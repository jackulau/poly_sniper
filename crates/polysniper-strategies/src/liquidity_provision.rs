//! Liquidity Provision Strategy
//!
//! Places orders on both sides of the book to earn the bid-ask spread
//! while managing inventory risk through position skewing.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use polysniper_core::{
    MarketId, OrderType, Outcome, Priority, Side, StateProvider, Strategy, StrategyError,
    SystemEvent, TokenId, TradeSignal,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Liquidity provision strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityProvisionConfig {
    pub enabled: bool,
    /// Minimum spread to quote (e.g., 0.02 = 2%)
    pub base_spread_pct: Decimal,
    /// Size per side in USD
    pub order_size_usd: Decimal,
    /// Maximum inventory per market in USD
    pub max_position_usd: Decimal,
    /// How much to skew prices with inventory (higher = more aggressive skew)
    #[serde(default = "default_inventory_skew_factor")]
    pub inventory_skew_factor: Decimal,
    /// How often to refresh quotes in seconds
    #[serde(default = "default_refresh_interval")]
    pub refresh_interval_secs: u64,
    /// Minimum market liquidity to provide quotes
    #[serde(default = "default_min_liquidity")]
    pub min_liquidity_usd: Decimal,
    /// Markets to provide liquidity for (empty = all)
    #[serde(default)]
    pub markets: Vec<String>,
    /// Cancel opposite side order when one side fills
    #[serde(default)]
    pub cancel_on_fill: bool,
    /// Price change threshold to trigger quote refresh (percentage)
    #[serde(default = "default_price_change_threshold")]
    pub price_change_threshold_pct: Decimal,
    /// Maximum total inventory across all markets
    #[serde(default = "default_max_total_position")]
    pub max_total_position_usd: Decimal,
}

fn default_inventory_skew_factor() -> Decimal {
    dec!(0.5)
}

fn default_refresh_interval() -> u64 {
    30
}

fn default_min_liquidity() -> Decimal {
    dec!(1000)
}

fn default_price_change_threshold() -> Decimal {
    dec!(1.0)
}

fn default_max_total_position() -> Decimal {
    dec!(5000)
}

impl Default for LiquidityProvisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_spread_pct: dec!(0.02),
            order_size_usd: dec!(50),
            max_position_usd: dec!(500),
            inventory_skew_factor: default_inventory_skew_factor(),
            refresh_interval_secs: default_refresh_interval(),
            min_liquidity_usd: default_min_liquidity(),
            markets: Vec::new(),
            cancel_on_fill: true,
            price_change_threshold_pct: default_price_change_threshold(),
            max_total_position_usd: default_max_total_position(),
        }
    }
}

/// Active quote state for a market
#[derive(Debug, Clone)]
struct QuoteState {
    /// Last mid price we quoted around
    last_mid_price: Decimal,
    /// Last time we refreshed quotes
    last_refresh: DateTime<Utc>,
    /// Current inventory in contracts (positive = long, negative = short)
    inventory: Decimal,
}

/// Calculated quote prices
#[derive(Debug, Clone)]
pub struct Quote {
    pub bid_price: Decimal,
    pub ask_price: Decimal,
    pub bid_size: Decimal,
    pub ask_size: Decimal,
}

/// Liquidity Provision Strategy
pub struct LiquidityProvisionStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: LiquidityProvisionConfig,
    /// Quote state per market (token_id -> state)
    quote_states: Arc<RwLock<HashMap<TokenId, QuoteState>>>,
    /// Token to market mapping
    token_market_map: Arc<RwLock<HashMap<TokenId, (MarketId, Outcome)>>>,
}

impl LiquidityProvisionStrategy {
    /// Create a new liquidity provision strategy
    pub fn new(config: LiquidityProvisionConfig) -> Self {
        let enabled = config.enabled;
        Self {
            id: "liquidity_provision".to_string(),
            name: "Liquidity Provision Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config,
            quote_states: Arc::new(RwLock::new(HashMap::new())),
            token_market_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a token for liquidity provision
    pub async fn register_token(&self, token_id: TokenId, market_id: MarketId, outcome: Outcome) {
        self.token_market_map
            .write()
            .await
            .insert(token_id, (market_id, outcome));
    }

    /// Check if we should provide liquidity for this market
    fn should_quote_market(&self, market_id: &str) -> bool {
        if self.config.markets.is_empty() {
            return true;
        }
        self.config.markets.contains(&market_id.to_string())
    }

    /// Calculate quote prices with inventory skew
    ///
    /// When we have positive inventory (long), we want to sell more than buy,
    /// so we lower both bid and ask prices to encourage selling.
    /// When we have negative inventory (short), we raise prices to encourage buying.
    pub fn calculate_quote(&self, mid_price: Decimal, inventory: Decimal) -> Option<Quote> {
        if mid_price.is_zero() {
            return None;
        }

        let half_spread = mid_price * self.config.base_spread_pct / dec!(2);

        // Calculate inventory skew
        // Skew is proportional to inventory relative to max position
        let inventory_ratio = if self.config.max_position_usd.is_zero() {
            Decimal::ZERO
        } else {
            inventory / self.config.max_position_usd
        };

        // Skew shifts both prices down when long (positive inventory)
        // and up when short (negative inventory)
        let skew_amount = mid_price * self.config.inventory_skew_factor * inventory_ratio;

        let bid_price = mid_price - half_spread - skew_amount;
        let ask_price = mid_price + half_spread - skew_amount;

        // Ensure prices are valid (between 0 and 1 for prediction markets)
        let bid_price = bid_price.max(dec!(0.01)).min(dec!(0.99));
        let ask_price = ask_price.max(dec!(0.01)).min(dec!(0.99));

        // Ensure ask > bid
        if ask_price <= bid_price {
            return None;
        }

        // Calculate sizes based on USD amount
        let bid_size = if bid_price.is_zero() {
            Decimal::ZERO
        } else {
            self.config.order_size_usd / bid_price
        };

        let ask_size = if ask_price.is_zero() {
            Decimal::ZERO
        } else {
            self.config.order_size_usd / ask_price
        };

        Some(Quote {
            bid_price,
            ask_price,
            bid_size,
            ask_size,
        })
    }

    /// Check if quotes need to be refreshed
    fn should_refresh_quotes(&self, state: &QuoteState, current_mid: Decimal) -> bool {
        let now = Utc::now();

        // Check time-based refresh
        let elapsed =
            (now - state.last_refresh).num_seconds() as u64 >= self.config.refresh_interval_secs;

        if elapsed {
            return true;
        }

        // Check price-based refresh
        if state.last_mid_price.is_zero() {
            return true;
        }

        let price_change_pct =
            ((current_mid - state.last_mid_price) / state.last_mid_price).abs() * dec!(100);
        price_change_pct >= self.config.price_change_threshold_pct
    }

    /// Check if inventory is within limits
    fn is_within_position_limits(&self, inventory_usd: Decimal) -> bool {
        inventory_usd.abs() <= self.config.max_position_usd
    }

    /// Update inventory from a fill
    pub async fn update_inventory(
        &self,
        token_id: &TokenId,
        side: Side,
        size: Decimal,
        price: Decimal,
    ) {
        let mut states = self.quote_states.write().await;
        if let Some(state) = states.get_mut(token_id) {
            let position_change = match side {
                Side::Buy => size * price,
                Side::Sell => -size * price,
            };
            state.inventory += position_change;
            debug!(
                token_id = %token_id,
                side = ?side,
                size = %size,
                new_inventory = %state.inventory,
                "Updated inventory"
            );
        }
    }

    /// Get current inventory for a token
    pub async fn get_inventory(&self, token_id: &TokenId) -> Decimal {
        self.quote_states
            .read()
            .await
            .get(token_id)
            .map(|s| s.inventory)
            .unwrap_or(Decimal::ZERO)
    }

    /// Get total inventory across all markets
    pub async fn get_total_inventory(&self) -> Decimal {
        self.quote_states
            .read()
            .await
            .values()
            .map(|s| s.inventory.abs())
            .sum()
    }

    /// Generate signals for a market
    async fn generate_quote_signals(
        &self,
        market_id: &MarketId,
        token_id: &TokenId,
        mid_price: Decimal,
        outcome: Outcome,
    ) -> Vec<TradeSignal> {
        let mut signals = Vec::new();
        let now = Utc::now();

        // Get or create quote state
        let mut states = self.quote_states.write().await;
        let state = states.entry(token_id.clone()).or_insert(QuoteState {
            last_mid_price: mid_price,
            last_refresh: now,
            inventory: Decimal::ZERO,
        });

        // Check if we should refresh
        if !self.should_refresh_quotes(state, mid_price) {
            return signals;
        }

        // Check position limits
        let inventory_usd = state.inventory;
        if !self.is_within_position_limits(inventory_usd) {
            warn!(
                token_id = %token_id,
                inventory = %inventory_usd,
                max = %self.config.max_position_usd,
                "Position limit reached, not quoting"
            );
            return signals;
        }

        // Calculate quotes
        let quote = match self.calculate_quote(mid_price, state.inventory) {
            Some(q) => q,
            None => {
                debug!(token_id = %token_id, "Could not calculate valid quote");
                return signals;
            }
        };

        info!(
            token_id = %token_id,
            mid = %mid_price,
            bid = %quote.bid_price,
            ask = %quote.ask_price,
            inventory = %state.inventory,
            "Generating liquidity provision quotes"
        );

        // Generate bid signal
        let bid_signal = TradeSignal {
            id: format!(
                "sig_lp_bid_{}_{}_{}",
                token_id,
                now.timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: market_id.clone(),
            token_id: token_id.clone(),
            outcome,
            side: Side::Buy,
            price: Some(quote.bid_price),
            size: quote.bid_size,
            size_usd: self.config.order_size_usd,
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: now,
            reason: format!(
                "LP bid at {:.4} (mid: {:.4}, inventory: {:.2})",
                quote.bid_price, mid_price, state.inventory
            ),
            metadata: serde_json::json!({
                "quote_type": "bid",
                "mid_price": mid_price.to_string(),
                "spread_pct": self.config.base_spread_pct.to_string(),
                "inventory": state.inventory.to_string(),
                "skew_factor": self.config.inventory_skew_factor.to_string(),
            }),
        };
        signals.push(bid_signal);

        // Generate ask signal
        let ask_signal = TradeSignal {
            id: format!(
                "sig_lp_ask_{}_{}_{}",
                token_id,
                now.timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: market_id.clone(),
            token_id: token_id.clone(),
            outcome,
            side: Side::Sell,
            price: Some(quote.ask_price),
            size: quote.ask_size,
            size_usd: self.config.order_size_usd,
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: now,
            reason: format!(
                "LP ask at {:.4} (mid: {:.4}, inventory: {:.2})",
                quote.ask_price, mid_price, state.inventory
            ),
            metadata: serde_json::json!({
                "quote_type": "ask",
                "mid_price": mid_price.to_string(),
                "spread_pct": self.config.base_spread_pct.to_string(),
                "inventory": state.inventory.to_string(),
                "skew_factor": self.config.inventory_skew_factor.to_string(),
            }),
        };
        signals.push(ask_signal);

        // Update state
        state.last_mid_price = mid_price;
        state.last_refresh = now;

        signals
    }
}

#[async_trait]
impl Strategy for LiquidityProvisionStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn process_event(
        &self,
        event: &SystemEvent,
        state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        let mut signals = Vec::new();

        // Extract price and token info from orderbook updates
        let (token_id, mid_price, market_id) = match event {
            SystemEvent::OrderbookUpdate(e) => {
                if let Some(mid) = e.orderbook.mid_price() {
                    (e.token_id.clone(), mid, e.market_id.clone())
                } else {
                    return Ok(signals);
                }
            }
            SystemEvent::PriceChange(e) => (e.token_id.clone(), e.new_price, e.market_id.clone()),
            _ => return Ok(signals),
        };

        // Check if we should quote this market
        if !self.should_quote_market(&market_id) {
            return Ok(signals);
        }

        // Check market liquidity
        if let Some(market) = state.get_market(&market_id).await {
            if market.liquidity < self.config.min_liquidity_usd {
                debug!(
                    market_id = %market_id,
                    liquidity = %market.liquidity,
                    min_required = %self.config.min_liquidity_usd,
                    "Insufficient liquidity, skipping"
                );
                return Ok(signals);
            }

            if !market.active || market.closed {
                debug!(market_id = %market_id, "Market not active or closed");
                return Ok(signals);
            }
        }

        // Check total position across all markets
        let total_inventory = self.get_total_inventory().await;
        if total_inventory >= self.config.max_total_position_usd {
            warn!(
                total_inventory = %total_inventory,
                max = %self.config.max_total_position_usd,
                "Total position limit reached, not quoting"
            );
            return Ok(signals);
        }

        // Get outcome from token map
        let token_map = self.token_market_map.read().await;
        let outcome = token_map
            .get(&token_id)
            .map(|(_, o)| *o)
            .unwrap_or(Outcome::Yes);
        drop(token_map);

        // Generate quote signals
        let quote_signals = self
            .generate_quote_signals(&market_id, &token_id, mid_price, outcome)
            .await;
        signals.extend(quote_signals);

        Ok(signals)
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(
            event,
            SystemEvent::OrderbookUpdate(_) | SystemEvent::PriceChange(_)
        )
    }

    async fn initialize(&mut self, state: &dyn StateProvider) -> Result<(), StrategyError> {
        info!(
            strategy_id = %self.id,
            spread_pct = %self.config.base_spread_pct,
            order_size = %self.config.order_size_usd,
            max_position = %self.config.max_position_usd,
            "Initializing liquidity provision strategy"
        );

        // Register all known markets
        for market in state.get_all_markets().await {
            self.register_token(
                market.yes_token_id.clone(),
                market.condition_id.clone(),
                Outcome::Yes,
            )
            .await;
            self.register_token(market.no_token_id.clone(), market.condition_id, Outcome::No)
                .await;
        }

        // Initialize inventory from current positions
        for position in state.get_all_positions().await {
            let inventory_usd = position.size * position.avg_price;
            let mut states = self.quote_states.write().await;
            states.insert(
                position.token_id.clone(),
                QuoteState {
                    last_mid_price: position.avg_price,
                    last_refresh: Utc::now(),
                    inventory: inventory_usd,
                },
            );
        }

        info!(
            strategy_id = %self.id,
            "Liquidity provision strategy initialized"
        );

        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }
}

fn rand_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{:08x}", nanos)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> LiquidityProvisionConfig {
        LiquidityProvisionConfig {
            enabled: true,
            base_spread_pct: dec!(0.02), // 2%
            order_size_usd: dec!(100),
            max_position_usd: dec!(500),
            inventory_skew_factor: dec!(0.5),
            refresh_interval_secs: 30,
            min_liquidity_usd: dec!(1000),
            markets: vec![],
            cancel_on_fill: true,
            price_change_threshold_pct: dec!(1.0),
            max_total_position_usd: dec!(5000),
        }
    }

    #[test]
    fn test_calculate_quote_no_inventory() {
        let strategy = LiquidityProvisionStrategy::new(default_config());
        let mid_price = dec!(0.50);
        let inventory = dec!(0);

        let quote = strategy.calculate_quote(mid_price, inventory).unwrap();

        // With 2% spread around 0.50:
        // half_spread = 0.50 * 0.02 / 2 = 0.005
        // bid = 0.50 - 0.005 = 0.495
        // ask = 0.50 + 0.005 = 0.505
        assert_eq!(quote.bid_price, dec!(0.495));
        assert_eq!(quote.ask_price, dec!(0.505));
    }

    #[test]
    fn test_calculate_quote_with_long_inventory() {
        let strategy = LiquidityProvisionStrategy::new(default_config());
        let mid_price = dec!(0.50);
        let inventory = dec!(250); // 50% of max position

        let quote = strategy.calculate_quote(mid_price, inventory).unwrap();

        // With long inventory, prices should be skewed down
        // Skew = 0.50 * 0.5 * (250/500) = 0.50 * 0.5 * 0.5 = 0.125
        // Bid = 0.50 - 0.005 - 0.125 = 0.37
        // Ask = 0.50 + 0.005 - 0.125 = 0.38
        assert!(quote.bid_price < dec!(0.49));
        assert!(quote.ask_price < dec!(0.51));
        assert!(quote.ask_price > quote.bid_price);
    }

    #[test]
    fn test_calculate_quote_with_short_inventory() {
        let strategy = LiquidityProvisionStrategy::new(default_config());
        let mid_price = dec!(0.50);
        let inventory = dec!(-250); // -50% of max position

        let quote = strategy.calculate_quote(mid_price, inventory).unwrap();

        // With short inventory, prices should be skewed up
        assert!(quote.bid_price > dec!(0.49));
        assert!(quote.ask_price > dec!(0.51));
        assert!(quote.ask_price > quote.bid_price);
    }

    #[test]
    fn test_calculate_quote_clamps_prices() {
        let strategy = LiquidityProvisionStrategy::new(default_config());

        // Very low mid price
        let quote = strategy.calculate_quote(dec!(0.02), dec!(0)).unwrap();
        assert!(quote.bid_price >= dec!(0.01));

        // Very high mid price
        let quote = strategy.calculate_quote(dec!(0.98), dec!(0)).unwrap();
        assert!(quote.ask_price <= dec!(0.99));
    }

    #[test]
    fn test_calculate_quote_zero_mid_price() {
        let strategy = LiquidityProvisionStrategy::new(default_config());
        let quote = strategy.calculate_quote(dec!(0), dec!(0));
        assert!(quote.is_none());
    }

    #[test]
    fn test_is_within_position_limits() {
        let strategy = LiquidityProvisionStrategy::new(default_config());

        assert!(strategy.is_within_position_limits(dec!(400)));
        assert!(strategy.is_within_position_limits(dec!(500)));
        assert!(!strategy.is_within_position_limits(dec!(501)));
        assert!(strategy.is_within_position_limits(dec!(-400)));
        assert!(!strategy.is_within_position_limits(dec!(-501)));
    }

    #[test]
    fn test_should_quote_market_all_markets() {
        let strategy = LiquidityProvisionStrategy::new(default_config());
        assert!(strategy.should_quote_market("any-market"));
    }

    #[test]
    fn test_should_quote_market_specific_markets() {
        let mut config = default_config();
        config.markets = vec!["market-1".to_string(), "market-2".to_string()];
        let strategy = LiquidityProvisionStrategy::new(config);

        assert!(strategy.should_quote_market("market-1"));
        assert!(strategy.should_quote_market("market-2"));
        assert!(!strategy.should_quote_market("market-3"));
    }

    #[test]
    fn test_should_refresh_quotes_time_based() {
        let strategy = LiquidityProvisionStrategy::new(default_config());
        let now = Utc::now();

        // Fresh state - should not refresh
        let fresh_state = QuoteState {
            last_mid_price: dec!(0.50),
            last_refresh: now,
            inventory: dec!(0),
        };
        assert!(!strategy.should_refresh_quotes(&fresh_state, dec!(0.50)));

        // Stale state - should refresh
        let stale_state = QuoteState {
            last_mid_price: dec!(0.50),
            last_refresh: now - chrono::Duration::seconds(60),
            inventory: dec!(0),
        };
        assert!(strategy.should_refresh_quotes(&stale_state, dec!(0.50)));
    }

    #[test]
    fn test_should_refresh_quotes_price_based() {
        let strategy = LiquidityProvisionStrategy::new(default_config());
        let now = Utc::now();

        let state = QuoteState {
            last_mid_price: dec!(0.50),
            last_refresh: now,
            inventory: dec!(0),
        };

        // Small price change - should not refresh
        assert!(!strategy.should_refresh_quotes(&state, dec!(0.504)));

        // Large price change - should refresh (> 1%)
        assert!(strategy.should_refresh_quotes(&state, dec!(0.51)));
    }

    #[test]
    fn test_quote_sizes_calculated_correctly() {
        let strategy = LiquidityProvisionStrategy::new(default_config());
        let quote = strategy.calculate_quote(dec!(0.50), dec!(0)).unwrap();

        // Size = order_size_usd / price
        // Bid size = 100 / 0.49 ≈ 204.08
        // Ask size = 100 / 0.51 ≈ 196.08
        assert!(quote.bid_size > dec!(200));
        assert!(quote.ask_size > dec!(195));
    }

    #[tokio::test]
    async fn test_inventory_tracking() {
        let strategy = LiquidityProvisionStrategy::new(default_config());
        let token_id = "test-token".to_string();

        // Initialize state
        {
            let mut states = strategy.quote_states.write().await;
            states.insert(
                token_id.clone(),
                QuoteState {
                    last_mid_price: dec!(0.50),
                    last_refresh: Utc::now(),
                    inventory: dec!(0),
                },
            );
        }

        // Buy adds to inventory
        strategy
            .update_inventory(&token_id, Side::Buy, dec!(100), dec!(0.50))
            .await;
        assert_eq!(strategy.get_inventory(&token_id).await, dec!(50));

        // Sell removes from inventory
        strategy
            .update_inventory(&token_id, Side::Sell, dec!(100), dec!(0.50))
            .await;
        assert_eq!(strategy.get_inventory(&token_id).await, dec!(0));
    }
}
