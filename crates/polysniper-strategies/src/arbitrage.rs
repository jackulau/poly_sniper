//! YES/NO Token Arbitrage Strategy
//!
//! Detects and exploits price discrepancies between YES/NO tokens when their
//! combined prices deviate from $1.00.

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

/// Arbitrage strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageConfig {
    pub enabled: bool,
    /// Minimum edge percentage deviation from $1.00 to trigger (e.g., 1.0 = 1%)
    pub min_edge_pct: Decimal,
    /// Order size in USD per leg
    pub order_size_usd: Decimal,
    /// Minimum liquidity in USD to trade
    #[serde(default = "default_min_liquidity")]
    pub min_liquidity_usd: Decimal,
    /// Specific markets to monitor (empty = all markets)
    #[serde(default)]
    pub markets: Vec<String>,
    /// Cooldown period in seconds after detecting an opportunity
    #[serde(default = "default_cooldown")]
    pub cooldown_secs: u64,
    /// Maximum acceptable slippage percentage
    #[serde(default = "default_max_slippage")]
    pub max_slippage_pct: Decimal,
}

fn default_min_liquidity() -> Decimal {
    dec!(1000)
}

fn default_cooldown() -> u64 {
    30
}

fn default_max_slippage() -> Decimal {
    dec!(0.5)
}

impl Default for ArbitrageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_edge_pct: dec!(1.0),
            order_size_usd: dec!(100),
            min_liquidity_usd: default_min_liquidity(),
            markets: Vec::new(),
            cooldown_secs: default_cooldown(),
            max_slippage_pct: default_max_slippage(),
        }
    }
}

/// Market price pair tracking
#[derive(Debug, Clone)]
struct MarketPrices {
    yes_token_id: TokenId,
    no_token_id: TokenId,
    yes_price: Option<Decimal>,
    no_price: Option<Decimal>,
    yes_spread: Option<Decimal>,
    no_spread: Option<Decimal>,
    last_updated: DateTime<Utc>,
}

impl MarketPrices {
    fn new(yes_token_id: TokenId, no_token_id: TokenId) -> Self {
        Self {
            yes_token_id,
            no_token_id,
            yes_price: None,
            no_price: None,
            yes_spread: None,
            no_spread: None,
            last_updated: Utc::now(),
        }
    }

    fn combined_price(&self) -> Option<Decimal> {
        match (self.yes_price, self.no_price) {
            (Some(yes), Some(no)) => Some(yes + no),
            _ => None,
        }
    }

    fn total_spread_cost(&self) -> Decimal {
        let yes_spread = self.yes_spread.unwrap_or(Decimal::ZERO);
        let no_spread = self.no_spread.unwrap_or(Decimal::ZERO);
        yes_spread + no_spread
    }
}

/// Cooldown tracking
#[derive(Debug, Clone)]
struct CooldownEntry {
    until: DateTime<Utc>,
}

/// YES/NO Token Arbitrage Strategy
pub struct ArbitrageStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: ArbitrageConfig,
    /// Price pairs per market
    market_prices: Arc<RwLock<HashMap<MarketId, MarketPrices>>>,
    /// Token to market mapping
    token_to_market: Arc<RwLock<HashMap<TokenId, (MarketId, Outcome)>>>,
    /// Cooldowns per market
    cooldowns: Arc<RwLock<HashMap<MarketId, CooldownEntry>>>,
}

impl ArbitrageStrategy {
    /// Create a new arbitrage strategy
    pub fn new(config: ArbitrageConfig) -> Self {
        let enabled = config.enabled;
        Self {
            id: "arbitrage".to_string(),
            name: "YES/NO Token Arbitrage Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config,
            market_prices: Arc::new(RwLock::new(HashMap::new())),
            token_to_market: Arc::new(RwLock::new(HashMap::new())),
            cooldowns: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a market for monitoring
    pub async fn register_market(
        &self,
        market_id: MarketId,
        yes_token_id: TokenId,
        no_token_id: TokenId,
    ) {
        let mut prices = self.market_prices.write().await;
        prices.insert(
            market_id.clone(),
            MarketPrices::new(yes_token_id.clone(), no_token_id.clone()),
        );

        let mut token_map = self.token_to_market.write().await;
        token_map.insert(yes_token_id, (market_id.clone(), Outcome::Yes));
        token_map.insert(no_token_id, (market_id, Outcome::No));
    }

    /// Check if market is in cooldown
    async fn is_in_cooldown(&self, market_id: &MarketId) -> bool {
        let cooldowns = self.cooldowns.read().await;
        if let Some(entry) = cooldowns.get(market_id) {
            return Utc::now() < entry.until;
        }
        false
    }

    /// Set cooldown for a market
    async fn set_cooldown(&self, market_id: &MarketId) {
        let until = Utc::now() + chrono::Duration::seconds(self.config.cooldown_secs as i64);
        self.cooldowns
            .write()
            .await
            .insert(market_id.clone(), CooldownEntry { until });
    }

    /// Update price for a token
    async fn update_price(
        &self,
        token_id: &TokenId,
        price: Decimal,
        spread: Option<Decimal>,
    ) -> Option<MarketId> {
        let token_map = self.token_to_market.read().await;
        let (market_id, outcome) = token_map.get(token_id)?;
        let market_id = market_id.clone();
        let outcome = *outcome;
        drop(token_map);

        let mut prices = self.market_prices.write().await;
        if let Some(market_prices) = prices.get_mut(&market_id) {
            match outcome {
                Outcome::Yes => {
                    market_prices.yes_price = Some(price);
                    market_prices.yes_spread = spread;
                }
                Outcome::No => {
                    market_prices.no_price = Some(price);
                    market_prices.no_spread = spread;
                }
            }
            market_prices.last_updated = Utc::now();
        }

        Some(market_id)
    }

    /// Check for arbitrage opportunity in a market
    async fn check_arbitrage_opportunity(&self, market_id: &MarketId) -> Option<ArbitrageOpportunity> {
        let prices = self.market_prices.read().await;
        let market_prices = prices.get(market_id)?;

        let combined = market_prices.combined_price()?;
        let spread_cost = market_prices.total_spread_cost();

        // Calculate edge: how much below $1.00 the combined price is
        // We need combined + spread_cost < 1.0 - min_edge_pct to have a profitable arb
        let effective_cost = combined + spread_cost;
        let edge = dec!(1.0) - effective_cost;
        let edge_pct = edge * dec!(100);

        debug!(
            market_id = %market_id,
            yes_price = %market_prices.yes_price.unwrap_or(Decimal::ZERO),
            no_price = %market_prices.no_price.unwrap_or(Decimal::ZERO),
            combined = %combined,
            spread_cost = %spread_cost,
            effective_cost = %effective_cost,
            edge_pct = %edge_pct,
            "Checking arbitrage opportunity"
        );

        // Only profitable if combined price + spreads < 1.0
        if edge_pct < self.config.min_edge_pct {
            return None;
        }

        Some(ArbitrageOpportunity {
            market_id: market_id.clone(),
            yes_token_id: market_prices.yes_token_id.clone(),
            no_token_id: market_prices.no_token_id.clone(),
            yes_price: market_prices.yes_price?,
            no_price: market_prices.no_price?,
            combined_price: combined,
            spread_cost,
            edge_pct,
        })
    }

    /// Generate trade signals for an arbitrage opportunity
    fn generate_signals(&self, opportunity: ArbitrageOpportunity) -> Vec<TradeSignal> {
        let now = Utc::now();
        let timestamp_ms = now.timestamp_millis();

        // Calculate size per leg
        let yes_size = if opportunity.yes_price.is_zero() {
            Decimal::ZERO
        } else {
            self.config.order_size_usd / opportunity.yes_price
        };

        let no_size = if opportunity.no_price.is_zero() {
            Decimal::ZERO
        } else {
            self.config.order_size_usd / opportunity.no_price
        };

        let reason = format!(
            "Arbitrage opportunity: YES({:.4}) + NO({:.4}) = {:.4}, edge {:.2}%",
            opportunity.yes_price,
            opportunity.no_price,
            opportunity.combined_price,
            opportunity.edge_pct
        );

        let metadata = serde_json::json!({
            "combined_price": opportunity.combined_price.to_string(),
            "edge_pct": opportunity.edge_pct.to_string(),
            "spread_cost": opportunity.spread_cost.to_string(),
            "yes_price": opportunity.yes_price.to_string(),
            "no_price": opportunity.no_price.to_string(),
        });

        vec![
            TradeSignal {
                id: format!(
                    "sig_arb_yes_{}_{}_{}",
                    opportunity.market_id,
                    timestamp_ms,
                    rand_suffix()
                ),
                strategy_id: self.id.clone(),
                market_id: opportunity.market_id.clone(),
                token_id: opportunity.yes_token_id,
                outcome: Outcome::Yes,
                side: Side::Buy,
                price: Some(opportunity.yes_price),
                size: yes_size,
                size_usd: self.config.order_size_usd,
                order_type: OrderType::Fok,
                priority: Priority::High,
                timestamp: now,
                reason: reason.clone(),
                metadata: metadata.clone(),
            },
            TradeSignal {
                id: format!(
                    "sig_arb_no_{}_{}_{}",
                    opportunity.market_id,
                    timestamp_ms,
                    rand_suffix()
                ),
                strategy_id: self.id.clone(),
                market_id: opportunity.market_id,
                token_id: opportunity.no_token_id,
                outcome: Outcome::No,
                side: Side::Buy,
                price: Some(opportunity.no_price),
                size: no_size,
                size_usd: self.config.order_size_usd,
                order_type: OrderType::Fok,
                priority: Priority::High,
                timestamp: now,
                reason,
                metadata,
            },
        ]
    }

    /// Determine if we should monitor this market
    fn should_monitor_market(&self, market_id: &str) -> bool {
        if self.config.markets.is_empty() {
            return true;
        }
        self.config.markets.contains(&market_id.to_string())
    }
}

/// Arbitrage opportunity details
#[derive(Debug, Clone)]
struct ArbitrageOpportunity {
    market_id: MarketId,
    yes_token_id: TokenId,
    no_token_id: TokenId,
    yes_price: Decimal,
    no_price: Decimal,
    combined_price: Decimal,
    spread_cost: Decimal,
    edge_pct: Decimal,
}

#[async_trait]
impl Strategy for ArbitrageStrategy {
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

        // Extract price and token info
        let (token_id, price, spread, market_id) = match event {
            SystemEvent::PriceChange(e) => {
                (e.token_id.clone(), e.new_price, None, e.market_id.clone())
            }
            SystemEvent::OrderbookUpdate(e) => {
                if let Some(mid) = e.orderbook.mid_price() {
                    let spread = e.orderbook.spread();
                    (e.token_id.clone(), mid, spread, e.market_id.clone())
                } else {
                    return Ok(signals);
                }
            }
            _ => return Ok(signals),
        };

        // Check if we should monitor this market
        if !self.should_monitor_market(&market_id) {
            return Ok(signals);
        }

        // Update price
        let Some(updated_market_id) = self.update_price(&token_id, price, spread).await else {
            return Ok(signals);
        };

        // Check cooldown
        if self.is_in_cooldown(&updated_market_id).await {
            debug!(market_id = %updated_market_id, "Market in cooldown, skipping");
            return Ok(signals);
        }

        // Check for arbitrage opportunity
        if let Some(opportunity) = self.check_arbitrage_opportunity(&updated_market_id).await {
            info!(
                market_id = %opportunity.market_id,
                combined_price = %opportunity.combined_price,
                edge_pct = %opportunity.edge_pct,
                "Arbitrage opportunity detected!"
            );

            // Check liquidity
            if let Some(market) = state.get_market(&updated_market_id).await {
                if market.liquidity < self.config.min_liquidity_usd {
                    warn!(
                        market_id = %updated_market_id,
                        liquidity = %market.liquidity,
                        min_required = %self.config.min_liquidity_usd,
                        "Insufficient liquidity, skipping opportunity"
                    );
                    return Ok(signals);
                }
            }

            signals = self.generate_signals(opportunity);
            self.set_cooldown(&updated_market_id).await;
        }

        Ok(signals)
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(
            event,
            SystemEvent::PriceChange(_) | SystemEvent::OrderbookUpdate(_)
        )
    }

    async fn initialize(&mut self, state: &dyn StateProvider) -> Result<(), StrategyError> {
        info!(
            strategy_id = %self.id,
            min_edge_pct = %self.config.min_edge_pct,
            order_size_usd = %self.config.order_size_usd,
            "Initializing arbitrage strategy"
        );

        // Register all known markets
        for market in state.get_all_markets().await {
            self.register_market(
                market.condition_id,
                market.yes_token_id,
                market.no_token_id,
            )
            .await;
        }

        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }

    async fn reload_config(&mut self, config_content: &str) -> Result<(), StrategyError> {
        let new_config: ArbitrageConfig = toml::from_str(config_content)
            .map_err(|e| StrategyError::ConfigError(format!("Failed to parse config: {}", e)))?;

        self.config = new_config;
        tracing::info!(strategy_id = %self.id, "Reloaded arbitrage strategy config");
        Ok(())
    }

    fn config_name(&self) -> &str {
        "arbitrage"
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
    use polysniper_core::{OrderbookUpdateEvent, Orderbook, PriceLevel, PriceChangeEvent};
    use std::collections::HashMap;

    /// Mock state provider for testing
    struct MockStateProvider {
        markets: HashMap<MarketId, polysniper_core::Market>,
    }

    impl MockStateProvider {
        fn new() -> Self {
            Self {
                markets: HashMap::new(),
            }
        }

        fn add_market(&mut self, market: polysniper_core::Market) {
            self.markets.insert(market.condition_id.clone(), market);
        }
    }

    #[async_trait]
    impl StateProvider for MockStateProvider {
        async fn get_market(&self, market_id: &MarketId) -> Option<polysniper_core::Market> {
            self.markets.get(market_id).cloned()
        }

        async fn get_all_markets(&self) -> Vec<polysniper_core::Market> {
            self.markets.values().cloned().collect()
        }

        async fn get_orderbook(&self, _token_id: &TokenId) -> Option<Orderbook> {
            None
        }

        async fn get_price(&self, _token_id: &TokenId) -> Option<Decimal> {
            None
        }

        async fn get_position(&self, _market_id: &MarketId) -> Option<polysniper_core::Position> {
            None
        }

        async fn get_all_positions(&self) -> Vec<polysniper_core::Position> {
            Vec::new()
        }

        async fn get_price_history(
            &self,
            _token_id: &TokenId,
            _limit: usize,
        ) -> Vec<(DateTime<Utc>, Decimal)> {
            Vec::new()
        }

        async fn get_portfolio_value(&self) -> Decimal {
            Decimal::ZERO
        }

        async fn get_daily_pnl(&self) -> Decimal {
            Decimal::ZERO
        }
    }

    fn create_test_market() -> polysniper_core::Market {
        polysniper_core::Market {
            condition_id: "test_market_1".to_string(),
            question: "Test Market?".to_string(),
            description: Some("A test market".to_string()),
            tags: vec!["test".to_string()],
            yes_token_id: "yes_token_1".to_string(),
            no_token_id: "no_token_1".to_string(),
            created_at: Utc::now(),
            end_date: None,
            active: true,
            closed: false,
            volume: dec!(10000),
            liquidity: dec!(5000),
        }
    }

    fn create_orderbook(token_id: &str, market_id: &str, bid: Decimal, ask: Decimal) -> Orderbook {
        Orderbook {
            token_id: token_id.to_string(),
            market_id: market_id.to_string(),
            bids: vec![PriceLevel { price: bid, size: dec!(100) }],
            asks: vec![PriceLevel { price: ask, size: dec!(100) }],
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_config_default() {
        let config = ArbitrageConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_edge_pct, dec!(1.0));
        assert_eq!(config.order_size_usd, dec!(100));
        assert_eq!(config.cooldown_secs, 30);
    }

    #[tokio::test]
    async fn test_strategy_initialization() {
        let config = ArbitrageConfig::default();
        let mut strategy = ArbitrageStrategy::new(config);

        let mut state = MockStateProvider::new();
        state.add_market(create_test_market());

        strategy.initialize(&state).await.unwrap();

        assert_eq!(strategy.id(), "arbitrage");
        assert!(strategy.is_enabled());
    }

    #[tokio::test]
    async fn test_edge_detection_no_opportunity() {
        let config = ArbitrageConfig {
            min_edge_pct: dec!(1.0),
            ..Default::default()
        };
        let mut strategy = ArbitrageStrategy::new(config);

        let mut state = MockStateProvider::new();
        state.add_market(create_test_market());
        strategy.initialize(&state).await.unwrap();

        // YES = 0.50, NO = 0.50, combined = 1.00 (no edge)
        let yes_orderbook = create_orderbook("yes_token_1", "test_market_1", dec!(0.49), dec!(0.51));
        let no_orderbook = create_orderbook("no_token_1", "test_market_1", dec!(0.49), dec!(0.51));

        let yes_event = SystemEvent::OrderbookUpdate(OrderbookUpdateEvent {
            market_id: "test_market_1".to_string(),
            token_id: "yes_token_1".to_string(),
            orderbook: yes_orderbook,
            timestamp: Utc::now(),
        });

        let signals = strategy.process_event(&yes_event, &state).await.unwrap();
        assert!(signals.is_empty());

        let no_event = SystemEvent::OrderbookUpdate(OrderbookUpdateEvent {
            market_id: "test_market_1".to_string(),
            token_id: "no_token_1".to_string(),
            orderbook: no_orderbook,
            timestamp: Utc::now(),
        });

        let signals = strategy.process_event(&no_event, &state).await.unwrap();
        assert!(signals.is_empty());
    }

    #[tokio::test]
    async fn test_edge_detection_with_opportunity() {
        let config = ArbitrageConfig {
            min_edge_pct: dec!(1.0),
            cooldown_secs: 0,
            ..Default::default()
        };
        let mut strategy = ArbitrageStrategy::new(config);

        let mut state = MockStateProvider::new();
        state.add_market(create_test_market());
        strategy.initialize(&state).await.unwrap();

        // YES = 0.45, NO = 0.45, combined = 0.90 (10% edge before spreads)
        let yes_orderbook = create_orderbook("yes_token_1", "test_market_1", dec!(0.44), dec!(0.46));
        let no_orderbook = create_orderbook("no_token_1", "test_market_1", dec!(0.44), dec!(0.46));

        // Update YES price
        let yes_event = SystemEvent::OrderbookUpdate(OrderbookUpdateEvent {
            market_id: "test_market_1".to_string(),
            token_id: "yes_token_1".to_string(),
            orderbook: yes_orderbook,
            timestamp: Utc::now(),
        });
        strategy.process_event(&yes_event, &state).await.unwrap();

        // Update NO price - should trigger opportunity
        let no_event = SystemEvent::OrderbookUpdate(OrderbookUpdateEvent {
            market_id: "test_market_1".to_string(),
            token_id: "no_token_1".to_string(),
            orderbook: no_orderbook,
            timestamp: Utc::now(),
        });

        let signals = strategy.process_event(&no_event, &state).await.unwrap();

        // Should get 2 signals (one for YES, one for NO)
        assert_eq!(signals.len(), 2);

        let yes_signal = signals.iter().find(|s| s.outcome == Outcome::Yes).unwrap();
        let no_signal = signals.iter().find(|s| s.outcome == Outcome::No).unwrap();

        assert_eq!(yes_signal.side, Side::Buy);
        assert_eq!(no_signal.side, Side::Buy);
        assert_eq!(yes_signal.strategy_id, "arbitrage");
        assert_eq!(no_signal.strategy_id, "arbitrage");
    }

    #[tokio::test]
    async fn test_cooldown_behavior() {
        let config = ArbitrageConfig {
            min_edge_pct: dec!(1.0),
            cooldown_secs: 60,
            ..Default::default()
        };
        let mut strategy = ArbitrageStrategy::new(config);

        let mut state = MockStateProvider::new();
        state.add_market(create_test_market());
        strategy.initialize(&state).await.unwrap();

        // Create opportunity
        let yes_orderbook = create_orderbook("yes_token_1", "test_market_1", dec!(0.44), dec!(0.46));
        let no_orderbook = create_orderbook("no_token_1", "test_market_1", dec!(0.44), dec!(0.46));

        let yes_event = SystemEvent::OrderbookUpdate(OrderbookUpdateEvent {
            market_id: "test_market_1".to_string(),
            token_id: "yes_token_1".to_string(),
            orderbook: yes_orderbook.clone(),
            timestamp: Utc::now(),
        });
        strategy.process_event(&yes_event, &state).await.unwrap();

        let no_event = SystemEvent::OrderbookUpdate(OrderbookUpdateEvent {
            market_id: "test_market_1".to_string(),
            token_id: "no_token_1".to_string(),
            orderbook: no_orderbook.clone(),
            timestamp: Utc::now(),
        });

        // First trigger should generate signals
        let signals = strategy.process_event(&no_event, &state).await.unwrap();
        assert_eq!(signals.len(), 2);

        // Second trigger should be blocked by cooldown
        let signals = strategy.process_event(&no_event, &state).await.unwrap();
        assert!(signals.is_empty());
    }

    #[tokio::test]
    async fn test_market_filtering() {
        let config = ArbitrageConfig {
            markets: vec!["other_market".to_string()],
            ..Default::default()
        };
        let mut strategy = ArbitrageStrategy::new(config);

        let mut state = MockStateProvider::new();
        state.add_market(create_test_market());
        strategy.initialize(&state).await.unwrap();

        let yes_event = SystemEvent::PriceChange(PriceChangeEvent {
            market_id: "test_market_1".to_string(),
            token_id: "yes_token_1".to_string(),
            old_price: Some(dec!(0.50)),
            new_price: dec!(0.45),
            price_change_pct: Some(dec!(-10.0)),
            timestamp: Utc::now(),
        });

        let signals = strategy.process_event(&yes_event, &state).await.unwrap();
        assert!(signals.is_empty());
    }

    #[tokio::test]
    async fn test_liquidity_check() {
        let config = ArbitrageConfig {
            min_edge_pct: dec!(1.0),
            min_liquidity_usd: dec!(10000),
            cooldown_secs: 0,
            ..Default::default()
        };
        let mut strategy = ArbitrageStrategy::new(config);

        let mut state = MockStateProvider::new();
        let mut market = create_test_market();
        market.liquidity = dec!(1000); // Below minimum
        state.add_market(market);
        strategy.initialize(&state).await.unwrap();

        // Create opportunity
        let yes_orderbook = create_orderbook("yes_token_1", "test_market_1", dec!(0.44), dec!(0.46));
        let no_orderbook = create_orderbook("no_token_1", "test_market_1", dec!(0.44), dec!(0.46));

        let yes_event = SystemEvent::OrderbookUpdate(OrderbookUpdateEvent {
            market_id: "test_market_1".to_string(),
            token_id: "yes_token_1".to_string(),
            orderbook: yes_orderbook,
            timestamp: Utc::now(),
        });
        strategy.process_event(&yes_event, &state).await.unwrap();

        let no_event = SystemEvent::OrderbookUpdate(OrderbookUpdateEvent {
            market_id: "test_market_1".to_string(),
            token_id: "no_token_1".to_string(),
            orderbook: no_orderbook,
            timestamp: Utc::now(),
        });

        // Should not generate signals due to insufficient liquidity
        let signals = strategy.process_event(&no_event, &state).await.unwrap();
        assert!(signals.is_empty());
    }

    #[tokio::test]
    async fn test_accepts_event() {
        let strategy = ArbitrageStrategy::new(ArbitrageConfig::default());

        let price_event = SystemEvent::PriceChange(PriceChangeEvent {
            market_id: "test".to_string(),
            token_id: "token".to_string(),
            old_price: Some(dec!(0.5)),
            new_price: dec!(0.6),
            price_change_pct: Some(dec!(20.0)),
            timestamp: Utc::now(),
        });
        assert!(strategy.accepts_event(&price_event));

        let orderbook = create_orderbook("token", "test", dec!(0.5), dec!(0.6));
        let orderbook_event = SystemEvent::OrderbookUpdate(OrderbookUpdateEvent {
            market_id: "test".to_string(),
            token_id: "token".to_string(),
            orderbook,
            timestamp: Utc::now(),
        });
        assert!(strategy.accepts_event(&orderbook_event));
    }

    #[tokio::test]
    async fn test_signal_generation_details() {
        let config = ArbitrageConfig {
            min_edge_pct: dec!(1.0),
            order_size_usd: dec!(50),
            cooldown_secs: 0,
            ..Default::default()
        };
        let mut strategy = ArbitrageStrategy::new(config);

        let mut state = MockStateProvider::new();
        state.add_market(create_test_market());
        strategy.initialize(&state).await.unwrap();

        // YES = 0.40, NO = 0.40, combined = 0.80 (20% edge)
        let yes_orderbook = create_orderbook("yes_token_1", "test_market_1", dec!(0.39), dec!(0.41));
        let no_orderbook = create_orderbook("no_token_1", "test_market_1", dec!(0.39), dec!(0.41));

        let yes_event = SystemEvent::OrderbookUpdate(OrderbookUpdateEvent {
            market_id: "test_market_1".to_string(),
            token_id: "yes_token_1".to_string(),
            orderbook: yes_orderbook,
            timestamp: Utc::now(),
        });
        strategy.process_event(&yes_event, &state).await.unwrap();

        let no_event = SystemEvent::OrderbookUpdate(OrderbookUpdateEvent {
            market_id: "test_market_1".to_string(),
            token_id: "no_token_1".to_string(),
            orderbook: no_orderbook,
            timestamp: Utc::now(),
        });

        let signals = strategy.process_event(&no_event, &state).await.unwrap();
        assert_eq!(signals.len(), 2);

        for signal in &signals {
            assert_eq!(signal.size_usd, dec!(50));
            assert_eq!(signal.order_type, OrderType::Fok);
            assert_eq!(signal.priority, Priority::High);
            assert!(signal.reason.contains("Arbitrage opportunity"));
        }
    }
}
