//! Order Book Imbalance Strategy
//!
//! Detects significant bid/ask volume asymmetry and trades in the direction
//! of the imbalance, anticipating price movement.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use polysniper_core::{
    OrderType, Orderbook, Outcome, Priority, Side, StateProvider, Strategy, StrategyError,
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

/// Orderbook imbalance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookImbalanceConfig {
    pub enabled: bool,
    /// Minimum imbalance ratio to trigger (e.g., 2.0 = 2:1 bid/ask ratio)
    pub imbalance_threshold: Decimal,
    /// Number of price levels to analyze
    #[serde(default = "default_depth_levels")]
    pub depth_levels: usize,
    /// Order size in USD
    pub order_size_usd: Decimal,
    /// Minimum liquidity to trade
    #[serde(default = "default_min_liquidity")]
    pub min_liquidity_usd: Decimal,
    /// Cooldown period in seconds between signals for the same token
    #[serde(default = "default_cooldown")]
    pub cooldown_secs: u64,
    /// Markets to monitor (empty = all)
    #[serde(default)]
    pub markets: Vec<String>,
    /// Weight by value (price * size) instead of just size
    #[serde(default = "default_value_weighting")]
    pub use_value_weighting: bool,
}

fn default_depth_levels() -> usize {
    5
}

fn default_min_liquidity() -> Decimal {
    dec!(1000)
}

fn default_cooldown() -> u64 {
    60
}

fn default_value_weighting() -> bool {
    true
}

impl Default for OrderbookImbalanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            imbalance_threshold: dec!(2.0),
            depth_levels: default_depth_levels(),
            order_size_usd: dec!(100),
            min_liquidity_usd: default_min_liquidity(),
            cooldown_secs: default_cooldown(),
            markets: Vec::new(),
            use_value_weighting: default_value_weighting(),
        }
    }
}

/// Result of imbalance calculation
#[derive(Debug, Clone)]
struct ImbalanceResult {
    bid_volume: Decimal,
    ask_volume: Decimal,
    ratio: Decimal,
}

/// Cooldown entry
#[derive(Debug, Clone)]
struct CooldownEntry {
    until: DateTime<Utc>,
}

/// Order Book Imbalance Strategy
pub struct OrderbookImbalanceStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: OrderbookImbalanceConfig,
    /// Cooldowns per token
    cooldowns: Arc<RwLock<HashMap<TokenId, CooldownEntry>>>,
    /// Token to market mapping
    token_market_map: Arc<RwLock<HashMap<TokenId, (String, Outcome)>>>,
}

impl OrderbookImbalanceStrategy {
    /// Create a new orderbook imbalance strategy
    pub fn new(config: OrderbookImbalanceConfig) -> Self {
        let enabled = config.enabled;
        Self {
            id: "orderbook_imbalance".to_string(),
            name: "Order Book Imbalance Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config,
            cooldowns: Arc::new(RwLock::new(HashMap::new())),
            token_market_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a token for monitoring
    pub async fn register_token(&self, token_id: TokenId, market_id: String, outcome: Outcome) {
        self.token_market_map
            .write()
            .await
            .insert(token_id, (market_id, outcome));
    }

    /// Check if token is in cooldown
    async fn is_in_cooldown(&self, token_id: &TokenId) -> bool {
        let cooldowns = self.cooldowns.read().await;
        if let Some(entry) = cooldowns.get(token_id) {
            return Utc::now() < entry.until;
        }
        false
    }

    /// Set cooldown for a token
    async fn set_cooldown(&self, token_id: &TokenId) {
        let until = Utc::now() + chrono::Duration::seconds(self.config.cooldown_secs as i64);
        self.cooldowns
            .write()
            .await
            .insert(token_id.clone(), CooldownEntry { until });
    }

    /// Calculate orderbook imbalance
    fn calculate_imbalance(&self, orderbook: &Orderbook) -> Option<ImbalanceResult> {
        // Sum bid volume across depth_levels
        let bid_volume: Decimal = orderbook
            .bids
            .iter()
            .take(self.config.depth_levels)
            .map(|l| {
                if self.config.use_value_weighting {
                    l.price * l.size
                } else {
                    l.size
                }
            })
            .sum();

        // Sum ask volume across depth_levels
        let ask_volume: Decimal = orderbook
            .asks
            .iter()
            .take(self.config.depth_levels)
            .map(|l| {
                if self.config.use_value_weighting {
                    l.price * l.size
                } else {
                    l.size
                }
            })
            .sum();

        // Need both sides to calculate ratio
        if bid_volume.is_zero() || ask_volume.is_zero() {
            return None;
        }

        let ratio = bid_volume / ask_volume;

        Some(ImbalanceResult {
            bid_volume,
            ask_volume,
            ratio,
        })
    }

    /// Determine if we should monitor this market
    fn should_monitor_market(&self, market_id: &str) -> bool {
        if self.config.markets.is_empty() {
            return true;
        }
        self.config.markets.contains(&market_id.to_string())
    }
}

#[async_trait]
impl Strategy for OrderbookImbalanceStrategy {
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

        // Only process orderbook updates
        let (token_id, market_id, orderbook) = match event {
            SystemEvent::OrderbookUpdate(e) => {
                (e.token_id.clone(), e.market_id.clone(), &e.orderbook)
            }
            _ => return Ok(signals),
        };

        // Check if we should monitor this market
        if !self.should_monitor_market(&market_id) {
            return Ok(signals);
        }

        // Check cooldown
        if self.is_in_cooldown(&token_id).await {
            debug!(token_id = %token_id, "Token in cooldown, skipping");
            return Ok(signals);
        }

        // Calculate imbalance
        let imbalance = match self.calculate_imbalance(orderbook) {
            Some(result) => result,
            None => {
                debug!(token_id = %token_id, "Could not calculate imbalance (empty orderbook)");
                return Ok(signals);
            }
        };

        debug!(
            token_id = %token_id,
            bid_volume = %imbalance.bid_volume,
            ask_volume = %imbalance.ask_volume,
            ratio = %imbalance.ratio,
            threshold = %self.config.imbalance_threshold,
            "Calculated orderbook imbalance"
        );

        // Check if imbalance exceeds threshold
        let inverse_threshold = Decimal::ONE / self.config.imbalance_threshold;
        let (should_signal, side, direction) =
            if imbalance.ratio >= self.config.imbalance_threshold {
                // Heavy bids: buy pressure, price likely to rise -> Buy
                (true, Side::Buy, "bullish")
            } else if imbalance.ratio <= inverse_threshold {
                // Heavy asks: sell pressure, price likely to fall -> Sell
                (true, Side::Sell, "bearish")
            } else {
                (false, Side::Buy, "neutral")
            };

        if !should_signal {
            return Ok(signals);
        }

        info!(
            token_id = %token_id,
            market_id = %market_id,
            ratio = %imbalance.ratio,
            direction = %direction,
            side = ?side,
            "Orderbook imbalance detected!"
        );

        // Get market info for outcome
        let token_map = self.token_market_map.read().await;
        let outcome = token_map
            .get(&token_id)
            .map(|(_, o)| *o)
            .unwrap_or(Outcome::Yes);
        drop(token_map);

        // Check liquidity if we have state
        if let Some(market) = state.get_market(&market_id).await {
            if market.liquidity < self.config.min_liquidity_usd {
                warn!(
                    market_id = %market_id,
                    liquidity = %market.liquidity,
                    min_required = %self.config.min_liquidity_usd,
                    "Insufficient liquidity, skipping signal"
                );
                return Ok(signals);
            }
        }

        // Get price from orderbook mid
        let price = orderbook.mid_price();

        // Calculate size
        let size = match price {
            Some(p) if !p.is_zero() => self.config.order_size_usd / p,
            _ => Decimal::ZERO,
        };

        let signal = TradeSignal {
            id: format!(
                "sig_imbalance_{}_{}_{}",
                token_id,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id,
            token_id: token_id.clone(),
            outcome,
            side,
            price,
            size,
            size_usd: self.config.order_size_usd,
            order_type: OrderType::Fok, // Use FOK for imbalance trading
            priority: Priority::High,   // High priority for time-sensitive
            timestamp: Utc::now(),
            reason: format!(
                "Orderbook imbalance {:.2}:1 detected ({} bias)",
                imbalance.ratio, direction
            ),
            metadata: serde_json::json!({
                "imbalance_ratio": imbalance.ratio.to_string(),
                "bid_volume": imbalance.bid_volume.to_string(),
                "ask_volume": imbalance.ask_volume.to_string(),
                "direction": direction,
                "threshold": self.config.imbalance_threshold.to_string(),
                "depth_levels": self.config.depth_levels,
                "value_weighted": self.config.use_value_weighting,
            }),
        };

        signals.push(signal);

        // Set cooldown
        self.set_cooldown(&token_id).await;

        Ok(signals)
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(event, SystemEvent::OrderbookUpdate(_))
    }

    async fn initialize(&mut self, state: &dyn StateProvider) -> Result<(), StrategyError> {
        info!(
            strategy_id = %self.id,
            threshold = %self.config.imbalance_threshold,
            depth_levels = %self.config.depth_levels,
            value_weighted = %self.config.use_value_weighting,
            "Initializing orderbook imbalance strategy"
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

        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }

    async fn reload_config(&mut self, config_content: &str) -> Result<(), StrategyError> {
        let new_config: OrderbookImbalanceConfig = toml::from_str(config_content)
            .map_err(|e| StrategyError::ConfigError(format!("Failed to parse config: {}", e)))?;

        self.config = new_config;
        tracing::info!(strategy_id = %self.id, "Reloaded orderbook imbalance strategy config");
        Ok(())
    }

    fn config_name(&self) -> &str {
        "orderbook_imbalance"
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
    use polysniper_core::{Orderbook, PriceLevel};

    fn create_test_orderbook(bids: Vec<(Decimal, Decimal)>, asks: Vec<(Decimal, Decimal)>) -> Orderbook {
        Orderbook {
            token_id: "test_token".to_string(),
            market_id: "test_market".to_string(),
            bids: bids
                .into_iter()
                .map(|(price, size)| PriceLevel { price, size })
                .collect(),
            asks: asks
                .into_iter()
                .map(|(price, size)| PriceLevel { price, size })
                .collect(),
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_imbalance_calculation_size_weighted() {
        let config = OrderbookImbalanceConfig {
            use_value_weighting: false,
            depth_levels: 3,
            ..Default::default()
        };
        let strategy = OrderbookImbalanceStrategy::new(config);

        // Bids: 100 + 200 + 150 = 450
        // Asks: 50 + 75 + 25 = 150
        // Ratio: 450/150 = 3.0
        let orderbook = create_test_orderbook(
            vec![
                (dec!(0.50), dec!(100)),
                (dec!(0.49), dec!(200)),
                (dec!(0.48), dec!(150)),
            ],
            vec![
                (dec!(0.51), dec!(50)),
                (dec!(0.52), dec!(75)),
                (dec!(0.53), dec!(25)),
            ],
        );

        let result = strategy.calculate_imbalance(&orderbook).unwrap();
        assert_eq!(result.bid_volume, dec!(450));
        assert_eq!(result.ask_volume, dec!(150));
        assert_eq!(result.ratio, dec!(3));
    }

    #[test]
    fn test_imbalance_calculation_value_weighted() {
        let config = OrderbookImbalanceConfig {
            use_value_weighting: true,
            depth_levels: 2,
            ..Default::default()
        };
        let strategy = OrderbookImbalanceStrategy::new(config);

        // Bids: (0.50 * 100) + (0.49 * 100) = 50 + 49 = 99
        // Asks: (0.51 * 100) + (0.52 * 100) = 51 + 52 = 103
        // Ratio: 99/103 ≈ 0.96
        let orderbook = create_test_orderbook(
            vec![(dec!(0.50), dec!(100)), (dec!(0.49), dec!(100))],
            vec![(dec!(0.51), dec!(100)), (dec!(0.52), dec!(100))],
        );

        let result = strategy.calculate_imbalance(&orderbook).unwrap();
        assert_eq!(result.bid_volume, dec!(99));
        assert_eq!(result.ask_volume, dec!(103));
    }

    #[test]
    fn test_imbalance_empty_orderbook() {
        let config = OrderbookImbalanceConfig::default();
        let strategy = OrderbookImbalanceStrategy::new(config);

        let orderbook = create_test_orderbook(vec![], vec![]);
        assert!(strategy.calculate_imbalance(&orderbook).is_none());
    }

    #[test]
    fn test_imbalance_one_side_empty() {
        let config = OrderbookImbalanceConfig::default();
        let strategy = OrderbookImbalanceStrategy::new(config);

        let orderbook = create_test_orderbook(
            vec![(dec!(0.50), dec!(100))],
            vec![],
        );
        assert!(strategy.calculate_imbalance(&orderbook).is_none());
    }

    #[test]
    fn test_bullish_imbalance_detection() {
        let config = OrderbookImbalanceConfig {
            imbalance_threshold: dec!(2.0),
            use_value_weighting: false,
            depth_levels: 3,
            ..Default::default()
        };
        let strategy = OrderbookImbalanceStrategy::new(config);

        // Ratio = 3:1 -> exceeds 2.0 threshold -> bullish
        let orderbook = create_test_orderbook(
            vec![
                (dec!(0.50), dec!(100)),
                (dec!(0.49), dec!(100)),
                (dec!(0.48), dec!(100)),
            ],
            vec![
                (dec!(0.51), dec!(50)),
                (dec!(0.52), dec!(25)),
                (dec!(0.53), dec!(25)),
            ],
        );

        let result = strategy.calculate_imbalance(&orderbook).unwrap();
        assert!(result.ratio >= dec!(2.0));
    }

    #[test]
    fn test_bearish_imbalance_detection() {
        let config = OrderbookImbalanceConfig {
            imbalance_threshold: dec!(2.0),
            use_value_weighting: false,
            depth_levels: 3,
            ..Default::default()
        };
        let strategy = OrderbookImbalanceStrategy::new(config);

        // Ratio = 1/3 = 0.33 -> below 1/2.0 = 0.5 threshold -> bearish
        let orderbook = create_test_orderbook(
            vec![
                (dec!(0.50), dec!(50)),
                (dec!(0.49), dec!(25)),
                (dec!(0.48), dec!(25)),
            ],
            vec![
                (dec!(0.51), dec!(100)),
                (dec!(0.52), dec!(100)),
                (dec!(0.53), dec!(100)),
            ],
        );

        let result = strategy.calculate_imbalance(&orderbook).unwrap();
        assert!(result.ratio <= dec!(0.5));
    }

    #[test]
    fn test_neutral_imbalance() {
        let config = OrderbookImbalanceConfig {
            imbalance_threshold: dec!(2.0),
            use_value_weighting: false,
            depth_levels: 2,
            ..Default::default()
        };
        let strategy = OrderbookImbalanceStrategy::new(config);

        // Ratio = 1.5 -> between 0.5 and 2.0 -> neutral (no signal)
        let orderbook = create_test_orderbook(
            vec![(dec!(0.50), dec!(150)), (dec!(0.49), dec!(150))],
            vec![(dec!(0.51), dec!(100)), (dec!(0.52), dec!(100))],
        );

        let result = strategy.calculate_imbalance(&orderbook).unwrap();
        assert!(result.ratio > dec!(0.5) && result.ratio < dec!(2.0));
    }

    #[test]
    fn test_depth_levels_limiting() {
        let config = OrderbookImbalanceConfig {
            use_value_weighting: false,
            depth_levels: 2,
            ..Default::default()
        };
        let strategy = OrderbookImbalanceStrategy::new(config);

        // Only first 2 levels should be counted
        // Bids: 100 + 100 = 200 (third level ignored)
        // Asks: 50 + 50 = 100 (third level ignored)
        let orderbook = create_test_orderbook(
            vec![
                (dec!(0.50), dec!(100)),
                (dec!(0.49), dec!(100)),
                (dec!(0.48), dec!(1000)), // Should be ignored
            ],
            vec![
                (dec!(0.51), dec!(50)),
                (dec!(0.52), dec!(50)),
                (dec!(0.53), dec!(1000)), // Should be ignored
            ],
        );

        let result = strategy.calculate_imbalance(&orderbook).unwrap();
        assert_eq!(result.bid_volume, dec!(200));
        assert_eq!(result.ask_volume, dec!(100));
    }

    #[test]
    fn test_market_filter() {
        let config = OrderbookImbalanceConfig {
            markets: vec!["market_a".to_string(), "market_b".to_string()],
            ..Default::default()
        };
        let strategy = OrderbookImbalanceStrategy::new(config);

        assert!(strategy.should_monitor_market("market_a"));
        assert!(strategy.should_monitor_market("market_b"));
        assert!(!strategy.should_monitor_market("market_c"));
    }

    #[test]
    fn test_market_filter_empty() {
        let config = OrderbookImbalanceConfig {
            markets: vec![],
            ..Default::default()
        };
        let strategy = OrderbookImbalanceStrategy::new(config);

        // Empty filter = all markets
        assert!(strategy.should_monitor_market("any_market"));
    }
}
