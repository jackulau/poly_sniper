//! Whale Following Strategy
//!
//! Generates trading signals based on detected whale activity, with support
//! for both following whales and fading detected spoofing.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use polysniper_core::{
    OrderType, Outcome, Priority, Side, StateProvider, Strategy, StrategyError,
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

use crate::whale_detector::{WhaleAlert, WhaleAlertType, WhaleDetectorConfig, WhaleDetector};

/// Type alias for backward compatibility
pub type WhaleConfig = WhaleDetectorConfig;
/// Type alias for backward compatibility
pub type WhaleActivity = WhaleAlert;
/// Type alias for backward compatibility
pub type WhaleActivityType = WhaleAlertType;

/// Configuration for whale-following strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleStrategyConfig {
    /// Whether the strategy is enabled
    pub enabled: bool,
    /// Minimum confidence score to generate a signal (0.0 to 1.0)
    pub min_confidence: Decimal,
    /// Whether to follow whales (trade in same direction as detected whales)
    pub follow_whales: bool,
    /// Whether to fade detected spoofing (trade opposite direction)
    pub fade_spoofing: bool,
    /// Order size in USD for whale-following trades
    pub order_size_usd: Decimal,
    /// Delay in seconds before acting on signals (to avoid front-running)
    pub signal_delay_secs: u64,
    /// Cooldown between signals for the same token (in seconds)
    #[serde(default = "default_signal_cooldown")]
    pub signal_cooldown_secs: u64,
    /// Markets to monitor (empty = all)
    #[serde(default)]
    pub markets: Vec<String>,
    /// Activity types to generate signals for
    #[serde(default = "default_activity_types")]
    pub signal_on_activities: Vec<String>,
}

fn default_signal_cooldown() -> u64 {
    300 // 5 minutes
}

fn default_activity_types() -> Vec<String> {
    vec![
        "SingleLargeTrade".to_string(),
        "CumulativeActivity".to_string(),
    ]
}

impl Default for WhaleStrategyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_confidence: dec!(0.7),
            follow_whales: true,
            fade_spoofing: false,
            order_size_usd: dec!(100),
            signal_delay_secs: 30,
            signal_cooldown_secs: default_signal_cooldown(),
            markets: Vec::new(),
            signal_on_activities: default_activity_types(),
        }
    }
}

impl WhaleStrategyConfig {
    /// Check if we should signal on a given activity type
    pub fn should_signal_on(&self, activity_type: &WhaleActivityType) -> bool {
        let type_str = match activity_type {
            WhaleActivityType::SingleLargeTrade => "SingleLargeTrade",
            WhaleActivityType::CumulativeActivity => "CumulativeActivity",
            WhaleActivityType::KnownWhaleActive => "KnownWhaleActive",
            WhaleActivityType::WhaleReversal => "WhaleReversal",
        };
        self.signal_on_activities.contains(&type_str.to_string())
    }
}

/// Pending signal awaiting delay
#[derive(Debug, Clone)]
struct PendingSignal {
    activity: WhaleActivity,
    token_id: TokenId,
    market_id: String,
    outcome: Outcome,
    execute_at: DateTime<Utc>,
}

/// Signal cooldown entry
#[derive(Debug, Clone)]
struct SignalCooldown {
    until: DateTime<Utc>,
}

/// Whale Following Strategy
pub struct WhaleStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: Arc<RwLock<WhaleStrategyConfig>>,
    detector: Arc<WhaleDetector>,
    /// Token to market mapping
    token_market_map: Arc<RwLock<HashMap<TokenId, (String, Outcome)>>>,
    /// Pending signals waiting for delay
    pending_signals: Arc<RwLock<Vec<PendingSignal>>>,
    /// Signal cooldowns per token
    cooldowns: Arc<RwLock<HashMap<TokenId, SignalCooldown>>>,
}

impl WhaleStrategy {
    /// Create a new whale strategy
    pub fn new(
        strategy_config: WhaleStrategyConfig,
        detector_config: WhaleConfig,
    ) -> Self {
        let enabled = strategy_config.enabled;
        Self {
            id: "whale_strategy".to_string(),
            name: "Whale Following Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config: Arc::new(RwLock::new(strategy_config)),
            detector: Arc::new(WhaleDetector::new(detector_config)),
            token_market_map: Arc::new(RwLock::new(HashMap::new())),
            pending_signals: Arc::new(RwLock::new(Vec::new())),
            cooldowns: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with existing detector
    pub fn with_detector(
        strategy_config: WhaleStrategyConfig,
        detector: Arc<WhaleDetector>,
    ) -> Self {
        let enabled = strategy_config.enabled;
        Self {
            id: "whale_strategy".to_string(),
            name: "Whale Following Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config: Arc::new(RwLock::new(strategy_config)),
            detector,
            token_market_map: Arc::new(RwLock::new(HashMap::new())),
            pending_signals: Arc::new(RwLock::new(Vec::new())),
            cooldowns: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the whale detector
    pub fn detector(&self) -> Arc<WhaleDetector> {
        self.detector.clone()
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
        let config = self.config.read().await;
        let until = Utc::now() + chrono::Duration::seconds(config.signal_cooldown_secs as i64);
        self.cooldowns
            .write()
            .await
            .insert(token_id.clone(), SignalCooldown { until });
    }

    /// Check if we should monitor this market
    fn should_monitor_market(&self, market_id: &str, markets: &[String]) -> bool {
        if markets.is_empty() {
            return true;
        }
        markets.contains(&market_id.to_string())
    }

    /// Process detected whale activity and potentially queue a signal
    async fn process_whale_activity(
        &self,
        activity: WhaleActivity,
        token_id: &TokenId,
        market_id: &str,
        outcome: Outcome,
    ) {
        let config = self.config.read().await;

        // Check confidence threshold
        if activity.confidence < config.min_confidence {
            debug!(
                token_id = %token_id,
                confidence = %activity.confidence,
                min_confidence = %config.min_confidence,
                "Activity below confidence threshold"
            );
            return;
        }

        // Check activity type filter
        if !config.should_signal_on(&activity.alert_type) {
            debug!(
                token_id = %token_id,
                activity_type = ?activity.alert_type,
                "Activity type not in signal list"
            );
            return;
        }

        // Handle whale reversal separately (similar to spoofing)
        if activity.alert_type == WhaleActivityType::WhaleReversal {
            if !config.fade_spoofing {
                debug!(token_id = %token_id, "Whale reversal detected but fade_spoofing disabled");
                return;
            }
        } else if !config.follow_whales {
            debug!(token_id = %token_id, "Whale activity detected but follow_whales disabled");
            return;
        }

        // Check cooldown
        if self.is_in_cooldown(token_id).await {
            debug!(token_id = %token_id, "Token in signal cooldown");
            return;
        }

        // Queue pending signal with delay
        let execute_at = Utc::now() + chrono::Duration::seconds(config.signal_delay_secs as i64);

        let pending = PendingSignal {
            activity,
            token_id: token_id.clone(),
            market_id: market_id.to_string(),
            outcome,
            execute_at,
        };

        self.pending_signals.write().await.push(pending);

        info!(
            token_id = %token_id,
            market_id = %market_id,
            delay_secs = %config.signal_delay_secs,
            "Queued whale signal for delayed execution"
        );
    }

    /// Check and emit any pending signals that are ready
    async fn emit_ready_signals(&self, state: &dyn StateProvider) -> Vec<TradeSignal> {
        let now = Utc::now();
        let config = self.config.read().await;
        let mut signals = Vec::new();

        let mut pending = self.pending_signals.write().await;

        // Partition into ready and not-ready
        let (ready, not_ready): (Vec<_>, Vec<_>) = pending
            .drain(..)
            .partition(|s| s.execute_at <= now);

        *pending = not_ready;

        for ps in ready {
            // Double-check cooldown
            if self.is_in_cooldown(&ps.token_id).await {
                continue;
            }

            // Get current price from state
            let price = state.get_price(&ps.token_id).await;

            // Determine side based on activity
            let side = self.determine_side(&ps.activity, &config);

            // Calculate size
            let size = match price {
                Some(p) if !p.is_zero() => config.order_size_usd / p,
                _ => Decimal::ZERO,
            };

            if size.is_zero() {
                warn!(token_id = %ps.token_id, "Cannot calculate size, skipping signal");
                continue;
            }

            let signal = TradeSignal {
                id: format!(
                    "sig_whale_{}_{}_{}",
                    ps.token_id,
                    Utc::now().timestamp_millis(),
                    rand_suffix()
                ),
                strategy_id: self.id.clone(),
                market_id: ps.market_id.clone(),
                token_id: ps.token_id.clone(),
                outcome: ps.outcome,
                side,
                price,
                size,
                size_usd: config.order_size_usd,
                order_type: OrderType::Gtc, // Use GTC for patient execution
                priority: Priority::Normal,
                timestamp: Utc::now(),
                reason: format!(
                    "Whale {:?} detected: {:?} side, ${} total, {} confidence",
                    ps.activity.alert_type,
                    ps.activity.whale_trade.side,
                    ps.activity.whale_trade.size_usd,
                    ps.activity.confidence
                ),
                metadata: serde_json::json!({
                    "whale_activity_type": format!("{:?}", ps.activity.alert_type),
                    "whale_side": format!("{:?}", ps.activity.whale_trade.side),
                    "whale_total_size_usd": ps.activity.whale_trade.size_usd.to_string(),
                    "whale_price": ps.activity.whale_trade.price.to_string(),
                    "whale_confidence": ps.activity.confidence.to_string(),
                    "signal_delay_secs": config.signal_delay_secs,
                }),
            };

            signals.push(signal);

            // Set cooldown
            self.set_cooldown(&ps.token_id).await;

            info!(
                token_id = %ps.token_id,
                market_id = %ps.market_id,
                side = ?side,
                size_usd = %config.order_size_usd,
                "Emitted whale following signal"
            );
        }

        signals
    }

    /// Determine trade side based on whale activity and config
    fn determine_side(&self, activity: &WhaleActivity, config: &WhaleStrategyConfig) -> Side {
        let whale_side = activity.whale_trade.side;
        if activity.alert_type == WhaleActivityType::WhaleReversal && config.fade_spoofing {
            // Fade reversal: trade opposite direction
            match whale_side {
                Side::Buy => Side::Sell,
                Side::Sell => Side::Buy,
            }
        } else {
            // Follow whales: trade same direction
            whale_side
        }
    }
}

#[async_trait]
impl Strategy for WhaleStrategy {
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
        // First, check for any ready pending signals
        let signals = self.emit_ready_signals(state).await;

        // Only process orderbook updates
        let (token_id, market_id, orderbook) = match event {
            SystemEvent::OrderbookUpdate(e) => {
                (e.token_id.clone(), e.market_id.clone(), &e.orderbook)
            }
            _ => return Ok(signals),
        };

        // Check market filter
        {
            let config = self.config.read().await;
            if !self.should_monitor_market(&market_id, &config.markets) {
                return Ok(signals);
            }
        }

        // Get market info for outcome
        let outcome = {
            let token_map = self.token_market_map.read().await;
            token_map
                .get(&token_id)
                .map(|(_, o)| *o)
                .unwrap_or(Outcome::Yes)
        };

        // Note: The whale detector processes individual trades, not orderbooks
        // For orderbook-based whale detection, we would need a different approach
        // For now, we just return any pending ready signals
        let _ = (token_id, orderbook, outcome);

        Ok(signals)
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(event, SystemEvent::OrderbookUpdate(_))
    }

    async fn initialize(&mut self, state: &dyn StateProvider) -> Result<(), StrategyError> {
        let config = self.config.read().await;
        info!(
            strategy_id = %self.id,
            min_confidence = %config.min_confidence,
            follow_whales = %config.follow_whales,
            fade_spoofing = %config.fade_spoofing,
            order_size_usd = %config.order_size_usd,
            signal_delay_secs = %config.signal_delay_secs,
            "Initializing whale following strategy"
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
        // Parse the combined config file
        #[derive(Deserialize)]
        struct CombinedConfig {
            whale_detector: Option<WhaleConfig>,
            whale_strategy: Option<WhaleStrategyConfig>,
        }

        let parsed: CombinedConfig = toml::from_str(config_content)
            .map_err(|e| StrategyError::ConfigError(format!("Failed to parse config: {}", e)))?;

        if let Some(_detector_config) = parsed.whale_detector {
            // Note: WhaleDetector reload_config requires mutable access
            // which is not available through Arc. Config updates would need
            // to be handled differently (e.g., using RwLock wrapper)
            warn!("Whale detector config reload not implemented for Arc<WhaleDetector>");
        }

        if let Some(strategy_config) = parsed.whale_strategy {
            *self.config.write().await = strategy_config;
        }

        info!(strategy_id = %self.id, "Reloaded whale strategy config");
        Ok(())
    }

    fn config_name(&self) -> &str {
        "whale_tracking"
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
    use polysniper_core::PriceLevel;

    fn create_test_orderbook(
        token_id: &str,
        bids: Vec<(Decimal, Decimal)>,
        asks: Vec<(Decimal, Decimal)>,
    ) -> polysniper_core::Orderbook {
        polysniper_core::Orderbook {
            token_id: token_id.to_string(),
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
    fn test_config_should_signal_on() {
        let config = WhaleStrategyConfig {
            signal_on_activities: vec!["LargeResting".to_string(), "Accumulation".to_string()],
            ..Default::default()
        };

        assert!(config.should_signal_on(&WhaleActivityType::LargeResting));
        assert!(config.should_signal_on(&WhaleActivityType::Accumulation));
        assert!(!config.should_signal_on(&WhaleActivityType::Iceberg));
        assert!(!config.should_signal_on(&WhaleActivityType::Spoofing));
    }

    #[test]
    fn test_determine_side_follow() {
        let config = WhaleStrategyConfig {
            follow_whales: true,
            fade_spoofing: false,
            ..Default::default()
        };
        let detector_config = WhaleConfig::default();
        let strategy = WhaleStrategy::new(config.clone(), detector_config);

        // Following whale buy activity should result in buy
        let buy_activity = WhaleActivity {
            activity_type: WhaleActivityType::LargeResting,
            side: Side::Buy,
            total_size_usd: dec!(10000),
            num_orders: 1,
            avg_price: dec!(0.50),
            first_seen: Utc::now(),
            last_seen: Utc::now(),
            confidence: dec!(0.9),
        };

        assert_eq!(strategy.determine_side(&buy_activity, &config), Side::Buy);

        // Following whale sell activity should result in sell
        let sell_activity = WhaleActivity {
            activity_type: WhaleActivityType::LargeResting,
            side: Side::Sell,
            total_size_usd: dec!(10000),
            num_orders: 1,
            avg_price: dec!(0.50),
            first_seen: Utc::now(),
            last_seen: Utc::now(),
            confidence: dec!(0.9),
        };

        assert_eq!(strategy.determine_side(&sell_activity, &config), Side::Sell);
    }

    #[test]
    fn test_determine_side_fade_spoofing() {
        let config = WhaleStrategyConfig {
            follow_whales: true,
            fade_spoofing: true,
            ..Default::default()
        };
        let detector_config = WhaleConfig::default();
        let strategy = WhaleStrategy::new(config.clone(), detector_config);

        // Fading buy spoofing should result in sell
        let spoof_buy = WhaleActivity {
            activity_type: WhaleActivityType::Spoofing,
            side: Side::Buy,
            total_size_usd: dec!(10000),
            num_orders: 1,
            avg_price: dec!(0.50),
            first_seen: Utc::now(),
            last_seen: Utc::now(),
            confidence: dec!(0.9),
        };

        assert_eq!(strategy.determine_side(&spoof_buy, &config), Side::Sell);

        // Fading sell spoofing should result in buy
        let spoof_sell = WhaleActivity {
            activity_type: WhaleActivityType::Spoofing,
            side: Side::Sell,
            total_size_usd: dec!(10000),
            num_orders: 1,
            avg_price: dec!(0.50),
            first_seen: Utc::now(),
            last_seen: Utc::now(),
            confidence: dec!(0.9),
        };

        assert_eq!(strategy.determine_side(&spoof_sell, &config), Side::Buy);
    }

    #[test]
    fn test_market_filter() {
        let config = WhaleStrategyConfig {
            markets: vec!["market_a".to_string(), "market_b".to_string()],
            ..Default::default()
        };
        let detector_config = WhaleConfig::default();
        let strategy = WhaleStrategy::new(config, detector_config);

        assert!(strategy.should_monitor_market("market_a", &["market_a".to_string(), "market_b".to_string()]));
        assert!(strategy.should_monitor_market("market_b", &["market_a".to_string(), "market_b".to_string()]));
        assert!(!strategy.should_monitor_market("market_c", &["market_a".to_string(), "market_b".to_string()]));
    }

    #[test]
    fn test_market_filter_empty() {
        let config = WhaleStrategyConfig {
            markets: vec![],
            ..Default::default()
        };
        let detector_config = WhaleConfig::default();
        let strategy = WhaleStrategy::new(config, detector_config);

        // Empty filter = all markets
        assert!(strategy.should_monitor_market("any_market", &[]));
    }

    #[tokio::test]
    async fn test_cooldown_enforcement() {
        let config = WhaleStrategyConfig {
            signal_cooldown_secs: 600,
            ..Default::default()
        };
        let detector_config = WhaleConfig::default();
        let strategy = WhaleStrategy::new(config, detector_config);

        let token_id = "test_token".to_string();

        // Initially not in cooldown
        assert!(!strategy.is_in_cooldown(&token_id).await);

        // Set cooldown
        strategy.set_cooldown(&token_id).await;

        // Now in cooldown
        assert!(strategy.is_in_cooldown(&token_id).await);
    }

    #[tokio::test]
    async fn test_register_token() {
        let config = WhaleStrategyConfig::default();
        let detector_config = WhaleConfig::default();
        let strategy = WhaleStrategy::new(config, detector_config);

        let token_id = "test_token".to_string();
        let market_id = "test_market".to_string();

        strategy
            .register_token(token_id.clone(), market_id.clone(), Outcome::Yes)
            .await;

        let map = strategy.token_market_map.read().await;
        assert!(map.contains_key(&token_id));

        let (stored_market, stored_outcome) = map.get(&token_id).unwrap();
        assert_eq!(stored_market, &market_id);
        assert_eq!(*stored_outcome, Outcome::Yes);
    }

    #[tokio::test]
    async fn test_detector_integration() {
        let strategy_config = WhaleStrategyConfig {
            min_confidence: dec!(0.5),
            signal_delay_secs: 0, // No delay for testing
            signal_cooldown_secs: 0, // No cooldown for testing
            ..Default::default()
        };
        let detector_config = WhaleConfig::default();

        let strategy = WhaleStrategy::new(strategy_config, detector_config);
        let detector = strategy.detector();

        // Verify detector is available
        assert!(detector.is_enabled());
    }
}
