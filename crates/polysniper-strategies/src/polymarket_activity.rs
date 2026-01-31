//! Polymarket Activity Strategy
//!
//! Trading strategy that processes smart money signals, volume anomalies,
//! and comment activity to generate trade signals.

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use polysniper_core::{
    CommentActivityEvent, MarketId, OrderType, Outcome, Priority, Side, SmartMoneySignalEvent,
    StateProvider, Strategy, StrategyError, SystemEvent, TradeSignal, TraderAction,
    VolumeAnomalyEvent,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Configuration for smart money following
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartMoneyConfig {
    /// Enable smart money tracking
    pub enabled: bool,
    /// Only track traders in top N on leaderboard
    pub min_trader_rank: u32,
    /// Minimum position size in USD to trigger signal
    pub min_position_size_usd: Decimal,
    /// How aggressively to follow (0.0-1.0)
    pub follow_strength: Decimal,
    /// Cooldown between signals for same market in seconds
    pub cooldown_secs: u64,
    /// Order size in USD for follow trades
    pub order_size_usd: Decimal,
}

impl Default for SmartMoneyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_trader_rank: 100,
            min_position_size_usd: dec!(1000),
            follow_strength: dec!(0.5),
            cooldown_secs: 600,
            order_size_usd: dec!(25),
        }
    }
}

/// Configuration for volume anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeAnomalyConfig {
    /// Enable volume anomaly detection
    pub enabled: bool,
    /// Minimum volume ratio (current/average) to trigger
    pub min_volume_ratio: Decimal,
    /// Number of periods for average calculation
    pub lookback_periods: u32,
    /// Cooldown between signals in seconds
    pub cooldown_secs: u64,
    /// Order size in USD
    pub order_size_usd: Decimal,
}

impl Default for VolumeAnomalyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_volume_ratio: dec!(3.0),
            lookback_periods: 24,
            cooldown_secs: 300,
            order_size_usd: dec!(50),
        }
    }
}

/// Configuration for comment activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommentActivityConfig {
    /// Enable comment activity tracking
    pub enabled: bool,
    /// Minimum comments per hour to trigger
    pub min_comments_per_hour: u32,
    /// Weight to give sentiment (0.0-1.0)
    pub sentiment_weight: Decimal,
    /// Cooldown between signals in seconds
    pub cooldown_secs: u64,
    /// Order size in USD
    pub order_size_usd: Decimal,
}

impl Default for CommentActivityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_comments_per_hour: 10,
            sentiment_weight: dec!(0.3),
            cooldown_secs: 600,
            order_size_usd: dec!(25),
        }
    }
}

/// Main strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolymarketActivityStrategyConfig {
    /// Enable/disable the strategy
    pub enabled: bool,
    /// Smart money tracking configuration
    #[serde(default)]
    pub smart_money_tracking: SmartMoneyConfig,
    /// Volume anomaly detection configuration
    #[serde(default)]
    pub volume_anomaly: VolumeAnomalyConfig,
    /// Comment activity configuration
    #[serde(default)]
    pub comment_activity: CommentActivityConfig,
    /// Maximum entry price for trades
    #[serde(default = "default_max_entry_price")]
    pub max_entry_price: Decimal,
    /// Priority for generated signals
    #[serde(default)]
    pub signal_priority: Priority,
}

fn default_max_entry_price() -> Decimal {
    dec!(0.90)
}

impl Default for PolymarketActivityStrategyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            smart_money_tracking: SmartMoneyConfig::default(),
            volume_anomaly: VolumeAnomalyConfig::default(),
            comment_activity: CommentActivityConfig::default(),
            max_entry_price: default_max_entry_price(),
            signal_priority: Priority::High,
        }
    }
}

/// Polymarket activity-based trading strategy
pub struct PolymarketActivityStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: Arc<RwLock<PolymarketActivityStrategyConfig>>,
    /// Last signal time per market for cooldown
    last_smart_money_signal: Arc<RwLock<HashMap<MarketId, DateTime<Utc>>>>,
    last_volume_signal: Arc<RwLock<HashMap<MarketId, DateTime<Utc>>>>,
    last_comment_signal: Arc<RwLock<HashMap<MarketId, DateTime<Utc>>>>,
    /// Track smart money positions for consensus
    smart_money_positions: Arc<RwLock<HashMap<MarketId, SmartMoneyConsensus>>>,
}

/// Tracks consensus among smart money traders
#[derive(Debug, Clone, Default)]
struct SmartMoneyConsensus {
    yes_weight: Decimal,
    no_weight: Decimal,
    last_updated: Option<DateTime<Utc>>,
}

#[allow(dead_code)]
impl SmartMoneyConsensus {
    fn add_signal(&mut self, outcome: Outcome, size_usd: Decimal, rank: u32) {
        // Weight by inverse of rank (top traders count more)
        let weight = size_usd / Decimal::from(rank.max(1));
        match outcome {
            Outcome::Yes => self.yes_weight += weight,
            Outcome::No => self.no_weight += weight,
        }
        self.last_updated = Some(Utc::now());
    }

    fn get_consensus(&self) -> Option<Outcome> {
        if self.yes_weight > self.no_weight * dec!(1.5) {
            Some(Outcome::Yes)
        } else if self.no_weight > self.yes_weight * dec!(1.5) {
            Some(Outcome::No)
        } else {
            None
        }
    }

    fn confidence(&self) -> Decimal {
        let total = self.yes_weight + self.no_weight;
        if total.is_zero() {
            return Decimal::ZERO;
        }
        let diff = (self.yes_weight - self.no_weight).abs();
        diff / total
    }
}

impl PolymarketActivityStrategy {
    /// Create a new Polymarket activity strategy
    pub fn new(config: PolymarketActivityStrategyConfig) -> Self {
        let enabled = config.enabled;

        Self {
            id: "polymarket_activity".to_string(),
            name: "Polymarket Activity Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config: Arc::new(RwLock::new(config)),
            last_smart_money_signal: Arc::new(RwLock::new(HashMap::new())),
            last_volume_signal: Arc::new(RwLock::new(HashMap::new())),
            last_comment_signal: Arc::new(RwLock::new(HashMap::new())),
            smart_money_positions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check cooldown for a signal type
    async fn check_cooldown(
        last_signals: &RwLock<HashMap<MarketId, DateTime<Utc>>>,
        market_id: &MarketId,
        cooldown_secs: u64,
    ) -> bool {
        let last = last_signals.read().await;
        if let Some(last_time) = last.get(market_id) {
            let cooldown = Duration::seconds(cooldown_secs as i64);
            if Utc::now() - *last_time < cooldown {
                return false;
            }
        }
        true
    }

    /// Record signal time for cooldown
    async fn record_signal_time(
        last_signals: &RwLock<HashMap<MarketId, DateTime<Utc>>>,
        market_id: &MarketId,
    ) {
        let mut last = last_signals.write().await;
        last.insert(market_id.clone(), Utc::now());
    }

    /// Process a smart money signal
    async fn process_smart_money_signal(
        &self,
        event: &SmartMoneySignalEvent,
        state: &dyn StateProvider,
    ) -> Result<Option<TradeSignal>, StrategyError> {
        let config = self.config.read().await;

        if !config.smart_money_tracking.enabled {
            return Ok(None);
        }

        // Check trader rank
        if event.trader_rank > config.smart_money_tracking.min_trader_rank {
            debug!(
                rank = event.trader_rank,
                max = config.smart_money_tracking.min_trader_rank,
                "Trader rank too low, skipping"
            );
            return Ok(None);
        }

        // Check position size
        if event.size_usd < config.smart_money_tracking.min_position_size_usd {
            debug!(
                size = %event.size_usd,
                min = %config.smart_money_tracking.min_position_size_usd,
                "Position size too small, skipping"
            );
            return Ok(None);
        }

        // Check cooldown
        if !Self::check_cooldown(
            &self.last_smart_money_signal,
            &event.market_id,
            config.smart_money_tracking.cooldown_secs,
        )
        .await
        {
            debug!(market = %event.market_id, "Cooldown active for smart money signal");
            return Ok(None);
        }

        // Update consensus tracking
        {
            let mut positions = self.smart_money_positions.write().await;
            let consensus = positions
                .entry(event.market_id.clone())
                .or_insert_with(SmartMoneyConsensus::default);

            match event.action {
                TraderAction::Buy | TraderAction::NewPosition => {
                    consensus.add_signal(event.outcome, event.size_usd, event.trader_rank);
                }
                TraderAction::Sell | TraderAction::ClosePosition => {
                    let opposite = match event.outcome {
                        Outcome::Yes => Outcome::No,
                        Outcome::No => Outcome::Yes,
                    };
                    consensus.add_signal(opposite, event.size_usd, event.trader_rank);
                }
            }
        }

        // Determine trade direction
        let (side, outcome) = match event.action {
            TraderAction::Buy | TraderAction::NewPosition => (Side::Buy, event.outcome),
            TraderAction::Sell | TraderAction::ClosePosition => {
                // When trader sells, we might want to follow
                (Side::Sell, event.outcome)
            }
        };

        // Only generate buy signals (following into positions)
        if side != Side::Buy {
            return Ok(None);
        }

        // Get market info
        let market = match state.get_market(&event.market_id).await {
            Some(m) => m,
            None => {
                warn!(market = %event.market_id, "Market not found");
                return Ok(None);
            }
        };

        // Get token ID based on outcome
        let token_id = match outcome {
            Outcome::Yes => &market.yes_token_id,
            Outcome::No => &market.no_token_id,
        };

        // Check current price
        let current_price = state.get_price(token_id).await;
        if let Some(price) = current_price {
            if price > config.max_entry_price {
                debug!(
                    price = %price,
                    max = %config.max_entry_price,
                    "Price too high for entry"
                );
                return Ok(None);
            }
        }

        let entry_price = current_price.unwrap_or(config.max_entry_price);
        let order_size_usd = config.smart_money_tracking.order_size_usd
            * config.smart_money_tracking.follow_strength;

        let size = if entry_price.is_zero() {
            Decimal::ZERO
        } else {
            order_size_usd / entry_price
        };

        // Record signal time
        Self::record_signal_time(&self.last_smart_money_signal, &event.market_id).await;

        let signal = TradeSignal {
            id: format!(
                "sig_smartmoney_{}_{}_{}",
                event.market_id,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: event.market_id.clone(),
            token_id: token_id.clone(),
            outcome,
            side,
            price: Some(entry_price),
            size,
            size_usd: order_size_usd,
            order_type: OrderType::Fok,
            priority: config.signal_priority,
            timestamp: Utc::now(),
            reason: format!(
                "Smart money follow: Trader #{} (${:.2} profit) {} {} position ${:.2}",
                event.trader_rank,
                event.trader_profit,
                match event.action {
                    TraderAction::Buy => "buying",
                    TraderAction::NewPosition => "opening",
                    TraderAction::Sell => "selling",
                    TraderAction::ClosePosition => "closing",
                },
                outcome,
                event.size_usd
            ),
            metadata: serde_json::json!({
                "trader_rank": event.trader_rank,
                "trader_profit": event.trader_profit.to_string(),
                "trader_address": event.trader_address,
                "action": format!("{:?}", event.action),
                "original_size_usd": event.size_usd.to_string(),
            }),
        };

        info!(
            signal_id = %signal.id,
            market = %event.market_id,
            trader_rank = event.trader_rank,
            outcome = ?outcome,
            "Generated smart money follow signal"
        );

        Ok(Some(signal))
    }

    /// Process a volume anomaly event
    async fn process_volume_anomaly(
        &self,
        event: &VolumeAnomalyEvent,
        state: &dyn StateProvider,
    ) -> Result<Option<TradeSignal>, StrategyError> {
        let config = self.config.read().await;

        if !config.volume_anomaly.enabled {
            return Ok(None);
        }

        // Check volume ratio
        if event.volume_ratio < config.volume_anomaly.min_volume_ratio {
            return Ok(None);
        }

        // Check cooldown
        if !Self::check_cooldown(
            &self.last_volume_signal,
            &event.market_id,
            config.volume_anomaly.cooldown_secs,
        )
        .await
        {
            debug!(market = %event.market_id, "Cooldown active for volume signal");
            return Ok(None);
        }

        // Get market info
        let market = match state.get_market(&event.market_id).await {
            Some(m) => m,
            None => {
                warn!(market = %event.market_id, "Market not found");
                return Ok(None);
            }
        };

        // Determine direction based on net flow if available
        let outcome = if let Some(net_flow) = event.net_flow {
            if net_flow > Decimal::ZERO {
                Outcome::Yes
            } else {
                Outcome::No
            }
        } else {
            // Check smart money consensus
            let positions = self.smart_money_positions.read().await;
            if let Some(consensus) = positions.get(&event.market_id) {
                match consensus.get_consensus() {
                    Some(o) => o,
                    None => {
                        debug!(market = %event.market_id, "No clear direction for volume anomaly");
                        return Ok(None);
                    }
                }
            } else {
                debug!(market = %event.market_id, "No consensus data for volume anomaly");
                return Ok(None);
            }
        };

        let token_id = match outcome {
            Outcome::Yes => &market.yes_token_id,
            Outcome::No => &market.no_token_id,
        };

        // Check current price
        let current_price = state.get_price(token_id).await;
        if let Some(price) = current_price {
            if price > config.max_entry_price {
                debug!(
                    price = %price,
                    max = %config.max_entry_price,
                    "Price too high for entry"
                );
                return Ok(None);
            }
        }

        let entry_price = current_price.unwrap_or(config.max_entry_price);
        let order_size_usd = config.volume_anomaly.order_size_usd;

        let size = if entry_price.is_zero() {
            Decimal::ZERO
        } else {
            order_size_usd / entry_price
        };

        // Record signal time
        Self::record_signal_time(&self.last_volume_signal, &event.market_id).await;

        let signal = TradeSignal {
            id: format!(
                "sig_volume_{}_{}_{}",
                event.market_id,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: event.market_id.clone(),
            token_id: token_id.clone(),
            outcome,
            side: Side::Buy,
            price: Some(entry_price),
            size,
            size_usd: order_size_usd,
            order_type: OrderType::Fok,
            priority: config.signal_priority,
            timestamp: Utc::now(),
            reason: format!(
                "Volume anomaly: {:.1}x normal (${:.2} vs ${:.2} avg), {} trades",
                event.volume_ratio, event.current_volume, event.avg_volume, event.trade_count
            ),
            metadata: serde_json::json!({
                "volume_ratio": event.volume_ratio.to_string(),
                "current_volume": event.current_volume.to_string(),
                "avg_volume": event.avg_volume.to_string(),
                "trade_count": event.trade_count,
            }),
        };

        info!(
            signal_id = %signal.id,
            market = %event.market_id,
            volume_ratio = %event.volume_ratio,
            outcome = ?outcome,
            "Generated volume anomaly signal"
        );

        Ok(Some(signal))
    }

    /// Process a comment activity spike
    async fn process_comment_activity(
        &self,
        event: &CommentActivityEvent,
        state: &dyn StateProvider,
    ) -> Result<Option<TradeSignal>, StrategyError> {
        let config = self.config.read().await;

        if !config.comment_activity.enabled {
            return Ok(None);
        }

        // Check velocity
        if event.comment_velocity < Decimal::from(config.comment_activity.min_comments_per_hour) {
            return Ok(None);
        }

        // Check cooldown
        if !Self::check_cooldown(
            &self.last_comment_signal,
            &event.market_id,
            config.comment_activity.cooldown_secs,
        )
        .await
        {
            debug!(market = %event.market_id, "Cooldown active for comment signal");
            return Ok(None);
        }

        // Get market info
        let market = match state.get_market(&event.market_id).await {
            Some(m) => m,
            None => {
                warn!(market = %event.market_id, "Market not found");
                return Ok(None);
            }
        };

        // Determine direction from sentiment hint or smart money consensus
        let outcome = if let Some(ref hint) = event.sentiment_hint {
            let hint_lower = hint.to_lowercase();
            if hint_lower.contains("bullish") || hint_lower.contains("positive") {
                Outcome::Yes
            } else if hint_lower.contains("bearish") || hint_lower.contains("negative") {
                Outcome::No
            } else {
                // Fall back to smart money consensus
                let positions = self.smart_money_positions.read().await;
                if let Some(consensus) = positions.get(&event.market_id) {
                    match consensus.get_consensus() {
                        Some(o) => o,
                        None => return Ok(None),
                    }
                } else {
                    return Ok(None);
                }
            }
        } else {
            return Ok(None);
        };

        let token_id = match outcome {
            Outcome::Yes => &market.yes_token_id,
            Outcome::No => &market.no_token_id,
        };

        let current_price = state.get_price(token_id).await;
        if let Some(price) = current_price {
            if price > config.max_entry_price {
                return Ok(None);
            }
        }

        let entry_price = current_price.unwrap_or(config.max_entry_price);
        let order_size_usd = config.comment_activity.order_size_usd;

        let size = if entry_price.is_zero() {
            Decimal::ZERO
        } else {
            order_size_usd / entry_price
        };

        // Record signal time
        Self::record_signal_time(&self.last_comment_signal, &event.market_id).await;

        let signal = TradeSignal {
            id: format!(
                "sig_comment_{}_{}_{}",
                event.market_id,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: event.market_id.clone(),
            token_id: token_id.clone(),
            outcome,
            side: Side::Buy,
            price: Some(entry_price),
            size,
            size_usd: order_size_usd,
            order_type: OrderType::Fok,
            priority: config.signal_priority,
            timestamp: Utc::now(),
            reason: format!(
                "Comment activity spike: {} comments ({:.1}/hr)",
                event.comment_count, event.comment_velocity
            ),
            metadata: serde_json::json!({
                "comment_count": event.comment_count,
                "comment_velocity": event.comment_velocity.to_string(),
                "sentiment_hint": event.sentiment_hint,
            }),
        };

        info!(
            signal_id = %signal.id,
            market = %event.market_id,
            velocity = %event.comment_velocity,
            outcome = ?outcome,
            "Generated comment activity signal"
        );

        Ok(Some(signal))
    }
}

#[async_trait]
impl Strategy for PolymarketActivityStrategy {
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

        match event {
            SystemEvent::SmartMoneySignal(e) => {
                if let Some(signal) = self.process_smart_money_signal(e, state).await? {
                    signals.push(signal);
                }
            }
            SystemEvent::VolumeAnomalyDetected(e) => {
                if let Some(signal) = self.process_volume_anomaly(e, state).await? {
                    signals.push(signal);
                }
            }
            SystemEvent::CommentActivitySpike(e) => {
                if let Some(signal) = self.process_comment_activity(e, state).await? {
                    signals.push(signal);
                }
            }
            _ => {}
        }

        Ok(signals)
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(
            event,
            SystemEvent::SmartMoneySignal(_)
                | SystemEvent::VolumeAnomalyDetected(_)
                | SystemEvent::CommentActivitySpike(_)
        )
    }

    async fn initialize(&mut self, _state: &dyn StateProvider) -> Result<(), StrategyError> {
        let config = self.config.read().await;
        info!(
            strategy_id = %self.id,
            smart_money_enabled = config.smart_money_tracking.enabled,
            volume_anomaly_enabled = config.volume_anomaly.enabled,
            comment_activity_enabled = config.comment_activity.enabled,
            "Initializing Polymarket activity strategy"
        );
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }

    async fn reload_config(&mut self, config_content: &str) -> Result<(), StrategyError> {
        let new_config: PolymarketActivityStrategyConfig = toml::from_str(config_content)
            .map_err(|e| StrategyError::ConfigError(format!("Failed to parse config: {}", e)))?;

        let mut config = self.config.write().await;
        *config = new_config;
        info!(strategy_id = %self.id, "Reloaded Polymarket activity strategy config");
        Ok(())
    }

    fn config_name(&self) -> &str {
        "polymarket_activity"
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
    use polysniper_core::{Market, Orderbook, Position, TokenId};

    struct MockStateProvider {
        markets: HashMap<MarketId, Market>,
        prices: HashMap<TokenId, Decimal>,
    }

    impl MockStateProvider {
        fn new() -> Self {
            Self {
                markets: HashMap::new(),
                prices: HashMap::new(),
            }
        }

        fn with_market(mut self, market: Market) -> Self {
            self.markets.insert(market.condition_id.clone(), market);
            self
        }

        fn with_price(mut self, token_id: TokenId, price: Decimal) -> Self {
            self.prices.insert(token_id, price);
            self
        }
    }

    #[async_trait]
    impl StateProvider for MockStateProvider {
        async fn get_market(&self, market_id: &MarketId) -> Option<Market> {
            self.markets.get(market_id).cloned()
        }

        async fn get_all_markets(&self) -> Vec<Market> {
            self.markets.values().cloned().collect()
        }

        async fn get_orderbook(&self, _token_id: &TokenId) -> Option<Orderbook> {
            None
        }

        async fn get_price(&self, token_id: &TokenId) -> Option<Decimal> {
            self.prices.get(token_id).copied()
        }

        async fn get_position(&self, _market_id: &MarketId) -> Option<Position> {
            None
        }

        async fn get_all_positions(&self) -> Vec<Position> {
            Vec::new()
        }

        async fn get_price_history(
            &self,
            _token_id: &TokenId,
            _limit: usize,
        ) -> Vec<(chrono::DateTime<chrono::Utc>, Decimal)> {
            Vec::new()
        }

        async fn get_portfolio_value(&self) -> Decimal {
            Decimal::ZERO
        }

        async fn get_daily_pnl(&self) -> Decimal {
            Decimal::ZERO
        }
    }

    fn test_market() -> Market {
        Market {
            condition_id: "test-market".to_string(),
            question: "Test market?".to_string(),
            description: None,
            tags: vec![],
            yes_token_id: "yes-token".to_string(),
            no_token_id: "no-token".to_string(),
            created_at: Utc::now(),
            end_date: None,
            active: true,
            closed: false,
            volume: dec!(100000),
            liquidity: dec!(50000),
        }
    }

    fn test_config() -> PolymarketActivityStrategyConfig {
        let mut config = PolymarketActivityStrategyConfig::default();
        config.smart_money_tracking.cooldown_secs = 0;
        config.smart_money_tracking.min_position_size_usd = dec!(100);
        config.volume_anomaly.cooldown_secs = 0;
        config
    }

    #[tokio::test]
    async fn test_smart_money_signal() {
        let config = test_config();
        let strategy = PolymarketActivityStrategy::new(config);
        let state = MockStateProvider::new()
            .with_market(test_market())
            .with_price("yes-token".to_string(), dec!(0.50));

        let event = SystemEvent::SmartMoneySignal(SmartMoneySignalEvent {
            market_id: "test-market".to_string(),
            token_id: "yes-token".to_string(),
            trader_address: "0x123".to_string(),
            trader_username: Some("top_trader".to_string()),
            trader_rank: 5,
            trader_profit: dec!(50000),
            action: TraderAction::Buy,
            outcome: Outcome::Yes,
            size_usd: dec!(5000),
            timestamp: Utc::now(),
        });

        let signals = strategy.process_event(&event, &state).await.unwrap();
        assert!(!signals.is_empty());

        let signal = &signals[0];
        assert_eq!(signal.market_id, "test-market");
        assert_eq!(signal.outcome, Outcome::Yes);
        assert_eq!(signal.side, Side::Buy);
    }

    #[tokio::test]
    async fn test_smart_money_rank_filter() {
        let mut config = test_config();
        config.smart_money_tracking.min_trader_rank = 10;
        let strategy = PolymarketActivityStrategy::new(config);
        let state = MockStateProvider::new()
            .with_market(test_market())
            .with_price("yes-token".to_string(), dec!(0.50));

        // Trader with rank 50 should be filtered out
        let event = SystemEvent::SmartMoneySignal(SmartMoneySignalEvent {
            market_id: "test-market".to_string(),
            token_id: "yes-token".to_string(),
            trader_address: "0x123".to_string(),
            trader_username: None,
            trader_rank: 50,
            trader_profit: dec!(5000),
            action: TraderAction::Buy,
            outcome: Outcome::Yes,
            size_usd: dec!(1000),
            timestamp: Utc::now(),
        });

        let signals = strategy.process_event(&event, &state).await.unwrap();
        assert!(signals.is_empty());
    }

    #[tokio::test]
    async fn test_volume_anomaly_signal() {
        let mut config = test_config();
        config.volume_anomaly.min_volume_ratio = dec!(2.0);
        let strategy = PolymarketActivityStrategy::new(config);
        let state = MockStateProvider::new()
            .with_market(test_market())
            .with_price("yes-token".to_string(), dec!(0.50));

        // First add some smart money consensus
        {
            let mut positions = strategy.smart_money_positions.write().await;
            let consensus = positions
                .entry("test-market".to_string())
                .or_insert_with(SmartMoneyConsensus::default);
            consensus.add_signal(Outcome::Yes, dec!(10000), 1);
        }

        let event = SystemEvent::VolumeAnomalyDetected(VolumeAnomalyEvent {
            market_id: "test-market".to_string(),
            current_volume: dec!(10000),
            avg_volume: dec!(2000),
            volume_ratio: dec!(5.0),
            trade_count: 100,
            net_flow: Some(dec!(5000)),
            timestamp: Utc::now(),
        });

        let signals = strategy.process_event(&event, &state).await.unwrap();
        assert!(!signals.is_empty());
    }

    #[tokio::test]
    async fn test_accepts_correct_events() {
        let config = test_config();
        let strategy = PolymarketActivityStrategy::new(config);

        assert!(strategy.accepts_event(&SystemEvent::SmartMoneySignal(SmartMoneySignalEvent {
            market_id: "test".to_string(),
            token_id: "token".to_string(),
            trader_address: "0x".to_string(),
            trader_username: None,
            trader_rank: 1,
            trader_profit: dec!(0),
            action: TraderAction::Buy,
            outcome: Outcome::Yes,
            size_usd: dec!(0),
            timestamp: Utc::now(),
        })));

        assert!(strategy.accepts_event(&SystemEvent::VolumeAnomalyDetected(VolumeAnomalyEvent {
            market_id: "test".to_string(),
            current_volume: dec!(0),
            avg_volume: dec!(0),
            volume_ratio: dec!(0),
            trade_count: 0,
            net_flow: None,
            timestamp: Utc::now(),
        })));

        assert!(!strategy.accepts_event(&SystemEvent::Heartbeat(polysniper_core::HeartbeatEvent {
            source: "test".to_string(),
            timestamp: Utc::now(),
        })));
    }

    #[test]
    fn test_smart_money_consensus() {
        let mut consensus = SmartMoneyConsensus::default();

        // Add yes signals
        consensus.add_signal(Outcome::Yes, dec!(1000), 1);
        consensus.add_signal(Outcome::Yes, dec!(500), 5);

        // Add no signal
        consensus.add_signal(Outcome::No, dec!(200), 10);

        assert_eq!(consensus.get_consensus(), Some(Outcome::Yes));
        assert!(consensus.confidence() > Decimal::ZERO);
    }
}
