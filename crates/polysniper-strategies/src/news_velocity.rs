//! News Velocity Strategy
//!
//! Trading strategy that reacts to news velocity signals, generating trade signals
//! based on acceleration (breaking news) or deceleration (fading story) patterns.

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use polysniper_core::{
    MarketId, NewsVelocitySignalEvent, OrderType, Outcome, Priority, Side, StateProvider, Strategy,
    StrategyError, SystemEvent, TradeSignal, VelocityDirection,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Configuration for news velocity strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsVelocityStrategyConfig {
    /// Whether the strategy is enabled
    pub enabled: bool,
    /// Configuration for acceleration (breaking news) signals
    #[serde(default)]
    pub acceleration: AccelerationConfig,
    /// Configuration for deceleration (fading) signals
    #[serde(default)]
    pub deceleration: DecelerationConfig,
    /// Keyword to market mapping
    #[serde(default)]
    pub market_mappings: HashMap<String, VelocityMarketMapping>,
    /// Priority for generated signals
    #[serde(default)]
    pub signal_priority: Priority,
    /// Global cooldown between signals for any keyword (seconds)
    #[serde(default = "default_global_cooldown")]
    pub global_cooldown_secs: u64,
}

fn default_global_cooldown() -> u64 {
    300 // 5 minutes
}

impl Default for NewsVelocityStrategyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            acceleration: AccelerationConfig::default(),
            deceleration: DecelerationConfig::default(),
            market_mappings: HashMap::new(),
            signal_priority: Priority::High,
            global_cooldown_secs: default_global_cooldown(),
        }
    }
}

/// Configuration for acceleration signals (breaking news)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerationConfig {
    /// Whether acceleration signals are enabled
    pub enabled: bool,
    /// Minimum acceleration factor to trigger signal
    #[serde(default = "default_min_acceleration")]
    pub min_acceleration: Decimal,
    /// Minimum articles in last hour to trigger signal
    #[serde(default = "default_min_articles_1h")]
    pub min_articles_1h: u32,
    /// Trade direction for acceleration signals (usually Buy for momentum)
    #[serde(default = "default_buy")]
    pub signal_direction: Side,
    /// Default order size in USD
    #[serde(default = "default_order_size")]
    pub order_size_usd: Decimal,
    /// Maximum entry price (don't chase too high)
    #[serde(default = "default_max_entry")]
    pub max_entry_price: Decimal,
    /// Cooldown between acceleration signals per keyword (seconds)
    #[serde(default = "default_cooldown")]
    pub cooldown_secs: u64,
}

fn default_min_acceleration() -> Decimal {
    dec!(2.0)
}

fn default_min_articles_1h() -> u32 {
    5
}

fn default_order_size() -> Decimal {
    dec!(50)
}

fn default_max_entry() -> Decimal {
    dec!(0.85)
}

fn default_buy() -> Side {
    Side::Buy
}

fn default_cooldown() -> u64 {
    1800 // 30 minutes
}

impl Default for AccelerationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_acceleration: default_min_acceleration(),
            min_articles_1h: default_min_articles_1h(),
            signal_direction: Side::Buy,
            order_size_usd: default_order_size(),
            max_entry_price: default_max_entry(),
            cooldown_secs: default_cooldown(),
        }
    }
}

/// Configuration for deceleration signals (fading story)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecelerationConfig {
    /// Whether deceleration signals are enabled
    pub enabled: bool,
    /// Maximum velocity ratio to trigger signal (e.g., 0.3 = 30% of baseline)
    #[serde(default = "default_max_velocity_ratio")]
    pub max_velocity_ratio: Decimal,
    /// Trade direction for deceleration signals (usually Sell for fading)
    #[serde(default = "default_sell")]
    pub signal_direction: Side,
    /// Order size in USD (0 = close position only)
    #[serde(default)]
    pub order_size_usd: Decimal,
    /// Minimum position profit percentage to exit
    #[serde(default = "default_min_profit")]
    pub min_position_profit_pct: Decimal,
    /// Cooldown between deceleration signals per keyword (seconds)
    #[serde(default = "default_decel_cooldown")]
    pub cooldown_secs: u64,
}

fn default_max_velocity_ratio() -> Decimal {
    dec!(0.3)
}

fn default_sell() -> Side {
    Side::Sell
}

fn default_min_profit() -> Decimal {
    dec!(5.0)
}

fn default_decel_cooldown() -> u64 {
    3600 // 1 hour
}

impl Default for DecelerationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_velocity_ratio: default_max_velocity_ratio(),
            signal_direction: Side::Sell,
            order_size_usd: Decimal::ZERO,
            min_position_profit_pct: default_min_profit(),
            cooldown_secs: default_decel_cooldown(),
        }
    }
}

/// Market mapping for velocity-triggered trades
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityMarketMapping {
    /// Market condition ID
    pub market_id: MarketId,
    /// YES token ID
    pub yes_token_id: String,
    /// NO token ID
    pub no_token_id: String,
    /// Override order size for this market
    pub order_size_usd: Option<Decimal>,
    /// Override max entry price for this market
    pub max_entry_price: Option<Decimal>,
    /// Whether acceleration signals are enabled for this market
    #[serde(default = "default_true")]
    pub acceleration_enabled: bool,
    /// Whether deceleration signals are enabled for this market
    #[serde(default = "default_true")]
    pub deceleration_enabled: bool,
}

fn default_true() -> bool {
    true
}

/// News velocity trading strategy
pub struct NewsVelocityStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: Arc<RwLock<NewsVelocityStrategyConfig>>,
    /// Last signal time per keyword and direction
    signal_cooldowns: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    /// Last global signal time
    last_global_signal: Arc<RwLock<Option<DateTime<Utc>>>>,
}

impl NewsVelocityStrategy {
    /// Create a new news velocity strategy
    pub fn new(config: NewsVelocityStrategyConfig) -> Self {
        let enabled = config.enabled;
        Self {
            id: "news_velocity".to_string(),
            name: "News Velocity Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config: Arc::new(RwLock::new(config)),
            signal_cooldowns: Arc::new(RwLock::new(HashMap::new())),
            last_global_signal: Arc::new(RwLock::new(None)),
        }
    }

    /// Generate a cooldown key for tracking
    fn cooldown_key(keyword: &str, direction: VelocityDirection) -> String {
        format!("{}:{}", keyword, direction.as_str())
    }

    /// Check if cooldown has passed
    async fn check_cooldown(&self, keyword: &str, direction: VelocityDirection) -> bool {
        let config = self.config.read().await;
        let cooldown_secs = match direction {
            VelocityDirection::Accelerating => config.acceleration.cooldown_secs,
            VelocityDirection::Decelerating => config.deceleration.cooldown_secs,
            VelocityDirection::Stable => return true,
        };

        let key = Self::cooldown_key(keyword, direction);
        let cooldowns = self.signal_cooldowns.read().await;

        if let Some(last_time) = cooldowns.get(&key) {
            let cooldown = Duration::seconds(cooldown_secs as i64);
            if Utc::now() - *last_time < cooldown {
                return false;
            }
        }

        // Also check global cooldown
        let last_global = self.last_global_signal.read().await;
        if let Some(last_time) = *last_global {
            let global_cooldown = Duration::seconds(config.global_cooldown_secs as i64);
            if Utc::now() - last_time < global_cooldown {
                return false;
            }
        }

        true
    }

    /// Record signal time for cooldown tracking
    async fn record_signal_time(&self, keyword: &str, direction: VelocityDirection) {
        let key = Self::cooldown_key(keyword, direction);
        let now = Utc::now();

        let mut cooldowns = self.signal_cooldowns.write().await;
        cooldowns.insert(key, now);

        let mut last_global = self.last_global_signal.write().await;
        *last_global = Some(now);
    }

    /// Process a news velocity signal event
    async fn process_velocity_signal(
        &self,
        signal: &NewsVelocitySignalEvent,
        state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        let config = self.config.read().await;
        let mut trade_signals = Vec::new();

        // Check direction-specific configuration
        match signal.direction {
            VelocityDirection::Accelerating => {
                if !config.acceleration.enabled {
                    return Ok(trade_signals);
                }
                if signal.acceleration < config.acceleration.min_acceleration {
                    debug!(
                        keyword = %signal.keyword,
                        acceleration = %signal.acceleration,
                        threshold = %config.acceleration.min_acceleration,
                        "Acceleration below threshold"
                    );
                    return Ok(trade_signals);
                }
                if signal.article_count_1h < config.acceleration.min_articles_1h {
                    debug!(
                        keyword = %signal.keyword,
                        count = %signal.article_count_1h,
                        min = %config.acceleration.min_articles_1h,
                        "Article count below threshold"
                    );
                    return Ok(trade_signals);
                }
            }
            VelocityDirection::Decelerating => {
                if !config.deceleration.enabled {
                    return Ok(trade_signals);
                }
                if signal.acceleration > config.deceleration.max_velocity_ratio {
                    debug!(
                        keyword = %signal.keyword,
                        acceleration = %signal.acceleration,
                        threshold = %config.deceleration.max_velocity_ratio,
                        "Velocity ratio above threshold"
                    );
                    return Ok(trade_signals);
                }
            }
            VelocityDirection::Stable => {
                return Ok(trade_signals);
            }
        }

        // Check cooldown
        if !self.check_cooldown(&signal.keyword, signal.direction).await {
            debug!(
                keyword = %signal.keyword,
                direction = %signal.direction,
                "Signal cooldown active"
            );
            return Ok(trade_signals);
        }

        // Try to find market mapping
        let mapping = config.market_mappings.get(&signal.keyword);

        // If no mapping, try signal's market_ids
        let market_ids = if let Some(m) = mapping {
            // Check if direction is enabled for this market
            match signal.direction {
                VelocityDirection::Accelerating if !m.acceleration_enabled => {
                    return Ok(trade_signals);
                }
                VelocityDirection::Decelerating if !m.deceleration_enabled => {
                    return Ok(trade_signals);
                }
                _ => {}
            }
            vec![m.market_id.clone()]
        } else if !signal.market_ids.is_empty() {
            signal.market_ids.clone()
        } else {
            // Try to find markets by keyword in state
            let markets = state.get_all_markets().await;
            let keyword_lower = signal.keyword.to_lowercase();
            markets
                .iter()
                .filter(|m| m.question.to_lowercase().contains(&keyword_lower))
                .map(|m| m.condition_id.clone())
                .collect::<Vec<_>>()
        };

        if market_ids.is_empty() {
            warn!(
                keyword = %signal.keyword,
                "No market mapping found for keyword"
            );
            return Ok(trade_signals);
        }

        // Generate trade signals for each market
        for market_id in market_ids {
            if let Some(trade_signal) =
                self.create_trade_signal(&signal, &market_id, mapping, state, &config)
                    .await?
            {
                trade_signals.push(trade_signal);
            }
        }

        // Record signal time if we generated signals
        if !trade_signals.is_empty() {
            self.record_signal_time(&signal.keyword, signal.direction)
                .await;
        }

        Ok(trade_signals)
    }

    /// Create a trade signal for a specific market
    async fn create_trade_signal(
        &self,
        velocity_signal: &NewsVelocitySignalEvent,
        market_id: &MarketId,
        mapping: Option<&VelocityMarketMapping>,
        state: &dyn StateProvider,
        config: &NewsVelocityStrategyConfig,
    ) -> Result<Option<TradeSignal>, StrategyError> {
        // Get market details
        let market = match state.get_market(market_id).await {
            Some(m) => m,
            None => {
                warn!(
                    market_id = %market_id,
                    "Market not found"
                );
                return Ok(None);
            }
        };

        // Determine trade parameters based on direction
        let (side, outcome, token_id) = match velocity_signal.direction {
            VelocityDirection::Accelerating => {
                // Buy YES on acceleration (momentum following)
                (
                    config.acceleration.signal_direction,
                    Outcome::Yes,
                    mapping
                        .map(|m| m.yes_token_id.clone())
                        .unwrap_or_else(|| market.yes_token_id.clone()),
                )
            }
            VelocityDirection::Decelerating => {
                // Sell on deceleration (fading story)
                (
                    config.deceleration.signal_direction,
                    Outcome::Yes,
                    mapping
                        .map(|m| m.yes_token_id.clone())
                        .unwrap_or_else(|| market.yes_token_id.clone()),
                )
            }
            VelocityDirection::Stable => return Ok(None),
        };

        // Get current price
        let current_price = state.get_price(&token_id).await;

        // Determine order size
        let order_size_usd = mapping
            .and_then(|m| m.order_size_usd)
            .unwrap_or_else(|| match velocity_signal.direction {
                VelocityDirection::Accelerating => config.acceleration.order_size_usd,
                VelocityDirection::Decelerating => config.deceleration.order_size_usd,
                VelocityDirection::Stable => Decimal::ZERO,
            });

        // For deceleration with 0 size, we need an existing position
        if velocity_signal.direction == VelocityDirection::Decelerating
            && order_size_usd.is_zero()
        {
            let position = state.get_position(market_id).await;
            if position.is_none() || position.as_ref().map(|p| p.size.is_zero()).unwrap_or(true) {
                debug!(
                    market_id = %market_id,
                    "No position to exit on deceleration"
                );
                return Ok(None);
            }
        }

        // Check max entry price for buys
        let max_entry = mapping
            .and_then(|m| m.max_entry_price)
            .unwrap_or(config.acceleration.max_entry_price);

        if side == Side::Buy {
            if let Some(price) = current_price {
                if price > max_entry {
                    warn!(
                        market_id = %market_id,
                        price = %price,
                        max = %max_entry,
                        "Price too high for velocity entry"
                    );
                    return Ok(None);
                }
            }
        }

        let entry_price = current_price.unwrap_or(max_entry);
        let size = if entry_price.is_zero() {
            Decimal::ZERO
        } else {
            order_size_usd / entry_price
        };

        // Build reason string
        let headlines_str = if velocity_signal.sample_headlines.is_empty() {
            String::new()
        } else {
            format!(
                " Headlines: {}",
                velocity_signal.sample_headlines.join("; ")
            )
        };

        let signal = TradeSignal {
            id: format!(
                "sig_velocity_{}_{}_{}",
                market_id,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: market_id.clone(),
            token_id,
            outcome,
            side,
            price: Some(entry_price),
            size,
            size_usd: order_size_usd,
            order_type: OrderType::Fok,
            priority: config.signal_priority,
            timestamp: Utc::now(),
            reason: format!(
                "News velocity {}: {} ({}x baseline, {} articles/hr).{}",
                velocity_signal.direction,
                velocity_signal.keyword,
                velocity_signal.acceleration,
                velocity_signal.current_velocity,
                headlines_str
            ),
            metadata: serde_json::json!({
                "keyword": velocity_signal.keyword,
                "direction": velocity_signal.direction.as_str(),
                "acceleration": velocity_signal.acceleration.to_string(),
                "current_velocity": velocity_signal.current_velocity.to_string(),
                "baseline_velocity": velocity_signal.baseline_velocity.to_string(),
                "article_count_1h": velocity_signal.article_count_1h,
                "article_count_24h": velocity_signal.article_count_24h,
            }),
        };

        info!(
            signal_id = %signal.id,
            market_id = %market_id,
            keyword = %velocity_signal.keyword,
            direction = %velocity_signal.direction,
            outcome = ?outcome,
            acceleration = %velocity_signal.acceleration,
            "Generated news velocity trade signal"
        );

        Ok(Some(signal))
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

#[async_trait]
impl Strategy for NewsVelocityStrategy {
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
        match event {
            SystemEvent::NewsVelocitySignal(signal) => {
                self.process_velocity_signal(signal, state).await
            }
            _ => Ok(Vec::new()),
        }
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(event, SystemEvent::NewsVelocitySignal(_))
    }

    async fn initialize(&mut self, _state: &dyn StateProvider) -> Result<(), StrategyError> {
        let config = self.config.read().await;
        info!(
            strategy_id = %self.id,
            market_mappings = %config.market_mappings.len(),
            acceleration_enabled = %config.acceleration.enabled,
            deceleration_enabled = %config.deceleration.enabled,
            "Initializing news velocity strategy"
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
        let new_config: NewsVelocityStrategyConfig = toml::from_str(config_content)
            .map_err(|e| StrategyError::ConfigError(format!("Failed to parse config: {}", e)))?;

        self.enabled
            .store(new_config.enabled, Ordering::SeqCst);

        let mut config = self.config.write().await;
        *config = new_config;

        info!(strategy_id = %self.id, "Reloaded news velocity strategy config");
        Ok(())
    }

    fn config_name(&self) -> &str {
        "news_velocity"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polysniper_core::{Market, Orderbook, Position, TokenId};
    use rust_decimal_macros::dec;
    use std::collections::HashMap;

    struct MockStateProvider {
        markets: HashMap<MarketId, Market>,
        prices: HashMap<TokenId, Decimal>,
        positions: HashMap<MarketId, Position>,
    }

    impl MockStateProvider {
        fn new() -> Self {
            Self {
                markets: HashMap::new(),
                prices: HashMap::new(),
                positions: HashMap::new(),
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

        fn with_position(mut self, position: Position) -> Self {
            self.positions
                .insert(position.market_id.clone(), position);
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

        async fn get_position(&self, market_id: &MarketId) -> Option<Position> {
            self.positions.get(market_id).cloned()
        }

        async fn get_all_positions(&self) -> Vec<Position> {
            self.positions.values().cloned().collect()
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

    fn test_market() -> Market {
        Market {
            condition_id: "trump-market".to_string(),
            question: "Will Trump win?".to_string(),
            description: Some("Test market".to_string()),
            tags: vec!["politics".to_string()],
            yes_token_id: "trump-yes".to_string(),
            no_token_id: "trump-no".to_string(),
            created_at: Utc::now(),
            end_date: None,
            active: true,
            closed: false,
            volume: dec!(1000000),
            liquidity: dec!(50000),
        }
    }

    fn test_config() -> NewsVelocityStrategyConfig {
        let mut config = NewsVelocityStrategyConfig::default();
        config.enabled = true;
        config.global_cooldown_secs = 1;
        config.acceleration.cooldown_secs = 1;
        config.deceleration.cooldown_secs = 1;
        config.market_mappings.insert(
            "trump".to_string(),
            VelocityMarketMapping {
                market_id: "trump-market".to_string(),
                yes_token_id: "trump-yes".to_string(),
                no_token_id: "trump-no".to_string(),
                order_size_usd: Some(dec!(100)),
                max_entry_price: Some(dec!(0.90)),
                acceleration_enabled: true,
                deceleration_enabled: true,
            },
        );
        config
    }

    #[tokio::test]
    async fn test_acceleration_signal() {
        let config = test_config();
        let strategy = NewsVelocityStrategy::new(config);
        let state = MockStateProvider::new()
            .with_market(test_market())
            .with_price("trump-yes".to_string(), dec!(0.50));

        let signal = NewsVelocitySignalEvent::new(
            "trump".to_string(),
            vec!["trump-market".to_string()],
            VelocityDirection::Accelerating,
            dec!(5.0),  // current velocity
            dec!(1.0),  // baseline
            dec!(5.0),  // acceleration
            10,         // 1h count
            50,         // 24h count
            vec!["Trump breaking news!".to_string()],
        );

        let event = SystemEvent::NewsVelocitySignal(signal);
        let signals = strategy.process_event(&event, &state).await.unwrap();

        assert!(!signals.is_empty());
        let trade = &signals[0];
        assert_eq!(trade.market_id, "trump-market");
        assert_eq!(trade.side, Side::Buy);
        assert_eq!(trade.outcome, Outcome::Yes);
    }

    #[tokio::test]
    async fn test_deceleration_signal_requires_position() {
        let mut config = test_config();
        config.deceleration.order_size_usd = Decimal::ZERO; // Close only
        // Remove order_size from mapping so the deceleration config is used
        if let Some(mapping) = config.market_mappings.get_mut("trump") {
            mapping.order_size_usd = None;
        }

        let strategy = NewsVelocityStrategy::new(config);
        let state = MockStateProvider::new()
            .with_market(test_market())
            .with_price("trump-yes".to_string(), dec!(0.60));

        let signal = NewsVelocitySignalEvent::new(
            "trump".to_string(),
            vec!["trump-market".to_string()],
            VelocityDirection::Decelerating,
            dec!(0.2),  // current velocity
            dec!(1.0),  // baseline
            dec!(0.2),  // acceleration (20% of baseline)
            2,          // 1h count
            20,         // 24h count
            vec![],
        );

        let event = SystemEvent::NewsVelocitySignal(signal);
        let signals = strategy.process_event(&event, &state).await.unwrap();

        // Should not generate signal without position
        assert!(signals.is_empty());
    }

    #[tokio::test]
    async fn test_price_too_high() {
        let config = test_config();
        let strategy = NewsVelocityStrategy::new(config);
        let state = MockStateProvider::new()
            .with_market(test_market())
            .with_price("trump-yes".to_string(), dec!(0.95)); // Above max entry

        let signal = NewsVelocitySignalEvent::new(
            "trump".to_string(),
            vec!["trump-market".to_string()],
            VelocityDirection::Accelerating,
            dec!(5.0),
            dec!(1.0),
            dec!(5.0),
            10,
            50,
            vec![],
        );

        let event = SystemEvent::NewsVelocitySignal(signal);
        let signals = strategy.process_event(&event, &state).await.unwrap();

        assert!(signals.is_empty());
    }

    #[tokio::test]
    async fn test_cooldown() {
        let mut config = test_config();
        config.acceleration.cooldown_secs = 3600; // Long cooldown
        config.global_cooldown_secs = 3600;

        let strategy = NewsVelocityStrategy::new(config);

        // Record a signal
        strategy
            .record_signal_time("trump", VelocityDirection::Accelerating)
            .await;

        // Should be on cooldown
        assert!(
            !strategy
                .check_cooldown("trump", VelocityDirection::Accelerating)
                .await
        );

        // Different keyword should also be on global cooldown
        assert!(
            !strategy
                .check_cooldown("bitcoin", VelocityDirection::Accelerating)
                .await
        );
    }

    #[tokio::test]
    async fn test_stable_direction_ignored() {
        let config = test_config();
        let strategy = NewsVelocityStrategy::new(config);
        let state = MockStateProvider::new().with_market(test_market());

        let signal = NewsVelocitySignalEvent::new(
            "trump".to_string(),
            vec!["trump-market".to_string()],
            VelocityDirection::Stable,
            dec!(1.0),
            dec!(1.0),
            dec!(1.0),
            5,
            50,
            vec![],
        );

        let event = SystemEvent::NewsVelocitySignal(signal);
        let signals = strategy.process_event(&event, &state).await.unwrap();

        assert!(signals.is_empty());
    }

    #[tokio::test]
    async fn test_accepts_only_velocity_signals() {
        let config = test_config();
        let strategy = NewsVelocityStrategy::new(config);

        assert!(
            strategy.accepts_event(&SystemEvent::NewsVelocitySignal(NewsVelocitySignalEvent::new(
                "trump".to_string(),
                vec![],
                VelocityDirection::Accelerating,
                Decimal::ZERO,
                Decimal::ZERO,
                Decimal::ZERO,
                0,
                0,
                vec![],
            )))
        );

        assert!(
            !strategy.accepts_event(&SystemEvent::Heartbeat(polysniper_core::HeartbeatEvent {
                source: "test".to_string(),
                timestamp: Utc::now(),
            }))
        );
    }

    #[tokio::test]
    async fn test_disabled_strategy() {
        let mut config = test_config();
        config.enabled = false;

        let strategy = NewsVelocityStrategy::new(config);
        assert!(!strategy.is_enabled());
    }

    #[tokio::test]
    async fn test_below_min_articles() {
        let mut config = test_config();
        config.acceleration.min_articles_1h = 10;

        let strategy = NewsVelocityStrategy::new(config);
        let state = MockStateProvider::new()
            .with_market(test_market())
            .with_price("trump-yes".to_string(), dec!(0.50));

        let signal = NewsVelocitySignalEvent::new(
            "trump".to_string(),
            vec!["trump-market".to_string()],
            VelocityDirection::Accelerating,
            dec!(5.0),
            dec!(1.0),
            dec!(5.0),
            5, // Below threshold
            50,
            vec![],
        );

        let event = SystemEvent::NewsVelocitySignal(signal);
        let signals = strategy.process_event(&event, &state).await.unwrap();

        assert!(signals.is_empty());
    }
}
