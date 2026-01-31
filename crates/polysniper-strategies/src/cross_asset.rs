//! Cross-Asset Strategy
//!
//! Trading strategy that monitors cryptocurrency price movements and generates
//! trade signals for correlated Polymarket markets. For example, ETH price surges
//! may predict ETF approval odds increases.

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use polysniper_core::{
    CryptoPriceUpdateEvent, OrderType, Outcome, Priority, Side, StateProvider, Strategy,
    StrategyError, SystemEvent, TradeSignal,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Correlation type between crypto and prediction market
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CorrelationType {
    /// Crypto up -> Market YES up
    Positive,
    /// Crypto up -> Market YES down
    Negative,
}

impl Default for CorrelationType {
    fn default() -> Self {
        CorrelationType::Positive
    }
}

/// Configuration for a single asset correlation mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetCorrelation {
    /// Crypto symbol to watch (e.g., "ETH")
    pub crypto_symbol: String,
    /// Keywords to match in market questions (e.g., ["eth", "ethereum", "etf"])
    pub market_keywords: Vec<String>,
    /// Type of correlation
    #[serde(default)]
    pub correlation_type: CorrelationType,
    /// Minimum confidence threshold for generating signals
    #[serde(default = "default_min_confidence")]
    pub min_confidence: Decimal,
    /// Order size in USD
    #[serde(default = "default_order_size")]
    pub order_size_usd: Decimal,
}

fn default_min_confidence() -> Decimal {
    dec!(0.7)
}

fn default_order_size() -> Decimal {
    dec!(50)
}

/// Configuration for the cross-asset strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossAssetConfig {
    /// Whether the strategy is enabled
    pub enabled: bool,
    /// Correlation rules
    #[serde(default)]
    pub correlations: Vec<AssetCorrelation>,
    /// Minimum crypto price change percentage to trigger a signal
    #[serde(default = "default_min_price_change")]
    pub min_price_change_pct: Decimal,
    /// Delay in seconds before generating signal after price movement
    #[serde(default = "default_signal_delay")]
    pub signal_delay_secs: u64,
    /// Cooldown in seconds between signals for same correlation
    #[serde(default = "default_cooldown")]
    pub cooldown_secs: u64,
    /// Maximum entry price for buying
    #[serde(default = "default_max_entry_price")]
    pub max_entry_price: Decimal,
    /// Price history window in minutes for velocity calculation
    #[serde(default = "default_history_window")]
    pub price_history_minutes: u64,
    /// Priority for generated signals
    #[serde(default)]
    pub signal_priority: Priority,
}

fn default_min_price_change() -> Decimal {
    dec!(5.0)
}

fn default_signal_delay() -> u64 {
    60
}

fn default_cooldown() -> u64 {
    300
}

fn default_max_entry_price() -> Decimal {
    dec!(0.85)
}

fn default_history_window() -> u64 {
    60
}

impl Default for CrossAssetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            correlations: Vec::new(),
            min_price_change_pct: default_min_price_change(),
            signal_delay_secs: default_signal_delay(),
            cooldown_secs: default_cooldown(),
            max_entry_price: default_max_entry_price(),
            price_history_minutes: default_history_window(),
            signal_priority: Priority::High,
        }
    }
}

/// Pending signal waiting for delay to elapse
struct PendingSignal {
    crypto_symbol: String,
    correlation: AssetCorrelation,
    price_change: Decimal,
    triggered_at: DateTime<Utc>,
}

/// Price history entry
#[derive(Debug, Clone)]
struct PriceEntry {
    #[allow(dead_code)]
    price: Decimal,
    timestamp: DateTime<Utc>,
}

/// Cross-asset trading strategy
pub struct CrossAssetStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: Arc<RwLock<CrossAssetConfig>>,
    /// Price history per crypto symbol
    price_history: Arc<RwLock<HashMap<String, VecDeque<PriceEntry>>>>,
    /// Cooldowns per correlation key (symbol + market_keyword_hash)
    cooldowns: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    /// Pending signals waiting for delay
    pending_signals: Arc<RwLock<Vec<PendingSignal>>>,
    /// Processed signal hashes to prevent duplicates
    processed_signals: Arc<RwLock<lru::LruCache<String, ()>>>,
}

impl CrossAssetStrategy {
    /// Create a new cross-asset strategy
    pub fn new(config: CrossAssetConfig) -> Self {
        let enabled = config.enabled;

        Self {
            id: "cross_asset".to_string(),
            name: "Cross-Asset Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config: Arc::new(RwLock::new(config)),
            price_history: Arc::new(RwLock::new(HashMap::new())),
            cooldowns: Arc::new(RwLock::new(HashMap::new())),
            pending_signals: Arc::new(RwLock::new(Vec::new())),
            processed_signals: Arc::new(RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(1000).unwrap(),
            ))),
        }
    }

    /// Generate a cooldown key for a correlation
    fn cooldown_key(symbol: &str, keywords: &[String]) -> String {
        let mut sorted_keywords = keywords.to_vec();
        sorted_keywords.sort();
        format!("{}:{}", symbol, sorted_keywords.join(","))
    }

    /// Check if cooldown is active for a correlation
    async fn is_on_cooldown(&self, key: &str) -> bool {
        let cooldowns = self.cooldowns.read().await;
        if let Some(last_signal) = cooldowns.get(key) {
            let config = self.config.read().await;
            let cooldown = Duration::seconds(config.cooldown_secs as i64);
            if Utc::now() - *last_signal < cooldown {
                return true;
            }
        }
        false
    }

    /// Record signal time for cooldown
    async fn record_signal(&self, key: &str) {
        let mut cooldowns = self.cooldowns.write().await;
        cooldowns.insert(key.to_string(), Utc::now());
    }

    /// Update price history
    async fn update_price_history(&self, event: &CryptoPriceUpdateEvent) {
        let mut history = self.price_history.write().await;
        let config = self.config.read().await;

        let entries = history
            .entry(event.symbol.clone())
            .or_insert_with(VecDeque::new);

        entries.push_back(PriceEntry {
            price: event.price,
            timestamp: event.timestamp,
        });

        // Trim old entries beyond the window
        let window = Duration::minutes(config.price_history_minutes as i64);
        let cutoff = Utc::now() - window;
        while entries.front().map_or(false, |e| e.timestamp < cutoff) {
            entries.pop_front();
        }
    }

    /// Find matching correlations for a crypto symbol
    async fn find_correlations(&self, symbol: &str) -> Vec<AssetCorrelation> {
        let config = self.config.read().await;
        config
            .correlations
            .iter()
            .filter(|c| c.crypto_symbol.eq_ignore_ascii_case(symbol))
            .cloned()
            .collect()
    }

    /// Find markets matching keywords in state
    async fn find_matching_markets(
        &self,
        keywords: &[String],
        state: &dyn StateProvider,
    ) -> Vec<polysniper_core::Market> {
        let markets = state.get_all_markets().await;
        markets
            .into_iter()
            .filter(|m| {
                let question_lower = m.question.to_lowercase();
                keywords
                    .iter()
                    .any(|kw| question_lower.contains(&kw.to_lowercase()))
            })
            .collect()
    }

    /// Process pending signals and generate trade signals
    async fn process_pending_signals(
        &self,
        state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        let mut signals = Vec::new();
        let now = Utc::now();
        let config = self.config.read().await;
        let delay = Duration::seconds(config.signal_delay_secs as i64);
        let max_entry = config.max_entry_price;
        let priority = config.signal_priority;
        drop(config);

        let mut pending = self.pending_signals.write().await;
        let mut to_remove = Vec::new();

        for (idx, pending_signal) in pending.iter().enumerate() {
            // Check if delay has elapsed
            if now - pending_signal.triggered_at < delay {
                continue;
            }

            // Check cooldown
            let cooldown_key = Self::cooldown_key(
                &pending_signal.crypto_symbol,
                &pending_signal.correlation.market_keywords,
            );

            if self.is_on_cooldown(&cooldown_key).await {
                debug!(
                    symbol = %pending_signal.crypto_symbol,
                    "Cooldown active, skipping signal"
                );
                to_remove.push(idx);
                continue;
            }

            // Find matching markets
            let markets = self
                .find_matching_markets(&pending_signal.correlation.market_keywords, state)
                .await;

            if markets.is_empty() {
                debug!(
                    keywords = ?pending_signal.correlation.market_keywords,
                    "No matching markets found"
                );
                to_remove.push(idx);
                continue;
            }

            // Generate signal for each matching market
            for market in &markets {
                // Determine direction based on correlation type and price movement
                let (outcome, token_id) = match pending_signal.correlation.correlation_type {
                    CorrelationType::Positive => {
                        if pending_signal.price_change > Decimal::ZERO {
                            (Outcome::Yes, &market.yes_token_id)
                        } else {
                            (Outcome::No, &market.no_token_id)
                        }
                    }
                    CorrelationType::Negative => {
                        if pending_signal.price_change > Decimal::ZERO {
                            (Outcome::No, &market.no_token_id)
                        } else {
                            (Outcome::Yes, &market.yes_token_id)
                        }
                    }
                };

                // Check current price
                let current_price = state.get_price(token_id).await;
                if let Some(price) = current_price {
                    if price > max_entry {
                        warn!(
                            market_id = %market.condition_id,
                            price = %price,
                            max_entry = %max_entry,
                            "Price too high for entry"
                        );
                        continue;
                    }
                }

                let entry_price = current_price.unwrap_or(max_entry);
                let order_size = pending_signal.correlation.order_size_usd;
                let size = if entry_price.is_zero() {
                    Decimal::ZERO
                } else {
                    order_size / entry_price
                };

                let signal = TradeSignal {
                    id: format!(
                        "sig_cross_asset_{}_{}_{}_{}",
                        pending_signal.crypto_symbol,
                        market.condition_id,
                        Utc::now().timestamp_millis(),
                        rand_suffix()
                    ),
                    strategy_id: self.id.clone(),
                    market_id: market.condition_id.clone(),
                    token_id: token_id.clone(),
                    outcome,
                    side: Side::Buy,
                    price: Some(entry_price),
                    size,
                    size_usd: order_size,
                    order_type: OrderType::Fok,
                    priority,
                    timestamp: Utc::now(),
                    reason: format!(
                        "Cross-asset signal: {} {} {:.2}% -> {} {}",
                        pending_signal.crypto_symbol,
                        if pending_signal.price_change > Decimal::ZERO {
                            "up"
                        } else {
                            "down"
                        },
                        pending_signal.price_change.abs(),
                        outcome,
                        market.question
                    ),
                    metadata: serde_json::json!({
                        "crypto_symbol": pending_signal.crypto_symbol,
                        "price_change_pct": pending_signal.price_change.to_string(),
                        "correlation_type": format!("{:?}", pending_signal.correlation.correlation_type),
                        "keywords": pending_signal.correlation.market_keywords,
                    }),
                };

                info!(
                    signal_id = %signal.id,
                    market_id = %market.condition_id,
                    crypto = %pending_signal.crypto_symbol,
                    change = %pending_signal.price_change,
                    outcome = ?outcome,
                    "Generated cross-asset trade signal"
                );

                signals.push(signal);
            }

            // Record signal for cooldown
            self.record_signal(&cooldown_key).await;
            to_remove.push(idx);
        }

        // Remove processed pending signals (in reverse order to preserve indices)
        to_remove.sort_by(|a, b| b.cmp(a));
        for idx in to_remove {
            pending.remove(idx);
        }

        Ok(signals)
    }

    /// Check if price movement is significant enough
    async fn is_significant_movement(&self, price_change: Decimal) -> bool {
        let config = self.config.read().await;
        price_change.abs() >= config.min_price_change_pct
    }

    /// Generate a signal hash for deduplication
    fn signal_hash(symbol: &str, price_change: Decimal) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        symbol.hash(&mut hasher);
        // Round to avoid floating point issues
        let rounded = (price_change * dec!(10)).trunc();
        rounded.to_string().hash(&mut hasher);
        // Include minute for time-based uniqueness
        let minute = Utc::now().timestamp() / 60;
        minute.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

#[async_trait]
impl Strategy for CrossAssetStrategy {
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

        // Process crypto price updates
        if let SystemEvent::CryptoPriceUpdate(crypto_event) = event {
            // Update price history
            self.update_price_history(crypto_event).await;

            // Check for significant movement
            let price_change = crypto_event.price_change_1h;
            if !self.is_significant_movement(price_change).await {
                // Also check 24h change if 1h is not available/significant
                let price_change_24h = crypto_event.price_change_24h;
                if !self.is_significant_movement(price_change_24h).await {
                    debug!(
                        symbol = %crypto_event.symbol,
                        change_1h = %price_change,
                        change_24h = %price_change_24h,
                        "Price movement below threshold"
                    );
                    // Still process pending signals
                    return self.process_pending_signals(state).await;
                }
            }

            // Use whichever change is more significant
            let effective_change = if price_change.abs() > crypto_event.price_change_24h.abs() {
                price_change
            } else {
                crypto_event.price_change_24h
            };

            // Check for duplicate signals
            let signal_hash = Self::signal_hash(&crypto_event.symbol, effective_change);
            {
                let mut processed = self.processed_signals.write().await;
                if processed.contains(&signal_hash) {
                    debug!(
                        symbol = %crypto_event.symbol,
                        "Duplicate price signal, skipping"
                    );
                    return self.process_pending_signals(state).await;
                }
                processed.put(signal_hash, ());
            }

            // Find matching correlations
            let correlations = self.find_correlations(&crypto_event.symbol).await;
            if correlations.is_empty() {
                debug!(
                    symbol = %crypto_event.symbol,
                    "No correlations configured for symbol"
                );
                return self.process_pending_signals(state).await;
            }

            info!(
                symbol = %crypto_event.symbol,
                price = %crypto_event.price,
                change = %effective_change,
                correlations = correlations.len(),
                "Significant crypto movement detected"
            );

            // Create pending signals for each correlation
            let mut pending = self.pending_signals.write().await;
            for correlation in correlations {
                pending.push(PendingSignal {
                    crypto_symbol: crypto_event.symbol.clone(),
                    correlation,
                    price_change: effective_change,
                    triggered_at: Utc::now(),
                });
            }
        }

        // Always process pending signals
        signals.extend(self.process_pending_signals(state).await?);

        Ok(signals)
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(event, SystemEvent::CryptoPriceUpdate(_))
    }

    async fn initialize(&mut self, _state: &dyn StateProvider) -> Result<(), StrategyError> {
        let config = self.config.read().await;
        info!(
            strategy_id = %self.id,
            correlations = config.correlations.len(),
            min_change = %config.min_price_change_pct,
            delay_secs = config.signal_delay_secs,
            cooldown_secs = config.cooldown_secs,
            "Initializing cross-asset strategy"
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
        let new_config: CrossAssetConfig = toml::from_str(config_content)
            .map_err(|e| StrategyError::ConfigError(format!("Failed to parse config: {}", e)))?;

        let mut config = self.config.write().await;
        *config = new_config;
        info!(strategy_id = %self.id, "Reloaded cross-asset strategy config");
        Ok(())
    }

    fn config_name(&self) -> &str {
        "cross_asset"
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
    use polysniper_core::{Market, MarketId, Orderbook, Position, TokenId};
    use rust_decimal_macros::dec;
    use std::collections::HashMap;

    /// Mock state provider for testing
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

        async fn get_trade_outcomes(&self, _limit: usize) -> Vec<(Decimal, Decimal)> {
            Vec::new()
        }
    }

    fn test_config() -> CrossAssetConfig {
        CrossAssetConfig {
            enabled: true,
            correlations: vec![
                AssetCorrelation {
                    crypto_symbol: "ETH".to_string(),
                    market_keywords: vec!["ethereum".to_string(), "eth".to_string(), "etf".to_string()],
                    correlation_type: CorrelationType::Positive,
                    min_confidence: dec!(0.7),
                    order_size_usd: dec!(50),
                },
                AssetCorrelation {
                    crypto_symbol: "BTC".to_string(),
                    market_keywords: vec!["bitcoin".to_string(), "btc".to_string()],
                    correlation_type: CorrelationType::Positive,
                    min_confidence: dec!(0.7),
                    order_size_usd: dec!(50),
                },
            ],
            min_price_change_pct: dec!(5.0),
            signal_delay_secs: 0, // No delay for tests
            cooldown_secs: 1,     // Short cooldown for tests
            max_entry_price: dec!(0.85),
            price_history_minutes: 60,
            signal_priority: Priority::High,
        }
    }

    fn test_market() -> Market {
        Market {
            condition_id: "eth-etf-market".to_string(),
            question: "Will an Ethereum ETF be approved by end of 2025?".to_string(),
            description: Some("ETF approval market".to_string()),
            tags: vec!["crypto".to_string(), "etf".to_string()],
            yes_token_id: "eth-etf-yes".to_string(),
            no_token_id: "eth-etf-no".to_string(),
            created_at: Utc::now(),
            end_date: None,
            active: true,
            closed: false,
            volume: dec!(1000000),
            liquidity: dec!(50000),
        }
    }

    #[test]
    fn test_config_default() {
        let config = CrossAssetConfig::default();
        assert!(!config.enabled);
        assert!(config.correlations.is_empty());
        assert_eq!(config.min_price_change_pct, dec!(5.0));
        assert_eq!(config.signal_delay_secs, 60);
        assert_eq!(config.cooldown_secs, 300);
    }

    #[test]
    fn test_cooldown_key() {
        let key = CrossAssetStrategy::cooldown_key("ETH", &["ethereum".to_string(), "etf".to_string()]);
        assert!(key.starts_with("ETH:"));
        assert!(key.contains("ethereum"));
        assert!(key.contains("etf"));
    }

    #[tokio::test]
    async fn test_process_significant_movement() {
        let config = test_config();
        let strategy = CrossAssetStrategy::new(config);
        let state = MockStateProvider::new()
            .with_market(test_market())
            .with_price("eth-etf-yes".to_string(), dec!(0.50));

        // Significant positive movement
        let event = SystemEvent::CryptoPriceUpdate(CryptoPriceUpdateEvent::new(
            "ETH".to_string(),
            dec!(2500),
            dec!(7.5), // Above 5% threshold
            dec!(10.0),
            dec!(1000000000),
        ));

        let signals = strategy.process_event(&event, &state).await.unwrap();

        // Should generate signal with no delay
        assert!(!signals.is_empty());
        let signal = &signals[0];
        assert_eq!(signal.market_id, "eth-etf-market");
        assert_eq!(signal.outcome, Outcome::Yes); // Positive correlation, price up
    }

    #[tokio::test]
    async fn test_insignificant_movement_no_signal() {
        let config = test_config();
        let strategy = CrossAssetStrategy::new(config);
        let state = MockStateProvider::new().with_market(test_market());

        // Insignificant movement (below 5% threshold)
        let event = SystemEvent::CryptoPriceUpdate(CryptoPriceUpdateEvent::new(
            "ETH".to_string(),
            dec!(2500),
            dec!(2.0), // Below threshold
            dec!(3.0), // Also below threshold
            dec!(1000000000),
        ));

        let signals = strategy.process_event(&event, &state).await.unwrap();
        assert!(signals.is_empty());
    }

    #[tokio::test]
    async fn test_negative_correlation() {
        let config = CrossAssetConfig {
            enabled: true,
            correlations: vec![AssetCorrelation {
                crypto_symbol: "BTC".to_string(),
                market_keywords: vec!["regulation".to_string()],
                correlation_type: CorrelationType::Negative,
                min_confidence: dec!(0.7),
                order_size_usd: dec!(50),
            }],
            min_price_change_pct: dec!(5.0),
            signal_delay_secs: 0,
            cooldown_secs: 1,
            max_entry_price: dec!(0.85),
            price_history_minutes: 60,
            signal_priority: Priority::High,
        };

        let strategy = CrossAssetStrategy::new(config);

        let market = Market {
            condition_id: "crypto-regulation".to_string(),
            question: "Will crypto regulation increase?".to_string(),
            description: None,
            tags: vec![],
            yes_token_id: "reg-yes".to_string(),
            no_token_id: "reg-no".to_string(),
            created_at: Utc::now(),
            end_date: None,
            active: true,
            closed: false,
            volume: dec!(1000000),
            liquidity: dec!(50000),
        };

        let state = MockStateProvider::new()
            .with_market(market)
            .with_price("reg-no".to_string(), dec!(0.50));

        // BTC price goes up
        let event = SystemEvent::CryptoPriceUpdate(CryptoPriceUpdateEvent::new(
            "BTC".to_string(),
            dec!(50000),
            dec!(8.0),
            dec!(12.0),
            dec!(5000000000),
        ));

        let signals = strategy.process_event(&event, &state).await.unwrap();

        // Negative correlation: price up -> NO outcome
        if !signals.is_empty() {
            assert_eq!(signals[0].outcome, Outcome::No);
        }
    }

    #[tokio::test]
    async fn test_no_matching_market() {
        let config = test_config();
        let strategy = CrossAssetStrategy::new(config);
        let state = MockStateProvider::new(); // No markets

        let event = SystemEvent::CryptoPriceUpdate(CryptoPriceUpdateEvent::new(
            "ETH".to_string(),
            dec!(2500),
            dec!(10.0),
            dec!(15.0),
            dec!(1000000000),
        ));

        let signals = strategy.process_event(&event, &state).await.unwrap();
        assert!(signals.is_empty());
    }

    #[tokio::test]
    async fn test_accepts_only_crypto_events() {
        let config = test_config();
        let strategy = CrossAssetStrategy::new(config);

        assert!(strategy.accepts_event(&SystemEvent::CryptoPriceUpdate(
            CryptoPriceUpdateEvent::new(
                "ETH".to_string(),
                dec!(2500),
                dec!(5.0),
                dec!(10.0),
                dec!(1000000000),
            )
        )));

        assert!(!strategy.accepts_event(&SystemEvent::Heartbeat(
            polysniper_core::HeartbeatEvent {
                source: "test".to_string(),
                timestamp: Utc::now(),
            }
        )));
    }

    #[tokio::test]
    async fn test_price_too_high_blocked() {
        let config = test_config();
        let strategy = CrossAssetStrategy::new(config);
        let state = MockStateProvider::new()
            .with_market(test_market())
            .with_price("eth-etf-yes".to_string(), dec!(0.95)); // Above max entry

        let event = SystemEvent::CryptoPriceUpdate(CryptoPriceUpdateEvent::new(
            "ETH".to_string(),
            dec!(2500),
            dec!(10.0),
            dec!(15.0),
            dec!(1000000000),
        ));

        let signals = strategy.process_event(&event, &state).await.unwrap();
        assert!(signals.is_empty()); // Blocked by price check
    }
}
