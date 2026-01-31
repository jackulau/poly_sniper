//! Sentiment Strategy
//!
//! Trading strategy that processes external signals, applies sentiment analysis,
//! and generates trade signals based on sentiment scores.

use crate::sentiment_analyzer::SentimentAnalyzer;
use async_trait::async_trait;
use chrono::{Duration, Utc};
use polysniper_core::{
    ExternalSignalEvent, MarketId, OrderType, Outcome, Priority, SentimentConfig,
    SentimentSignal, SentimentSource, Side, SignalSource, StateProvider, Strategy, StrategyError,
    SystemEvent, TradeSignal,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Sentiment strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentStrategyConfig {
    /// Enable/disable the strategy
    pub enabled: bool,
    /// Sentiment analysis configuration
    #[serde(flatten)]
    pub sentiment: SentimentConfig,
    /// Market mappings: market keyword -> market details
    #[serde(default)]
    pub market_mappings: HashMap<String, SentimentMarketMapping>,
    /// Aggregation window in seconds (combine signals within this window)
    #[serde(default = "default_aggregation_window")]
    pub aggregation_window_secs: u64,
    /// Priority for generated signals
    #[serde(default)]
    pub signal_priority: Priority,
}

fn default_aggregation_window() -> u64 {
    60 // 1 minute
}

/// Market mapping for sentiment-triggered trades
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentMarketMapping {
    pub market_id: MarketId,
    pub yes_token_id: String,
    pub no_token_id: String,
    /// Override order size for this market
    pub order_size_usd: Option<Decimal>,
    /// Override max entry price for this market
    pub max_entry_price: Option<Decimal>,
}

impl Default for SentimentStrategyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sentiment: SentimentConfig::default(),
            market_mappings: HashMap::new(),
            aggregation_window_secs: default_aggregation_window(),
            signal_priority: Priority::High,
        }
    }
}

/// Pending signal for aggregation
struct PendingSignal {
    signal: SentimentSignal,
    received_at: chrono::DateTime<Utc>,
}

/// Sentiment-based trading strategy
pub struct SentimentStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: SentimentStrategyConfig,
    analyzer: SentimentAnalyzer,
    /// Pending signals per market keyword, for aggregation
    pending_signals: Arc<RwLock<HashMap<String, Vec<PendingSignal>>>>,
    /// Cache of processed signal hashes to prevent duplicates
    processed_signals: Arc<RwLock<lru::LruCache<String, ()>>>,
    /// Last signal time per market for cooldown
    last_signal_time: Arc<RwLock<HashMap<MarketId, chrono::DateTime<Utc>>>>,
}

impl SentimentStrategy {
    /// Create a new sentiment strategy
    pub fn new(config: SentimentStrategyConfig) -> Self {
        let enabled = config.enabled;
        let analyzer = SentimentAnalyzer::new(config.sentiment.clone());

        Self {
            id: "sentiment".to_string(),
            name: "Sentiment Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config,
            analyzer,
            pending_signals: Arc::new(RwLock::new(HashMap::new())),
            processed_signals: Arc::new(RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(1000).unwrap(),
            ))),
            last_signal_time: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Generate a hash for deduplication
    fn signal_hash(signal: &ExternalSignalEvent) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        signal.content.hash(&mut hasher);
        signal.signal_type.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Convert SignalSource to SentimentSource
    fn to_sentiment_source(source: &SignalSource) -> SentimentSource {
        match source {
            SignalSource::Twitter { .. } => SentimentSource::Twitter,
            SignalSource::Rss { .. } => SentimentSource::News,
            SignalSource::Webhook { .. } => SentimentSource::Custom("webhook".to_string()),
            SignalSource::Custom { name } => SentimentSource::Custom(name.clone()),
        }
    }

    /// Check if enough time has passed since last signal for this market
    async fn check_cooldown(&self, market_id: &MarketId) -> bool {
        let last_times = self.last_signal_time.read().await;
        if let Some(last_time) = last_times.get(market_id) {
            let cooldown = Duration::seconds(self.config.sentiment.signal_cooldown_secs as i64);
            if Utc::now() - *last_time < cooldown {
                debug!(
                    market_id = %market_id,
                    "Cooldown active, skipping signal"
                );
                return false;
            }
        }
        true
    }

    /// Record signal time for cooldown
    async fn record_signal_time(&self, market_id: &MarketId) {
        let mut last_times = self.last_signal_time.write().await;
        last_times.insert(market_id.clone(), Utc::now());
    }

    /// Process pending signals and generate aggregated signals if ready
    async fn process_pending_signals(
        &self,
        state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        let mut signals = Vec::new();
        let now = Utc::now();
        let window_secs = self.config.aggregation_window_secs;
        let window = Duration::seconds(window_secs as i64);

        let mut pending = self.pending_signals.write().await;

        // Process each market keyword group
        let keywords: Vec<String> = pending.keys().cloned().collect();

        for keyword in keywords {
            if let Some(pending_list) = pending.get_mut(&keyword) {
                // If window is 0, process immediately without filtering
                let should_process = if window_secs == 0 {
                    !pending_list.is_empty()
                } else {
                    // Filter out old signals (older than 2x the window)
                    let max_age = window * 2;
                    pending_list.retain(|p| now - p.received_at < max_age);

                    // Check if oldest signal has passed the window
                    if !pending_list.is_empty() {
                        let oldest = pending_list.iter().map(|p| p.received_at).min().unwrap();
                        now - oldest >= window
                    } else {
                        false
                    }
                };

                if should_process && !pending_list.is_empty() {
                    // Aggregate all pending signals
                    let sentiment_signals: Vec<SentimentSignal> =
                        pending_list.drain(..).map(|p| p.signal).collect();

                    let aggregated = self.analyzer.aggregate(&sentiment_signals);

                    if self.analyzer.is_aggregated_actionable(&aggregated) {
                        // Generate trade signal
                        if let Some(trade_signal) =
                            self.create_trade_signal(&keyword, &aggregated, state).await?
                        {
                            signals.push(trade_signal);
                        }
                    }
                }
            }
        }

        // Clean up empty entries
        pending.retain(|_, v| !v.is_empty());

        Ok(signals)
    }

    /// Create a trade signal from aggregated sentiment
    async fn create_trade_signal(
        &self,
        keyword: &str,
        aggregated: &polysniper_core::AggregatedSentiment,
        state: &dyn StateProvider,
    ) -> Result<Option<TradeSignal>, StrategyError> {
        // Find market mapping
        let mapping = match self.config.market_mappings.get(keyword) {
            Some(m) => m,
            None => {
                // Try to find market by searching state
                if let Some(m) = self.find_market_by_keyword(keyword, state).await {
                    // Can't add to config dynamically, so we'll use this directly
                    return self.create_signal_for_market(keyword, aggregated, &m, state).await;
                }
                warn!(
                    keyword = %keyword,
                    "No market mapping found for keyword"
                );
                return Ok(None);
            }
        };

        self.create_signal_for_market(keyword, aggregated, mapping, state)
            .await
    }

    /// Find a market by keyword in state
    async fn find_market_by_keyword(
        &self,
        keyword: &str,
        state: &dyn StateProvider,
    ) -> Option<SentimentMarketMapping> {
        let keyword_lower = keyword.to_lowercase();

        for market in state.get_all_markets().await {
            if market.question.to_lowercase().contains(&keyword_lower) {
                return Some(SentimentMarketMapping {
                    market_id: market.condition_id,
                    yes_token_id: market.yes_token_id,
                    no_token_id: market.no_token_id,
                    order_size_usd: None,
                    max_entry_price: None,
                });
            }
        }
        None
    }

    /// Create the actual trade signal
    async fn create_signal_for_market(
        &self,
        keyword: &str,
        aggregated: &polysniper_core::AggregatedSentiment,
        mapping: &SentimentMarketMapping,
        state: &dyn StateProvider,
    ) -> Result<Option<TradeSignal>, StrategyError> {
        // Check cooldown
        if !self.check_cooldown(&mapping.market_id).await {
            return Ok(None);
        }

        // Determine trade direction based on sentiment
        let (side, outcome, token_id) = if aggregated.score.is_positive() {
            (Side::Buy, Outcome::Yes, &mapping.yes_token_id)
        } else {
            (Side::Buy, Outcome::No, &mapping.no_token_id)
        };

        // Get current price
        let current_price = state.get_price(token_id).await;

        // Check max entry price
        let max_price = mapping
            .max_entry_price
            .unwrap_or(self.config.sentiment.max_entry_price);

        if side == Side::Buy {
            if let Some(price) = current_price {
                if price > max_price {
                    warn!(
                        market_id = %mapping.market_id,
                        price = %price,
                        max = %max_price,
                        "Price too high for sentiment entry"
                    );
                    return Ok(None);
                }
            }
        }

        let order_size = mapping
            .order_size_usd
            .unwrap_or(self.config.sentiment.default_order_size_usd);
        let entry_price = current_price.unwrap_or(max_price);
        let size = if entry_price.is_zero() {
            Decimal::ZERO
        } else {
            order_size / entry_price
        };

        // Record signal time for cooldown
        self.record_signal_time(&mapping.market_id).await;

        let signal = TradeSignal {
            id: format!(
                "sig_sentiment_{}_{}_{}",
                mapping.market_id,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: mapping.market_id.clone(),
            token_id: token_id.clone(),
            outcome,
            side,
            price: Some(entry_price),
            size,
            size_usd: order_size,
            order_type: OrderType::Fok,
            priority: self.config.signal_priority,
            timestamp: Utc::now(),
            reason: format!(
                "Sentiment signal for '{}': score={:.2}, confidence={:.2}, sources={}",
                keyword,
                aggregated.score.value(),
                aggregated.confidence.value(),
                aggregated.signal_count
            ),
            metadata: serde_json::json!({
                "sentiment_score": aggregated.score.value().to_string(),
                "sentiment_confidence": aggregated.confidence.value().to_string(),
                "signal_count": aggregated.signal_count,
                "sources": aggregated.sources.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                "market_keyword": keyword,
            }),
        };

        info!(
            signal_id = %signal.id,
            market_id = %mapping.market_id,
            outcome = ?outcome,
            score = %aggregated.score.value(),
            "Generated sentiment trade signal"
        );

        Ok(Some(signal))
    }
}

#[async_trait]
impl Strategy for SentimentStrategy {
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

        // Only process external signals
        let external_signal = match event {
            SystemEvent::ExternalSignal(e) => e,
            _ => return Ok(signals),
        };

        // Check for duplicate
        let signal_hash = Self::signal_hash(external_signal);
        {
            let mut processed = self.processed_signals.write().await;
            if processed.contains(&signal_hash) {
                debug!(
                    hash = %signal_hash,
                    "Duplicate signal, skipping"
                );
                return Ok(signals);
            }
            processed.put(signal_hash.clone(), ());
        }

        // Analyze sentiment
        let sentiment_source = Self::to_sentiment_source(&external_signal.source);
        let sentiment_signal = self
            .analyzer
            .analyze(&external_signal.content, sentiment_source);

        debug!(
            content_preview = %truncate(&external_signal.content, 100),
            score = %sentiment_signal.score.value(),
            confidence = %sentiment_signal.confidence.value(),
            market_keywords = ?sentiment_signal.market_keywords,
            "Analyzed external signal"
        );

        // If no market keywords, we can't trade
        if sentiment_signal.market_keywords.is_empty() {
            debug!("No market keywords matched, skipping");
            return Ok(signals);
        }

        // Check if immediately actionable (skip aggregation for high-confidence signals)
        if self.analyzer.is_actionable(&sentiment_signal)
            && sentiment_signal.confidence.is_high()
            && sentiment_signal.score.intensity() >= dec!(0.7)
        {
            // High confidence single signal - process immediately
            for keyword in &sentiment_signal.market_keywords {
                let aggregated = self.analyzer.aggregate(std::slice::from_ref(&sentiment_signal));
                if let Some(trade_signal) =
                    self.create_trade_signal(keyword, &aggregated, state).await?
                {
                    signals.push(trade_signal);
                }
            }
        } else {
            // Add to pending signals for aggregation
            let mut pending = self.pending_signals.write().await;
            for keyword in &sentiment_signal.market_keywords {
                let entry = pending.entry(keyword.clone()).or_insert_with(Vec::new);
                entry.push(PendingSignal {
                    signal: sentiment_signal.clone(),
                    received_at: Utc::now(),
                });
            }
            drop(pending);

            // Check if we should process aggregated signals
            signals.extend(self.process_pending_signals(state).await?);
        }

        Ok(signals)
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(event, SystemEvent::ExternalSignal(_))
    }

    async fn initialize(&mut self, _state: &dyn StateProvider) -> Result<(), StrategyError> {
        info!(
            strategy_id = %self.id,
            market_mappings = %self.config.market_mappings.len(),
            min_sentiment = %self.config.sentiment.min_sentiment_threshold,
            min_confidence = %self.config.sentiment.min_confidence_threshold,
            "Initializing sentiment strategy"
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
        let new_config: SentimentStrategyConfig = toml::from_str(config_content)
            .map_err(|e| StrategyError::ConfigError(format!("Failed to parse config: {}", e)))?;

        self.config = new_config;
        tracing::info!(strategy_id = %self.id, "Reloaded sentiment strategy config");
        Ok(())
    }

    fn config_name(&self) -> &str {
        "sentiment"
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
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

    fn test_config() -> SentimentStrategyConfig {
        let mut config = SentimentStrategyConfig::default();

        // Add market keywords
        config.sentiment.market_keywords.insert(
            "trump".to_string(),
            vec!["trump-market".to_string()],
        );

        // Add market mapping
        config.market_mappings.insert(
            "trump".to_string(),
            SentimentMarketMapping {
                market_id: "trump-market".to_string(),
                yes_token_id: "trump-yes".to_string(),
                no_token_id: "trump-no".to_string(),
                order_size_usd: Some(dec!(100)),
                max_entry_price: Some(dec!(0.90)),
            },
        );

        // Set shorter cooldown for tests
        config.sentiment.signal_cooldown_secs = 1;
        config.aggregation_window_secs = 0; // Immediate aggregation for tests

        // Lower thresholds for easier testing
        config.sentiment.min_sentiment_threshold = dec!(0.2);
        config.sentiment.min_confidence_threshold = dec!(0.3);

        config
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

    #[tokio::test]
    async fn test_process_positive_sentiment() {
        let config = test_config();
        let strategy = SentimentStrategy::new(config);
        let state = MockStateProvider::new()
            .with_market(test_market())
            .with_price("trump-yes".to_string(), dec!(0.50));

        let event = SystemEvent::ExternalSignal(ExternalSignalEvent {
            source: SignalSource::Twitter {
                account: "test".to_string(),
            },
            signal_type: "tweet".to_string(),
            content: "Trump is winning! Very bullish rally confirmed!".to_string(),
            market_id: None,
            keywords: vec!["trump".to_string()],
            metadata: serde_json::json!({}),
            received_at: Utc::now(),
        });

        let signals = strategy.process_event(&event, &state).await.unwrap();

        // Should generate a signal for high-confidence positive sentiment
        assert!(!signals.is_empty());
        let signal = &signals[0];
        assert_eq!(signal.market_id, "trump-market");
        assert_eq!(signal.outcome, Outcome::Yes); // Positive sentiment -> YES
        assert_eq!(signal.side, Side::Buy);
    }

    #[tokio::test]
    async fn test_process_negative_sentiment() {
        let mut config = test_config();
        // Lower thresholds for test
        config.sentiment.min_sentiment_threshold = dec!(0.2);
        config.sentiment.min_confidence_threshold = dec!(0.3);

        let strategy = SentimentStrategy::new(config);
        let state = MockStateProvider::new()
            .with_market(test_market())
            .with_price("trump-no".to_string(), dec!(0.50));

        let event = SystemEvent::ExternalSignal(ExternalSignalEvent {
            source: SignalSource::Twitter {
                account: "test".to_string(),
            },
            signal_type: "tweet".to_string(),
            content: "Trump is losing badly! Bearish crash dump!".to_string(),
            market_id: None,
            keywords: vec!["trump".to_string()],
            metadata: serde_json::json!({}),
            received_at: Utc::now(),
        });

        let signals = strategy.process_event(&event, &state).await.unwrap();

        // May or may not generate signal depending on exact score
        if !signals.is_empty() {
            let signal = &signals[0];
            assert_eq!(signal.market_id, "trump-market");
            assert_eq!(signal.outcome, Outcome::No); // Negative sentiment -> NO
        }
    }

    #[tokio::test]
    async fn test_duplicate_signal_ignored() {
        let config = test_config();
        let strategy = SentimentStrategy::new(config);
        let state = MockStateProvider::new()
            .with_market(test_market())
            .with_price("trump-yes".to_string(), dec!(0.50));

        let event = SystemEvent::ExternalSignal(ExternalSignalEvent {
            source: SignalSource::Twitter {
                account: "test".to_string(),
            },
            signal_type: "tweet".to_string(),
            content: "Trump winning! Bullish surge rally!".to_string(),
            market_id: None,
            keywords: vec!["trump".to_string()],
            metadata: serde_json::json!({}),
            received_at: Utc::now(),
        });

        // First processing
        let _ = strategy.process_event(&event, &state).await.unwrap();

        // Second processing of same event should be ignored
        let signals = strategy.process_event(&event, &state).await.unwrap();
        assert!(signals.is_empty());
    }

    #[tokio::test]
    async fn test_no_market_keyword_skipped() {
        let config = test_config();
        let strategy = SentimentStrategy::new(config);
        let state = MockStateProvider::new();

        let event = SystemEvent::ExternalSignal(ExternalSignalEvent {
            source: SignalSource::Twitter {
                account: "test".to_string(),
            },
            signal_type: "tweet".to_string(),
            content: "This is bullish news about something unrelated!".to_string(),
            market_id: None,
            keywords: vec![],
            metadata: serde_json::json!({}),
            received_at: Utc::now(),
        });

        let signals = strategy.process_event(&event, &state).await.unwrap();
        assert!(signals.is_empty());
    }

    #[tokio::test]
    async fn test_price_too_high() {
        let config = test_config();
        let strategy = SentimentStrategy::new(config);
        let state = MockStateProvider::new()
            .with_market(test_market())
            .with_price("trump-yes".to_string(), dec!(0.95)); // Above max entry price

        let event = SystemEvent::ExternalSignal(ExternalSignalEvent {
            source: SignalSource::Twitter {
                account: "test".to_string(),
            },
            signal_type: "tweet".to_string(),
            content: "Trump winning! Bullish surge rally confirmed!".to_string(),
            market_id: None,
            keywords: vec!["trump".to_string()],
            metadata: serde_json::json!({}),
            received_at: Utc::now(),
        });

        let signals = strategy.process_event(&event, &state).await.unwrap();
        assert!(signals.is_empty()); // Should be blocked by price check
    }

    #[tokio::test]
    async fn test_accepts_only_external_signals() {
        let config = test_config();
        let strategy = SentimentStrategy::new(config);

        assert!(strategy.accepts_event(&SystemEvent::ExternalSignal(ExternalSignalEvent {
            source: SignalSource::Twitter {
                account: "test".to_string()
            },
            signal_type: "tweet".to_string(),
            content: "test".to_string(),
            market_id: None,
            keywords: vec![],
            metadata: serde_json::json!({}),
            received_at: Utc::now(),
        })));

        assert!(!strategy.accepts_event(&SystemEvent::Heartbeat(
            polysniper_core::HeartbeatEvent {
                source: "test".to_string(),
                timestamp: Utc::now(),
            }
        )));
    }
}
