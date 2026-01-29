//! Event-Based Strategy
//!
//! Reacts to external signals (webhooks, RSS feeds, etc.) to generate trades.

use async_trait::async_trait;
use chrono::Utc;
use polysniper_core::{
    ExternalSignalEvent, MarketId, OrderType, Outcome, Priority, Side, SignalSource, StateProvider,
    Strategy, StrategyError, SystemEvent, TradeSignal,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::ml_processor::MlSignalProcessor;

/// Event-based strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventBasedConfig {
    pub enabled: bool,
    /// Signal rules for matching and action
    pub rules: Vec<SignalRule>,
    /// Default order size in USD
    pub default_order_size_usd: Decimal,
    /// Market search keywords to market ID mapping
    #[serde(default)]
    pub market_mappings: HashMap<String, MarketMapping>,
    /// ML signal processing configuration
    #[serde(default)]
    pub ml_config: MlConfig,
}

/// Rule for processing external signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalRule {
    /// Rule name
    pub name: String,
    /// Keywords to match in signal content
    pub keywords: Vec<String>,
    /// Signal source filter (optional)
    pub source_filter: Option<SourceFilter>,
    /// Action to take when matched
    pub action: RuleAction,
    /// Priority for this rule
    #[serde(default = "default_priority")]
    pub priority: Priority,
}

fn default_priority() -> Priority {
    Priority::High
}

/// Filter for signal sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFilter {
    /// Allowed webhook endpoints
    pub webhook_endpoints: Option<Vec<String>>,
    /// Allowed RSS feeds
    pub rss_feeds: Option<Vec<String>>,
    /// Allowed Twitter accounts
    pub twitter_accounts: Option<Vec<String>>,
}

/// Action to take on rule match
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleAction {
    /// Market ID to trade (if known)
    pub market_id: Option<String>,
    /// Keywords to search for market
    pub market_keywords: Option<Vec<String>>,
    /// Side to trade
    pub side: Side,
    /// Outcome to trade
    pub outcome: Outcome,
    /// Order size in USD (overrides default)
    pub order_size_usd: Option<Decimal>,
    /// Maximum price for entry
    #[serde(default = "default_max_price")]
    pub max_entry_price: Decimal,
}

fn default_max_price() -> Decimal {
    dec!(0.90)
}

/// Market mapping from keywords to market info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMapping {
    pub market_id: MarketId,
    pub yes_token_id: String,
    pub no_token_id: String,
}

impl Default for EventBasedConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: Vec::new(),
            default_order_size_usd: dec!(100),
            market_mappings: HashMap::new(),
            ml_config: MlConfig::default(),
        }
    }
}

/// Event-Based Strategy
pub struct EventBasedStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: Arc<RwLock<EventBasedConfig>>,
    /// Cache of matched signals to prevent duplicates
    processed_signals: Arc<RwLock<lru::LruCache<String, ()>>>,
    /// ML signal processor
    ml_processor: MlSignalProcessor,
}

impl EventBasedStrategy {
    /// Create a new event-based strategy
    pub fn new(config: EventBasedConfig) -> Self {
        let enabled = config.enabled;
        let ml_processor = MlSignalProcessor::new(config.ml_config.clone());
        Self {
            id: "event_based".to_string(),
            name: "Event-Based Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config: Arc::new(RwLock::new(config)),
            processed_signals: Arc::new(RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(1000).unwrap(),
            ))),
            ml_processor,
        }
    }

    /// Get reference to the ML processor
    pub fn ml_processor(&self) -> &MlSignalProcessor {
        &self.ml_processor
    }

    /// Process ML signal and generate trade signal if applicable
    async fn process_ml_signal(
        &self,
        external_signal: &ExternalSignalEvent,
        state: &dyn StateProvider,
    ) -> Result<Option<TradeSignal>, StrategyError> {
        // Try to process as ML signal
        let ml_result = match self.ml_processor.process_signal(external_signal).await {
            Some(result) => result,
            None => return Ok(None), // Not an ML signal
        };

        if !ml_result.should_execute {
            debug!(
                reason = ?ml_result.rejection_reason,
                "ML signal rejected"
            );
            return Ok(None);
        }

        // Get market from prediction
        let prediction = &ml_result.prediction;
        let market_id = match &prediction.market_id {
            Some(id) => id.clone(),
            None => {
                // Try to find market by keywords
                for market in state.get_all_markets().await {
                    let question_lower = market.question.to_lowercase();
                    if prediction
                        .market_keywords
                        .iter()
                        .any(|kw| question_lower.contains(&kw.to_lowercase()))
                    {
                        return self
                            .create_ml_trade_signal(
                                &market.condition_id,
                                &market.yes_token_id,
                                &market.no_token_id,
                                &ml_result,
                                state,
                            )
                            .await;
                    }
                }
                warn!(
                    prediction_id = %prediction.id,
                    "Could not find market for ML prediction"
                );
                return Ok(None);
            }
        };

        // Get market details
        let market = match state.get_market(&market_id).await {
            Some(m) => m,
            None => {
                warn!(
                    market_id = %market_id,
                    "Market not found for ML prediction"
                );
                return Ok(None);
            }
        };

        self.create_ml_trade_signal(
            &market.condition_id,
            &market.yes_token_id,
            &market.no_token_id,
            &ml_result,
            state,
        )
        .await
    }

    /// Create a trade signal from ML processing result
    async fn create_ml_trade_signal(
        &self,
        market_id: &MarketId,
        yes_token_id: &str,
        no_token_id: &str,
        ml_result: &crate::ml_processor::MlProcessingResult,
        state: &dyn StateProvider,
    ) -> Result<Option<TradeSignal>, StrategyError> {
        let prediction = &ml_result.prediction;
        let outcome = ml_result.suggested_outcome;

        let token_id = match outcome {
            Outcome::Yes => yes_token_id.to_string(),
            Outcome::No => no_token_id.to_string(),
        };

        // Get current price
        let current_price = state.get_price(&token_id).await;
        let entry_price = current_price.unwrap_or(dec!(0.50));

        // Calculate order size with ML confidence multiplier
        let base_size_usd = self.config.default_order_size_usd;
        let adjusted_size_usd = base_size_usd * ml_result.size_multiplier;

        let size = if entry_price.is_zero() {
            Decimal::ZERO
        } else {
            adjusted_size_usd / entry_price
        };

        let signal = TradeSignal {
            id: format!(
                "sig_ml_{}_{}_{}_{}",
                prediction.model_id,
                market_id,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: market_id.clone(),
            token_id,
            outcome,
            side: Side::Buy,
            price: Some(entry_price),
            size,
            size_usd: adjusted_size_usd,
            order_type: OrderType::Fok,
            priority: Priority::High,
            timestamp: Utc::now(),
            reason: format!(
                "ML prediction from model '{}' with {:.1}% confidence",
                prediction.model_id,
                prediction.confidence * dec!(100)
            ),
            metadata: serde_json::json!({
                "ml_signal": true,
                "model_id": prediction.model_id,
                "model_version": prediction.model_version,
                "prediction_id": prediction.id,
                "confidence": prediction.confidence.to_string(),
                "size_multiplier": ml_result.size_multiplier.to_string(),
                "base_size_usd": base_size_usd.to_string(),
                "adjusted_size_usd": adjusted_size_usd.to_string(),
            }),
        };

        info!(
            signal_id = %signal.id,
            model_id = %prediction.model_id,
            confidence = %prediction.confidence,
            size_multiplier = %ml_result.size_multiplier,
            outcome = ?outcome,
            size_usd = %adjusted_size_usd,
            "Generated ML trade signal"
        );

        Ok(Some(signal))
    }

    /// Generate a signal hash for deduplication
    fn signal_hash(signal: &ExternalSignalEvent) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        signal.content.hash(&mut hasher);
        signal.signal_type.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Check if signal matches source filter
    fn matches_source_filter(&self, signal: &ExternalSignalEvent, filter: &SourceFilter) -> bool {
        match &signal.source {
            SignalSource::Webhook { endpoint } => {
                if let Some(allowed) = &filter.webhook_endpoints {
                    return allowed.iter().any(|e| endpoint.contains(e));
                }
            }
            SignalSource::Rss { feed_url } => {
                if let Some(allowed) = &filter.rss_feeds {
                    return allowed.iter().any(|f| feed_url.contains(f));
                }
            }
            SignalSource::Twitter { account } => {
                if let Some(allowed) = &filter.twitter_accounts {
                    return allowed.iter().any(|a| account.eq_ignore_ascii_case(a));
                }
            }
            SignalSource::Custom { .. } => return true,
        }
        // If no specific filter for this source type, allow it
        true
    }

    /// Check if signal content matches keywords
    fn matches_keywords(&self, signal: &ExternalSignalEvent, keywords: &[String]) -> bool {
        let content_lower = signal.content.to_lowercase();

        // Also check signal keywords
        let signal_keywords: Vec<String> =
            signal.keywords.iter().map(|k| k.to_lowercase()).collect();

        keywords.iter().any(|kw| {
            let kw_lower = kw.to_lowercase();
            content_lower.contains(&kw_lower)
                || signal_keywords.iter().any(|sk| sk.contains(&kw_lower))
        })
    }

    /// Find matching rule for signal (using pre-fetched rules)
    fn find_matching_rule_sync<'a>(
        &self,
        signal: &ExternalSignalEvent,
        rules: &'a [SignalRule],
    ) -> Option<&'a SignalRule> {
        for rule in rules {
            // Check source filter
            if let Some(filter) = &rule.source_filter {
                if !self.matches_source_filter(signal, filter) {
                    continue;
                }
            }

            // Check keywords
            if self.matches_keywords(signal, &rule.keywords) {
                return Some(rule);
            }
        }
        None
    }

    /// Resolve market from action (using pre-fetched market_mappings)
    async fn resolve_market(
        &self,
        action: &RuleAction,
        signal: &ExternalSignalEvent,
        state: &dyn StateProvider,
        market_mappings: &HashMap<String, MarketMapping>,
    ) -> Option<(MarketId, String, String)> {
        // If signal already has market ID
        if let Some(market_id) = &signal.market_id {
            if let Some(market) = state.get_market(market_id).await {
                return Some((market.condition_id, market.yes_token_id, market.no_token_id));
            }
        }

        // If action has explicit market ID
        if let Some(market_id) = &action.market_id {
            if let Some(market) = state.get_market(market_id).await {
                return Some((market.condition_id, market.yes_token_id, market.no_token_id));
            }
        }

        // Try market mappings
        if let Some(keywords) = &action.market_keywords {
            for keyword in keywords {
                if let Some(mapping) = market_mappings.get(keyword) {
                    return Some((
                        mapping.market_id.clone(),
                        mapping.yes_token_id.clone(),
                        mapping.no_token_id.clone(),
                    ));
                }
            }
        }

        // Try to find market by searching state
        // This is a simplified search - in production, would use Gamma API
        for market in state.get_all_markets().await {
            if let Some(keywords) = &action.market_keywords {
                let question_lower = market.question.to_lowercase();
                if keywords
                    .iter()
                    .any(|kw| question_lower.contains(&kw.to_lowercase()))
                {
                    return Some((market.condition_id, market.yes_token_id, market.no_token_id));
                }
            }
        }

        None
    }
}

#[async_trait]
impl Strategy for EventBasedStrategy {
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

        // First, try to process as ML signal
        if self.ml_processor.is_ml_signal(external_signal) {
            if let Some(ml_signal) = self.process_ml_signal(external_signal, state).await? {
                signals.push(ml_signal);
                return Ok(signals);
            }
            // If ML processing returned None but it was an ML signal, don't continue to rule matching
            return Ok(signals);
        }

        // Find matching rule (for non-ML signals)
        let rule = match self.find_matching_rule(external_signal) {
            Some(r) => r,
            None => {
                debug!(
                    content = %external_signal.content,
                    "No matching rule for signal"
                );
                return Ok(signals);
            }
        };

        info!(
            rule_name = %rule.name,
            content = %external_signal.content,
            "Signal matched rule"
        );

        // Resolve market
        let (market_id, yes_token_id, no_token_id) = match self
            .resolve_market(&rule.action, external_signal, state)
            .await
        {
            Some(m) => m,
            None => {
                warn!(
                    rule_name = %rule.name,
                    "Could not resolve market for signal"
                );
                return Ok(signals);
            }
        };

        let token_id = match rule.action.outcome {
            Outcome::Yes => yes_token_id,
            Outcome::No => no_token_id,
        };

        // Get current price
        let current_price = state.get_price(&token_id).await;

        // Check max entry price for buys
        if rule.action.side == Side::Buy {
            if let Some(price) = current_price {
                if price > rule.action.max_entry_price {
                    warn!(
                        market_id = %market_id,
                        price = %price,
                        max = %rule.action.max_entry_price,
                        "Price too high for entry"
                    );
                    return Ok(signals);
                }
            }
        }

        let order_size = rule.action.order_size_usd.unwrap_or(default_order_size_usd);
        let entry_price = current_price.unwrap_or(rule.action.max_entry_price);
        let size = if entry_price.is_zero() {
            Decimal::ZERO
        } else {
            order_size / entry_price
        };

        let signal = TradeSignal {
            id: format!(
                "sig_event_{}_{}_{}",
                market_id,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id,
            token_id,
            outcome: rule.action.outcome,
            side: rule.action.side,
            price: Some(entry_price),
            size,
            size_usd: order_size,
            order_type: OrderType::Fok,
            priority: rule.priority,
            timestamp: Utc::now(),
            reason: format!(
                "External signal matched rule '{}': {}",
                rule.name,
                truncate(&external_signal.content, 100)
            ),
            metadata: serde_json::json!({
                "rule_name": rule.name,
                "signal_type": external_signal.signal_type,
                "signal_content": truncate(&external_signal.content, 500),
                "source": format!("{:?}", external_signal.source),
            }),
        };

        signals.push(signal);

        Ok(signals)
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(event, SystemEvent::ExternalSignal(_))
    }

    async fn initialize(&mut self, _state: &dyn StateProvider) -> Result<(), StrategyError> {
        let config = self.config.read().await;
        info!(
            strategy_id = %self.id,
            rules_count = %config.rules.len(),
            "Initializing event-based strategy"
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
        let new_config: EventBasedConfig = toml::from_str(config_content).map_err(|e| {
            StrategyError::ConfigError(format!("Failed to parse event_based config: {}", e))
        })?;

        // Validate config
        if new_config.default_order_size_usd <= Decimal::ZERO {
            return Err(StrategyError::ConfigError(
                "default_order_size_usd must be positive".to_string(),
            ));
        }

        // Validate rules
        for rule in &new_config.rules {
            if rule.keywords.is_empty() {
                return Err(StrategyError::ConfigError(format!(
                    "Rule '{}' must have at least one keyword",
                    rule.name
                )));
            }
        }

        // Update enabled state
        self.enabled.store(new_config.enabled, Ordering::SeqCst);

        // Update config atomically
        let mut config = self.config.write().await;
        *config = new_config;

        info!(
            strategy_id = %self.id,
            rules_count = %config.rules.len(),
            "Reloaded event_based configuration"
        );

        Ok(())
    }

    fn config_name(&self) -> &str {
        "event_based"
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
