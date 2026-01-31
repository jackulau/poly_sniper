//! LLM Prediction Strategy
//!
//! Analyzes Polymarket markets using an LLM and generates trade signals
//! based on confidence scores and price edge.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use polysniper_core::{
    Market, MarketId, OrderType, Outcome, Priority, Side, StateProvider, Strategy, StrategyError,
    SystemEvent, TradeSignal,
};
use polysniper_data::{
    ChatCompletionRequest, ChatMessage, OpenRouterClient, OpenRouterConfig, ResponseFormat,
};
use polysniper_ml::{FeatureStore, FeatureStoreConfig};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{sleep, Duration};
use tracing::{debug, info, warn};

/// LLM Prediction Strategy Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmPredictionConfig {
    /// Whether the strategy is enabled
    pub enabled: bool,
    /// LLM model to use (e.g., "x-ai/grok-3-latest")
    pub model: String,
    /// Environment variable containing the API key
    pub api_key_env: String,
    /// Temperature for LLM responses (0.0 - 1.0)
    pub temperature: f32,
    /// Maximum tokens for LLM response
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Interval between analysis runs in seconds
    pub analysis_interval_secs: u64,
    /// Maximum markets to analyze per interval
    pub max_markets_per_interval: usize,
    /// Delay between API calls in milliseconds
    #[serde(default = "default_api_delay")]
    pub api_call_delay_ms: u64,
    /// Minimum confidence threshold to generate signals (0.0 - 1.0)
    pub confidence_threshold: f64,
    /// Minimum price edge required (difference between LLM fair value and market price)
    pub min_price_edge: Decimal,
    /// Order size in USD
    pub order_size_usd: Decimal,
    /// Minimum market liquidity in USD
    pub min_liquidity_usd: Decimal,
    /// Order type (Gtc, Fok, Gtd)
    #[serde(default = "default_order_type")]
    pub order_type: String,
    /// Specific market IDs to analyze (empty = all markets)
    #[serde(default)]
    pub markets: Vec<String>,
    /// Filter markets by tags
    #[serde(default)]
    pub tags: Vec<String>,
    /// Filter markets by keywords in question
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Override the default system prompt
    #[serde(default)]
    pub system_prompt_override: Option<String>,
    /// Whether to use the feature store for enhanced prompts
    #[serde(default)]
    pub use_feature_store: bool,
}

fn default_max_tokens() -> u32 {
    1024
}

fn default_api_delay() -> u64 {
    500
}

fn default_order_type() -> String {
    "Gtc".to_string()
}

impl Default for LlmPredictionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "x-ai/grok-3-latest".to_string(),
            api_key_env: "OPENROUTER_API_KEY".to_string(),
            temperature: 0.3,
            max_tokens: 1024,
            analysis_interval_secs: 300,
            max_markets_per_interval: 10,
            api_call_delay_ms: 500,
            confidence_threshold: 0.75,
            min_price_edge: dec!(0.05),
            order_size_usd: dec!(50),
            min_liquidity_usd: dec!(5000),
            order_type: "Gtc".to_string(),
            markets: Vec::new(),
            tags: Vec::new(),
            keywords: Vec::new(),
            system_prompt_override: None,
            use_feature_store: false,
        }
    }
}

/// LLM prediction response
#[derive(Debug, Clone, Deserialize)]
pub struct LlmPrediction {
    /// Prediction: "yes", "no", or "hold"
    pub prediction: String,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    /// Reasoning for the prediction
    pub reasoning: String,
    /// Optional target price (fair value estimate)
    #[serde(default)]
    pub target_price: Option<f64>,
}

/// Cached prediction with timestamp
struct CachedPrediction {
    prediction: LlmPrediction,
    timestamp: DateTime<Utc>,
}

/// LLM Prediction Strategy
pub struct LlmPredictionStrategy {
    id: String,
    config: LlmPredictionConfig,
    enabled: Arc<AtomicBool>,
    openrouter_client: OpenRouterClient,
    prediction_cache: Arc<RwLock<HashMap<MarketId, CachedPrediction>>>,
    last_analysis_time: Arc<RwLock<DateTime<Utc>>>,
    feature_store: Option<Arc<FeatureStore>>,
}

impl LlmPredictionStrategy {
    /// Create a new LLM prediction strategy
    pub fn new(config: LlmPredictionConfig) -> Result<Self, StrategyError> {
        // Get API key from environment
        let api_key = std::env::var(&config.api_key_env).map_err(|_| {
            StrategyError::ConfigError(format!(
                "Environment variable {} not set",
                config.api_key_env
            ))
        })?;

        let openrouter_config = OpenRouterConfig {
            api_key,
            default_model: config.model.clone(),
            ..Default::default()
        };

        let openrouter_client = OpenRouterClient::new(openrouter_config)
            .map_err(|e| StrategyError::InitializationError(e.to_string()))?;

        let enabled = config.enabled;
        let use_feature_store = config.use_feature_store;

        // Initialize feature store if configured
        let feature_store = if use_feature_store {
            let fs_config = FeatureStoreConfig::default();
            let fs = FeatureStore::new(fs_config);
            // Register all standard feature computers
            let mut registry = polysniper_ml::FeatureRegistry::new();
            polysniper_ml::register_all_computers(&mut registry);
            Some(Arc::new(fs))
        } else {
            None
        };

        Ok(Self {
            id: "llm_prediction".to_string(),
            config,
            enabled: Arc::new(AtomicBool::new(enabled)),
            openrouter_client,
            prediction_cache: Arc::new(RwLock::new(HashMap::new())),
            last_analysis_time: Arc::new(RwLock::new(Utc::now() - chrono::Duration::hours(1))),
            feature_store,
        })
    }

    /// Create a new LLM prediction strategy with a pre-configured feature store
    pub fn with_feature_store(
        config: LlmPredictionConfig,
        feature_store: Arc<FeatureStore>,
    ) -> Result<Self, StrategyError> {
        // Get API key from environment
        let api_key = std::env::var(&config.api_key_env).map_err(|_| {
            StrategyError::ConfigError(format!(
                "Environment variable {} not set",
                config.api_key_env
            ))
        })?;

        let openrouter_config = OpenRouterConfig {
            api_key,
            default_model: config.model.clone(),
            ..Default::default()
        };

        let openrouter_client = OpenRouterClient::new(openrouter_config)
            .map_err(|e| StrategyError::InitializationError(e.to_string()))?;

        let enabled = config.enabled;

        Ok(Self {
            id: "llm_prediction".to_string(),
            config,
            enabled: Arc::new(AtomicBool::new(enabled)),
            openrouter_client,
            prediction_cache: Arc::new(RwLock::new(HashMap::new())),
            last_analysis_time: Arc::new(RwLock::new(Utc::now() - chrono::Duration::hours(1))),
            feature_store: Some(feature_store),
        })
    }

    /// Build the system prompt for market analysis
    fn build_system_prompt(&self) -> String {
        if let Some(override_prompt) = &self.config.system_prompt_override {
            return override_prompt.clone();
        }

        r#"You are a prediction market analyst. Analyze the market and provide your prediction.

Respond with JSON only:
{
  "prediction": "yes" | "no" | "hold",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "target_price": 0.0-1.0 (optional, your fair value estimate)
}

Guidelines:
- "yes" means you believe the outcome will happen
- "no" means you believe the outcome will NOT happen
- "hold" means insufficient information or no edge
- confidence should reflect your certainty (0.5 = uncertain, 1.0 = certain)
- Only predict "yes" or "no" if confidence > 0.6

Be concise. Focus on key factors that will determine the outcome."#
            .to_string()
    }

    /// Build the user prompt for a specific market
    fn build_user_prompt(&self, market: &Market, yes_price: Decimal) -> String {
        let no_price = Decimal::ONE - yes_price;
        let yes_probability = (yes_price * dec!(100)).round();

        let time_remaining = match &market.end_date {
            Some(end_date) => {
                let now = Utc::now();
                if *end_date > now {
                    let duration = *end_date - now;
                    let days = duration.num_days();
                    let hours = duration.num_hours() % 24;
                    if days > 0 {
                        format!("{} days {} hours", days, hours)
                    } else {
                        format!("{} hours", duration.num_hours())
                    }
                } else {
                    "Expired".to_string()
                }
            }
            None => "Unknown".to_string(),
        };

        let description = market
            .description
            .as_deref()
            .unwrap_or("No description available");

        format!(
            r#"Market: {question}

Description: {description}

Current YES price: {yes_price} (implies {yes_probability}% probability)
Current NO price: {no_price}

24h Volume: ${volume}
Liquidity: ${liquidity}
Time until resolution: {time_remaining}

Analyze this market and provide your prediction."#,
            question = market.question,
            description = description,
            yes_price = yes_price,
            yes_probability = yes_probability,
            no_price = no_price,
            volume = market.volume.round(),
            liquidity = market.liquidity.round(),
            time_remaining = time_remaining,
        )
    }

    /// Build an enhanced user prompt with feature store data
    async fn build_enhanced_prompt(
        &self,
        market: &Market,
        yes_price: Decimal,
        state: &dyn StateProvider,
    ) -> String {
        // Start with the base prompt
        let base_prompt = self.build_user_prompt(market, yes_price);

        // If no feature store, return base prompt
        let feature_store = match &self.feature_store {
            Some(fs) => fs,
            None => return base_prompt,
        };

        // Try to get features
        let features = match feature_store
            .get_current_features(
                &market.condition_id,
                &["orderbook", "sentiment", "temporal", "price_history"],
                state,
            )
            .await
        {
            Ok(f) => f,
            Err(e) => {
                debug!(error = %e, "Failed to get features, using base prompt");
                return base_prompt;
            }
        };

        // Build enhanced prompt with feature data
        let mut enhanced = base_prompt;

        // Add orderbook features if available
        if let Some(orderbook) = features.get("orderbook") {
            enhanced.push_str("\n\nOrderbook Analysis:");
            if let Some(imbalance) = orderbook.value.get("imbalance_ratio") {
                enhanced.push_str(&format!("\n- Order imbalance: {}", imbalance));
            }
            if let Some(spread) = orderbook.value.get("spread") {
                enhanced.push_str(&format!("\n- Spread: {}", spread));
            }
            if let Some(bid_depth) = orderbook.value.get("bid_depth") {
                enhanced.push_str(&format!("\n- Bid depth: ${}", bid_depth));
            }
            if let Some(ask_depth) = orderbook.value.get("ask_depth") {
                enhanced.push_str(&format!("\n- Ask depth: ${}", ask_depth));
            }
        }

        // Add sentiment features if available
        if let Some(sentiment) = features.get("sentiment") {
            enhanced.push_str("\n\nSentiment Analysis:");
            if let Some(score) = sentiment.value.get("sentiment_score") {
                enhanced.push_str(&format!("\n- Sentiment score: {}", score));
            }
            if let Some(uncertainty) = sentiment.value.get("uncertainty_count") {
                enhanced.push_str(&format!("\n- Uncertainty indicators: {}", uncertainty));
            }
        }

        // Add price history features if available
        if let Some(price_hist) = features.get("price_history") {
            enhanced.push_str("\n\nPrice History Analysis:");
            if let Some(volatility) = price_hist.value.get("volatility") {
                enhanced.push_str(&format!("\n- Recent volatility: {}", volatility));
            }
            if let Some(momentum) = price_hist.value.get("momentum") {
                enhanced.push_str(&format!("\n- Momentum: {}", momentum));
            }
            if let Some(trend) = price_hist.value.get("trend_strength") {
                enhanced.push_str(&format!("\n- Trend strength: {}", trend));
            }
        }

        enhanced
    }

    /// Parse the LLM response into a prediction
    fn parse_llm_response(&self, content: &str) -> Result<LlmPrediction, StrategyError> {
        // Try to extract JSON from the response
        let json_content = if let Some(start) = content.find('{') {
            if let Some(end) = content.rfind('}') {
                &content[start..=end]
            } else {
                content
            }
        } else {
            content
        };

        serde_json::from_str(json_content).map_err(|e| {
            StrategyError::ProcessingError(format!(
                "Failed to parse LLM response as JSON: {}. Content: {}",
                e,
                truncate(content, 200)
            ))
        })
    }

    /// Check if a prediction should generate a signal
    fn should_generate_signal(&self, prediction: &LlmPrediction, current_price: Decimal) -> bool {
        // Skip "hold" predictions
        if prediction.prediction.to_lowercase() == "hold" {
            return false;
        }

        // Check confidence threshold
        if prediction.confidence < self.config.confidence_threshold {
            return false;
        }

        // Check price edge if target price is provided
        if let Some(target) = prediction.target_price {
            let target_decimal = Decimal::try_from(target).unwrap_or(current_price);
            let edge = match prediction.prediction.to_lowercase().as_str() {
                "yes" => target_decimal - current_price,
                "no" => (Decimal::ONE - target_decimal) - (Decimal::ONE - current_price),
                _ => return false,
            };

            if edge < self.config.min_price_edge {
                return false;
            }
        }

        true
    }

    /// Filter markets based on configuration
    fn filter_markets(&self, markets: &[Market]) -> Vec<Market> {
        markets
            .iter()
            .filter(|market| {
                // Skip inactive or closed markets
                if !market.active || market.closed {
                    return false;
                }

                // Check minimum liquidity
                if market.liquidity < self.config.min_liquidity_usd {
                    return false;
                }

                // Filter by specific market IDs if configured
                if !self.config.markets.is_empty()
                    && !self.config.markets.contains(&market.condition_id)
                {
                    return false;
                }

                // Filter by tags if configured
                if !self.config.tags.is_empty() {
                    let has_matching_tag = self
                        .config
                        .tags
                        .iter()
                        .any(|t| market.tags.iter().any(|mt| mt.eq_ignore_ascii_case(t)));
                    if !has_matching_tag {
                        return false;
                    }
                }

                // Filter by keywords if configured
                if !self.config.keywords.is_empty() {
                    let question_lower = market.question.to_lowercase();
                    let has_matching_keyword = self
                        .config
                        .keywords
                        .iter()
                        .any(|k| question_lower.contains(&k.to_lowercase()));
                    if !has_matching_keyword {
                        return false;
                    }
                }

                true
            })
            .cloned()
            .collect()
    }

    /// Analyze a market using the LLM
    async fn analyze_market(
        &self,
        market: &Market,
        state: &dyn StateProvider,
    ) -> Result<Option<LlmPrediction>, StrategyError> {
        // Get current price
        let current_price = match state.get_price(&market.yes_token_id).await {
            Some(p) => p,
            None => {
                debug!(
                    market_id = %market.condition_id,
                    "No price available for market, skipping"
                );
                return Ok(None);
            }
        };

        // Build prompts (use enhanced prompt if feature store is available)
        let system_prompt = self.build_system_prompt();
        let user_prompt = if self.feature_store.is_some() {
            self.build_enhanced_prompt(market, current_price, state).await
        } else {
            self.build_user_prompt(market, current_price)
        };

        // Create chat completion request
        let request = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages: vec![
                ChatMessage::system(&system_prompt),
                ChatMessage::user(&user_prompt),
            ],
            temperature: Some(self.config.temperature),
            max_tokens: Some(self.config.max_tokens),
            response_format: Some(ResponseFormat::json()),
        };

        // Call OpenRouter
        let response = self
            .openrouter_client
            .chat_completion(request)
            .await
            .map_err(|e| StrategyError::ProcessingError(format!("LLM API error: {}", e)))?;

        // Extract response content
        let content = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| StrategyError::ProcessingError("Empty response from LLM".to_string()))?;

        // Parse response
        let prediction = self.parse_llm_response(&content)?;

        debug!(
            market_id = %market.condition_id,
            prediction = %prediction.prediction,
            confidence = %prediction.confidence,
            "LLM analysis complete"
        );

        Ok(Some(prediction))
    }

    /// Generate a trade signal from a prediction
    fn generate_signal(
        &self,
        market: &Market,
        prediction: &LlmPrediction,
        current_price: Decimal,
    ) -> TradeSignal {
        let (side, outcome, token_id, price) = match prediction.prediction.to_lowercase().as_str() {
            "yes" => (
                Side::Buy,
                Outcome::Yes,
                market.yes_token_id.clone(),
                current_price,
            ),
            "no" => (
                Side::Buy,
                Outcome::No,
                market.no_token_id.clone(),
                Decimal::ONE - current_price,
            ),
            _ => unreachable!(), // "hold" filtered out earlier
        };

        let size = if price.is_zero() {
            Decimal::ZERO
        } else {
            self.config.order_size_usd / price
        };

        let order_type = match self.config.order_type.to_lowercase().as_str() {
            "gtc" => OrderType::Gtc,
            "fok" => OrderType::Fok,
            "gtd" => OrderType::Gtd,
            _ => OrderType::Gtc,
        };

        TradeSignal {
            id: format!(
                "sig_llm_{}_{}_{}",
                market.condition_id,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: market.condition_id.clone(),
            token_id,
            outcome,
            side,
            price: Some(price),
            size,
            size_usd: self.config.order_size_usd,
            order_type,
            priority: Priority::Normal,
            timestamp: Utc::now(),
            reason: format!(
                "LLM predicts {} with {:.0}% confidence. Reasoning: {}",
                prediction.prediction,
                prediction.confidence * 100.0,
                truncate(&prediction.reasoning, 200)
            ),
            metadata: serde_json::json!({
                "llm_model": self.config.model,
                "confidence": prediction.confidence,
                "target_price": prediction.target_price,
                "full_reasoning": prediction.reasoning,
            }),
        }
    }

    /// Check if cached prediction is still valid
    fn is_cache_valid(&self, cached: &CachedPrediction) -> bool {
        let cache_duration = chrono::Duration::seconds(self.config.analysis_interval_secs as i64);
        Utc::now() - cached.timestamp < cache_duration
    }
}

#[async_trait]
impl Strategy for LlmPredictionStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        "LLM Prediction Strategy"
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(event, SystemEvent::Heartbeat(_))
    }

    async fn process_event(
        &self,
        event: &SystemEvent,
        state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        let mut signals = Vec::new();

        // Only process heartbeat events
        if !matches!(event, SystemEvent::Heartbeat(_)) {
            return Ok(signals);
        }

        // Check if enough time has passed since last analysis
        let interval = chrono::Duration::seconds(self.config.analysis_interval_secs as i64);
        {
            let last_time = self.last_analysis_time.read().await;
            if Utc::now() - *last_time < interval {
                debug!("Analysis interval not reached, skipping");
                return Ok(signals);
            }
        }

        // Update last analysis time
        {
            let mut last_time = self.last_analysis_time.write().await;
            *last_time = Utc::now();
        }

        info!("Starting LLM market analysis");

        // Get all markets and filter
        let all_markets = state.get_all_markets().await;
        let filtered_markets = self.filter_markets(&all_markets);

        // Limit to max_markets_per_interval
        let markets_to_analyze: Vec<_> = filtered_markets
            .into_iter()
            .take(self.config.max_markets_per_interval)
            .collect();

        info!(
            total_markets = all_markets.len(),
            filtered_markets = markets_to_analyze.len(),
            "Analyzing markets"
        );

        // Process each market
        for market in &markets_to_analyze {
            // Check cache first
            {
                let cache = self.prediction_cache.read().await;
                if let Some(cached) = cache.get(&market.condition_id) {
                    if self.is_cache_valid(cached) {
                        debug!(
                            market_id = %market.condition_id,
                            "Using cached prediction"
                        );

                        // Get current price and check if we should generate signal
                        if let Some(current_price) = state.get_price(&market.yes_token_id).await {
                            if self.should_generate_signal(&cached.prediction, current_price) {
                                let signal =
                                    self.generate_signal(market, &cached.prediction, current_price);
                                signals.push(signal);
                            }
                        }
                        continue;
                    }
                }
            }

            // Analyze market
            match self.analyze_market(market, state).await {
                Ok(Some(prediction)) => {
                    // Cache the prediction
                    {
                        let mut cache = self.prediction_cache.write().await;
                        cache.insert(
                            market.condition_id.clone(),
                            CachedPrediction {
                                prediction: prediction.clone(),
                                timestamp: Utc::now(),
                            },
                        );
                    }

                    // Check if we should generate a signal
                    if let Some(current_price) = state.get_price(&market.yes_token_id).await {
                        if self.should_generate_signal(&prediction, current_price) {
                            let signal = self.generate_signal(market, &prediction, current_price);
                            info!(
                                market_id = %market.condition_id,
                                prediction = %prediction.prediction,
                                confidence = %prediction.confidence,
                                "Generated trade signal"
                            );
                            signals.push(signal);
                        }
                    }
                }
                Ok(None) => {
                    debug!(
                        market_id = %market.condition_id,
                        "No prediction generated"
                    );
                }
                Err(e) => {
                    warn!(
                        market_id = %market.condition_id,
                        error = %e,
                        "Failed to analyze market"
                    );
                }
            }

            // Delay between API calls
            if self.config.api_call_delay_ms > 0 {
                sleep(Duration::from_millis(self.config.api_call_delay_ms)).await;
            }
        }

        info!(signals_generated = signals.len(), "LLM analysis complete");

        Ok(signals)
    }

    async fn initialize(&mut self, _state: &dyn StateProvider) -> Result<(), StrategyError> {
        info!(
            strategy_id = %self.id,
            model = %self.config.model,
            interval_secs = %self.config.analysis_interval_secs,
            confidence_threshold = %self.config.confidence_threshold,
            "Initializing LLM prediction strategy"
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
        let new_config: LlmPredictionConfig = toml::from_str(config_content)
            .map_err(|e| StrategyError::ConfigError(format!("Failed to parse config: {}", e)))?;

        self.config = new_config;
        tracing::info!(strategy_id = %self.id, "Reloaded LLM prediction strategy config");
        Ok(())
    }

    fn config_name(&self) -> &str {
        "llm_prediction"
    }
}

/// Truncate a string to a maximum length
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

/// Generate a random suffix for signal IDs
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

    #[test]
    fn test_config_default() {
        let config = LlmPredictionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.model, "x-ai/grok-3-latest");
        assert_eq!(config.api_key_env, "OPENROUTER_API_KEY");
        assert_eq!(config.temperature, 0.3);
        assert_eq!(config.max_tokens, 1024);
        assert_eq!(config.analysis_interval_secs, 300);
        assert_eq!(config.max_markets_per_interval, 10);
        assert_eq!(config.confidence_threshold, 0.75);
        assert_eq!(config.min_price_edge, dec!(0.05));
        assert_eq!(config.order_size_usd, dec!(50));
        assert_eq!(config.min_liquidity_usd, dec!(5000));
        assert_eq!(config.order_type, "Gtc");
    }

    #[test]
    fn test_parse_llm_response_valid() {
        // We can't easily test this without a strategy instance, but we can test JSON parsing
        let json = r#"{"prediction": "yes", "confidence": 0.85, "reasoning": "Test reasoning", "target_price": 0.65}"#;
        let prediction: LlmPrediction = serde_json::from_str(json).unwrap();
        assert_eq!(prediction.prediction, "yes");
        assert_eq!(prediction.confidence, 0.85);
        assert_eq!(prediction.reasoning, "Test reasoning");
        assert_eq!(prediction.target_price, Some(0.65));
    }

    #[test]
    fn test_parse_llm_response_without_target_price() {
        let json = r#"{"prediction": "no", "confidence": 0.7, "reasoning": "Test"}"#;
        let prediction: LlmPrediction = serde_json::from_str(json).unwrap();
        assert_eq!(prediction.prediction, "no");
        assert_eq!(prediction.target_price, None);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello...");
        assert_eq!(truncate("", 5), "");
    }
}
