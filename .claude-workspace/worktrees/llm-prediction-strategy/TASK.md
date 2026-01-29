---
id: llm-prediction-strategy
name: LLM Prediction Strategy Implementation
wave: 2
priority: 1
dependencies: [openrouter-client]
estimated_hours: 5
tags: [backend, strategy, llm]
---

## Objective

Implement the LLM prediction strategy that analyzes Polymarket markets using an LLM and generates trade signals based on confidence scores.

## Context

This strategy uses the OpenRouter client (from `openrouter-client` task) to send market analysis prompts to Grok/other LLMs. It triggers on Heartbeat events (timer-based), analyzes markets periodically, and generates trade signals when confidence exceeds the threshold and sufficient price edge exists.

## Implementation

### 1. Create `crates/polysniper-strategies/src/llm_prediction.rs`

**Configuration Struct:**
```rust
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmPredictionConfig {
    pub enabled: bool,
    pub model: String,
    pub api_key_env: String,
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    pub analysis_interval_secs: u64,
    pub max_markets_per_interval: usize,
    #[serde(default = "default_api_delay")]
    pub api_call_delay_ms: u64,
    pub confidence_threshold: f64,
    pub min_price_edge: Decimal,
    pub order_size_usd: Decimal,
    pub min_liquidity_usd: Decimal,
    #[serde(default = "default_order_type")]
    pub order_type: String,
    #[serde(default)]
    pub markets: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub system_prompt_override: Option<String>,
}

fn default_max_tokens() -> u32 { 1024 }
fn default_api_delay() -> u64 { 500 }
fn default_order_type() -> String { "Gtc".to_string() }

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
            min_price_edge: Decimal::new(5, 2),  // 0.05
            order_size_usd: Decimal::new(50, 0),
            min_liquidity_usd: Decimal::new(5000, 0),
            order_type: "Gtc".to_string(),
            markets: Vec::new(),
            tags: Vec::new(),
            keywords: Vec::new(),
            system_prompt_override: None,
        }
    }
}
```

**LLM Response Struct:**
```rust
#[derive(Debug, Clone, Deserialize)]
pub struct LlmPrediction {
    pub prediction: String,  // "yes" | "no" | "hold"
    pub confidence: f64,     // 0.0 - 1.0
    pub reasoning: String,
    #[serde(default)]
    pub target_price: Option<f64>,
}
```

**Prediction Cache:**
```rust
struct CachedPrediction {
    prediction: LlmPrediction,
    timestamp: DateTime<Utc>,
}
```

**Strategy Implementation:**
```rust
pub struct LlmPredictionStrategy {
    id: String,
    config: LlmPredictionConfig,
    enabled: Arc<AtomicBool>,
    openrouter_client: OpenRouterClient,
    prediction_cache: Arc<RwLock<HashMap<String, CachedPrediction>>>,
    last_analysis_time: Arc<RwLock<DateTime<Utc>>>,
}

impl LlmPredictionStrategy {
    pub fn new(config: LlmPredictionConfig) -> Result<Self, StrategyError>;

    fn build_system_prompt(&self) -> String;
    fn build_user_prompt(&self, market: &Market, price: Decimal) -> String;
    fn parse_llm_response(&self, content: &str) -> Result<LlmPrediction, StrategyError>;
    fn should_generate_signal(&self, prediction: &LlmPrediction, current_price: Decimal) -> bool;
    fn filter_markets(&self, markets: &[Market]) -> Vec<Market>;

    async fn analyze_market(
        &self,
        market: &Market,
        state: &dyn StateProvider,
    ) -> Result<Option<LlmPrediction>, StrategyError>;
}
```

**System Prompt (default):**
```text
You are a prediction market analyst. Analyze the market and provide your prediction.

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

Be concise. Focus on key factors that will determine the outcome.
```

**User Prompt Template:**
```text
Market: {question}

Description: {description}

Current YES price: {yes_price} (implies {yes_price*100}% probability)
Current NO price: {no_price}

24h Volume: ${volume}
Liquidity: ${liquidity}
Time until resolution: {time_remaining}

Recent price trend: {trend_direction} ({price_change}% in last 24h)

Analyze this market and provide your prediction.
```

**Strategy Trait Implementation:**
```rust
#[async_trait]
impl Strategy for LlmPredictionStrategy {
    fn id(&self) -> &str { &self.id }
    fn name(&self) -> &str { "LLM Prediction Strategy" }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(event, SystemEvent::Heartbeat(_))
    }

    async fn process_event(
        &self,
        event: &SystemEvent,
        state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        // 1. Check if enough time has passed since last analysis
        // 2. Get all markets from state
        // 3. Filter markets by config (tags, keywords, specific IDs)
        // 4. Limit to max_markets_per_interval
        // 5. For each market:
        //    a. Check cache for recent prediction
        //    b. If not cached, call LLM
        //    c. Parse response
        //    d. Cache prediction
        //    e. If confidence >= threshold AND price edge >= min_edge:
        //       Generate TradeSignal
        // 6. Return all generated signals
    }

    async fn initialize(&mut self, _state: &dyn StateProvider) -> Result<(), StrategyError> {
        info!("Initializing LLM prediction strategy");
        Ok(())
    }

    fn is_enabled(&self) -> bool { self.enabled.load(Ordering::SeqCst) }
    fn set_enabled(&mut self, enabled: bool) { self.enabled.store(enabled, Ordering::SeqCst); }
}
```

**Signal Generation Logic:**
```rust
fn generate_signal(
    &self,
    market: &Market,
    prediction: &LlmPrediction,
    current_price: Decimal,
) -> TradeSignal {
    let (side, outcome, price) = match prediction.prediction.as_str() {
        "yes" => (Side::Buy, Outcome::Yes, current_price),
        "no" => (Side::Buy, Outcome::No, Decimal::ONE - current_price),
        _ => unreachable!(), // "hold" filtered out earlier
    };

    let size = self.config.order_size_usd / price;

    TradeSignal {
        id: format!("sig_llm_{}_{}", market.condition_id, Utc::now().timestamp_millis()),
        strategy_id: self.id.clone(),
        market_id: market.condition_id.clone(),
        token_id: market.get_token_id(outcome),
        outcome,
        side,
        price: Some(price),
        size,
        size_usd: self.config.order_size_usd,
        order_type: OrderType::from_str(&self.config.order_type),
        priority: Priority::Normal,
        timestamp: Utc::now(),
        reason: format!(
            "LLM predicts {} with {:.0}% confidence. Reasoning: {}",
            prediction.prediction,
            prediction.confidence * 100.0,
            prediction.reasoning
        ),
        metadata: serde_json::json!({
            "llm_model": self.config.model,
            "confidence": prediction.confidence,
            "target_price": prediction.target_price,
        }),
    }
}
```

### 2. Update `crates/polysniper-strategies/src/lib.rs`

Add module and re-exports:
```rust
pub mod llm_prediction;

pub use llm_prediction::{LlmPredictionConfig, LlmPredictionStrategy};
```

### 3. Update `crates/polysniper-strategies/Cargo.toml`

Add dependency on polysniper-data (for OpenRouterClient):
```toml
[dependencies]
polysniper-data = { path = "../polysniper-data" }
```

## Acceptance Criteria

- [ ] `LlmPredictionConfig` deserializes from TOML config
- [ ] `LlmPredictionStrategy` implements `Strategy` trait
- [ ] Strategy accepts only `SystemEvent::Heartbeat` events
- [ ] System and user prompts constructed correctly with market data
- [ ] LLM JSON response parsed into `LlmPrediction` struct
- [ ] Prediction caching prevents redundant API calls
- [ ] Analysis interval respected (no spam on every heartbeat)
- [ ] Market filtering works (tags, keywords, specific IDs)
- [ ] `max_markets_per_interval` limit enforced
- [ ] Signals generated only when confidence >= threshold AND edge >= min_edge
- [ ] TradeSignal includes LLM reasoning and metadata
- [ ] Module exported from strategies lib.rs
- [ ] Code compiles: `cargo build -p polysniper-strategies`

## Files to Create/Modify

- `crates/polysniper-strategies/src/llm_prediction.rs` - **Create** - Strategy implementation
- `crates/polysniper-strategies/src/lib.rs` - **Modify** - Add module and re-exports
- `crates/polysniper-strategies/Cargo.toml` - **Modify** - Add polysniper-data dependency

## Integration Points

- **Provides**: `LlmPredictionStrategy` for main.rs to load
- **Consumes**: `OpenRouterClient` from polysniper-data, `StateProvider` for market data
- **Conflicts**: Avoid modifying existing strategy files

## Testing Notes

For manual testing:
1. Set `OPENROUTER_API_KEY` environment variable
2. Enable strategy in config with low thresholds for testing
3. Run with `dry_run = true` in execution config
4. Watch logs for "LLM analysis" and "Generated signal" messages
