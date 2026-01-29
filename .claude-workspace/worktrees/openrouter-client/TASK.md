---
id: openrouter-client
name: OpenRouter API Client
wave: 1
priority: 1
dependencies: []
estimated_hours: 3
tags: [backend, data, http-client]
---

## Objective

Implement an HTTP client for OpenRouter API that handles chat completions with retry logic and rate limiting.

## Context

This client will be used by the LLM prediction strategy to send market analysis prompts to LLM models (defaulting to Grok). It follows the existing pattern established by `GammaClient` in the data crate, but adds retry logic similar to `OrderSubmitter`.

## Implementation

### 1. Create `crates/polysniper-data/src/openrouter_client.rs`

**Request/Response Structs:**
```rust
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,  // "system" | "user" | "assistant"
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,  // "json_object"
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
```

**Configuration:**
```rust
#[derive(Debug, Clone)]
pub struct OpenRouterConfig {
    pub api_key: String,
    pub base_url: String,  // https://openrouter.ai/api/v1
    pub default_model: String,
    pub timeout_secs: u64,
    pub max_retries: u32,
    pub retry_delay_ms: u64,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            default_model: "x-ai/grok-3-latest".to_string(),
            timeout_secs: 60,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }
}
```

**Client Implementation:**
```rust
pub struct OpenRouterClient {
    client: reqwest::Client,
    config: OpenRouterConfig,
}

impl OpenRouterClient {
    pub fn new(config: OpenRouterConfig) -> Result<Self, DataSourceError>;

    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, DataSourceError>;

    async fn chat_completion_with_retry(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, DataSourceError>;
}
```

**Error Handling:**
- Use existing `DataSourceError` from polysniper-core
- Add specific handling for:
  - HTTP 429 (rate limited) - exponential backoff
  - HTTP 5xx (server errors) - retry with delay
  - HTTP 4xx (client errors) - fail immediately (except 429)
  - Timeout - retry with delay

**Retry Logic:**
- Exponential backoff: `delay * 2^attempt`
- Max retries configurable (default: 3)
- Log each retry attempt with tracing

### 2. Update `crates/polysniper-data/src/lib.rs`

Add module declaration and re-exports:
```rust
pub mod openrouter_client;

pub use openrouter_client::{
    OpenRouterClient, OpenRouterConfig,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, ResponseFormat,
};
```

### 3. Update `crates/polysniper-data/Cargo.toml`

Ensure dependencies are present (should already have reqwest, serde, tokio).

## Acceptance Criteria

- [ ] `OpenRouterClient` struct with configuration
- [ ] `chat_completion()` method sends requests to OpenRouter API
- [ ] Request serialization matches OpenRouter API spec
- [ ] Response deserialization handles all fields
- [ ] Retry logic with exponential backoff for 429/5xx errors
- [ ] Rate limit (429) detection and handling
- [ ] Proper error mapping to `DataSourceError`
- [ ] API key passed via Authorization header
- [ ] Timeout configuration (default 60s for LLM responses)
- [ ] Module exported from lib.rs
- [ ] Code compiles without errors: `cargo build -p polysniper-data`

## Files to Create/Modify

- `crates/polysniper-data/src/openrouter_client.rs` - **Create** - Main client implementation
- `crates/polysniper-data/src/lib.rs` - **Modify** - Add module and re-exports

## Integration Points

- **Provides**: `OpenRouterClient` for LLM API calls
- **Consumes**: Nothing (standalone HTTP client)
- **Conflicts**: None - new file with no overlap

## Testing Notes

For manual testing, set `OPENROUTER_API_KEY` and use a simple test request:
```rust
let config = OpenRouterConfig {
    api_key: std::env::var("OPENROUTER_API_KEY").unwrap(),
    ..Default::default()
};
let client = OpenRouterClient::new(config)?;
let response = client.chat_completion(ChatCompletionRequest {
    model: "x-ai/grok-3-latest".to_string(),
    messages: vec![ChatMessage {
        role: "user".to_string(),
        content: "Say hello".to_string(),
    }],
    ..Default::default()
}).await?;
```

## Reference

- OpenRouter API docs: https://openrouter.ai/docs
- Existing pattern: `crates/polysniper-data/src/gamma_client.rs`
- Retry pattern: `crates/polysniper-execution/src/submitter.rs`
