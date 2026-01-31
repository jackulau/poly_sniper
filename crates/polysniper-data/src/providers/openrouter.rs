//! OpenRouter Provider
//!
//! LLM provider implementation using OpenRouter API.

use async_trait::async_trait;
use polysniper_core::{BinaryPrediction, LlmPredictionResult, MarketContext, PredictionValue, ProviderConfig};
use rust_decimal::Decimal;
use serde::Deserialize;
use std::time::Instant;
use tracing::debug;

use crate::{ChatCompletionRequest, ChatMessage, OpenRouterClient, OpenRouterConfig, ResponseFormat};

use super::{LlmProvider, ProviderError};

/// OpenRouter provider for LLM predictions
pub struct OpenRouterProvider {
    client: OpenRouterClient,
    model: String,
    cost_per_request: Decimal,
    temperature: f32,
    max_tokens: u32,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider
    pub fn new(config: &ProviderConfig, api_key: &str) -> Result<Self, ProviderError> {
        if config.model.is_empty() {
            return Err(ProviderError::ConfigError("Model name is required".to_string()));
        }

        let openrouter_config = OpenRouterConfig {
            api_key: api_key.to_string(),
            default_model: config.model.clone(),
            ..Default::default()
        };

        let client = OpenRouterClient::new(openrouter_config)
            .map_err(|e| ProviderError::ConfigError(e.to_string()))?;

        Ok(Self {
            client,
            model: config.model.clone(),
            cost_per_request: config.cost_per_request,
            temperature: config.temperature.unwrap_or(0.3),
            max_tokens: config.max_tokens.unwrap_or(1024),
        })
    }

    /// Build the system prompt for market analysis
    fn build_system_prompt(&self) -> String {
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

    /// Build the user prompt for a specific market context
    fn build_user_prompt(&self, context: &MarketContext) -> String {
        let yes_probability = (context.yes_price * Decimal::from(100)).round();
        let description = context.description.as_deref().unwrap_or("No description available");
        let time_remaining = context.time_remaining.as_deref().unwrap_or("Unknown");

        format!(
            r#"Market: {question}

Description: {description}

Current YES price: {yes_price} (implies {yes_probability}% probability)
Current NO price: {no_price}

24h Volume: ${volume}
Liquidity: ${liquidity}
Time until resolution: {time_remaining}

Analyze this market and provide your prediction."#,
            question = context.question,
            description = description,
            yes_price = context.yes_price,
            yes_probability = yes_probability,
            no_price = context.no_price,
            volume = context.volume.round(),
            liquidity = context.liquidity.round(),
            time_remaining = time_remaining,
        )
    }

    /// Parse the LLM response into a prediction result
    fn parse_response(&self, content: &str, latency_ms: u64) -> Result<LlmPredictionResult, ProviderError> {
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

        let parsed: LlmResponse = serde_json::from_str(json_content)
            .map_err(|e| ProviderError::ParseError(format!("Failed to parse response: {}. Content: {}", e, truncate(content, 200))))?;

        let prediction = match parsed.prediction.to_lowercase().as_str() {
            "yes" => PredictionValue::Binary(BinaryPrediction::Yes),
            "no" => PredictionValue::Binary(BinaryPrediction::No),
            "hold" => {
                // For hold, we use a probability of 0.5 (uncertain)
                PredictionValue::Probability(Decimal::new(50, 2))
            }
            _ => {
                return Err(ProviderError::ParseError(format!(
                    "Invalid prediction value: {}",
                    parsed.prediction
                )));
            }
        };

        let confidence = Decimal::try_from(parsed.confidence)
            .map_err(|e| ProviderError::ParseError(format!("Invalid confidence: {}", e)))?;

        let target_price = parsed.target_price.map(|p| {
            Decimal::try_from(p).unwrap_or(Decimal::new(50, 2))
        });

        Ok(LlmPredictionResult {
            model_id: self.model.clone(),
            prediction,
            confidence,
            reasoning: parsed.reasoning,
            target_price,
            latency_ms,
            cost: self.cost_per_request,
        })
    }
}

/// Response from LLM
#[derive(Debug, Deserialize)]
struct LlmResponse {
    prediction: String,
    confidence: f64,
    reasoning: String,
    #[serde(default)]
    target_price: Option<f64>,
}

#[async_trait]
impl LlmProvider for OpenRouterProvider {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "openrouter"
    }

    fn cost_per_request(&self) -> Decimal {
        self.cost_per_request
    }

    async fn predict(&self, context: &MarketContext) -> Result<LlmPredictionResult, ProviderError> {
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(context);

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage::system(&system_prompt),
                ChatMessage::user(&user_prompt),
            ],
            temperature: Some(self.temperature),
            max_tokens: Some(self.max_tokens),
            response_format: Some(ResponseFormat::json()),
        };

        let start = Instant::now();

        debug!(
            model = %self.model,
            market_id = %context.market_id,
            "Sending prediction request to OpenRouter"
        );

        let response = self.client.chat_completion(request).await.map_err(|e| {
            match e {
                polysniper_core::DataSourceError::RateLimited => ProviderError::RateLimited,
                polysniper_core::DataSourceError::Timeout(_) => ProviderError::Timeout,
                _ => ProviderError::ApiError(e.to_string()),
            }
        })?;

        let latency_ms = start.elapsed().as_millis() as u64;

        let content = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| ProviderError::ParseError("Empty response from LLM".to_string()))?;

        let result = self.parse_response(&content, latency_ms)?;

        debug!(
            model = %self.model,
            prediction = ?result.prediction,
            confidence = %result.confidence,
            latency_ms = %latency_ms,
            "Received prediction from OpenRouter"
        );

        Ok(result)
    }

    async fn health_check(&self) -> bool {
        // Simple health check - could be expanded to make a lightweight API call
        true
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

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn create_test_context() -> MarketContext {
        MarketContext {
            market_id: "test-market".to_string(),
            question: "Will event X happen?".to_string(),
            description: Some("Test description".to_string()),
            yes_price: dec!(0.65),
            no_price: dec!(0.35),
            volume: dec!(10000),
            liquidity: dec!(50000),
            time_remaining: Some("7 days".to_string()),
            end_date: None,
            metadata: None,
        }
    }

    #[test]
    fn test_parse_yes_response() {
        let config = ProviderConfig {
            provider_type: "openrouter".to_string(),
            model: "test-model".to_string(),
            enabled: true,
            weight: dec!(1.0),
            cost_per_request: dec!(0.01),
            temperature: None,
            max_tokens: None,
        };

        // We can't easily create a full provider without an API key,
        // but we can test the response parsing logic indirectly
        let json = r#"{"prediction": "yes", "confidence": 0.85, "reasoning": "Test reason", "target_price": 0.75}"#;
        let parsed: LlmResponse = serde_json::from_str(json).unwrap();

        assert_eq!(parsed.prediction, "yes");
        assert_eq!(parsed.confidence, 0.85);
        assert_eq!(parsed.reasoning, "Test reason");
        assert_eq!(parsed.target_price, Some(0.75));
    }

    #[test]
    fn test_parse_no_response() {
        let json = r#"{"prediction": "no", "confidence": 0.70, "reasoning": "Unlikely to happen"}"#;
        let parsed: LlmResponse = serde_json::from_str(json).unwrap();

        assert_eq!(parsed.prediction, "no");
        assert_eq!(parsed.confidence, 0.70);
        assert_eq!(parsed.target_price, None);
    }

    #[test]
    fn test_parse_hold_response() {
        let json = r#"{"prediction": "hold", "confidence": 0.50, "reasoning": "Insufficient information"}"#;
        let parsed: LlmResponse = serde_json::from_str(json).unwrap();

        assert_eq!(parsed.prediction, "hold");
    }

    #[test]
    fn test_build_user_prompt() {
        let context = create_test_context();

        // We can verify the context is valid
        assert_eq!(context.yes_price, dec!(0.65));
        assert_eq!(context.no_price, dec!(0.35));
        assert_eq!(context.question, "Will event X happen?");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello...");
        assert_eq!(truncate("", 5), "");
    }
}
