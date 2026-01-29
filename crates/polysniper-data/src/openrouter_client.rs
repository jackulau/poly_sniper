//! OpenRouter API client for LLM chat completions

use polysniper_core::DataSourceError;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};

/// Chat completion request for OpenRouter API
#[derive(Debug, Clone, Default, Serialize)]
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

/// Chat message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Response format specification
#[derive(Debug, Clone, Serialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
}

impl ResponseFormat {
    pub fn json() -> Self {
        Self {
            format_type: "json_object".to_string(),
        }
    }
}

/// Chat completion response from OpenRouter API
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

/// A choice in the completion response
#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

/// Response message content
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

/// Token usage statistics
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Configuration for OpenRouter client
#[derive(Debug, Clone)]
pub struct OpenRouterConfig {
    pub api_key: String,
    pub base_url: String,
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

/// OpenRouter API client for LLM chat completions
pub struct OpenRouterClient {
    client: Client,
    config: OpenRouterConfig,
}

impl OpenRouterClient {
    /// Create a new OpenRouter client with the given configuration
    pub fn new(config: OpenRouterConfig) -> Result<Self, DataSourceError> {
        if config.api_key.is_empty() {
            return Err(DataSourceError::AuthError(
                "OpenRouter API key is required".to_string(),
            ));
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| DataSourceError::ConnectionError(e.to_string()))?;

        Ok(Self { client, config })
    }

    /// Send a chat completion request to OpenRouter
    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, DataSourceError> {
        self.chat_completion_with_retry(&request).await
    }

    /// Internal method with retry logic
    async fn chat_completion_with_retry(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, DataSourceError> {
        let mut attempt = 0;
        let mut delay_ms = self.config.retry_delay_ms;

        loop {
            attempt += 1;

            match self.send_request(request).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    let should_retry = match &e {
                        DataSourceError::RateLimited => true,
                        DataSourceError::HttpError(msg) if msg.contains("5") => true,
                        DataSourceError::Timeout(_) => true,
                        _ => false,
                    };

                    if !should_retry || attempt >= self.config.max_retries {
                        return Err(e);
                    }

                    warn!(
                        attempt = attempt,
                        max_retries = self.config.max_retries,
                        delay_ms = delay_ms,
                        error = %e,
                        "OpenRouter request failed, retrying..."
                    );

                    sleep(Duration::from_millis(delay_ms)).await;
                    delay_ms *= 2; // Exponential backoff
                }
            }
        }
    }

    /// Send a single request to OpenRouter
    async fn send_request(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, DataSourceError> {
        let url = format!("{}/chat/completions", self.config.base_url);

        debug!(
            model = %request.model,
            messages_count = request.messages.len(),
            "Sending chat completion request to OpenRouter"
        );

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    DataSourceError::Timeout(e.to_string())
                } else {
                    DataSourceError::HttpError(e.to_string())
                }
            })?;

        let status = response.status();

        if status.as_u16() == 429 {
            return Err(DataSourceError::RateLimited);
        }

        if status.is_server_error() {
            let body = response.text().await.unwrap_or_default();
            return Err(DataSourceError::HttpError(format!(
                "Server error {}: {}",
                status.as_u16(),
                body
            )));
        }

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DataSourceError::HttpError(format!(
                "HTTP {}: {}",
                status.as_u16(),
                body
            )));
        }

        let completion: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| DataSourceError::ParseError(e.to_string()))?;

        debug!(
            id = %completion.id,
            choices = completion.choices.len(),
            "Received chat completion response"
        );

        Ok(completion)
    }

    /// Get the default model from configuration
    pub fn default_model(&self) -> &str {
        &self.config.default_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = OpenRouterConfig::default();
        assert_eq!(config.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(config.default_model, "x-ai/grok-3-latest");
        assert_eq!(config.timeout_secs, 60);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_delay_ms, 1000);
    }

    #[test]
    fn test_chat_message_helpers() {
        let system = ChatMessage::system("You are an assistant");
        assert_eq!(system.role, "system");
        assert_eq!(system.content, "You are an assistant");

        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, "user");
        assert_eq!(user.content, "Hello");

        let assistant = ChatMessage::assistant("Hi there");
        assert_eq!(assistant.role, "assistant");
        assert_eq!(assistant.content, "Hi there");
    }

    #[test]
    fn test_response_format_json() {
        let format = ResponseFormat::json();
        assert_eq!(format.format_type, "json_object");
    }

    #[test]
    fn test_client_requires_api_key() {
        let config = OpenRouterConfig::default();
        let result = OpenRouterClient::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_client_with_api_key() {
        let config = OpenRouterConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        };
        let result = OpenRouterClient::new(config);
        assert!(result.is_ok());
    }
}
