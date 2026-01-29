//! Discord webhook error types

use thiserror::Error;

/// Discord webhook errors
#[derive(Error, Debug)]
pub enum DiscordError {
    #[error("HTTP request failed: {0}")]
    HttpError(String),

    #[error("Rate limited, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },

    #[error("Invalid webhook URL: {0}")]
    InvalidWebhookUrl(String),

    #[error("Request serialization failed: {0}")]
    SerializationError(String),

    #[error("Response parse error: {0}")]
    ParseError(String),

    #[error("Retry exhausted after {attempts} attempts: {message}")]
    RetryExhausted { attempts: u32, message: String },

    #[error("Webhook returned error: {status} - {message}")]
    WebhookError { status: u16, message: String },

    #[error("Request timeout")]
    Timeout,
}

impl From<reqwest::Error> for DiscordError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            DiscordError::Timeout
        } else {
            DiscordError::HttpError(err.to_string())
        }
    }
}

impl From<serde_json::Error> for DiscordError {
    fn from(err: serde_json::Error) -> Self {
        DiscordError::SerializationError(err.to_string())
    }
}
