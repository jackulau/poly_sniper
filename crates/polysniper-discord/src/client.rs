//! Discord webhook HTTP client with retry and rate limit handling

use crate::error::DiscordError;
use crate::types::{DiscordMessage, WebhookResponse};
use reqwest::Client;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_MAX_RETRIES: u32 = 3;
const DEFAULT_RETRY_DELAY_MS: u64 = 1000;

/// Configuration for the Discord webhook client
#[derive(Debug, Clone)]
pub struct DiscordClientConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,

    /// Base delay between retries in milliseconds
    pub retry_delay_ms: u64,

    /// HTTP request timeout
    pub timeout: Duration,

    /// Whether to run in dry-run mode (no actual requests)
    pub dry_run: bool,
}

impl Default for DiscordClientConfig {
    fn default() -> Self {
        Self {
            max_retries: DEFAULT_MAX_RETRIES,
            retry_delay_ms: DEFAULT_RETRY_DELAY_MS,
            timeout: DEFAULT_TIMEOUT,
            dry_run: false,
        }
    }
}

/// Discord webhook client for sending messages
pub struct DiscordWebhookClient {
    webhook_url: String,
    client: Client,
    config: DiscordClientConfig,
    dry_run: Arc<AtomicBool>,
}

impl DiscordWebhookClient {
    /// Create a new Discord webhook client
    pub fn new(webhook_url: impl Into<String>) -> Result<Self, DiscordError> {
        Self::with_config(webhook_url, DiscordClientConfig::default())
    }

    /// Create a new Discord webhook client with custom configuration
    pub fn with_config(
        webhook_url: impl Into<String>,
        config: DiscordClientConfig,
    ) -> Result<Self, DiscordError> {
        let webhook_url = webhook_url.into();

        // Validate webhook URL format
        if !webhook_url.starts_with("https://discord.com/api/webhooks/")
            && !webhook_url.starts_with("https://discordapp.com/api/webhooks/")
        {
            return Err(DiscordError::InvalidWebhookUrl(
                "URL must start with https://discord.com/api/webhooks/ or https://discordapp.com/api/webhooks/".to_string(),
            ));
        }

        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| DiscordError::HttpError(e.to_string()))?;

        let dry_run = Arc::new(AtomicBool::new(config.dry_run));

        Ok(Self {
            webhook_url,
            client,
            config,
            dry_run,
        })
    }

    /// Set dry-run mode
    pub fn set_dry_run(&self, dry_run: bool) {
        self.dry_run.store(dry_run, Ordering::SeqCst);
    }

    /// Check if in dry-run mode
    pub fn is_dry_run(&self) -> bool {
        self.dry_run.load(Ordering::SeqCst)
    }

    /// Send a message to the Discord webhook
    pub async fn send_message(&self, message: &DiscordMessage) -> Result<(), DiscordError> {
        if self.dry_run.load(Ordering::SeqCst) {
            let json = serde_json::to_string_pretty(message)?;
            info!(
                webhook_url = %self.webhook_url,
                "[DRY RUN] Would send Discord message:\n{}",
                json
            );
            return Ok(());
        }

        self.send_with_retry(message).await
    }

    /// Send message with retry logic
    async fn send_with_retry(&self, message: &DiscordMessage) -> Result<(), DiscordError> {
        let mut attempts = 0;

        loop {
            attempts += 1;

            match self.send_once(message).await {
                Ok(()) => {
                    if attempts > 1 {
                        info!(
                            attempts = attempts,
                            "Discord message sent successfully after retries"
                        );
                    } else {
                        debug!("Discord message sent successfully");
                    }
                    return Ok(());
                }
                Err(e) => {
                    // Handle rate limiting specially
                    if let DiscordError::RateLimited { retry_after_ms } = &e {
                        if attempts < self.config.max_retries {
                            warn!(
                                retry_after_ms = retry_after_ms,
                                attempt = attempts,
                                "Rate limited by Discord, waiting before retry"
                            );
                            sleep(Duration::from_millis(*retry_after_ms)).await;
                            continue;
                        }
                    }

                    if attempts >= self.config.max_retries {
                        error!(
                            attempts = attempts,
                            error = %e,
                            "Discord message send failed after max retries"
                        );
                        return Err(DiscordError::RetryExhausted {
                            attempts,
                            message: e.to_string(),
                        });
                    }

                    warn!(
                        attempt = attempts,
                        error = %e,
                        "Discord message send failed, retrying..."
                    );

                    // Exponential backoff
                    let delay = self.config.retry_delay_ms * 2u64.pow(attempts - 1);
                    sleep(Duration::from_millis(delay)).await;
                }
            }
        }
    }

    /// Send a single message attempt
    async fn send_once(&self, message: &DiscordMessage) -> Result<(), DiscordError> {
        debug!(webhook_url = %self.webhook_url, "Sending Discord webhook message");

        let response = self
            .client
            .post(&self.webhook_url)
            .json(message)
            .send()
            .await?;

        let status = response.status();

        // Handle rate limiting (429)
        if status.as_u16() == 429 {
            let body: WebhookResponse = response.json().await.unwrap_or(WebhookResponse {
                id: None,
                code: None,
                message: Some("Rate limited".to_string()),
                retry_after: Some(5000.0),
            });

            let retry_after_ms = body
                .retry_after
                .map(|r| (r * 1000.0) as u64)
                .unwrap_or(5000);

            return Err(DiscordError::RateLimited { retry_after_ms });
        }

        // Success - 204 No Content is the normal response
        if status.is_success() {
            return Ok(());
        }

        // Handle other errors
        let body = response.text().await.unwrap_or_default();
        Err(DiscordError::WebhookError {
            status: status.as_u16(),
            message: body,
        })
    }

    /// Send a simple text message
    pub async fn send_text(&self, content: impl Into<String>) -> Result<(), DiscordError> {
        let message = DiscordMessage::text(content);
        self.send_message(&message).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_webhook_url() {
        let result = DiscordWebhookClient::new("https://example.com/webhook");
        assert!(result.is_err());
        if let Err(DiscordError::InvalidWebhookUrl(_)) = result {
            // Expected
        } else {
            panic!("Expected InvalidWebhookUrl error");
        }
    }

    #[test]
    fn test_valid_webhook_url() {
        let result = DiscordWebhookClient::new("https://discord.com/api/webhooks/123456/abcdef");
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_discordapp_url() {
        let result = DiscordWebhookClient::new("https://discordapp.com/api/webhooks/123456/abcdef");
        assert!(result.is_ok());
    }

    #[test]
    fn test_dry_run_mode() {
        let client =
            DiscordWebhookClient::new("https://discord.com/api/webhooks/123456/abcdef").unwrap();
        assert!(!client.is_dry_run());

        client.set_dry_run(true);
        assert!(client.is_dry_run());

        client.set_dry_run(false);
        assert!(!client.is_dry_run());
    }

    #[test]
    fn test_config_with_dry_run() {
        let config = DiscordClientConfig {
            dry_run: true,
            ..Default::default()
        };

        let client = DiscordWebhookClient::with_config(
            "https://discord.com/api/webhooks/123456/abcdef",
            config,
        )
        .unwrap();

        assert!(client.is_dry_run());
    }

    #[tokio::test]
    async fn test_dry_run_send() {
        let config = DiscordClientConfig {
            dry_run: true,
            ..Default::default()
        };

        let client = DiscordWebhookClient::with_config(
            "https://discord.com/api/webhooks/123456/abcdef",
            config,
        )
        .unwrap();

        // Should succeed without making real HTTP request
        let result = client.send_text("Test message").await;
        assert!(result.is_ok());
    }
}
