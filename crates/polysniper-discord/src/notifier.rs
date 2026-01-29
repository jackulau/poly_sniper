//! Discord notification service with event bus integration
//!
//! Subscribes to SystemEvents and sends appropriate Discord notifications
//! based on configuration settings.

use crate::embed::{Embed, EmbedBuilder, EmbedColor};
use crate::templates::{ErrorEmbed, RiskEmbed, StatusEmbed};
use crate::types::DiscordMessage;
use crate::{DiscordClientConfig, DiscordError, DiscordWebhookClient};
use polysniper_core::{
    ConnectionState, ConnectionStatusEvent, DiscordConfig, SystemEvent, TradeExecutedEvent,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, Mutex, RwLock};
use tracing::{debug, error, info, warn};

/// Discord notifier service that subscribes to the event bus
/// and sends formatted notifications for configured event types.
pub struct DiscordNotifier {
    client: DiscordWebhookClient,
    config: DiscordConfig,
    rate_limiter: Arc<RwLock<RateLimiter>>,
    pending_messages: Arc<Mutex<VecDeque<DiscordMessage>>>,
    running: Arc<AtomicBool>,
}

/// Rate limiter for Discord messages
struct RateLimiter {
    /// Timestamps of messages sent in the current window
    message_times: VecDeque<Instant>,
    /// Maximum messages per minute
    limit_per_minute: u32,
}

impl RateLimiter {
    fn new(limit_per_minute: u32) -> Self {
        Self {
            message_times: VecDeque::new(),
            limit_per_minute,
        }
    }

    /// Check if we can send a message (and record it if so)
    fn try_send(&mut self) -> bool {
        let now = Instant::now();
        let window = Duration::from_secs(60);

        // Remove old timestamps outside the window
        while let Some(front) = self.message_times.front() {
            if now.duration_since(*front) > window {
                self.message_times.pop_front();
            } else {
                break;
            }
        }

        // Check if we're under the limit
        if self.message_times.len() < self.limit_per_minute as usize {
            self.message_times.push_back(now);
            true
        } else {
            false
        }
    }

    /// Get the number of messages sent in the current window
    fn current_count(&self) -> usize {
        self.message_times.len()
    }
}

impl DiscordNotifier {
    /// Create a new Discord notifier from configuration
    pub fn new(config: DiscordConfig) -> Result<Self, DiscordError> {
        let webhook_url = Self::get_webhook_url(&config)?;

        let client_config = DiscordClientConfig {
            dry_run: config.dry_run,
            ..Default::default()
        };

        let client = DiscordWebhookClient::with_config(webhook_url, client_config)?;

        Ok(Self {
            client,
            config: config.clone(),
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new(config.rate_limit_per_minute))),
            pending_messages: Arc::new(Mutex::new(VecDeque::new())),
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Get webhook URL from config or environment variable
    fn get_webhook_url(config: &DiscordConfig) -> Result<String, DiscordError> {
        // First try config
        if let Some(ref url) = config.webhook_url {
            if !url.is_empty() {
                return Ok(url.clone());
            }
        }

        // Fall back to environment variable
        std::env::var("DISCORD_WEBHOOK_URL").map_err(|_| {
            DiscordError::InvalidWebhookUrl(
                "No webhook URL provided. Set DISCORD_WEBHOOK_URL env var or discord.webhook_url in config".to_string(),
            )
        })
    }

    /// Start the notifier event loop
    pub async fn run(&self, mut event_rx: broadcast::Receiver<SystemEvent>) {
        self.running.store(true, Ordering::SeqCst);
        info!("Discord notifier started");

        // Send startup notification
        if let Err(e) = self.send_startup_notification().await {
            warn!(error = %e, "Failed to send startup notification");
        }

        while self.running.load(Ordering::SeqCst) {
            tokio::select! {
                event = event_rx.recv() => {
                    match event {
                        Ok(event) => {
                            if let Err(e) = self.handle_event(&event).await {
                                error!(error = %e, "Error handling event for Discord notification");
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            warn!(lagged = n, "Discord notifier lagged behind event bus");
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            info!("Event bus closed, shutting down Discord notifier");
                            break;
                        }
                    }
                }
            }
        }

        // Flush pending messages on shutdown
        self.flush_pending().await;

        info!("Discord notifier stopped");
    }

    /// Signal the notifier to stop
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if the notifier is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Handle a system event
    async fn handle_event(&self, event: &SystemEvent) -> Result<(), DiscordError> {
        // Filter based on event type and config
        let embed = match event {
            SystemEvent::TradeExecuted(trade) if self.config.notify_on_trade => {
                Some(self.build_trade_embed(trade))
            }
            SystemEvent::ConnectionStatus(status) if self.config.notify_on_connection_status => {
                Some(self.build_connection_embed(status))
            }
            _ => None,
        };

        if let Some(embed) = embed {
            self.send_embed(embed).await?;
        }

        Ok(())
    }

    /// Build embed for trade executed event
    fn build_trade_embed(&self, trade: &TradeExecutedEvent) -> Embed {
        EmbedBuilder::new()
            .title("Trade Executed")
            .color(EmbedColor::Success)
            .inline_field("Market", truncate_str(&trade.market_id, 40))
            .inline_field("Token", truncate_str(&trade.token_id, 40))
            .inline_field("Price", format!("{:.2}¢", trade.executed_price * rust_decimal::Decimal::from(100)))
            .inline_field("Size", format!("{} contracts", trade.executed_size))
            .inline_field("Fees", format!("${:.4}", trade.fees))
            .inline_field("Order ID", truncate_str(&trade.order_id, 20))
            .footer(format!("Signal: {}", trade.signal.strategy_id))
            .timestamp_now()
            .build()
    }

    /// Build embed for connection status event
    fn build_connection_embed(&self, status: &ConnectionStatusEvent) -> Embed {
        let (title, color) = match status.status {
            ConnectionState::Connected => ("Connected", EmbedColor::Success),
            ConnectionState::Disconnected => ("Disconnected", EmbedColor::Error),
            ConnectionState::Reconnecting => ("Reconnecting", EmbedColor::Warning),
            ConnectionState::Error => ("Connection Error", EmbedColor::Error),
        };

        let mut builder = EmbedBuilder::new()
            .title(title)
            .color(color)
            .inline_field("Service", &status.source);

        if let Some(ref msg) = status.message {
            builder = builder.description(msg);
        }

        builder.timestamp_now().build()
    }

    /// Send an embed notification
    async fn send_embed(&self, embed: Embed) -> Result<(), DiscordError> {
        // Convert our Embed to DiscordMessage with the wire format embed
        let wire_embed = crate::types::DiscordEmbed {
            title: embed.title,
            description: embed.description,
            url: embed.url,
            color: embed.color,
            timestamp: embed.timestamp,
            footer: embed.footer.map(|f| crate::types::DiscordEmbedFooter {
                text: f.text,
                icon_url: f.icon_url,
            }),
            thumbnail: embed.thumbnail.map(|t| crate::types::DiscordEmbedImage { url: t.url }),
            image: embed.image.map(|i| crate::types::DiscordEmbedImage { url: i.url }),
            author: embed.author.map(|a| crate::types::DiscordEmbedAuthor {
                name: a.name,
                url: a.url,
                icon_url: a.icon_url,
            }),
            fields: embed
                .fields
                .into_iter()
                .map(|f| crate::types::DiscordEmbedField {
                    name: f.name,
                    value: f.value,
                    inline: f.inline,
                })
                .collect(),
        };

        let message = DiscordMessage::embed(wire_embed);
        self.send_message(message).await
    }

    /// Send a message with rate limiting
    async fn send_message(&self, message: DiscordMessage) -> Result<(), DiscordError> {
        // Try to acquire rate limit slot
        let can_send = {
            let mut limiter = self.rate_limiter.write().await;
            limiter.try_send()
        };

        if can_send {
            self.client.send_message(&message).await
        } else {
            // Queue the message for later
            let mut pending = self.pending_messages.lock().await;
            if pending.len() < 100 {
                // Cap pending queue
                pending.push_back(message);
                debug!(
                    pending_count = pending.len(),
                    "Message queued due to rate limit"
                );
                Ok(())
            } else {
                warn!("Pending message queue full, dropping message");
                Ok(()) // Don't fail, just drop
            }
        }
    }

    /// Send startup notification
    async fn send_startup_notification(&self) -> Result<(), DiscordError> {
        let embed = StatusEmbed::startup(env!("CARGO_PKG_VERSION"));
        self.send_embed(embed).await
    }

    /// Flush pending messages on shutdown
    async fn flush_pending(&self) {
        let pending: Vec<DiscordMessage> = {
            let mut queue = self.pending_messages.lock().await;
            queue.drain(..).collect()
        };

        if !pending.is_empty() {
            info!(count = pending.len(), "Flushing pending Discord messages");
            for message in pending {
                // Ignore rate limiting during flush
                if let Err(e) = self.client.send_message(&message).await {
                    warn!(error = %e, "Failed to flush pending message");
                }
                // Small delay to avoid hitting rate limits
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }

    /// Send an error notification
    pub async fn notify_error(&self, title: &str, message: &str) -> Result<(), DiscordError> {
        if !self.config.notify_on_error {
            return Ok(());
        }

        let embed = ErrorEmbed::create(title, message);
        self.send_embed(embed).await
    }

    /// Send a critical error notification (bypasses rate limiting)
    pub async fn notify_critical(&self, title: &str, message: &str) -> Result<(), DiscordError> {
        if !self.config.notify_on_error {
            return Ok(());
        }

        let embed = ErrorEmbed::critical(title, message);
        // Convert and send directly, bypassing rate limit for critical alerts
        let wire_embed = crate::types::DiscordEmbed {
            title: embed.title,
            description: embed.description,
            url: embed.url,
            color: embed.color,
            timestamp: embed.timestamp,
            footer: embed.footer.map(|f| crate::types::DiscordEmbedFooter {
                text: f.text,
                icon_url: f.icon_url,
            }),
            thumbnail: embed.thumbnail.map(|t| crate::types::DiscordEmbedImage { url: t.url }),
            image: embed.image.map(|i| crate::types::DiscordEmbedImage { url: i.url }),
            author: embed.author.map(|a| crate::types::DiscordEmbedAuthor {
                name: a.name,
                url: a.url,
                icon_url: a.icon_url,
            }),
            fields: embed
                .fields
                .into_iter()
                .map(|f| crate::types::DiscordEmbedField {
                    name: f.name,
                    value: f.value,
                    inline: f.inline,
                })
                .collect(),
        };
        self.client.send_message(&DiscordMessage::embed(wire_embed)).await
    }

    /// Send a risk alert notification
    pub async fn notify_risk_event(
        &self,
        signal_id: &str,
        reason: &str,
    ) -> Result<(), DiscordError> {
        if !self.config.notify_on_risk_events {
            return Ok(());
        }

        let embed = RiskEmbed::rejected(signal_id, reason);
        self.send_embed(embed).await
    }

    /// Send a circuit breaker notification
    pub async fn notify_circuit_breaker(
        &self,
        reason: &str,
        loss_amount: rust_decimal::Decimal,
    ) -> Result<(), DiscordError> {
        if !self.config.notify_on_risk_events {
            return Ok(());
        }

        let embed = RiskEmbed::circuit_breaker(reason, loss_amount);
        // Circuit breaker bypasses rate limiting
        let wire_embed = crate::types::DiscordEmbed {
            title: embed.title,
            description: embed.description,
            url: embed.url,
            color: embed.color,
            timestamp: embed.timestamp,
            footer: embed.footer.map(|f| crate::types::DiscordEmbedFooter {
                text: f.text,
                icon_url: f.icon_url,
            }),
            thumbnail: embed.thumbnail.map(|t| crate::types::DiscordEmbedImage { url: t.url }),
            image: embed.image.map(|i| crate::types::DiscordEmbedImage { url: i.url }),
            author: embed.author.map(|a| crate::types::DiscordEmbedAuthor {
                name: a.name,
                url: a.url,
                icon_url: a.icon_url,
            }),
            fields: embed
                .fields
                .into_iter()
                .map(|f| crate::types::DiscordEmbedField {
                    name: f.name,
                    value: f.value,
                    inline: f.inline,
                })
                .collect(),
        };
        self.client.send_message(&DiscordMessage::embed(wire_embed)).await
    }

    /// Get current rate limiter status
    pub async fn rate_limit_status(&self) -> (usize, u32) {
        let limiter = self.rate_limiter.read().await;
        (limiter.current_count(), limiter.limit_per_minute)
    }
}

/// Truncate a string to a maximum length, adding ellipsis if needed
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use polysniper_core::{OrderType, Outcome, Priority, Side, TradeSignal};
    use rust_decimal_macros::dec;

    fn test_config() -> DiscordConfig {
        DiscordConfig {
            enabled: true,
            webhook_url: Some("https://discord.com/api/webhooks/123456/abcdef".to_string()),
            notify_on_trade: true,
            notify_on_error: true,
            notify_on_risk_events: true,
            notify_on_connection_status: true,
            dry_run: true,
            rate_limit_per_minute: 30,
        }
    }

    fn test_trade_signal() -> TradeSignal {
        TradeSignal {
            id: "sig_123".to_string(),
            strategy_id: "test_strategy".to_string(),
            market_id: "market_123".to_string(),
            token_id: "token_yes".to_string(),
            outcome: Outcome::Yes,
            side: Side::Buy,
            price: Some(dec!(0.65)),
            size: dec!(100),
            size_usd: dec!(65.00),
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: Utc::now(),
            reason: "Test signal".to_string(),
            metadata: serde_json::json!({}),
        }
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(5);

        // Should allow first 5 messages
        for _ in 0..5 {
            assert!(limiter.try_send());
        }

        // 6th should be blocked
        assert!(!limiter.try_send());
    }

    #[test]
    fn test_truncate_str() {
        assert_eq!(truncate_str("short", 10), "short");
        assert_eq!(truncate_str("this is long", 10), "this is...");
        assert_eq!(truncate_str("exactly", 7), "exactly");
    }

    #[tokio::test]
    async fn test_notifier_creation() {
        let config = test_config();
        let notifier = DiscordNotifier::new(config);
        assert!(notifier.is_ok());
    }

    #[tokio::test]
    async fn test_notifier_dry_run() {
        let config = test_config();
        let notifier = DiscordNotifier::new(config).unwrap();

        // Should succeed in dry run mode
        let result = notifier.notify_error("Test", "Test message").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limit_status() {
        let config = test_config();
        let notifier = DiscordNotifier::new(config).unwrap();

        let (count, limit) = notifier.rate_limit_status().await;
        assert_eq!(count, 0);
        assert_eq!(limit, 30);
    }

    #[tokio::test]
    async fn test_build_trade_embed() {
        let config = test_config();
        let notifier = DiscordNotifier::new(config).unwrap();

        let signal = test_trade_signal();
        let trade = TradeExecutedEvent {
            order_id: "order_123".to_string(),
            signal,
            market_id: "market_123".to_string(),
            token_id: "token_yes".to_string(),
            executed_price: dec!(0.65),
            executed_size: dec!(100),
            fees: dec!(0.01),
            timestamp: Utc::now(),
        };

        let embed = notifier.build_trade_embed(&trade);
        assert!(embed.title.is_some());
        assert!(embed.title.unwrap().contains("Trade"));
    }

    #[tokio::test]
    async fn test_build_connection_embed() {
        let config = test_config();
        let notifier = DiscordNotifier::new(config).unwrap();

        let status = ConnectionStatusEvent {
            source: "WebSocket".to_string(),
            status: ConnectionState::Connected,
            message: Some("Successfully connected".to_string()),
            timestamp: Utc::now(),
        };

        let embed = notifier.build_connection_embed(&status);
        assert!(embed.title.is_some());
        assert!(embed.title.unwrap().contains("Connected"));
    }

    #[test]
    fn test_config_notify_flags() {
        let mut config = test_config();
        config.notify_on_trade = false;
        config.notify_on_error = false;
        config.notify_on_risk_events = false;
        config.notify_on_connection_status = false;

        // All notifications should be disabled
        assert!(!config.notify_on_trade);
        assert!(!config.notify_on_error);
        assert!(!config.notify_on_risk_events);
        assert!(!config.notify_on_connection_status);
    }
}
