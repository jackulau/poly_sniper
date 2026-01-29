//! Alerting module for Slack and Telegram notifications

use crate::metrics::{record_alert_failure, record_alert_sent};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Alerting errors
#[derive(Debug, Error)]
pub enum AlertError {
    #[error("HTTP request failed: {0}")]
    Http(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Channel not configured: {0}")]
    NotConfigured(String),

    #[error("Rate limited")]
    RateLimited,
}

/// Alert severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
}

impl std::fmt::Display for AlertLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// Alert message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub level: AlertLevel,
    pub category: String,
    pub title: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

impl Alert {
    pub fn new(level: AlertLevel, category: &str, title: &str, message: &str) -> Self {
        Self {
            level,
            category: category.to_string(),
            title: title.to_string(),
            message: message.to_string(),
            timestamp: Utc::now(),
            metadata: None,
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn info(category: &str, title: &str, message: &str) -> Self {
        Self::new(AlertLevel::Info, category, title, message)
    }

    pub fn warning(category: &str, title: &str, message: &str) -> Self {
        Self::new(AlertLevel::Warning, category, title, message)
    }

    pub fn critical(category: &str, title: &str, message: &str) -> Self {
        Self::new(AlertLevel::Critical, category, title, message)
    }
}

/// Alert channel trait
pub trait AlertChannel: Send + Sync {
    fn name(&self) -> &str;
    fn send(&self, alert: &Alert) -> Result<(), AlertError>;
    fn is_enabled(&self) -> bool;
}

/// Slack configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackConfig {
    pub enabled: bool,
    pub webhook_url: String,
    pub channel: Option<String>,
    pub username: Option<String>,
    pub icon_emoji: Option<String>,
}

impl Default for SlackConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            webhook_url: String::new(),
            channel: None,
            username: Some("Polysniper".to_string()),
            icon_emoji: Some(":chart_with_upwards_trend:".to_string()),
        }
    }
}

/// Slack channel implementation
pub struct SlackChannel {
    config: SlackConfig,
}

impl SlackChannel {
    pub fn new(config: SlackConfig) -> Self {
        Self { config }
    }

    fn level_to_color(&self, level: AlertLevel) -> &str {
        match level {
            AlertLevel::Info => "#36a64f",     // green
            AlertLevel::Warning => "#ff9800",  // orange
            AlertLevel::Critical => "#ff0000", // red
        }
    }

    fn level_to_emoji(&self, level: AlertLevel) -> &str {
        match level {
            AlertLevel::Info => ":information_source:",
            AlertLevel::Warning => ":warning:",
            AlertLevel::Critical => ":rotating_light:",
        }
    }
}

impl AlertChannel for SlackChannel {
    fn name(&self) -> &str {
        "slack"
    }

    fn send(&self, alert: &Alert) -> Result<(), AlertError> {
        if !self.is_enabled() {
            return Err(AlertError::NotConfigured("slack".to_string()));
        }

        let payload = serde_json::json!({
            "channel": self.config.channel,
            "username": self.config.username,
            "icon_emoji": self.config.icon_emoji,
            "attachments": [{
                "color": self.level_to_color(alert.level),
                "title": format!("{} {}", self.level_to_emoji(alert.level), alert.title),
                "text": alert.message,
                "fields": [
                    {
                        "title": "Category",
                        "value": &alert.category,
                        "short": true
                    },
                    {
                        "title": "Level",
                        "value": format!("{:?}", alert.level),
                        "short": true
                    }
                ],
                "footer": "Polysniper",
                "ts": alert.timestamp.timestamp()
            }]
        });

        let response = ureq::post(&self.config.webhook_url)
            .set("Content-Type", "application/json")
            .send_string(&payload.to_string());

        match response {
            Ok(_) => {
                record_alert_sent(&alert.level.to_string(), "slack");
                Ok(())
            }
            Err(e) => {
                record_alert_failure("slack");
                Err(AlertError::Http(e.to_string()))
            }
        }
    }

    fn is_enabled(&self) -> bool {
        self.config.enabled && !self.config.webhook_url.is_empty()
    }
}

/// Telegram configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelegramConfig {
    pub enabled: bool,
    pub bot_token: String,
    pub chat_id: String,
    pub parse_mode: Option<String>,
}

impl Default for TelegramConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bot_token: String::new(),
            chat_id: String::new(),
            parse_mode: Some("HTML".to_string()),
        }
    }
}

/// Telegram channel implementation
pub struct TelegramChannel {
    config: TelegramConfig,
}

impl TelegramChannel {
    pub fn new(config: TelegramConfig) -> Self {
        Self { config }
    }

    fn level_to_emoji(&self, level: AlertLevel) -> &str {
        match level {
            AlertLevel::Info => "\u{2139}\u{fe0f}",    // ℹ️
            AlertLevel::Warning => "\u{26a0}\u{fe0f}", // ⚠️
            AlertLevel::Critical => "\u{1f6a8}",       // 🚨
        }
    }

    fn format_message(&self, alert: &Alert) -> String {
        format!(
            "{} <b>{}</b>\n\n{}\n\n<i>Category:</i> {}\n<i>Time:</i> {}",
            self.level_to_emoji(alert.level),
            html_escape(&alert.title),
            html_escape(&alert.message),
            html_escape(&alert.category),
            alert.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        )
    }
}

impl AlertChannel for TelegramChannel {
    fn name(&self) -> &str {
        "telegram"
    }

    fn send(&self, alert: &Alert) -> Result<(), AlertError> {
        if !self.is_enabled() {
            return Err(AlertError::NotConfigured("telegram".to_string()));
        }

        let url = format!(
            "https://api.telegram.org/bot{}/sendMessage",
            self.config.bot_token
        );

        let payload = serde_json::json!({
            "chat_id": self.config.chat_id,
            "text": self.format_message(alert),
            "parse_mode": self.config.parse_mode
        });

        let response = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_string(&payload.to_string());

        match response {
            Ok(_) => {
                record_alert_sent(&alert.level.to_string(), "telegram");
                Ok(())
            }
            Err(e) => {
                record_alert_failure("telegram");
                Err(AlertError::Http(e.to_string()))
            }
        }
    }

    fn is_enabled(&self) -> bool {
        self.config.enabled && !self.config.bot_token.is_empty() && !self.config.chat_id.is_empty()
    }
}

/// Simple HTML escape for Telegram
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    pub enabled: bool,
    pub min_level: AlertLevel,
    pub slack: SlackConfig,
    pub telegram: TelegramConfig,
    pub rate_limit_seconds: u64,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_level: AlertLevel::Warning,
            slack: SlackConfig::default(),
            telegram: TelegramConfig::default(),
            rate_limit_seconds: 60,
        }
    }
}

/// Alert manager for sending alerts through multiple channels
pub struct AlertManager {
    config: AlertingConfig,
    channels: Vec<Box<dyn AlertChannel>>,
    last_alerts: Arc<RwLock<std::collections::HashMap<String, DateTime<Utc>>>>,
}

impl AlertManager {
    pub fn new(config: AlertingConfig) -> Self {
        let mut channels: Vec<Box<dyn AlertChannel>> = Vec::new();

        if config.slack.enabled {
            channels.push(Box::new(SlackChannel::new(config.slack.clone())));
        }

        if config.telegram.enabled {
            channels.push(Box::new(TelegramChannel::new(config.telegram.clone())));
        }

        info!(channels = channels.len(), "Alert manager initialized");

        Self {
            config,
            channels,
            last_alerts: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Check if alert should be rate limited
    async fn is_rate_limited(&self, key: &str) -> bool {
        let last_alerts = self.last_alerts.read().await;
        if let Some(last_time) = last_alerts.get(key) {
            let elapsed = Utc::now().signed_duration_since(*last_time);
            elapsed.num_seconds() < self.config.rate_limit_seconds as i64
        } else {
            false
        }
    }

    /// Record alert sent time
    async fn record_alert(&self, key: &str) {
        let mut last_alerts = self.last_alerts.write().await;
        last_alerts.insert(key.to_string(), Utc::now());
    }

    /// Send an alert through all enabled channels
    pub async fn send(&self, alert: Alert) -> Result<(), AlertError> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check minimum level
        if alert.level < self.config.min_level {
            return Ok(());
        }

        // Check rate limiting
        let key = format!("{}:{}", alert.category, alert.title);
        if self.is_rate_limited(&key).await {
            warn!(
                category = %alert.category,
                title = %alert.title,
                "Alert rate limited"
            );
            return Err(AlertError::RateLimited);
        }

        // Send through all channels
        let mut last_error = None;
        for channel in &self.channels {
            if channel.is_enabled() {
                match channel.send(&alert) {
                    Ok(_) => {
                        info!(
                            channel = channel.name(),
                            level = %alert.level,
                            title = %alert.title,
                            "Alert sent"
                        );
                    }
                    Err(e) => {
                        error!(
                            channel = channel.name(),
                            error = %e,
                            "Failed to send alert"
                        );
                        last_error = Some(e);
                    }
                }
            }
        }

        // Record that we sent this alert
        self.record_alert(&key).await;

        match last_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Send a critical alert (bypasses rate limiting)
    pub async fn send_critical(&self, alert: Alert) -> Result<(), AlertError> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut last_error = None;
        for channel in &self.channels {
            if channel.is_enabled() {
                if let Err(e) = channel.send(&alert) {
                    error!(
                        channel = channel.name(),
                        error = %e,
                        "Failed to send critical alert"
                    );
                    last_error = Some(e);
                }
            }
        }

        match last_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Quick helper for sending info alerts
    pub async fn info(&self, category: &str, title: &str, message: &str) {
        let alert = Alert::info(category, title, message);
        let _ = self.send(alert).await;
    }

    /// Quick helper for sending warning alerts
    pub async fn warning(&self, category: &str, title: &str, message: &str) {
        let alert = Alert::warning(category, title, message);
        let _ = self.send(alert).await;
    }

    /// Quick helper for sending critical alerts
    pub async fn critical(&self, category: &str, title: &str, message: &str) {
        let alert = Alert::critical(category, title, message);
        let _ = self.send_critical(alert).await;
    }
}

// Pre-defined alert categories
pub mod categories {
    pub const TRADING: &str = "trading";
    pub const RISK: &str = "risk";
    pub const SYSTEM: &str = "system";
    pub const CONNECTION: &str = "connection";
    pub const MARKET: &str = "market";
}

// Pre-defined alert helpers
impl AlertManager {
    /// Alert for circuit breaker trigger
    pub async fn alert_circuit_breaker(&self, reason: &str, daily_pnl: f64) {
        self.critical(
            categories::RISK,
            "Circuit Breaker Triggered",
            &format!(
                "Trading has been halted.\nReason: {}\nDaily P&L: ${:.2}",
                reason, daily_pnl
            ),
        )
        .await;
    }

    /// Alert for large trade
    pub async fn alert_large_trade(&self, market_id: &str, side: &str, size_usd: f64) {
        self.info(
            categories::TRADING,
            "Large Trade Executed",
            &format!(
                "Market: {}\nSide: {}\nSize: ${:.2}",
                market_id, side, size_usd
            ),
        )
        .await;
    }

    /// Alert for connection loss
    pub async fn alert_connection_lost(&self, source: &str) {
        self.warning(
            categories::CONNECTION,
            "Connection Lost",
            &format!("Lost connection to {}", source),
        )
        .await;
    }

    /// Alert for new market discovery
    pub async fn alert_new_market(&self, market_id: &str, question: &str) {
        self.info(
            categories::MARKET,
            "New Market Discovered",
            &format!("ID: {}\nQuestion: {}", market_id, question),
        )
        .await;
    }

    /// Alert for high daily loss
    pub async fn alert_high_loss(&self, current_pnl: f64, limit: f64) {
        self.warning(
            categories::RISK,
            "Approaching Loss Limit",
            &format!(
                "Current daily P&L: ${:.2}\nLimit: ${:.2}\nRemaining: ${:.2}",
                current_pnl,
                limit,
                limit + current_pnl
            ),
        )
        .await;
    }

    /// Alert for strategy error
    pub async fn alert_strategy_error(&self, strategy_id: &str, error: &str) {
        self.warning(
            categories::SYSTEM,
            "Strategy Error",
            &format!("Strategy: {}\nError: {}", strategy_id, error),
        )
        .await;
    }
}
