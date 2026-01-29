//! Preset embed templates for Polysniper events
//!
//! Provides ready-to-use embed builders for common notification types.

use chrono::Utc;
use polysniper_core::{Outcome, Side, TradeSignal};
use rust_decimal::Decimal;

use crate::embed::{Embed, EmbedBuilder, EmbedColor};

/// Trade execution embed template
pub struct TradeEmbed;

impl TradeEmbed {
    /// Create an embed from a trade signal
    pub fn from_signal(signal: &TradeSignal) -> Embed {
        let side_emoji = match signal.side {
            Side::Buy => "📈",
            Side::Sell => "📉",
        };

        let outcome_emoji = match signal.outcome {
            Outcome::Yes => "✅",
            Outcome::No => "❌",
        };

        let color = match signal.side {
            Side::Buy => EmbedColor::Success,
            Side::Sell => EmbedColor::Warning,
        };

        let price_str = signal
            .price
            .map(|p| format!("{:.2}¢", p * Decimal::from(100)))
            .unwrap_or_else(|| "Market".to_string());

        EmbedBuilder::new()
            .title(format!("{} Trade Executed", side_emoji))
            .description(&signal.reason)
            .color(color)
            .inline_field("Market", truncate_str(&signal.market_id, 40))
            .inline_field(
                "Outcome",
                format!("{} {}", outcome_emoji, signal.outcome),
            )
            .inline_field("Side", signal.side.to_string())
            .inline_field("Price", price_str)
            .inline_field("Size", format!("{} contracts", signal.size))
            .inline_field("Value", format!("${:.2}", signal.size_usd))
            .footer(format!("Strategy: {} | Signal: {}", signal.strategy_id, signal.id))
            .timestamp_now()
            .build()
    }

    /// Create a simple trade notification embed
    pub fn simple(
        market: &str,
        side: Side,
        outcome: Outcome,
        size_usd: Decimal,
        reason: &str,
    ) -> Embed {
        let side_emoji = match side {
            Side::Buy => "📈",
            Side::Sell => "📉",
        };

        let color = match side {
            Side::Buy => EmbedColor::Success,
            Side::Sell => EmbedColor::Warning,
        };

        EmbedBuilder::new()
            .title(format!("{} {} {}", side_emoji, side, outcome))
            .description(reason)
            .color(color)
            .inline_field("Market", truncate_str(market, 50))
            .inline_field("Value", format!("${:.2}", size_usd))
            .timestamp_now()
            .build()
    }
}

/// Error notification embed template
pub struct ErrorEmbed;

impl ErrorEmbed {
    /// Create an embed from an error
    pub fn from_error(error: &dyn std::error::Error) -> Embed {
        Self::create("Error", &error.to_string())
    }

    /// Create an error embed with custom title and message
    pub fn create(title: &str, message: &str) -> Embed {
        EmbedBuilder::new()
            .title(format!("🚨 {}", title))
            .description(message)
            .color(EmbedColor::Error)
            .timestamp_now()
            .build()
    }

    /// Create an error embed with context
    pub fn with_context(title: &str, message: &str, context: &[(&str, &str)]) -> Embed {
        let mut builder = EmbedBuilder::new()
            .title(format!("🚨 {}", title))
            .description(message)
            .color(EmbedColor::Error);

        for (name, value) in context {
            builder = builder.inline_field(*name, *value);
        }

        builder.timestamp_now().build()
    }

    /// Create a critical error embed (for circuit breakers, etc.)
    pub fn critical(title: &str, message: &str) -> Embed {
        EmbedBuilder::new()
            .title(format!("🔴 CRITICAL: {}", title))
            .description(message)
            .color(EmbedColor::Custom(0x8B0000)) // Dark red
            .footer("Immediate attention required")
            .timestamp_now()
            .build()
    }
}

/// Risk decision embed template
pub struct RiskEmbed;

/// Result of a risk validation decision
#[derive(Debug, Clone)]
pub enum RiskDecision {
    Approved,
    Rejected { reason: String },
    PartiallyApproved { reason: String, adjusted_size: Decimal },
}

impl RiskEmbed {
    /// Create an embed from a risk decision
    pub fn from_decision(signal_id: &str, decision: &RiskDecision) -> Embed {
        match decision {
            RiskDecision::Approved => Self::approved(signal_id),
            RiskDecision::Rejected { reason } => Self::rejected(signal_id, reason),
            RiskDecision::PartiallyApproved {
                reason,
                adjusted_size,
            } => Self::partial(signal_id, reason, *adjusted_size),
        }
    }

    /// Create an approved risk embed
    pub fn approved(signal_id: &str) -> Embed {
        EmbedBuilder::new()
            .title("✅ Risk Check Passed")
            .description("Signal approved for execution")
            .color(EmbedColor::Success)
            .inline_field("Signal", signal_id)
            .inline_field("Status", "Approved")
            .timestamp_now()
            .build()
    }

    /// Create a rejected risk embed
    pub fn rejected(signal_id: &str, reason: &str) -> Embed {
        EmbedBuilder::new()
            .title("❌ Risk Check Failed")
            .description(reason)
            .color(EmbedColor::Error)
            .inline_field("Signal", signal_id)
            .inline_field("Status", "Rejected")
            .timestamp_now()
            .build()
    }

    /// Create a partially approved risk embed
    pub fn partial(signal_id: &str, reason: &str, adjusted_size: Decimal) -> Embed {
        EmbedBuilder::new()
            .title("⚠️ Risk Check: Partial Approval")
            .description(reason)
            .color(EmbedColor::Warning)
            .inline_field("Signal", signal_id)
            .inline_field("Status", "Partial")
            .inline_field("Adjusted Size", format!("{}", adjusted_size))
            .timestamp_now()
            .build()
    }

    /// Create a circuit breaker triggered embed
    pub fn circuit_breaker(reason: &str, loss_amount: Decimal) -> Embed {
        EmbedBuilder::new()
            .title("🔴 Circuit Breaker Triggered")
            .description(reason)
            .color(EmbedColor::Custom(0x8B0000))
            .inline_field("Daily Loss", format!("${:.2}", loss_amount))
            .inline_field("Status", "Trading Halted")
            .footer("Manual intervention required to resume")
            .timestamp_now()
            .build()
    }
}

/// Status/connection embed template
pub struct StatusEmbed;

impl StatusEmbed {
    /// Create a status embed builder
    pub fn builder() -> StatusEmbedBuilder {
        StatusEmbedBuilder::default()
    }

    /// Create a connected status embed
    pub fn connected(service: &str) -> Embed {
        EmbedBuilder::new()
            .title("🟢 Connected")
            .description(format!("Successfully connected to {}", service))
            .color(EmbedColor::Success)
            .timestamp_now()
            .build()
    }

    /// Create a disconnected status embed
    pub fn disconnected(service: &str, reason: Option<&str>) -> Embed {
        let description = match reason {
            Some(r) => format!("Disconnected from {}: {}", service, r),
            None => format!("Disconnected from {}", service),
        };

        EmbedBuilder::new()
            .title("🔴 Disconnected")
            .description(description)
            .color(EmbedColor::Error)
            .timestamp_now()
            .build()
    }

    /// Create a reconnecting status embed
    pub fn reconnecting(service: &str, attempt: u32) -> Embed {
        EmbedBuilder::new()
            .title("🟡 Reconnecting")
            .description(format!("Attempting to reconnect to {}", service))
            .color(EmbedColor::Warning)
            .inline_field("Attempt", format!("#{}", attempt))
            .timestamp_now()
            .build()
    }

    /// Create a system startup embed
    pub fn startup(version: &str) -> Embed {
        EmbedBuilder::new()
            .title("🚀 Polysniper Started")
            .description("Trading bot initialized and ready")
            .color(EmbedColor::Polymarket)
            .inline_field("Version", version)
            .inline_field("Mode", "Live")
            .timestamp(Utc::now())
            .build()
    }

    /// Create a system shutdown embed
    pub fn shutdown(reason: Option<&str>) -> Embed {
        let description = match reason {
            Some(r) => format!("Shutting down: {}", r),
            None => "Shutting down gracefully".to_string(),
        };

        EmbedBuilder::new()
            .title("⏹️ Polysniper Stopped")
            .description(description)
            .color(EmbedColor::Info)
            .timestamp_now()
            .build()
    }

    /// Create a heartbeat/health check embed
    pub fn heartbeat(uptime_secs: u64, active_orders: u32, positions: u32) -> Embed {
        let uptime = format_duration(uptime_secs);

        EmbedBuilder::new()
            .title("💓 Heartbeat")
            .color(EmbedColor::Info)
            .inline_field("Uptime", uptime)
            .inline_field("Active Orders", format!("{}", active_orders))
            .inline_field("Positions", format!("{}", positions))
            .timestamp_now()
            .build()
    }
}

/// Builder for custom status embeds
#[derive(Default)]
pub struct StatusEmbedBuilder {
    title: Option<String>,
    status: Option<StatusType>,
    services: Vec<(String, ServiceStatus)>,
    metrics: Vec<(String, String)>,
}

#[derive(Clone, Copy)]
pub enum StatusType {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Clone, Copy)]
pub enum ServiceStatus {
    Online,
    Offline,
    Degraded,
}

impl StatusEmbedBuilder {
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn status(mut self, status: StatusType) -> Self {
        self.status = Some(status);
        self
    }

    pub fn service(mut self, name: impl Into<String>, status: ServiceStatus) -> Self {
        self.services.push((name.into(), status));
        self
    }

    pub fn metric(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.metrics.push((name.into(), value.into()));
        self
    }

    pub fn build(self) -> Embed {
        let (emoji, color) = match self.status.unwrap_or(StatusType::Healthy) {
            StatusType::Healthy => ("🟢", EmbedColor::Success),
            StatusType::Degraded => ("🟡", EmbedColor::Warning),
            StatusType::Unhealthy => ("🔴", EmbedColor::Error),
        };

        let title = self
            .title
            .unwrap_or_else(|| "System Status".to_string());

        let mut builder = EmbedBuilder::new()
            .title(format!("{} {}", emoji, title))
            .color(color);

        // Add services
        for (name, status) in self.services {
            let status_str = match status {
                ServiceStatus::Online => "🟢 Online",
                ServiceStatus::Offline => "🔴 Offline",
                ServiceStatus::Degraded => "🟡 Degraded",
            };
            builder = builder.inline_field(name, status_str);
        }

        // Add metrics
        for (name, value) in self.metrics {
            builder = builder.inline_field(name, value);
        }

        builder.timestamp_now().build()
    }
}

impl Default for StatusEmbed {
    fn default() -> Self {
        Self
    }
}

/// Truncate a string to a maximum length, adding ellipsis if needed
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Format a duration in seconds to a human-readable string
fn format_duration(secs: u64) -> String {
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;

    if days > 0 {
        format!("{}d {}h {}m", days, hours, minutes)
    } else if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m", minutes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use polysniper_core::{OrderType, Outcome, Priority, Side};
    use rust_decimal_macros::dec;

    fn create_test_signal() -> TradeSignal {
        TradeSignal {
            id: "sig_123".to_string(),
            strategy_id: "momentum_v1".to_string(),
            market_id: "0x1234567890abcdef".to_string(),
            token_id: "token_yes".to_string(),
            outcome: Outcome::Yes,
            side: Side::Buy,
            price: Some(dec!(0.65)),
            size: dec!(100),
            size_usd: dec!(65.00),
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: Utc::now(),
            reason: "Strong momentum detected on BTC market".to_string(),
            metadata: serde_json::json!({}),
        }
    }

    #[test]
    fn test_trade_embed_from_signal() {
        let signal = create_test_signal();
        let embed = TradeEmbed::from_signal(&signal);

        assert!(embed.title.unwrap().contains("Trade Executed"));
        assert_eq!(embed.fields.len(), 6);
        assert!(embed.footer.is_some());
        assert!(embed.timestamp.is_some());
    }

    #[test]
    fn test_trade_embed_simple() {
        let embed = TradeEmbed::simple(
            "Will BTC reach $100k?",
            Side::Buy,
            Outcome::Yes,
            dec!(50.00),
            "Testing trade",
        );

        assert!(embed.title.is_some());
        assert_eq!(embed.fields.len(), 2);
    }

    #[test]
    fn test_error_embed() {
        let embed = ErrorEmbed::create("Connection Failed", "Could not connect to CLOB API");

        assert!(embed.title.unwrap().contains("Connection Failed"));
        assert!(embed.color.is_some());
    }

    #[test]
    fn test_error_embed_with_context() {
        let embed = ErrorEmbed::with_context(
            "Order Failed",
            "Insufficient funds",
            &[("Order ID", "ord_123"), ("Amount", "$100")],
        );

        assert_eq!(embed.fields.len(), 2);
    }

    #[test]
    fn test_error_embed_critical() {
        let embed = ErrorEmbed::critical("System Failure", "Database connection lost");

        assert!(embed.title.unwrap().contains("CRITICAL"));
        assert!(embed.footer.is_some());
    }

    #[test]
    fn test_risk_embed_approved() {
        let embed = RiskEmbed::approved("sig_123");

        assert!(embed.title.unwrap().contains("Passed"));
        assert_eq!(embed.fields.len(), 2);
    }

    #[test]
    fn test_risk_embed_rejected() {
        let embed = RiskEmbed::rejected("sig_123", "Position limit exceeded");

        assert!(embed.title.unwrap().contains("Failed"));
    }

    #[test]
    fn test_risk_embed_from_decision() {
        let decision = RiskDecision::PartiallyApproved {
            reason: "Size reduced due to daily limit".to_string(),
            adjusted_size: dec!(50),
        };

        let embed = RiskEmbed::from_decision("sig_123", &decision);
        assert!(embed.title.unwrap().contains("Partial"));
    }

    #[test]
    fn test_risk_embed_circuit_breaker() {
        let embed = RiskEmbed::circuit_breaker("Daily loss limit reached", dec!(500.00));

        assert!(embed.title.unwrap().contains("Circuit Breaker"));
        assert!(embed.footer.is_some());
    }

    #[test]
    fn test_status_embed_connected() {
        let embed = StatusEmbed::connected("Polymarket CLOB");

        assert!(embed.title.unwrap().contains("Connected"));
    }

    #[test]
    fn test_status_embed_disconnected() {
        let embed = StatusEmbed::disconnected("WebSocket", Some("Connection timeout"));

        assert!(embed.title.unwrap().contains("Disconnected"));
        assert!(embed.description.unwrap().contains("timeout"));
    }

    #[test]
    fn test_status_embed_reconnecting() {
        let embed = StatusEmbed::reconnecting("WebSocket", 3);

        assert!(embed.title.unwrap().contains("Reconnecting"));
        assert!(embed.fields[0].value.contains("3"));
    }

    #[test]
    fn test_status_embed_startup() {
        let embed = StatusEmbed::startup("1.0.0");

        assert!(embed.title.unwrap().contains("Started"));
        assert!(embed.fields.iter().any(|f| f.value == "1.0.0"));
    }

    #[test]
    fn test_status_embed_shutdown() {
        let embed = StatusEmbed::shutdown(Some("User requested"));

        assert!(embed.title.unwrap().contains("Stopped"));
    }

    #[test]
    fn test_status_embed_heartbeat() {
        let embed = StatusEmbed::heartbeat(3661, 5, 2); // 1h 1m 1s

        assert!(embed.title.unwrap().contains("Heartbeat"));
        assert_eq!(embed.fields.len(), 3);
        assert!(embed.fields[0].value.contains("1h"));
    }

    #[test]
    fn test_status_embed_builder() {
        let embed = StatusEmbed::builder()
            .title("API Status")
            .status(StatusType::Degraded)
            .service("CLOB", ServiceStatus::Online)
            .service("WebSocket", ServiceStatus::Degraded)
            .metric("Latency", "150ms")
            .build();

        assert!(embed.title.unwrap().contains("API Status"));
        assert_eq!(embed.fields.len(), 3);
    }

    #[test]
    fn test_truncate_str() {
        assert_eq!(truncate_str("short", 10), "short");
        assert_eq!(truncate_str("this is a long string", 10), "this is...");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30), "0m");
        assert_eq!(format_duration(3600), "1h 0m");
        assert_eq!(format_duration(90000), "1d 1h 0m");
    }
}
