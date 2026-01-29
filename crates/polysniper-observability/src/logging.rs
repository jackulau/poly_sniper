//! Logging configuration using tracing

use tracing::Level;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};

/// Logging format
#[derive(Debug, Clone, Copy)]
pub enum LogFormat {
    /// Human-readable format
    Pretty,
    /// JSON format for log aggregation
    Json,
    /// Compact format
    Compact,
}

/// Initialize logging with the specified format
pub fn init_logging(format: LogFormat, default_level: Level) {
    let env_filter = EnvFilter::builder()
        .with_default_directive(default_level.into())
        .from_env_lossy()
        .add_directive("hyper=warn".parse().unwrap())
        .add_directive("reqwest=warn".parse().unwrap())
        .add_directive("tokio_tungstenite=warn".parse().unwrap())
        .add_directive("tungstenite=warn".parse().unwrap());

    match format {
        LogFormat::Pretty => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(
                    fmt::layer()
                        .with_target(true)
                        .with_thread_ids(false)
                        .with_file(false)
                        .with_line_number(false)
                        .with_span_events(FmtSpan::CLOSE),
                )
                .init();
        }
        LogFormat::Json => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt::layer().json().with_span_events(FmtSpan::CLOSE))
                .init();
        }
        LogFormat::Compact => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(
                    fmt::layer()
                        .compact()
                        .with_target(false)
                        .with_thread_ids(false),
                )
                .init();
        }
    }
}

/// Initialize logging with default settings
pub fn init_default_logging() {
    init_logging(LogFormat::Pretty, Level::INFO);
}
