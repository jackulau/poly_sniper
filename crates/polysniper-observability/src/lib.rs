//! Polysniper Observability
//!
//! Logging, metrics, alerting, and health checks.

pub mod alerting;
pub mod health;
pub mod logging;
pub mod metrics;

pub use alerting::{AlertChannel, AlertManager, AlertingConfig, SlackConfig, TelegramConfig};
pub use logging::{init_default_logging, init_logging, LogFormat};
pub use metrics::{
    gather_metrics,
    // Helper functions
    record_alert_failure,
    record_alert_sent,
    record_api_request,
    record_batch_deduplication,
    record_batch_processed,
    record_batch_processing,
    record_event_processing,
    record_new_market,
    record_order,
    record_risk_rejection,
    record_signal,
    record_strategy_error,
    record_strategy_processing,
    record_trade,
    record_websocket_reconnect,
    register_metrics,
    start_metrics_server,
    update_markets_monitored,
    update_pnl,
    update_uptime,
    update_websocket_status,
    MetricsConfig,
};
