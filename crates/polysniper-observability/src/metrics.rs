//! Prometheus metrics for Polysniper

use axum::{routing::get, Router};
use lazy_static::lazy_static;
use prometheus::{
    CounterVec, Gauge, GaugeVec, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge,
    Opts, Registry, TextEncoder,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tracing::info;

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    // Trading metrics
    pub static ref SIGNALS_GENERATED: IntCounterVec = IntCounterVec::new(
        Opts::new("polysniper_signals_generated_total", "Total trade signals generated"),
        &["strategy", "side"]
    ).unwrap();

    pub static ref ORDERS_SUBMITTED: IntCounterVec = IntCounterVec::new(
        Opts::new("polysniper_orders_submitted_total", "Total orders submitted"),
        &["strategy", "side", "status"]
    ).unwrap();

    pub static ref TRADES_EXECUTED: IntCounterVec = IntCounterVec::new(
        Opts::new("polysniper_trades_executed_total", "Total trades executed"),
        &["strategy", "side", "outcome"]
    ).unwrap();

    pub static ref TRADE_VOLUME_USD: CounterVec = CounterVec::new(
        Opts::new("polysniper_trade_volume_usd_total", "Total trading volume in USD"),
        &["strategy", "side"]
    ).unwrap();

    pub static ref REALIZED_PNL_USD: Gauge = Gauge::new(
        "polysniper_realized_pnl_usd",
        "Realized P&L in USD"
    ).unwrap();

    pub static ref UNREALIZED_PNL_USD: Gauge = Gauge::new(
        "polysniper_unrealized_pnl_usd",
        "Unrealized P&L in USD"
    ).unwrap();

    pub static ref DAILY_PNL_USD: Gauge = Gauge::new(
        "polysniper_daily_pnl_usd",
        "Daily P&L in USD"
    ).unwrap();

    // Risk metrics
    pub static ref RISK_REJECTIONS: IntCounterVec = IntCounterVec::new(
        Opts::new("polysniper_risk_rejections_total", "Total signals rejected by risk manager"),
        &["reason"]
    ).unwrap();

    pub static ref RISK_MODIFICATIONS: IntCounter = IntCounter::new(
        "polysniper_risk_modifications_total",
        "Total signals modified by risk manager"
    ).unwrap();

    pub static ref CIRCUIT_BREAKER_TRIGGERED: IntCounter = IntCounter::new(
        "polysniper_circuit_breaker_triggered_total",
        "Total circuit breaker triggers"
    ).unwrap();

    pub static ref CURRENT_POSITION_USD: GaugeVec = GaugeVec::new(
        Opts::new("polysniper_current_position_usd", "Current position size in USD"),
        &["market_id"]
    ).unwrap();

    pub static ref ORDERS_PER_MINUTE: IntGauge = IntGauge::new(
        "polysniper_orders_per_minute",
        "Current orders per minute rate"
    ).unwrap();

    // Event processing metrics
    pub static ref EVENTS_PROCESSED: IntCounterVec = IntCounterVec::new(
        Opts::new("polysniper_events_processed_total", "Total events processed"),
        &["event_type"]
    ).unwrap();

    pub static ref EVENT_PROCESSING_DURATION: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "polysniper_event_processing_duration_seconds",
            "Event processing duration in seconds"
        ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        &["event_type"]
    ).unwrap();

    pub static ref EVENT_BUS_LAG: IntGauge = IntGauge::new(
        "polysniper_event_bus_lag",
        "Number of lagged messages in event bus"
    ).unwrap();

    // Strategy metrics
    pub static ref STRATEGY_PROCESSING_DURATION: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "polysniper_strategy_processing_duration_seconds",
            "Strategy processing duration in seconds"
        ).buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
        &["strategy"]
    ).unwrap();

    pub static ref STRATEGY_ERRORS: IntCounterVec = IntCounterVec::new(
        Opts::new("polysniper_strategy_errors_total", "Total strategy processing errors"),
        &["strategy"]
    ).unwrap();

    // Market metrics
    pub static ref MARKETS_MONITORED: IntGauge = IntGauge::new(
        "polysniper_markets_monitored",
        "Number of markets currently monitored"
    ).unwrap();

    pub static ref NEW_MARKETS_DISCOVERED: IntCounter = IntCounter::new(
        "polysniper_new_markets_discovered_total",
        "Total new markets discovered"
    ).unwrap();

    pub static ref PRICE_UPDATES: IntCounterVec = IntCounterVec::new(
        Opts::new("polysniper_price_updates_total", "Total price updates received"),
        &["token_id"]
    ).unwrap();

    // Connection metrics
    pub static ref WEBSOCKET_CONNECTED: IntGauge = IntGauge::new(
        "polysniper_websocket_connected",
        "WebSocket connection status (1=connected, 0=disconnected)"
    ).unwrap();

    pub static ref WEBSOCKET_RECONNECTS: IntCounter = IntCounter::new(
        "polysniper_websocket_reconnects_total",
        "Total WebSocket reconnection attempts"
    ).unwrap();

    pub static ref API_REQUESTS: IntCounterVec = IntCounterVec::new(
        Opts::new("polysniper_api_requests_total", "Total API requests made"),
        &["endpoint", "status"]
    ).unwrap();

    pub static ref API_REQUEST_DURATION: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "polysniper_api_request_duration_seconds",
            "API request duration in seconds"
        ).buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
        &["endpoint"]
    ).unwrap();

    // Alerting metrics
    pub static ref ALERTS_SENT: IntCounterVec = IntCounterVec::new(
        Opts::new("polysniper_alerts_sent_total", "Total alerts sent"),
        &["level", "channel"]
    ).unwrap();

    pub static ref ALERT_SEND_FAILURES: IntCounterVec = IntCounterVec::new(
        Opts::new("polysniper_alert_send_failures_total", "Total alert send failures"),
        &["channel"]
    ).unwrap();

    // System metrics
    pub static ref UPTIME_SECONDS: IntGauge = IntGauge::new(
        "polysniper_uptime_seconds",
        "System uptime in seconds"
    ).unwrap();
}

/// Register all metrics with the registry
pub fn register_metrics() {
    // Trading metrics
    REGISTRY.register(Box::new(SIGNALS_GENERATED.clone())).ok();
    REGISTRY.register(Box::new(ORDERS_SUBMITTED.clone())).ok();
    REGISTRY.register(Box::new(TRADES_EXECUTED.clone())).ok();
    REGISTRY.register(Box::new(TRADE_VOLUME_USD.clone())).ok();
    REGISTRY.register(Box::new(REALIZED_PNL_USD.clone())).ok();
    REGISTRY.register(Box::new(UNREALIZED_PNL_USD.clone())).ok();
    REGISTRY.register(Box::new(DAILY_PNL_USD.clone())).ok();

    // Risk metrics
    REGISTRY.register(Box::new(RISK_REJECTIONS.clone())).ok();
    REGISTRY.register(Box::new(RISK_MODIFICATIONS.clone())).ok();
    REGISTRY
        .register(Box::new(CIRCUIT_BREAKER_TRIGGERED.clone()))
        .ok();
    REGISTRY
        .register(Box::new(CURRENT_POSITION_USD.clone()))
        .ok();
    REGISTRY.register(Box::new(ORDERS_PER_MINUTE.clone())).ok();

    // Event metrics
    REGISTRY.register(Box::new(EVENTS_PROCESSED.clone())).ok();
    REGISTRY
        .register(Box::new(EVENT_PROCESSING_DURATION.clone()))
        .ok();
    REGISTRY.register(Box::new(EVENT_BUS_LAG.clone())).ok();

    // Strategy metrics
    REGISTRY
        .register(Box::new(STRATEGY_PROCESSING_DURATION.clone()))
        .ok();
    REGISTRY.register(Box::new(STRATEGY_ERRORS.clone())).ok();

    // Market metrics
    REGISTRY.register(Box::new(MARKETS_MONITORED.clone())).ok();
    REGISTRY
        .register(Box::new(NEW_MARKETS_DISCOVERED.clone()))
        .ok();
    REGISTRY.register(Box::new(PRICE_UPDATES.clone())).ok();

    // Connection metrics
    REGISTRY
        .register(Box::new(WEBSOCKET_CONNECTED.clone()))
        .ok();
    REGISTRY
        .register(Box::new(WEBSOCKET_RECONNECTS.clone()))
        .ok();
    REGISTRY.register(Box::new(API_REQUESTS.clone())).ok();
    REGISTRY
        .register(Box::new(API_REQUEST_DURATION.clone()))
        .ok();

    // Alerting metrics
    REGISTRY.register(Box::new(ALERTS_SENT.clone())).ok();
    REGISTRY
        .register(Box::new(ALERT_SEND_FAILURES.clone()))
        .ok();

    // System metrics
    REGISTRY.register(Box::new(UPTIME_SECONDS.clone())).ok();
}

/// Get metrics as Prometheus text format
pub fn gather_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    encoder
        .encode_to_string(&metric_families)
        .unwrap_or_default()
}

/// Metrics HTTP handler
async fn metrics_handler() -> String {
    gather_metrics()
}

/// Health check handler
async fn health_handler() -> &'static str {
    "OK"
}

/// Start the metrics HTTP server
pub async fn start_metrics_server(port: u16) -> tokio::task::JoinHandle<()> {
    register_metrics();

    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler));

    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    info!(port = %port, "Starting metrics server");

    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    })
}

// Helper functions for recording metrics

/// Record a trade signal
pub fn record_signal(strategy: &str, side: &str) {
    SIGNALS_GENERATED.with_label_values(&[strategy, side]).inc();
}

/// Record an order submission
pub fn record_order(strategy: &str, side: &str, status: &str) {
    ORDERS_SUBMITTED
        .with_label_values(&[strategy, side, status])
        .inc();
}

/// Record a trade execution
pub fn record_trade(strategy: &str, side: &str, outcome: &str, volume_usd: f64) {
    TRADES_EXECUTED
        .with_label_values(&[strategy, side, outcome])
        .inc();
    TRADE_VOLUME_USD
        .with_label_values(&[strategy, side])
        .inc_by(volume_usd);
}

/// Record risk rejection
pub fn record_risk_rejection(reason: &str) {
    RISK_REJECTIONS.with_label_values(&[reason]).inc();
}

/// Update P&L metrics
pub fn update_pnl(realized: f64, unrealized: f64, daily: f64) {
    REALIZED_PNL_USD.set(realized);
    UNREALIZED_PNL_USD.set(unrealized);
    DAILY_PNL_USD.set(daily);
}

/// Record event processing
pub fn record_event_processing(event_type: &str, duration_secs: f64) {
    EVENTS_PROCESSED.with_label_values(&[event_type]).inc();
    EVENT_PROCESSING_DURATION
        .with_label_values(&[event_type])
        .observe(duration_secs);
}

/// Record strategy processing
pub fn record_strategy_processing(strategy: &str, duration_secs: f64) {
    STRATEGY_PROCESSING_DURATION
        .with_label_values(&[strategy])
        .observe(duration_secs);
}

/// Record strategy error
pub fn record_strategy_error(strategy: &str) {
    STRATEGY_ERRORS.with_label_values(&[strategy]).inc();
}

/// Update market count
pub fn update_markets_monitored(count: i64) {
    MARKETS_MONITORED.set(count);
}

/// Record new market discovery
pub fn record_new_market() {
    NEW_MARKETS_DISCOVERED.inc();
}

/// Update WebSocket connection status
pub fn update_websocket_status(connected: bool) {
    WEBSOCKET_CONNECTED.set(if connected { 1 } else { 0 });
}

/// Record WebSocket reconnect
pub fn record_websocket_reconnect() {
    WEBSOCKET_RECONNECTS.inc();
}

/// Record API request
pub fn record_api_request(endpoint: &str, status: &str, duration_secs: f64) {
    API_REQUESTS.with_label_values(&[endpoint, status]).inc();
    API_REQUEST_DURATION
        .with_label_values(&[endpoint])
        .observe(duration_secs);
}

/// Record alert sent
pub fn record_alert_sent(level: &str, channel: &str) {
    ALERTS_SENT.with_label_values(&[level, channel]).inc();
}

/// Record alert send failure
pub fn record_alert_failure(channel: &str) {
    ALERT_SEND_FAILURES.with_label_values(&[channel]).inc();
}

/// Update uptime
pub fn update_uptime(seconds: i64) {
    UPTIME_SECONDS.set(seconds);
}

/// Configuration for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Whether metrics are enabled
    pub enabled: bool,
    /// Port for metrics HTTP server
    pub port: u16,
    /// Collection interval in seconds
    pub collection_interval_secs: u64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 9090,
            collection_interval_secs: 60,
        }
    }
}
