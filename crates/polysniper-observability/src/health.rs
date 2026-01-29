//! Health check HTTP server for Polysniper
//!
//! Provides HTTP endpoints for monitoring and orchestration systems.

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::watch;
use tracing::{error, info};

/// Health check server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    pub enabled: bool,
    pub bind_address: String,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bind_address: "127.0.0.1:8080".to_string(),
        }
    }
}

/// Health status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Overall status: "healthy", "degraded", or "unhealthy"
    pub status: String,
    /// Whether trading is enabled (inverse of risk manager halt)
    pub trading_enabled: bool,
    /// Whether WebSocket is connected
    pub websocket_connected: bool,
    /// Number of markets loaded
    pub markets_loaded: u32,
    /// Uptime in seconds
    pub uptime_secs: u64,
    /// Timestamp of last event received
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_event_timestamp: Option<DateTime<Utc>>,
}

impl HealthStatus {
    fn compute_status(&mut self) {
        if !self.websocket_connected {
            self.status = "unhealthy".to_string();
        } else if self.markets_loaded == 0 || !self.trading_enabled {
            self.status = "degraded".to_string();
        } else {
            self.status = "healthy".to_string();
        }
    }
}

/// Shared state for the health server
#[derive(Clone)]
pub struct HealthState {
    inner: Arc<HealthStateInner>,
}

struct HealthStateInner {
    start_time: Instant,
    trading_enabled: AtomicBool,
    websocket_connected: AtomicBool,
    markets_loaded: AtomicU64,
    last_event_timestamp: tokio::sync::RwLock<Option<DateTime<Utc>>>,
}

impl HealthState {
    /// Create new health state
    pub fn new() -> Self {
        Self {
            inner: Arc::new(HealthStateInner {
                start_time: Instant::now(),
                trading_enabled: AtomicBool::new(true),
                websocket_connected: AtomicBool::new(false),
                markets_loaded: AtomicU64::new(0),
                last_event_timestamp: tokio::sync::RwLock::new(None),
            }),
        }
    }

    /// Set trading enabled status
    pub fn set_trading_enabled(&self, enabled: bool) {
        self.inner.trading_enabled.store(enabled, Ordering::SeqCst);
    }

    /// Set WebSocket connected status
    pub fn set_websocket_connected(&self, connected: bool) {
        self.inner
            .websocket_connected
            .store(connected, Ordering::SeqCst);
    }

    /// Set number of markets loaded
    pub fn set_markets_loaded(&self, count: u32) {
        self.inner
            .markets_loaded
            .store(count as u64, Ordering::SeqCst);
    }

    /// Record an event timestamp
    pub async fn record_event(&self) {
        let mut ts = self.inner.last_event_timestamp.write().await;
        *ts = Some(Utc::now());
    }

    /// Get current health status
    pub async fn get_status(&self) -> HealthStatus {
        let uptime_secs = self.inner.start_time.elapsed().as_secs();
        let last_event_timestamp = *self.inner.last_event_timestamp.read().await;

        let mut status = HealthStatus {
            status: String::new(),
            trading_enabled: self.inner.trading_enabled.load(Ordering::SeqCst),
            websocket_connected: self.inner.websocket_connected.load(Ordering::SeqCst),
            markets_loaded: self.inner.markets_loaded.load(Ordering::SeqCst) as u32,
            uptime_secs,
            last_event_timestamp,
        };
        status.compute_status();
        status
    }
}

impl Default for HealthState {
    fn default() -> Self {
        Self::new()
    }
}

/// Health check server
pub struct HealthServer {
    bind_address: String,
    state: HealthState,
    shutdown_rx: watch::Receiver<bool>,
}

impl HealthServer {
    /// Create a new health server
    pub fn new(
        config: &HealthConfig,
        state: HealthState,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        Self {
            bind_address: config.bind_address.clone(),
            state,
            shutdown_rx,
        }
    }

    /// Start the health server
    pub async fn run(self) -> Result<(), std::io::Error> {
        let app = create_router(self.state);
        let addr: std::net::SocketAddr = self
            .bind_address
            .parse()
            .expect("Invalid bind address for health server");

        info!(address = %addr, "Starting health check server");

        let listener = tokio::net::TcpListener::bind(addr).await?;

        let mut shutdown_rx = self.shutdown_rx;
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.changed().await;
                info!("Health server shutting down");
            })
            .await
    }

    /// Start the health server in a background task
    pub fn spawn(self) -> tokio::task::JoinHandle<Result<(), std::io::Error>> {
        tokio::spawn(self.run())
    }
}

/// Create the Axum router for health endpoints
pub fn create_router(state: HealthState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/health/ready", get(readiness_check))
        .route("/health/status", get(status_check))
        .with_state(state)
}

/// Basic liveness check - returns 200 OK if server is running
async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

/// Readiness check - returns 200 if ready to serve traffic
async fn readiness_check(State(state): State<HealthState>) -> impl IntoResponse {
    let status = state.get_status().await;

    if status.websocket_connected && status.markets_loaded > 0 {
        (StatusCode::OK, "Ready")
    } else {
        let mut reasons = Vec::new();
        if !status.websocket_connected {
            reasons.push("WebSocket not connected");
        }
        if status.markets_loaded == 0 {
            reasons.push("No markets loaded");
        }
        (
            StatusCode::SERVICE_UNAVAILABLE,
            reasons.join(", ").leak() as &'static str,
        )
    }
}

/// Detailed status check - returns full JSON health status
async fn status_check(State(state): State<HealthState>) -> impl IntoResponse {
    let status = state.get_status().await;
    Json(status)
}

/// Start health server with default shutdown signal
pub async fn start_health_server(
    config: &HealthConfig,
    state: HealthState,
) -> Option<tokio::task::JoinHandle<Result<(), std::io::Error>>> {
    if !config.enabled {
        info!("Health server disabled by configuration");
        return None;
    }

    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Set up ctrl+c handler for graceful shutdown
    tokio::spawn(async move {
        if let Err(e) = tokio::signal::ctrl_c().await {
            error!(error = %e, "Failed to install ctrl+c handler");
        }
        let _ = shutdown_tx.send(true);
    });

    let server = HealthServer::new(config, state, shutdown_rx);
    Some(server.spawn())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_state_defaults() {
        let state = HealthState::new();
        let status = state.get_status().await;

        assert!(!status.websocket_connected);
        assert!(status.trading_enabled);
        assert_eq!(status.markets_loaded, 0);
        assert_eq!(status.status, "unhealthy");
    }

    #[tokio::test]
    async fn test_health_state_healthy() {
        let state = HealthState::new();
        state.set_websocket_connected(true);
        state.set_markets_loaded(10);
        state.set_trading_enabled(true);

        let status = state.get_status().await;

        assert!(status.websocket_connected);
        assert!(status.trading_enabled);
        assert_eq!(status.markets_loaded, 10);
        assert_eq!(status.status, "healthy");
    }

    #[tokio::test]
    async fn test_health_state_degraded() {
        let state = HealthState::new();
        state.set_websocket_connected(true);
        state.set_markets_loaded(0);

        let status = state.get_status().await;
        assert_eq!(status.status, "degraded");

        state.set_markets_loaded(10);
        state.set_trading_enabled(false);

        let status = state.get_status().await;
        assert_eq!(status.status, "degraded");
    }

    #[tokio::test]
    async fn test_record_event() {
        let state = HealthState::new();
        assert!(state.get_status().await.last_event_timestamp.is_none());

        state.record_event().await;
        assert!(state.get_status().await.last_event_timestamp.is_some());
    }
}
