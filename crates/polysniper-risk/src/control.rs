//! Emergency kill switch control server
//!
//! Provides HTTP API and Unix signal handlers for emergency trading control.

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use polysniper_core::{ControlConfig, Position, RiskValidator, StateProvider};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tracing::{error, info, warn};

use crate::RiskManager;

/// Control server error
#[derive(Debug, Error)]
pub enum ControlError {
    #[error("Failed to bind to address: {0}")]
    BindError(#[from] std::io::Error),
}

/// Shared state for the control server
pub struct ControlState {
    risk_manager: Arc<RiskManager>,
    state_provider: Arc<dyn StateProvider>,
    #[allow(dead_code)] // Kept alive to prevent broadcast channel closure
    shutdown_tx: broadcast::Sender<()>,
}

impl ControlState {
    pub fn new(
        risk_manager: Arc<RiskManager>,
        state_provider: Arc<dyn StateProvider>,
        shutdown_tx: broadcast::Sender<()>,
    ) -> Self {
        Self {
            risk_manager,
            state_provider,
            shutdown_tx,
        }
    }
}

/// Control server for emergency kill switch
pub struct ControlServer {
    config: ControlConfig,
    state: Arc<ControlState>,
}

impl ControlServer {
    /// Create a new control server
    pub fn new(
        config: ControlConfig,
        risk_manager: Arc<RiskManager>,
        state_provider: Arc<dyn StateProvider>,
        shutdown_tx: broadcast::Sender<()>,
    ) -> Self {
        let state = Arc::new(ControlState::new(risk_manager, state_provider, shutdown_tx));
        Self { config, state }
    }

    /// Build the axum router
    fn build_router(&self) -> Router {
        let auth_token = self.config.auth_token.clone();

        let router = Router::new()
            .route("/halt", post(halt_handler))
            .route("/resume", post(resume_handler))
            .route("/status", get(status_handler))
            .route("/positions", get(positions_handler))
            .route("/close-all", post(close_all_handler))
            .route("/health", get(health_handler))
            .with_state(self.state.clone());

        // Add auth middleware if token is configured
        if let Some(token) = auth_token {
            router.layer(middleware::from_fn(move |req, next| {
                auth_middleware(req, next, token.clone())
            }))
        } else {
            router
        }
    }

    /// Run the control server
    pub async fn run(self) -> Result<(), ControlError> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = TcpListener::bind(&addr).await?;
        info!(address = %addr, "Control server listening");

        let router = self.build_router();
        if let Err(e) = axum::serve(listener, router).await {
            error!(error = %e, "Control server error");
        }

        Ok(())
    }

    /// Get the configured address
    pub fn address(&self) -> String {
        format!("{}:{}", self.config.host, self.config.port)
    }
}

/// Authentication middleware
async fn auth_middleware(req: Request, next: Next, expected_token: String) -> Response {
    // Allow health check without auth
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    let auth_header = req
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = header.trim_start_matches("Bearer ");
            if token == expected_token {
                next.run(req).await
            } else {
                (StatusCode::UNAUTHORIZED, "Invalid token").into_response()
            }
        }
        _ => (StatusCode::UNAUTHORIZED, "Missing or invalid Authorization header").into_response(),
    }
}

/// Response for status endpoint
#[derive(Debug, Serialize, Deserialize)]
pub struct StatusResponse {
    pub halted: bool,
    pub halt_reason: Option<String>,
    pub daily_pnl: Decimal,
    pub portfolio_value: Decimal,
    pub position_count: usize,
}

/// Response for positions endpoint
#[derive(Debug, Serialize, Deserialize)]
pub struct PositionsResponse {
    pub positions: Vec<Position>,
    pub total_value: Decimal,
}

/// Request for halt endpoint
#[derive(Debug, Deserialize)]
pub struct HaltRequest {
    pub reason: Option<String>,
}

/// Request for resume endpoint
#[derive(Debug, Deserialize)]
pub struct ResumeRequest {
    pub confirmation: Option<String>,
}

/// Response for generic operations
#[derive(Debug, Serialize)]
pub struct OperationResponse {
    pub success: bool,
    pub message: String,
}

/// Health check handler
async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// POST /halt - Immediately halt all trading
async fn halt_handler(
    State(state): State<Arc<ControlState>>,
    Json(req): Json<HaltRequest>,
) -> impl IntoResponse {
    let reason = req.reason.unwrap_or_else(|| "Manual halt via API".to_string());
    state.risk_manager.halt(&reason);

    info!(reason = %reason, "Trading halted via control API");

    Json(OperationResponse {
        success: true,
        message: format!("Trading halted: {}", reason),
    })
}

/// POST /resume - Resume trading
async fn resume_handler(
    State(state): State<Arc<ControlState>>,
    Json(req): Json<ResumeRequest>,
) -> impl IntoResponse {
    // Require explicit confirmation
    if req.confirmation.as_deref() != Some("CONFIRM_RESUME") {
        return (
            StatusCode::BAD_REQUEST,
            Json(OperationResponse {
                success: false,
                message: "Must provide confirmation: CONFIRM_RESUME".to_string(),
            }),
        );
    }

    state.risk_manager.resume();
    info!("Trading resumed via control API");

    (
        StatusCode::OK,
        Json(OperationResponse {
            success: true,
            message: "Trading resumed".to_string(),
        }),
    )
}

/// GET /status - Get current system status
async fn status_handler(State(state): State<Arc<ControlState>>) -> impl IntoResponse {
    let halted = state.risk_manager.is_halted();
    let halt_reason = state.risk_manager.get_halt_reason().await;
    let daily_pnl = state.state_provider.get_daily_pnl().await;
    let portfolio_value = state.state_provider.get_portfolio_value().await;
    let positions = state.state_provider.get_all_positions().await;

    Json(StatusResponse {
        halted,
        halt_reason,
        daily_pnl,
        portfolio_value,
        position_count: positions.len(),
    })
}

/// GET /positions - Get all open positions
async fn positions_handler(State(state): State<Arc<ControlState>>) -> impl IntoResponse {
    let positions = state.state_provider.get_all_positions().await;
    let total_value = positions
        .iter()
        .map(|p| p.size * p.avg_price)
        .sum::<Decimal>();

    Json(PositionsResponse {
        positions,
        total_value,
    })
}

/// POST /close-all - Signal to close all positions (dry-run safe)
async fn close_all_handler(State(state): State<Arc<ControlState>>) -> impl IntoResponse {
    // First halt trading
    state.risk_manager.halt("Close-all initiated");

    let positions = state.state_provider.get_all_positions().await;

    info!(
        position_count = positions.len(),
        "Close-all initiated via control API"
    );

    // Note: Actual position closing would be handled by the main app
    // This endpoint signals intent and halts new trading
    Json(OperationResponse {
        success: true,
        message: format!(
            "Close-all initiated. {} positions flagged for closing. Trading halted.",
            positions.len()
        ),
    })
}

/// Signal handler for Unix signals
pub struct SignalHandler {
    risk_manager: Arc<RiskManager>,
    state_provider: Arc<dyn StateProvider>,
    shutdown_tx: broadcast::Sender<()>,
}

impl SignalHandler {
    pub fn new(
        risk_manager: Arc<RiskManager>,
        state_provider: Arc<dyn StateProvider>,
        shutdown_tx: broadcast::Sender<()>,
    ) -> Self {
        Self {
            risk_manager,
            state_provider,
            shutdown_tx,
        }
    }

    /// Start signal handlers
    pub async fn start(self) {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};

            let risk_manager = self.risk_manager.clone();
            let state_provider = self.state_provider.clone();

            // SIGUSR1 - Toggle halt/resume
            let rm_usr1 = risk_manager.clone();
            tokio::spawn(async move {
                let mut stream = signal(SignalKind::user_defined1())
                    .expect("Failed to register SIGUSR1 handler");
                loop {
                    stream.recv().await;
                    if rm_usr1.is_halted() {
                        rm_usr1.resume();
                        info!("Trading resumed via SIGUSR1");
                    } else {
                        rm_usr1.halt("Manual halt via SIGUSR1");
                        warn!("Trading halted via SIGUSR1");
                    }
                }
            });

            // SIGUSR2 - Print status
            let rm_usr2 = risk_manager.clone();
            let sp_usr2 = state_provider.clone();
            tokio::spawn(async move {
                let mut stream = signal(SignalKind::user_defined2())
                    .expect("Failed to register SIGUSR2 handler");
                loop {
                    stream.recv().await;
                    let halted = rm_usr2.is_halted();
                    let daily_pnl = sp_usr2.get_daily_pnl().await;
                    let portfolio_value = sp_usr2.get_portfolio_value().await;
                    let positions = sp_usr2.get_all_positions().await;

                    info!(
                        halted = halted,
                        daily_pnl = %daily_pnl,
                        portfolio_value = %portfolio_value,
                        position_count = positions.len(),
                        "Status dump via SIGUSR2"
                    );
                }
            });

            // SIGTERM - Graceful shutdown
            let rm_term = risk_manager;
            let sp_term = state_provider;
            let shutdown_tx = self.shutdown_tx;
            tokio::spawn(async move {
                let mut stream =
                    signal(SignalKind::terminate()).expect("Failed to register SIGTERM handler");
                stream.recv().await;

                // Dump final status
                let daily_pnl = sp_term.get_daily_pnl().await;
                let portfolio_value = sp_term.get_portfolio_value().await;
                let positions = sp_term.get_all_positions().await;

                info!(
                    daily_pnl = %daily_pnl,
                    portfolio_value = %portfolio_value,
                    position_count = positions.len(),
                    "SIGTERM received - initiating graceful shutdown"
                );

                // Halt trading first
                rm_term.halt("SIGTERM shutdown");

                // Send shutdown signal
                if let Err(e) = shutdown_tx.send(()) {
                    error!(error = %e, "Failed to send shutdown signal");
                }
            });

            info!("Unix signal handlers registered (SIGUSR1, SIGUSR2, SIGTERM)");
        }

        #[cfg(not(unix))]
        {
            warn!("Unix signal handlers not available on this platform");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use polysniper_core::{Market, MarketId, Orderbook, RiskConfig, TokenId};
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    /// Mock state provider for testing
    struct MockStateProvider {
        positions: RwLock<Vec<Position>>,
        daily_pnl: RwLock<Decimal>,
        portfolio_value: RwLock<Decimal>,
    }

    impl MockStateProvider {
        fn new() -> Self {
            Self {
                positions: RwLock::new(vec![]),
                daily_pnl: RwLock::new(Decimal::ZERO),
                portfolio_value: RwLock::new(Decimal::new(1000, 0)),
            }
        }
    }

    #[async_trait::async_trait]
    impl StateProvider for MockStateProvider {
        async fn get_market(&self, _market_id: &MarketId) -> Option<Market> {
            None
        }

        async fn get_all_markets(&self) -> Vec<Market> {
            vec![]
        }

        async fn get_orderbook(&self, _token_id: &TokenId) -> Option<Orderbook> {
            None
        }

        async fn get_price(&self, _token_id: &TokenId) -> Option<Decimal> {
            None
        }

        async fn get_position(&self, _market_id: &MarketId) -> Option<Position> {
            None
        }

        async fn get_all_positions(&self) -> Vec<Position> {
            self.positions.read().await.clone()
        }

        async fn get_price_history(
            &self,
            _token_id: &TokenId,
            _limit: usize,
        ) -> Vec<(chrono::DateTime<chrono::Utc>, Decimal)> {
            vec![]
        }

        async fn get_portfolio_value(&self) -> Decimal {
            *self.portfolio_value.read().await
        }

        async fn get_daily_pnl(&self) -> Decimal {
            *self.daily_pnl.read().await
        }
    }

    fn create_test_state() -> (Arc<ControlState>, Arc<RiskManager>, broadcast::Sender<()>) {
        let risk_manager = Arc::new(RiskManager::new(RiskConfig::default()));
        let state_provider: Arc<dyn StateProvider> = Arc::new(MockStateProvider::new());
        let (shutdown_tx, _) = broadcast::channel(1);

        let control_state = Arc::new(ControlState::new(
            risk_manager.clone(),
            state_provider,
            shutdown_tx.clone(),
        ));

        (control_state, risk_manager, shutdown_tx)
    }

    fn create_test_router(state: Arc<ControlState>) -> Router {
        Router::new()
            .route("/halt", post(halt_handler))
            .route("/resume", post(resume_handler))
            .route("/status", get(status_handler))
            .route("/positions", get(positions_handler))
            .route("/health", get(health_handler))
            .with_state(state)
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let (state, _, _) = create_test_state();
        let router = create_test_router(state);

        let response = router
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_halt_endpoint() {
        let (state, risk_manager, _) = create_test_state();
        let router = create_test_router(state);

        assert!(!risk_manager.is_halted());

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/halt")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"reason": "Test halt"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(risk_manager.is_halted());
    }

    #[tokio::test]
    async fn test_resume_endpoint_requires_confirmation() {
        let (state, risk_manager, _) = create_test_state();
        let router = create_test_router(state);

        // Halt first
        risk_manager.halt("Test");

        // Try to resume without confirmation
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/resume")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(risk_manager.is_halted());

        // Resume with confirmation
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/resume")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"confirmation": "CONFIRM_RESUME"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(!risk_manager.is_halted());
    }

    #[tokio::test]
    async fn test_status_endpoint() {
        let (state, _, _) = create_test_state();
        let router = create_test_router(state);

        let response = router
            .oneshot(Request::builder().uri("/status").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let status: StatusResponse = serde_json::from_slice(&body).unwrap();

        assert!(!status.halted);
        assert_eq!(status.position_count, 0);
    }

    #[tokio::test]
    async fn test_positions_endpoint() {
        let (state, _, _) = create_test_state();
        let router = create_test_router(state);

        let response = router
            .oneshot(
                Request::builder()
                    .uri("/positions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let positions: PositionsResponse = serde_json::from_slice(&body).unwrap();

        assert!(positions.positions.is_empty());
        assert_eq!(positions.total_value, Decimal::ZERO);
    }

    #[tokio::test]
    async fn test_auth_middleware() {
        let risk_manager = Arc::new(RiskManager::new(RiskConfig::default()));
        let state_provider: Arc<dyn StateProvider> = Arc::new(MockStateProvider::new());
        let (shutdown_tx, _) = broadcast::channel(1);

        let state = Arc::new(ControlState::new(
            risk_manager,
            state_provider,
            shutdown_tx,
        ));

        let token = "secret-token".to_string();
        let router = Router::new()
            .route("/status", get(status_handler))
            .route("/health", get(health_handler))
            .with_state(state)
            .layer(middleware::from_fn(move |req, next| {
                auth_middleware(req, next, token.clone())
            }));

        // Health should work without auth
        let response = router
            .clone()
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Status without auth should fail
        let response = router
            .clone()
            .oneshot(Request::builder().uri("/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        // Status with wrong token should fail
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/status")
                    .header("Authorization", "Bearer wrong-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        // Status with correct token should work
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/status")
                    .header("Authorization", "Bearer secret-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
