//! Webhook server for receiving ML model predictions
//!
//! Provides an HTTP endpoint for external ML models to send predictions
//! which are converted to ExternalSignal events for strategy consumption.

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use polysniper_core::{EventBus, ExternalSignalEvent, SignalSource, SystemEvent, WebhookConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{error, info};

/// ML prediction request from external models
#[derive(Debug, Clone, Deserialize)]
pub struct MlPredictionRequest {
    /// Market ID (condition_id) for the prediction
    pub market_id: String,
    /// Prediction direction: "up", "down", "buy", "sell", or numeric value
    pub prediction: String,
    /// Confidence score from 0.0 to 1.0
    pub confidence: f64,
    /// Optional model identifier
    #[serde(default)]
    pub model_id: Option<String>,
    /// Optional additional metadata
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Webhook response
#[derive(Debug, Clone, Serialize)]
pub struct WebhookResponse {
    /// Status: "success" or "error"
    pub status: String,
    /// Human-readable message
    pub message: String,
    /// Optional signal ID if successfully created
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_id: Option<String>,
}

impl WebhookResponse {
    fn success(message: impl Into<String>, signal_id: Option<String>) -> Self {
        Self {
            status: "success".to_string(),
            message: message.into(),
            signal_id,
        }
    }

    fn error(message: impl Into<String>) -> Self {
        Self {
            status: "error".to_string(),
            message: message.into(),
            signal_id: None,
        }
    }
}

/// Shared state for webhook handlers
#[derive(Clone)]
pub struct WebhookState<E: EventBus + Send + Sync + 'static> {
    event_bus: Arc<E>,
    api_key: Option<String>,
}

/// Webhook server that receives ML predictions
pub struct WebhookServer<E: EventBus + Send + Sync + Clone + 'static> {
    config: WebhookConfig,
    event_bus: Arc<E>,
}

impl<E: EventBus + Send + Sync + Clone + 'static> WebhookServer<E> {
    /// Create a new webhook server
    pub fn new(config: WebhookConfig, event_bus: Arc<E>) -> Self {
        Self { config, event_bus }
    }

    /// Build the router with all routes
    fn build_router(&self) -> Router {
        let state = WebhookState {
            event_bus: self.event_bus.clone(),
            api_key: self.config.api_key.clone(),
        };

        let ml_route = post(handle_ml_prediction::<E>);

        // Apply API key middleware if configured
        if self.config.api_key.is_some() {
            Router::new()
                .route("/webhook/ml", ml_route)
                .layer(middleware::from_fn_with_state(
                    state.clone(),
                    api_key_middleware::<E>,
                ))
                .route("/health", get(handle_health))
                .with_state(state)
        } else {
            Router::new()
                .route("/webhook/ml", ml_route)
                .route("/health", get(handle_health))
                .with_state(state)
        }
    }

    /// Start the webhook server
    pub async fn run(self) -> anyhow::Result<()> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = TcpListener::bind(&addr).await?;

        info!(
            address = %addr,
            api_key_required = self.config.api_key.is_some(),
            "Webhook server started"
        );

        let router = self.build_router();

        axum::serve(listener, router).await?;

        Ok(())
    }

    /// Start the webhook server and return a handle
    pub async fn start(self) -> anyhow::Result<tokio::task::JoinHandle<()>> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = TcpListener::bind(&addr).await?;

        info!(
            address = %addr,
            api_key_required = self.config.api_key.is_some(),
            "Webhook server started"
        );

        let router = self.build_router();

        let handle = tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, router).await {
                error!(error = %e, "Webhook server error");
            }
        });

        Ok(handle)
    }
}

/// API key authentication middleware
async fn api_key_middleware<E: EventBus + Send + Sync + 'static>(
    State(state): State<WebhookState<E>>,
    request: Request,
    next: Next,
) -> Response {
    if let Some(expected_key) = &state.api_key {
        let provided_key = request
            .headers()
            .get("x-api-key")
            .and_then(|v| v.to_str().ok());

        match provided_key {
            Some(key) if key == expected_key => next.run(request).await,
            Some(_) => (
                StatusCode::UNAUTHORIZED,
                Json(WebhookResponse::error("Invalid API key")),
            )
                .into_response(),
            None => (
                StatusCode::UNAUTHORIZED,
                Json(WebhookResponse::error("Missing X-API-Key header")),
            )
                .into_response(),
        }
    } else {
        next.run(request).await
    }
}

/// Handle ML prediction requests
async fn handle_ml_prediction<E: EventBus + Send + Sync + 'static>(
    State(state): State<WebhookState<E>>,
    Json(payload): Json<MlPredictionRequest>,
) -> Result<Json<WebhookResponse>, (StatusCode, Json<WebhookResponse>)> {
    // Validate required fields
    if payload.market_id.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(WebhookResponse::error("market_id is required")),
        ));
    }

    if payload.prediction.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(WebhookResponse::error("prediction is required")),
        ));
    }

    if !(0.0..=1.0).contains(&payload.confidence) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(WebhookResponse::error(
                "confidence must be between 0.0 and 1.0",
            )),
        ));
    }

    // Generate signal ID
    let signal_id = uuid::Uuid::new_v4().to_string();

    // Build metadata
    let mut metadata = payload.metadata.unwrap_or(serde_json::json!({}));
    if let Some(obj) = metadata.as_object_mut() {
        obj.insert("model_id".to_string(), serde_json::json!(payload.model_id));
        obj.insert(
            "confidence".to_string(),
            serde_json::json!(payload.confidence),
        );
        obj.insert("signal_id".to_string(), serde_json::json!(&signal_id));
    }

    // Create external signal event
    let event = SystemEvent::ExternalSignal(ExternalSignalEvent {
        source: SignalSource::Webhook {
            endpoint: "/webhook/ml".to_string(),
        },
        signal_type: "ml_prediction".to_string(),
        content: payload.prediction.clone(),
        market_id: Some(payload.market_id.clone()),
        keywords: vec![
            "ml".to_string(),
            "prediction".to_string(),
            payload.prediction.clone(),
        ],
        metadata,
        received_at: Utc::now(),
    });

    // Publish to event bus
    state.event_bus.publish(event);

    info!(
        market_id = %payload.market_id,
        prediction = %payload.prediction,
        confidence = %payload.confidence,
        model_id = ?payload.model_id,
        signal_id = %signal_id,
        "ML prediction received"
    );

    Ok(Json(WebhookResponse::success(
        "Prediction received and published",
        Some(signal_id),
    )))
}

/// Health check endpoint
async fn handle_health<E: EventBus + Send + Sync + 'static>(
    State(_state): State<WebhookState<E>>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "polysniper-webhook",
        "timestamp": Utc::now().to_rfc3339()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_prediction_request_parsing() {
        let json = r#"{
            "market_id": "0x123",
            "prediction": "up",
            "confidence": 0.85
        }"#;

        let request: MlPredictionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.market_id, "0x123");
        assert_eq!(request.prediction, "up");
        assert!((request.confidence - 0.85).abs() < 0.001);
        assert!(request.model_id.is_none());
        assert!(request.metadata.is_none());
    }

    #[test]
    fn test_ml_prediction_request_with_optional_fields() {
        let json = r#"{
            "market_id": "0x456",
            "prediction": "sell",
            "confidence": 0.92,
            "model_id": "gpt-4-predictor",
            "metadata": {"source": "twitter", "tweet_id": "123456"}
        }"#;

        let request: MlPredictionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.market_id, "0x456");
        assert_eq!(request.prediction, "sell");
        assert!((request.confidence - 0.92).abs() < 0.001);
        assert_eq!(request.model_id, Some("gpt-4-predictor".to_string()));
        assert!(request.metadata.is_some());
        let meta = request.metadata.unwrap();
        assert_eq!(meta["source"], "twitter");
    }

    #[test]
    fn test_webhook_response_success() {
        let response = WebhookResponse::success("Test message", Some("sig-123".to_string()));
        assert_eq!(response.status, "success");
        assert_eq!(response.message, "Test message");
        assert_eq!(response.signal_id, Some("sig-123".to_string()));
    }

    #[test]
    fn test_webhook_response_error() {
        let response = WebhookResponse::error("Something went wrong");
        assert_eq!(response.status, "error");
        assert_eq!(response.message, "Something went wrong");
        assert!(response.signal_id.is_none());
    }

    #[test]
    fn test_confidence_validation_bounds() {
        // Test that we handle edge cases for confidence bounds
        let valid_low = 0.0;
        let valid_high = 1.0;
        let invalid_low = -0.1;
        let invalid_high = 1.1;

        assert!((0.0..=1.0).contains(&valid_low));
        assert!((0.0..=1.0).contains(&valid_high));
        assert!(!(0.0..=1.0).contains(&invalid_low));
        assert!(!(0.0..=1.0).contains(&invalid_high));
    }
}
