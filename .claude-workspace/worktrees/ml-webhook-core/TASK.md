---
id: ml-webhook-core
name: ML Webhook Core - HTTP Endpoint for Model Predictions
wave: 1
priority: 1
dependencies: []
estimated_hours: 4
tags: [backend, api, ml-integration]
---

## Objective

Create an Axum-based HTTP webhook endpoint to receive ML model predictions and convert them to ExternalSignal events.

## Context

The Polysniper codebase already has Axum 0.8 as a dependency and the `ExternalSignal` event type with `SignalSource::Webhook` variant. This task implements the actual HTTP server and endpoint handlers to receive predictions from external ML models.

## Implementation

1. Create `/crates/polysniper-data/src/webhook_server.rs`:
   - Axum router with `/webhook/ml` POST endpoint
   - Request validation and parsing
   - Convert incoming payloads to `ExternalSignal` events
   - Publish events via `EventBus`

2. Add webhook configuration to `/crates/polysniper-core/src/types.rs`:
   - `WebhookConfig` struct with host, port, auth settings
   - Add to `AppConfig`

3. Create request/response types:
   - `MlPredictionRequest` with market_id, prediction, confidence, model_id, metadata
   - `WebhookResponse` with status and message

4. Integrate server startup in main.rs

## Acceptance Criteria

- [ ] POST `/webhook/ml` endpoint accepts ML predictions
- [ ] Request body validated (required fields: market_id, prediction, confidence)
- [ ] Optional API key authentication via header
- [ ] Predictions converted to `ExternalSignal` with `SignalSource::Webhook`
- [ ] Events published to EventBus for strategy consumption
- [ ] Configurable host/port via TOML config
- [ ] Health endpoint at GET `/health`
- [ ] Errors return appropriate HTTP status codes
- [ ] Unit tests for request parsing

## Files to Create/Modify

- `crates/polysniper-data/src/webhook_server.rs` - **CREATE** - Webhook server implementation
- `crates/polysniper-data/src/lib.rs` - **MODIFY** - Export webhook module
- `crates/polysniper-core/src/types.rs` - **MODIFY** - Add WebhookConfig
- `config/default.toml` - **MODIFY** - Add webhook configuration section
- `src/main.rs` - **MODIFY** - Start webhook server

## Integration Points

- **Provides**: HTTP endpoint for external ML models to send predictions
- **Consumes**: EventBus for publishing ExternalSignal events
- **Conflicts**: None - new file creation
