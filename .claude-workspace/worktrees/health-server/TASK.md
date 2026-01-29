---
id: health-server
name: Health Check HTTP Endpoint
wave: 1
priority: 1
dependencies: []
estimated_hours: 4
tags: [backend, http, monitoring]
---

## Objective

Create an HTTP server exposing health check and status endpoints for monitoring and orchestration systems.

## Context

Polysniper has no external visibility into its runtime state. Orchestration systems (Kubernetes, Docker, systemd) need health endpoints to determine if the service is alive and functioning. The codebase already has:
- `axum 0.8` in workspace dependencies (unused)
- `tower` and `tower-http` for middleware
- `prometheus 0.13` for metrics (unused)
- Existing `HeartbeatEvent` and `ConnectionStatusEvent` infrastructure
- `RiskManager` with `is_halted()` method
- `MarketCache` with market count, position, and P&L data

## Implementation

1. **Create health types** in `crates/polysniper-core/src/types.rs`:
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct HealthConfig {
       pub enabled: bool,
       pub bind_address: String,  // default: "127.0.0.1:8080"
   }
   ```
   Add `health: HealthConfig` field to `AppConfig`

2. **Create health server crate** or module:
   - Option A: New crate `polysniper-health` (recommended for separation)
   - Option B: Add to `polysniper-observability`
   
   For simplicity, add to `polysniper-observability`.

3. **Create health server module** in `crates/polysniper-observability/src/health.rs`:
   - `HealthServer` struct holding:
     - Bind address
     - Shared references to state (Arc)
     - Shutdown signal receiver
   - `HealthStatus` response struct:
     - `status`: "healthy" | "degraded" | "unhealthy"
     - `trading_enabled`: bool (inverse of risk manager halt)
     - `websocket_connected`: bool
     - `markets_loaded`: u32
     - `uptime_secs`: u64
     - `last_event_timestamp`: Option<DateTime>
   - Routes:
     - `GET /health` - Basic liveness (200 OK if server running)
     - `GET /health/ready` - Readiness (checks WS connection, markets loaded)
     - `GET /health/status` - Detailed JSON status
     - `GET /metrics` - Prometheus metrics (optional stretch goal)

4. **Implement Axum router** in health module:
   ```rust
   pub fn create_router(state: AppState) -> Router {
       Router::new()
           .route("/health", get(health_check))
           .route("/health/ready", get(readiness_check))
           .route("/health/status", get(status_check))
           .with_state(state)
   }
   ```

5. **Add config parsing** in `config/default.toml`:
   ```toml
   [health]
   enabled = true
   bind_address = "127.0.0.1:8080"
   ```

6. **Update Cargo.toml** for polysniper-observability:
   - Add axum, tower-http dependencies
   - Enable serde feature for response types

## Acceptance Criteria

- [ ] `HealthConfig` added to core types
- [ ] Health server module created in polysniper-observability
- [ ] `GET /health` returns 200 OK when server is running
- [ ] `GET /health/ready` returns 200 when WS connected and markets loaded, 503 otherwise
- [ ] `GET /health/status` returns detailed JSON with all status fields
- [ ] Server binds to configurable address (default 127.0.0.1:8080)
- [ ] Server can be disabled via config (`enabled = false`)
- [ ] Graceful shutdown on application termination
- [ ] No panics on missing state data
- [ ] `cargo clippy` passes with no warnings

## Files to Create/Modify

- `crates/polysniper-core/src/types.rs` - Add HealthConfig struct and to AppConfig
- `crates/polysniper-observability/Cargo.toml` - Add axum, tower-http deps
- `crates/polysniper-observability/src/health.rs` - New health server module
- `crates/polysniper-observability/src/lib.rs` - Export health module
- `config/default.toml` - Add [health] section

## Integration Points

- **Provides**: `HealthServer`, health check routes, `HealthStatus` response type
- **Consumes**: `StateProvider` trait, `RiskValidator` trait (for halt status)
- **Conflicts**: None - new module, config section addition only

## Technical Notes

- Use `axum::extract::State` for shared state access
- Use `tokio::select!` for graceful shutdown with shutdown signal
- Health check should be fast (no heavy computations)
- Readiness check should verify essential systems (WS, markets)
- Status endpoint can include more detail (strategy states, risk limits)
- Consider adding request tracing via tower-http TracingLayer
