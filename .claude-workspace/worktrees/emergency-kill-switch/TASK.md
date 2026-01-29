---
id: emergency-kill-switch
name: Emergency Kill Switch
wave: 1
priority: 1
dependencies: []
estimated_hours: 5
tags: [risk, emergency, http-api, signals]
---

## Objective

Implement an emergency kill switch with HTTP endpoint and signal handler for immediate shutdown of all trading activity.

## Context

The current system has a basic circuit breaker based on daily losses, but lacks manual override capability. This task adds:
- HTTP API endpoint for remote halt/resume
- Unix signal handlers (SIGTERM, SIGUSR1) for CLI control
- Status endpoint for monitoring
- Graceful shutdown with position reporting

The codebase already has axum and tower dependencies available but not used. This task creates a minimal HTTP server for control operations.

## Implementation

1. **Create control server module** in `crates/polysniper-risk/src/control.rs`:
   - Define `ControlServer` struct wrapping axum router
   - Implement endpoints: `/halt`, `/resume`, `/status`, `/positions`
   - Share state with RiskManager via Arc

2. **Add control config** to `RiskConfig` in `crates/polysniper-core/src/types.rs`:
   ```rust
   pub struct ControlConfig {
       pub enabled: bool,
       pub port: u16,                    // Default 9876
       pub host: String,                 // Default "127.0.0.1"
       pub auth_token: Option<String>,   // Optional bearer token
       pub signal_handlers: bool,        // Enable SIGUSR1/SIGUSR2 handlers
   }
   ```

3. **Implement HTTP endpoints** in `control.rs`:
   ```rust
   // POST /halt - Immediately halt all trading
   // POST /resume - Resume trading (requires confirmation)
   // GET /status - Return current state (halted, positions, P&L)
   // GET /positions - Return all open positions
   // POST /close-all - Signal to close all positions (dry-run safe)
   ```

4. **Implement signal handlers** in `control.rs`:
   - `SIGUSR1` - Toggle halt/resume
   - `SIGUSR2` - Print status to logs
   - `SIGTERM` - Graceful shutdown with status dump
   - Use `tokio::signal` for async signal handling

5. **Integrate with RiskManager** (`crates/polysniper-risk/src/validator.rs`):
   - Control server gets reference to RiskManager's `halted` flag
   - `/halt` sets the atomic bool that blocks all trading
   - `/resume` clears it with logging

6. **Integrate into main.rs**:
   - Spawn control server task alongside ws_manager
   - Register signal handlers in main runtime
   - Log control server URL on startup

7. **Add config section** in `config/default.toml`:
   ```toml
   [control]
   enabled = true
   port = 9876
   host = "127.0.0.1"
   auth_token = null  # Set via CONTROL_AUTH_TOKEN env var
   signal_handlers = true
   ```

8. **Add unit tests** in `crates/polysniper-risk/src/control.rs`:
   - Test halt/resume state changes
   - Test endpoint responses
   - Test auth token validation

## Acceptance Criteria

- [ ] HTTP server starts on configured port
- [ ] `/halt` endpoint immediately stops all trading
- [ ] `/resume` endpoint re-enables trading
- [ ] `/status` returns current state and key metrics
- [ ] Signal handlers work for SIGUSR1/SIGUSR2
- [ ] Optional auth token protects endpoints
- [ ] Server gracefully shuts down on SIGTERM
- [ ] All tests passing
- [ ] Feature can be disabled via config

## Files to Create/Modify

- `crates/polysniper-risk/src/control.rs` - Create control server module
- `crates/polysniper-risk/src/lib.rs` - Export control module
- `crates/polysniper-risk/Cargo.toml` - Add axum, tower dependencies (move from workspace)
- `crates/polysniper-core/src/types.rs` - Add ControlConfig struct
- `src/main.rs` - Spawn control server and register signal handlers
- `config/default.toml` - Add control config section

## Integration Points

- **Provides**: HTTP control interface, signal-based control
- **Consumes**: `RiskManager::halt()`, `RiskManager::resume()`, `RiskManager::is_halted()`
- **Conflicts**: Coordinates with main.rs app lifecycle - be careful with shutdown order
