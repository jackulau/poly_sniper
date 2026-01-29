---
id: main-integration
name: Main Application Integration
wave: 3
priority: 1
dependencies: [llm-prediction-strategy, llm-strategy-config]
estimated_hours: 2
tags: [backend, integration]
---

## Objective

Integrate the LLM prediction strategy into the main application, including config loading, strategy registration, and heartbeat event emission.

## Context

This task brings together the OpenRouter client, LLM prediction strategy, and configuration to make the feature fully functional. It modifies main.rs to load and register the new strategy.

## Implementation

### 1. Update `src/main.rs`

**Add imports:**
```rust
use polysniper_strategies::{
    // ... existing imports ...
    LlmPredictionConfig, LlmPredictionStrategy,
};
```

**Update `load_strategies()` function:**

Add after existing strategy loading (around line ~200):
```rust
// Load LLM Prediction Strategy
let llm_prediction_config = Self::load_strategy_config::<LlmPredictionConfig>("llm_prediction")?
    .unwrap_or_default();
if llm_prediction_config.enabled {
    match LlmPredictionStrategy::new(llm_prediction_config) {
        Ok(strategy) => {
            strategies.push(Box::new(strategy));
            info!("Loaded LLM Prediction Strategy");
        }
        Err(e) => {
            warn!("Failed to initialize LLM Prediction Strategy: {}. Skipping.", e);
        }
    }
}
```

**Ensure Heartbeat event emission exists:**

Check if heartbeat timer already exists. If not, add in the main event loop (around the `tokio::select!` block):

```rust
// Add heartbeat interval if not present
let heartbeat_interval = tokio::time::interval(Duration::from_secs(60));

// In the select! block:
_ = heartbeat_interval.tick() => {
    let heartbeat = SystemEvent::Heartbeat(HeartbeatEvent {
        source: "main".to_string(),
        timestamp: Utc::now(),
    });
    if let Err(e) = event_bus.publish(heartbeat).await {
        warn!("Failed to publish heartbeat: {}", e);
    }
}
```

**Note:** The heartbeat interval (60s) is shorter than the strategy's analysis interval (300s). The strategy itself tracks when to run analysis, so it won't analyze on every heartbeat.

### 2. Verify Dependencies in `Cargo.toml`

Ensure root Cargo.toml has polysniper-strategies as a workspace member and src/main.rs imports work.

## Acceptance Criteria

- [ ] `LlmPredictionConfig` imported from polysniper-strategies
- [ ] `LlmPredictionStrategy` imported from polysniper-strategies
- [ ] Config loaded from `config/strategies/llm_prediction.toml`
- [ ] Strategy registered in `load_strategies()` when enabled
- [ ] Graceful handling if strategy initialization fails (warn + skip, don't crash)
- [ ] Heartbeat events emitted at regular interval (60s or existing)
- [ ] Application builds: `cargo build`
- [ ] Application runs without errors with strategy disabled
- [ ] Application runs without errors with strategy enabled (requires API key)

## Files to Create/Modify

- `src/main.rs` - **Modify** - Add imports, config loading, strategy registration, heartbeat

## Integration Points

- **Provides**: Full LLM prediction strategy integration
- **Consumes**: `LlmPredictionConfig`, `LlmPredictionStrategy` from strategies crate
- **Conflicts**: Modifies main.rs - coordinate with any other main.rs changes

## Testing Verification

1. **Build test:**
   ```bash
   cargo build
   ```

2. **Run with strategy disabled (no API key needed):**
   ```bash
   cargo run
   ```
   Expected: No errors, strategy not loaded (since `enabled = false`)

3. **Run with strategy enabled:**
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-..."
   # Edit config/strategies/llm_prediction.toml: enabled = true
   cargo run
   ```
   Expected logs:
   - "Loaded LLM Prediction Strategy"
   - "Initializing LLM prediction strategy"
   - Heartbeat events published
   - LLM analysis runs every 300 seconds (or configured interval)

4. **Run with dry_run enabled:**
   Ensure `config/default.toml` has `dry_run = true` in execution section.
   Expected: Trade signals logged but no actual orders submitted.

## Notes

- The heartbeat interval in main.rs should be shorter than the strategy's analysis_interval_secs
- The strategy internally tracks when to run analysis, so heartbeats just provide the "tick"
- If OPENROUTER_API_KEY is not set when strategy is enabled, it should fail gracefully at initialization
