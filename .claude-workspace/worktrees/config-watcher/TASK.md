---
id: config-watcher
name: Config File Watcher Service
wave: 1
priority: 1
dependencies: []
estimated_hours: 4
tags: [backend, config, async]
---

## Objective

Create a file system watcher service that monitors config files and emits events when they change.

## Context

Polysniper currently loads configuration once at startup from TOML files. This task creates the infrastructure to detect config file changes at runtime. The watcher will monitor both the main config (`config/default.toml`) and strategy configs (`config/strategies/*.toml`), emitting structured events that can trigger reload logic.

The codebase already uses:
- Tokio async runtime
- `notify` crate is NOT present (needs to be added)
- Broadcast event bus pattern for event distribution
- `Arc<RwLock<>>` for shared state

## Implementation

1. **Add `notify` dependency** to `polysniper-core/Cargo.toml`:
   ```toml
   notify = { version = "6.1", default-features = false, features = ["macos_kqueue"] }
   ```

2. **Create config events** in `crates/polysniper-core/src/events.rs`:
   - Add `ConfigChanged(ConfigChangedEvent)` variant to `SystemEvent`
   - Define `ConfigChangedEvent` with fields: `path`, `config_type` (Main/Strategy), `timestamp`
   - Define `ConfigType` enum: `Main`, `Strategy(String)`

3. **Create config watcher module** in `crates/polysniper-core/src/config_watcher.rs`:
   - `ConfigWatcher` struct with:
     - Config directory paths
     - Event sender (broadcast channel)
     - Debounce duration (avoid rapid re-fires)
   - `new(config_dir, strategies_dir, event_tx)` constructor
   - `async fn run(&self)` - main watch loop
   - File change debouncing (500ms default)
   - Error handling for missing files/directories

4. **Add module export** in `crates/polysniper-core/src/lib.rs`

5. **Add tests** in `crates/polysniper-core/src/config_watcher.rs`:
   - Test file modification detection
   - Test debounce behavior
   - Test multiple file changes

## Acceptance Criteria

- [ ] `notify` crate added to dependencies
- [ ] `ConfigChangedEvent` defined in events module
- [ ] `ConfigWatcher` struct with async `run()` method
- [ ] Watches `config/default.toml` for main config changes
- [ ] Watches `config/strategies/*.toml` for strategy config changes
- [ ] Debounces rapid file changes (configurable, default 500ms)
- [ ] Emits `ConfigChanged` events via broadcast channel
- [ ] Handles missing files gracefully (logs warning, continues)
- [ ] Unit tests pass
- [ ] `cargo clippy` passes with no warnings

## Files to Create/Modify

- `crates/polysniper-core/Cargo.toml` - Add notify dependency
- `crates/polysniper-core/src/events.rs` - Add ConfigChangedEvent
- `crates/polysniper-core/src/config_watcher.rs` - New config watcher module
- `crates/polysniper-core/src/lib.rs` - Export new module

## Integration Points

- **Provides**: `ConfigWatcher` service, `ConfigChangedEvent` event type
- **Consumes**: Broadcast event bus (existing pattern)
- **Conflicts**: None - new module with no file overlap

## Technical Notes

- Use `notify::RecommendedWatcher` with `EventKind::Modify` filter
- Implement debouncing using `tokio::time::sleep` and tracking last event time
- Config type detection: parse path to determine if main config or strategy
- Strategy name extraction: derive from filename (e.g., `price_spike.toml` -> "price_spike")
