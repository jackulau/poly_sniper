---
id: hot-reload-integration
name: Hot Reload Integration with Strategies
wave: 2
priority: 2
dependencies: [config-watcher]
estimated_hours: 5
tags: [backend, strategies, integration]
---

## Objective

Integrate the config watcher with strategies and risk manager to enable runtime parameter updates without restart.

## Context

This task connects the `ConfigWatcher` (from config-watcher task) to the application's runtime components. When a config file changes, the affected components should reload their parameters. The codebase patterns that support this:
- Strategies use `Arc<RwLock<Config>>` pattern for internal state
- `RiskManager` has `Arc<RwLock<>>` for halt reason
- `Strategy` trait has `set_enabled()` for runtime enable/disable
- Event bus pattern allows decoupled notification

Key challenge: Strategy configs are loaded at construction time and not retained for updates. Need to add a `reload_config()` method to the Strategy trait.

## Implementation

1. **Extend Strategy trait** in `crates/polysniper-core/src/traits.rs`:
   ```rust
   #[async_trait]
   pub trait Strategy: Send + Sync {
       // ... existing methods ...
       
       /// Reload configuration from provided TOML content
       /// Returns Ok(()) on success, Err if config is invalid
       async fn reload_config(&mut self, config_content: &str) -> Result<(), StrategyError>;
       
       /// Get the strategy's config file name (e.g., "price_spike")
       fn config_name(&self) -> &str;
   }
   ```

2. **Implement reload_config** for each strategy in `crates/polysniper-strategies/src/`:
   - `target_price.rs`: Parse new `TargetPriceConfig`, update internal targets
   - `price_spike.rs`: Parse new `PriceSpikeConfig`, update thresholds/cooldowns
   - `new_market.rs`: Parse new `NewMarketConfig`, update filters
   - `event_based.rs`: Parse new `EventBasedConfig`, update rules/mappings

3. **Add reloadable config to RiskManager** in `crates/polysniper-risk/src/validator.rs`:
   ```rust
   impl RiskManager {
       pub fn update_config(&self, config: RiskConfig) {
           // Update internal Arc<RwLock<RiskConfig>>
       }
   }
   ```
   - Wrap `RiskConfig` in `Arc<RwLock<>>` instead of owned value
   - Add `update_config()` method

4. **Create config reload handler** in `src/main.rs`:
   - Subscribe to `ConfigChangedEvent` from event bus
   - On main config change: reload and update RiskManager, ExecutionConfig
   - On strategy config change: find matching strategy by name, call `reload_config()`
   - Log success/failure of reload attempts

5. **Spawn config watcher** in main event loop:
   ```rust
   // In App::run()
   let config_watcher = ConfigWatcher::new(
       "config".into(),
       "config/strategies".into(),
       event_bus.clone(),
   );
   tokio::spawn(async move { config_watcher.run().await });
   ```

6. **Add config reload to event processing** in main loop:
   ```rust
   SystemEvent::ConfigChanged(event) => {
       self.handle_config_change(event).await;
   }
   ```

7. **Validation and error handling**:
   - Parse config before applying (don't update on invalid config)
   - Log which config was reloaded and what changed
   - Emit metrics/events for monitoring config reloads

## Acceptance Criteria

- [ ] `Strategy` trait extended with `reload_config()` and `config_name()` methods
- [ ] All 4 strategies implement `reload_config()` correctly
- [ ] `RiskManager` supports runtime config updates
- [ ] `ConfigWatcher` spawned in main event loop
- [ ] `ConfigChangedEvent` handled in main event processing
- [ ] Strategy parameter changes take effect within 1 second
- [ ] Risk config changes (limits, circuit breaker) apply immediately
- [ ] Invalid config files are rejected with error log (no crash)
- [ ] Successful reloads logged at INFO level
- [ ] Integration tests verify hot reload works end-to-end
- [ ] `cargo clippy` passes with no warnings

## Files to Create/Modify

- `crates/polysniper-core/src/traits.rs` - Extend Strategy trait
- `crates/polysniper-strategies/src/target_price.rs` - Implement reload_config
- `crates/polysniper-strategies/src/price_spike.rs` - Implement reload_config
- `crates/polysniper-strategies/src/new_market.rs` - Implement reload_config
- `crates/polysniper-strategies/src/event_based.rs` - Implement reload_config
- `crates/polysniper-risk/src/validator.rs` - Add update_config method
- `src/main.rs` - Spawn config watcher, handle ConfigChangedEvent

## Integration Points

- **Provides**: Hot reload capability for strategies and risk config
- **Consumes**: `ConfigWatcher` from config-watcher task, `ConfigChangedEvent`
- **Conflicts**: Modifies Strategy trait (coordinate with any other strategy work)

## Technical Notes

- Use `toml::from_str()` for parsing config content
- Strategy `reload_config` should be idempotent (safe to call multiple times)
- Consider adding a `ConfigReloadEvent` to notify successful reloads
- RiskManager config update should NOT reset rate limiting state
- Keep track of last successful config hash to avoid no-op reloads
- Strategies should validate config before applying (e.g., positive values)
