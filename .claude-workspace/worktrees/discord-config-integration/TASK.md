---
id: discord-config-integration
name: Discord Configuration and AppConfig Integration
wave: 1
priority: 3
dependencies: []
estimated_hours: 2
tags: [backend, config, discord]
---

## Objective

Add Discord webhook configuration to the application config system, following existing TOML-based configuration patterns.

## Context

Polysniper uses TOML configuration files with `serde` deserialization. This task adds Discord-specific configuration options to `AppConfig` and creates a sample configuration section. This is a Wave 1 task that can run independently - it only touches configuration, not the Discord implementation.

## Implementation

1. Add `DiscordConfig` struct to `polysniper-core/src/types.rs`:
   ```rust
   #[derive(Debug, Clone, Deserialize, Default)]
   pub struct DiscordConfig {
       pub enabled: bool,
       pub webhook_url: Option<String>,
       pub notify_on_trade: bool,
       pub notify_on_error: bool,
       pub notify_on_risk_events: bool,
       pub notify_on_connection_status: bool,
       pub dry_run: bool,
       pub rate_limit_per_minute: u32,
   }
   ```

2. Add `discord` field to `AppConfig`:
   ```rust
   pub struct AppConfig {
       // ... existing fields
       pub discord: Option<DiscordConfig>,
   }
   ```

3. Update `config/default.toml` with Discord section:
   ```toml
   [discord]
   enabled = false
   # webhook_url = "https://discord.com/api/webhooks/..."
   notify_on_trade = true
   notify_on_error = true
   notify_on_risk_events = true
   notify_on_connection_status = false
   dry_run = false
   rate_limit_per_minute = 30
   ```

4. Add environment variable support for webhook URL (security):
   - `DISCORD_WEBHOOK_URL` env var override
   - Document in config comments

5. Add config validation in loading:
   - Warn if enabled but no webhook URL
   - Validate URL format if provided

## Acceptance Criteria

- [ ] `DiscordConfig` struct added to types with all fields
- [ ] `AppConfig` includes optional `discord` field
- [ ] Default config file includes commented Discord section
- [ ] Config loads without errors with/without Discord section
- [ ] Environment variable override works for webhook URL
- [ ] Invalid config (enabled=true, no URL) produces warning
- [ ] Existing tests still pass

## Files to Create/Modify

- `crates/polysniper-core/src/types.rs` - Add DiscordConfig struct and AppConfig field
- `config/default.toml` - Add Discord configuration section
- `src/main.rs` - Add env var loading for webhook URL (optional)

## Integration Points

- **Provides**: `DiscordConfig` type, configuration loading
- **Consumes**: Existing `serde` and config patterns
- **Conflicts**: Touches `types.rs` - coordinate if other tasks modify it

## Reference Files

- `/Users/jacklau/Polysniper/crates/polysniper-core/src/types.rs` - Existing config types
- `/Users/jacklau/Polysniper/config/default.toml` - Existing configuration
- `/Users/jacklau/Polysniper/src/main.rs` - Config loading pattern
