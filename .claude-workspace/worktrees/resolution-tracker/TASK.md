---
id: resolution-tracker
name: Market Resolution Tracker - Monitor Resolution Events
wave: 1
priority: 1
dependencies: []
estimated_hours: 4
tags: [backend, data-source, resolution]
---

## Objective

Create a service that monitors markets approaching resolution and emits `MarketStateChange` events when markets transition to Resolved state.

## Context

The codebase has `MarketStateChange` event with `Resolved` state but no active monitoring for when markets resolve. The Gamma API provides market end dates and resolution status. This task adds proactive resolution detection.

## Implementation

1. Create `/crates/polysniper-data/src/resolution_monitor.rs`:
   - Track markets with positions (from StateProvider)
   - Poll Gamma API for market status changes
   - Configurable poll interval (default 30s)
   - Emit `MarketStateChange` when status changes

2. Create `/crates/polysniper-core/src/resolution.rs`:
   - `ResolutionConfig` with thresholds and timing
   - `ResolutionStatus` enum (Far, Approaching, Imminent, Resolved)
   - Helper functions for time-to-resolution calculations

3. Enhance GammaClient:
   - Add `fetch_market_status(market_id)` method
   - Parse resolution timestamp from response
   - Return current market state

4. Add resolution warnings:
   - Emit events when market is approaching resolution
   - Configurable warning thresholds (24h, 1h, 15m)

## Acceptance Criteria

- [ ] Monitors all markets where user has positions
- [ ] Detects market resolution within poll interval
- [ ] Emits `MarketStateChange` event on resolution
- [ ] Emits warning events at configurable thresholds
- [ ] Tracks time-to-resolution for each position
- [ ] Handles markets without clear end dates gracefully
- [ ] Configurable poll interval
- [ ] Unit tests for resolution calculations

## Files to Create/Modify

- `crates/polysniper-data/src/resolution_monitor.rs` - **CREATE** - Resolution monitoring service
- `crates/polysniper-data/src/lib.rs` - **MODIFY** - Export resolution module
- `crates/polysniper-data/src/gamma_client.rs` - **MODIFY** - Add status fetch method
- `crates/polysniper-core/src/resolution.rs` - **CREATE** - Resolution types and config
- `crates/polysniper-core/src/lib.rs` - **MODIFY** - Export resolution module
- `config/default.toml` - **MODIFY** - Add resolution tracking config

## Integration Points

- **Provides**: Market resolution events and warnings
- **Consumes**: StateProvider for positions, GammaClient for status
- **Conflicts**: None - new file creation primarily
