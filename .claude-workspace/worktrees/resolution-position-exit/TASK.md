---
id: resolution-position-exit
name: Resolution Position Exit - Auto-Exit Before Market Resolution
wave: 2
priority: 2
dependencies: [resolution-tracker]
estimated_hours: 4
tags: [backend, strategy, risk-management]
---

## Objective

Create a strategy that automatically exits positions before market resolution based on configurable time thresholds and risk rules.

## Context

Once resolution-tracker monitors market states and emits warning events, this task implements the logic to auto-exit positions. This prevents positions from being held through resolution when outcome is uncertain.

## Implementation

1. Create `/crates/polysniper-strategies/src/resolution_exit.rs`:
   - New strategy implementing `Strategy` trait
   - Listens to `MarketStateChange` and resolution warning events
   - Generates exit signals when thresholds reached
   - Configurable exit timing (e.g., 1h before resolution)

2. Create exit decision logic:
   - Time-based: Exit X minutes before resolution
   - P&L-based: Exit if unrealized P&L < threshold
   - Configurable per-market overrides
   - Priority: HIGH for resolution exits

3. Add position exit configuration:
   - Default exit time before resolution
   - Per-market override settings
   - Exit order type (market vs limit)
   - P&L floor triggering early exit

4. Integrate with risk management:
   - Notify RiskManager of pending exits
   - Update position tracking post-exit
   - Log exit reasons for analysis

## Acceptance Criteria

- [ ] Monitors positions approaching resolution
- [ ] Auto-generates exit signals at configured threshold
- [ ] Supports market-specific exit timing overrides
- [ ] P&L-based early exit option
- [ ] Exit signals use HIGH priority for fast execution
- [ ] Tracks exit success/failure
- [ ] Logs all auto-exits with reasons
- [ ] Configurable exit order type (FOK recommended)
- [ ] Unit tests for exit timing calculations
- [ ] Does not exit positions with explicit "hold through" flag

## Files to Create/Modify

- `crates/polysniper-strategies/src/resolution_exit.rs` - **CREATE** - Resolution exit strategy
- `crates/polysniper-strategies/src/lib.rs` - **MODIFY** - Export resolution module
- `crates/polysniper-core/src/resolution.rs` - **MODIFY** - Add exit config types
- `config/strategies/resolution_exit.toml` - **CREATE** - Exit strategy config

## Integration Points

- **Provides**: Automatic position exits before resolution
- **Consumes**: MarketStateChange events from resolution-tracker
- **Conflicts**: May modify resolution.rs (coordinate with resolution-tracker)
