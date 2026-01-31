---
id: kelly-criterion-sizing
name: Kelly Criterion Position Sizing
wave: 1
priority: 1
dependencies: []
estimated_hours: 4
tags: [risk, sizing, performance]
---

## Objective

Implement Kelly criterion position sizing to optimize bet sizes based on historical edge and win rate, maximizing long-term growth while managing risk.

## Context

The codebase already has volatility-based position sizing in `polysniper-risk/src/volatility.rs`. Kelly criterion provides a mathematically optimal sizing formula based on:
- Win rate (probability of winning)
- Average win/loss ratio
- Edge calculation

This complements volatility sizing by adding edge-aware position management. The formula `f* = (bp - q) / b` determines optimal fraction where:
- b = odds (average win / average loss)
- p = probability of winning
- q = probability of losing (1 - p)

## Implementation

1. Create `/crates/polysniper-risk/src/kelly.rs`:
   - `KellyCalculator` struct with configurable parameters
   - `KellyConfig` with fractional Kelly support (half-Kelly, quarter-Kelly)
   - Calculate optimal position fraction from trade history
   - Rolling window for recent trades (configurable, e.g., last 50-100 trades)
   - Minimum sample size before Kelly kicks in (e.g., 20 trades)

2. Integrate with `RiskManager` in `/crates/polysniper-risk/src/validator.rs`:
   - Add Kelly sizing as optional adjustment after volatility sizing
   - Apply Kelly fraction as multiplier to calculated size
   - Respect existing min/max size bounds

3. Add configuration to `/crates/polysniper-core/src/types.rs`:
   - `KellyConfig` struct with enable flag, fraction (0.25-1.0), window size
   - Add to `RiskConfig` struct

4. Update `/config/default.toml`:
   - Add `[risk.kelly]` section with sensible defaults
   - Default to half-Kelly (0.5) for conservative sizing

5. Add comprehensive tests in kelly.rs:
   - Test edge calculation with known values
   - Test fractional Kelly scaling
   - Test minimum sample size behavior
   - Test integration with volatility sizing

## Acceptance Criteria

- [ ] KellyCalculator correctly implements Kelly criterion formula
- [ ] Fractional Kelly support (configurable 0.25 to 1.0 fraction)
- [ ] Rolling window tracks recent N trades for dynamic edge calculation
- [ ] Graceful degradation when insufficient trade history (skip Kelly sizing)
- [ ] Integration with existing RiskManager validation pipeline
- [ ] Configuration via TOML with hot-reload support
- [ ] All existing tests still pass
- [ ] New unit tests for Kelly calculator (>80% coverage)
- [ ] No f64 usage - all Decimal arithmetic

## Files to Create/Modify

**Create:**
- `crates/polysniper-risk/src/kelly.rs` - Kelly criterion calculator

**Modify:**
- `crates/polysniper-risk/src/lib.rs` - Export kelly module
- `crates/polysniper-risk/src/validator.rs` - Integrate Kelly sizing
- `crates/polysniper-core/src/types.rs` - Add KellyConfig
- `config/default.toml` - Add Kelly configuration section

## Integration Points

- **Provides**: Edge-aware position sizing multiplier for RiskManager
- **Consumes**: Trade history from persistence layer (PnL data)
- **Conflicts**: None - complements existing volatility sizing

## Technical Notes

- Use `rust_decimal::Decimal` for all calculations (no f64)
- Follow existing `VolatilityCalculator` pattern for structure
- Kelly fraction should be applied AFTER volatility adjustment
- Consider adding `KellyEdge` event type for observability
