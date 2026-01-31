---
id: drawdown-scaling
name: Drawdown-Triggered Position Scaling
wave: 1
priority: 1
dependencies: []
estimated_hours: 4
tags: [risk, sizing, drawdown]
---

## Objective

Implement progressive position size reduction based on drawdown depth, automatically scaling down exposure as portfolio losses accumulate to protect capital during adverse conditions.

## Context

The codebase already has volatility-based sizing in `polysniper-risk/src/volatility.rs` and circuit breaker logic in `validator.rs`. Drawdown scaling adds a complementary layer that:
- Reduces position sizes progressively as drawdown deepens
- Provides smooth degradation rather than binary halt/resume
- Allows trading to continue with reduced risk during drawdowns
- Automatically recovers sizing as PnL improves

The scaling formula uses a continuous function:
- 0-5% drawdown: 100% of normal size
- 5-10% drawdown: Linear reduction to 75%
- 10-20% drawdown: Linear reduction to 50%
- 20-30% drawdown: Linear reduction to 25%
- >30% drawdown: Minimum size (10%) or halt

## Implementation

1. Create `/crates/polysniper-risk/src/drawdown.rs`:
   - `DrawdownCalculator` struct to track portfolio high-water mark
   - `DrawdownConfig` with configurable thresholds and reduction tiers
   - Calculate current drawdown percentage from peak
   - Return size multiplier based on drawdown level
   - Track peak equity and current equity

2. Add configuration types to `/crates/polysniper-core/src/types.rs`:
   - `DrawdownConfig` struct with:
     - `enabled: bool`
     - `tier_1_threshold_pct: Decimal` (default 5.0)
     - `tier_1_multiplier: Decimal` (default 0.75)
     - `tier_2_threshold_pct: Decimal` (default 10.0)
     - `tier_2_multiplier: Decimal` (default 0.50)
     - `tier_3_threshold_pct: Decimal` (default 20.0)
     - `tier_3_multiplier: Decimal` (default 0.25)
     - `max_drawdown_pct: Decimal` (default 30.0)
     - `min_multiplier: Decimal` (default 0.10)
     - `recovery_buffer_pct: Decimal` (default 2.0) - hysteresis buffer

3. Integrate with `RiskManager` in `/crates/polysniper-risk/src/validator.rs`:
   - Add DrawdownCalculator as a component
   - Apply drawdown multiplier after volatility/Kelly sizing
   - Track portfolio value updates to maintain high-water mark
   - Add `update_equity(new_value: Decimal)` method

4. Add StateProvider methods if needed:
   - `get_peak_portfolio_value() -> Decimal`
   - Or track internally in DrawdownCalculator

5. Update `/config/default.toml`:
   - Add `[risk.drawdown]` section with tier configuration
   - Default to conservative thresholds

6. Add comprehensive tests:
   - Test multiplier calculation at each tier
   - Test smooth interpolation between tiers
   - Test recovery behavior with hysteresis
   - Test edge cases (0% drawdown, exactly at threshold)

## Acceptance Criteria

- [ ] DrawdownCalculator correctly tracks high-water mark
- [ ] Drawdown percentage calculated accurately from peak
- [ ] Size multiplier reduces smoothly through tiers
- [ ] Hysteresis buffer prevents rapid oscillation during recovery
- [ ] Integration with RiskManager validation pipeline
- [ ] Configuration via TOML with hot-reload support
- [ ] All existing tests still pass
- [ ] New unit tests for drawdown calculations (>80% coverage)
- [ ] No f64 usage - all Decimal arithmetic

## Files to Create/Modify

**Create:**
- `crates/polysniper-risk/src/drawdown.rs` - Drawdown calculator and scaling logic

**Modify:**
- `crates/polysniper-risk/src/lib.rs` - Export drawdown module
- `crates/polysniper-risk/src/validator.rs` - Integrate drawdown scaling
- `crates/polysniper-core/src/types.rs` - Add DrawdownConfig
- `config/default.toml` - Add drawdown configuration section

## Integration Points

- **Provides**: Drawdown-aware size multiplier for RiskManager
- **Consumes**: Portfolio value from StateProvider
- **Conflicts**: None - complements existing volatility sizing

## Technical Notes

- Use `rust_decimal::Decimal` for all calculations (no f64)
- Follow existing `VolatilityCalculator` pattern for structure
- Drawdown multiplier should be applied AFTER volatility adjustment
- Consider using linear interpolation between tiers for smooth transitions
- High-water mark should persist across restarts (consider DB storage)
