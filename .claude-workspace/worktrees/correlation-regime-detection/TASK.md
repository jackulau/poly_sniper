---
id: correlation-regime-detection
name: Correlation Regime Detection
wave: 1
priority: 2
dependencies: []
estimated_hours: 5
tags: [risk, correlation, regime]
---

## Objective

Implement dynamic correlation regime detection to identify when market correlations spike during stress periods, automatically adjusting exposure limits when the correlation environment shifts from normal to stressed.

## Context

The codebase already has static correlation tracking in `polysniper-risk/src/correlation.rs` that calculates Pearson correlation from price history. However, correlations are not constant - they tend to spike dramatically during market stress ("correlations go to 1 in a crisis"). This task adds:

- Rolling correlation calculation with configurable windows
- Regime classification (normal, elevated, crisis)
- Dynamic adjustment of correlation exposure limits per regime
- Detection of correlation breakouts from historical norms

Key insight: A 0.3 correlation in normal times might become 0.8+ during stress. Static limits don't account for this regime-dependent behavior.

## Implementation

1. Create `/crates/polysniper-risk/src/correlation_regime.rs`:
   - `CorrelationRegimeDetector` struct with rolling windows
   - `CorrelationRegime` enum: Normal, Elevated, Crisis
   - Track average correlation across all position pairs
   - Detect regime using z-score or percentile thresholds
   - Calculate regime-adjusted exposure limits

2. Add configuration types to `/crates/polysniper-core/src/types.rs`:
   - `CorrelationRegimeConfig` struct with:
     - `enabled: bool`
     - `short_window_secs: u64` (default 300 - 5 min)
     - `long_window_secs: u64` (default 86400 - 24 hours)
     - `elevated_threshold: Decimal` (default 0.5 - 50% above baseline)
     - `crisis_threshold: Decimal` (default 1.0 - 100% above baseline)
     - `normal_limit_multiplier: Decimal` (default 1.0)
     - `elevated_limit_multiplier: Decimal` (default 0.7)
     - `crisis_limit_multiplier: Decimal` (default 0.4)
     - `min_samples: usize` (default 100)

3. Integrate with `CorrelationTracker` in `/crates/polysniper-risk/src/correlation.rs`:
   - Add `CorrelationRegimeDetector` as component
   - Track short-term vs long-term average correlation
   - Adjust `max_correlated_exposure_usd` based on current regime
   - Emit events when regime changes

4. Add to `RiskManager` in `/crates/polysniper-risk/src/validator.rs`:
   - Periodically update regime detection
   - Apply regime-adjusted limits to correlation checks
   - Log regime transitions

5. Add new event type (optional):
   - `SystemEvent::CorrelationRegimeChange { old: CorrelationRegime, new: CorrelationRegime, avg_correlation: Decimal }`

6. Update `/config/default.toml`:
   - Add `[risk.correlation.regime]` section
   - Default to conservative crisis multipliers

7. Add comprehensive tests:
   - Test regime detection with synthetic data
   - Test limit adjustment per regime
   - Test transitions between regimes
   - Test hysteresis to prevent regime flickering

## Acceptance Criteria

- [ ] CorrelationRegimeDetector calculates rolling average correlation
- [ ] Regime correctly classified (Normal/Elevated/Crisis) based on thresholds
- [ ] Exposure limits dynamically adjusted per regime
- [ ] Hysteresis prevents rapid regime oscillation
- [ ] Integration with existing CorrelationTracker
- [ ] Configuration via TOML with hot-reload support
- [ ] All existing correlation tests still pass
- [ ] New unit tests for regime detection (>80% coverage)
- [ ] No f64 usage - all Decimal arithmetic

## Files to Create/Modify

**Create:**
- `crates/polysniper-risk/src/correlation_regime.rs` - Regime detection logic

**Modify:**
- `crates/polysniper-risk/src/lib.rs` - Export correlation_regime module
- `crates/polysniper-risk/src/correlation.rs` - Integrate regime detection
- `crates/polysniper-risk/src/validator.rs` - Use regime-adjusted limits
- `crates/polysniper-core/src/types.rs` - Add CorrelationRegimeConfig
- `crates/polysniper-core/src/events.rs` - Add CorrelationRegimeChange event (optional)
- `config/default.toml` - Add regime configuration section

## Integration Points

- **Provides**: Dynamic correlation regime classification and limit adjustment
- **Consumes**: Price history from StateProvider, correlation calculations from CorrelationTracker
- **Conflicts**: Extends correlation.rs - minimal overlap

## Technical Notes

- Use `rust_decimal::Decimal` for all calculations
- Rolling window should use efficient ring buffer (VecDeque)
- Consider exponentially weighted moving average for smoother transitions
- Regime detection should run on a timer (e.g., every 60s) not per-trade
- Store regime history for analysis and backtesting
- Z-score calculation: `(current_avg - long_term_avg) / long_term_std`
