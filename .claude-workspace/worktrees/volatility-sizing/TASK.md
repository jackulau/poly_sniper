---
id: volatility-sizing
name: Volatility-Adjusted Position Sizing
wave: 1
priority: 1
dependencies: []
estimated_hours: 4
tags: [risk, sizing, volatility]
---

## Objective

Implement volatility-adjusted position sizing that automatically reduces position sizes in volatile markets.

## Context

The current system uses fixed USD-based sizing configured per strategy. This task adds dynamic sizing that considers market volatility - reducing exposure in high-volatility conditions and potentially increasing it in stable markets.

The codebase already has price history tracking (`MarketCache::price_history`) and percentage change calculations (`get_price_change_pct`). This task extends that to compute rolling volatility and apply it to position sizing.

## Implementation

1. **Create volatility calculator module** in `crates/polysniper-risk/src/volatility.rs`:
   - Calculate rolling standard deviation from price history
   - Compute volatility percentile across all tracked markets
   - Define volatility buckets (low, medium, high, extreme)

2. **Add volatility config** to `RiskConfig` in `crates/polysniper-core/src/types.rs`:
   ```rust
   pub struct VolatilityConfig {
       pub enabled: bool,
       pub window_secs: u64,           // Rolling window (default 300)
       pub base_volatility_pct: Decimal,  // Reference volatility
       pub min_size_multiplier: Decimal,  // Floor (e.g., 0.25 = 25% of normal)
       pub max_size_multiplier: Decimal,  // Ceiling (e.g., 1.5 = 150% of normal)
   }
   ```

3. **Integrate into RiskManager** (`crates/polysniper-risk/src/validator.rs`):
   - Add method `calculate_volatility_adjusted_size()`
   - Call from `validate()` to modify signal size based on current volatility
   - Return `RiskDecision::Modified` with volatility-adjusted size

4. **Update config loading** in `config/default.toml`:
   ```toml
   [risk.volatility]
   enabled = true
   window_secs = 300
   base_volatility_pct = 5.0
   min_size_multiplier = 0.25
   max_size_multiplier = 1.5
   ```

5. **Add unit tests** in `crates/polysniper-risk/src/volatility.rs`:
   - Test volatility calculation with known price series
   - Test size adjustment for different volatility levels
   - Test edge cases (empty history, single price, extreme volatility)

## Acceptance Criteria

- [ ] Volatility calculated correctly from price history using standard deviation
- [ ] Position sizes reduced proportionally in high-volatility markets
- [ ] Size multiplier clamped to configured min/max bounds
- [ ] Configuration properly loaded from TOML
- [ ] Volatility adjustment logged for observability
- [ ] All tests passing
- [ ] Feature can be disabled via config

## Files to Create/Modify

- `crates/polysniper-risk/src/volatility.rs` - Create new volatility calculation module
- `crates/polysniper-risk/src/lib.rs` - Export volatility module
- `crates/polysniper-risk/src/validator.rs` - Integrate volatility-based sizing
- `crates/polysniper-core/src/types.rs` - Add VolatilityConfig struct
- `config/default.toml` - Add volatility config section

## Integration Points

- **Provides**: `VolatilityCalculator` for computing market volatility
- **Consumes**: `StateProvider::get_price_history()` for historical prices
- **Conflicts**: Avoid editing order execution or strategy code (separate concerns)
