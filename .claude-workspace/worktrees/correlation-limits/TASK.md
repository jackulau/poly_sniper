---
id: correlation-limits
name: Correlation-Aware Position Limits
wave: 1
priority: 2
dependencies: []
estimated_hours: 5
tags: [risk, correlation, position-management]
---

## Objective

Implement correlation-aware position limits that reduce total exposure when holding positions in correlated markets.

## Context

The current risk system checks position limits per market independently. However, correlated markets (e.g., related political outcomes, similar event categories) compound risk exposure. This task adds correlation tracking and aggregate exposure limits for correlated positions.

Polymarket markets often have natural correlations:
- Same event, different outcomes (Yes/No of same question)
- Related events (e.g., multiple election markets)
- Time-series events (same event at different dates)

## Implementation

1. **Create correlation module** in `crates/polysniper-risk/src/correlation.rs`:
   - Define `CorrelationGroup` struct to group related markets
   - Implement `CorrelationTracker` to track correlation between markets
   - Calculate correlation from price movements over time window
   - Group markets by correlation threshold (e.g., >0.7 correlation)

2. **Add correlation config** to `RiskConfig` in `crates/polysniper-core/src/types.rs`:
   ```rust
   pub struct CorrelationConfig {
       pub enabled: bool,
       pub correlation_threshold: Decimal,    // Minimum correlation to group (0.7)
       pub window_secs: u64,                  // Time window for correlation (3600)
       pub max_correlated_exposure_usd: Decimal, // Max total across correlated (3000)
       pub correlation_groups: Vec<Vec<String>>, // Manual groupings (market slugs)
   }
   ```

3. **Implement correlation calculation** in `correlation.rs`:
   - `calculate_correlation(market_a, market_b)` - Pearson correlation coefficient
   - `get_correlated_positions(market_id)` - Find all correlated positions
   - `get_correlated_exposure(market_id)` - Sum USD exposure in correlated group

4. **Integrate into RiskManager** (`crates/polysniper-risk/src/validator.rs`):
   - Add method `check_correlated_exposure()`
   - Before approving, sum exposure across correlated markets
   - Reject or reduce size if correlated exposure would exceed limit
   - Add reason to RiskDecision for transparency

5. **Support manual correlation groups** in config:
   ```toml
   [risk.correlation]
   enabled = true
   correlation_threshold = 0.7
   max_correlated_exposure_usd = 3000
   
   [[risk.correlation.groups]]
   markets = ["presidential-election-winner", "electoral-vote-count", "swing-state-*"]
   ```

6. **Add unit tests** in `crates/polysniper-risk/src/correlation.rs`:
   - Test correlation calculation with known price series
   - Test correlated exposure aggregation
   - Test position rejection when limit exceeded
   - Test manual grouping override

## Acceptance Criteria

- [ ] Correlation calculated correctly using Pearson coefficient
- [ ] Markets automatically grouped when correlation exceeds threshold
- [ ] Total exposure across correlated positions enforced
- [ ] Manual correlation groups respected from config
- [ ] Position rejected/reduced when correlated limit exceeded
- [ ] Correlation data logged for observability
- [ ] All tests passing
- [ ] Feature can be disabled via config

## Files to Create/Modify

- `crates/polysniper-risk/src/correlation.rs` - Create new correlation tracking module
- `crates/polysniper-risk/src/lib.rs` - Export correlation module
- `crates/polysniper-risk/src/validator.rs` - Integrate correlation checks
- `crates/polysniper-core/src/types.rs` - Add CorrelationConfig struct
- `config/default.toml` - Add correlation config section

## Integration Points

- **Provides**: `CorrelationTracker` for grouping markets and calculating exposure
- **Consumes**: `StateProvider::get_all_positions()`, `StateProvider::get_price_history()`
- **Conflicts**: Avoid editing order execution or strategy code
