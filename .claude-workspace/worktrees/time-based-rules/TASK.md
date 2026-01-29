---
id: time-based-rules
name: Time-Based Risk Rules
wave: 1
priority: 3
dependencies: []
estimated_hours: 4
tags: [risk, timing, market-events]
---

## Objective

Implement time-based risk rules that reduce trading activity before market resolution or major events.

## Context

Prediction markets have known event times (resolution dates, election days, etc.). Trading near these events carries heightened risk due to:
- Reduced liquidity as resolution approaches
- Increased volatility from last-minute information
- Higher slippage and wider spreads

This task adds time-aware risk rules to reduce or halt activity as events approach.

## Implementation

1. **Create time rules module** in `crates/polysniper-risk/src/time_rules.rs`:
   - Define `TimeBasedRule` struct for configurable rules
   - Implement rule matching based on time-to-event
   - Calculate time until market resolution from `Market.end_date`

2. **Add time rules config** to `RiskConfig` in `crates/polysniper-core/src/types.rs`:
   ```rust
   pub struct TimeRulesConfig {
       pub enabled: bool,
       pub rules: Vec<TimeRule>,
   }
   
   pub struct TimeRule {
       pub name: String,
       pub hours_before: u64,            // Hours before event
       pub action: TimeRuleAction,       // What to do
       pub applies_to: Vec<String>,      // Market patterns (glob)
   }
   
   pub enum TimeRuleAction {
       ReduceSize { multiplier: Decimal }, // Reduce position size
       BlockNew,                           // Block new positions (allow exits)
       HaltAll,                           // No trading at all
   }
   ```

3. **Implement time rule checking** in `time_rules.rs`:
   - `TimeRuleEngine::check_market(market)` - Check if rules apply
   - `TimeRuleEngine::get_size_modifier(market)` - Get size multiplier
   - `TimeRuleEngine::is_blocked(market)` - Check if trading blocked
   - Use glob patterns for market matching

4. **Integrate into RiskManager** (`crates/polysniper-risk/src/validator.rs`):
   - Add time rule checking before other validations
   - Return `RiskDecision::Rejected` if blocked
   - Apply size reduction if rule specifies multiplier
   - Include time rule reason in decision

5. **Add config section** in `config/default.toml`:
   ```toml
   [risk.time_rules]
   enabled = true
   
   [[risk.time_rules.rules]]
   name = "pre_resolution_reduction"
   hours_before = 24
   action = { type = "ReduceSize", multiplier = 0.5 }
   applies_to = ["*"]
   
   [[risk.time_rules.rules]]
   name = "resolution_block"
   hours_before = 2
   action = "BlockNew"
   applies_to = ["*"]
   ```

6. **Add unit tests** in `crates/polysniper-risk/src/time_rules.rs`:
   - Test rule matching with various time-to-event values
   - Test glob pattern matching for market filtering
   - Test size modification calculation
   - Test rule priority (most restrictive wins)

## Acceptance Criteria

- [ ] Time-to-resolution calculated correctly from market end_date
- [ ] Rules applied in order of restrictiveness
- [ ] Size reduction applied at configured thresholds
- [ ] Trading blocked when within block window
- [ ] Glob patterns work for market matching
- [ ] Time rule application logged for observability
- [ ] All tests passing
- [ ] Feature can be disabled via config

## Files to Create/Modify

- `crates/polysniper-risk/src/time_rules.rs` - Create new time rules module
- `crates/polysniper-risk/src/lib.rs` - Export time_rules module
- `crates/polysniper-risk/src/validator.rs` - Integrate time rule checks
- `crates/polysniper-core/src/types.rs` - Add TimeRulesConfig structs
- `config/default.toml` - Add time rules config section

## Integration Points

- **Provides**: `TimeRuleEngine` for time-based trading restrictions
- **Consumes**: `Market.end_date` from StateProvider
- **Conflicts**: Avoid editing strategy code or order execution
