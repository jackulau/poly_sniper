---
id: implementation-shortfall
name: Implementation Shortfall Minimization
wave: 1
priority: 1
dependencies: []
estimated_hours: 6
tags: [execution, metrics, optimization]
---

## Objective

Track slippage vs. decision price and optimize execution to reduce implementation shortfall.

## Context

Implementation shortfall is the difference between the decision price (price when trade signal was generated) and the actual execution price. This is a key measure of execution quality. The codebase already has TWAP/VWAP algorithms with slippage tracking, but lacks dedicated implementation shortfall measurement and minimization logic.

## Implementation

### 1. Create Implementation Shortfall Tracker

Create new file `crates/polysniper-execution/src/shortfall_tracker.rs`:

```rust
pub struct ShortfallTracker {
    // Track decision prices for each parent order
    decision_prices: HashMap<String, ShortfallRecord>,
    config: ShortfallConfig,
}

pub struct ShortfallRecord {
    parent_order_id: String,
    decision_price: Decimal,     // Price when signal generated
    decision_time: DateTime<Utc>,
    side: Side,
    total_size: Decimal,
    executed_size: Decimal,
    vwap: Decimal,
    shortfall_bps: Decimal,      // Implementation shortfall in basis points
    components: ShortfallComponents,
}

pub struct ShortfallComponents {
    timing_delay: Decimal,       // Cost from delay between decision and first fill
    market_impact: Decimal,      // Cost from our own order pressure
    spread_cost: Decimal,        // Cost from bid-ask spread
    opportunity_cost: Decimal,   // Cost from unfilled portion
}
```

### 2. Integrate with TWAP/VWAP Executors

Modify `algorithms/twap.rs` and `algorithms/vwap.rs` to:
- Capture decision price at execution start
- Calculate shortfall components after each fill
- Report final shortfall on completion

### 3. Add Shortfall Metrics

Add to `polysniper-observability/src/metrics.rs`:
- `execution_shortfall_bps` histogram
- `shortfall_by_component` gauges
- `shortfall_by_algorithm` counters

### 4. Add Adaptive Execution Speed

Implement logic to speed up or slow down execution based on:
- Current shortfall trend (if losing to adverse selection, execute faster)
- Favorable price movements (if price is better, can be patient)

## Acceptance Criteria

- [ ] ShortfallTracker struct implemented with all components
- [ ] Decision price captured at signal generation time
- [ ] Shortfall calculated correctly for buy and sell orders
- [ ] Components (timing, impact, spread, opportunity) measured separately
- [ ] Prometheus metrics exported for shortfall monitoring
- [ ] TWAP/VWAP executors integrated with shortfall tracking
- [ ] Adaptive execution speed based on shortfall trend
- [ ] All tests passing

## Files to Create/Modify

- `crates/polysniper-execution/src/shortfall_tracker.rs` - Create new file
- `crates/polysniper-execution/src/lib.rs` - Add module export
- `crates/polysniper-execution/src/algorithms/twap.rs` - Integrate tracker
- `crates/polysniper-execution/src/algorithms/vwap.rs` - Integrate tracker
- `crates/polysniper-execution/src/algorithms/mod.rs` - Update ExecutionStats
- `crates/polysniper-observability/src/metrics.rs` - Add shortfall metrics
- `crates/polysniper-core/src/types.rs` - Add ShortfallConfig to AppConfig

## Integration Points

- **Provides**: Shortfall tracking for all algorithmic executions
- **Consumes**: Decision prices from TradeSignal, fill data from executors
- **Conflicts**: None - new functionality
