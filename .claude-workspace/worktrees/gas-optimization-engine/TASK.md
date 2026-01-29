---
id: gas-optimization-engine
name: Gas Optimization Engine - Smart Transaction Timing
wave: 2
priority: 2
dependencies: [gas-fee-tracker]
estimated_hours: 5
tags: [backend, execution, optimization]
---

## Objective

Create an optimization layer that uses gas price data to intelligently time order submissions, batch orders when beneficial, and select optimal execution strategies based on current network conditions.

## Context

Once gas-fee-tracker provides real-time gas data, this task implements the decision logic to optimize when and how orders are submitted. Non-urgent orders can wait for lower gas, while urgent signals execute immediately.

## Implementation

1. Create `/crates/polysniper-execution/src/gas_optimizer.rs`:
   - `GasOptimizer` that wraps OrderSubmitter
   - Queue for non-urgent orders
   - Gas price threshold checks before submission
   - Batch order accumulation and submission

2. Implement order queuing:
   - Priority-based queue (CRITICAL immediate, LOW can wait)
   - Max queue time before forced execution
   - Gas price triggers for queue flush

3. Add batch optimization:
   - Accumulate orders for same token/market
   - Combine into single transaction when possible
   - Track batch savings

4. Create timing strategies:
   - `Immediate`: Execute regardless of gas
   - `WaitForLow`: Queue until gas below threshold
   - `TimeWindow`: Execute within X minutes at best gas
   - Auto-select based on signal priority

5. Integrate with execution pipeline:
   - Wrap existing OrderSubmitter
   - Add gas optimization config
   - Track savings metrics

## Acceptance Criteria

- [ ] Non-urgent orders queue until gas price favorable
- [ ] Urgent (HIGH/CRITICAL priority) orders execute immediately
- [ ] Maximum queue time prevents stale orders
- [ ] Gas threshold triggers configurable per priority
- [ ] Batch similar orders when gas-efficient
- [ ] Tracks gas savings vs immediate execution
- [ ] Transparent to strategies (wraps existing executor)
- [ ] Configurable optimization strategies
- [ ] Unit tests for queue management
- [ ] Metrics/logging for optimization decisions

## Files to Create/Modify

- `crates/polysniper-execution/src/gas_optimizer.rs` - **CREATE** - Gas optimization logic
- `crates/polysniper-execution/src/lib.rs` - **MODIFY** - Export optimizer module
- `crates/polysniper-core/src/gas.rs` - **MODIFY** - Add optimization config types
- `config/default.toml` - **MODIFY** - Add gas optimization settings
- `src/main.rs` - **MODIFY** - Wire optimizer into execution pipeline

## Integration Points

- **Provides**: Gas-optimized order execution
- **Consumes**: GasPriceUpdate events from gas-fee-tracker, Orders from strategies
- **Conflicts**: May modify execution pipeline in main.rs (coordinate carefully)
