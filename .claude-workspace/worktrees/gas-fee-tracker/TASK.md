---
id: gas-fee-tracker
name: Gas Fee Tracker - Monitor Polygon Gas Costs
wave: 1
priority: 1
dependencies: []
estimated_hours: 4
tags: [backend, blockchain, optimization]
---

## Objective

Create a service that monitors Polygon network gas prices and transaction costs, providing real-time fee data for order optimization decisions.

## Context

Polysniper executes trades on Polymarket which runs on Polygon. While Polymarket's CLOB handles order matching, understanding gas costs helps optimize when and how to submit orders. This task adds gas price monitoring.

## Implementation

1. Create `/crates/polysniper-execution/src/gas_tracker.rs`:
   - Poll Polygon gas price APIs (Polygonscan, Alchemy)
   - Track gas price history (fast, standard, slow)
   - Calculate average gas costs for order types
   - Emit `GasPriceUpdate` events

2. Create `/crates/polysniper-core/src/gas.rs`:
   - `GasConfig` with API endpoints and thresholds
   - `GasPrice` struct (fast, standard, slow in gwei)
   - `GasCostEstimate` for order operations
   - `GasCondition` enum (Low, Normal, High, Extreme)

3. Add gas price data source:
   - Primary: Polygon RPC `eth_gasPrice`
   - Fallback: Polygonscan Gas Oracle API
   - Caching with TTL (10 seconds)

4. Create gas history tracking:
   - Store recent gas prices in memory
   - Calculate rolling averages
   - Detect gas spikes

## Acceptance Criteria

- [ ] Polls Polygon gas prices every 10 seconds
- [ ] Tracks fast/standard/slow gas tiers
- [ ] Maintains gas price history (last 1 hour)
- [ ] Calculates rolling averages and percentiles
- [ ] Emits events when gas exceeds thresholds
- [ ] Provides current gas condition classification
- [ ] Handles API failures gracefully with fallbacks
- [ ] Configurable poll interval and thresholds
- [ ] Unit tests for gas calculations

## Files to Create/Modify

- `crates/polysniper-execution/src/gas_tracker.rs` - **CREATE** - Gas monitoring service
- `crates/polysniper-execution/src/lib.rs` - **MODIFY** - Export gas module
- `crates/polysniper-core/src/gas.rs` - **CREATE** - Gas types and config
- `crates/polysniper-core/src/lib.rs` - **MODIFY** - Export gas module
- `crates/polysniper-core/src/events.rs` - **MODIFY** - Add GasPriceUpdate event
- `config/default.toml` - **MODIFY** - Add gas tracking config

## Integration Points

- **Provides**: Real-time gas price data and classifications
- **Consumes**: External APIs (Polygon RPC, Polygonscan)
- **Conflicts**: May touch events.rs (coordinate with other tasks)
