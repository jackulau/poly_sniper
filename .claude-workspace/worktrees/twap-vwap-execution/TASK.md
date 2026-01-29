---
id: twap-vwap-execution
name: TWAP/VWAP Execution Algorithms
wave: 1
priority: 4
dependencies: []
estimated_hours: 6
tags: [execution, twap, vwap, slippage]
---

## Objective

Implement Time-Weighted Average Price (TWAP) and Volume-Weighted Average Price (VWAP) execution algorithms to break large orders into smaller chunks over time, reducing market impact and slippage.

## Context

Large orders can significantly move the market if executed all at once. TWAP/VWAP algorithms help by:
- **TWAP**: Splitting orders evenly across time intervals
- **VWAP**: Splitting orders proportionally to historical volume patterns

This is implemented in the execution layer, not as a strategy. It transforms large trade signals into multiple smaller child orders.

## Implementation

### 1. Create Execution Algorithm Module
**File:** `crates/polysniper-execution/src/algorithms/mod.rs`
**File:** `crates/polysniper-execution/src/algorithms/twap.rs`
**File:** `crates/polysniper-execution/src/algorithms/vwap.rs`

### 2. TWAP Implementation

```rust
pub struct TwapConfig {
    pub total_duration_secs: u64,     // Total execution window
    pub num_slices: u32,              // Number of child orders
    pub randomize_timing: bool,       // Add random jitter to timing
    pub randomize_size: bool,         // Vary child order sizes
    pub max_participation_rate: Decimal,  // Max % of volume per interval
}

pub struct TwapExecutor {
    config: TwapConfig,
    active_orders: Arc<RwLock<HashMap<String, TwapState>>>,
}

impl TwapExecutor {
    // Split parent order into child orders
    pub fn create_schedule(&self, signal: &TradeSignal) -> Vec<ChildOrder>;

    // Get next child order to execute
    pub fn get_next_order(&self, parent_id: &str) -> Option<ChildOrder>;

    // Update state after child order execution
    pub fn record_fill(&self, parent_id: &str, filled_size: Decimal, fill_price: Decimal);

    // Check if execution is complete
    pub fn is_complete(&self, parent_id: &str) -> bool;

    // Get execution statistics
    pub fn get_stats(&self, parent_id: &str) -> ExecutionStats;
}
```

### 3. VWAP Implementation

```rust
pub struct VwapConfig {
    pub total_duration_secs: u64,
    pub volume_profile: VolumeProfile,    // Historical volume pattern
    pub max_participation_rate: Decimal,
    pub adaptive: bool,                   // Adjust based on real-time volume
}

pub enum VolumeProfile {
    Uniform,                              // Same as TWAP
    Custom(Vec<Decimal>),                 // Custom percentage per interval
    Historical,                           // Load from historical data
}

pub struct VwapExecutor {
    config: VwapConfig,
    active_orders: Arc<RwLock<HashMap<String, VwapState>>>,
}
```

### 4. Execution Statistics

```rust
pub struct ExecutionStats {
    pub parent_id: String,
    pub total_size: Decimal,
    pub executed_size: Decimal,
    pub remaining_size: Decimal,
    pub num_fills: u32,
    pub avg_fill_price: Decimal,
    pub vwap: Decimal,                    // Volume-weighted average price
    pub slippage_pct: Decimal,            // vs initial price
    pub started_at: DateTime<Utc>,
    pub estimated_completion: DateTime<Utc>,
}
```

### 5. Integration with Order Submitter

Modify `submitter.rs` to support algorithm-based execution:
- Add `ExecutionAlgorithm` enum to `TradeSignal` or create wrapper
- Route large orders through TWAP/VWAP executors
- Track parent-child order relationships

### 6. Create Config
**File:** `config/execution.toml` - Add algorithm configuration section

### 7. Add Tests
Unit tests for:
- TWAP schedule generation
- VWAP volume distribution
- Fill tracking and stats calculation
- Edge cases (partial fills, timeouts)

## Acceptance Criteria

- [ ] TwapExecutor correctly splits orders across time intervals
- [ ] VwapExecutor distributes orders according to volume profile
- [ ] Child orders are generated with correct sizes and timing
- [ ] Fill tracking accurately updates execution state
- [ ] ExecutionStats provides accurate VWAP and slippage metrics
- [ ] Configuration loads from TOML file
- [ ] All unit tests passing
- [ ] Integration with existing execution pipeline

## Files to Create/Modify

- `crates/polysniper-execution/src/algorithms/mod.rs` - **CREATE** - Module definition
- `crates/polysniper-execution/src/algorithms/twap.rs` - **CREATE** - TWAP executor
- `crates/polysniper-execution/src/algorithms/vwap.rs` - **CREATE** - VWAP executor
- `crates/polysniper-execution/src/lib.rs` - **MODIFY** - Add algorithms module
- `crates/polysniper-core/src/types.rs` - **MODIFY** - Add ExecutionAlgorithm enum (optional)
- `config/execution.toml` - **MODIFY** - Add algorithm settings

## Integration Points

- **Provides**: Order slicing and execution scheduling, execution statistics
- **Consumes**: TradeSignal, order execution feedback
- **Conflicts**: May modify types.rs (coordinate with other tasks touching this file)

## Notes

- Consider adding market participation rate limits (don't exceed X% of volume)
- Real VWAP requires historical volume data - may need to start with TWAP or uniform
- Should handle cancellation of in-progress parent orders
- Consider adaptive algorithms that adjust based on real-time conditions
