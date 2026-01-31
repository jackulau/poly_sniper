---
id: memory-mapped-orderbook
name: Memory-Mapped Orderbook for High-Frequency Updates
wave: 1
priority: 1
dependencies: []
estimated_hours: 6
tags: [performance, data-structures, orderbook]
---

## Objective

Replace the current Vec-based orderbook implementation with a memory-efficient, zero-copy structure that minimizes allocations during high-frequency updates.

## Context

Current orderbook implementation in `crates/polysniper-core/src/types.rs` uses `Vec<PriceLevel>` which requires:
- Full clone on every read (16 clone operations identified)
- New allocations on every update
- O(n) lookups for price-based operations

For high-frequency trading, this causes:
- Memory churn from frequent allocations
- Cache misses from scattered allocations
- Latency spikes during GC pressure

## Implementation

### 1. Create new orderbook module

**File**: `crates/polysniper-core/src/orderbook.rs`

Implement an arena-based orderbook with these characteristics:

```rust
use std::sync::Arc;

/// Price level stored in contiguous memory
#[derive(Clone, Copy)]
pub struct PackedPriceLevel {
    pub price: i64,      // Price in basis points (avoid Decimal overhead)
    pub size: i64,       // Size in smallest units
}

/// Memory-efficient orderbook with COW semantics
pub struct FastOrderbook {
    token_id: TokenId,
    market_id: MarketId,
    /// Sorted bids (descending by price), pre-allocated capacity
    bids: Box<[PackedPriceLevel]>,
    /// Sorted asks (ascending by price), pre-allocated capacity
    asks: Box<[PackedPriceLevel]>,
    bid_count: u16,
    ask_count: u16,
    timestamp: i64,  // Unix timestamp in millis
}

impl FastOrderbook {
    /// Create with pre-allocated capacity
    pub fn with_capacity(bid_capacity: usize, ask_capacity: usize) -> Self;

    /// Update in place without allocation (returns bool if successful)
    pub fn apply_delta(&mut self, delta: &OrderbookDelta) -> bool;

    /// Binary search for price level
    pub fn bid_at_price(&self, price: i64) -> Option<&PackedPriceLevel>;
    pub fn ask_at_price(&self, price: i64) -> Option<&PackedPriceLevel>;

    /// Zero-copy access to top N levels
    pub fn top_bids(&self, n: usize) -> &[PackedPriceLevel];
    pub fn top_asks(&self, n: usize) -> &[PackedPriceLevel];

    /// Convert to legacy format for compatibility
    pub fn to_legacy(&self) -> Orderbook;
}
```

### 2. Add Copy-on-Write wrapper for thread-safe sharing

```rust
/// Thread-safe orderbook with COW semantics
pub struct SharedOrderbook {
    inner: Arc<FastOrderbook>,
}

impl SharedOrderbook {
    /// Get read-only reference (no clone needed)
    pub fn get(&self) -> &FastOrderbook;

    /// Clone-on-write for mutations
    pub fn get_mut(&mut self) -> &mut FastOrderbook;
}
```

### 3. Integrate with MarketCache

**File**: `crates/polysniper-data/src/market_cache.rs`

- Replace `HashMap<TokenId, Orderbook>` with `HashMap<TokenId, SharedOrderbook>`
- Update `get_orderbook()` to return `&FastOrderbook` reference instead of clone
- Add `get_orderbook_snapshot()` for cases requiring owned data

### 4. Update consumers to use references

- `orderbook_imbalance.rs` - Use `top_bids()`/`top_asks()` slices
- `depth_analyzer.rs` - Use binary search methods
- `order_manager.rs` - Use reference-based access

### 5. Add benchmarks

**File**: `crates/polysniper-core/benches/orderbook_bench.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_orderbook_update(c: &mut Criterion) {
    // Compare Vec-based vs FastOrderbook update performance
}

fn bench_orderbook_access(c: &mut Criterion) {
    // Compare clone vs reference access patterns
}
```

## Acceptance Criteria

- [ ] FastOrderbook struct with pre-allocated storage
- [ ] Binary search for O(log n) price lookups
- [ ] Zero-copy top-N level access
- [ ] COW wrapper for thread-safe sharing
- [ ] MarketCache integration with reference-based access
- [ ] Backward compatibility via `to_legacy()` method
- [ ] Benchmark showing 5x+ improvement in update throughput
- [ ] All existing tests pass
- [ ] No memory leaks (run with `cargo test --release` + valgrind)

## Files to Create/Modify

- `crates/polysniper-core/src/orderbook.rs` - **CREATE** - New orderbook implementation
- `crates/polysniper-core/src/lib.rs` - Add `pub mod orderbook;`
- `crates/polysniper-core/src/types.rs` - Keep legacy `Orderbook` for compatibility
- `crates/polysniper-data/src/market_cache.rs` - Use SharedOrderbook
- `crates/polysniper-strategies/src/orderbook_imbalance.rs` - Use slice access
- `crates/polysniper-execution/src/depth_analyzer.rs` - Use binary search
- `crates/polysniper-core/benches/orderbook_bench.rs` - **CREATE** - Benchmarks

## Integration Points

- **Provides**: High-performance orderbook for all strategies and execution code
- **Consumes**: Raw orderbook data from WebSocket manager
- **Conflicts**: None - new module, backward compatible

## Technical Notes

1. Use `i64` for prices/sizes instead of `Decimal` in hot path (convert at boundaries)
2. Pre-allocate for typical depth (50-100 levels) to avoid runtime allocations
3. Consider `smallvec` for small orderbooks that fit in stack
4. Add `#[repr(C)]` for predictable memory layout
5. Use `unsafe` carefully for zero-copy slices if needed
