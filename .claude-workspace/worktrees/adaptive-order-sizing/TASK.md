---
id: adaptive-order-sizing
name: Adaptive Order Sizing Based on Orderbook Depth
wave: 1
priority: 1
dependencies: []
estimated_hours: 4
tags: [execution, orderbook, sizing]
---

## Objective

Implement intelligent order sizing that analyzes orderbook depth to determine optimal order sizes, preventing excessive slippage and market impact.

## Context

Currently, orders use fixed sizes specified in trade signals. This task adds dynamic sizing based on available liquidity at each price level in the orderbook. The system should analyze depth on both sides and size orders appropriately to avoid moving the market or getting poor fills.

## Implementation

### 1. Create Depth Analyzer Module

**File**: `crates/polysniper-execution/src/depth_analyzer.rs`

```rust
pub struct DepthAnalyzer {
    max_market_impact_bps: Decimal,  // Max acceptable price impact in basis points
    min_liquidity_ratio: Decimal,    // Min ratio of our order to available depth
}

impl DepthAnalyzer {
    /// Analyze orderbook depth and return recommended order size
    pub fn calculate_optimal_size(
        &self,
        orderbook: &Orderbook,
        side: Side,
        target_price: Decimal,
        max_size: Decimal,
    ) -> OrderSizeRecommendation;
    
    /// Calculate available liquidity up to a price level
    pub fn liquidity_to_price(
        &self,
        orderbook: &Orderbook,
        side: Side,
        limit_price: Decimal,
    ) -> Decimal;
    
    /// Estimate price impact for a given order size
    pub fn estimate_price_impact(
        &self,
        orderbook: &Orderbook,
        side: Side,
        size: Decimal,
    ) -> PriceImpact;
}

pub struct OrderSizeRecommendation {
    pub recommended_size: Decimal,
    pub max_safe_size: Decimal,
    pub estimated_avg_price: Decimal,
    pub estimated_impact_bps: Decimal,
    pub liquidity_score: f64,  // 0.0 to 1.0, how liquid the book is
}

pub struct PriceImpact {
    pub impact_bps: Decimal,
    pub avg_fill_price: Decimal,
    pub levels_consumed: usize,
}
```

### 2. Integrate with OrderBuilder

**File**: `crates/polysniper-execution/src/order_builder.rs`

- Add `DepthAnalyzer` as an optional component
- Modify `build_from_signal()` to optionally use depth analysis
- Add configuration for adaptive sizing behavior

### 3. Add Configuration Options

**File**: `crates/polysniper-core/src/types.rs` (extend ExecutionConfig)

```rust
pub struct AdaptiveSizingConfig {
    pub enabled: bool,
    pub max_market_impact_bps: Decimal,  // Default: 50 bps
    pub min_liquidity_ratio: Decimal,    // Default: 0.1 (10% of depth)
    pub size_reduction_factor: Decimal,  // Default: 0.8 (reduce by 20% if thin)
}
```

### 4. Update Configuration File

**File**: `config/default.toml`

```toml
[execution.adaptive_sizing]
enabled = true
max_market_impact_bps = 50
min_liquidity_ratio = 0.1
size_reduction_factor = 0.8
```

## Acceptance Criteria

- [ ] DepthAnalyzer correctly calculates available liquidity at price levels
- [ ] Order sizes are reduced when orderbook is thin
- [ ] Price impact estimation is accurate within 10% tolerance
- [ ] Configuration allows disabling adaptive sizing
- [ ] Unit tests cover edge cases (empty book, single level, etc.)
- [ ] Integration with OrderBuilder maintains backward compatibility
- [ ] Logging shows sizing decisions for debugging

## Files to Create/Modify

- `crates/polysniper-execution/src/depth_analyzer.rs` - **CREATE** - Core depth analysis logic
- `crates/polysniper-execution/src/lib.rs` - **MODIFY** - Export new module
- `crates/polysniper-execution/src/order_builder.rs` - **MODIFY** - Integrate depth analyzer
- `crates/polysniper-core/src/types.rs` - **MODIFY** - Add config struct
- `config/default.toml` - **MODIFY** - Add configuration section

## Integration Points

- **Provides**: `DepthAnalyzer` for order sizing decisions
- **Consumes**: `Orderbook` from `StateProvider`
- **Conflicts**: Avoid modifying `submitter.rs` (handled by partial-fill-handler)

## Testing Notes

- Use in-memory orderbooks with various depth scenarios
- Test edge cases: empty book, single-sided book, very thin liquidity
- Verify backward compatibility with fixed sizing when disabled
