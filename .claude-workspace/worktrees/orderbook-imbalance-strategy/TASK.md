---
id: orderbook-imbalance-strategy
name: Order Book Imbalance Detection Strategy
wave: 1
priority: 3
dependencies: []
estimated_hours: 4
tags: [strategy, orderbook, imbalance]
---

## Objective

Implement a strategy that detects significant bid/ask volume asymmetry in the order book and trades in the direction of the imbalance, anticipating price movement.

## Context

Order book imbalance is a leading indicator of price direction:
- If bids >> asks: More buying pressure, price likely to rise → Buy
- If asks >> bids: More selling pressure, price likely to fall → Sell

This strategy analyzes order book depth to identify these imbalances and generate directional trades before the price moves.

## Implementation

### 1. Create Strategy File
**File:** `crates/polysniper-strategies/src/orderbook_imbalance.rs`

```rust
// Key components:
// - OrderbookImbalanceConfig struct
// - OrderbookImbalanceStrategy implementing Strategy trait
// - Imbalance calculation across multiple price levels
// - Signal generation with confidence weighting
```

### 2. Configuration Structure
```rust
pub struct OrderbookImbalanceConfig {
    pub enabled: bool,
    pub imbalance_threshold: Decimal,      // Min ratio (e.g., 2.0 = 2:1 imbalance)
    pub depth_levels: usize,               // Number of price levels to analyze
    pub order_size_usd: Decimal,           // Trade size
    pub min_liquidity_usd: Decimal,        // Minimum book depth
    pub cooldown_secs: u64,                // Cooldown between signals
    pub markets: Vec<String>,              // Empty = all markets
    pub use_value_weighting: bool,         // Weight by price*size vs just size
}
```

### 3. Core Logic

**Imbalance Calculation:**
```rust
fn calculate_imbalance(&self, orderbook: &Orderbook) -> ImbalanceResult {
    // Sum bid volume across depth_levels
    let bid_volume: Decimal = orderbook.bids
        .iter()
        .take(self.config.depth_levels)
        .map(|l| if use_value_weighting { l.price * l.size } else { l.size })
        .sum();

    // Sum ask volume across depth_levels
    let ask_volume: Decimal = orderbook.asks
        .iter()
        .take(self.config.depth_levels)
        .map(|l| if use_value_weighting { l.price * l.size } else { l.size })
        .sum();

    // Calculate ratio
    let ratio = bid_volume / ask_volume;

    ImbalanceResult { bid_volume, ask_volume, ratio }
}
```

**Signal Generation:**
- If ratio > threshold: Buy signal (heavy bids indicate upward pressure)
- If ratio < 1/threshold: Sell signal (heavy asks indicate downward pressure)
- Include imbalance ratio in signal metadata for confidence tracking

**State Management:**
- Track last signal time per token for cooldown
- Optionally track historical imbalance accuracy for adaptive thresholds

### 4. Create Config File
**File:** `config/strategies/orderbook_imbalance.toml`

### 5. Update Strategy Module
**File:** `crates/polysniper-strategies/src/lib.rs` - Add module export

### 6. Add Tests
Unit tests for:
- Imbalance calculation with various order books
- Threshold triggering logic
- Cooldown enforcement
- Value weighting vs size weighting

## Acceptance Criteria

- [ ] OrderbookImbalanceStrategy implements the Strategy trait correctly
- [ ] Calculates bid/ask volume imbalance across configurable depth
- [ ] Triggers signals when imbalance exceeds threshold
- [ ] Supports both size-weighted and value-weighted calculations
- [ ] Implements per-token cooldown
- [ ] Configuration loads from TOML file
- [ ] All unit tests passing
- [ ] Signal metadata includes imbalance ratio

## Files to Create/Modify

- `crates/polysniper-strategies/src/orderbook_imbalance.rs` - **CREATE** - Main strategy implementation
- `crates/polysniper-strategies/src/lib.rs` - **MODIFY** - Add module export
- `config/strategies/orderbook_imbalance.toml` - **CREATE** - Default configuration

## Integration Points

- **Provides**: Directional trade signals based on order book analysis
- **Consumes**: StateProvider (orderbooks), OrderbookUpdate events
- **Conflicts**: None - new strategy module

## Notes

- Consider adding support for time-weighted imbalance (imbalance persistence)
- May want to combine with price momentum for confirmation
- Large orders can significantly skew imbalance - consider outlier handling
