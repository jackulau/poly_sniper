---
id: liquidity-provision-strategy
name: Liquidity Provision Market-Making Strategy
wave: 1
priority: 2
dependencies: []
estimated_hours: 6
tags: [strategy, market-making, liquidity]
---

## Objective

Implement a liquidity provision strategy that places orders on both sides of the book to earn the bid-ask spread while managing inventory risk.

## Context

Market-making involves quoting both buy and sell prices, profiting from the spread between them. This strategy will:
- Place limit orders on both bid and ask sides
- Adjust quotes based on inventory position
- Manage risk through position limits and skewing
- Earn the spread when both sides get filled

This is more complex than directional strategies because it requires active inventory management.

## Implementation

### 1. Create Strategy File
**File:** `crates/polysniper-strategies/src/liquidity_provision.rs`

```rust
// Key components:
// - LiquidityProvisionConfig struct
// - LiquidityProvisionStrategy implementing Strategy trait
// - Quote calculation with inventory skew
// - Order refresh logic
// - Position tracking and limits
```

### 2. Configuration Structure
```rust
pub struct LiquidityProvisionConfig {
    pub enabled: bool,
    pub base_spread_pct: Decimal,         // Minimum spread to quote (e.g., 2%)
    pub order_size_usd: Decimal,          // Size per side
    pub max_position_usd: Decimal,        // Maximum inventory
    pub inventory_skew_factor: Decimal,   // How much to skew prices with inventory
    pub refresh_interval_secs: u64,       // How often to refresh quotes
    pub min_liquidity_usd: Decimal,       // Minimum market liquidity
    pub markets: Vec<String>,             // Markets to provide liquidity
    pub cancel_on_fill: bool,             // Cancel other side when one fills
}
```

### 3. Core Logic

**Quote Calculation:**
- Get mid price from orderbook
- Apply base spread: `bid = mid - spread/2`, `ask = mid + spread/2`
- Apply inventory skew: shift both quotes based on current position
  - Long position → lower prices (encourage sells)
  - Short position → higher prices (encourage buys)

**Order Management:**
- Track active orders per market
- Refresh quotes on price changes beyond threshold
- Cancel stale orders periodically
- Handle partial fills

**Risk Controls:**
- Maximum position per market
- Maximum total inventory across all markets
- Minimum profit threshold before quoting

### 4. Create Config File
**File:** `config/strategies/liquidity_provision.toml`

### 5. Update Strategy Module
**File:** `crates/polysniper-strategies/src/lib.rs` - Add module export

### 6. Add Tests
Unit tests for:
- Quote calculation
- Inventory skew logic
- Position limit enforcement
- Order refresh triggers

## Acceptance Criteria

- [ ] LiquidityProvisionStrategy implements the Strategy trait correctly
- [ ] Places both bid and ask orders around mid price
- [ ] Calculates spread correctly with configurable base spread
- [ ] Implements inventory skew to manage position risk
- [ ] Respects maximum position limits
- [ ] Refreshes quotes on significant price movements
- [ ] Configuration loads from TOML file
- [ ] All unit tests passing
- [ ] Handles partial fills appropriately

## Files to Create/Modify

- `crates/polysniper-strategies/src/liquidity_provision.rs` - **CREATE** - Main strategy implementation
- `crates/polysniper-strategies/src/lib.rs` - **MODIFY** - Add module export
- `config/strategies/liquidity_provision.toml` - **CREATE** - Default configuration

## Integration Points

- **Provides**: Two-sided liquidity quotes, spread earning
- **Consumes**: StateProvider (orderbooks, positions), SystemEvent stream
- **Conflicts**: None - new strategy module

## Notes

- This strategy generates GTC (Good-til-Cancelled) limit orders, not FOK
- Consider implementing order cancellation signals (may need new signal type)
- Inventory tracking requires integration with position updates
