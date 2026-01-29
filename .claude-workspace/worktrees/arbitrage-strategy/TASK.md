---
id: arbitrage-strategy
name: YES/NO Token Arbitrage Strategy
wave: 1
priority: 1
dependencies: []
estimated_hours: 4
tags: [strategy, arbitrage, core]
---

## Objective

Implement an arbitrage strategy that detects and exploits price discrepancies between YES/NO tokens when their combined prices deviate from $1.00.

## Context

In Polymarket, YES and NO tokens for any market should theoretically sum to $1.00 (minus spread). When the combined prices deviate significantly from this, there's an arbitrage opportunity:
- If YES + NO < $1.00 - fees: Buy both tokens, guaranteed profit at resolution
- If YES + NO > $1.00 + fees: This shouldn't happen often, but indicates mispricing

This strategy monitors all markets for these pricing inefficiencies and generates appropriate trade signals.

## Implementation

### 1. Create Strategy File
**File:** `crates/polysniper-strategies/src/arbitrage.rs`

```rust
// Key components:
// - ArbitrageConfig struct with configurable thresholds
// - ArbitrageStrategy implementing Strategy trait
// - Price pair monitoring and discrepancy detection
// - Dual-leg trade signal generation
```

### 2. Configuration Structure
```rust
pub struct ArbitrageConfig {
    pub enabled: bool,
    pub min_edge_pct: Decimal,           // Minimum deviation from $1 (e.g., 1%)
    pub order_size_usd: Decimal,         // Size per leg
    pub min_liquidity_usd: Decimal,      // Minimum market liquidity
    pub markets: Vec<String>,            // Empty = all markets
    pub cooldown_secs: u64,              // Per-market cooldown
    pub max_slippage_pct: Decimal,       // Maximum acceptable slippage
}
```

### 3. Core Logic
- Listen to `OrderbookUpdate` and `PriceChange` events
- For each market, fetch both YES and NO token prices
- Calculate combined price: `yes_mid + no_mid`
- If `combined < (1.0 - min_edge_pct)`: Generate buy signals for both tokens
- Track cooldowns to avoid repeated triggers
- Include spread costs in edge calculation

### 4. Create Config File
**File:** `config/strategies/arbitrage.toml`

### 5. Update Strategy Module
**File:** `crates/polysniper-strategies/src/lib.rs` - Add module export

### 6. Add Tests
Unit tests for:
- Edge detection logic
- Signal generation
- Cooldown behavior
- Configuration parsing

## Acceptance Criteria

- [ ] ArbitrageStrategy implements the Strategy trait correctly
- [ ] Detects price discrepancies when YES + NO deviates from $1.00
- [ ] Generates paired buy signals for both tokens when edge exists
- [ ] Respects minimum edge threshold (configurable)
- [ ] Accounts for bid-ask spreads in edge calculation
- [ ] Implements per-market cooldown to avoid spam
- [ ] Configuration loads from TOML file
- [ ] All unit tests passing
- [ ] Integrates with existing event bus

## Files to Create/Modify

- `crates/polysniper-strategies/src/arbitrage.rs` - **CREATE** - Main strategy implementation
- `crates/polysniper-strategies/src/lib.rs` - **MODIFY** - Add module export
- `config/strategies/arbitrage.toml` - **CREATE** - Default configuration

## Integration Points

- **Provides**: Arbitrage detection and dual-leg trade signals
- **Consumes**: StateProvider (market data, orderbooks), SystemEvent stream
- **Conflicts**: None - new strategy module
