---
id: multi-leg-strategy
name: Multi-Leg Correlated Market Strategy
wave: 2
priority: 5
dependencies: [arbitrage-strategy]
estimated_hours: 5
tags: [strategy, correlation, multi-market]
---

## Objective

Implement a strategy that identifies and trades correlated positions across related markets, such as election outcomes where results in one market imply results in another.

## Context

Many Polymarket markets are correlated:
- **Election markets**: "Will Biden win?" and "Will Trump win?" are inversely correlated
- **Sequential events**: "Will X happen by March?" implies "Will X happen by December?"
- **Conditional markets**: "Will Biden win AND Democrats control Senate?" depends on multiple outcomes

This strategy monitors relationships between markets and generates coordinated trades when mispricings occur across related markets.

## Implementation

### 1. Create Strategy File
**File:** `crates/polysniper-strategies/src/multi_leg.rs`

```rust
// Key components:
// - MultiLegConfig with correlation rules
// - MultiLegStrategy implementing Strategy trait
// - Correlation pair/group monitoring
// - Coordinated signal generation
```

### 2. Configuration Structure

```rust
pub struct MultiLegConfig {
    pub enabled: bool,
    pub correlation_rules: Vec<CorrelationRule>,
    pub min_edge_pct: Decimal,           // Minimum mispricing to trade
    pub order_size_usd: Decimal,         // Size per leg
    pub cooldown_secs: u64,
}

pub struct CorrelationRule {
    pub name: String,
    pub legs: Vec<CorrelationLeg>,
    pub relationship: CorrelationRelationship,
    pub expected_sum: Option<Decimal>,   // For mutually exclusive markets
}

pub struct CorrelationLeg {
    pub market_id: String,
    pub token: Outcome,                  // Yes or No
    pub weight: Decimal,                 // For complex relationships
}

pub enum CorrelationRelationship {
    MutuallyExclusive,        // Sum of YES tokens should = 1
    Inverse,                  // A going up means B goes down
    Conditional,              // A implies B (A price <= B price for YES)
    Custom { formula: String }, // Advanced expressions
}
```

### 3. Core Logic

**Mutually Exclusive Detection:**
```rust
// Example: Biden win + Trump win + Third party win = 1
fn check_mutually_exclusive(&self, rule: &CorrelationRule, state: &dyn StateProvider) -> Option<Vec<TradeSignal>> {
    let total: Decimal = rule.legs.iter()
        .filter_map(|leg| state.get_price(&leg.token_id()))
        .sum();

    let expected = rule.expected_sum.unwrap_or(Decimal::ONE);
    let edge = (expected - total).abs();

    if edge >= self.config.min_edge_pct {
        // Generate buy signals for underpriced legs
        // or identify specific mispriced leg
    }
}
```

**Inverse Correlation:**
```rust
// Example: Biden YES should roughly equal Trump NO
fn check_inverse(&self, rule: &CorrelationRule, state: &dyn StateProvider) -> Option<Vec<TradeSignal>> {
    // Get prices for both legs
    // Check if they deviate from expected inverse relationship
    // Generate spread trade signals
}
```

**Conditional/Implied:**
```rust
// Example: "Biden wins" YES price should be >= "Biden wins AND Dems control Senate" YES price
fn check_conditional(&self, rule: &CorrelationRule, state: &dyn StateProvider) -> Option<Vec<TradeSignal>> {
    // Verify implication pricing holds
    // Trade when it doesn't
}
```

### 4. Signal Coordination

Multi-leg trades need to be coordinated:
- Generate signals with shared `parent_id` to link legs
- Include leg index and total legs in metadata
- Consider adding `TradeSignalGroup` type for atomic multi-leg execution

### 5. Create Config File
**File:** `config/strategies/multi_leg.toml`

Example configuration:
```toml
[strategy]
enabled = true
min_edge_pct = "0.02"
order_size_usd = "50"
cooldown_secs = 120

[[strategy.correlation_rules]]
name = "2024 Presidential"
relationship = "MutuallyExclusive"
expected_sum = "1.0"

[[strategy.correlation_rules.legs]]
market_id = "biden-2024"
token = "Yes"
weight = "1.0"

[[strategy.correlation_rules.legs]]
market_id = "trump-2024"
token = "Yes"
weight = "1.0"
```

### 6. Update Strategy Module
**File:** `crates/polysniper-strategies/src/lib.rs` - Add module export

### 7. Add Tests
Unit tests for:
- Mutually exclusive calculation
- Inverse correlation detection
- Conditional pricing validation
- Multi-leg signal generation

## Acceptance Criteria

- [ ] MultiLegStrategy implements the Strategy trait correctly
- [ ] Supports mutually exclusive market relationship
- [ ] Supports inverse correlation relationship
- [ ] Supports conditional/implied relationships
- [ ] Generates coordinated multi-leg signals with shared parent ID
- [ ] Configuration loads from TOML file with correlation rules
- [ ] All unit tests passing
- [ ] Handles markets with different token IDs correctly

## Files to Create/Modify

- `crates/polysniper-strategies/src/multi_leg.rs` - **CREATE** - Main strategy implementation
- `crates/polysniper-strategies/src/lib.rs` - **MODIFY** - Add module export
- `config/strategies/multi_leg.toml` - **CREATE** - Default configuration with example rules

## Integration Points

- **Provides**: Correlated trade signals across multiple markets
- **Consumes**: StateProvider (multiple markets/prices), SystemEvent stream
- **Depends on**: arbitrage-strategy (shares some pricing logic patterns)
- **Conflicts**: None - new strategy module

## Notes

- May want to leverage arbitrage strategy's price comparison logic
- Consider market discovery for automatic correlation detection
- Real-world correlations may be imperfect - need tolerance settings
- Execution timing across legs is important - may need TWAP/VWAP integration
