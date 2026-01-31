---
id: vpin-calculator
name: VPIN (Volume-Synchronized Probability of Informed Trading) Calculator
wave: 1
priority: 1
dependencies: []
estimated_hours: 6
tags: [analysis, core, order-flow]
---

## Objective

Implement a VPIN calculator that measures order flow toxicity by tracking the probability of informed trading based on volume-synchronized buckets.

## Context

VPIN is a metric that estimates the probability that informed traders are actively trading. It measures the imbalance between buy-initiated and sell-initiated volume across volume-synchronized buckets. High VPIN values indicate elevated risk of adverse selection (trading against informed participants).

This is a foundational component for the market microstructure analysis system. Other components (strategy, events) will consume VPIN signals.

## Implementation

### 1. Create new module: `crates/polysniper-strategies/src/vpin.rs`

**Core Components:**

```rust
pub struct VpinConfig {
    pub enabled: bool,
    pub bucket_size_usd: Decimal,        // Volume per bucket (e.g., $1000)
    pub lookback_buckets: usize,          // Number of buckets for VPIN calc (e.g., 50)
    pub high_toxicity_threshold: Decimal, // Alert threshold (e.g., 0.7)
    pub low_toxicity_threshold: Decimal,  // Safe threshold (e.g., 0.3)
    pub trade_classification: TradeClassificationMethod,
}

pub enum TradeClassificationMethod {
    TickRule,      // Compare trade price to previous trade
    QuoteMidpoint, // Compare to bid-ask midpoint
    BulkVolume,    // Lee-Ready with volume weighting
}

pub struct VpinCalculator {
    config: VpinConfig,
    // Rolling bucket state per token
    buckets: HashMap<TokenId, VecDeque<VolumeBucket>>,
    current_bucket: HashMap<TokenId, VolumeBucket>,
}

pub struct VolumeBucket {
    pub buy_volume: Decimal,
    pub sell_volume: Decimal,
    pub total_volume: Decimal,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub trade_count: u32,
}

pub struct VpinResult {
    pub token_id: TokenId,
    pub vpin: Decimal,                    // 0.0 to 1.0
    pub toxicity_level: ToxicityLevel,
    pub buy_volume_pct: Decimal,
    pub sell_volume_pct: Decimal,
    pub bucket_count: usize,
    pub timestamp: DateTime<Utc>,
}

pub enum ToxicityLevel {
    Low,      // VPIN < 0.3
    Normal,   // 0.3 <= VPIN < 0.5
    Elevated, // 0.5 <= VPIN < 0.7
    High,     // VPIN >= 0.7
}
```

### 2. Implement trade classification

```rust
impl VpinCalculator {
    /// Classify a trade as buy-initiated or sell-initiated
    fn classify_trade(
        &self,
        trade_price: Decimal,
        prev_price: Option<Decimal>,
        bid: Decimal,
        ask: Decimal,
    ) -> TradeSide {
        match self.config.trade_classification {
            TradeClassificationMethod::TickRule => {
                // If price > prev_price, it's a buy
                // If price < prev_price, it's a sell
                // If equal, use previous classification
            }
            TradeClassificationMethod::QuoteMidpoint => {
                let mid = (bid + ask) / dec!(2);
                if trade_price >= mid { TradeSide::Buy } else { TradeSide::Sell }
            }
            TradeClassificationMethod::BulkVolume => {
                // Lee-Ready algorithm with bulk volume classification
            }
        }
    }

    /// Process a trade and update bucket state
    pub fn process_trade(
        &mut self,
        token_id: &TokenId,
        price: Decimal,
        size: Decimal,
        side: TradeSide,
    ) -> Option<VpinResult> {
        // Add volume to current bucket
        // If bucket is full, finalize and start new bucket
        // Recalculate VPIN across lookback window
    }

    /// Calculate VPIN from completed buckets
    pub fn calculate_vpin(&self, token_id: &TokenId) -> Option<VpinResult> {
        // VPIN = sum(|buy_vol - sell_vol|) / (2 * total_vol) across all buckets
    }
}
```

### 3. Create configuration file: `config/strategies/vpin.toml`

```toml
[strategy]
enabled = true
id = "vpin"
name = "VPIN Calculator"

[strategy.vpin]
bucket_size_usd = "1000"
lookback_buckets = 50
high_toxicity_threshold = "0.7"
low_toxicity_threshold = "0.3"
trade_classification = "QuoteMidpoint"

# Per-market overrides (optional)
[strategy.vpin.market_overrides]
# For volatile markets, use smaller buckets
# "trump-*" = { bucket_size_usd = "500", lookback_buckets = 30 }
```

### 4. Add tests in the same file

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> VpinConfig { ... }

    #[test]
    fn test_vpin_calculation_balanced_flow() {
        // Equal buy/sell volume should give VPIN ≈ 0
    }

    #[test]
    fn test_vpin_calculation_buy_dominated() {
        // All buys should give VPIN ≈ 1
    }

    #[test]
    fn test_bucket_transition() {
        // Test bucket finalization when volume threshold reached
    }

    #[test]
    fn test_trade_classification_tick_rule() { ... }

    #[test]
    fn test_trade_classification_quote_midpoint() { ... }

    #[tokio::test]
    async fn test_vpin_with_mock_state() { ... }
}
```

### 5. Export from strategies crate

Update `crates/polysniper-strategies/src/lib.rs` to export the VPIN module.

## Acceptance Criteria

- [ ] VpinCalculator correctly accumulates volume into buckets
- [ ] Trade classification works for all three methods
- [ ] VPIN calculation produces values in [0, 1] range
- [ ] Bucket rotation occurs when volume threshold is reached
- [ ] Per-token state is properly isolated
- [ ] Configuration loads from TOML file
- [ ] All tests pass (`cargo test -p polysniper-strategies`)
- [ ] Code follows existing patterns (Decimal types, async_trait, etc.)
- [ ] No clippy warnings

## Files to Create/Modify

- `crates/polysniper-strategies/src/vpin.rs` - **CREATE** - Core VPIN implementation
- `crates/polysniper-strategies/src/lib.rs` - **MODIFY** - Add `pub mod vpin;` export
- `config/strategies/vpin.toml` - **CREATE** - Configuration file

## Integration Points

- **Provides**: `VpinCalculator`, `VpinResult`, `ToxicityLevel` for use by strategy and events
- **Consumes**: Trade data (from `PartialFillEvent`, `FullFillEvent`) and orderbook data (for quote midpoint)
- **Conflicts**: None - standalone module
