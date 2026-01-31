---
id: enhanced-fill-probability
name: Enhanced Fill Probability Modeling
wave: 1
priority: 3
dependencies: []
estimated_hours: 5
tags: [execution, orderbook, probability]
---

## Objective

Implement more accurate fill probability modeling with active position tracking, order age decay, and price level analysis.

## Context

The existing `queue_estimator.rs` has basic fill rate tracking and queue position estimation. This enhancement adds:
- **Active position tracking**: Monitor our orders' queue positions as orderbook changes
- **Order age decay**: Bi-modal model where older orders are more likely at front initially, then decay
- **Price level analysis**: Track how often price touches levels and fill behavior when touched
- **Combined probability methods**: Multiple estimation approaches with confidence scoring

## Implementation

### 1. Create Price Level Analyzer

Create `crates/polysniper-execution/src/price_level_analyzer.rs`:

```rust
use rust_decimal::Decimal;
use std::collections::HashMap;
use tokio::sync::RwLock;

pub struct PriceLevelAnalyzer {
    // Track behavior at each price level per token
    price_levels: RwLock<HashMap<TokenId, HashMap<Decimal, PriceLevelStats>>>,
    config: PriceLevelConfig,
}

pub struct PriceLevelConfig {
    pub history_window_secs: u64,        // How far back to track (default: 3600)
    pub min_touches_for_stats: u32,      // Minimum touches for valid stats (default: 5)
}

pub struct PriceLevelStats {
    pub touches: u32,                     // Times price reached this level
    pub time_at_level_ms: u64,            // Cumulative time at this level
    pub volume_traded: Decimal,           // Volume traded at this level
    pub orders_filled: u32,               // Orders completely filled
    pub orders_partially_filled: u32,     // Orders partially filled
    pub avg_fill_pct_when_touched: Decimal, // Average fill % when price touches
    pub last_touch: DateTime<Utc>,
}

impl PriceLevelAnalyzer {
    pub fn new(config: PriceLevelConfig) -> Self;

    /// Record that price touched a level
    pub async fn record_touch(&self, token_id: &TokenId, price: Decimal, duration_ms: u64);

    /// Record a fill at a price level
    pub async fn record_fill(&self, token_id: &TokenId, price: Decimal, fill_pct: Decimal);

    /// Get probability of fill if price touches our level
    pub fn fill_probability_on_touch(&self, token_id: &TokenId, price: Decimal) -> Option<Decimal>;

    /// Get average time price spends at a level
    pub fn avg_time_at_level(&self, token_id: &TokenId, price: Decimal) -> Option<u64>;
}
```

### 2. Enhance Queue Estimator with Active Tracking

Modify `crates/polysniper-execution/src/queue_estimator.rs`:

```rust
pub struct QueuePositionState {
    pub order_id: OrderId,
    pub token_id: TokenId,
    pub price: Decimal,
    pub side: Side,
    pub size: Decimal,
    pub initial_size_ahead: Decimal,
    pub current_size_ahead: Decimal,
    pub order_placed_at: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
    pub observed_departures: u32,        // Orders ahead that left queue
}

impl QueueEstimator {
    /// Start tracking an order's queue position
    pub async fn start_tracking(&self, order: &Order);

    /// Update tracked positions from orderbook change
    pub async fn update_positions(&self, token_id: &TokenId, orderbook: &Orderbook);

    /// Get tracked position state
    pub async fn get_tracked_position(&self, order_id: &OrderId) -> Option<QueuePositionState>;

    /// Stop tracking an order (on fill or cancel)
    pub async fn stop_tracking(&self, order_id: &OrderId);
}
```

### 3. Add Age Decay Model

```rust
impl QueueEstimator {
    /// Apply age-based adjustment to fill probability
    fn apply_age_factor(&self, state: &QueuePositionState) -> Decimal {
        let age_secs = (Utc::now() - state.order_placed_at).num_seconds() as f64;

        // Bi-modal: first 5 mins older = better (queue priority), then decay
        if age_secs < 300.0 {
            // Linear increase: 1.0 to 1.2 over 5 minutes
            Decimal::ONE + Decimal::try_from(age_secs / 1500.0).unwrap_or(Decimal::ZERO)
        } else {
            // Exponential decay after 5 minutes (half-life ~30 mins)
            let decay = (-(age_secs - 300.0) / 1800.0).exp();
            Decimal::try_from(1.2 * decay).unwrap_or(dec!(0.5))
        }
    }
}
```

### 4. Implement Combined Probability Method

```rust
pub struct FillProbability {
    pub probability: Decimal,              // Final probability 0.0-1.0
    pub confidence: Decimal,               // Confidence 0.0-1.0
    pub expected_time_secs: Option<u64>,   // Expected fill time if filled
    pub method: ProbabilityMethod,
    pub components: ProbabilityComponents,
}

pub struct ProbabilityComponents {
    pub historical_rate_prob: Option<Decimal>,
    pub queue_position_prob: Option<Decimal>,
    pub price_level_prob: Option<Decimal>,
    pub age_factor: Decimal,
}

pub enum ProbabilityMethod {
    HistoricalRate,      // Based on fill rate history only
    QueuePosition,       // Based on position in queue only
    PriceLevel,          // Based on price level statistics
    Combined,            // Weighted combination
    Insufficient,        // Not enough data for estimate
}

impl QueueEstimator {
    pub async fn calculate_fill_probability(
        &self,
        order_id: &OrderId,
        time_horizon_secs: u64,
        price_analyzer: &PriceLevelAnalyzer,
    ) -> FillProbability {
        // Combine multiple probability sources with confidence weighting
    }
}
```

### 5. Add Metrics

Add to `crates/polysniper-observability/src/metrics.rs`:
- `fill_probability_estimate` histogram with labels for method
- `fill_probability_confidence` histogram
- `queue_position_tracked_orders` gauge
- `price_level_touches` counter
- `fill_probability_accuracy` histogram (compare estimates to actuals)

## Acceptance Criteria

- [ ] PriceLevelAnalyzer tracks touch statistics per price level
- [ ] QueueEstimator actively tracks our orders' positions
- [ ] Position updates correctly when orderbook changes
- [ ] Age decay model applied correctly (increase then decay)
- [ ] Combined probability uses multiple methods with confidence weighting
- [ ] Metrics exported for monitoring and accuracy tracking
- [ ] All existing tests still pass
- [ ] New tests for price level and position tracking

## Files to Create/Modify

- `crates/polysniper-execution/src/price_level_analyzer.rs` - Create new file
- `crates/polysniper-execution/src/queue_estimator.rs` - Enhance with active tracking
- `crates/polysniper-execution/src/lib.rs` - Add module export
- `crates/polysniper-observability/src/metrics.rs` - Add probability metrics
- `crates/polysniper-core/src/types.rs` - Add PriceLevelConfig

## Integration Points

- **Provides**: Enhanced fill probability for order decisions
- **Consumes**: Orderbook updates, fill events
- **Conflicts**: Enhances queue_estimator.rs (no other tasks touch this file)
