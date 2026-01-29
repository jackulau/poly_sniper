---
id: queue-position-estimation
name: Queue Position Estimation for Limit Orders
wave: 1
priority: 2
dependencies: []
estimated_hours: 4
tags: [execution, orderbook, estimation]
---

## Objective

Implement queue position estimation that tracks where limit orders sit in the orderbook queue and estimates time-to-fill based on historical fill rates and current market activity.

## Context

When placing limit orders, it's valuable to know your position in the queue and estimated time to fill. This helps strategies decide whether to place passive orders or cross the spread. The estimator uses:
1. Orderbook depth analysis to determine queue position
2. Historical trade flow to estimate fill rates
3. Market activity metrics to adjust predictions

## Implementation

### 1. Create Queue Position Estimator Module

**File**: `crates/polysniper-execution/src/queue_estimator.rs`

```rust
use std::collections::{HashMap, VecDeque};

pub struct QueueEstimator {
    /// Historical fill rates per price level
    fill_rate_history: RwLock<HashMap<TokenId, FillRateTracker>>,
    /// Configuration
    config: QueueEstimatorConfig,
}

pub struct FillRateTracker {
    /// Recent fills at each price level
    fills_by_price: HashMap<Decimal, VecDeque<FillRecord>>,
    /// Overall fill rate (size/second)
    avg_fill_rate: Decimal,
    /// Last update timestamp
    last_update: DateTime<Utc>,
}

pub struct FillRecord {
    pub size: Decimal,
    pub timestamp: DateTime<Utc>,
}

pub struct QueueEstimatorConfig {
    pub history_window_secs: u64,      // How far back to look for fill rates
    pub min_samples_for_estimate: u32, // Minimum fills needed for estimation
    pub confidence_decay_factor: f64,  // How quickly confidence decays with time
}

pub struct QueuePosition {
    pub price_level: Decimal,
    pub size_ahead: Decimal,           // Total size ahead in queue
    pub estimated_position: u32,       // Approximate position in queue
    pub estimated_time_to_fill_secs: Option<f64>,
    pub confidence: f64,               // 0.0 to 1.0
    pub fill_probability_1min: f64,    // Probability of fill in next minute
    pub fill_probability_5min: f64,    // Probability of fill in 5 minutes
}

impl QueueEstimator {
    /// Estimate queue position for a hypothetical order
    pub async fn estimate_position(
        &self,
        token_id: &TokenId,
        orderbook: &Orderbook,
        side: Side,
        price: Decimal,
        size: Decimal,
    ) -> QueuePosition;
    
    /// Update fill rate history from observed trade
    pub async fn record_fill(
        &self,
        token_id: &TokenId,
        price: Decimal,
        size: Decimal,
    );
    
    /// Get current estimated fill rate at a price level
    pub async fn get_fill_rate(
        &self,
        token_id: &TokenId,
        price: Decimal,
    ) -> Option<Decimal>;
    
    /// Clean up old history entries
    pub async fn cleanup_old_entries(&self);
}
```

### 2. Integrate with WebSocket Trade Feed

**File**: `crates/polysniper-data/src/ws_manager.rs`

- Add handler for trade messages to feed fill rate tracker
- Parse trade events and forward to QueueEstimator

### 3. Add Queue Position to Order Tracking

**File**: `crates/polysniper-execution/src/fill_manager.rs` (if exists from partial-fill-handler)

Or create integration point in submitter:

```rust
pub struct OrderWithQueueInfo {
    pub order: Order,
    pub queue_position: Option<QueuePosition>,
    pub last_queue_update: DateTime<Utc>,
}
```

### 4. Add Configuration

**File**: `config/default.toml`

```toml
[execution.queue_estimation]
enabled = true
history_window_secs = 300
min_samples_for_estimate = 10
confidence_decay_factor = 0.95
```

### 5. Add Queue Events

**File**: `crates/polysniper-core/src/events.rs`

```rust
pub struct QueueUpdateEvent {
    pub order_id: String,
    pub token_id: TokenId,
    pub position: QueuePosition,
}
```

## Acceptance Criteria

- [ ] Queue position accurately reflects size ahead at price level
- [ ] Time-to-fill estimates are reasonable based on historical fill rates
- [ ] Fill rates update in real-time from trade feed
- [ ] Confidence decreases appropriately when data is stale
- [ ] Memory is bounded with rolling window cleanup
- [ ] Configuration allows disabling estimation
- [ ] Unit tests cover various market conditions

## Files to Create/Modify

- `crates/polysniper-execution/src/queue_estimator.rs` - **CREATE** - Core estimation logic
- `crates/polysniper-execution/src/lib.rs` - **MODIFY** - Export new module
- `crates/polysniper-data/src/ws_manager.rs` - **MODIFY** - Feed trade data to estimator
- `crates/polysniper-core/src/events.rs` - **MODIFY** - Add queue events
- `crates/polysniper-core/src/types.rs` - **MODIFY** - Add config struct
- `config/default.toml` - **MODIFY** - Add configuration section

## Integration Points

- **Provides**: Queue position estimates for order placement decisions
- **Consumes**: Orderbook from StateProvider, trade feed from WebSocket
- **Conflicts**: Avoid major changes to ws_manager message handling (just add new handler)

## Testing Notes

- Create mock trade sequences to test fill rate calculation
- Test queue position with various orderbook depths
- Verify confidence decay over time
- Test cleanup of old history entries
