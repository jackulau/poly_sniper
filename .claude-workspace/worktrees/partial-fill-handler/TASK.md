---
id: partial-fill-handler
name: Partial Fill Tracking and Management
wave: 1
priority: 1
dependencies: []
estimated_hours: 5
tags: [execution, fills, tracking]
---

## Objective

Implement comprehensive partial fill tracking that monitors order status, detects partial fills, and manages remaining quantities with optional auto-resubmission.

## Context

The current system tracks `filled_size` and `remaining_size` in `OrderStatusResponse` but lacks automated handling of partial fills. This task adds a fill manager that:
1. Tracks all active orders and their fill status
2. Detects partial fills from status updates
3. Optionally resubmits remaining quantities
4. Emits events for fill notifications

## Implementation

### 1. Create Fill Manager Module

**File**: `crates/polysniper-execution/src/fill_manager.rs`

```rust
use std::collections::HashMap;
use tokio::sync::RwLock;

pub struct FillManager {
    active_orders: RwLock<HashMap<String, TrackedOrder>>,
    config: FillManagerConfig,
    event_sender: broadcast::Sender<FillEvent>,
}

pub struct TrackedOrder {
    pub order: Order,
    pub original_size: Decimal,
    pub filled_size: Decimal,
    pub remaining_size: Decimal,
    pub fills: Vec<Fill>,
    pub status: TrackedOrderStatus,
    pub created_at: DateTime<Utc>,
    pub last_checked: DateTime<Utc>,
}

pub struct Fill {
    pub size: Decimal,
    pub price: Decimal,
    pub timestamp: DateTime<Utc>,
}

pub enum TrackedOrderStatus {
    Active,
    PartiallyFilled,
    FullyFilled,
    Cancelled,
    Expired,
}

pub struct FillManagerConfig {
    pub auto_resubmit: bool,
    pub min_resubmit_size: Decimal,
    pub poll_interval_ms: u64,
    pub max_resubmit_attempts: u32,
}

pub enum FillEvent {
    PartialFill { order_id: String, fill: Fill, remaining: Decimal },
    FullFill { order_id: String, avg_price: Decimal, total_size: Decimal },
    OrderExpired { order_id: String, filled: Decimal, unfilled: Decimal },
    ResubmitTriggered { original_id: String, new_order: Order },
}

impl FillManager {
    /// Start tracking an order
    pub async fn track_order(&self, order: Order);
    
    /// Update order status from CLOB response
    pub async fn update_status(&self, order_id: &str, response: OrderStatusResponse);
    
    /// Check and handle partial fills
    pub async fn process_fills(&self) -> Vec<FillEvent>;
    
    /// Get current status of tracked order
    pub async fn get_tracked_order(&self, order_id: &str) -> Option<TrackedOrder>;
    
    /// Cancel tracking (order cancelled externally)
    pub async fn stop_tracking(&self, order_id: &str);
    
    /// Calculate VWAP for an order's fills
    pub fn calculate_vwap(fills: &[Fill]) -> Decimal;
}
```

### 2. Create Fill Polling Service

**File**: `crates/polysniper-execution/src/fill_poller.rs`

```rust
pub struct FillPoller {
    fill_manager: Arc<FillManager>,
    order_executor: Arc<dyn OrderExecutor>,
    poll_interval: Duration,
}

impl FillPoller {
    /// Run the polling loop (call from main event loop)
    pub async fn poll_once(&self) -> Result<Vec<FillEvent>, ExecutionError>;
    
    /// Start background polling task
    pub fn spawn_polling_task(self: Arc<Self>) -> JoinHandle<()>;
}
```

### 3. Integrate with Order Submission Flow

**File**: `crates/polysniper-execution/src/submitter.rs`

- After successful submission, register order with FillManager
- Add method to query fill status

### 4. Add Configuration

**File**: `config/default.toml`

```toml
[execution.fill_management]
enabled = true
auto_resubmit = false
min_resubmit_size = 10.0
poll_interval_ms = 1000
max_resubmit_attempts = 3
```

### 5. Add Fill Events to Event System

**File**: `crates/polysniper-core/src/events.rs`

Add new event variants for fill notifications that strategies can subscribe to.

## Acceptance Criteria

- [ ] FillManager correctly tracks active orders and their fill status
- [ ] Partial fills are detected and FillEvent::PartialFill emitted
- [ ] VWAP calculation is accurate for multi-fill orders
- [ ] Auto-resubmit (when enabled) creates new orders for remaining size
- [ ] Orders are cleaned up when fully filled, cancelled, or expired
- [ ] Configuration allows disabling fill management
- [ ] Unit tests cover partial fill scenarios
- [ ] Memory is bounded (orders cleaned up after completion)

## Files to Create/Modify

- `crates/polysniper-execution/src/fill_manager.rs` - **CREATE** - Core fill tracking logic
- `crates/polysniper-execution/src/fill_poller.rs` - **CREATE** - Polling service
- `crates/polysniper-execution/src/lib.rs` - **MODIFY** - Export new modules
- `crates/polysniper-execution/src/submitter.rs` - **MODIFY** - Integrate fill tracking
- `crates/polysniper-core/src/events.rs` - **MODIFY** - Add fill event types
- `crates/polysniper-core/src/types.rs` - **MODIFY** - Add config struct
- `config/default.toml` - **MODIFY** - Add configuration section

## Integration Points

- **Provides**: Fill tracking and events for other components
- **Consumes**: `OrderStatusResponse` from `OrderExecutor`
- **Conflicts**: Coordinate with cancel-replace-logic on order state management

## Testing Notes

- Mock CLOB responses with various fill scenarios
- Test auto-resubmit with min size thresholds
- Verify memory cleanup after order completion
- Test concurrent order tracking
