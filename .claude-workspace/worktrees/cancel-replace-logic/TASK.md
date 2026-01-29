---
id: cancel-replace-logic
name: Cancel-and-Replace Order Management
wave: 2
priority: 1
dependencies: [partial-fill-handler]
estimated_hours: 5
tags: [execution, orders, management]
---

## Objective

Implement cancel-and-replace logic that automatically adjusts resting limit orders as market conditions change, maintaining optimal positioning without manual intervention.

## Context

When market prices move, existing limit orders may become stale or suboptimal. This task adds an order manager that:
1. Monitors market prices relative to resting orders
2. Decides when to cancel and replace orders
3. Handles the cancel-replace sequence atomically
4. Preserves partial fills during replacement

This task depends on partial-fill-handler because it needs to preserve fill state when replacing orders.

## Implementation

### 1. Create Order Manager Module

**File**: `crates/polysniper-execution/src/order_manager.rs`

```rust
pub struct OrderManager {
    /// Active orders being managed
    managed_orders: RwLock<HashMap<String, ManagedOrder>>,
    /// Fill manager for tracking partial fills
    fill_manager: Arc<FillManager>,
    /// Order executor for cancel/submit
    executor: Arc<dyn OrderExecutor>,
    /// Configuration
    config: OrderManagerConfig,
}

pub struct ManagedOrder {
    pub order: Order,
    pub management_policy: ManagementPolicy,
    pub replace_count: u32,
    pub total_filled: Decimal,
    pub created_at: DateTime<Utc>,
    pub last_replaced: Option<DateTime<Utc>>,
}

pub struct ManagementPolicy {
    /// How far price can move before triggering replace (in bps)
    pub price_drift_threshold_bps: Decimal,
    /// Minimum time between replacements
    pub min_replace_interval_ms: u64,
    /// Maximum number of replacements
    pub max_replacements: u32,
    /// Whether to chase price (follow market direction)
    pub chase_enabled: bool,
    /// How aggressively to chase (0.0 = stay at original, 1.0 = follow mid)
    pub chase_aggression: f64,
}

pub struct OrderManagerConfig {
    pub enabled: bool,
    pub default_policy: ManagementPolicy,
    pub check_interval_ms: u64,
}

pub enum ReplaceDecision {
    Hold,
    Replace { new_price: Decimal, reason: String },
    Cancel { reason: String },
}

impl OrderManager {
    /// Start managing an order
    pub async fn manage_order(
        &self,
        order: Order,
        policy: Option<ManagementPolicy>,
    ) -> Result<String, ExecutionError>;
    
    /// Stop managing an order (but don't cancel it)
    pub async fn stop_managing(&self, order_id: &str);
    
    /// Check all managed orders and perform replacements
    pub async fn check_and_replace(&self, state: &dyn StateProvider) 
        -> Vec<ReplaceResult>;
    
    /// Evaluate whether an order should be replaced
    fn evaluate_replace(
        &self,
        managed: &ManagedOrder,
        current_mid: Decimal,
        orderbook: &Orderbook,
    ) -> ReplaceDecision;
    
    /// Execute cancel-and-replace sequence
    async fn execute_replace(
        &self,
        managed: &mut ManagedOrder,
        new_price: Decimal,
    ) -> Result<ReplaceResult, ExecutionError>;
}

pub struct ReplaceResult {
    pub original_order_id: String,
    pub new_order_id: Option<String>,
    pub action: ReplaceAction,
    pub preserved_fill: Decimal,
}

pub enum ReplaceAction {
    Replaced { old_price: Decimal, new_price: Decimal },
    Cancelled { reason: String },
    Failed { error: String },
    Skipped { reason: String },
}
```

### 2. Add Replace Events

**File**: `crates/polysniper-core/src/events.rs`

```rust
pub struct OrderReplacedEvent {
    pub original_order_id: String,
    pub new_order_id: String,
    pub old_price: Decimal,
    pub new_price: Decimal,
    pub preserved_fill: Decimal,
    pub reason: String,
}
```

### 3. Integration with Main Loop

**File**: `src/main.rs`

- Add OrderManager to application state
- Call `check_and_replace()` periodically in event loop
- Handle ReplaceResult events

### 4. Add Configuration

**File**: `config/default.toml`

```toml
[execution.order_management]
enabled = true
check_interval_ms = 500

[execution.order_management.default_policy]
price_drift_threshold_bps = 25
min_replace_interval_ms = 1000
max_replacements = 10
chase_enabled = true
chase_aggression = 0.5
```

## Acceptance Criteria

- [ ] Orders are replaced when price drifts beyond threshold
- [ ] Partial fills are preserved during replacement
- [ ] Minimum interval between replacements is respected
- [ ] Maximum replacement count is enforced
- [ ] Chase logic correctly follows market direction
- [ ] Cancel-replace sequence handles failures gracefully
- [ ] Events emitted for all replace actions
- [ ] Configuration allows disabling management
- [ ] Unit tests cover replace decision logic

## Files to Create/Modify

- `crates/polysniper-execution/src/order_manager.rs` - **CREATE** - Core order management
- `crates/polysniper-execution/src/lib.rs` - **MODIFY** - Export new module
- `crates/polysniper-core/src/events.rs` - **MODIFY** - Add replace events
- `crates/polysniper-core/src/types.rs` - **MODIFY** - Add config struct
- `src/main.rs` - **MODIFY** - Integrate order manager
- `config/default.toml` - **MODIFY** - Add configuration section

## Integration Points

- **Provides**: Automated order management for passive strategies
- **Consumes**: FillManager for fill tracking, OrderExecutor for submissions
- **Depends On**: partial-fill-handler must be complete for fill preservation
- **Conflicts**: Coordinate changes to main.rs event loop

## Testing Notes

- Mock price movements to test replace triggers
- Test partial fill preservation across replacements
- Verify rate limiting (min interval, max replacements)
- Test failure scenarios (cancel fails, submit fails)
- Verify chase logic calculations
