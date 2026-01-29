//! Order Manager - Cancel-and-Replace Logic for Managing Resting Orders
//!
//! This module provides automatic order management that monitors market prices
//! relative to resting orders and performs cancel-and-replace operations when
//! prices drift beyond configured thresholds.

use crate::FillManager;
use chrono::{DateTime, Utc};
use polysniper_core::{
    events::OrderReplacedEvent, ExecutionError, Order, OrderExecutor, OrderManagementConfig,
    Orderbook, Side, StateProvider, SystemEvent,
};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, info, warn};

/// Policy for managing individual orders
#[derive(Debug, Clone)]
pub struct ManagementPolicy {
    /// How far price can move before triggering replace (in basis points)
    pub price_drift_threshold_bps: Decimal,
    /// Minimum time between replacements (milliseconds)
    pub min_replace_interval_ms: u64,
    /// Maximum number of replacements allowed
    pub max_replacements: u32,
    /// Whether to chase price (follow market direction)
    pub chase_enabled: bool,
    /// How aggressively to chase (0.0 = stay at original, 1.0 = follow mid)
    pub chase_aggression: f64,
}

impl Default for ManagementPolicy {
    fn default() -> Self {
        Self {
            price_drift_threshold_bps: Decimal::new(25, 0), // 25 bps = 0.25%
            min_replace_interval_ms: 1000,
            max_replacements: 10,
            chase_enabled: true,
            chase_aggression: 0.5,
        }
    }
}

impl From<&OrderManagementConfig> for ManagementPolicy {
    fn from(config: &OrderManagementConfig) -> Self {
        Self {
            price_drift_threshold_bps: config.default_policy.price_drift_threshold_bps,
            min_replace_interval_ms: config.default_policy.min_replace_interval_ms,
            max_replacements: config.default_policy.max_replacements,
            chase_enabled: config.default_policy.chase_enabled,
            chase_aggression: config.default_policy.chase_aggression,
        }
    }
}

/// A managed order with tracking metadata
#[derive(Debug, Clone)]
pub struct ManagedOrder {
    pub order: Order,
    pub management_policy: ManagementPolicy,
    pub replace_count: u32,
    pub total_filled: Decimal,
    pub created_at: DateTime<Utc>,
    pub last_replaced: Option<DateTime<Utc>>,
    pub original_price: Decimal,
}

impl ManagedOrder {
    fn new(order: Order, policy: ManagementPolicy) -> Self {
        let original_price = order.price;
        Self {
            order,
            management_policy: policy,
            replace_count: 0,
            total_filled: Decimal::ZERO,
            created_at: Utc::now(),
            last_replaced: None,
            original_price,
        }
    }

    fn can_replace(&self) -> bool {
        if self.replace_count >= self.management_policy.max_replacements {
            return false;
        }

        if let Some(last_replaced) = self.last_replaced {
            let elapsed_ms = (Utc::now() - last_replaced).num_milliseconds() as u64;
            if elapsed_ms < self.management_policy.min_replace_interval_ms {
                return false;
            }
        }

        true
    }
}

/// Decision from evaluating whether to replace an order
#[derive(Debug, Clone)]
pub enum ReplaceDecision {
    /// Keep the order as-is
    Hold,
    /// Replace with a new price
    Replace { new_price: Decimal, reason: String },
    /// Cancel the order entirely
    Cancel { reason: String },
}

/// Result of a replace operation
#[derive(Debug, Clone)]
pub struct ReplaceResult {
    pub original_order_id: String,
    pub new_order_id: Option<String>,
    pub action: ReplaceAction,
    pub preserved_fill: Decimal,
}

/// The action taken during replacement
#[derive(Debug, Clone)]
pub enum ReplaceAction {
    Replaced {
        old_price: Decimal,
        new_price: Decimal,
    },
    Cancelled {
        reason: String,
    },
    Failed {
        error: String,
    },
    Skipped {
        reason: String,
    },
}

/// Order Manager for automatic cancel-and-replace operations
pub struct OrderManager {
    managed_orders: RwLock<HashMap<String, ManagedOrder>>,
    fill_manager: Arc<FillManager>,
    executor: Arc<dyn OrderExecutor>,
    config: OrderManagementConfig,
    event_sender: broadcast::Sender<SystemEvent>,
}

impl OrderManager {
    /// Create a new OrderManager
    pub fn new(
        fill_manager: Arc<FillManager>,
        executor: Arc<dyn OrderExecutor>,
        config: OrderManagementConfig,
    ) -> Self {
        let (event_sender, _) = broadcast::channel(1024);
        Self {
            managed_orders: RwLock::new(HashMap::new()),
            fill_manager,
            executor,
            config,
            event_sender,
        }
    }

    /// Subscribe to order management events
    pub fn subscribe(&self) -> broadcast::Receiver<SystemEvent> {
        self.event_sender.subscribe()
    }

    /// Start managing an order with optional custom policy
    pub async fn manage_order(
        &self,
        order: Order,
        policy: Option<ManagementPolicy>,
    ) -> Result<String, ExecutionError> {
        if !self.config.enabled {
            return Ok(order.id.clone());
        }

        let order_id = order.id.clone();
        let policy = policy.unwrap_or_else(|| ManagementPolicy::from(&self.config));
        let managed = ManagedOrder::new(order.clone(), policy);

        debug!(order_id = %order_id, "Starting to manage order");

        // Also track with fill manager for partial fill detection
        self.fill_manager.track_order(order).await;

        let mut orders = self.managed_orders.write().await;
        orders.insert(order_id.clone(), managed);

        Ok(order_id)
    }

    /// Stop managing an order without cancelling it
    pub async fn stop_managing(&self, order_id: &str) {
        let mut orders = self.managed_orders.write().await;
        if orders.remove(order_id).is_some() {
            debug!(order_id = %order_id, "Stopped managing order");
        }
    }

    /// Check all managed orders and perform replacements as needed
    pub async fn check_and_replace(&self, state: &dyn StateProvider) -> Vec<ReplaceResult> {
        if !self.config.enabled {
            return Vec::new();
        }

        let mut results = Vec::new();

        // Get a snapshot of orders to check
        let order_ids: Vec<String> = {
            let orders = self.managed_orders.read().await;
            orders.keys().cloned().collect()
        };

        for order_id in order_ids {
            if let Some(result) = self.check_single_order(&order_id, state).await {
                results.push(result);
            }
        }

        results
    }

    /// Check a single order and potentially replace it
    async fn check_single_order(
        &self,
        order_id: &str,
        state: &dyn StateProvider,
    ) -> Option<ReplaceResult> {
        // Get the managed order
        let managed = {
            let orders = self.managed_orders.read().await;
            orders.get(order_id)?.clone()
        };

        // Get current market data
        let orderbook = state.get_orderbook(&managed.order.token_id).await?;
        let current_mid = orderbook.mid_price()?;

        // Evaluate whether to replace
        let decision = self.evaluate_replace(&managed, current_mid, &orderbook);

        match decision {
            ReplaceDecision::Hold => None,
            ReplaceDecision::Replace { new_price, reason } => {
                Some(self.execute_replace(order_id, new_price, &reason).await)
            }
            ReplaceDecision::Cancel { reason } => {
                Some(self.execute_cancel(order_id, &reason).await)
            }
        }
    }

    /// Evaluate whether an order should be replaced based on market conditions
    fn evaluate_replace(
        &self,
        managed: &ManagedOrder,
        current_mid: Decimal,
        orderbook: &Orderbook,
    ) -> ReplaceDecision {
        // Check if replacement is allowed
        if !managed.can_replace() {
            return ReplaceDecision::Hold;
        }

        let order_price = managed.order.price;
        let threshold_bps = managed.management_policy.price_drift_threshold_bps;

        // Calculate price drift in basis points
        let drift_bps = if order_price.is_zero() {
            Decimal::ZERO
        } else {
            ((current_mid - order_price).abs() / order_price) * Decimal::new(10000, 0)
        };

        if drift_bps < threshold_bps {
            return ReplaceDecision::Hold;
        }

        // Calculate new price based on chase settings
        let new_price = if managed.management_policy.chase_enabled {
            self.calculate_chase_price(managed, current_mid, orderbook)
        } else {
            // Just adjust to be at the threshold boundary
            self.calculate_threshold_price(managed, current_mid)
        };

        // Validate new price is different enough to warrant replacement
        if (new_price - order_price).abs() < Decimal::new(1, 3) {
            // Less than 0.001 difference
            return ReplaceDecision::Hold;
        }

        ReplaceDecision::Replace {
            new_price,
            reason: format!(
                "Price drifted {}bps from order (mid: {}, order: {})",
                drift_bps.round_dp(2),
                current_mid,
                order_price
            ),
        }
    }

    /// Calculate chase price based on aggression setting
    fn calculate_chase_price(
        &self,
        managed: &ManagedOrder,
        current_mid: Decimal,
        orderbook: &Orderbook,
    ) -> Decimal {
        let aggression = Decimal::try_from(managed.management_policy.chase_aggression)
            .unwrap_or(Decimal::new(5, 1)); // Default 0.5

        let original_price = managed.original_price;

        // For buys, we want to stay at or below mid
        // For sells, we want to stay at or above mid
        let target = match managed.order.side {
            Side::Buy => {
                // Chase up towards the ask (more aggressive = closer to ask)
                let best_ask = orderbook.best_ask().unwrap_or(current_mid);
                current_mid + (best_ask - current_mid) * aggression
            }
            Side::Sell => {
                // Chase down towards the bid (more aggressive = closer to bid)
                let best_bid = orderbook.best_bid().unwrap_or(current_mid);
                current_mid - (current_mid - best_bid) * aggression
            }
        };

        // Blend between original price and target based on aggression
        original_price + (target - original_price) * aggression
    }

    /// Calculate price at threshold boundary (non-chase mode)
    fn calculate_threshold_price(&self, managed: &ManagedOrder, current_mid: Decimal) -> Decimal {
        let threshold_pct =
            managed.management_policy.price_drift_threshold_bps / Decimal::new(10000, 0);
        let order_price = managed.order.price;

        match managed.order.side {
            Side::Buy => {
                // If mid moved up, adjust our buy up but stay below mid
                if current_mid > order_price {
                    (current_mid * (Decimal::ONE - threshold_pct)).min(current_mid)
                } else {
                    order_price
                }
            }
            Side::Sell => {
                // If mid moved down, adjust our sell down but stay above mid
                if current_mid < order_price {
                    (current_mid * (Decimal::ONE + threshold_pct)).max(current_mid)
                } else {
                    order_price
                }
            }
        }
    }

    /// Execute a cancel-and-replace operation
    async fn execute_replace(
        &self,
        order_id: &str,
        new_price: Decimal,
        reason: &str,
    ) -> ReplaceResult {
        let mut orders = self.managed_orders.write().await;

        let Some(managed) = orders.get_mut(order_id) else {
            return ReplaceResult {
                original_order_id: order_id.to_string(),
                new_order_id: None,
                action: ReplaceAction::Failed {
                    error: "Order not found in managed orders".to_string(),
                },
                preserved_fill: Decimal::ZERO,
            };
        };

        // Get current fill state
        let tracked = self.fill_manager.get_tracked_order(order_id).await;
        let preserved_fill = tracked.map(|t| t.filled_size).unwrap_or(Decimal::ZERO);

        let old_price = managed.order.price;

        // Cancel the existing order
        if let Err(e) = self.executor.cancel_order(order_id).await {
            warn!(order_id = %order_id, error = %e, "Failed to cancel order for replacement");
            return ReplaceResult {
                original_order_id: order_id.to_string(),
                new_order_id: None,
                action: ReplaceAction::Failed {
                    error: format!("Cancel failed: {}", e),
                },
                preserved_fill,
            };
        }

        // Calculate remaining size (original size minus fills)
        let remaining_size = managed.order.size - preserved_fill;
        if remaining_size <= Decimal::ZERO {
            info!(order_id = %order_id, "Order fully filled, no replacement needed");
            orders.remove(order_id);
            return ReplaceResult {
                original_order_id: order_id.to_string(),
                new_order_id: None,
                action: ReplaceAction::Skipped {
                    reason: "Order fully filled".to_string(),
                },
                preserved_fill,
            };
        }

        // Create replacement order
        let new_order = Order {
            id: format!("{}_replace_{}", order_id, managed.replace_count + 1),
            market_id: managed.order.market_id.clone(),
            token_id: managed.order.token_id.clone(),
            side: managed.order.side,
            price: new_price,
            size: remaining_size,
            order_type: managed.order.order_type,
            signal_id: managed.order.signal_id.clone(),
            created_at: Utc::now(),
        };

        // Submit the replacement
        let new_order_id = match self.executor.submit_order(new_order.clone()).await {
            Ok(id) => id,
            Err(e) => {
                warn!(order_id = %order_id, error = %e, "Failed to submit replacement order");
                orders.remove(order_id);
                return ReplaceResult {
                    original_order_id: order_id.to_string(),
                    new_order_id: None,
                    action: ReplaceAction::Failed {
                        error: format!("Submit replacement failed: {}", e),
                    },
                    preserved_fill,
                };
            }
        };

        info!(
            original_order_id = %order_id,
            new_order_id = %new_order_id,
            old_price = %old_price,
            new_price = %new_price,
            preserved_fill = %preserved_fill,
            reason = %reason,
            "Order replaced"
        );

        // Update tracking
        let market_id = managed.order.market_id.clone();
        managed.order = new_order.clone();
        managed.order.id = new_order_id.clone();
        managed.replace_count += 1;
        managed.total_filled = preserved_fill;
        managed.last_replaced = Some(Utc::now());

        // Track the new order with fill manager
        self.fill_manager.track_order(new_order).await;

        // Remove old entry, insert with new ID
        let updated_managed = managed.clone();
        orders.remove(order_id);
        orders.insert(new_order_id.clone(), updated_managed);

        // Publish event
        let event = SystemEvent::OrderReplaced(OrderReplacedEvent {
            original_order_id: order_id.to_string(),
            new_order_id: new_order_id.clone(),
            market_id,
            old_price,
            new_price,
            preserved_fill,
            reason: reason.to_string(),
            timestamp: Utc::now(),
        });
        let _ = self.event_sender.send(event);

        ReplaceResult {
            original_order_id: order_id.to_string(),
            new_order_id: Some(new_order_id),
            action: ReplaceAction::Replaced {
                old_price,
                new_price,
            },
            preserved_fill,
        }
    }

    /// Execute a cancel operation
    async fn execute_cancel(&self, order_id: &str, reason: &str) -> ReplaceResult {
        let tracked = self.fill_manager.get_tracked_order(order_id).await;
        let preserved_fill = tracked.map(|t| t.filled_size).unwrap_or(Decimal::ZERO);

        if let Err(e) = self.executor.cancel_order(order_id).await {
            warn!(order_id = %order_id, error = %e, "Failed to cancel order");
            return ReplaceResult {
                original_order_id: order_id.to_string(),
                new_order_id: None,
                action: ReplaceAction::Failed {
                    error: format!("Cancel failed: {}", e),
                },
                preserved_fill,
            };
        }

        // Remove from tracking
        let mut orders = self.managed_orders.write().await;
        orders.remove(order_id);
        self.fill_manager.stop_tracking(order_id).await;

        info!(order_id = %order_id, reason = %reason, "Order cancelled");

        ReplaceResult {
            original_order_id: order_id.to_string(),
            new_order_id: None,
            action: ReplaceAction::Cancelled {
                reason: reason.to_string(),
            },
            preserved_fill,
        }
    }

    /// Get the number of managed orders
    pub async fn managed_count(&self) -> usize {
        let orders = self.managed_orders.read().await;
        orders.len()
    }

    /// Get a managed order by ID
    pub async fn get_managed_order(&self, order_id: &str) -> Option<ManagedOrder> {
        let orders = self.managed_orders.read().await;
        orders.get(order_id).cloned()
    }

    /// Check if order management is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polysniper_core::{
        FillManagementConfig, OrderStatus, OrderStatusResponse, OrderType, Side,
    };
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    // Mock executor for testing
    struct MockExecutor {
        dry_run: AtomicBool,
        cancel_count: AtomicU32,
        submit_count: AtomicU32,
        should_fail_cancel: AtomicBool,
        should_fail_submit: AtomicBool,
    }

    impl MockExecutor {
        fn new() -> Self {
            Self {
                dry_run: AtomicBool::new(true),
                cancel_count: AtomicU32::new(0),
                submit_count: AtomicU32::new(0),
                should_fail_cancel: AtomicBool::new(false),
                should_fail_submit: AtomicBool::new(false),
            }
        }
    }

    #[async_trait::async_trait]
    impl OrderExecutor for MockExecutor {
        async fn submit_order(&self, order: Order) -> Result<String, ExecutionError> {
            if self.should_fail_submit.load(Ordering::SeqCst) {
                return Err(ExecutionError::SubmissionError(
                    "Mock submit failure".to_string(),
                ));
            }
            self.submit_count.fetch_add(1, Ordering::SeqCst);
            Ok(format!("mock_{}", order.id))
        }

        async fn cancel_order(&self, _order_id: &str) -> Result<(), ExecutionError> {
            if self.should_fail_cancel.load(Ordering::SeqCst) {
                return Err(ExecutionError::Cancelled("Mock cancel failure".to_string()));
            }
            self.cancel_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn get_order_status(
            &self,
            order_id: &str,
        ) -> Result<OrderStatusResponse, ExecutionError> {
            Ok(OrderStatusResponse {
                order_id: order_id.to_string(),
                status: OrderStatus::Live,
                filled_size: Decimal::ZERO,
                remaining_size: Decimal::new(100, 0),
                avg_fill_price: None,
            })
        }

        fn is_dry_run(&self) -> bool {
            self.dry_run.load(Ordering::SeqCst)
        }
    }

    fn create_test_order(id: &str, price: Decimal, side: Side) -> Order {
        Order {
            id: id.to_string(),
            market_id: "test_market".to_string(),
            token_id: "test_token".to_string(),
            side,
            price,
            size: Decimal::new(100, 0),
            order_type: OrderType::Gtc,
            signal_id: "test_signal".to_string(),
            created_at: Utc::now(),
        }
    }

    fn create_test_config() -> OrderManagementConfig {
        OrderManagementConfig {
            enabled: true,
            check_interval_ms: 500,
            default_policy: polysniper_core::ManagementPolicyConfig {
                price_drift_threshold_bps: Decimal::new(25, 0),
                min_replace_interval_ms: 100, // Short for testing
                max_replacements: 10,
                chase_enabled: true,
                chase_aggression: 0.5,
            },
        }
    }

    fn create_fill_config() -> FillManagementConfig {
        FillManagementConfig {
            enabled: true,
            auto_resubmit: false,
            min_resubmit_size: Decimal::new(10, 0),
            poll_interval_ms: 1000,
            max_resubmit_attempts: 3,
        }
    }

    #[tokio::test]
    async fn test_manage_order() {
        let fill_manager = Arc::new(FillManager::new(create_fill_config()));
        let executor = Arc::new(MockExecutor::new());
        let manager = OrderManager::new(fill_manager, executor, create_test_config());

        let order = create_test_order("order1", Decimal::new(50, 2), Side::Buy);
        let result = manager.manage_order(order, None).await;

        assert!(result.is_ok());
        assert_eq!(manager.managed_count().await, 1);
    }

    #[tokio::test]
    async fn test_stop_managing() {
        let fill_manager = Arc::new(FillManager::new(create_fill_config()));
        let executor = Arc::new(MockExecutor::new());
        let manager = OrderManager::new(fill_manager, executor, create_test_config());

        let order = create_test_order("order1", Decimal::new(50, 2), Side::Buy);
        manager.manage_order(order, None).await.unwrap();

        assert_eq!(manager.managed_count().await, 1);

        manager.stop_managing("order1").await;

        assert_eq!(manager.managed_count().await, 0);
    }

    #[tokio::test]
    async fn test_evaluate_replace_hold_within_threshold() {
        let fill_manager = Arc::new(FillManager::new(create_fill_config()));
        let executor = Arc::new(MockExecutor::new());
        let manager = OrderManager::new(fill_manager, executor, create_test_config());

        let order = create_test_order("order1", Decimal::new(50, 2), Side::Buy);
        let managed = ManagedOrder::new(order, ManagementPolicy::default());

        // Mid price very close to order price (within 25 bps threshold)
        let current_mid = Decimal::new(501, 3); // 0.501
        let orderbook = Orderbook {
            token_id: "test_token".to_string(),
            market_id: "test_market".to_string(),
            bids: vec![],
            asks: vec![],
            timestamp: Utc::now(),
        };

        let decision = manager.evaluate_replace(&managed, current_mid, &orderbook);
        assert!(matches!(decision, ReplaceDecision::Hold));
    }

    #[tokio::test]
    async fn test_evaluate_replace_triggers_on_drift() {
        let fill_manager = Arc::new(FillManager::new(create_fill_config()));
        let executor = Arc::new(MockExecutor::new());
        let manager = OrderManager::new(fill_manager, executor, create_test_config());

        let order = create_test_order("order1", Decimal::new(50, 2), Side::Buy); // 0.50
        let managed = ManagedOrder::new(order, ManagementPolicy::default());

        // Mid price drifted significantly (more than 25 bps = 0.25%)
        // 0.50 + 0.25% = 0.50125, so 0.51 is definitely beyond threshold
        let current_mid = Decimal::new(51, 2); // 0.51 = 2% drift
        let orderbook = Orderbook {
            token_id: "test_token".to_string(),
            market_id: "test_market".to_string(),
            bids: vec![polysniper_core::PriceLevel {
                price: Decimal::new(505, 3),
                size: Decimal::new(100, 0),
            }],
            asks: vec![polysniper_core::PriceLevel {
                price: Decimal::new(515, 3),
                size: Decimal::new(100, 0),
            }],
            timestamp: Utc::now(),
        };

        let decision = manager.evaluate_replace(&managed, current_mid, &orderbook);
        assert!(matches!(decision, ReplaceDecision::Replace { .. }));
    }

    #[tokio::test]
    async fn test_max_replacements_enforced() {
        let fill_manager = Arc::new(FillManager::new(create_fill_config()));
        let executor = Arc::new(MockExecutor::new());
        let manager = OrderManager::new(fill_manager, executor, create_test_config());

        let order = create_test_order("order1", Decimal::new(50, 2), Side::Buy);
        let mut managed = ManagedOrder::new(
            order,
            ManagementPolicy {
                max_replacements: 2,
                ..ManagementPolicy::default()
            },
        );

        // Exhaust replacements
        managed.replace_count = 2;

        let current_mid = Decimal::new(55, 2); // Significant drift
        let orderbook = Orderbook {
            token_id: "test_token".to_string(),
            market_id: "test_market".to_string(),
            bids: vec![],
            asks: vec![],
            timestamp: Utc::now(),
        };

        let decision = manager.evaluate_replace(&managed, current_mid, &orderbook);
        assert!(matches!(decision, ReplaceDecision::Hold));
    }

    #[tokio::test]
    async fn test_min_interval_enforced() {
        let fill_manager = Arc::new(FillManager::new(create_fill_config()));
        let executor = Arc::new(MockExecutor::new());
        let manager = OrderManager::new(fill_manager, executor, create_test_config());

        let order = create_test_order("order1", Decimal::new(50, 2), Side::Buy);
        let mut managed = ManagedOrder::new(
            order,
            ManagementPolicy {
                min_replace_interval_ms: 10000, // 10 seconds
                ..ManagementPolicy::default()
            },
        );

        // Set last replaced to now
        managed.last_replaced = Some(Utc::now());

        let current_mid = Decimal::new(55, 2); // Significant drift
        let orderbook = Orderbook {
            token_id: "test_token".to_string(),
            market_id: "test_market".to_string(),
            bids: vec![],
            asks: vec![],
            timestamp: Utc::now(),
        };

        let decision = manager.evaluate_replace(&managed, current_mid, &orderbook);
        assert!(matches!(decision, ReplaceDecision::Hold));
    }

    #[tokio::test]
    async fn test_disabled_manager() {
        let fill_manager = Arc::new(FillManager::new(create_fill_config()));
        let executor = Arc::new(MockExecutor::new());
        let mut config = create_test_config();
        config.enabled = false;

        let manager = OrderManager::new(fill_manager, executor, config);

        let order = create_test_order("order1", Decimal::new(50, 2), Side::Buy);
        manager.manage_order(order, None).await.unwrap();

        // Should not track when disabled
        assert_eq!(manager.managed_count().await, 0);
    }

    #[tokio::test]
    async fn test_chase_price_calculation_buy() {
        let fill_manager = Arc::new(FillManager::new(create_fill_config()));
        let executor = Arc::new(MockExecutor::new());
        let manager = OrderManager::new(fill_manager, executor, create_test_config());

        let order = create_test_order("order1", Decimal::new(50, 2), Side::Buy);
        let managed = ManagedOrder::new(
            order,
            ManagementPolicy {
                chase_aggression: 0.5,
                ..ManagementPolicy::default()
            },
        );

        let current_mid = Decimal::new(52, 2); // 0.52
        let orderbook = Orderbook {
            token_id: "test_token".to_string(),
            market_id: "test_market".to_string(),
            bids: vec![polysniper_core::PriceLevel {
                price: Decimal::new(51, 2),
                size: Decimal::new(100, 0),
            }],
            asks: vec![polysniper_core::PriceLevel {
                price: Decimal::new(53, 2),
                size: Decimal::new(100, 0),
            }],
            timestamp: Utc::now(),
        };

        let new_price = manager.calculate_chase_price(&managed, current_mid, &orderbook);

        // New price should be between original and mid, influenced by aggression
        assert!(new_price > managed.original_price);
        assert!(new_price <= current_mid);
    }
}
