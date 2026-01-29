//! Fill Manager - Tracks order fills and manages partial fill handling
//!
//! This module provides comprehensive partial fill tracking that monitors
//! order status, detects partial fills, and manages remaining quantities
//! with optional auto-resubmission.

use chrono::{DateTime, Utc};
use polysniper_core::{
    events::{Fill, FullFillEvent, OrderExpiredEvent, PartialFillEvent, ResubmitTriggeredEvent},
    FillManagementConfig, Order, OrderStatus, OrderStatusResponse, SystemEvent,
};
use rust_decimal::Decimal;
use std::collections::HashMap;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, info, warn};

/// Status of a tracked order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackedOrderStatus {
    Active,
    PartiallyFilled,
    FullyFilled,
    Cancelled,
    Expired,
}

/// A tracked order with fill history
#[derive(Debug, Clone)]
pub struct TrackedOrder {
    pub order: Order,
    pub original_size: Decimal,
    pub filled_size: Decimal,
    pub remaining_size: Decimal,
    pub fills: Vec<Fill>,
    pub status: TrackedOrderStatus,
    pub created_at: DateTime<Utc>,
    pub last_checked: DateTime<Utc>,
    pub resubmit_count: u32,
}

impl TrackedOrder {
    fn new(order: Order) -> Self {
        let original_size = order.size;
        Self {
            order,
            original_size,
            filled_size: Decimal::ZERO,
            remaining_size: original_size,
            fills: Vec::new(),
            status: TrackedOrderStatus::Active,
            created_at: Utc::now(),
            last_checked: Utc::now(),
            resubmit_count: 0,
        }
    }

    fn is_complete(&self) -> bool {
        matches!(
            self.status,
            TrackedOrderStatus::FullyFilled
                | TrackedOrderStatus::Cancelled
                | TrackedOrderStatus::Expired
        )
    }
}

/// Fill Manager for tracking order fills
pub struct FillManager {
    active_orders: RwLock<HashMap<String, TrackedOrder>>,
    config: FillManagementConfig,
    event_sender: broadcast::Sender<SystemEvent>,
}

impl FillManager {
    /// Create a new FillManager
    pub fn new(config: FillManagementConfig) -> Self {
        let (event_sender, _) = broadcast::channel(1024);
        Self {
            active_orders: RwLock::new(HashMap::new()),
            config,
            event_sender,
        }
    }

    /// Subscribe to fill events
    pub fn subscribe(&self) -> broadcast::Receiver<SystemEvent> {
        self.event_sender.subscribe()
    }

    /// Start tracking an order
    pub async fn track_order(&self, order: Order) {
        if !self.config.enabled {
            return;
        }

        let order_id = order.id.clone();
        let tracked = TrackedOrder::new(order);

        debug!(order_id = %order_id, "Starting to track order");

        let mut orders = self.active_orders.write().await;
        orders.insert(order_id, tracked);
    }

    /// Update order status from CLOB response
    pub async fn update_status(
        &self,
        order_id: &str,
        response: OrderStatusResponse,
    ) -> Vec<SystemEvent> {
        let mut events = Vec::new();
        let mut orders = self.active_orders.write().await;

        let Some(tracked) = orders.get_mut(order_id) else {
            debug!(order_id = %order_id, "Order not tracked, ignoring status update");
            return events;
        };

        tracked.last_checked = Utc::now();

        // Check for new fills
        let prev_filled = tracked.filled_size;
        let new_filled = response.filled_size;

        if new_filled > prev_filled {
            // Calculate the new fill
            let fill_size = new_filled - prev_filled;
            let fill_price = response.avg_fill_price.unwrap_or(tracked.order.price);

            let fill = Fill {
                size: fill_size,
                price: fill_price,
                timestamp: Utc::now(),
            };

            tracked.fills.push(fill.clone());
            tracked.filled_size = new_filled;
            tracked.remaining_size = response.remaining_size;

            info!(
                order_id = %order_id,
                fill_size = %fill_size,
                total_filled = %new_filled,
                remaining = %response.remaining_size,
                "Order received fill"
            );

            // Determine if partial or full
            if response.remaining_size.is_zero() {
                tracked.status = TrackedOrderStatus::FullyFilled;

                let avg_price = Self::calculate_vwap(&tracked.fills);
                let event = SystemEvent::FullFill(FullFillEvent {
                    order_id: order_id.to_string(),
                    market_id: tracked.order.market_id.clone(),
                    token_id: tracked.order.token_id.clone(),
                    avg_price,
                    total_size: tracked.filled_size,
                    fill_count: tracked.fills.len(),
                    timestamp: Utc::now(),
                });
                events.push(event);
            } else {
                tracked.status = TrackedOrderStatus::PartiallyFilled;

                let event = SystemEvent::PartialFill(PartialFillEvent {
                    order_id: order_id.to_string(),
                    market_id: tracked.order.market_id.clone(),
                    token_id: tracked.order.token_id.clone(),
                    fill,
                    total_filled: tracked.filled_size,
                    remaining: tracked.remaining_size,
                    timestamp: Utc::now(),
                });
                events.push(event);
            }
        }

        // Handle status transitions
        match response.status {
            OrderStatus::Matched => {
                if tracked.remaining_size.is_zero() {
                    tracked.status = TrackedOrderStatus::FullyFilled;
                }
            }
            OrderStatus::Cancelled => {
                if tracked.status != TrackedOrderStatus::Cancelled {
                    tracked.status = TrackedOrderStatus::Cancelled;
                    info!(
                        order_id = %order_id,
                        filled = %tracked.filled_size,
                        unfilled = %tracked.remaining_size,
                        "Order cancelled"
                    );
                }
            }
            OrderStatus::Expired => {
                if tracked.status != TrackedOrderStatus::Expired {
                    tracked.status = TrackedOrderStatus::Expired;

                    let event = SystemEvent::OrderExpired(OrderExpiredEvent {
                        order_id: order_id.to_string(),
                        market_id: tracked.order.market_id.clone(),
                        token_id: tracked.order.token_id.clone(),
                        filled: tracked.filled_size,
                        unfilled: tracked.remaining_size,
                        timestamp: Utc::now(),
                    });
                    events.push(event);

                    info!(
                        order_id = %order_id,
                        filled = %tracked.filled_size,
                        unfilled = %tracked.remaining_size,
                        "Order expired"
                    );
                }
            }
            OrderStatus::Live => {
                // Still active, nothing to update
            }
        }

        // Publish events
        for event in &events {
            let _ = self.event_sender.send(event.clone());
        }

        events
    }

    /// Check if auto-resubmit should be triggered for an order
    pub async fn check_resubmit(&self, order_id: &str) -> Option<Order> {
        if !self.config.auto_resubmit {
            return None;
        }

        let mut orders = self.active_orders.write().await;
        let tracked = orders.get_mut(order_id)?;

        // Only resubmit for expired orders with remaining size
        if tracked.status != TrackedOrderStatus::Expired {
            return None;
        }

        // Check min size threshold
        if tracked.remaining_size < self.config.min_resubmit_size {
            debug!(
                order_id = %order_id,
                remaining = %tracked.remaining_size,
                min_size = %self.config.min_resubmit_size,
                "Remaining size below minimum, skipping resubmit"
            );
            return None;
        }

        // Check max attempts
        if tracked.resubmit_count >= self.config.max_resubmit_attempts {
            warn!(
                order_id = %order_id,
                attempts = tracked.resubmit_count,
                "Max resubmit attempts reached"
            );
            return None;
        }

        tracked.resubmit_count += 1;

        // Create new order for remaining size
        let new_order = Order {
            id: format!("{}_resubmit_{}", tracked.order.id, tracked.resubmit_count),
            market_id: tracked.order.market_id.clone(),
            token_id: tracked.order.token_id.clone(),
            side: tracked.order.side,
            price: tracked.order.price,
            size: tracked.remaining_size,
            order_type: tracked.order.order_type,
            signal_id: tracked.order.signal_id.clone(),
            created_at: Utc::now(),
        };

        let event = SystemEvent::ResubmitTriggered(ResubmitTriggeredEvent {
            original_order_id: order_id.to_string(),
            new_order: new_order.clone(),
            market_id: tracked.order.market_id.clone(),
            remaining_size: tracked.remaining_size,
            resubmit_attempt: tracked.resubmit_count,
            timestamp: Utc::now(),
        });
        let _ = self.event_sender.send(event);

        info!(
            original_order_id = %order_id,
            new_order_id = %new_order.id,
            remaining_size = %tracked.remaining_size,
            attempt = tracked.resubmit_count,
            "Triggered resubmit for expired order"
        );

        Some(new_order)
    }

    /// Get current status of a tracked order
    pub async fn get_tracked_order(&self, order_id: &str) -> Option<TrackedOrder> {
        let orders = self.active_orders.read().await;
        orders.get(order_id).cloned()
    }

    /// Stop tracking an order (e.g., when cancelled externally)
    pub async fn stop_tracking(&self, order_id: &str) {
        let mut orders = self.active_orders.write().await;
        if orders.remove(order_id).is_some() {
            debug!(order_id = %order_id, "Stopped tracking order");
        }
    }

    /// Get all active order IDs for polling
    pub async fn get_active_order_ids(&self) -> Vec<String> {
        let orders = self.active_orders.read().await;
        orders
            .iter()
            .filter(|(_, tracked)| !tracked.is_complete())
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Clean up completed orders (memory management)
    pub async fn cleanup_completed(&self) -> usize {
        let mut orders = self.active_orders.write().await;
        let before = orders.len();
        orders.retain(|_, tracked| !tracked.is_complete());
        let removed = before - orders.len();

        if removed > 0 {
            debug!(removed = removed, remaining = orders.len(), "Cleaned up completed orders");
        }

        removed
    }

    /// Get count of active orders being tracked
    pub async fn active_count(&self) -> usize {
        let orders = self.active_orders.read().await;
        orders.iter().filter(|(_, t)| !t.is_complete()).count()
    }

    /// Get total count of tracked orders (including completed)
    pub async fn total_count(&self) -> usize {
        let orders = self.active_orders.read().await;
        orders.len()
    }

    /// Calculate Volume Weighted Average Price for fills
    pub fn calculate_vwap(fills: &[Fill]) -> Decimal {
        if fills.is_empty() {
            return Decimal::ZERO;
        }

        let total_value: Decimal = fills.iter().map(|f| f.price * f.size).sum();
        let total_size: Decimal = fills.iter().map(|f| f.size).sum();

        if total_size.is_zero() {
            return Decimal::ZERO;
        }

        total_value / total_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polysniper_core::{OrderType, Side};

    fn create_test_order(id: &str, size: Decimal) -> Order {
        Order {
            id: id.to_string(),
            market_id: "test_market".to_string(),
            token_id: "test_token".to_string(),
            side: Side::Buy,
            price: Decimal::new(50, 2), // 0.50
            size,
            order_type: OrderType::Gtc,
            signal_id: "test_signal".to_string(),
            created_at: Utc::now(),
        }
    }

    fn create_test_config() -> FillManagementConfig {
        FillManagementConfig {
            enabled: true,
            auto_resubmit: false,
            min_resubmit_size: Decimal::new(10, 0),
            poll_interval_ms: 1000,
            max_resubmit_attempts: 3,
        }
    }

    #[tokio::test]
    async fn test_track_order() {
        let manager = FillManager::new(create_test_config());
        let order = create_test_order("order1", Decimal::new(100, 0));

        manager.track_order(order.clone()).await;

        let tracked = manager.get_tracked_order("order1").await.unwrap();
        assert_eq!(tracked.order.id, "order1");
        assert_eq!(tracked.original_size, Decimal::new(100, 0));
        assert_eq!(tracked.filled_size, Decimal::ZERO);
        assert_eq!(tracked.status, TrackedOrderStatus::Active);
    }

    #[tokio::test]
    async fn test_partial_fill_detection() {
        let manager = FillManager::new(create_test_config());
        let order = create_test_order("order1", Decimal::new(100, 0));

        manager.track_order(order).await;

        // Simulate partial fill
        let response = OrderStatusResponse {
            order_id: "order1".to_string(),
            status: OrderStatus::Live,
            filled_size: Decimal::new(30, 0),
            remaining_size: Decimal::new(70, 0),
            avg_fill_price: Some(Decimal::new(50, 2)),
        };

        let events = manager.update_status("order1", response).await;

        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], SystemEvent::PartialFill(_)));

        let tracked = manager.get_tracked_order("order1").await.unwrap();
        assert_eq!(tracked.filled_size, Decimal::new(30, 0));
        assert_eq!(tracked.remaining_size, Decimal::new(70, 0));
        assert_eq!(tracked.status, TrackedOrderStatus::PartiallyFilled);
        assert_eq!(tracked.fills.len(), 1);
    }

    #[tokio::test]
    async fn test_full_fill_detection() {
        let manager = FillManager::new(create_test_config());
        let order = create_test_order("order1", Decimal::new(100, 0));

        manager.track_order(order).await;

        // Simulate full fill
        let response = OrderStatusResponse {
            order_id: "order1".to_string(),
            status: OrderStatus::Matched,
            filled_size: Decimal::new(100, 0),
            remaining_size: Decimal::ZERO,
            avg_fill_price: Some(Decimal::new(50, 2)),
        };

        let events = manager.update_status("order1", response).await;

        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], SystemEvent::FullFill(_)));

        let tracked = manager.get_tracked_order("order1").await.unwrap();
        assert_eq!(tracked.status, TrackedOrderStatus::FullyFilled);
    }

    #[tokio::test]
    async fn test_multiple_partial_fills() {
        let manager = FillManager::new(create_test_config());
        let order = create_test_order("order1", Decimal::new(100, 0));

        manager.track_order(order).await;

        // First partial fill
        let response1 = OrderStatusResponse {
            order_id: "order1".to_string(),
            status: OrderStatus::Live,
            filled_size: Decimal::new(30, 0),
            remaining_size: Decimal::new(70, 0),
            avg_fill_price: Some(Decimal::new(50, 2)),
        };
        manager.update_status("order1", response1).await;

        // Second partial fill
        let response2 = OrderStatusResponse {
            order_id: "order1".to_string(),
            status: OrderStatus::Live,
            filled_size: Decimal::new(60, 0),
            remaining_size: Decimal::new(40, 0),
            avg_fill_price: Some(Decimal::new(51, 2)),
        };
        manager.update_status("order1", response2).await;

        let tracked = manager.get_tracked_order("order1").await.unwrap();
        assert_eq!(tracked.fills.len(), 2);
        assert_eq!(tracked.filled_size, Decimal::new(60, 0));
    }

    #[tokio::test]
    async fn test_order_expired() {
        let manager = FillManager::new(create_test_config());
        let order = create_test_order("order1", Decimal::new(100, 0));

        manager.track_order(order).await;

        // Partial fill then expire
        let response1 = OrderStatusResponse {
            order_id: "order1".to_string(),
            status: OrderStatus::Live,
            filled_size: Decimal::new(30, 0),
            remaining_size: Decimal::new(70, 0),
            avg_fill_price: Some(Decimal::new(50, 2)),
        };
        manager.update_status("order1", response1).await;

        let response2 = OrderStatusResponse {
            order_id: "order1".to_string(),
            status: OrderStatus::Expired,
            filled_size: Decimal::new(30, 0),
            remaining_size: Decimal::new(70, 0),
            avg_fill_price: Some(Decimal::new(50, 2)),
        };
        let events = manager.update_status("order1", response2).await;

        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], SystemEvent::OrderExpired(_)));

        let tracked = manager.get_tracked_order("order1").await.unwrap();
        assert_eq!(tracked.status, TrackedOrderStatus::Expired);
    }

    #[tokio::test]
    async fn test_vwap_calculation() {
        let fills = vec![
            Fill {
                size: Decimal::new(30, 0),
                price: Decimal::new(50, 2),
                timestamp: Utc::now(),
            },
            Fill {
                size: Decimal::new(70, 0),
                price: Decimal::new(52, 2),
                timestamp: Utc::now(),
            },
        ];

        // VWAP = (30 * 0.50 + 70 * 0.52) / 100 = (15 + 36.4) / 100 = 0.514
        let vwap = FillManager::calculate_vwap(&fills);
        assert_eq!(vwap, Decimal::new(514, 3));
    }

    #[tokio::test]
    async fn test_vwap_empty_fills() {
        let vwap = FillManager::calculate_vwap(&[]);
        assert_eq!(vwap, Decimal::ZERO);
    }

    #[tokio::test]
    async fn test_auto_resubmit() {
        let mut config = create_test_config();
        config.auto_resubmit = true;
        config.min_resubmit_size = Decimal::new(10, 0);

        let manager = FillManager::new(config);
        let order = create_test_order("order1", Decimal::new(100, 0));

        manager.track_order(order).await;

        // Partial fill then expire
        let response = OrderStatusResponse {
            order_id: "order1".to_string(),
            status: OrderStatus::Expired,
            filled_size: Decimal::new(30, 0),
            remaining_size: Decimal::new(70, 0),
            avg_fill_price: Some(Decimal::new(50, 2)),
        };
        manager.update_status("order1", response).await;

        let new_order = manager.check_resubmit("order1").await;
        assert!(new_order.is_some());
        let new_order = new_order.unwrap();
        assert_eq!(new_order.size, Decimal::new(70, 0));
        assert!(new_order.id.contains("resubmit_1"));
    }

    #[tokio::test]
    async fn test_no_resubmit_below_min_size() {
        let mut config = create_test_config();
        config.auto_resubmit = true;
        config.min_resubmit_size = Decimal::new(100, 0);

        let manager = FillManager::new(config);
        let order = create_test_order("order1", Decimal::new(100, 0));

        manager.track_order(order).await;

        let response = OrderStatusResponse {
            order_id: "order1".to_string(),
            status: OrderStatus::Expired,
            filled_size: Decimal::new(50, 0),
            remaining_size: Decimal::new(50, 0),
            avg_fill_price: Some(Decimal::new(50, 2)),
        };
        manager.update_status("order1", response).await;

        let new_order = manager.check_resubmit("order1").await;
        assert!(new_order.is_none());
    }

    #[tokio::test]
    async fn test_cleanup_completed() {
        let manager = FillManager::new(create_test_config());

        manager
            .track_order(create_test_order("order1", Decimal::new(100, 0)))
            .await;
        manager
            .track_order(create_test_order("order2", Decimal::new(100, 0)))
            .await;

        // Complete order1
        let response = OrderStatusResponse {
            order_id: "order1".to_string(),
            status: OrderStatus::Matched,
            filled_size: Decimal::new(100, 0),
            remaining_size: Decimal::ZERO,
            avg_fill_price: Some(Decimal::new(50, 2)),
        };
        manager.update_status("order1", response).await;

        assert_eq!(manager.total_count().await, 2);
        assert_eq!(manager.active_count().await, 1);

        let removed = manager.cleanup_completed().await;
        assert_eq!(removed, 1);
        assert_eq!(manager.total_count().await, 1);
    }

    #[tokio::test]
    async fn test_get_active_order_ids() {
        let manager = FillManager::new(create_test_config());

        manager
            .track_order(create_test_order("order1", Decimal::new(100, 0)))
            .await;
        manager
            .track_order(create_test_order("order2", Decimal::new(100, 0)))
            .await;

        // Complete order1
        let response = OrderStatusResponse {
            order_id: "order1".to_string(),
            status: OrderStatus::Matched,
            filled_size: Decimal::new(100, 0),
            remaining_size: Decimal::ZERO,
            avg_fill_price: Some(Decimal::new(50, 2)),
        };
        manager.update_status("order1", response).await;

        let active_ids = manager.get_active_order_ids().await;
        assert_eq!(active_ids.len(), 1);
        assert_eq!(active_ids[0], "order2");
    }

    #[tokio::test]
    async fn test_disabled_manager() {
        let mut config = create_test_config();
        config.enabled = false;

        let manager = FillManager::new(config);
        manager
            .track_order(create_test_order("order1", Decimal::new(100, 0)))
            .await;

        // Should not track when disabled
        assert!(manager.get_tracked_order("order1").await.is_none());
    }
}
