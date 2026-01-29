//! Fill Poller - Periodic polling service for order fill status
//!
//! This module provides a polling service that periodically checks
//! the status of active orders and updates the FillManager.

use crate::fill_manager::FillManager;
use polysniper_core::{ExecutionError, OrderExecutor, SystemEvent};
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
use tracing::{debug, error, warn};

/// Fill Poller for periodic order status checks
pub struct FillPoller {
    fill_manager: Arc<FillManager>,
    order_executor: Arc<dyn OrderExecutor>,
    poll_interval: Duration,
    cleanup_interval_polls: u32,
}

impl FillPoller {
    /// Create a new FillPoller
    pub fn new(
        fill_manager: Arc<FillManager>,
        order_executor: Arc<dyn OrderExecutor>,
        poll_interval_ms: u64,
    ) -> Self {
        Self {
            fill_manager,
            order_executor,
            poll_interval: Duration::from_millis(poll_interval_ms),
            cleanup_interval_polls: 100, // Cleanup every 100 polls
        }
    }

    /// Set the cleanup interval (number of polls between cleanups)
    pub fn with_cleanup_interval(mut self, interval: u32) -> Self {
        self.cleanup_interval_polls = interval;
        self
    }

    /// Get the poll interval
    pub fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    /// Run a single poll cycle for all active orders
    pub async fn poll_once(&self) -> Result<Vec<SystemEvent>, ExecutionError> {
        let order_ids = self.fill_manager.get_active_order_ids().await;

        if order_ids.is_empty() {
            debug!("No active orders to poll");
            return Ok(Vec::new());
        }

        debug!(count = order_ids.len(), "Polling order statuses");

        let mut all_events = Vec::new();

        for order_id in order_ids {
            match self.order_executor.get_order_status(&order_id).await {
                Ok(status) => {
                    let events = self.fill_manager.update_status(&order_id, status).await;
                    all_events.extend(events);

                    // Check if resubmit is needed
                    if let Some(new_order) = self.fill_manager.check_resubmit(&order_id).await {
                        // Track the new order
                        self.fill_manager.track_order(new_order.clone()).await;

                        // Submit the new order
                        match self.order_executor.submit_order(new_order.clone()).await {
                            Ok(submitted_id) => {
                                debug!(
                                    original_id = %order_id,
                                    new_id = %submitted_id,
                                    "Resubmitted order"
                                );
                            }
                            Err(e) => {
                                warn!(
                                    order_id = %new_order.id,
                                    error = %e,
                                    "Failed to resubmit order"
                                );
                            }
                        }
                    }
                }
                Err(ExecutionError::NotFound(_)) => {
                    debug!(order_id = %order_id, "Order not found, stopping tracking");
                    self.fill_manager.stop_tracking(&order_id).await;
                }
                Err(e) => {
                    warn!(order_id = %order_id, error = %e, "Failed to get order status");
                }
            }
        }

        Ok(all_events)
    }

    /// Spawn a background polling task
    pub fn spawn_polling_task(self: Arc<Self>) -> JoinHandle<()> {
        let this = self.clone();
        tokio::spawn(async move {
            let mut poll_count: u32 = 0;

            loop {
                // Poll for updates
                if let Err(e) = this.poll_once().await {
                    error!(error = %e, "Polling cycle failed");
                }

                poll_count = poll_count.wrapping_add(1);

                // Periodic cleanup
                if poll_count.is_multiple_of(this.cleanup_interval_polls) {
                    let cleaned = this.fill_manager.cleanup_completed().await;
                    if cleaned > 0 {
                        debug!(cleaned = cleaned, "Cleaned up completed orders");
                    }
                }

                // Wait for next poll
                tokio::time::sleep(this.poll_interval).await;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use polysniper_core::{
        FillManagementConfig, Order, OrderStatus, OrderStatusResponse, OrderType, Side,
    };
    use rust_decimal::Decimal;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::sync::Mutex;

    struct MockOrderExecutor {
        responses: Mutex<Vec<Result<OrderStatusResponse, ExecutionError>>>,
        call_count: AtomicU32,
    }

    impl MockOrderExecutor {
        fn new(responses: Vec<Result<OrderStatusResponse, ExecutionError>>) -> Self {
            Self {
                responses: Mutex::new(responses),
                call_count: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl OrderExecutor for MockOrderExecutor {
        async fn submit_order(&self, _order: Order) -> Result<String, ExecutionError> {
            Ok("mock_order_id".to_string())
        }

        async fn cancel_order(&self, _order_id: &str) -> Result<(), ExecutionError> {
            Ok(())
        }

        async fn get_order_status(
            &self,
            _order_id: &str,
        ) -> Result<OrderStatusResponse, ExecutionError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut responses = self.responses.lock().await;
            if responses.is_empty() {
                Ok(OrderStatusResponse {
                    order_id: "order1".to_string(),
                    status: OrderStatus::Live,
                    filled_size: Decimal::ZERO,
                    remaining_size: Decimal::new(100, 0),
                    avg_fill_price: None,
                })
            } else {
                responses.remove(0)
            }
        }

        fn is_dry_run(&self) -> bool {
            true
        }
    }

    fn create_test_config() -> FillManagementConfig {
        FillManagementConfig {
            enabled: true,
            auto_resubmit: false,
            min_resubmit_size: Decimal::new(10, 0),
            poll_interval_ms: 100,
            max_resubmit_attempts: 3,
        }
    }

    fn create_test_order() -> Order {
        Order {
            id: "order1".to_string(),
            market_id: "test_market".to_string(),
            token_id: "test_token".to_string(),
            side: Side::Buy,
            price: Decimal::new(50, 2),
            size: Decimal::new(100, 0),
            order_type: OrderType::Gtc,
            signal_id: "test_signal".to_string(),
            created_at: chrono::Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_poll_once_no_orders() {
        let manager = Arc::new(FillManager::new(create_test_config()));
        let executor = Arc::new(MockOrderExecutor::new(vec![]));

        let poller = FillPoller::new(manager, executor, 100);
        let events = poller.poll_once().await.unwrap();

        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn test_poll_once_with_fill() {
        let manager = Arc::new(FillManager::new(create_test_config()));
        let executor = Arc::new(MockOrderExecutor::new(vec![Ok(OrderStatusResponse {
            order_id: "order1".to_string(),
            status: OrderStatus::Live,
            filled_size: Decimal::new(50, 0),
            remaining_size: Decimal::new(50, 0),
            avg_fill_price: Some(Decimal::new(50, 2)),
        })]));

        manager.track_order(create_test_order()).await;

        let poller = FillPoller::new(manager.clone(), executor, 100);
        let events = poller.poll_once().await.unwrap();

        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], SystemEvent::PartialFill(_)));
    }

    #[tokio::test]
    async fn test_poll_once_order_not_found() {
        let manager = Arc::new(FillManager::new(create_test_config()));
        let executor = Arc::new(MockOrderExecutor::new(vec![Err(ExecutionError::NotFound(
            "order1".to_string(),
        ))]));

        manager.track_order(create_test_order()).await;

        let poller = FillPoller::new(manager.clone(), executor, 100);
        let events = poller.poll_once().await.unwrap();

        assert!(events.is_empty());
        // Order should be removed from tracking
        assert!(manager.get_tracked_order("order1").await.is_none());
    }

    #[tokio::test]
    async fn test_poll_interval() {
        let manager = Arc::new(FillManager::new(create_test_config()));
        let executor = Arc::new(MockOrderExecutor::new(vec![]));

        let poller = FillPoller::new(manager, executor, 500);
        assert_eq!(poller.poll_interval(), Duration::from_millis(500));
    }
}
