//! Gas optimization layer for intelligent order timing
//!
//! Wraps the order executor to queue orders based on gas conditions and priority,
//! implementing timing strategies for cost-optimized execution.

use anyhow::Result;
use chrono::{DateTime, Utc};
use polysniper_core::{
    ExecutionError, ExecutionStrategy, GasCondition, GasOptimizationConfig, GasOptimizationMetrics,
    GasPrice, Order, OrderExecutor, OrderStatusResponse, Priority,
};
use rust_decimal::Decimal;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

/// A queued order waiting for favorable gas conditions
#[derive(Debug, Clone)]
pub struct QueuedOrder {
    /// The order to execute
    pub order: Order,
    /// Priority level for gas threshold selection
    pub priority: Priority,
    /// When the order was queued
    pub queued_at: Instant,
    /// Timestamp for tracking
    pub queued_timestamp: DateTime<Utc>,
    /// Gas price when queued (for savings calculation)
    pub gas_at_queue: Option<Decimal>,
    /// Best gas price observed while in queue
    pub best_gas_observed: Option<Decimal>,
}

impl QueuedOrder {
    pub fn new(order: Order, priority: Priority, current_gas: Option<Decimal>) -> Self {
        Self {
            order,
            priority,
            queued_at: Instant::now(),
            queued_timestamp: Utc::now(),
            gas_at_queue: current_gas,
            best_gas_observed: current_gas,
        }
    }

    /// Update best observed gas price
    pub fn observe_gas(&mut self, gas: Decimal) {
        match self.best_gas_observed {
            Some(best) if gas < best => self.best_gas_observed = Some(gas),
            None => self.best_gas_observed = Some(gas),
            _ => {}
        }
    }

    /// Get time spent in queue
    pub fn queue_duration(&self) -> Duration {
        self.queued_at.elapsed()
    }

    /// Check if the order has exceeded its max queue time
    pub fn is_expired(&self, max_queue_time_secs: u64) -> bool {
        self.queue_duration().as_secs() >= max_queue_time_secs
    }

    /// Calculate estimated gas savings
    pub fn estimated_savings(&self, execution_gas: Decimal) -> Decimal {
        self.gas_at_queue
            .map(|queued| {
                if execution_gas < queued {
                    queued - execution_gas
                } else {
                    Decimal::ZERO
                }
            })
            .unwrap_or(Decimal::ZERO)
    }
}

/// Commands sent to the optimizer task
#[derive(Debug)]
enum OptimizerCommand {
    /// Submit an order for gas-optimized execution
    SubmitOrder {
        order: Order,
        priority: Priority,
        response: tokio::sync::oneshot::Sender<Result<String, ExecutionError>>,
    },
    /// Update the current gas price
    UpdateGasPrice(GasPrice),
    /// Update gas condition
    UpdateGasCondition(GasCondition),
    /// Shutdown the optimizer
    Shutdown,
}

/// Internal state for the optimizer
struct OptimizerState {
    /// Queue of orders waiting for favorable gas
    queue: VecDeque<QueuedOrder>,
    /// Current gas price
    current_gas: Option<GasPrice>,
    /// Current gas condition
    current_condition: Option<GasCondition>,
    /// Optimization metrics
    metrics: GasOptimizationMetrics,
}

impl OptimizerState {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            current_gas: None,
            current_condition: None,
            metrics: GasOptimizationMetrics::new(),
        }
    }
}

/// Gas optimizer that wraps an order executor
pub struct GasOptimizer {
    /// Inner order executor
    inner: Arc<dyn OrderExecutor>,
    /// Configuration
    config: GasOptimizationConfig,
    /// Command channel sender
    command_tx: mpsc::Sender<OptimizerCommand>,
    /// Shared state for metrics access
    state: Arc<RwLock<OptimizerState>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

impl GasOptimizer {
    /// Create a new gas optimizer wrapping the given executor
    pub fn new(
        inner: Arc<dyn OrderExecutor>,
        config: GasOptimizationConfig,
    ) -> (Self, GasOptimizerHandle) {
        let (command_tx, command_rx) = mpsc::channel(100);
        let state = Arc::new(RwLock::new(OptimizerState::new()));
        let shutdown = Arc::new(AtomicBool::new(false));

        let optimizer = Self {
            inner: inner.clone(),
            config: config.clone(),
            command_tx: command_tx.clone(),
            state: state.clone(),
            shutdown: shutdown.clone(),
        };

        let handle = GasOptimizerHandle {
            command_tx,
            command_rx: Some(command_rx),
            inner,
            config,
            state,
            shutdown,
        };

        (optimizer, handle)
    }

    /// Submit an order with gas optimization based on priority
    pub async fn submit_order_with_priority(
        &self,
        order: Order,
        priority: Priority,
    ) -> Result<String, ExecutionError> {
        if !self.config.enabled {
            // Optimization disabled, submit directly
            return self.inner.submit_order(order).await;
        }

        let settings = self.config.settings_for_priority(&priority);

        // Critical orders with Immediate strategy bypass the queue
        if matches!(settings.strategy, ExecutionStrategy::Immediate) {
            return self.inner.submit_order(order).await;
        }

        // For other strategies, use the async command channel
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.command_tx
            .send(OptimizerCommand::SubmitOrder {
                order,
                priority,
                response: response_tx,
            })
            .await
            .map_err(|_| ExecutionError::SubmissionError("Optimizer channel closed".to_string()))?;

        response_rx
            .await
            .map_err(|_| ExecutionError::SubmissionError("Optimizer response lost".to_string()))?
    }

    /// Update gas price from GasPriceUpdateEvent
    pub async fn update_gas_price(&self, gas_price: GasPrice) {
        let _ = self
            .command_tx
            .send(OptimizerCommand::UpdateGasPrice(gas_price))
            .await;
    }

    /// Update gas condition
    pub async fn update_gas_condition(&self, condition: GasCondition) {
        let _ = self
            .command_tx
            .send(OptimizerCommand::UpdateGasCondition(condition))
            .await;
    }

    /// Get current optimization metrics
    pub async fn metrics(&self) -> GasOptimizationMetrics {
        self.state.read().await.metrics.clone()
    }

    /// Get current queue length
    pub async fn queue_length(&self) -> usize {
        self.state.read().await.queue.len()
    }

    /// Shutdown the optimizer
    pub async fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.command_tx.send(OptimizerCommand::Shutdown).await;
    }
}

#[async_trait::async_trait]
impl OrderExecutor for GasOptimizer {
    async fn submit_order(&self, order: Order) -> Result<String, ExecutionError> {
        // Default to Normal priority when called through the trait
        self.submit_order_with_priority(order, Priority::Normal)
            .await
    }

    async fn cancel_order(&self, order_id: &str) -> Result<(), ExecutionError> {
        self.inner.cancel_order(order_id).await
    }

    async fn get_order_status(&self, order_id: &str) -> Result<OrderStatusResponse, ExecutionError> {
        self.inner.get_order_status(order_id).await
    }

    fn is_dry_run(&self) -> bool {
        self.inner.is_dry_run()
    }
}

/// Handle for running the optimizer's background processing task
pub struct GasOptimizerHandle {
    #[allow(dead_code)]
    command_tx: mpsc::Sender<OptimizerCommand>,
    command_rx: Option<mpsc::Receiver<OptimizerCommand>>,
    inner: Arc<dyn OrderExecutor>,
    config: GasOptimizationConfig,
    state: Arc<RwLock<OptimizerState>>,
    shutdown: Arc<AtomicBool>,
}

impl GasOptimizerHandle {
    /// Run the optimizer's background processing loop
    pub async fn run(mut self) -> Result<()> {
        let mut command_rx = self
            .command_rx
            .take()
            .ok_or_else(|| anyhow::anyhow!("Handle already running"))?;

        let process_interval =
            Duration::from_millis(self.config.queue_process_interval_ms);
        let mut interval = tokio::time::interval(process_interval);

        info!(
            "Gas optimizer started with {}ms process interval",
            self.config.queue_process_interval_ms
        );

        loop {
            tokio::select! {
                Some(cmd) = command_rx.recv() => {
                    match cmd {
                        OptimizerCommand::SubmitOrder { order, priority, response } => {
                            let result = self.handle_submit(order, priority).await;
                            let _ = response.send(result);
                        }
                        OptimizerCommand::UpdateGasPrice(gas_price) => {
                            self.handle_gas_update(gas_price).await;
                        }
                        OptimizerCommand::UpdateGasCondition(condition) => {
                            let mut state = self.state.write().await;
                            state.current_condition = Some(condition);
                        }
                        OptimizerCommand::Shutdown => {
                            info!("Gas optimizer shutting down, processing remaining queue");
                            self.flush_queue().await;
                            break;
                        }
                    }
                }
                _ = interval.tick() => {
                    if self.shutdown.load(Ordering::SeqCst) {
                        break;
                    }
                    self.process_queue().await;
                }
            }
        }

        info!("Gas optimizer stopped");
        Ok(())
    }

    /// Handle a new order submission
    async fn handle_submit(
        &self,
        order: Order,
        priority: Priority,
    ) -> Result<String, ExecutionError> {
        let settings = self.config.settings_for_priority(&priority);
        let state = self.state.read().await;
        let current_gas = state.current_gas.as_ref().map(|g| g.standard);
        let condition = state.current_condition;
        drop(state);

        // Check if we should execute immediately
        let should_execute = match settings.strategy {
            ExecutionStrategy::Immediate => true,
            ExecutionStrategy::WaitForLow => {
                // Execute if gas is below threshold or favorable
                current_gas
                    .map(|g| g <= settings.max_gas_gwei)
                    .unwrap_or(true)
                    || condition.map(|c| c.is_favorable()).unwrap_or(true)
            }
            ExecutionStrategy::TimeWindow => {
                // Start with queue, will execute at best time within window
                false
            }
        };

        if should_execute {
            debug!(
                order_id = %order.id,
                priority = ?priority,
                gas = ?current_gas,
                "Executing order immediately"
            );

            let mut state = self.state.write().await;
            state.metrics.record_immediate();
            drop(state);

            return self.inner.submit_order(order).await;
        }

        // Queue the order
        debug!(
            order_id = %order.id,
            priority = ?priority,
            gas = ?current_gas,
            max_gas = %settings.max_gas_gwei,
            "Queueing order for gas optimization"
        );

        let queued_order = QueuedOrder::new(order.clone(), priority, current_gas);
        let mut state = self.state.write().await;
        state.queue.push_back(queued_order);
        state.metrics.record_queued();
        drop(state);

        // Return pending status - caller can poll for completion
        Ok(format!("queued:{}", order.id))
    }

    /// Handle gas price update
    async fn handle_gas_update(&self, gas_price: GasPrice) {
        let mut state = self.state.write().await;
        state.current_gas = Some(gas_price.clone());

        // Update best observed gas for all queued orders
        for order in state.queue.iter_mut() {
            order.observe_gas(gas_price.standard);
        }
    }

    /// Process the queue, executing orders that meet criteria
    async fn process_queue(&self) {
        let state = self.state.read().await;
        let current_gas = state.current_gas.as_ref().map(|g| g.standard);
        let queue_len = state.queue.len();
        drop(state);

        if queue_len == 0 {
            return;
        }

        let mut orders_to_execute = Vec::new();
        let mut state = self.state.write().await;

        // Process queue from front (oldest first)
        let mut i = 0;
        while i < state.queue.len() {
            let order = &state.queue[i];
            let settings = self.config.settings_for_priority(&order.priority);

            let should_execute = if order.is_expired(settings.max_queue_time_secs) {
                // Force execute expired orders
                debug!(
                    order_id = %order.order.id,
                    queue_time = ?order.queue_duration(),
                    "Force executing expired order"
                );
                true
            } else {
                // Check if gas conditions are favorable
                match settings.strategy {
                    ExecutionStrategy::WaitForLow => {
                        current_gas
                            .map(|g| g <= settings.max_gas_gwei)
                            .unwrap_or(false)
                    }
                    ExecutionStrategy::TimeWindow => {
                        // For time window, execute when we've observed good enough gas
                        // or when we're approaching the deadline
                        let time_remaining = settings
                            .max_queue_time_secs
                            .saturating_sub(order.queue_duration().as_secs());
                        let approaching_deadline = time_remaining < 30;

                        current_gas
                            .map(|g| g <= settings.max_gas_gwei || approaching_deadline)
                            .unwrap_or(approaching_deadline)
                    }
                    ExecutionStrategy::Immediate => true,
                }
            };

            if should_execute {
                orders_to_execute.push(state.queue.remove(i).unwrap());
            } else {
                i += 1;
            }
        }

        drop(state);

        // Execute orders outside the lock
        for queued_order in orders_to_execute {
            let wait_secs = queued_order.queue_duration().as_secs_f64();
            let savings = current_gas
                .map(|g| queued_order.estimated_savings(g))
                .unwrap_or(Decimal::ZERO);

            match self.inner.submit_order(queued_order.order.clone()).await {
                Ok(order_id) => {
                    info!(
                        order_id = %order_id,
                        wait_secs = %wait_secs,
                        savings_gwei = %savings,
                        "Executed queued order"
                    );

                    let mut state = self.state.write().await;
                    if queued_order.is_expired(
                        self.config
                            .settings_for_priority(&queued_order.priority)
                            .max_queue_time_secs,
                    ) {
                        state.metrics.record_force_executed();
                    } else {
                        state.metrics.record_from_queue(wait_secs);
                    }
                    state.metrics.record_gas_saved(savings);
                }
                Err(e) => {
                    warn!(
                        order_id = %queued_order.order.id,
                        error = %e,
                        "Failed to execute queued order"
                    );
                }
            }
        }
    }

    /// Flush all remaining orders in queue (on shutdown)
    async fn flush_queue(&self) {
        let state = self.state.read().await;
        let remaining = state.queue.len();
        drop(state);

        if remaining > 0 {
            info!("Flushing {} remaining orders from queue", remaining);
        }

        loop {
            let mut state = self.state.write().await;
            let order = state.queue.pop_front();
            drop(state);

            match order {
                Some(queued_order) => {
                    if let Err(e) = self.inner.submit_order(queued_order.order.clone()).await {
                        warn!(
                            order_id = %queued_order.order.id,
                            error = %e,
                            "Failed to flush queued order"
                        );
                    }
                }
                None => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polysniper_core::{OrderType, Side};
    use rust_decimal_macros::dec;
    use std::sync::atomic::AtomicU64;

    /// Mock order executor for testing
    struct MockExecutor {
        submissions: Arc<AtomicU64>,
        dry_run: bool,
    }

    impl MockExecutor {
        fn new() -> Self {
            Self {
                submissions: Arc::new(AtomicU64::new(0)),
                dry_run: true,
            }
        }
    }

    #[async_trait::async_trait]
    impl OrderExecutor for MockExecutor {
        async fn submit_order(&self, order: Order) -> Result<String, ExecutionError> {
            self.submissions.fetch_add(1, Ordering::SeqCst);
            Ok(format!("mock_{}", order.id))
        }

        async fn cancel_order(&self, _order_id: &str) -> Result<(), ExecutionError> {
            Ok(())
        }

        async fn get_order_status(
            &self,
            order_id: &str,
        ) -> Result<OrderStatusResponse, ExecutionError> {
            Ok(OrderStatusResponse {
                order_id: order_id.to_string(),
                status: polysniper_core::OrderStatus::Live,
                filled_size: Decimal::ZERO,
                remaining_size: Decimal::ONE,
                avg_fill_price: None,
            })
        }

        fn is_dry_run(&self) -> bool {
            self.dry_run
        }
    }

    fn create_test_order(id: &str) -> Order {
        Order {
            id: id.to_string(),
            market_id: "test_market".to_string(),
            token_id: "test_token".to_string(),
            side: Side::Buy,
            price: dec!(0.50),
            size: dec!(10),
            order_type: OrderType::Gtc,
            signal_id: "test_signal".to_string(),
            created_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_queued_order_expiry() {
        let order = create_test_order("test1");
        let queued = QueuedOrder::new(order, Priority::Low, Some(dec!(50)));

        // Should not be expired immediately
        assert!(!queued.is_expired(60));

        // With 0 max time, should be expired
        assert!(queued.is_expired(0));
    }

    #[tokio::test]
    async fn test_queued_order_observe_gas() {
        let order = create_test_order("test1");
        let mut queued = QueuedOrder::new(order, Priority::Normal, Some(dec!(100)));

        queued.observe_gas(dec!(80));
        assert_eq!(queued.best_gas_observed, Some(dec!(80)));

        queued.observe_gas(dec!(90)); // Higher, should not update
        assert_eq!(queued.best_gas_observed, Some(dec!(80)));

        queued.observe_gas(dec!(70)); // Lower, should update
        assert_eq!(queued.best_gas_observed, Some(dec!(70)));
    }

    #[tokio::test]
    async fn test_queued_order_savings() {
        let order = create_test_order("test1");
        let queued = QueuedOrder::new(order, Priority::Normal, Some(dec!(100)));

        // Executed at lower gas = savings
        assert_eq!(queued.estimated_savings(dec!(60)), dec!(40));

        // Executed at same gas = no savings
        assert_eq!(queued.estimated_savings(dec!(100)), dec!(0));

        // Executed at higher gas = no savings (not negative)
        assert_eq!(queued.estimated_savings(dec!(120)), dec!(0));
    }

    #[tokio::test]
    async fn test_optimizer_immediate_priority() {
        let mock = Arc::new(MockExecutor::new());
        let config = GasOptimizationConfig::default();
        let (optimizer, _handle) = GasOptimizer::new(mock.clone(), config);

        let order = create_test_order("test1");

        // Critical priority should execute immediately
        let result = optimizer
            .submit_order_with_priority(order, Priority::Critical)
            .await;

        assert!(result.is_ok());
        assert_eq!(mock.submissions.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_optimizer_disabled() {
        let mock = Arc::new(MockExecutor::new());
        let mut config = GasOptimizationConfig::default();
        config.enabled = false;
        let (optimizer, _handle) = GasOptimizer::new(mock.clone(), config);

        let order = create_test_order("test1");

        // Should bypass queue when disabled
        let result = optimizer
            .submit_order_with_priority(order, Priority::Low)
            .await;

        assert!(result.is_ok());
        assert_eq!(mock.submissions.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_optimizer_metrics() {
        let mock = Arc::new(MockExecutor::new());
        let config = GasOptimizationConfig::default();
        let (optimizer, _handle) = GasOptimizer::new(mock, config);

        // Initial metrics should be zero
        let metrics = optimizer.metrics().await;
        assert_eq!(metrics.orders_processed, 0);
        assert_eq!(metrics.orders_immediate, 0);
        assert_eq!(metrics.orders_queued, 0);
    }
}
