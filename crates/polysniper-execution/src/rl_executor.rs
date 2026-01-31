//! Reinforcement Learning Enhanced Executor
//!
//! Wraps base execution algorithms (TWAP/VWAP) with RL-based timing optimization.
//! The RL agent learns when to place orders based on market microstructure.

use crate::algorithms::{ChildOrder, ExecutionStats, TwapExecutor};
use chrono::{Timelike, Utc};
use polysniper_core::{Orderbook, Side, TradeSignal};
use polysniper_ml::{
    AgentStats, ExecutionAction, ExecutionReward, ExecutionState, QTableAgent, RlConfig, Urgency,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Configuration for the RL-enhanced executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlExecutorConfig {
    /// Whether RL enhancement is enabled
    pub enabled: bool,
    /// RL agent configuration
    pub rl_config: RlConfig,
    /// Decision interval in milliseconds
    pub decision_interval_ms: u64,
    /// Minimum size to place per decision
    pub min_order_size: Decimal,
    /// Maximum time without progress before forcing action (seconds)
    pub max_wait_secs: u64,
}

impl Default for RlExecutorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rl_config: RlConfig::default(),
            decision_interval_ms: 1000,
            min_order_size: dec!(10),
            max_wait_secs: 30,
        }
    }
}

/// Context for execution decisions
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Parent order/signal ID
    pub parent_id: String,
    /// Market ID
    pub market_id: String,
    /// Token ID
    pub token_id: String,
    /// Order side
    pub side: Side,
    /// Total order size
    pub total_size: Decimal,
    /// Remaining size to execute
    pub remaining_size: Decimal,
    /// Target price
    pub target_price: Option<Decimal>,
    /// Mid price from orderbook
    pub mid_price: Decimal,
    /// Best bid
    pub best_bid: Option<Decimal>,
    /// Best ask
    pub best_ask: Option<Decimal>,
    /// Current spread
    pub spread: Decimal,
    /// Orderbook imbalance (-1 to 1)
    pub orderbook_imbalance: Decimal,
    /// Time elapsed as fraction of total window
    pub time_elapsed_pct: f64,
    /// Urgency level
    pub urgency: Urgency,
    /// When execution started
    pub started_at: chrono::DateTime<Utc>,
    /// Expected completion time
    pub expected_completion: chrono::DateTime<Utc>,
}

impl ExecutionContext {
    /// Build execution state for RL agent
    pub fn to_rl_state(&self) -> ExecutionState {
        let remaining_fraction = if self.total_size.is_zero() {
            Decimal::ZERO
        } else {
            self.remaining_size / self.total_size
        };

        ExecutionState {
            remaining_size: remaining_fraction,
            time_elapsed_pct: self.time_elapsed_pct,
            urgency: self.urgency,
            bid_ask_spread: self.spread,
            orderbook_imbalance: self.orderbook_imbalance,
            recent_volatility: dec!(0.05), // TODO: Calculate from market data
            queue_depth_at_price: 0,       // TODO: Get from orderbook
            hour_of_day: Utc::now().hour() as u8,
            minute_of_hour: Utc::now().minute() as u8,
            seconds_since_last_trade: 0, // TODO: Track
            recent_fill_rate: Decimal::ONE,
            avg_slippage_last_n: Decimal::ZERO,
        }
    }
}

/// State for tracking an active RL execution
#[derive(Debug)]
struct RlExecutionState {
    /// Execution context
    context: ExecutionContext,
    /// Previous state (for learning)
    prev_state: Option<ExecutionState>,
    /// Previous action (for learning)
    prev_action: Option<ExecutionAction>,
    /// Child orders created by this execution
    child_orders: Vec<ChildOrder>,
    /// Total filled size
    filled_size: Decimal,
    /// Total cost (size * price)
    total_cost: Decimal,
    /// Number of fills
    num_fills: u32,
    /// Cumulative reward
    cumulative_reward: f64,
    /// Reference price at start
    reference_price: Decimal,
    /// Last decision time
    last_decision_at: chrono::DateTime<Utc>,
    /// Whether execution is complete
    is_complete: bool,
}

/// RL-enhanced execution algorithm.
///
/// Uses reinforcement learning to optimize the timing of order placement
/// while respecting the overall execution strategy (TWAP/VWAP).
pub struct RlEnhancedExecutor {
    /// RL agent for timing decisions
    rl_agent: Arc<QTableAgent>,
    /// Configuration
    config: RlExecutorConfig,
    /// Active executions
    active_executions: Arc<RwLock<HashMap<String, RlExecutionState>>>,
    /// Base TWAP executor (for fallback and comparison)
    #[allow(dead_code)]
    base_executor: Arc<TwapExecutor>,
    /// Recent fills for learning (parent_id -> fills)
    recent_fills: Arc<RwLock<HashMap<String, Vec<FillRecord>>>>,
}

/// Record of a fill for learning
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct FillRecord {
    child_id: String,
    filled_size: Decimal,
    fill_price: Decimal,
    mid_price_at_fill: Decimal,
    timestamp: chrono::DateTime<Utc>,
}

impl RlEnhancedExecutor {
    /// Create a new RL-enhanced executor
    pub fn new() -> Self {
        Self::with_config(RlExecutorConfig::default())
    }

    /// Create a new RL-enhanced executor with custom configuration
    pub fn with_config(config: RlExecutorConfig) -> Self {
        let rl_agent = QTableAgent::with_config(config.rl_config.clone());

        Self {
            rl_agent: Arc::new(rl_agent),
            config,
            active_executions: Arc::new(RwLock::new(HashMap::new())),
            base_executor: Arc::new(TwapExecutor::new()),
            recent_fills: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start a new RL-enhanced execution
    pub async fn start_execution(
        &self,
        signal: &TradeSignal,
        reference_price: Decimal,
        urgency: Urgency,
        duration_secs: u64,
    ) -> String {
        let now = Utc::now();

        let context = ExecutionContext {
            parent_id: signal.id.clone(),
            market_id: signal.market_id.clone(),
            token_id: signal.token_id.clone(),
            side: signal.side,
            total_size: signal.size,
            remaining_size: signal.size,
            target_price: signal.price,
            mid_price: reference_price,
            best_bid: None,
            best_ask: None,
            spread: dec!(0.01),
            orderbook_imbalance: Decimal::ZERO,
            time_elapsed_pct: 0.0,
            urgency,
            started_at: now,
            expected_completion: now + chrono::Duration::seconds(duration_secs as i64),
        };

        let state = RlExecutionState {
            context,
            prev_state: None,
            prev_action: None,
            child_orders: Vec::new(),
            filled_size: Decimal::ZERO,
            total_cost: Decimal::ZERO,
            num_fills: 0,
            cumulative_reward: 0.0,
            reference_price,
            last_decision_at: now,
            is_complete: false,
        };

        let parent_id = signal.id.clone();
        self.active_executions
            .write()
            .await
            .insert(parent_id.clone(), state);

        info!(
            parent_id = %signal.id,
            total_size = %signal.size,
            reference_price = %reference_price,
            duration_secs = duration_secs,
            "Started RL-enhanced execution"
        );

        parent_id
    }

    /// Update market data for an execution
    pub async fn update_market_data(&self, parent_id: &str, orderbook: &Orderbook) {
        let mut executions = self.active_executions.write().await;

        if let Some(state) = executions.get_mut(parent_id) {
            state.context.mid_price = orderbook.mid_price().unwrap_or(state.context.mid_price);
            state.context.best_bid = orderbook.best_bid();
            state.context.best_ask = orderbook.best_ask();
            state.context.spread = orderbook.spread().unwrap_or(dec!(0.01));
            state.context.orderbook_imbalance = calculate_imbalance(orderbook);

            // Update time elapsed
            let elapsed = Utc::now() - state.context.started_at;
            let total = state.context.expected_completion - state.context.started_at;
            state.context.time_elapsed_pct = if total.num_milliseconds() > 0 {
                elapsed.num_milliseconds() as f64 / total.num_milliseconds() as f64
            } else {
                1.0
            };
        }
    }

    /// Get the next order to place (if any) using RL decision
    pub async fn get_next_order(&self, parent_id: &str) -> Option<ChildOrder> {
        let mut executions = self.active_executions.write().await;
        let state = executions.get_mut(parent_id)?;

        if state.is_complete {
            return None;
        }

        // Check if enough time has passed since last decision
        let now = Utc::now();
        let since_last = (now - state.last_decision_at).num_milliseconds() as u64;
        if since_last < self.config.decision_interval_ms {
            return None;
        }

        // Build current state for RL
        let rl_state = state.context.to_rl_state();

        // Learn from previous action if applicable
        if let (Some(prev_state), Some(prev_action)) =
            (state.prev_state.take(), state.prev_action.take())
        {
            // Calculate reward for previous action
            let reward = self.calculate_reward(state, &prev_action);
            state.cumulative_reward += reward.total();

            // Update RL agent
            let done = state.context.remaining_size <= Decimal::ZERO;
            self.rl_agent
                .update(&prev_state, prev_action, reward.total(), &rl_state, done)
                .await;
        }

        // Get RL decision
        let action = self.rl_agent.select_action(&rl_state).await;

        debug!(
            parent_id = parent_id,
            action = ?action,
            remaining = %state.context.remaining_size,
            time_pct = state.context.time_elapsed_pct,
            "RL decision"
        );

        // Store for next learning step
        state.prev_state = Some(rl_state.clone());
        state.prev_action = Some(action);
        state.last_decision_at = now;

        // Convert action to child order
        match action {
            ExecutionAction::Wait => None,
            ExecutionAction::Cancel => {
                // For now, just wait - could implement order cancellation
                None
            }
            action if action.places_order() => {
                let size = (state.context.remaining_size * action.size_fraction())
                    .max(self.config.min_order_size)
                    .min(state.context.remaining_size);

                if size < self.config.min_order_size {
                    return None;
                }

                let child_id = format!("{}_{}", parent_id, state.child_orders.len());

                let child_order = ChildOrder {
                    id: child_id.clone(),
                    parent_id: parent_id.to_string(),
                    slice_index: state.child_orders.len() as u32,
                    market_id: state.context.market_id.clone(),
                    token_id: state.context.token_id.clone(),
                    side: state.context.side,
                    price: state.context.target_price,
                    size,
                    scheduled_at: now,
                    submitted: false,
                    submitted_at: None,
                    filled_size: Decimal::ZERO,
                    avg_fill_price: None,
                };

                state.child_orders.push(child_order.clone());
                Some(child_order)
            }
            _ => None,
        }
    }

    /// Record a fill for learning
    pub async fn record_fill(
        &self,
        parent_id: &str,
        child_id: &str,
        filled_size: Decimal,
        fill_price: Decimal,
    ) {
        // Collect data we need for the fill record before releasing the lock
        let (mid_price, learning_data) = {
            let mut executions = self.active_executions.write().await;

            if let Some(state) = executions.get_mut(parent_id) {
                // Update child order
                if let Some(child) = state.child_orders.iter_mut().find(|c| c.id == child_id) {
                    child.filled_size += filled_size;
                    if let Some(prev_avg) = child.avg_fill_price {
                        let prev_cost = prev_avg * (child.filled_size - filled_size);
                        let new_cost = prev_cost + (fill_price * filled_size);
                        child.avg_fill_price = Some(new_cost / child.filled_size);
                    } else {
                        child.avg_fill_price = Some(fill_price);
                    }
                }

                // Update execution state
                state.filled_size += filled_size;
                state.total_cost += filled_size * fill_price;
                state.num_fills += 1;
                state.context.remaining_size -= filled_size;

                let mid_price = state.context.mid_price;

                // Check completion
                let learning_data = if state.context.remaining_size <= Decimal::ZERO {
                    state.is_complete = true;

                    // Final learning update
                    if let (Some(prev_state), Some(prev_action)) =
                        (state.prev_state.take(), state.prev_action.take())
                    {
                        let final_state = state.context.to_rl_state();
                        let reward = self.calculate_reward(state, &prev_action);

                        // Add completion bonus
                        let vwap = if state.filled_size > Decimal::ZERO {
                            state.total_cost / state.filled_size
                        } else {
                            state.reference_price
                        };
                        let slippage = ((vwap - state.reference_price) / state.reference_price)
                            .abs()
                            .to_string()
                            .parse::<f64>()
                            .unwrap_or(0.0);

                        let completion_reward =
                            ExecutionReward::completion_bonus(slippage, state.context.time_elapsed_pct);

                        let total_reward = reward.total() + completion_reward.total();
                        state.cumulative_reward += total_reward;

                        info!(
                            parent_id = parent_id,
                            filled_size = %state.filled_size,
                            num_fills = state.num_fills,
                            cumulative_reward = state.cumulative_reward,
                            "RL execution complete"
                        );

                        Some((prev_state, prev_action, total_reward, final_state))
                    } else {
                        None
                    }
                } else {
                    None
                };

                (Some(mid_price), learning_data)
            } else {
                (None, None)
            }
        };

        // Perform RL update outside the lock
        if let Some((prev_state, prev_action, total_reward, final_state)) = learning_data {
            self.rl_agent
                .update(&prev_state, prev_action, total_reward, &final_state, true)
                .await;
        }

        // Store fill record
        if let Some(mid_price) = mid_price {
            let mut fills = self.recent_fills.write().await;
            fills.entry(parent_id.to_string()).or_default().push(FillRecord {
                child_id: child_id.to_string(),
                filled_size,
                fill_price,
                mid_price_at_fill: mid_price,
                timestamp: Utc::now(),
            });
        }
    }

    /// Calculate reward for an action
    fn calculate_reward(&self, state: &RlExecutionState, action: &ExecutionAction) -> ExecutionReward {
        if !action.places_order() {
            // Waiting penalty
            return ExecutionReward::wait_penalty(state.context.time_elapsed_pct);
        }

        // For order placement, calculate based on recent fill if available
        if state.num_fills > 0 && state.filled_size > Decimal::ZERO {
            let vwap = state.total_cost / state.filled_size;
            ExecutionReward::from_fill(
                state.filled_size,
                state.context.total_size,
                vwap,
                state.reference_price,
                state.context.time_elapsed_pct,
                None,
                &self.config.rl_config.reward,
            )
        } else {
            ExecutionReward::zero()
        }
    }

    /// Check if an execution is complete
    pub async fn is_complete(&self, parent_id: &str) -> bool {
        let executions = self.active_executions.read().await;
        executions
            .get(parent_id)
            .map(|s| s.is_complete)
            .unwrap_or(true)
    }

    /// Get execution statistics
    pub async fn get_stats(&self, parent_id: &str) -> Option<ExecutionStats> {
        let executions = self.active_executions.read().await;
        let state = executions.get(parent_id)?;

        let vwap = if state.filled_size > Decimal::ZERO {
            state.total_cost / state.filled_size
        } else {
            state.reference_price
        };

        let slippage_pct = if state.reference_price > Decimal::ZERO {
            match state.context.side {
                Side::Buy => ((vwap - state.reference_price) / state.reference_price) * dec!(100),
                Side::Sell => ((state.reference_price - vwap) / state.reference_price) * dec!(100),
            }
        } else {
            Decimal::ZERO
        };

        Some(ExecutionStats {
            parent_id: parent_id.to_string(),
            algorithm: "RL-Enhanced".to_string(),
            total_size: state.context.total_size,
            executed_size: state.filled_size,
            remaining_size: state.context.remaining_size,
            num_fills: state.num_fills,
            total_slices: state.child_orders.len() as u32,
            completed_slices: state
                .child_orders
                .iter()
                .filter(|c| c.is_filled())
                .count() as u32,
            avg_fill_price: vwap,
            vwap,
            reference_price: state.reference_price,
            slippage_pct,
            started_at: state.context.started_at,
            estimated_completion: state.context.expected_completion,
            completed_at: if state.is_complete {
                Some(Utc::now())
            } else {
                None
            },
            shortfall_bps: None,
            shortfall_timing_delay_bps: None,
            shortfall_market_impact_bps: None,
            shortfall_spread_cost_bps: None,
            shortfall_opportunity_cost_bps: None,
            speed_adjustment: None,
        })
    }

    /// Cancel an execution
    pub async fn cancel(&self, parent_id: &str) {
        let mut executions = self.active_executions.write().await;
        if let Some(state) = executions.get_mut(parent_id) {
            state.is_complete = true;
            info!(parent_id = parent_id, "Cancelled RL execution");
        }
    }

    /// Get the RL agent for inspection/persistence
    pub fn agent(&self) -> &Arc<QTableAgent> {
        &self.rl_agent
    }

    /// Enable training mode
    pub async fn enable_training_mode(&self) {
        self.rl_agent.enable_training_mode().await;
    }

    /// Enable production mode
    pub async fn enable_production_mode(&self) {
        self.rl_agent.enable_production_mode().await;
    }

    /// Export agent model to JSON
    pub async fn export_model(&self) -> serde_json::Result<String> {
        self.rl_agent.export_to_json().await
    }

    /// Import agent model from JSON
    pub async fn import_model(&self, json: &str) -> serde_json::Result<()> {
        self.rl_agent.import_from_json(json).await
    }

    /// Get agent statistics
    pub async fn get_agent_stats(&self) -> AgentStats {
        self.rl_agent.get_stats().await
    }

    /// Cleanup completed executions
    pub async fn cleanup_completed(&self) {
        let mut executions = self.active_executions.write().await;
        executions.retain(|_, state| !state.is_complete);
    }

    /// Get all active execution IDs
    pub async fn active_executions(&self) -> Vec<String> {
        self.active_executions
            .read()
            .await
            .keys()
            .cloned()
            .collect()
    }
}

impl Default for RlEnhancedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate orderbook imbalance (-1 to 1)
fn calculate_imbalance(orderbook: &Orderbook) -> Decimal {
    let bid_volume: Decimal = orderbook.bids.iter().take(5).map(|l| l.size).sum();
    let ask_volume: Decimal = orderbook.asks.iter().take(5).map(|l| l.size).sum();

    let total = bid_volume + ask_volume;
    if total.is_zero() {
        Decimal::ZERO
    } else {
        (bid_volume - ask_volume) / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polysniper_core::{OrderType, Outcome, Priority};

    fn create_test_signal(size: Decimal) -> TradeSignal {
        TradeSignal {
            id: "test_signal_1".to_string(),
            strategy_id: "test_strategy".to_string(),
            market_id: "market_1".to_string(),
            token_id: "token_1".to_string(),
            outcome: Outcome::Yes,
            side: Side::Buy,
            price: Some(dec!(0.50)),
            size,
            size_usd: size * dec!(0.50),
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: Utc::now(),
            reason: "Test signal".to_string(),
            metadata: serde_json::Value::Null,
        }
    }

    fn create_test_orderbook() -> Orderbook {
        Orderbook {
            token_id: "token_1".to_string(),
            market_id: "market_1".to_string(),
            bids: vec![
                polysniper_core::PriceLevel {
                    price: dec!(0.49),
                    size: dec!(100),
                },
                polysniper_core::PriceLevel {
                    price: dec!(0.48),
                    size: dec!(200),
                },
            ],
            asks: vec![
                polysniper_core::PriceLevel {
                    price: dec!(0.51),
                    size: dec!(100),
                },
                polysniper_core::PriceLevel {
                    price: dec!(0.52),
                    size: dec!(150),
                },
            ],
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_executor_creation() {
        let executor = RlEnhancedExecutor::new();
        assert!(executor.active_executions().await.is_empty());
    }

    #[tokio::test]
    async fn test_start_execution() {
        let executor = RlEnhancedExecutor::new();
        let signal = create_test_signal(dec!(1000));

        let parent_id = executor
            .start_execution(&signal, dec!(0.50), Urgency::Normal, 300)
            .await;

        assert_eq!(parent_id, "test_signal_1");
        assert!(!executor.is_complete(&parent_id).await);

        let active = executor.active_executions().await;
        assert_eq!(active.len(), 1);
        assert!(active.contains(&parent_id));
    }

    #[tokio::test]
    async fn test_update_market_data() {
        let executor = RlEnhancedExecutor::new();
        let signal = create_test_signal(dec!(1000));

        let parent_id = executor
            .start_execution(&signal, dec!(0.50), Urgency::Normal, 300)
            .await;

        let orderbook = create_test_orderbook();
        executor.update_market_data(&parent_id, &orderbook).await;

        let stats = executor.get_stats(&parent_id).await.unwrap();
        assert_eq!(stats.total_size, dec!(1000));
    }

    #[tokio::test]
    async fn test_get_next_order() {
        let config = RlExecutorConfig {
            decision_interval_ms: 0, // No delay for testing
            min_order_size: dec!(1),
            ..Default::default()
        };

        let executor = RlEnhancedExecutor::with_config(config);
        let signal = create_test_signal(dec!(100));

        let parent_id = executor
            .start_execution(&signal, dec!(0.50), Urgency::High, 60)
            .await;

        // Update with market data
        let orderbook = create_test_orderbook();
        executor.update_market_data(&parent_id, &orderbook).await;

        // Should get an order (RL agent will make a decision)
        // Note: Due to exploration, we might get Wait too
        let _order = executor.get_next_order(&parent_id).await;
        // Order may or may not be placed depending on RL decision
    }

    #[tokio::test]
    async fn test_record_fill() {
        let config = RlExecutorConfig {
            decision_interval_ms: 0,
            min_order_size: dec!(1),
            ..Default::default()
        };

        let executor = RlEnhancedExecutor::with_config(config);
        let signal = create_test_signal(dec!(100));

        let parent_id = executor
            .start_execution(&signal, dec!(0.50), Urgency::Normal, 300)
            .await;

        // Simulate getting an order and filling it
        if let Some(order) = executor.get_next_order(&parent_id).await {
            executor
                .record_fill(&parent_id, &order.id, order.size, dec!(0.51))
                .await;

            let stats = executor.get_stats(&parent_id).await.unwrap();
            assert!(stats.executed_size > Decimal::ZERO);
        }
    }

    #[tokio::test]
    async fn test_cancel_execution() {
        let executor = RlEnhancedExecutor::new();
        let signal = create_test_signal(dec!(1000));

        let parent_id = executor
            .start_execution(&signal, dec!(0.50), Urgency::Normal, 300)
            .await;

        assert!(!executor.is_complete(&parent_id).await);

        executor.cancel(&parent_id).await;

        assert!(executor.is_complete(&parent_id).await);
    }

    #[tokio::test]
    async fn test_mode_switching() {
        let executor = RlEnhancedExecutor::new();

        executor.enable_training_mode().await;
        let training_epsilon = executor.rl_agent.get_epsilon().await;
        assert_eq!(training_epsilon, 0.3);

        executor.enable_production_mode().await;
        let prod_epsilon = executor.rl_agent.get_epsilon().await;
        assert_eq!(prod_epsilon, 0.05);
    }

    #[tokio::test]
    async fn test_export_import_model() {
        let executor = RlEnhancedExecutor::new();

        // Export
        let json = executor.export_model().await.unwrap();
        assert!(!json.is_empty());

        // Import
        let executor2 = RlEnhancedExecutor::new();
        executor2.import_model(&json).await.unwrap();
    }

    #[test]
    fn test_calculate_imbalance() {
        let orderbook = create_test_orderbook();
        let imbalance = calculate_imbalance(&orderbook);

        // Bids: 100 + 200 = 300, Asks: 100 + 150 = 250
        // Imbalance = (300 - 250) / 550 = 50/550 ≈ 0.09
        assert!(imbalance > Decimal::ZERO);
        assert!(imbalance < dec!(0.2));
    }

    #[test]
    fn test_execution_context_to_rl_state() {
        let context = ExecutionContext {
            parent_id: "test".to_string(),
            market_id: "market_1".to_string(),
            token_id: "token_1".to_string(),
            side: Side::Buy,
            total_size: dec!(100),
            remaining_size: dec!(50),
            target_price: Some(dec!(0.50)),
            mid_price: dec!(0.50),
            best_bid: Some(dec!(0.49)),
            best_ask: Some(dec!(0.51)),
            spread: dec!(0.02),
            orderbook_imbalance: dec!(0.1),
            time_elapsed_pct: 0.5,
            urgency: Urgency::Normal,
            started_at: Utc::now(),
            expected_completion: Utc::now() + chrono::Duration::seconds(300),
        };

        let state = context.to_rl_state();

        assert_eq!(state.remaining_size, dec!(0.5)); // 50/100
        assert_eq!(state.time_elapsed_pct, 0.5);
        assert_eq!(state.urgency, Urgency::Normal);
        assert_eq!(state.bid_ask_spread, dec!(0.02));
        assert_eq!(state.orderbook_imbalance, dec!(0.1));
    }
}
