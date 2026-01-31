//! Time-Weighted Average Price (TWAP) Execution Algorithm
//!
//! Splits large orders evenly across time intervals to reduce market impact.
//! Supports adaptive participation rates based on real-time volume conditions.

use super::{ChildOrder, ExecutionStats};
use crate::shortfall_tracker::{ShortfallRecord, ShortfallTracker};
use crate::participation_adapter::{ParticipationAdapter, ParticipationRate};
use crate::volume_monitor::VolumeMonitor;
use chrono::{DateTime, Duration, Utc};
use polysniper_core::{Priority, Side, TradeSignal};
use rand::Rng;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// TWAP execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapConfig {
    /// Total execution window in seconds
    pub total_duration_secs: u64,
    /// Number of child order slices
    pub num_slices: u32,
    /// Add random jitter to timing (+/- percentage of interval)
    pub randomize_timing: bool,
    /// Timing jitter as percentage (0.0-1.0)
    pub timing_jitter_pct: f64,
    /// Vary child order sizes randomly
    pub randomize_size: bool,
    /// Size variation as percentage (0.0-1.0)
    pub size_jitter_pct: f64,
    /// Maximum participation rate (fraction of interval volume)
    pub max_participation_rate: Decimal,
    /// Enable adaptive participation rate based on real-time volume
    pub adaptive_participation: bool,
}

impl Default for TwapConfig {
    fn default() -> Self {
        Self {
            total_duration_secs: 300,      // 5 minutes
            num_slices: 10,                // 10 child orders
            randomize_timing: true,        // Add timing jitter
            timing_jitter_pct: 0.2,        // +/- 20% timing variation
            randomize_size: true,          // Vary sizes
            size_jitter_pct: 0.1,          // +/- 10% size variation
            max_participation_rate: dec!(0.1), // 10% max of volume
            adaptive_participation: false, // Disabled by default for backwards compatibility
        }
    }
}

/// State for a TWAP execution
#[derive(Debug, Clone)]
pub struct TwapState {
    /// Parent signal ID
    pub parent_id: String,
    /// Configuration used
    pub config: TwapConfig,
    /// Reference price at start (decision price for shortfall calculation)
    pub reference_price: Decimal,
    /// Decision time when signal was generated
    pub decision_time: DateTime<Utc>,
    /// Child orders in this execution
    pub child_orders: Vec<ChildOrder>,
    /// When execution started
    pub started_at: DateTime<Utc>,
    /// When execution should complete
    pub end_time: DateTime<Utc>,
    /// Total size to execute
    pub total_size: Decimal,
    /// Size executed so far
    pub executed_size: Decimal,
    /// Total cost (size * price) for VWAP calculation
    pub total_cost: Decimal,
    /// Number of fills received
    pub num_fills: u32,
    /// Whether execution is cancelled
    pub cancelled: bool,
    /// Order side for shortfall calculation
    pub side: Side,
    /// Arrival price (mid-price at execution start)
    pub arrival_price: Option<Decimal>,
    /// Current speed adjustment from shortfall tracking
    pub speed_adjustment: Decimal,
    /// Token ID for adaptive participation
    pub token_id: String,
    /// Priority level for adaptive participation
    pub priority: Priority,
    /// Participation rates used for each slice (for analysis)
    pub participation_rates: Vec<Decimal>,
}

/// TWAP Executor for managing TWAP executions
pub struct TwapExecutor {
    config: TwapConfig,
    active_orders: Arc<RwLock<HashMap<String, TwapState>>>,
    shortfall_tracker: Arc<ShortfallTracker>,
    /// Optional participation adapter for adaptive rates
    participation_adapter: Option<Arc<ParticipationAdapter>>,
}

impl TwapExecutor {
    /// Create a new TWAP executor with default configuration
    pub fn new() -> Self {
        Self {
            config: TwapConfig::default(),
            active_orders: Arc::new(RwLock::new(HashMap::new())),
            shortfall_tracker: Arc::new(ShortfallTracker::new()),
            participation_adapter: None,
        }
    }

    /// Create a new TWAP executor with custom configuration
    pub fn with_config(config: TwapConfig) -> Self {
        Self {
            config,
            active_orders: Arc::new(RwLock::new(HashMap::new())),
            shortfall_tracker: Arc::new(ShortfallTracker::new()),
        }
    }

    /// Create a new TWAP executor with custom configuration and shortfall tracker
    pub fn with_config_and_tracker(config: TwapConfig, shortfall_tracker: Arc<ShortfallTracker>) -> Self {
        Self {
            config,
            active_orders: Arc::new(RwLock::new(HashMap::new())),
            shortfall_tracker,
        }
    }

    /// Get the shortfall tracker
    pub fn shortfall_tracker(&self) -> &Arc<ShortfallTracker> {
        &self.shortfall_tracker
    }

    /// Create a new TWAP executor with adaptive participation support
    pub fn with_adaptive_participation(
        config: TwapConfig,
        volume_monitor: Arc<RwLock<VolumeMonitor>>,
    ) -> Self {
        use crate::participation_adapter::ParticipationConfig;
        let adapter = Arc::new(ParticipationAdapter::new(
            volume_monitor,
            ParticipationConfig::default(),
        ));
        Self {
            config,
            active_orders: Arc::new(RwLock::new(HashMap::new())),
            participation_adapter: Some(adapter),
        }
    }

    /// Set the participation adapter for adaptive rates
    pub fn set_participation_adapter(&mut self, adapter: Arc<ParticipationAdapter>) {
        self.participation_adapter = Some(adapter);
    }

    /// Create execution schedule from a trade signal
    pub fn create_schedule(&self, signal: &TradeSignal, _reference_price: Decimal) -> Vec<ChildOrder> {
        let now = Utc::now();
        let interval_secs = self.config.total_duration_secs / self.config.num_slices as u64;
        let base_size = signal.size / Decimal::from(self.config.num_slices);

        let mut child_orders = Vec::with_capacity(self.config.num_slices as usize);
        let mut rng = rand::thread_rng();

        // Track size adjustments to ensure total equals signal.size
        let mut total_allocated = Decimal::ZERO;

        for i in 0..self.config.num_slices {
            // Calculate scheduled time with optional jitter
            let base_offset_secs = interval_secs * i as u64;
            let scheduled_offset = if self.config.randomize_timing && i > 0 {
                let jitter_range = (interval_secs as f64 * self.config.timing_jitter_pct) as i64;
                let jitter: i64 = rng.gen_range(-jitter_range..=jitter_range);
                (base_offset_secs as i64 + jitter).max(0) as u64
            } else {
                base_offset_secs
            };

            let scheduled_at = now + Duration::seconds(scheduled_offset as i64);

            // Calculate size with optional variation
            let size = if i == self.config.num_slices - 1 {
                // Last slice gets remaining to ensure exact total
                signal.size - total_allocated
            } else if self.config.randomize_size {
                let jitter_factor = 1.0 + rng.gen_range(-self.config.size_jitter_pct..self.config.size_jitter_pct);
                let jittered = base_size * Decimal::from_f64_retain(jitter_factor).unwrap_or(Decimal::ONE);
                // Ensure non-negative and track
                let adjusted = jittered.max(Decimal::ZERO);
                total_allocated += adjusted;
                adjusted
            } else {
                total_allocated += base_size;
                base_size
            };

            let child_order = ChildOrder {
                id: format!("{}_twap_{}", signal.id, i),
                parent_id: signal.id.clone(),
                slice_index: i,
                market_id: signal.market_id.clone(),
                token_id: signal.token_id.clone(),
                side: signal.side,
                price: signal.price,
                size,
                scheduled_at,
                submitted: false,
                submitted_at: None,
                filled_size: Decimal::ZERO,
                avg_fill_price: None,
            };

            child_orders.push(child_order);
        }

        debug!(
            parent_id = %signal.id,
            num_slices = self.config.num_slices,
            total_duration_secs = self.config.total_duration_secs,
            total_size = %signal.size,
            "Created TWAP schedule"
        );

        child_orders
    }

    /// Start a new TWAP execution for a signal
    pub async fn start_execution(&self, signal: &TradeSignal, reference_price: Decimal) -> String {
        self.start_execution_with_arrival(signal, reference_price, None).await
    }

    /// Start a new TWAP execution for a signal with arrival price
    pub async fn start_execution_with_arrival(
        &self,
        signal: &TradeSignal,
        reference_price: Decimal,
        arrival_price: Option<Decimal>,
    ) -> String {
        let child_orders = self.create_schedule(signal, reference_price);
        let now = Utc::now();

        let state = TwapState {
            parent_id: signal.id.clone(),
            config: self.config.clone(),
            reference_price,
            decision_time: signal.timestamp,
            end_time: now + Duration::seconds(self.config.total_duration_secs as i64),
            started_at: now,
            total_size: signal.size,
            executed_size: Decimal::ZERO,
            total_cost: Decimal::ZERO,
            num_fills: 0,
            cancelled: false,
            child_orders,
            side: signal.side,
            arrival_price,
            speed_adjustment: Decimal::ONE,
            token_id: signal.token_id.clone(),
            priority: signal.priority,
            participation_rates: Vec::new(),
        };

        let parent_id = signal.id.clone();
        self.active_orders.write().await.insert(parent_id.clone(), state);

        // Start shortfall tracking
        self.shortfall_tracker
            .start_tracking(
                parent_id.clone(),
                reference_price,
                signal.timestamp,
                signal.side,
                signal.size,
                arrival_price,
            )
            .await;

        info!(
            parent_id = %signal.id,
            total_size = %signal.size,
            reference_price = %reference_price,
            decision_time = %signal.timestamp,
            num_slices = self.config.num_slices,
            "Started TWAP execution with shortfall tracking"
            adaptive = self.config.adaptive_participation,
            "Started TWAP execution"
        );

        parent_id
    }

    /// Calculate the adaptive participation rate for a given execution
    ///
    /// Returns the participation rate to use for the current slice based on
    /// real-time volume conditions, urgency, and time remaining.
    pub async fn get_adaptive_participation_rate(&self, parent_id: &str) -> Option<ParticipationRate> {
        if !self.config.adaptive_participation {
            return None;
        }

        let adapter = self.participation_adapter.as_ref()?;

        let orders = self.active_orders.read().await;
        let state = orders.get(parent_id)?;

        // Calculate remaining time percentage
        let now = Utc::now();
        let total_duration = (state.end_time - state.started_at).num_seconds() as f64;
        let remaining_duration = (state.end_time - now).num_seconds().max(0) as f64;
        let remaining_time_pct = if total_duration > 0.0 {
            Decimal::from_f64_retain(remaining_duration / total_duration).unwrap_or(Decimal::ONE)
        } else {
            Decimal::ZERO
        };

        let rate = adapter
            .calculate_rate(&state.token_id, state.priority, remaining_time_pct)
            .await;

        Some(rate)
    }

    /// Record the participation rate used for a slice
    pub async fn record_participation_rate(&self, parent_id: &str, rate: Decimal) {
        let mut orders = self.active_orders.write().await;
        if let Some(state) = orders.get_mut(parent_id) {
            state.participation_rates.push(rate);
            debug!(
                parent_id = %parent_id,
                rate = %rate,
                slice = state.participation_rates.len(),
                "Recorded participation rate for TWAP slice"
            );
        }
    }

    /// Get the average participation rate used for an execution
    pub async fn get_average_participation_rate(&self, parent_id: &str) -> Option<Decimal> {
        let orders = self.active_orders.read().await;
        let state = orders.get(parent_id)?;

        if state.participation_rates.is_empty() {
            return Some(self.config.max_participation_rate);
        }

        let sum: Decimal = state.participation_rates.iter().sum();
        Some(sum / Decimal::from(state.participation_rates.len()))
    }

    /// Get the next child order that should be executed
    pub async fn get_next_order(&self, parent_id: &str) -> Option<ChildOrder> {
        let orders = self.active_orders.read().await;
        let state = orders.get(parent_id)?;

        if state.cancelled {
            return None;
        }

        let now = Utc::now();

        // Find the next unsubmitted order whose scheduled time has passed
        state
            .child_orders
            .iter()
            .find(|o| !o.submitted && o.scheduled_at <= now)
            .cloned()
    }

    /// Get all pending child orders (scheduled but not yet submitted)
    pub async fn get_pending_orders(&self, parent_id: &str) -> Vec<ChildOrder> {
        let orders = self.active_orders.read().await;
        if let Some(state) = orders.get(parent_id) {
            state
                .child_orders
                .iter()
                .filter(|o| !o.submitted)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Mark a child order as submitted
    pub async fn mark_submitted(&self, parent_id: &str, child_id: &str) {
        let mut orders = self.active_orders.write().await;
        if let Some(state) = orders.get_mut(parent_id) {
            if let Some(child) = state.child_orders.iter_mut().find(|o| o.id == child_id) {
                child.submitted = true;
                child.submitted_at = Some(Utc::now());
                debug!(parent_id = %parent_id, child_id = %child_id, "Marked child order as submitted");
            }
        }
    }

    /// Record a fill for a child order
    pub async fn record_fill(&self, parent_id: &str, child_id: &str, filled_size: Decimal, fill_price: Decimal) {
        self.record_fill_with_market_price(parent_id, child_id, filled_size, fill_price, None).await;
    }

    /// Record a fill for a child order with current market mid-price
    pub async fn record_fill_with_market_price(
        &self,
        parent_id: &str,
        child_id: &str,
        filled_size: Decimal,
        fill_price: Decimal,
        current_mid_price: Option<Decimal>,
    ) {
        let mut orders = self.active_orders.write().await;
        if let Some(state) = orders.get_mut(parent_id) {
            // Update child order
            if let Some(child) = state.child_orders.iter_mut().find(|o| o.id == child_id) {
                child.filled_size += filled_size;
                // Update average fill price
                if let Some(prev_avg) = child.avg_fill_price {
                    let prev_cost = prev_avg * (child.filled_size - filled_size);
                    let new_cost = prev_cost + (fill_price * filled_size);
                    child.avg_fill_price = Some(new_cost / child.filled_size);
                } else {
                    child.avg_fill_price = Some(fill_price);
                }
            }

            // Update parent state
            state.executed_size += filled_size;
            state.total_cost += filled_size * fill_price;
            state.num_fills += 1;

            // Record to shortfall tracker
            drop(orders); // Release lock before async call
            self.shortfall_tracker
                .record_fill(parent_id, filled_size, fill_price, current_mid_price)
                .await;

            // Update speed adjustment from shortfall tracker
            let speed_adjustment = self.shortfall_tracker.get_speed_adjustment(parent_id).await;
            if let Some(state) = self.active_orders.write().await.get_mut(parent_id) {
                state.speed_adjustment = speed_adjustment;
            }

            info!(
                parent_id = %parent_id,
                child_id = %child_id,
                filled_size = %filled_size,
                fill_price = %fill_price,
                speed_adjustment = %speed_adjustment,
                "Recorded TWAP fill with shortfall tracking"
            );
        }
    }

    /// Check if a TWAP execution is complete
    pub async fn is_complete(&self, parent_id: &str) -> bool {
        let orders = self.active_orders.read().await;
        if let Some(state) = orders.get(parent_id) {
            state.executed_size >= state.total_size || state.cancelled
        } else {
            true // Not found means complete or never existed
        }
    }

    /// Cancel a TWAP execution
    pub async fn cancel(&self, parent_id: &str) {
        let mut orders = self.active_orders.write().await;
        if let Some(state) = orders.get_mut(parent_id) {
            state.cancelled = true;
            drop(orders);
            // Cancel shortfall tracking
            self.shortfall_tracker.cancel(parent_id).await;
            info!(parent_id = %parent_id, "Cancelled TWAP execution");
        }
    }

    /// Get the shortfall record for an execution
    pub async fn get_shortfall(&self, parent_id: &str) -> Option<ShortfallRecord> {
        self.shortfall_tracker.get_record(parent_id).await
    }

    /// Get the current speed adjustment recommendation
    pub async fn get_speed_adjustment(&self, parent_id: &str) -> Decimal {
        self.active_orders
            .read()
            .await
            .get(parent_id)
            .map(|s| s.speed_adjustment)
            .unwrap_or(Decimal::ONE)
    }

    /// Update market price for shortfall calculation
    pub async fn update_market_price(&self, parent_id: &str, current_mid_price: Decimal) {
        self.shortfall_tracker
            .update_market_price(parent_id, current_mid_price)
            .await;

        // Update speed adjustment
        let speed_adjustment = self.shortfall_tracker.get_speed_adjustment(parent_id).await;
        if let Some(state) = self.active_orders.write().await.get_mut(parent_id) {
            state.speed_adjustment = speed_adjustment;
        }
    }

    /// Apply adaptive execution speed adjustment to remaining orders
    ///
    /// This method reschedules remaining unsubmitted orders based on the current
    /// speed adjustment from shortfall tracking. A speed adjustment > 1.0 means
    /// faster execution (shorter intervals), while < 1.0 means slower execution.
    pub async fn apply_speed_adjustment(&self, parent_id: &str) {
        let mut orders = self.active_orders.write().await;
        if let Some(state) = orders.get_mut(parent_id) {
            let speed = state.speed_adjustment;
            if speed == Decimal::ONE {
                return; // No adjustment needed
            }

            let now = Utc::now();
            let remaining_orders: Vec<usize> = state
                .child_orders
                .iter()
                .enumerate()
                .filter(|(_, o)| !o.submitted && o.scheduled_at > now)
                .map(|(i, _)| i)
                .collect();

            if remaining_orders.is_empty() {
                return;
            }

            // Calculate base interval
            let base_interval_secs = self.config.total_duration_secs / self.config.num_slices as u64;
            // Adjust interval: higher speed = shorter interval
            let adjusted_interval_secs = if speed > Decimal::ZERO {
                let adjusted = Decimal::from(base_interval_secs) / speed;
                adjusted.to_string().parse::<f64>().unwrap_or(base_interval_secs as f64) as i64
            } else {
                base_interval_secs as i64
            };

            // Reschedule remaining orders
            for (offset, &idx) in remaining_orders.iter().enumerate() {
                let new_time = now + Duration::seconds(adjusted_interval_secs * (offset as i64 + 1));
                state.child_orders[idx].scheduled_at = new_time;
            }

            debug!(
                parent_id = %parent_id,
                speed_adjustment = %speed,
                remaining_orders = remaining_orders.len(),
                adjusted_interval_secs,
                "Applied adaptive execution speed adjustment"
            );
        }
    }

    /// Get next order with adaptive timing consideration
    ///
    /// This checks for speed adjustment and may reschedule remaining orders
    /// before returning the next order.
    pub async fn get_next_order_adaptive(&self, parent_id: &str) -> Option<ChildOrder> {
        // First apply any speed adjustment
        self.apply_speed_adjustment(parent_id).await;

        // Then get the next order
        self.get_next_order(parent_id).await
    }

    /// Get execution statistics for a parent order
    pub async fn get_stats(&self, parent_id: &str) -> Option<ExecutionStats> {
        let orders = self.active_orders.read().await;
        let state = orders.get(parent_id)?;

        let vwap = if state.executed_size > Decimal::ZERO {
            state.total_cost / state.executed_size
        } else {
            state.reference_price
        };

        let avg_fill_price = vwap; // For TWAP, VWAP is our average

        let slippage_pct = if state.reference_price > Decimal::ZERO {
            match state.child_orders.first().map(|o| o.side) {
                Some(Side::Buy) => ((vwap - state.reference_price) / state.reference_price) * dec!(100),
                Some(Side::Sell) => ((state.reference_price - vwap) / state.reference_price) * dec!(100),
                None => Decimal::ZERO,
            }
        } else {
            Decimal::ZERO
        };

        let completed_slices = state
            .child_orders
            .iter()
            .filter(|o| o.is_filled())
            .count() as u32;

        let speed_adjustment = state.speed_adjustment;
        drop(orders); // Release lock before async call

        // Get shortfall info
        let shortfall_record = self.shortfall_tracker.get_record(parent_id).await;
        let (shortfall_bps, timing_delay_bps, market_impact_bps, spread_cost_bps, opportunity_cost_bps) =
            if let Some(ref record) = shortfall_record {
                (
                    Some(record.shortfall_bps),
                    Some(record.components.timing_delay_bps),
                    Some(record.components.market_impact_bps),
                    Some(record.components.spread_cost_bps),
                    Some(record.components.opportunity_cost_bps),
                )
            } else {
                (None, None, None, None, None)
            };

        let orders = self.active_orders.read().await;
        let state = orders.get(parent_id)?;

        Some(ExecutionStats {
            parent_id: parent_id.to_string(),
            algorithm: "TWAP".to_string(),
            total_size: state.total_size,
            executed_size: state.executed_size,
            remaining_size: state.total_size - state.executed_size,
            num_fills: state.num_fills,
            total_slices: self.config.num_slices,
            completed_slices,
            avg_fill_price,
            vwap,
            reference_price: state.reference_price,
            slippage_pct,
            started_at: state.started_at,
            estimated_completion: state.end_time,
            completed_at: if state.executed_size >= state.total_size {
                Some(Utc::now())
            } else {
                None
            },
            shortfall_bps,
            shortfall_timing_delay_bps: timing_delay_bps,
            shortfall_market_impact_bps: market_impact_bps,
            shortfall_spread_cost_bps: spread_cost_bps,
            shortfall_opportunity_cost_bps: opportunity_cost_bps,
            speed_adjustment: Some(speed_adjustment),
        })
    }

    /// Remove completed executions from tracking
    pub async fn cleanup_completed(&self) {
        let mut orders = self.active_orders.write().await;
        orders.retain(|_, state| {
            state.executed_size < state.total_size && !state.cancelled
        });
    }

    /// Get all active execution IDs
    pub async fn active_executions(&self) -> Vec<String> {
        self.active_orders.read().await.keys().cloned().collect()
    }
}

impl Default for TwapExecutor {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn test_create_schedule_basic() {
        let config = TwapConfig {
            total_duration_secs: 100,
            num_slices: 10,
            randomize_timing: false,
            timing_jitter_pct: 0.0,
            randomize_size: false,
            size_jitter_pct: 0.0,
            max_participation_rate: dec!(0.1),
            adaptive_participation: false,
        };

        let executor = TwapExecutor::with_config(config);
        let signal = create_test_signal(dec!(1000));

        let schedule = executor.create_schedule(&signal, dec!(0.50));

        assert_eq!(schedule.len(), 10);

        // Each slice should have exactly 100 units
        for (i, order) in schedule.iter().enumerate() {
            assert_eq!(order.size, dec!(100));
            assert_eq!(order.slice_index, i as u32);
            assert_eq!(order.parent_id, "test_signal_1");
            assert!(!order.submitted);
        }

        // Total should equal signal size
        let total: Decimal = schedule.iter().map(|o| o.size).sum();
        assert_eq!(total, dec!(1000));
    }

    #[test]
    fn test_create_schedule_with_randomization() {
        let config = TwapConfig {
            total_duration_secs: 100,
            num_slices: 10,
            randomize_timing: true,
            timing_jitter_pct: 0.2,
            randomize_size: true,
            size_jitter_pct: 0.1,
            max_participation_rate: dec!(0.1),
            adaptive_participation: false,
        };

        let executor = TwapExecutor::with_config(config);
        let signal = create_test_signal(dec!(1000));

        let schedule = executor.create_schedule(&signal, dec!(0.50));

        assert_eq!(schedule.len(), 10);

        // Total should still equal signal size (last slice adjusts)
        let total: Decimal = schedule.iter().map(|o| o.size).sum();
        assert_eq!(total, dec!(1000));

        // First order should be at now (no jitter for first)
        assert!(!schedule[0].submitted);
    }

    #[tokio::test]
    async fn test_execution_lifecycle() {
        let config = TwapConfig {
            total_duration_secs: 10,
            num_slices: 5,
            randomize_timing: false,
            timing_jitter_pct: 0.0,
            randomize_size: false,
            size_jitter_pct: 0.0,
            max_participation_rate: dec!(0.1),
            adaptive_participation: false,
        };

        let executor = TwapExecutor::with_config(config);
        let signal = create_test_signal(dec!(500));

        // Start execution
        let parent_id = executor.start_execution(&signal, dec!(0.50)).await;
        assert_eq!(parent_id, "test_signal_1");

        // Should not be complete yet
        assert!(!executor.is_complete(&parent_id).await);

        // Get next order (first one should be immediately available)
        let next = executor.get_next_order(&parent_id).await;
        assert!(next.is_some());
        let first_order = next.unwrap();
        assert_eq!(first_order.slice_index, 0);
        assert_eq!(first_order.size, dec!(100));

        // Mark as submitted
        executor.mark_submitted(&parent_id, &first_order.id).await;

        // Record fill
        executor
            .record_fill(&parent_id, &first_order.id, dec!(100), dec!(0.51))
            .await;

        // Check stats
        let stats = executor.get_stats(&parent_id).await.unwrap();
        assert_eq!(stats.executed_size, dec!(100));
        assert_eq!(stats.remaining_size, dec!(400));
        assert_eq!(stats.num_fills, 1);
        assert_eq!(stats.vwap, dec!(0.51));
    }

    #[tokio::test]
    async fn test_slippage_calculation() {
        let config = TwapConfig {
            total_duration_secs: 10,
            num_slices: 2,
            randomize_timing: false,
            timing_jitter_pct: 0.0,
            randomize_size: false,
            size_jitter_pct: 0.0,
            max_participation_rate: dec!(0.1),
            adaptive_participation: false,
        };

        let executor = TwapExecutor::with_config(config);
        let signal = create_test_signal(dec!(200));

        let parent_id = executor.start_execution(&signal, dec!(0.50)).await;

        // Record fills at higher prices (for buy, this is negative slippage)
        executor
            .record_fill(&parent_id, "test_signal_1_twap_0", dec!(100), dec!(0.52))
            .await;
        executor
            .record_fill(&parent_id, "test_signal_1_twap_1", dec!(100), dec!(0.54))
            .await;

        let stats = executor.get_stats(&parent_id).await.unwrap();

        // VWAP should be (100*0.52 + 100*0.54) / 200 = 0.53
        assert_eq!(stats.vwap, dec!(0.53));

        // Slippage for buy: (0.53 - 0.50) / 0.50 * 100 = 6%
        assert_eq!(stats.slippage_pct, dec!(6));
    }

    #[tokio::test]
    async fn test_cancel_execution() {
        let executor = TwapExecutor::new();
        let signal = create_test_signal(dec!(1000));

        let parent_id = executor.start_execution(&signal, dec!(0.50)).await;

        // Cancel
        executor.cancel(&parent_id).await;

        // Should be complete (cancelled)
        assert!(executor.is_complete(&parent_id).await);

        // No more orders should be returned
        assert!(executor.get_next_order(&parent_id).await.is_none());
    }
}
