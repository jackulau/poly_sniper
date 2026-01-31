//! Volume-Weighted Average Price (VWAP) Execution Algorithm
//!
//! Splits orders proportionally to historical or expected volume patterns.
//! Supports adaptive participation rates based on real-time volume conditions.

use super::{ChildOrder, ExecutionStats};
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
use tracing::{debug, info, warn};

/// Volume profile for VWAP execution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum VolumeProfile {
    /// Uniform distribution (same as TWAP)
    #[default]
    Uniform,
    /// Front-loaded: Execute more at the beginning
    FrontLoaded {
        /// Decay factor (higher = more front-loaded)
        decay: f64,
    },
    /// Back-loaded: Execute more at the end
    BackLoaded {
        /// Growth factor (higher = more back-loaded)
        growth: f64,
    },
    /// U-shaped: More at beginning and end (like market open/close)
    UShaped {
        /// Depth of the U (0.0-1.0, higher = deeper U)
        depth: f64,
    },
    /// Custom percentage distribution per interval
    Custom {
        /// Percentages for each interval (should sum to ~1.0)
        weights: Vec<Decimal>,
    },
}


impl VolumeProfile {
    /// Calculate weights for each slice based on the profile
    pub fn calculate_weights(&self, num_slices: u32) -> Vec<Decimal> {
        match self {
            VolumeProfile::Uniform => {
                let weight = Decimal::ONE / Decimal::from(num_slices);
                vec![weight; num_slices as usize]
            }
            VolumeProfile::FrontLoaded { decay } => {
                let mut weights = Vec::with_capacity(num_slices as usize);
                let mut total = 0.0_f64;

                for i in 0..num_slices {
                    let w = (-(*decay) * i as f64).exp();
                    weights.push(w);
                    total += w;
                }

                // Normalize
                weights
                    .iter()
                    .map(|w| Decimal::from_f64_retain(w / total).unwrap_or(Decimal::ZERO))
                    .collect()
            }
            VolumeProfile::BackLoaded { growth } => {
                let mut weights = Vec::with_capacity(num_slices as usize);
                let mut total = 0.0_f64;

                for i in 0..num_slices {
                    let w = ((*growth) * i as f64 / num_slices as f64).exp();
                    weights.push(w);
                    total += w;
                }

                // Normalize
                weights
                    .iter()
                    .map(|w| Decimal::from_f64_retain(w / total).unwrap_or(Decimal::ZERO))
                    .collect()
            }
            VolumeProfile::UShaped { depth } => {
                let mut weights = Vec::with_capacity(num_slices as usize);
                let mut total = 0.0_f64;
                let mid = (num_slices - 1) as f64 / 2.0;

                for i in 0..num_slices {
                    let distance_from_mid = ((i as f64 - mid) / mid).abs();
                    let w = 1.0 - depth * (1.0 - distance_from_mid * distance_from_mid);
                    weights.push(w.max(0.1)); // Ensure minimum weight
                    total += weights.last().unwrap();
                }

                // Normalize
                weights
                    .iter()
                    .map(|w| Decimal::from_f64_retain(w / total).unwrap_or(Decimal::ZERO))
                    .collect()
            }
            VolumeProfile::Custom { weights } => {
                if weights.len() != num_slices as usize {
                    warn!(
                        expected = num_slices,
                        actual = weights.len(),
                        "Custom weights count mismatch, using uniform"
                    );
                    let weight = Decimal::ONE / Decimal::from(num_slices);
                    return vec![weight; num_slices as usize];
                }

                // Normalize to ensure they sum to 1
                let total: Decimal = weights.iter().sum();
                if total.is_zero() {
                    let weight = Decimal::ONE / Decimal::from(num_slices);
                    return vec![weight; num_slices as usize];
                }

                weights.iter().map(|w| *w / total).collect()
            }
        }
    }
}

/// VWAP execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VwapConfig {
    /// Total execution window in seconds
    pub total_duration_secs: u64,
    /// Number of child order slices
    pub num_slices: u32,
    /// Volume profile to use
    pub volume_profile: VolumeProfile,
    /// Maximum participation rate
    pub max_participation_rate: Decimal,
    /// Adapt execution based on real-time volume
    pub adaptive: bool,
    /// Add random jitter to timing
    pub randomize_timing: bool,
    /// Timing jitter as percentage (0.0-1.0)
    pub timing_jitter_pct: f64,
    /// Enable adaptive participation rate based on real-time volume
    pub adaptive_participation: bool,
}

impl Default for VwapConfig {
    fn default() -> Self {
        Self {
            total_duration_secs: 300,          // 5 minutes
            num_slices: 10,                    // 10 child orders
            volume_profile: VolumeProfile::Uniform,
            max_participation_rate: dec!(0.1), // 10% max of volume
            adaptive: false,                   // Start with non-adaptive
            randomize_timing: true,
            timing_jitter_pct: 0.15,
            adaptive_participation: false,     // Disabled by default for backwards compatibility
        }
    }
}

/// State for a VWAP execution
#[derive(Debug, Clone)]
pub struct VwapState {
    /// Parent signal ID
    pub parent_id: String,
    /// Configuration used
    pub config: VwapConfig,
    /// Reference price at start
    pub reference_price: Decimal,
    /// Volume weights for each slice
    pub weights: Vec<Decimal>,
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
    /// Real-time volume observed (for adaptive mode)
    pub observed_volume: Vec<Decimal>,
    /// Token ID for adaptive participation
    pub token_id: String,
    /// Priority level for adaptive participation
    pub priority: Priority,
    /// Participation rates used for each slice (for analysis)
    pub participation_rates: Vec<Decimal>,
}

/// VWAP Executor for managing VWAP executions
pub struct VwapExecutor {
    config: VwapConfig,
    active_orders: Arc<RwLock<HashMap<String, VwapState>>>,
    /// Optional participation adapter for adaptive rates
    participation_adapter: Option<Arc<ParticipationAdapter>>,
}

impl VwapExecutor {
    /// Create a new VWAP executor with default configuration
    pub fn new() -> Self {
        Self {
            config: VwapConfig::default(),
            active_orders: Arc::new(RwLock::new(HashMap::new())),
            participation_adapter: None,
        }
    }

    /// Create a new VWAP executor with custom configuration
    pub fn with_config(config: VwapConfig) -> Self {
        Self {
            config,
            active_orders: Arc::new(RwLock::new(HashMap::new())),
            participation_adapter: None,
        }
    }

    /// Create a new VWAP executor with adaptive participation support
    pub fn with_adaptive_participation(
        config: VwapConfig,
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
        let weights = self.config.volume_profile.calculate_weights(self.config.num_slices);

        let mut child_orders = Vec::with_capacity(self.config.num_slices as usize);
        let mut rng = rand::thread_rng();
        let mut total_allocated = Decimal::ZERO;

        for (i, weight) in weights.iter().enumerate() {
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

            // Calculate size based on weight
            let size = if i == self.config.num_slices as usize - 1 {
                // Last slice gets remaining to ensure exact total
                signal.size - total_allocated
            } else {
                let weighted_size = signal.size * weight;
                total_allocated += weighted_size;
                weighted_size
            };

            let child_order = ChildOrder {
                id: format!("{}_vwap_{}", signal.id, i),
                parent_id: signal.id.clone(),
                slice_index: i as u32,
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
            profile = ?self.config.volume_profile,
            "Created VWAP schedule"
        );

        child_orders
    }

    /// Start a new VWAP execution for a signal
    pub async fn start_execution(&self, signal: &TradeSignal, reference_price: Decimal) -> String {
        let weights = self.config.volume_profile.calculate_weights(self.config.num_slices);
        let child_orders = self.create_schedule(signal, reference_price);
        let now = Utc::now();

        let state = VwapState {
            parent_id: signal.id.clone(),
            config: self.config.clone(),
            reference_price,
            weights,
            end_time: now + Duration::seconds(self.config.total_duration_secs as i64),
            started_at: now,
            total_size: signal.size,
            executed_size: Decimal::ZERO,
            total_cost: Decimal::ZERO,
            num_fills: 0,
            cancelled: false,
            child_orders,
            observed_volume: vec![Decimal::ZERO; self.config.num_slices as usize],
            token_id: signal.token_id.clone(),
            priority: signal.priority,
            participation_rates: Vec::new(),
        };

        let parent_id = signal.id.clone();
        self.active_orders.write().await.insert(parent_id.clone(), state);

        info!(
            parent_id = %signal.id,
            total_size = %signal.size,
            reference_price = %reference_price,
            num_slices = self.config.num_slices,
            profile = ?self.config.volume_profile,
            adaptive_participation = self.config.adaptive_participation,
            "Started VWAP execution"
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
                "Recorded participation rate for VWAP slice"
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

    /// Get all pending child orders
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
                debug!(parent_id = %parent_id, child_id = %child_id, "Marked VWAP child order as submitted");
            }
        }
    }

    /// Record a fill for a child order
    pub async fn record_fill(&self, parent_id: &str, child_id: &str, filled_size: Decimal, fill_price: Decimal) {
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

            info!(
                parent_id = %parent_id,
                child_id = %child_id,
                filled_size = %filled_size,
                fill_price = %fill_price,
                total_executed = %state.executed_size,
                remaining = %(state.total_size - state.executed_size),
                "Recorded VWAP fill"
            );
        }
    }

    /// Update observed volume for adaptive execution
    pub async fn update_observed_volume(&self, parent_id: &str, slice_index: usize, volume: Decimal) {
        let mut orders = self.active_orders.write().await;
        if let Some(state) = orders.get_mut(parent_id) {
            if slice_index < state.observed_volume.len() {
                state.observed_volume[slice_index] = volume;

                // If adaptive mode is enabled, potentially adjust remaining orders
                if state.config.adaptive {
                    self.adapt_remaining_orders(state, slice_index);
                }
            }
        }
    }

    /// Adapt remaining orders based on observed volume (internal)
    fn adapt_remaining_orders(&self, state: &mut VwapState, current_slice: usize) {
        // Simple adaptation: if volume is higher than expected, increase remaining sizes
        // This is a basic implementation - production would be more sophisticated
        let remaining_slices = state.child_orders.len() - current_slice - 1;
        if remaining_slices == 0 {
            return;
        }

        let expected_weight = state.weights.get(current_slice).copied().unwrap_or(Decimal::ZERO);
        let observed_volume = state.observed_volume.get(current_slice).copied().unwrap_or(Decimal::ZERO);

        // Calculate total expected volume based on this slice's observation
        let total_expected = if expected_weight > Decimal::ZERO {
            observed_volume / expected_weight
        } else {
            return;
        };

        // Remaining size to distribute
        let remaining_size = state.total_size - state.executed_size;

        // Redistribute based on remaining weights
        let remaining_weight: Decimal = state.weights[current_slice + 1..].iter().sum();
        if remaining_weight.is_zero() {
            return;
        }

        for (i, child) in state.child_orders.iter_mut().enumerate() {
            if i <= current_slice || child.submitted {
                continue;
            }

            let slice_weight = state.weights.get(i).copied().unwrap_or(Decimal::ZERO);
            let new_size = remaining_size * (slice_weight / remaining_weight);
            child.size = new_size;
        }

        debug!(
            parent_id = %state.parent_id,
            current_slice,
            observed_volume = %observed_volume,
            total_expected = %total_expected,
            "Adapted VWAP order sizes based on observed volume"
        );
    }

    /// Check if a VWAP execution is complete
    pub async fn is_complete(&self, parent_id: &str) -> bool {
        let orders = self.active_orders.read().await;
        if let Some(state) = orders.get(parent_id) {
            state.executed_size >= state.total_size || state.cancelled
        } else {
            true
        }
    }

    /// Cancel a VWAP execution
    pub async fn cancel(&self, parent_id: &str) {
        let mut orders = self.active_orders.write().await;
        if let Some(state) = orders.get_mut(parent_id) {
            state.cancelled = true;
            info!(parent_id = %parent_id, "Cancelled VWAP execution");
        }
    }

    /// Get execution statistics
    pub async fn get_stats(&self, parent_id: &str) -> Option<ExecutionStats> {
        let orders = self.active_orders.read().await;
        let state = orders.get(parent_id)?;

        let vwap = if state.executed_size > Decimal::ZERO {
            state.total_cost / state.executed_size
        } else {
            state.reference_price
        };

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

        Some(ExecutionStats {
            parent_id: parent_id.to_string(),
            algorithm: "VWAP".to_string(),
            total_size: state.total_size,
            executed_size: state.executed_size,
            remaining_size: state.total_size - state.executed_size,
            num_fills: state.num_fills,
            total_slices: state.config.num_slices,
            completed_slices,
            avg_fill_price: vwap,
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
        })
    }

    /// Remove completed executions
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

impl Default for VwapExecutor {
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
    fn test_uniform_weights() {
        let profile = VolumeProfile::Uniform;
        let weights = profile.calculate_weights(5);

        assert_eq!(weights.len(), 5);
        for w in &weights {
            assert_eq!(*w, dec!(0.2));
        }
    }

    #[test]
    fn test_front_loaded_weights() {
        let profile = VolumeProfile::FrontLoaded { decay: 0.5 };
        let weights = profile.calculate_weights(5);

        assert_eq!(weights.len(), 5);

        // First weight should be largest
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
        assert!(weights[2] > weights[3]);
        assert!(weights[3] > weights[4]);

        // Sum should be approximately 1
        let sum: Decimal = weights.iter().sum();
        assert!((sum - Decimal::ONE).abs() < dec!(0.001));
    }

    #[test]
    fn test_back_loaded_weights() {
        let profile = VolumeProfile::BackLoaded { growth: 1.0 };
        let weights = profile.calculate_weights(5);

        assert_eq!(weights.len(), 5);

        // Last weight should be largest
        assert!(weights[4] > weights[3]);
        assert!(weights[3] > weights[2]);
        assert!(weights[2] > weights[1]);
        assert!(weights[1] > weights[0]);

        // Sum should be approximately 1
        let sum: Decimal = weights.iter().sum();
        assert!((sum - Decimal::ONE).abs() < dec!(0.001));
    }

    #[test]
    fn test_u_shaped_weights() {
        let profile = VolumeProfile::UShaped { depth: 0.5 };
        let weights = profile.calculate_weights(5);

        assert_eq!(weights.len(), 5);

        // First and last should be larger than middle
        assert!(weights[0] > weights[2]);
        assert!(weights[4] > weights[2]);

        // Sum should be approximately 1
        let sum: Decimal = weights.iter().sum();
        assert!((sum - Decimal::ONE).abs() < dec!(0.001));
    }

    #[test]
    fn test_custom_weights() {
        let profile = VolumeProfile::Custom {
            weights: vec![dec!(0.1), dec!(0.2), dec!(0.3), dec!(0.2), dec!(0.2)],
        };
        let weights = profile.calculate_weights(5);

        assert_eq!(weights.len(), 5);
        assert_eq!(weights[0], dec!(0.1));
        assert_eq!(weights[1], dec!(0.2));
        assert_eq!(weights[2], dec!(0.3));

        // Sum should equal 1
        let sum: Decimal = weights.iter().sum();
        assert_eq!(sum, Decimal::ONE);
    }

    #[test]
    fn test_create_schedule_with_weights() {
        let config = VwapConfig {
            total_duration_secs: 100,
            num_slices: 4,
            volume_profile: VolumeProfile::Custom {
                weights: vec![dec!(0.4), dec!(0.3), dec!(0.2), dec!(0.1)],
            },
            max_participation_rate: dec!(0.1),
            adaptive: false,
            randomize_timing: false,
            timing_jitter_pct: 0.0,
            adaptive_participation: false,
        };

        let executor = VwapExecutor::with_config(config);
        let signal = create_test_signal(dec!(1000));

        let schedule = executor.create_schedule(&signal, dec!(0.50));

        assert_eq!(schedule.len(), 4);

        // Verify sizes match weights
        assert_eq!(schedule[0].size, dec!(400));
        assert_eq!(schedule[1].size, dec!(300));
        assert_eq!(schedule[2].size, dec!(200));
        assert_eq!(schedule[3].size, dec!(100));

        // Total should equal signal size
        let total: Decimal = schedule.iter().map(|o| o.size).sum();
        assert_eq!(total, dec!(1000));
    }

    #[tokio::test]
    async fn test_vwap_execution_lifecycle() {
        let config = VwapConfig {
            total_duration_secs: 10,
            num_slices: 4,
            volume_profile: VolumeProfile::Uniform,
            max_participation_rate: dec!(0.1),
            adaptive: false,
            randomize_timing: false,
            timing_jitter_pct: 0.0,
            adaptive_participation: false,
        };

        let executor = VwapExecutor::with_config(config);
        let signal = create_test_signal(dec!(400));

        // Start execution
        let parent_id = executor.start_execution(&signal, dec!(0.50)).await;
        assert_eq!(parent_id, "test_signal_1");

        // Should not be complete
        assert!(!executor.is_complete(&parent_id).await);

        // Get first order
        let next = executor.get_next_order(&parent_id).await;
        assert!(next.is_some());
        let first_order = next.unwrap();
        assert_eq!(first_order.size, dec!(100)); // 400 / 4

        // Submit and fill
        executor.mark_submitted(&parent_id, &first_order.id).await;
        executor
            .record_fill(&parent_id, &first_order.id, dec!(100), dec!(0.52))
            .await;

        // Check stats
        let stats = executor.get_stats(&parent_id).await.unwrap();
        assert_eq!(stats.executed_size, dec!(100));
        assert_eq!(stats.remaining_size, dec!(300));
        assert_eq!(stats.algorithm, "VWAP");
    }

    #[tokio::test]
    async fn test_vwap_slippage() {
        let config = VwapConfig {
            total_duration_secs: 10,
            num_slices: 2,
            volume_profile: VolumeProfile::Uniform,
            max_participation_rate: dec!(0.1),
            adaptive: false,
            randomize_timing: false,
            timing_jitter_pct: 0.0,
            adaptive_participation: false,
        };

        let executor = VwapExecutor::with_config(config);
        let mut signal = create_test_signal(dec!(200));
        signal.side = Side::Sell;

        let parent_id = executor.start_execution(&signal, dec!(0.50)).await;

        // Record fills at lower prices (for sell, this is negative slippage)
        executor
            .record_fill(&parent_id, "test_signal_1_vwap_0", dec!(100), dec!(0.48))
            .await;
        executor
            .record_fill(&parent_id, "test_signal_1_vwap_1", dec!(100), dec!(0.46))
            .await;

        let stats = executor.get_stats(&parent_id).await.unwrap();

        // VWAP = (100*0.48 + 100*0.46) / 200 = 0.47
        assert_eq!(stats.vwap, dec!(0.47));

        // Slippage for sell: (0.50 - 0.47) / 0.50 * 100 = 6%
        assert_eq!(stats.slippage_pct, dec!(6));
    }
}
