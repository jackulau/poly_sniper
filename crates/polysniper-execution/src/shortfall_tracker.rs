//! Implementation Shortfall Tracker
//!
//! Tracks slippage vs. decision price and provides metrics for optimizing
//! execution to reduce implementation shortfall.

use chrono::{DateTime, Duration, Utc};
use polysniper_core::Side;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Configuration for shortfall tracking and adaptive execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortfallConfig {
    /// Enable shortfall tracking
    pub enabled: bool,
    /// Shortfall threshold in basis points to trigger faster execution
    pub adverse_threshold_bps: Decimal,
    /// Shortfall threshold in basis points for favorable price movement
    pub favorable_threshold_bps: Decimal,
    /// Speed multiplier when experiencing adverse selection
    pub adverse_speed_multiplier: Decimal,
    /// Speed multiplier when price is favorable
    pub favorable_speed_multiplier: Decimal,
    /// Minimum speed multiplier (can't go slower than this)
    pub min_speed_multiplier: Decimal,
    /// Maximum speed multiplier (can't go faster than this)
    pub max_speed_multiplier: Decimal,
}

impl Default for ShortfallConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            adverse_threshold_bps: dec!(20),     // 20 bps adverse triggers faster execution
            favorable_threshold_bps: dec!(10),   // 10 bps favorable allows slower execution
            adverse_speed_multiplier: dec!(1.5), // 1.5x faster when adverse
            favorable_speed_multiplier: dec!(0.7), // 0.7x slower when favorable
            min_speed_multiplier: dec!(0.5),     // Can't go slower than 0.5x
            max_speed_multiplier: dec!(2.0),     // Can't go faster than 2x
        }
    }
}

/// Components of implementation shortfall
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShortfallComponents {
    /// Cost from delay between decision and first fill (in basis points)
    pub timing_delay_bps: Decimal,
    /// Cost from our own order pressure (in basis points)
    pub market_impact_bps: Decimal,
    /// Cost from bid-ask spread (in basis points)
    pub spread_cost_bps: Decimal,
    /// Cost from unfilled portion (in basis points)
    pub opportunity_cost_bps: Decimal,
}

impl ShortfallComponents {
    /// Get total shortfall from all components
    pub fn total_bps(&self) -> Decimal {
        self.timing_delay_bps + self.market_impact_bps + self.spread_cost_bps + self.opportunity_cost_bps
    }
}

/// Record for tracking implementation shortfall of a single execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortfallRecord {
    /// Parent order ID this record tracks
    pub parent_order_id: String,
    /// Price when trade signal was generated
    pub decision_price: Decimal,
    /// Time when trade signal was generated
    pub decision_time: DateTime<Utc>,
    /// Order side (buy/sell)
    pub side: Side,
    /// Total size to execute
    pub total_size: Decimal,
    /// Size executed so far
    pub executed_size: Decimal,
    /// Volume-weighted average price of fills
    pub vwap: Decimal,
    /// Total cost (size * price) for VWAP calculation
    pub total_cost: Decimal,
    /// Number of fills received
    pub num_fills: u32,
    /// Implementation shortfall in basis points
    pub shortfall_bps: Decimal,
    /// Breakdown of shortfall components
    pub components: ShortfallComponents,
    /// Time of first fill
    pub first_fill_time: Option<DateTime<Utc>>,
    /// Price at first fill (for timing delay calculation)
    pub first_fill_price: Option<Decimal>,
    /// Arrival price (mid-price at execution start)
    pub arrival_price: Option<Decimal>,
    /// Current market mid-price (for opportunity cost)
    pub current_market_price: Option<Decimal>,
    /// Whether execution is complete
    pub is_complete: bool,
    /// When execution completed
    pub completed_at: Option<DateTime<Utc>>,
    /// Current speed adjustment recommendation
    pub speed_adjustment: Decimal,
}

impl ShortfallRecord {
    /// Create a new shortfall record
    pub fn new(
        parent_order_id: String,
        decision_price: Decimal,
        decision_time: DateTime<Utc>,
        side: Side,
        total_size: Decimal,
        arrival_price: Option<Decimal>,
    ) -> Self {
        Self {
            parent_order_id,
            decision_price,
            decision_time,
            side,
            total_size,
            executed_size: Decimal::ZERO,
            vwap: Decimal::ZERO,
            total_cost: Decimal::ZERO,
            num_fills: 0,
            shortfall_bps: Decimal::ZERO,
            components: ShortfallComponents::default(),
            first_fill_time: None,
            first_fill_price: None,
            arrival_price,
            current_market_price: None,
            is_complete: false,
            completed_at: None,
            speed_adjustment: Decimal::ONE,
        }
    }

    /// Record a fill and update shortfall calculations
    pub fn record_fill(&mut self, fill_size: Decimal, fill_price: Decimal, current_mid_price: Option<Decimal>) {
        let is_first_fill = self.num_fills == 0;

        // Update fill tracking
        self.executed_size += fill_size;
        self.total_cost += fill_size * fill_price;
        self.num_fills += 1;

        // Update VWAP
        if self.executed_size > Decimal::ZERO {
            self.vwap = self.total_cost / self.executed_size;
        }

        // Record first fill data
        if is_first_fill {
            self.first_fill_time = Some(Utc::now());
            self.first_fill_price = Some(fill_price);
        }

        // Update current market price
        if let Some(mid) = current_mid_price {
            self.current_market_price = Some(mid);
        }

        // Recalculate shortfall
        self.calculate_shortfall();

        // Check if complete
        if self.executed_size >= self.total_size {
            self.is_complete = true;
            self.completed_at = Some(Utc::now());
        }

        debug!(
            parent_id = %self.parent_order_id,
            fill_size = %fill_size,
            fill_price = %fill_price,
            vwap = %self.vwap,
            shortfall_bps = %self.shortfall_bps,
            executed = %self.executed_size,
            total = %self.total_size,
            "Recorded fill for shortfall tracking"
        );
    }

    /// Calculate implementation shortfall in basis points
    fn calculate_shortfall(&mut self) {
        if self.decision_price.is_zero() {
            return;
        }

        // Calculate total shortfall
        // For buys: shortfall = (VWAP - decision_price) / decision_price * 10000
        // For sells: shortfall = (decision_price - VWAP) / decision_price * 10000
        self.shortfall_bps = match self.side {
            Side::Buy => {
                if self.vwap > Decimal::ZERO {
                    ((self.vwap - self.decision_price) / self.decision_price) * dec!(10000)
                } else {
                    Decimal::ZERO
                }
            }
            Side::Sell => {
                if self.vwap > Decimal::ZERO {
                    ((self.decision_price - self.vwap) / self.decision_price) * dec!(10000)
                } else {
                    Decimal::ZERO
                }
            }
        };

        // Calculate components
        self.calculate_components();
    }

    /// Calculate individual shortfall components
    fn calculate_components(&mut self) {
        if self.decision_price.is_zero() {
            return;
        }

        // 1. Timing delay: Cost from delay between decision and first fill
        // Uses arrival price vs first fill price
        if let (Some(arrival), Some(first_fill)) = (self.arrival_price, self.first_fill_price) {
            self.components.timing_delay_bps = match self.side {
                Side::Buy => ((first_fill - arrival) / self.decision_price) * dec!(10000),
                Side::Sell => ((arrival - first_fill) / self.decision_price) * dec!(10000),
            };
            // Ensure non-negative
            self.components.timing_delay_bps = self.components.timing_delay_bps.max(Decimal::ZERO);
        }

        // 2. Spread cost: Approximated as half the typical spread paid
        // Uses arrival price vs decision price
        if let Some(arrival) = self.arrival_price {
            let spread_component = match self.side {
                Side::Buy => ((arrival - self.decision_price) / self.decision_price) * dec!(10000),
                Side::Sell => ((self.decision_price - arrival) / self.decision_price) * dec!(10000),
            };
            self.components.spread_cost_bps = spread_component.max(Decimal::ZERO);
        }

        // 3. Market impact: Cost from our own order pressure
        // Approximated as difference between VWAP and arrival price (adjusted for side)
        if let Some(arrival) = self.arrival_price {
            if self.vwap > Decimal::ZERO {
                self.components.market_impact_bps = match self.side {
                    Side::Buy => ((self.vwap - arrival) / self.decision_price) * dec!(10000),
                    Side::Sell => ((arrival - self.vwap) / self.decision_price) * dec!(10000),
                };
                self.components.market_impact_bps = self.components.market_impact_bps.max(Decimal::ZERO);
            }
        }

        // 4. Opportunity cost: Cost from unfilled portion
        // Based on where the market moved vs what we could have executed
        let remaining_size = self.total_size - self.executed_size;
        if remaining_size > Decimal::ZERO && self.total_size > Decimal::ZERO {
            if let Some(current_market) = self.current_market_price {
                let unfilled_fraction = remaining_size / self.total_size;
                let price_move = match self.side {
                    Side::Buy => current_market - self.decision_price,
                    Side::Sell => self.decision_price - current_market,
                };
                // Only count as opportunity cost if market moved against us
                if price_move > Decimal::ZERO {
                    self.components.opportunity_cost_bps =
                        (price_move / self.decision_price) * unfilled_fraction * dec!(10000);
                } else {
                    self.components.opportunity_cost_bps = Decimal::ZERO;
                }
            }
        }
    }

    /// Get remaining size to execute
    pub fn remaining_size(&self) -> Decimal {
        self.total_size - self.executed_size
    }

    /// Get execution progress as percentage
    pub fn progress_pct(&self) -> Decimal {
        if self.total_size.is_zero() {
            return dec!(100);
        }
        (self.executed_size / self.total_size) * dec!(100)
    }

    /// Check if execution is experiencing adverse selection
    pub fn is_adverse(&self, threshold_bps: Decimal) -> bool {
        self.shortfall_bps > threshold_bps
    }

    /// Check if price movement is favorable
    pub fn is_favorable(&self, threshold_bps: Decimal) -> bool {
        self.shortfall_bps < -threshold_bps
    }
}

/// Tracker for implementation shortfall across multiple executions
pub struct ShortfallTracker {
    /// Active shortfall records by parent order ID
    records: Arc<RwLock<HashMap<String, ShortfallRecord>>>,
    /// Configuration
    config: ShortfallConfig,
    /// Historical completed records (limited size)
    completed_history: Arc<RwLock<Vec<ShortfallRecord>>>,
    /// Maximum history size
    max_history_size: usize,
}

impl ShortfallTracker {
    /// Create a new shortfall tracker with default config
    pub fn new() -> Self {
        Self {
            records: Arc::new(RwLock::new(HashMap::new())),
            config: ShortfallConfig::default(),
            completed_history: Arc::new(RwLock::new(Vec::new())),
            max_history_size: 1000,
        }
    }

    /// Create a new shortfall tracker with custom config
    pub fn with_config(config: ShortfallConfig) -> Self {
        Self {
            records: Arc::new(RwLock::new(HashMap::new())),
            config,
            completed_history: Arc::new(RwLock::new(Vec::new())),
            max_history_size: 1000,
        }
    }

    /// Start tracking a new execution
    pub async fn start_tracking(
        &self,
        parent_order_id: String,
        decision_price: Decimal,
        decision_time: DateTime<Utc>,
        side: Side,
        total_size: Decimal,
        arrival_price: Option<Decimal>,
    ) {
        let record = ShortfallRecord::new(
            parent_order_id.clone(),
            decision_price,
            decision_time,
            side,
            total_size,
            arrival_price,
        );

        self.records.write().await.insert(parent_order_id.clone(), record);

        info!(
            parent_id = %parent_order_id,
            decision_price = %decision_price,
            side = ?side,
            total_size = %total_size,
            "Started shortfall tracking"
        );
    }

    /// Record a fill for an execution
    pub async fn record_fill(
        &self,
        parent_order_id: &str,
        fill_size: Decimal,
        fill_price: Decimal,
        current_mid_price: Option<Decimal>,
    ) {
        let mut records = self.records.write().await;
        if let Some(record) = records.get_mut(parent_order_id) {
            record.record_fill(fill_size, fill_price, current_mid_price);

            // Update speed adjustment recommendation
            record.speed_adjustment = self.calculate_speed_adjustment(record);

            // Move to history if complete
            if record.is_complete {
                let completed_record = record.clone();
                drop(records); // Release lock before acquiring completed_history lock

                // Move to history
                let mut history = self.completed_history.write().await;
                history.push(completed_record);

                // Trim history if needed
                while history.len() > self.max_history_size {
                    history.remove(0);
                }

                // Remove from active records
                self.records.write().await.remove(parent_order_id);
            }
        } else {
            warn!(
                parent_id = %parent_order_id,
                "Attempted to record fill for unknown order"
            );
        }
    }

    /// Update the current market price for an execution
    pub async fn update_market_price(&self, parent_order_id: &str, current_mid_price: Decimal) {
        let mut records = self.records.write().await;
        if let Some(record) = records.get_mut(parent_order_id) {
            record.current_market_price = Some(current_mid_price);
            record.calculate_shortfall();
            record.speed_adjustment = self.calculate_speed_adjustment(record);
        }
    }

    /// Get the current shortfall record for an execution
    pub async fn get_record(&self, parent_order_id: &str) -> Option<ShortfallRecord> {
        self.records.read().await.get(parent_order_id).cloned()
    }

    /// Get the recommended speed adjustment for an execution
    pub async fn get_speed_adjustment(&self, parent_order_id: &str) -> Decimal {
        self.records
            .read()
            .await
            .get(parent_order_id)
            .map(|r| r.speed_adjustment)
            .unwrap_or(Decimal::ONE)
    }

    /// Calculate speed adjustment based on shortfall
    fn calculate_speed_adjustment(&self, record: &ShortfallRecord) -> Decimal {
        if !self.config.enabled {
            return Decimal::ONE;
        }

        let adjustment = if record.is_adverse(self.config.adverse_threshold_bps) {
            // Adverse selection - speed up
            self.config.adverse_speed_multiplier
        } else if record.is_favorable(self.config.favorable_threshold_bps) {
            // Favorable movement - slow down
            self.config.favorable_speed_multiplier
        } else {
            // Neutral - normal speed
            Decimal::ONE
        };

        // Clamp to configured bounds
        adjustment
            .max(self.config.min_speed_multiplier)
            .min(self.config.max_speed_multiplier)
    }

    /// Get all active shortfall records
    pub async fn get_all_active(&self) -> Vec<ShortfallRecord> {
        self.records.read().await.values().cloned().collect()
    }

    /// Get execution statistics summary
    pub async fn get_stats_summary(&self) -> ShortfallSummary {
        let records = self.records.read().await;
        let history = self.completed_history.read().await;

        let all_records: Vec<&ShortfallRecord> = records
            .values()
            .chain(history.iter())
            .collect();

        if all_records.is_empty() {
            return ShortfallSummary::default();
        }

        let total_shortfall: Decimal = all_records.iter().map(|r| r.shortfall_bps).sum();
        let count = Decimal::from(all_records.len() as i64);
        let avg_shortfall = total_shortfall / count;

        let timing_delay: Decimal = all_records.iter().map(|r| r.components.timing_delay_bps).sum();
        let market_impact: Decimal = all_records.iter().map(|r| r.components.market_impact_bps).sum();
        let spread_cost: Decimal = all_records.iter().map(|r| r.components.spread_cost_bps).sum();
        let opportunity_cost: Decimal = all_records.iter().map(|r| r.components.opportunity_cost_bps).sum();

        ShortfallSummary {
            total_executions: all_records.len(),
            active_executions: records.len(),
            avg_shortfall_bps: avg_shortfall,
            total_shortfall_bps: total_shortfall,
            avg_timing_delay_bps: timing_delay / count,
            avg_market_impact_bps: market_impact / count,
            avg_spread_cost_bps: spread_cost / count,
            avg_opportunity_cost_bps: opportunity_cost / count,
        }
    }

    /// Mark an execution as cancelled
    pub async fn cancel(&self, parent_order_id: &str) {
        let mut records = self.records.write().await;
        if let Some(record) = records.get_mut(parent_order_id) {
            record.is_complete = true;
            record.completed_at = Some(Utc::now());

            let completed_record = record.clone();
            drop(records);

            // Move to history
            let mut history = self.completed_history.write().await;
            history.push(completed_record);

            // Trim history if needed
            while history.len() > self.max_history_size {
                history.remove(0);
            }

            // Remove from active records
            self.records.write().await.remove(parent_order_id);

            info!(parent_id = %parent_order_id, "Cancelled shortfall tracking");
        }
    }

    /// Clean up old completed records
    pub async fn cleanup_old_records(&self, max_age: Duration) {
        let cutoff = Utc::now() - max_age;
        let mut history = self.completed_history.write().await;
        history.retain(|r| {
            r.completed_at
                .map(|t| t > cutoff)
                .unwrap_or(true)
        });
    }

    /// Get the configuration
    pub fn config(&self) -> &ShortfallConfig {
        &self.config
    }
}

impl Default for ShortfallTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for shortfall tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShortfallSummary {
    /// Total number of executions tracked
    pub total_executions: usize,
    /// Number of currently active executions
    pub active_executions: usize,
    /// Average implementation shortfall in basis points
    pub avg_shortfall_bps: Decimal,
    /// Total implementation shortfall in basis points
    pub total_shortfall_bps: Decimal,
    /// Average timing delay component in basis points
    pub avg_timing_delay_bps: Decimal,
    /// Average market impact component in basis points
    pub avg_market_impact_bps: Decimal,
    /// Average spread cost component in basis points
    pub avg_spread_cost_bps: Decimal,
    /// Average opportunity cost component in basis points
    pub avg_opportunity_cost_bps: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shortfall_record_creation() {
        let record = ShortfallRecord::new(
            "test_order".to_string(),
            dec!(0.50),
            Utc::now(),
            Side::Buy,
            dec!(1000),
            Some(dec!(0.50)),
        );

        assert_eq!(record.parent_order_id, "test_order");
        assert_eq!(record.decision_price, dec!(0.50));
        assert_eq!(record.total_size, dec!(1000));
        assert_eq!(record.executed_size, Decimal::ZERO);
        assert!(!record.is_complete);
    }

    #[test]
    fn test_shortfall_record_fill() {
        let mut record = ShortfallRecord::new(
            "test_order".to_string(),
            dec!(0.50),
            Utc::now(),
            Side::Buy,
            dec!(1000),
            Some(dec!(0.50)),
        );

        // Record a fill at a higher price (slippage for buy)
        record.record_fill(dec!(500), dec!(0.52), Some(dec!(0.51)));

        assert_eq!(record.executed_size, dec!(500));
        assert_eq!(record.vwap, dec!(0.52));
        assert_eq!(record.num_fills, 1);
        assert!(record.first_fill_time.is_some());
        assert_eq!(record.first_fill_price, Some(dec!(0.52)));

        // Shortfall should be positive (paid more than decision price)
        // (0.52 - 0.50) / 0.50 * 10000 = 400 bps
        assert_eq!(record.shortfall_bps, dec!(400));
    }

    #[test]
    fn test_shortfall_record_multiple_fills() {
        let mut record = ShortfallRecord::new(
            "test_order".to_string(),
            dec!(0.50),
            Utc::now(),
            Side::Buy,
            dec!(1000),
            Some(dec!(0.50)),
        );

        // First fill
        record.record_fill(dec!(500), dec!(0.51), Some(dec!(0.51)));
        // Second fill
        record.record_fill(dec!(500), dec!(0.53), Some(dec!(0.52)));

        assert_eq!(record.executed_size, dec!(1000));
        // VWAP = (500 * 0.51 + 500 * 0.53) / 1000 = 0.52
        assert_eq!(record.vwap, dec!(0.52));
        assert_eq!(record.num_fills, 2);
        assert!(record.is_complete);

        // Shortfall = (0.52 - 0.50) / 0.50 * 10000 = 400 bps
        assert_eq!(record.shortfall_bps, dec!(400));
    }

    #[test]
    fn test_shortfall_sell_side() {
        let mut record = ShortfallRecord::new(
            "test_order".to_string(),
            dec!(0.50),
            Utc::now(),
            Side::Sell,
            dec!(1000),
            Some(dec!(0.50)),
        );

        // Record a fill at a lower price (slippage for sell)
        record.record_fill(dec!(1000), dec!(0.48), Some(dec!(0.49)));

        // Shortfall should be positive (received less than decision price)
        // (0.50 - 0.48) / 0.50 * 10000 = 400 bps
        assert_eq!(record.shortfall_bps, dec!(400));
    }

    #[test]
    fn test_favorable_price_movement() {
        let mut record = ShortfallRecord::new(
            "test_order".to_string(),
            dec!(0.50),
            Utc::now(),
            Side::Buy,
            dec!(1000),
            Some(dec!(0.50)),
        );

        // Record a fill at a lower price (favorable for buy)
        record.record_fill(dec!(1000), dec!(0.48), Some(dec!(0.48)));

        // Shortfall should be negative (paid less than decision price)
        // (0.48 - 0.50) / 0.50 * 10000 = -400 bps
        assert_eq!(record.shortfall_bps, dec!(-400));
        assert!(record.is_favorable(dec!(10)));
    }

    #[test]
    fn test_adverse_detection() {
        let mut record = ShortfallRecord::new(
            "test_order".to_string(),
            dec!(0.50),
            Utc::now(),
            Side::Buy,
            dec!(1000),
            Some(dec!(0.50)),
        );

        // Fill at higher price (adverse for buy)
        record.record_fill(dec!(500), dec!(0.52), Some(dec!(0.52)));

        assert!(record.is_adverse(dec!(20))); // > 20 bps threshold
        assert!(!record.is_favorable(dec!(10)));
    }

    #[tokio::test]
    async fn test_shortfall_tracker() {
        let tracker = ShortfallTracker::new();

        let now = Utc::now();
        tracker
            .start_tracking(
                "order_1".to_string(),
                dec!(0.50),
                now,
                Side::Buy,
                dec!(1000),
                Some(dec!(0.50)),
            )
            .await;

        // Record fills
        tracker
            .record_fill("order_1", dec!(500), dec!(0.51), Some(dec!(0.51)))
            .await;

        let record = tracker.get_record("order_1").await.unwrap();
        assert_eq!(record.executed_size, dec!(500));
        assert_eq!(record.vwap, dec!(0.51));

        // Get speed adjustment
        let adjustment = tracker.get_speed_adjustment("order_1").await;
        // With 200 bps shortfall and 20 bps threshold, should be adverse
        assert!(adjustment > Decimal::ONE);
    }

    #[tokio::test]
    async fn test_tracker_completion() {
        let tracker = ShortfallTracker::new();

        let now = Utc::now();
        tracker
            .start_tracking(
                "order_1".to_string(),
                dec!(0.50),
                now,
                Side::Buy,
                dec!(100),
                Some(dec!(0.50)),
            )
            .await;

        // Complete the order
        tracker
            .record_fill("order_1", dec!(100), dec!(0.51), Some(dec!(0.51)))
            .await;

        // Should be moved to history
        let record = tracker.get_record("order_1").await;
        assert!(record.is_none());

        // Check summary
        let summary = tracker.get_stats_summary().await;
        assert_eq!(summary.total_executions, 1);
        assert_eq!(summary.active_executions, 0);
    }

    #[test]
    fn test_shortfall_components() {
        let mut record = ShortfallRecord::new(
            "test_order".to_string(),
            dec!(0.50),
            Utc::now(),
            Side::Buy,
            dec!(1000),
            Some(dec!(0.505)), // Arrival price slightly higher
        );

        // First fill
        record.record_fill(dec!(500), dec!(0.51), Some(dec!(0.515)));

        // Components should be calculated
        assert!(record.components.timing_delay_bps >= Decimal::ZERO);
        assert!(record.components.spread_cost_bps >= Decimal::ZERO);

        // Update with opportunity cost scenario
        record.current_market_price = Some(dec!(0.53)); // Market moved against us
        record.calculate_shortfall();
        assert!(record.components.opportunity_cost_bps >= Decimal::ZERO);
    }

    #[test]
    fn test_progress_calculation() {
        let mut record = ShortfallRecord::new(
            "test_order".to_string(),
            dec!(0.50),
            Utc::now(),
            Side::Buy,
            dec!(1000),
            None,
        );

        assert_eq!(record.progress_pct(), dec!(0));

        record.record_fill(dec!(500), dec!(0.50), None);
        assert_eq!(record.progress_pct(), dec!(50));

        record.record_fill(dec!(500), dec!(0.50), None);
        assert_eq!(record.progress_pct(), dec!(100));
    }

    #[test]
    fn test_config_defaults() {
        let config = ShortfallConfig::default();
        assert!(config.enabled);
        assert_eq!(config.adverse_threshold_bps, dec!(20));
        assert_eq!(config.favorable_threshold_bps, dec!(10));
    }
}
