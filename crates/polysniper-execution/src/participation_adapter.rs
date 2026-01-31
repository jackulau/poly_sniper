//! Participation Rate Adapter for dynamic participation adjustment
//!
//! Calculates adaptive participation rates based on real-time volume conditions,
//! order urgency, and time pressure to optimize execution quality.

use crate::volume_monitor::VolumeMonitor;
use polysniper_core::Priority;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, instrument};

/// Configuration for adaptive participation rate calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipationConfig {
    /// Base participation rate (default: 0.10 = 10%)
    pub base_rate: Decimal,
    /// Minimum participation rate floor (default: 0.02 = 2%)
    pub min_rate: Decimal,
    /// Maximum participation rate ceiling (default: 0.25 = 25%)
    pub max_rate: Decimal,
    /// How aggressively to scale with volume (default: 0.5)
    /// Higher values make participation more responsive to volume changes
    pub volume_scaling_factor: Decimal,
    /// Additional participation boost for urgent orders (default: 0.05 = 5%)
    pub urgency_boost: Decimal,
    /// Boost participation in final 20% of execution window (default: 1.3 = 30% boost)
    pub time_pressure_boost: Decimal,
    /// Threshold for time pressure (remaining time percentage below which boost applies)
    pub time_pressure_threshold: Decimal,
}

impl Default for ParticipationConfig {
    fn default() -> Self {
        Self {
            base_rate: dec!(0.10),           // 10% base participation
            min_rate: dec!(0.02),            // 2% minimum
            max_rate: dec!(0.25),            // 25% maximum
            volume_scaling_factor: dec!(0.5), // Moderate scaling
            urgency_boost: dec!(0.05),       // 5% boost for urgency
            time_pressure_boost: dec!(1.3),  // 30% boost near deadline
            time_pressure_threshold: dec!(0.2), // Apply in final 20% of time
        }
    }
}

/// Result of a participation rate calculation
#[derive(Debug, Clone)]
pub struct ParticipationRate {
    /// Final calculated participation rate
    pub rate: Decimal,
    /// Base rate before adjustments
    pub base_rate: Decimal,
    /// Volume adjustment factor
    pub volume_factor: Decimal,
    /// Urgency adjustment factor
    pub urgency_factor: Decimal,
    /// Time pressure adjustment factor
    pub time_pressure_factor: Decimal,
    /// Whether minimum bound was applied
    pub clamped_min: bool,
    /// Whether maximum bound was applied
    pub clamped_max: bool,
}

/// Calculates adaptive participation rates based on market conditions
pub struct ParticipationAdapter {
    volume_monitor: Arc<RwLock<VolumeMonitor>>,
    config: ParticipationConfig,
}

impl ParticipationAdapter {
    /// Create a new participation adapter
    pub fn new(
        volume_monitor: Arc<RwLock<VolumeMonitor>>,
        config: ParticipationConfig,
    ) -> Self {
        Self {
            volume_monitor,
            config,
        }
    }

    /// Create a participation adapter with default configuration
    pub fn with_defaults(volume_monitor: Arc<RwLock<VolumeMonitor>>) -> Self {
        Self::new(volume_monitor, ParticipationConfig::default())
    }

    /// Calculate the adaptive participation rate for a given token
    ///
    /// # Arguments
    /// * `token_id` - The token being traded
    /// * `urgency` - Priority level of the order
    /// * `remaining_time_pct` - Percentage of execution window remaining (0.0 to 1.0)
    ///
    /// # Returns
    /// A ParticipationRate with the calculated rate and breakdown
    #[instrument(skip(self), fields(token_id = %token_id, urgency = ?urgency, remaining_time_pct = %remaining_time_pct))]
    pub async fn calculate_rate(
        &self,
        token_id: &str,
        urgency: Priority,
        remaining_time_pct: Decimal,
    ) -> ParticipationRate {
        // Get volume ratio from monitor
        let volume_ratio = {
            let monitor = self.volume_monitor.read().await;
            monitor.get_volume_ratio(&token_id.to_string())
        };

        // Calculate volume-scaled rate
        // high volume = higher participation allowed
        // formula: base_rate * volume_ratio^scaling_factor
        let volume_factor = self.calculate_volume_factor(volume_ratio);
        let scaled_rate = self.config.base_rate * volume_factor;

        // Apply urgency multiplier
        let urgency_factor = self.get_urgency_multiplier(urgency);

        // Apply time pressure boost if near end of window
        let time_pressure_factor = if remaining_time_pct < self.config.time_pressure_threshold {
            self.config.time_pressure_boost
        } else {
            Decimal::ONE
        };

        // Calculate final rate
        let raw_rate = scaled_rate * urgency_factor * time_pressure_factor;

        // Clamp to bounds
        let clamped_min = raw_rate < self.config.min_rate;
        let clamped_max = raw_rate > self.config.max_rate;
        let final_rate = raw_rate.max(self.config.min_rate).min(self.config.max_rate);

        debug!(
            token_id = %token_id,
            volume_ratio = %volume_ratio,
            volume_factor = %volume_factor,
            urgency_factor = %urgency_factor,
            time_pressure_factor = %time_pressure_factor,
            raw_rate = %raw_rate,
            final_rate = %final_rate,
            clamped_min,
            clamped_max,
            "Calculated adaptive participation rate"
        );

        ParticipationRate {
            rate: final_rate,
            base_rate: self.config.base_rate,
            volume_factor,
            urgency_factor,
            time_pressure_factor,
            clamped_min,
            clamped_max,
        }
    }

    /// Calculate the simple participation rate without async
    /// Use when you already have the volume ratio
    pub fn calculate_rate_sync(
        &self,
        volume_ratio: Decimal,
        urgency: Priority,
        remaining_time_pct: Decimal,
    ) -> ParticipationRate {
        let volume_factor = self.calculate_volume_factor(volume_ratio);
        let scaled_rate = self.config.base_rate * volume_factor;
        let urgency_factor = self.get_urgency_multiplier(urgency);
        let time_pressure_factor = if remaining_time_pct < self.config.time_pressure_threshold {
            self.config.time_pressure_boost
        } else {
            Decimal::ONE
        };

        let raw_rate = scaled_rate * urgency_factor * time_pressure_factor;
        let clamped_min = raw_rate < self.config.min_rate;
        let clamped_max = raw_rate > self.config.max_rate;
        let final_rate = raw_rate.max(self.config.min_rate).min(self.config.max_rate);

        ParticipationRate {
            rate: final_rate,
            base_rate: self.config.base_rate,
            volume_factor,
            urgency_factor,
            time_pressure_factor,
            clamped_min,
            clamped_max,
        }
    }

    /// Calculate the volume adjustment factor using power function
    fn calculate_volume_factor(&self, volume_ratio: Decimal) -> Decimal {
        // Use approximation: volume_ratio^scaling_factor
        // For decimal, we use a linear interpolation approach for the power
        let scaling = self.config.volume_scaling_factor;

        if volume_ratio == Decimal::ONE {
            return Decimal::ONE;
        }

        // Simple approximation: 1 + scaling * (ratio - 1)
        // This gives a linear approximation of the power function near 1
        // For ratio=2 and scaling=0.5: 1 + 0.5 * 1 = 1.5 (actual 2^0.5 ≈ 1.41)
        // For ratio=0.5 and scaling=0.5: 1 + 0.5 * -0.5 = 0.75 (actual 0.5^0.5 ≈ 0.71)
        let factor = Decimal::ONE + scaling * (volume_ratio - Decimal::ONE);
        factor.max(dec!(0.1)) // Ensure positive
    }

    /// Get the urgency multiplier based on priority level
    fn get_urgency_multiplier(&self, urgency: Priority) -> Decimal {
        match urgency {
            Priority::Critical => dec!(1.5),  // 50% boost
            Priority::High => dec!(1.25),     // 25% boost
            Priority::Normal => dec!(1.0),    // No change
            Priority::Low => dec!(0.8),       // 20% reduction
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &ParticipationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume_monitor::VolumeMonitorConfig;

    async fn create_test_adapter() -> ParticipationAdapter {
        let monitor = Arc::new(RwLock::new(VolumeMonitor::new(VolumeMonitorConfig {
            history_window_secs: 300,
            observation_interval_secs: 60,
            smoothing_factor: dec!(0.2),
        })));
        ParticipationAdapter::with_defaults(monitor)
    }

    #[tokio::test]
    async fn test_base_rate() {
        let adapter = create_test_adapter().await;

        // With no volume data, should return base rate
        let result = adapter
            .calculate_rate("token_1", Priority::Normal, dec!(0.5))
            .await;

        assert_eq!(result.rate, dec!(0.10)); // Default base rate
        assert_eq!(result.urgency_factor, Decimal::ONE);
        assert_eq!(result.time_pressure_factor, Decimal::ONE);
    }

    #[tokio::test]
    async fn test_urgency_boost_critical() {
        let adapter = create_test_adapter().await;

        let result = adapter
            .calculate_rate("token_1", Priority::Critical, dec!(0.5))
            .await;

        // Base 10% * 1.5 (critical) = 15%
        assert_eq!(result.rate, dec!(0.15));
        assert_eq!(result.urgency_factor, dec!(1.5));
    }

    #[tokio::test]
    async fn test_urgency_reduction_low() {
        let adapter = create_test_adapter().await;

        let result = adapter
            .calculate_rate("token_1", Priority::Low, dec!(0.5))
            .await;

        // Base 10% * 0.8 (low) = 8%
        assert_eq!(result.rate, dec!(0.08));
        assert_eq!(result.urgency_factor, dec!(0.8));
    }

    #[tokio::test]
    async fn test_time_pressure_boost() {
        let adapter = create_test_adapter().await;

        // 10% remaining (below 20% threshold)
        let result = adapter
            .calculate_rate("token_1", Priority::Normal, dec!(0.10))
            .await;

        // Base 10% * 1.3 (time pressure) = 13%
        assert_eq!(result.rate, dec!(0.13));
        assert_eq!(result.time_pressure_factor, dec!(1.3));
    }

    #[tokio::test]
    async fn test_no_time_pressure_above_threshold() {
        let adapter = create_test_adapter().await;

        // 50% remaining (above 20% threshold)
        let result = adapter
            .calculate_rate("token_1", Priority::Normal, dec!(0.50))
            .await;

        assert_eq!(result.time_pressure_factor, Decimal::ONE);
    }

    #[tokio::test]
    async fn test_max_rate_clamping() {
        let monitor = Arc::new(RwLock::new(VolumeMonitor::with_defaults()));
        let config = ParticipationConfig {
            base_rate: dec!(0.30), // High base rate
            max_rate: dec!(0.25),
            ..Default::default()
        };
        let adapter = ParticipationAdapter::new(monitor, config);

        let result = adapter
            .calculate_rate("token_1", Priority::Critical, dec!(0.10)) // Many boosts
            .await;

        assert_eq!(result.rate, dec!(0.25)); // Clamped to max
        assert!(result.clamped_max);
    }

    #[tokio::test]
    async fn test_min_rate_clamping() {
        let monitor = Arc::new(RwLock::new(VolumeMonitor::with_defaults()));
        let config = ParticipationConfig {
            base_rate: dec!(0.01), // Very low base rate
            min_rate: dec!(0.02),
            ..Default::default()
        };
        let adapter = ParticipationAdapter::new(monitor, config);

        let result = adapter
            .calculate_rate("token_1", Priority::Low, dec!(0.5))
            .await;

        assert_eq!(result.rate, dec!(0.02)); // Clamped to min
        assert!(result.clamped_min);
    }

    #[test]
    fn test_volume_factor_calculation() {
        let monitor = Arc::new(tokio::sync::RwLock::new(VolumeMonitor::with_defaults()));
        let adapter = ParticipationAdapter::with_defaults(monitor);

        // Normal volume
        let factor = adapter.calculate_volume_factor(Decimal::ONE);
        assert_eq!(factor, Decimal::ONE);

        // High volume (2x)
        let factor = adapter.calculate_volume_factor(dec!(2.0));
        assert!(factor > Decimal::ONE);

        // Low volume (0.5x)
        let factor = adapter.calculate_volume_factor(dec!(0.5));
        assert!(factor < Decimal::ONE);
    }

    #[test]
    fn test_sync_calculation() {
        let monitor = Arc::new(tokio::sync::RwLock::new(VolumeMonitor::with_defaults()));
        let adapter = ParticipationAdapter::with_defaults(monitor);

        let result = adapter.calculate_rate_sync(
            dec!(2.0),           // High volume
            Priority::High,      // High priority
            dec!(0.10),          // Low remaining time
        );

        // Should have all boosts applied
        assert!(result.rate > dec!(0.10)); // Above base rate
        assert!(result.volume_factor > Decimal::ONE);
        assert_eq!(result.urgency_factor, dec!(1.25));
        assert_eq!(result.time_pressure_factor, dec!(1.3));
    }

    #[tokio::test]
    async fn test_with_recorded_volume() {
        let monitor = Arc::new(RwLock::new(VolumeMonitor::with_defaults()));

        // Record some volume data
        {
            let mut m = monitor.write().await;
            for _ in 0..5 {
                m.record_volume(&"token_1".to_string(), dec!(6000));
            }
        }

        let adapter = ParticipationAdapter::with_defaults(monitor);

        let result = adapter
            .calculate_rate("token_1", Priority::Normal, dec!(0.5))
            .await;

        // With consistent volume, should be close to base rate
        assert!(result.rate > dec!(0.05));
        assert!(result.rate < dec!(0.20));
    }
}
