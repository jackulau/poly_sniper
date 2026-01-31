//! Volume Monitor for tracking real-time trading volume
//!
//! Provides rolling volume history to enable dynamic participation rate adaptation
//! based on actual market activity rather than static configuration.

use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tracing::{debug, instrument};

/// Unique identifier for a token
pub type TokenId = String;

/// A single volume observation at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeObservation {
    /// When the observation was recorded
    pub timestamp: DateTime<Utc>,
    /// Volume observed during this interval
    pub volume: Decimal,
    /// Duration of the observation interval in seconds
    pub interval_secs: u64,
}

/// Configuration for the volume monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeMonitorConfig {
    /// How far back to look for volume history (default: 3600 seconds = 1 hour)
    pub history_window_secs: u64,
    /// Granularity of volume observations (default: 60 seconds)
    pub observation_interval_secs: u64,
    /// Exponential moving average smoothing factor (default: 0.2)
    /// Higher values give more weight to recent observations
    pub smoothing_factor: Decimal,
}

impl Default for VolumeMonitorConfig {
    fn default() -> Self {
        Self {
            history_window_secs: 3600,      // 1 hour
            observation_interval_secs: 60, // 1 minute
            smoothing_factor: dec!(0.2),   // EMA alpha
        }
    }
}

/// Monitors and tracks rolling volume for tokens
pub struct VolumeMonitor {
    /// Rolling volume observations per token
    volume_history: HashMap<TokenId, VecDeque<VolumeObservation>>,
    /// EMA of volume rate per token (volume per second)
    ema_volume_rate: HashMap<TokenId, Decimal>,
    /// Configuration
    config: VolumeMonitorConfig,
}

impl VolumeMonitor {
    /// Create a new volume monitor with the given configuration
    pub fn new(config: VolumeMonitorConfig) -> Self {
        Self {
            volume_history: HashMap::new(),
            ema_volume_rate: HashMap::new(),
            config,
        }
    }

    /// Create a volume monitor with default configuration
    pub fn with_defaults() -> Self {
        Self::new(VolumeMonitorConfig::default())
    }

    /// Record a volume observation for a token
    #[instrument(skip(self), fields(token_id = %token_id, volume = %volume))]
    pub fn record_volume(&mut self, token_id: &TokenId, volume: Decimal) {
        let now = Utc::now();
        let observation = VolumeObservation {
            timestamp: now,
            volume,
            interval_secs: self.config.observation_interval_secs,
        };

        // Get or create history for this token
        let history = self
            .volume_history
            .entry(token_id.clone())
            .or_default();

        history.push_back(observation);

        // Prune old observations inline to avoid borrow issues
        let cutoff = Utc::now() - Duration::seconds(self.config.history_window_secs as i64);
        while let Some(front) = history.front() {
            if front.timestamp < cutoff {
                history.pop_front();
            } else {
                break;
            }
        }

        let history_len = history.len();

        // Update EMA
        let volume_rate = volume / Decimal::from(self.config.observation_interval_secs);
        let alpha = self.config.smoothing_factor;

        let new_ema = match self.ema_volume_rate.get(token_id) {
            Some(&current_ema) => {
                // EMA = alpha * new_value + (1 - alpha) * old_ema
                alpha * volume_rate + (Decimal::ONE - alpha) * current_ema
            }
            None => {
                // First observation, use raw value
                volume_rate
            }
        };

        self.ema_volume_rate.insert(token_id.clone(), new_ema);

        debug!(
            token_id = %token_id,
            volume = %volume,
            history_len = history_len,
            "Recorded volume observation"
        );
    }

    /// Get the current volume rate (volume per second) based on recent observations
    pub fn get_current_volume_rate(&self, token_id: &TokenId) -> Decimal {
        // Use EMA as the current volume rate
        self.ema_volume_rate
            .get(token_id)
            .copied()
            .unwrap_or(Decimal::ZERO)
    }

    /// Get the average volume rate over the history window
    pub fn get_average_volume_rate(&self, token_id: &TokenId) -> Decimal {
        let history = match self.volume_history.get(token_id) {
            Some(h) if !h.is_empty() => h,
            _ => return Decimal::ZERO,
        };

        let total_volume: Decimal = history.iter().map(|o| o.volume).sum();
        let total_time: u64 = history.iter().map(|o| o.interval_secs).sum();

        if total_time == 0 {
            return Decimal::ZERO;
        }

        total_volume / Decimal::from(total_time)
    }

    /// Get the volume ratio (current / average)
    ///
    /// Returns 1.0 if insufficient data. Values > 1.0 indicate above-average volume.
    pub fn get_volume_ratio(&self, token_id: &TokenId) -> Decimal {
        let current = self.get_current_volume_rate(token_id);
        let average = self.get_average_volume_rate(token_id);

        if average.is_zero() {
            // No historical data, assume normal volume
            return Decimal::ONE;
        }

        // Clamp to reasonable bounds to prevent extreme values
        let ratio = current / average;
        ratio.max(dec!(0.1)).min(dec!(10.0))
    }

    /// Get the total volume in the observation window
    pub fn get_total_volume(&self, token_id: &TokenId) -> Decimal {
        self.volume_history
            .get(token_id)
            .map(|h| h.iter().map(|o| o.volume).sum())
            .unwrap_or(Decimal::ZERO)
    }

    /// Get the number of observations for a token
    pub fn get_observation_count(&self, token_id: &TokenId) -> usize {
        self.volume_history
            .get(token_id)
            .map(|h| h.len())
            .unwrap_or(0)
    }

    /// Check if there is sufficient data for reliable estimation
    pub fn has_sufficient_data(&self, token_id: &TokenId) -> bool {
        // Need at least 5 observations for reasonable estimates
        self.get_observation_count(token_id) >= 5
    }

    /// Clear all history for a token
    pub fn clear_token(&mut self, token_id: &TokenId) {
        self.volume_history.remove(token_id);
        self.ema_volume_rate.remove(token_id);
    }

    /// Clear all history
    pub fn clear_all(&mut self) {
        self.volume_history.clear();
        self.ema_volume_rate.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_monitor() -> VolumeMonitor {
        VolumeMonitor::new(VolumeMonitorConfig {
            history_window_secs: 300, // 5 minutes for testing
            observation_interval_secs: 60,
            smoothing_factor: dec!(0.3),
        })
    }

    #[test]
    fn test_record_volume() {
        let mut monitor = create_test_monitor();
        let token_id = "token_1".to_string();

        monitor.record_volume(&token_id, dec!(1000));
        assert_eq!(monitor.get_observation_count(&token_id), 1);
        assert_eq!(monitor.get_total_volume(&token_id), dec!(1000));

        monitor.record_volume(&token_id, dec!(2000));
        assert_eq!(monitor.get_observation_count(&token_id), 2);
        assert_eq!(monitor.get_total_volume(&token_id), dec!(3000));
    }

    #[test]
    fn test_volume_rate_calculation() {
        let mut monitor = create_test_monitor();
        let token_id = "token_1".to_string();

        // Record volume of 6000 over a 60-second interval = 100/sec rate
        monitor.record_volume(&token_id, dec!(6000));

        let current_rate = monitor.get_current_volume_rate(&token_id);
        // 6000 / 60 = 100
        assert_eq!(current_rate, dec!(100));
    }

    #[test]
    fn test_volume_ratio_no_data() {
        let monitor = create_test_monitor();
        let token_id = "token_1".to_string();

        // No data should return 1.0 (normal volume)
        let ratio = monitor.get_volume_ratio(&token_id);
        assert_eq!(ratio, Decimal::ONE);
    }

    #[test]
    fn test_volume_ratio_with_data() {
        let mut monitor = create_test_monitor();
        let token_id = "token_1".to_string();

        // Record consistent volume
        for _ in 0..5 {
            monitor.record_volume(&token_id, dec!(6000));
        }

        // With consistent volume, ratio should be close to 1
        let ratio = monitor.get_volume_ratio(&token_id);
        assert!(ratio > dec!(0.5) && ratio < dec!(2.0));
    }

    #[test]
    fn test_ema_smoothing() {
        let mut monitor = create_test_monitor();
        let token_id = "token_1".to_string();

        // First observation sets the baseline
        monitor.record_volume(&token_id, dec!(6000)); // 100/sec
        let rate1 = monitor.get_current_volume_rate(&token_id);
        assert_eq!(rate1, dec!(100));

        // Second observation with higher volume should increase EMA
        monitor.record_volume(&token_id, dec!(12000)); // 200/sec
        let rate2 = monitor.get_current_volume_rate(&token_id);
        assert!(rate2 > rate1);
        assert!(rate2 < dec!(200)); // Should be smoothed, not jump to 200
    }

    #[test]
    fn test_ratio_clamping() {
        let mut monitor = VolumeMonitor::new(VolumeMonitorConfig {
            history_window_secs: 3600,
            observation_interval_secs: 60,
            smoothing_factor: dec!(1.0), // No smoothing for this test
        });
        let token_id = "token_1".to_string();

        // Create artificial scenario with extreme ratio
        // First, add many low-volume observations
        for _ in 0..10 {
            monitor.record_volume(&token_id, dec!(100));
        }

        // Ratio should be clamped between 0.1 and 10.0
        let ratio = monitor.get_volume_ratio(&token_id);
        assert!(ratio >= dec!(0.1));
        assert!(ratio <= dec!(10.0));
    }

    #[test]
    fn test_has_sufficient_data() {
        let mut monitor = create_test_monitor();
        let token_id = "token_1".to_string();

        assert!(!monitor.has_sufficient_data(&token_id));

        for _ in 0..4 {
            monitor.record_volume(&token_id, dec!(1000));
        }
        assert!(!monitor.has_sufficient_data(&token_id));

        monitor.record_volume(&token_id, dec!(1000));
        assert!(monitor.has_sufficient_data(&token_id));
    }

    #[test]
    fn test_clear_token() {
        let mut monitor = create_test_monitor();
        let token_id = "token_1".to_string();

        monitor.record_volume(&token_id, dec!(1000));
        assert_eq!(monitor.get_observation_count(&token_id), 1);

        monitor.clear_token(&token_id);
        assert_eq!(monitor.get_observation_count(&token_id), 0);
        assert_eq!(monitor.get_current_volume_rate(&token_id), Decimal::ZERO);
    }

    #[test]
    fn test_multiple_tokens() {
        let mut monitor = create_test_monitor();
        let token_1 = "token_1".to_string();
        let token_2 = "token_2".to_string();

        monitor.record_volume(&token_1, dec!(1000));
        monitor.record_volume(&token_2, dec!(2000));

        assert_eq!(monitor.get_total_volume(&token_1), dec!(1000));
        assert_eq!(monitor.get_total_volume(&token_2), dec!(2000));
    }
}
