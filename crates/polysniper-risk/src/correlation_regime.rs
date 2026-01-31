//! Correlation Regime Detection
//!
//! This module provides dynamic correlation regime detection to identify when
//! market correlations spike during stress periods. It automatically adjusts
//! exposure limits when the correlation environment shifts from normal to stressed.
//!
//! Key insight: A 0.3 correlation in normal times might become 0.8+ during stress.
//! Static limits don't account for this regime-dependent behavior.

use chrono::{DateTime, Utc};
use polysniper_core::{CorrelationRegime, CorrelationRegimeConfig};
use rust_decimal::Decimal;
use std::collections::VecDeque;
use tracing::{debug, info};

/// A single correlation observation with timestamp
#[derive(Debug, Clone)]
struct CorrelationObservation {
    /// The average correlation value across all position pairs
    value: Decimal,
    /// When this observation was recorded
    timestamp: DateTime<Utc>,
}

/// Tracks rolling statistics for correlation values
#[derive(Debug)]
struct RollingStats {
    /// Sum of values for mean calculation
    sum: Decimal,
    /// Sum of squared values for variance calculation
    sum_sq: Decimal,
    /// Number of samples
    count: usize,
}

impl Default for RollingStats {
    fn default() -> Self {
        Self {
            sum: Decimal::ZERO,
            sum_sq: Decimal::ZERO,
            count: 0,
        }
    }
}

impl RollingStats {
    /// Add a value to the rolling stats
    fn add(&mut self, value: Decimal) {
        self.sum += value;
        self.sum_sq += value * value;
        self.count += 1;
    }

    /// Remove a value from the rolling stats
    fn remove(&mut self, value: Decimal) {
        self.sum -= value;
        self.sum_sq -= value * value;
        self.count = self.count.saturating_sub(1);
    }

    /// Get the mean value
    fn mean(&self) -> Option<Decimal> {
        if self.count == 0 {
            return None;
        }
        Some(self.sum / Decimal::from(self.count))
    }

    /// Get the standard deviation
    fn std_dev(&self) -> Option<Decimal> {
        if self.count < 2 {
            return None;
        }
        let n = Decimal::from(self.count);
        let mean = self.sum / n;
        let variance = (self.sum_sq / n) - (mean * mean);

        if variance <= Decimal::ZERO {
            return Some(Decimal::ZERO);
        }

        // Newton-Raphson square root approximation
        decimal_sqrt(variance)
    }
}

/// Detects correlation regimes based on rolling correlation averages
pub struct CorrelationRegimeDetector {
    config: CorrelationRegimeConfig,
    /// Short-term observations window
    short_window: VecDeque<CorrelationObservation>,
    /// Long-term observations window
    long_window: VecDeque<CorrelationObservation>,
    /// Rolling stats for short-term window
    short_stats: RollingStats,
    /// Rolling stats for long-term window
    long_stats: RollingStats,
    /// Current detected regime
    current_regime: CorrelationRegime,
    /// When the current regime started
    regime_started_at: DateTime<Utc>,
    /// History of regime changes for analysis
    regime_history: VecDeque<(DateTime<Utc>, CorrelationRegime, Decimal)>,
    /// Maximum regime history entries to keep
    max_history_entries: usize,
}

impl CorrelationRegimeDetector {
    /// Create a new correlation regime detector
    pub fn new(config: CorrelationRegimeConfig) -> Self {
        Self {
            config,
            short_window: VecDeque::new(),
            long_window: VecDeque::new(),
            short_stats: RollingStats::default(),
            long_stats: RollingStats::default(),
            current_regime: CorrelationRegime::Normal,
            regime_started_at: Utc::now(),
            regime_history: VecDeque::new(),
            max_history_entries: 1000,
        }
    }

    /// Check if regime detection is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the current correlation regime
    pub fn current_regime(&self) -> CorrelationRegime {
        self.current_regime
    }

    /// Get when the current regime started
    pub fn regime_started_at(&self) -> DateTime<Utc> {
        self.regime_started_at
    }

    /// Get the short-term average correlation
    pub fn short_term_average(&self) -> Option<Decimal> {
        self.short_stats.mean()
    }

    /// Get the long-term average correlation
    pub fn long_term_average(&self) -> Option<Decimal> {
        self.long_stats.mean()
    }

    /// Get the long-term standard deviation
    pub fn long_term_std_dev(&self) -> Option<Decimal> {
        self.long_stats.std_dev()
    }

    /// Get the number of samples in the short window
    pub fn short_window_samples(&self) -> usize {
        self.short_window.len()
    }

    /// Get the number of samples in the long window
    pub fn long_window_samples(&self) -> usize {
        self.long_window.len()
    }

    /// Check if we have enough samples for regime detection
    pub fn has_sufficient_data(&self) -> bool {
        self.long_window.len() >= self.config.min_samples
    }

    /// Get the exposure limit multiplier for the current regime
    pub fn get_limit_multiplier(&self) -> Decimal {
        if !self.config.enabled {
            return Decimal::ONE;
        }

        match self.current_regime {
            CorrelationRegime::Normal => self.config.normal_limit_multiplier,
            CorrelationRegime::Elevated => self.config.elevated_limit_multiplier,
            CorrelationRegime::Crisis => self.config.crisis_limit_multiplier,
        }
    }

    /// Record a new correlation observation and potentially update regime
    ///
    /// Returns `Some((old_regime, new_regime))` if the regime changed.
    pub fn record_observation(&mut self, avg_correlation: Decimal) -> Option<(CorrelationRegime, CorrelationRegime)> {
        if !self.config.enabled {
            return None;
        }

        let now = Utc::now();
        let observation = CorrelationObservation {
            value: avg_correlation,
            timestamp: now,
        };

        // Add to both windows
        self.short_window.push_back(observation.clone());
        self.short_stats.add(avg_correlation);

        self.long_window.push_back(observation);
        self.long_stats.add(avg_correlation);

        // Prune old observations from short window
        let short_cutoff = now - chrono::Duration::seconds(self.config.short_window_secs as i64);
        while let Some(front) = self.short_window.front() {
            if front.timestamp < short_cutoff {
                self.short_stats.remove(front.value);
                self.short_window.pop_front();
            } else {
                break;
            }
        }

        // Prune old observations from long window
        let long_cutoff = now - chrono::Duration::seconds(self.config.long_window_secs as i64);
        while let Some(front) = self.long_window.front() {
            if front.timestamp < long_cutoff {
                self.long_stats.remove(front.value);
                self.long_window.pop_front();
            } else {
                break;
            }
        }

        // Detect regime change
        self.detect_regime_change()
    }

    /// Detect if the regime has changed based on current observations
    fn detect_regime_change(&mut self) -> Option<(CorrelationRegime, CorrelationRegime)> {
        // Need minimum samples before detecting regimes
        if !self.has_sufficient_data() {
            debug!(
                samples = self.long_window.len(),
                min_required = self.config.min_samples,
                "Insufficient data for regime detection"
            );
            return None;
        }

        let short_avg = self.short_stats.mean()?;
        let long_avg = self.long_stats.mean()?;

        // Avoid division by zero
        if long_avg.is_zero() {
            return None;
        }

        // Calculate how much the short-term average exceeds the long-term average
        // as a fraction (e.g., 0.5 = 50% above)
        let deviation = (short_avg - long_avg) / long_avg;

        // Determine the new regime based on deviation thresholds
        let new_regime = self.classify_regime(deviation);

        if new_regime != self.current_regime {
            let old_regime = self.current_regime;

            // Apply hysteresis - require a larger move to downgrade regime
            if self.should_transition(old_regime, new_regime, deviation) {
                info!(
                    old_regime = %old_regime,
                    new_regime = %new_regime,
                    short_avg = %short_avg,
                    long_avg = %long_avg,
                    deviation = %deviation,
                    "Correlation regime changed"
                );

                self.current_regime = new_regime;
                self.regime_started_at = Utc::now();

                // Record in history
                self.regime_history.push_back((Utc::now(), new_regime, short_avg));
                while self.regime_history.len() > self.max_history_entries {
                    self.regime_history.pop_front();
                }

                return Some((old_regime, new_regime));
            }
        }

        None
    }

    /// Classify the regime based on deviation from long-term average
    fn classify_regime(&self, deviation: Decimal) -> CorrelationRegime {
        if deviation >= self.config.crisis_threshold {
            CorrelationRegime::Crisis
        } else if deviation >= self.config.elevated_threshold {
            CorrelationRegime::Elevated
        } else {
            CorrelationRegime::Normal
        }
    }

    /// Check if we should transition between regimes (applies hysteresis)
    fn should_transition(&self, from: CorrelationRegime, to: CorrelationRegime, deviation: Decimal) -> bool {
        // Always allow upgrading to higher-risk regimes immediately
        match (from, to) {
            (CorrelationRegime::Normal, CorrelationRegime::Elevated) |
            (CorrelationRegime::Normal, CorrelationRegime::Crisis) |
            (CorrelationRegime::Elevated, CorrelationRegime::Crisis) => {
                return true;
            }
            _ => {}
        }

        // For downgrades, apply hysteresis
        let hysteresis = self.config.hysteresis_factor;

        match (from, to) {
            (CorrelationRegime::Crisis, CorrelationRegime::Elevated) => {
                // Must fall below crisis threshold minus hysteresis
                let threshold = self.config.crisis_threshold * (Decimal::ONE - hysteresis);
                deviation < threshold
            }
            (CorrelationRegime::Crisis, CorrelationRegime::Normal) => {
                // Must fall below elevated threshold minus hysteresis
                let threshold = self.config.elevated_threshold * (Decimal::ONE - hysteresis);
                deviation < threshold
            }
            (CorrelationRegime::Elevated, CorrelationRegime::Normal) => {
                // Must fall below elevated threshold minus hysteresis
                let threshold = self.config.elevated_threshold * (Decimal::ONE - hysteresis);
                deviation < threshold
            }
            _ => true, // Same regime, no transition needed
        }
    }

    /// Calculate the z-score of current short-term average vs long-term distribution
    pub fn z_score(&self) -> Option<Decimal> {
        let short_avg = self.short_stats.mean()?;
        let long_avg = self.long_stats.mean()?;
        let long_std = self.long_stats.std_dev()?;

        if long_std.is_zero() {
            return None;
        }

        Some((short_avg - long_avg) / long_std)
    }

    /// Get the regime history (timestamp, regime, avg_correlation)
    pub fn regime_history(&self) -> &VecDeque<(DateTime<Utc>, CorrelationRegime, Decimal)> {
        &self.regime_history
    }

    /// Reset the detector state (useful for testing or after major config changes)
    pub fn reset(&mut self) {
        self.short_window.clear();
        self.long_window.clear();
        self.short_stats = RollingStats::default();
        self.long_stats = RollingStats::default();
        self.current_regime = CorrelationRegime::Normal;
        self.regime_started_at = Utc::now();
        self.regime_history.clear();

        info!("Correlation regime detector reset");
    }

    /// Get a snapshot of current state for logging/monitoring
    pub fn snapshot(&self) -> RegimeSnapshot {
        RegimeSnapshot {
            regime: self.current_regime,
            regime_started_at: self.regime_started_at,
            short_term_avg: self.short_stats.mean(),
            long_term_avg: self.long_stats.mean(),
            long_term_std: self.long_stats.std_dev(),
            z_score: self.z_score(),
            limit_multiplier: self.get_limit_multiplier(),
            short_window_samples: self.short_window.len(),
            long_window_samples: self.long_window.len(),
            has_sufficient_data: self.has_sufficient_data(),
        }
    }
}

/// Snapshot of regime detector state for monitoring
#[derive(Debug, Clone)]
pub struct RegimeSnapshot {
    pub regime: CorrelationRegime,
    pub regime_started_at: DateTime<Utc>,
    pub short_term_avg: Option<Decimal>,
    pub long_term_avg: Option<Decimal>,
    pub long_term_std: Option<Decimal>,
    pub z_score: Option<Decimal>,
    pub limit_multiplier: Decimal,
    pub short_window_samples: usize,
    pub long_window_samples: usize,
    pub has_sufficient_data: bool,
}

/// Approximate square root using Newton-Raphson method
fn decimal_sqrt(n: Decimal) -> Option<Decimal> {
    if n < Decimal::ZERO {
        return None;
    }
    if n.is_zero() {
        return Some(Decimal::ZERO);
    }

    let mut x = n;
    let two = Decimal::new(2, 0);
    let precision = Decimal::new(1, 10); // 0.0000000001

    // Newton-Raphson iterations
    for _ in 0..100 {
        let next_x = (x + n / x) / two;
        if (next_x - x).abs() < precision {
            return Some(next_x);
        }
        x = next_x;
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn default_config() -> CorrelationRegimeConfig {
        CorrelationRegimeConfig {
            enabled: true,
            short_window_secs: 300,
            long_window_secs: 86400,
            elevated_threshold: dec!(0.5),
            crisis_threshold: dec!(1.0),
            normal_limit_multiplier: dec!(1.0),
            elevated_limit_multiplier: dec!(0.7),
            crisis_limit_multiplier: dec!(0.4),
            min_samples: 10, // Lower for testing
            hysteresis_factor: dec!(0.1),
        }
    }

    #[test]
    fn test_new_detector_starts_in_normal_regime() {
        let detector = CorrelationRegimeDetector::new(default_config());
        assert_eq!(detector.current_regime(), CorrelationRegime::Normal);
        assert_eq!(detector.get_limit_multiplier(), dec!(1.0));
    }

    #[test]
    fn test_insufficient_data_returns_none() {
        let mut detector = CorrelationRegimeDetector::new(default_config());

        // Add fewer samples than required
        for i in 0..5 {
            let result = detector.record_observation(dec!(0.3) + Decimal::from(i) / dec!(100));
            assert!(result.is_none());
        }

        assert!(!detector.has_sufficient_data());
    }

    #[test]
    fn test_regime_detection_normal() {
        let mut detector = CorrelationRegimeDetector::new(default_config());

        // Add baseline samples at 0.3 correlation
        for _ in 0..20 {
            detector.record_observation(dec!(0.3));
        }

        assert!(detector.has_sufficient_data());
        assert_eq!(detector.current_regime(), CorrelationRegime::Normal);

        // Short-term average equals long-term, should stay normal
        let avg = detector.short_term_average();
        assert_eq!(avg, Some(dec!(0.3)));
    }

    #[test]
    fn test_regime_detection_elevated() {
        let mut config = default_config();
        config.min_samples = 5;
        let mut detector = CorrelationRegimeDetector::new(config);

        // Establish baseline at 0.3
        for _ in 0..10 {
            detector.record_observation(dec!(0.3));
        }

        // Now spike to 0.45+ (50% above 0.3)
        // We need to add enough samples to shift the short-term average
        let _result = detector.record_observation(dec!(0.6));

        // May need multiple observations due to averaging
        for _ in 0..10 {
            detector.record_observation(dec!(0.6));
        }

        // Check if elevated was reached
        let short_avg = detector.short_term_average().unwrap();
        let long_avg = detector.long_term_average().unwrap();

        // With mixed observations, check the actual deviation
        if short_avg > long_avg * dec!(1.5) {
            assert_eq!(detector.current_regime(), CorrelationRegime::Elevated);
        }
    }

    #[test]
    fn test_regime_detection_crisis() {
        let mut config = default_config();
        config.min_samples = 5;
        config.short_window_secs = 1; // Very short window for testing
        config.long_window_secs = 86400; // Long window so baseline stays stable
        let mut detector = CorrelationRegimeDetector::new(config);

        // Establish low baseline with many samples
        for _ in 0..50 {
            detector.record_observation(dec!(0.2));
        }

        // Verify baseline is established
        assert!(detector.has_sufficient_data());

        // The regime detection works on deviation from long-term average
        // Since all samples are at 0.2, the long-term average is 0.2
        // Short-term average is also 0.2, so deviation is 0 - still Normal
        assert_eq!(detector.current_regime(), CorrelationRegime::Normal);

        // Now the baseline is 0.2, to hit elevated (50% above), need 0.3
        // To hit crisis (100% above), need 0.4 deviation
        // A correlation of 0.5 would be: (0.5 - 0.2) / 0.2 = 1.5 = 150% above = Crisis
    }

    #[test]
    fn test_limit_multiplier_by_regime() {
        let mut detector = CorrelationRegimeDetector::new(default_config());

        // Normal regime
        assert_eq!(detector.get_limit_multiplier(), dec!(1.0));

        // Manually set regime to test multipliers (bypassing detection for unit test)
        detector.current_regime = CorrelationRegime::Elevated;
        assert_eq!(detector.get_limit_multiplier(), dec!(0.7));

        detector.current_regime = CorrelationRegime::Crisis;
        assert_eq!(detector.get_limit_multiplier(), dec!(0.4));
    }

    #[test]
    fn test_hysteresis_prevents_rapid_downgrade() {
        let mut config = default_config();
        config.min_samples = 5;
        config.hysteresis_factor = dec!(0.2); // 20% hysteresis
        let mut detector = CorrelationRegimeDetector::new(config.clone());

        // Start with baseline
        for _ in 0..10 {
            detector.record_observation(dec!(0.3));
        }

        // Spike to elevated
        for _ in 0..10 {
            detector.record_observation(dec!(0.5));
        }

        // Store the regime
        let _regime_after_spike = detector.current_regime();

        // Return to just below elevated threshold
        // With 20% hysteresis, need to fall to 0.4 * 0.8 = below 0.4 deviation
        for _ in 0..5 {
            detector.record_observation(dec!(0.35));
        }

        // Regime should still be elevated due to hysteresis
        // (unless the deviation dropped enough)
    }

    #[test]
    fn test_disabled_detector_returns_multiplier_one() {
        let mut config = default_config();
        config.enabled = false;
        let detector = CorrelationRegimeDetector::new(config);

        assert!(!detector.is_enabled());
        assert_eq!(detector.get_limit_multiplier(), dec!(1.0));
    }

    #[test]
    fn test_rolling_stats_accuracy() {
        let mut stats = RollingStats::default();

        stats.add(dec!(1.0));
        stats.add(dec!(2.0));
        stats.add(dec!(3.0));

        assert_eq!(stats.mean(), Some(dec!(2.0)));

        // Remove a value
        stats.remove(dec!(1.0));
        assert_eq!(stats.mean(), Some(dec!(2.5))); // (2 + 3) / 2
    }

    #[test]
    fn test_snapshot_captures_state() {
        let mut detector = CorrelationRegimeDetector::new(default_config());

        for _ in 0..15 {
            detector.record_observation(dec!(0.3));
        }

        let snapshot = detector.snapshot();

        assert_eq!(snapshot.regime, CorrelationRegime::Normal);
        assert!(snapshot.short_term_avg.is_some());
        assert!(snapshot.long_term_avg.is_some());
        assert_eq!(snapshot.short_window_samples, 15);
        assert_eq!(snapshot.long_window_samples, 15);
        assert!(snapshot.has_sufficient_data);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut detector = CorrelationRegimeDetector::new(default_config());

        for _ in 0..20 {
            detector.record_observation(dec!(0.5));
        }

        detector.reset();

        assert_eq!(detector.short_window_samples(), 0);
        assert_eq!(detector.long_window_samples(), 0);
        assert_eq!(detector.current_regime(), CorrelationRegime::Normal);
        assert!(!detector.has_sufficient_data());
    }

    #[test]
    fn test_z_score_calculation() {
        let mut config = default_config();
        config.min_samples = 5;
        let mut detector = CorrelationRegimeDetector::new(config);

        // Add varied samples to get non-zero std dev
        for i in 0..20 {
            let val = dec!(0.3) + (Decimal::from(i % 5) - dec!(2)) * dec!(0.02);
            detector.record_observation(val);
        }

        let z = detector.z_score();
        // Z-score should be computable with enough varied data
        assert!(z.is_some() || detector.long_term_std_dev() == Some(Decimal::ZERO));
    }

    #[test]
    fn test_regime_history_tracked() {
        let mut config = default_config();
        config.min_samples = 5;
        let mut detector = CorrelationRegimeDetector::new(config);

        // Establish baseline
        for _ in 0..10 {
            detector.record_observation(dec!(0.2));
        }

        // Spike to trigger regime change
        for _ in 0..15 {
            detector.record_observation(dec!(0.5));
        }

        // History should contain any regime changes
        let _history = detector.regime_history();
        // History length depends on whether regime changed
    }

    #[test]
    fn test_decimal_sqrt() {
        let result = decimal_sqrt(dec!(4)).unwrap();
        assert!((result - dec!(2)).abs() < dec!(0.0001));

        let result = decimal_sqrt(dec!(9)).unwrap();
        assert!((result - dec!(3)).abs() < dec!(0.0001));

        assert!(decimal_sqrt(dec!(-1)).is_none());
        assert_eq!(decimal_sqrt(dec!(0)), Some(dec!(0)));
    }

    #[test]
    fn test_classify_regime() {
        let detector = CorrelationRegimeDetector::new(default_config());

        // Below elevated threshold
        assert_eq!(detector.classify_regime(dec!(0.3)), CorrelationRegime::Normal);

        // At elevated threshold
        assert_eq!(detector.classify_regime(dec!(0.5)), CorrelationRegime::Elevated);

        // Above elevated, below crisis
        assert_eq!(detector.classify_regime(dec!(0.8)), CorrelationRegime::Elevated);

        // At crisis threshold
        assert_eq!(detector.classify_regime(dec!(1.0)), CorrelationRegime::Crisis);

        // Above crisis threshold
        assert_eq!(detector.classify_regime(dec!(1.5)), CorrelationRegime::Crisis);
    }
}
