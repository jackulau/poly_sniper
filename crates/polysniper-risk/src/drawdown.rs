//! Drawdown-triggered position scaling
//!
//! Calculates portfolio drawdown from high-water mark and adjusts position
//! sizes accordingly - progressively reducing exposure as drawdown deepens
//! to protect capital during adverse market conditions.

use polysniper_core::DrawdownConfig;
use rust_decimal::Decimal;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;
use tracing::debug;

/// Tracks portfolio high-water mark and calculates drawdown-based size scaling
#[derive(Debug)]
pub struct DrawdownCalculator {
    config: DrawdownConfig,
    /// Peak portfolio value (high-water mark)
    peak_value: RwLock<Decimal>,
    /// Current portfolio value
    current_value: RwLock<Decimal>,
    /// Last applied multiplier (for hysteresis tracking)
    last_multiplier: RwLock<Decimal>,
    /// Flag to track if we're in recovery mode (for hysteresis)
    in_recovery: AtomicBool,
}

impl DrawdownCalculator {
    /// Create a new drawdown calculator with the given config
    pub fn new(config: DrawdownConfig) -> Self {
        Self {
            config,
            peak_value: RwLock::new(Decimal::ZERO),
            current_value: RwLock::new(Decimal::ZERO),
            last_multiplier: RwLock::new(Decimal::ONE),
            in_recovery: AtomicBool::new(false),
        }
    }

    /// Initialize or update the portfolio value
    ///
    /// This should be called whenever portfolio value changes (after fills, mark-to-market, etc.)
    pub fn update_equity(&self, new_value: Decimal) {
        let mut current = self.current_value.write().unwrap();
        let mut peak = self.peak_value.write().unwrap();

        *current = new_value;

        // Update high-water mark if we have a new peak
        if new_value > *peak {
            *peak = new_value;
            // If we hit a new high, we're definitely not in recovery anymore
            self.in_recovery.store(false, Ordering::SeqCst);
            debug!(
                new_peak = %new_value,
                "New high-water mark"
            );
        }
    }

    /// Set the initial peak value (e.g., loaded from persistence)
    pub fn set_peak_value(&self, peak: Decimal) {
        let mut peak_value = self.peak_value.write().unwrap();
        *peak_value = peak;
    }

    /// Get current peak value
    pub fn get_peak_value(&self) -> Decimal {
        *self.peak_value.read().unwrap()
    }

    /// Get current portfolio value
    pub fn get_current_value(&self) -> Decimal {
        *self.current_value.read().unwrap()
    }

    /// Calculate the current drawdown percentage from peak
    ///
    /// Returns a positive percentage (e.g., 10.0 for 10% drawdown)
    pub fn calculate_drawdown_pct(&self) -> Decimal {
        let current = *self.current_value.read().unwrap();
        let peak = *self.peak_value.read().unwrap();

        if peak.is_zero() || current >= peak {
            return Decimal::ZERO;
        }

        ((peak - current) / peak) * Decimal::ONE_HUNDRED
    }

    /// Calculate the size multiplier based on current drawdown
    ///
    /// Uses linear interpolation between tiers for smooth scaling:
    /// - 0% to tier_1_threshold: 100% size
    /// - tier_1_threshold to tier_2_threshold: linear from tier_1_multiplier to tier_2_multiplier
    /// - tier_2_threshold to tier_3_threshold: linear from tier_2_multiplier to tier_3_multiplier
    /// - tier_3_threshold to max_drawdown: linear from tier_3_multiplier to min_multiplier
    /// - Above max_drawdown: min_multiplier
    pub fn calculate_size_multiplier(&self, drawdown_pct: Decimal) -> Decimal {
        if !self.config.enabled {
            return Decimal::ONE;
        }

        // Check for recovery hysteresis
        let last_mult = *self.last_multiplier.read().unwrap();
        let in_recovery = self.in_recovery.load(Ordering::SeqCst);

        // Calculate raw multiplier based on drawdown
        let raw_multiplier = self.calculate_raw_multiplier(drawdown_pct);

        // Apply hysteresis during recovery
        let final_multiplier = if in_recovery && raw_multiplier > last_mult {
            // We're recovering - only increase if we've recovered by the buffer amount
            let effective_drawdown = self.calculate_effective_drawdown_for_recovery(drawdown_pct);
            let recovery_multiplier = self.calculate_raw_multiplier(effective_drawdown);

            if recovery_multiplier > last_mult {
                // Recovery threshold met, can increase multiplier
                self.in_recovery.store(false, Ordering::SeqCst);
                raw_multiplier
            } else {
                // Still in hysteresis zone, keep last multiplier
                last_mult
            }
        } else {
            // Going down or not in recovery mode
            if raw_multiplier < last_mult {
                // Drawdown is increasing, mark as in recovery for next time
                self.in_recovery.store(true, Ordering::SeqCst);
            }
            raw_multiplier
        };

        // Update last multiplier
        *self.last_multiplier.write().unwrap() = final_multiplier;

        debug!(
            drawdown_pct = %drawdown_pct,
            raw_multiplier = %raw_multiplier,
            final_multiplier = %final_multiplier,
            in_recovery = %in_recovery,
            "Calculated drawdown size multiplier"
        );

        final_multiplier
    }

    /// Calculate raw multiplier without hysteresis
    fn calculate_raw_multiplier(&self, drawdown_pct: Decimal) -> Decimal {
        if drawdown_pct <= Decimal::ZERO {
            return Decimal::ONE;
        }

        // Above max drawdown - use minimum
        if drawdown_pct >= self.config.max_drawdown_pct {
            return self.config.min_multiplier;
        }

        // Below tier 1 threshold - full size
        if drawdown_pct < self.config.tier_1_threshold_pct {
            return Decimal::ONE;
        }

        // Tier 1: between tier_1_threshold and tier_2_threshold
        if drawdown_pct < self.config.tier_2_threshold_pct {
            return self.interpolate(
                drawdown_pct,
                self.config.tier_1_threshold_pct,
                self.config.tier_2_threshold_pct,
                Decimal::ONE,
                self.config.tier_1_multiplier,
            );
        }

        // Tier 2: between tier_2_threshold and tier_3_threshold
        if drawdown_pct < self.config.tier_3_threshold_pct {
            return self.interpolate(
                drawdown_pct,
                self.config.tier_2_threshold_pct,
                self.config.tier_3_threshold_pct,
                self.config.tier_1_multiplier,
                self.config.tier_2_multiplier,
            );
        }

        // Tier 3: between tier_3_threshold and max_drawdown
        self.interpolate(
            drawdown_pct,
            self.config.tier_3_threshold_pct,
            self.config.max_drawdown_pct,
            self.config.tier_2_multiplier,
            self.config.tier_3_multiplier,
        )
    }

    /// Linear interpolation between two points
    fn interpolate(
        &self,
        value: Decimal,
        start_x: Decimal,
        end_x: Decimal,
        start_y: Decimal,
        end_y: Decimal,
    ) -> Decimal {
        if end_x == start_x {
            return start_y;
        }

        let progress = (value - start_x) / (end_x - start_x);
        start_y + (end_y - start_y) * progress
    }

    /// Calculate effective drawdown for recovery hysteresis
    ///
    /// Adds recovery buffer to current drawdown to prevent rapid oscillation
    fn calculate_effective_drawdown_for_recovery(&self, current_drawdown: Decimal) -> Decimal {
        // To increase the multiplier, we need to be recovery_buffer_pct below
        // the threshold for the next tier up
        (current_drawdown + self.config.recovery_buffer_pct).max(Decimal::ZERO)
    }

    /// Get the size multiplier for the current portfolio state
    pub fn get_current_multiplier(&self) -> Decimal {
        let drawdown = self.calculate_drawdown_pct();
        self.calculate_size_multiplier(drawdown)
    }

    /// Check if drawdown scaling is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Check if we're at or above max drawdown
    pub fn is_max_drawdown(&self) -> bool {
        self.calculate_drawdown_pct() >= self.config.max_drawdown_pct
    }

    /// Get a status summary for logging/metrics
    pub fn status_summary(&self) -> DrawdownStatus {
        let current_value = self.get_current_value();
        let peak_value = self.get_peak_value();
        let drawdown_pct = self.calculate_drawdown_pct();
        let multiplier = self.get_current_multiplier();

        DrawdownStatus {
            current_value,
            peak_value,
            drawdown_pct,
            multiplier,
            is_max_drawdown: self.is_max_drawdown(),
        }
    }
}

/// Status summary for drawdown state
#[derive(Debug, Clone)]
pub struct DrawdownStatus {
    pub current_value: Decimal,
    pub peak_value: Decimal,
    pub drawdown_pct: Decimal,
    pub multiplier: Decimal,
    pub is_max_drawdown: bool,
}

impl std::fmt::Display for DrawdownStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Drawdown: {:.2}% (${:.2} from peak ${:.2}), multiplier: {:.2}{}",
            self.drawdown_pct,
            self.current_value,
            self.peak_value,
            self.multiplier,
            if self.is_max_drawdown { " [MAX]" } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn default_config() -> DrawdownConfig {
        DrawdownConfig {
            enabled: true,
            tier_1_threshold_pct: dec!(5),
            tier_1_multiplier: dec!(0.75),
            tier_2_threshold_pct: dec!(10),
            tier_2_multiplier: dec!(0.50),
            tier_3_threshold_pct: dec!(20),
            tier_3_multiplier: dec!(0.25),
            max_drawdown_pct: dec!(30),
            min_multiplier: dec!(0.10),
            recovery_buffer_pct: dec!(2),
        }
    }

    #[test]
    fn test_zero_drawdown() {
        let calc = DrawdownCalculator::new(default_config());
        calc.update_equity(dec!(10000));

        let drawdown = calc.calculate_drawdown_pct();
        assert_eq!(drawdown, Decimal::ZERO);

        let multiplier = calc.get_current_multiplier();
        assert_eq!(multiplier, Decimal::ONE);
    }

    #[test]
    fn test_small_drawdown_below_tier1() {
        let calc = DrawdownCalculator::new(default_config());
        calc.update_equity(dec!(10000)); // Set peak
        calc.update_equity(dec!(9700)); // 3% drawdown

        let drawdown = calc.calculate_drawdown_pct();
        assert_eq!(drawdown, dec!(3));

        let multiplier = calc.get_current_multiplier();
        assert_eq!(multiplier, Decimal::ONE);
    }

    #[test]
    fn test_exactly_at_tier1_threshold() {
        let calc = DrawdownCalculator::new(default_config());
        calc.update_equity(dec!(10000));
        calc.update_equity(dec!(9500)); // Exactly 5% drawdown

        let drawdown = calc.calculate_drawdown_pct();
        assert_eq!(drawdown, dec!(5));

        let multiplier = calc.get_current_multiplier();
        // At exactly tier 1 threshold, should start interpolating toward tier_1_multiplier
        // At 5% (start of tier 1), multiplier should be 1.0
        assert_eq!(multiplier, Decimal::ONE);
    }

    #[test]
    fn test_mid_tier1_drawdown() {
        let calc = DrawdownCalculator::new(default_config());
        calc.update_equity(dec!(10000));
        calc.update_equity(dec!(9250)); // 7.5% drawdown (midpoint of tier 1: 5-10%)

        let drawdown = calc.calculate_drawdown_pct();
        assert_eq!(drawdown, dec!(7.5));

        let multiplier = calc.get_current_multiplier();
        // Interpolation: at 50% through tier 1, should be 0.875 (midpoint between 1.0 and 0.75)
        assert_eq!(multiplier, dec!(0.875));
    }

    #[test]
    fn test_at_tier2_threshold() {
        let calc = DrawdownCalculator::new(default_config());
        calc.update_equity(dec!(10000));
        calc.update_equity(dec!(9000)); // 10% drawdown

        let drawdown = calc.calculate_drawdown_pct();
        assert_eq!(drawdown, dec!(10));

        let multiplier = calc.get_current_multiplier();
        // At tier 2 threshold, multiplier should be tier_1_multiplier (0.75)
        assert_eq!(multiplier, dec!(0.75));
    }

    #[test]
    fn test_mid_tier2_drawdown() {
        let calc = DrawdownCalculator::new(default_config());
        calc.update_equity(dec!(10000));
        calc.update_equity(dec!(8500)); // 15% drawdown (midpoint of tier 2: 10-20%)

        let drawdown = calc.calculate_drawdown_pct();
        assert_eq!(drawdown, dec!(15));

        let multiplier = calc.get_current_multiplier();
        // Interpolation: at 50% through tier 2, should be 0.625 (midpoint between 0.75 and 0.50)
        assert_eq!(multiplier, dec!(0.625));
    }

    #[test]
    fn test_at_tier3_threshold() {
        let calc = DrawdownCalculator::new(default_config());
        calc.update_equity(dec!(10000));
        calc.update_equity(dec!(8000)); // 20% drawdown

        let drawdown = calc.calculate_drawdown_pct();
        assert_eq!(drawdown, dec!(20));

        let multiplier = calc.get_current_multiplier();
        // At tier 3 threshold, should be tier_2_multiplier (0.50)
        assert_eq!(multiplier, dec!(0.50));
    }

    #[test]
    fn test_mid_tier3_drawdown() {
        let calc = DrawdownCalculator::new(default_config());
        calc.update_equity(dec!(10000));
        calc.update_equity(dec!(7500)); // 25% drawdown (midpoint of tier 3: 20-30%)

        let drawdown = calc.calculate_drawdown_pct();
        assert_eq!(drawdown, dec!(25));

        let multiplier = calc.get_current_multiplier();
        // Interpolation: at 50% through tier 3, should be 0.375 (midpoint between 0.50 and 0.25)
        assert_eq!(multiplier, dec!(0.375));
    }

    #[test]
    fn test_at_max_drawdown() {
        let calc = DrawdownCalculator::new(default_config());
        calc.update_equity(dec!(10000));
        calc.update_equity(dec!(7000)); // 30% drawdown

        let drawdown = calc.calculate_drawdown_pct();
        assert_eq!(drawdown, dec!(30));

        let multiplier = calc.get_current_multiplier();
        // At max drawdown, should be min_multiplier
        assert_eq!(multiplier, dec!(0.10));
    }

    #[test]
    fn test_beyond_max_drawdown() {
        let calc = DrawdownCalculator::new(default_config());
        calc.update_equity(dec!(10000));
        calc.update_equity(dec!(5000)); // 50% drawdown

        let drawdown = calc.calculate_drawdown_pct();
        assert_eq!(drawdown, dec!(50));

        let multiplier = calc.get_current_multiplier();
        // Beyond max drawdown, should stay at min_multiplier
        assert_eq!(multiplier, dec!(0.10));

        assert!(calc.is_max_drawdown());
    }

    #[test]
    fn test_high_water_mark_updates() {
        let calc = DrawdownCalculator::new(default_config());
        calc.update_equity(dec!(10000));
        assert_eq!(calc.get_peak_value(), dec!(10000));

        // New high
        calc.update_equity(dec!(11000));
        assert_eq!(calc.get_peak_value(), dec!(11000));

        // Drawdown
        calc.update_equity(dec!(10500));
        assert_eq!(calc.get_peak_value(), dec!(11000)); // Peak unchanged

        // Another new high
        calc.update_equity(dec!(12000));
        assert_eq!(calc.get_peak_value(), dec!(12000));
    }

    #[test]
    fn test_drawdown_disabled() {
        let mut config = default_config();
        config.enabled = false;
        let calc = DrawdownCalculator::new(config);

        calc.update_equity(dec!(10000));
        calc.update_equity(dec!(5000)); // 50% drawdown

        let multiplier = calc.get_current_multiplier();
        // When disabled, always return 1.0
        assert_eq!(multiplier, Decimal::ONE);
    }

    #[test]
    fn test_set_peak_value() {
        let calc = DrawdownCalculator::new(default_config());
        calc.set_peak_value(dec!(15000));
        calc.update_equity(dec!(12000));

        let drawdown = calc.calculate_drawdown_pct();
        assert_eq!(drawdown, dec!(20)); // 3000/15000 = 20%
    }

    #[test]
    fn test_recovery_hysteresis() {
        let calc = DrawdownCalculator::new(default_config());

        // Start at peak
        calc.update_equity(dec!(10000));

        // Drop to 15% drawdown
        calc.update_equity(dec!(8500));
        let mult1 = calc.get_current_multiplier();
        assert_eq!(mult1, dec!(0.625)); // Mid tier 2

        // Recover slightly to 14% drawdown
        calc.update_equity(dec!(8600));
        let mult2 = calc.get_current_multiplier();
        // Due to hysteresis, should not immediately improve
        // Recovery needs to be more than recovery_buffer_pct (2%)
        assert_eq!(mult2, dec!(0.625)); // Still at previous level

        // Recover more significantly (to 10% drawdown, which with 2% buffer = 12%)
        calc.update_equity(dec!(9000));
        let mult3 = calc.get_current_multiplier();
        // At 10% drawdown + 2% buffer = 12%, interpolate at tier 2
        // Should be 0.75 (at threshold, which triggers recovery)
        assert_eq!(mult3, dec!(0.75));
    }

    #[test]
    fn test_status_summary() {
        let calc = DrawdownCalculator::new(default_config());
        calc.update_equity(dec!(10000));
        calc.update_equity(dec!(8500));

        let status = calc.status_summary();
        assert_eq!(status.current_value, dec!(8500));
        assert_eq!(status.peak_value, dec!(10000));
        assert_eq!(status.drawdown_pct, dec!(15));
        assert_eq!(status.multiplier, dec!(0.625));
        assert!(!status.is_max_drawdown);
    }

    #[test]
    fn test_zero_peak_value() {
        let calc = DrawdownCalculator::new(default_config());
        // Don't set any value - peak is zero

        let drawdown = calc.calculate_drawdown_pct();
        assert_eq!(drawdown, Decimal::ZERO);

        let multiplier = calc.get_current_multiplier();
        assert_eq!(multiplier, Decimal::ONE);
    }

    #[test]
    fn test_current_above_peak() {
        let calc = DrawdownCalculator::new(default_config());
        calc.set_peak_value(dec!(10000));
        calc.update_equity(dec!(11000)); // Above peak

        let drawdown = calc.calculate_drawdown_pct();
        assert_eq!(drawdown, Decimal::ZERO);

        let multiplier = calc.get_current_multiplier();
        assert_eq!(multiplier, Decimal::ONE);
    }
}
