//! Volatility calculation and position sizing adjustment
//!
//! Calculates rolling volatility from price history and adjusts position
//! sizes accordingly - reducing size in high-volatility markets and
//! potentially increasing it in stable markets.

use chrono::{DateTime, Utc};
use polysniper_core::VolatilityConfig;
use rust_decimal::Decimal;
use tracing::{debug, info};

/// Volatility calculator for market price data
#[derive(Debug, Clone)]
pub struct VolatilityCalculator {
    config: VolatilityConfig,
}

impl VolatilityCalculator {
    /// Create a new volatility calculator with the given config
    pub fn new(config: VolatilityConfig) -> Self {
        Self { config }
    }

    /// Calculate rolling standard deviation of returns from price history
    ///
    /// Returns the volatility as a percentage (e.g., 5.0 for 5% volatility)
    pub fn calculate_volatility(
        &self,
        price_history: &[(DateTime<Utc>, Decimal)],
    ) -> Option<Decimal> {
        // Filter to prices within the window
        let cutoff = Utc::now() - chrono::Duration::seconds(self.config.window_secs as i64);
        let recent_prices: Vec<Decimal> = price_history
            .iter()
            .filter(|(ts, _)| *ts >= cutoff)
            .map(|(_, price)| *price)
            .collect();

        self.calculate_stddev_pct(&recent_prices)
    }

    /// Calculate standard deviation of percentage returns
    fn calculate_stddev_pct(&self, prices: &[Decimal]) -> Option<Decimal> {
        if prices.len() < 2 {
            return None;
        }

        // Calculate percentage returns
        let returns: Vec<Decimal> = prices
            .windows(2)
            .filter_map(|w| {
                let prev = w[0];
                let curr = w[1];
                if prev.is_zero() {
                    None
                } else {
                    Some(((curr - prev) / prev) * Decimal::ONE_HUNDRED)
                }
            })
            .collect();

        if returns.is_empty() {
            return None;
        }

        // Calculate mean
        let n = Decimal::from(returns.len());
        let sum: Decimal = returns.iter().copied().sum();
        let mean = sum / n;

        // Calculate variance
        let variance_sum: Decimal = returns.iter().map(|r| (*r - mean) * (*r - mean)).sum();

        // Use sample variance (n-1) for better estimation
        let divisor = if returns.len() > 1 {
            Decimal::from(returns.len() - 1)
        } else {
            Decimal::ONE
        };
        let variance = variance_sum / divisor;

        // Calculate standard deviation using Newton-Raphson approximation
        let stddev = self.sqrt_decimal(variance)?;

        Some(stddev)
    }

    /// Calculate square root using Newton-Raphson method
    fn sqrt_decimal(&self, value: Decimal) -> Option<Decimal> {
        if value.is_zero() {
            return Some(Decimal::ZERO);
        }
        if value.is_sign_negative() {
            return None;
        }

        // Initial guess
        let mut guess = value / Decimal::TWO;
        if guess.is_zero() {
            guess = Decimal::new(1, 10); // Very small number for small inputs
        }

        // Newton-Raphson iterations
        for _ in 0..20 {
            let next_guess = (guess + value / guess) / Decimal::TWO;
            if (next_guess - guess).abs() < Decimal::new(1, 10) {
                return Some(next_guess);
            }
            guess = next_guess;
        }

        Some(guess)
    }

    /// Calculate the size multiplier based on current volatility
    ///
    /// Higher volatility -> lower multiplier (smaller positions)
    /// Lower volatility -> higher multiplier (larger positions)
    pub fn calculate_size_multiplier(&self, current_volatility_pct: Decimal) -> Decimal {
        if !self.config.enabled {
            return Decimal::ONE;
        }

        if current_volatility_pct.is_zero() {
            // No volatility data, use max multiplier
            return self.config.max_size_multiplier;
        }

        // multiplier = base_volatility / current_volatility
        // If volatility is 2x base, multiplier = 0.5 (half size)
        // If volatility is 0.5x base, multiplier = 2.0 (double size, clamped to max)
        let raw_multiplier = self.config.base_volatility_pct / current_volatility_pct;

        // Clamp to configured bounds
        let clamped = raw_multiplier
            .max(self.config.min_size_multiplier)
            .min(self.config.max_size_multiplier);

        debug!(
            current_vol = %current_volatility_pct,
            base_vol = %self.config.base_volatility_pct,
            raw_mult = %raw_multiplier,
            clamped_mult = %clamped,
            "Calculated volatility size multiplier"
        );

        clamped
    }

    /// Calculate the adjusted position size based on volatility
    ///
    /// Returns (adjusted_size, multiplier_used, volatility_pct)
    pub fn adjust_size(
        &self,
        original_size: Decimal,
        price_history: &[(DateTime<Utc>, Decimal)],
    ) -> (Decimal, Decimal, Option<Decimal>) {
        if !self.config.enabled {
            return (original_size, Decimal::ONE, None);
        }

        let volatility = self.calculate_volatility(price_history);
        let multiplier = match volatility {
            Some(vol) => self.calculate_size_multiplier(vol),
            None => Decimal::ONE, // No history, no adjustment
        };

        let adjusted = original_size * multiplier;

        if multiplier != Decimal::ONE {
            info!(
                original_size = %original_size,
                adjusted_size = %adjusted,
                multiplier = %multiplier,
                volatility = ?volatility,
                "Volatility-adjusted position size"
            );
        }

        (adjusted, multiplier, volatility)
    }

    /// Check if volatility adjustment is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use rust_decimal::prelude::FromPrimitive;

    fn default_config() -> VolatilityConfig {
        VolatilityConfig {
            enabled: true,
            window_secs: 300,
            base_volatility_pct: Decimal::new(5, 0),  // 5%
            min_size_multiplier: Decimal::new(25, 2), // 0.25
            max_size_multiplier: Decimal::new(15, 1), // 1.5
        }
    }

    fn make_price_history(prices: &[f64]) -> Vec<(DateTime<Utc>, Decimal)> {
        let now = Utc::now();
        prices
            .iter()
            .enumerate()
            .map(|(i, p)| {
                (
                    now - Duration::seconds((prices.len() - 1 - i) as i64 * 10),
                    Decimal::from_f64(*p).unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn test_volatility_calculation_stable_market() {
        let calc = VolatilityCalculator::new(default_config());
        // Prices that are very stable (small movements)
        let history = make_price_history(&[0.50, 0.51, 0.50, 0.51, 0.50, 0.51]);

        let vol = calc.calculate_volatility(&history).unwrap();
        // Volatility should be low (around 2%)
        assert!(
            vol < Decimal::new(5, 0),
            "Expected low volatility, got {}",
            vol
        );
    }

    #[test]
    fn test_volatility_calculation_volatile_market() {
        let calc = VolatilityCalculator::new(default_config());
        // Prices with high volatility
        let history = make_price_history(&[0.30, 0.50, 0.35, 0.55, 0.40, 0.60]);

        let vol = calc.calculate_volatility(&history).unwrap();
        // Volatility should be high
        assert!(
            vol > Decimal::new(10, 0),
            "Expected high volatility, got {}",
            vol
        );
    }

    #[test]
    fn test_volatility_empty_history() {
        let calc = VolatilityCalculator::new(default_config());
        let history: Vec<(DateTime<Utc>, Decimal)> = vec![];

        let vol = calc.calculate_volatility(&history);
        assert!(vol.is_none());
    }

    #[test]
    fn test_volatility_single_price() {
        let calc = VolatilityCalculator::new(default_config());
        let history = make_price_history(&[0.50]);

        let vol = calc.calculate_volatility(&history);
        assert!(vol.is_none());
    }

    #[test]
    fn test_size_multiplier_high_volatility() {
        let calc = VolatilityCalculator::new(default_config());
        // 10% volatility (2x base) -> multiplier should be ~0.5
        let multiplier = calc.calculate_size_multiplier(Decimal::new(10, 0));
        assert_eq!(multiplier, Decimal::new(5, 1)); // 0.5
    }

    #[test]
    fn test_size_multiplier_low_volatility() {
        let calc = VolatilityCalculator::new(default_config());
        // 2.5% volatility (0.5x base) -> multiplier should be 2.0, clamped to 1.5
        let multiplier = calc.calculate_size_multiplier(Decimal::new(25, 1));
        assert_eq!(multiplier, Decimal::new(15, 1)); // 1.5 (clamped)
    }

    #[test]
    fn test_size_multiplier_baseline_volatility() {
        let calc = VolatilityCalculator::new(default_config());
        // Exactly at base volatility -> multiplier should be 1.0
        let multiplier = calc.calculate_size_multiplier(Decimal::new(5, 0));
        assert_eq!(multiplier, Decimal::ONE);
    }

    #[test]
    fn test_size_multiplier_extreme_volatility() {
        let calc = VolatilityCalculator::new(default_config());
        // 50% volatility (10x base) -> multiplier should be 0.1, clamped to 0.25
        let multiplier = calc.calculate_size_multiplier(Decimal::new(50, 0));
        assert_eq!(multiplier, Decimal::new(25, 2)); // 0.25 (clamped to min)
    }

    #[test]
    fn test_size_multiplier_zero_volatility() {
        let calc = VolatilityCalculator::new(default_config());
        // Zero volatility -> max multiplier
        let multiplier = calc.calculate_size_multiplier(Decimal::ZERO);
        assert_eq!(multiplier, Decimal::new(15, 1)); // 1.5 (max)
    }

    #[test]
    fn test_adjust_size_high_volatility() {
        let calc = VolatilityCalculator::new(default_config());
        let original_size = Decimal::new(100, 0);

        // Create volatile price history
        let history = make_price_history(&[0.30, 0.50, 0.35, 0.55, 0.40, 0.60]);

        let (adjusted, multiplier, vol) = calc.adjust_size(original_size, &history);

        // Size should be reduced due to high volatility
        assert!(adjusted < original_size);
        assert!(multiplier < Decimal::ONE);
        assert!(vol.is_some());
    }

    #[test]
    fn test_adjust_size_disabled() {
        let mut config = default_config();
        config.enabled = false;
        let calc = VolatilityCalculator::new(config);

        let original_size = Decimal::new(100, 0);
        let history = make_price_history(&[0.30, 0.50, 0.35, 0.55, 0.40, 0.60]);

        let (adjusted, multiplier, vol) = calc.adjust_size(original_size, &history);

        // No adjustment when disabled
        assert_eq!(adjusted, original_size);
        assert_eq!(multiplier, Decimal::ONE);
        assert!(vol.is_none());
    }

    #[test]
    fn test_stddev_calculation_known_values() {
        let calc = VolatilityCalculator::new(default_config());

        // Prices: 100, 102, 98, 104, 96
        // Returns: +2%, -3.92%, +6.12%, -7.69%
        let prices = vec![
            Decimal::new(100, 0),
            Decimal::new(102, 0),
            Decimal::new(98, 0),
            Decimal::new(104, 0),
            Decimal::new(96, 0),
        ];

        let stddev = calc.calculate_stddev_pct(&prices).unwrap();

        // Should be around 5-6% standard deviation
        assert!(stddev > Decimal::new(4, 0), "Stddev too low: {}", stddev);
        assert!(stddev < Decimal::new(8, 0), "Stddev too high: {}", stddev);
    }

    #[test]
    fn test_sqrt_decimal() {
        let calc = VolatilityCalculator::new(default_config());

        // sqrt(4) = 2
        let result = calc.sqrt_decimal(Decimal::new(4, 0)).unwrap();
        assert!((result - Decimal::TWO).abs() < Decimal::new(1, 8));

        // sqrt(9) = 3
        let result = calc.sqrt_decimal(Decimal::new(9, 0)).unwrap();
        assert!((result - Decimal::new(3, 0)).abs() < Decimal::new(1, 8));

        // sqrt(0) = 0
        let result = calc.sqrt_decimal(Decimal::ZERO).unwrap();
        assert_eq!(result, Decimal::ZERO);
    }
}
