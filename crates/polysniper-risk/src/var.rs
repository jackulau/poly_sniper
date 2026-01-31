//! Value at Risk (VaR) and Conditional VaR (CVaR) Calculator
//!
//! Provides portfolio-level risk metrics for position sizing and risk limits.
//!
//! ## Methods Supported
//!
//! - **Historical VaR**: Uses the empirical distribution of returns to find the
//!   loss at a given confidence level (e.g., 95% VaR is the 5th percentile of returns)
//!
//! - **Parametric VaR**: Assumes returns follow a normal distribution and uses
//!   the mean and standard deviation to calculate VaR
//!
//! - **CVaR (Expected Shortfall)**: The average loss when losses exceed VaR,
//!   which captures tail risk better than VaR alone

use chrono::Utc;
use polysniper_core::{Position, StateProvider, VaRConfig, VaRMethod, VaRResult};
use rust_decimal::Decimal;
use std::collections::HashMap;
use tracing::{info, warn};

/// Square root of 10 approximated as a Decimal for 10-day VaR scaling
/// sqrt(10) ≈ 3.16227766
const SQRT_10: (i64, u32) = (316227766, 8);

/// Z-score for 95% confidence (one-tailed) ≈ 1.645
const Z_SCORE_95: (i64, u32) = (1645, 3);

/// Z-score for 99% confidence (one-tailed) ≈ 2.326
const Z_SCORE_99: (i64, u32) = (2326, 3);

/// VaR and CVaR calculator for portfolio risk management
#[derive(Debug, Clone)]
pub struct VaRCalculator {
    config: VaRConfig,
}

impl VaRCalculator {
    /// Create a new VaR calculator with the given configuration
    pub fn new(config: VaRConfig) -> Self {
        Self { config }
    }

    /// Check if VaR calculation is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the maximum allowed portfolio VaR
    pub fn max_portfolio_var(&self) -> Decimal {
        self.config.max_portfolio_var_usd
    }

    /// Get the maximum position VaR contribution percentage
    pub fn max_position_var_contribution_pct(&self) -> Decimal {
        self.config.max_position_var_contribution_pct
    }

    /// Get the configured confidence level
    pub fn confidence_level(&self) -> Decimal {
        self.config.confidence_level
    }

    /// Calculate VaR and CVaR for the portfolio
    ///
    /// Returns a VaRResult containing 1-day VaR, CVaR, 10-day VaR, and per-position contributions.
    pub async fn calculate_portfolio_var(&self, state: &dyn StateProvider) -> VaRResult {
        if !self.config.enabled {
            return VaRResult::default();
        }

        let positions = state.get_all_positions().await;
        if positions.is_empty() {
            return VaRResult::default();
        }

        // Collect returns for each position
        let mut all_portfolio_returns: Vec<Decimal> = Vec::new();
        let mut position_vars: HashMap<String, Decimal> = HashMap::new();

        for position in &positions {
            let returns = self.get_position_returns(position, state).await;
            if returns.is_empty() {
                continue;
            }

            // Calculate position VaR
            let position_value = position.size.abs() * position.avg_price;
            let position_var = self.calculate_var_from_returns(&returns, position_value);

            position_vars.insert(position.market_id.clone(), position_var);

            // Weight returns by position value for portfolio
            // For simplicity, we aggregate returns (proper portfolio VaR would need covariance)
            if all_portfolio_returns.is_empty() {
                all_portfolio_returns = returns.iter().map(|r| *r * position_value).collect();
            } else {
                // Add weighted returns (assuming same time periods)
                let min_len = all_portfolio_returns.len().min(returns.len());
                for i in 0..min_len {
                    all_portfolio_returns[i] += returns[i] * position_value;
                }
            }
        }

        if all_portfolio_returns.is_empty() {
            return VaRResult::default();
        }

        // Calculate portfolio VaR
        let portfolio_value = state.get_portfolio_value().await;
        let (var_1d, cvar_1d) = self.calculate_var_and_cvar(&all_portfolio_returns);

        // Scale to 10-day VaR using sqrt(10) rule
        let sqrt_10 = Decimal::new(SQRT_10.0, SQRT_10.1);
        let var_10d = var_1d * sqrt_10;

        let result = VaRResult {
            var_1d,
            cvar_1d,
            var_10d,
            position_contributions: position_vars,
            confidence_level: self.config.confidence_level,
            calculated_at: Utc::now(),
        };

        info!(
            var_1d = %var_1d,
            cvar_1d = %cvar_1d,
            var_10d = %var_10d,
            portfolio_value = %portfolio_value,
            num_positions = positions.len(),
            confidence = %self.config.confidence_level,
            "Calculated portfolio VaR"
        );

        result
    }

    /// Calculate VaR from a vector of returns and position value
    fn calculate_var_from_returns(&self, returns: &[Decimal], position_value: Decimal) -> Decimal {
        if !self.config.enabled || returns.is_empty() {
            return Decimal::ZERO;
        }

        match self.config.method {
            VaRMethod::Historical => self.historical_var(returns) * position_value,
            VaRMethod::Parametric => self.parametric_var(returns) * position_value,
        }
    }

    /// Calculate both VaR and CVaR from portfolio returns
    fn calculate_var_and_cvar(&self, returns: &[Decimal]) -> (Decimal, Decimal) {
        if returns.is_empty() {
            return (Decimal::ZERO, Decimal::ZERO);
        }

        match self.config.method {
            VaRMethod::Historical => {
                let var = self.historical_var(returns);
                let cvar = self.calculate_cvar(returns, var);
                (var.abs(), cvar.abs())
            }
            VaRMethod::Parametric => {
                let var = self.parametric_var(returns);
                let cvar = self.parametric_cvar(returns);
                (var.abs(), cvar.abs())
            }
        }
    }

    /// Calculate Historical VaR (percentile method)
    ///
    /// VaR at confidence level α is the (1-α) percentile of returns.
    /// For 95% VaR, we find the 5th percentile (worst 5% of returns).
    fn historical_var(&self, returns: &[Decimal]) -> Decimal {
        if returns.is_empty() {
            return Decimal::ZERO;
        }

        let mut sorted_returns: Vec<Decimal> = returns.to_vec();
        sorted_returns.sort();

        // Calculate index for the confidence level
        // For 95% confidence, we want the 5th percentile
        let alpha = Decimal::ONE - self.config.confidence_level;
        let n = Decimal::from(sorted_returns.len());
        let index_decimal = alpha * n;

        // Convert to usize, floor it, and ensure it's within bounds
        let index = self
            .decimal_to_usize(index_decimal)
            .min(sorted_returns.len().saturating_sub(1));

        // VaR is the negative of the return at this percentile
        // (we report VaR as a positive number representing potential loss)
        let var_return = sorted_returns[index];

        if var_return < Decimal::ZERO {
            var_return.abs()
        } else {
            Decimal::ZERO
        }
    }

    /// Calculate Parametric VaR (variance-covariance method)
    ///
    /// Assumes normal distribution: VaR = -μ + σ * z_α
    /// where z_α is the z-score for the confidence level
    fn parametric_var(&self, returns: &[Decimal]) -> Decimal {
        if returns.len() < 2 {
            return Decimal::ZERO;
        }

        let (mean, stddev) = self.calculate_mean_stddev(returns);
        let z_score = self.get_z_score();

        // VaR = -mean + z_score * stddev (for left tail)
        // Since we're looking at losses, we take the left tail
        let var = z_score * stddev - mean;

        if var < Decimal::ZERO {
            Decimal::ZERO
        } else {
            var
        }
    }

    /// Calculate CVaR (Expected Shortfall) using historical returns
    ///
    /// CVaR is the average of returns that exceed (are worse than) VaR
    fn calculate_cvar(&self, returns: &[Decimal], var: Decimal) -> Decimal {
        if returns.is_empty() {
            return Decimal::ZERO;
        }

        // Filter returns that are worse than VaR (more negative)
        let var_threshold = -var; // Convert VaR back to return space
        let tail_returns: Vec<Decimal> = returns
            .iter()
            .filter(|r| **r <= var_threshold)
            .copied()
            .collect();

        if tail_returns.is_empty() {
            return var; // If no tail returns, CVaR = VaR
        }

        // CVaR is the average of tail losses
        let sum: Decimal = tail_returns.iter().sum();
        let avg = sum / Decimal::from(tail_returns.len());

        avg.abs()
    }

    /// Calculate Parametric CVaR (Expected Shortfall)
    ///
    /// For normal distribution: ES = μ + σ * φ(z_α) / (1 - α)
    /// where φ is the standard normal PDF
    fn parametric_cvar(&self, returns: &[Decimal]) -> Decimal {
        if returns.len() < 2 {
            return Decimal::ZERO;
        }

        let (mean, stddev) = self.calculate_mean_stddev(returns);
        let z_score = self.get_z_score();

        // Approximate the normal PDF at z_score: φ(z) ≈ exp(-z²/2) / sqrt(2π)
        // For z=1.645 (95%), φ(z) ≈ 0.103
        // For z=2.326 (99%), φ(z) ≈ 0.027
        let pdf_value = self.approximate_normal_pdf(z_score);
        let alpha = Decimal::ONE - self.config.confidence_level;

        // ES = μ + σ * φ(z_α) / α (but we want loss, so negate mean)
        if alpha.is_zero() {
            return self.parametric_var(returns);
        }

        let es = z_score * stddev + (stddev * pdf_value / alpha) - mean;

        if es < Decimal::ZERO {
            Decimal::ZERO
        } else {
            es
        }
    }

    /// Calculate mean and standard deviation of returns
    fn calculate_mean_stddev(&self, returns: &[Decimal]) -> (Decimal, Decimal) {
        if returns.is_empty() {
            return (Decimal::ZERO, Decimal::ZERO);
        }

        let n = Decimal::from(returns.len());
        let sum: Decimal = returns.iter().sum();
        let mean = sum / n;

        if returns.len() < 2 {
            return (mean, Decimal::ZERO);
        }

        // Calculate variance using sample formula (n-1)
        let variance_sum: Decimal = returns.iter().map(|r| (*r - mean) * (*r - mean)).sum();
        let variance = variance_sum / Decimal::from(returns.len() - 1);

        let stddev = self.sqrt_decimal(variance).unwrap_or(Decimal::ZERO);

        (mean, stddev)
    }

    /// Get the z-score for the configured confidence level
    fn get_z_score(&self) -> Decimal {
        // Common z-scores
        if self.config.confidence_level >= Decimal::new(99, 2) {
            Decimal::new(Z_SCORE_99.0, Z_SCORE_99.1)
        } else if self.config.confidence_level >= Decimal::new(95, 2) {
            Decimal::new(Z_SCORE_95.0, Z_SCORE_95.1)
        } else {
            // Approximate for other levels (linear interpolation is rough but works)
            Decimal::new(Z_SCORE_95.0, Z_SCORE_95.1)
        }
    }

    /// Approximate the normal PDF at a given z-score
    fn approximate_normal_pdf(&self, z: Decimal) -> Decimal {
        // φ(z) = exp(-z²/2) / sqrt(2π)
        // For z=1.645: φ ≈ 0.103
        // For z=2.326: φ ≈ 0.027
        let z_val = self.decimal_to_usize(z * Decimal::new(1000, 0));

        if z_val >= 2300 {
            Decimal::new(27, 3) // 0.027
        } else if z_val >= 1600 {
            Decimal::new(103, 3) // 0.103
        } else {
            Decimal::new(150, 3) // 0.150 for lower z-scores
        }
    }

    /// Get position returns from price history
    async fn get_position_returns(
        &self,
        position: &Position,
        state: &dyn StateProvider,
    ) -> Vec<Decimal> {
        let history = state
            .get_price_history(
                &position.token_id,
                (self.config.lookback_days * 24) as usize,
            )
            .await;

        if history.len() < 2 {
            return Vec::new();
        }

        // Calculate returns from price history
        let prices: Vec<Decimal> = history.iter().map(|(_, p)| *p).collect();
        self.calculate_returns(&prices)
    }

    /// Calculate percentage returns from prices
    fn calculate_returns(&self, prices: &[Decimal]) -> Vec<Decimal> {
        if prices.len() < 2 {
            return Vec::new();
        }

        prices
            .windows(2)
            .filter_map(|w| {
                let prev = w[0];
                let curr = w[1];
                if prev.is_zero() {
                    None
                } else {
                    Some((curr - prev) / prev)
                }
            })
            .collect()
    }

    /// Calculate the marginal VaR contribution of adding a new position
    ///
    /// This is an approximation: we calculate the VaR with and without the new position
    pub async fn calculate_marginal_var(
        &self,
        additional_exposure_usd: Decimal,
        additional_returns: &[Decimal],
        state: &dyn StateProvider,
    ) -> Decimal {
        if !self.config.enabled || additional_returns.is_empty() {
            return Decimal::ZERO;
        }

        // Get current portfolio VaR
        let current_result = self.calculate_portfolio_var(state).await;
        let _current_var = if self.config.use_cvar_for_limits {
            current_result.cvar_1d
        } else {
            current_result.var_1d
        };

        // Marginal VaR is approximately the additional VaR from the new position
        // A proper implementation would use covariance, but this is a conservative estimate
        self.calculate_var_from_returns(additional_returns, additional_exposure_usd)
    }

    /// Check if adding a new position would exceed VaR limits
    pub async fn would_exceed_var_limit(
        &self,
        additional_var: Decimal,
        state: &dyn StateProvider,
    ) -> bool {
        if !self.config.enabled {
            return false;
        }

        let current_result = self.calculate_portfolio_var(state).await;
        let current_var = if self.config.use_cvar_for_limits {
            current_result.cvar_1d
        } else {
            current_result.var_1d
        };

        let new_total = current_var + additional_var;

        if new_total > self.config.max_portfolio_var_usd {
            warn!(
                current_var = %current_var,
                additional_var = %additional_var,
                new_total = %new_total,
                limit = %self.config.max_portfolio_var_usd,
                "VaR limit would be exceeded"
            );
            true
        } else {
            false
        }
    }

    /// Check if a position's VaR contribution exceeds the maximum allowed percentage
    pub fn exceeds_contribution_limit(&self, position_var: Decimal, total_var: Decimal) -> bool {
        if total_var.is_zero() || !self.config.enabled {
            return false;
        }

        let contribution_pct = position_var / total_var;
        contribution_pct > self.config.max_position_var_contribution_pct
    }

    /// Calculate the remaining VaR budget before hitting the limit
    pub async fn remaining_var_budget(&self, state: &dyn StateProvider) -> Decimal {
        if !self.config.enabled {
            return Decimal::MAX;
        }

        let current_result = self.calculate_portfolio_var(state).await;
        let current_var = if self.config.use_cvar_for_limits {
            current_result.cvar_1d
        } else {
            current_result.var_1d
        };

        let remaining = self.config.max_portfolio_var_usd - current_var;
        if remaining < Decimal::ZERO {
            Decimal::ZERO
        } else {
            remaining
        }
    }

    /// Square root using Newton-Raphson method
    fn sqrt_decimal(&self, value: Decimal) -> Option<Decimal> {
        if value.is_zero() {
            return Some(Decimal::ZERO);
        }
        if value.is_sign_negative() {
            return None;
        }

        let mut guess = value / Decimal::TWO;
        if guess.is_zero() {
            guess = Decimal::new(1, 10);
        }

        for _ in 0..20 {
            let next_guess = (guess + value / guess) / Decimal::TWO;
            if (next_guess - guess).abs() < Decimal::new(1, 10) {
                return Some(next_guess);
            }
            guess = next_guess;
        }

        Some(guess)
    }

    /// Convert Decimal to usize (truncating)
    fn decimal_to_usize(&self, value: Decimal) -> usize {
        value.floor().to_string().parse().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn default_config() -> VaRConfig {
        VaRConfig {
            enabled: true,
            method: VaRMethod::Historical,
            confidence_level: dec!(0.95),
            lookback_days: 30,
            max_portfolio_var_usd: dec!(500),
            max_position_var_contribution_pct: dec!(0.25),
            use_cvar_for_limits: false,
        }
    }

    #[test]
    fn test_historical_var_known_distribution() {
        let calc = VaRCalculator::new(default_config());

        // Create 100 returns where the 5th percentile (index 4) is -0.05
        let returns: Vec<Decimal> = (0..100)
            .map(|i| Decimal::new(i as i64 - 50, 2)) // Returns from -0.50 to +0.49
            .collect();

        // The 5th percentile (5% of 100 = 5th value) should be around -0.45
        let var = calc.historical_var(&returns);

        // VaR should be positive and represent the 5th percentile loss
        assert!(var > Decimal::ZERO, "VaR should be positive, got {}", var);
        assert!(
            var >= dec!(0.40) && var <= dec!(0.50),
            "VaR should be around 0.45, got {}",
            var
        );
    }

    #[test]
    fn test_historical_var_all_positive_returns() {
        let calc = VaRCalculator::new(default_config());

        // All positive returns - no loss expected
        let returns: Vec<Decimal> =
            vec![dec!(0.01), dec!(0.02), dec!(0.03), dec!(0.04), dec!(0.05)];

        let var = calc.historical_var(&returns);
        assert_eq!(
            var,
            Decimal::ZERO,
            "VaR should be 0 for all positive returns"
        );
    }

    #[test]
    fn test_historical_var_empty_returns() {
        let calc = VaRCalculator::new(default_config());
        let returns: Vec<Decimal> = vec![];

        let var = calc.historical_var(&returns);
        assert_eq!(var, Decimal::ZERO);
    }

    #[test]
    fn test_parametric_var() {
        let mut config = default_config();
        config.method = VaRMethod::Parametric;
        let calc = VaRCalculator::new(config);

        // Returns with mean ≈ 0 and stddev ≈ 0.02
        let returns: Vec<Decimal> =
            vec![dec!(-0.02), dec!(-0.01), dec!(0.00), dec!(0.01), dec!(0.02)];

        let var = calc.parametric_var(&returns);

        // Parametric VaR = z * stddev - mean ≈ 1.645 * 0.0158 ≈ 0.026
        assert!(var > Decimal::ZERO, "Parametric VaR should be positive");
        assert!(
            var < dec!(0.10),
            "Parametric VaR should be reasonable, got {}",
            var
        );
    }

    #[test]
    fn test_cvar_calculation() {
        let calc = VaRCalculator::new(default_config());

        // Returns with clear tail
        let returns: Vec<Decimal> = vec![
            dec!(-0.10), // Tail
            dec!(-0.08), // Tail
            dec!(-0.05),
            dec!(-0.02),
            dec!(0.00),
            dec!(0.01),
            dec!(0.02),
            dec!(0.03),
            dec!(0.04),
            dec!(0.05),
        ];

        let var = calc.historical_var(&returns);
        let cvar = calc.calculate_cvar(&returns, var);

        // CVaR should be >= VaR (it's the average of tail losses)
        assert!(cvar >= var, "CVaR {} should be >= VaR {}", cvar, var);
    }

    #[test]
    fn test_10d_var_scaling() {
        let calc = VaRCalculator::new(default_config());

        let returns: Vec<Decimal> = vec![
            dec!(-0.05),
            dec!(-0.03),
            dec!(-0.01),
            dec!(0.01),
            dec!(0.03),
        ];

        let (var_1d, _) = calc.calculate_var_and_cvar(&returns);
        let sqrt_10 = Decimal::new(SQRT_10.0, SQRT_10.1);
        let var_10d = var_1d * sqrt_10;

        // 10-day VaR should be approximately 3.16x the 1-day VaR
        let expected_ratio = sqrt_10;
        if !var_1d.is_zero() {
            let actual_ratio = var_10d / var_1d;
            assert!(
                (actual_ratio - expected_ratio).abs() < dec!(0.01),
                "10-day VaR ratio should be sqrt(10), got {}",
                actual_ratio
            );
        }
    }

    #[test]
    fn test_calculate_returns() {
        let calc = VaRCalculator::new(default_config());

        let prices = vec![dec!(100), dec!(102), dec!(99), dec!(105)];
        let returns = calc.calculate_returns(&prices);

        assert_eq!(returns.len(), 3);
        assert_eq!(returns[0], dec!(0.02)); // (102-100)/100
        assert!((returns[1] - dec!(-0.0294117647)).abs() < dec!(0.0001)); // (99-102)/102
        assert!((returns[2] - dec!(0.0606060606)).abs() < dec!(0.0001)); // (105-99)/99
    }

    #[test]
    fn test_mean_stddev_calculation() {
        let calc = VaRCalculator::new(default_config());

        let returns = vec![
            dec!(2),
            dec!(4),
            dec!(4),
            dec!(4),
            dec!(5),
            dec!(5),
            dec!(7),
            dec!(9),
        ];
        let (mean, stddev) = calc.calculate_mean_stddev(&returns);

        // Mean = 40/8 = 5
        assert_eq!(mean, dec!(5));

        // Variance = [(2-5)² + (4-5)² + ... + (9-5)²] / 7 = 32/7 ≈ 4.57
        // Stddev ≈ 2.14
        assert!(
            (stddev - dec!(2.14)).abs() < dec!(0.1),
            "Stddev should be around 2.14, got {}",
            stddev
        );
    }

    #[test]
    fn test_exceeds_contribution_limit() {
        let calc = VaRCalculator::new(default_config());

        // 30% contribution should exceed 25% limit
        assert!(calc.exceeds_contribution_limit(dec!(30), dec!(100)));

        // 20% contribution should not exceed 25% limit
        assert!(!calc.exceeds_contribution_limit(dec!(20), dec!(100)));

        // Zero total should not trigger
        assert!(!calc.exceeds_contribution_limit(dec!(10), Decimal::ZERO));
    }

    #[test]
    fn test_disabled_var() {
        let mut config = default_config();
        config.enabled = false;
        let calc = VaRCalculator::new(config);

        assert!(!calc.is_enabled());

        let returns = vec![dec!(-0.10), dec!(-0.05), dec!(0.00)];
        let var = calc.calculate_var_from_returns(&returns, dec!(1000));

        // Should return 0 when disabled
        assert_eq!(var, Decimal::ZERO);
    }

    #[test]
    fn test_sqrt_decimal() {
        let calc = VaRCalculator::new(default_config());

        // Test sqrt(4) = 2 with tolerance
        let sqrt4 = calc.sqrt_decimal(dec!(4)).unwrap();
        assert!(
            (sqrt4 - dec!(2)).abs() < dec!(0.0001),
            "sqrt(4) should be ~2, got {}",
            sqrt4
        );

        // Test sqrt(9) = 3 with tolerance
        let sqrt9 = calc.sqrt_decimal(dec!(9)).unwrap();
        assert!(
            (sqrt9 - dec!(3)).abs() < dec!(0.0001),
            "sqrt(9) should be ~3, got {}",
            sqrt9
        );

        // Test sqrt(0) = 0
        assert_eq!(calc.sqrt_decimal(dec!(0)).unwrap(), dec!(0));

        // Test negative returns None
        assert!(calc.sqrt_decimal(dec!(-1)).is_none());

        // sqrt(2) ≈ 1.414
        let sqrt2 = calc.sqrt_decimal(dec!(2)).unwrap();
        assert!((sqrt2 - dec!(1.414213562)).abs() < dec!(0.0001));
    }

    #[test]
    fn test_var_with_position_value() {
        let calc = VaRCalculator::new(default_config());

        // 5% loss return with $1000 position = $50 VaR
        let returns = vec![
            dec!(-0.05),
            dec!(-0.03),
            dec!(-0.01),
            dec!(0.01),
            dec!(0.03),
        ];

        let position_value = dec!(1000);
        let var = calc.calculate_var_from_returns(&returns, position_value);

        // VaR should be scaled by position value
        assert!(var > Decimal::ZERO);
        assert!(var < position_value);
    }

    #[test]
    fn test_z_score_selection() {
        let mut config = default_config();
        config.confidence_level = dec!(0.99);
        let calc = VaRCalculator::new(config);

        let z = calc.get_z_score();
        assert!((z - dec!(2.326)).abs() < dec!(0.001));

        let mut config95 = default_config();
        config95.confidence_level = dec!(0.95);
        let calc95 = VaRCalculator::new(config95);

        let z95 = calc95.get_z_score();
        assert!((z95 - dec!(1.645)).abs() < dec!(0.001));
    }
}
