//! Kelly criterion position sizing
//!
//! Implements the Kelly criterion formula for optimal position sizing based on
//! historical edge and win rate. The Kelly formula maximizes long-term growth
//! while managing risk.
//!
//! The full Kelly formula is: f* = (bp - q) / b
//! where:
//!   - b = odds (average win / average loss)
//!   - p = probability of winning
//!   - q = probability of losing (1 - p)
//!   - f* = optimal fraction of capital to bet
//!
//! This implementation supports fractional Kelly (e.g., half-Kelly) for more
//! conservative position sizing.

use polysniper_core::KellyConfig;
use rust_decimal::Decimal;
use tracing::{debug, info};

/// A completed trade outcome for Kelly calculation
#[derive(Debug, Clone)]
pub struct TradeOutcome {
    /// The P&L of the trade (positive = win, negative = loss)
    pub pnl: Decimal,
    /// The trade size in USD (used for normalizing returns)
    pub size_usd: Decimal,
}

impl TradeOutcome {
    /// Create a new trade outcome
    pub fn new(pnl: Decimal, size_usd: Decimal) -> Self {
        Self { pnl, size_usd }
    }

    /// Check if this was a winning trade
    pub fn is_win(&self) -> bool {
        self.pnl > Decimal::ZERO
    }

    /// Get the return percentage (pnl / size)
    pub fn return_pct(&self) -> Option<Decimal> {
        if self.size_usd.is_zero() {
            None
        } else {
            Some(self.pnl / self.size_usd)
        }
    }
}

/// Result of Kelly calculation
#[derive(Debug, Clone)]
pub struct KellyResult {
    /// The calculated Kelly multiplier (before config bounds)
    pub raw_multiplier: Decimal,
    /// The final multiplier after applying config bounds
    pub multiplier: Decimal,
    /// Estimated win rate (0.0 - 1.0)
    pub win_rate: Decimal,
    /// Average win/loss ratio (odds)
    pub odds: Decimal,
    /// Calculated edge ((odds * win_rate) - (1 - win_rate))
    pub edge: Decimal,
    /// Number of trades used in calculation
    pub sample_size: usize,
    /// Whether minimum sample size was met
    pub sufficient_data: bool,
}

/// Kelly criterion calculator for position sizing
#[derive(Debug, Clone)]
pub struct KellyCalculator {
    config: KellyConfig,
}

impl KellyCalculator {
    /// Create a new Kelly calculator with the given config
    pub fn new(config: KellyConfig) -> Self {
        Self { config }
    }

    /// Check if Kelly sizing is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the configured fractional Kelly value
    pub fn fraction(&self) -> Decimal {
        self.config.fraction
    }

    /// Calculate Kelly sizing multiplier from trade history
    ///
    /// Returns None if Kelly is disabled or there's insufficient data.
    /// The multiplier should be applied to the position size.
    pub fn calculate(&self, trades: &[TradeOutcome]) -> Option<KellyResult> {
        if !self.config.enabled {
            return None;
        }

        let sample_size = trades.len();
        let min_trades = self.config.min_trades as usize;

        // Check minimum sample size
        if sample_size < min_trades {
            debug!(
                sample_size = sample_size,
                min_trades = min_trades,
                "Insufficient trade history for Kelly calculation"
            );
            return Some(KellyResult {
                raw_multiplier: Decimal::ONE,
                multiplier: Decimal::ONE,
                win_rate: Decimal::ZERO,
                odds: Decimal::ZERO,
                edge: Decimal::ZERO,
                sample_size,
                sufficient_data: false,
            });
        }

        // Take only the most recent trades within the window
        let window_size = self.config.window_size as usize;
        let recent_trades: Vec<_> = if trades.len() > window_size {
            trades.iter().rev().take(window_size).collect()
        } else {
            trades.iter().collect()
        };

        // Calculate win rate
        let wins: Vec<_> = recent_trades.iter().filter(|t| t.is_win()).collect();
        let losses: Vec<_> = recent_trades.iter().filter(|t| !t.is_win()).collect();

        let total_trades = Decimal::from(recent_trades.len());
        let win_count = Decimal::from(wins.len());

        if total_trades.is_zero() {
            return None;
        }

        let win_rate = win_count / total_trades;
        let loss_rate = Decimal::ONE - win_rate;

        // Calculate average win and average loss (as percentages of size)
        let avg_win_pct = if !wins.is_empty() {
            let total_win_pct: Decimal = wins
                .iter()
                .filter_map(|t| t.return_pct())
                .sum();
            total_win_pct / Decimal::from(wins.len())
        } else {
            Decimal::ZERO
        };

        let avg_loss_pct = if !losses.is_empty() {
            let total_loss_pct: Decimal = losses
                .iter()
                .filter_map(|t| t.return_pct())
                .map(|r| r.abs())
                .sum();
            total_loss_pct / Decimal::from(losses.len())
        } else {
            Decimal::ZERO
        };

        // Calculate odds (b = avg_win / avg_loss)
        let odds = if avg_loss_pct.is_zero() {
            // No losses - very good but use conservative odds
            Decimal::new(2, 0) // 2:1 odds
        } else {
            avg_win_pct / avg_loss_pct
        };

        // Calculate edge: edge = (b * p) - q
        // where b = odds, p = win_rate, q = loss_rate
        let edge = (odds * win_rate) - loss_rate;

        // Calculate Kelly fraction: f* = (bp - q) / b = edge / b
        let kelly_fraction = if odds.is_zero() {
            Decimal::ZERO
        } else {
            edge / odds
        };

        // Apply fractional Kelly
        let adjusted_kelly = kelly_fraction * self.config.fraction;

        // Clamp to config bounds
        let clamped = adjusted_kelly
            .max(self.config.min_multiplier)
            .min(self.config.max_multiplier);

        info!(
            win_rate = %win_rate,
            odds = %odds,
            edge = %edge,
            kelly_fraction = %kelly_fraction,
            fractional_kelly = %adjusted_kelly,
            multiplier = %clamped,
            sample_size = sample_size,
            "Kelly criterion calculation"
        );

        Some(KellyResult {
            raw_multiplier: adjusted_kelly,
            multiplier: clamped,
            win_rate,
            odds,
            edge,
            sample_size,
            sufficient_data: true,
        })
    }

    /// Calculate the size multiplier to apply to a position
    ///
    /// Returns (multiplier, reason) tuple. The multiplier is 1.0 if Kelly
    /// is disabled or there's insufficient data.
    pub fn calculate_size_multiplier(
        &self,
        trades: &[TradeOutcome],
    ) -> (Decimal, Option<String>) {
        if !self.config.enabled {
            return (Decimal::ONE, None);
        }

        match self.calculate(trades) {
            Some(result) if result.sufficient_data => {
                let reason = format!(
                    "Kelly sizing: edge {:.2}%, win rate {:.1}%, odds {:.2}:1, multiplier {:.2}x",
                    result.edge * Decimal::ONE_HUNDRED,
                    result.win_rate * Decimal::ONE_HUNDRED,
                    result.odds,
                    result.multiplier
                );
                (result.multiplier, Some(reason))
            }
            Some(result) => {
                debug!(
                    sample_size = result.sample_size,
                    min_required = self.config.min_trades,
                    "Skipping Kelly sizing - insufficient trade history"
                );
                (Decimal::ONE, None)
            }
            None => (Decimal::ONE, None),
        }
    }

    /// Adjust a position size based on Kelly criterion
    ///
    /// Returns (adjusted_size, multiplier, reason)
    pub fn adjust_size(
        &self,
        original_size: Decimal,
        trades: &[TradeOutcome],
    ) -> (Decimal, Decimal, Option<String>) {
        let (multiplier, reason) = self.calculate_size_multiplier(trades);
        let adjusted = original_size * multiplier;

        if multiplier != Decimal::ONE {
            info!(
                original_size = %original_size,
                adjusted_size = %adjusted,
                multiplier = %multiplier,
                "Applied Kelly criterion position sizing"
            );
        }

        (adjusted, multiplier, reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn default_config() -> KellyConfig {
        KellyConfig {
            enabled: true,
            fraction: dec!(0.5),      // Half-Kelly
            window_size: 100,
            min_trades: 20,
            max_multiplier: dec!(2.0),
            min_multiplier: dec!(0.25),
        }
    }

    fn make_trades(outcomes: &[(f64, f64)]) -> Vec<TradeOutcome> {
        outcomes
            .iter()
            .map(|(pnl, size)| TradeOutcome::new(
                Decimal::from_f64_retain(*pnl).unwrap(),
                Decimal::from_f64_retain(*size).unwrap(),
            ))
            .collect()
    }

    #[test]
    fn test_kelly_disabled() {
        let mut config = default_config();
        config.enabled = false;
        let calc = KellyCalculator::new(config);

        let trades = make_trades(&[(10.0, 100.0); 30]);
        let result = calc.calculate(&trades);
        assert!(result.is_none());
    }

    #[test]
    fn test_insufficient_trades() {
        let calc = KellyCalculator::new(default_config());

        // Only 10 trades, need 20
        let trades = make_trades(&[(10.0, 100.0); 10]);
        let result = calc.calculate(&trades).unwrap();

        assert!(!result.sufficient_data);
        assert_eq!(result.multiplier, Decimal::ONE);
        assert_eq!(result.sample_size, 10);
    }

    #[test]
    fn test_all_winning_trades() {
        let calc = KellyCalculator::new(default_config());

        // All wins with 10% return
        let trades = make_trades(&[(10.0, 100.0); 30]);
        let result = calc.calculate(&trades).unwrap();

        assert!(result.sufficient_data);
        assert_eq!(result.win_rate, Decimal::ONE);
        // With 100% win rate and conservative 2:1 odds, edge = 2*1 - 0 = 2
        // Kelly = edge/odds = 2/2 = 1.0, half-Kelly = 0.5
        // But clamped to max of 2.0
        assert!(result.multiplier <= dec!(2.0));
    }

    #[test]
    fn test_50_50_win_rate_equal_wins_losses() {
        let calc = KellyCalculator::new(default_config());

        // 50% win rate, equal win/loss amounts (0 edge)
        let mut trades = Vec::new();
        for _ in 0..15 {
            trades.push(TradeOutcome::new(dec!(10), dec!(100))); // Win $10
        }
        for _ in 0..15 {
            trades.push(TradeOutcome::new(dec!(-10), dec!(100))); // Lose $10
        }

        let result = calc.calculate(&trades).unwrap();

        assert!(result.sufficient_data);
        assert_eq!(result.win_rate, dec!(0.5));
        assert_eq!(result.odds, Decimal::ONE); // 10/10 = 1:1 odds
        // Edge = (1 * 0.5) - 0.5 = 0
        assert_eq!(result.edge, Decimal::ZERO);
        // Kelly = 0/1 = 0, clamped to min 0.25
        assert_eq!(result.multiplier, dec!(0.25));
    }

    #[test]
    fn test_positive_edge() {
        let calc = KellyCalculator::new(default_config());

        // 60% win rate, 2:1 win/loss ratio (strong positive edge)
        let mut trades = Vec::new();
        for _ in 0..18 {
            trades.push(TradeOutcome::new(dec!(20), dec!(100))); // Win $20 (20% return)
        }
        for _ in 0..12 {
            trades.push(TradeOutcome::new(dec!(-10), dec!(100))); // Lose $10 (10% loss)
        }

        let result = calc.calculate(&trades).unwrap();

        assert!(result.sufficient_data);
        assert_eq!(result.win_rate, dec!(0.6));
        assert_eq!(result.odds, dec!(2)); // 20% / 10% = 2:1 odds
        // Edge = (2 * 0.6) - 0.4 = 1.2 - 0.4 = 0.8
        assert_eq!(result.edge, dec!(0.8));
        // Kelly = 0.8 / 2 = 0.4, half-Kelly = 0.2
        // But should be between min and max
        assert!(result.multiplier >= dec!(0.25));
        assert!(result.multiplier <= dec!(2.0));
    }

    #[test]
    fn test_negative_edge_clamped_to_min() {
        let calc = KellyCalculator::new(default_config());

        // 30% win rate, 1:1 odds (negative edge)
        let mut trades = Vec::new();
        for _ in 0..9 {
            trades.push(TradeOutcome::new(dec!(10), dec!(100))); // Win
        }
        for _ in 0..21 {
            trades.push(TradeOutcome::new(dec!(-10), dec!(100))); // Lose
        }

        let result = calc.calculate(&trades).unwrap();

        assert!(result.sufficient_data);
        assert_eq!(result.win_rate, dec!(0.3));
        // Edge = (1 * 0.3) - 0.7 = -0.4 (negative)
        assert!(result.edge < Decimal::ZERO);
        // Kelly would be negative, clamped to min
        assert_eq!(result.multiplier, dec!(0.25));
    }

    #[test]
    fn test_window_size_limits() {
        let mut config = default_config();
        config.window_size = 25; // Only use last 25 trades
        let calc = KellyCalculator::new(config);

        // First 30 trades are losses
        let mut trades = Vec::new();
        for _ in 0..30 {
            trades.push(TradeOutcome::new(dec!(-10), dec!(100)));
        }
        // Last 25 trades are wins
        for _ in 0..25 {
            trades.push(TradeOutcome::new(dec!(10), dec!(100)));
        }

        let result = calc.calculate(&trades).unwrap();

        // Should only consider last 25 trades (all wins)
        assert!(result.sufficient_data);
        assert_eq!(result.win_rate, Decimal::ONE);
        assert_eq!(result.sample_size, 55); // Total trades
    }

    #[test]
    fn test_size_multiplier_method() {
        let calc = KellyCalculator::new(default_config());

        // Strong positive edge
        let mut trades = Vec::new();
        for _ in 0..20 {
            trades.push(TradeOutcome::new(dec!(20), dec!(100)));
        }
        for _ in 0..10 {
            trades.push(TradeOutcome::new(dec!(-10), dec!(100)));
        }

        let (multiplier, reason) = calc.calculate_size_multiplier(&trades);

        assert!(multiplier > Decimal::ONE || multiplier >= dec!(0.25));
        assert!(reason.is_some());
    }

    #[test]
    fn test_adjust_size() {
        let calc = KellyCalculator::new(default_config());

        let original_size = dec!(100);
        let trades = make_trades(&[(10.0, 100.0); 30]);

        let (adjusted, multiplier, reason) = calc.adjust_size(original_size, &trades);

        // Adjusted size should be original * multiplier
        assert_eq!(adjusted, original_size * multiplier);
        assert!(reason.is_some());
    }

    #[test]
    fn test_trade_outcome_return_pct() {
        let win = TradeOutcome::new(dec!(10), dec!(100));
        assert_eq!(win.return_pct(), Some(dec!(0.1)));
        assert!(win.is_win());

        let loss = TradeOutcome::new(dec!(-5), dec!(100));
        assert_eq!(loss.return_pct(), Some(dec!(-0.05)));
        assert!(!loss.is_win());

        let zero_size = TradeOutcome::new(dec!(10), dec!(0));
        assert!(zero_size.return_pct().is_none());
    }

    #[test]
    fn test_fractional_kelly_values() {
        // Test quarter-Kelly
        let mut config = default_config();
        config.fraction = dec!(0.25);
        let calc = KellyCalculator::new(config);

        let mut trades = Vec::new();
        for _ in 0..18 {
            trades.push(TradeOutcome::new(dec!(20), dec!(100)));
        }
        for _ in 0..12 {
            trades.push(TradeOutcome::new(dec!(-10), dec!(100)));
        }

        let quarter_result = calc.calculate(&trades).unwrap();

        // Test full Kelly
        let mut config = default_config();
        config.fraction = dec!(1.0);
        let calc = KellyCalculator::new(config);

        let full_result = calc.calculate(&trades).unwrap();

        // Quarter-Kelly raw multiplier should be 1/4 of full Kelly
        // (before clamping)
        assert!(quarter_result.raw_multiplier < full_result.raw_multiplier);
    }
}
