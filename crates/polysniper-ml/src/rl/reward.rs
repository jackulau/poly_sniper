//! Reward calculation for reinforcement learning execution timing.
//!
//! Defines the reward signal components and calculation logic.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Configuration for reward calculation weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardConfig {
    /// Weight for fill rate reward component
    pub fill_weight: f64,
    /// Weight for slippage penalty component
    pub slippage_weight: f64,
    /// Weight for time penalty component
    pub time_weight: f64,
    /// Weight for market impact penalty component
    pub impact_weight: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            fill_weight: 10.0,
            slippage_weight: 100.0,
            time_weight: 50.0,
            impact_weight: 50.0,
        }
    }
}

/// Reward signal after taking an action.
///
/// Combines multiple components to form a composite reward that balances
/// fill rate, cost (slippage), speed (time), and market impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReward {
    /// Reward for achieving fills (higher fill rate = better)
    pub fill_rate_reward: f64,
    /// Penalty for slippage from mid price (lower slippage = better)
    pub slippage_penalty: f64,
    /// Penalty for taking too long (exponential cost for delay)
    pub time_penalty: f64,
    /// Penalty for moving the market price
    pub market_impact_penalty: f64,
}

impl ExecutionReward {
    /// Create a new reward with zero values
    pub fn zero() -> Self {
        Self {
            fill_rate_reward: 0.0,
            slippage_penalty: 0.0,
            time_penalty: 0.0,
            market_impact_penalty: 0.0,
        }
    }

    /// Calculate total reward (sum of rewards minus penalties)
    pub fn total(&self) -> f64 {
        self.fill_rate_reward - self.slippage_penalty - self.time_penalty - self.market_impact_penalty
    }

    /// Create a reward for a successful fill
    pub fn from_fill(
        filled_size: Decimal,
        order_size: Decimal,
        fill_price: Decimal,
        mid_price: Decimal,
        time_elapsed_pct: f64,
        price_after_fill: Option<Decimal>,
        config: &RewardConfig,
    ) -> Self {
        // Fill rate reward: higher is better
        let fill_rate = if order_size.is_zero() {
            0.0
        } else {
            (filled_size / order_size)
                .to_string()
                .parse::<f64>()
                .unwrap_or(0.0)
        };
        let fill_rate_reward = fill_rate * config.fill_weight;

        // Slippage penalty: deviation from mid price
        let slippage = if mid_price.is_zero() {
            0.0
        } else {
            ((fill_price - mid_price) / mid_price)
                .abs()
                .to_string()
                .parse::<f64>()
                .unwrap_or(0.0)
        };
        let slippage_penalty = slippage * config.slippage_weight;

        // Time penalty: exponential cost for delay after 90% of time window
        let time_penalty = if time_elapsed_pct > 0.9 {
            (time_elapsed_pct - 0.9) * config.time_weight
        } else {
            0.0
        };

        // Market impact penalty: price movement after our order
        let market_impact_penalty = price_after_fill
            .map(|price_after| {
                if fill_price.is_zero() {
                    0.0
                } else {
                    let impact = ((price_after - fill_price) / fill_price)
                        .abs()
                        .to_string()
                        .parse::<f64>()
                        .unwrap_or(0.0);
                    impact * config.impact_weight
                }
            })
            .unwrap_or(0.0);

        Self {
            fill_rate_reward,
            slippage_penalty,
            time_penalty,
            market_impact_penalty,
        }
    }

    /// Create a small penalty for waiting (opportunity cost)
    pub fn wait_penalty(time_elapsed_pct: f64) -> Self {
        // Small penalty for waiting that increases as time runs out
        let base_penalty = 0.1;
        let time_factor = if time_elapsed_pct > 0.5 {
            1.0 + (time_elapsed_pct - 0.5) * 2.0
        } else {
            1.0
        };

        Self {
            fill_rate_reward: 0.0,
            slippage_penalty: 0.0,
            time_penalty: base_penalty * time_factor,
            market_impact_penalty: 0.0,
        }
    }

    /// Create a penalty for cancellation
    pub fn cancel_penalty() -> Self {
        Self {
            fill_rate_reward: 0.0,
            slippage_penalty: 0.0,
            time_penalty: 0.5, // Moderate penalty for cancelling
            market_impact_penalty: 0.0,
        }
    }

    /// Create a reward for completing the full execution
    pub fn completion_bonus(avg_slippage: f64, total_time_pct: f64) -> Self {
        // Bonus for completing execution, scaled by quality
        let slippage_factor = (1.0 - avg_slippage * 10.0).max(0.0);
        let time_factor = (1.0 - total_time_pct).max(0.0) * 0.5 + 0.5;

        Self {
            fill_rate_reward: 5.0 * slippage_factor * time_factor,
            slippage_penalty: 0.0,
            time_penalty: 0.0,
            market_impact_penalty: 0.0,
        }
    }
}

impl std::ops::Add for ExecutionReward {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            fill_rate_reward: self.fill_rate_reward + other.fill_rate_reward,
            slippage_penalty: self.slippage_penalty + other.slippage_penalty,
            time_penalty: self.time_penalty + other.time_penalty,
            market_impact_penalty: self.market_impact_penalty + other.market_impact_penalty,
        }
    }
}

impl std::ops::AddAssign for ExecutionReward {
    fn add_assign(&mut self, other: Self) {
        self.fill_rate_reward += other.fill_rate_reward;
        self.slippage_penalty += other.slippage_penalty;
        self.time_penalty += other.time_penalty;
        self.market_impact_penalty += other.market_impact_penalty;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_reward_total() {
        let reward = ExecutionReward {
            fill_rate_reward: 10.0,
            slippage_penalty: 2.0,
            time_penalty: 1.0,
            market_impact_penalty: 0.5,
        };

        assert_eq!(reward.total(), 6.5); // 10 - 2 - 1 - 0.5
    }

    #[test]
    fn test_reward_from_fill() {
        let config = RewardConfig::default();
        let reward = ExecutionReward::from_fill(
            dec!(100),  // filled
            dec!(100),  // order size
            dec!(0.51), // fill price
            dec!(0.50), // mid price
            0.5,        // time elapsed
            None,       // no price after
            &config,
        );

        // Full fill should give max fill reward
        assert!(reward.fill_rate_reward > 9.0);

        // 2% slippage should give notable penalty
        assert!(reward.slippage_penalty > 1.0);

        // Time elapsed 50% should have no time penalty
        assert_eq!(reward.time_penalty, 0.0);

        // Total should still be positive for a good fill
        assert!(reward.total() > 0.0);
    }

    #[test]
    fn test_reward_time_penalty() {
        let config = RewardConfig::default();

        // At 95% time elapsed, should have time penalty
        let reward = ExecutionReward::from_fill(
            dec!(100),
            dec!(100),
            dec!(0.50), // No slippage
            dec!(0.50),
            0.95,
            None,
            &config,
        );

        assert!(reward.time_penalty > 0.0);
    }

    #[test]
    fn test_wait_penalty_increases_over_time() {
        let early_penalty = ExecutionReward::wait_penalty(0.3);
        let late_penalty = ExecutionReward::wait_penalty(0.8);

        assert!(late_penalty.time_penalty > early_penalty.time_penalty);
    }

    #[test]
    fn test_reward_addition() {
        let r1 = ExecutionReward {
            fill_rate_reward: 5.0,
            slippage_penalty: 1.0,
            time_penalty: 0.5,
            market_impact_penalty: 0.0,
        };

        let r2 = ExecutionReward {
            fill_rate_reward: 3.0,
            slippage_penalty: 0.5,
            time_penalty: 0.0,
            market_impact_penalty: 0.2,
        };

        let sum = r1 + r2;
        assert_eq!(sum.fill_rate_reward, 8.0);
        assert_eq!(sum.slippage_penalty, 1.5);
        assert_eq!(sum.time_penalty, 0.5);
        assert_eq!(sum.market_impact_penalty, 0.2);
    }
}
