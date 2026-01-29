//! Backtest configuration

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Configuration for a backtest run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Start time for the backtest
    pub start_time: DateTime<Utc>,
    /// End time for the backtest
    pub end_time: DateTime<Utc>,
    /// Initial capital in USD
    pub initial_capital: Decimal,
    /// Fee configuration
    pub fees: FeeConfig,
    /// Slippage configuration
    pub slippage: SlippageConfig,
    /// Optional filter for specific market IDs
    pub market_filter: Option<Vec<String>>,
    /// Optional filter for specific token IDs
    pub token_filter: Option<Vec<String>>,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            start_time: Utc::now() - chrono::Duration::days(30),
            end_time: Utc::now(),
            initial_capital: Decimal::new(10000, 0),
            fees: FeeConfig::default(),
            slippage: SlippageConfig::default(),
            market_filter: None,
            token_filter: None,
        }
    }
}

/// Fee configuration for simulated trades
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeConfig {
    /// Trading fee as a percentage (e.g., 0.002 for 0.2%)
    pub trading_fee_pct: Decimal,
    /// Minimum fee in USD
    pub min_fee_usd: Decimal,
}

impl Default for FeeConfig {
    fn default() -> Self {
        Self {
            trading_fee_pct: Decimal::new(2, 3), // 0.2%
            min_fee_usd: Decimal::new(1, 2),    // $0.01
        }
    }
}

/// Slippage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageConfig {
    /// Slippage model to use
    pub model: SlippageModel,
    /// Base slippage percentage
    pub base_slippage_pct: Decimal,
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            model: SlippageModel::Fixed,
            base_slippage_pct: Decimal::new(1, 3), // 0.1%
        }
    }
}

/// Slippage models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlippageModel {
    /// No slippage (instant fill at signal price)
    None,
    /// Fixed slippage percentage
    Fixed,
    /// Slippage proportional to order size
    SizeProportional,
}
