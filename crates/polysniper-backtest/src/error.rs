//! Backtest error types

use thiserror::Error;

/// Backtest result type alias
pub type Result<T> = std::result::Result<T, BacktestError>;

/// Backtest errors
#[derive(Error, Debug)]
pub enum BacktestError {
    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("No data available for the specified time range")]
    NoData,

    #[error("Strategy error: {0}")]
    StrategyError(String),

    #[error("Invalid time range: {0}")]
    InvalidTimeRange(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl From<polysniper_persistence::PersistenceError> for BacktestError {
    fn from(err: polysniper_persistence::PersistenceError) -> Self {
        BacktestError::DatabaseError(err.to_string())
    }
}

impl From<polysniper_core::StrategyError> for BacktestError {
    fn from(err: polysniper_core::StrategyError) -> Self {
        BacktestError::StrategyError(err.to_string())
    }
}

impl From<serde_json::Error> for BacktestError {
    fn from(err: serde_json::Error) -> Self {
        BacktestError::SerializationError(err.to_string())
    }
}
