//! Polysniper Backtest
//!
//! Historical backtesting engine for strategy evaluation.

pub mod config;
pub mod data_loader;
pub mod engine;
pub mod error;
pub mod results;

pub use config::BacktestConfig;
pub use data_loader::DataLoader;
pub use engine::BacktestEngine;
pub use error::{BacktestError, Result};
pub use results::{BacktestResults, TradeResult};
