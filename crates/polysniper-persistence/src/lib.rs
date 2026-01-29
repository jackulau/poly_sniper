//! Polysniper Persistence
//!
//! SQLite database persistence for trade history, orders, and strategy state.

mod database;
mod error;
mod models;
pub mod pnl_calculator;
mod repositories;

pub use database::Database;
pub use error::{PersistenceError, Result};
pub use models::*;
pub use pnl_calculator::{
    calculate_trade_pnl, calculate_unrealized_pnl, CostBasisMethod, Lot, PnlResult,
    PositionTracker,
};
pub use repositories::*;
