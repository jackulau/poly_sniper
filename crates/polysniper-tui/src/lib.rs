//! Terminal UI for Polysniper orderbook visualization
//!
//! This crate provides a real-time terminal-based visualization of orderbook
//! depth, price tickers, and recent trades.

pub mod app;
mod event;
mod ui;
pub mod widgets;

pub use app::{App, AppAction, TradeRecord};
pub use event::{Event, EventHandler};
pub use ui::draw;
