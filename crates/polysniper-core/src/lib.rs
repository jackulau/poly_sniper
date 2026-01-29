//! Polysniper Core
//!
//! Core types, traits, and events for the Polysniper trading system.

pub mod config_watcher;
pub mod error;
pub mod events;
pub mod resolution;
pub mod traits;
pub mod types;

// Re-export commonly used types
pub use config_watcher::*;
pub use error::*;
pub use events::*;
pub use resolution::*;
pub use traits::*;
pub use types::*;
