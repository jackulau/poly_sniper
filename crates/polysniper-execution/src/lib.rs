//! Polysniper Execution
//!
//! Order building and submission to CLOB.

pub mod gas_optimizer;
pub mod gas_tracker;
pub mod order_builder;
pub mod order_manager;
pub mod submitter;

pub use gas_optimizer::{GasOptimizer, GasOptimizerHandle, QueuedOrder};
pub use gas_tracker::{GasHistoryStats, GasTracker};
pub use order_builder::OrderBuilder;
pub use order_manager::{
    ManagedOrder, ManagementPolicy, OrderManager, ReplaceAction, ReplaceDecision, ReplaceResult,
};
pub use submitter::OrderSubmitter;
