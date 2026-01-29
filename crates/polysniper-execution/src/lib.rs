//! Polysniper Execution
//!
//! Order building, submission to CLOB, and execution algorithms (TWAP/VWAP).

pub mod algorithms;
pub mod order_builder;
pub mod submitter;

pub use algorithms::{
    AlgorithmConfig, AlgorithmType, ChildOrder, ExecutionStats, TwapConfig, TwapExecutor,
    VolumeProfile, VwapConfig, VwapExecutor,
};
pub use order_builder::OrderBuilder;
pub use submitter::OrderSubmitter;
