//! Polysniper Execution
//!
//! Order building, submission to CLOB, and execution algorithms (TWAP/VWAP).

pub mod algorithms;
pub mod depth_analyzer;
pub mod fill_manager;
pub mod fill_poller;
pub mod gas_optimizer;
pub mod gas_tracker;
pub mod order_builder;
pub mod order_manager;
pub mod price_level_analyzer;
pub mod queue_estimator;
pub mod submitter;

pub use algorithms::{
    AlgorithmConfig, AlgorithmType, ChildOrder, ExecutionStats, TwapConfig, TwapExecutor,
    VolumeProfile, VwapConfig, VwapExecutor,
};
pub use depth_analyzer::{DepthAnalyzer, DepthAnalyzerConfig};
pub use fill_manager::FillManager;
pub use fill_poller::FillPoller;
pub use gas_optimizer::GasOptimizer;
pub use gas_tracker::GasTracker;
pub use order_builder::OrderBuilder;
pub use order_manager::OrderManager;
pub use price_level_analyzer::{PriceLevelAnalyzer, PriceLevelStats};
pub use queue_estimator::{FillProbability, ProbabilityComponents, ProbabilityMethod, QueueEstimator, QueuePositionState};
pub use submitter::OrderSubmitter;
