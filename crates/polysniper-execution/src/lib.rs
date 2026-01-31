//! Polysniper Execution
//!
//! Order building, submission to CLOB, and execution algorithms (TWAP/VWAP).

pub mod algorithms;
pub mod depth_analyzer;
pub mod fast_submitter;
pub mod fill_manager;
pub mod fill_poller;
pub mod gas_optimizer;
pub mod gas_tracker;
pub mod market_impact;
pub mod order_builder;
pub mod order_manager;
pub mod price_level_analyzer;
pub mod participation_adapter;
pub mod queue_estimator;
pub mod shortfall_tracker;
pub mod submitter;
pub mod volume_monitor;

pub use algorithms::{
    AlgorithmConfig, AlgorithmType, ChildOrder, ExecutionStats, TwapConfig, TwapExecutor,
    VolumeProfile, VwapConfig, VwapExecutor,
};
pub use depth_analyzer::{DepthAnalyzer, DepthAnalyzerConfig};
pub use fast_submitter::{FastSubmitter, FastSubmitterConfig, OrderCache, OrderTemplate, SignedOrderData, SubmissionStats};
pub use fill_manager::FillManager;
pub use fill_poller::FillPoller;
pub use gas_optimizer::GasOptimizer;
pub use gas_tracker::GasTracker;
pub use market_impact::{
    ImpactModelType, ImpactObservation, ImpactParameters, ImpactPrediction, ImpactRecommendation,
    MarketConditions, MarketImpactConfig, MarketImpactEstimator,
};
pub use order_builder::OrderBuilder;
pub use order_manager::OrderManager;
pub use price_level_analyzer::{PriceLevelAnalyzer, PriceLevelStats};
pub use queue_estimator::{FillProbability, ProbabilityComponents, ProbabilityMethod, QueueEstimator, QueuePositionState};
pub use participation_adapter::{ParticipationAdapter, ParticipationConfig, ParticipationRate};
pub use queue_estimator::QueueEstimator;
pub use shortfall_tracker::{
    ShortfallComponents, ShortfallConfig, ShortfallRecord, ShortfallSummary, ShortfallTracker,
};
pub use submitter::OrderSubmitter;
pub use volume_monitor::{VolumeMonitor, VolumeMonitorConfig, VolumeObservation};
