//! Polysniper Execution
//!
//! Order building and submission to CLOB.

pub mod depth_analyzer;
pub mod order_builder;
pub mod submitter;

pub use depth_analyzer::{
    DepthAnalyzer, DepthAnalyzerConfig, OrderSizeRecommendation, PriceImpact,
};
pub use order_builder::OrderBuilder;
pub use submitter::OrderSubmitter;
