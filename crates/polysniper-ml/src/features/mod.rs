//! Feature computers for the feature store
//!
//! Each feature computer implements the FeatureComputer trait and computes
//! specific features from market data.

mod market;
mod orderbook;
mod price_history;
mod sentiment;
mod temporal;

pub use market::MarketFeatureComputer;
pub use orderbook::OrderbookFeatureComputer;
pub use price_history::PriceHistoryFeatureComputer;
pub use sentiment::SentimentFeatureComputer;
pub use temporal::TemporalFeatureComputer;

use crate::feature_store::{FeatureComputer, FeatureRegistry};
use std::sync::Arc;

/// Register all standard feature computers with the registry
pub fn register_all_computers(registry: &mut FeatureRegistry) {
    registry.register(Arc::new(MarketFeatureComputer::new()) as Arc<dyn FeatureComputer>);
    registry.register(Arc::new(OrderbookFeatureComputer::new()) as Arc<dyn FeatureComputer>);
    registry.register(Arc::new(SentimentFeatureComputer::new()) as Arc<dyn FeatureComputer>);
    registry.register(Arc::new(TemporalFeatureComputer::new()) as Arc<dyn FeatureComputer>);
    registry.register(Arc::new(PriceHistoryFeatureComputer::default()) as Arc<dyn FeatureComputer>);
}
