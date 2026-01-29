//! Polysniper Strategies
//!
//! Trading strategy implementations.

pub mod event_based;
pub mod multi_leg;
pub mod new_market;
pub mod price_spike;
pub mod target_price;

pub use event_based::{EventBasedConfig, EventBasedStrategy};
pub use multi_leg::{
    CorrelationLeg, CorrelationRelationship, CorrelationRule, MultiLegConfig, MultiLegStrategy,
};
pub use new_market::{NewMarketConfig, NewMarketStrategy};
pub use price_spike::{PriceSpikeConfig, PriceSpikeStrategy};
pub use target_price::{PriceTarget, TargetDirection, TargetPriceConfig, TargetPriceStrategy};
