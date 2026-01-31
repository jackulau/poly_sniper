//! Polysniper Strategies
//!
//! Trading strategy implementations.

pub mod arbitrage;
pub mod event_based;
pub mod liquidity_provision;
pub mod llm_prediction;
pub mod ml_processor;
pub mod multi_leg;
pub mod new_market;
pub mod orderbook_imbalance;
pub mod polymarket_activity;
pub mod price_spike;
pub mod resolution_exit;
pub mod sentiment_analyzer;
pub mod sentiment_strategy;
pub mod target_price;

pub use arbitrage::{ArbitrageConfig, ArbitrageStrategy};
pub use event_based::{EventBasedConfig, EventBasedStrategy};
pub use liquidity_provision::{LiquidityProvisionConfig, LiquidityProvisionStrategy};
pub use llm_prediction::{LlmPredictionConfig, LlmPredictionStrategy};
pub use ml_processor::{MlProcessingResult, MlSignalProcessor};
pub use multi_leg::{
    CorrelationLeg, CorrelationRelationship, CorrelationRule, MultiLegConfig, MultiLegStrategy,
};
pub use new_market::{NewMarketConfig, NewMarketStrategy};
pub use orderbook_imbalance::{OrderbookImbalanceConfig, OrderbookImbalanceStrategy};
pub use price_spike::{PriceSpikeConfig, PriceSpikeStrategy};
pub use polymarket_activity::{
    CommentActivityConfig, PolymarketActivityStrategy, PolymarketActivityStrategyConfig,
    SmartMoneyConfig, VolumeAnomalyConfig,
};
pub use resolution_exit::ResolutionExitStrategy;
pub use sentiment_analyzer::SentimentAnalyzer;
pub use sentiment_strategy::{SentimentMarketMapping, SentimentStrategy, SentimentStrategyConfig};
pub use target_price::{PriceTarget, TargetDirection, TargetPriceConfig, TargetPriceStrategy};
