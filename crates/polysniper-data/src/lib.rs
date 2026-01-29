//! Polysniper Data
//!
//! Data ingestion layer for WebSocket and REST API clients.

pub mod event_bus;
pub mod feed_aggregator;
pub mod feed_types;
pub mod gamma_client;
pub mod market_cache;
pub mod rss_client;
pub mod twitter_client;
pub mod ws_manager;

pub use event_bus::BroadcastEventBus;
pub use feed_aggregator::FeedAggregator;
pub use feed_types::{FeedConfig, FeedError, FeedItem, FeedSource, RssConfig, RssFeedConfig, TwitterConfig, TwitterQueryConfig};
pub use gamma_client::GammaClient;
pub use market_cache::MarketCache;
pub use rss_client::RssClient;
pub use twitter_client::TwitterClient;
pub use ws_manager::WsManager;
