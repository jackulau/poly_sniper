//! Polysniper Data
//!
//! Data ingestion layer for WebSocket and REST API clients.

pub mod event_bus;
pub mod feed_aggregator;
pub mod feed_types;
pub mod gamma_client;
pub mod market_cache;
pub mod news_velocity;
pub mod openrouter_client;
pub mod resolution_monitor;
pub mod rss_client;
pub mod twitter_client;
pub mod webhook_server;
pub mod ws_manager;

pub use event_bus::BroadcastEventBus;
pub use feed_aggregator::FeedAggregator;
pub use feed_types::{FeedConfig, FeedError, FeedItem, FeedSource, RssConfig, RssFeedConfig, TwitterConfig, TwitterQueryConfig};
pub use gamma_client::GammaClient;
pub use market_cache::MarketCache;
pub use openrouter_client::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, OpenRouterClient,
    OpenRouterConfig, ResponseFormat, ResponseMessage, Usage,
};
pub use resolution_monitor::ResolutionMonitor;
pub use rss_client::RssClient;
pub use twitter_client::TwitterClient;
pub use webhook_server::WebhookServer;
pub use ws_manager::WsManager;
pub use news_velocity::{
    ArticleTimestamp, KeywordTracking, NewsCategory, NewsVelocityConfig, NewsVelocityTracker,
    VelocityMetrics, VelocityWindows,
};
