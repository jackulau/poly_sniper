//! Polysniper Data
//!
//! Data ingestion layer for WebSocket and REST API clients.

pub mod connection_health;
pub mod crypto_price_client;
pub mod connection_pool;
pub mod event_bus;
pub mod event_bus_fast;
pub mod external_markets;
pub mod feed_aggregator;
pub mod feed_types;
pub mod gamma_client;
pub mod http_pool;
pub mod market_cache;
pub mod news_velocity;
pub mod openrouter_client;
pub mod providers;
pub mod polymarket_activity;
pub mod prediction_aggregator;
pub mod resolution_monitor;
pub mod rss_client;
pub mod twitter_client;
pub mod webhook_server;
pub mod ws_manager;

pub use connection_health::{ConnectionHealth, HealthSnapshot, HealthStatus};
pub use crypto_price_client::{CryptoApiProvider, CryptoPrice, CryptoPriceClient, CryptoPriceConfig};
pub use connection_pool::{
    ClobConnection, ConnectionPool, ConnectionPoolConfig, HealthTracker, LatencyStats,
    RpcConnection,
};
pub use event_bus::BroadcastEventBus;
pub use event_bus_fast::{
    AsyncEventSubscription, BackpressureConfig, EventBusMetrics, EventSubscription,
    EventSubscriptionExt, LockFreeEventBus, UnifiedEventBus,
};
pub use external_markets::{
    ExternalMarketError, ExternalMarketPrice, KalshiClient, KalshiConfig, KalshiEvent,
    KalshiMarket, MetaculusClient, MetaculusConfig, MetaculusPrediction, Platform,
    PredictItClient, PredictItConfig, PredictItContract, PredictItMarket,
};
pub use feed_aggregator::FeedAggregator;
pub use feed_types::{FeedConfig, FeedError, FeedItem, FeedSource, RssConfig, RssFeedConfig, TwitterConfig, TwitterQueryConfig};
pub use gamma_client::GammaClient;
pub use http_pool::{HttpPool, HttpPoolConfig, HttpPoolError, WarmupResult};
pub use market_cache::MarketCache;
pub use market_cache::{LargeOrder, MarketCache, OrderbookHistory, OrderbookSnapshot};
pub use openrouter_client::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, OpenRouterClient,
    OpenRouterConfig, ResponseFormat, ResponseMessage, Usage,
};
pub use providers::{BoxedProvider, LlmProvider, OpenRouterProvider, ProviderError, ProviderFactory};
pub use polymarket_activity::{
    PolymarketActivityClient, PolymarketActivityConfig, TraderPosition, TraderProfile,
    VolumeSnapshot,
};
pub use prediction_aggregator::{
    AggregatedPrice, ArbitrageOpportunity, ArbitrageType, MarketMapping, PlatformWeights,
    PredictionAggregator, PredictionAggregatorConfig, Side,
};
pub use resolution_monitor::ResolutionMonitor;
pub use rss_client::RssClient;
pub use twitter_client::TwitterClient;
pub use webhook_server::WebhookServer;
pub use ws_manager::{ConnectionConfig, WsManager};
pub use ws_manager::{WsManager, WsManagerConfig};
pub use ws_manager::WsManager;
pub use news_velocity::{
    ArticleTimestamp, KeywordTracking, NewsCategory, NewsVelocityConfig, NewsVelocityTracker,
    VelocityMetrics, VelocityWindows,
};
