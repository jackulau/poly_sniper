use crate::types::{Market, MarketId, Orderbook, Position, QueuePosition, TokenId, TradeSignal};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// System-wide event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemEvent {
    /// Price change event from RTDS or orderbook
    PriceChange(PriceChangeEvent),

    /// Orderbook update event
    OrderbookUpdate(OrderbookUpdateEvent),

    /// New market discovered
    NewMarket(NewMarketEvent),

    /// Market state changed
    MarketStateChange(MarketStateChangeEvent),

    /// External signal (webhook, RSS, etc.)
    ExternalSignal(ExternalSignalEvent),

    /// Position update
    PositionUpdate(PositionUpdateEvent),

    /// Trade executed
    TradeExecuted(TradeExecutedEvent),

    /// Queue position update for a tracked order
    QueueUpdate(QueueUpdateEvent),

    /// Connection status change
    ConnectionStatus(ConnectionStatusEvent),

    /// Heartbeat for health checks
    Heartbeat(HeartbeatEvent),

    /// Feed item received from Twitter, RSS, etc.
    FeedItemReceived(FeedItemReceivedEvent),

    /// Config file changed
    ConfigChanged(ConfigChangedEvent),
}

impl SystemEvent {
    /// Get the event type as a string
    pub fn event_type(&self) -> &'static str {
        match self {
            SystemEvent::PriceChange(_) => "price_change",
            SystemEvent::OrderbookUpdate(_) => "orderbook_update",
            SystemEvent::NewMarket(_) => "new_market",
            SystemEvent::MarketStateChange(_) => "market_state_change",
            SystemEvent::ExternalSignal(_) => "external_signal",
            SystemEvent::PositionUpdate(_) => "position_update",
            SystemEvent::TradeExecuted(_) => "trade_executed",
            SystemEvent::QueueUpdate(_) => "queue_update",
            SystemEvent::ConnectionStatus(_) => "connection_status",
            SystemEvent::Heartbeat(_) => "heartbeat",
            SystemEvent::FeedItemReceived(_) => "feed_item_received",
            SystemEvent::ConfigChanged(_) => "config_changed",
        }
    }

    /// Get the timestamp of the event
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            SystemEvent::PriceChange(e) => e.timestamp,
            SystemEvent::OrderbookUpdate(e) => e.timestamp,
            SystemEvent::NewMarket(e) => e.discovered_at,
            SystemEvent::MarketStateChange(e) => e.timestamp,
            SystemEvent::ExternalSignal(e) => e.received_at,
            SystemEvent::PositionUpdate(e) => e.timestamp,
            SystemEvent::TradeExecuted(e) => e.timestamp,
            SystemEvent::QueueUpdate(e) => e.timestamp,
            SystemEvent::ConnectionStatus(e) => e.timestamp,
            SystemEvent::Heartbeat(e) => e.timestamp,
            SystemEvent::FeedItemReceived(e) => e.received_at,
            SystemEvent::ConfigChanged(e) => e.timestamp,
        }
    }

    /// Get the market ID if applicable
    pub fn market_id(&self) -> Option<&MarketId> {
        match self {
            SystemEvent::PriceChange(e) => Some(&e.market_id),
            SystemEvent::OrderbookUpdate(e) => Some(&e.market_id),
            SystemEvent::NewMarket(e) => Some(&e.market.condition_id),
            SystemEvent::MarketStateChange(e) => Some(&e.market_id),
            SystemEvent::ExternalSignal(e) => e.market_id.as_ref(),
            SystemEvent::PositionUpdate(e) => Some(&e.market_id),
            SystemEvent::TradeExecuted(e) => Some(&e.market_id),
            SystemEvent::QueueUpdate(e) => Some(&e.market_id),
            SystemEvent::ConnectionStatus(_) => None,
            SystemEvent::Heartbeat(_) => None,
            SystemEvent::FeedItemReceived(_) => None,
            SystemEvent::ConfigChanged(_) => None,
        }
    }
}

/// Price change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceChangeEvent {
    pub market_id: MarketId,
    pub token_id: TokenId,
    pub old_price: Option<Decimal>,
    pub new_price: Decimal,
    pub price_change_pct: Option<Decimal>,
    pub timestamp: DateTime<Utc>,
}

impl PriceChangeEvent {
    /// Create a new price change event
    pub fn new(
        market_id: MarketId,
        token_id: TokenId,
        old_price: Option<Decimal>,
        new_price: Decimal,
    ) -> Self {
        let price_change_pct = old_price.map(|old| {
            if old.is_zero() {
                Decimal::ZERO
            } else {
                ((new_price - old) / old) * Decimal::ONE_HUNDRED
            }
        });

        Self {
            market_id,
            token_id,
            old_price,
            new_price,
            price_change_pct,
            timestamp: Utc::now(),
        }
    }
}

/// Orderbook update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookUpdateEvent {
    pub market_id: MarketId,
    pub token_id: TokenId,
    pub orderbook: Orderbook,
    pub timestamp: DateTime<Utc>,
}

/// New market discovered event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMarketEvent {
    pub market: Market,
    pub discovered_at: DateTime<Utc>,
}

/// Market state change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStateChangeEvent {
    pub market_id: MarketId,
    pub old_state: MarketState,
    pub new_state: MarketState,
    pub timestamp: DateTime<Utc>,
}

/// Market states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketState {
    Active,
    Paused,
    Closed,
    Resolved,
}

/// External signal event (webhooks, RSS, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalSignalEvent {
    pub source: SignalSource,
    pub signal_type: String,
    pub content: String,
    pub market_id: Option<MarketId>,
    pub keywords: Vec<String>,
    pub metadata: serde_json::Value,
    pub received_at: DateTime<Utc>,
}

/// Source of external signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalSource {
    Webhook { endpoint: String },
    Rss { feed_url: String },
    Twitter { account: String },
    Custom { name: String },
}

/// Position update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionUpdateEvent {
    pub market_id: MarketId,
    pub position: Position,
    pub timestamp: DateTime<Utc>,
}

/// Trade executed event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeExecutedEvent {
    pub order_id: String,
    pub signal: TradeSignal,
    pub market_id: MarketId,
    pub token_id: TokenId,
    pub executed_price: Decimal,
    pub executed_size: Decimal,
    pub fees: Decimal,
    pub timestamp: DateTime<Utc>,
}

/// Connection status event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStatusEvent {
    pub source: String,
    pub status: ConnectionState,
    pub message: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Connection states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionState {
    Connected,
    Disconnected,
    Reconnecting,
    Error,
}

/// Heartbeat event for health checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatEvent {
    pub source: String,
    pub timestamp: DateTime<Utc>,
}

/// Source of a feed item
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FeedItemSource {
    Twitter {
        account: Option<String>,
        query: Option<String>,
    },
    Rss {
        feed_url: String,
        feed_title: Option<String>,
    },
}

impl FeedItemSource {
    pub fn source_name(&self) -> &'static str {
        match self {
            FeedItemSource::Twitter { .. } => "twitter",
            FeedItemSource::Rss { .. } => "rss",
        }
    }
}

/// A feed item from any source (Twitter, RSS, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedItem {
    /// Unique identifier for this item (from source)
    pub id: String,
    /// Source of this feed item
    pub source: FeedItemSource,
    /// Main content/text of the item
    pub content: String,
    /// Title (for RSS items)
    pub title: Option<String>,
    /// Author/username
    pub author: Option<String>,
    /// URL to the original content
    pub url: Option<String>,
    /// When the content was published
    pub published_at: DateTime<Utc>,
    /// When we received this item
    pub received_at: DateTime<Utc>,
    /// Content hash for deduplication
    pub content_hash: String,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Matched keywords that triggered this item
    pub matched_keywords: Vec<String>,
}

/// Feed item received event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedItemReceivedEvent {
    /// The feed item that was received
    pub item: FeedItem,
    /// When the item was received
    pub received_at: DateTime<Utc>,
}

/// Queue position update event for a tracked order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueUpdateEvent {
    /// Order ID being tracked
    pub order_id: String,
    /// Token ID for the order
    pub token_id: TokenId,
    /// Market ID for the order
    pub market_id: MarketId,
    /// Updated queue position information
    pub position: QueuePosition,
    /// When this update was calculated
    pub timestamp: DateTime<Utc>,
}

/// Type of configuration that changed
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfigType {
    /// Main application config (config/default.toml)
    Main,
    /// Strategy config (config/strategies/*.toml)
    Strategy(String),
}

/// Config file changed event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigChangedEvent {
    /// Path to the changed config file
    pub path: std::path::PathBuf,
    /// Type of config that changed
    pub config_type: ConfigType,
    /// When the change was detected
    pub timestamp: DateTime<Utc>,
}

impl ConfigChangedEvent {
    /// Create a new config changed event
    pub fn new(path: std::path::PathBuf, config_type: ConfigType) -> Self {
        Self {
            path,
            config_type,
            timestamp: Utc::now(),
        }
    }
}
