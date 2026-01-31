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

    /// Full order fill
    FullFill(FullFillEvent),

    /// Partial order fill
    PartialFill(PartialFillEvent),

    /// Order expired
    OrderExpired(OrderExpiredEvent),

    /// Order resubmit triggered
    ResubmitTriggered(ResubmitTriggeredEvent),

    /// Gas price update
    GasPriceUpdate(GasPriceUpdateEvent),

    /// Order replaced
    OrderReplaced(OrderReplacedEvent),

    /// Smart money signal from top trader activity
    SmartMoneySignal(SmartMoneySignalEvent),

    /// Volume anomaly detected for a market
    VolumeAnomalyDetected(VolumeAnomalyEvent),

    /// Comment activity spike detected
    CommentActivitySpike(CommentActivityEvent),
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
            SystemEvent::FullFill(_) => "full_fill",
            SystemEvent::PartialFill(_) => "partial_fill",
            SystemEvent::OrderExpired(_) => "order_expired",
            SystemEvent::ResubmitTriggered(_) => "resubmit_triggered",
            SystemEvent::GasPriceUpdate(_) => "gas_price_update",
            SystemEvent::OrderReplaced(_) => "order_replaced",
            SystemEvent::SmartMoneySignal(_) => "smart_money_signal",
            SystemEvent::VolumeAnomalyDetected(_) => "volume_anomaly_detected",
            SystemEvent::CommentActivitySpike(_) => "comment_activity_spike",
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
            SystemEvent::FullFill(e) => e.timestamp,
            SystemEvent::PartialFill(e) => e.timestamp,
            SystemEvent::OrderExpired(e) => e.timestamp,
            SystemEvent::ResubmitTriggered(e) => e.timestamp,
            SystemEvent::GasPriceUpdate(e) => e.timestamp,
            SystemEvent::OrderReplaced(e) => e.timestamp,
            SystemEvent::SmartMoneySignal(e) => e.timestamp,
            SystemEvent::VolumeAnomalyDetected(e) => e.timestamp,
            SystemEvent::CommentActivitySpike(e) => e.timestamp,
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
            SystemEvent::FullFill(e) => Some(&e.market_id),
            SystemEvent::PartialFill(e) => Some(&e.market_id),
            SystemEvent::OrderExpired(e) => Some(&e.market_id),
            SystemEvent::ResubmitTriggered(e) => Some(&e.market_id),
            SystemEvent::GasPriceUpdate(_) => None,
            SystemEvent::OrderReplaced(e) => Some(&e.market_id),
            SystemEvent::SmartMoneySignal(e) => Some(&e.market_id),
            SystemEvent::VolumeAnomalyDetected(e) => Some(&e.market_id),
            SystemEvent::CommentActivitySpike(e) => Some(&e.market_id),
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

/// Fill information for an order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    /// Fill size
    pub size: Decimal,
    /// Fill price
    pub price: Decimal,
    /// When the fill occurred
    pub timestamp: DateTime<Utc>,
}

/// Full order fill event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullFillEvent {
    /// Order ID that was fully filled
    pub order_id: String,
    /// Market ID
    pub market_id: MarketId,
    /// Token ID
    pub token_id: TokenId,
    /// Volume-weighted average price across all fills
    pub avg_price: Decimal,
    /// Total size filled
    pub total_size: Decimal,
    /// Number of individual fills
    pub fill_count: usize,
    /// When the fill was detected
    pub timestamp: DateTime<Utc>,
}

/// Partial order fill event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialFillEvent {
    /// Order ID that was partially filled
    pub order_id: String,
    /// Market ID
    pub market_id: MarketId,
    /// Token ID
    pub token_id: TokenId,
    /// Fill details
    pub fill: Fill,
    /// Total filled size so far
    pub total_filled: Decimal,
    /// Remaining size after this fill
    pub remaining: Decimal,
    /// When the fill was detected
    pub timestamp: DateTime<Utc>,
}

/// Order expired event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderExpiredEvent {
    /// Order ID that expired
    pub order_id: String,
    /// Market ID
    pub market_id: MarketId,
    /// Token ID
    pub token_id: TokenId,
    /// Total size filled before expiry
    pub filled: Decimal,
    /// Size that was unfilled
    pub unfilled: Decimal,
    /// When the expiry was detected
    pub timestamp: DateTime<Utc>,
}

/// Resubmit triggered event for partial fills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResubmitTriggeredEvent {
    /// Original order ID
    pub original_order_id: String,
    /// New order being submitted
    pub new_order: crate::Order,
    /// Market ID
    pub market_id: MarketId,
    /// Remaining size being resubmitted
    pub remaining_size: Decimal,
    /// Which resubmit attempt this is
    pub resubmit_attempt: u32,
    /// When the resubmit was triggered
    pub timestamp: DateTime<Utc>,
}

/// Gas price update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasPriceUpdateEvent {
    /// Current gas price
    pub gas_price: crate::GasPrice,
    /// Previous gas price (if available)
    pub previous_price: Option<crate::GasPrice>,
    /// Current gas condition
    pub condition: crate::GasCondition,
    /// Previous gas condition (if available)
    pub previous_condition: Option<crate::GasCondition>,
    /// Whether this is a spike
    pub is_spike: bool,
    /// Average gas price in gwei (for comparison)
    pub average_gwei: Option<Decimal>,
    /// When the update occurred
    pub timestamp: DateTime<Utc>,
}

impl GasPriceUpdateEvent {
    /// Create a new gas price update event
    pub fn new(
        gas_price: crate::GasPrice,
        previous_price: Option<crate::GasPrice>,
        condition: crate::GasCondition,
        previous_condition: Option<crate::GasCondition>,
        is_spike: bool,
        average_gwei: Option<Decimal>,
    ) -> Self {
        Self {
            gas_price,
            previous_price,
            condition,
            previous_condition,
            is_spike,
            average_gwei,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Check if the gas condition changed
    pub fn condition_changed(&self) -> bool {
        self.previous_condition.map_or(false, |prev| prev != self.condition)
    }
}

/// Order replaced event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderReplacedEvent {
    /// Original order ID being replaced
    pub original_order_id: String,
    /// New order ID
    pub new_order_id: String,
    /// Market ID
    pub market_id: MarketId,
    /// Old price
    pub old_price: Decimal,
    /// New price
    pub new_price: Decimal,
    /// Filled size preserved from original order
    pub preserved_fill: Decimal,
    /// Reason for replacement
    pub reason: String,
    /// When the replacement occurred
    pub timestamp: DateTime<Utc>,
}

/// Type of action taken by a trader
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraderAction {
    /// Buying into a position
    Buy,
    /// Selling out of a position
    Sell,
    /// Opening a new position
    NewPosition,
    /// Closing an existing position
    ClosePosition,
}

/// Smart money signal event from top trader activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartMoneySignalEvent {
    /// Market ID
    pub market_id: MarketId,
    /// Token ID
    pub token_id: TokenId,
    /// Trader's address
    pub trader_address: String,
    /// Trader's username if available
    pub trader_username: Option<String>,
    /// Trader's leaderboard rank
    pub trader_rank: u32,
    /// Trader's total profit/PnL
    pub trader_profit: Decimal,
    /// Action taken by the trader
    pub action: TraderAction,
    /// Outcome being traded
    pub outcome: crate::types::Outcome,
    /// Size of the position in USD
    pub size_usd: Decimal,
    /// When the signal was generated
    pub timestamp: DateTime<Utc>,
}

/// Volume anomaly detected event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeAnomalyEvent {
    /// Market ID
    pub market_id: MarketId,
    /// Current volume in USD
    pub current_volume: Decimal,
    /// Average volume over lookback period
    pub avg_volume: Decimal,
    /// Ratio of current to average volume
    pub volume_ratio: Decimal,
    /// Number of trades in the period
    pub trade_count: u32,
    /// Net flow direction: positive = more buying, negative = more selling
    pub net_flow: Option<Decimal>,
    /// When the anomaly was detected
    pub timestamp: DateTime<Utc>,
}

/// Comment activity spike event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommentActivityEvent {
    /// Market ID
    pub market_id: MarketId,
    /// Number of comments in the period
    pub comment_count: u32,
    /// Comments per hour velocity
    pub comment_velocity: Decimal,
    /// Brief sentiment hint from comments (if available)
    pub sentiment_hint: Option<String>,
    /// When the spike was detected
    pub timestamp: DateTime<Utc>,
}
