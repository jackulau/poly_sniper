use crate::types::{Market, MarketId, Order, Orderbook, Position, TokenId, TradeSignal};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

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

    /// Connection status change
    ConnectionStatus(ConnectionStatusEvent),

    /// Heartbeat for health checks
    Heartbeat(HeartbeatEvent),

    /// Partial fill event
    PartialFill(PartialFillEvent),

    /// Full fill event
    FullFill(FullFillEvent),

    /// Order expired event
    OrderExpired(OrderExpiredEvent),

    /// Order resubmit triggered
    ResubmitTriggered(ResubmitTriggeredEvent),
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
            SystemEvent::ConnectionStatus(_) => "connection_status",
            SystemEvent::Heartbeat(_) => "heartbeat",
            SystemEvent::PartialFill(_) => "partial_fill",
            SystemEvent::FullFill(_) => "full_fill",
            SystemEvent::OrderExpired(_) => "order_expired",
            SystemEvent::ResubmitTriggered(_) => "resubmit_triggered",
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
            SystemEvent::ConnectionStatus(e) => e.timestamp,
            SystemEvent::Heartbeat(e) => e.timestamp,
            SystemEvent::PartialFill(e) => e.timestamp,
            SystemEvent::FullFill(e) => e.timestamp,
            SystemEvent::OrderExpired(e) => e.timestamp,
            SystemEvent::ResubmitTriggered(e) => e.timestamp,
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
            SystemEvent::ConnectionStatus(_) => None,
            SystemEvent::Heartbeat(_) => None,
            SystemEvent::PartialFill(e) => Some(&e.market_id),
            SystemEvent::FullFill(e) => Some(&e.market_id),
            SystemEvent::OrderExpired(e) => Some(&e.market_id),
            SystemEvent::ResubmitTriggered(e) => Some(&e.market_id),
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

/// A single fill on an order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub size: Decimal,
    pub price: Decimal,
    pub timestamp: DateTime<Utc>,
}

/// Partial fill event - order partially executed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialFillEvent {
    pub order_id: String,
    pub market_id: MarketId,
    pub token_id: TokenId,
    pub fill: Fill,
    pub total_filled: Decimal,
    pub remaining: Decimal,
    pub timestamp: DateTime<Utc>,
}

/// Full fill event - order completely executed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullFillEvent {
    pub order_id: String,
    pub market_id: MarketId,
    pub token_id: TokenId,
    pub avg_price: Decimal,
    pub total_size: Decimal,
    pub fill_count: usize,
    pub timestamp: DateTime<Utc>,
}

/// Order expired event - order expired with unfilled portion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderExpiredEvent {
    pub order_id: String,
    pub market_id: MarketId,
    pub token_id: TokenId,
    pub filled: Decimal,
    pub unfilled: Decimal,
    pub timestamp: DateTime<Utc>,
}

/// Resubmit triggered event - new order created for remaining size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResubmitTriggeredEvent {
    pub original_order_id: String,
    pub new_order: Order,
    pub market_id: MarketId,
    pub remaining_size: Decimal,
    pub resubmit_attempt: u32,
    pub timestamp: DateTime<Utc>,
}
