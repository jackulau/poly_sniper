use crate::types::{CorrelationRegime, Market, MarketId, Orderbook, Position, QueuePosition, TokenId, TradeSignal};
use crate::types::{Market, MarketId, Orderbook, Position, QueuePosition, Side, TokenId, TradeSignal};
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

    /// Correlation regime changed
    CorrelationRegimeChange(CorrelationRegimeChangeEvent),
    /// Crypto price update from external market
    CryptoPriceUpdate(CryptoPriceUpdateEvent),
    /// Microstructure analysis events (VPIN, whale detection, market impact)
    Microstructure(MicrostructureEvent),
    /// News velocity signal (acceleration/deceleration in coverage)
    NewsVelocitySignal(NewsVelocitySignalEvent),
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
            SystemEvent::CorrelationRegimeChange(_) => "correlation_regime_change",
            SystemEvent::CryptoPriceUpdate(_) => "crypto_price_update",
            SystemEvent::Microstructure(e) => e.event_type(),
            SystemEvent::NewsVelocitySignal(_) => "news_velocity_signal",
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
            SystemEvent::CorrelationRegimeChange(e) => e.timestamp,
            SystemEvent::CryptoPriceUpdate(e) => e.timestamp,
            SystemEvent::Microstructure(e) => e.timestamp(),
            SystemEvent::NewsVelocitySignal(e) => e.timestamp,
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
            SystemEvent::CorrelationRegimeChange(_) => None,
            SystemEvent::CryptoPriceUpdate(_) => None,
            SystemEvent::Microstructure(e) => e.market_id(),
            SystemEvent::NewsVelocitySignal(e) => e.market_ids.first(),
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
        self.previous_condition.is_some_and(|prev| prev != self.condition)
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

/// Correlation regime change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationRegimeChangeEvent {
    /// Previous regime
    pub old_regime: CorrelationRegime,
    /// New regime
    pub new_regime: CorrelationRegime,
    /// Current average correlation across position pairs
    pub avg_correlation: Decimal,
    /// Short-term average correlation
    pub short_term_avg: Option<Decimal>,
    /// Long-term average correlation
    pub long_term_avg: Option<Decimal>,
    /// New exposure limit multiplier
    pub limit_multiplier: Decimal,
// ============================================================================
// Microstructure Events
// ============================================================================

/// Wrapper for all microstructure-related events (VPIN, whale detection, market impact)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MicrostructureEvent {
    /// VPIN calculation update
    VpinUpdate(VpinUpdateEvent),

    /// Whale activity detected
    WhaleDetected(WhaleDetectedEvent),

    /// Market impact prediction
    ImpactPrediction(ImpactPredictionEvent),

    /// Toxicity level change (crossing threshold)
    ToxicityChange(ToxicityChangeEvent),

    /// Combined microstructure signal for strategy use
    MicrostructureSignal(MicrostructureSignalEvent),
}

impl MicrostructureEvent {
    /// Get the event type as a string
    pub fn event_type(&self) -> &'static str {
        match self {
            MicrostructureEvent::VpinUpdate(_) => "vpin_update",
            MicrostructureEvent::WhaleDetected(_) => "whale_detected",
            MicrostructureEvent::ImpactPrediction(_) => "impact_prediction",
            MicrostructureEvent::ToxicityChange(_) => "toxicity_change",
            MicrostructureEvent::MicrostructureSignal(_) => "microstructure_signal",
        }
    }

    /// Get the timestamp of the event
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            MicrostructureEvent::VpinUpdate(e) => e.timestamp,
            MicrostructureEvent::WhaleDetected(e) => e.timestamp,
            MicrostructureEvent::ImpactPrediction(e) => e.timestamp,
            MicrostructureEvent::ToxicityChange(e) => e.timestamp,
            MicrostructureEvent::MicrostructureSignal(e) => e.timestamp,
        }
    }

    /// Get the market ID if applicable
    pub fn market_id(&self) -> Option<&MarketId> {
        match self {
            MicrostructureEvent::VpinUpdate(e) => Some(&e.market_id),
            MicrostructureEvent::WhaleDetected(e) => Some(&e.market_id),
            MicrostructureEvent::ImpactPrediction(e) => Some(&e.market_id),
            MicrostructureEvent::ToxicityChange(e) => Some(&e.market_id),
            MicrostructureEvent::MicrostructureSignal(e) => Some(&e.market_id),
        }
    }

    /// Get the token ID if applicable
    pub fn token_id(&self) -> Option<&TokenId> {
        match self {
            MicrostructureEvent::VpinUpdate(e) => Some(&e.token_id),
            MicrostructureEvent::WhaleDetected(e) => Some(&e.token_id),
            MicrostructureEvent::ImpactPrediction(e) => Some(&e.token_id),
            MicrostructureEvent::ToxicityChange(e) => Some(&e.token_id),
            MicrostructureEvent::MicrostructureSignal(e) => Some(&e.token_id),
        }
    }
}

/// Toxicity level classification based on VPIN value
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToxicityLevel {
    /// VPIN < low_threshold (e.g., < 0.3) - favorable trading conditions
    Low,
    /// low_threshold <= VPIN < 0.5 - normal market conditions
    Normal,
    /// 0.5 <= VPIN < high_threshold - elevated informed trading activity
    Elevated,
    /// VPIN >= high_threshold (e.g., >= 0.7) - high informed trading probability
    High,
}

impl std::fmt::Display for ToxicityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToxicityLevel::Low => write!(f, "Low"),
            ToxicityLevel::Normal => write!(f, "Normal"),
            ToxicityLevel::Elevated => write!(f, "Elevated"),
            ToxicityLevel::High => write!(f, "High"),
        }
    }
}

/// Type of whale alert
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhaleAlertType {
    /// Single large trade detected
    SingleLargeTrade,
    /// Cumulative activity threshold exceeded
    CumulativeActivity,
    /// Known whale address is active
    KnownWhaleActive,
    /// Whale changed trading direction
    WhaleReversal,
}

/// Classification of a whale address based on trading patterns
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhaleClassification {
    /// Unknown pattern - not enough data
    #[default]
    Unknown,
    /// Consistently buys - accumulating position
    Accumulator,
    /// Consistently sells - distributing position
    Distributor,
    /// Quick in/out trades - short-term trader
    Flipper,
    /// High win rate trader - informed trader
    Informed,
}

/// Recommended action based on whale activity
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhaleAction {
    /// No action needed
    None,
    /// Reduce position by multiplier
    ReducePosition {
        /// Multiplier to apply to position size (e.g., 0.5 = reduce to 50%)
        multiplier: Decimal,
    },
    /// Stop opening new trades
    HaltNewTrades,
    /// Follow the whale's direction
    FollowWhale {
        /// Direction to follow
        direction: Side,
    },
    /// Generate alert only
    Alert,
}

/// VPIN calculation update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpinUpdateEvent {
    /// Token ID for this calculation
    pub token_id: TokenId,
    /// Market ID
    pub market_id: MarketId,
    /// VPIN value (0.0 to 1.0)
    pub vpin: Decimal,
    /// Toxicity level classification
    pub toxicity_level: ToxicityLevel,
    /// Percentage of volume from buy-initiated trades
    pub buy_volume_pct: Decimal,
    /// Percentage of volume from sell-initiated trades
    pub sell_volume_pct: Decimal,
    /// Number of buckets used in calculation
    pub bucket_count: usize,
    /// When this calculation was made
    pub timestamp: DateTime<Utc>,
}

impl VpinUpdateEvent {
    /// Create a new VPIN update event
    pub fn new(
        token_id: TokenId,
        market_id: MarketId,
        vpin: Decimal,
        toxicity_level: ToxicityLevel,
        buy_volume_pct: Decimal,
        sell_volume_pct: Decimal,
        bucket_count: usize,
    ) -> Self {
        Self {
            token_id,
            market_id,
            vpin,
            toxicity_level,
            buy_volume_pct,
            sell_volume_pct,
            bucket_count,
            timestamp: Utc::now(),
        }
    }
}

/// Whale activity detected event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleDetectedEvent {
    /// Token involved
    pub token_id: TokenId,
    /// Market involved
    pub market_id: MarketId,
    /// Type of whale alert
    pub alert_type: WhaleAlertType,
    /// Trade size in USD that triggered this alert
    pub trade_size_usd: Decimal,
    /// Cumulative size in USD if applicable
    pub cumulative_size_usd: Option<Decimal>,
    /// Side of the whale trade
    pub side: Side,
    /// Wallet address if available
    pub address: Option<String>,
    /// Classification of the whale if known
    pub classification: Option<WhaleClassification>,
    /// Recommended action
    pub recommended_action: WhaleAction,
    /// Confidence in the recommendation (0.0 to 1.0)
    pub confidence: Decimal,
    /// When the alert was generated
    pub timestamp: DateTime<Utc>,
}

impl WhaleDetectedEvent {
    /// Create a new whale detected event
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        token_id: TokenId,
        market_id: MarketId,
        alert_type: WhaleAlertType,
        trade_size_usd: Decimal,
        side: Side,
        recommended_action: WhaleAction,
        confidence: Decimal,
    ) -> Self {
        Self {
            token_id,
            market_id,
            alert_type,
            trade_size_usd,
            cumulative_size_usd: None,
            side,
            address: None,
            classification: None,
            recommended_action,
            confidence,
            timestamp: Utc::now(),
        }
    }

    /// Set cumulative size for aggregate whale detection
    pub fn with_cumulative_size(mut self, cumulative_size_usd: Decimal) -> Self {
        self.cumulative_size_usd = Some(cumulative_size_usd);
        self
    }

    /// Set address of the whale
    pub fn with_address(mut self, address: String) -> Self {
        self.address = Some(address);
        self
    }

    /// Set classification of the whale
    pub fn with_classification(mut self, classification: WhaleClassification) -> Self {
        self.classification = Some(classification);
        self
    }
}

/// Market impact prediction event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactPredictionEvent {
    /// Token being traded
    pub token_id: TokenId,
    /// Market ID
    pub market_id: MarketId,
    /// Proposed trade size in USD
    pub proposed_size_usd: Decimal,
    /// Expected total impact in basis points
    pub expected_impact_bps: Decimal,
    /// Expected temporary impact in basis points
    pub temporary_impact_bps: Decimal,
    /// Expected permanent impact in basis points
    pub permanent_impact_bps: Decimal,
    /// Expected time for temporary impact to recover (seconds)
    pub expected_recovery_secs: u64,
    /// Confidence in the prediction (0.0 to 1.0)
    pub model_confidence: Decimal,
    /// When the prediction was made
    pub timestamp: DateTime<Utc>,
}

impl ImpactPredictionEvent {
    /// Create a new impact prediction event
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        token_id: TokenId,
        market_id: MarketId,
        proposed_size_usd: Decimal,
        expected_impact_bps: Decimal,
        temporary_impact_bps: Decimal,
        permanent_impact_bps: Decimal,
        expected_recovery_secs: u64,
        model_confidence: Decimal,
    ) -> Self {
        Self {
            token_id,
            market_id,
            proposed_size_usd,
            expected_impact_bps,
            temporary_impact_bps,
            permanent_impact_bps,
            expected_recovery_secs,
            model_confidence,
            timestamp: Utc::now(),
        }
    }
}

/// Toxicity level change event (crossing threshold)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToxicityChangeEvent {
    /// Token ID
    pub token_id: TokenId,
    /// Market ID
    pub market_id: MarketId,
    /// Previous toxicity level
    pub previous_level: ToxicityLevel,
    /// New toxicity level
    pub new_level: ToxicityLevel,
    /// Current VPIN value
    pub vpin: Decimal,
    /// Reason for the trigger
    pub trigger_reason: String,
    /// When the change occurred
    pub timestamp: DateTime<Utc>,
}

impl CorrelationRegimeChangeEvent {
    /// Create a new correlation regime change event
    pub fn new(
        old_regime: CorrelationRegime,
        new_regime: CorrelationRegime,
        avg_correlation: Decimal,
        short_term_avg: Option<Decimal>,
        long_term_avg: Option<Decimal>,
        limit_multiplier: Decimal,
    ) -> Self {
        Self {
            old_regime,
            new_regime,
            avg_correlation,
            short_term_avg,
            long_term_avg,
            limit_multiplier,
            timestamp: Utc::now(),
        }
    }
/// Crypto price update event from external markets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoPriceUpdateEvent {
    /// Crypto symbol (e.g., "ETH", "BTC", "SOL")
    pub symbol: String,
    /// Current price in USD
    pub price: Decimal,
    /// 1-hour price change percentage
    pub price_change_1h: Decimal,
    /// 24-hour price change percentage
    pub price_change_24h: Decimal,
    /// 24-hour trading volume in USD
    pub volume_24h: Decimal,
    /// When this update occurred
    pub timestamp: DateTime<Utc>,
}

impl CryptoPriceUpdateEvent {
    /// Create a new crypto price update event
    pub fn new(
        symbol: String,
        price: Decimal,
        price_change_1h: Decimal,
        price_change_24h: Decimal,
        volume_24h: Decimal,
    ) -> Self {
        Self {
            symbol,
            price,
            price_change_1h,
            price_change_24h,
            volume_24h,
impl ToxicityChangeEvent {
    /// Create a new toxicity change event
    pub fn new(
        token_id: TokenId,
        market_id: MarketId,
        previous_level: ToxicityLevel,
        new_level: ToxicityLevel,
        vpin: Decimal,
        trigger_reason: String,
    ) -> Self {
        Self {
            token_id,
            market_id,
            previous_level,
            new_level,
            vpin,
            trigger_reason,
            timestamp: Utc::now(),
        }
    }

    /// Check if toxicity increased
    pub fn is_increase(&self) -> bool {
        self.level_to_score(self.new_level) > self.level_to_score(self.previous_level)
    }

    /// Check if toxicity decreased
    pub fn is_decrease(&self) -> bool {
        self.level_to_score(self.new_level) < self.level_to_score(self.previous_level)
    }

    fn level_to_score(&self, level: ToxicityLevel) -> u8 {
        match level {
            ToxicityLevel::Low => 0,
            ToxicityLevel::Normal => 1,
            ToxicityLevel::Elevated => 2,
            ToxicityLevel::High => 3,
        }
    }
}

/// Type of combined microstructure signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MicrostructureSignalType {
    /// Good conditions for trading
    Favorable,
    /// Poor conditions, reduce activity
    Unfavorable,
    /// Follow detected whale
    WhaleFollow,
    /// Avoid trading against whale
    WhaleAvoid,
    /// Elevated informed trading detected
    HighToxicity,
}

/// Summary of whale activity for signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleActivitySummary {
    /// Number of recent whale trades
    pub recent_whale_trades: u32,
    /// Net whale direction
    pub net_whale_direction: Side,
    /// Total whale volume in USD
    pub total_whale_volume_usd: Decimal,
}

/// Components that contributed to a microstructure signal
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MicrostructureComponents {
    /// VPIN value if available
    pub vpin: Option<Decimal>,
    /// Toxicity level if available
    pub toxicity_level: Option<ToxicityLevel>,
    /// Whale activity summary if available
    pub whale_activity: Option<WhaleActivitySummary>,
    /// Expected market impact in basis points if available
    pub expected_impact_bps: Option<Decimal>,
}

/// Recommended action based on microstructure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MicrostructureAction {
    /// No action needed
    None,
    /// Reduce position/order size by multiplier
    ReduceSize {
        /// Multiplier to apply (e.g., 0.5 = reduce to 50%)
        multiplier: Decimal,
    },
    /// Halt all trading temporarily
    HaltTrading,
    /// Increase position/order size (favorable conditions)
    IncreaseSize {
        /// Multiplier to apply (e.g., 1.5 = increase by 50%)
        multiplier: Decimal,
    },
    /// Urgent action required
    Urgent {
        /// Reason for urgency
        reason: String,
    },
}

/// Combined microstructure signal for strategy use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureSignalEvent {
    /// Token ID
    pub token_id: TokenId,
    /// Market ID
    pub market_id: MarketId,
    /// Type of signal
    pub signal_type: MicrostructureSignalType,
    /// Signal strength (-1.0 to 1.0, bearish to bullish)
    pub strength: Decimal,
    /// Confidence in the signal (0.0 to 1.0)
    pub confidence: Decimal,
    /// Components that contributed to this signal
    pub components: MicrostructureComponents,
    /// Recommended action
    pub recommended_action: MicrostructureAction,
/// Direction of news velocity change
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VelocityDirection {
    /// Coverage increasing rapidly (breaking news)
    Accelerating,
    /// Coverage decreasing (story fading)
    Decelerating,
    /// Normal/stable coverage
    Stable,
}

impl VelocityDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            VelocityDirection::Accelerating => "accelerating",
            VelocityDirection::Decelerating => "decelerating",
            VelocityDirection::Stable => "stable",
        }
    }
}

impl std::fmt::Display for VelocityDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// News velocity signal event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsVelocitySignalEvent {
    /// Keyword that triggered this signal
    pub keyword: String,
    /// Market IDs mapped to this keyword
    pub market_ids: Vec<MarketId>,
    /// Direction of velocity change
    pub direction: VelocityDirection,
    /// Current velocity (articles per hour)
    pub current_velocity: Decimal,
    /// Historical baseline velocity
    pub baseline_velocity: Decimal,
    /// Acceleration factor (current / baseline)
    pub acceleration: Decimal,
    /// Number of articles in the last hour
    pub article_count_1h: u32,
    /// Number of articles in the last 24 hours
    pub article_count_24h: u32,
    /// Sample headlines for context
    pub sample_headlines: Vec<String>,
    /// When the signal was generated
    pub timestamp: DateTime<Utc>,
}

impl MicrostructureSignalEvent {
    /// Create a new microstructure signal event
    pub fn new(
        token_id: TokenId,
        market_id: MarketId,
        signal_type: MicrostructureSignalType,
        strength: Decimal,
        confidence: Decimal,
        components: MicrostructureComponents,
        recommended_action: MicrostructureAction,
    ) -> Self {
        Self {
            token_id,
            market_id,
            signal_type,
            strength,
            confidence,
            components,
            recommended_action,
            timestamp: Utc::now(),
        }
    }

    /// Check if this represents a significant price movement
    pub fn is_significant_move(&self, threshold_pct: Decimal) -> bool {
        self.price_change_1h.abs() >= threshold_pct
    /// Check if conditions are favorable for trading
    pub fn is_favorable(&self) -> bool {
        matches!(self.signal_type, MicrostructureSignalType::Favorable)
    }

    /// Check if conditions are unfavorable for trading
    pub fn is_unfavorable(&self) -> bool {
        matches!(
            self.signal_type,
            MicrostructureSignalType::Unfavorable | MicrostructureSignalType::HighToxicity
        )
    }
impl NewsVelocitySignalEvent {
    /// Create a new news velocity signal event
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        keyword: String,
        market_ids: Vec<MarketId>,
        direction: VelocityDirection,
        current_velocity: Decimal,
        baseline_velocity: Decimal,
        acceleration: Decimal,
        article_count_1h: u32,
        article_count_24h: u32,
        sample_headlines: Vec<String>,
    ) -> Self {
        Self {
            keyword,
            market_ids,
            direction,
            current_velocity,
            baseline_velocity,
            acceleration,
            article_count_1h,
            article_count_24h,
            sample_headlines,
            timestamp: Utc::now(),
        }
    }
}
