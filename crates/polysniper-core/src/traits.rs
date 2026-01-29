use crate::error::{DataSourceError, ExecutionError, RiskError, StrategyError};
use crate::events::SystemEvent;
use crate::types::{Market, MarketId, Order, Orderbook, Position, TokenId, TradeSignal};
use async_trait::async_trait;
use rust_decimal::Decimal;
use tokio::sync::broadcast;

/// Strategy trait for implementing trading strategies
#[async_trait]
pub trait Strategy: Send + Sync {
    /// Unique identifier for this strategy
    fn id(&self) -> &str;

    /// Human-readable name
    fn name(&self) -> &str;

    /// Process an incoming event and potentially generate trade signals
    async fn process_event(
        &self,
        event: &SystemEvent,
        state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError>;

    /// Check if this strategy is interested in a particular event type
    fn accepts_event(&self, event: &SystemEvent) -> bool;

    /// Initialize the strategy with current state
    async fn initialize(&mut self, state: &dyn StateProvider) -> Result<(), StrategyError>;

    /// Shutdown the strategy gracefully
    async fn shutdown(&self) -> Result<(), StrategyError> {
        Ok(())
    }

    /// Check if the strategy is currently enabled
    fn is_enabled(&self) -> bool;

    /// Enable or disable the strategy
    fn set_enabled(&mut self, enabled: bool);
}

/// Data source trait for market data feeds
#[async_trait]
pub trait DataSource: Send + Sync {
    /// Connect to the data source
    async fn connect(&mut self) -> Result<(), DataSourceError>;

    /// Disconnect from the data source
    async fn disconnect(&mut self) -> Result<(), DataSourceError>;

    /// Get a receiver for system events
    fn subscribe(&self) -> broadcast::Receiver<SystemEvent>;

    /// Check if currently connected
    fn is_connected(&self) -> bool;

    /// Get the name of this data source
    fn name(&self) -> &str;
}

/// State provider trait for accessing market state
#[async_trait]
pub trait StateProvider: Send + Sync {
    /// Get a market by ID
    async fn get_market(&self, market_id: &MarketId) -> Option<Market>;

    /// Get all known markets
    async fn get_all_markets(&self) -> Vec<Market>;

    /// Get orderbook for a token
    async fn get_orderbook(&self, token_id: &TokenId) -> Option<Orderbook>;

    /// Get current price for a token
    async fn get_price(&self, token_id: &TokenId) -> Option<Decimal>;

    /// Get position for a market
    async fn get_position(&self, market_id: &MarketId) -> Option<Position>;

    /// Get all positions
    async fn get_all_positions(&self) -> Vec<Position>;

    /// Get price history for a token
    async fn get_price_history(
        &self,
        token_id: &TokenId,
        limit: usize,
    ) -> Vec<(chrono::DateTime<chrono::Utc>, Decimal)>;

    /// Get total portfolio value in USD
    async fn get_portfolio_value(&self) -> Decimal;

    /// Get daily P&L
    async fn get_daily_pnl(&self) -> Decimal;
}

/// State manager trait for updating market state
#[async_trait]
pub trait StateManager: StateProvider {
    /// Update market data
    async fn update_market(&self, market: Market);

    /// Update orderbook
    async fn update_orderbook(&self, orderbook: Orderbook);

    /// Update price
    async fn update_price(&self, token_id: TokenId, price: Decimal);

    /// Update position
    async fn update_position(&self, position: Position);

    /// Record a price snapshot for history
    async fn record_price_snapshot(&self, token_id: TokenId, price: Decimal);

    /// Clear all state
    async fn clear(&self);
}

/// Risk validator trait
#[async_trait]
pub trait RiskValidator: Send + Sync {
    /// Validate a trade signal against risk rules
    async fn validate(
        &self,
        signal: &TradeSignal,
        state: &dyn StateProvider,
    ) -> Result<RiskDecision, RiskError>;

    /// Check if trading is currently halted
    fn is_halted(&self) -> bool;

    /// Halt all trading (circuit breaker)
    fn halt(&self, reason: &str);

    /// Resume trading
    fn resume(&self);
}

/// Risk decision result
#[derive(Debug, Clone)]
pub enum RiskDecision {
    /// Signal is approved
    Approved,
    /// Signal is approved with modifications
    Modified {
        new_size: Decimal,
        reason: String,
    },
    /// Signal is rejected
    Rejected {
        reason: String,
    },
}

impl RiskDecision {
    pub fn is_approved(&self) -> bool {
        matches!(self, RiskDecision::Approved | RiskDecision::Modified { .. })
    }
}

/// Order executor trait
#[async_trait]
pub trait OrderExecutor: Send + Sync {
    /// Submit an order
    async fn submit_order(&self, order: Order) -> Result<String, ExecutionError>;

    /// Cancel an order
    async fn cancel_order(&self, order_id: &str) -> Result<(), ExecutionError>;

    /// Get order status
    async fn get_order_status(&self, order_id: &str) -> Result<OrderStatusResponse, ExecutionError>;

    /// Check if in dry run mode
    fn is_dry_run(&self) -> bool;
}

/// Order status response from executor
#[derive(Debug, Clone)]
pub struct OrderStatusResponse {
    pub order_id: String,
    pub status: crate::types::OrderStatus,
    pub filled_size: Decimal,
    pub remaining_size: Decimal,
    pub avg_fill_price: Option<Decimal>,
}

/// Market data poller trait (for REST-based polling)
#[async_trait]
pub trait MarketPoller: Send + Sync {
    /// Poll for new markets
    async fn poll_markets(&self) -> Result<Vec<Market>, DataSourceError>;

    /// Poll for market details
    async fn poll_market(&self, market_id: &MarketId) -> Result<Option<Market>, DataSourceError>;

    /// Search markets by keyword
    async fn search_markets(&self, query: &str) -> Result<Vec<Market>, DataSourceError>;
}

/// Event bus for publishing and subscribing to events
pub trait EventBus: Send + Sync {
    /// Publish an event
    fn publish(&self, event: SystemEvent);

    /// Subscribe to events
    fn subscribe(&self) -> broadcast::Receiver<SystemEvent>;

    /// Get the number of subscribers
    fn subscriber_count(&self) -> usize;
}
