//! In-memory market state cache

use chrono::{DateTime, Utc};
use polysniper_core::{
    Market, MarketId, Orderbook, Position, Side, StateManager, StateProvider, TokenId,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

const MAX_PRICE_HISTORY: usize = 1000;
const MAX_ORDERBOOK_SNAPSHOTS: usize = 100;

/// Price history entry
#[derive(Debug, Clone)]
struct PriceEntry {
    timestamp: DateTime<Utc>,
    price: Decimal,
}

/// A large order detected in the orderbook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeOrder {
    /// Price level of the order
    pub price: Decimal,
    /// Size in contracts
    pub size: Decimal,
    /// Size in USD (price * size)
    pub size_usd: Decimal,
    /// Side of the order (bid = Buy, ask = Sell)
    pub side: Side,
}

/// Orderbook snapshot for historical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookSnapshot {
    /// When this snapshot was taken
    pub timestamp: DateTime<Utc>,
    /// Total bid size in USD
    pub total_bid_size_usd: Decimal,
    /// Total ask size in USD
    pub total_ask_size_usd: Decimal,
    /// Large bids detected in this snapshot (orders above threshold)
    pub large_bids: Vec<LargeOrder>,
    /// Large asks detected in this snapshot (orders above threshold)
    pub large_asks: Vec<LargeOrder>,
    /// Best bid price at snapshot time
    pub best_bid: Option<Decimal>,
    /// Best ask price at snapshot time
    pub best_ask: Option<Decimal>,
    /// Mid price at snapshot time
    pub mid_price: Option<Decimal>,
    /// Spread at snapshot time
    pub spread: Option<Decimal>,
}

impl OrderbookSnapshot {
    /// Create a new orderbook snapshot from an orderbook
    pub fn from_orderbook(orderbook: &Orderbook, large_order_threshold_usd: Decimal) -> Self {
        let total_bid_size_usd: Decimal = orderbook
            .bids
            .iter()
            .map(|l| l.price * l.size)
            .sum();
        let total_ask_size_usd: Decimal = orderbook
            .asks
            .iter()
            .map(|l| l.price * l.size)
            .sum();

        let large_bids: Vec<LargeOrder> = orderbook
            .bids
            .iter()
            .filter_map(|l| {
                let size_usd = l.price * l.size;
                if size_usd >= large_order_threshold_usd {
                    Some(LargeOrder {
                        price: l.price,
                        size: l.size,
                        size_usd,
                        side: Side::Buy,
                    })
                } else {
                    None
                }
            })
            .collect();

        let large_asks: Vec<LargeOrder> = orderbook
            .asks
            .iter()
            .filter_map(|l| {
                let size_usd = l.price * l.size;
                if size_usd >= large_order_threshold_usd {
                    Some(LargeOrder {
                        price: l.price,
                        size: l.size,
                        size_usd,
                        side: Side::Sell,
                    })
                } else {
                    None
                }
            })
            .collect();

        Self {
            timestamp: orderbook.timestamp,
            total_bid_size_usd,
            total_ask_size_usd,
            large_bids,
            large_asks,
            best_bid: orderbook.best_bid(),
            best_ask: orderbook.best_ask(),
            mid_price: orderbook.mid_price(),
            spread: orderbook.spread(),
        }
    }

    /// Get the bid/ask imbalance ratio
    pub fn imbalance_ratio(&self) -> Option<Decimal> {
        if self.total_ask_size_usd.is_zero() {
            None
        } else {
            Some(self.total_bid_size_usd / self.total_ask_size_usd)
        }
    }

    /// Check if there are any large orders
    pub fn has_large_orders(&self) -> bool {
        !self.large_bids.is_empty() || !self.large_asks.is_empty()
    }

    /// Get total large order value on bid side
    pub fn total_large_bid_value(&self) -> Decimal {
        self.large_bids.iter().map(|o| o.size_usd).sum()
    }

    /// Get total large order value on ask side
    pub fn total_large_ask_value(&self) -> Decimal {
        self.large_asks.iter().map(|o| o.size_usd).sum()
    }
}

/// Orderbook history for a token
#[derive(Debug, Clone)]
pub struct OrderbookHistory {
    snapshots: VecDeque<OrderbookSnapshot>,
    max_snapshots: usize,
}

impl OrderbookHistory {
    /// Create a new orderbook history with default max snapshots
    pub fn new() -> Self {
        Self {
            snapshots: VecDeque::new(),
            max_snapshots: MAX_ORDERBOOK_SNAPSHOTS,
        }
    }

    /// Create a new orderbook history with custom max snapshots
    pub fn with_max_snapshots(max_snapshots: usize) -> Self {
        Self {
            snapshots: VecDeque::new(),
            max_snapshots,
        }
    }

    /// Add a snapshot
    pub fn push(&mut self, snapshot: OrderbookSnapshot) {
        self.snapshots.push_back(snapshot);
        while self.snapshots.len() > self.max_snapshots {
            self.snapshots.pop_front();
        }
    }

    /// Get the most recent snapshot
    pub fn latest(&self) -> Option<&OrderbookSnapshot> {
        self.snapshots.back()
    }

    /// Get all snapshots
    pub fn snapshots(&self) -> &VecDeque<OrderbookSnapshot> {
        &self.snapshots
    }

    /// Get snapshots within a time window
    pub fn snapshots_since(&self, since: DateTime<Utc>) -> Vec<&OrderbookSnapshot> {
        self.snapshots
            .iter()
            .filter(|s| s.timestamp >= since)
            .collect()
    }

    /// Get the number of snapshots
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Clear all snapshots
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }

    /// Get snapshots with large orders within a time window
    pub fn large_order_snapshots_since(&self, since: DateTime<Utc>) -> Vec<&OrderbookSnapshot> {
        self.snapshots
            .iter()
            .filter(|s| s.timestamp >= since && s.has_large_orders())
            .collect()
    }
}

impl Default for OrderbookHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// In-memory market state cache
pub struct MarketCache {
    markets: Arc<RwLock<HashMap<MarketId, Market>>>,
    orderbooks: Arc<RwLock<HashMap<TokenId, Orderbook>>>,
    prices: Arc<RwLock<HashMap<TokenId, Decimal>>>,
    positions: Arc<RwLock<HashMap<MarketId, Position>>>,
    price_history: Arc<RwLock<HashMap<TokenId, VecDeque<PriceEntry>>>>,
    daily_pnl: Arc<RwLock<Decimal>>,
    orderbook_history: Arc<RwLock<HashMap<TokenId, OrderbookHistory>>>,
    large_order_threshold_usd: Decimal,
}

impl MarketCache {
    /// Create a new market cache
    pub fn new() -> Self {
        Self {
            markets: Arc::new(RwLock::new(HashMap::new())),
            orderbooks: Arc::new(RwLock::new(HashMap::new())),
            prices: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            price_history: Arc::new(RwLock::new(HashMap::new())),
            daily_pnl: Arc::new(RwLock::new(Decimal::ZERO)),
            orderbook_history: Arc::new(RwLock::new(HashMap::new())),
            large_order_threshold_usd: Decimal::new(5000, 0), // Default $5000 threshold
        }
    }

    /// Create a new market cache with custom large order threshold
    pub fn with_large_order_threshold(threshold_usd: Decimal) -> Self {
        Self {
            markets: Arc::new(RwLock::new(HashMap::new())),
            orderbooks: Arc::new(RwLock::new(HashMap::new())),
            prices: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            price_history: Arc::new(RwLock::new(HashMap::new())),
            daily_pnl: Arc::new(RwLock::new(Decimal::ZERO)),
            orderbook_history: Arc::new(RwLock::new(HashMap::new())),
            large_order_threshold_usd: threshold_usd,
        }
    }

    /// Get the number of cached markets
    pub async fn market_count(&self) -> usize {
        self.markets.read().await.len()
    }

    /// Get the number of cached orderbooks
    pub async fn orderbook_count(&self) -> usize {
        self.orderbooks.read().await.len()
    }

    /// Get all market IDs
    pub async fn get_market_ids(&self) -> Vec<MarketId> {
        self.markets.read().await.keys().cloned().collect()
    }

    /// Check if a market exists
    pub async fn has_market(&self, market_id: &MarketId) -> bool {
        self.markets.read().await.contains_key(market_id)
    }

    /// Get price change percentage over a time window
    pub async fn get_price_change_pct(
        &self,
        token_id: &TokenId,
        window_secs: u64,
    ) -> Option<Decimal> {
        let history = self.price_history.read().await;
        let entries = history.get(token_id)?;

        if entries.is_empty() {
            return None;
        }

        let current = entries.back()?;
        let cutoff = Utc::now() - chrono::Duration::seconds(window_secs as i64);

        // Find the oldest price within the window
        let old_entry = entries.iter().find(|e| e.timestamp >= cutoff)?;

        if old_entry.price.is_zero() {
            return None;
        }

        Some(((current.price - old_entry.price) / old_entry.price) * Decimal::ONE_HUNDRED)
    }

    /// Update daily P&L
    pub async fn update_daily_pnl(&self, delta: Decimal) {
        let mut pnl = self.daily_pnl.write().await;
        *pnl += delta;
    }

    /// Reset daily P&L
    pub async fn reset_daily_pnl(&self) {
        let mut pnl = self.daily_pnl.write().await;
        *pnl = Decimal::ZERO;
    }

    /// Get orderbook history for a token
    pub async fn get_orderbook_history(&self, token_id: &TokenId) -> Option<OrderbookHistory> {
        self.orderbook_history.read().await.get(token_id).cloned()
    }

    /// Get orderbook snapshots since a given time
    pub async fn get_orderbook_snapshots_since(
        &self,
        token_id: &TokenId,
        since: DateTime<Utc>,
    ) -> Vec<OrderbookSnapshot> {
        let history = self.orderbook_history.read().await;
        history
            .get(token_id)
            .map(|h| h.snapshots_since(since).into_iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get the most recent orderbook snapshot
    pub async fn get_latest_orderbook_snapshot(
        &self,
        token_id: &TokenId,
    ) -> Option<OrderbookSnapshot> {
        let history = self.orderbook_history.read().await;
        history.get(token_id).and_then(|h| h.latest().cloned())
    }

    /// Get large orders from recent snapshots
    pub async fn get_recent_large_orders(
        &self,
        token_id: &TokenId,
        window_secs: u64,
    ) -> (Vec<LargeOrder>, Vec<LargeOrder>) {
        let since = Utc::now() - chrono::Duration::seconds(window_secs as i64);
        let history = self.orderbook_history.read().await;

        let mut large_bids = Vec::new();
        let mut large_asks = Vec::new();

        if let Some(h) = history.get(token_id) {
            for snapshot in h.large_order_snapshots_since(since) {
                large_bids.extend(snapshot.large_bids.iter().cloned());
                large_asks.extend(snapshot.large_asks.iter().cloned());
            }
        }

        (large_bids, large_asks)
    }

    /// Set the large order threshold
    pub fn set_large_order_threshold(&mut self, threshold_usd: Decimal) {
        self.large_order_threshold_usd = threshold_usd;
    }

    /// Clear orderbook history for a token
    pub async fn clear_orderbook_history(&self, token_id: &TokenId) {
        self.orderbook_history.write().await.remove(token_id);
    }

    /// Clear all orderbook history
    pub async fn clear_all_orderbook_history(&self) {
        self.orderbook_history.write().await.clear();
    }
}

impl Default for MarketCache {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl StateProvider for MarketCache {
    async fn get_market(&self, market_id: &MarketId) -> Option<Market> {
        self.markets.read().await.get(market_id).cloned()
    }

    async fn get_all_markets(&self) -> Vec<Market> {
        self.markets.read().await.values().cloned().collect()
    }

    async fn get_orderbook(&self, token_id: &TokenId) -> Option<Orderbook> {
        self.orderbooks.read().await.get(token_id).cloned()
    }

    async fn get_price(&self, token_id: &TokenId) -> Option<Decimal> {
        self.prices.read().await.get(token_id).copied()
    }

    async fn get_position(&self, market_id: &MarketId) -> Option<Position> {
        self.positions.read().await.get(market_id).cloned()
    }

    async fn get_all_positions(&self) -> Vec<Position> {
        self.positions.read().await.values().cloned().collect()
    }

    async fn get_price_history(
        &self,
        token_id: &TokenId,
        limit: usize,
    ) -> Vec<(DateTime<Utc>, Decimal)> {
        let history = self.price_history.read().await;
        match history.get(token_id) {
            Some(entries) => entries
                .iter()
                .rev()
                .take(limit)
                .map(|e| (e.timestamp, e.price))
                .collect(),
            None => Vec::new(),
        }
    }

    async fn get_portfolio_value(&self) -> Decimal {
        let positions = self.positions.read().await;
        let prices = self.prices.read().await;

        positions
            .values()
            .filter_map(|pos| {
                let price = prices.get(&pos.token_id)?;
                Some(pos.size * price)
            })
            .sum()
    }

    async fn get_daily_pnl(&self) -> Decimal {
        *self.daily_pnl.read().await
    }
}

#[async_trait::async_trait]
impl StateManager for MarketCache {
    async fn update_market(&self, market: Market) {
        self.markets
            .write()
            .await
            .insert(market.condition_id.clone(), market);
    }

    async fn update_orderbook(&self, orderbook: Orderbook) {
        // Also update price from orderbook
        if let Some(mid_price) = orderbook.mid_price() {
            self.update_price(orderbook.token_id.clone(), mid_price)
                .await;
        }

        // Record orderbook snapshot for history
        let snapshot = OrderbookSnapshot::from_orderbook(&orderbook, self.large_order_threshold_usd);
        {
            let mut history = self.orderbook_history.write().await;
            let token_history = history
                .entry(orderbook.token_id.clone())
                .or_insert_with(OrderbookHistory::new);
            token_history.push(snapshot);
        }

        self.orderbooks
            .write()
            .await
            .insert(orderbook.token_id.clone(), orderbook);
    }

    async fn update_price(&self, token_id: TokenId, price: Decimal) {
        self.prices.write().await.insert(token_id.clone(), price);
        self.record_price_snapshot(token_id, price).await;
    }

    async fn update_position(&self, position: Position) {
        self.positions
            .write()
            .await
            .insert(position.market_id.clone(), position);
    }

    async fn record_price_snapshot(&self, token_id: TokenId, price: Decimal) {
        let mut history = self.price_history.write().await;
        let entries = history.entry(token_id).or_insert_with(VecDeque::new);

        entries.push_back(PriceEntry {
            timestamp: Utc::now(),
            price,
        });

        // Trim to max size
        while entries.len() > MAX_PRICE_HISTORY {
            entries.pop_front();
        }
    }

    async fn clear(&self) {
        self.markets.write().await.clear();
        self.orderbooks.write().await.clear();
        self.prices.write().await.clear();
        self.positions.write().await.clear();
        self.price_history.write().await.clear();
        self.orderbook_history.write().await.clear();
        *self.daily_pnl.write().await = Decimal::ZERO;
    }
}
