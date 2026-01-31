//! In-memory market state cache

use chrono::{DateTime, Utc};
use polysniper_core::{
    Market, MarketId, Orderbook, Position, StateManager, StateProvider, TokenId,
};
use rust_decimal::Decimal;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

const MAX_PRICE_HISTORY: usize = 1000;

/// Price history entry
#[derive(Debug, Clone)]
struct PriceEntry {
    timestamp: DateTime<Utc>,
    price: Decimal,
}

/// In-memory market state cache
pub struct MarketCache {
    markets: Arc<RwLock<HashMap<MarketId, Market>>>,
    orderbooks: Arc<RwLock<HashMap<TokenId, Orderbook>>>,
    prices: Arc<RwLock<HashMap<TokenId, Decimal>>>,
    positions: Arc<RwLock<HashMap<MarketId, Position>>>,
    price_history: Arc<RwLock<HashMap<TokenId, VecDeque<PriceEntry>>>>,
    daily_pnl: Arc<RwLock<Decimal>>,
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

    async fn get_trade_outcomes(&self, _limit: usize) -> Vec<(Decimal, Decimal)> {
        // MarketCache doesn't track trade history directly
        // This would need to be fetched from the persistence layer
        // For now, return empty - Kelly sizing will skip if no data
        Vec::new()
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
        *self.daily_pnl.write().await = Decimal::ZERO;
    }
}
