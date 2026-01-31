//! Backtest engine for replaying historical data through strategies

use crate::config::{BacktestConfig, SlippageModel};
use crate::data_loader::DataLoader;
use crate::error::Result;
use crate::results::{BacktestResults, TradeResult};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use polysniper_core::{
    Market, MarketId, Orderbook, Position, StateProvider, Strategy, SystemEvent, TokenId,
    TradeSignal,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Backtest engine that replays historical events through strategies
pub struct BacktestEngine {
    config: BacktestConfig,
    data_loader: DataLoader,
    state: Arc<SimulatedState>,
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub async fn new(config: BacktestConfig, db_path: &str) -> Result<Self> {
        let data_loader = DataLoader::new(db_path).await?;
        let state = Arc::new(SimulatedState::new(config.initial_capital));

        Ok(Self {
            config,
            data_loader,
            state,
        })
    }

    /// Create a backtest engine from an existing data loader
    pub fn from_data_loader(config: BacktestConfig, data_loader: DataLoader) -> Self {
        let state = Arc::new(SimulatedState::new(config.initial_capital));
        Self {
            config,
            data_loader,
            state,
        }
    }

    /// Run a backtest with a single strategy
    pub async fn run(&self, strategy: &dyn Strategy) -> Result<BacktestResults> {
        info!(
            strategy_id = %strategy.id(),
            start = %self.config.start_time,
            end = %self.config.end_time,
            "Starting backtest"
        );

        // Reset state
        self.state.reset(self.config.initial_capital).await;

        // Load historical events
        let events = self
            .data_loader
            .load_events(
                self.config.start_time,
                self.config.end_time,
                self.config.token_filter.as_deref(),
            )
            .await?;

        info!(event_count = events.len(), "Loaded historical events");

        let mut trades = Vec::new();

        // Process each event
        for event in &events {
            // Update simulated time
            self.state.set_current_time(event.timestamp()).await;

            // Update state based on event
            self.update_state(event).await;

            // Check if strategy accepts this event
            if !strategy.accepts_event(event) {
                continue;
            }

            // Process event through strategy
            match strategy.process_event(event, self.state.as_ref()).await {
                Ok(signals) => {
                    for signal in signals {
                        if let Some(trade) = self.execute_signal(&signal).await? {
                            trades.push(trade);
                        }
                    }
                }
                Err(e) => {
                    debug!(
                        strategy_id = %strategy.id(),
                        error = %e,
                        "Strategy error during backtest"
                    );
                }
            }
        }

        // Close any open positions at the end
        let final_trades = self.close_all_positions().await?;
        trades.extend(final_trades);

        let results = BacktestResults::from_trades(
            strategy.id().to_string(),
            self.config.start_time,
            self.config.end_time,
            self.config.initial_capital,
            trades,
        );

        info!(
            strategy_id = %strategy.id(),
            total_pnl = %results.metrics.total_pnl,
            trade_count = results.metrics.trade_count,
            win_rate = %results.metrics.win_rate,
            sharpe = %results.metrics.sharpe_ratio,
            "Backtest complete"
        );

        Ok(results)
    }

    /// Run backtest with multiple strategies
    pub async fn run_multiple(
        &self,
        strategies: &[&dyn Strategy],
    ) -> Result<Vec<BacktestResults>> {
        let mut all_results = Vec::with_capacity(strategies.len());

        for strategy in strategies {
            let results = self.run(*strategy).await?;
            all_results.push(results);
        }

        Ok(all_results)
    }

    async fn update_state(&self, event: &SystemEvent) {
        match event {
            SystemEvent::PriceChange(e) => {
                self.state
                    .update_price(e.token_id.clone(), e.new_price)
                    .await;
            }
            SystemEvent::OrderbookUpdate(e) => {
                self.state.update_orderbook(e.orderbook.clone()).await;
            }
            SystemEvent::NewMarket(e) => {
                self.state.update_market(e.market.clone()).await;
            }
            _ => {}
        }
    }

    async fn execute_signal(&self, signal: &TradeSignal) -> Result<Option<TradeResult>> {
        // Get current price
        let current_price = self
            .state
            .get_price(&signal.token_id)
            .await
            .unwrap_or(signal.price.unwrap_or(dec!(0.50)));

        // Apply slippage
        let execution_price = self.apply_slippage(current_price, signal);

        // Calculate fees
        let fees = self.calculate_fees(signal.size_usd);

        // Create trade result
        let pnl = self.calculate_trade_pnl(signal, execution_price, fees);

        // Update simulated balance
        self.state.update_balance(pnl - fees).await;

        // Track position
        self.state
            .update_position_from_signal(signal, execution_price)
            .await;

        let trade = TradeResult {
            id: uuid::Uuid::new_v4().to_string(),
            signal_id: signal.id.clone(),
            market_id: signal.market_id.clone(),
            token_id: signal.token_id.clone(),
            is_buy: signal.side == polysniper_core::Side::Buy,
            entry_price: execution_price,
            exit_price: None,
            size: signal.size,
            size_usd: signal.size_usd,
            fees,
            pnl,
            timestamp: signal.timestamp,
            reason: signal.reason.clone(),
        };

        debug!(
            signal_id = %signal.id,
            price = %execution_price,
            size = %signal.size,
            pnl = %pnl,
            "Executed simulated trade"
        );

        Ok(Some(trade))
    }

    fn apply_slippage(&self, price: Decimal, signal: &TradeSignal) -> Decimal {
        match self.config.slippage.model {
            SlippageModel::None => price,
            SlippageModel::Fixed => {
                let slippage = price * self.config.slippage.base_slippage_pct;
                match signal.side {
                    polysniper_core::Side::Buy => price + slippage,
                    polysniper_core::Side::Sell => price - slippage,
                }
            }
            SlippageModel::SizeProportional => {
                // More slippage for larger orders
                let size_factor = signal.size_usd / dec!(1000);
                let slippage = price * self.config.slippage.base_slippage_pct * (dec!(1) + size_factor);
                match signal.side {
                    polysniper_core::Side::Buy => price + slippage,
                    polysniper_core::Side::Sell => price - slippage,
                }
            }
        }
    }

    fn calculate_fees(&self, size_usd: Decimal) -> Decimal {
        let fee = size_usd * self.config.fees.trading_fee_pct;
        fee.max(self.config.fees.min_fee_usd)
    }

    fn calculate_trade_pnl(
        &self,
        _signal: &TradeSignal,
        _execution_price: Decimal,
        fees: Decimal,
    ) -> Decimal {
        // For prediction markets, P&L depends on whether the position is eventually correct
        // For simplicity in backtesting, we calculate based on price movement from entry
        // In a full simulation, we would track positions and close them at market resolution

        // Since we're replaying price data, we assume the trade is closed at a later price
        // For now, return negative fees as a placeholder (actual P&L calculated on position close)
        -fees
    }

    async fn close_all_positions(&self) -> Result<Vec<TradeResult>> {
        let positions = self.state.get_all_positions().await;
        let mut trades = Vec::new();

        for position in positions {
            if position.size.is_zero() {
                continue;
            }

            let current_price = self
                .state
                .get_price(&position.token_id)
                .await
                .unwrap_or(position.avg_price);

            let pnl = if position.size > Decimal::ZERO {
                // Long position
                (current_price - position.avg_price) * position.size
            } else {
                // Short position
                (position.avg_price - current_price) * position.size.abs()
            };

            let trade = TradeResult {
                id: uuid::Uuid::new_v4().to_string(),
                signal_id: "close_position".to_string(),
                market_id: position.market_id.clone(),
                token_id: position.token_id.clone(),
                is_buy: position.size < Decimal::ZERO, // Closing a short = buy
                entry_price: position.avg_price,
                exit_price: Some(current_price),
                size: position.size.abs(),
                size_usd: position.size.abs() * current_price,
                fees: Decimal::ZERO,
                pnl,
                timestamp: self.state.current_time().await,
                reason: "Position closed at end of backtest".to_string(),
            };

            trades.push(trade);
        }

        Ok(trades)
    }

    /// Get the available data range
    pub async fn get_data_range(&self) -> Result<Option<(DateTime<Utc>, DateTime<Utc>)>> {
        self.data_loader.get_data_range().await
    }
}

/// Simulated market state for backtesting
pub struct SimulatedState {
    prices: RwLock<HashMap<TokenId, Decimal>>,
    orderbooks: RwLock<HashMap<TokenId, Orderbook>>,
    markets: RwLock<HashMap<MarketId, Market>>,
    positions: RwLock<HashMap<MarketId, Position>>,
    balance: RwLock<Decimal>,
    current_time: RwLock<DateTime<Utc>>,
    price_history: RwLock<HashMap<TokenId, Vec<(DateTime<Utc>, Decimal)>>>,
}

impl SimulatedState {
    pub fn new(initial_balance: Decimal) -> Self {
        Self {
            prices: RwLock::new(HashMap::new()),
            orderbooks: RwLock::new(HashMap::new()),
            markets: RwLock::new(HashMap::new()),
            positions: RwLock::new(HashMap::new()),
            balance: RwLock::new(initial_balance),
            current_time: RwLock::new(Utc::now()),
            price_history: RwLock::new(HashMap::new()),
        }
    }

    pub async fn reset(&self, initial_balance: Decimal) {
        self.prices.write().await.clear();
        self.orderbooks.write().await.clear();
        self.positions.write().await.clear();
        self.price_history.write().await.clear();
        *self.balance.write().await = initial_balance;
    }

    pub async fn update_price(&self, token_id: TokenId, price: Decimal) {
        self.prices.write().await.insert(token_id.clone(), price);

        // Track price history
        let time = *self.current_time.read().await;
        self.price_history
            .write()
            .await
            .entry(token_id)
            .or_default()
            .push((time, price));
    }

    pub async fn update_orderbook(&self, orderbook: Orderbook) {
        self.orderbooks
            .write()
            .await
            .insert(orderbook.token_id.clone(), orderbook);
    }

    pub async fn update_market(&self, market: Market) {
        self.markets
            .write()
            .await
            .insert(market.condition_id.clone(), market);
    }

    pub async fn update_balance(&self, delta: Decimal) {
        *self.balance.write().await += delta;
    }

    pub async fn set_current_time(&self, time: DateTime<Utc>) {
        *self.current_time.write().await = time;
    }

    pub async fn current_time(&self) -> DateTime<Utc> {
        *self.current_time.read().await
    }

    pub async fn update_position_from_signal(&self, signal: &TradeSignal, price: Decimal) {
        let mut positions = self.positions.write().await;
        let position = positions.entry(signal.market_id.clone()).or_insert_with(|| {
            Position {
                market_id: signal.market_id.clone(),
                token_id: signal.token_id.clone(),
                outcome: signal.outcome,
                size: Decimal::ZERO,
                avg_price: Decimal::ZERO,
                realized_pnl: Decimal::ZERO,
                unrealized_pnl: Decimal::ZERO,
                updated_at: Utc::now(),
            }
        });

        let size_delta = match signal.side {
            polysniper_core::Side::Buy => signal.size,
            polysniper_core::Side::Sell => -signal.size,
        };

        // Update average price
        if (position.size > Decimal::ZERO && size_delta > Decimal::ZERO)
            || (position.size < Decimal::ZERO && size_delta < Decimal::ZERO)
        {
            // Adding to position - recalculate average
            let total_cost = position.size * position.avg_price + size_delta * price;
            let new_size = position.size + size_delta;
            if !new_size.is_zero() {
                position.avg_price = total_cost / new_size;
            }
        }

        position.size += size_delta;
        position.updated_at = *self.current_time.read().await;
    }
}

#[async_trait]
impl StateProvider for SimulatedState {
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
        self.price_history
            .read()
            .await
            .get(token_id)
            .map(|h| h.iter().rev().take(limit).cloned().collect())
            .unwrap_or_default()
    }

    async fn get_portfolio_value(&self) -> Decimal {
        *self.balance.read().await
    }

    async fn get_daily_pnl(&self) -> Decimal {
        Decimal::ZERO // Not tracked in simulation
    }

    async fn get_trade_outcomes(&self, _limit: usize) -> Vec<(Decimal, Decimal)> {
        // Trade history not tracked in basic simulation
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_simulated_state() {
        let state = SimulatedState::new(dec!(10000));

        state.update_price("token1".to_string(), dec!(0.50)).await;
        assert_eq!(state.get_price(&"token1".to_string()).await, Some(dec!(0.50)));

        state.update_balance(dec!(100)).await;
        assert_eq!(state.get_portfolio_value().await, dec!(10100));
    }

    #[tokio::test]
    async fn test_state_reset() {
        let state = SimulatedState::new(dec!(10000));

        state.update_price("token1".to_string(), dec!(0.50)).await;
        state.update_balance(dec!(100)).await;

        state.reset(dec!(5000)).await;

        assert_eq!(state.get_price(&"token1".to_string()).await, None);
        assert_eq!(state.get_portfolio_value().await, dec!(5000));
    }
}
