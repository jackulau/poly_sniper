//! Risk validation implementation

use crate::correlation::CorrelationTracker;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use polysniper_core::{
    RiskConfig, RiskDecision, RiskError, RiskValidator, StateProvider, TradeSignal,
};
use rust_decimal::Decimal;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Order tracking for rate limiting
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct OrderRecord {
    timestamp: DateTime<Utc>,
    market_id: String,
}

/// Risk manager implementation
pub struct RiskManager {
    config: RiskConfig,
    halted: Arc<AtomicBool>,
    halt_reason: Arc<RwLock<Option<String>>>,
    /// Recent orders for rate limiting
    recent_orders: Arc<RwLock<VecDeque<OrderRecord>>>,
    /// Correlation tracker for correlated exposure limits
    correlation_tracker: CorrelationTracker,
}

impl RiskManager {
    /// Create a new risk manager
    pub fn new(config: RiskConfig) -> Self {
        let correlation_tracker = CorrelationTracker::new(config.correlation.clone());
        Self {
            config,
            halted: Arc::new(AtomicBool::new(false)),
            halt_reason: Arc::new(RwLock::new(None)),
            recent_orders: Arc::new(RwLock::new(VecDeque::new())),
            correlation_tracker,
        }
    }

    /// Get a reference to the correlation tracker
    pub fn correlation_tracker(&self) -> &CorrelationTracker {
        &self.correlation_tracker
    }

    /// Get a mutable reference to the correlation tracker
    pub fn correlation_tracker_mut(&mut self) -> &mut CorrelationTracker {
        &mut self.correlation_tracker
    }

    /// Record an order for rate limiting
    pub async fn record_order(&self, market_id: &str) {
        let mut orders = self.recent_orders.write().await;
        orders.push_back(OrderRecord {
            timestamp: Utc::now(),
            market_id: market_id.to_string(),
        });

        // Clean old entries (older than 1 minute)
        let cutoff = Utc::now() - chrono::Duration::seconds(60);
        while orders
            .front()
            .map(|o| o.timestamp < cutoff)
            .unwrap_or(false)
        {
            orders.pop_front();
        }
    }

    /// Get orders per minute
    async fn orders_per_minute(&self) -> u32 {
        let orders = self.recent_orders.read().await;
        let cutoff = Utc::now() - chrono::Duration::seconds(60);
        orders.iter().filter(|o| o.timestamp >= cutoff).count() as u32
    }

    /// Check rate limit
    async fn check_rate_limit(&self) -> Result<(), RiskError> {
        let current_rate = self.orders_per_minute().await;
        if current_rate >= self.config.max_orders_per_minute {
            return Err(RiskError::RateLimitExceeded);
        }
        Ok(())
    }

    /// Check order size limit
    fn check_order_size(&self, signal: &TradeSignal) -> Result<(), RiskError> {
        if signal.size_usd > self.config.max_order_size_usd {
            return Err(RiskError::OrderSizeLimitExceeded(format!(
                "Order size ${} exceeds limit ${}",
                signal.size_usd, self.config.max_order_size_usd
            )));
        }
        Ok(())
    }

    /// Check position limit
    async fn check_position_limit(
        &self,
        signal: &TradeSignal,
        state: &dyn StateProvider,
    ) -> Result<(), RiskError> {
        let current_position = state.get_position(&signal.market_id).await;
        let current_value = current_position
            .map(|p| p.size * signal.price.unwrap_or(Decimal::ZERO))
            .unwrap_or(Decimal::ZERO);

        let new_value = current_value + signal.size_usd;

        if new_value > self.config.max_position_size_usd {
            return Err(RiskError::PositionLimitExceeded(format!(
                "Position would be ${} exceeding limit ${}",
                new_value, self.config.max_position_size_usd
            )));
        }
        Ok(())
    }

    /// Check daily loss limit
    async fn check_daily_loss(&self, state: &dyn StateProvider) -> Result<(), RiskError> {
        let daily_pnl = state.get_daily_pnl().await;

        // Check circuit breaker
        if daily_pnl <= -self.config.circuit_breaker_loss_usd {
            self.halt(&format!(
                "Circuit breaker triggered: daily loss ${} exceeds ${}",
                daily_pnl.abs(),
                self.config.circuit_breaker_loss_usd
            ));
            return Err(RiskError::CircuitBreakerTriggered(format!(
                "Daily loss ${} exceeds circuit breaker threshold ${}",
                daily_pnl.abs(),
                self.config.circuit_breaker_loss_usd
            )));
        }

        // Check daily loss limit
        if daily_pnl <= -self.config.daily_loss_limit_usd {
            return Err(RiskError::DailyLossLimitExceeded(format!(
                "Daily loss ${} exceeds limit ${}",
                daily_pnl.abs(),
                self.config.daily_loss_limit_usd
            )));
        }

        Ok(())
    }

    /// Calculate adjusted size if needed
    fn calculate_adjusted_size(
        &self,
        signal: &TradeSignal,
        max_allowed: Decimal,
    ) -> Option<Decimal> {
        if signal.size_usd > max_allowed && max_allowed > Decimal::ZERO {
            let price = signal.price.unwrap_or(Decimal::ONE);
            if price.is_zero() {
                return None;
            }
            Some(max_allowed / price)
        } else {
            None
        }
    }

    /// Check correlated exposure limit
    async fn check_correlated_exposure(
        &self,
        signal: &TradeSignal,
        state: &dyn StateProvider,
    ) -> Result<Option<Decimal>, RiskError> {
        if !self.correlation_tracker.is_enabled() {
            return Ok(None);
        }

        // Check if this market is in any correlation group
        let groups = self.correlation_tracker.get_groups_for_market(&signal.market_id);
        if groups.is_empty() {
            debug!(
                market_id = %signal.market_id,
                "Market not in any correlation group, skipping correlation check"
            );
            return Ok(None);
        }

        // Calculate current correlated exposure
        let current_exposure = self
            .correlation_tracker
            .calculate_correlated_exposure(&signal.market_id, state)
            .await;

        let max_exposure = self.correlation_tracker.max_correlated_exposure();
        let new_total = current_exposure + signal.size_usd;

        debug!(
            market_id = %signal.market_id,
            current_exposure = %current_exposure,
            order_size = %signal.size_usd,
            new_total = %new_total,
            limit = %max_exposure,
            "Checking correlated exposure"
        );

        if new_total > max_exposure {
            let remaining_room = max_exposure - current_exposure;
            if remaining_room <= Decimal::ZERO {
                return Err(RiskError::CorrelatedExposureExceeded(format!(
                    "Correlated exposure ${} already at/above limit ${}",
                    current_exposure, max_exposure
                )));
            }
            // Return the remaining room for potential size adjustment
            return Ok(Some(remaining_room));
        }

        Ok(None)
    }
}

#[async_trait]
impl RiskValidator for RiskManager {
    async fn validate(
        &self,
        signal: &TradeSignal,
        state: &dyn StateProvider,
    ) -> Result<RiskDecision, RiskError> {
        // Check if halted
        if self.is_halted() {
            let reason = self
                .halt_reason
                .read()
                .await
                .clone()
                .unwrap_or_else(|| "Unknown reason".to_string());
            return Err(RiskError::TradingHalted(reason));
        }

        // Check rate limit
        self.check_rate_limit().await?;

        // Check daily loss
        self.check_daily_loss(state).await?;

        // Check order size
        if let Err(e) = self.check_order_size(signal) {
            // Try to adjust size
            if let Some(new_size) =
                self.calculate_adjusted_size(signal, self.config.max_order_size_usd)
            {
                warn!(
                    signal_id = %signal.id,
                    original_size = %signal.size,
                    new_size = %new_size,
                    "Adjusting order size due to limit"
                );
                return Ok(RiskDecision::Modified {
                    new_size,
                    reason: format!(
                        "Order size reduced from {} to {} due to max order size limit",
                        signal.size, new_size
                    ),
                });
            }
            return Err(e);
        }

        // Check position limit
        if let Err(e) = self.check_position_limit(signal, state).await {
            // Calculate how much room we have
            let current_position = state.get_position(&signal.market_id).await;
            let current_value = current_position
                .map(|p| p.size * signal.price.unwrap_or(Decimal::ZERO))
                .unwrap_or(Decimal::ZERO);

            let remaining_room = self.config.max_position_size_usd - current_value;

            if let Some(new_size) = self.calculate_adjusted_size(signal, remaining_room) {
                if new_size > Decimal::ZERO {
                    warn!(
                        signal_id = %signal.id,
                        original_size = %signal.size,
                        new_size = %new_size,
                        "Adjusting order size due to position limit"
                    );
                    return Ok(RiskDecision::Modified {
                        new_size,
                        reason: format!(
                            "Order size reduced from {} to {} due to position limit",
                            signal.size, new_size
                        ),
                    });
                }
            }
            return Err(e);
        }

        // Check correlated exposure limit
        match self.check_correlated_exposure(signal, state).await {
            Ok(Some(remaining_room)) => {
                // Need to reduce size due to correlated exposure
                if let Some(new_size) = self.calculate_adjusted_size(signal, remaining_room) {
                    if new_size > Decimal::ZERO {
                        warn!(
                            signal_id = %signal.id,
                            original_size = %signal.size,
                            new_size = %new_size,
                            remaining_room = %remaining_room,
                            "Adjusting order size due to correlated exposure limit"
                        );
                        return Ok(RiskDecision::Modified {
                            new_size,
                            reason: format!(
                                "Order size reduced from {} to {} due to correlated exposure limit (${} remaining)",
                                signal.size, new_size, remaining_room
                            ),
                        });
                    }
                }
                // No room at all
                return Err(RiskError::CorrelatedExposureExceeded(format!(
                    "No remaining room in correlated exposure limit (${} limit)",
                    self.correlation_tracker.max_correlated_exposure()
                )));
            }
            Ok(None) => {
                // No adjustment needed
            }
            Err(e) => return Err(e),
        }

        Ok(RiskDecision::Approved)
    }

    fn is_halted(&self) -> bool {
        self.halted.load(Ordering::SeqCst)
    }

    fn halt(&self, reason: &str) {
        self.halted.store(true, Ordering::SeqCst);
        // Use blocking write since halt can be called from sync context
        if let Ok(mut halt_reason) = self.halt_reason.try_write() {
            *halt_reason = Some(reason.to_string());
        }
        warn!(reason = %reason, "Trading halted!");
    }

    fn resume(&self) {
        self.halted.store(false, Ordering::SeqCst);
        if let Ok(mut halt_reason) = self.halt_reason.try_write() {
            *halt_reason = None;
        }
        info!("Trading resumed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use chrono::Utc;
    use polysniper_core::{
        CorrelationConfig, CorrelationGroupConfig, Market, Orderbook, Outcome, Position, Priority,
        Side, OrderType,
    };
    use rust_decimal_macros::dec;
    use std::collections::HashMap;
    use std::sync::RwLock as StdRwLock;

    /// Mock state provider for testing
    struct MockStateProvider {
        positions: StdRwLock<HashMap<String, Position>>,
        prices: StdRwLock<HashMap<String, Decimal>>,
        daily_pnl: Decimal,
    }

    impl MockStateProvider {
        fn new() -> Self {
            Self {
                positions: StdRwLock::new(HashMap::new()),
                prices: StdRwLock::new(HashMap::new()),
                daily_pnl: Decimal::ZERO,
            }
        }

        fn with_position(self, market_id: &str, token_id: &str, size: Decimal, price: Decimal) -> Self {
            let position = Position {
                market_id: market_id.to_string(),
                token_id: token_id.to_string(),
                outcome: Outcome::Yes,
                size,
                avg_price: price,
                realized_pnl: Decimal::ZERO,
                unrealized_pnl: Decimal::ZERO,
                updated_at: Utc::now(),
            };
            self.positions.write().unwrap().insert(market_id.to_string(), position);
            self.prices.write().unwrap().insert(token_id.to_string(), price);
            self
        }

        #[allow(dead_code)]
        fn with_daily_pnl(mut self, pnl: Decimal) -> Self {
            self.daily_pnl = pnl;
            self
        }
    }

    #[async_trait]
    impl StateProvider for MockStateProvider {
        async fn get_market(&self, _market_id: &String) -> Option<Market> {
            None
        }

        async fn get_all_markets(&self) -> Vec<Market> {
            Vec::new()
        }

        async fn get_orderbook(&self, _token_id: &String) -> Option<Orderbook> {
            None
        }

        async fn get_price(&self, token_id: &String) -> Option<Decimal> {
            self.prices.read().unwrap().get(token_id).copied()
        }

        async fn get_position(&self, market_id: &String) -> Option<Position> {
            self.positions.read().unwrap().get(market_id).cloned()
        }

        async fn get_all_positions(&self) -> Vec<Position> {
            self.positions.read().unwrap().values().cloned().collect()
        }

        async fn get_price_history(
            &self,
            _token_id: &String,
            _limit: usize,
        ) -> Vec<(chrono::DateTime<chrono::Utc>, Decimal)> {
            Vec::new()
        }

        async fn get_portfolio_value(&self) -> Decimal {
            Decimal::ZERO
        }

        async fn get_daily_pnl(&self) -> Decimal {
            self.daily_pnl
        }
    }

    fn create_test_signal(market_id: &str, token_id: &str, size_usd: Decimal, price: Decimal) -> TradeSignal {
        TradeSignal {
            id: "test-signal-1".to_string(),
            strategy_id: "test-strategy".to_string(),
            market_id: market_id.to_string(),
            token_id: token_id.to_string(),
            outcome: Outcome::Yes,
            side: Side::Buy,
            price: Some(price),
            size: size_usd / price,
            size_usd,
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: Utc::now(),
            reason: "test".to_string(),
            metadata: serde_json::Value::Null,
        }
    }

    fn create_config_with_correlation(groups: Vec<CorrelationGroupConfig>) -> RiskConfig {
        RiskConfig {
            max_position_size_usd: dec!(5000),
            max_order_size_usd: dec!(500),
            daily_loss_limit_usd: dec!(500),
            circuit_breaker_loss_usd: dec!(300),
            max_orders_per_minute: 60,
            correlation: CorrelationConfig {
                enabled: true,
                correlation_threshold: dec!(0.7),
                window_secs: 3600,
                max_correlated_exposure_usd: dec!(3000),
                groups,
            },
        }
    }

    #[tokio::test]
    async fn test_approve_order_not_in_correlation_group() {
        let config = create_config_with_correlation(vec![
            CorrelationGroupConfig {
                name: "election".to_string(),
                markets: vec!["presidential-winner".to_string()],
            },
        ]);

        let risk_manager = RiskManager::new(config);
        let state = MockStateProvider::new();

        // Order for a market not in any correlation group
        let signal = create_test_signal("other-market", "token-1", dec!(100), dec!(0.5));

        let result = risk_manager.validate(&signal, &state).await;
        assert!(matches!(result, Ok(RiskDecision::Approved)));
    }

    #[tokio::test]
    async fn test_approve_order_within_correlation_limit() {
        let config = create_config_with_correlation(vec![
            CorrelationGroupConfig {
                name: "election".to_string(),
                markets: vec![
                    "presidential-winner".to_string(),
                    "electoral-count".to_string(),
                ],
            },
        ]);

        let risk_manager = RiskManager::new(config);

        // Existing position of $1000 in a correlated market
        let state = MockStateProvider::new()
            .with_position("electoral-count", "token-ec", dec!(2000), dec!(0.5));

        // New order of $500 in another correlated market (total $1500 < $3000 limit)
        let signal = create_test_signal("presidential-winner", "token-pw", dec!(500), dec!(0.5));

        let result = risk_manager.validate(&signal, &state).await;
        assert!(matches!(result, Ok(RiskDecision::Approved)));
    }

    #[tokio::test]
    async fn test_reduce_order_when_correlation_limit_exceeded() {
        let config = create_config_with_correlation(vec![
            CorrelationGroupConfig {
                name: "election".to_string(),
                markets: vec![
                    "presidential-winner".to_string(),
                    "electoral-count".to_string(),
                ],
            },
        ]);

        let risk_manager = RiskManager::new(config);

        // Existing position of $2800 in a correlated market (5600 contracts * $0.5)
        let state = MockStateProvider::new()
            .with_position("electoral-count", "token-ec", dec!(5600), dec!(0.5));

        // New order of $500 (max order size) would exceed $3000 correlated limit
        // Current correlated exposure: $2800, new order: $500, total: $3300 > $3000
        // Remaining room: $200
        let signal = create_test_signal("presidential-winner", "token-pw", dec!(500), dec!(0.5));

        let result = risk_manager.validate(&signal, &state).await;

        match result {
            Ok(RiskDecision::Modified { new_size, reason }) => {
                assert!(
                    reason.contains("correlated exposure limit"),
                    "Expected reason to contain 'correlated exposure limit', got: {}",
                    reason
                );
                // $200 remaining / $0.5 price = 400 contracts
                assert_eq!(new_size, dec!(400));
            }
            other => panic!("Expected Modified decision, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_reject_order_when_at_correlation_limit() {
        let config = create_config_with_correlation(vec![
            CorrelationGroupConfig {
                name: "election".to_string(),
                markets: vec![
                    "presidential-winner".to_string(),
                    "electoral-count".to_string(),
                ],
            },
        ]);

        let risk_manager = RiskManager::new(config);

        // Existing position exactly at the $3000 limit
        let state = MockStateProvider::new()
            .with_position("electoral-count", "token-ec", dec!(6000), dec!(0.5));

        // Any new order should be rejected
        let signal = create_test_signal("presidential-winner", "token-pw", dec!(100), dec!(0.5));

        let result = risk_manager.validate(&signal, &state).await;

        match result {
            Err(RiskError::CorrelatedExposureExceeded(msg)) => {
                assert!(msg.contains("already at/above limit"));
            }
            other => panic!("Expected CorrelatedExposureExceeded error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_pattern_matching_in_correlation_groups() {
        let config = create_config_with_correlation(vec![
            CorrelationGroupConfig {
                name: "swing-states".to_string(),
                markets: vec!["swing-state-*".to_string()],
            },
        ]);

        let risk_manager = RiskManager::new(config);

        // Existing positions in pattern-matched markets
        let state = MockStateProvider::new()
            .with_position("swing-state-pa", "token-pa", dec!(4000), dec!(0.5))
            .with_position("swing-state-mi", "token-mi", dec!(2000), dec!(0.5));

        // New order in another pattern-matched market (total would be $4000 > $3000)
        let signal = create_test_signal("swing-state-wi", "token-wi", dec!(1000), dec!(0.5));

        let result = risk_manager.validate(&signal, &state).await;

        // Should be rejected or modified since we're already at $3000
        match result {
            Err(RiskError::CorrelatedExposureExceeded(_)) => {}
            Ok(RiskDecision::Modified { .. }) => {}
            other => panic!("Expected rejection or modification, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_correlation_disabled() {
        let mut config = create_config_with_correlation(vec![
            CorrelationGroupConfig {
                name: "election".to_string(),
                markets: vec![
                    "presidential-winner".to_string(),
                    "electoral-count".to_string(),
                ],
            },
        ]);
        config.correlation.enabled = false;

        let risk_manager = RiskManager::new(config);

        // Large position that would exceed limit if correlation was enabled
        let state = MockStateProvider::new()
            .with_position("electoral-count", "token-ec", dec!(10000), dec!(0.5));

        let signal = create_test_signal("presidential-winner", "token-pw", dec!(500), dec!(0.5));

        // Should be approved since correlation is disabled
        let result = risk_manager.validate(&signal, &state).await;
        assert!(matches!(result, Ok(RiskDecision::Approved)));
    }

    #[tokio::test]
    async fn test_correlation_tracker_access() {
        let config = create_config_with_correlation(vec![
            CorrelationGroupConfig {
                name: "test".to_string(),
                markets: vec!["market-a".to_string()],
            },
        ]);

        let risk_manager = RiskManager::new(config);

        assert!(risk_manager.correlation_tracker().is_enabled());
        assert_eq!(risk_manager.correlation_tracker().max_correlated_exposure(), dec!(3000));
    }
}
