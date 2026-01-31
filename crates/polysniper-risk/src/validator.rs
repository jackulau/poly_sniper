//! Risk validation implementation

use crate::correlation::CorrelationTracker;
use crate::kelly::{KellyCalculator, TradeOutcome};
use crate::time_rules::{TimeRuleEngine, TimeRuleResult};
use crate::volatility::VolatilityCalculator;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use polysniper_core::{
    Market, RiskConfig, RiskDecision, RiskError, RiskValidator, StateProvider, TradeSignal,
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
    config: Arc<RwLock<RiskConfig>>,
    halted: Arc<AtomicBool>,
    halt_reason: Arc<RwLock<Option<String>>>,
    /// Recent orders for rate limiting
    recent_orders: Arc<RwLock<VecDeque<OrderRecord>>>,
    /// Volatility calculator for position sizing
    volatility_calculator: VolatilityCalculator,
    /// Kelly criterion calculator for edge-based position sizing
    kelly_calculator: KellyCalculator,
    /// Time-based rule engine
    time_rule_engine: TimeRuleEngine,
    /// Correlation tracker for correlated position limits
    #[allow(dead_code)]
    correlation_tracker: CorrelationTracker,
}

impl RiskManager {
    /// Create a new risk manager
    pub fn new(config: RiskConfig) -> Self {
        let volatility_calculator = VolatilityCalculator::new(config.volatility.clone());
        let kelly_calculator = KellyCalculator::new(config.kelly.clone());
        let time_rule_engine = TimeRuleEngine::new(config.time_rules.clone());
        let correlation_tracker = CorrelationTracker::new(config.correlation.clone());
        Self {
            config: Arc::new(RwLock::new(config)),
            halted: Arc::new(AtomicBool::new(false)),
            halt_reason: Arc::new(RwLock::new(None)),
            recent_orders: Arc::new(RwLock::new(VecDeque::new())),
            volatility_calculator,
            kelly_calculator,
            time_rule_engine,
            correlation_tracker,
        }
    }

    /// Get a reference to the Kelly calculator
    pub fn kelly_calculator(&self) -> &KellyCalculator {
        &self.kelly_calculator
    }

    /// Get a reference to the time rule engine
    pub fn time_rule_engine(&self) -> &TimeRuleEngine {
        &self.time_rule_engine
    }

    /// Get a reference to the correlation tracker
    pub fn correlation_tracker(&self) -> &CorrelationTracker {
        &self.correlation_tracker
    }

    /// Check time rules for a signal and market
    ///
    /// Returns Ok(Some(decision)) if a time rule requires action,
    /// or Ok(None) if no time rules apply and other checks should continue.
    async fn check_time_rules(
        &self,
        signal: &TradeSignal,
        state: &dyn StateProvider,
    ) -> Result<Option<RiskDecision>, RiskError> {
        let market = match state.get_market(&signal.market_id).await {
            Some(m) => m,
            None => {
                // No market info available, skip time rules
                return Ok(None);
            }
        };

        let time_result = self.time_rule_engine.check_signal(signal, &market);

        match time_result {
            TimeRuleResult::Allowed => Ok(None),
            TimeRuleResult::ReduceSize {
                multiplier,
                rule_name,
            } => {
                let new_size = signal.size * multiplier;
                let new_size_usd = signal.size_usd * multiplier;

                if new_size_usd < Decimal::ONE {
                    // Size too small after reduction
                    warn!(
                        signal_id = %signal.id,
                        rule = %rule_name,
                        "Time rule reduced size below minimum, rejecting"
                    );
                    return Ok(Some(RiskDecision::Rejected {
                        reason: format!(
                            "Time rule '{}' reduced size below minimum (multiplier: {})",
                            rule_name, multiplier
                        ),
                    }));
                }

                info!(
                    signal_id = %signal.id,
                    rule = %rule_name,
                    original_size = %signal.size,
                    new_size = %new_size,
                    multiplier = %multiplier,
                    "Time rule applied size reduction"
                );
                Ok(Some(RiskDecision::Modified {
                    new_size,
                    reason: format!(
                        "Time rule '{}' reduced size by {} ({}h before resolution)",
                        rule_name,
                        multiplier,
                        self.hours_until_end(&market)
                    ),
                }))
            }
            TimeRuleResult::BlockNew { rule_name } => {
                warn!(
                    signal_id = %signal.id,
                    rule = %rule_name,
                    "Time rule blocking new position"
                );
                Ok(Some(RiskDecision::Rejected {
                    reason: format!(
                        "Time rule '{}' blocks new positions ({}h before resolution)",
                        rule_name,
                        self.hours_until_end(&market)
                    ),
                }))
            }
            TimeRuleResult::HaltAll { rule_name } => {
                warn!(
                    signal_id = %signal.id,
                    rule = %rule_name,
                    "Time rule halting all trading"
                );
                Ok(Some(RiskDecision::Rejected {
                    reason: format!(
                        "Time rule '{}' halts all trading ({}h before resolution)",
                        rule_name,
                        self.hours_until_end(&market)
                    ),
                }))
            }
        }
    }

    /// Calculate hours until market ends
    fn hours_until_end(&self, market: &Market) -> i64 {
        match market.end_date {
            Some(end) => {
                let duration = end.signed_duration_since(Utc::now());
                duration.num_hours()
            }
            None => -1,
        }
    }

    /// Update the risk configuration at runtime
    ///
    /// This updates risk limits without resetting rate limiting state.
    pub async fn update_config(&self, new_config: RiskConfig) {
        let mut config = self.config.write().await;
        *config = new_config;
        info!(
            max_order_size = %config.max_order_size_usd,
            max_position_size = %config.max_position_size_usd,
            daily_loss_limit = %config.daily_loss_limit_usd,
            circuit_breaker_loss = %config.circuit_breaker_loss_usd,
            max_orders_per_minute = %config.max_orders_per_minute,
            "Risk configuration updated"
        );
    }

    /// Get current config (for inspection)
    pub async fn get_config(&self) -> RiskConfig {
        self.config.read().await.clone()
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

    /// Get the current halt reason, if halted
    pub async fn get_halt_reason(&self) -> Option<String> {
        self.halt_reason.read().await.clone()
    }

    /// Check rate limit
    async fn check_rate_limit(&self, max_orders_per_minute: u32) -> Result<(), RiskError> {
        let current_rate = self.orders_per_minute().await;
        if current_rate >= max_orders_per_minute {
            return Err(RiskError::RateLimitExceeded);
        }
        Ok(())
    }

    /// Check order size limit
    fn check_order_size(
        &self,
        signal: &TradeSignal,
        max_order_size_usd: Decimal,
    ) -> Result<(), RiskError> {
        if signal.size_usd > max_order_size_usd {
            return Err(RiskError::OrderSizeLimitExceeded(format!(
                "Order size ${} exceeds limit ${}",
                signal.size_usd, max_order_size_usd
            )));
        }
        Ok(())
    }

    /// Check position limit
    async fn check_position_limit(
        &self,
        signal: &TradeSignal,
        state: &dyn StateProvider,
        max_position_size_usd: Decimal,
    ) -> Result<(), RiskError> {
        let current_position = state.get_position(&signal.market_id).await;
        let current_value = current_position
            .map(|p| p.size * signal.price.unwrap_or(Decimal::ZERO))
            .unwrap_or(Decimal::ZERO);

        let new_value = current_value + signal.size_usd;

        if new_value > max_position_size_usd {
            return Err(RiskError::PositionLimitExceeded(format!(
                "Position would be ${} exceeding limit ${}",
                new_value, max_position_size_usd
            )));
        }
        Ok(())
    }

    /// Check daily loss limit
    async fn check_daily_loss(
        &self,
        state: &dyn StateProvider,
        daily_loss_limit_usd: Decimal,
        circuit_breaker_loss_usd: Decimal,
    ) -> Result<(), RiskError> {
        let daily_pnl = state.get_daily_pnl().await;

        // Check circuit breaker
        if daily_pnl <= -circuit_breaker_loss_usd {
            self.halt(&format!(
                "Circuit breaker triggered: daily loss ${} exceeds ${}",
                daily_pnl.abs(),
                circuit_breaker_loss_usd
            ));
            return Err(RiskError::CircuitBreakerTriggered(format!(
                "Daily loss ${} exceeds circuit breaker threshold ${}",
                daily_pnl.abs(),
                circuit_breaker_loss_usd
            )));
        }

        // Check daily loss limit
        if daily_pnl <= -daily_loss_limit_usd {
            return Err(RiskError::DailyLossLimitExceeded(format!(
                "Daily loss ${} exceeds limit ${}",
                daily_pnl.abs(),
                daily_loss_limit_usd
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

    /// Calculate volatility-adjusted size based on market conditions
    ///
    /// Returns (adjusted_size, was_modified, reason) if volatility adjustment applies
    async fn calculate_volatility_adjusted_size(
        &self,
        signal: &TradeSignal,
        state: &dyn StateProvider,
    ) -> Option<(Decimal, String)> {
        if !self.volatility_calculator.is_enabled() {
            return None;
        }

        // Get price history for the token
        let price_history = state.get_price_history(&signal.token_id, 1000).await;

        if price_history.is_empty() {
            debug!(
                token_id = %signal.token_id,
                "No price history for volatility calculation, skipping adjustment"
            );
            return None;
        }

        let (adjusted_size, multiplier, volatility) = self
            .volatility_calculator
            .adjust_size(signal.size, &price_history);

        // Only report modification if multiplier is not 1.0
        if multiplier != Decimal::ONE {
            let vol_str = volatility
                .map(|v| format!("{:.2}%", v))
                .unwrap_or_else(|| "unknown".to_string());

            let reason = format!(
                "Volatility-adjusted: size {} -> {} (multiplier: {:.2}, volatility: {})",
                signal.size, adjusted_size, multiplier, vol_str
            );

            info!(
                signal_id = %signal.id,
                token_id = %signal.token_id,
                original_size = %signal.size,
                adjusted_size = %adjusted_size,
                multiplier = %multiplier,
                volatility = ?volatility,
                "Applying volatility-based position sizing"
            );

            return Some((adjusted_size, reason));
        }

        None
    }

    /// Calculate Kelly-adjusted size based on trade history edge
    ///
    /// Returns (adjusted_size, reason) if Kelly adjustment applies.
    /// Kelly sizing is applied AFTER volatility adjustment.
    async fn calculate_kelly_adjusted_size(
        &self,
        current_size: Decimal,
        state: &dyn StateProvider,
    ) -> Option<(Decimal, String)> {
        if !self.kelly_calculator.is_enabled() {
            return None;
        }

        // Get trade history for Kelly calculation
        let config = self.config.read().await;
        let window_size = config.kelly.window_size as usize;
        drop(config);

        let trade_outcomes_raw = state.get_trade_outcomes(window_size).await;

        if trade_outcomes_raw.is_empty() {
            debug!("No trade history for Kelly calculation, skipping adjustment");
            return None;
        }

        // Convert to TradeOutcome structs
        let trade_outcomes: Vec<TradeOutcome> = trade_outcomes_raw
            .iter()
            .map(|(pnl, size_usd)| TradeOutcome::new(*pnl, *size_usd))
            .collect();

        let (adjusted_size, multiplier, reason) = self
            .kelly_calculator
            .adjust_size(current_size, &trade_outcomes);

        // Only report modification if multiplier is not 1.0
        if multiplier != Decimal::ONE {
            if let Some(reason_str) = reason {
                info!(
                    original_size = %current_size,
                    adjusted_size = %adjusted_size,
                    multiplier = %multiplier,
                    trade_count = trade_outcomes.len(),
                    "Applying Kelly criterion position sizing"
                );
                return Some((adjusted_size, reason_str));
            }
        }

        None
    }
}

#[async_trait]
impl RiskValidator for RiskManager {
    async fn validate(
        &self,
        signal: &TradeSignal,
        state: &dyn StateProvider,
    ) -> Result<RiskDecision, RiskError> {
        // Extract config values
        let config = self.config.read().await;
        let max_orders_per_minute = config.max_orders_per_minute;
        let max_order_size_usd = config.max_order_size_usd;
        let max_position_size_usd = config.max_position_size_usd;
        let daily_loss_limit_usd = config.daily_loss_limit_usd;
        let circuit_breaker_loss_usd = config.circuit_breaker_loss_usd;
        drop(config);

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

        // Check time rules first (before other validations)
        if let Some(decision) = self.check_time_rules(signal, state).await? {
            // If time rule requires rejection or modification, return early
            // For Modified, we still need to check other limits on the reduced size
            match &decision {
                RiskDecision::Rejected { .. } => return Ok(decision),
                RiskDecision::Modified { new_size, reason } => {
                    // Create a modified signal to check other limits
                    let price = signal.price.unwrap_or(Decimal::ONE);
                    let new_size_usd = *new_size * price;

                    // Check if modified size passes other limits
                    if new_size_usd > max_order_size_usd {
                        // Still too large, adjust further
                        let final_size = max_order_size_usd / price;
                        return Ok(RiskDecision::Modified {
                            new_size: final_size,
                            reason: format!(
                                "{}; further reduced to order size limit",
                                reason
                            ),
                        });
                    }

                    // The time rule modification is the final result
                    return Ok(decision);
                }
                RiskDecision::Approved => {}
            }
        }

        // Check rate limit
        self.check_rate_limit(max_orders_per_minute).await?;

        // Check daily loss
        self.check_daily_loss(state, daily_loss_limit_usd, circuit_breaker_loss_usd)
            .await?;

        // Check order size
        if let Err(e) = self.check_order_size(signal, max_order_size_usd) {
            // Try to adjust size
            if let Some(new_size) =
                self.calculate_adjusted_size(signal, max_order_size_usd)
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
        if let Err(e) = self.check_position_limit(signal, state, max_position_size_usd).await {
            // Calculate how much room we have
            let current_position = state.get_position(&signal.market_id).await;
            let current_value = current_position
                .map(|p| p.size * signal.price.unwrap_or(Decimal::ZERO))
                .unwrap_or(Decimal::ZERO);

            let remaining_room = max_position_size_usd - current_value;

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

        // Apply volatility-based position sizing first
        let (current_size, mut reasons) = match self
            .calculate_volatility_adjusted_size(signal, state)
            .await
        {
            Some((adjusted_size, reason)) => (adjusted_size, vec![reason]),
            None => (signal.size, vec![]),
        };

        // Apply Kelly criterion sizing after volatility adjustment
        let (final_size, final_reasons) = match self
            .calculate_kelly_adjusted_size(current_size, state)
            .await
        {
            Some((kelly_adjusted, kelly_reason)) => {
                reasons.push(kelly_reason);
                (kelly_adjusted, reasons)
            }
            None => (current_size, reasons),
        };

        // Return modified decision if any sizing adjustments were made
        if !final_reasons.is_empty() {
            return Ok(RiskDecision::Modified {
                new_size: final_size,
                reason: final_reasons.join("; "),
            });
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

        async fn get_trade_outcomes(&self, _limit: usize) -> Vec<(Decimal, Decimal)> {
            // Return empty for existing tests (no Kelly adjustment)
            Vec::new()
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
            volatility: Default::default(),
            kelly: Default::default(),
            time_rules: Default::default(),
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
