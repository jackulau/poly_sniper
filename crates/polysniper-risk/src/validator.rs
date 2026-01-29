//! Risk validation implementation

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
use tracing::{info, warn};

use crate::time_rules::{TimeRuleEngine, TimeRuleResult};

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
    /// Time rule engine for time-based restrictions
    time_rule_engine: TimeRuleEngine,
}

impl RiskManager {
    /// Create a new risk manager
    pub fn new(config: RiskConfig) -> Self {
        let time_rule_engine = TimeRuleEngine::new(config.time_rules.clone());
        Self {
            config,
            halted: Arc::new(AtomicBool::new(false)),
            halt_reason: Arc::new(RwLock::new(None)),
            recent_orders: Arc::new(RwLock::new(VecDeque::new())),
            time_rule_engine,
        }
    }

    /// Get a reference to the time rule engine
    pub fn time_rule_engine(&self) -> &TimeRuleEngine {
        &self.time_rule_engine
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
                    if new_size_usd > self.config.max_order_size_usd {
                        // Still too large, adjust further
                        let final_size = self.config.max_order_size_usd / price;
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
        self.check_rate_limit().await?;

        // Check daily loss
        self.check_daily_loss(state).await?;

        // Check order size
        if let Err(e) = self.check_order_size(signal) {
            // Try to adjust size
            if let Some(new_size) = self.calculate_adjusted_size(signal, self.config.max_order_size_usd) {
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
