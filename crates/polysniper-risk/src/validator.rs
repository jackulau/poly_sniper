//! Risk validation implementation

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
use tracing::{info, warn};

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
}

impl RiskManager {
    /// Create a new risk manager
    pub fn new(config: RiskConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            halted: Arc::new(AtomicBool::new(false)),
            halt_reason: Arc::new(RwLock::new(None)),
            recent_orders: Arc::new(RwLock::new(VecDeque::new())),
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

        // Read config once for this validation
        let config = self.config.read().await;
        let max_orders_per_minute = config.max_orders_per_minute;
        let max_order_size_usd = config.max_order_size_usd;
        let max_position_size_usd = config.max_position_size_usd;
        let daily_loss_limit_usd = config.daily_loss_limit_usd;
        let circuit_breaker_loss_usd = config.circuit_breaker_loss_usd;
        drop(config);

        // Check rate limit
        self.check_rate_limit(max_orders_per_minute).await?;

        // Check daily loss
        self.check_daily_loss(state, daily_loss_limit_usd, circuit_breaker_loss_usd)
            .await?;

        // Check order size
        if let Err(e) = self.check_order_size(signal, max_order_size_usd) {
            // Try to adjust size
            if let Some(new_size) = self.calculate_adjusted_size(signal, max_order_size_usd) {
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
