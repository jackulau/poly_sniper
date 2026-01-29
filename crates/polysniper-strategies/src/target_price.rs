//! Target Price Strategy
//!
//! Executes trades when price reaches configured target levels.

use async_trait::async_trait;
use chrono::Utc;
use polysniper_core::{
    OrderType, Outcome, Priority, Side, StateProvider, Strategy, StrategyError, SystemEvent,
    TokenId, TradeSignal,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Target price configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetPriceConfig {
    pub enabled: bool,
    pub targets: Vec<PriceTarget>,
}

/// Individual price target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceTarget {
    /// Market condition ID
    pub market_id: String,
    /// Token ID to trade
    pub token_id: String,
    /// Outcome (Yes/No)
    pub outcome: Outcome,
    /// Target price to trigger at
    pub target_price: Decimal,
    /// Direction: buy below target, sell above target
    pub direction: TargetDirection,
    /// Order size in USD
    pub size_usd: Decimal,
    /// Order type
    #[serde(default = "default_order_type")]
    pub order_type: OrderType,
    /// Whether this target is one-shot or recurring
    #[serde(default)]
    pub one_shot: bool,
}

fn default_order_type() -> OrderType {
    OrderType::Gtc
}

/// Direction for price targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetDirection {
    /// Buy when price falls below target
    BuyBelow,
    /// Sell when price rises above target
    SellAbove,
}

/// Target Price Strategy
pub struct TargetPriceStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    targets: Arc<RwLock<HashMap<TokenId, Vec<PriceTarget>>>>,
    triggered: Arc<RwLock<HashMap<String, bool>>>,
}

impl TargetPriceStrategy {
    /// Create a new target price strategy
    pub fn new() -> Self {
        Self {
            id: "target_price".to_string(),
            name: "Target Price Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(true)),
            targets: Arc::new(RwLock::new(HashMap::new())),
            triggered: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create from configuration
    pub fn from_config(config: TargetPriceConfig) -> Self {
        let strategy = Self::new();
        strategy.enabled.store(config.enabled, Ordering::SeqCst);

        // Group targets by token ID
        let mut targets_map: HashMap<TokenId, Vec<PriceTarget>> = HashMap::new();
        for target in config.targets {
            targets_map
                .entry(target.token_id.clone())
                .or_default()
                .push(target);
        }

        // Initialize synchronously by creating a new RwLock
        Self {
            targets: Arc::new(RwLock::new(targets_map)),
            ..strategy
        }
    }

    /// Add a price target
    pub async fn add_target(&self, target: PriceTarget) {
        let mut targets = self.targets.write().await;
        targets
            .entry(target.token_id.clone())
            .or_default()
            .push(target);
    }

    /// Remove all targets for a token
    pub async fn remove_targets(&self, token_id: &TokenId) {
        self.targets.write().await.remove(token_id);
    }

    /// Get all targets
    pub async fn get_targets(&self) -> HashMap<TokenId, Vec<PriceTarget>> {
        self.targets.read().await.clone()
    }

    /// Generate a unique key for a target
    fn target_key(target: &PriceTarget) -> String {
        format!(
            "{}:{}:{}:{:?}",
            target.market_id, target.token_id, target.target_price, target.direction
        )
    }

    /// Check if a target should trigger based on current price
    fn should_trigger(target: &PriceTarget, current_price: Decimal) -> bool {
        match target.direction {
            TargetDirection::BuyBelow => current_price <= target.target_price,
            TargetDirection::SellAbove => current_price >= target.target_price,
        }
    }
}

impl Default for TargetPriceStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Strategy for TargetPriceStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn process_event(
        &self,
        event: &SystemEvent,
        _state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        let mut signals = Vec::new();

        // Only process price change events
        let price_event = match event {
            SystemEvent::PriceChange(e) => e,
            SystemEvent::OrderbookUpdate(e) => {
                // Also check orderbook mid price
                if let Some(mid) = e.orderbook.mid_price() {
                    // Create a synthetic price event
                    let targets = self.targets.read().await;
                    if let Some(token_targets) = targets.get(&e.token_id) {
                        for target in token_targets {
                            if Self::should_trigger(target, mid) {
                                let key = Self::target_key(target);

                                // Check if already triggered (for one-shot)
                                let triggered = self.triggered.read().await;
                                if target.one_shot && triggered.get(&key).copied().unwrap_or(false) {
                                    continue;
                                }
                                drop(triggered);

                                let signal = self.create_signal(target, mid);
                                signals.push(signal);

                                // Mark as triggered
                                if target.one_shot {
                                    self.triggered.write().await.insert(key, true);
                                }
                            }
                        }
                    }
                }
                return Ok(signals);
            }
            _ => return Ok(signals),
        };

        let current_price = price_event.new_price;
        let token_id = &price_event.token_id;

        // Check if any targets are hit
        let targets = self.targets.read().await;
        if let Some(token_targets) = targets.get(token_id) {
            for target in token_targets {
                if Self::should_trigger(target, current_price) {
                    let key = Self::target_key(target);

                    // Check if already triggered (for one-shot)
                    let triggered = self.triggered.read().await;
                    if target.one_shot && triggered.get(&key).copied().unwrap_or(false) {
                        debug!(
                            target_key = %key,
                            "Target already triggered (one-shot)"
                        );
                        continue;
                    }
                    drop(triggered);

                    info!(
                        market_id = %target.market_id,
                        token_id = %target.token_id,
                        target_price = %target.target_price,
                        current_price = %current_price,
                        direction = ?target.direction,
                        "Price target hit!"
                    );

                    let signal = self.create_signal(target, current_price);
                    signals.push(signal);

                    // Mark as triggered
                    if target.one_shot {
                        self.triggered.write().await.insert(key, true);
                    }
                }
            }
        }

        Ok(signals)
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(
            event,
            SystemEvent::PriceChange(_) | SystemEvent::OrderbookUpdate(_)
        )
    }

    async fn initialize(&mut self, _state: &dyn StateProvider) -> Result<(), StrategyError> {
        info!(
            strategy_id = %self.id,
            "Initializing target price strategy"
        );
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }

    async fn reload_config(&mut self, config_content: &str) -> Result<(), StrategyError> {
        let config: TargetPriceConfig = toml::from_str(config_content).map_err(|e| {
            StrategyError::ConfigError(format!("Failed to parse target_price config: {}", e))
        })?;

        // Validate config
        for target in &config.targets {
            if target.size_usd <= Decimal::ZERO {
                return Err(StrategyError::ConfigError(format!(
                    "Invalid size_usd {} for target {}",
                    target.size_usd, target.market_id
                )));
            }
            if target.target_price <= Decimal::ZERO || target.target_price >= Decimal::ONE {
                return Err(StrategyError::ConfigError(format!(
                    "Invalid target_price {} for target {} (must be between 0 and 1)",
                    target.target_price, target.market_id
                )));
            }
        }

        // Update enabled state
        self.enabled.store(config.enabled, Ordering::SeqCst);

        // Group targets by token ID
        let mut targets_map: HashMap<TokenId, Vec<PriceTarget>> = HashMap::new();
        for target in config.targets {
            targets_map
                .entry(target.token_id.clone())
                .or_default()
                .push(target);
        }

        // Update targets atomically
        let mut targets = self.targets.write().await;
        *targets = targets_map;

        info!(
            strategy_id = %self.id,
            target_count = %targets.values().map(|v| v.len()).sum::<usize>(),
            "Reloaded target_price configuration"
        );

        Ok(())
    }

    fn config_name(&self) -> &str {
        "target_price"
    }
}

impl TargetPriceStrategy {
    fn create_signal(&self, target: &PriceTarget, current_price: Decimal) -> TradeSignal {
        let side = match target.direction {
            TargetDirection::BuyBelow => Side::Buy,
            TargetDirection::SellAbove => Side::Sell,
        };

        // Calculate size from USD
        let price_for_calc = if current_price.is_zero() {
            target.target_price
        } else {
            current_price
        };
        let size = if price_for_calc.is_zero() {
            Decimal::ZERO
        } else {
            target.size_usd / price_for_calc
        };

        TradeSignal {
            id: format!(
                "sig_tp_{}_{}_{}",
                target.market_id,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: target.market_id.clone(),
            token_id: target.token_id.clone(),
            outcome: target.outcome,
            side,
            price: Some(target.target_price),
            size,
            size_usd: target.size_usd,
            order_type: target.order_type,
            priority: Priority::Normal,
            timestamp: Utc::now(),
            reason: format!(
                "Price {} target {} at {} (current: {})",
                match target.direction {
                    TargetDirection::BuyBelow => "below",
                    TargetDirection::SellAbove => "above",
                },
                target.target_price,
                target.token_id,
                current_price
            ),
            metadata: serde_json::json!({
                "target_price": target.target_price.to_string(),
                "current_price": current_price.to_string(),
                "direction": format!("{:?}", target.direction),
            }),
        }
    }
}

fn rand_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{:08x}", nanos)
}
