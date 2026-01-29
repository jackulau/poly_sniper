//! Price Spike Strategy
//!
//! Detects sudden price movements and trades in the direction of momentum.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use polysniper_core::{
    OrderType, Outcome, Priority, Side, StateProvider, Strategy, StrategyError, SystemEvent,
    TokenId, TradeSignal,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Price spike configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceSpikeConfig {
    pub enabled: bool,
    /// Minimum price change percentage to trigger
    pub spike_threshold_pct: Decimal,
    /// Time window in seconds to measure price change
    pub time_window_secs: u64,
    /// Order size in USD
    pub order_size_usd: Decimal,
    /// Cooldown period in seconds after a spike is detected
    #[serde(default = "default_cooldown")]
    pub cooldown_secs: u64,
    /// Minimum liquidity to trade
    #[serde(default = "default_min_liquidity")]
    pub min_liquidity_usd: Decimal,
    /// Markets to monitor (empty = all)
    #[serde(default)]
    pub markets: Vec<String>,
    /// Direction to trade: "momentum" (follow spike) or "reversion" (fade spike)
    #[serde(default = "default_direction")]
    pub trade_direction: TradeDirection,
}

fn default_cooldown() -> u64 {
    60
}

fn default_min_liquidity() -> Decimal {
    dec!(1000)
}

fn default_direction() -> TradeDirection {
    TradeDirection::Momentum
}

/// Trade direction on spike
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TradeDirection {
    /// Trade in direction of spike (momentum)
    Momentum,
    /// Trade against spike (mean reversion)
    Reversion,
}

impl Default for PriceSpikeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            spike_threshold_pct: dec!(5.0),
            time_window_secs: 10,
            order_size_usd: dec!(100),
            cooldown_secs: default_cooldown(),
            min_liquidity_usd: default_min_liquidity(),
            markets: Vec::new(),
            trade_direction: default_direction(),
        }
    }
}

/// Price entry for history tracking
#[derive(Debug, Clone)]
struct PriceEntry {
    price: Decimal,
    timestamp: DateTime<Utc>,
}

/// Cooldown entry
#[derive(Debug, Clone)]
struct CooldownEntry {
    until: DateTime<Utc>,
}

/// Price Spike Strategy
pub struct PriceSpikeStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: PriceSpikeConfig,
    /// Price history per token
    price_history: Arc<RwLock<HashMap<TokenId, VecDeque<PriceEntry>>>>,
    /// Cooldowns per token
    cooldowns: Arc<RwLock<HashMap<TokenId, CooldownEntry>>>,
    /// Token to market mapping
    token_market_map: Arc<RwLock<HashMap<TokenId, (String, Outcome)>>>,
}

impl PriceSpikeStrategy {
    /// Create a new price spike strategy
    pub fn new(config: PriceSpikeConfig) -> Self {
        let enabled = config.enabled;
        Self {
            id: "price_spike".to_string(),
            name: "Price Spike Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config,
            price_history: Arc::new(RwLock::new(HashMap::new())),
            cooldowns: Arc::new(RwLock::new(HashMap::new())),
            token_market_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a token for monitoring
    pub async fn register_token(&self, token_id: TokenId, market_id: String, outcome: Outcome) {
        self.token_market_map
            .write()
            .await
            .insert(token_id, (market_id, outcome));
    }

    /// Check if token is in cooldown
    async fn is_in_cooldown(&self, token_id: &TokenId) -> bool {
        let cooldowns = self.cooldowns.read().await;
        if let Some(entry) = cooldowns.get(token_id) {
            return Utc::now() < entry.until;
        }
        false
    }

    /// Set cooldown for a token
    async fn set_cooldown(&self, token_id: &TokenId) {
        let until = Utc::now() + chrono::Duration::seconds(self.config.cooldown_secs as i64);
        self.cooldowns
            .write()
            .await
            .insert(token_id.clone(), CooldownEntry { until });
    }

    /// Record a price and check for spike
    async fn check_for_spike(&self, token_id: &TokenId, price: Decimal) -> Option<(Decimal, bool)> {
        let mut history = self.price_history.write().await;
        let entries = history
            .entry(token_id.clone())
            .or_insert_with(VecDeque::new);

        let now = Utc::now();
        entries.push_back(PriceEntry {
            price,
            timestamp: now,
        });

        // Remove old entries outside the window
        let cutoff = now - chrono::Duration::seconds(self.config.time_window_secs as i64);
        while entries
            .front()
            .map(|e| e.timestamp < cutoff)
            .unwrap_or(false)
        {
            entries.pop_front();
        }

        // Need at least 2 data points
        if entries.len() < 2 {
            return None;
        }

        // Get oldest price in window
        let oldest = entries.front()?;
        let oldest_price = oldest.price;

        if oldest_price.is_zero() {
            return None;
        }

        // Calculate percentage change
        let change_pct = ((price - oldest_price) / oldest_price) * dec!(100);
        let abs_change = change_pct.abs();

        debug!(
            token_id = %token_id,
            oldest_price = %oldest_price,
            current_price = %price,
            change_pct = %change_pct,
            threshold = %self.config.spike_threshold_pct,
            "Checking for price spike"
        );

        if abs_change >= self.config.spike_threshold_pct {
            let is_spike_up = change_pct > Decimal::ZERO;
            Some((change_pct, is_spike_up))
        } else {
            None
        }
    }

    /// Determine if we should monitor this market
    fn should_monitor_market(&self, market_id: &str) -> bool {
        if self.config.markets.is_empty() {
            return true;
        }
        self.config.markets.contains(&market_id.to_string())
    }
}

#[async_trait]
impl Strategy for PriceSpikeStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn process_event(
        &self,
        event: &SystemEvent,
        state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        let mut signals = Vec::new();

        // Extract price and token info
        let (token_id, price, market_id) = match event {
            SystemEvent::PriceChange(e) => (e.token_id.clone(), e.new_price, e.market_id.clone()),
            SystemEvent::OrderbookUpdate(e) => {
                if let Some(mid) = e.orderbook.mid_price() {
                    (e.token_id.clone(), mid, e.market_id.clone())
                } else {
                    return Ok(signals);
                }
            }
            _ => return Ok(signals),
        };

        // Check if we should monitor this market
        if !self.should_monitor_market(&market_id) {
            return Ok(signals);
        }

        // Check cooldown
        if self.is_in_cooldown(&token_id).await {
            debug!(token_id = %token_id, "Token in cooldown, skipping");
            return Ok(signals);
        }

        // Check for spike
        if let Some((change_pct, is_spike_up)) = self.check_for_spike(&token_id, price).await {
            info!(
                token_id = %token_id,
                market_id = %market_id,
                change_pct = %change_pct,
                direction = if is_spike_up { "UP" } else { "DOWN" },
                "Price spike detected!"
            );

            // Get market info for outcome
            let token_map = self.token_market_map.read().await;
            let outcome = token_map
                .get(&token_id)
                .map(|(_, o)| *o)
                .unwrap_or(Outcome::Yes);
            drop(token_map);

            // Check liquidity if we have state
            if let Some(market) = state.get_market(&market_id).await {
                if market.liquidity < self.config.min_liquidity_usd {
                    warn!(
                        market_id = %market_id,
                        liquidity = %market.liquidity,
                        min_required = %self.config.min_liquidity_usd,
                        "Insufficient liquidity, skipping signal"
                    );
                    return Ok(signals);
                }
            }

            // Determine trade side based on direction config
            let side = match self.config.trade_direction {
                TradeDirection::Momentum => {
                    if is_spike_up {
                        Side::Buy
                    } else {
                        Side::Sell
                    }
                }
                TradeDirection::Reversion => {
                    if is_spike_up {
                        Side::Sell
                    } else {
                        Side::Buy
                    }
                }
            };

            // Calculate size
            let size = if price.is_zero() {
                Decimal::ZERO
            } else {
                self.config.order_size_usd / price
            };

            let signal = TradeSignal {
                id: format!(
                    "sig_spike_{}_{}_{}",
                    token_id,
                    Utc::now().timestamp_millis(),
                    rand_suffix()
                ),
                strategy_id: self.id.clone(),
                market_id,
                token_id: token_id.clone(),
                outcome,
                side,
                price: Some(price),
                size,
                size_usd: self.config.order_size_usd,
                order_type: OrderType::Fok, // Use FOK for spike trading
                priority: Priority::High,   // High priority for time-sensitive
                timestamp: Utc::now(),
                reason: format!(
                    "Price spike {:.2}% {} detected, trading {:?}",
                    change_pct,
                    if is_spike_up { "UP" } else { "DOWN" },
                    self.config.trade_direction
                ),
                metadata: serde_json::json!({
                    "change_pct": change_pct.to_string(),
                    "spike_direction": if is_spike_up { "up" } else { "down" },
                    "trade_direction": format!("{:?}", self.config.trade_direction),
                    "threshold_pct": self.config.spike_threshold_pct.to_string(),
                }),
            };

            signals.push(signal);

            // Set cooldown
            self.set_cooldown(&token_id).await;
        }

        Ok(signals)
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(
            event,
            SystemEvent::PriceChange(_) | SystemEvent::OrderbookUpdate(_)
        )
    }

    async fn initialize(&mut self, state: &dyn StateProvider) -> Result<(), StrategyError> {
        info!(
            strategy_id = %self.id,
            threshold = %self.config.spike_threshold_pct,
            window_secs = %self.config.time_window_secs,
            "Initializing price spike strategy"
        );

        // Register all known markets
        for market in state.get_all_markets().await {
            self.register_token(
                market.yes_token_id.clone(),
                market.condition_id.clone(),
                Outcome::Yes,
            )
            .await;
            self.register_token(market.no_token_id.clone(), market.condition_id, Outcome::No)
                .await;
        }

        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
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
