//! New Market Strategy
//!
//! Detects and enters newly created markets early.

use async_trait::async_trait;
use chrono::Utc;
use polysniper_core::{
    Market, MarketId, OrderType, Outcome, Priority, Side, StateProvider, Strategy, StrategyError,
    SystemEvent, TradeSignal,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// New market strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMarketConfig {
    pub enabled: bool,
    /// Order size in USD
    pub order_size_usd: Decimal,
    /// Maximum market age in seconds to consider "new"
    pub max_age_secs: u64,
    /// Minimum liquidity before entering
    #[serde(default = "default_min_liquidity")]
    pub min_liquidity_usd: Decimal,
    /// Keywords to filter (empty = all markets)
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Categories to filter
    #[serde(default)]
    pub categories: Vec<String>,
    /// Default side to take
    #[serde(default = "default_side")]
    pub default_side: Side,
    /// Default outcome to trade
    #[serde(default = "default_outcome")]
    pub default_outcome: Outcome,
    /// Maximum price to pay (for buying Yes)
    #[serde(default = "default_max_entry_price")]
    pub max_entry_price: Decimal,
}

fn default_min_liquidity() -> Decimal {
    dec!(100)
}

fn default_side() -> Side {
    Side::Buy
}

fn default_outcome() -> Outcome {
    Outcome::Yes
}

fn default_max_entry_price() -> Decimal {
    dec!(0.50)
}

impl Default for NewMarketConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            order_size_usd: dec!(100),
            max_age_secs: 300, // 5 minutes
            min_liquidity_usd: default_min_liquidity(),
            keywords: Vec::new(),
            categories: Vec::new(),
            default_side: default_side(),
            default_outcome: default_outcome(),
            max_entry_price: default_max_entry_price(),
        }
    }
}

/// New Market Strategy
pub struct NewMarketStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: Arc<RwLock<NewMarketConfig>>,
    /// Set of known market IDs
    known_markets: Arc<RwLock<HashSet<MarketId>>>,
    /// Set of markets we've already generated signals for
    signaled_markets: Arc<RwLock<HashSet<MarketId>>>,
}

impl NewMarketStrategy {
    /// Create a new market strategy
    pub fn new(config: NewMarketConfig) -> Self {
        let enabled = config.enabled;
        Self {
            id: "new_market".to_string(),
            name: "New Market Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config: Arc::new(RwLock::new(config)),
            known_markets: Arc::new(RwLock::new(HashSet::new())),
            signaled_markets: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Add a known market
    pub async fn add_known_market(&self, market_id: MarketId) {
        self.known_markets.write().await.insert(market_id);
    }

    /// Check if market is new (not previously known)
    async fn is_new_market(&self, market_id: &MarketId) -> bool {
        !self.known_markets.read().await.contains(market_id)
    }

    /// Check if we've already signaled for this market
    async fn has_signaled(&self, market_id: &MarketId) -> bool {
        self.signaled_markets.read().await.contains(market_id)
    }

    /// Mark market as signaled
    async fn mark_signaled(&self, market_id: &MarketId) {
        self.signaled_markets
            .write()
            .await
            .insert(market_id.clone());
    }

    /// Check if market matches our filters (using pre-fetched config values)
    fn matches_filters_sync(
        &self,
        market: &Market,
        keywords: &[String],
        categories: &[String],
    ) -> bool {
        // Check keywords
        if !keywords.is_empty() {
            let question_lower = market.question.to_lowercase();
            let desc_lower = market
                .description
                .as_ref()
                .map(|d| d.to_lowercase())
                .unwrap_or_default();

            let has_keyword = keywords.iter().any(|kw| {
                let kw_lower = kw.to_lowercase();
                question_lower.contains(&kw_lower) || desc_lower.contains(&kw_lower)
            });

            if !has_keyword {
                return false;
            }
        }

        // Check categories/tags
        if !self.config.categories.is_empty() {
            let has_category = self
                .config
                .categories
                .iter()
                .any(|cat| market.tags.iter().any(|tag| tag.eq_ignore_ascii_case(cat)));

            if !has_category {
                return false;
            }
        }

        true
    }

    /// Check if market is young enough (using pre-fetched config value)
    fn is_young_enough_sync(&self, market: &Market, max_age_secs: u64) -> bool {
        let age = Utc::now() - market.created_at;
        age.num_seconds() <= max_age_secs as i64
    }
}

#[async_trait]
impl Strategy for NewMarketStrategy {
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

        // Only process new market events
        let new_market_event = match event {
            SystemEvent::NewMarket(e) => e,
            _ => return Ok(signals),
        };

        let market = &new_market_event.market;

        // Check if we already know this market
        if !self.is_new_market(&market.condition_id).await {
            debug!(
                market_id = %market.condition_id,
                "Market already known, skipping"
            );
            return Ok(signals);
        }

        // Add to known markets
        self.add_known_market(market.condition_id.clone()).await;

        // Check if we've already signaled
        if self.has_signaled(&market.condition_id).await {
            debug!(
                market_id = %market.condition_id,
                "Already generated signal for this market"
            );
            return Ok(signals);
        }

        // Check if market is active
        if !market.active || market.closed {
            debug!(
                market_id = %market.condition_id,
                "Market not active or closed, skipping"
            );
            return Ok(signals);
        }

        // Read config once for this event processing
        let config = self.config.read().await;
        let max_age_secs = config.max_age_secs;
        let keywords = config.keywords.clone();
        let categories = config.categories.clone();
        let min_liquidity_usd = config.min_liquidity_usd;
        let default_outcome = config.default_outcome;
        let default_side = config.default_side;
        let max_entry_price = config.max_entry_price;
        let order_size_usd = config.order_size_usd;
        drop(config);

        // Check age
        if !self.is_young_enough_sync(market, max_age_secs) {
            debug!(
                market_id = %market.condition_id,
                "Market too old, skipping"
            );
            return Ok(signals);
        }

        // Check filters
        if !self.matches_filters_sync(market, &keywords, &categories) {
            debug!(
                market_id = %market.condition_id,
                "Market doesn't match filters, skipping"
            );
            return Ok(signals);
        }

        // Check liquidity
        if market.liquidity < min_liquidity_usd {
            debug!(
                market_id = %market.condition_id,
                liquidity = %market.liquidity,
                required = %min_liquidity_usd,
                "Insufficient liquidity, skipping"
            );
            return Ok(signals);
        }

        // Check current price if we can
        let token_id = match default_outcome {
            Outcome::Yes => &market.yes_token_id,
            Outcome::No => &market.no_token_id,
        };

        let current_price = state.get_price(token_id).await;

        // For buying, check max entry price
        if default_side == Side::Buy {
            if let Some(price) = current_price {
                if price > max_entry_price {
                    warn!(
                        market_id = %market.condition_id,
                        price = %price,
                        max_price = %max_entry_price,
                        "Price too high for entry, skipping"
                    );
                    return Ok(signals);
                }
            }
        }

        info!(
            market_id = %market.condition_id,
            question = %market.question,
            liquidity = %market.liquidity,
            "New market detected! Generating signal"
        );

        // Calculate size
        let entry_price = current_price.unwrap_or(max_entry_price);
        let size = if entry_price.is_zero() {
            Decimal::ZERO
        } else {
            order_size_usd / entry_price
        };

        let signal = TradeSignal {
            id: format!(
                "sig_new_{}_{}_{}",
                market.condition_id,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: market.condition_id.clone(),
            token_id: token_id.clone(),
            outcome: default_outcome,
            side: default_side,
            price: Some(entry_price),
            size,
            size_usd: self.config.order_size_usd,
            order_type: OrderType::Fok,   // Use FOK for speed
            priority: Priority::Critical, // Critical priority for new markets
            timestamp: Utc::now(),
            reason: format!("New market detected: {}", market.question),
            metadata: serde_json::json!({
                "question": market.question,
                "liquidity": market.liquidity.to_string(),
                "created_at": market.created_at.to_rfc3339(),
                "tags": market.tags,
            }),
        };

        signals.push(signal);

        // Mark as signaled
        self.mark_signaled(&market.condition_id).await;

        Ok(signals)
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(event, SystemEvent::NewMarket(_))
    }

    async fn initialize(&mut self, state: &dyn StateProvider) -> Result<(), StrategyError> {
        let config = self.config.read().await;
        info!(
            strategy_id = %self.id,
            max_age_secs = %config.max_age_secs,
            "Initializing new market strategy"
        );
        drop(config);

        // Load all existing markets as "known"
        for market in state.get_all_markets().await {
            self.add_known_market(market.condition_id).await;
        }

        let known_count = self.known_markets.read().await.len();
        info!(
            strategy_id = %self.id,
            known_markets = known_count,
            "Loaded existing markets"
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
        let new_config: NewMarketConfig = toml::from_str(config_content).map_err(|e| {
            StrategyError::ConfigError(format!("Failed to parse new_market config: {}", e))
        })?;

        // Validate config
        if new_config.order_size_usd <= Decimal::ZERO {
            return Err(StrategyError::ConfigError(
                "order_size_usd must be positive".to_string(),
            ));
        }
        if new_config.max_age_secs == 0 {
            return Err(StrategyError::ConfigError(
                "max_age_secs must be positive".to_string(),
            ));
        }
        if new_config.max_entry_price <= Decimal::ZERO || new_config.max_entry_price >= Decimal::ONE
        {
            return Err(StrategyError::ConfigError(
                "max_entry_price must be between 0 and 1".to_string(),
            ));
        }

        // Update enabled state
        self.enabled.store(new_config.enabled, Ordering::SeqCst);

        // Update config atomically
        let mut config = self.config.write().await;
        *config = new_config;

        info!(
            strategy_id = %self.id,
            max_age_secs = %config.max_age_secs,
            order_size_usd = %config.order_size_usd,
            "Reloaded new_market configuration"
        );

        Ok(())
    }

    fn config_name(&self) -> &str {
        "new_market"
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
