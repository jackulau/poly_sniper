//! Multi-Leg Correlated Market Strategy
//!
//! Identifies and trades correlated positions across related markets,
//! such as election outcomes where results in one market imply results in another.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use polysniper_core::{
    MarketId, OrderType, Outcome, Priority, Side, StateProvider, Strategy, StrategyError,
    SystemEvent, TokenId, TradeSignal,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Multi-leg strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLegConfig {
    pub enabled: bool,
    /// Correlation rules defining relationships between markets
    pub correlation_rules: Vec<CorrelationRule>,
    /// Minimum edge (mispricing) percentage to trigger a trade
    #[serde(default = "default_min_edge")]
    pub min_edge_pct: Decimal,
    /// Order size in USD per leg
    #[serde(default = "default_order_size")]
    pub order_size_usd: Decimal,
    /// Cooldown period in seconds after generating signals for a rule
    #[serde(default = "default_cooldown")]
    pub cooldown_secs: u64,
    /// Minimum liquidity in USD required for markets
    #[serde(default = "default_min_liquidity")]
    pub min_liquidity_usd: Decimal,
}

fn default_min_edge() -> Decimal {
    dec!(0.02) // 2% minimum edge
}

fn default_order_size() -> Decimal {
    dec!(50)
}

fn default_cooldown() -> u64 {
    120 // 2 minutes
}

fn default_min_liquidity() -> Decimal {
    dec!(1000)
}

impl Default for MultiLegConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            correlation_rules: Vec::new(),
            min_edge_pct: default_min_edge(),
            order_size_usd: default_order_size(),
            cooldown_secs: default_cooldown(),
            min_liquidity_usd: default_min_liquidity(),
        }
    }
}

/// A correlation rule defining how multiple markets relate to each other
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationRule {
    /// Human-readable name for this rule
    pub name: String,
    /// The legs (markets) involved in this correlation
    pub legs: Vec<CorrelationLeg>,
    /// The type of relationship between the markets
    pub relationship: CorrelationRelationship,
    /// Expected sum for mutually exclusive markets (default: 1.0)
    #[serde(default)]
    pub expected_sum: Option<Decimal>,
    /// Tolerance for pricing deviations (default: 0.01)
    #[serde(default = "default_tolerance")]
    pub tolerance: Decimal,
}

fn default_tolerance() -> Decimal {
    dec!(0.01) // 1% tolerance
}

/// A single leg in a correlation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationLeg {
    /// Market condition ID
    pub market_id: String,
    /// Which outcome token to use (Yes or No)
    pub token: Outcome,
    /// Weight multiplier for this leg (default: 1.0)
    #[serde(default = "default_weight")]
    pub weight: Decimal,
}

fn default_weight() -> Decimal {
    Decimal::ONE
}

/// Types of correlation relationships between markets
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrelationRelationship {
    /// Sum of YES tokens should equal expected_sum (e.g., Biden + Trump + Other = 1)
    MutuallyExclusive,
    /// Markets move in opposite directions (e.g., A YES ~ B NO)
    Inverse,
    /// A implies B: A's YES price should be <= B's YES price
    Conditional,
}

/// Cached market data for correlation calculations
#[derive(Debug, Clone)]
struct MarketCache {
    market_id: MarketId,
    yes_token_id: TokenId,
    no_token_id: TokenId,
    yes_price: Option<Decimal>,
    no_price: Option<Decimal>,
    liquidity: Decimal,
}

/// Cooldown entry for a rule
struct CooldownEntry {
    until: DateTime<Utc>,
}

/// Multi-Leg Correlated Market Strategy
pub struct MultiLegStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: MultiLegConfig,
    /// Cooldowns per rule (by rule name)
    cooldowns: Arc<RwLock<HashMap<String, CooldownEntry>>>,
    /// Counter for generating unique signal group IDs
    signal_counter: Arc<RwLock<u64>>,
}

impl MultiLegStrategy {
    /// Create a new multi-leg strategy
    pub fn new(config: MultiLegConfig) -> Self {
        let enabled = config.enabled;
        Self {
            id: "multi_leg".to_string(),
            name: "Multi-Leg Correlated Market Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config,
            cooldowns: Arc::new(RwLock::new(HashMap::new())),
            signal_counter: Arc::new(RwLock::new(0)),
        }
    }

    /// Check if a rule is in cooldown
    async fn is_in_cooldown(&self, rule_name: &str) -> bool {
        let cooldowns = self.cooldowns.read().await;
        if let Some(entry) = cooldowns.get(rule_name) {
            return Utc::now() < entry.until;
        }
        false
    }

    /// Set cooldown for a rule
    async fn set_cooldown(&self, rule_name: &str) {
        let until = Utc::now() + chrono::Duration::seconds(self.config.cooldown_secs as i64);
        self.cooldowns
            .write()
            .await
            .insert(rule_name.to_string(), CooldownEntry { until });
    }

    /// Generate a unique signal group ID
    async fn next_signal_group_id(&self) -> String {
        let mut counter = self.signal_counter.write().await;
        *counter += 1;
        format!(
            "grp_multileg_{}_{}",
            Utc::now().timestamp_millis(),
            *counter
        )
    }

    /// Get market cache for a leg
    async fn get_leg_market_cache(
        &self,
        leg: &CorrelationLeg,
        state: &dyn StateProvider,
    ) -> Option<MarketCache> {
        let market = state.get_market(&leg.market_id).await?;

        let yes_price = state.get_price(&market.yes_token_id).await;
        let no_price = state.get_price(&market.no_token_id).await;

        Some(MarketCache {
            market_id: market.condition_id,
            yes_token_id: market.yes_token_id,
            no_token_id: market.no_token_id,
            yes_price,
            no_price,
            liquidity: market.liquidity,
        })
    }

    /// Get the relevant price for a leg based on its token type
    fn get_leg_price(leg: &CorrelationLeg, cache: &MarketCache) -> Option<Decimal> {
        match leg.token {
            Outcome::Yes => cache.yes_price,
            Outcome::No => cache.no_price,
        }
    }

    /// Check mutually exclusive relationship
    /// For markets that sum to 1 (e.g., candidate A + candidate B + candidate C = 1)
    async fn check_mutually_exclusive(
        &self,
        rule: &CorrelationRule,
        state: &dyn StateProvider,
    ) -> Option<Vec<TradeSignal>> {
        let mut caches = Vec::new();
        let mut total_price = Decimal::ZERO;
        let mut min_liquidity = Decimal::MAX;

        // Gather prices for all legs
        for leg in &rule.legs {
            let cache = self.get_leg_market_cache(leg, state).await?;
            let price = Self::get_leg_price(leg, &cache)?;
            total_price += price * leg.weight;
            min_liquidity = min_liquidity.min(cache.liquidity);
            caches.push((leg.clone(), cache, price));
        }

        // Check liquidity requirement
        if min_liquidity < self.config.min_liquidity_usd {
            debug!(
                rule = %rule.name,
                min_liquidity = %min_liquidity,
                required = %self.config.min_liquidity_usd,
                "Insufficient liquidity for mutually exclusive rule"
            );
            return None;
        }

        let expected = rule.expected_sum.unwrap_or(Decimal::ONE);
        let deviation = total_price - expected;
        let edge = deviation.abs();

        debug!(
            rule = %rule.name,
            total_price = %total_price,
            expected = %expected,
            edge = %edge,
            "Checking mutually exclusive relationship"
        );

        if edge < self.config.min_edge_pct {
            return None;
        }

        info!(
            rule = %rule.name,
            total_price = %total_price,
            expected = %expected,
            edge = %edge,
            "Mutually exclusive mispricing detected!"
        );

        // Generate signals based on deviation direction
        let group_id = self.next_signal_group_id().await;
        let mut signals = Vec::new();
        let leg_count = caches.len();

        if deviation > Decimal::ZERO {
            // Total is above expected - look for overpriced legs to sell
            // Find the most overpriced leg (highest price relative to fair value)
            if let Some((idx, (leg, cache, price))) = caches
                .iter()
                .enumerate()
                .max_by(|(_, (_, _, a)), (_, (_, _, b))| a.cmp(b))
            {
                let signal = self.create_trade_signal(
                    &group_id,
                    idx,
                    leg_count,
                    leg,
                    cache,
                    *price,
                    Side::Sell,
                    &rule.name,
                    format!(
                        "Mutually exclusive overpriced: sum {:.4} > expected {:.4}",
                        total_price, expected
                    ),
                );
                signals.push(signal);
            }
        } else {
            // Total is below expected - look for underpriced legs to buy
            // Find the most underpriced leg (lowest price relative to fair value)
            if let Some((idx, (leg, cache, price))) = caches
                .iter()
                .enumerate()
                .min_by(|(_, (_, _, a)), (_, (_, _, b))| a.cmp(b))
            {
                let signal = self.create_trade_signal(
                    &group_id,
                    idx,
                    leg_count,
                    leg,
                    cache,
                    *price,
                    Side::Buy,
                    &rule.name,
                    format!(
                        "Mutually exclusive underpriced: sum {:.4} < expected {:.4}",
                        total_price, expected
                    ),
                );
                signals.push(signal);
            }
        }

        if signals.is_empty() {
            None
        } else {
            Some(signals)
        }
    }

    /// Check inverse correlation relationship
    /// For markets where A YES ~ B NO (or A goes up when B goes down)
    async fn check_inverse(
        &self,
        rule: &CorrelationRule,
        state: &dyn StateProvider,
    ) -> Option<Vec<TradeSignal>> {
        if rule.legs.len() != 2 {
            warn!(
                rule = %rule.name,
                leg_count = rule.legs.len(),
                "Inverse relationship requires exactly 2 legs"
            );
            return None;
        }

        let leg_a = &rule.legs[0];
        let leg_b = &rule.legs[1];

        let cache_a = self.get_leg_market_cache(leg_a, state).await?;
        let cache_b = self.get_leg_market_cache(leg_b, state).await?;

        let price_a = Self::get_leg_price(leg_a, &cache_a)?;
        let price_b = Self::get_leg_price(leg_b, &cache_b)?;

        // Check liquidity
        let min_liquidity = cache_a.liquidity.min(cache_b.liquidity);
        if min_liquidity < self.config.min_liquidity_usd {
            debug!(
                rule = %rule.name,
                min_liquidity = %min_liquidity,
                "Insufficient liquidity for inverse rule"
            );
            return None;
        }

        // For inverse correlation: price_a + price_b should be close to 1.0
        // (if A YES is high, B YES should be low, and vice versa)
        let sum = (price_a * leg_a.weight) + (price_b * leg_b.weight);
        let expected = rule.expected_sum.unwrap_or(Decimal::ONE);
        let deviation = sum - expected;
        let edge = deviation.abs();

        debug!(
            rule = %rule.name,
            price_a = %price_a,
            price_b = %price_b,
            sum = %sum,
            expected = %expected,
            edge = %edge,
            "Checking inverse relationship"
        );

        if edge < self.config.min_edge_pct {
            return None;
        }

        info!(
            rule = %rule.name,
            price_a = %price_a,
            price_b = %price_b,
            edge = %edge,
            "Inverse correlation mispricing detected!"
        );

        // Generate coordinated spread trade
        let group_id = self.next_signal_group_id().await;
        let mut signals = Vec::new();

        if deviation > Decimal::ZERO {
            // Sum is too high - sell both (or sell the more expensive one)
            // Typically in inverse correlation, we'd do a spread: sell high, buy low
            // Sell the more expensive leg
            let (sell_leg, sell_cache, sell_price, buy_leg, buy_cache, buy_price) =
                if price_a > price_b {
                    (leg_a, &cache_a, price_a, leg_b, &cache_b, price_b)
                } else {
                    (leg_b, &cache_b, price_b, leg_a, &cache_a, price_a)
                };

            signals.push(self.create_trade_signal(
                &group_id,
                0,
                2,
                sell_leg,
                sell_cache,
                sell_price,
                Side::Sell,
                &rule.name,
                format!("Inverse spread: sell high leg at {:.4}", sell_price),
            ));

            signals.push(self.create_trade_signal(
                &group_id,
                1,
                2,
                buy_leg,
                buy_cache,
                buy_price,
                Side::Buy,
                &rule.name,
                format!("Inverse spread: buy low leg at {:.4}", buy_price),
            ));
        } else {
            // Sum is too low - buy both would be unusual
            // In inverse correlation, this means both are underpriced
            // Buy the one with more upside potential (lower price)
            let (buy_leg, buy_cache, buy_price) = if price_a < price_b {
                (leg_a, &cache_a, price_a)
            } else {
                (leg_b, &cache_b, price_b)
            };

            signals.push(self.create_trade_signal(
                &group_id,
                0,
                1,
                buy_leg,
                buy_cache,
                buy_price,
                Side::Buy,
                &rule.name,
                format!("Inverse underpriced: buy at {:.4}", buy_price),
            ));
        }

        if signals.is_empty() {
            None
        } else {
            Some(signals)
        }
    }

    /// Check conditional/implied relationship
    /// For markets where A implies B (e.g., "Biden wins" implies "Democrat wins")
    /// A's YES price should be <= B's YES price
    async fn check_conditional(
        &self,
        rule: &CorrelationRule,
        state: &dyn StateProvider,
    ) -> Option<Vec<TradeSignal>> {
        if rule.legs.len() != 2 {
            warn!(
                rule = %rule.name,
                leg_count = rule.legs.len(),
                "Conditional relationship requires exactly 2 legs (implication: A => B)"
            );
            return None;
        }

        // First leg is the antecedent (A), second is the consequent (B)
        // A => B means P(A) <= P(B) for YES outcomes
        let leg_a = &rule.legs[0]; // The specific outcome (e.g., "Biden wins")
        let leg_b = &rule.legs[1]; // The implied outcome (e.g., "Democrat wins")

        let cache_a = self.get_leg_market_cache(leg_a, state).await?;
        let cache_b = self.get_leg_market_cache(leg_b, state).await?;

        let price_a = Self::get_leg_price(leg_a, &cache_a)?;
        let price_b = Self::get_leg_price(leg_b, &cache_b)?;

        // Check liquidity
        let min_liquidity = cache_a.liquidity.min(cache_b.liquidity);
        if min_liquidity < self.config.min_liquidity_usd {
            debug!(
                rule = %rule.name,
                min_liquidity = %min_liquidity,
                "Insufficient liquidity for conditional rule"
            );
            return None;
        }

        // Apply weights
        let weighted_a = price_a * leg_a.weight;
        let weighted_b = price_b * leg_b.weight;

        // For implication A => B: price_a should be <= price_b (with tolerance)
        // If price_a > price_b + tolerance, there's a mispricing
        let violation = weighted_a - weighted_b - rule.tolerance;

        debug!(
            rule = %rule.name,
            price_a = %price_a,
            price_b = %price_b,
            weighted_a = %weighted_a,
            weighted_b = %weighted_b,
            violation = %violation,
            "Checking conditional relationship"
        );

        if violation <= Decimal::ZERO {
            // No violation - implication holds
            return None;
        }

        if violation < self.config.min_edge_pct {
            // Violation exists but not large enough
            return None;
        }

        info!(
            rule = %rule.name,
            price_a = %price_a,
            price_b = %price_b,
            violation = %violation,
            "Conditional implication violated!"
        );

        // Generate spread trade: buy the consequent (B), sell the antecedent (A)
        // This exploits the mispricing where A is overpriced relative to B
        let group_id = self.next_signal_group_id().await;
        let mut signals = Vec::new();

        // Sell A (overpriced antecedent)
        signals.push(self.create_trade_signal(
            &group_id,
            0,
            2,
            leg_a,
            &cache_a,
            price_a,
            Side::Sell,
            &rule.name,
            format!(
                "Conditional violation: sell antecedent at {:.4} (> consequent {:.4})",
                price_a, price_b
            ),
        ));

        // Buy B (underpriced consequent)
        signals.push(self.create_trade_signal(
            &group_id,
            1,
            2,
            leg_b,
            &cache_b,
            price_b,
            Side::Buy,
            &rule.name,
            format!(
                "Conditional violation: buy consequent at {:.4} (should be >= antecedent)",
                price_b
            ),
        ));

        Some(signals)
    }

    /// Create a trade signal for a leg
    #[allow(clippy::too_many_arguments)]
    fn create_trade_signal(
        &self,
        group_id: &str,
        leg_index: usize,
        total_legs: usize,
        leg: &CorrelationLeg,
        cache: &MarketCache,
        price: Decimal,
        side: Side,
        rule_name: &str,
        reason: String,
    ) -> TradeSignal {
        let token_id = match leg.token {
            Outcome::Yes => cache.yes_token_id.clone(),
            Outcome::No => cache.no_token_id.clone(),
        };

        let size = if price.is_zero() {
            Decimal::ZERO
        } else {
            self.config.order_size_usd / price
        };

        TradeSignal {
            id: format!(
                "sig_multileg_{}_{}_{}_{}",
                cache.market_id,
                leg_index,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: cache.market_id.clone(),
            token_id,
            outcome: leg.token,
            side,
            price: Some(price),
            size,
            size_usd: self.config.order_size_usd,
            order_type: OrderType::Gtc, // Use GTC for spread trades
            priority: Priority::Normal,
            timestamp: Utc::now(),
            reason,
            metadata: serde_json::json!({
                "rule_name": rule_name,
                "group_id": group_id,
                "leg_index": leg_index,
                "total_legs": total_legs,
                "leg_weight": leg.weight.to_string(),
            }),
        }
    }

    /// Process a correlation rule
    async fn process_rule(
        &self,
        rule: &CorrelationRule,
        state: &dyn StateProvider,
    ) -> Option<Vec<TradeSignal>> {
        // Check cooldown
        if self.is_in_cooldown(&rule.name).await {
            debug!(rule = %rule.name, "Rule in cooldown, skipping");
            return None;
        }

        let signals = match &rule.relationship {
            CorrelationRelationship::MutuallyExclusive => {
                self.check_mutually_exclusive(rule, state).await
            }
            CorrelationRelationship::Inverse => self.check_inverse(rule, state).await,
            CorrelationRelationship::Conditional => self.check_conditional(rule, state).await,
        };

        // Set cooldown if signals were generated
        if signals.is_some() {
            self.set_cooldown(&rule.name).await;
        }

        signals
    }
}

#[async_trait]
impl Strategy for MultiLegStrategy {
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
        let mut all_signals = Vec::new();

        // We process on price changes or orderbook updates
        let market_id = match event {
            SystemEvent::PriceChange(e) => Some(&e.market_id),
            SystemEvent::OrderbookUpdate(e) => Some(&e.market_id),
            _ => None,
        };

        // Check if any of our rules involve this market
        let Some(market_id) = market_id else {
            return Ok(all_signals);
        };

        for rule in &self.config.correlation_rules {
            // Only process rule if the changed market is part of it
            let involves_market = rule.legs.iter().any(|leg| &leg.market_id == market_id);
            if !involves_market {
                continue;
            }

            if let Some(signals) = self.process_rule(rule, state).await {
                info!(
                    rule = %rule.name,
                    signal_count = signals.len(),
                    "Generated multi-leg signals"
                );
                all_signals.extend(signals);
            }
        }

        Ok(all_signals)
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
            rules_count = self.config.correlation_rules.len(),
            min_edge = %self.config.min_edge_pct,
            order_size = %self.config.order_size_usd,
            "Initializing multi-leg correlated market strategy"
        );

        for rule in &self.config.correlation_rules {
            info!(
                rule_name = %rule.name,
                relationship = ?rule.relationship,
                leg_count = rule.legs.len(),
                "Registered correlation rule"
            );
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

#[cfg(test)]
mod tests {
    use super::*;
    use polysniper_core::{Market, Orderbook, Position};
    use std::collections::HashMap;
    use tokio::sync::RwLock;

    /// Mock state provider for testing
    struct MockStateProvider {
        markets: HashMap<MarketId, Market>,
        prices: RwLock<HashMap<TokenId, Decimal>>,
    }

    impl MockStateProvider {
        fn new() -> Self {
            Self {
                markets: HashMap::new(),
                prices: RwLock::new(HashMap::new()),
            }
        }

        fn with_market(mut self, market: Market) -> Self {
            self.markets.insert(market.condition_id.clone(), market);
            self
        }

        async fn set_price(&self, token_id: &str, price: Decimal) {
            self.prices
                .write()
                .await
                .insert(token_id.to_string(), price);
        }
    }

    #[async_trait]
    impl StateProvider for MockStateProvider {
        async fn get_market(&self, market_id: &MarketId) -> Option<Market> {
            self.markets.get(market_id).cloned()
        }

        async fn get_all_markets(&self) -> Vec<Market> {
            self.markets.values().cloned().collect()
        }

        async fn get_orderbook(&self, _token_id: &TokenId) -> Option<Orderbook> {
            None
        }

        async fn get_price(&self, token_id: &TokenId) -> Option<Decimal> {
            self.prices.read().await.get(token_id).copied()
        }

        async fn get_position(&self, _market_id: &MarketId) -> Option<Position> {
            None
        }

        async fn get_all_positions(&self) -> Vec<Position> {
            Vec::new()
        }

        async fn get_price_history(
            &self,
            _token_id: &TokenId,
            _limit: usize,
        ) -> Vec<(chrono::DateTime<chrono::Utc>, Decimal)> {
            Vec::new()
        }

        async fn get_portfolio_value(&self) -> Decimal {
            Decimal::ZERO
        }

        async fn get_daily_pnl(&self) -> Decimal {
            Decimal::ZERO
        }
    }

    fn create_test_market(id: &str, yes_token: &str, no_token: &str) -> Market {
        Market {
            condition_id: id.to_string(),
            question: format!("Test market {}", id),
            description: None,
            tags: vec![],
            yes_token_id: yes_token.to_string(),
            no_token_id: no_token.to_string(),
            created_at: Utc::now(),
            end_date: None,
            active: true,
            closed: false,
            volume: dec!(100000),
            liquidity: dec!(50000),
        }
    }

    #[tokio::test]
    async fn test_mutually_exclusive_overpriced() {
        // Setup: Three candidates that should sum to 1.0
        let state = MockStateProvider::new()
            .with_market(create_test_market("biden-2024", "biden-yes", "biden-no"))
            .with_market(create_test_market("trump-2024", "trump-yes", "trump-no"))
            .with_market(create_test_market("other-2024", "other-yes", "other-no"));

        // Prices sum to 1.10 (overpriced by 0.10)
        state.set_price("biden-yes", dec!(0.45)).await;
        state.set_price("trump-yes", dec!(0.50)).await;
        state.set_price("other-yes", dec!(0.15)).await;

        let config = MultiLegConfig {
            enabled: true,
            correlation_rules: vec![CorrelationRule {
                name: "2024 Presidential".to_string(),
                legs: vec![
                    CorrelationLeg {
                        market_id: "biden-2024".to_string(),
                        token: Outcome::Yes,
                        weight: Decimal::ONE,
                    },
                    CorrelationLeg {
                        market_id: "trump-2024".to_string(),
                        token: Outcome::Yes,
                        weight: Decimal::ONE,
                    },
                    CorrelationLeg {
                        market_id: "other-2024".to_string(),
                        token: Outcome::Yes,
                        weight: Decimal::ONE,
                    },
                ],
                relationship: CorrelationRelationship::MutuallyExclusive,
                expected_sum: Some(Decimal::ONE),
                tolerance: dec!(0.01),
            }],
            min_edge_pct: dec!(0.05), // 5% minimum edge
            order_size_usd: dec!(100),
            cooldown_secs: 0,
            min_liquidity_usd: dec!(1000),
        };

        let strategy = MultiLegStrategy::new(config);
        let rule = &strategy.config.correlation_rules[0];

        let signals = strategy.check_mutually_exclusive(rule, &state).await;

        assert!(signals.is_some());
        let signals = signals.unwrap();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].side, Side::Sell);
        // Should sell the highest priced leg (trump at 0.50)
        assert_eq!(signals[0].market_id, "trump-2024");
    }

    #[tokio::test]
    async fn test_mutually_exclusive_underpriced() {
        let state = MockStateProvider::new()
            .with_market(create_test_market("biden-2024", "biden-yes", "biden-no"))
            .with_market(create_test_market("trump-2024", "trump-yes", "trump-no"));

        // Prices sum to 0.85 (underpriced by 0.15)
        state.set_price("biden-yes", dec!(0.40)).await;
        state.set_price("trump-yes", dec!(0.45)).await;

        let config = MultiLegConfig {
            enabled: true,
            correlation_rules: vec![CorrelationRule {
                name: "Binary Election".to_string(),
                legs: vec![
                    CorrelationLeg {
                        market_id: "biden-2024".to_string(),
                        token: Outcome::Yes,
                        weight: Decimal::ONE,
                    },
                    CorrelationLeg {
                        market_id: "trump-2024".to_string(),
                        token: Outcome::Yes,
                        weight: Decimal::ONE,
                    },
                ],
                relationship: CorrelationRelationship::MutuallyExclusive,
                expected_sum: Some(Decimal::ONE),
                tolerance: dec!(0.01),
            }],
            min_edge_pct: dec!(0.05),
            order_size_usd: dec!(100),
            cooldown_secs: 0,
            min_liquidity_usd: dec!(1000),
        };

        let strategy = MultiLegStrategy::new(config);
        let rule = &strategy.config.correlation_rules[0];

        let signals = strategy.check_mutually_exclusive(rule, &state).await;

        assert!(signals.is_some());
        let signals = signals.unwrap();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].side, Side::Buy);
        // Should buy the lowest priced leg (biden at 0.40)
        assert_eq!(signals[0].market_id, "biden-2024");
    }

    #[tokio::test]
    async fn test_inverse_correlation() {
        let state = MockStateProvider::new()
            .with_market(create_test_market("biden-2024", "biden-yes", "biden-no"))
            .with_market(create_test_market("trump-2024", "trump-yes", "trump-no"));

        // For inverse: Biden YES should correlate with Trump NO
        // If Biden YES = 0.60, Trump NO should also be ~0.60
        // So Biden YES + Trump YES should sum to ~1.0 in a two-horse race
        // Here they sum to 1.20 (overpriced)
        state.set_price("biden-yes", dec!(0.60)).await;
        state.set_price("trump-yes", dec!(0.60)).await;

        let config = MultiLegConfig {
            enabled: true,
            correlation_rules: vec![CorrelationRule {
                name: "Inverse Test".to_string(),
                legs: vec![
                    CorrelationLeg {
                        market_id: "biden-2024".to_string(),
                        token: Outcome::Yes,
                        weight: Decimal::ONE,
                    },
                    CorrelationLeg {
                        market_id: "trump-2024".to_string(),
                        token: Outcome::Yes,
                        weight: Decimal::ONE,
                    },
                ],
                relationship: CorrelationRelationship::Inverse,
                expected_sum: Some(Decimal::ONE),
                tolerance: dec!(0.01),
            }],
            min_edge_pct: dec!(0.05),
            order_size_usd: dec!(100),
            cooldown_secs: 0,
            min_liquidity_usd: dec!(1000),
        };

        let strategy = MultiLegStrategy::new(config);
        let rule = &strategy.config.correlation_rules[0];

        let signals = strategy.check_inverse(rule, &state).await;

        assert!(signals.is_some());
        let signals = signals.unwrap();
        // Should generate sell for overpriced and buy for the other
        assert_eq!(signals.len(), 2);
    }

    #[tokio::test]
    async fn test_conditional_violation() {
        // "Biden wins" implies "Democrat wins"
        // So Biden YES price should be <= Democrat YES price
        let state = MockStateProvider::new()
            .with_market(create_test_market("biden-2024", "biden-yes", "biden-no"))
            .with_market(create_test_market("dem-wins", "dem-yes", "dem-no"));

        // Violation: Biden at 0.50, Democrat at 0.40
        // Biden winning implies Democrat wins, so Biden can't be > Democrat
        state.set_price("biden-yes", dec!(0.50)).await;
        state.set_price("dem-yes", dec!(0.40)).await;

        let config = MultiLegConfig {
            enabled: true,
            correlation_rules: vec![CorrelationRule {
                name: "Biden implies Democrat".to_string(),
                legs: vec![
                    CorrelationLeg {
                        market_id: "biden-2024".to_string(),
                        token: Outcome::Yes,
                        weight: Decimal::ONE,
                    },
                    CorrelationLeg {
                        market_id: "dem-wins".to_string(),
                        token: Outcome::Yes,
                        weight: Decimal::ONE,
                    },
                ],
                relationship: CorrelationRelationship::Conditional,
                expected_sum: None,
                tolerance: dec!(0.01),
            }],
            min_edge_pct: dec!(0.05),
            order_size_usd: dec!(100),
            cooldown_secs: 0,
            min_liquidity_usd: dec!(1000),
        };

        let strategy = MultiLegStrategy::new(config);
        let rule = &strategy.config.correlation_rules[0];

        let signals = strategy.check_conditional(rule, &state).await;

        assert!(signals.is_some());
        let signals = signals.unwrap();
        assert_eq!(signals.len(), 2);

        // Should sell Biden (overpriced antecedent) and buy Democrat (underpriced consequent)
        let sell_signal = signals.iter().find(|s| s.side == Side::Sell).unwrap();
        let buy_signal = signals.iter().find(|s| s.side == Side::Buy).unwrap();

        assert_eq!(sell_signal.market_id, "biden-2024");
        assert_eq!(buy_signal.market_id, "dem-wins");
    }

    #[tokio::test]
    async fn test_conditional_no_violation() {
        let state = MockStateProvider::new()
            .with_market(create_test_market("biden-2024", "biden-yes", "biden-no"))
            .with_market(create_test_market("dem-wins", "dem-yes", "dem-no"));

        // No violation: Biden at 0.40, Democrat at 0.50
        // This is correct - Biden implies Democrat, and Biden < Democrat
        state.set_price("biden-yes", dec!(0.40)).await;
        state.set_price("dem-yes", dec!(0.50)).await;

        let config = MultiLegConfig {
            enabled: true,
            correlation_rules: vec![CorrelationRule {
                name: "Biden implies Democrat".to_string(),
                legs: vec![
                    CorrelationLeg {
                        market_id: "biden-2024".to_string(),
                        token: Outcome::Yes,
                        weight: Decimal::ONE,
                    },
                    CorrelationLeg {
                        market_id: "dem-wins".to_string(),
                        token: Outcome::Yes,
                        weight: Decimal::ONE,
                    },
                ],
                relationship: CorrelationRelationship::Conditional,
                expected_sum: None,
                tolerance: dec!(0.01),
            }],
            min_edge_pct: dec!(0.05),
            order_size_usd: dec!(100),
            cooldown_secs: 0,
            min_liquidity_usd: dec!(1000),
        };

        let strategy = MultiLegStrategy::new(config);
        let rule = &strategy.config.correlation_rules[0];

        let signals = strategy.check_conditional(rule, &state).await;

        // No violation, no signals
        assert!(signals.is_none());
    }

    #[tokio::test]
    async fn test_signal_group_id() {
        let config = MultiLegConfig::default();
        let strategy = MultiLegStrategy::new(config);

        let id1 = strategy.next_signal_group_id().await;
        let id2 = strategy.next_signal_group_id().await;

        assert!(id1.starts_with("grp_multileg_"));
        assert!(id2.starts_with("grp_multileg_"));
        assert_ne!(id1, id2);
    }

    #[tokio::test]
    async fn test_cooldown() {
        let config = MultiLegConfig {
            cooldown_secs: 1,
            ..Default::default()
        };
        let strategy = MultiLegStrategy::new(config);

        assert!(!strategy.is_in_cooldown("test_rule").await);

        strategy.set_cooldown("test_rule").await;
        assert!(strategy.is_in_cooldown("test_rule").await);

        // Wait for cooldown to expire
        tokio::time::sleep(tokio::time::Duration::from_millis(1100)).await;
        assert!(!strategy.is_in_cooldown("test_rule").await);
    }

    #[tokio::test]
    async fn test_insufficient_liquidity() {
        let mut market = create_test_market("biden-2024", "biden-yes", "biden-no");
        market.liquidity = dec!(500); // Below minimum

        let state = MockStateProvider::new().with_market(market);
        state.set_price("biden-yes", dec!(0.50)).await;

        let config = MultiLegConfig {
            enabled: true,
            correlation_rules: vec![CorrelationRule {
                name: "Low Liquidity Test".to_string(),
                legs: vec![CorrelationLeg {
                    market_id: "biden-2024".to_string(),
                    token: Outcome::Yes,
                    weight: Decimal::ONE,
                }],
                relationship: CorrelationRelationship::MutuallyExclusive,
                expected_sum: Some(Decimal::ONE),
                tolerance: dec!(0.01),
            }],
            min_edge_pct: dec!(0.05),
            order_size_usd: dec!(100),
            cooldown_secs: 0,
            min_liquidity_usd: dec!(1000), // Higher than market liquidity
        };

        let strategy = MultiLegStrategy::new(config);
        let rule = &strategy.config.correlation_rules[0];

        let signals = strategy.check_mutually_exclusive(rule, &state).await;

        // Should return None due to insufficient liquidity
        assert!(signals.is_none());
    }
}
