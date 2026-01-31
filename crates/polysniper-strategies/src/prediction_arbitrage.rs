//! Prediction Aggregator Arbitrage Strategy
//!
//! Compares Polymarket odds with external prediction markets (Metaculus, PredictIt, Kalshi)
//! to detect pricing discrepancies and edge opportunities.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use polysniper_core::{
    events::{ArbitrageType, ExternalPlatform},
    OrderType, Outcome, Priority, Side, StateProvider, Strategy, StrategyError, SystemEvent,
    TradeSignal,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Prediction arbitrage strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionArbitrageConfig {
    /// Whether the strategy is enabled
    pub enabled: bool,
    /// Minimum edge percentage to trigger (e.g., 5.0 = 5%)
    #[serde(default = "default_min_edge_pct")]
    pub min_edge_pct: Decimal,
    /// Platform weights for consensus calculation
    #[serde(default)]
    pub platform_weights: PlatformWeights,
    /// Soft arbitrage configuration
    #[serde(default)]
    pub soft_arb: SoftArbConfig,
    /// Convergence trading configuration
    #[serde(default)]
    pub convergence: ConvergenceConfig,
    /// Cooldown period in seconds after generating a signal
    #[serde(default = "default_cooldown")]
    pub cooldown_secs: u64,
}

fn default_min_edge_pct() -> Decimal {
    dec!(5.0)
}

fn default_cooldown() -> u64 {
    1800 // 30 minutes
}

impl Default for PredictionArbitrageConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_edge_pct: default_min_edge_pct(),
            platform_weights: PlatformWeights::default(),
            soft_arb: SoftArbConfig::default(),
            convergence: ConvergenceConfig::default(),
            cooldown_secs: default_cooldown(),
        }
    }
}

/// Platform weights for consensus calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformWeights {
    /// Metaculus weight (experts, typically higher trust)
    #[serde(default = "default_metaculus_weight")]
    pub metaculus: Decimal,
    /// PredictIt weight (retail traders, capped contracts)
    #[serde(default = "default_predictit_weight")]
    pub predictit: Decimal,
    /// Kalshi weight (regulated, good liquidity)
    #[serde(default = "default_kalshi_weight")]
    pub kalshi: Decimal,
}

fn default_metaculus_weight() -> Decimal {
    dec!(1.2)
}

fn default_predictit_weight() -> Decimal {
    dec!(0.9)
}

fn default_kalshi_weight() -> Decimal {
    dec!(1.0)
}

impl Default for PlatformWeights {
    fn default() -> Self {
        Self {
            metaculus: default_metaculus_weight(),
            predictit: default_predictit_weight(),
            kalshi: default_kalshi_weight(),
        }
    }
}

impl PlatformWeights {
    /// Get weight for a platform
    pub fn get(&self, platform: ExternalPlatform) -> Decimal {
        match platform {
            ExternalPlatform::Metaculus => self.metaculus,
            ExternalPlatform::PredictIt => self.predictit,
            ExternalPlatform::Kalshi => self.kalshi,
        }
    }
}

/// Soft arbitrage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftArbConfig {
    /// Whether soft arbitrage is enabled
    pub enabled: bool,
    /// Minimum edge percentage for soft arb (e.g., 5.0 = 5%)
    #[serde(default = "default_soft_arb_edge")]
    pub min_edge_pct: Decimal,
    /// Maximum entry price (don't buy above this)
    #[serde(default = "default_max_entry_price")]
    pub max_entry_price: Decimal,
    /// Order size in USD
    #[serde(default = "default_order_size")]
    pub order_size_usd: Decimal,
}

fn default_soft_arb_edge() -> Decimal {
    dec!(5.0)
}

fn default_max_entry_price() -> Decimal {
    dec!(0.90)
}

fn default_order_size() -> Decimal {
    dec!(50.0)
}

impl Default for SoftArbConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_edge_pct: default_soft_arb_edge(),
            max_entry_price: default_max_entry_price(),
            order_size_usd: default_order_size(),
        }
    }
}

/// Convergence trading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceConfig {
    /// Whether convergence trading is enabled
    pub enabled: bool,
    /// Minimum spread z-score to trigger (e.g., 2.0 = 2 standard deviations)
    #[serde(default = "default_zscore")]
    pub min_spread_zscore: Decimal,
    /// Number of historical periods to compare
    #[serde(default = "default_lookback")]
    pub lookback_periods: u32,
    /// Order size in USD
    #[serde(default = "default_convergence_size")]
    pub order_size_usd: Decimal,
}

fn default_zscore() -> Decimal {
    dec!(2.0)
}

fn default_lookback() -> u32 {
    50
}

fn default_convergence_size() -> Decimal {
    dec!(25.0)
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_spread_zscore: default_zscore(),
            lookback_periods: default_lookback(),
            order_size_usd: default_convergence_size(),
        }
    }
}

/// External price data for a market
#[derive(Debug, Clone)]
struct ExternalPriceData {
    #[allow(dead_code)]
    platform: ExternalPlatform,
    price: Decimal,
    #[allow(dead_code)]
    timestamp: DateTime<Utc>,
}

/// Historical spread data for convergence trading
#[derive(Debug, Clone, Default)]
struct SpreadHistory {
    spreads: VecDeque<Decimal>,
    max_size: usize,
}

impl SpreadHistory {
    fn new(max_size: usize) -> Self {
        Self {
            spreads: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    fn add(&mut self, spread: Decimal) {
        if self.spreads.len() >= self.max_size {
            self.spreads.pop_front();
        }
        self.spreads.push_back(spread);
    }

    fn mean(&self) -> Option<Decimal> {
        if self.spreads.is_empty() {
            return None;
        }
        let sum: Decimal = self.spreads.iter().sum();
        Some(sum / Decimal::from(self.spreads.len()))
    }

    fn std_dev(&self) -> Option<Decimal> {
        let mean = self.mean()?;
        if self.spreads.len() < 2 {
            return None;
        }

        let variance: Decimal = self
            .spreads
            .iter()
            .map(|x| (*x - mean) * (*x - mean))
            .sum::<Decimal>()
            / Decimal::from(self.spreads.len() - 1);

        // Approximate square root using Newton's method
        let mut guess = variance / Decimal::TWO;
        for _ in 0..10 {
            if guess.is_zero() {
                break;
            }
            guess = (guess + variance / guess) / Decimal::TWO;
        }
        Some(guess)
    }

    fn zscore(&self, value: Decimal) -> Option<Decimal> {
        let mean = self.mean()?;
        let std_dev = self.std_dev()?;
        if std_dev.is_zero() {
            return None;
        }
        Some((value - mean) / std_dev)
    }

    fn len(&self) -> usize {
        self.spreads.len()
    }
}

/// Prediction arbitrage strategy
pub struct PredictionArbitrageStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: Arc<RwLock<PredictionArbitrageConfig>>,
    /// External prices by market ID and platform
    external_prices: Arc<RwLock<HashMap<String, HashMap<ExternalPlatform, ExternalPriceData>>>>,
    /// Signal cooldowns by market ID
    signal_cooldowns: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    /// Historical spreads by market ID
    historical_spreads: Arc<RwLock<HashMap<String, SpreadHistory>>>,
}

impl PredictionArbitrageStrategy {
    /// Create a new prediction arbitrage strategy
    pub fn new(config: PredictionArbitrageConfig) -> Self {
        let enabled = config.enabled;
        Self {
            id: "prediction_arbitrage".to_string(),
            name: "Prediction Aggregator Arbitrage Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config: Arc::new(RwLock::new(config)),
            external_prices: Arc::new(RwLock::new(HashMap::new())),
            signal_cooldowns: Arc::new(RwLock::new(HashMap::new())),
            historical_spreads: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Process an external price update
    async fn process_external_price(
        &self,
        market_id: &str,
        platform: ExternalPlatform,
        price: Decimal,
    ) {
        let mut prices = self.external_prices.write().await;
        let market_prices = prices.entry(market_id.to_string()).or_default();
        market_prices.insert(
            platform,
            ExternalPriceData {
                platform,
                price,
                timestamp: Utc::now(),
            },
        );
    }

    /// Calculate weighted consensus price from external sources
    async fn calculate_consensus(&self, market_id: &str) -> Option<Decimal> {
        let config = self.config.read().await;
        let prices = self.external_prices.read().await;

        let market_prices = prices.get(market_id)?;
        if market_prices.is_empty() {
            return None;
        }

        let mut weighted_sum = Decimal::ZERO;
        let mut total_weight = Decimal::ZERO;

        for (platform, data) in market_prices {
            let weight = config.platform_weights.get(*platform);
            weighted_sum += data.price * weight;
            total_weight += weight;
        }

        if total_weight.is_zero() {
            return None;
        }

        Some(weighted_sum / total_weight)
    }

    /// Check if a market is in cooldown
    async fn is_in_cooldown(&self, market_id: &str) -> bool {
        let config = self.config.read().await;
        let cooldowns = self.signal_cooldowns.read().await;

        if let Some(last_signal) = cooldowns.get(market_id) {
            let elapsed = Utc::now() - *last_signal;
            return elapsed.num_seconds() < config.cooldown_secs as i64;
        }

        false
    }

    /// Set cooldown for a market
    async fn set_cooldown(&self, market_id: &str) {
        let mut cooldowns = self.signal_cooldowns.write().await;
        cooldowns.insert(market_id.to_string(), Utc::now());
    }

    /// Update historical spread and return z-score
    async fn update_spread_history(&self, market_id: &str, spread: Decimal) -> Option<Decimal> {
        let config = self.config.read().await;
        let mut histories = self.historical_spreads.write().await;

        let history = histories
            .entry(market_id.to_string())
            .or_insert_with(|| SpreadHistory::new(config.convergence.lookback_periods as usize));

        let zscore = if history.len() >= 10 {
            history.zscore(spread)
        } else {
            None
        };

        history.add(spread);
        zscore
    }

    /// Generate a trade signal for soft arbitrage
    fn generate_soft_arb_signal(
        &self,
        config: &PredictionArbitrageConfig,
        market_id: &str,
        token_id: &str,
        poly_price: Decimal,
        consensus: Decimal,
        edge_pct: Decimal,
        best_platform: ExternalPlatform,
    ) -> Option<TradeSignal> {
        let side = if poly_price < consensus {
            // Polymarket is cheap - BUY
            if poly_price > config.soft_arb.max_entry_price {
                debug!(
                    market_id = %market_id,
                    poly_price = %poly_price,
                    "Price above max entry, skipping"
                );
                return None;
            }
            Side::Buy
        } else {
            // Polymarket is expensive - SELL
            Side::Sell
        };

        let size = config.soft_arb.order_size_usd / poly_price;
        let now = Utc::now();
        let timestamp_ms = now.timestamp_millis();

        Some(TradeSignal {
            id: format!(
                "sig_pred_arb_{}_{}_{}",
                market_id,
                timestamp_ms,
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: market_id.to_string(),
            token_id: token_id.to_string(),
            outcome: if side == Side::Buy {
                Outcome::Yes
            } else {
                Outcome::No
            },
            side,
            price: Some(poly_price),
            size,
            size_usd: config.soft_arb.order_size_usd,
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: now,
            reason: format!(
                "Soft arb: Poly={:.2}, Consensus={:.2}, Edge={:.1}% vs {}",
                poly_price, consensus, edge_pct, best_platform
            ),
            metadata: serde_json::json!({
                "arbitrage_type": "soft",
                "polymarket_price": poly_price.to_string(),
                "consensus_price": consensus.to_string(),
                "edge_pct": edge_pct.to_string(),
                "best_platform": best_platform.as_str(),
            }),
        })
    }

    /// Generate a trade signal for convergence trading
    fn generate_convergence_signal(
        &self,
        config: &PredictionArbitrageConfig,
        market_id: &str,
        token_id: &str,
        poly_price: Decimal,
        consensus: Decimal,
        zscore: Decimal,
    ) -> Option<TradeSignal> {
        // If z-score is positive (spread is above mean), expect convergence (spread to decrease)
        // If Polymarket > consensus (positive spread), expect Polymarket to decrease -> SELL
        // If Polymarket < consensus (negative spread), expect Polymarket to increase -> BUY

        let spread = poly_price - consensus;
        let side = if spread > Decimal::ZERO {
            Side::Sell
        } else {
            Side::Buy
        };

        let size = config.convergence.order_size_usd / poly_price;
        let now = Utc::now();
        let timestamp_ms = now.timestamp_millis();

        Some(TradeSignal {
            id: format!(
                "sig_pred_conv_{}_{}_{}",
                market_id,
                timestamp_ms,
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: market_id.to_string(),
            token_id: token_id.to_string(),
            outcome: if side == Side::Buy {
                Outcome::Yes
            } else {
                Outcome::No
            },
            side,
            price: Some(poly_price),
            size,
            size_usd: config.convergence.order_size_usd,
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: now,
            reason: format!(
                "Convergence: Poly={:.2}, Consensus={:.2}, Z-score={:.2}",
                poly_price, consensus, zscore
            ),
            metadata: serde_json::json!({
                "arbitrage_type": "convergence",
                "polymarket_price": poly_price.to_string(),
                "consensus_price": consensus.to_string(),
                "zscore": zscore.to_string(),
            }),
        })
    }

    /// Find the platform with the largest price discrepancy
    async fn find_best_edge(
        &self,
        market_id: &str,
        poly_price: Decimal,
    ) -> Option<(ExternalPlatform, Decimal)> {
        let prices = self.external_prices.read().await;
        let market_prices = prices.get(market_id)?;

        let mut best_platform = ExternalPlatform::Metaculus;
        let mut max_diff = Decimal::ZERO;

        for (platform, data) in market_prices {
            let diff = (poly_price - data.price).abs();
            if diff > max_diff {
                max_diff = diff;
                best_platform = *platform;
            }
        }

        if max_diff > Decimal::ZERO {
            Some((best_platform, market_prices.get(&best_platform)?.price))
        } else {
            None
        }
    }
}

#[async_trait]
impl Strategy for PredictionArbitrageStrategy {
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

        let config = self.config.read().await;
        if !config.enabled {
            return Ok(signals);
        }

        match event {
            SystemEvent::ExternalPriceUpdate(e) => {
                // Store the external price
                if let Some(ref mapping) = e.polymarket_mapping {
                    self.process_external_price(mapping, e.platform, e.yes_price)
                        .await;
                }
            }

            SystemEvent::PredictionArbitrageDetected(e) => {
                // Check cooldown
                if self.is_in_cooldown(&e.polymarket_id).await {
                    debug!(
                        market_id = %e.polymarket_id,
                        "Market in cooldown, skipping signal"
                    );
                    return Ok(signals);
                }

                // Check minimum edge
                if e.edge_pct < config.min_edge_pct {
                    return Ok(signals);
                }

                // Generate signal based on arbitrage type
                let signal = match e.arbitrage_type {
                    ArbitrageType::SoftArbitrage if config.soft_arb.enabled => {
                        if e.edge_pct >= config.soft_arb.min_edge_pct {
                            self.generate_soft_arb_signal(
                                &config,
                                &e.polymarket_id,
                                &e.polymarket_id, // Use condition_id as token_id placeholder
                                e.polymarket_price,
                                e.external_price,
                                e.edge_pct,
                                e.platform,
                            )
                        } else {
                            None
                        }
                    }
                    ArbitrageType::Convergence if config.convergence.enabled => {
                        // For convergence, we need z-score from history
                        let spread = e.polymarket_price - e.external_price;
                        if let Some(zscore) =
                            self.update_spread_history(&e.polymarket_id, spread).await
                        {
                            if zscore.abs() >= config.convergence.min_spread_zscore {
                                self.generate_convergence_signal(
                                    &config,
                                    &e.polymarket_id,
                                    &e.polymarket_id,
                                    e.polymarket_price,
                                    e.external_price,
                                    zscore,
                                )
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if let Some(s) = signal {
                    self.set_cooldown(&e.polymarket_id).await;
                    info!(
                        market_id = %e.polymarket_id,
                        edge_pct = %e.edge_pct,
                        arbitrage_type = ?e.arbitrage_type,
                        "Generated prediction arbitrage signal"
                    );
                    signals.push(s);
                }
            }

            // For price updates, check for arbitrage opportunities
            SystemEvent::PriceChange(e) => {
                let consensus = self.calculate_consensus(&e.market_id).await;
                if let Some(consensus_price) = consensus {
                    let spread = e.new_price - consensus_price;
                    let edge_pct = if !consensus_price.is_zero() {
                        (spread.abs() / consensus_price) * Decimal::ONE_HUNDRED
                    } else {
                        Decimal::ZERO
                    };

                    // Update spread history for convergence
                    let zscore = self.update_spread_history(&e.market_id, spread).await;

                    // Check for soft arbitrage
                    if config.soft_arb.enabled && edge_pct >= config.soft_arb.min_edge_pct {
                        if !self.is_in_cooldown(&e.market_id).await {
                            if let Some((best_platform, _)) =
                                self.find_best_edge(&e.market_id, e.new_price).await
                            {
                                let signal = self.generate_soft_arb_signal(
                                    &config,
                                    &e.market_id,
                                    &e.token_id,
                                    e.new_price,
                                    consensus_price,
                                    edge_pct,
                                    best_platform,
                                );

                                if let Some(s) = signal {
                                    self.set_cooldown(&e.market_id).await;
                                    signals.push(s);
                                    return Ok(signals);
                                }
                            }
                        }
                    }

                    // Check for convergence opportunity
                    if config.convergence.enabled {
                        if let Some(z) = zscore {
                            if z.abs() >= config.convergence.min_spread_zscore {
                                if !self.is_in_cooldown(&e.market_id).await {
                                    let signal = self.generate_convergence_signal(
                                        &config,
                                        &e.market_id,
                                        &e.token_id,
                                        e.new_price,
                                        consensus_price,
                                        z,
                                    );

                                    if let Some(s) = signal {
                                        self.set_cooldown(&e.market_id).await;
                                        signals.push(s);
                                        return Ok(signals);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            _ => {}
        }

        Ok(signals)
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(
            event,
            SystemEvent::PriceChange(_)
                | SystemEvent::ExternalPriceUpdate(_)
                | SystemEvent::PredictionArbitrageDetected(_)
        )
    }

    async fn initialize(&mut self, _state: &dyn StateProvider) -> Result<(), StrategyError> {
        let config = self.config.read().await;
        info!(
            strategy_id = %self.id,
            min_edge_pct = %config.min_edge_pct,
            soft_arb_enabled = config.soft_arb.enabled,
            convergence_enabled = config.convergence.enabled,
            "Initializing prediction arbitrage strategy"
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
        let new_config: PredictionArbitrageConfig = toml::from_str(config_content)
            .map_err(|e| StrategyError::ConfigError(format!("Failed to parse config: {}", e)))?;

        self.enabled.store(new_config.enabled, Ordering::SeqCst);

        {
            let mut config = self.config.write().await;
            *config = new_config;
        }

        info!(strategy_id = %self.id, "Reloaded prediction arbitrage strategy config");
        Ok(())
    }

    fn config_name(&self) -> &str {
        "prediction_arbitrage"
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

    #[test]
    fn test_default_config() {
        let config = PredictionArbitrageConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.min_edge_pct, dec!(5.0));
        assert_eq!(config.cooldown_secs, 1800);
        assert!(config.soft_arb.enabled);
        assert!(config.convergence.enabled);
    }

    #[test]
    fn test_platform_weights() {
        let weights = PlatformWeights::default();
        assert_eq!(weights.get(ExternalPlatform::Metaculus), dec!(1.2));
        assert_eq!(weights.get(ExternalPlatform::PredictIt), dec!(0.9));
        assert_eq!(weights.get(ExternalPlatform::Kalshi), dec!(1.0));
    }

    #[test]
    fn test_spread_history() {
        let mut history = SpreadHistory::new(10);

        // Add some spreads
        for i in 1..=5 {
            history.add(Decimal::from(i));
        }

        assert_eq!(history.len(), 5);
        assert_eq!(history.mean(), Some(dec!(3))); // (1+2+3+4+5)/5 = 3

        // Add more to trigger rotation
        for i in 6..=15 {
            history.add(Decimal::from(i));
        }

        assert_eq!(history.len(), 10);
    }

    #[test]
    fn test_spread_history_zscore() {
        let mut history = SpreadHistory::new(100);

        // Add a set of values
        for i in 0..50 {
            history.add(Decimal::from(i) / Decimal::from(100)); // 0.00 to 0.49
        }

        let mean = history.mean().unwrap();
        let zscore_mean = history.zscore(mean);
        if let Some(z) = zscore_mean {
            // Z-score of mean should be close to 0
            assert!(z.abs() < dec!(0.1));
        }
    }

    #[test]
    fn test_strategy_creation() {
        let config = PredictionArbitrageConfig::default();
        let strategy = PredictionArbitrageStrategy::new(config);

        assert_eq!(strategy.id(), "prediction_arbitrage");
        assert_eq!(strategy.name(), "Prediction Aggregator Arbitrage Strategy");
        assert!(!strategy.is_enabled());
    }

    #[tokio::test]
    async fn test_cooldown() {
        let mut config = PredictionArbitrageConfig::default();
        config.cooldown_secs = 1; // 1 second for testing
        let strategy = PredictionArbitrageStrategy::new(config);

        assert!(!strategy.is_in_cooldown("test_market").await);

        strategy.set_cooldown("test_market").await;
        assert!(strategy.is_in_cooldown("test_market").await);

        // Wait for cooldown to expire
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        assert!(!strategy.is_in_cooldown("test_market").await);
    }

    #[tokio::test]
    async fn test_external_price_storage() {
        let config = PredictionArbitrageConfig::default();
        let strategy = PredictionArbitrageStrategy::new(config);

        strategy
            .process_external_price("test_market", ExternalPlatform::Metaculus, dec!(0.65))
            .await;

        strategy
            .process_external_price("test_market", ExternalPlatform::Kalshi, dec!(0.70))
            .await;

        let consensus = strategy.calculate_consensus("test_market").await;
        assert!(consensus.is_some());

        // Consensus should be weighted average: (0.65 * 1.2 + 0.70 * 1.0) / (1.2 + 1.0)
        // = (0.78 + 0.70) / 2.2 = 1.48 / 2.2 = 0.6727...
        let c = consensus.unwrap();
        assert!(c > dec!(0.67) && c < dec!(0.68));
    }

    #[test]
    fn test_accepts_event() {
        let strategy = PredictionArbitrageStrategy::new(PredictionArbitrageConfig::default());

        let price_event = SystemEvent::PriceChange(polysniper_core::events::PriceChangeEvent {
            market_id: "test".to_string(),
            token_id: "token".to_string(),
            old_price: Some(dec!(0.5)),
            new_price: dec!(0.6),
            price_change_pct: Some(dec!(20.0)),
            timestamp: Utc::now(),
        });
        assert!(strategy.accepts_event(&price_event));

        let external_price_event =
            SystemEvent::ExternalPriceUpdate(polysniper_core::events::ExternalPriceUpdateEvent {
                platform: ExternalPlatform::Metaculus,
                question_id: "12345".to_string(),
                yes_price: dec!(0.65),
                polymarket_mapping: Some("test".to_string()),
                timestamp: Utc::now(),
            });
        assert!(strategy.accepts_event(&external_price_event));
    }
}
