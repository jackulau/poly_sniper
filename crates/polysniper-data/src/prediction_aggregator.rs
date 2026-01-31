//! Prediction Aggregator Service
//!
//! Aggregates prices from multiple prediction market platforms
//! and detects arbitrage opportunities.

use crate::external_markets::{
    ExternalMarketError, ExternalMarketPrice, KalshiClient, KalshiConfig, MetaculusClient,
    MetaculusConfig, Platform, PredictItClient, PredictItConfig,
};
use chrono::{DateTime, Utc};
use polysniper_core::events::SystemEvent;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, info, warn};

/// Configuration for the prediction aggregator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionAggregatorConfig {
    /// Whether the aggregator is enabled
    pub enabled: bool,
    /// Minimum edge (%) to consider significant
    #[serde(default = "default_min_edge_pct")]
    pub min_edge_pct: Decimal,
    /// Fee adjustment for cross-platform comparisons
    #[serde(default)]
    pub default_fee_adjustment: Decimal,
    /// Metaculus client config
    #[serde(default)]
    pub metaculus: MetaculusConfig,
    /// PredictIt client config
    #[serde(default)]
    pub predictit: PredictItConfig,
    /// Kalshi client config
    #[serde(default)]
    pub kalshi: KalshiConfig,
    /// Market mappings
    #[serde(default)]
    pub mappings: Vec<MarketMapping>,
}

fn default_min_edge_pct() -> Decimal {
    Decimal::new(5, 0) // 5%
}

impl Default for PredictionAggregatorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_edge_pct: default_min_edge_pct(),
            default_fee_adjustment: Decimal::ZERO,
            metaculus: MetaculusConfig::default(),
            predictit: PredictItConfig::default(),
            kalshi: KalshiConfig::default(),
            mappings: Vec::new(),
        }
    }
}

/// Mapping between Polymarket and external markets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMapping {
    /// Human-readable name
    pub name: String,
    /// Polymarket condition_id
    pub polymarket_id: String,
    /// Metaculus question ID (if available)
    pub metaculus_id: Option<String>,
    /// PredictIt contract ID (if available)
    pub predictit_contract: Option<i64>,
    /// Kalshi market ticker (if available)
    pub kalshi_ticker: Option<String>,
    /// Fee/spread adjustment for this specific mapping
    #[serde(default)]
    pub price_adjustment: Decimal,
}

impl MarketMapping {
    /// Check if this mapping has any external sources
    pub fn has_external_sources(&self) -> bool {
        self.metaculus_id.is_some()
            || self.predictit_contract.is_some()
            || self.kalshi_ticker.is_some()
    }

    /// Get the number of external sources
    pub fn external_source_count(&self) -> usize {
        let mut count = 0;
        if self.metaculus_id.is_some() {
            count += 1;
        }
        if self.predictit_contract.is_some() {
            count += 1;
        }
        if self.kalshi_ticker.is_some() {
            count += 1;
        }
        count
    }
}

/// Aggregated price data from all platforms
#[derive(Debug, Clone)]
pub struct AggregatedPrice {
    /// Mapping name
    pub name: String,
    /// Polymarket condition ID
    pub polymarket_id: String,
    /// Polymarket price (if available)
    pub polymarket_price: Option<Decimal>,
    /// External prices by platform
    pub external_prices: HashMap<Platform, ExternalMarketPrice>,
    /// Weighted consensus price
    pub consensus_price: Option<Decimal>,
    /// Timestamp of aggregation
    pub timestamp: DateTime<Utc>,
}

impl AggregatedPrice {
    /// Calculate the edge between Polymarket and consensus
    pub fn edge(&self) -> Option<Decimal> {
        match (self.polymarket_price, self.consensus_price) {
            (Some(poly), Some(consensus)) => Some(poly - consensus),
            _ => None,
        }
    }

    /// Calculate the edge percentage
    pub fn edge_pct(&self) -> Option<Decimal> {
        match (self.polymarket_price, self.consensus_price) {
            (Some(poly), Some(consensus)) if !consensus.is_zero() => {
                Some(((poly - consensus) / consensus) * Decimal::ONE_HUNDRED)
            }
            _ => None,
        }
    }

    /// Get the external price spread (max - min)
    pub fn external_spread(&self) -> Option<Decimal> {
        if self.external_prices.is_empty() {
            return None;
        }

        let prices: Vec<Decimal> = self
            .external_prices
            .values()
            .map(|p| p.yes_price)
            .collect();

        let min = prices.iter().min()?;
        let max = prices.iter().max()?;

        Some(*max - *min)
    }
}

/// Detected arbitrage opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    /// Mapping name
    pub name: String,
    /// Polymarket condition ID
    pub polymarket_id: String,
    /// Polymarket price
    pub polymarket_price: Decimal,
    /// External platform with best edge
    pub external_platform: Platform,
    /// External price
    pub external_price: Decimal,
    /// Price difference (absolute)
    pub price_difference: Decimal,
    /// Edge percentage
    pub edge_pct: Decimal,
    /// Type of arbitrage opportunity
    pub arbitrage_type: ArbitrageType,
    /// Recommended side (buy/sell on Polymarket)
    pub recommended_side: Side,
    /// Confidence score (0.0-1.0)
    pub confidence: Decimal,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Type of arbitrage opportunity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArbitrageType {
    /// True risk-free arbitrage (rare, requires positions on both platforms)
    HardArbitrage,
    /// Soft arbitrage - edge suggesting mispricing
    SoftArbitrage,
    /// Convergence opportunity - expect prices to converge
    Convergence,
}

/// Trading side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

/// Platform weights for consensus calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformWeights {
    /// Metaculus weight (experts, typically higher trust)
    pub metaculus: Decimal,
    /// PredictIt weight (retail traders, capped contracts)
    pub predictit: Decimal,
    /// Kalshi weight (regulated, good liquidity)
    pub kalshi: Decimal,
}

impl Default for PlatformWeights {
    fn default() -> Self {
        Self {
            metaculus: Decimal::new(12, 1), // 1.2
            predictit: Decimal::new(9, 1),  // 0.9
            kalshi: Decimal::ONE,           // 1.0
        }
    }
}

impl PlatformWeights {
    /// Get weight for a platform
    pub fn get(&self, platform: Platform) -> Decimal {
        match platform {
            Platform::Metaculus => self.metaculus,
            Platform::PredictIt => self.predictit,
            Platform::Kalshi => self.kalshi,
        }
    }
}

/// Prediction aggregator service
pub struct PredictionAggregator {
    config: Arc<RwLock<PredictionAggregatorConfig>>,
    metaculus: Option<MetaculusClient>,
    predictit: Option<PredictItClient>,
    kalshi: Option<KalshiClient>,
    event_tx: broadcast::Sender<SystemEvent>,
    platform_weights: PlatformWeights,
    mappings: Arc<RwLock<Vec<MarketMapping>>>,
}

impl PredictionAggregator {
    /// Create a new prediction aggregator
    pub fn new(
        config: PredictionAggregatorConfig,
        event_tx: broadcast::Sender<SystemEvent>,
    ) -> Self {
        let metaculus = if config.metaculus.enabled {
            Some(MetaculusClient::new(config.metaculus.clone()))
        } else {
            None
        };

        let predictit = if config.predictit.enabled {
            Some(PredictItClient::new(config.predictit.clone()))
        } else {
            None
        };

        let kalshi = if config.kalshi.enabled {
            Some(KalshiClient::new(config.kalshi.clone()))
        } else {
            None
        };

        let mappings = config.mappings.clone();

        Self {
            config: Arc::new(RwLock::new(config)),
            metaculus,
            predictit,
            kalshi,
            event_tx,
            platform_weights: PlatformWeights::default(),
            mappings: Arc::new(RwLock::new(mappings)),
        }
    }

    /// Check if the aggregator is enabled
    pub async fn is_enabled(&self) -> bool {
        let config = self.config.read().await;
        config.enabled
    }

    /// Set platform weights
    pub fn set_platform_weights(&mut self, weights: PlatformWeights) {
        self.platform_weights = weights;
    }

    /// Add a market mapping
    pub async fn add_mapping(&self, mapping: MarketMapping) {
        let mut mappings = self.mappings.write().await;
        mappings.push(mapping);
    }

    /// Get all mappings
    pub async fn get_mappings(&self) -> Vec<MarketMapping> {
        let mappings = self.mappings.read().await;
        mappings.clone()
    }

    /// Poll all external markets and compare with Polymarket prices
    pub async fn poll_and_compare(
        &self,
        polymarket_prices: &HashMap<String, Decimal>,
    ) -> Vec<ArbitrageOpportunity> {
        let mappings = self.mappings.read().await;
        let config = self.config.read().await;
        let mut opportunities = Vec::new();

        for mapping in mappings.iter() {
            // Get external prices for this mapping
            let external_prices = self.get_external_prices(mapping).await;

            if external_prices.is_empty() {
                continue;
            }

            // Get Polymarket price
            let poly_price = match polymarket_prices.get(&mapping.polymarket_id) {
                Some(p) => *p,
                None => continue,
            };

            // Calculate weighted consensus
            let consensus = self.calculate_consensus(&external_prices);

            if let Some(consensus_price) = consensus {
                // Adjust for fees
                let adjusted_consensus =
                    consensus_price - mapping.price_adjustment - config.default_fee_adjustment;

                // Calculate edge
                let edge = poly_price - adjusted_consensus;
                let edge_pct = if !adjusted_consensus.is_zero() {
                    (edge.abs() / adjusted_consensus) * Decimal::ONE_HUNDRED
                } else {
                    Decimal::ZERO
                };

                // Check if edge exceeds threshold
                if edge_pct >= config.min_edge_pct {
                    // Find the platform with the largest discrepancy
                    let (best_platform, best_price, max_diff) =
                        self.find_best_edge(poly_price, &external_prices);

                    let (arbitrage_type, recommended_side) = if edge > Decimal::ZERO {
                        // Polymarket is higher - sell on Polymarket
                        (ArbitrageType::SoftArbitrage, Side::Sell)
                    } else {
                        // Polymarket is lower - buy on Polymarket
                        (ArbitrageType::SoftArbitrage, Side::Buy)
                    };

                    // Calculate confidence based on:
                    // - Number of platforms agreeing
                    // - Spread between external sources
                    let confidence = self.calculate_confidence(&external_prices);

                    let opportunity = ArbitrageOpportunity {
                        name: mapping.name.clone(),
                        polymarket_id: mapping.polymarket_id.clone(),
                        polymarket_price: poly_price,
                        external_platform: best_platform,
                        external_price: best_price,
                        price_difference: max_diff,
                        edge_pct,
                        arbitrage_type,
                        recommended_side,
                        confidence,
                        timestamp: Utc::now(),
                    };

                    info!(
                        name = %mapping.name,
                        poly_price = %poly_price,
                        consensus = %adjusted_consensus,
                        edge_pct = %edge_pct,
                        side = ?recommended_side,
                        "Arbitrage opportunity detected"
                    );

                    opportunities.push(opportunity.clone());

                    // Emit event
                    self.emit_arbitrage_signal(&opportunity).await;
                }
            }
        }

        opportunities
    }

    /// Fetch external prices for a mapping
    async fn get_external_prices(
        &self,
        mapping: &MarketMapping,
    ) -> HashMap<Platform, ExternalMarketPrice> {
        let mut prices = HashMap::new();

        // Fetch from Metaculus
        if let (Some(client), Some(question_id)) = (&self.metaculus, &mapping.metaculus_id) {
            match client.get_question(question_id).await {
                Ok(prediction) => {
                    prices.insert(Platform::Metaculus, prediction.to_external_price());
                }
                Err(e) => {
                    debug!(question_id = %question_id, error = %e, "Failed to fetch Metaculus price");
                }
            }
        }

        // Fetch from PredictIt
        if let (Some(client), Some(contract_id)) = (&self.predictit, &mapping.predictit_contract) {
            // PredictIt doesn't have a direct contract endpoint, so we fetch all markets
            // and filter. In production, you'd want to cache this.
            match client.get_all_markets().await {
                Ok(markets) => {
                    for market in markets {
                        for contract in &market.contracts {
                            if contract.contract_id == *contract_id {
                                prices.insert(
                                    Platform::PredictIt,
                                    contract.to_external_price(&market.name),
                                );
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    debug!(contract_id = %contract_id, error = %e, "Failed to fetch PredictIt price");
                }
            }
        }

        // Fetch from Kalshi
        if let (Some(client), Some(ticker)) = (&self.kalshi, &mapping.kalshi_ticker) {
            match client.get_market(ticker).await {
                Ok(market) => {
                    prices.insert(Platform::Kalshi, market.to_external_price());
                }
                Err(e) => {
                    debug!(ticker = %ticker, error = %e, "Failed to fetch Kalshi price");
                }
            }
        }

        prices
    }

    /// Calculate weighted consensus price from external sources
    fn calculate_consensus(&self, prices: &HashMap<Platform, ExternalMarketPrice>) -> Option<Decimal> {
        if prices.is_empty() {
            return None;
        }

        let mut weighted_sum = Decimal::ZERO;
        let mut total_weight = Decimal::ZERO;

        for (platform, price) in prices {
            let weight = self.platform_weights.get(*platform);
            weighted_sum += price.yes_price * weight;
            total_weight += weight;
        }

        if total_weight.is_zero() {
            return None;
        }

        Some(weighted_sum / total_weight)
    }

    /// Find the platform with the largest edge
    fn find_best_edge(
        &self,
        poly_price: Decimal,
        external_prices: &HashMap<Platform, ExternalMarketPrice>,
    ) -> (Platform, Decimal, Decimal) {
        let mut best_platform = Platform::Metaculus;
        let mut best_price = Decimal::ZERO;
        let mut max_diff = Decimal::ZERO;

        for (platform, price) in external_prices {
            let diff = (poly_price - price.yes_price).abs();
            if diff > max_diff {
                max_diff = diff;
                best_platform = *platform;
                best_price = price.yes_price;
            }
        }

        (best_platform, best_price, max_diff)
    }

    /// Calculate confidence score based on external price agreement
    fn calculate_confidence(&self, prices: &HashMap<Platform, ExternalMarketPrice>) -> Decimal {
        if prices.is_empty() {
            return Decimal::ZERO;
        }

        // Base confidence on number of sources
        let source_confidence = match prices.len() {
            1 => Decimal::new(5, 1),  // 0.5
            2 => Decimal::new(7, 1),  // 0.7
            3 => Decimal::new(9, 1),  // 0.9
            _ => Decimal::ONE,
        };

        // Reduce confidence if external sources disagree
        let price_values: Vec<Decimal> = prices.values().map(|p| p.yes_price).collect();
        if price_values.len() > 1 {
            let min = price_values.iter().min().unwrap();
            let max = price_values.iter().max().unwrap();
            let spread = *max - *min;

            // If spread is large (> 10%), reduce confidence
            if spread > Decimal::new(1, 1) {
                return source_confidence * Decimal::new(7, 1); // Reduce by 30%
            }
        }

        source_confidence
    }

    /// Emit an arbitrage signal event
    async fn emit_arbitrage_signal(&self, opportunity: &ArbitrageOpportunity) {
        // Create the event (this would be a new SystemEvent variant)
        // For now, we use ExternalSignal as a carrier
        let event = SystemEvent::ExternalSignal(polysniper_core::events::ExternalSignalEvent {
            source: polysniper_core::events::SignalSource::Custom {
                name: "prediction_aggregator".to_string(),
            },
            signal_type: "arbitrage_detected".to_string(),
            content: format!(
                "Arbitrage: {} - Poly: {}, {}: {}, Edge: {}%",
                opportunity.name,
                opportunity.polymarket_price,
                opportunity.external_platform,
                opportunity.external_price,
                opportunity.edge_pct
            ),
            market_id: Some(opportunity.polymarket_id.clone()),
            keywords: vec!["arbitrage".to_string(), opportunity.name.clone()],
            metadata: serde_json::to_value(opportunity).unwrap_or_default(),
            received_at: Utc::now(),
        });

        if let Err(e) = self.event_tx.send(event) {
            warn!(error = %e, "Failed to emit arbitrage signal");
        }
    }

    /// Initialize the aggregator (authenticate with platforms if needed)
    pub async fn initialize(&self) -> Result<(), ExternalMarketError> {
        // Authenticate with Kalshi if configured
        if let Some(client) = &self.kalshi {
            if let Err(e) = client.authenticate().await {
                warn!(error = %e, "Failed to authenticate with Kalshi");
                // Don't fail initialization, just log the warning
            }
        }

        info!("Prediction aggregator initialized");
        Ok(())
    }

    /// Get aggregated prices for all mappings
    pub async fn get_aggregated_prices(
        &self,
        polymarket_prices: &HashMap<String, Decimal>,
    ) -> Vec<AggregatedPrice> {
        let mappings = self.mappings.read().await;
        let mut results = Vec::new();

        for mapping in mappings.iter() {
            let external_prices = self.get_external_prices(mapping).await;
            let consensus = self.calculate_consensus(&external_prices);

            results.push(AggregatedPrice {
                name: mapping.name.clone(),
                polymarket_id: mapping.polymarket_id.clone(),
                polymarket_price: polymarket_prices.get(&mapping.polymarket_id).copied(),
                external_prices,
                consensus_price: consensus,
                timestamp: Utc::now(),
            });
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_market_mapping_has_sources() {
        let mapping = MarketMapping {
            name: "Test".to_string(),
            polymarket_id: "0x123".to_string(),
            metaculus_id: Some("12345".to_string()),
            predictit_contract: None,
            kalshi_ticker: Some("TEST".to_string()),
            price_adjustment: Decimal::ZERO,
        };

        assert!(mapping.has_external_sources());
        assert_eq!(mapping.external_source_count(), 2);
    }

    #[test]
    fn test_platform_weights() {
        let weights = PlatformWeights::default();
        assert_eq!(weights.get(Platform::Metaculus), dec!(1.2));
        assert_eq!(weights.get(Platform::PredictIt), dec!(0.9));
        assert_eq!(weights.get(Platform::Kalshi), dec!(1.0));
    }

    #[test]
    fn test_aggregated_price_edge() {
        let mut external_prices = HashMap::new();
        external_prices.insert(
            Platform::Metaculus,
            ExternalMarketPrice::new(
                Platform::Metaculus,
                "123".to_string(),
                "Test".to_string(),
                dec!(0.60),
            ),
        );

        let aggregated = AggregatedPrice {
            name: "Test".to_string(),
            polymarket_id: "0x123".to_string(),
            polymarket_price: Some(dec!(0.65)),
            external_prices,
            consensus_price: Some(dec!(0.60)),
            timestamp: Utc::now(),
        };

        assert_eq!(aggregated.edge(), Some(dec!(0.05)));

        // Edge pct = (0.65 - 0.60) / 0.60 * 100 = 8.33...%
        let edge_pct = aggregated.edge_pct().unwrap();
        assert!(edge_pct > dec!(8) && edge_pct < dec!(9));
    }

    #[test]
    fn test_arbitrage_type_serialization() {
        let arb_type = ArbitrageType::SoftArbitrage;
        let json = serde_json::to_string(&arb_type).unwrap();
        assert_eq!(json, "\"SoftArbitrage\"");
    }
}
