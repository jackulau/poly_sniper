//! Gas price types and configuration for Polygon network monitoring.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Gas price configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasConfig {
    /// Whether gas tracking is enabled
    pub enabled: bool,
    /// Poll interval in seconds
    pub poll_interval_secs: u64,
    /// Primary RPC endpoint for eth_gasPrice
    pub rpc_endpoint: String,
    /// Fallback Polygonscan Gas Oracle API endpoint
    pub fallback_endpoint: Option<String>,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    /// History duration in seconds (default 1 hour)
    pub history_duration_secs: u64,
    /// Gas condition thresholds in gwei
    pub thresholds: GasThresholds,
}

impl Default for GasConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_secs: 10,
            rpc_endpoint: "https://polygon-rpc.com".to_string(),
            fallback_endpoint: None,
            cache_ttl_secs: 10,
            history_duration_secs: 3600, // 1 hour
            thresholds: GasThresholds::default(),
        }
    }
}

/// Thresholds for gas condition classification (in gwei)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasThresholds {
    /// Below this is considered Low
    pub low: Decimal,
    /// Below this is considered Normal (above low)
    pub normal: Decimal,
    /// Below this is considered High (above normal)
    pub high: Decimal,
    // Above high is considered Extreme
}

impl Default for GasThresholds {
    fn default() -> Self {
        Self {
            low: Decimal::new(30, 0),     // 30 gwei
            normal: Decimal::new(100, 0), // 100 gwei
            high: Decimal::new(300, 0),   // 300 gwei
        }
    }
}

/// Gas price tiers (fast, standard, slow)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasPrice {
    /// Fast gas price in gwei (typically ~95th percentile)
    pub fast: Decimal,
    /// Standard gas price in gwei (typically ~50th percentile)
    pub standard: Decimal,
    /// Slow gas price in gwei (typically ~25th percentile)
    pub slow: Decimal,
    /// Timestamp when this price was fetched
    pub timestamp: DateTime<Utc>,
    /// Source of the gas price data
    pub source: GasPriceSource,
}

impl GasPrice {
    /// Create a new gas price with all tiers set to the same value
    pub fn uniform(price_gwei: Decimal, source: GasPriceSource) -> Self {
        Self {
            fast: price_gwei,
            standard: price_gwei,
            slow: price_gwei,
            timestamp: Utc::now(),
            source,
        }
    }

    /// Create a new gas price with tiered values
    pub fn tiered(fast: Decimal, standard: Decimal, slow: Decimal, source: GasPriceSource) -> Self {
        Self {
            fast,
            standard,
            slow,
            timestamp: Utc::now(),
            source,
        }
    }
}

/// Source of gas price data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GasPriceSource {
    /// Primary RPC endpoint (eth_gasPrice)
    Rpc,
    /// Polygonscan Gas Oracle API
    Polygonscan,
    /// Cached value
    Cache,
}

impl std::fmt::Display for GasPriceSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GasPriceSource::Rpc => write!(f, "RPC"),
            GasPriceSource::Polygonscan => write!(f, "Polygonscan"),
            GasPriceSource::Cache => write!(f, "Cache"),
        }
    }
}

/// Estimated gas cost for an operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasCostEstimate {
    /// Operation type
    pub operation: GasOperation,
    /// Estimated gas units
    pub gas_units: u64,
    /// Cost in MATIC using fast gas price
    pub cost_fast_matic: Decimal,
    /// Cost in MATIC using standard gas price
    pub cost_standard_matic: Decimal,
    /// Cost in MATIC using slow gas price
    pub cost_slow_matic: Decimal,
    /// Cost in USD using standard gas price (if MATIC price available)
    pub cost_usd: Option<Decimal>,
    /// Timestamp of estimate
    pub timestamp: DateTime<Utc>,
}

/// Types of operations for gas estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GasOperation {
    /// Approve token spending
    Approve,
    /// Submit a limit order
    LimitOrder,
    /// Cancel an order
    CancelOrder,
    /// Market order execution
    MarketOrder,
}

impl GasOperation {
    /// Get estimated gas units for this operation type on Polygon
    pub fn estimated_gas_units(&self) -> u64 {
        match self {
            GasOperation::Approve => 46_000,
            GasOperation::LimitOrder => 150_000,
            GasOperation::CancelOrder => 80_000,
            GasOperation::MarketOrder => 200_000,
        }
    }
}

/// Gas condition classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GasCondition {
    /// Gas prices are low - good time to transact
    Low,
    /// Gas prices are normal
    Normal,
    /// Gas prices are high - consider waiting
    High,
    /// Gas prices are extreme - avoid transactions if possible
    Extreme,
}

impl GasCondition {
    /// Classify gas condition based on standard gas price and thresholds
    pub fn classify(gas_price_gwei: Decimal, thresholds: &GasThresholds) -> Self {
        if gas_price_gwei < thresholds.low {
            GasCondition::Low
        } else if gas_price_gwei < thresholds.normal {
            GasCondition::Normal
        } else if gas_price_gwei < thresholds.high {
            GasCondition::High
        } else {
            GasCondition::Extreme
        }
    }

    /// Check if it's a good time to transact
    pub fn is_favorable(&self) -> bool {
        matches!(self, GasCondition::Low | GasCondition::Normal)
    }
}

impl std::fmt::Display for GasCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GasCondition::Low => write!(f, "Low"),
            GasCondition::Normal => write!(f, "Normal"),
            GasCondition::High => write!(f, "High"),
            GasCondition::Extreme => write!(f, "Extreme"),
        }
    }
}

/// Gas price history for tracking and analysis
#[derive(Debug, Clone)]
pub struct GasPriceHistory {
    /// Historical gas prices
    prices: VecDeque<GasPrice>,
    /// Maximum history duration in seconds
    max_duration_secs: u64,
}

impl GasPriceHistory {
    /// Create a new gas price history with default duration (1 hour)
    pub fn new() -> Self {
        Self::with_duration(3600)
    }

    /// Create a new gas price history with custom duration
    pub fn with_duration(max_duration_secs: u64) -> Self {
        Self {
            prices: VecDeque::new(),
            max_duration_secs,
        }
    }

    /// Add a new gas price to history
    pub fn add(&mut self, price: GasPrice) {
        self.prices.push_back(price);
        self.prune_old();
    }

    /// Remove prices older than max_duration_secs
    fn prune_old(&mut self) {
        let cutoff = Utc::now() - chrono::Duration::seconds(self.max_duration_secs as i64);
        while let Some(front) = self.prices.front() {
            if front.timestamp < cutoff {
                self.prices.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get the number of prices in history
    pub fn len(&self) -> usize {
        self.prices.len()
    }

    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }

    /// Get the latest gas price
    pub fn latest(&self) -> Option<&GasPrice> {
        self.prices.back()
    }

    /// Calculate rolling average of standard gas prices
    pub fn average_standard(&self) -> Option<Decimal> {
        if self.prices.is_empty() {
            return None;
        }
        let sum: Decimal = self.prices.iter().map(|p| p.standard).sum();
        Some(sum / Decimal::from(self.prices.len()))
    }

    /// Calculate rolling average of fast gas prices
    pub fn average_fast(&self) -> Option<Decimal> {
        if self.prices.is_empty() {
            return None;
        }
        let sum: Decimal = self.prices.iter().map(|p| p.fast).sum();
        Some(sum / Decimal::from(self.prices.len()))
    }

    /// Calculate percentile of standard gas prices (0-100)
    pub fn percentile_standard(&self, percentile: u8) -> Option<Decimal> {
        if self.prices.is_empty() || percentile > 100 {
            return None;
        }
        let mut prices: Vec<Decimal> = self.prices.iter().map(|p| p.standard).collect();
        prices.sort();
        let index = (prices.len() - 1) * percentile as usize / 100;
        Some(prices[index])
    }

    /// Detect if current price is a spike (above 95th percentile of history)
    pub fn is_spike(&self, current: &GasPrice) -> bool {
        if let Some(p95) = self.percentile_standard(95) {
            current.standard > p95
        } else {
            false
        }
    }

    /// Get min and max standard gas prices in history
    pub fn min_max_standard(&self) -> Option<(Decimal, Decimal)> {
        if self.prices.is_empty() {
            return None;
        }
        let min = self.prices.iter().map(|p| p.standard).min()?;
        let max = self.prices.iter().map(|p| p.standard).max()?;
        Some((min, max))
    }
}

impl Default for GasPriceHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_gas_condition_classify() {
        let thresholds = GasThresholds::default();

        assert_eq!(
            GasCondition::classify(dec!(20), &thresholds),
            GasCondition::Low
        );
        assert_eq!(
            GasCondition::classify(dec!(50), &thresholds),
            GasCondition::Normal
        );
        assert_eq!(
            GasCondition::classify(dec!(150), &thresholds),
            GasCondition::High
        );
        assert_eq!(
            GasCondition::classify(dec!(500), &thresholds),
            GasCondition::Extreme
        );
    }

    #[test]
    fn test_gas_condition_is_favorable() {
        assert!(GasCondition::Low.is_favorable());
        assert!(GasCondition::Normal.is_favorable());
        assert!(!GasCondition::High.is_favorable());
        assert!(!GasCondition::Extreme.is_favorable());
    }

    #[test]
    fn test_gas_price_uniform() {
        let price = GasPrice::uniform(dec!(50), GasPriceSource::Rpc);
        assert_eq!(price.fast, dec!(50));
        assert_eq!(price.standard, dec!(50));
        assert_eq!(price.slow, dec!(50));
    }

    #[test]
    fn test_gas_price_tiered() {
        let price = GasPrice::tiered(dec!(100), dec!(50), dec!(30), GasPriceSource::Polygonscan);
        assert_eq!(price.fast, dec!(100));
        assert_eq!(price.standard, dec!(50));
        assert_eq!(price.slow, dec!(30));
    }

    #[test]
    fn test_gas_operation_estimated_units() {
        assert_eq!(GasOperation::Approve.estimated_gas_units(), 46_000);
        assert_eq!(GasOperation::LimitOrder.estimated_gas_units(), 150_000);
        assert_eq!(GasOperation::CancelOrder.estimated_gas_units(), 80_000);
        assert_eq!(GasOperation::MarketOrder.estimated_gas_units(), 200_000);
    }

    #[test]
    fn test_gas_price_history_average() {
        let mut history = GasPriceHistory::new();

        history.add(GasPrice::uniform(dec!(50), GasPriceSource::Rpc));
        history.add(GasPrice::uniform(dec!(100), GasPriceSource::Rpc));
        history.add(GasPrice::uniform(dec!(150), GasPriceSource::Rpc));

        assert_eq!(history.average_standard(), Some(dec!(100)));
    }

    #[test]
    fn test_gas_price_history_percentile() {
        let mut history = GasPriceHistory::new();

        for i in 1..=100 {
            history.add(GasPrice::uniform(Decimal::from(i), GasPriceSource::Rpc));
        }

        // 50th percentile should be around 50
        let p50 = history.percentile_standard(50).unwrap();
        assert!(p50 >= dec!(49) && p50 <= dec!(51));

        // 95th percentile should be around 95
        let p95 = history.percentile_standard(95).unwrap();
        assert!(p95 >= dec!(94) && p95 <= dec!(96));
    }

    #[test]
    fn test_gas_price_history_spike_detection() {
        let mut history = GasPriceHistory::new();

        // Add normal prices
        for _ in 0..20 {
            history.add(GasPrice::uniform(dec!(50), GasPriceSource::Rpc));
        }

        let normal_price = GasPrice::uniform(dec!(50), GasPriceSource::Rpc);
        assert!(!history.is_spike(&normal_price));

        let spike_price = GasPrice::uniform(dec!(200), GasPriceSource::Rpc);
        assert!(history.is_spike(&spike_price));
    }

    #[test]
    fn test_gas_price_history_min_max() {
        let mut history = GasPriceHistory::new();

        history.add(GasPrice::uniform(dec!(50), GasPriceSource::Rpc));
        history.add(GasPrice::uniform(dec!(100), GasPriceSource::Rpc));
        history.add(GasPrice::uniform(dec!(25), GasPriceSource::Rpc));

        let (min, max) = history.min_max_standard().unwrap();
        assert_eq!(min, dec!(25));
        assert_eq!(max, dec!(100));
    }
}
