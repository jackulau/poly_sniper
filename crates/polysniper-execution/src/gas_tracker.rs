//! Gas price tracker for Polygon network
//!
//! Monitors gas prices from Polygon RPC and Polygonscan APIs,
//! tracks history, calculates averages, and emits events.

use anyhow::{Context, Result};
use chrono::Utc;
use polysniper_core::{
    GasCondition, GasConfig, GasCostEstimate, GasOperation, GasPrice, GasPriceHistory,
    GasPriceSource, GasPriceUpdateEvent, SystemEvent,
};
use rust_decimal::Decimal;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info, warn};

/// Polygonscan Gas Oracle API response
#[derive(Debug, Deserialize)]
struct PolygonscanGasResponse {
    status: String,
    result: PolygonscanGasResult,
}

#[derive(Debug, Deserialize)]
struct PolygonscanGasResult {
    #[serde(rename = "SafeGasPrice")]
    safe_gas_price: String,
    #[serde(rename = "ProposeGasPrice")]
    propose_gas_price: String,
    #[serde(rename = "FastGasPrice")]
    fast_gas_price: String,
}

/// Gas tracker state
struct GasTrackerState {
    /// Current gas price
    current_price: Option<GasPrice>,
    /// Gas price history
    history: GasPriceHistory,
    /// Current gas condition
    current_condition: Option<GasCondition>,
    /// Last successful fetch time
    last_fetch: Option<chrono::DateTime<Utc>>,
}

impl GasTrackerState {
    fn new(history_duration_secs: u64) -> Self {
        Self {
            current_price: None,
            history: GasPriceHistory::with_duration(history_duration_secs),
            current_condition: None,
            last_fetch: None,
        }
    }
}

/// Gas tracker service for monitoring Polygon gas prices
pub struct GasTracker {
    /// Configuration
    config: GasConfig,
    /// HTTP client
    client: reqwest::Client,
    /// Shared state
    state: Arc<RwLock<GasTrackerState>>,
    /// Event sender
    event_tx: broadcast::Sender<SystemEvent>,
}

impl GasTracker {
    /// Create a new gas tracker
    pub fn new(config: GasConfig, event_tx: broadcast::Sender<SystemEvent>) -> Self {
        let state = GasTrackerState::new(config.history_duration_secs);

        Self {
            config,
            client: reqwest::Client::new(),
            state: Arc::new(RwLock::new(state)),
            event_tx,
        }
    }

    /// Start the gas tracking loop
    pub async fn run(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Gas tracking is disabled");
            return Ok(());
        }

        info!(
            "Starting gas tracker with {}s poll interval",
            self.config.poll_interval_secs
        );

        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(self.config.poll_interval_secs));

        loop {
            interval.tick().await;

            if let Err(e) = self.poll_gas_price().await {
                error!("Failed to poll gas price: {}", e);
            }
        }
    }

    /// Poll gas price from RPC or fallback
    async fn poll_gas_price(&self) -> Result<()> {
        // Try primary RPC first
        let gas_price = match self.fetch_from_rpc().await {
            Ok(price) => price,
            Err(e) => {
                warn!("Primary RPC failed: {}, trying fallback", e);
                self.fetch_from_polygonscan().await?
            }
        };

        self.process_gas_price(gas_price).await
    }

    /// Fetch gas price from Polygon RPC (eth_gasPrice)
    async fn fetch_from_rpc(&self) -> Result<GasPrice> {
        let request_body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "eth_gasPrice",
            "params": [],
            "id": 1
        });

        let response = self
            .client
            .post(&self.config.rpc_endpoint)
            .json(&request_body)
            .send()
            .await
            .context("Failed to send RPC request")?;

        let json: serde_json::Value = response.json().await.context("Failed to parse RPC response")?;

        let result = json
            .get("result")
            .and_then(|r| r.as_str())
            .context("Missing result in RPC response")?;

        // Parse hex gas price (in wei) to gwei
        let gas_wei = u128::from_str_radix(result.trim_start_matches("0x"), 16)
            .context("Failed to parse hex gas price")?;

        let gas_gwei = Decimal::from(gas_wei) / Decimal::from(1_000_000_000u64);

        debug!("Fetched gas price from RPC: {} gwei", gas_gwei);

        // RPC only gives a single value, use it for all tiers with adjustments
        Ok(GasPrice::tiered(
            gas_gwei * Decimal::new(120, 2), // fast = 1.2x
            gas_gwei,                         // standard = base
            gas_gwei * Decimal::new(80, 2),  // slow = 0.8x
            GasPriceSource::Rpc,
        ))
    }

    /// Fetch gas price from Polygonscan Gas Oracle API
    async fn fetch_from_polygonscan(&self) -> Result<GasPrice> {
        let endpoint = self
            .config
            .fallback_endpoint
            .as_ref()
            .context("No fallback endpoint configured")?;

        let response = self
            .client
            .get(endpoint)
            .send()
            .await
            .context("Failed to send Polygonscan request")?;

        let gas_data: PolygonscanGasResponse = response
            .json()
            .await
            .context("Failed to parse Polygonscan response")?;

        if gas_data.status != "1" {
            anyhow::bail!("Polygonscan API returned error status");
        }

        let fast: Decimal = gas_data
            .result
            .fast_gas_price
            .parse()
            .context("Failed to parse fast gas price")?;
        let standard: Decimal = gas_data
            .result
            .propose_gas_price
            .parse()
            .context("Failed to parse propose gas price")?;
        let slow: Decimal = gas_data
            .result
            .safe_gas_price
            .parse()
            .context("Failed to parse safe gas price")?;

        debug!(
            "Fetched gas price from Polygonscan: fast={}, standard={}, slow={}",
            fast, standard, slow
        );

        Ok(GasPrice::tiered(fast, standard, slow, GasPriceSource::Polygonscan))
    }

    /// Process a new gas price update
    async fn process_gas_price(&self, gas_price: GasPrice) -> Result<()> {
        let mut state = self.state.write().await;

        let previous_price = state.current_price.clone();
        let previous_condition = state.current_condition;

        // Update history
        state.history.add(gas_price.clone());

        // Classify gas condition
        let condition = GasCondition::classify(gas_price.standard, &self.config.thresholds);

        // Check for spike
        let is_spike = state.history.is_spike(&gas_price);

        // Calculate rolling average
        let average_gwei = state.history.average_standard();

        // Update current state
        state.current_price = Some(gas_price.clone());
        state.current_condition = Some(condition);
        state.last_fetch = Some(Utc::now());

        // Create and emit event
        let event = GasPriceUpdateEvent::new(
            gas_price,
            previous_price,
            condition,
            previous_condition,
            is_spike,
            average_gwei,
        );

        // Log significant changes
        if event.condition_changed() {
            info!(
                "Gas condition changed: {:?} -> {}",
                previous_condition, condition
            );
        }

        if is_spike {
            warn!(
                "Gas spike detected! Current: {} gwei, Average: {:?} gwei",
                event.gas_price.standard, average_gwei
            );
        }

        // Emit event (ignore send errors if no receivers)
        let _ = self.event_tx.send(SystemEvent::GasPriceUpdate(event));

        Ok(())
    }

    /// Get current gas price
    pub async fn current_price(&self) -> Option<GasPrice> {
        self.state.read().await.current_price.clone()
    }

    /// Get current gas condition
    pub async fn current_condition(&self) -> Option<GasCondition> {
        self.state.read().await.current_condition
    }

    /// Get gas price history
    pub async fn history_stats(&self) -> GasHistoryStats {
        let state = self.state.read().await;
        GasHistoryStats {
            count: state.history.len(),
            average_standard: state.history.average_standard(),
            average_fast: state.history.average_fast(),
            min_max: state.history.min_max_standard(),
            p50: state.history.percentile_standard(50),
            p95: state.history.percentile_standard(95),
        }
    }

    /// Estimate gas cost for an operation
    pub async fn estimate_cost(&self, operation: GasOperation) -> Option<GasCostEstimate> {
        let state = self.state.read().await;
        let gas_price = state.current_price.as_ref()?;

        let gas_units = operation.estimated_gas_units();
        let gas_units_decimal = Decimal::from(gas_units);

        // Convert gwei to MATIC (1 MATIC = 1e9 gwei)
        let gwei_to_matic = Decimal::from(1_000_000_000u64);

        Some(GasCostEstimate {
            operation,
            gas_units,
            cost_fast_matic: (gas_price.fast * gas_units_decimal) / gwei_to_matic,
            cost_standard_matic: (gas_price.standard * gas_units_decimal) / gwei_to_matic,
            cost_slow_matic: (gas_price.slow * gas_units_decimal) / gwei_to_matic,
            cost_usd: None, // Would need MATIC/USD price feed
            timestamp: Utc::now(),
        })
    }

    /// Check if current gas conditions are favorable for trading
    pub async fn is_favorable(&self) -> bool {
        self.state
            .read()
            .await
            .current_condition
            .map(|c| c.is_favorable())
            .unwrap_or(true) // Default to true if no data yet
    }
}

/// Statistics about gas price history
#[derive(Debug, Clone)]
pub struct GasHistoryStats {
    /// Number of samples in history
    pub count: usize,
    /// Average standard gas price
    pub average_standard: Option<Decimal>,
    /// Average fast gas price
    pub average_fast: Option<Decimal>,
    /// Min and max standard gas prices
    pub min_max: Option<(Decimal, Decimal)>,
    /// 50th percentile
    pub p50: Option<Decimal>,
    /// 95th percentile
    pub p95: Option<Decimal>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use tokio::sync::broadcast;

    fn create_test_config() -> GasConfig {
        GasConfig {
            enabled: true,
            poll_interval_secs: 10,
            rpc_endpoint: "https://polygon-rpc.com".to_string(),
            fallback_endpoint: None,
            cache_ttl_secs: 10,
            history_duration_secs: 3600,
            thresholds: polysniper_core::GasThresholds::default(),
            optimization: polysniper_core::GasOptimizationConfig::default(),
        }
    }

    #[tokio::test]
    async fn test_gas_tracker_creation() {
        let config = create_test_config();
        let (tx, _rx) = broadcast::channel(100);
        let tracker = GasTracker::new(config, tx);

        assert!(tracker.current_price().await.is_none());
        assert!(tracker.current_condition().await.is_none());
    }

    #[tokio::test]
    async fn test_process_gas_price() {
        let config = create_test_config();
        let (tx, mut rx) = broadcast::channel(100);
        let tracker = GasTracker::new(config, tx);

        let gas_price = GasPrice::uniform(dec!(50), GasPriceSource::Rpc);
        tracker.process_gas_price(gas_price.clone()).await.unwrap();

        // Check state was updated
        let current = tracker.current_price().await.unwrap();
        assert_eq!(current.standard, dec!(50));

        let condition = tracker.current_condition().await.unwrap();
        assert_eq!(condition, GasCondition::Normal);

        // Check event was emitted
        let event = rx.try_recv().unwrap();
        match event {
            SystemEvent::GasPriceUpdate(e) => {
                assert_eq!(e.gas_price.standard, dec!(50));
                assert_eq!(e.condition, GasCondition::Normal);
            }
            _ => panic!("Expected GasPriceUpdate event"),
        }
    }

    #[tokio::test]
    async fn test_estimate_cost() {
        let config = create_test_config();
        let (tx, _rx) = broadcast::channel(100);
        let tracker = GasTracker::new(config, tx);

        // No price yet
        assert!(tracker.estimate_cost(GasOperation::LimitOrder).await.is_none());

        // Add a price
        let gas_price = GasPrice::uniform(dec!(50), GasPriceSource::Rpc);
        tracker.process_gas_price(gas_price).await.unwrap();

        let estimate = tracker.estimate_cost(GasOperation::LimitOrder).await.unwrap();
        assert_eq!(estimate.gas_units, 150_000);
        // 50 gwei * 150000 / 1e9 = 0.0075 MATIC
        assert_eq!(estimate.cost_standard_matic, dec!(0.0075));
    }

    #[tokio::test]
    async fn test_history_stats() {
        let config = create_test_config();
        let (tx, _rx) = broadcast::channel(100);
        let tracker = GasTracker::new(config, tx);

        // Add multiple prices
        for i in 1..=10 {
            let gas_price = GasPrice::uniform(Decimal::from(i * 10), GasPriceSource::Rpc);
            tracker.process_gas_price(gas_price).await.unwrap();
        }

        let stats = tracker.history_stats().await;
        assert_eq!(stats.count, 10);
        assert_eq!(stats.average_standard, Some(dec!(55))); // (10+20+...+100)/10 = 55
        assert_eq!(stats.min_max, Some((dec!(10), dec!(100))));
    }

    #[tokio::test]
    async fn test_is_favorable() {
        let config = create_test_config();
        let (tx, _rx) = broadcast::channel(100);
        let tracker = GasTracker::new(config, tx);

        // Default to true when no data
        assert!(tracker.is_favorable().await);

        // Low gas = favorable
        let gas_price = GasPrice::uniform(dec!(20), GasPriceSource::Rpc);
        tracker.process_gas_price(gas_price).await.unwrap();
        assert!(tracker.is_favorable().await);

        // Extreme gas = not favorable
        let gas_price = GasPrice::uniform(dec!(500), GasPriceSource::Rpc);
        tracker.process_gas_price(gas_price).await.unwrap();
        assert!(!tracker.is_favorable().await);
    }

    #[tokio::test]
    async fn test_condition_change_detection() {
        let config = create_test_config();
        let (tx, mut rx) = broadcast::channel(100);
        let tracker = GasTracker::new(config, tx);

        // First update - no previous condition
        let gas_price = GasPrice::uniform(dec!(20), GasPriceSource::Rpc);
        tracker.process_gas_price(gas_price).await.unwrap();

        let event1 = rx.try_recv().unwrap();
        match event1 {
            SystemEvent::GasPriceUpdate(e) => {
                assert!(!e.condition_changed()); // No previous condition
            }
            _ => panic!("Expected GasPriceUpdate event"),
        }

        // Second update - condition changes from Low to Extreme
        let gas_price = GasPrice::uniform(dec!(500), GasPriceSource::Rpc);
        tracker.process_gas_price(gas_price).await.unwrap();

        let event2 = rx.try_recv().unwrap();
        match event2 {
            SystemEvent::GasPriceUpdate(e) => {
                assert!(e.condition_changed());
                assert_eq!(e.previous_condition, Some(GasCondition::Low));
                assert_eq!(e.condition, GasCondition::Extreme);
            }
            _ => panic!("Expected GasPriceUpdate event"),
        }
    }
}
