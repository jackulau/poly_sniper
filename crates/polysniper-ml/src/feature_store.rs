//! Feature Store for consistent feature computation
//!
//! Provides a centralized store for computing and caching features,
//! ensuring consistency between backtesting and live trading.

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use polysniper_core::{Market, Orderbook, StateProvider};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Errors that can occur in the feature store
#[derive(Error, Debug)]
pub enum FeatureStoreError {
    #[error("Feature not found: {0}")]
    FeatureNotFound(String),
    #[error("Feature computation failed: {0}")]
    ComputationError(String),
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
    #[error("Missing dependency: {0}")]
    MissingDependency(String),
}

pub type Result<T> = std::result::Result<T, FeatureStoreError>;

/// Key for looking up cached features
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct FeatureKey {
    pub market_id: String,
    pub feature_name: String,
    pub timestamp: DateTime<Utc>,
    pub version: String,
}

impl FeatureKey {
    pub fn new(market_id: &str, feature_name: &str, timestamp: DateTime<Utc>, version: &str) -> Self {
        Self {
            market_id: market_id.to_string(),
            feature_name: feature_name.to_string(),
            timestamp,
            version: version.to_string(),
        }
    }

    /// Create a key for current time
    pub fn current(market_id: &str, feature_name: &str, version: &str) -> Self {
        Self::new(market_id, feature_name, Utc::now(), version)
    }
}

/// Value stored for a computed feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureValue {
    pub value: serde_json::Value,
    pub computed_at: DateTime<Utc>,
    pub ttl: Duration,
    pub metadata: HashMap<String, String>,
}

impl FeatureValue {
    pub fn new(value: serde_json::Value, ttl: Duration) -> Self {
        Self {
            value,
            computed_at: Utc::now(),
            ttl,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn is_expired(&self) -> bool {
        Utc::now() > self.computed_at + self.ttl
    }

    /// Get the value as a Decimal if possible
    pub fn as_decimal(&self) -> Option<Decimal> {
        match &self.value {
            serde_json::Value::Number(n) => {
                n.as_f64().and_then(|f| Decimal::try_from(f).ok())
            }
            serde_json::Value::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Get the value as an f64 if possible
    pub fn as_f64(&self) -> Option<f64> {
        self.value.as_f64()
    }
}

/// Context provided to feature computers
#[derive(Debug, Clone)]
pub struct FeatureContext {
    pub market: Market,
    pub orderbook: Option<Orderbook>,
    pub price: Option<Decimal>,
    pub price_history: Vec<(DateTime<Utc>, Decimal)>,
    pub timestamp: DateTime<Utc>,
    pub computed_features: HashMap<String, FeatureValue>,
}

impl FeatureContext {
    pub fn new(market: Market, timestamp: DateTime<Utc>) -> Self {
        Self {
            market,
            orderbook: None,
            price: None,
            price_history: Vec::new(),
            timestamp,
            computed_features: HashMap::new(),
        }
    }

    pub fn with_orderbook(mut self, orderbook: Orderbook) -> Self {
        self.orderbook = Some(orderbook);
        self
    }

    pub fn with_price(mut self, price: Decimal) -> Self {
        self.price = Some(price);
        self
    }

    pub fn with_price_history(mut self, history: Vec<(DateTime<Utc>, Decimal)>) -> Self {
        self.price_history = history;
        self
    }
}

/// Trait for computing features
#[async_trait]
pub trait FeatureComputer: Send + Sync {
    /// Name of this feature
    fn name(&self) -> &str;

    /// Version of this feature computation (for versioning)
    fn version(&self) -> &str;

    /// Compute the feature given the context
    async fn compute(&self, context: &FeatureContext) -> Result<serde_json::Value>;

    /// List of feature names this computer depends on
    fn dependencies(&self) -> Vec<&str>;

    /// Default TTL for cached values
    fn default_ttl(&self) -> Duration {
        Duration::seconds(60)
    }
}

/// Registry of available feature computers
pub struct FeatureRegistry {
    computers: HashMap<String, Arc<dyn FeatureComputer>>,
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureRegistry {
    pub fn new() -> Self {
        Self {
            computers: HashMap::new(),
        }
    }

    pub fn register(&mut self, computer: Arc<dyn FeatureComputer>) {
        self.computers.insert(computer.name().to_string(), computer);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn FeatureComputer>> {
        self.computers.get(name).cloned()
    }

    pub fn all(&self) -> Vec<Arc<dyn FeatureComputer>> {
        self.computers.values().cloned().collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.computers.keys().map(|s| s.as_str()).collect()
    }
}

/// Configuration for the feature store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStoreConfig {
    pub enabled: bool,
    pub cache_ttl_secs: u64,
    pub persistence_enabled: bool,
    pub db_path: String,
}

impl Default for FeatureStoreConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_ttl_secs: 60,
            persistence_enabled: true,
            db_path: "data/features.db".to_string(),
        }
    }
}

/// Simplified cache key for in-memory storage (without timestamp for current values)
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct CacheKey {
    market_id: String,
    feature_name: String,
    version: String,
}

/// The main feature store
pub struct FeatureStore {
    cache: Arc<RwLock<HashMap<CacheKey, FeatureValue>>>,
    registry: Arc<RwLock<FeatureRegistry>>,
    config: FeatureStoreConfig,
    historical_features: Arc<RwLock<HashMap<FeatureKey, FeatureValue>>>,
}

impl FeatureStore {
    pub fn new(config: FeatureStoreConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            registry: Arc::new(RwLock::new(FeatureRegistry::new())),
            config,
            historical_features: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a feature computer
    pub async fn register_computer(&self, computer: Arc<dyn FeatureComputer>) {
        self.registry.write().await.register(computer);
    }

    /// Get current features for a market
    pub async fn get_current_features(
        &self,
        market_id: &str,
        feature_names: &[&str],
        state: &dyn StateProvider,
    ) -> Result<HashMap<String, FeatureValue>> {
        let mut results = HashMap::new();

        // Get market data
        let market = state
            .get_market(&market_id.to_string())
            .await
            .ok_or_else(|| FeatureStoreError::ComputationError(format!("Market not found: {}", market_id)))?;

        // Build context
        let mut context = FeatureContext::new(market.clone(), Utc::now());

        // Add orderbook if available
        if let Some(orderbook) = state.get_orderbook(&market.yes_token_id).await {
            context = context.with_orderbook(orderbook);
        }

        // Add price if available
        if let Some(price) = state.get_price(&market.yes_token_id).await {
            context = context.with_price(price);
        }

        // Add price history
        let history = state.get_price_history(&market.yes_token_id, 100).await;
        context = context.with_price_history(history);

        // Compute each requested feature
        for feature_name in feature_names {
            match self.compute_feature(market_id, feature_name, &context).await {
                Ok(value) => {
                    results.insert(feature_name.to_string(), value);
                }
                Err(e) => {
                    warn!(
                        market_id = %market_id,
                        feature = %feature_name,
                        error = %e,
                        "Failed to compute feature"
                    );
                }
            }
        }

        Ok(results)
    }

    /// Get features as they would have been at a specific point in time (for backtesting)
    pub async fn get_features_at(
        &self,
        market_id: &str,
        timestamp: DateTime<Utc>,
        feature_names: &[&str],
    ) -> Result<HashMap<String, FeatureValue>> {
        let mut results = HashMap::new();
        let historical = self.historical_features.read().await;

        for feature_name in feature_names {
            // Find the most recent feature value before or at the timestamp
            let registry = self.registry.read().await;
            let version = registry
                .get(feature_name)
                .map(|c| c.version().to_string())
                .unwrap_or_else(|| "1.0".to_string());

            let key = FeatureKey::new(market_id, feature_name, timestamp, &version);

            // Look for exact match first
            if let Some(value) = historical.get(&key) {
                results.insert(feature_name.to_string(), value.clone());
                continue;
            }

            // Otherwise, find the closest timestamp before the requested time
            let matching_features: Vec<_> = historical
                .iter()
                .filter(|(k, _)| {
                    k.market_id == market_id
                        && k.feature_name == *feature_name
                        && k.version == version
                        && k.timestamp <= timestamp
                })
                .collect();

            if let Some((_, value)) = matching_features
                .into_iter()
                .max_by_key(|(k, _)| k.timestamp)
            {
                results.insert(feature_name.to_string(), value.clone());
            }
        }

        Ok(results)
    }

    /// Store features for future backtesting
    pub async fn store_features(
        &self,
        market_id: &str,
        features: HashMap<String, serde_json::Value>,
        timestamp: DateTime<Utc>,
    ) -> Result<()> {
        let registry = self.registry.read().await;
        let mut historical = self.historical_features.write().await;

        for (feature_name, value) in features {
            let version = registry
                .get(&feature_name)
                .map(|c| c.version().to_string())
                .unwrap_or_else(|| "1.0".to_string());

            let key = FeatureKey::new(market_id, &feature_name, timestamp, &version);
            let feature_value = FeatureValue::new(
                value,
                Duration::seconds(self.config.cache_ttl_secs as i64),
            );

            historical.insert(key, feature_value);
        }

        Ok(())
    }

    /// Compute a single feature
    async fn compute_feature(
        &self,
        market_id: &str,
        feature_name: &str,
        context: &FeatureContext,
    ) -> Result<FeatureValue> {
        let registry = self.registry.read().await;
        let computer = registry
            .get(feature_name)
            .ok_or_else(|| FeatureStoreError::FeatureNotFound(feature_name.to_string()))?;

        let cache_key = CacheKey {
            market_id: market_id.to_string(),
            feature_name: feature_name.to_string(),
            version: computer.version().to_string(),
        };

        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                if !cached.is_expired() {
                    debug!(
                        market_id = %market_id,
                        feature = %feature_name,
                        "Cache hit"
                    );
                    return Ok(cached.clone());
                }
            }
        }

        // Compute dependencies first
        let mut context_with_deps = context.clone();
        for dep_name in computer.dependencies() {
            if !context_with_deps.computed_features.contains_key(dep_name) {
                if let Ok(dep_value) = Box::pin(self.compute_feature(market_id, dep_name, context)).await {
                    context_with_deps.computed_features.insert(dep_name.to_string(), dep_value);
                }
            }
        }

        // Compute the feature
        let value = computer.compute(&context_with_deps).await?;
        let feature_value = FeatureValue::new(value, computer.default_ttl());

        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, feature_value.clone());
        }

        debug!(
            market_id = %market_id,
            feature = %feature_name,
            "Computed feature"
        );

        Ok(feature_value)
    }

    /// Clear expired entries from the cache
    pub async fn clear_expired(&self) {
        let mut cache = self.cache.write().await;
        cache.retain(|_, v| !v.is_expired());
    }

    /// Clear all cached values for a specific market
    pub async fn invalidate_market(&self, market_id: &str) {
        let mut cache = self.cache.write().await;
        cache.retain(|k, _| k.market_id != market_id);
    }

    /// Get all registered feature names
    pub async fn available_features(&self) -> Vec<String> {
        let registry = self.registry.read().await;
        registry.names().into_iter().map(|s| s.to_string()).collect()
    }

    /// Compute all available features for a market
    pub async fn compute_all_features(
        &self,
        market_id: &str,
        state: &dyn StateProvider,
    ) -> Result<HashMap<String, FeatureValue>> {
        let feature_names: Vec<String> = self.available_features().await;
        let feature_refs: Vec<&str> = feature_names.iter().map(|s| s.as_str()).collect();
        self.get_current_features(market_id, &feature_refs, state).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    struct MockFeatureComputer;

    #[async_trait]
    impl FeatureComputer for MockFeatureComputer {
        fn name(&self) -> &str {
            "test_feature"
        }

        fn version(&self) -> &str {
            "1.0"
        }

        async fn compute(&self, _context: &FeatureContext) -> Result<serde_json::Value> {
            Ok(serde_json::json!(42.0))
        }

        fn dependencies(&self) -> Vec<&str> {
            vec![]
        }
    }

    #[test]
    fn test_feature_key() {
        let key1 = FeatureKey::new("market1", "feature1", Utc::now(), "1.0");
        let key2 = FeatureKey::new("market1", "feature1", Utc::now(), "1.0");
        assert_eq!(key1.market_id, key2.market_id);
        assert_eq!(key1.feature_name, key2.feature_name);
    }

    #[test]
    fn test_feature_value_expiry() {
        let value = FeatureValue::new(serde_json::json!(1.0), Duration::seconds(60));
        assert!(!value.is_expired());

        let old_value = FeatureValue {
            value: serde_json::json!(1.0),
            computed_at: Utc::now() - Duration::seconds(120),
            ttl: Duration::seconds(60),
            metadata: HashMap::new(),
        };
        assert!(old_value.is_expired());
    }

    #[test]
    fn test_feature_value_as_decimal() {
        let value = FeatureValue::new(serde_json::json!(42.5), Duration::seconds(60));
        let decimal = value.as_decimal().unwrap();
        assert!(decimal > dec!(42) && decimal < dec!(43));
    }

    #[test]
    fn test_registry() {
        let mut registry = FeatureRegistry::new();
        let computer = Arc::new(MockFeatureComputer) as Arc<dyn FeatureComputer>;
        registry.register(computer);

        assert!(registry.get("test_feature").is_some());
        assert!(registry.get("nonexistent").is_none());
        assert_eq!(registry.names().len(), 1);
    }
}
