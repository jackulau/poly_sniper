---
id: feature-store
name: Feature Store for Consistent Feature Computation
wave: 1
priority: 1
dependencies: []
estimated_hours: 6
tags: [ml, backend, infrastructure]
---

## Objective

Build a feature store that provides consistent feature computation across backtesting and live trading, ensuring ML models receive identical feature values regardless of execution context.

## Context

Currently, features are computed ad-hoc in different places (LLM prediction strategy, sentiment analyzer, ML processor). This leads to potential train-serve skew where features computed during backtesting differ from live trading. A centralized feature store ensures:
- Identical feature computation logic for training/backtesting/live
- Feature versioning and lineage tracking
- Point-in-time feature retrieval for backtesting
- Caching for real-time feature serving

## Implementation

### 1. Create Feature Store Core (`crates/polysniper-ml/src/feature_store.rs`)

```rust
pub struct FeatureStore {
    // Cached computed features
    cache: Arc<RwLock<HashMap<FeatureKey, FeatureValue>>>,
    // Feature definitions registry
    registry: FeatureRegistry,
    // Persistence for historical features
    db: Arc<Database>,
    // Feature computation engine
    compute_engine: FeatureComputeEngine,
}

pub struct FeatureKey {
    pub market_id: String,
    pub feature_name: String,
    pub timestamp: DateTime<Utc>,
    pub version: String,
}

pub struct FeatureValue {
    pub value: serde_json::Value,
    pub computed_at: DateTime<Utc>,
    pub ttl: Duration,
    pub metadata: HashMap<String, String>,
}

pub trait FeatureComputer: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn compute(&self, context: &FeatureContext) -> Result<serde_json::Value>;
    fn dependencies(&self) -> Vec<&str>;
}
```

### 2. Define Standard Feature Computers (`crates/polysniper-ml/src/features/`)

```rust
// market_features.rs
pub struct MarketFeatureComputer;
impl FeatureComputer for MarketFeatureComputer {
    // liquidity, volume, time_to_expiry, current_price, price_spread
}

// orderbook_features.rs
pub struct OrderbookFeatureComputer;
impl FeatureComputer for OrderbookFeatureComputer {
    // bid_depth, ask_depth, imbalance_ratio, microprice, weighted_mid
}

// sentiment_features.rs
pub struct SentimentFeatureComputer;
impl FeatureComputer for SentimentFeatureComputer {
    // sentiment_score, keyword_count, source_diversity
}

// temporal_features.rs
pub struct TemporalFeatureComputer;
impl FeatureComputer for TemporalFeatureComputer {
    // hour_of_day, day_of_week, time_to_resolution, market_age
}

// price_history_features.rs
pub struct PriceHistoryFeatureComputer;
impl FeatureComputer for PriceHistoryFeatureComputer {
    // volatility, momentum, mean_reversion_signal, price_velocity
}
```

### 3. Point-in-Time Feature Retrieval for Backtesting

```rust
impl FeatureStore {
    /// Get features as they would have been at a specific point in time
    pub async fn get_features_at(
        &self,
        market_id: &str,
        timestamp: DateTime<Utc>,
        feature_names: &[&str],
    ) -> Result<HashMap<String, FeatureValue>> {
        // Uses historical snapshots, not current data
    }

    /// Persist features for future backtesting
    pub async fn store_features(
        &self,
        market_id: &str,
        features: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        // Store with timestamp for point-in-time retrieval
    }
}
```

### 4. Integration with Existing ML Components

Modify `crates/polysniper-strategies/src/llm_prediction.rs`:
```rust
// Use feature store instead of inline computation
let features = self.feature_store.get_current_features(
    &market.condition_id,
    &["liquidity", "volatility", "sentiment", "orderbook_imbalance"]
).await?;
```

Modify `crates/polysniper-backtest/src/engine.rs`:
```rust
// Load historical features for backtesting
let features = self.feature_store.get_features_at(
    &market_id,
    event_timestamp,
    required_features
).await?;
```

### 5. Configuration (`config/feature_store.toml`)

```toml
[feature_store]
enabled = true
cache_ttl_secs = 60
persistence_enabled = true
db_path = "data/features.db"

[feature_store.features]
market = { enabled = true, version = "1.0" }
orderbook = { enabled = true, version = "1.0" }
sentiment = { enabled = true, version = "1.0" }
temporal = { enabled = true, version = "1.0" }
price_history = { enabled = true, version = "1.0", window_secs = 3600 }
```

## Acceptance Criteria

- [ ] Feature store core implemented with caching and persistence
- [ ] At least 5 feature computers implemented (market, orderbook, sentiment, temporal, price_history)
- [ ] Point-in-time feature retrieval works for backtesting
- [ ] Feature versioning prevents train-serve skew
- [ ] Integration tests verify feature consistency between live and backtest
- [ ] LLM prediction strategy refactored to use feature store
- [ ] All existing tests pass
- [ ] No security vulnerabilities introduced

## Files to Create/Modify

**Create:**
- `crates/polysniper-ml/Cargo.toml` - New crate for ML infrastructure
- `crates/polysniper-ml/src/lib.rs` - Crate entry point
- `crates/polysniper-ml/src/feature_store.rs` - Core feature store
- `crates/polysniper-ml/src/features/mod.rs` - Feature computers module
- `crates/polysniper-ml/src/features/market.rs` - Market features
- `crates/polysniper-ml/src/features/orderbook.rs` - Orderbook features
- `crates/polysniper-ml/src/features/sentiment.rs` - Sentiment features
- `crates/polysniper-ml/src/features/temporal.rs` - Temporal features
- `crates/polysniper-ml/src/features/price_history.rs` - Price history features
- `config/feature_store.toml` - Configuration

**Modify:**
- `Cargo.toml` - Add polysniper-ml to workspace members
- `crates/polysniper-strategies/Cargo.toml` - Add polysniper-ml dependency
- `crates/polysniper-strategies/src/llm_prediction.rs` - Use feature store
- `crates/polysniper-backtest/Cargo.toml` - Add polysniper-ml dependency
- `crates/polysniper-backtest/src/engine.rs` - Use feature store for backtesting

## Integration Points

- **Provides**: Consistent feature computation API for all ML components
- **Consumes**: StateProvider (market cache), Database (persistence)
- **Conflicts**: Avoid modifying ml_processor.rs (handled by ensemble-predictions task)

## Testing Strategy

1. Unit tests for each feature computer
2. Integration tests verifying feature consistency
3. Backtest tests using point-in-time features
4. Cache invalidation tests
