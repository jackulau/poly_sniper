---
id: cross-asset-signals
name: Cross-Asset Signals from Crypto Price Movements
wave: 1
priority: 1
dependencies: []
estimated_hours: 6
tags: [alpha, crypto, signals, data-source]
---

## Objective

Implement cross-asset signal generation that monitors crypto price movements (ETH, BTC, SOL) to predict related Polymarket market movements (e.g., ETH price surge → ETF approval odds increase).

## Context

Crypto prices often lead prediction market movements. For example:
- ETH price pumps may precede ETF approval odds increases
- BTC movements correlate with crypto-related regulation markets
- SOL price action may affect Solana ecosystem prediction markets

This task adds a new data source and strategy to detect and act on these correlations.

## Implementation

### 1. Create Crypto Price Client

**File:** `crates/polysniper-data/src/crypto_price_client.rs`

```rust
pub struct CryptoPriceClient {
    http_client: reqwest::Client,
    event_tx: broadcast::Sender<SystemEvent>,
    config: CryptoPriceConfig,
    price_cache: Arc<RwLock<HashMap<String, CryptoPrice>>>,
}

pub struct CryptoPrice {
    pub symbol: String,           // "ETH", "BTC", "SOL"
    pub price_usd: Decimal,
    pub price_change_1h: Decimal, // Percentage
    pub price_change_24h: Decimal,
    pub volume_24h: Decimal,
    pub timestamp: DateTime<Utc>,
}

pub struct CryptoPriceConfig {
    pub enabled: bool,
    pub symbols: Vec<String>,          // ["ETH", "BTC", "SOL"]
    pub poll_interval_secs: u64,       // Default: 30
    pub api_provider: CryptoApiProvider, // CoinGecko, Binance, etc.
    pub api_key: Option<String>,
}

pub enum CryptoApiProvider {
    CoinGecko,
    Binance,
    CoinMarketCap,
}
```

- Implement polling loop fetching from CoinGecko (free tier) or Binance
- Emit `CryptoPriceUpdate` events to EventBus
- Track price history for velocity calculations

### 2. Add SystemEvent Variant

**File:** `crates/polysniper-core/src/events.rs`

Add new event type:
```rust
CryptoPriceUpdate(CryptoPriceUpdateEvent),

pub struct CryptoPriceUpdateEvent {
    pub symbol: String,
    pub price: Decimal,
    pub price_change_1h: Decimal,
    pub price_change_24h: Decimal,
    pub volume_24h: Decimal,
    pub timestamp: DateTime<Utc>,
}
```

### 3. Create Cross-Asset Strategy

**File:** `crates/polysniper-strategies/src/cross_asset.rs`

```rust
pub struct CrossAssetStrategy {
    config: Arc<RwLock<CrossAssetConfig>>,
    price_history: HashMap<String, VecDeque<(DateTime<Utc>, Decimal)>>,
    cooldowns: HashMap<String, DateTime<Utc>>,
}

pub struct CrossAssetConfig {
    pub enabled: bool,
    pub correlations: Vec<AssetCorrelation>,
    pub min_price_change_pct: Decimal,      // Minimum crypto move to trigger
    pub signal_delay_secs: u64,             // Wait before generating signal
    pub cooldown_secs: u64,                 // Per-correlation cooldown
}

pub struct AssetCorrelation {
    pub crypto_symbol: String,              // "ETH"
    pub market_keywords: Vec<String>,       // ["eth", "ethereum", "etf"]
    pub correlation_type: CorrelationType,  // Positive or Negative
    pub min_confidence: Decimal,
    pub order_size_usd: Decimal,
}

pub enum CorrelationType {
    Positive,  // Crypto up → Market YES up
    Negative,  // Crypto up → Market YES down
}
```

- Subscribe to `CryptoPriceUpdate` events
- Detect significant price movements (configurable threshold)
- Match to correlated Polymarket markets via keyword mapping
- Generate trade signals with appropriate delay

### 4. Configuration

**File:** `config/strategies/cross_asset.toml`

```toml
[cross_asset]
enabled = true
min_price_change_pct = 5.0
signal_delay_secs = 60
cooldown_secs = 300

[[cross_asset.correlations]]
crypto_symbol = "ETH"
market_keywords = ["eth", "ethereum", "etf", "spot etf"]
correlation_type = "Positive"
min_confidence = 0.7
order_size_usd = 50.0

[[cross_asset.correlations]]
crypto_symbol = "BTC"
market_keywords = ["bitcoin", "btc", "strategic reserve"]
correlation_type = "Positive"
min_confidence = 0.7
order_size_usd = 50.0
```

### 5. Update Module Exports

- Add `crypto_price_client` to `crates/polysniper-data/src/lib.rs`
- Add `cross_asset` to `crates/polysniper-strategies/src/lib.rs`
- Register strategy in main application

## Acceptance Criteria

- [ ] CryptoPriceClient fetches prices from CoinGecko/Binance API
- [ ] CryptoPriceUpdate events are published to EventBus
- [ ] CrossAssetStrategy correctly maps crypto movements to markets
- [ ] Configurable correlation rules work correctly
- [ ] Signal delay and cooldown mechanisms function properly
- [ ] All new code has unit tests with mocked API responses
- [ ] Integration with existing event bus architecture
- [ ] No breaking changes to existing strategies

## Files to Create/Modify

**Create:**
- `crates/polysniper-data/src/crypto_price_client.rs`
- `crates/polysniper-strategies/src/cross_asset.rs`
- `config/strategies/cross_asset.toml`

**Modify:**
- `crates/polysniper-core/src/events.rs` - Add CryptoPriceUpdate event
- `crates/polysniper-data/src/lib.rs` - Export crypto_price_client
- `crates/polysniper-strategies/src/lib.rs` - Export cross_asset

## Integration Points

- **Provides**: CryptoPriceUpdate events for other strategies, cross-asset trade signals
- **Consumes**: EventBus for publishing, StateProvider for market lookups
- **Conflicts**: None - new module with clean separation

## Testing Requirements

```rust
#[cfg(test)]
mod tests {
    // Test price parsing from API response
    #[test]
    fn test_parse_coingecko_response() { ... }

    // Test correlation detection
    #[tokio::test]
    async fn test_positive_correlation_signal() { ... }

    // Test cooldown enforcement
    #[tokio::test]
    async fn test_cooldown_prevents_duplicate_signals() { ... }

    // Test market keyword matching
    #[tokio::test]
    async fn test_market_keyword_matching() { ... }
}
```
