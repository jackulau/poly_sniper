---
id: prediction-aggregator-arb
name: Prediction Aggregator Arbitrage (Metaculus, PredictIt, Kalshi)
wave: 1
priority: 4
dependencies: []
estimated_hours: 7
tags: [alpha, arbitrage, cross-platform, prediction-markets]
---

## Objective

Implement prediction aggregator arbitrage that compares Polymarket odds with Metaculus, PredictIt, and Kalshi to detect pricing discrepancies and edge opportunities.

## Context

Different prediction markets price the same underlying events differently due to:
- Different user bases and biases
- Liquidity differences
- Fee structures
- Market timing (hours, access restrictions)

When Polymarket shows significantly different odds than other platforms, this can indicate:
- **True arbitrage**: Risk-free profit across platforms
- **Soft arbitrage**: Edge where one market is likely mispriced
- **Convergence opportunity**: Bet that odds will converge

This task adds data sources for external prediction markets and a strategy to detect and act on discrepancies.

## Implementation

### 1. Create External Prediction Market Clients

**File:** `crates/polysniper-data/src/external_markets/mod.rs`

```rust
pub mod metaculus;
pub mod predictit;
pub mod kalshi;

pub use metaculus::MetaculusClient;
pub use predictit::PredictItClient;
pub use kalshi::KalshiClient;

pub struct ExternalMarketPrice {
    pub platform: Platform,
    pub question_id: String,
    pub question: String,
    pub yes_price: Decimal,
    pub no_price: Option<Decimal>,
    pub volume: Option<Decimal>,
    pub last_updated: DateTime<Utc>,
    pub market_close: Option<DateTime<Utc>>,
}

pub enum Platform {
    Metaculus,
    PredictIt,
    Kalshi,
}
```

**File:** `crates/polysniper-data/src/external_markets/metaculus.rs`

```rust
pub struct MetaculusClient {
    http_client: reqwest::Client,
    config: MetaculusConfig,
    cache: Arc<RwLock<HashMap<String, CachedQuestion>>>,
}

pub struct MetaculusConfig {
    pub enabled: bool,
    pub api_base_url: String,           // "https://www.metaculus.com/api2"
    pub poll_interval_secs: u64,
    pub tracked_questions: Vec<String>, // Question IDs or search terms
}

impl MetaculusClient {
    /// Fetch current community prediction for a question
    pub async fn get_question(&self, question_id: &str) -> Result<MetaculusPrediction> {
        // GET /api2/questions/{id}/
        // Extract: community_prediction (probability), resolution_criteria
    }

    /// Search for questions matching keywords
    pub async fn search_questions(&self, query: &str) -> Result<Vec<MetaculusPrediction>> {
        // GET /api2/questions/?search={query}
    }
}

pub struct MetaculusPrediction {
    pub question_id: String,
    pub title: String,
    pub community_prediction: Decimal,  // 0.0-1.0 probability
    pub prediction_count: u32,
    pub close_time: DateTime<Utc>,
    pub resolution_date: Option<DateTime<Utc>>,
}
```

**File:** `crates/polysniper-data/src/external_markets/predictit.rs`

```rust
pub struct PredictItClient {
    http_client: reqwest::Client,
    config: PredictItConfig,
    cache: Arc<RwLock<HashMap<String, CachedContract>>>,
}

pub struct PredictItConfig {
    pub enabled: bool,
    pub api_base_url: String,           // "https://www.predictit.org/api"
    pub poll_interval_secs: u64,
    pub tracked_markets: Vec<i32>,      // Market IDs
}

impl PredictItClient {
    /// Fetch all markets
    pub async fn get_all_markets(&self) -> Result<Vec<PredictItMarket>> {
        // GET /api/marketdata/all/
    }

    /// Fetch specific market
    pub async fn get_market(&self, market_id: i32) -> Result<PredictItMarket> {
        // GET /api/marketdata/markets/{id}
    }
}

pub struct PredictItMarket {
    pub market_id: i32,
    pub name: String,
    pub contracts: Vec<PredictItContract>,
    pub status: String,                 // Open, Closed
}

pub struct PredictItContract {
    pub contract_id: i64,
    pub name: String,
    pub best_buy_yes: Option<Decimal>,
    pub best_sell_yes: Option<Decimal>,
    pub best_buy_no: Option<Decimal>,
    pub best_sell_no: Option<Decimal>,
    pub last_trade_price: Decimal,
    pub last_close_price: Decimal,
}
```

**File:** `crates/polysniper-data/src/external_markets/kalshi.rs`

```rust
pub struct KalshiClient {
    http_client: reqwest::Client,
    config: KalshiConfig,
    auth_token: Option<String>,
    cache: Arc<RwLock<HashMap<String, CachedEvent>>>,
}

pub struct KalshiConfig {
    pub enabled: bool,
    pub api_base_url: String,           // "https://trading-api.kalshi.com/trade-api/v2"
    pub email: Option<String>,          // For authenticated requests
    pub password_env: String,           // Environment variable name
    pub poll_interval_secs: u64,
    pub tracked_series: Vec<String>,    // Series tickers
}

impl KalshiClient {
    /// Get events for a series
    pub async fn get_events(&self, series_ticker: &str) -> Result<Vec<KalshiEvent>> {
        // GET /trade-api/v2/events?series_ticker={ticker}
    }

    /// Get markets for an event
    pub async fn get_markets(&self, event_ticker: &str) -> Result<Vec<KalshiMarket>> {
        // GET /trade-api/v2/markets?event_ticker={ticker}
    }
}

pub struct KalshiEvent {
    pub event_ticker: String,
    pub series_ticker: String,
    pub title: String,
    pub category: String,
    pub markets: Vec<KalshiMarket>,
}

pub struct KalshiMarket {
    pub ticker: String,
    pub subtitle: String,
    pub yes_bid: Decimal,
    pub yes_ask: Decimal,
    pub no_bid: Decimal,
    pub no_ask: Decimal,
    pub last_price: Decimal,
    pub volume: u64,
    pub open_interest: u64,
    pub close_time: DateTime<Utc>,
}
```

### 2. Create Prediction Aggregator Service

**File:** `crates/polysniper-data/src/prediction_aggregator.rs`

```rust
pub struct PredictionAggregator {
    metaculus: Option<MetaculusClient>,
    predictit: Option<PredictItClient>,
    kalshi: Option<KalshiClient>,
    event_tx: broadcast::Sender<SystemEvent>,
    mappings: MarketMappings,
}

pub struct MarketMappings {
    /// Map Polymarket condition_id to external market identifiers
    pub mappings: Vec<MarketMapping>,
}

pub struct MarketMapping {
    pub name: String,                       // Human-readable name
    pub polymarket_id: String,              // Polymarket condition_id
    pub metaculus_id: Option<String>,       // Metaculus question ID
    pub predictit_contract: Option<i64>,    // PredictIt contract ID
    pub kalshi_ticker: Option<String>,      // Kalshi market ticker
    pub price_adjustment: Decimal,          // Fee/spread adjustment
}

impl PredictionAggregator {
    /// Poll all external markets and compare with Polymarket
    pub async fn poll_and_compare(&self, polymarket_prices: &HashMap<String, Decimal>) {
        for mapping in &self.mappings.mappings {
            let external_prices = self.get_external_prices(mapping).await;

            if let Some(poly_price) = polymarket_prices.get(&mapping.polymarket_id) {
                let discrepancies = self.find_discrepancies(
                    *poly_price,
                    &external_prices,
                    mapping.price_adjustment,
                );

                if let Some(arb) = discrepancies {
                    self.emit_arbitrage_signal(mapping, arb).await;
                }
            }
        }
    }
}
```

### 3. Add SystemEvent Variants

**File:** `crates/polysniper-core/src/events.rs`

Add new event types:
```rust
ExternalPriceUpdate(ExternalPriceUpdateEvent),
PredictionArbitrageDetected(PredictionArbitrageEvent),

pub struct ExternalPriceUpdateEvent {
    pub platform: Platform,
    pub question_id: String,
    pub yes_price: Decimal,
    pub polymarket_mapping: Option<String>,
    pub timestamp: DateTime<Utc>,
}

pub struct PredictionArbitrageEvent {
    pub polymarket_id: String,
    pub polymarket_price: Decimal,
    pub external_platform: Platform,
    pub external_price: Decimal,
    pub price_difference: Decimal,          // Absolute difference
    pub edge_pct: Decimal,                  // Percentage edge
    pub arbitrage_type: ArbitrageType,
    pub recommended_side: Side,             // Buy/Sell on Polymarket
    pub confidence: Decimal,
    pub timestamp: DateTime<Utc>,
}

pub enum ArbitrageType {
    HardArbitrage,      // True risk-free arb (rare)
    SoftArbitrage,      // Edge suggesting mispricing
    Convergence,        // Expect prices to converge
}
```

### 4. Create Prediction Arbitrage Strategy

**File:** `crates/polysniper-strategies/src/prediction_arbitrage.rs`

```rust
pub struct PredictionArbitrageStrategy {
    config: Arc<RwLock<PredictionArbitrageConfig>>,
    signal_cooldowns: HashMap<String, DateTime<Utc>>,
    historical_spreads: HashMap<String, VecDeque<Decimal>>,
}

pub struct PredictionArbitrageConfig {
    pub enabled: bool,
    pub min_edge_pct: Decimal,              // Minimum edge to trigger (e.g., 5%)
    pub platform_weights: PlatformWeights,  // Trust weights per platform
    pub soft_arb_config: SoftArbConfig,
    pub convergence_config: ConvergenceConfig,
    pub cooldown_secs: u64,
}

pub struct PlatformWeights {
    pub metaculus: Decimal,                 // e.g., 1.2 (experts, high trust)
    pub predictit: Decimal,                 // e.g., 0.9 (retail, less liquidity)
    pub kalshi: Decimal,                    // e.g., 1.0 (regulated, good liquidity)
}

pub struct SoftArbConfig {
    pub enabled: bool,
    pub min_edge_pct: Decimal,              // e.g., 5%
    pub max_entry_price: Decimal,           // Don't buy above this
    pub order_size_usd: Decimal,
}

pub struct ConvergenceConfig {
    pub enabled: bool,
    pub min_spread_zscore: Decimal,         // Spread must be N std devs from mean
    pub lookback_periods: u32,              // Historical spread comparison
    pub order_size_usd: Decimal,
}
```

**Strategy Logic:**

1. **Platform Comparison:**
   - Weight external prices by platform reliability
   - Calculate consensus price from external sources
   - Compare with Polymarket price

2. **Edge Detection:**
   - If Polymarket < consensus: BUY opportunity
   - If Polymarket > consensus: SELL opportunity
   - Weight by confidence based on:
     - Number of platforms agreeing
     - Historical accuracy of platforms
     - Spread between external sources

3. **Convergence Trading:**
   - Track historical spread between Polymarket and consensus
   - When spread exceeds historical norms (z-score)
   - Bet on convergence (mean reversion)

### 5. Configuration

**File:** `config/strategies/prediction_arbitrage.toml`

```toml
[prediction_arbitrage]
enabled = true
min_edge_pct = 5.0
cooldown_secs = 1800

[prediction_arbitrage.platform_weights]
metaculus = 1.2       # Expert forecasters, higher trust
predictit = 0.9       # Retail traders, capped contracts
kalshi = 1.0          # Regulated, good liquidity

[prediction_arbitrage.soft_arb]
enabled = true
min_edge_pct = 5.0
max_entry_price = 0.90
order_size_usd = 50.0

[prediction_arbitrage.convergence]
enabled = true
min_spread_zscore = 2.0
lookback_periods = 50
order_size_usd = 25.0

# Market mappings
[[prediction_arbitrage.mappings]]
name = "2024 Presidential Election"
polymarket_id = "0x123..."
metaculus_id = "12345"
predictit_contract = 6867
kalshi_ticker = "PRES-24"
price_adjustment = 0.02  # Account for fees

[[prediction_arbitrage.mappings]]
name = "Fed Rate Decision"
polymarket_id = "0x456..."
metaculus_id = "23456"
kalshi_ticker = "FED-25JAN"
price_adjustment = 0.01

# External market API configuration
[external_markets.metaculus]
enabled = true
api_base_url = "https://www.metaculus.com/api2"
poll_interval_secs = 300

[external_markets.predictit]
enabled = true
api_base_url = "https://www.predictit.org/api"
poll_interval_secs = 60

[external_markets.kalshi]
enabled = true
api_base_url = "https://trading-api.kalshi.com/trade-api/v2"
poll_interval_secs = 30
email_env = "KALSHI_EMAIL"
password_env = "KALSHI_PASSWORD"
```

### 6. Market Matching Service

**File:** `crates/polysniper-data/src/market_matcher.rs`

Optional: Automatic matching of similar questions across platforms:

```rust
pub struct MarketMatcher {
    /// Use semantic similarity to find matching markets
    pub async fn find_matches(
        polymarket: &Market,
        external: &[ExternalMarketPrice],
    ) -> Vec<PotentialMatch> {
        // Use fuzzy string matching or embeddings
        // Return candidates for manual review
    }
}
```

### 7. Update Module Exports

- Add `external_markets` module to `crates/polysniper-data/src/lib.rs`
- Add `prediction_aggregator` to `crates/polysniper-data/src/lib.rs`
- Add `prediction_arbitrage` strategy to `crates/polysniper-strategies/src/lib.rs`
- Register strategy in main application

## Acceptance Criteria

- [ ] MetaculusClient fetches community predictions
- [ ] PredictItClient fetches market prices
- [ ] KalshiClient fetches market prices (with optional auth)
- [ ] PredictionAggregator coordinates polling across platforms
- [ ] Market mappings correctly link across platforms
- [ ] Edge detection accurately identifies discrepancies
- [ ] PredictionArbitrageDetected events are published
- [ ] Strategy generates appropriate signals based on edge
- [ ] Platform weighting affects signal confidence
- [ ] Convergence trading tracks historical spreads
- [ ] Cooldowns prevent signal spam
- [ ] All new code has unit tests
- [ ] Graceful handling when external APIs are unavailable
- [ ] Rate limiting respects external API limits

## Files to Create/Modify

**Create:**
- `crates/polysniper-data/src/external_markets/mod.rs`
- `crates/polysniper-data/src/external_markets/metaculus.rs`
- `crates/polysniper-data/src/external_markets/predictit.rs`
- `crates/polysniper-data/src/external_markets/kalshi.rs`
- `crates/polysniper-data/src/prediction_aggregator.rs`
- `crates/polysniper-strategies/src/prediction_arbitrage.rs`
- `config/strategies/prediction_arbitrage.toml`

**Modify:**
- `crates/polysniper-core/src/events.rs` - Add new event types
- `crates/polysniper-data/src/lib.rs` - Export modules
- `crates/polysniper-strategies/src/lib.rs` - Export strategy

## Integration Points

- **Provides**: External market prices, arbitrage signals, cross-platform edge detection
- **Consumes**: Polymarket prices from state, EventBus for publishing
- **Conflicts**: None - new module with clean separation

## Testing Requirements

```rust
#[cfg(test)]
mod tests {
    // Test Metaculus API parsing
    #[test]
    fn test_parse_metaculus_response() { ... }

    // Test PredictIt API parsing
    #[test]
    fn test_parse_predictit_response() { ... }

    // Test Kalshi API parsing
    #[test]
    fn test_parse_kalshi_response() { ... }

    // Test edge calculation
    #[test]
    fn test_edge_calculation() { ... }

    // Test weighted consensus
    #[test]
    fn test_weighted_consensus_price() { ... }

    // Test arbitrage signal generation
    #[tokio::test]
    async fn test_arbitrage_signal_on_edge() { ... }

    // Test convergence z-score
    #[test]
    fn test_convergence_zscore_calculation() { ... }

    // Test cooldown enforcement
    #[tokio::test]
    async fn test_arb_cooldown() { ... }
}
```

## Notes on External API Access

1. **Metaculus**: Public API, no auth required for reading predictions
2. **PredictIt**: Public API for market data, no auth for reading
3. **Kalshi**: Requires authentication for most endpoints
   - Store credentials in environment variables
   - Handle token refresh

4. **Rate Limits**:
   - Metaculus: Respect robots.txt, ~1 req/sec
   - PredictIt: ~60 req/min
   - Kalshi: See API docs for limits

5. **Data Lag**:
   - External markets may have delayed data
   - Factor this into edge calculations
