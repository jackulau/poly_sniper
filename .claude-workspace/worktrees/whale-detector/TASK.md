---
id: whale-detector
name: Whale Detection and Large Trade Tracking
wave: 1
priority: 1
dependencies: []
estimated_hours: 5
tags: [analysis, core, whale-tracking]
---

## Objective

Implement a whale detection system that identifies and tracks large trades and wallet addresses exhibiting significant trading activity, providing leading indicators for position management.

## Context

Whale tracking monitors large market participants whose trades can significantly impact prices. By detecting whale activity early, the system can:
1. Avoid trading against large informed participants
2. Potentially follow whale momentum
3. Reduce position sizes during high whale activity
4. Generate alerts for manual review

## Implementation

### 1. Create new module: `crates/polysniper-strategies/src/whale_detector.rs`

**Core Components:**

```rust
pub struct WhaleDetectorConfig {
    pub enabled: bool,
    pub whale_threshold_usd: Decimal,           // Single trade threshold (e.g., $5000)
    pub aggregate_threshold_usd: Decimal,       // Cumulative threshold (e.g., $10000)
    pub aggregate_window_secs: u64,             // Window for aggregation (e.g., 300s)
    pub alert_on_detection: bool,
    pub track_addresses: bool,                  // Enable address profiling
    pub address_history_limit: usize,           // Max trades per address to track
    pub cooldown_secs: u64,                     // Cooldown between same-address alerts
}

pub struct WhaleDetector {
    config: WhaleDetectorConfig,
    // Track recent large trades per token
    recent_whale_trades: HashMap<TokenId, VecDeque<WhaleTrade>>,
    // Track cumulative activity per address (if enabled)
    address_profiles: HashMap<String, AddressProfile>,
    // Cooldown tracking
    last_alerts: HashMap<String, DateTime<Utc>>,
}

pub struct WhaleTrade {
    pub trade_id: String,
    pub token_id: TokenId,
    pub market_id: MarketId,
    pub side: Side,
    pub size_usd: Decimal,
    pub price: Decimal,
    pub timestamp: DateTime<Utc>,
    pub address: Option<String>,            // Wallet address if available
    pub is_single_whale: bool,              // Met threshold in single trade
    pub cumulative_context: Option<CumulativeContext>,
}

pub struct CumulativeContext {
    pub total_volume_usd: Decimal,
    pub trade_count: u32,
    pub window_start: DateTime<Utc>,
    pub dominant_side: Side,
    pub side_ratio: Decimal,                // Buy volume / total volume
}

pub struct AddressProfile {
    pub address: String,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub total_volume_usd: Decimal,
    pub trade_count: u32,
    pub avg_trade_size_usd: Decimal,
    pub win_rate: Option<Decimal>,          // If we can track outcomes
    pub recent_trades: VecDeque<WhaleTrade>,
    pub classification: WhaleClassification,
}

pub enum WhaleClassification {
    Unknown,
    Accumulator,    // Consistently buys
    Distributor,    // Consistently sells
    Flipper,        // Quick in/out trades
    Informed,       // High win rate
}

pub struct WhaleAlert {
    pub alert_type: WhaleAlertType,
    pub token_id: TokenId,
    pub market_id: MarketId,
    pub whale_trade: WhaleTrade,
    pub recommended_action: WhaleAction,
    pub confidence: Decimal,
    pub timestamp: DateTime<Utc>,
}

pub enum WhaleAlertType {
    SingleLargeTrade,
    CumulativeActivity,
    KnownWhaleActive,
    WhaleReversal,      // Whale changes direction
}

pub enum WhaleAction {
    None,
    ReducePosition { multiplier: Decimal },
    HaltNewTrades,
    FollowWhale { direction: Side },
    Alert,
}
```

### 2. Implement detection logic

```rust
impl WhaleDetector {
    /// Process a trade and check for whale activity
    pub fn process_trade(
        &mut self,
        token_id: &TokenId,
        market_id: &MarketId,
        side: Side,
        size_usd: Decimal,
        price: Decimal,
        address: Option<&str>,
        timestamp: DateTime<Utc>,
    ) -> Option<WhaleAlert> {
        // Check single trade threshold
        if size_usd >= self.config.whale_threshold_usd {
            return self.create_single_whale_alert(...);
        }

        // Track trade and check cumulative
        self.track_trade(...);

        // Check cumulative threshold
        if let Some(cumulative) = self.check_cumulative_threshold(token_id) {
            return self.create_cumulative_alert(...);
        }

        // Check known whale activity
        if let Some(profile) = address.and_then(|a| self.address_profiles.get(a)) {
            if profile.classification == WhaleClassification::Informed {
                return self.create_known_whale_alert(...);
            }
        }

        None
    }

    /// Get cumulative whale activity for a token
    pub fn get_whale_activity(&self, token_id: &TokenId) -> Option<CumulativeContext> {
        let trades = self.recent_whale_trades.get(token_id)?;
        let now = Utc::now();
        let window_start = now - Duration::seconds(self.config.aggregate_window_secs as i64);

        let recent: Vec<_> = trades.iter()
            .filter(|t| t.timestamp >= window_start)
            .collect();

        // Calculate cumulative stats
    }

    /// Update address profile with new trade
    fn update_address_profile(&mut self, address: &str, trade: &WhaleTrade) {
        let profile = self.address_profiles
            .entry(address.to_string())
            .or_insert_with(|| AddressProfile::new(address));

        profile.add_trade(trade);
        profile.update_classification();
    }

    /// Classify an address based on trading patterns
    fn classify_address(profile: &AddressProfile) -> WhaleClassification {
        // Analyze trade history to determine classification
    }
}
```

### 3. Create configuration file: `config/strategies/whale_detector.toml`

```toml
[strategy]
enabled = true
id = "whale_detector"
name = "Whale Detection"

[strategy.whale]
whale_threshold_usd = "5000"
aggregate_threshold_usd = "10000"
aggregate_window_secs = 300
alert_on_detection = true
track_addresses = true
address_history_limit = 100
cooldown_secs = 60

# Actions to take on whale detection
[strategy.whale.actions]
reduce_position_on_whale = true
reduction_multiplier = "0.5"
halt_on_cumulative = false

# Known whale addresses (manual configuration)
[strategy.whale.known_addresses]
# "0xabc123..." = { classification = "Informed", notes = "Historical accuracy" }
```

### 4. Add tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> WhaleDetectorConfig { ... }

    #[test]
    fn test_single_whale_detection() {
        // Trade above threshold triggers alert
    }

    #[test]
    fn test_cumulative_whale_detection() {
        // Multiple smaller trades exceeding aggregate threshold
    }

    #[test]
    fn test_address_profiling() {
        // Address classification based on trading patterns
    }

    #[test]
    fn test_cooldown_behavior() {
        // No duplicate alerts within cooldown window
    }

    #[test]
    fn test_window_expiration() {
        // Old trades expire from cumulative calculation
    }

    #[tokio::test]
    async fn test_whale_detector_with_mock_state() { ... }
}
```

### 5. Export from strategies crate

Update `crates/polysniper-strategies/src/lib.rs` to export the whale_detector module.

## Acceptance Criteria

- [ ] Single large trade detection works at configured threshold
- [ ] Cumulative activity detection tracks trades within time window
- [ ] Address profiling correctly classifies trading patterns
- [ ] Cooldown prevents alert spam for same address/token
- [ ] Window expiration removes stale trades from calculations
- [ ] Configuration loads from TOML file
- [ ] All tests pass (`cargo test -p polysniper-strategies`)
- [ ] Code follows existing patterns
- [ ] No clippy warnings

## Files to Create/Modify

- `crates/polysniper-strategies/src/whale_detector.rs` - **CREATE** - Core whale detection
- `crates/polysniper-strategies/src/lib.rs` - **MODIFY** - Add `pub mod whale_detector;` export
- `config/strategies/whale_detector.toml` - **CREATE** - Configuration file

## Integration Points

- **Provides**: `WhaleDetector`, `WhaleAlert`, `WhaleAction` for use by strategy and events
- **Consumes**: Trade data (from `TradeExecutedEvent`, order fills) with address info if available
- **Conflicts**: None - standalone module
