---
id: polymarket-signals
name: Polymarket-Specific Signals (Trader Leaderboards, Activity, Volume)
wave: 1
priority: 2
dependencies: []
estimated_hours: 6
tags: [alpha, polymarket, signals, smart-money]
---

## Objective

Implement Polymarket-specific signal generation by tracking trader leaderboards, comment activity, and volume patterns to identify smart money flow and market sentiment shifts.

## Context

Polymarket has unique on-platform signals that can provide alpha:
- **Trader leaderboards**: Track top performers and their positions
- **Comment activity**: Spikes in comments may precede price moves
- **Volume patterns**: Unusual volume can signal incoming news/resolution
- **Smart money tracking**: Follow profitable traders' positions

This task adds data sources and strategies to capture these Polymarket-specific signals.

## Implementation

### 1. Create Polymarket Activity Client

**File:** `crates/polysniper-data/src/polymarket_activity.rs`

```rust
pub struct PolymarketActivityClient {
    http_client: reqwest::Client,
    event_tx: broadcast::Sender<SystemEvent>,
    config: PolymarketActivityConfig,
    trader_cache: Arc<RwLock<HashMap<String, TraderProfile>>>,
    volume_history: Arc<RwLock<HashMap<String, VecDeque<VolumeSnapshot>>>>,
}

pub struct PolymarketActivityConfig {
    pub enabled: bool,
    pub poll_interval_secs: u64,          // Default: 60
    pub track_top_traders: u32,           // Number of top traders to track
    pub volume_lookback_periods: u32,     // For detecting unusual volume
    pub comment_activity_threshold: u32,  // Min comments to trigger signal
}

pub struct TraderProfile {
    pub address: String,
    pub username: Option<String>,
    pub profit_pnl: Decimal,
    pub volume_traded: Decimal,
    pub win_rate: Decimal,
    pub recent_positions: Vec<TraderPosition>,
    pub last_updated: DateTime<Utc>,
}

pub struct TraderPosition {
    pub market_id: String,
    pub token_id: String,
    pub outcome: Outcome,
    pub size: Decimal,
    pub avg_price: Decimal,
    pub timestamp: DateTime<Utc>,
}

pub struct VolumeSnapshot {
    pub market_id: String,
    pub volume_usd: Decimal,
    pub trade_count: u32,
    pub timestamp: DateTime<Utc>,
}
```

### 2. Add SystemEvent Variants

**File:** `crates/polysniper-core/src/events.rs`

Add new event types:
```rust
SmartMoneySignal(SmartMoneySignalEvent),
VolumeAnomalyDetected(VolumeAnomalyEvent),
CommentActivitySpike(CommentActivityEvent),

pub struct SmartMoneySignalEvent {
    pub market_id: String,
    pub trader_address: String,
    pub trader_rank: u32,           // Leaderboard position
    pub trader_profit: Decimal,     // Total PnL
    pub action: TraderAction,       // Buy/Sell
    pub outcome: Outcome,           // Yes/No
    pub size_usd: Decimal,
    pub timestamp: DateTime<Utc>,
}

pub enum TraderAction {
    Buy,
    Sell,
    NewPosition,
    ClosePosition,
}

pub struct VolumeAnomalyEvent {
    pub market_id: String,
    pub current_volume: Decimal,
    pub avg_volume: Decimal,
    pub volume_ratio: Decimal,      // current / avg
    pub trade_count: u32,
    pub timestamp: DateTime<Utc>,
}

pub struct CommentActivityEvent {
    pub market_id: String,
    pub comment_count: u32,
    pub comment_velocity: Decimal,  // comments per hour
    pub sentiment_hint: Option<String>,  // Brief sentiment from comments
    pub timestamp: DateTime<Utc>,
}
```

### 3. Create Polymarket Activity Strategy

**File:** `crates/polysniper-strategies/src/polymarket_activity.rs`

```rust
pub struct PolymarketActivityStrategy {
    config: Arc<RwLock<PolymarketActivityStrategyConfig>>,
    signal_history: HashMap<String, DateTime<Utc>>,
    smart_money_positions: HashMap<String, Vec<TraderPosition>>,
}

pub struct PolymarketActivityStrategyConfig {
    pub enabled: bool,
    pub smart_money_tracking: SmartMoneyConfig,
    pub volume_anomaly: VolumeAnomalyConfig,
    pub comment_activity: CommentActivityConfig,
}

pub struct SmartMoneyConfig {
    pub enabled: bool,
    pub min_trader_rank: u32,           // Only track top N traders
    pub min_position_size_usd: Decimal, // Minimum position to trigger
    pub follow_strength: Decimal,       // 0.0-1.0, how much to follow
    pub cooldown_secs: u64,
    pub order_size_usd: Decimal,
}

pub struct VolumeAnomalyConfig {
    pub enabled: bool,
    pub min_volume_ratio: Decimal,      // e.g., 3.0 = 3x normal volume
    pub lookback_periods: u32,          // Number of periods for avg
    pub cooldown_secs: u64,
    pub order_size_usd: Decimal,
}

pub struct CommentActivityConfig {
    pub enabled: bool,
    pub min_comments_per_hour: u32,
    pub sentiment_weight: Decimal,      // How much to weight sentiment
    pub cooldown_secs: u64,
    pub order_size_usd: Decimal,
}
```

**Strategy Logic:**

1. **Smart Money Following:**
   - Monitor top trader positions
   - When top trader takes significant position, generate follow signal
   - Weight by trader profitability and position size

2. **Volume Anomaly Detection:**
   - Track rolling average volume per market
   - Detect when current volume exceeds threshold (e.g., 3x average)
   - Generate signal in direction of volume flow

3. **Comment Activity:**
   - Monitor comment velocity per market
   - Spike in comments may indicate imminent news/resolution
   - Optionally analyze comment sentiment

### 4. Configuration

**File:** `config/strategies/polymarket_activity.toml`

```toml
[polymarket_activity]
enabled = true

[polymarket_activity.smart_money_tracking]
enabled = true
min_trader_rank = 100           # Track top 100 traders
min_position_size_usd = 1000.0
follow_strength = 0.5
cooldown_secs = 600
order_size_usd = 25.0

[polymarket_activity.volume_anomaly]
enabled = true
min_volume_ratio = 3.0
lookback_periods = 24
cooldown_secs = 300
order_size_usd = 50.0

[polymarket_activity.comment_activity]
enabled = false                 # Requires API access
min_comments_per_hour = 10
sentiment_weight = 0.3
cooldown_secs = 600
order_size_usd = 25.0
```

### 5. Polymarket API Integration Notes

The Polymarket API endpoints to use:
- `/leaderboard` - Top trader rankings
- `/markets/{id}/activity` - Market-specific activity
- `/markets/{id}/comments` - Comment data (if available)
- `/users/{address}/positions` - Individual trader positions

If official API doesn't expose these, consider:
- Subgraph queries for on-chain position data
- Web scraping (with appropriate rate limiting)
- Third-party data providers

### 6. Update Module Exports

- Add `polymarket_activity` to `crates/polysniper-data/src/lib.rs`
- Add `polymarket_activity` strategy to `crates/polysniper-strategies/src/lib.rs`
- Register strategy in main application

## Acceptance Criteria

- [ ] PolymarketActivityClient fetches trader leaderboard data
- [ ] Trader positions are tracked and updated
- [ ] Volume anomaly detection works with configurable thresholds
- [ ] SmartMoneySignal events are published correctly
- [ ] Strategy generates appropriate follow signals
- [ ] Cooldown mechanisms prevent signal spam
- [ ] All new code has unit tests
- [ ] Graceful handling when API data is unavailable
- [ ] No breaking changes to existing code

## Files to Create/Modify

**Create:**
- `crates/polysniper-data/src/polymarket_activity.rs`
- `crates/polysniper-strategies/src/polymarket_activity.rs`
- `config/strategies/polymarket_activity.toml`

**Modify:**
- `crates/polysniper-core/src/events.rs` - Add new event types
- `crates/polysniper-data/src/lib.rs` - Export polymarket_activity
- `crates/polysniper-strategies/src/lib.rs` - Export strategy

## Integration Points

- **Provides**: Smart money signals, volume anomaly events, activity-based trade signals
- **Consumes**: EventBus for publishing, Gamma client for market data
- **Conflicts**: None - new module with clean separation

## Testing Requirements

```rust
#[cfg(test)]
mod tests {
    // Test trader profile parsing
    #[test]
    fn test_parse_trader_profile() { ... }

    // Test volume anomaly detection
    #[tokio::test]
    async fn test_volume_anomaly_detection() { ... }

    // Test smart money signal generation
    #[tokio::test]
    async fn test_smart_money_follow_signal() { ... }

    // Test cooldown enforcement
    #[tokio::test]
    async fn test_cooldown_prevents_spam() { ... }
}
```
