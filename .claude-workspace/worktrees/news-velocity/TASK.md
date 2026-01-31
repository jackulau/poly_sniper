---
id: news-velocity
name: News Velocity Scoring (Rate-of-Change in Coverage)
wave: 1
priority: 3
dependencies: []
estimated_hours: 5
tags: [alpha, sentiment, news, velocity]
---

## Objective

Implement news velocity scoring that measures not just sentiment but the rate-of-change in news coverage for Polymarket topics, enabling early detection of emerging narratives before they're priced in.

## Context

Current sentiment analysis only captures point-in-time sentiment. However, the **velocity** of news coverage is often more predictive:
- Sudden increase in coverage → Breaking news → Price movement
- Accelerating coverage → Growing narrative momentum
- Declining coverage velocity → Story fading → Mean reversion opportunity

This task extends the existing feed aggregator to track coverage velocity and generate signals based on acceleration/deceleration patterns.

## Implementation

### 1. Create News Velocity Tracker

**File:** `crates/polysniper-data/src/news_velocity.rs`

```rust
pub struct NewsVelocityTracker {
    config: NewsVelocityConfig,
    article_history: HashMap<String, VecDeque<ArticleTimestamp>>,
    velocity_cache: HashMap<String, VelocityMetrics>,
    event_tx: broadcast::Sender<SystemEvent>,
}

pub struct NewsVelocityConfig {
    pub enabled: bool,
    pub tracking_keywords: Vec<KeywordTracking>,
    pub velocity_windows: Vec<Duration>,    // [1h, 6h, 24h]
    pub acceleration_threshold: Decimal,    // e.g., 2.0 = 2x baseline
    pub deceleration_threshold: Decimal,    // e.g., 0.5 = half baseline
    pub min_articles_for_signal: u32,       // Minimum coverage
    pub market_mappings: HashMap<String, Vec<String>>,  // keyword → market_ids
}

pub struct KeywordTracking {
    pub keyword: String,
    pub aliases: Vec<String>,               // e.g., ["Trump", "Donald Trump", "POTUS"]
    pub category: NewsCategory,
}

pub enum NewsCategory {
    Politics,
    Crypto,
    Sports,
    Finance,
    Technology,
    Other,
}

pub struct ArticleTimestamp {
    pub source: String,
    pub timestamp: DateTime<Utc>,
    pub title_hash: String,                 // For deduplication
    pub sentiment: Option<Decimal>,         // If sentiment analysis available
}

pub struct VelocityMetrics {
    pub keyword: String,
    pub current_velocity: Decimal,          // Articles per hour
    pub velocity_1h: Decimal,
    pub velocity_6h: Decimal,
    pub velocity_24h: Decimal,
    pub acceleration: Decimal,              // Rate of change
    pub baseline_velocity: Decimal,         // Historical average
    pub last_updated: DateTime<Utc>,
}
```

### 2. Velocity Calculation Logic

```rust
impl NewsVelocityTracker {
    /// Calculate velocity (articles per hour) for a time window
    fn calculate_velocity(&self, keyword: &str, window: Duration) -> Decimal {
        let history = self.article_history.get(keyword)?;
        let cutoff = Utc::now() - window;
        let count = history.iter().filter(|a| a.timestamp > cutoff).count();
        let hours = window.num_hours() as u64;
        Decimal::from(count) / Decimal::from(hours)
    }

    /// Calculate acceleration (change in velocity)
    fn calculate_acceleration(&self, keyword: &str) -> Decimal {
        let velocity_1h = self.calculate_velocity(keyword, Duration::hours(1));
        let velocity_6h = self.calculate_velocity(keyword, Duration::hours(6));

        if velocity_6h.is_zero() {
            return velocity_1h; // Avoid division by zero
        }

        // Acceleration = (recent velocity - older velocity) / older velocity
        (velocity_1h - velocity_6h) / velocity_6h
    }

    /// Process incoming feed item and update velocity tracking
    pub async fn process_feed_item(&mut self, item: &FeedItem) {
        for keyword in &self.config.tracking_keywords {
            if self.matches_keyword(item, keyword) {
                self.record_article(keyword, item);

                let metrics = self.update_metrics(&keyword.keyword);

                // Check for velocity signal
                if metrics.acceleration > self.config.acceleration_threshold {
                    self.emit_velocity_signal(keyword, &metrics, VelocityDirection::Accelerating);
                } else if metrics.acceleration < -self.config.deceleration_threshold.abs() {
                    self.emit_velocity_signal(keyword, &metrics, VelocityDirection::Decelerating);
                }
            }
        }
    }
}
```

### 3. Add SystemEvent Variant

**File:** `crates/polysniper-core/src/events.rs`

Add new event type:
```rust
NewsVelocitySignal(NewsVelocitySignalEvent),

pub struct NewsVelocitySignalEvent {
    pub keyword: String,
    pub market_ids: Vec<String>,            // Mapped markets
    pub direction: VelocityDirection,
    pub current_velocity: Decimal,          // Articles/hour
    pub baseline_velocity: Decimal,
    pub acceleration: Decimal,
    pub article_count_1h: u32,
    pub article_count_24h: u32,
    pub sample_headlines: Vec<String>,      // Recent headlines for context
    pub timestamp: DateTime<Utc>,
}

pub enum VelocityDirection {
    Accelerating,   // Coverage increasing rapidly
    Decelerating,   // Coverage decreasing
    Stable,         // Normal coverage
}
```

### 4. Create News Velocity Strategy

**File:** `crates/polysniper-strategies/src/news_velocity.rs`

```rust
pub struct NewsVelocityStrategy {
    config: Arc<RwLock<NewsVelocityStrategyConfig>>,
    signal_cooldowns: HashMap<String, DateTime<Utc>>,
}

pub struct NewsVelocityStrategyConfig {
    pub enabled: bool,
    pub acceleration_config: AccelerationConfig,
    pub deceleration_config: DecelerationConfig,
}

pub struct AccelerationConfig {
    pub enabled: bool,
    pub min_acceleration: Decimal,          // e.g., 2.0 = 2x baseline
    pub min_articles_1h: u32,               // Minimum recent articles
    pub signal_direction: Side,             // Usually Buy (momentum)
    pub order_size_usd: Decimal,
    pub max_entry_price: Decimal,           // Don't chase too high
    pub cooldown_secs: u64,
}

pub struct DecelerationConfig {
    pub enabled: bool,
    pub max_velocity_ratio: Decimal,        // e.g., 0.3 = 30% of baseline
    pub signal_direction: Side,             // Usually Sell (fading)
    pub order_size_usd: Decimal,
    pub min_position_profit_pct: Decimal,   // Only exit if profitable
    pub cooldown_secs: u64,
}
```

**Strategy Logic:**

1. **Acceleration Signals (Breaking News):**
   - When coverage velocity exceeds threshold
   - Generate BUY signal (momentum-following)
   - Weight by magnitude of acceleration
   - Early entry before full pricing

2. **Deceleration Signals (Fading Story):**
   - When coverage velocity drops significantly
   - Generate SELL signal if holding position
   - Mean reversion on overbought markets

### 5. Configuration

**File:** `config/strategies/news_velocity.toml`

```toml
[news_velocity]
enabled = true

[[news_velocity.tracking_keywords]]
keyword = "trump"
aliases = ["Donald Trump", "POTUS", "Trump administration"]
category = "Politics"

[[news_velocity.tracking_keywords]]
keyword = "bitcoin etf"
aliases = ["spot btc etf", "bitcoin spot etf"]
category = "Crypto"

[[news_velocity.tracking_keywords]]
keyword = "fed rate"
aliases = ["federal reserve", "interest rate", "rate cut", "rate hike"]
category = "Finance"

[news_velocity.velocity_windows]
short = 3600        # 1 hour
medium = 21600      # 6 hours
long = 86400        # 24 hours

[news_velocity.acceleration_config]
enabled = true
min_acceleration = 2.0
min_articles_1h = 5
signal_direction = "Buy"
order_size_usd = 50.0
max_entry_price = 0.85
cooldown_secs = 1800

[news_velocity.deceleration_config]
enabled = true
max_velocity_ratio = 0.3
signal_direction = "Sell"
order_size_usd = 0.0          # Close position only
min_position_profit_pct = 5.0
cooldown_secs = 3600

[news_velocity.market_mappings]
trump = ["0xabc123...", "0xdef456..."]
"bitcoin etf" = ["0x789ghi..."]
"fed rate" = ["0xjkl012..."]
```

### 6. Integration with Existing Feed Aggregator

Modify `crates/polysniper-data/src/feed_aggregator.rs` to:
- Pass items to NewsVelocityTracker before/after publishing
- Share deduplication cache with velocity tracker
- Coordinate keyword matching

### 7. Update Module Exports

- Add `news_velocity` to `crates/polysniper-data/src/lib.rs`
- Add `news_velocity` strategy to `crates/polysniper-strategies/src/lib.rs`
- Register strategy in main application

## Acceptance Criteria

- [ ] NewsVelocityTracker accurately calculates article velocity
- [ ] Acceleration/deceleration detection works correctly
- [ ] Keyword aliases are properly matched
- [ ] Market mappings link keywords to Polymarket markets
- [ ] NewsVelocitySignal events are published to EventBus
- [ ] Strategy generates appropriate signals based on velocity
- [ ] Cooldowns prevent signal spam
- [ ] Historical velocity baseline is calculated correctly
- [ ] All new code has unit tests
- [ ] Integration with existing feed aggregator works

## Files to Create/Modify

**Create:**
- `crates/polysniper-data/src/news_velocity.rs`
- `crates/polysniper-strategies/src/news_velocity.rs`
- `config/strategies/news_velocity.toml`

**Modify:**
- `crates/polysniper-core/src/events.rs` - Add NewsVelocitySignal event
- `crates/polysniper-data/src/feed_aggregator.rs` - Integration point
- `crates/polysniper-data/src/lib.rs` - Export news_velocity
- `crates/polysniper-strategies/src/lib.rs` - Export strategy

## Integration Points

- **Provides**: News velocity metrics, acceleration signals
- **Consumes**: FeedItemReceived events from feed aggregator
- **Conflicts**: Coordinates with sentiment analysis (avoid duplicate signals)

## Testing Requirements

```rust
#[cfg(test)]
mod tests {
    // Test velocity calculation
    #[test]
    fn test_velocity_calculation() { ... }

    // Test acceleration detection
    #[test]
    fn test_acceleration_detection() { ... }

    // Test keyword alias matching
    #[test]
    fn test_keyword_alias_matching() { ... }

    // Test signal generation on acceleration
    #[tokio::test]
    async fn test_acceleration_signal_generation() { ... }

    // Test deceleration signal
    #[tokio::test]
    async fn test_deceleration_signal() { ... }

    // Test cooldown enforcement
    #[tokio::test]
    async fn test_velocity_cooldown() { ... }
}
```
