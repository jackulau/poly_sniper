---
id: whale-tracking
name: Whale Order Tracking and Detection
wave: 1
priority: 2
dependencies: []
estimated_hours: 5
tags: [strategy, detection, alpha, leading-indicator]
---

## Objective

Implement whale tracking to detect large orders and accumulation patterns in the orderbook, using these as leading indicators for market direction.

## Context

Large traders ("whales") often have information advantages or market-moving power. Detecting their activity early provides valuable signals:
- Large resting orders indicate strong conviction at a price level
- Accumulation patterns (multiple large orders appearing) signal institutional interest
- Order book changes before major moves can predict direction

The codebase already has:
- `DepthAnalyzer` for orderbook depth analysis
- `QueueEstimator` for fill rate tracking
- Real-time orderbook updates via WebSocket
- Orderbook structure with bid/ask price levels

## Implementation

### 1. Create Whale Detector

**File:** `crates/polysniper-strategies/src/whale_detector.rs`

```rust
pub struct WhaleDetector {
    config: Arc<RwLock<WhaleConfig>>,
    orderbook_history: HashMap<TokenId, VecDeque<OrderbookSnapshot>>,
    detected_whales: HashMap<TokenId, Vec<WhaleActivity>>,
}

pub struct WhaleConfig {
    pub enabled: bool,
    pub min_order_size_usd: Decimal,         // Minimum to qualify as "whale" (e.g., $5000)
    pub min_relative_size_pct: Decimal,      // Minimum % of total depth (e.g., 10%)
    pub accumulation_window_secs: u64,       // Window to detect patterns (e.g., 300s)
    pub min_accumulation_orders: u32,        // Minimum orders for accumulation signal
    pub alert_cooldown_secs: u64,            // Prevent duplicate alerts
}

pub struct WhaleActivity {
    pub activity_type: WhaleActivityType,
    pub side: Side,                          // Buy or Sell
    pub total_size_usd: Decimal,
    pub num_orders: u32,
    pub avg_price: Decimal,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub confidence: Decimal,                 // 0.0 to 1.0
}

pub enum WhaleActivityType {
    LargeResting,        // Single large order detected
    Accumulation,        // Multiple large orders appearing
    Iceberg,             // Large order being worked (detected via fills)
    Spoofing,            // Large order placed then quickly cancelled
}
```

### 2. Create Whale Strategy

**File:** `crates/polysniper-strategies/src/whale_strategy.rs`

```rust
pub struct WhaleStrategy {
    detector: Arc<WhaleDetector>,
    config: Arc<RwLock<WhaleStrategyConfig>>,
    cooldowns: HashMap<String, DateTime<Utc>>,
}

pub struct WhaleStrategyConfig {
    pub enabled: bool,
    pub min_confidence: Decimal,             // Minimum confidence to generate signal
    pub follow_whales: bool,                 // Trade in same direction
    pub fade_spoofing: bool,                 // Trade against detected spoofing
    pub order_size_usd: Decimal,             // Size for whale-following trades
    pub signal_delay_secs: u64,              // Wait before acting (avoid front-running)
}
```

- Subscribe to OrderbookUpdate events
- Use WhaleDetector to identify whale activity
- Generate TradeSignal when criteria met
- Support both follow (same direction) and fade (opposite) strategies

### 3. Add Orderbook History Tracking

**Modify:** `crates/polysniper-data/src/market_cache.rs`

Add orderbook snapshot history:
```rust
pub struct OrderbookHistory {
    snapshots: VecDeque<OrderbookSnapshot>,
    max_snapshots: usize,
}

pub struct OrderbookSnapshot {
    pub timestamp: DateTime<Utc>,
    pub total_bid_size: Decimal,
    pub total_ask_size: Decimal,
    pub large_bids: Vec<LargeOrder>,
    pub large_asks: Vec<LargeOrder>,
}

pub struct LargeOrder {
    pub price: Decimal,
    pub size: Decimal,
    pub size_usd: Decimal,
}
```

### 4. Add SystemEvent for Whale Detection

**File:** `crates/polysniper-core/src/events.rs`

```rust
WhaleDetected(WhaleDetectedEvent),

pub struct WhaleDetectedEvent {
    pub token_id: TokenId,
    pub market_id: MarketId,
    pub activity: WhaleActivity,
    pub timestamp: DateTime<Utc>,
}
```

### 5. Configuration

**File:** `config/strategies/whale_tracking.toml`

```toml
[whale_detector]
enabled = true
min_order_size_usd = 5000.0
min_relative_size_pct = 10.0
accumulation_window_secs = 300
min_accumulation_orders = 3
alert_cooldown_secs = 600

[whale_strategy]
enabled = true
min_confidence = 0.7
follow_whales = true
fade_spoofing = false
order_size_usd = 100.0
signal_delay_secs = 30
```

## Acceptance Criteria

- [ ] WhaleDetector correctly identifies large orders (absolute and relative thresholds)
- [ ] Accumulation pattern detection works with configurable window
- [ ] WhaleStrategy generates signals based on detected activity
- [ ] Orderbook history tracking captures snapshots for analysis
- [ ] WhaleDetected events published to EventBus
- [ ] Configurable follow/fade strategy modes
- [ ] All new code has unit tests
- [ ] Integration with existing event bus
- [ ] No breaking changes to existing orderbook handling

## Files to Create/Modify

**Create:**
- `crates/polysniper-strategies/src/whale_detector.rs` - Detection logic
- `crates/polysniper-strategies/src/whale_strategy.rs` - Strategy implementation
- `config/strategies/whale_tracking.toml` - Configuration

**Modify:**
- `crates/polysniper-core/src/events.rs` - Add WhaleDetected event
- `crates/polysniper-data/src/market_cache.rs` - Add orderbook history
- `crates/polysniper-strategies/src/lib.rs` - Export new modules

## Integration Points

- **Provides**: WhaleDetected events, whale-following trade signals
- **Consumes**: OrderbookUpdate events, StateProvider for market data
- **Conflicts**: None - new module with clean separation

## Testing Requirements

```rust
#[cfg(test)]
mod tests {
    // Test large order detection
    #[test]
    fn test_detect_large_resting_order() { ... }

    // Test accumulation pattern
    #[tokio::test]
    async fn test_detect_accumulation() { ... }

    // Test strategy signal generation
    #[tokio::test]
    async fn test_whale_following_signal() { ... }

    // Test cooldown enforcement
    #[tokio::test]
    async fn test_cooldown_prevents_duplicate() { ... }
}
```
