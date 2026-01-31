---
id: batch-signal-processing
name: Batch Signal Processing for Market Updates
wave: 1
priority: 1
dependencies: []
estimated_hours: 5
tags: [performance, signal-processing, throughput]
---

## Objective

Implement batch processing for market updates to reduce per-event overhead and improve throughput during high-frequency update periods.

## Context

Current event processing in `src/main.rs` handles events one at a time:
- Each event triggers full strategy evaluation loop
- Sequential signal collection via `signals.extend()`
- Risk validation per signal, not batched
- Database persistence per trade

During high-activity periods (market opens, news events), this creates bottlenecks:
- Event bus lag warnings (10K buffer fills)
- Strategy processing can't keep up with update rate
- Latency spikes from sequential processing

## Implementation

### 1. Create batch processor module

**File**: `crates/polysniper-core/src/batch_processor.rs`

```rust
use std::time::{Duration, Instant};

/// Configuration for batch processing
#[derive(Clone)]
pub struct BatchConfig {
    /// Maximum events per batch
    pub max_batch_size: usize,
    /// Maximum time to wait for batch to fill
    pub max_batch_duration: Duration,
    /// Minimum events before processing (avoid single-event batches)
    pub min_batch_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            max_batch_duration: Duration::from_millis(10),
            min_batch_size: 1,
        }
    }
}

/// Batch of events for processing
pub struct EventBatch {
    events: Vec<SystemEvent>,
    created_at: Instant,
    /// Deduplicated token IDs that have updates
    affected_tokens: HashSet<TokenId>,
}

impl EventBatch {
    pub fn new() -> Self;

    /// Add event to batch, returns true if batch is ready
    pub fn push(&mut self, event: SystemEvent, config: &BatchConfig) -> bool;

    /// Check if batch should be flushed
    pub fn should_flush(&self, config: &BatchConfig) -> bool;

    /// Get events grouped by type for efficient processing
    pub fn by_type(&self) -> EventsByType;

    /// Drain events
    pub fn drain(&mut self) -> Vec<SystemEvent>;
}

/// Events grouped by type for batch processing
pub struct EventsByType {
    pub price_changes: Vec<PriceChangeEvent>,
    pub orderbook_updates: Vec<OrderbookUpdateEvent>,
    pub other: Vec<SystemEvent>,
}
```

### 2. Implement batch-aware strategy processing

**File**: `crates/polysniper-core/src/traits.rs`

Add batch processing trait method:

```rust
#[async_trait]
pub trait Strategy: Send + Sync {
    // Existing method
    async fn process_event(&self, event: &SystemEvent, state: &dyn TradingState)
        -> Result<Vec<TradeSignal>>;

    // NEW: Batch processing (default falls back to single-event)
    async fn process_batch(&self, events: &[SystemEvent], state: &dyn TradingState)
        -> Result<Vec<TradeSignal>> {
        let mut signals = Vec::new();
        for event in events {
            if self.accepts_event(event) {
                signals.extend(self.process_event(event, state).await?);
            }
        }
        Ok(signals)
    }

    /// Whether strategy supports optimized batch processing
    fn supports_batch(&self) -> bool {
        false
    }
}
```

### 3. Optimize key strategies for batch processing

**File**: `crates/polysniper-strategies/src/orderbook_imbalance.rs`

```rust
impl Strategy for OrderbookImbalanceStrategy {
    fn supports_batch(&self) -> bool {
        true
    }

    async fn process_batch(&self, events: &[SystemEvent], state: &dyn TradingState)
        -> Result<Vec<TradeSignal>> {
        // Group orderbook updates by token
        let updates_by_token = self.group_by_token(events);

        // Process only latest update per token (skip intermediate updates)
        let mut signals = Vec::with_capacity(updates_by_token.len());

        for (token_id, updates) in updates_by_token {
            // Only process the most recent orderbook for each token
            if let Some(latest) = updates.last() {
                if let Some(signal) = self.check_imbalance(latest).await? {
                    signals.push(signal);
                }
            }
        }

        Ok(signals)
    }
}
```

**File**: `crates/polysniper-strategies/src/price_spike.rs`

```rust
impl Strategy for PriceSpikeStrategy {
    fn supports_batch(&self) -> bool {
        true
    }

    async fn process_batch(&self, events: &[SystemEvent], state: &dyn TradingState)
        -> Result<Vec<TradeSignal>> {
        // Batch update price history for all tokens
        {
            let mut history = self.price_history.write().await;
            for event in events {
                if let SystemEvent::PriceChange(e) = event {
                    self.update_history_entry(&mut history, &e.token_id, e.price);
                }
            }
        }

        // Check for spikes on affected tokens only
        let affected_tokens: HashSet<_> = events
            .iter()
            .filter_map(|e| match e {
                SystemEvent::PriceChange(e) => Some(e.token_id.clone()),
                _ => None,
            })
            .collect();

        let mut signals = Vec::new();
        for token_id in affected_tokens {
            if let Some(signal) = self.check_spike(&token_id).await? {
                signals.push(signal);
            }
        }

        Ok(signals)
    }
}
```

### 4. Update main event loop for batching

**File**: `src/main.rs`

```rust
async fn run_event_loop(&mut self) -> Result<()> {
    let mut batch = EventBatch::new();
    let batch_config = BatchConfig::default();
    let mut flush_interval = tokio::time::interval(batch_config.max_batch_duration);

    loop {
        tokio::select! {
            biased;  // Prioritize shutdown

            _ = shutdown_rx.recv() => break,

            // Timer-based flush
            _ = flush_interval.tick() => {
                if !batch.is_empty() {
                    self.process_batch(batch.drain()).await?;
                }
            }

            // Event arrival
            event = event_rx.recv() => {
                match event {
                    Ok(event) => {
                        if batch.push(event, &batch_config) {
                            // Batch full, process immediately
                            self.process_batch(batch.drain()).await?;
                        }
                    }
                    Err(RecvError::Lagged(n)) => {
                        warn!("Event bus lagged by {} messages", n);
                    }
                    Err(RecvError::Closed) => break,
                }
            }
        }
    }

    Ok(())
}

async fn process_batch(&mut self, events: Vec<SystemEvent>) -> Result<()> {
    let batch_start = Instant::now();
    let batch_size = events.len();

    // Update state for all events first
    for event in &events {
        self.update_state(event).await;
    }

    // Collect signals from all strategies
    let mut all_signals = Vec::new();

    for strategy in &self.strategies {
        if !strategy.is_enabled() {
            continue;
        }

        let strategy_start = Instant::now();

        let signals = if strategy.supports_batch() {
            strategy.process_batch(&events, self.state.as_ref()).await?
        } else {
            // Fallback to single-event processing
            let mut signals = Vec::new();
            for event in &events {
                if strategy.accepts_event(event) {
                    signals.extend(strategy.process_event(event, self.state.as_ref()).await?);
                }
            }
            signals
        };

        record_strategy_processing(strategy.id(), strategy_start.elapsed().as_secs_f64());
        all_signals.extend(signals);
    }

    // Process collected signals
    if !all_signals.is_empty() {
        self.process_signals(all_signals).await?;
    }

    // Record batch metrics
    record_batch_processing(batch_size, batch_start.elapsed().as_secs_f64());

    Ok(())
}
```

### 5. Add batch metrics

**File**: `crates/polysniper-observability/src/metrics.rs`

```rust
pub static BATCH_SIZE: LazyLock<Histogram> = LazyLock::new(|| {
    register_histogram!(
        "batch_size",
        "Number of events per batch",
        vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    ).unwrap()
});

pub static BATCH_PROCESSING_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    register_histogram!(
        "batch_processing_duration_seconds",
        "Time to process a batch of events"
    ).unwrap()
});
```

## Acceptance Criteria

- [ ] BatchConfig with configurable batch size and duration
- [ ] EventBatch with deduplication by token
- [ ] Strategy trait extended with `process_batch()` method
- [ ] OrderbookImbalanceStrategy optimized for batch (skip intermediate)
- [ ] PriceSpikeStrategy optimized for batch (single history update)
- [ ] Main event loop uses batching with timer-based flush
- [ ] Batch metrics (size histogram, processing duration)
- [ ] All existing tests pass
- [ ] Benchmark showing 3x+ throughput improvement under load

## Files to Create/Modify

- `crates/polysniper-core/src/batch_processor.rs` - **CREATE** - Batch types and logic
- `crates/polysniper-core/src/lib.rs` - Export batch_processor
- `crates/polysniper-core/src/traits.rs` - Add process_batch to Strategy trait
- `crates/polysniper-strategies/src/orderbook_imbalance.rs` - Implement batch processing
- `crates/polysniper-strategies/src/price_spike.rs` - Implement batch processing
- `src/main.rs` - Batch event loop
- `crates/polysniper-observability/src/metrics.rs` - Batch metrics

## Integration Points

- **Provides**: Higher throughput event processing
- **Consumes**: Events from event bus
- **Conflicts**: Minimal - main.rs event loop changes only

## Technical Notes

1. Batching is most effective for price updates (many per second)
2. Heartbeat events should bypass batching (time-sensitive)
3. Consider per-strategy batch config (some need every update)
4. Deduplication key should include market_id for multi-market tokens
5. Monitor batch_size histogram to tune configuration
