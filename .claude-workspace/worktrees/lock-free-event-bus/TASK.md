---
id: lock-free-event-bus
name: Lock-Free Event Bus for Lower Latency
wave: 1
priority: 1
dependencies: []
estimated_hours: 5
tags: [performance, concurrency, event-bus]
---

## Objective

Replace the tokio broadcast channel-based event bus with a lock-free alternative to reduce latency for high-frequency event dispatching.

## Context

Current event bus in `crates/polysniper-data/src/event_bus.rs` uses `tokio::sync::broadcast` which:
- Has 10,000 message capacity (can cause lag under heavy load)
- Requires synchronization on every publish/subscribe
- Drops messages when subscribers lag (RecvError::Lagged)
- Single-threaded event processing in main loop

For trading systems, we need:
- Sub-microsecond publish latency
- No message drops (bounded backpressure instead)
- SPMC (single-producer, multi-consumer) pattern
- Predictable latency without lock contention

## Implementation

### 1. Add lock-free channel dependency

**File**: `Cargo.toml` (workspace)

```toml
[workspace.dependencies]
crossbeam-channel = "0.5"
```

### 2. Create lock-free event bus

**File**: `crates/polysniper-data/src/event_bus_fast.rs`

```rust
use crossbeam_channel::{bounded, Sender, Receiver, TrySendError};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Lock-free event bus using crossbeam channels
pub struct LockFreeEventBus {
    /// Senders for each subscriber (one-to-many via cloned senders not supported)
    subscribers: Arc<parking_lot::RwLock<Vec<Sender<SystemEvent>>>>,
    subscriber_count: AtomicUsize,
    /// Metrics
    dropped_count: AtomicUsize,
    publish_count: AtomicUsize,
}

/// Subscriber handle with bounded channel
pub struct EventSubscription {
    receiver: Receiver<SystemEvent>,
    id: usize,
}

impl LockFreeEventBus {
    /// Create new event bus
    pub fn new() -> Self;

    /// Subscribe with specified buffer capacity
    pub fn subscribe(&self, capacity: usize) -> EventSubscription;

    /// Publish event to all subscribers (non-blocking)
    /// Returns number of subscribers that received the event
    pub fn publish(&self, event: SystemEvent) -> usize;

    /// Try publish with timeout
    pub fn publish_timeout(&self, event: SystemEvent, timeout: Duration) -> usize;

    /// Get metrics
    pub fn metrics(&self) -> EventBusMetrics;
}

impl EventSubscription {
    /// Receive next event (blocking)
    pub fn recv(&self) -> Result<SystemEvent, RecvError>;

    /// Try receive (non-blocking)
    pub fn try_recv(&self) -> Option<SystemEvent>;

    /// Receive with timeout
    pub fn recv_timeout(&self, timeout: Duration) -> Result<SystemEvent, RecvTimeoutError>;

    /// Check pending message count
    pub fn pending(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct EventBusMetrics {
    pub total_published: usize,
    pub total_dropped: usize,
    pub subscriber_count: usize,
    pub avg_queue_depth: f64,
}
```

### 3. Implement backpressure strategy

```rust
/// Backpressure configuration
pub struct BackpressureConfig {
    /// Drop oldest messages when full (default: false, block instead)
    pub drop_oldest: bool,
    /// Warn when queue exceeds this threshold
    pub warn_threshold: usize,
    /// Maximum time to block on publish
    pub max_block_duration: Duration,
}
```

### 4. Add async wrapper for tokio compatibility

```rust
/// Async-compatible wrapper using tokio::sync::Notify
pub struct AsyncEventSubscription {
    inner: EventSubscription,
    notify: Arc<tokio::sync::Notify>,
}

impl AsyncEventSubscription {
    /// Async receive for use in tokio::select!
    pub async fn recv(&self) -> Result<SystemEvent, RecvError>;
}
```

### 5. Update main event loop

**File**: `src/main.rs`

```rust
// Replace tokio::select! with crossbeam_channel::select!
// Or use async wrapper for compatibility

let subscription = event_bus.subscribe(10_000);

loop {
    crossbeam_channel::select! {
        recv(shutdown_rx) -> _ => break,
        recv(subscription.receiver) -> event => {
            if let Ok(event) = event {
                self.process_event(event).await?;
            }
        }
    }
}
```

### 6. Add metrics integration

**File**: `crates/polysniper-observability/src/metrics.rs`

Add new metrics:
```rust
pub static EVENT_BUS_QUEUE_DEPTH: LazyLock<IntGaugeVec> = ...;
pub static EVENT_BUS_DROPPED: LazyLock<IntCounter> = ...;
pub static EVENT_BUS_PUBLISH_LATENCY: LazyLock<Histogram> = ...;
```

### 7. Implement feature flag for gradual rollout

```rust
/// EventBus trait implementation for both backends
pub enum EventBusImpl {
    Broadcast(BroadcastEventBus),
    LockFree(LockFreeEventBus),
}

impl EventBus for EventBusImpl {
    // Delegate to underlying implementation
}
```

## Acceptance Criteria

- [ ] LockFreeEventBus with crossbeam channels
- [ ] Bounded backpressure (no silent message drops)
- [ ] Per-subscriber queue depth metrics
- [ ] Async wrapper for tokio::select! compatibility
- [ ] Feature flag to switch between implementations
- [ ] Benchmark showing 10x+ latency improvement for publish
- [ ] All existing tests pass
- [ ] Integration with existing metrics infrastructure

## Files to Create/Modify

- `crates/polysniper-data/src/event_bus_fast.rs` - **CREATE** - Lock-free implementation
- `crates/polysniper-data/src/event_bus.rs` - Add trait abstraction, keep broadcast impl
- `crates/polysniper-data/src/lib.rs` - Export new module
- `crates/polysniper-data/Cargo.toml` - Add crossbeam-channel dependency
- `src/main.rs` - Update event loop to use new bus
- `crates/polysniper-observability/src/metrics.rs` - Add queue depth metrics
- `crates/polysniper-data/benches/event_bus_bench.rs` - **CREATE** - Benchmarks

## Integration Points

- **Provides**: Low-latency event dispatch for all system components
- **Consumes**: Events from WebSocket manager, strategies, fill manager
- **Conflicts**: None - backward compatible via trait abstraction

## Technical Notes

1. crossbeam-channel is wait-free for single producer scenarios
2. Consider `flume` as alternative (pure Rust, similar performance)
3. Use `parking_lot::RwLock` for subscriber list (faster than std)
4. Pre-allocate subscriber vector capacity
5. Consider SPMC ring buffer for truly zero-allocation publish
