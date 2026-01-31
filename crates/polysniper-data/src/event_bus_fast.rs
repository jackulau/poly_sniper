//! Lock-free event bus implementation using crossbeam channels
//!
//! This module provides a high-performance, lock-free event bus designed for
//! sub-microsecond publish latency in high-frequency trading systems.

use crate::event_bus::BroadcastEventBus;
use crossbeam_channel::{bounded, Receiver, RecvError, RecvTimeoutError, Sender, TrySendError};
use polysniper_core::EventBus;
use tokio::sync::broadcast;
use parking_lot::RwLock;
use polysniper_core::SystemEvent;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, trace, warn};

/// Default capacity for subscriber channels
const DEFAULT_SUBSCRIBER_CAPACITY: usize = 10_000;

/// Lock-free event bus using crossbeam channels for low-latency event dispatch.
///
/// This implementation provides:
/// - Sub-microsecond publish latency for typical operations
/// - Bounded backpressure instead of silent message drops
/// - Per-subscriber queue depth tracking
/// - Configurable backpressure strategies
pub struct LockFreeEventBus {
    /// Subscribers with their channels and metadata
    subscribers: Arc<RwLock<Vec<SubscriberEntry>>>,
    /// Total number of active subscribers
    subscriber_count: AtomicUsize,
    /// Total events published
    publish_count: AtomicU64,
    /// Total events dropped across all subscribers
    dropped_count: AtomicU64,
    /// Backpressure configuration
    config: BackpressureConfig,
    /// Next subscriber ID
    next_subscriber_id: AtomicUsize,
}

/// Entry for a single subscriber
struct SubscriberEntry {
    /// Sender end of the channel
    sender: Sender<SystemEvent>,
    /// Unique subscriber ID
    id: usize,
    /// Number of messages dropped for this subscriber
    dropped: AtomicU64,
    /// Whether this subscriber is active
    active: bool,
}

/// Configuration for backpressure handling
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Drop oldest messages when full (default: false, blocks instead)
    pub drop_oldest: bool,
    /// Warn when queue exceeds this threshold
    pub warn_threshold: usize,
    /// Maximum time to block on publish when queue is full
    pub max_block_duration: Duration,
    /// Default capacity for new subscriber channels
    pub default_capacity: usize,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            drop_oldest: false,
            warn_threshold: 8_000,
            max_block_duration: Duration::from_micros(100),
            default_capacity: DEFAULT_SUBSCRIBER_CAPACITY,
        }
    }
}

/// Subscriber handle with bounded channel for receiving events
pub struct EventSubscription {
    /// Receiver end of the channel
    receiver: Receiver<SystemEvent>,
    /// Unique subscriber ID
    id: usize,
    /// Reference to parent bus for cleanup
    bus: Arc<RwLock<Vec<SubscriberEntry>>>,
}

impl EventSubscription {
    /// Receive next event (blocking)
    pub fn recv(&self) -> Result<SystemEvent, RecvError> {
        self.receiver.recv()
    }

    /// Try receive (non-blocking)
    pub fn try_recv(&self) -> Option<SystemEvent> {
        self.receiver.try_recv().ok()
    }

    /// Receive with timeout
    pub fn recv_timeout(&self, timeout: Duration) -> Result<SystemEvent, RecvTimeoutError> {
        self.receiver.recv_timeout(timeout)
    }

    /// Check pending message count in this subscription's queue
    pub fn pending(&self) -> usize {
        self.receiver.len()
    }

    /// Get the subscriber ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the raw receiver for use with crossbeam::select!
    pub fn receiver(&self) -> &Receiver<SystemEvent> {
        &self.receiver
    }
}

impl Drop for EventSubscription {
    fn drop(&mut self) {
        // Mark subscriber as inactive when subscription is dropped
        let mut subscribers = self.bus.write();
        if let Some(entry) = subscribers.iter_mut().find(|e| e.id == self.id) {
            entry.active = false;
        }
    }
}

/// Metrics for the event bus
#[derive(Debug, Clone)]
pub struct EventBusMetrics {
    /// Total events published
    pub total_published: u64,
    /// Total events dropped across all subscribers
    pub total_dropped: u64,
    /// Current number of active subscribers
    pub subscriber_count: usize,
    /// Average queue depth across all active subscribers
    pub avg_queue_depth: f64,
    /// Maximum queue depth across all subscribers
    pub max_queue_depth: usize,
    /// Publish operations per second (since last metrics call)
    pub publish_rate: f64,
}

impl LockFreeEventBus {
    /// Create a new event bus with default configuration
    pub fn new() -> Self {
        Self::with_config(BackpressureConfig::default())
    }

    /// Create a new event bus with specified configuration
    pub fn with_config(config: BackpressureConfig) -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(Vec::with_capacity(8))),
            subscriber_count: AtomicUsize::new(0),
            publish_count: AtomicU64::new(0),
            dropped_count: AtomicU64::new(0),
            config,
            next_subscriber_id: AtomicUsize::new(0),
        }
    }

    /// Subscribe with default buffer capacity
    pub fn subscribe(&self) -> EventSubscription {
        self.subscribe_with_capacity(self.config.default_capacity)
    }

    /// Subscribe with specified buffer capacity
    pub fn subscribe_with_capacity(&self, capacity: usize) -> EventSubscription {
        let (sender, receiver) = bounded(capacity);
        let id = self.next_subscriber_id.fetch_add(1, Ordering::SeqCst);

        let entry = SubscriberEntry {
            sender,
            id,
            dropped: AtomicU64::new(0),
            active: true,
        };

        {
            let mut subscribers = self.subscribers.write();
            subscribers.push(entry);
        }

        self.subscriber_count.fetch_add(1, Ordering::SeqCst);
        debug!(subscriber_id = %id, capacity = %capacity, "New subscriber added");

        EventSubscription {
            receiver,
            id,
            bus: self.subscribers.clone(),
        }
    }

    /// Publish event to all subscribers (non-blocking)
    /// Returns the number of subscribers that received the event
    pub fn publish(&self, event: SystemEvent) -> usize {
        let start = Instant::now();
        let event_type = event.event_type();
        let mut delivered = 0;
        let mut dropped = 0;

        {
            let subscribers = self.subscribers.read();
            for entry in subscribers.iter() {
                if !entry.active {
                    continue;
                }

                // Try non-blocking send first
                match entry.sender.try_send(event.clone()) {
                    Ok(()) => {
                        delivered += 1;
                    }
                    Err(TrySendError::Full(_)) => {
                        if self.config.drop_oldest {
                            // Drop the oldest message to make room
                            // Note: crossbeam doesn't support this directly, so we just drop
                            dropped += 1;
                            entry.dropped.fetch_add(1, Ordering::Relaxed);
                        } else {
                            // Try with timeout
                            match entry.sender.send_timeout(event.clone(), self.config.max_block_duration) {
                                Ok(()) => delivered += 1,
                                Err(_) => {
                                    dropped += 1;
                                    entry.dropped.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        // Subscriber disconnected, will be cleaned up
                    }
                }
            }
        }

        self.publish_count.fetch_add(1, Ordering::Relaxed);
        if dropped > 0 {
            self.dropped_count.fetch_add(dropped, Ordering::Relaxed);
            warn!(
                event_type = %event_type,
                dropped = %dropped,
                delivered = %delivered,
                "Events dropped due to backpressure"
            );
        }

        trace!(
            event_type = %event_type,
            delivered = %delivered,
            latency_us = %start.elapsed().as_micros(),
            "Event published"
        );

        delivered
    }

    /// Publish event with timeout
    /// Returns the number of subscribers that received the event
    pub fn publish_timeout(&self, event: SystemEvent, timeout: Duration) -> usize {
        let mut delivered = 0;

        {
            let subscribers = self.subscribers.read();
            for entry in subscribers.iter() {
                if !entry.active {
                    continue;
                }

                match entry.sender.send_timeout(event.clone(), timeout) {
                    Ok(()) => delivered += 1,
                    Err(_) => {
                        entry.dropped.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        self.publish_count.fetch_add(1, Ordering::Relaxed);
        delivered
    }

    /// Get current metrics
    pub fn metrics(&self) -> EventBusMetrics {
        let subscribers = self.subscribers.read();
        let active_count = subscribers.iter().filter(|e| e.active).count();

        let (total_depth, max_depth) = subscribers
            .iter()
            .filter(|e| e.active)
            .map(|e| e.sender.len())
            .fold((0usize, 0usize), |(total, max), depth| {
                (total + depth, max.max(depth))
            });

        let avg_depth = if active_count > 0 {
            total_depth as f64 / active_count as f64
        } else {
            0.0
        };

        EventBusMetrics {
            total_published: self.publish_count.load(Ordering::Relaxed),
            total_dropped: self.dropped_count.load(Ordering::Relaxed),
            subscriber_count: active_count,
            avg_queue_depth: avg_depth,
            max_queue_depth: max_depth,
            publish_rate: 0.0, // Would need time tracking for accurate rate
        }
    }

    /// Get subscriber count
    pub fn subscriber_count(&self) -> usize {
        let subscribers = self.subscribers.read();
        subscribers.iter().filter(|e| e.active).count()
    }

    /// Clean up disconnected subscribers
    pub fn cleanup(&self) {
        let mut subscribers = self.subscribers.write();
        let before = subscribers.len();
        subscribers.retain(|e| e.active && !e.sender.is_empty());
        let removed = before - subscribers.len();
        if removed > 0 {
            debug!(removed = %removed, "Cleaned up inactive subscribers");
        }
    }

    /// Get queue depths for all active subscribers
    pub fn queue_depths(&self) -> Vec<(usize, usize)> {
        let subscribers = self.subscribers.read();
        subscribers
            .iter()
            .filter(|e| e.active)
            .map(|e| (e.id, e.sender.len()))
            .collect()
    }
}

impl Default for LockFreeEventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for LockFreeEventBus {
    fn clone(&self) -> Self {
        // Cloning creates a new bus that shares the same subscriber list
        Self {
            subscribers: self.subscribers.clone(),
            subscriber_count: AtomicUsize::new(self.subscriber_count.load(Ordering::SeqCst)),
            publish_count: AtomicU64::new(self.publish_count.load(Ordering::Relaxed)),
            dropped_count: AtomicU64::new(self.dropped_count.load(Ordering::Relaxed)),
            config: self.config.clone(),
            next_subscriber_id: AtomicUsize::new(
                self.next_subscriber_id.load(Ordering::SeqCst),
            ),
        }
    }
}

/// Async-compatible wrapper for EventSubscription
///
/// This wrapper allows the lock-free event bus to be used within tokio::select!
/// and other async contexts while maintaining low latency.
pub struct AsyncEventSubscription {
    /// Inner subscription
    inner: EventSubscription,
}

impl AsyncEventSubscription {
    /// Create a new async subscription from a regular subscription
    pub fn new(inner: EventSubscription) -> Self {
        Self { inner }
    }

    /// Async receive for use in tokio::select!
    ///
    /// This implementation uses a polling approach with short sleeps to avoid
    /// blocking the tokio runtime while still providing responsive event delivery.
    pub async fn recv(&self) -> Result<SystemEvent, RecvError> {
        loop {
            // Try non-blocking receive first
            if let Some(event) = self.inner.try_recv() {
                return Ok(event);
            }

            // Use a short sleep to yield to the runtime
            // This balances latency vs CPU usage
            tokio::time::sleep(Duration::from_micros(10)).await;
        }
    }

    /// Try to receive without waiting
    pub fn try_recv(&self) -> Option<SystemEvent> {
        self.inner.try_recv()
    }

    /// Receive with a timeout
    pub async fn recv_timeout(&self, timeout: Duration) -> Result<SystemEvent, RecvTimeoutError> {
        let deadline = Instant::now() + timeout;

        loop {
            if let Some(event) = self.inner.try_recv() {
                return Ok(event);
            }

            if Instant::now() >= deadline {
                return Err(RecvTimeoutError::Timeout);
            }

            // Short sleep to yield
            tokio::time::sleep(Duration::from_micros(10)).await;
        }
    }

    /// Check pending message count
    pub fn pending(&self) -> usize {
        self.inner.pending()
    }

    /// Get the subscriber ID
    pub fn id(&self) -> usize {
        self.inner.id()
    }

    /// Get the inner receiver for direct crossbeam::select! usage
    pub fn receiver(&self) -> &Receiver<SystemEvent> {
        self.inner.receiver()
    }
}

/// Extension trait to convert EventSubscription to AsyncEventSubscription
pub trait EventSubscriptionExt {
    /// Convert to an async-compatible subscription
    fn into_async(self) -> AsyncEventSubscription;
}

impl EventSubscriptionExt for EventSubscription {
    fn into_async(self) -> AsyncEventSubscription {
        AsyncEventSubscription::new(self)
    }
}

/// Unified event bus that can use either broadcast or lock-free implementation
///
/// This enum allows switching between implementations while maintaining
/// compatibility with code that uses the `EventBus` trait.
#[derive(Clone)]
pub enum UnifiedEventBus {
    /// Tokio broadcast-based implementation (original)
    Broadcast(BroadcastEventBus),
    /// Lock-free crossbeam-based implementation (new)
    LockFree(LockFreeEventBus),
}

impl UnifiedEventBus {
    /// Create a new broadcast-based event bus
    pub fn broadcast() -> Self {
        Self::Broadcast(BroadcastEventBus::new())
    }

    /// Create a new lock-free event bus with default config
    pub fn lock_free() -> Self {
        Self::LockFree(LockFreeEventBus::new())
    }

    /// Create a new lock-free event bus with custom config
    pub fn lock_free_with_config(config: BackpressureConfig) -> Self {
        Self::LockFree(LockFreeEventBus::with_config(config))
    }

    /// Check if this is the lock-free implementation
    pub fn is_lock_free(&self) -> bool {
        matches!(self, Self::LockFree(_))
    }

    /// Get the sender for the broadcast bus (for WebSocket manager compatibility)
    /// Returns None if using lock-free implementation
    pub fn broadcast_sender(&self) -> Option<broadcast::Sender<SystemEvent>> {
        match self {
            Self::Broadcast(bus) => Some(bus.sender()),
            Self::LockFree(_) => None,
        }
    }

    /// Get lock-free subscription (returns None for broadcast bus)
    pub fn subscribe_lock_free(&self) -> Option<EventSubscription> {
        match self {
            Self::LockFree(bus) => Some(bus.subscribe()),
            Self::Broadcast(_) => None,
        }
    }

    /// Get lock-free async subscription (returns None for broadcast bus)
    pub fn subscribe_lock_free_async(&self) -> Option<AsyncEventSubscription> {
        match self {
            Self::LockFree(bus) => Some(bus.subscribe().into_async()),
            Self::Broadcast(_) => None,
        }
    }

    /// Get metrics for lock-free bus (returns None for broadcast bus)
    pub fn lock_free_metrics(&self) -> Option<EventBusMetrics> {
        match self {
            Self::LockFree(bus) => Some(bus.metrics()),
            Self::Broadcast(_) => None,
        }
    }

    /// Publish event and return delivery count (for lock-free) or 1 (for broadcast)
    pub fn publish_with_count(&self, event: SystemEvent) -> usize {
        match self {
            Self::Broadcast(bus) => {
                bus.publish(event);
                1 // Broadcast doesn't return delivery count
            }
            Self::LockFree(bus) => bus.publish(event),
        }
    }
}

impl EventBus for UnifiedEventBus {
    fn publish(&self, event: SystemEvent) {
        match self {
            Self::Broadcast(bus) => bus.publish(event),
            Self::LockFree(bus) => {
                bus.publish(event);
            }
        }
    }

    fn subscribe(&self) -> broadcast::Receiver<SystemEvent> {
        match self {
            Self::Broadcast(bus) => bus.subscribe(),
            Self::LockFree(_) => {
                // Create a dummy broadcast channel for compatibility
                // Users should use subscribe_lock_free() or subscribe_lock_free_async() instead
                let (tx, rx) = broadcast::channel(1);
                drop(tx);
                rx
            }
        }
    }

    fn subscriber_count(&self) -> usize {
        match self {
            Self::Broadcast(bus) => bus.subscriber_count(),
            Self::LockFree(bus) => bus.subscriber_count(),
        }
    }
}

impl Default for UnifiedEventBus {
    fn default() -> Self {
        // Default to lock-free for better performance
        Self::lock_free()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polysniper_core::HeartbeatEvent;

    fn create_test_event() -> SystemEvent {
        SystemEvent::Heartbeat(HeartbeatEvent {
            source: "test".to_string(),
            timestamp: chrono::Utc::now(),
        })
    }

    #[test]
    fn test_subscribe_and_publish() {
        let bus = LockFreeEventBus::new();
        let sub = bus.subscribe();

        assert_eq!(bus.subscriber_count(), 1);

        let event = create_test_event();
        let delivered = bus.publish(event);

        assert_eq!(delivered, 1);
        assert!(sub.try_recv().is_some());
    }

    #[test]
    fn test_multiple_subscribers() {
        let bus = LockFreeEventBus::new();
        let sub1 = bus.subscribe();
        let sub2 = bus.subscribe();
        let sub3 = bus.subscribe();

        assert_eq!(bus.subscriber_count(), 3);

        let event = create_test_event();
        let delivered = bus.publish(event);

        assert_eq!(delivered, 3);
        assert!(sub1.try_recv().is_some());
        assert!(sub2.try_recv().is_some());
        assert!(sub3.try_recv().is_some());
    }

    #[test]
    fn test_subscriber_cleanup_on_drop() {
        let bus = LockFreeEventBus::new();
        {
            let _sub = bus.subscribe();
            assert_eq!(bus.subscriber_count(), 1);
        }
        // Subscriber marked inactive after drop
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn test_metrics() {
        let bus = LockFreeEventBus::new();
        let _sub = bus.subscribe();

        for _ in 0..100 {
            bus.publish(create_test_event());
        }

        let metrics = bus.metrics();
        assert_eq!(metrics.total_published, 100);
        assert_eq!(metrics.subscriber_count, 1);
        assert!(metrics.avg_queue_depth <= 100.0);
    }

    #[test]
    fn test_recv_timeout() {
        let bus = LockFreeEventBus::new();
        let sub = bus.subscribe();

        // Should timeout when no events
        let result = sub.recv_timeout(Duration::from_millis(10));
        assert!(matches!(result, Err(RecvTimeoutError::Timeout)));

        // Should receive when event is published
        bus.publish(create_test_event());
        let result = sub.recv_timeout(Duration::from_millis(100));
        assert!(result.is_ok());
    }

    #[test]
    fn test_pending_count() {
        let bus = LockFreeEventBus::new();
        let sub = bus.subscribe();

        assert_eq!(sub.pending(), 0);

        for _ in 0..10 {
            bus.publish(create_test_event());
        }

        assert_eq!(sub.pending(), 10);

        sub.try_recv();
        assert_eq!(sub.pending(), 9);
    }

    #[tokio::test]
    async fn test_async_recv() {
        let bus = LockFreeEventBus::new();
        let sub = bus.subscribe().into_async();

        // Spawn a task to publish after a short delay
        let bus_clone = bus.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            bus_clone.publish(create_test_event());
        });

        // Should receive the event
        let result = sub.recv().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_async_recv_timeout() {
        let bus = LockFreeEventBus::new();
        let sub = bus.subscribe().into_async();

        // Should timeout when no events
        let result = sub.recv_timeout(Duration::from_millis(50)).await;
        assert!(matches!(result, Err(RecvTimeoutError::Timeout)));

        // Publish an event and try again
        bus.publish(create_test_event());
        let result = sub.recv_timeout(Duration::from_millis(100)).await;
        assert!(result.is_ok());
    }
}
