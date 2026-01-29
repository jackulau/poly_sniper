//! Event bus implementation using tokio broadcast

use polysniper_core::{EventBus, SystemEvent};
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::broadcast;
use tracing::debug;

const DEFAULT_CHANNEL_CAPACITY: usize = 10000;

/// Broadcast-based event bus
pub struct BroadcastEventBus {
    sender: broadcast::Sender<SystemEvent>,
    subscriber_count: AtomicUsize,
}

impl BroadcastEventBus {
    /// Create a new event bus with default capacity
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CHANNEL_CAPACITY)
    }

    /// Create a new event bus with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sender,
            subscriber_count: AtomicUsize::new(0),
        }
    }

    /// Get the sender for this event bus
    pub fn sender(&self) -> broadcast::Sender<SystemEvent> {
        self.sender.clone()
    }
}

impl Default for BroadcastEventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBus for BroadcastEventBus {
    fn publish(&self, event: SystemEvent) {
        debug!("Publishing event: {}", event.event_type());
        // Ignore send errors (no subscribers)
        let _ = self.sender.send(event);
    }

    fn subscribe(&self) -> broadcast::Receiver<SystemEvent> {
        self.subscriber_count.fetch_add(1, Ordering::SeqCst);
        self.sender.subscribe()
    }

    fn subscriber_count(&self) -> usize {
        self.subscriber_count.load(Ordering::SeqCst)
    }
}

impl Clone for BroadcastEventBus {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            subscriber_count: AtomicUsize::new(self.subscriber_count.load(Ordering::SeqCst)),
        }
    }
}
