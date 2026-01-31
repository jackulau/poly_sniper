//! Batch processor for high-throughput event processing
//!
//! Batches market update events to reduce per-event overhead and improve
//! throughput during high-frequency update periods.

use crate::events::{OrderbookUpdateEvent, PriceChangeEvent, SystemEvent};
use crate::types::TokenId;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Configuration for batch processing
#[derive(Debug, Clone)]
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

impl BatchConfig {
    /// Create a config optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            max_batch_size: 50,
            max_batch_duration: Duration::from_millis(5),
            min_batch_size: 1,
        }
    }

    /// Create a config optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            max_batch_size: 200,
            max_batch_duration: Duration::from_millis(20),
            min_batch_size: 5,
        }
    }
}

/// Events grouped by type for efficient batch processing
#[derive(Debug, Default)]
pub struct EventsByType {
    pub price_changes: Vec<PriceChangeEvent>,
    pub orderbook_updates: Vec<OrderbookUpdateEvent>,
    pub other: Vec<SystemEvent>,
}

impl EventsByType {
    /// Get the total number of events
    pub fn len(&self) -> usize {
        self.price_changes.len() + self.orderbook_updates.len() + self.other.len()
    }

    /// Check if there are no events
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Batch of events for processing
#[derive(Debug)]
pub struct EventBatch {
    events: Vec<SystemEvent>,
    created_at: Instant,
    /// Deduplicated token IDs that have updates
    affected_tokens: HashSet<TokenId>,
}

impl Default for EventBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBatch {
    /// Create a new empty batch
    pub fn new() -> Self {
        Self {
            events: Vec::with_capacity(100),
            created_at: Instant::now(),
            affected_tokens: HashSet::new(),
        }
    }

    /// Add event to batch, returns true if batch is ready to be processed
    pub fn push(&mut self, event: SystemEvent, config: &BatchConfig) -> bool {
        // Track affected tokens for deduplication
        match &event {
            SystemEvent::PriceChange(e) => {
                self.affected_tokens.insert(e.token_id.clone());
            }
            SystemEvent::OrderbookUpdate(e) => {
                self.affected_tokens.insert(e.token_id.clone());
            }
            _ => {}
        }

        self.events.push(event);

        // Check if batch should be flushed
        self.should_flush(config)
    }

    /// Check if batch should be flushed based on config
    pub fn should_flush(&self, config: &BatchConfig) -> bool {
        // Flush if at max size
        if self.events.len() >= config.max_batch_size {
            return true;
        }

        // Flush if duration exceeded and we have at least min_batch_size
        if self.events.len() >= config.min_batch_size
            && self.created_at.elapsed() >= config.max_batch_duration
        {
            return true;
        }

        false
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get the number of events in the batch
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Get the set of affected token IDs
    pub fn affected_tokens(&self) -> &HashSet<TokenId> {
        &self.affected_tokens
    }

    /// Get events grouped by type for efficient processing
    pub fn by_type(&self) -> EventsByType {
        let mut result = EventsByType::default();

        for event in &self.events {
            match event {
                SystemEvent::PriceChange(e) => {
                    result.price_changes.push(e.clone());
                }
                SystemEvent::OrderbookUpdate(e) => {
                    result.orderbook_updates.push(e.clone());
                }
                other => {
                    result.other.push(other.clone());
                }
            }
        }

        result
    }

    /// Get the latest price change for each token
    pub fn latest_price_changes(&self) -> HashMap<TokenId, PriceChangeEvent> {
        let mut latest: HashMap<TokenId, PriceChangeEvent> = HashMap::new();

        for event in &self.events {
            if let SystemEvent::PriceChange(e) = event {
                latest.insert(e.token_id.clone(), e.clone());
            }
        }

        latest
    }

    /// Get the latest orderbook update for each token
    pub fn latest_orderbook_updates(&self) -> HashMap<TokenId, OrderbookUpdateEvent> {
        let mut latest: HashMap<TokenId, OrderbookUpdateEvent> = HashMap::new();

        for event in &self.events {
            if let SystemEvent::OrderbookUpdate(e) = event {
                latest.insert(e.token_id.clone(), e.clone());
            }
        }

        latest
    }

    /// Drain all events from the batch and reset
    pub fn drain(&mut self) -> Vec<SystemEvent> {
        self.affected_tokens.clear();
        self.created_at = Instant::now();
        std::mem::take(&mut self.events)
    }

    /// Get elapsed time since batch was created
    pub fn elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get a reference to all events
    pub fn events(&self) -> &[SystemEvent] {
        &self.events
    }
}

/// Check if an event should bypass batching (time-sensitive events)
pub fn should_bypass_batching(event: &SystemEvent) -> bool {
    matches!(
        event,
        SystemEvent::Heartbeat(_)
            | SystemEvent::ConfigChanged(_)
            | SystemEvent::ConnectionStatus(_)
            | SystemEvent::FullFill(_)
            | SystemEvent::PartialFill(_)
            | SystemEvent::OrderExpired(_)
            | SystemEvent::ResubmitTriggered(_)
            | SystemEvent::OrderReplaced(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{HeartbeatEvent, PriceChangeEvent};
    use chrono::Utc;
    use rust_decimal_macros::dec;

    fn create_price_event(token_id: &str, price: rust_decimal::Decimal) -> SystemEvent {
        SystemEvent::PriceChange(PriceChangeEvent {
            market_id: "test_market".to_string(),
            token_id: token_id.to_string(),
            old_price: None,
            new_price: price,
            price_change_pct: None,
            timestamp: Utc::now(),
        })
    }

    #[test]
    fn test_batch_creation() {
        let batch = EventBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_batch_push_and_size() {
        let mut batch = EventBatch::new();
        let config = BatchConfig::default();

        let event = create_price_event("token1", dec!(0.5));
        let ready = batch.push(event, &config);

        assert!(!ready); // Not at max size
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_max_size_flush() {
        let mut batch = EventBatch::new();
        let config = BatchConfig {
            max_batch_size: 3,
            ..Default::default()
        };

        for i in 0..2 {
            let event = create_price_event(&format!("token{}", i), dec!(0.5));
            assert!(!batch.push(event, &config));
        }

        // Third event should trigger flush
        let event = create_price_event("token2", dec!(0.5));
        assert!(batch.push(event, &config));
    }

    #[test]
    fn test_affected_tokens_tracking() {
        let mut batch = EventBatch::new();
        let config = BatchConfig::default();

        batch.push(create_price_event("token1", dec!(0.5)), &config);
        batch.push(create_price_event("token2", dec!(0.6)), &config);
        batch.push(create_price_event("token1", dec!(0.55)), &config); // Duplicate token

        assert_eq!(batch.affected_tokens().len(), 2);
        assert!(batch.affected_tokens().contains("token1"));
        assert!(batch.affected_tokens().contains("token2"));
    }

    #[test]
    fn test_latest_price_changes() {
        let mut batch = EventBatch::new();
        let config = BatchConfig::default();

        batch.push(create_price_event("token1", dec!(0.5)), &config);
        batch.push(create_price_event("token1", dec!(0.55)), &config); // Later update
        batch.push(create_price_event("token2", dec!(0.6)), &config);

        let latest = batch.latest_price_changes();
        assert_eq!(latest.len(), 2);
        assert_eq!(latest.get("token1").unwrap().new_price, dec!(0.55));
        assert_eq!(latest.get("token2").unwrap().new_price, dec!(0.6));
    }

    #[test]
    fn test_events_by_type() {
        let mut batch = EventBatch::new();
        let config = BatchConfig::default();

        batch.push(create_price_event("token1", dec!(0.5)), &config);
        batch.push(
            SystemEvent::Heartbeat(HeartbeatEvent {
                source: "test".to_string(),
                timestamp: Utc::now(),
            }),
            &config,
        );

        let by_type = batch.by_type();
        assert_eq!(by_type.price_changes.len(), 1);
        assert_eq!(by_type.other.len(), 1);
        assert_eq!(by_type.orderbook_updates.len(), 0);
    }

    #[test]
    fn test_batch_drain() {
        let mut batch = EventBatch::new();
        let config = BatchConfig::default();

        batch.push(create_price_event("token1", dec!(0.5)), &config);
        batch.push(create_price_event("token2", dec!(0.6)), &config);

        let events = batch.drain();
        assert_eq!(events.len(), 2);
        assert!(batch.is_empty());
        assert!(batch.affected_tokens().is_empty());
    }

    #[test]
    fn test_should_bypass_batching() {
        // Heartbeat should bypass
        let heartbeat = SystemEvent::Heartbeat(HeartbeatEvent {
            source: "test".to_string(),
            timestamp: Utc::now(),
        });
        assert!(should_bypass_batching(&heartbeat));

        // Price change should not bypass
        let price = create_price_event("token1", dec!(0.5));
        assert!(!should_bypass_batching(&price));
    }

    #[test]
    fn test_config_presets() {
        let low_latency = BatchConfig::low_latency();
        assert!(low_latency.max_batch_duration < BatchConfig::default().max_batch_duration);

        let high_throughput = BatchConfig::high_throughput();
        assert!(high_throughput.max_batch_size > BatchConfig::default().max_batch_size);
    }
}
