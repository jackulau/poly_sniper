---
id: microstructure-events
name: Microstructure Event Types and Event Bus Integration
wave: 2
priority: 2
dependencies: [vpin-calculator, whale-detector, market-impact-estimator]
estimated_hours: 4
tags: [integration, events, core]
---

## Objective

Define microstructure-specific event types and integrate them with the existing event bus system, enabling other components (risk manager, strategies, alerting) to react to VPIN, whale, and impact signals.

## Context

The Polysniper event bus uses `SystemEvent` enum with broadcast channels. This task creates new event variants for microstructure analysis that flow through the existing infrastructure:
- `MicrostructureEvent` - Wrapper for all microstructure signals
- Enables risk manager to adjust sizing based on toxicity/whale activity
- Enables alerting system to notify on significant microstructure changes
- Enables strategies to incorporate microstructure signals

## Implementation

### 1. Add new event types to `crates/polysniper-core/src/events.rs`

**New Event Variants:**

```rust
// Add to SystemEvent enum
pub enum SystemEvent {
    // ... existing variants ...

    /// Microstructure analysis events
    Microstructure(MicrostructureEvent),
}

/// Wrapper for all microstructure-related events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MicrostructureEvent {
    /// VPIN calculation update
    VpinUpdate(VpinUpdateEvent),

    /// Whale activity detected
    WhaleDetected(WhaleDetectedEvent),

    /// Market impact prediction
    ImpactPrediction(ImpactPredictionEvent),

    /// Toxicity level change (crossing threshold)
    ToxicityChange(ToxicityChangeEvent),

    /// Combined microstructure signal
    MicrostructureSignal(MicrostructureSignalEvent),
}

/// VPIN calculation update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpinUpdateEvent {
    pub token_id: TokenId,
    pub market_id: MarketId,
    pub vpin: Decimal,
    pub toxicity_level: ToxicityLevel,
    pub buy_volume_pct: Decimal,
    pub sell_volume_pct: Decimal,
    pub bucket_count: usize,
    pub timestamp: DateTime<Utc>,
}

/// Whale activity detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleDetectedEvent {
    pub token_id: TokenId,
    pub market_id: MarketId,
    pub alert_type: WhaleAlertType,
    pub trade_size_usd: Decimal,
    pub cumulative_size_usd: Option<Decimal>,
    pub side: Side,
    pub address: Option<String>,
    pub classification: Option<WhaleClassification>,
    pub recommended_action: WhaleAction,
    pub confidence: Decimal,
    pub timestamp: DateTime<Utc>,
}

/// Market impact prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactPredictionEvent {
    pub token_id: TokenId,
    pub market_id: MarketId,
    pub proposed_size_usd: Decimal,
    pub expected_impact_bps: Decimal,
    pub temporary_impact_bps: Decimal,
    pub permanent_impact_bps: Decimal,
    pub expected_recovery_secs: u64,
    pub model_confidence: Decimal,
    pub timestamp: DateTime<Utc>,
}

/// Toxicity level change (crossing threshold)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToxicityChangeEvent {
    pub token_id: TokenId,
    pub market_id: MarketId,
    pub previous_level: ToxicityLevel,
    pub new_level: ToxicityLevel,
    pub vpin: Decimal,
    pub trigger_reason: String,
    pub timestamp: DateTime<Utc>,
}

/// Combined microstructure signal for strategy use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureSignalEvent {
    pub token_id: TokenId,
    pub market_id: MarketId,
    pub signal_type: MicrostructureSignalType,
    pub strength: Decimal,              // -1.0 to 1.0 (bearish to bullish)
    pub confidence: Decimal,            // 0.0 to 1.0
    pub components: MicrostructureComponents,
    pub recommended_action: MicrostructureAction,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MicrostructureSignalType {
    Favorable,      // Good conditions for trading
    Unfavorable,    // Poor conditions, reduce activity
    WhaleFollow,    // Follow detected whale
    WhaleAvoid,     // Avoid trading against whale
    HighToxicity,   // Elevated informed trading
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureComponents {
    pub vpin: Option<Decimal>,
    pub toxicity_level: Option<ToxicityLevel>,
    pub whale_activity: Option<WhaleActivitySummary>,
    pub expected_impact_bps: Option<Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleActivitySummary {
    pub recent_whale_trades: u32,
    pub net_whale_direction: Side,
    pub total_whale_volume_usd: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MicrostructureAction {
    None,
    ReduceSize { multiplier: Decimal },
    HaltTrading,
    IncreaseSize { multiplier: Decimal },
    Urgent { reason: String },
}

// Re-export types from VPIN and Whale modules
pub use crate::vpin::{ToxicityLevel, VpinResult};
pub use crate::whale_detector::{WhaleAlertType, WhaleAction, WhaleClassification};
```

### 2. Create event publisher helper in `crates/polysniper-core/src/microstructure_publisher.rs`

```rust
use tokio::sync::broadcast;

pub struct MicrostructurePublisher {
    event_tx: broadcast::Sender<SystemEvent>,
}

impl MicrostructurePublisher {
    pub fn new(event_tx: broadcast::Sender<SystemEvent>) -> Self {
        Self { event_tx }
    }

    /// Publish a VPIN update
    pub fn publish_vpin_update(&self, event: VpinUpdateEvent) -> Result<(), PublishError> {
        self.event_tx
            .send(SystemEvent::Microstructure(MicrostructureEvent::VpinUpdate(event)))
            .map(|_| ())
            .map_err(|_| PublishError::ChannelClosed)
    }

    /// Publish a whale detection
    pub fn publish_whale_detected(&self, event: WhaleDetectedEvent) -> Result<(), PublishError> {
        self.event_tx
            .send(SystemEvent::Microstructure(MicrostructureEvent::WhaleDetected(event)))
            .map(|_| ())
            .map_err(|_| PublishError::ChannelClosed)
    }

    /// Publish an impact prediction
    pub fn publish_impact_prediction(&self, event: ImpactPredictionEvent) -> Result<(), PublishError> {
        self.event_tx
            .send(SystemEvent::Microstructure(MicrostructureEvent::ImpactPrediction(event)))
            .map(|_| ())
            .map_err(|_| PublishError::ChannelClosed)
    }

    /// Publish a toxicity level change
    pub fn publish_toxicity_change(&self, event: ToxicityChangeEvent) -> Result<(), PublishError> {
        self.event_tx
            .send(SystemEvent::Microstructure(MicrostructureEvent::ToxicityChange(event)))
            .map(|_| ())
            .map_err(|_| PublishError::ChannelClosed)
    }

    /// Publish a combined microstructure signal
    pub fn publish_signal(&self, event: MicrostructureSignalEvent) -> Result<(), PublishError> {
        self.event_tx
            .send(SystemEvent::Microstructure(MicrostructureEvent::MicrostructureSignal(event)))
            .map(|_| ())
            .map_err(|_| PublishError::ChannelClosed)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PublishError {
    #[error("Event channel closed")]
    ChannelClosed,
}
```

### 3. Update event matching in strategies

Update the `accepts_event` pattern for strategies that want to consume microstructure events:

```rust
// Example in a strategy
fn accepts_event(&self, event: &SystemEvent) -> bool {
    matches!(
        event,
        SystemEvent::PriceChange(_)
            | SystemEvent::OrderbookUpdate(_)
            | SystemEvent::Microstructure(_)  // NEW: Accept microstructure events
    )
}

// Example processing
async fn process_event(&self, event: &SystemEvent, state: &dyn StateProvider) -> Result<Vec<TradeSignal>, StrategyError> {
    match event {
        SystemEvent::Microstructure(micro) => match micro {
            MicrostructureEvent::WhaleDetected(whale) => {
                // React to whale detection
            }
            MicrostructureEvent::ToxicityChange(toxicity) => {
                // React to toxicity changes
            }
            _ => Ok(vec![]),
        },
        _ => Ok(vec![]),
    }
}
```

### 4. Add tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microstructure_event_serialization() {
        // Ensure all events serialize/deserialize correctly
    }

    #[test]
    fn test_vpin_update_event() {
        let event = VpinUpdateEvent {
            token_id: "test-token".to_string(),
            market_id: "test-market".to_string(),
            vpin: dec!(0.65),
            toxicity_level: ToxicityLevel::Elevated,
            // ...
        };
        // Verify fields
    }

    #[tokio::test]
    async fn test_publisher_broadcasts_events() {
        let (tx, mut rx) = broadcast::channel(100);
        let publisher = MicrostructurePublisher::new(tx);

        publisher.publish_vpin_update(test_vpin_event()).unwrap();

        let received = rx.recv().await.unwrap();
        assert!(matches!(received, SystemEvent::Microstructure(MicrostructureEvent::VpinUpdate(_))));
    }

    #[test]
    fn test_microstructure_action_variants() {
        // Test all action types
    }
}
```

## Acceptance Criteria

- [ ] All new event types are defined in `events.rs`
- [ ] Events serialize/deserialize correctly with serde
- [ ] `MicrostructurePublisher` helper works with broadcast channel
- [ ] Event matching works in strategy pattern
- [ ] All tests pass (`cargo test -p polysniper-core`)
- [ ] No breaking changes to existing event handling
- [ ] Code follows existing patterns
- [ ] No clippy warnings

## Files to Create/Modify

- `crates/polysniper-core/src/events.rs` - **MODIFY** - Add new event types
- `crates/polysniper-core/src/microstructure_publisher.rs` - **CREATE** - Event publisher helper
- `crates/polysniper-core/src/lib.rs` - **MODIFY** - Export new module

## Integration Points

- **Provides**: Event types for microstructure signals, publisher helper
- **Consumes**: Results from VPIN calculator, whale detector, market impact estimator
- **Depends on**: vpin-calculator, whale-detector, market-impact-estimator (for types)
- **Conflicts**: Modifies shared `events.rs` - coordinate with other changes
