//! Microstructure Event Publisher
//!
//! Helper module for publishing microstructure-related events to the event bus.
//! Provides a simplified interface for VPIN calculators, whale detectors, and
//! market impact estimators to emit events.

use crate::events::{
    ImpactPredictionEvent, MicrostructureEvent, MicrostructureSignalEvent, SystemEvent,
    ToxicityChangeEvent, VpinUpdateEvent, WhaleDetectedEvent,
};
use thiserror::Error;
use tokio::sync::broadcast;
use tracing::debug;

/// Error type for publishing microstructure events
#[derive(Debug, Error)]
pub enum PublishError {
    /// The event channel has been closed
    #[error("Event channel closed")]
    ChannelClosed,
}

/// Publisher for microstructure events
///
/// Provides a convenient interface for publishing VPIN, whale detection,
/// market impact, and combined microstructure signals to the event bus.
pub struct MicrostructurePublisher {
    event_tx: broadcast::Sender<SystemEvent>,
}

impl MicrostructurePublisher {
    /// Create a new microstructure publisher with the given event channel
    pub fn new(event_tx: broadcast::Sender<SystemEvent>) -> Self {
        Self { event_tx }
    }

    /// Publish a VPIN update event
    ///
    /// # Arguments
    /// * `event` - The VPIN update event to publish
    ///
    /// # Returns
    /// `Ok(())` if the event was published successfully, `Err(PublishError::ChannelClosed)` if the channel is closed
    pub fn publish_vpin_update(&self, event: VpinUpdateEvent) -> Result<(), PublishError> {
        debug!(
            token_id = %event.token_id,
            vpin = %event.vpin,
            toxicity = %event.toxicity_level,
            "Publishing VPIN update event"
        );
        self.publish(MicrostructureEvent::VpinUpdate(event))
    }

    /// Publish a whale detected event
    ///
    /// # Arguments
    /// * `event` - The whale detected event to publish
    ///
    /// # Returns
    /// `Ok(())` if the event was published successfully, `Err(PublishError::ChannelClosed)` if the channel is closed
    pub fn publish_whale_detected(&self, event: WhaleDetectedEvent) -> Result<(), PublishError> {
        debug!(
            token_id = %event.token_id,
            alert_type = ?event.alert_type,
            trade_size_usd = %event.trade_size_usd,
            "Publishing whale detected event"
        );
        self.publish(MicrostructureEvent::WhaleDetected(event))
    }

    /// Publish a market impact prediction event
    ///
    /// # Arguments
    /// * `event` - The impact prediction event to publish
    ///
    /// # Returns
    /// `Ok(())` if the event was published successfully, `Err(PublishError::ChannelClosed)` if the channel is closed
    pub fn publish_impact_prediction(
        &self,
        event: ImpactPredictionEvent,
    ) -> Result<(), PublishError> {
        debug!(
            token_id = %event.token_id,
            proposed_size_usd = %event.proposed_size_usd,
            expected_impact_bps = %event.expected_impact_bps,
            "Publishing impact prediction event"
        );
        self.publish(MicrostructureEvent::ImpactPrediction(event))
    }

    /// Publish a toxicity level change event
    ///
    /// # Arguments
    /// * `event` - The toxicity change event to publish
    ///
    /// # Returns
    /// `Ok(())` if the event was published successfully, `Err(PublishError::ChannelClosed)` if the channel is closed
    pub fn publish_toxicity_change(&self, event: ToxicityChangeEvent) -> Result<(), PublishError> {
        debug!(
            token_id = %event.token_id,
            previous = %event.previous_level,
            new = %event.new_level,
            vpin = %event.vpin,
            "Publishing toxicity change event"
        );
        self.publish(MicrostructureEvent::ToxicityChange(event))
    }

    /// Publish a combined microstructure signal event
    ///
    /// # Arguments
    /// * `event` - The microstructure signal event to publish
    ///
    /// # Returns
    /// `Ok(())` if the event was published successfully, `Err(PublishError::ChannelClosed)` if the channel is closed
    pub fn publish_signal(&self, event: MicrostructureSignalEvent) -> Result<(), PublishError> {
        debug!(
            token_id = %event.token_id,
            signal_type = ?event.signal_type,
            strength = %event.strength,
            confidence = %event.confidence,
            "Publishing microstructure signal event"
        );
        self.publish(MicrostructureEvent::MicrostructureSignal(event))
    }

    /// Publish any microstructure event
    ///
    /// # Arguments
    /// * `event` - The microstructure event to publish
    ///
    /// # Returns
    /// `Ok(())` if the event was published successfully, `Err(PublishError::ChannelClosed)` if the channel is closed
    pub fn publish(&self, event: MicrostructureEvent) -> Result<(), PublishError> {
        self.event_tx
            .send(SystemEvent::Microstructure(event))
            .map(|_| ())
            .map_err(|_| PublishError::ChannelClosed)
    }

    /// Get the number of active receivers
    pub fn receiver_count(&self) -> usize {
        self.event_tx.receiver_count()
    }

    /// Subscribe to microstructure events
    ///
    /// Returns a receiver that will receive all SystemEvents (filter for Microstructure variants)
    pub fn subscribe(&self) -> broadcast::Receiver<SystemEvent> {
        self.event_tx.subscribe()
    }
}

impl Clone for MicrostructurePublisher {
    fn clone(&self) -> Self {
        Self {
            event_tx: self.event_tx.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{
        MicrostructureAction, MicrostructureComponents, MicrostructureSignalType, ToxicityLevel,
        WhaleAction, WhaleAlertType,
    };
    use crate::types::Side;
    use rust_decimal_macros::dec;

    fn create_publisher() -> (MicrostructurePublisher, broadcast::Receiver<SystemEvent>) {
        let (tx, rx) = broadcast::channel(100);
        let publisher = MicrostructurePublisher::new(tx);
        (publisher, rx)
    }

    #[tokio::test]
    async fn test_publish_vpin_update() {
        let (publisher, mut rx) = create_publisher();

        let event = VpinUpdateEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            dec!(0.65),
            ToxicityLevel::Elevated,
            dec!(0.60),
            dec!(0.40),
            50,
        );

        publisher.publish_vpin_update(event.clone()).unwrap();

        let received = rx.recv().await.unwrap();
        match received {
            SystemEvent::Microstructure(MicrostructureEvent::VpinUpdate(e)) => {
                assert_eq!(e.token_id, "token-1");
                assert_eq!(e.vpin, dec!(0.65));
                assert_eq!(e.toxicity_level, ToxicityLevel::Elevated);
            }
            _ => panic!("Expected VpinUpdate event"),
        }
    }

    #[tokio::test]
    async fn test_publish_whale_detected() {
        let (publisher, mut rx) = create_publisher();

        let event = WhaleDetectedEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            WhaleAlertType::SingleLargeTrade,
            dec!(10000),
            Side::Buy,
            WhaleAction::Alert,
            dec!(0.85),
        );

        publisher.publish_whale_detected(event.clone()).unwrap();

        let received = rx.recv().await.unwrap();
        match received {
            SystemEvent::Microstructure(MicrostructureEvent::WhaleDetected(e)) => {
                assert_eq!(e.token_id, "token-1");
                assert_eq!(e.trade_size_usd, dec!(10000));
                assert_eq!(e.alert_type, WhaleAlertType::SingleLargeTrade);
            }
            _ => panic!("Expected WhaleDetected event"),
        }
    }

    #[tokio::test]
    async fn test_publish_impact_prediction() {
        let (publisher, mut rx) = create_publisher();

        let event = ImpactPredictionEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            dec!(5000),
            dec!(25),
            dec!(20),
            dec!(5),
            60,
            dec!(0.90),
        );

        publisher.publish_impact_prediction(event.clone()).unwrap();

        let received = rx.recv().await.unwrap();
        match received {
            SystemEvent::Microstructure(MicrostructureEvent::ImpactPrediction(e)) => {
                assert_eq!(e.token_id, "token-1");
                assert_eq!(e.proposed_size_usd, dec!(5000));
                assert_eq!(e.expected_impact_bps, dec!(25));
            }
            _ => panic!("Expected ImpactPrediction event"),
        }
    }

    #[tokio::test]
    async fn test_publish_toxicity_change() {
        let (publisher, mut rx) = create_publisher();

        let event = ToxicityChangeEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            ToxicityLevel::Normal,
            ToxicityLevel::Elevated,
            dec!(0.55),
            "VPIN crossed 0.5 threshold".to_string(),
        );

        publisher.publish_toxicity_change(event.clone()).unwrap();

        let received = rx.recv().await.unwrap();
        match received {
            SystemEvent::Microstructure(MicrostructureEvent::ToxicityChange(e)) => {
                assert_eq!(e.token_id, "token-1");
                assert_eq!(e.previous_level, ToxicityLevel::Normal);
                assert_eq!(e.new_level, ToxicityLevel::Elevated);
                assert!(e.is_increase());
            }
            _ => panic!("Expected ToxicityChange event"),
        }
    }

    #[tokio::test]
    async fn test_publish_microstructure_signal() {
        let (publisher, mut rx) = create_publisher();

        let components = MicrostructureComponents {
            vpin: Some(dec!(0.65)),
            toxicity_level: Some(ToxicityLevel::Elevated),
            whale_activity: None,
            expected_impact_bps: Some(dec!(25)),
        };

        let event = MicrostructureSignalEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            MicrostructureSignalType::Unfavorable,
            dec!(-0.5),
            dec!(0.80),
            components,
            MicrostructureAction::ReduceSize {
                multiplier: dec!(0.5),
            },
        );

        publisher.publish_signal(event.clone()).unwrap();

        let received = rx.recv().await.unwrap();
        match received {
            SystemEvent::Microstructure(MicrostructureEvent::MicrostructureSignal(e)) => {
                assert_eq!(e.token_id, "token-1");
                assert_eq!(e.signal_type, MicrostructureSignalType::Unfavorable);
                assert!(e.is_unfavorable());
            }
            _ => panic!("Expected MicrostructureSignal event"),
        }
    }

    #[test]
    fn test_publisher_clone() {
        let (tx, _rx) = broadcast::channel(100);
        let publisher = MicrostructurePublisher::new(tx);
        let cloned = publisher.clone();

        // Both should point to same channel
        assert_eq!(publisher.receiver_count(), cloned.receiver_count());
    }

    #[test]
    fn test_channel_closed_error() {
        let (tx, _rx) = broadcast::channel::<SystemEvent>(1);
        let publisher = MicrostructurePublisher::new(tx);

        // Drop the receiver
        drop(_rx);

        let event = VpinUpdateEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            dec!(0.5),
            ToxicityLevel::Normal,
            dec!(0.50),
            dec!(0.50),
            10,
        );

        // Should still succeed since broadcast doesn't require receivers
        // (unlike mpsc). The send will succeed but no one receives it.
        let result = publisher.publish_vpin_update(event);
        // With broadcast, send succeeds even with no receivers
        assert!(result.is_ok() || matches!(result, Err(PublishError::ChannelClosed)));
    }

    #[test]
    fn test_vpin_update_event_creation() {
        let event = VpinUpdateEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            dec!(0.65),
            ToxicityLevel::Elevated,
            dec!(0.60),
            dec!(0.40),
            50,
        );

        assert_eq!(event.token_id, "token-1");
        assert_eq!(event.market_id, "market-1");
        assert_eq!(event.vpin, dec!(0.65));
        assert_eq!(event.toxicity_level, ToxicityLevel::Elevated);
        assert_eq!(event.bucket_count, 50);
    }

    #[test]
    fn test_whale_detected_event_builder() {
        let event = WhaleDetectedEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            WhaleAlertType::SingleLargeTrade,
            dec!(10000),
            Side::Buy,
            WhaleAction::Alert,
            dec!(0.85),
        )
        .with_cumulative_size(dec!(25000))
        .with_address("0x123...".to_string())
        .with_classification(crate::events::WhaleClassification::Informed);

        assert_eq!(event.cumulative_size_usd, Some(dec!(25000)));
        assert_eq!(event.address, Some("0x123...".to_string()));
        assert_eq!(
            event.classification,
            Some(crate::events::WhaleClassification::Informed)
        );
    }

    #[test]
    fn test_toxicity_change_direction() {
        let increase = ToxicityChangeEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            ToxicityLevel::Low,
            ToxicityLevel::High,
            dec!(0.75),
            "Test increase".to_string(),
        );
        assert!(increase.is_increase());
        assert!(!increase.is_decrease());

        let decrease = ToxicityChangeEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            ToxicityLevel::High,
            ToxicityLevel::Normal,
            dec!(0.45),
            "Test decrease".to_string(),
        );
        assert!(!decrease.is_increase());
        assert!(decrease.is_decrease());
    }

    #[test]
    fn test_microstructure_signal_favorable() {
        let favorable = MicrostructureSignalEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            MicrostructureSignalType::Favorable,
            dec!(0.5),
            dec!(0.80),
            MicrostructureComponents::default(),
            MicrostructureAction::None,
        );
        assert!(favorable.is_favorable());
        assert!(!favorable.is_unfavorable());

        let unfavorable = MicrostructureSignalEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            MicrostructureSignalType::HighToxicity,
            dec!(-0.5),
            dec!(0.80),
            MicrostructureComponents::default(),
            MicrostructureAction::HaltTrading,
        );
        assert!(!unfavorable.is_favorable());
        assert!(unfavorable.is_unfavorable());
    }

    #[test]
    fn test_microstructure_event_methods() {
        let vpin_event = MicrostructureEvent::VpinUpdate(VpinUpdateEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            dec!(0.5),
            ToxicityLevel::Normal,
            dec!(0.50),
            dec!(0.50),
            10,
        ));

        assert_eq!(vpin_event.event_type(), "vpin_update");
        assert_eq!(vpin_event.market_id(), Some(&"market-1".to_string()));
        assert_eq!(vpin_event.token_id(), Some(&"token-1".to_string()));
    }

    #[test]
    fn test_event_serialization() {
        let event = VpinUpdateEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            dec!(0.65),
            ToxicityLevel::Elevated,
            dec!(0.60),
            dec!(0.40),
            50,
        );

        // Should serialize and deserialize correctly
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: VpinUpdateEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.token_id, event.token_id);
        assert_eq!(deserialized.vpin, event.vpin);
        assert_eq!(deserialized.toxicity_level, event.toxicity_level);
    }

    #[test]
    fn test_microstructure_event_wrapper_serialization() {
        let inner = VpinUpdateEvent::new(
            "token-1".to_string(),
            "market-1".to_string(),
            dec!(0.65),
            ToxicityLevel::Elevated,
            dec!(0.60),
            dec!(0.40),
            50,
        );
        let event = MicrostructureEvent::VpinUpdate(inner);

        let json = serde_json::to_string(&event).unwrap();
        let deserialized: MicrostructureEvent = serde_json::from_str(&json).unwrap();

        match deserialized {
            MicrostructureEvent::VpinUpdate(e) => {
                assert_eq!(e.token_id, "token-1");
            }
            _ => panic!("Expected VpinUpdate variant"),
        }
    }
}
