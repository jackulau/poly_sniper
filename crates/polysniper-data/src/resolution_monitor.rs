//! Resolution Monitor Service
//!
//! Monitors markets with open positions for resolution events and emits warnings
//! as markets approach their resolution time.

use crate::gamma_client::MarketStatus;
use crate::GammaClient;
use chrono::Utc;
use polysniper_core::{
    EventBus, MarketId, MarketState, MarketStateChangeEvent, ResolutionConfig, ResolutionInfo,
    ResolutionStatus, ResolutionWarning, StateProvider, SystemEvent,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration as TokioDuration};
use tracing::{debug, info, warn};

/// Resolution warning event wrapper for SystemEvent
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResolutionWarningEvent {
    pub warning: ResolutionWarning,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Resolution monitor service that tracks market resolution status
pub struct ResolutionMonitor<E: EventBus, S: StateProvider> {
    /// Gamma API client for fetching market status
    gamma_client: Arc<GammaClient>,
    /// Event bus for publishing events
    event_bus: Arc<E>,
    /// State provider for getting positions
    state_provider: Arc<S>,
    /// Configuration for resolution tracking
    config: ResolutionConfig,
    /// Tracked market resolution info
    tracked_markets: Arc<RwLock<HashMap<MarketId, ResolutionInfo>>>,
    /// Previous market states for detecting changes
    previous_states: Arc<RwLock<HashMap<MarketId, MarketState>>>,
    /// Whether the monitor is running
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl<E: EventBus + 'static, S: StateProvider + 'static> ResolutionMonitor<E, S> {
    /// Create a new resolution monitor
    pub fn new(
        gamma_client: Arc<GammaClient>,
        event_bus: Arc<E>,
        state_provider: Arc<S>,
        config: ResolutionConfig,
    ) -> Self {
        Self {
            gamma_client,
            event_bus,
            state_provider,
            config,
            tracked_markets: Arc::new(RwLock::new(HashMap::new())),
            previous_states: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Start the resolution monitor polling loop
    pub async fn start(&self) {
        if !self.config.enabled {
            info!("Resolution monitor is disabled");
            return;
        }

        self.running
            .store(true, std::sync::atomic::Ordering::SeqCst);
        info!(
            "Starting resolution monitor with {}s poll interval",
            self.config.poll_interval_secs
        );

        let mut poll_interval =
            interval(TokioDuration::from_secs(self.config.poll_interval_secs));

        while self.running.load(std::sync::atomic::Ordering::SeqCst) {
            poll_interval.tick().await;

            if let Err(e) = self.poll_and_check().await {
                warn!("Resolution monitor poll error: {}", e);
            }
        }

        info!("Resolution monitor stopped");
    }

    /// Stop the resolution monitor
    pub fn stop(&self) {
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if the monitor is running
    pub fn is_running(&self) -> bool {
        self.running.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Perform a single poll and check cycle
    async fn poll_and_check(&self) -> Result<(), String> {
        // Get all positions from state provider
        let positions = self.state_provider.get_all_positions().await;
        let market_ids: Vec<MarketId> = positions.iter().map(|p| p.market_id.clone()).collect();

        if market_ids.is_empty() {
            debug!("No positions to monitor for resolution");
            return Ok(());
        }

        debug!("Checking resolution status for {} markets", market_ids.len());

        // Fetch status for all markets with positions
        let statuses = self.gamma_client.fetch_market_statuses(&market_ids).await;

        // Process each market status
        for (market_id, status_result) in market_ids.iter().zip(statuses.into_iter()) {
            match status_result {
                Ok(Some(status)) => {
                    self.process_market_status(market_id, status).await;
                }
                Ok(None) => {
                    warn!("Market {} not found in Gamma API", market_id);
                }
                Err(e) => {
                    warn!("Failed to fetch status for market {}: {}", market_id, e);
                }
            }
        }

        Ok(())
    }

    /// Process a market status update
    async fn process_market_status(&self, market_id: &MarketId, status: MarketStatus) {
        // Check for state change
        self.check_state_change(market_id, &status).await;

        // Update tracked market info and check for warnings
        self.update_tracked_market(market_id, &status).await;
    }

    /// Check if market state has changed and emit event
    async fn check_state_change(&self, market_id: &MarketId, status: &MarketStatus) {
        let mut previous_states = self.previous_states.write().await;

        if let Some(previous_state) = previous_states.get(market_id) {
            if *previous_state != status.state {
                info!(
                    "Market {} state changed: {:?} -> {:?}",
                    market_id, previous_state, status.state
                );

                // Emit state change event
                let event = SystemEvent::MarketStateChange(MarketStateChangeEvent {
                    market_id: market_id.clone(),
                    old_state: *previous_state,
                    new_state: status.state,
                    timestamp: Utc::now(),
                });

                self.event_bus.publish(event);
            }
        }

        // Update previous state
        previous_states.insert(market_id.clone(), status.state);
    }

    /// Update tracked market info and check for resolution warnings
    async fn update_tracked_market(&self, market_id: &MarketId, status: &MarketStatus) {
        let mut tracked_markets = self.tracked_markets.write().await;

        // Get or create resolution info
        let info = tracked_markets
            .entry(market_id.clone())
            .or_insert_with(|| ResolutionInfo::new(market_id.clone(), status.end_date));

        // Update the info with current time
        info.update(&self.config);

        // Check if we should emit a warning
        if let Some(threshold) = info.check_warning(&self.config) {
            let warning = ResolutionWarning {
                market_id: market_id.clone(),
                market_question: Some(status.question.clone()),
                status: info.status,
                time_remaining_secs: info
                    .time_remaining
                    .map(|d| d.num_seconds())
                    .unwrap_or(0),
                warning_threshold_secs: threshold,
                time_remaining_formatted: info.format_time_remaining(),
            };

            self.emit_resolution_warning(warning);
        }
    }

    /// Emit a resolution warning event
    fn emit_resolution_warning(&self, warning: ResolutionWarning) {
        info!(
            "Resolution warning for market {}: {} until resolution (threshold: {}s)",
            warning.market_id, warning.time_remaining_formatted, warning.warning_threshold_secs
        );

        // Create an external signal event for the warning
        let event = SystemEvent::ExternalSignal(polysniper_core::ExternalSignalEvent {
            source: polysniper_core::SignalSource::Custom {
                name: "resolution_monitor".to_string(),
            },
            signal_type: "resolution_warning".to_string(),
            content: format!(
                "Market '{}' resolution in {}",
                warning
                    .market_question
                    .as_deref()
                    .unwrap_or(&warning.market_id),
                warning.time_remaining_formatted
            ),
            market_id: Some(warning.market_id.clone()),
            keywords: vec!["resolution".to_string(), "warning".to_string()],
            metadata: serde_json::to_value(&warning).unwrap_or(serde_json::Value::Null),
            received_at: Utc::now(),
        });

        self.event_bus.publish(event);
    }

    /// Get resolution info for a specific market
    pub async fn get_resolution_info(&self, market_id: &MarketId) -> Option<ResolutionInfo> {
        let tracked_markets = self.tracked_markets.read().await;
        tracked_markets.get(market_id).cloned()
    }

    /// Get all tracked markets
    pub async fn get_all_tracked_markets(&self) -> HashMap<MarketId, ResolutionInfo> {
        let tracked_markets = self.tracked_markets.read().await;
        tracked_markets.clone()
    }

    /// Manually add a market to track
    pub async fn track_market(&self, market_id: MarketId, end_date: Option<chrono::DateTime<Utc>>) {
        let mut tracked_markets = self.tracked_markets.write().await;
        tracked_markets.insert(market_id.clone(), ResolutionInfo::new(market_id, end_date));
    }

    /// Remove a market from tracking
    pub async fn untrack_market(&self, market_id: &MarketId) {
        let mut tracked_markets = self.tracked_markets.write().await;
        tracked_markets.remove(market_id);
    }

    /// Get markets approaching resolution (within threshold)
    pub async fn get_approaching_resolution(&self) -> Vec<(MarketId, ResolutionInfo)> {
        let tracked_markets = self.tracked_markets.read().await;
        tracked_markets
            .iter()
            .filter(|(_, info)| {
                matches!(
                    info.status,
                    ResolutionStatus::Approaching | ResolutionStatus::Imminent
                )
            })
            .map(|(id, info)| (id.clone(), info.clone()))
            .collect()
    }

    /// Force a status refresh for a specific market
    pub async fn refresh_market(&self, market_id: &MarketId) -> Result<Option<ResolutionInfo>, String> {
        let status = self
            .gamma_client
            .fetch_market_status(market_id)
            .await
            .map_err(|e| e.to_string())?;

        if let Some(status) = status {
            self.process_market_status(market_id, status).await;
            Ok(self.get_resolution_info(market_id).await)
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use polysniper_core::{EventBus, Position};
    use rust_decimal::Decimal;
    use std::sync::atomic::AtomicUsize;
    use tokio::sync::broadcast;

    // Mock event bus for testing
    #[allow(dead_code)]
    struct MockEventBus {
        sender: broadcast::Sender<SystemEvent>,
        publish_count: AtomicUsize,
    }

    impl MockEventBus {
        fn new() -> Self {
            let (sender, _) = broadcast::channel(100);
            Self {
                sender,
                publish_count: AtomicUsize::new(0),
            }
        }

        fn get_publish_count(&self) -> usize {
            self.publish_count.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl EventBus for MockEventBus {
        fn publish(&self, event: SystemEvent) {
            self.publish_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = self.sender.send(event);
        }

        fn subscribe(&self) -> broadcast::Receiver<SystemEvent> {
            self.sender.subscribe()
        }

        fn subscriber_count(&self) -> usize {
            self.sender.receiver_count()
        }
    }

    // Mock state provider for testing
    #[allow(dead_code)]
    struct MockStateProvider {
        positions: Vec<Position>,
    }

    impl MockStateProvider {
        fn new(positions: Vec<Position>) -> Self {
            Self { positions }
        }
    }

    #[async_trait::async_trait]
    impl StateProvider for MockStateProvider {
        async fn get_market(&self, _market_id: &MarketId) -> Option<polysniper_core::Market> {
            None
        }

        async fn get_all_markets(&self) -> Vec<polysniper_core::Market> {
            vec![]
        }

        async fn get_orderbook(&self, _token_id: &polysniper_core::TokenId) -> Option<polysniper_core::Orderbook> {
            None
        }

        async fn get_price(&self, _token_id: &polysniper_core::TokenId) -> Option<Decimal> {
            None
        }

        async fn get_position(&self, market_id: &MarketId) -> Option<Position> {
            self.positions.iter().find(|p| &p.market_id == market_id).cloned()
        }

        async fn get_all_positions(&self) -> Vec<Position> {
            self.positions.clone()
        }

        async fn get_price_history(
            &self,
            _token_id: &polysniper_core::TokenId,
            _limit: usize,
        ) -> Vec<(chrono::DateTime<Utc>, Decimal)> {
            vec![]
        }

        async fn get_portfolio_value(&self) -> Decimal {
            Decimal::ZERO
        }

        async fn get_daily_pnl(&self) -> Decimal {
            Decimal::ZERO
        }
    }

    #[test]
    fn test_resolution_info_tracking() {
        let now = Utc::now();
        let end_date = now + Duration::hours(12);

        let info = ResolutionInfo::new("test_market".to_string(), Some(end_date));

        assert_eq!(info.market_id, "test_market");
        assert_eq!(info.status, ResolutionStatus::Approaching);
        assert!(info.time_remaining.is_some());
    }

    #[test]
    fn test_resolution_config_defaults() {
        let config = ResolutionConfig::default();

        assert!(config.enabled);
        assert_eq!(config.poll_interval_secs, 30);
        assert_eq!(config.approaching_threshold_secs, 24 * 60 * 60);
        assert_eq!(config.imminent_threshold_secs, 60 * 60);
        assert_eq!(config.warning_thresholds_secs.len(), 3);
    }
}
