//! Resolution Exit Strategy
//!
//! Automatically exits positions before market resolution based on configurable
//! time thresholds and risk rules.

use async_trait::async_trait;
use chrono::{Duration, Utc};
use polysniper_core::{
    ExitOrderType, ExitReason, MarketId, MarketState, MarketStateChangeEvent, OrderType, Outcome,
    PnlFloorType, Priority, ResolutionExitConfig, ResolutionInfo, Side, StateProvider, Strategy,
    StrategyError, SystemEvent, TrackedPosition, TradeSignal,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Configuration for the resolution exit strategy
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResolutionExitStrategyConfig {
    /// Base resolution exit configuration
    #[serde(flatten)]
    pub exit_config: ResolutionExitConfig,
}

/// Resolution Exit Strategy
///
/// Monitors positions approaching resolution and generates exit signals
/// based on time and P&L thresholds.
pub struct ResolutionExitStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: ResolutionExitConfig,
    /// Tracked positions with resolution timing
    tracked_positions: Arc<RwLock<HashMap<MarketId, TrackedPosition>>>,
    /// Resolution info for markets (reserved for future use)
    #[allow(dead_code)]
    resolution_info: Arc<RwLock<HashMap<MarketId, ResolutionInfo>>>,
    /// Markets that have already had exit signals generated
    exited_markets: Arc<RwLock<HashMap<MarketId, ExitReason>>>,
}

impl ResolutionExitStrategy {
    /// Create a new resolution exit strategy
    pub fn new(config: ResolutionExitConfig) -> Self {
        let enabled = config.enabled;
        Self {
            id: "resolution_exit".to_string(),
            name: "Resolution Exit Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config,
            tracked_positions: Arc::new(RwLock::new(HashMap::new())),
            resolution_info: Arc::new(RwLock::new(HashMap::new())),
            exited_markets: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create from strategy config
    pub fn from_config(config: ResolutionExitStrategyConfig) -> Self {
        Self::new(config.exit_config)
    }

    /// Convert ExitOrderType to OrderType
    fn to_order_type(exit_type: ExitOrderType) -> OrderType {
        match exit_type {
            ExitOrderType::Fok => OrderType::Fok,
            ExitOrderType::Gtc => OrderType::Gtc,
            ExitOrderType::Market => OrderType::Fok, // Market orders use FOK
        }
    }

    /// Get the order type for a specific market
    fn get_order_type(&self, market_id: &str) -> OrderType {
        // Check for market-specific override
        if let Some(override_config) = self
            .config
            .market_overrides
            .iter()
            .find(|o| o.market_id == market_id)
        {
            if let Some(order_type) = override_config.order_type {
                return Self::to_order_type(order_type);
            }
        }
        Self::to_order_type(self.config.exit_order_type)
    }

    /// Update tracked positions from state
    async fn update_tracked_positions(&self, state: &dyn StateProvider) {
        let positions = state.get_all_positions().await;
        let mut tracked = self.tracked_positions.write().await;
        let exited = self.exited_markets.read().await;

        for position in &positions {
            // Skip if already exited
            if exited.contains_key(&position.market_id) {
                continue;
            }

            // Skip zero-size positions
            if position.size.is_zero() {
                tracked.remove(&position.market_id);
                continue;
            }

            // Get market info for end date
            let market = state.get_market(&position.market_id).await;
            let end_date = market.as_ref().and_then(|m| m.end_date);

            // Check if should hold through
            let hold_through = self.config.should_hold_through(&position.market_id);

            // Update or insert tracked position
            tracked.insert(
                position.market_id.clone(),
                TrackedPosition {
                    market_id: position.market_id.clone(),
                    token_id: position.token_id.clone(),
                    size: position.size,
                    avg_price: position.avg_price,
                    unrealized_pnl: position.unrealized_pnl,
                    end_date,
                    exit_signal_generated: false,
                    exit_reason: None,
                    hold_through,
                },
            );
        }

        // Remove positions that no longer exist
        let current_market_ids: Vec<MarketId> =
            positions.iter().map(|p| p.market_id.clone()).collect();
        tracked.retain(|k, _| current_market_ids.contains(k));
    }

    /// Check if a position should exit based on time
    fn check_time_exit(&self, position: &TrackedPosition) -> Option<ExitReason> {
        if position.hold_through {
            debug!(
                market_id = %position.market_id,
                "Position flagged for hold-through, skipping time check"
            );
            return None;
        }

        let time_remaining_secs = position.time_remaining_secs()?;
        if time_remaining_secs <= 0 {
            return None; // Already resolved
        }

        let threshold = self.config.get_exit_before_secs(&position.market_id);
        if self
            .config
            .should_exit_on_time(&position.market_id, time_remaining_secs)
        {
            return Some(ExitReason::TimeThreshold {
                time_remaining_secs,
                threshold_secs: threshold,
            });
        }

        None
    }

    /// Check if a position should exit based on P&L
    fn check_pnl_exit(&self, position: &TrackedPosition) -> Option<ExitReason> {
        if position.hold_through {
            return None;
        }

        let position_value = position.position_value();

        // Check market-specific P&L floor override
        if let Some(override_config) = self
            .config
            .market_overrides
            .iter()
            .find(|o| o.market_id == position.market_id)
        {
            if let Some(floor) = override_config.pnl_floor_usd {
                if position.unrealized_pnl < floor {
                    return Some(ExitReason::PnlFloor {
                        unrealized_pnl: position.unrealized_pnl,
                        floor,
                        floor_type: PnlFloorType::Absolute,
                    });
                }
            }
        }

        // Check absolute USD floor
        if let Some(floor) = self.config.pnl_floor_usd {
            if position.unrealized_pnl < floor {
                return Some(ExitReason::PnlFloor {
                    unrealized_pnl: position.unrealized_pnl,
                    floor,
                    floor_type: PnlFloorType::Absolute,
                });
            }
        }

        // Check percentage floor
        if let Some(floor_pct) = self.config.pnl_floor_pct {
            if !position_value.is_zero() {
                let pnl_pct = (position.unrealized_pnl / position_value) * Decimal::ONE_HUNDRED;
                if pnl_pct < floor_pct {
                    return Some(ExitReason::PnlFloor {
                        unrealized_pnl: position.unrealized_pnl,
                        floor: floor_pct,
                        floor_type: PnlFloorType::Percentage,
                    });
                }
            }
        }

        None
    }

    /// Generate an exit signal for a position
    fn generate_exit_signal(
        &self,
        position: &TrackedPosition,
        reason: &ExitReason,
        current_price: Option<Decimal>,
    ) -> TradeSignal {
        let order_type = self.get_order_type(&position.market_id);
        let price = current_price.unwrap_or(position.avg_price);

        // Calculate exit price with slippage for FOK orders
        let exit_price = if order_type == OrderType::Fok {
            // For sell orders, allow slippage below current price
            price * (Decimal::ONE - self.config.max_slippage)
        } else {
            price
        };

        let size_usd = position.size * exit_price;

        TradeSignal {
            id: format!(
                "sig_rexit_{}_{}_{}",
                position.market_id,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: position.market_id.clone(),
            token_id: position.token_id.clone(),
            outcome: Outcome::Yes, // Will be determined by actual token
            side: Side::Sell,      // Always selling to exit
            price: Some(exit_price),
            size: position.size,
            size_usd,
            order_type,
            priority: Priority::High, // Resolution exits are high priority
            timestamp: Utc::now(),
            reason: format!("Resolution exit: {}", reason),
            metadata: serde_json::json!({
                "exit_reason": format!("{:?}", reason),
                "time_remaining_formatted": position
                    .time_remaining()
                    .map(format_duration)
                    .unwrap_or_else(|| "Unknown".to_string()),
                "unrealized_pnl": position.unrealized_pnl.to_string(),
                "hold_through": position.hold_through,
            }),
        }
    }

    /// Handle market state change events
    async fn handle_market_state_change(
        &self,
        event: &MarketStateChangeEvent,
        state: &dyn StateProvider,
    ) -> Vec<TradeSignal> {
        let mut signals = Vec::new();

        // Check if market is approaching resolution
        if event.new_state == MarketState::Closed || event.new_state == MarketState::Resolved {
            // Get position for this market
            if let Some(position) = state.get_position(&event.market_id).await {
                if !position.size.is_zero() {
                    // Check if already exited
                    let exited = self.exited_markets.read().await;
                    if exited.contains_key(&event.market_id) {
                        return signals;
                    }
                    drop(exited);

                    // Check if should hold through
                    if self.config.should_hold_through(&event.market_id) {
                        info!(
                            market_id = %event.market_id,
                            "Market resolving but hold-through flag is set"
                        );
                        return signals;
                    }

                    // Generate immediate exit
                    let reason = ExitReason::MarketResolved;
                    let current_price = state.get_price(&position.token_id).await;

                    let tracked = TrackedPosition {
                        market_id: position.market_id.clone(),
                        token_id: position.token_id.clone(),
                        size: position.size,
                        avg_price: position.avg_price,
                        unrealized_pnl: position.unrealized_pnl,
                        end_date: None,
                        exit_signal_generated: true,
                        exit_reason: Some(reason.clone()),
                        hold_through: false,
                    };

                    let signal = self.generate_exit_signal(&tracked, &reason, current_price);

                    warn!(
                        market_id = %event.market_id,
                        state = ?event.new_state,
                        "Market state changed to resolved/closed, generating exit signal"
                    );

                    // Record exit
                    self.exited_markets
                        .write()
                        .await
                        .insert(event.market_id.clone(), reason);

                    signals.push(signal);
                }
            }
        }

        signals
    }

    /// Check all tracked positions for exit conditions
    async fn check_all_positions(&self, state: &dyn StateProvider) -> Vec<TradeSignal> {
        let mut signals = Vec::new();

        // Update positions first
        self.update_tracked_positions(state).await;

        let tracked = self.tracked_positions.read().await;
        let exited = self.exited_markets.read().await;

        for (market_id, position) in tracked.iter() {
            // Skip if already exited
            if exited.contains_key(market_id) {
                continue;
            }

            // Skip hold-through positions
            if position.hold_through {
                continue;
            }

            // Check time-based exit first
            if let Some(reason) = self.check_time_exit(position) {
                let current_price = state.get_price(&position.token_id).await;
                let signal = self.generate_exit_signal(position, &reason, current_price);

                if self.config.log_exits {
                    info!(
                        market_id = %market_id,
                        reason = %reason,
                        size = %position.size,
                        "Generating time-based exit signal"
                    );
                }

                signals.push(signal);
                continue; // Only one exit per position
            }

            // Check P&L-based exit
            if let Some(reason) = self.check_pnl_exit(position) {
                let current_price = state.get_price(&position.token_id).await;
                let signal = self.generate_exit_signal(position, &reason, current_price);

                if self.config.log_exits {
                    info!(
                        market_id = %market_id,
                        reason = %reason,
                        size = %position.size,
                        unrealized_pnl = %position.unrealized_pnl,
                        "Generating P&L-based exit signal"
                    );
                }

                signals.push(signal);
            }
        }

        drop(tracked);
        drop(exited);

        // Record exits
        if !signals.is_empty() {
            let mut exited = self.exited_markets.write().await;
            let tracked = self.tracked_positions.read().await;

            for signal in &signals {
                if let Some(position) = tracked.get(&signal.market_id) {
                    if let Some(reason) = &position.exit_reason {
                        exited.insert(signal.market_id.clone(), reason.clone());
                    } else if let Some(reason) = self.check_time_exit(position) {
                        exited.insert(signal.market_id.clone(), reason);
                    } else if let Some(reason) = self.check_pnl_exit(position) {
                        exited.insert(signal.market_id.clone(), reason);
                    }
                }
            }
        }

        signals
    }

    /// Get tracked positions (for monitoring/debugging)
    pub async fn get_tracked_positions(&self) -> HashMap<MarketId, TrackedPosition> {
        self.tracked_positions.read().await.clone()
    }

    /// Get exited markets (for monitoring/debugging)
    pub async fn get_exited_markets(&self) -> HashMap<MarketId, ExitReason> {
        self.exited_markets.read().await.clone()
    }

    /// Clear exit history for a market (allows re-entry)
    pub async fn clear_exit(&self, market_id: &MarketId) {
        self.exited_markets.write().await.remove(market_id);
    }
}

#[async_trait]
impl Strategy for ResolutionExitStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn process_event(
        &self,
        event: &SystemEvent,
        state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        let mut signals = Vec::new();

        match event {
            // Handle market state changes
            SystemEvent::MarketStateChange(e) => {
                signals.extend(self.handle_market_state_change(e, state).await);
            }

            // On position updates, check all positions
            SystemEvent::PositionUpdate(_) => {
                signals.extend(self.check_all_positions(state).await);
            }

            // On heartbeat, check all positions periodically
            SystemEvent::Heartbeat(_) => {
                signals.extend(self.check_all_positions(state).await);
            }

            // On price changes, check P&L-based exits
            SystemEvent::PriceChange(e) => {
                let tracked = self.tracked_positions.read().await;
                if let Some(position) = tracked
                    .values()
                    .find(|p| p.token_id == e.token_id || p.market_id == e.market_id)
                {
                    let market_id = position.market_id.clone();
                    drop(tracked);

                    // Update position P&L with new price
                    let mut tracked = self.tracked_positions.write().await;
                    if let Some(pos) = tracked.get_mut(&market_id) {
                        // Recalculate unrealized P&L
                        pos.unrealized_pnl = (e.new_price - pos.avg_price) * pos.size;
                    }
                    drop(tracked);

                    // Check for P&L exit
                    signals.extend(self.check_all_positions(state).await);
                }
            }

            _ => {}
        }

        Ok(signals)
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(
            event,
            SystemEvent::MarketStateChange(_)
                | SystemEvent::PositionUpdate(_)
                | SystemEvent::Heartbeat(_)
                | SystemEvent::PriceChange(_)
        )
    }

    async fn initialize(&mut self, state: &dyn StateProvider) -> Result<(), StrategyError> {
        info!(
            strategy_id = %self.id,
            exit_before_secs = %self.config.default_exit_before_secs,
            pnl_floor_usd = ?self.config.pnl_floor_usd,
            hold_through_count = %self.config.hold_through_markets.len(),
            "Initializing resolution exit strategy"
        );

        // Load current positions
        self.update_tracked_positions(state).await;

        let tracked = self.tracked_positions.read().await;
        info!(
            positions_tracked = %tracked.len(),
            "Resolution exit strategy initialized with positions"
        );

        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }
}

/// Format a duration as human-readable string
fn format_duration(duration: Duration) -> String {
    if duration <= Duration::zero() {
        return "Resolved".to_string();
    }

    let total_secs = duration.num_seconds();
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let minutes = (total_secs % 3600) / 60;

    if days > 0 {
        format!("{}d {}h", days, hours)
    } else if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m", minutes)
    }
}

/// Generate a random suffix for signal IDs
fn rand_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{:08x}", nanos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_strategy_creation() {
        let config = ResolutionExitConfig::default();
        let strategy = ResolutionExitStrategy::new(config);

        assert_eq!(strategy.id(), "resolution_exit");
        assert_eq!(strategy.name(), "Resolution Exit Strategy");
        assert!(strategy.is_enabled());
    }

    #[test]
    fn test_check_time_exit() {
        let config = ResolutionExitConfig {
            default_exit_before_secs: 3600, // 1 hour
            ..Default::default()
        };
        let strategy = ResolutionExitStrategy::new(config);

        // Position 30 minutes from resolution - should exit
        let position_near = TrackedPosition {
            market_id: "market1".to_string(),
            token_id: "token1".to_string(),
            size: dec!(100),
            avg_price: dec!(0.5),
            unrealized_pnl: dec!(10),
            end_date: Some(Utc::now() + Duration::minutes(30)),
            exit_signal_generated: false,
            exit_reason: None,
            hold_through: false,
        };

        let exit_reason = strategy.check_time_exit(&position_near);
        assert!(exit_reason.is_some());
        assert!(matches!(
            exit_reason,
            Some(ExitReason::TimeThreshold { .. })
        ));

        // Position 2 hours from resolution - should not exit
        let position_far = TrackedPosition {
            market_id: "market2".to_string(),
            token_id: "token2".to_string(),
            size: dec!(100),
            avg_price: dec!(0.5),
            unrealized_pnl: dec!(10),
            end_date: Some(Utc::now() + Duration::hours(2)),
            exit_signal_generated: false,
            exit_reason: None,
            hold_through: false,
        };

        let exit_reason = strategy.check_time_exit(&position_far);
        assert!(exit_reason.is_none());
    }

    #[test]
    fn test_check_time_exit_hold_through() {
        let config = ResolutionExitConfig::default();
        let strategy = ResolutionExitStrategy::new(config);

        // Position near resolution but with hold-through flag
        let position = TrackedPosition {
            market_id: "market1".to_string(),
            token_id: "token1".to_string(),
            size: dec!(100),
            avg_price: dec!(0.5),
            unrealized_pnl: dec!(10),
            end_date: Some(Utc::now() + Duration::minutes(30)),
            exit_signal_generated: false,
            exit_reason: None,
            hold_through: true, // Should skip exit
        };

        let exit_reason = strategy.check_time_exit(&position);
        assert!(exit_reason.is_none());
    }

    #[test]
    fn test_check_pnl_exit_absolute() {
        let config = ResolutionExitConfig {
            pnl_floor_usd: Some(dec!(-50)),
            ..Default::default()
        };
        let strategy = ResolutionExitStrategy::new(config);

        // Position with P&L below floor
        let position_loss = TrackedPosition {
            market_id: "market1".to_string(),
            token_id: "token1".to_string(),
            size: dec!(100),
            avg_price: dec!(0.5),
            unrealized_pnl: dec!(-75), // Below -50 floor
            end_date: Some(Utc::now() + Duration::hours(24)),
            exit_signal_generated: false,
            exit_reason: None,
            hold_through: false,
        };

        let exit_reason = strategy.check_pnl_exit(&position_loss);
        assert!(exit_reason.is_some());
        assert!(matches!(
            exit_reason,
            Some(ExitReason::PnlFloor {
                floor_type: PnlFloorType::Absolute,
                ..
            })
        ));

        // Position with P&L above floor
        let position_ok = TrackedPosition {
            unrealized_pnl: dec!(-25), // Above -50 floor
            ..position_loss
        };

        let exit_reason = strategy.check_pnl_exit(&position_ok);
        assert!(exit_reason.is_none());
    }

    #[test]
    fn test_check_pnl_exit_percentage() {
        let config = ResolutionExitConfig {
            pnl_floor_pct: Some(dec!(-10)), // -10%
            ..Default::default()
        };
        let strategy = ResolutionExitStrategy::new(config);

        // Position with -20% P&L (100 size * 0.5 avg = 50 value, -10 P&L = -20%)
        let position_loss = TrackedPosition {
            market_id: "market1".to_string(),
            token_id: "token1".to_string(),
            size: dec!(100),
            avg_price: dec!(0.5),
            unrealized_pnl: dec!(-10), // -20% of 50 position value
            end_date: Some(Utc::now() + Duration::hours(24)),
            exit_signal_generated: false,
            exit_reason: None,
            hold_through: false,
        };

        let exit_reason = strategy.check_pnl_exit(&position_loss);
        assert!(exit_reason.is_some());
        assert!(matches!(
            exit_reason,
            Some(ExitReason::PnlFloor {
                floor_type: PnlFloorType::Percentage,
                ..
            })
        ));
    }

    #[test]
    fn test_generate_exit_signal() {
        let config = ResolutionExitConfig::default();
        let strategy = ResolutionExitStrategy::new(config);

        let position = TrackedPosition {
            market_id: "market1".to_string(),
            token_id: "token1".to_string(),
            size: dec!(100),
            avg_price: dec!(0.5),
            unrealized_pnl: dec!(10),
            end_date: Some(Utc::now() + Duration::minutes(30)),
            exit_signal_generated: false,
            exit_reason: None,
            hold_through: false,
        };

        let reason = ExitReason::TimeThreshold {
            time_remaining_secs: 1800,
            threshold_secs: 3600,
        };

        let signal = strategy.generate_exit_signal(&position, &reason, Some(dec!(0.55)));

        assert_eq!(signal.strategy_id, "resolution_exit");
        assert_eq!(signal.market_id, "market1");
        assert_eq!(signal.side, Side::Sell);
        assert_eq!(signal.priority, Priority::High);
        assert_eq!(signal.size, dec!(100));
        assert!(signal.id.starts_with("sig_rexit_"));
    }

    #[test]
    fn test_order_type_conversion() {
        assert_eq!(
            ResolutionExitStrategy::to_order_type(ExitOrderType::Fok),
            OrderType::Fok
        );
        assert_eq!(
            ResolutionExitStrategy::to_order_type(ExitOrderType::Gtc),
            OrderType::Gtc
        );
        assert_eq!(
            ResolutionExitStrategy::to_order_type(ExitOrderType::Market),
            OrderType::Fok
        );
    }
}
