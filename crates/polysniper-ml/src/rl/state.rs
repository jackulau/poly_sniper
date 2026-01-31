//! State representation for reinforcement learning execution timing.
//!
//! Defines the state space, action space, and state discretization for the Q-table.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Urgency level for order execution
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Urgency {
    /// No time pressure, can wait for optimal conditions
    Low,
    /// Normal urgency, balance speed and cost
    #[default]
    Normal,
    /// High urgency, prioritize speed over cost
    High,
    /// Critical urgency, execute immediately
    Critical,
}

impl Urgency {
    /// Convert urgency to a numeric value for discretization
    pub fn to_bucket(&self) -> u8 {
        match self {
            Urgency::Low => 0,
            Urgency::Normal => 1,
            Urgency::High => 2,
            Urgency::Critical => 3,
        }
    }
}

/// State representation for execution timing decisions.
///
/// Captures order characteristics, market microstructure, temporal features,
/// and historical context to inform the RL agent's decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    // Order characteristics
    /// Remaining size as fraction of original order (0.0 to 1.0)
    pub remaining_size: Decimal,
    /// Time elapsed as percentage of total window (0.0 to 1.0)
    pub time_elapsed_pct: f64,
    /// Urgency level of the order
    pub urgency: Urgency,

    // Market microstructure
    /// Current bid-ask spread
    pub bid_ask_spread: Decimal,
    /// Orderbook imbalance (-1.0 to 1.0, positive = more bids)
    pub orderbook_imbalance: Decimal,
    /// Recent volatility (standard deviation of returns)
    pub recent_volatility: Decimal,
    /// Queue depth at our price level
    pub queue_depth_at_price: u64,

    // Temporal features
    /// Hour of day (0-23)
    pub hour_of_day: u8,
    /// Minute of hour (0-59)
    pub minute_of_hour: u8,
    /// Seconds since last trade in this market
    pub seconds_since_last_trade: u64,

    // Historical context
    /// Recent fill rate (fills per minute)
    pub recent_fill_rate: Decimal,
    /// Average slippage in recent orders
    pub avg_slippage_last_n: Decimal,
}

impl Default for ExecutionState {
    fn default() -> Self {
        Self {
            remaining_size: Decimal::ONE,
            time_elapsed_pct: 0.0,
            urgency: Urgency::Normal,
            bid_ask_spread: dec!(0.01),
            orderbook_imbalance: Decimal::ZERO,
            recent_volatility: dec!(0.05),
            queue_depth_at_price: 0,
            hour_of_day: 12,
            minute_of_hour: 0,
            seconds_since_last_trade: 0,
            recent_fill_rate: Decimal::ONE,
            avg_slippage_last_n: Decimal::ZERO,
        }
    }
}

impl ExecutionState {
    /// Create a new execution state with the given parameters
    pub fn new(
        remaining_size: Decimal,
        time_elapsed_pct: f64,
        urgency: Urgency,
        bid_ask_spread: Decimal,
        orderbook_imbalance: Decimal,
    ) -> Self {
        Self {
            remaining_size,
            time_elapsed_pct,
            urgency,
            bid_ask_spread,
            orderbook_imbalance,
            ..Default::default()
        }
    }

    /// Convert continuous state to discrete key for Q-table lookup
    pub fn to_key(&self) -> StateKey {
        StateKey {
            remaining_bucket: self.discretize_remaining(),
            time_bucket: self.discretize_time(),
            spread_bucket: self.discretize_spread(),
            imbalance_bucket: self.discretize_imbalance(),
            hour_bucket: self.hour_of_day / 4, // 6 buckets (4-hour periods)
            urgency_bucket: self.urgency.to_bucket(),
        }
    }

    /// Discretize remaining size into buckets
    fn discretize_remaining(&self) -> u8 {
        if self.remaining_size < dec!(0.1) {
            0 // Almost done
        } else if self.remaining_size < dec!(0.25) {
            1 // Quarter remaining
        } else if self.remaining_size < dec!(0.5) {
            2 // Half remaining
        } else if self.remaining_size < dec!(0.75) {
            3 // Three quarters remaining
        } else {
            4 // Most remaining
        }
    }

    /// Discretize time elapsed into buckets
    fn discretize_time(&self) -> u8 {
        if self.time_elapsed_pct < 0.1 {
            0 // Just started
        } else if self.time_elapsed_pct < 0.25 {
            1 // Early
        } else if self.time_elapsed_pct < 0.5 {
            2 // Middle
        } else if self.time_elapsed_pct < 0.75 {
            3 // Late
        } else if self.time_elapsed_pct < 0.9 {
            4 // Very late
        } else {
            5 // Almost done
        }
    }

    /// Discretize spread into buckets
    fn discretize_spread(&self) -> u8 {
        if self.bid_ask_spread < dec!(0.005) {
            0 // Very tight
        } else if self.bid_ask_spread < dec!(0.01) {
            1 // Tight
        } else if self.bid_ask_spread < dec!(0.02) {
            2 // Normal
        } else if self.bid_ask_spread < dec!(0.05) {
            3 // Wide
        } else {
            4 // Very wide
        }
    }

    /// Discretize orderbook imbalance into buckets
    fn discretize_imbalance(&self) -> u8 {
        if self.orderbook_imbalance < dec!(-0.5) {
            0 // Strong sell pressure
        } else if self.orderbook_imbalance < dec!(-0.2) {
            1 // Moderate sell pressure
        } else if self.orderbook_imbalance < dec!(0.2) {
            2 // Balanced
        } else if self.orderbook_imbalance < dec!(0.5) {
            3 // Moderate buy pressure
        } else {
            4 // Strong buy pressure
        }
    }
}

/// Discrete state key for Q-table lookup.
///
/// This is a compact representation of the state that can be used as a HashMap key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateKey {
    /// Remaining size bucket (0-4)
    pub remaining_bucket: u8,
    /// Time elapsed bucket (0-5)
    pub time_bucket: u8,
    /// Spread bucket (0-4)
    pub spread_bucket: u8,
    /// Imbalance bucket (0-4)
    pub imbalance_bucket: u8,
    /// Hour bucket (0-5, representing 4-hour periods)
    pub hour_bucket: u8,
    /// Urgency bucket (0-3)
    pub urgency_bucket: u8,
}

impl Hash for StateKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Pack into a single u32 for efficient hashing
        let packed: u32 = (self.remaining_bucket as u32)
            | ((self.time_bucket as u32) << 4)
            | ((self.spread_bucket as u32) << 8)
            | ((self.imbalance_bucket as u32) << 12)
            | ((self.hour_bucket as u32) << 16)
            | ((self.urgency_bucket as u32) << 20);
        packed.hash(state);
    }
}

impl StateKey {
    /// Get the total number of possible state combinations
    /// This is useful for pre-allocating Q-table capacity
    pub fn state_space_size() -> usize {
        5 * 6 * 5 * 5 * 6 * 4 // remaining * time * spread * imbalance * hour * urgency
    }
}

/// Actions the agent can take for order execution timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionAction {
    /// Wait and don't place an order yet
    Wait,
    /// Place a small order (10% of remaining)
    PlaceSmall,
    /// Place a medium order (25% of remaining)
    PlaceMedium,
    /// Place a large order (50% of remaining)
    PlaceLarge,
    /// Place the entire remaining size
    PlaceAll,
    /// Cancel existing order and wait for better opportunity
    Cancel,
}

impl ExecutionAction {
    /// Get all possible actions
    pub fn all() -> &'static [ExecutionAction] {
        &[
            ExecutionAction::Wait,
            ExecutionAction::PlaceSmall,
            ExecutionAction::PlaceMedium,
            ExecutionAction::PlaceLarge,
            ExecutionAction::PlaceAll,
            ExecutionAction::Cancel,
        ]
    }

    /// Get the size fraction for this action
    pub fn size_fraction(&self) -> Decimal {
        match self {
            ExecutionAction::Wait => Decimal::ZERO,
            ExecutionAction::PlaceSmall => dec!(0.10),
            ExecutionAction::PlaceMedium => dec!(0.25),
            ExecutionAction::PlaceLarge => dec!(0.50),
            ExecutionAction::PlaceAll => Decimal::ONE,
            ExecutionAction::Cancel => Decimal::ZERO,
        }
    }

    /// Check if this action places an order
    pub fn places_order(&self) -> bool {
        matches!(
            self,
            ExecutionAction::PlaceSmall
                | ExecutionAction::PlaceMedium
                | ExecutionAction::PlaceLarge
                | ExecutionAction::PlaceAll
        )
    }

    /// Get action from index (for random selection)
    pub fn from_index(index: usize) -> Option<Self> {
        Self::all().get(index).copied()
    }

    /// Get index of this action
    pub fn to_index(&self) -> usize {
        match self {
            ExecutionAction::Wait => 0,
            ExecutionAction::PlaceSmall => 1,
            ExecutionAction::PlaceMedium => 2,
            ExecutionAction::PlaceLarge => 3,
            ExecutionAction::PlaceAll => 4,
            ExecutionAction::Cancel => 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_discretization() {
        let state = ExecutionState {
            remaining_size: dec!(0.60),
            time_elapsed_pct: 0.3,
            urgency: Urgency::Normal,
            bid_ask_spread: dec!(0.015),
            orderbook_imbalance: dec!(0.1),
            hour_of_day: 10,
            ..Default::default()
        };

        let key = state.to_key();
        assert_eq!(key.remaining_bucket, 3); // 0.5-0.75
        assert_eq!(key.time_bucket, 2); // 0.25-0.5
        assert_eq!(key.spread_bucket, 2); // 0.01-0.02
        assert_eq!(key.imbalance_bucket, 2); // -0.2 to 0.2
        assert_eq!(key.hour_bucket, 2); // Hour 10 / 4 = 2
        assert_eq!(key.urgency_bucket, 1); // Normal
    }

    #[test]
    fn test_state_key_hash_equality() {
        let state1 = ExecutionState {
            remaining_size: dec!(0.60),
            time_elapsed_pct: 0.3,
            ..Default::default()
        };

        let state2 = ExecutionState {
            remaining_size: dec!(0.55), // Different but same bucket
            time_elapsed_pct: 0.35, // Different but same bucket
            ..Default::default()
        };

        assert_eq!(state1.to_key(), state2.to_key());
    }

    #[test]
    fn test_execution_action_size_fraction() {
        assert_eq!(ExecutionAction::Wait.size_fraction(), Decimal::ZERO);
        assert_eq!(ExecutionAction::PlaceSmall.size_fraction(), dec!(0.10));
        assert_eq!(ExecutionAction::PlaceMedium.size_fraction(), dec!(0.25));
        assert_eq!(ExecutionAction::PlaceLarge.size_fraction(), dec!(0.50));
        assert_eq!(ExecutionAction::PlaceAll.size_fraction(), Decimal::ONE);
        assert_eq!(ExecutionAction::Cancel.size_fraction(), Decimal::ZERO);
    }

    #[test]
    fn test_action_places_order() {
        assert!(!ExecutionAction::Wait.places_order());
        assert!(ExecutionAction::PlaceSmall.places_order());
        assert!(ExecutionAction::PlaceMedium.places_order());
        assert!(ExecutionAction::PlaceLarge.places_order());
        assert!(ExecutionAction::PlaceAll.places_order());
        assert!(!ExecutionAction::Cancel.places_order());
    }

    #[test]
    fn test_state_space_size() {
        // Should be 5 * 6 * 5 * 5 * 6 * 4 = 18,000 states
        assert_eq!(StateKey::state_space_size(), 18000);
    }

    #[test]
    fn test_action_index_roundtrip() {
        for (i, action) in ExecutionAction::all().iter().enumerate() {
            assert_eq!(action.to_index(), i);
            assert_eq!(ExecutionAction::from_index(i), Some(*action));
        }
    }
}
