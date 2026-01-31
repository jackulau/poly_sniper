//! Experience replay buffer for reinforcement learning.
//!
//! Stores past experiences for batch learning to improve sample efficiency
//! and break correlations between consecutive samples.

use super::state::{ExecutionAction, ExecutionState};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A single experience tuple (s, a, r, s', done)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// State before taking action
    pub state: ExecutionState,
    /// Action taken
    pub action: ExecutionAction,
    /// Reward received
    pub reward: f64,
    /// State after taking action
    pub next_state: ExecutionState,
    /// Whether this was a terminal state (execution complete)
    pub done: bool,
}

impl Experience {
    /// Create a new experience tuple
    pub fn new(
        state: ExecutionState,
        action: ExecutionAction,
        reward: f64,
        next_state: ExecutionState,
        done: bool,
    ) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            done,
        }
    }
}

/// Configuration for the experience replay buffer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBufferConfig {
    /// Maximum number of experiences to store
    pub capacity: usize,
    /// Batch size for sampling
    pub batch_size: usize,
    /// Minimum experiences before learning starts
    pub min_experiences: usize,
}

impl Default for ReplayBufferConfig {
    fn default() -> Self {
        Self {
            capacity: 10000,
            batch_size: 32,
            min_experiences: 100,
        }
    }
}

/// Experience replay buffer for storing and sampling past experiences.
///
/// Uses a circular buffer (VecDeque) to efficiently add new experiences
/// and remove old ones when capacity is reached.
#[derive(Debug)]
pub struct ExperienceReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
    config: ReplayBufferConfig,
}

impl ExperienceReplayBuffer {
    /// Create a new replay buffer with default configuration
    pub fn new() -> Self {
        Self::with_config(ReplayBufferConfig::default())
    }

    /// Create a new replay buffer with custom configuration
    pub fn with_config(config: ReplayBufferConfig) -> Self {
        Self {
            buffer: VecDeque::with_capacity(config.capacity),
            capacity: config.capacity,
            config,
        }
    }

    /// Create a new replay buffer with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_config(ReplayBufferConfig {
            capacity,
            ..Default::default()
        })
    }

    /// Add a new experience to the buffer.
    ///
    /// If the buffer is full, the oldest experience is removed.
    pub fn add(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    /// Push an experience (alias for add)
    pub fn push(&mut self, experience: Experience) {
        self.add(experience);
    }

    /// Sample a random batch of experiences.
    ///
    /// Returns references to avoid cloning large amounts of data.
    pub fn sample(&self, batch_size: usize) -> Vec<&Experience> {
        let mut rng = rand::thread_rng();
        let samples: Vec<&Experience> = self.buffer.iter().collect();
        samples
            .choose_multiple(&mut rng, batch_size.min(self.buffer.len()))
            .cloned()
            .collect()
    }

    /// Sample the configured batch size
    pub fn sample_batch(&self) -> Vec<&Experience> {
        self.sample(self.config.batch_size)
    }

    /// Get the current number of experiences in the buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Check if there are enough experiences for learning
    pub fn ready_for_learning(&self) -> bool {
        self.buffer.len() >= self.config.min_experiences
    }

    /// Clear all experiences from the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get the buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get all experiences (for persistence)
    pub fn all_experiences(&self) -> impl Iterator<Item = &Experience> {
        self.buffer.iter()
    }

    /// Load experiences from a saved state
    pub fn load_experiences(&mut self, experiences: Vec<Experience>) {
        self.buffer.clear();
        for exp in experiences {
            self.add(exp);
        }
    }

    /// Get the most recent N experiences
    pub fn recent(&self, n: usize) -> Vec<&Experience> {
        self.buffer.iter().rev().take(n).collect()
    }

    /// Calculate average reward across all experiences
    pub fn average_reward(&self) -> f64 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.buffer.iter().map(|e| e.reward).sum();
        sum / self.buffer.len() as f64
    }

    /// Get action distribution in the buffer
    pub fn action_distribution(&self) -> std::collections::HashMap<ExecutionAction, usize> {
        let mut dist = std::collections::HashMap::new();
        for exp in &self.buffer {
            *dist.entry(exp.action).or_insert(0) += 1;
        }
        dist
    }
}

impl Default for ExperienceReplayBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn create_test_experience(reward: f64, done: bool) -> Experience {
        Experience {
            state: ExecutionState {
                remaining_size: dec!(0.5),
                time_elapsed_pct: 0.3,
                ..Default::default()
            },
            action: ExecutionAction::PlaceMedium,
            reward,
            next_state: ExecutionState {
                remaining_size: dec!(0.25),
                time_elapsed_pct: 0.4,
                ..Default::default()
            },
            done,
        }
    }

    #[test]
    fn test_buffer_add_and_len() {
        let mut buffer = ExperienceReplayBuffer::with_capacity(100);

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);

        buffer.add(create_test_experience(1.0, false));
        assert_eq!(buffer.len(), 1);

        buffer.add(create_test_experience(2.0, false));
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_buffer_capacity_overflow() {
        let mut buffer = ExperienceReplayBuffer::with_capacity(3);

        // Fill buffer
        buffer.add(create_test_experience(1.0, false));
        buffer.add(create_test_experience(2.0, false));
        buffer.add(create_test_experience(3.0, false));
        assert_eq!(buffer.len(), 3);

        // Add one more - should remove oldest
        buffer.add(create_test_experience(4.0, false));
        assert_eq!(buffer.len(), 3);

        // Check oldest was removed
        let rewards: Vec<f64> = buffer.all_experiences().map(|e| e.reward).collect();
        assert!(!rewards.contains(&1.0));
        assert!(rewards.contains(&2.0));
        assert!(rewards.contains(&3.0));
        assert!(rewards.contains(&4.0));
    }

    #[test]
    fn test_buffer_sample() {
        let mut buffer = ExperienceReplayBuffer::with_capacity(100);

        for i in 0..50 {
            buffer.add(create_test_experience(i as f64, false));
        }

        let sample = buffer.sample(10);
        assert_eq!(sample.len(), 10);

        // Sample more than available
        let sample = buffer.sample(100);
        assert_eq!(sample.len(), 50);
    }

    #[test]
    fn test_ready_for_learning() {
        let config = ReplayBufferConfig {
            capacity: 100,
            batch_size: 32,
            min_experiences: 10,
        };
        let mut buffer = ExperienceReplayBuffer::with_config(config);

        assert!(!buffer.ready_for_learning());

        for i in 0..9 {
            buffer.add(create_test_experience(i as f64, false));
        }
        assert!(!buffer.ready_for_learning());

        buffer.add(create_test_experience(9.0, false));
        assert!(buffer.ready_for_learning());
    }

    #[test]
    fn test_average_reward() {
        let mut buffer = ExperienceReplayBuffer::with_capacity(100);

        buffer.add(create_test_experience(1.0, false));
        buffer.add(create_test_experience(2.0, false));
        buffer.add(create_test_experience(3.0, false));

        assert_eq!(buffer.average_reward(), 2.0);
    }

    #[test]
    fn test_action_distribution() {
        let mut buffer = ExperienceReplayBuffer::with_capacity(100);

        for _ in 0..3 {
            let mut exp = create_test_experience(1.0, false);
            exp.action = ExecutionAction::Wait;
            buffer.add(exp);
        }

        for _ in 0..2 {
            let mut exp = create_test_experience(1.0, false);
            exp.action = ExecutionAction::PlaceSmall;
            buffer.add(exp);
        }

        let dist = buffer.action_distribution();
        assert_eq!(dist.get(&ExecutionAction::Wait), Some(&3));
        assert_eq!(dist.get(&ExecutionAction::PlaceSmall), Some(&2));
    }

    #[test]
    fn test_recent_experiences() {
        let mut buffer = ExperienceReplayBuffer::with_capacity(100);

        for i in 0..10 {
            buffer.add(create_test_experience(i as f64, false));
        }

        let recent = buffer.recent(3);
        assert_eq!(recent.len(), 3);

        // Most recent should be reward 9.0
        assert_eq!(recent[0].reward, 9.0);
        assert_eq!(recent[1].reward, 8.0);
        assert_eq!(recent[2].reward, 7.0);
    }

    #[test]
    fn test_clear() {
        let mut buffer = ExperienceReplayBuffer::with_capacity(100);

        buffer.add(create_test_experience(1.0, false));
        buffer.add(create_test_experience(2.0, false));
        assert_eq!(buffer.len(), 2);

        buffer.clear();
        assert!(buffer.is_empty());
    }
}
