//! Q-Learning agent for execution timing optimization.
//!
//! Implements tabular Q-learning with epsilon-greedy exploration and
//! experience replay for learning optimal order execution timing.

use super::replay_buffer::{Experience, ExperienceReplayBuffer, ReplayBufferConfig};
use super::reward::RewardConfig;
use super::state::{ExecutionAction, ExecutionState, StateKey};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Configuration for the Q-Learning agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlConfig {
    /// Whether RL execution is enabled
    pub enabled: bool,
    /// Operating mode
    pub mode: RlMode,
    /// Learning rate (alpha) - how quickly to update Q-values
    pub alpha: f64,
    /// Discount factor (gamma) - importance of future rewards
    pub gamma: f64,
    /// Exploration rate (epsilon) - probability of random action
    pub epsilon: f64,
    /// Minimum exploration rate
    pub epsilon_min: f64,
    /// Exploration decay rate per episode
    pub epsilon_decay: f64,
    /// Replay buffer configuration
    pub replay_buffer: ReplayBufferConfig,
    /// Reward calculation configuration
    pub reward: RewardConfig,
    /// How often to update from replay buffer (in steps)
    pub update_frequency: usize,
    /// Path to save/load the model
    pub model_path: String,
    /// How often to save the model (in seconds)
    pub save_interval_secs: u64,
}

impl Default for RlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: RlMode::Training,
            alpha: 0.1,
            gamma: 0.99,
            epsilon: 0.3,
            epsilon_min: 0.05,
            epsilon_decay: 0.995,
            replay_buffer: ReplayBufferConfig::default(),
            reward: RewardConfig::default(),
            update_frequency: 10,
            model_path: "data/rl_execution_model.json".to_string(),
            save_interval_secs: 300,
        }
    }
}

/// Operating mode for the RL agent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RlMode {
    /// Training mode - higher exploration, learning enabled
    Training,
    /// Production mode - low exploration, learning enabled
    Production,
    /// Evaluation mode - no exploration, no learning
    Evaluation,
}

/// Statistics about the agent's performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentStats {
    /// Total number of actions taken
    pub total_actions: u64,
    /// Number of exploration (random) actions
    pub exploration_actions: u64,
    /// Number of exploitation (greedy) actions
    pub exploitation_actions: u64,
    /// Total reward accumulated
    pub total_reward: f64,
    /// Number of Q-value updates
    pub num_updates: u64,
    /// Current exploration rate
    pub current_epsilon: f64,
    /// Number of unique states visited
    pub states_visited: usize,
    /// Average Q-value across all state-action pairs
    pub avg_q_value: f64,
}

/// Q-Table agent for learning optimal execution timing.
///
/// Uses tabular Q-learning with a HashMap-based Q-table, epsilon-greedy
/// exploration, and experience replay for improved sample efficiency.
pub struct QTableAgent {
    /// State-action value table
    q_table: Arc<RwLock<HashMap<StateKey, HashMap<ExecutionAction, f64>>>>,
    /// Configuration
    config: RlConfig,
    /// Current exploration rate
    epsilon: Arc<RwLock<f64>>,
    /// Experience replay buffer
    replay_buffer: Arc<RwLock<ExperienceReplayBuffer>>,
    /// Step counter for update frequency
    step_counter: Arc<RwLock<u64>>,
    /// Statistics
    stats: Arc<RwLock<AgentStats>>,
}

impl QTableAgent {
    /// Create a new Q-table agent with default configuration
    pub fn new() -> Self {
        Self::with_config(RlConfig::default())
    }

    /// Create a new Q-table agent with custom configuration
    pub fn with_config(config: RlConfig) -> Self {
        let epsilon = config.epsilon;
        let replay_buffer = ExperienceReplayBuffer::with_config(config.replay_buffer.clone());

        Self {
            q_table: Arc::new(RwLock::new(HashMap::new())),
            config,
            epsilon: Arc::new(RwLock::new(epsilon)),
            replay_buffer: Arc::new(RwLock::new(replay_buffer)),
            step_counter: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(AgentStats::default())),
        }
    }

    /// Select an action using epsilon-greedy policy
    pub async fn select_action(&self, state: &ExecutionState) -> ExecutionAction {
        let mut rng = rand::thread_rng();
        let epsilon = *self.epsilon.read().await;

        let action = if rng.gen::<f64>() < epsilon {
            // Explore: random action
            let idx = rng.gen_range(0..ExecutionAction::all().len());
            let action = ExecutionAction::from_index(idx).unwrap_or(ExecutionAction::Wait);

            let mut stats = self.stats.write().await;
            stats.exploration_actions += 1;
            stats.total_actions += 1;

            debug!(action = ?action, epsilon = epsilon, "Exploration: random action");
            action
        } else {
            // Exploit: best known action
            let action = self.best_action(state).await;

            let mut stats = self.stats.write().await;
            stats.exploitation_actions += 1;
            stats.total_actions += 1;

            debug!(action = ?action, "Exploitation: best action");
            action
        };

        action
    }

    /// Get the best action for a state (greedy policy)
    pub async fn best_action(&self, state: &ExecutionState) -> ExecutionAction {
        let state_key = state.to_key();
        let q_table = self.q_table.read().await;

        if let Some(action_values) = q_table.get(&state_key) {
            action_values
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(action, _)| *action)
                .unwrap_or(ExecutionAction::Wait)
        } else {
            // No data for this state, use default heuristic
            self.default_action(state)
        }
    }

    /// Default action based on heuristics when no Q-values available
    fn default_action(&self, state: &ExecutionState) -> ExecutionAction {
        // If very late in execution window, place remaining
        if state.time_elapsed_pct > 0.9 {
            return ExecutionAction::PlaceAll;
        }

        // If spread is tight and have time, place medium
        if state.bid_ask_spread < rust_decimal_macros::dec!(0.01) && state.time_elapsed_pct < 0.7 {
            return ExecutionAction::PlaceMedium;
        }

        // Default to waiting if conditions aren't favorable
        ExecutionAction::Wait
    }

    /// Update Q-value after receiving a reward (single update)
    pub async fn update(
        &self,
        state: &ExecutionState,
        action: ExecutionAction,
        reward: f64,
        next_state: &ExecutionState,
        done: bool,
    ) {
        // Don't learn in evaluation mode
        if self.config.mode == RlMode::Evaluation {
            return;
        }

        // Store experience
        {
            let mut buffer = self.replay_buffer.write().await;
            buffer.add(Experience::new(
                state.clone(),
                action,
                reward,
                next_state.clone(),
                done,
            ));
        }

        // Increment step counter and check if we should do batch update
        let should_update = {
            let mut counter = self.step_counter.write().await;
            *counter += 1;
            *counter % self.config.update_frequency as u64 == 0
        };

        if should_update {
            self.batch_update().await;
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_reward += reward;
    }

    /// Perform batch update from replay buffer
    async fn batch_update(&self) {
        // Sample and clone experiences while holding the lock briefly
        let batch: Vec<Experience> = {
            let buffer = self.replay_buffer.read().await;
            if !buffer.ready_for_learning() {
                return;
            }
            buffer.sample_batch().into_iter().cloned().collect()
        };

        for exp in batch {
            self.update_q_value(
                &exp.state,
                exp.action,
                exp.reward,
                &exp.next_state,
                exp.done,
            )
            .await;
        }

        // Decay exploration rate
        self.decay_epsilon().await;
    }

    /// Update a single Q-value using the Bellman equation
    async fn update_q_value(
        &self,
        state: &ExecutionState,
        action: ExecutionAction,
        reward: f64,
        next_state: &ExecutionState,
        done: bool,
    ) {
        let state_key = state.to_key();
        let next_key = next_state.to_key();

        let mut q_table = self.q_table.write().await;

        // Get current Q-value
        let current_q = q_table
            .get(&state_key)
            .and_then(|actions| actions.get(&action))
            .copied()
            .unwrap_or(0.0);

        // Get max Q-value for next state (0 if terminal)
        let max_next_q = if done {
            0.0
        } else {
            q_table
                .get(&next_key)
                .map(|actions| {
                    actions
                        .values()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .unwrap_or(0.0)
                .max(0.0) // Treat unknown states as 0
        };

        // Q(s,a) = Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
        let new_q =
            current_q + self.config.alpha * (reward + self.config.gamma * max_next_q - current_q);

        q_table
            .entry(state_key)
            .or_insert_with(HashMap::new)
            .insert(action, new_q);

        // Update stats
        drop(q_table);
        let mut stats = self.stats.write().await;
        stats.num_updates += 1;
    }

    /// Decay the exploration rate
    async fn decay_epsilon(&self) {
        let mut epsilon = self.epsilon.write().await;
        *epsilon = (*epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);

        let mut stats = self.stats.write().await;
        stats.current_epsilon = *epsilon;
    }

    /// Set the exploration rate
    pub async fn set_epsilon(&self, new_epsilon: f64) {
        let mut epsilon = self.epsilon.write().await;
        *epsilon = new_epsilon.clamp(self.config.epsilon_min, 1.0);
    }

    /// Get the current exploration rate
    pub async fn get_epsilon(&self) -> f64 {
        *self.epsilon.read().await
    }

    /// Enable training mode
    pub async fn enable_training_mode(&self) {
        self.set_epsilon(0.3).await;
        info!("RL agent switched to training mode (epsilon=0.3)");
    }

    /// Enable production mode
    pub async fn enable_production_mode(&self) {
        self.set_epsilon(0.05).await;
        info!("RL agent switched to production mode (epsilon=0.05)");
    }

    /// Get the Q-table for persistence or analysis
    pub async fn get_q_table(&self) -> HashMap<StateKey, HashMap<ExecutionAction, f64>> {
        self.q_table.read().await.clone()
    }

    /// Load a Q-table from saved state
    pub async fn load_q_table(&self, q_table: HashMap<StateKey, HashMap<ExecutionAction, f64>>) {
        let mut table = self.q_table.write().await;
        *table = q_table;
        info!(num_states = table.len(), "Loaded Q-table");
    }

    /// Get agent statistics
    pub async fn get_stats(&self) -> AgentStats {
        let mut stats = self.stats.read().await.clone();
        let q_table = self.q_table.read().await;

        stats.states_visited = q_table.len();

        // Calculate average Q-value
        let mut total_q = 0.0;
        let mut count = 0;
        for action_values in q_table.values() {
            for q in action_values.values() {
                total_q += q;
                count += 1;
            }
        }
        stats.avg_q_value = if count > 0 {
            total_q / count as f64
        } else {
            0.0
        };

        stats
    }

    /// Get the Q-value for a specific state-action pair
    pub async fn get_q_value(&self, state: &ExecutionState, action: ExecutionAction) -> f64 {
        let state_key = state.to_key();
        let q_table = self.q_table.read().await;

        q_table
            .get(&state_key)
            .and_then(|actions| actions.get(&action))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get all Q-values for a state
    pub async fn get_state_values(
        &self,
        state: &ExecutionState,
    ) -> HashMap<ExecutionAction, f64> {
        let state_key = state.to_key();
        let q_table = self.q_table.read().await;

        q_table
            .get(&state_key)
            .cloned()
            .unwrap_or_else(HashMap::new)
    }

    /// Get the action distribution (frequency of each action being best)
    pub async fn get_action_distribution(&self) -> HashMap<ExecutionAction, usize> {
        let q_table = self.q_table.read().await;
        let mut distribution = HashMap::new();

        for action_values in q_table.values() {
            if let Some((best_action, _)) = action_values
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                *distribution.entry(*best_action).or_insert(0) += 1;
            }
        }

        distribution
    }

    /// Export the model to JSON
    pub async fn export_to_json(&self) -> serde_json::Result<String> {
        let q_table = self.q_table.read().await;
        let epsilon = *self.epsilon.read().await;
        let stats = self.stats.read().await.clone();

        let export = ModelExport {
            q_table: q_table.clone(),
            epsilon,
            stats,
            config: self.config.clone(),
        };

        serde_json::to_string_pretty(&export)
    }

    /// Import the model from JSON
    pub async fn import_from_json(&self, json: &str) -> serde_json::Result<()> {
        let import: ModelExport = serde_json::from_str(json)?;

        let mut q_table = self.q_table.write().await;
        *q_table = import.q_table;

        let mut epsilon = self.epsilon.write().await;
        *epsilon = import.epsilon;

        let mut stats = self.stats.write().await;
        *stats = import.stats;

        info!(num_states = q_table.len(), "Imported Q-table from JSON");
        Ok(())
    }
}

impl Default for QTableAgent {
    fn default() -> Self {
        Self::new()
    }
}

/// Model export format for persistence
#[derive(Serialize, Deserialize)]
struct ModelExport {
    q_table: HashMap<StateKey, HashMap<ExecutionAction, f64>>,
    epsilon: f64,
    stats: AgentStats,
    config: RlConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn create_test_state(remaining: f64, time_pct: f64) -> ExecutionState {
        ExecutionState {
            remaining_size: rust_decimal::Decimal::from_f64_retain(remaining).unwrap(),
            time_elapsed_pct: time_pct,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_agent_creation() {
        let agent = QTableAgent::new();
        let stats = agent.get_stats().await;

        assert_eq!(stats.total_actions, 0);
        assert_eq!(stats.states_visited, 0);
    }

    #[tokio::test]
    async fn test_select_action() {
        let agent = QTableAgent::new();
        let state = create_test_state(0.5, 0.3);

        // Should be able to select an action
        let action = agent.select_action(&state).await;
        assert!(ExecutionAction::all().contains(&action));

        // Stats should be updated
        let stats = agent.get_stats().await;
        assert_eq!(stats.total_actions, 1);
    }

    #[tokio::test]
    async fn test_q_value_update() {
        let config = RlConfig {
            alpha: 0.5, // High learning rate for testing
            gamma: 0.9,
            epsilon: 0.0, // No exploration
            update_frequency: 1,
            replay_buffer: ReplayBufferConfig {
                capacity: 100,
                batch_size: 1,
                min_experiences: 1,
            },
            ..Default::default()
        };

        let agent = QTableAgent::with_config(config);
        let state = create_test_state(0.5, 0.3);
        let next_state = create_test_state(0.25, 0.4);

        // Initial Q-value should be 0
        let initial_q = agent
            .get_q_value(&state, ExecutionAction::PlaceMedium)
            .await;
        assert_eq!(initial_q, 0.0);

        // Update with positive reward
        agent
            .update(&state, ExecutionAction::PlaceMedium, 10.0, &next_state, false)
            .await;

        // Q-value should be updated
        let updated_q = agent
            .get_q_value(&state, ExecutionAction::PlaceMedium)
            .await;
        assert!(updated_q > 0.0);
    }

    #[tokio::test]
    async fn test_epsilon_decay() {
        let config = RlConfig {
            epsilon: 1.0,
            epsilon_min: 0.1,
            epsilon_decay: 0.5,
            ..Default::default()
        };

        let agent = QTableAgent::with_config(config);

        let initial = agent.get_epsilon().await;
        assert_eq!(initial, 1.0);

        agent.decay_epsilon().await;

        let after_decay = agent.get_epsilon().await;
        assert_eq!(after_decay, 0.5);

        // Multiple decays
        agent.decay_epsilon().await;
        agent.decay_epsilon().await;

        let after_multiple = agent.get_epsilon().await;
        assert!(after_multiple >= 0.1); // Should not go below min
    }

    #[tokio::test]
    async fn test_best_action_with_data() {
        // Test directly setting Q-values to verify best_action logic
        let agent = QTableAgent::new();
        let state = create_test_state(0.5, 0.3);
        let state_key = state.to_key();

        // Directly set Q-values in the table
        {
            let mut q_table = agent.q_table.write().await;
            let mut action_values = HashMap::new();
            action_values.insert(ExecutionAction::Wait, 1.0);
            action_values.insert(ExecutionAction::PlaceSmall, 5.0);
            action_values.insert(ExecutionAction::PlaceMedium, 10.0);
            q_table.insert(state_key, action_values);
        }

        // Best action should be PlaceMedium (highest Q-value)
        let best = agent.best_action(&state).await;
        assert_eq!(best, ExecutionAction::PlaceMedium);
    }

    #[tokio::test]
    async fn test_mode_switching() {
        let agent = QTableAgent::new();

        agent.enable_training_mode().await;
        let training_epsilon = agent.get_epsilon().await;
        assert_eq!(training_epsilon, 0.3);

        agent.enable_production_mode().await;
        let prod_epsilon = agent.get_epsilon().await;
        assert_eq!(prod_epsilon, 0.05);
    }

    #[tokio::test]
    async fn test_export_import() {
        let agent = QTableAgent::new();
        let state = create_test_state(0.5, 0.3);
        let next_state = create_test_state(0.25, 0.4);

        // Add some data
        agent
            .update(
                &state,
                ExecutionAction::PlaceMedium,
                5.0,
                &next_state,
                true,
            )
            .await;

        // Export
        let json = agent.export_to_json().await.unwrap();

        // Create new agent and import
        let agent2 = QTableAgent::new();
        agent2.import_from_json(&json).await.unwrap();

        // Should have same Q-table
        let q1 = agent.get_q_table().await;
        let q2 = agent2.get_q_table().await;
        assert_eq!(q1.len(), q2.len());
    }

    #[tokio::test]
    async fn test_action_distribution() {
        // Test directly setting Q-values to verify action distribution logic
        let agent = QTableAgent::new();

        // Set up multiple states with different best actions
        let state1 = create_test_state(0.9, 0.1);
        let state2 = create_test_state(0.5, 0.5);
        let state3 = create_test_state(0.1, 0.9);

        {
            let mut q_table = agent.q_table.write().await;

            // State 1: Wait is best
            let mut actions1 = HashMap::new();
            actions1.insert(ExecutionAction::Wait, 10.0);
            actions1.insert(ExecutionAction::PlaceSmall, 1.0);
            q_table.insert(state1.to_key(), actions1);

            // State 2: PlaceMedium is best
            let mut actions2 = HashMap::new();
            actions2.insert(ExecutionAction::Wait, 1.0);
            actions2.insert(ExecutionAction::PlaceMedium, 10.0);
            q_table.insert(state2.to_key(), actions2);

            // State 3: PlaceAll is best
            let mut actions3 = HashMap::new();
            actions3.insert(ExecutionAction::Wait, 1.0);
            actions3.insert(ExecutionAction::PlaceAll, 10.0);
            q_table.insert(state3.to_key(), actions3);
        }

        let dist = agent.get_action_distribution().await;
        assert_eq!(dist.get(&ExecutionAction::Wait), Some(&1));
        assert_eq!(dist.get(&ExecutionAction::PlaceMedium), Some(&1));
        assert_eq!(dist.get(&ExecutionAction::PlaceAll), Some(&1));
    }
}
