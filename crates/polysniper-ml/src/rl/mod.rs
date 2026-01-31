//! Reinforcement Learning Module
//!
//! Implements Q-Learning for optimal order execution timing.

mod q_learning;
mod replay_buffer;
mod reward;
mod state;

pub use q_learning::{AgentStats, QTableAgent, RlConfig};
pub use replay_buffer::{Experience, ExperienceReplayBuffer};
pub use reward::ExecutionReward;
pub use state::{ExecutionAction, ExecutionState, StateKey, Urgency};
