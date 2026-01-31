---
id: rl-execution
name: Reinforcement Learning for Optimal Execution Timing
wave: 1
priority: 2
dependencies: []
estimated_hours: 7
tags: [ml, execution, reinforcement-learning]
---

## Objective

Implement a reinforcement learning system that learns optimal execution timing for orders, adapting to market microstructure patterns to minimize slippage and maximize fill rates.

## Context

The existing TWAP/VWAP execution algorithms in `polysniper-execution` use fixed or randomized timing. They don't learn from:
- Historical fill patterns (which times have better fill rates)
- Queue position dynamics
- Orderbook imbalance patterns before execution
- Price impact of previous orders

An RL agent can learn a policy that optimizes when to place orders based on learned patterns.

## Implementation

### 1. Define State, Action, Reward (`crates/polysniper-ml/src/rl/mod.rs`)

```rust
/// State representation for execution timing
#[derive(Clone)]
pub struct ExecutionState {
    // Order characteristics
    pub remaining_size: Decimal,
    pub time_elapsed_pct: f64,
    pub urgency: Urgency,

    // Market microstructure
    pub bid_ask_spread: Decimal,
    pub orderbook_imbalance: Decimal,
    pub recent_volatility: Decimal,
    pub queue_depth_at_price: u64,

    // Temporal features
    pub hour_of_day: u8,
    pub minute_of_hour: u8,
    pub seconds_since_last_trade: u64,

    // Historical context
    pub recent_fill_rate: Decimal,
    pub avg_slippage_last_n: Decimal,
}

/// Actions the agent can take
#[derive(Clone, Copy, PartialEq)]
pub enum ExecutionAction {
    Wait,           // Don't place order yet
    PlaceSmall,     // Place 10% of remaining
    PlaceMedium,    // Place 25% of remaining
    PlaceLarge,     // Place 50% of remaining
    PlaceAll,       // Place full remaining size
    Cancel,         // Cancel and wait for better opportunity
}

/// Reward signal after taking action
pub struct ExecutionReward {
    pub fill_rate_reward: f64,      // Higher fill rate = better
    pub slippage_penalty: f64,      // Lower slippage = better
    pub time_penalty: f64,          // Penalty for taking too long
    pub market_impact_penalty: f64, // Penalty for moving price
}

impl ExecutionReward {
    pub fn total(&self) -> f64 {
        self.fill_rate_reward
        - self.slippage_penalty
        - self.time_penalty
        - self.market_impact_penalty
    }
}
```

### 2. Implement Q-Learning Agent (`crates/polysniper-ml/src/rl/q_learning.rs`)

```rust
pub struct QTableAgent {
    // State-action value table
    q_table: Arc<RwLock<HashMap<StateKey, HashMap<ExecutionAction, f64>>>>,
    // Learning parameters
    alpha: f64,        // Learning rate
    gamma: f64,        // Discount factor
    epsilon: f64,      // Exploration rate
    // Exploration decay
    epsilon_min: f64,
    epsilon_decay: f64,
    // Persistence
    db: Arc<Database>,
}

impl QTableAgent {
    /// Choose action using epsilon-greedy policy
    pub async fn select_action(&self, state: &ExecutionState) -> ExecutionAction {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < self.epsilon {
            // Explore: random action
            self.random_action()
        } else {
            // Exploit: best known action
            self.best_action(state).await
        }
    }

    /// Update Q-value after receiving reward
    pub async fn update(
        &self,
        state: &ExecutionState,
        action: ExecutionAction,
        reward: f64,
        next_state: &ExecutionState,
    ) {
        let state_key = state.to_key();
        let next_key = next_state.to_key();

        let mut q_table = self.q_table.write().await;

        // Q(s,a) = Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
        let current_q = q_table
            .get(&state_key)
            .and_then(|actions| actions.get(&action))
            .copied()
            .unwrap_or(0.0);

        let max_next_q = q_table
            .get(&next_key)
            .map(|actions| actions.values().cloned().fold(f64::NEG_INFINITY, f64::max))
            .unwrap_or(0.0);

        let new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q);

        q_table
            .entry(state_key)
            .or_insert_with(HashMap::new)
            .insert(action, new_q);

        // Decay exploration
        self.decay_epsilon();
    }
}
```

### 3. State Discretization for Q-Table

```rust
impl ExecutionState {
    /// Convert continuous state to discrete key for Q-table
    pub fn to_key(&self) -> StateKey {
        StateKey {
            remaining_bucket: self.discretize_remaining(),
            time_bucket: self.discretize_time(),
            spread_bucket: self.discretize_spread(),
            imbalance_bucket: self.discretize_imbalance(),
            hour_bucket: self.hour_of_day / 4, // 6 buckets
        }
    }

    fn discretize_remaining(&self) -> u8 {
        match self.remaining_size {
            x if x < dec!(0.1) => 0,
            x if x < dec!(0.25) => 1,
            x if x < dec!(0.5) => 2,
            x if x < dec!(0.75) => 3,
            _ => 4,
        }
    }

    // ... similar for other dimensions
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct StateKey {
    remaining_bucket: u8,
    time_bucket: u8,
    spread_bucket: u8,
    imbalance_bucket: u8,
    hour_bucket: u8,
}
```

### 4. Experience Replay Buffer (`crates/polysniper-ml/src/rl/replay_buffer.rs`)

```rust
pub struct ExperienceReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

pub struct Experience {
    pub state: ExecutionState,
    pub action: ExecutionAction,
    pub reward: f64,
    pub next_state: ExecutionState,
    pub done: bool,
}

impl ExperienceReplayBuffer {
    pub fn add(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Experience> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.buffer.iter()
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect()
    }
}
```

### 5. Integration with TWAP/VWAP (`crates/polysniper-execution/src/rl_executor.rs`)

```rust
pub struct RlEnhancedExecutor {
    // Underlying executor (TWAP or VWAP)
    base_executor: Box<dyn ExecutionAlgorithm>,
    // RL agent for timing
    rl_agent: Arc<QTableAgent>,
    // State builder
    state_builder: StateBuilder,
    // Config
    config: RlExecutorConfig,
}

#[async_trait]
impl ExecutionAlgorithm for RlEnhancedExecutor {
    async fn get_next_order(
        &self,
        parent_id: &str,
        context: &ExecutionContext,
    ) -> Option<Order> {
        // Build current state
        let state = self.state_builder.build(context).await;

        // Get RL decision
        let action = self.rl_agent.select_action(&state).await;

        match action {
            ExecutionAction::Wait => None,
            ExecutionAction::PlaceSmall => self.create_order(context, dec!(0.10)).await,
            ExecutionAction::PlaceMedium => self.create_order(context, dec!(0.25)).await,
            ExecutionAction::PlaceLarge => self.create_order(context, dec!(0.50)).await,
            ExecutionAction::PlaceAll => self.create_order(context, dec!(1.0)).await,
            ExecutionAction::Cancel => {
                // Cancel existing and reschedule
                self.cancel_and_reschedule(parent_id).await;
                None
            }
        }
    }

    async fn record_fill(
        &self,
        parent_id: &str,
        order_id: &str,
        filled: Decimal,
        price: Decimal,
        context: &ExecutionContext,
    ) {
        // Calculate reward
        let reward = self.calculate_reward(context, filled, price);

        // Get state transition
        let prev_state = self.state_builder.get_previous_state(parent_id);
        let curr_state = self.state_builder.build(context).await;

        // Update RL agent
        if let Some(prev) = prev_state {
            let action = self.last_action.get(parent_id);
            if let Some(action) = action {
                self.rl_agent.update(&prev, *action, reward.total(), &curr_state).await;
            }
        }
    }
}
```

### 6. Reward Calculation

```rust
impl RlEnhancedExecutor {
    fn calculate_reward(&self, context: &ExecutionContext, filled: Decimal, price: Decimal) -> ExecutionReward {
        // Fill rate reward: higher is better
        let fill_rate = filled / context.order_size;
        let fill_rate_reward = fill_rate.to_f64().unwrap() * 10.0;

        // Slippage penalty: deviation from mid price
        let mid_price = context.mid_price;
        let slippage = ((price - mid_price) / mid_price).abs();
        let slippage_penalty = slippage.to_f64().unwrap() * 100.0;

        // Time penalty: exponential cost for delay
        let time_pct = context.time_elapsed_pct;
        let time_penalty = if time_pct > 0.9 { (time_pct - 0.9) * 50.0 } else { 0.0 };

        // Market impact: price moved after our order
        let impact = context.price_after_fill.map(|p| {
            ((p - price) / price).abs().to_f64().unwrap()
        }).unwrap_or(0.0);
        let market_impact_penalty = impact * 50.0;

        ExecutionReward {
            fill_rate_reward,
            slippage_penalty,
            time_penalty,
            market_impact_penalty,
        }
    }
}
```

### 7. Training Mode and Evaluation

```rust
impl RlEnhancedExecutor {
    /// Run in training mode (higher exploration)
    pub fn enable_training_mode(&mut self) {
        self.rl_agent.set_epsilon(0.3); // 30% exploration
    }

    /// Run in production mode (exploitation)
    pub fn enable_production_mode(&mut self) {
        self.rl_agent.set_epsilon(0.05); // 5% exploration
    }

    /// Evaluate current policy performance
    pub async fn evaluate_policy(&self) -> PolicyMetrics {
        let q_table = self.rl_agent.get_q_table().await;

        PolicyMetrics {
            num_states_visited: q_table.len(),
            avg_q_value: self.calculate_avg_q(&q_table),
            action_distribution: self.calculate_action_distribution(&q_table),
            estimated_improvement: self.estimate_improvement_over_baseline(),
        }
    }
}
```

### 8. Configuration (`config/rl_execution.toml`)

```toml
[rl_execution]
enabled = true
mode = "training"  # training, production, evaluation

[rl_execution.q_learning]
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
epsilon = 0.3        # Initial exploration rate
epsilon_min = 0.05   # Minimum exploration rate
epsilon_decay = 0.995

[rl_execution.state]
spread_buckets = 5
imbalance_buckets = 5
time_buckets = 10
remaining_buckets = 5

[rl_execution.reward]
fill_weight = 10.0
slippage_weight = 100.0
time_weight = 50.0
impact_weight = 50.0

[rl_execution.replay_buffer]
capacity = 10000
batch_size = 32
update_frequency = 10

[rl_execution.persistence]
save_interval_secs = 300
model_path = "data/rl_execution_model.json"
```

## Acceptance Criteria

- [ ] Q-Learning agent implemented with epsilon-greedy policy
- [ ] State representation captures market microstructure
- [ ] Reward function balances fill rate, slippage, and timing
- [ ] Experience replay buffer for stable learning
- [ ] Integration with TWAP/VWAP executors
- [ ] Training/production mode switching
- [ ] Persistence of learned Q-table
- [ ] Unit tests for Q-learning update logic
- [ ] Integration tests for execution flow
- [ ] All existing tests pass

## Files to Create/Modify

**Create:**
- `crates/polysniper-ml/src/rl/mod.rs` - RL module entry
- `crates/polysniper-ml/src/rl/q_learning.rs` - Q-learning implementation
- `crates/polysniper-ml/src/rl/state.rs` - State representation
- `crates/polysniper-ml/src/rl/replay_buffer.rs` - Experience replay
- `crates/polysniper-ml/src/rl/reward.rs` - Reward calculation
- `crates/polysniper-execution/src/rl_executor.rs` - RL-enhanced executor
- `config/rl_execution.toml` - Configuration

**Modify:**
- `crates/polysniper-ml/src/lib.rs` - Export RL module
- `crates/polysniper-execution/Cargo.toml` - Add polysniper-ml dependency
- `crates/polysniper-execution/src/lib.rs` - Export RL executor
- `crates/polysniper-execution/src/mod.rs` - Register RL executor

## Integration Points

- **Provides**: RL-enhanced execution algorithm
- **Consumes**: StateProvider (orderbook, prices), TWAP/VWAP base executors
- **Conflicts**: Avoid modifying twap.rs/vwap.rs core logic (wrap instead)

## Testing Strategy

1. Unit tests for Q-learning convergence
2. Unit tests for state discretization
3. Unit tests for reward calculation
4. Mock executor tests for action selection
5. Simulation tests with synthetic market data
6. Comparison tests: RL vs baseline TWAP/VWAP
