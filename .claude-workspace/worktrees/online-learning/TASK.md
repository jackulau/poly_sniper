---
id: online-learning
name: Online Learning with Trade Outcome Updates
wave: 1
priority: 1
dependencies: []
estimated_hours: 6
tags: [ml, backend, learning]
---

## Objective

Implement an online learning system that continuously updates model weights and parameters based on recent trade outcomes, allowing the trading system to adapt to changing market conditions in real-time.

## Context

Currently, the ML processor tracks model performance (accuracy, P&L) but doesn't use this feedback to adjust behavior. The existing `ModelPerformance` struct in `ml_types.rs` tracks predictions and outcomes, but the system lacks:
- Automatic weight adjustment based on rolling accuracy
- Decay mechanisms for stale performance data
- Adaptive confidence thresholds that learn from outcomes
- Multi-armed bandit approach for model selection

## Implementation

### 1. Create Online Learning Core (`crates/polysniper-ml/src/online_learning.rs`)

```rust
pub struct OnlineLearner {
    // Model performance with temporal decay
    model_stats: Arc<RwLock<HashMap<String, AdaptiveModelStats>>>,
    // Learning parameters
    config: OnlineLearningConfig,
    // Database for persistence
    db: Arc<Database>,
    // Event channel for outcome notifications
    outcome_rx: broadcast::Receiver<TradeOutcome>,
}

pub struct AdaptiveModelStats {
    pub model_id: String,
    // Exponential moving average of accuracy
    pub ema_accuracy: Decimal,
    // Recent prediction window for learning
    pub recent_predictions: VecDeque<(MlPrediction, Option<bool>)>,
    // Learned confidence threshold (adapts over time)
    pub adaptive_threshold: Decimal,
    // Learned weight (for ensemble)
    pub adaptive_weight: Decimal,
    // Thompson Sampling parameters (alpha, beta for Beta distribution)
    pub thompson_alpha: f64,
    pub thompson_beta: f64,
    // Last update timestamp
    pub updated_at: DateTime<Utc>,
}

pub struct TradeOutcome {
    pub prediction_id: String,
    pub model_id: String,
    pub predicted_outcome: Outcome,
    pub actual_outcome: Outcome,
    pub pnl: Decimal,
    pub resolved_at: DateTime<Utc>,
}
```

### 2. Outcome Tracking Integration

```rust
impl OnlineLearner {
    /// Process a resolved market and update model stats
    pub async fn record_outcome(&self, outcome: TradeOutcome) -> Result<()> {
        let mut stats = self.model_stats.write().await;
        let model = stats.entry(outcome.model_id.clone())
            .or_insert_with(|| AdaptiveModelStats::new(&outcome.model_id));

        // Update EMA accuracy
        let is_correct = outcome.predicted_outcome == outcome.actual_outcome;
        model.ema_accuracy = self.config.ema_alpha * Decimal::from(is_correct as u8)
            + (Decimal::ONE - self.config.ema_alpha) * model.ema_accuracy;

        // Update Thompson Sampling parameters
        if is_correct {
            model.thompson_alpha += 1.0;
        } else {
            model.thompson_beta += 1.0;
        }

        // Adapt confidence threshold based on calibration
        model.update_adaptive_threshold(&self.config);

        // Persist to database
        self.persist_stats(&model).await?;
        Ok(())
    }

    /// Get current adaptive parameters for a model
    pub async fn get_adaptive_params(&self, model_id: &str) -> Option<AdaptiveParams> {
        let stats = self.model_stats.read().await;
        stats.get(model_id).map(|s| AdaptiveParams {
            confidence_threshold: s.adaptive_threshold,
            weight: s.adaptive_weight,
            ema_accuracy: s.ema_accuracy,
        })
    }
}
```

### 3. Thompson Sampling for Model Selection

```rust
impl OnlineLearner {
    /// Sample from posterior to select which model(s) to use
    pub async fn thompson_sample(&self) -> Vec<(String, f64)> {
        use rand::distributions::{Beta, Distribution};
        let mut rng = rand::thread_rng();
        let stats = self.model_stats.read().await;

        let mut samples: Vec<(String, f64)> = stats.iter()
            .filter(|(_, s)| s.is_eligible())
            .map(|(id, s)| {
                let beta = Beta::new(s.thompson_alpha, s.thompson_beta).unwrap();
                (id.clone(), beta.sample(&mut rng))
            })
            .collect();

        // Sort by sampled value (highest first)
        samples.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        samples
    }
}
```

### 4. Adaptive Confidence Thresholds

```rust
impl AdaptiveModelStats {
    /// Update threshold based on calibration error
    pub fn update_adaptive_threshold(&mut self, config: &OnlineLearningConfig) {
        // Track calibration: predicted confidence vs actual accuracy
        let calibration_error = self.calculate_calibration_error();

        // If model is overconfident, raise threshold
        // If model is underconfident, lower threshold
        if calibration_error > Decimal::ZERO {
            self.adaptive_threshold = (self.adaptive_threshold
                + config.threshold_adjustment_rate * calibration_error)
                .min(config.max_threshold);
        } else {
            self.adaptive_threshold = (self.adaptive_threshold
                + config.threshold_adjustment_rate * calibration_error)
                .max(config.min_threshold);
        }
    }
}
```

### 5. Integration with ML Processor

Modify `crates/polysniper-strategies/src/ml_processor.rs`:

```rust
impl MlSignalProcessor {
    pub async fn process_signal_with_learning(
        &self,
        signal: &ExternalSignalEvent,
        learner: &OnlineLearner,
    ) -> Result<Option<MlProcessingResult>> {
        let prediction = self.parse_prediction(signal)?;

        // Get adaptive parameters instead of static config
        let params = learner.get_adaptive_params(&prediction.model_id).await
            .unwrap_or_else(|| self.default_params());

        // Use adaptive threshold
        if prediction.confidence < params.confidence_threshold {
            return Ok(None);
        }

        // Apply adaptive weight to size multiplier
        let size_multiplier = prediction.confidence * params.weight;

        // ... rest of processing
    }
}
```

### 6. Resolution Event Listener

```rust
impl OnlineLearner {
    /// Listen for market resolution events and update model stats
    pub async fn start_outcome_listener(
        self: Arc<Self>,
        mut event_rx: broadcast::Receiver<SystemEvent>,
    ) {
        tokio::spawn(async move {
            while let Ok(event) = event_rx.recv().await {
                if let SystemEvent::MarketResolved(resolution) = event {
                    if let Some(outcome) = self.find_prediction_outcome(&resolution).await {
                        if let Err(e) = self.record_outcome(outcome).await {
                            tracing::error!("Failed to record outcome: {}", e);
                        }
                    }
                }
            }
        });
    }
}
```

### 7. Configuration (`config/online_learning.toml`)

```toml
[online_learning]
enabled = true
ema_alpha = 0.1  # Weight for new observations
min_predictions_for_adaptation = 10
decay_half_life_hours = 168  # 1 week

[online_learning.thresholds]
min_threshold = 0.40
max_threshold = 0.90
initial_threshold = 0.60
adjustment_rate = 0.01

[online_learning.thompson_sampling]
enabled = true
prior_alpha = 1.0  # Initial successes (prior)
prior_beta = 1.0   # Initial failures (prior)

[online_learning.persistence]
save_interval_secs = 60
db_table = "model_learning_stats"
```

## Acceptance Criteria

- [ ] OnlineLearner core implemented with EMA accuracy tracking
- [ ] Thompson Sampling for model selection
- [ ] Adaptive confidence thresholds based on calibration
- [ ] Integration with resolution events for outcome tracking
- [ ] Persistence of learning state to database
- [ ] ML processor updated to use adaptive parameters
- [ ] Unit tests for all learning algorithms
- [ ] Integration tests for outcome tracking flow
- [ ] All existing tests pass

## Files to Create/Modify

**Create:**
- `crates/polysniper-ml/src/online_learning.rs` - Core online learning
- `crates/polysniper-ml/src/thompson_sampling.rs` - Thompson Sampling implementation
- `crates/polysniper-ml/src/calibration.rs` - Calibration analysis
- `config/online_learning.toml` - Configuration

**Modify:**
- `crates/polysniper-ml/src/lib.rs` - Export online learning module
- `crates/polysniper-core/src/events.rs` - Add MarketResolved event if not exists
- `crates/polysniper-strategies/src/ml_processor.rs` - Use adaptive parameters
- `crates/polysniper-persistence/src/models.rs` - Add learning stats table

## Integration Points

- **Provides**: Adaptive model parameters, Thompson Sampling model selection
- **Consumes**: SystemEvent (MarketResolved), MlPrediction outcomes
- **Conflicts**: Avoid modifying ml_types.rs extensively (add new types instead)

## Testing Strategy

1. Unit tests for EMA calculation
2. Unit tests for Thompson Sampling convergence
3. Unit tests for threshold adaptation
4. Integration tests for outcome tracking
5. Simulation tests with synthetic outcomes
