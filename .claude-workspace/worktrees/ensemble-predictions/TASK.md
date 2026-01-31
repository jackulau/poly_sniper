---
id: ensemble-predictions
name: Ensemble Predictions with Multi-LLM Weighting
wave: 1
priority: 1
dependencies: []
estimated_hours: 5
tags: [ml, llm, ensemble]
---

## Objective

Implement an ensemble prediction system that combines predictions from multiple LLM providers (OpenRouter models, Claude, GPT-4, etc.), weighting them by historical accuracy and confidence calibration.

## Context

Currently, `llm_prediction.rs` uses a single LLM model (Grok-3 via OpenRouter). While `ml_types.rs` and `ml_processor.rs` support per-model weighting, there's no orchestration layer to:
- Query multiple LLM providers in parallel
- Aggregate predictions with learned weights
- Handle disagreements between models
- Optimize model selection based on cost and latency

## Implementation

### 1. Create Ensemble Orchestrator (`crates/polysniper-ml/src/ensemble.rs`)

```rust
pub struct EnsembleOrchestrator {
    // LLM providers
    providers: Vec<Arc<dyn LlmProvider>>,
    // Model weights (can be updated by OnlineLearner)
    weights: Arc<RwLock<HashMap<String, ModelWeight>>>,
    // Aggregation strategy
    aggregation: AggregationStrategy,
    // Configuration
    config: EnsembleConfig,
}

pub struct ModelWeight {
    pub model_id: String,
    pub accuracy_weight: Decimal,
    pub confidence_weight: Decimal,
    pub cost_per_request: Decimal,
    pub avg_latency_ms: u64,
    pub enabled: bool,
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn model_id(&self) -> &str;
    fn provider_name(&self) -> &str;
    async fn predict(&self, context: &MarketContext) -> Result<LlmPrediction>;
    fn cost_per_request(&self) -> Decimal;
}

pub enum AggregationStrategy {
    WeightedAverage,     // Weighted sum of probabilities
    Voting,              // Majority vote with confidence threshold
    StackedEnsemble,     // Meta-model combines predictions
    ConfidenceMaximum,   // Use highest confidence prediction
    BayesianAggregation, // Bayesian combination of posteriors
}
```

### 2. Implement LLM Providers

```rust
// openrouter_provider.rs
pub struct OpenRouterProvider {
    client: OpenRouterClient,
    model_name: String,
    config: OpenRouterConfig,
}

#[async_trait]
impl LlmProvider for OpenRouterProvider {
    fn model_id(&self) -> &str { &self.model_name }
    fn provider_name(&self) -> &str { "openrouter" }

    async fn predict(&self, context: &MarketContext) -> Result<LlmPrediction> {
        let prompt = self.build_prompt(context);
        let response = self.client.chat_completion(&prompt).await?;
        self.parse_response(&response)
    }
}

// Additional providers (can be added later):
// - ClaudeProvider (direct Anthropic API)
// - OpenAIProvider (direct OpenAI API)
// - LocalModelProvider (local models via llama.cpp)
```

### 3. Parallel Query Execution

```rust
impl EnsembleOrchestrator {
    /// Query all enabled providers in parallel
    pub async fn get_predictions(
        &self,
        context: &MarketContext,
    ) -> Result<Vec<(String, LlmPrediction)>> {
        let weights = self.weights.read().await;
        let enabled_providers: Vec<_> = self.providers.iter()
            .filter(|p| weights.get(p.model_id()).map(|w| w.enabled).unwrap_or(false))
            .collect();

        let futures: Vec<_> = enabled_providers.iter()
            .map(|provider| {
                let ctx = context.clone();
                let provider = Arc::clone(provider);
                async move {
                    let result = tokio::time::timeout(
                        Duration::from_secs(self.config.timeout_secs),
                        provider.predict(&ctx)
                    ).await;

                    match result {
                        Ok(Ok(pred)) => Some((provider.model_id().to_string(), pred)),
                        Ok(Err(e)) => {
                            tracing::warn!("Provider {} failed: {}", provider.model_id(), e);
                            None
                        }
                        Err(_) => {
                            tracing::warn!("Provider {} timed out", provider.model_id());
                            None
                        }
                    }
                }
            })
            .collect();

        let results = futures::future::join_all(futures).await;
        Ok(results.into_iter().flatten().collect())
    }
}
```

### 4. Weighted Aggregation

```rust
impl EnsembleOrchestrator {
    /// Aggregate predictions using configured strategy
    pub async fn aggregate(
        &self,
        predictions: &[(String, LlmPrediction)],
    ) -> Result<EnsemblePrediction> {
        let weights = self.weights.read().await;

        match self.aggregation {
            AggregationStrategy::WeightedAverage => {
                self.weighted_average_aggregate(predictions, &weights)
            }
            AggregationStrategy::Voting => {
                self.voting_aggregate(predictions, &weights)
            }
            AggregationStrategy::BayesianAggregation => {
                self.bayesian_aggregate(predictions, &weights)
            }
            // ... other strategies
        }
    }

    fn weighted_average_aggregate(
        &self,
        predictions: &[(String, LlmPrediction)],
        weights: &HashMap<String, ModelWeight>,
    ) -> Result<EnsemblePrediction> {
        let mut weighted_yes_prob = Decimal::ZERO;
        let mut total_weight = Decimal::ZERO;
        let mut combined_reasoning = Vec::new();

        for (model_id, pred) in predictions {
            let weight = weights.get(model_id)
                .map(|w| w.accuracy_weight)
                .unwrap_or(Decimal::ONE);

            let prob = match &pred.prediction {
                PredictionValue::Binary(b) => {
                    if *b == BinaryPrediction::Yes { pred.confidence } else { Decimal::ONE - pred.confidence }
                }
                PredictionValue::Probability(p) => *p,
                _ => continue,
            };

            weighted_yes_prob += prob * weight;
            total_weight += weight;
            combined_reasoning.push(format!("{}: {}", model_id, pred.reasoning));
        }

        let final_prob = weighted_yes_prob / total_weight;

        Ok(EnsemblePrediction {
            prediction: if final_prob > dec!(0.5) {
                PredictionValue::Binary(BinaryPrediction::Yes)
            } else {
                PredictionValue::Binary(BinaryPrediction::No)
            },
            probability: final_prob,
            confidence: self.calculate_ensemble_confidence(predictions, weights),
            reasoning: combined_reasoning.join("\n---\n"),
            model_contributions: predictions.iter()
                .map(|(id, p)| (id.clone(), weights.get(id).map(|w| w.accuracy_weight).unwrap_or(Decimal::ONE)))
                .collect(),
            agreement_score: self.calculate_agreement(predictions),
        })
    }
}
```

### 5. Disagreement Handling

```rust
impl EnsembleOrchestrator {
    /// Calculate agreement score (0-1)
    fn calculate_agreement(&self, predictions: &[(String, LlmPrediction)]) -> Decimal {
        let yes_votes: usize = predictions.iter()
            .filter(|(_, p)| matches!(p.prediction, PredictionValue::Binary(BinaryPrediction::Yes)))
            .count();

        let total = predictions.len() as f64;
        let majority = (yes_votes as f64).max(total - yes_votes as f64);

        Decimal::from_f64(majority / total).unwrap_or(Decimal::ZERO)
    }

    /// Handle low agreement scenarios
    pub async fn handle_disagreement(
        &self,
        predictions: &[(String, LlmPrediction)],
    ) -> DisagreementResolution {
        let agreement = self.calculate_agreement(predictions);

        if agreement < self.config.min_agreement_threshold {
            DisagreementResolution::Abstain {
                reason: "Model disagreement too high".to_string(),
                agreement_score: agreement,
            }
        } else if agreement < self.config.high_agreement_threshold {
            DisagreementResolution::ReduceSize {
                size_multiplier: agreement,
                reason: "Moderate model disagreement".to_string(),
            }
        } else {
            DisagreementResolution::Proceed
        }
    }
}

pub enum DisagreementResolution {
    Proceed,
    ReduceSize { size_multiplier: Decimal, reason: String },
    Abstain { reason: String, agreement_score: Decimal },
}
```

### 6. Cost-Aware Model Selection

```rust
impl EnsembleOrchestrator {
    /// Select models based on cost budget and performance
    pub async fn select_models_for_query(
        &self,
        budget: Decimal,
    ) -> Vec<String> {
        let weights = self.weights.read().await;

        // Sort by value = accuracy_weight / cost
        let mut models: Vec<_> = weights.iter()
            .filter(|(_, w)| w.enabled && w.cost_per_request <= budget)
            .map(|(id, w)| {
                let value = w.accuracy_weight / w.cost_per_request.max(dec!(0.001));
                (id.clone(), w.cost_per_request, value)
            })
            .collect();

        models.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // Greedily select models within budget
        let mut selected = Vec::new();
        let mut remaining = budget;

        for (id, cost, _) in models {
            if cost <= remaining {
                selected.push(id);
                remaining -= cost;
            }
        }

        selected
    }
}
```

### 7. Configuration (`config/ensemble.toml`)

```toml
[ensemble]
enabled = true
aggregation_strategy = "weighted_average"  # weighted_average, voting, bayesian
min_agreement_threshold = 0.6
high_agreement_threshold = 0.8
timeout_secs = 30
max_cost_per_query = 0.10

[[ensemble.providers]]
type = "openrouter"
model = "x-ai/grok-3-latest"
enabled = true
weight = 1.0

[[ensemble.providers]]
type = "openrouter"
model = "anthropic/claude-3.5-sonnet"
enabled = true
weight = 1.2

[[ensemble.providers]]
type = "openrouter"
model = "openai/gpt-4-turbo"
enabled = true
weight = 1.1

[[ensemble.providers]]
type = "openrouter"
model = "google/gemini-pro-1.5"
enabled = false
weight = 0.9
```

## Acceptance Criteria

- [ ] EnsembleOrchestrator implemented with parallel querying
- [ ] At least 3 providers configured (can all use OpenRouter)
- [ ] Weighted average aggregation working
- [ ] Agreement score calculation and disagreement handling
- [ ] Cost-aware model selection
- [ ] Integration with existing LlmPredictionStrategy
- [ ] Unit tests for aggregation strategies
- [ ] Integration tests for ensemble prediction flow
- [ ] All existing tests pass

## Files to Create/Modify

**Create:**
- `crates/polysniper-ml/src/ensemble.rs` - Ensemble orchestrator
- `crates/polysniper-ml/src/providers/mod.rs` - Provider trait and implementations
- `crates/polysniper-ml/src/providers/openrouter.rs` - OpenRouter provider
- `crates/polysniper-ml/src/aggregation.rs` - Aggregation strategies
- `config/ensemble.toml` - Configuration

**Modify:**
- `crates/polysniper-ml/src/lib.rs` - Export ensemble module
- `crates/polysniper-strategies/src/llm_prediction.rs` - Use ensemble instead of single model
- `crates/polysniper-data/src/openrouter_client.rs` - Ensure it supports multiple models

## Integration Points

- **Provides**: Ensemble predictions with confidence and agreement scores
- **Consumes**: OpenRouterClient, MarketContext, OnlineLearner weights
- **Conflicts**: Keep ml_processor.rs changes minimal (it handles external ML signals)

## Testing Strategy

1. Unit tests for each aggregation strategy
2. Mock provider tests for parallel execution
3. Disagreement handling tests
4. Cost optimization tests
5. Integration tests with real OpenRouter (optional, requires API key)
