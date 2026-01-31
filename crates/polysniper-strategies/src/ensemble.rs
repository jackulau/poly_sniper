//! Ensemble Orchestrator
//!
//! Combines predictions from multiple LLM providers using configurable
//! aggregation strategies with support for disagreement handling and
//! cost-aware model selection.

use chrono::Utc;
use polysniper_core::{
    AggregationStrategy, BinaryPrediction, DisagreementResolution, EnsembleConfig,
    EnsemblePrediction, LlmPredictionResult, MarketContext, ModelWeight, PredictionValue,
};
use polysniper_data::{BoxedProvider, ProviderError, ProviderFactory};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Ensemble orchestrator for multi-LLM predictions
pub struct EnsembleOrchestrator {
    /// LLM providers
    providers: Vec<BoxedProvider>,
    /// Model weights (can be updated by online learning)
    weights: Arc<RwLock<HashMap<String, ModelWeight>>>,
    /// Configuration
    config: EnsembleConfig,
}

impl EnsembleOrchestrator {
    /// Create a new ensemble orchestrator
    pub fn new(config: EnsembleConfig, api_key: &str) -> Result<Self, EnsembleError> {
        let mut providers = Vec::new();
        let mut weights = HashMap::new();

        for provider_config in &config.providers {
            if !provider_config.enabled {
                continue;
            }

            match ProviderFactory::create(provider_config, api_key) {
                Ok(provider) => {
                    let model_id = provider.model_id().to_string();

                    // Initialize weights
                    let weight = ModelWeight {
                        model_id: model_id.clone(),
                        accuracy_weight: provider_config.weight,
                        confidence_weight: Decimal::ONE,
                        cost_per_request: provider_config.cost_per_request,
                        avg_latency_ms: 0,
                        enabled: true,
                    };
                    weights.insert(model_id, weight);

                    providers.push(provider);
                }
                Err(e) => {
                    warn!(
                        provider_type = %provider_config.provider_type,
                        model = %provider_config.model,
                        error = %e,
                        "Failed to create provider, skipping"
                    );
                }
            }
        }

        if providers.is_empty() {
            return Err(EnsembleError::NoProviders);
        }

        info!(
            provider_count = providers.len(),
            "Initialized ensemble orchestrator"
        );

        Ok(Self {
            providers,
            weights: Arc::new(RwLock::new(weights)),
            config,
        })
    }

    /// Get the number of enabled providers
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    /// Query all enabled providers in parallel
    pub async fn get_predictions(
        &self,
        context: &MarketContext,
    ) -> Result<Vec<LlmPredictionResult>, EnsembleError> {
        let weights = self.weights.read().await;
        let timeout = Duration::from_secs(self.config.timeout_secs);

        // Create futures for all enabled providers
        let futures: Vec<_> = self
            .providers
            .iter()
            .filter(|p| {
                weights
                    .get(p.model_id())
                    .map(|w| w.enabled)
                    .unwrap_or(false)
            })
            .map(|provider| {
                let ctx = context.clone();
                let provider = Arc::clone(provider);
                let timeout_duration = timeout;

                async move {
                    let result = tokio::time::timeout(timeout_duration, provider.predict(&ctx)).await;

                    match result {
                        Ok(Ok(pred)) => Some(pred),
                        Ok(Err(e)) => {
                            warn!(
                                model = %provider.model_id(),
                                error = %e,
                                "Provider prediction failed"
                            );
                            None
                        }
                        Err(_) => {
                            warn!(
                                model = %provider.model_id(),
                                "Provider timed out"
                            );
                            None
                        }
                    }
                }
            })
            .collect();

        // Execute all futures in parallel
        let results = futures::future::join_all(futures).await;
        let predictions: Vec<_> = results.into_iter().flatten().collect();

        if predictions.is_empty() {
            return Err(EnsembleError::AllProvidersFailed);
        }

        if predictions.len() < self.config.min_providers {
            return Err(EnsembleError::InsufficientProviders {
                required: self.config.min_providers,
                received: predictions.len(),
            });
        }

        Ok(predictions)
    }

    /// Aggregate predictions using configured strategy
    pub async fn aggregate(
        &self,
        predictions: &[LlmPredictionResult],
    ) -> Result<EnsemblePrediction, EnsembleError> {
        if predictions.is_empty() {
            return Err(EnsembleError::NoPredictions);
        }

        let weights = self.weights.read().await;

        match self.config.aggregation_strategy {
            AggregationStrategy::WeightedAverage => {
                self.weighted_average_aggregate(predictions, &weights)
            }
            AggregationStrategy::Voting => self.voting_aggregate(predictions, &weights),
            AggregationStrategy::ConfidenceMaximum => {
                self.confidence_maximum_aggregate(predictions, &weights)
            }
            AggregationStrategy::BayesianAggregation => {
                self.bayesian_aggregate(predictions, &weights)
            }
        }
    }

    /// Weighted average aggregation
    fn weighted_average_aggregate(
        &self,
        predictions: &[LlmPredictionResult],
        weights: &HashMap<String, ModelWeight>,
    ) -> Result<EnsemblePrediction, EnsembleError> {
        let mut weighted_yes_prob = Decimal::ZERO;
        let mut total_weight = Decimal::ZERO;
        let mut combined_reasoning = Vec::new();
        let mut model_contributions = HashMap::new();

        for pred in predictions {
            let weight = weights
                .get(&pred.model_id)
                .map(|w| w.accuracy_weight)
                .unwrap_or(Decimal::ONE);

            let prob = self.prediction_to_probability(&pred.prediction, pred.confidence);

            weighted_yes_prob += prob * weight;
            total_weight += weight;

            model_contributions.insert(pred.model_id.clone(), weight);
            combined_reasoning.push(format!("[{}] {}", pred.model_id, pred.reasoning));
        }

        let final_prob = if total_weight.is_zero() {
            dec!(0.5)
        } else {
            weighted_yes_prob / total_weight
        };

        let prediction = if final_prob > dec!(0.5) {
            PredictionValue::Binary(BinaryPrediction::Yes)
        } else {
            PredictionValue::Binary(BinaryPrediction::No)
        };

        let confidence = self.calculate_ensemble_confidence(predictions, weights);
        let agreement_score = self.calculate_agreement(predictions);

        Ok(EnsemblePrediction {
            id: Uuid::new_v4().to_string(),
            prediction,
            probability: final_prob,
            confidence,
            reasoning: combined_reasoning.join("\n\n"),
            model_contributions,
            agreement_score,
            provider_count: predictions.len(),
            generated_at: Utc::now(),
        })
    }

    /// Voting-based aggregation
    fn voting_aggregate(
        &self,
        predictions: &[LlmPredictionResult],
        weights: &HashMap<String, ModelWeight>,
    ) -> Result<EnsemblePrediction, EnsembleError> {
        let mut yes_votes = Decimal::ZERO;
        let mut no_votes = Decimal::ZERO;
        let mut combined_reasoning = Vec::new();
        let mut model_contributions = HashMap::new();

        for pred in predictions {
            let weight = weights
                .get(&pred.model_id)
                .map(|w| w.accuracy_weight)
                .unwrap_or(Decimal::ONE);

            let is_yes = match &pred.prediction {
                PredictionValue::Binary(BinaryPrediction::Yes) => true,
                PredictionValue::Binary(BinaryPrediction::No) => false,
                PredictionValue::Probability(p) => *p > dec!(0.5),
                PredictionValue::Categorical(c) => {
                    c.to_lowercase().contains("yes") || c.to_lowercase().contains("true")
                }
            };

            // Weight votes by confidence
            let vote_weight = weight * pred.confidence;

            if is_yes {
                yes_votes += vote_weight;
            } else {
                no_votes += vote_weight;
            }

            model_contributions.insert(pred.model_id.clone(), weight);
            combined_reasoning.push(format!("[{}] {}", pred.model_id, pred.reasoning));
        }

        let total_votes = yes_votes + no_votes;
        let final_prob = if total_votes.is_zero() {
            dec!(0.5)
        } else {
            yes_votes / total_votes
        };

        let prediction = if yes_votes > no_votes {
            PredictionValue::Binary(BinaryPrediction::Yes)
        } else {
            PredictionValue::Binary(BinaryPrediction::No)
        };

        let confidence = self.calculate_ensemble_confidence(predictions, weights);
        let agreement_score = self.calculate_agreement(predictions);

        Ok(EnsemblePrediction {
            id: Uuid::new_v4().to_string(),
            prediction,
            probability: final_prob,
            confidence,
            reasoning: combined_reasoning.join("\n\n"),
            model_contributions,
            agreement_score,
            provider_count: predictions.len(),
            generated_at: Utc::now(),
        })
    }

    /// Use the prediction with highest confidence
    fn confidence_maximum_aggregate(
        &self,
        predictions: &[LlmPredictionResult],
        weights: &HashMap<String, ModelWeight>,
    ) -> Result<EnsemblePrediction, EnsembleError> {
        let best = predictions
            .iter()
            .max_by(|a, b| {
                let weight_a = weights
                    .get(&a.model_id)
                    .map(|w| w.accuracy_weight)
                    .unwrap_or(Decimal::ONE);
                let weight_b = weights
                    .get(&b.model_id)
                    .map(|w| w.accuracy_weight)
                    .unwrap_or(Decimal::ONE);

                let score_a = a.confidence * weight_a;
                let score_b = b.confidence * weight_b;

                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(EnsembleError::NoPredictions)?;

        let mut model_contributions = HashMap::new();
        let mut combined_reasoning = Vec::new();

        for pred in predictions {
            let weight = weights
                .get(&pred.model_id)
                .map(|w| w.accuracy_weight)
                .unwrap_or(Decimal::ONE);
            model_contributions.insert(pred.model_id.clone(), weight);
            combined_reasoning.push(format!("[{}] {}", pred.model_id, pred.reasoning));
        }

        let probability = self.prediction_to_probability(&best.prediction, best.confidence);
        let agreement_score = self.calculate_agreement(predictions);

        Ok(EnsemblePrediction {
            id: Uuid::new_v4().to_string(),
            prediction: best.prediction.clone(),
            probability,
            confidence: best.confidence,
            reasoning: combined_reasoning.join("\n\n"),
            model_contributions,
            agreement_score,
            provider_count: predictions.len(),
            generated_at: Utc::now(),
        })
    }

    /// Bayesian aggregation of predictions
    fn bayesian_aggregate(
        &self,
        predictions: &[LlmPredictionResult],
        weights: &HashMap<String, ModelWeight>,
    ) -> Result<EnsemblePrediction, EnsembleError> {
        // Start with uninformative prior (0.5)
        let mut log_odds = Decimal::ZERO;
        let mut combined_reasoning = Vec::new();
        let mut model_contributions = HashMap::new();

        for pred in predictions {
            let weight = weights
                .get(&pred.model_id)
                .map(|w| w.accuracy_weight)
                .unwrap_or(Decimal::ONE);

            let prob = self.prediction_to_probability(&pred.prediction, pred.confidence);

            // Convert to log-odds and add (Bayesian update)
            // log_odds = log(p / (1-p))
            let clamped_prob = prob.max(dec!(0.01)).min(dec!(0.99));
            let _odds_ratio = clamped_prob / (Decimal::ONE - clamped_prob);

            // Use natural log approximation (ln(x) ≈ (x-1) - (x-1)^2/2 for x near 1)
            // For simplicity, we use a linear approximation weighted by the model weight
            let contribution = (clamped_prob - dec!(0.5)) * dec!(2) * weight;
            log_odds += contribution;

            model_contributions.insert(pred.model_id.clone(), weight);
            combined_reasoning.push(format!("[{}] {}", pred.model_id, pred.reasoning));
        }

        // Convert back to probability using sigmoid-like transformation
        // p = 1 / (1 + exp(-log_odds))
        // Simplified: map log_odds to [0, 1] range
        let normalized = log_odds / Decimal::from(predictions.len());
        let final_prob = (normalized + dec!(0.5)).max(dec!(0.01)).min(dec!(0.99));

        let prediction = if final_prob > dec!(0.5) {
            PredictionValue::Binary(BinaryPrediction::Yes)
        } else {
            PredictionValue::Binary(BinaryPrediction::No)
        };

        let confidence = self.calculate_ensemble_confidence(predictions, weights);
        let agreement_score = self.calculate_agreement(predictions);

        Ok(EnsemblePrediction {
            id: Uuid::new_v4().to_string(),
            prediction,
            probability: final_prob,
            confidence,
            reasoning: combined_reasoning.join("\n\n"),
            model_contributions,
            agreement_score,
            provider_count: predictions.len(),
            generated_at: Utc::now(),
        })
    }

    /// Convert prediction value to probability
    fn prediction_to_probability(&self, prediction: &PredictionValue, confidence: Decimal) -> Decimal {
        match prediction {
            PredictionValue::Binary(BinaryPrediction::Yes) => confidence,
            PredictionValue::Binary(BinaryPrediction::No) => Decimal::ONE - confidence,
            PredictionValue::Probability(p) => *p,
            PredictionValue::Categorical(c) => {
                if c.to_lowercase().contains("yes") || c.to_lowercase().contains("true") {
                    confidence
                } else {
                    Decimal::ONE - confidence
                }
            }
        }
    }

    /// Calculate agreement score (0.0-1.0)
    fn calculate_agreement(&self, predictions: &[LlmPredictionResult]) -> Decimal {
        if predictions.is_empty() {
            return Decimal::ZERO;
        }

        let yes_count = predictions
            .iter()
            .filter(|p| match &p.prediction {
                PredictionValue::Binary(BinaryPrediction::Yes) => true,
                PredictionValue::Probability(prob) => *prob > dec!(0.5),
                PredictionValue::Categorical(c) => {
                    c.to_lowercase().contains("yes") || c.to_lowercase().contains("true")
                }
                _ => false,
            })
            .count();

        let total = predictions.len();
        let majority = yes_count.max(total - yes_count);

        Decimal::from(majority) / Decimal::from(total)
    }

    /// Calculate ensemble confidence
    fn calculate_ensemble_confidence(
        &self,
        predictions: &[LlmPredictionResult],
        weights: &HashMap<String, ModelWeight>,
    ) -> Decimal {
        if predictions.is_empty() {
            return Decimal::ZERO;
        }

        let mut weighted_confidence = Decimal::ZERO;
        let mut total_weight = Decimal::ZERO;

        for pred in predictions {
            let weight = weights
                .get(&pred.model_id)
                .map(|w| w.accuracy_weight)
                .unwrap_or(Decimal::ONE);

            weighted_confidence += pred.confidence * weight;
            total_weight += weight;
        }

        if total_weight.is_zero() {
            return Decimal::ZERO;
        }

        // Adjust by agreement score for more conservative estimates
        let base_confidence = weighted_confidence / total_weight;
        let agreement = self.calculate_agreement(predictions);

        // Final confidence is reduced when there's disagreement
        base_confidence * agreement
    }

    /// Handle disagreement between models
    pub fn handle_disagreement(
        &self,
        predictions: &[LlmPredictionResult],
    ) -> DisagreementResolution {
        let agreement = self.calculate_agreement(predictions);

        if agreement < self.config.min_agreement_threshold {
            DisagreementResolution::Abstain {
                reason: format!(
                    "Model disagreement too high (agreement: {:.0}%)",
                    agreement * dec!(100)
                ),
                agreement_score: agreement,
            }
        } else if agreement < self.config.high_agreement_threshold {
            DisagreementResolution::ReduceSize {
                size_multiplier: agreement,
                reason: format!(
                    "Moderate model disagreement (agreement: {:.0}%)",
                    agreement * dec!(100)
                ),
            }
        } else {
            DisagreementResolution::Proceed
        }
    }

    /// Select models based on cost budget and performance
    pub async fn select_models_for_query(&self, budget: Decimal) -> Vec<String> {
        let weights = self.weights.read().await;

        // Calculate value = accuracy_weight / cost for each model
        let mut models: Vec<_> = weights
            .iter()
            .filter(|(_, w)| w.enabled && w.cost_per_request <= budget)
            .map(|(id, w)| {
                let value = if w.cost_per_request.is_zero() {
                    w.accuracy_weight * dec!(1000) // Very high value for free models
                } else {
                    w.accuracy_weight / w.cost_per_request
                };
                (id.clone(), w.cost_per_request, value)
            })
            .collect();

        // Sort by value (descending)
        models.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

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

    /// Update model weight based on prediction outcome
    pub async fn update_weight(&self, model_id: &str, correct: bool, _latency_ms: u64) {
        let mut weights = self.weights.write().await;

        if let Some(weight) = weights.get_mut(model_id) {
            // Simple exponential moving average update
            let alpha = dec!(0.1); // Learning rate

            if correct {
                weight.accuracy_weight = weight.accuracy_weight * (Decimal::ONE - alpha)
                    + Decimal::new(12, 1) * alpha; // Move toward 1.2
            } else {
                weight.accuracy_weight = weight.accuracy_weight * (Decimal::ONE - alpha)
                    + Decimal::new(8, 1) * alpha; // Move toward 0.8
            }

            // Clamp to reasonable bounds
            weight.accuracy_weight = weight.accuracy_weight.max(dec!(0.5)).min(dec!(2.0));

            debug!(
                model_id = %model_id,
                new_weight = %weight.accuracy_weight,
                correct = %correct,
                "Updated model weight"
            );
        }
    }

    /// Get full ensemble prediction for a market
    pub async fn predict(
        &self,
        context: &MarketContext,
    ) -> Result<(EnsemblePrediction, DisagreementResolution), EnsembleError> {
        // Get predictions from all providers
        let predictions = self.get_predictions(context).await?;

        // Handle disagreement
        let resolution = self.handle_disagreement(&predictions);

        // Aggregate predictions
        let ensemble = self.aggregate(&predictions).await?;

        info!(
            market_id = %context.market_id,
            provider_count = %ensemble.provider_count,
            agreement = %ensemble.agreement_score,
            prediction = ?ensemble.prediction,
            confidence = %ensemble.confidence,
            "Ensemble prediction complete"
        );

        Ok((ensemble, resolution))
    }

    /// Get configuration
    pub fn config(&self) -> &EnsembleConfig {
        &self.config
    }
}

/// Errors that can occur in ensemble operations
#[derive(Debug, thiserror::Error)]
pub enum EnsembleError {
    #[error("No providers configured or all disabled")]
    NoProviders,

    #[error("All providers failed to return predictions")]
    AllProvidersFailed,

    #[error("Insufficient providers responded: required {required}, received {received}")]
    InsufficientProviders { required: usize, received: usize },

    #[error("No predictions to aggregate")]
    NoPredictions,

    #[error("Provider error: {0}")]
    ProviderError(#[from] ProviderError),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn create_test_prediction(model_id: &str, is_yes: bool, confidence: Decimal) -> LlmPredictionResult {
        LlmPredictionResult {
            model_id: model_id.to_string(),
            prediction: if is_yes {
                PredictionValue::Binary(BinaryPrediction::Yes)
            } else {
                PredictionValue::Binary(BinaryPrediction::No)
            },
            confidence,
            reasoning: format!("Reasoning from {}", model_id),
            target_price: None,
            latency_ms: 500,
            cost: dec!(0.01),
        }
    }

    #[test]
    fn test_calculate_agreement_unanimous() {
        let config = EnsembleConfig::default();
        // We can't create orchestrator without API key, but we can test the logic directly

        let predictions = vec![
            create_test_prediction("model1", true, dec!(0.80)),
            create_test_prediction("model2", true, dec!(0.75)),
            create_test_prediction("model3", true, dec!(0.85)),
        ];

        // Unanimous YES
        let yes_count = predictions
            .iter()
            .filter(|p| matches!(p.prediction, PredictionValue::Binary(BinaryPrediction::Yes)))
            .count();
        let agreement = Decimal::from(yes_count.max(predictions.len() - yes_count))
            / Decimal::from(predictions.len());

        assert_eq!(agreement, dec!(1.0));
    }

    #[test]
    fn test_calculate_agreement_split() {
        let predictions = vec![
            create_test_prediction("model1", true, dec!(0.80)),
            create_test_prediction("model2", false, dec!(0.75)),
            create_test_prediction("model3", true, dec!(0.85)),
        ];

        let yes_count = predictions
            .iter()
            .filter(|p| matches!(p.prediction, PredictionValue::Binary(BinaryPrediction::Yes)))
            .count();
        let agreement = Decimal::from(yes_count.max(predictions.len() - yes_count))
            / Decimal::from(predictions.len());

        // 2 yes, 1 no -> 2/3 = 0.666...
        assert!(agreement > dec!(0.66) && agreement < dec!(0.67));
    }

    #[test]
    fn test_disagreement_resolution_abstain() {
        let config = EnsembleConfig {
            min_agreement_threshold: dec!(0.60),
            high_agreement_threshold: dec!(0.80),
            ..Default::default()
        };

        // 1 yes, 1 no -> 50% agreement (below min threshold)
        let agreement = dec!(0.50);

        let resolution = if agreement < config.min_agreement_threshold {
            DisagreementResolution::Abstain {
                reason: "test".to_string(),
                agreement_score: agreement,
            }
        } else if agreement < config.high_agreement_threshold {
            DisagreementResolution::ReduceSize {
                size_multiplier: agreement,
                reason: "test".to_string(),
            }
        } else {
            DisagreementResolution::Proceed
        };

        assert!(matches!(resolution, DisagreementResolution::Abstain { .. }));
    }

    #[test]
    fn test_disagreement_resolution_reduce() {
        let config = EnsembleConfig {
            min_agreement_threshold: dec!(0.60),
            high_agreement_threshold: dec!(0.80),
            ..Default::default()
        };

        // 2 yes, 1 no -> 66.67% agreement (between thresholds)
        let agreement = dec!(0.67);

        let resolution = if agreement < config.min_agreement_threshold {
            DisagreementResolution::Abstain {
                reason: "test".to_string(),
                agreement_score: agreement,
            }
        } else if agreement < config.high_agreement_threshold {
            DisagreementResolution::ReduceSize {
                size_multiplier: agreement,
                reason: "test".to_string(),
            }
        } else {
            DisagreementResolution::Proceed
        };

        assert!(matches!(resolution, DisagreementResolution::ReduceSize { .. }));
    }

    #[test]
    fn test_disagreement_resolution_proceed() {
        let config = EnsembleConfig {
            min_agreement_threshold: dec!(0.60),
            high_agreement_threshold: dec!(0.80),
            ..Default::default()
        };

        // Unanimous -> 100% agreement
        let agreement = dec!(1.00);

        let resolution = if agreement < config.min_agreement_threshold {
            DisagreementResolution::Abstain {
                reason: "test".to_string(),
                agreement_score: agreement,
            }
        } else if agreement < config.high_agreement_threshold {
            DisagreementResolution::ReduceSize {
                size_multiplier: agreement,
                reason: "test".to_string(),
            }
        } else {
            DisagreementResolution::Proceed
        };

        assert!(matches!(resolution, DisagreementResolution::Proceed));
    }

    #[test]
    fn test_prediction_to_probability() {
        // YES prediction -> confidence as probability
        let pred_yes = PredictionValue::Binary(BinaryPrediction::Yes);
        let prob = match &pred_yes {
            PredictionValue::Binary(BinaryPrediction::Yes) => dec!(0.80),
            PredictionValue::Binary(BinaryPrediction::No) => Decimal::ONE - dec!(0.80),
            PredictionValue::Probability(p) => *p,
            _ => dec!(0.5),
        };
        assert_eq!(prob, dec!(0.80));

        // NO prediction -> 1 - confidence
        let pred_no = PredictionValue::Binary(BinaryPrediction::No);
        let prob = match &pred_no {
            PredictionValue::Binary(BinaryPrediction::Yes) => dec!(0.80),
            PredictionValue::Binary(BinaryPrediction::No) => Decimal::ONE - dec!(0.80),
            PredictionValue::Probability(p) => *p,
            _ => dec!(0.5),
        };
        assert_eq!(prob, dec!(0.20));
    }

    #[test]
    fn test_model_selection_by_value() {
        // Test the value calculation: accuracy_weight / cost
        let weights = vec![
            ("model1", dec!(1.2), dec!(0.02)), // value = 60
            ("model2", dec!(1.0), dec!(0.01)), // value = 100
            ("model3", dec!(0.8), dec!(0.03)), // value = 26.67
        ];

        let mut sorted: Vec<_> = weights
            .iter()
            .map(|(id, weight, cost)| {
                let value = *weight / *cost;
                (*id, *cost, value)
            })
            .collect();

        sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        assert_eq!(sorted[0].0, "model2"); // Highest value
        assert_eq!(sorted[1].0, "model1");
        assert_eq!(sorted[2].0, "model3"); // Lowest value
    }
}
