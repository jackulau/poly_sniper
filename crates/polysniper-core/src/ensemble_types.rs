//! Ensemble Types
//!
//! Types for multi-LLM ensemble prediction systems.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{BinaryPrediction, PredictionValue};

/// Ensemble orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Whether ensemble mode is enabled
    #[serde(default)]
    pub enabled: bool,
    /// Aggregation strategy to use
    #[serde(default)]
    pub aggregation_strategy: AggregationStrategy,
    /// Minimum agreement threshold (0.0-1.0) to proceed with signal
    #[serde(default = "default_min_agreement")]
    pub min_agreement_threshold: Decimal,
    /// High agreement threshold (above this, full confidence)
    #[serde(default = "default_high_agreement")]
    pub high_agreement_threshold: Decimal,
    /// Timeout for each provider in seconds
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    /// Maximum cost per query across all providers
    #[serde(default = "default_max_cost")]
    pub max_cost_per_query: Decimal,
    /// Minimum number of providers required for a valid ensemble
    #[serde(default = "default_min_providers")]
    pub min_providers: usize,
    /// Provider configurations
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
}

fn default_min_agreement() -> Decimal {
    Decimal::new(60, 2) // 0.60
}

fn default_high_agreement() -> Decimal {
    Decimal::new(80, 2) // 0.80
}

fn default_timeout_secs() -> u64 {
    30
}

fn default_max_cost() -> Decimal {
    Decimal::new(10, 2) // 0.10 USD
}

fn default_min_providers() -> usize {
    2
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            aggregation_strategy: AggregationStrategy::default(),
            min_agreement_threshold: default_min_agreement(),
            high_agreement_threshold: default_high_agreement(),
            timeout_secs: default_timeout_secs(),
            max_cost_per_query: default_max_cost(),
            min_providers: default_min_providers(),
            providers: Vec::new(),
        }
    }
}

/// Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider type (e.g., "openrouter")
    #[serde(rename = "type")]
    pub provider_type: String,
    /// Model name/ID
    pub model: String,
    /// Whether this provider is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Weight for this provider's predictions
    #[serde(default = "default_weight")]
    pub weight: Decimal,
    /// Cost per request in USD
    #[serde(default)]
    pub cost_per_request: Decimal,
    /// Custom temperature for this provider
    pub temperature: Option<f32>,
    /// Custom max tokens for this provider
    pub max_tokens: Option<u32>,
}

fn default_true() -> bool {
    true
}

fn default_weight() -> Decimal {
    Decimal::ONE
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            provider_type: "openrouter".to_string(),
            model: String::new(),
            enabled: true,
            weight: Decimal::ONE,
            cost_per_request: Decimal::ZERO,
            temperature: None,
            max_tokens: None,
        }
    }
}

/// Aggregation strategy for combining predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AggregationStrategy {
    /// Weighted average of probabilities
    #[default]
    WeightedAverage,
    /// Majority vote with confidence weighting
    Voting,
    /// Use highest confidence prediction
    ConfidenceMaximum,
    /// Bayesian combination of posteriors
    BayesianAggregation,
}

/// Weight configuration for a model in the ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeight {
    /// Model identifier
    pub model_id: String,
    /// Weight based on historical accuracy
    pub accuracy_weight: Decimal,
    /// Weight based on confidence calibration
    pub confidence_weight: Decimal,
    /// Cost per request in USD
    pub cost_per_request: Decimal,
    /// Average latency in milliseconds
    pub avg_latency_ms: u64,
    /// Whether this model is enabled
    pub enabled: bool,
}

impl ModelWeight {
    /// Create a new ModelWeight with default values
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            accuracy_weight: Decimal::ONE,
            confidence_weight: Decimal::ONE,
            cost_per_request: Decimal::ZERO,
            avg_latency_ms: 0,
            enabled: true,
        }
    }

    /// Calculate combined weight
    pub fn combined_weight(&self) -> Decimal {
        self.accuracy_weight * self.confidence_weight
    }
}

/// Result of ensemble prediction aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePrediction {
    /// Unique identifier
    pub id: String,
    /// Final aggregated prediction
    pub prediction: PredictionValue,
    /// Aggregated probability for YES outcome
    pub probability: Decimal,
    /// Combined confidence score
    pub confidence: Decimal,
    /// Combined reasoning from all models
    pub reasoning: String,
    /// Individual model contributions (model_id -> weight used)
    pub model_contributions: HashMap<String, Decimal>,
    /// Agreement score between models (0.0-1.0)
    pub agreement_score: Decimal,
    /// Number of providers that responded
    pub provider_count: usize,
    /// Timestamp of ensemble prediction
    pub generated_at: DateTime<Utc>,
}

impl EnsemblePrediction {
    /// Check if ensemble has high agreement
    pub fn has_high_agreement(&self, threshold: Decimal) -> bool {
        self.agreement_score >= threshold
    }

    /// Check if the prediction is for YES
    pub fn is_yes(&self) -> bool {
        matches!(
            self.prediction,
            PredictionValue::Binary(BinaryPrediction::Yes)
        ) || matches!(&self.prediction, PredictionValue::Probability(p) if *p > Decimal::new(50, 2))
    }
}

/// Individual LLM prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmPredictionResult {
    /// Model identifier
    pub model_id: String,
    /// Prediction value
    pub prediction: PredictionValue,
    /// Confidence score (0.0-1.0)
    pub confidence: Decimal,
    /// Reasoning/explanation
    pub reasoning: String,
    /// Target price if provided
    pub target_price: Option<Decimal>,
    /// Time taken for this prediction
    pub latency_ms: u64,
    /// Cost of this request
    pub cost: Decimal,
}

/// Resolution for handling model disagreement
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum DisagreementResolution {
    /// Proceed with the aggregated prediction
    Proceed,
    /// Reduce position size due to disagreement
    ReduceSize {
        /// Multiplier to apply to size (0.0-1.0)
        size_multiplier: Decimal,
        /// Reason for reduction
        reason: String,
    },
    /// Abstain from trading due to high disagreement
    Abstain {
        /// Reason for abstaining
        reason: String,
        /// Agreement score when abstaining
        agreement_score: Decimal,
    },
}

impl DisagreementResolution {
    /// Check if we should proceed with trading
    pub fn should_proceed(&self) -> bool {
        matches!(self, DisagreementResolution::Proceed)
    }

    /// Get the size multiplier (1.0 if proceeding normally)
    pub fn size_multiplier(&self) -> Decimal {
        match self {
            DisagreementResolution::Proceed => Decimal::ONE,
            DisagreementResolution::ReduceSize {
                size_multiplier, ..
            } => *size_multiplier,
            DisagreementResolution::Abstain { .. } => Decimal::ZERO,
        }
    }
}

/// Market context for LLM analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketContext {
    /// Market condition ID
    pub market_id: String,
    /// Market question
    pub question: String,
    /// Market description
    pub description: Option<String>,
    /// Current YES price
    pub yes_price: Decimal,
    /// Current NO price
    pub no_price: Decimal,
    /// 24h trading volume
    pub volume: Decimal,
    /// Market liquidity
    pub liquidity: Decimal,
    /// Time until resolution
    pub time_remaining: Option<String>,
    /// Market end date
    pub end_date: Option<DateTime<Utc>>,
    /// Additional context/metadata
    pub metadata: Option<serde_json::Value>,
}

impl MarketContext {
    /// Calculate implied probability from YES price
    pub fn implied_probability(&self) -> Decimal {
        self.yes_price
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_ensemble_config_defaults() {
        let config = EnsembleConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.min_agreement_threshold, dec!(0.60));
        assert_eq!(config.high_agreement_threshold, dec!(0.80));
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_cost_per_query, dec!(0.10));
        assert_eq!(config.min_providers, 2);
    }

    #[test]
    fn test_model_weight_combined() {
        let weight = ModelWeight {
            model_id: "test".to_string(),
            accuracy_weight: dec!(1.2),
            confidence_weight: dec!(0.8),
            cost_per_request: dec!(0.01),
            avg_latency_ms: 500,
            enabled: true,
        };
        assert_eq!(weight.combined_weight(), dec!(0.96));
    }

    #[test]
    fn test_disagreement_resolution_size() {
        let proceed = DisagreementResolution::Proceed;
        assert!(proceed.should_proceed());
        assert_eq!(proceed.size_multiplier(), dec!(1.0));

        let reduce = DisagreementResolution::ReduceSize {
            size_multiplier: dec!(0.5),
            reason: "test".to_string(),
        };
        assert!(!reduce.should_proceed());
        assert_eq!(reduce.size_multiplier(), dec!(0.5));

        let abstain = DisagreementResolution::Abstain {
            reason: "test".to_string(),
            agreement_score: dec!(0.4),
        };
        assert!(!abstain.should_proceed());
        assert_eq!(abstain.size_multiplier(), dec!(0.0));
    }

    #[test]
    fn test_ensemble_prediction_is_yes() {
        let pred_yes = EnsemblePrediction {
            id: "test".to_string(),
            prediction: PredictionValue::Binary(BinaryPrediction::Yes),
            probability: dec!(0.75),
            confidence: dec!(0.80),
            reasoning: "test".to_string(),
            model_contributions: HashMap::new(),
            agreement_score: dec!(0.90),
            provider_count: 3,
            generated_at: Utc::now(),
        };
        assert!(pred_yes.is_yes());

        let pred_no = EnsemblePrediction {
            prediction: PredictionValue::Binary(BinaryPrediction::No),
            ..pred_yes.clone()
        };
        assert!(!pred_no.is_yes());

        let pred_prob_high = EnsemblePrediction {
            prediction: PredictionValue::Probability(dec!(0.70)),
            ..pred_yes.clone()
        };
        assert!(pred_prob_high.is_yes());

        let pred_prob_low = EnsemblePrediction {
            prediction: PredictionValue::Probability(dec!(0.30)),
            ..pred_yes
        };
        assert!(!pred_prob_low.is_yes());
    }

    #[test]
    fn test_aggregation_strategy_serde() {
        let json = r#""weighted_average""#;
        let strategy: AggregationStrategy = serde_json::from_str(json).unwrap();
        assert_eq!(strategy, AggregationStrategy::WeightedAverage);

        let json = r#""voting""#;
        let strategy: AggregationStrategy = serde_json::from_str(json).unwrap();
        assert_eq!(strategy, AggregationStrategy::Voting);
    }
}
