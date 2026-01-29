//! ML Types
//!
//! Types for ML model predictions and signal processing.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ML model prediction from external source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlPrediction {
    /// Unique identifier for this prediction
    pub id: String,
    /// Model identifier that generated this prediction
    pub model_id: String,
    /// Model version (optional)
    pub model_version: Option<String>,
    /// Predicted outcome (e.g., "yes", "no", or probability value)
    pub prediction: PredictionValue,
    /// Confidence score (0.0 to 1.0)
    pub confidence: Decimal,
    /// Target market ID (if known)
    pub market_id: Option<String>,
    /// Target market keywords (for market lookup)
    pub market_keywords: Vec<String>,
    /// Features used for prediction (for debugging/tracking)
    pub features: Option<serde_json::Value>,
    /// Additional metadata from the model
    pub metadata: serde_json::Value,
    /// When the prediction was generated
    pub generated_at: DateTime<Utc>,
    /// Expiration time for the prediction (optional)
    pub expires_at: Option<DateTime<Utc>>,
}

impl MlPrediction {
    /// Check if prediction is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Utc::now() > expires_at
        } else {
            false
        }
    }

    /// Check if confidence meets minimum threshold
    pub fn meets_threshold(&self, min_confidence: Decimal) -> bool {
        self.confidence >= min_confidence
    }
}

/// Prediction value types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PredictionValue {
    /// Binary yes/no prediction
    Binary(BinaryPrediction),
    /// Probability prediction (0.0 to 1.0 for YES outcome)
    Probability(Decimal),
    /// Categorical prediction
    Categorical(String),
}

/// Binary prediction outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BinaryPrediction {
    Yes,
    No,
}

/// ML configuration for signal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlConfig {
    /// Whether ML signal processing is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Global minimum confidence threshold (0.0 to 1.0)
    #[serde(default = "default_min_confidence")]
    pub min_confidence: Decimal,
    /// Maximum confidence cap (for risk management)
    #[serde(default = "default_max_confidence")]
    pub max_confidence: Decimal,
    /// Size multiplier for confidence scaling
    /// Size = base_size * (confidence * confidence_multiplier)
    #[serde(default = "default_confidence_multiplier")]
    pub confidence_multiplier: Decimal,
    /// Minimum size multiplier (prevents tiny orders)
    #[serde(default = "default_min_size_multiplier")]
    pub min_size_multiplier: Decimal,
    /// Maximum size multiplier (caps large orders)
    #[serde(default = "default_max_size_multiplier")]
    pub max_size_multiplier: Decimal,
    /// Default cooldown between signals from same model (seconds)
    #[serde(default = "default_cooldown_secs")]
    pub default_cooldown_secs: u64,
    /// Per-model configuration overrides
    #[serde(default)]
    pub model_configs: HashMap<String, ModelConfig>,
    /// Webhook source pattern for identifying ML signals
    #[serde(default = "default_ml_source_pattern")]
    pub ml_source_pattern: String,
}

fn default_true() -> bool {
    true
}

fn default_min_confidence() -> Decimal {
    Decimal::new(50, 2) // 0.50
}

fn default_max_confidence() -> Decimal {
    Decimal::ONE // 1.0
}

fn default_confidence_multiplier() -> Decimal {
    Decimal::ONE // 1.0
}

fn default_min_size_multiplier() -> Decimal {
    Decimal::new(25, 2) // 0.25
}

fn default_max_size_multiplier() -> Decimal {
    Decimal::new(150, 2) // 1.50
}

fn default_cooldown_secs() -> u64 {
    60 // 1 minute
}

fn default_ml_source_pattern() -> String {
    "ml-webhook".to_string()
}

impl Default for MlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_confidence: default_min_confidence(),
            max_confidence: default_max_confidence(),
            confidence_multiplier: default_confidence_multiplier(),
            min_size_multiplier: default_min_size_multiplier(),
            max_size_multiplier: default_max_size_multiplier(),
            default_cooldown_secs: default_cooldown_secs(),
            model_configs: HashMap::new(),
            ml_source_pattern: default_ml_source_pattern(),
        }
    }
}

/// Per-model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model display name
    pub name: Option<String>,
    /// Whether this model is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Override minimum confidence for this model
    pub min_confidence: Option<Decimal>,
    /// Override confidence multiplier for this model
    pub confidence_multiplier: Option<Decimal>,
    /// Override cooldown for this model (seconds)
    pub cooldown_secs: Option<u64>,
    /// Maximum signals per day from this model
    pub max_signals_per_day: Option<u32>,
    /// Weight multiplier for this model (for ensemble)
    #[serde(default = "default_weight")]
    pub weight: Decimal,
}

fn default_weight() -> Decimal {
    Decimal::ONE
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: None,
            enabled: true,
            min_confidence: None,
            confidence_multiplier: None,
            cooldown_secs: None,
            max_signals_per_day: None,
            weight: Decimal::ONE,
        }
    }
}

/// Model performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    /// Model identifier
    pub model_id: String,
    /// Total number of predictions
    pub total_predictions: u64,
    /// Number of correct predictions
    pub correct_predictions: u64,
    /// Number of incorrect predictions
    pub incorrect_predictions: u64,
    /// Number of pending predictions (not yet resolved)
    pub pending_predictions: u64,
    /// Total profit/loss from this model's signals
    pub total_pnl: Decimal,
    /// Average confidence of predictions
    pub avg_confidence: Decimal,
    /// Rolling accuracy (last N predictions)
    pub rolling_accuracy: Option<Decimal>,
    /// Last prediction time
    pub last_prediction_at: Option<DateTime<Utc>>,
    /// When tracking started
    pub tracking_started_at: DateTime<Utc>,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

impl ModelPerformance {
    /// Create new performance tracker for a model
    pub fn new(model_id: String) -> Self {
        let now = Utc::now();
        Self {
            model_id,
            total_predictions: 0,
            correct_predictions: 0,
            incorrect_predictions: 0,
            pending_predictions: 0,
            total_pnl: Decimal::ZERO,
            avg_confidence: Decimal::ZERO,
            rolling_accuracy: None,
            last_prediction_at: None,
            tracking_started_at: now,
            updated_at: now,
        }
    }

    /// Calculate accuracy as a decimal (0.0 to 1.0)
    pub fn accuracy(&self) -> Option<Decimal> {
        let resolved = self.correct_predictions + self.incorrect_predictions;
        if resolved == 0 {
            None
        } else {
            Some(Decimal::from(self.correct_predictions) / Decimal::from(resolved))
        }
    }

    /// Record a new prediction
    pub fn record_prediction(&mut self, confidence: Decimal) {
        self.total_predictions += 1;
        self.pending_predictions += 1;

        // Update rolling average confidence
        let total = Decimal::from(self.total_predictions);
        self.avg_confidence =
            (self.avg_confidence * (total - Decimal::ONE) + confidence) / total;

        self.last_prediction_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }

    /// Record outcome of a prediction
    pub fn record_outcome(&mut self, correct: bool, pnl: Decimal) {
        if self.pending_predictions > 0 {
            self.pending_predictions -= 1;
        }

        if correct {
            self.correct_predictions += 1;
        } else {
            self.incorrect_predictions += 1;
        }

        self.total_pnl += pnl;
        self.updated_at = Utc::now();
    }

    /// Check if model should be trusted based on historical performance
    pub fn is_reliable(&self, min_accuracy: Decimal, min_predictions: u64) -> bool {
        let resolved = self.correct_predictions + self.incorrect_predictions;
        if resolved < min_predictions {
            return true; // Not enough data, trust it
        }

        self.accuracy().is_some_and(|acc| acc >= min_accuracy)
    }
}

/// Signal outcome for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalOutcome {
    /// Original prediction ID
    pub prediction_id: String,
    /// Model that made the prediction
    pub model_id: String,
    /// Market ID
    pub market_id: String,
    /// Predicted outcome
    pub predicted: PredictionValue,
    /// Confidence of prediction
    pub confidence: Decimal,
    /// Actual outcome (once resolved)
    pub actual: Option<ActualOutcome>,
    /// Entry price
    pub entry_price: Decimal,
    /// Exit price (if closed)
    pub exit_price: Option<Decimal>,
    /// Position size in USD
    pub size_usd: Decimal,
    /// Realized P&L (if closed)
    pub realized_pnl: Option<Decimal>,
    /// When signal was generated
    pub signal_at: DateTime<Utc>,
    /// When outcome was resolved (if resolved)
    pub resolved_at: Option<DateTime<Utc>>,
}

/// Actual market outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActualOutcome {
    /// Market resolved YES
    Yes,
    /// Market resolved NO
    No,
    /// Position was closed before resolution
    ClosedEarly { exit_price: Decimal },
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_prediction_threshold() {
        let prediction = MlPrediction {
            id: "test".to_string(),
            model_id: "model1".to_string(),
            model_version: None,
            prediction: PredictionValue::Binary(BinaryPrediction::Yes),
            confidence: dec!(0.75),
            market_id: None,
            market_keywords: vec![],
            features: None,
            metadata: serde_json::Value::Null,
            generated_at: Utc::now(),
            expires_at: None,
        };

        assert!(prediction.meets_threshold(dec!(0.50)));
        assert!(prediction.meets_threshold(dec!(0.75)));
        assert!(!prediction.meets_threshold(dec!(0.80)));
    }

    #[test]
    fn test_model_performance_accuracy() {
        let mut perf = ModelPerformance::new("test_model".to_string());

        // No predictions yet
        assert_eq!(perf.accuracy(), None);

        // Record predictions
        perf.record_prediction(dec!(0.80));
        perf.record_prediction(dec!(0.70));
        perf.record_prediction(dec!(0.90));

        assert_eq!(perf.total_predictions, 3);
        assert_eq!(perf.pending_predictions, 3);

        // Record outcomes: 2 correct, 1 incorrect
        perf.record_outcome(true, dec!(50.0));
        perf.record_outcome(true, dec!(30.0));
        perf.record_outcome(false, dec!(-20.0));

        assert_eq!(perf.correct_predictions, 2);
        assert_eq!(perf.incorrect_predictions, 1);
        assert_eq!(perf.pending_predictions, 0);
        assert_eq!(perf.total_pnl, dec!(60.0));

        // Accuracy should be 2/3 = 0.666...
        let accuracy = perf.accuracy().unwrap();
        assert!(accuracy > dec!(0.66) && accuracy < dec!(0.67));
    }

    #[test]
    fn test_ml_config_defaults() {
        let config = MlConfig::default();

        assert!(config.enabled);
        assert_eq!(config.min_confidence, dec!(0.50));
        assert_eq!(config.max_confidence, dec!(1.0));
        assert_eq!(config.confidence_multiplier, dec!(1.0));
        assert_eq!(config.min_size_multiplier, dec!(0.25));
        assert_eq!(config.max_size_multiplier, dec!(1.50));
        assert_eq!(config.default_cooldown_secs, 60);
    }
}
