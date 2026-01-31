//! Online Learning Module
//!
//! Implements an online learning system that continuously updates model weights
//! and parameters based on recent trade outcomes, allowing the trading system
//! to adapt to changing market conditions in real-time.

use chrono::{DateTime, Duration, Utc};
use polysniper_core::Outcome;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::calibration::CalibrationAnalyzer;
use crate::thompson_sampling::ThompsonSampler;

/// Configuration for online learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineLearningConfig {
    /// Whether online learning is enabled
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Alpha for exponential moving average (weight for new observations)
    #[serde(default = "default_ema_alpha")]
    pub ema_alpha: Decimal,

    /// Minimum predictions before adaptation kicks in
    #[serde(default = "default_min_predictions")]
    pub min_predictions_for_adaptation: u32,

    /// Half-life for temporal decay in hours
    #[serde(default = "default_decay_half_life")]
    pub decay_half_life_hours: u64,

    /// Maximum recent predictions to keep in memory
    #[serde(default = "default_max_recent")]
    pub max_recent_predictions: usize,

    /// Threshold configuration
    #[serde(default)]
    pub thresholds: ThresholdConfig,

    /// Thompson Sampling configuration
    #[serde(default)]
    pub thompson_sampling: ThompsonSamplingConfig,

    /// Persistence configuration
    #[serde(default)]
    pub persistence: PersistenceConfig,
}

fn default_enabled() -> bool {
    true
}

fn default_ema_alpha() -> Decimal {
    dec!(0.1)
}

fn default_min_predictions() -> u32 {
    10
}

fn default_decay_half_life() -> u64 {
    168 // 1 week
}

fn default_max_recent() -> usize {
    100
}

/// Threshold adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    /// Minimum confidence threshold
    #[serde(default = "default_min_threshold")]
    pub min_threshold: Decimal,

    /// Maximum confidence threshold
    #[serde(default = "default_max_threshold")]
    pub max_threshold: Decimal,

    /// Initial/default threshold
    #[serde(default = "default_initial_threshold")]
    pub initial_threshold: Decimal,

    /// Rate of threshold adjustment per calibration update
    #[serde(default = "default_adjustment_rate")]
    pub adjustment_rate: Decimal,
}

fn default_min_threshold() -> Decimal {
    dec!(0.40)
}

fn default_max_threshold() -> Decimal {
    dec!(0.90)
}

fn default_initial_threshold() -> Decimal {
    dec!(0.60)
}

fn default_adjustment_rate() -> Decimal {
    dec!(0.01)
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            min_threshold: default_min_threshold(),
            max_threshold: default_max_threshold(),
            initial_threshold: default_initial_threshold(),
            adjustment_rate: default_adjustment_rate(),
        }
    }
}

/// Thompson Sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThompsonSamplingConfig {
    /// Whether Thompson Sampling is enabled
    #[serde(default = "default_ts_enabled")]
    pub enabled: bool,

    /// Prior alpha (initial successes)
    #[serde(default = "default_prior_alpha")]
    pub prior_alpha: f64,

    /// Prior beta (initial failures)
    #[serde(default = "default_prior_beta")]
    pub prior_beta: f64,

    /// Minimum samples before model is eligible for selection
    #[serde(default = "default_min_samples")]
    pub min_samples_for_eligibility: u32,
}

fn default_ts_enabled() -> bool {
    true
}

fn default_prior_alpha() -> f64 {
    1.0
}

fn default_prior_beta() -> f64 {
    1.0
}

fn default_min_samples() -> u32 {
    5
}

impl Default for ThompsonSamplingConfig {
    fn default() -> Self {
        Self {
            enabled: default_ts_enabled(),
            prior_alpha: default_prior_alpha(),
            prior_beta: default_prior_beta(),
            min_samples_for_eligibility: default_min_samples(),
        }
    }
}

/// Persistence configuration for learning state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Interval between saves in seconds
    #[serde(default = "default_save_interval")]
    pub save_interval_secs: u64,

    /// Database table name
    #[serde(default = "default_db_table")]
    pub db_table: String,
}

fn default_save_interval() -> u64 {
    60
}

fn default_db_table() -> String {
    "model_learning_stats".to_string()
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            save_interval_secs: default_save_interval(),
            db_table: default_db_table(),
        }
    }
}

impl Default for OnlineLearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ema_alpha: default_ema_alpha(),
            min_predictions_for_adaptation: default_min_predictions(),
            decay_half_life_hours: default_decay_half_life(),
            max_recent_predictions: default_max_recent(),
            thresholds: ThresholdConfig::default(),
            thompson_sampling: ThompsonSamplingConfig::default(),
            persistence: PersistenceConfig::default(),
        }
    }
}

/// Trade outcome for updating model stats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeOutcome {
    /// Unique prediction ID
    pub prediction_id: String,
    /// Model that made the prediction
    pub model_id: String,
    /// Predicted outcome
    pub predicted_outcome: Outcome,
    /// Actual market outcome
    pub actual_outcome: Outcome,
    /// Confidence of the original prediction
    pub confidence: Decimal,
    /// Realized P&L from this prediction
    pub pnl: Decimal,
    /// When the outcome was resolved
    pub resolved_at: DateTime<Utc>,
}

impl TradeOutcome {
    /// Check if the prediction was correct
    pub fn is_correct(&self) -> bool {
        self.predicted_outcome == self.actual_outcome
    }
}

/// A single prediction record for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    /// Prediction ID
    pub prediction_id: String,
    /// Confidence at prediction time
    pub confidence: Decimal,
    /// Predicted outcome
    pub predicted: Outcome,
    /// Actual outcome (None if pending)
    pub actual: Option<Outcome>,
    /// Whether prediction was correct (None if pending)
    pub correct: Option<bool>,
    /// P&L from this prediction (None if pending)
    pub pnl: Option<Decimal>,
    /// When prediction was made
    pub predicted_at: DateTime<Utc>,
    /// When outcome was resolved (None if pending)
    pub resolved_at: Option<DateTime<Utc>>,
}

/// Adaptive model statistics with temporal decay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveModelStats {
    /// Model identifier
    pub model_id: String,

    /// Exponential moving average of accuracy
    pub ema_accuracy: Decimal,

    /// Recent predictions for rolling analysis
    #[serde(default)]
    pub recent_predictions: VecDeque<PredictionRecord>,

    /// Learned/adaptive confidence threshold
    pub adaptive_threshold: Decimal,

    /// Learned weight for ensemble voting
    pub adaptive_weight: Decimal,

    /// Thompson Sampling alpha (successes + prior)
    pub thompson_alpha: f64,

    /// Thompson Sampling beta (failures + prior)
    pub thompson_beta: f64,

    /// Total prediction count
    pub total_predictions: u64,

    /// Correct prediction count
    pub correct_predictions: u64,

    /// Total P&L from this model
    pub total_pnl: Decimal,

    /// Average confidence of predictions
    pub avg_confidence: Decimal,

    /// When this model was first seen
    pub first_seen_at: DateTime<Utc>,

    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl AdaptiveModelStats {
    /// Create new stats for a model
    pub fn new(model_id: &str, config: &OnlineLearningConfig) -> Self {
        let now = Utc::now();
        Self {
            model_id: model_id.to_string(),
            ema_accuracy: dec!(0.5), // Start neutral
            recent_predictions: VecDeque::with_capacity(config.max_recent_predictions),
            adaptive_threshold: config.thresholds.initial_threshold,
            adaptive_weight: Decimal::ONE,
            thompson_alpha: config.thompson_sampling.prior_alpha,
            thompson_beta: config.thompson_sampling.prior_beta,
            total_predictions: 0,
            correct_predictions: 0,
            total_pnl: Decimal::ZERO,
            avg_confidence: Decimal::ZERO,
            first_seen_at: now,
            updated_at: now,
        }
    }

    /// Calculate current accuracy
    pub fn accuracy(&self) -> Option<Decimal> {
        if self.total_predictions == 0 {
            return None;
        }
        Some(Decimal::from(self.correct_predictions) / Decimal::from(self.total_predictions))
    }

    /// Check if model has enough data for adaptation
    pub fn is_eligible(&self, min_predictions: u32) -> bool {
        self.total_predictions >= min_predictions as u64
    }

    /// Record a new prediction (before outcome is known)
    pub fn record_prediction(
        &mut self,
        prediction_id: &str,
        confidence: Decimal,
        predicted: Outcome,
        max_recent: usize,
    ) {
        // Update average confidence
        let total = Decimal::from(self.total_predictions + 1);
        let old_total = Decimal::from(self.total_predictions);
        self.avg_confidence = (self.avg_confidence * old_total + confidence) / total;

        // Add to recent predictions
        let record = PredictionRecord {
            prediction_id: prediction_id.to_string(),
            confidence,
            predicted,
            actual: None,
            correct: None,
            pnl: None,
            predicted_at: Utc::now(),
            resolved_at: None,
        };

        self.recent_predictions.push_back(record);

        // Trim if needed
        while self.recent_predictions.len() > max_recent {
            self.recent_predictions.pop_front();
        }

        self.updated_at = Utc::now();
    }

    /// Update statistics with a resolved outcome
    pub fn record_outcome(&mut self, outcome: &TradeOutcome, config: &OnlineLearningConfig) {
        let is_correct = outcome.is_correct();

        // Update counts
        self.total_predictions += 1;
        if is_correct {
            self.correct_predictions += 1;
        }

        // Update total P&L
        self.total_pnl += outcome.pnl;

        // Update EMA accuracy
        let correct_value = if is_correct {
            Decimal::ONE
        } else {
            Decimal::ZERO
        };
        self.ema_accuracy =
            config.ema_alpha * correct_value + (Decimal::ONE - config.ema_alpha) * self.ema_accuracy;

        // Update Thompson Sampling parameters
        if is_correct {
            self.thompson_alpha += 1.0;
        } else {
            self.thompson_beta += 1.0;
        }

        // Update the corresponding prediction record if it exists
        if let Some(record) = self
            .recent_predictions
            .iter_mut()
            .find(|r| r.prediction_id == outcome.prediction_id)
        {
            record.actual = Some(outcome.actual_outcome);
            record.correct = Some(is_correct);
            record.pnl = Some(outcome.pnl);
            record.resolved_at = Some(outcome.resolved_at);
        }

        // Update adaptive threshold based on calibration
        self.update_adaptive_threshold(config);

        // Update adaptive weight based on performance
        self.update_adaptive_weight(config);

        self.updated_at = Utc::now();
    }

    /// Update adaptive threshold based on calibration error
    pub fn update_adaptive_threshold(&mut self, config: &OnlineLearningConfig) {
        // Calculate calibration error from recent resolved predictions
        let resolved: Vec<_> = self
            .recent_predictions
            .iter()
            .filter(|r| r.correct.is_some())
            .collect();

        if resolved.len() < config.min_predictions_for_adaptation as usize {
            return;
        }

        // Calculate calibration error:
        // Positive = overconfident (predicted higher than actual accuracy)
        // Negative = underconfident
        let calibration_error = CalibrationAnalyzer::calculate_calibration_error(&resolved);

        // Adjust threshold based on calibration
        // If overconfident (positive error), raise threshold
        // If underconfident (negative error), lower threshold
        let adjustment = config.thresholds.adjustment_rate * calibration_error;
        let new_threshold = (self.adaptive_threshold + adjustment)
            .max(config.thresholds.min_threshold)
            .min(config.thresholds.max_threshold);

        if new_threshold != self.adaptive_threshold {
            debug!(
                model_id = %self.model_id,
                old_threshold = %self.adaptive_threshold,
                new_threshold = %new_threshold,
                calibration_error = %calibration_error,
                "Updated adaptive threshold"
            );
            self.adaptive_threshold = new_threshold;
        }
    }

    /// Update adaptive weight based on EMA accuracy
    fn update_adaptive_weight(&mut self, config: &OnlineLearningConfig) {
        if self.total_predictions < config.min_predictions_for_adaptation as u64 {
            return;
        }

        // Weight is proportional to EMA accuracy, clamped to reasonable bounds
        // If accuracy > 0.5, weight increases; if < 0.5, weight decreases
        let accuracy_factor = (self.ema_accuracy - dec!(0.5)) * dec!(2.0); // Range: -1.0 to 1.0
        let new_weight = (Decimal::ONE + accuracy_factor)
            .max(dec!(0.25))
            .min(dec!(2.0));

        self.adaptive_weight = new_weight;
    }

    /// Apply temporal decay to old predictions
    pub fn apply_temporal_decay(&mut self, half_life_hours: u64) {
        let now = Utc::now();
        let _half_life = Duration::hours(half_life_hours as i64);

        // Filter out very old predictions
        let cutoff = now - Duration::hours(half_life_hours as i64 * 4); // Keep 4 half-lives
        self.recent_predictions.retain(|p| p.predicted_at > cutoff);

        // Apply decay to Thompson Sampling parameters
        // Decay factor based on time since last update
        if let Ok(duration) = now.signed_duration_since(self.updated_at).to_std() {
            let hours_elapsed = duration.as_secs_f64() / 3600.0;
            let decay_factor = 0.5_f64.powf(hours_elapsed / half_life_hours as f64);

            // Decay towards prior (1.0, 1.0)
            self.thompson_alpha = 1.0 + (self.thompson_alpha - 1.0) * decay_factor;
            self.thompson_beta = 1.0 + (self.thompson_beta - 1.0) * decay_factor;
        }
    }
}

/// Adaptive parameters returned for signal processing
#[derive(Debug, Clone)]
pub struct AdaptiveParams {
    /// Adaptive confidence threshold
    pub confidence_threshold: Decimal,
    /// Adaptive weight for sizing
    pub weight: Decimal,
    /// Current EMA accuracy
    pub ema_accuracy: Decimal,
    /// Whether adaptation is active (has enough data)
    pub adaptation_active: bool,
}

/// Online learner that tracks and updates model performance
pub struct OnlineLearner {
    /// Model statistics with temporal decay
    model_stats: Arc<RwLock<HashMap<String, AdaptiveModelStats>>>,
    /// Learning configuration
    config: OnlineLearningConfig,
    /// Thompson sampler for model selection
    thompson_sampler: ThompsonSampler,
}

impl OnlineLearner {
    /// Create a new online learner
    pub fn new(config: OnlineLearningConfig) -> Self {
        let thompson_sampler = ThompsonSampler::new(
            config.thompson_sampling.prior_alpha,
            config.thompson_sampling.prior_beta,
        );

        Self {
            model_stats: Arc::new(RwLock::new(HashMap::new())),
            config,
            thompson_sampler,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &OnlineLearningConfig {
        &self.config
    }

    /// Record a new prediction (before outcome is known)
    pub async fn record_prediction(
        &self,
        model_id: &str,
        prediction_id: &str,
        confidence: Decimal,
        predicted: Outcome,
    ) {
        let mut stats = self.model_stats.write().await;
        let model = stats
            .entry(model_id.to_string())
            .or_insert_with(|| AdaptiveModelStats::new(model_id, &self.config));

        model.record_prediction(
            prediction_id,
            confidence,
            predicted,
            self.config.max_recent_predictions,
        );

        debug!(
            model_id = %model_id,
            prediction_id = %prediction_id,
            confidence = %confidence,
            "Recorded new prediction"
        );
    }

    /// Process a resolved market and update model stats
    pub async fn record_outcome(&self, outcome: TradeOutcome) -> Result<(), String> {
        let mut stats = self.model_stats.write().await;

        let model = stats
            .entry(outcome.model_id.clone())
            .or_insert_with(|| AdaptiveModelStats::new(&outcome.model_id, &self.config));

        let is_correct = outcome.is_correct();
        let pnl = outcome.pnl;

        model.record_outcome(&outcome, &self.config);

        info!(
            prediction_id = %outcome.prediction_id,
            model_id = %outcome.model_id,
            correct = is_correct,
            pnl = %pnl,
            ema_accuracy = %model.ema_accuracy,
            total_predictions = model.total_predictions,
            "Recorded outcome and updated model stats"
        );

        Ok(())
    }

    /// Get current adaptive parameters for a model
    pub async fn get_adaptive_params(&self, model_id: &str) -> Option<AdaptiveParams> {
        let stats = self.model_stats.read().await;
        stats.get(model_id).map(|s| AdaptiveParams {
            confidence_threshold: s.adaptive_threshold,
            weight: s.adaptive_weight,
            ema_accuracy: s.ema_accuracy,
            adaptation_active: s.is_eligible(self.config.min_predictions_for_adaptation),
        })
    }

    /// Get default adaptive parameters (when no model stats exist)
    pub fn default_params(&self) -> AdaptiveParams {
        AdaptiveParams {
            confidence_threshold: self.config.thresholds.initial_threshold,
            weight: Decimal::ONE,
            ema_accuracy: dec!(0.5),
            adaptation_active: false,
        }
    }

    /// Sample from posterior using Thompson Sampling to select models
    /// Returns models sorted by sampled value (highest first)
    pub async fn thompson_sample(&self) -> Vec<(String, f64)> {
        if !self.config.thompson_sampling.enabled {
            return Vec::new();
        }

        let stats = self.model_stats.read().await;
        let min_samples = self.config.thompson_sampling.min_samples_for_eligibility;

        let mut samples: Vec<(String, f64)> = stats
            .iter()
            .filter(|(_, s)| s.is_eligible(min_samples))
            .map(|(id, s)| {
                let sample = self.thompson_sampler.sample(s.thompson_alpha, s.thompson_beta);
                (id.clone(), sample)
            })
            .collect();

        // Sort by sampled value (highest first)
        samples.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        samples
    }

    /// Get statistics for all models
    pub async fn get_all_stats(&self) -> HashMap<String, AdaptiveModelStats> {
        let stats = self.model_stats.read().await;
        stats.clone()
    }

    /// Get statistics for a specific model
    pub async fn get_model_stats(&self, model_id: &str) -> Option<AdaptiveModelStats> {
        let stats = self.model_stats.read().await;
        stats.get(model_id).cloned()
    }

    /// Apply temporal decay to all models
    pub async fn apply_decay(&self) {
        let mut stats = self.model_stats.write().await;
        for model in stats.values_mut() {
            model.apply_temporal_decay(self.config.decay_half_life_hours);
        }
    }

    /// Load stats from persistence (to be called at startup)
    pub async fn load_stats(&self, stats: HashMap<String, AdaptiveModelStats>) {
        let mut current = self.model_stats.write().await;
        *current = stats;
        info!(
            models_loaded = current.len(),
            "Loaded model stats from persistence"
        );
    }

    /// Get stats for persistence (to be called periodically)
    pub async fn get_stats_for_persistence(&self) -> HashMap<String, AdaptiveModelStats> {
        self.model_stats.read().await.clone()
    }

    /// Check if a model should be used based on current adaptive parameters
    pub async fn should_use_model(&self, model_id: &str, confidence: Decimal) -> bool {
        let params = self.get_adaptive_params(model_id).await;

        match params {
            Some(p) => {
                if !p.adaptation_active {
                    // Not enough data yet, use default threshold
                    confidence >= self.config.thresholds.initial_threshold
                } else {
                    confidence >= p.confidence_threshold
                }
            }
            None => {
                // No stats yet, use default threshold
                confidence >= self.config.thresholds.initial_threshold
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_outcome_is_correct() {
        let correct_outcome = TradeOutcome {
            prediction_id: "test1".to_string(),
            model_id: "model1".to_string(),
            predicted_outcome: Outcome::Yes,
            actual_outcome: Outcome::Yes,
            confidence: dec!(0.8),
            pnl: dec!(100.0),
            resolved_at: Utc::now(),
        };
        assert!(correct_outcome.is_correct());

        let incorrect_outcome = TradeOutcome {
            prediction_id: "test2".to_string(),
            model_id: "model1".to_string(),
            predicted_outcome: Outcome::Yes,
            actual_outcome: Outcome::No,
            confidence: dec!(0.8),
            pnl: dec!(-50.0),
            resolved_at: Utc::now(),
        };
        assert!(!incorrect_outcome.is_correct());
    }

    #[test]
    fn test_adaptive_model_stats_new() {
        let config = OnlineLearningConfig::default();
        let stats = AdaptiveModelStats::new("test_model", &config);

        assert_eq!(stats.model_id, "test_model");
        assert_eq!(stats.ema_accuracy, dec!(0.5));
        assert_eq!(stats.adaptive_threshold, config.thresholds.initial_threshold);
        assert_eq!(stats.total_predictions, 0);
        assert_eq!(stats.correct_predictions, 0);
    }

    #[tokio::test]
    async fn test_online_learner_record_outcome() {
        let config = OnlineLearningConfig::default();
        let learner = OnlineLearner::new(config);

        let outcome = TradeOutcome {
            prediction_id: "pred1".to_string(),
            model_id: "model1".to_string(),
            predicted_outcome: Outcome::Yes,
            actual_outcome: Outcome::Yes,
            confidence: dec!(0.75),
            pnl: dec!(50.0),
            resolved_at: Utc::now(),
        };

        learner.record_outcome(outcome).await.unwrap();

        let stats = learner.get_model_stats("model1").await.unwrap();
        assert_eq!(stats.total_predictions, 1);
        assert_eq!(stats.correct_predictions, 1);
        assert!(stats.ema_accuracy > dec!(0.5)); // Should increase
    }

    #[tokio::test]
    async fn test_online_learner_adaptive_params() {
        let config = OnlineLearningConfig::default();
        let learner = OnlineLearner::new(config.clone());

        // No stats yet
        let params = learner.get_adaptive_params("unknown_model").await;
        assert!(params.is_none());

        // Record some outcomes
        for i in 0..15 {
            let outcome = TradeOutcome {
                prediction_id: format!("pred{}", i),
                model_id: "model1".to_string(),
                predicted_outcome: if i % 2 == 0 { Outcome::Yes } else { Outcome::No },
                actual_outcome: if i % 3 != 0 {
                    if i % 2 == 0 { Outcome::Yes } else { Outcome::No }
                } else {
                    Outcome::Yes
                },
                confidence: dec!(0.7),
                pnl: if i % 3 != 0 { dec!(10.0) } else { dec!(-10.0) },
                resolved_at: Utc::now(),
            };
            learner.record_outcome(outcome).await.unwrap();
        }

        // Now should have adaptive params
        let params = learner.get_adaptive_params("model1").await.unwrap();
        assert!(params.adaptation_active);
    }

    #[tokio::test]
    async fn test_thompson_sampling() {
        let config = OnlineLearningConfig::default();
        let learner = OnlineLearner::new(config);

        // Record outcomes for multiple models
        for model_num in 0..3 {
            for i in 0..10 {
                let is_correct = if model_num == 0 {
                    i % 5 != 0 // 80% accuracy
                } else if model_num == 1 {
                    i % 2 == 0 // 50% accuracy
                } else {
                    i % 10 == 0 // 10% accuracy
                };

                let outcome = TradeOutcome {
                    prediction_id: format!("pred_{}_{}", model_num, i),
                    model_id: format!("model{}", model_num),
                    predicted_outcome: Outcome::Yes,
                    actual_outcome: if is_correct { Outcome::Yes } else { Outcome::No },
                    confidence: dec!(0.7),
                    pnl: if is_correct { dec!(10.0) } else { dec!(-10.0) },
                    resolved_at: Utc::now(),
                };
                learner.record_outcome(outcome).await.unwrap();
            }
        }

        // Thompson sample should generally favor higher-accuracy models
        let samples = learner.thompson_sample().await;
        assert_eq!(samples.len(), 3);
        // Note: Due to randomness, we can't assert exact ordering
    }
}
