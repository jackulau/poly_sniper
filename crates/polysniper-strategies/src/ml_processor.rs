//! ML Signal Processor
//!
//! Processes ML model predictions with confidence-based sizing,
//! threshold filtering, and model performance tracking.

use chrono::{DateTime, Duration, Utc};
use polysniper_core::{
    BinaryPrediction, ExternalSignalEvent, MlConfig, MlPrediction, ModelConfig, ModelPerformance,
    Outcome, PredictionValue, SignalOutcome, SignalSource,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// ML Signal Processor
///
/// Handles ML model predictions with:
/// - Confidence-based position sizing
/// - Minimum confidence threshold filtering
/// - Per-model cooldown enforcement
/// - Model performance tracking
pub struct MlSignalProcessor {
    config: MlConfig,
    /// Model performance trackers
    model_performance: Arc<RwLock<HashMap<String, ModelPerformance>>>,
    /// Last signal time per model (for cooldown)
    last_signal_time: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    /// Signal count per model per day (for rate limiting)
    daily_signal_count: Arc<RwLock<HashMap<String, DailyCount>>>,
    /// Pending signal outcomes (for tracking)
    pending_outcomes: Arc<RwLock<HashMap<String, SignalOutcome>>>,
}

/// Daily signal count tracker
#[derive(Debug, Clone)]
struct DailyCount {
    date: chrono::NaiveDate,
    count: u32,
}

/// Result of processing an ML signal
#[derive(Debug, Clone)]
pub struct MlProcessingResult {
    /// Whether the signal should be executed
    pub should_execute: bool,
    /// Rejection reason (if not executing)
    pub rejection_reason: Option<String>,
    /// Calculated size multiplier (based on confidence)
    pub size_multiplier: Decimal,
    /// The parsed prediction
    pub prediction: MlPrediction,
    /// Suggested outcome based on prediction
    pub suggested_outcome: Outcome,
}

impl MlSignalProcessor {
    /// Create a new ML signal processor
    pub fn new(config: MlConfig) -> Self {
        Self {
            config,
            model_performance: Arc::new(RwLock::new(HashMap::new())),
            last_signal_time: Arc::new(RwLock::new(HashMap::new())),
            daily_signal_count: Arc::new(RwLock::new(HashMap::new())),
            pending_outcomes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if an external signal is from an ML source
    pub fn is_ml_signal(&self, signal: &ExternalSignalEvent) -> bool {
        match &signal.source {
            SignalSource::Webhook { endpoint } => {
                endpoint.contains(&self.config.ml_source_pattern)
            }
            SignalSource::Custom { name } => name.contains(&self.config.ml_source_pattern),
            _ => false,
        }
    }

    /// Try to parse ML prediction from external signal
    pub fn parse_prediction(&self, signal: &ExternalSignalEvent) -> Option<MlPrediction> {
        // First try to parse from metadata
        if let Ok(prediction) = serde_json::from_value::<MlPrediction>(signal.metadata.clone()) {
            return Some(prediction);
        }

        // Try to extract from content (JSON string)
        if let Ok(prediction) = serde_json::from_str::<MlPrediction>(&signal.content) {
            return Some(prediction);
        }

        // Try to parse from nested "prediction" field in metadata
        if let Some(pred_value) = signal.metadata.get("prediction") {
            if let Ok(prediction) = serde_json::from_value::<MlPrediction>(pred_value.clone()) {
                return Some(prediction);
            }
        }

        debug!(
            content = %signal.content,
            "Failed to parse ML prediction from signal"
        );
        None
    }

    /// Process an ML signal and determine if it should be executed
    pub async fn process_signal(
        &self,
        signal: &ExternalSignalEvent,
    ) -> Option<MlProcessingResult> {
        // Check if ML processing is enabled
        if !self.config.enabled {
            debug!("ML processing is disabled");
            return None;
        }

        // Check if this is an ML signal
        if !self.is_ml_signal(signal) {
            return None;
        }

        // Parse prediction
        let prediction = self.parse_prediction(signal)?;

        // Check if prediction is expired
        if prediction.is_expired() {
            info!(
                prediction_id = %prediction.id,
                model_id = %prediction.model_id,
                "ML prediction is expired, skipping"
            );
            return Some(MlProcessingResult {
                should_execute: false,
                rejection_reason: Some("Prediction expired".to_string()),
                size_multiplier: Decimal::ZERO,
                prediction,
                suggested_outcome: Outcome::Yes,
            });
        }

        // Get model-specific config (or defaults)
        let model_config = self.config.model_configs.get(&prediction.model_id);

        // Check if model is enabled
        if let Some(cfg) = model_config {
            if !cfg.enabled {
                info!(
                    model_id = %prediction.model_id,
                    "Model is disabled, skipping signal"
                );
                return Some(MlProcessingResult {
                    should_execute: false,
                    rejection_reason: Some(format!("Model {} is disabled", prediction.model_id)),
                    size_multiplier: Decimal::ZERO,
                    prediction,
                    suggested_outcome: Outcome::Yes,
                });
            }
        }

        // Check confidence threshold
        let min_confidence = model_config
            .and_then(|c| c.min_confidence)
            .unwrap_or(self.config.min_confidence);

        if !prediction.meets_threshold(min_confidence) {
            info!(
                prediction_id = %prediction.id,
                model_id = %prediction.model_id,
                confidence = %prediction.confidence,
                min_confidence = %min_confidence,
                "ML prediction below confidence threshold"
            );
            return Some(MlProcessingResult {
                should_execute: false,
                rejection_reason: Some(format!(
                    "Confidence {} below threshold {}",
                    prediction.confidence, min_confidence
                )),
                size_multiplier: Decimal::ZERO,
                prediction,
                suggested_outcome: Outcome::Yes,
            });
        }

        // Check cooldown
        if let Some(rejection) = self.check_cooldown(&prediction, model_config).await {
            return Some(MlProcessingResult {
                should_execute: false,
                rejection_reason: Some(rejection),
                size_multiplier: Decimal::ZERO,
                prediction,
                suggested_outcome: Outcome::Yes,
            });
        }

        // Check daily limit
        if let Some(rejection) = self.check_daily_limit(&prediction, model_config).await {
            return Some(MlProcessingResult {
                should_execute: false,
                rejection_reason: Some(rejection),
                size_multiplier: Decimal::ZERO,
                prediction,
                suggested_outcome: Outcome::Yes,
            });
        }

        // Calculate size multiplier based on confidence
        let size_multiplier = self.calculate_size_multiplier(&prediction, model_config);

        // Determine suggested outcome from prediction
        let suggested_outcome = self.prediction_to_outcome(&prediction);

        // Record the signal
        self.record_signal(&prediction).await;

        info!(
            prediction_id = %prediction.id,
            model_id = %prediction.model_id,
            confidence = %prediction.confidence,
            size_multiplier = %size_multiplier,
            outcome = ?suggested_outcome,
            "ML prediction accepted"
        );

        Some(MlProcessingResult {
            should_execute: true,
            rejection_reason: None,
            size_multiplier,
            prediction,
            suggested_outcome,
        })
    }

    /// Check if model is in cooldown
    async fn check_cooldown(
        &self,
        prediction: &MlPrediction,
        model_config: Option<&ModelConfig>,
    ) -> Option<String> {
        let cooldown_secs = model_config
            .and_then(|c| c.cooldown_secs)
            .unwrap_or(self.config.default_cooldown_secs);

        if cooldown_secs == 0 {
            return None;
        }

        let last_time = self.last_signal_time.read().await;
        if let Some(last) = last_time.get(&prediction.model_id) {
            let elapsed = Utc::now().signed_duration_since(*last);
            let cooldown = Duration::seconds(cooldown_secs as i64);

            if elapsed < cooldown {
                let remaining = (cooldown - elapsed).num_seconds();
                return Some(format!(
                    "Model {} in cooldown, {} seconds remaining",
                    prediction.model_id, remaining
                ));
            }
        }
        None
    }

    /// Check daily signal limit
    async fn check_daily_limit(
        &self,
        prediction: &MlPrediction,
        model_config: Option<&ModelConfig>,
    ) -> Option<String> {
        let max_per_day = model_config.and_then(|c| c.max_signals_per_day);

        let max_per_day = match max_per_day {
            Some(m) => m,
            None => return None, // No limit configured
        };

        let today = Utc::now().date_naive();
        let counts = self.daily_signal_count.read().await;

        if let Some(daily) = counts.get(&prediction.model_id) {
            if daily.date == today && daily.count >= max_per_day {
                return Some(format!(
                    "Model {} has reached daily limit of {} signals",
                    prediction.model_id, max_per_day
                ));
            }
        }
        None
    }

    /// Calculate size multiplier based on confidence
    fn calculate_size_multiplier(
        &self,
        prediction: &MlPrediction,
        model_config: Option<&ModelConfig>,
    ) -> Decimal {
        // Get confidence multiplier (how much confidence affects size)
        let confidence_multiplier = model_config
            .and_then(|c| c.confidence_multiplier)
            .unwrap_or(self.config.confidence_multiplier);

        // Clamp confidence to max
        let effective_confidence = prediction.confidence.min(self.config.max_confidence);

        // Base calculation: size_multiplier = confidence * multiplier
        let raw_multiplier = effective_confidence * confidence_multiplier;

        // Apply model weight if available
        let weighted_multiplier = match model_config {
            Some(cfg) => raw_multiplier * cfg.weight,
            None => raw_multiplier,
        };

        // Clamp to min/max bounds
        weighted_multiplier
            .max(self.config.min_size_multiplier)
            .min(self.config.max_size_multiplier)
    }

    /// Convert prediction to trading outcome
    fn prediction_to_outcome(&self, prediction: &MlPrediction) -> Outcome {
        match &prediction.prediction {
            PredictionValue::Binary(binary) => match binary {
                BinaryPrediction::Yes => Outcome::Yes,
                BinaryPrediction::No => Outcome::No,
            },
            PredictionValue::Probability(prob) => {
                // If probability > 0.5, predict YES
                if *prob > dec!(0.5) {
                    Outcome::Yes
                } else {
                    Outcome::No
                }
            }
            PredictionValue::Categorical(cat) => {
                // Try to match common patterns
                let lower = cat.to_lowercase();
                if lower.contains("yes") || lower.contains("true") || lower.contains("positive") {
                    Outcome::Yes
                } else {
                    Outcome::No
                }
            }
        }
    }

    /// Record a signal for tracking
    async fn record_signal(&self, prediction: &MlPrediction) {
        // Update last signal time
        {
            let mut last_time = self.last_signal_time.write().await;
            last_time.insert(prediction.model_id.clone(), Utc::now());
        }

        // Update daily count
        {
            let today = Utc::now().date_naive();
            let mut counts = self.daily_signal_count.write().await;

            let entry = counts
                .entry(prediction.model_id.clone())
                .or_insert(DailyCount {
                    date: today,
                    count: 0,
                });

            if entry.date != today {
                // Reset for new day
                entry.date = today;
                entry.count = 1;
            } else {
                entry.count += 1;
            }
        }

        // Update model performance
        {
            let mut perf = self.model_performance.write().await;
            let entry = perf
                .entry(prediction.model_id.clone())
                .or_insert_with(|| ModelPerformance::new(prediction.model_id.clone()));
            entry.record_prediction(prediction.confidence);
        }
    }

    /// Record outcome of a prediction for performance tracking
    pub async fn record_outcome(&self, prediction_id: &str, correct: bool, pnl: Decimal) {
        let mut outcomes = self.pending_outcomes.write().await;

        if let Some(outcome) = outcomes.remove(prediction_id) {
            let mut perf = self.model_performance.write().await;
            if let Some(model_perf) = perf.get_mut(&outcome.model_id) {
                model_perf.record_outcome(correct, pnl);
            }

            info!(
                prediction_id = %prediction_id,
                model_id = %outcome.model_id,
                correct = %correct,
                pnl = %pnl,
                "Recorded prediction outcome"
            );
        } else {
            warn!(
                prediction_id = %prediction_id,
                "No pending outcome found for prediction"
            );
        }
    }

    /// Store pending outcome for later resolution
    pub async fn store_pending_outcome(&self, outcome: SignalOutcome) {
        let mut outcomes = self.pending_outcomes.write().await;
        outcomes.insert(outcome.prediction_id.clone(), outcome);
    }

    /// Get model performance statistics
    pub async fn get_model_performance(&self, model_id: &str) -> Option<ModelPerformance> {
        let perf = self.model_performance.read().await;
        perf.get(model_id).cloned()
    }

    /// Get all model performance statistics
    pub async fn get_all_model_performance(&self) -> HashMap<String, ModelPerformance> {
        let perf = self.model_performance.read().await;
        perf.clone()
    }

    /// Get the configuration
    pub fn config(&self) -> &MlConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: MlConfig) {
        self.config = config;
    }
}

/// Builder for MlProcessingResult (for testing)
#[derive(Debug)]
pub struct MlProcessingResultBuilder {
    should_execute: bool,
    rejection_reason: Option<String>,
    size_multiplier: Decimal,
    prediction: Option<MlPrediction>,
    suggested_outcome: Outcome,
}

impl Default for MlProcessingResultBuilder {
    fn default() -> Self {
        Self {
            should_execute: true,
            rejection_reason: None,
            size_multiplier: Decimal::ONE,
            prediction: None,
            suggested_outcome: Outcome::Yes,
        }
    }
}

impl MlProcessingResultBuilder {
    pub fn new() -> Self {
        Self {
            should_execute: true,
            rejection_reason: None,
            size_multiplier: Decimal::ONE,
            prediction: None,
            suggested_outcome: Outcome::Yes,
        }
    }

    pub fn should_execute(mut self, value: bool) -> Self {
        self.should_execute = value;
        self
    }

    pub fn rejection_reason(mut self, reason: impl Into<String>) -> Self {
        self.rejection_reason = Some(reason.into());
        self
    }

    pub fn size_multiplier(mut self, value: Decimal) -> Self {
        self.size_multiplier = value;
        self
    }

    pub fn prediction(mut self, pred: MlPrediction) -> Self {
        self.prediction = Some(pred);
        self
    }

    pub fn suggested_outcome(mut self, outcome: Outcome) -> Self {
        self.suggested_outcome = outcome;
        self
    }

    pub fn build(self) -> MlProcessingResult {
        MlProcessingResult {
            should_execute: self.should_execute,
            rejection_reason: self.rejection_reason,
            size_multiplier: self.size_multiplier,
            prediction: self.prediction.expect("prediction is required"),
            suggested_outcome: self.suggested_outcome,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn create_test_signal(model_id: &str, confidence: Decimal) -> ExternalSignalEvent {
        let prediction = MlPrediction {
            id: format!("pred_{}", model_id),
            model_id: model_id.to_string(),
            model_version: None,
            prediction: PredictionValue::Binary(BinaryPrediction::Yes),
            confidence,
            market_id: Some("market123".to_string()),
            market_keywords: vec!["test".to_string()],
            features: None,
            metadata: serde_json::Value::Null,
            generated_at: Utc::now(),
            expires_at: None,
        };

        ExternalSignalEvent {
            source: SignalSource::Webhook {
                endpoint: "/ml-webhook".to_string(),
            },
            signal_type: "ml_prediction".to_string(),
            content: String::new(),
            market_id: None,
            keywords: vec![],
            metadata: serde_json::to_value(&prediction).unwrap(),
            received_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_confidence_threshold_filtering() {
        let config = MlConfig {
            enabled: true,
            min_confidence: dec!(0.60),
            ..Default::default()
        };
        let processor = MlSignalProcessor::new(config);

        // Signal with high confidence should be accepted
        let high_conf_signal = create_test_signal("model1", dec!(0.80));
        let result = processor.process_signal(&high_conf_signal).await;
        assert!(result.is_some());
        assert!(result.unwrap().should_execute);

        // Signal with low confidence should be rejected
        let low_conf_signal = create_test_signal("model2", dec!(0.40));
        let result = processor.process_signal(&low_conf_signal).await;
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(!result.should_execute);
        assert!(result.rejection_reason.is_some());
    }

    #[tokio::test]
    async fn test_size_multiplier_calculation() {
        let config = MlConfig {
            enabled: true,
            min_confidence: dec!(0.30),
            confidence_multiplier: dec!(1.0),
            min_size_multiplier: dec!(0.25),
            max_size_multiplier: dec!(1.50),
            ..Default::default()
        };
        let processor = MlSignalProcessor::new(config);

        // Test with 0.50 confidence
        let signal_50 = create_test_signal("model1", dec!(0.50));
        let result = processor.process_signal(&signal_50).await.unwrap();
        assert_eq!(result.size_multiplier, dec!(0.50));

        // Test with 0.80 confidence
        let signal_80 = create_test_signal("model2", dec!(0.80));
        let result = processor.process_signal(&signal_80).await.unwrap();
        assert_eq!(result.size_multiplier, dec!(0.80));

        // Test min clamping (0.20 should become 0.25)
        let signal_20 = create_test_signal("model3", dec!(0.30)); // Just above min_confidence
        let result = processor.process_signal(&signal_20).await.unwrap();
        assert_eq!(result.size_multiplier, dec!(0.30));
    }

    #[tokio::test]
    async fn test_cooldown_enforcement() {
        let config = MlConfig {
            enabled: true,
            min_confidence: dec!(0.30),
            default_cooldown_secs: 60,
            ..Default::default()
        };
        let processor = MlSignalProcessor::new(config);

        // First signal should be accepted
        let signal1 = create_test_signal("model1", dec!(0.80));
        let result = processor.process_signal(&signal1).await.unwrap();
        assert!(result.should_execute);

        // Second signal from same model should be rejected (cooldown)
        let signal2 = create_test_signal("model1", dec!(0.85));
        let result = processor.process_signal(&signal2).await.unwrap();
        assert!(!result.should_execute);
        assert!(result
            .rejection_reason
            .as_ref()
            .unwrap()
            .contains("cooldown"));

        // Signal from different model should be accepted
        let signal3 = create_test_signal("model2", dec!(0.75));
        let result = processor.process_signal(&signal3).await.unwrap();
        assert!(result.should_execute);
    }

    #[tokio::test]
    async fn test_model_performance_tracking() {
        let processor = MlSignalProcessor::new(MlConfig::default());

        // Process a signal
        let signal = create_test_signal("test_model", dec!(0.80));
        let result = processor.process_signal(&signal).await.unwrap();
        assert!(result.should_execute);

        // Check performance was recorded
        let perf = processor.get_model_performance("test_model").await;
        assert!(perf.is_some());
        let perf = perf.unwrap();
        assert_eq!(perf.total_predictions, 1);
        assert_eq!(perf.pending_predictions, 1);
    }

    #[tokio::test]
    async fn test_is_ml_signal() {
        let processor = MlSignalProcessor::new(MlConfig::default());

        // ML webhook signal
        let ml_signal = ExternalSignalEvent {
            source: SignalSource::Webhook {
                endpoint: "/api/ml-webhook/v1".to_string(),
            },
            signal_type: "prediction".to_string(),
            content: String::new(),
            market_id: None,
            keywords: vec![],
            metadata: serde_json::Value::Null,
            received_at: Utc::now(),
        };
        assert!(processor.is_ml_signal(&ml_signal));

        // Non-ML webhook signal
        let regular_signal = ExternalSignalEvent {
            source: SignalSource::Webhook {
                endpoint: "/api/news".to_string(),
            },
            signal_type: "news".to_string(),
            content: String::new(),
            market_id: None,
            keywords: vec![],
            metadata: serde_json::Value::Null,
            received_at: Utc::now(),
        };
        assert!(!processor.is_ml_signal(&regular_signal));

        // RSS signal (not ML)
        let rss_signal = ExternalSignalEvent {
            source: SignalSource::Rss {
                feed_url: "https://news.com/feed".to_string(),
            },
            signal_type: "news".to_string(),
            content: String::new(),
            market_id: None,
            keywords: vec![],
            metadata: serde_json::Value::Null,
            received_at: Utc::now(),
        };
        assert!(!processor.is_ml_signal(&rss_signal));
    }

    #[test]
    fn test_prediction_to_outcome() {
        let processor = MlSignalProcessor::new(MlConfig::default());

        // Binary YES
        let pred_yes = MlPrediction {
            id: "test".to_string(),
            model_id: "model".to_string(),
            model_version: None,
            prediction: PredictionValue::Binary(BinaryPrediction::Yes),
            confidence: dec!(0.80),
            market_id: None,
            market_keywords: vec![],
            features: None,
            metadata: serde_json::Value::Null,
            generated_at: Utc::now(),
            expires_at: None,
        };
        assert_eq!(processor.prediction_to_outcome(&pred_yes), Outcome::Yes);

        // Binary NO
        let pred_no = MlPrediction {
            prediction: PredictionValue::Binary(BinaryPrediction::No),
            ..pred_yes.clone()
        };
        assert_eq!(processor.prediction_to_outcome(&pred_no), Outcome::No);

        // Probability > 0.5
        let pred_prob_high = MlPrediction {
            prediction: PredictionValue::Probability(dec!(0.70)),
            ..pred_yes.clone()
        };
        assert_eq!(
            processor.prediction_to_outcome(&pred_prob_high),
            Outcome::Yes
        );

        // Probability < 0.5
        let pred_prob_low = MlPrediction {
            prediction: PredictionValue::Probability(dec!(0.30)),
            ..pred_yes.clone()
        };
        assert_eq!(
            processor.prediction_to_outcome(&pred_prob_low),
            Outcome::No
        );
    }
}
