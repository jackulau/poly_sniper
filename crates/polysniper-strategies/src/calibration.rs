//! Calibration Analysis Module
//!
//! Analyzes model calibration by comparing predicted confidence levels
//! with actual accuracy. A well-calibrated model should have predictions
//! at 70% confidence that are correct 70% of the time.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

use crate::online_learning::PredictionRecord;

/// Calibration analyzer for model confidence vs accuracy
#[derive(Debug, Clone)]
pub struct CalibrationAnalyzer;

impl CalibrationAnalyzer {
    /// Calculate calibration error from prediction records
    /// Returns positive value if overconfident, negative if underconfident
    pub fn calculate_calibration_error(records: &[&PredictionRecord]) -> Decimal {
        if records.is_empty() {
            return Decimal::ZERO;
        }

        let resolved: Vec<_> = records
            .iter()
            .filter(|r| r.correct.is_some())
            .collect();

        if resolved.is_empty() {
            return Decimal::ZERO;
        }

        // Calculate average confidence and average accuracy
        let mut total_confidence = Decimal::ZERO;
        let mut total_correct = 0u64;

        for record in &resolved {
            total_confidence += record.confidence;
            if record.correct.unwrap_or(false) {
                total_correct += 1;
            }
        }

        let avg_confidence = total_confidence / Decimal::from(resolved.len() as u64);
        let avg_accuracy = Decimal::from(total_correct) / Decimal::from(resolved.len() as u64);

        // Calibration error = avg_confidence - avg_accuracy
        // Positive = overconfident, Negative = underconfident
        avg_confidence - avg_accuracy
    }

    /// Calculate binned calibration metrics (ECE - Expected Calibration Error)
    /// Groups predictions into bins by confidence and compares with accuracy
    pub fn calculate_ece(records: &[&PredictionRecord], num_bins: usize) -> CalibrationMetrics {
        let resolved: Vec<_> = records
            .iter()
            .filter(|r| r.correct.is_some())
            .collect();

        if resolved.is_empty() {
            return CalibrationMetrics::default();
        }

        let bin_width = dec!(1.0) / Decimal::from(num_bins as u64);
        let mut bins: Vec<CalibrationBin> = (0..num_bins)
            .map(|i| CalibrationBin {
                lower_bound: bin_width * Decimal::from(i as u64),
                upper_bound: bin_width * Decimal::from((i + 1) as u64),
                total_confidence: Decimal::ZERO,
                correct_count: 0,
                total_count: 0,
            })
            .collect();

        // Assign predictions to bins
        for record in &resolved {
            let conf = record.confidence;
            let bin_idx = ((conf / bin_width).floor())
                .min(Decimal::from((num_bins - 1) as u64))
                .to_string()
                .parse::<usize>()
                .unwrap_or(0);

            bins[bin_idx].total_confidence += conf;
            bins[bin_idx].total_count += 1;
            if record.correct.unwrap_or(false) {
                bins[bin_idx].correct_count += 1;
            }
        }

        // Calculate ECE
        let total_samples = resolved.len() as u64;
        let mut ece = Decimal::ZERO;
        let mut max_calibration_error = Decimal::ZERO;

        for bin in &bins {
            if bin.total_count > 0 {
                let avg_conf = bin.total_confidence / Decimal::from(bin.total_count);
                let accuracy = Decimal::from(bin.correct_count) / Decimal::from(bin.total_count);
                let bin_error = (avg_conf - accuracy).abs();
                let weight = Decimal::from(bin.total_count) / Decimal::from(total_samples);
                ece += weight * bin_error;
                max_calibration_error = max_calibration_error.max(bin_error);
            }
        }

        CalibrationMetrics {
            expected_calibration_error: ece,
            max_calibration_error,
            bins,
            total_samples,
        }
    }

    /// Calculate reliability diagram data for visualization
    pub fn reliability_diagram(
        records: &[&PredictionRecord],
        num_bins: usize,
    ) -> Vec<ReliabilityPoint> {
        let metrics = Self::calculate_ece(records, num_bins);

        metrics
            .bins
            .iter()
            .filter(|b| b.total_count > 0)
            .map(|b| {
                let avg_conf = b.total_confidence / Decimal::from(b.total_count);
                let accuracy = Decimal::from(b.correct_count) / Decimal::from(b.total_count);
                ReliabilityPoint {
                    mean_predicted_confidence: avg_conf,
                    actual_accuracy: accuracy,
                    sample_count: b.total_count,
                    bin_lower: b.lower_bound,
                    bin_upper: b.upper_bound,
                }
            })
            .collect()
    }

    /// Determine if model is overconfident, underconfident, or well-calibrated
    pub fn assess_calibration(error: Decimal) -> CalibrationAssessment {
        let threshold = dec!(0.05); // 5% threshold

        if error > threshold {
            CalibrationAssessment::Overconfident
        } else if error < -threshold {
            CalibrationAssessment::Underconfident
        } else {
            CalibrationAssessment::WellCalibrated
        }
    }

    /// Calculate Brier score (mean squared error of probability predictions)
    pub fn brier_score(records: &[&PredictionRecord]) -> Option<Decimal> {
        let resolved: Vec<_> = records
            .iter()
            .filter(|r| r.correct.is_some())
            .collect();

        if resolved.is_empty() {
            return None;
        }

        let mut total_squared_error = Decimal::ZERO;

        for record in &resolved {
            let confidence = record.confidence;
            let actual = if record.correct.unwrap_or(false) {
                Decimal::ONE
            } else {
                Decimal::ZERO
            };
            let error = confidence - actual;
            total_squared_error += error * error;
        }

        Some(total_squared_error / Decimal::from(resolved.len() as u64))
    }

    /// Decompose Brier score into reliability, resolution, and uncertainty
    pub fn brier_decomposition(records: &[&PredictionRecord]) -> Option<BrierDecomposition> {
        let metrics = Self::calculate_ece(records, 10);

        if metrics.total_samples == 0 {
            return None;
        }

        let n = metrics.total_samples as f64;

        // Overall base rate (fraction of correct predictions)
        let resolved: Vec<_> = records
            .iter()
            .filter(|r| r.correct.is_some())
            .collect();

        let total_correct: u64 = resolved
            .iter()
            .filter(|r| r.correct.unwrap_or(false))
            .count() as u64;

        let base_rate = total_correct as f64 / n;

        // Uncertainty = p(1-p) where p is base rate
        let uncertainty = base_rate * (1.0 - base_rate);

        // Calculate reliability and resolution from bins
        let mut reliability = 0.0;
        let mut resolution = 0.0;

        for bin in &metrics.bins {
            if bin.total_count > 0 {
                let n_k = bin.total_count as f64;
                let avg_conf = (bin.total_confidence / Decimal::from(bin.total_count))
                    .to_string()
                    .parse::<f64>()
                    .unwrap_or(0.0);
                let accuracy = bin.correct_count as f64 / n_k;

                reliability += n_k * (avg_conf - accuracy).powi(2);
                resolution += n_k * (accuracy - base_rate).powi(2);
            }
        }

        reliability /= n;
        resolution /= n;

        Some(BrierDecomposition {
            reliability: Decimal::try_from(reliability).unwrap_or(Decimal::ZERO),
            resolution: Decimal::try_from(resolution).unwrap_or(Decimal::ZERO),
            uncertainty: Decimal::try_from(uncertainty).unwrap_or(Decimal::ZERO),
        })
    }
}

/// Calibration metrics calculated from predictions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error (weighted average of bin calibration errors)
    pub expected_calibration_error: Decimal,
    /// Maximum calibration error across any bin
    pub max_calibration_error: Decimal,
    /// Individual bin statistics
    pub bins: Vec<CalibrationBin>,
    /// Total number of samples analyzed
    pub total_samples: u64,
}

/// A single bin for calibration analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalibrationBin {
    /// Lower bound of confidence range
    pub lower_bound: Decimal,
    /// Upper bound of confidence range
    pub upper_bound: Decimal,
    /// Sum of all confidence values in bin
    pub total_confidence: Decimal,
    /// Number of correct predictions in bin
    pub correct_count: u64,
    /// Total predictions in bin
    pub total_count: u64,
}

impl CalibrationBin {
    /// Get the average confidence in this bin
    pub fn avg_confidence(&self) -> Option<Decimal> {
        if self.total_count > 0 {
            Some(self.total_confidence / Decimal::from(self.total_count))
        } else {
            None
        }
    }

    /// Get the accuracy in this bin
    pub fn accuracy(&self) -> Option<Decimal> {
        if self.total_count > 0 {
            Some(Decimal::from(self.correct_count) / Decimal::from(self.total_count))
        } else {
            None
        }
    }
}

/// Point in a reliability diagram
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityPoint {
    /// Mean predicted confidence in this bin
    pub mean_predicted_confidence: Decimal,
    /// Actual fraction of correct predictions
    pub actual_accuracy: Decimal,
    /// Number of samples in this bin
    pub sample_count: u64,
    /// Bin lower bound
    pub bin_lower: Decimal,
    /// Bin upper bound
    pub bin_upper: Decimal,
}

/// Assessment of model calibration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationAssessment {
    /// Model predictions are too confident
    Overconfident,
    /// Model predictions are not confident enough
    Underconfident,
    /// Model is well-calibrated
    WellCalibrated,
}

impl std::fmt::Display for CalibrationAssessment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrationAssessment::Overconfident => write!(f, "Overconfident"),
            CalibrationAssessment::Underconfident => write!(f, "Underconfident"),
            CalibrationAssessment::WellCalibrated => write!(f, "Well-Calibrated"),
        }
    }
}

/// Brier score decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrierDecomposition {
    /// Reliability (calibration component) - lower is better
    pub reliability: Decimal,
    /// Resolution (discrimination component) - higher is better
    pub resolution: Decimal,
    /// Uncertainty (inherent variability) - fixed for dataset
    pub uncertainty: Decimal,
}

impl BrierDecomposition {
    /// Calculate total Brier score from components
    /// Brier = Reliability - Resolution + Uncertainty
    pub fn brier_score(&self) -> Decimal {
        self.reliability - self.resolution + self.uncertainty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use polysniper_core::Outcome;

    fn make_record(confidence: Decimal, correct: bool) -> PredictionRecord {
        PredictionRecord {
            prediction_id: "test".to_string(),
            confidence,
            predicted: Outcome::Yes,
            actual: Some(if correct { Outcome::Yes } else { Outcome::No }),
            correct: Some(correct),
            pnl: Some(if correct { dec!(10.0) } else { dec!(-10.0) }),
            predicted_at: Utc::now(),
            resolved_at: Some(Utc::now()),
        }
    }

    #[test]
    fn test_calibration_error_overconfident() {
        // High confidence, low accuracy = overconfident
        let records: Vec<PredictionRecord> = (0..10)
            .map(|i| make_record(dec!(0.9), i < 5)) // 90% confidence, 50% accuracy
            .collect();

        let refs: Vec<_> = records.iter().collect();
        let error = CalibrationAnalyzer::calculate_calibration_error(&refs);

        // Should be positive (overconfident)
        assert!(error > dec!(0.3));
        assert_eq!(
            CalibrationAnalyzer::assess_calibration(error),
            CalibrationAssessment::Overconfident
        );
    }

    #[test]
    fn test_calibration_error_underconfident() {
        // Low confidence, high accuracy = underconfident
        let records: Vec<PredictionRecord> = (0..10)
            .map(|i| make_record(dec!(0.5), i < 9)) // 50% confidence, 90% accuracy
            .collect();

        let refs: Vec<_> = records.iter().collect();
        let error = CalibrationAnalyzer::calculate_calibration_error(&refs);

        // Should be negative (underconfident)
        assert!(error < dec!(-0.3));
        assert_eq!(
            CalibrationAnalyzer::assess_calibration(error),
            CalibrationAssessment::Underconfident
        );
    }

    #[test]
    fn test_calibration_error_well_calibrated() {
        // Confidence matches accuracy = well calibrated
        let records: Vec<PredictionRecord> = (0..10)
            .map(|i| make_record(dec!(0.7), i < 7)) // 70% confidence, 70% accuracy
            .collect();

        let refs: Vec<_> = records.iter().collect();
        let error = CalibrationAnalyzer::calculate_calibration_error(&refs);

        // Should be close to zero
        assert!(error.abs() < dec!(0.05));
        assert_eq!(
            CalibrationAnalyzer::assess_calibration(error),
            CalibrationAssessment::WellCalibrated
        );
    }

    #[test]
    fn test_ece_calculation() {
        // Create records with varying confidence
        let mut records = Vec::new();

        // Bin 0.0-0.2: 2 predictions, 0 correct (0% accuracy)
        records.push(make_record(dec!(0.1), false));
        records.push(make_record(dec!(0.15), false));

        // Bin 0.4-0.6: 4 predictions, 2 correct (50% accuracy)
        records.push(make_record(dec!(0.5), true));
        records.push(make_record(dec!(0.55), true));
        records.push(make_record(dec!(0.45), false));
        records.push(make_record(dec!(0.5), false));

        // Bin 0.8-1.0: 4 predictions, 4 correct (100% accuracy)
        records.push(make_record(dec!(0.9), true));
        records.push(make_record(dec!(0.85), true));
        records.push(make_record(dec!(0.95), true));
        records.push(make_record(dec!(0.9), true));

        let refs: Vec<_> = records.iter().collect();
        let metrics = CalibrationAnalyzer::calculate_ece(&refs, 5);

        assert_eq!(metrics.total_samples, 10);
        assert!(metrics.expected_calibration_error >= Decimal::ZERO);
    }

    #[test]
    fn test_brier_score() {
        // Perfect predictions
        let perfect: Vec<PredictionRecord> = (0..10)
            .map(|_| make_record(dec!(1.0), true))
            .collect();

        let refs: Vec<_> = perfect.iter().collect();
        let score = CalibrationAnalyzer::brier_score(&refs).unwrap();
        assert_eq!(score, Decimal::ZERO);

        // Worst predictions (100% confident, 0% correct)
        let worst: Vec<PredictionRecord> = (0..10)
            .map(|_| make_record(dec!(1.0), false))
            .collect();

        let refs: Vec<_> = worst.iter().collect();
        let score = CalibrationAnalyzer::brier_score(&refs).unwrap();
        assert_eq!(score, Decimal::ONE);
    }

    #[test]
    fn test_reliability_diagram() {
        let records: Vec<PredictionRecord> = vec![
            make_record(dec!(0.3), false),
            make_record(dec!(0.5), true),
            make_record(dec!(0.7), true),
            make_record(dec!(0.9), true),
        ];

        let refs: Vec<_> = records.iter().collect();
        let diagram = CalibrationAnalyzer::reliability_diagram(&refs, 5);

        assert!(!diagram.is_empty());
        for point in &diagram {
            assert!(point.mean_predicted_confidence >= Decimal::ZERO);
            assert!(point.mean_predicted_confidence <= Decimal::ONE);
            assert!(point.actual_accuracy >= Decimal::ZERO);
            assert!(point.actual_accuracy <= Decimal::ONE);
        }
    }
}
