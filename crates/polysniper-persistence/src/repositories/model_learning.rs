//! Model Learning Stats Repository
//!
//! Repository for persisting and retrieving online learning model statistics.

use crate::error::Result;
use crate::models::{ModelLearningStatsRecord, PredictionOutcomeRecord};
use chrono::Utc;
use rust_decimal::Decimal;
use sqlx::{Pool, Row, Sqlite};
use std::collections::HashMap;
use std::str::FromStr;
use tracing::{debug, info};

/// Repository for model learning statistics
pub struct ModelLearningRepository {
    pool: Pool<Sqlite>,
}

impl ModelLearningRepository {
    /// Create a new repository
    pub fn new(pool: Pool<Sqlite>) -> Self {
        Self { pool }
    }

    /// Save or update model learning stats
    pub async fn save_stats(&self, stats: &ModelLearningStatsRecord) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO model_learning_stats (
                model_id, ema_accuracy, adaptive_threshold, adaptive_weight,
                thompson_alpha, thompson_beta, total_predictions, correct_predictions,
                total_pnl, avg_confidence, recent_predictions, first_seen_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                ema_accuracy = excluded.ema_accuracy,
                adaptive_threshold = excluded.adaptive_threshold,
                adaptive_weight = excluded.adaptive_weight,
                thompson_alpha = excluded.thompson_alpha,
                thompson_beta = excluded.thompson_beta,
                total_predictions = excluded.total_predictions,
                correct_predictions = excluded.correct_predictions,
                total_pnl = excluded.total_pnl,
                avg_confidence = excluded.avg_confidence,
                recent_predictions = excluded.recent_predictions,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&stats.model_id)
        .bind(stats.ema_accuracy.to_string())
        .bind(stats.adaptive_threshold.to_string())
        .bind(stats.adaptive_weight.to_string())
        .bind(stats.thompson_alpha)
        .bind(stats.thompson_beta)
        .bind(stats.total_predictions)
        .bind(stats.correct_predictions)
        .bind(stats.total_pnl.to_string())
        .bind(stats.avg_confidence.to_string())
        .bind(&stats.recent_predictions)
        .bind(stats.first_seen_at.to_rfc3339())
        .bind(stats.updated_at.to_rfc3339())
        .execute(&self.pool)
        .await?;

        debug!(model_id = %stats.model_id, "Saved model learning stats");
        Ok(())
    }

    /// Save multiple model stats in a batch
    pub async fn save_all_stats(&self, stats: &HashMap<String, ModelLearningStatsRecord>) -> Result<()> {
        for record in stats.values() {
            self.save_stats(record).await?;
        }
        info!(count = stats.len(), "Saved all model learning stats");
        Ok(())
    }

    /// Load stats for a specific model
    pub async fn load_stats(&self, model_id: &str) -> Result<Option<ModelLearningStatsRecord>> {
        let row = sqlx::query(
            r#"
            SELECT model_id, ema_accuracy, adaptive_threshold, adaptive_weight,
                   thompson_alpha, thompson_beta, total_predictions, correct_predictions,
                   total_pnl, avg_confidence, recent_predictions, first_seen_at, updated_at
            FROM model_learning_stats
            WHERE model_id = ?
            "#,
        )
        .bind(model_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => Ok(Some(ModelLearningStatsRecord {
                model_id: row.get("model_id"),
                ema_accuracy: parse_decimal(row.get("ema_accuracy")),
                adaptive_threshold: parse_decimal(row.get("adaptive_threshold")),
                adaptive_weight: parse_decimal(row.get("adaptive_weight")),
                thompson_alpha: row.get("thompson_alpha"),
                thompson_beta: row.get("thompson_beta"),
                total_predictions: row.get("total_predictions"),
                correct_predictions: row.get("correct_predictions"),
                total_pnl: parse_decimal(row.get("total_pnl")),
                avg_confidence: parse_decimal(row.get("avg_confidence")),
                recent_predictions: row.get("recent_predictions"),
                first_seen_at: chrono::DateTime::parse_from_rfc3339(row.get("first_seen_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                updated_at: chrono::DateTime::parse_from_rfc3339(row.get("updated_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            })),
            None => Ok(None),
        }
    }

    /// Load all model learning stats
    pub async fn load_all_stats(&self) -> Result<HashMap<String, ModelLearningStatsRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT model_id, ema_accuracy, adaptive_threshold, adaptive_weight,
                   thompson_alpha, thompson_beta, total_predictions, correct_predictions,
                   total_pnl, avg_confidence, recent_predictions, first_seen_at, updated_at
            FROM model_learning_stats
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let mut stats = HashMap::new();
        for row in rows {
            let model_id: String = row.get("model_id");
            let record = ModelLearningStatsRecord {
                model_id: model_id.clone(),
                ema_accuracy: parse_decimal(row.get("ema_accuracy")),
                adaptive_threshold: parse_decimal(row.get("adaptive_threshold")),
                adaptive_weight: parse_decimal(row.get("adaptive_weight")),
                thompson_alpha: row.get("thompson_alpha"),
                thompson_beta: row.get("thompson_beta"),
                total_predictions: row.get("total_predictions"),
                correct_predictions: row.get("correct_predictions"),
                total_pnl: parse_decimal(row.get("total_pnl")),
                avg_confidence: parse_decimal(row.get("avg_confidence")),
                recent_predictions: row.get("recent_predictions"),
                first_seen_at: chrono::DateTime::parse_from_rfc3339(row.get("first_seen_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                updated_at: chrono::DateTime::parse_from_rfc3339(row.get("updated_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            };
            stats.insert(model_id, record);
        }

        info!(count = stats.len(), "Loaded all model learning stats");
        Ok(stats)
    }

    /// Delete stats for a model
    pub async fn delete_stats(&self, model_id: &str) -> Result<()> {
        sqlx::query("DELETE FROM model_learning_stats WHERE model_id = ?")
            .bind(model_id)
            .execute(&self.pool)
            .await?;

        debug!(model_id = %model_id, "Deleted model learning stats");
        Ok(())
    }

    /// Save a prediction outcome
    pub async fn save_prediction_outcome(&self, outcome: &PredictionOutcomeRecord) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO prediction_outcomes (
                prediction_id, model_id, market_id, confidence,
                predicted_outcome, actual_outcome, is_correct, pnl,
                predicted_at, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prediction_id) DO UPDATE SET
                actual_outcome = excluded.actual_outcome,
                is_correct = excluded.is_correct,
                pnl = excluded.pnl,
                resolved_at = excluded.resolved_at
            "#,
        )
        .bind(&outcome.prediction_id)
        .bind(&outcome.model_id)
        .bind(&outcome.market_id)
        .bind(outcome.confidence.to_string())
        .bind(&outcome.predicted_outcome)
        .bind(&outcome.actual_outcome)
        .bind(outcome.is_correct)
        .bind(outcome.pnl.map(|d| d.to_string()))
        .bind(outcome.predicted_at.to_rfc3339())
        .bind(outcome.resolved_at.map(|dt| dt.to_rfc3339()))
        .execute(&self.pool)
        .await?;

        debug!(prediction_id = %outcome.prediction_id, "Saved prediction outcome");
        Ok(())
    }

    /// Get pending predictions for a model
    pub async fn get_pending_predictions(&self, model_id: &str) -> Result<Vec<PredictionOutcomeRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, prediction_id, model_id, market_id, confidence,
                   predicted_outcome, actual_outcome, is_correct, pnl,
                   predicted_at, resolved_at
            FROM prediction_outcomes
            WHERE model_id = ? AND actual_outcome IS NULL
            ORDER BY predicted_at DESC
            "#,
        )
        .bind(model_id)
        .fetch_all(&self.pool)
        .await?;

        let outcomes = rows
            .into_iter()
            .map(|row| PredictionOutcomeRecord {
                id: row.get("id"),
                prediction_id: row.get("prediction_id"),
                model_id: row.get("model_id"),
                market_id: row.get("market_id"),
                confidence: parse_decimal(row.get("confidence")),
                predicted_outcome: row.get("predicted_outcome"),
                actual_outcome: row.get("actual_outcome"),
                is_correct: row.get("is_correct"),
                pnl: row
                    .get::<Option<String>, _>("pnl")
                    .map(|s| parse_decimal(&s)),
                predicted_at: chrono::DateTime::parse_from_rfc3339(row.get("predicted_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                resolved_at: row
                    .get::<Option<String>, _>("resolved_at")
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                    .map(|dt| dt.with_timezone(&Utc)),
            })
            .collect();

        Ok(outcomes)
    }

    /// Get predictions for a market
    pub async fn get_predictions_for_market(&self, market_id: &str) -> Result<Vec<PredictionOutcomeRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, prediction_id, model_id, market_id, confidence,
                   predicted_outcome, actual_outcome, is_correct, pnl,
                   predicted_at, resolved_at
            FROM prediction_outcomes
            WHERE market_id = ?
            ORDER BY predicted_at DESC
            "#,
        )
        .bind(market_id)
        .fetch_all(&self.pool)
        .await?;

        let outcomes = rows
            .into_iter()
            .map(|row| PredictionOutcomeRecord {
                id: row.get("id"),
                prediction_id: row.get("prediction_id"),
                model_id: row.get("model_id"),
                market_id: row.get("market_id"),
                confidence: parse_decimal(row.get("confidence")),
                predicted_outcome: row.get("predicted_outcome"),
                actual_outcome: row.get("actual_outcome"),
                is_correct: row.get("is_correct"),
                pnl: row
                    .get::<Option<String>, _>("pnl")
                    .map(|s| parse_decimal(&s)),
                predicted_at: chrono::DateTime::parse_from_rfc3339(row.get("predicted_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                resolved_at: row
                    .get::<Option<String>, _>("resolved_at")
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                    .map(|dt| dt.with_timezone(&Utc)),
            })
            .collect();

        Ok(outcomes)
    }

    /// Get model accuracy summary
    pub async fn get_model_accuracy_summary(&self) -> Result<Vec<ModelAccuracySummary>> {
        let rows = sqlx::query(
            r#"
            SELECT model_id,
                   total_predictions,
                   correct_predictions,
                   CASE WHEN total_predictions > 0
                        THEN CAST(correct_predictions AS REAL) / total_predictions
                        ELSE 0 END as accuracy,
                   total_pnl,
                   ema_accuracy,
                   updated_at
            FROM model_learning_stats
            ORDER BY accuracy DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let summaries = rows
            .into_iter()
            .map(|row| ModelAccuracySummary {
                model_id: row.get("model_id"),
                total_predictions: row.get("total_predictions"),
                correct_predictions: row.get("correct_predictions"),
                accuracy: row.get("accuracy"),
                total_pnl: parse_decimal(row.get("total_pnl")),
                ema_accuracy: parse_decimal(row.get("ema_accuracy")),
                updated_at: chrono::DateTime::parse_from_rfc3339(row.get("updated_at"))
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            })
            .collect();

        Ok(summaries)
    }

    /// Clean up old prediction records (keep last N days)
    pub async fn cleanup_old_predictions(&self, days_to_keep: i64) -> Result<u64> {
        let cutoff = Utc::now() - chrono::Duration::days(days_to_keep);

        let result = sqlx::query(
            r#"
            DELETE FROM prediction_outcomes
            WHERE predicted_at < ? AND resolved_at IS NOT NULL
            "#,
        )
        .bind(cutoff.to_rfc3339())
        .execute(&self.pool)
        .await?;

        let deleted = result.rows_affected();
        if deleted > 0 {
            info!(deleted = deleted, days_to_keep = days_to_keep, "Cleaned up old predictions");
        }

        Ok(deleted)
    }
}

/// Model accuracy summary
#[derive(Debug, Clone)]
pub struct ModelAccuracySummary {
    pub model_id: String,
    pub total_predictions: i64,
    pub correct_predictions: i64,
    pub accuracy: f64,
    pub total_pnl: Decimal,
    pub ema_accuracy: Decimal,
    pub updated_at: chrono::DateTime<Utc>,
}

/// Helper to parse decimal from string
fn parse_decimal(s: &str) -> Decimal {
    Decimal::from_str(s).unwrap_or(Decimal::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Database;
    use rust_decimal_macros::dec;

    async fn setup_test_db() -> Database {
        Database::in_memory().await.unwrap()
    }

    #[tokio::test]
    async fn test_save_and_load_stats() {
        let db = setup_test_db().await;
        let repo = ModelLearningRepository::new(db.pool().clone());

        let stats = ModelLearningStatsRecord {
            model_id: "test_model".to_string(),
            ema_accuracy: dec!(0.75),
            adaptive_threshold: dec!(0.6),
            adaptive_weight: dec!(1.2),
            thompson_alpha: 10.0,
            thompson_beta: 3.0,
            total_predictions: 13,
            correct_predictions: 10,
            total_pnl: dec!(150.0),
            avg_confidence: dec!(0.72),
            recent_predictions: None,
            first_seen_at: Utc::now(),
            updated_at: Utc::now(),
        };

        repo.save_stats(&stats).await.unwrap();

        let loaded = repo.load_stats("test_model").await.unwrap().unwrap();
        assert_eq!(loaded.model_id, "test_model");
        assert_eq!(loaded.ema_accuracy, dec!(0.75));
        assert_eq!(loaded.total_predictions, 13);
    }

    #[tokio::test]
    async fn test_save_prediction_outcome() {
        let db = setup_test_db().await;
        let repo = ModelLearningRepository::new(db.pool().clone());

        let outcome = PredictionOutcomeRecord {
            id: None,
            prediction_id: "pred_1".to_string(),
            model_id: "model_1".to_string(),
            market_id: Some("market_1".to_string()),
            confidence: dec!(0.8),
            predicted_outcome: "YES".to_string(),
            actual_outcome: None,
            is_correct: None,
            pnl: None,
            predicted_at: Utc::now(),
            resolved_at: None,
        };

        repo.save_prediction_outcome(&outcome).await.unwrap();

        let pending = repo.get_pending_predictions("model_1").await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].prediction_id, "pred_1");
    }

    #[tokio::test]
    async fn test_load_all_stats() {
        let db = setup_test_db().await;
        let repo = ModelLearningRepository::new(db.pool().clone());

        // Save multiple models
        for i in 0..3 {
            let stats = ModelLearningStatsRecord {
                model_id: format!("model_{}", i),
                ema_accuracy: dec!(0.5) + Decimal::from(i) * dec!(0.1),
                adaptive_threshold: dec!(0.6),
                adaptive_weight: dec!(1.0),
                thompson_alpha: 1.0 + i as f64,
                thompson_beta: 1.0,
                total_predictions: i + 1,
                correct_predictions: i,
                total_pnl: Decimal::ZERO,
                avg_confidence: dec!(0.7),
                recent_predictions: None,
                first_seen_at: Utc::now(),
                updated_at: Utc::now(),
            };
            repo.save_stats(&stats).await.unwrap();
        }

        let all = repo.load_all_stats().await.unwrap();
        assert_eq!(all.len(), 3);
    }
}
