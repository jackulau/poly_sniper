//! Alert repository

use crate::{
    error::Result,
    models::{AlertLevel, AlertRecord},
    Database,
};
use chrono::{DateTime, Utc};
use sqlx::Row;
use std::str::FromStr;

/// Repository for alert records
pub struct AlertRepository<'a> {
    db: &'a Database,
}

impl<'a> AlertRepository<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Insert a new alert
    pub async fn insert(&self, alert: &AlertRecord) -> Result<i64> {
        let result = sqlx::query(
            r#"
            INSERT INTO alerts (level, category, message, metadata, sent, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(alert.level.to_string())
        .bind(&alert.category)
        .bind(&alert.message)
        .bind(alert.metadata.as_ref().map(|m| m.to_string()))
        .bind(alert.sent as i32)
        .bind(alert.created_at.to_rfc3339())
        .execute(self.db.pool())
        .await?;

        Ok(result.last_insert_rowid())
    }

    /// Mark alert as sent
    pub async fn mark_sent(&self, id: i64) -> Result<()> {
        sqlx::query("UPDATE alerts SET sent = 1 WHERE id = ?")
            .bind(id)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    /// Get unsent alerts
    pub async fn get_unsent(&self) -> Result<Vec<AlertRecord>> {
        let rows = sqlx::query("SELECT * FROM alerts WHERE sent = 0 ORDER BY created_at ASC")
            .fetch_all(self.db.pool())
            .await?;

        rows.iter().map(Self::row_to_alert).collect()
    }

    /// Get alerts by level
    pub async fn get_by_level(&self, level: AlertLevel, limit: i64) -> Result<Vec<AlertRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM alerts WHERE level = ? ORDER BY created_at DESC LIMIT ?",
        )
        .bind(level.to_string())
        .bind(limit)
        .fetch_all(self.db.pool())
        .await?;

        rows.iter().map(Self::row_to_alert).collect()
    }

    /// Get recent alerts
    pub async fn get_recent(&self, limit: i64) -> Result<Vec<AlertRecord>> {
        let rows = sqlx::query("SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?")
            .bind(limit)
            .fetch_all(self.db.pool())
            .await?;

        rows.iter().map(Self::row_to_alert).collect()
    }

    /// Get alerts within time range
    pub async fn get_by_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<AlertRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM alerts WHERE created_at >= ? AND created_at <= ? ORDER BY created_at DESC",
        )
        .bind(start.to_rfc3339())
        .bind(end.to_rfc3339())
        .fetch_all(self.db.pool())
        .await?;

        rows.iter().map(Self::row_to_alert).collect()
    }

    /// Count alerts by level
    pub async fn count_by_level(&self, level: AlertLevel) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM alerts WHERE level = ?")
            .bind(level.to_string())
            .fetch_one(self.db.pool())
            .await?;

        Ok(row.get::<i64, _>("count"))
    }

    /// Cleanup old alerts
    pub async fn cleanup_old(&self, keep_days: i64) -> Result<u64> {
        let cutoff = Utc::now() - chrono::Duration::days(keep_days);

        let result = sqlx::query("DELETE FROM alerts WHERE created_at < ?")
            .bind(cutoff.to_rfc3339())
            .execute(self.db.pool())
            .await?;

        Ok(result.rows_affected())
    }

    fn row_to_alert(row: &sqlx::sqlite::SqliteRow) -> Result<AlertRecord> {
        Ok(AlertRecord {
            id: Some(row.get("id")),
            level: AlertLevel::from_str(row.get::<&str, _>("level")).unwrap_or(AlertLevel::Info),
            category: row.get("category"),
            message: row.get("message"),
            metadata: row
                .get::<Option<String>, _>("metadata")
                .and_then(|s| serde_json::from_str(&s).ok()),
            sent: row.get::<i32, _>("sent") == 1,
            created_at: DateTime::parse_from_rfc3339(row.get::<&str, _>("created_at"))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }
}
