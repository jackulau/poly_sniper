//! Price snapshot repository

use crate::{error::Result, models::PriceSnapshotRecord, Database};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use sqlx::Row;
use std::str::FromStr;

/// Repository for price snapshots
pub struct PriceSnapshotRepository<'a> {
    db: &'a Database,
}

impl<'a> PriceSnapshotRepository<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Insert a new price snapshot
    pub async fn insert(&self, snapshot: &PriceSnapshotRecord) -> Result<i64> {
        let result = sqlx::query(
            r#"
            INSERT INTO price_snapshots (market_id, token_id, price, timestamp)
            VALUES (?, ?, ?, ?)
            "#,
        )
        .bind(&snapshot.market_id)
        .bind(&snapshot.token_id)
        .bind(snapshot.price.to_string())
        .bind(snapshot.timestamp.to_rfc3339())
        .execute(self.db.pool())
        .await?;

        Ok(result.last_insert_rowid())
    }

    /// Get price history for a token
    pub async fn get_by_token(
        &self,
        token_id: &str,
        limit: i64,
    ) -> Result<Vec<PriceSnapshotRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM price_snapshots WHERE token_id = ? ORDER BY timestamp DESC LIMIT ?",
        )
        .bind(token_id)
        .bind(limit)
        .fetch_all(self.db.pool())
        .await?;

        rows.iter().map(Self::row_to_snapshot).collect()
    }

    /// Get price history within time range
    pub async fn get_by_time_range(
        &self,
        token_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<PriceSnapshotRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT * FROM price_snapshots
            WHERE token_id = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            "#,
        )
        .bind(token_id)
        .bind(start.to_rfc3339())
        .bind(end.to_rfc3339())
        .fetch_all(self.db.pool())
        .await?;

        rows.iter().map(Self::row_to_snapshot).collect()
    }

    /// Get latest price for a token
    pub async fn get_latest(&self, token_id: &str) -> Result<Option<PriceSnapshotRecord>> {
        let row = sqlx::query(
            "SELECT * FROM price_snapshots WHERE token_id = ? ORDER BY timestamp DESC LIMIT 1",
        )
        .bind(token_id)
        .fetch_optional(self.db.pool())
        .await?;

        match row {
            Some(r) => Ok(Some(Self::row_to_snapshot(&r)?)),
            None => Ok(None),
        }
    }

    /// Cleanup old snapshots, keeping only the most recent N per token
    pub async fn cleanup(&self, keep_per_token: i64) -> Result<u64> {
        // Get all token IDs
        let token_rows = sqlx::query("SELECT DISTINCT token_id FROM price_snapshots")
            .fetch_all(self.db.pool())
            .await?;

        let mut total_deleted = 0u64;

        for row in token_rows {
            let token_id: String = row.get("token_id");

            // Delete all but the most recent N snapshots for this token
            let result = sqlx::query(
                r#"
                DELETE FROM price_snapshots
                WHERE token_id = ? AND id NOT IN (
                    SELECT id FROM price_snapshots
                    WHERE token_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
                "#,
            )
            .bind(&token_id)
            .bind(&token_id)
            .bind(keep_per_token)
            .execute(self.db.pool())
            .await?;

            total_deleted += result.rows_affected();
        }

        Ok(total_deleted)
    }

    /// Get total snapshot count
    pub async fn count(&self) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM price_snapshots")
            .fetch_one(self.db.pool())
            .await?;

        Ok(row.get::<i64, _>("count"))
    }

    fn row_to_snapshot(row: &sqlx::sqlite::SqliteRow) -> Result<PriceSnapshotRecord> {
        Ok(PriceSnapshotRecord {
            id: Some(row.get("id")),
            market_id: row.get("market_id"),
            token_id: row.get("token_id"),
            price: Decimal::from_str(row.get::<&str, _>("price")).unwrap_or(Decimal::ZERO),
            timestamp: DateTime::parse_from_rfc3339(row.get::<&str, _>("timestamp"))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }
}
