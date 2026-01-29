//! Historical data loading from SQLite database

use crate::error::{BacktestError, Result};
use chrono::{DateTime, Utc};
use polysniper_core::{PriceChangeEvent, SystemEvent};
use polysniper_persistence::{Database, PriceSnapshotRecord, PriceSnapshotRepository};
use sqlx::Row;
use tracing::info;

/// Loads historical data from the persistence layer
pub struct DataLoader {
    db: Database,
}

impl DataLoader {
    /// Create a new data loader from a database path
    pub async fn new(db_path: &str) -> Result<Self> {
        let db = Database::new(db_path)
            .await
            .map_err(|e| BacktestError::DatabaseError(e.to_string()))?;
        Ok(Self { db })
    }

    /// Create a data loader from an existing database connection
    pub fn from_database(db: Database) -> Self {
        Self { db }
    }

    /// Load price snapshots within a time range
    pub async fn load_price_snapshots(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        token_filter: Option<&[String]>,
    ) -> Result<Vec<PriceSnapshotRecord>> {
        let repo = PriceSnapshotRepository::new(&self.db);

        let mut all_snapshots = Vec::new();

        if let Some(tokens) = token_filter {
            // Load snapshots for specific tokens
            for token_id in tokens {
                let snapshots = repo
                    .get_by_time_range(token_id, start, end)
                    .await
                    .map_err(|e| BacktestError::DatabaseError(e.to_string()))?;
                all_snapshots.extend(snapshots);
            }
        } else {
            // Load all snapshots in range - need to get all token IDs first
            let all_tokens = self.get_all_token_ids().await?;
            for token_id in all_tokens {
                let snapshots = repo
                    .get_by_time_range(&token_id, start, end)
                    .await
                    .map_err(|e| BacktestError::DatabaseError(e.to_string()))?;
                all_snapshots.extend(snapshots);
            }
        }

        // Sort by timestamp
        all_snapshots.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        info!(
            count = all_snapshots.len(),
            start = %start,
            end = %end,
            "Loaded price snapshots"
        );

        Ok(all_snapshots)
    }

    /// Convert price snapshots to system events for replay
    pub fn snapshots_to_events(&self, snapshots: &[PriceSnapshotRecord]) -> Vec<SystemEvent> {
        let mut events = Vec::with_capacity(snapshots.len());
        let mut prev_prices: std::collections::HashMap<String, rust_decimal::Decimal> =
            std::collections::HashMap::new();

        for snapshot in snapshots {
            let old_price = prev_prices.get(&snapshot.token_id).copied();

            let event = SystemEvent::PriceChange(PriceChangeEvent {
                market_id: snapshot.market_id.clone(),
                token_id: snapshot.token_id.clone(),
                old_price,
                new_price: snapshot.price,
                price_change_pct: old_price.map(|old| {
                    if old.is_zero() {
                        rust_decimal::Decimal::ZERO
                    } else {
                        ((snapshot.price - old) / old) * rust_decimal::Decimal::ONE_HUNDRED
                    }
                }),
                timestamp: snapshot.timestamp,
            });

            prev_prices.insert(snapshot.token_id.clone(), snapshot.price);
            events.push(event);
        }

        events
    }

    /// Load events in chronological order for replay
    pub async fn load_events(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        token_filter: Option<&[String]>,
    ) -> Result<Vec<SystemEvent>> {
        let snapshots = self.load_price_snapshots(start, end, token_filter).await?;

        if snapshots.is_empty() {
            return Err(BacktestError::NoData);
        }

        Ok(self.snapshots_to_events(&snapshots))
    }

    /// Get all unique token IDs from the database
    async fn get_all_token_ids(&self) -> Result<Vec<String>> {
        let rows = sqlx::query("SELECT DISTINCT token_id FROM price_snapshots")
            .fetch_all(self.db.pool())
            .await
            .map_err(|e| BacktestError::DatabaseError(e.to_string()))?;

        Ok(rows.iter().map(|r| r.get::<String, _>("token_id")).collect())
    }

    /// Get the time range of available data
    pub async fn get_data_range(&self) -> Result<Option<(DateTime<Utc>, DateTime<Utc>)>> {
        let row = sqlx::query(
            r#"
            SELECT
                MIN(timestamp) as min_ts,
                MAX(timestamp) as max_ts
            FROM price_snapshots
            "#,
        )
        .fetch_one(self.db.pool())
        .await
        .map_err(|e| BacktestError::DatabaseError(e.to_string()))?;

        let min_ts: Option<String> = row.get("min_ts");
        let max_ts: Option<String> = row.get("max_ts");

        match (min_ts, max_ts) {
            (Some(ref min), Some(ref max)) => {
                let start = DateTime::parse_from_rfc3339(min)
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| BacktestError::DatabaseError(e.to_string()))?;
                let end = DateTime::parse_from_rfc3339(max)
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| BacktestError::DatabaseError(e.to_string()))?;
                Ok(Some((start, end)))
            }
            _ => Ok(None),
        }
    }

    /// Get count of snapshots in a time range
    pub async fn count_snapshots(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<i64> {
        let row = sqlx::query(
            r#"
            SELECT COUNT(*) as count FROM price_snapshots
            WHERE timestamp >= ? AND timestamp <= ?
            "#,
        )
        .bind(start.to_rfc3339())
        .bind(end.to_rfc3339())
        .fetch_one(self.db.pool())
        .await
        .map_err(|e| BacktestError::DatabaseError(e.to_string()))?;

        Ok(row.get::<i64, _>("count"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_price_change_calculation() {
        let old_price = dec!(0.50);
        let new_price = dec!(0.55);
        let change_pct = ((new_price - old_price) / old_price) * rust_decimal::Decimal::ONE_HUNDRED;
        assert_eq!(change_pct, dec!(10)); // 10% increase
    }

    #[tokio::test]
    async fn test_data_loader_with_in_memory_db() {
        use polysniper_persistence::Database;

        // Create an in-memory database for testing
        let db = Database::in_memory().await.unwrap();
        let loader = DataLoader::from_database(db);

        // Test that we get no data from empty database
        let range = loader.get_data_range().await.unwrap();
        assert!(range.is_none());
    }
}
