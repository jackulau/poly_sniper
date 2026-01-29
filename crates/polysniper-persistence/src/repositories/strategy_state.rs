//! Strategy state repository

use crate::{error::Result, models::StrategyStateRecord, Database};
use chrono::{DateTime, Utc};
use sqlx::Row;

/// Repository for strategy state persistence
pub struct StrategyStateRepository<'a> {
    db: &'a Database,
}

impl<'a> StrategyStateRepository<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Save or update strategy state
    pub async fn save(&self, state: &StrategyStateRecord) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO strategy_state (strategy_id, state_data, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(strategy_id) DO UPDATE SET
                state_data = excluded.state_data,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&state.strategy_id)
        .bind(state.state_data.to_string())
        .bind(state.updated_at.to_rfc3339())
        .execute(self.db.pool())
        .await?;

        Ok(())
    }

    /// Load strategy state
    pub async fn load(&self, strategy_id: &str) -> Result<Option<StrategyStateRecord>> {
        let row = sqlx::query("SELECT * FROM strategy_state WHERE strategy_id = ?")
            .bind(strategy_id)
            .fetch_optional(self.db.pool())
            .await?;

        match row {
            Some(r) => Ok(Some(Self::row_to_state(&r)?)),
            None => Ok(None),
        }
    }

    /// Load all strategy states
    pub async fn load_all(&self) -> Result<Vec<StrategyStateRecord>> {
        let rows = sqlx::query("SELECT * FROM strategy_state")
            .fetch_all(self.db.pool())
            .await?;

        rows.iter().map(Self::row_to_state).collect()
    }

    /// Delete strategy state
    pub async fn delete(&self, strategy_id: &str) -> Result<()> {
        sqlx::query("DELETE FROM strategy_state WHERE strategy_id = ?")
            .bind(strategy_id)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    fn row_to_state(row: &sqlx::sqlite::SqliteRow) -> Result<StrategyStateRecord> {
        let state_data_str: String = row.get("state_data");
        let state_data: serde_json::Value = serde_json::from_str(&state_data_str)?;

        Ok(StrategyStateRecord {
            strategy_id: row.get("strategy_id"),
            state_data,
            updated_at: DateTime::parse_from_rfc3339(row.get::<&str, _>("updated_at"))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }
}
