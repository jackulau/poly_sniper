//! Position history repository
//!
//! Tracks position lifecycle from open to close for analysis.

use crate::{error::Result, Database};
use chrono::{DateTime, Utc};
use polysniper_core::Side;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::Row;
use std::str::FromStr;

/// Position history record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionHistoryRecord {
    pub id: Option<i64>,
    pub market_id: String,
    pub token_id: String,
    pub side: Side,
    pub entry_price: Decimal,
    pub exit_price: Option<Decimal>,
    pub size: Decimal,
    pub realized_pnl: Option<Decimal>,
    pub fees: Decimal,
    pub opened_at: DateTime<Utc>,
    pub closed_at: Option<DateTime<Utc>>,
    pub strategy_id: Option<String>,
}

/// Position summary for a market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSummary {
    pub market_id: String,
    pub total_positions: i64,
    pub open_positions: i64,
    pub closed_positions: i64,
    pub total_realized_pnl: Decimal,
    pub total_fees: Decimal,
    pub avg_holding_time_hours: Option<f64>,
}

/// Repository for position history
pub struct PositionHistoryRepository<'a> {
    db: &'a Database,
}

impl<'a> PositionHistoryRepository<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Open a new position
    #[allow(clippy::too_many_arguments)]
    pub async fn open_position(
        &self,
        market_id: &str,
        token_id: &str,
        side: Side,
        entry_price: Decimal,
        size: Decimal,
        fees: Decimal,
        strategy_id: Option<&str>,
    ) -> Result<i64> {
        let now = Utc::now();

        let result = sqlx::query(
            r#"
            INSERT INTO position_history (
                market_id, token_id, side, entry_price, size, fees, opened_at, strategy_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(market_id)
        .bind(token_id)
        .bind(side.to_string())
        .bind(entry_price.to_string())
        .bind(size.to_string())
        .bind(fees.to_string())
        .bind(now.to_rfc3339())
        .bind(strategy_id)
        .execute(self.db.pool())
        .await?;

        Ok(result.last_insert_rowid())
    }

    /// Close a position
    pub async fn close_position(
        &self,
        id: i64,
        exit_price: Decimal,
        realized_pnl: Decimal,
        exit_fees: Decimal,
    ) -> Result<()> {
        let now = Utc::now();

        sqlx::query(
            r#"
            UPDATE position_history SET
                exit_price = ?,
                realized_pnl = ?,
                fees = CAST((CAST(fees AS REAL) + ?) AS TEXT),
                closed_at = ?
            WHERE id = ?
            "#,
        )
        .bind(exit_price.to_string())
        .bind(realized_pnl.to_string())
        .bind(exit_fees.to_string())
        .bind(now.to_rfc3339())
        .bind(id)
        .execute(self.db.pool())
        .await?;

        Ok(())
    }

    /// Get position by ID
    pub async fn get_by_id(&self, id: i64) -> Result<Option<PositionHistoryRecord>> {
        let row = sqlx::query("SELECT * FROM position_history WHERE id = ?")
            .bind(id)
            .fetch_optional(self.db.pool())
            .await?;

        match row {
            Some(r) => Ok(Some(Self::row_to_record(&r)?)),
            None => Ok(None),
        }
    }

    /// Get open position for a market
    pub async fn get_open_position(&self, market_id: &str, token_id: &str) -> Result<Option<PositionHistoryRecord>> {
        let row = sqlx::query(
            "SELECT * FROM position_history WHERE market_id = ? AND token_id = ? AND closed_at IS NULL ORDER BY opened_at DESC LIMIT 1"
        )
        .bind(market_id)
        .bind(token_id)
        .fetch_optional(self.db.pool())
        .await?;

        match row {
            Some(r) => Ok(Some(Self::row_to_record(&r)?)),
            None => Ok(None),
        }
    }

    /// Get all open positions
    pub async fn get_open_positions(&self) -> Result<Vec<PositionHistoryRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM position_history WHERE closed_at IS NULL ORDER BY opened_at DESC"
        )
        .fetch_all(self.db.pool())
        .await?;

        rows.iter().map(Self::row_to_record).collect()
    }

    /// Get closed positions for a market
    pub async fn get_closed_positions(&self, market_id: &str, limit: i64) -> Result<Vec<PositionHistoryRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM position_history WHERE market_id = ? AND closed_at IS NOT NULL ORDER BY closed_at DESC LIMIT ?"
        )
        .bind(market_id)
        .bind(limit)
        .fetch_all(self.db.pool())
        .await?;

        rows.iter().map(Self::row_to_record).collect()
    }

    /// Get positions by strategy
    pub async fn get_by_strategy(&self, strategy_id: &str, limit: i64) -> Result<Vec<PositionHistoryRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM position_history WHERE strategy_id = ? ORDER BY opened_at DESC LIMIT ?"
        )
        .bind(strategy_id)
        .bind(limit)
        .fetch_all(self.db.pool())
        .await?;

        rows.iter().map(Self::row_to_record).collect()
    }

    /// Get positions within time range
    pub async fn get_by_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<PositionHistoryRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM position_history WHERE opened_at >= ? AND opened_at <= ? ORDER BY opened_at DESC"
        )
        .bind(start.to_rfc3339())
        .bind(end.to_rfc3339())
        .fetch_all(self.db.pool())
        .await?;

        rows.iter().map(Self::row_to_record).collect()
    }

    /// Get recent positions
    pub async fn get_recent(&self, limit: i64) -> Result<Vec<PositionHistoryRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM position_history ORDER BY opened_at DESC LIMIT ?"
        )
        .bind(limit)
        .fetch_all(self.db.pool())
        .await?;

        rows.iter().map(Self::row_to_record).collect()
    }

    /// Get position summary for a market
    pub async fn get_market_summary(&self, market_id: &str) -> Result<PositionSummary> {
        let row = sqlx::query(
            r#"
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN closed_at IS NULL THEN 1 END) as open_count,
                COUNT(CASE WHEN closed_at IS NOT NULL THEN 1 END) as closed_count,
                COALESCE(SUM(CASE WHEN realized_pnl IS NOT NULL THEN CAST(realized_pnl AS REAL) ELSE 0 END), 0) as total_pnl,
                COALESCE(SUM(CAST(fees AS REAL)), 0) as total_fees,
                AVG(
                    CASE WHEN closed_at IS NOT NULL THEN
                        (julianday(closed_at) - julianday(opened_at)) * 24
                    END
                ) as avg_holding_hours
            FROM position_history
            WHERE market_id = ?
            "#,
        )
        .bind(market_id)
        .fetch_one(self.db.pool())
        .await?;

        Ok(PositionSummary {
            market_id: market_id.to_string(),
            total_positions: row.get("total"),
            open_positions: row.get("open_count"),
            closed_positions: row.get("closed_count"),
            total_realized_pnl: Decimal::from_f64_retain(row.get::<f64, _>("total_pnl"))
                .unwrap_or(Decimal::ZERO),
            total_fees: Decimal::from_f64_retain(row.get::<f64, _>("total_fees"))
                .unwrap_or(Decimal::ZERO),
            avg_holding_time_hours: row.get("avg_holding_hours"),
        })
    }

    /// Get overall position statistics
    pub async fn get_overall_stats(&self) -> Result<(i64, i64, Decimal)> {
        let row = sqlx::query(
            r#"
            SELECT
                COUNT(CASE WHEN closed_at IS NULL THEN 1 END) as open_count,
                COUNT(CASE WHEN closed_at IS NOT NULL THEN 1 END) as closed_count,
                COALESCE(SUM(CASE WHEN realized_pnl IS NOT NULL THEN CAST(realized_pnl AS REAL) ELSE 0 END), 0) as total_pnl
            FROM position_history
            "#,
        )
        .fetch_one(self.db.pool())
        .await?;

        let open: i64 = row.get("open_count");
        let closed: i64 = row.get("closed_count");
        let pnl: f64 = row.get("total_pnl");

        Ok((open, closed, Decimal::from_f64_retain(pnl).unwrap_or(Decimal::ZERO)))
    }

    /// Update position size (for partial closes)
    pub async fn update_size(&self, id: i64, new_size: Decimal) -> Result<()> {
        sqlx::query("UPDATE position_history SET size = ? WHERE id = ?")
            .bind(new_size.to_string())
            .bind(id)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    /// Update position entry price (for averaging)
    pub async fn update_entry_price(&self, id: i64, new_entry_price: Decimal, additional_fees: Decimal) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE position_history SET
                entry_price = ?,
                fees = CAST((CAST(fees AS REAL) + ?) AS TEXT)
            WHERE id = ?
            "#,
        )
        .bind(new_entry_price.to_string())
        .bind(additional_fees.to_string())
        .bind(id)
        .execute(self.db.pool())
        .await?;

        Ok(())
    }

    fn row_to_record(row: &sqlx::sqlite::SqliteRow) -> Result<PositionHistoryRecord> {
        Ok(PositionHistoryRecord {
            id: Some(row.get("id")),
            market_id: row.get("market_id"),
            token_id: row.get("token_id"),
            side: match row.get::<String, _>("side").as_str() {
                "BUY" => Side::Buy,
                _ => Side::Sell,
            },
            entry_price: Decimal::from_str(row.get::<&str, _>("entry_price"))
                .unwrap_or(Decimal::ZERO),
            exit_price: row
                .get::<Option<String>, _>("exit_price")
                .and_then(|s| Decimal::from_str(&s).ok()),
            size: Decimal::from_str(row.get::<&str, _>("size")).unwrap_or(Decimal::ZERO),
            realized_pnl: row
                .get::<Option<String>, _>("realized_pnl")
                .and_then(|s| Decimal::from_str(&s).ok()),
            fees: Decimal::from_str(row.get::<&str, _>("fees")).unwrap_or(Decimal::ZERO),
            opened_at: DateTime::parse_from_rfc3339(row.get::<&str, _>("opened_at"))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            closed_at: row
                .get::<Option<String>, _>("closed_at")
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&Utc)),
            strategy_id: row.get("strategy_id"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_history_record_default() {
        let record = PositionHistoryRecord {
            id: None,
            market_id: "test".to_string(),
            token_id: "token".to_string(),
            side: Side::Buy,
            entry_price: Decimal::from(50),
            exit_price: None,
            size: Decimal::from(100),
            realized_pnl: None,
            fees: Decimal::ZERO,
            opened_at: Utc::now(),
            closed_at: None,
            strategy_id: None,
        };

        assert!(record.closed_at.is_none());
        assert!(record.realized_pnl.is_none());
    }
}
