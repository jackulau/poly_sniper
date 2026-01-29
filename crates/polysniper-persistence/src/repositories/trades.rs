//! Trade repository

use crate::{error::Result, models::TradeRecord, Database};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use sqlx::Row;

/// Repository for trade records
pub struct TradeRepository<'a> {
    db: &'a Database,
}

impl<'a> TradeRepository<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Insert a new trade record
    pub async fn insert(&self, trade: &TradeRecord) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO trades (
                id, order_id, signal_id, strategy_id, market_id, token_id,
                side, executed_price, executed_size, size_usd, fees,
                realized_pnl, timestamp, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&trade.id)
        .bind(&trade.order_id)
        .bind(&trade.signal_id)
        .bind(&trade.strategy_id)
        .bind(&trade.market_id)
        .bind(&trade.token_id)
        .bind(trade.side.to_string())
        .bind(trade.executed_price.to_string())
        .bind(trade.executed_size.to_string())
        .bind(trade.size_usd.to_string())
        .bind(trade.fees.to_string())
        .bind(trade.realized_pnl.map(|p| p.to_string()))
        .bind(trade.timestamp.to_rfc3339())
        .bind(trade.metadata.as_ref().map(|m| m.to_string()))
        .execute(self.db.pool())
        .await?;

        Ok(())
    }

    /// Get trade by ID
    pub async fn get_by_id(&self, id: &str) -> Result<Option<TradeRecord>> {
        let row = sqlx::query("SELECT * FROM trades WHERE id = ?")
            .bind(id)
            .fetch_optional(self.db.pool())
            .await?;

        match row {
            Some(r) => Ok(Some(Self::row_to_trade(&r)?)),
            None => Ok(None),
        }
    }

    /// Get trades by market ID
    pub async fn get_by_market(&self, market_id: &str, limit: i64) -> Result<Vec<TradeRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM trades WHERE market_id = ? ORDER BY timestamp DESC LIMIT ?",
        )
        .bind(market_id)
        .bind(limit)
        .fetch_all(self.db.pool())
        .await?;

        rows.iter().map(Self::row_to_trade).collect()
    }

    /// Get trades by strategy ID
    pub async fn get_by_strategy(&self, strategy_id: &str, limit: i64) -> Result<Vec<TradeRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM trades WHERE strategy_id = ? ORDER BY timestamp DESC LIMIT ?",
        )
        .bind(strategy_id)
        .bind(limit)
        .fetch_all(self.db.pool())
        .await?;

        rows.iter().map(Self::row_to_trade).collect()
    }

    /// Get trades within time range
    pub async fn get_by_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<TradeRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM trades WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp DESC",
        )
        .bind(start.to_rfc3339())
        .bind(end.to_rfc3339())
        .fetch_all(self.db.pool())
        .await?;

        rows.iter().map(Self::row_to_trade).collect()
    }

    /// Get recent trades
    pub async fn get_recent(&self, limit: i64) -> Result<Vec<TradeRecord>> {
        let rows = sqlx::query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?")
            .bind(limit)
            .fetch_all(self.db.pool())
            .await?;

        rows.iter().map(Self::row_to_trade).collect()
    }

    /// Get total trade count
    pub async fn count(&self) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM trades")
            .fetch_one(self.db.pool())
            .await?;

        Ok(row.get::<i64, _>("count"))
    }

    /// Get total volume
    pub async fn total_volume(&self) -> Result<Decimal> {
        let row = sqlx::query("SELECT COALESCE(SUM(CAST(size_usd AS REAL)), 0) as total FROM trades")
            .fetch_one(self.db.pool())
            .await?;

        let total: f64 = row.get("total");
        Ok(Decimal::from_f64_retain(total).unwrap_or(Decimal::ZERO))
    }

    fn row_to_trade(row: &sqlx::sqlite::SqliteRow) -> Result<TradeRecord> {
        use polysniper_core::Side;
        use std::str::FromStr;

        Ok(TradeRecord {
            id: row.get("id"),
            order_id: row.get("order_id"),
            signal_id: row.get("signal_id"),
            strategy_id: row.get("strategy_id"),
            market_id: row.get("market_id"),
            token_id: row.get("token_id"),
            side: match row.get::<String, _>("side").as_str() {
                "BUY" => Side::Buy,
                _ => Side::Sell,
            },
            executed_price: Decimal::from_str(row.get::<&str, _>("executed_price"))
                .unwrap_or(Decimal::ZERO),
            executed_size: Decimal::from_str(row.get::<&str, _>("executed_size"))
                .unwrap_or(Decimal::ZERO),
            size_usd: Decimal::from_str(row.get::<&str, _>("size_usd")).unwrap_or(Decimal::ZERO),
            fees: Decimal::from_str(row.get::<&str, _>("fees")).unwrap_or(Decimal::ZERO),
            realized_pnl: row
                .get::<Option<String>, _>("realized_pnl")
                .and_then(|s| Decimal::from_str(&s).ok()),
            timestamp: DateTime::parse_from_rfc3339(row.get::<&str, _>("timestamp"))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            metadata: row
                .get::<Option<String>, _>("metadata")
                .and_then(|s| serde_json::from_str(&s).ok()),
        })
    }
}
