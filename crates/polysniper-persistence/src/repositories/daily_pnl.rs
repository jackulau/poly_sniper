//! Daily P&L repository

use crate::{error::Result, models::DailyPnlRecord, Database};
use chrono::Utc;
use rust_decimal::Decimal;
use sqlx::Row;
use std::str::FromStr;

/// Repository for daily P&L records
pub struct DailyPnlRepository<'a> {
    db: &'a Database,
}

impl<'a> DailyPnlRepository<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Get or create today's P&L record
    pub async fn get_or_create_today(&self, starting_balance: Decimal) -> Result<DailyPnlRecord> {
        let today = Utc::now().format("%Y-%m-%d").to_string();

        // Try to get existing record
        if let Some(record) = self.get_by_date(&today).await? {
            return Ok(record);
        }

        // Create new record for today
        let record = DailyPnlRecord {
            date: today.clone(),
            starting_balance,
            ending_balance: None,
            realized_pnl: Decimal::ZERO,
            unrealized_pnl: Decimal::ZERO,
            trade_count: 0,
            win_count: 0,
            loss_count: 0,
            circuit_breaker_hit: false,
        };

        sqlx::query(
            r#"
            INSERT INTO daily_pnl (
                date, starting_balance, realized_pnl, unrealized_pnl,
                trade_count, win_count, loss_count, circuit_breaker_hit
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&record.date)
        .bind(record.starting_balance.to_string())
        .bind(record.realized_pnl.to_string())
        .bind(record.unrealized_pnl.to_string())
        .bind(record.trade_count)
        .bind(record.win_count)
        .bind(record.loss_count)
        .bind(record.circuit_breaker_hit as i32)
        .execute(self.db.pool())
        .await?;

        Ok(record)
    }

    /// Get P&L record by date
    pub async fn get_by_date(&self, date: &str) -> Result<Option<DailyPnlRecord>> {
        let row = sqlx::query("SELECT * FROM daily_pnl WHERE date = ?")
            .bind(date)
            .fetch_optional(self.db.pool())
            .await?;

        match row {
            Some(r) => Ok(Some(Self::row_to_record(&r)?)),
            None => Ok(None),
        }
    }

    /// Update realized P&L
    pub async fn add_realized_pnl(&self, date: &str, pnl: Decimal, is_win: bool) -> Result<()> {
        let (win_inc, loss_inc) = if is_win { (1, 0) } else { (0, 1) };

        sqlx::query(
            r#"
            UPDATE daily_pnl SET
                realized_pnl = CAST((CAST(realized_pnl AS REAL) + ?) AS TEXT),
                trade_count = trade_count + 1,
                win_count = win_count + ?,
                loss_count = loss_count + ?
            WHERE date = ?
            "#,
        )
        .bind(pnl.to_string())
        .bind(win_inc)
        .bind(loss_inc)
        .bind(date)
        .execute(self.db.pool())
        .await?;

        Ok(())
    }

    /// Update unrealized P&L
    pub async fn update_unrealized_pnl(&self, date: &str, pnl: Decimal) -> Result<()> {
        sqlx::query("UPDATE daily_pnl SET unrealized_pnl = ? WHERE date = ?")
            .bind(pnl.to_string())
            .bind(date)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    /// Mark circuit breaker hit
    pub async fn mark_circuit_breaker(&self, date: &str) -> Result<()> {
        sqlx::query("UPDATE daily_pnl SET circuit_breaker_hit = 1 WHERE date = ?")
            .bind(date)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    /// Close the day with ending balance
    pub async fn close_day(&self, date: &str, ending_balance: Decimal) -> Result<()> {
        sqlx::query("UPDATE daily_pnl SET ending_balance = ? WHERE date = ?")
            .bind(ending_balance.to_string())
            .bind(date)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    /// Get recent P&L records
    pub async fn get_recent(&self, days: i64) -> Result<Vec<DailyPnlRecord>> {
        let rows = sqlx::query("SELECT * FROM daily_pnl ORDER BY date DESC LIMIT ?")
            .bind(days)
            .fetch_all(self.db.pool())
            .await?;

        rows.iter().map(Self::row_to_record).collect()
    }

    /// Get total realized P&L across all days
    pub async fn total_realized_pnl(&self) -> Result<Decimal> {
        let row = sqlx::query(
            "SELECT COALESCE(SUM(CAST(realized_pnl AS REAL)), 0) as total FROM daily_pnl",
        )
        .fetch_one(self.db.pool())
        .await?;

        let total: f64 = row.get("total");
        Ok(Decimal::from_f64_retain(total).unwrap_or(Decimal::ZERO))
    }

    fn row_to_record(row: &sqlx::sqlite::SqliteRow) -> Result<DailyPnlRecord> {
        Ok(DailyPnlRecord {
            date: row.get("date"),
            starting_balance: Decimal::from_str(row.get::<&str, _>("starting_balance"))
                .unwrap_or(Decimal::ZERO),
            ending_balance: row
                .get::<Option<String>, _>("ending_balance")
                .and_then(|s| Decimal::from_str(&s).ok()),
            realized_pnl: Decimal::from_str(row.get::<&str, _>("realized_pnl"))
                .unwrap_or(Decimal::ZERO),
            unrealized_pnl: Decimal::from_str(row.get::<&str, _>("unrealized_pnl"))
                .unwrap_or(Decimal::ZERO),
            trade_count: row.get("trade_count"),
            win_count: row.get("win_count"),
            loss_count: row.get("loss_count"),
            circuit_breaker_hit: row.get::<i32, _>("circuit_breaker_hit") == 1,
        })
    }
}
