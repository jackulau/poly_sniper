//! Performance metrics repository
//!
//! Calculates and queries trading performance metrics from historical data.

use crate::{error::Result, Database};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::Row;
use std::str::FromStr;

/// Aggregated performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total realized P&L across all trades
    pub total_realized_pnl: Decimal,
    /// Current unrealized P&L (from daily_pnl)
    pub total_unrealized_pnl: Decimal,
    /// Win rate as a percentage (0-100)
    pub win_rate: Decimal,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: Option<Decimal>,
    /// Average winning trade size
    pub avg_win: Decimal,
    /// Average losing trade size
    pub avg_loss: Decimal,
    /// Largest winning trade
    pub largest_win: Decimal,
    /// Largest losing trade
    pub largest_loss: Decimal,
    /// Total number of trades
    pub total_trades: i64,
    /// Number of winning trades
    pub win_count: i64,
    /// Number of losing trades
    pub loss_count: i64,
    /// Total fees paid
    pub total_fees: Decimal,
    /// Return on investment percentage
    pub roi: Option<Decimal>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_realized_pnl: Decimal::ZERO,
            total_unrealized_pnl: Decimal::ZERO,
            win_rate: Decimal::ZERO,
            profit_factor: None,
            avg_win: Decimal::ZERO,
            avg_loss: Decimal::ZERO,
            largest_win: Decimal::ZERO,
            largest_loss: Decimal::ZERO,
            total_trades: 0,
            win_count: 0,
            loss_count: 0,
            total_fees: Decimal::ZERO,
            roi: None,
        }
    }
}

/// Equity curve data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub date: String,
    pub balance: Decimal,
    pub realized_pnl: Decimal,
    pub unrealized_pnl: Decimal,
}

/// Repository for performance metrics queries
pub struct PerformanceMetricsRepository<'a> {
    db: &'a Database,
}

impl<'a> PerformanceMetricsRepository<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Calculate overall performance metrics
    pub async fn calculate_metrics(&self, starting_capital: Option<Decimal>) -> Result<PerformanceMetrics> {
        // Get aggregated daily P&L data
        let daily_row = sqlx::query(
            r#"
            SELECT
                COALESCE(SUM(CAST(realized_pnl AS REAL)), 0) as total_realized,
                COALESCE((SELECT unrealized_pnl FROM daily_pnl ORDER BY date DESC LIMIT 1), '0') as current_unrealized,
                COALESCE(SUM(win_count), 0) as total_wins,
                COALESCE(SUM(loss_count), 0) as total_losses,
                COALESCE(SUM(trade_count), 0) as total_trades
            FROM daily_pnl
            "#,
        )
        .fetch_one(self.db.pool())
        .await?;

        let total_realized: f64 = daily_row.get("total_realized");
        let current_unrealized: String = daily_row.get("current_unrealized");
        let total_wins: i64 = daily_row.get("total_wins");
        let total_losses: i64 = daily_row.get("total_losses");
        let total_trades: i64 = daily_row.get("total_trades");

        // Calculate win/loss aggregates from trades
        let trade_stats = sqlx::query(
            r#"
            SELECT
                COALESCE(SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN CAST(realized_pnl AS REAL) ELSE 0 END), 0) as gross_profit,
                COALESCE(SUM(CASE WHEN CAST(realized_pnl AS REAL) < 0 THEN ABS(CAST(realized_pnl AS REAL)) ELSE 0 END), 0) as gross_loss,
                COALESCE(MAX(CAST(realized_pnl AS REAL)), 0) as largest_win,
                COALESCE(MIN(CAST(realized_pnl AS REAL)), 0) as largest_loss,
                COALESCE(SUM(CAST(fees AS REAL)), 0) as total_fees,
                COUNT(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN 1 END) as win_trades,
                COUNT(CASE WHEN CAST(realized_pnl AS REAL) < 0 THEN 1 END) as loss_trades
            FROM trades
            WHERE realized_pnl IS NOT NULL
            "#,
        )
        .fetch_one(self.db.pool())
        .await?;

        let gross_profit: f64 = trade_stats.get("gross_profit");
        let gross_loss: f64 = trade_stats.get("gross_loss");
        let largest_win: f64 = trade_stats.get("largest_win");
        let largest_loss: f64 = trade_stats.get("largest_loss");
        let total_fees: f64 = trade_stats.get("total_fees");
        let win_trades: i64 = trade_stats.get("win_trades");
        let loss_trades: i64 = trade_stats.get("loss_trades");

        // Calculate derived metrics
        let win_rate = if total_wins + total_losses > 0 {
            Decimal::from(total_wins * 100) / Decimal::from(total_wins + total_losses)
        } else {
            Decimal::ZERO
        };

        let profit_factor = if gross_loss > 0.0 {
            Some(Decimal::from_f64_retain(gross_profit / gross_loss).unwrap_or(Decimal::ZERO))
        } else {
            None // Undefined or infinite profit factor when no losses
        };

        let avg_win = if win_trades > 0 {
            Decimal::from_f64_retain(gross_profit / win_trades as f64).unwrap_or(Decimal::ZERO)
        } else {
            Decimal::ZERO
        };

        let avg_loss = if loss_trades > 0 {
            Decimal::from_f64_retain(gross_loss / loss_trades as f64).unwrap_or(Decimal::ZERO)
        } else {
            Decimal::ZERO
        };

        let roi = starting_capital.and_then(|cap| {
            if cap > Decimal::ZERO {
                Some(Decimal::from_f64_retain(total_realized).unwrap_or(Decimal::ZERO) * Decimal::from(100) / cap)
            } else {
                None
            }
        });

        Ok(PerformanceMetrics {
            total_realized_pnl: Decimal::from_f64_retain(total_realized).unwrap_or(Decimal::ZERO),
            total_unrealized_pnl: Decimal::from_str(&current_unrealized).unwrap_or(Decimal::ZERO),
            win_rate,
            profit_factor,
            avg_win,
            avg_loss,
            largest_win: Decimal::from_f64_retain(largest_win).unwrap_or(Decimal::ZERO),
            largest_loss: Decimal::from_f64_retain(largest_loss.abs()).unwrap_or(Decimal::ZERO),
            total_trades,
            win_count: total_wins,
            loss_count: total_losses,
            total_fees: Decimal::from_f64_retain(total_fees).unwrap_or(Decimal::ZERO),
            roi,
        })
    }

    /// Get P&L by date range
    pub async fn get_pnl_by_date_range(
        &self,
        from: &str,
        to: &str,
    ) -> Result<PerformanceMetrics> {
        let daily_row = sqlx::query(
            r#"
            SELECT
                COALESCE(SUM(CAST(realized_pnl AS REAL)), 0) as total_realized,
                COALESCE(SUM(win_count), 0) as total_wins,
                COALESCE(SUM(loss_count), 0) as total_losses,
                COALESCE(SUM(trade_count), 0) as total_trades
            FROM daily_pnl
            WHERE date >= ? AND date <= ?
            "#,
        )
        .bind(from)
        .bind(to)
        .fetch_one(self.db.pool())
        .await?;

        let total_realized: f64 = daily_row.get("total_realized");
        let total_wins: i64 = daily_row.get("total_wins");
        let total_losses: i64 = daily_row.get("total_losses");
        let total_trades: i64 = daily_row.get("total_trades");

        let win_rate = if total_wins + total_losses > 0 {
            Decimal::from(total_wins * 100) / Decimal::from(total_wins + total_losses)
        } else {
            Decimal::ZERO
        };

        Ok(PerformanceMetrics {
            total_realized_pnl: Decimal::from_f64_retain(total_realized).unwrap_or(Decimal::ZERO),
            win_rate,
            total_trades,
            win_count: total_wins,
            loss_count: total_losses,
            ..Default::default()
        })
    }

    /// Get P&L by strategy
    pub async fn get_pnl_by_strategy(&self, strategy_id: &str) -> Result<PerformanceMetrics> {
        let row = sqlx::query(
            r#"
            SELECT
                COALESCE(SUM(CAST(realized_pnl AS REAL)), 0) as total_realized,
                COALESCE(SUM(CAST(fees AS REAL)), 0) as total_fees,
                COUNT(*) as total_trades,
                COUNT(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN CAST(realized_pnl AS REAL) < 0 THEN 1 END) as losses,
                COALESCE(MAX(CAST(realized_pnl AS REAL)), 0) as largest_win,
                COALESCE(MIN(CAST(realized_pnl AS REAL)), 0) as largest_loss
            FROM trades
            WHERE strategy_id = ? AND realized_pnl IS NOT NULL
            "#,
        )
        .bind(strategy_id)
        .fetch_one(self.db.pool())
        .await?;

        let total_realized: f64 = row.get("total_realized");
        let total_fees: f64 = row.get("total_fees");
        let total_trades: i64 = row.get("total_trades");
        let wins: i64 = row.get("wins");
        let losses: i64 = row.get("losses");
        let largest_win: f64 = row.get("largest_win");
        let largest_loss: f64 = row.get("largest_loss");

        let win_rate = if wins + losses > 0 {
            Decimal::from(wins * 100) / Decimal::from(wins + losses)
        } else {
            Decimal::ZERO
        };

        Ok(PerformanceMetrics {
            total_realized_pnl: Decimal::from_f64_retain(total_realized).unwrap_or(Decimal::ZERO),
            win_rate,
            total_trades,
            win_count: wins,
            loss_count: losses,
            total_fees: Decimal::from_f64_retain(total_fees).unwrap_or(Decimal::ZERO),
            largest_win: Decimal::from_f64_retain(largest_win).unwrap_or(Decimal::ZERO),
            largest_loss: Decimal::from_f64_retain(largest_loss.abs()).unwrap_or(Decimal::ZERO),
            ..Default::default()
        })
    }

    /// Get P&L by market
    pub async fn get_pnl_by_market(&self, market_id: &str) -> Result<PerformanceMetrics> {
        let row = sqlx::query(
            r#"
            SELECT
                COALESCE(SUM(CAST(realized_pnl AS REAL)), 0) as total_realized,
                COALESCE(SUM(CAST(fees AS REAL)), 0) as total_fees,
                COUNT(*) as total_trades,
                COUNT(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN CAST(realized_pnl AS REAL) < 0 THEN 1 END) as losses
            FROM trades
            WHERE market_id = ? AND realized_pnl IS NOT NULL
            "#,
        )
        .bind(market_id)
        .fetch_one(self.db.pool())
        .await?;

        let total_realized: f64 = row.get("total_realized");
        let total_fees: f64 = row.get("total_fees");
        let total_trades: i64 = row.get("total_trades");
        let wins: i64 = row.get("wins");
        let losses: i64 = row.get("losses");

        let win_rate = if wins + losses > 0 {
            Decimal::from(wins * 100) / Decimal::from(wins + losses)
        } else {
            Decimal::ZERO
        };

        Ok(PerformanceMetrics {
            total_realized_pnl: Decimal::from_f64_retain(total_realized).unwrap_or(Decimal::ZERO),
            win_rate,
            total_trades,
            win_count: wins,
            loss_count: losses,
            total_fees: Decimal::from_f64_retain(total_fees).unwrap_or(Decimal::ZERO),
            ..Default::default()
        })
    }

    /// Get cumulative P&L over time
    pub async fn get_cumulative_pnl(&self) -> Result<Vec<(String, Decimal)>> {
        let rows = sqlx::query(
            r#"
            SELECT
                date,
                SUM(CAST(realized_pnl AS REAL)) OVER (ORDER BY date) as cumulative_pnl
            FROM daily_pnl
            ORDER BY date
            "#,
        )
        .fetch_all(self.db.pool())
        .await?;

        let mut result = Vec::new();
        for row in rows {
            let date: String = row.get("date");
            let pnl: f64 = row.get("cumulative_pnl");
            result.push((date, Decimal::from_f64_retain(pnl).unwrap_or(Decimal::ZERO)));
        }

        Ok(result)
    }

    /// Get equity curve (daily ending balances)
    pub async fn get_equity_curve(&self) -> Result<Vec<EquityPoint>> {
        let rows = sqlx::query(
            r#"
            SELECT
                date,
                COALESCE(ending_balance, starting_balance) as balance,
                realized_pnl,
                unrealized_pnl
            FROM daily_pnl
            ORDER BY date
            "#,
        )
        .fetch_all(self.db.pool())
        .await?;

        let mut result = Vec::new();
        for row in rows {
            let date: String = row.get("date");
            let balance: String = row.get("balance");
            let realized: String = row.get("realized_pnl");
            let unrealized: String = row.get("unrealized_pnl");

            result.push(EquityPoint {
                date,
                balance: Decimal::from_str(&balance).unwrap_or(Decimal::ZERO),
                realized_pnl: Decimal::from_str(&realized).unwrap_or(Decimal::ZERO),
                unrealized_pnl: Decimal::from_str(&unrealized).unwrap_or(Decimal::ZERO),
            });
        }

        Ok(result)
    }

    /// Get P&L summary for today
    pub async fn get_today_pnl(&self) -> Result<Option<(Decimal, Decimal)>> {
        let today = Utc::now().format("%Y-%m-%d").to_string();

        let row = sqlx::query(
            "SELECT realized_pnl, unrealized_pnl FROM daily_pnl WHERE date = ?"
        )
        .bind(&today)
        .fetch_optional(self.db.pool())
        .await?;

        match row {
            Some(r) => {
                let realized: String = r.get("realized_pnl");
                let unrealized: String = r.get("unrealized_pnl");
                Ok(Some((
                    Decimal::from_str(&realized).unwrap_or(Decimal::ZERO),
                    Decimal::from_str(&unrealized).unwrap_or(Decimal::ZERO),
                )))
            }
            None => Ok(None),
        }
    }

    /// Get trades with realized P&L within a time range
    pub async fn get_trades_with_pnl(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<(String, String, Decimal, Decimal)>> {
        let rows = sqlx::query(
            r#"
            SELECT
                id,
                strategy_id,
                COALESCE(realized_pnl, '0') as realized_pnl,
                fees
            FROM trades
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
            "#,
        )
        .bind(from.to_rfc3339())
        .bind(to.to_rfc3339())
        .fetch_all(self.db.pool())
        .await?;

        let mut result = Vec::new();
        for row in rows {
            let id: String = row.get("id");
            let strategy: String = row.get("strategy_id");
            let pnl: String = row.get("realized_pnl");
            let fees: String = row.get("fees");

            result.push((
                id,
                strategy,
                Decimal::from_str(&pnl).unwrap_or(Decimal::ZERO),
                Decimal::from_str(&fees).unwrap_or(Decimal::ZERO),
            ));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_default() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.total_realized_pnl, Decimal::ZERO);
        assert_eq!(metrics.win_rate, Decimal::ZERO);
        assert_eq!(metrics.total_trades, 0);
    }
}
