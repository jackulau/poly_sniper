//! Strategy metrics repository for aggregated performance data

use crate::{error::Result, Database};
use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::Row;
use std::str::FromStr;

/// Aggregated strategy performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    /// Strategy identifier
    pub strategy_id: String,
    /// Total number of trades
    pub total_trades: i64,
    /// Number of winning trades (realized_pnl > 0)
    pub win_count: i64,
    /// Number of losing trades (realized_pnl < 0)
    pub loss_count: i64,
    /// Sum of all positive P&L
    pub gross_profit: Decimal,
    /// Sum of all negative P&L (stored as negative value)
    pub gross_loss: Decimal,
    /// Net P&L (gross_profit + gross_loss)
    pub net_pnl: Decimal,
    /// Win rate (win_count / total_trades with realized_pnl)
    pub win_rate: f64,
    /// Profit factor (|gross_profit / gross_loss|)
    pub profit_factor: Option<f64>,
    /// Average trade size in USD
    pub avg_trade_size: Decimal,
    /// Average winning trade P&L
    pub avg_win: Option<Decimal>,
    /// Average losing trade P&L
    pub avg_loss: Option<Decimal>,
    /// Maximum consecutive wins
    pub max_consecutive_wins: i64,
    /// Maximum consecutive losses
    pub max_consecutive_losses: i64,
    /// Largest single win
    pub largest_win: Option<Decimal>,
    /// Largest single loss
    pub largest_loss: Option<Decimal>,
    /// Sharpe ratio (if enough data)
    pub sharpe_ratio: Option<f64>,
    /// Total volume traded in USD
    pub total_volume: Decimal,
}

impl Default for StrategyMetrics {
    fn default() -> Self {
        Self {
            strategy_id: String::new(),
            total_trades: 0,
            win_count: 0,
            loss_count: 0,
            gross_profit: Decimal::ZERO,
            gross_loss: Decimal::ZERO,
            net_pnl: Decimal::ZERO,
            win_rate: 0.0,
            profit_factor: None,
            avg_trade_size: Decimal::ZERO,
            avg_win: None,
            avg_loss: None,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            largest_win: None,
            largest_loss: None,
            sharpe_ratio: None,
            total_volume: Decimal::ZERO,
        }
    }
}

/// Time period filter for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimePeriod {
    Today,
    Week,
    Month,
    All,
    Custom,
}

impl FromStr for TimePeriod {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "today" => Ok(Self::Today),
            "week" => Ok(Self::Week),
            "month" => Ok(Self::Month),
            "all" => Ok(Self::All),
            _ => Err(format!("Invalid time period: {}", s)),
        }
    }
}

/// Metric to rank strategies by
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankingMetric {
    NetPnl,
    WinRate,
    ProfitFactor,
    SharpeRatio,
    TotalTrades,
    TotalVolume,
}

impl FromStr for RankingMetric {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "net_pnl" | "netpnl" | "pnl" => Ok(Self::NetPnl),
            "win_rate" | "winrate" => Ok(Self::WinRate),
            "profit_factor" | "profitfactor" => Ok(Self::ProfitFactor),
            "sharpe" | "sharpe_ratio" => Ok(Self::SharpeRatio),
            "trades" | "total_trades" => Ok(Self::TotalTrades),
            "volume" | "total_volume" => Ok(Self::TotalVolume),
            _ => Err(format!("Invalid ranking metric: {}", s)),
        }
    }
}

/// Repository for strategy performance metrics
pub struct StrategyMetricsRepository<'a> {
    db: &'a Database,
}

impl<'a> StrategyMetricsRepository<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Get aggregated metrics for all strategies
    pub async fn get_all_metrics(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<Vec<StrategyMetrics>> {
        let strategy_ids = self.get_distinct_strategies(start, end).await?;

        let mut metrics = Vec::new();
        for strategy_id in strategy_ids {
            if let Some(m) = self.get_metrics(&strategy_id, start, end).await? {
                metrics.push(m);
            }
        }

        Ok(metrics)
    }

    /// Get metrics for a single strategy
    pub async fn get_metrics(
        &self,
        strategy_id: &str,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<Option<StrategyMetrics>> {
        // Build the time filter clause
        let (time_clause, start_str, end_str) = self.build_time_filter(start, end);

        // Get basic aggregates
        let query = format!(
            r#"
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl IS NOT NULL AND CAST(realized_pnl AS REAL) > 0 THEN 1 ELSE 0 END) as win_count,
                SUM(CASE WHEN realized_pnl IS NOT NULL AND CAST(realized_pnl AS REAL) < 0 THEN 1 ELSE 0 END) as loss_count,
                COALESCE(SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN CAST(realized_pnl AS REAL) ELSE 0 END), 0) as gross_profit,
                COALESCE(SUM(CASE WHEN CAST(realized_pnl AS REAL) < 0 THEN CAST(realized_pnl AS REAL) ELSE 0 END), 0) as gross_loss,
                COALESCE(SUM(CAST(realized_pnl AS REAL)), 0) as net_pnl,
                COALESCE(AVG(CAST(size_usd AS REAL)), 0) as avg_trade_size,
                COALESCE(SUM(CAST(size_usd AS REAL)), 0) as total_volume,
                MAX(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN CAST(realized_pnl AS REAL) END) as largest_win,
                MIN(CASE WHEN CAST(realized_pnl AS REAL) < 0 THEN CAST(realized_pnl AS REAL) END) as largest_loss,
                AVG(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN CAST(realized_pnl AS REAL) END) as avg_win,
                AVG(CASE WHEN CAST(realized_pnl AS REAL) < 0 THEN CAST(realized_pnl AS REAL) END) as avg_loss
            FROM trades
            WHERE strategy_id = ?
            {}
            "#,
            time_clause
        );

        let mut query_builder = sqlx::query(&query).bind(strategy_id);

        if let Some(ref s) = start_str {
            query_builder = query_builder.bind(s);
        }
        if let Some(ref e) = end_str {
            query_builder = query_builder.bind(e);
        }

        let row = query_builder.fetch_optional(self.db.pool()).await?;

        let Some(row) = row else {
            return Ok(None);
        };

        let total_trades: i64 = row.try_get("total_trades").unwrap_or(0);

        if total_trades == 0 {
            return Ok(Some(StrategyMetrics {
                strategy_id: strategy_id.to_string(),
                ..Default::default()
            }));
        }

        let win_count: i64 = row.try_get("win_count").unwrap_or(0);
        let loss_count: i64 = row.try_get("loss_count").unwrap_or(0);
        let gross_profit: f64 = row.try_get("gross_profit").unwrap_or(0.0);
        let gross_loss: f64 = row.try_get("gross_loss").unwrap_or(0.0);
        let net_pnl: f64 = row.try_get("net_pnl").unwrap_or(0.0);
        let avg_trade_size: f64 = row.try_get("avg_trade_size").unwrap_or(0.0);
        let total_volume: f64 = row.try_get("total_volume").unwrap_or(0.0);
        let largest_win: Option<f64> = row.try_get("largest_win").ok();
        let largest_loss: Option<f64> = row.try_get("largest_loss").ok();
        let avg_win: Option<f64> = row.try_get("avg_win").ok();
        let avg_loss: Option<f64> = row.try_get("avg_loss").ok();

        // Calculate win rate
        let trades_with_pnl = win_count + loss_count;
        let win_rate = if trades_with_pnl > 0 {
            (win_count as f64) / (trades_with_pnl as f64) * 100.0
        } else {
            0.0
        };

        // Calculate profit factor
        let profit_factor = if gross_loss.abs() > 0.0 {
            Some(gross_profit / gross_loss.abs())
        } else if gross_profit > 0.0 {
            Some(f64::INFINITY)
        } else {
            None
        };

        // Get consecutive wins/losses
        let (max_consecutive_wins, max_consecutive_losses) =
            self.calculate_consecutive_streaks(strategy_id, start, end).await?;

        // Calculate Sharpe ratio
        let sharpe_ratio = self.calculate_sharpe_ratio(strategy_id, start, end).await?;

        Ok(Some(StrategyMetrics {
            strategy_id: strategy_id.to_string(),
            total_trades,
            win_count,
            loss_count,
            gross_profit: Decimal::from_f64_retain(gross_profit).unwrap_or(Decimal::ZERO),
            gross_loss: Decimal::from_f64_retain(gross_loss).unwrap_or(Decimal::ZERO),
            net_pnl: Decimal::from_f64_retain(net_pnl).unwrap_or(Decimal::ZERO),
            win_rate,
            profit_factor,
            avg_trade_size: Decimal::from_f64_retain(avg_trade_size).unwrap_or(Decimal::ZERO),
            avg_win: avg_win.and_then(Decimal::from_f64_retain),
            avg_loss: avg_loss.and_then(Decimal::from_f64_retain),
            max_consecutive_wins,
            max_consecutive_losses,
            largest_win: largest_win.and_then(Decimal::from_f64_retain),
            largest_loss: largest_loss.and_then(Decimal::from_f64_retain),
            sharpe_ratio,
            total_volume: Decimal::from_f64_retain(total_volume).unwrap_or(Decimal::ZERO),
        }))
    }

    /// Get distinct strategy IDs from trades
    async fn get_distinct_strategies(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<Vec<String>> {
        let (time_clause, start_str, end_str) = self.build_time_filter(start, end);

        let query = format!(
            "SELECT DISTINCT strategy_id FROM trades WHERE 1=1 {}",
            time_clause
        );

        let mut query_builder = sqlx::query(&query);

        if let Some(ref s) = start_str {
            query_builder = query_builder.bind(s);
        }
        if let Some(ref e) = end_str {
            query_builder = query_builder.bind(e);
        }

        let rows = query_builder.fetch_all(self.db.pool()).await?;

        Ok(rows
            .iter()
            .map(|r| r.get::<String, _>("strategy_id"))
            .collect())
    }

    /// Calculate maximum consecutive wins and losses
    async fn calculate_consecutive_streaks(
        &self,
        strategy_id: &str,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<(i64, i64)> {
        let (time_clause, start_str, end_str) = self.build_time_filter(start, end);

        let query = format!(
            r#"
            SELECT realized_pnl
            FROM trades
            WHERE strategy_id = ? AND realized_pnl IS NOT NULL
            {}
            ORDER BY timestamp ASC
            "#,
            time_clause
        );

        let mut query_builder = sqlx::query(&query).bind(strategy_id);

        if let Some(ref s) = start_str {
            query_builder = query_builder.bind(s);
        }
        if let Some(ref e) = end_str {
            query_builder = query_builder.bind(e);
        }

        let rows = query_builder.fetch_all(self.db.pool()).await?;

        let mut max_wins = 0i64;
        let mut max_losses = 0i64;
        let mut current_wins = 0i64;
        let mut current_losses = 0i64;

        for row in rows {
            let pnl_str: String = row.get("realized_pnl");
            let pnl = pnl_str.parse::<f64>().unwrap_or(0.0);

            if pnl > 0.0 {
                current_wins += 1;
                current_losses = 0;
                max_wins = max_wins.max(current_wins);
            } else if pnl < 0.0 {
                current_losses += 1;
                current_wins = 0;
                max_losses = max_losses.max(current_losses);
            }
        }

        Ok((max_wins, max_losses))
    }

    /// Calculate Sharpe ratio for a strategy
    /// Sharpe = (mean return - risk_free_rate) / std_dev(returns)
    /// Using 0 as risk-free rate for simplicity
    async fn calculate_sharpe_ratio(
        &self,
        strategy_id: &str,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<Option<f64>> {
        let (time_clause, start_str, end_str) = self.build_time_filter(start, end);

        let query = format!(
            r#"
            SELECT realized_pnl
            FROM trades
            WHERE strategy_id = ? AND realized_pnl IS NOT NULL
            {}
            "#,
            time_clause
        );

        let mut query_builder = sqlx::query(&query).bind(strategy_id);

        if let Some(ref s) = start_str {
            query_builder = query_builder.bind(s);
        }
        if let Some(ref e) = end_str {
            query_builder = query_builder.bind(e);
        }

        let rows = query_builder.fetch_all(self.db.pool()).await?;

        if rows.len() < 2 {
            return Ok(None);
        }

        let returns: Vec<f64> = rows
            .iter()
            .filter_map(|r| {
                let pnl_str: String = r.get("realized_pnl");
                pnl_str.parse::<f64>().ok()
            })
            .collect();

        if returns.len() < 2 {
            return Ok(None);
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok(None);
        }

        // Annualized Sharpe (assuming daily trades, ~252 trading days)
        // For simplicity, we just compute the ratio without annualization
        Ok(Some(mean / std_dev))
    }

    /// Get aggregated metrics across all strategies (totals)
    pub async fn get_total_metrics(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<StrategyMetrics> {
        let all_metrics = self.get_all_metrics(start, end).await?;

        if all_metrics.is_empty() {
            return Ok(StrategyMetrics {
                strategy_id: "TOTAL".to_string(),
                ..Default::default()
            });
        }

        let total_trades: i64 = all_metrics.iter().map(|m| m.total_trades).sum();
        let win_count: i64 = all_metrics.iter().map(|m| m.win_count).sum();
        let loss_count: i64 = all_metrics.iter().map(|m| m.loss_count).sum();
        let gross_profit: Decimal = all_metrics.iter().map(|m| m.gross_profit).sum();
        let gross_loss: Decimal = all_metrics.iter().map(|m| m.gross_loss).sum();
        let net_pnl: Decimal = all_metrics.iter().map(|m| m.net_pnl).sum();
        let total_volume: Decimal = all_metrics.iter().map(|m| m.total_volume).sum();

        let trades_with_pnl = win_count + loss_count;
        let win_rate = if trades_with_pnl > 0 {
            (win_count as f64) / (trades_with_pnl as f64) * 100.0
        } else {
            0.0
        };

        let gross_loss_f64 = gross_loss.to_string().parse::<f64>().unwrap_or(0.0);
        let gross_profit_f64 = gross_profit.to_string().parse::<f64>().unwrap_or(0.0);

        let profit_factor = if gross_loss_f64.abs() > 0.0 {
            Some(gross_profit_f64 / gross_loss_f64.abs())
        } else if gross_profit_f64 > 0.0 {
            Some(f64::INFINITY)
        } else {
            None
        };

        let avg_trade_size = if total_trades > 0 {
            total_volume / Decimal::from(total_trades)
        } else {
            Decimal::ZERO
        };

        // Calculate overall Sharpe from all trades
        let sharpe_ratio = self.calculate_sharpe_ratio_all(start, end).await?;

        Ok(StrategyMetrics {
            strategy_id: "TOTAL".to_string(),
            total_trades,
            win_count,
            loss_count,
            gross_profit,
            gross_loss,
            net_pnl,
            win_rate,
            profit_factor,
            avg_trade_size,
            avg_win: None,
            avg_loss: None,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            largest_win: None,
            largest_loss: None,
            sharpe_ratio,
            total_volume,
        })
    }

    /// Calculate Sharpe ratio across all strategies
    async fn calculate_sharpe_ratio_all(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<Option<f64>> {
        let (time_clause, start_str, end_str) = self.build_time_filter(start, end);

        let query = format!(
            r#"
            SELECT realized_pnl
            FROM trades
            WHERE realized_pnl IS NOT NULL
            {}
            "#,
            time_clause
        );

        let mut query_builder = sqlx::query(&query);

        if let Some(ref s) = start_str {
            query_builder = query_builder.bind(s);
        }
        if let Some(ref e) = end_str {
            query_builder = query_builder.bind(e);
        }

        let rows = query_builder.fetch_all(self.db.pool()).await?;

        if rows.len() < 2 {
            return Ok(None);
        }

        let returns: Vec<f64> = rows
            .iter()
            .filter_map(|r| {
                let pnl_str: String = r.get("realized_pnl");
                pnl_str.parse::<f64>().ok()
            })
            .collect();

        if returns.len() < 2 {
            return Ok(None);
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok(None);
        }

        Ok(Some(mean / std_dev))
    }

    /// Sort metrics by the specified ranking metric
    pub fn sort_by_metric(metrics: &mut [StrategyMetrics], rank_by: RankingMetric, descending: bool) {
        metrics.sort_by(|a, b| {
            let cmp = match rank_by {
                RankingMetric::NetPnl => a.net_pnl.cmp(&b.net_pnl),
                RankingMetric::WinRate => a.win_rate.partial_cmp(&b.win_rate).unwrap_or(std::cmp::Ordering::Equal),
                RankingMetric::ProfitFactor => {
                    let a_pf = a.profit_factor.unwrap_or(0.0);
                    let b_pf = b.profit_factor.unwrap_or(0.0);
                    a_pf.partial_cmp(&b_pf).unwrap_or(std::cmp::Ordering::Equal)
                }
                RankingMetric::SharpeRatio => {
                    let a_sr = a.sharpe_ratio.unwrap_or(f64::NEG_INFINITY);
                    let b_sr = b.sharpe_ratio.unwrap_or(f64::NEG_INFINITY);
                    a_sr.partial_cmp(&b_sr).unwrap_or(std::cmp::Ordering::Equal)
                }
                RankingMetric::TotalTrades => a.total_trades.cmp(&b.total_trades),
                RankingMetric::TotalVolume => a.total_volume.cmp(&b.total_volume),
            };

            if descending {
                cmp.reverse()
            } else {
                cmp
            }
        });
    }

    /// Get time bounds based on period
    pub fn get_time_bounds(period: TimePeriod) -> (Option<DateTime<Utc>>, Option<DateTime<Utc>>) {
        use chrono::{Duration, TimeZone};

        let now = Utc::now();
        let today_start = Utc
            .from_utc_datetime(&now.date_naive().and_hms_opt(0, 0, 0).unwrap());

        match period {
            TimePeriod::Today => (Some(today_start), Some(now)),
            TimePeriod::Week => (Some(now - Duration::days(7)), Some(now)),
            TimePeriod::Month => (Some(now - Duration::days(30)), Some(now)),
            TimePeriod::All => (None, None),
            TimePeriod::Custom => (None, None), // Caller provides bounds
        }
    }

    /// Build time filter SQL clause
    fn build_time_filter(
        &self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> (String, Option<String>, Option<String>) {
        let mut clause = String::new();
        let start_str = start.map(|s| s.to_rfc3339());
        let end_str = end.map(|e| e.to_rfc3339());

        if start_str.is_some() {
            clause.push_str(" AND timestamp >= ?");
        }
        if end_str.is_some() {
            clause.push_str(" AND timestamp <= ?");
        }

        (clause, start_str, end_str)
    }

    /// Get metrics for a specific date range
    pub async fn get_metrics_for_date_range(
        &self,
        strategy_id: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<Option<StrategyMetrics>> {
        use chrono::TimeZone;

        let start = Utc
            .from_utc_datetime(&start_date.and_hms_opt(0, 0, 0).unwrap());
        let end = Utc
            .from_utc_datetime(&end_date.and_hms_opt(23, 59, 59).unwrap());

        self.get_metrics(strategy_id, Some(start), Some(end)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Database, TradeRecord, TradeRepository};
    use chrono::Utc;
    use polysniper_core::Side;
    use rust_decimal_macros::dec;

    async fn setup_test_db() -> Database {
        Database::in_memory().await.unwrap()
    }

    async fn insert_test_trade(
        db: &Database,
        strategy_id: &str,
        realized_pnl: Option<Decimal>,
        size_usd: Decimal,
    ) {
        let trade = TradeRecord {
            id: uuid::Uuid::new_v4().to_string(),
            order_id: "test_order".to_string(),
            signal_id: "test_signal".to_string(),
            strategy_id: strategy_id.to_string(),
            market_id: "test_market".to_string(),
            token_id: "test_token".to_string(),
            side: Side::Buy,
            executed_price: dec!(0.50),
            executed_size: dec!(100),
            size_usd,
            fees: dec!(0.01),
            realized_pnl,
            timestamp: Utc::now(),
            metadata: None,
        };
        let repo = TradeRepository::new(db);
        repo.insert(&trade).await.unwrap();
    }

    #[tokio::test]
    async fn test_empty_metrics() {
        let db = setup_test_db().await;
        let repo = StrategyMetricsRepository::new(&db);

        let metrics = repo.get_metrics("nonexistent", None, None).await.unwrap();
        assert!(metrics.is_none() || metrics.unwrap().total_trades == 0);
    }

    #[tokio::test]
    async fn test_basic_metrics() {
        let db = setup_test_db().await;

        // Insert test trades
        insert_test_trade(&db, "test_strategy", Some(dec!(10)), dec!(100)).await;
        insert_test_trade(&db, "test_strategy", Some(dec!(-5)), dec!(50)).await;
        insert_test_trade(&db, "test_strategy", Some(dec!(15)), dec!(150)).await;

        let repo = StrategyMetricsRepository::new(&db);
        let metrics = repo
            .get_metrics("test_strategy", None, None)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(metrics.total_trades, 3);
        assert_eq!(metrics.win_count, 2);
        assert_eq!(metrics.loss_count, 1);
        assert_eq!(metrics.net_pnl, dec!(20)); // 10 - 5 + 15
        assert_eq!(metrics.gross_profit, dec!(25)); // 10 + 15
    }

    #[tokio::test]
    async fn test_win_rate_calculation() {
        let db = setup_test_db().await;

        // 3 wins, 1 loss = 75% win rate
        insert_test_trade(&db, "test_strategy", Some(dec!(10)), dec!(100)).await;
        insert_test_trade(&db, "test_strategy", Some(dec!(5)), dec!(100)).await;
        insert_test_trade(&db, "test_strategy", Some(dec!(15)), dec!(100)).await;
        insert_test_trade(&db, "test_strategy", Some(dec!(-5)), dec!(100)).await;

        let repo = StrategyMetricsRepository::new(&db);
        let metrics = repo
            .get_metrics("test_strategy", None, None)
            .await
            .unwrap()
            .unwrap();

        assert!((metrics.win_rate - 75.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_profit_factor() {
        let db = setup_test_db().await;

        // Gross profit = 30, Gross loss = -10, Profit factor = 3.0
        insert_test_trade(&db, "test_strategy", Some(dec!(20)), dec!(100)).await;
        insert_test_trade(&db, "test_strategy", Some(dec!(10)), dec!(100)).await;
        insert_test_trade(&db, "test_strategy", Some(dec!(-10)), dec!(100)).await;

        let repo = StrategyMetricsRepository::new(&db);
        let metrics = repo
            .get_metrics("test_strategy", None, None)
            .await
            .unwrap()
            .unwrap();

        let pf = metrics.profit_factor.unwrap();
        assert!((pf - 3.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_sorting_by_metric() {
        let mut metrics = vec![
            StrategyMetrics {
                strategy_id: "a".to_string(),
                net_pnl: dec!(100),
                win_rate: 50.0,
                ..Default::default()
            },
            StrategyMetrics {
                strategy_id: "b".to_string(),
                net_pnl: dec!(200),
                win_rate: 60.0,
                ..Default::default()
            },
            StrategyMetrics {
                strategy_id: "c".to_string(),
                net_pnl: dec!(50),
                win_rate: 80.0,
                ..Default::default()
            },
        ];

        // Sort by net PnL descending
        StrategyMetricsRepository::sort_by_metric(&mut metrics, RankingMetric::NetPnl, true);
        assert_eq!(metrics[0].strategy_id, "b");
        assert_eq!(metrics[1].strategy_id, "a");
        assert_eq!(metrics[2].strategy_id, "c");

        // Sort by win rate descending
        StrategyMetricsRepository::sort_by_metric(&mut metrics, RankingMetric::WinRate, true);
        assert_eq!(metrics[0].strategy_id, "c");
        assert_eq!(metrics[1].strategy_id, "b");
        assert_eq!(metrics[2].strategy_id, "a");
    }
}
