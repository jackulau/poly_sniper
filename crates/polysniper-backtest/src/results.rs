//! Backtest results and performance metrics

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Results of a backtest run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    /// Strategy ID
    pub strategy_id: String,
    /// Start time of backtest
    pub start_time: DateTime<Utc>,
    /// End time of backtest
    pub end_time: DateTime<Utc>,
    /// Initial capital
    pub initial_capital: Decimal,
    /// Final capital
    pub final_capital: Decimal,
    /// Individual trade results
    pub trades: Vec<TradeResult>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Per-market breakdown
    pub market_breakdown: HashMap<String, MarketPerformance>,
}

impl BacktestResults {
    /// Create new results from trades
    pub fn from_trades(
        strategy_id: String,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        initial_capital: Decimal,
        trades: Vec<TradeResult>,
    ) -> Self {
        let final_capital = Self::calculate_final_capital(initial_capital, &trades);
        let metrics = PerformanceMetrics::calculate(&trades, initial_capital, final_capital);
        let market_breakdown = Self::calculate_market_breakdown(&trades);

        Self {
            strategy_id,
            start_time,
            end_time,
            initial_capital,
            final_capital,
            trades,
            metrics,
            market_breakdown,
        }
    }

    fn calculate_final_capital(initial: Decimal, trades: &[TradeResult]) -> Decimal {
        let total_pnl: Decimal = trades.iter().map(|t| t.pnl).sum();
        initial + total_pnl
    }

    fn calculate_market_breakdown(trades: &[TradeResult]) -> HashMap<String, MarketPerformance> {
        let mut breakdown: HashMap<String, Vec<&TradeResult>> = HashMap::new();

        for trade in trades {
            breakdown
                .entry(trade.market_id.clone())
                .or_default()
                .push(trade);
        }

        breakdown
            .into_iter()
            .map(|(market_id, market_trades)| {
                let trade_count = market_trades.len();
                let total_pnl: Decimal = market_trades.iter().map(|t| t.pnl).sum();
                let wins = market_trades.iter().filter(|t| t.pnl > Decimal::ZERO).count();

                (
                    market_id,
                    MarketPerformance {
                        trade_count,
                        total_pnl,
                        win_rate: if trade_count > 0 {
                            Decimal::from(wins) / Decimal::from(trade_count)
                        } else {
                            Decimal::ZERO
                        },
                    },
                )
            })
            .collect()
    }

    /// Export results to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export results to CSV string
    pub fn trades_to_csv(&self) -> String {
        let mut csv = String::from("timestamp,market_id,token_id,side,entry_price,exit_price,size,pnl,fees\n");
        for trade in &self.trades {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{}\n",
                trade.timestamp.to_rfc3339(),
                trade.market_id,
                trade.token_id,
                if trade.is_buy { "BUY" } else { "SELL" },
                trade.entry_price,
                trade.exit_price.unwrap_or(trade.entry_price),
                trade.size,
                trade.pnl,
                trade.fees,
            ));
        }
        csv
    }
}

/// Individual trade result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeResult {
    /// Trade ID
    pub id: String,
    /// Signal ID that generated this trade
    pub signal_id: String,
    /// Market ID
    pub market_id: String,
    /// Token ID
    pub token_id: String,
    /// Whether this was a buy
    pub is_buy: bool,
    /// Entry price
    pub entry_price: Decimal,
    /// Exit price (if closed)
    pub exit_price: Option<Decimal>,
    /// Trade size in contracts
    pub size: Decimal,
    /// Trade size in USD
    pub size_usd: Decimal,
    /// Fees paid
    pub fees: Decimal,
    /// Profit/Loss
    pub pnl: Decimal,
    /// Timestamp of the trade
    pub timestamp: DateTime<Utc>,
    /// Strategy reason
    pub reason: String,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total profit/loss
    pub total_pnl: Decimal,
    /// Total realized P&L
    pub realized_pnl: Decimal,
    /// Total unrealized P&L (open positions)
    pub unrealized_pnl: Decimal,
    /// Return on capital percentage
    pub return_pct: Decimal,
    /// Win rate (percentage of profitable trades)
    pub win_rate: Decimal,
    /// Total number of trades
    pub trade_count: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Average trade P&L
    pub avg_trade_pnl: Decimal,
    /// Average winning trade
    pub avg_win: Decimal,
    /// Average losing trade
    pub avg_loss: Decimal,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: Decimal,
    /// Maximum drawdown percentage
    pub max_drawdown_pct: Decimal,
    /// Maximum drawdown in USD
    pub max_drawdown_usd: Decimal,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: Decimal,
    /// Total fees paid
    pub total_fees: Decimal,
    /// Average trade size in USD
    pub avg_trade_size_usd: Decimal,
}

impl PerformanceMetrics {
    /// Calculate metrics from trades
    pub fn calculate(
        trades: &[TradeResult],
        initial_capital: Decimal,
        final_capital: Decimal,
    ) -> Self {
        let trade_count = trades.len();
        let total_pnl = final_capital - initial_capital;
        let return_pct = if initial_capital.is_zero() {
            Decimal::ZERO
        } else {
            (total_pnl / initial_capital) * dec!(100)
        };

        let winning_trades: Vec<_> = trades.iter().filter(|t| t.pnl > Decimal::ZERO).collect();
        let losing_trades: Vec<_> = trades.iter().filter(|t| t.pnl < Decimal::ZERO).collect();

        let win_count = winning_trades.len();
        let loss_count = losing_trades.len();

        let win_rate = if trade_count > 0 {
            Decimal::from(win_count) / Decimal::from(trade_count) * dec!(100)
        } else {
            Decimal::ZERO
        };

        let total_fees: Decimal = trades.iter().map(|t| t.fees).sum();
        let total_size_usd: Decimal = trades.iter().map(|t| t.size_usd).sum();

        let avg_trade_pnl = if trade_count > 0 {
            total_pnl / Decimal::from(trade_count)
        } else {
            Decimal::ZERO
        };

        let avg_win = if win_count > 0 {
            winning_trades.iter().map(|t| t.pnl).sum::<Decimal>() / Decimal::from(win_count)
        } else {
            Decimal::ZERO
        };

        let avg_loss = if loss_count > 0 {
            losing_trades.iter().map(|t| t.pnl).sum::<Decimal>() / Decimal::from(loss_count)
        } else {
            Decimal::ZERO
        };

        let gross_profit: Decimal = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_loss: Decimal = losing_trades.iter().map(|t| t.pnl.abs()).sum();

        let profit_factor = if gross_loss.is_zero() {
            if gross_profit.is_zero() {
                Decimal::ZERO
            } else {
                dec!(999.99) // Effectively infinite
            }
        } else {
            gross_profit / gross_loss
        };

        let avg_trade_size_usd = if trade_count > 0 {
            total_size_usd / Decimal::from(trade_count)
        } else {
            Decimal::ZERO
        };

        // Calculate max drawdown
        let (max_drawdown_pct, max_drawdown_usd) =
            Self::calculate_max_drawdown(trades, initial_capital);

        // Calculate Sharpe ratio
        let sharpe_ratio = Self::calculate_sharpe_ratio(trades, initial_capital);

        Self {
            total_pnl,
            realized_pnl: total_pnl,
            unrealized_pnl: Decimal::ZERO,
            return_pct,
            win_rate,
            trade_count,
            winning_trades: win_count,
            losing_trades: loss_count,
            avg_trade_pnl,
            avg_win,
            avg_loss,
            profit_factor,
            max_drawdown_pct,
            max_drawdown_usd,
            sharpe_ratio,
            total_fees,
            avg_trade_size_usd,
        }
    }

    fn calculate_max_drawdown(trades: &[TradeResult], initial_capital: Decimal) -> (Decimal, Decimal) {
        if trades.is_empty() {
            return (Decimal::ZERO, Decimal::ZERO);
        }

        let mut equity = initial_capital;
        let mut peak = equity;
        let mut max_drawdown_usd = Decimal::ZERO;
        let mut max_drawdown_pct = Decimal::ZERO;

        for trade in trades {
            equity += trade.pnl;

            if equity > peak {
                peak = equity;
            }

            let drawdown_usd = peak - equity;
            let drawdown_pct = if peak.is_zero() {
                Decimal::ZERO
            } else {
                (drawdown_usd / peak) * dec!(100)
            };

            if drawdown_usd > max_drawdown_usd {
                max_drawdown_usd = drawdown_usd;
                max_drawdown_pct = drawdown_pct;
            }
        }

        (max_drawdown_pct, max_drawdown_usd)
    }

    fn calculate_sharpe_ratio(trades: &[TradeResult], initial_capital: Decimal) -> Decimal {
        if trades.len() < 2 {
            return Decimal::ZERO;
        }

        let returns: Vec<Decimal> = trades
            .iter()
            .map(|t| {
                if initial_capital.is_zero() {
                    Decimal::ZERO
                } else {
                    t.pnl / initial_capital
                }
            })
            .collect();

        let n = Decimal::from(returns.len());
        let mean_return = returns.iter().sum::<Decimal>() / n;

        // Calculate standard deviation
        let variance = returns
            .iter()
            .map(|r| {
                let diff = *r - mean_return;
                diff * diff
            })
            .sum::<Decimal>()
            / n;

        // Use Newton-Raphson approximation for square root
        let std_dev = Self::decimal_sqrt(variance);

        if std_dev.is_zero() {
            return Decimal::ZERO;
        }

        // Annualize (assume ~252 trading days per year)
        // Sharpe = (mean_return * sqrt(252)) / (std_dev * sqrt(252))
        //        = mean_return / std_dev * sqrt(252)
        let sqrt_252 = dec!(15.87); // sqrt(252) ≈ 15.87

        (mean_return / std_dev) * sqrt_252
    }

    fn decimal_sqrt(n: Decimal) -> Decimal {
        if n.is_zero() || n.is_sign_negative() {
            return Decimal::ZERO;
        }

        // Newton-Raphson method
        let mut x = n / dec!(2);
        for _ in 0..20 {
            let next_x = (x + n / x) / dec!(2);
            if (next_x - x).abs() < dec!(0.0000001) {
                break;
            }
            x = next_x;
        }
        x
    }
}

/// Per-market performance breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketPerformance {
    /// Number of trades in this market
    pub trade_count: usize,
    /// Total P&L for this market
    pub total_pnl: Decimal,
    /// Win rate for this market
    pub win_rate: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn make_trade(pnl: Decimal, size_usd: Decimal, fees: Decimal) -> TradeResult {
        TradeResult {
            id: "test".to_string(),
            signal_id: "sig".to_string(),
            market_id: "market".to_string(),
            token_id: "token".to_string(),
            is_buy: true,
            entry_price: dec!(0.50),
            exit_price: Some(dec!(0.55)),
            size: dec!(100),
            size_usd,
            fees,
            pnl,
            timestamp: Utc::now(),
            reason: "test".to_string(),
        }
    }

    #[test]
    fn test_metrics_calculation() {
        let trades = vec![
            make_trade(dec!(10), dec!(100), dec!(0.20)),
            make_trade(dec!(-5), dec!(100), dec!(0.20)),
            make_trade(dec!(15), dec!(100), dec!(0.20)),
            make_trade(dec!(-3), dec!(100), dec!(0.20)),
        ];

        let initial = dec!(1000);
        let final_cap = dec!(1017); // 1000 + 10 - 5 + 15 - 3

        let metrics = PerformanceMetrics::calculate(&trades, initial, final_cap);

        assert_eq!(metrics.trade_count, 4);
        assert_eq!(metrics.winning_trades, 2);
        assert_eq!(metrics.losing_trades, 2);
        assert_eq!(metrics.total_pnl, dec!(17));
        assert_eq!(metrics.win_rate, dec!(50));
        assert_eq!(metrics.total_fees, dec!(0.80));
    }

    #[test]
    fn test_max_drawdown() {
        let trades = vec![
            make_trade(dec!(10), dec!(100), dec!(0)),
            make_trade(dec!(-20), dec!(100), dec!(0)),
            make_trade(dec!(5), dec!(100), dec!(0)),
        ];

        let (_dd_pct, dd_usd) = PerformanceMetrics::calculate_max_drawdown(&trades, dec!(100));

        // Peak was 110, then dropped to 90, so drawdown = 20
        assert_eq!(dd_usd, dec!(20));
    }

    #[test]
    fn test_profit_factor() {
        let trades = vec![
            make_trade(dec!(20), dec!(100), dec!(0)),
            make_trade(dec!(-10), dec!(100), dec!(0)),
        ];

        let metrics = PerformanceMetrics::calculate(&trades, dec!(100), dec!(110));

        // Profit factor = 20 / 10 = 2
        assert_eq!(metrics.profit_factor, dec!(2));
    }

    #[test]
    fn test_decimal_sqrt() {
        let result = PerformanceMetrics::decimal_sqrt(dec!(16));
        assert!((result - dec!(4)).abs() < dec!(0.0001));

        let result = PerformanceMetrics::decimal_sqrt(dec!(2));
        assert!((result - dec!(1.4142)).abs() < dec!(0.001));
    }
}
