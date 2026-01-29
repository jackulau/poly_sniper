//! Stats CLI command for strategy performance comparison

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use clap::{Args, Subcommand, ValueEnum};
use comfy_table::{presets::UTF8_FULL, Attribute, Cell, Color, ContentArrangement, Table};
use polysniper_persistence::{
    Database, RankingMetric, StrategyMetrics, StrategyMetricsRepository, TimePeriod,
};
use rust_decimal::Decimal;
use serde::Serialize;

/// Stats command for viewing strategy performance
#[derive(Debug, Args)]
pub struct StatsCommand {
    #[command(subcommand)]
    pub command: StatsSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum StatsSubcommand {
    /// Show strategy performance comparison
    Strategies(StrategiesArgs),
}

#[derive(Debug, Args)]
pub struct StrategiesArgs {
    /// Time period filter
    #[arg(long, short, default_value = "month")]
    pub period: Period,

    /// Start date (YYYY-MM-DD) for custom range
    #[arg(long, requires = "to")]
    pub from: Option<NaiveDate>,

    /// End date (YYYY-MM-DD) for custom range
    #[arg(long, requires = "from")]
    pub to: Option<NaiveDate>,

    /// Strategy ID to show detailed stats for
    #[arg(long, short)]
    pub strategy: Option<String>,

    /// Metric to rank strategies by
    #[arg(long, short, default_value = "net-pnl")]
    pub rank_by: RankBy,

    /// Output format
    #[arg(long, short, default_value = "table")]
    pub format: OutputFormat,

    /// Path to database file
    #[arg(long, default_value = "data/polysniper.db")]
    pub db_path: String,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Period {
    Today,
    Week,
    Month,
    All,
}

impl From<Period> for TimePeriod {
    fn from(p: Period) -> Self {
        match p {
            Period::Today => TimePeriod::Today,
            Period::Week => TimePeriod::Week,
            Period::Month => TimePeriod::Month,
            Period::All => TimePeriod::All,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum RankBy {
    NetPnl,
    WinRate,
    ProfitFactor,
    Sharpe,
    Trades,
    Volume,
}

impl From<RankBy> for RankingMetric {
    fn from(r: RankBy) -> Self {
        match r {
            RankBy::NetPnl => RankingMetric::NetPnl,
            RankBy::WinRate => RankingMetric::WinRate,
            RankBy::ProfitFactor => RankingMetric::ProfitFactor,
            RankBy::Sharpe => RankingMetric::SharpeRatio,
            RankBy::Trades => RankingMetric::TotalTrades,
            RankBy::Volume => RankingMetric::TotalVolume,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    Table,
    Json,
    Csv,
}

impl StatsCommand {
    pub async fn run(&self) -> Result<()> {
        match &self.command {
            StatsSubcommand::Strategies(args) => run_strategies_stats(args).await,
        }
    }
}

async fn run_strategies_stats(args: &StrategiesArgs) -> Result<()> {
    // Connect to database
    let db = Database::new(&args.db_path)
        .await
        .context("Failed to connect to database")?;

    let repo = StrategyMetricsRepository::new(&db);

    // Determine time bounds
    let (start, end) = if args.from.is_some() && args.to.is_some() {
        use chrono::TimeZone;
        let start = Utc.from_utc_datetime(
            &args
                .from
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
        );
        let end = Utc.from_utc_datetime(
            &args
                .to
                .unwrap()
                .and_hms_opt(23, 59, 59)
                .unwrap(),
        );
        (Some(start), Some(end))
    } else {
        StrategyMetricsRepository::get_time_bounds(args.period.into())
    };

    // Get metrics
    let mut metrics = if let Some(strategy_id) = &args.strategy {
        // Single strategy detail
        match repo.get_metrics(strategy_id, start, end).await? {
            Some(m) => vec![m],
            None => vec![],
        }
    } else {
        // All strategies
        repo.get_all_metrics(start, end).await?
    };

    // Sort by ranking metric
    StrategyMetricsRepository::sort_by_metric(&mut metrics, args.rank_by.into(), true);

    // Get totals
    let totals = repo.get_total_metrics(start, end).await?;

    // Output based on format
    match args.format {
        OutputFormat::Table => output_table(&metrics, &totals, args),
        OutputFormat::Json => output_json(&metrics, &totals)?,
        OutputFormat::Csv => output_csv(&metrics)?,
    }

    Ok(())
}

fn output_table(metrics: &[StrategyMetrics], totals: &StrategyMetrics, args: &StrategiesArgs) {
    let period_label = match args.period {
        Period::Today => "Today",
        Period::Week => "Last 7 Days",
        Period::Month => "Last 30 Days",
        Period::All => "All Time",
    };

    println!();
    println!("Strategy Performance ({})", period_label);
    println!();

    if metrics.is_empty() {
        println!("No trades found for the selected period.");
        return;
    }

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Strategy").add_attribute(Attribute::Bold),
            Cell::new("Trades").add_attribute(Attribute::Bold),
            Cell::new("Win%").add_attribute(Attribute::Bold),
            Cell::new("Net P&L").add_attribute(Attribute::Bold),
            Cell::new("Profit Factor").add_attribute(Attribute::Bold),
            Cell::new("Sharpe").add_attribute(Attribute::Bold),
            Cell::new("Volume").add_attribute(Attribute::Bold),
        ]);

    for m in metrics {
        let pnl_color = if m.net_pnl >= Decimal::ZERO {
            Color::Green
        } else {
            Color::Red
        };

        let pnl_str = format_pnl(m.net_pnl);
        let pf_str = m
            .profit_factor
            .map(|pf| {
                if pf.is_infinite() {
                    "∞".to_string()
                } else {
                    format!("{:.2}", pf)
                }
            })
            .unwrap_or_else(|| "-".to_string());
        let sharpe_str = m
            .sharpe_ratio
            .map(|s| format!("{:.2}", s))
            .unwrap_or_else(|| "-".to_string());

        table.add_row(vec![
            Cell::new(&m.strategy_id),
            Cell::new(m.total_trades),
            Cell::new(format!("{:.1}%", m.win_rate)),
            Cell::new(&pnl_str).fg(pnl_color),
            Cell::new(&pf_str),
            Cell::new(&sharpe_str),
            Cell::new(format_volume(m.total_volume)),
        ]);
    }

    // Add totals row
    if metrics.len() > 1 {
        let totals_pnl_color = if totals.net_pnl >= Decimal::ZERO {
            Color::Green
        } else {
            Color::Red
        };

        let totals_pf_str = totals
            .profit_factor
            .map(|pf| {
                if pf.is_infinite() {
                    "∞".to_string()
                } else {
                    format!("{:.2}", pf)
                }
            })
            .unwrap_or_else(|| "-".to_string());
        let totals_sharpe_str = totals
            .sharpe_ratio
            .map(|s| format!("{:.2}", s))
            .unwrap_or_else(|| "-".to_string());

        table.add_row(vec![
            Cell::new("TOTAL").add_attribute(Attribute::Bold),
            Cell::new(totals.total_trades).add_attribute(Attribute::Bold),
            Cell::new(format!("{:.1}%", totals.win_rate)).add_attribute(Attribute::Bold),
            Cell::new(format_pnl(totals.net_pnl))
                .fg(totals_pnl_color)
                .add_attribute(Attribute::Bold),
            Cell::new(&totals_pf_str).add_attribute(Attribute::Bold),
            Cell::new(&totals_sharpe_str).add_attribute(Attribute::Bold),
            Cell::new(format_volume(totals.total_volume)).add_attribute(Attribute::Bold),
        ]);
    }

    println!("{table}");

    // Show detailed info for single strategy
    if let Some(strategy_id) = &args.strategy {
        if let Some(m) = metrics.first() {
            println!();
            println!("Detailed Stats for {}", strategy_id);
            println!();
            println!("  Wins: {} | Losses: {}", m.win_count, m.loss_count);
            println!(
                "  Gross Profit: {} | Gross Loss: {}",
                format_pnl(m.gross_profit),
                format_pnl(m.gross_loss)
            );
            if let Some(avg_win) = m.avg_win {
                println!("  Average Win: {}", format_pnl(avg_win));
            }
            if let Some(avg_loss) = m.avg_loss {
                println!("  Average Loss: {}", format_pnl(avg_loss));
            }
            if let Some(largest_win) = m.largest_win {
                println!("  Largest Win: {}", format_pnl(largest_win));
            }
            if let Some(largest_loss) = m.largest_loss {
                println!("  Largest Loss: {}", format_pnl(largest_loss));
            }
            println!(
                "  Max Consecutive: {} wins, {} losses",
                m.max_consecutive_wins, m.max_consecutive_losses
            );
            println!("  Avg Trade Size: {}", format_volume(m.avg_trade_size));
        }
    }

    println!();
}

fn output_json(metrics: &[StrategyMetrics], totals: &StrategyMetrics) -> Result<()> {
    #[derive(Serialize)]
    struct Output<'a> {
        strategies: &'a [StrategyMetrics],
        totals: &'a StrategyMetrics,
        generated_at: DateTime<Utc>,
    }

    let output = Output {
        strategies: metrics,
        totals,
        generated_at: Utc::now(),
    };

    let json = serde_json::to_string_pretty(&output)?;
    println!("{}", json);
    Ok(())
}

fn output_csv(metrics: &[StrategyMetrics]) -> Result<()> {
    let mut wtr = csv::Writer::from_writer(std::io::stdout());

    // Write header
    wtr.write_record([
        "strategy_id",
        "total_trades",
        "win_count",
        "loss_count",
        "gross_profit",
        "gross_loss",
        "net_pnl",
        "win_rate",
        "profit_factor",
        "avg_trade_size",
        "sharpe_ratio",
        "total_volume",
    ])?;

    for m in metrics {
        wtr.write_record([
            &m.strategy_id,
            &m.total_trades.to_string(),
            &m.win_count.to_string(),
            &m.loss_count.to_string(),
            &m.gross_profit.to_string(),
            &m.gross_loss.to_string(),
            &m.net_pnl.to_string(),
            &format!("{:.2}", m.win_rate),
            &m.profit_factor
                .map(|pf| format!("{:.4}", pf))
                .unwrap_or_default(),
            &m.avg_trade_size.to_string(),
            &m.sharpe_ratio
                .map(|s| format!("{:.4}", s))
                .unwrap_or_default(),
            &m.total_volume.to_string(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

fn format_pnl(value: Decimal) -> String {
    if value >= Decimal::ZERO {
        format!("+${:.2}", value)
    } else {
        format!("-${:.2}", value.abs())
    }
}

fn format_volume(value: Decimal) -> String {
    if value >= Decimal::from(1_000_000) {
        format!("${:.2}M", value / Decimal::from(1_000_000))
    } else if value >= Decimal::from(1_000) {
        format!("${:.2}K", value / Decimal::from(1_000))
    } else {
        format!("${:.2}", value)
    }
}
