//! Backtest CLI command implementation

use anyhow::{Context, Result};
use chrono::{NaiveDate, TimeZone, Utc};
use polysniper_backtest::{BacktestConfig, BacktestEngine};
use polysniper_core::Strategy;
use polysniper_strategies::{
    EventBasedConfig, EventBasedStrategy, NewMarketConfig, NewMarketStrategy, PriceSpikeConfig,
    PriceSpikeStrategy, TargetPriceConfig, TargetPriceStrategy,
};
use rust_decimal::Decimal;
use std::str::FromStr;
use tracing::info;

const STRATEGIES_CONFIG_DIR: &str = "config/strategies";

/// Run a backtest with the specified parameters
pub async fn run_backtest(
    strategy_name: &str,
    from: Option<String>,
    to: Option<String>,
    capital: f64,
    db_path: &str,
    output_format: &str,
) -> Result<()> {
    info!(
        strategy = %strategy_name,
        db_path = %db_path,
        "Starting backtest"
    );

    // Parse dates
    let start_time = match from {
        Some(date_str) => {
            let date = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
                .with_context(|| format!("Invalid start date format: {}", date_str))?;
            Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap())
        }
        None => Utc::now() - chrono::Duration::days(30),
    };

    let end_time = match to {
        Some(date_str) => {
            let date = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
                .with_context(|| format!("Invalid end date format: {}", date_str))?;
            Utc.from_utc_datetime(&date.and_hms_opt(23, 59, 59).unwrap())
        }
        None => Utc::now(),
    };

    // Validate date range
    if start_time >= end_time {
        anyhow::bail!("Start date must be before end date");
    }

    // Create backtest config
    let config = BacktestConfig {
        start_time,
        end_time,
        initial_capital: Decimal::from_str(&format!("{:.2}", capital))
            .unwrap_or(Decimal::new(10000, 0)),
        ..Default::default()
    };

    // Create backtest engine
    let engine = BacktestEngine::new(config, db_path)
        .await
        .context("Failed to create backtest engine")?;

    // Check data availability
    if let Some((data_start, data_end)) = engine.get_data_range().await? {
        info!(
            data_start = %data_start,
            data_end = %data_end,
            "Available data range"
        );

        if start_time < data_start || end_time > data_end {
            eprintln!(
                "Warning: Requested time range ({} to {}) partially outside available data ({} to {})",
                start_time.format("%Y-%m-%d"),
                end_time.format("%Y-%m-%d"),
                data_start.format("%Y-%m-%d"),
                data_end.format("%Y-%m-%d")
            );
        }
    } else {
        anyhow::bail!("No historical data available in database");
    }

    // Load strategy
    let mut strategy = load_strategy(strategy_name)?;
    strategy
        .initialize(&polysniper_backtest::engine::SimulatedState::new(
            Decimal::new(10000, 0),
        ))
        .await
        .context("Failed to initialize strategy")?;

    // Run backtest
    let results = engine.run(strategy.as_ref()).await?;

    // Output results
    match output_format {
        "json" => {
            println!("{}", results.to_json()?);
        }
        "csv" => {
            println!("{}", results.trades_to_csv());
        }
        _ => {
            print_results_text(&results);
        }
    }

    Ok(())
}

fn load_strategy(name: &str) -> Result<Box<dyn Strategy>> {
    match name {
        "target_price" => {
            let config = load_strategy_config::<TargetPriceConfig>("target_price")?
                .unwrap_or_else(|| TargetPriceConfig {
                    enabled: true,
                    targets: Vec::new(),
                });
            Ok(Box::new(TargetPriceStrategy::from_config(config)))
        }
        "price_spike" => {
            let config =
                load_strategy_config::<PriceSpikeConfig>("price_spike")?.unwrap_or_default();
            Ok(Box::new(PriceSpikeStrategy::new(config)))
        }
        "new_market" => {
            let config =
                load_strategy_config::<NewMarketConfig>("new_market")?.unwrap_or_default();
            Ok(Box::new(NewMarketStrategy::new(config)))
        }
        "event_based" => {
            let config =
                load_strategy_config::<EventBasedConfig>("event_based")?.unwrap_or_default();
            Ok(Box::new(EventBasedStrategy::new(config)))
        }
        _ => anyhow::bail!("Unknown strategy: {}. Available: target_price, price_spike, new_market, event_based", name),
    }
}

fn load_strategy_config<T: serde::de::DeserializeOwned>(name: &str) -> Result<Option<T>> {
    let config_path = format!("{}/{}.toml", STRATEGIES_CONFIG_DIR, name);
    if std::path::Path::new(&config_path).exists() {
        let content = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read strategy config: {}", config_path))?;
        let config: T = toml::from_str(&content)
            .with_context(|| format!("Failed to parse strategy config: {}", config_path))?;
        Ok(Some(config))
    } else {
        Ok(None)
    }
}

fn print_results_text(results: &polysniper_backtest::BacktestResults) {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    BACKTEST RESULTS                            ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Strategy:        {:<44} ║",
        results.strategy_id
    );
    println!(
        "║ Period:          {} to {} ║",
        results.start_time.format("%Y-%m-%d"),
        results.end_time.format("%Y-%m-%d")
    );
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                    PERFORMANCE METRICS                         ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Initial Capital:  ${:<43.2} ║",
        results.initial_capital
    );
    println!(
        "║ Final Capital:    ${:<43.2} ║",
        results.final_capital
    );
    println!(
        "║ Total P&L:        ${:<43.2} ║",
        results.metrics.total_pnl
    );
    println!(
        "║ Return:           {:<43.2}% ║",
        results.metrics.return_pct
    );
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Trade Count:      {:<44} ║",
        results.metrics.trade_count
    );
    println!(
        "║ Winning Trades:   {:<44} ║",
        results.metrics.winning_trades
    );
    println!(
        "║ Losing Trades:    {:<44} ║",
        results.metrics.losing_trades
    );
    println!(
        "║ Win Rate:         {:<43.2}% ║",
        results.metrics.win_rate
    );
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Avg Trade P&L:    ${:<43.2} ║",
        results.metrics.avg_trade_pnl
    );
    println!(
        "║ Avg Win:          ${:<43.2} ║",
        results.metrics.avg_win
    );
    println!(
        "║ Avg Loss:         ${:<43.2} ║",
        results.metrics.avg_loss
    );
    println!(
        "║ Profit Factor:    {:<44.2} ║",
        results.metrics.profit_factor
    );
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Max Drawdown:     {:<43.2}% ║",
        results.metrics.max_drawdown_pct
    );
    println!(
        "║ Max Drawdown $:   ${:<43.2} ║",
        results.metrics.max_drawdown_usd
    );
    println!(
        "║ Sharpe Ratio:     {:<44.2} ║",
        results.metrics.sharpe_ratio
    );
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Total Fees:       ${:<43.2} ║",
        results.metrics.total_fees
    );
    println!(
        "║ Avg Trade Size:   ${:<43.2} ║",
        results.metrics.avg_trade_size_usd
    );
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Market breakdown
    if !results.market_breakdown.is_empty() {
        println!("Market Breakdown:");
        println!("─────────────────────────────────────────────────────────────────");
        for (market_id, perf) in &results.market_breakdown {
            let market_display = if market_id.len() > 30 {
                format!("{}...", &market_id[..27])
            } else {
                market_id.clone()
            };
            println!(
                "  {}: {} trades, P&L: ${:.2}, Win Rate: {:.1}%",
                market_display,
                perf.trade_count,
                perf.total_pnl,
                perf.win_rate * rust_decimal::Decimal::ONE_HUNDRED
            );
        }
        println!();
    }
}
