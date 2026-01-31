//! Price history feature computer
//!
//! Computes features from historical price data like volatility, momentum, and trends.

use crate::feature_store::{FeatureComputer, FeatureContext, Result, FeatureStoreError};
use async_trait::async_trait;
use chrono::Duration;
use rust_decimal::Decimal;

/// Configuration for price history feature computation
#[derive(Debug, Clone)]
pub struct PriceHistoryConfig {
    /// Window size in seconds for volatility calculation
    pub window_secs: u64,
    /// Minimum number of price points required for computation
    pub min_samples: usize,
}

impl Default for PriceHistoryConfig {
    fn default() -> Self {
        Self {
            window_secs: 3600, // 1 hour
            min_samples: 10,
        }
    }
}

/// Computes features from price history
pub struct PriceHistoryFeatureComputer {
    config: PriceHistoryConfig,
}

impl PriceHistoryFeatureComputer {
    pub fn new(config: PriceHistoryConfig) -> Self {
        Self { config }
    }
}

impl Default for PriceHistoryFeatureComputer {
    fn default() -> Self {
        Self::new(PriceHistoryConfig::default())
    }
}

#[async_trait]
impl FeatureComputer for PriceHistoryFeatureComputer {
    fn name(&self) -> &str {
        "price_history"
    }

    fn version(&self) -> &str {
        "1.0"
    }

    async fn compute(&self, context: &FeatureContext) -> Result<serde_json::Value> {
        let history = &context.price_history;

        if history.len() < self.config.min_samples {
            return Err(FeatureStoreError::ComputationError(format!(
                "Insufficient price history: {} samples, need {}",
                history.len(),
                self.config.min_samples
            )));
        }

        let prices: Vec<f64> = history
            .iter()
            .filter_map(|(_, p)| p.to_string().parse::<f64>().ok())
            .collect();

        if prices.is_empty() {
            return Err(FeatureStoreError::ComputationError(
                "No valid prices in history".to_string(),
            ));
        }

        // Calculate volatility (standard deviation of returns)
        let volatility = calculate_volatility(&prices);

        // Calculate momentum (recent price change)
        let momentum = calculate_momentum(&prices);

        // Calculate mean reversion signal
        let mean_reversion = calculate_mean_reversion(&prices);

        // Calculate price velocity (rate of change)
        let velocity = calculate_velocity(&prices, history);

        // Price range
        let (min_price, max_price) = min_max(&prices);
        let price_range = max_price - min_price;

        // Current price relative to range
        let current_price = prices.last().copied().unwrap_or(0.5);
        let range_position = if price_range > 0.0 {
            (current_price - min_price) / price_range
        } else {
            0.5
        };

        // Moving averages
        let sma_short = simple_moving_average(&prices, 5);
        let sma_long = simple_moving_average(&prices, 20);
        let sma_crossover = sma_short.map(|s| sma_long.map(|l| s > l)).flatten();

        // Trend strength (absolute value of momentum)
        let trend_strength = momentum.map(|m| m.abs());

        // Number of price points
        let sample_count = prices.len();

        Ok(serde_json::json!({
            "volatility": volatility,
            "momentum": momentum,
            "mean_reversion_signal": mean_reversion,
            "price_velocity": velocity,
            "price_range": price_range,
            "range_position": range_position,
            "min_price": min_price,
            "max_price": max_price,
            "current_price": current_price,
            "sma_short": sma_short,
            "sma_long": sma_long,
            "sma_crossover": sma_crossover,
            "trend_strength": trend_strength,
            "sample_count": sample_count,
        }))
    }

    fn dependencies(&self) -> Vec<&str> {
        vec![]
    }

    fn default_ttl(&self) -> Duration {
        Duration::seconds(30)
    }
}

/// Calculate volatility as standard deviation of returns
fn calculate_volatility(prices: &[f64]) -> Option<f64> {
    if prices.len() < 2 {
        return None;
    }

    // Calculate returns
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    if returns.is_empty() {
        return None;
    }

    // Calculate standard deviation
    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

    Some(variance.sqrt())
}

/// Calculate momentum as percentage change from oldest to newest
fn calculate_momentum(prices: &[f64]) -> Option<f64> {
    if prices.len() < 2 {
        return None;
    }

    let oldest = prices.first()?;
    let newest = prices.last()?;

    if *oldest == 0.0 {
        return None;
    }

    Some((newest - oldest) / oldest)
}

/// Calculate mean reversion signal (distance from mean)
fn calculate_mean_reversion(prices: &[f64]) -> Option<f64> {
    if prices.is_empty() {
        return None;
    }

    let mean: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
    let current = *prices.last()?;

    if mean == 0.0 {
        return None;
    }

    // Positive signal = price below mean (expect reversion up)
    // Negative signal = price above mean (expect reversion down)
    Some((mean - current) / mean)
}

/// Calculate price velocity (rate of change per time unit)
fn calculate_velocity(
    prices: &[f64],
    history: &[(chrono::DateTime<chrono::Utc>, Decimal)],
) -> Option<f64> {
    if history.len() < 2 {
        return None;
    }

    let first_time = history.first()?.0;
    let last_time = history.last()?.0;
    let duration_secs = (last_time - first_time).num_seconds() as f64;

    if duration_secs <= 0.0 {
        return None;
    }

    let first_price = prices.first()?;
    let last_price = prices.last()?;

    Some((last_price - first_price) / duration_secs)
}

/// Get min and max values
fn min_max(prices: &[f64]) -> (f64, f64) {
    let min = prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

/// Calculate simple moving average
fn simple_moving_average(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period {
        return None;
    }

    let recent: Vec<f64> = prices.iter().rev().take(period).cloned().collect();
    Some(recent.iter().sum::<f64>() / recent.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use polysniper_core::Market;
    use rust_decimal_macros::dec;

    fn create_test_market() -> Market {
        Market {
            condition_id: "test_market".to_string(),
            question: "Test?".to_string(),
            description: None,
            tags: vec![],
            yes_token_id: "yes_token".to_string(),
            no_token_id: "no_token".to_string(),
            created_at: Utc::now(),
            end_date: None,
            active: true,
            closed: false,
            volume: dec!(1000),
            liquidity: dec!(500),
        }
    }

    fn create_price_history() -> Vec<(chrono::DateTime<chrono::Utc>, Decimal)> {
        let now = Utc::now();
        vec![
            (now - chrono::Duration::minutes(10), dec!(0.45)),
            (now - chrono::Duration::minutes(9), dec!(0.46)),
            (now - chrono::Duration::minutes(8), dec!(0.47)),
            (now - chrono::Duration::minutes(7), dec!(0.48)),
            (now - chrono::Duration::minutes(6), dec!(0.49)),
            (now - chrono::Duration::minutes(5), dec!(0.50)),
            (now - chrono::Duration::minutes(4), dec!(0.51)),
            (now - chrono::Duration::minutes(3), dec!(0.52)),
            (now - chrono::Duration::minutes(2), dec!(0.53)),
            (now - chrono::Duration::minutes(1), dec!(0.54)),
            (now, dec!(0.55)),
        ]
    }

    #[tokio::test]
    async fn test_price_history_features() {
        let computer = PriceHistoryFeatureComputer::default();
        let market = create_test_market();
        let history = create_price_history();
        let context = FeatureContext::new(market, Utc::now()).with_price_history(history);

        let result = computer.compute(&context).await.unwrap();

        assert!(result.get("volatility").is_some());
        assert!(result.get("momentum").is_some());
        assert!(result.get("mean_reversion_signal").is_some());
    }

    #[tokio::test]
    async fn test_momentum_calculation() {
        let prices = vec![0.40, 0.42, 0.45, 0.48, 0.50];
        let momentum = calculate_momentum(&prices).unwrap();
        // (0.50 - 0.40) / 0.40 = 0.25
        assert!((momentum - 0.25).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_insufficient_history() {
        let computer = PriceHistoryFeatureComputer::default();
        let market = create_test_market();
        let history = vec![(Utc::now(), dec!(0.50))]; // Only 1 point
        let context = FeatureContext::new(market, Utc::now()).with_price_history(history);

        let result = computer.compute(&context).await;
        assert!(result.is_err());
    }
}
