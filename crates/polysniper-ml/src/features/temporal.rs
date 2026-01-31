//! Temporal feature computer
//!
//! Computes time-based features like hour of day, day of week, time to resolution.

use crate::feature_store::{FeatureComputer, FeatureContext, Result};
use async_trait::async_trait;
use chrono::{Datelike, Duration, Timelike};

/// Computes temporal features
pub struct TemporalFeatureComputer;

impl TemporalFeatureComputer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TemporalFeatureComputer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FeatureComputer for TemporalFeatureComputer {
    fn name(&self) -> &str {
        "temporal"
    }

    fn version(&self) -> &str {
        "1.0"
    }

    async fn compute(&self, context: &FeatureContext) -> Result<serde_json::Value> {
        let market = &context.market;
        let now = context.timestamp;

        // Hour of day (0-23)
        let hour_of_day = now.hour() as i32;

        // Day of week (0 = Monday, 6 = Sunday)
        let day_of_week = now.weekday().num_days_from_monday() as i32;

        // Is weekend
        let is_weekend = day_of_week >= 5;

        // Is US market hours (9:30 AM - 4:00 PM ET, roughly 14:30 - 21:00 UTC)
        let is_us_market_hours = hour_of_day >= 14 && hour_of_day < 21;

        // Time to resolution
        let (time_to_resolution_secs, time_to_resolution_hours, time_to_resolution_days) =
            if let Some(end_date) = market.end_date {
                let remaining = end_date - now;
                let secs = remaining.num_seconds();
                let hours = remaining.num_hours() as f64;
                let days = remaining.num_days() as f64;
                (Some(secs), Some(hours), Some(days))
            } else {
                (None, None, None)
            };

        // Market age in days
        let market_age_days = (now - market.created_at).num_days() as f64;

        // Is near resolution (within 24 hours)
        let is_near_resolution = time_to_resolution_hours
            .map(|h| h <= 24.0 && h > 0.0)
            .unwrap_or(false);

        // Is very close to resolution (within 1 hour)
        let is_very_close = time_to_resolution_hours
            .map(|h| h <= 1.0 && h > 0.0)
            .unwrap_or(false);

        // Month of year (1-12)
        let month = now.month() as i32;

        // Quarter (1-4)
        let quarter = ((month - 1) / 3) + 1;

        // Is end of month (last 3 days)
        let is_end_of_month = now.day() >= 28;

        // Is market mature (more than 7 days old)
        let is_mature = market_age_days >= 7.0;

        Ok(serde_json::json!({
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_us_market_hours": is_us_market_hours,
            "time_to_resolution_secs": time_to_resolution_secs,
            "time_to_resolution_hours": time_to_resolution_hours,
            "time_to_resolution_days": time_to_resolution_days,
            "market_age_days": market_age_days,
            "is_near_resolution": is_near_resolution,
            "is_very_close": is_very_close,
            "month": month,
            "quarter": quarter,
            "is_end_of_month": is_end_of_month,
            "is_mature": is_mature,
        }))
    }

    fn dependencies(&self) -> Vec<&str> {
        vec![]
    }

    fn default_ttl(&self) -> Duration {
        Duration::seconds(60) // Temporal features change with time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use polysniper_core::Market;
    use rust_decimal_macros::dec;

    fn create_test_market(end_in_hours: Option<i64>) -> Market {
        let end_date = end_in_hours.map(|h| Utc::now() + chrono::Duration::hours(h));
        Market {
            condition_id: "test_market".to_string(),
            question: "Test?".to_string(),
            description: None,
            tags: vec![],
            yes_token_id: "yes_token".to_string(),
            no_token_id: "no_token".to_string(),
            created_at: Utc::now() - chrono::Duration::days(10),
            end_date,
            active: true,
            closed: false,
            volume: dec!(1000),
            liquidity: dec!(500),
        }
    }

    #[tokio::test]
    async fn test_temporal_features() {
        let computer = TemporalFeatureComputer::new();
        let market = create_test_market(Some(48));
        let context = FeatureContext::new(market, Utc::now());

        let result = computer.compute(&context).await.unwrap();

        assert!(result.get("hour_of_day").is_some());
        assert!(result.get("day_of_week").is_some());
        assert!(result.get("time_to_resolution_hours").is_some());
        assert!(result.get("market_age_days").is_some());
    }

    #[tokio::test]
    async fn test_near_resolution() {
        let computer = TemporalFeatureComputer::new();
        let market = create_test_market(Some(12)); // 12 hours until resolution
        let context = FeatureContext::new(market, Utc::now());

        let result = computer.compute(&context).await.unwrap();

        assert!(result.get("is_near_resolution").unwrap().as_bool().unwrap());
        assert!(!result.get("is_very_close").unwrap().as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_no_end_date() {
        let computer = TemporalFeatureComputer::new();
        let market = create_test_market(None);
        let context = FeatureContext::new(market, Utc::now());

        let result = computer.compute(&context).await.unwrap();

        assert!(result.get("time_to_resolution_secs").unwrap().is_null());
    }
}
