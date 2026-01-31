//! Market feature computer
//!
//! Computes features related to market metadata like liquidity, volume, and price.

use crate::feature_store::{FeatureComputer, FeatureContext, Result};
use async_trait::async_trait;
use chrono::Duration;
use rust_decimal::Decimal;

/// Computes market-level features
pub struct MarketFeatureComputer;

impl MarketFeatureComputer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MarketFeatureComputer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FeatureComputer for MarketFeatureComputer {
    fn name(&self) -> &str {
        "market"
    }

    fn version(&self) -> &str {
        "1.0"
    }

    async fn compute(&self, context: &FeatureContext) -> Result<serde_json::Value> {
        let market = &context.market;

        // Current price (YES token)
        let current_price = context.price.unwrap_or(Decimal::new(50, 2)); // 0.50 default

        // Price spread (YES + NO = 1.0, so spread is distance from either extreme)
        let price_spread = if current_price <= Decimal::new(50, 2) {
            current_price
        } else {
            Decimal::ONE - current_price
        };

        // Time to expiry in hours
        let time_to_expiry_hours = market.end_date.map(|end| {
            let remaining = end - context.timestamp;
            remaining.num_hours() as f64
        });

        Ok(serde_json::json!({
            "liquidity": market.liquidity.to_string(),
            "volume": market.volume.to_string(),
            "current_price": current_price.to_string(),
            "price_spread": price_spread.to_string(),
            "time_to_expiry_hours": time_to_expiry_hours,
            "is_active": market.active,
            "is_closed": market.closed,
            "tag_count": market.tags.len(),
        }))
    }

    fn dependencies(&self) -> Vec<&str> {
        vec![]
    }

    fn default_ttl(&self) -> Duration {
        Duration::seconds(30) // Market features can change frequently
    }
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
            question: "Will this happen?".to_string(),
            description: Some("Test description".to_string()),
            tags: vec!["politics".to_string()],
            yes_token_id: "yes_token".to_string(),
            no_token_id: "no_token".to_string(),
            created_at: Utc::now(),
            end_date: Some(Utc::now() + chrono::Duration::days(7)),
            active: true,
            closed: false,
            volume: dec!(10000),
            liquidity: dec!(5000),
        }
    }

    #[tokio::test]
    async fn test_market_features() {
        let computer = MarketFeatureComputer::new();
        let market = create_test_market();
        let context = FeatureContext::new(market, Utc::now()).with_price(dec!(0.65));

        let result = computer.compute(&context).await.unwrap();

        assert!(result.get("liquidity").is_some());
        assert!(result.get("volume").is_some());
        assert!(result.get("current_price").is_some());
        assert!(result.get("time_to_expiry_hours").is_some());
    }
}
