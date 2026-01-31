//! Orderbook feature computer
//!
//! Computes features from orderbook data like depth, imbalance, and microprice.

use crate::feature_store::{FeatureComputer, FeatureContext, Result, FeatureStoreError};
use async_trait::async_trait;
use chrono::Duration;
use rust_decimal::Decimal;

/// Computes orderbook-related features
pub struct OrderbookFeatureComputer;

impl OrderbookFeatureComputer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OrderbookFeatureComputer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FeatureComputer for OrderbookFeatureComputer {
    fn name(&self) -> &str {
        "orderbook"
    }

    fn version(&self) -> &str {
        "1.0"
    }

    async fn compute(&self, context: &FeatureContext) -> Result<serde_json::Value> {
        let orderbook = context.orderbook.as_ref().ok_or_else(|| {
            FeatureStoreError::ComputationError("Orderbook not available".to_string())
        })?;

        // Calculate bid depth (total size on bid side)
        let bid_depth: Decimal = orderbook.bids.iter().map(|l| l.size).sum();

        // Calculate ask depth (total size on ask side)
        let ask_depth: Decimal = orderbook.asks.iter().map(|l| l.size).sum();

        // Calculate imbalance ratio
        let total_depth = bid_depth + ask_depth;
        let imbalance_ratio = if total_depth > Decimal::ZERO {
            (bid_depth - ask_depth) / total_depth
        } else {
            Decimal::ZERO
        };

        // Calculate microprice (volume-weighted mid price)
        let microprice = calculate_microprice(orderbook);

        // Calculate weighted mid price (considers deeper levels)
        let weighted_mid = calculate_weighted_mid(orderbook);

        // Get spread
        let spread = orderbook.spread().unwrap_or(Decimal::ZERO);

        // Best prices
        let best_bid = orderbook.best_bid().unwrap_or(Decimal::ZERO);
        let best_ask = orderbook.best_ask().unwrap_or(Decimal::ONE);

        // Depth at top levels (first 3)
        let top_bid_depth: Decimal = orderbook.bids.iter().take(3).map(|l| l.size).sum();
        let top_ask_depth: Decimal = orderbook.asks.iter().take(3).map(|l| l.size).sum();

        Ok(serde_json::json!({
            "bid_depth": bid_depth.to_string(),
            "ask_depth": ask_depth.to_string(),
            "imbalance_ratio": imbalance_ratio.to_string(),
            "microprice": microprice.map(|p| p.to_string()),
            "weighted_mid": weighted_mid.map(|p| p.to_string()),
            "spread": spread.to_string(),
            "best_bid": best_bid.to_string(),
            "best_ask": best_ask.to_string(),
            "top_bid_depth": top_bid_depth.to_string(),
            "top_ask_depth": top_ask_depth.to_string(),
            "bid_levels": orderbook.bids.len(),
            "ask_levels": orderbook.asks.len(),
        }))
    }

    fn dependencies(&self) -> Vec<&str> {
        vec![]
    }

    fn default_ttl(&self) -> Duration {
        Duration::seconds(5) // Orderbook features change very frequently
    }
}

/// Calculate microprice (volume-weighted price at top of book)
fn calculate_microprice(orderbook: &polysniper_core::Orderbook) -> Option<Decimal> {
    let best_bid = orderbook.bids.first()?;
    let best_ask = orderbook.asks.first()?;

    let total_size = best_bid.size + best_ask.size;
    if total_size.is_zero() {
        return None;
    }

    // Microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
    let microprice = (best_bid.price * best_ask.size + best_ask.price * best_bid.size) / total_size;
    Some(microprice)
}

/// Calculate weighted mid price considering multiple levels
fn calculate_weighted_mid(orderbook: &polysniper_core::Orderbook) -> Option<Decimal> {
    if orderbook.bids.is_empty() || orderbook.asks.is_empty() {
        return None;
    }

    // Take up to 5 levels on each side
    let bid_levels: Vec<_> = orderbook.bids.iter().take(5).collect();
    let ask_levels: Vec<_> = orderbook.asks.iter().take(5).collect();

    let mut weighted_bid_sum = Decimal::ZERO;
    let mut bid_weight_sum = Decimal::ZERO;
    let mut weighted_ask_sum = Decimal::ZERO;
    let mut ask_weight_sum = Decimal::ZERO;

    for (i, level) in bid_levels.iter().enumerate() {
        let weight = Decimal::from(5 - i as i64); // Decreasing weight for deeper levels
        weighted_bid_sum += level.price * level.size * weight;
        bid_weight_sum += level.size * weight;
    }

    for (i, level) in ask_levels.iter().enumerate() {
        let weight = Decimal::from(5 - i as i64);
        weighted_ask_sum += level.price * level.size * weight;
        ask_weight_sum += level.size * weight;
    }

    if bid_weight_sum.is_zero() || ask_weight_sum.is_zero() {
        return None;
    }

    let weighted_bid = weighted_bid_sum / bid_weight_sum;
    let weighted_ask = weighted_ask_sum / ask_weight_sum;

    Some((weighted_bid + weighted_ask) / Decimal::TWO)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use polysniper_core::{Market, Orderbook, PriceLevel};
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

    fn create_test_orderbook() -> Orderbook {
        Orderbook {
            token_id: "yes_token".to_string(),
            market_id: "test_market".to_string(),
            bids: vec![
                PriceLevel { price: dec!(0.48), size: dec!(100) },
                PriceLevel { price: dec!(0.47), size: dec!(200) },
            ],
            asks: vec![
                PriceLevel { price: dec!(0.52), size: dec!(100) },
                PriceLevel { price: dec!(0.53), size: dec!(150) },
            ],
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_orderbook_features() {
        let computer = OrderbookFeatureComputer::new();
        let market = create_test_market();
        let orderbook = create_test_orderbook();
        let context = FeatureContext::new(market, Utc::now()).with_orderbook(orderbook);

        let result = computer.compute(&context).await.unwrap();

        assert!(result.get("bid_depth").is_some());
        assert!(result.get("ask_depth").is_some());
        assert!(result.get("imbalance_ratio").is_some());
        assert!(result.get("microprice").is_some());
    }

    #[tokio::test]
    async fn test_orderbook_without_data() {
        let computer = OrderbookFeatureComputer::new();
        let market = create_test_market();
        let context = FeatureContext::new(market, Utc::now());

        let result = computer.compute(&context).await;
        assert!(result.is_err());
    }
}
