//! Depth analyzer for orderbook-based order sizing
//!
//! Analyzes orderbook depth to determine optimal order sizes,
//! preventing excessive slippage and market impact.

use polysniper_core::{Orderbook, Side};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tracing::{debug, instrument};

/// Configuration for adaptive order sizing
#[derive(Debug, Clone)]
pub struct DepthAnalyzerConfig {
    /// Maximum acceptable price impact in basis points
    pub max_market_impact_bps: Decimal,
    /// Minimum ratio of our order to available depth (0.0 to 1.0)
    pub min_liquidity_ratio: Decimal,
    /// Factor to reduce size when book is thin (0.0 to 1.0)
    pub size_reduction_factor: Decimal,
}

impl Default for DepthAnalyzerConfig {
    fn default() -> Self {
        Self {
            max_market_impact_bps: dec!(50),  // 50 bps = 0.5%
            min_liquidity_ratio: dec!(0.1),   // Max 10% of available depth
            size_reduction_factor: dec!(0.8), // Reduce by 20% if thin
        }
    }
}

/// Recommendation for order sizing based on depth analysis
#[derive(Debug, Clone)]
pub struct OrderSizeRecommendation {
    /// Recommended order size
    pub recommended_size: Decimal,
    /// Maximum safe size that won't exceed impact limits
    pub max_safe_size: Decimal,
    /// Estimated average fill price for recommended size
    pub estimated_avg_price: Decimal,
    /// Estimated price impact in basis points
    pub estimated_impact_bps: Decimal,
    /// Liquidity score from 0.0 (illiquid) to 1.0 (very liquid)
    pub liquidity_score: f64,
}

/// Price impact estimation result
#[derive(Debug, Clone)]
pub struct PriceImpact {
    /// Impact in basis points
    pub impact_bps: Decimal,
    /// Average fill price
    pub avg_fill_price: Decimal,
    /// Number of price levels consumed
    pub levels_consumed: usize,
}

/// Analyzes orderbook depth for optimal order sizing
pub struct DepthAnalyzer {
    config: DepthAnalyzerConfig,
}

impl DepthAnalyzer {
    /// Create a new depth analyzer with the given configuration
    pub fn new(config: DepthAnalyzerConfig) -> Self {
        Self { config }
    }

    /// Create a depth analyzer with default configuration
    pub fn with_defaults() -> Self {
        Self::new(DepthAnalyzerConfig::default())
    }

    /// Analyze orderbook depth and return recommended order size
    ///
    /// # Arguments
    /// * `orderbook` - Current orderbook snapshot
    /// * `side` - Side of the order (Buy or Sell)
    /// * `target_price` - Target price for the order
    /// * `max_size` - Maximum desired order size
    #[instrument(skip(self, orderbook), fields(token_id = %orderbook.token_id, side = ?side))]
    pub fn calculate_optimal_size(
        &self,
        orderbook: &Orderbook,
        side: Side,
        target_price: Decimal,
        max_size: Decimal,
    ) -> OrderSizeRecommendation {
        // Get relevant side of the book
        let levels = match side {
            Side::Buy => &orderbook.asks,  // Buying takes from asks
            Side::Sell => &orderbook.bids, // Selling takes from bids
        };

        if levels.is_empty() {
            debug!("Empty orderbook side, returning zero size");
            return OrderSizeRecommendation {
                recommended_size: Decimal::ZERO,
                max_safe_size: Decimal::ZERO,
                estimated_avg_price: target_price,
                estimated_impact_bps: Decimal::ZERO,
                liquidity_score: 0.0,
            };
        }

        // Calculate total available liquidity
        let total_liquidity = self.liquidity_to_price(orderbook, side, target_price);

        // Calculate max safe size based on liquidity ratio
        let max_from_liquidity = total_liquidity * self.config.min_liquidity_ratio;

        // Find max size that stays within impact limits
        let max_from_impact =
            self.find_max_size_for_impact(orderbook, side, self.config.max_market_impact_bps);

        // Take the minimum of all constraints
        let max_safe_size = max_size.min(max_from_liquidity).min(max_from_impact);

        // Calculate recommended size (apply reduction factor if book is thin)
        let liquidity_score = self.calculate_liquidity_score(orderbook, side, max_size);
        let recommended_size = if liquidity_score < 0.5 {
            max_safe_size * self.config.size_reduction_factor
        } else {
            max_safe_size
        };

        // Estimate impact for recommended size
        let impact = self.estimate_price_impact(orderbook, side, recommended_size);

        debug!(
            recommended_size = %recommended_size,
            max_safe_size = %max_safe_size,
            liquidity_score = liquidity_score,
            impact_bps = %impact.impact_bps,
            "Calculated optimal order size"
        );

        OrderSizeRecommendation {
            recommended_size,
            max_safe_size,
            estimated_avg_price: impact.avg_fill_price,
            estimated_impact_bps: impact.impact_bps,
            liquidity_score,
        }
    }

    /// Calculate available liquidity up to a price level
    ///
    /// For buys, sums ask liquidity up to the limit price.
    /// For sells, sums bid liquidity down to the limit price.
    pub fn liquidity_to_price(
        &self,
        orderbook: &Orderbook,
        side: Side,
        limit_price: Decimal,
    ) -> Decimal {
        let levels = match side {
            Side::Buy => &orderbook.asks,
            Side::Sell => &orderbook.bids,
        };

        levels
            .iter()
            .filter(|level| match side {
                Side::Buy => level.price <= limit_price,
                Side::Sell => level.price >= limit_price,
            })
            .map(|level| level.size)
            .sum()
    }

    /// Estimate price impact for a given order size
    ///
    /// Returns the average fill price and impact in basis points
    pub fn estimate_price_impact(
        &self,
        orderbook: &Orderbook,
        side: Side,
        size: Decimal,
    ) -> PriceImpact {
        let levels = match side {
            Side::Buy => &orderbook.asks,
            Side::Sell => &orderbook.bids,
        };

        if levels.is_empty() || size.is_zero() {
            return PriceImpact {
                impact_bps: Decimal::ZERO,
                avg_fill_price: Decimal::ZERO,
                levels_consumed: 0,
            };
        }

        let reference_price = levels[0].price;
        let mut remaining = size;
        let mut total_cost = Decimal::ZERO;
        let mut levels_consumed = 0;

        for level in levels {
            if remaining.is_zero() {
                break;
            }

            let fill_size = remaining.min(level.size);
            total_cost += fill_size * level.price;
            remaining -= fill_size;
            levels_consumed += 1;
        }

        let filled_size = size - remaining;
        if filled_size.is_zero() {
            return PriceImpact {
                impact_bps: Decimal::ZERO,
                avg_fill_price: reference_price,
                levels_consumed: 0,
            };
        }

        let avg_fill_price = total_cost / filled_size;

        // Calculate impact in basis points
        // For buys: (avg_price - best_ask) / best_ask * 10000
        // For sells: (best_bid - avg_price) / best_bid * 10000
        let impact_bps = match side {
            Side::Buy => ((avg_fill_price - reference_price) / reference_price) * dec!(10000),
            Side::Sell => ((reference_price - avg_fill_price) / reference_price) * dec!(10000),
        };

        PriceImpact {
            impact_bps: impact_bps.max(Decimal::ZERO),
            avg_fill_price,
            levels_consumed,
        }
    }

    /// Find the maximum order size that stays within the impact limit
    fn find_max_size_for_impact(
        &self,
        orderbook: &Orderbook,
        side: Side,
        max_impact_bps: Decimal,
    ) -> Decimal {
        let levels = match side {
            Side::Buy => &orderbook.asks,
            Side::Sell => &orderbook.bids,
        };

        if levels.is_empty() {
            return Decimal::ZERO;
        }

        let reference_price = levels[0].price;
        let mut cumulative_size = Decimal::ZERO;
        let mut cumulative_cost = Decimal::ZERO;

        for level in levels {
            // Calculate what would happen if we filled through this level
            let new_cumulative_cost = cumulative_cost + (level.size * level.price);
            let new_cumulative_size = cumulative_size + level.size;
            let avg_price = new_cumulative_cost / new_cumulative_size;

            let impact = match side {
                Side::Buy => ((avg_price - reference_price) / reference_price) * dec!(10000),
                Side::Sell => ((reference_price - avg_price) / reference_price) * dec!(10000),
            };

            if impact > max_impact_bps {
                // Binary search within this level to find exact size
                return self.binary_search_size_in_level(
                    reference_price,
                    cumulative_size,
                    cumulative_cost,
                    level.price,
                    level.size,
                    max_impact_bps,
                    side,
                );
            }

            cumulative_size = new_cumulative_size;
            cumulative_cost = new_cumulative_cost;
        }

        // All levels fit within impact limit
        cumulative_size
    }

    /// Binary search to find exact size within a level that hits impact limit
    #[allow(clippy::too_many_arguments)]
    fn binary_search_size_in_level(
        &self,
        reference_price: Decimal,
        base_size: Decimal,
        base_cost: Decimal,
        level_price: Decimal,
        level_size: Decimal,
        target_impact_bps: Decimal,
        side: Side,
    ) -> Decimal {
        let mut low = Decimal::ZERO;
        let mut high = level_size;
        let tolerance = dec!(0.01); // 0.01 contract tolerance

        while high - low > tolerance {
            let mid = (low + high) / dec!(2);
            let total_size = base_size + mid;
            let total_cost = base_cost + (mid * level_price);
            let avg_price = total_cost / total_size;

            let impact = match side {
                Side::Buy => ((avg_price - reference_price) / reference_price) * dec!(10000),
                Side::Sell => ((reference_price - avg_price) / reference_price) * dec!(10000),
            };

            if impact > target_impact_bps {
                high = mid;
            } else {
                low = mid;
            }
        }

        base_size + low
    }

    /// Calculate a liquidity score from 0.0 to 1.0
    ///
    /// Based on how much depth is available relative to the order size
    fn calculate_liquidity_score(
        &self,
        orderbook: &Orderbook,
        side: Side,
        order_size: Decimal,
    ) -> f64 {
        let levels = match side {
            Side::Buy => &orderbook.asks,
            Side::Sell => &orderbook.bids,
        };

        if levels.is_empty() || order_size.is_zero() {
            return 0.0;
        }

        // Sum total available liquidity within a reasonable price range
        // Use 5% from best price as "reasonable"
        let best_price = levels[0].price;
        let limit_price = match side {
            Side::Buy => best_price * dec!(1.05),
            Side::Sell => best_price * dec!(0.95),
        };

        let available_liquidity = self.liquidity_to_price(orderbook, side, limit_price);

        // Score is ratio of available liquidity to order size, capped at 1.0
        let ratio = available_liquidity / order_size;
        let score = ratio.to_string().parse::<f64>().unwrap_or(0.0);
        score.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use polysniper_core::PriceLevel;

    fn create_test_orderbook() -> Orderbook {
        Orderbook {
            token_id: "test_token".to_string(),
            market_id: "test_market".to_string(),
            bids: vec![
                PriceLevel {
                    price: dec!(0.50),
                    size: dec!(100),
                },
                PriceLevel {
                    price: dec!(0.49),
                    size: dec!(200),
                },
                PriceLevel {
                    price: dec!(0.48),
                    size: dec!(300),
                },
            ],
            asks: vec![
                PriceLevel {
                    price: dec!(0.51),
                    size: dec!(100),
                },
                PriceLevel {
                    price: dec!(0.52),
                    size: dec!(200),
                },
                PriceLevel {
                    price: dec!(0.53),
                    size: dec!(300),
                },
            ],
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_liquidity_to_price_buy() {
        let analyzer = DepthAnalyzer::with_defaults();
        let orderbook = create_test_orderbook();

        // Buying up to 0.52 should include first two ask levels
        let liquidity = analyzer.liquidity_to_price(&orderbook, Side::Buy, dec!(0.52));
        assert_eq!(liquidity, dec!(300)); // 100 + 200

        // Buying up to 0.55 should include all ask levels
        let liquidity = analyzer.liquidity_to_price(&orderbook, Side::Buy, dec!(0.55));
        assert_eq!(liquidity, dec!(600)); // 100 + 200 + 300
    }

    #[test]
    fn test_liquidity_to_price_sell() {
        let analyzer = DepthAnalyzer::with_defaults();
        let orderbook = create_test_orderbook();

        // Selling down to 0.49 should include first two bid levels
        let liquidity = analyzer.liquidity_to_price(&orderbook, Side::Sell, dec!(0.49));
        assert_eq!(liquidity, dec!(300)); // 100 + 200

        // Selling down to 0.45 should include all bid levels
        let liquidity = analyzer.liquidity_to_price(&orderbook, Side::Sell, dec!(0.45));
        assert_eq!(liquidity, dec!(600)); // 100 + 200 + 300
    }

    #[test]
    fn test_estimate_price_impact_no_impact() {
        let analyzer = DepthAnalyzer::with_defaults();
        let orderbook = create_test_orderbook();

        // Small order that only fills at best level
        let impact = analyzer.estimate_price_impact(&orderbook, Side::Buy, dec!(50));
        assert_eq!(impact.avg_fill_price, dec!(0.51));
        assert_eq!(impact.impact_bps, Decimal::ZERO);
        assert_eq!(impact.levels_consumed, 1);
    }

    #[test]
    fn test_estimate_price_impact_multiple_levels() {
        let analyzer = DepthAnalyzer::with_defaults();
        let orderbook = create_test_orderbook();

        // Order that spans multiple levels: 100 @ 0.51 + 50 @ 0.52
        let impact = analyzer.estimate_price_impact(&orderbook, Side::Buy, dec!(150));

        // Expected avg price: (100 * 0.51 + 50 * 0.52) / 150 = 77 / 150 = 0.513333...
        let expected_avg = (dec!(100) * dec!(0.51) + dec!(50) * dec!(0.52)) / dec!(150);
        assert_eq!(impact.avg_fill_price, expected_avg);
        assert_eq!(impact.levels_consumed, 2);

        // Impact should be positive
        assert!(impact.impact_bps > Decimal::ZERO);
    }

    #[test]
    fn test_calculate_optimal_size_liquid_book() {
        let analyzer = DepthAnalyzer::with_defaults();
        let orderbook = create_test_orderbook();

        let recommendation = analyzer.calculate_optimal_size(
            &orderbook,
            Side::Buy,
            dec!(0.55),
            dec!(50), // Small order relative to book
        );

        // Should recommend close to requested size for liquid book
        assert!(recommendation.recommended_size > Decimal::ZERO);
        assert!(recommendation.recommended_size <= dec!(50));
        assert!(recommendation.liquidity_score > 0.5);
    }

    #[test]
    fn test_calculate_optimal_size_thin_book() {
        let analyzer = DepthAnalyzer::with_defaults();
        let orderbook = create_test_orderbook();

        let recommendation = analyzer.calculate_optimal_size(
            &orderbook,
            Side::Buy,
            dec!(0.55),
            dec!(5000), // Large order relative to book
        );

        // Should recommend much less than requested for thin book
        assert!(recommendation.recommended_size < dec!(5000));
        assert!(recommendation.liquidity_score < 0.5);
    }

    #[test]
    fn test_empty_orderbook() {
        let analyzer = DepthAnalyzer::with_defaults();
        let orderbook = Orderbook {
            token_id: "test_token".to_string(),
            market_id: "test_market".to_string(),
            bids: vec![],
            asks: vec![],
            timestamp: Utc::now(),
        };

        let recommendation =
            analyzer.calculate_optimal_size(&orderbook, Side::Buy, dec!(0.50), dec!(100));

        assert_eq!(recommendation.recommended_size, Decimal::ZERO);
        assert_eq!(recommendation.max_safe_size, Decimal::ZERO);
        assert_eq!(recommendation.liquidity_score, 0.0);
    }

    #[test]
    fn test_single_level_orderbook() {
        let analyzer = DepthAnalyzer::with_defaults();
        let orderbook = Orderbook {
            token_id: "test_token".to_string(),
            market_id: "test_market".to_string(),
            bids: vec![PriceLevel {
                price: dec!(0.50),
                size: dec!(100),
            }],
            asks: vec![PriceLevel {
                price: dec!(0.51),
                size: dec!(100),
            }],
            timestamp: Utc::now(),
        };

        let recommendation =
            analyzer.calculate_optimal_size(&orderbook, Side::Buy, dec!(0.55), dec!(50));

        assert!(recommendation.recommended_size > Decimal::ZERO);
        assert!(recommendation.recommended_size <= dec!(50));
    }

    #[test]
    fn test_impact_within_tolerance() {
        let analyzer = DepthAnalyzer::new(DepthAnalyzerConfig {
            max_market_impact_bps: dec!(100), // 1% impact limit
            min_liquidity_ratio: dec!(0.5),
            size_reduction_factor: dec!(0.8),
        });

        let orderbook = create_test_orderbook();

        let recommendation =
            analyzer.calculate_optimal_size(&orderbook, Side::Buy, dec!(0.55), dec!(100));

        // Impact should be within configured limit
        assert!(recommendation.estimated_impact_bps <= dec!(100));
    }
}
