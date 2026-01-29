//! Order builder for constructing CLOB orders

use chrono::Utc;
use polysniper_core::{AdaptiveSizingConfig, Order, OrderType, Orderbook, Side, TradeSignal};
use rust_decimal::Decimal;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info};

use crate::depth_analyzer::{DepthAnalyzer, DepthAnalyzerConfig};

static ORDER_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique order ID
fn generate_order_id() -> String {
    let counter = ORDER_COUNTER.fetch_add(1, Ordering::SeqCst);
    let timestamp = Utc::now().timestamp_millis();
    format!("ord_{}_{}", timestamp, counter)
}

/// Order builder for creating orders from signals
pub struct OrderBuilder {
    /// Default order type
    default_order_type: OrderType,
    /// Minimum tick size for prices
    tick_size: Decimal,
    /// Minimum size increment
    size_increment: Decimal,
    /// Optional depth analyzer for adaptive sizing
    depth_analyzer: Option<DepthAnalyzer>,
    /// Whether adaptive sizing is enabled
    adaptive_sizing_enabled: bool,
}

impl OrderBuilder {
    /// Create a new order builder with default settings
    pub fn new() -> Self {
        Self {
            default_order_type: OrderType::Gtc,
            tick_size: Decimal::new(1, 2),      // 0.01
            size_increment: Decimal::new(1, 0), // 1
            depth_analyzer: None,
            adaptive_sizing_enabled: false,
        }
    }

    /// Set the default order type
    pub fn with_order_type(mut self, order_type: OrderType) -> Self {
        self.default_order_type = order_type;
        self
    }

    /// Set the tick size
    pub fn with_tick_size(mut self, tick_size: Decimal) -> Self {
        self.tick_size = tick_size;
        self
    }

    /// Set the size increment
    pub fn with_size_increment(mut self, size_increment: Decimal) -> Self {
        self.size_increment = size_increment;
        self
    }

    /// Configure adaptive sizing from config
    pub fn with_adaptive_sizing(mut self, config: &AdaptiveSizingConfig) -> Self {
        self.adaptive_sizing_enabled = config.enabled;
        if config.enabled {
            let analyzer_config = DepthAnalyzerConfig {
                max_market_impact_bps: config.max_market_impact_bps,
                min_liquidity_ratio: config.min_liquidity_ratio,
                size_reduction_factor: config.size_reduction_factor,
            };
            self.depth_analyzer = Some(DepthAnalyzer::new(analyzer_config));
            info!(
                max_impact_bps = %config.max_market_impact_bps,
                min_liquidity_ratio = %config.min_liquidity_ratio,
                "Adaptive sizing enabled"
            );
        }
        self
    }

    /// Build an order from a trade signal (without orderbook - uses fixed sizing)
    pub fn build_from_signal(&self, signal: &TradeSignal) -> Order {
        let price = self.round_price(signal.price.unwrap_or(self.calculate_market_price(signal)));
        let size = self.round_size(signal.size);

        Order {
            id: generate_order_id(),
            market_id: signal.market_id.clone(),
            token_id: signal.token_id.clone(),
            side: signal.side,
            price,
            size,
            order_type: signal.order_type,
            signal_id: signal.id.clone(),
            created_at: Utc::now(),
        }
    }

    /// Build an order from a trade signal with adaptive sizing based on orderbook depth
    ///
    /// If adaptive sizing is disabled or no orderbook is provided, falls back to fixed sizing.
    pub fn build_from_signal_with_orderbook(
        &self,
        signal: &TradeSignal,
        orderbook: Option<&Orderbook>,
    ) -> Order {
        let price = self.round_price(signal.price.unwrap_or(self.calculate_market_price(signal)));

        // Determine size: use adaptive sizing if enabled and orderbook available
        let size = match (&self.depth_analyzer, orderbook) {
            (Some(analyzer), Some(book)) if self.adaptive_sizing_enabled => {
                let recommendation =
                    analyzer.calculate_optimal_size(book, signal.side, price, signal.size);

                debug!(
                    signal_id = %signal.id,
                    requested_size = %signal.size,
                    recommended_size = %recommendation.recommended_size,
                    max_safe_size = %recommendation.max_safe_size,
                    estimated_impact_bps = %recommendation.estimated_impact_bps,
                    liquidity_score = recommendation.liquidity_score,
                    "Adaptive sizing applied"
                );

                self.round_size(recommendation.recommended_size)
            }
            _ => {
                debug!(
                    signal_id = %signal.id,
                    size = %signal.size,
                    "Using fixed sizing (adaptive disabled or no orderbook)"
                );
                self.round_size(signal.size)
            }
        };

        Order {
            id: generate_order_id(),
            market_id: signal.market_id.clone(),
            token_id: signal.token_id.clone(),
            side: signal.side,
            price,
            size,
            order_type: signal.order_type,
            signal_id: signal.id.clone(),
            created_at: Utc::now(),
        }
    }

    /// Check if adaptive sizing is enabled
    pub fn is_adaptive_sizing_enabled(&self) -> bool {
        self.adaptive_sizing_enabled
    }

    /// Get a reference to the depth analyzer, if configured
    pub fn depth_analyzer(&self) -> Option<&DepthAnalyzer> {
        self.depth_analyzer.as_ref()
    }

    /// Build a limit order
    pub fn build_limit_order(
        &self,
        market_id: String,
        token_id: String,
        side: Side,
        price: Decimal,
        size: Decimal,
        signal_id: String,
    ) -> Order {
        Order {
            id: generate_order_id(),
            market_id,
            token_id,
            side,
            price: self.round_price(price),
            size: self.round_size(size),
            order_type: OrderType::Gtc,
            signal_id,
            created_at: Utc::now(),
        }
    }

    /// Build a market order (FOK at aggressive price)
    pub fn build_market_order(
        &self,
        market_id: String,
        token_id: String,
        side: Side,
        size: Decimal,
        signal_id: String,
    ) -> Order {
        // For market orders, use aggressive price
        let price = match side {
            Side::Buy => Decimal::new(99, 2), // 0.99 (max buy price)
            Side::Sell => Decimal::new(1, 2), // 0.01 (min sell price)
        };

        Order {
            id: generate_order_id(),
            market_id,
            token_id,
            side,
            price,
            size: self.round_size(size),
            order_type: OrderType::Fok,
            signal_id,
            created_at: Utc::now(),
        }
    }

    /// Round price to tick size
    fn round_price(&self, price: Decimal) -> Decimal {
        (price / self.tick_size).round() * self.tick_size
    }

    /// Round size to size increment
    fn round_size(&self, size: Decimal) -> Decimal {
        (size / self.size_increment).floor() * self.size_increment
    }

    /// Calculate aggressive market price for immediate fills
    fn calculate_market_price(&self, signal: &TradeSignal) -> Decimal {
        match signal.side {
            Side::Buy => Decimal::new(99, 2), // 0.99
            Side::Sell => Decimal::new(1, 2), // 0.01
        }
    }
}

impl Default for OrderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polysniper_core::PriceLevel;
    use rust_decimal_macros::dec;

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
            ],
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_round_price() {
        let builder = OrderBuilder::new();

        // Standard rounding: 0.525 rounds to 0.52 (banker's rounding)
        assert_eq!(
            builder.round_price(Decimal::new(525, 3)),
            Decimal::new(52, 2)
        ); // 0.525 -> 0.52
        assert_eq!(
            builder.round_price(Decimal::new(524, 3)),
            Decimal::new(52, 2)
        ); // 0.524 -> 0.52
        assert_eq!(
            builder.round_price(Decimal::new(526, 3)),
            Decimal::new(53, 2)
        ); // 0.526 -> 0.53
    }

    #[test]
    fn test_build_market_order() {
        let builder = OrderBuilder::new();

        let order = builder.build_market_order(
            "market_1".to_string(),
            "token_1".to_string(),
            Side::Buy,
            Decimal::new(100, 0),
            "signal_1".to_string(),
        );

        assert_eq!(order.order_type, OrderType::Fok);
        assert_eq!(order.price, Decimal::new(99, 2)); // 0.99 for buy
    }

    #[test]
    fn test_adaptive_sizing_disabled_by_default() {
        let builder = OrderBuilder::new();
        assert!(!builder.is_adaptive_sizing_enabled());
        assert!(builder.depth_analyzer().is_none());
    }

    #[test]
    fn test_adaptive_sizing_enabled() {
        let config = AdaptiveSizingConfig {
            enabled: true,
            max_market_impact_bps: dec!(50),
            min_liquidity_ratio: dec!(0.1),
            size_reduction_factor: dec!(0.8),
        };

        let builder = OrderBuilder::new().with_adaptive_sizing(&config);
        assert!(builder.is_adaptive_sizing_enabled());
        assert!(builder.depth_analyzer().is_some());
    }

    #[test]
    fn test_adaptive_sizing_config_disabled() {
        let config = AdaptiveSizingConfig {
            enabled: false,
            max_market_impact_bps: dec!(50),
            min_liquidity_ratio: dec!(0.1),
            size_reduction_factor: dec!(0.8),
        };

        let builder = OrderBuilder::new().with_adaptive_sizing(&config);
        assert!(!builder.is_adaptive_sizing_enabled());
        assert!(builder.depth_analyzer().is_none());
    }

    #[test]
    fn test_build_from_signal_with_orderbook_no_orderbook() {
        use polysniper_core::{Outcome, Priority};

        let config = AdaptiveSizingConfig::default();
        let builder = OrderBuilder::new().with_adaptive_sizing(&config);

        let signal = TradeSignal {
            id: "sig_1".to_string(),
            strategy_id: "test_strategy".to_string(),
            market_id: "market_1".to_string(),
            token_id: "token_1".to_string(),
            outcome: Outcome::Yes,
            side: Side::Buy,
            price: Some(dec!(0.51)),
            size: dec!(100),
            size_usd: dec!(51),
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: Utc::now(),
            reason: "Test signal".to_string(),
            metadata: serde_json::json!({}),
        };

        // Without orderbook, should use fixed sizing
        let order = builder.build_from_signal_with_orderbook(&signal, None);
        assert_eq!(order.size, dec!(100));
    }

    #[test]
    fn test_build_from_signal_with_orderbook_adaptive() {
        use polysniper_core::{Outcome, Priority};

        let config = AdaptiveSizingConfig {
            enabled: true,
            max_market_impact_bps: dec!(50),
            min_liquidity_ratio: dec!(0.1), // Max 10% of depth
            size_reduction_factor: dec!(0.8),
        };
        let builder = OrderBuilder::new().with_adaptive_sizing(&config);

        let signal = TradeSignal {
            id: "sig_1".to_string(),
            strategy_id: "test_strategy".to_string(),
            market_id: "test_market".to_string(),
            token_id: "test_token".to_string(),
            outcome: Outcome::Yes,
            side: Side::Buy,
            price: Some(dec!(0.55)),
            size: dec!(5000), // Large order relative to book
            size_usd: dec!(2750),
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: Utc::now(),
            reason: "Test signal".to_string(),
            metadata: serde_json::json!({}),
        };

        let orderbook = create_test_orderbook();
        let order = builder.build_from_signal_with_orderbook(&signal, Some(&orderbook));

        // Should reduce size due to thin book
        assert!(order.size < dec!(5000));
        assert!(order.size > Decimal::ZERO);
    }

    #[test]
    fn test_backward_compatibility() {
        use polysniper_core::{Outcome, Priority};

        // Ensure old API still works
        let builder = OrderBuilder::new();

        let signal = TradeSignal {
            id: "sig_1".to_string(),
            strategy_id: "test_strategy".to_string(),
            market_id: "market_1".to_string(),
            token_id: "token_1".to_string(),
            outcome: Outcome::Yes,
            side: Side::Buy,
            price: Some(dec!(0.51)),
            size: dec!(100),
            size_usd: dec!(51),
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: Utc::now(),
            reason: "Test signal".to_string(),
            metadata: serde_json::json!({}),
        };

        let order = builder.build_from_signal(&signal);
        assert_eq!(order.size, dec!(100));
        assert_eq!(order.price, dec!(0.51));
    }
}
