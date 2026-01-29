//! Order builder for constructing CLOB orders

use chrono::Utc;
use polysniper_core::{Order, OrderType, Side, TradeSignal};
use rust_decimal::Decimal;
use std::sync::atomic::{AtomicU64, Ordering};

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
}

impl OrderBuilder {
    /// Create a new order builder with default settings
    pub fn new() -> Self {
        Self {
            default_order_type: OrderType::Gtc,
            tick_size: Decimal::new(1, 2),      // 0.01
            size_increment: Decimal::new(1, 0), // 1
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

    /// Build an order from a trade signal
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
}
