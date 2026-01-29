//! P&L calculation logic
//!
//! Provides realized and unrealized P&L calculations using FIFO and average cost basis methods.

use polysniper_core::Side;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Cost basis method for P&L calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CostBasisMethod {
    /// First-in, first-out: oldest lots are sold first
    Fifo,
    /// Average cost: all shares have the same cost basis
    #[default]
    Average,
}

/// Represents a single lot (purchase) of shares
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lot {
    /// Size of this lot
    pub size: Decimal,
    /// Price paid per share
    pub price: Decimal,
    /// Fees paid on acquisition
    pub fees: Decimal,
}

impl Lot {
    pub fn new(size: Decimal, price: Decimal, fees: Decimal) -> Self {
        Self { size, price, fees }
    }

    /// Total cost of this lot including fees
    pub fn total_cost(&self) -> Decimal {
        (self.size * self.price) + self.fees
    }
}

/// Tracks position for P&L calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionTracker {
    /// Current lots held (for FIFO)
    lots: Vec<Lot>,
    /// Total size held
    total_size: Decimal,
    /// Total cost (for average cost)
    total_cost: Decimal,
    /// Accumulated realized P&L
    realized_pnl: Decimal,
    /// Accumulated fees
    total_fees: Decimal,
    /// Cost basis method
    method: CostBasisMethod,
}

impl Default for PositionTracker {
    fn default() -> Self {
        Self::new(CostBasisMethod::Average)
    }
}

impl PositionTracker {
    pub fn new(method: CostBasisMethod) -> Self {
        Self {
            lots: Vec::new(),
            total_size: Decimal::ZERO,
            total_cost: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            total_fees: Decimal::ZERO,
            method,
        }
    }

    /// Get current position size
    pub fn size(&self) -> Decimal {
        self.total_size
    }

    /// Get average entry price
    pub fn avg_price(&self) -> Decimal {
        if self.total_size.is_zero() {
            Decimal::ZERO
        } else {
            self.total_cost / self.total_size
        }
    }

    /// Get total realized P&L
    pub fn realized_pnl(&self) -> Decimal {
        self.realized_pnl
    }

    /// Get total fees paid
    pub fn total_fees(&self) -> Decimal {
        self.total_fees
    }

    /// Calculate unrealized P&L at given price
    pub fn unrealized_pnl(&self, current_price: Decimal) -> Decimal {
        if self.total_size.is_zero() {
            return Decimal::ZERO;
        }
        (current_price * self.total_size) - self.total_cost
    }

    /// Add to position (buy)
    pub fn add(&mut self, size: Decimal, price: Decimal, fees: Decimal) {
        let lot = Lot::new(size, price, fees);
        let cost = lot.total_cost();

        self.lots.push(lot);
        self.total_size += size;
        self.total_cost += cost;
        self.total_fees += fees;
    }

    /// Reduce position (sell) and return realized P&L
    pub fn reduce(&mut self, size: Decimal, exit_price: Decimal, fees: Decimal) -> Decimal {
        if size > self.total_size {
            // Can't sell more than we have
            return Decimal::ZERO;
        }

        self.total_fees += fees;

        let pnl = match self.method {
            CostBasisMethod::Fifo => self.reduce_fifo(size, exit_price, fees),
            CostBasisMethod::Average => self.reduce_average(size, exit_price, fees),
        };

        self.realized_pnl += pnl;
        pnl
    }

    /// Reduce using FIFO method
    fn reduce_fifo(&mut self, mut size: Decimal, exit_price: Decimal, fees: Decimal) -> Decimal {
        let mut realized = Decimal::ZERO;

        while size > Decimal::ZERO && !self.lots.is_empty() {
            let lot = &mut self.lots[0];

            if lot.size <= size {
                // Consume entire lot
                let lot_cost = lot.total_cost();
                let lot_size = lot.size;
                let lot_proceeds = lot_size * exit_price;

                self.total_size -= lot_size;
                self.total_cost -= lot_cost;
                size -= lot_size;

                realized += lot_proceeds - lot_cost;
                self.lots.remove(0);
            } else {
                // Partially consume lot
                let fraction = size / lot.size;
                let partial_cost = lot.total_cost() * fraction;
                let partial_proceeds = size * exit_price;

                lot.size -= size;
                lot.fees -= lot.fees * fraction;

                self.total_size -= size;
                self.total_cost -= partial_cost;

                realized += partial_proceeds - partial_cost;
                size = Decimal::ZERO;
            }
        }

        // Subtract exit fees from realized P&L
        realized -= fees;
        realized
    }

    /// Reduce using average cost method
    fn reduce_average(&mut self, size: Decimal, exit_price: Decimal, fees: Decimal) -> Decimal {
        let avg_cost = self.avg_price();
        let cost_basis = size * avg_cost;
        let proceeds = size * exit_price;

        // Adjust total cost proportionally
        let fraction = size / self.total_size;
        let removed_cost = self.total_cost * fraction;

        self.total_size -= size;
        self.total_cost -= removed_cost;

        // Update lots for average method (combine into single lot if needed)
        if !self.lots.is_empty() {
            self.lots.clear();
            if self.total_size > Decimal::ZERO {
                self.lots.push(Lot::new(self.total_size, avg_cost, Decimal::ZERO));
            }
        }

        proceeds - cost_basis - fees
    }

    /// Close entire position and return realized P&L
    pub fn close(&mut self, exit_price: Decimal, fees: Decimal) -> Decimal {
        self.reduce(self.total_size, exit_price, fees)
    }

    /// Check if position is empty
    pub fn is_empty(&self) -> bool {
        self.total_size.is_zero()
    }
}

/// Result of a P&L calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnlResult {
    /// Realized P&L from closing/reducing position
    pub realized_pnl: Decimal,
    /// Whether this trade was profitable
    pub is_win: bool,
    /// Fees paid on this trade
    pub fees: Decimal,
    /// New position size after trade
    pub new_size: Decimal,
    /// New average price after trade
    pub new_avg_price: Decimal,
}

/// Calculate P&L for a trade given position state
pub fn calculate_trade_pnl(
    current_size: Decimal,
    current_avg_price: Decimal,
    trade_side: Side,
    trade_price: Decimal,
    trade_size: Decimal,
    fees: Decimal,
    method: CostBasisMethod,
) -> PnlResult {
    let mut tracker = PositionTracker::new(method);

    // Reconstruct position
    if current_size > Decimal::ZERO {
        tracker.add(current_size, current_avg_price, Decimal::ZERO);
    }

    let (realized_pnl, new_size, new_avg_price) = match trade_side {
        Side::Buy => {
            // Adding to position
            tracker.add(trade_size, trade_price, fees);
            (Decimal::ZERO, tracker.size(), tracker.avg_price())
        }
        Side::Sell => {
            // Reducing or closing position
            let pnl = tracker.reduce(trade_size, trade_price, fees);
            (pnl, tracker.size(), tracker.avg_price())
        }
    };

    PnlResult {
        realized_pnl,
        is_win: realized_pnl > Decimal::ZERO,
        fees,
        new_size,
        new_avg_price,
    }
}

/// Calculate unrealized P&L for a position
pub fn calculate_unrealized_pnl(
    size: Decimal,
    avg_price: Decimal,
    current_price: Decimal,
) -> Decimal {
    if size.is_zero() {
        return Decimal::ZERO;
    }
    (current_price - avg_price) * size
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_position_tracker_add() {
        let mut tracker = PositionTracker::new(CostBasisMethod::Average);
        tracker.add(dec!(100), dec!(0.50), dec!(0.10));

        assert_eq!(tracker.size(), dec!(100));
        assert_eq!(tracker.avg_price(), dec!(0.501)); // (100 * 0.50 + 0.10) / 100
    }

    #[test]
    fn test_position_tracker_average_cost() {
        let mut tracker = PositionTracker::new(CostBasisMethod::Average);

        // Buy 100 @ 0.50
        tracker.add(dec!(100), dec!(0.50), Decimal::ZERO);
        assert_eq!(tracker.avg_price(), dec!(0.50));

        // Buy 100 @ 0.60
        tracker.add(dec!(100), dec!(0.60), Decimal::ZERO);
        assert_eq!(tracker.size(), dec!(200));
        assert_eq!(tracker.avg_price(), dec!(0.55)); // (50 + 60) / 200

        // Sell 100 @ 0.70
        let pnl = tracker.reduce(dec!(100), dec!(0.70), Decimal::ZERO);
        assert_eq!(pnl, dec!(15)); // (0.70 - 0.55) * 100
        assert_eq!(tracker.size(), dec!(100));
    }

    #[test]
    fn test_position_tracker_fifo() {
        let mut tracker = PositionTracker::new(CostBasisMethod::Fifo);

        // Buy 100 @ 0.50
        tracker.add(dec!(100), dec!(0.50), Decimal::ZERO);

        // Buy 100 @ 0.60
        tracker.add(dec!(100), dec!(0.60), Decimal::ZERO);

        // Sell 100 @ 0.70 (sells the first lot @ 0.50)
        let pnl = tracker.reduce(dec!(100), dec!(0.70), Decimal::ZERO);
        assert_eq!(pnl, dec!(20)); // (0.70 - 0.50) * 100
        assert_eq!(tracker.size(), dec!(100));

        // Remaining lot should be @ 0.60
        assert_eq!(tracker.avg_price(), dec!(0.60));
    }

    #[test]
    fn test_position_tracker_with_fees() {
        let mut tracker = PositionTracker::new(CostBasisMethod::Average);

        // Buy 100 @ 0.50 with $1 fee
        tracker.add(dec!(100), dec!(0.50), dec!(1));
        assert_eq!(tracker.total_fees(), dec!(1));

        // Sell 100 @ 0.60 with $1 fee
        let pnl = tracker.reduce(dec!(100), dec!(0.60), dec!(1));
        // Gross: (0.60 - 0.51) * 100 = 9
        // Net: 9 - 1 (exit fee) = 8
        assert_eq!(pnl, dec!(8));
        assert_eq!(tracker.total_fees(), dec!(2));
    }

    #[test]
    fn test_unrealized_pnl() {
        let pnl = calculate_unrealized_pnl(dec!(100), dec!(0.50), dec!(0.60));
        assert_eq!(pnl, dec!(10)); // (0.60 - 0.50) * 100

        let pnl = calculate_unrealized_pnl(dec!(100), dec!(0.60), dec!(0.50));
        assert_eq!(pnl, dec!(-10)); // (0.50 - 0.60) * 100
    }

    #[test]
    fn test_calculate_trade_pnl_buy() {
        let result = calculate_trade_pnl(
            dec!(100),     // current size
            dec!(0.50),    // current avg price
            Side::Buy,     // buying more
            dec!(0.60),    // buy price
            dec!(100),     // buy size
            dec!(1),       // fees
            CostBasisMethod::Average,
        );

        assert_eq!(result.realized_pnl, Decimal::ZERO); // No P&L on buys
        assert_eq!(result.new_size, dec!(200));
        // Avg price: (50 + 60 + 1) / 200 = 0.555
        assert_eq!(result.new_avg_price, dec!(0.555));
    }

    #[test]
    fn test_calculate_trade_pnl_sell() {
        let result = calculate_trade_pnl(
            dec!(100),     // current size
            dec!(0.50),    // current avg price
            Side::Sell,    // selling
            dec!(0.70),    // sell price
            dec!(50),      // sell size
            dec!(1),       // fees
            CostBasisMethod::Average,
        );

        // Realized: (0.70 - 0.50) * 50 - 1 = 9
        assert_eq!(result.realized_pnl, dec!(9));
        assert!(result.is_win);
        assert_eq!(result.new_size, dec!(50));
    }

    #[test]
    fn test_close_position() {
        let mut tracker = PositionTracker::new(CostBasisMethod::Average);
        tracker.add(dec!(100), dec!(0.50), Decimal::ZERO);

        let pnl = tracker.close(dec!(0.70), dec!(1));
        assert_eq!(pnl, dec!(19)); // (0.70 - 0.50) * 100 - 1
        assert!(tracker.is_empty());
    }
}
