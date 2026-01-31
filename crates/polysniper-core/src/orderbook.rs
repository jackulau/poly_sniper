//! High-performance orderbook implementation for HFT workloads
//!
//! This module provides a memory-efficient, zero-copy orderbook structure
//! that minimizes allocations during high-frequency updates.
//!
//! Key features:
//! - Pre-allocated storage to avoid runtime allocations
//! - Binary search for O(log n) price lookups
//! - Zero-copy access to top N levels via slices
//! - Copy-on-write semantics for thread-safe sharing
//! - Backward compatibility with legacy `Orderbook` type

use crate::types::{MarketId, Orderbook, PriceLevel, TokenId};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::sync::Arc;

/// Default capacity for bid/ask sides (typical depth is 50-100 levels)
pub const DEFAULT_ORDERBOOK_CAPACITY: usize = 100;

/// Packed price level for efficient memory layout
///
/// Uses `i64` instead of `Decimal` for hot-path operations:
/// - Price stored as basis points (0.01 = 100, 0.50 = 5000, 1.00 = 10000)
/// - Size stored in smallest units (typically 1 contract = 1_000_000)
///
/// This avoids `Decimal` overhead in tight loops while maintaining precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)] // Predictable memory layout for cache efficiency
pub struct PackedPriceLevel {
    /// Price in basis points (1 bp = 0.0001, so 0.50 = 5000)
    pub price: i64,
    /// Size in micro-units (1 contract = 1_000_000)
    pub size: i64,
}

impl PackedPriceLevel {
    /// Price multiplier: 10000 basis points = 1.0
    pub const PRICE_MULTIPLIER: i64 = 10_000;
    /// Size multiplier: 1_000_000 micro-units = 1.0 contract
    pub const SIZE_MULTIPLIER: i64 = 1_000_000;

    /// Create a new packed price level
    #[inline]
    pub const fn new(price: i64, size: i64) -> Self {
        Self { price, size }
    }

    /// Create from Decimal values
    #[inline]
    pub fn from_decimal(price: Decimal, size: Decimal) -> Self {
        Self {
            price: decimal_to_i64(price, Self::PRICE_MULTIPLIER),
            size: decimal_to_i64(size, Self::SIZE_MULTIPLIER),
        }
    }

    /// Convert price back to Decimal
    #[inline]
    pub fn price_decimal(&self) -> Decimal {
        i64_to_decimal(self.price, Self::PRICE_MULTIPLIER)
    }

    /// Convert size back to Decimal
    #[inline]
    pub fn size_decimal(&self) -> Decimal {
        i64_to_decimal(self.size, Self::SIZE_MULTIPLIER)
    }

    /// Convert to legacy PriceLevel
    #[inline]
    pub fn to_legacy(&self) -> PriceLevel {
        PriceLevel {
            price: self.price_decimal(),
            size: self.size_decimal(),
        }
    }

    /// Check if this level is empty (zero size)
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }
}

/// Delta update for orderbook
#[derive(Debug, Clone)]
pub struct OrderbookDelta {
    /// Bid levels to update (price, size) - size of 0 means remove
    pub bids: Vec<PackedPriceLevel>,
    /// Ask levels to update (price, size) - size of 0 means remove
    pub asks: Vec<PackedPriceLevel>,
    /// Timestamp of the delta
    pub timestamp: i64,
}

impl OrderbookDelta {
    /// Create an empty delta
    pub fn new(timestamp: i64) -> Self {
        Self {
            bids: Vec::new(),
            asks: Vec::new(),
            timestamp,
        }
    }

    /// Add a bid update
    pub fn add_bid(&mut self, price: i64, size: i64) {
        self.bids.push(PackedPriceLevel::new(price, size));
    }

    /// Add an ask update
    pub fn add_ask(&mut self, price: i64, size: i64) {
        self.asks.push(PackedPriceLevel::new(price, size));
    }
}

/// Memory-efficient orderbook with pre-allocated storage
///
/// Designed for high-frequency updates with minimal allocations:
/// - Fixed-capacity arrays avoid reallocation
/// - Sorted storage enables binary search lookups
/// - Zero-copy slice access for top-N queries
#[derive(Clone)]
pub struct FastOrderbook {
    /// Token ID for this orderbook
    token_id: TokenId,
    /// Market ID
    market_id: MarketId,
    /// Pre-allocated bid storage (sorted descending by price)
    bids: Box<[PackedPriceLevel]>,
    /// Pre-allocated ask storage (sorted ascending by price)
    asks: Box<[PackedPriceLevel]>,
    /// Number of active bid levels (valid entries in bids[0..bid_count])
    bid_count: u16,
    /// Number of active ask levels (valid entries in asks[0..ask_count])
    ask_count: u16,
    /// Timestamp in milliseconds since epoch
    timestamp: i64,
}

impl FastOrderbook {
    /// Create a new orderbook with the given capacity
    pub fn with_capacity(
        token_id: TokenId,
        market_id: MarketId,
        bid_capacity: usize,
        ask_capacity: usize,
    ) -> Self {
        Self {
            token_id,
            market_id,
            bids: vec![PackedPriceLevel::new(0, 0); bid_capacity].into_boxed_slice(),
            asks: vec![PackedPriceLevel::new(0, 0); ask_capacity].into_boxed_slice(),
            bid_count: 0,
            ask_count: 0,
            timestamp: 0,
        }
    }

    /// Create with default capacity (100 levels per side)
    pub fn new(token_id: TokenId, market_id: MarketId) -> Self {
        Self::with_capacity(
            token_id,
            market_id,
            DEFAULT_ORDERBOOK_CAPACITY,
            DEFAULT_ORDERBOOK_CAPACITY,
        )
    }

    /// Create from a legacy Orderbook
    pub fn from_legacy(orderbook: &Orderbook) -> Self {
        let bid_capacity = orderbook.bids.len().max(DEFAULT_ORDERBOOK_CAPACITY);
        let ask_capacity = orderbook.asks.len().max(DEFAULT_ORDERBOOK_CAPACITY);

        let mut fast = Self::with_capacity(
            orderbook.token_id.clone(),
            orderbook.market_id.clone(),
            bid_capacity,
            ask_capacity,
        );

        // Copy bids
        for (i, level) in orderbook.bids.iter().enumerate() {
            if i >= fast.bids.len() {
                break;
            }
            fast.bids[i] = PackedPriceLevel::from_decimal(level.price, level.size);
        }
        fast.bid_count = orderbook.bids.len().min(fast.bids.len()) as u16;

        // Copy asks
        for (i, level) in orderbook.asks.iter().enumerate() {
            if i >= fast.asks.len() {
                break;
            }
            fast.asks[i] = PackedPriceLevel::from_decimal(level.price, level.size);
        }
        fast.ask_count = orderbook.asks.len().min(fast.asks.len()) as u16;

        fast.timestamp = orderbook.timestamp.timestamp_millis();

        fast
    }

    /// Get the token ID
    #[inline]
    pub fn token_id(&self) -> &TokenId {
        &self.token_id
    }

    /// Get the market ID
    #[inline]
    pub fn market_id(&self) -> &MarketId {
        &self.market_id
    }

    /// Get the timestamp
    #[inline]
    pub fn timestamp(&self) -> i64 {
        self.timestamp
    }

    /// Get timestamp as DateTime
    pub fn timestamp_datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.timestamp).unwrap_or_else(Utc::now)
    }

    /// Set the timestamp
    #[inline]
    pub fn set_timestamp(&mut self, timestamp: i64) {
        self.timestamp = timestamp;
    }

    /// Get the number of bid levels
    #[inline]
    pub fn bid_count(&self) -> usize {
        self.bid_count as usize
    }

    /// Get the number of ask levels
    #[inline]
    pub fn ask_count(&self) -> usize {
        self.ask_count as usize
    }

    /// Check if the orderbook is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bid_count == 0 && self.ask_count == 0
    }

    /// Zero-copy access to top N bid levels
    ///
    /// Returns a slice of up to `n` best bid levels (highest prices first).
    #[inline]
    pub fn top_bids(&self, n: usize) -> &[PackedPriceLevel] {
        let count = n.min(self.bid_count as usize);
        &self.bids[..count]
    }

    /// Zero-copy access to top N ask levels
    ///
    /// Returns a slice of up to `n` best ask levels (lowest prices first).
    #[inline]
    pub fn top_asks(&self, n: usize) -> &[PackedPriceLevel] {
        let count = n.min(self.ask_count as usize);
        &self.asks[..count]
    }

    /// Get all active bid levels
    #[inline]
    pub fn bids(&self) -> &[PackedPriceLevel] {
        &self.bids[..self.bid_count as usize]
    }

    /// Get all active ask levels
    #[inline]
    pub fn asks(&self) -> &[PackedPriceLevel] {
        &self.asks[..self.ask_count as usize]
    }

    /// Get best bid level
    #[inline]
    pub fn best_bid(&self) -> Option<&PackedPriceLevel> {
        if self.bid_count > 0 {
            Some(&self.bids[0])
        } else {
            None
        }
    }

    /// Get best ask level
    #[inline]
    pub fn best_ask(&self) -> Option<&PackedPriceLevel> {
        if self.ask_count > 0 {
            Some(&self.asks[0])
        } else {
            None
        }
    }

    /// Get best bid price as Decimal
    #[inline]
    pub fn best_bid_price(&self) -> Option<Decimal> {
        self.best_bid().map(|l| l.price_decimal())
    }

    /// Get best ask price as Decimal
    #[inline]
    pub fn best_ask_price(&self) -> Option<Decimal> {
        self.best_ask().map(|l| l.price_decimal())
    }

    /// Get mid price as Decimal
    pub fn mid_price(&self) -> Option<Decimal> {
        match (self.best_bid_price(), self.best_ask_price()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / Decimal::TWO),
            (Some(bid), None) => Some(bid),
            (None, Some(ask)) => Some(ask),
            (None, None) => None,
        }
    }

    /// Get spread as Decimal
    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_bid_price(), self.best_ask_price()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Binary search for a bid at a specific price
    ///
    /// Returns the level if found, None otherwise.
    /// Time complexity: O(log n)
    pub fn bid_at_price(&self, price: i64) -> Option<&PackedPriceLevel> {
        let bids = &self.bids[..self.bid_count as usize];
        // Bids are sorted descending, so we search in reverse order
        let result = bids.binary_search_by(|level| {
            // Reverse comparison for descending order
            price.cmp(&level.price)
        });
        match result {
            Ok(idx) => Some(&bids[idx]),
            Err(_) => None,
        }
    }

    /// Binary search for an ask at a specific price
    ///
    /// Returns the level if found, None otherwise.
    /// Time complexity: O(log n)
    pub fn ask_at_price(&self, price: i64) -> Option<&PackedPriceLevel> {
        let asks = &self.asks[..self.ask_count as usize];
        // Asks are sorted ascending
        let result = asks.binary_search_by(|level| level.price.cmp(&price));
        match result {
            Ok(idx) => Some(&asks[idx]),
            Err(_) => None,
        }
    }

    /// Apply a delta update to the orderbook
    ///
    /// Returns true if successful, false if the update couldn't be applied
    /// (e.g., capacity exceeded).
    pub fn apply_delta(&mut self, delta: &OrderbookDelta) -> bool {
        // Apply bid updates
        for update in &delta.bids {
            if update.size == 0 {
                self.remove_bid(update.price);
            } else if !self.upsert_bid(*update) {
                return false;
            }
        }

        // Apply ask updates
        for update in &delta.asks {
            if update.size == 0 {
                self.remove_ask(update.price);
            } else if !self.upsert_ask(*update) {
                return false;
            }
        }

        self.timestamp = delta.timestamp;
        true
    }

    /// Insert or update a bid level
    ///
    /// Maintains descending sort order. Returns false if capacity exceeded.
    fn upsert_bid(&mut self, level: PackedPriceLevel) -> bool {
        let bids = &mut self.bids[..];
        let count = self.bid_count as usize;

        // Binary search for insertion point (descending order)
        let result = bids[..count].binary_search_by(|l| level.price.cmp(&l.price));

        match result {
            Ok(idx) => {
                // Update existing level
                bids[idx] = level;
                true
            }
            Err(idx) => {
                // Insert new level
                if count >= bids.len() {
                    return false; // Capacity exceeded
                }
                // Shift elements to make room
                bids.copy_within(idx..count, idx + 1);
                bids[idx] = level;
                self.bid_count += 1;
                true
            }
        }
    }

    /// Insert or update an ask level
    ///
    /// Maintains ascending sort order. Returns false if capacity exceeded.
    fn upsert_ask(&mut self, level: PackedPriceLevel) -> bool {
        let asks = &mut self.asks[..];
        let count = self.ask_count as usize;

        // Binary search for insertion point (ascending order)
        let result = asks[..count].binary_search_by(|l| l.price.cmp(&level.price));

        match result {
            Ok(idx) => {
                // Update existing level
                asks[idx] = level;
                true
            }
            Err(idx) => {
                // Insert new level
                if count >= asks.len() {
                    return false; // Capacity exceeded
                }
                // Shift elements to make room
                asks.copy_within(idx..count, idx + 1);
                asks[idx] = level;
                self.ask_count += 1;
                true
            }
        }
    }

    /// Remove a bid level at the given price
    fn remove_bid(&mut self, price: i64) {
        let count = self.bid_count as usize;
        let result = self.bids[..count].binary_search_by(|l| price.cmp(&l.price));

        if let Ok(idx) = result {
            // Shift elements to close the gap
            self.bids.copy_within(idx + 1..count, idx);
            self.bid_count -= 1;
        }
    }

    /// Remove an ask level at the given price
    fn remove_ask(&mut self, price: i64) {
        let count = self.ask_count as usize;
        let result = self.asks[..count].binary_search_by(|l| l.price.cmp(&price));

        if let Ok(idx) = result {
            // Shift elements to close the gap
            self.asks.copy_within(idx + 1..count, idx);
            self.ask_count -= 1;
        }
    }

    /// Clear all levels
    pub fn clear(&mut self) {
        self.bid_count = 0;
        self.ask_count = 0;
    }

    /// Set all bids from a slice (clears existing bids)
    ///
    /// Expects input to be sorted descending by price.
    pub fn set_bids(&mut self, levels: &[PackedPriceLevel]) {
        let count = levels.len().min(self.bids.len());
        self.bids[..count].copy_from_slice(&levels[..count]);
        self.bid_count = count as u16;
    }

    /// Set all asks from a slice (clears existing asks)
    ///
    /// Expects input to be sorted ascending by price.
    pub fn set_asks(&mut self, levels: &[PackedPriceLevel]) {
        let count = levels.len().min(self.asks.len());
        self.asks[..count].copy_from_slice(&levels[..count]);
        self.ask_count = count as u16;
    }

    /// Convert to legacy Orderbook format for compatibility
    pub fn to_legacy(&self) -> Orderbook {
        Orderbook {
            token_id: self.token_id.clone(),
            market_id: self.market_id.clone(),
            bids: self.bids().iter().map(|l| l.to_legacy()).collect(),
            asks: self.asks().iter().map(|l| l.to_legacy()).collect(),
            timestamp: self.timestamp_datetime(),
        }
    }
}

impl std::fmt::Debug for FastOrderbook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastOrderbook")
            .field("token_id", &self.token_id)
            .field("market_id", &self.market_id)
            .field("bid_count", &self.bid_count)
            .field("ask_count", &self.ask_count)
            .field("timestamp", &self.timestamp)
            .field("best_bid", &self.best_bid())
            .field("best_ask", &self.best_ask())
            .finish()
    }
}

/// Thread-safe orderbook wrapper with copy-on-write semantics
///
/// Enables efficient sharing across threads:
/// - Multiple readers can access the orderbook without cloning
/// - Writers get an exclusive mutable copy
/// - Atomic reference counting for memory safety
#[derive(Clone)]
pub struct SharedOrderbook {
    inner: Arc<FastOrderbook>,
}

impl SharedOrderbook {
    /// Create a new shared orderbook
    pub fn new(orderbook: FastOrderbook) -> Self {
        Self {
            inner: Arc::new(orderbook),
        }
    }

    /// Create with default capacity
    pub fn with_defaults(token_id: TokenId, market_id: MarketId) -> Self {
        Self::new(FastOrderbook::new(token_id, market_id))
    }

    /// Get read-only reference to the orderbook (no clone needed)
    #[inline]
    pub fn get(&self) -> &FastOrderbook {
        &self.inner
    }

    /// Get mutable reference with copy-on-write semantics
    ///
    /// If this is the only reference, returns the inner orderbook directly.
    /// Otherwise, clones the data to get an exclusive copy.
    #[inline]
    pub fn get_mut(&mut self) -> &mut FastOrderbook {
        Arc::make_mut(&mut self.inner)
    }

    /// Check if this is the only reference
    #[inline]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Get the reference count
    #[inline]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Replace the inner orderbook
    pub fn replace(&mut self, orderbook: FastOrderbook) {
        self.inner = Arc::new(orderbook);
    }

    /// Convert to legacy Orderbook format
    pub fn to_legacy(&self) -> Orderbook {
        self.inner.to_legacy()
    }
}

impl std::fmt::Debug for SharedOrderbook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedOrderbook")
            .field("inner", &self.inner)
            .field("ref_count", &Arc::strong_count(&self.inner))
            .finish()
    }
}

impl From<FastOrderbook> for SharedOrderbook {
    fn from(orderbook: FastOrderbook) -> Self {
        Self::new(orderbook)
    }
}

impl From<&Orderbook> for SharedOrderbook {
    fn from(orderbook: &Orderbook) -> Self {
        Self::new(FastOrderbook::from_legacy(orderbook))
    }
}

// Helper functions for Decimal <-> i64 conversion

/// Convert Decimal to i64 with the given multiplier
#[inline]
fn decimal_to_i64(value: Decimal, multiplier: i64) -> i64 {
    let scaled = value * Decimal::from(multiplier);
    let mantissa = scaled.mantissa();
    let divisor = 10i128.pow(scaled.scale());
    (mantissa / divisor) as i64
}

/// Convert i64 to Decimal with the given multiplier
#[inline]
fn i64_to_decimal(value: i64, multiplier: i64) -> Decimal {
    Decimal::from(value) / Decimal::from(multiplier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_packed_price_level_conversion() {
        let level = PackedPriceLevel::from_decimal(dec!(0.50), dec!(100.5));
        assert_eq!(level.price, 5000); // 0.50 * 10000
        assert_eq!(level.size, 100_500_000); // 100.5 * 1_000_000

        // Round-trip conversion
        assert_eq!(level.price_decimal(), dec!(0.5000));
        assert_eq!(level.size_decimal(), dec!(100.5));
    }

    #[test]
    fn test_fast_orderbook_creation() {
        let ob = FastOrderbook::new("token1".to_string(), "market1".to_string());
        assert_eq!(ob.token_id(), "token1");
        assert_eq!(ob.market_id(), "market1");
        assert_eq!(ob.bid_count(), 0);
        assert_eq!(ob.ask_count(), 0);
        assert!(ob.is_empty());
    }

    #[test]
    fn test_fast_orderbook_from_legacy() {
        let legacy = Orderbook {
            token_id: "token1".to_string(),
            market_id: "market1".to_string(),
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
                    size: dec!(150),
                },
                PriceLevel {
                    price: dec!(0.52),
                    size: dec!(250),
                },
            ],
            timestamp: Utc::now(),
        };

        let fast = FastOrderbook::from_legacy(&legacy);
        assert_eq!(fast.bid_count(), 2);
        assert_eq!(fast.ask_count(), 2);
        assert_eq!(fast.best_bid_price(), Some(dec!(0.5000)));
        assert_eq!(fast.best_ask_price(), Some(dec!(0.51)));
    }

    #[test]
    fn test_top_n_access() {
        let legacy = Orderbook {
            token_id: "token1".to_string(),
            market_id: "market1".to_string(),
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
                    size: dec!(150),
                },
            ],
            timestamp: Utc::now(),
        };

        let fast = FastOrderbook::from_legacy(&legacy);

        // Test top_bids with various N values
        assert_eq!(fast.top_bids(2).len(), 2);
        assert_eq!(fast.top_bids(5).len(), 3); // Only 3 available
        assert_eq!(fast.top_bids(0).len(), 0);

        // Verify ordering (best bid first)
        let top2 = fast.top_bids(2);
        assert_eq!(top2[0].price, 5000); // 0.50
        assert_eq!(top2[1].price, 4900); // 0.49
    }

    #[test]
    fn test_binary_search() {
        let legacy = Orderbook {
            token_id: "token1".to_string(),
            market_id: "market1".to_string(),
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
                    size: dec!(150),
                },
                PriceLevel {
                    price: dec!(0.52),
                    size: dec!(250),
                },
            ],
            timestamp: Utc::now(),
        };

        let fast = FastOrderbook::from_legacy(&legacy);

        // Find existing levels
        let bid = fast.bid_at_price(4900);
        assert!(bid.is_some());
        assert_eq!(bid.unwrap().size, 200_000_000);

        let ask = fast.ask_at_price(5200);
        assert!(ask.is_some());
        assert_eq!(ask.unwrap().size, 250_000_000);

        // Search for non-existent levels
        assert!(fast.bid_at_price(4700).is_none());
        assert!(fast.ask_at_price(5300).is_none());
    }

    #[test]
    fn test_apply_delta() {
        let mut ob = FastOrderbook::new("token1".to_string(), "market1".to_string());

        // Initial delta with some levels
        let mut delta1 = OrderbookDelta::new(1000);
        delta1.add_bid(5000, 100_000_000); // 0.50, 100
        delta1.add_bid(4900, 200_000_000); // 0.49, 200
        delta1.add_ask(5100, 150_000_000); // 0.51, 150

        assert!(ob.apply_delta(&delta1));
        assert_eq!(ob.bid_count(), 2);
        assert_eq!(ob.ask_count(), 1);

        // Update delta - modify existing and add new
        let mut delta2 = OrderbookDelta::new(2000);
        delta2.add_bid(5000, 150_000_000); // Update 0.50 to 150
        delta2.add_bid(4800, 300_000_000); // Add 0.48, 300
        delta2.add_ask(5200, 250_000_000); // Add 0.52, 250

        assert!(ob.apply_delta(&delta2));
        assert_eq!(ob.bid_count(), 3);
        assert_eq!(ob.ask_count(), 2);

        // Verify update
        let bid_50 = ob.bid_at_price(5000).unwrap();
        assert_eq!(bid_50.size, 150_000_000);

        // Remove delta
        let mut delta3 = OrderbookDelta::new(3000);
        delta3.add_bid(4900, 0); // Remove 0.49

        assert!(ob.apply_delta(&delta3));
        assert_eq!(ob.bid_count(), 2);
        assert!(ob.bid_at_price(4900).is_none());
    }

    #[test]
    fn test_to_legacy_roundtrip() {
        let original = Orderbook {
            token_id: "token1".to_string(),
            market_id: "market1".to_string(),
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
                    size: dec!(150),
                },
            ],
            timestamp: Utc::now(),
        };

        let fast = FastOrderbook::from_legacy(&original);
        let roundtrip = fast.to_legacy();

        assert_eq!(roundtrip.token_id, original.token_id);
        assert_eq!(roundtrip.market_id, original.market_id);
        assert_eq!(roundtrip.bids.len(), original.bids.len());
        assert_eq!(roundtrip.asks.len(), original.asks.len());

        // Verify values (with some precision tolerance)
        for (orig, rt) in original.bids.iter().zip(roundtrip.bids.iter()) {
            assert_eq!(orig.price, rt.price);
            assert_eq!(orig.size, rt.size);
        }
    }

    #[test]
    fn test_shared_orderbook_cow() {
        let fast = FastOrderbook::new("token1".to_string(), "market1".to_string());
        let mut shared1 = SharedOrderbook::new(fast);

        // Initially unique
        assert!(shared1.is_unique());

        // Clone creates another reference
        let shared2 = shared1.clone();
        assert!(!shared1.is_unique());
        assert_eq!(shared1.ref_count(), 2);

        // get_mut on shared1 should trigger copy
        let inner = shared1.get_mut();
        inner.set_timestamp(12345);

        // Now shared1 has its own copy
        assert!(shared1.is_unique());
        assert_eq!(shared1.get().timestamp(), 12345);

        // shared2 still has original (timestamp 0)
        assert_eq!(shared2.get().timestamp(), 0);
    }

    #[test]
    fn test_mid_price_and_spread() {
        let legacy = Orderbook {
            token_id: "token1".to_string(),
            market_id: "market1".to_string(),
            bids: vec![PriceLevel {
                price: dec!(0.50),
                size: dec!(100),
            }],
            asks: vec![PriceLevel {
                price: dec!(0.52),
                size: dec!(100),
            }],
            timestamp: Utc::now(),
        };

        let fast = FastOrderbook::from_legacy(&legacy);

        let mid = fast.mid_price().unwrap();
        // (0.50 + 0.52) / 2 = 0.51
        assert_eq!(mid, dec!(0.51));

        let spread = fast.spread().unwrap();
        // 0.52 - 0.50 = 0.02
        assert_eq!(spread, dec!(0.02));
    }

    #[test]
    fn test_empty_orderbook_accessors() {
        let ob = FastOrderbook::new("token1".to_string(), "market1".to_string());

        assert!(ob.best_bid().is_none());
        assert!(ob.best_ask().is_none());
        assert!(ob.best_bid_price().is_none());
        assert!(ob.best_ask_price().is_none());
        assert!(ob.mid_price().is_none());
        assert!(ob.spread().is_none());
        assert!(ob.bid_at_price(5000).is_none());
        assert!(ob.ask_at_price(5100).is_none());
        assert!(ob.top_bids(5).is_empty());
        assert!(ob.top_asks(5).is_empty());
    }
}
