//! Queue Position Estimator
//!
//! Tracks queue position for limit orders and estimates time-to-fill based on
//! historical fill rates and current market activity.

use chrono::{DateTime, Duration, Utc};
use polysniper_core::{Orderbook, QueueEstimatorConfig, QueuePosition, Side, TokenId};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::collections::{HashMap, VecDeque};
use tokio::sync::RwLock;
use tracing::debug;

/// Record of a single fill event
#[derive(Debug, Clone)]
pub struct FillRecord {
    /// Size of the fill
    pub size: Decimal,
    /// Price at which fill occurred
    pub price: Decimal,
    /// When the fill occurred
    pub timestamp: DateTime<Utc>,
}

/// Tracks fill rates for a single token
#[derive(Debug)]
pub struct FillRateTracker {
    /// Recent fills indexed by price level
    fills_by_price: HashMap<String, VecDeque<FillRecord>>,
    /// Total fills across all prices (for overall rate)
    all_fills: VecDeque<FillRecord>,
    /// Overall average fill rate (size per second)
    avg_fill_rate: Decimal,
    /// Last update timestamp
    last_update: DateTime<Utc>,
    /// History window duration
    history_window: Duration,
}

impl FillRateTracker {
    /// Create a new fill rate tracker
    pub fn new(history_window_secs: u64) -> Self {
        Self {
            fills_by_price: HashMap::new(),
            all_fills: VecDeque::new(),
            avg_fill_rate: Decimal::ZERO,
            last_update: Utc::now(),
            history_window: Duration::seconds(history_window_secs as i64),
        }
    }

    /// Record a new fill
    pub fn record_fill(&mut self, price: Decimal, size: Decimal) {
        let now = Utc::now();
        let record = FillRecord {
            size,
            price,
            timestamp: now,
        };

        // Store by price level (using string key for HashMap)
        let price_key = price.to_string();
        self.fills_by_price
            .entry(price_key)
            .or_default()
            .push_back(record.clone());

        // Store in all fills
        self.all_fills.push_back(record);

        // Update overall fill rate
        self.update_fill_rate();
        self.last_update = now;
    }

    /// Clean up old entries outside the history window
    pub fn cleanup(&mut self) {
        let cutoff = Utc::now() - self.history_window;

        // Clean up per-price fills
        for fills in self.fills_by_price.values_mut() {
            while fills.front().map(|f| f.timestamp < cutoff).unwrap_or(false) {
                fills.pop_front();
            }
        }

        // Remove empty price levels
        self.fills_by_price.retain(|_, fills| !fills.is_empty());

        // Clean up all fills
        while self
            .all_fills
            .front()
            .map(|f| f.timestamp < cutoff)
            .unwrap_or(false)
        {
            self.all_fills.pop_front();
        }

        // Recalculate fill rate
        self.update_fill_rate();
    }

    /// Update the average fill rate calculation
    fn update_fill_rate(&mut self) {
        if self.all_fills.is_empty() {
            self.avg_fill_rate = Decimal::ZERO;
            return;
        }

        let total_size: Decimal = self.all_fills.iter().map(|f| f.size).sum();

        // Calculate time span of fills
        if let (Some(first), Some(last)) = (self.all_fills.front(), self.all_fills.back()) {
            let duration_secs = (last.timestamp - first.timestamp).num_seconds();
            if duration_secs > 0 {
                self.avg_fill_rate =
                    total_size / Decimal::from(duration_secs);
            } else {
                // All fills at same instant - use 1 second as minimum
                self.avg_fill_rate = total_size;
            }
        }
    }

    /// Get fill rate at a specific price level (size per second)
    pub fn get_fill_rate_at_price(&self, price: Decimal) -> Option<Decimal> {
        let price_key = price.to_string();
        let fills = self.fills_by_price.get(&price_key)?;

        if fills.len() < 2 {
            return None;
        }

        let total_size: Decimal = fills.iter().map(|f| f.size).sum();
        let first = fills.front()?;
        let last = fills.back()?;
        let duration_secs = (last.timestamp - first.timestamp).num_seconds();

        if duration_secs > 0 {
            Some(total_size / Decimal::from(duration_secs))
        } else {
            Some(total_size)
        }
    }

    /// Get number of samples at a price level
    pub fn sample_count_at_price(&self, price: Decimal) -> usize {
        let price_key = price.to_string();
        self.fills_by_price
            .get(&price_key)
            .map(|fills| fills.len())
            .unwrap_or(0)
    }

    /// Get total sample count
    pub fn total_sample_count(&self) -> usize {
        self.all_fills.len()
    }

    /// Get seconds since last update
    pub fn seconds_since_last_update(&self) -> i64 {
        (Utc::now() - self.last_update).num_seconds()
    }
}

/// Queue position estimator for limit orders
pub struct QueueEstimator {
    /// Historical fill rates per token
    fill_rate_history: RwLock<HashMap<TokenId, FillRateTracker>>,
    /// Configuration
    config: QueueEstimatorConfig,
}

impl QueueEstimator {
    /// Create a new queue estimator
    pub fn new(config: QueueEstimatorConfig) -> Self {
        Self {
            fill_rate_history: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Estimate queue position for a hypothetical order
    pub async fn estimate_position(
        &self,
        token_id: &TokenId,
        orderbook: &Orderbook,
        side: Side,
        price: Decimal,
        _size: Decimal,
    ) -> QueuePosition {
        // Calculate size ahead in the queue
        let size_ahead = self.calculate_size_ahead(orderbook, side, price);

        // Estimate position (number of orders ahead - simplified as size / avg order size)
        let estimated_position = self.estimate_queue_position(size_ahead);

        // Get fill rate and calculate time-to-fill
        let fill_rate = self.get_fill_rate(token_id, price).await;
        let (estimated_time, confidence, prob_1min, prob_5min) =
            self.calculate_fill_estimates(token_id, size_ahead, fill_rate).await;

        QueuePosition {
            price_level: price,
            size_ahead,
            estimated_position,
            estimated_time_to_fill_secs: estimated_time,
            confidence,
            fill_probability_1min: prob_1min,
            fill_probability_5min: prob_5min,
        }
    }

    /// Record a fill event from the trade feed
    pub async fn record_fill(&self, token_id: &TokenId, price: Decimal, size: Decimal) {
        let mut history = self.fill_rate_history.write().await;
        let tracker = history
            .entry(token_id.clone())
            .or_insert_with(|| FillRateTracker::new(self.config.history_window_secs));
        tracker.record_fill(price, size);
        debug!(
            token_id = %token_id,
            price = %price,
            size = %size,
            "Recorded fill for queue estimation"
        );
    }

    /// Get current estimated fill rate at a price level
    pub async fn get_fill_rate(&self, token_id: &TokenId, price: Decimal) -> Option<Decimal> {
        let history = self.fill_rate_history.read().await;
        let tracker = history.get(token_id)?;

        // Try price-specific rate first, fall back to overall rate
        tracker
            .get_fill_rate_at_price(price)
            .or(Some(tracker.avg_fill_rate))
    }

    /// Clean up old history entries
    pub async fn cleanup_old_entries(&self) {
        let mut history = self.fill_rate_history.write().await;
        for tracker in history.values_mut() {
            tracker.cleanup();
        }
        // Remove empty trackers
        history.retain(|_, tracker| tracker.total_sample_count() > 0);
    }

    /// Check if estimation is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Calculate size ahead in queue at given price level
    fn calculate_size_ahead(&self, orderbook: &Orderbook, side: Side, price: Decimal) -> Decimal {
        match side {
            Side::Buy => {
                // For buy orders, count bid size at prices >= our price
                orderbook
                    .bids
                    .iter()
                    .filter(|level| level.price >= price)
                    .map(|level| level.size)
                    .sum()
            }
            Side::Sell => {
                // For sell orders, count ask size at prices <= our price
                orderbook
                    .asks
                    .iter()
                    .filter(|level| level.price <= price)
                    .map(|level| level.size)
                    .sum()
            }
        }
    }

    /// Estimate queue position from size ahead
    fn estimate_queue_position(&self, size_ahead: Decimal) -> u32 {
        // Assume average order size of 100 units (this could be made configurable)
        let avg_order_size = Decimal::from(100);
        let position = size_ahead / avg_order_size;
        position.to_u32().unwrap_or(u32::MAX)
    }

    /// Calculate fill time estimates and probabilities
    async fn calculate_fill_estimates(
        &self,
        token_id: &TokenId,
        size_ahead: Decimal,
        fill_rate: Option<Decimal>,
    ) -> (Option<f64>, f64, f64, f64) {
        let history = self.fill_rate_history.read().await;
        let tracker = history.get(token_id);

        // Check if we have enough samples
        let sample_count = tracker.map(|t| t.total_sample_count()).unwrap_or(0);
        let has_sufficient_samples = sample_count >= self.config.min_samples_for_estimate as usize;

        if !has_sufficient_samples || fill_rate.is_none() {
            return (None, 0.0, 0.0, 0.0);
        }

        let rate = fill_rate.unwrap();
        if rate.is_zero() {
            return (None, 0.0, 0.0, 0.0);
        }

        // Calculate estimated time to fill
        let time_to_fill_secs = (size_ahead / rate).to_f64().unwrap_or(f64::MAX);

        // Calculate confidence based on sample count and staleness
        let seconds_since_update = tracker
            .map(|t| t.seconds_since_last_update())
            .unwrap_or(i64::MAX);

        let base_confidence = (sample_count as f64 / 100.0).min(1.0);
        let staleness_factor = self.config.confidence_decay_factor.powi(
            (seconds_since_update / 60) as i32  // Decay per minute
        );
        let confidence = (base_confidence * staleness_factor).clamp(0.0, 1.0);

        // Calculate fill probabilities using exponential distribution
        // P(fill in t seconds) = 1 - exp(-rate * t / size_ahead)
        let rate_f64 = rate.to_f64().unwrap_or(0.0);
        let size_f64 = size_ahead.to_f64().unwrap_or(1.0).max(0.001);
        let lambda = rate_f64 / size_f64;

        let prob_1min = 1.0 - (-lambda * 60.0).exp();
        let prob_5min = 1.0 - (-lambda * 300.0).exp();

        (
            Some(time_to_fill_secs),
            confidence,
            prob_1min.clamp(0.0, 1.0),
            prob_5min.clamp(0.0, 1.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polysniper_core::PriceLevel;

    fn create_test_config() -> QueueEstimatorConfig {
        QueueEstimatorConfig {
            enabled: true,
            history_window_secs: 300,
            min_samples_for_estimate: 5,
            confidence_decay_factor: 0.95,
        }
    }

    fn create_test_orderbook() -> Orderbook {
        Orderbook {
            token_id: "test_token".to_string(),
            market_id: "test_market".to_string(),
            bids: vec![
                PriceLevel {
                    price: Decimal::new(50, 2), // 0.50
                    size: Decimal::from(1000),
                },
                PriceLevel {
                    price: Decimal::new(49, 2), // 0.49
                    size: Decimal::from(2000),
                },
                PriceLevel {
                    price: Decimal::new(48, 2), // 0.48
                    size: Decimal::from(1500),
                },
            ],
            asks: vec![
                PriceLevel {
                    price: Decimal::new(51, 2), // 0.51
                    size: Decimal::from(800),
                },
                PriceLevel {
                    price: Decimal::new(52, 2), // 0.52
                    size: Decimal::from(1200),
                },
                PriceLevel {
                    price: Decimal::new(53, 2), // 0.53
                    size: Decimal::from(1000),
                },
            ],
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_estimate_position_buy() {
        let config = create_test_config();
        let estimator = QueueEstimator::new(config);
        let orderbook = create_test_orderbook();

        // Place a buy order at 0.50 - should see 1000 size ahead (the existing 0.50 bid)
        let position = estimator
            .estimate_position(
                &"test_token".to_string(),
                &orderbook,
                Side::Buy,
                Decimal::new(50, 2),
                Decimal::from(100),
            )
            .await;

        assert_eq!(position.price_level, Decimal::new(50, 2));
        assert_eq!(position.size_ahead, Decimal::from(1000));
        // Without fill history, no time estimate
        assert!(position.estimated_time_to_fill_secs.is_none());
        assert_eq!(position.confidence, 0.0);
    }

    #[tokio::test]
    async fn test_estimate_position_sell() {
        let config = create_test_config();
        let estimator = QueueEstimator::new(config);
        let orderbook = create_test_orderbook();

        // Place a sell order at 0.51 - should see 800 size ahead (the existing 0.51 ask)
        let position = estimator
            .estimate_position(
                &"test_token".to_string(),
                &orderbook,
                Side::Sell,
                Decimal::new(51, 2),
                Decimal::from(100),
            )
            .await;

        assert_eq!(position.price_level, Decimal::new(51, 2));
        assert_eq!(position.size_ahead, Decimal::from(800));
    }

    #[tokio::test]
    async fn test_record_fill_and_get_rate() {
        let config = create_test_config();
        let estimator = QueueEstimator::new(config);
        let token_id = "test_token".to_string();
        let price = Decimal::new(50, 2);

        // Record some fills
        for _ in 0..10 {
            estimator.record_fill(&token_id, price, Decimal::from(100)).await;
        }

        // Should now have a fill rate
        let rate = estimator.get_fill_rate(&token_id, price).await;
        assert!(rate.is_some());
        assert!(rate.unwrap() > Decimal::ZERO);
    }

    #[tokio::test]
    async fn test_fill_rate_tracker_cleanup() {
        let mut tracker = FillRateTracker::new(1); // 1 second window for testing

        // Record a fill
        tracker.record_fill(Decimal::new(50, 2), Decimal::from(100));
        assert_eq!(tracker.total_sample_count(), 1);

        // Wait and cleanup
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        tracker.cleanup();

        // Should be cleaned up
        assert_eq!(tracker.total_sample_count(), 0);
    }

    #[tokio::test]
    async fn test_estimate_with_fill_history() {
        let config = QueueEstimatorConfig {
            enabled: true,
            history_window_secs: 300,
            min_samples_for_estimate: 3, // Lower threshold for test
            confidence_decay_factor: 0.95,
        };
        let estimator = QueueEstimator::new(config);
        let token_id = "test_token".to_string();
        let orderbook = create_test_orderbook();

        // Record enough fills to enable estimation
        for _ in 0..5 {
            estimator
                .record_fill(&token_id, Decimal::new(50, 2), Decimal::from(100))
                .await;
        }

        let position = estimator
            .estimate_position(
                &token_id,
                &orderbook,
                Side::Buy,
                Decimal::new(50, 2),
                Decimal::from(100),
            )
            .await;

        // Should now have estimates
        assert!(position.estimated_time_to_fill_secs.is_some());
        assert!(position.confidence > 0.0);
        assert!(position.fill_probability_1min >= 0.0);
        assert!(position.fill_probability_5min >= 0.0);
    }

    #[test]
    fn test_queue_position_calculation() {
        let config = create_test_config();
        let estimator = QueueEstimator::new(config);
        let orderbook = create_test_orderbook();

        // Test buy order at lower price - should see more size ahead
        let size_ahead = estimator.calculate_size_ahead(&orderbook, Side::Buy, Decimal::new(48, 2));
        // At 0.48, we see 0.50 (1000) + 0.49 (2000) + 0.48 (1500) = 4500
        assert_eq!(size_ahead, Decimal::from(4500));

        // Test sell order at higher price - should see more size ahead
        let size_ahead = estimator.calculate_size_ahead(&orderbook, Side::Sell, Decimal::new(53, 2));
        // At 0.53, we see 0.51 (800) + 0.52 (1200) + 0.53 (1000) = 3000
        assert_eq!(size_ahead, Decimal::from(3000));
    }
}
