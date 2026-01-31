//! Queue Position Estimator
//!
//! Tracks queue position for limit orders and estimates time-to-fill based on
//! historical fill rates, active position tracking, and price level analysis.

use crate::price_level_analyzer::PriceLevelAnalyzer;
use chrono::{DateTime, Duration, Utc};
use polysniper_core::{Orderbook, OrderId, QueueEstimatorConfig, QueuePosition, Side, TokenId};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
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
    pub avg_fill_rate: Decimal,
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
                self.avg_fill_rate = total_size / Decimal::from(duration_secs);
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

/// State for an actively tracked order position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuePositionState {
    /// Order ID being tracked
    pub order_id: OrderId,
    /// Token ID for this order
    pub token_id: TokenId,
    /// Price level of the order
    pub price: Decimal,
    /// Side of the order (buy/sell)
    pub side: Side,
    /// Size of the order
    pub size: Decimal,
    /// Initial size ahead when order was placed
    pub initial_size_ahead: Decimal,
    /// Current estimated size ahead
    pub current_size_ahead: Decimal,
    /// When the order was placed
    pub order_placed_at: DateTime<Utc>,
    /// Last time position was updated
    pub last_update: DateTime<Utc>,
    /// Number of orders observed leaving the queue ahead
    pub observed_departures: u32,
}

impl QueuePositionState {
    /// Get the age of this order in seconds
    pub fn age_secs(&self) -> i64 {
        (Utc::now() - self.order_placed_at).num_seconds()
    }

    /// Calculate progress through queue (0.0 = back, 1.0 = front)
    pub fn queue_progress(&self) -> Decimal {
        if self.initial_size_ahead.is_zero() {
            return Decimal::ONE;
        }
        let processed = self.initial_size_ahead - self.current_size_ahead;
        (processed / self.initial_size_ahead).max(Decimal::ZERO).min(Decimal::ONE)
    }
}

/// Method used to calculate fill probability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbabilityMethod {
    /// Based on historical fill rate only
    HistoricalRate,
    /// Based on queue position only
    QueuePosition,
    /// Based on price level statistics
    PriceLevel,
    /// Weighted combination of all methods
    Combined,
    /// Not enough data for reliable estimate
    Insufficient,
}

/// Components that contributed to the probability calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityComponents {
    /// Probability from historical fill rate analysis
    pub historical_rate_prob: Option<Decimal>,
    /// Probability from queue position analysis
    pub queue_position_prob: Option<Decimal>,
    /// Probability from price level statistics
    pub price_level_prob: Option<Decimal>,
    /// Age-based adjustment factor
    pub age_factor: Decimal,
}

impl Default for ProbabilityComponents {
    fn default() -> Self {
        Self {
            historical_rate_prob: None,
            queue_position_prob: None,
            price_level_prob: None,
            age_factor: Decimal::ONE,
        }
    }
}

/// Enhanced fill probability estimate with multiple data sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillProbability {
    /// Final probability estimate (0.0 - 1.0)
    pub probability: Decimal,
    /// Confidence in the estimate (0.0 - 1.0)
    pub confidence: Decimal,
    /// Expected time to fill in seconds (if filled)
    pub expected_time_secs: Option<u64>,
    /// Method used for the estimate
    pub method: ProbabilityMethod,
    /// Component probabilities that contributed
    pub components: ProbabilityComponents,
}

impl FillProbability {
    /// Create an insufficient data response
    pub fn insufficient() -> Self {
        Self {
            probability: Decimal::ZERO,
            confidence: Decimal::ZERO,
            expected_time_secs: None,
            method: ProbabilityMethod::Insufficient,
            components: ProbabilityComponents::default(),
        }
    }
}

/// Queue position estimator for limit orders
pub struct QueueEstimator {
    /// Historical fill rates per token
    fill_rate_history: RwLock<HashMap<TokenId, FillRateTracker>>,
    /// Actively tracked order positions
    tracked_positions: RwLock<HashMap<OrderId, QueuePositionState>>,
    /// Configuration
    config: QueueEstimatorConfig,
}

impl QueueEstimator {
    /// Create a new queue estimator
    pub fn new(config: QueueEstimatorConfig) -> Self {
        Self {
            fill_rate_history: RwLock::new(HashMap::new()),
            tracked_positions: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Start tracking an order's queue position
    pub async fn start_tracking(
        &self,
        order_id: &OrderId,
        token_id: &TokenId,
        orderbook: &Orderbook,
        side: Side,
        price: Decimal,
        size: Decimal,
    ) {
        let size_ahead = self.calculate_size_ahead(orderbook, side, price);
        let now = Utc::now();

        let state = QueuePositionState {
            order_id: order_id.clone(),
            token_id: token_id.clone(),
            price,
            side,
            size,
            initial_size_ahead: size_ahead,
            current_size_ahead: size_ahead,
            order_placed_at: now,
            last_update: now,
            observed_departures: 0,
        };

        let mut positions = self.tracked_positions.write().await;
        positions.insert(order_id.clone(), state);

        debug!(
            order_id = %order_id,
            token_id = %token_id,
            price = %price,
            size_ahead = %size_ahead,
            "Started tracking order queue position"
        );
    }

    /// Update tracked positions from orderbook change
    pub async fn update_positions(&self, token_id: &TokenId, orderbook: &Orderbook) {
        let mut positions = self.tracked_positions.write().await;
        let now = Utc::now();

        for state in positions.values_mut() {
            if &state.token_id != token_id {
                continue;
            }

            let old_size_ahead = state.current_size_ahead;
            let new_size_ahead = self.calculate_size_ahead(orderbook, state.side, state.price);

            // Track observed departures (size that left queue)
            if new_size_ahead < old_size_ahead {
                // Estimate number of orders that departed (assuming avg 100 size per order)
                let departed_size = old_size_ahead - new_size_ahead;
                let estimated_departed = (departed_size / Decimal::from(100))
                    .to_u32()
                    .unwrap_or(1)
                    .max(1);
                state.observed_departures += estimated_departed;
            }

            state.current_size_ahead = new_size_ahead;
            state.last_update = now;
        }
    }

    /// Get tracked position state for an order
    pub async fn get_tracked_position(&self, order_id: &OrderId) -> Option<QueuePositionState> {
        let positions = self.tracked_positions.read().await;
        positions.get(order_id).cloned()
    }

    /// Stop tracking an order (on fill or cancel)
    pub async fn stop_tracking(&self, order_id: &OrderId) {
        let mut positions = self.tracked_positions.write().await;
        if positions.remove(order_id).is_some() {
            debug!(order_id = %order_id, "Stopped tracking order queue position");
        }
    }

    /// Get number of actively tracked orders
    pub async fn tracked_order_count(&self) -> usize {
        self.tracked_positions.read().await.len()
    }

    /// Apply age-based adjustment to fill probability
    ///
    /// Bi-modal model:
    /// - First 5 minutes: older orders are more likely to fill (queue priority)
    /// - After 5 minutes: exponential decay (orders may be stale/market moved)
    fn apply_age_factor(&self, state: &QueuePositionState) -> Decimal {
        let age_secs = state.age_secs() as f64;

        if age_secs < 300.0 {
            // First 5 minutes: linear increase from 1.0 to 1.2
            // This reflects queue priority - older orders at front
            let factor = 1.0 + (age_secs / 1500.0); // 0.2 over 300 secs
            Decimal::try_from(factor).unwrap_or(Decimal::ONE)
        } else {
            // After 5 minutes: exponential decay with half-life of ~30 minutes
            // This reflects that very old orders may indicate wrong price
            let decay_age = age_secs - 300.0;
            let decay = (-decay_age / 1800.0).exp();
            let factor = 1.2 * decay; // Start from 1.2 and decay
            Decimal::try_from(factor.max(0.1)).unwrap_or(dec!(0.5))
        }
    }

    /// Calculate enhanced fill probability using multiple methods
    pub async fn calculate_fill_probability(
        &self,
        order_id: &OrderId,
        time_horizon_secs: u64,
        price_analyzer: Option<&PriceLevelAnalyzer>,
    ) -> FillProbability {
        // Get tracked state
        let state = match self.get_tracked_position(order_id).await {
            Some(s) => s,
            None => return FillProbability::insufficient(),
        };

        let mut components = ProbabilityComponents {
            age_factor: self.apply_age_factor(&state),
            ..Default::default()
        };

        // 1. Historical rate probability
        let (historical_prob, sample_count) = {
            let history = self.fill_rate_history.read().await;
            let tracker = history.get(&state.token_id);

            let prob = if let Some(tracker) = tracker {
                if tracker.total_sample_count() >= self.config.min_samples_for_estimate as usize {
                    let rate = tracker
                        .get_fill_rate_at_price(state.price)
                        .unwrap_or(tracker.avg_fill_rate);

                    if !rate.is_zero() {
                        // P(fill) = 1 - exp(-rate * time / size_ahead)
                        let rate_f64 = rate.to_f64().unwrap_or(0.0);
                        let size_f64 = state.current_size_ahead.to_f64().unwrap_or(1.0).max(0.001);
                        let lambda = rate_f64 / size_f64;
                        let prob = 1.0 - (-lambda * time_horizon_secs as f64).exp();
                        Some(Decimal::try_from(prob.clamp(0.0, 1.0)).unwrap_or(Decimal::ZERO))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let count = tracker.map(|t| t.total_sample_count()).unwrap_or(0);
            (prob, count)
        };

        components.historical_rate_prob = historical_prob;

        // 2. Queue position probability
        let queue_prob = if !state.initial_size_ahead.is_zero() {
            // Simple model: probability proportional to progress through queue
            let progress = state.queue_progress();
            // Higher progress = higher probability, scaled by time horizon
            let time_factor = Decimal::try_from((time_horizon_secs as f64 / 300.0).min(1.0))
                .unwrap_or(Decimal::ONE);
            Some((progress * time_factor).min(Decimal::ONE))
        } else {
            // No queue ahead - high probability
            Some(dec!(0.95))
        };
        components.queue_position_prob = queue_prob;

        // 3. Price level probability (if analyzer provided)
        let price_level_prob = if let Some(analyzer) = price_analyzer {
            analyzer
                .fill_probability_on_touch(&state.token_id, state.price)
                .await
        } else {
            None
        };
        components.price_level_prob = price_level_prob;

        // Determine method and calculate combined probability
        let (method, raw_probability) = self.combine_probabilities(&components);

        // Apply age factor
        let final_probability = (raw_probability * components.age_factor).min(Decimal::ONE);

        // Calculate confidence based on available data sources
        let confidence = self.calculate_confidence(&components, sample_count);

        // Estimate expected time to fill
        let expected_time = self.estimate_fill_time(&state, &components);

        FillProbability {
            probability: final_probability,
            confidence,
            expected_time_secs: expected_time,
            method,
            components,
        }
    }

    /// Combine probability estimates from multiple sources
    fn combine_probabilities(&self, components: &ProbabilityComponents) -> (ProbabilityMethod, Decimal) {
        let mut sources = Vec::new();
        let mut weights = Vec::new();

        // Historical rate: highest weight when available (most reliable)
        if let Some(prob) = components.historical_rate_prob {
            sources.push(prob);
            weights.push(dec!(0.5));
        }

        // Queue position: medium weight
        if let Some(prob) = components.queue_position_prob {
            sources.push(prob);
            weights.push(dec!(0.3));
        }

        // Price level: lower weight (supplementary)
        if let Some(prob) = components.price_level_prob {
            sources.push(prob);
            weights.push(dec!(0.2));
        }

        if sources.is_empty() {
            return (ProbabilityMethod::Insufficient, Decimal::ZERO);
        }

        if sources.len() == 1 {
            let method = if components.historical_rate_prob.is_some() {
                ProbabilityMethod::HistoricalRate
            } else if components.queue_position_prob.is_some() {
                ProbabilityMethod::QueuePosition
            } else {
                ProbabilityMethod::PriceLevel
            };
            return (method, sources[0]);
        }

        // Normalize weights and calculate weighted average
        let total_weight: Decimal = weights.iter().take(sources.len()).sum();
        let weighted_sum: Decimal = sources
            .iter()
            .zip(weights.iter())
            .map(|(prob, weight)| *prob * *weight)
            .sum();

        let combined = weighted_sum / total_weight;
        (ProbabilityMethod::Combined, combined)
    }

    /// Calculate confidence in the estimate
    fn calculate_confidence(&self, components: &ProbabilityComponents, sample_count: usize) -> Decimal {
        let mut confidence = Decimal::ZERO;

        // Base confidence from sample count (up to 0.4)
        let sample_confidence = Decimal::try_from((sample_count as f64 / 100.0).min(0.4))
            .unwrap_or(Decimal::ZERO);
        confidence += sample_confidence;

        // Add confidence for each available data source
        if components.historical_rate_prob.is_some() {
            confidence += dec!(0.3);
        }
        if components.queue_position_prob.is_some() {
            confidence += dec!(0.2);
        }
        if components.price_level_prob.is_some() {
            confidence += dec!(0.1);
        }

        confidence.min(Decimal::ONE)
    }

    /// Estimate time to fill based on available data
    fn estimate_fill_time(
        &self,
        state: &QueuePositionState,
        components: &ProbabilityComponents,
    ) -> Option<u64> {
        // If historical rate available, use it
        if let Some(prob) = components.historical_rate_prob {
            if prob > Decimal::ZERO {
                // Invert exponential: t = -ln(1-p) * size / rate
                // Approximate: t ≈ size_ahead / rate when p is moderate
                let time_estimate = (state.current_size_ahead / prob)
                    .to_f64()
                    .unwrap_or(0.0)
                    .abs() as u64;
                return Some(time_estimate.min(86400)); // Cap at 24 hours
            }
        }

        // Fall back to progress-based estimate
        if !state.initial_size_ahead.is_zero() {
            let progress = state.queue_progress();
            if progress > Decimal::ZERO {
                let elapsed = state.age_secs() as u64;
                let remaining_fraction = Decimal::ONE - progress;
                let estimated_total = Decimal::from(elapsed) / progress;
                let remaining = (estimated_total * remaining_fraction)
                    .to_u64()
                    .unwrap_or(0);
                return Some(remaining.min(86400));
            }
        }

        None
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
            (seconds_since_update / 60) as i32, // Decay per minute
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
    use polysniper_core::{PriceLevel, PriceLevelConfig};

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
            estimator
                .record_fill(&token_id, price, Decimal::from(100))
                .await;
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
        let size_ahead =
            estimator.calculate_size_ahead(&orderbook, Side::Buy, Decimal::new(48, 2));
        // At 0.48, we see 0.50 (1000) + 0.49 (2000) + 0.48 (1500) = 4500
        assert_eq!(size_ahead, Decimal::from(4500));

        // Test sell order at higher price - should see more size ahead
        let size_ahead =
            estimator.calculate_size_ahead(&orderbook, Side::Sell, Decimal::new(53, 2));
        // At 0.53, we see 0.51 (800) + 0.52 (1200) + 0.53 (1000) = 3000
        assert_eq!(size_ahead, Decimal::from(3000));
    }

    #[tokio::test]
    async fn test_active_position_tracking() {
        let config = create_test_config();
        let estimator = QueueEstimator::new(config);
        let orderbook = create_test_orderbook();
        let order_id = "order_1".to_string();
        let token_id = "test_token".to_string();

        // Start tracking
        estimator
            .start_tracking(
                &order_id,
                &token_id,
                &orderbook,
                Side::Buy,
                Decimal::new(50, 2),
                Decimal::from(100),
            )
            .await;

        // Should have position
        let state = estimator.get_tracked_position(&order_id).await;
        assert!(state.is_some());
        let state = state.unwrap();
        assert_eq!(state.initial_size_ahead, Decimal::from(1000));
        assert_eq!(state.current_size_ahead, Decimal::from(1000));

        // Stop tracking
        estimator.stop_tracking(&order_id).await;
        assert!(estimator.get_tracked_position(&order_id).await.is_none());
    }

    #[tokio::test]
    async fn test_position_update_on_orderbook_change() {
        let config = create_test_config();
        let estimator = QueueEstimator::new(config);
        let orderbook = create_test_orderbook();
        let order_id = "order_1".to_string();
        let token_id = "test_token".to_string();

        // Start tracking
        estimator
            .start_tracking(
                &order_id,
                &token_id,
                &orderbook,
                Side::Buy,
                Decimal::new(50, 2),
                Decimal::from(100),
            )
            .await;

        // Create new orderbook with less size ahead
        let mut updated_orderbook = orderbook.clone();
        updated_orderbook.bids[0].size = Decimal::from(500); // Reduced from 1000

        // Update positions
        estimator
            .update_positions(&token_id, &updated_orderbook)
            .await;

        // Check updated position
        let state = estimator.get_tracked_position(&order_id).await.unwrap();
        assert_eq!(state.current_size_ahead, Decimal::from(500));
        assert!(state.observed_departures > 0);
    }

    #[test]
    fn test_age_factor_early() {
        let config = create_test_config();
        let estimator = QueueEstimator::new(config);

        // Create a fresh order (0 seconds old)
        let state = QueuePositionState {
            order_id: "test".to_string(),
            token_id: "token".to_string(),
            price: dec!(0.50),
            side: Side::Buy,
            size: dec!(100),
            initial_size_ahead: dec!(1000),
            current_size_ahead: dec!(1000),
            order_placed_at: Utc::now(),
            last_update: Utc::now(),
            observed_departures: 0,
        };

        let factor = estimator.apply_age_factor(&state);
        // Should be close to 1.0 for fresh orders
        assert!(factor >= dec!(0.99) && factor <= dec!(1.01));
    }

    #[test]
    fn test_queue_progress() {
        let state = QueuePositionState {
            order_id: "test".to_string(),
            token_id: "token".to_string(),
            price: dec!(0.50),
            side: Side::Buy,
            size: dec!(100),
            initial_size_ahead: dec!(1000),
            current_size_ahead: dec!(250), // 75% progress
            order_placed_at: Utc::now(),
            last_update: Utc::now(),
            observed_departures: 7,
        };

        let progress = state.queue_progress();
        assert_eq!(progress, dec!(0.75));
    }

    #[tokio::test]
    async fn test_calculate_fill_probability() {
        let config = QueueEstimatorConfig {
            enabled: true,
            history_window_secs: 300,
            min_samples_for_estimate: 3,
            confidence_decay_factor: 0.95,
        };
        let estimator = QueueEstimator::new(config);
        let orderbook = create_test_orderbook();
        let order_id = "order_1".to_string();
        let token_id = "test_token".to_string();

        // Start tracking
        estimator
            .start_tracking(
                &order_id,
                &token_id,
                &orderbook,
                Side::Buy,
                Decimal::new(50, 2),
                Decimal::from(100),
            )
            .await;

        // Record fills to build history
        for _ in 0..10 {
            estimator
                .record_fill(&token_id, Decimal::new(50, 2), Decimal::from(100))
                .await;
        }

        // Calculate probability
        let prob = estimator
            .calculate_fill_probability(&order_id, 300, None)
            .await;

        assert!(prob.probability > Decimal::ZERO);
        assert!(prob.confidence > Decimal::ZERO);
        assert!(matches!(
            prob.method,
            ProbabilityMethod::Combined | ProbabilityMethod::HistoricalRate
        ));
    }

    #[tokio::test]
    async fn test_calculate_fill_probability_with_price_analyzer() {
        let config = QueueEstimatorConfig {
            enabled: true,
            history_window_secs: 300,
            min_samples_for_estimate: 3,
            confidence_decay_factor: 0.95,
        };
        let estimator = QueueEstimator::new(config);

        let price_config = PriceLevelConfig {
            history_window_secs: 3600,
            min_touches_for_stats: 3,
        };
        let price_analyzer = PriceLevelAnalyzer::new(price_config);

        let orderbook = create_test_orderbook();
        let order_id = "order_1".to_string();
        let token_id = "test_token".to_string();
        let price = Decimal::new(50, 2);

        // Start tracking
        estimator
            .start_tracking(&order_id, &token_id, &orderbook, Side::Buy, price, Decimal::from(100))
            .await;

        // Build price level history
        for _ in 0..5 {
            price_analyzer.record_touch(&token_id, price, 100).await;
            price_analyzer
                .record_fill(&token_id, price, dec!(0.8), dec!(100))
                .await;
        }

        // Calculate probability with price analyzer
        let prob = estimator
            .calculate_fill_probability(&order_id, 300, Some(&price_analyzer))
            .await;

        assert!(prob.probability > Decimal::ZERO);
        assert!(prob.components.price_level_prob.is_some());
    }

    #[tokio::test]
    async fn test_insufficient_data() {
        let config = create_test_config();
        let estimator = QueueEstimator::new(config);

        // Try to get probability for non-existent order
        let prob = estimator
            .calculate_fill_probability(&"nonexistent".to_string(), 300, None)
            .await;

        assert_eq!(prob.method, ProbabilityMethod::Insufficient);
        assert_eq!(prob.probability, Decimal::ZERO);
        assert_eq!(prob.confidence, Decimal::ZERO);
    }
}
