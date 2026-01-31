//! Price Level Analyzer
//!
//! Tracks behavior at specific price levels including touch frequency,
//! time spent at level, and fill statistics for improved fill probability modeling.

use chrono::{DateTime, Duration, Utc};
use polysniper_core::{PriceLevelConfig, TokenId};
use rust_decimal::Decimal;
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::debug;

/// Statistics tracked for each price level
#[derive(Debug, Clone)]
pub struct PriceLevelStats {
    /// Number of times price touched this level
    pub touches: u32,
    /// Cumulative time spent at this level in milliseconds
    pub time_at_level_ms: u64,
    /// Total volume traded at this level
    pub volume_traded: Decimal,
    /// Number of orders completely filled at this level
    pub orders_filled: u32,
    /// Number of orders partially filled at this level
    pub orders_partially_filled: u32,
    /// Average fill percentage when price touches this level
    pub avg_fill_pct_when_touched: Decimal,
    /// Last time price touched this level
    pub last_touch: DateTime<Utc>,
    /// First touch time (for time window calculations)
    first_touch: DateTime<Utc>,
    /// Sum of fill percentages (for calculating average)
    fill_pct_sum: Decimal,
    /// Number of fill events (for calculating average)
    fill_events: u32,
}

impl Default for PriceLevelStats {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            touches: 0,
            time_at_level_ms: 0,
            volume_traded: Decimal::ZERO,
            orders_filled: 0,
            orders_partially_filled: 0,
            avg_fill_pct_when_touched: Decimal::ZERO,
            last_touch: now,
            first_touch: now,
            fill_pct_sum: Decimal::ZERO,
            fill_events: 0,
        }
    }
}

impl PriceLevelStats {
    /// Create new stats with initial touch
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a touch at this price level
    pub fn record_touch(&mut self, duration_ms: u64) {
        if self.touches == 0 {
            self.first_touch = Utc::now();
        }
        self.touches += 1;
        self.time_at_level_ms += duration_ms;
        self.last_touch = Utc::now();
    }

    /// Record a fill event at this price level
    pub fn record_fill(&mut self, fill_pct: Decimal, volume: Decimal) {
        self.volume_traded += volume;
        self.fill_pct_sum += fill_pct;
        self.fill_events += 1;

        if fill_pct >= Decimal::ONE {
            self.orders_filled += 1;
        } else if fill_pct > Decimal::ZERO {
            self.orders_partially_filled += 1;
        }

        // Update average fill percentage
        if self.fill_events > 0 {
            self.avg_fill_pct_when_touched =
                self.fill_pct_sum / Decimal::from(self.fill_events);
        }
    }

    /// Get average time per touch in milliseconds
    pub fn avg_time_per_touch_ms(&self) -> u64 {
        if self.touches > 0 {
            self.time_at_level_ms / self.touches as u64
        } else {
            0
        }
    }

    /// Check if stats are stale (older than window)
    pub fn is_stale(&self, window: Duration) -> bool {
        Utc::now() - self.last_touch > window
    }
}

/// Analyzes price level behavior for fill probability estimation
pub struct PriceLevelAnalyzer {
    /// Statistics per price level per token
    price_levels: RwLock<HashMap<TokenId, HashMap<String, PriceLevelStats>>>,
    /// Configuration
    config: PriceLevelConfig,
}

impl PriceLevelAnalyzer {
    /// Create a new price level analyzer
    pub fn new(config: PriceLevelConfig) -> Self {
        Self {
            price_levels: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Record that price touched a level
    pub async fn record_touch(&self, token_id: &TokenId, price: Decimal, duration_ms: u64) {
        let mut levels = self.price_levels.write().await;
        let token_levels = levels.entry(token_id.clone()).or_default();
        let price_key = price.to_string();

        let stats = token_levels.entry(price_key).or_default();
        stats.record_touch(duration_ms);

        debug!(
            token_id = %token_id,
            price = %price,
            duration_ms = duration_ms,
            touches = stats.touches,
            "Recorded price level touch"
        );
    }

    /// Record a fill event at a price level
    pub async fn record_fill(
        &self,
        token_id: &TokenId,
        price: Decimal,
        fill_pct: Decimal,
        volume: Decimal,
    ) {
        let mut levels = self.price_levels.write().await;
        let token_levels = levels.entry(token_id.clone()).or_default();
        let price_key = price.to_string();

        let stats = token_levels.entry(price_key).or_default();
        stats.record_fill(fill_pct, volume);

        debug!(
            token_id = %token_id,
            price = %price,
            fill_pct = %fill_pct,
            volume = %volume,
            "Recorded fill at price level"
        );
    }

    /// Get probability of fill if price touches our level
    ///
    /// Returns None if insufficient data (less than min_touches_for_stats touches)
    pub async fn fill_probability_on_touch(
        &self,
        token_id: &TokenId,
        price: Decimal,
    ) -> Option<Decimal> {
        let levels = self.price_levels.read().await;
        let token_levels = levels.get(token_id)?;
        let price_key = price.to_string();
        let stats = token_levels.get(&price_key)?;

        // Need minimum touches for valid statistics
        if stats.touches < self.config.min_touches_for_stats {
            return None;
        }

        // Use average fill percentage as probability
        Some(stats.avg_fill_pct_when_touched)
    }

    /// Get average time price spends at a level in milliseconds
    pub async fn avg_time_at_level(&self, token_id: &TokenId, price: Decimal) -> Option<u64> {
        let levels = self.price_levels.read().await;
        let token_levels = levels.get(token_id)?;
        let price_key = price.to_string();
        let stats = token_levels.get(&price_key)?;

        if stats.touches < self.config.min_touches_for_stats {
            return None;
        }

        Some(stats.avg_time_per_touch_ms())
    }

    /// Get statistics for a price level (returns None if no data)
    pub async fn get_stats(&self, token_id: &TokenId, price: Decimal) -> Option<PriceLevelStats> {
        let levels = self.price_levels.read().await;
        let token_levels = levels.get(token_id)?;
        let price_key = price.to_string();
        token_levels.get(&price_key).cloned()
    }

    /// Get touch count at a price level
    pub async fn touch_count(&self, token_id: &TokenId, price: Decimal) -> u32 {
        let levels = self.price_levels.read().await;
        levels
            .get(token_id)
            .and_then(|t| t.get(&price.to_string()))
            .map(|s| s.touches)
            .unwrap_or(0)
    }

    /// Clean up old entries outside the history window
    pub async fn cleanup_old_entries(&self) {
        let window = Duration::seconds(self.config.history_window_secs as i64);
        let mut levels = self.price_levels.write().await;

        for token_levels in levels.values_mut() {
            token_levels.retain(|_, stats| !stats.is_stale(window));
        }

        // Remove empty token entries
        levels.retain(|_, token_levels| !token_levels.is_empty());
    }

    /// Get total number of tracked price levels across all tokens
    pub async fn total_price_levels(&self) -> usize {
        let levels = self.price_levels.read().await;
        levels.values().map(|t| t.len()).sum()
    }

    /// Check if we have sufficient data for a price level
    pub async fn has_sufficient_data(&self, token_id: &TokenId, price: Decimal) -> bool {
        self.touch_count(token_id, price).await >= self.config.min_touches_for_stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn test_config() -> PriceLevelConfig {
        PriceLevelConfig {
            history_window_secs: 3600,
            min_touches_for_stats: 5,
        }
    }

    #[tokio::test]
    async fn test_record_touch() {
        let analyzer = PriceLevelAnalyzer::new(test_config());
        let token_id = "test_token".to_string();
        let price = dec!(0.50);

        analyzer.record_touch(&token_id, price, 100).await;
        analyzer.record_touch(&token_id, price, 200).await;
        analyzer.record_touch(&token_id, price, 150).await;

        let stats = analyzer.get_stats(&token_id, price).await.unwrap();
        assert_eq!(stats.touches, 3);
        assert_eq!(stats.time_at_level_ms, 450);
    }

    #[tokio::test]
    async fn test_record_fill() {
        let analyzer = PriceLevelAnalyzer::new(test_config());
        let token_id = "test_token".to_string();
        let price = dec!(0.50);

        analyzer
            .record_fill(&token_id, price, dec!(1.0), dec!(100))
            .await;
        analyzer
            .record_fill(&token_id, price, dec!(0.5), dec!(50))
            .await;

        let stats = analyzer.get_stats(&token_id, price).await.unwrap();
        assert_eq!(stats.orders_filled, 1);
        assert_eq!(stats.orders_partially_filled, 1);
        assert_eq!(stats.volume_traded, dec!(150));
        assert_eq!(stats.avg_fill_pct_when_touched, dec!(0.75)); // (1.0 + 0.5) / 2
    }

    #[tokio::test]
    async fn test_fill_probability_requires_min_touches() {
        let analyzer = PriceLevelAnalyzer::new(test_config());
        let token_id = "test_token".to_string();
        let price = dec!(0.50);

        // Record only 3 touches (below min of 5)
        for _ in 0..3 {
            analyzer.record_touch(&token_id, price, 100).await;
            analyzer
                .record_fill(&token_id, price, dec!(0.8), dec!(100))
                .await;
        }

        // Should return None due to insufficient data
        let prob = analyzer.fill_probability_on_touch(&token_id, price).await;
        assert!(prob.is_none());

        // Record 2 more touches to meet minimum
        for _ in 0..2 {
            analyzer.record_touch(&token_id, price, 100).await;
            analyzer
                .record_fill(&token_id, price, dec!(0.8), dec!(100))
                .await;
        }

        // Now should have data
        let prob = analyzer.fill_probability_on_touch(&token_id, price).await;
        assert!(prob.is_some());
        assert_eq!(prob.unwrap(), dec!(0.8));
    }

    #[tokio::test]
    async fn test_avg_time_at_level() {
        let analyzer = PriceLevelAnalyzer::new(test_config());
        let token_id = "test_token".to_string();
        let price = dec!(0.50);

        // Record enough touches for valid stats
        for _ in 0..5 {
            analyzer.record_touch(&token_id, price, 100).await;
        }

        let avg_time = analyzer.avg_time_at_level(&token_id, price).await;
        assert!(avg_time.is_some());
        assert_eq!(avg_time.unwrap(), 100);
    }

    #[tokio::test]
    async fn test_multiple_tokens() {
        let analyzer = PriceLevelAnalyzer::new(test_config());
        let token1 = "token1".to_string();
        let token2 = "token2".to_string();
        let price = dec!(0.50);

        analyzer.record_touch(&token1, price, 100).await;
        analyzer.record_touch(&token2, price, 200).await;

        let stats1 = analyzer.get_stats(&token1, price).await.unwrap();
        let stats2 = analyzer.get_stats(&token2, price).await.unwrap();

        assert_eq!(stats1.touches, 1);
        assert_eq!(stats2.touches, 1);
        assert_eq!(stats1.time_at_level_ms, 100);
        assert_eq!(stats2.time_at_level_ms, 200);
    }

    #[tokio::test]
    async fn test_has_sufficient_data() {
        let analyzer = PriceLevelAnalyzer::new(test_config());
        let token_id = "test_token".to_string();
        let price = dec!(0.50);

        assert!(!analyzer.has_sufficient_data(&token_id, price).await);

        for _ in 0..5 {
            analyzer.record_touch(&token_id, price, 100).await;
        }

        assert!(analyzer.has_sufficient_data(&token_id, price).await);
    }

    #[tokio::test]
    async fn test_total_price_levels() {
        let analyzer = PriceLevelAnalyzer::new(test_config());
        let token1 = "token1".to_string();
        let token2 = "token2".to_string();

        analyzer.record_touch(&token1, dec!(0.50), 100).await;
        analyzer.record_touch(&token1, dec!(0.51), 100).await;
        analyzer.record_touch(&token2, dec!(0.50), 100).await;

        assert_eq!(analyzer.total_price_levels().await, 3);
    }
}
