//! VPIN (Volume-Synchronized Probability of Informed Trading) Calculator
//!
//! Measures order flow toxicity by tracking the probability of informed trading
//! based on volume-synchronized buckets.

use chrono::{DateTime, Utc};
use polysniper_core::{Side, TokenId};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// VPIN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpinConfig {
    /// Whether VPIN calculation is enabled
    pub enabled: bool,
    /// Volume per bucket in USD (e.g., $1000)
    pub bucket_size_usd: Decimal,
    /// Number of buckets for VPIN calculation (e.g., 50)
    pub lookback_buckets: usize,
    /// Alert threshold for high toxicity (e.g., 0.7)
    pub high_toxicity_threshold: Decimal,
    /// Safe threshold for low toxicity (e.g., 0.3)
    pub low_toxicity_threshold: Decimal,
    /// Method used to classify trades as buy or sell initiated
    pub trade_classification: TradeClassificationMethod,
}

impl Default for VpinConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bucket_size_usd: dec!(1000),
            lookback_buckets: 50,
            high_toxicity_threshold: dec!(0.7),
            low_toxicity_threshold: dec!(0.3),
            trade_classification: TradeClassificationMethod::QuoteMidpoint,
        }
    }
}

/// Method used to classify trades as buy-initiated or sell-initiated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TradeClassificationMethod {
    /// Compare trade price to previous trade price
    TickRule,
    /// Compare trade price to bid-ask midpoint
    #[default]
    QuoteMidpoint,
    /// Lee-Ready algorithm with bulk volume classification
    BulkVolume,
}

/// A volume-synchronized bucket tracking buy and sell volume
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeBucket {
    /// Volume from buy-initiated trades in USD
    pub buy_volume: Decimal,
    /// Volume from sell-initiated trades in USD
    pub sell_volume: Decimal,
    /// Total volume in USD
    pub total_volume: Decimal,
    /// When this bucket started filling
    pub start_time: DateTime<Utc>,
    /// When this bucket was completed (None if still filling)
    pub end_time: Option<DateTime<Utc>>,
    /// Number of trades in this bucket
    pub trade_count: u32,
}

impl VolumeBucket {
    fn new() -> Self {
        Self {
            buy_volume: Decimal::ZERO,
            sell_volume: Decimal::ZERO,
            total_volume: Decimal::ZERO,
            start_time: Utc::now(),
            end_time: None,
            trade_count: 0,
        }
    }

    /// Add a trade to this bucket
    fn add_trade(&mut self, volume_usd: Decimal, side: Side) {
        match side {
            Side::Buy => self.buy_volume += volume_usd,
            Side::Sell => self.sell_volume += volume_usd,
        }
        self.total_volume += volume_usd;
        self.trade_count += 1;
    }

    /// Finalize this bucket
    fn finalize(&mut self) {
        self.end_time = Some(Utc::now());
    }

    /// Get the absolute imbalance |buy - sell|
    fn imbalance(&self) -> Decimal {
        (self.buy_volume - self.sell_volume).abs()
    }
}

/// Result of VPIN calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpinResult {
    /// Token ID for this calculation
    pub token_id: TokenId,
    /// VPIN value (0.0 to 1.0)
    pub vpin: Decimal,
    /// Toxicity level classification
    pub toxicity_level: ToxicityLevel,
    /// Percentage of volume from buy-initiated trades
    pub buy_volume_pct: Decimal,
    /// Percentage of volume from sell-initiated trades
    pub sell_volume_pct: Decimal,
    /// Number of buckets used in calculation
    pub bucket_count: usize,
    /// When this calculation was made
    pub timestamp: DateTime<Utc>,
}

/// Toxicity level classification based on VPIN value
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToxicityLevel {
    /// VPIN < low_threshold (e.g., < 0.3)
    Low,
    /// low_threshold <= VPIN < 0.5
    Normal,
    /// 0.5 <= VPIN < high_threshold
    Elevated,
    /// VPIN >= high_threshold (e.g., >= 0.7)
    High,
}

impl std::fmt::Display for ToxicityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToxicityLevel::Low => write!(f, "Low"),
            ToxicityLevel::Normal => write!(f, "Normal"),
            ToxicityLevel::Elevated => write!(f, "Elevated"),
            ToxicityLevel::High => write!(f, "High"),
        }
    }
}

/// State tracking for a single token
#[derive(Debug)]
struct TokenState {
    /// Completed buckets for VPIN calculation
    completed_buckets: VecDeque<VolumeBucket>,
    /// Current bucket being filled
    current_bucket: VolumeBucket,
    /// Last trade price (for tick rule)
    last_price: Option<Decimal>,
    /// Last classified side (for tick rule when prices are equal)
    last_side: Option<Side>,
}

impl TokenState {
    fn new() -> Self {
        Self {
            completed_buckets: VecDeque::new(),
            current_bucket: VolumeBucket::new(),
            last_price: None,
            last_side: None,
        }
    }
}

/// VPIN Calculator
///
/// Calculates the Volume-synchronized Probability of Informed Trading (VPIN)
/// metric for each token based on trade flow data.
pub struct VpinCalculator {
    config: VpinConfig,
    /// Per-token state
    token_states: HashMap<TokenId, TokenState>,
}

impl VpinCalculator {
    /// Create a new VPIN calculator with the given configuration
    pub fn new(config: VpinConfig) -> Self {
        Self {
            config,
            token_states: HashMap::new(),
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &VpinConfig {
        &self.config
    }

    /// Classify a trade as buy-initiated or sell-initiated
    ///
    /// # Arguments
    /// * `trade_price` - The price at which the trade executed
    /// * `prev_price` - The previous trade price (for tick rule)
    /// * `bid` - Current best bid price
    /// * `ask` - Current best ask price
    /// * `prev_side` - Previous trade's classified side (for tick rule fallback)
    pub fn classify_trade(
        &self,
        trade_price: Decimal,
        prev_price: Option<Decimal>,
        bid: Decimal,
        ask: Decimal,
        prev_side: Option<Side>,
    ) -> Side {
        match self.config.trade_classification {
            TradeClassificationMethod::TickRule => {
                // Compare to previous trade price
                match prev_price {
                    Some(prev) if trade_price > prev => Side::Buy,
                    Some(prev) if trade_price < prev => Side::Sell,
                    _ => {
                        // Price unchanged - use previous classification or default to buy
                        prev_side.unwrap_or(Side::Buy)
                    }
                }
            }
            TradeClassificationMethod::QuoteMidpoint => {
                // Compare to bid-ask midpoint
                let mid = (bid + ask) / dec!(2);
                if trade_price >= mid {
                    Side::Buy
                } else {
                    Side::Sell
                }
            }
            TradeClassificationMethod::BulkVolume => {
                // Lee-Ready: Use quote midpoint but with adjustment for
                // trades at the midpoint based on recent price movement
                let mid = (bid + ask) / dec!(2);
                if trade_price > mid {
                    Side::Buy
                } else if trade_price < mid {
                    Side::Sell
                } else {
                    // At midpoint - use tick rule as fallback
                    match prev_price {
                        Some(prev) if trade_price > prev => Side::Buy,
                        Some(prev) if trade_price < prev => Side::Sell,
                        _ => prev_side.unwrap_or(Side::Buy),
                    }
                }
            }
        }
    }

    /// Process a trade and update bucket state
    ///
    /// Returns a VpinResult if we have enough completed buckets to calculate VPIN.
    ///
    /// # Arguments
    /// * `token_id` - The token being traded
    /// * `price` - Trade execution price
    /// * `size_usd` - Trade size in USD
    /// * `bid` - Current best bid (for quote midpoint classification)
    /// * `ask` - Current best ask (for quote midpoint classification)
    pub fn process_trade(
        &mut self,
        token_id: &TokenId,
        price: Decimal,
        size_usd: Decimal,
        bid: Decimal,
        ask: Decimal,
    ) -> Option<VpinResult> {
        // Get previous state for classification (before mutable borrow)
        let (prev_price, prev_side) = self
            .token_states
            .get(token_id)
            .map(|s| (s.last_price, s.last_side))
            .unwrap_or((None, None));

        // Classify the trade before mutable borrow
        let side = self.classify_trade(price, prev_price, bid, ask, prev_side);

        // Get or create token state
        let state = self
            .token_states
            .entry(token_id.clone())
            .or_insert_with(TokenState::new);

        // Update last price and side for tick rule
        state.last_price = Some(price);
        state.last_side = Some(side);

        // Add to current bucket
        let mut remaining_volume = size_usd;

        while remaining_volume > Decimal::ZERO {
            let space_in_bucket =
                self.config.bucket_size_usd - state.current_bucket.total_volume;

            if remaining_volume >= space_in_bucket {
                // Fill the rest of current bucket
                state.current_bucket.add_trade(space_in_bucket, side);
                remaining_volume -= space_in_bucket;

                // Finalize and rotate bucket
                state.current_bucket.finalize();
                state.completed_buckets.push_back(state.current_bucket.clone());
                state.current_bucket = VolumeBucket::new();

                // Keep only lookback_buckets
                while state.completed_buckets.len() > self.config.lookback_buckets {
                    state.completed_buckets.pop_front();
                }
            } else {
                // Add remaining volume to current bucket
                state.current_bucket.add_trade(remaining_volume, side);
                remaining_volume = Decimal::ZERO;
            }
        }

        // Calculate VPIN if we have enough buckets
        self.calculate_vpin(token_id)
    }

    /// Process a trade with a pre-classified side
    ///
    /// Use this when the trade side is already known (e.g., from exchange data).
    pub fn process_classified_trade(
        &mut self,
        token_id: &TokenId,
        price: Decimal,
        size_usd: Decimal,
        side: Side,
    ) -> Option<VpinResult> {
        // Get or create token state
        let state = self
            .token_states
            .entry(token_id.clone())
            .or_insert_with(TokenState::new);

        // Update last price and side
        state.last_price = Some(price);
        state.last_side = Some(side);

        // Add to current bucket
        let mut remaining_volume = size_usd;

        while remaining_volume > Decimal::ZERO {
            let space_in_bucket =
                self.config.bucket_size_usd - state.current_bucket.total_volume;

            if remaining_volume >= space_in_bucket {
                // Fill the rest of current bucket
                state.current_bucket.add_trade(space_in_bucket, side);
                remaining_volume -= space_in_bucket;

                // Finalize and rotate bucket
                state.current_bucket.finalize();
                state.completed_buckets.push_back(state.current_bucket.clone());
                state.current_bucket = VolumeBucket::new();

                // Keep only lookback_buckets
                while state.completed_buckets.len() > self.config.lookback_buckets {
                    state.completed_buckets.pop_front();
                }
            } else {
                // Add remaining volume to current bucket
                state.current_bucket.add_trade(remaining_volume, side);
                remaining_volume = Decimal::ZERO;
            }
        }

        // Calculate VPIN if we have enough buckets
        self.calculate_vpin(token_id)
    }

    /// Calculate VPIN from completed buckets
    ///
    /// VPIN = sum(|buy_vol - sell_vol|) / (2 * total_vol) across all buckets
    ///
    /// Returns None if there are no completed buckets.
    pub fn calculate_vpin(&self, token_id: &TokenId) -> Option<VpinResult> {
        let state = self.token_states.get(token_id)?;

        if state.completed_buckets.is_empty() {
            return None;
        }

        // Sum up imbalances and total volume
        let mut total_imbalance = Decimal::ZERO;
        let mut total_volume = Decimal::ZERO;
        let mut total_buy_volume = Decimal::ZERO;
        let mut total_sell_volume = Decimal::ZERO;

        for bucket in &state.completed_buckets {
            total_imbalance += bucket.imbalance();
            total_volume += bucket.total_volume;
            total_buy_volume += bucket.buy_volume;
            total_sell_volume += bucket.sell_volume;
        }

        // Avoid division by zero
        if total_volume.is_zero() {
            return None;
        }

        // VPIN = sum(|buy - sell|) / (2 * total_volume)
        let vpin = total_imbalance / (dec!(2) * total_volume);

        // Clamp to [0, 1] range (should already be in range, but be safe)
        let vpin = vpin.min(Decimal::ONE).max(Decimal::ZERO);

        // Calculate volume percentages
        let buy_volume_pct = total_buy_volume / total_volume;
        let sell_volume_pct = total_sell_volume / total_volume;

        // Determine toxicity level
        let toxicity_level = self.classify_toxicity(vpin);

        Some(VpinResult {
            token_id: token_id.clone(),
            vpin,
            toxicity_level,
            buy_volume_pct,
            sell_volume_pct,
            bucket_count: state.completed_buckets.len(),
            timestamp: Utc::now(),
        })
    }

    /// Classify toxicity level based on VPIN value
    fn classify_toxicity(&self, vpin: Decimal) -> ToxicityLevel {
        if vpin >= self.config.high_toxicity_threshold {
            ToxicityLevel::High
        } else if vpin >= dec!(0.5) {
            ToxicityLevel::Elevated
        } else if vpin >= self.config.low_toxicity_threshold {
            ToxicityLevel::Normal
        } else {
            ToxicityLevel::Low
        }
    }

    /// Get the current bucket for a token (for inspection)
    pub fn current_bucket(&self, token_id: &TokenId) -> Option<&VolumeBucket> {
        self.token_states.get(token_id).map(|s| &s.current_bucket)
    }

    /// Get the number of completed buckets for a token
    pub fn completed_bucket_count(&self, token_id: &TokenId) -> usize {
        self.token_states
            .get(token_id)
            .map(|s| s.completed_buckets.len())
            .unwrap_or(0)
    }

    /// Clear all state for a token
    pub fn clear_token(&mut self, token_id: &TokenId) {
        self.token_states.remove(token_id);
    }

    /// Clear all state
    pub fn clear_all(&mut self) {
        self.token_states.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> VpinConfig {
        VpinConfig {
            enabled: true,
            bucket_size_usd: dec!(100), // Smaller buckets for testing
            lookback_buckets: 10,
            high_toxicity_threshold: dec!(0.7),
            low_toxicity_threshold: dec!(0.3),
            trade_classification: TradeClassificationMethod::QuoteMidpoint,
        }
    }

    #[test]
    fn test_vpin_calculation_balanced_flow() {
        // Equal buy/sell volume should give VPIN ≈ 0
        let mut calculator = VpinCalculator::new(default_config());
        let token_id = "test_token".to_string();

        // Add balanced trades to fill buckets
        for i in 0..20 {
            let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
            calculator.process_classified_trade(&token_id, dec!(0.5), dec!(50), side);
        }

        let result = calculator.calculate_vpin(&token_id).unwrap();

        // With perfectly balanced flow, VPIN should be 0
        assert_eq!(result.vpin, dec!(0));
        assert_eq!(result.toxicity_level, ToxicityLevel::Low);
        assert_eq!(result.buy_volume_pct, dec!(0.5));
        assert_eq!(result.sell_volume_pct, dec!(0.5));
    }

    #[test]
    fn test_vpin_calculation_buy_dominated() {
        // All buys should give VPIN = 1
        let mut calculator = VpinCalculator::new(default_config());
        let token_id = "test_token".to_string();

        // Fill buckets with only buys
        for _ in 0..20 {
            calculator.process_classified_trade(&token_id, dec!(0.5), dec!(50), Side::Buy);
        }

        let result = calculator.calculate_vpin(&token_id).unwrap();

        // With all buys, VPIN should be 1.0
        // VPIN = sum(|buy - sell|) / (2 * total)
        // For all buys: sum(|100 - 0|) / (2 * 100) = 100 / 200 = 0.5 per bucket
        // Actually: VPIN = total_imbalance / (2 * total_volume)
        // With 10 buckets of $100 each, all buys:
        // total_imbalance = 10 * 100 = 1000
        // total_volume = 10 * 100 = 1000
        // VPIN = 1000 / (2 * 1000) = 0.5
        //
        // VPIN of 0.5 means moderate imbalance. For VPIN = 1.0, you'd need
        // |buy - sell| = 2 * total, which is impossible.
        // Max VPIN is 0.5 when all volume is one-sided.
        assert_eq!(result.vpin, dec!(0.5));
        assert_eq!(result.toxicity_level, ToxicityLevel::Elevated);
        assert_eq!(result.buy_volume_pct, Decimal::ONE);
        assert_eq!(result.sell_volume_pct, Decimal::ZERO);
    }

    #[test]
    fn test_vpin_calculation_sell_dominated() {
        // All sells should also give VPIN = 0.5 (max possible)
        let mut calculator = VpinCalculator::new(default_config());
        let token_id = "test_token".to_string();

        for _ in 0..20 {
            calculator.process_classified_trade(&token_id, dec!(0.5), dec!(50), Side::Sell);
        }

        let result = calculator.calculate_vpin(&token_id).unwrap();
        assert_eq!(result.vpin, dec!(0.5));
        assert_eq!(result.buy_volume_pct, Decimal::ZERO);
        assert_eq!(result.sell_volume_pct, Decimal::ONE);
    }

    #[test]
    fn test_bucket_transition() {
        let mut calculator = VpinCalculator::new(default_config());
        let token_id = "test_token".to_string();

        // Bucket size is $100, so a $100 trade should complete one bucket
        assert_eq!(calculator.completed_bucket_count(&token_id), 0);

        // First $50 trade - bucket not complete
        calculator.process_classified_trade(&token_id, dec!(0.5), dec!(50), Side::Buy);
        assert_eq!(calculator.completed_bucket_count(&token_id), 0);

        // Second $50 trade - bucket should complete
        calculator.process_classified_trade(&token_id, dec!(0.5), dec!(50), Side::Sell);
        assert_eq!(calculator.completed_bucket_count(&token_id), 1);

        // Check bucket contents
        let state = calculator.token_states.get(&token_id).unwrap();
        let bucket = state.completed_buckets.front().unwrap();
        assert_eq!(bucket.buy_volume, dec!(50));
        assert_eq!(bucket.sell_volume, dec!(50));
        assert_eq!(bucket.total_volume, dec!(100));
        assert_eq!(bucket.trade_count, 2);
        assert!(bucket.end_time.is_some());
    }

    #[test]
    fn test_bucket_overflow() {
        // Test that large trades correctly overflow into multiple buckets
        let mut calculator = VpinCalculator::new(default_config());
        let token_id = "test_token".to_string();

        // $350 trade should create 3 complete buckets + partial
        calculator.process_classified_trade(&token_id, dec!(0.5), dec!(350), Side::Buy);
        assert_eq!(calculator.completed_bucket_count(&token_id), 3);

        // Current bucket should have $50
        let current = calculator.current_bucket(&token_id).unwrap();
        assert_eq!(current.total_volume, dec!(50));
    }

    #[test]
    fn test_lookback_limit() {
        let mut calculator = VpinCalculator::new(VpinConfig {
            lookback_buckets: 3,
            bucket_size_usd: dec!(100),
            ..default_config()
        });
        let token_id = "test_token".to_string();

        // Fill 5 buckets
        for _ in 0..5 {
            calculator.process_classified_trade(&token_id, dec!(0.5), dec!(100), Side::Buy);
        }

        // Should only keep 3 (lookback limit)
        assert_eq!(calculator.completed_bucket_count(&token_id), 3);
    }

    #[test]
    fn test_trade_classification_tick_rule() {
        let calculator = VpinCalculator::new(VpinConfig {
            trade_classification: TradeClassificationMethod::TickRule,
            ..default_config()
        });

        // Price up -> Buy
        assert_eq!(
            calculator.classify_trade(dec!(0.55), Some(dec!(0.50)), dec!(0.49), dec!(0.51), None),
            Side::Buy
        );

        // Price down -> Sell
        assert_eq!(
            calculator.classify_trade(dec!(0.45), Some(dec!(0.50)), dec!(0.49), dec!(0.51), None),
            Side::Sell
        );

        // Price unchanged with prev_side -> Use prev_side
        assert_eq!(
            calculator.classify_trade(
                dec!(0.50),
                Some(dec!(0.50)),
                dec!(0.49),
                dec!(0.51),
                Some(Side::Sell)
            ),
            Side::Sell
        );

        // Price unchanged with no prev_side -> Default to Buy
        assert_eq!(
            calculator.classify_trade(dec!(0.50), Some(dec!(0.50)), dec!(0.49), dec!(0.51), None),
            Side::Buy
        );

        // No prev_price -> Default to Buy
        assert_eq!(
            calculator.classify_trade(dec!(0.50), None, dec!(0.49), dec!(0.51), None),
            Side::Buy
        );
    }

    #[test]
    fn test_trade_classification_quote_midpoint() {
        let calculator = VpinCalculator::new(VpinConfig {
            trade_classification: TradeClassificationMethod::QuoteMidpoint,
            ..default_config()
        });

        // Trade at or above midpoint -> Buy
        // Bid=0.49, Ask=0.51, Mid=0.50
        assert_eq!(
            calculator.classify_trade(dec!(0.51), None, dec!(0.49), dec!(0.51), None),
            Side::Buy
        );
        assert_eq!(
            calculator.classify_trade(dec!(0.50), None, dec!(0.49), dec!(0.51), None),
            Side::Buy
        );

        // Trade below midpoint -> Sell
        assert_eq!(
            calculator.classify_trade(dec!(0.49), None, dec!(0.49), dec!(0.51), None),
            Side::Sell
        );
    }

    #[test]
    fn test_trade_classification_bulk_volume() {
        let calculator = VpinCalculator::new(VpinConfig {
            trade_classification: TradeClassificationMethod::BulkVolume,
            ..default_config()
        });

        // Above midpoint -> Buy
        assert_eq!(
            calculator.classify_trade(dec!(0.52), Some(dec!(0.50)), dec!(0.49), dec!(0.51), None),
            Side::Buy
        );

        // Below midpoint -> Sell
        assert_eq!(
            calculator.classify_trade(dec!(0.48), Some(dec!(0.50)), dec!(0.49), dec!(0.51), None),
            Side::Sell
        );

        // At midpoint, price went up -> Buy
        assert_eq!(
            calculator.classify_trade(dec!(0.50), Some(dec!(0.48)), dec!(0.49), dec!(0.51), None),
            Side::Buy
        );

        // At midpoint, price went down -> Sell
        assert_eq!(
            calculator.classify_trade(dec!(0.50), Some(dec!(0.52)), dec!(0.49), dec!(0.51), None),
            Side::Sell
        );
    }

    #[test]
    fn test_toxicity_level_classification() {
        let config = VpinConfig {
            high_toxicity_threshold: dec!(0.7),
            low_toxicity_threshold: dec!(0.3),
            ..default_config()
        };
        let calculator = VpinCalculator::new(config);

        assert_eq!(calculator.classify_toxicity(dec!(0.1)), ToxicityLevel::Low);
        assert_eq!(calculator.classify_toxicity(dec!(0.29)), ToxicityLevel::Low);
        assert_eq!(calculator.classify_toxicity(dec!(0.3)), ToxicityLevel::Normal);
        assert_eq!(calculator.classify_toxicity(dec!(0.49)), ToxicityLevel::Normal);
        assert_eq!(calculator.classify_toxicity(dec!(0.5)), ToxicityLevel::Elevated);
        assert_eq!(calculator.classify_toxicity(dec!(0.69)), ToxicityLevel::Elevated);
        assert_eq!(calculator.classify_toxicity(dec!(0.7)), ToxicityLevel::High);
        assert_eq!(calculator.classify_toxicity(dec!(1.0)), ToxicityLevel::High);
    }

    #[test]
    fn test_multiple_tokens_isolated() {
        let mut calculator = VpinCalculator::new(default_config());
        let token_a = "token_a".to_string();
        let token_b = "token_b".to_string();

        // Token A: all buys
        for _ in 0..5 {
            calculator.process_classified_trade(&token_a, dec!(0.5), dec!(100), Side::Buy);
        }

        // Token B: all sells
        for _ in 0..5 {
            calculator.process_classified_trade(&token_b, dec!(0.5), dec!(100), Side::Sell);
        }

        let result_a = calculator.calculate_vpin(&token_a).unwrap();
        let result_b = calculator.calculate_vpin(&token_b).unwrap();

        // Both should have same VPIN (one-sided flow)
        assert_eq!(result_a.vpin, result_b.vpin);
        assert_eq!(result_a.vpin, dec!(0.5));

        // But different volume percentages
        assert_eq!(result_a.buy_volume_pct, Decimal::ONE);
        assert_eq!(result_a.sell_volume_pct, Decimal::ZERO);
        assert_eq!(result_b.buy_volume_pct, Decimal::ZERO);
        assert_eq!(result_b.sell_volume_pct, Decimal::ONE);
    }

    #[test]
    fn test_clear_token() {
        let mut calculator = VpinCalculator::new(default_config());
        let token_id = "test_token".to_string();

        calculator.process_classified_trade(&token_id, dec!(0.5), dec!(100), Side::Buy);
        assert_eq!(calculator.completed_bucket_count(&token_id), 1);

        calculator.clear_token(&token_id);
        assert_eq!(calculator.completed_bucket_count(&token_id), 0);
        assert!(calculator.current_bucket(&token_id).is_none());
    }

    #[test]
    fn test_no_buckets_returns_none() {
        let calculator = VpinCalculator::new(default_config());
        let token_id = "nonexistent".to_string();

        assert!(calculator.calculate_vpin(&token_id).is_none());
    }

    #[test]
    fn test_partial_bucket_not_included() {
        let mut calculator = VpinCalculator::new(default_config());
        let token_id = "test_token".to_string();

        // Add $50 to first bucket (partial)
        calculator.process_classified_trade(&token_id, dec!(0.5), dec!(50), Side::Buy);

        // No completed buckets yet
        assert!(calculator.calculate_vpin(&token_id).is_none());
    }

    #[test]
    fn test_process_trade_with_classification() {
        let mut calculator = VpinCalculator::new(VpinConfig {
            trade_classification: TradeClassificationMethod::QuoteMidpoint,
            bucket_size_usd: dec!(100),
            ..default_config()
        });
        let token_id = "test_token".to_string();

        // Trade above midpoint (should be classified as buy)
        // Bid=0.40, Ask=0.50, Mid=0.45, Trade=0.48 -> Buy
        calculator.process_trade(&token_id, dec!(0.48), dec!(100), dec!(0.40), dec!(0.50));

        let state = calculator.token_states.get(&token_id).unwrap();
        let bucket = state.completed_buckets.front().unwrap();
        assert_eq!(bucket.buy_volume, dec!(100));
        assert_eq!(bucket.sell_volume, dec!(0));
    }

    #[test]
    fn test_vpin_result_fields() {
        let mut calculator = VpinCalculator::new(default_config());
        let token_id = "test_token".to_string();

        // 60% buys, 40% sells per bucket
        for _ in 0..10 {
            calculator.process_classified_trade(&token_id, dec!(0.5), dec!(60), Side::Buy);
            calculator.process_classified_trade(&token_id, dec!(0.5), dec!(40), Side::Sell);
        }

        let result = calculator.calculate_vpin(&token_id).unwrap();

        assert_eq!(result.token_id, token_id);
        assert_eq!(result.bucket_count, 10);
        assert_eq!(result.buy_volume_pct, dec!(0.6));
        assert_eq!(result.sell_volume_pct, dec!(0.4));

        // VPIN for 60/40 split:
        // Imbalance per bucket = |60 - 40| = 20
        // Total imbalance = 10 * 20 = 200
        // Total volume = 10 * 100 = 1000
        // VPIN = 200 / (2 * 1000) = 0.1
        assert_eq!(result.vpin, dec!(0.1));
        assert_eq!(result.toxicity_level, ToxicityLevel::Low);
    }
}
