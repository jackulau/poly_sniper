//! Whale Detection and Large Trade Tracking
//!
//! Identifies and tracks large trades and wallet addresses exhibiting significant
//! trading activity, providing leading indicators for position management.

use chrono::{DateTime, Duration, Utc};
use polysniper_core::{MarketId, Side, TokenId};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tracing::{debug, info};

/// Whale detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleDetectorConfig {
    /// Whether whale detection is enabled
    pub enabled: bool,
    /// Single trade threshold in USD (e.g., $5000)
    pub whale_threshold_usd: Decimal,
    /// Cumulative threshold in USD (e.g., $10000)
    pub aggregate_threshold_usd: Decimal,
    /// Window for aggregation in seconds (e.g., 300s)
    pub aggregate_window_secs: u64,
    /// Whether to generate alerts on detection
    pub alert_on_detection: bool,
    /// Enable address profiling
    pub track_addresses: bool,
    /// Max trades per address to track
    #[serde(default = "default_address_history_limit")]
    pub address_history_limit: usize,
    /// Cooldown between same-address alerts in seconds
    #[serde(default = "default_cooldown_secs")]
    pub cooldown_secs: u64,
}

fn default_address_history_limit() -> usize {
    100
}

fn default_cooldown_secs() -> u64 {
    60
}

impl Default for WhaleDetectorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            whale_threshold_usd: dec!(5000),
            aggregate_threshold_usd: dec!(10000),
            aggregate_window_secs: 300,
            alert_on_detection: true,
            track_addresses: true,
            address_history_limit: default_address_history_limit(),
            cooldown_secs: default_cooldown_secs(),
        }
    }
}

/// A single whale trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleTrade {
    /// Unique trade identifier
    pub trade_id: String,
    /// Token ID traded
    pub token_id: TokenId,
    /// Market ID
    pub market_id: MarketId,
    /// Buy or sell
    pub side: Side,
    /// Trade size in USD
    pub size_usd: Decimal,
    /// Trade price
    pub price: Decimal,
    /// When the trade occurred
    pub timestamp: DateTime<Utc>,
    /// Wallet address if available
    pub address: Option<String>,
    /// Whether this met the single-trade threshold
    pub is_single_whale: bool,
    /// Cumulative context if applicable
    pub cumulative_context: Option<CumulativeContext>,
}

/// Context for cumulative whale activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CumulativeContext {
    /// Total volume in USD during the window
    pub total_volume_usd: Decimal,
    /// Number of trades in the window
    pub trade_count: u32,
    /// Start of the aggregation window
    pub window_start: DateTime<Utc>,
    /// Dominant side during the window
    pub dominant_side: Side,
    /// Buy volume as a ratio of total volume (0.0 to 1.0)
    pub side_ratio: Decimal,
}

/// Classification of a whale address based on trading patterns
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhaleClassification {
    /// Unknown pattern
    #[default]
    Unknown,
    /// Consistently buys
    Accumulator,
    /// Consistently sells
    Distributor,
    /// Quick in/out trades
    Flipper,
    /// High win rate trader
    Informed,
}

/// Profile of a whale address
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddressProfile {
    /// Wallet address
    pub address: String,
    /// When first observed
    pub first_seen: DateTime<Utc>,
    /// When last observed
    pub last_seen: DateTime<Utc>,
    /// Total volume traded in USD
    pub total_volume_usd: Decimal,
    /// Total number of trades
    pub trade_count: u32,
    /// Average trade size in USD
    pub avg_trade_size_usd: Decimal,
    /// Win rate if trackable
    pub win_rate: Option<Decimal>,
    /// Recent trades for analysis
    pub recent_trades: VecDeque<WhaleTrade>,
    /// Classification based on patterns
    pub classification: WhaleClassification,
}

impl AddressProfile {
    /// Create a new address profile
    pub fn new(address: &str) -> Self {
        let now = Utc::now();
        Self {
            address: address.to_string(),
            first_seen: now,
            last_seen: now,
            total_volume_usd: Decimal::ZERO,
            trade_count: 0,
            avg_trade_size_usd: Decimal::ZERO,
            win_rate: None,
            recent_trades: VecDeque::new(),
            classification: WhaleClassification::Unknown,
        }
    }

    /// Add a trade to the profile
    pub fn add_trade(&mut self, trade: &WhaleTrade, history_limit: usize) {
        self.last_seen = trade.timestamp;
        self.total_volume_usd += trade.size_usd;
        self.trade_count += 1;
        self.avg_trade_size_usd = self.total_volume_usd / Decimal::from(self.trade_count);

        self.recent_trades.push_back(trade.clone());
        while self.recent_trades.len() > history_limit {
            self.recent_trades.pop_front();
        }
    }

    /// Update classification based on trading patterns
    pub fn update_classification(&mut self) {
        if self.trade_count < 5 {
            self.classification = WhaleClassification::Unknown;
            return;
        }

        let mut buy_volume = Decimal::ZERO;
        let mut sell_volume = Decimal::ZERO;
        let mut quick_flips = 0u32;
        let mut last_side: Option<Side> = None;
        let mut last_timestamp: Option<DateTime<Utc>> = None;

        for trade in &self.recent_trades {
            match trade.side {
                Side::Buy => buy_volume += trade.size_usd,
                Side::Sell => sell_volume += trade.size_usd,
            }

            // Check for quick flips (opposite direction within 5 minutes)
            if let (Some(prev_side), Some(prev_time)) = (last_side, last_timestamp) {
                if prev_side != trade.side
                    && (trade.timestamp - prev_time).num_seconds() < 300
                {
                    quick_flips += 1;
                }
            }
            last_side = Some(trade.side);
            last_timestamp = Some(trade.timestamp);
        }

        let total = buy_volume + sell_volume;
        if total.is_zero() {
            self.classification = WhaleClassification::Unknown;
            return;
        }

        let buy_ratio = buy_volume / total;
        let flip_ratio = Decimal::from(quick_flips) / Decimal::from(self.recent_trades.len() as u32);

        // Classify based on patterns
        if flip_ratio > dec!(0.3) {
            self.classification = WhaleClassification::Flipper;
        } else if buy_ratio > dec!(0.7) {
            self.classification = WhaleClassification::Accumulator;
        } else if buy_ratio < dec!(0.3) {
            self.classification = WhaleClassification::Distributor;
        } else if let Some(win_rate) = self.win_rate {
            if win_rate > dec!(0.6) {
                self.classification = WhaleClassification::Informed;
            } else {
                self.classification = WhaleClassification::Unknown;
            }
        } else {
            self.classification = WhaleClassification::Unknown;
        }
    }
}

/// Type of whale alert
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhaleAlertType {
    /// Single large trade detected
    SingleLargeTrade,
    /// Cumulative activity threshold exceeded
    CumulativeActivity,
    /// Known whale address is active
    KnownWhaleActive,
    /// Whale changed trading direction
    WhaleReversal,
}

/// Recommended action based on whale activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WhaleAction {
    /// No action needed
    None,
    /// Reduce position by multiplier
    ReducePosition { multiplier: Decimal },
    /// Stop opening new trades
    HaltNewTrades,
    /// Follow the whale's direction
    FollowWhale { direction: Side },
    /// Generate alert only
    Alert,
}

/// Whale activity alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleAlert {
    /// Type of alert
    pub alert_type: WhaleAlertType,
    /// Token involved
    pub token_id: TokenId,
    /// Market involved
    pub market_id: MarketId,
    /// The whale trade that triggered this
    pub whale_trade: WhaleTrade,
    /// Recommended action
    pub recommended_action: WhaleAction,
    /// Confidence in the recommendation (0.0 to 1.0)
    pub confidence: Decimal,
    /// When the alert was generated
    pub timestamp: DateTime<Utc>,
}

/// Whale detector for monitoring large trades
pub struct WhaleDetector {
    config: WhaleDetectorConfig,
    /// Recent whale trades per token
    recent_whale_trades: HashMap<TokenId, VecDeque<WhaleTrade>>,
    /// Address profiles for tracked addresses
    address_profiles: HashMap<String, AddressProfile>,
    /// Last alert time per address (for cooldown)
    last_alerts: HashMap<String, DateTime<Utc>>,
}

impl WhaleDetector {
    /// Create a new whale detector with the given configuration
    pub fn new(config: WhaleDetectorConfig) -> Self {
        info!(
            threshold_usd = %config.whale_threshold_usd,
            aggregate_usd = %config.aggregate_threshold_usd,
            window_secs = %config.aggregate_window_secs,
            "Initializing whale detector"
        );
        Self {
            config,
            recent_whale_trades: HashMap::new(),
            address_profiles: HashMap::new(),
            last_alerts: HashMap::new(),
        }
    }

    /// Check if an address is in cooldown
    fn is_in_cooldown(&self, address: &str) -> bool {
        if let Some(last_alert) = self.last_alerts.get(address) {
            let cooldown_duration = Duration::seconds(self.config.cooldown_secs as i64);
            return Utc::now() < *last_alert + cooldown_duration;
        }
        false
    }

    /// Set cooldown for an address
    fn set_cooldown(&mut self, address: &str) {
        self.last_alerts.insert(address.to_string(), Utc::now());
    }

    /// Process a trade and check for whale activity
    #[allow(clippy::too_many_arguments)]
    pub fn process_trade(
        &mut self,
        token_id: &TokenId,
        market_id: &MarketId,
        side: Side,
        size_usd: Decimal,
        price: Decimal,
        address: Option<&str>,
        timestamp: DateTime<Utc>,
    ) -> Option<WhaleAlert> {
        if !self.config.enabled {
            return None;
        }

        let trade_id = format!(
            "wt_{}_{}_{:08x}",
            token_id,
            timestamp.timestamp_millis(),
            rand_suffix()
        );

        // Check for single whale trade
        let is_single_whale = size_usd >= self.config.whale_threshold_usd;

        if is_single_whale {
            debug!(
                token_id = %token_id,
                size_usd = %size_usd,
                threshold = %self.config.whale_threshold_usd,
                "Single whale trade detected"
            );
        }

        // Create the whale trade record
        let whale_trade = WhaleTrade {
            trade_id,
            token_id: token_id.clone(),
            market_id: market_id.clone(),
            side,
            size_usd,
            price,
            timestamp,
            address: address.map(|a| a.to_string()),
            is_single_whale,
            cumulative_context: None,
        };

        // Track the trade
        self.track_trade(&whale_trade);

        // Update address profile if tracking is enabled
        if self.config.track_addresses {
            if let Some(addr) = address {
                self.update_address_profile(addr, &whale_trade);
            }
        }

        // Check for cooldown
        if let Some(addr) = address {
            if self.is_in_cooldown(addr) {
                debug!(address = %addr, "Address in cooldown, skipping alert");
                return None;
            }
        }

        // Generate alert based on detection
        let alert = if is_single_whale {
            Some(self.create_single_whale_alert(&whale_trade))
        } else if let Some(cumulative) = self.check_cumulative_threshold(token_id) {
            // Update the trade with cumulative context
            let mut trade_with_context = whale_trade.clone();
            trade_with_context.cumulative_context = Some(cumulative.clone());
            Some(self.create_cumulative_alert(&trade_with_context, &cumulative))
        } else if let Some(addr) = address {
            // Check if this is a known whale
            if let Some(profile) = self.address_profiles.get(addr) {
                if profile.classification == WhaleClassification::Informed {
                    Some(self.create_known_whale_alert(&whale_trade, profile))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Set cooldown if alert was generated
        if alert.is_some() {
            if let Some(addr) = address {
                self.set_cooldown(addr);
            }
        }

        alert
    }

    /// Track a trade in the recent trades list
    fn track_trade(&mut self, trade: &WhaleTrade) {
        let trades = self
            .recent_whale_trades
            .entry(trade.token_id.clone())
            .or_default();

        trades.push_back(trade.clone());

        // Clean up old trades outside the window
        let window_start = Utc::now() - Duration::seconds(self.config.aggregate_window_secs as i64);
        while let Some(front) = trades.front() {
            if front.timestamp < window_start {
                trades.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get cumulative whale activity for a token within the window
    pub fn get_whale_activity(&self, token_id: &TokenId) -> Option<CumulativeContext> {
        let trades = self.recent_whale_trades.get(token_id)?;
        let now = Utc::now();
        let window_start = now - Duration::seconds(self.config.aggregate_window_secs as i64);

        let recent: Vec<_> = trades.iter().filter(|t| t.timestamp >= window_start).collect();

        if recent.is_empty() {
            return None;
        }

        let mut buy_volume = Decimal::ZERO;
        let mut sell_volume = Decimal::ZERO;

        for trade in &recent {
            match trade.side {
                Side::Buy => buy_volume += trade.size_usd,
                Side::Sell => sell_volume += trade.size_usd,
            }
        }

        let total_volume = buy_volume + sell_volume;
        let trade_count = recent.len() as u32;
        let dominant_side = if buy_volume >= sell_volume {
            Side::Buy
        } else {
            Side::Sell
        };
        let side_ratio = if total_volume.is_zero() {
            dec!(0.5)
        } else {
            buy_volume / total_volume
        };

        Some(CumulativeContext {
            total_volume_usd: total_volume,
            trade_count,
            window_start,
            dominant_side,
            side_ratio,
        })
    }

    /// Check if cumulative activity exceeds the threshold
    fn check_cumulative_threshold(&self, token_id: &TokenId) -> Option<CumulativeContext> {
        let activity = self.get_whale_activity(token_id)?;

        if activity.total_volume_usd >= self.config.aggregate_threshold_usd {
            info!(
                token_id = %token_id,
                total_volume = %activity.total_volume_usd,
                threshold = %self.config.aggregate_threshold_usd,
                trade_count = %activity.trade_count,
                "Cumulative whale activity threshold exceeded"
            );
            Some(activity)
        } else {
            None
        }
    }

    /// Update address profile with a new trade
    fn update_address_profile(&mut self, address: &str, trade: &WhaleTrade) {
        let history_limit = self.config.address_history_limit;
        let profile = self
            .address_profiles
            .entry(address.to_string())
            .or_insert_with(|| AddressProfile::new(address));

        profile.add_trade(trade, history_limit);
        profile.update_classification();

        debug!(
            address = %address,
            classification = ?profile.classification,
            total_volume = %profile.total_volume_usd,
            trade_count = %profile.trade_count,
            "Updated address profile"
        );
    }

    /// Create an alert for a single large trade
    fn create_single_whale_alert(&self, trade: &WhaleTrade) -> WhaleAlert {
        let confidence = self.calculate_single_trade_confidence(trade);
        let recommended_action = self.determine_action_for_single_trade(trade, confidence);

        info!(
            token_id = %trade.token_id,
            size_usd = %trade.size_usd,
            side = ?trade.side,
            confidence = %confidence,
            "Single whale trade alert"
        );

        WhaleAlert {
            alert_type: WhaleAlertType::SingleLargeTrade,
            token_id: trade.token_id.clone(),
            market_id: trade.market_id.clone(),
            whale_trade: trade.clone(),
            recommended_action,
            confidence,
            timestamp: Utc::now(),
        }
    }

    /// Create an alert for cumulative activity
    fn create_cumulative_alert(
        &self,
        trade: &WhaleTrade,
        context: &CumulativeContext,
    ) -> WhaleAlert {
        let confidence = self.calculate_cumulative_confidence(context);
        let recommended_action = self.determine_action_for_cumulative(context, confidence);

        info!(
            token_id = %trade.token_id,
            total_volume = %context.total_volume_usd,
            trade_count = %context.trade_count,
            dominant_side = ?context.dominant_side,
            confidence = %confidence,
            "Cumulative whale activity alert"
        );

        WhaleAlert {
            alert_type: WhaleAlertType::CumulativeActivity,
            token_id: trade.token_id.clone(),
            market_id: trade.market_id.clone(),
            whale_trade: trade.clone(),
            recommended_action,
            confidence,
            timestamp: Utc::now(),
        }
    }

    /// Create an alert for known whale activity
    fn create_known_whale_alert(&self, trade: &WhaleTrade, profile: &AddressProfile) -> WhaleAlert {
        let confidence = profile.win_rate.unwrap_or(dec!(0.6));
        let recommended_action = WhaleAction::FollowWhale {
            direction: trade.side,
        };

        info!(
            token_id = %trade.token_id,
            address = %profile.address,
            classification = ?profile.classification,
            confidence = %confidence,
            "Known whale active alert"
        );

        WhaleAlert {
            alert_type: WhaleAlertType::KnownWhaleActive,
            token_id: trade.token_id.clone(),
            market_id: trade.market_id.clone(),
            whale_trade: trade.clone(),
            recommended_action,
            confidence,
            timestamp: Utc::now(),
        }
    }

    /// Calculate confidence for a single large trade
    fn calculate_single_trade_confidence(&self, trade: &WhaleTrade) -> Decimal {
        // Higher confidence for larger trades relative to threshold
        let ratio = trade.size_usd / self.config.whale_threshold_usd;
        let base_confidence = if ratio > dec!(3) {
            dec!(0.9)
        } else if ratio > dec!(2) {
            dec!(0.8)
        } else if ratio > dec!(1.5) {
            dec!(0.7)
        } else {
            dec!(0.6)
        };

        // Boost if we know the address
        if let Some(addr) = &trade.address {
            if let Some(profile) = self.address_profiles.get(addr) {
                if profile.classification == WhaleClassification::Informed {
                    return base_confidence + dec!(0.1);
                }
            }
        }

        base_confidence
    }

    /// Calculate confidence for cumulative activity
    fn calculate_cumulative_confidence(&self, context: &CumulativeContext) -> Decimal {
        // Higher confidence for more one-sided activity
        let side_bias = if context.side_ratio > dec!(0.5) {
            context.side_ratio
        } else {
            Decimal::ONE - context.side_ratio
        };

        // Base on number of trades and bias
        let trade_factor = if context.trade_count > 10 {
            dec!(0.2)
        } else if context.trade_count > 5 {
            dec!(0.1)
        } else {
            dec!(0)
        };

        let volume_ratio = context.total_volume_usd / self.config.aggregate_threshold_usd;
        let volume_factor = if volume_ratio > dec!(2) {
            dec!(0.1)
        } else {
            dec!(0)
        };

        (side_bias * dec!(0.7) + trade_factor + volume_factor).min(dec!(0.95))
    }

    /// Determine action for a single large trade
    fn determine_action_for_single_trade(
        &self,
        trade: &WhaleTrade,
        confidence: Decimal,
    ) -> WhaleAction {
        if !self.config.alert_on_detection {
            return WhaleAction::None;
        }

        // For very large trades, suggest reducing position
        let ratio = trade.size_usd / self.config.whale_threshold_usd;
        if ratio > dec!(3) && confidence > dec!(0.7) {
            WhaleAction::ReducePosition {
                multiplier: dec!(0.5),
            }
        } else if ratio > dec!(2) && confidence > dec!(0.6) {
            WhaleAction::ReducePosition {
                multiplier: dec!(0.7),
            }
        } else {
            WhaleAction::Alert
        }
    }

    /// Determine action for cumulative activity
    fn determine_action_for_cumulative(
        &self,
        context: &CumulativeContext,
        confidence: Decimal,
    ) -> WhaleAction {
        if !self.config.alert_on_detection {
            return WhaleAction::None;
        }

        let volume_ratio = context.total_volume_usd / self.config.aggregate_threshold_usd;
        let side_bias = if context.side_ratio > dec!(0.5) {
            context.side_ratio
        } else {
            Decimal::ONE - context.side_ratio
        };

        if volume_ratio > dec!(2) && side_bias > dec!(0.8) && confidence > dec!(0.7) {
            WhaleAction::FollowWhale {
                direction: context.dominant_side,
            }
        } else if volume_ratio > dec!(1.5) && confidence > dec!(0.6) {
            WhaleAction::ReducePosition {
                multiplier: dec!(0.7),
            }
        } else {
            WhaleAction::Alert
        }
    }

    /// Get the address profile for a given address
    pub fn get_address_profile(&self, address: &str) -> Option<&AddressProfile> {
        self.address_profiles.get(address)
    }

    /// Get all known whale addresses
    pub fn get_known_whales(&self) -> Vec<&AddressProfile> {
        self.address_profiles
            .values()
            .filter(|p| {
                p.classification != WhaleClassification::Unknown
                    && p.total_volume_usd >= self.config.whale_threshold_usd
            })
            .collect()
    }

    /// Clean up old data outside the aggregation window
    pub fn cleanup_stale_data(&mut self) {
        let window_start = Utc::now() - Duration::seconds(self.config.aggregate_window_secs as i64);

        for trades in self.recent_whale_trades.values_mut() {
            while let Some(front) = trades.front() {
                if front.timestamp < window_start {
                    trades.pop_front();
                } else {
                    break;
                }
            }
        }

        // Clean up empty token entries
        self.recent_whale_trades.retain(|_, trades| !trades.is_empty());

        // Clean up old cooldowns
        let cooldown_duration = Duration::seconds(self.config.cooldown_secs as i64);
        let cutoff = Utc::now() - cooldown_duration;
        self.last_alerts.retain(|_, time| *time > cutoff);
    }

    /// Check if detector is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the current configuration
    pub fn config(&self) -> &WhaleDetectorConfig {
        &self.config
    }

    /// Reload configuration
    pub fn reload_config(&mut self, config: WhaleDetectorConfig) {
        info!(
            old_threshold = %self.config.whale_threshold_usd,
            new_threshold = %config.whale_threshold_usd,
            "Reloading whale detector config"
        );
        self.config = config;
    }
}

fn rand_suffix() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> WhaleDetectorConfig {
        WhaleDetectorConfig {
            enabled: true,
            whale_threshold_usd: dec!(5000),
            aggregate_threshold_usd: dec!(10000),
            aggregate_window_secs: 300,
            alert_on_detection: true,
            track_addresses: true,
            address_history_limit: 100,
            cooldown_secs: 60,
        }
    }

    #[test]
    fn test_single_whale_detection() {
        let config = default_config();
        let mut detector = WhaleDetector::new(config);

        // Trade above threshold should trigger alert
        let alert = detector.process_trade(
            &"token_yes".to_string(),
            &"market_1".to_string(),
            Side::Buy,
            dec!(6000), // Above $5000 threshold
            dec!(0.50),
            None,
            Utc::now(),
        );

        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.alert_type, WhaleAlertType::SingleLargeTrade);
        assert!(alert.whale_trade.is_single_whale);
        assert!(alert.confidence >= dec!(0.6));
    }

    #[test]
    fn test_no_alert_below_threshold() {
        let config = default_config();
        let mut detector = WhaleDetector::new(config);

        // Trade below threshold should not trigger single whale alert
        // Also below cumulative threshold with just one trade
        let alert = detector.process_trade(
            &"token_yes".to_string(),
            &"market_1".to_string(),
            Side::Buy,
            dec!(1000), // Below $5000 threshold
            dec!(0.50),
            None,
            Utc::now(),
        );

        assert!(alert.is_none());
    }

    #[test]
    fn test_cumulative_whale_detection() {
        let config = WhaleDetectorConfig {
            whale_threshold_usd: dec!(5000),
            aggregate_threshold_usd: dec!(8000),
            ..default_config()
        };
        let mut detector = WhaleDetector::new(config);

        let token_id = "token_yes".to_string();
        let market_id = "market_1".to_string();

        // First trade - below single threshold but contributes to cumulative
        let alert1 = detector.process_trade(
            &token_id,
            &market_id,
            Side::Buy,
            dec!(3000),
            dec!(0.50),
            None,
            Utc::now(),
        );
        assert!(alert1.is_none());

        // Second trade - still below single but now cumulative exceeds threshold
        let alert2 = detector.process_trade(
            &token_id,
            &market_id,
            Side::Buy,
            dec!(3000),
            dec!(0.50),
            None,
            Utc::now(),
        );
        assert!(alert2.is_none()); // 6000 < 8000

        // Third trade - cumulative now exceeds aggregate threshold
        let alert3 = detector.process_trade(
            &token_id,
            &market_id,
            Side::Buy,
            dec!(3000),
            dec!(0.50),
            None,
            Utc::now(),
        );
        assert!(alert3.is_some()); // 9000 >= 8000

        let alert = alert3.unwrap();
        assert_eq!(alert.alert_type, WhaleAlertType::CumulativeActivity);
        assert!(alert.whale_trade.cumulative_context.is_some());

        let context = alert.whale_trade.cumulative_context.unwrap();
        assert_eq!(context.total_volume_usd, dec!(9000));
        assert_eq!(context.trade_count, 3);
        assert_eq!(context.dominant_side, Side::Buy);
    }

    #[test]
    fn test_address_profiling() {
        let config = default_config();
        let mut detector = WhaleDetector::new(config);

        let address = "0xwhale123";
        let token_id = "token_yes".to_string();
        let market_id = "market_1".to_string();

        // Add several buy trades
        for _ in 0..10 {
            detector.process_trade(
                &token_id,
                &market_id,
                Side::Buy,
                dec!(1000),
                dec!(0.50),
                Some(address),
                Utc::now(),
            );
        }

        let profile = detector.get_address_profile(address);
        assert!(profile.is_some());

        let profile = profile.unwrap();
        assert_eq!(profile.address, address);
        assert_eq!(profile.trade_count, 10);
        assert_eq!(profile.total_volume_usd, dec!(10000));
        assert_eq!(profile.avg_trade_size_usd, dec!(1000));
        // With 100% buys, should be classified as Accumulator
        assert_eq!(profile.classification, WhaleClassification::Accumulator);
    }

    #[test]
    fn test_address_classification_distributor() {
        let config = default_config();
        let mut detector = WhaleDetector::new(config);

        let address = "0xseller123";
        let token_id = "token_yes".to_string();
        let market_id = "market_1".to_string();

        // Add mostly sell trades
        for _ in 0..10 {
            detector.process_trade(
                &token_id,
                &market_id,
                Side::Sell,
                dec!(1000),
                dec!(0.50),
                Some(address),
                Utc::now(),
            );
        }

        let profile = detector.get_address_profile(address).unwrap();
        assert_eq!(profile.classification, WhaleClassification::Distributor);
    }

    #[test]
    fn test_address_classification_flipper() {
        let config = default_config();
        let mut detector = WhaleDetector::new(config);

        let address = "0xflipper123";
        let token_id = "token_yes".to_string();
        let market_id = "market_1".to_string();
        let now = Utc::now();

        // Add alternating buy/sell trades within short time periods
        for i in 0..10 {
            let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
            // Each trade is 30 seconds apart (within 5 min flip window)
            let timestamp = now + Duration::seconds(i * 30);
            detector.process_trade(
                &token_id,
                &market_id,
                side,
                dec!(1000),
                dec!(0.50),
                Some(address),
                timestamp,
            );
        }

        let profile = detector.get_address_profile(address).unwrap();
        assert_eq!(profile.classification, WhaleClassification::Flipper);
    }

    #[test]
    fn test_cooldown_behavior() {
        let config = WhaleDetectorConfig {
            cooldown_secs: 60,
            ..default_config()
        };
        let mut detector = WhaleDetector::new(config);

        let address = "0xwhale456";
        let token_id = "token_yes".to_string();
        let market_id = "market_1".to_string();

        // First large trade should generate alert
        let alert1 = detector.process_trade(
            &token_id,
            &market_id,
            Side::Buy,
            dec!(6000),
            dec!(0.50),
            Some(address),
            Utc::now(),
        );
        assert!(alert1.is_some());

        // Second trade from same address within cooldown should not alert
        let alert2 = detector.process_trade(
            &token_id,
            &market_id,
            Side::Buy,
            dec!(7000),
            dec!(0.50),
            Some(address),
            Utc::now(),
        );
        assert!(alert2.is_none());

        // Trade from different address should still work
        let alert3 = detector.process_trade(
            &token_id,
            &market_id,
            Side::Buy,
            dec!(6000),
            dec!(0.50),
            Some("0xother789"),
            Utc::now(),
        );
        assert!(alert3.is_some());
    }

    #[test]
    fn test_window_expiration() {
        let config = WhaleDetectorConfig {
            aggregate_window_secs: 60, // 1 minute window
            aggregate_threshold_usd: dec!(5000),
            whale_threshold_usd: dec!(10000), // High single threshold
            ..default_config()
        };
        let mut detector = WhaleDetector::new(config);

        let token_id = "token_yes".to_string();
        let market_id = "market_1".to_string();
        let now = Utc::now();

        // Add trade that's old (outside window)
        let old_time = now - Duration::seconds(120); // 2 minutes ago
        detector.process_trade(
            &token_id,
            &market_id,
            Side::Buy,
            dec!(3000),
            dec!(0.50),
            None,
            old_time,
        );

        // Clean up stale data
        detector.cleanup_stale_data();

        // Get activity - should be empty since trade is outside window
        let activity = detector.get_whale_activity(&token_id);
        assert!(activity.is_none());

        // Add new trade within window
        detector.process_trade(
            &token_id,
            &market_id,
            Side::Buy,
            dec!(3000),
            dec!(0.50),
            None,
            now,
        );

        // Now should have activity
        let activity = detector.get_whale_activity(&token_id);
        assert!(activity.is_some());
        assert_eq!(activity.unwrap().total_volume_usd, dec!(3000));
    }

    #[test]
    fn test_disabled_detector() {
        let config = WhaleDetectorConfig {
            enabled: false,
            ..default_config()
        };
        let mut detector = WhaleDetector::new(config);

        // Even a large trade should not generate alert when disabled
        let alert = detector.process_trade(
            &"token_yes".to_string(),
            &"market_1".to_string(),
            Side::Buy,
            dec!(100000),
            dec!(0.50),
            None,
            Utc::now(),
        );

        assert!(alert.is_none());
        assert!(!detector.is_enabled());
    }

    #[test]
    fn test_whale_action_types() {
        let config = default_config();
        let mut detector = WhaleDetector::new(config);

        // Very large trade should recommend position reduction
        let alert = detector.process_trade(
            &"token_yes".to_string(),
            &"market_1".to_string(),
            Side::Buy,
            dec!(20000), // 4x threshold
            dec!(0.50),
            None,
            Utc::now(),
        );

        assert!(alert.is_some());
        let alert = alert.unwrap();
        match alert.recommended_action {
            WhaleAction::ReducePosition { multiplier } => {
                assert!(multiplier <= dec!(0.7));
            }
            _ => panic!("Expected ReducePosition action"),
        }
    }

    #[test]
    fn test_get_known_whales() {
        let config = default_config();
        let mut detector = WhaleDetector::new(config);

        // Add a whale with significant activity
        let address = "0xbigwhale";
        for _ in 0..10 {
            detector.process_trade(
                &"token_yes".to_string(),
                &"market_1".to_string(),
                Side::Buy,
                dec!(1000),
                dec!(0.50),
                Some(address),
                Utc::now(),
            );
        }

        let whales = detector.get_known_whales();
        assert_eq!(whales.len(), 1);
        assert_eq!(whales[0].address, address);
        assert_eq!(whales[0].total_volume_usd, dec!(10000));
    }

    #[test]
    fn test_cumulative_context_side_ratio() {
        let config = WhaleDetectorConfig {
            aggregate_threshold_usd: dec!(5000),
            whale_threshold_usd: dec!(10000),
            ..default_config()
        };
        let mut detector = WhaleDetector::new(config);

        let token_id = "token_yes".to_string();
        let market_id = "market_1".to_string();

        // Add mixed trades: 3 buys, 1 sell
        for _ in 0..3 {
            detector.process_trade(
                &token_id,
                &market_id,
                Side::Buy,
                dec!(1500),
                dec!(0.50),
                None,
                Utc::now(),
            );
        }
        detector.process_trade(
            &token_id,
            &market_id,
            Side::Sell,
            dec!(1500),
            dec!(0.50),
            None,
            Utc::now(),
        );

        let activity = detector.get_whale_activity(&token_id).unwrap();
        // Total: 6000, Buy: 4500, Sell: 1500
        // Side ratio = 4500/6000 = 0.75
        assert_eq!(activity.total_volume_usd, dec!(6000));
        assert_eq!(activity.side_ratio, dec!(0.75));
        assert_eq!(activity.dominant_side, Side::Buy);
    }

    #[test]
    fn test_address_profile_history_limit() {
        let config = WhaleDetectorConfig {
            address_history_limit: 5,
            ..default_config()
        };
        let mut detector = WhaleDetector::new(config);

        let address = "0xlimitedhistory";

        // Add more trades than history limit
        for i in 0..10 {
            detector.process_trade(
                &"token_yes".to_string(),
                &"market_1".to_string(),
                Side::Buy,
                dec!(1000),
                dec!(0.50),
                Some(address),
                Utc::now() + Duration::seconds(i),
            );
        }

        let profile = detector.get_address_profile(address).unwrap();
        assert_eq!(profile.trade_count, 10); // Total count includes all
        assert_eq!(profile.recent_trades.len(), 5); // But only last 5 in history
    }

    #[test]
    fn test_reload_config() {
        let config = default_config();
        let mut detector = WhaleDetector::new(config);

        assert_eq!(detector.config().whale_threshold_usd, dec!(5000));

        let new_config = WhaleDetectorConfig {
            whale_threshold_usd: dec!(10000),
            ..default_config()
        };
        detector.reload_config(new_config);

        assert_eq!(detector.config().whale_threshold_usd, dec!(10000));
    }
}
