//! Whale Order Detection
//!
//! Detects large orders ("whales") and accumulation patterns in the orderbook,
//! using these as leading indicators for market direction.

use chrono::{DateTime, Duration, Utc};
use polysniper_core::{Orderbook, PriceLevel, Side, TokenId};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Configuration for whale detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleConfig {
    /// Whether whale detection is enabled
    pub enabled: bool,
    /// Minimum order size in USD to qualify as "whale" (e.g., $5000)
    pub min_order_size_usd: Decimal,
    /// Minimum % of total depth for relative size threshold (e.g., 10%)
    pub min_relative_size_pct: Decimal,
    /// Time window for detecting accumulation patterns (in seconds)
    pub accumulation_window_secs: u64,
    /// Minimum number of large orders within window for accumulation signal
    pub min_accumulation_orders: u32,
    /// Cooldown between alerts for the same token (in seconds)
    pub alert_cooldown_secs: u64,
    /// Maximum orderbook snapshots to retain for analysis
    #[serde(default = "default_max_snapshots")]
    pub max_snapshots: usize,
}

fn default_max_snapshots() -> usize {
    100
}

impl Default for WhaleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_order_size_usd: dec!(5000),
            min_relative_size_pct: dec!(10),
            accumulation_window_secs: 300,
            min_accumulation_orders: 3,
            alert_cooldown_secs: 600,
            max_snapshots: default_max_snapshots(),
        }
    }
}

/// Type of whale activity detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhaleActivityType {
    /// Single large resting order detected
    LargeResting,
    /// Multiple large orders appearing within a time window (accumulation)
    Accumulation,
    /// Large order being worked (detected via repeated fills at same level)
    Iceberg,
    /// Large order placed then quickly cancelled (spoofing)
    Spoofing,
}

impl std::fmt::Display for WhaleActivityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WhaleActivityType::LargeResting => write!(f, "Large Resting Order"),
            WhaleActivityType::Accumulation => write!(f, "Accumulation"),
            WhaleActivityType::Iceberg => write!(f, "Iceberg Order"),
            WhaleActivityType::Spoofing => write!(f, "Spoofing"),
        }
    }
}

/// Information about detected whale activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleActivity {
    /// Type of whale activity
    pub activity_type: WhaleActivityType,
    /// Buy or Sell side
    pub side: Side,
    /// Total size in USD of whale orders
    pub total_size_usd: Decimal,
    /// Number of whale orders contributing to this activity
    pub num_orders: u32,
    /// Average price of whale orders
    pub avg_price: Decimal,
    /// When the activity was first detected
    pub first_seen: DateTime<Utc>,
    /// When the activity was last updated
    pub last_seen: DateTime<Utc>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: Decimal,
}

impl WhaleActivity {
    /// Calculate the duration of this activity
    pub fn duration(&self) -> Duration {
        self.last_seen - self.first_seen
    }

    /// Calculate the average order size
    pub fn avg_order_size_usd(&self) -> Decimal {
        if self.num_orders == 0 {
            Decimal::ZERO
        } else {
            self.total_size_usd / Decimal::from(self.num_orders)
        }
    }
}

/// A large order detected in the orderbook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeOrder {
    /// Price level of the order
    pub price: Decimal,
    /// Size in contracts
    pub size: Decimal,
    /// Size in USD
    pub size_usd: Decimal,
    /// Side of the order
    pub side: Side,
    /// When the order was first seen
    pub first_seen: DateTime<Utc>,
    /// When the order was last seen
    pub last_seen: DateTime<Utc>,
}

/// Orderbook snapshot for historical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookSnapshot {
    /// When this snapshot was taken
    pub timestamp: DateTime<Utc>,
    /// Total bid size in USD
    pub total_bid_size_usd: Decimal,
    /// Total ask size in USD
    pub total_ask_size_usd: Decimal,
    /// Large bids detected in this snapshot
    pub large_bids: Vec<LargeOrder>,
    /// Large asks detected in this snapshot
    pub large_asks: Vec<LargeOrder>,
}

/// Cooldown tracking
struct AlertCooldown {
    last_alert: DateTime<Utc>,
    activity_type: WhaleActivityType,
}

/// Whale order detector
pub struct WhaleDetector {
    /// Configuration
    config: Arc<RwLock<WhaleConfig>>,
    /// Historical orderbook snapshots by token
    orderbook_history: Arc<RwLock<HashMap<TokenId, VecDeque<OrderbookSnapshot>>>>,
    /// Detected whale activities by token
    detected_whales: Arc<RwLock<HashMap<TokenId, Vec<WhaleActivity>>>>,
    /// Cooldown tracking by token
    cooldowns: Arc<RwLock<HashMap<TokenId, AlertCooldown>>>,
    /// Tracked large orders for spoofing detection
    tracked_orders: Arc<RwLock<HashMap<TokenId, HashMap<String, LargeOrder>>>>,
}

impl WhaleDetector {
    /// Create a new whale detector
    pub fn new(config: WhaleConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            orderbook_history: Arc::new(RwLock::new(HashMap::new())),
            detected_whales: Arc::new(RwLock::new(HashMap::new())),
            cooldowns: Arc::new(RwLock::new(HashMap::new())),
            tracked_orders: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Update configuration
    pub async fn update_config(&self, config: WhaleConfig) {
        *self.config.write().await = config;
    }

    /// Get current configuration
    pub async fn get_config(&self) -> WhaleConfig {
        self.config.read().await.clone()
    }

    /// Check if detector is enabled
    pub async fn is_enabled(&self) -> bool {
        self.config.read().await.enabled
    }

    /// Process an orderbook update and detect whale activity
    pub async fn process_orderbook(
        &self,
        token_id: &TokenId,
        orderbook: &Orderbook,
    ) -> Vec<WhaleActivity> {
        let config = self.config.read().await.clone();
        if !config.enabled {
            return Vec::new();
        }

        // Calculate total orderbook depth for relative sizing
        let total_bid_depth = Self::calculate_total_depth(&orderbook.bids);
        let total_ask_depth = Self::calculate_total_depth(&orderbook.asks);

        // Find large orders
        let now = Utc::now();
        let large_bids = self.find_large_orders(
            &orderbook.bids,
            Side::Buy,
            total_bid_depth,
            &config,
            now,
        );
        let large_asks = self.find_large_orders(
            &orderbook.asks,
            Side::Sell,
            total_ask_depth,
            &config,
            now,
        );

        // Create snapshot
        let snapshot = OrderbookSnapshot {
            timestamp: now,
            total_bid_size_usd: total_bid_depth,
            total_ask_size_usd: total_ask_depth,
            large_bids: large_bids.clone(),
            large_asks: large_asks.clone(),
        };

        // Store snapshot
        self.store_snapshot(token_id, snapshot, config.max_snapshots)
            .await;

        // Detect spoofing (large orders that disappeared)
        self.detect_spoofing(token_id, &large_bids, &large_asks, &config)
            .await;

        // Update tracked orders
        self.update_tracked_orders(token_id, &large_bids, &large_asks)
            .await;

        // Detect activities
        let mut activities = Vec::new();

        // Detect large resting orders
        for order in large_bids.iter().chain(large_asks.iter()) {
            if self.should_alert(token_id, WhaleActivityType::LargeResting, &config).await {
                let activity = WhaleActivity {
                    activity_type: WhaleActivityType::LargeResting,
                    side: order.side,
                    total_size_usd: order.size_usd,
                    num_orders: 1,
                    avg_price: order.price,
                    first_seen: order.first_seen,
                    last_seen: order.last_seen,
                    confidence: self.calculate_confidence(order.size_usd, &config),
                };
                activities.push(activity);
            }
        }

        // Detect accumulation patterns
        if let Some(accumulation) = self.detect_accumulation(token_id, &config).await {
            if self.should_alert(token_id, WhaleActivityType::Accumulation, &config).await {
                activities.push(accumulation);
            }
        }

        // Store detected activities
        if !activities.is_empty() {
            let mut whales = self.detected_whales.write().await;
            let token_activities = whales.entry(token_id.clone()).or_insert_with(Vec::new);
            token_activities.extend(activities.clone());

            // Keep only recent activities (last hour)
            let cutoff = now - Duration::hours(1);
            token_activities.retain(|a| a.last_seen > cutoff);
        }

        activities
    }

    /// Calculate total depth in USD
    fn calculate_total_depth(levels: &[PriceLevel]) -> Decimal {
        levels
            .iter()
            .map(|l| l.price * l.size)
            .sum()
    }

    /// Find large orders in price levels
    fn find_large_orders(
        &self,
        levels: &[PriceLevel],
        side: Side,
        total_depth: Decimal,
        config: &WhaleConfig,
        now: DateTime<Utc>,
    ) -> Vec<LargeOrder> {
        let mut large_orders = Vec::new();
        let relative_threshold = total_depth * config.min_relative_size_pct / dec!(100);

        for level in levels {
            let size_usd = level.price * level.size;

            // Check both absolute and relative thresholds
            if size_usd >= config.min_order_size_usd || size_usd >= relative_threshold {
                large_orders.push(LargeOrder {
                    price: level.price,
                    size: level.size,
                    size_usd,
                    side,
                    first_seen: now,
                    last_seen: now,
                });
            }
        }

        large_orders
    }

    /// Store an orderbook snapshot
    async fn store_snapshot(&self, token_id: &TokenId, snapshot: OrderbookSnapshot, max_snapshots: usize) {
        let mut history = self.orderbook_history.write().await;
        let snapshots = history.entry(token_id.clone()).or_insert_with(VecDeque::new);

        snapshots.push_back(snapshot);

        // Trim to max size
        while snapshots.len() > max_snapshots {
            snapshots.pop_front();
        }
    }

    /// Detect accumulation patterns (multiple large orders appearing)
    async fn detect_accumulation(
        &self,
        token_id: &TokenId,
        config: &WhaleConfig,
    ) -> Option<WhaleActivity> {
        let history = self.orderbook_history.read().await;
        let snapshots = history.get(token_id)?;

        if snapshots.is_empty() {
            return None;
        }

        let now = Utc::now();
        let window_start = now - Duration::seconds(config.accumulation_window_secs as i64);

        // Count large orders within the window
        let mut bid_count = 0u32;
        let mut ask_count = 0u32;
        let mut total_bid_size_usd = Decimal::ZERO;
        let mut total_ask_size_usd = Decimal::ZERO;
        let mut bid_prices = Vec::new();
        let mut ask_prices = Vec::new();
        let mut first_seen: Option<DateTime<Utc>> = None;

        for snapshot in snapshots.iter() {
            if snapshot.timestamp < window_start {
                continue;
            }

            if first_seen.is_none() {
                first_seen = Some(snapshot.timestamp);
            }

            for order in &snapshot.large_bids {
                bid_count += 1;
                total_bid_size_usd += order.size_usd;
                bid_prices.push(order.price);
            }

            for order in &snapshot.large_asks {
                ask_count += 1;
                total_ask_size_usd += order.size_usd;
                ask_prices.push(order.price);
            }
        }

        // Determine if there's significant accumulation on either side
        let min_orders = config.min_accumulation_orders;

        if bid_count >= min_orders && bid_count > ask_count {
            // Bullish accumulation
            let avg_price = if bid_prices.is_empty() {
                Decimal::ZERO
            } else {
                bid_prices.iter().copied().sum::<Decimal>() / Decimal::from(bid_prices.len() as u32)
            };

            return Some(WhaleActivity {
                activity_type: WhaleActivityType::Accumulation,
                side: Side::Buy,
                total_size_usd: total_bid_size_usd,
                num_orders: bid_count,
                avg_price,
                first_seen: first_seen.unwrap_or(now),
                last_seen: now,
                confidence: self.calculate_accumulation_confidence(bid_count, min_orders),
            });
        } else if ask_count >= min_orders && ask_count > bid_count {
            // Bearish accumulation
            let avg_price = if ask_prices.is_empty() {
                Decimal::ZERO
            } else {
                ask_prices.iter().copied().sum::<Decimal>() / Decimal::from(ask_prices.len() as u32)
            };

            return Some(WhaleActivity {
                activity_type: WhaleActivityType::Accumulation,
                side: Side::Sell,
                total_size_usd: total_ask_size_usd,
                num_orders: ask_count,
                avg_price,
                first_seen: first_seen.unwrap_or(now),
                last_seen: now,
                confidence: self.calculate_accumulation_confidence(ask_count, min_orders),
            });
        }

        None
    }

    /// Detect spoofing (orders that were placed then quickly removed)
    async fn detect_spoofing(
        &self,
        token_id: &TokenId,
        current_bids: &[LargeOrder],
        current_asks: &[LargeOrder],
        config: &WhaleConfig,
    ) {
        let mut tracked = self.tracked_orders.write().await;
        let token_orders = match tracked.get_mut(token_id) {
            Some(orders) => orders,
            None => return,
        };

        let now = Utc::now();
        let spoof_threshold = Duration::seconds(10); // Order must disappear within 10 seconds

        // Create sets of current order keys for quick lookup
        let current_bid_keys: std::collections::HashSet<_> = current_bids
            .iter()
            .map(|o| format!("bid_{}", o.price))
            .collect();
        let current_ask_keys: std::collections::HashSet<_> = current_asks
            .iter()
            .map(|o| format!("ask_{}", o.price))
            .collect();

        // Find orders that disappeared quickly
        let mut spoofed = Vec::new();
        for (key, order) in token_orders.iter() {
            let is_present = if order.side == Side::Buy {
                current_bid_keys.contains(key)
            } else {
                current_ask_keys.contains(key)
            };

            if !is_present {
                let age = now - order.first_seen;
                if age < spoof_threshold && order.size_usd >= config.min_order_size_usd {
                    spoofed.push(key.clone());
                    info!(
                        token_id = %token_id,
                        side = ?order.side,
                        size_usd = %order.size_usd,
                        price = %order.price,
                        "Potential spoofing detected: large order removed after {:?}",
                        age
                    );
                }
            }
        }

        // Remove spoofed orders from tracking
        for key in spoofed {
            token_orders.remove(&key);
        }
    }

    /// Update tracked orders with current large orders
    async fn update_tracked_orders(
        &self,
        token_id: &TokenId,
        bids: &[LargeOrder],
        asks: &[LargeOrder],
    ) {
        let mut tracked = self.tracked_orders.write().await;
        let token_orders = tracked.entry(token_id.clone()).or_insert_with(HashMap::new);
        let now = Utc::now();

        // Update or add bids
        for order in bids {
            let key = format!("bid_{}", order.price);
            if let Some(existing) = token_orders.get_mut(&key) {
                existing.last_seen = now;
            } else {
                token_orders.insert(key, order.clone());
            }
        }

        // Update or add asks
        for order in asks {
            let key = format!("ask_{}", order.price);
            if let Some(existing) = token_orders.get_mut(&key) {
                existing.last_seen = now;
            } else {
                token_orders.insert(key, order.clone());
            }
        }

        // Clean up old tracked orders (older than 1 minute)
        let cutoff = now - Duration::minutes(1);
        token_orders.retain(|_, order| order.last_seen > cutoff);
    }

    /// Check if we should send an alert (respecting cooldown)
    async fn should_alert(
        &self,
        token_id: &TokenId,
        activity_type: WhaleActivityType,
        config: &WhaleConfig,
    ) -> bool {
        let mut cooldowns = self.cooldowns.write().await;
        let now = Utc::now();
        let cooldown_duration = Duration::seconds(config.alert_cooldown_secs as i64);

        if let Some(cooldown) = cooldowns.get(token_id) {
            if now - cooldown.last_alert < cooldown_duration
                && cooldown.activity_type == activity_type {
                debug!(
                    token_id = %token_id,
                    activity_type = ?activity_type,
                    "Alert suppressed due to cooldown"
                );
                return false;
            }
        }

        // Update cooldown
        cooldowns.insert(
            token_id.clone(),
            AlertCooldown {
                last_alert: now,
                activity_type,
            },
        );

        true
    }

    /// Calculate confidence score based on order size
    fn calculate_confidence(&self, size_usd: Decimal, config: &WhaleConfig) -> Decimal {
        // Higher size = higher confidence, capped at 1.0
        let ratio = size_usd / config.min_order_size_usd;
        if ratio >= dec!(3) {
            dec!(1.0)
        } else if ratio >= dec!(2) {
            dec!(0.9)
        } else if ratio >= dec!(1.5) {
            dec!(0.8)
        } else {
            dec!(0.7)
        }
    }

    /// Calculate confidence for accumulation patterns
    fn calculate_accumulation_confidence(&self, order_count: u32, min_orders: u32) -> Decimal {
        let ratio = Decimal::from(order_count) / Decimal::from(min_orders);
        if ratio >= dec!(3) {
            dec!(1.0)
        } else if ratio >= dec!(2) {
            dec!(0.9)
        } else {
            dec!(0.7) + (ratio - dec!(1)) * dec!(0.1)
        }
    }

    /// Get recent whale activities for a token
    pub async fn get_recent_activities(&self, token_id: &TokenId) -> Vec<WhaleActivity> {
        let whales = self.detected_whales.read().await;
        whales.get(token_id).cloned().unwrap_or_default()
    }

    /// Get orderbook history for a token
    pub async fn get_orderbook_history(&self, token_id: &TokenId) -> Vec<OrderbookSnapshot> {
        let history = self.orderbook_history.read().await;
        history
            .get(token_id)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Clear all tracked data for a token
    pub async fn clear_token_data(&self, token_id: &TokenId) {
        self.orderbook_history.write().await.remove(token_id);
        self.detected_whales.write().await.remove(token_id);
        self.cooldowns.write().await.remove(token_id);
        self.tracked_orders.write().await.remove(token_id);
    }

    /// Clear all tracked data
    pub async fn clear_all(&self) {
        self.orderbook_history.write().await.clear();
        self.detected_whales.write().await.clear();
        self.cooldowns.write().await.clear();
        self.tracked_orders.write().await.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polysniper_core::PriceLevel;

    fn create_orderbook(
        token_id: &str,
        bids: Vec<(Decimal, Decimal)>,
        asks: Vec<(Decimal, Decimal)>,
    ) -> Orderbook {
        Orderbook {
            token_id: token_id.to_string(),
            market_id: "test_market".to_string(),
            bids: bids
                .into_iter()
                .map(|(price, size)| PriceLevel { price, size })
                .collect(),
            asks: asks
                .into_iter()
                .map(|(price, size)| PriceLevel { price, size })
                .collect(),
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_calculate_total_depth() {
        let levels = vec![
            PriceLevel { price: dec!(0.50), size: dec!(100) },
            PriceLevel { price: dec!(0.49), size: dec!(200) },
        ];
        let depth = WhaleDetector::calculate_total_depth(&levels);
        // 0.50 * 100 + 0.49 * 200 = 50 + 98 = 148
        assert_eq!(depth, dec!(148));
    }

    #[tokio::test]
    async fn test_detect_large_resting_order() {
        let config = WhaleConfig {
            enabled: true,
            min_order_size_usd: dec!(5000),
            min_relative_size_pct: dec!(10),
            alert_cooldown_secs: 0, // No cooldown for testing
            ..Default::default()
        };
        let detector = WhaleDetector::new(config);

        // Create orderbook with one large bid ($7500 at 0.50)
        let orderbook = create_orderbook(
            "test_token",
            vec![(dec!(0.50), dec!(15000))], // 0.50 * 15000 = $7500
            vec![(dec!(0.55), dec!(1000))],   // 0.55 * 1000 = $550
        );

        let activities = detector.process_orderbook(&"test_token".to_string(), &orderbook).await;

        assert!(!activities.is_empty(), "Should detect at least one whale activity");

        let large_resting = activities.iter().find(|a| a.activity_type == WhaleActivityType::LargeResting);
        assert!(large_resting.is_some(), "Should detect large resting order");

        let activity = large_resting.unwrap();
        assert_eq!(activity.side, Side::Buy);
        assert!(activity.total_size_usd >= dec!(5000));
    }

    #[tokio::test]
    async fn test_no_whale_small_orders() {
        let config = WhaleConfig {
            enabled: true,
            min_order_size_usd: dec!(5000),
            // Relative threshold is per-side, so we need multiple orders per side
            // to ensure no single order exceeds the threshold
            min_relative_size_pct: dec!(60), // 60% threshold
            alert_cooldown_secs: 0,
            ..Default::default()
        };
        let detector = WhaleDetector::new(config);

        // Multiple small orders per side, each less than 60% of that side's depth
        // and each less than $5000 absolute
        // Bid side: total = $50+$40 = $90, each order is <60% of side depth
        // Ask side: total = $55+$50 = $105, each order is <60% of side depth
        let orderbook = create_orderbook(
            "test_token",
            vec![
                (dec!(0.50), dec!(100)), // $50 = 55% of $90 bid depth - under 60%
                (dec!(0.40), dec!(100)), // $40 = 44% of $90 bid depth - under 60%
            ],
            vec![
                (dec!(0.55), dec!(100)), // $55 = 52% of $105 ask depth - under 60%
                (dec!(0.50), dec!(100)), // $50 = 48% of $105 ask depth - under 60%
            ],
        );

        let activities = detector.process_orderbook(&"test_token".to_string(), &orderbook).await;

        assert!(activities.is_empty(), "Should not detect whale activity for small orders");
    }

    #[tokio::test]
    async fn test_detect_accumulation() {
        let config = WhaleConfig {
            enabled: true,
            min_order_size_usd: dec!(2000), // Set high enough that small asks don't qualify
            min_relative_size_pct: dec!(80), // Very high so relative threshold doesn't trigger on small orders
            accumulation_window_secs: 300,
            min_accumulation_orders: 3,
            alert_cooldown_secs: 0, // No cooldown so we can detect accumulation
            ..Default::default()
        };
        let detector = WhaleDetector::new(config);
        let token_id = "test_token".to_string();

        // Process multiple orderbooks with large bids and truly small asks
        // Bids: 0.50 * 5000 = $2500, 0.51 * 5000 = $2550, etc. - all > $2000
        // Asks: multiple small orders so no single one is > 80% of ask depth
        let mut found_accumulation = false;
        for i in 0..5 {
            let orderbook = create_orderbook(
                "test_token",
                vec![(dec!(0.50) + Decimal::from(i) * dec!(0.01), dec!(5000))], // $2500+ bids
                vec![
                    (dec!(0.55), dec!(50)),  // $27.50
                    (dec!(0.56), dec!(50)),  // $28.00
                    (dec!(0.57), dec!(50)),  // $28.50
                ], // Multiple small asks, none > $2000 and none > 80% of ~$84 ask depth
            );
            let activities = detector.process_orderbook(&token_id, &orderbook).await;

            // After processing 3+ orderbooks, we should start seeing accumulation signals
            if activities.iter().any(|a| a.activity_type == WhaleActivityType::Accumulation) {
                found_accumulation = true;
            }
        }

        // Also check stored activities
        let stored_activities = detector.get_recent_activities(&token_id).await;
        if stored_activities.iter().any(|a| a.activity_type == WhaleActivityType::Accumulation) {
            found_accumulation = true;
        }

        assert!(found_accumulation, "Should detect accumulation pattern after multiple large orders");
    }

    #[tokio::test]
    async fn test_cooldown_enforcement() {
        let config = WhaleConfig {
            enabled: true,
            min_order_size_usd: dec!(1000),
            min_relative_size_pct: dec!(10),
            alert_cooldown_secs: 600, // 10 minute cooldown
            ..Default::default()
        };
        let detector = WhaleDetector::new(config);
        let token_id = "test_token".to_string();

        let orderbook = create_orderbook(
            "test_token",
            vec![(dec!(0.50), dec!(10000))], // Large bid
            vec![(dec!(0.55), dec!(100))],
        );

        // First detection should work
        let activities1 = detector.process_orderbook(&token_id, &orderbook).await;
        assert!(!activities1.is_empty(), "First detection should work");

        // Second detection should be suppressed by cooldown
        let activities2 = detector.process_orderbook(&token_id, &orderbook).await;
        assert!(activities2.is_empty(), "Second detection should be suppressed by cooldown");
    }

    #[tokio::test]
    async fn test_disabled_detector() {
        let config = WhaleConfig {
            enabled: false,
            ..Default::default()
        };
        let detector = WhaleDetector::new(config);

        let orderbook = create_orderbook(
            "test_token",
            vec![(dec!(0.50), dec!(100000))], // Huge order
            vec![(dec!(0.55), dec!(100))],
        );

        let activities = detector.process_orderbook(&"test_token".to_string(), &orderbook).await;
        assert!(activities.is_empty(), "Disabled detector should not produce activities");
    }

    #[test]
    fn test_confidence_calculation() {
        let config = WhaleConfig::default();
        let detector = WhaleDetector::new(config.clone());

        // 3x minimum size = max confidence
        assert_eq!(detector.calculate_confidence(dec!(15000), &config), dec!(1.0));

        // 2x minimum size
        assert_eq!(detector.calculate_confidence(dec!(10000), &config), dec!(0.9));

        // 1.5x minimum size
        assert_eq!(detector.calculate_confidence(dec!(7500), &config), dec!(0.8));

        // At minimum size
        assert_eq!(detector.calculate_confidence(dec!(5000), &config), dec!(0.7));
    }

    #[tokio::test]
    async fn test_clear_token_data() {
        let config = WhaleConfig {
            enabled: true,
            min_order_size_usd: dec!(1000),
            alert_cooldown_secs: 0,
            ..Default::default()
        };
        let detector = WhaleDetector::new(config);
        let token_id = "test_token".to_string();

        let orderbook = create_orderbook(
            "test_token",
            vec![(dec!(0.50), dec!(10000))],
            vec![(dec!(0.55), dec!(100))],
        );

        detector.process_orderbook(&token_id, &orderbook).await;

        // Verify data exists
        assert!(!detector.get_recent_activities(&token_id).await.is_empty());
        assert!(!detector.get_orderbook_history(&token_id).await.is_empty());

        // Clear data
        detector.clear_token_data(&token_id).await;

        // Verify data is cleared
        assert!(detector.get_recent_activities(&token_id).await.is_empty());
        assert!(detector.get_orderbook_history(&token_id).await.is_empty());
    }
}
