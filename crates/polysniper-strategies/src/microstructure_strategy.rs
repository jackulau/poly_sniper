//! Market Microstructure Trading Strategy
//!
//! A comprehensive trading strategy that combines VPIN, whale detection, and market impact
//! signals to generate intelligent trade signals and risk adjustments.

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use polysniper_core::{
    events::{
        MicrostructureAction, MicrostructureComponents, MicrostructureEvent,
        MicrostructureSignalEvent, MicrostructureSignalType, ToxicityChangeEvent,
        ToxicityLevel as CoreToxicityLevel, VpinUpdateEvent, WhaleAction, WhaleActivitySummary,
        WhaleAlertType, WhaleDetectedEvent,
    },
    MicrostructurePublisher, OrderType, Outcome, Priority, Side, StateProvider, Strategy,
    StrategyError, SystemEvent, TokenId, TradeSignal,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::vpin::{ToxicityLevel, VpinCalculator, VpinConfig, VpinResult};
use crate::whale_detector::{WhaleAlert, WhaleDetector, WhaleDetectorConfig};

/// Action to take based on toxicity level
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToxicityAction {
    /// No action
    #[default]
    None,
    /// Reduce position sizes by multiplier
    ReducePositions { multiplier: Decimal },
    /// Halt new trades entirely
    HaltNewTrades,
    /// Trade against the dominant flow (contrarian)
    FadeTheCrowd,
    /// Trade with the dominant flow (momentum)
    FollowTheCrowd,
}

/// VPIN-based trading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpinTradingConfig {
    /// Whether VPIN-based trading is enabled
    pub enabled: bool,
    /// Action to take when toxicity is high
    pub high_toxicity_action: ToxicityAction,
    /// Action to take when toxicity is low
    pub low_toxicity_action: ToxicityAction,
    /// Position size multipliers per toxicity level
    #[serde(default)]
    pub toxicity_position_multiplier: HashMap<String, Decimal>,
}

impl Default for VpinTradingConfig {
    fn default() -> Self {
        let mut multipliers = HashMap::new();
        multipliers.insert("low".to_string(), dec!(1.2));
        multipliers.insert("normal".to_string(), dec!(1.0));
        multipliers.insert("elevated".to_string(), dec!(0.7));
        multipliers.insert("high".to_string(), dec!(0.4));

        Self {
            enabled: true,
            high_toxicity_action: ToxicityAction::ReducePositions {
                multiplier: dec!(0.5),
            },
            low_toxicity_action: ToxicityAction::None,
            toxicity_position_multiplier: multipliers,
        }
    }
}

/// Whale-based trading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleTradingConfig {
    /// Whether whale-based trading is enabled
    pub enabled: bool,
    /// Whether to follow whale trades
    pub follow_whale: bool,
    /// Minimum confidence to follow a whale
    pub follow_confidence_threshold: Decimal,
    /// Threshold in USD to consider avoiding whale impact
    pub avoid_whale_threshold_usd: Decimal,
    /// Position multiplier when whale activity detected
    pub whale_position_multiplier: Decimal,
}

impl Default for WhaleTradingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            follow_whale: true,
            follow_confidence_threshold: dec!(0.7),
            avoid_whale_threshold_usd: dec!(10000),
            whale_position_multiplier: dec!(0.6),
        }
    }
}

/// Market impact adjustment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAdjustmentConfig {
    /// Whether impact adjustments are enabled
    pub enabled: bool,
    /// Maximum acceptable impact in basis points
    pub max_acceptable_impact_bps: Decimal,
    /// Impact threshold to trigger auto-slicing
    pub auto_slice_above_impact_bps: Decimal,
    /// Delay in seconds when high impact detected
    pub delay_high_impact_secs: u64,
}

impl Default for ImpactAdjustmentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_acceptable_impact_bps: dec!(50),
            auto_slice_above_impact_bps: dec!(25),
            delay_high_impact_secs: 60,
        }
    }
}

/// Configuration for the microstructure strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureStrategyConfig {
    /// Whether the strategy is enabled
    pub enabled: bool,
    /// VPIN-based trading configuration
    #[serde(default)]
    pub vpin_trading: VpinTradingConfig,
    /// Whale-based trading configuration
    #[serde(default)]
    pub whale_trading: WhaleTradingConfig,
    /// Impact-based adjustments
    #[serde(default)]
    pub impact_adjustments: ImpactAdjustmentConfig,
    /// Minimum confidence threshold for signals
    #[serde(default = "default_min_confidence")]
    pub min_confidence: Decimal,
    /// Cooldown between signals in seconds
    #[serde(default = "default_cooldown_secs")]
    pub cooldown_secs: u64,
    /// Maximum signals per hour
    #[serde(default = "default_max_signals_per_hour")]
    pub max_signals_per_hour: u32,
    /// Order size in USD
    #[serde(default = "default_order_size_usd")]
    pub order_size_usd: Decimal,
    /// VPIN calculator configuration
    #[serde(default)]
    pub vpin_config: VpinConfig,
    /// Whale detector configuration
    #[serde(default)]
    pub whale_detector_config: WhaleDetectorConfig,
}

fn default_min_confidence() -> Decimal {
    dec!(0.6)
}

fn default_cooldown_secs() -> u64 {
    300
}

fn default_max_signals_per_hour() -> u32 {
    10
}

fn default_order_size_usd() -> Decimal {
    dec!(100)
}

impl Default for MicrostructureStrategyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            vpin_trading: VpinTradingConfig::default(),
            whale_trading: WhaleTradingConfig::default(),
            impact_adjustments: ImpactAdjustmentConfig::default(),
            min_confidence: default_min_confidence(),
            cooldown_secs: default_cooldown_secs(),
            max_signals_per_hour: default_max_signals_per_hour(),
            order_size_usd: default_order_size_usd(),
            vpin_config: VpinConfig::default(),
            whale_detector_config: WhaleDetectorConfig::default(),
        }
    }
}

/// Market Microstructure Strategy
///
/// Combines VPIN analysis, whale detection, and market impact modeling to generate
/// intelligent trading signals and risk-adjusted position sizing.
pub struct MicrostructureStrategy {
    id: String,
    name: String,
    enabled: Arc<AtomicBool>,
    config: Arc<RwLock<MicrostructureStrategyConfig>>,

    /// VPIN calculator for each token
    vpin_calculator: Arc<RwLock<VpinCalculator>>,

    /// Whale detector
    whale_detector: Arc<RwLock<WhaleDetector>>,

    /// Current toxicity level per token
    current_toxicity: Arc<RwLock<HashMap<TokenId, ToxicityLevel>>>,

    /// Recent whale alerts per token
    recent_whale_alerts: Arc<RwLock<HashMap<TokenId, VecDeque<WhaleAlert>>>>,

    /// Signal cooldowns per token
    signal_cooldowns: Arc<RwLock<HashMap<TokenId, DateTime<Utc>>>>,

    /// Hourly signal counts
    hourly_signal_counts: Arc<RwLock<HashMap<String, u32>>>,

    /// Hour start for signal counting
    hour_start: Arc<RwLock<DateTime<Utc>>>,

    /// Token to market mapping
    token_market_map: Arc<RwLock<HashMap<TokenId, (String, Outcome)>>>,

    /// Event publisher (optional)
    publisher: Arc<RwLock<Option<MicrostructurePublisher>>>,
}

impl MicrostructureStrategy {
    /// Create a new microstructure strategy with the given configuration
    pub fn new(config: MicrostructureStrategyConfig) -> Self {
        let enabled = config.enabled;
        let vpin_calculator = VpinCalculator::new(config.vpin_config.clone());
        let whale_detector = WhaleDetector::new(config.whale_detector_config.clone());

        Self {
            id: "microstructure".to_string(),
            name: "Market Microstructure Strategy".to_string(),
            enabled: Arc::new(AtomicBool::new(enabled)),
            config: Arc::new(RwLock::new(config)),
            vpin_calculator: Arc::new(RwLock::new(vpin_calculator)),
            whale_detector: Arc::new(RwLock::new(whale_detector)),
            current_toxicity: Arc::new(RwLock::new(HashMap::new())),
            recent_whale_alerts: Arc::new(RwLock::new(HashMap::new())),
            signal_cooldowns: Arc::new(RwLock::new(HashMap::new())),
            hourly_signal_counts: Arc::new(RwLock::new(HashMap::new())),
            hour_start: Arc::new(RwLock::new(Utc::now())),
            token_market_map: Arc::new(RwLock::new(HashMap::new())),
            publisher: Arc::new(RwLock::new(None)),
        }
    }

    /// Set the event publisher
    pub async fn set_publisher(&self, publisher: MicrostructurePublisher) {
        let mut p = self.publisher.write().await;
        *p = Some(publisher);
    }

    /// Register a token for monitoring
    pub async fn register_token(&self, token_id: TokenId, market_id: String, outcome: Outcome) {
        self.token_market_map
            .write()
            .await
            .insert(token_id, (market_id, outcome));
    }

    /// Check if token is in cooldown
    async fn is_in_cooldown(&self, token_id: &TokenId) -> bool {
        let cooldowns = self.signal_cooldowns.read().await;
        let config = self.config.read().await;

        if let Some(last_signal) = cooldowns.get(token_id) {
            return Utc::now() < *last_signal + Duration::seconds(config.cooldown_secs as i64);
        }
        false
    }

    /// Set cooldown for a token
    async fn set_cooldown(&self, token_id: &TokenId) {
        self.signal_cooldowns
            .write()
            .await
            .insert(token_id.clone(), Utc::now());
    }

    /// Check if hourly signal limit is exceeded
    async fn is_hourly_limit_exceeded(&self) -> bool {
        let mut hour_start = self.hour_start.write().await;
        let mut counts = self.hourly_signal_counts.write().await;
        let config = self.config.read().await;

        // Reset if new hour
        if Utc::now() >= *hour_start + Duration::hours(1) {
            *hour_start = Utc::now();
            counts.clear();
        }

        let total: u32 = counts.values().sum();
        total >= config.max_signals_per_hour
    }

    /// Increment hourly signal count
    async fn increment_signal_count(&self, token_id: &TokenId) {
        let mut counts = self.hourly_signal_counts.write().await;
        *counts.entry(token_id.clone()).or_insert(0) += 1;
    }

    /// Process a trade for VPIN and whale detection
    #[allow(clippy::too_many_arguments)]
    pub async fn process_trade(
        &self,
        token_id: &TokenId,
        market_id: &str,
        side: Side,
        size_usd: Decimal,
        price: Decimal,
        bid: Decimal,
        ask: Decimal,
        address: Option<&str>,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        let mut signals = vec![];

        // Update VPIN
        let vpin_result = {
            let mut calculator = self.vpin_calculator.write().await;
            calculator.process_trade(token_id, price, size_usd, bid, ask)
        };

        if let Some(vpin_result) = vpin_result {
            // Check for toxicity level change
            let prev_level = {
                let toxicity = self.current_toxicity.read().await;
                toxicity.get(token_id).copied()
            };

            if prev_level != Some(vpin_result.toxicity_level) {
                // Publish toxicity change event
                if let Some(publisher) = self.publisher.read().await.as_ref() {
                    let event = ToxicityChangeEvent::new(
                        token_id.clone(),
                        market_id.to_string(),
                        convert_toxicity_level(prev_level.unwrap_or(ToxicityLevel::Normal)),
                        convert_toxicity_level(vpin_result.toxicity_level),
                        vpin_result.vpin,
                        "VPIN threshold crossed".to_string(),
                    );
                    let _ = publisher.publish_toxicity_change(event);
                }

                // Generate signal based on toxicity change
                if let Some(signal) = self
                    .generate_toxicity_signal(&vpin_result, market_id)
                    .await?
                {
                    signals.push(signal);
                }
            }

            // Update current toxicity
            self.current_toxicity
                .write()
                .await
                .insert(token_id.clone(), vpin_result.toxicity_level);

            // Publish VPIN update
            if let Some(publisher) = self.publisher.read().await.as_ref() {
                let event = VpinUpdateEvent::new(
                    token_id.clone(),
                    market_id.to_string(),
                    vpin_result.vpin,
                    convert_toxicity_level(vpin_result.toxicity_level),
                    vpin_result.buy_volume_pct,
                    vpin_result.sell_volume_pct,
                    vpin_result.bucket_count,
                );
                let _ = publisher.publish_vpin_update(event);
            }
        }

        // Check for whale activity
        let whale_alert = {
            let mut detector = self.whale_detector.write().await;
            let market_id_string = market_id.to_string();
            detector.process_trade(
                token_id, &market_id_string, side, size_usd, price, address, timestamp,
            )
        };

        if let Some(whale_alert) = whale_alert {
            // Store whale alert
            {
                let mut alerts = self.recent_whale_alerts.write().await;
                let token_alerts = alerts.entry(token_id.clone()).or_default();
                token_alerts.push_back(whale_alert.clone());

                // Keep only last 10 alerts per token
                while token_alerts.len() > 10 {
                    token_alerts.pop_front();
                }
            }

            // Publish whale detection
            if let Some(publisher) = self.publisher.read().await.as_ref() {
                let event = WhaleDetectedEvent::new(
                    token_id.clone(),
                    market_id.to_string(),
                    convert_whale_alert_type(&whale_alert.alert_type),
                    whale_alert.whale_trade.size_usd,
                    whale_alert.whale_trade.side,
                    convert_whale_action(&whale_alert.recommended_action),
                    whale_alert.confidence,
                );
                let _ = publisher.publish_whale_detected(event);
            }

            // Generate whale signal if configured
            if let Some(signal) = self
                .generate_whale_signal(&whale_alert, market_id)
                .await?
            {
                signals.push(signal);
            }
        }

        Ok(signals)
    }

    /// Generate signal based on toxicity level
    async fn generate_toxicity_signal(
        &self,
        vpin_result: &VpinResult,
        market_id: &str,
    ) -> Result<Option<TradeSignal>, StrategyError> {
        let config = self.config.read().await;

        if !config.vpin_trading.enabled {
            return Ok(None);
        }

        // Check cooldown
        if self.is_in_cooldown(&vpin_result.token_id).await {
            debug!(
                token_id = %vpin_result.token_id,
                "Token in cooldown, skipping toxicity signal"
            );
            return Ok(None);
        }

        // Check hourly limit
        if self.is_hourly_limit_exceeded().await {
            debug!("Hourly signal limit exceeded");
            return Ok(None);
        }

        let action = match vpin_result.toxicity_level {
            ToxicityLevel::High => &config.vpin_trading.high_toxicity_action,
            ToxicityLevel::Low => &config.vpin_trading.low_toxicity_action,
            _ => return Ok(None),
        };

        let signal = match action {
            ToxicityAction::FadeTheCrowd => {
                // Trade against the dominant flow direction
                let side = if vpin_result.buy_volume_pct > dec!(0.6) {
                    Side::Sell // Fade the buyers
                } else if vpin_result.sell_volume_pct > dec!(0.6) {
                    Side::Buy // Fade the sellers
                } else {
                    return Ok(None);
                };

                self.build_signal(
                    &vpin_result.token_id,
                    market_id,
                    side,
                    format!(
                        "Fade crowd: VPIN={:.2}, toxicity={:?}",
                        vpin_result.vpin, vpin_result.toxicity_level
                    ),
                    dec!(0.7),
                )
                .await
            }
            ToxicityAction::FollowTheCrowd => {
                // Trade with the dominant flow direction
                let side = if vpin_result.buy_volume_pct > dec!(0.6) {
                    Side::Buy
                } else if vpin_result.sell_volume_pct > dec!(0.6) {
                    Side::Sell
                } else {
                    return Ok(None);
                };

                self.build_signal(
                    &vpin_result.token_id,
                    market_id,
                    side,
                    format!("Follow flow: VPIN={:.2}", vpin_result.vpin),
                    dec!(0.65),
                )
                .await
            }
            _ => return Ok(None),
        };

        if let Some(ref s) = signal {
            self.set_cooldown(&vpin_result.token_id).await;
            self.increment_signal_count(&vpin_result.token_id).await;

            info!(
                token_id = %s.token_id,
                side = ?s.side,
                reason = %s.reason,
                "Generated toxicity-based trade signal"
            );
        }

        Ok(signal)
    }

    /// Generate signal based on whale detection
    async fn generate_whale_signal(
        &self,
        whale_alert: &WhaleAlert,
        market_id: &str,
    ) -> Result<Option<TradeSignal>, StrategyError> {
        let config = self.config.read().await;

        if !config.whale_trading.enabled {
            return Ok(None);
        }

        if !config.whale_trading.follow_whale {
            return Ok(None);
        }

        if whale_alert.confidence < config.whale_trading.follow_confidence_threshold {
            debug!(
                confidence = %whale_alert.confidence,
                threshold = %config.whale_trading.follow_confidence_threshold,
                "Whale confidence below threshold"
            );
            return Ok(None);
        }

        // Check cooldown
        if self.is_in_cooldown(&whale_alert.token_id).await {
            return Ok(None);
        }

        // Check hourly limit
        if self.is_hourly_limit_exceeded().await {
            return Ok(None);
        }

        // Follow the whale's direction
        let side = whale_alert.whale_trade.side;

        let signal = self
            .build_signal(
                &whale_alert.token_id,
                market_id,
                side,
                format!(
                    "Follow whale: ${:.0} {:?}",
                    whale_alert.whale_trade.size_usd, whale_alert.alert_type
                ),
                whale_alert.confidence,
            )
            .await;

        if let Some(ref s) = signal {
            self.set_cooldown(&whale_alert.token_id).await;
            self.increment_signal_count(&whale_alert.token_id).await;

            info!(
                token_id = %s.token_id,
                side = ?s.side,
                whale_size_usd = %whale_alert.whale_trade.size_usd,
                "Generated whale-following trade signal"
            );
        }

        Ok(signal)
    }

    /// Build a trade signal
    async fn build_signal(
        &self,
        token_id: &TokenId,
        market_id: &str,
        side: Side,
        reason: String,
        confidence: Decimal,
    ) -> Option<TradeSignal> {
        let config = self.config.read().await;

        if confidence < config.min_confidence {
            return None;
        }

        let token_map = self.token_market_map.read().await;
        let outcome = token_map
            .get(token_id)
            .map(|(_, o)| *o)
            .unwrap_or(Outcome::Yes);
        drop(token_map);

        // Apply size multiplier
        let multiplier = self.get_size_multiplier(token_id).await;
        let size_usd = config.order_size_usd * multiplier;

        Some(TradeSignal {
            id: format!(
                "sig_micro_{}_{}_{}",
                token_id,
                Utc::now().timestamp_millis(),
                rand_suffix()
            ),
            strategy_id: self.id.clone(),
            market_id: market_id.to_string(),
            token_id: token_id.clone(),
            outcome,
            side,
            price: None, // Market order
            size: Decimal::ZERO, // Will be calculated from size_usd
            size_usd,
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: Utc::now(),
            reason,
            metadata: serde_json::json!({
                "strategy": "microstructure",
                "confidence": confidence.to_string(),
                "size_multiplier": multiplier.to_string(),
            }),
        })
    }

    /// Get the position size multiplier based on current microstructure conditions
    pub async fn get_size_multiplier(&self, token_id: &TokenId) -> Decimal {
        let config = self.config.read().await;
        let mut multiplier = Decimal::ONE;

        // Apply toxicity multiplier
        let toxicity = self.current_toxicity.read().await;
        if let Some(level) = toxicity.get(token_id) {
            let level_key = match level {
                ToxicityLevel::Low => "low",
                ToxicityLevel::Normal => "normal",
                ToxicityLevel::Elevated => "elevated",
                ToxicityLevel::High => "high",
            };
            if let Some(m) = config.vpin_trading.toxicity_position_multiplier.get(level_key) {
                multiplier *= *m;
            }
        }
        drop(toxicity);

        // Apply whale multiplier if recent whale activity
        let alerts = self.recent_whale_alerts.read().await;
        if let Some(token_alerts) = alerts.get(token_id) {
            let recent_count = token_alerts
                .iter()
                .filter(|a| Utc::now() - a.timestamp < Duration::minutes(5))
                .count();
            if recent_count > 0 {
                multiplier *= config.whale_trading.whale_position_multiplier;
            }
        }

        // Never reduce below 10%
        multiplier.max(dec!(0.1))
    }

    /// Get current toxicity level for a token
    pub async fn get_toxicity(&self, token_id: &TokenId) -> Option<ToxicityLevel> {
        self.current_toxicity.read().await.get(token_id).copied()
    }

    /// Get recent whale alerts for a token
    pub async fn get_recent_whale_alerts(&self, token_id: &TokenId) -> Vec<WhaleAlert> {
        self.recent_whale_alerts
            .read()
            .await
            .get(token_id)
            .map(|alerts| alerts.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Generate a combined microstructure signal
    pub async fn generate_microstructure_signal(
        &self,
        token_id: &TokenId,
        market_id: &str,
    ) -> Option<MicrostructureSignalEvent> {
        let toxicity = self.current_toxicity.read().await;
        let toxicity_level = toxicity.get(token_id).copied();
        drop(toxicity);

        let alerts = self.recent_whale_alerts.read().await;
        let whale_activity = alerts.get(token_id).and_then(|token_alerts| {
            let recent: Vec<_> = token_alerts
                .iter()
                .filter(|a| Utc::now() - a.timestamp < Duration::minutes(5))
                .collect();

            if recent.is_empty() {
                return None;
            }

            let mut buy_volume = Decimal::ZERO;
            let mut sell_volume = Decimal::ZERO;

            for alert in &recent {
                match alert.whale_trade.side {
                    Side::Buy => buy_volume += alert.whale_trade.size_usd,
                    Side::Sell => sell_volume += alert.whale_trade.size_usd,
                }
            }

            let total = buy_volume + sell_volume;
            let net_direction = if buy_volume >= sell_volume {
                Side::Buy
            } else {
                Side::Sell
            };

            Some(WhaleActivitySummary {
                recent_whale_trades: recent.len() as u32,
                net_whale_direction: net_direction,
                total_whale_volume_usd: total,
            })
        });
        drop(alerts);

        // Get VPIN
        let vpin_result = self.vpin_calculator.read().await.calculate_vpin(token_id);
        let vpin = vpin_result.as_ref().map(|r| r.vpin);

        let components = MicrostructureComponents {
            vpin,
            toxicity_level: toxicity_level.map(convert_toxicity_level),
            whale_activity: whale_activity.clone(),
            expected_impact_bps: None, // Would need market impact estimator
        };

        // Determine signal type and strength
        let (signal_type, strength, action) = self
            .evaluate_microstructure_conditions(&toxicity_level, &whale_activity)
            .await;

        let confidence = self.calculate_signal_confidence(&components).await;

        Some(MicrostructureSignalEvent::new(
            token_id.clone(),
            market_id.to_string(),
            signal_type,
            strength,
            confidence,
            components,
            action,
        ))
    }

    /// Evaluate microstructure conditions to determine signal
    async fn evaluate_microstructure_conditions(
        &self,
        toxicity: &Option<ToxicityLevel>,
        whale_activity: &Option<WhaleActivitySummary>,
    ) -> (MicrostructureSignalType, Decimal, MicrostructureAction) {
        let config = self.config.read().await;

        // High toxicity is unfavorable
        if let Some(ToxicityLevel::High) = toxicity {
            return (
                MicrostructureSignalType::HighToxicity,
                dec!(-0.8),
                MicrostructureAction::ReduceSize {
                    multiplier: dec!(0.4),
                },
            );
        }

        // Significant whale activity
        if let Some(ref activity) = whale_activity {
            if activity.total_whale_volume_usd >= config.whale_trading.avoid_whale_threshold_usd {
                if config.whale_trading.follow_whale {
                    let strength = match activity.net_whale_direction {
                        Side::Buy => dec!(0.6),
                        Side::Sell => dec!(-0.6),
                    };
                    return (
                        MicrostructureSignalType::WhaleFollow,
                        strength,
                        MicrostructureAction::None,
                    );
                } else {
                    return (
                        MicrostructureSignalType::WhaleAvoid,
                        dec!(-0.4),
                        MicrostructureAction::ReduceSize {
                            multiplier: dec!(0.6),
                        },
                    );
                }
            }
        }

        // Low toxicity is favorable
        if let Some(ToxicityLevel::Low) = toxicity {
            return (
                MicrostructureSignalType::Favorable,
                dec!(0.5),
                MicrostructureAction::IncreaseSize {
                    multiplier: dec!(1.2),
                },
            );
        }

        // Elevated toxicity
        if let Some(ToxicityLevel::Elevated) = toxicity {
            return (
                MicrostructureSignalType::Unfavorable,
                dec!(-0.3),
                MicrostructureAction::ReduceSize {
                    multiplier: dec!(0.7),
                },
            );
        }

        // Normal conditions
        (
            MicrostructureSignalType::Favorable,
            dec!(0.1),
            MicrostructureAction::None,
        )
    }

    /// Calculate confidence in the microstructure signal
    async fn calculate_signal_confidence(&self, components: &MicrostructureComponents) -> Decimal {
        let mut confidence = dec!(0.5);

        // VPIN adds confidence
        if components.vpin.is_some() {
            confidence += dec!(0.2);
        }

        // Whale activity adds confidence
        if let Some(ref activity) = components.whale_activity {
            if activity.recent_whale_trades >= 3 {
                confidence += dec!(0.2);
            } else if activity.recent_whale_trades >= 1 {
                confidence += dec!(0.1);
            }
        }

        confidence.min(dec!(0.95))
    }
}

#[async_trait]
impl Strategy for MicrostructureStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn process_event(
        &self,
        event: &SystemEvent,
        _state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        // Process microstructure events directly
        if let SystemEvent::Microstructure(micro_event) = event {
            return self.handle_microstructure_event(micro_event).await;
        }

        // Process trade events for VPIN/whale updates
        if let SystemEvent::TradeExecuted(trade) = event {
            // We would need bid/ask from orderbook state, but for now we skip this
            // In practice, the VPIN calculator would be fed from orderbook updates
            debug!(
                token_id = %trade.token_id,
                "TradeExecuted event received, would update VPIN/whale detection"
            );
        }

        Ok(vec![])
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(
            event,
            SystemEvent::OrderbookUpdate(_)
                | SystemEvent::TradeExecuted(_)
                | SystemEvent::Microstructure(_)
                | SystemEvent::PartialFill(_)
                | SystemEvent::FullFill(_)
        )
    }

    async fn initialize(&mut self, state: &dyn StateProvider) -> Result<(), StrategyError> {
        info!(
            strategy_id = %self.id,
            "Initializing microstructure strategy"
        );

        // Register all known markets
        for market in state.get_all_markets().await {
            self.register_token(
                market.yes_token_id.clone(),
                market.condition_id.clone(),
                Outcome::Yes,
            )
            .await;
            self.register_token(market.no_token_id.clone(), market.condition_id, Outcome::No)
                .await;
        }

        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }

    async fn reload_config(&mut self, config_content: &str) -> Result<(), StrategyError> {
        let new_config: MicrostructureStrategyConfig = toml::from_str(config_content)
            .map_err(|e| StrategyError::ConfigError(format!("Failed to parse config: {}", e)))?;

        // Update enabled state
        self.enabled.store(new_config.enabled, Ordering::SeqCst);

        // Update VPIN calculator config
        {
            let mut calculator = self.vpin_calculator.write().await;
            *calculator = VpinCalculator::new(new_config.vpin_config.clone());
        }

        // Update whale detector config
        {
            let mut detector = self.whale_detector.write().await;
            detector.reload_config(new_config.whale_detector_config.clone());
        }

        // Update main config
        {
            let mut config = self.config.write().await;
            *config = new_config;
        }

        info!(strategy_id = %self.id, "Reloaded microstructure strategy config");
        Ok(())
    }

    fn config_name(&self) -> &str {
        "microstructure"
    }
}

impl MicrostructureStrategy {
    /// Handle microstructure events from other components
    async fn handle_microstructure_event(
        &self,
        event: &MicrostructureEvent,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        let signals = vec![];

        match event {
            MicrostructureEvent::VpinUpdate(e) => {
                // Update our tracked toxicity level
                self.current_toxicity
                    .write()
                    .await
                    .insert(e.token_id.clone(), convert_core_toxicity_level(e.toxicity_level));
            }
            MicrostructureEvent::ToxicityChange(e) => {
                // Generate signal if toxicity changed significantly
                if e.is_increase() && e.new_level == CoreToxicityLevel::High {
                    warn!(
                        token_id = %e.token_id,
                        vpin = %e.vpin,
                        "High toxicity detected, consider reducing positions"
                    );
                }
            }
            MicrostructureEvent::WhaleDetected(e) => {
                // Track whale detection
                debug!(
                    token_id = %e.token_id,
                    alert_type = ?e.alert_type,
                    size_usd = %e.trade_size_usd,
                    "Whale detected event received"
                );
            }
            MicrostructureEvent::MicrostructureSignal(e) => {
                // React to combined signals
                if e.is_unfavorable() {
                    debug!(
                        token_id = %e.token_id,
                        signal_type = ?e.signal_type,
                        "Unfavorable microstructure conditions"
                    );
                }
            }
            MicrostructureEvent::ImpactPrediction(_) => {
                // Impact predictions could inform execution strategy
            }
        }

        Ok(signals)
    }
}

/// Convert vpin ToxicityLevel to core events ToxicityLevel
fn convert_toxicity_level(level: ToxicityLevel) -> CoreToxicityLevel {
    match level {
        ToxicityLevel::Low => CoreToxicityLevel::Low,
        ToxicityLevel::Normal => CoreToxicityLevel::Normal,
        ToxicityLevel::Elevated => CoreToxicityLevel::Elevated,
        ToxicityLevel::High => CoreToxicityLevel::High,
    }
}

/// Convert core ToxicityLevel to vpin ToxicityLevel
fn convert_core_toxicity_level(level: CoreToxicityLevel) -> ToxicityLevel {
    match level {
        CoreToxicityLevel::Low => ToxicityLevel::Low,
        CoreToxicityLevel::Normal => ToxicityLevel::Normal,
        CoreToxicityLevel::Elevated => ToxicityLevel::Elevated,
        CoreToxicityLevel::High => ToxicityLevel::High,
    }
}

/// Convert internal WhaleAlertType to events WhaleAlertType
fn convert_whale_alert_type(
    alert_type: &crate::whale_detector::WhaleAlertType,
) -> WhaleAlertType {
    match alert_type {
        crate::whale_detector::WhaleAlertType::SingleLargeTrade => {
            WhaleAlertType::SingleLargeTrade
        }
        crate::whale_detector::WhaleAlertType::CumulativeActivity => {
            WhaleAlertType::CumulativeActivity
        }
        crate::whale_detector::WhaleAlertType::KnownWhaleActive => {
            WhaleAlertType::KnownWhaleActive
        }
        crate::whale_detector::WhaleAlertType::WhaleReversal => WhaleAlertType::WhaleReversal,
    }
}

/// Convert internal WhaleAction to events WhaleAction
fn convert_whale_action(action: &crate::whale_detector::WhaleAction) -> WhaleAction {
    match action {
        crate::whale_detector::WhaleAction::None => WhaleAction::None,
        crate::whale_detector::WhaleAction::ReducePosition { multiplier } => {
            WhaleAction::ReducePosition {
                multiplier: *multiplier,
            }
        }
        crate::whale_detector::WhaleAction::HaltNewTrades => WhaleAction::HaltNewTrades,
        crate::whale_detector::WhaleAction::FollowWhale { direction } => {
            WhaleAction::FollowWhale {
                direction: *direction,
            }
        }
        crate::whale_detector::WhaleAction::Alert => WhaleAction::Alert,
    }
}

fn rand_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{:08x}", nanos)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> MicrostructureStrategyConfig {
        MicrostructureStrategyConfig::default()
    }

    #[tokio::test]
    async fn test_strategy_creation() {
        let config = default_config();
        let strategy = MicrostructureStrategy::new(config);

        assert_eq!(strategy.id(), "microstructure");
        assert_eq!(strategy.name(), "Market Microstructure Strategy");
        assert!(strategy.is_enabled());
    }

    #[tokio::test]
    async fn test_size_multiplier_default() {
        let config = default_config();
        let strategy = MicrostructureStrategy::new(config);

        // With no toxicity or whale data, multiplier should be 1.0
        let multiplier = strategy.get_size_multiplier(&"test_token".to_string()).await;
        assert_eq!(multiplier, dec!(1.0));
    }

    #[tokio::test]
    async fn test_size_multiplier_with_toxicity() {
        let config = default_config();
        let strategy = MicrostructureStrategy::new(config);

        // Set high toxicity
        strategy
            .current_toxicity
            .write()
            .await
            .insert("test_token".to_string(), ToxicityLevel::High);

        let multiplier = strategy.get_size_multiplier(&"test_token".to_string()).await;
        assert_eq!(multiplier, dec!(0.4)); // High toxicity = 0.4 multiplier
    }

    #[tokio::test]
    async fn test_size_multiplier_with_low_toxicity() {
        let config = default_config();
        let strategy = MicrostructureStrategy::new(config);

        // Set low toxicity
        strategy
            .current_toxicity
            .write()
            .await
            .insert("test_token".to_string(), ToxicityLevel::Low);

        let multiplier = strategy.get_size_multiplier(&"test_token".to_string()).await;
        assert_eq!(multiplier, dec!(1.2)); // Low toxicity = 1.2 multiplier (favorable)
    }

    #[tokio::test]
    async fn test_cooldown_behavior() {
        let config = MicrostructureStrategyConfig {
            cooldown_secs: 60,
            ..default_config()
        };
        let strategy = MicrostructureStrategy::new(config);
        let token_id = "test_token".to_string();

        // Initially not in cooldown
        assert!(!strategy.is_in_cooldown(&token_id).await);

        // Set cooldown
        strategy.set_cooldown(&token_id).await;

        // Now should be in cooldown
        assert!(strategy.is_in_cooldown(&token_id).await);
    }

    #[tokio::test]
    async fn test_hourly_limit() {
        let config = MicrostructureStrategyConfig {
            max_signals_per_hour: 2,
            ..default_config()
        };
        let strategy = MicrostructureStrategy::new(config);
        let token_id = "test_token".to_string();

        // Initially not exceeded
        assert!(!strategy.is_hourly_limit_exceeded().await);

        // Increment twice
        strategy.increment_signal_count(&token_id).await;
        strategy.increment_signal_count(&token_id).await;

        // Now should be exceeded
        assert!(strategy.is_hourly_limit_exceeded().await);
    }

    #[tokio::test]
    async fn test_register_token() {
        let config = default_config();
        let strategy = MicrostructureStrategy::new(config);

        strategy
            .register_token(
                "token_yes".to_string(),
                "market_1".to_string(),
                Outcome::Yes,
            )
            .await;

        let map = strategy.token_market_map.read().await;
        assert!(map.contains_key("token_yes"));
        let (market_id, outcome) = map.get("token_yes").unwrap();
        assert_eq!(market_id, "market_1");
        assert_eq!(*outcome, Outcome::Yes);
    }

    #[tokio::test]
    async fn test_toxicity_action_parsing() {
        let toml_str = r#"
            enabled = true
            [vpin_trading]
            enabled = true
            high_toxicity_action = { reduce_positions = { multiplier = "0.5" } }
            low_toxicity_action = "none"
        "#;

        let config: MicrostructureStrategyConfig =
            toml::from_str(toml_str).expect("Failed to parse config");

        assert!(config.vpin_trading.enabled);
        assert_eq!(
            config.vpin_trading.high_toxicity_action,
            ToxicityAction::ReducePositions {
                multiplier: dec!(0.5)
            }
        );
        assert_eq!(config.vpin_trading.low_toxicity_action, ToxicityAction::None);
    }

    #[tokio::test]
    async fn test_config_reload() {
        let config = default_config();
        let mut strategy = MicrostructureStrategy::new(config);

        let new_config_str = r#"
            enabled = false
            min_confidence = "0.8"
            cooldown_secs = 600
            max_signals_per_hour = 5
            order_size_usd = "200"

            [vpin_trading]
            enabled = true
            high_toxicity_action = "none"
            low_toxicity_action = "none"

            [whale_trading]
            enabled = false
            follow_whale = false
            follow_confidence_threshold = "0.8"
            avoid_whale_threshold_usd = "20000"
            whale_position_multiplier = "0.5"

            [impact_adjustments]
            enabled = true
            max_acceptable_impact_bps = "100"
            auto_slice_above_impact_bps = "50"
            delay_high_impact_secs = 120

            [vpin_config]
            enabled = true
            bucket_size_usd = "2000"
            lookback_buckets = 100
            high_toxicity_threshold = "0.7"
            low_toxicity_threshold = "0.3"
            trade_classification = "quote_midpoint"

            [whale_detector_config]
            enabled = true
            whale_threshold_usd = "10000"
            aggregate_threshold_usd = "20000"
            aggregate_window_secs = 300
            alert_on_detection = true
            track_addresses = true
            address_history_limit = 100
            cooldown_secs = 60
        "#;

        strategy.reload_config(new_config_str).await.unwrap();

        assert!(!strategy.is_enabled());

        let config = strategy.config.read().await;
        assert_eq!(config.min_confidence, dec!(0.8));
        assert_eq!(config.cooldown_secs, 600);
        assert!(!config.whale_trading.enabled);
    }

    #[tokio::test]
    async fn test_accepts_event() {
        let config = default_config();
        let strategy = MicrostructureStrategy::new(config);

        // Should accept these events
        assert!(strategy.accepts_event(&SystemEvent::TradeExecuted(
            polysniper_core::events::TradeExecutedEvent {
                order_id: "test".to_string(),
                signal: TradeSignal {
                    id: "test".to_string(),
                    strategy_id: "test".to_string(),
                    market_id: "test".to_string(),
                    token_id: "test".to_string(),
                    outcome: Outcome::Yes,
                    side: Side::Buy,
                    price: None,
                    size: Decimal::ZERO,
                    size_usd: Decimal::ZERO,
                    order_type: OrderType::Gtc,
                    priority: Priority::Normal,
                    timestamp: Utc::now(),
                    reason: "test".to_string(),
                    metadata: serde_json::Value::Null,
                },
                market_id: "test".to_string(),
                token_id: "test".to_string(),
                executed_price: dec!(0.5),
                executed_size: dec!(100),
                fees: Decimal::ZERO,
                timestamp: Utc::now(),
            }
        )));

        // Should not accept heartbeat
        assert!(!strategy.accepts_event(&SystemEvent::Heartbeat(
            polysniper_core::events::HeartbeatEvent {
                source: "test".to_string(),
                timestamp: Utc::now(),
            }
        )));
    }

    #[tokio::test]
    async fn test_evaluate_microstructure_conditions_high_toxicity() {
        let config = default_config();
        let strategy = MicrostructureStrategy::new(config);

        let (signal_type, strength, action) = strategy
            .evaluate_microstructure_conditions(&Some(ToxicityLevel::High), &None)
            .await;

        assert_eq!(signal_type, MicrostructureSignalType::HighToxicity);
        assert!(strength < Decimal::ZERO);
        matches!(action, MicrostructureAction::ReduceSize { .. });
    }

    #[tokio::test]
    async fn test_evaluate_microstructure_conditions_low_toxicity() {
        let config = default_config();
        let strategy = MicrostructureStrategy::new(config);

        let (signal_type, strength, action) = strategy
            .evaluate_microstructure_conditions(&Some(ToxicityLevel::Low), &None)
            .await;

        assert_eq!(signal_type, MicrostructureSignalType::Favorable);
        assert!(strength > Decimal::ZERO);
        matches!(action, MicrostructureAction::IncreaseSize { .. });
    }

    #[tokio::test]
    async fn test_evaluate_microstructure_conditions_whale_follow() {
        let config = default_config();
        let strategy = MicrostructureStrategy::new(config);

        let whale_activity = Some(WhaleActivitySummary {
            recent_whale_trades: 5,
            net_whale_direction: Side::Buy,
            total_whale_volume_usd: dec!(15000),
        });

        let (signal_type, strength, _action) = strategy
            .evaluate_microstructure_conditions(&Some(ToxicityLevel::Normal), &whale_activity)
            .await;

        assert_eq!(signal_type, MicrostructureSignalType::WhaleFollow);
        assert!(strength > Decimal::ZERO); // Bullish whale activity
    }

    #[tokio::test]
    async fn test_signal_confidence_calculation() {
        let config = default_config();
        let strategy = MicrostructureStrategy::new(config);

        // Minimal components
        let minimal = MicrostructureComponents {
            vpin: None,
            toxicity_level: None,
            whale_activity: None,
            expected_impact_bps: None,
        };
        let minimal_confidence = strategy.calculate_signal_confidence(&minimal).await;

        // Rich components
        let rich = MicrostructureComponents {
            vpin: Some(dec!(0.5)),
            toxicity_level: Some(CoreToxicityLevel::Normal),
            whale_activity: Some(WhaleActivitySummary {
                recent_whale_trades: 5,
                net_whale_direction: Side::Buy,
                total_whale_volume_usd: dec!(10000),
            }),
            expected_impact_bps: Some(dec!(10)),
        };
        let rich_confidence = strategy.calculate_signal_confidence(&rich).await;

        assert!(rich_confidence > minimal_confidence);
        assert!(rich_confidence <= dec!(0.95)); // Capped
    }

    #[tokio::test]
    async fn test_minimum_multiplier() {
        let config = MicrostructureStrategyConfig {
            vpin_trading: VpinTradingConfig {
                enabled: true,
                toxicity_position_multiplier: {
                    let mut m = HashMap::new();
                    m.insert("high".to_string(), dec!(0.01)); // Very low multiplier
                    m
                },
                ..Default::default()
            },
            whale_trading: WhaleTradingConfig {
                whale_position_multiplier: dec!(0.01), // Very low multiplier
                ..Default::default()
            },
            ..default_config()
        };
        let strategy = MicrostructureStrategy::new(config);

        // Set conditions that would result in very low multiplier
        strategy
            .current_toxicity
            .write()
            .await
            .insert("test_token".to_string(), ToxicityLevel::High);

        let multiplier = strategy.get_size_multiplier(&"test_token".to_string()).await;

        // Should be clamped to minimum 0.1
        assert_eq!(multiplier, dec!(0.1));
    }
}
