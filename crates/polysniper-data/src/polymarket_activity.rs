//! Polymarket Activity Client
//!
//! Tracks trader leaderboards, comment activity, and volume patterns
//! to identify smart money flow and market sentiment shifts.

use chrono::{DateTime, Utc};
use polysniper_core::{
    CommentActivityEvent, DataSourceError, MarketId, SmartMoneySignalEvent, SystemEvent,
    TokenId, TraderAction, VolumeAnomalyEvent,
};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration as StdDuration;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, info, warn};

const DEFAULT_TIMEOUT: StdDuration = StdDuration::from_secs(30);
const MAX_VOLUME_HISTORY: usize = 100;

/// Configuration for the Polymarket activity client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolymarketActivityConfig {
    /// Whether the activity client is enabled
    pub enabled: bool,
    /// Poll interval in seconds
    pub poll_interval_secs: u64,
    /// Number of top traders to track from leaderboard
    pub track_top_traders: u32,
    /// Number of periods to look back for volume anomaly detection
    pub volume_lookback_periods: u32,
    /// Minimum comments per hour to trigger activity spike
    pub comment_activity_threshold: u32,
    /// Minimum volume ratio to trigger anomaly (current/average)
    pub min_volume_ratio: Decimal,
    /// Base URL for Polymarket API
    pub api_base_url: String,
}

impl Default for PolymarketActivityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_secs: 60,
            track_top_traders: 100,
            volume_lookback_periods: 24,
            comment_activity_threshold: 10,
            min_volume_ratio: Decimal::new(3, 0),
            api_base_url: "https://gamma-api.polymarket.com".to_string(),
        }
    }
}

/// A trader's position in a market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraderPosition {
    /// Market ID
    pub market_id: MarketId,
    /// Token ID
    pub token_id: TokenId,
    /// Outcome (Yes/No)
    pub outcome: polysniper_core::Outcome,
    /// Position size in contracts
    pub size: Decimal,
    /// Average entry price
    pub avg_price: Decimal,
    /// When the position was opened/last updated
    pub timestamp: DateTime<Utc>,
}

/// A trader's profile from the leaderboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraderProfile {
    /// Trader's address
    pub address: String,
    /// Username if available
    pub username: Option<String>,
    /// Total profit/PnL
    pub profit_pnl: Decimal,
    /// Total volume traded
    pub volume_traded: Decimal,
    /// Win rate (0.0 - 1.0)
    pub win_rate: Decimal,
    /// Leaderboard rank
    pub rank: u32,
    /// Recent positions
    pub recent_positions: Vec<TraderPosition>,
    /// When the profile was last updated
    pub last_updated: DateTime<Utc>,
}

/// A snapshot of volume for a market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeSnapshot {
    /// Market ID
    pub market_id: MarketId,
    /// Volume in USD
    pub volume_usd: Decimal,
    /// Number of trades
    pub trade_count: u32,
    /// When the snapshot was taken
    pub timestamp: DateTime<Utc>,
}

/// Leaderboard response from API
#[derive(Debug, Clone, Deserialize)]
struct LeaderboardResponse {
    #[serde(default)]
    data: Vec<LeaderboardEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct LeaderboardEntry {
    #[serde(default)]
    address: String,
    #[serde(default)]
    username: Option<String>,
    #[serde(default)]
    profit: String,
    #[serde(default)]
    volume: String,
    #[serde(default)]
    win_rate: Option<String>,
    #[serde(default)]
    rank: Option<u32>,
}

/// Position response from API
#[derive(Debug, Clone, Deserialize)]
struct PositionsResponse {
    #[serde(default)]
    data: Vec<PositionEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct PositionEntry {
    #[serde(default)]
    market_id: String,
    #[serde(default)]
    token_id: String,
    #[serde(default)]
    outcome: String,
    #[serde(default)]
    size: String,
    #[serde(default)]
    avg_price: String,
}

/// Market activity response
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct MarketActivityResponse {
    #[serde(default)]
    volume_24h: Option<String>,
    #[serde(default)]
    trade_count_24h: Option<u32>,
    #[serde(default)]
    comment_count: Option<u32>,
}

/// Polymarket activity client for tracking smart money and market signals
pub struct PolymarketActivityClient {
    http_client: Client,
    event_tx: broadcast::Sender<SystemEvent>,
    config: PolymarketActivityConfig,
    trader_cache: Arc<RwLock<HashMap<String, TraderProfile>>>,
    volume_history: Arc<RwLock<HashMap<MarketId, VecDeque<VolumeSnapshot>>>>,
    previous_positions: Arc<RwLock<HashMap<String, Vec<TraderPosition>>>>,
}

impl PolymarketActivityClient {
    /// Create a new Polymarket activity client
    pub fn new(
        config: PolymarketActivityConfig,
        event_tx: broadcast::Sender<SystemEvent>,
    ) -> Self {
        let http_client = Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            http_client,
            event_tx,
            config,
            trader_cache: Arc::new(RwLock::new(HashMap::new())),
            volume_history: Arc::new(RwLock::new(HashMap::new())),
            previous_positions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: PolymarketActivityConfig) {
        self.config = config;
    }

    /// Fetch and process the trader leaderboard
    pub async fn poll_leaderboard(&self) -> Result<Vec<TraderProfile>, DataSourceError> {
        let url = format!(
            "{}/leaderboard?limit={}",
            self.config.api_base_url, self.config.track_top_traders
        );

        debug!("Fetching leaderboard from: {}", url);

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| DataSourceError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(DataSourceError::HttpError(format!(
                "Leaderboard API returned status {}",
                response.status()
            )));
        }

        let leaderboard: LeaderboardResponse = response
            .json()
            .await
            .map_err(|e| DataSourceError::ParseError(e.to_string()))?;

        let mut profiles = Vec::new();
        for (idx, entry) in leaderboard.data.into_iter().enumerate() {
            let profit = entry.profit.parse::<Decimal>().unwrap_or(Decimal::ZERO);
            let volume = entry.volume.parse::<Decimal>().unwrap_or(Decimal::ZERO);
            let win_rate = entry
                .win_rate
                .as_ref()
                .and_then(|s| s.parse::<Decimal>().ok())
                .unwrap_or(Decimal::ZERO);

            profiles.push(TraderProfile {
                address: entry.address,
                username: entry.username,
                profit_pnl: profit,
                volume_traded: volume,
                win_rate,
                rank: entry.rank.unwrap_or((idx + 1) as u32),
                recent_positions: Vec::new(),
                last_updated: Utc::now(),
            });
        }

        info!("Fetched {} traders from leaderboard", profiles.len());
        Ok(profiles)
    }

    /// Fetch positions for a specific trader
    pub async fn fetch_trader_positions(
        &self,
        address: &str,
    ) -> Result<Vec<TraderPosition>, DataSourceError> {
        let url = format!(
            "{}/users/{}/positions",
            self.config.api_base_url, address
        );

        debug!("Fetching positions for trader: {}", address);

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| DataSourceError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            if response.status().as_u16() == 404 {
                return Ok(Vec::new());
            }
            return Err(DataSourceError::HttpError(format!(
                "Positions API returned status {}",
                response.status()
            )));
        }

        let positions_resp: PositionsResponse = response
            .json()
            .await
            .map_err(|e| DataSourceError::ParseError(e.to_string()))?;

        let positions: Vec<TraderPosition> = positions_resp
            .data
            .into_iter()
            .filter_map(|p| {
                let outcome = if p.outcome.eq_ignore_ascii_case("yes") {
                    polysniper_core::Outcome::Yes
                } else if p.outcome.eq_ignore_ascii_case("no") {
                    polysniper_core::Outcome::No
                } else {
                    return None;
                };

                Some(TraderPosition {
                    market_id: p.market_id,
                    token_id: p.token_id,
                    outcome,
                    size: p.size.parse().unwrap_or(Decimal::ZERO),
                    avg_price: p.avg_price.parse().unwrap_or(Decimal::ZERO),
                    timestamp: Utc::now(),
                })
            })
            .collect();

        Ok(positions)
    }

    /// Fetch market activity data
    pub async fn fetch_market_activity(
        &self,
        market_id: &MarketId,
    ) -> Result<Option<VolumeSnapshot>, DataSourceError> {
        let url = format!(
            "{}/markets/{}/activity",
            self.config.api_base_url, market_id
        );

        debug!("Fetching market activity for: {}", market_id);

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| DataSourceError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            if response.status().as_u16() == 404 {
                return Ok(None);
            }
            return Err(DataSourceError::HttpError(format!(
                "Market activity API returned status {}",
                response.status()
            )));
        }

        let activity: MarketActivityResponse = response
            .json()
            .await
            .map_err(|e| DataSourceError::ParseError(e.to_string()))?;

        let volume = activity
            .volume_24h
            .as_ref()
            .and_then(|s| s.parse::<Decimal>().ok())
            .unwrap_or(Decimal::ZERO);

        Ok(Some(VolumeSnapshot {
            market_id: market_id.clone(),
            volume_usd: volume,
            trade_count: activity.trade_count_24h.unwrap_or(0),
            timestamp: Utc::now(),
        }))
    }

    /// Detect position changes and emit smart money signals
    pub async fn detect_position_changes(&self, profiles: &[TraderProfile]) {
        let mut previous = self.previous_positions.write().await;

        for profile in profiles {
            if let Ok(current_positions) = self.fetch_trader_positions(&profile.address).await {
                if let Some(prev_positions) = previous.get(&profile.address) {
                    // Compare positions to detect changes
                    for current in &current_positions {
                        let prev = prev_positions
                            .iter()
                            .find(|p| p.market_id == current.market_id && p.token_id == current.token_id);

                        let (action, size_change) = match prev {
                            Some(p) if current.size > p.size => {
                                (TraderAction::Buy, current.size - p.size)
                            }
                            Some(p) if current.size < p.size => {
                                (TraderAction::Sell, p.size - current.size)
                            }
                            None if current.size > Decimal::ZERO => {
                                (TraderAction::NewPosition, current.size)
                            }
                            _ => continue,
                        };

                        let size_usd = size_change * current.avg_price;

                        // Only signal significant positions
                        if size_usd < Decimal::new(100, 0) {
                            continue;
                        }

                        let event = SmartMoneySignalEvent {
                            market_id: current.market_id.clone(),
                            token_id: current.token_id.clone(),
                            trader_address: profile.address.clone(),
                            trader_username: profile.username.clone(),
                            trader_rank: profile.rank,
                            trader_profit: profile.profit_pnl,
                            action,
                            outcome: current.outcome,
                            size_usd,
                            timestamp: Utc::now(),
                        };

                        info!(
                            trader = %profile.address,
                            rank = profile.rank,
                            market = %current.market_id,
                            action = ?action,
                            size_usd = %size_usd,
                            "Smart money signal detected"
                        );

                        let _ = self.event_tx.send(SystemEvent::SmartMoneySignal(event));
                    }

                    // Detect closed positions
                    for prev_pos in prev_positions {
                        let still_exists = current_positions.iter().any(|c| {
                            c.market_id == prev_pos.market_id && c.token_id == prev_pos.token_id
                        });

                        if !still_exists && prev_pos.size > Decimal::ZERO {
                            let size_usd = prev_pos.size * prev_pos.avg_price;

                            if size_usd < Decimal::new(100, 0) {
                                continue;
                            }

                            let event = SmartMoneySignalEvent {
                                market_id: prev_pos.market_id.clone(),
                                token_id: prev_pos.token_id.clone(),
                                trader_address: profile.address.clone(),
                                trader_username: profile.username.clone(),
                                trader_rank: profile.rank,
                                trader_profit: profile.profit_pnl,
                                action: TraderAction::ClosePosition,
                                outcome: prev_pos.outcome,
                                size_usd,
                                timestamp: Utc::now(),
                            };

                            info!(
                                trader = %profile.address,
                                rank = profile.rank,
                                market = %prev_pos.market_id,
                                "Position closed by top trader"
                            );

                            let _ = self.event_tx.send(SystemEvent::SmartMoneySignal(event));
                        }
                    }
                }

                // Update previous positions
                previous.insert(profile.address.clone(), current_positions);
            }
        }
    }

    /// Record a volume snapshot and check for anomalies
    pub async fn record_volume_and_check_anomaly(
        &self,
        snapshot: VolumeSnapshot,
    ) -> Option<VolumeAnomalyEvent> {
        let mut history = self.volume_history.write().await;
        let market_history = history
            .entry(snapshot.market_id.clone())
            .or_insert_with(VecDeque::new);

        // Calculate average volume from history
        let lookback = self.config.volume_lookback_periods as usize;
        let avg_volume = if market_history.len() >= lookback {
            let sum: Decimal = market_history
                .iter()
                .take(lookback)
                .map(|s| s.volume_usd)
                .sum();
            sum / Decimal::from(lookback)
        } else if !market_history.is_empty() {
            let sum: Decimal = market_history.iter().map(|s| s.volume_usd).sum();
            sum / Decimal::from(market_history.len())
        } else {
            Decimal::ZERO
        };

        // Add current snapshot
        market_history.push_front(snapshot.clone());
        if market_history.len() > MAX_VOLUME_HISTORY {
            market_history.pop_back();
        }

        // Check for anomaly
        if avg_volume > Decimal::ZERO {
            let ratio = snapshot.volume_usd / avg_volume;
            if ratio >= self.config.min_volume_ratio {
                let event = VolumeAnomalyEvent {
                    market_id: snapshot.market_id,
                    current_volume: snapshot.volume_usd,
                    avg_volume,
                    volume_ratio: ratio,
                    trade_count: snapshot.trade_count,
                    net_flow: None,
                    timestamp: Utc::now(),
                };

                info!(
                    market = %event.market_id,
                    current = %event.current_volume,
                    average = %avg_volume,
                    ratio = %ratio,
                    "Volume anomaly detected"
                );

                let _ = self.event_tx.send(SystemEvent::VolumeAnomalyDetected(event.clone()));
                return Some(event);
            }
        }

        None
    }

    /// Check for comment activity spikes
    pub async fn check_comment_activity(
        &self,
        market_id: &MarketId,
        comment_count: u32,
        period_hours: Decimal,
    ) -> Option<CommentActivityEvent> {
        let velocity = if period_hours > Decimal::ZERO {
            Decimal::from(comment_count) / period_hours
        } else {
            Decimal::ZERO
        };

        if velocity >= Decimal::from(self.config.comment_activity_threshold) {
            let event = CommentActivityEvent {
                market_id: market_id.clone(),
                comment_count,
                comment_velocity: velocity,
                sentiment_hint: None,
                timestamp: Utc::now(),
            };

            info!(
                market = %market_id,
                count = comment_count,
                velocity = %velocity,
                "Comment activity spike detected"
            );

            let _ = self.event_tx.send(SystemEvent::CommentActivitySpike(event.clone()));
            return Some(event);
        }

        None
    }

    /// Run a full poll cycle
    pub async fn poll(&self) -> Result<(), DataSourceError> {
        if !self.config.enabled {
            return Ok(());
        }

        // Fetch leaderboard and update trader cache
        let profiles = self.poll_leaderboard().await?;

        // Update cache
        {
            let mut cache = self.trader_cache.write().await;
            for profile in &profiles {
                cache.insert(profile.address.clone(), profile.clone());
            }
        }

        // Detect position changes for top traders
        self.detect_position_changes(&profiles).await;

        Ok(())
    }

    /// Start the polling loop
    pub async fn start_polling(&self) {
        let interval = tokio::time::Duration::from_secs(self.config.poll_interval_secs);
        let mut ticker = tokio::time::interval(interval);

        loop {
            ticker.tick().await;

            if let Err(e) = self.poll().await {
                warn!("Polymarket activity poll error: {}", e);
            }
        }
    }

    /// Get cached trader profiles
    pub async fn get_cached_traders(&self) -> Vec<TraderProfile> {
        let cache = self.trader_cache.read().await;
        cache.values().cloned().collect()
    }

    /// Get a specific trader profile from cache
    pub async fn get_trader(&self, address: &str) -> Option<TraderProfile> {
        let cache = self.trader_cache.read().await;
        cache.get(address).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use tokio::sync::broadcast;

    fn test_config() -> PolymarketActivityConfig {
        PolymarketActivityConfig {
            enabled: true,
            poll_interval_secs: 60,
            track_top_traders: 10,
            volume_lookback_periods: 5,
            comment_activity_threshold: 5,
            min_volume_ratio: Decimal::new(2, 0),
            api_base_url: "https://test.api".to_string(),
        }
    }

    #[tokio::test]
    async fn test_volume_anomaly_detection() {
        let (tx, _rx) = broadcast::channel(100);
        let client = PolymarketActivityClient::new(test_config(), tx);

        // Record baseline snapshots
        for i in 1..=5 {
            let snapshot = VolumeSnapshot {
                market_id: "test-market".to_string(),
                volume_usd: Decimal::new(1000, 0),
                trade_count: 10,
                timestamp: Utc::now() - Duration::hours(i),
            };
            client.record_volume_and_check_anomaly(snapshot).await;
        }

        // Record anomalous volume
        let anomaly_snapshot = VolumeSnapshot {
            market_id: "test-market".to_string(),
            volume_usd: Decimal::new(5000, 0), // 5x normal
            trade_count: 50,
            timestamp: Utc::now(),
        };

        let result = client.record_volume_and_check_anomaly(anomaly_snapshot).await;
        assert!(result.is_some());

        let event = result.unwrap();
        assert_eq!(event.market_id, "test-market");
        assert!(event.volume_ratio >= Decimal::new(2, 0));
    }

    #[tokio::test]
    async fn test_comment_activity_spike() {
        let (tx, _rx) = broadcast::channel(100);
        let client = PolymarketActivityClient::new(test_config(), tx);

        // Test with high comment velocity
        let market_id = "test-market".to_string();
        let result = client
            .check_comment_activity(&market_id, 20, Decimal::ONE)
            .await;
        assert!(result.is_some());

        // Test with low comment velocity
        let result = client
            .check_comment_activity(&market_id, 2, Decimal::ONE)
            .await;
        assert!(result.is_none());
    }

    #[test]
    fn test_trader_profile_serialization() {
        let profile = TraderProfile {
            address: "0x123".to_string(),
            username: Some("trader1".to_string()),
            profit_pnl: Decimal::new(10000, 0),
            volume_traded: Decimal::new(100000, 0),
            win_rate: Decimal::new(65, 2),
            rank: 1,
            recent_positions: vec![],
            last_updated: Utc::now(),
        };

        let json = serde_json::to_string(&profile).unwrap();
        let parsed: TraderProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.address, profile.address);
        assert_eq!(parsed.profit_pnl, profile.profit_pnl);
    }

    #[test]
    fn test_volume_snapshot_serialization() {
        let snapshot = VolumeSnapshot {
            market_id: "test-market".to_string(),
            volume_usd: Decimal::new(5000, 0),
            trade_count: 100,
            timestamp: Utc::now(),
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        let parsed: VolumeSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.market_id, snapshot.market_id);
        assert_eq!(parsed.volume_usd, snapshot.volume_usd);
    }

    #[test]
    fn test_default_config() {
        let config = PolymarketActivityConfig::default();
        assert!(config.enabled);
        assert_eq!(config.poll_interval_secs, 60);
        assert_eq!(config.track_top_traders, 100);
        assert_eq!(config.min_volume_ratio, Decimal::new(3, 0));
    }
}
