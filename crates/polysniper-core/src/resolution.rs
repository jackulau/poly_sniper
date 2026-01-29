//! Resolution tracking types and configuration
//!
//! Types for monitoring market resolution and time-to-resolution calculations,
//! as well as configuration for automatic position exits before resolution.

use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Resolution status enum for markets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionStatus {
    /// Market is far from resolution (> 24h)
    Far,
    /// Market is approaching resolution (24h - 1h)
    Approaching,
    /// Market resolution is imminent (< 1h)
    Imminent,
    /// Market has resolved
    Resolved,
    /// Market has no known end date
    Unknown,
}

impl ResolutionStatus {
    /// Get the warning level for this status (higher = more urgent)
    pub fn warning_level(&self) -> u8 {
        match self {
            ResolutionStatus::Unknown => 0,
            ResolutionStatus::Far => 1,
            ResolutionStatus::Approaching => 2,
            ResolutionStatus::Imminent => 3,
            ResolutionStatus::Resolved => 4,
        }
    }

    /// Check if this status requires a warning notification
    pub fn requires_warning(&self) -> bool {
        matches!(
            self,
            ResolutionStatus::Approaching | ResolutionStatus::Imminent
        )
    }
}

impl std::fmt::Display for ResolutionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolutionStatus::Far => write!(f, "Far"),
            ResolutionStatus::Approaching => write!(f, "Approaching"),
            ResolutionStatus::Imminent => write!(f, "Imminent"),
            ResolutionStatus::Resolved => write!(f, "Resolved"),
            ResolutionStatus::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Configuration for resolution tracking thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionConfig {
    /// Whether resolution tracking is enabled
    pub enabled: bool,
    /// Poll interval for checking market status in seconds
    pub poll_interval_secs: u64,
    /// Threshold for "approaching" status in seconds (default 24 hours)
    pub approaching_threshold_secs: u64,
    /// Threshold for "imminent" status in seconds (default 1 hour)
    pub imminent_threshold_secs: u64,
    /// Warning thresholds to emit events at (in seconds before resolution)
    pub warning_thresholds_secs: Vec<u64>,
}

impl Default for ResolutionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_secs: 30,
            approaching_threshold_secs: 24 * 60 * 60, // 24 hours
            imminent_threshold_secs: 60 * 60,         // 1 hour
            warning_thresholds_secs: vec![
                24 * 60 * 60, // 24 hours
                60 * 60,      // 1 hour
                15 * 60,      // 15 minutes
            ],
        }
    }
}

impl ResolutionConfig {
    /// Calculate the resolution status based on time remaining
    pub fn calculate_status(&self, time_to_resolution: Option<Duration>) -> ResolutionStatus {
        match time_to_resolution {
            None => ResolutionStatus::Unknown,
            Some(duration) => {
                if duration <= Duration::zero() {
                    ResolutionStatus::Resolved
                } else if duration.num_seconds() <= self.imminent_threshold_secs as i64 {
                    ResolutionStatus::Imminent
                } else if duration.num_seconds() <= self.approaching_threshold_secs as i64 {
                    ResolutionStatus::Approaching
                } else {
                    ResolutionStatus::Far
                }
            }
        }
    }

    /// Check if a warning should be emitted for the given time remaining
    pub fn should_emit_warning(&self, time_remaining_secs: i64) -> Option<u64> {
        // Find the highest threshold that we're at or below
        self.warning_thresholds_secs
            .iter()
            .find(|&&threshold| time_remaining_secs <= threshold as i64 && time_remaining_secs > 0)
            .copied()
    }
}

/// Information about market resolution timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionInfo {
    /// Market condition ID
    pub market_id: String,
    /// When the market is expected to resolve
    pub end_date: Option<DateTime<Utc>>,
    /// Current resolution status
    pub status: ResolutionStatus,
    /// Time remaining until resolution
    pub time_remaining: Option<Duration>,
    /// Last time this info was updated
    pub last_checked: DateTime<Utc>,
    /// Warnings that have already been emitted (thresholds in seconds)
    pub emitted_warnings: Vec<u64>,
}

impl ResolutionInfo {
    /// Create new resolution info for a market
    pub fn new(market_id: String, end_date: Option<DateTime<Utc>>) -> Self {
        let now = Utc::now();
        let time_remaining = end_date.map(|end| end - now);
        let status = ResolutionConfig::default().calculate_status(time_remaining);

        Self {
            market_id,
            end_date,
            status,
            time_remaining,
            last_checked: now,
            emitted_warnings: Vec::new(),
        }
    }

    /// Update the resolution info with current time
    pub fn update(&mut self, config: &ResolutionConfig) {
        let now = Utc::now();
        self.time_remaining = self.end_date.map(|end| end - now);
        self.status = config.calculate_status(self.time_remaining);
        self.last_checked = now;
    }

    /// Check if we should emit a warning and which threshold
    pub fn check_warning(&mut self, config: &ResolutionConfig) -> Option<u64> {
        let time_remaining_secs = self.time_remaining?.num_seconds();
        if time_remaining_secs <= 0 {
            return None;
        }

        // Check each threshold from highest to lowest
        let mut sorted_thresholds = config.warning_thresholds_secs.clone();
        sorted_thresholds.sort_by(|a, b| b.cmp(a));

        for threshold in sorted_thresholds {
            if time_remaining_secs <= threshold as i64
                && !self.emitted_warnings.contains(&threshold)
            {
                self.emitted_warnings.push(threshold);
                return Some(threshold);
            }
        }

        None
    }

    /// Format time remaining as human-readable string
    pub fn format_time_remaining(&self) -> String {
        match self.time_remaining {
            None => "Unknown".to_string(),
            Some(duration) => {
                if duration <= Duration::zero() {
                    "Resolved".to_string()
                } else {
                    let total_secs = duration.num_seconds();
                    let days = total_secs / 86400;
                    let hours = (total_secs % 86400) / 3600;
                    let minutes = (total_secs % 3600) / 60;

                    if days > 0 {
                        format!("{}d {}h", days, hours)
                    } else if hours > 0 {
                        format!("{}h {}m", hours, minutes)
                    } else {
                        format!("{}m", minutes)
                    }
                }
            }
        }
    }
}

/// Resolution warning event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionWarning {
    /// Market condition ID
    pub market_id: String,
    /// Human-readable market question
    pub market_question: Option<String>,
    /// Current resolution status
    pub status: ResolutionStatus,
    /// Time remaining until resolution
    pub time_remaining_secs: i64,
    /// Warning threshold that was crossed (in seconds)
    pub warning_threshold_secs: u64,
    /// Formatted time remaining
    pub time_remaining_formatted: String,
}

// ============================================================================
// Resolution Exit Configuration Types
// ============================================================================

/// Order type for resolution exits
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitOrderType {
    /// Fill-or-kill (recommended for resolution exits)
    #[default]
    Fok,
    /// Good-til-cancelled limit order
    Gtc,
    /// Market order (immediate execution)
    Market,
}

/// Configuration for automatic position exits before resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionExitConfig {
    /// Whether the resolution exit strategy is enabled
    pub enabled: bool,

    /// Default time before resolution to exit (in seconds)
    /// Default: 3600 (1 hour)
    pub default_exit_before_secs: u64,

    /// Order type for exit orders (FOK recommended)
    #[serde(default)]
    pub exit_order_type: ExitOrderType,

    /// P&L floor in USD - exit early if unrealized P&L falls below this
    /// None means no P&L-based exit
    pub pnl_floor_usd: Option<Decimal>,

    /// P&L floor percentage - exit if unrealized P&L % is below this
    /// None means no percentage-based exit
    pub pnl_floor_pct: Option<Decimal>,

    /// Per-market exit timing overrides
    #[serde(default)]
    pub market_overrides: Vec<MarketExitOverride>,

    /// Markets explicitly flagged to hold through resolution
    #[serde(default)]
    pub hold_through_markets: Vec<String>,

    /// Maximum slippage tolerance for FOK orders (as decimal, e.g., 0.02 = 2%)
    #[serde(default = "default_max_slippage")]
    pub max_slippage: Decimal,

    /// Whether to log all exit decisions for analysis
    #[serde(default = "default_true")]
    pub log_exits: bool,
}

fn default_max_slippage() -> Decimal {
    Decimal::new(2, 2) // 0.02 = 2%
}

fn default_true() -> bool {
    true
}

impl Default for ResolutionExitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_exit_before_secs: 3600, // 1 hour
            exit_order_type: ExitOrderType::Fok,
            pnl_floor_usd: None,
            pnl_floor_pct: None,
            market_overrides: Vec::new(),
            hold_through_markets: Vec::new(),
            max_slippage: default_max_slippage(),
            log_exits: true,
        }
    }
}

impl ResolutionExitConfig {
    /// Get the exit time for a specific market (uses override if available)
    pub fn get_exit_before_secs(&self, market_id: &str) -> u64 {
        self.market_overrides
            .iter()
            .find(|o| o.market_id == market_id)
            .map(|o| o.exit_before_secs)
            .unwrap_or(self.default_exit_before_secs)
    }

    /// Check if a market should be held through resolution
    pub fn should_hold_through(&self, market_id: &str) -> bool {
        self.hold_through_markets.iter().any(|id| id == market_id)
    }

    /// Check if we should exit based on P&L thresholds
    pub fn should_exit_on_pnl(&self, unrealized_pnl: Decimal, position_value: Decimal) -> bool {
        // Check absolute USD floor
        if let Some(floor) = self.pnl_floor_usd {
            if unrealized_pnl < floor {
                return true;
            }
        }

        // Check percentage floor
        if let Some(floor_pct) = self.pnl_floor_pct {
            if !position_value.is_zero() {
                let pnl_pct = (unrealized_pnl / position_value) * Decimal::ONE_HUNDRED;
                if pnl_pct < floor_pct {
                    return true;
                }
            }
        }

        false
    }

    /// Check if we should exit based on time remaining
    pub fn should_exit_on_time(&self, market_id: &str, time_remaining_secs: i64) -> bool {
        if time_remaining_secs <= 0 {
            return false; // Already resolved
        }

        let exit_threshold = self.get_exit_before_secs(market_id) as i64;
        time_remaining_secs <= exit_threshold
    }
}

/// Per-market exit timing override
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketExitOverride {
    /// Market condition ID
    pub market_id: String,
    /// Time before resolution to exit (in seconds)
    pub exit_before_secs: u64,
    /// Optional custom order type for this market
    pub order_type: Option<ExitOrderType>,
    /// Optional custom P&L floor for this market
    pub pnl_floor_usd: Option<Decimal>,
}

/// Reason for a resolution exit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExitReason {
    /// Exit triggered by time threshold
    TimeThreshold {
        time_remaining_secs: i64,
        threshold_secs: u64,
    },
    /// Exit triggered by P&L floor
    PnlFloor {
        unrealized_pnl: Decimal,
        floor: Decimal,
        floor_type: PnlFloorType,
    },
    /// Exit triggered by market resolution
    MarketResolved,
    /// Manual exit requested
    Manual { reason: String },
}

impl std::fmt::Display for ExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExitReason::TimeThreshold {
                time_remaining_secs,
                threshold_secs,
            } => {
                write!(
                    f,
                    "Time threshold reached: {}s remaining (threshold: {}s)",
                    time_remaining_secs, threshold_secs
                )
            }
            ExitReason::PnlFloor {
                unrealized_pnl,
                floor,
                floor_type,
            } => {
                write!(
                    f,
                    "P&L floor breached: {} < {} ({})",
                    unrealized_pnl, floor, floor_type
                )
            }
            ExitReason::MarketResolved => write!(f, "Market resolved"),
            ExitReason::Manual { reason } => write!(f, "Manual exit: {}", reason),
        }
    }
}

/// Type of P&L floor that was breached
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PnlFloorType {
    /// Absolute USD floor
    Absolute,
    /// Percentage floor
    Percentage,
}

impl std::fmt::Display for PnlFloorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PnlFloorType::Absolute => write!(f, "USD"),
            PnlFloorType::Percentage => write!(f, "percentage"),
        }
    }
}

/// Tracked position for resolution exit strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedPosition {
    /// Market condition ID
    pub market_id: String,
    /// Token ID being held
    pub token_id: String,
    /// Position size
    pub size: Decimal,
    /// Average entry price
    pub avg_price: Decimal,
    /// Current unrealized P&L
    pub unrealized_pnl: Decimal,
    /// Market end date
    pub end_date: Option<DateTime<Utc>>,
    /// Whether exit signal has been generated
    pub exit_signal_generated: bool,
    /// Exit reason if signal was generated
    pub exit_reason: Option<ExitReason>,
    /// Whether this position should be held through resolution
    pub hold_through: bool,
}

impl TrackedPosition {
    /// Calculate position value
    pub fn position_value(&self) -> Decimal {
        self.size * self.avg_price
    }

    /// Get time remaining until resolution
    pub fn time_remaining(&self) -> Option<Duration> {
        self.end_date.map(|end| end - Utc::now())
    }

    /// Get time remaining in seconds
    pub fn time_remaining_secs(&self) -> Option<i64> {
        self.time_remaining().map(|d| d.num_seconds())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_resolution_status_from_time() {
        let config = ResolutionConfig::default();

        // Far (more than 24 hours)
        let far_duration = Duration::hours(48);
        assert_eq!(
            config.calculate_status(Some(far_duration)),
            ResolutionStatus::Far
        );

        // Approaching (between 1-24 hours)
        let approaching_duration = Duration::hours(12);
        assert_eq!(
            config.calculate_status(Some(approaching_duration)),
            ResolutionStatus::Approaching
        );

        // Imminent (less than 1 hour)
        let imminent_duration = Duration::minutes(30);
        assert_eq!(
            config.calculate_status(Some(imminent_duration)),
            ResolutionStatus::Imminent
        );

        // Resolved (zero or negative)
        let resolved_duration = Duration::minutes(-10);
        assert_eq!(
            config.calculate_status(Some(resolved_duration)),
            ResolutionStatus::Resolved
        );

        // Unknown (None)
        assert_eq!(config.calculate_status(None), ResolutionStatus::Unknown);
    }

    #[test]
    fn test_resolution_info_warnings() {
        let config = ResolutionConfig::default();
        let end_date = Utc::now() + Duration::hours(12); // 12 hours from now

        let mut info = ResolutionInfo::new("test_market".to_string(), Some(end_date));

        // Should emit 24-hour warning (since we're within 24 hours)
        let warning = info.check_warning(&config);
        assert_eq!(warning, Some(24 * 60 * 60));

        // Should not emit same warning again
        let warning2 = info.check_warning(&config);
        assert_eq!(warning2, None);

        // Simulate time passing to within 1 hour
        info.time_remaining = Some(Duration::minutes(45));
        let warning3 = info.check_warning(&config);
        assert_eq!(warning3, Some(60 * 60));

        // Simulate time passing to within 15 minutes
        info.time_remaining = Some(Duration::minutes(10));
        let warning4 = info.check_warning(&config);
        assert_eq!(warning4, Some(15 * 60));
    }

    #[test]
    fn test_format_time_remaining() {
        let mut info = ResolutionInfo::new("test".to_string(), None);

        info.time_remaining = Some(Duration::days(2) + Duration::hours(5));
        assert_eq!(info.format_time_remaining(), "2d 5h");

        info.time_remaining = Some(Duration::hours(3) + Duration::minutes(30));
        assert_eq!(info.format_time_remaining(), "3h 30m");

        info.time_remaining = Some(Duration::minutes(15));
        assert_eq!(info.format_time_remaining(), "15m");

        info.time_remaining = None;
        assert_eq!(info.format_time_remaining(), "Unknown");

        info.time_remaining = Some(Duration::minutes(-5));
        assert_eq!(info.format_time_remaining(), "Resolved");
    }

    #[test]
    fn test_warning_level() {
        assert_eq!(ResolutionStatus::Unknown.warning_level(), 0);
        assert_eq!(ResolutionStatus::Far.warning_level(), 1);
        assert_eq!(ResolutionStatus::Approaching.warning_level(), 2);
        assert_eq!(ResolutionStatus::Imminent.warning_level(), 3);
        assert_eq!(ResolutionStatus::Resolved.warning_level(), 4);
    }

    #[test]
    fn test_exit_config_defaults() {
        let config = ResolutionExitConfig::default();
        assert!(config.enabled);
        assert_eq!(config.default_exit_before_secs, 3600);
        assert_eq!(config.exit_order_type, ExitOrderType::Fok);
        assert!(config.pnl_floor_usd.is_none());
        assert!(config.market_overrides.is_empty());
    }

    #[test]
    fn test_exit_config_market_override() {
        let config = ResolutionExitConfig {
            market_overrides: vec![MarketExitOverride {
                market_id: "special_market".to_string(),
                exit_before_secs: 7200, // 2 hours
                order_type: None,
                pnl_floor_usd: None,
            }],
            ..Default::default()
        };

        assert_eq!(config.get_exit_before_secs("special_market"), 7200);
        assert_eq!(config.get_exit_before_secs("other_market"), 3600);
    }

    #[test]
    fn test_hold_through() {
        let config = ResolutionExitConfig {
            hold_through_markets: vec!["hold_market".to_string()],
            ..Default::default()
        };

        assert!(config.should_hold_through("hold_market"));
        assert!(!config.should_hold_through("other_market"));
    }

    #[test]
    fn test_pnl_floor_absolute() {
        let config = ResolutionExitConfig {
            pnl_floor_usd: Some(dec!(-50)),
            ..Default::default()
        };

        // Should exit: P&L is below floor
        assert!(config.should_exit_on_pnl(dec!(-100), dec!(500)));

        // Should not exit: P&L is above floor
        assert!(!config.should_exit_on_pnl(dec!(-25), dec!(500)));
        assert!(!config.should_exit_on_pnl(dec!(50), dec!(500)));
    }

    #[test]
    fn test_pnl_floor_percentage() {
        let config = ResolutionExitConfig {
            pnl_floor_pct: Some(dec!(-10)), // -10%
            ..Default::default()
        };

        // Should exit: -20% P&L (100 loss on 500 position)
        assert!(config.should_exit_on_pnl(dec!(-100), dec!(500)));

        // Should not exit: -5% P&L (25 loss on 500 position)
        assert!(!config.should_exit_on_pnl(dec!(-25), dec!(500)));
    }

    #[test]
    fn test_should_exit_on_time() {
        let config = ResolutionExitConfig::default(); // 1 hour default

        // Should exit: 30 minutes remaining
        assert!(config.should_exit_on_time("any_market", 30 * 60));

        // Should exit: exactly at threshold
        assert!(config.should_exit_on_time("any_market", 3600));

        // Should not exit: 2 hours remaining
        assert!(!config.should_exit_on_time("any_market", 2 * 3600));

        // Should not exit: already resolved
        assert!(!config.should_exit_on_time("any_market", 0));
        assert!(!config.should_exit_on_time("any_market", -100));
    }

    #[test]
    fn test_tracked_position() {
        let end_date = Utc::now() + Duration::hours(2);
        let position = TrackedPosition {
            market_id: "test_market".to_string(),
            token_id: "token_123".to_string(),
            size: dec!(100),
            avg_price: dec!(0.5),
            unrealized_pnl: dec!(10),
            end_date: Some(end_date),
            exit_signal_generated: false,
            exit_reason: None,
            hold_through: false,
        };

        assert_eq!(position.position_value(), dec!(50));
        assert!(position.time_remaining().is_some());
        let time_remaining = position.time_remaining_secs().unwrap();
        assert!(time_remaining > 7000 && time_remaining < 7300); // ~2 hours
    }
}
