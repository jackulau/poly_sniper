//! Resolution tracking types and configuration
//!
//! Types for monitoring market resolution and time-to-resolution calculations.

use chrono::{DateTime, Duration, Utc};
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
        for &threshold in &self.warning_thresholds_secs {
            if time_remaining_secs <= threshold as i64 && time_remaining_secs > 0 {
                return Some(threshold);
            }
        }
        None
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
            if time_remaining_secs <= threshold as i64 && !self.emitted_warnings.contains(&threshold)
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
