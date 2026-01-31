//! Connection health monitoring and RTT tracking

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Maximum number of RTT samples to keep for statistics
const MAX_RTT_SAMPLES: usize = 100;

/// Connection health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Connection is healthy - recent pongs received, low failure count
    Healthy,
    /// Connection is degraded - delayed pongs or some failures
    Degraded,
    /// Connection is unhealthy - multiple failures or extended silence
    Unhealthy,
}

impl HealthStatus {
    /// Convert to numeric value for metrics (0=unhealthy, 1=degraded, 2=healthy)
    pub fn as_metric_value(&self) -> i64 {
        match self {
            HealthStatus::Unhealthy => 0,
            HealthStatus::Degraded => 1,
            HealthStatus::Healthy => 2,
        }
    }
}

/// Tracks connection health metrics
pub struct ConnectionHealth {
    /// Timestamp of last successful message (as unix millis)
    last_message_ms: AtomicU64,
    /// Timestamp of last successful pong (as unix millis)
    last_pong_ms: AtomicU64,
    /// Consecutive failure count
    failure_count: AtomicU32,
    /// RTT samples for latency tracking
    rtt_samples: Mutex<VecDeque<Duration>>,
    /// Reference instant for relative timing
    start_instant: Instant,
}

impl ConnectionHealth {
    /// Create a new connection health tracker
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            last_message_ms: AtomicU64::new(0),
            last_pong_ms: AtomicU64::new(0),
            failure_count: AtomicU32::new(0),
            rtt_samples: Mutex::new(VecDeque::with_capacity(MAX_RTT_SAMPLES)),
            start_instant: now,
        }
    }

    /// Record successful message receipt
    pub fn record_message(&self) {
        let elapsed = self.start_instant.elapsed().as_millis() as u64;
        self.last_message_ms.store(elapsed, Ordering::Release);
    }

    /// Record successful pong with RTT measurement
    pub fn record_pong(&self, rtt: Duration) {
        let elapsed = self.start_instant.elapsed().as_millis() as u64;
        self.last_pong_ms.store(elapsed, Ordering::Release);

        // Reset failure count on successful pong
        self.failure_count.store(0, Ordering::Release);

        // Record RTT sample
        let mut samples = self.rtt_samples.lock();
        if samples.len() >= MAX_RTT_SAMPLES {
            samples.pop_front();
        }
        samples.push_back(rtt);
    }

    /// Record a connection failure
    pub fn record_failure(&self) {
        self.failure_count.fetch_add(1, Ordering::AcqRel);
    }

    /// Reset health tracking (call on new connection)
    pub fn reset(&self) {
        let elapsed = self.start_instant.elapsed().as_millis() as u64;
        self.last_message_ms.store(elapsed, Ordering::Release);
        self.last_pong_ms.store(elapsed, Ordering::Release);
        self.failure_count.store(0, Ordering::Release);
        self.rtt_samples.lock().clear();
    }

    /// Get time since last message
    pub fn time_since_message(&self) -> Duration {
        let last = self.last_message_ms.load(Ordering::Acquire);
        let now = self.start_instant.elapsed().as_millis() as u64;
        Duration::from_millis(now.saturating_sub(last))
    }

    /// Get time since last pong
    pub fn time_since_pong(&self) -> Duration {
        let last = self.last_pong_ms.load(Ordering::Acquire);
        let now = self.start_instant.elapsed().as_millis() as u64;
        Duration::from_millis(now.saturating_sub(last))
    }

    /// Get consecutive failure count
    pub fn failure_count(&self) -> u32 {
        self.failure_count.load(Ordering::Acquire)
    }

    /// Get current health status
    pub fn status(&self) -> HealthStatus {
        let failures = self.failure_count.load(Ordering::Acquire);
        let since_pong = self.time_since_pong();

        // More than 3 consecutive failures = unhealthy
        if failures > 3 {
            return HealthStatus::Unhealthy;
        }

        // No pong for more than 60 seconds = degraded
        if since_pong > Duration::from_secs(60) {
            return HealthStatus::Degraded;
        }

        // Any failures = degraded
        if failures > 0 {
            return HealthStatus::Degraded;
        }

        // No pong for more than 30 seconds = degraded
        if since_pong > Duration::from_secs(30) {
            return HealthStatus::Degraded;
        }

        HealthStatus::Healthy
    }

    /// Get average RTT across all samples
    pub fn avg_rtt(&self) -> Option<Duration> {
        let samples = self.rtt_samples.lock();
        if samples.is_empty() {
            return None;
        }

        let total: Duration = samples.iter().sum();
        Some(total / samples.len() as u32)
    }

    /// Get median RTT (P50)
    pub fn median_rtt(&self) -> Option<Duration> {
        let samples = self.rtt_samples.lock();
        if samples.is_empty() {
            return None;
        }

        let mut sorted: Vec<Duration> = samples.iter().copied().collect();
        sorted.sort();
        Some(sorted[sorted.len() / 2])
    }

    /// Get P95 RTT
    pub fn p95_rtt(&self) -> Option<Duration> {
        let samples = self.rtt_samples.lock();
        if samples.is_empty() {
            return None;
        }

        let mut sorted: Vec<Duration> = samples.iter().copied().collect();
        sorted.sort();

        let idx = (sorted.len() as f64 * 0.95).ceil() as usize - 1;
        Some(sorted[idx.min(sorted.len() - 1)])
    }

    /// Get P99 RTT
    pub fn p99_rtt(&self) -> Option<Duration> {
        let samples = self.rtt_samples.lock();
        if samples.is_empty() {
            return None;
        }

        let mut sorted: Vec<Duration> = samples.iter().copied().collect();
        sorted.sort();

        let idx = (sorted.len() as f64 * 0.99).ceil() as usize - 1;
        Some(sorted[idx.min(sorted.len() - 1)])
    }

    /// Get minimum RTT
    pub fn min_rtt(&self) -> Option<Duration> {
        self.rtt_samples.lock().iter().min().copied()
    }

    /// Get maximum RTT
    pub fn max_rtt(&self) -> Option<Duration> {
        self.rtt_samples.lock().iter().max().copied()
    }

    /// Get number of RTT samples
    pub fn sample_count(&self) -> usize {
        self.rtt_samples.lock().len()
    }

    /// Get a snapshot of health metrics
    pub fn snapshot(&self) -> HealthSnapshot {
        HealthSnapshot {
            status: self.status(),
            failure_count: self.failure_count(),
            time_since_message: self.time_since_message(),
            time_since_pong: self.time_since_pong(),
            avg_rtt: self.avg_rtt(),
            p99_rtt: self.p99_rtt(),
            sample_count: self.sample_count(),
        }
    }
}

impl Default for ConnectionHealth {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of connection health metrics
#[derive(Debug, Clone)]
pub struct HealthSnapshot {
    pub status: HealthStatus,
    pub failure_count: u32,
    pub time_since_message: Duration,
    pub time_since_pong: Duration,
    pub avg_rtt: Option<Duration>,
    pub p99_rtt: Option<Duration>,
    pub sample_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let health = ConnectionHealth::new();
        assert_eq!(health.failure_count(), 0);
        assert_eq!(health.sample_count(), 0);
        // Status might be Degraded initially since no pong received yet
    }

    #[test]
    fn test_record_pong() {
        let health = ConnectionHealth::new();
        health.reset();

        health.record_pong(Duration::from_millis(10));
        health.record_pong(Duration::from_millis(20));
        health.record_pong(Duration::from_millis(15));

        assert_eq!(health.sample_count(), 3);
        assert!(health.avg_rtt().is_some());
        assert_eq!(health.status(), HealthStatus::Healthy);
    }

    #[test]
    fn test_record_failure() {
        let health = ConnectionHealth::new();
        health.reset();
        health.record_pong(Duration::from_millis(10));

        health.record_failure();
        assert_eq!(health.failure_count(), 1);
        assert_eq!(health.status(), HealthStatus::Degraded);

        health.record_failure();
        health.record_failure();
        health.record_failure();
        assert_eq!(health.failure_count(), 4);
        assert_eq!(health.status(), HealthStatus::Unhealthy);
    }

    #[test]
    fn test_failure_reset_on_pong() {
        let health = ConnectionHealth::new();
        health.record_failure();
        health.record_failure();
        assert_eq!(health.failure_count(), 2);

        health.record_pong(Duration::from_millis(10));
        assert_eq!(health.failure_count(), 0);
    }

    #[test]
    fn test_rtt_percentiles() {
        let health = ConnectionHealth::new();

        // Add 100 samples
        for i in 1..=100 {
            health.record_pong(Duration::from_millis(i));
        }

        assert_eq!(health.sample_count(), 100);
        assert_eq!(health.min_rtt(), Some(Duration::from_millis(1)));
        assert_eq!(health.max_rtt(), Some(Duration::from_millis(100)));

        // Median should be around 50-51
        let median = health.median_rtt().unwrap();
        assert!(median >= Duration::from_millis(50) && median <= Duration::from_millis(51));

        // P95 should be around 95
        let p95 = health.p95_rtt().unwrap();
        assert!(p95 >= Duration::from_millis(94) && p95 <= Duration::from_millis(96));

        // P99 should be around 99
        let p99 = health.p99_rtt().unwrap();
        assert!(p99 >= Duration::from_millis(98) && p99 <= Duration::from_millis(100));
    }

    #[test]
    fn test_max_samples() {
        let health = ConnectionHealth::new();

        // Add more than MAX_RTT_SAMPLES
        for i in 1..=150 {
            health.record_pong(Duration::from_millis(i));
        }

        // Should only keep MAX_RTT_SAMPLES
        assert_eq!(health.sample_count(), MAX_RTT_SAMPLES);

        // First samples should be dropped, so min should be 51
        assert_eq!(health.min_rtt(), Some(Duration::from_millis(51)));
    }

    #[test]
    fn test_health_status_metric_value() {
        assert_eq!(HealthStatus::Healthy.as_metric_value(), 2);
        assert_eq!(HealthStatus::Degraded.as_metric_value(), 1);
        assert_eq!(HealthStatus::Unhealthy.as_metric_value(), 0);
    }

    #[test]
    fn test_snapshot() {
        let health = ConnectionHealth::new();
        health.reset();
        health.record_pong(Duration::from_millis(25));

        let snapshot = health.snapshot();
        assert_eq!(snapshot.status, HealthStatus::Healthy);
        assert_eq!(snapshot.failure_count, 0);
        assert_eq!(snapshot.sample_count, 1);
        assert_eq!(snapshot.avg_rtt, Some(Duration::from_millis(25)));
    }
}
