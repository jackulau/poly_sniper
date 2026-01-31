//! Latency metrics for tracking execution pipeline performance.
//!
//! Provides histogram-based tracking for:
//! - Event receipt to signal generation
//! - Signal to order submission
//! - Submission to confirmation
//! - End-to-end latency

use lazy_static::lazy_static;
use prometheus::{HistogramOpts, HistogramVec, Registry};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

lazy_static! {
    /// Latency from event receipt to signal generation
    pub static ref EVENT_TO_SIGNAL_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "polysniper_event_to_signal_latency_us",
            "Latency from event receipt to signal generation in microseconds"
        ).buckets(vec![10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]),
        &["strategy"]
    ).unwrap();

    /// Latency from signal generation to order submission
    pub static ref SIGNAL_TO_SUBMIT_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "polysniper_signal_to_submit_latency_us",
            "Latency from signal generation to order submission in microseconds"
        ).buckets(vec![100.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 25000.0, 50000.0, 100000.0]),
        &["order_type"]
    ).unwrap();

    /// Latency from order submission to confirmation
    pub static ref SUBMIT_TO_CONFIRM_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "polysniper_submit_to_confirm_latency_us",
            "Latency from order submission to confirmation in microseconds"
        ).buckets(vec![1000.0, 5000.0, 10000.0, 25000.0, 50000.0, 100000.0, 250000.0, 500000.0, 1000000.0]),
        &["endpoint"]
    ).unwrap();

    /// End-to-end latency from event to confirmation
    pub static ref END_TO_END_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "polysniper_end_to_end_latency_us",
            "End-to-end latency from event to order confirmation in microseconds"
        ).buckets(vec![1000.0, 5000.0, 10000.0, 25000.0, 50000.0, 100000.0, 250000.0, 500000.0, 1000000.0, 2000000.0]),
        &["strategy"]
    ).unwrap();

    /// WebSocket message parsing latency
    pub static ref WS_MESSAGE_PARSE_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "polysniper_ws_message_parse_latency_us",
            "WebSocket message parsing latency in microseconds"
        ).buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]),
        &["message_type"]
    ).unwrap();

    /// Connection pool routing latency
    pub static ref CONNECTION_ROUTING_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "polysniper_connection_routing_latency_us",
            "Connection pool routing decision latency in microseconds"
        ).buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0]),
        &["pool_type"]
    ).unwrap();
}

/// Latency statistics at various percentiles
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyStats {
    /// P50 (median) latency in microseconds
    pub p50_us: u64,
    /// P90 latency in microseconds
    pub p90_us: u64,
    /// P99 latency in microseconds
    pub p99_us: u64,
    /// Average latency in microseconds
    pub avg_us: u64,
    /// Minimum latency in microseconds
    pub min_us: u64,
    /// Maximum latency in microseconds
    pub max_us: u64,
    /// Sample count
    pub count: u64,
}

/// Latency metrics collector for pipeline stages
pub struct LatencyMetrics {
    /// Internal samples for event to signal latency
    event_to_signal_samples: std::sync::RwLock<Vec<u64>>,
    /// Internal samples for signal to submit latency
    signal_to_submit_samples: std::sync::RwLock<Vec<u64>>,
    /// Internal samples for submit to confirm latency
    submit_to_confirm_samples: std::sync::RwLock<Vec<u64>>,
    /// Internal samples for end-to-end latency
    end_to_end_samples: std::sync::RwLock<Vec<u64>>,
    /// Internal samples for WebSocket message parsing
    ws_parse_samples: std::sync::RwLock<Vec<u64>>,
    /// Maximum samples to keep
    max_samples: usize,
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self::new(10000)
    }
}

impl LatencyMetrics {
    /// Create a new latency metrics collector
    pub fn new(max_samples: usize) -> Self {
        Self {
            event_to_signal_samples: std::sync::RwLock::new(Vec::with_capacity(max_samples)),
            signal_to_submit_samples: std::sync::RwLock::new(Vec::with_capacity(max_samples)),
            submit_to_confirm_samples: std::sync::RwLock::new(Vec::with_capacity(max_samples)),
            end_to_end_samples: std::sync::RwLock::new(Vec::with_capacity(max_samples)),
            ws_parse_samples: std::sync::RwLock::new(Vec::with_capacity(max_samples)),
            max_samples,
        }
    }

    /// Record event to signal latency
    pub fn record_event_to_signal(&self, duration: Duration, strategy: &str) {
        let micros = duration.as_micros() as u64;
        EVENT_TO_SIGNAL_LATENCY
            .with_label_values(&[strategy])
            .observe(micros as f64);

        self.add_sample(&self.event_to_signal_samples, micros);
        debug!(latency_us = micros, strategy = strategy, "Event to signal latency");
    }

    /// Record signal to submit latency
    pub fn record_signal_to_submit(&self, duration: Duration, order_type: &str) {
        let micros = duration.as_micros() as u64;
        SIGNAL_TO_SUBMIT_LATENCY
            .with_label_values(&[order_type])
            .observe(micros as f64);

        self.add_sample(&self.signal_to_submit_samples, micros);
        debug!(latency_us = micros, order_type = order_type, "Signal to submit latency");
    }

    /// Record submit to confirm latency
    pub fn record_submit_to_confirm(&self, duration: Duration, endpoint: &str) {
        let micros = duration.as_micros() as u64;
        SUBMIT_TO_CONFIRM_LATENCY
            .with_label_values(&[endpoint])
            .observe(micros as f64);

        self.add_sample(&self.submit_to_confirm_samples, micros);
        debug!(latency_us = micros, endpoint = endpoint, "Submit to confirm latency");
    }

    /// Record end-to-end latency
    pub fn record_end_to_end(&self, duration: Duration, strategy: &str) {
        let micros = duration.as_micros() as u64;
        END_TO_END_LATENCY
            .with_label_values(&[strategy])
            .observe(micros as f64);

        self.add_sample(&self.end_to_end_samples, micros);
        debug!(latency_us = micros, strategy = strategy, "End-to-end latency");
    }

    /// Record WebSocket message parse latency
    pub fn record_ws_parse(&self, duration: Duration, message_type: &str) {
        let micros = duration.as_micros() as u64;
        WS_MESSAGE_PARSE_LATENCY
            .with_label_values(&[message_type])
            .observe(micros as f64);

        self.add_sample(&self.ws_parse_samples, micros);
    }

    /// Record connection routing latency
    pub fn record_connection_routing(&self, duration: Duration, pool_type: &str) {
        let micros = duration.as_micros() as u64;
        CONNECTION_ROUTING_LATENCY
            .with_label_values(&[pool_type])
            .observe(micros as f64);
    }

    /// Add a sample to a sample vector
    fn add_sample(&self, samples: &std::sync::RwLock<Vec<u64>>, value: u64) {
        let mut samples = samples.write().unwrap();
        if samples.len() >= self.max_samples {
            samples.remove(0);
        }
        samples.push(value);
    }

    /// Calculate statistics from samples
    fn calculate_stats(samples: &[u64]) -> LatencyStats {
        if samples.is_empty() {
            return LatencyStats::default();
        }

        let mut sorted = samples.to_vec();
        sorted.sort_unstable();

        let count = sorted.len();
        let sum: u64 = sorted.iter().sum();

        LatencyStats {
            p50_us: sorted[count / 2],
            p90_us: sorted[(count * 90) / 100],
            p99_us: sorted[(count * 99) / 100],
            avg_us: sum / count as u64,
            min_us: sorted[0],
            max_us: sorted[count - 1],
            count: count as u64,
        }
    }

    /// Get event to signal latency statistics
    pub fn get_event_to_signal_stats(&self) -> LatencyStats {
        let samples = self.event_to_signal_samples.read().unwrap();
        Self::calculate_stats(&samples)
    }

    /// Get signal to submit latency statistics
    pub fn get_signal_to_submit_stats(&self) -> LatencyStats {
        let samples = self.signal_to_submit_samples.read().unwrap();
        Self::calculate_stats(&samples)
    }

    /// Get submit to confirm latency statistics
    pub fn get_submit_to_confirm_stats(&self) -> LatencyStats {
        let samples = self.submit_to_confirm_samples.read().unwrap();
        Self::calculate_stats(&samples)
    }

    /// Get end-to-end latency statistics
    pub fn get_end_to_end_stats(&self) -> LatencyStats {
        let samples = self.end_to_end_samples.read().unwrap();
        Self::calculate_stats(&samples)
    }

    /// Get WebSocket parse latency statistics
    pub fn get_ws_parse_stats(&self) -> LatencyStats {
        let samples = self.ws_parse_samples.read().unwrap();
        Self::calculate_stats(&samples)
    }

    /// Get P50 latencies for all stages
    pub fn get_p50(&self) -> PipelineLatencies {
        PipelineLatencies {
            event_to_signal_us: self.get_event_to_signal_stats().p50_us,
            signal_to_submit_us: self.get_signal_to_submit_stats().p50_us,
            submit_to_confirm_us: self.get_submit_to_confirm_stats().p50_us,
            end_to_end_us: self.get_end_to_end_stats().p50_us,
        }
    }

    /// Get P99 latencies for all stages
    pub fn get_p99(&self) -> PipelineLatencies {
        PipelineLatencies {
            event_to_signal_us: self.get_event_to_signal_stats().p99_us,
            signal_to_submit_us: self.get_signal_to_submit_stats().p99_us,
            submit_to_confirm_us: self.get_submit_to_confirm_stats().p99_us,
            end_to_end_us: self.get_end_to_end_stats().p99_us,
        }
    }

    /// Clear all samples
    pub fn clear(&self) {
        self.event_to_signal_samples.write().unwrap().clear();
        self.signal_to_submit_samples.write().unwrap().clear();
        self.submit_to_confirm_samples.write().unwrap().clear();
        self.end_to_end_samples.write().unwrap().clear();
        self.ws_parse_samples.write().unwrap().clear();
    }
}

/// Pipeline latencies at a specific percentile
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineLatencies {
    /// Event to signal latency in microseconds
    pub event_to_signal_us: u64,
    /// Signal to submit latency in microseconds
    pub signal_to_submit_us: u64,
    /// Submit to confirm latency in microseconds
    pub submit_to_confirm_us: u64,
    /// End-to-end latency in microseconds
    pub end_to_end_us: u64,
}

/// Register latency metrics with the Prometheus registry
pub fn register_latency_metrics(registry: &Registry) {
    registry
        .register(Box::new(EVENT_TO_SIGNAL_LATENCY.clone()))
        .ok();
    registry
        .register(Box::new(SIGNAL_TO_SUBMIT_LATENCY.clone()))
        .ok();
    registry
        .register(Box::new(SUBMIT_TO_CONFIRM_LATENCY.clone()))
        .ok();
    registry
        .register(Box::new(END_TO_END_LATENCY.clone()))
        .ok();
    registry
        .register(Box::new(WS_MESSAGE_PARSE_LATENCY.clone()))
        .ok();
    registry
        .register(Box::new(CONNECTION_ROUTING_LATENCY.clone()))
        .ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_stats_calculation() {
        let samples = vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
        let stats = LatencyMetrics::calculate_stats(&samples);

        assert_eq!(stats.count, 10);
        assert_eq!(stats.min_us, 100);
        assert_eq!(stats.max_us, 1000);
        assert_eq!(stats.avg_us, 550);
        assert_eq!(stats.p50_us, 600); // Middle element
    }

    #[test]
    fn test_latency_stats_empty() {
        let stats = LatencyMetrics::calculate_stats(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.avg_us, 0);
    }

    #[test]
    fn test_latency_metrics_recording() {
        let metrics = LatencyMetrics::new(100);

        metrics.record_event_to_signal(Duration::from_micros(100), "test_strategy");
        metrics.record_event_to_signal(Duration::from_micros(200), "test_strategy");
        metrics.record_event_to_signal(Duration::from_micros(300), "test_strategy");

        let stats = metrics.get_event_to_signal_stats();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.avg_us, 200);
    }

    #[test]
    fn test_latency_metrics_max_samples() {
        let metrics = LatencyMetrics::new(5);

        for i in 1..=10 {
            metrics.record_event_to_signal(Duration::from_micros(i * 100), "test");
        }

        let stats = metrics.get_event_to_signal_stats();
        assert_eq!(stats.count, 5); // Only keeps 5 samples
        assert_eq!(stats.min_us, 600); // Oldest samples were removed
    }

    #[test]
    fn test_pipeline_latencies() {
        let metrics = LatencyMetrics::new(100);

        metrics.record_event_to_signal(Duration::from_micros(100), "test");
        metrics.record_signal_to_submit(Duration::from_micros(500), "GTC");
        metrics.record_submit_to_confirm(Duration::from_micros(10000), "clob");
        metrics.record_end_to_end(Duration::from_micros(15000), "test");

        let p50 = metrics.get_p50();
        assert_eq!(p50.event_to_signal_us, 100);
        assert_eq!(p50.signal_to_submit_us, 500);
        assert_eq!(p50.submit_to_confirm_us, 10000);
        assert_eq!(p50.end_to_end_us, 15000);
    }

    #[test]
    fn test_clear_metrics() {
        let metrics = LatencyMetrics::new(100);

        metrics.record_event_to_signal(Duration::from_micros(100), "test");
        assert_eq!(metrics.get_event_to_signal_stats().count, 1);

        metrics.clear();
        assert_eq!(metrics.get_event_to_signal_stats().count, 0);
    }
}
