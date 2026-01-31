//! Connection pool for low-latency connections to CLOB and RPC endpoints.
//!
//! Provides warm connection management, latency-based routing, and automatic failover.

use chrono::{DateTime, Utc};
use reqwest::{Client, ClientBuilder};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Maximum latency samples to keep per connection
const MAX_LATENCY_SAMPLES: usize = 100;

/// Default connection timeout
const DEFAULT_CONNECTION_TIMEOUT_MS: u64 = 5000;

/// Default request timeout
const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 2000;

/// Configuration for the connection pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolConfig {
    /// Number of CLOB REST connections to maintain
    pub clob_pool_size: usize,
    /// Number of Polygon RPC connections to maintain
    pub rpc_pool_size: usize,
    /// Connection establishment timeout in milliseconds
    pub connection_timeout_ms: u64,
    /// Per-request timeout in milliseconds
    pub request_timeout_ms: u64,
    /// Health check interval in seconds
    pub health_check_interval_secs: u64,
    /// Route requests to the fastest responding connection
    pub prefer_fastest: bool,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            clob_pool_size: 3,
            rpc_pool_size: 2,
            connection_timeout_ms: DEFAULT_CONNECTION_TIMEOUT_MS,
            request_timeout_ms: DEFAULT_REQUEST_TIMEOUT_MS,
            health_check_interval_secs: 30,
            prefer_fastest: true,
        }
    }
}

/// Latency statistics for a connection
#[derive(Debug, Clone, Default)]
pub struct LatencyStats {
    /// Average latency in microseconds
    pub avg_us: u64,
    /// P50 latency in microseconds
    pub p50_us: u64,
    /// P99 latency in microseconds
    pub p99_us: u64,
    /// Minimum latency in microseconds
    pub min_us: u64,
    /// Maximum latency in microseconds
    pub max_us: u64,
    /// Number of samples
    pub sample_count: usize,
}

/// Tracks health and latency metrics for connections
#[derive(Debug)]
pub struct HealthTracker {
    latencies: HashMap<String, VecDeque<Duration>>,
    failures: HashMap<String, u32>,
    last_check: DateTime<Utc>,
}

impl Default for HealthTracker {
    fn default() -> Self {
        Self {
            latencies: HashMap::new(),
            failures: HashMap::new(),
            last_check: Utc::now(),
        }
    }
}

impl HealthTracker {
    /// Create a new health tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful request latency
    pub fn record_latency(&mut self, connection_id: &str, latency: Duration) {
        let samples = self
            .latencies
            .entry(connection_id.to_string())
            .or_default();

        samples.push_back(latency);
        if samples.len() > MAX_LATENCY_SAMPLES {
            samples.pop_front();
        }

        // Reset failure count on success
        self.failures.insert(connection_id.to_string(), 0);
    }

    /// Record a failure for a connection
    pub fn record_failure(&mut self, connection_id: &str) {
        *self.failures.entry(connection_id.to_string()).or_insert(0) += 1;
    }

    /// Get the failure count for a connection
    pub fn failure_count(&self, connection_id: &str) -> u32 {
        self.failures.get(connection_id).copied().unwrap_or(0)
    }

    /// Get latency statistics for a connection
    pub fn get_stats(&self, connection_id: &str) -> LatencyStats {
        let Some(samples) = self.latencies.get(connection_id) else {
            return LatencyStats::default();
        };

        if samples.is_empty() {
            return LatencyStats::default();
        }

        let mut sorted: Vec<u64> = samples.iter().map(|d| d.as_micros() as u64).collect();
        sorted.sort_unstable();

        let sum: u64 = sorted.iter().sum();
        let count = sorted.len();

        LatencyStats {
            avg_us: sum / count as u64,
            p50_us: sorted[count / 2],
            p99_us: sorted[(count * 99) / 100],
            min_us: sorted[0],
            max_us: sorted[count - 1],
            sample_count: count,
        }
    }

    /// Get the average latency for a connection in microseconds
    pub fn avg_latency_us(&self, connection_id: &str) -> Option<u64> {
        let samples = self.latencies.get(connection_id)?;
        if samples.is_empty() {
            return None;
        }
        let sum: u64 = samples.iter().map(|d| d.as_micros() as u64).sum();
        Some(sum / samples.len() as u64)
    }

    /// Check if a connection is healthy (less than 3 consecutive failures)
    pub fn is_healthy(&self, connection_id: &str) -> bool {
        self.failure_count(connection_id) < 3
    }

    /// Update the last check time
    pub fn update_check_time(&mut self) {
        self.last_check = Utc::now();
    }
}

/// A single connection to the CLOB REST API
#[derive(Debug)]
pub struct ClobConnection {
    /// Connection identifier
    pub id: String,
    /// Base URL for this connection
    pub base_url: String,
    /// HTTP client for this connection
    pub client: Client,
}

impl ClobConnection {
    /// Create a new CLOB connection
    pub fn new(id: String, base_url: String, config: &ConnectionPoolConfig) -> Self {
        let client = ClientBuilder::new()
            .connect_timeout(Duration::from_millis(config.connection_timeout_ms))
            .timeout(Duration::from_millis(config.request_timeout_ms))
            .tcp_nodelay(true)
            .pool_max_idle_per_host(1)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            id,
            base_url,
            client,
        }
    }
}

/// A single connection to a Polygon RPC endpoint
#[derive(Debug)]
pub struct RpcConnection {
    /// Connection identifier
    pub id: String,
    /// RPC endpoint URL
    pub endpoint_url: String,
    /// HTTP client for this connection
    pub client: Client,
}

impl RpcConnection {
    /// Create a new RPC connection
    pub fn new(id: String, endpoint_url: String, config: &ConnectionPoolConfig) -> Self {
        let client = ClientBuilder::new()
            .connect_timeout(Duration::from_millis(config.connection_timeout_ms))
            .timeout(Duration::from_millis(config.request_timeout_ms))
            .tcp_nodelay(true)
            .pool_max_idle_per_host(1)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            id,
            endpoint_url,
            client,
        }
    }
}

/// Connection pool for managing multiple warm connections
pub struct ConnectionPool {
    clob_connections: Vec<ClobConnection>,
    rpc_connections: Vec<RpcConnection>,
    config: ConnectionPoolConfig,
    health_tracker: Arc<RwLock<HealthTracker>>,
    clob_round_robin: AtomicUsize,
    rpc_round_robin: AtomicUsize,
}

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new(
        clob_base_url: String,
        rpc_endpoint_urls: Vec<String>,
        config: ConnectionPoolConfig,
    ) -> Self {
        // Create CLOB connections
        let clob_connections: Vec<ClobConnection> = (0..config.clob_pool_size)
            .map(|i| ClobConnection::new(format!("clob_{}", i), clob_base_url.clone(), &config))
            .collect();

        // Create RPC connections - distribute across provided endpoints
        let rpc_connections: Vec<RpcConnection> = (0..config.rpc_pool_size)
            .map(|i| {
                let endpoint = if rpc_endpoint_urls.is_empty() {
                    "https://polygon-rpc.com".to_string()
                } else {
                    rpc_endpoint_urls[i % rpc_endpoint_urls.len()].clone()
                };
                RpcConnection::new(format!("rpc_{}", i), endpoint, &config)
            })
            .collect();

        info!(
            clob_pool_size = clob_connections.len(),
            rpc_pool_size = rpc_connections.len(),
            "Connection pool initialized"
        );

        Self {
            clob_connections,
            rpc_connections,
            config,
            health_tracker: Arc::new(RwLock::new(HealthTracker::new())),
            clob_round_robin: AtomicUsize::new(0),
            rpc_round_robin: AtomicUsize::new(0),
        }
    }

    /// Get the best CLOB connection based on latency or round-robin
    pub async fn get_clob_connection(&self) -> Option<&ClobConnection> {
        if self.clob_connections.is_empty() {
            return None;
        }

        if self.config.prefer_fastest {
            // Find the fastest healthy connection
            let tracker = self.health_tracker.read().await;
            let mut best_conn: Option<&ClobConnection> = None;
            let mut best_latency = u64::MAX;

            for conn in &self.clob_connections {
                if tracker.is_healthy(&conn.id) {
                    let latency = tracker.avg_latency_us(&conn.id).unwrap_or(u64::MAX / 2);
                    if latency < best_latency {
                        best_latency = latency;
                        best_conn = Some(conn);
                    }
                }
            }

            // Fall back to round-robin if no healthy connection with latency data
            if best_conn.is_some() {
                return best_conn;
            }
        }

        // Round-robin fallback
        let idx = self.clob_round_robin.fetch_add(1, Ordering::Relaxed) % self.clob_connections.len();
        Some(&self.clob_connections[idx])
    }

    /// Get the best RPC connection based on latency or round-robin
    pub async fn get_rpc_connection(&self) -> Option<&RpcConnection> {
        if self.rpc_connections.is_empty() {
            return None;
        }

        if self.config.prefer_fastest {
            // Find the fastest healthy connection
            let tracker = self.health_tracker.read().await;
            let mut best_conn: Option<&RpcConnection> = None;
            let mut best_latency = u64::MAX;

            for conn in &self.rpc_connections {
                if tracker.is_healthy(&conn.id) {
                    let latency = tracker.avg_latency_us(&conn.id).unwrap_or(u64::MAX / 2);
                    if latency < best_latency {
                        best_latency = latency;
                        best_conn = Some(conn);
                    }
                }
            }

            // Fall back to round-robin if no healthy connection with latency data
            if best_conn.is_some() {
                return best_conn;
            }
        }

        // Round-robin fallback
        let idx = self.rpc_round_robin.fetch_add(1, Ordering::Relaxed) % self.rpc_connections.len();
        Some(&self.rpc_connections[idx])
    }

    /// Record a successful request latency
    pub async fn record_latency(&self, connection_id: &str, latency: Duration) {
        let mut tracker = self.health_tracker.write().await;
        tracker.record_latency(connection_id, latency);
        debug!(
            connection_id = connection_id,
            latency_us = latency.as_micros(),
            "Recorded latency"
        );
    }

    /// Record a failure for a connection
    pub async fn record_failure(&self, connection_id: &str) {
        let mut tracker = self.health_tracker.write().await;
        tracker.record_failure(connection_id);
        let failures = tracker.failure_count(connection_id);
        warn!(
            connection_id = connection_id,
            failures = failures,
            "Recorded connection failure"
        );
    }

    /// Get latency statistics for a connection
    pub async fn get_stats(&self, connection_id: &str) -> LatencyStats {
        let tracker = self.health_tracker.read().await;
        tracker.get_stats(connection_id)
    }

    /// Get statistics for all CLOB connections
    pub async fn get_all_clob_stats(&self) -> Vec<(String, LatencyStats)> {
        let tracker = self.health_tracker.read().await;
        self.clob_connections
            .iter()
            .map(|conn| (conn.id.clone(), tracker.get_stats(&conn.id)))
            .collect()
    }

    /// Get statistics for all RPC connections
    pub async fn get_all_rpc_stats(&self) -> Vec<(String, LatencyStats)> {
        let tracker = self.health_tracker.read().await;
        self.rpc_connections
            .iter()
            .map(|conn| (conn.id.clone(), tracker.get_stats(&conn.id)))
            .collect()
    }

    /// Execute a CLOB request with automatic latency tracking
    pub async fn execute_clob_request<F, Fut, T, E>(
        &self,
        request_fn: F,
    ) -> Result<T, E>
    where
        F: FnOnce(&ClobConnection) -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Debug,
    {
        let conn = self.get_clob_connection().await.expect("No CLOB connections available");
        let start = Instant::now();

        let result = request_fn(conn).await;
        let latency = start.elapsed();

        match &result {
            Ok(_) => {
                self.record_latency(&conn.id, latency).await;
            }
            Err(_) => {
                self.record_failure(&conn.id).await;
            }
        }

        result
    }

    /// Execute an RPC request with automatic latency tracking
    pub async fn execute_rpc_request<F, Fut, T, E>(
        &self,
        request_fn: F,
    ) -> Result<T, E>
    where
        F: FnOnce(&RpcConnection) -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Debug,
    {
        let conn = self.get_rpc_connection().await.expect("No RPC connections available");
        let start = Instant::now();

        let result = request_fn(conn).await;
        let latency = start.elapsed();

        match &result {
            Ok(_) => {
                self.record_latency(&conn.id, latency).await;
            }
            Err(_) => {
                self.record_failure(&conn.id).await;
            }
        }

        result
    }

    /// Run health checks on all connections
    pub async fn health_check(&self) {
        debug!("Running connection pool health checks");

        // Health check CLOB connections
        for conn in &self.clob_connections {
            let start = Instant::now();
            let result = conn
                .client
                .get(format!("{}/health", conn.base_url))
                .send()
                .await;

            let latency = start.elapsed();

            match result {
                Ok(response) if response.status().is_success() => {
                    self.record_latency(&conn.id, latency).await;
                    debug!(connection_id = conn.id, "CLOB health check passed");
                }
                _ => {
                    self.record_failure(&conn.id).await;
                    warn!(connection_id = conn.id, "CLOB health check failed");
                }
            }
        }

        // Health check RPC connections
        for conn in &self.rpc_connections {
            let start = Instant::now();
            let result = conn
                .client
                .post(&conn.endpoint_url)
                .json(&serde_json::json!({
                    "jsonrpc": "2.0",
                    "method": "eth_blockNumber",
                    "params": [],
                    "id": 1
                }))
                .send()
                .await;

            let latency = start.elapsed();

            match result {
                Ok(response) if response.status().is_success() => {
                    self.record_latency(&conn.id, latency).await;
                    debug!(connection_id = conn.id, "RPC health check passed");
                }
                _ => {
                    self.record_failure(&conn.id).await;
                    warn!(connection_id = conn.id, "RPC health check failed");
                }
            }
        }

        let mut tracker = self.health_tracker.write().await;
        tracker.update_check_time();
    }

    /// Start background health checking task
    pub fn start_health_checker(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let interval_secs = self.config.health_check_interval_secs;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
            loop {
                interval.tick().await;
                self.health_check().await;
            }
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &ConnectionPoolConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_pool_config_default() {
        let config = ConnectionPoolConfig::default();
        assert_eq!(config.clob_pool_size, 3);
        assert_eq!(config.rpc_pool_size, 2);
        assert_eq!(config.connection_timeout_ms, 5000);
        assert_eq!(config.request_timeout_ms, 2000);
        assert!(config.prefer_fastest);
    }

    #[test]
    fn test_health_tracker_record_latency() {
        let mut tracker = HealthTracker::new();
        tracker.record_latency("conn_1", Duration::from_micros(100));
        tracker.record_latency("conn_1", Duration::from_micros(200));
        tracker.record_latency("conn_1", Duration::from_micros(300));

        let stats = tracker.get_stats("conn_1");
        assert_eq!(stats.sample_count, 3);
        assert_eq!(stats.avg_us, 200);
        assert_eq!(stats.min_us, 100);
        assert_eq!(stats.max_us, 300);
    }

    #[test]
    fn test_health_tracker_failures() {
        let mut tracker = HealthTracker::new();
        assert!(tracker.is_healthy("conn_1"));

        tracker.record_failure("conn_1");
        tracker.record_failure("conn_1");
        assert!(tracker.is_healthy("conn_1"));

        tracker.record_failure("conn_1");
        assert!(!tracker.is_healthy("conn_1"));
    }

    #[test]
    fn test_health_tracker_reset_on_success() {
        let mut tracker = HealthTracker::new();
        tracker.record_failure("conn_1");
        tracker.record_failure("conn_1");
        tracker.record_latency("conn_1", Duration::from_micros(100));

        assert_eq!(tracker.failure_count("conn_1"), 0);
        assert!(tracker.is_healthy("conn_1"));
    }

    #[tokio::test]
    async fn test_connection_pool_creation() {
        let config = ConnectionPoolConfig::default();
        let pool = ConnectionPool::new(
            "https://clob.polymarket.com".to_string(),
            vec!["https://polygon-rpc.com".to_string()],
            config,
        );

        assert!(pool.get_clob_connection().await.is_some());
        assert!(pool.get_rpc_connection().await.is_some());
    }

    #[tokio::test]
    async fn test_fastest_connection_routing() {
        let config = ConnectionPoolConfig {
            clob_pool_size: 2,
            prefer_fastest: true,
            ..Default::default()
        };
        let pool = ConnectionPool::new(
            "https://clob.polymarket.com".to_string(),
            vec![],
            config,
        );

        // Record different latencies for connections
        pool.record_latency("clob_0", Duration::from_micros(500)).await;
        pool.record_latency("clob_1", Duration::from_micros(100)).await;

        // Should prefer the faster connection
        let conn = pool.get_clob_connection().await.unwrap();
        assert_eq!(conn.id, "clob_1");
    }
}
