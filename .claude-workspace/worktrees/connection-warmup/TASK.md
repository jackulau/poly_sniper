---
id: connection-warmup
name: Connection Warmup with Heartbeats
wave: 1
priority: 2
dependencies: []
estimated_hours: 4
tags: [performance, networking, websocket]
---

## Objective

Implement connection warmup and keep-alive mechanisms to maintain hot connections and reduce cold-start latency for WebSocket and HTTP connections.

## Context

Current connection handling in `crates/polysniper-data/src/ws_manager.rs`:
- Ping interval is 30 seconds (passive)
- No active connection warming
- 1-second fixed reconnect delay
- No connection pooling for HTTP clients
- Cold connections can add 100-500ms latency on first request

For trading systems, we need:
- Pre-warmed connections ready for instant use
- Proactive health checks before connections go stale
- Exponential backoff with jitter for reconnection
- HTTP connection pooling for API calls

## Implementation

### 1. Enhance WebSocket connection management

**File**: `crates/polysniper-data/src/ws_manager.rs`

```rust
/// Connection warmup configuration
#[derive(Clone)]
pub struct ConnectionConfig {
    /// Ping interval for keep-alive
    pub ping_interval: Duration,
    /// Pong timeout before considering connection dead
    pub pong_timeout: Duration,
    /// Initial reconnect delay
    pub initial_reconnect_delay: Duration,
    /// Maximum reconnect delay
    pub max_reconnect_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Add jitter to prevent thundering herd
    pub jitter_factor: f64,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            ping_interval: Duration::from_secs(15),      // More aggressive than 30s
            pong_timeout: Duration::from_secs(5),
            initial_reconnect_delay: Duration::from_millis(100),
            max_reconnect_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

impl WsManager {
    /// Active ping sender (not just interval tracking)
    async fn ping_loop(&self, write: &mut SplitSink<...>, config: &ConnectionConfig) {
        let mut interval = tokio::time::interval(config.ping_interval);
        let mut last_pong = Instant::now();

        loop {
            interval.tick().await;

            // Check if we received pong recently
            if last_pong.elapsed() > config.pong_timeout {
                warn!("Pong timeout, connection may be dead");
                self.connected.store(false, Ordering::SeqCst);
                break;
            }

            // Send ping
            if let Err(e) = write.send(Message::Ping(vec![])).await {
                warn!("Failed to send ping: {}", e);
                break;
            }
        }
    }

    /// Exponential backoff with jitter
    fn calculate_reconnect_delay(&self, attempt: u32, config: &ConnectionConfig) -> Duration {
        let base_delay = config.initial_reconnect_delay.as_secs_f64()
            * config.backoff_multiplier.powi(attempt as i32);
        let capped_delay = base_delay.min(config.max_reconnect_delay.as_secs_f64());

        // Add jitter
        let jitter = capped_delay * config.jitter_factor * rand::random::<f64>();
        Duration::from_secs_f64(capped_delay + jitter)
    }
}
```

### 2. Create HTTP connection pool

**File**: `crates/polysniper-data/src/http_pool.rs`

```rust
use reqwest::{Client, ClientBuilder};
use std::time::Duration;

/// Pooled HTTP client with connection warmup
pub struct HttpPool {
    client: Client,
    warmup_urls: Vec<String>,
}

impl HttpPool {
    pub fn new(config: HttpPoolConfig) -> Self {
        let client = ClientBuilder::new()
            .pool_max_idle_per_host(config.max_idle_per_host)
            .pool_idle_timeout(config.idle_timeout)
            .connect_timeout(config.connect_timeout)
            .tcp_keepalive(config.tcp_keepalive)
            .tcp_nodelay(true)  // Disable Nagle's algorithm
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            warmup_urls: config.warmup_urls,
        }
    }

    /// Warm up connections to known endpoints
    pub async fn warmup(&self) -> Result<(), HttpPoolError> {
        for url in &self.warmup_urls {
            // HEAD request to establish connection without body
            match self.client.head(url).send().await {
                Ok(_) => debug!("Warmed connection to {}", url),
                Err(e) => warn!("Failed to warm connection to {}: {}", url, e),
            }
        }
        Ok(())
    }

    /// Get the pooled client
    pub fn client(&self) -> &Client {
        &self.client
    }
}

#[derive(Clone)]
pub struct HttpPoolConfig {
    /// Maximum idle connections per host
    pub max_idle_per_host: usize,
    /// How long to keep idle connections
    pub idle_timeout: Duration,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// TCP keepalive interval
    pub tcp_keepalive: Duration,
    /// URLs to warm up on startup
    pub warmup_urls: Vec<String>,
}

impl Default for HttpPoolConfig {
    fn default() -> Self {
        Self {
            max_idle_per_host: 10,
            idle_timeout: Duration::from_secs(90),
            connect_timeout: Duration::from_secs(5),
            tcp_keepalive: Duration::from_secs(60),
            warmup_urls: vec![],
        }
    }
}
```

### 3. Add connection health monitor

**File**: `crates/polysniper-data/src/connection_health.rs`

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Tracks connection health metrics
pub struct ConnectionHealth {
    /// Last successful message timestamp
    last_message: AtomicU64,
    /// Last successful ping/pong roundtrip
    last_pong: AtomicU64,
    /// Consecutive failures
    failure_count: AtomicU32,
    /// RTT measurements for latency tracking
    rtt_samples: parking_lot::Mutex<VecDeque<Duration>>,
}

impl ConnectionHealth {
    pub fn new() -> Self;

    /// Record successful message receipt
    pub fn record_message(&self);

    /// Record successful pong with RTT
    pub fn record_pong(&self, rtt: Duration);

    /// Record connection failure
    pub fn record_failure(&self);

    /// Reset on reconnection
    pub fn reset(&self);

    /// Get health status
    pub fn status(&self) -> HealthStatus {
        let since_message = self.time_since_message();
        let since_pong = self.time_since_pong();
        let failures = self.failure_count.load(Ordering::Relaxed);

        if failures > 3 {
            HealthStatus::Unhealthy
        } else if since_pong > Duration::from_secs(60) {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }

    /// Get average RTT
    pub fn avg_rtt(&self) -> Option<Duration>;

    /// Get P99 RTT
    pub fn p99_rtt(&self) -> Option<Duration>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}
```

### 4. Update configuration

**File**: `config/default.toml`

```toml
[connection]
# WebSocket settings
ws_ping_interval_secs = 15
ws_pong_timeout_secs = 5
ws_initial_reconnect_delay_ms = 100
ws_max_reconnect_delay_secs = 30

# HTTP pool settings
http_max_idle_per_host = 10
http_idle_timeout_secs = 90
http_connect_timeout_secs = 5
http_tcp_keepalive_secs = 60

# Warmup endpoints (establish connections on startup)
warmup_urls = [
    "https://clob.polymarket.com/",
    "https://gamma-api.polymarket.com/"
]
```

### 5. Integrate with main application

**File**: `src/main.rs`

```rust
// During initialization
let http_pool = HttpPool::new(HttpPoolConfig {
    warmup_urls: vec![
        config.endpoints.clob_api.clone(),
        config.endpoints.gamma_api.clone(),
    ],
    ..Default::default()
});

// Warm up connections before starting trading
info!("Warming up HTTP connections...");
http_pool.warmup().await?;

// Pass pooled client to components that need it
let order_executor = OrderExecutor::new(http_pool.client().clone(), ...);
```

### 6. Add connection metrics

**File**: `crates/polysniper-observability/src/metrics.rs`

```rust
pub static CONNECTION_RTT: LazyLock<Histogram> = LazyLock::new(|| {
    register_histogram!(
        "connection_rtt_seconds",
        "WebSocket ping-pong roundtrip time",
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    ).unwrap()
});

pub static CONNECTION_HEALTH: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    register_int_gauge_vec!(
        "connection_health",
        "Connection health status (0=unhealthy, 1=degraded, 2=healthy)",
        &["connection"]
    ).unwrap()
});

pub static RECONNECT_BACKOFF: LazyLock<Histogram> = LazyLock::new(|| {
    register_histogram!(
        "reconnect_backoff_seconds",
        "Time waited before reconnection attempt"
    ).unwrap()
});
```

## Acceptance Criteria

- [ ] Active ping sending every 15 seconds
- [ ] Pong timeout detection (5 seconds)
- [ ] Exponential backoff with jitter for reconnection
- [ ] HTTP connection pool with configurable settings
- [ ] Connection warmup on startup
- [ ] ConnectionHealth tracker with RTT measurements
- [ ] Configuration via TOML file
- [ ] Metrics for RTT, health status, backoff times
- [ ] All existing tests pass
- [ ] Integration tests for reconnection scenarios

## Files to Create/Modify

- `crates/polysniper-data/src/ws_manager.rs` - Enhanced ping/pong, backoff logic
- `crates/polysniper-data/src/http_pool.rs` - **CREATE** - HTTP connection pooling
- `crates/polysniper-data/src/connection_health.rs` - **CREATE** - Health monitoring
- `crates/polysniper-data/src/lib.rs` - Export new modules
- `config/default.toml` - Connection configuration
- `src/main.rs` - Warmup integration
- `crates/polysniper-observability/src/metrics.rs` - Connection metrics

## Integration Points

- **Provides**: Reliable, low-latency connections for all network operations
- **Consumes**: Configuration, metrics infrastructure
- **Conflicts**: ws_manager.rs modifications (isolated to connection handling)

## Technical Notes

1. TCP_NODELAY disables Nagle's algorithm for lower latency
2. Jitter prevents thundering herd on mass reconnection
3. Keep HTTP connections warm to avoid TLS handshake overhead
4. Consider separate pools for different endpoints (different SLA)
5. Monitor P99 RTT to detect network degradation early
