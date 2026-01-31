---
id: latency-optimization
name: Latency Optimization for Better Fills
wave: 1
priority: 1
dependencies: []
estimated_hours: 5
tags: [execution, performance, latency, fills]
---

## Objective

Optimize execution latency across the system to achieve better fills through faster order submission, reduced processing time, and smarter connection management.

## Context

Lower latency = better fills in competitive markets. The codebase already has:
- WebSocket manager with reconnection logic
- Order submission via REST API
- TWAP/VWAP execution algorithms
- Gas optimizer for Polygon transactions
- Queue estimator for fill rate prediction

This task focuses on reducing end-to-end latency from signal generation to order confirmation.

## Implementation

### 1. Connection Pool Optimization

**File:** `crates/polysniper-data/src/connection_pool.rs`

```rust
pub struct ConnectionPool {
    clob_connections: Vec<ClobConnection>,
    rpc_connections: Vec<RpcConnection>,
    config: ConnectionPoolConfig,
    health_tracker: Arc<HealthTracker>,
}

pub struct ConnectionPoolConfig {
    pub clob_pool_size: usize,              // Number of CLOB REST connections
    pub rpc_pool_size: usize,               // Number of Polygon RPC connections
    pub connection_timeout_ms: u64,         // Connection establishment timeout
    pub request_timeout_ms: u64,            // Per-request timeout
    pub health_check_interval_secs: u64,    // Connection health monitoring
    pub prefer_fastest: bool,               // Route to fastest connection
}

pub struct HealthTracker {
    latencies: HashMap<String, VecDeque<Duration>>,
    failures: HashMap<String, u32>,
    last_check: DateTime<Utc>,
}
```

- Maintain multiple warm connections to CLOB API
- Route requests to fastest responding endpoint
- Automatic failover on connection issues
- Track latency metrics per connection

### 2. WebSocket Optimization

**Modify:** `crates/polysniper-data/src/ws_manager.rs`

Add optimizations:
```rust
pub struct WsManagerConfig {
    // Existing fields...
    pub enable_compression: bool,           // WebSocket compression
    pub tcp_nodelay: bool,                  // Disable Nagle's algorithm
    pub buffer_size: usize,                 // Receive buffer size
    pub priority_subscriptions: Vec<String>, // High-priority token subscriptions
}
```

- Enable TCP_NODELAY to reduce latency
- Implement message batching for subscriptions
- Priority queue for high-value market updates
- Pre-parse message formats for faster processing

### 3. Order Submission Pipeline

**File:** `crates/polysniper-execution/src/fast_submitter.rs`

```rust
pub struct FastSubmitter {
    connection_pool: Arc<ConnectionPool>,
    order_cache: OrderCache,
    config: FastSubmitterConfig,
}

pub struct FastSubmitterConfig {
    pub parallel_submissions: bool,         // Submit to multiple endpoints
    pub pre_sign_orders: bool,              // Pre-sign common order templates
    pub batch_small_orders: bool,           // Batch small orders together
    pub speculative_nonce: bool,            // Pre-increment nonce speculatively
}

pub struct OrderCache {
    // Pre-computed order templates for common scenarios
    templates: HashMap<OrderTemplate, SignedOrder>,
    nonce_cache: AtomicU64,
}
```

- Pre-sign order templates for common scenarios
- Speculative nonce management to avoid lookups
- Parallel submission to multiple endpoints (first response wins)
- Order batching for efficiency

### 4. Signal Processing Optimization

**Modify:** `crates/polysniper-strategies/src/lib.rs`

Add fast-path processing:
```rust
pub struct StrategyProcessor {
    // Existing fields...
    fast_path_strategies: Vec<Arc<dyn FastStrategy>>,
    slow_path_strategies: Vec<Arc<dyn Strategy>>,
}

pub trait FastStrategy: Strategy {
    // Synchronous fast-path processing
    fn process_event_sync(&self, event: &SystemEvent) -> Option<TradeSignal>;
}
```

- Implement fast-path for time-critical strategies (arbitrage, price spike)
- Avoid async overhead for simple signal generation
- Priority queue for signal processing

### 5. Latency Metrics and Monitoring

**File:** `crates/polysniper-observability/src/latency_metrics.rs`

```rust
pub struct LatencyMetrics {
    event_to_signal_us: Histogram,          // Event receipt to signal generation
    signal_to_submit_us: Histogram,         // Signal to order submission
    submit_to_confirm_us: Histogram,        // Submission to confirmation
    end_to_end_us: Histogram,               // Total latency
    ws_message_parse_us: Histogram,         // WebSocket message parsing
}

impl LatencyMetrics {
    pub fn record_event_to_signal(&self, duration: Duration);
    pub fn record_signal_to_submit(&self, duration: Duration);
    pub fn record_submit_to_confirm(&self, duration: Duration);
    pub fn record_end_to_end(&self, duration: Duration);
    pub fn get_p50(&self) -> LatencyStats;
    pub fn get_p99(&self) -> LatencyStats;
}
```

### 6. Configuration

**Modify:** `config/default.toml`

```toml
[latency]
enabled = true

[latency.connection_pool]
clob_pool_size = 3
rpc_pool_size = 2
connection_timeout_ms = 5000
request_timeout_ms = 2000
health_check_interval_secs = 30
prefer_fastest = true

[latency.websocket]
enable_compression = false      # Compression adds CPU overhead
tcp_nodelay = true              # Disable Nagle's algorithm
buffer_size = 65536
priority_subscriptions = []

[latency.execution]
parallel_submissions = true
pre_sign_orders = true
batch_small_orders = false
speculative_nonce = true
```

## Acceptance Criteria

- [ ] Connection pool maintains warm connections to CLOB and RPC
- [ ] Latency-based routing sends requests to fastest endpoint
- [ ] TCP_NODELAY enabled on WebSocket connections
- [ ] Order pre-signing reduces submission latency
- [ ] Latency metrics tracked at each pipeline stage
- [ ] P50 and P99 latencies visible in metrics/dashboard
- [ ] Fast-path processing for time-critical strategies
- [ ] All new code has unit tests
- [ ] No breaking changes to existing execution flow
- [ ] Benchmark tests demonstrate latency improvements

## Files to Create/Modify

**Create:**
- `crates/polysniper-data/src/connection_pool.rs` - Connection pooling
- `crates/polysniper-execution/src/fast_submitter.rs` - Optimized submission
- `crates/polysniper-observability/src/latency_metrics.rs` - Latency tracking

**Modify:**
- `crates/polysniper-data/src/ws_manager.rs` - WebSocket optimizations
- `crates/polysniper-data/src/lib.rs` - Export connection_pool
- `crates/polysniper-execution/src/lib.rs` - Export fast_submitter
- `crates/polysniper-observability/src/lib.rs` - Export latency_metrics
- `config/default.toml` - Add latency configuration section

## Integration Points

- **Provides**: Faster order execution, latency metrics, connection management
- **Consumes**: Existing submitter interface, WebSocket events
- **Conflicts**: Modifies ws_manager.rs - ensure coordination with other tasks

## Testing Requirements

```rust
#[cfg(test)]
mod tests {
    // Test connection pool routing
    #[tokio::test]
    async fn test_fastest_connection_routing() { ... }

    // Test health tracking
    #[test]
    fn test_connection_health_tracking() { ... }

    // Test pre-signed order caching
    #[test]
    fn test_order_template_caching() { ... }

    // Test latency metrics recording
    #[test]
    fn test_latency_histogram() { ... }
}
```

## Benchmarks

Add criterion benchmarks:
```rust
// benches/latency_benchmarks.rs
fn benchmark_order_submission(c: &mut Criterion) {
    c.bench_function("order_submission_baseline", |b| { ... });
    c.bench_function("order_submission_optimized", |b| { ... });
}

fn benchmark_ws_message_parsing(c: &mut Criterion) {
    c.bench_function("parse_orderbook_update", |b| { ... });
}
```
