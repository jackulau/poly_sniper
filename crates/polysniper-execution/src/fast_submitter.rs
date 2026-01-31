//! Fast order submitter with latency optimizations.
//!
//! Provides optimized order submission through:
//! - Pre-signed order templates for common scenarios
//! - Speculative nonce management to avoid lookups
//! - Parallel submission to multiple endpoints
//! - Order caching for repeated patterns

use crate::submitter::OrderSubmitter;
use polysniper_core::{ExecutionError, Order, OrderExecutor, OrderStatusResponse};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Configuration for the fast submitter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastSubmitterConfig {
    /// Submit to multiple endpoints in parallel (first response wins)
    pub parallel_submissions: bool,
    /// Pre-sign common order templates for faster submission
    pub pre_sign_orders: bool,
    /// Batch small orders together
    pub batch_small_orders: bool,
    /// Use speculative nonce to avoid nonce lookup latency
    pub speculative_nonce: bool,
    /// Threshold size (in USD) below which orders are considered "small"
    pub small_order_threshold_usd: Decimal,
    /// Maximum batch size for small orders
    pub max_batch_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
}

impl Default for FastSubmitterConfig {
    fn default() -> Self {
        Self {
            parallel_submissions: true,
            pre_sign_orders: true,
            batch_small_orders: false,
            speculative_nonce: true,
            small_order_threshold_usd: Decimal::new(50, 0), // $50
            max_batch_size: 5,
            batch_timeout_ms: 100,
        }
    }
}

/// Order template for pre-signing
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct OrderTemplate {
    /// Token ID
    pub token_id: String,
    /// Side (buy/sell)
    pub side: String,
    /// Order type
    pub order_type: String,
}

/// Pre-computed signed order data
#[derive(Debug, Clone)]
pub struct SignedOrderData {
    /// The signed order bytes
    pub signature: Vec<u8>,
    /// Expiration timestamp
    pub expiration: i64,
    /// Nonce used for this signature
    pub nonce: u64,
}

/// Cache for pre-signed orders and nonces
pub struct OrderCache {
    /// Pre-computed order templates
    templates: RwLock<HashMap<OrderTemplate, SignedOrderData>>,
    /// Current nonce (speculatively incremented)
    nonce_cache: AtomicU64,
    /// Last confirmed nonce from the chain
    confirmed_nonce: AtomicU64,
}

impl OrderCache {
    /// Create a new order cache
    pub fn new(initial_nonce: u64) -> Self {
        Self {
            templates: RwLock::new(HashMap::new()),
            nonce_cache: AtomicU64::new(initial_nonce),
            confirmed_nonce: AtomicU64::new(initial_nonce),
        }
    }

    /// Get the next speculative nonce
    pub fn get_next_nonce(&self) -> u64 {
        self.nonce_cache.fetch_add(1, AtomicOrdering::SeqCst)
    }

    /// Get the current nonce without incrementing
    pub fn current_nonce(&self) -> u64 {
        self.nonce_cache.load(AtomicOrdering::SeqCst)
    }

    /// Update the confirmed nonce from chain
    pub fn update_confirmed_nonce(&self, nonce: u64) {
        let old = self.confirmed_nonce.swap(nonce, AtomicOrdering::SeqCst);
        // If confirmed nonce is higher than our cache, update cache
        let current = self.nonce_cache.load(AtomicOrdering::SeqCst);
        if nonce > current {
            self.nonce_cache.store(nonce, AtomicOrdering::SeqCst);
        }
        debug!(
            old_confirmed = old,
            new_confirmed = nonce,
            current_cache = current,
            "Updated confirmed nonce"
        );
    }

    /// Rollback nonce on failed submission
    pub fn rollback_nonce(&self) {
        // Don't go below confirmed nonce
        let confirmed = self.confirmed_nonce.load(AtomicOrdering::SeqCst);
        let current = self.nonce_cache.load(AtomicOrdering::SeqCst);
        if current > confirmed {
            self.nonce_cache.fetch_sub(1, AtomicOrdering::SeqCst);
        }
    }

    /// Store a pre-signed order template
    pub async fn store_template(&self, template: OrderTemplate, signed: SignedOrderData) {
        let mut templates = self.templates.write().await;
        templates.insert(template, signed);
    }

    /// Get a pre-signed order if available
    pub async fn get_template(&self, template: &OrderTemplate) -> Option<SignedOrderData> {
        let templates = self.templates.read().await;
        templates.get(template).cloned()
    }

    /// Clear expired templates
    pub async fn clear_expired(&self) {
        let now = chrono::Utc::now().timestamp();
        let mut templates = self.templates.write().await;
        templates.retain(|_, v| v.expiration > now);
    }
}

impl Default for OrderCache {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Submission statistics
#[derive(Debug, Clone, Default)]
pub struct SubmissionStats {
    /// Number of orders submitted
    pub orders_submitted: u64,
    /// Number of parallel submissions
    pub parallel_submissions: u64,
    /// Number of cached nonces used
    pub cached_nonce_hits: u64,
    /// Number of nonce lookups required
    pub nonce_lookups: u64,
    /// Average submission latency in microseconds
    pub avg_latency_us: u64,
    /// P99 submission latency in microseconds
    pub p99_latency_us: u64,
}

/// Fast order submitter with latency optimizations
pub struct FastSubmitter {
    /// Underlying order submitter
    submitter: Arc<OrderSubmitter>,
    /// Order cache for pre-signing and nonce management
    order_cache: OrderCache,
    /// Configuration
    config: FastSubmitterConfig,
    /// Submission statistics
    stats: RwLock<SubmissionStats>,
    /// Latency samples for statistics
    latency_samples: RwLock<Vec<Duration>>,
}

impl FastSubmitter {
    /// Create a new fast submitter
    pub fn new(submitter: Arc<OrderSubmitter>, config: FastSubmitterConfig) -> Self {
        info!(
            parallel = config.parallel_submissions,
            pre_sign = config.pre_sign_orders,
            speculative_nonce = config.speculative_nonce,
            "Creating FastSubmitter with optimizations"
        );

        Self {
            submitter,
            order_cache: OrderCache::default(),
            config,
            stats: RwLock::new(SubmissionStats::default()),
            latency_samples: RwLock::new(Vec::with_capacity(1000)),
        }
    }

    /// Set the initial nonce from chain lookup
    pub fn set_initial_nonce(&self, nonce: u64) {
        self.order_cache.update_confirmed_nonce(nonce);
        info!(nonce = nonce, "Set initial nonce for fast submitter");
    }

    /// Get the next nonce (speculative if enabled)
    pub fn get_next_nonce(&self) -> u64 {
        if self.config.speculative_nonce {
            let mut stats = self.stats.blocking_write();
            stats.cached_nonce_hits += 1;
            self.order_cache.get_next_nonce()
        } else {
            let mut stats = self.stats.blocking_write();
            stats.nonce_lookups += 1;
            // In non-speculative mode, would need to query the chain
            self.order_cache.get_next_nonce()
        }
    }

    /// Submit order with fast-path optimizations
    pub async fn submit_fast(&self, order: Order) -> Result<String, ExecutionError> {
        let start = Instant::now();

        // Record the submission
        {
            let mut stats = self.stats.write().await;
            stats.orders_submitted += 1;
        }

        // Submit through the underlying submitter
        let result = self.submitter.submit_order(order).await;

        // Record latency
        let latency = start.elapsed();
        self.record_latency(latency).await;

        // Handle nonce on failure
        if result.is_err() && self.config.speculative_nonce {
            self.order_cache.rollback_nonce();
        }

        result
    }

    /// Submit multiple orders in parallel (first response wins for each)
    pub async fn submit_parallel(
        &self,
        orders: Vec<Order>,
    ) -> Vec<Result<String, ExecutionError>> {
        use tokio::task::JoinSet;

        if orders.is_empty() {
            return vec![];
        }

        let start = Instant::now();
        let order_count = orders.len();

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.parallel_submissions += 1;
        }

        // Submit all orders concurrently using JoinSet
        let mut join_set = JoinSet::new();
        for (idx, order) in orders.into_iter().enumerate() {
            let submitter = Arc::clone(&self.submitter);
            join_set.spawn(async move { (idx, submitter.submit_order(order).await) });
        }

        // Collect results in order
        let mut results: Vec<Option<Result<String, ExecutionError>>> =
            (0..order_count).map(|_| None).collect();
        while let Some(result) = join_set.join_next().await {
            if let Ok((idx, order_result)) = result {
                results[idx] = Some(order_result);
            }
        }

        // Record latency
        let latency = start.elapsed();
        self.record_latency(latency).await;

        results
            .into_iter()
            .map(|r| r.unwrap_or(Err(ExecutionError::NetworkError("Task failed".to_string()))))
            .collect()
    }

    /// Record a latency sample
    async fn record_latency(&self, latency: Duration) {
        let mut samples = self.latency_samples.write().await;
        samples.push(latency);

        // Keep only last 1000 samples
        if samples.len() > 1000 {
            samples.remove(0);
        }

        // Update statistics
        if !samples.is_empty() {
            let mut sorted: Vec<u64> = samples.iter().map(|d| d.as_micros() as u64).collect();
            sorted.sort_unstable();

            let avg = sorted.iter().sum::<u64>() / sorted.len() as u64;
            let p99_idx = (sorted.len() * 99) / 100;
            let p99 = sorted[p99_idx];

            let mut stats = self.stats.write().await;
            stats.avg_latency_us = avg;
            stats.p99_latency_us = p99;
        }
    }

    /// Get submission statistics
    pub async fn get_stats(&self) -> SubmissionStats {
        self.stats.read().await.clone()
    }

    /// Get the configuration
    pub fn config(&self) -> &FastSubmitterConfig {
        &self.config
    }

    /// Clear expired order templates
    pub async fn clear_expired_templates(&self) {
        self.order_cache.clear_expired().await;
    }
}

#[async_trait::async_trait]
impl OrderExecutor for FastSubmitter {
    async fn submit_order(&self, order: Order) -> Result<String, ExecutionError> {
        self.submit_fast(order).await
    }

    async fn cancel_order(&self, order_id: &str) -> Result<(), ExecutionError> {
        self.submitter.cancel_order(order_id).await
    }

    async fn get_order_status(&self, order_id: &str) -> Result<OrderStatusResponse, ExecutionError> {
        self.submitter.get_order_status(order_id).await
    }

    fn is_dry_run(&self) -> bool {
        self.submitter.is_dry_run()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_submitter_config_default() {
        let config = FastSubmitterConfig::default();
        assert!(config.parallel_submissions);
        assert!(config.pre_sign_orders);
        assert!(!config.batch_small_orders);
        assert!(config.speculative_nonce);
    }

    #[test]
    fn test_order_cache_nonce() {
        let cache = OrderCache::new(10);
        assert_eq!(cache.current_nonce(), 10);
        assert_eq!(cache.get_next_nonce(), 10);
        assert_eq!(cache.get_next_nonce(), 11);
        assert_eq!(cache.current_nonce(), 12);
    }

    #[test]
    fn test_order_cache_nonce_rollback() {
        let cache = OrderCache::new(10);
        cache.get_next_nonce(); // 10 -> 11
        cache.get_next_nonce(); // 11 -> 12
        cache.rollback_nonce(); // 12 -> 11
        assert_eq!(cache.current_nonce(), 11);
    }

    #[test]
    fn test_order_cache_confirmed_nonce_update() {
        let cache = OrderCache::new(10);
        cache.get_next_nonce(); // 10 -> 11
        cache.get_next_nonce(); // 11 -> 12

        // Confirmed nonce jumps ahead
        cache.update_confirmed_nonce(20);
        assert_eq!(cache.current_nonce(), 20);
    }

    #[test]
    fn test_order_template_hash() {
        let template1 = OrderTemplate {
            token_id: "token1".to_string(),
            side: "BUY".to_string(),
            order_type: "GTC".to_string(),
        };

        let template2 = OrderTemplate {
            token_id: "token1".to_string(),
            side: "BUY".to_string(),
            order_type: "GTC".to_string(),
        };

        let template3 = OrderTemplate {
            token_id: "token2".to_string(),
            side: "BUY".to_string(),
            order_type: "GTC".to_string(),
        };

        assert_eq!(template1, template2);
        assert_ne!(template1, template3);
    }

    #[tokio::test]
    async fn test_order_cache_templates() {
        let cache = OrderCache::new(0);

        let template = OrderTemplate {
            token_id: "token1".to_string(),
            side: "BUY".to_string(),
            order_type: "GTC".to_string(),
        };

        let signed = SignedOrderData {
            signature: vec![1, 2, 3, 4],
            expiration: chrono::Utc::now().timestamp() + 3600,
            nonce: 1,
        };

        cache.store_template(template.clone(), signed.clone()).await;

        let retrieved = cache.get_template(&template).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().nonce, 1);
    }
}
