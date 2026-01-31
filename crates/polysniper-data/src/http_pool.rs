//! HTTP connection pool with warmup for low-latency API calls

use reqwest::{Client, ClientBuilder};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors that can occur with the HTTP pool
#[derive(Error, Debug)]
pub enum HttpPoolError {
    #[error("Failed to build HTTP client: {0}")]
    BuildError(String),
    #[error("Warmup request failed for {url}: {error}")]
    WarmupError { url: String, error: String },
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),
}

/// HTTP connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpPoolConfig {
    /// Maximum idle connections per host
    #[serde(default = "default_max_idle_per_host")]
    pub max_idle_per_host: usize,
    /// How long to keep idle connections
    #[serde(default = "default_idle_timeout_secs")]
    pub idle_timeout_secs: u64,
    /// Connection timeout in seconds
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,
    /// TCP keepalive interval in seconds
    #[serde(default = "default_tcp_keepalive_secs")]
    pub tcp_keepalive_secs: u64,
    /// Request timeout in seconds
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    /// URLs to warm up on startup
    #[serde(default)]
    pub warmup_urls: Vec<String>,
}

fn default_max_idle_per_host() -> usize {
    10
}
fn default_idle_timeout_secs() -> u64 {
    90
}
fn default_connect_timeout_secs() -> u64 {
    5
}
fn default_tcp_keepalive_secs() -> u64 {
    60
}
fn default_request_timeout_secs() -> u64 {
    30
}

impl Default for HttpPoolConfig {
    fn default() -> Self {
        Self {
            max_idle_per_host: default_max_idle_per_host(),
            idle_timeout_secs: default_idle_timeout_secs(),
            connect_timeout_secs: default_connect_timeout_secs(),
            tcp_keepalive_secs: default_tcp_keepalive_secs(),
            request_timeout_secs: default_request_timeout_secs(),
            warmup_urls: vec![],
        }
    }
}

/// Pooled HTTP client with connection warmup
pub struct HttpPool {
    client: Client,
    warmup_urls: Vec<String>,
    config: HttpPoolConfig,
}

impl HttpPool {
    /// Create a new HTTP connection pool
    pub fn new(config: HttpPoolConfig) -> Result<Self, HttpPoolError> {
        let client = ClientBuilder::new()
            .pool_max_idle_per_host(config.max_idle_per_host)
            .pool_idle_timeout(Duration::from_secs(config.idle_timeout_secs))
            .connect_timeout(Duration::from_secs(config.connect_timeout_secs))
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .tcp_keepalive(Duration::from_secs(config.tcp_keepalive_secs))
            .tcp_nodelay(true) // Disable Nagle's algorithm for lower latency
            .build()
            .map_err(|e| HttpPoolError::BuildError(e.to_string()))?;

        let warmup_urls = config.warmup_urls.clone();

        info!(
            max_idle = config.max_idle_per_host,
            idle_timeout_secs = config.idle_timeout_secs,
            connect_timeout_secs = config.connect_timeout_secs,
            tcp_nodelay = true,
            warmup_count = warmup_urls.len(),
            "HTTP pool created"
        );

        Ok(Self {
            client,
            warmup_urls,
            config,
        })
    }

    /// Create with default configuration
    pub fn with_defaults() -> Result<Self, HttpPoolError> {
        Self::new(HttpPoolConfig::default())
    }

    /// Create with warmup URLs
    pub fn with_warmup_urls(urls: Vec<String>) -> Result<Self, HttpPoolError> {
        Self::new(HttpPoolConfig {
            warmup_urls: urls,
            ..Default::default()
        })
    }

    /// Warm up connections to known endpoints
    ///
    /// This establishes TCP connections and completes TLS handshakes
    /// so subsequent requests have minimal latency.
    pub async fn warmup(&self) -> Result<WarmupResult, HttpPoolError> {
        let mut result = WarmupResult::default();

        if self.warmup_urls.is_empty() {
            debug!("No warmup URLs configured");
            return Ok(result);
        }

        info!(urls = self.warmup_urls.len(), "Warming up HTTP connections");

        for url in &self.warmup_urls {
            let start = std::time::Instant::now();

            // HEAD request to establish connection without transferring body
            match self.client.head(url).send().await {
                Ok(response) => {
                    let elapsed = start.elapsed();
                    let status = response.status();

                    if status.is_success() || status.is_redirection() {
                        debug!(
                            url = url,
                            status = %status,
                            elapsed_ms = elapsed.as_millis() as u64,
                            "Warmed connection"
                        );
                        result.successful += 1;
                        result.total_latency_ms += elapsed.as_millis() as u64;
                    } else {
                        warn!(
                            url = url,
                            status = %status,
                            "Warmup request returned non-success status"
                        );
                        result.failed += 1;
                    }
                }
                Err(e) => {
                    warn!(
                        url = url,
                        error = %e,
                        "Failed to warm connection"
                    );
                    result.failed += 1;
                }
            }
        }

        info!(
            successful = result.successful,
            failed = result.failed,
            avg_latency_ms = if result.successful > 0 {
                result.total_latency_ms / result.successful as u64
            } else {
                0
            },
            "Connection warmup complete"
        );

        Ok(result)
    }

    /// Get the pooled client for making requests
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Get a cloned client (for moving into async tasks)
    pub fn client_clone(&self) -> Client {
        self.client.clone()
    }

    /// Get the pool configuration
    pub fn config(&self) -> &HttpPoolConfig {
        &self.config
    }

    /// Check if the pool has warmup URLs configured
    pub fn has_warmup_urls(&self) -> bool {
        !self.warmup_urls.is_empty()
    }
}

/// Result of a warmup operation
#[derive(Debug, Default, Clone)]
pub struct WarmupResult {
    /// Number of successful warmup connections
    pub successful: u32,
    /// Number of failed warmup connections
    pub failed: u32,
    /// Total latency of successful warmups in milliseconds
    pub total_latency_ms: u64,
}

impl WarmupResult {
    /// Check if all warmup connections succeeded
    pub fn all_successful(&self) -> bool {
        self.failed == 0 && self.successful > 0
    }

    /// Get average warmup latency in milliseconds
    pub fn avg_latency_ms(&self) -> Option<u64> {
        if self.successful > 0 {
            Some(self.total_latency_ms / self.successful as u64)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HttpPoolConfig::default();
        assert_eq!(config.max_idle_per_host, 10);
        assert_eq!(config.idle_timeout_secs, 90);
        assert_eq!(config.connect_timeout_secs, 5);
        assert_eq!(config.tcp_keepalive_secs, 60);
        assert!(config.warmup_urls.is_empty());
    }

    #[tokio::test]
    async fn test_pool_creation() {
        let pool = HttpPool::with_defaults().unwrap();
        assert!(!pool.has_warmup_urls());
    }

    #[tokio::test]
    async fn test_pool_with_warmup_urls() {
        let urls = vec!["https://example.com".to_string()];
        let pool = HttpPool::with_warmup_urls(urls).unwrap();
        assert!(pool.has_warmup_urls());
    }

    #[test]
    fn test_warmup_result() {
        let result = WarmupResult {
            successful: 3,
            failed: 1,
            total_latency_ms: 300,
        };
        assert!(!result.all_successful());
        assert_eq!(result.avg_latency_ms(), Some(100));

        let all_success = WarmupResult {
            successful: 2,
            failed: 0,
            total_latency_ms: 200,
        };
        assert!(all_success.all_successful());
    }
}
