//! Twitter API v2 client for feed ingestion
//!
//! Polls Twitter's search API for tweets matching configured keywords.
//! Implements rate limiting compliance (15 requests per 15 minutes).

use crate::feed_types::{FeedError, FeedItem, FeedSource, TwitterConfig, TwitterQueryConfig};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

const TWITTER_API_BASE: &str = "https://api.twitter.com/2";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

// Rate limit: 15 requests per 15 minutes for search endpoint
const RATE_LIMIT_REQUESTS: u32 = 15;
const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(15 * 60);

/// Twitter API v2 response for search
#[derive(Debug, Deserialize)]
struct TwitterSearchResponse {
    #[serde(default)]
    data: Option<Vec<TweetData>>,
    #[serde(default)]
    includes: Option<TwitterIncludes>,
    #[serde(default)]
    meta: Option<TwitterMeta>,
}

#[derive(Debug, Deserialize)]
struct TweetData {
    id: String,
    text: String,
    author_id: Option<String>,
    created_at: Option<String>,
    #[serde(default)]
    public_metrics: Option<TweetMetrics>,
}

#[derive(Debug, Deserialize)]
struct TweetMetrics {
    #[serde(default)]
    like_count: u64,
    #[serde(default)]
    retweet_count: u64,
    #[serde(default)]
    reply_count: u64,
}

#[derive(Debug, Deserialize)]
struct TwitterIncludes {
    #[serde(default)]
    users: Option<Vec<TwitterUser>>,
}

#[derive(Debug, Deserialize)]
struct TwitterUser {
    id: String,
    username: String,
    #[allow(dead_code)]
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TwitterMeta {
    #[allow(dead_code)]
    #[serde(default)]
    result_count: u32,
    #[serde(default)]
    newest_id: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    oldest_id: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    next_token: Option<String>,
}

/// Rate limiter for Twitter API
struct RateLimiter {
    requests: AtomicU64,
    window_start: RwLock<Instant>,
}

impl RateLimiter {
    fn new() -> Self {
        Self {
            requests: AtomicU64::new(0),
            window_start: RwLock::new(Instant::now()),
        }
    }

    async fn check_and_increment(&self) -> Result<(), FeedError> {
        let mut window_start = self.window_start.write().await;
        let now = Instant::now();

        // Reset window if expired
        if now.duration_since(*window_start) >= RATE_LIMIT_WINDOW {
            *window_start = now;
            self.requests.store(0, Ordering::SeqCst);
        }

        let current = self.requests.fetch_add(1, Ordering::SeqCst);
        if current >= RATE_LIMIT_REQUESTS as u64 {
            let remaining = RATE_LIMIT_WINDOW - now.duration_since(*window_start);
            warn!(
                "Twitter rate limit reached, {} requests in window",
                current
            );
            return Err(FeedError::RateLimited {
                retry_after: Some(remaining),
            });
        }

        Ok(())
    }

    async fn time_until_reset(&self) -> Duration {
        let window_start = self.window_start.read().await;
        let elapsed = Instant::now().duration_since(*window_start);
        if elapsed >= RATE_LIMIT_WINDOW {
            Duration::ZERO
        } else {
            RATE_LIMIT_WINDOW - elapsed
        }
    }
}

/// Twitter API v2 client
pub struct TwitterClient {
    client: Client,
    bearer_token: String,
    config: TwitterConfig,
    rate_limiter: Arc<RateLimiter>,
    /// Track the newest tweet ID seen for each query to avoid duplicates
    since_ids: RwLock<HashMap<String, String>>,
}

impl TwitterClient {
    /// Create a new Twitter client
    pub fn new(config: TwitterConfig) -> Result<Self, FeedError> {
        let bearer_token = std::env::var(&config.bearer_token_env).map_err(|_| {
            FeedError::NotConfigured(format!(
                "Twitter bearer token env var '{}' not set",
                config.bearer_token_env
            ))
        })?;

        let client = Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .map_err(|e| FeedError::NetworkError(e.to_string()))?;

        Ok(Self {
            client,
            bearer_token,
            config,
            rate_limiter: Arc::new(RateLimiter::new()),
            since_ids: RwLock::new(HashMap::new()),
        })
    }

    /// Create a client with a provided bearer token (for testing)
    pub fn with_token(config: TwitterConfig, bearer_token: String) -> Result<Self, FeedError> {
        let client = Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .map_err(|e| FeedError::NetworkError(e.to_string()))?;

        Ok(Self {
            client,
            bearer_token,
            config,
            rate_limiter: Arc::new(RateLimiter::new()),
            since_ids: RwLock::new(HashMap::new()),
        })
    }

    /// Poll all configured queries and return new tweets
    pub async fn poll(&self) -> Result<Vec<FeedItem>, FeedError> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        let mut all_items = Vec::new();

        for query_config in &self.config.queries {
            match self.search_tweets(query_config).await {
                Ok(items) => {
                    info!(
                        "Fetched {} tweets for query '{}'",
                        items.len(),
                        query_config.query
                    );
                    all_items.extend(items);
                }
                Err(FeedError::RateLimited { retry_after }) => {
                    warn!(
                        "Rate limited on Twitter API, retry after {:?}",
                        retry_after
                    );
                    // Stop processing more queries when rate limited
                    return Err(FeedError::RateLimited { retry_after });
                }
                Err(e) => {
                    warn!("Failed to fetch tweets for query '{}': {}", query_config.query, e);
                    // Continue with other queries
                }
            }
        }

        Ok(all_items)
    }

    /// Search for tweets matching a query
    async fn search_tweets(&self, query_config: &TwitterQueryConfig) -> Result<Vec<FeedItem>, FeedError> {
        // Check rate limit
        self.rate_limiter.check_and_increment().await?;

        // Build the search query
        let mut query = query_config.query.clone();
        if let Some(ref account) = query_config.from_account {
            query = format!("from:{} {}", account, query);
        }
        if !query_config.include_retweets {
            query = format!("{} -is:retweet", query);
        }

        // Get since_id for this query to avoid duplicates
        let since_id = {
            let since_ids = self.since_ids.read().await;
            since_ids.get(&query_config.query).cloned()
        };

        let mut url = format!(
            "{}/tweets/search/recent?query={}&max_results={}&tweet.fields=created_at,author_id,public_metrics&expansions=author_id&user.fields=username,name",
            TWITTER_API_BASE,
            urlencoding::encode(&query),
            self.config.max_results.min(100) // API max is 100
        );

        if let Some(ref id) = since_id {
            url.push_str(&format!("&since_id={}", id));
        }

        debug!("Searching Twitter: {}", url);

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.bearer_token)
            .send()
            .await
            .map_err(|e| FeedError::NetworkError(e.to_string()))?;

        let status = response.status();
        if status.as_u16() == 429 {
            let retry_after = response
                .headers()
                .get("x-rate-limit-reset")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<i64>().ok())
                .map(|ts| {
                    let now = Utc::now().timestamp();
                    Duration::from_secs((ts - now).max(0) as u64)
                });
            return Err(FeedError::RateLimited { retry_after });
        }

        if status.as_u16() == 401 {
            return Err(FeedError::AuthError("Invalid Twitter bearer token".to_string()));
        }

        if !status.is_success() {
            return Err(FeedError::FetchError(format!(
                "Twitter API returned status {}",
                status
            )));
        }

        let twitter_response: TwitterSearchResponse = response
            .json()
            .await
            .map_err(|e| FeedError::ParseError(e.to_string()))?;

        // Build user lookup map
        let user_map: HashMap<String, &TwitterUser> = twitter_response
            .includes
            .as_ref()
            .and_then(|i| i.users.as_ref())
            .map(|users| users.iter().map(|u| (u.id.clone(), u)).collect())
            .unwrap_or_default();

        // Update since_id for next poll
        if let Some(ref meta) = twitter_response.meta {
            if let Some(ref newest_id) = meta.newest_id {
                let mut since_ids = self.since_ids.write().await;
                since_ids.insert(query_config.query.clone(), newest_id.clone());
            }
        }

        // Convert tweets to FeedItems
        let items: Vec<FeedItem> = twitter_response
            .data
            .unwrap_or_default()
            .into_iter()
            .filter_map(|tweet| self.convert_tweet(tweet, &user_map, query_config))
            .collect();

        Ok(items)
    }

    fn convert_tweet(
        &self,
        tweet: TweetData,
        user_map: &HashMap<String, &TwitterUser>,
        query_config: &TwitterQueryConfig,
    ) -> Option<FeedItem> {
        let author = tweet
            .author_id
            .as_ref()
            .and_then(|id| user_map.get(id))
            .map(|u| u.username.clone());

        let published_at = tweet
            .created_at
            .as_ref()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        let url = author.as_ref().map(|username| {
            format!("https://twitter.com/{}/status/{}", username, tweet.id)
        });

        let mut item = FeedItem::new(
            tweet.id.clone(),
            FeedSource::Twitter {
                account: author.clone(),
                query: Some(query_config.query.clone()),
            },
            tweet.text,
            None,
            author,
            url,
            published_at,
        );

        // Add metrics as metadata
        if let Some(metrics) = tweet.public_metrics {
            item = item
                .with_metadata("likes", serde_json::json!(metrics.like_count))
                .with_metadata("retweets", serde_json::json!(metrics.retweet_count))
                .with_metadata("replies", serde_json::json!(metrics.reply_count));
        }

        Some(item)
    }

    /// Get the polling interval from config
    pub fn poll_interval(&self) -> Duration {
        Duration::from_secs(self.config.poll_interval_secs)
    }

    /// Check if the client is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get time until rate limit resets
    pub async fn time_until_reset(&self) -> Duration {
        self.rate_limiter.time_until_reset().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let limiter = RateLimiter::new();

            // Should allow first requests
            for _ in 0..RATE_LIMIT_REQUESTS {
                assert!(limiter.check_and_increment().await.is_ok());
            }

            // Should reject after limit
            let result = limiter.check_and_increment().await;
            assert!(matches!(result, Err(FeedError::RateLimited { .. })));
        });
    }

    #[test]
    fn test_feed_item_creation() {
        let item = FeedItem::new(
            "12345".to_string(),
            FeedSource::Twitter {
                account: Some("testuser".to_string()),
                query: Some("polymarket".to_string()),
            },
            "This is a test tweet about polymarket".to_string(),
            None,
            Some("testuser".to_string()),
            Some("https://twitter.com/testuser/status/12345".to_string()),
            Utc::now(),
        );

        assert_eq!(item.id, "12345");
        assert!(item.content.contains("polymarket"));
        assert!(!item.content_hash.is_empty());
    }

    #[test]
    fn test_query_building() {
        let query_config = TwitterQueryConfig {
            query: "polymarket".to_string(),
            from_account: Some("PolymarketHQ".to_string()),
            include_retweets: false,
        };

        let mut query = query_config.query.clone();
        if let Some(ref account) = query_config.from_account {
            query = format!("from:{} {}", account, query);
        }
        if !query_config.include_retweets {
            query = format!("{} -is:retweet", query);
        }

        assert!(query.contains("from:PolymarketHQ"));
        assert!(query.contains("-is:retweet"));
        assert!(query.contains("polymarket"));
    }
}
