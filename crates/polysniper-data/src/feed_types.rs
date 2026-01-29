//! Feed data types for sentiment analysis
//!
//! Defines FeedItem, FeedConfig, and feed-specific error types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::Duration;
use thiserror::Error;

/// Errors specific to feed operations
#[derive(Error, Debug)]
pub enum FeedError {
    #[error("Failed to fetch feed: {0}")]
    FetchError(String),

    #[error("Failed to parse feed: {0}")]
    ParseError(String),

    #[error("Rate limited: retry after {retry_after:?}")]
    RateLimited { retry_after: Option<Duration> },

    #[error("Authentication failed: {0}")]
    AuthError(String),

    #[error("Feed source not configured: {0}")]
    NotConfigured(String),

    #[error("Invalid feed URL: {0}")]
    InvalidUrl(String),

    #[error("Network error: {0}")]
    NetworkError(String),
}

/// Source of a feed item
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FeedSource {
    Twitter {
        account: Option<String>,
        query: Option<String>,
    },
    Rss {
        feed_url: String,
        feed_title: Option<String>,
    },
}

impl FeedSource {
    pub fn source_name(&self) -> &'static str {
        match self {
            FeedSource::Twitter { .. } => "twitter",
            FeedSource::Rss { .. } => "rss",
        }
    }
}

/// A feed item from any source (Twitter, RSS, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedItem {
    /// Unique identifier for this item (from source)
    pub id: String,
    /// Source of this feed item
    pub source: FeedSource,
    /// Main content/text of the item
    pub content: String,
    /// Title (for RSS items)
    pub title: Option<String>,
    /// Author/username
    pub author: Option<String>,
    /// URL to the original content
    pub url: Option<String>,
    /// When the content was published
    pub published_at: DateTime<Utc>,
    /// When we received this item
    pub received_at: DateTime<Utc>,
    /// Content hash for deduplication
    pub content_hash: String,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Matched keywords that triggered this item
    pub matched_keywords: Vec<String>,
}

impl FeedItem {
    /// Create a new feed item and compute its content hash
    pub fn new(
        id: String,
        source: FeedSource,
        content: String,
        title: Option<String>,
        author: Option<String>,
        url: Option<String>,
        published_at: DateTime<Utc>,
    ) -> Self {
        let content_hash = Self::compute_hash(&content, &url);
        Self {
            id,
            source,
            content,
            title,
            author,
            url,
            published_at,
            received_at: Utc::now(),
            content_hash,
            metadata: HashMap::new(),
            matched_keywords: Vec::new(),
        }
    }

    /// Compute a hash of the content for deduplication
    pub fn compute_hash(content: &str, url: &Option<String>) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        if let Some(u) = url {
            hasher.update(u.as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// Add metadata to this feed item
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Add matched keywords
    pub fn with_matched_keywords(mut self, keywords: Vec<String>) -> Self {
        self.matched_keywords = keywords;
        self
    }
}

/// Configuration for a single Twitter search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwitterQueryConfig {
    /// Search query (e.g., "polymarket OR prediction market")
    pub query: String,
    /// Optional account filter
    pub from_account: Option<String>,
    /// Whether to include retweets
    #[serde(default = "default_false")]
    pub include_retweets: bool,
}

fn default_false() -> bool {
    false
}

/// Configuration for RSS feed sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RssFeedConfig {
    /// URL of the RSS/Atom feed
    pub url: String,
    /// Display name for this feed
    pub name: Option<String>,
    /// Keywords to filter for (empty = all items)
    #[serde(default)]
    pub keywords: Vec<String>,
}

/// Configuration for feed polling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedConfig {
    /// Whether feed ingestion is enabled
    #[serde(default = "default_false")]
    pub enabled: bool,

    /// Twitter configuration
    #[serde(default)]
    pub twitter: TwitterConfig,

    /// RSS feed configuration
    #[serde(default)]
    pub rss: RssConfig,

    /// Global keywords to track across all sources
    #[serde(default)]
    pub keywords: Vec<String>,

    /// Maximum items to keep in deduplication cache
    #[serde(default = "default_dedup_cache_size")]
    pub dedup_cache_size: usize,
}

fn default_dedup_cache_size() -> usize {
    10000
}

impl Default for FeedConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            twitter: TwitterConfig::default(),
            rss: RssConfig::default(),
            keywords: Vec::new(),
            dedup_cache_size: default_dedup_cache_size(),
        }
    }
}

/// Twitter-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwitterConfig {
    /// Whether Twitter polling is enabled
    #[serde(default = "default_false")]
    pub enabled: bool,

    /// Environment variable name for bearer token
    #[serde(default = "default_twitter_bearer_env")]
    pub bearer_token_env: String,

    /// Polling interval in seconds
    #[serde(default = "default_twitter_poll_interval")]
    pub poll_interval_secs: u64,

    /// Search queries to monitor
    #[serde(default)]
    pub queries: Vec<TwitterQueryConfig>,

    /// Maximum results per query
    #[serde(default = "default_max_results")]
    pub max_results: u32,
}

fn default_twitter_bearer_env() -> String {
    "TWITTER_BEARER_TOKEN".to_string()
}

fn default_twitter_poll_interval() -> u64 {
    60 // 1 minute (respecting rate limits)
}

fn default_max_results() -> u32 {
    100
}

impl Default for TwitterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bearer_token_env: default_twitter_bearer_env(),
            poll_interval_secs: default_twitter_poll_interval(),
            queries: Vec::new(),
            max_results: default_max_results(),
        }
    }
}

/// RSS-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RssConfig {
    /// Whether RSS polling is enabled
    #[serde(default = "default_false")]
    pub enabled: bool,

    /// Polling interval in seconds
    #[serde(default = "default_rss_poll_interval")]
    pub poll_interval_secs: u64,

    /// List of RSS feeds to monitor
    #[serde(default)]
    pub feeds: Vec<RssFeedConfig>,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_rss_poll_interval() -> u64 {
    300 // 5 minutes
}

fn default_timeout() -> u64 {
    30
}

impl Default for RssConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            poll_interval_secs: default_rss_poll_interval(),
            feeds: Vec::new(),
            timeout_secs: default_timeout(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_item_hash() {
        let item1 = FeedItem::new(
            "1".to_string(),
            FeedSource::Twitter {
                account: Some("test".to_string()),
                query: None,
            },
            "Hello world".to_string(),
            None,
            Some("test".to_string()),
            Some("https://twitter.com/test/1".to_string()),
            Utc::now(),
        );

        let item2 = FeedItem::new(
            "2".to_string(),
            FeedSource::Twitter {
                account: Some("test".to_string()),
                query: None,
            },
            "Hello world".to_string(),
            None,
            Some("test".to_string()),
            Some("https://twitter.com/test/1".to_string()),
            Utc::now(),
        );

        // Same content + URL = same hash
        assert_eq!(item1.content_hash, item2.content_hash);

        let item3 = FeedItem::new(
            "3".to_string(),
            FeedSource::Twitter {
                account: Some("test".to_string()),
                query: None,
            },
            "Hello world".to_string(),
            None,
            Some("test".to_string()),
            Some("https://twitter.com/test/2".to_string()), // Different URL
            Utc::now(),
        );

        // Different URL = different hash
        assert_ne!(item1.content_hash, item3.content_hash);
    }

    #[test]
    fn test_feed_source_name() {
        let twitter = FeedSource::Twitter {
            account: Some("test".to_string()),
            query: None,
        };
        assert_eq!(twitter.source_name(), "twitter");

        let rss = FeedSource::Rss {
            feed_url: "https://example.com/feed.xml".to_string(),
            feed_title: None,
        };
        assert_eq!(rss.source_name(), "rss");
    }

    #[test]
    fn test_feed_item_with_metadata() {
        let item = FeedItem::new(
            "1".to_string(),
            FeedSource::Twitter {
                account: None,
                query: Some("test query".to_string()),
            },
            "Test content".to_string(),
            None,
            None,
            None,
            Utc::now(),
        )
        .with_metadata("likes", serde_json::json!(100))
        .with_matched_keywords(vec!["test".to_string()]);

        assert_eq!(item.metadata.get("likes"), Some(&serde_json::json!(100)));
        assert_eq!(item.matched_keywords, vec!["test".to_string()]);
    }

    #[test]
    fn test_default_config() {
        let config = FeedConfig::default();
        assert!(!config.enabled);
        assert!(!config.twitter.enabled);
        assert!(!config.rss.enabled);
        assert_eq!(config.dedup_cache_size, 10000);
    }
}
