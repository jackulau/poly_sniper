//! RSS/Atom feed client for feed ingestion
//!
//! Polls RSS and Atom feeds for content matching configured keywords.

use crate::feed_types::{FeedError, FeedItem, FeedSource, RssConfig, RssFeedConfig};
use chrono::Utc;
use feed_rs::parser;
use reqwest::Client;
use std::collections::HashSet;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// RSS/Atom feed client
pub struct RssClient {
    client: Client,
    config: RssConfig,
    /// Track seen item IDs to avoid duplicates between polls
    seen_ids: RwLock<HashSet<String>>,
}

impl RssClient {
    /// Create a new RSS client
    pub fn new(config: RssConfig) -> Result<Self, FeedError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .user_agent("Polysniper/1.0 (RSS Feed Reader)")
            .build()
            .map_err(|e| FeedError::NetworkError(e.to_string()))?;

        Ok(Self {
            client,
            config,
            seen_ids: RwLock::new(HashSet::new()),
        })
    }

    /// Poll all configured feeds and return new items
    pub async fn poll(&self) -> Result<Vec<FeedItem>, FeedError> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        let mut all_items = Vec::new();

        for feed_config in &self.config.feeds {
            match self.fetch_feed(feed_config).await {
                Ok(items) => {
                    info!(
                        "Fetched {} items from RSS feed '{}'",
                        items.len(),
                        feed_config.name.as_deref().unwrap_or(&feed_config.url)
                    );
                    all_items.extend(items);
                }
                Err(e) => {
                    warn!(
                        "Failed to fetch RSS feed '{}': {}",
                        feed_config.name.as_deref().unwrap_or(&feed_config.url),
                        e
                    );
                    // Continue with other feeds
                }
            }
        }

        Ok(all_items)
    }

    /// Fetch and parse a single RSS/Atom feed
    async fn fetch_feed(&self, feed_config: &RssFeedConfig) -> Result<Vec<FeedItem>, FeedError> {
        debug!("Fetching RSS feed: {}", feed_config.url);

        let response = self
            .client
            .get(&feed_config.url)
            .send()
            .await
            .map_err(|e| FeedError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(FeedError::FetchError(format!(
                "RSS feed returned status {}",
                response.status()
            )));
        }

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        // Validate content type (allow various RSS/Atom/XML types)
        let valid_types = [
            "application/rss+xml",
            "application/atom+xml",
            "application/xml",
            "text/xml",
            "application/rdf+xml",
        ];
        let is_valid_type = valid_types.iter().any(|t| content_type.contains(t))
            || content_type.is_empty(); // Some servers don't set content-type

        if !is_valid_type && !content_type.contains("xml") {
            warn!(
                "Unexpected content type '{}' for RSS feed, attempting to parse anyway",
                content_type
            );
        }

        let body = response
            .bytes()
            .await
            .map_err(|e| FeedError::NetworkError(e.to_string()))?;

        let feed = parser::parse(&body[..])
            .map_err(|e| FeedError::ParseError(format!("Failed to parse feed: {}", e)))?;

        let feed_title = feed.title.map(|t| t.content);

        // Get seen IDs to filter duplicates
        let seen_ids = self.seen_ids.read().await;

        let items: Vec<FeedItem> = feed
            .entries
            .into_iter()
            .filter_map(|entry| {
                let entry_id = entry.id.clone();

                // Skip if already seen
                if seen_ids.contains(&entry_id) {
                    return None;
                }

                self.convert_entry(entry, feed_config, feed_title.as_deref())
            })
            .filter(|item| {
                // Filter by keywords if configured
                if feed_config.keywords.is_empty() {
                    return true;
                }
                let content_lower = item.content.to_lowercase();
                let title_lower = item.title.as_deref().unwrap_or("").to_lowercase();
                feed_config.keywords.iter().any(|kw| {
                    let kw_lower = kw.to_lowercase();
                    content_lower.contains(&kw_lower) || title_lower.contains(&kw_lower)
                })
            })
            .collect();

        // Update seen IDs
        drop(seen_ids);
        let mut seen_ids = self.seen_ids.write().await;
        for item in &items {
            seen_ids.insert(item.id.clone());
        }

        // Limit cache size
        if seen_ids.len() > 10000 {
            // Keep only recent items by clearing oldest (simple approach)
            seen_ids.clear();
            for item in &items {
                seen_ids.insert(item.id.clone());
            }
        }

        Ok(items)
    }

    fn convert_entry(
        &self,
        entry: feed_rs::model::Entry,
        feed_config: &RssFeedConfig,
        feed_title: Option<&str>,
    ) -> Option<FeedItem> {
        // Get content from various possible fields
        let content = entry
            .content
            .and_then(|c| c.body)
            .or_else(|| entry.summary.map(|s| s.content))
            .unwrap_or_default();

        // Get the first link
        let url = entry.links.first().map(|l| l.href.clone());

        // Get published date
        let published_at = entry
            .published
            .or(entry.updated)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        // Get author
        let author = entry
            .authors
            .first()
            .map(|a| a.name.clone());

        // Get title
        let title = entry.title.map(|t| t.content);

        // Strip HTML tags from content for cleaner text
        let clean_content = strip_html_tags(&content);

        let mut item = FeedItem::new(
            entry.id,
            FeedSource::Rss {
                feed_url: feed_config.url.clone(),
                feed_title: feed_config.name.clone().or_else(|| feed_title.map(|s| s.to_string())),
            },
            clean_content,
            title,
            author,
            url,
            published_at,
        );

        // Add categories as metadata
        if !entry.categories.is_empty() {
            let categories: Vec<String> = entry
                .categories
                .into_iter()
                .map(|c| c.term)
                .collect();
            item = item.with_metadata("categories", serde_json::json!(categories));
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

    /// Clear the seen IDs cache (useful for testing or full refresh)
    pub async fn clear_cache(&self) {
        let mut seen_ids = self.seen_ids.write().await;
        seen_ids.clear();
    }
}

/// Strip HTML tags from text content
fn strip_html_tags(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut in_tag = false;
    let mut in_entity = false;
    let mut entity_buf = String::new();

    for c in input.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            '&' if !in_tag => {
                in_entity = true;
                entity_buf.clear();
            }
            ';' if in_entity => {
                in_entity = false;
                // Decode common HTML entities
                match entity_buf.as_str() {
                    "amp" => result.push('&'),
                    "lt" => result.push('<'),
                    "gt" => result.push('>'),
                    "quot" => result.push('"'),
                    "apos" => result.push('\''),
                    "nbsp" => result.push(' '),
                    _ if entity_buf.starts_with('#') => {
                        // Numeric entity
                        if let Ok(code) = entity_buf[1..].parse::<u32>() {
                            if let Some(ch) = char::from_u32(code) {
                                result.push(ch);
                            }
                        }
                    }
                    _ => {
                        // Unknown entity, keep as-is
                        result.push('&');
                        result.push_str(&entity_buf);
                        result.push(';');
                    }
                }
                entity_buf.clear();
            }
            _ if in_entity => entity_buf.push(c),
            _ if !in_tag => result.push(c),
            _ => {}
        }
    }

    // Collapse multiple whitespace to single space
    let mut prev_whitespace = false;
    result
        .chars()
        .filter(|c| {
            let is_ws = c.is_whitespace();
            if is_ws && prev_whitespace {
                false
            } else {
                prev_whitespace = is_ws;
                true
            }
        })
        .collect::<String>()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(
            strip_html_tags("<p>Hello <b>world</b>!</p>"),
            "Hello world!"
        );
        assert_eq!(
            strip_html_tags("No tags here"),
            "No tags here"
        );
        assert_eq!(
            strip_html_tags("<div class=\"test\">Content</div>"),
            "Content"
        );
        assert_eq!(
            strip_html_tags("&amp; &lt; &gt; &quot;"),
            "& < > \""
        );
        assert_eq!(
            strip_html_tags("Multiple   spaces   here"),
            "Multiple spaces here"
        );
    }

    #[test]
    fn test_rss_feed_source() {
        let source = FeedSource::Rss {
            feed_url: "https://example.com/feed.xml".to_string(),
            feed_title: Some("Example Feed".to_string()),
        };
        assert_eq!(source.source_name(), "rss");
    }

    #[test]
    fn test_rss_config_defaults() {
        let config = RssConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.poll_interval_secs, 300);
        assert!(config.feeds.is_empty());
    }

    #[tokio::test]
    async fn test_seen_ids_cache() {
        let config = RssConfig {
            enabled: false,
            poll_interval_secs: 60,
            feeds: vec![],
            timeout_secs: 30,
        };
        let client = RssClient::new(config).unwrap();

        // Add some IDs
        {
            let mut seen = client.seen_ids.write().await;
            seen.insert("id1".to_string());
            seen.insert("id2".to_string());
        }

        // Verify they're there
        {
            let seen = client.seen_ids.read().await;
            assert!(seen.contains("id1"));
            assert!(seen.contains("id2"));
        }

        // Clear cache
        client.clear_cache().await;

        // Verify they're gone
        {
            let seen = client.seen_ids.read().await;
            assert!(!seen.contains("id1"));
            assert!(!seen.contains("id2"));
        }
    }
}
