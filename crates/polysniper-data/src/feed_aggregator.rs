//! Feed aggregator for managing multiple feed sources
//!
//! Coordinates polling across Twitter and RSS feeds, handles deduplication,
//! and publishes FeedItemReceived events to the event bus.

use crate::feed_types::{FeedConfig, FeedError, FeedItem};
use crate::rss_client::RssClient;
use crate::twitter_client::TwitterClient;
use polysniper_core::{EventBus, SystemEvent};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::{interval, MissedTickBehavior};
use tracing::{debug, error, info, warn};

/// Feed aggregator that manages multiple feed sources
pub struct FeedAggregator<E: EventBus> {
    config: FeedConfig,
    twitter_client: Option<TwitterClient>,
    rss_client: Option<RssClient>,
    event_bus: Arc<E>,
    /// Content hashes of recently seen items for deduplication
    seen_hashes: RwLock<HashSet<String>>,
    /// Global keywords to match across all sources
    global_keywords: Vec<String>,
}

impl<E: EventBus + 'static> FeedAggregator<E> {
    /// Create a new feed aggregator
    pub fn new(config: FeedConfig, event_bus: Arc<E>) -> Result<Self, FeedError> {
        let twitter_client = if config.twitter.enabled {
            match TwitterClient::new(config.twitter.clone()) {
                Ok(client) => Some(client),
                Err(e) => {
                    warn!("Failed to create Twitter client: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let rss_client = if config.rss.enabled {
            match RssClient::new(config.rss.clone()) {
                Ok(client) => Some(client),
                Err(e) => {
                    warn!("Failed to create RSS client: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let global_keywords = config.keywords.clone();

        Ok(Self {
            config,
            twitter_client,
            rss_client,
            event_bus,
            seen_hashes: RwLock::new(HashSet::new()),
            global_keywords,
        })
    }

    /// Start the feed aggregator polling loop
    ///
    /// This will spawn background tasks for each enabled feed source.
    pub async fn start(self: Arc<Self>) -> Result<(), FeedError> {
        if !self.config.enabled {
            info!("Feed aggregator is disabled");
            return Ok(());
        }

        info!("Starting feed aggregator");

        let mut handles = Vec::new();

        // Start Twitter polling if enabled
        if self.twitter_client.is_some() {
            let aggregator = Arc::clone(&self);
            let handle = tokio::spawn(async move {
                aggregator.poll_twitter_loop().await;
            });
            handles.push(handle);
            info!("Started Twitter polling");
        }

        // Start RSS polling if enabled
        if self.rss_client.is_some() {
            let aggregator = Arc::clone(&self);
            let handle = tokio::spawn(async move {
                aggregator.poll_rss_loop().await;
            });
            handles.push(handle);
            info!("Started RSS polling");
        }

        // Wait for all tasks (they run indefinitely)
        for handle in handles {
            if let Err(e) = handle.await {
                error!("Feed polling task failed: {}", e);
            }
        }

        Ok(())
    }

    /// Poll Twitter in a loop
    async fn poll_twitter_loop(&self) {
        let client = match &self.twitter_client {
            Some(c) => c,
            None => return,
        };

        let mut poll_interval = interval(client.poll_interval());
        poll_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            poll_interval.tick().await;

            match client.poll().await {
                Ok(items) => {
                    self.process_items(items).await;
                }
                Err(FeedError::RateLimited { retry_after }) => {
                    let wait_time = retry_after.unwrap_or(Duration::from_secs(60));
                    warn!("Twitter rate limited, waiting {:?}", wait_time);
                    tokio::time::sleep(wait_time).await;
                }
                Err(e) => {
                    warn!("Twitter poll failed: {}", e);
                }
            }
        }
    }

    /// Poll RSS feeds in a loop
    async fn poll_rss_loop(&self) {
        let client = match &self.rss_client {
            Some(c) => c,
            None => return,
        };

        let mut poll_interval = interval(client.poll_interval());
        poll_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            poll_interval.tick().await;

            match client.poll().await {
                Ok(items) => {
                    self.process_items(items).await;
                }
                Err(e) => {
                    warn!("RSS poll failed: {}", e);
                }
            }
        }
    }

    /// Process feed items: deduplicate, match keywords, and publish events
    async fn process_items(&self, items: Vec<FeedItem>) {
        let mut seen_hashes = self.seen_hashes.write().await;

        for mut item in items {
            // Deduplicate by content hash
            if seen_hashes.contains(&item.content_hash) {
                debug!("Skipping duplicate item: {}", item.id);
                continue;
            }

            // Match global keywords
            let matched = self.match_keywords(&item);
            if !matched.is_empty() {
                item = item.with_matched_keywords(matched);
            }

            // Add to seen hashes
            seen_hashes.insert(item.content_hash.clone());

            // Publish event
            self.publish_item(&item);

            debug!(
                "Published feed item: {} from {:?}",
                item.id,
                item.source.source_name()
            );
        }

        // Limit cache size
        if seen_hashes.len() > self.config.dedup_cache_size {
            // Simple approach: clear and rely on source-level dedup
            let excess = seen_hashes.len() - self.config.dedup_cache_size;
            let to_remove: Vec<String> = seen_hashes.iter().take(excess).cloned().collect();
            for hash in to_remove {
                seen_hashes.remove(&hash);
            }
        }
    }

    /// Match item content against global keywords
    fn match_keywords(&self, item: &FeedItem) -> Vec<String> {
        if self.global_keywords.is_empty() {
            return Vec::new();
        }

        let content_lower = item.content.to_lowercase();
        let title_lower = item.title.as_deref().unwrap_or("").to_lowercase();

        self.global_keywords
            .iter()
            .filter(|kw| {
                let kw_lower = kw.to_lowercase();
                content_lower.contains(&kw_lower) || title_lower.contains(&kw_lower)
            })
            .cloned()
            .collect()
    }

    /// Publish a feed item as a system event
    fn publish_item(&self, item: &FeedItem) {
        use polysniper_core::{FeedItem as CoreFeedItem, FeedItemReceivedEvent, FeedItemSource};

        // Convert local FeedItem to core FeedItem
        let core_source = match &item.source {
            crate::feed_types::FeedSource::Twitter { account, query } => FeedItemSource::Twitter {
                account: account.clone(),
                query: query.clone(),
            },
            crate::feed_types::FeedSource::Rss { feed_url, feed_title } => FeedItemSource::Rss {
                feed_url: feed_url.clone(),
                feed_title: feed_title.clone(),
            },
        };

        let core_item = CoreFeedItem {
            id: item.id.clone(),
            source: core_source,
            content: item.content.clone(),
            title: item.title.clone(),
            author: item.author.clone(),
            url: item.url.clone(),
            published_at: item.published_at,
            received_at: item.received_at,
            content_hash: item.content_hash.clone(),
            metadata: item.metadata.clone(),
            matched_keywords: item.matched_keywords.clone(),
        };

        let event = SystemEvent::FeedItemReceived(FeedItemReceivedEvent {
            item: core_item,
            received_at: item.received_at,
        });

        self.event_bus.publish(event);
    }

    /// Poll all sources once (for testing or manual triggers)
    pub async fn poll_once(&self) -> Result<Vec<FeedItem>, FeedError> {
        let mut all_items = Vec::new();

        if let Some(ref client) = self.twitter_client {
            match client.poll().await {
                Ok(items) => all_items.extend(items),
                Err(e) => warn!("Twitter poll failed: {}", e),
            }
        }

        if let Some(ref client) = self.rss_client {
            match client.poll().await {
                Ok(items) => all_items.extend(items),
                Err(e) => warn!("RSS poll failed: {}", e),
            }
        }

        // Process and deduplicate
        self.process_items(all_items.clone()).await;

        Ok(all_items)
    }

    /// Check if the aggregator is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the number of unique items seen
    pub async fn seen_count(&self) -> usize {
        self.seen_hashes.read().await.len()
    }

    /// Clear the deduplication cache
    pub async fn clear_cache(&self) {
        let mut seen_hashes = self.seen_hashes.write().await;
        seen_hashes.clear();

        if let Some(ref client) = self.rss_client {
            client.clear_cache().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::broadcast;

    /// Mock event bus for testing
    struct MockEventBus {
        sender: broadcast::Sender<SystemEvent>,
        publish_count: AtomicUsize,
    }

    impl MockEventBus {
        fn new() -> Self {
            let (sender, _) = broadcast::channel(100);
            Self {
                sender,
                publish_count: AtomicUsize::new(0),
            }
        }

        fn publish_count(&self) -> usize {
            self.publish_count.load(Ordering::SeqCst)
        }
    }

    impl EventBus for MockEventBus {
        fn publish(&self, event: SystemEvent) {
            self.publish_count.fetch_add(1, Ordering::SeqCst);
            let _ = self.sender.send(event);
        }

        fn subscribe(&self) -> broadcast::Receiver<SystemEvent> {
            self.sender.subscribe()
        }

        fn subscriber_count(&self) -> usize {
            0
        }
    }

    #[tokio::test]
    async fn test_feed_aggregator_disabled() {
        let config = FeedConfig::default();
        let event_bus = Arc::new(MockEventBus::new());
        let aggregator = FeedAggregator::new(config, event_bus.clone()).unwrap();

        assert!(!aggregator.is_enabled());
    }

    #[tokio::test]
    async fn test_deduplication() {
        let mut config = FeedConfig::default();
        config.enabled = true;
        config.dedup_cache_size = 100;

        let event_bus = Arc::new(MockEventBus::new());
        let aggregator = FeedAggregator::new(config, event_bus.clone()).unwrap();

        // Create duplicate items
        let item1 = FeedItem::new(
            "1".to_string(),
            crate::feed_types::FeedSource::Twitter {
                account: Some("test".to_string()),
                query: None,
            },
            "Same content".to_string(),
            None,
            None,
            Some("https://example.com/1".to_string()),
            chrono::Utc::now(),
        );

        let item2 = FeedItem::new(
            "2".to_string(),
            crate::feed_types::FeedSource::Twitter {
                account: Some("test".to_string()),
                query: None,
            },
            "Same content".to_string(),
            None,
            None,
            Some("https://example.com/1".to_string()), // Same URL = same hash
            chrono::Utc::now(),
        );

        // Process first item
        aggregator.process_items(vec![item1]).await;
        assert_eq!(event_bus.publish_count(), 1);
        assert_eq!(aggregator.seen_count().await, 1);

        // Process duplicate - should be skipped
        aggregator.process_items(vec![item2]).await;
        assert_eq!(event_bus.publish_count(), 1); // Still 1
        assert_eq!(aggregator.seen_count().await, 1);
    }

    #[tokio::test]
    async fn test_keyword_matching() {
        let mut config = FeedConfig::default();
        config.enabled = true;
        config.keywords = vec!["polymarket".to_string(), "prediction".to_string()];

        let event_bus = Arc::new(MockEventBus::new());
        let aggregator = FeedAggregator::new(config, event_bus).unwrap();

        let item = FeedItem::new(
            "1".to_string(),
            crate::feed_types::FeedSource::Rss {
                feed_url: "https://example.com/feed".to_string(),
                feed_title: None,
            },
            "This is about Polymarket and prediction markets".to_string(),
            None,
            None,
            None,
            chrono::Utc::now(),
        );

        let matched = aggregator.match_keywords(&item);
        assert!(matched.contains(&"polymarket".to_string()));
        assert!(matched.contains(&"prediction".to_string()));
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let mut config = FeedConfig::default();
        config.enabled = true;

        let event_bus = Arc::new(MockEventBus::new());
        let aggregator = FeedAggregator::new(config, event_bus).unwrap();

        let item = FeedItem::new(
            "1".to_string(),
            crate::feed_types::FeedSource::Twitter {
                account: None,
                query: None,
            },
            "Test content".to_string(),
            None,
            None,
            None,
            chrono::Utc::now(),
        );

        aggregator.process_items(vec![item]).await;
        assert_eq!(aggregator.seen_count().await, 1);

        aggregator.clear_cache().await;
        assert_eq!(aggregator.seen_count().await, 0);
    }
}
