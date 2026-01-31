//! News Velocity Tracking
//!
//! Tracks the rate-of-change in news coverage for keywords, enabling early detection
//! of emerging narratives before they're priced in.

use chrono::{DateTime, Duration, Utc};
use polysniper_core::{
    EventBus, FeedItem, MarketId, NewsVelocitySignalEvent, SystemEvent, VelocityDirection,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Configuration for news velocity tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsVelocityConfig {
    /// Whether velocity tracking is enabled
    pub enabled: bool,
    /// Keywords to track with their aliases
    #[serde(default)]
    pub tracking_keywords: Vec<KeywordTracking>,
    /// Time windows for velocity calculation (in seconds)
    #[serde(default = "default_velocity_windows")]
    pub velocity_windows: VelocityWindows,
    /// Acceleration threshold (e.g., 2.0 = 2x baseline)
    #[serde(default = "default_acceleration_threshold")]
    pub acceleration_threshold: Decimal,
    /// Deceleration threshold (e.g., 0.5 = half baseline)
    #[serde(default = "default_deceleration_threshold")]
    pub deceleration_threshold: Decimal,
    /// Minimum articles for signal generation
    #[serde(default = "default_min_articles")]
    pub min_articles_for_signal: u32,
    /// Keyword to market ID mappings
    #[serde(default)]
    pub market_mappings: HashMap<String, Vec<MarketId>>,
    /// Maximum number of articles to keep in history per keyword
    #[serde(default = "default_max_history")]
    pub max_history_per_keyword: usize,
    /// Cooldown between signals for the same keyword (seconds)
    #[serde(default = "default_signal_cooldown")]
    pub signal_cooldown_secs: u64,
    /// Maximum headlines to include in signal
    #[serde(default = "default_max_headlines")]
    pub max_sample_headlines: usize,
}

fn default_velocity_windows() -> VelocityWindows {
    VelocityWindows::default()
}

fn default_acceleration_threshold() -> Decimal {
    Decimal::new(2, 0) // 2.0
}

fn default_deceleration_threshold() -> Decimal {
    Decimal::new(5, 1) // 0.5
}

fn default_min_articles() -> u32 {
    5
}

fn default_max_history() -> usize {
    1000
}

fn default_signal_cooldown() -> u64 {
    1800 // 30 minutes
}

fn default_max_headlines() -> usize {
    5
}

impl Default for NewsVelocityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tracking_keywords: Vec::new(),
            velocity_windows: VelocityWindows::default(),
            acceleration_threshold: default_acceleration_threshold(),
            deceleration_threshold: default_deceleration_threshold(),
            min_articles_for_signal: default_min_articles(),
            market_mappings: HashMap::new(),
            max_history_per_keyword: default_max_history(),
            signal_cooldown_secs: default_signal_cooldown(),
            max_sample_headlines: default_max_headlines(),
        }
    }
}

/// Time windows for velocity calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityWindows {
    /// Short window in seconds (default: 1 hour)
    pub short_secs: u64,
    /// Medium window in seconds (default: 6 hours)
    pub medium_secs: u64,
    /// Long window in seconds (default: 24 hours)
    pub long_secs: u64,
}

impl Default for VelocityWindows {
    fn default() -> Self {
        Self {
            short_secs: 3600,   // 1 hour
            medium_secs: 21600, // 6 hours
            long_secs: 86400,   // 24 hours
        }
    }
}

/// Keyword tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordTracking {
    /// Primary keyword to track
    pub keyword: String,
    /// Aliases that also match (case-insensitive)
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Category of the keyword
    #[serde(default)]
    pub category: NewsCategory,
}

impl KeywordTracking {
    /// Create a new keyword tracking config
    pub fn new(keyword: impl Into<String>) -> Self {
        Self {
            keyword: keyword.into(),
            aliases: Vec::new(),
            category: NewsCategory::Other,
        }
    }

    /// Add aliases
    pub fn with_aliases(mut self, aliases: Vec<String>) -> Self {
        self.aliases = aliases;
        self
    }

    /// Set category
    pub fn with_category(mut self, category: NewsCategory) -> Self {
        self.category = category;
        self
    }

    /// Check if text matches this keyword or its aliases
    pub fn matches(&self, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        let keyword_lower = self.keyword.to_lowercase();

        if text_lower.contains(&keyword_lower) {
            return true;
        }

        for alias in &self.aliases {
            if text_lower.contains(&alias.to_lowercase()) {
                return true;
            }
        }

        false
    }
}

/// News category for classification
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NewsCategory {
    Politics,
    Crypto,
    Sports,
    Finance,
    Technology,
    #[default]
    Other,
}

impl NewsCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            NewsCategory::Politics => "politics",
            NewsCategory::Crypto => "crypto",
            NewsCategory::Sports => "sports",
            NewsCategory::Finance => "finance",
            NewsCategory::Technology => "technology",
            NewsCategory::Other => "other",
        }
    }
}

/// A timestamped article record for velocity tracking
#[derive(Debug, Clone)]
pub struct ArticleTimestamp {
    /// Source of the article (e.g., "twitter", "rss")
    pub source: String,
    /// When the article was published/received
    pub timestamp: DateTime<Utc>,
    /// Hash of the title for deduplication
    pub title_hash: String,
    /// Title for sample headlines
    pub title: Option<String>,
    /// Sentiment score if available
    pub sentiment: Option<Decimal>,
}

impl ArticleTimestamp {
    /// Create a new article timestamp
    pub fn new(source: String, timestamp: DateTime<Utc>, title: Option<String>) -> Self {
        let title_hash = title
            .as_ref()
            .map(|t| {
                let mut hasher = Sha256::new();
                hasher.update(t.as_bytes());
                format!("{:x}", hasher.finalize())[..16].to_string()
            })
            .unwrap_or_default();

        Self {
            source,
            timestamp,
            title_hash,
            title,
            sentiment: None,
        }
    }

    /// Add sentiment score
    pub fn with_sentiment(mut self, sentiment: Decimal) -> Self {
        self.sentiment = Some(sentiment);
        self
    }
}

/// Velocity metrics for a keyword
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityMetrics {
    /// Keyword being tracked
    pub keyword: String,
    /// Current velocity (articles per hour, from short window)
    pub current_velocity: Decimal,
    /// Velocity over 1 hour
    pub velocity_1h: Decimal,
    /// Velocity over 6 hours
    pub velocity_6h: Decimal,
    /// Velocity over 24 hours
    pub velocity_24h: Decimal,
    /// Acceleration (rate of change relative to baseline)
    pub acceleration: Decimal,
    /// Historical baseline velocity
    pub baseline_velocity: Decimal,
    /// Article count in last hour
    pub article_count_1h: u32,
    /// Article count in last 24 hours
    pub article_count_24h: u32,
    /// When metrics were last updated
    pub last_updated: DateTime<Utc>,
}

impl Default for VelocityMetrics {
    fn default() -> Self {
        Self {
            keyword: String::new(),
            current_velocity: Decimal::ZERO,
            velocity_1h: Decimal::ZERO,
            velocity_6h: Decimal::ZERO,
            velocity_24h: Decimal::ZERO,
            acceleration: Decimal::ZERO,
            baseline_velocity: Decimal::ZERO,
            article_count_1h: 0,
            article_count_24h: 0,
            last_updated: Utc::now(),
        }
    }
}

/// News velocity tracker
pub struct NewsVelocityTracker<E: EventBus> {
    config: NewsVelocityConfig,
    /// Article history per keyword
    article_history: RwLock<HashMap<String, VecDeque<ArticleTimestamp>>>,
    /// Cached velocity metrics per keyword
    velocity_cache: RwLock<HashMap<String, VelocityMetrics>>,
    /// Event bus for publishing signals
    event_bus: Arc<E>,
    /// Last signal time per keyword (for cooldown)
    last_signal_time: RwLock<HashMap<String, DateTime<Utc>>>,
    /// Seen title hashes for deduplication
    seen_hashes: RwLock<HashMap<String, DateTime<Utc>>>,
}

impl<E: EventBus + 'static> NewsVelocityTracker<E> {
    /// Create a new news velocity tracker
    pub fn new(config: NewsVelocityConfig, event_bus: Arc<E>) -> Self {
        Self {
            config,
            article_history: RwLock::new(HashMap::new()),
            velocity_cache: RwLock::new(HashMap::new()),
            event_bus,
            last_signal_time: RwLock::new(HashMap::new()),
            seen_hashes: RwLock::new(HashMap::new()),
        }
    }

    /// Check if tracking is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Process a feed item and update velocity tracking
    pub async fn process_feed_item(&self, item: &FeedItem) {
        if !self.config.enabled {
            return;
        }

        for keyword_config in &self.config.tracking_keywords {
            if self.matches_keyword(item, keyword_config) {
                // Check for duplicate
                let title = item.title.clone().or_else(|| {
                    let content = &item.content;
                    if content.len() > 100 {
                        Some(content[..100].to_string())
                    } else {
                        Some(content.clone())
                    }
                });

                let article = ArticleTimestamp::new(
                    item.source.source_name().to_string(),
                    item.published_at,
                    title,
                );

                // Skip if we've seen this article recently
                if self.is_duplicate(&article).await {
                    debug!(
                        keyword = %keyword_config.keyword,
                        "Skipping duplicate article"
                    );
                    continue;
                }

                // Record the article
                self.record_article(&keyword_config.keyword, article).await;

                // Update metrics and check for signals
                let metrics = self.update_metrics(&keyword_config.keyword).await;

                debug!(
                    keyword = %keyword_config.keyword,
                    velocity_1h = %metrics.velocity_1h,
                    acceleration = %metrics.acceleration,
                    article_count_1h = %metrics.article_count_1h,
                    "Updated velocity metrics"
                );

                // Check for velocity signal
                self.check_and_emit_signal(&keyword_config.keyword, &metrics)
                    .await;
            }
        }
    }

    /// Check if a feed item matches a keyword configuration
    fn matches_keyword(&self, item: &FeedItem, keyword_config: &KeywordTracking) -> bool {
        let content = &item.content;
        let title = item.title.as_deref().unwrap_or("");
        let combined = format!("{} {}", title, content);

        keyword_config.matches(&combined)
    }

    /// Check if an article is a duplicate
    async fn is_duplicate(&self, article: &ArticleTimestamp) -> bool {
        if article.title_hash.is_empty() {
            return false;
        }

        let mut seen = self.seen_hashes.write().await;
        let cutoff = Utc::now() - Duration::hours(24);

        // Clean old entries
        seen.retain(|_, ts| *ts > cutoff);

        if seen.contains_key(&article.title_hash) {
            return true;
        }

        seen.insert(article.title_hash.clone(), article.timestamp);
        false
    }

    /// Record an article for a keyword
    async fn record_article(&self, keyword: &str, article: ArticleTimestamp) {
        let mut history = self.article_history.write().await;
        let articles = history
            .entry(keyword.to_string())
            .or_insert_with(VecDeque::new);

        articles.push_back(article);

        // Limit history size
        while articles.len() > self.config.max_history_per_keyword {
            articles.pop_front();
        }
    }

    /// Update velocity metrics for a keyword
    async fn update_metrics(&self, keyword: &str) -> VelocityMetrics {
        let history = self.article_history.read().await;
        let articles = history.get(keyword);

        let now = Utc::now();

        let (velocity_1h, count_1h) =
            self.calculate_velocity_and_count(articles, Duration::hours(1), now);
        let (velocity_6h, _) = self.calculate_velocity_and_count(articles, Duration::hours(6), now);
        let (velocity_24h, count_24h) =
            self.calculate_velocity_and_count(articles, Duration::hours(24), now);

        // Use 6h or 24h velocity as baseline, whichever is larger
        let baseline = velocity_6h.max(velocity_24h);
        let baseline = if baseline.is_zero() {
            Decimal::new(1, 1) // 0.1 - Minimum baseline to avoid division by zero
        } else {
            baseline
        };

        // Calculate acceleration as ratio of current (1h) to baseline
        let acceleration = if baseline.is_zero() {
            velocity_1h
        } else {
            velocity_1h / baseline
        };

        let metrics = VelocityMetrics {
            keyword: keyword.to_string(),
            current_velocity: velocity_1h,
            velocity_1h,
            velocity_6h,
            velocity_24h,
            acceleration,
            baseline_velocity: baseline,
            article_count_1h: count_1h,
            article_count_24h: count_24h,
            last_updated: now,
        };

        // Update cache
        let mut cache = self.velocity_cache.write().await;
        cache.insert(keyword.to_string(), metrics.clone());

        metrics
    }

    /// Calculate velocity (articles per hour) and count for a time window
    fn calculate_velocity_and_count(
        &self,
        articles: Option<&VecDeque<ArticleTimestamp>>,
        window: Duration,
        now: DateTime<Utc>,
    ) -> (Decimal, u32) {
        let articles = match articles {
            Some(a) => a,
            None => return (Decimal::ZERO, 0),
        };

        let cutoff = now - window;
        let count = articles.iter().filter(|a| a.timestamp > cutoff).count() as u32;

        let hours = window.num_hours().max(1) as u64;
        let velocity = Decimal::from(count) / Decimal::from(hours);

        (velocity, count)
    }

    /// Check for velocity signal and emit if thresholds are met
    async fn check_and_emit_signal(&self, keyword: &str, metrics: &VelocityMetrics) {
        // Check minimum article count
        if metrics.article_count_1h < self.config.min_articles_for_signal {
            return;
        }

        // Check cooldown
        if !self.check_cooldown(keyword).await {
            debug!(
                keyword = %keyword,
                "Signal cooldown active"
            );
            return;
        }

        let direction = if metrics.acceleration >= self.config.acceleration_threshold {
            VelocityDirection::Accelerating
        } else if metrics.acceleration <= self.config.deceleration_threshold {
            VelocityDirection::Decelerating
        } else {
            return; // No signal needed for stable velocity
        };

        // Get market IDs for this keyword
        let market_ids = self
            .config
            .market_mappings
            .get(keyword)
            .cloned()
            .unwrap_or_default();

        // Get sample headlines
        let sample_headlines = self.get_sample_headlines(keyword).await;

        info!(
            keyword = %keyword,
            direction = %direction,
            velocity = %metrics.current_velocity,
            acceleration = %metrics.acceleration,
            article_count_1h = %metrics.article_count_1h,
            "Emitting news velocity signal"
        );

        // Record signal time for cooldown
        self.record_signal_time(keyword).await;

        // Emit the signal
        let signal = NewsVelocitySignalEvent::new(
            keyword.to_string(),
            market_ids,
            direction,
            metrics.current_velocity,
            metrics.baseline_velocity,
            metrics.acceleration,
            metrics.article_count_1h,
            metrics.article_count_24h,
            sample_headlines,
        );

        self.event_bus
            .publish(SystemEvent::NewsVelocitySignal(signal));
    }

    /// Check if cooldown has passed for a keyword
    async fn check_cooldown(&self, keyword: &str) -> bool {
        let last_times = self.last_signal_time.read().await;
        if let Some(last_time) = last_times.get(keyword) {
            let cooldown = Duration::seconds(self.config.signal_cooldown_secs as i64);
            if Utc::now() - *last_time < cooldown {
                return false;
            }
        }
        true
    }

    /// Record signal time for cooldown tracking
    async fn record_signal_time(&self, keyword: &str) {
        let mut last_times = self.last_signal_time.write().await;
        last_times.insert(keyword.to_string(), Utc::now());
    }

    /// Get sample headlines for a keyword
    async fn get_sample_headlines(&self, keyword: &str) -> Vec<String> {
        let history = self.article_history.read().await;
        let articles = match history.get(keyword) {
            Some(a) => a,
            None => return Vec::new(),
        };

        let cutoff = Utc::now() - Duration::hours(1);
        articles
            .iter()
            .rev()
            .filter(|a| a.timestamp > cutoff && a.title.is_some())
            .take(self.config.max_sample_headlines)
            .filter_map(|a| a.title.clone())
            .collect()
    }

    /// Get current velocity metrics for a keyword
    pub async fn get_metrics(&self, keyword: &str) -> Option<VelocityMetrics> {
        let cache = self.velocity_cache.read().await;
        cache.get(keyword).cloned()
    }

    /// Get all tracked keywords
    pub fn tracked_keywords(&self) -> Vec<&str> {
        self.config
            .tracking_keywords
            .iter()
            .map(|k| k.keyword.as_str())
            .collect()
    }

    /// Get article count for a keyword in a time window
    pub async fn get_article_count(&self, keyword: &str, hours: i64) -> u32 {
        let history = self.article_history.read().await;
        let articles = match history.get(keyword) {
            Some(a) => a,
            None => return 0,
        };

        let cutoff = Utc::now() - Duration::hours(hours);
        articles.iter().filter(|a| a.timestamp > cutoff).count() as u32
    }

    /// Clear all tracked data
    pub async fn clear(&self) {
        let mut history = self.article_history.write().await;
        history.clear();

        let mut cache = self.velocity_cache.write().await;
        cache.clear();

        let mut seen = self.seen_hashes.write().await;
        seen.clear();

        let mut last_times = self.last_signal_time.write().await;
        last_times.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::broadcast;

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

    fn test_config() -> NewsVelocityConfig {
        let mut config = NewsVelocityConfig::default();
        config.enabled = true;
        config.min_articles_for_signal = 2;
        config.signal_cooldown_secs = 1;
        config.tracking_keywords = vec![
            KeywordTracking::new("trump")
                .with_aliases(vec!["Donald Trump".to_string(), "POTUS".to_string()])
                .with_category(NewsCategory::Politics),
            KeywordTracking::new("bitcoin")
                .with_aliases(vec!["BTC".to_string(), "crypto".to_string()])
                .with_category(NewsCategory::Crypto),
        ];
        config
            .market_mappings
            .insert("trump".to_string(), vec!["trump-market-id".to_string()]);
        config
    }

    fn create_feed_item(content: &str, title: Option<&str>) -> FeedItem {
        FeedItem {
            id: uuid::Uuid::new_v4().to_string(),
            source: polysniper_core::FeedItemSource::Twitter {
                account: Some("test".to_string()),
                query: None,
            },
            content: content.to_string(),
            title: title.map(String::from),
            author: Some("test_author".to_string()),
            url: Some("https://example.com".to_string()),
            published_at: Utc::now(),
            received_at: Utc::now(),
            content_hash: format!("{:x}", Sha256::new().chain_update(content).finalize()),
            metadata: HashMap::new(),
            matched_keywords: Vec::new(),
        }
    }

    #[tokio::test]
    async fn test_keyword_matching() {
        let config = test_config();
        let event_bus = Arc::new(MockEventBus::new());
        let tracker = NewsVelocityTracker::new(config, event_bus);

        let trump_keyword = &tracker.config.tracking_keywords[0];

        // Direct match
        assert!(trump_keyword.matches("Trump is winning"));

        // Alias match
        assert!(trump_keyword.matches("Donald Trump announces"));
        assert!(trump_keyword.matches("POTUS speaks today"));

        // Case insensitive
        assert!(trump_keyword.matches("TRUMP rallies supporters"));

        // No match
        assert!(!trump_keyword.matches("Biden announces policy"));
    }

    #[tokio::test]
    async fn test_velocity_calculation() {
        let config = test_config();
        let event_bus = Arc::new(MockEventBus::new());
        let tracker = NewsVelocityTracker::new(config, event_bus);

        // Add some articles
        for i in 0..5 {
            let article = ArticleTimestamp::new(
                "twitter".to_string(),
                Utc::now() - Duration::minutes(i * 10),
                Some(format!("Trump headline {}", i)),
            );
            tracker.record_article("trump", article).await;
        }

        let metrics = tracker.update_metrics("trump").await;

        assert_eq!(metrics.keyword, "trump");
        assert!(metrics.velocity_1h > Decimal::ZERO);
        assert_eq!(metrics.article_count_1h, 5);
    }

    #[tokio::test]
    async fn test_acceleration_signal() {
        let mut config = test_config();
        config.acceleration_threshold = Decimal::new(15, 1); // 1.5
        config.min_articles_for_signal = 3;

        let event_bus = Arc::new(MockEventBus::new());
        let tracker = NewsVelocityTracker::new(config, event_bus.clone());

        // Add articles to trigger acceleration signal
        // Recent articles (last hour) - high velocity
        for i in 0..5 {
            let article = ArticleTimestamp::new(
                "twitter".to_string(),
                Utc::now() - Duration::minutes(i * 5),
                Some(format!("Breaking Trump news {}", i)),
            );
            tracker.record_article("trump", article).await;
        }

        // Older articles (6-24 hours ago) - lower baseline
        for i in 0..2 {
            let article = ArticleTimestamp::new(
                "twitter".to_string(),
                Utc::now() - Duration::hours(12 + i),
                Some(format!("Old Trump news {}", i)),
            );
            tracker.record_article("trump", article).await;
        }

        let metrics = tracker.update_metrics("trump").await;
        tracker.check_and_emit_signal("trump", &metrics).await;

        // Should emit an acceleration signal
        assert!(event_bus.publish_count() > 0);
    }

    #[tokio::test]
    async fn test_cooldown() {
        let mut config = test_config();
        config.signal_cooldown_secs = 60;

        let event_bus = Arc::new(MockEventBus::new());
        let tracker = NewsVelocityTracker::new(config, event_bus.clone());

        // First signal should work
        tracker.record_signal_time("trump").await;
        assert!(!tracker.check_cooldown("trump").await);

        // Different keyword should not be affected
        assert!(tracker.check_cooldown("bitcoin").await);
    }

    #[tokio::test]
    async fn test_deduplication() {
        let config = test_config();
        let event_bus = Arc::new(MockEventBus::new());
        let tracker = NewsVelocityTracker::new(config, event_bus);

        let article1 = ArticleTimestamp::new(
            "twitter".to_string(),
            Utc::now(),
            Some("Same headline".to_string()),
        );

        let article2 = ArticleTimestamp::new(
            "rss".to_string(),
            Utc::now(),
            Some("Same headline".to_string()),
        );

        assert!(!tracker.is_duplicate(&article1).await);
        assert!(tracker.is_duplicate(&article2).await);
    }

    #[tokio::test]
    async fn test_process_feed_item() {
        let config = test_config();
        let event_bus = Arc::new(MockEventBus::new());
        let tracker = NewsVelocityTracker::new(config, event_bus);

        let item = create_feed_item(
            "Trump announces new policy initiative",
            Some("Trump Policy Update"),
        );

        tracker.process_feed_item(&item).await;

        let count = tracker.get_article_count("trump", 1).await;
        assert_eq!(count, 1);
    }

    #[test]
    fn test_news_category() {
        assert_eq!(NewsCategory::Politics.as_str(), "politics");
        assert_eq!(NewsCategory::Crypto.as_str(), "crypto");
        assert_eq!(NewsCategory::Other.as_str(), "other");
    }

    #[test]
    fn test_velocity_direction() {
        assert_eq!(VelocityDirection::Accelerating.as_str(), "accelerating");
        assert_eq!(VelocityDirection::Decelerating.as_str(), "decelerating");
        assert_eq!(VelocityDirection::Stable.as_str(), "stable");
    }
}
