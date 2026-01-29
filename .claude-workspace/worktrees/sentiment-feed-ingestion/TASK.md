---
id: sentiment-feed-ingestion
name: Sentiment Feed Ingestion - Twitter/News Feed Polling
wave: 1
priority: 1
dependencies: []
estimated_hours: 5
tags: [backend, data-source, sentiment]
---

## Objective

Create data source modules to poll Twitter API and RSS/news feeds, extract content, and publish raw feed data as events for sentiment analysis.

## Context

The codebase defines `SignalSource::Twitter` and `SignalSource::Rss` variants but lacks the actual polling implementations. This task adds feed ingestion that works alongside the existing `GammaClient` polling pattern.

## Implementation

1. Create `/crates/polysniper-data/src/twitter_client.rs`:
   - Twitter API v2 client using reqwest
   - Search tweets by keywords/accounts
   - Rate limiting compliance (15 requests/15 min for search)
   - Convert tweets to `FeedItem` struct

2. Create `/crates/polysniper-data/src/rss_client.rs`:
   - RSS/Atom feed parser
   - Polling interval configuration
   - Support multiple feed URLs
   - Convert entries to `FeedItem` struct

3. Create `/crates/polysniper-data/src/feed_types.rs`:
   - `FeedItem` struct (source, content, timestamp, url, metadata)
   - `FeedConfig` for polling settings
   - Feed-specific error types

4. Create `/crates/polysniper-data/src/feed_aggregator.rs`:
   - Manages multiple feed sources
   - Deduplication using content hash
   - Publishes raw `FeedItemReceived` events

5. Add feed configuration to config:
   - Twitter API credentials (env vars)
   - RSS feed URLs list
   - Keywords to track
   - Polling intervals

## Acceptance Criteria

- [ ] Twitter client polls search API for configured keywords
- [ ] RSS client parses Atom and RSS 2.0 feeds
- [ ] Feed items deduplicated by content hash
- [ ] Items published as events for downstream processing
- [ ] Configurable polling intervals per source
- [ ] Respects Twitter API rate limits
- [ ] Graceful handling of feed errors (continue on failure)
- [ ] Supports multiple concurrent feed sources
- [ ] Unit tests for feed parsing

## Files to Create/Modify

- `crates/polysniper-data/src/twitter_client.rs` - **CREATE** - Twitter API client
- `crates/polysniper-data/src/rss_client.rs` - **CREATE** - RSS feed client
- `crates/polysniper-data/src/feed_types.rs` - **CREATE** - Feed data types
- `crates/polysniper-data/src/feed_aggregator.rs` - **CREATE** - Feed management
- `crates/polysniper-data/src/lib.rs` - **MODIFY** - Export feed modules
- `crates/polysniper-data/Cargo.toml` - **MODIFY** - Add feed-rss dependency
- `crates/polysniper-core/src/events.rs` - **MODIFY** - Add FeedItemReceived event
- `config/default.toml` - **MODIFY** - Add feeds configuration

## Integration Points

- **Provides**: Raw feed data events for sentiment analysis
- **Consumes**: EventBus for publishing, configuration
- **Conflicts**: May touch events.rs (coordinate with other tasks)
