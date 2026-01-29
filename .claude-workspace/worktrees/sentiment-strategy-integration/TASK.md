---
id: sentiment-strategy-integration
name: Sentiment Strategy Integration - Event-Based Trading on Sentiment
wave: 2
priority: 2
dependencies: [sentiment-feed-ingestion]
estimated_hours: 5
tags: [backend, strategy, sentiment]
---

## Objective

Create a sentiment analysis layer that processes feed items, scores sentiment, and generates ExternalSignal events for the EventBasedStrategy to act upon.

## Context

Once sentiment-feed-ingestion provides raw feed data, this task adds sentiment scoring and converts high-confidence sentiment signals into trading events. Uses keyword-based sentiment analysis (no external ML required).

## Implementation

1. Create `/crates/polysniper-strategies/src/sentiment_analyzer.rs`:
   - Keyword-based sentiment scoring
   - Configurable positive/negative word lists
   - Market keyword mapping (e.g., "Trump" -> Trump markets)
   - Aggregate sentiment from multiple sources

2. Create `/crates/polysniper-strategies/src/sentiment_strategy.rs`:
   - New strategy implementing `Strategy` trait
   - Consumes `FeedItemReceived` events
   - Applies sentiment analysis
   - Generates `ExternalSignal` events with sentiment metadata

3. Create `/crates/polysniper-core/src/sentiment.rs`:
   - `SentimentScore` (-1.0 to 1.0)
   - `SentimentConfig` with thresholds and keywords
   - `SentimentSignal` combining score + confidence + source

4. Add sentiment strategy configuration:
   - Positive/negative keyword dictionaries
   - Market keyword mappings
   - Minimum sentiment threshold for signals
   - Source weighting (Twitter vs News)

## Acceptance Criteria

- [ ] Processes feed items and calculates sentiment scores
- [ ] Maps feed content to relevant markets via keywords
- [ ] Generates ExternalSignal with sentiment metadata
- [ ] Configurable sentiment thresholds trigger trading
- [ ] Multiple sources aggregated (weighted average)
- [ ] Prevents duplicate signals for same event
- [ ] Source-specific sentiment weighting
- [ ] Unit tests for sentiment calculations
- [ ] Integration with EventBasedStrategy rule matching

## Files to Create/Modify

- `crates/polysniper-strategies/src/sentiment_analyzer.rs` - **CREATE** - Sentiment scoring
- `crates/polysniper-strategies/src/sentiment_strategy.rs` - **CREATE** - Sentiment strategy
- `crates/polysniper-strategies/src/lib.rs` - **MODIFY** - Export sentiment modules
- `crates/polysniper-core/src/sentiment.rs` - **CREATE** - Sentiment types
- `crates/polysniper-core/src/lib.rs` - **MODIFY** - Export sentiment module
- `config/strategies/sentiment.toml` - **CREATE** - Sentiment strategy config

## Integration Points

- **Provides**: Sentiment-based trading signals
- **Consumes**: FeedItemReceived events from sentiment-feed-ingestion
- **Conflicts**: None - new strategy implementation
