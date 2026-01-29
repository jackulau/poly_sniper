//! Sentiment Analysis Types
//!
//! Core types for sentiment analysis in the Polysniper trading system.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sentiment score ranging from -1.0 (very negative) to 1.0 (very positive)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SentimentScore(Decimal);

impl SentimentScore {
    /// Create a new sentiment score, clamping to [-1.0, 1.0]
    pub fn new(score: Decimal) -> Self {
        let clamped = if score < Decimal::NEGATIVE_ONE {
            Decimal::NEGATIVE_ONE
        } else if score > Decimal::ONE {
            Decimal::ONE
        } else {
            score
        };
        Self(clamped)
    }

    /// Get the raw score value
    pub fn value(&self) -> Decimal {
        self.0
    }

    /// Check if sentiment is positive (> 0)
    pub fn is_positive(&self) -> bool {
        self.0 > Decimal::ZERO
    }

    /// Check if sentiment is negative (< 0)
    pub fn is_negative(&self) -> bool {
        self.0 < Decimal::ZERO
    }

    /// Check if sentiment is neutral (== 0)
    pub fn is_neutral(&self) -> bool {
        self.0 == Decimal::ZERO
    }

    /// Get absolute value of the score (intensity)
    pub fn intensity(&self) -> Decimal {
        self.0.abs()
    }

    /// Neutral sentiment score
    pub const NEUTRAL: SentimentScore = SentimentScore(Decimal::ZERO);
}

impl Default for SentimentScore {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

impl From<Decimal> for SentimentScore {
    fn from(score: Decimal) -> Self {
        Self::new(score)
    }
}

impl From<f64> for SentimentScore {
    fn from(score: f64) -> Self {
        Self::new(Decimal::try_from(score).unwrap_or(Decimal::ZERO))
    }
}

/// Confidence level for a sentiment signal (0.0 to 1.0)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Confidence(Decimal);

impl Confidence {
    /// Create a new confidence value, clamping to [0.0, 1.0]
    pub fn new(value: Decimal) -> Self {
        let clamped = if value < Decimal::ZERO {
            Decimal::ZERO
        } else if value > Decimal::ONE {
            Decimal::ONE
        } else {
            value
        };
        Self(clamped)
    }

    /// Get the raw confidence value
    pub fn value(&self) -> Decimal {
        self.0
    }

    /// High confidence threshold (0.7)
    pub fn is_high(&self) -> bool {
        self.0 >= Decimal::new(7, 1)
    }

    /// Medium confidence threshold (0.4)
    pub fn is_medium(&self) -> bool {
        self.0 >= Decimal::new(4, 1)
    }

    /// Low confidence
    pub fn is_low(&self) -> bool {
        self.0 < Decimal::new(4, 1)
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self(Decimal::ZERO)
    }
}

impl From<Decimal> for Confidence {
    fn from(value: Decimal) -> Self {
        Self::new(value)
    }
}

impl From<f64> for Confidence {
    fn from(value: f64) -> Self {
        Self::new(Decimal::try_from(value).unwrap_or(Decimal::ZERO))
    }
}

/// Source type for sentiment signals
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SentimentSource {
    /// Twitter/X feed
    Twitter,
    /// RSS news feed
    News,
    /// Reddit
    Reddit,
    /// Custom/other source
    Custom(String),
}

impl SentimentSource {
    /// Get a string representation of the source
    pub fn as_str(&self) -> &str {
        match self {
            SentimentSource::Twitter => "twitter",
            SentimentSource::News => "news",
            SentimentSource::Reddit => "reddit",
            SentimentSource::Custom(name) => name.as_str(),
        }
    }
}

impl Default for SentimentSource {
    fn default() -> Self {
        SentimentSource::Custom("unknown".to_string())
    }
}

/// A sentiment signal combining score, confidence, and source information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentSignal {
    /// The calculated sentiment score
    pub score: SentimentScore,
    /// Confidence in the score
    pub confidence: Confidence,
    /// Source of the signal
    pub source: SentimentSource,
    /// Original content that was analyzed
    pub content: String,
    /// Keywords matched in the content
    pub matched_keywords: Vec<String>,
    /// Market keywords matched (for market mapping)
    pub market_keywords: Vec<String>,
    /// When the signal was created
    pub timestamp: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl SentimentSignal {
    /// Create a new sentiment signal
    pub fn new(
        score: SentimentScore,
        confidence: Confidence,
        source: SentimentSource,
        content: String,
    ) -> Self {
        Self {
            score,
            confidence,
            source,
            content,
            matched_keywords: Vec::new(),
            market_keywords: Vec::new(),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add matched keywords
    pub fn with_matched_keywords(mut self, keywords: Vec<String>) -> Self {
        self.matched_keywords = keywords;
        self
    }

    /// Add market keywords
    pub fn with_market_keywords(mut self, keywords: Vec<String>) -> Self {
        self.market_keywords = keywords;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Check if this signal meets minimum thresholds for trading
    pub fn is_actionable(&self, min_score: Decimal, min_confidence: Decimal) -> bool {
        self.score.intensity() >= min_score && self.confidence.value() >= min_confidence
    }
}

/// Configuration for sentiment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentConfig {
    /// Enabled flag
    pub enabled: bool,
    /// Positive keywords with optional weights
    pub positive_keywords: HashMap<String, Decimal>,
    /// Negative keywords with optional weights
    pub negative_keywords: HashMap<String, Decimal>,
    /// Market keyword mappings (keyword -> market identifiers)
    pub market_keywords: HashMap<String, Vec<String>>,
    /// Source weights for aggregation (source -> weight)
    pub source_weights: HashMap<String, Decimal>,
    /// Minimum sentiment score to generate signal
    pub min_sentiment_threshold: Decimal,
    /// Minimum confidence to generate signal
    pub min_confidence_threshold: Decimal,
    /// Cooldown between signals for same market (seconds)
    pub signal_cooldown_secs: u64,
    /// Default order size in USD for sentiment signals
    pub default_order_size_usd: Decimal,
    /// Maximum entry price for sentiment-based buys
    pub max_entry_price: Decimal,
}

impl Default for SentimentConfig {
    fn default() -> Self {
        let mut positive_keywords = HashMap::new();
        positive_keywords.insert("bullish".to_string(), Decimal::ONE);
        positive_keywords.insert("positive".to_string(), Decimal::new(8, 1));
        positive_keywords.insert("surge".to_string(), Decimal::new(9, 1));
        positive_keywords.insert("rally".to_string(), Decimal::new(9, 1));
        positive_keywords.insert("win".to_string(), Decimal::new(7, 1));
        positive_keywords.insert("winning".to_string(), Decimal::new(7, 1));
        positive_keywords.insert("confirmed".to_string(), Decimal::new(8, 1));
        positive_keywords.insert("approved".to_string(), Decimal::new(8, 1));
        positive_keywords.insert("success".to_string(), Decimal::new(8, 1));

        let mut negative_keywords = HashMap::new();
        negative_keywords.insert("bearish".to_string(), Decimal::ONE);
        negative_keywords.insert("negative".to_string(), Decimal::new(8, 1));
        negative_keywords.insert("crash".to_string(), Decimal::new(9, 1));
        negative_keywords.insert("dump".to_string(), Decimal::new(9, 1));
        negative_keywords.insert("lose".to_string(), Decimal::new(7, 1));
        negative_keywords.insert("losing".to_string(), Decimal::new(7, 1));
        negative_keywords.insert("rejected".to_string(), Decimal::new(8, 1));
        negative_keywords.insert("denied".to_string(), Decimal::new(8, 1));
        negative_keywords.insert("failed".to_string(), Decimal::new(8, 1));

        let mut source_weights = HashMap::new();
        source_weights.insert("twitter".to_string(), Decimal::new(12, 1)); // 1.2x for Twitter
        source_weights.insert("news".to_string(), Decimal::ONE); // 1.0x for news
        source_weights.insert("reddit".to_string(), Decimal::new(8, 1)); // 0.8x for Reddit

        Self {
            enabled: true,
            positive_keywords,
            negative_keywords,
            market_keywords: HashMap::new(),
            source_weights,
            min_sentiment_threshold: Decimal::new(3, 1), // 0.3
            min_confidence_threshold: Decimal::new(5, 1), // 0.5
            signal_cooldown_secs: 300,                    // 5 minutes
            default_order_size_usd: Decimal::new(50, 0),  // $50
            max_entry_price: Decimal::new(85, 2),         // 0.85
        }
    }
}

/// Result of aggregating multiple sentiment signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedSentiment {
    /// Weighted average sentiment score
    pub score: SentimentScore,
    /// Combined confidence (based on number of sources and their weights)
    pub confidence: Confidence,
    /// Number of signals aggregated
    pub signal_count: usize,
    /// Sources that contributed
    pub sources: Vec<SentimentSource>,
    /// All matched market keywords
    pub market_keywords: Vec<String>,
    /// When the aggregation was performed
    pub timestamp: DateTime<Utc>,
}

impl AggregatedSentiment {
    /// Create an empty aggregation result
    pub fn empty() -> Self {
        Self {
            score: SentimentScore::NEUTRAL,
            confidence: Confidence::new(Decimal::ZERO),
            signal_count: 0,
            sources: Vec::new(),
            market_keywords: Vec::new(),
            timestamp: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_sentiment_score_clamping() {
        let score = SentimentScore::new(dec!(1.5));
        assert_eq!(score.value(), Decimal::ONE);

        let score = SentimentScore::new(dec!(-1.5));
        assert_eq!(score.value(), Decimal::NEGATIVE_ONE);

        let score = SentimentScore::new(dec!(0.5));
        assert_eq!(score.value(), dec!(0.5));
    }

    #[test]
    fn test_sentiment_score_classification() {
        let positive = SentimentScore::new(dec!(0.5));
        assert!(positive.is_positive());
        assert!(!positive.is_negative());
        assert!(!positive.is_neutral());

        let negative = SentimentScore::new(dec!(-0.5));
        assert!(!negative.is_positive());
        assert!(negative.is_negative());
        assert!(!negative.is_neutral());

        let neutral = SentimentScore::NEUTRAL;
        assert!(!neutral.is_positive());
        assert!(!neutral.is_negative());
        assert!(neutral.is_neutral());
    }

    #[test]
    fn test_sentiment_score_intensity() {
        let positive = SentimentScore::new(dec!(0.7));
        assert_eq!(positive.intensity(), dec!(0.7));

        let negative = SentimentScore::new(dec!(-0.8));
        assert_eq!(negative.intensity(), dec!(0.8));
    }

    #[test]
    fn test_confidence_clamping() {
        let confidence = Confidence::new(dec!(1.5));
        assert_eq!(confidence.value(), Decimal::ONE);

        let confidence = Confidence::new(dec!(-0.5));
        assert_eq!(confidence.value(), Decimal::ZERO);
    }

    #[test]
    fn test_confidence_levels() {
        let high = Confidence::new(dec!(0.8));
        assert!(high.is_high());
        assert!(high.is_medium());
        assert!(!high.is_low());

        let medium = Confidence::new(dec!(0.5));
        assert!(!medium.is_high());
        assert!(medium.is_medium());
        assert!(!medium.is_low());

        let low = Confidence::new(dec!(0.2));
        assert!(!low.is_high());
        assert!(!low.is_medium());
        assert!(low.is_low());
    }

    #[test]
    fn test_sentiment_signal_actionable() {
        let signal = SentimentSignal::new(
            SentimentScore::new(dec!(0.6)),
            Confidence::new(dec!(0.7)),
            SentimentSource::Twitter,
            "Test content".to_string(),
        );

        assert!(signal.is_actionable(dec!(0.3), dec!(0.5)));
        assert!(!signal.is_actionable(dec!(0.8), dec!(0.5)));
        assert!(!signal.is_actionable(dec!(0.3), dec!(0.9)));
    }

    #[test]
    fn test_sentiment_source() {
        assert_eq!(SentimentSource::Twitter.as_str(), "twitter");
        assert_eq!(SentimentSource::News.as_str(), "news");
        assert_eq!(SentimentSource::Reddit.as_str(), "reddit");
        assert_eq!(
            SentimentSource::Custom("telegram".to_string()).as_str(),
            "telegram"
        );
    }
}
