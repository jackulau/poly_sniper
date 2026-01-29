//! Sentiment Analyzer
//!
//! Keyword-based sentiment scoring for external signals.

use chrono::Utc;
use polysniper_core::{
    AggregatedSentiment, Confidence, SentimentConfig, SentimentScore, SentimentSignal,
    SentimentSource,
};
use rust_decimal::Decimal;
use std::collections::HashMap;
use tracing::{debug, trace};

/// Sentiment analyzer for processing text content
pub struct SentimentAnalyzer {
    config: SentimentConfig,
}

impl SentimentAnalyzer {
    /// Create a new sentiment analyzer with the given configuration
    pub fn new(config: SentimentConfig) -> Self {
        Self { config }
    }

    /// Analyze text content and return a sentiment signal
    pub fn analyze(&self, content: &str, source: SentimentSource) -> SentimentSignal {
        let content_lower = content.to_lowercase();

        // Calculate positive score
        let (positive_score, positive_keywords) =
            self.calculate_keyword_score(&content_lower, &self.config.positive_keywords);

        // Calculate negative score
        let (negative_score, negative_keywords) =
            self.calculate_keyword_score(&content_lower, &self.config.negative_keywords);

        // Combine scores: positive minus negative
        let raw_score = positive_score - negative_score;

        // Normalize score to [-1, 1] range
        // Use max possible score as denominator (sum of all keyword weights)
        let max_positive: Decimal = self.config.positive_keywords.values().sum();
        let max_negative: Decimal = self.config.negative_keywords.values().sum();
        let max_score = max_positive.max(max_negative).max(Decimal::ONE);

        let normalized_score = raw_score / max_score;
        let sentiment_score = SentimentScore::new(normalized_score);

        // Calculate confidence based on keyword matches and content length
        let total_matches = positive_keywords.len() + negative_keywords.len();
        let confidence = self.calculate_confidence(
            total_matches,
            content.len(),
            positive_score.abs() + negative_score.abs(),
        );

        // Find market keywords
        let market_keywords = self.find_market_keywords(&content_lower);

        let mut matched_keywords = positive_keywords;
        matched_keywords.extend(negative_keywords);

        trace!(
            content_preview = %truncate(content, 100),
            score = %sentiment_score.value(),
            confidence = %confidence.value(),
            matched = ?matched_keywords,
            "Analyzed sentiment"
        );

        SentimentSignal::new(sentiment_score, confidence, source, content.to_string())
            .with_matched_keywords(matched_keywords)
            .with_market_keywords(market_keywords)
    }

    /// Calculate score from keyword matches
    fn calculate_keyword_score(
        &self,
        content: &str,
        keywords: &HashMap<String, Decimal>,
    ) -> (Decimal, Vec<String>) {
        let mut score = Decimal::ZERO;
        let mut matched = Vec::new();

        for (keyword, weight) in keywords {
            let keyword_lower = keyword.to_lowercase();
            // Count occurrences of keyword in content
            let count = content.matches(&keyword_lower).count();
            if count > 0 {
                // Apply diminishing returns for multiple matches
                let match_score = *weight * Decimal::from(count.min(3) as i64);
                score += match_score;
                matched.push(keyword.clone());
            }
        }

        (score, matched)
    }

    /// Calculate confidence based on various factors
    fn calculate_confidence(
        &self,
        keyword_matches: usize,
        content_length: usize,
        total_score: Decimal,
    ) -> Confidence {
        // Base confidence from keyword matches
        let match_factor = match keyword_matches {
            0 => Decimal::ZERO,
            1 => Decimal::new(3, 1),  // 0.3
            2 => Decimal::new(5, 1),  // 0.5
            3 => Decimal::new(7, 1),  // 0.7
            _ => Decimal::new(85, 2), // 0.85 (cap at 4+ matches)
        };

        // Content length factor (longer content = more reliable)
        let length_factor = if content_length < 50 {
            Decimal::new(6, 1) // 0.6 for very short
        } else if content_length < 200 {
            Decimal::new(8, 1) // 0.8 for short
        } else {
            Decimal::ONE // 1.0 for longer content
        };

        // Score magnitude factor (stronger signals = more confident)
        let score_factor = if total_score < Decimal::new(5, 1) {
            Decimal::new(7, 1) // 0.7
        } else {
            Decimal::ONE // 1.0
        };

        let confidence = match_factor * length_factor * score_factor;
        Confidence::new(confidence)
    }

    /// Find market-related keywords in content
    fn find_market_keywords(&self, content: &str) -> Vec<String> {
        let mut found = Vec::new();

        for keyword in self.config.market_keywords.keys() {
            let keyword_lower = keyword.to_lowercase();
            if content.contains(&keyword_lower) {
                found.push(keyword.clone());
            }
        }

        found
    }

    /// Aggregate multiple sentiment signals into a weighted average
    pub fn aggregate(&self, signals: &[SentimentSignal]) -> AggregatedSentiment {
        if signals.is_empty() {
            return AggregatedSentiment::empty();
        }

        let mut total_weight = Decimal::ZERO;
        let mut weighted_score = Decimal::ZERO;
        let mut weighted_confidence = Decimal::ZERO;
        let mut sources = Vec::new();
        let mut all_market_keywords: Vec<String> = Vec::new();

        for signal in signals {
            // Get source weight
            let source_weight = self
                .config
                .source_weights
                .get(signal.source.as_str())
                .copied()
                .unwrap_or(Decimal::ONE);

            // Weight by both source weight and individual confidence
            let weight = source_weight * signal.confidence.value();

            weighted_score += signal.score.value() * weight;
            weighted_confidence += signal.confidence.value() * weight;
            total_weight += weight;

            if !sources.contains(&signal.source) {
                sources.push(signal.source.clone());
            }

            for keyword in &signal.market_keywords {
                if !all_market_keywords.contains(keyword) {
                    all_market_keywords.push(keyword.clone());
                }
            }
        }

        let (final_score, final_confidence) = if total_weight > Decimal::ZERO {
            (
                SentimentScore::new(weighted_score / total_weight),
                Confidence::new(weighted_confidence / total_weight),
            )
        } else {
            (SentimentScore::NEUTRAL, Confidence::new(Decimal::ZERO))
        };

        debug!(
            signal_count = signals.len(),
            score = %final_score.value(),
            confidence = %final_confidence.value(),
            sources = ?sources.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            "Aggregated sentiment"
        );

        AggregatedSentiment {
            score: final_score,
            confidence: final_confidence,
            signal_count: signals.len(),
            sources,
            market_keywords: all_market_keywords,
            timestamp: Utc::now(),
        }
    }

    /// Get market identifiers for a given keyword
    pub fn get_market_identifiers(&self, keyword: &str) -> Option<&Vec<String>> {
        self.config.market_keywords.get(keyword)
    }

    /// Check if a signal meets the minimum thresholds for action
    pub fn is_actionable(&self, signal: &SentimentSignal) -> bool {
        signal.is_actionable(
            self.config.min_sentiment_threshold,
            self.config.min_confidence_threshold,
        )
    }

    /// Check if an aggregated sentiment meets the minimum thresholds
    pub fn is_aggregated_actionable(&self, aggregated: &AggregatedSentiment) -> bool {
        aggregated.score.intensity() >= self.config.min_sentiment_threshold
            && aggregated.confidence.value() >= self.config.min_confidence_threshold
    }

    /// Get the sentiment configuration
    pub fn config(&self) -> &SentimentConfig {
        &self.config
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn test_config() -> SentimentConfig {
        let mut config = SentimentConfig::default();
        // Add some market keywords for testing
        config.market_keywords.insert(
            "trump".to_string(),
            vec!["trump-market-id".to_string()],
        );
        config.market_keywords.insert(
            "bitcoin".to_string(),
            vec!["btc-market-id".to_string()],
        );
        config
    }

    #[test]
    fn test_analyze_positive_sentiment() {
        let analyzer = SentimentAnalyzer::new(test_config());
        let signal = analyzer.analyze(
            "This is very bullish news! Markets will rally and surge higher.",
            SentimentSource::Twitter,
        );

        assert!(signal.score.is_positive());
        assert!(!signal.matched_keywords.is_empty());
        assert!(signal.matched_keywords.contains(&"bullish".to_string()));
        assert!(signal.matched_keywords.contains(&"rally".to_string()));
        assert!(signal.matched_keywords.contains(&"surge".to_string()));
    }

    #[test]
    fn test_analyze_negative_sentiment() {
        let analyzer = SentimentAnalyzer::new(test_config());
        let signal = analyzer.analyze(
            "Bearish outlook. The market will crash and dump. Totally failed.",
            SentimentSource::News,
        );

        assert!(signal.score.is_negative());
        assert!(!signal.matched_keywords.is_empty());
        assert!(signal.matched_keywords.contains(&"bearish".to_string()));
        assert!(signal.matched_keywords.contains(&"crash".to_string()));
        assert!(signal.matched_keywords.contains(&"dump".to_string()));
        assert!(signal.matched_keywords.contains(&"failed".to_string()));
    }

    #[test]
    fn test_analyze_neutral_content() {
        let analyzer = SentimentAnalyzer::new(test_config());
        let signal = analyzer.analyze(
            "The weather is nice today. No market news.",
            SentimentSource::News,
        );

        assert!(signal.score.is_neutral());
        assert!(signal.matched_keywords.is_empty());
    }

    #[test]
    fn test_analyze_mixed_sentiment() {
        let analyzer = SentimentAnalyzer::new(test_config());
        let signal = analyzer.analyze(
            "Bullish on one hand, but crash fears persist. Market is winning but losing momentum.",
            SentimentSource::Twitter,
        );

        // Mixed content should have moderate or near-neutral score
        assert!(signal.score.intensity() < dec!(0.5));
    }

    #[test]
    fn test_market_keyword_detection() {
        let analyzer = SentimentAnalyzer::new(test_config());
        let signal = analyzer.analyze(
            "Trump announces new policy. Very bullish for markets.",
            SentimentSource::News,
        );

        assert!(signal.market_keywords.contains(&"trump".to_string()));
    }

    #[test]
    fn test_aggregate_signals() {
        let analyzer = SentimentAnalyzer::new(test_config());

        let signal1 = SentimentSignal::new(
            SentimentScore::new(dec!(0.8)),
            Confidence::new(dec!(0.7)),
            SentimentSource::Twitter,
            "Bullish news".to_string(),
        )
        .with_market_keywords(vec!["trump".to_string()]);

        let signal2 = SentimentSignal::new(
            SentimentScore::new(dec!(0.6)),
            Confidence::new(dec!(0.8)),
            SentimentSource::News,
            "Positive outlook".to_string(),
        )
        .with_market_keywords(vec!["trump".to_string()]);

        let aggregated = analyzer.aggregate(&[signal1, signal2]);

        assert_eq!(aggregated.signal_count, 2);
        assert!(aggregated.score.is_positive());
        assert!(aggregated.confidence.value() > Decimal::ZERO);
        assert_eq!(aggregated.sources.len(), 2);
        assert!(aggregated.market_keywords.contains(&"trump".to_string()));
    }

    #[test]
    fn test_aggregate_empty() {
        let analyzer = SentimentAnalyzer::new(test_config());
        let aggregated = analyzer.aggregate(&[]);

        assert_eq!(aggregated.signal_count, 0);
        assert!(aggregated.score.is_neutral());
        assert_eq!(aggregated.confidence.value(), Decimal::ZERO);
    }

    #[test]
    fn test_is_actionable() {
        let analyzer = SentimentAnalyzer::new(test_config());

        // Actionable signal
        let actionable = SentimentSignal::new(
            SentimentScore::new(dec!(0.5)),
            Confidence::new(dec!(0.6)),
            SentimentSource::Twitter,
            "Content".to_string(),
        );
        assert!(analyzer.is_actionable(&actionable));

        // Not actionable - low score
        let low_score = SentimentSignal::new(
            SentimentScore::new(dec!(0.1)),
            Confidence::new(dec!(0.6)),
            SentimentSource::Twitter,
            "Content".to_string(),
        );
        assert!(!analyzer.is_actionable(&low_score));

        // Not actionable - low confidence
        let low_confidence = SentimentSignal::new(
            SentimentScore::new(dec!(0.5)),
            Confidence::new(dec!(0.2)),
            SentimentSource::Twitter,
            "Content".to_string(),
        );
        assert!(!analyzer.is_actionable(&low_confidence));
    }

    #[test]
    fn test_source_weights() {
        let mut config = test_config();
        config
            .source_weights
            .insert("twitter".to_string(), dec!(2.0));
        config.source_weights.insert("news".to_string(), dec!(1.0));

        let analyzer = SentimentAnalyzer::new(config);

        // Twitter signal with same score/confidence should have more weight
        let twitter_signal = SentimentSignal::new(
            SentimentScore::new(dec!(0.8)),
            Confidence::new(dec!(0.5)),
            SentimentSource::Twitter,
            "Bullish".to_string(),
        );

        let news_signal = SentimentSignal::new(
            SentimentScore::new(dec!(0.4)),
            Confidence::new(dec!(0.5)),
            SentimentSource::News,
            "Slightly positive".to_string(),
        );

        let aggregated = analyzer.aggregate(&[twitter_signal, news_signal]);

        // Weighted average should be closer to Twitter's score (0.8) due to 2x weight
        assert!(aggregated.score.value() > dec!(0.5));
    }
}
