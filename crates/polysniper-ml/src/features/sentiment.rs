//! Sentiment feature computer
//!
//! Computes features related to market sentiment derived from tags and metadata.

use crate::feature_store::{FeatureComputer, FeatureContext, Result};
use async_trait::async_trait;
use chrono::Duration;

/// Computes sentiment-related features from market metadata
pub struct SentimentFeatureComputer;

impl SentimentFeatureComputer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SentimentFeatureComputer {
    fn default() -> Self {
        Self::new()
    }
}

/// Keywords that might indicate positive sentiment
const POSITIVE_KEYWORDS: &[&str] = &[
    "win", "success", "approve", "pass", "yes", "confirm", "agree",
    "growth", "increase", "rise", "gain", "positive", "up",
];

/// Keywords that might indicate negative sentiment
const NEGATIVE_KEYWORDS: &[&str] = &[
    "fail", "lose", "reject", "deny", "no", "refuse", "decline",
    "drop", "fall", "decrease", "negative", "down", "crash",
];

/// Keywords indicating high uncertainty
const UNCERTAINTY_KEYWORDS: &[&str] = &[
    "may", "might", "could", "possibly", "uncertain", "unclear",
    "depends", "conditional", "if", "whether",
];

#[async_trait]
impl FeatureComputer for SentimentFeatureComputer {
    fn name(&self) -> &str {
        "sentiment"
    }

    fn version(&self) -> &str {
        "1.0"
    }

    async fn compute(&self, context: &FeatureContext) -> Result<serde_json::Value> {
        let market = &context.market;

        // Combine question and description for analysis
        let text = format!(
            "{} {}",
            market.question,
            market.description.as_deref().unwrap_or("")
        )
        .to_lowercase();

        // Count keyword occurrences
        let positive_count = count_keywords(&text, POSITIVE_KEYWORDS);
        let negative_count = count_keywords(&text, NEGATIVE_KEYWORDS);
        let uncertainty_count = count_keywords(&text, UNCERTAINTY_KEYWORDS);

        // Calculate sentiment score (-1.0 to 1.0)
        let total_sentiment_keywords = positive_count + negative_count;
        let sentiment_score = if total_sentiment_keywords > 0 {
            (positive_count as f64 - negative_count as f64) / total_sentiment_keywords as f64
        } else {
            0.0
        };

        // Analyze tags
        let tag_diversity = market.tags.len();
        let has_politics_tag = market.tags.iter().any(|t| t.to_lowercase().contains("politic"));
        let has_sports_tag = market.tags.iter().any(|t| t.to_lowercase().contains("sport"));
        let has_crypto_tag = market.tags.iter().any(|t| {
            let lower = t.to_lowercase();
            lower.contains("crypto") || lower.contains("bitcoin") || lower.contains("eth")
        });

        // Calculate question complexity (word count)
        let word_count = market.question.split_whitespace().count();
        let has_question_mark = market.question.contains('?');

        // Source diversity (based on available metadata)
        let source_diversity = if market.description.is_some() { 2 } else { 1 };

        Ok(serde_json::json!({
            "sentiment_score": sentiment_score,
            "positive_keyword_count": positive_count,
            "negative_keyword_count": negative_count,
            "uncertainty_count": uncertainty_count,
            "tag_diversity": tag_diversity,
            "has_politics_tag": has_politics_tag,
            "has_sports_tag": has_sports_tag,
            "has_crypto_tag": has_crypto_tag,
            "question_word_count": word_count,
            "has_question_mark": has_question_mark,
            "source_diversity": source_diversity,
        }))
    }

    fn dependencies(&self) -> Vec<&str> {
        vec![]
    }

    fn default_ttl(&self) -> Duration {
        Duration::minutes(5) // Sentiment from metadata doesn't change often
    }
}

fn count_keywords(text: &str, keywords: &[&str]) -> usize {
    keywords.iter().filter(|kw| text.contains(*kw)).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use polysniper_core::Market;
    use rust_decimal_macros::dec;

    fn create_test_market(question: &str, description: Option<&str>) -> Market {
        Market {
            condition_id: "test_market".to_string(),
            question: question.to_string(),
            description: description.map(|s| s.to_string()),
            tags: vec!["politics".to_string(), "election".to_string()],
            yes_token_id: "yes_token".to_string(),
            no_token_id: "no_token".to_string(),
            created_at: Utc::now(),
            end_date: None,
            active: true,
            closed: false,
            volume: dec!(1000),
            liquidity: dec!(500),
        }
    }

    #[tokio::test]
    async fn test_sentiment_features_positive() {
        let computer = SentimentFeatureComputer::new();
        let market = create_test_market(
            "Will the candidate win the election?",
            Some("The candidate is expected to succeed with strong growth in polls."),
        );
        let context = FeatureContext::new(market, Utc::now());

        let result = computer.compute(&context).await.unwrap();

        let score = result.get("sentiment_score").unwrap().as_f64().unwrap();
        assert!(score > 0.0, "Expected positive sentiment");
    }

    #[tokio::test]
    async fn test_sentiment_features_negative() {
        let computer = SentimentFeatureComputer::new();
        let market = create_test_market(
            "Will the proposal fail?",
            Some("The proposal is likely to be rejected and decline in support."),
        );
        let context = FeatureContext::new(market, Utc::now());

        let result = computer.compute(&context).await.unwrap();

        let score = result.get("sentiment_score").unwrap().as_f64().unwrap();
        assert!(score < 0.0, "Expected negative sentiment");
    }

    #[tokio::test]
    async fn test_tag_analysis() {
        let computer = SentimentFeatureComputer::new();
        let market = create_test_market("Will this happen?", None);
        let context = FeatureContext::new(market, Utc::now());

        let result = computer.compute(&context).await.unwrap();

        assert!(result.get("has_politics_tag").unwrap().as_bool().unwrap());
        assert_eq!(result.get("tag_diversity").unwrap().as_u64().unwrap(), 2);
    }
}
