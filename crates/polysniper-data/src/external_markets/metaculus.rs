//! Metaculus API Client
//!
//! Client for fetching community predictions from Metaculus.
//! Metaculus is a community of expert forecasters.

use super::{ExternalMarketError, ExternalMarketPrice, Platform};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Configuration for the Metaculus client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaculusConfig {
    /// Whether the Metaculus client is enabled
    pub enabled: bool,
    /// Base URL for the Metaculus API
    #[serde(default = "default_api_base_url")]
    pub api_base_url: String,
    /// How often to poll for updates (seconds)
    #[serde(default = "default_poll_interval")]
    pub poll_interval_secs: u64,
    /// Question IDs to track
    #[serde(default)]
    pub tracked_questions: Vec<String>,
}

fn default_api_base_url() -> String {
    "https://www.metaculus.com/api2".to_string()
}

fn default_poll_interval() -> u64 {
    300 // 5 minutes
}

impl Default for MetaculusConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_base_url: default_api_base_url(),
            poll_interval_secs: default_poll_interval(),
            tracked_questions: Vec::new(),
        }
    }
}

/// Cached question data
#[derive(Debug, Clone)]
struct CachedQuestion {
    prediction: MetaculusPrediction,
    cached_at: DateTime<Utc>,
}

/// Metaculus prediction data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaculusPrediction {
    /// Question ID
    pub question_id: String,
    /// Question title
    pub title: String,
    /// Community prediction (0.0-1.0 probability)
    pub community_prediction: Decimal,
    /// Number of predictions made
    pub prediction_count: u32,
    /// When the question closes for predictions
    pub close_time: Option<DateTime<Utc>>,
    /// Expected resolution date
    pub resolution_date: Option<DateTime<Utc>>,
    /// Question URL
    pub url: Option<String>,
}

impl MetaculusPrediction {
    /// Convert to ExternalMarketPrice
    pub fn to_external_price(&self) -> ExternalMarketPrice {
        let mut price = ExternalMarketPrice::new(
            Platform::Metaculus,
            self.question_id.clone(),
            self.title.clone(),
            self.community_prediction,
        );

        if let Some(close_time) = self.close_time {
            price = price.with_market_close(close_time);
        }

        price
    }
}

/// Raw API response for a question
#[derive(Debug, Deserialize)]
struct QuestionResponse {
    id: i64,
    title: String,
    #[serde(default)]
    community_prediction: Option<CommunityPrediction>,
    #[serde(default)]
    number_of_predictions: Option<u32>,
    #[serde(default)]
    close_time: Option<String>,
    #[serde(default)]
    resolve_time: Option<String>,
    #[serde(default)]
    page_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CommunityPrediction {
    #[serde(default)]
    full: Option<PredictionValue>,
}

#[derive(Debug, Deserialize)]
struct PredictionValue {
    #[serde(default)]
    q2: Option<f64>, // Median prediction
}

/// Search response from Metaculus API
#[derive(Debug, Deserialize)]
struct SearchResponse {
    results: Vec<QuestionResponse>,
}

/// Client for the Metaculus prediction market API
pub struct MetaculusClient {
    http_client: reqwest::Client,
    config: MetaculusConfig,
    cache: Arc<RwLock<HashMap<String, CachedQuestion>>>,
}

impl MetaculusClient {
    /// Create a new Metaculus client
    pub fn new(config: MetaculusConfig) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("Polysniper/1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            http_client,
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if the client is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the poll interval
    pub fn poll_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.config.poll_interval_secs)
    }

    /// Fetch a specific question by ID
    pub async fn get_question(
        &self,
        question_id: &str,
    ) -> Result<MetaculusPrediction, ExternalMarketError> {
        let url = format!("{}/questions/{}/", self.config.api_base_url, question_id);

        debug!(url = %url, "Fetching Metaculus question");

        let response = self.http_client.get(&url).send().await?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(ExternalMarketError::NotFound(format!(
                "Question {} not found",
                question_id
            )));
        }

        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ExternalMarketError::RateLimited {
                retry_after_secs: 60,
            });
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ExternalMarketError::Api(format!(
                "Status {}: {}",
                status, body
            )));
        }

        let question: QuestionResponse = response.json().await.map_err(|e| {
            ExternalMarketError::Parse(format!("Failed to parse question response: {}", e))
        })?;

        let prediction = self.parse_question(question)?;

        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.insert(
                question_id.to_string(),
                CachedQuestion {
                    prediction: prediction.clone(),
                    cached_at: Utc::now(),
                },
            );
        }

        Ok(prediction)
    }

    /// Search for questions matching a query
    pub async fn search_questions(
        &self,
        query: &str,
    ) -> Result<Vec<MetaculusPrediction>, ExternalMarketError> {
        let url = format!(
            "{}/questions/?search={}",
            self.config.api_base_url,
            urlencoding::encode(query)
        );

        debug!(url = %url, "Searching Metaculus questions");

        let response = self.http_client.get(&url).send().await?;

        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ExternalMarketError::RateLimited {
                retry_after_secs: 60,
            });
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ExternalMarketError::Api(format!(
                "Status {}: {}",
                status, body
            )));
        }

        let search_response: SearchResponse = response.json().await.map_err(|e| {
            ExternalMarketError::Parse(format!("Failed to parse search response: {}", e))
        })?;

        let mut predictions = Vec::new();
        for question in search_response.results {
            match self.parse_question(question) {
                Ok(pred) => predictions.push(pred),
                Err(e) => warn!(error = %e, "Failed to parse question in search results"),
            }
        }

        Ok(predictions)
    }

    /// Fetch all tracked questions
    pub async fn fetch_tracked_questions(
        &self,
    ) -> Result<Vec<MetaculusPrediction>, ExternalMarketError> {
        let mut predictions = Vec::new();

        for question_id in &self.config.tracked_questions {
            match self.get_question(question_id).await {
                Ok(pred) => predictions.push(pred),
                Err(e) => warn!(question_id = %question_id, error = %e, "Failed to fetch tracked question"),
            }
        }

        Ok(predictions)
    }

    /// Get cached prediction if available and not stale
    pub async fn get_cached(&self, question_id: &str) -> Option<MetaculusPrediction> {
        let cache = self.cache.read().await;
        if let Some(cached) = cache.get(question_id) {
            let age = Utc::now() - cached.cached_at;
            if age.num_seconds() < self.config.poll_interval_secs as i64 {
                return Some(cached.prediction.clone());
            }
        }
        None
    }

    /// Parse a question response into a MetaculusPrediction
    fn parse_question(
        &self,
        question: QuestionResponse,
    ) -> Result<MetaculusPrediction, ExternalMarketError> {
        let community_prediction = question
            .community_prediction
            .and_then(|cp| cp.full)
            .and_then(|f| f.q2)
            .ok_or_else(|| {
                ExternalMarketError::Parse("No community prediction available".to_string())
            })?;

        let close_time = question
            .close_time
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        let resolution_date = question
            .resolve_time
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        Ok(MetaculusPrediction {
            question_id: question.id.to_string(),
            title: question.title,
            community_prediction: Decimal::from_f64_retain(community_prediction)
                .unwrap_or(Decimal::ZERO),
            prediction_count: question.number_of_predictions.unwrap_or(0),
            close_time,
            resolution_date,
            url: question.page_url,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_default_config() {
        let config = MetaculusConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.api_base_url, "https://www.metaculus.com/api2");
        assert_eq!(config.poll_interval_secs, 300);
        assert!(config.tracked_questions.is_empty());
    }

    #[test]
    fn test_prediction_to_external_price() {
        let prediction = MetaculusPrediction {
            question_id: "12345".to_string(),
            title: "Test Question".to_string(),
            community_prediction: dec!(0.75),
            prediction_count: 100,
            close_time: None,
            resolution_date: None,
            url: Some("https://metaculus.com/q/12345".to_string()),
        };

        let price = prediction.to_external_price();
        assert_eq!(price.platform, Platform::Metaculus);
        assert_eq!(price.question_id, "12345");
        assert_eq!(price.yes_price, dec!(0.75));
    }
}
