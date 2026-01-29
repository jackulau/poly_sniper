//! Gamma API client for market discovery

use chrono::{DateTime, Utc};
use polysniper_core::{DataSourceError, Market, MarketId, MarketPoller};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::Deserialize;
use std::time::Duration;
use tracing::{debug, info};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_LIMIT: u32 = 100;

/// Gamma API response for markets
#[derive(Debug, Clone, Deserialize)]
struct GammaMarketsResponse {
    #[serde(default)]
    data: Vec<GammaMarket>,
    #[serde(default)]
    next_cursor: Option<String>,
}

/// Gamma market data
#[derive(Debug, Clone, Deserialize)]
struct GammaMarket {
    condition_id: String,
    question: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    #[allow(dead_code)]
    outcomes: Vec<String>,
    #[serde(default)]
    tokens: Vec<GammaToken>,
    #[serde(default)]
    active: bool,
    #[serde(default)]
    closed: bool,
    #[serde(default)]
    volume: String,
    #[serde(default)]
    liquidity: String,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    end_date_iso: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct GammaToken {
    token_id: String,
    outcome: String,
}

/// Gamma API client
pub struct GammaClient {
    base_url: String,
    client: Client,
}

impl GammaClient {
    /// Create a new Gamma client
    pub fn new(base_url: String) -> Self {
        let client = Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .expect("Failed to create HTTP client");

        Self { base_url, client }
    }

    /// Fetch active markets with pagination
    pub async fn fetch_markets(
        &self,
        limit: Option<u32>,
        cursor: Option<&str>,
    ) -> Result<(Vec<Market>, Option<String>), DataSourceError> {
        let limit = limit.unwrap_or(DEFAULT_LIMIT);
        let mut url = format!("{}/markets?limit={}&active=true", self.base_url, limit);

        if let Some(c) = cursor {
            url.push_str(&format!("&cursor={}", c));
        }

        debug!("Fetching markets from Gamma: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| DataSourceError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(DataSourceError::HttpError(format!(
                "Gamma API returned status {}",
                response.status()
            )));
        }

        let gamma_response: GammaMarketsResponse = response
            .json()
            .await
            .map_err(|e| DataSourceError::ParseError(e.to_string()))?;

        let markets: Vec<Market> = gamma_response
            .data
            .into_iter()
            .filter_map(|gm| self.convert_market(gm))
            .collect();

        info!("Fetched {} markets from Gamma", markets.len());

        Ok((markets, gamma_response.next_cursor))
    }

    /// Fetch a single market by condition ID
    pub async fn fetch_market(&self, condition_id: &str) -> Result<Option<Market>, DataSourceError> {
        let url = format!("{}/markets/{}", self.base_url, condition_id);

        debug!("Fetching market from Gamma: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| DataSourceError::HttpError(e.to_string()))?;

        if response.status().as_u16() == 404 {
            return Ok(None);
        }

        if !response.status().is_success() {
            return Err(DataSourceError::HttpError(format!(
                "Gamma API returned status {}",
                response.status()
            )));
        }

        let gamma_market: GammaMarket = response
            .json()
            .await
            .map_err(|e| DataSourceError::ParseError(e.to_string()))?;

        Ok(self.convert_market(gamma_market))
    }

    /// Search markets by keyword
    pub async fn search(&self, query: &str) -> Result<Vec<Market>, DataSourceError> {
        let url = format!(
            "{}/markets?search={}&active=true&limit=50",
            self.base_url,
            urlencoding::encode(query)
        );

        debug!("Searching markets on Gamma: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| DataSourceError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(DataSourceError::HttpError(format!(
                "Gamma API returned status {}",
                response.status()
            )));
        }

        let gamma_response: GammaMarketsResponse = response
            .json()
            .await
            .map_err(|e| DataSourceError::ParseError(e.to_string()))?;

        let markets: Vec<Market> = gamma_response
            .data
            .into_iter()
            .filter_map(|gm| self.convert_market(gm))
            .collect();

        info!("Found {} markets for query '{}'", markets.len(), query);

        Ok(markets)
    }

    fn convert_market(&self, gm: GammaMarket) -> Option<Market> {
        let (yes_token_id, no_token_id) = self.extract_token_ids(&gm.tokens)?;

        let created_at = gm
            .created_at
            .as_ref()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        let end_date = gm
            .end_date_iso
            .as_ref()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        let volume = gm.volume.parse::<Decimal>().unwrap_or(Decimal::ZERO);
        let liquidity = gm.liquidity.parse::<Decimal>().unwrap_or(Decimal::ZERO);

        Some(Market {
            condition_id: gm.condition_id,
            question: gm.question,
            description: gm.description,
            tags: gm.tags,
            yes_token_id,
            no_token_id,
            created_at,
            end_date,
            active: gm.active,
            closed: gm.closed,
            volume,
            liquidity,
        })
    }

    fn extract_token_ids(&self, tokens: &[GammaToken]) -> Option<(String, String)> {
        let yes_token = tokens
            .iter()
            .find(|t| t.outcome.eq_ignore_ascii_case("yes"))?;
        let no_token = tokens
            .iter()
            .find(|t| t.outcome.eq_ignore_ascii_case("no"))?;

        Some((yes_token.token_id.clone(), no_token.token_id.clone()))
    }
}

#[async_trait::async_trait]
impl MarketPoller for GammaClient {
    async fn poll_markets(&self) -> Result<Vec<Market>, DataSourceError> {
        let (markets, _) = self.fetch_markets(None, None).await?;
        Ok(markets)
    }

    async fn poll_market(&self, market_id: &MarketId) -> Result<Option<Market>, DataSourceError> {
        self.fetch_market(market_id).await
    }

    async fn search_markets(&self, query: &str) -> Result<Vec<Market>, DataSourceError> {
        self.search(query).await
    }
}
