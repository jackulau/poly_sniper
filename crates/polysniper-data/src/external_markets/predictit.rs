//! PredictIt API Client
//!
//! Client for fetching market prices from PredictIt.
//! PredictIt is a US-based prediction market with capped contracts.

use super::{ExternalMarketError, ExternalMarketPrice, Platform};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Configuration for the PredictIt client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictItConfig {
    /// Whether the PredictIt client is enabled
    pub enabled: bool,
    /// Base URL for the PredictIt API
    #[serde(default = "default_api_base_url")]
    pub api_base_url: String,
    /// How often to poll for updates (seconds)
    #[serde(default = "default_poll_interval")]
    pub poll_interval_secs: u64,
    /// Market IDs to track
    #[serde(default)]
    pub tracked_markets: Vec<i32>,
}

fn default_api_base_url() -> String {
    "https://www.predictit.org/api".to_string()
}

fn default_poll_interval() -> u64 {
    60 // 1 minute
}

impl Default for PredictItConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_base_url: default_api_base_url(),
            poll_interval_secs: default_poll_interval(),
            tracked_markets: Vec::new(),
        }
    }
}

/// Cached contract data
#[derive(Debug, Clone)]
struct CachedContract {
    market: PredictItMarket,
    cached_at: DateTime<Utc>,
}

/// PredictIt market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictItMarket {
    /// Market ID
    pub market_id: i32,
    /// Market name/question
    pub name: String,
    /// Short name
    pub short_name: Option<String>,
    /// Market URL
    pub url: Option<String>,
    /// Contracts in this market
    pub contracts: Vec<PredictItContract>,
    /// Market status (Open, Closed)
    pub status: String,
    /// When the market was last updated
    pub timestamp: DateTime<Utc>,
}

impl PredictItMarket {
    /// Get the primary contract (first one, usually the main question)
    pub fn primary_contract(&self) -> Option<&PredictItContract> {
        self.contracts.first()
    }

    /// Convert primary contract to ExternalMarketPrice
    pub fn to_external_price(&self) -> Option<ExternalMarketPrice> {
        let contract = self.primary_contract()?;

        let yes_price = contract.last_trade_price;
        let no_price = Decimal::ONE - yes_price;

        let mut price = ExternalMarketPrice::new(
            Platform::PredictIt,
            contract.contract_id.to_string(),
            format!("{} - {}", self.name, contract.name),
            yes_price,
        )
        .with_no_price(no_price)
        .with_last_updated(self.timestamp);

        if let Some(close) = contract.date_end {
            price = price.with_market_close(close);
        }

        Some(price)
    }
}

/// PredictIt contract (individual outcome within a market)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictItContract {
    /// Contract ID
    pub contract_id: i64,
    /// Contract name (outcome description)
    pub name: String,
    /// Short name
    pub short_name: Option<String>,
    /// Best buy price for YES
    pub best_buy_yes: Option<Decimal>,
    /// Best sell price for YES
    pub best_sell_yes: Option<Decimal>,
    /// Best buy price for NO
    pub best_buy_no: Option<Decimal>,
    /// Best sell price for NO
    pub best_sell_no: Option<Decimal>,
    /// Last trade price
    pub last_trade_price: Decimal,
    /// Last close price
    pub last_close_price: Decimal,
    /// Contract end date
    pub date_end: Option<DateTime<Utc>>,
}

impl PredictItContract {
    /// Get the mid price for YES
    pub fn yes_mid_price(&self) -> Option<Decimal> {
        match (self.best_buy_yes, self.best_sell_yes) {
            (Some(buy), Some(sell)) => Some((buy + sell) / Decimal::TWO),
            _ => None,
        }
    }

    /// Get the spread
    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_buy_yes, self.best_sell_yes) {
            (Some(buy), Some(sell)) => Some(sell - buy),
            _ => None,
        }
    }

    /// Convert to ExternalMarketPrice
    pub fn to_external_price(&self, market_name: &str) -> ExternalMarketPrice {
        let yes_price = self.yes_mid_price().unwrap_or(self.last_trade_price);
        let no_price = Decimal::ONE - yes_price;

        let mut price = ExternalMarketPrice::new(
            Platform::PredictIt,
            self.contract_id.to_string(),
            format!("{} - {}", market_name, self.name),
            yes_price,
        )
        .with_no_price(no_price);

        if let Some(close) = self.date_end {
            price = price.with_market_close(close);
        }

        price
    }
}

/// Raw API response for all markets
#[derive(Debug, Deserialize)]
struct AllMarketsResponse {
    markets: Vec<RawMarket>,
}

/// Raw API response for a single market
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawMarket {
    id: i32,
    name: String,
    short_name: Option<String>,
    url: Option<String>,
    contracts: Vec<RawContract>,
    status: String,
    timestamp: Option<String>,
}

/// Raw contract from API
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawContract {
    id: i64,
    name: String,
    short_name: Option<String>,
    best_buy_yes_cost: Option<f64>,
    best_sell_yes_cost: Option<f64>,
    best_buy_no_cost: Option<f64>,
    best_sell_no_cost: Option<f64>,
    last_trade_price: f64,
    last_close_price: f64,
    date_end: Option<String>,
}

/// Client for the PredictIt prediction market API
pub struct PredictItClient {
    http_client: reqwest::Client,
    config: PredictItConfig,
    cache: Arc<RwLock<HashMap<i32, CachedContract>>>,
}

impl PredictItClient {
    /// Create a new PredictIt client
    pub fn new(config: PredictItConfig) -> Self {
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

    /// Fetch all markets
    pub async fn get_all_markets(&self) -> Result<Vec<PredictItMarket>, ExternalMarketError> {
        let url = format!("{}/marketdata/all/", self.config.api_base_url);

        debug!(url = %url, "Fetching all PredictIt markets");

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

        let all_markets: AllMarketsResponse = response.json().await.map_err(|e| {
            ExternalMarketError::Parse(format!("Failed to parse markets response: {}", e))
        })?;

        let markets: Vec<PredictItMarket> = all_markets
            .markets
            .into_iter()
            .filter_map(|m| self.parse_market(m).ok())
            .collect();

        // Update cache
        {
            let mut cache = self.cache.write().await;
            for market in &markets {
                cache.insert(
                    market.market_id,
                    CachedContract {
                        market: market.clone(),
                        cached_at: Utc::now(),
                    },
                );
            }
        }

        Ok(markets)
    }

    /// Fetch a specific market by ID
    pub async fn get_market(&self, market_id: i32) -> Result<PredictItMarket, ExternalMarketError> {
        let url = format!("{}/marketdata/markets/{}", self.config.api_base_url, market_id);

        debug!(url = %url, "Fetching PredictIt market");

        let response = self.http_client.get(&url).send().await?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(ExternalMarketError::NotFound(format!(
                "Market {} not found",
                market_id
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

        let raw_market: RawMarket = response.json().await.map_err(|e| {
            ExternalMarketError::Parse(format!("Failed to parse market response: {}", e))
        })?;

        let market = self.parse_market(raw_market)?;

        // Update cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(
                market_id,
                CachedContract {
                    market: market.clone(),
                    cached_at: Utc::now(),
                },
            );
        }

        Ok(market)
    }

    /// Fetch all tracked markets
    pub async fn fetch_tracked_markets(&self) -> Result<Vec<PredictItMarket>, ExternalMarketError> {
        let mut markets = Vec::new();

        for market_id in &self.config.tracked_markets {
            match self.get_market(*market_id).await {
                Ok(market) => markets.push(market),
                Err(e) => warn!(market_id = %market_id, error = %e, "Failed to fetch tracked market"),
            }
        }

        Ok(markets)
    }

    /// Get cached market if available and not stale
    pub async fn get_cached(&self, market_id: i32) -> Option<PredictItMarket> {
        let cache = self.cache.read().await;
        if let Some(cached) = cache.get(&market_id) {
            let age = Utc::now() - cached.cached_at;
            if age.num_seconds() < self.config.poll_interval_secs as i64 {
                return Some(cached.market.clone());
            }
        }
        None
    }

    /// Parse a raw market into PredictItMarket
    fn parse_market(&self, raw: RawMarket) -> Result<PredictItMarket, ExternalMarketError> {
        let timestamp = raw
            .timestamp
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        let contracts: Vec<PredictItContract> = raw
            .contracts
            .into_iter()
            .map(|c| self.parse_contract(c))
            .collect();

        Ok(PredictItMarket {
            market_id: raw.id,
            name: raw.name,
            short_name: raw.short_name,
            url: raw.url,
            contracts,
            status: raw.status,
            timestamp,
        })
    }

    /// Parse a raw contract
    fn parse_contract(&self, raw: RawContract) -> PredictItContract {
        let date_end = raw
            .date_end
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        PredictItContract {
            contract_id: raw.id,
            name: raw.name,
            short_name: raw.short_name,
            best_buy_yes: raw
                .best_buy_yes_cost
                .and_then(Decimal::from_f64_retain),
            best_sell_yes: raw
                .best_sell_yes_cost
                .and_then(Decimal::from_f64_retain),
            best_buy_no: raw.best_buy_no_cost.and_then(Decimal::from_f64_retain),
            best_sell_no: raw
                .best_sell_no_cost
                .and_then(Decimal::from_f64_retain),
            last_trade_price: Decimal::from_f64_retain(raw.last_trade_price)
                .unwrap_or(Decimal::ZERO),
            last_close_price: Decimal::from_f64_retain(raw.last_close_price)
                .unwrap_or(Decimal::ZERO),
            date_end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_default_config() {
        let config = PredictItConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.api_base_url, "https://www.predictit.org/api");
        assert_eq!(config.poll_interval_secs, 60);
        assert!(config.tracked_markets.is_empty());
    }

    #[test]
    fn test_contract_mid_price() {
        let contract = PredictItContract {
            contract_id: 12345,
            name: "Test Contract".to_string(),
            short_name: None,
            best_buy_yes: Some(dec!(0.50)),
            best_sell_yes: Some(dec!(0.55)),
            best_buy_no: None,
            best_sell_no: None,
            last_trade_price: dec!(0.52),
            last_close_price: dec!(0.51),
            date_end: None,
        };

        assert_eq!(contract.yes_mid_price(), Some(dec!(0.525)));
        assert_eq!(contract.spread(), Some(dec!(0.05)));
    }

    #[test]
    fn test_contract_to_external_price() {
        let contract = PredictItContract {
            contract_id: 12345,
            name: "Test Contract".to_string(),
            short_name: None,
            best_buy_yes: Some(dec!(0.60)),
            best_sell_yes: Some(dec!(0.65)),
            best_buy_no: None,
            best_sell_no: None,
            last_trade_price: dec!(0.62),
            last_close_price: dec!(0.61),
            date_end: None,
        };

        let price = contract.to_external_price("Test Market");
        assert_eq!(price.platform, Platform::PredictIt);
        assert_eq!(price.yes_price, dec!(0.625)); // mid price
        assert_eq!(price.no_price, Some(dec!(0.375)));
    }
}
