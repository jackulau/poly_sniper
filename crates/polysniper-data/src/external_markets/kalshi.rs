//! Kalshi API Client
//!
//! Client for fetching market prices from Kalshi.
//! Kalshi is a regulated US prediction market exchange.

use super::{ExternalMarketError, ExternalMarketPrice, Platform};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Configuration for the Kalshi client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalshiConfig {
    /// Whether the Kalshi client is enabled
    pub enabled: bool,
    /// Base URL for the Kalshi API
    #[serde(default = "default_api_base_url")]
    pub api_base_url: String,
    /// Email for authentication (optional - some endpoints are public)
    pub email: Option<String>,
    /// Environment variable name containing the password
    #[serde(default = "default_password_env")]
    pub password_env: String,
    /// How often to poll for updates (seconds)
    #[serde(default = "default_poll_interval")]
    pub poll_interval_secs: u64,
    /// Series tickers to track
    #[serde(default)]
    pub tracked_series: Vec<String>,
    /// Specific market tickers to track
    #[serde(default)]
    pub tracked_markets: Vec<String>,
}

fn default_api_base_url() -> String {
    "https://trading-api.kalshi.com/trade-api/v2".to_string()
}

fn default_password_env() -> String {
    "KALSHI_PASSWORD".to_string()
}

fn default_poll_interval() -> u64 {
    30 // 30 seconds
}

impl Default for KalshiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_base_url: default_api_base_url(),
            email: None,
            password_env: default_password_env(),
            poll_interval_secs: default_poll_interval(),
            tracked_series: Vec::new(),
            tracked_markets: Vec::new(),
        }
    }
}

/// Cached event data
#[derive(Debug, Clone)]
struct CachedEvent {
    event: KalshiEvent,
    cached_at: DateTime<Utc>,
}

/// Kalshi event (contains multiple markets)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalshiEvent {
    /// Event ticker (unique identifier)
    pub event_ticker: String,
    /// Series ticker this event belongs to
    pub series_ticker: String,
    /// Event title
    pub title: String,
    /// Category
    pub category: String,
    /// Sub-title (more specific description)
    pub sub_title: Option<String>,
    /// Markets within this event
    pub markets: Vec<KalshiMarket>,
    /// Event status
    pub status: String,
}

impl KalshiEvent {
    /// Get the primary market (first one)
    pub fn primary_market(&self) -> Option<&KalshiMarket> {
        self.markets.first()
    }
}

/// Kalshi market (individual contract)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalshiMarket {
    /// Market ticker (unique identifier)
    pub ticker: String,
    /// Event ticker this market belongs to
    pub event_ticker: String,
    /// Market subtitle/description
    pub subtitle: String,
    /// Best bid price for YES
    pub yes_bid: Decimal,
    /// Best ask price for YES
    pub yes_ask: Decimal,
    /// Best bid price for NO
    pub no_bid: Decimal,
    /// Best ask price for NO
    pub no_ask: Decimal,
    /// Last trade price
    pub last_price: Decimal,
    /// 24h volume in contracts
    pub volume: u64,
    /// 24h volume in dollars
    pub volume_24h: Option<Decimal>,
    /// Open interest
    pub open_interest: u64,
    /// When the market closes
    pub close_time: Option<DateTime<Utc>>,
    /// Market status
    pub status: String,
}

impl KalshiMarket {
    /// Get the mid price for YES
    pub fn yes_mid_price(&self) -> Decimal {
        (self.yes_bid + self.yes_ask) / Decimal::TWO
    }

    /// Get the spread
    pub fn spread(&self) -> Decimal {
        self.yes_ask - self.yes_bid
    }

    /// Convert to ExternalMarketPrice
    pub fn to_external_price(&self) -> ExternalMarketPrice {
        let yes_price = self.yes_mid_price();
        let no_price = (self.no_bid + self.no_ask) / Decimal::TWO;

        let mut price = ExternalMarketPrice::new(
            Platform::Kalshi,
            self.ticker.clone(),
            self.subtitle.clone(),
            yes_price,
        )
        .with_no_price(no_price)
        .with_volume(Decimal::from(self.volume));

        if let Some(close_time) = self.close_time {
            price = price.with_market_close(close_time);
        }

        price
    }
}

/// Raw API response for events
#[derive(Debug, Deserialize)]
struct EventsResponse {
    events: Vec<RawEvent>,
    #[allow(dead_code)]
    cursor: Option<String>,
}

/// Raw event from API
#[derive(Debug, Deserialize)]
struct RawEvent {
    event_ticker: String,
    series_ticker: String,
    title: String,
    category: String,
    sub_title: Option<String>,
    status: String,
}

/// Raw API response for markets
#[derive(Debug, Deserialize)]
struct MarketsResponse {
    markets: Vec<RawMarket>,
    #[allow(dead_code)]
    cursor: Option<String>,
}

/// Raw market from API
#[derive(Debug, Deserialize)]
struct RawMarket {
    ticker: String,
    event_ticker: String,
    subtitle: String,
    yes_bid: Option<f64>,
    yes_ask: Option<f64>,
    no_bid: Option<f64>,
    no_ask: Option<f64>,
    last_price: Option<f64>,
    volume: Option<u64>,
    volume_24h: Option<f64>,
    open_interest: Option<u64>,
    close_time: Option<String>,
    status: String,
}

/// Authentication response
#[derive(Debug, Deserialize)]
struct LoginResponse {
    token: String,
    #[allow(dead_code)]
    member_id: String,
}

/// Client for the Kalshi prediction market API
pub struct KalshiClient {
    http_client: reqwest::Client,
    config: KalshiConfig,
    auth_token: Arc<RwLock<Option<String>>>,
    cache: Arc<RwLock<HashMap<String, CachedEvent>>>,
}

impl KalshiClient {
    /// Create a new Kalshi client
    pub fn new(config: KalshiConfig) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("Polysniper/1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            http_client,
            config,
            auth_token: Arc::new(RwLock::new(None)),
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

    /// Authenticate with Kalshi (if credentials are configured)
    pub async fn authenticate(&self) -> Result<(), ExternalMarketError> {
        let email = match &self.config.email {
            Some(e) => e.clone(),
            None => {
                debug!("No Kalshi credentials configured, using unauthenticated access");
                return Ok(());
            }
        };

        let password = std::env::var(&self.config.password_env).map_err(|_| {
            ExternalMarketError::Auth(format!(
                "Password environment variable {} not set",
                self.config.password_env
            ))
        })?;

        let url = format!("{}/login", self.config.api_base_url);

        debug!(url = %url, "Authenticating with Kalshi");

        let response = self
            .http_client
            .post(&url)
            .json(&serde_json::json!({
                "email": email,
                "password": password
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ExternalMarketError::Auth(format!(
                "Login failed: {} - {}",
                status, body
            )));
        }

        let login: LoginResponse = response.json().await.map_err(|e| {
            ExternalMarketError::Parse(format!("Failed to parse login response: {}", e))
        })?;

        {
            let mut token = self.auth_token.write().await;
            *token = Some(login.token);
        }

        info!("Kalshi authentication successful");
        Ok(())
    }

    /// Get events for a series
    pub async fn get_events(
        &self,
        series_ticker: &str,
    ) -> Result<Vec<KalshiEvent>, ExternalMarketError> {
        let url = format!(
            "{}/events?series_ticker={}",
            self.config.api_base_url, series_ticker
        );

        debug!(url = %url, "Fetching Kalshi events");

        let mut request = self.http_client.get(&url);

        // Add auth header if available
        {
            let token = self.auth_token.read().await;
            if let Some(t) = token.as_ref() {
                request = request.header("Authorization", format!("Bearer {}", t));
            }
        }

        let response = request.send().await?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(ExternalMarketError::Auth("Unauthorized".to_string()));
        }

        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ExternalMarketError::RateLimited {
                retry_after_secs: 30,
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

        let events_response: EventsResponse = response.json().await.map_err(|e| {
            ExternalMarketError::Parse(format!("Failed to parse events response: {}", e))
        })?;

        let mut events = Vec::new();
        for raw_event in events_response.events {
            // Fetch markets for each event
            match self.get_markets(&raw_event.event_ticker).await {
                Ok(markets) => {
                    let event = KalshiEvent {
                        event_ticker: raw_event.event_ticker.clone(),
                        series_ticker: raw_event.series_ticker,
                        title: raw_event.title,
                        category: raw_event.category,
                        sub_title: raw_event.sub_title,
                        markets,
                        status: raw_event.status,
                    };

                    // Cache the event
                    {
                        let mut cache = self.cache.write().await;
                        cache.insert(
                            raw_event.event_ticker,
                            CachedEvent {
                                event: event.clone(),
                                cached_at: Utc::now(),
                            },
                        );
                    }

                    events.push(event);
                }
                Err(e) => {
                    warn!(event_ticker = %raw_event.event_ticker, error = %e, "Failed to fetch markets for event");
                }
            }
        }

        Ok(events)
    }

    /// Get markets for an event
    pub async fn get_markets(
        &self,
        event_ticker: &str,
    ) -> Result<Vec<KalshiMarket>, ExternalMarketError> {
        let url = format!(
            "{}/markets?event_ticker={}",
            self.config.api_base_url, event_ticker
        );

        debug!(url = %url, "Fetching Kalshi markets");

        let mut request = self.http_client.get(&url);

        // Add auth header if available
        {
            let token = self.auth_token.read().await;
            if let Some(t) = token.as_ref() {
                request = request.header("Authorization", format!("Bearer {}", t));
            }
        }

        let response = request.send().await?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(ExternalMarketError::Auth("Unauthorized".to_string()));
        }

        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ExternalMarketError::RateLimited {
                retry_after_secs: 30,
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

        let markets_response: MarketsResponse = response.json().await.map_err(|e| {
            ExternalMarketError::Parse(format!("Failed to parse markets response: {}", e))
        })?;

        let markets: Vec<KalshiMarket> = markets_response
            .markets
            .into_iter()
            .map(|m| self.parse_market(m))
            .collect();

        Ok(markets)
    }

    /// Get a specific market by ticker
    pub async fn get_market(&self, ticker: &str) -> Result<KalshiMarket, ExternalMarketError> {
        let url = format!("{}/markets/{}", self.config.api_base_url, ticker);

        debug!(url = %url, "Fetching Kalshi market");

        let mut request = self.http_client.get(&url);

        // Add auth header if available
        {
            let token = self.auth_token.read().await;
            if let Some(t) = token.as_ref() {
                request = request.header("Authorization", format!("Bearer {}", t));
            }
        }

        let response = request.send().await?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(ExternalMarketError::NotFound(format!(
                "Market {} not found",
                ticker
            )));
        }

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(ExternalMarketError::Auth("Unauthorized".to_string()));
        }

        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ExternalMarketError::RateLimited {
                retry_after_secs: 30,
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

        // Single market response wraps in { "market": {...} }
        #[derive(Debug, Deserialize)]
        struct SingleMarketResponse {
            market: RawMarket,
        }

        let market_response: SingleMarketResponse = response.json().await.map_err(|e| {
            ExternalMarketError::Parse(format!("Failed to parse market response: {}", e))
        })?;

        Ok(self.parse_market(market_response.market))
    }

    /// Fetch all tracked series and markets
    pub async fn fetch_tracked(&self) -> Result<Vec<KalshiMarket>, ExternalMarketError> {
        let mut all_markets = Vec::new();

        // Fetch tracked series
        for series in &self.config.tracked_series {
            match self.get_events(series).await {
                Ok(events) => {
                    for event in events {
                        all_markets.extend(event.markets);
                    }
                }
                Err(e) => warn!(series = %series, error = %e, "Failed to fetch tracked series"),
            }
        }

        // Fetch individual tracked markets
        for ticker in &self.config.tracked_markets {
            match self.get_market(ticker).await {
                Ok(market) => all_markets.push(market),
                Err(e) => warn!(ticker = %ticker, error = %e, "Failed to fetch tracked market"),
            }
        }

        Ok(all_markets)
    }

    /// Get cached event if available and not stale
    pub async fn get_cached(&self, event_ticker: &str) -> Option<KalshiEvent> {
        let cache = self.cache.read().await;
        if let Some(cached) = cache.get(event_ticker) {
            let age = Utc::now() - cached.cached_at;
            if age.num_seconds() < self.config.poll_interval_secs as i64 {
                return Some(cached.event.clone());
            }
        }
        None
    }

    /// Parse a raw market into KalshiMarket
    fn parse_market(&self, raw: RawMarket) -> KalshiMarket {
        let close_time = raw
            .close_time
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        // Kalshi prices are in cents (0-100), convert to decimal (0.0-1.0)
        let yes_bid = raw
            .yes_bid
            .map(|p| Decimal::from_f64_retain(p / 100.0).unwrap_or(Decimal::ZERO))
            .unwrap_or(Decimal::ZERO);
        let yes_ask = raw
            .yes_ask
            .map(|p| Decimal::from_f64_retain(p / 100.0).unwrap_or(Decimal::ONE))
            .unwrap_or(Decimal::ONE);
        let no_bid = raw
            .no_bid
            .map(|p| Decimal::from_f64_retain(p / 100.0).unwrap_or(Decimal::ZERO))
            .unwrap_or(Decimal::ZERO);
        let no_ask = raw
            .no_ask
            .map(|p| Decimal::from_f64_retain(p / 100.0).unwrap_or(Decimal::ONE))
            .unwrap_or(Decimal::ONE);
        let last_price = raw
            .last_price
            .map(|p| Decimal::from_f64_retain(p / 100.0).unwrap_or(Decimal::ZERO))
            .unwrap_or(Decimal::ZERO);

        KalshiMarket {
            ticker: raw.ticker,
            event_ticker: raw.event_ticker,
            subtitle: raw.subtitle,
            yes_bid,
            yes_ask,
            no_bid,
            no_ask,
            last_price,
            volume: raw.volume.unwrap_or(0),
            volume_24h: raw.volume_24h.and_then(Decimal::from_f64_retain),
            open_interest: raw.open_interest.unwrap_or(0),
            close_time,
            status: raw.status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_default_config() {
        let config = KalshiConfig::default();
        assert!(!config.enabled);
        assert_eq!(
            config.api_base_url,
            "https://trading-api.kalshi.com/trade-api/v2"
        );
        assert_eq!(config.poll_interval_secs, 30);
        assert!(config.tracked_series.is_empty());
    }

    #[test]
    fn test_market_mid_price() {
        let market = KalshiMarket {
            ticker: "TEST-MARKET".to_string(),
            event_ticker: "TEST-EVENT".to_string(),
            subtitle: "Test Market".to_string(),
            yes_bid: dec!(0.45),
            yes_ask: dec!(0.50),
            no_bid: dec!(0.50),
            no_ask: dec!(0.55),
            last_price: dec!(0.47),
            volume: 1000,
            volume_24h: Some(dec!(5000)),
            open_interest: 500,
            close_time: None,
            status: "active".to_string(),
        };

        assert_eq!(market.yes_mid_price(), dec!(0.475));
        assert_eq!(market.spread(), dec!(0.05));
    }

    #[test]
    fn test_market_to_external_price() {
        let market = KalshiMarket {
            ticker: "FED-25JAN".to_string(),
            event_ticker: "FED-RATE".to_string(),
            subtitle: "Fed Rate Decision".to_string(),
            yes_bid: dec!(0.60),
            yes_ask: dec!(0.65),
            no_bid: dec!(0.35),
            no_ask: dec!(0.40),
            last_price: dec!(0.62),
            volume: 2000,
            volume_24h: Some(dec!(10000)),
            open_interest: 1000,
            close_time: None,
            status: "active".to_string(),
        };

        let price = market.to_external_price();
        assert_eq!(price.platform, Platform::Kalshi);
        assert_eq!(price.question_id, "FED-25JAN");
        assert_eq!(price.yes_price, dec!(0.625)); // mid price
        assert_eq!(price.no_price, Some(dec!(0.375))); // mid price
        assert_eq!(price.volume, Some(dec!(2000)));
    }
}
