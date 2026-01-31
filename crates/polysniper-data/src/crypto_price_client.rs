//! Crypto Price Client
//!
//! Fetches cryptocurrency prices from external APIs (CoinGecko, Binance)
//! and publishes CryptoPriceUpdate events to the event bus.

use crate::BroadcastEventBus;
use chrono::{DateTime, Utc};
use polysniper_core::{CryptoPriceUpdateEvent, DataSourceError, EventBus, SystemEvent};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// API provider for crypto prices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CryptoApiProvider {
    #[default]
    CoinGecko,
    Binance,
}

/// Configuration for crypto price client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoPriceConfig {
    /// Whether the client is enabled
    pub enabled: bool,
    /// Symbols to track (e.g., ["ETH", "BTC", "SOL"])
    pub symbols: Vec<String>,
    /// Poll interval in seconds
    pub poll_interval_secs: u64,
    /// API provider to use
    pub api_provider: CryptoApiProvider,
    /// Optional API key (required for some providers)
    pub api_key: Option<String>,
}

impl Default for CryptoPriceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            symbols: vec!["ETH".to_string(), "BTC".to_string(), "SOL".to_string()],
            poll_interval_secs: 30,
            api_provider: CryptoApiProvider::default(),
            api_key: None,
        }
    }
}

/// Cached crypto price data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoPrice {
    /// Symbol (e.g., "ETH")
    pub symbol: String,
    /// Price in USD
    pub price_usd: Decimal,
    /// 1-hour price change percentage
    pub price_change_1h: Decimal,
    /// 24-hour price change percentage
    pub price_change_24h: Decimal,
    /// 24-hour volume in USD
    pub volume_24h: Decimal,
    /// When this price was fetched
    pub timestamp: DateTime<Utc>,
}

/// CoinGecko API response for simple price endpoint
#[derive(Debug, Deserialize)]
struct CoinGeckoSimplePrice {
    usd: f64,
    #[serde(default)]
    usd_24h_change: Option<f64>,
    #[serde(default)]
    usd_24h_vol: Option<f64>,
}

/// Binance API ticker response
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BinanceTicker {
    symbol: String,
    last_price: String,
    price_change_percent: String,
    quote_volume: String,
}

/// Crypto price client for fetching external crypto prices
pub struct CryptoPriceClient {
    http_client: Client,
    event_bus: Arc<BroadcastEventBus>,
    config: CryptoPriceConfig,
    price_cache: Arc<RwLock<HashMap<String, CryptoPrice>>>,
    running: Arc<RwLock<bool>>,
}

impl CryptoPriceClient {
    /// Create a new crypto price client
    pub fn new(config: CryptoPriceConfig, event_bus: Arc<BroadcastEventBus>) -> Self {
        let http_client = Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            http_client,
            event_bus,
            config,
            price_cache: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the polling loop
    pub async fn start(&self) {
        if !self.config.enabled {
            info!("Crypto price client is disabled");
            return;
        }

        {
            let mut running = self.running.write().await;
            if *running {
                warn!("Crypto price client already running");
                return;
            }
            *running = true;
        }

        info!(
            symbols = ?self.config.symbols,
            provider = ?self.config.api_provider,
            "Starting crypto price client"
        );

        let poll_interval = Duration::from_secs(self.config.poll_interval_secs);
        let mut interval = tokio::time::interval(poll_interval);

        loop {
            interval.tick().await;

            if !*self.running.read().await {
                info!("Crypto price client stopped");
                break;
            }

            if let Err(e) = self.poll_prices().await {
                error!(error = %e, "Failed to poll crypto prices");
            }
        }
    }

    /// Stop the polling loop
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
        info!("Stopping crypto price client");
    }

    /// Poll prices from the configured API provider
    async fn poll_prices(&self) -> Result<(), DataSourceError> {
        let prices = match self.config.api_provider {
            CryptoApiProvider::CoinGecko => self.fetch_coingecko_prices().await?,
            CryptoApiProvider::Binance => self.fetch_binance_prices().await?,
        };

        let mut cache = self.price_cache.write().await;
        for price in prices {
            let event = CryptoPriceUpdateEvent::new(
                price.symbol.clone(),
                price.price_usd,
                price.price_change_1h,
                price.price_change_24h,
                price.volume_24h,
            );

            debug!(
                symbol = %price.symbol,
                price = %price.price_usd,
                change_1h = %price.price_change_1h,
                "Publishing crypto price update"
            );

            self.event_bus.publish(SystemEvent::CryptoPriceUpdate(event));
            cache.insert(price.symbol.clone(), price);
        }

        Ok(())
    }

    /// Fetch prices from CoinGecko API
    async fn fetch_coingecko_prices(&self) -> Result<Vec<CryptoPrice>, DataSourceError> {
        let ids: Vec<&str> = self
            .config
            .symbols
            .iter()
            .filter_map(|s| symbol_to_coingecko_id(s))
            .collect();

        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let ids_param = ids.join(",");
        let url = format!(
            "https://api.coingecko.com/api/v3/simple/price?ids={}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true",
            ids_param
        );

        debug!("Fetching CoinGecko prices: {}", url);

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| DataSourceError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(DataSourceError::HttpError(format!(
                "CoinGecko API returned status {}",
                response.status()
            )));
        }

        let data: HashMap<String, CoinGeckoSimplePrice> = response
            .json()
            .await
            .map_err(|e| DataSourceError::ParseError(e.to_string()))?;

        let now = Utc::now();
        let prices: Vec<CryptoPrice> = self
            .config
            .symbols
            .iter()
            .filter_map(|symbol| {
                let cg_id = symbol_to_coingecko_id(symbol)?;
                let price_data = data.get(cg_id)?;

                Some(CryptoPrice {
                    symbol: symbol.clone(),
                    price_usd: Decimal::try_from(price_data.usd).unwrap_or(Decimal::ZERO),
                    price_change_1h: Decimal::ZERO, // CoinGecko free tier doesn't have 1h change
                    price_change_24h: price_data
                        .usd_24h_change
                        .map(|v| Decimal::try_from(v).unwrap_or(Decimal::ZERO))
                        .unwrap_or(Decimal::ZERO),
                    volume_24h: price_data
                        .usd_24h_vol
                        .map(|v| Decimal::try_from(v).unwrap_or(Decimal::ZERO))
                        .unwrap_or(Decimal::ZERO),
                    timestamp: now,
                })
            })
            .collect();

        info!("Fetched {} crypto prices from CoinGecko", prices.len());
        Ok(prices)
    }

    /// Fetch prices from Binance API
    async fn fetch_binance_prices(&self) -> Result<Vec<CryptoPrice>, DataSourceError> {
        let url = "https://api.binance.com/api/v3/ticker/24hr";

        debug!("Fetching Binance prices");

        let response = self
            .http_client
            .get(url)
            .send()
            .await
            .map_err(|e| DataSourceError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(DataSourceError::HttpError(format!(
                "Binance API returned status {}",
                response.status()
            )));
        }

        let tickers: Vec<BinanceTicker> = response
            .json()
            .await
            .map_err(|e| DataSourceError::ParseError(e.to_string()))?;

        let now = Utc::now();
        let prices: Vec<CryptoPrice> = self
            .config
            .symbols
            .iter()
            .filter_map(|symbol| {
                let pair = format!("{}USDT", symbol.to_uppercase());
                let ticker = tickers.iter().find(|t| t.symbol == pair)?;

                let price = ticker.last_price.parse::<Decimal>().ok()?;
                let change_24h = ticker.price_change_percent.parse::<Decimal>().ok()?;
                let volume = ticker.quote_volume.parse::<Decimal>().ok()?;

                Some(CryptoPrice {
                    symbol: symbol.clone(),
                    price_usd: price,
                    price_change_1h: Decimal::ZERO, // Would need separate endpoint
                    price_change_24h: change_24h,
                    volume_24h: volume,
                    timestamp: now,
                })
            })
            .collect();

        info!("Fetched {} crypto prices from Binance", prices.len());
        Ok(prices)
    }

    /// Get cached price for a symbol
    pub async fn get_price(&self, symbol: &str) -> Option<CryptoPrice> {
        let cache = self.price_cache.read().await;
        cache.get(symbol).cloned()
    }

    /// Get all cached prices
    pub async fn get_all_prices(&self) -> Vec<CryptoPrice> {
        let cache = self.price_cache.read().await;
        cache.values().cloned().collect()
    }

    /// Check if client is running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }
}

/// Map symbol to CoinGecko ID
fn symbol_to_coingecko_id(symbol: &str) -> Option<&'static str> {
    match symbol.to_uppercase().as_str() {
        "BTC" => Some("bitcoin"),
        "ETH" => Some("ethereum"),
        "SOL" => Some("solana"),
        "MATIC" => Some("matic-network"),
        "AVAX" => Some("avalanche-2"),
        "LINK" => Some("chainlink"),
        "UNI" => Some("uniswap"),
        "AAVE" => Some("aave"),
        "DOT" => Some("polkadot"),
        "ATOM" => Some("cosmos"),
        "ADA" => Some("cardano"),
        "DOGE" => Some("dogecoin"),
        "XRP" => Some("ripple"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_symbol_to_coingecko_id() {
        assert_eq!(symbol_to_coingecko_id("BTC"), Some("bitcoin"));
        assert_eq!(symbol_to_coingecko_id("ETH"), Some("ethereum"));
        assert_eq!(symbol_to_coingecko_id("SOL"), Some("solana"));
        assert_eq!(symbol_to_coingecko_id("eth"), Some("ethereum"));
        assert_eq!(symbol_to_coingecko_id("UNKNOWN"), None);
    }

    #[test]
    fn test_crypto_price_config_default() {
        let config = CryptoPriceConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.symbols.len(), 3);
        assert_eq!(config.poll_interval_secs, 30);
        assert_eq!(config.api_provider, CryptoApiProvider::CoinGecko);
    }

    #[test]
    fn test_crypto_price_update_event() {
        let event = CryptoPriceUpdateEvent::new(
            "ETH".to_string(),
            dec!(2500.0),
            dec!(5.5),
            dec!(10.2),
            dec!(1000000000),
        );

        assert_eq!(event.symbol, "ETH");
        assert!(event.is_significant_move(dec!(5.0)));
        assert!(!event.is_significant_move(dec!(6.0)));
    }

    #[tokio::test]
    async fn test_crypto_price_client_disabled() {
        let config = CryptoPriceConfig {
            enabled: false,
            ..Default::default()
        };
        let event_bus = Arc::new(BroadcastEventBus::new());
        let client = CryptoPriceClient::new(config, event_bus);

        // Should not start when disabled
        assert!(!client.is_running().await);
    }

    #[tokio::test]
    async fn test_price_cache() {
        let config = CryptoPriceConfig::default();
        let event_bus = Arc::new(BroadcastEventBus::new());
        let client = CryptoPriceClient::new(config, event_bus);

        // Initially empty
        assert!(client.get_price("ETH").await.is_none());
        assert!(client.get_all_prices().await.is_empty());

        // Manually insert into cache for testing
        {
            let mut cache = client.price_cache.write().await;
            cache.insert(
                "ETH".to_string(),
                CryptoPrice {
                    symbol: "ETH".to_string(),
                    price_usd: dec!(2500),
                    price_change_1h: dec!(1.5),
                    price_change_24h: dec!(5.0),
                    volume_24h: dec!(1000000000),
                    timestamp: Utc::now(),
                },
            );
        }

        // Now should be cached
        let price = client.get_price("ETH").await.unwrap();
        assert_eq!(price.symbol, "ETH");
        assert_eq!(price.price_usd, dec!(2500));

        let all = client.get_all_prices().await;
        assert_eq!(all.len(), 1);
    }
}
