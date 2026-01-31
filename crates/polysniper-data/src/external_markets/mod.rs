//! External Prediction Market Clients
//!
//! Clients for fetching prices from external prediction markets
//! like Metaculus, PredictIt, and Kalshi.

pub mod kalshi;
pub mod metaculus;
pub mod predictit;

pub use kalshi::{KalshiClient, KalshiConfig, KalshiEvent, KalshiMarket};
pub use metaculus::{MetaculusClient, MetaculusConfig, MetaculusPrediction};
pub use predictit::{PredictItClient, PredictItConfig, PredictItContract, PredictItMarket};

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// External prediction market platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Platform {
    /// Metaculus - Expert forecaster community
    Metaculus,
    /// PredictIt - US-based prediction market
    PredictIt,
    /// Kalshi - Regulated US prediction market
    Kalshi,
}

impl Platform {
    /// Get the platform name as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            Platform::Metaculus => "Metaculus",
            Platform::PredictIt => "PredictIt",
            Platform::Kalshi => "Kalshi",
        }
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Price information from an external prediction market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalMarketPrice {
    /// The platform this price is from
    pub platform: Platform,
    /// Platform-specific question/market identifier
    pub question_id: String,
    /// Human-readable question/market name
    pub question: String,
    /// YES price (probability 0.0-1.0)
    pub yes_price: Decimal,
    /// NO price (if available)
    pub no_price: Option<Decimal>,
    /// Trading volume (if available)
    pub volume: Option<Decimal>,
    /// When the price was last updated
    pub last_updated: DateTime<Utc>,
    /// When the market closes (if available)
    pub market_close: Option<DateTime<Utc>>,
}

impl ExternalMarketPrice {
    /// Create a new external market price
    pub fn new(
        platform: Platform,
        question_id: String,
        question: String,
        yes_price: Decimal,
    ) -> Self {
        Self {
            platform,
            question_id,
            question,
            yes_price,
            no_price: None,
            volume: None,
            last_updated: Utc::now(),
            market_close: None,
        }
    }

    /// Set the NO price
    pub fn with_no_price(mut self, no_price: Decimal) -> Self {
        self.no_price = Some(no_price);
        self
    }

    /// Set the volume
    pub fn with_volume(mut self, volume: Decimal) -> Self {
        self.volume = Some(volume);
        self
    }

    /// Set the market close time
    pub fn with_market_close(mut self, market_close: DateTime<Utc>) -> Self {
        self.market_close = Some(market_close);
        self
    }

    /// Set the last updated time
    pub fn with_last_updated(mut self, last_updated: DateTime<Utc>) -> Self {
        self.last_updated = last_updated;
        self
    }
}

/// Error type for external market operations
#[derive(Debug, thiserror::Error)]
pub enum ExternalMarketError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Failed to parse response: {0}")]
    Parse(String),

    #[error("Authentication failed: {0}")]
    Auth(String),

    #[error("Rate limited: retry after {retry_after_secs} seconds")]
    RateLimited { retry_after_secs: u64 },

    #[error("Market not found: {0}")]
    NotFound(String),

    #[error("API error: {0}")]
    Api(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_platform_as_str() {
        assert_eq!(Platform::Metaculus.as_str(), "Metaculus");
        assert_eq!(Platform::PredictIt.as_str(), "PredictIt");
        assert_eq!(Platform::Kalshi.as_str(), "Kalshi");
    }

    #[test]
    fn test_external_market_price_builder() {
        let price = ExternalMarketPrice::new(
            Platform::Metaculus,
            "12345".to_string(),
            "Test Question".to_string(),
            dec!(0.65),
        )
        .with_no_price(dec!(0.35))
        .with_volume(dec!(1000));

        assert_eq!(price.platform, Platform::Metaculus);
        assert_eq!(price.yes_price, dec!(0.65));
        assert_eq!(price.no_price, Some(dec!(0.35)));
        assert_eq!(price.volume, Some(dec!(1000)));
    }
}
