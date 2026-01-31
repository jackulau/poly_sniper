//! LLM Provider Implementations
//!
//! Trait and implementations for LLM providers in the ensemble system.

mod openrouter;

pub use openrouter::OpenRouterProvider;

use async_trait::async_trait;
use polysniper_core::{LlmPredictionResult, MarketContext, ProviderConfig};
use rust_decimal::Decimal;
use std::sync::Arc;

/// Error type for LLM provider operations
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Timeout")]
    Timeout,
    #[error("Rate limited")]
    RateLimited,
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Trait for LLM prediction providers
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Get the unique model identifier
    fn model_id(&self) -> &str;

    /// Get the provider name (e.g., "openrouter", "anthropic")
    fn provider_name(&self) -> &str;

    /// Get the cost per request in USD
    fn cost_per_request(&self) -> Decimal;

    /// Make a prediction for the given market context
    async fn predict(&self, context: &MarketContext) -> Result<LlmPredictionResult, ProviderError>;

    /// Check if the provider is healthy/available
    async fn health_check(&self) -> bool {
        true
    }
}

/// Type alias for boxed provider
pub type BoxedProvider = Arc<dyn LlmProvider>;

/// Factory for creating providers from configuration
pub struct ProviderFactory;

impl ProviderFactory {
    /// Create a provider from configuration
    pub fn create(config: &ProviderConfig, api_key: &str) -> Result<BoxedProvider, ProviderError> {
        match config.provider_type.as_str() {
            "openrouter" => {
                let provider = OpenRouterProvider::new(config, api_key)?;
                Ok(Arc::new(provider))
            }
            other => Err(ProviderError::ConfigError(format!(
                "Unknown provider type: {}",
                other
            ))),
        }
    }

    /// Create multiple providers from configurations
    pub fn create_all(
        configs: &[ProviderConfig],
        api_key: &str,
    ) -> Vec<(ProviderConfig, Result<BoxedProvider, ProviderError>)> {
        configs
            .iter()
            .filter(|c| c.enabled)
            .map(|config| (config.clone(), Self::create(config, api_key)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_provider_factory_unknown_type() {
        let config = ProviderConfig {
            provider_type: "unknown".to_string(),
            model: "test".to_string(),
            enabled: true,
            weight: dec!(1.0),
            cost_per_request: dec!(0.01),
            temperature: None,
            max_tokens: None,
        };

        let result = ProviderFactory::create(&config, "test-key");
        assert!(result.is_err());
        match result {
            Err(ProviderError::ConfigError(_)) => (),
            _ => panic!("Expected ConfigError"),
        }
    }
}
