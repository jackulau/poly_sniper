//! Discord integration for Polysniper
//!
//! Provides Discord webhook client, rich embed building, and preset templates
//! for Discord webhook notifications.

// Webhook client
mod client;
mod error;
mod types;

// Embed builder and templates
pub mod embed;
pub mod templates;

// Re-exports from client
pub use client::{DiscordClientConfig, DiscordWebhookClient};
pub use error::DiscordError;
pub use types::{colors, DiscordEmbed, DiscordEmbedField, DiscordMessage, WebhookResponse};

// Re-exports from embed builder
pub use embed::{
    Embed, EmbedAuthor, EmbedBuilder, EmbedColor, EmbedField, EmbedFooter, EmbedImage,
    EmbedProvider, EmbedThumbnail, EmbedVideo,
};
pub use templates::{ErrorEmbed, RiskEmbed, StatusEmbed, TradeEmbed};
