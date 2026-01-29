//! Discord webhook client for Polysniper alerts and notifications

mod client;
mod error;
mod types;

pub use client::{DiscordClientConfig, DiscordWebhookClient};
pub use error::DiscordError;
pub use types::{colors, DiscordEmbed, DiscordEmbedField, DiscordMessage, WebhookResponse};
