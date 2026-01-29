//! Discord integration for Polysniper
//!
//! Provides rich embed building and preset templates for Discord webhook notifications.

pub mod embed;
pub mod templates;

pub use embed::{
    Embed, EmbedAuthor, EmbedBuilder, EmbedColor, EmbedField, EmbedFooter, EmbedImage,
    EmbedProvider, EmbedThumbnail, EmbedVideo,
};
pub use templates::{ErrorEmbed, RiskEmbed, StatusEmbed, TradeEmbed};
