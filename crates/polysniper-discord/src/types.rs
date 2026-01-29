//! Discord webhook types for message formatting

use serde::{Deserialize, Serialize};

/// Discord webhook message payload
#[derive(Debug, Clone, Serialize, Default)]
pub struct DiscordMessage {
    /// Plain text content (up to 2000 characters)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Username to display for this message (overrides webhook default)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,

    /// Avatar URL to display for this message (overrides webhook default)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avatar_url: Option<String>,

    /// Rich embeds (up to 10)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub embeds: Vec<DiscordEmbed>,

    /// Whether this is a TTS message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tts: Option<bool>,
}

impl DiscordMessage {
    /// Create a new message with plain text content
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            ..Default::default()
        }
    }

    /// Create a new message with a single embed
    pub fn embed(embed: DiscordEmbed) -> Self {
        Self {
            embeds: vec![embed],
            ..Default::default()
        }
    }

    /// Add an embed to the message
    pub fn with_embed(mut self, embed: DiscordEmbed) -> Self {
        self.embeds.push(embed);
        self
    }

    /// Set the username for this message
    pub fn with_username(mut self, username: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self
    }

    /// Set the avatar URL for this message
    pub fn with_avatar_url(mut self, avatar_url: impl Into<String>) -> Self {
        self.avatar_url = Some(avatar_url.into());
        self
    }
}

/// Discord embed for rich message formatting
#[derive(Debug, Clone, Serialize, Default)]
pub struct DiscordEmbed {
    /// Embed title (up to 256 characters)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Embed description (up to 4096 characters)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// URL for the title to link to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Embed color as integer (decimal representation of hex color)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<u32>,

    /// ISO8601 timestamp for the embed footer
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,

    /// Footer information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub footer: Option<DiscordEmbedFooter>,

    /// Thumbnail image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail: Option<DiscordEmbedImage>,

    /// Main image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<DiscordEmbedImage>,

    /// Author information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<DiscordEmbedAuthor>,

    /// Fields (up to 25)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<DiscordEmbedField>,
}

impl DiscordEmbed {
    /// Create a new embed with a title
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the embed title
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the embed description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the embed URL
    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Set the embed color from hex (e.g., 0xFF5733)
    pub fn color(mut self, color: u32) -> Self {
        self.color = Some(color);
        self
    }

    /// Set the timestamp
    pub fn timestamp(mut self, timestamp: impl Into<String>) -> Self {
        self.timestamp = Some(timestamp.into());
        self
    }

    /// Add a field to the embed
    pub fn field(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
        inline: bool,
    ) -> Self {
        self.fields.push(DiscordEmbedField {
            name: name.into(),
            value: value.into(),
            inline,
        });
        self
    }

    /// Set the footer
    pub fn footer(mut self, text: impl Into<String>) -> Self {
        self.footer = Some(DiscordEmbedFooter {
            text: text.into(),
            icon_url: None,
        });
        self
    }

    /// Set the footer with an icon
    pub fn footer_with_icon(
        mut self,
        text: impl Into<String>,
        icon_url: impl Into<String>,
    ) -> Self {
        self.footer = Some(DiscordEmbedFooter {
            text: text.into(),
            icon_url: Some(icon_url.into()),
        });
        self
    }

    /// Set the thumbnail image
    pub fn thumbnail(mut self, url: impl Into<String>) -> Self {
        self.thumbnail = Some(DiscordEmbedImage { url: url.into() });
        self
    }

    /// Set the main image
    pub fn image(mut self, url: impl Into<String>) -> Self {
        self.image = Some(DiscordEmbedImage { url: url.into() });
        self
    }

    /// Set the author
    pub fn author(mut self, name: impl Into<String>) -> Self {
        self.author = Some(DiscordEmbedAuthor {
            name: name.into(),
            url: None,
            icon_url: None,
        });
        self
    }

    /// Set the author with URL and icon
    pub fn author_with_url(
        mut self,
        name: impl Into<String>,
        url: impl Into<String>,
        icon_url: Option<String>,
    ) -> Self {
        self.author = Some(DiscordEmbedAuthor {
            name: name.into(),
            url: Some(url.into()),
            icon_url,
        });
        self
    }
}

/// Discord embed field
#[derive(Debug, Clone, Serialize)]
pub struct DiscordEmbedField {
    /// Field name (up to 256 characters)
    pub name: String,

    /// Field value (up to 1024 characters)
    pub value: String,

    /// Whether this field should be displayed inline
    #[serde(default)]
    pub inline: bool,
}

/// Discord embed footer
#[derive(Debug, Clone, Serialize)]
pub struct DiscordEmbedFooter {
    /// Footer text (up to 2048 characters)
    pub text: String,

    /// Footer icon URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon_url: Option<String>,
}

/// Discord embed image
#[derive(Debug, Clone, Serialize)]
pub struct DiscordEmbedImage {
    /// Image URL
    pub url: String,
}

/// Discord embed author
#[derive(Debug, Clone, Serialize)]
pub struct DiscordEmbedAuthor {
    /// Author name (up to 256 characters)
    pub name: String,

    /// Author URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Author icon URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon_url: Option<String>,
}

/// Response from Discord webhook API
#[derive(Debug, Clone, Deserialize)]
pub struct WebhookResponse {
    /// Message ID (only present if ?wait=true)
    pub id: Option<String>,

    /// Error code (if request failed)
    pub code: Option<u32>,

    /// Error message (if request failed)
    pub message: Option<String>,

    /// Retry after (ms) for rate limits
    pub retry_after: Option<f64>,
}

/// Common embed colors for alerts
#[allow(dead_code)]
pub mod colors {
    /// Green - use for success messages
    pub const SUCCESS: u32 = 0x57F287;
    /// Yellow - use for warning messages
    pub const WARNING: u32 = 0xFEE75C;
    /// Red - use for error messages
    pub const ERROR: u32 = 0xED4245;
    /// Blurple (Discord brand color) - use for informational messages
    pub const INFO: u32 = 0x5865F2;
    /// Gray - use for neutral messages
    pub const NEUTRAL: u32 = 0x99AAB5;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_text_serialization() {
        let msg = DiscordMessage::text("Hello, World!");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"content\":\"Hello, World!\""));
        assert!(!json.contains("embeds"));
    }

    #[test]
    fn test_message_with_username() {
        let msg = DiscordMessage::text("Test")
            .with_username("Polysniper Bot")
            .with_avatar_url("https://example.com/avatar.png");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"username\":\"Polysniper Bot\""));
        assert!(json.contains("\"avatar_url\":\"https://example.com/avatar.png\""));
    }

    #[test]
    fn test_embed_serialization() {
        let embed = DiscordEmbed::new()
            .title("Test Title")
            .description("Test Description")
            .color(colors::SUCCESS)
            .field("Field 1", "Value 1", true)
            .field("Field 2", "Value 2", false)
            .footer("Footer Text");

        let json = serde_json::to_string(&embed).unwrap();
        assert!(json.contains("\"title\":\"Test Title\""));
        assert!(json.contains("\"description\":\"Test Description\""));
        assert!(json.contains("\"color\":5763719")); // 0x57F287 in decimal
        assert!(json.contains("\"name\":\"Field 1\""));
        assert!(json.contains("\"value\":\"Value 1\""));
        assert!(json.contains("\"inline\":true"));
        assert!(json.contains("\"text\":\"Footer Text\""));
    }

    #[test]
    fn test_message_with_embed() {
        let embed = DiscordEmbed::new()
            .title("Alert")
            .description("Something happened");

        let msg = DiscordMessage::embed(embed);
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"embeds\""));
        assert!(json.contains("\"title\":\"Alert\""));
    }

    #[test]
    fn test_empty_fields_not_serialized() {
        let msg = DiscordMessage::default();
        let json = serde_json::to_string(&msg).unwrap();
        // Empty message should serialize to just {}
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_embed_builder_chain() {
        let embed = DiscordEmbed::new()
            .title("Order Executed")
            .description("Your order has been filled")
            .url("https://polymarket.com")
            .color(colors::SUCCESS)
            .field("Market", "Election 2024", false)
            .field("Side", "YES", true)
            .field("Price", "$0.65", true)
            .thumbnail("https://example.com/thumb.png")
            .footer_with_icon("Polysniper", "https://example.com/icon.png")
            .timestamp("2024-01-15T12:00:00Z");

        let json = serde_json::to_string(&embed).unwrap();
        assert!(json.contains("\"title\":\"Order Executed\""));
        assert!(json.contains("\"url\":\"https://polymarket.com\""));
        assert!(json.contains("\"timestamp\":\"2024-01-15T12:00:00Z\""));
        assert!(json.contains("\"thumbnail\""));
    }
}
