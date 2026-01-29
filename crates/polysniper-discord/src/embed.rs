//! Discord embed types and fluent builder API
//!
//! Implements the Discord embed specification:
//! https://discord.com/developers/docs/resources/message#embed-object

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Discord embed field limits
pub mod limits {
    pub const TITLE_MAX: usize = 256;
    pub const DESCRIPTION_MAX: usize = 4096;
    pub const FIELDS_MAX: usize = 25;
    pub const FIELD_NAME_MAX: usize = 256;
    pub const FIELD_VALUE_MAX: usize = 1024;
    pub const FOOTER_TEXT_MAX: usize = 2048;
    pub const AUTHOR_NAME_MAX: usize = 256;
    pub const TOTAL_CHARS_MAX: usize = 6000;
}

/// Errors that can occur when building embeds
#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("Title exceeds maximum length of {limit} characters (got {actual})")]
    TitleTooLong { limit: usize, actual: usize },

    #[error("Description exceeds maximum length of {limit} characters (got {actual})")]
    DescriptionTooLong { limit: usize, actual: usize },

    #[error("Too many fields: maximum is {limit}, got {actual}")]
    TooManyFields { limit: usize, actual: usize },

    #[error("Field name exceeds maximum length of {limit} characters (got {actual})")]
    FieldNameTooLong { limit: usize, actual: usize },

    #[error("Field value exceeds maximum length of {limit} characters (got {actual})")]
    FieldValueTooLong { limit: usize, actual: usize },

    #[error("Footer text exceeds maximum length of {limit} characters (got {actual})")]
    FooterTextTooLong { limit: usize, actual: usize },

    #[error("Author name exceeds maximum length of {limit} characters (got {actual})")]
    AuthorNameTooLong { limit: usize, actual: usize },

    #[error("Total embed size exceeds maximum of {limit} characters (got {actual})")]
    TotalSizeTooLarge { limit: usize, actual: usize },

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),
}

/// Predefined embed colors for common use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedColor {
    /// Green - for successful operations
    Success,
    /// Yellow/Orange - for warnings
    Warning,
    /// Red - for errors
    Error,
    /// Blue - for informational messages
    Info,
    /// Purple - for Polymarket branding
    Polymarket,
    /// Custom hex color (without # prefix)
    Custom(u32),
}

impl EmbedColor {
    /// Convert to Discord color integer format
    pub fn to_discord_color(self) -> u32 {
        match self {
            EmbedColor::Success => 0x2ECC71,   // Green
            EmbedColor::Warning => 0xF39C12,   // Orange
            EmbedColor::Error => 0xE74C3C,     // Red
            EmbedColor::Info => 0x3498DB,      // Blue
            EmbedColor::Polymarket => 0x7B3FE4, // Purple
            EmbedColor::Custom(color) => color,
        }
    }

    /// Create from hex string (with or without # prefix)
    pub fn from_hex(hex: &str) -> Result<Self, EmbedError> {
        let hex = hex.trim_start_matches('#');
        u32::from_str_radix(hex, 16)
            .map(EmbedColor::Custom)
            .map_err(|_| EmbedError::InvalidUrl(format!("Invalid hex color: {}", hex)))
    }
}

/// Embed field structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbedField {
    pub name: String,
    pub value: String,
    #[serde(default)]
    pub inline: bool,
}

impl EmbedField {
    pub fn new(name: impl Into<String>, value: impl Into<String>, inline: bool) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            inline,
        }
    }
}

/// Embed author structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbedAuthor {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy_icon_url: Option<String>,
}

/// Embed footer structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbedFooter {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy_icon_url: Option<String>,
}

/// Embed image structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbedImage {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
}

/// Embed thumbnail structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbedThumbnail {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
}

/// Embed video structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbedVideo {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
}

/// Embed provider structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbedProvider {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Complete Discord embed structure
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Embed {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub embed_type: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub footer: Option<EmbedFooter>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<EmbedImage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail: Option<EmbedThumbnail>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub video: Option<EmbedVideo>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<EmbedProvider>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<EmbedAuthor>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<EmbedField>,
}

impl Embed {
    /// Calculate the total character count for embed size validation
    pub fn total_chars(&self) -> usize {
        let mut total = 0;

        if let Some(ref title) = self.title {
            total += title.len();
        }
        if let Some(ref description) = self.description {
            total += description.len();
        }
        if let Some(ref footer) = self.footer {
            total += footer.text.len();
        }
        if let Some(ref author) = self.author {
            total += author.name.len();
        }
        for field in &self.fields {
            total += field.name.len() + field.value.len();
        }

        total
    }
}

/// Fluent builder for Discord embeds
#[derive(Debug, Clone, Default)]
pub struct EmbedBuilder {
    embed: Embed,
}

impl EmbedBuilder {
    /// Create a new embed builder
    pub fn new() -> Self {
        Self {
            embed: Embed {
                embed_type: Some("rich".to_string()),
                ..Default::default()
            },
        }
    }

    /// Set the embed title
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.embed.title = Some(title.into());
        self
    }

    /// Set the embed description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.embed.description = Some(description.into());
        self
    }

    /// Set the embed URL (makes title clickable)
    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.embed.url = Some(url.into());
        self
    }

    /// Set the embed color
    pub fn color(mut self, color: EmbedColor) -> Self {
        self.embed.color = Some(color.to_discord_color());
        self
    }

    /// Set the embed color from a raw u32 value
    pub fn color_raw(mut self, color: u32) -> Self {
        self.embed.color = Some(color);
        self
    }

    /// Set the embed timestamp
    pub fn timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.embed.timestamp = Some(timestamp.to_rfc3339());
        self
    }

    /// Set the embed timestamp to now
    pub fn timestamp_now(self) -> Self {
        self.timestamp(Utc::now())
    }

    /// Add a field to the embed
    pub fn field(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
        inline: bool,
    ) -> Self {
        self.embed
            .fields
            .push(EmbedField::new(name, value, inline));
        self
    }

    /// Add an inline field (shorthand for field with inline=true)
    pub fn inline_field(self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.field(name, value, true)
    }

    /// Set the embed footer
    pub fn footer(mut self, text: impl Into<String>) -> Self {
        self.embed.footer = Some(EmbedFooter {
            text: text.into(),
            icon_url: None,
            proxy_icon_url: None,
        });
        self
    }

    /// Set the embed footer with icon
    pub fn footer_with_icon(mut self, text: impl Into<String>, icon_url: impl Into<String>) -> Self {
        self.embed.footer = Some(EmbedFooter {
            text: text.into(),
            icon_url: Some(icon_url.into()),
            proxy_icon_url: None,
        });
        self
    }

    /// Set the embed author
    pub fn author(mut self, name: impl Into<String>) -> Self {
        self.embed.author = Some(EmbedAuthor {
            name: name.into(),
            url: None,
            icon_url: None,
            proxy_icon_url: None,
        });
        self
    }

    /// Set the embed author with URL
    pub fn author_with_url(mut self, name: impl Into<String>, url: impl Into<String>) -> Self {
        self.embed.author = Some(EmbedAuthor {
            name: name.into(),
            url: Some(url.into()),
            icon_url: None,
            proxy_icon_url: None,
        });
        self
    }

    /// Set the embed author with full details
    pub fn author_full(
        mut self,
        name: impl Into<String>,
        url: Option<String>,
        icon_url: Option<String>,
    ) -> Self {
        self.embed.author = Some(EmbedAuthor {
            name: name.into(),
            url,
            icon_url,
            proxy_icon_url: None,
        });
        self
    }

    /// Set the embed image
    pub fn image(mut self, url: impl Into<String>) -> Self {
        self.embed.image = Some(EmbedImage {
            url: url.into(),
            proxy_url: None,
            height: None,
            width: None,
        });
        self
    }

    /// Set the embed thumbnail
    pub fn thumbnail(mut self, url: impl Into<String>) -> Self {
        self.embed.thumbnail = Some(EmbedThumbnail {
            url: url.into(),
            proxy_url: None,
            height: None,
            width: None,
        });
        self
    }

    /// Build the embed without validation (may exceed Discord limits)
    pub fn build(self) -> Embed {
        self.embed
    }

    /// Build the embed with validation
    pub fn try_build(self) -> Result<Embed, EmbedError> {
        self.validate()?;
        Ok(self.embed)
    }

    /// Validate the embed against Discord limits
    pub fn validate(&self) -> Result<(), EmbedError> {
        if let Some(ref title) = self.embed.title {
            if title.len() > limits::TITLE_MAX {
                return Err(EmbedError::TitleTooLong {
                    limit: limits::TITLE_MAX,
                    actual: title.len(),
                });
            }
        }

        if let Some(ref description) = self.embed.description {
            if description.len() > limits::DESCRIPTION_MAX {
                return Err(EmbedError::DescriptionTooLong {
                    limit: limits::DESCRIPTION_MAX,
                    actual: description.len(),
                });
            }
        }

        if self.embed.fields.len() > limits::FIELDS_MAX {
            return Err(EmbedError::TooManyFields {
                limit: limits::FIELDS_MAX,
                actual: self.embed.fields.len(),
            });
        }

        for field in &self.embed.fields {
            if field.name.len() > limits::FIELD_NAME_MAX {
                return Err(EmbedError::FieldNameTooLong {
                    limit: limits::FIELD_NAME_MAX,
                    actual: field.name.len(),
                });
            }
            if field.value.len() > limits::FIELD_VALUE_MAX {
                return Err(EmbedError::FieldValueTooLong {
                    limit: limits::FIELD_VALUE_MAX,
                    actual: field.value.len(),
                });
            }
        }

        if let Some(ref footer) = self.embed.footer {
            if footer.text.len() > limits::FOOTER_TEXT_MAX {
                return Err(EmbedError::FooterTextTooLong {
                    limit: limits::FOOTER_TEXT_MAX,
                    actual: footer.text.len(),
                });
            }
        }

        if let Some(ref author) = self.embed.author {
            if author.name.len() > limits::AUTHOR_NAME_MAX {
                return Err(EmbedError::AuthorNameTooLong {
                    limit: limits::AUTHOR_NAME_MAX,
                    actual: author.name.len(),
                });
            }
        }

        let total_chars = self.embed.total_chars();
        if total_chars > limits::TOTAL_CHARS_MAX {
            return Err(EmbedError::TotalSizeTooLarge {
                limit: limits::TOTAL_CHARS_MAX,
                actual: total_chars,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_builder_basic() {
        let embed = EmbedBuilder::new()
            .title("Test Title")
            .description("Test Description")
            .color(EmbedColor::Success)
            .build();

        assert_eq!(embed.title, Some("Test Title".to_string()));
        assert_eq!(embed.description, Some("Test Description".to_string()));
        assert_eq!(embed.color, Some(0x2ECC71));
        assert_eq!(embed.embed_type, Some("rich".to_string()));
    }

    #[test]
    fn test_embed_builder_with_fields() {
        let embed = EmbedBuilder::new()
            .title("Trade Executed")
            .field("Market", "Will BTC reach $100k?", true)
            .field("Side", "YES", true)
            .field("Amount", "$50.00", true)
            .build();

        assert_eq!(embed.fields.len(), 3);
        assert!(embed.fields[0].inline);
        assert_eq!(embed.fields[0].name, "Market");
    }

    #[test]
    fn test_embed_builder_with_footer_and_author() {
        let embed = EmbedBuilder::new()
            .author("Polysniper Bot")
            .footer("Powered by Polysniper")
            .timestamp_now()
            .build();

        assert!(embed.author.is_some());
        assert!(embed.footer.is_some());
        assert!(embed.timestamp.is_some());
    }

    #[test]
    fn test_embed_color_conversion() {
        assert_eq!(EmbedColor::Success.to_discord_color(), 0x2ECC71);
        assert_eq!(EmbedColor::Error.to_discord_color(), 0xE74C3C);
        assert_eq!(EmbedColor::Custom(0xFFFFFF).to_discord_color(), 0xFFFFFF);
    }

    #[test]
    fn test_embed_color_from_hex() {
        assert_eq!(
            EmbedColor::from_hex("#FF0000").unwrap().to_discord_color(),
            0xFF0000
        );
        assert_eq!(
            EmbedColor::from_hex("00FF00").unwrap().to_discord_color(),
            0x00FF00
        );
    }

    #[test]
    fn test_validation_title_too_long() {
        let long_title = "x".repeat(300);
        let builder = EmbedBuilder::new().title(long_title);

        let result = builder.validate();
        assert!(matches!(result, Err(EmbedError::TitleTooLong { .. })));
    }

    #[test]
    fn test_validation_too_many_fields() {
        let mut builder = EmbedBuilder::new();
        for i in 0..30 {
            builder = builder.field(format!("Field {}", i), "value", false);
        }

        let result = builder.validate();
        assert!(matches!(result, Err(EmbedError::TooManyFields { .. })));
    }

    #[test]
    fn test_json_serialization() {
        let embed = EmbedBuilder::new()
            .title("Test")
            .description("Description")
            .color(EmbedColor::Info)
            .field("Field1", "Value1", true)
            .footer("Footer text")
            .build();

        let json = serde_json::to_string(&embed).unwrap();
        assert!(json.contains("\"title\":\"Test\""));
        assert!(json.contains("\"type\":\"rich\""));
        assert!(json.contains("\"color\":3447003")); // 0x3498DB in decimal

        // Verify it can be deserialized back
        let deserialized: Embed = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.title, embed.title);
    }

    #[test]
    fn test_embed_total_chars() {
        let embed = EmbedBuilder::new()
            .title("Title") // 5 chars
            .description("Description") // 11 chars
            .footer("Footer") // 6 chars
            .author("Author") // 6 chars
            .field("Name", "Value", false) // 9 chars
            .build();

        assert_eq!(embed.total_chars(), 37);
    }

    #[test]
    fn test_inline_field_shorthand() {
        let embed = EmbedBuilder::new()
            .inline_field("Key", "Value")
            .build();

        assert_eq!(embed.fields.len(), 1);
        assert!(embed.fields[0].inline);
    }

    #[test]
    fn test_author_with_url() {
        let embed = EmbedBuilder::new()
            .author_with_url("Bot Name", "https://example.com")
            .build();

        let author = embed.author.unwrap();
        assert_eq!(author.name, "Bot Name");
        assert_eq!(author.url, Some("https://example.com".to_string()));
    }

    #[test]
    fn test_footer_with_icon() {
        let embed = EmbedBuilder::new()
            .footer_with_icon("Footer", "https://example.com/icon.png")
            .build();

        let footer = embed.footer.unwrap();
        assert_eq!(footer.text, "Footer");
        assert_eq!(
            footer.icon_url,
            Some("https://example.com/icon.png".to_string())
        );
    }

    #[test]
    fn test_image_and_thumbnail() {
        let embed = EmbedBuilder::new()
            .image("https://example.com/image.png")
            .thumbnail("https://example.com/thumb.png")
            .build();

        assert_eq!(
            embed.image.unwrap().url,
            "https://example.com/image.png"
        );
        assert_eq!(
            embed.thumbnail.unwrap().url,
            "https://example.com/thumb.png"
        );
    }

    #[test]
    fn test_try_build_valid() {
        let result = EmbedBuilder::new()
            .title("Valid Title")
            .description("Valid Description")
            .try_build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_try_build_invalid() {
        let result = EmbedBuilder::new()
            .title("x".repeat(300))
            .try_build();

        assert!(result.is_err());
    }
}
