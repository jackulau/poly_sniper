---
id: discord-embed-builder
name: Discord Rich Embed Builder
wave: 1
priority: 2
dependencies: []
estimated_hours: 3
tags: [backend, discord, embeds]
---

## Objective

Create a fluent builder API for constructing Discord rich embeds with full support for Discord's embed specification.

## Context

Discord embeds allow rich, formatted messages with colors, fields, images, and more. This task creates a type-safe builder pattern for constructing embeds that match Discord's API specification. This is a Wave 1 task that can run in parallel with `discord-webhook-core` - the types will be integrated later.

## Implementation

1. Create embed types module in `polysniper-discord` crate (or standalone file if crate doesn't exist yet):
   - `Embed` struct with all Discord embed fields
   - `EmbedField` for embed fields (name, value, inline)
   - `EmbedAuthor` for author information
   - `EmbedFooter` for footer with text and icon
   - `EmbedImage` / `EmbedThumbnail` for images
   - `EmbedColor` helper for hex color conversion

2. Implement `EmbedBuilder` with fluent API:
   ```rust
   EmbedBuilder::new()
       .title("Trade Executed")
       .description("BTC prediction market")
       .color(EmbedColor::Success)
       .field("Market", "Will BTC reach $100k?", true)
       .field("Side", "YES", true)
       .field("Amount", "$50.00", true)
       .footer("Polysniper Bot")
       .timestamp(Utc::now())
       .build()
   ```

3. Create preset embed templates for common notifications:
   - `TradeEmbed::from_signal(signal)` - Trade execution notification
   - `ErrorEmbed::from_error(error)` - Error alerts
   - `RiskEmbed::from_decision(decision)` - Risk validation results
   - `StatusEmbed::new()` - Connection/status updates

4. Add comprehensive unit tests for builder and serialization

## Acceptance Criteria

- [ ] `EmbedBuilder` compiles and creates valid embed structures
- [ ] All Discord embed fields supported (title, description, color, fields, author, footer, image, thumbnail, timestamp)
- [ ] Builder validates field limits (title: 256 chars, description: 4096 chars, fields: 25 max)
- [ ] Preset templates generate appropriate embeds for Polysniper events
- [ ] JSON serialization matches Discord webhook API format
- [ ] Unit tests cover all builder methods and edge cases

## Files to Create/Modify

- `crates/polysniper-discord/src/embed.rs` - Embed types and builder
- `crates/polysniper-discord/src/templates.rs` - Preset embed templates
- `crates/polysniper-discord/src/lib.rs` - Export new modules

## Integration Points

- **Provides**: `EmbedBuilder`, `Embed` type, preset templates
- **Consumes**: `chrono::DateTime` for timestamps, `serde` for serialization
- **Conflicts**: May touch `polysniper-discord/src/lib.rs` - coordinate with discord-webhook-core

## Reference Files

- Discord Embed specification: https://discord.com/developers/docs/resources/message#embed-object
- `/Users/jacklau/Polysniper/crates/polysniper-core/src/types.rs` - Type patterns
- `/Users/jacklau/Polysniper/crates/polysniper-execution/src/order_builder.rs` - Builder pattern reference
