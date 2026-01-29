---
id: discord-webhook-core
name: Discord Webhook Core Types and Client
wave: 1
priority: 1
dependencies: []
estimated_hours: 3
tags: [backend, core, discord]
---

## Objective

Create the core Discord webhook client with HTTP request functionality and type definitions for Discord's webhook API.

## Context

This task establishes the foundational types and HTTP client for sending Discord webhooks. It follows the existing patterns in the codebase (using `reqwest` for HTTP, `serde` for serialization). This is a Wave 1 task with no dependencies - other Discord tasks will build on this foundation.

## Implementation

1. Create new crate `polysniper-discord` in `crates/` directory
2. Define Discord webhook types:
   - `DiscordWebhook` struct with `webhook_url`
   - `DiscordMessage` struct for basic messages
   - `DiscordEmbed` struct for rich embeds (placeholder, will be expanded in embed-builder task)
   - `WebhookResponse` for API responses
   - `DiscordError` error type using `thiserror`
3. Implement `DiscordWebhookClient`:
   - Constructor with `reqwest::Client` initialization (follow `GammaClient` pattern)
   - `send_message()` async method for basic webhook posts
   - Retry logic following `OrderSubmitter` pattern
   - Rate limiting awareness (429 response handling)
4. Add crate to workspace `Cargo.toml`
5. Add unit tests for serialization and error handling

## Acceptance Criteria

- [ ] `polysniper-discord` crate compiles without errors
- [ ] Types serialize correctly to Discord webhook JSON format
- [ ] Client handles HTTP errors gracefully with proper error types
- [ ] Rate limit (429) responses are handled with backoff
- [ ] Dry-run mode support for testing without real Discord requests
- [ ] Unit tests for type serialization pass

## Files to Create/Modify

- `crates/polysniper-discord/Cargo.toml` - New crate manifest
- `crates/polysniper-discord/src/lib.rs` - Module exports
- `crates/polysniper-discord/src/types.rs` - Discord webhook types
- `crates/polysniper-discord/src/client.rs` - HTTP client implementation
- `crates/polysniper-discord/src/error.rs` - Error types
- `Cargo.toml` - Add workspace member

## Integration Points

- **Provides**: `DiscordWebhookClient`, core types for Discord API
- **Consumes**: `reqwest` HTTP client (existing workspace dependency)
- **Conflicts**: None - this is a new crate

## Reference Files

- `/Users/jacklau/Polysniper/crates/polysniper-data/src/gamma_client.rs` - HTTP client pattern
- `/Users/jacklau/Polysniper/crates/polysniper-execution/src/submitter.rs` - Retry logic pattern
- `/Users/jacklau/Polysniper/crates/polysniper-core/src/error.rs` - Error type pattern
