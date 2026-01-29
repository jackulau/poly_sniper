---
id: discord-notifier-service
name: Discord Notifier Service with Event Bus Integration
wave: 2
priority: 1
dependencies: [discord-webhook-core, discord-embed-builder, discord-config-integration]
estimated_hours: 4
tags: [backend, service, discord, integration]
---

## Objective

Create a Discord notification service that subscribes to the event bus and sends formatted notifications for configured event types.

## Context

This is the integration layer that ties together the Discord client, embed builder, and configuration. It subscribes to `SystemEvent` from the event bus and sends appropriate Discord notifications. This is a Wave 2 task that depends on all Wave 1 tasks completing first.

## Implementation

1. Create `DiscordNotifier` service in `polysniper-discord` crate:
   ```rust
   pub struct DiscordNotifier {
       client: DiscordWebhookClient,
       config: DiscordConfig,
       event_rx: broadcast::Receiver<SystemEvent>,
   }
   ```

2. Implement event handling loop:
   - Subscribe to event bus on construction
   - Filter events based on config (`notify_on_trade`, `notify_on_error`, etc.)
   - Convert events to appropriate embeds using templates
   - Send via webhook client with rate limiting

3. Handle specific event types:
   - `SystemEvent::TradeExecuted` → Trade notification embed
   - `SystemEvent::ConnectionStatus` → Status change embed
   - Strategy errors → Error alert embed
   - Risk validation failures → Risk alert embed

4. Implement rate limiting:
   - Track messages sent per minute
   - Queue/drop messages exceeding rate limit
   - Use config `rate_limit_per_minute`

5. Add graceful shutdown handling:
   - Respond to shutdown signal
   - Flush pending notifications

6. Integrate into main application:
   - Initialize in `main.rs` if Discord enabled
   - Spawn as background task with `tokio::spawn`
   - Pass event bus subscriber

7. Add integration tests with mock event bus

## Acceptance Criteria

- [ ] `DiscordNotifier` compiles and initializes from config
- [ ] Service subscribes to event bus and receives events
- [ ] Configured event types trigger Discord notifications
- [ ] Rate limiting prevents exceeding Discord limits
- [ ] Dry-run mode logs but doesn't send real requests
- [ ] Graceful shutdown completes without hanging
- [ ] Integration with main.rs works when Discord enabled
- [ ] Tests verify event filtering and notification logic

## Files to Create/Modify

- `crates/polysniper-discord/src/notifier.rs` - Main notifier service
- `crates/polysniper-discord/src/lib.rs` - Export notifier
- `src/main.rs` - Initialize and spawn Discord notifier
- `Cargo.toml` (root) - Add polysniper-discord dependency to main binary

## Integration Points

- **Provides**: `DiscordNotifier` service, automatic event notifications
- **Consumes**:
  - `DiscordWebhookClient` from discord-webhook-core
  - `EmbedBuilder` and templates from discord-embed-builder
  - `DiscordConfig` from discord-config-integration
  - `EventBus` from polysniper-data
  - `SystemEvent` from polysniper-core
- **Conflicts**: Touches `main.rs` - coordinate carefully

## Reference Files

- `/Users/jacklau/Polysniper/crates/polysniper-data/src/event_bus.rs` - Event bus pattern
- `/Users/jacklau/Polysniper/crates/polysniper-core/src/events.rs` - SystemEvent types
- `/Users/jacklau/Polysniper/src/main.rs` - Service initialization pattern
- `/Users/jacklau/Polysniper/crates/polysniper-strategies/src/event_based.rs` - Event subscription pattern
