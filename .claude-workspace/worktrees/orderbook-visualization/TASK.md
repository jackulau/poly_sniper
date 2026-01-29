---
id: orderbook-visualization
name: Terminal Orderbook Depth Visualization
wave: 1
priority: 2
dependencies: []
estimated_hours: 5
tags: [tui, visualization, orderbook]
---

## Objective

Build terminal-based charts showing orderbook bid/ask depth levels using a TUI library.

## Context

Polysniper already has:
- Real-time orderbook data via WebSocket (`OrderbookUpdate` events)
- `Orderbook` type with bids/asks as `Vec<OrderbookLevel>` (price, size)
- MarketCache storing latest orderbook per token
- Structured logging but no visualization

This task adds a terminal UI for visualizing orderbook depth in real-time.

## Implementation

### 1. Add TUI Dependencies

In workspace `Cargo.toml`:
```toml
ratatui = "0.28"      # Modern TUI framework (tui-rs successor)
crossterm = "0.28"    # Terminal backend
```

### 2. Create TUI Crate

Create `crates/polysniper-tui/`:

```
polysniper-tui/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── app.rs           # Application state
│   ├── ui.rs            # Layout and rendering
│   ├── widgets/
│   │   ├── mod.rs
│   │   ├── orderbook.rs # Depth chart widget
│   │   ├── price.rs     # Price ticker widget
│   │   └── trades.rs    # Recent trades widget
│   └── event.rs         # Input handling
```

### 3. Implement Orderbook Depth Widget (`widgets/orderbook.rs`)

Display:
- Horizontal bar chart showing bid/ask depth
- Bids in green (left side), asks in red (right side)
- Price levels on Y-axis, cumulative size on X-axis
- Spread indicator at center
- Top 10-20 levels visible (configurable)

ASCII visualization concept:
```
     BIDS            │ SPREAD │            ASKS
████████████  0.45   │  0.02  │   0.47  ██████████████
██████████    0.44   │        │   0.48  ████████████████
████████      0.43   │        │   0.49  ██████████
██████        0.42   │        │   0.50  ████████████████████
```

### 4. Implement Price Ticker Widget (`widgets/price.rs`)

Display:
- Current mid price
- 24h change (if available)
- Bid/Ask spread
- Last trade price and size

### 5. Implement Recent Trades Widget (`widgets/trades.rs`)

Display:
- Last N trades executed by the bot
- Side (buy/sell), price, size, timestamp
- Color-coded by side

### 6. Main TUI Layout (`ui.rs`)

Layout structure:
```
┌─────────────────────────────────────────────┐
│  Market: Election 2024 | Token: YES         │
├─────────────────────────────────────────────┤
│                                             │
│           Orderbook Depth Chart             │
│                                             │
├──────────────────────┬──────────────────────┤
│   Price Ticker       │   Recent Trades      │
│   Mid: 0.456         │   BUY  0.45 100      │
│   Spread: 0.02       │   SELL 0.46  50      │
└──────────────────────┴──────────────────────┘
```

### 7. Integration with Main App

Add TUI mode to main binary:
```bash
polysniper tui --market <market_id>
```

- Subscribe to orderbook updates for selected market
- Render at 10 FPS (100ms refresh)
- Support keyboard navigation (q=quit, arrows=scroll)

## Acceptance Criteria

- [ ] Orderbook depth chart renders correctly in terminal
- [ ] Bids and asks visually distinguished (color-coded)
- [ ] Updates in real-time as WebSocket data arrives
- [ ] Price ticker shows current mid price and spread
- [ ] Recent trades list shows bot's executed trades
- [ ] Keyboard controls work (quit, scroll, select market)
- [ ] Graceful terminal restore on exit
- [ ] Works on standard 80x24 terminal minimum

## Files to Create/Modify

**Create:**
- `crates/polysniper-tui/Cargo.toml` - Crate manifest with ratatui
- `crates/polysniper-tui/src/lib.rs` - Crate exports
- `crates/polysniper-tui/src/app.rs` - TUI application state
- `crates/polysniper-tui/src/ui.rs` - Main layout renderer
- `crates/polysniper-tui/src/event.rs` - Input event handling
- `crates/polysniper-tui/src/widgets/mod.rs` - Widget exports
- `crates/polysniper-tui/src/widgets/orderbook.rs` - Depth chart
- `crates/polysniper-tui/src/widgets/price.rs` - Price ticker
- `crates/polysniper-tui/src/widgets/trades.rs` - Trades list

**Modify:**
- `Cargo.toml` (workspace) - Add tui crate and dependencies
- `src/main.rs` - Add TUI subcommand

## Integration Points

- **Provides**: Visual monitoring of orderbook state
- **Consumes**:
  - `polysniper-data` for orderbook and price data
  - `polysniper-core` for event types
- **Conflicts**: None - entirely new visualization layer

## Technical Notes

- Use crossterm backend for cross-platform support
- Handle terminal resize events
- Consider async rendering with tokio
- Test on both light and dark terminal themes
- Add fallback for terminals without color support
