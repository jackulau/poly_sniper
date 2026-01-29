---
id: pnl-tracking-sqlite
name: Complete P&L Tracking with SQLite
wave: 1
priority: 1
dependencies: []
estimated_hours: 5
tags: [backend, persistence, metrics]
---

## Objective

Complete the P&L tracking system with SQLite persistence, trade history, and performance metrics calculation.

## Context

Polysniper already has:
- `daily_pnl` repository with win_count, loss_count, trade_count fields
- `trades` repository storing executed trades with realized_pnl field
- `orders` repository tracking order lifecycle
- Position tracking in MarketCache with realized/unrealized P&L fields
- Risk manager that checks daily loss limits

This task completes the integration and adds missing calculations.

## Implementation

### 1. Wire P&L Recording to Main Loop

In `src/main.rs`, after successful trade execution:
- Record trade to trades repository with realized_pnl
- Update daily_pnl with `add_realized_pnl()`
- Update position in MarketCache

### 2. Implement Position P&L Calculator

Create `crates/polysniper-persistence/src/pnl_calculator.rs`:
- Calculate realized P&L on position close/reduce
- Calculate unrealized P&L from current prices
- Support FIFO and average cost basis methods
- Track fees separately

### 3. Add Performance Metrics Repository

Create `crates/polysniper-persistence/src/repositories/performance_metrics.rs`:

Store and calculate:
- Total realized P&L
- Total unrealized P&L
- Win rate = win_count / (win_count + loss_count)
- Profit factor = gross_profit / gross_loss
- Average win/loss size
- Largest win/loss
- ROI = total_pnl / starting_capital

### 4. Add Position History Table

New table `position_history`:
```sql
CREATE TABLE position_history (
    id INTEGER PRIMARY KEY,
    market_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price DECIMAL NOT NULL,
    exit_price DECIMAL,
    size DECIMAL NOT NULL,
    realized_pnl DECIMAL,
    opened_at TIMESTAMP NOT NULL,
    closed_at TIMESTAMP,
    strategy_id TEXT
);
```

### 5. Add API Methods for P&L Queries

In repositories:
- `get_pnl_by_date_range(from, to)`
- `get_pnl_by_strategy(strategy_id)`
- `get_pnl_by_market(market_id)`
- `get_cumulative_pnl()`
- `get_equity_curve()` - daily ending balances

## Acceptance Criteria

- [ ] Trades are recorded to SQLite after execution
- [ ] Daily P&L updates automatically on each trade
- [ ] Realized P&L calculated correctly on position close
- [ ] Unrealized P&L reflects current market prices
- [ ] Position history tracks entry/exit for analysis
- [ ] Performance metrics queryable by date range/strategy
- [ ] Unit tests for P&L calculations
- [ ] Integration test: execute trades → query P&L

## Files to Create/Modify

**Create:**
- `crates/polysniper-persistence/src/pnl_calculator.rs` - P&L calculation logic
- `crates/polysniper-persistence/src/repositories/performance_metrics.rs` - Metrics queries
- `crates/polysniper-persistence/src/repositories/position_history.rs` - Position lifecycle

**Modify:**
- `crates/polysniper-persistence/src/lib.rs` - Export new modules
- `crates/polysniper-persistence/src/database.rs` - Add migrations
- `src/main.rs` - Wire P&L recording after trades
- `crates/polysniper-data/src/market_cache.rs` - Integrate P&L updates

## Integration Points

- **Provides**: Complete P&L data for strategy comparison and visualization
- **Consumes**:
  - `polysniper-core` for types (Position, Trade)
  - `polysniper-data` for current prices
- **Conflicts**: Minor overlap with `market_cache.rs` - coordinate with any concurrent changes

## Technical Notes

- Use transactions for trade + P&L update atomicity
- Handle partial fills correctly
- Consider timezone handling for daily P&L resets
- Fee calculation: maker fee = 0%, taker fee = ~0.1% (configurable)
