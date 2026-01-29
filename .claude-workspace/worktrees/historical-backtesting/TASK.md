---
id: historical-backtesting
name: Historical Backtesting Engine
wave: 1
priority: 1
dependencies: []
estimated_hours: 6
tags: [backend, strategies, testing]
---

## Objective

Build a backtesting engine that replays historical market data to test strategies before live deployment.

## Context

Polysniper already has:
- Price snapshot storage in SQLite (`price_snapshots` table)
- In-memory price history in MarketCache (VecDeque with 1000 entries per token)
- Event-driven architecture via broadcast channels
- Strategy trait with `process_event()` method

This task creates a replay system that can feed historical data through strategies to evaluate performance.

## Implementation

### 1. Create Backtesting Crate Structure

Create new crate `crates/polysniper-backtest/`:

```
polysniper-backtest/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── engine.rs        # Core replay engine
│   ├── data_loader.rs   # Load historical data from SQLite
│   ├── results.rs       # Backtest results and metrics
│   └── config.rs        # Backtest configuration
```

### 2. Implement Data Loader (`data_loader.rs`)

- Load price snapshots from SQLite by time range
- Load orderbook snapshots if available
- Convert historical records to SystemEvent format
- Support filtering by market_id or token_id

### 3. Implement Backtest Engine (`engine.rs`)

- Accept strategy instances and historical data range
- Replay events in chronological order
- Track simulated positions and P&L
- Support configurable slippage and fee models
- Mock order execution (instant fill at price or realistic queue simulation)

### 4. Implement Results Collector (`results.rs`)

Calculate and report:
- Total P&L (realized + unrealized)
- Win rate (% of profitable trades)
- Sharpe ratio
- Max drawdown
- Trade count and average trade size
- Return on capital

### 5. Add CLI Integration

Add backtest command to main binary:
```bash
polysniper backtest --strategy target_price --from 2024-01-01 --to 2024-01-31
```

## Acceptance Criteria

- [ ] Can load historical price data from SQLite
- [ ] Can replay events through any strategy implementing Strategy trait
- [ ] Calculates accurate simulated P&L with configurable fees
- [ ] Reports key performance metrics (win rate, Sharpe, drawdown)
- [ ] CLI command runs backtest and outputs results
- [ ] Unit tests for data loader and results calculations
- [ ] Integration test with sample data

## Files to Create/Modify

**Create:**
- `crates/polysniper-backtest/Cargo.toml` - Crate manifest
- `crates/polysniper-backtest/src/lib.rs` - Crate exports
- `crates/polysniper-backtest/src/engine.rs` - Replay engine
- `crates/polysniper-backtest/src/data_loader.rs` - Historical data loader
- `crates/polysniper-backtest/src/results.rs` - Results and metrics
- `crates/polysniper-backtest/src/config.rs` - Configuration

**Modify:**
- `Cargo.toml` (workspace) - Add backtest crate to workspace
- `src/main.rs` - Add backtest CLI subcommand

## Integration Points

- **Provides**: Backtesting capability for all strategies
- **Consumes**:
  - `polysniper-persistence` for historical data
  - `polysniper-strategies` for strategy implementations
  - `polysniper-core` for types and events
- **Conflicts**: None - new crate with minimal overlap

## Technical Notes

- Use simulated time (not real clock) during replay
- Event timestamps drive the simulation clock
- Consider parallel backtest runs for parameter optimization
- Results should be exportable to JSON/CSV for analysis
