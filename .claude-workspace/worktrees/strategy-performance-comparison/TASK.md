---
id: strategy-performance-comparison
name: Strategy Performance Comparison Dashboard
wave: 2
priority: 2
dependencies: [pnl-tracking-sqlite]
estimated_hours: 4
tags: [metrics, strategies, analysis]
---

## Objective

Build a system to track and compare which strategies are profitable, with metrics and ranking.

## Context

Polysniper already has:
- 4 strategies (target_price, price_spike, new_market, event_based)
- Trades table with `strategy_id` field
- Daily P&L tracking
- Strategy state persistence

This task adds aggregated performance metrics per strategy and comparison tools.

## Implementation

### 1. Create Strategy Metrics Repository

Create `crates/polysniper-persistence/src/repositories/strategy_metrics.rs`:

Track per strategy:
- Total trades
- Win count / Loss count
- Gross profit / Gross loss
- Net P&L
- Win rate
- Profit factor
- Average trade size
- Average win / Average loss
- Max consecutive wins / losses
- Largest win / Largest loss
- Sharpe ratio (if enough data)

### 2. Add Strategy Performance View

SQL view or query for easy access:
```sql
SELECT
    strategy_id,
    COUNT(*) as total_trades,
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
    SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END) as gross_profit,
    SUM(CASE WHEN realized_pnl < 0 THEN realized_pnl ELSE 0 END) as gross_loss,
    SUM(realized_pnl) as net_pnl
FROM trades
GROUP BY strategy_id;
```

### 3. Implement Comparison CLI Command

Add command:
```bash
polysniper stats strategies
```

Output table:
```
Strategy Performance (Last 30 days)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy       | Trades | Win% | Net P&L | Profit Factor | Sharpe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
target_price   |    45  | 62%  | +$234   |     1.65      |  1.23
price_spike    |    23  | 48%  | -$45    |     0.92      | -0.34
new_market     |    12  | 75%  | +$156   |     3.00      |  2.10
event_based    |     8  | 50%  | +$12    |     1.08      |  0.45
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL          |    88  | 58%  | +$357   |     1.42      |  0.86
```

### 4. Add Time-Based Filtering

Support filtering:
- `--period today|week|month|all`
- `--from YYYY-MM-DD --to YYYY-MM-DD`
- `--strategy <strategy_id>` for single strategy detail

### 5. Add Strategy Ranking

Rank strategies by configurable metric:
```bash
polysniper stats strategies --rank-by profit_factor
polysniper stats strategies --rank-by sharpe
polysniper stats strategies --rank-by net_pnl
```

### 6. Export to JSON/CSV

```bash
polysniper stats strategies --format json > strategy_stats.json
polysniper stats strategies --format csv > strategy_stats.csv
```

### 7. Optional: Add to TUI

If orderbook-visualization is complete, add strategy stats panel:
- Real-time P&L per strategy
- Toggle view with keyboard shortcut

## Acceptance Criteria

- [ ] Strategy metrics calculated correctly from trades table
- [ ] CLI displays formatted comparison table
- [ ] Time-based filtering works (today/week/month/custom)
- [ ] Strategies rankable by different metrics
- [ ] Export to JSON and CSV formats
- [ ] Metrics update as new trades execute
- [ ] Unit tests for metric calculations

## Files to Create/Modify

**Create:**
- `crates/polysniper-persistence/src/repositories/strategy_metrics.rs` - Metrics queries
- `src/commands/stats.rs` - Stats CLI command (if using subcommand pattern)

**Modify:**
- `crates/polysniper-persistence/src/lib.rs` - Export strategy_metrics
- `crates/polysniper-persistence/src/repositories/mod.rs` - Add module
- `src/main.rs` - Add stats subcommand

## Integration Points

- **Provides**: Performance insights for strategy optimization
- **Consumes**:
  - `polysniper-persistence` trades and P&L data (from pnl-tracking-sqlite)
  - `polysniper-strategies` for strategy metadata
- **Conflicts**: None - queries existing data

## Dependencies

**Depends on:** `pnl-tracking-sqlite` - Needs complete P&L tracking to calculate metrics

This task should start after pnl-tracking-sqlite is complete, as it relies on:
- Trades having realized_pnl populated
- Position history for accurate calculations

## Technical Notes

- Use window functions for Sharpe ratio calculation if SQLite supports
- Cache metrics with TTL to avoid recomputation on every query
- Consider materialized view pattern for complex aggregations
- Handle strategies with zero trades gracefully
