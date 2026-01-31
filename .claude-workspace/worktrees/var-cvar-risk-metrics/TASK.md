---
id: var-cvar-risk-metrics
name: VaR and CVaR Portfolio Risk Metrics
wave: 1
priority: 2
dependencies: []
estimated_hours: 5
tags: [risk, portfolio, metrics, var]
---

## Objective

Implement Value at Risk (VaR) and Conditional Value at Risk (CVaR/Expected Shortfall) calculations for portfolio-level risk monitoring and position sizing constraints.

## Context

The codebase tracks position-level risk (order size, position limits) but lacks portfolio-level risk metrics. VaR and CVaR provide:

- **VaR**: Maximum expected loss at a given confidence level (e.g., 95% VaR = we expect to lose no more than X with 95% probability)
- **CVaR**: Average loss when losses exceed VaR (captures tail risk)

These metrics enable:
- Portfolio-level risk limits (e.g., daily VaR < $500)
- Position sizing based on marginal VaR contribution
- Risk budgeting across strategies
- Regulatory-style risk reporting

Methods to implement:
1. **Historical VaR**: Percentile of historical returns
2. **Parametric VaR**: Assumes normal distribution (quick approximation)
3. **CVaR**: Average of returns beyond VaR threshold

## Implementation

1. Create `/crates/polysniper-risk/src/var.rs`:
   - `VaRCalculator` struct with configurable methods
   - `VaRConfig` with confidence levels and lookback periods
   - `VaRMethod` enum: Historical, Parametric, MonteCarlo (future)
   - Calculate portfolio VaR from position returns
   - Calculate CVaR (Expected Shortfall)
   - Calculate marginal VaR for individual positions

2. Add configuration types to `/crates/polysniper-core/src/types.rs`:
   - `VaRConfig` struct with:
     - `enabled: bool`
     - `method: VaRMethod` (default Historical)
     - `confidence_level: Decimal` (default 0.95)
     - `lookback_days: u32` (default 30)
     - `max_portfolio_var_usd: Decimal` (daily VaR limit)
     - `max_position_var_contribution_pct: Decimal` (default 0.25)
     - `use_cvar_for_limits: bool` (default false)

3. Add `VaRResult` struct:
   - `var_1d: Decimal` (1-day VaR)
   - `cvar_1d: Decimal` (1-day CVaR/ES)
   - `var_10d: Decimal` (10-day VaR, scaled)
   - `position_contributions: HashMap<MarketId, Decimal>`
   - `confidence_level: Decimal`
   - `calculated_at: DateTime<Utc>`

4. Integrate with `RiskManager` in `/crates/polysniper-risk/src/validator.rs`:
   - Add VaRCalculator as component
   - Check if new position would breach VaR limit
   - Calculate marginal VaR contribution of proposed trade
   - Optionally reject/modify if VaR limit exceeded

5. Add StateProvider support:
   - `get_portfolio_returns(lookback_days: u32) -> Vec<Decimal>` or compute from PnL history
   - `get_position_returns(market_id: &str, lookback_days: u32) -> Vec<Decimal>`

6. Update `/config/default.toml`:
   - Add `[risk.var]` section with defaults
   - Conservative daily VaR limit

7. Add metrics/observability:
   - Prometheus gauge for current VaR/CVaR
   - Event on VaR limit breach

8. Add comprehensive tests:
   - Test historical VaR calculation with known data
   - Test CVaR as average of tail losses
   - Test parametric VaR with normal distribution
   - Test marginal VaR contribution
   - Test 10-day VaR scaling (sqrt(10) rule)

## Acceptance Criteria

- [ ] Historical VaR correctly calculates percentile of returns
- [ ] CVaR correctly averages losses beyond VaR threshold
- [ ] Parametric VaR uses normal distribution approximation
- [ ] Marginal VaR contribution calculated per position
- [ ] VaR limit enforced in RiskManager validation
- [ ] 10-day VaR scaled correctly from 1-day VaR
- [ ] Configuration via TOML with hot-reload support
- [ ] All existing tests still pass
- [ ] New unit tests for VaR calculations (>80% coverage)
- [ ] No f64 usage - all Decimal arithmetic

## Files to Create/Modify

**Create:**
- `crates/polysniper-risk/src/var.rs` - VaR/CVaR calculator

**Modify:**
- `crates/polysniper-risk/src/lib.rs` - Export var module
- `crates/polysniper-risk/src/validator.rs` - Integrate VaR checks
- `crates/polysniper-core/src/types.rs` - Add VaRConfig, VaRMethod
- `config/default.toml` - Add VaR configuration section

## Integration Points

- **Provides**: Portfolio-level VaR/CVaR metrics and limit enforcement
- **Consumes**: Historical returns from persistence layer or computed from PnL
- **Conflicts**: None - adds new risk dimension

## Technical Notes

- Use `rust_decimal::Decimal` for all calculations
- Historical VaR requires sorting returns - use stable sort
- Parametric VaR needs mean and std dev of returns
- CVaR = average of returns below VaR percentile
- 10-day VaR ~ 1-day VaR * sqrt(10) (assumes IID returns)
- Consider caching VaR calculations (expensive to compute frequently)
- Marginal VaR: change in portfolio VaR from adding position

## Mathematical Formulas

**Historical VaR (95%)**:
- Sort returns ascending
- VaR = -1 * returns[floor(n * 0.05)]

**Parametric VaR (95%)**:
- VaR = mean - 1.645 * std_dev (z-score for 95%)

**CVaR (95%)**:
- CVaR = average of returns below VaR threshold

**10-day VaR**:
- VaR_10d = VaR_1d * sqrt(10)
