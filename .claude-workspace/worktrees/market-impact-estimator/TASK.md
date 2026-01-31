---
id: market-impact-estimator
name: Market Impact Modeling and Prediction
wave: 1
priority: 1
dependencies: []
estimated_hours: 5
tags: [analysis, core, execution]
---

## Objective

Implement an enhanced market impact estimator that predicts price impact before execution, considering historical patterns, current market conditions, and order flow dynamics to optimize entry timing.

## Context

Market impact is the price movement caused by a trade. The existing `depth_analyzer.rs` provides basic impact estimation from the orderbook, but we need a more sophisticated model that:
1. Considers temporary vs. permanent impact
2. Uses historical trade data for calibration
3. Accounts for market conditions (volatility, liquidity, time of day)
4. Predicts recovery time after impact

This module enhances execution quality by providing better impact predictions for order sizing and timing.

## Implementation

### 1. Create new module: `crates/polysniper-execution/src/market_impact.rs`

**Core Components:**

```rust
pub struct MarketImpactConfig {
    pub enabled: bool,
    pub model_type: ImpactModelType,
    pub lookback_trades: usize,             // Historical trades for calibration
    pub decay_halflife_secs: u64,           // Impact decay half-life
    pub permanent_impact_ratio: Decimal,     // Permanent vs temporary ratio (e.g., 0.3)
    pub volatility_adjustment: bool,
    pub liquidity_adjustment: bool,
    pub max_acceptable_impact_bps: Decimal,
}

pub enum ImpactModelType {
    SquareRoot,     // σ * sqrt(Q/V) - Almgren-Chriss style
    Linear,         // Simple linear model
    Logarithmic,    // Log impact model
    Adaptive,       // Uses historical data to calibrate
}

pub struct MarketImpactEstimator {
    config: MarketImpactConfig,
    // Historical impact observations per token
    impact_history: HashMap<TokenId, VecDeque<ImpactObservation>>,
    // Calibrated parameters per token
    calibrated_params: HashMap<TokenId, ImpactParameters>,
    // Current market conditions
    market_conditions: HashMap<TokenId, MarketConditions>,
}

pub struct ImpactObservation {
    pub token_id: TokenId,
    pub trade_size_usd: Decimal,
    pub pre_trade_price: Decimal,
    pub post_trade_price: Decimal,
    pub mid_price_before: Decimal,
    pub mid_price_after: Decimal,
    pub realized_impact_bps: Decimal,
    pub timestamp: DateTime<Utc>,
    pub recovery_price: Option<Decimal>,
    pub recovery_time_secs: Option<u64>,
    pub volatility_at_time: Option<Decimal>,
    pub liquidity_at_time: Option<Decimal>,
}

pub struct ImpactParameters {
    pub alpha: Decimal,             // Temporary impact coefficient
    pub beta: Decimal,              // Permanent impact coefficient
    pub gamma: Decimal,             // Size exponent (e.g., 0.5 for sqrt)
    pub decay_rate: Decimal,        // Impact decay rate
    pub r_squared: Decimal,         // Model fit quality
    pub last_calibrated: DateTime<Utc>,
    pub observation_count: usize,
}

pub struct MarketConditions {
    pub token_id: TokenId,
    pub current_volatility: Decimal,
    pub current_spread_bps: Decimal,
    pub current_depth_usd: Decimal,
    pub recent_volume_usd: Decimal,
    pub timestamp: DateTime<Utc>,
}

pub struct ImpactPrediction {
    pub token_id: TokenId,
    pub trade_size_usd: Decimal,
    pub expected_impact_bps: Decimal,
    pub impact_range: (Decimal, Decimal),   // Confidence interval
    pub temporary_impact_bps: Decimal,
    pub permanent_impact_bps: Decimal,
    pub expected_recovery_secs: u64,
    pub optimal_execution_time: Option<DateTime<Utc>>,
    pub confidence: Decimal,
    pub model_used: ImpactModelType,
    pub market_conditions: MarketConditions,
}

pub struct ImpactRecommendation {
    pub should_execute: bool,
    pub recommended_size: Decimal,          // May be reduced from requested
    pub recommended_slices: u32,            // Number of child orders
    pub recommended_interval_secs: u64,     // Time between slices
    pub reason: String,
}
```

### 2. Implement impact models

```rust
impl MarketImpactEstimator {
    /// Predict market impact for a proposed trade
    pub fn predict_impact(
        &self,
        token_id: &TokenId,
        size_usd: Decimal,
        side: Side,
        orderbook: &Orderbook,
    ) -> ImpactPrediction {
        let conditions = self.get_market_conditions(token_id, orderbook);
        let params = self.get_calibrated_params(token_id);

        match self.config.model_type {
            ImpactModelType::SquareRoot => self.sqrt_model_predict(size_usd, &conditions, &params),
            ImpactModelType::Linear => self.linear_model_predict(size_usd, &conditions, &params),
            ImpactModelType::Logarithmic => self.log_model_predict(size_usd, &conditions, &params),
            ImpactModelType::Adaptive => self.adaptive_model_predict(size_usd, &conditions, &params),
        }
    }

    /// Square root impact model: impact = α * σ * sqrt(Q/V)
    fn sqrt_model_predict(
        &self,
        size_usd: Decimal,
        conditions: &MarketConditions,
        params: &ImpactParameters,
    ) -> ImpactPrediction {
        let relative_size = size_usd / conditions.current_depth_usd;
        let sqrt_size = relative_size.sqrt().unwrap_or(relative_size);

        let base_impact = params.alpha * conditions.current_volatility * sqrt_size;
        let temporary = base_impact * (Decimal::ONE - self.config.permanent_impact_ratio);
        let permanent = base_impact * self.config.permanent_impact_ratio;

        // Adjust for current conditions
        let adjusted_impact = self.adjust_for_conditions(base_impact, conditions);

        ImpactPrediction { ... }
    }

    /// Record an observed trade impact for model calibration
    pub fn record_impact(
        &mut self,
        token_id: &TokenId,
        observation: ImpactObservation,
    ) {
        let history = self.impact_history
            .entry(token_id.clone())
            .or_insert_with(VecDeque::new);

        history.push_back(observation);

        // Trim to lookback limit
        while history.len() > self.config.lookback_trades {
            history.pop_front();
        }

        // Recalibrate if enough observations
        if history.len() >= 10 {
            self.calibrate_parameters(token_id);
        }
    }

    /// Calibrate model parameters from historical observations
    fn calibrate_parameters(&mut self, token_id: &TokenId) {
        let history = match self.impact_history.get(token_id) {
            Some(h) => h,
            None => return,
        };

        // Simple regression to fit parameters
        // In a real implementation, use proper regression
        let params = self.fit_impact_model(history);
        self.calibrated_params.insert(token_id.clone(), params);
    }

    /// Get execution recommendation based on impact prediction
    pub fn get_recommendation(
        &self,
        prediction: &ImpactPrediction,
    ) -> ImpactRecommendation {
        let max_impact = self.config.max_acceptable_impact_bps;

        if prediction.expected_impact_bps <= max_impact {
            return ImpactRecommendation {
                should_execute: true,
                recommended_size: prediction.trade_size_usd,
                recommended_slices: 1,
                recommended_interval_secs: 0,
                reason: "Impact within acceptable range".to_string(),
            };
        }

        // Calculate reduced size to stay within impact limit
        let reduction_factor = max_impact / prediction.expected_impact_bps;
        let reduced_size = prediction.trade_size_usd * reduction_factor;

        // Or split into slices
        let slices = (prediction.expected_impact_bps / max_impact)
            .ceil()
            .to_u32()
            .unwrap_or(5)
            .min(10);

        ImpactRecommendation {
            should_execute: true,
            recommended_size: reduced_size,
            recommended_slices: slices,
            recommended_interval_secs: prediction.expected_recovery_secs,
            reason: format!(
                "Impact {:.1} bps exceeds limit {:.1} bps",
                prediction.expected_impact_bps, max_impact
            ),
        }
    }
}
```

### 3. Create configuration file: `config/execution/market_impact.toml`

```toml
[execution.market_impact]
enabled = true

# Model configuration
model_type = "SquareRoot"       # SquareRoot, Linear, Logarithmic, Adaptive
lookback_trades = 100           # Historical trades for calibration
decay_halflife_secs = 60        # Impact decay half-life
permanent_impact_ratio = "0.3"  # 30% permanent, 70% temporary

# Adjustments
volatility_adjustment = true
liquidity_adjustment = true

# Limits
max_acceptable_impact_bps = "50"

# Default parameters (used before calibration)
[execution.market_impact.defaults]
alpha = "0.1"                   # Impact coefficient
beta = "0.03"                   # Permanent impact coefficient
gamma = "0.5"                   # Size exponent
```

### 4. Add tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> MarketImpactConfig { ... }
    fn test_orderbook() -> Orderbook { ... }

    #[test]
    fn test_sqrt_model_prediction() {
        // Larger trades have proportionally larger impact
    }

    #[test]
    fn test_impact_increases_with_size() {
        // Verify impact scales correctly with trade size
    }

    #[test]
    fn test_volatility_adjustment() {
        // Higher volatility increases expected impact
    }

    #[test]
    fn test_calibration_from_observations() {
        // Parameters improve with historical data
    }

    #[test]
    fn test_recommendation_within_limits() {
        // Low impact trades execute fully
    }

    #[test]
    fn test_recommendation_size_reduction() {
        // High impact trades get size reduction
    }

    #[tokio::test]
    async fn test_impact_recording_and_calibration() { ... }
}
```

### 5. Integration with existing depth_analyzer

Enhance integration with existing `depth_analyzer.rs`:
- Use `DepthAnalyzer::estimate_price_impact()` as input to the model
- Feed observed impacts back to calibrate the model
- Provide `MarketImpactEstimator` as optional enhancement to `OrderBuilder`

## Acceptance Criteria

- [ ] Square root impact model produces reasonable predictions
- [ ] Impact scales appropriately with trade size
- [ ] Volatility and liquidity adjustments work correctly
- [ ] Historical calibration improves predictions over time
- [ ] Recommendations correctly reduce size/slice for high impact
- [ ] Integration with existing depth_analyzer works
- [ ] Configuration loads from TOML file
- [ ] All tests pass (`cargo test -p polysniper-execution`)
- [ ] Code follows existing patterns
- [ ] No clippy warnings

## Files to Create/Modify

- `crates/polysniper-execution/src/market_impact.rs` - **CREATE** - Core impact modeling
- `crates/polysniper-execution/src/lib.rs` - **MODIFY** - Add `pub mod market_impact;` export
- `config/execution/market_impact.toml` - **CREATE** - Configuration file

## Integration Points

- **Provides**: `MarketImpactEstimator`, `ImpactPrediction`, `ImpactRecommendation` for execution optimization
- **Consumes**: Orderbook data, historical trade data
- **Enhances**: `OrderBuilder` and `DepthAnalyzer` with better impact predictions
- **Conflicts**: None - complements existing depth_analyzer
