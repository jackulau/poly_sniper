//! Market Impact Modeling and Prediction
//!
//! Implements an enhanced market impact estimator that predicts price impact
//! before execution, considering historical patterns, current market conditions,
//! and order flow dynamics to optimize entry timing.
//!
//! Key features:
//! - Multiple impact models: Square root (Almgren-Chriss), Linear, Logarithmic, Adaptive
//! - Temporary vs. permanent impact decomposition
//! - Historical trade data calibration
//! - Market condition adjustments (volatility, liquidity)
//! - Recovery time prediction
//! - Execution recommendations for order sizing

use chrono::{DateTime, Utc};
use polysniper_core::{Orderbook, Side, TokenId};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tracing::{debug, instrument};

/// Configuration for the market impact estimator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactConfig {
    /// Whether market impact estimation is enabled
    pub enabled: bool,
    /// Type of impact model to use
    pub model_type: ImpactModelType,
    /// Number of historical trades to use for calibration
    pub lookback_trades: usize,
    /// Impact decay half-life in seconds
    pub decay_halflife_secs: u64,
    /// Ratio of permanent to total impact (e.g., 0.3 = 30% permanent)
    pub permanent_impact_ratio: Decimal,
    /// Whether to adjust for current volatility
    pub volatility_adjustment: bool,
    /// Whether to adjust for current liquidity
    pub liquidity_adjustment: bool,
    /// Maximum acceptable impact in basis points
    pub max_acceptable_impact_bps: Decimal,
    /// Default alpha (temporary impact coefficient)
    pub default_alpha: Decimal,
    /// Default beta (permanent impact coefficient)
    pub default_beta: Decimal,
    /// Default gamma (size exponent, e.g., 0.5 for sqrt)
    pub default_gamma: Decimal,
}

impl Default for MarketImpactConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_type: ImpactModelType::SquareRoot,
            lookback_trades: 100,
            decay_halflife_secs: 60,
            permanent_impact_ratio: dec!(0.3),
            volatility_adjustment: true,
            liquidity_adjustment: true,
            max_acceptable_impact_bps: dec!(50),
            default_alpha: dec!(0.1),
            default_beta: dec!(0.03),
            default_gamma: dec!(0.5),
        }
    }
}

/// Type of impact model to use for predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImpactModelType {
    /// Square root model: σ * sqrt(Q/V) - Almgren-Chriss style
    #[default]
    SquareRoot,
    /// Simple linear model
    Linear,
    /// Logarithmic impact model
    Logarithmic,
    /// Adaptive model using historical data for calibration
    Adaptive,
}

/// Observation of actual trade impact for calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactObservation {
    /// Token that was traded
    pub token_id: TokenId,
    /// Trade size in USD
    pub trade_size_usd: Decimal,
    /// Price at which the trade executed
    pub pre_trade_price: Decimal,
    /// Price immediately after trade
    pub post_trade_price: Decimal,
    /// Mid price before the trade
    pub mid_price_before: Decimal,
    /// Mid price after the trade
    pub mid_price_after: Decimal,
    /// Realized impact in basis points
    pub realized_impact_bps: Decimal,
    /// When the trade occurred
    pub timestamp: DateTime<Utc>,
    /// Price after recovery (if observed)
    pub recovery_price: Option<Decimal>,
    /// Time to recover in seconds (if observed)
    pub recovery_time_secs: Option<u64>,
    /// Volatility at time of trade
    pub volatility_at_time: Option<Decimal>,
    /// Liquidity at time of trade (depth in USD)
    pub liquidity_at_time: Option<Decimal>,
}

/// Calibrated parameters for the impact model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactParameters {
    /// Temporary impact coefficient
    pub alpha: Decimal,
    /// Permanent impact coefficient
    pub beta: Decimal,
    /// Size exponent (e.g., 0.5 for sqrt model)
    pub gamma: Decimal,
    /// Impact decay rate per second
    pub decay_rate: Decimal,
    /// R-squared fit quality (0.0 to 1.0)
    pub r_squared: Decimal,
    /// When parameters were last calibrated
    pub last_calibrated: DateTime<Utc>,
    /// Number of observations used for calibration
    pub observation_count: usize,
}

impl ImpactParameters {
    /// Create default parameters
    pub fn default_params(config: &MarketImpactConfig) -> Self {
        Self {
            alpha: config.default_alpha,
            beta: config.default_beta,
            gamma: config.default_gamma,
            decay_rate: Decimal::ONE / Decimal::from(config.decay_halflife_secs),
            r_squared: Decimal::ZERO,
            last_calibrated: Utc::now(),
            observation_count: 0,
        }
    }
}

/// Current market conditions for a token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    /// Token ID
    pub token_id: TokenId,
    /// Current volatility estimate (annualized %)
    pub current_volatility: Decimal,
    /// Current bid-ask spread in basis points
    pub current_spread_bps: Decimal,
    /// Current orderbook depth in USD (within reasonable price range)
    pub current_depth_usd: Decimal,
    /// Recent trading volume in USD
    pub recent_volume_usd: Decimal,
    /// When conditions were measured
    pub timestamp: DateTime<Utc>,
}

impl MarketConditions {
    /// Extract market conditions from an orderbook
    pub fn from_orderbook(orderbook: &Orderbook) -> Self {
        let best_bid = orderbook.best_bid().unwrap_or(Decimal::ZERO);
        let best_ask = orderbook.best_ask().unwrap_or(Decimal::ONE);
        let mid_price = orderbook.mid_price().unwrap_or(dec!(0.5));

        // Calculate spread in basis points
        let spread = if mid_price > Decimal::ZERO {
            ((best_ask - best_bid) / mid_price) * dec!(10000)
        } else {
            Decimal::ZERO
        };

        // Calculate total depth within 5% of mid price
        let bid_depth: Decimal = orderbook
            .bids
            .iter()
            .filter(|l| l.price >= mid_price * dec!(0.95))
            .map(|l| l.size * l.price)
            .sum();

        let ask_depth: Decimal = orderbook
            .asks
            .iter()
            .filter(|l| l.price <= mid_price * dec!(1.05))
            .map(|l| l.size * l.price)
            .sum();

        Self {
            token_id: orderbook.token_id.clone(),
            current_volatility: dec!(0.1), // Default 10% vol, should be provided externally
            current_spread_bps: spread,
            current_depth_usd: bid_depth + ask_depth,
            recent_volume_usd: Decimal::ZERO, // Should be provided externally
            timestamp: orderbook.timestamp,
        }
    }

    /// Create with explicit volatility
    pub fn with_volatility(mut self, volatility: Decimal) -> Self {
        self.current_volatility = volatility;
        self
    }

    /// Create with explicit volume
    pub fn with_volume(mut self, volume: Decimal) -> Self {
        self.recent_volume_usd = volume;
        self
    }
}

/// Prediction of market impact for a proposed trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactPrediction {
    /// Token being traded
    pub token_id: TokenId,
    /// Proposed trade size in USD
    pub trade_size_usd: Decimal,
    /// Expected total impact in basis points
    pub expected_impact_bps: Decimal,
    /// Confidence interval for impact (low, high)
    pub impact_range: (Decimal, Decimal),
    /// Expected temporary impact in basis points
    pub temporary_impact_bps: Decimal,
    /// Expected permanent impact in basis points
    pub permanent_impact_bps: Decimal,
    /// Expected time for temporary impact to decay (seconds)
    pub expected_recovery_secs: u64,
    /// Optimal time to execute (if applicable)
    pub optimal_execution_time: Option<DateTime<Utc>>,
    /// Confidence in the prediction (0.0 to 1.0)
    pub confidence: Decimal,
    /// Model used for prediction
    pub model_used: ImpactModelType,
    /// Market conditions at time of prediction
    pub market_conditions: MarketConditions,
}

/// Recommendation for how to execute a trade given impact prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactRecommendation {
    /// Whether the trade should be executed
    pub should_execute: bool,
    /// Recommended size (may be reduced from requested)
    pub recommended_size: Decimal,
    /// Recommended number of child order slices
    pub recommended_slices: u32,
    /// Recommended interval between slices in seconds
    pub recommended_interval_secs: u64,
    /// Reason for the recommendation
    pub reason: String,
}

/// Market impact estimator using various models
pub struct MarketImpactEstimator {
    config: MarketImpactConfig,
    /// Historical impact observations per token
    impact_history: HashMap<TokenId, VecDeque<ImpactObservation>>,
    /// Calibrated model parameters per token
    calibrated_params: HashMap<TokenId, ImpactParameters>,
    /// Current market conditions per token
    market_conditions: HashMap<TokenId, MarketConditions>,
}

impl MarketImpactEstimator {
    /// Create a new market impact estimator with the given configuration
    pub fn new(config: MarketImpactConfig) -> Self {
        Self {
            config,
            impact_history: HashMap::new(),
            calibrated_params: HashMap::new(),
            market_conditions: HashMap::new(),
        }
    }

    /// Create an estimator with default configuration
    pub fn with_defaults() -> Self {
        Self::new(MarketImpactConfig::default())
    }

    /// Get the configuration
    pub fn config(&self) -> &MarketImpactConfig {
        &self.config
    }

    /// Update market conditions for a token
    pub fn update_conditions(&mut self, conditions: MarketConditions) {
        self.market_conditions
            .insert(conditions.token_id.clone(), conditions);
    }

    /// Get market conditions for a token
    fn get_market_conditions(&self, token_id: &TokenId, orderbook: &Orderbook) -> MarketConditions {
        // Start with orderbook-derived conditions
        let mut conditions = MarketConditions::from_orderbook(orderbook);

        // Override with stored conditions if available (they may have volatility data)
        if let Some(stored) = self.market_conditions.get(token_id) {
            conditions.current_volatility = stored.current_volatility;
            conditions.recent_volume_usd = stored.recent_volume_usd;
        }

        conditions
    }

    /// Get calibrated parameters for a token, or defaults if not calibrated
    fn get_calibrated_params(&self, token_id: &TokenId) -> ImpactParameters {
        self.calibrated_params
            .get(token_id)
            .cloned()
            .unwrap_or_else(|| ImpactParameters::default_params(&self.config))
    }

    /// Predict market impact for a proposed trade
    #[instrument(skip(self, orderbook), fields(token_id = %orderbook.token_id, size = %size_usd))]
    pub fn predict_impact(
        &self,
        token_id: &TokenId,
        size_usd: Decimal,
        _side: Side,
        orderbook: &Orderbook,
    ) -> ImpactPrediction {
        let conditions = self.get_market_conditions(token_id, orderbook);
        let params = self.get_calibrated_params(token_id);

        match self.config.model_type {
            ImpactModelType::SquareRoot => {
                self.sqrt_model_predict(size_usd, &conditions, &params)
            }
            ImpactModelType::Linear => self.linear_model_predict(size_usd, &conditions, &params),
            ImpactModelType::Logarithmic => self.log_model_predict(size_usd, &conditions, &params),
            ImpactModelType::Adaptive => {
                self.adaptive_model_predict(token_id, size_usd, &conditions, &params)
            }
        }
    }

    /// Square root impact model: impact = α * σ * sqrt(Q/V)
    ///
    /// Based on Almgren-Chriss model where impact scales with square root of relative size.
    fn sqrt_model_predict(
        &self,
        size_usd: Decimal,
        conditions: &MarketConditions,
        params: &ImpactParameters,
    ) -> ImpactPrediction {
        // Avoid division by zero
        let depth = conditions.current_depth_usd.max(dec!(1));

        // Relative size: Q/V
        let relative_size = size_usd / depth;

        // Square root of relative size
        let sqrt_size = self.decimal_sqrt(relative_size);

        // Base impact = α * σ * sqrt(Q/V)
        let base_impact = params.alpha * conditions.current_volatility * sqrt_size * dec!(10000); // Convert to bps

        // Apply adjustments
        let adjusted_impact = self.adjust_for_conditions(base_impact, conditions);

        // Split into temporary and permanent
        let permanent = adjusted_impact * self.config.permanent_impact_ratio;
        let temporary = adjusted_impact - permanent;

        // Calculate recovery time based on decay half-life
        let recovery_secs = self.config.decay_halflife_secs * 3; // ~87.5% recovery

        // Confidence based on parameter calibration quality
        let confidence = params.r_squared.max(dec!(0.3)); // At least 0.3 for default

        // Impact range: ±30% uncertainty
        let uncertainty = adjusted_impact * dec!(0.3);
        let impact_range = (
            (adjusted_impact - uncertainty).max(Decimal::ZERO),
            adjusted_impact + uncertainty,
        );

        debug!(
            base_impact = %base_impact,
            adjusted = %adjusted_impact,
            temporary = %temporary,
            permanent = %permanent,
            "Square root model prediction"
        );

        ImpactPrediction {
            token_id: conditions.token_id.clone(),
            trade_size_usd: size_usd,
            expected_impact_bps: adjusted_impact,
            impact_range,
            temporary_impact_bps: temporary,
            permanent_impact_bps: permanent,
            expected_recovery_secs: recovery_secs,
            optimal_execution_time: None,
            confidence,
            model_used: ImpactModelType::SquareRoot,
            market_conditions: conditions.clone(),
        }
    }

    /// Linear impact model: impact = α * (Q/V)
    fn linear_model_predict(
        &self,
        size_usd: Decimal,
        conditions: &MarketConditions,
        params: &ImpactParameters,
    ) -> ImpactPrediction {
        let depth = conditions.current_depth_usd.max(dec!(1));
        let relative_size = size_usd / depth;

        // Linear impact
        let base_impact = params.alpha * relative_size * dec!(10000);
        let adjusted_impact = self.adjust_for_conditions(base_impact, conditions);

        let permanent = adjusted_impact * self.config.permanent_impact_ratio;
        let temporary = adjusted_impact - permanent;
        let recovery_secs = self.config.decay_halflife_secs * 3;

        let confidence = params.r_squared.max(dec!(0.3));
        let uncertainty = adjusted_impact * dec!(0.35);

        ImpactPrediction {
            token_id: conditions.token_id.clone(),
            trade_size_usd: size_usd,
            expected_impact_bps: adjusted_impact,
            impact_range: (
                (adjusted_impact - uncertainty).max(Decimal::ZERO),
                adjusted_impact + uncertainty,
            ),
            temporary_impact_bps: temporary,
            permanent_impact_bps: permanent,
            expected_recovery_secs: recovery_secs,
            optimal_execution_time: None,
            confidence,
            model_used: ImpactModelType::Linear,
            market_conditions: conditions.clone(),
        }
    }

    /// Logarithmic impact model: impact = α * log(1 + Q/V)
    fn log_model_predict(
        &self,
        size_usd: Decimal,
        conditions: &MarketConditions,
        params: &ImpactParameters,
    ) -> ImpactPrediction {
        let depth = conditions.current_depth_usd.max(dec!(1));
        let relative_size = size_usd / depth;

        // Log impact: α * ln(1 + Q/V)
        let log_arg = (Decimal::ONE + relative_size)
            .to_string()
            .parse::<f64>()
            .unwrap_or(1.0);
        let log_val = Decimal::from_str_exact(&log_arg.ln().to_string()).unwrap_or(Decimal::ZERO);

        let base_impact = params.alpha * log_val * dec!(10000);
        let adjusted_impact = self.adjust_for_conditions(base_impact, conditions);

        let permanent = adjusted_impact * self.config.permanent_impact_ratio;
        let temporary = adjusted_impact - permanent;
        let recovery_secs = self.config.decay_halflife_secs * 3;

        let confidence = params.r_squared.max(dec!(0.3));
        let uncertainty = adjusted_impact * dec!(0.35);

        ImpactPrediction {
            token_id: conditions.token_id.clone(),
            trade_size_usd: size_usd,
            expected_impact_bps: adjusted_impact,
            impact_range: (
                (adjusted_impact - uncertainty).max(Decimal::ZERO),
                adjusted_impact + uncertainty,
            ),
            temporary_impact_bps: temporary,
            permanent_impact_bps: permanent,
            expected_recovery_secs: recovery_secs,
            optimal_execution_time: None,
            confidence,
            model_used: ImpactModelType::Logarithmic,
            market_conditions: conditions.clone(),
        }
    }

    /// Adaptive model: Uses historical data to choose best model
    fn adaptive_model_predict(
        &self,
        token_id: &TokenId,
        size_usd: Decimal,
        conditions: &MarketConditions,
        params: &ImpactParameters,
    ) -> ImpactPrediction {
        // If we have enough history with good fit, use calibrated params
        // Otherwise fall back to sqrt model as default
        let use_sqrt = params.observation_count < 10 || params.r_squared < dec!(0.5);

        if use_sqrt {
            let mut prediction = self.sqrt_model_predict(size_usd, conditions, params);
            prediction.model_used = ImpactModelType::Adaptive;
            prediction
        } else {
            // Use power law with calibrated gamma: impact = α * (Q/V)^γ
            let depth = conditions.current_depth_usd.max(dec!(1));
            let relative_size = size_usd / depth;

            let power = self.decimal_pow(relative_size, params.gamma);
            let base_impact = params.alpha * conditions.current_volatility * power * dec!(10000);
            let adjusted_impact = self.adjust_for_conditions(base_impact, conditions);

            let permanent = adjusted_impact * self.config.permanent_impact_ratio;
            let temporary = adjusted_impact - permanent;
            let recovery_secs = self.config.decay_halflife_secs * 3;

            // Higher confidence with calibrated model
            let confidence = params.r_squared;

            // Tighter bounds with calibrated model
            let uncertainty = adjusted_impact * (Decimal::ONE - params.r_squared) * dec!(0.5);

            debug!(
                token_id = %token_id,
                gamma = %params.gamma,
                r_squared = %params.r_squared,
                "Adaptive model using calibrated parameters"
            );

            ImpactPrediction {
                token_id: conditions.token_id.clone(),
                trade_size_usd: size_usd,
                expected_impact_bps: adjusted_impact,
                impact_range: (
                    (adjusted_impact - uncertainty).max(Decimal::ZERO),
                    adjusted_impact + uncertainty,
                ),
                temporary_impact_bps: temporary,
                permanent_impact_bps: permanent,
                expected_recovery_secs: recovery_secs,
                optimal_execution_time: None,
                confidence,
                model_used: ImpactModelType::Adaptive,
                market_conditions: conditions.clone(),
            }
        }
    }

    /// Adjust impact for current market conditions
    fn adjust_for_conditions(&self, base_impact: Decimal, conditions: &MarketConditions) -> Decimal {
        let mut impact = base_impact;

        // Volatility adjustment: higher vol = higher impact
        if self.config.volatility_adjustment {
            // Baseline volatility assumption: 10%
            let vol_factor = conditions.current_volatility / dec!(0.1);
            impact *= vol_factor.max(dec!(0.5)).min(dec!(2.0));
        }

        // Liquidity adjustment: lower liquidity = higher impact
        if self.config.liquidity_adjustment {
            // Spread adjustment: wider spreads indicate lower liquidity
            let spread_factor = Decimal::ONE + (conditions.current_spread_bps / dec!(100));
            impact *= spread_factor.min(dec!(2.0));
        }

        impact.max(Decimal::ZERO)
    }

    /// Record an observed trade impact for model calibration
    pub fn record_impact(&mut self, token_id: &TokenId, observation: ImpactObservation) {
        let history = self
            .impact_history
            .entry(token_id.clone())
            .or_default();

        history.push_back(observation);

        // Trim to lookback limit
        while history.len() > self.config.lookback_trades {
            history.pop_front();
        }

        // Recalibrate if we have enough observations
        if history.len() >= 10 {
            self.calibrate_parameters(token_id);
        }
    }

    /// Calibrate model parameters from historical observations
    fn calibrate_parameters(&mut self, token_id: &TokenId) {
        let history = match self.impact_history.get(token_id) {
            Some(h) if h.len() >= 10 => h,
            _ => return,
        };

        let params = self.fit_impact_model(history);
        self.calibrated_params.insert(token_id.clone(), params);
    }

    /// Fit impact model parameters using simple regression
    fn fit_impact_model(&self, history: &VecDeque<ImpactObservation>) -> ImpactParameters {
        // Collect data points: (relative_size, realized_impact)
        let data_points: Vec<(f64, f64)> = history
            .iter()
            .filter_map(|obs| {
                let liquidity = obs.liquidity_at_time.unwrap_or(dec!(1000));
                if liquidity <= Decimal::ZERO {
                    return None;
                }
                let relative_size = (obs.trade_size_usd / liquidity)
                    .to_string()
                    .parse::<f64>()
                    .ok()?;
                let impact = obs.realized_impact_bps.to_string().parse::<f64>().ok()?;
                Some((relative_size, impact))
            })
            .collect();

        if data_points.is_empty() {
            return ImpactParameters::default_params(&self.config);
        }

        // Fit power law: impact = α * x^γ
        // Take log: ln(impact) = ln(α) + γ * ln(x)
        // Simple linear regression on log-transformed data

        let log_data: Vec<(f64, f64)> = data_points
            .iter()
            .filter(|(x, y)| *x > 0.0 && *y > 0.0)
            .map(|(x, y)| (x.ln(), y.ln()))
            .collect();

        if log_data.len() < 5 {
            return ImpactParameters::default_params(&self.config);
        }

        // Linear regression: y = a + b*x
        let n = log_data.len() as f64;
        let sum_x: f64 = log_data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = log_data.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = log_data.iter().map(|(x, y)| x * y).sum();
        let sum_xx: f64 = log_data.iter().map(|(x, _)| x * x).sum();
        let sum_yy: f64 = log_data.iter().map(|(_, y)| y * y).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return ImpactParameters::default_params(&self.config);
        }

        let gamma = (n * sum_xy - sum_x * sum_y) / denom;
        let ln_alpha = (sum_y - gamma * sum_x) / n;
        let alpha = ln_alpha.exp();

        // Calculate R-squared
        let y_mean = sum_y / n;
        let ss_tot = sum_yy - n * y_mean * y_mean;
        let ss_res: f64 = log_data
            .iter()
            .map(|(x, y)| {
                let predicted = ln_alpha + gamma * x;
                (y - predicted).powi(2)
            })
            .sum();

        let r_squared = (if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        })
        .clamp(0.0, 1.0);

        debug!(
            alpha = alpha,
            gamma = gamma,
            r_squared = r_squared,
            n_obs = history.len(),
            "Calibrated impact model parameters"
        );

        ImpactParameters {
            alpha: Decimal::from_str_exact(&format!("{:.6}", alpha.clamp(0.001, 1.0)))
                .unwrap_or(self.config.default_alpha),
            beta: self.config.default_beta, // Beta requires observing permanent impact
            gamma: Decimal::from_str_exact(&format!("{:.3}", gamma.clamp(0.1, 1.0)))
                .unwrap_or(self.config.default_gamma),
            decay_rate: Decimal::ONE / Decimal::from(self.config.decay_halflife_secs),
            r_squared: Decimal::from_str_exact(&format!("{:.3}", r_squared))
                .unwrap_or(Decimal::ZERO),
            last_calibrated: Utc::now(),
            observation_count: history.len(),
        }
    }

    /// Get execution recommendation based on impact prediction
    pub fn get_recommendation(&self, prediction: &ImpactPrediction) -> ImpactRecommendation {
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

        // Impact exceeds limit - calculate how to reduce it

        // Option 1: Reduce size to stay within impact limit
        // For sqrt model: impact ∝ sqrt(size), so size ∝ impact^2
        // Reduction factor = (max_impact / predicted_impact)^2 for sqrt
        // For linear: factor = max_impact / predicted_impact
        let reduction_factor = match prediction.model_used {
            ImpactModelType::SquareRoot | ImpactModelType::Adaptive => {
                let ratio = max_impact / prediction.expected_impact_bps;
                ratio * ratio // Square for sqrt model
            }
            ImpactModelType::Linear => max_impact / prediction.expected_impact_bps,
            ImpactModelType::Logarithmic => {
                // Approximate: impact ∝ ln(size), harder to invert
                max_impact / prediction.expected_impact_bps
            }
        };

        let reduced_size = prediction.trade_size_usd * reduction_factor;

        // Option 2: Split into slices with time spacing
        // Number of slices to get each slice under impact limit
        let impact_ratio = prediction.expected_impact_bps / max_impact;
        let slices = impact_ratio
            .ceil()
            .to_string()
            .parse::<u32>()
            .unwrap_or(5)
            .clamp(2, 10);

        // Interval should allow temporary impact to decay
        let interval_secs = prediction.expected_recovery_secs / 2;

        ImpactRecommendation {
            should_execute: true,
            recommended_size: reduced_size,
            recommended_slices: slices,
            recommended_interval_secs: interval_secs,
            reason: format!(
                "Impact {:.1} bps exceeds limit {:.1} bps. Recommended: reduce size to ${:.2} or split into {} slices with {}s intervals",
                prediction.expected_impact_bps, max_impact, reduced_size, slices, interval_secs
            ),
        }
    }

    /// Helper: compute decimal square root using Newton's method
    fn decimal_sqrt(&self, x: Decimal) -> Decimal {
        if x <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        // Convert to f64, sqrt, convert back
        let x_f64 = x.to_string().parse::<f64>().unwrap_or(0.0);
        let sqrt_f64 = x_f64.sqrt();

        Decimal::from_str_exact(&format!("{:.10}", sqrt_f64)).unwrap_or(Decimal::ZERO)
    }

    /// Helper: compute decimal power x^y
    fn decimal_pow(&self, x: Decimal, y: Decimal) -> Decimal {
        if x <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        let x_f64 = x.to_string().parse::<f64>().unwrap_or(0.0);
        let y_f64 = y.to_string().parse::<f64>().unwrap_or(0.5);
        let pow_f64 = x_f64.powf(y_f64);

        Decimal::from_str_exact(&format!("{:.10}", pow_f64)).unwrap_or(Decimal::ZERO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polysniper_core::PriceLevel;
    use rust_decimal_macros::dec;

    fn default_config() -> MarketImpactConfig {
        MarketImpactConfig::default()
    }

    fn test_orderbook() -> Orderbook {
        Orderbook {
            token_id: "test_token".to_string(),
            market_id: "test_market".to_string(),
            bids: vec![
                PriceLevel {
                    price: dec!(0.50),
                    size: dec!(1000),
                },
                PriceLevel {
                    price: dec!(0.49),
                    size: dec!(2000),
                },
                PriceLevel {
                    price: dec!(0.48),
                    size: dec!(3000),
                },
            ],
            asks: vec![
                PriceLevel {
                    price: dec!(0.51),
                    size: dec!(1000),
                },
                PriceLevel {
                    price: dec!(0.52),
                    size: dec!(2000),
                },
                PriceLevel {
                    price: dec!(0.53),
                    size: dec!(3000),
                },
            ],
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_sqrt_model_prediction() {
        let estimator = MarketImpactEstimator::new(default_config());
        let orderbook = test_orderbook();

        let prediction = estimator.predict_impact(
            &"test_token".to_string(),
            dec!(100), // $100 trade
            Side::Buy,
            &orderbook,
        );

        assert!(prediction.expected_impact_bps > Decimal::ZERO);
        assert!(prediction.temporary_impact_bps > Decimal::ZERO);
        assert!(prediction.permanent_impact_bps > Decimal::ZERO);
        assert_eq!(prediction.model_used, ImpactModelType::SquareRoot);
    }

    #[test]
    fn test_impact_increases_with_size() {
        let estimator = MarketImpactEstimator::new(default_config());
        let orderbook = test_orderbook();

        let small_prediction = estimator.predict_impact(
            &"test_token".to_string(),
            dec!(10),
            Side::Buy,
            &orderbook,
        );

        let large_prediction = estimator.predict_impact(
            &"test_token".to_string(),
            dec!(1000),
            Side::Buy,
            &orderbook,
        );

        assert!(
            large_prediction.expected_impact_bps > small_prediction.expected_impact_bps,
            "Larger trade should have larger impact"
        );
    }

    #[test]
    fn test_volatility_adjustment() {
        let config = MarketImpactConfig {
            volatility_adjustment: true,
            ..default_config()
        };
        let estimator = MarketImpactEstimator::new(config);
        let orderbook = test_orderbook();

        // Create conditions with different volatilities
        let low_vol_conditions = MarketConditions::from_orderbook(&orderbook)
            .with_volatility(dec!(0.05)); // 5% vol
        let high_vol_conditions = MarketConditions::from_orderbook(&orderbook)
            .with_volatility(dec!(0.20)); // 20% vol

        let _params = ImpactParameters::default_params(estimator.config());

        let low_vol_impact = estimator.adjust_for_conditions(dec!(10), &low_vol_conditions);
        let high_vol_impact = estimator.adjust_for_conditions(dec!(10), &high_vol_conditions);

        assert!(
            high_vol_impact > low_vol_impact,
            "Higher volatility should increase impact"
        );
    }

    #[test]
    fn test_recommendation_within_limits() {
        let config = MarketImpactConfig {
            max_acceptable_impact_bps: dec!(100),
            ..default_config()
        };
        let estimator = MarketImpactEstimator::new(config);

        let prediction = ImpactPrediction {
            token_id: "test".to_string(),
            trade_size_usd: dec!(100),
            expected_impact_bps: dec!(50), // Under limit
            impact_range: (dec!(35), dec!(65)),
            temporary_impact_bps: dec!(35),
            permanent_impact_bps: dec!(15),
            expected_recovery_secs: 180,
            optimal_execution_time: None,
            confidence: dec!(0.8),
            model_used: ImpactModelType::SquareRoot,
            market_conditions: MarketConditions {
                token_id: "test".to_string(),
                current_volatility: dec!(0.1),
                current_spread_bps: dec!(10),
                current_depth_usd: dec!(5000),
                recent_volume_usd: dec!(10000),
                timestamp: Utc::now(),
            },
        };

        let recommendation = estimator.get_recommendation(&prediction);

        assert!(recommendation.should_execute);
        assert_eq!(recommendation.recommended_size, dec!(100));
        assert_eq!(recommendation.recommended_slices, 1);
        assert!(recommendation.reason.contains("within acceptable range"));
    }

    #[test]
    fn test_recommendation_size_reduction() {
        let config = MarketImpactConfig {
            max_acceptable_impact_bps: dec!(50),
            ..default_config()
        };
        let estimator = MarketImpactEstimator::new(config);

        let prediction = ImpactPrediction {
            token_id: "test".to_string(),
            trade_size_usd: dec!(1000),
            expected_impact_bps: dec!(200), // Way over limit
            impact_range: (dec!(150), dec!(250)),
            temporary_impact_bps: dec!(140),
            permanent_impact_bps: dec!(60),
            expected_recovery_secs: 180,
            optimal_execution_time: None,
            confidence: dec!(0.7),
            model_used: ImpactModelType::SquareRoot,
            market_conditions: MarketConditions {
                token_id: "test".to_string(),
                current_volatility: dec!(0.1),
                current_spread_bps: dec!(10),
                current_depth_usd: dec!(5000),
                recent_volume_usd: dec!(10000),
                timestamp: Utc::now(),
            },
        };

        let recommendation = estimator.get_recommendation(&prediction);

        assert!(recommendation.should_execute);
        assert!(recommendation.recommended_size < dec!(1000), "Should reduce size");
        assert!(recommendation.recommended_slices > 1, "Should recommend slicing");
        assert!(recommendation.reason.contains("exceeds limit"));
    }

    #[test]
    fn test_calibration_from_observations() {
        let mut estimator = MarketImpactEstimator::new(default_config());
        let token_id = "test_token".to_string();

        // Record enough observations for calibration
        for i in 1..=15 {
            let size = Decimal::from(i * 100);
            // Simulate sqrt-like impact: impact ∝ sqrt(size/liquidity)
            let liquidity = dec!(5000);
            let relative_size = size / liquidity;
            let sqrt_rel = (relative_size.to_string().parse::<f64>().unwrap_or(0.0)).sqrt();
            let impact = Decimal::from_str_exact(&format!("{:.2}", sqrt_rel * 100.0))
                .unwrap_or(dec!(5));

            estimator.record_impact(
                &token_id,
                ImpactObservation {
                    token_id: token_id.clone(),
                    trade_size_usd: size,
                    pre_trade_price: dec!(0.50),
                    post_trade_price: dec!(0.51),
                    mid_price_before: dec!(0.505),
                    mid_price_after: dec!(0.510),
                    realized_impact_bps: impact,
                    timestamp: Utc::now(),
                    recovery_price: Some(dec!(0.505)),
                    recovery_time_secs: Some(60),
                    volatility_at_time: Some(dec!(0.1)),
                    liquidity_at_time: Some(liquidity),
                },
            );
        }

        // Check that calibration occurred
        let params = estimator.get_calibrated_params(&token_id);
        assert!(params.observation_count > 0);
        // Gamma should be close to 0.5 for sqrt-like data
        assert!(params.gamma > dec!(0.3) && params.gamma < dec!(0.7));
    }

    #[test]
    fn test_linear_model_prediction() {
        let config = MarketImpactConfig {
            model_type: ImpactModelType::Linear,
            ..default_config()
        };
        let estimator = MarketImpactEstimator::new(config);
        let orderbook = test_orderbook();

        let prediction = estimator.predict_impact(
            &"test_token".to_string(),
            dec!(100),
            Side::Buy,
            &orderbook,
        );

        assert!(prediction.expected_impact_bps > Decimal::ZERO);
        assert_eq!(prediction.model_used, ImpactModelType::Linear);
    }

    #[test]
    fn test_log_model_prediction() {
        let config = MarketImpactConfig {
            model_type: ImpactModelType::Logarithmic,
            ..default_config()
        };
        let estimator = MarketImpactEstimator::new(config);
        let orderbook = test_orderbook();

        let prediction = estimator.predict_impact(
            &"test_token".to_string(),
            dec!(100),
            Side::Buy,
            &orderbook,
        );

        assert!(prediction.expected_impact_bps > Decimal::ZERO);
        assert_eq!(prediction.model_used, ImpactModelType::Logarithmic);
    }

    #[test]
    fn test_market_conditions_from_orderbook() {
        let orderbook = test_orderbook();
        let conditions = MarketConditions::from_orderbook(&orderbook);

        assert_eq!(conditions.token_id, "test_token");
        assert!(conditions.current_spread_bps > Decimal::ZERO);
        assert!(conditions.current_depth_usd > Decimal::ZERO);
    }

    #[test]
    fn test_impact_range_contains_expected() {
        let estimator = MarketImpactEstimator::new(default_config());
        let orderbook = test_orderbook();

        let prediction = estimator.predict_impact(
            &"test_token".to_string(),
            dec!(500),
            Side::Buy,
            &orderbook,
        );

        let (low, high) = prediction.impact_range;
        assert!(low <= prediction.expected_impact_bps);
        assert!(high >= prediction.expected_impact_bps);
    }

    #[test]
    fn test_zero_size_trade() {
        let estimator = MarketImpactEstimator::new(default_config());
        let orderbook = test_orderbook();

        let prediction = estimator.predict_impact(
            &"test_token".to_string(),
            Decimal::ZERO,
            Side::Buy,
            &orderbook,
        );

        assert_eq!(prediction.expected_impact_bps, Decimal::ZERO);
    }

    #[test]
    fn test_adaptive_model_uses_sqrt_when_uncalibrated() {
        let config = MarketImpactConfig {
            model_type: ImpactModelType::Adaptive,
            ..default_config()
        };
        let estimator = MarketImpactEstimator::new(config);
        let orderbook = test_orderbook();

        let prediction = estimator.predict_impact(
            &"test_token".to_string(),
            dec!(100),
            Side::Buy,
            &orderbook,
        );

        // Should use adaptive but fall back to sqrt behavior
        assert_eq!(prediction.model_used, ImpactModelType::Adaptive);
        assert!(prediction.expected_impact_bps > Decimal::ZERO);
    }
}
