---
id: microstructure-strategy
name: Market Microstructure Trading Strategy
wave: 2
priority: 2
dependencies: [vpin-calculator, whale-detector, market-impact-estimator, microstructure-events]
estimated_hours: 6
tags: [strategy, integration, trading]
---

## Objective

Implement a comprehensive trading strategy that combines VPIN, whale detection, and market impact signals to generate intelligent trade signals and risk adjustments.

## Context

This strategy serves as the integration point for all microstructure analysis components. It:
1. Processes VPIN, whale, and impact signals
2. Generates trade signals based on microstructure conditions
3. Adjusts position sizing based on toxicity levels
4. Follows or avoids whale activity based on configuration
5. Provides risk recommendations to the risk manager

## Implementation

### 1. Create strategy: `crates/polysniper-strategies/src/microstructure_strategy.rs`

**Core Components:**

```rust
pub struct MicrostructureStrategyConfig {
    pub enabled: bool,

    // VPIN-based trading
    pub vpin_trading: VpinTradingConfig,

    // Whale-based trading
    pub whale_trading: WhaleTradingConfig,

    // Impact-based adjustments
    pub impact_adjustments: ImpactAdjustmentConfig,

    // General settings
    pub min_confidence: Decimal,
    pub cooldown_secs: u64,
    pub max_signals_per_hour: u32,
}

pub struct VpinTradingConfig {
    pub enabled: bool,
    pub high_toxicity_action: ToxicityAction,
    pub low_toxicity_action: ToxicityAction,
    pub toxicity_position_multiplier: HashMap<ToxicityLevel, Decimal>,
}

pub enum ToxicityAction {
    None,
    ReducePositions { multiplier: Decimal },
    HaltNewTrades,
    FadeTheCrowd,  // Trade against high buy/sell imbalance
    FollowTheCrowd,
}

pub struct WhaleTradingConfig {
    pub enabled: bool,
    pub follow_whale: bool,
    pub follow_confidence_threshold: Decimal,
    pub avoid_whale_threshold_usd: Decimal,
    pub whale_position_multiplier: Decimal,
}

pub struct ImpactAdjustmentConfig {
    pub enabled: bool,
    pub max_acceptable_impact_bps: Decimal,
    pub auto_slice_above_impact_bps: Decimal,
    pub delay_high_impact_secs: u64,
}

pub struct MicrostructureStrategy {
    config: MicrostructureStrategyConfig,

    // Analysis components
    vpin_calculator: VpinCalculator,
    whale_detector: WhaleDetector,
    impact_estimator: MarketImpactEstimator,

    // State tracking
    current_toxicity: HashMap<TokenId, ToxicityLevel>,
    recent_whale_alerts: HashMap<TokenId, VecDeque<WhaleAlert>>,
    signal_cooldowns: HashMap<TokenId, DateTime<Utc>>,
    hourly_signal_counts: HashMap<String, u32>,

    // Event publisher
    publisher: MicrostructurePublisher,
}
```

### 2. Implement Strategy trait

```rust
#[async_trait]
impl Strategy for MicrostructureStrategy {
    fn name(&self) -> &str {
        "microstructure"
    }

    fn accepts_event(&self, event: &SystemEvent) -> bool {
        matches!(
            event,
            SystemEvent::OrderbookUpdate(_)
                | SystemEvent::PriceChange(_)
                | SystemEvent::PartialFill(_)
                | SystemEvent::FullFill(_)
                | SystemEvent::TradeExecuted(_)
                | SystemEvent::Microstructure(_)
        )
    }

    async fn process_event(
        &self,
        event: &SystemEvent,
        state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        match event {
            // Process orderbook updates for VPIN
            SystemEvent::OrderbookUpdate(ob_event) => {
                self.process_orderbook_update(ob_event, state).await
            }

            // Process fills for whale detection and impact calibration
            SystemEvent::PartialFill(fill) | SystemEvent::FullFill(fill) => {
                self.process_fill(fill, state).await
            }

            // Process trade executions
            SystemEvent::TradeExecuted(trade) => {
                self.process_trade(trade, state).await
            }

            // Process microstructure events from other components
            SystemEvent::Microstructure(micro) => {
                self.process_microstructure_event(micro, state).await
            }

            _ => Ok(vec![]),
        }
    }

    async fn reload_config(&mut self, config_content: &str) -> Result<(), StrategyError> {
        let config: MicrostructureStrategyConfig = toml::from_str(config_content)?;
        self.config = config;
        Ok(())
    }
}
```

### 3. Implement signal generation

```rust
impl MicrostructureStrategy {
    /// Process orderbook update - update VPIN and check for signals
    async fn process_orderbook_update(
        &mut self,
        event: &OrderbookUpdateEvent,
        state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        let mut signals = vec![];

        // Update VPIN calculation
        if let Some(vpin_result) = self.vpin_calculator.process_orderbook_update(event) {
            // Check for toxicity level changes
            let prev_level = self.current_toxicity.get(&event.token_id).copied();

            if prev_level != Some(vpin_result.toxicity_level) {
                // Toxicity level changed - publish event
                self.publisher.publish_toxicity_change(ToxicityChangeEvent {
                    token_id: event.token_id.clone(),
                    market_id: event.market_id.clone(),
                    previous_level: prev_level.unwrap_or(ToxicityLevel::Normal),
                    new_level: vpin_result.toxicity_level,
                    vpin: vpin_result.vpin,
                    trigger_reason: "VPIN threshold crossed".to_string(),
                    timestamp: Utc::now(),
                })?;

                // Generate signals based on toxicity change
                if let Some(signal) = self.generate_toxicity_signal(&vpin_result, state).await? {
                    signals.push(signal);
                }
            }

            self.current_toxicity.insert(event.token_id.clone(), vpin_result.toxicity_level);

            // Publish VPIN update
            self.publisher.publish_vpin_update(VpinUpdateEvent::from(vpin_result))?;
        }

        Ok(signals)
    }

    /// Process trade fill - update whale detector and impact estimator
    async fn process_fill(
        &mut self,
        fill: &FillEvent,
        state: &dyn StateProvider,
    ) -> Result<Vec<TradeSignal>, StrategyError> {
        let mut signals = vec![];

        // Check for whale activity
        if let Some(whale_alert) = self.whale_detector.process_trade(
            &fill.token_id,
            &fill.market_id,
            fill.side,
            fill.size_usd,
            fill.price,
            fill.address.as_deref(),
            fill.timestamp,
        ) {
            // Publish whale detection
            self.publisher.publish_whale_detected(WhaleDetectedEvent::from(&whale_alert))?;

            // Generate whale-based signal if configured
            if let Some(signal) = self.generate_whale_signal(&whale_alert, state).await? {
                signals.push(signal);
            }
        }

        // Record impact observation for calibration
        if let Some(pre_price) = state.get_price(&fill.token_id).await {
            self.impact_estimator.record_impact(&fill.token_id, ImpactObservation {
                token_id: fill.token_id.clone(),
                trade_size_usd: fill.size_usd,
                pre_trade_price: pre_price,
                post_trade_price: fill.price,
                // ... other fields
            });
        }

        Ok(signals)
    }

    /// Generate signal based on toxicity level
    async fn generate_toxicity_signal(
        &self,
        vpin_result: &VpinResult,
        state: &dyn StateProvider,
    ) -> Result<Option<TradeSignal>, StrategyError> {
        if !self.config.vpin_trading.enabled {
            return Ok(None);
        }

        // Check cooldown
        if let Some(last_signal) = self.signal_cooldowns.get(&vpin_result.token_id) {
            if Utc::now() - *last_signal < Duration::seconds(self.config.cooldown_secs as i64) {
                return Ok(None);
            }
        }

        let action = match vpin_result.toxicity_level {
            ToxicityLevel::High => &self.config.vpin_trading.high_toxicity_action,
            ToxicityLevel::Low => &self.config.vpin_trading.low_toxicity_action,
            _ => return Ok(None),
        };

        match action {
            ToxicityAction::FadeTheCrowd => {
                // Trade against the dominant flow direction
                let side = if vpin_result.buy_volume_pct > dec!(0.6) {
                    Side::Sell  // Fade the buyers
                } else if vpin_result.sell_volume_pct > dec!(0.6) {
                    Side::Buy   // Fade the sellers
                } else {
                    return Ok(None);
                };

                // Build signal
                Ok(Some(self.build_signal(
                    &vpin_result.token_id,
                    side,
                    format!("Fade crowd: VPIN={:.2}, toxicity={:?}", vpin_result.vpin, vpin_result.toxicity_level),
                    state,
                ).await?))
            }
            ToxicityAction::FollowTheCrowd => {
                // Trade with the dominant flow direction
                let side = if vpin_result.buy_volume_pct > dec!(0.6) {
                    Side::Buy
                } else if vpin_result.sell_volume_pct > dec!(0.6) {
                    Side::Sell
                } else {
                    return Ok(None);
                };

                Ok(Some(self.build_signal(
                    &vpin_result.token_id,
                    side,
                    format!("Follow flow: VPIN={:.2}", vpin_result.vpin),
                    state,
                ).await?))
            }
            _ => Ok(None),
        }
    }

    /// Generate signal based on whale detection
    async fn generate_whale_signal(
        &self,
        whale_alert: &WhaleAlert,
        state: &dyn StateProvider,
    ) -> Result<Option<TradeSignal>, StrategyError> {
        if !self.config.whale_trading.enabled {
            return Ok(None);
        }

        if !self.config.whale_trading.follow_whale {
            return Ok(None);
        }

        if whale_alert.confidence < self.config.whale_trading.follow_confidence_threshold {
            return Ok(None);
        }

        // Follow the whale's direction
        let side = match &whale_alert.whale_trade.side {
            Side::Buy => Side::Buy,
            Side::Sell => Side::Sell,
        };

        Ok(Some(self.build_signal(
            &whale_alert.token_id,
            side,
            format!("Follow whale: ${:.0} {:?}", whale_alert.whale_trade.size_usd, whale_alert.alert_type),
            state,
        ).await?))
    }

    /// Apply microstructure-based position sizing
    pub fn get_size_multiplier(&self, token_id: &TokenId) -> Decimal {
        let mut multiplier = Decimal::ONE;

        // Apply toxicity multiplier
        if let Some(level) = self.current_toxicity.get(token_id) {
            if let Some(m) = self.config.vpin_trading.toxicity_position_multiplier.get(level) {
                multiplier *= *m;
            }
        }

        // Apply whale multiplier if recent whale activity
        if let Some(alerts) = self.recent_whale_alerts.get(token_id) {
            let recent = alerts.iter()
                .filter(|a| Utc::now() - a.timestamp < Duration::minutes(5))
                .count();
            if recent > 0 {
                multiplier *= self.config.whale_trading.whale_position_multiplier;
            }
        }

        multiplier.max(dec!(0.1))  // Never reduce below 10%
    }
}
```

### 4. Create configuration file: `config/strategies/microstructure.toml`

```toml
[strategy]
enabled = true
id = "microstructure"
name = "Market Microstructure Strategy"

# General settings
min_confidence = "0.6"
cooldown_secs = 300
max_signals_per_hour = 10

# VPIN-based trading
[strategy.vpin_trading]
enabled = true
high_toxicity_action = "ReducePositions"
low_toxicity_action = "None"

[strategy.vpin_trading.toxicity_position_multiplier]
Low = "1.2"       # Increase size in low toxicity
Normal = "1.0"    # Normal sizing
Elevated = "0.7"  # Reduce size
High = "0.4"      # Significantly reduce

# Whale-based trading
[strategy.whale_trading]
enabled = true
follow_whale = true
follow_confidence_threshold = "0.7"
avoid_whale_threshold_usd = "10000"
whale_position_multiplier = "0.6"

# Impact adjustments
[strategy.impact_adjustments]
enabled = true
max_acceptable_impact_bps = "50"
auto_slice_above_impact_bps = "25"
delay_high_impact_secs = 60
```

### 5. Add tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> MicrostructureStrategyConfig { ... }

    struct MockStateProvider { ... }

    #[tokio::test]
    async fn test_toxicity_signal_generation() {
        // High toxicity should reduce positions
    }

    #[tokio::test]
    async fn test_whale_follow_signal() {
        // Whale detection should generate follow signal
    }

    #[tokio::test]
    async fn test_size_multiplier_calculation() {
        // Combined toxicity and whale should stack
    }

    #[tokio::test]
    async fn test_cooldown_prevents_spam() {
        // Multiple signals within cooldown should be ignored
    }

    #[tokio::test]
    async fn test_config_reload() {
        // Config changes should apply
    }

    #[tokio::test]
    async fn test_fade_the_crowd_signal() {
        // Low toxicity with imbalance should generate contrarian signal
    }
}
```

### 6. Export from strategies crate

Update `crates/polysniper-strategies/src/lib.rs` to export the microstructure strategy.

## Acceptance Criteria

- [ ] Strategy implements Strategy trait correctly
- [ ] VPIN signals generate appropriate trade signals
- [ ] Whale detection triggers follow/avoid signals
- [ ] Size multipliers stack correctly from multiple sources
- [ ] Cooldown prevents signal spam
- [ ] Configuration reloads work
- [ ] Events are published to event bus
- [ ] All tests pass (`cargo test -p polysniper-strategies`)
- [ ] Integration with VPIN, whale, and impact components works
- [ ] Code follows existing patterns
- [ ] No clippy warnings

## Files to Create/Modify

- `crates/polysniper-strategies/src/microstructure_strategy.rs` - **CREATE** - Strategy implementation
- `crates/polysniper-strategies/src/lib.rs` - **MODIFY** - Export strategy
- `config/strategies/microstructure.toml` - **CREATE** - Configuration file

## Integration Points

- **Provides**: Trading signals, size multipliers, risk recommendations
- **Consumes**: VPIN calculator, whale detector, market impact estimator, event publisher
- **Depends on**: vpin-calculator, whale-detector, market-impact-estimator, microstructure-events
- **Conflicts**: None - new strategy module
