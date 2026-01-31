---
id: participation-rate-adaptation
name: Dynamic Participation Rate Adaptation
wave: 1
priority: 2
dependencies: []
estimated_hours: 5
tags: [execution, volume, adaptation]
---

## Objective

Dynamically adjust participation rates based on real-time volume rather than fixed rates.

## Context

The current TWAP/VWAP executors use a fixed `max_participation_rate` (default 10%). This is suboptimal because:
- Low volume periods: Fixed rate may still cause excessive market impact
- High volume periods: Fixed rate leaves alpha on the table
- Volume spikes: Opportunity to execute more without impact

The existing `depth_analyzer.rs` has liquidity analysis, but participation adaptation should be volume-based, not just depth-based.

## Implementation

### 1. Create Volume Profile Monitor

Create new file `crates/polysniper-execution/src/volume_monitor.rs`:

```rust
pub struct VolumeMonitor {
    // Rolling volume observations per token
    volume_history: HashMap<TokenId, VecDeque<VolumeObservation>>,
    config: VolumeMonitorConfig,
}

pub struct VolumeObservation {
    timestamp: DateTime<Utc>,
    volume: Decimal,
    interval_secs: u64,
}

pub struct VolumeMonitorConfig {
    history_window_secs: u64,      // How far back to look (default: 3600)
    observation_interval_secs: u64, // Granularity (default: 60)
    smoothing_factor: Decimal,     // EMA smoothing (default: 0.2)
}

impl VolumeMonitor {
    pub fn get_current_volume_rate(&self, token_id: &TokenId) -> Decimal;
    pub fn get_average_volume_rate(&self, token_id: &TokenId) -> Decimal;
    pub fn get_volume_ratio(&self, token_id: &TokenId) -> Decimal; // current / average
    pub fn record_volume(&mut self, token_id: &TokenId, volume: Decimal);
}
```

### 2. Create Adaptive Participation Calculator

Create new file `crates/polysniper-execution/src/participation_adapter.rs`:

```rust
pub struct ParticipationAdapter {
    volume_monitor: Arc<RwLock<VolumeMonitor>>,
    config: ParticipationConfig,
}

pub struct ParticipationConfig {
    base_rate: Decimal,            // Base participation rate (default: 0.10)
    min_rate: Decimal,             // Floor participation (default: 0.02)
    max_rate: Decimal,             // Ceiling participation (default: 0.25)
    volume_scaling_factor: Decimal, // How aggressively to scale (default: 0.5)
    urgency_boost: Decimal,        // Boost for urgent orders (default: 0.05)
}

impl ParticipationAdapter {
    /// Calculate adaptive participation rate based on current conditions
    pub fn calculate_rate(
        &self,
        token_id: &TokenId,
        urgency: Priority,
        remaining_time_pct: Decimal,
    ) -> Decimal {
        let volume_ratio = self.volume_monitor.get_volume_ratio(token_id);

        // Scale: high volume = higher participation allowed
        let scaled_rate = self.config.base_rate * volume_ratio.powf(self.config.volume_scaling_factor);

        // Apply urgency boost
        let urgency_multiplier = match urgency {
            Priority::Critical => dec!(1.5),
            Priority::High => dec!(1.25),
            Priority::Normal => dec!(1.0),
            Priority::Low => dec!(0.8),
        };

        // Apply time pressure (accelerate near end of window)
        let time_pressure = if remaining_time_pct < dec!(0.2) {
            dec!(1.3)  // 30% boost in final 20% of time
        } else {
            dec!(1.0)
        };

        (scaled_rate * urgency_multiplier * time_pressure)
            .min(self.config.max_rate)
            .max(self.config.min_rate)
    }
}
```

### 3. Integrate with TWAP/VWAP

Modify executors to use `ParticipationAdapter` instead of fixed rate:
- Before each slice, calculate adaptive rate
- Adjust slice size based on current participation allowance
- Log participation rate decisions for analysis

### 4. Add Volume-Based Metrics

Add to observability:
- `volume_ratio_current` gauge per token
- `participation_rate_used` histogram
- `participation_adjustments` counter

## Acceptance Criteria

- [ ] VolumeMonitor tracks rolling volume history
- [ ] ParticipationAdapter calculates dynamic rates
- [ ] Rates scale with volume ratio (high volume = higher participation)
- [ ] Urgency and time pressure factored in
- [ ] Min/max bounds enforced
- [ ] TWAP/VWAP use adaptive rates per slice
- [ ] Metrics exported for monitoring
- [ ] All tests passing

## Files to Create/Modify

- `crates/polysniper-execution/src/volume_monitor.rs` - Create new file
- `crates/polysniper-execution/src/participation_adapter.rs` - Create new file
- `crates/polysniper-execution/src/lib.rs` - Add module exports
- `crates/polysniper-execution/src/algorithms/twap.rs` - Use adapter
- `crates/polysniper-execution/src/algorithms/vwap.rs` - Use adapter
- `crates/polysniper-core/src/types.rs` - Add VolumeMonitorConfig, ParticipationConfig
- `crates/polysniper-observability/src/metrics.rs` - Add volume metrics

## Integration Points

- **Provides**: Adaptive participation rates for execution algorithms
- **Consumes**: Trade volume data from WebSocket/fills, Priority from TradeSignal
- **Conflicts**: None - new functionality
