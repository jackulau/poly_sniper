---
id: ml-signal-processor
name: ML Signal Processor - Strategy Integration for Model Predictions
wave: 2
priority: 2
dependencies: [ml-webhook-core]
estimated_hours: 4
tags: [backend, strategy, ml-integration]
---

## Objective

Enhance the EventBasedStrategy to intelligently process ML model predictions, with confidence-based sizing, model tracking, and signal quality filtering.

## Context

Once ml-webhook-core provides the HTTP endpoint, predictions arrive as `ExternalSignal` events. This task enhances how those signals are processed by strategies, adding ML-specific logic for confidence weighting and model performance tracking.

## Implementation

1. Create `/crates/polysniper-strategies/src/ml_processor.rs`:
   - `MlSignalProcessor` that wraps signal handling
   - Confidence-based position sizing (higher confidence = larger size)
   - Minimum confidence threshold filtering
   - Model ID tracking for performance analysis

2. Enhance EventBasedStrategy:
   - Detect ML webhook signals via source matching
   - Apply confidence multiplier to order size
   - Track signal outcomes by model_id
   - Configurable model-specific rules

3. Create `/crates/polysniper-core/src/ml_types.rs`:
   - `MlPrediction` struct with all model output fields
   - `MlConfig` with confidence thresholds, sizing rules
   - `ModelPerformance` for tracking accuracy

4. Add ML signal rules to strategy config:
   - Per-model confidence thresholds
   - Size multiplier curves
   - Cooldown between signals from same model

## Acceptance Criteria

- [ ] ML signals processed with confidence-aware sizing
- [ ] Minimum confidence threshold filtering (configurable)
- [ ] Size scaled by confidence (e.g., 50% conf = 50% of max size)
- [ ] Model ID tracked in signal metadata
- [ ] Per-model cooldown prevents rapid-fire signals
- [ ] Signal outcomes tracked for model performance
- [ ] Integrates cleanly with existing EventBasedStrategy
- [ ] Configuration via strategy config files
- [ ] Unit tests for confidence calculations

## Files to Create/Modify

- `crates/polysniper-strategies/src/ml_processor.rs` - **CREATE** - ML signal processing
- `crates/polysniper-strategies/src/event_based.rs` - **MODIFY** - Integrate ML processing
- `crates/polysniper-strategies/src/lib.rs` - **MODIFY** - Export ML module
- `crates/polysniper-core/src/ml_types.rs` - **CREATE** - ML-specific types
- `crates/polysniper-core/src/lib.rs` - **MODIFY** - Export ML types
- `config/strategies/event_based.toml` - **MODIFY** - Add ML signal rules

## Integration Points

- **Provides**: Enhanced ML signal processing for strategies
- **Consumes**: ExternalSignal events from ml-webhook-core
- **Conflicts**: Modifies event_based.rs (coordinate with other strategy tasks)
