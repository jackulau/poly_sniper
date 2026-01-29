---
id: llm-strategy-config
name: LLM Strategy Configuration File
wave: 1
priority: 2
dependencies: []
estimated_hours: 1
tags: [config, strategy]
---

## Objective

Create the TOML configuration file for the LLM prediction strategy with sensible defaults.

## Context

This configuration file defines all tunable parameters for the LLM prediction strategy. It follows the existing config patterns in `config/strategies/` and will be loaded by main.rs during strategy initialization.

## Implementation

### 1. Create `config/strategies/llm_prediction.toml`

```toml
# LLM Prediction Strategy Configuration
# Uses OpenRouter API to analyze markets with LLM and generate trade signals

[strategy]
# Enable/disable the strategy (default: disabled for safety)
enabled = false

# OpenRouter API configuration
model = "x-ai/grok-3-latest"
api_key_env = "OPENROUTER_API_KEY"

# LLM parameters
temperature = 0.3
max_tokens = 1024

# Analysis parameters
# How often to run analysis (in seconds)
analysis_interval_secs = 300

# Maximum markets to analyze per interval (cost control)
max_markets_per_interval = 10

# Delay between API calls to avoid rate limits (milliseconds)
api_call_delay_ms = 500

# Trading parameters
# Minimum confidence score (0.0-1.0) to generate a signal
confidence_threshold = 0.75

# Minimum edge (difference between LLM predicted price and current price)
min_price_edge = "0.05"

# Order size in USD
order_size_usd = "50"

# Minimum market liquidity to consider (USD)
min_liquidity_usd = "5000"

# Order type: "Gtc" (good-til-canceled) or "Fok" (fill-or-kill)
order_type = "Gtc"

# Market filtering (optional)
# If empty, analyze all active markets
# markets = ["0x...", "0x..."]  # Specific market condition IDs
markets = []

# Filter by tags (optional)
# tags = ["politics", "sports"]
tags = []

# Keywords to filter markets by question text (optional)
# keywords = ["election", "president"]
keywords = []

# Prompt customization (optional)
# Override the default system prompt for specialized analysis
# system_prompt_override = "You are a specialized political analyst..."
```

## Acceptance Criteria

- [ ] Config file created at `config/strategies/llm_prediction.toml`
- [ ] All parameters documented with comments
- [ ] Default values are conservative (enabled = false, high thresholds)
- [ ] Follows existing config file patterns (see `event_based.toml`)
- [ ] TOML syntax is valid

## Files to Create/Modify

- `config/strategies/llm_prediction.toml` - **Create** - Strategy configuration

## Integration Points

- **Provides**: Configuration file for LLM prediction strategy
- **Consumes**: Nothing
- **Conflicts**: None - new file

## Notes

Key design decisions:
1. `enabled = false` by default for safety
2. Conservative thresholds (0.75 confidence, 0.05 edge) to minimize false signals
3. Cost controls (300s interval, 10 markets max, 500ms delay)
4. `x-ai/grok-3-latest` as default per user request
5. Empty filter arrays = analyze all markets (flexible)
