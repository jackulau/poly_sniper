# Polysniper

A high-performance Polymarket trading bot written in Rust, designed for sub-100ms order latency.

## Overview

Polysniper is an automated trading system for [Polymarket](https://polymarket.com), the leading prediction market platform. It monitors markets in real-time via WebSocket connections and executes trades based on configurable strategies.

## Features

### Trading Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Target Price** | Execute trades when price reaches configured levels | Set buy/sell targets for specific markets |
| **Price Spike** | Detect and trade on sudden price movements | Momentum or mean-reversion trading |
| **New Market** | Enter newly created markets early | First-mover advantage on new questions |
| **Event-Based** | React to external signals (webhooks, RSS) | News-driven trading |

### Risk Management

- **Position Limits** - Maximum exposure per market and total portfolio
- **Order Size Limits** - Cap on individual order sizes
- **Daily Loss Limits** - Stop trading when daily losses exceed threshold
- **Circuit Breaker** - Halt all trading on severe losses
- **Rate Limiting** - Respect API limits and prevent over-trading

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL DATA SOURCES                         │
├─────────────┬─────────────┬─────────────┬───────────────────────┤
│ CLOB WS     │ Gamma REST  │ RTDS WS     │ Webhooks/RSS          │
└──────┬──────┴──────┬──────┴──────┬──────┴───────────┬───────────┘
       │             │             │                   │
       ▼             ▼             ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│              DATA INGESTION LAYER (async)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVENT BUS (tokio broadcast)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ STRATEGY ENGINE │ │  STATE MANAGER  │ │  RISK MANAGER   │
└────────┬────────┘ └─────────────────┘ └────────┬────────┘
         │                                       │
         └───────────────────┬───────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION ENGINE                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    CLOB REST API
```

## Project Structure

```
polysniper/
├── Cargo.toml                    # Workspace manifest
├── LICENSE                       # Apache 2.0 License
├── README.md                     # This file
├── config/
│   ├── default.toml              # Main configuration
│   └── strategies/               # Strategy-specific configs
│       ├── target_price.toml
│       ├── price_spike.toml
│       ├── new_market.toml
│       └── event_based.toml
├── crates/
│   ├── polysniper-core/          # Types, traits, events
│   ├── polysniper-data/          # WebSocket & API clients
│   ├── polysniper-strategies/    # Strategy implementations
│   ├── polysniper-execution/     # Order building & submission
│   ├── polysniper-risk/          # Risk management
│   ├── polysniper-observability/ # Logging & metrics
│   └── polysniper-persistence/   # Database (Phase 5)
└── src/
    └── main.rs                   # Application entry point
```

## Getting Started

### Prerequisites

- **Rust 1.70+** - Install via [rustup](https://rustup.rs/)
- **Polymarket Account** - With API access enabled
- **Private Key** - EOA wallet for signing orders

### Installation

```bash
# Clone the repository
git clone https://github.com/jacklau/Polysniper.git
cd Polysniper

# Build in release mode
cargo build --release
```

### Configuration

1. **Set your private key:**
```bash
export POLYMARKET_PRIVATE_KEY="0x..."
```

2. **Configure the main settings** in `config/default.toml`:
```toml
[execution]
dry_run = true  # Start with dry run!

[risk]
max_position_size_usd = 5000
max_order_size_usd = 500
daily_loss_limit_usd = 500
```

3. **Enable and configure strategies** in `config/strategies/`

### Running

```bash
# Dry run mode (default - logs orders without submitting)
./target/release/polysniper

# With custom config path
POLYSNIPER_CONFIG=config/local.toml ./target/release/polysniper

# With debug logging
LOG_LEVEL=DEBUG ./target/release/polysniper

# JSON log format (for log aggregation)
LOG_FORMAT=json ./target/release/polysniper
```

## Configuration Reference

### Main Configuration

| Section | Key | Description | Default |
|---------|-----|-------------|---------|
| `endpoints` | `clob_rest` | CLOB REST API URL | `https://clob.polymarket.com` |
| `endpoints` | `clob_ws` | CLOB WebSocket URL | `wss://ws-subscriptions-clob.polymarket.com/ws/` |
| `endpoints` | `gamma_api` | Gamma API URL | `https://gamma-api.polymarket.com` |
| `auth` | `private_key_env` | Env var for private key | `POLYMARKET_PRIVATE_KEY` |
| `auth` | `signature_type` | 0=EOA, 1=PolyProxy, 2=GnosisSafe | `0` |
| `risk` | `max_position_size_usd` | Max position per market | `5000` |
| `risk` | `max_order_size_usd` | Max single order size | `500` |
| `risk` | `daily_loss_limit_usd` | Daily loss limit | `500` |
| `risk` | `circuit_breaker_loss_usd` | Halt threshold | `300` |
| `execution` | `dry_run` | Log orders without submitting | `true` |
| `execution` | `max_retries` | Order submission retries | `3` |

### Strategy: Target Price

Execute trades when prices reach specified levels.

```toml
[strategy]
enabled = true

[[strategy.targets]]
market_id = "0x..."           # Condition ID
token_id = "0x..."            # YES or NO token
outcome = "Yes"               # "Yes" or "No"
target_price = "0.30"         # Trigger price
direction = "buy_below"       # "buy_below" or "sell_above"
size_usd = "100"              # Order size in USD
one_shot = true               # Only trigger once
```

### Strategy: Price Spike

Trade on sudden price movements.

```toml
[strategy]
enabled = true
spike_threshold_pct = "5.0"   # Minimum % change to trigger
time_window_secs = 10         # Time window for measurement
order_size_usd = "100"        # Order size in USD
cooldown_secs = 60            # Cooldown after trigger
trade_direction = "momentum"  # "momentum" or "reversion"
```

### Strategy: New Market

Enter newly created markets.

```toml
[strategy]
enabled = true
order_size_usd = "100"        # Order size in USD
max_age_secs = 300            # Max market age (5 min)
keywords = ["Trump", "election"]  # Filter by keywords
max_entry_price = "0.50"      # Max price for entry
```

### Strategy: Event-Based

React to external signals.

```toml
[strategy]
enabled = true

[[strategy.rules]]
name = "Breaking News"
keywords = ["breaking", "confirmed"]
[strategy.rules.action]
market_keywords = ["election"]
side = "Buy"
outcome = "Yes"
order_size_usd = "50"
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `POLYMARKET_PRIVATE_KEY` | Wallet private key for signing | Yes |
| `POLYSNIPER_CONFIG` | Path to config file | No |
| `LOG_LEVEL` | DEBUG, INFO, WARN, ERROR | No |
| `LOG_FORMAT` | pretty, json, compact | No |

## Safety Guidelines

1. **Always start with `dry_run = true`** to verify your configuration
2. **Use small order sizes** ($5-10) when first going live
3. **Monitor logs closely** during initial live trading
4. **Set conservative risk limits** until you understand the system
5. **Test strategies on paper** before deploying capital

The circuit breaker will automatically halt trading if daily losses exceed the configured threshold.

## Development

### Running Tests

```bash
cargo test
```

### Building Documentation

```bash
cargo doc --open
```

### Code Structure

- **polysniper-core**: Shared types, traits, and error definitions
- **polysniper-data**: WebSocket clients, API clients, state management
- **polysniper-strategies**: Trading strategy implementations
- **polysniper-execution**: Order building and submission
- **polysniper-risk**: Risk validation and circuit breakers
- **polysniper-observability**: Structured logging, Prometheus metrics, alerting
- **polysniper-persistence**: SQLite database for trade history and state

## Observability & Monitoring

### Prometheus Metrics (Phase 6)

Metrics are exposed on port 9090 by default. Available at `http://localhost:9090/metrics`.

**Key Metrics:**
- `polysniper_trades_executed_total` - Trade counts by strategy and outcome
- `polysniper_signals_generated_total` - Signal generation rates
- `polysniper_risk_rejections_total` - Risk manager rejections
- `polysniper_event_processing_duration_seconds` - Processing latency
- `polysniper_daily_pnl_usd` - Current daily P&L

```toml
[metrics]
enabled = true
port = 9090
collection_interval_secs = 60
```

### Alerting (Phase 7)

Configure Slack and/or Telegram notifications for critical events.

```toml
[alerting]
enabled = true
min_level = "warning"  # info, warning, critical

[alerting.slack]
enabled = true
webhook_url = "https://hooks.slack.com/services/..."
channel = "#polysniper-alerts"

[alerting.telegram]
enabled = true
bot_token = "123456:ABC..."
chat_id = "-100123456789"
```

**Alert Types:**
- Circuit breaker triggered
- Connection lost
- High daily loss warnings
- Strategy errors
- New market discoveries

### Persistence (Phase 5)

Trade history, orders, and strategy state are persisted to SQLite.

```toml
[persistence]
enabled = true
db_path = "data/polysniper.db"
price_snapshot_interval_secs = 60
max_price_snapshots = 10000
```

## Roadmap

- [x] Phase 1: Core infrastructure + Target Price strategy
- [x] Phase 2: All 4 trading strategies
- [x] Phase 3: Risk management
- [x] Phase 4: Observability (logging)
- [x] Phase 5: Persistence (SQLite)
- [x] Phase 6: Metrics (Prometheus)
- [x] Phase 7: Alerting (Slack/Telegram)

## Disclaimer

This software is provided for educational and research purposes. Trading on prediction markets involves financial risk. The authors are not responsible for any losses incurred from using this software. Always understand the risks before trading.

## License

Copyright 2026 Polysniper Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
