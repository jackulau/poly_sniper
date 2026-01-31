//! Polysniper - High-Performance Polymarket Sniper
//!
//! A Rust-based trading bot for Polymarket with multiple sniping strategies.

mod commands;

use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Parser, Subcommand};
use commands::StatsCommand;
use polysniper_core::{
    AppConfig, ConfigChangedEvent, ConfigType, DiscordConfig, EventBus, HeartbeatEvent,
    OrderExecutor, RiskDecision, RiskValidator, StateManager, StateProvider, Strategy,
    SystemEvent, TradeSignal,
};
use polysniper_data::{BroadcastEventBus, ConnectionConfig, GammaClient, HttpPool, HttpPoolConfig, MarketCache, WebhookServer, WsManager};
use polysniper_discord::DiscordNotifier;
use polysniper_execution::{GasOptimizer, OrderBuilder, OrderSubmitter};
use polysniper_observability::{
    init_logging, record_event_processing, record_new_market, record_order, record_risk_rejection,
    record_signal, record_strategy_error, record_strategy_processing, start_metrics_server,
    update_markets_monitored, update_uptime, AlertManager, AlertingConfig, LogFormat, SlackConfig,
    TelegramConfig,
};
use polysniper_persistence::{
    calculate_trade_pnl, CostBasisMethod, DailyPnlRepository, Database,
    PositionHistoryRepository, TradeRecord, TradeRepository,
};
use polysniper_risk::{ControlServer, RiskManager, SignalHandler};
use polysniper_strategies::{
    EventBasedConfig, EventBasedStrategy, LlmPredictionConfig, LlmPredictionStrategy,
    NewMarketConfig, NewMarketStrategy, PriceSpikeConfig, PriceSpikeStrategy, TargetPriceConfig,
    TargetPriceStrategy,
};
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tokio::time::interval;
use tracing::{error, info, warn, Level};

/// Polysniper CLI
#[derive(Parser)]
#[command(name = "polysniper")]
#[command(about = "High-Performance Polymarket Sniper", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the trading bot (default)
    Run,
    /// Run a backtest with historical data
    Backtest {
        /// Strategy to backtest
        #[arg(short, long, default_value = "target_price")]
        strategy: String,

        /// Start date (YYYY-MM-DD format)
        #[arg(long)]
        from: Option<String>,

        /// End date (YYYY-MM-DD format)
        #[arg(long)]
        to: Option<String>,

        /// Initial capital in USD
        #[arg(long, default_value = "10000")]
        capital: f64,

        /// Database path
        #[arg(long, default_value = "data/polysniper.db")]
        db_path: String,

        /// Output format (text, json, csv)
        #[arg(long, default_value = "text")]
        output: String,
    },
    /// Launch the terminal UI for orderbook visualization
    Tui {
        /// Market ID to monitor (optional, will show market selector if not provided)
        #[arg(short, long)]
        market: Option<String>,
    },
    /// View statistics and performance metrics
    Stats(StatsCommand),
}

/// Configuration file paths
const DEFAULT_CONFIG_PATH: &str = "config/default.toml";
const STRATEGIES_CONFIG_DIR: &str = "config/strategies";

/// Main application state
struct App {
    config: AppConfig,
    event_bus: Arc<BroadcastEventBus>,
    state: Arc<MarketCache>,
    strategies: Vec<Box<dyn Strategy>>,
    risk_manager: Arc<RiskManager>,
    order_builder: OrderBuilder,
    order_executor: Arc<dyn OrderExecutor>,
    gas_optimizer: Option<Arc<GasOptimizer>>,
    gamma_client: Arc<GammaClient>,
    http_pool: Arc<HttpPool>,
    database: Option<Arc<Database>>,
    alert_manager: Option<Arc<AlertManager>>,
    discord_notifier: Option<Arc<DiscordNotifier>>,
    start_time: Instant,
}

impl App {
    /// Create a new application instance
    async fn new() -> Result<Self> {
        // Load configuration
        let config = Self::load_config()?;

        // Initialize components
        let event_bus = Arc::new(BroadcastEventBus::new());
        let state = Arc::new(MarketCache::new());

        // Initialize HTTP connection pool with warmup
        let http_pool_config = HttpPoolConfig {
            max_idle_per_host: config.connection.http_max_idle_per_host,
            idle_timeout_secs: config.connection.http_idle_timeout_secs,
            connect_timeout_secs: config.connection.http_connect_timeout_secs,
            tcp_keepalive_secs: config.connection.http_tcp_keepalive_secs,
            request_timeout_secs: config.connection.http_request_timeout_secs,
            warmup_urls: config.connection.warmup_urls.clone(),
        };
        let http_pool = Arc::new(
            HttpPool::new(http_pool_config).context("Failed to create HTTP pool")?,
        );

        // Initialize Gamma client for market discovery
        let gamma_client = Arc::new(GammaClient::new(config.endpoints.gamma_api.clone()));

        // Initialize risk manager
        let risk_manager = Arc::new(RiskManager::new(config.risk.clone()));

        // Initialize order builder and executor
        let order_builder = OrderBuilder::new();

        // Get wallet address from private key (in production, derive from key)
        let wallet_address = std::env::var(&config.auth.private_key_env)
            .map(|_| "0x0000000000000000000000000000000000000000".to_string()) // Placeholder
            .unwrap_or_else(|_| "0x0000000000000000000000000000000000000000".to_string());

        let base_executor: Arc<dyn OrderExecutor> = Arc::new(OrderSubmitter::new(
            config.endpoints.clob_rest.clone(),
            wallet_address,
            config.auth.signature_type,
            config.execution.clone(),
        ));

        // Wrap with gas optimizer if enabled
        let (order_executor, gas_optimizer): (Arc<dyn OrderExecutor>, Option<Arc<GasOptimizer>>) =
            if config.gas.optimization.enabled {
                let (optimizer, _handle) =
                    GasOptimizer::new(base_executor, config.gas.optimization.clone());
                let optimizer = Arc::new(optimizer);
                (optimizer.clone(), Some(optimizer))
            } else {
                (base_executor, None)
            };

        // Load strategies
        let strategies = Self::load_strategies()?;

        // Initialize database (Phase 5)
        let database = if config.persistence.enabled {
            match Database::new(&config.persistence.db_path).await {
                Ok(db) => {
                    info!(db_path = %config.persistence.db_path, "Database initialized");
                    Some(Arc::new(db))
                }
                Err(e) => {
                    error!(error = %e, "Failed to initialize database, continuing without persistence");
                    None
                }
            }
        } else {
            None
        };

        // Initialize alert manager (Phase 7)
        let alert_manager = if config.alerting.enabled {
            let alerting_config = Self::build_alerting_config(&config);
            Some(Arc::new(AlertManager::new(alerting_config)))
        } else {
            None
        };

        // Initialize Discord notifier
        let discord_notifier = if config.discord.enabled {
            match DiscordNotifier::new(config.discord.clone()) {
                Ok(notifier) => {
                    info!("Discord notifier initialized");
                    Some(Arc::new(notifier))
                }
                Err(e) => {
                    warn!(error = %e, "Failed to initialize Discord notifier, continuing without it");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config,
            event_bus,
            state,
            strategies,
            risk_manager,
            order_builder,
            order_executor,
            gas_optimizer,
            gamma_client,
            http_pool,
            database,
            alert_manager,
            discord_notifier,
            start_time: Instant::now(),
        })
    }

    /// Build alerting config from app config
    fn build_alerting_config(config: &AppConfig) -> AlertingConfig {
        use polysniper_observability::alerting::AlertLevel;

        AlertingConfig {
            enabled: config.alerting.enabled,
            min_level: match config.alerting.min_level.as_str() {
                "info" => AlertLevel::Info,
                "critical" => AlertLevel::Critical,
                _ => AlertLevel::Warning,
            },
            slack: SlackConfig {
                enabled: config.alerting.slack.enabled,
                webhook_url: config.alerting.slack.webhook_url.clone(),
                channel: config.alerting.slack.channel.clone(),
                username: config.alerting.slack.username.clone(),
                icon_emoji: config.alerting.slack.icon_emoji.clone(),
            },
            telegram: TelegramConfig {
                enabled: config.alerting.telegram.enabled,
                bot_token: config.alerting.telegram.bot_token.clone(),
                chat_id: config.alerting.telegram.chat_id.clone(),
                parse_mode: config.alerting.telegram.parse_mode.clone(),
            },
            rate_limit_seconds: config.alerting.rate_limit_seconds,
        }
    }

    /// Load main configuration
    fn load_config() -> Result<AppConfig> {
        let config_path =
            std::env::var("POLYSNIPER_CONFIG").unwrap_or_else(|_| DEFAULT_CONFIG_PATH.to_string());

        let mut config = if std::path::Path::new(&config_path).exists() {
            let content = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config file: {}", config_path))?;
            toml::from_str(&content)
                .with_context(|| format!("Failed to parse config file: {}", config_path))?
        } else {
            info!("Config file not found, using defaults");
            AppConfig::default()
        };

        // Apply Discord webhook URL from environment variable (security best practice)
        if let Ok(webhook_url) = std::env::var("DISCORD_WEBHOOK_URL") {
            if !webhook_url.is_empty() {
                config.discord.webhook_url = Some(webhook_url);
            }
        }

        // Validate Discord configuration
        Self::validate_discord_config(&config.discord);

        Ok(config)
    }

    /// Validate Discord configuration and log warnings
    fn validate_discord_config(config: &DiscordConfig) {
        if config.enabled {
            match &config.webhook_url {
                Some(url) if url.is_empty() => {
                    warn!("Discord is enabled but webhook_url is empty. Set DISCORD_WEBHOOK_URL environment variable.");
                }
                Some(url) if !url.starts_with("https://discord.com/api/webhooks/") => {
                    warn!(
                        "Discord webhook URL does not match expected format (https://discord.com/api/webhooks/...). URL: {}",
                        url
                    );
                }
                None => {
                    warn!("Discord is enabled but no webhook_url provided. Set DISCORD_WEBHOOK_URL environment variable.");
                }
                _ => {}
            }
        }
    }

    /// Load strategy configurations
    fn load_strategies() -> Result<Vec<Box<dyn Strategy>>> {
        let mut strategies: Vec<Box<dyn Strategy>> = Vec::new();

        // Load Target Price Strategy
        let target_price_config = Self::load_strategy_config::<TargetPriceConfig>("target_price")?
            .unwrap_or_else(|| TargetPriceConfig {
                enabled: false,
                targets: Vec::new(),
            });
        if target_price_config.enabled {
            strategies.push(Box::new(TargetPriceStrategy::from_config(
                target_price_config,
            )));
            info!("Loaded Target Price strategy");
        }

        // Load Price Spike Strategy
        let price_spike_config =
            Self::load_strategy_config::<PriceSpikeConfig>("price_spike")?.unwrap_or_default();
        if price_spike_config.enabled {
            strategies.push(Box::new(PriceSpikeStrategy::new(price_spike_config)));
            info!("Loaded Price Spike strategy");
        }

        // Load New Market Strategy
        let new_market_config =
            Self::load_strategy_config::<NewMarketConfig>("new_market")?.unwrap_or_default();
        if new_market_config.enabled {
            strategies.push(Box::new(NewMarketStrategy::new(new_market_config)));
            info!("Loaded New Market strategy");
        }

        // Load Event-Based Strategy
        let event_based_config =
            Self::load_strategy_config::<EventBasedConfig>("event_based")?.unwrap_or_default();
        if event_based_config.enabled {
            strategies.push(Box::new(EventBasedStrategy::new(event_based_config)));
            info!("Loaded Event-Based strategy");
        }

        // Load LLM Prediction Strategy
        let llm_prediction_config =
            Self::load_strategy_config::<LlmPredictionConfig>("llm_prediction")?
                .unwrap_or_default();
        if llm_prediction_config.enabled {
            match LlmPredictionStrategy::new(llm_prediction_config) {
                Ok(strategy) => {
                    strategies.push(Box::new(strategy));
                    info!("Loaded LLM Prediction Strategy");
                }
                Err(e) => {
                    warn!(
                        "Failed to initialize LLM Prediction Strategy: {}. Skipping.",
                        e
                    );
                }
            }
        }

        Ok(strategies)
    }

    /// Load a specific strategy configuration
    fn load_strategy_config<T: serde::de::DeserializeOwned>(name: &str) -> Result<Option<T>> {
        let config_path = format!("{}/{}.toml", STRATEGIES_CONFIG_DIR, name);
        if std::path::Path::new(&config_path).exists() {
            let content = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read strategy config: {}", config_path))?;
            let config: T = toml::from_str(&content)
                .with_context(|| format!("Failed to parse strategy config: {}", config_path))?;
            Ok(Some(config))
        } else {
            Ok(None)
        }
    }

    /// Initialize all strategies
    async fn initialize_strategies(&mut self) -> Result<()> {
        for strategy in &mut self.strategies {
            strategy
                .initialize(self.state.as_ref())
                .await
                .with_context(|| format!("Failed to initialize strategy: {}", strategy.id()))?;
        }
        Ok(())
    }

    /// Start the main event loop
    async fn run(&mut self) -> Result<()> {
        info!("Starting Polysniper...");

        // Initialize strategies
        self.initialize_strategies().await?;

        // Warm up HTTP connections before starting trading
        if self.http_pool.has_warmup_urls() {
            info!("Warming up HTTP connections...");
            match self.http_pool.warmup().await {
                Ok(result) => {
                    if result.all_successful() {
                        info!(
                            count = result.successful,
                            avg_latency_ms = result.avg_latency_ms().unwrap_or(0),
                            "HTTP connections warmed up successfully"
                        );
                    } else {
                        warn!(
                            successful = result.successful,
                            failed = result.failed,
                            "Some HTTP warmup connections failed"
                        );
                    }
                }
                Err(e) => {
                    warn!(error = %e, "HTTP connection warmup failed, continuing anyway");
                }
            }
        }

        // Load initial markets from Gamma
        self.load_initial_markets().await?;

        // Start metrics server (Phase 6)
        if self.config.metrics.enabled {
            let _metrics_handle = start_metrics_server(self.config.metrics.port).await;
            info!(port = %self.config.metrics.port, "Metrics server started");
        }

        // Start webhook server for ML predictions
        if self.config.webhook.enabled {
            let webhook_server =
                WebhookServer::new(self.config.webhook.clone(), self.event_bus.clone());
            match webhook_server.start().await {
                Ok(_handle) => {
                    info!(
                        host = %self.config.webhook.host,
                        port = %self.config.webhook.port,
                        "Webhook server started"
                    );
                }
                Err(e) => {
                    error!(error = %e, "Failed to start webhook server");
                }
            }
        }

        // Create shutdown signal
        let (shutdown_tx, mut shutdown_rx) = broadcast::channel::<()>(1);

        // Start control server (emergency kill switch)
        if self.config.control.enabled {
            let control_config = self.config.control.clone();
            let control_server = ControlServer::new(
                control_config.clone(),
                self.risk_manager.clone(),
                self.state.clone(),
                shutdown_tx.clone(),
            );
            let control_addr = control_server.address();

            tokio::spawn(async move {
                if let Err(e) = control_server.run().await {
                    error!("Control server error: {}", e);
                }
            });

            info!(
                address = %control_addr,
                "Control server started (endpoints: /halt, /resume, /status, /positions)"
            );

            // Start signal handlers if enabled
            if control_config.signal_handlers {
                let signal_handler = SignalHandler::new(
                    self.risk_manager.clone(),
                    self.state.clone(),
                    shutdown_tx.clone(),
                );
                signal_handler.start().await;
            }
        }

        // Spawn WebSocket manager with connection warmup configuration
        let ws_config = ConnectionConfig {
            ping_interval: std::time::Duration::from_millis(self.config.connection.ws_ping_interval_ms),
            pong_timeout: std::time::Duration::from_millis(self.config.connection.ws_pong_timeout_ms),
            initial_reconnect_delay: std::time::Duration::from_millis(self.config.connection.ws_initial_reconnect_delay_ms),
            max_reconnect_delay: std::time::Duration::from_millis(self.config.connection.ws_max_reconnect_delay_ms),
            backoff_multiplier: self.config.connection.ws_backoff_multiplier,
            jitter_factor: self.config.connection.ws_jitter_factor,
            message_timeout: std::time::Duration::from_secs(60),
        };
        let ws_manager = WsManager::with_config(
            self.config.endpoints.clob_ws.clone(),
            self.event_bus.sender(),
            ws_config,
        );

        // Subscribe to markets for price updates
        for market in self.state.get_all_markets().await {
            ws_manager
                .subscribe_token(market.yes_token_id.clone(), market.condition_id.clone())
                .await;
            ws_manager
                .subscribe_token(market.no_token_id, market.condition_id)
                .await;
        }

        // Spawn WebSocket task
        let ws_handle = tokio::spawn(async move {
            if let Err(e) = ws_manager.run().await {
                error!("WebSocket error: {}", e);
            }
        });

        // Spawn Gamma polling task for new markets
        let gamma_client = self.gamma_client.clone();
        let state = self.state.clone();
        let event_bus = self.event_bus.clone();
        let alert_manager = self.alert_manager.clone();
        let gamma_handle = tokio::spawn(async move {
            let mut poll_interval = interval(Duration::from_secs(5));
            loop {
                poll_interval.tick().await;
                if let Err(e) =
                    poll_gamma_markets(&gamma_client, &state, &event_bus, alert_manager.as_ref())
                        .await
                {
                    warn!("Gamma polling error: {}", e);
                }
            }
        });

        // Spawn uptime tracker
        let start_time = self.start_time;
        tokio::spawn(async move {
            let mut uptime_interval = interval(Duration::from_secs(60));
            loop {
                uptime_interval.tick().await;
                update_uptime(start_time.elapsed().as_secs() as i64);
            }
        });

        // Spawn heartbeat emitter for strategies (e.g., LLM prediction)
        let heartbeat_event_bus = self.event_bus.clone();
        tokio::spawn(async move {
            let mut heartbeat_interval = interval(Duration::from_secs(60));
            loop {
                heartbeat_interval.tick().await;
                let heartbeat = SystemEvent::Heartbeat(HeartbeatEvent {
                    source: "main".to_string(),
                    timestamp: Utc::now(),
                });
                heartbeat_event_bus.publish(heartbeat);
            }
        });

        // Main event processing loop
        let mut event_rx = self.event_bus.subscribe();

        info!(
            "Polysniper started. Dry run mode: {}",
            self.config.execution.dry_run
        );
        info!("Loaded {} strategies", self.strategies.len());
        info!("Monitoring {} markets", self.state.market_count().await);

        // Update initial market count metric
        update_markets_monitored(self.state.market_count().await as i64);

        // Handle Ctrl+C for graceful shutdown
        let shutdown_tx_clone = shutdown_tx.clone();
        tokio::spawn(async move {
            tokio::signal::ctrl_c().await.ok();
            info!("Shutdown signal received");
            let _ = shutdown_tx_clone.send(());
        });

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("Shutting down...");
                    break;
                }
                event = event_rx.recv() => {
                    match event {
                        Ok(event) => {
                            let event_start = Instant::now();
                            let event_type = event.event_type().to_string();

                            if let Err(e) = self.process_event(event).await {
                                error!("Error processing event: {}", e);
                            }

                            // Record event processing metrics
                            record_event_processing(
                                &event_type,
                                event_start.elapsed().as_secs_f64()
                            );
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            warn!("Event bus lagged by {} messages", n);
                            polysniper_observability::metrics::EVENT_BUS_LAG.set(n as i64);
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            error!("Event bus closed");
                            break;
                        }
                    }
                }
            }
        }

        // Cleanup
        ws_handle.abort();
        gamma_handle.abort();
        // config_watcher_handle.abort(); // TODO: Implement config watcher

        // Close database connection
        if let Some(db) = &self.database {
            db.close().await;
        }

        info!("Polysniper stopped");
        Ok(())
    }

    /// Load initial markets from Gamma API
    async fn load_initial_markets(&self) -> Result<()> {
        info!("Loading markets from Gamma API...");

        let (markets, _) = self
            .gamma_client
            .fetch_markets(Some(100), None)
            .await
            .context("Failed to fetch markets from Gamma")?;

        for market in markets {
            self.state.update_market(market).await;
        }

        info!("Loaded {} markets", self.state.market_count().await);
        Ok(())
    }

    /// Handle config change event
    async fn handle_config_change(&mut self, event: &ConfigChangedEvent) {
        info!(
            path = %event.path.display(),
            config_type = ?event.config_type,
            "Config change detected"
        );

        // Read the file content
        let content = match std::fs::read_to_string(&event.path) {
            Ok(c) => c,
            Err(e) => {
                error!(
                    path = %event.path.display(),
                    error = %e,
                    "Failed to read config file"
                );
                return;
            }
        };

        match &event.config_type {
            ConfigType::Main => {
                // Reload main config - update risk manager and other settings
                match toml::from_str::<AppConfig>(&content) {
                    Ok(new_config) => {
                        // Update risk manager config
                        self.risk_manager.update_config(new_config.risk.clone()).await;
                        info!("Reloaded main configuration");

                        // Record config reload metric
                        polysniper_observability::metrics::CONFIG_RELOADS.inc();
                    }
                    Err(e) => {
                        error!(
                            error = %e,
                            "Failed to parse main config, skipping reload"
                        );
                    }
                }
            }
            ConfigType::Strategy(strategy_name) => {
                // Find the matching strategy and reload its config
                let mut found = false;
                for strategy in &mut self.strategies {
                    if strategy.config_name() == strategy_name {
                        found = true;
                        match strategy.reload_config(&content).await {
                            Ok(()) => {
                                info!(
                                    strategy = %strategy_name,
                                    "Reloaded strategy configuration"
                                );
                                polysniper_observability::metrics::CONFIG_RELOADS.inc();
                            }
                            Err(e) => {
                                error!(
                                    strategy = %strategy_name,
                                    error = %e,
                                    "Failed to reload strategy config"
                                );
                            }
                        }
                        break;
                    }
                }
                if !found {
                    warn!(
                        strategy = %strategy_name,
                        "No matching strategy found for config change"
                    );
                }
            }
        }
    }

    /// Process a single event
    async fn process_event(&mut self, event: SystemEvent) -> Result<()> {
        // Handle config change events separately
        if let SystemEvent::ConfigChanged(ref config_event) = event {
            self.handle_config_change(config_event).await;
            return Ok(());
        }

        // Update state based on event
        match &event {
            SystemEvent::OrderbookUpdate(e) => {
                self.state.update_orderbook(e.orderbook.clone()).await;
            }
            SystemEvent::PriceChange(e) => {
                self.state
                    .update_price(e.token_id.clone(), e.new_price)
                    .await;
            }
            SystemEvent::NewMarket(e) => {
                self.state.update_market(e.market.clone()).await;
                record_new_market();
                update_markets_monitored(self.state.market_count().await as i64);
            }
            SystemEvent::GasPriceUpdate(e) => {
                // Forward gas price updates to the optimizer
                if let Some(optimizer) = &self.gas_optimizer {
                    optimizer.update_gas_price(e.gas_price.clone()).await;
                    optimizer.update_gas_condition(e.condition).await;
                }
            }
            _ => {}
        }

        // Collect signals from all strategies
        let mut signals: Vec<TradeSignal> = Vec::new();

        for strategy in &self.strategies {
            if !strategy.is_enabled() {
                continue;
            }

            if !strategy.accepts_event(&event) {
                continue;
            }

            let strategy_start = Instant::now();
            match strategy.process_event(&event, self.state.as_ref()).await {
                Ok(strategy_signals) => {
                    record_strategy_processing(
                        strategy.id(),
                        strategy_start.elapsed().as_secs_f64(),
                    );
                    signals.extend(strategy_signals);
                }
                Err(e) => {
                    warn!(
                        strategy_id = %strategy.id(),
                        error = %e,
                        "Strategy error processing event"
                    );
                    record_strategy_error(strategy.id());

                    // Alert on strategy error
                    if let Some(alert_mgr) = &self.alert_manager {
                        alert_mgr
                            .alert_strategy_error(strategy.id(), &e.to_string())
                            .await;
                    }

                    // Also notify via Discord
                    if let Some(ref discord) = self.discord_notifier {
                        if let Err(discord_err) = discord
                            .notify_error(&format!("Strategy Error: {}", strategy.id()), &e.to_string())
                            .await
                        {
                            warn!(error = %discord_err, "Failed to send Discord strategy error alert");
                        }
                    }
                }
            }
        }

        // Process signals in priority order
        if !signals.is_empty() {
            self.process_signals(signals).await?;
        }

        Ok(())
    }

    /// Process trade signals
    async fn process_signals(&self, signals: Vec<TradeSignal>) -> Result<()> {
        // Sort by priority (highest first)
        let mut sorted_signals = signals;
        sorted_signals.sort_by(|a, b| b.priority.cmp(&a.priority));

        for signal in sorted_signals {
            info!(
                signal_id = %signal.id,
                strategy = %signal.strategy_id,
                market = %signal.market_id,
                side = %signal.side,
                size = %signal.size,
                price = ?signal.price,
                priority = ?signal.priority,
                "Processing trade signal"
            );

            // Record signal metric
            record_signal(&signal.strategy_id, &signal.side.to_string());

            // Validate with risk manager
            match self
                .risk_manager
                .validate(&signal, self.state.as_ref())
                .await
            {
                Ok(RiskDecision::Approved) => {
                    // Build and submit order
                    let order = self.order_builder.build_from_signal(&signal);

                    // Persist order if database is available
                    if let Some(db) = &self.database {
                        use polysniper_persistence::OrderRepository;
                        let order_repo = OrderRepository::new(db);
                        if let Err(e) = order_repo.insert(&order).await {
                            warn!(error = %e, "Failed to persist order");
                        }
                    }

                    // Track with order manager for GTC limit orders
                    // TODO: Implement order management
                    // let should_manage =
                    //     matches!(signal.order_type, polysniper_core::OrderType::Gtc);
                    // if should_manage {
                    //     if let Err(e) = self.order_manager.manage_order(order.clone(), None).await {
                    //         warn!(error = %e, "Failed to start managing order");
                    //     }
                    // }

                    match self.order_executor.submit_order(order).await {
                        Ok(order_id) => {
                            info!(
                                signal_id = %signal.id,
                                order_id = %order_id,
                                "Order submitted successfully"
                            );
                            self.risk_manager.record_order(&signal.market_id).await;
                            record_order(
                                &signal.strategy_id,
                                &signal.side.to_string(),
                                "submitted",
                            );

                            // Persist trade and update P&L if database is available
                            if let Some(db) = &self.database {
                                // Get current position to calculate P&L
                                let current_position = self.state.get_position(&signal.market_id).await;
                                let (current_size, current_avg_price) = current_position
                                    .as_ref()
                                    .map(|p| (p.size, p.avg_price))
                                    .unwrap_or((Decimal::ZERO, Decimal::ZERO));

                                // Calculate estimated fees (0.1% taker fee)
                                let executed_price = signal.price.unwrap_or_default();
                                let fees = signal.size_usd * Decimal::new(1, 3); // 0.001 = 0.1%

                                // Calculate P&L for this trade
                                let pnl_result = calculate_trade_pnl(
                                    current_size,
                                    current_avg_price,
                                    signal.side,
                                    executed_price,
                                    signal.size,
                                    fees,
                                    CostBasisMethod::Average,
                                );

                                let trade = TradeRecord {
                                    id: uuid::Uuid::new_v4().to_string(),
                                    order_id: order_id.clone(),
                                    signal_id: signal.id.clone(),
                                    strategy_id: signal.strategy_id.clone(),
                                    market_id: signal.market_id.clone(),
                                    token_id: signal.token_id.clone(),
                                    side: signal.side,
                                    executed_price,
                                    executed_size: signal.size,
                                    size_usd: signal.size_usd,
                                    fees,
                                    realized_pnl: if pnl_result.realized_pnl != Decimal::ZERO {
                                        Some(pnl_result.realized_pnl)
                                    } else {
                                        None
                                    },
                                    timestamp: chrono::Utc::now(),
                                    metadata: Some(signal.metadata.clone()),
                                };

                                // Insert trade record
                                let trade_repo = TradeRepository::new(db);
                                if let Err(e) = trade_repo.insert(&trade).await {
                                    warn!(error = %e, "Failed to persist trade");
                                }

                                // Update daily P&L if there was realized P&L
                                if pnl_result.realized_pnl != Decimal::ZERO {
                                    let daily_pnl_repo = DailyPnlRepository::new(db);
                                    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();

                                    // Ensure today's record exists
                                    let starting_balance = self.state.get_portfolio_value().await;
                                    if let Err(e) = daily_pnl_repo.get_or_create_today(starting_balance).await {
                                        warn!(error = %e, "Failed to create daily P&L record");
                                    }

                                    // Add realized P&L
                                    if let Err(e) = daily_pnl_repo
                                        .add_realized_pnl(&today, pnl_result.realized_pnl, pnl_result.is_win)
                                        .await
                                    {
                                        warn!(error = %e, "Failed to update daily P&L");
                                    }

                                    // Update in-memory daily P&L
                                    self.state.update_daily_pnl(pnl_result.realized_pnl).await;

                                    info!(
                                        trade_id = %trade.id,
                                        realized_pnl = %pnl_result.realized_pnl,
                                        is_win = %pnl_result.is_win,
                                        "Recorded realized P&L"
                                    );
                                }

                                // Update position in MarketCache
                                let updated_position = polysniper_core::Position {
                                    market_id: signal.market_id.clone(),
                                    token_id: signal.token_id.clone(),
                                    outcome: signal.outcome,
                                    size: pnl_result.new_size,
                                    avg_price: pnl_result.new_avg_price,
                                    realized_pnl: current_position
                                        .as_ref()
                                        .map(|p| p.realized_pnl + pnl_result.realized_pnl)
                                        .unwrap_or(pnl_result.realized_pnl),
                                    unrealized_pnl: Decimal::ZERO, // Will be updated by price changes
                                    updated_at: chrono::Utc::now(),
                                };
                                self.state.update_position(updated_position).await;

                                // Track position history
                                let pos_history_repo = PositionHistoryRepository::new(db);
                                match signal.side {
                                    polysniper_core::Side::Buy => {
                                        // Opening or adding to position
                                        if current_size.is_zero() {
                                            // New position
                                            if let Err(e) = pos_history_repo
                                                .open_position(
                                                    &signal.market_id,
                                                    &signal.token_id,
                                                    signal.side,
                                                    executed_price,
                                                    signal.size,
                                                    fees,
                                                    Some(&signal.strategy_id),
                                                )
                                                .await
                                            {
                                                warn!(error = %e, "Failed to record position open");
                                            }
                                        }
                                    }
                                    polysniper_core::Side::Sell => {
                                        // Reducing or closing position
                                        if pnl_result.new_size.is_zero() {
                                            // Position closed - find and update position history
                                            if let Ok(Some(open_pos)) = pos_history_repo
                                                .get_open_position(&signal.market_id, &signal.token_id)
                                                .await
                                            {
                                                if let Some(pos_id) = open_pos.id {
                                                    if let Err(e) = pos_history_repo
                                                        .close_position(
                                                            pos_id,
                                                            executed_price,
                                                            pnl_result.realized_pnl,
                                                            fees,
                                                        )
                                                        .await
                                                    {
                                                        warn!(error = %e, "Failed to record position close");
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            error!(
                                signal_id = %signal.id,
                                error = %e,
                                "Order submission failed"
                            );
                            record_order(&signal.strategy_id, &signal.side.to_string(), "failed");
                        }
                    }
                }
                Ok(RiskDecision::Modified { new_size, reason }) => {
                    info!(
                        signal_id = %signal.id,
                        new_size = %new_size,
                        reason = %reason,
                        "Signal modified by risk manager"
                    );
                    polysniper_observability::metrics::RISK_MODIFICATIONS.inc();

                    // Build order with modified size
                    let mut modified_signal = signal.clone();
                    modified_signal.size = new_size;
                    let order = self.order_builder.build_from_signal(&modified_signal);
                    match self.order_executor.submit_order(order).await {
                        Ok(order_id) => {
                            info!(
                                signal_id = %signal.id,
                                order_id = %order_id,
                                "Modified order submitted"
                            );
                            self.risk_manager.record_order(&signal.market_id).await;
                            record_order(&signal.strategy_id, &signal.side.to_string(), "modified");
                        }
                        Err(e) => {
                            error!(
                                signal_id = %signal.id,
                                error = %e,
                                "Modified order submission failed"
                            );
                        }
                    }
                }
                Ok(RiskDecision::Rejected { reason }) => {
                    warn!(
                        signal_id = %signal.id,
                        reason = %reason,
                        "Signal rejected by risk manager"
                    );
                    record_risk_rejection(&reason);

                    // Notify via Discord for risk rejections
                    if let Some(ref discord) = self.discord_notifier {
                        if let Err(discord_err) = discord
                            .notify_risk_event(&signal.id, &reason)
                            .await
                        {
                            warn!(error = %discord_err, "Failed to send Discord risk rejection alert");
                        }
                    }
                }
                Err(e) => {
                    error!(
                        signal_id = %signal.id,
                        error = %e,
                        "Risk validation error"
                    );

                    // Check for circuit breaker
                    if e.to_string().contains("circuit breaker") {
                        polysniper_observability::metrics::CIRCUIT_BREAKER_TRIGGERED.inc();
                        let daily_pnl = self.state.get_daily_pnl().await;

                        // Alert via existing alert manager
                        if let Some(alert_mgr) = &self.alert_manager {
                            let daily_pnl =
                                self.state.get_daily_pnl().await.to_f64().unwrap_or(0.0);
                            alert_mgr
                                .alert_circuit_breaker(&e.to_string(), daily_pnl.to_f64().unwrap_or(0.0))
                                .await;
                        }

                        // Also notify via Discord
                        if let Some(ref discord) = self.discord_notifier {
                            if let Err(discord_err) = discord
                                .notify_circuit_breaker(&e.to_string(), daily_pnl)
                                .await
                            {
                                warn!(error = %discord_err, "Failed to send Discord circuit breaker alert");
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Poll Gamma for new markets
async fn poll_gamma_markets(
    gamma_client: &GammaClient,
    state: &MarketCache,
    event_bus: &BroadcastEventBus,
    alert_manager: Option<&Arc<AlertManager>>,
) -> Result<()> {
    let (markets, _) = gamma_client.fetch_markets(Some(50), None).await?;

    for market in markets {
        // Check if this is a new market
        if !state.has_market(&market.condition_id).await {
            info!(
                market_id = %market.condition_id,
                question = %market.question,
                "New market discovered"
            );

            // Alert on new market discovery
            if let Some(alert_mgr) = alert_manager {
                alert_mgr
                    .alert_new_market(&market.condition_id, &market.question)
                    .await;
            }

            // Publish new market event
            event_bus.publish(SystemEvent::NewMarket(polysniper_core::NewMarketEvent {
                market: market.clone(),
                discovered_at: chrono::Utc::now(),
            }));

            // Add to state
            state.update_market(market).await;
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Stats(stats_cmd)) => {
            // Stats command doesn't need full logging setup
            stats_cmd.run().await
        }
        Some(Commands::Backtest { strategy, from, to, capital, db_path, output }) => {
            // TODO: Implement backtest command
            println!("Backtesting {} strategy with capital ${}", strategy, capital);
            println!("From: {:?}, To: {:?}", from, to);
            println!("Database: {}, Output: {}", db_path, output);
            Ok(())
        }
        Some(Commands::Tui { market }) => {
            // TODO: Launch TUI
            println!("Launching TUI for market: {:?}", market);
            Ok(())
        }
        Some(Commands::Run) | None => {
            // Initialize logging for the trading bot
            let log_format = std::env::var("LOG_FORMAT")
                .map(|f| match f.as_str() {
                    "json" => LogFormat::Json,
                    "compact" => LogFormat::Compact,
                    _ => LogFormat::Pretty,
                })
                .unwrap_or(LogFormat::Pretty);

            let log_level = std::env::var("LOG_LEVEL")
                .map(|l| match l.to_uppercase().as_str() {
                    "DEBUG" => Level::DEBUG,
                    "TRACE" => Level::TRACE,
                    "WARN" => Level::WARN,
                    "ERROR" => Level::ERROR,
                    _ => Level::INFO,
                })
                .unwrap_or(Level::INFO);

            init_logging(log_format, log_level);

            // Run application
            let mut app = App::new().await?;
            app.run().await
        }
    }
}
