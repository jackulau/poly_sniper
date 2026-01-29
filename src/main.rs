//! Polysniper - High-Performance Polymarket Sniper
//!
//! A Rust-based trading bot for Polymarket with multiple sniping strategies.

use anyhow::{Context, Result};
use polysniper_core::{
    AppConfig, EventBus, OrderExecutor, RiskDecision, RiskValidator, StateManager, StateProvider,
    Strategy, SystemEvent, TradeSignal,
};
use polysniper_data::{BroadcastEventBus, GammaClient, MarketCache, WsManager};
use polysniper_execution::{OrderBuilder, OrderSubmitter};
use polysniper_observability::{
    init_logging, record_event_processing, record_new_market, record_order, record_risk_rejection,
    record_signal, record_strategy_error, record_strategy_processing, start_metrics_server,
    update_markets_monitored, update_uptime, AlertManager, AlertingConfig, LogFormat, SlackConfig,
    TelegramConfig,
};
use polysniper_persistence::{Database, TradeRecord, TradeRepository};
use polysniper_risk::RiskManager;
use polysniper_strategies::{
    EventBasedConfig, EventBasedStrategy, NewMarketConfig, NewMarketStrategy, PriceSpikeConfig,
    PriceSpikeStrategy, TargetPriceConfig, TargetPriceStrategy,
};
use rust_decimal::prelude::ToPrimitive;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tokio::time::interval;
use tracing::{error, info, warn, Level};

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
    gamma_client: Arc<GammaClient>,
    database: Option<Arc<Database>>,
    alert_manager: Option<Arc<AlertManager>>,
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

        let order_executor: Arc<dyn OrderExecutor> = Arc::new(OrderSubmitter::new(
            config.endpoints.clob_rest.clone(),
            wallet_address,
            config.auth.signature_type,
            config.execution.clone(),
        ));

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

        Ok(Self {
            config,
            event_bus,
            state,
            strategies,
            risk_manager,
            order_builder,
            order_executor,
            gamma_client,
            database,
            alert_manager,
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

        if std::path::Path::new(&config_path).exists() {
            let content = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config file: {}", config_path))?;
            toml::from_str(&content)
                .with_context(|| format!("Failed to parse config file: {}", config_path))
        } else {
            info!("Config file not found, using defaults");
            Ok(AppConfig::default())
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

        // Load initial markets from Gamma
        self.load_initial_markets().await?;

        // Start metrics server (Phase 6)
        if self.config.metrics.enabled {
            let _metrics_handle = start_metrics_server(self.config.metrics.port).await;
            info!(port = %self.config.metrics.port, "Metrics server started");
        }

        // Create shutdown signal
        let (shutdown_tx, mut shutdown_rx) = broadcast::channel::<()>(1);

        // Spawn WebSocket manager
        let ws_manager = WsManager::new(
            self.config.endpoints.clob_ws.clone(),
            self.event_bus.sender(),
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

    /// Process a single event
    async fn process_event(&self, event: SystemEvent) -> Result<()> {
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

                            // Persist trade if database is available
                            if let Some(db) = &self.database {
                                let trade = TradeRecord {
                                    id: uuid::Uuid::new_v4().to_string(),
                                    order_id: order_id.clone(),
                                    signal_id: signal.id.clone(),
                                    strategy_id: signal.strategy_id.clone(),
                                    market_id: signal.market_id.clone(),
                                    token_id: signal.token_id.clone(),
                                    side: signal.side,
                                    executed_price: signal.price.unwrap_or_default(),
                                    executed_size: signal.size,
                                    size_usd: signal.size_usd,
                                    fees: rust_decimal::Decimal::ZERO,
                                    realized_pnl: None,
                                    timestamp: chrono::Utc::now(),
                                    metadata: Some(signal.metadata.clone()),
                                };
                                let trade_repo = TradeRepository::new(db);
                                if let Err(e) = trade_repo.insert(&trade).await {
                                    warn!(error = %e, "Failed to persist trade");
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
                        if let Some(alert_mgr) = &self.alert_manager {
                            let daily_pnl =
                                self.state.get_daily_pnl().await.to_f64().unwrap_or(0.0);
                            alert_mgr
                                .alert_circuit_breaker(&e.to_string(), daily_pnl)
                                .await;
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
    // Initialize logging
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
