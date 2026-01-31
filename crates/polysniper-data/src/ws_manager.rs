//! WebSocket manager for CLOB real-time data with connection warmup
//! WebSocket manager for CLOB real-time data
//!
//! Optimized for low-latency message processing with configurable options
//! for TCP_NODELAY, compression, and priority-based message handling.

use futures::{SinkExt, StreamExt};
use polysniper_core::{
    DataSourceError, MarketId, Orderbook, PriceChangeEvent, PriceLevel, SystemEvent, TokenId,
};
use rand::Rng;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::time::timeout;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

use crate::connection_health::{ConnectionHealth, HealthStatus};

/// Connection warmup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    /// Ping interval for keep-alive (more aggressive than passive 30s)
    #[serde(with = "humantime_serde", default = "default_ping_interval")]
    pub ping_interval: Duration,
    /// Pong timeout before considering connection dead
    #[serde(with = "humantime_serde", default = "default_pong_timeout")]
    pub pong_timeout: Duration,
    /// Initial reconnect delay
    #[serde(with = "humantime_serde", default = "default_initial_reconnect_delay")]
    pub initial_reconnect_delay: Duration,
    /// Maximum reconnect delay
    #[serde(with = "humantime_serde", default = "default_max_reconnect_delay")]
    pub max_reconnect_delay: Duration,
    /// Backoff multiplier for exponential backoff
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: f64,
    /// Jitter factor to prevent thundering herd (0.0 to 1.0)
    #[serde(default = "default_jitter_factor")]
    pub jitter_factor: f64,
    /// Message timeout (considers connection dead if no message received)
    #[serde(with = "humantime_serde", default = "default_message_timeout")]
    pub message_timeout: Duration,
}

fn default_ping_interval() -> Duration {
    Duration::from_secs(15)
}
fn default_pong_timeout() -> Duration {
    Duration::from_secs(5)
}
fn default_initial_reconnect_delay() -> Duration {
    Duration::from_millis(100)
}
fn default_max_reconnect_delay() -> Duration {
    Duration::from_secs(30)
}
fn default_backoff_multiplier() -> f64 {
    2.0
}
fn default_jitter_factor() -> f64 {
    0.1
}
fn default_message_timeout() -> Duration {
    Duration::from_secs(60)
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            ping_interval: default_ping_interval(),
            pong_timeout: default_pong_timeout(),
            initial_reconnect_delay: default_initial_reconnect_delay(),
            max_reconnect_delay: default_max_reconnect_delay(),
            backoff_multiplier: default_backoff_multiplier(),
            jitter_factor: default_jitter_factor(),
            message_timeout: default_message_timeout(),
        }
    }
}
use std::time::Duration;
use tokio::sync::{broadcast, RwLock};
use tokio::time::{interval, timeout};
use tokio_tungstenite::{connect_async_with_config, tungstenite::protocol::WebSocketConfig, tungstenite::Message};
use tracing::{debug, error, info, warn};

const RECONNECT_DELAY: Duration = Duration::from_secs(1);
const PING_INTERVAL: Duration = Duration::from_secs(30);
const MESSAGE_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_BUFFER_SIZE: usize = 65536;

/// WebSocket message types from CLOB
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsMessage {
    /// Subscription confirmation
    Subscribed {
        channel: String,
        assets: Vec<String>,
    },
    /// Price change notification
    PriceChange {
        asset_id: String,
        price: String,
        timestamp: String,
    },
    /// Orderbook snapshot
    Book {
        asset_id: String,
        market: String,
        bids: Vec<BookLevel>,
        asks: Vec<BookLevel>,
        timestamp: String,
    },
    /// Orderbook delta update
    BookDelta {
        asset_id: String,
        market: String,
        bids: Vec<BookLevel>,
        asks: Vec<BookLevel>,
    },
    /// Heartbeat
    Heartbeat,
    /// Error message
    Error { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookLevel {
    pub price: String,
    pub size: String,
}

/// Subscription request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeRequest {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub channel: String,
    pub assets: Vec<String>,
}

impl SubscribeRequest {
    pub fn book(token_ids: Vec<String>) -> Self {
        Self {
            msg_type: "subscribe".to_string(),
            channel: "book".to_string(),
            assets: token_ids,
        }
    }

    pub fn price(token_ids: Vec<String>) -> Self {
        Self {
            msg_type: "subscribe".to_string(),
            channel: "price".to_string(),
            assets: token_ids,
        }
    }
}

/// Internal commands for the ping loop
enum PingCommand {
    RecordPong { rtt: Duration },
    Shutdown,
}

/// WebSocket manager for CLOB data with connection warmup
/// Configuration for WebSocket latency optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsManagerConfig {
    /// Enable WebSocket compression (disabled by default as it adds CPU overhead)
    pub enable_compression: bool,
    /// Enable TCP_NODELAY to disable Nagle's algorithm (reduces latency)
    pub tcp_nodelay: bool,
    /// Receive buffer size in bytes
    pub buffer_size: usize,
    /// High-priority token subscriptions (processed first)
    pub priority_subscriptions: Vec<String>,
}

impl Default for WsManagerConfig {
    fn default() -> Self {
        Self {
            enable_compression: false,
            tcp_nodelay: true,
            buffer_size: DEFAULT_BUFFER_SIZE,
            priority_subscriptions: Vec::new(),
        }
    }
}

/// WebSocket manager for CLOB data
pub struct WsManager {
    ws_url: String,
    event_tx: broadcast::Sender<SystemEvent>,
    connected: Arc<AtomicBool>,
    subscribed_tokens: Arc<RwLock<HashSet<TokenId>>>,
    market_token_map: Arc<RwLock<std::collections::HashMap<TokenId, MarketId>>>,
    config: ConnectionConfig,
    reconnect_attempts: AtomicU32,
    health: Arc<ConnectionHealth>,
    config: WsManagerConfig,
}

impl WsManager {
    /// Create a new WebSocket manager with default configuration
    pub fn new(ws_url: String, event_tx: broadcast::Sender<SystemEvent>) -> Self {
        Self::with_config(ws_url, event_tx, ConnectionConfig::default())
        Self::with_config(ws_url, event_tx, WsManagerConfig::default())
    }

    /// Create a new WebSocket manager with custom configuration
    pub fn with_config(
        ws_url: String,
        event_tx: broadcast::Sender<SystemEvent>,
        config: ConnectionConfig,
    ) -> Self {
        config: WsManagerConfig,
    ) -> Self {
        info!(
            tcp_nodelay = config.tcp_nodelay,
            compression = config.enable_compression,
            buffer_size = config.buffer_size,
            "Creating WebSocket manager with latency optimizations"
        );
        Self {
            ws_url,
            event_tx,
            connected: Arc::new(AtomicBool::new(false)),
            subscribed_tokens: Arc::new(RwLock::new(HashSet::new())),
            market_token_map: Arc::new(RwLock::new(std::collections::HashMap::new())),
            config,
            reconnect_attempts: AtomicU32::new(0),
            health: Arc::new(ConnectionHealth::new()),
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &WsManagerConfig {
        &self.config
    }

    /// Check if a token is a priority subscription
    pub fn is_priority_token(&self, token_id: &str) -> bool {
        self.config.priority_subscriptions.contains(&token_id.to_string())
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    /// Get connection health status
    pub fn health_status(&self) -> HealthStatus {
        self.health.status()
    }

    /// Get connection health metrics
    pub fn health(&self) -> &ConnectionHealth {
        &self.health
    }

    /// Subscribe to a token's book and price updates
    pub async fn subscribe_token(&self, token_id: TokenId, market_id: MarketId) {
        self.subscribed_tokens
            .write()
            .await
            .insert(token_id.clone());
        self.market_token_map
            .write()
            .await
            .insert(token_id, market_id);
    }

    /// Unsubscribe from a token
    pub async fn unsubscribe_token(&self, token_id: &TokenId) {
        self.subscribed_tokens.write().await.remove(token_id);
        self.market_token_map.write().await.remove(token_id);
    }

    /// Calculate reconnect delay with exponential backoff and jitter
    fn calculate_reconnect_delay(&self, attempt: u32) -> Duration {
        let base_delay = self.config.initial_reconnect_delay.as_secs_f64()
            * self.config.backoff_multiplier.powi(attempt as i32);
        let capped_delay = base_delay.min(self.config.max_reconnect_delay.as_secs_f64());

        // Add jitter to prevent thundering herd
        let mut rng = rand::thread_rng();
        let jitter = capped_delay * self.config.jitter_factor * rng.gen::<f64>();
        let final_delay = capped_delay + jitter;

        debug!(
            attempt = attempt,
            base_delay_ms = (base_delay * 1000.0) as u64,
            final_delay_ms = (final_delay * 1000.0) as u64,
            "Calculated reconnect delay"
        );

        Duration::from_secs_f64(final_delay)
    }

    /// Start the WebSocket connection with automatic reconnection
    pub async fn run(&self) -> Result<(), DataSourceError> {
        loop {
            match self.connect_and_listen().await {
                Ok(()) => {
                    info!("WebSocket connection closed gracefully");
                    break;
                }
                Err(e) => {
                    let attempt = self.reconnect_attempts.fetch_add(1, Ordering::SeqCst);
                    let delay = self.calculate_reconnect_delay(attempt);

                    error!(
                        error = %e,
                        attempt = attempt,
                        delay_ms = delay.as_millis() as u64,
                        "WebSocket error, reconnecting with backoff"
                    );

                    self.connected.store(false, Ordering::SeqCst);
                    self.health.record_failure();
                    self.publish_connection_status(false, Some(e.to_string()));

                    tokio::time::sleep(delay).await;
                }
            }
        }
        Ok(())
    }

    async fn connect_and_listen(&self) -> Result<(), DataSourceError> {
        info!("Connecting to WebSocket: {}", self.ws_url);

        // Configure WebSocket for low latency
        let mut ws_config = WebSocketConfig::default();
        ws_config.max_message_size = Some(16 * 1024 * 1024); // 16 MB
        ws_config.max_frame_size = Some(4 * 1024 * 1024);    // 4 MB
        ws_config.accept_unmasked_frames = false;

        let (ws_stream, _) = connect_async_with_config(&self.ws_url, Some(ws_config), self.config.tcp_nodelay)
            .await
            .map_err(|e| DataSourceError::ConnectionError(e.to_string()))?;

        let (mut write, mut read) = ws_stream.split();

        // Reset reconnect attempts on successful connection
        self.reconnect_attempts.store(0, Ordering::SeqCst);
        self.connected.store(true, Ordering::SeqCst);
        self.health.reset();
        self.publish_connection_status(true, None);
        info!("WebSocket connected");

        // Subscribe to all tracked tokens
        let tokens: Vec<String> = self
            .subscribed_tokens
            .read()
            .await
            .iter()
            .cloned()
            .collect();
        if !tokens.is_empty() {
            let book_sub = SubscribeRequest::book(tokens.clone());
            let price_sub = SubscribeRequest::price(tokens);

            let book_msg = serde_json::to_string(&book_sub)
                .map_err(|e| DataSourceError::ParseError(e.to_string()))?;
            let price_msg = serde_json::to_string(&price_sub)
                .map_err(|e| DataSourceError::ParseError(e.to_string()))?;

            write
                .send(Message::Text(book_msg.into()))
                .await
                .map_err(|e| DataSourceError::WebSocketError(e.to_string()))?;
            write
                .send(Message::Text(price_msg.into()))
                .await
                .map_err(|e| DataSourceError::WebSocketError(e.to_string()))?;

            debug!("Sent subscription requests");
        }

        // Create channel for ping commands
        let (ping_cmd_tx, mut ping_cmd_rx) = mpsc::channel::<PingCommand>(16);

        // Spawn active ping loop
        let connected_flag = self.connected.clone();
        let ping_interval = self.config.ping_interval;
        let pong_timeout = self.config.pong_timeout;
        let health = self.health.clone();

        let ping_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(ping_interval);
            let mut pending_ping: Option<Instant> = None;
            let mut last_pong = Instant::now();

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Check if we're still connected
                        if !connected_flag.load(Ordering::SeqCst) {
                            break;
                        }

                        // Check if pending ping timed out
                        if let Some(ping_time) = pending_ping {
                            if ping_time.elapsed() > pong_timeout {
                                warn!(
                                    timeout_ms = pong_timeout.as_millis() as u64,
                                    "Pong timeout, connection may be dead"
                                );
                                health.record_failure();
                                connected_flag.store(false, Ordering::SeqCst);
                                break;
                            }
                        }

                        // Check last pong time
                        if last_pong.elapsed() > pong_timeout * 3 {
                            warn!("Extended pong timeout, marking connection unhealthy");
                            health.record_failure();
                        }

                        // Record that we're about to send a ping
                        pending_ping = Some(Instant::now());
                    }
                    cmd = ping_cmd_rx.recv() => {
                        match cmd {
                            Some(PingCommand::RecordPong { rtt }) => {
                                pending_ping = None;
                                last_pong = Instant::now();
                                health.record_pong(rtt);
                                debug!(rtt_ms = rtt.as_millis() as u64, "Recorded pong");
                            }
                            Some(PingCommand::Shutdown) | None => {
                                break;
                            }
                        }
                    }
                }
            }
        });

        // Track when we sent pings for RTT calculation
        let mut last_ping_time: Option<Instant> = None;

        // Main message loop with active ping sending
        let mut ping_interval = tokio::time::interval(self.config.ping_interval);

        loop {
            tokio::select! {
                // Periodic ping sending
                _ = ping_interval.tick() => {
                    if let Err(e) = write.send(Message::Ping(vec![].into())).await {
                        warn!(error = %e, "Failed to send ping");
                        break;
                    }
                    last_ping_time = Some(Instant::now());
                    debug!("Sent ping");
                }

                // Message receiving
                msg = timeout(self.config.message_timeout, read.next()) => {
                    match msg {
                        Ok(Some(Ok(Message::Text(text)))) => {
                            self.health.record_message();
                            self.handle_message(&text).await;
                        }
                        Ok(Some(Ok(Message::Ping(data)))) => {
                            // Respond to server pings
                            if let Err(e) = write.send(Message::Pong(data)).await {
                                warn!(error = %e, "Failed to send pong response");
                            }
                            debug!("Received ping, sent pong");
                        }
                        Ok(Some(Ok(Message::Pong(_)))) => {
                            // Calculate RTT if we have a pending ping
                            if let Some(ping_time) = last_ping_time.take() {
                                let rtt = ping_time.elapsed();
                                let _ = ping_cmd_tx.send(PingCommand::RecordPong { rtt }).await;
                            }
                            debug!("Received pong");
                        }
                        Ok(Some(Ok(Message::Close(_)))) => {
                            info!("WebSocket closed by server");
                            break;
                        }
                        Ok(Some(Err(e))) => {
                            error!(error = %e, "WebSocket error");
                            break;
                        }
                        Ok(None) => {
                            info!("WebSocket stream ended");
                            break;
                        }
                        Err(_) => {
                            warn!(
                                timeout_secs = self.config.message_timeout.as_secs(),
                                "WebSocket message timeout"
                            );
                            self.health.record_failure();
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Cleanup
        self.connected.store(false, Ordering::SeqCst);
        let _ = ping_cmd_tx.send(PingCommand::Shutdown).await;
        ping_handle.abort();

        Ok(())
    }

    async fn handle_message(&self, text: &str) {
        match serde_json::from_str::<WsMessage>(text) {
            Ok(msg) => match msg {
                WsMessage::PriceChange {
                    asset_id,
                    price,
                    timestamp: _,
                } => {
                    if let Ok(new_price) = price.parse::<Decimal>() {
                        if let Some(market_id) =
                            self.market_token_map.read().await.get(&asset_id).cloned()
                        {
                            let event = SystemEvent::PriceChange(PriceChangeEvent::new(
                                market_id, asset_id,
                                None, // Old price would need state tracking
                                new_price,
                            ));
                            let _ = self.event_tx.send(event);
                        }
                    }
                }
                WsMessage::Book {
                    asset_id,
                    market,
                    bids,
                    asks,
                    timestamp: _,
                } => {
                    let orderbook = self.parse_orderbook(&asset_id, &market, bids, asks);
                    let event =
                        SystemEvent::OrderbookUpdate(polysniper_core::OrderbookUpdateEvent {
                            market_id: market,
                            token_id: asset_id,
                            orderbook,
                            timestamp: chrono::Utc::now(),
                        });
                    let _ = self.event_tx.send(event);
                }
                WsMessage::BookDelta {
                    asset_id,
                    market,
                    bids,
                    asks,
                } => {
                    // Delta updates would need to be applied to existing orderbook
                    // For simplicity, treating as full update
                    let orderbook = self.parse_orderbook(&asset_id, &market, bids, asks);
                    let event =
                        SystemEvent::OrderbookUpdate(polysniper_core::OrderbookUpdateEvent {
                            market_id: market,
                            token_id: asset_id,
                            orderbook,
                            timestamp: chrono::Utc::now(),
                        });
                    let _ = self.event_tx.send(event);
                }
                WsMessage::Subscribed { channel, assets } => {
                    info!("Subscribed to {} for {} assets", channel, assets.len());
                }
                WsMessage::Heartbeat => {
                    debug!("Received heartbeat");
                    self.health.record_message();
                }
                WsMessage::Error { message } => {
                    error!("WebSocket error message: {}", message);
                }
            },
            Err(e) => {
                debug!("Failed to parse WebSocket message: {} - {}", e, text);
            }
        }
    }

    fn parse_orderbook(
        &self,
        token_id: &str,
        market_id: &str,
        bids: Vec<BookLevel>,
        asks: Vec<BookLevel>,
    ) -> Orderbook {
        let parse_levels = |levels: Vec<BookLevel>| -> Vec<PriceLevel> {
            levels
                .into_iter()
                .filter_map(|l| {
                    let price = l.price.parse::<Decimal>().ok()?;
                    let size = l.size.parse::<Decimal>().ok()?;
                    Some(PriceLevel { price, size })
                })
                .collect()
        };

        Orderbook {
            token_id: token_id.to_string(),
            market_id: market_id.to_string(),
            bids: parse_levels(bids),
            asks: parse_levels(asks),
            timestamp: chrono::Utc::now(),
        }
    }

    fn publish_connection_status(&self, connected: bool, message: Option<String>) {
        let event = SystemEvent::ConnectionStatus(polysniper_core::ConnectionStatusEvent {
            source: "clob_ws".to_string(),
            status: if connected {
                polysniper_core::ConnectionState::Connected
            } else {
                polysniper_core::ConnectionState::Disconnected
            },
            message,
            timestamp: chrono::Utc::now(),
        });
        let _ = self.event_tx.send(event);
    }
}

/// Module for humantime serde deserialization
mod humantime_serde {
    use serde::{self, Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let ms = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(ms))
    }
}
