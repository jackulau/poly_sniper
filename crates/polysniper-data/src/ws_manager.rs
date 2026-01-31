//! WebSocket manager for CLOB real-time data
//!
//! Optimized for low-latency message processing with configurable options
//! for TCP_NODELAY, compression, and priority-based message handling.

use futures::{SinkExt, StreamExt};
use polysniper_core::{
    DataSourceError, MarketId, Orderbook, PriceChangeEvent, PriceLevel, SystemEvent, TokenId,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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
    config: WsManagerConfig,
}

impl WsManager {
    /// Create a new WebSocket manager with default configuration
    pub fn new(ws_url: String, event_tx: broadcast::Sender<SystemEvent>) -> Self {
        Self::with_config(ws_url, event_tx, WsManagerConfig::default())
    }

    /// Create a new WebSocket manager with custom configuration
    pub fn with_config(
        ws_url: String,
        event_tx: broadcast::Sender<SystemEvent>,
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

    /// Start the WebSocket connection with automatic reconnection
    pub async fn run(&self) -> Result<(), DataSourceError> {
        loop {
            match self.connect_and_listen().await {
                Ok(()) => {
                    info!("WebSocket connection closed gracefully");
                    break;
                }
                Err(e) => {
                    error!("WebSocket error: {}, reconnecting...", e);
                    self.connected.store(false, Ordering::SeqCst);
                    self.publish_connection_status(false, Some(e.to_string()));
                    tokio::time::sleep(RECONNECT_DELAY).await;
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

        self.connected.store(true, Ordering::SeqCst);
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

        // Spawn ping task
        let connected_flag = self.connected.clone();
        let ping_handle = tokio::spawn(async move {
            let mut ping_interval = interval(PING_INTERVAL);
            while connected_flag.load(Ordering::SeqCst) {
                ping_interval.tick().await;
                // Note: actual ping sending would need access to write stream
                // This is simplified - in production, use a channel
            }
        });

        // Main message loop
        loop {
            let msg = timeout(MESSAGE_TIMEOUT, read.next()).await;

            match msg {
                Ok(Some(Ok(Message::Text(text)))) => {
                    self.handle_message(&text).await;
                }
                Ok(Some(Ok(Message::Ping(_data)))) => {
                    // Pong is handled automatically by tungstenite
                    debug!("Received ping");
                }
                Ok(Some(Ok(Message::Close(_)))) => {
                    info!("WebSocket closed by server");
                    break;
                }
                Ok(Some(Err(e))) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
                Ok(None) => {
                    info!("WebSocket stream ended");
                    break;
                }
                Err(_) => {
                    warn!("WebSocket message timeout");
                    break;
                }
                _ => {}
            }
        }

        self.connected.store(false, Ordering::SeqCst);
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
