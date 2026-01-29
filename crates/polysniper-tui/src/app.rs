//! TUI application state

use chrono::{DateTime, Utc};
use polysniper_core::{Market, Orderbook, Side};
use rust_decimal::Decimal;
use std::collections::VecDeque;

/// Maximum number of recent trades to display
const MAX_RECENT_TRADES: usize = 20;

/// Maximum number of orderbook levels to display
pub const MAX_ORDERBOOK_LEVELS: usize = 15;

/// A recorded trade for display
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub side: Side,
    pub price: Decimal,
    pub size: Decimal,
    pub timestamp: DateTime<Utc>,
}

/// Actions the app can take
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppAction {
    /// Quit the application
    Quit,
    /// Scroll orderbook up
    ScrollUp,
    /// Scroll orderbook down
    ScrollDown,
    /// Select next market
    NextMarket,
    /// Select previous market
    PrevMarket,
    /// Toggle help display
    ToggleHelp,
    /// No action
    None,
}

/// TUI application state
pub struct App {
    /// Whether the app should quit
    pub should_quit: bool,

    /// Currently selected market
    pub current_market: Option<Market>,

    /// Current orderbook for the selected market (YES token)
    pub orderbook: Option<Orderbook>,

    /// Recent trades executed by the bot
    pub recent_trades: VecDeque<TradeRecord>,

    /// Current scroll offset for orderbook
    pub orderbook_scroll: usize,

    /// Available markets to select from
    pub available_markets: Vec<Market>,

    /// Index of currently selected market in available_markets
    pub selected_market_index: usize,

    /// Whether to show help overlay
    pub show_help: bool,

    /// Last update timestamp
    pub last_update: DateTime<Utc>,

    /// Connection status
    pub connected: bool,
}

impl App {
    /// Create a new application instance
    pub fn new() -> Self {
        Self {
            should_quit: false,
            current_market: None,
            orderbook: None,
            recent_trades: VecDeque::new(),
            orderbook_scroll: 0,
            available_markets: Vec::new(),
            selected_market_index: 0,
            show_help: false,
            last_update: Utc::now(),
            connected: false,
        }
    }

    /// Handle an application action
    pub fn handle_action(&mut self, action: AppAction) {
        match action {
            AppAction::Quit => self.should_quit = true,
            AppAction::ScrollUp => {
                if self.orderbook_scroll > 0 {
                    self.orderbook_scroll -= 1;
                }
            }
            AppAction::ScrollDown => {
                if let Some(ref ob) = self.orderbook {
                    let max_scroll = ob.bids.len().max(ob.asks.len()).saturating_sub(MAX_ORDERBOOK_LEVELS);
                    if self.orderbook_scroll < max_scroll {
                        self.orderbook_scroll += 1;
                    }
                }
            }
            AppAction::NextMarket => {
                if !self.available_markets.is_empty() {
                    self.selected_market_index =
                        (self.selected_market_index + 1) % self.available_markets.len();
                    self.select_market_at_index();
                }
            }
            AppAction::PrevMarket => {
                if !self.available_markets.is_empty() {
                    self.selected_market_index = if self.selected_market_index == 0 {
                        self.available_markets.len() - 1
                    } else {
                        self.selected_market_index - 1
                    };
                    self.select_market_at_index();
                }
            }
            AppAction::ToggleHelp => {
                self.show_help = !self.show_help;
            }
            AppAction::None => {}
        }
    }

    /// Select the market at the current index
    fn select_market_at_index(&mut self) {
        if let Some(market) = self.available_markets.get(self.selected_market_index) {
            self.current_market = Some(market.clone());
            self.orderbook = None;
            self.orderbook_scroll = 0;
        }
    }

    /// Update the orderbook
    pub fn update_orderbook(&mut self, orderbook: Orderbook) {
        self.orderbook = Some(orderbook);
        self.last_update = Utc::now();
    }

    /// Update the current market
    pub fn update_market(&mut self, market: Market) {
        self.current_market = Some(market);
        self.last_update = Utc::now();
    }

    /// Add a trade to recent trades
    pub fn add_trade(&mut self, trade: TradeRecord) {
        self.recent_trades.push_front(trade);
        while self.recent_trades.len() > MAX_RECENT_TRADES {
            self.recent_trades.pop_back();
        }
    }

    /// Set available markets
    pub fn set_available_markets(&mut self, markets: Vec<Market>) {
        self.available_markets = markets;
        if self.current_market.is_none() && !self.available_markets.is_empty() {
            self.select_market_at_index();
        }
    }

    /// Set connection status
    pub fn set_connected(&mut self, connected: bool) {
        self.connected = connected;
    }

    /// Get the current mid price
    pub fn mid_price(&self) -> Option<Decimal> {
        self.orderbook.as_ref().and_then(|ob| ob.mid_price())
    }

    /// Get the current spread
    pub fn spread(&self) -> Option<Decimal> {
        self.orderbook.as_ref().and_then(|ob| ob.spread())
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<Decimal> {
        self.orderbook.as_ref().and_then(|ob| ob.best_bid())
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<Decimal> {
        self.orderbook.as_ref().and_then(|ob| ob.best_ask())
    }

    /// Get the last trade if any
    pub fn last_trade(&self) -> Option<&TradeRecord> {
        self.recent_trades.front()
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
