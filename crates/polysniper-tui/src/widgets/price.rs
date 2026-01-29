//! Price ticker widget

use crate::app::{App, TradeRecord};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};
use rust_decimal::Decimal;

/// Widget for displaying current price information
pub struct PriceTickerWidget<'a> {
    mid_price: Option<Decimal>,
    spread: Option<Decimal>,
    best_bid: Option<Decimal>,
    best_ask: Option<Decimal>,
    last_trade: Option<&'a TradeRecord>,
    connected: bool,
}

impl<'a> PriceTickerWidget<'a> {
    /// Create a new price ticker widget from app state
    pub fn from_app(app: &'a App) -> Self {
        Self {
            mid_price: app.mid_price(),
            spread: app.spread(),
            best_bid: app.best_bid(),
            best_ask: app.best_ask(),
            last_trade: app.last_trade(),
            connected: app.connected,
        }
    }
}

impl Widget for PriceTickerWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Price Ticker ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));

        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines = Vec::new();

        // Connection status
        let status_style = if self.connected {
            Style::default().fg(Color::Green)
        } else {
            Style::default().fg(Color::Red)
        };
        let status_text = if self.connected {
            "● CONNECTED"
        } else {
            "○ DISCONNECTED"
        };
        lines.push(Line::from(Span::styled(status_text, status_style)));
        lines.push(Line::from(""));

        // Mid price
        if let Some(mid) = self.mid_price {
            lines.push(Line::from(vec![
                Span::raw("Mid Price: "),
                Span::styled(
                    format!("{:.4}", mid),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
        } else {
            lines.push(Line::from(Span::styled(
                "Mid Price: --",
                Style::default().fg(Color::DarkGray),
            )));
        }

        // Spread
        if let Some(spread) = self.spread {
            let spread_pct = self
                .mid_price
                .map(|m| {
                    if m.is_zero() {
                        Decimal::ZERO
                    } else {
                        (spread / m) * Decimal::from(100)
                    }
                })
                .unwrap_or(Decimal::ZERO);

            lines.push(Line::from(vec![
                Span::raw("Spread:    "),
                Span::styled(
                    format!("{:.4} ({:.2}%)", spread, spread_pct),
                    Style::default().fg(Color::Cyan),
                ),
            ]));
        } else {
            lines.push(Line::from(Span::styled(
                "Spread:    --",
                Style::default().fg(Color::DarkGray),
            )));
        }

        lines.push(Line::from(""));

        // Best bid/ask
        if let Some(bid) = self.best_bid {
            lines.push(Line::from(vec![
                Span::raw("Best Bid:  "),
                Span::styled(format!("{:.4}", bid), Style::default().fg(Color::Green)),
            ]));
        }

        if let Some(ask) = self.best_ask {
            lines.push(Line::from(vec![
                Span::raw("Best Ask:  "),
                Span::styled(format!("{:.4}", ask), Style::default().fg(Color::Red)),
            ]));
        }

        lines.push(Line::from(""));

        // Last trade
        if let Some(trade) = self.last_trade {
            let (side_str, side_color) = match trade.side {
                polysniper_core::Side::Buy => ("BUY ", Color::Green),
                polysniper_core::Side::Sell => ("SELL", Color::Red),
            };

            lines.push(Line::from(Span::raw("Last Trade:")));
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(side_str, Style::default().fg(side_color)),
                Span::raw(format!(" {:.4} x {:.2}", trade.price, trade.size)),
            ]));
        } else {
            lines.push(Line::from(Span::styled(
                "Last Trade: --",
                Style::default().fg(Color::DarkGray),
            )));
        }

        let paragraph = Paragraph::new(lines);
        paragraph.render(inner, buf);
    }
}
