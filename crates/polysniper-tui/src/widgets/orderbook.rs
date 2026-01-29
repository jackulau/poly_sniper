//! Orderbook depth chart widget

use crate::app::MAX_ORDERBOOK_LEVELS;
use polysniper_core::Orderbook;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};
use rust_decimal::prelude::*;

/// Widget for displaying orderbook depth as a horizontal bar chart
pub struct OrderbookWidget<'a> {
    orderbook: Option<&'a Orderbook>,
    scroll_offset: usize,
}

impl<'a> OrderbookWidget<'a> {
    /// Create a new orderbook widget
    pub fn new(orderbook: Option<&'a Orderbook>, scroll_offset: usize) -> Self {
        Self {
            orderbook,
            scroll_offset,
        }
    }

    /// Render the orderbook as text lines
    fn render_lines(&self, width: u16) -> Vec<Line<'static>> {
        let Some(orderbook) = self.orderbook else {
            return vec![Line::from("No orderbook data")];
        };

        let mut lines = Vec::new();

        // Calculate the maximum size for scaling bars
        let max_bid_size = orderbook
            .bids
            .iter()
            .map(|l| l.size)
            .max()
            .unwrap_or(Decimal::ONE);
        let max_ask_size = orderbook
            .asks
            .iter()
            .map(|l| l.size)
            .max()
            .unwrap_or(Decimal::ONE);
        let max_size = max_bid_size.max(max_ask_size);

        // Header with spread info
        let spread = orderbook.spread().unwrap_or(Decimal::ZERO);
        let mid = orderbook.mid_price().unwrap_or(Decimal::ZERO);
        lines.push(Line::from(vec![
            Span::styled("     BIDS     ", Style::default().fg(Color::Green)),
            Span::raw("│"),
            Span::styled(
                format!(" SPREAD: {:.4} ", spread),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw("│"),
            Span::styled("     ASKS     ", Style::default().fg(Color::Red)),
        ]));

        lines.push(Line::from(vec![
            Span::raw("─────────────────"),
            Span::raw("┼"),
            Span::styled(
                format!("  MID: {:.4}  ", mid),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw("┼"),
            Span::raw("─────────────────"),
        ]));

        // Calculate available width for each side's bar
        // Layout: [bar][price] | spread | [price][bar]
        // Each side gets approximately (width - spread_col) / 2
        let side_width = ((width as usize).saturating_sub(20)) / 2;
        let bar_width = side_width.saturating_sub(8); // Leave room for price

        // Get visible levels based on scroll
        let visible_bids: Vec<_> = orderbook
            .bids
            .iter()
            .skip(self.scroll_offset)
            .take(MAX_ORDERBOOK_LEVELS)
            .collect();
        let visible_asks: Vec<_> = orderbook
            .asks
            .iter()
            .skip(self.scroll_offset)
            .take(MAX_ORDERBOOK_LEVELS)
            .collect();

        let num_levels = visible_bids.len().max(visible_asks.len());

        for i in 0..num_levels {
            let mut spans = Vec::new();

            // Bid side (right-aligned bar, then price)
            if let Some(bid) = visible_bids.get(i) {
                let bar_len = if max_size.is_zero() {
                    0
                } else {
                    ((bid.size / max_size) * Decimal::from(bar_width))
                        .to_usize()
                        .unwrap_or(0)
                        .min(bar_width)
                };

                // Create bar (right-aligned)
                let padding = bar_width.saturating_sub(bar_len);
                spans.push(Span::raw(" ".repeat(padding)));
                spans.push(Span::styled(
                    "█".repeat(bar_len),
                    Style::default().fg(Color::Green),
                ));
                spans.push(Span::styled(
                    format!(" {:.4}", bid.price),
                    Style::default().fg(Color::Green),
                ));
            } else {
                spans.push(Span::raw(" ".repeat(side_width)));
            }

            // Separator
            spans.push(Span::raw(" │ "));

            // Ask side (price, then bar)
            if let Some(ask) = visible_asks.get(i) {
                let bar_len = if max_size.is_zero() {
                    0
                } else {
                    ((ask.size / max_size) * Decimal::from(bar_width))
                        .to_usize()
                        .unwrap_or(0)
                        .min(bar_width)
                };

                spans.push(Span::styled(
                    format!("{:.4} ", ask.price),
                    Style::default().fg(Color::Red),
                ));
                spans.push(Span::styled(
                    "█".repeat(bar_len),
                    Style::default().fg(Color::Red),
                ));
            }

            lines.push(Line::from(spans));
        }

        // Show scroll indicator if there are more levels
        let total_levels = orderbook.bids.len().max(orderbook.asks.len());
        if total_levels > MAX_ORDERBOOK_LEVELS {
            lines.push(Line::from(Span::styled(
                format!(
                    "  [{}-{} of {} levels] ↑/↓ to scroll",
                    self.scroll_offset + 1,
                    (self.scroll_offset + num_levels).min(total_levels),
                    total_levels
                ),
                Style::default().fg(Color::DarkGray),
            )));
        }

        lines
    }
}

impl Widget for OrderbookWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Orderbook Depth ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));

        let inner = block.inner(area);
        block.render(area, buf);

        let lines = self.render_lines(inner.width);
        let paragraph = Paragraph::new(lines);
        paragraph.render(inner, buf);
    }
}
