//! Recent trades widget

use crate::app::TradeRecord;
use polysniper_core::Side;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};
use std::collections::VecDeque;

/// Widget for displaying recent trades
pub struct TradesWidget<'a> {
    trades: &'a VecDeque<TradeRecord>,
}

impl<'a> TradesWidget<'a> {
    /// Create a new trades widget
    pub fn new(trades: &'a VecDeque<TradeRecord>) -> Self {
        Self { trades }
    }
}

impl Widget for TradesWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Recent Trades ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));

        let inner = block.inner(area);
        block.render(area, buf);

        if self.trades.is_empty() {
            let paragraph = Paragraph::new(Line::from(Span::styled(
                "No trades yet",
                Style::default().fg(Color::DarkGray),
            )));
            paragraph.render(inner, buf);
            return;
        }

        let mut lines = Vec::new();

        // Header
        lines.push(Line::from(vec![
            Span::styled(
                "SIDE   PRICE     SIZE      TIME",
                Style::default().fg(Color::DarkGray),
            ),
        ]));

        // Calculate how many trades can fit
        let max_trades = (inner.height as usize).saturating_sub(1);

        for trade in self.trades.iter().take(max_trades) {
            let (side_str, side_color) = match trade.side {
                Side::Buy => ("BUY ", Color::Green),
                Side::Sell => ("SELL", Color::Red),
            };

            let time_str = trade.timestamp.format("%H:%M:%S").to_string();

            lines.push(Line::from(vec![
                Span::styled(side_str, Style::default().fg(side_color)),
                Span::raw("   "),
                Span::styled(
                    format!("{:>8.4}", trade.price),
                    Style::default().fg(side_color),
                ),
                Span::raw("  "),
                Span::raw(format!("{:>8.2}", trade.size)),
                Span::raw("  "),
                Span::styled(time_str, Style::default().fg(Color::DarkGray)),
            ]));
        }

        let paragraph = Paragraph::new(lines);
        paragraph.render(inner, buf);
    }
}
