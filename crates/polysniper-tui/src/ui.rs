//! Main TUI layout and rendering

use crate::{
    app::App,
    widgets::{OrderbookWidget, PriceTickerWidget, TradesWidget},
};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

/// Draw the TUI to the terminal
pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Main layout: header, content, footer
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(10),    // Content
            Constraint::Length(1),  // Footer
        ])
        .split(area);

    // Draw header
    draw_header(frame, app, main_chunks[0]);

    // Content layout: orderbook on top, price/trades on bottom
    let content_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(60), // Orderbook
            Constraint::Percentage(40), // Price ticker + trades
        ])
        .split(main_chunks[1]);

    // Draw orderbook widget
    let orderbook_widget =
        OrderbookWidget::new(app.orderbook.as_ref(), app.orderbook_scroll);
    frame.render_widget(orderbook_widget, content_chunks[0]);

    // Bottom section: price ticker + trades side by side
    let bottom_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // Price ticker
            Constraint::Percentage(60), // Recent trades
        ])
        .split(content_chunks[1]);

    // Draw price ticker
    let price_widget = PriceTickerWidget::from_app(app);
    frame.render_widget(price_widget, bottom_chunks[0]);

    // Draw recent trades
    let trades_widget = TradesWidget::new(&app.recent_trades);
    frame.render_widget(trades_widget, bottom_chunks[1]);

    // Draw footer
    draw_footer(frame, main_chunks[2]);

    // Draw help overlay if enabled
    if app.show_help {
        draw_help_overlay(frame, area);
    }
}

/// Draw the header with market info
fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let market_info = if let Some(ref market) = app.current_market {
        let question = if market.question.len() > 60 {
            format!("{}...", &market.question[..57])
        } else {
            market.question.clone()
        };
        format!(" Market: {} ", question)
    } else {
        " No market selected ".to_string()
    };

    let market_selector = if !app.available_markets.is_empty() {
        format!(
            " [{}/{}] ←/→ to switch ",
            app.selected_market_index + 1,
            app.available_markets.len()
        )
    } else {
        String::new()
    };

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            market_info,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(market_selector, Style::default().fg(Color::DarkGray)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" Polysniper TUI "),
    );

    frame.render_widget(header, area);
}

/// Draw the footer with keybindings
fn draw_footer(frame: &mut Frame, area: Rect) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" q", Style::default().fg(Color::Yellow)),
        Span::raw(": quit  "),
        Span::styled("↑↓/jk", Style::default().fg(Color::Yellow)),
        Span::raw(": scroll  "),
        Span::styled("←→/hl", Style::default().fg(Color::Yellow)),
        Span::raw(": switch market  "),
        Span::styled("?", Style::default().fg(Color::Yellow)),
        Span::raw(": help"),
    ]));

    frame.render_widget(footer, area);
}

/// Draw the help overlay
fn draw_help_overlay(frame: &mut Frame, area: Rect) {
    // Center the help box
    let width = 50.min(area.width.saturating_sub(4));
    let height = 15.min(area.height.saturating_sub(4));
    let x = (area.width.saturating_sub(width)) / 2;
    let y = (area.height.saturating_sub(height)) / 2;

    let help_area = Rect::new(x, y, width, height);

    // Clear the background
    frame.render_widget(Clear, help_area);

    let help_text = vec![
        Line::from(Span::styled(
            "Keyboard Shortcuts",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  q / Esc     ", Style::default().fg(Color::Yellow)),
            Span::raw("Quit"),
        ]),
        Line::from(vec![
            Span::styled("  Ctrl+C      ", Style::default().fg(Color::Yellow)),
            Span::raw("Force quit"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ↑ / k       ", Style::default().fg(Color::Yellow)),
            Span::raw("Scroll orderbook up"),
        ]),
        Line::from(vec![
            Span::styled("  ↓ / j       ", Style::default().fg(Color::Yellow)),
            Span::raw("Scroll orderbook down"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ← / h       ", Style::default().fg(Color::Yellow)),
            Span::raw("Previous market"),
        ]),
        Line::from(vec![
            Span::styled("  → / l / Tab ", Style::default().fg(Color::Yellow)),
            Span::raw("Next market"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ?           ", Style::default().fg(Color::Yellow)),
            Span::raw("Toggle this help"),
        ]),
    ];

    let help = Paragraph::new(help_text).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .style(Style::default().bg(Color::Black))
            .title(" Help "),
    );

    frame.render_widget(help, help_area);
}
