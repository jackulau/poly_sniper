//! Input event handling for the TUI

use crate::AppAction;
use crossterm::event::{self, Event as CrosstermEvent, KeyCode, KeyEvent, KeyModifiers};
use std::time::Duration;

/// Events that can occur in the TUI
#[derive(Debug, Clone)]
pub enum Event {
    /// A key was pressed
    Key(KeyEvent),
    /// Terminal was resized
    Resize(u16, u16),
    /// Tick event for refreshing UI
    Tick,
}

/// Handles input events from the terminal
pub struct EventHandler {
    /// Tick rate in milliseconds
    tick_rate: Duration,
}

impl EventHandler {
    /// Create a new event handler
    pub fn new(tick_rate_ms: u64) -> Self {
        Self {
            tick_rate: Duration::from_millis(tick_rate_ms),
        }
    }

    /// Poll for the next event
    pub fn next_event(&self) -> anyhow::Result<Event> {
        if event::poll(self.tick_rate)? {
            match event::read()? {
                CrosstermEvent::Key(key) => Ok(Event::Key(key)),
                CrosstermEvent::Resize(w, h) => Ok(Event::Resize(w, h)),
                _ => Ok(Event::Tick),
            }
        } else {
            Ok(Event::Tick)
        }
    }

    /// Convert a key event to an app action
    pub fn key_to_action(key: KeyEvent) -> AppAction {
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => AppAction::Quit,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => AppAction::Quit,
            KeyCode::Up | KeyCode::Char('k') => AppAction::ScrollUp,
            KeyCode::Down | KeyCode::Char('j') => AppAction::ScrollDown,
            KeyCode::Right | KeyCode::Char('l') | KeyCode::Tab => AppAction::NextMarket,
            KeyCode::Left | KeyCode::Char('h') | KeyCode::BackTab => AppAction::PrevMarket,
            KeyCode::Char('?') => AppAction::ToggleHelp,
            _ => AppAction::None,
        }
    }
}

impl Default for EventHandler {
    fn default() -> Self {
        Self::new(100) // 10 FPS
    }
}
