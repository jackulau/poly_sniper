//! Config file watcher service
//!
//! Monitors configuration files for changes and emits events when they are modified.

use crate::events::{ConfigChangedEvent, ConfigType, SystemEvent};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, error, info, warn};

/// Default debounce duration for file change events
const DEFAULT_DEBOUNCE_DURATION: Duration = Duration::from_millis(500);

/// Config file watcher that monitors config files and emits events when they change
pub struct ConfigWatcher {
    /// Path to main config file
    config_path: PathBuf,
    /// Path to strategies directory
    strategies_dir: PathBuf,
    /// Event sender for broadcasting config changes
    event_tx: broadcast::Sender<SystemEvent>,
    /// Debounce duration to avoid rapid re-fires
    debounce_duration: Duration,
}

impl ConfigWatcher {
    /// Create a new config watcher
    ///
    /// # Arguments
    /// * `config_path` - Path to the main config file (e.g., config/default.toml)
    /// * `strategies_dir` - Path to the strategies directory (e.g., config/strategies)
    /// * `event_tx` - Broadcast channel sender for emitting events
    pub fn new(
        config_path: impl Into<PathBuf>,
        strategies_dir: impl Into<PathBuf>,
        event_tx: broadcast::Sender<SystemEvent>,
    ) -> Self {
        Self {
            config_path: config_path.into(),
            strategies_dir: strategies_dir.into(),
            event_tx,
            debounce_duration: DEFAULT_DEBOUNCE_DURATION,
        }
    }

    /// Set custom debounce duration
    pub fn with_debounce(mut self, duration: Duration) -> Self {
        self.debounce_duration = duration;
        self
    }

    /// Run the config watcher
    ///
    /// This method watches the config files and emits events when they change.
    /// It runs until the watcher is dropped or an unrecoverable error occurs.
    pub async fn run(&self) -> Result<(), ConfigWatcherError> {
        let (tx, mut rx) = mpsc::channel::<notify::Result<Event>>(100);

        // Create the watcher
        let mut watcher = RecommendedWatcher::new(
            move |res| {
                let _ = tx.blocking_send(res);
            },
            Config::default(),
        )
        .map_err(|e| ConfigWatcherError::WatcherInit(e.to_string()))?;

        // Watch main config file
        if self.config_path.exists() {
            watcher
                .watch(&self.config_path, RecursiveMode::NonRecursive)
                .map_err(|e| ConfigWatcherError::WatchPath {
                    path: self.config_path.clone(),
                    error: e.to_string(),
                })?;
            info!("Watching main config: {:?}", self.config_path);
        } else {
            warn!(
                "Main config file does not exist, will not watch: {:?}",
                self.config_path
            );
        }

        // Watch strategies directory
        if self.strategies_dir.exists() {
            watcher
                .watch(&self.strategies_dir, RecursiveMode::Recursive)
                .map_err(|e| ConfigWatcherError::WatchPath {
                    path: self.strategies_dir.clone(),
                    error: e.to_string(),
                })?;
            info!("Watching strategies directory: {:?}", self.strategies_dir);
        } else {
            warn!(
                "Strategies directory does not exist, will not watch: {:?}",
                self.strategies_dir
            );
        }

        // Track last event time per path for debouncing
        let last_events: Arc<RwLock<HashMap<PathBuf, Instant>>> =
            Arc::new(RwLock::new(HashMap::new()));

        // Process events
        while let Some(res) = rx.recv().await {
            match res {
                Ok(event) => {
                    self.handle_event(event, &last_events).await;
                }
                Err(e) => {
                    error!("Watch error: {:?}", e);
                }
            }
        }

        Ok(())
    }

    /// Handle a file system event
    async fn handle_event(
        &self,
        event: Event,
        last_events: &Arc<RwLock<HashMap<PathBuf, Instant>>>,
    ) {
        // Only process modify events
        if !matches!(
            event.kind,
            EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
        ) {
            return;
        }

        for path in event.paths {
            // Only process .toml files
            if path.extension().is_none_or(|ext| ext != "toml") {
                continue;
            }

            // Check debounce
            let now = Instant::now();
            {
                let last = last_events.read().await;
                if let Some(&last_time) = last.get(&path) {
                    if now.duration_since(last_time) < self.debounce_duration {
                        debug!("Debouncing event for {:?}", path);
                        continue;
                    }
                }
            }

            // Update last event time
            {
                let mut last = last_events.write().await;
                last.insert(path.clone(), now);
            }

            // Determine config type
            let config_type = self.determine_config_type(&path);

            debug!("Config changed: {:?} ({:?})", path, config_type);

            // Emit event
            let event = ConfigChangedEvent::new(path, config_type);
            if let Err(e) = self.event_tx.send(SystemEvent::ConfigChanged(event)) {
                warn!("Failed to send config changed event: {:?}", e);
            }
        }
    }

    /// Determine the type of config based on the path
    fn determine_config_type(&self, path: &Path) -> ConfigType {
        // Check if it's the main config
        if path == self.config_path {
            return ConfigType::Main;
        }

        // Check if it's a strategy config
        if path.starts_with(&self.strategies_dir) {
            let strategy_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            return ConfigType::Strategy(strategy_name);
        }

        // Default to main if we can't determine
        ConfigType::Main
    }
}

/// Errors that can occur in the config watcher
#[derive(Debug, thiserror::Error)]
pub enum ConfigWatcherError {
    #[error("Failed to initialize watcher: {0}")]
    WatcherInit(String),

    #[error("Failed to watch path {path:?}: {error}")]
    WatchPath { path: PathBuf, error: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_config_type_detection() {
        let (tx, _rx) = broadcast::channel(16);
        let watcher = ConfigWatcher::new("config/default.toml", "config/strategies", tx);

        // Test main config detection
        let config_type = watcher.determine_config_type(Path::new("config/default.toml"));
        assert_eq!(config_type, ConfigType::Main);

        // Test strategy config detection
        let config_type =
            watcher.determine_config_type(Path::new("config/strategies/price_spike.toml"));
        assert_eq!(config_type, ConfigType::Strategy("price_spike".to_string()));
    }

    #[tokio::test]
    async fn test_file_modification_detection() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("default.toml");
        let strategies_dir = temp_dir.path().join("strategies");

        // Create initial files
        fs::write(&config_path, "initial = true").unwrap();
        fs::create_dir(&strategies_dir).unwrap();

        let (tx, mut rx) = broadcast::channel(16);
        let watcher = ConfigWatcher::new(&config_path, &strategies_dir, tx)
            .with_debounce(Duration::from_millis(50));

        // Spawn watcher in background
        let watcher_handle = tokio::spawn(async move {
            let _ = watcher.run().await;
        });

        // Give watcher time to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Modify the config file
        fs::write(&config_path, "modified = true").unwrap();

        // Wait for event with timeout
        let result = timeout(Duration::from_secs(2), rx.recv()).await;

        assert!(result.is_ok(), "Should receive event within timeout");
        let event = result.unwrap().unwrap();

        match event {
            SystemEvent::ConfigChanged(e) => {
                assert_eq!(e.config_type, ConfigType::Main);
                assert_eq!(e.path, config_path);
            }
            _ => panic!("Expected ConfigChanged event"),
        }

        watcher_handle.abort();
    }

    #[tokio::test]
    async fn test_debounce_behavior() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("default.toml");
        let strategies_dir = temp_dir.path().join("strategies");

        fs::write(&config_path, "initial = true").unwrap();
        fs::create_dir(&strategies_dir).unwrap();

        let (tx, mut rx) = broadcast::channel(16);
        let watcher = ConfigWatcher::new(&config_path, &strategies_dir, tx)
            .with_debounce(Duration::from_millis(200));

        let watcher_handle = tokio::spawn(async move {
            let _ = watcher.run().await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Rapid modifications (should be debounced)
        for i in 0..5 {
            fs::write(&config_path, format!("rapid = {}", i)).unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Wait for potential events
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Count received events
        let mut event_count = 0;
        while let Ok(result) = rx.try_recv() {
            if matches!(result, SystemEvent::ConfigChanged(_)) {
                event_count += 1;
            }
        }

        // Should have at most 2 events due to debouncing (possibly 1-2 depending on timing)
        assert!(
            event_count <= 2,
            "Expected at most 2 events due to debouncing, got {}",
            event_count
        );

        watcher_handle.abort();
    }

    #[tokio::test]
    async fn test_strategy_file_change() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("default.toml");
        let strategies_dir = temp_dir.path().join("strategies");

        fs::write(&config_path, "main = true").unwrap();
        fs::create_dir(&strategies_dir).unwrap();
        let strategy_path = strategies_dir.join("momentum.toml");
        fs::write(&strategy_path, "strategy = true").unwrap();

        let (tx, mut rx) = broadcast::channel(16);
        let watcher = ConfigWatcher::new(&config_path, &strategies_dir, tx)
            .with_debounce(Duration::from_millis(50));

        let watcher_handle = tokio::spawn(async move {
            let _ = watcher.run().await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Modify strategy file
        fs::write(&strategy_path, "strategy = modified").unwrap();

        let result = timeout(Duration::from_secs(2), rx.recv()).await;

        assert!(result.is_ok(), "Should receive event within timeout");
        let event = result.unwrap().unwrap();

        match event {
            SystemEvent::ConfigChanged(e) => {
                assert_eq!(e.config_type, ConfigType::Strategy("momentum".to_string()));
            }
            _ => panic!("Expected ConfigChanged event"),
        }

        watcher_handle.abort();
    }

    #[tokio::test]
    async fn test_missing_files_graceful_handling() {
        let (tx, _rx) = broadcast::channel(16);
        let watcher = ConfigWatcher::new(
            "/nonexistent/config.toml",
            "/nonexistent/strategies",
            tx,
        );

        // Run with timeout - watcher will start but watch nothing
        // Since there are no valid paths, it may block forever waiting for events
        // so we use a timeout to verify it starts without panicking
        let watcher_handle = tokio::spawn(async move {
            watcher.run().await
        });

        // Give it time to initialize
        tokio::time::sleep(Duration::from_millis(100)).await;

        // If we got here without panicking, the graceful handling worked
        // Abort the watcher since it would run forever
        watcher_handle.abort();
    }
}
