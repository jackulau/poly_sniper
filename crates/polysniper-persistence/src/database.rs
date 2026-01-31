//! Database connection and schema management

use crate::error::Result;
use sqlx::{sqlite::SqlitePoolOptions, Pool, Sqlite};
use std::path::Path;
use tracing::info;

/// Database connection pool wrapper
#[derive(Clone)]
pub struct Database {
    pool: Pool<Sqlite>,
}

impl Database {
    /// Create a new database connection
    pub async fn new(db_path: &str) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = Path::new(db_path).parent() {
            std::fs::create_dir_all(parent).ok();
        }

        let connection_string = format!("sqlite:{}?mode=rwc", db_path);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&connection_string)
            .await?;

        let db = Self { pool };
        db.initialize_schema().await?;

        info!(db_path = %db_path, "Database initialized");
        Ok(db)
    }

    /// Create an in-memory database (for testing)
    pub async fn in_memory() -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await?;

        let db = Self { pool };
        db.initialize_schema().await?;

        info!("In-memory database initialized");
        Ok(db)
    }

    /// Get the connection pool
    pub fn pool(&self) -> &Pool<Sqlite> {
        &self.pool
    }

    /// Initialize database schema
    async fn initialize_schema(&self) -> Result<()> {
        // Trades table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                signal_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                side TEXT NOT NULL,
                executed_price TEXT NOT NULL,
                executed_size TEXT NOT NULL,
                size_usd TEXT NOT NULL,
                fees TEXT NOT NULL,
                realized_pnl TEXT,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Orders table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                signal_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                side TEXT NOT NULL,
                price TEXT NOT NULL,
                size TEXT NOT NULL,
                order_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                filled_size TEXT DEFAULT '0',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Price snapshots table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS price_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                price TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Strategy state table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS strategy_state (
                strategy_id TEXT PRIMARY KEY,
                state_data TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Daily PnL table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT PRIMARY KEY,
                starting_balance TEXT NOT NULL,
                ending_balance TEXT,
                realized_pnl TEXT NOT NULL DEFAULT '0',
                unrealized_pnl TEXT NOT NULL DEFAULT '0',
                trade_count INTEGER NOT NULL DEFAULT 0,
                win_count INTEGER NOT NULL DEFAULT 0,
                loss_count INTEGER NOT NULL DEFAULT 0,
                circuit_breaker_hit INTEGER NOT NULL DEFAULT 0
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Alerts table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                category TEXT NOT NULL,
                message TEXT NOT NULL,
                metadata TEXT,
                sent INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Position history table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS position_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price TEXT NOT NULL,
                exit_price TEXT,
                size TEXT NOT NULL,
                realized_pnl TEXT,
                fees TEXT NOT NULL DEFAULT '0',
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                strategy_id TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Model learning stats table for online learning
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS model_learning_stats (
                model_id TEXT PRIMARY KEY,
                ema_accuracy TEXT NOT NULL DEFAULT '0.5',
                adaptive_threshold TEXT NOT NULL DEFAULT '0.6',
                adaptive_weight TEXT NOT NULL DEFAULT '1.0',
                thompson_alpha REAL NOT NULL DEFAULT 1.0,
                thompson_beta REAL NOT NULL DEFAULT 1.0,
                total_predictions INTEGER NOT NULL DEFAULT 0,
                correct_predictions INTEGER NOT NULL DEFAULT 0,
                total_pnl TEXT NOT NULL DEFAULT '0',
                avg_confidence TEXT NOT NULL DEFAULT '0',
                recent_predictions TEXT,
                first_seen_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Prediction outcomes table for tracking individual predictions
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS prediction_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT NOT NULL UNIQUE,
                model_id TEXT NOT NULL,
                market_id TEXT,
                confidence TEXT NOT NULL,
                predicted_outcome TEXT NOT NULL,
                actual_outcome TEXT,
                is_correct INTEGER,
                pnl TEXT,
                predicted_at TEXT NOT NULL,
                resolved_at TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create indexes for common queries
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id);
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id);
            CREATE INDEX IF NOT EXISTS idx_orders_market ON orders(market_id);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            CREATE INDEX IF NOT EXISTS idx_price_snapshots_token ON price_snapshots(token_id);
            CREATE INDEX IF NOT EXISTS idx_price_snapshots_timestamp ON price_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(level);
            CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
            CREATE INDEX IF NOT EXISTS idx_position_history_market ON position_history(market_id);
            CREATE INDEX IF NOT EXISTS idx_position_history_token ON position_history(token_id);
            CREATE INDEX IF NOT EXISTS idx_position_history_strategy ON position_history(strategy_id);
            CREATE INDEX IF NOT EXISTS idx_position_history_opened ON position_history(opened_at);
            CREATE INDEX IF NOT EXISTS idx_position_history_closed ON position_history(closed_at);
            CREATE INDEX IF NOT EXISTS idx_model_learning_stats_updated ON model_learning_stats(updated_at);
            CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_model ON prediction_outcomes(model_id);
            CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_market ON prediction_outcomes(market_id);
            CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_predicted_at ON prediction_outcomes(predicted_at);
            "#,
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Close the database connection
    pub async fn close(&self) {
        self.pool.close().await;
    }
}
