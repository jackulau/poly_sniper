//! Persistence error types

use thiserror::Error;

/// Result type for persistence operations
pub type Result<T> = std::result::Result<T, PersistenceError>;

/// Persistence layer errors
#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Migration error: {0}")]
    Migration(#[from] sqlx::migrate::MigrateError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Configuration error: {0}")]
    Config(String),
}
