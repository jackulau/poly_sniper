use thiserror::Error;

/// Strategy errors
#[derive(Error, Debug)]
pub enum StrategyError {
    #[error("Strategy initialization failed: {0}")]
    InitializationError(String),

    #[error("Strategy configuration error: {0}")]
    ConfigError(String),

    #[error("Strategy processing error: {0}")]
    ProcessingError(String),

    #[error("State access error: {0}")]
    StateError(String),

    #[error("Strategy is disabled")]
    Disabled,
}

/// Data source errors
#[derive(Error, Debug)]
pub enum DataSourceError {
    #[error("Connection failed: {0}")]
    ConnectionError(String),

    #[error("Authentication failed: {0}")]
    AuthError(String),

    #[error("WebSocket error: {0}")]
    WebSocketError(String),

    #[error("HTTP request failed: {0}")]
    HttpError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Disconnected")]
    Disconnected,
}

/// Risk management errors
#[derive(Error, Debug)]
pub enum RiskError {
    #[error("Position limit exceeded: {0}")]
    PositionLimitExceeded(String),

    #[error("Order size limit exceeded: {0}")]
    OrderSizeLimitExceeded(String),

    #[error("Daily loss limit exceeded: {0}")]
    DailyLossLimitExceeded(String),

    #[error("Circuit breaker triggered: {0}")]
    CircuitBreakerTriggered(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Trading halted: {0}")]
    TradingHalted(String),

    #[error("Validation error: {0}")]
    ValidationError(String),
}

/// Execution errors
#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("Order submission failed: {0}")]
    SubmissionError(String),

    #[error("Order signing failed: {0}")]
    SigningError(String),

    #[error("Order cancelled: {0}")]
    Cancelled(String),

    #[error("Order expired")]
    Expired,

    #[error("Insufficient balance: {0}")]
    InsufficientBalance(String),

    #[error("Invalid order: {0}")]
    InvalidOrder(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Retry exhausted after {attempts} attempts: {message}")]
    RetryExhausted { attempts: u32, message: String },

    #[error("Order not found: {0}")]
    NotFound(String),
}

/// Persistence errors
#[derive(Error, Debug)]
pub enum PersistenceError {
    #[error("Database connection error: {0}")]
    ConnectionError(String),

    #[error("Query error: {0}")]
    QueryError(String),

    #[error("Migration error: {0}")]
    MigrationError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Not found: {0}")]
    NotFound(String),
}

/// Configuration errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Configuration file not found: {0}")]
    FileNotFound(String),

    #[error("Configuration parse error: {0}")]
    ParseError(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid value for {field}: {message}")]
    InvalidValue { field: String, message: String },

    #[error("Environment variable not set: {0}")]
    EnvVarNotSet(String),
}
