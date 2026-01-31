//! Database models

use chrono::{DateTime, Utc};
use polysniper_core::{OrderType, Side};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Persisted trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub id: String,
    pub order_id: String,
    pub signal_id: String,
    pub strategy_id: String,
    pub market_id: String,
    pub token_id: String,
    pub side: Side,
    pub executed_price: Decimal,
    pub executed_size: Decimal,
    pub size_usd: Decimal,
    pub fees: Decimal,
    pub realized_pnl: Option<Decimal>,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

/// Persisted order record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRecord {
    pub id: String,
    pub signal_id: String,
    pub market_id: String,
    pub token_id: String,
    pub side: Side,
    pub price: Decimal,
    pub size: Decimal,
    pub order_type: OrderType,
    pub status: OrderStatusDb,
    pub filled_size: Decimal,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

/// Order status for database
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderStatusDb {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Expired,
    Failed,
}

impl std::fmt::Display for OrderStatusDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Submitted => write!(f, "submitted"),
            Self::PartiallyFilled => write!(f, "partially_filled"),
            Self::Filled => write!(f, "filled"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Expired => write!(f, "expired"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

impl std::str::FromStr for OrderStatusDb {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "submitted" => Ok(Self::Submitted),
            "partially_filled" => Ok(Self::PartiallyFilled),
            "filled" => Ok(Self::Filled),
            "cancelled" => Ok(Self::Cancelled),
            "expired" => Ok(Self::Expired),
            "failed" => Ok(Self::Failed),
            _ => Err(format!("Invalid order status: {}", s)),
        }
    }
}

/// Persisted price snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceSnapshotRecord {
    pub id: Option<i64>,
    pub market_id: String,
    pub token_id: String,
    pub price: Decimal,
    pub timestamp: DateTime<Utc>,
}

/// Strategy state for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyStateRecord {
    pub strategy_id: String,
    pub state_data: serde_json::Value,
    pub updated_at: DateTime<Utc>,
}

/// Daily P&L record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyPnlRecord {
    pub date: String,
    pub starting_balance: Decimal,
    pub ending_balance: Option<Decimal>,
    pub realized_pnl: Decimal,
    pub unrealized_pnl: Decimal,
    pub trade_count: i32,
    pub win_count: i32,
    pub loss_count: i32,
    pub circuit_breaker_hit: bool,
}

/// Alert record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRecord {
    pub id: Option<i64>,
    pub level: AlertLevel,
    pub category: String,
    pub message: String,
    pub metadata: Option<serde_json::Value>,
    pub sent: bool,
    pub created_at: DateTime<Utc>,
}

/// Alert severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
}

impl std::fmt::Display for AlertLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

impl std::str::FromStr for AlertLevel {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "info" => Ok(Self::Info),
            "warning" => Ok(Self::Warning),
            "critical" => Ok(Self::Critical),
            _ => Err(format!("Invalid alert level: {}", s)),
        }
    }
}

/// Configuration for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Path to SQLite database file
    pub db_path: String,
    /// Whether persistence is enabled
    pub enabled: bool,
    /// Price snapshot interval in seconds
    pub price_snapshot_interval_secs: u64,
    /// Maximum price snapshots to keep per token
    pub max_price_snapshots: i64,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            db_path: "data/polysniper.db".to_string(),
            enabled: true,
            price_snapshot_interval_secs: 60,
            max_price_snapshots: 10000,
        }
    }
}

/// Model learning stats record for database persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLearningStatsRecord {
    /// Model identifier
    pub model_id: String,
    /// Exponential moving average of accuracy
    pub ema_accuracy: Decimal,
    /// Adaptive confidence threshold
    pub adaptive_threshold: Decimal,
    /// Adaptive weight for ensemble
    pub adaptive_weight: Decimal,
    /// Thompson Sampling alpha parameter
    pub thompson_alpha: f64,
    /// Thompson Sampling beta parameter
    pub thompson_beta: f64,
    /// Total prediction count
    pub total_predictions: i64,
    /// Correct prediction count
    pub correct_predictions: i64,
    /// Total P&L from this model
    pub total_pnl: Decimal,
    /// Average confidence of predictions
    pub avg_confidence: Decimal,
    /// Recent predictions (JSON serialized)
    pub recent_predictions: Option<String>,
    /// When this model was first seen
    pub first_seen_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

/// Prediction outcome record for database persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionOutcomeRecord {
    /// Auto-generated ID
    pub id: Option<i64>,
    /// Unique prediction identifier
    pub prediction_id: String,
    /// Model that made the prediction
    pub model_id: String,
    /// Market ID (optional)
    pub market_id: Option<String>,
    /// Confidence of the prediction
    pub confidence: Decimal,
    /// Predicted outcome (YES/NO)
    pub predicted_outcome: String,
    /// Actual outcome (YES/NO, None if pending)
    pub actual_outcome: Option<String>,
    /// Whether prediction was correct (None if pending)
    pub is_correct: Option<bool>,
    /// Realized P&L from this prediction
    pub pnl: Option<Decimal>,
    /// When prediction was made
    pub predicted_at: DateTime<Utc>,
    /// When outcome was resolved (None if pending)
    pub resolved_at: Option<DateTime<Utc>>,
}
