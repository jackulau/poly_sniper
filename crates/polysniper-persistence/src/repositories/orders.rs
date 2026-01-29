//! Order repository

use crate::{error::Result, models::OrderRecord, models::OrderStatusDb, Database};
use chrono::{DateTime, Utc};
use polysniper_core::{Order, OrderType, Side};
use rust_decimal::Decimal;
use sqlx::Row;
use std::str::FromStr;

/// Repository for order records
pub struct OrderRepository<'a> {
    db: &'a Database,
}

impl<'a> OrderRepository<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Insert a new order
    pub async fn insert(&self, order: &Order) -> Result<()> {
        let now = Utc::now();
        sqlx::query(
            r#"
            INSERT INTO orders (
                id, signal_id, market_id, token_id, side, price, size,
                order_type, status, filled_size, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&order.id)
        .bind(&order.signal_id)
        .bind(&order.market_id)
        .bind(&order.token_id)
        .bind(order.side.to_string())
        .bind(order.price.to_string())
        .bind(order.size.to_string())
        .bind(format!("{:?}", order.order_type))
        .bind(OrderStatusDb::Pending.to_string())
        .bind("0")
        .bind(order.created_at.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(self.db.pool())
        .await?;

        Ok(())
    }

    /// Update order status
    pub async fn update_status(&self, order_id: &str, status: OrderStatusDb) -> Result<()> {
        sqlx::query("UPDATE orders SET status = ?, updated_at = ? WHERE id = ?")
            .bind(status.to_string())
            .bind(Utc::now().to_rfc3339())
            .bind(order_id)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    /// Update filled size
    pub async fn update_filled(&self, order_id: &str, filled_size: Decimal) -> Result<()> {
        sqlx::query("UPDATE orders SET filled_size = ?, updated_at = ? WHERE id = ?")
            .bind(filled_size.to_string())
            .bind(Utc::now().to_rfc3339())
            .bind(order_id)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    /// Get order by ID
    pub async fn get_by_id(&self, id: &str) -> Result<Option<OrderRecord>> {
        let row = sqlx::query("SELECT * FROM orders WHERE id = ?")
            .bind(id)
            .fetch_optional(self.db.pool())
            .await?;

        match row {
            Some(r) => Ok(Some(Self::row_to_order(&r)?)),
            None => Ok(None),
        }
    }

    /// Get orders by status
    pub async fn get_by_status(&self, status: OrderStatusDb) -> Result<Vec<OrderRecord>> {
        let rows = sqlx::query("SELECT * FROM orders WHERE status = ? ORDER BY created_at DESC")
            .bind(status.to_string())
            .fetch_all(self.db.pool())
            .await?;

        rows.iter().map(Self::row_to_order).collect()
    }

    /// Get orders by market
    pub async fn get_by_market(&self, market_id: &str, limit: i64) -> Result<Vec<OrderRecord>> {
        let rows = sqlx::query(
            "SELECT * FROM orders WHERE market_id = ? ORDER BY created_at DESC LIMIT ?",
        )
        .bind(market_id)
        .bind(limit)
        .fetch_all(self.db.pool())
        .await?;

        rows.iter().map(Self::row_to_order).collect()
    }

    /// Get recent orders
    pub async fn get_recent(&self, limit: i64) -> Result<Vec<OrderRecord>> {
        let rows = sqlx::query("SELECT * FROM orders ORDER BY created_at DESC LIMIT ?")
            .bind(limit)
            .fetch_all(self.db.pool())
            .await?;

        rows.iter().map(Self::row_to_order).collect()
    }

    /// Get pending orders count
    pub async fn pending_count(&self) -> Result<i64> {
        let row =
            sqlx::query("SELECT COUNT(*) as count FROM orders WHERE status IN ('pending', 'submitted')")
                .fetch_one(self.db.pool())
                .await?;

        Ok(row.get::<i64, _>("count"))
    }

    fn row_to_order(row: &sqlx::sqlite::SqliteRow) -> Result<OrderRecord> {
        Ok(OrderRecord {
            id: row.get("id"),
            signal_id: row.get("signal_id"),
            market_id: row.get("market_id"),
            token_id: row.get("token_id"),
            side: match row.get::<String, _>("side").as_str() {
                "BUY" => Side::Buy,
                _ => Side::Sell,
            },
            price: Decimal::from_str(row.get::<&str, _>("price")).unwrap_or(Decimal::ZERO),
            size: Decimal::from_str(row.get::<&str, _>("size")).unwrap_or(Decimal::ZERO),
            order_type: match row.get::<String, _>("order_type").as_str() {
                "Gtc" => OrderType::Gtc,
                "Fok" => OrderType::Fok,
                _ => OrderType::Gtd,
            },
            status: OrderStatusDb::from_str(row.get::<&str, _>("status"))
                .unwrap_or(OrderStatusDb::Pending),
            filled_size: Decimal::from_str(row.get::<&str, _>("filled_size"))
                .unwrap_or(Decimal::ZERO),
            created_at: DateTime::parse_from_rfc3339(row.get::<&str, _>("created_at"))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: DateTime::parse_from_rfc3339(row.get::<&str, _>("updated_at"))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            metadata: row
                .get::<Option<String>, _>("metadata")
                .and_then(|s| serde_json::from_str(&s).ok()),
        })
    }
}
