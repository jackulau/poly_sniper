//! Order submitter with retry logic

use polysniper_core::{
    ExecutionConfig, ExecutionError, Order, OrderExecutor, OrderStatus, OrderStatusResponse,
};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

/// CLOB order request
#[derive(Debug, Clone, Serialize)]
struct ClobOrderRequest {
    order: ClobOrder,
    owner: String,
    #[serde(rename = "orderType")]
    order_type: String,
}

#[derive(Debug, Clone, Serialize)]
struct ClobOrder {
    salt: String,
    maker: String,
    signer: String,
    taker: String,
    #[serde(rename = "tokenId")]
    token_id: String,
    #[serde(rename = "makerAmount")]
    maker_amount: String,
    #[serde(rename = "takerAmount")]
    taker_amount: String,
    expiration: String,
    nonce: String,
    #[serde(rename = "feeRateBps")]
    fee_rate_bps: String,
    side: String,
    #[serde(rename = "signatureType")]
    signature_type: u8,
    signature: String,
}

/// CLOB order response
#[derive(Debug, Clone, Deserialize)]
struct ClobOrderResponse {
    #[serde(rename = "orderID")]
    order_id: Option<String>,
    success: bool,
    #[serde(rename = "errorMsg")]
    error_msg: Option<String>,
}

/// Order cancel response
#[derive(Debug, Clone, Deserialize)]
struct ClobCancelResponse {
    success: bool,
    #[serde(rename = "errorMsg")]
    error_msg: Option<String>,
}

/// Order status response
#[derive(Debug, Clone, Deserialize)]
struct ClobOrderStatusResponse {
    #[serde(rename = "orderID")]
    order_id: String,
    status: String,
    #[serde(rename = "filledSize")]
    filled_size: String,
    #[serde(rename = "remainingSize")]
    remaining_size: String,
    #[serde(rename = "avgFillPrice")]
    avg_fill_price: Option<String>,
}

/// Order submitter with retry logic
pub struct OrderSubmitter {
    base_url: String,
    client: Client,
    config: ExecutionConfig,
    dry_run: Arc<AtomicBool>,
    wallet_address: String,
    signature_type: u8,
}

impl OrderSubmitter {
    /// Create a new order submitter
    pub fn new(
        base_url: String,
        wallet_address: String,
        signature_type: u8,
        config: ExecutionConfig,
    ) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");

        let dry_run = Arc::new(AtomicBool::new(config.dry_run));

        Self {
            base_url,
            client,
            config,
            dry_run,
            wallet_address,
            signature_type,
        }
    }

    /// Set dry run mode
    pub fn set_dry_run(&self, dry_run: bool) {
        self.dry_run.store(dry_run, AtomicOrdering::SeqCst);
    }

    /// Submit order with retry logic
    async fn submit_with_retry(&self, order: &Order) -> Result<String, ExecutionError> {
        let mut attempts = 0;

        loop {
            attempts += 1;

            match self.submit_once(order).await {
                Ok(order_id) => {
                    info!(
                        order_id = %order_id,
                        attempts = attempts,
                        "Order submitted successfully"
                    );
                    return Ok(order_id);
                }
                Err(e) => {
                    if attempts >= self.config.max_retries {
                        error!(
                            order_id = %order.id,
                            attempts = attempts,
                            error = %e,
                            "Order submission failed after max retries"
                        );
                        return Err(ExecutionError::RetryExhausted {
                            attempts,
                            message: e.to_string(),
                        });
                    }

                    warn!(
                        order_id = %order.id,
                        attempt = attempts,
                        error = %e,
                        "Order submission failed, retrying..."
                    );

                    sleep(Duration::from_millis(self.config.retry_delay_ms)).await;
                }
            }
        }
    }

    /// Submit a single order attempt
    async fn submit_once(&self, order: &Order) -> Result<String, ExecutionError> {
        // In production, this would:
        // 1. Build the EIP-712 typed data structure
        // 2. Sign with the wallet's private key
        // 3. Submit to CLOB API

        // For now, simulate the submission
        let url = format!("{}/order", self.base_url);

        debug!(
            order_id = %order.id,
            market_id = %order.market_id,
            token_id = %order.token_id,
            side = %order.side,
            price = %order.price,
            size = %order.size,
            "Submitting order"
        );

        // Build placeholder order request
        // In production, this would be properly signed
        let request = ClobOrderRequest {
            order: ClobOrder {
                salt: chrono::Utc::now().timestamp_millis().to_string(),
                maker: self.wallet_address.clone(),
                signer: self.wallet_address.clone(),
                taker: "0x0000000000000000000000000000000000000000".to_string(),
                token_id: order.token_id.clone(),
                maker_amount: (order.size * Decimal::new(1000000, 0)).to_string(),
                taker_amount: (order.size * order.price * Decimal::new(1000000, 0)).to_string(),
                expiration: "0".to_string(),
                nonce: "0".to_string(),
                fee_rate_bps: "0".to_string(),
                side: order.side.to_string(),
                signature_type: self.signature_type,
                signature: "0x".to_string(), // Would be actual signature
            },
            owner: self.wallet_address.clone(),
            order_type: format!("{:?}", order.order_type).to_uppercase(),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ExecutionError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ExecutionError::SubmissionError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        let clob_response: ClobOrderResponse = response
            .json()
            .await
            .map_err(|e| ExecutionError::SubmissionError(e.to_string()))?;

        if !clob_response.success {
            return Err(ExecutionError::SubmissionError(
                clob_response
                    .error_msg
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ));
        }

        Ok(clob_response.order_id.unwrap_or_else(|| order.id.clone()))
    }
}

#[async_trait::async_trait]
impl OrderExecutor for OrderSubmitter {
    async fn submit_order(&self, order: Order) -> Result<String, ExecutionError> {
        if self.dry_run.load(AtomicOrdering::SeqCst) {
            info!(
                order_id = %order.id,
                market_id = %order.market_id,
                token_id = %order.token_id,
                side = %order.side,
                price = %order.price,
                size = %order.size,
                "[DRY RUN] Would submit order"
            );
            return Ok(format!("dry_run_{}", order.id));
        }

        self.submit_with_retry(&order).await
    }

    async fn cancel_order(&self, order_id: &str) -> Result<(), ExecutionError> {
        if self.dry_run.load(AtomicOrdering::SeqCst) {
            info!(order_id = %order_id, "[DRY RUN] Would cancel order");
            return Ok(());
        }

        let url = format!("{}/order/{}", self.base_url, order_id);

        let response = self
            .client
            .delete(&url)
            .send()
            .await
            .map_err(|e| ExecutionError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ExecutionError::Cancelled(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        let cancel_response: ClobCancelResponse = response
            .json()
            .await
            .map_err(|e| ExecutionError::SubmissionError(e.to_string()))?;

        if !cancel_response.success {
            return Err(ExecutionError::Cancelled(
                cancel_response
                    .error_msg
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ));
        }

        info!(order_id = %order_id, "Order cancelled");
        Ok(())
    }

    async fn get_order_status(
        &self,
        order_id: &str,
    ) -> Result<OrderStatusResponse, ExecutionError> {
        let url = format!("{}/order/{}", self.base_url, order_id);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ExecutionError::NetworkError(e.to_string()))?;

        if response.status().as_u16() == 404 {
            return Err(ExecutionError::NotFound(order_id.to_string()));
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ExecutionError::SubmissionError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        let status_response: ClobOrderStatusResponse = response
            .json()
            .await
            .map_err(|e| ExecutionError::SubmissionError(e.to_string()))?;

        let status = match status_response.status.as_str() {
            "LIVE" => OrderStatus::Live,
            "MATCHED" => OrderStatus::Matched,
            "CANCELLED" => OrderStatus::Cancelled,
            "EXPIRED" => OrderStatus::Expired,
            _ => OrderStatus::Live,
        };

        let filled_size = status_response
            .filled_size
            .parse::<Decimal>()
            .unwrap_or(Decimal::ZERO);
        let remaining_size = status_response
            .remaining_size
            .parse::<Decimal>()
            .unwrap_or(Decimal::ZERO);
        let avg_fill_price = status_response
            .avg_fill_price
            .and_then(|s| s.parse::<Decimal>().ok());

        Ok(OrderStatusResponse {
            order_id: status_response.order_id,
            status,
            filled_size,
            remaining_size,
            avg_fill_price,
        })
    }

    fn is_dry_run(&self) -> bool {
        self.dry_run.load(AtomicOrdering::SeqCst)
    }
}
