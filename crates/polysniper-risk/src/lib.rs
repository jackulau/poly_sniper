//! Polysniper Risk Management
//!
//! Risk validation and position management.

pub mod control;
pub mod correlation;
pub mod correlation_regime;
pub mod drawdown;
pub mod kelly;
pub mod time_rules;
pub mod validator;
pub mod volatility;

pub use control::*;
pub use correlation::*;
pub use correlation_regime::{CorrelationRegimeDetector, RegimeSnapshot};
pub use drawdown::{DrawdownCalculator, DrawdownStatus};
pub use kelly::{KellyCalculator, KellyResult, TradeOutcome};
pub use time_rules::{TimeRuleEngine, TimeRuleResult};
pub use validator::RiskManager;
pub use volatility::VolatilityCalculator;
