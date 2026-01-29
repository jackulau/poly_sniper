//! Time-based risk rules for trading restrictions near market events
//!
//! This module provides time-aware risk rules that can reduce or halt trading
//! activity as market events (like resolution dates) approach.

use chrono::{DateTime, Utc};
use polysniper_core::{Market, Side, TimeRuleAction, TimeRulesConfig, TradeSignal};
use rust_decimal::Decimal;
use tracing::{debug, info};

/// Result of checking time rules against a market
#[derive(Debug, Clone)]
pub enum TimeRuleResult {
    /// No rules apply, trading allowed normally
    Allowed,
    /// Size should be reduced by a multiplier
    ReduceSize {
        multiplier: Decimal,
        rule_name: String,
    },
    /// New positions blocked (exits still allowed)
    BlockNew { rule_name: String },
    /// All trading halted
    HaltAll { rule_name: String },
}

impl TimeRuleResult {
    /// Returns true if trading is completely blocked
    pub fn is_blocked(&self) -> bool {
        matches!(self, TimeRuleResult::HaltAll { .. })
    }

    /// Returns true if new positions are blocked
    pub fn is_new_blocked(&self) -> bool {
        matches!(
            self,
            TimeRuleResult::BlockNew { .. } | TimeRuleResult::HaltAll { .. }
        )
    }

    /// Get the size multiplier (1.0 if no reduction)
    pub fn size_multiplier(&self) -> Decimal {
        match self {
            TimeRuleResult::ReduceSize { multiplier, .. } => *multiplier,
            _ => Decimal::ONE,
        }
    }

    /// Get the rule name if a rule matched
    pub fn rule_name(&self) -> Option<&str> {
        match self {
            TimeRuleResult::Allowed => None,
            TimeRuleResult::ReduceSize { rule_name, .. } => Some(rule_name),
            TimeRuleResult::BlockNew { rule_name } => Some(rule_name),
            TimeRuleResult::HaltAll { rule_name } => Some(rule_name),
        }
    }
}

/// Engine for evaluating time-based risk rules
#[derive(Debug, Clone)]
pub struct TimeRuleEngine {
    config: TimeRulesConfig,
}

impl TimeRuleEngine {
    /// Create a new TimeRuleEngine with the given configuration
    pub fn new(config: TimeRulesConfig) -> Self {
        Self { config }
    }

    /// Check if time rules are enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Check time rules for a market and return the most restrictive result
    ///
    /// Rules are evaluated against the market's end_date. The most restrictive
    /// matching rule is applied (HaltAll > BlockNew > ReduceSize > Allowed).
    pub fn check_market(&self, market: &Market) -> TimeRuleResult {
        self.check_market_at(market, Utc::now())
    }

    /// Check time rules for a market at a specific time (useful for testing)
    pub fn check_market_at(&self, market: &Market, now: DateTime<Utc>) -> TimeRuleResult {
        if !self.config.enabled {
            return TimeRuleResult::Allowed;
        }

        let end_date = match market.end_date {
            Some(date) => date,
            None => {
                debug!(market_id = %market.condition_id, "Market has no end_date, skipping time rules");
                return TimeRuleResult::Allowed;
            }
        };

        // Calculate hours until market ends
        let duration = end_date.signed_duration_since(now);
        let hours_until = duration.num_hours();

        // If market already ended, halt all trading
        if hours_until < 0 {
            return TimeRuleResult::HaltAll {
                rule_name: "market_ended".to_string(),
            };
        }

        let hours_until = hours_until as u64;

        // Find matching rules and apply the most restrictive
        let mut result = TimeRuleResult::Allowed;
        let mut smallest_multiplier = Decimal::ONE;

        for rule in &self.config.rules {
            // Check if rule applies to this time window
            if hours_until > rule.hours_before {
                continue;
            }

            // Check if market matches any of the patterns
            if !self.matches_patterns(market, &rule.applies_to) {
                continue;
            }

            debug!(
                rule = %rule.name,
                hours_until = hours_until,
                market = %market.condition_id,
                "Time rule matched"
            );

            // Apply most restrictive rule
            match &rule.action {
                TimeRuleAction::HaltAll => {
                    // HaltAll is most restrictive, return immediately
                    info!(
                        rule = %rule.name,
                        market = %market.condition_id,
                        hours_until = hours_until,
                        "Time rule: halting all trading"
                    );
                    return TimeRuleResult::HaltAll {
                        rule_name: rule.name.clone(),
                    };
                }
                TimeRuleAction::BlockNew => {
                    // BlockNew is more restrictive than ReduceSize
                    if !matches!(result, TimeRuleResult::BlockNew { .. }) {
                        info!(
                            rule = %rule.name,
                            market = %market.condition_id,
                            hours_until = hours_until,
                            "Time rule: blocking new positions"
                        );
                        result = TimeRuleResult::BlockNew {
                            rule_name: rule.name.clone(),
                        };
                    }
                }
                TimeRuleAction::ReduceSize { multiplier } => {
                    // Use smallest multiplier across all matching rules
                    if *multiplier < smallest_multiplier
                        && !matches!(
                            result,
                            TimeRuleResult::BlockNew { .. } | TimeRuleResult::HaltAll { .. }
                        )
                    {
                        smallest_multiplier = *multiplier;
                        info!(
                            rule = %rule.name,
                            market = %market.condition_id,
                            hours_until = hours_until,
                            multiplier = %multiplier,
                            "Time rule: reducing position size"
                        );
                        result = TimeRuleResult::ReduceSize {
                            multiplier: *multiplier,
                            rule_name: rule.name.clone(),
                        };
                    }
                }
            }
        }

        result
    }

    /// Check if trading is blocked for a market
    pub fn is_blocked(&self, market: &Market) -> bool {
        self.check_market(market).is_blocked()
    }

    /// Check if new positions are blocked for a market
    pub fn is_new_blocked(&self, market: &Market) -> bool {
        self.check_market(market).is_new_blocked()
    }

    /// Get the size modifier for a market (1.0 if no reduction applies)
    pub fn get_size_modifier(&self, market: &Market) -> Decimal {
        self.check_market(market).size_multiplier()
    }

    /// Check if a signal should be allowed based on time rules
    ///
    /// Exits (sells) are allowed even when BlockNew is active, but blocked under HaltAll.
    pub fn check_signal(&self, signal: &TradeSignal, market: &Market) -> TimeRuleResult {
        let result = self.check_market(market);

        // If blocking new positions, allow exits (sells)
        if matches!(result, TimeRuleResult::BlockNew { .. }) && signal.side == Side::Sell {
            debug!(
                signal_id = %signal.id,
                market = %market.condition_id,
                "Allowing exit despite BlockNew"
            );
            return TimeRuleResult::Allowed;
        }

        result
    }

    /// Match market against glob patterns
    fn matches_patterns(&self, market: &Market, patterns: &[String]) -> bool {
        for pattern in patterns {
            if self.matches_glob(pattern, market) {
                return true;
            }
        }
        false
    }

    /// Simple glob matching against market identifiers
    ///
    /// Supports:
    /// - `*` matches everything
    /// - `foo*` matches anything starting with "foo"
    /// - `*foo` matches anything ending with "foo"
    /// - `*foo*` matches anything containing "foo"
    /// - Exact match otherwise
    fn matches_glob(&self, pattern: &str, market: &Market) -> bool {
        // Check against question and tags
        let targets = [
            market.question.to_lowercase(),
            market.condition_id.to_lowercase(),
        ];

        let pattern_lower = pattern.to_lowercase();

        // Wildcard matches everything
        if pattern == "*" {
            return true;
        }

        // Handle glob patterns
        let (prefix_wild, suffix_wild) = (
            pattern_lower.starts_with('*'),
            pattern_lower.ends_with('*'),
        );

        let core_pattern = pattern_lower
            .trim_start_matches('*')
            .trim_end_matches('*');

        for target in &targets {
            let matched = match (prefix_wild, suffix_wild) {
                (true, true) => target.contains(core_pattern),
                (true, false) => target.ends_with(core_pattern),
                (false, true) => target.starts_with(core_pattern),
                (false, false) => target == core_pattern,
            };

            if matched {
                return true;
            }
        }

        // Also check tags
        for tag in &market.tags {
            let tag_lower = tag.to_lowercase();
            let matched = match (prefix_wild, suffix_wild) {
                (true, true) => tag_lower.contains(core_pattern),
                (true, false) => tag_lower.ends_with(core_pattern),
                (false, true) => tag_lower.starts_with(core_pattern),
                (false, false) => tag_lower == core_pattern,
            };

            if matched {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use polysniper_core::TimeRule;
    use rust_decimal_macros::dec;

    fn create_test_market(
        condition_id: &str,
        question: &str,
        tags: Vec<&str>,
        hours_until_end: i64,
    ) -> Market {
        let now = Utc::now();
        Market {
            condition_id: condition_id.to_string(),
            question: question.to_string(),
            description: None,
            tags: tags.into_iter().map(|s| s.to_string()).collect(),
            yes_token_id: "yes_token".to_string(),
            no_token_id: "no_token".to_string(),
            created_at: now - Duration::days(30),
            end_date: Some(now + Duration::hours(hours_until_end)),
            active: true,
            closed: false,
            volume: dec!(100000),
            liquidity: dec!(50000),
        }
    }

    fn create_test_market_no_end(condition_id: &str, question: &str) -> Market {
        Market {
            condition_id: condition_id.to_string(),
            question: question.to_string(),
            description: None,
            tags: vec![],
            yes_token_id: "yes_token".to_string(),
            no_token_id: "no_token".to_string(),
            created_at: Utc::now(),
            end_date: None,
            active: true,
            closed: false,
            volume: dec!(100000),
            liquidity: dec!(50000),
        }
    }

    fn default_config() -> TimeRulesConfig {
        TimeRulesConfig {
            enabled: true,
            rules: vec![
                TimeRule {
                    name: "pre_resolution_reduction".to_string(),
                    hours_before: 24,
                    action: TimeRuleAction::ReduceSize {
                        multiplier: dec!(0.5),
                    },
                    applies_to: vec!["*".to_string()],
                },
                TimeRule {
                    name: "resolution_block".to_string(),
                    hours_before: 2,
                    action: TimeRuleAction::BlockNew,
                    applies_to: vec!["*".to_string()],
                },
            ],
        }
    }

    #[test]
    fn test_no_rules_when_disabled() {
        let config = TimeRulesConfig {
            enabled: false,
            rules: vec![TimeRule {
                name: "test".to_string(),
                hours_before: 100,
                action: TimeRuleAction::HaltAll,
                applies_to: vec!["*".to_string()],
            }],
        };
        let engine = TimeRuleEngine::new(config);
        let market = create_test_market("id1", "Test Market", vec![], 1);

        let result = engine.check_market(&market);
        assert!(matches!(result, TimeRuleResult::Allowed));
    }

    #[test]
    fn test_no_rules_without_end_date() {
        let config = default_config();
        let engine = TimeRuleEngine::new(config);
        let market = create_test_market_no_end("id1", "Test Market");

        let result = engine.check_market(&market);
        assert!(matches!(result, TimeRuleResult::Allowed));
    }

    #[test]
    fn test_reduce_size_rule() {
        let config = default_config();
        let engine = TimeRuleEngine::new(config);
        // 12 hours until end - should trigger 24h reduction rule
        let market = create_test_market("id1", "Test Market", vec![], 12);

        let result = engine.check_market(&market);
        assert!(matches!(
            result,
            TimeRuleResult::ReduceSize { multiplier, .. } if multiplier == dec!(0.5)
        ));
    }

    #[test]
    fn test_block_new_rule() {
        let config = default_config();
        let engine = TimeRuleEngine::new(config);
        // 1 hour until end - should trigger 2h block rule
        let market = create_test_market("id1", "Test Market", vec![], 1);

        let result = engine.check_market(&market);
        assert!(matches!(result, TimeRuleResult::BlockNew { .. }));
    }

    #[test]
    fn test_halt_after_market_ends() {
        let config = default_config();
        let engine = TimeRuleEngine::new(config);
        // Market ended 2 hours ago
        let market = create_test_market("id1", "Test Market", vec![], -2);

        let result = engine.check_market(&market);
        assert!(matches!(result, TimeRuleResult::HaltAll { .. }));
    }

    #[test]
    fn test_no_rules_far_from_end() {
        let config = default_config();
        let engine = TimeRuleEngine::new(config);
        // 72 hours until end - no rules should apply
        let market = create_test_market("id1", "Test Market", vec![], 72);

        let result = engine.check_market(&market);
        assert!(matches!(result, TimeRuleResult::Allowed));
    }

    #[test]
    fn test_glob_wildcard_matches_all() {
        let config = TimeRulesConfig {
            enabled: true,
            rules: vec![TimeRule {
                name: "test".to_string(),
                hours_before: 100,
                action: TimeRuleAction::BlockNew,
                applies_to: vec!["*".to_string()],
            }],
        };
        let engine = TimeRuleEngine::new(config);
        let market = create_test_market("any_id", "Any Question", vec!["politics"], 50);

        let result = engine.check_market(&market);
        assert!(matches!(result, TimeRuleResult::BlockNew { .. }));
    }

    #[test]
    fn test_glob_prefix_match() {
        let config = TimeRulesConfig {
            enabled: true,
            rules: vec![TimeRule {
                name: "election_rule".to_string(),
                hours_before: 100,
                action: TimeRuleAction::BlockNew,
                applies_to: vec!["election*".to_string()],
            }],
        };
        let engine = TimeRuleEngine::new(config);

        let market1 = create_test_market("id1", "Election 2024 Results", vec![], 50);
        let market2 = create_test_market("id2", "Bitcoin Price", vec![], 50);

        assert!(matches!(
            engine.check_market(&market1),
            TimeRuleResult::BlockNew { .. }
        ));
        assert!(matches!(
            engine.check_market(&market2),
            TimeRuleResult::Allowed
        ));
    }

    #[test]
    fn test_glob_suffix_match() {
        let config = TimeRulesConfig {
            enabled: true,
            rules: vec![TimeRule {
                name: "crypto_rule".to_string(),
                hours_before: 100,
                action: TimeRuleAction::BlockNew,
                applies_to: vec!["*price".to_string()],
            }],
        };
        let engine = TimeRuleEngine::new(config);

        let market1 = create_test_market("id1", "Bitcoin Price", vec![], 50);
        let market2 = create_test_market("id2", "Election Results", vec![], 50);

        assert!(matches!(
            engine.check_market(&market1),
            TimeRuleResult::BlockNew { .. }
        ));
        assert!(matches!(
            engine.check_market(&market2),
            TimeRuleResult::Allowed
        ));
    }

    #[test]
    fn test_glob_contains_match() {
        let config = TimeRulesConfig {
            enabled: true,
            rules: vec![TimeRule {
                name: "trump_rule".to_string(),
                hours_before: 100,
                action: TimeRuleAction::BlockNew,
                applies_to: vec!["*trump*".to_string()],
            }],
        };
        let engine = TimeRuleEngine::new(config);

        let market1 = create_test_market("id1", "Will Trump win the election?", vec![], 50);
        let market2 = create_test_market("id2", "Bitcoin Price", vec![], 50);

        assert!(matches!(
            engine.check_market(&market1),
            TimeRuleResult::BlockNew { .. }
        ));
        assert!(matches!(
            engine.check_market(&market2),
            TimeRuleResult::Allowed
        ));
    }

    #[test]
    fn test_glob_tag_match() {
        let config = TimeRulesConfig {
            enabled: true,
            rules: vec![TimeRule {
                name: "politics_rule".to_string(),
                hours_before: 100,
                action: TimeRuleAction::BlockNew,
                applies_to: vec!["politics".to_string()],
            }],
        };
        let engine = TimeRuleEngine::new(config);

        let market1 = create_test_market("id1", "Some Question", vec!["politics", "usa"], 50);
        let market2 = create_test_market("id2", "Bitcoin Price", vec!["crypto"], 50);

        assert!(matches!(
            engine.check_market(&market1),
            TimeRuleResult::BlockNew { .. }
        ));
        assert!(matches!(
            engine.check_market(&market2),
            TimeRuleResult::Allowed
        ));
    }

    #[test]
    fn test_most_restrictive_wins() {
        let config = TimeRulesConfig {
            enabled: true,
            rules: vec![
                TimeRule {
                    name: "reduce".to_string(),
                    hours_before: 100,
                    action: TimeRuleAction::ReduceSize {
                        multiplier: dec!(0.5),
                    },
                    applies_to: vec!["*".to_string()],
                },
                TimeRule {
                    name: "halt".to_string(),
                    hours_before: 10,
                    action: TimeRuleAction::HaltAll,
                    applies_to: vec!["*".to_string()],
                },
            ],
        };
        let engine = TimeRuleEngine::new(config);
        let market = create_test_market("id1", "Test Market", vec![], 5);

        let result = engine.check_market(&market);
        assert!(matches!(result, TimeRuleResult::HaltAll { rule_name } if rule_name == "halt"));
    }

    #[test]
    fn test_smallest_multiplier_wins() {
        let config = TimeRulesConfig {
            enabled: true,
            rules: vec![
                TimeRule {
                    name: "reduce_50".to_string(),
                    hours_before: 100,
                    action: TimeRuleAction::ReduceSize {
                        multiplier: dec!(0.5),
                    },
                    applies_to: vec!["*".to_string()],
                },
                TimeRule {
                    name: "reduce_25".to_string(),
                    hours_before: 50,
                    action: TimeRuleAction::ReduceSize {
                        multiplier: dec!(0.25),
                    },
                    applies_to: vec!["*".to_string()],
                },
            ],
        };
        let engine = TimeRuleEngine::new(config);
        let market = create_test_market("id1", "Test Market", vec![], 40);

        let result = engine.check_market(&market);
        assert!(matches!(
            result,
            TimeRuleResult::ReduceSize { multiplier, rule_name }
            if multiplier == dec!(0.25) && rule_name == "reduce_25"
        ));
    }

    #[test]
    fn test_size_modifier() {
        let config = default_config();
        let engine = TimeRuleEngine::new(config);

        // Far from end - no reduction
        let market1 = create_test_market("id1", "Test", vec![], 72);
        assert_eq!(engine.get_size_modifier(&market1), Decimal::ONE);

        // Within 24h - 50% reduction
        let market2 = create_test_market("id2", "Test", vec![], 12);
        assert_eq!(engine.get_size_modifier(&market2), dec!(0.5));

        // Within 2h - blocked (size multiplier still available)
        let market3 = create_test_market("id3", "Test", vec![], 1);
        assert!(engine.is_new_blocked(&market3));
    }

    #[test]
    fn test_is_blocked() {
        let config = TimeRulesConfig {
            enabled: true,
            rules: vec![TimeRule {
                name: "halt".to_string(),
                hours_before: 10,
                action: TimeRuleAction::HaltAll,
                applies_to: vec!["*".to_string()],
            }],
        };
        let engine = TimeRuleEngine::new(config);

        let market1 = create_test_market("id1", "Test", vec![], 5);
        let market2 = create_test_market("id2", "Test", vec![], 20);

        assert!(engine.is_blocked(&market1));
        assert!(!engine.is_blocked(&market2));
    }

    #[test]
    fn test_signal_exit_allowed_during_block_new() {
        use polysniper_core::{OrderType, Outcome, Priority};

        let config = TimeRulesConfig {
            enabled: true,
            rules: vec![TimeRule {
                name: "block".to_string(),
                hours_before: 10,
                action: TimeRuleAction::BlockNew,
                applies_to: vec!["*".to_string()],
            }],
        };
        let engine = TimeRuleEngine::new(config);
        let market = create_test_market("id1", "Test", vec![], 5);

        let buy_signal = TradeSignal {
            id: "sig1".to_string(),
            strategy_id: "strat1".to_string(),
            market_id: "id1".to_string(),
            token_id: "token1".to_string(),
            outcome: Outcome::Yes,
            side: Side::Buy,
            price: Some(dec!(0.5)),
            size: dec!(100),
            size_usd: dec!(50),
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: Utc::now(),
            reason: "test".to_string(),
            metadata: serde_json::Value::Null,
        };

        let sell_signal = TradeSignal {
            side: Side::Sell,
            ..buy_signal.clone()
        };

        // Buy should be blocked
        let buy_result = engine.check_signal(&buy_signal, &market);
        assert!(matches!(buy_result, TimeRuleResult::BlockNew { .. }));

        // Sell (exit) should be allowed
        let sell_result = engine.check_signal(&sell_signal, &market);
        assert!(matches!(sell_result, TimeRuleResult::Allowed));
    }

    #[test]
    fn test_signal_exit_blocked_during_halt() {
        use polysniper_core::{OrderType, Outcome, Priority};

        let config = TimeRulesConfig {
            enabled: true,
            rules: vec![TimeRule {
                name: "halt".to_string(),
                hours_before: 10,
                action: TimeRuleAction::HaltAll,
                applies_to: vec!["*".to_string()],
            }],
        };
        let engine = TimeRuleEngine::new(config);
        let market = create_test_market("id1", "Test", vec![], 5);

        let sell_signal = TradeSignal {
            id: "sig1".to_string(),
            strategy_id: "strat1".to_string(),
            market_id: "id1".to_string(),
            token_id: "token1".to_string(),
            outcome: Outcome::Yes,
            side: Side::Sell,
            price: Some(dec!(0.5)),
            size: dec!(100),
            size_usd: dec!(50),
            order_type: OrderType::Gtc,
            priority: Priority::Normal,
            timestamp: Utc::now(),
            reason: "test".to_string(),
            metadata: serde_json::Value::Null,
        };

        // Even sells should be blocked during HaltAll
        let result = engine.check_signal(&sell_signal, &market);
        assert!(matches!(result, TimeRuleResult::HaltAll { .. }));
    }

    #[test]
    fn test_case_insensitive_matching() {
        let config = TimeRulesConfig {
            enabled: true,
            rules: vec![TimeRule {
                name: "test".to_string(),
                hours_before: 100,
                action: TimeRuleAction::BlockNew,
                applies_to: vec!["*ELECTION*".to_string()],
            }],
        };
        let engine = TimeRuleEngine::new(config);
        let market = create_test_market("id1", "Will the election happen?", vec![], 50);

        let result = engine.check_market(&market);
        assert!(matches!(result, TimeRuleResult::BlockNew { .. }));
    }
}
