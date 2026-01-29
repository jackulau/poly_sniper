//! Correlation tracking for position limit management
//!
//! This module provides correlation-aware position limits by tracking
//! relationships between markets and enforcing aggregate exposure limits
//! for correlated positions.

use polysniper_core::{CorrelationConfig, Position, StateProvider};
use rust_decimal::Decimal;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

/// A group of correlated markets
#[derive(Debug, Clone)]
pub struct CorrelationGroup {
    /// Unique identifier for this group
    pub id: String,
    /// Market IDs in this group
    pub market_ids: HashSet<String>,
    /// Whether this is a manually configured group
    pub is_manual: bool,
}

impl CorrelationGroup {
    /// Create a new correlation group
    pub fn new(id: String, market_ids: HashSet<String>, is_manual: bool) -> Self {
        Self {
            id,
            market_ids,
            is_manual,
        }
    }

    /// Check if a market is in this group
    pub fn contains(&self, market_id: &str) -> bool {
        self.market_ids.contains(market_id)
    }

    /// Add a market to this group
    pub fn add_market(&mut self, market_id: String) {
        self.market_ids.insert(market_id);
    }
}

/// Tracks correlations between markets and calculates correlated exposure
pub struct CorrelationTracker {
    config: CorrelationConfig,
    /// Manual groups from configuration
    manual_groups: Vec<CorrelationGroup>,
    /// Dynamically calculated groups (from price correlation)
    dynamic_groups: Vec<CorrelationGroup>,
    /// Cache of market to group mappings for fast lookup
    market_to_groups: HashMap<String, Vec<String>>,
}

impl CorrelationTracker {
    /// Create a new correlation tracker from configuration
    pub fn new(config: CorrelationConfig) -> Self {
        let mut tracker = Self {
            config: config.clone(),
            manual_groups: Vec::new(),
            dynamic_groups: Vec::new(),
            market_to_groups: HashMap::new(),
        };

        // Initialize manual groups from config
        for group_config in config.groups.iter() {
            let group = CorrelationGroup::new(
                group_config.name.clone(),
                group_config.markets.iter().cloned().collect(),
                true,
            );
            tracker.add_manual_group(group);
            debug!(
                group_name = %group_config.name,
                markets = ?group_config.markets,
                "Loaded manual correlation group"
            );
        }

        info!(
            manual_groups = tracker.manual_groups.len(),
            "Correlation tracker initialized"
        );

        tracker
    }

    /// Add a manually configured correlation group
    fn add_manual_group(&mut self, group: CorrelationGroup) {
        let group_id = group.id.clone();
        for market_id in &group.market_ids {
            self.market_to_groups
                .entry(market_id.clone())
                .or_default()
                .push(group_id.clone());
        }
        self.manual_groups.push(group);
    }

    /// Check if correlation tracking is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the maximum correlated exposure limit
    pub fn max_correlated_exposure(&self) -> Decimal {
        self.config.max_correlated_exposure_usd
    }

    /// Calculate Pearson correlation coefficient between two price series
    ///
    /// Returns a value between -1 and 1, where:
    /// - 1 indicates perfect positive correlation
    /// - -1 indicates perfect negative correlation
    /// - 0 indicates no correlation
    pub fn calculate_correlation(prices_a: &[Decimal], prices_b: &[Decimal]) -> Option<Decimal> {
        if prices_a.len() != prices_b.len() || prices_a.len() < 2 {
            return None;
        }

        let n = Decimal::from(prices_a.len());

        // Calculate means
        let sum_a: Decimal = prices_a.iter().sum();
        let sum_b: Decimal = prices_b.iter().sum();
        let mean_a = sum_a / n;
        let mean_b = sum_b / n;

        // Calculate covariance and standard deviations
        let mut cov = Decimal::ZERO;
        let mut var_a = Decimal::ZERO;
        let mut var_b = Decimal::ZERO;

        for (a, b) in prices_a.iter().zip(prices_b.iter()) {
            let diff_a = *a - mean_a;
            let diff_b = *b - mean_b;
            cov += diff_a * diff_b;
            var_a += diff_a * diff_a;
            var_b += diff_b * diff_b;
        }

        if var_a.is_zero() || var_b.is_zero() {
            return None;
        }

        // Pearson correlation = cov / (std_a * std_b)
        // We use an approximation for sqrt since Decimal doesn't have native sqrt
        let std_a = Self::decimal_sqrt(var_a)?;
        let std_b = Self::decimal_sqrt(var_b)?;

        if std_a.is_zero() || std_b.is_zero() {
            return None;
        }

        Some(cov / (std_a * std_b))
    }

    /// Approximate square root using Newton-Raphson method
    fn decimal_sqrt(n: Decimal) -> Option<Decimal> {
        if n < Decimal::ZERO {
            return None;
        }
        if n.is_zero() {
            return Some(Decimal::ZERO);
        }

        let mut x = n;
        let two = Decimal::new(2, 0);
        let precision = Decimal::new(1, 10); // 0.0000000001

        // Newton-Raphson iterations
        for _ in 0..100 {
            let next_x = (x + n / x) / two;
            if (next_x - x).abs() < precision {
                return Some(next_x);
            }
            x = next_x;
        }

        Some(x)
    }

    /// Get all groups that contain the specified market
    pub fn get_groups_for_market(&self, market_id: &str) -> Vec<&CorrelationGroup> {
        let mut groups = Vec::new();

        // Check manual groups first (including pattern matching)
        for group in &self.manual_groups {
            if group.contains(market_id) || self.matches_pattern(market_id, &group.market_ids) {
                groups.push(group);
            }
        }

        // Check dynamic groups
        for group in &self.dynamic_groups {
            if group.contains(market_id) {
                groups.push(group);
            }
        }

        groups
    }

    /// Check if a market ID matches any pattern in the group
    fn matches_pattern(&self, market_id: &str, patterns: &HashSet<String>) -> bool {
        for pattern in patterns {
            if pattern.contains('*') {
                // Simple glob pattern matching
                let prefix = pattern.trim_end_matches('*');
                if market_id.starts_with(prefix) {
                    return true;
                }
            }
        }
        false
    }

    /// Get all correlated market IDs for a given market
    pub fn get_correlated_markets(&self, market_id: &str) -> HashSet<String> {
        let mut correlated = HashSet::new();

        for group in self.get_groups_for_market(market_id) {
            for id in &group.market_ids {
                if id != market_id && !id.contains('*') {
                    correlated.insert(id.clone());
                }
            }
        }

        correlated
    }

    /// Calculate total exposure across all correlated positions for a market
    pub async fn calculate_correlated_exposure(
        &self,
        market_id: &str,
        state: &dyn StateProvider,
    ) -> Decimal {
        if !self.config.enabled {
            return Decimal::ZERO;
        }

        let mut total_exposure = Decimal::ZERO;
        let mut counted_markets = HashSet::new();

        // Get all groups this market belongs to
        let groups = self.get_groups_for_market(market_id);

        for group in &groups {
            // Add exposure from all markets in the group
            for correlated_market_id in &group.market_ids {
                // Skip patterns and already counted markets
                if correlated_market_id.contains('*') || counted_markets.contains(correlated_market_id) {
                    continue;
                }

                if let Some(position) = state.get_position(correlated_market_id).await {
                    let exposure = self.calculate_position_exposure(&position, state).await;
                    total_exposure += exposure;
                    counted_markets.insert(correlated_market_id.clone());
                }
            }
        }

        // Also check pattern matches against actual positions
        let all_positions = state.get_all_positions().await;
        for position in all_positions {
            if counted_markets.contains(&position.market_id) {
                continue;
            }

            // Check if this position matches any pattern in the groups
            for group in &groups {
                if self.matches_pattern(&position.market_id, &group.market_ids) {
                    let exposure = self.calculate_position_exposure(&position, state).await;
                    total_exposure += exposure;
                    counted_markets.insert(position.market_id.clone());
                    break;
                }
            }
        }

        debug!(
            market_id = %market_id,
            correlated_markets = counted_markets.len(),
            total_exposure = %total_exposure,
            "Calculated correlated exposure"
        );

        total_exposure
    }

    /// Calculate exposure for a single position
    async fn calculate_position_exposure(&self, position: &Position, state: &dyn StateProvider) -> Decimal {
        // Get current price for the position's token
        if let Some(price) = state.get_price(&position.token_id).await {
            position.size.abs() * price
        } else {
            // Fallback to average price if current price unavailable
            position.size.abs() * position.avg_price
        }
    }

    /// Calculate remaining room for new exposure in correlated group
    pub async fn calculate_remaining_room(
        &self,
        market_id: &str,
        state: &dyn StateProvider,
    ) -> Decimal {
        if !self.config.enabled {
            return Decimal::MAX;
        }

        let current_exposure = self.calculate_correlated_exposure(market_id, state).await;
        let remaining = self.config.max_correlated_exposure_usd - current_exposure;

        if remaining < Decimal::ZERO {
            Decimal::ZERO
        } else {
            remaining
        }
    }

    /// Check if adding a new position would exceed correlated exposure limit
    pub async fn would_exceed_limit(
        &self,
        market_id: &str,
        additional_exposure_usd: Decimal,
        state: &dyn StateProvider,
    ) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Only check if market is in a correlation group
        if self.get_groups_for_market(market_id).is_empty() {
            return false;
        }

        let current_exposure = self.calculate_correlated_exposure(market_id, state).await;
        let new_total = current_exposure + additional_exposure_usd;

        if new_total > self.config.max_correlated_exposure_usd {
            warn!(
                market_id = %market_id,
                current_exposure = %current_exposure,
                additional = %additional_exposure_usd,
                new_total = %new_total,
                limit = %self.config.max_correlated_exposure_usd,
                "Correlated exposure limit would be exceeded"
            );
            true
        } else {
            false
        }
    }

    /// Update dynamic correlation groups based on price history
    ///
    /// This should be called periodically to recalculate correlations
    pub async fn update_dynamic_groups(&mut self, state: &dyn StateProvider) {
        if !self.config.enabled {
            return;
        }

        let all_positions = state.get_all_positions().await;
        if all_positions.len() < 2 {
            return;
        }

        // Collect price histories for all positions
        let mut price_histories: HashMap<String, Vec<Decimal>> = HashMap::new();

        for position in &all_positions {
            let history = state
                .get_price_history(&position.token_id, 100)
                .await;
            if history.len() >= 10 {
                price_histories.insert(
                    position.market_id.clone(),
                    history.iter().map(|(_, p)| *p).collect(),
                );
            }
        }

        // Calculate pairwise correlations
        let market_ids: Vec<_> = price_histories.keys().cloned().collect();
        let mut new_groups: HashMap<String, HashSet<String>> = HashMap::new();
        let mut group_counter = 0;

        for i in 0..market_ids.len() {
            for j in (i + 1)..market_ids.len() {
                let id_a = &market_ids[i];
                let id_b = &market_ids[j];

                if let (Some(prices_a), Some(prices_b)) =
                    (price_histories.get(id_a), price_histories.get(id_b))
                {
                    // Ensure same length
                    let min_len = prices_a.len().min(prices_b.len());
                    if min_len < 10 {
                        continue;
                    }

                    let prices_a: Vec<_> = prices_a.iter().take(min_len).copied().collect();
                    let prices_b: Vec<_> = prices_b.iter().take(min_len).copied().collect();

                    if let Some(correlation) = Self::calculate_correlation(&prices_a, &prices_b) {
                        if correlation.abs() >= self.config.correlation_threshold {
                            // Find or create group for these markets
                            let group_id = format!("dynamic_{}", group_counter);
                            group_counter += 1;

                            let group = new_groups.entry(group_id).or_default();
                            group.insert(id_a.clone());
                            group.insert(id_b.clone());

                            debug!(
                                market_a = %id_a,
                                market_b = %id_b,
                                correlation = %correlation,
                                "Found correlated markets"
                            );
                        }
                    }
                }
            }
        }

        // Merge overlapping groups
        let merged_groups = Self::merge_overlapping_groups(new_groups);

        // Update dynamic groups
        self.dynamic_groups = merged_groups
            .into_iter()
            .map(|(id, markets)| CorrelationGroup::new(id, markets, false))
            .collect();

        // Rebuild market-to-groups cache for dynamic groups
        for group in &self.dynamic_groups {
            for market_id in &group.market_ids {
                self.market_to_groups
                    .entry(market_id.clone())
                    .or_default()
                    .push(group.id.clone());
            }
        }

        info!(
            dynamic_groups = self.dynamic_groups.len(),
            "Updated dynamic correlation groups"
        );
    }

    /// Merge groups that have overlapping markets
    fn merge_overlapping_groups(
        groups: HashMap<String, HashSet<String>>,
    ) -> HashMap<String, HashSet<String>> {
        let mut merged: Vec<HashSet<String>> = Vec::new();

        for (_, markets) in groups {
            // Find any existing group that overlaps
            let mut found_idx = None;
            for (idx, existing) in merged.iter().enumerate() {
                if markets.iter().any(|m| existing.contains(m)) {
                    found_idx = Some(idx);
                    break;
                }
            }

            if let Some(idx) = found_idx {
                // Merge into existing group
                merged[idx].extend(markets);
            } else {
                // Create new group
                merged.push(markets);
            }
        }

        // Convert back to HashMap with generated IDs
        merged
            .into_iter()
            .enumerate()
            .map(|(idx, markets)| (format!("dynamic_{}", idx), markets))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polysniper_core::CorrelationGroupConfig;
    use rust_decimal_macros::dec;

    #[test]
    fn test_correlation_calculation_perfect_positive() {
        let prices_a = vec![dec!(1.0), dec!(2.0), dec!(3.0), dec!(4.0), dec!(5.0)];
        let prices_b = vec![dec!(2.0), dec!(4.0), dec!(6.0), dec!(8.0), dec!(10.0)];

        let corr = CorrelationTracker::calculate_correlation(&prices_a, &prices_b).unwrap();
        assert!(corr > dec!(0.99), "Expected perfect positive correlation, got {}", corr);
    }

    #[test]
    fn test_correlation_calculation_perfect_negative() {
        let prices_a = vec![dec!(1.0), dec!(2.0), dec!(3.0), dec!(4.0), dec!(5.0)];
        let prices_b = vec![dec!(10.0), dec!(8.0), dec!(6.0), dec!(4.0), dec!(2.0)];

        let corr = CorrelationTracker::calculate_correlation(&prices_a, &prices_b).unwrap();
        assert!(corr < dec!(-0.99), "Expected perfect negative correlation, got {}", corr);
    }

    #[test]
    fn test_correlation_calculation_no_correlation() {
        let prices_a = vec![dec!(1.0), dec!(2.0), dec!(3.0), dec!(2.0), dec!(1.0)];
        let prices_b = vec![dec!(5.0), dec!(5.0), dec!(5.0), dec!(5.0), dec!(5.0)];

        let corr = CorrelationTracker::calculate_correlation(&prices_a, &prices_b);
        // Should return None because std_b is 0
        assert!(corr.is_none());
    }

    #[test]
    fn test_correlation_mismatched_lengths() {
        let prices_a = vec![dec!(1.0), dec!(2.0), dec!(3.0)];
        let prices_b = vec![dec!(1.0), dec!(2.0)];

        let corr = CorrelationTracker::calculate_correlation(&prices_a, &prices_b);
        assert!(corr.is_none());
    }

    #[test]
    fn test_correlation_too_few_samples() {
        let prices_a = vec![dec!(1.0)];
        let prices_b = vec![dec!(2.0)];

        let corr = CorrelationTracker::calculate_correlation(&prices_a, &prices_b);
        assert!(corr.is_none());
    }

    #[test]
    fn test_manual_group_contains() {
        let mut markets = HashSet::new();
        markets.insert("market-a".to_string());
        markets.insert("market-b".to_string());

        let group = CorrelationGroup::new("test-group".to_string(), markets, true);

        assert!(group.contains("market-a"));
        assert!(group.contains("market-b"));
        assert!(!group.contains("market-c"));
    }

    #[test]
    fn test_tracker_with_manual_groups() {
        let config = CorrelationConfig {
            enabled: true,
            correlation_threshold: dec!(0.7),
            window_secs: 3600,
            max_correlated_exposure_usd: dec!(3000),
            groups: vec![CorrelationGroupConfig {
                name: "election".to_string(),
                markets: vec![
                    "presidential-winner".to_string(),
                    "electoral-count".to_string(),
                ],
            }],
        };

        let tracker = CorrelationTracker::new(config);

        assert!(tracker.is_enabled());
        assert_eq!(tracker.max_correlated_exposure(), dec!(3000));

        let groups = tracker.get_groups_for_market("presidential-winner");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].id, "election");
    }

    #[test]
    fn test_pattern_matching() {
        let config = CorrelationConfig {
            enabled: true,
            correlation_threshold: dec!(0.7),
            window_secs: 3600,
            max_correlated_exposure_usd: dec!(3000),
            groups: vec![CorrelationGroupConfig {
                name: "swing-states".to_string(),
                markets: vec!["swing-state-*".to_string()],
            }],
        };

        let tracker = CorrelationTracker::new(config);

        // Pattern should match
        let groups = tracker.get_groups_for_market("swing-state-pennsylvania");
        assert_eq!(groups.len(), 1);

        // Pattern should not match
        let groups = tracker.get_groups_for_market("other-market");
        assert!(groups.is_empty());
    }

    #[test]
    fn test_get_correlated_markets() {
        let config = CorrelationConfig {
            enabled: true,
            correlation_threshold: dec!(0.7),
            window_secs: 3600,
            max_correlated_exposure_usd: dec!(3000),
            groups: vec![CorrelationGroupConfig {
                name: "test".to_string(),
                markets: vec![
                    "market-a".to_string(),
                    "market-b".to_string(),
                    "market-c".to_string(),
                ],
            }],
        };

        let tracker = CorrelationTracker::new(config);

        let correlated = tracker.get_correlated_markets("market-a");
        assert_eq!(correlated.len(), 2);
        assert!(correlated.contains("market-b"));
        assert!(correlated.contains("market-c"));
    }

    #[test]
    fn test_decimal_sqrt() {
        let result = CorrelationTracker::decimal_sqrt(dec!(4)).unwrap();
        assert!((result - dec!(2)).abs() < dec!(0.0001));

        let result = CorrelationTracker::decimal_sqrt(dec!(9)).unwrap();
        assert!((result - dec!(3)).abs() < dec!(0.0001));

        let result = CorrelationTracker::decimal_sqrt(dec!(2)).unwrap();
        assert!((result - dec!(1.414213562)).abs() < dec!(0.0001));

        assert!(CorrelationTracker::decimal_sqrt(dec!(-1)).is_none());
        assert_eq!(CorrelationTracker::decimal_sqrt(dec!(0)), Some(dec!(0)));
    }

    #[test]
    fn test_merge_overlapping_groups() {
        let mut groups: HashMap<String, HashSet<String>> = HashMap::new();

        // Group 1: a, b
        let mut g1 = HashSet::new();
        g1.insert("a".to_string());
        g1.insert("b".to_string());
        groups.insert("g1".to_string(), g1);

        // Group 2: b, c (overlaps with g1 via "b")
        let mut g2 = HashSet::new();
        g2.insert("b".to_string());
        g2.insert("c".to_string());
        groups.insert("g2".to_string(), g2);

        // Group 3: d, e (no overlap)
        let mut g3 = HashSet::new();
        g3.insert("d".to_string());
        g3.insert("e".to_string());
        groups.insert("g3".to_string(), g3);

        let merged = CorrelationTracker::merge_overlapping_groups(groups);

        // Should have 2 groups after merge
        assert_eq!(merged.len(), 2);

        // One group should have a, b, c
        let has_abc = merged.values().any(|g| g.len() == 3 && g.contains("a") && g.contains("b") && g.contains("c"));
        assert!(has_abc, "Expected merged group with a, b, c");

        // One group should have d, e
        let has_de = merged.values().any(|g| g.len() == 2 && g.contains("d") && g.contains("e"));
        assert!(has_de, "Expected group with d, e");
    }
}
