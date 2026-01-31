//! Thompson Sampling Implementation
//!
//! Implements Thompson Sampling using Beta distributions for multi-armed
//! bandit-style model selection. This allows the trading system to explore
//! different models while exploiting the best performers.

use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::f64::consts::PI;

/// Thompson Sampler using Beta distribution
#[derive(Debug, Clone)]
pub struct ThompsonSampler {
    /// Prior alpha (initial successes)
    prior_alpha: f64,
    /// Prior beta (initial failures)
    prior_beta: f64,
}

impl ThompsonSampler {
    /// Create a new Thompson sampler with given priors
    pub fn new(prior_alpha: f64, prior_beta: f64) -> Self {
        Self {
            prior_alpha,
            prior_beta,
        }
    }

    /// Sample from Beta(alpha, beta) distribution
    /// Uses the ratio of Gamma random variables method
    pub fn sample(&self, alpha: f64, beta: f64) -> f64 {
        let mut rng = rand::thread_rng();
        sample_beta(&mut rng, alpha, beta)
    }

    /// Sample from posterior with prior included
    pub fn sample_with_prior(&self, successes: u64, failures: u64) -> f64 {
        let alpha = self.prior_alpha + successes as f64;
        let beta = self.prior_beta + failures as f64;
        self.sample(alpha, beta)
    }

    /// Get the expected value (mean) of Beta(alpha, beta)
    pub fn expected_value(alpha: f64, beta: f64) -> f64 {
        alpha / (alpha + beta)
    }

    /// Get the variance of Beta(alpha, beta)
    pub fn variance(alpha: f64, beta: f64) -> f64 {
        let sum = alpha + beta;
        (alpha * beta) / (sum * sum * (sum + 1.0))
    }

    /// Get the mode of Beta(alpha, beta)
    /// Returns None if alpha or beta <= 1
    pub fn mode(alpha: f64, beta: f64) -> Option<f64> {
        if alpha > 1.0 && beta > 1.0 {
            Some((alpha - 1.0) / (alpha + beta - 2.0))
        } else {
            None
        }
    }

    /// Calculate Upper Confidence Bound (UCB) for a model
    /// Useful for comparison with other exploration strategies
    pub fn ucb(alpha: f64, beta: f64, total_pulls: u64, c: f64) -> f64 {
        let mean = Self::expected_value(alpha, beta);
        let n = alpha + beta - 2.0; // Number of observations (minus priors)
        if n <= 0.0 || total_pulls == 0 {
            return f64::INFINITY; // Encourage exploration of new arms
        }
        mean + c * (2.0 * (total_pulls as f64).ln() / n).sqrt()
    }
}

/// Sample from Beta distribution using Gamma ratio method
fn sample_beta<R: Rng>(rng: &mut R, alpha: f64, beta: f64) -> f64 {
    let x = sample_gamma(rng, alpha);
    let y = sample_gamma(rng, beta);
    x / (x + y)
}

/// Sample from Gamma distribution using Marsaglia and Tsang's method
fn sample_gamma<R: Rng>(rng: &mut R, shape: f64) -> f64 {
    if shape < 1.0 {
        // Use Ahrens-Dieter method for shape < 1
        let u = rng.gen::<f64>();
        sample_gamma(rng, shape + 1.0) * u.powf(1.0 / shape)
    } else {
        // Marsaglia and Tsang's method
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let x = sample_standard_normal(rng);
            let v = 1.0 + c * x;

            if v > 0.0 {
                let v = v * v * v;
                let u = rng.gen::<f64>();

                if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                    return d * v;
                }

                if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                    return d * v;
                }
            }
        }
    }
}

/// Sample from standard normal distribution using Box-Muller transform
fn sample_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    let uniform = Uniform::new(0.0_f64, 1.0);
    let u1 = uniform.sample(rng);
    let u2 = uniform.sample(rng);

    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Multi-armed bandit state for multiple models
#[derive(Debug, Clone)]
pub struct MultiArmedBandit {
    /// Thompson sampler
    sampler: ThompsonSampler,
    /// Arms (models) with their success/failure counts
    arms: Vec<ArmState>,
}

/// State of a single arm in the bandit
#[derive(Debug, Clone)]
pub struct ArmState {
    /// Arm identifier
    pub id: String,
    /// Number of successes (correct predictions)
    pub successes: u64,
    /// Number of failures (incorrect predictions)
    pub failures: u64,
    /// Total reward accumulated
    pub total_reward: f64,
}

impl ArmState {
    /// Create a new arm
    pub fn new(id: String) -> Self {
        Self {
            id,
            successes: 0,
            failures: 0,
            total_reward: 0.0,
        }
    }

    /// Get total pulls for this arm
    pub fn total_pulls(&self) -> u64 {
        self.successes + self.failures
    }

    /// Get empirical success rate
    pub fn success_rate(&self) -> Option<f64> {
        let total = self.total_pulls();
        if total == 0 {
            None
        } else {
            Some(self.successes as f64 / total as f64)
        }
    }
}

impl MultiArmedBandit {
    /// Create a new multi-armed bandit
    pub fn new(prior_alpha: f64, prior_beta: f64) -> Self {
        Self {
            sampler: ThompsonSampler::new(prior_alpha, prior_beta),
            arms: Vec::new(),
        }
    }

    /// Add a new arm (model)
    pub fn add_arm(&mut self, id: String) {
        if !self.arms.iter().any(|a| a.id == id) {
            self.arms.push(ArmState::new(id));
        }
    }

    /// Select an arm using Thompson Sampling
    /// Returns the arm ID with the highest sampled value
    pub fn select_arm(&self) -> Option<String> {
        if self.arms.is_empty() {
            return None;
        }

        let mut best_arm = None;
        let mut best_sample = f64::NEG_INFINITY;

        for arm in &self.arms {
            let sample = self.sampler.sample_with_prior(arm.successes, arm.failures);
            if sample > best_sample {
                best_sample = sample;
                best_arm = Some(arm.id.clone());
            }
        }

        best_arm
    }

    /// Select top k arms using Thompson Sampling
    pub fn select_top_k(&self, k: usize) -> Vec<(String, f64)> {
        let mut samples: Vec<(String, f64)> = self
            .arms
            .iter()
            .map(|arm| {
                let sample = self.sampler.sample_with_prior(arm.successes, arm.failures);
                (arm.id.clone(), sample)
            })
            .collect();

        samples.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        samples.truncate(k);
        samples
    }

    /// Update an arm with the outcome of a pull
    pub fn update_arm(&mut self, id: &str, success: bool, reward: f64) {
        if let Some(arm) = self.arms.iter_mut().find(|a| a.id == id) {
            if success {
                arm.successes += 1;
            } else {
                arm.failures += 1;
            }
            arm.total_reward += reward;
        }
    }

    /// Get the arm with the highest expected value
    pub fn best_arm_by_expected_value(&self) -> Option<&ArmState> {
        self.arms.iter().max_by(|a, b| {
            let alpha_a = self.sampler.prior_alpha + a.successes as f64;
            let beta_a = self.sampler.prior_beta + a.failures as f64;
            let alpha_b = self.sampler.prior_alpha + b.successes as f64;
            let beta_b = self.sampler.prior_beta + b.failures as f64;

            let ev_a = ThompsonSampler::expected_value(alpha_a, beta_a);
            let ev_b = ThompsonSampler::expected_value(alpha_b, beta_b);

            ev_a.partial_cmp(&ev_b).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get all arms
    pub fn arms(&self) -> &[ArmState] {
        &self.arms
    }

    /// Get an arm by ID
    pub fn get_arm(&self, id: &str) -> Option<&ArmState> {
        self.arms.iter().find(|a| a.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thompson_sampler_sample_in_range() {
        let sampler = ThompsonSampler::new(1.0, 1.0);

        // Sample should always be in [0, 1]
        for _ in 0..100 {
            let sample = sampler.sample(2.0, 3.0);
            assert!(sample >= 0.0 && sample <= 1.0);
        }
    }

    #[test]
    fn test_expected_value() {
        // Beta(2, 2) should have mean 0.5
        let ev = ThompsonSampler::expected_value(2.0, 2.0);
        assert!((ev - 0.5).abs() < 0.0001);

        // Beta(8, 2) should have mean 0.8
        let ev = ThompsonSampler::expected_value(8.0, 2.0);
        assert!((ev - 0.8).abs() < 0.0001);
    }

    #[test]
    fn test_variance() {
        // Beta(1, 1) should have variance 1/12 ≈ 0.0833
        let var = ThompsonSampler::variance(1.0, 1.0);
        assert!((var - 1.0 / 12.0).abs() < 0.0001);
    }

    #[test]
    fn test_mode() {
        // Beta(2, 2) should have mode 0.5
        let mode = ThompsonSampler::mode(2.0, 2.0);
        assert_eq!(mode, Some(0.5));

        // Beta(1, 1) should not have a mode
        let mode = ThompsonSampler::mode(1.0, 1.0);
        assert_eq!(mode, None);
    }

    #[test]
    fn test_multi_armed_bandit() {
        let mut bandit = MultiArmedBandit::new(1.0, 1.0);

        bandit.add_arm("model_a".to_string());
        bandit.add_arm("model_b".to_string());
        bandit.add_arm("model_c".to_string());

        // Simulate some outcomes - model_a is best
        for _ in 0..10 {
            bandit.update_arm("model_a", true, 1.0);
        }
        for _ in 0..3 {
            bandit.update_arm("model_a", false, 0.0);
        }

        for _ in 0..5 {
            bandit.update_arm("model_b", true, 1.0);
        }
        for _ in 0..5 {
            bandit.update_arm("model_b", false, 0.0);
        }

        for _ in 0..2 {
            bandit.update_arm("model_c", true, 1.0);
        }
        for _ in 0..8 {
            bandit.update_arm("model_c", false, 0.0);
        }

        // Best arm by expected value should be model_a
        let best = bandit.best_arm_by_expected_value().unwrap();
        assert_eq!(best.id, "model_a");
    }

    #[test]
    fn test_select_arm() {
        let mut bandit = MultiArmedBandit::new(1.0, 1.0);

        bandit.add_arm("good".to_string());
        bandit.add_arm("bad".to_string());

        // Make "good" clearly better
        for _ in 0..100 {
            bandit.update_arm("good", true, 1.0);
        }
        for _ in 0..100 {
            bandit.update_arm("bad", false, 0.0);
        }

        // Selection should heavily favor "good" arm
        let mut good_count = 0;
        for _ in 0..100 {
            if bandit.select_arm() == Some("good".to_string()) {
                good_count += 1;
            }
        }

        // Should select "good" most of the time (but not always due to exploration)
        assert!(good_count > 80);
    }

    #[test]
    fn test_sample_with_prior() {
        let sampler = ThompsonSampler::new(1.0, 1.0);

        // With 10 successes and 0 failures, samples should be high
        let mut high_samples = 0;
        for _ in 0..100 {
            if sampler.sample_with_prior(10, 0) > 0.7 {
                high_samples += 1;
            }
        }
        assert!(high_samples > 80);

        // With 0 successes and 10 failures, samples should be low
        let mut low_samples = 0;
        for _ in 0..100 {
            if sampler.sample_with_prior(0, 10) < 0.3 {
                low_samples += 1;
            }
        }
        assert!(low_samples > 80);
    }
}
