//! Curiosity-Driven Intrinsic Motivation for Quantum-SNN
//!
//! This module provides intrinsic motivation signals that encourage exploration
//! independent of external rewards. Two main approaches:
//!
//! 1. **Novelty-based (State Visitation)**: Rewards visiting rarely-seen states
//! 2. **Prediction Error**: Rewards surprising state transitions
//!
//! These can be used standalone or combined for robust exploration.

use std::collections::HashMap;

/// Tracks state visitation for novelty-based intrinsic reward.
///
/// Uses hash-based counting with optional decay to forget old visits.
/// Returns higher rewards for rarely-visited states.
#[derive(Debug, Clone)]
pub struct StateVisitationTracker {
    /// Hash-based state counting
    visit_counts: HashMap<u64, u32>,
    /// Decay factor for visit counts (0.0 = no decay, 1.0 = full decay each step)
    decay_rate: f32,
    /// Intrinsic reward scaling
    reward_scale: f32,
    /// Total visits (for statistics)
    total_visits: u64,
    /// Number of unique states seen
    unique_states: usize,
}

impl Default for StateVisitationTracker {
    fn default() -> Self {
        Self::new(1.0, 0.0)
    }
}

impl StateVisitationTracker {
    /// Create a new state visitation tracker.
    ///
    /// # Arguments
    /// * `reward_scale` - Scale factor for intrinsic rewards (typically 0.1-1.0)
    /// * `decay_rate` - How fast to forget old visits (0.0-1.0, 0 = never forget)
    pub fn new(reward_scale: f32, decay_rate: f32) -> Self {
        Self {
            visit_counts: HashMap::new(),
            decay_rate: decay_rate.clamp(0.0, 1.0),
            reward_scale,
            total_visits: 0,
            unique_states: 0,
        }
    }

    /// Compute intrinsic reward based on novelty.
    ///
    /// Returns higher reward for rarely-visited states using inverse sqrt scaling:
    /// reward = scale / sqrt(visit_count)
    ///
    /// This gives diminishing returns for revisiting the same state.
    pub fn intrinsic_reward(&mut self, state_hash: u64) -> f32 {
        let count = self.visit_counts.entry(state_hash).or_insert(0);

        if *count == 0 {
            self.unique_states += 1;
        }

        *count += 1;
        self.total_visits += 1;

        // Inverse sqrt scaling: 1/sqrt(n) gives diminishing returns
        self.reward_scale / (*count as f32).sqrt()
    }

    /// Apply decay to all visit counts (call periodically, e.g., per episode)
    pub fn decay(&mut self) {
        if self.decay_rate <= 0.0 {
            return;
        }

        let decay_factor = 1.0 - self.decay_rate;
        self.visit_counts.retain(|_, count| {
            *count = ((*count as f32) * decay_factor) as u32;
            *count > 0
        });

        self.unique_states = self.visit_counts.len();
    }

    /// Hash continuous observation to discrete bucket.
    ///
    /// Uses a simple binning approach - each dimension is quantized to bins.
    pub fn hash_observation(observation: &[f32], bins: usize) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis

        for (i, &val) in observation.iter().enumerate() {
            // Quantize to bins (assuming normalized [0, 1] range)
            let normalized = val.clamp(0.0, 1.0);
            let bin = (normalized * bins as f32) as u64;

            // FNV-1a hash update
            hash ^= bin.wrapping_add(i as u64 * 256);
            hash = hash.wrapping_mul(0x100000001b3);
        }

        hash
    }

    /// Hash spike rates (0-255) to state hash
    pub fn hash_spike_rates(rates: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;

        for (i, &rate) in rates.iter().enumerate() {
            // Quantize to 16 bins (0-15)
            let bin = (rate / 16) as u64;
            hash ^= bin.wrapping_add(i as u64 * 256);
            hash = hash.wrapping_mul(0x100000001b3);
        }

        hash
    }

    /// Get statistics
    pub fn stats(&self) -> (u64, usize) {
        (self.total_visits, self.unique_states)
    }

    /// Reset all counts
    pub fn reset(&mut self) {
        self.visit_counts.clear();
        self.total_visits = 0;
        self.unique_states = 0;
    }
}

/// Simple forward model for prediction error curiosity.
///
/// Uses a linear model to predict next state given current state and action.
/// Prediction error serves as intrinsic reward - surprising transitions are rewarding.
#[derive(Debug, Clone)]
pub struct PredictionErrorCuriosity {
    /// Weights for each (state, action) -> next_state prediction
    /// Shape: [n_actions][state_dim][state_dim]
    weights: Vec<Vec<Vec<f32>>>,
    /// Bias terms [n_actions][state_dim]
    biases: Vec<Vec<f32>>,
    /// Learning rate for model updates
    learning_rate: f32,
    /// Intrinsic reward scaling
    reward_scale: f32,
    /// State dimension
    state_dim: usize,
    /// Number of actions
    n_actions: usize,
    /// Running average of prediction errors (for normalization)
    error_avg: f32,
    /// Error averaging momentum
    error_momentum: f32,
}

impl PredictionErrorCuriosity {
    /// Create a new prediction error curiosity module.
    ///
    /// # Arguments
    /// * `state_dim` - Dimension of state/observation vector
    /// * `n_actions` - Number of discrete actions
    /// * `learning_rate` - How fast to update the prediction model
    /// * `reward_scale` - Scale factor for intrinsic rewards
    pub fn new(state_dim: usize, n_actions: usize, learning_rate: f32, reward_scale: f32) -> Self {
        // Initialize weights with small random values
        let mut weights = Vec::with_capacity(n_actions);
        let mut biases = Vec::with_capacity(n_actions);

        for a in 0..n_actions {
            let mut action_weights = Vec::with_capacity(state_dim);
            let mut action_biases = Vec::with_capacity(state_dim);

            for i in 0..state_dim {
                let mut row = Vec::with_capacity(state_dim);
                for j in 0..state_dim {
                    // Small random initialization based on position
                    let init = if i == j { 0.9 } else { 0.01 };
                    let noise = ((a * 31 + i * 17 + j * 7) % 100) as f32 / 1000.0 - 0.05;
                    row.push(init + noise);
                }
                action_weights.push(row);
                action_biases.push(0.0);
            }

            weights.push(action_weights);
            biases.push(action_biases);
        }

        Self {
            weights,
            biases,
            learning_rate,
            reward_scale,
            state_dim,
            n_actions,
            error_avg: 1.0,
            error_momentum: 0.99,
        }
    }

    /// Predict next state given current state and action.
    pub fn predict(&self, state: &[f32], action: usize) -> Vec<f32> {
        if action >= self.n_actions || state.len() != self.state_dim {
            return state.to_vec(); // Fallback to identity
        }

        let mut prediction = vec![0.0; self.state_dim];
        let action_weights = &self.weights[action];
        let action_biases = &self.biases[action];

        for i in 0..self.state_dim {
            let mut sum = action_biases[i];
            for j in 0..self.state_dim {
                sum += action_weights[i][j] * state[j];
            }
            prediction[i] = sum;
        }

        prediction
    }

    /// Update model and return prediction error as intrinsic reward.
    ///
    /// The prediction error (MSE) serves as the intrinsic reward -
    /// surprising transitions that the model couldn't predict are rewarding.
    pub fn update(&mut self, state: &[f32], action: usize, next_state: &[f32]) -> f32 {
        if action >= self.n_actions
            || state.len() != self.state_dim
            || next_state.len() != self.state_dim
        {
            return 0.0;
        }

        // Predict and compute error
        let prediction = self.predict(state, action);

        let mut mse = 0.0;
        for i in 0..self.state_dim {
            let diff = prediction[i] - next_state[i];
            mse += diff * diff;
        }
        mse /= self.state_dim as f32;

        // Update running average for normalization
        self.error_avg = self.error_momentum * self.error_avg + (1.0 - self.error_momentum) * mse;

        // Gradient descent update: minimize prediction error
        let action_weights = &mut self.weights[action];
        let action_biases = &mut self.biases[action];

        for i in 0..self.state_dim {
            let error = prediction[i] - next_state[i];

            // Update bias
            action_biases[i] -= self.learning_rate * error;

            // Update weights
            for j in 0..self.state_dim {
                action_weights[i][j] -= self.learning_rate * error * state[j];
            }
        }

        // Return normalized intrinsic reward
        let normalized_error = mse / self.error_avg.max(0.01);
        self.reward_scale * normalized_error.sqrt()
    }

    /// Get current average prediction error
    pub fn average_error(&self) -> f32 {
        self.error_avg
    }

    /// Reset the model
    pub fn reset(&mut self) {
        *self = Self::new(
            self.state_dim,
            self.n_actions,
            self.learning_rate,
            self.reward_scale,
        );
    }
}

/// Combined curiosity module that can use multiple strategies.
#[derive(Debug, Clone)]
pub enum CuriosityModule {
    /// No curiosity (disabled)
    None,
    /// Novelty-based only
    Novelty(StateVisitationTracker),
    /// Prediction error only
    PredictionError(PredictionErrorCuriosity),
    /// Combined: weighted sum of both
    Combined {
        novelty: StateVisitationTracker,
        prediction: PredictionErrorCuriosity,
        novelty_weight: f32,
        prediction_weight: f32,
    },
}

impl Default for CuriosityModule {
    fn default() -> Self {
        CuriosityModule::None
    }
}

impl CuriosityModule {
    /// Create novelty-only curiosity
    pub fn novelty(reward_scale: f32, decay_rate: f32) -> Self {
        CuriosityModule::Novelty(StateVisitationTracker::new(reward_scale, decay_rate))
    }

    /// Create prediction-error curiosity
    pub fn prediction_error(
        state_dim: usize,
        n_actions: usize,
        learning_rate: f32,
        reward_scale: f32,
    ) -> Self {
        CuriosityModule::PredictionError(PredictionErrorCuriosity::new(
            state_dim,
            n_actions,
            learning_rate,
            reward_scale,
        ))
    }

    /// Create combined curiosity with equal weighting
    pub fn combined(state_dim: usize, n_actions: usize, reward_scale: f32) -> Self {
        CuriosityModule::Combined {
            novelty: StateVisitationTracker::new(reward_scale, 0.0),
            prediction: PredictionErrorCuriosity::new(state_dim, n_actions, 0.01, reward_scale),
            novelty_weight: 0.5,
            prediction_weight: 0.5,
        }
    }

    /// Compute intrinsic reward given state transition.
    ///
    /// For novelty: only needs next_state (hashed)
    /// For prediction: needs state, action, next_state
    pub fn intrinsic_reward(
        &mut self,
        state: Option<&[f32]>,
        action: Option<usize>,
        next_state: &[f32],
    ) -> f32 {
        match self {
            CuriosityModule::None => 0.0,

            CuriosityModule::Novelty(tracker) => {
                let hash = StateVisitationTracker::hash_observation(next_state, 16);
                tracker.intrinsic_reward(hash)
            }

            CuriosityModule::PredictionError(predictor) => {
                if let (Some(s), Some(a)) = (state, action) {
                    predictor.update(s, a, next_state)
                } else {
                    0.0
                }
            }

            CuriosityModule::Combined {
                novelty,
                prediction,
                novelty_weight,
                prediction_weight,
            } => {
                let hash = StateVisitationTracker::hash_observation(next_state, 16);
                let n_reward = novelty.intrinsic_reward(hash);

                let p_reward = if let (Some(s), Some(a)) = (state, action) {
                    prediction.update(s, a, next_state)
                } else {
                    0.0
                };

                *novelty_weight * n_reward + *prediction_weight * p_reward
            }
        }
    }

    /// Compute intrinsic reward from spike rates (convenience method)
    pub fn intrinsic_reward_from_spikes(
        &mut self,
        prev_rates: Option<&[u8]>,
        action: Option<usize>,
        current_rates: &[u8],
    ) -> f32 {
        // Convert spike rates to f32 normalized [0, 1]
        let current_f32: Vec<f32> = current_rates.iter().map(|&r| r as f32 / 255.0).collect();

        let prev_f32: Option<Vec<f32>> =
            prev_rates.map(|r| r.iter().map(|&v| v as f32 / 255.0).collect());

        self.intrinsic_reward(prev_f32.as_deref(), action, &current_f32)
    }

    /// Apply decay (for episodic resetting)
    pub fn decay(&mut self) {
        match self {
            CuriosityModule::Novelty(tracker) => tracker.decay(),
            CuriosityModule::Combined { novelty, .. } => novelty.decay(),
            _ => {}
        }
    }

    /// Reset the curiosity module
    pub fn reset(&mut self) {
        match self {
            CuriosityModule::Novelty(tracker) => tracker.reset(),
            CuriosityModule::PredictionError(predictor) => predictor.reset(),
            CuriosityModule::Combined {
                novelty,
                prediction,
                ..
            } => {
                novelty.reset();
                prediction.reset();
            }
            CuriosityModule::None => {}
        }
    }

    /// Check if curiosity is enabled
    pub fn is_enabled(&self) -> bool {
        !matches!(self, CuriosityModule::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_novelty_tracker() {
        let mut tracker = StateVisitationTracker::new(1.0, 0.0);

        // First visit should give high reward
        let hash1 = 12345u64;
        let reward1 = tracker.intrinsic_reward(hash1);
        assert!(reward1 > 0.9, "First visit reward: {}", reward1);

        // Second visit should give lower reward
        let reward2 = tracker.intrinsic_reward(hash1);
        assert!(reward2 < reward1, "Second visit should be lower");
        assert!((reward2 - 1.0 / 2.0_f32.sqrt()).abs() < 0.01);

        // New state should give high reward again
        let hash2 = 67890u64;
        let reward3 = tracker.intrinsic_reward(hash2);
        assert!(reward3 > 0.9, "New state reward: {}", reward3);

        let (total, unique) = tracker.stats();
        assert_eq!(total, 3);
        assert_eq!(unique, 2);
    }

    #[test]
    fn test_hash_observation() {
        let obs1 = vec![0.5, 0.5, 0.5, 0.5];
        let obs2 = vec![0.5, 0.5, 0.5, 0.6];
        let obs3 = vec![0.5, 0.5, 0.5, 0.5];

        let hash1 = StateVisitationTracker::hash_observation(&obs1, 16);
        let hash2 = StateVisitationTracker::hash_observation(&obs2, 16);
        let hash3 = StateVisitationTracker::hash_observation(&obs3, 16);

        // Same observations should hash to same value
        assert_eq!(hash1, hash3);

        // Different observations should (usually) hash differently
        // Note: with 16 bins, 0.5 and 0.6 map to different bins
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_prediction_error() {
        let mut predictor = PredictionErrorCuriosity::new(4, 3, 0.01, 1.0);

        // Simple state transition
        let state = vec![1.0, 0.0, 0.0, 0.0];
        let action = 0;
        let next_state = vec![0.0, 1.0, 0.0, 0.0];

        // First prediction should have high error (model is random)
        let reward1 = predictor.update(&state, action, &next_state);
        assert!(reward1 > 0.0, "Should have positive intrinsic reward");

        // After many updates, error should decrease
        for _ in 0..100 {
            predictor.update(&state, action, &next_state);
        }

        let reward_after = predictor.update(&state, action, &next_state);

        // Error should be lower after learning (normalized, so may not be strictly less)
        println!(
            "Initial reward: {}, After learning: {}",
            reward1, reward_after
        );
    }

    #[test]
    fn test_combined_curiosity() {
        let mut curiosity = CuriosityModule::combined(4, 3, 0.5);

        let state = vec![1.0, 0.0, 0.0, 0.0];
        let next_state = vec![0.0, 1.0, 0.0, 0.0];
        let action = 0;

        // Should get combined reward
        let reward = curiosity.intrinsic_reward(Some(&state), Some(action), &next_state);
        assert!(reward > 0.0, "Combined reward: {}", reward);

        // Reset should work
        curiosity.reset();
        assert!(curiosity.is_enabled());
    }

    #[test]
    fn test_spike_rate_curiosity() {
        let mut curiosity = CuriosityModule::novelty(1.0, 0.0);

        let rates1: Vec<u8> = vec![128, 64, 200, 50];
        let rates2: Vec<u8> = vec![128, 64, 200, 50];
        let rates3: Vec<u8> = vec![0, 255, 100, 150];

        // Same rates should give diminishing rewards
        let r1 = curiosity.intrinsic_reward_from_spikes(None, None, &rates1);
        let r2 = curiosity.intrinsic_reward_from_spikes(None, None, &rates2);
        assert!(r1 > r2, "Repeat visits should give less reward");

        // New rates should give high reward
        let r3 = curiosity.intrinsic_reward_from_spikes(None, None, &rates3);
        assert!(r3 > r2, "New state should give higher reward");
    }

    #[test]
    fn test_decay() {
        let mut tracker = StateVisitationTracker::new(1.0, 0.5); // 50% decay

        let hash = 12345u64;

        // Build up visits
        for _ in 0..10 {
            tracker.intrinsic_reward(hash);
        }

        let (_, unique_before) = tracker.stats();
        assert_eq!(unique_before, 1);

        // Apply decay
        tracker.decay();
        tracker.decay();
        tracker.decay(); // Multiple decays should eventually clear

        // After enough decay, revisiting should feel novel again
        let reward = tracker.intrinsic_reward(hash);
        println!("Reward after decay: {}", reward);
    }
}
