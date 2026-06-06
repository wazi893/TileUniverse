//! TD-Style Credit Assignment for Quantum-SNN
//!
//! This module provides temporal difference (TD) learning style credit assignment
//! to improve handling of delayed rewards in the Quantum-SNN hybrid.
//!
//! # Key Concepts
//!
//! - **Value Function**: Estimates expected future reward from a state
//! - **TD Error**: δ = r + γV(s') - V(s), the "surprise" in reward
//! - **Eligibility Modulation**: Use TD error instead of raw reward for R-STDP
//!
//! This helps the network learn from delayed rewards by bootstrapping from
//! value estimates rather than waiting for actual rewards.

/// Simple linear value function for TD-style credit assignment.
///
/// Uses a linear model V(s) = w·s + b to estimate state values.
/// Updates via TD(0): w ← w + α·δ·s where δ = r + γV(s') - V(s)
#[derive(Debug, Clone)]
pub struct ValueFunction {
    /// Weights for linear value function
    weights: Vec<f32>,
    /// Bias term
    bias: f32,
    /// Learning rate
    alpha: f32,
    /// Discount factor (0.0-1.0)
    gamma: f32,
    /// State dimension
    state_dim: usize,
    /// Last computed value (for debugging)
    last_value: f32,
    /// Running average of TD errors (for normalization)
    td_error_avg: f32,
    /// Averaging momentum
    momentum: f32,
}

impl ValueFunction {
    /// Create a new value function.
    ///
    /// # Arguments
    /// * `state_dim` - Dimension of state/observation
    /// * `alpha` - Learning rate (typically 0.01-0.1)
    /// * `gamma` - Discount factor (typically 0.95-0.99)
    pub fn new(state_dim: usize, alpha: f32, gamma: f32) -> Self {
        Self {
            weights: vec![0.0; state_dim],
            bias: 0.0,
            alpha,
            gamma,
            state_dim,
            last_value: 0.0,
            td_error_avg: 1.0,
            momentum: 0.99,
        }
    }

    /// Estimate state value V(s).
    pub fn value(&self, state: &[f32]) -> f32 {
        if state.len() != self.state_dim {
            return 0.0;
        }

        let mut v = self.bias;
        for (w, s) in self.weights.iter().zip(state.iter()) {
            v += w * s;
        }
        v
    }

    /// TD(0) update: compute TD error and update weights.
    ///
    /// # Arguments
    /// * `state` - Current state s
    /// * `reward` - Immediate reward r
    /// * `next_state` - Next state s'
    /// * `done` - Whether episode ended (terminal state)
    ///
    /// # Returns
    /// The TD error δ = r + γV(s') - V(s)
    pub fn update(&mut self, state: &[f32], reward: f32, next_state: &[f32], done: bool) -> f32 {
        if state.len() != self.state_dim || next_state.len() != self.state_dim {
            return 0.0;
        }

        // Compute current and next values
        let v = self.value(state);
        let v_next = if done { 0.0 } else { self.value(next_state) };

        // TD error
        let td_error = reward + self.gamma * v_next - v;

        // Update running average
        self.td_error_avg =
            self.momentum * self.td_error_avg + (1.0 - self.momentum) * td_error.abs();

        // Update weights: w ← w + α·δ·s
        for (w, s) in self.weights.iter_mut().zip(state.iter()) {
            *w += self.alpha * td_error * s;
        }
        self.bias += self.alpha * td_error;

        self.last_value = v;
        td_error
    }

    /// Get normalized TD error (for use as R-STDP signal).
    ///
    /// Normalizes by running average to keep signal in reasonable range.
    pub fn normalized_td_error(&self, td_error: f32) -> f32 {
        td_error / self.td_error_avg.max(0.1)
    }

    /// Get the last computed value
    pub fn last_value(&self) -> f32 {
        self.last_value
    }

    /// Get average TD error magnitude
    pub fn average_td_error(&self) -> f32 {
        self.td_error_avg
    }

    /// Reset the value function
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.bias = 0.0;
        self.last_value = 0.0;
        self.td_error_avg = 1.0;
    }

    /// Get current weights (for debugging/visualization)
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}

/// TD-modulated eligibility trace for R-STDP.
///
/// Instead of modulating synaptic changes by raw reward, we use the TD error.
/// This provides a better learning signal for delayed rewards because:
/// - Positive TD error = "better than expected" → strengthen recent actions
/// - Negative TD error = "worse than expected" → weaken recent actions
/// - Zero TD error = "as expected" → no change
#[derive(Debug, Clone)]
pub struct TDEligibilityTrace {
    /// Trace value (0-255)
    pub value: u8,
    /// Decay rate per tick (0.0-1.0)
    pub decay_rate: f32,
}

impl Default for TDEligibilityTrace {
    fn default() -> Self {
        Self {
            value: 0,
            decay_rate: 0.95,
        }
    }
}

impl TDEligibilityTrace {
    /// Create a new trace with custom decay
    pub fn new(decay_rate: f32) -> Self {
        Self {
            value: 0,
            decay_rate: decay_rate.clamp(0.0, 1.0),
        }
    }

    /// Mark this trace (spike occurred)
    pub fn mark(&mut self) {
        self.value = 255;
    }

    /// Decay the trace
    pub fn decay(&mut self) {
        let new_val = (self.value as f32 * self.decay_rate) as u8;
        self.value = new_val;
    }

    /// Get modulated weight update.
    ///
    /// Uses TD error instead of raw reward for better credit assignment.
    pub fn modulated_update(&self, td_error: f32, learning_rate: f32) -> f32 {
        (self.value as f32 / 255.0) * td_error * learning_rate
    }
}

/// Combines value function with eligibility traces for TD-enhanced R-STDP.
#[derive(Debug, Clone)]
pub struct TDCreditAssignment {
    /// Value function for bootstrapping
    value_fn: ValueFunction,
    /// Scale factor for TD error → R-STDP signal
    td_scale: f32,
    /// Previous state (for TD update)
    prev_state: Option<Vec<f32>>,
    /// Last TD error
    last_td_error: f32,
}

impl TDCreditAssignment {
    /// Create a new TD credit assignment module.
    ///
    /// # Arguments
    /// * `state_dim` - Dimension of state/observation
    /// * `alpha` - Value function learning rate
    /// * `gamma` - Discount factor
    /// * `td_scale` - Scale factor for converting TD error to R-STDP signal
    pub fn new(state_dim: usize, alpha: f32, gamma: f32, td_scale: f32) -> Self {
        Self {
            value_fn: ValueFunction::new(state_dim, alpha, gamma),
            td_scale,
            prev_state: None,
            last_td_error: 0.0,
        }
    }

    /// Compute TD-modulated reward signal.
    ///
    /// # Arguments
    /// * `reward` - Immediate reward from environment
    /// * `current_state` - Current state
    /// * `done` - Whether episode ended
    ///
    /// # Returns
    /// TD-modulated reward signal (scaled TD error)
    pub fn compute_signal(&mut self, reward: f32, current_state: &[f32], done: bool) -> f32 {
        let td_error = if let Some(ref prev) = self.prev_state {
            self.value_fn.update(prev, reward, current_state, done)
        } else {
            // First step: use reward directly
            reward
        };

        self.last_td_error = td_error;

        // Store current state for next iteration
        self.prev_state = if done {
            None // Reset on episode end
        } else {
            Some(current_state.to_vec())
        };

        // Return scaled TD error as R-STDP signal
        let normalized = self.value_fn.normalized_td_error(td_error);
        (normalized * self.td_scale).clamp(-127.0, 127.0)
    }

    /// Convert TD signal to i8 for R-STDP
    pub fn compute_signal_i8(&mut self, reward: f32, current_state: &[f32], done: bool) -> i8 {
        self.compute_signal(reward, current_state, done) as i8
    }

    /// Get the last TD error
    pub fn last_td_error(&self) -> f32 {
        self.last_td_error
    }

    /// Get estimated value of a state
    pub fn value(&self, state: &[f32]) -> f32 {
        self.value_fn.value(state)
    }

    /// Reset at episode start
    pub fn reset_episode(&mut self) {
        self.prev_state = None;
        self.last_td_error = 0.0;
    }

    /// Full reset (including learned value function)
    pub fn reset(&mut self) {
        self.value_fn.reset();
        self.prev_state = None;
        self.last_td_error = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_function() {
        let mut vf = ValueFunction::new(4, 0.1, 0.95);

        let state = vec![1.0, 0.0, 0.0, 0.0];
        let next_state = vec![0.0, 1.0, 0.0, 0.0];

        // Initial value should be near 0
        let v0 = vf.value(&state);
        assert!(v0.abs() < 0.01, "Initial value: {}", v0);

        // TD update with positive reward should increase value
        let td_error = vf.update(&state, 1.0, &next_state, false);
        println!("TD error: {}", td_error);

        // After update, value should be higher
        let v1 = vf.value(&state);
        assert!(v1 > v0, "Value should increase: {} -> {}", v0, v1);
    }

    #[test]
    fn test_td_learning_sequence() {
        let mut vf = ValueFunction::new(2, 0.1, 0.9);

        // Simulate a sequence where state [1,0] leads to reward
        let s0 = vec![1.0, 0.0];
        let s1 = vec![0.0, 1.0];

        // Train on many episodes
        for _ in 0..100 {
            // Episode: s0 -> s1 -> terminal with reward
            vf.update(&s0, 0.0, &s1, false); // Step to s1, no reward
            vf.update(&s1, 1.0, &s1, true); // Terminal with reward
        }

        // After training, s0 should have positive value (leads to reward)
        let v_s0 = vf.value(&s0);
        let v_s1 = vf.value(&s1);

        println!("V(s0) = {}, V(s1) = {}", v_s0, v_s1);

        // s1 is closer to reward, should have higher value
        assert!(v_s1 > v_s0, "s1 should have higher value");
        assert!(v_s0 > 0.0, "s0 should have positive value");
    }

    #[test]
    fn test_eligibility_trace() {
        let mut trace = TDEligibilityTrace::new(0.9);

        // Initial trace should be 0
        assert_eq!(trace.value, 0);

        // Mark the trace
        trace.mark();
        assert_eq!(trace.value, 255);

        // Decay should reduce it
        trace.decay();
        assert!(trace.value < 255);

        // Multiple decays should approach 0
        for _ in 0..50 {
            trace.decay();
        }
        assert!(trace.value < 5, "Trace should decay to near 0");
    }

    #[test]
    fn test_td_credit_assignment() {
        let mut tdc = TDCreditAssignment::new(4, 0.1, 0.95, 50.0);

        let s0 = vec![1.0, 0.0, 0.0, 0.0];
        let s1 = vec![0.0, 1.0, 0.0, 0.0];

        // First signal (no previous state)
        let signal1 = tdc.compute_signal(0.5, &s0, false);
        println!("Signal 1: {}", signal1);

        // Second signal (has previous state)
        let signal2 = tdc.compute_signal(1.0, &s1, false);
        println!("Signal 2: {}", signal2);

        // End of episode
        let signal3 = tdc.compute_signal(10.0, &s1, true);
        println!("Signal 3 (terminal): {}", signal3);

        // Reset should clear previous state
        tdc.reset_episode();
        assert!(tdc.prev_state.is_none());
    }

    #[test]
    fn test_delayed_reward_credit() {
        let mut tdc = TDCreditAssignment::new(2, 0.1, 0.9, 50.0);

        let states = vec![
            vec![1.0, 0.0], // s0
            vec![0.5, 0.5], // s1
            vec![0.0, 1.0], // s2 (gets reward)
        ];

        // Train on many episodes with delayed reward
        for _ in 0..100 {
            tdc.reset_episode();
            tdc.compute_signal(0.0, &states[0], false); // s0, no reward
            tdc.compute_signal(0.0, &states[1], false); // s1, no reward
            tdc.compute_signal(1.0, &states[2], true); // s2, reward
        }

        // Check that value propagates back
        let v0 = tdc.value(&states[0]);
        let v1 = tdc.value(&states[1]);
        let v2 = tdc.value(&states[2]);

        println!("V(s0)={}, V(s1)={}, V(s2)={}", v0, v1, v2);

        // Values should decrease as we go further from reward
        assert!(v2 > v1, "s2 should have highest value");
        assert!(v1 > v0, "s1 should have higher value than s0");
        assert!(v0 > 0.0, "s0 should have positive value");
    }
}
