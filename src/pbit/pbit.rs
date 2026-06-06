//! Single P-bit Implementation
//!
//! A p-bit is a probabilistic bit that fluctuates between states 0 and 1
//! according to a sigmoid activation function of its input.

use crate::qec::noise::SimpleRng;

/// A single probabilistic bit
///
/// The p-bit's probability of being in state 1 is:
/// P(m = 1) = sigmoid(I) = 1 / (1 + exp(-I))
///
/// where I is the total input (bias + weighted sum of neighbors)
#[derive(Debug, Clone)]
pub struct PBit {
    /// Current state: 0 or 1
    pub state: u8,
    /// External bias term (h_i in Ising model)
    pub bias: f64,
    /// Cached input from last update
    input: f64,
    /// Cached probability from last update
    probability: f64,
}

impl Default for PBit {
    fn default() -> Self {
        Self::new()
    }
}

impl PBit {
    /// Create a new p-bit initialized to state 0
    pub fn new() -> Self {
        Self {
            state: 0,
            bias: 0.0,
            input: 0.0,
            probability: 0.5,
        }
    }

    /// Create a p-bit with initial bias
    pub fn with_bias(bias: f64) -> Self {
        Self {
            state: 0,
            bias,
            input: bias,
            probability: sigmoid(bias),
        }
    }

    /// Get current state as i8 (-1 or +1) for Ising model
    #[inline]
    pub fn spin(&self) -> i8 {
        2 * (self.state as i8) - 1
    }

    /// Get current state as f64 (-1.0 or +1.0)
    #[inline]
    pub fn spin_f64(&self) -> f64 {
        2.0 * (self.state as f64) - 1.0
    }

    /// Update the p-bit given total input (bias + neighbor contributions)
    ///
    /// Returns true if state changed
    #[inline]
    pub fn update(&mut self, total_input: f64, rng: &mut SimpleRng) -> bool {
        self.input = total_input;
        self.probability = sigmoid(total_input);

        let old_state = self.state;
        self.state = if rng.next_f64() < self.probability {
            1
        } else {
            0
        };

        self.state != old_state
    }

    /// Update with inverse temperature beta (for simulated annealing)
    ///
    /// P(m=1) = sigmoid(beta * I)
    #[inline]
    pub fn update_with_beta(&mut self, total_input: f64, beta: f64, rng: &mut SimpleRng) -> bool {
        self.input = total_input;
        self.probability = sigmoid(beta * total_input);

        let old_state = self.state;
        self.state = if rng.next_f64() < self.probability {
            1
        } else {
            0
        };

        self.state != old_state
    }

    /// Get the probability of being in state 1
    #[inline]
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Get the last computed input
    #[inline]
    pub fn input(&self) -> f64 {
        self.input
    }

    /// Force the p-bit to a specific state
    pub fn set_state(&mut self, state: u8) {
        debug_assert!(state <= 1);
        self.state = state;
    }

    /// Flip the p-bit state
    #[inline]
    pub fn flip(&mut self) {
        self.state = 1 - self.state;
    }
}

/// Sigmoid activation function
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    // Numerically stable sigmoid
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Inverse sigmoid (logit function)
#[inline]
pub fn logit(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0);
    (p / (1.0 - p)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);

        // Symmetry
        assert!((sigmoid(2.0) + sigmoid(-2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pbit_update() {
        let mut rng = SimpleRng::new(42);
        let mut pbit = PBit::new();

        // Strong positive input should usually result in state 1
        let mut ones = 0;
        for _ in 0..1000 {
            pbit.update(5.0, &mut rng);
            ones += pbit.state as u32;
        }
        assert!(ones > 900, "Expected mostly 1s, got {}", ones);

        // Strong negative input should usually result in state 0
        let mut zeros = 0;
        for _ in 0..1000 {
            pbit.update(-5.0, &mut rng);
            zeros += (1 - pbit.state) as u32;
        }
        assert!(zeros > 900, "Expected mostly 0s, got {}", zeros);
    }

    #[test]
    fn test_spin_encoding() {
        let mut pbit = PBit::new();
        pbit.state = 0;
        assert_eq!(pbit.spin(), -1);
        pbit.state = 1;
        assert_eq!(pbit.spin(), 1);
    }

    #[test]
    fn test_beta_scaling() {
        let mut rng = SimpleRng::new(42);
        let mut pbit = PBit::new();

        // High beta (low temperature) should give more deterministic behavior
        pbit.update_with_beta(1.0, 10.0, &mut rng);
        let high_beta_prob = pbit.probability();

        pbit.update_with_beta(1.0, 0.1, &mut rng);
        let low_beta_prob = pbit.probability();

        // High beta should push probability closer to 1
        assert!(high_beta_prob > low_beta_prob);
    }
}
