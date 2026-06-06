//! Handicapped classical neural network - matched to quantum circuit constraints.
//!
//! EPIC 91.5: Fair comparison by crippling classical NN to match quantum disadvantages:
//! - Reduced parameter count (61 vs 117)
//! - Poor initialization (uniform [-1,1] vs Xavier)
//! - Discrete mutations (quantize to 0.1 vs smooth Gaussian)

use crate::classical_brain::Move;
use crate::quantum::QRng;

/// Handicapped classical feedforward neural network.
///
/// Architecture: 8 inputs → 4 hidden (ReLU) → 5 outputs (softmax)
/// Parameters: 61 (closer to quantum's ~50)
#[derive(Clone, Debug)]
pub struct HandicappedClassicalBrain {
    /// Input to hidden weights [hidden][input] = [4][8]
    pub weights_ih: [[f32; 8]; 4],
    /// Hidden layer biases [4]
    pub bias_h: [f32; 4],
    /// Hidden to output weights [output][hidden] = [5][4]
    pub weights_ho: [[f32; 4]; 5],
    /// Output layer biases [5]
    pub bias_o: [f32; 5],
}

impl HandicappedClassicalBrain {
    /// Create a new brain with POOR random initialization (uniform [-1, 1]).
    pub fn random(rng: &mut QRng) -> Self {
        Self {
            weights_ih: std::array::from_fn(|_| {
                std::array::from_fn(|_| rng.next_f32() * 2.0 - 1.0)
            }), // Uniform [-1, 1]
            bias_h: [0.0; 4],
            weights_ho: std::array::from_fn(|_| {
                std::array::from_fn(|_| rng.next_f32() * 2.0 - 1.0)
            }),
            bias_o: [0.0; 5],
        }
    }

    /// Count total parameters (for comparison with quantum circuits)
    pub fn parameter_count(&self) -> usize {
        8 * 4 + 4 + 4 * 5 + 5 // = 61
    }

    /// Forward pass: sensors → movement decision
    pub fn forward(&self, inputs: &[f32; 8], rng: &mut QRng) -> Move {
        // Hidden layer with ReLU activation
        let mut hidden = [0.0_f32; 4];
        for i in 0..4 {
            let mut sum = self.bias_h[i];
            for j in 0..8 {
                sum += inputs[j] * self.weights_ih[i][j];
            }
            hidden[i] = sum.max(0.0); // ReLU
        }

        // Output layer (logits)
        let mut logits = [0.0_f32; 5];
        for i in 0..5 {
            let mut sum = self.bias_o[i];
            for j in 0..4 {
                sum += hidden[j] * self.weights_ho[i][j];
            }
            logits[i] = sum;
        }

        // Softmax with numerical stability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_vals = [0.0_f32; 5];
        let mut exp_sum = 0.0_f32;
        for i in 0..5 {
            exp_vals[i] = (logits[i] - max_logit).exp();
            exp_sum += exp_vals[i];
        }

        // Sample from probability distribution
        let r = rng.next_f32();
        let mut cumsum = 0.0;
        for i in 0..5 {
            cumsum += exp_vals[i] / exp_sum;
            if r < cumsum {
                return Move::from_index(i);
            }
        }
        Move::Stay
    }

    /// Mutate with DISCRETE quantization (like quantum gate swaps).
    ///
    /// Each mutation:
    /// 1. Add Gaussian perturbation
    /// 2. Quantize to nearest 0.1 (simulates discrete gate space)
    pub fn mutate(&mut self, rng: &mut QRng, mutation_rate: f32, mutation_scale: f32) {
        // Mutate input→hidden weights
        for row in &mut self.weights_ih {
            for w in row {
                if rng.next_f32() < mutation_rate {
                    *w += (rng.next_f32() * 2.0 - 1.0) * mutation_scale;
                    *w = (*w * 10.0).round() / 10.0; // Quantize to 0.1
                }
            }
        }

        // Mutate hidden biases
        for b in &mut self.bias_h {
            if rng.next_f32() < mutation_rate {
                *b += (rng.next_f32() * 2.0 - 1.0) * mutation_scale;
                *b = (*b * 10.0).round() / 10.0;
            }
        }

        // Mutate hidden→output weights
        for row in &mut self.weights_ho {
            for w in row {
                if rng.next_f32() < mutation_rate {
                    *w += (rng.next_f32() * 2.0 - 1.0) * mutation_scale;
                    *w = (*w * 10.0).round() / 10.0;
                }
            }
        }

        // Mutate output biases
        for b in &mut self.bias_o {
            if rng.next_f32() < mutation_rate {
                *b += (rng.next_f32() * 2.0 - 1.0) * mutation_scale;
                *b = (*b * 10.0).round() / 10.0;
            }
        }
    }

    /// Create offspring with mutation
    pub fn reproduce(&self, rng: &mut QRng, mutation_rate: f32, mutation_scale: f32) -> Self {
        let mut child = self.clone();
        child.mutate(rng, mutation_rate, mutation_scale);
        child
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_count() {
        let mut rng = QRng::new(42);
        let brain = HandicappedClassicalBrain::random(&mut rng);
        assert_eq!(brain.parameter_count(), 61);
    }

    #[test]
    fn test_quantization() {
        let mut rng = QRng::new(42);
        let mut brain = HandicappedClassicalBrain::random(&mut rng);

        brain.mutate(&mut rng, 1.0, 0.5);

        // Check that all weights are quantized to 0.1
        for row in &brain.weights_ih {
            for &w in row {
                assert_eq!(w, (w * 10.0).round() / 10.0, "Weight not quantized: {}", w);
            }
        }
    }
}
