//! Classical feedforward neural network brain for organisms.
//!
//! Architecture: 8 inputs → 8 hidden (ReLU) → 5 outputs (softmax)
//! Parameters: 117 weights (comparable to quantum circuit parameter count)

use crate::quantum::QRng;

/// Movement directions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Up,
    Down,
    Left,
    Right,
    Stay,
}

impl Move {
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Move::Up,
            1 => Move::Down,
            2 => Move::Left,
            3 => Move::Right,
            _ => Move::Stay,
        }
    }

    pub fn to_index(self) -> usize {
        match self {
            Move::Up => 0,
            Move::Down => 1,
            Move::Left => 2,
            Move::Right => 3,
            Move::Stay => 4,
        }
    }
}

/// Classical feedforward neural network
///
/// Architecture matches quantum circuit complexity:
/// - 8 sensor inputs (heat + charge, 4 directions each)
/// - 8 hidden neurons with ReLU activation
/// - 5 output neurons (movement directions) with softmax
/// - Total: 117 parameters vs ~20-50 in quantum circuits
#[derive(Clone, Debug)]
pub struct ClassicalBrain {
    /// Input to hidden weights [hidden][input] = [8][8]
    pub weights_ih: [[f32; 8]; 8],
    /// Hidden layer biases [8]
    pub bias_h: [f32; 8],
    /// Hidden to output weights [output][hidden] = [5][8]
    pub weights_ho: [[f32; 5]; 8],
    /// Output layer biases [5]
    pub bias_o: [f32; 5],
}

impl ClassicalBrain {
    /// Create a new brain with Xavier-initialized random weights
    pub fn random(rng: &mut QRng) -> Self {
        let scale_ih = (2.0 / 8.0_f32).sqrt();
        let scale_ho = (2.0 / 8.0_f32).sqrt();

        Self {
            weights_ih: std::array::from_fn(|_| {
                std::array::from_fn(|_| (rng.next_f32() * 2.0 - 1.0) * scale_ih)
            }),
            bias_h: [0.0; 8],
            weights_ho: std::array::from_fn(|_| {
                std::array::from_fn(|_| (rng.next_f32() * 2.0 - 1.0) * scale_ho)
            }),
            bias_o: [0.0; 5],
        }
    }

    /// Count total parameters (for comparison with quantum circuits)
    pub fn parameter_count(&self) -> usize {
        8 * 8 + 8 + 8 * 5 + 5 // = 117
    }

    /// Forward pass: sensors → movement decision
    ///
    /// # Arguments
    /// * `inputs` - 8 sensor values [heat_up, heat_down, heat_left, heat_right,
    ///                               charge_up, charge_down, charge_left, charge_right]
    /// * `rng` - Random number generator for sampling from softmax
    ///
    /// # Returns
    /// Sampled movement direction
    pub fn forward(&self, inputs: &[f32; 8], rng: &mut QRng) -> Move {
        // Hidden layer with ReLU activation
        let mut hidden = [0.0_f32; 8];
        for i in 0..8 {
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
            for j in 0..8 {
                sum += hidden[j] * self.weights_ho[j][i];
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

    /// Forward pass returning probabilities (for analysis)
    pub fn forward_probs(&self, inputs: &[f32; 8]) -> [f32; 5] {
        // Hidden layer with ReLU activation
        let mut hidden = [0.0_f32; 8];
        for i in 0..8 {
            let mut sum = self.bias_h[i];
            for j in 0..8 {
                sum += inputs[j] * self.weights_ih[i][j];
            }
            hidden[i] = sum.max(0.0);
        }

        // Output layer (logits)
        let mut logits = [0.0_f32; 5];
        for i in 0..5 {
            let mut sum = self.bias_o[i];
            for j in 0..8 {
                sum += hidden[j] * self.weights_ho[j][i];
            }
            logits[i] = sum;
        }

        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = [0.0_f32; 5];
        let mut exp_sum = 0.0_f32;
        for i in 0..5 {
            probs[i] = (logits[i] - max_logit).exp();
            exp_sum += probs[i];
        }
        for i in 0..5 {
            probs[i] /= exp_sum;
        }
        probs
    }

    /// Mutate the brain (evolutionary operator)
    ///
    /// Uses Gaussian perturbation on weights, similar to how quantum
    /// circuits mutate gate parameters.
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `mutation_rate` - Probability of mutating each weight (0.0-1.0)
    /// * `mutation_scale` - Standard deviation of Gaussian perturbation
    pub fn mutate(&mut self, rng: &mut QRng, mutation_rate: f32, mutation_scale: f32) {
        // Mutate input→hidden weights
        for row in &mut self.weights_ih {
            for w in row {
                if rng.next_f32() < mutation_rate {
                    *w += (rng.next_f32() * 2.0 - 1.0) * mutation_scale;
                }
            }
        }

        // Mutate hidden biases
        for b in &mut self.bias_h {
            if rng.next_f32() < mutation_rate {
                *b += (rng.next_f32() * 2.0 - 1.0) * mutation_scale;
            }
        }

        // Mutate hidden→output weights
        for row in &mut self.weights_ho {
            for w in row {
                if rng.next_f32() < mutation_rate {
                    *w += (rng.next_f32() * 2.0 - 1.0) * mutation_scale;
                }
            }
        }

        // Mutate output biases
        for b in &mut self.bias_o {
            if rng.next_f32() < mutation_rate {
                *b += (rng.next_f32() * 2.0 - 1.0) * mutation_scale;
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
    fn test_brain_creation() {
        let mut rng = QRng::new(42);
        let brain = ClassicalBrain::random(&mut rng);
        assert_eq!(brain.parameter_count(), 117);
    }

    #[test]
    fn test_forward_deterministic() {
        let mut rng = QRng::new(42);
        let brain = ClassicalBrain::random(&mut rng);
        let inputs = [0.5, 0.3, 0.8, 0.1, 0.2, 0.4, 0.6, 0.7];

        // Same seed should give same output
        let mut rng1 = QRng::new(123);
        let mut rng2 = QRng::new(123);

        let move1 = brain.forward(&inputs, &mut rng1);
        let move2 = brain.forward(&inputs, &mut rng2);

        assert_eq!(move1, move2);
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let mut rng = QRng::new(42);
        let brain = ClassicalBrain::random(&mut rng);
        let inputs = [0.5, 0.3, 0.8, 0.1, 0.2, 0.4, 0.6, 0.7];

        let probs = brain.forward_probs(&inputs);
        let sum: f32 = probs.iter().sum();

        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mutation() {
        let mut rng = QRng::new(42);
        let brain = ClassicalBrain::random(&mut rng);
        let mut child = brain.clone();

        child.mutate(&mut rng, 1.0, 0.1); // 100% mutation rate

        // At least some weights should have changed
        let mut different = false;
        for i in 0..8 {
            for j in 0..8 {
                if (brain.weights_ih[i][j] - child.weights_ih[i][j]).abs() > 1e-10 {
                    different = true;
                    break;
                }
            }
        }
        assert!(different);
    }

    #[test]
    fn test_output_distribution() {
        let mut rng = QRng::new(42);
        let brain = ClassicalBrain::random(&mut rng);
        let inputs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Strong up signal

        // Run many trials and check distribution
        let mut counts = [0u32; 5];
        for _ in 0..1000 {
            let m = brain.forward(&inputs, &mut rng);
            counts[m.to_index()] += 1;
        }

        // Should have some variety (not all same move)
        let nonzero = counts.iter().filter(|&&c| c > 0).count();
        assert!(
            nonzero >= 2,
            "Expected variety in outputs, got {:?}",
            counts
        );
    }
}
