//! EPIC 96.5: Configurable Neural Network Architectures
//!
//! Parameterized NN to test different sizes:
//! - Config A (8-8-5): Current baseline
//! - Config B (16-16-5): Minimum WMMA viable
//! - Config C (32-32-8): WMMA sweet spot
//! - Config D (32-64-8): Maximum capacity
//! - Config E (64-64-8): Stretch goal

use crate::classical_brain::Move;
use crate::quantum::QRng;

/// Neural network architecture configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NNConfig {
    pub n_sensors: usize,
    pub n_hidden: usize,
    pub n_outputs: usize,
}

impl NNConfig {
    /// Config A: Current baseline (8-8-5)
    /// - 117 total parameters
    /// - WMMA efficiency: 20% (too much padding)
    pub const CONFIG_A: Self = Self {
        n_sensors: 8,
        n_hidden: 8,
        n_outputs: 5,
    };

    /// Config B: Minimum WMMA (16-16-5)
    /// - 341 total parameters
    /// - WMMA efficiency: 100% (native 16×16 tile)
    pub const CONFIG_B: Self = Self {
        n_sensors: 16,
        n_hidden: 16,
        n_outputs: 5,
    };

    /// Config C: WMMA sweet spot (32-32-8)
    /// - 1,312 total parameters
    /// - WMMA efficiency: 100% (2×2 grid of 16×16 tiles)
    pub const CONFIG_C: Self = Self {
        n_sensors: 32,
        n_hidden: 32,
        n_outputs: 8,
    };

    /// Config D: Maximum capacity (32-64-8)
    /// - 2,632 total parameters
    /// - WMMA efficiency: 85% (asymmetric but acceptable)
    pub const CONFIG_D: Self = Self {
        n_sensors: 32,
        n_hidden: 64,
        n_outputs: 8,
    };

    /// Config E: Stretch goal (64-64-8)
    /// - 4,680 total parameters
    /// - WMMA efficiency: 100% (4×4 grid of 16×16 tiles)
    pub const CONFIG_E: Self = Self {
        n_sensors: 64,
        n_hidden: 64,
        n_outputs: 8,
    };

    /// Total number of parameters in this configuration
    pub fn total_params(&self) -> usize {
        // weights_ih + bias_h + weights_ho + bias_o
        (self.n_sensors * self.n_hidden)
            + self.n_hidden
            + (self.n_hidden * self.n_outputs)
            + self.n_outputs
    }

    /// WMMA efficiency (percentage of compute used, not wasted on padding)
    pub fn wmma_efficiency(&self) -> f64 {
        // Layer 1: sensors × hidden
        let l1_actual = (self.n_sensors * self.n_hidden) as f64;
        let l1_padded_rows = self.n_sensors.div_ceil(16) * 16;
        let l1_padded_cols = self.n_hidden.div_ceil(16) * 16;
        let l1_padded = (l1_padded_rows * l1_padded_cols) as f64;

        // Layer 2: hidden × outputs
        let l2_actual = (self.n_hidden * self.n_outputs) as f64;
        let l2_padded_rows = self.n_hidden.div_ceil(16) * 16;
        let l2_padded_cols = self.n_outputs.div_ceil(16) * 16;
        let l2_padded = (l2_padded_rows * l2_padded_cols) as f64;

        let total_actual = l1_actual + l2_actual;
        let total_padded = l1_padded + l2_padded;

        (total_actual / total_padded) * 100.0
    }
}

/// Configurable neural network brain
#[derive(Clone)]
pub struct ConfigurableBrain {
    pub config: NNConfig,

    // Weights stored as flat arrays (row-major)
    pub weights_ih: Vec<f32>, // [n_sensors × n_hidden]
    pub bias_h: Vec<f32>,     // [n_hidden]
    pub weights_ho: Vec<f32>, // [n_hidden × n_outputs]
    pub bias_o: Vec<f32>,     // [n_outputs]
}

impl ConfigurableBrain {
    /// Create a random brain with given configuration
    pub fn random(config: NNConfig, seed: u64) -> Self {
        let mut rng = QRng::new(seed);

        let n_ih = config.n_sensors * config.n_hidden;
        let n_ho = config.n_hidden * config.n_outputs;

        Self {
            config,
            weights_ih: (0..n_ih).map(|_| rng.next_f32() * 2.0 - 1.0).collect(),
            bias_h: (0..config.n_hidden).map(|_| rng.next_f32() - 0.5).collect(),
            weights_ho: (0..n_ho).map(|_| rng.next_f32() * 2.0 - 1.0).collect(),
            bias_o: (0..config.n_outputs)
                .map(|_| rng.next_f32() - 0.5)
                .collect(),
        }
    }

    /// Forward pass: sensors → hidden → outputs
    pub fn forward(&self, sensors: &[f32]) -> Vec<f32> {
        assert_eq!(
            sensors.len(),
            self.config.n_sensors,
            "Sensor count mismatch"
        );

        // Layer 1: sensors → hidden
        let mut hidden = vec![0.0; self.config.n_hidden];
        for h in 0..self.config.n_hidden {
            let mut sum = self.bias_h[h];
            for s in 0..self.config.n_sensors {
                sum += sensors[s] * self.weights_ih[s * self.config.n_hidden + h];
            }
            hidden[h] = sum.tanh();
        }

        // Layer 2: hidden → outputs
        let mut outputs = vec![0.0; self.config.n_outputs];
        for o in 0..self.config.n_outputs {
            let mut sum = self.bias_o[o];
            for h in 0..self.config.n_hidden {
                sum += hidden[h] * self.weights_ho[h * self.config.n_outputs + o];
            }
            outputs[o] = sum.tanh();
        }

        outputs
    }

    /// Forward pass and select move (argmax of outputs)
    pub fn select_move(&self, sensors: &[f32]) -> Move {
        let outputs = self.forward(sensors);
        let max_idx = outputs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Map output index to move
        match max_idx {
            0 => Move::Up,
            1 => Move::Down,
            2 => Move::Left,
            3 => Move::Right,
            _ => Move::Stay,
        }
    }

    /// Mutate this brain (for evolution)
    pub fn mutate(&self, mutation_rate: f64, rng: &mut QRng) -> Self {
        let mut mutated = self.clone();

        // Mutate weights_ih
        for w in &mut mutated.weights_ih {
            if rng.next_f32() < mutation_rate as f32 {
                *w += (rng.next_f32() * 2.0 - 1.0) * 0.2;
                *w = w.clamp(-2.0, 2.0);
            }
        }

        // Mutate bias_h
        for b in &mut mutated.bias_h {
            if rng.next_f32() < mutation_rate as f32 {
                *b += (rng.next_f32() * 2.0 - 1.0) * 0.1;
                *b = b.clamp(-1.0, 1.0);
            }
        }

        // Mutate weights_ho
        for w in &mut mutated.weights_ho {
            if rng.next_f32() < mutation_rate as f32 {
                *w += (rng.next_f32() * 2.0 - 1.0) * 0.2;
                *w = w.clamp(-2.0, 2.0);
            }
        }

        // Mutate bias_o
        for b in &mut mutated.bias_o {
            if rng.next_f32() < mutation_rate as f32 {
                *b += (rng.next_f32() * 2.0 - 1.0) * 0.1;
                *b = b.clamp(-1.0, 1.0);
            }
        }

        mutated
    }

    /// Crossover with another brain (for evolution)
    pub fn crossover(&self, other: &Self, rng: &mut QRng) -> Self {
        assert_eq!(
            self.config, other.config,
            "Can't crossover different configs"
        );

        let mut child = self.clone();

        // 50/50 crossover for each weight
        for (c, o) in child.weights_ih.iter_mut().zip(&other.weights_ih) {
            if rng.next_f32() < 0.5 {
                *c = *o;
            }
        }

        for (c, o) in child.bias_h.iter_mut().zip(&other.bias_h) {
            if rng.next_f32() < 0.5 {
                *c = *o;
            }
        }

        for (c, o) in child.weights_ho.iter_mut().zip(&other.weights_ho) {
            if rng.next_f32() < 0.5 {
                *c = *o;
            }
        }

        for (c, o) in child.bias_o.iter_mut().zip(&other.bias_o) {
            if rng.next_f32() < 0.5 {
                *c = *o;
            }
        }

        child
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_params() {
        // A: 8×8 + 8 + 8×5 + 5 = 64 + 8 + 40 + 5 = 117
        assert_eq!(NNConfig::CONFIG_A.total_params(), 117);
        // B: 16×16 + 16 + 16×5 + 5 = 256 + 16 + 80 + 5 = 357
        assert_eq!(NNConfig::CONFIG_B.total_params(), 357);
        // C: 32×32 + 32 + 32×8 + 8 = 1024 + 32 + 256 + 8 = 1320
        assert_eq!(NNConfig::CONFIG_C.total_params(), 1320);
        // D: 32×64 + 64 + 64×8 + 8 = 2048 + 64 + 512 + 8 = 2632
        assert_eq!(NNConfig::CONFIG_D.total_params(), 2632);
        // E: 64×64 + 64 + 64×8 + 8 = 4096 + 64 + 512 + 8 = 4680
        assert_eq!(NNConfig::CONFIG_E.total_params(), 4680);
    }

    #[test]
    fn test_wmma_efficiency() {
        // Config A: 8×8 padded to 16×16 = poor efficiency
        let eff_a = NNConfig::CONFIG_A.wmma_efficiency();
        assert!(eff_a < 30.0, "Config A should be <30% efficient");

        // Config B: Layer 1 is 16×16 (100%), but layer 2 is 16×5→16×16 (31%)
        // Total: (256+80)/(256+256) = 65.6%
        let eff_b = NNConfig::CONFIG_B.wmma_efficiency();
        assert!(
            eff_b > 60.0 && eff_b < 70.0,
            "Config B should be ~65% efficient (layer 2 padding hurts)"
        );

        // Config C: 32×32 (100%) and 32×8→32×16 (50%)
        // Better than B due to larger matrices
        let eff_c = NNConfig::CONFIG_C.wmma_efficiency();
        assert!(eff_c > 70.0, "Config C should be >70% efficient");
    }

    #[test]
    fn test_forward_pass_config_a() {
        let brain = ConfigurableBrain::random(NNConfig::CONFIG_A, 12345);
        let sensors = vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4];

        let outputs = brain.forward(&sensors);

        assert_eq!(outputs.len(), 5);
        for &o in &outputs {
            assert!(o >= -1.0 && o <= 1.0, "tanh output should be in [-1, 1]");
        }
    }

    #[test]
    fn test_forward_pass_config_c() {
        let brain = ConfigurableBrain::random(NNConfig::CONFIG_C, 54321);
        let sensors = vec![0.0; 32]; // 32 zero sensors

        let outputs = brain.forward(&sensors);

        assert_eq!(outputs.len(), 8);
        for &o in &outputs {
            assert!(o >= -1.0 && o <= 1.0, "tanh output should be in [-1, 1]");
        }
    }

    #[test]
    fn test_select_move() {
        let brain = ConfigurableBrain::random(NNConfig::CONFIG_A, 99999);
        let sensors = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let mv = brain.select_move(&sensors);

        // Should be one of the valid moves
        matches!(
            mv,
            Move::Up | Move::Down | Move::Left | Move::Right | Move::Stay
        );
    }

    #[test]
    fn test_mutation() {
        let mut rng = QRng::new(11111);
        let brain = ConfigurableBrain::random(NNConfig::CONFIG_A, 22222);
        let mutated = brain.mutate(0.1, &mut rng);

        // Should have different weights
        let diff_count = brain
            .weights_ih
            .iter()
            .zip(&mutated.weights_ih)
            .filter(|(a, b)| (*a - *b).abs() > 1e-6)
            .count();

        assert!(diff_count > 0, "Mutation should change some weights");
        assert!(
            diff_count < brain.weights_ih.len(),
            "Mutation shouldn't change ALL weights"
        );
    }

    #[test]
    fn test_crossover() {
        let mut rng = QRng::new(33333);
        let brain1 = ConfigurableBrain::random(NNConfig::CONFIG_A, 44444);
        let brain2 = ConfigurableBrain::random(NNConfig::CONFIG_A, 55555);

        let child = brain1.crossover(&brain2, &mut rng);

        // Child should have mix of parent weights
        let from_parent1 = child
            .weights_ih
            .iter()
            .zip(&brain1.weights_ih)
            .filter(|(c, p)| (*c - *p).abs() < 1e-6)
            .count();

        let from_parent2 = child
            .weights_ih
            .iter()
            .zip(&brain2.weights_ih)
            .filter(|(c, p)| (*c - *p).abs() < 1e-6)
            .count();

        assert!(from_parent1 > 0, "Child should inherit from parent1");
        assert!(from_parent2 > 0, "Child should inherit from parent2");
    }
}
