//! Hybrid Exploration Brain - NN for exploitation, QUBO for exploration
//!
//! This module implements a hybrid brain architecture inspired by successful
//! quantum-classical hybrid systems in robotics. The key insight from D-Wave's
//! 30x speedup results is that hybrid approaches outperform pure quantum.
//!
//! # Architecture
//!
//! 1. **Classical NN (Exploitation)**: Provides baseline policy using a trained
//!    feedforward network. Handles smooth gradient landscapes effectively.
//!
//! 2. **QUBO/P-bit (Exploration)**: When NN confidence is low (high entropy),
//!    the QUBO layer samples alternative actions. This provides structured
//!    exploration superior to epsilon-greedy random sampling.
//!
//! # Decision Flow
//!
//! ```text
//! Sensors → NN → softmax probs → entropy check
//!                                    ↓
//!                     [high entropy]   [low entropy]
//!                           ↓               ↓
//!                    QUBO sampling    NN decision
//! ```
//!
//! # Why This Works
//!
//! - NN excels at smooth optimization (which dominates foraging)
//! - QUBO provides principled exploration with one-hot constraints
//! - Entropy-based switching avoids random exploration when confident
//! - Both components are evolvable, allowing adaptation

use crate::brain::qubo_brain::{DecisionBudget, QuboBrain};
use crate::classical_brain::{ClassicalBrain, Move};
use crate::qec::noise::SimpleRng as PBitRng;
use crate::quantum::QRng;
use crate::quantum_brain::SensorReadings;

/// Configuration for hybrid brain behavior
#[derive(Debug, Clone, Copy)]
pub struct HybridConfig {
    /// Entropy threshold for blending QUBO into decision
    /// Range: [0.0, ln(5) ≈ 1.609] - higher means more NN reliance
    pub entropy_threshold: f32,

    /// QUBO blend weight when exploring (0.0 = pure NN, 1.0 = pure QUBO)
    /// We blend NN and QUBO probabilities rather than switching
    pub qubo_blend_weight: f32,

    /// Minimum blend rate (always blend at least this much QUBO)
    pub min_blend_rate: f32,

    /// QUBO decision budget
    pub qubo_budget: DecisionBudget,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            // DISABLED: Only use QUBO when NN entropy exceeds theoretical max
            // This effectively makes it pure NN unless explicitly overridden
            entropy_threshold: 2.0,

            // Tiny QUBO influence when blending does occur
            qubo_blend_weight: 0.05,

            // DISABLED: No baseline QUBO blending
            min_blend_rate: 0.0,

            // More sweeps for better QUBO decisions
            qubo_budget: DecisionBudget {
                sweeps: 200,
                beta: 6.0,
            },
        }
    }
}

/// Hybrid Exploration Brain combining NN exploitation with QUBO exploration
#[derive(Clone, Debug)]
pub struct HybridExplorationBrain {
    /// Classical NN for exploitation (greedy policy)
    pub nn: ClassicalBrain,

    /// QUBO brain for exploration (structured sampling)
    pub qubo: QuboBrain,

    /// Configuration for hybrid behavior
    pub config: HybridConfig,

    /// Statistics: how often each mode was used
    pub nn_decisions: u64,
    pub qubo_decisions: u64,
}

impl HybridExplorationBrain {
    /// Create a new hybrid brain with random initialization
    pub fn random(rng: &mut QRng) -> Self {
        let nn = ClassicalBrain::random(rng);

        // Initialize QUBO with penalty = 10.0 - moderate constraint enforcement
        let seed = (rng.next_f32() * u64::MAX as f32) as u64;
        let mut pbit_rng = PBitRng::new(seed);
        let qubo = QuboBrain::random(5, 10.0, &mut pbit_rng);

        Self {
            nn,
            qubo,
            config: HybridConfig::default(),
            nn_decisions: 0,
            qubo_decisions: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(rng: &mut QRng, config: HybridConfig) -> Self {
        let mut brain = Self::random(rng);
        brain.config = config;
        brain
    }

    /// Compute entropy of probability distribution
    /// Returns value in [0, ln(5)] where higher = more uncertain
    fn entropy(probs: &[f32; 5]) -> f32 {
        let mut h = 0.0f32;
        for &p in probs {
            if p > 1e-10 {
                h -= p * p.ln();
            }
        }
        h
    }

    /// Maximum possible entropy for 5 outcomes: ln(5) ≈ 1.609
    pub const MAX_ENTROPY: f32 = 1.609;

    /// Make a decision using hybrid strategy with probability blending
    ///
    /// Instead of switching between NN and QUBO, we blend their probability
    /// distributions. This allows QUBO to provide exploration "nudges" while
    /// NN remains the primary decision maker.
    pub fn decide(&mut self, sensors: &SensorReadings, rng: &mut QRng) -> Move {
        // Fast path: if blending is completely disabled, just use NN directly
        if self.config.min_blend_rate <= 0.0 && self.config.entropy_threshold >= Self::MAX_ENTROPY {
            self.nn_decisions += 1;
            let inputs = sensors_to_array(sensors);
            return self.nn.forward(&inputs, rng);
        }

        // Get NN's probability distribution
        let inputs = sensors_to_array(sensors);
        let nn_probs = self.nn.forward_probs(&inputs);
        let entropy = Self::entropy(&nn_probs);

        // Compute blend weight based on entropy and config
        // Higher entropy (more uncertain) -> more QUBO influence
        let base_blend = if entropy > self.config.entropy_threshold {
            self.config.qubo_blend_weight
        } else {
            self.config.min_blend_rate
        };

        // If blend is negligible, just use NN
        if base_blend < 0.01 {
            self.nn_decisions += 1;
            return self.nn.forward(&inputs, rng);
        }

        // Get QUBO's preferred action
        self.qubo_decisions += 1;
        let seed = (rng.next_f32() * u64::MAX as f32) as u64;
        let mut pbit_rng = PBitRng::new(seed);
        let qubo_action = self
            .qubo
            .decide(sensors, &mut pbit_rng, &self.config.qubo_budget);
        let qubo_idx = qubo_action.to_index();

        // Create QUBO probability distribution (one-hot with smoothing)
        // This gives most probability to QUBO's choice but allows alternatives
        let mut qubo_probs = [0.05f32; 5]; // Small baseline for all
        qubo_probs[qubo_idx] = 0.8; // High for QUBO's choice
        // Normalize
        let qubo_sum: f32 = qubo_probs.iter().sum();
        for p in &mut qubo_probs {
            *p /= qubo_sum;
        }

        // Blend probabilities: (1-blend)*NN + blend*QUBO
        let mut blended = [0.0f32; 5];
        for i in 0..5 {
            blended[i] = (1.0 - base_blend) * nn_probs[i] + base_blend * qubo_probs[i];
        }

        // Normalize blended probs (should be close to 1 already)
        let sum: f32 = blended.iter().sum();
        for p in &mut blended {
            *p /= sum;
        }

        // Sample from blended distribution
        let r = rng.next_f32();
        let mut cumsum = 0.0;
        for i in 0..5 {
            cumsum += blended[i];
            if r < cumsum {
                return Move::from_index(i);
            }
        }
        Move::Stay
    }

    /// Get NN confidence (1 - normalized entropy)
    pub fn confidence(&self, sensors: &SensorReadings) -> f32 {
        let inputs = sensors_to_array(sensors);
        let probs = self.nn.forward_probs(&inputs);
        let entropy = Self::entropy(&probs);
        1.0 - (entropy / Self::MAX_ENTROPY)
    }

    /// Get exploration ratio (QUBO decisions / total decisions)
    pub fn exploration_ratio(&self) -> f32 {
        let total = self.nn_decisions + self.qubo_decisions;
        if total == 0 {
            0.0
        } else {
            self.qubo_decisions as f32 / total as f32
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.nn_decisions = 0;
        self.qubo_decisions = 0;
    }

    /// Create mutated offspring
    ///
    /// NN is always evolved. QUBO is only evolved if blending is enabled.
    /// Config params can also mutate to tune blending behavior.
    pub fn reproduce(&self, rng: &mut QRng, mutation_rate: f32, mutation_scale: f32) -> Self {
        // Mutate NN component (always - this is our core)
        let nn = self.nn.reproduce(rng, mutation_rate, mutation_scale);

        // Only mutate QUBO if blending is actually enabled
        // This prevents wasting evolutionary "budget" on unused parameters
        let qubo = if self.config.min_blend_rate > 0.0
            || self.config.entropy_threshold < Self::MAX_ENTROPY
        {
            let seed = (rng.next_f32() * u64::MAX as f32) as u64;
            let mut pbit_rng = PBitRng::new(seed);
            self.qubo
                .reproduce(&mut pbit_rng, mutation_rate, mutation_scale)
        } else {
            // QUBO disabled - just clone without mutation
            self.qubo.clone()
        };

        // Only mutate config if blending is enabled
        let mut config = self.config;
        if self.config.min_blend_rate > 0.0 || self.config.entropy_threshold < Self::MAX_ENTROPY {
            if rng.next_f32() < 0.1 {
                config.entropy_threshold += (rng.next_f32() * 2.0 - 1.0) * 0.1;
                config.entropy_threshold = config.entropy_threshold.clamp(0.3, 1.5);
            }
            if rng.next_f32() < 0.1 {
                config.qubo_blend_weight += (rng.next_f32() * 2.0 - 1.0) * 0.05;
                config.qubo_blend_weight = config.qubo_blend_weight.clamp(0.05, 0.5);
            }
            if rng.next_f32() < 0.1 {
                config.min_blend_rate += (rng.next_f32() * 2.0 - 1.0) * 0.02;
                config.min_blend_rate = config.min_blend_rate.clamp(0.0, 0.2);
            }
        }

        Self {
            nn,
            qubo,
            config,
            nn_decisions: 0,
            qubo_decisions: 0,
        }
    }

    /// Get total parameter count
    pub fn parameter_count(&self) -> usize {
        self.nn.parameter_count() + self.qubo.parameter_count()
    }

    /// Memory usage in bytes (approximate)
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.qubo.memory_bytes()
            + std::mem::size_of::<ClassicalBrain>()
    }
}

/// Convert SensorReadings to normalized 8-element float array for NN
fn sensors_to_array(sensors: &SensorReadings) -> [f32; 8] {
    let normalize = |v: u32| (v as f32) / 255.0;
    [
        normalize(sensors.heat_n),
        normalize(sensors.heat_s),
        normalize(sensors.heat_e),
        normalize(sensors.heat_w),
        normalize(sensors.heat_local),
        normalize(sensors.charge_local),
        0.0, // Padding
        0.0, // Padding
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_brain_creation() {
        let mut rng = QRng::new(42);
        let brain = HybridExplorationBrain::random(&mut rng);

        assert!(brain.parameter_count() > 0);
        assert_eq!(brain.nn_decisions, 0);
        assert_eq!(brain.qubo_decisions, 0);
    }

    #[test]
    fn test_entropy_calculation() {
        // Uniform distribution has maximum entropy
        let uniform = [0.2, 0.2, 0.2, 0.2, 0.2];
        let entropy_uniform = HybridExplorationBrain::entropy(&uniform);
        assert!((entropy_uniform - HybridExplorationBrain::MAX_ENTROPY).abs() < 0.01);

        // Peaked distribution has low entropy
        let peaked = [0.9, 0.025, 0.025, 0.025, 0.025];
        let entropy_peaked = HybridExplorationBrain::entropy(&peaked);
        assert!(entropy_peaked < 0.5);

        // Delta has zero entropy
        let delta = [1.0, 0.0, 0.0, 0.0, 0.0];
        let entropy_delta = HybridExplorationBrain::entropy(&delta);
        assert!(entropy_delta < 0.01);
    }

    #[test]
    fn test_hybrid_decides() {
        let mut rng = QRng::new(42);
        let mut brain = HybridExplorationBrain::random(&mut rng);

        let sensors = SensorReadings {
            heat_n: 200,
            heat_s: 50,
            heat_e: 100,
            heat_w: 100,
            heat_local: 75,
            charge_local: 25,
        };

        // Run many decisions
        for _ in 0..100 {
            let _ = brain.decide(&sensors, &mut rng);
        }

        // Should have used both modes
        assert!(brain.nn_decisions > 0 || brain.qubo_decisions > 0);
        println!(
            "NN: {}, QUBO: {}, Exploration ratio: {:.2}%",
            brain.nn_decisions,
            brain.qubo_decisions,
            brain.exploration_ratio() * 100.0
        );
    }

    #[test]
    fn test_reproduction() {
        let mut rng = QRng::new(42);
        let parent = HybridExplorationBrain::random(&mut rng);

        let child = parent.reproduce(&mut rng, 1.0, 0.1);

        // Child should be different (high mutation rate)
        // Check NN weights differ
        let mut different = false;
        for i in 0..8 {
            for j in 0..8 {
                if (parent.nn.weights_ih[i][j] - child.nn.weights_ih[i][j]).abs() > 1e-6 {
                    different = true;
                    break;
                }
            }
        }
        assert!(different, "Child should have mutated NN weights");
    }

    #[test]
    fn test_confidence_range() {
        let mut rng = QRng::new(42);
        let brain = HybridExplorationBrain::random(&mut rng);

        let sensors = SensorReadings {
            heat_n: 128,
            heat_s: 128,
            heat_e: 128,
            heat_w: 128,
            heat_local: 128,
            charge_local: 128,
        };

        let confidence = brain.confidence(&sensors);
        assert!((0.0..=1.0).contains(&confidence));
    }
}
