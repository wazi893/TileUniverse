//! Population Coding for Smooth Action Distributions
//!
//! This module provides population coding for the output layer of the Quantum-SNN,
//! enabling smoother action probability distributions compared to one-hot encoding.
//!
//! # Key Concepts
//!
//! - **One-hot encoding**: Each action has a single dedicated neuron
//!   - Pros: Simple, sparse
//!   - Cons: Discrete jumps in probability
//!
//! - **Population coding**: Each action is represented by multiple neurons with
//!   overlapping tuning curves (typically Gaussian)
//!   - Pros: Smoother gradients, better generalization
//!   - Cons: More neurons needed
//!
//! # Example
//!
//! ```ignore
//! let encoder = PopulationEncoder::new(3, 8, 1.0); // 3 actions, 8 neurons/action
//!
//! // Spike counts from 24 output neurons (8 per action)
//! let spike_counts: Vec<u32> = get_output_spikes();
//!
//! // Decode to action probabilities
//! let probs = encoder.decode(&spike_counts);
//! ```

/// Population encoder/decoder for smooth action distributions.
///
/// Uses Gaussian tuning curves centered on each action, allowing
/// neurons to contribute to multiple nearby actions.
#[derive(Debug, Clone)]
pub struct PopulationEncoder {
    /// Number of discrete actions
    n_actions: usize,
    /// Number of neurons per action
    neurons_per_action: usize,
    /// Total neurons (n_actions * neurons_per_action)
    total_neurons: usize,
    /// Tuning curve width (Gaussian sigma)
    #[allow(dead_code)]
    sigma: f32,
    /// Precomputed Gaussian weights for each neuron
    weights: Vec<Vec<f32>>,
}

impl PopulationEncoder {
    /// Create a new population encoder.
    ///
    /// # Arguments
    /// * `n_actions` - Number of discrete actions to encode
    /// * `neurons_per_action` - How many neurons represent each action
    /// * `sigma` - Width of Gaussian tuning curves (higher = more overlap)
    pub fn new(n_actions: usize, neurons_per_action: usize, sigma: f32) -> Self {
        let total_neurons = n_actions * neurons_per_action;
        let sigma = sigma.max(0.1); // Avoid division by zero

        // Precompute Gaussian weights
        // For each neuron, compute its contribution to each action
        let mut weights = Vec::with_capacity(total_neurons);

        for neuron_idx in 0..total_neurons {
            let action_idx = neuron_idx / neurons_per_action;
            let position_in_action = neuron_idx % neurons_per_action;

            // Neuron's preferred position (center of its receptive field)
            let center = action_idx as f32
                + (position_in_action as f32 - neurons_per_action as f32 / 2.0)
                    / neurons_per_action as f32;

            let mut neuron_weights = Vec::with_capacity(n_actions);
            for action in 0..n_actions {
                // Gaussian weight based on distance to action
                let distance = (action as f32 - center).abs();
                let weight = (-distance * distance / (2.0 * sigma * sigma)).exp();
                neuron_weights.push(weight);
            }
            weights.push(neuron_weights);
        }

        Self {
            n_actions,
            neurons_per_action,
            total_neurons,
            sigma,
            weights,
        }
    }

    /// Create a simple encoder with no overlap (equivalent to one-hot)
    pub fn one_hot(n_actions: usize) -> Self {
        Self::new(n_actions, 1, 0.1)
    }

    /// Create a recommended configuration for RL tasks
    pub fn for_rl(n_actions: usize) -> Self {
        // 4 neurons per action with moderate overlap
        Self::new(n_actions, 4, 0.5)
    }

    /// Decode spike counts to action probabilities.
    ///
    /// # Arguments
    /// * `spike_counts` - Spike counts for each output neuron (length = total_neurons)
    ///
    /// # Returns
    /// Probability distribution over actions (normalized, sums to 1)
    pub fn decode(&self, spike_counts: &[u32]) -> Vec<f64> {
        if spike_counts.len() != self.total_neurons {
            // Fallback: assume spike_counts are per-action
            let mut probs = vec![0.0; self.n_actions];
            for (i, &count) in spike_counts.iter().enumerate() {
                if i < self.n_actions {
                    probs[i] = count as f64;
                }
            }
            let sum: f64 = probs.iter().sum();
            if sum > 0.0 {
                for p in &mut probs {
                    *p /= sum;
                }
            } else {
                let uniform = 1.0 / self.n_actions as f64;
                probs = vec![uniform; self.n_actions];
            }
            return probs;
        }

        // Weighted sum of spike counts for each action
        let mut action_scores = vec![0.0f64; self.n_actions];

        for (neuron_idx, &count) in spike_counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let count_f64 = count as f64;
            for (action, &weight) in self.weights[neuron_idx].iter().enumerate() {
                action_scores[action] += count_f64 * weight as f64;
            }
        }

        // Normalize to probabilities
        let sum: f64 = action_scores.iter().sum();
        if sum > 1e-10 {
            for score in &mut action_scores {
                *score /= sum;
            }
        } else {
            // No spikes: uniform distribution
            let uniform = 1.0 / self.n_actions as f64;
            action_scores = vec![uniform; self.n_actions];
        }

        action_scores
    }

    /// Encode action probabilities to target spike rates.
    ///
    /// Useful for training: given desired action probabilities,
    /// compute target spike rates for each output neuron.
    ///
    /// # Arguments
    /// * `action_probs` - Desired probability for each action
    /// * `max_rate` - Maximum spike rate (typically 255)
    ///
    /// # Returns
    /// Target spike rates for each output neuron
    pub fn encode(&self, action_probs: &[f64], max_rate: u8) -> Vec<u8> {
        if action_probs.len() != self.n_actions {
            return vec![0; self.total_neurons];
        }

        let mut rates = Vec::with_capacity(self.total_neurons);

        for neuron_idx in 0..self.total_neurons {
            // Compute weighted sum of action probabilities
            let mut rate = 0.0f64;
            for (action, &weight) in self.weights[neuron_idx].iter().enumerate() {
                rate += action_probs[action] * weight as f64;
            }

            // Scale to [0, max_rate]
            let scaled = (rate * max_rate as f64).clamp(0.0, max_rate as f64) as u8;
            rates.push(scaled);
        }

        rates
    }

    /// Get the total number of output neurons needed
    pub fn total_neurons(&self) -> usize {
        self.total_neurons
    }

    /// Get number of actions
    pub fn n_actions(&self) -> usize {
        self.n_actions
    }

    /// Get neurons per action
    pub fn neurons_per_action(&self) -> usize {
        self.neurons_per_action
    }
}

/// Configuration for population-coded outputs
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OutputEncoding {
    /// One-hot basis states |2^i⟩ (current default)
    #[default]
    OneHot,
    /// Population code with Gaussian tuning curves
    Population {
        neurons_per_action: usize,
        sigma: f32,
    },
}

impl OutputEncoding {
    /// Create a population encoding configuration
    pub fn population(neurons_per_action: usize, sigma: f32) -> Self {
        OutputEncoding::Population {
            neurons_per_action,
            sigma,
        }
    }

    /// Create the recommended population encoding for RL
    pub fn population_rl() -> Self {
        OutputEncoding::Population {
            neurons_per_action: 4,
            sigma: 0.5,
        }
    }

    /// Get the number of output neurons needed for n actions
    pub fn total_neurons(&self, n_actions: usize) -> usize {
        match self {
            OutputEncoding::OneHot => n_actions,
            OutputEncoding::Population {
                neurons_per_action, ..
            } => n_actions * neurons_per_action,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot_equivalent() {
        let encoder = PopulationEncoder::one_hot(4);

        // With one neuron per action, should behave like one-hot
        let spikes = vec![10, 5, 0, 3];
        let probs = encoder.decode(&spikes);

        // Should be proportional to spike counts
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 0.01, "Probs should sum to 1");

        // Highest count should have highest probability
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 0, "Action 0 should have highest probability");
    }

    #[test]
    fn test_population_decoding() {
        // 3 actions, 4 neurons per action = 12 total neurons
        let encoder = PopulationEncoder::new(3, 4, 0.5);

        assert_eq!(encoder.total_neurons(), 12);
        assert_eq!(encoder.n_actions(), 3);

        // Spike counts with strong activity for action 1
        let mut spikes = vec![0u32; 12];
        // Neurons 4-7 represent action 1
        spikes[4] = 10;
        spikes[5] = 20;
        spikes[6] = 15;
        spikes[7] = 5;

        let probs = encoder.decode(&spikes);

        // Action 1 should dominate
        assert_eq!(probs.len(), 3);
        assert!(probs[1] > probs[0], "Action 1 should be higher than 0");
        assert!(probs[1] > probs[2], "Action 1 should be higher than 2");

        println!("Population probs: {:?}", probs);
    }

    #[test]
    fn test_smooth_transition() {
        // With population coding, nearby actions should get some probability
        let encoder = PopulationEncoder::new(3, 4, 1.0); // Wide sigma for overlap

        // Activity between action 0 and 1
        let mut spikes = vec![0u32; 12];
        // Neurons at boundary of action 0 and 1
        spikes[3] = 10; // Last neuron of action 0
        spikes[4] = 10; // First neuron of action 1

        let probs = encoder.decode(&spikes);

        // Both action 0 and 1 should have significant probability
        assert!(
            probs[0] > 0.2,
            "Action 0 should have probability: {}",
            probs[0]
        );
        assert!(
            probs[1] > 0.2,
            "Action 1 should have probability: {}",
            probs[1]
        );

        // Action 2 should have less (farther away)
        assert!(
            probs[2] < probs[1],
            "Action 2 should be lower than action 1"
        );

        println!("Smooth transition probs: {:?}", probs);
    }

    #[test]
    fn test_encode_decode() {
        let encoder = PopulationEncoder::for_rl(4);

        // Encode a specific action probability
        let target_probs = vec![0.7, 0.2, 0.1, 0.0];
        let spike_rates = encoder.encode(&target_probs, 255);

        // Decode back
        let spike_counts: Vec<u32> = spike_rates.iter().map(|&r| r as u32).collect();
        let decoded_probs = encoder.decode(&spike_counts);

        // Should be close to original (not exact due to discretization)
        assert!(
            decoded_probs[0] > decoded_probs[1],
            "Order should be preserved"
        );
        assert!(
            decoded_probs[1] > decoded_probs[2],
            "Order should be preserved"
        );

        println!("Original: {:?}", target_probs);
        println!("Encoded rates: {:?}", spike_rates);
        println!("Decoded: {:?}", decoded_probs);
    }

    #[test]
    fn test_uniform_spikes() {
        let encoder = PopulationEncoder::new(3, 4, 0.5);

        // Equal spikes everywhere should give uniform probabilities
        let spikes = vec![10u32; 12];
        let probs = encoder.decode(&spikes);

        // Should be roughly uniform
        for p in &probs {
            assert!(
                (*p - 1.0 / 3.0).abs() < 0.1,
                "Should be roughly uniform: {:?}",
                probs
            );
        }
    }

    #[test]
    fn test_no_spikes() {
        let encoder = PopulationEncoder::new(4, 4, 0.5);

        // No spikes should give uniform distribution
        let spikes = vec![0u32; 16];
        let probs = encoder.decode(&spikes);

        let uniform = 1.0 / 4.0;
        for p in &probs {
            assert_eq!(*p, uniform, "Should be uniform with no spikes");
        }
    }

    #[test]
    fn test_output_encoding_enum() {
        let one_hot = OutputEncoding::OneHot;
        assert_eq!(one_hot.total_neurons(4), 4);

        let pop = OutputEncoding::population(4, 0.5);
        assert_eq!(pop.total_neurons(4), 16);

        let pop_rl = OutputEncoding::population_rl();
        assert_eq!(pop_rl.total_neurons(3), 12);
    }
}
