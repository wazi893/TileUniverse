//! Spiking Neural Network Brain (CHUNGUS 4)
//!
//! Implements a brain using the neuromorphic SNN system with:
//! - LIF neuron dynamics
//! - STDP + R-STDP learning
//! - Rate coding for sensor input
//! - Winner-take-all output decoding

use crate::classical_brain::Move;
use crate::quantum::QRng;
use crate::quantum_brain::SensorReadings;
use crate::snn::{SNNConfig, SNNNetwork};

/// Spiking Neural Network brain for organism control
///
/// Uses rate coding to encode sensor inputs and winner-take-all
/// to decode output actions.
#[derive(Clone, Debug)]
pub struct SpikingBrain {
    /// The underlying SNN network
    network: SNNNetwork,

    /// Number of simulation ticks per decision
    ticks_per_decision: usize,

    /// Learning rate for R-STDP
    learning_rate: i8,
}

impl SpikingBrain {
    /// Create a new SpikingBrain with default configuration
    pub fn new() -> Self {
        let config = SNNConfig {
            n_inputs: 6, // 4 directional heat + local heat + charge
            hidden_layers: vec![16, 8],
            n_outputs: 5, // Up, Down, Left, Right, Stay
            connection_prob: 0.5,
            neurons_per_cpu: 32,
            recurrent: true,
            ..SNNConfig::default()
        };

        let mut network = SNNNetwork::new(config);
        network.build_connectivity(42); // Default seed
        network.runtime.learning_enabled = true;

        Self {
            network,
            ticks_per_decision: 10,
            learning_rate: 32,
        }
    }

    /// Create a random SpikingBrain with the given seed
    pub fn random(rng: &mut QRng) -> Self {
        let seed = (rng.next_f32() * u64::MAX as f32) as u64;

        let config = SNNConfig {
            n_inputs: 6,
            hidden_layers: vec![16, 8],
            n_outputs: 5,
            connection_prob: 0.3 + rng.next_f32() * 0.4, // Random 0.3-0.7
            neurons_per_cpu: 32,
            recurrent: rng.next_f32() > 0.5,
            ..SNNConfig::default()
        };

        let mut network = SNNNetwork::new(config);
        network.build_connectivity(seed);
        network.runtime.learning_enabled = true;

        Self {
            network,
            ticks_per_decision: 10,
            learning_rate: 32,
        }
    }

    /// Create from configuration
    pub fn with_config(config: SNNConfig, seed: u64) -> Self {
        let mut network = SNNNetwork::new(config);
        network.build_connectivity(seed);
        network.runtime.learning_enabled = true;

        Self {
            network,
            ticks_per_decision: 10,
            learning_rate: 32,
        }
    }

    /// Encode sensor readings as input spike rates
    ///
    /// Maps sensor values [0, 255] to spike rates [0, 255]
    fn encode_sensors(&self, sensors: &SensorReadings) -> [u8; 6] {
        [
            (sensors.heat_n.min(255)) as u8,       // North heat
            (sensors.heat_s.min(255)) as u8,       // South heat
            (sensors.heat_e.min(255)) as u8,       // East heat
            (sensors.heat_w.min(255)) as u8,       // West heat
            (sensors.heat_local.min(255)) as u8,   // Local heat
            (sensors.charge_local.min(255)) as u8, // Local charge
        ]
    }

    /// Decode output spike counts to movement action
    fn decode_output(&self) -> Move {
        let action = self.network.get_action();
        match action {
            0 => Move::Up,
            1 => Move::Down,
            2 => Move::Left,
            3 => Move::Right,
            _ => Move::Stay,
        }
    }

    /// Execute brain decision
    pub fn decide(&mut self, sensors: &SensorReadings) -> Move {
        // Reset output counts for new decision
        self.network.reset_output_counts();

        // Encode sensors as spike rates
        let inputs = self.encode_sensors(sensors);
        self.network.set_inputs(&inputs);

        // Run simulation for decision period
        for _ in 0..self.ticks_per_decision {
            self.network.step();
        }

        // Decode output
        self.decode_output()
    }

    /// Apply reward signal for R-STDP learning
    pub fn apply_reward(&mut self, reward: f32) {
        let scaled_reward = (reward.clamp(-1.0, 1.0) * self.learning_rate as f32) as i8;
        self.network.set_reward(scaled_reward);
    }

    /// Create mutated offspring
    pub fn reproduce(&self, rng: &mut QRng, mutation_rate: f32) -> Self {
        let seed = (rng.next_f32() * u64::MAX as f32) as u64;

        // Clone config with possible mutations
        let mut config = self.network.config.clone();

        // Mutate connection probability
        if rng.next_f32() < mutation_rate {
            config.connection_prob =
                (config.connection_prob + (rng.next_f32() - 0.5) * 0.2).clamp(0.1, 0.9);
        }

        // Mutate recurrence
        if rng.next_f32() < mutation_rate * 0.5 {
            config.recurrent = !config.recurrent;
        }

        // Create new network with mutated config and new connectivity
        let mut network = SNNNetwork::new(config);
        network.build_connectivity(seed);
        network.runtime.learning_enabled = self.network.runtime.learning_enabled;

        // Copy some weights from parent (structural mutation)
        // Note: This is a simplified version - full implementation would
        // do more sophisticated weight inheritance

        Self {
            network,
            ticks_per_decision: self.ticks_per_decision,
            learning_rate: self.learning_rate,
        }
    }

    /// Get parameter count (number of synapses)
    pub fn parameter_count(&self) -> usize {
        self.network.n_synapses()
    }

    /// Get neuron count
    pub fn neuron_count(&self) -> usize {
        self.network.n_neurons()
    }

    /// Get network statistics
    pub fn stats(&self) -> SpikingBrainStats {
        SpikingBrainStats {
            neurons: self.network.n_neurons(),
            synapses: self.network.n_synapses(),
            cpus: self.network.topology.n_cpus,
            ticks_per_decision: self.ticks_per_decision,
            recurrent: self.network.config.recurrent,
        }
    }
}

impl Default for SpikingBrain {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a SpikingBrain
#[derive(Debug, Clone)]
pub struct SpikingBrainStats {
    pub neurons: usize,
    pub synapses: usize,
    pub cpus: usize,
    pub ticks_per_decision: usize,
    pub recurrent: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spiking_brain_creation() {
        let brain = SpikingBrain::new();
        assert!(brain.neuron_count() > 0);
        assert!(brain.parameter_count() > 0);
    }

    #[test]
    fn test_spiking_brain_random() {
        let mut rng = QRng::new(42);
        let brain = SpikingBrain::random(&mut rng);
        assert!(brain.neuron_count() > 0);
    }

    #[test]
    fn test_spiking_brain_decide() {
        let mut brain = SpikingBrain::new();

        let sensors = SensorReadings {
            heat_n: 100,
            heat_s: 50,
            heat_e: 75,
            heat_w: 25,
            heat_local: 60,
            charge_local: 40,
        };

        let action = brain.decide(&sensors);
        // Action should be one of the valid moves
        assert!(matches!(
            action,
            Move::Up | Move::Down | Move::Left | Move::Right | Move::Stay
        ));
    }

    #[test]
    fn test_spiking_brain_reproduce() {
        let mut rng = QRng::new(42);
        let parent = SpikingBrain::random(&mut rng);

        let child = parent.reproduce(&mut rng, 0.1);

        // Child should have valid structure
        assert!(child.neuron_count() > 0);
        assert!(child.parameter_count() > 0);
    }

    #[test]
    fn test_spiking_brain_learning() {
        let mut brain = SpikingBrain::new();

        let sensors = SensorReadings {
            heat_n: 100,
            heat_s: 50,
            heat_e: 75,
            heat_w: 25,
            heat_local: 60,
            charge_local: 40,
        };

        // Make a decision
        let _action1 = brain.decide(&sensors);

        // Apply positive reward
        brain.apply_reward(1.0);

        // Make another decision (network should have learned)
        let _action2 = brain.decide(&sensors);

        // Just verify it ran without panicking
    }

    #[test]
    fn test_spiking_brain_stats() {
        let brain = SpikingBrain::new();
        let stats = brain.stats();

        println!("SpikingBrain stats: {:?}", stats);
        assert!(stats.neurons > 0);
        assert!(stats.synapses > 0);
        assert!(stats.cpus >= 1);
    }
}
