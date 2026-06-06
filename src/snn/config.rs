//! SNN Configuration Structures
//!
//! Configuration for network topology, neuron parameters, and learning rules.

use super::neuron::LIFNeuron;

/// STDP (Spike-Timing-Dependent Plasticity) configuration
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct STDPConfig {
    /// LTP (Long-Term Potentiation) time constant in ticks
    /// Typical: 20 ticks (~20ms at 1kHz tick rate)
    pub tau_plus: u8,

    /// LTD (Long-Term Depression) time constant in ticks
    /// Typical: 20 ticks
    pub tau_minus: u8,

    /// Maximum potentiation magnitude (Q1.7 fixed-point)
    /// Typical: 10-20
    pub a_plus: i8,

    /// Maximum depression magnitude (Q1.7 fixed-point, should be negative or positive depending on convention)
    /// Typical: 10-15 (will be applied as negative)
    pub a_minus: i8,

    /// Minimum weight bound (Q1.7)
    pub w_min: i8,

    /// Maximum weight bound (Q1.7)
    pub w_max: i8,

    /// Eligibility trace decay rate (Q0.8, for R-STDP)
    /// How fast the eligibility trace decays before reward arrives
    pub eligibility_decay: u8,
}

impl Default for STDPConfig {
    fn default() -> Self {
        Self {
            tau_plus: 20,
            tau_minus: 20,
            a_plus: 15,
            a_minus: 12,
            w_min: -100,
            w_max: 100,
            eligibility_decay: 230, // ~0.9 decay per tick
        }
    }
}

impl STDPConfig {
    /// Create a balanced STDP configuration
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Create a configuration biased toward potentiation (faster learning)
    pub fn potentiation_biased() -> Self {
        Self {
            a_plus: 20,
            a_minus: 10,
            ..Self::default()
        }
    }

    /// Create a configuration biased toward depression (stability)
    pub fn depression_biased() -> Self {
        Self {
            a_plus: 10,
            a_minus: 20,
            ..Self::default()
        }
    }

    /// Create a configuration with no learning (fixed weights)
    pub fn frozen() -> Self {
        Self {
            a_plus: 0,
            a_minus: 0,
            ..Self::default()
        }
    }

    /// Get tau_plus as f32 for calculations
    pub fn tau_plus_f32(&self) -> f32 {
        self.tau_plus as f32
    }

    /// Get tau_minus as f32
    pub fn tau_minus_f32(&self) -> f32 {
        self.tau_minus as f32
    }
}

/// Neuron parameter configuration
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuronConfig {
    /// Leak decay factor (Q0.8)
    pub leak: u8,

    /// Firing threshold (Q8.8)
    pub threshold: i16,

    /// Refractory period in ticks
    pub refractory_period: u8,

    /// Reset potential after spike (Q8.8)
    pub v_reset: i16,
}

impl Default for NeuronConfig {
    fn default() -> Self {
        Self {
            leak: LIFNeuron::LEAK_DEFAULT,
            threshold: LIFNeuron::THRESHOLD_DEFAULT,
            refractory_period: LIFNeuron::REFRACTORY_DEFAULT,
            v_reset: LIFNeuron::V_RESET,
        }
    }
}

impl NeuronConfig {
    /// Fast-spiking neuron (low threshold, short refractory)
    pub fn fast_spiking() -> Self {
        Self {
            leak: 245,      // ~0.96, slow leak
            threshold: 200, // ~0.78, lower threshold
            refractory_period: 1,
            v_reset: -64, // Less hyperpolarization
        }
    }

    /// Regular spiking neuron (balanced)
    pub fn regular_spiking() -> Self {
        Self::default()
    }

    /// Burst-capable neuron (slow recovery)
    pub fn bursting() -> Self {
        Self {
            leak: 250,      // ~0.98, very slow leak
            threshold: 300, // ~1.17, higher threshold
            refractory_period: 4,
            v_reset: -200, // Strong hyperpolarization
        }
    }
}

/// Network topology configuration
#[derive(Clone, Debug, PartialEq)]
pub struct SNNConfig {
    /// Number of input neurons (from sensors)
    pub n_inputs: usize,

    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,

    /// Number of output neurons (for action decoding)
    pub n_outputs: usize,

    /// Connection probability for random connectivity (0.0 - 1.0)
    pub connection_prob: f32,

    /// Whether to include recurrent connections within layers
    pub recurrent: bool,

    /// Neuron parameters
    pub neuron_config: NeuronConfig,

    /// STDP learning parameters
    pub stdp_config: STDPConfig,

    /// Number of neurons per CPU (128 or 256)
    pub neurons_per_cpu: usize,

    /// Enable reward-modulated STDP
    pub use_reward_modulation: bool,

    /// Duplicate inputs for each output channel
    /// When true, creates n_inputs * n_outputs input neurons
    /// Each channel gets its own dedicated copy of inputs
    /// This enables channel-aware learning with shared observations
    pub duplicate_inputs: bool,
}

impl Default for SNNConfig {
    fn default() -> Self {
        Self {
            n_inputs: 8,
            hidden_layers: vec![64, 64],
            n_outputs: 5,
            connection_prob: 0.3,
            recurrent: true,
            neuron_config: NeuronConfig::default(),
            stdp_config: STDPConfig::default(),
            neurons_per_cpu: 128,
            use_reward_modulation: true,
            duplicate_inputs: false,
        }
    }
}

impl SNNConfig {
    /// Create a new SNN configuration
    pub fn new(n_inputs: usize, hidden_layers: Vec<usize>, n_outputs: usize) -> Self {
        Self {
            n_inputs,
            hidden_layers,
            n_outputs,
            ..Self::default()
        }
    }

    /// Create a minimal configuration for testing
    pub fn minimal() -> Self {
        Self {
            n_inputs: 4,
            hidden_layers: vec![8],
            n_outputs: 2,
            connection_prob: 0.5,
            recurrent: false,
            neurons_per_cpu: 128,
            ..Self::default()
        }
    }

    /// Create a large-scale configuration
    pub fn large_scale() -> Self {
        Self {
            n_inputs: 16,
            hidden_layers: vec![256, 256, 128],
            n_outputs: 8,
            connection_prob: 0.1, // Sparse for large networks
            recurrent: true,
            neurons_per_cpu: 128,
            ..Self::default()
        }
    }

    /// DVS-optimized configuration for MNIST classification
    ///
    /// # Architecture
    ///
    /// - Input: 28×28×2 = 1568 neurons (ON + OFF per pixel)
    /// - Hidden: 128 neurons (feature layer)
    /// - Output: 10 neurons (digits 0-9)
    ///
    /// # Parameters
    ///
    /// - Sparse connectivity (30% probability)
    /// - High threshold for DVS sparsity
    /// - Slower leak for temporal integration
    /// - Strong STDP for unsupervised feature learning
    /// - No recurrent connections (feedforward only)
    ///
    /// # Expected Performance
    ///
    /// - >85% accuracy on MNIST test set
    /// - ~5-15% DVS event sparsity
    /// - 50-100 ticks per image presentation
    pub fn for_dvs_mnist() -> Self {
        Self {
            n_inputs: 28 * 28 * 2,                       // 1568 (ON + OFF per pixel)
            hidden_layers: vec![128],                    // Feature layer
            n_outputs: 10,                               // 10 digits
            connection_prob: 0.7, // Dense connectivity (was 0.3 - too sparse!)
            recurrent: false,     // Feedforward only
            neuron_config: NeuronConfig::fast_spiking(), // Proven preset (threshold 200 vs 250)
            stdp_config: STDPConfig {
                a_plus: 20,  // Strong potentiation
                a_minus: 15, // Moderate depression
                tau_plus: 20,
                tau_minus: 20,
                w_min: -128,
                w_max: 127,
                ..STDPConfig::default()
            },
            neurons_per_cpu: 128,
            use_reward_modulation: true, // R-STDP for credit assignment
            duplicate_inputs: false,
        }
    }

    /// Total number of neurons in the network
    pub fn total_neurons(&self) -> usize {
        self.actual_input_count() + self.hidden_layers.iter().sum::<usize>() + self.n_outputs
    }

    /// Actual number of input neurons (accounting for duplication)
    pub fn actual_input_count(&self) -> usize {
        if self.duplicate_inputs {
            self.n_inputs * self.n_outputs
        } else {
            self.n_inputs
        }
    }

    /// Number of CPUs required to host this network
    pub fn required_cpus(&self) -> usize {
        let total = self.total_neurons();
        (total + self.neurons_per_cpu - 1) / self.neurons_per_cpu
    }

    /// Get layer sizes as a vector [inputs, hidden1, hidden2, ..., outputs]
    pub fn layer_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![self.actual_input_count()];
        sizes.extend(&self.hidden_layers);
        sizes.push(self.n_outputs);
        sizes
    }

    /// Estimated number of synapses based on connection probability
    pub fn estimated_synapses(&self) -> usize {
        let layers = self.layer_sizes();
        let mut total = 0;

        // Feedforward connections
        for i in 0..layers.len() - 1 {
            let connections = (layers[i] * layers[i + 1]) as f32 * self.connection_prob;
            total += connections as usize;
        }

        // Recurrent connections within hidden layers
        if self.recurrent {
            for &size in &self.hidden_layers {
                let connections = (size * size) as f32 * self.connection_prob * 0.5;
                total += connections as usize;
            }
        }

        total
    }
}

/// Runtime configuration for SNN execution
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuntimeConfig {
    /// Number of ticks per decision (for brain integration)
    pub ticks_per_decision: u32,

    /// Enable STDP learning during execution
    pub learning_enabled: bool,

    /// Global reward signal scaling factor (Q0.8)
    pub reward_scale: u8,

    /// Use GPU acceleration if available
    pub use_gpu: bool,

    /// Spike buffer size per CPU
    pub spike_buffer_size: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            ticks_per_decision: 10,
            learning_enabled: true,
            reward_scale: 128, // 0.5 in Q0.8
            use_gpu: false,
            spike_buffer_size: 4096,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snn_config_total_neurons() {
        let config = SNNConfig::new(8, vec![64, 32], 5);
        assert_eq!(config.total_neurons(), 8 + 64 + 32 + 5);
    }

    #[test]
    fn test_snn_config_required_cpus() {
        let config = SNNConfig {
            n_inputs: 8,
            hidden_layers: vec![120],
            n_outputs: 0,
            neurons_per_cpu: 128,
            ..Default::default()
        };
        // 128 neurons, 128 per CPU = 1 CPU
        assert_eq!(config.required_cpus(), 1);

        let config2 = SNNConfig {
            n_inputs: 8,
            hidden_layers: vec![256],
            n_outputs: 8,
            neurons_per_cpu: 128,
            ..Default::default()
        };
        // 272 neurons, 128 per CPU = 3 CPUs
        assert_eq!(config2.required_cpus(), 3);
    }

    #[test]
    fn test_layer_sizes() {
        let config = SNNConfig::new(8, vec![64, 32], 5);
        assert_eq!(config.layer_sizes(), vec![8, 64, 32, 5]);
    }

    #[test]
    fn test_stdp_config_presets() {
        let balanced = STDPConfig::balanced();
        assert_eq!(balanced.a_plus, 15);
        assert_eq!(balanced.a_minus, 12);

        let frozen = STDPConfig::frozen();
        assert_eq!(frozen.a_plus, 0);
        assert_eq!(frozen.a_minus, 0);
    }
}
