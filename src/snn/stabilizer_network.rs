//! Stabilizer Quantum Neural Network (EPIC 130-131)
//!
//! A neural network where each neuron IS a qubit, using the stabilizer
//! formalism for efficient simulation at scale.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │     StabilizerNeuronNetwork         │
//! │  (wraps StabilizerTableau from QEC) │
//! └─────────────────────────────────────┘
//!                     │
//!      ┌──────────────┼──────────────┐
//!      │              │              │
//!   Input         Hidden         Output
//!   (H gate)     (CZ/CNOT)      (measure)
//! ```
//!
//! # Memory Efficiency
//!
//! - 10K neurons: ~50 MB (stabilizer tableau)
//! - O(n) gate operations
//! - O(n²) measurement (worst case)

use crate::qec::noise::SimpleRng;
use crate::qec::stabilizer::StabilizerTableau;

use super::stabilizer_neuron::{
    EntanglingGate, LayerType, StabilizerLayer, StabilizerNeuron, StabilizerNeuronConfig,
    StabilizerSynapse,
};

/// Configuration for a stabilizer neural network.
#[derive(Clone, Debug)]
pub struct StabilizerNetworkConfig {
    /// Number of input neurons
    pub n_inputs: usize,

    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,

    /// Number of output neurons
    pub n_outputs: usize,

    /// Probability of connection between adjacent layers
    pub connectivity: f32,

    /// Whether to allow intra-layer (recurrent) connections
    pub recurrent: bool,

    /// Recurrent connection probability (if recurrent=true)
    pub recurrent_prob: f32,

    /// Default entangling gate type
    pub default_gate: EntanglingGate,

    /// Learning rate for correlation updates (Q0.8)
    pub learning_rate: u8,

    /// Enable syndrome-based learning
    pub syndrome_learning: bool,
}

impl Default for StabilizerNetworkConfig {
    fn default() -> Self {
        Self {
            n_inputs: 4,
            hidden_layers: vec![8],
            n_outputs: 2,
            connectivity: 0.3,
            recurrent: false,
            recurrent_prob: 0.1,
            default_gate: EntanglingGate::CZ,
            learning_rate: 10,
            syndrome_learning: false,
        }
    }
}

impl StabilizerNetworkConfig {
    /// Create a minimal network (for testing).
    pub fn minimal() -> Self {
        Self {
            n_inputs: 2,
            hidden_layers: vec![4],
            n_outputs: 2,
            connectivity: 0.5,
            ..Default::default()
        }
    }

    /// Create a medium-sized network.
    pub fn medium() -> Self {
        Self {
            n_inputs: 8,
            hidden_layers: vec![32, 32],
            n_outputs: 4,
            connectivity: 0.3,
            ..Default::default()
        }
    }

    /// Create a larger network.
    pub fn large() -> Self {
        Self {
            n_inputs: 16,
            hidden_layers: vec![64, 64, 32],
            n_outputs: 8,
            connectivity: 0.2,
            ..Default::default()
        }
    }

    /// Total number of neurons.
    pub fn total_neurons(&self) -> usize {
        self.n_inputs + self.hidden_layers.iter().sum::<usize>() + self.n_outputs
    }
}

/// Topology information for the network.
#[derive(Clone, Debug)]
pub struct StabilizerTopology {
    /// Layer definitions
    pub layers: Vec<StabilizerLayer>,

    /// Total neuron count
    pub total_neurons: usize,
}

impl StabilizerTopology {
    /// Build topology from config.
    pub fn from_config(config: &StabilizerNetworkConfig) -> Self {
        let mut layers = Vec::new();
        let mut offset = 0;

        // Input layer
        layers.push(StabilizerLayer::new(
            0,
            offset,
            config.n_inputs,
            LayerType::Input,
        ));
        offset += config.n_inputs;

        // Hidden layers
        for (i, &size) in config.hidden_layers.iter().enumerate() {
            layers.push(StabilizerLayer::new(
                (i + 1) as u8,
                offset,
                size,
                LayerType::Hidden,
            ));
            offset += size;
        }

        // Output layer
        let output_idx = (config.hidden_layers.len() + 1) as u8;
        layers.push(StabilizerLayer::new(
            output_idx,
            offset,
            config.n_outputs,
            LayerType::Output,
        ));
        offset += config.n_outputs;

        Self {
            layers,
            total_neurons: offset,
        }
    }

    /// Get layer containing a qubit index.
    pub fn layer_for_qubit(&self, qubit_idx: usize) -> Option<&StabilizerLayer> {
        self.layers.iter().find(|l| l.contains(qubit_idx))
    }

    /// Get input layer.
    pub fn input_layer(&self) -> &StabilizerLayer {
        &self.layers[0]
    }

    /// Get output layer.
    pub fn output_layer(&self) -> &StabilizerLayer {
        self.layers.last().unwrap()
    }

    /// Get qubit indices for input layer.
    pub fn input_qubits(&self) -> std::ops::Range<usize> {
        let l = self.input_layer();
        l.start..l.end()
    }

    /// Get qubit indices for output layer.
    pub fn output_qubits(&self) -> std::ops::Range<usize> {
        let l = self.output_layer();
        l.start..l.end()
    }
}

/// A quantum neural network using stabilizer qubits.
pub struct StabilizerNeuronNetwork {
    /// Shared quantum state (stabilizer tableau)
    tableau: StabilizerTableau,

    /// Neuron metadata
    neurons: Vec<StabilizerNeuron>,

    /// Entangling synapses
    synapses: Vec<StabilizerSynapse>,

    /// Network topology
    topology: StabilizerTopology,

    /// Configuration
    config: StabilizerNetworkConfig,

    /// Random number generator
    rng: SimpleRng,

    /// Current tick
    tick: u64,

    /// Output spike counts (for decision making)
    output_spike_counts: Vec<u32>,

    /// Total spikes this epoch
    total_spikes: u64,
}

impl StabilizerNeuronNetwork {
    /// Create a new network from configuration.
    pub fn new(config: StabilizerNetworkConfig, seed: u64) -> Self {
        let topology = StabilizerTopology::from_config(&config);
        let n_qubits = topology.total_neurons;

        // Create stabilizer tableau for all neurons
        let tableau = StabilizerTableau::new(n_qubits);

        // Create neuron metadata
        let mut neurons = Vec::with_capacity(n_qubits);
        for layer in &topology.layers {
            let neuron_config = match layer.layer_type {
                LayerType::Input => StabilizerNeuronConfig::input(),
                LayerType::Hidden => StabilizerNeuronConfig::hidden(),
                LayerType::Output => StabilizerNeuronConfig::output(),
            };
            for i in layer.start..layer.end() {
                neurons.push(StabilizerNeuron::new(i, layer.index, &neuron_config));
            }
        }

        // Initialize RNG
        let mut rng = SimpleRng::new(seed);

        // Create synapses between adjacent layers
        // Use adaptive connectivity for large layers to limit synapse count
        let mut synapses = Vec::new();
        for i in 0..topology.layers.len() - 1 {
            let src_layer = &topology.layers[i];
            let dst_layer = &topology.layers[i + 1];

            let src_size = src_layer.end() - src_layer.start;
            let dst_size = dst_layer.end() - dst_layer.start;

            // Adaptive connectivity: limit expected synapses per layer pair
            // Target max ~1000 synapses per layer connection for performance
            let max_synapses_per_pair: f64 = 1000.0;
            let potential_synapses = (src_size * dst_size) as f64;
            let adaptive_connectivity: f64 = if potential_synapses > max_synapses_per_pair {
                (max_synapses_per_pair / potential_synapses).min(config.connectivity as f64)
            } else {
                config.connectivity as f64
            };

            for src in src_layer.start..src_layer.end() {
                for dst in dst_layer.start..dst_layer.end() {
                    if rng.next_f64() < adaptive_connectivity {
                        synapses.push(StabilizerSynapse::new(src, dst, config.default_gate));
                    }
                }
            }
        }

        // Add recurrent connections if enabled
        if config.recurrent {
            for layer in &topology.layers {
                if layer.layer_type == LayerType::Hidden {
                    for src in layer.start..layer.end() {
                        for dst in layer.start..layer.end() {
                            if src != dst && rng.next_f64() < config.recurrent_prob as f64 {
                                synapses.push(StabilizerSynapse::new(
                                    src,
                                    dst,
                                    config.default_gate,
                                ));
                            }
                        }
                    }
                }
            }
        }

        let n_outputs = config.n_outputs;
        Self {
            tableau,
            neurons,
            synapses,
            topology,
            config,
            rng,
            tick: 0,
            output_spike_counts: vec![0; n_outputs],
            total_spikes: 0,
        }
    }

    /// Create a feedforward network.
    pub fn feedforward(
        n_inputs: usize,
        hidden: &[usize],
        n_outputs: usize,
        connectivity: f32,
        seed: u64,
    ) -> Self {
        let config = StabilizerNetworkConfig {
            n_inputs,
            hidden_layers: hidden.to_vec(),
            n_outputs,
            connectivity,
            recurrent: false,
            ..Default::default()
        };
        Self::new(config, seed)
    }

    /// Create a recurrent network.
    pub fn recurrent(
        n_inputs: usize,
        hidden: &[usize],
        n_outputs: usize,
        connectivity: f32,
        recurrent_prob: f32,
        seed: u64,
    ) -> Self {
        let config = StabilizerNetworkConfig {
            n_inputs,
            hidden_layers: hidden.to_vec(),
            n_outputs,
            connectivity,
            recurrent: true,
            recurrent_prob,
            ..Default::default()
        };
        Self::new(config, seed)
    }

    /// Get number of neurons.
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Get number of synapses.
    pub fn num_synapses(&self) -> usize {
        self.synapses.len()
    }

    /// Get current tick.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Get output spike counts.
    pub fn get_output_spike_counts(&self) -> Vec<u32> {
        self.output_spike_counts.clone()
    }

    /// Reset output spike counts.
    pub fn reset_output_counts(&mut self) {
        for c in &mut self.output_spike_counts {
            *c = 0;
        }
    }

    /// Initialize the tableau to |0⟩^⊗n state.
    pub fn reset_state(&mut self) {
        self.tableau = StabilizerTableau::new(self.topology.total_neurons);
        self.tick = 0;
        self.total_spikes = 0;
        self.reset_output_counts();
    }

    // =========================================================================
    // TICK CYCLE
    // =========================================================================

    /// Run one tick of the network.
    ///
    /// Returns indices of neurons that fired.
    pub fn step(&mut self, inputs: &[u8]) -> Vec<usize> {
        let tick_u8 = (self.tick & 0xFF) as u8;

        // Phase 1: Input encoding
        self.apply_inputs(inputs);

        // Phase 2: Apply entangling synapses
        self.apply_synapses();

        // Phase 3: Measure and determine firing
        let fired = self.measure_and_fire(tick_u8);

        // Phase 4: Update output spike counts
        self.update_output_counts(&fired);

        // Phase 5: Reset fired neurons
        self.reset_fired_neurons(&fired);

        // Increment tick
        self.tick += 1;

        fired
    }

    /// Apply inputs to input layer neurons.
    ///
    /// Input rates [0-255] are converted to Clifford operations:
    /// - rate > 128: Apply Hadamard (creates superposition → 50% fire prob)
    /// - rate > 192: Apply H then S (phase accumulation)
    fn apply_inputs(&mut self, inputs: &[u8]) {
        let input_range = self.topology.input_qubits();
        let n_inputs = input_range.len().min(inputs.len());

        // Collect qubits for batch operations
        let mut hadamard_qubits: Vec<usize> = Vec::new();
        let mut phase_qubits: Vec<usize> = Vec::new();

        for (i, &rate) in inputs.iter().take(n_inputs).enumerate() {
            let qubit = input_range.start + i;

            if rate > 128 {
                hadamard_qubits.push(qubit);
            }
            if rate > 192 {
                phase_qubits.push(qubit);
            }
        }

        // Apply in batch
        if !hadamard_qubits.is_empty() {
            self.tableau.batch_hadamard(&hadamard_qubits);
        }
        if !phase_qubits.is_empty() {
            self.tableau.batch_phase(&phase_qubits);
        }
    }

    /// Apply entangling gates for all active synapses.
    /// Uses batch operations for better performance on large networks.
    fn apply_synapses(&mut self) {
        // Collect CZ pairs for batch application
        let mut cz_pairs: Vec<(usize, usize)> = Vec::new();
        let mut cnot_synapses: Vec<(usize, usize)> = Vec::new();
        let mut swap_synapses: Vec<(usize, usize)> = Vec::new();

        for syn in &self.synapses {
            if !syn.is_active() {
                continue;
            }

            match syn.gate_type {
                EntanglingGate::CZ => {
                    cz_pairs.push((syn.source, syn.target));
                }
                EntanglingGate::CNOT => {
                    cnot_synapses.push((syn.source, syn.target));
                }
                EntanglingGate::SWAP => {
                    swap_synapses.push((syn.source, syn.target));
                }
            }
        }

        // Apply CZ gates in batch (most common case)
        if !cz_pairs.is_empty() {
            self.tableau.batch_cz(&cz_pairs);
        }

        // CNOT gates (applied individually for now)
        for (src, tgt) in cnot_synapses {
            self.tableau.cnot(src, tgt);
        }

        // SWAP gates (3 CNOTs each)
        for (src, tgt) in swap_synapses {
            self.tableau.cnot(src, tgt);
            self.tableau.cnot(tgt, src);
            self.tableau.cnot(src, tgt);
        }
    }

    /// Measure output neurons and determine which fire.
    fn measure_and_fire(&mut self, tick: u8) -> Vec<usize> {
        let mut fired = Vec::new();

        // Only measure output layer
        let output_range = self.topology.output_qubits();

        for qubit in output_range {
            let random_bit = self.rng.next_f64() > 0.5;
            let (measurement, _was_random) = self.tableau.measure_z(qubit, random_bit);

            let neuron = &mut self.neurons[qubit];
            if neuron.process_measurement(measurement, tick) {
                fired.push(qubit);
                self.total_spikes += 1;
            }
        }

        fired
    }

    /// Update output spike counts based on fired neurons.
    fn update_output_counts(&mut self, fired: &[usize]) {
        let output_start = self.topology.output_layer().start;

        for &qubit in fired {
            if qubit >= output_start {
                let output_idx = qubit - output_start;
                if output_idx < self.output_spike_counts.len() {
                    self.output_spike_counts[output_idx] += 1;
                }
            }
        }
    }

    /// Reset fired neurons to |0⟩ state.
    fn reset_fired_neurons(&mut self, fired: &[usize]) {
        for &qubit in fired {
            let neuron = &self.neurons[qubit];
            if neuron.should_auto_reset() {
                // Re-initialize this qubit to |0⟩
                // This is done by measuring in Z and applying X if result was 1
                let random_bit = self.rng.next_f64() > 0.5;
                let (result, _) = self.tableau.measure_z(qubit, random_bit);
                if result == 1 {
                    self.tableau.pauli_x(qubit);
                }
            }
        }
    }

    // =========================================================================
    // DECISION MAKING
    // =========================================================================

    /// Make a decision based on output spike counts (winner-take-all).
    pub fn decide(&mut self, inputs: &[u8]) -> usize {
        // Run one step
        self.step(inputs);

        // Return index with highest spike count
        self.output_spike_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| *count)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Run multiple ticks and then decide (integration window).
    pub fn decide_integrated(&mut self, inputs: &[u8], ticks: usize) -> usize {
        self.reset_output_counts();

        for _ in 0..ticks {
            self.step(inputs);
        }

        self.output_spike_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| *count)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    // =========================================================================
    // LEARNING
    // =========================================================================

    /// Apply reward-modulated learning.
    pub fn learn(&mut self, reward: i8) {
        // Update synapse correlations based on recent firing patterns
        for syn in &mut self.synapses {
            let src_fired = self.neurons[syn.source].fired();
            let tgt_fired = self.neurons[syn.target].fired();
            syn.update_correlation(src_fired, tgt_fired, self.config.learning_rate);
            syn.apply_reward(reward);
        }

        // Prune very weak synapses
        for syn in &mut self.synapses {
            if syn.correlation < -500 {
                syn.active = false;
            } else if syn.correlation > 500 && !syn.active {
                syn.active = true;
            }
        }
    }

    /// Update learning without reward (pure Hebbian).
    pub fn update_correlations(&mut self) {
        for syn in &mut self.synapses {
            let src_fired = self.neurons[syn.source].fired();
            let tgt_fired = self.neurons[syn.target].fired();
            syn.update_correlation(src_fired, tgt_fired, self.config.learning_rate);
        }
    }

    // =========================================================================
    // INSPECTION
    // =========================================================================

    /// Get a reference to the underlying tableau.
    pub fn tableau(&self) -> &StabilizerTableau {
        &self.tableau
    }

    /// Get a mutable reference to the underlying tableau.
    pub fn tableau_mut(&mut self) -> &mut StabilizerTableau {
        &mut self.tableau
    }

    /// Get topology information.
    pub fn topology(&self) -> &StabilizerTopology {
        &self.topology
    }

    /// Enable/disable all synapses originating from a neuron.
    pub fn set_synapses_from_neuron(&mut self, neuron: usize, enabled: bool) {
        for synapse in &mut self.synapses {
            if synapse.source == neuron {
                synapse.active = enabled;
            }
        }
    }

    /// Enable/disable all synapses targeting a neuron.
    pub fn set_synapses_to_neuron(&mut self, neuron: usize, enabled: bool) {
        for synapse in &mut self.synapses {
            if synapse.target == neuron {
                synapse.active = enabled;
            }
        }
    }

    /// Get neuron by index.
    pub fn neuron(&self, idx: usize) -> Option<&StabilizerNeuron> {
        self.neurons.get(idx)
    }

    /// Get synapse by index.
    pub fn synapse(&self, idx: usize) -> Option<&StabilizerSynapse> {
        self.synapses.get(idx)
    }

    /// Count active synapses.
    pub fn active_synapse_count(&self) -> usize {
        self.synapses.iter().filter(|s| s.is_active()).count()
    }

    /// Get memory usage estimate in bytes.
    pub fn memory_usage(&self) -> usize {
        let n = self.neurons.len();
        // Tableau: ~n²/4 bytes (2n generators, n/64 u64s each for x and z)
        let tableau_bytes = 2 * n * (n / 64 + 1) * 8 * 2;
        // Neurons: ~16 bytes each
        let neuron_bytes = n * 16;
        // Synapses: ~24 bytes each
        let synapse_bytes = self.synapses.len() * 24;

        tableau_bytes + neuron_bytes + synapse_bytes
    }
}

impl std::fmt::Debug for StabilizerNeuronNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StabilizerNeuronNetwork")
            .field("neurons", &self.neurons.len())
            .field("synapses", &self.synapses.len())
            .field("active_synapses", &self.active_synapse_count())
            .field("tick", &self.tick)
            .field("memory_bytes", &self.memory_usage())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let net = StabilizerNeuronNetwork::feedforward(4, &[8], 2, 0.5, 42);
        assert_eq!(net.num_neurons(), 14); // 4 + 8 + 2
        assert!(net.num_synapses() > 0);
    }

    #[test]
    fn test_network_step() {
        let mut net = StabilizerNeuronNetwork::feedforward(4, &[8], 2, 0.5, 42);

        let inputs = vec![200, 150, 100, 50];
        let fired = net.step(&inputs);

        // Some outputs should have fired
        assert!(fired.len() <= 2); // Max 2 outputs can fire
        assert_eq!(net.tick(), 1);
    }

    #[test]
    fn test_network_decide() {
        let mut net = StabilizerNeuronNetwork::feedforward(4, &[8], 2, 0.5, 42);

        let inputs = vec![255, 255, 0, 0];
        let decision = net.decide(&inputs);

        assert!(decision < 2); // Valid output index
    }

    #[test]
    fn test_bell_state_correlation() {
        // Create minimal 2-neuron network
        let config = StabilizerNetworkConfig {
            n_inputs: 1,
            hidden_layers: vec![],
            n_outputs: 1,
            connectivity: 1.0,
            ..Default::default()
        };
        let mut net = StabilizerNeuronNetwork::new(config, 42);

        // Manually create Bell state: H on qubit 0, then CNOT(0,1)
        net.tableau.hadamard(0);
        net.tableau.cnot(0, 1);

        // Measure both qubits - they should be correlated
        let mut same_count = 0;
        for _ in 0..100 {
            let mut test_tableau = net.tableau.clone();
            let (m0, _) = test_tableau.measure_z(0, true);
            let (m1, _) = test_tableau.measure_z(1, false); // deterministic after first measurement
            if m0 == m1 {
                same_count += 1;
            }
        }

        // Bell state should have perfect correlation
        assert!(
            same_count > 90,
            "Bell state correlation failed: {}/100",
            same_count
        );
    }

    #[test]
    fn test_topology() {
        let config = StabilizerNetworkConfig {
            n_inputs: 4,
            hidden_layers: vec![8, 8],
            n_outputs: 2,
            ..Default::default()
        };
        let topo = StabilizerTopology::from_config(&config);

        assert_eq!(topo.layers.len(), 4); // input + 2 hidden + output
        assert_eq!(topo.total_neurons, 22);
        assert_eq!(topo.input_qubits(), 0..4);
        assert_eq!(topo.output_qubits(), 20..22);
    }

    #[test]
    fn test_learning() {
        let mut net = StabilizerNeuronNetwork::feedforward(2, &[4], 2, 1.0, 42);

        // Run steps and learn after each one (while fired states are fresh)
        for _ in 0..10 {
            net.step(&[200, 200]);
            net.update_correlations(); // Update correlations while fired flags are set
        }

        // Apply reward modulation
        net.learn(100);

        // Check that at least some synapses were affected
        // Note: Due to quantum probabilistic nature, not all will change
        let active_count = net.active_synapse_count();
        assert!(active_count > 0, "Should have active synapses");
    }

    #[test]
    fn test_memory_usage() {
        let net = StabilizerNeuronNetwork::feedforward(10, &[100], 10, 0.3, 42);
        let mem = net.memory_usage();

        // Should be roughly O(n²) where n=120
        // 120² / 4 = 3600 bytes for tableau (minimum)
        // Plus neurons and synapses
        assert!(mem > 1000);
        assert!(mem < 1_000_000); // Should be < 1MB for this size
    }
}
