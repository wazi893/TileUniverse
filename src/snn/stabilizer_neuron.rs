//! Stabilizer Quantum Neurons (EPIC 130)
//!
//! Neurons that ARE qubits, using the stabilizer formalism for efficient
//! simulation of quantum neuromorphic systems at scale.
//!
//! # Key Concept
//!
//! Each neuron corresponds to a qubit index in a shared `StabilizerTableau`.
//! Synaptic connections are represented as entangling gates (CZ, CNOT).
//! Firing is determined by Z-basis measurement.
//!
//! # Memory Efficiency
//!
//! - StabilizerNeuron: ~8 bytes (metadata only)
//! - Shared tableau: O(n²) bits for n neurons
//! - 10K neurons: ~50 MB total
//!
//! # Example
//!
//! ```ignore
//! use engine::snn::stabilizer_neuron::{StabilizerNeuron, StabilizerNeuronConfig};
//!
//! let config = StabilizerNeuronConfig::default();
//! let neuron = StabilizerNeuron::new(0, &config);
//! ```

use std::fmt;

/// Configuration for stabilizer-based quantum neurons.
#[derive(Clone, Debug)]
pub struct StabilizerNeuronConfig {
    /// Fire when Z-measurement returns 1 (|1⟩ state)
    pub fire_on_measure_one: bool,

    /// Number of ticks to remain in refractory state after firing
    pub refractory_ticks: u8,

    /// Automatically reset neuron to |0⟩ after firing
    pub auto_reset: bool,

    /// Apply Hadamard on input neurons to create superposition
    pub hadamard_on_input: bool,
}

impl Default for StabilizerNeuronConfig {
    fn default() -> Self {
        Self {
            fire_on_measure_one: true,
            refractory_ticks: 2,
            auto_reset: true,
            hadamard_on_input: true,
        }
    }
}

impl StabilizerNeuronConfig {
    /// Create config for input layer neurons
    pub fn input() -> Self {
        Self {
            fire_on_measure_one: true,
            refractory_ticks: 1,
            auto_reset: true,
            hadamard_on_input: true,
        }
    }

    /// Create config for hidden layer neurons
    pub fn hidden() -> Self {
        Self {
            fire_on_measure_one: true,
            refractory_ticks: 2,
            auto_reset: true,
            hadamard_on_input: false, // Hidden neurons get entangled, not H-gated
        }
    }

    /// Create config for output layer neurons
    pub fn output() -> Self {
        Self {
            fire_on_measure_one: true,
            refractory_ticks: 2,
            auto_reset: true,
            hadamard_on_input: false,
        }
    }
}

/// A quantum neuron backed by a stabilizer qubit.
///
/// The actual quantum state is stored in a shared `StabilizerTableau`;
/// this struct holds only the metadata and index into that tableau.
#[derive(Clone, Debug)]
pub struct StabilizerNeuron {
    /// Index into the shared StabilizerTableau
    pub qubit_idx: usize,

    /// Layer this neuron belongs to (0 = input, 1+ = hidden/output)
    pub layer: u8,

    /// Refractory period counter (0 = ready, >0 = in refractory)
    pub refractory: u8,

    /// Refractory period length from config
    refractory_ticks: u8,

    /// Last tick when this neuron fired (for learning)
    pub last_spike_time: u8,

    /// Whether this neuron fired on the current tick
    pub fired: bool,

    /// Fire when measurement is 1
    fire_on_one: bool,

    /// Auto-reset after firing
    auto_reset: bool,
}

impl StabilizerNeuron {
    /// Create a new stabilizer neuron at the given qubit index.
    pub fn new(qubit_idx: usize, layer: u8, config: &StabilizerNeuronConfig) -> Self {
        Self {
            qubit_idx,
            layer,
            refractory: 0,
            refractory_ticks: config.refractory_ticks,
            last_spike_time: 0,
            fired: false,
            fire_on_one: config.fire_on_measure_one,
            auto_reset: config.auto_reset,
        }
    }

    /// Check if the neuron is ready to fire (not in refractory period).
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.refractory == 0
    }

    /// Check if the neuron fired on the current tick.
    #[inline]
    pub fn fired(&self) -> bool {
        self.fired
    }

    /// Process a measurement result and determine if the neuron fires.
    ///
    /// Returns `true` if the neuron fires (and should be reset).
    pub fn process_measurement(&mut self, measurement: u8, tick: u8) -> bool {
        // Clear previous tick's fire status
        self.fired = false;

        // Decrement refractory if active
        if self.refractory > 0 {
            self.refractory -= 1;
            return false;
        }

        // Check if measurement triggers firing
        let should_fire = if self.fire_on_one {
            measurement == 1
        } else {
            measurement == 0
        };

        if should_fire {
            self.fired = true;
            self.last_spike_time = tick;
            self.refractory = self.refractory_ticks;
            return true;
        }

        false
    }

    /// Manually enter refractory period (used when resetting).
    pub fn enter_refractory(&mut self) {
        self.refractory = self.refractory_ticks;
    }

    /// Check if auto-reset is enabled.
    #[inline]
    pub fn should_auto_reset(&self) -> bool {
        self.auto_reset
    }

    /// Get time since last spike (wrapping arithmetic for u8).
    pub fn time_since_spike(&self, current_tick: u8) -> u8 {
        current_tick.wrapping_sub(self.last_spike_time)
    }
}

impl fmt::Display for StabilizerNeuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StabNeuron(q={}, layer={}, {})",
            self.qubit_idx,
            self.layer,
            if self.fired {
                "FIRED"
            } else if self.refractory > 0 {
                "refractory"
            } else {
                "ready"
            }
        )
    }
}

/// Types of entangling gates for synaptic connections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntanglingGate {
    /// Controlled-Z gate (symmetric)
    CZ,
    /// CNOT gate (source=control, target=target)
    CNOT,
    /// SWAP gate (exchanges qubit states)
    SWAP,
}

impl Default for EntanglingGate {
    fn default() -> Self {
        Self::CZ // CZ is symmetric, good default
    }
}

impl fmt::Display for EntanglingGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CZ => write!(f, "CZ"),
            Self::CNOT => write!(f, "CNOT"),
            Self::SWAP => write!(f, "SWAP"),
        }
    }
}

/// A synaptic connection represented as an entangling gate.
#[derive(Clone, Debug)]
pub struct StabilizerSynapse {
    /// Source qubit index (control for CNOT)
    pub source: usize,

    /// Target qubit index
    pub target: usize,

    /// Type of entangling gate
    pub gate_type: EntanglingGate,

    /// Whether this synapse is active (can be disabled for learning)
    pub active: bool,

    /// Correlation strength (for learning, Q1.7 fixed point)
    pub correlation: i16,
}

impl StabilizerSynapse {
    /// Create a new synapse between two qubits.
    pub fn new(source: usize, target: usize, gate_type: EntanglingGate) -> Self {
        Self {
            source,
            target,
            gate_type,
            active: true,
            correlation: 0,
        }
    }

    /// Create a CZ synapse (most common).
    pub fn cz(source: usize, target: usize) -> Self {
        Self::new(source, target, EntanglingGate::CZ)
    }

    /// Create a CNOT synapse.
    pub fn cnot(source: usize, target: usize) -> Self {
        Self::new(source, target, EntanglingGate::CNOT)
    }

    /// Check if synapse is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Update correlation based on spike observations.
    ///
    /// - Both fire: positive correlation
    /// - One fires: negative correlation
    /// - Neither fires: no change
    pub fn update_correlation(
        &mut self,
        source_fired: bool,
        target_fired: bool,
        learning_rate: u8,
    ) {
        let lr = learning_rate as i16;
        if source_fired && target_fired {
            // Both fired: strengthen
            self.correlation = self.correlation.saturating_add(lr);
        } else if source_fired != target_fired {
            // Only one fired: weaken
            self.correlation = self.correlation.saturating_sub(lr / 2);
        }
        // Clamp to reasonable range
        self.correlation = self.correlation.clamp(-1000, 1000);
    }

    /// Apply reward modulation to correlation.
    pub fn apply_reward(&mut self, reward: i8) {
        // Scale correlation by reward
        let scaled = (self.correlation as i32 * reward as i32) >> 7;
        self.correlation = scaled.clamp(-1000, 1000) as i16;
    }
}

impl fmt::Display for StabilizerSynapse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({}->{}, corr={}, {})",
            self.gate_type,
            self.source,
            self.target,
            self.correlation,
            if self.active { "ON" } else { "off" }
        )
    }
}

/// Layer information for network topology.
#[derive(Clone, Debug)]
pub struct StabilizerLayer {
    /// Layer index (0 = input)
    pub index: u8,

    /// Starting qubit index for this layer
    pub start: usize,

    /// Number of neurons in this layer
    pub count: usize,

    /// Layer type
    pub layer_type: LayerType,
}

/// Type of layer in the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    Input,
    Hidden,
    Output,
}

impl StabilizerLayer {
    /// Create a new layer.
    pub fn new(index: u8, start: usize, count: usize, layer_type: LayerType) -> Self {
        Self {
            index,
            start,
            count,
            layer_type,
        }
    }

    /// Get the ending qubit index (exclusive).
    pub fn end(&self) -> usize {
        self.start + self.count
    }

    /// Check if a qubit index belongs to this layer.
    pub fn contains(&self, qubit_idx: usize) -> bool {
        qubit_idx >= self.start && qubit_idx < self.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_size() {
        // Verify neuron metadata fits in ~16 bytes (close to 8-byte target)
        assert!(std::mem::size_of::<StabilizerNeuron>() <= 24);
    }

    #[test]
    fn test_neuron_creation() {
        let config = StabilizerNeuronConfig::default();
        let neuron = StabilizerNeuron::new(42, 0, &config);

        assert_eq!(neuron.qubit_idx, 42);
        assert_eq!(neuron.layer, 0);
        assert!(neuron.is_ready());
        assert!(!neuron.fired());
    }

    #[test]
    fn test_neuron_firing() {
        let config = StabilizerNeuronConfig::default();
        let mut neuron = StabilizerNeuron::new(0, 0, &config);

        // Measurement of 0 should not fire (fire_on_one=true)
        assert!(!neuron.process_measurement(0, 0));
        assert!(!neuron.fired());

        // Measurement of 1 should fire
        assert!(neuron.process_measurement(1, 1));
        assert!(neuron.fired());

        // Should be in refractory now
        assert!(!neuron.is_ready());
        assert_eq!(neuron.refractory, 2);

        // Should not fire while in refractory
        assert!(!neuron.process_measurement(1, 2));
        assert!(!neuron.fired());
        assert_eq!(neuron.refractory, 1);

        // Still refractory
        assert!(!neuron.process_measurement(1, 3));
        assert_eq!(neuron.refractory, 0);

        // Now ready again
        assert!(neuron.is_ready());
        assert!(neuron.process_measurement(1, 4));
        assert!(neuron.fired());
    }

    #[test]
    fn test_synapse_creation() {
        let syn = StabilizerSynapse::cz(0, 1);
        assert_eq!(syn.source, 0);
        assert_eq!(syn.target, 1);
        assert_eq!(syn.gate_type, EntanglingGate::CZ);
        assert!(syn.is_active());
        assert_eq!(syn.correlation, 0);
    }

    #[test]
    fn test_synapse_correlation() {
        let mut syn = StabilizerSynapse::cz(0, 1);

        // Both fire: strengthen
        syn.update_correlation(true, true, 10);
        assert_eq!(syn.correlation, 10);

        // One fires: weaken
        syn.update_correlation(true, false, 10);
        assert_eq!(syn.correlation, 5);

        // Neither fires: no change
        syn.update_correlation(false, false, 10);
        assert_eq!(syn.correlation, 5);
    }

    #[test]
    fn test_layer() {
        let layer = StabilizerLayer::new(0, 0, 10, LayerType::Input);
        assert_eq!(layer.end(), 10);
        assert!(layer.contains(0));
        assert!(layer.contains(9));
        assert!(!layer.contains(10));
    }

    #[test]
    fn test_config_presets() {
        let input = StabilizerNeuronConfig::input();
        assert_eq!(input.refractory_ticks, 1);
        assert!(input.hadamard_on_input);

        let hidden = StabilizerNeuronConfig::hidden();
        assert!(!hidden.hadamard_on_input);

        let output = StabilizerNeuronConfig::output();
        assert!(!output.hadamard_on_input);
    }
}
