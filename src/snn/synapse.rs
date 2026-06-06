//! Synapse Structure and Weight Management
//!
//! Synapses connect neurons and transmit spikes with configurable:
//! - Weight (excitatory or inhibitory)
//! - Transmission delay
//! - Cross-CPU routing information

use std::collections::HashMap;

/// Synapse flags packed into 4 bits
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynapseFlags(pub u8);

impl SynapseFlags {
    /// Excitatory synapse (positive current)
    pub const EXCITATORY: u8 = 0x00;
    /// Inhibitory synapse (negative current)
    pub const INHIBITORY: u8 = 0x10;
    /// Cross-CPU synapse (requires routing)
    pub const CROSS_CPU: u8 = 0x20;
    /// Plastic synapse (subject to STDP)
    pub const PLASTIC: u8 = 0x40;
    /// Active synapse (not pruned)
    pub const ACTIVE: u8 = 0x80;

    pub fn new() -> Self {
        Self(Self::EXCITATORY | Self::PLASTIC | Self::ACTIVE)
    }

    pub fn is_inhibitory(self) -> bool {
        (self.0 & Self::INHIBITORY) != 0
    }

    pub fn is_cross_cpu(self) -> bool {
        (self.0 & Self::CROSS_CPU) != 0
    }

    pub fn is_plastic(self) -> bool {
        (self.0 & Self::PLASTIC) != 0
    }

    pub fn is_active(self) -> bool {
        (self.0 & Self::ACTIVE) != 0
    }
}

impl Default for SynapseFlags {
    fn default() -> Self {
        Self::new()
    }
}

/// Synapse connecting two neurons (4 bytes)
///
/// For local synapses (within same CPU), `target` is the local neuron index.
/// For cross-CPU synapses, the `CROSS_CPU` flag is set and routing uses
/// `CrossCpuSynapse` for full addressing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Synapse {
    /// Target neuron index (local within CPU, or lookup key for cross-CPU)
    pub target: u16,

    /// Synaptic weight in Q1.7 fixed-point
    /// Range: -128 to 127 represents -1.0 to 0.992
    /// Positive = excitatory, Negative = inhibitory
    pub weight: i8,

    /// Delay (bits 0-3) and flags (bits 4-7)
    /// Delay: 0-15 ticks transmission delay
    /// Flags: See SynapseFlags
    pub delay_flags: u8,
}

impl Synapse {
    /// Create a new excitatory synapse
    pub fn excitatory(target: u16, weight: i8, delay: u8) -> Self {
        debug_assert!(delay <= 15, "Delay must be 0-15");
        Self {
            target,
            weight: weight.abs(), // Ensure positive
            delay_flags: (delay & 0x0F) | SynapseFlags::PLASTIC | SynapseFlags::ACTIVE,
        }
    }

    /// Create a new inhibitory synapse
    pub fn inhibitory(target: u16, weight: i8, delay: u8) -> Self {
        debug_assert!(delay <= 15, "Delay must be 0-15");
        Self {
            target,
            weight: -(weight.abs()), // Ensure negative
            delay_flags: (delay & 0x0F)
                | SynapseFlags::INHIBITORY
                | SynapseFlags::PLASTIC
                | SynapseFlags::ACTIVE,
        }
    }

    /// Get transmission delay (0-15 ticks)
    #[inline]
    pub fn delay(&self) -> u8 {
        self.delay_flags & 0x0F
    }

    /// Set transmission delay
    pub fn set_delay(&mut self, delay: u8) {
        debug_assert!(delay <= 15, "Delay must be 0-15");
        self.delay_flags = (self.delay_flags & 0xF0) | (delay & 0x0F);
    }

    /// Get flags
    #[inline]
    pub fn flags(&self) -> SynapseFlags {
        SynapseFlags(self.delay_flags & 0xF0)
    }

    /// Check if synapse is excitatory
    #[inline]
    pub fn is_excitatory(&self) -> bool {
        self.weight >= 0
    }

    /// Check if synapse is inhibitory
    #[inline]
    pub fn is_inhibitory(&self) -> bool {
        self.weight < 0
    }

    /// Check if this is a cross-CPU synapse
    #[inline]
    pub fn is_cross_cpu(&self) -> bool {
        (self.delay_flags & SynapseFlags::CROSS_CPU) != 0
    }

    /// Check if this synapse is plastic (subject to STDP)
    #[inline]
    pub fn is_plastic(&self) -> bool {
        (self.delay_flags & SynapseFlags::PLASTIC) != 0
    }

    /// Get weight as f32 (-1.0 to ~1.0)
    pub fn weight_f32(&self) -> f32 {
        self.weight as f32 / 128.0
    }

    /// Set weight from f32, clamped to valid range
    pub fn set_weight_f32(&mut self, w: f32) {
        self.weight = (w * 128.0).clamp(-128.0, 127.0) as i8;
    }

    /// Compute the current contribution from this synapse
    /// Returns Q8.8 current value
    #[inline]
    pub fn current(&self) -> i16 {
        // Scale weight from Q1.7 to Q8.8
        // With threshold=256, a weight of 64 gives current of ~128 (need ~2 active synapses to fire)
        // Multiply by 2 to make the network more responsive
        (self.weight as i16) * 2
    }
}

/// Cross-CPU synapse with full addressing (12 bytes)
///
/// Used for routing spikes between CPUs in a multi-CPU SNN grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrossCpuSynapse {
    /// Source CPU ID
    pub source_cpu: u32,
    /// Source neuron index within source CPU
    pub source_neuron: u8,
    /// Target CPU ID
    pub target_cpu: u32,
    /// Target neuron index within target CPU
    pub target_neuron: u8,
    /// Synaptic weight (Q1.7)
    pub weight: i8,
    /// Transmission delay (0-15 ticks)
    pub delay: u8,
}

impl CrossCpuSynapse {
    pub fn new(
        source_cpu: u32,
        source_neuron: u8,
        target_cpu: u32,
        target_neuron: u8,
        weight: i8,
        delay: u8,
    ) -> Self {
        Self {
            source_cpu,
            source_neuron,
            target_cpu,
            target_neuron,
            weight,
            delay: delay.min(15),
        }
    }

    /// Compute the current contribution
    #[inline]
    pub fn current(&self) -> i16 {
        (self.weight as i16) * 2
    }
}

/// Synapse table for a single CPU
///
/// Stores both local and outgoing cross-CPU synapses.
#[derive(Clone, Debug, Default)]
pub struct SynapseTable {
    /// Local synapses: source neuron -> list of (target neuron, synapse)
    /// Stored in adjacency list format for efficient spike propagation
    pub local: Vec<Vec<Synapse>>,

    /// Cross-CPU synapses: source neuron -> list of cross-CPU synapses
    pub cross_cpu: HashMap<u8, Vec<CrossCpuSynapse>>,

    /// Total synapse count (local + cross-CPU)
    pub total_count: usize,
}

impl SynapseTable {
    /// Create a new synapse table for `n_neurons` neurons
    pub fn new(n_neurons: usize) -> Self {
        Self {
            local: vec![Vec::new(); n_neurons],
            cross_cpu: HashMap::new(),
            total_count: 0,
        }
    }

    /// Add a local synapse from `source` to `target`
    pub fn add_local(&mut self, source: usize, synapse: Synapse) {
        if source < self.local.len() {
            self.local[source].push(synapse);
            self.total_count += 1;
        }
    }

    /// Add a cross-CPU synapse
    pub fn add_cross_cpu(&mut self, source_neuron: u8, synapse: CrossCpuSynapse) {
        self.cross_cpu
            .entry(source_neuron)
            .or_default()
            .push(synapse);
        self.total_count += 1;
    }

    /// Get all local synapses from a source neuron
    pub fn get_local(&self, source: usize) -> &[Synapse] {
        if source < self.local.len() {
            &self.local[source]
        } else {
            &[]
        }
    }

    /// Get mutable reference to local synapses (for STDP updates)
    pub fn get_local_mut(&mut self, source: usize) -> Option<&mut Vec<Synapse>> {
        self.local.get_mut(source)
    }

    /// Get cross-CPU synapses from a source neuron
    pub fn get_cross_cpu(&self, source_neuron: u8) -> Option<&Vec<CrossCpuSynapse>> {
        self.cross_cpu.get(&source_neuron)
    }

    /// Number of neurons in this table
    pub fn n_neurons(&self) -> usize {
        self.local.len()
    }

    /// Get the global synapse offset for synapses from a source neuron
    /// Used to index into eligibility trace arrays
    pub fn synapse_offset(&self, source: usize) -> usize {
        self.local.iter().take(source).map(|v| v.len()).sum()
    }

    /// Readonly version of synapse_offset (for use when self is borrowed)
    pub fn synapse_offset_readonly(&self, source: usize) -> usize {
        self.local.iter().take(source).map(|v| v.len()).sum()
    }

    /// Accumulate currents from all spikes in this tick
    ///
    /// Given a list of neurons that fired, compute the total current
    /// to each target neuron.
    pub fn accumulate_currents(&self, fired: &[usize], currents: &mut [i16]) {
        for &source in fired {
            for syn in self.get_local(source) {
                if (syn.target as usize) < currents.len() {
                    currents[syn.target as usize] =
                        currents[syn.target as usize].saturating_add(syn.current());
                }
            }
        }
    }
}

/// CSR (Compressed Sparse Row) format for GPU-friendly synapse storage
#[derive(Clone, Debug)]
pub struct SynapseCSR {
    /// Row pointers: syn_ptr[i] is the start index of synapses from neuron i
    pub syn_ptr: Vec<u32>,
    /// Target neuron indices
    pub targets: Vec<u16>,
    /// Synapse weights
    pub weights: Vec<i8>,
    /// Synapse delays
    pub delays: Vec<u8>,
}

impl SynapseCSR {
    /// Convert a SynapseTable to CSR format
    pub fn from_table(table: &SynapseTable) -> Self {
        let mut syn_ptr = Vec::with_capacity(table.n_neurons() + 1);
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        let mut delays = Vec::new();

        let mut ptr = 0u32;
        for synapses in &table.local {
            syn_ptr.push(ptr);
            for syn in synapses {
                targets.push(syn.target);
                weights.push(syn.weight);
                delays.push(syn.delay());
                ptr += 1;
            }
        }
        syn_ptr.push(ptr);

        Self {
            syn_ptr,
            targets,
            weights,
            delays,
        }
    }

    /// Number of neurons
    pub fn n_neurons(&self) -> usize {
        self.syn_ptr.len().saturating_sub(1)
    }

    /// Total number of synapses
    pub fn n_synapses(&self) -> usize {
        self.targets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synapse_excitatory() {
        let syn = Synapse::excitatory(10, 64, 2);

        assert_eq!(syn.target, 10);
        assert_eq!(syn.weight, 64);
        assert_eq!(syn.delay(), 2);
        assert!(syn.is_excitatory());
        assert!(!syn.is_inhibitory());
        assert!(syn.is_plastic());
    }

    #[test]
    fn test_synapse_inhibitory() {
        let syn = Synapse::inhibitory(20, 32, 5);

        assert_eq!(syn.target, 20);
        assert_eq!(syn.weight, -32);
        assert_eq!(syn.delay(), 5);
        assert!(!syn.is_excitatory());
        assert!(syn.is_inhibitory());
    }

    #[test]
    fn test_synapse_current() {
        let syn = Synapse::excitatory(0, 64, 0);
        assert_eq!(syn.current(), 128); // 64 << 1

        let syn_inh = Synapse::inhibitory(0, 32, 0);
        assert_eq!(syn_inh.current(), -64); // -32 << 1
    }

    #[test]
    fn test_synapse_size() {
        // Ensure struct is 4 bytes as designed
        assert_eq!(std::mem::size_of::<Synapse>(), 4);
    }

    #[test]
    fn test_synapse_table() {
        let mut table = SynapseTable::new(4);

        // Add local synapses: neuron 0 -> neurons 1, 2
        table.add_local(0, Synapse::excitatory(1, 50, 0));
        table.add_local(0, Synapse::excitatory(2, 30, 1));

        // Add local synapse: neuron 1 -> neuron 3
        table.add_local(1, Synapse::inhibitory(3, 40, 0));

        assert_eq!(table.total_count, 3);
        assert_eq!(table.get_local(0).len(), 2);
        assert_eq!(table.get_local(1).len(), 1);
        assert_eq!(table.get_local(2).len(), 0);
    }

    #[test]
    fn test_accumulate_currents() {
        let mut table = SynapseTable::new(4);

        // Neuron 0 excites neurons 1 and 2
        table.add_local(0, Synapse::excitatory(1, 50, 0));
        table.add_local(0, Synapse::excitatory(2, 30, 0));

        // Neuron 1 inhibits neuron 3
        table.add_local(1, Synapse::inhibitory(3, 40, 0));

        let mut currents = [0i16; 4];
        table.accumulate_currents(&[0, 1], &mut currents);

        assert_eq!(currents[0], 0); // No input
        assert_eq!(currents[1], 100); // 50 << 1
        assert_eq!(currents[2], 60); // 30 << 1
        assert_eq!(currents[3], -80); // -40 << 1
    }

    #[test]
    fn test_synapse_csr() {
        let mut table = SynapseTable::new(3);
        table.add_local(0, Synapse::excitatory(1, 50, 0));
        table.add_local(0, Synapse::excitatory(2, 30, 0));
        table.add_local(2, Synapse::excitatory(0, 20, 1));

        let csr = SynapseCSR::from_table(&table);

        assert_eq!(csr.n_neurons(), 3);
        assert_eq!(csr.n_synapses(), 3);
        assert_eq!(csr.syn_ptr, vec![0, 2, 2, 3]); // 2 from 0, 0 from 1, 1 from 2
    }
}
