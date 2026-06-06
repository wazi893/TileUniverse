//! Leaky Integrate-and-Fire (LIF) Neuron Model
//!
//! Implements the core LIF dynamics with:
//! - Membrane potential (Q8.8 fixed-point)
//! - Leak decay factor
//! - Firing threshold
//! - Refractory period
//! - Spike timing for STDP

/// Leaky Integrate-and-Fire neuron state (8 bytes per neuron)
///
/// Uses fixed-point arithmetic for efficiency:
/// - `v_mem`: Q8.8 format (-128.0 to 127.996)
/// - `leak`: Q0.8 format (0.0 to 0.996)
/// - `threshold`: Q8.8 format
///
/// Field order is optimized to minimize padding (i16 fields first).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LIFNeuron {
    /// Membrane potential in Q8.8 fixed-point
    /// Range: -32768 to 32767 represents -128.0 to 127.996
    pub v_mem: i16,

    /// Firing threshold in Q8.8 fixed-point
    /// Default: 256 = 1.0
    pub threshold: i16,

    /// Leak decay factor in Q0.8 fixed-point
    /// 255 = 0.996 (almost no leak), 230 = 0.9 (typical), 128 = 0.5
    pub leak: u8,

    /// Refractory counter (ticks remaining until neuron can fire again)
    /// 0 = ready to fire, >0 = refractory
    pub refractory: u8,

    /// Last spike time (tick modulo 256, for STDP)
    pub last_spike_time: u8,

    /// Spike flag: 1 if neuron fired this tick, 0 otherwise
    pub spiked: u8,
}

impl LIFNeuron {
    /// Resting membrane potential (0.0 in Q8.8)
    pub const V_REST: i16 = 0;

    /// Reset potential after spike (-0.5 in Q8.8 = -128)
    pub const V_RESET: i16 = -128;

    /// Default firing threshold (1.0 in Q8.8 = 256)
    pub const THRESHOLD_DEFAULT: i16 = 256;

    /// Default leak factor (0.9 in Q0.8 = 230)
    pub const LEAK_DEFAULT: u8 = 230;

    /// Default refractory period (2 ticks)
    pub const REFRACTORY_DEFAULT: u8 = 2;

    /// Create a new LIF neuron with default parameters
    pub fn new() -> Self {
        Self {
            v_mem: Self::V_REST,
            leak: Self::LEAK_DEFAULT,
            threshold: Self::THRESHOLD_DEFAULT,
            refractory: 0,
            last_spike_time: 0,
            spiked: 0,
        }
    }

    /// Create a neuron with custom parameters
    pub fn with_params(leak: u8, threshold: i16) -> Self {
        Self {
            v_mem: Self::V_REST,
            leak,
            threshold,
            refractory: 0,
            last_spike_time: 0,
            spiked: 0,
        }
    }

    /// Execute one tick of LIF dynamics
    ///
    /// # Arguments
    /// * `input_current` - Total synaptic current (Q8.8 fixed-point)
    /// * `tick` - Current simulation tick (for STDP timing)
    ///
    /// # Returns
    /// `true` if the neuron fired this tick, `false` otherwise
    #[inline]
    pub fn step(&mut self, input_current: i16, tick: u8) -> bool {
        // Check refractory period
        if self.refractory > 0 {
            self.refractory -= 1;
            self.spiked = 0;
            return false;
        }

        // Apply leak: v_mem = v_mem * leak (Q8.8 * Q0.8 >> 8)
        let leaked = ((self.v_mem as i32 * self.leak as i32) >> 8) as i16;
        self.v_mem = leaked;

        // Integrate input current
        self.v_mem = self.v_mem.saturating_add(input_current);

        // Check firing threshold
        if self.v_mem >= self.threshold {
            // Fire!
            self.v_mem = Self::V_RESET;
            self.refractory = Self::REFRACTORY_DEFAULT;
            self.last_spike_time = tick;
            self.spiked = 1;
            true
        } else {
            self.spiked = 0;
            false
        }
    }

    /// Check if the neuron fired this tick
    #[inline]
    pub fn fired(&self) -> bool {
        self.spiked != 0
    }

    /// Check if the neuron is in refractory period
    #[inline]
    pub fn is_refractory(&self) -> bool {
        self.refractory > 0
    }

    /// Reset the neuron to resting state
    pub fn reset(&mut self) {
        self.v_mem = Self::V_REST;
        self.refractory = 0;
        self.spiked = 0;
    }

    /// Get membrane potential as f32 (for debugging/visualization)
    pub fn v_mem_f32(&self) -> f32 {
        self.v_mem as f32 / 256.0
    }

    /// Get threshold as f32
    pub fn threshold_f32(&self) -> f32 {
        self.threshold as f32 / 256.0
    }

    /// Set membrane potential from f32
    pub fn set_v_mem_f32(&mut self, v: f32) {
        self.v_mem = (v * 256.0).clamp(-32768.0, 32767.0) as i16;
    }
}

/// A population of neurons (e.g., within a single CPU)
#[derive(Clone, Debug)]
pub struct NeuronPopulation {
    /// Array of neurons
    pub neurons: Vec<LIFNeuron>,

    /// Population-level spike count this tick
    pub spike_count: u32,

    /// Total spikes since creation
    pub total_spikes: u64,
}

impl NeuronPopulation {
    /// Create a population of `n` neurons with default parameters
    pub fn new(n: usize) -> Self {
        Self {
            neurons: vec![LIFNeuron::new(); n],
            spike_count: 0,
            total_spikes: 0,
        }
    }

    /// Create a population with custom neuron parameters
    pub fn with_params(n: usize, leak: u8, threshold: i16) -> Self {
        Self {
            neurons: vec![LIFNeuron::with_params(leak, threshold); n],
            spike_count: 0,
            total_spikes: 0,
        }
    }

    /// Step all neurons with given input currents
    pub fn step(&mut self, currents: &[i16], tick: u8) {
        debug_assert_eq!(currents.len(), self.neurons.len());

        self.spike_count = 0;
        for (neuron, &current) in self.neurons.iter_mut().zip(currents.iter()) {
            if neuron.step(current, tick) {
                self.spike_count += 1;
            }
        }
        self.total_spikes += self.spike_count as u64;
    }

    /// Get indices of neurons that fired this tick
    pub fn fired_indices(&self) -> Vec<usize> {
        self.neurons
            .iter()
            .enumerate()
            .filter(|(_, n)| n.fired())
            .map(|(i, _)| i)
            .collect()
    }

    /// Get number of neurons
    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    /// Check if population is empty
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_subthreshold() {
        let mut neuron = LIFNeuron::new();

        // Small input should not cause spike
        assert!(!neuron.step(100, 0));
        assert_eq!(neuron.spiked, 0);
        assert!(!neuron.is_refractory());
    }

    #[test]
    fn test_lif_integration_and_fire() {
        let mut neuron = LIFNeuron::new();
        neuron.threshold = 256; // 1.0
        neuron.leak = 255; // Almost no leak

        // Accumulate to threshold
        assert!(!neuron.step(100, 0)); // 0 + 100 = 100
        assert!(!neuron.step(100, 1)); // 100 + 100 = 200
        assert!(neuron.step(100, 2)); // 200 + 100 = 300 > 256 → FIRE!

        assert_eq!(neuron.spiked, 1);
        assert_eq!(neuron.last_spike_time, 2);
        assert!(neuron.is_refractory());
    }

    #[test]
    fn test_lif_refractory() {
        let mut neuron = LIFNeuron::new();
        neuron.threshold = 100; // Low threshold
        neuron.leak = 255; // Almost no leak

        // Fire
        assert!(neuron.step(200, 0));
        assert_eq!(neuron.refractory, LIFNeuron::REFRACTORY_DEFAULT);

        // Large input during refractory should not fire
        assert!(!neuron.step(500, 1));
        assert_eq!(neuron.refractory, 1);

        // Still refractory
        assert!(!neuron.step(500, 2));
        assert_eq!(neuron.refractory, 0);

        // Now can fire again - need larger input to overcome V_RESET (-128)
        // After reset, v_mem = -128. With input 300: -128 + 300 = 172 > 100
        assert!(neuron.step(300, 3));
    }

    #[test]
    fn test_lif_leak() {
        let mut neuron = LIFNeuron::new();
        neuron.v_mem = 256; // 1.0
        neuron.leak = 128; // 0.5 decay
        neuron.threshold = 1000; // High threshold (won't fire)

        neuron.step(0, 0);
        assert_eq!(neuron.v_mem, 128); // Decayed to 0.5

        neuron.step(0, 1);
        assert_eq!(neuron.v_mem, 64); // Decayed to 0.25

        neuron.step(0, 2);
        assert_eq!(neuron.v_mem, 32); // Decayed to 0.125
    }

    #[test]
    fn test_lif_reset() {
        let mut neuron = LIFNeuron::new();
        neuron.v_mem = 500;
        neuron.refractory = 5;
        neuron.spiked = 1;

        neuron.reset();

        assert_eq!(neuron.v_mem, LIFNeuron::V_REST);
        assert_eq!(neuron.refractory, 0);
        assert_eq!(neuron.spiked, 0);
    }

    #[test]
    fn test_population_step() {
        let mut pop = NeuronPopulation::new(4);
        // Set low threshold so we can test firing
        for n in &mut pop.neurons {
            n.threshold = 100;
            n.leak = 255;
        }

        // Only neurons 1 and 3 get enough current
        let currents = [50, 150, 50, 200];
        pop.step(&currents, 0);

        assert_eq!(pop.spike_count, 2);
        assert!(pop.neurons[1].fired());
        assert!(pop.neurons[3].fired());
        assert!(!pop.neurons[0].fired());
        assert!(!pop.neurons[2].fired());
    }

    #[test]
    fn test_neuron_size() {
        // Ensure struct is 8 bytes as designed
        assert_eq!(std::mem::size_of::<LIFNeuron>(), 8);
    }
}
