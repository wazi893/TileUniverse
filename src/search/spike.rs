//! SNN Spike Encoding for the search engine (Phase 2).
//!
//! Converts fitness values to spike trains for neural processing.
//! Implements threshold neurons and spike integration for
//! quantum-neural hybrid search.

use super::candidate::{Rng, Xorshift64};
use super::substrate::SearchSubstrate;

/// Spike encoding configuration
#[derive(Clone, Debug)]
pub struct SpikeConfig {
    /// Time window for spike generation (ticks)
    pub time_window: usize,
    /// Base firing rate for zero fitness (Hz equivalent)
    pub base_rate: f32,
    /// Maximum firing rate for optimal fitness (Hz equivalent)
    pub max_rate: f32,
    /// Threshold for spike integration
    pub threshold: f32,
    /// Leak rate for membrane potential (per tick)
    pub leak_rate: f32,
    /// Refractory period after spike (ticks)
    pub refractory_period: usize,
}

impl Default for SpikeConfig {
    fn default() -> Self {
        Self {
            time_window: 100,
            base_rate: 0.01,
            max_rate: 0.5,
            threshold: 1.0,
            leak_rate: 0.1,
            refractory_period: 2,
        }
    }
}

impl SpikeConfig {
    /// Create config optimized for fast convergence
    pub fn fast() -> Self {
        Self {
            time_window: 50,
            base_rate: 0.05,
            max_rate: 0.8,
            threshold: 0.8,
            leak_rate: 0.15,
            refractory_period: 1,
        }
    }

    /// Create config for precise discrimination
    pub fn precise() -> Self {
        Self {
            time_window: 200,
            base_rate: 0.01,
            max_rate: 0.3,
            threshold: 1.5,
            leak_rate: 0.05,
            refractory_period: 3,
        }
    }
}

/// Converts fitness values to spike rates
#[derive(Clone, Debug)]
pub struct FitnessToSpikeEncoder {
    config: SpikeConfig,
}

impl FitnessToSpikeEncoder {
    /// Create a new encoder with given config
    pub fn new(config: SpikeConfig) -> Self {
        Self { config }
    }

    /// Convert fitness (0-255) to spike rate (0.0-1.0)
    #[inline]
    pub fn fitness_to_rate(&self, fitness: u8) -> f32 {
        let normalized = fitness as f32 / 255.0;
        self.config.base_rate + normalized * (self.config.max_rate - self.config.base_rate)
    }

    /// Generate a spike train for a given fitness value
    pub fn generate_spike_train(&self, fitness: u8, rng: &mut impl Rng) -> SpikeTrain {
        let rate = self.fitness_to_rate(fitness);
        let mut spikes = Vec::with_capacity(self.config.time_window / 4);

        for t in 0..self.config.time_window {
            if rng.next_f32() < rate {
                spikes.push(t as u32);
            }
        }

        SpikeTrain {
            spikes,
            time_window: self.config.time_window as u32,
            source_fitness: fitness,
        }
    }

    /// Encode all candidates in a substrate to spike rates
    pub fn encode_substrate(&self, substrate: &SearchSubstrate) -> Vec<f32> {
        substrate
            .candidates()
            .iter()
            .map(|c| self.fitness_to_rate(c.fitness))
            .collect()
    }
}

impl Default for FitnessToSpikeEncoder {
    fn default() -> Self {
        Self::new(SpikeConfig::default())
    }
}

/// A train of spike times
#[derive(Clone, Debug)]
pub struct SpikeTrain {
    /// Spike times within the window
    pub spikes: Vec<u32>,
    /// Total time window
    pub time_window: u32,
    /// Original fitness value
    pub source_fitness: u8,
}

impl SpikeTrain {
    /// Get the spike rate (spikes per tick)
    pub fn rate(&self) -> f32 {
        if self.time_window > 0 {
            self.spikes.len() as f32 / self.time_window as f32
        } else {
            0.0
        }
    }

    /// Get spike count
    pub fn count(&self) -> usize {
        self.spikes.len()
    }

    /// Check if there's a spike at given time
    pub fn has_spike_at(&self, t: u32) -> bool {
        self.spikes.contains(&t)
    }

    /// Get inter-spike intervals
    pub fn intervals(&self) -> Vec<u32> {
        if self.spikes.len() < 2 {
            return vec![];
        }
        self.spikes.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Mean inter-spike interval
    pub fn mean_interval(&self) -> Option<f32> {
        let intervals = self.intervals();
        if intervals.is_empty() {
            None
        } else {
            Some(intervals.iter().sum::<u32>() as f32 / intervals.len() as f32)
        }
    }
}

/// Leaky Integrate-and-Fire (LIF) neuron
#[derive(Clone, Debug)]
pub struct LIFNeuron {
    /// Current membrane potential
    pub membrane_potential: f32,
    /// Firing threshold
    pub threshold: f32,
    /// Leak rate per tick
    pub leak_rate: f32,
    /// Refractory counter (ticks until can fire again)
    pub refractory: usize,
    /// Refractory period duration
    pub refractory_period: usize,
    /// Total spike count
    pub spike_count: u64,
    /// Last spike time
    pub last_spike: Option<u32>,
}

impl LIFNeuron {
    /// Create a new LIF neuron
    pub fn new(threshold: f32, leak_rate: f32, refractory_period: usize) -> Self {
        Self {
            membrane_potential: 0.0,
            threshold,
            leak_rate,
            refractory: 0,
            refractory_period,
            spike_count: 0,
            last_spike: None,
        }
    }

    /// Create from spike config
    pub fn from_config(config: &SpikeConfig) -> Self {
        Self::new(config.threshold, config.leak_rate, config.refractory_period)
    }

    /// Process one tick with given input current
    /// Returns true if the neuron fires
    pub fn tick(&mut self, input: f32, t: u32) -> bool {
        // Handle refractory period
        if self.refractory > 0 {
            self.refractory -= 1;
            self.membrane_potential = 0.0;
            return false;
        }

        // Leak
        self.membrane_potential *= 1.0 - self.leak_rate;

        // Integrate input
        self.membrane_potential += input;

        // Check threshold
        if self.membrane_potential >= self.threshold {
            self.membrane_potential = 0.0;
            self.refractory = self.refractory_period;
            self.spike_count += 1;
            self.last_spike = Some(t);
            true
        } else {
            false
        }
    }

    /// Reset the neuron state
    pub fn reset(&mut self) {
        self.membrane_potential = 0.0;
        self.refractory = 0;
        self.spike_count = 0;
        self.last_spike = None;
    }

    /// Get the current state as a tuple (potential, refractory, spike_count)
    pub fn state(&self) -> (f32, usize, u64) {
        (self.membrane_potential, self.refractory, self.spike_count)
    }
}

impl Default for LIFNeuron {
    fn default() -> Self {
        Self::new(1.0, 0.1, 2)
    }
}

/// Spike integration layer for combining multiple spike trains
#[derive(Clone, Debug)]
pub struct SpikeIntegrator {
    /// Integration neurons (one per output)
    neurons: Vec<LIFNeuron>,
    /// Connection weights from inputs to outputs
    weights: Vec<Vec<f32>>,
    /// Number of inputs
    num_inputs: usize,
    /// Number of outputs
    num_outputs: usize,
}

impl SpikeIntegrator {
    /// Create a new spike integrator
    pub fn new(num_inputs: usize, num_outputs: usize, config: &SpikeConfig) -> Self {
        let neurons = (0..num_outputs)
            .map(|_| LIFNeuron::from_config(config))
            .collect();

        // Initialize weights uniformly
        let weight = 1.0 / num_inputs as f32;
        let weights = (0..num_outputs).map(|_| vec![weight; num_inputs]).collect();

        Self {
            neurons,
            weights,
            num_inputs,
            num_outputs,
        }
    }

    /// Create with random weights
    pub fn with_random_weights(
        num_inputs: usize,
        num_outputs: usize,
        config: &SpikeConfig,
        rng: &mut impl Rng,
    ) -> Self {
        let neurons = (0..num_outputs)
            .map(|_| LIFNeuron::from_config(config))
            .collect();

        let weights = (0..num_outputs)
            .map(|_| {
                (0..num_inputs)
                    .map(|_| rng.next_f32() * 2.0 / num_inputs as f32)
                    .collect()
            })
            .collect();

        Self {
            neurons,
            weights,
            num_inputs,
            num_outputs,
        }
    }

    /// Process one tick with input spikes
    /// input_spikes[i] = true if input i fired this tick
    pub fn tick(&mut self, input_spikes: &[bool], t: u32) -> Vec<bool> {
        assert_eq!(input_spikes.len(), self.num_inputs);

        let mut output_spikes = Vec::with_capacity(self.num_outputs);

        for (neuron_idx, neuron) in self.neurons.iter_mut().enumerate() {
            // Compute weighted input
            let input: f32 = input_spikes
                .iter()
                .zip(self.weights[neuron_idx].iter())
                .filter_map(|(&spike, &weight)| if spike { Some(weight) } else { None })
                .sum();

            output_spikes.push(neuron.tick(input, t));
        }

        output_spikes
    }

    /// Process a full time window with spike trains
    pub fn process_trains(&mut self, trains: &[SpikeTrain]) -> Vec<SpikeTrain> {
        assert_eq!(trains.len(), self.num_inputs);

        let time_window = trains.first().map(|t| t.time_window).unwrap_or(100);

        // Reset neurons
        for neuron in &mut self.neurons {
            neuron.reset();
        }

        // Collect output spikes
        let mut output_spikes: Vec<Vec<u32>> = vec![vec![]; self.num_outputs];

        for t in 0..time_window {
            // Get input spikes at time t
            let input_spikes: Vec<bool> = trains.iter().map(|tr| tr.has_spike_at(t)).collect();

            // Process through integrator
            let outputs = self.tick(&input_spikes, t);

            // Record output spikes
            for (i, &fired) in outputs.iter().enumerate() {
                if fired {
                    output_spikes[i].push(t);
                }
            }
        }

        // Convert to spike trains
        output_spikes
            .into_iter()
            .map(|spikes| SpikeTrain {
                spikes,
                time_window,
                source_fitness: 0, // Integrated, no single source
            })
            .collect()
    }

    /// Get output spike counts
    pub fn spike_counts(&self) -> Vec<u64> {
        self.neurons.iter().map(|n| n.spike_count).collect()
    }

    /// Update weights using simple Hebbian learning
    /// Strengthens connections where pre and post fire together
    pub fn hebbian_update(&mut self, input_spikes: &[bool], output_spikes: &[bool], rate: f32) {
        for (out_idx, &out_spike) in output_spikes.iter().enumerate() {
            if out_spike {
                for (in_idx, &in_spike) in input_spikes.iter().enumerate() {
                    if in_spike {
                        self.weights[out_idx][in_idx] += rate;
                    }
                }
            }
        }

        // Normalize weights
        for weights in &mut self.weights {
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                for w in weights.iter_mut() {
                    *w /= sum;
                }
            }
        }
    }

    /// Get weight matrix
    pub fn weights(&self) -> &[Vec<f32>] {
        &self.weights
    }

    /// Set specific weight
    pub fn set_weight(&mut self, output: usize, input: usize, weight: f32) {
        self.weights[output][input] = weight;
    }
}

/// Population-level spike encoding
/// Encodes the entire search population as a spike layer
pub struct PopulationSpikeLayer {
    /// Encoder for fitness-to-spike conversion
    encoder: FitnessToSpikeEncoder,
    /// Current spike rates for each candidate
    spike_rates: Vec<f32>,
    /// Spike trains (lazily generated)
    spike_trains: Option<Vec<SpikeTrain>>,
    /// Random number generator
    rng: Xorshift64,
}

impl PopulationSpikeLayer {
    /// Create a new population spike layer
    pub fn new(population_size: usize, config: SpikeConfig, seed: u64) -> Self {
        Self {
            encoder: FitnessToSpikeEncoder::new(config),
            spike_rates: vec![0.0; population_size],
            spike_trains: None,
            rng: Xorshift64::new(seed),
        }
    }

    /// Update spike rates from substrate fitness values
    pub fn update_from_substrate(&mut self, substrate: &SearchSubstrate) {
        self.spike_rates = self.encoder.encode_substrate(substrate);
        self.spike_trains = None; // Invalidate cached trains
    }

    /// Get spike rates
    pub fn rates(&self) -> &[f32] {
        &self.spike_rates
    }

    /// Get or generate spike trains
    pub fn get_spike_trains(&mut self, substrate: &SearchSubstrate) -> &[SpikeTrain] {
        if self.spike_trains.is_none() {
            let trains: Vec<SpikeTrain> = substrate
                .candidates()
                .iter()
                .map(|c| self.encoder.generate_spike_train(c.fitness, &mut self.rng))
                .collect();
            self.spike_trains = Some(trains);
        }
        self.spike_trains.as_ref().unwrap()
    }

    /// Get spike-based selection probabilities
    /// Higher spike rates = higher selection probability
    pub fn selection_probabilities(&self) -> Vec<f32> {
        let total: f32 = self.spike_rates.iter().sum();
        if total > 0.0 {
            self.spike_rates.iter().map(|r| r / total).collect()
        } else {
            vec![1.0 / self.spike_rates.len() as f32; self.spike_rates.len()]
        }
    }

    /// Select candidates based on spike rates
    pub fn spike_select(&mut self, count: usize) -> Vec<usize> {
        let probs = self.selection_probabilities();
        let mut selected = Vec::with_capacity(count);

        for _ in 0..count {
            let target = self.rng.next_f32();
            let mut cumulative = 0.0;
            let mut found = false;

            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= target {
                    selected.push(i);
                    found = true;
                    break;
                }
            }

            // Fallback to last if no selection made
            if !found {
                selected.push(probs.len() - 1);
            }
        }

        selected
    }

    /// Get statistics about the spike layer
    pub fn statistics(&self) -> SpikeLayerStats {
        let n = self.spike_rates.len();
        if n == 0 {
            return SpikeLayerStats::default();
        }

        let mean_rate = self.spike_rates.iter().sum::<f32>() / n as f32;
        let max_rate = self.spike_rates.iter().cloned().fold(0.0f32, f32::max);
        let min_rate = self.spike_rates.iter().cloned().fold(f32::MAX, f32::min);

        let variance = self
            .spike_rates
            .iter()
            .map(|r| (r - mean_rate).powi(2))
            .sum::<f32>()
            / n as f32;

        SpikeLayerStats {
            population_size: n,
            mean_rate,
            max_rate,
            min_rate,
            std_rate: variance.sqrt(),
        }
    }
}

/// Statistics for a spike layer
#[derive(Clone, Debug, Default)]
pub struct SpikeLayerStats {
    pub population_size: usize,
    pub mean_rate: f32,
    pub max_rate: f32,
    pub min_rate: f32,
    pub std_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::fitness::OneMax;

    #[test]
    fn test_fitness_to_spike_rate() {
        let encoder = FitnessToSpikeEncoder::default();

        // Zero fitness should give base rate
        let rate_zero = encoder.fitness_to_rate(0);
        assert!((rate_zero - 0.01).abs() < 0.001);

        // Max fitness should give max rate
        let rate_max = encoder.fitness_to_rate(255);
        assert!((rate_max - 0.5).abs() < 0.001);

        // Middle fitness should give middle rate
        let rate_mid = encoder.fitness_to_rate(127);
        assert!(rate_mid > 0.2 && rate_mid < 0.3);
    }

    #[test]
    fn test_spike_train_generation() {
        let encoder = FitnessToSpikeEncoder::default();
        let mut rng = Xorshift64::new(42);

        // High fitness should produce more spikes
        let train_high = encoder.generate_spike_train(255, &mut rng);
        let train_low = encoder.generate_spike_train(50, &mut rng);

        // Statistically, high fitness should have more spikes
        // (may occasionally fail due to randomness, but very unlikely)
        assert!(
            train_high.count() > train_low.count() / 2,
            "High fitness: {}, Low fitness: {}",
            train_high.count(),
            train_low.count()
        );
    }

    #[test]
    fn test_lif_neuron() {
        let mut neuron = LIFNeuron::new(1.0, 0.1, 2);

        // Sub-threshold input should not fire
        assert!(!neuron.tick(0.5, 0));
        assert!(!neuron.tick(0.4, 1));

        // Accumulated input should fire
        assert!(neuron.tick(0.5, 2)); // Total ~0.86 + 0.5 > 1.0

        // Should be in refractory period
        assert!(!neuron.tick(2.0, 3));
        assert!(!neuron.tick(2.0, 4));

        // After refractory, should be able to fire again
        assert!(neuron.tick(1.5, 5));
    }

    #[test]
    fn test_spike_integrator() {
        let config = SpikeConfig::default();
        let mut integrator = SpikeIntegrator::new(4, 2, &config);

        // Process some input spikes
        let input1 = vec![true, false, true, false];
        let output1 = integrator.tick(&input1, 0);
        assert_eq!(output1.len(), 2);

        // After enough input, neurons should fire
        for t in 1..20 {
            let _ = integrator.tick(&input1, t as u32);
        }

        let counts = integrator.spike_counts();
        assert!(counts.iter().any(|&c| c > 0), "Should have some spikes");
    }

    #[test]
    fn test_population_spike_layer() {
        let mut substrate = SearchSubstrate::new(100, OneMax::new(16));
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let mut layer = PopulationSpikeLayer::new(100, SpikeConfig::default(), 42);
        layer.update_from_substrate(&substrate);

        let stats = layer.statistics();
        assert_eq!(stats.population_size, 100);
        assert!(stats.mean_rate > 0.0);
        assert!(stats.max_rate >= stats.min_rate);

        // Selection should work
        let selected = layer.spike_select(10);
        assert_eq!(selected.len(), 10);
        for &idx in &selected {
            assert!(idx < 100);
        }
    }

    #[test]
    fn test_selection_probabilities() {
        let mut substrate = SearchSubstrate::new(10, OneMax::new(8));
        substrate.initialize_random(42);
        substrate.evaluate_all();

        let mut layer = PopulationSpikeLayer::new(10, SpikeConfig::default(), 42);
        layer.update_from_substrate(&substrate);

        let probs = layer.selection_probabilities();
        assert_eq!(probs.len(), 10);

        // Probabilities should sum to ~1.0
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_spike_train_intervals() {
        let train = SpikeTrain {
            spikes: vec![0, 5, 10, 20, 25],
            time_window: 30,
            source_fitness: 100,
        };

        let intervals = train.intervals();
        assert_eq!(intervals, vec![5, 5, 10, 5]);

        let mean = train.mean_interval().unwrap();
        assert!((mean - 6.25).abs() < 0.01);
    }

    #[test]
    fn test_hebbian_learning() {
        let config = SpikeConfig::default();
        let mut integrator = SpikeIntegrator::new(4, 2, &config);

        let _initial_weights = integrator.weights()[0].clone();

        // Simulate co-firing
        let input = vec![true, false, true, false];
        let output = vec![true, false];

        integrator.hebbian_update(&input, &output, 0.1);

        // Weights for active inputs to active outputs should increase (relatively)
        let new_weights = &integrator.weights()[0];
        // Due to normalization, check relative changes
        assert!(new_weights[0] > new_weights[1]); // Input 0 fired, input 1 didn't
    }
}
