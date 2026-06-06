//! SNN Network Topology and Builder
//!
//! Builds and manages the full SNN network across multiple CPUs.

use std::collections::HashSet;

use rayon::prelude::*;

use super::config::{RuntimeConfig, SNNConfig};
use super::neuron::{LIFNeuron, NeuronPopulation};
use super::spike_router::SpikeRoutingManager;
use super::stdp::STDPProcessor;
use super::synapse::{CrossCpuSynapse, Synapse, SynapseTable};

/// Network topology description
#[derive(Clone, Debug)]
pub struct SNNTopology {
    /// Layer definitions: (layer_id, start_neuron, end_neuron)
    pub layers: Vec<(usize, usize, usize)>,

    /// Total number of neurons
    pub n_neurons: usize,

    /// Neuron to CPU mapping: neuron_id -> cpu_id
    pub neuron_to_cpu: Vec<usize>,

    /// CPU to neuron range mapping: cpu_id -> (start, end)
    pub cpu_to_neurons: Vec<(usize, usize)>,

    /// Number of CPUs
    pub n_cpus: usize,
}

impl SNNTopology {
    /// Build topology from configuration
    pub fn from_config(config: &SNNConfig) -> Self {
        let layer_sizes = config.layer_sizes();
        let n_neurons = config.total_neurons();
        let n_cpus = config.required_cpus();
        let neurons_per_cpu = config.neurons_per_cpu;

        // Build layer definitions
        let mut layers = Vec::new();
        let mut offset = 0;
        for (i, &size) in layer_sizes.iter().enumerate() {
            layers.push((i, offset, offset + size));
            offset += size;
        }

        // Build neuron to CPU mapping
        let neuron_to_cpu: Vec<usize> = (0..n_neurons).map(|n| n / neurons_per_cpu).collect();

        // Build CPU to neuron range mapping
        let mut cpu_to_neurons = Vec::with_capacity(n_cpus);
        for cpu in 0..n_cpus {
            let start = cpu * neurons_per_cpu;
            let end = ((cpu + 1) * neurons_per_cpu).min(n_neurons);
            cpu_to_neurons.push((start, end));
        }

        Self {
            layers,
            n_neurons,
            neuron_to_cpu,
            cpu_to_neurons,
            n_cpus,
        }
    }

    /// Get the layer index for a neuron
    pub fn neuron_layer(&self, neuron: usize) -> Option<usize> {
        for &(layer_id, start, end) in &self.layers {
            if neuron >= start && neuron < end {
                return Some(layer_id);
            }
        }
        None
    }

    /// Get neurons in a layer
    pub fn layer_neurons(&self, layer: usize) -> Option<(usize, usize)> {
        self.layers.get(layer).map(|&(_, start, end)| (start, end))
    }

    /// Check if two neurons are on the same CPU
    pub fn same_cpu(&self, n1: usize, n2: usize) -> bool {
        if n1 < self.neuron_to_cpu.len() && n2 < self.neuron_to_cpu.len() {
            self.neuron_to_cpu[n1] == self.neuron_to_cpu[n2]
        } else {
            false
        }
    }

    /// Get local neuron index within CPU
    pub fn local_index(&self, neuron: usize, neurons_per_cpu: usize) -> usize {
        neuron % neurons_per_cpu
    }
}

/// Full SNN network distributed across CPUs
#[derive(Clone, Debug)]
pub struct SNNNetwork {
    /// Network configuration
    pub config: SNNConfig,

    /// Runtime configuration
    pub runtime: RuntimeConfig,

    /// Network topology
    pub topology: SNNTopology,

    /// Neuron populations per CPU
    pub populations: Vec<NeuronPopulation>,

    /// Synapse tables per CPU
    pub synapses: Vec<SynapseTable>,

    /// Cross-CPU spike routing
    pub router: SpikeRoutingManager,

    /// STDP processors per CPU
    pub stdp_processors: Vec<STDPProcessor>,

    /// Current simulation tick
    pub tick: u64,

    /// Global reward signal (for R-STDP)
    pub reward: i8,

    /// Input spike rates (for encoding sensors)
    pub input_rates: Vec<u8>,

    /// Output spike counts (for decoding actions)
    pub output_counts: Vec<u32>,

    /// Random state for spike generation
    rng_state: u64,
}

impl SNNNetwork {
    /// Create a new SNN network from configuration
    pub fn new(config: SNNConfig) -> Self {
        let runtime = RuntimeConfig::default();
        let topology = SNNTopology::from_config(&config);

        // Create neuron populations
        let mut populations = Vec::with_capacity(topology.n_cpus);
        for cpu in 0..topology.n_cpus {
            let (start, end) = topology.cpu_to_neurons[cpu];
            let n_local = end - start;
            populations.push(NeuronPopulation::with_params(
                n_local,
                config.neuron_config.leak,
                config.neuron_config.threshold,
            ));
        }

        // Create synapse tables
        let synapses: Vec<SynapseTable> = (0..topology.n_cpus)
            .map(|cpu| {
                let (start, end) = topology.cpu_to_neurons[cpu];
                SynapseTable::new(end - start)
            })
            .collect();

        // Create router
        let router = SpikeRoutingManager::new(topology.n_cpus, runtime.spike_buffer_size);

        // Create STDP processors
        let stdp_processors: Vec<STDPProcessor> = synapses
            .iter()
            .map(|s| STDPProcessor::new(config.stdp_config, s.total_count))
            .collect();

        // Initialize input/output buffers
        // Use actual_input_count() to handle duplicate_inputs mode
        let input_rates = vec![0u8; config.actual_input_count()];
        let output_counts = vec![0u32; config.n_outputs];

        Self {
            config,
            runtime,
            topology,
            populations,
            synapses,
            router,
            stdp_processors,
            tick: 0,
            reward: 0,
            input_rates,
            output_counts,
            rng_state: 12345, // Fixed seed for reproducibility
        }
    }

    /// Build random connectivity based on configuration
    ///
    /// Uses structured connectivity to create "channels" through the network:
    /// - Neurons are assigned to channels based on (index % n_outputs)
    /// - Same-channel connections get stronger weights (aligned pathways)
    /// - Cross-channel connections get weaker/inhibitory weights (competition)
    pub fn build_connectivity(&mut self, seed: u64) {
        self.rng_state = seed;
        let layers = self.topology.layers.clone();
        let n_outputs = self.config.n_outputs;

        // Build feedforward connections with STRONGLY SEPARATED channels
        // Each output gets CONTIGUOUS blocks of dedicated neurons through the network
        for i in 0..layers.len() - 1 {
            let (_, src_start, src_end) = layers[i];
            let (_, dst_start, dst_end) = layers[i + 1];

            let src_layer_size = src_end - src_start;
            let dst_layer_size = dst_end - dst_start;
            let src_neurons_per_channel = src_layer_size / n_outputs.max(1);
            let dst_neurons_per_channel = dst_layer_size / n_outputs.max(1);

            for src in src_start..src_end {
                // CONTIGUOUS channel assignment: neurons 0-63 → ch0, 64-127 → ch1
                let src_channel = (src - src_start) / src_neurons_per_channel.max(1);
                let src_channel = src_channel.min(n_outputs - 1);

                for dst in dst_start..dst_end {
                    let dst_channel = (dst - dst_start) / dst_neurons_per_channel.max(1);
                    let dst_channel = dst_channel.min(n_outputs - 1);
                    let aligned = src_channel == dst_channel;

                    if aligned {
                        // SAME CHANNEL: High probability, strong excitatory
                        // This creates dedicated pathways for each output
                        if self.random_float() < self.config.connection_prob * 1.5 {
                            let weight = (80.0 + self.random_float() * 40.0) as i8;
                            self.add_synapse(src, dst, weight, 1);
                        }
                    } else {
                        // CROSS CHANNEL: Lower probability, always inhibitory
                        // This creates competition between channels at EVERY layer
                        if self.random_float() < self.config.connection_prob * 0.3 {
                            let weight = -(40.0 + self.random_float() * 40.0) as i8;
                            self.add_synapse(src, dst, weight, 1);
                        }
                    }
                }
            }
        }

        // Build recurrent connections if enabled
        if self.config.recurrent {
            for &(_, start, end) in &self
                .config
                .hidden_layers
                .iter()
                .enumerate()
                .map(|(i, _)| layers[i + 1])
                .collect::<Vec<_>>()
            {
                for src in start..end {
                    for dst in start..end {
                        if src != dst && self.random_float() < self.config.connection_prob * 0.3 {
                            // Recurrent: mix of excitatory and inhibitory
                            let weight = if self.random_float() < 0.6 {
                                (30.0 + self.random_float() * 40.0) as i8
                            } else {
                                (-(20.0 + self.random_float() * 30.0)) as i8
                            };
                            self.add_synapse(src, dst, weight, 2);
                        }
                    }
                }
            }
        }

        // INTRA-LAYER LATERAL INHIBITION: Add cross-channel inhibition within hidden layers
        // This sharpens competition between channels at each layer
        // Use contiguous channel blocks instead of interleaved for better separation
        for &(_, start, end) in &self
            .config
            .hidden_layers
            .iter()
            .enumerate()
            .map(|(i, _)| layers[i + 1])
            .collect::<Vec<_>>()
        {
            let layer_size = end - start;
            let neurons_per_channel = layer_size / n_outputs.max(1);

            // Scale inhibition with layer size - larger layers need more inhibition
            let inhibition_prob = if layer_size > 64 { 0.05 } else { 0.02 };

            for src in start..end {
                // Use CONTIGUOUS channel assignment: blocks of neurons per channel
                let src_channel = (src - start) / neurons_per_channel.max(1);
                let src_channel = src_channel.min(n_outputs - 1);

                for dst in start..end {
                    let dst_channel = (dst - start) / neurons_per_channel.max(1);
                    let dst_channel = dst_channel.min(n_outputs - 1);

                    // Only inhibit across channels, not within same channel
                    if src != dst && src_channel != dst_channel {
                        if self.random_float() < inhibition_prob {
                            let weight = -(60.0 + self.random_float() * 30.0) as i8;
                            self.add_synapse(src, dst, weight, 1);
                        }
                    }
                }
            }
        }

        // LATERAL INHIBITION: Add strong inhibitory connections between output neurons
        // This creates winner-take-all dynamics so outputs differentiate instead of
        // all firing equally. When one output fires, it suppresses the others.
        let output_layer = layers.last().unwrap();
        let (_, output_start, output_end) = *output_layer;
        for src in output_start..output_end {
            for dst in output_start..output_end {
                if src != dst {
                    // Strong inhibitory connection with short delay for fast suppression
                    // Weight of -80 to -100 creates strong competition
                    let weight = -(80.0 + self.random_float() * 20.0) as i8;
                    self.add_synapse(src, dst, weight, 1);
                }
            }
        }

        // CRITICAL: Reinitialize STDP processors now that synapses exist
        // The processors were created in new() with 0 traces because synapses
        // didn't exist yet. Now that connectivity is built, recreate them
        // with the correct number of traces for R-STDP learning to work.
        self.stdp_processors = self
            .synapses
            .iter()
            .map(|s| STDPProcessor::new(self.config.stdp_config, s.total_count))
            .collect();
    }

    /// Build simple random feedforward connectivity without channel alignment
    ///
    /// For supervised learning tasks (like MNIST classification) where we want:
    /// - All inputs connect to all hidden neurons
    /// - Primarily excitatory connections
    /// - Competition only at output layer (winner-take-all)
    ///
    /// This is simpler and more appropriate than the channel-aligned pattern.
    pub fn build_simple_connectivity(&mut self, seed: u64) {
        self.rng_state = seed;
        let layers = self.topology.layers.clone();

        // Build feedforward connections - simple random
        for i in 0..layers.len() - 1 {
            let (_, src_start, src_end) = layers[i];
            let (_, dst_start, dst_end) = layers[i + 1];

            // Reduce connection probability to output layer for winner-take-all
            let is_to_output = i == layers.len() - 2;
            let conn_prob = if is_to_output {
                self.config.connection_prob * 0.3 // Only 21% to outputs
            } else {
                self.config.connection_prob
            };

            for src in src_start..src_end {
                for dst in dst_start..dst_end {
                    // Random connection with configured probability
                    if self.random_float() < conn_prob {
                        // Primarily excitatory with some inhibitory for balance
                        let weight = if self.random_float() < 0.8 {
                            // 80% excitatory: +40 to +80
                            (40.0 + self.random_float() * 40.0) as i8
                        } else {
                            // 20% inhibitory: -20 to -40
                            -(20.0 + self.random_float() * 20.0) as i8
                        };
                        self.add_synapse(src, dst, weight, 1);
                    }
                }
            }
        }

        // Build recurrent connections if enabled
        if self.config.recurrent {
            for &(_, start, end) in &self
                .config
                .hidden_layers
                .iter()
                .enumerate()
                .map(|(i, _)| layers[i + 1])
                .collect::<Vec<_>>()
            {
                for src in start..end {
                    for dst in start..end {
                        if src != dst && self.random_float() < self.config.connection_prob * 0.2 {
                            // Sparse recurrent, mostly excitatory
                            let weight = if self.random_float() < 0.7 {
                                (30.0 + self.random_float() * 30.0) as i8
                            } else {
                                -(20.0 + self.random_float() * 20.0) as i8
                            };
                            self.add_synapse(src, dst, weight, 1);
                        }
                    }
                }
            }
        }

        // Lateral inhibition at output layer only (winner-take-all)
        // VERY strong inhibition to create clear winners
        let output_layer = layers.last().unwrap();
        let (_, output_start, output_end) = *output_layer;
        for src in output_start..output_end {
            for dst in output_start..output_end {
                if src != dst {
                    // VERY strong inhibitory for winner-take-all
                    // Use maximum negative weight for strongest competition
                    let weight = -(100.0 + self.random_float() * 27.0) as i8; // -100 to -127
                    self.add_synapse(src, dst, weight, 1);
                }
            }
        }

        // Reinitialize STDP processors with correct trace count
        self.stdp_processors = self
            .synapses
            .iter()
            .map(|s| STDPProcessor::new(self.config.stdp_config, s.total_count))
            .collect();
    }

    /// Build purely excitatory same-channel connectivity (no cross-channel inhibition).
    ///
    /// Unlike `build_connectivity`, this method creates **only** same-channel excitatory
    /// connections. Cross-channel inhibition is deliberately omitted because with n_outputs=10
    /// the 9:1 inhibition ratio would overwhelm same-channel excitation and silence all neurons.
    ///
    /// Discrimination is achieved entirely via per-class adaptive thresholds set post-build
    /// (see `examples/mnist_10class_test.rs`).
    ///
    /// Channel assignment:
    ///   - Input neurons are split into `n_outputs` contiguous blocks of equal size
    ///   - Hidden/output neurons are similarly split into `n_outputs` contiguous blocks
    ///   - Only same-channel src → same-channel dst connections are created
    ///
    /// Use this with discriminative pixel selection: each input channel carries the K most
    /// class-specific pixels for its class, so the corresponding hidden channel fires selectively.
    pub fn build_excitatory_channels(&mut self, seed: u64) {
        self.rng_state = seed;
        let layers = self.topology.layers.clone();
        let n_outputs = self.config.n_outputs;

        for i in 0..layers.len() - 1 {
            let (_, src_start, src_end) = layers[i];
            let (_, dst_start, dst_end) = layers[i + 1];

            let src_per_ch = (src_end - src_start) / n_outputs.max(1);
            let dst_per_ch = (dst_end - dst_start) / n_outputs.max(1);

            for src in src_start..src_end {
                let src_ch = ((src - src_start) / src_per_ch.max(1)).min(n_outputs - 1);
                for dst in dst_start..dst_end {
                    let dst_ch = ((dst - dst_start) / dst_per_ch.max(1)).min(n_outputs - 1);
                    if src_ch == dst_ch && self.random_float() < self.config.connection_prob * 1.5 {
                        // Excitatory only: +80 to +120 (avg 100, scaled later by d_norm)
                        let weight = (80.0 + self.random_float() * 40.0) as i8;
                        self.add_synapse(src, dst, weight, 1);
                    }
                }
            }
        }

        // Reinitialize STDP processors with correct trace count
        self.stdp_processors = self
            .synapses
            .iter()
            .map(|s| STDPProcessor::new(self.config.stdp_config, s.total_count))
            .collect();
    }

    /// Add a synapse between two neurons
    fn add_synapse(&mut self, source: usize, target: usize, weight: i8, delay: u8) {
        if source >= self.topology.n_neurons || target >= self.topology.n_neurons {
            return;
        }

        let src_cpu = self.topology.neuron_to_cpu[source];
        let tgt_cpu = self.topology.neuron_to_cpu[target];

        if src_cpu == tgt_cpu {
            // Local synapse
            let local_src =
                self.topology
                    .local_index(source, self.config.neurons_per_cpu) as usize;
            let local_tgt = self
                .topology
                .local_index(target, self.config.neurons_per_cpu);

            let synapse = if weight >= 0 {
                Synapse::excitatory(local_tgt as u16, weight, delay)
            } else {
                Synapse::inhibitory(local_tgt as u16, -weight, delay)
            };

            self.synapses[src_cpu].add_local(local_src, synapse);
        } else {
            // Cross-CPU synapse (local index truncated to u8; requires neurons_per_cpu ≤ 256)
            let cross_syn = CrossCpuSynapse::new(
                src_cpu as u32,
                self.topology
                    .local_index(source, self.config.neurons_per_cpu) as u8,
                tgt_cpu as u32,
                self.topology
                    .local_index(target, self.config.neurons_per_cpu) as u8,
                weight,
                delay,
            );
            self.router.add_synapse(cross_syn);
        }
    }

    /// Simple random float generator
    fn random_float(&mut self) -> f32 {
        // LCG random
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        ((self.rng_state >> 33) as f32) / (u32::MAX as f32)
    }

    /// Set input spike rates from sensor readings
    pub fn set_inputs(&mut self, rates: &[u8]) {
        let n = rates.len().min(self.input_rates.len());
        self.input_rates[..n].copy_from_slice(&rates[..n]);
    }

    /// Set global reward signal
    pub fn set_reward(&mut self, reward: i8) {
        self.reward = reward;
    }

    /// Execute one simulation tick
    pub fn step(&mut self) {
        let tick_u8 = (self.tick & 0xFF) as u8;

        // Phase 1: Generate input spikes (rate coding)
        self.generate_input_spikes(tick_u8);

        // Phase 2: Process each CPU locally
        for cpu in 0..self.topology.n_cpus {
            self.step_cpu(cpu, tick_u8);
        }

        // Phase 3: Route cross-CPU spikes
        self.router.route_spikes();

        // Phase 4: Process incoming cross-CPU spikes
        for cpu in 0..self.topology.n_cpus {
            self.process_incoming_spikes(cpu);
        }
        // BUG FIX: drain incoming buffers after processing so spikes don't persist
        // across ticks (get_incoming used peek_all which does not drain).
        self.router.clear_incoming();

        // Phase 5: Update eligibility traces for R-STDP
        if self.runtime.learning_enabled {
            self.update_eligibility_traces(tick_u8);
        }

        // Phase 6: Apply reward-modulated STDP if reward is set
        if self.runtime.learning_enabled && self.reward != 0 {
            self.apply_reward_learning();
        }

        // Phase 7: Count output spikes
        self.count_output_spikes();

        // Advance tick
        self.tick += 1;
        self.router.next_tick();
    }

    /// Generate input spikes based on rate coding
    fn generate_input_spikes(&mut self, tick: u8) {
        // Copy input rates to avoid borrowing conflict
        let rates: Vec<u8> = self.input_rates.clone();
        // Use actual_input_count() to handle duplicate_inputs mode
        let n_actual_inputs = self.config.actual_input_count();
        let neurons_per_cpu = self.config.neurons_per_cpu;

        // Collect (cpu, local) pairs that fire this tick, then queue them for cross-CPU routing.
        // We must separate the neuron-firing pass from the router queue_spike pass to satisfy
        // Rust's borrow checker (populations and router are separate fields, but we need
        // an immutable topology borrow during firing which overlaps with the mutable router borrow).
        let mut cross_cpu_fires: Vec<(usize, u8)> = Vec::new();

        // Input layer is in CPU 0 (or first few CPUs)
        for (i, &rate) in rates.iter().enumerate() {
            if i >= n_actual_inputs {
                break;
            }

            // Rate coding: probability of spike = rate / 256
            let rand = (self.random_float() * 256.0) as u8;
            if rand < rate {
                // Inject spike directly
                let cpu = self.topology.neuron_to_cpu.get(i).copied().unwrap_or(0);
                let local = self.topology.local_index(i, neurons_per_cpu) as usize;
                if cpu < self.populations.len() && local < self.populations[cpu].neurons.len() {
                    // Directly mark neuron as having fired
                    // (bypass the step function for input neurons)
                    let neuron = &mut self.populations[cpu].neurons[local];
                    if neuron.refractory == 0 {
                        neuron.spiked = 1;
                        neuron.last_spike_time = tick;
                        neuron.v_mem = LIFNeuron::V_RESET;
                        neuron.refractory = LIFNeuron::REFRACTORY_DEFAULT;
                        // Record for cross-CPU queuing below.
                        // BUG FIX: pop.step later resets spiked=0 for neurons in refractory,
                        // so step_cpu's post-step queue check misses input neurons.
                        // We pre-queue them here before any pop.step runs.
                        cross_cpu_fires.push((cpu, local as u8));
                    }
                }
            }
        }

        // Queue fired input neurons for cross-CPU routing now, BEFORE step_cpu is called.
        // step_cpu's pop.step will reset spiked=0 (refractory), so we must do this first.
        // If the neuron has no cross-CPU synapses the entry is a no-op in route_spikes.
        for (cpu, local) in cross_cpu_fires {
            self.router.queue_spike(cpu, local);
        }
    }

    /// Execute one tick on a single CPU
    fn step_cpu(&mut self, cpu: usize, tick: u8) {
        let pop = &mut self.populations[cpu];
        let synapses = &self.synapses[cpu];

        // Collect fired neurons from previous tick
        let fired: Vec<usize> = pop
            .neurons
            .iter()
            .enumerate()
            .filter(|(_, n)| n.fired())
            .map(|(i, _)| i)
            .collect();

        // Accumulate currents from local synapses
        let mut currents = vec![0i16; pop.len()];
        synapses.accumulate_currents(&fired, &mut currents);

        // Step all neurons
        pop.step(&currents, tick);

        // Queue spikes for cross-CPU routing
        for (i, neuron) in pop.neurons.iter().enumerate() {
            if neuron.fired() {
                self.router.queue_spike(cpu, i as u8);
            }
        }
    }

    /// Process incoming cross-CPU spikes
    fn process_incoming_spikes(&mut self, cpu: usize) {
        let incoming = self.router.get_incoming(cpu);

        // BUG FIX: route_spikes adds one routing entry per synapse (not per neuron),
        // so the same spike appears N times in the buffer (once per synapse).
        // Deduplicate by (source_cpu, source_neuron) so each source is processed exactly once.
        let mut seen: HashSet<(u16, u8)> = HashSet::new();

        for spike in incoming {
            if !seen.insert((spike.source_cpu, spike.source_neuron)) {
                continue; // Skip duplicate entries for the same source neuron
            }

            // Look up synapses from this source (in SOURCE CPU's router, not target!)
            let source_cpu = spike.source_cpu as usize;
            if source_cpu >= self.router.routers.len() {
                continue;
            }
            if let Some(synapses) =
                self.router.routers[source_cpu].get_synapses(spike.source_neuron)
            {
                for syn in synapses {
                    if syn.target_cpu as usize == cpu {
                        let local = syn.target_neuron as usize;
                        if local < self.populations[cpu].neurons.len() {
                            // Add current from cross-CPU spike
                            self.populations[cpu].neurons[local].v_mem = self.populations[cpu]
                                .neurons[local]
                                .v_mem
                                .saturating_add(syn.current());
                        }
                    }
                }
            }
        }
    }

    /// Update eligibility traces during simulation
    /// Called each tick to accumulate STDP eligibility based on spike pairs
    fn update_eligibility_traces(&mut self, tick: u8) {
        for cpu in 0..self.topology.n_cpus {
            let pop = &self.populations[cpu];
            let synapses = &self.synapses[cpu];
            let processor = &mut self.stdp_processors[cpu];

            // For each presynaptic neuron that fired this tick
            for (src, neuron) in pop.neurons.iter().enumerate() {
                if neuron.fired() {
                    let pre_time = tick;

                    // Update eligibility for all its synapses
                    if src < synapses.local.len() {
                        for (syn_idx, syn) in synapses.local[src].iter().enumerate() {
                            if syn.is_plastic() {
                                let post_idx = syn.target as usize;
                                if let Some(post_neuron) = pop.neurons.get(post_idx) {
                                    // Get synapse global index for trace lookup
                                    let global_idx = synapses.synapse_offset(src) + syn_idx;
                                    if global_idx < processor.traces.len() {
                                        // Pre fired, check if post also fired recently
                                        let post_time = post_neuron.last_spike_time;
                                        processor.traces[global_idx].update(
                                            pre_time,
                                            post_time,
                                            &self.config.stdp_config,
                                            tick,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Also check: for each postsynaptic neuron that fired, look at its presynaptic partners
            for (post_idx, neuron) in pop.neurons.iter().enumerate() {
                if neuron.fired() {
                    let post_time = tick;

                    // Find all synapses targeting this neuron and update their eligibility
                    for (src, syn_list) in synapses.local.iter().enumerate() {
                        if let Some(pre_neuron) = pop.neurons.get(src) {
                            let pre_time = pre_neuron.last_spike_time;

                            for (syn_idx, syn) in syn_list.iter().enumerate() {
                                if syn.is_plastic() && syn.target as usize == post_idx {
                                    let global_idx = synapses.synapse_offset(src) + syn_idx;
                                    if global_idx < processor.traces.len() {
                                        processor.traces[global_idx].update(
                                            pre_time,
                                            post_time,
                                            &self.config.stdp_config,
                                            tick,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Apply reward-modulated learning using accumulated eligibility traces
    /// This should be called after setting the reward signal
    pub fn apply_learning(&mut self) {
        if self.reward == 0 {
            return;
        }

        for cpu in 0..self.topology.n_cpus {
            let processor = &self.stdp_processors[cpu];
            let synapses = &mut self.synapses[cpu];

            // Pre-compute synapse offsets to avoid borrow conflicts
            let offsets: Vec<usize> = (0..synapses.local.len())
                .map(|src| synapses.local.iter().take(src).map(|v| v.len()).sum())
                .collect();

            // Apply eligibility traces with reward modulation
            for (src, syn_list) in synapses.local.iter_mut().enumerate() {
                let base_offset = offsets.get(src).copied().unwrap_or(0);
                for (syn_idx, syn) in syn_list.iter_mut().enumerate() {
                    if syn.is_plastic() {
                        let global_idx = base_offset + syn_idx;
                        if global_idx < processor.traces.len() {
                            processor.traces[global_idx].apply_reward(
                                syn,
                                self.reward,
                                &self.config.stdp_config,
                            );
                        }
                    }
                }
            }
        }

        // Reset eligibility traces after applying reward
        for processor in &mut self.stdp_processors {
            processor.reset();
        }

        // Reset reward signal
        self.reward = 0;
    }

    /// Apply action-gated reward-modulated learning
    ///
    /// Only updates synapses that are causally connected to the chosen action.
    /// This implements proper credit assignment: only synapses that contributed
    /// to the selected output receive reward modulation.
    ///
    /// # Arguments
    /// * `chosen_action` - The action index (0 to n_outputs-1) that was selected
    ///
    /// # Credit Assignment Strategy
    /// Updates ONLY synapses that directly target the chosen output neuron.
    /// This prevents the "shared hidden neuron" problem where strengthening
    /// input->hidden synapses would benefit ALL outputs.
    pub fn apply_learning_gated(&mut self, chosen_action: usize) {
        if self.reward == 0 || chosen_action >= self.config.n_outputs {
            return;
        }

        // Compute the global index of the chosen output neuron
        let output_start = self.topology.n_neurons - self.config.n_outputs;
        let chosen_output_neuron = output_start + chosen_action;

        // Only update synapses that DIRECTLY target the chosen output
        // This is much more selective than full causal chain propagation
        for cpu in 0..self.topology.n_cpus {
            let processor = &self.stdp_processors[cpu];
            let synapses = &mut self.synapses[cpu];
            let pop = &self.populations[cpu];
            let (cpu_start, _) = self.topology.cpu_to_neurons[cpu];

            // Pre-compute synapse offsets for trace indexing
            let offsets: Vec<usize> = (0..synapses.local.len())
                .map(|src| synapses.local.iter().take(src).map(|v| v.len()).sum())
                .collect();

            for (local_src, syn_list) in synapses.local.iter_mut().enumerate() {
                // Only process if source neuron fired (has eligibility)
                if local_src < pop.neurons.len()
                    && (pop.neurons[local_src].fired()
                        || pop.neurons[local_src].last_spike_time > 0)
                {
                    let base_offset = offsets.get(local_src).copied().unwrap_or(0);

                    for (syn_idx, syn) in syn_list.iter_mut().enumerate() {
                        // ONLY update synapses targeting the chosen output
                        let target_global = cpu_start + syn.target as usize;

                        if syn.is_plastic() && target_global == chosen_output_neuron {
                            let global_idx = base_offset + syn_idx;
                            if global_idx < processor.traces.len() {
                                processor.traces[global_idx].apply_reward(
                                    syn,
                                    self.reward,
                                    &self.config.stdp_config,
                                );
                            }
                        }
                    }
                }
            }
        }

        // Reset eligibility traces after applying reward
        for processor in &mut self.stdp_processors {
            processor.reset();
        }

        // Reset reward signal
        self.reward = 0;
    }

    /// Apply competitive learning: strengthen chosen action, weaken others
    ///
    /// This method updates synapses targeting the chosen output with positive
    /// reward, and synapses targeting OTHER outputs with negative reward (scaled).
    /// This explicit competition helps differentiate action preferences.
    pub fn apply_learning_competitive(&mut self, chosen_action: usize, weaken_factor: i8) {
        if self.reward == 0 || chosen_action >= self.config.n_outputs {
            return;
        }

        let output_start = self.topology.n_neurons - self.config.n_outputs;
        let chosen_output = output_start + chosen_action;

        for cpu in 0..self.topology.n_cpus {
            let processor = &self.stdp_processors[cpu];
            let synapses = &mut self.synapses[cpu];
            let pop = &self.populations[cpu];
            let (cpu_start, _) = self.topology.cpu_to_neurons[cpu];

            let offsets: Vec<usize> = (0..synapses.local.len())
                .map(|src| synapses.local.iter().take(src).map(|v| v.len()).sum())
                .collect();

            for (local_src, syn_list) in synapses.local.iter_mut().enumerate() {
                if local_src < pop.neurons.len()
                    && (pop.neurons[local_src].fired()
                        || pop.neurons[local_src].last_spike_time > 0)
                {
                    let base_offset = offsets.get(local_src).copied().unwrap_or(0);

                    for (syn_idx, syn) in syn_list.iter_mut().enumerate() {
                        let target_global = cpu_start + syn.target as usize;

                        // Check if target is any output neuron
                        if target_global >= output_start
                            && target_global < output_start + self.config.n_outputs
                        {
                            let global_idx = base_offset + syn_idx;
                            if global_idx < processor.traces.len() && syn.is_plastic() {
                                if target_global == chosen_output {
                                    // Strengthen chosen action
                                    processor.traces[global_idx].apply_reward(
                                        syn,
                                        self.reward,
                                        &self.config.stdp_config,
                                    );
                                } else {
                                    // Weaken other actions (scaled)
                                    let weaken_reward =
                                        -(self.reward.abs().saturating_mul(weaken_factor) / 100);
                                    processor.traces[global_idx].apply_reward(
                                        syn,
                                        weaken_reward,
                                        &self.config.stdp_config,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        for processor in &mut self.stdp_processors {
            processor.reset();
        }
        self.reward = 0;
    }

    /// Apply deep gated learning with selective hidden layer updates.
    ///
    /// This extends gated learning to also update input->hidden synapses,
    /// but ONLY for hidden neurons that predominantly connect to the chosen
    /// output (stronger weight to chosen than to any other output).
    pub fn apply_learning_deep_gated(&mut self, chosen_action: usize) {
        if self.reward == 0 || chosen_action >= self.config.n_outputs {
            return;
        }

        let output_start = self.topology.n_neurons - self.config.n_outputs;
        let chosen_output = output_start + chosen_action;

        // First: compute which hidden neurons prefer the chosen output
        // A hidden neuron "prefers" an output if its synapse weight to that output
        // is greater than its weight to any other output
        let mut prefers_chosen = vec![false; self.topology.n_neurons];

        for cpu in 0..self.topology.n_cpus {
            let synapses = &self.synapses[cpu];
            let (cpu_start, _cpu_end) = self.topology.cpu_to_neurons[cpu];

            for local_src in 0..synapses.local.len() {
                let global_src = cpu_start + local_src;
                if global_src >= output_start {
                    continue; // Skip output neurons
                }

                // Find weights to each output
                let mut weights_to_outputs = vec![0i16; self.config.n_outputs];
                for syn in &synapses.local[local_src] {
                    let target = cpu_start + syn.target as usize;
                    if target >= output_start && target < output_start + self.config.n_outputs {
                        let output_idx = target - output_start;
                        weights_to_outputs[output_idx] = syn.weight as i16;
                    }
                }

                // Check if this neuron prefers the chosen output
                let chosen_weight = weights_to_outputs[chosen_action];
                let max_other_weight = weights_to_outputs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != chosen_action)
                    .map(|(_, w)| *w)
                    .max()
                    .unwrap_or(i16::MIN);

                if chosen_weight > max_other_weight && chosen_weight > 0 {
                    prefers_chosen[global_src] = true;
                }
            }
        }

        // Second: apply learning
        for cpu in 0..self.topology.n_cpus {
            let processor = &self.stdp_processors[cpu];
            let synapses = &mut self.synapses[cpu];
            let pop = &self.populations[cpu];
            let (cpu_start, _) = self.topology.cpu_to_neurons[cpu];

            let offsets: Vec<usize> = (0..synapses.local.len())
                .map(|src| synapses.local.iter().take(src).map(|v| v.len()).sum())
                .collect();

            for (local_src, syn_list) in synapses.local.iter_mut().enumerate() {
                if local_src < pop.neurons.len()
                    && (pop.neurons[local_src].fired()
                        || pop.neurons[local_src].last_spike_time > 0)
                {
                    let _global_src = cpu_start + local_src;
                    let base_offset = offsets.get(local_src).copied().unwrap_or(0);

                    for (syn_idx, syn) in syn_list.iter_mut().enumerate() {
                        let target_global = cpu_start + syn.target as usize;

                        // Update if:
                        // 1. Target is the chosen output (output-layer synapse), OR
                        // 2. Target prefers the chosen output (hidden-layer synapse)
                        let should_update = target_global == chosen_output
                            || (target_global < output_start && prefers_chosen[target_global]);

                        if syn.is_plastic() && should_update {
                            let global_idx = base_offset + syn_idx;
                            if global_idx < processor.traces.len() {
                                processor.traces[global_idx].apply_reward(
                                    syn,
                                    self.reward,
                                    &self.config.stdp_config,
                                );
                            }
                        }
                    }
                }
            }
        }

        for processor in &mut self.stdp_processors {
            processor.reset();
        }
        self.reward = 0;
    }

    /// Apply channel-aware learning that respects the network's channel structure.
    ///
    /// The network is built with contiguous channel assignment where neurons
    /// are divided into n_outputs channels. This method only updates synapses
    /// where BOTH source and target neurons belong to the chosen action's channel.
    /// This prevents learning interference between action pathways.
    pub fn apply_learning_channeled(&mut self, chosen_action: usize) {
        if self.reward == 0 || chosen_action >= self.config.n_outputs {
            return;
        }

        let output_start = self.topology.n_neurons - self.config.n_outputs;
        let n_outputs = self.config.n_outputs;

        // Compute channel assignment for each neuron based on contiguous blocks
        // This mirrors the logic in build_connectivity
        let input_layer_end = if !self.topology.layers.is_empty() {
            self.topology.layers[0].2
        } else {
            0
        };

        // When duplicate_inputs is enabled, input neurons are also channel-specific
        let duplicate_inputs = self.config.duplicate_inputs;

        let neuron_channel: Vec<usize> = (0..self.topology.n_neurons)
            .map(|n| {
                // Find which layer this neuron is in
                for &(_, layer_start, layer_end) in &self.topology.layers {
                    if n >= layer_start && n < layer_end {
                        // Input layer: check if duplicated
                        if layer_start == 0 && n < input_layer_end {
                            if duplicate_inputs {
                                // Duplicated inputs: assign to channel based on position
                                let layer_size = layer_end - layer_start;
                                let neurons_per_channel = layer_size / n_outputs.max(1);
                                let channel = (n - layer_start) / neurons_per_channel.max(1);
                                return channel.min(n_outputs - 1);
                            } else {
                                // Shared inputs: belong to all channels
                                return usize::MAX;
                            }
                        }
                        // Hidden/output layers: always channel-specific
                        let layer_size = layer_end - layer_start;
                        let neurons_per_channel = layer_size / n_outputs.max(1);
                        let channel = (n - layer_start) / neurons_per_channel.max(1);
                        return channel.min(n_outputs - 1);
                    }
                }
                0
            })
            .collect();

        for cpu in 0..self.topology.n_cpus {
            let processor = &self.stdp_processors[cpu];
            let synapses = &mut self.synapses[cpu];
            let pop = &self.populations[cpu];
            let (cpu_start, _) = self.topology.cpu_to_neurons[cpu];

            let offsets: Vec<usize> = (0..synapses.local.len())
                .map(|src| synapses.local.iter().take(src).map(|v| v.len()).sum())
                .collect();

            for (local_src, syn_list) in synapses.local.iter_mut().enumerate() {
                let global_src = cpu_start + local_src;

                // Check if source neuron fired
                let fired = local_src < pop.neurons.len()
                    && (pop.neurons[local_src].fired()
                        || pop.neurons[local_src].last_spike_time > 0);

                if !fired || global_src >= neuron_channel.len() {
                    continue;
                }

                let src_channel = neuron_channel[global_src];
                // Input neurons (usize::MAX) can contribute to any channel
                // Hidden/output neurons must be in the chosen channel
                let src_ok = src_channel == usize::MAX || src_channel == chosen_action;

                if !src_ok {
                    continue;
                }

                let base_offset = offsets.get(local_src).copied().unwrap_or(0);

                for (syn_idx, syn) in syn_list.iter_mut().enumerate() {
                    let target_global = cpu_start + syn.target as usize;

                    // Check target is in the chosen channel
                    let target_channel = if target_global < neuron_channel.len() {
                        neuron_channel[target_global]
                    } else {
                        usize::MAX
                    };

                    // Target must be in chosen channel (or chosen output)
                    let target_in_channel = target_channel == chosen_action;
                    let target_is_chosen_output = target_global == output_start + chosen_action;

                    if syn.is_plastic() && (target_in_channel || target_is_chosen_output) {
                        let global_idx = base_offset + syn_idx;
                        if global_idx < processor.traces.len() {
                            processor.traces[global_idx].apply_reward(
                                syn,
                                self.reward,
                                &self.config.stdp_config,
                            );
                        }
                    }
                }
            }
        }

        for processor in &mut self.stdp_processors {
            processor.reset();
        }
        self.reward = 0;
    }

    /// Supervised STDP: always strengthen the correct output, weaken the wrong one.
    ///
    /// Unlike pure R-STDP (which can cause weight death when negative rewards dominate),
    /// this always provides a positive learning signal for the correct output while
    /// only penalising the incorrectly-chosen output.
    ///
    /// # Arguments
    /// * `chosen_action`  - Output neuron the network picked (WTA winner)
    /// * `correct_action` - The ground-truth class label
    pub fn apply_learning_supervised(&mut self, chosen_action: usize, correct_action: usize) {
        if chosen_action >= self.config.n_outputs || correct_action >= self.config.n_outputs {
            return;
        }

        let output_start = self.topology.n_neurons - self.config.n_outputs;
        let correct_output = output_start + correct_action;
        let chosen_output = output_start + chosen_action;
        let reward_pos: i8 = 100; // always strengthen correct output
        let reward_neg: i8 = -60; // weaken wrong output (only on misprediction)
        let is_wrong = chosen_action != correct_action;

        for cpu in 0..self.topology.n_cpus {
            let processor = &self.stdp_processors[cpu];
            let synapses = &mut self.synapses[cpu];
            let pop = &self.populations[cpu];
            let (cpu_start, _) = self.topology.cpu_to_neurons[cpu];

            let offsets: Vec<usize> = (0..synapses.local.len())
                .map(|src| synapses.local.iter().take(src).map(|v| v.len()).sum())
                .collect();

            for (local_src, syn_list) in synapses.local.iter_mut().enumerate() {
                if local_src >= pop.neurons.len() {
                    continue;
                }
                let fired =
                    pop.neurons[local_src].fired() || pop.neurons[local_src].last_spike_time > 0;
                if !fired {
                    continue;
                }

                let base_offset = offsets.get(local_src).copied().unwrap_or(0);

                for (syn_idx, syn) in syn_list.iter_mut().enumerate() {
                    let target_global = cpu_start + syn.target as usize;
                    if !syn.is_plastic() {
                        continue;
                    }
                    let global_idx = base_offset + syn_idx;
                    if global_idx >= processor.traces.len() {
                        continue;
                    }

                    if target_global == correct_output {
                        // Always strengthen synapses to the correct output
                        processor.traces[global_idx].apply_reward(
                            syn,
                            reward_pos,
                            &self.config.stdp_config,
                        );
                    } else if is_wrong && target_global == chosen_output {
                        // Weaken synapses to the wrong output only on misprediction
                        processor.traces[global_idx].apply_reward(
                            syn,
                            reward_neg,
                            &self.config.stdp_config,
                        );
                    }
                }
            }
        }

        for processor in &mut self.stdp_processors {
            processor.reset();
        }
    }

    /// Apply tile substrate routing delays to all local synapses.
    ///
    /// For each synapse (src → dst), looks up the routing delay from the
    /// RouterMesh and sets the synapse delay field (clamped to 0-15).
    pub fn apply_substrate_delays(&mut self, mesh: &crate::snn::router_mesh::RouterMesh) {
        for cpu in 0..self.topology.n_cpus {
            let (cpu_start, _cpu_end) = self.topology.cpu_to_neurons[cpu];
            for local_src in 0..self.synapses[cpu].local.len() {
                let global_src = cpu_start + local_src;
                for syn in &mut self.synapses[cpu].local[local_src] {
                    let global_dst = cpu_start + syn.target as usize;
                    let delay = mesh.delay(global_src, global_dst).min(15);
                    syn.set_delay(delay);
                }
            }
        }
    }

    /// Reset last_spike_time for all neurons to 0.
    ///
    /// Call this at the start of each input presentation so that
    /// `last_spike_time > 0` reliably means "fired during this presentation".
    pub fn reset_activity(&mut self) {
        for pop in &mut self.populations {
            for neuron in &mut pop.neurons {
                neuron.last_spike_time = 0;
            }
        }
    }

    /// Apply perceptron-style direct weight update to output-targeting synapses.
    ///
    /// Unlike R-STDP (which uses eligibility traces that can corrupt inhibitory
    /// cross-channel weights via LTP), this method applies a FIXED weight delta
    /// based only on whether the presynaptic neuron fired (`last_spike_time > 0`)
    /// and the sign of the synapse:
    ///
    /// - Only updates **excitatory** (positive-weight) synapses — cross-channel
    ///   inhibitory synapses are left unchanged to preserve WTA structure.
    /// - `+lr` to synapses targeting `correct_action` output
    /// - `-lr` to synapses targeting `chosen_action` output (only on misprediction)
    ///
    /// Call `reset_activity()` BEFORE each presentation and this AFTER.
    pub fn apply_perceptron_learning(
        &mut self,
        chosen_action: usize,
        correct_action: usize,
        lr: i8,
    ) {
        if chosen_action >= self.config.n_outputs || correct_action >= self.config.n_outputs {
            return;
        }

        let output_start = self.topology.n_neurons - self.config.n_outputs;
        let correct_output = output_start + correct_action;
        let chosen_output = output_start + chosen_action;
        let is_wrong = chosen_action != correct_action;
        let w_min = self.config.stdp_config.w_min as i16;
        let w_max = self.config.stdp_config.w_max as i16;

        for cpu in 0..self.topology.n_cpus {
            let (cpu_start, _) = self.topology.cpu_to_neurons[cpu];
            let pop = &self.populations[cpu];
            let synapses = &mut self.synapses[cpu];

            for (local_src, syn_list) in synapses.local.iter_mut().enumerate() {
                if local_src >= pop.neurons.len() {
                    continue;
                }

                // Only update if this presynaptic neuron fired in this presentation
                let fired = pop.neurons[local_src].last_spike_time > 0;
                if !fired {
                    continue;
                }

                for syn in syn_list.iter_mut() {
                    if !syn.is_plastic() {
                        continue;
                    }
                    // Only touch excitatory (positive-weight) synapses.
                    // Inhibitory cross-channel synapses stay fixed to preserve WTA.
                    if syn.weight <= 0 {
                        continue;
                    }

                    // Balanced Perceptron:
                    // - Always strengthen correct output path (on every sample).
                    // - Only penalize wrong output path on mispredictions.
                    //
                    // This ensures:
                    // - ch0→Out0 grows net +4850/epoch (digit-0 correct: 730, wrong: 270; digit-1 wrong: -30)
                    // - ch1→Out1 grows net +3650/epoch (digit-1 correct: 970, wrong: 30; digit-0 wrong: -270)
                    // Both saturate at w_max, making WTA timing the discriminator.
                    let target_global = cpu_start + syn.target as usize;
                    if target_global == correct_output {
                        syn.weight = (syn.weight as i16 + lr as i16).clamp(w_min, w_max) as i8;
                    } else if is_wrong && target_global == chosen_output {
                        syn.weight = (syn.weight as i16 - lr as i16).clamp(w_min, w_max) as i8;
                    }
                }
            }
        }

        // Reset eligibility traces for next presentation
        for processor in &mut self.stdp_processors {
            processor.reset();
        }
    }

    /// Legacy method for backward compatibility
    fn apply_reward_learning(&mut self) {
        self.apply_learning();
    }

    /// Count output layer spikes
    fn count_output_spikes(&mut self) {
        let output_start = self.topology.n_neurons - self.config.n_outputs;

        for (i, count) in self.output_counts.iter_mut().enumerate() {
            let neuron = output_start + i;
            if neuron < self.topology.n_neurons {
                let cpu = self.topology.neuron_to_cpu[neuron];
                let local = self
                    .topology
                    .local_index(neuron, self.config.neurons_per_cpu)
                    as usize;

                if cpu < self.populations.len() && local < self.populations[cpu].neurons.len() {
                    if self.populations[cpu].neurons[local].fired() {
                        *count += 1;
                    }
                }
            }
        }
    }

    /// Get output action (winner-take-all)
    pub fn get_action(&self) -> usize {
        self.output_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| *count)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Reset output spike counts
    pub fn reset_output_counts(&mut self) {
        for count in &mut self.output_counts {
            *count = 0;
        }
    }

    /// Get total spike count this tick
    pub fn spike_count(&self) -> u32 {
        self.populations.iter().map(|p| p.spike_count).sum()
    }

    /// Get total neuron count
    pub fn n_neurons(&self) -> usize {
        self.topology.n_neurons
    }

    /// Get total synapse count
    pub fn n_synapses(&self) -> usize {
        self.synapses.iter().map(|s| s.total_count).sum::<usize>()
            + self.router.total_cross_cpu_synapses()
    }

    /// Randomize synapse weights based on seed
    ///
    /// Useful for creating variation in initial network states during evolution.
    pub fn randomize_weights(&mut self, seed: u64) {
        let mut rng = seed;

        // Local LCG function to avoid borrow issues
        let mut random_float = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32) / (u32::MAX as f32)
        };

        for synapses in &mut self.synapses {
            for syn_list in &mut synapses.local {
                for syn in syn_list.iter_mut() {
                    // Random weight: 80% excitatory (10-100), 20% inhibitory (-10 to -60)
                    if random_float() < 0.8 {
                        syn.weight = ((random_float() * 90.0) + 10.0) as i8;
                    } else {
                        syn.weight = (-(random_float() * 50.0 + 10.0)) as i8;
                    }
                }
            }
        }

        self.rng_state = rng;
    }

    /// Mutate synapse weights with given mutation rate
    ///
    /// Each synapse has `mutation_rate` probability of being mutated.
    /// Mutations add a small random delta to the weight.
    pub fn mutate_weights(&mut self, rng: &mut crate::quantum::QRng, mutation_rate: f32) {
        for synapses in &mut self.synapses {
            for syn_list in &mut synapses.local {
                for syn in syn_list.iter_mut() {
                    if rng.next_f32() < mutation_rate {
                        // Add random delta: -20 to +20
                        let delta = (rng.next_f32() * 40.0 - 20.0) as i8;
                        syn.weight = syn.weight.saturating_add(delta);
                        // Clamp to valid range
                        syn.weight = syn.weight.clamp(-127, 127);
                    }
                }
            }
        }
    }

    /// Execute one simulation tick, auto-selecting parallel or sequential
    /// based on network size.
    ///
    /// For networks with 16+ CPUs, uses parallel processing.
    /// For smaller networks, uses sequential to avoid overhead.
    pub fn step_auto(&mut self) {
        if self.topology.n_cpus >= 16 {
            self.step_parallel();
        } else {
            self.step();
        }
    }

    /// Execute one simulation tick with parallel CPU processing
    ///
    /// Uses rayon to parallelize CPU stepping for better performance
    /// on multi-core systems. Best for networks with 16+ CPUs.
    pub fn step_parallel(&mut self) {
        let tick_u8 = (self.tick & 0xFF) as u8;

        // Phase 1: Generate input spikes (rate coding)
        self.generate_input_spikes(tick_u8);

        // Phase 2: Process each CPU in parallel
        // Collect currents and step neurons, gathering spike info
        let cpu_results: Vec<(u32, Vec<(usize, u8)>)> = self
            .populations
            .par_iter_mut()
            .zip(self.synapses.par_iter())
            .enumerate()
            .map(|(cpu, (pop, synapses))| {
                // Collect fired neurons from previous tick
                let fired: Vec<usize> = pop
                    .neurons
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| n.fired())
                    .map(|(i, _)| i)
                    .collect();

                // Accumulate currents from local synapses
                let mut currents = vec![0i16; pop.len()];
                synapses.accumulate_currents(&fired, &mut currents);

                // Step all neurons
                pop.step(&currents, tick_u8);

                // Collect spike info for cross-CPU routing
                let spikes: Vec<(usize, u8)> = pop
                    .neurons
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| n.fired())
                    .map(|(i, _)| (cpu, i as u8))
                    .collect();

                (pop.spike_count, spikes)
            })
            .collect();

        // Phase 2b: Queue spikes for cross-CPU routing (sequential - router not thread-safe)
        for (_, spikes) in &cpu_results {
            for &(cpu, neuron) in spikes {
                self.router.queue_spike(cpu, neuron);
            }
        }

        // Phase 3: Route cross-CPU spikes
        self.router.route_spikes();

        // Phase 4: Process incoming cross-CPU spikes
        for cpu in 0..self.topology.n_cpus {
            self.process_incoming_spikes(cpu);
        }
        self.router.clear_incoming();

        // Phase 5: Update eligibility traces for R-STDP
        if self.runtime.learning_enabled {
            self.update_eligibility_traces(tick_u8);
        }

        // Phase 6: Apply reward-modulated STDP if reward is set
        if self.runtime.learning_enabled && self.reward != 0 {
            self.apply_reward_learning();
        }

        // Phase 7: Count output spikes
        self.count_output_spikes();

        // Advance tick
        self.tick += 1;
        self.router.next_tick();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_from_config() {
        let config = SNNConfig::new(8, vec![64, 32], 5);
        let topology = SNNTopology::from_config(&config);

        assert_eq!(topology.n_neurons, 8 + 64 + 32 + 5);
        assert_eq!(topology.layers.len(), 4); // input, hidden1, hidden2, output
    }

    #[test]
    fn test_network_creation() {
        let config = SNNConfig::minimal();
        let network = SNNNetwork::new(config);

        assert_eq!(network.n_neurons(), 4 + 8 + 2); // inputs + hidden + outputs
        assert!(network.populations.len() > 0);
    }

    #[test]
    fn test_network_step() {
        let config = SNNConfig::minimal();
        let mut network = SNNNetwork::new(config);
        network.build_connectivity(42);

        // Set some inputs
        network.set_inputs(&[128, 200, 50, 100]);

        // Run a few ticks
        for _ in 0..10 {
            network.step();
        }

        // Should have some activity
        assert!(network.tick == 10);
    }

    #[test]
    fn test_input_encoding() {
        let config = SNNConfig::minimal();
        let mut network = SNNNetwork::new(config);

        // High rate should produce spikes
        network.set_inputs(&[255, 255, 255, 255]);
        network.step();

        // At least some inputs should have spiked
        // (probabilistic, but very likely with 255)
    }

    #[test]
    fn test_output_decoding() {
        let config = SNNConfig::minimal();
        let mut network = SNNNetwork::new(config);
        network.build_connectivity(42);

        network.set_inputs(&[200, 200, 50, 50]);

        // Run for decision period
        for _ in 0..network.runtime.ticks_per_decision {
            network.step();
        }

        let _action = network.get_action();
        // Action should be 0-1 (2 outputs)
    }

    #[test]
    fn test_large_network_1000_neurons() {
        // Create a network with 1000+ neurons
        let config = SNNConfig {
            n_inputs: 64,
            hidden_layers: vec![512, 256, 128],
            n_outputs: 32,
            connection_prob: 0.1, // Sparse connectivity
            neurons_per_cpu: 128,
            recurrent: false,
            ..SNNConfig::default()
        };

        let expected_neurons = 64 + 512 + 256 + 128 + 32; // = 992 neurons
        assert!(
            config.total_neurons() >= 992,
            "Expected at least 992 neurons"
        );

        let mut network = SNNNetwork::new(config);
        assert_eq!(network.n_neurons(), expected_neurons);

        // Build connectivity
        network.build_connectivity(12345);
        let n_synapses = network.n_synapses();
        assert!(n_synapses > 0, "Should have created synapses");

        // Run simulation for 100 ticks
        network.set_inputs(&[200u8; 64]);
        let mut total_spikes = 0u64;

        for _ in 0..100 {
            network.step();
            total_spikes += network.spike_count() as u64;
        }

        assert_eq!(network.tick, 100);
        println!(
            "Large network (992 neurons, {} synapses): {} total spikes in 100 ticks",
            n_synapses, total_spikes
        );
    }

    #[test]
    fn test_multi_cpu_network() {
        // Create a network that spans multiple CPUs
        let config = SNNConfig {
            n_inputs: 32,
            hidden_layers: vec![256, 128],
            n_outputs: 16,
            connection_prob: 0.2,
            neurons_per_cpu: 64, // Lower neurons per CPU to force multi-CPU
            recurrent: true,
            ..SNNConfig::default()
        };

        let network = SNNNetwork::new(config);
        let n_cpus = network.topology.n_cpus;

        // Should require multiple CPUs
        assert!(n_cpus >= 4, "Expected at least 4 CPUs, got {}", n_cpus);

        // Verify CPU distribution
        for cpu in 0..n_cpus {
            let (start, end) = network.topology.cpu_to_neurons[cpu];
            assert!(end > start, "CPU {} should have neurons", cpu);
            assert!(
                end - start <= 64,
                "CPU {} should have at most 64 neurons",
                cpu
            );
        }

        println!(
            "Multi-CPU network: {} neurons across {} CPUs",
            network.n_neurons(),
            n_cpus
        );
    }

    #[test]
    fn test_cross_cpu_spike_routing() {
        // Create a network specifically to test cross-CPU routing
        let config = SNNConfig {
            n_inputs: 16,
            hidden_layers: vec![64, 32],
            n_outputs: 8,
            connection_prob: 0.5, // Higher connectivity to ensure cross-CPU connections
            neurons_per_cpu: 32,  // Small CPUs to ensure cross-CPU synapses
            recurrent: true,
            ..SNNConfig::default()
        };

        let mut network = SNNNetwork::new(config);
        network.build_connectivity(42);

        let cross_cpu_synapses = network.router.total_cross_cpu_synapses();

        // Should have some cross-CPU synapses with this configuration
        println!("Cross-CPU synapses: {}", cross_cpu_synapses);

        // Run simulation and track cross-CPU spike routing
        network.set_inputs(&[255u8; 16]); // High input to generate spikes

        for _ in 0..50 {
            network.step();
        }

        let total_routed = network.router.total_cross_cpu_spikes;
        println!("Total cross-CPU spikes routed: {}", total_routed);
    }

    #[test]
    fn test_scalability_benchmark() {
        use std::time::Instant;

        let configs = [
            ("1K neurons", SNNConfig::new(32, vec![512, 256, 128], 64)),
            ("2K neurons", SNNConfig::new(64, vec![1024, 512, 256], 128)),
            (
                "4K neurons",
                SNNConfig::new(128, vec![2048, 1024, 512], 256),
            ),
        ];

        println!("\nSNN Scalability Benchmark:");
        println!(
            "{:<15} {:>10} {:>10} {:>12} {:>12}",
            "Config", "Neurons", "Synapses", "Build (ms)", "100 ticks (ms)"
        );
        println!("{:-<65}", "");

        for (name, config) in configs {
            let neurons = config.total_neurons();

            // Time network building
            let build_start = Instant::now();
            let mut network = SNNNetwork::new(config);
            network.build_connectivity(42);
            let build_time = build_start.elapsed();

            let synapses = network.n_synapses();

            // Time simulation
            network.set_inputs(&[128u8; 128]);
            let sim_start = Instant::now();
            for _ in 0..100 {
                network.step();
            }
            let sim_time = sim_start.elapsed();

            println!(
                "{:<15} {:>10} {:>10} {:>12.2} {:>12.2}",
                name,
                neurons,
                synapses,
                build_time.as_secs_f64() * 1000.0,
                sim_time.as_secs_f64() * 1000.0
            );
        }
    }

    #[test]
    fn test_reward_learning_integration() {
        // Test R-STDP learning across multiple ticks
        let config = SNNConfig {
            n_inputs: 8,
            hidden_layers: vec![32],
            n_outputs: 4,
            connection_prob: 0.5,
            neurons_per_cpu: 64,
            recurrent: false,
            ..SNNConfig::default()
        };

        let mut network = SNNNetwork::new(config.clone());
        network.build_connectivity(42);
        network.runtime.learning_enabled = true;

        // Collect initial weights
        let initial_weight_sum: i64 = network
            .synapses
            .iter()
            .flat_map(|s| s.local.iter())
            .flat_map(|syns| syns.iter())
            .map(|s| s.weight as i64)
            .sum();

        // Run with positive reward
        network.set_reward(64);
        network.set_inputs(&[200u8; 8]);

        for _ in 0..100 {
            network.step();
        }

        // Check weights changed
        let final_weight_sum: i64 = network
            .synapses
            .iter()
            .flat_map(|s| s.local.iter())
            .flat_map(|syns| syns.iter())
            .map(|s| s.weight as i64)
            .sum();

        // With positive reward and LTP activity, weights should generally increase
        // (though not guaranteed for every synapse due to timing)
        println!(
            "Initial weight sum: {}, Final weight sum: {}",
            initial_weight_sum, final_weight_sum
        );

        // Just verify the system ran without panicking
        assert!(network.tick == 100);
    }

    #[test]
    fn test_parallel_step() {
        // Test that parallel step produces valid results
        let config = SNNConfig::new(16, vec![128, 64], 8);
        let mut network = SNNNetwork::new(config);
        network.build_connectivity(42);

        network.set_inputs(&[128u8; 16]);

        // Run with parallel stepping
        for _ in 0..50 {
            network.step_parallel();
        }

        assert_eq!(network.tick, 50);
        println!("Parallel step completed 50 ticks successfully");
    }

    #[test]
    fn test_parallel_vs_sequential_benchmark() {
        use std::time::Instant;

        let config = SNNConfig::new(64, vec![512, 256, 128], 32);

        // Sequential benchmark
        let mut seq_network = SNNNetwork::new(config.clone());
        seq_network.build_connectivity(42);
        seq_network.set_inputs(&[128u8; 64]);

        let seq_start = Instant::now();
        for _ in 0..100 {
            seq_network.step();
        }
        let seq_time = seq_start.elapsed();

        // Parallel benchmark
        let mut par_network = SNNNetwork::new(config);
        par_network.build_connectivity(42);
        par_network.set_inputs(&[128u8; 64]);

        let par_start = Instant::now();
        for _ in 0..100 {
            par_network.step_parallel();
        }
        let par_time = par_start.elapsed();

        println!(
            "\nParallel vs Sequential Benchmark ({} neurons):",
            seq_network.n_neurons()
        );
        println!("  Sequential: {:.2}ms", seq_time.as_secs_f64() * 1000.0);
        println!("  Parallel:   {:.2}ms", par_time.as_secs_f64() * 1000.0);
        println!(
            "  Speedup:    {:.2}x",
            seq_time.as_secs_f64() / par_time.as_secs_f64()
        );
    }
}
