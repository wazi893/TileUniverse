//! Quantum Eligibility Traces (Sprint 59.0 EPIC 137)
//!
//! Bridges spiking neural network STDP with quantum phase evolution.
//!
//! # Key Insight
//!
//! Classical eligibility traces decay exponentially. Quantum states have
//! natural phase evolution. We use quantum phase as a "memory" of past
//! activity, enabling reward-modulated learning with quantum enhancement.
//!
//! # Architecture
//!
//! ```text
//! Pre-spike ──┐
//!             ├──> Eligibility Trace ──> Phase Rotation ──> Cluster CRz
//! Post-spike ─┘
//!               │
//!               ▼
//!           Reward Signal ──> Scale & Apply
//! ```

use std::f64::consts::PI;

/// Configuration for quantum eligibility traces.
#[derive(Clone, Debug)]
pub struct QuantumEligibilityConfig {
    /// Decay rate per tick (0.0 = instant decay, 1.0 = no decay)
    pub decay_rate: f64,

    /// Maximum phase accumulation (radians)
    pub max_phase: f64,

    /// Learning rate for STDP-like updates
    pub learning_rate: f64,

    /// Time window for potentiation (ticks)
    pub potentiation_window: u8,

    /// Time window for depression (ticks)
    pub depression_window: u8,

    /// Coupling strength to quantum cluster
    pub quantum_coupling: f64,
}

impl Default for QuantumEligibilityConfig {
    fn default() -> Self {
        Self {
            decay_rate: 0.95,
            max_phase: PI / 2.0,
            learning_rate: 0.1,
            potentiation_window: 20,
            depression_window: 20,
            quantum_coupling: 1.0,
        }
    }
}

impl QuantumEligibilityConfig {
    /// Fast learning configuration.
    pub fn fast() -> Self {
        Self {
            decay_rate: 0.9,
            learning_rate: 0.2,
            ..Default::default()
        }
    }

    /// Slow learning configuration.
    pub fn slow() -> Self {
        Self {
            decay_rate: 0.99,
            learning_rate: 0.05,
            ..Default::default()
        }
    }

    /// Strong quantum coupling.
    pub fn quantum_heavy() -> Self {
        Self {
            quantum_coupling: 2.0,
            max_phase: PI,
            ..Default::default()
        }
    }
}

/// A single eligibility trace for a synapse.
#[derive(Clone, Debug)]
pub struct EligibilityTrace {
    /// Accumulated phase (radians)
    pub phase: f64,

    /// Source neuron index
    pub source: usize,

    /// Target neuron index
    pub target: usize,

    /// Last pre-synaptic spike time
    last_pre_spike: Option<u64>,

    /// Last post-synaptic spike time
    last_post_spike: Option<u64>,
}

impl EligibilityTrace {
    /// Create a new eligibility trace for a synapse.
    pub fn new(source: usize, target: usize) -> Self {
        Self {
            phase: 0.0,
            source,
            target,
            last_pre_spike: None,
            last_post_spike: None,
        }
    }

    /// Record a pre-synaptic spike.
    pub fn record_pre_spike(&mut self, tick: u64) {
        self.last_pre_spike = Some(tick);
    }

    /// Record a post-synaptic spike.
    pub fn record_post_spike(&mut self, tick: u64) {
        self.last_post_spike = Some(tick);
    }

    /// Get time since last pre-synaptic spike.
    pub fn time_since_pre(&self, current_tick: u64) -> Option<u64> {
        self.last_pre_spike.map(|t| current_tick.saturating_sub(t))
    }

    /// Get time since last post-synaptic spike.
    pub fn time_since_post(&self, current_tick: u64) -> Option<u64> {
        self.last_post_spike.map(|t| current_tick.saturating_sub(t))
    }

    /// Check if this trace is active (non-zero phase).
    pub fn is_active(&self) -> bool {
        self.phase.abs() > 1e-6
    }
}

/// Collection of eligibility traces for a network.
#[derive(Clone, Debug)]
pub struct QuantumEligibilityTraces {
    /// Traces indexed by synapse
    traces: Vec<EligibilityTrace>,

    /// Configuration
    config: QuantumEligibilityConfig,

    /// Current tick
    tick: u64,
}

impl QuantumEligibilityTraces {
    /// Create traces for a set of synapses.
    pub fn new(synapse_pairs: &[(usize, usize)], config: QuantumEligibilityConfig) -> Self {
        let traces = synapse_pairs
            .iter()
            .map(|&(src, tgt)| EligibilityTrace::new(src, tgt))
            .collect();

        Self {
            traces,
            config,
            tick: 0,
        }
    }

    /// Create traces for a network with given synapse count.
    pub fn for_network(n_synapses: usize, config: QuantumEligibilityConfig) -> Self {
        // Placeholder synapses - will be updated when connected to real network
        let traces = (0..n_synapses)
            .map(|i| EligibilityTrace::new(i, i + 1))
            .collect();

        Self {
            traces,
            config,
            tick: 0,
        }
    }

    /// Get number of traces.
    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    /// Get trace by index.
    pub fn get(&self, idx: usize) -> Option<&EligibilityTrace> {
        self.traces.get(idx)
    }

    /// Get mutable trace by index.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut EligibilityTrace> {
        self.traces.get_mut(idx)
    }

    /// Update traces with spike information.
    ///
    /// Call this once per tick with the set of neurons that fired.
    pub fn update(&mut self, fired_neurons: &[usize]) {
        self.tick += 1;

        // Cache config values to avoid borrow issues
        let decay_rate = self.config.decay_rate;
        let learning_rate = self.config.learning_rate;
        let max_phase = self.config.max_phase;
        let potentiation_window = self.config.potentiation_window as i64;
        let depression_window = self.config.depression_window as i64;
        let current_tick = self.tick;

        // Update each trace
        for trace in &mut self.traces {
            // Decay existing phase first
            trace.phase *= decay_rate;

            // Check for new spikes this tick
            let pre_spiked = fired_neurons.contains(&trace.source);
            let post_spiked = fired_neurons.contains(&trace.target);

            // Record new spikes
            if pre_spiked {
                trace.record_pre_spike(current_tick);
            }
            if post_spiked {
                trace.record_post_spike(current_tick);
            }

            // Only apply STDP when a new spike completes a pair
            // (i.e., when the second spike of a pre-post or post-pre pair arrives)
            if (pre_spiked || post_spiked)
                && trace.last_pre_spike.is_some()
                && trace.last_post_spike.is_some()
            {
                let pre_time = trace.last_pre_spike.unwrap();
                let post_time = trace.last_post_spike.unwrap();
                let dt = post_time as i64 - pre_time as i64;

                // Potentiation: pre before post (dt > 0)
                if dt > 0 && dt <= potentiation_window {
                    let strength = Self::stdp_window_static(dt as f64);
                    trace.phase += learning_rate * strength;
                }
                // Depression: post before pre (dt < 0)
                else if dt < 0 && dt >= -depression_window {
                    let strength = Self::stdp_window_static(dt as f64);
                    trace.phase += learning_rate * strength;
                }
            }

            // Clamp phase
            trace.phase = trace.phase.clamp(-max_phase, max_phase);
        }
    }

    /// STDP learning window function (static version).
    ///
    /// Positive dt (pre before post) -> positive (potentiation)
    /// Negative dt (post before pre) -> negative (depression)
    fn stdp_window_static(dt: f64) -> f64 {
        let tau = 20.0; // Time constant
        if dt > 0.0 {
            // Pre before post: potentiation
            (-dt / tau).exp()
        } else {
            // Post before pre: depression (weaker)
            -0.5 * (dt / tau).exp()
        }
    }

    /// Apply reward modulation to all traces.
    pub fn apply_reward(&mut self, reward: f64) {
        let scale = reward.clamp(-1.0, 1.0);
        for trace in &mut self.traces {
            trace.phase *= 1.0 + scale * 0.5; // Modulate by reward
        }
    }

    /// Get phases for all traces.
    pub fn get_phases(&self) -> Vec<f64> {
        self.traces.iter().map(|t| t.phase).collect()
    }

    /// Get active trace count.
    pub fn active_count(&self) -> usize {
        self.traces.iter().filter(|t| t.is_active()).count()
    }

    /// Get total phase magnitude.
    pub fn total_phase_magnitude(&self) -> f64 {
        self.traces.iter().map(|t| t.phase.abs()).sum()
    }

    /// Reset all traces.
    pub fn reset(&mut self) {
        for trace in &mut self.traces {
            trace.phase = 0.0;
            trace.last_pre_spike = None;
            trace.last_post_spike = None;
        }
        self.tick = 0;
    }

    /// Get configuration.
    pub fn config(&self) -> &QuantumEligibilityConfig {
        &self.config
    }

    /// Get current tick.
    pub fn tick(&self) -> u64 {
        self.tick
    }
}

/// Applies eligibility traces to quantum clusters via controlled rotations.
pub struct QuantumTraceApplicator {
    /// Mapping from trace index to cluster qubit pairs
    /// (trace_idx, cluster_id, control_qubit, target_qubit)
    mappings: Vec<(usize, usize, usize, usize)>,
}

impl QuantumTraceApplicator {
    /// Create an applicator with explicit mappings.
    pub fn new(mappings: Vec<(usize, usize, usize, usize)>) -> Self {
        Self { mappings }
    }

    /// Create an applicator that maps traces to cluster in order.
    pub fn linear(n_traces: usize, cluster_id: usize, n_qubits: usize) -> Self {
        let mappings = (0..n_traces.min(n_qubits - 1))
            .map(|i| (i, cluster_id, i, (i + 1) % n_qubits))
            .collect();
        Self { mappings }
    }

    /// Get the controlled rotation angles for each mapping.
    pub fn get_rotations(
        &self,
        traces: &QuantumEligibilityTraces,
    ) -> Vec<(usize, usize, usize, f64)> {
        self.mappings
            .iter()
            .filter_map(|&(trace_idx, cluster_id, ctrl, tgt)| {
                traces.get(trace_idx).map(|trace| {
                    let angle = trace.phase * traces.config().quantum_coupling;
                    (cluster_id, ctrl, tgt, angle)
                })
            })
            .filter(|(_, _, _, angle)| angle.abs() > 1e-6) // Skip negligible rotations
            .collect()
    }
}

/// Integration helper for HybridQuantumNetwork.
pub mod integration {
    use super::*;

    /// Create eligibility traces from a synapse list.
    pub fn create_traces_from_synapses<S: AsRef<[(usize, usize)]>>(
        synapses: S,
        config: QuantumEligibilityConfig,
    ) -> QuantumEligibilityTraces {
        QuantumEligibilityTraces::new(synapses.as_ref(), config)
    }

    /// Compute phase update based on pre/post spike times.
    pub fn compute_phase_delta(
        pre_spike_tick: u64,
        post_spike_tick: u64,
        current_tick: u64,
        learning_rate: f64,
    ) -> f64 {
        let dt = post_spike_tick as i64 - pre_spike_tick as i64;
        let age_pre = current_tick.saturating_sub(pre_spike_tick) as f64;
        let age_post = current_tick.saturating_sub(post_spike_tick) as f64;

        // Weight by recency
        let recency_factor = (-(age_pre + age_post) / 40.0).exp();

        let tau = 20.0;
        let stdp = if dt > 0 {
            (-dt as f64 / tau).exp()
        } else {
            -0.5 * (dt as f64 / tau).exp()
        };

        learning_rate * stdp * recency_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_creation() {
        let trace = EligibilityTrace::new(0, 1);
        assert_eq!(trace.source, 0);
        assert_eq!(trace.target, 1);
        assert!(!trace.is_active());
    }

    #[test]
    fn test_trace_spike_recording() {
        let mut trace = EligibilityTrace::new(0, 1);

        trace.record_pre_spike(10);
        trace.record_post_spike(15);

        assert_eq!(trace.time_since_pre(20), Some(10));
        assert_eq!(trace.time_since_post(20), Some(5));
    }

    #[test]
    fn test_traces_creation() {
        let pairs = vec![(0, 1), (1, 2), (2, 3)];
        let traces = QuantumEligibilityTraces::new(&pairs, QuantumEligibilityConfig::default());

        assert_eq!(traces.len(), 3);
        assert_eq!(traces.get(0).unwrap().source, 0);
        assert_eq!(traces.get(0).unwrap().target, 1);
    }

    #[test]
    fn test_stdp_potentiation() {
        let pairs = vec![(0, 1)];
        let mut traces = QuantumEligibilityTraces::new(&pairs, QuantumEligibilityConfig::default());

        // Pre-synaptic spike first
        traces.update(&[0]);
        for _ in 0..5 {
            traces.update(&[]);
        }
        // Post-synaptic spike after
        traces.update(&[1]);

        // Should have positive phase (potentiation)
        let phase = traces.get(0).unwrap().phase;
        assert!(phase > 0.0, "Expected positive phase, got {}", phase);
    }

    #[test]
    fn test_stdp_depression() {
        let pairs = vec![(0, 1)];
        let mut traces = QuantumEligibilityTraces::new(&pairs, QuantumEligibilityConfig::default());

        // Post-synaptic spike first
        traces.update(&[1]);
        for _ in 0..5 {
            traces.update(&[]);
        }
        // Pre-synaptic spike after
        traces.update(&[0]);

        // Should have negative phase (depression)
        let phase = traces.get(0).unwrap().phase;
        assert!(phase < 0.0, "Expected negative phase, got {}", phase);
    }

    #[test]
    fn test_decay() {
        let pairs = vec![(0, 1)];
        let config = QuantumEligibilityConfig {
            decay_rate: 0.5,
            ..Default::default()
        };
        let mut traces = QuantumEligibilityTraces::new(&pairs, config);

        // Create phase with proper STDP timing (pre before post with gap)
        traces.update(&[0]); // Pre-spike at tick 1
        for _ in 0..3 {
            traces.update(&[]); // Wait
        }
        traces.update(&[1]); // Post-spike at tick 5 (dt = 4 > 0)

        let initial_phase = traces.get(0).unwrap().phase;
        assert!(
            initial_phase > 0.0,
            "Should have positive phase from potentiation, got {}",
            initial_phase
        );

        // Decay over several ticks
        for _ in 0..10 {
            traces.update(&[]);
        }

        let final_phase = traces.get(0).unwrap().phase;

        // Should have decayed significantly with decay_rate = 0.5
        // After 10 ticks: final = initial * 0.5^10 ≈ initial * 0.001
        assert!(
            final_phase.abs() < initial_phase.abs() * 0.1,
            "Phase should decay: initial={}, final={}",
            initial_phase,
            final_phase
        );
    }

    #[test]
    fn test_reward_modulation() {
        let pairs = vec![(0, 1)];
        let mut traces = QuantumEligibilityTraces::new(&pairs, QuantumEligibilityConfig::default());

        // Create some phase
        traces.update(&[0]);
        traces.update(&[1]);

        let before = traces.get(0).unwrap().phase;

        // Apply positive reward
        traces.apply_reward(1.0);

        let after = traces.get(0).unwrap().phase;

        // Positive reward should increase phase magnitude
        assert!(after.abs() > before.abs(), "Reward should enhance phase");
    }

    #[test]
    fn test_applicator_linear() {
        let applicator = QuantumTraceApplicator::linear(4, 0, 8);
        assert_eq!(applicator.mappings.len(), 4);
    }

    #[test]
    fn test_applicator_rotations() {
        let pairs = vec![(0, 1), (1, 2)];
        let mut traces = QuantumEligibilityTraces::new(&pairs, QuantumEligibilityConfig::default());

        // Create phase
        traces.update(&[0, 1]);
        traces.update(&[1, 2]);

        let applicator = QuantumTraceApplicator::linear(2, 0, 4);
        let rotations = applicator.get_rotations(&traces);

        // Should have some rotations
        assert!(!rotations.is_empty());
    }

    #[test]
    fn test_config_presets() {
        let fast = QuantumEligibilityConfig::fast();
        let slow = QuantumEligibilityConfig::slow();

        assert!(fast.decay_rate < slow.decay_rate);
        assert!(fast.learning_rate > slow.learning_rate);
    }

    #[test]
    fn test_reset() {
        let pairs = vec![(0, 1)];
        let mut traces = QuantumEligibilityTraces::new(&pairs, QuantumEligibilityConfig::default());

        // Create activity with proper STDP timing (pre before post)
        traces.update(&[0]); // Pre-spike
        traces.update(&[1]); // Post-spike (dt = 1 > 0)
        assert!(
            traces.active_count() > 0,
            "Should have activity after STDP potentiation"
        );

        traces.reset();
        assert_eq!(traces.active_count(), 0, "Should be inactive after reset");
        assert_eq!(traces.tick(), 0, "Tick should be 0 after reset");
    }

    #[test]
    fn test_total_phase_magnitude() {
        let pairs = vec![(0, 1), (1, 2), (2, 3)];
        let mut traces = QuantumEligibilityTraces::new(&pairs, QuantumEligibilityConfig::default());

        // Initially zero
        assert!((traces.total_phase_magnitude() - 0.0).abs() < 1e-10);

        // Create activity
        traces.update(&[0, 1, 2]);
        traces.update(&[1, 2, 3]);

        // Should have some phase
        assert!(traces.total_phase_magnitude() > 0.0);
    }
}
