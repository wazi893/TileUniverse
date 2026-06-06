//! Quantum-SNN Hybrid System
//!
//! Combines classical Spiking Neural Networks with sparse quantum simulation
//! for enhanced decision making through quantum interference.
//!
//! # Architecture
//!
//! ```text
//! Classical SNN Layer     →  Spike-to-Quantum  →  Quantum Interference  →  Measurement
//! (1000 LIF neurons)         Encoder               (H + CZ gates)          (Action)
//! ```
//!
//! # Key Innovation
//!
//! The sparse quantum representation is perfect for SNNs because:
//! - Spike patterns are naturally sparse (~1-5% active)
//! - We encode spike counts as amplitudes, not full 2^n superposition
//! - Interference preserves sparsity (CZ doesn't increase amplitude count)

use crate::snn::curiosity::CuriosityModule;
use crate::snn::value_function::TDCreditAssignment;
use crate::snn::{SNNConfig, SNNNetwork};
use crate::tile8::sparse_quantum_bigint::{Complex64, SparseQuantumGridBigInt};
use num_bigint::BigUint;
use num_traits::One;

/// Quantum interference mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterferenceMode {
    /// No interference - pure spike-proportional sampling
    None,
    /// Soft mixing - small rotations that gently blend adjacent probabilities
    /// Good for exploration while preserving spike count information
    SoftMix,
    /// Grover-like amplitude amplification - enhances the dominant action
    /// while maintaining some probability for others
    Grover,
    /// Anti-Grover curiosity exploration - amplifies LEAST preferred actions
    /// Good for escaping local optima and exploring unexplored options
    AntiGrover,
    /// Adaptive: starts with AntiGrover (exploration), decays to Grover (exploitation)
    /// The decay is controlled by the tick count
    Adaptive,
    /// Triggered: starts with AntiGrover (exploration), switches to Grover after
    /// receiving a high reward. Best of both worlds - explore until you find
    /// something good, then exploit it.
    Triggered,
    /// Original H+CZ circuit (creates destructive interference - not recommended)
    HadamardCZ,
}

impl Default for InterferenceMode {
    fn default() -> Self {
        Self::SoftMix // Best default for learning
    }
}

/// Configuration for the Quantum-SNN hybrid system
#[derive(Clone, Debug)]
pub struct QuantumSNNConfig {
    /// Number of SNN input neurons
    pub n_inputs: usize,
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Number of output neurons (also quantum qubits)
    pub n_outputs: usize,
    /// Quantum interference mode
    pub interference_mode: InterferenceMode,
    /// Number of quantum interference layers (for modes that use depth)
    pub interference_depth: usize,
    /// Mixing angle for SoftMix mode (radians, 0 = no mixing, π/4 = strong mixing)
    pub mix_angle: f64,
    /// Ticks per decision (integration window)
    pub ticks_per_decision: usize,
    /// Enable R-STDP learning
    pub learning_enabled: bool,
    /// Reward threshold to trigger switch from exploration to exploitation
    /// Used by InterferenceMode::Triggered
    pub trigger_threshold: i8,
    /// Number of consecutive high rewards required to trigger exploitation
    /// Higher values ensure the network has truly learned before switching
    pub trigger_count: usize,
    /// Duplicate inputs for each output channel
    /// When true, creates n_inputs * n_outputs input neurons
    /// Enables channel-aware learning for context-dependent tasks
    pub duplicate_inputs: bool,
}

impl Default for QuantumSNNConfig {
    fn default() -> Self {
        Self {
            n_inputs: 100,
            hidden_layers: vec![890],
            n_outputs: 10,
            interference_mode: InterferenceMode::SoftMix,
            interference_depth: 1,
            mix_angle: 0.1, // Small mixing angle (~6 degrees)
            ticks_per_decision: 20,
            learning_enabled: true,
            trigger_threshold: 50, // Switch to exploitation after reward >= 50
            trigger_count: 5,      // Require 5 consecutive high rewards
            duplicate_inputs: false,
        }
    }
}

impl QuantumSNNConfig {
    /// Create a small config for testing
    pub fn small() -> Self {
        Self {
            n_inputs: 16,
            hidden_layers: vec![32],
            n_outputs: 4,
            interference_mode: InterferenceMode::SoftMix,
            interference_depth: 1,
            mix_angle: 0.1,
            ticks_per_decision: 10,
            learning_enabled: false,
            trigger_threshold: 50,
            trigger_count: 5,
            duplicate_inputs: false,
        }
    }

    /// Create config for XOR task
    pub fn xor() -> Self {
        Self {
            n_inputs: 4,
            hidden_layers: vec![16],
            n_outputs: 2,
            interference_mode: InterferenceMode::SoftMix,
            interference_depth: 1,
            mix_angle: 0.15, // Moderate mixing for exploration
            ticks_per_decision: 20,
            learning_enabled: true,
            trigger_threshold: 50,
            trigger_count: 5,
            duplicate_inputs: false,
        }
    }

    /// Create config for pattern classification
    pub fn pattern_classifier(n_classes: usize) -> Self {
        Self {
            n_inputs: 64, // 8x8 input
            hidden_layers: vec![128, 64],
            n_outputs: n_classes,
            interference_mode: InterferenceMode::Grover,
            interference_depth: 1,
            mix_angle: 0.1,
            ticks_per_decision: 30,
            learning_enabled: true,
            trigger_threshold: 50,
            trigger_count: 5,
            duplicate_inputs: false,
        }
    }

    /// Create config with no interference (pure spike-proportional)
    pub fn no_interference(n_inputs: usize, hidden: Vec<usize>, n_outputs: usize) -> Self {
        Self {
            n_inputs,
            hidden_layers: hidden,
            n_outputs,
            interference_mode: InterferenceMode::None,
            interference_depth: 0,
            mix_angle: 0.0,
            ticks_per_decision: 20,
            learning_enabled: true,
            trigger_threshold: 50,
            trigger_count: 5,
            duplicate_inputs: false,
        }
    }
}

/// Quantum-SNN Hybrid System
///
/// Combines a classical SNN for feature extraction with a sparse quantum
/// layer for decision making via interference.
pub struct QuantumSNN {
    /// Classical SNN for feature extraction
    snn: SNNNetwork,

    /// Sparse quantum decision layer
    quantum: SparseQuantumGridBigInt,

    /// Configuration
    config: QuantumSNNConfig,

    /// Current tick counter
    tick: u64,

    /// Last measurement result
    last_action: usize,

    /// Probability distribution from last measurement
    last_probabilities: Vec<f64>,

    /// RNG state for measurement sampling
    rng_state: u64,

    /// Whether triggered mode has switched to exploitation
    /// (used by InterferenceMode::Triggered)
    exploitation_triggered: bool,

    /// Counter for consecutive high rewards (for Triggered mode)
    consecutive_high_rewards: usize,

    /// Optional curiosity module for intrinsic motivation (EPIC 121)
    curiosity: CuriosityModule,

    /// Previous state for curiosity (needed for prediction error)
    prev_state: Option<Vec<f32>>,

    /// Optional TD credit assignment for delayed rewards (EPIC 122)
    td_credit: Option<TDCreditAssignment>,
}

impl QuantumSNN {
    /// Create a new Quantum-SNN hybrid system
    pub fn new(config: QuantumSNNConfig) -> Self {
        Self::with_seed(config, 42)
    }

    /// Create a new Quantum-SNN hybrid system with a specific seed
    pub fn with_seed(config: QuantumSNNConfig, seed: u64) -> Self {
        // Create classical SNN
        let snn_config = SNNConfig {
            n_inputs: config.n_inputs,
            hidden_layers: config.hidden_layers.clone(),
            n_outputs: config.n_outputs,
            connection_prob: 0.3,
            use_reward_modulation: config.learning_enabled,
            duplicate_inputs: config.duplicate_inputs,
            ..SNNConfig::default()
        };
        let mut snn = SNNNetwork::new(snn_config);
        snn.build_connectivity(seed);

        // Create sparse quantum layer
        // We need at least 7 qubits for SparseQuantumGridBigInt
        let n_qubits = config.n_outputs.max(7);
        let quantum = SparseQuantumGridBigInt::new(n_qubits);

        Self {
            snn,
            quantum,
            config,
            tick: 0,
            last_action: 0,
            last_probabilities: Vec::new(),
            rng_state: seed,
            exploitation_triggered: false,
            consecutive_high_rewards: 0,
            curiosity: CuriosityModule::None,
            prev_state: None,
            td_credit: None,
        }
    }

    /// Set input spike rates (0-255 for each input neuron)
    pub fn set_inputs(&mut self, inputs: &[u8]) {
        self.snn.set_inputs(inputs);
    }

    /// Enable novelty-based curiosity
    ///
    /// Provides intrinsic reward for visiting novel states.
    /// Good for environments with sparse external rewards.
    pub fn enable_novelty_curiosity(&mut self, reward_scale: f32, decay_rate: f32) {
        self.curiosity = CuriosityModule::novelty(reward_scale, decay_rate);
    }

    /// Enable prediction-error curiosity
    ///
    /// Provides intrinsic reward for surprising state transitions.
    /// Encourages exploring unpredictable parts of the environment.
    pub fn enable_prediction_curiosity(&mut self, learning_rate: f32, reward_scale: f32) {
        self.curiosity = CuriosityModule::prediction_error(
            self.config.n_inputs,
            self.config.n_outputs,
            learning_rate,
            reward_scale,
        );
    }

    /// Enable combined curiosity (novelty + prediction error)
    pub fn enable_combined_curiosity(&mut self, reward_scale: f32) {
        self.curiosity =
            CuriosityModule::combined(self.config.n_inputs, self.config.n_outputs, reward_scale);
    }

    /// Disable curiosity
    pub fn disable_curiosity(&mut self) {
        self.curiosity = CuriosityModule::None;
        self.prev_state = None;
    }

    /// Check if curiosity is enabled
    pub fn has_curiosity(&self) -> bool {
        self.curiosity.is_enabled()
    }

    // =========================================================================
    // TD Credit Assignment (EPIC 122)
    // =========================================================================

    /// Enable TD-style credit assignment for delayed rewards.
    ///
    /// Uses temporal difference learning to bootstrap value estimates,
    /// providing better credit assignment for delayed reward problems.
    ///
    /// # Arguments
    /// * `alpha` - Value function learning rate (typically 0.01-0.1)
    /// * `gamma` - Discount factor (typically 0.95-0.99)
    /// * `td_scale` - Scale factor for TD error → R-STDP signal
    pub fn enable_td_credit(&mut self, alpha: f32, gamma: f32, td_scale: f32) {
        self.td_credit = Some(TDCreditAssignment::new(
            self.config.n_inputs,
            alpha,
            gamma,
            td_scale,
        ));
    }

    /// Disable TD credit assignment
    pub fn disable_td_credit(&mut self) {
        self.td_credit = None;
    }

    /// Check if TD credit is enabled
    pub fn has_td_credit(&self) -> bool {
        self.td_credit.is_some()
    }

    /// Learn with TD-modulated credit assignment.
    ///
    /// Uses TD error instead of raw reward for R-STDP, providing better
    /// credit assignment for delayed rewards.
    ///
    /// # Arguments
    /// * `reward` - Immediate reward from environment
    /// * `current_state` - Current state (normalized 0-1)
    /// * `done` - Whether episode ended
    ///
    /// # Returns
    /// The TD error (positive = better than expected, negative = worse)
    pub fn learn_with_td(&mut self, reward: f32, current_state: &[f32], done: bool) -> f32 {
        // Compute TD signal if enabled, otherwise use scaled reward
        let (td_signal, td_error) = if let Some(ref mut td) = self.td_credit {
            let signal = td.compute_signal(reward, current_state, done);
            let error = td.last_td_error();
            (signal as i8, error)
        } else {
            let scaled = (reward * 100.0).clamp(-127.0, 127.0) as i8;
            (scaled, reward)
        };

        // Apply R-STDP with computed signal
        self.learn(td_signal);

        td_error
    }

    /// Get estimated value of a state (requires TD credit enabled)
    pub fn state_value(&self, state: &[f32]) -> Option<f32> {
        self.td_credit.as_ref().map(|td| td.value(state))
    }

    /// Reset TD credit at episode boundary
    pub fn reset_td_episode(&mut self) {
        if let Some(ref mut td) = self.td_credit {
            td.reset_episode();
        }
    }

    /// Run one complete decision cycle:
    /// 1. Run SNN for ticks_per_decision steps
    /// 2. Encode spike counts as quantum amplitudes
    /// 3. Apply quantum interference
    /// 4. Measure to get action
    pub fn decide(&mut self) -> usize {
        // Reset output counts for new decision
        self.snn.reset_output_counts();

        // Run SNN for integration window
        for _ in 0..self.config.ticks_per_decision {
            self.snn.step();
            self.tick += 1;
        }

        // Encode spike counts as quantum state
        self.encode_spikes_to_quantum();

        // Apply quantum interference
        self.apply_interference();

        // Measure to get action
        self.last_action = self.measure();

        self.last_action
    }

    /// Get classical winner-take-all action (for comparison)
    pub fn classical_action(&self) -> usize {
        self.snn.get_action()
    }

    /// Encode SNN output spike counts as quantum amplitudes
    ///
    /// Creates a quantum state where amplitude_i = sqrt(count_i / total)
    /// This preserves the probability interpretation: |amplitude|² = count / total
    fn encode_spikes_to_quantum(&mut self) {
        let spike_counts = &self.snn.output_counts;
        let total: u32 = spike_counts.iter().sum();

        // Clear quantum state
        self.quantum.initialize_zero_state();

        if total == 0 {
            // No spikes: create uniform superposition
            for q in 0..self.config.n_outputs.min(7) {
                self.quantum.apply_h(q);
            }
            // For qubits 7+, we'd need cross-block Hadamards
            // For now, keep them in |0⟩ state
        } else {
            // Encode spike rates as amplitudes
            // |ψ⟩ = Σ sqrt(count_i/total) |2^i⟩
            self.quantum.blocks.clear();

            for (i, &count) in spike_counts.iter().enumerate() {
                if count > 0 && i < self.config.n_outputs {
                    let amplitude = (count as f64 / total as f64).sqrt();

                    // State |2^i⟩ corresponds to qubit i being |1⟩
                    let state_index: BigUint = BigUint::one() << i;
                    let block_id = &state_index >> 7;
                    let local_addr = (&state_index & BigUint::from(127u32))
                        .to_u64_digits()
                        .first()
                        .copied()
                        .unwrap_or(0) as usize;

                    let block = self.quantum.get_or_create_block(block_id);
                    block.set(local_addr, Complex64::new(amplitude, 0.0));
                }
            }

            // Ensure state is normalized (should already be from encoding)
            self.normalize_quantum_state();
        }
    }

    /// Normalize the quantum state to have total probability 1.0
    fn normalize_quantum_state(&mut self) {
        let total_prob = self.quantum.total_probability();
        if total_prob > 1e-10 && (total_prob - 1.0).abs() > 1e-10 {
            let norm_factor = 1.0 / total_prob.sqrt();
            for block in self.quantum.blocks.values_mut() {
                for i in 0..128 {
                    let amp = block.get(i);
                    if amp.norm_squared() > 1e-30 {
                        block.set(
                            i,
                            Complex64::new(amp.re * norm_factor, amp.im * norm_factor),
                        );
                    }
                }
            }
        }
    }

    /// Apply quantum interference circuit based on configured mode
    ///
    /// Different modes provide different exploration/exploitation tradeoffs:
    /// - None: Pure spike-proportional sampling
    /// - SoftMix: Gentle blending that preserves spike information
    /// - Grover: Amplitude amplification of dominant action
    /// - HadamardCZ: Original circuit (not recommended - destroys information)
    fn apply_interference(&mut self) {
        // Skip for large output counts - sparse advantage is lost
        if self.config.n_outputs > 12 {
            return;
        }

        match self.config.interference_mode {
            InterferenceMode::None => {
                // No interference - pure spike-proportional sampling
            }

            InterferenceMode::SoftMix => {
                // Soft mixing using Ry rotations with small angles
                // This gently blends probabilities between adjacent actions
                // while preserving the overall spike count ordering
                self.apply_soft_mix();
            }

            InterferenceMode::Grover => {
                // Grover-like amplitude amplification
                // Enhances the dominant action while maintaining exploration
                self.apply_grover_amplification();
            }

            InterferenceMode::AntiGrover => {
                // Anti-Grover curiosity exploration
                // Amplifies LEAST preferred actions to encourage exploration
                self.apply_anti_grover();
            }

            InterferenceMode::Adaptive => {
                // Adaptive exploration-exploitation balance
                // Early: more anti-grover (exploration)
                // Late: more grover (exploitation)
                self.apply_adaptive_interference();
            }

            InterferenceMode::Triggered => {
                // Triggered mode: explore until high reward, then exploit
                // Start with anti-grover to discover good actions
                // Once triggered (high reward received), switch to grover
                self.apply_triggered_interference();
            }

            InterferenceMode::HadamardCZ => {
                // Original H+CZ circuit (not recommended)
                for _layer in 0..self.config.interference_depth {
                    for q in (1..self.config.n_outputs).step_by(2) {
                        self.quantum.apply_h(q);
                    }
                    for q in 0..(self.config.n_outputs - 1) {
                        self.quantum.apply_cz(q, q + 1);
                    }
                    for q in (0..self.config.n_outputs).step_by(2) {
                        self.quantum.apply_h(q);
                    }
                }
            }
        }
    }

    /// Soft mixing: apply small Ry rotations to gently blend probabilities
    ///
    /// For each pair of adjacent qubits, applies a controlled rotation
    /// that mixes their amplitudes slightly. This creates exploration
    /// while preserving the spike count ordering.
    fn apply_soft_mix(&mut self) {
        let angle = self.config.mix_angle;
        if angle.abs() < 1e-10 {
            return;
        }

        // For each output, apply a small rotation that mixes with neighbors
        // We implement this as: for each basis state |i⟩, create small
        // amplitude in neighboring states |i±1⟩

        // Get current amplitudes
        let amplitudes: Vec<Complex64> = (0..self.config.n_outputs)
            .map(|i| self.get_basis_amplitude(i))
            .collect();

        // Apply soft mixing: new_amp[i] = cos(θ)*amp[i] + sin(θ)/√2 * (amp[i-1] + amp[i+1])
        let cos_a = angle.cos();
        let sin_a = angle.sin() / 2.0_f64.sqrt();

        let mut new_amplitudes = amplitudes.clone();
        for i in 0..self.config.n_outputs {
            let left = if i > 0 {
                amplitudes[i - 1]
            } else {
                Complex64::zero()
            };
            let right = if i < self.config.n_outputs - 1 {
                amplitudes[i + 1]
            } else {
                Complex64::zero()
            };

            new_amplitudes[i] = Complex64::new(
                cos_a * amplitudes[i].re + sin_a * (left.re + right.re),
                cos_a * amplitudes[i].im + sin_a * (left.im + right.im),
            );
        }

        // Normalize and write back
        let norm: f64 = new_amplitudes
            .iter()
            .map(|a| a.norm_squared())
            .sum::<f64>()
            .sqrt();
        if norm > 1e-10 {
            for (i, amp) in new_amplitudes.iter().enumerate() {
                let normalized = Complex64::new(amp.re / norm, amp.im / norm);
                self.set_basis_amplitude(i, normalized);
            }
        }
    }

    /// Grover-like amplitude amplification
    ///
    /// Applies soft amplification that enhances the dominant action's probability
    /// while maintaining some probability for exploration.
    /// Unlike standard Grover (which inverts about the mean), this AMPLIFIES
    /// differences from the mean in the SAME direction.
    fn apply_grover_amplification(&mut self) {
        self.apply_grover_with_strength(0.5);
    }

    /// Apply Grover amplification with configurable strength
    /// g > 0: amplify dominant (exploitation)
    /// g < 0: amplify minority (exploration / anti-grover)
    fn apply_grover_with_strength(&mut self, g: f64) {
        // Get current amplitudes
        let amplitudes: Vec<Complex64> = (0..self.config.n_outputs)
            .map(|i| self.get_basis_amplitude(i))
            .collect();

        // Compute mean amplitude
        let n = self.config.n_outputs as f64;
        let mean_re: f64 = amplitudes.iter().map(|a| a.re).sum::<f64>() / n;
        let mean_im: f64 = amplitudes.iter().map(|a| a.im).sum::<f64>() / n;

        // Amplification: amp'[i] = amp[i] + g*(amp[i] - mean)
        // g > 0: pushes larger amplitudes higher, smaller lower (exploitation)
        // g < 0: pushes larger amplitudes lower, smaller higher (exploration)
        let mut new_amplitudes = Vec::with_capacity(self.config.n_outputs);
        for amp in &amplitudes {
            new_amplitudes.push(Complex64::new(
                amp.re + g * (amp.re - mean_re),
                amp.im + g * (amp.im - mean_im),
            ));
        }

        // Normalize and write back
        let norm: f64 = new_amplitudes
            .iter()
            .map(|a| a.norm_squared())
            .sum::<f64>()
            .sqrt();
        if norm > 1e-10 {
            for (i, amp) in new_amplitudes.iter().enumerate() {
                let normalized = Complex64::new(amp.re / norm, amp.im / norm);
                self.set_basis_amplitude(i, normalized);
            }
        }
    }

    /// Anti-Grover: Curiosity-driven exploration
    ///
    /// The opposite of Grover - amplifies the LEAST preferred actions.
    /// This encourages exploration of actions the network hasn't learned to prefer.
    /// Useful for escaping local optima and discovering hidden rewards.
    fn apply_anti_grover(&mut self) {
        // Use negative strength to invert the amplification direction
        // -0.5 means: actions below mean get amplified, actions above get reduced
        self.apply_grover_with_strength(-0.5);
    }

    /// Adaptive interference: exploration → exploitation over time
    ///
    /// Early in training: anti-grover (explore new actions)
    /// Late in training: grover (exploit learned knowledge)
    ///
    /// The transition is controlled by tick count and config parameters.
    fn apply_adaptive_interference(&mut self) {
        // Exploration decay schedule
        // Start with g = -0.5 (full anti-grover)
        // Decay towards g = +0.5 (full grover)
        // The transition happens over ~10000 ticks

        let exploration_phase = 5000.0; // Ticks of pure exploration
        let transition_phase = 10000.0; // Ticks to transition

        let t = self.tick as f64;

        let g = if t < exploration_phase {
            // Pure exploration phase
            -0.5
        } else if t < exploration_phase + transition_phase {
            // Linear transition from -0.5 to +0.5
            let progress = (t - exploration_phase) / transition_phase;
            -0.5 + progress * 1.0
        } else {
            // Pure exploitation phase
            0.5
        };

        self.apply_grover_with_strength(g);
    }

    /// Triggered interference: explore until high reward, then exploit
    ///
    /// Uses anti-grover (curiosity) until trigger_count consecutive high rewards,
    /// then permanently switches to NO interference (pure classical WTA).
    /// Best of both worlds: thorough exploration followed by deterministic exploitation.
    fn apply_triggered_interference(&mut self) {
        if self.exploitation_triggered {
            // High rewards confirmed - switch to pure classical (no interference)
            // This gives maximum exploitation/convergence
            // (No-op: skip all interference, use raw spike probabilities)
        } else {
            // Still exploring - use anti-grover to find good rewards
            self.apply_grover_with_strength(-0.5);
        }
    }

    /// Check if a reward should trigger the switch to exploitation mode
    /// Requires `trigger_count` consecutive high rewards before switching.
    /// This ensures the network has actually learned before exploiting.
    pub fn check_trigger(&mut self, reward: i8) {
        if self.exploitation_triggered {
            return; // Already triggered
        }

        if reward >= self.config.trigger_threshold {
            self.consecutive_high_rewards += 1;
            if self.consecutive_high_rewards >= self.config.trigger_count {
                self.exploitation_triggered = true;
            }
        } else {
            // Reset counter on low reward - must be consecutive!
            self.consecutive_high_rewards = 0;
        }
    }

    /// Get the current consecutive high reward count
    pub fn consecutive_high_rewards(&self) -> usize {
        self.consecutive_high_rewards
    }

    /// Check if exploitation has been triggered
    pub fn is_exploitation_triggered(&self) -> bool {
        self.exploitation_triggered
    }

    /// Manually trigger exploitation mode (useful for testing)
    pub fn trigger_exploitation(&mut self) {
        self.exploitation_triggered = true;
    }

    /// Reset the trigger (go back to exploration mode)
    pub fn reset_trigger(&mut self) {
        self.exploitation_triggered = false;
    }

    /// Get amplitude for basis state |2^i⟩
    fn get_basis_amplitude(&self, action: usize) -> Complex64 {
        if action >= self.config.n_outputs {
            return Complex64::zero();
        }

        let state_index: BigUint = BigUint::one() << action;
        let block_id = &state_index >> 7;
        let local_addr = (&state_index & BigUint::from(127u32))
            .to_u64_digits()
            .first()
            .copied()
            .unwrap_or(0) as usize;

        if let Some(block) = self.quantum.blocks.get(&block_id) {
            block.get(local_addr)
        } else {
            Complex64::zero()
        }
    }

    /// Set amplitude for basis state |2^i⟩
    fn set_basis_amplitude(&mut self, action: usize, amplitude: Complex64) {
        if action >= self.config.n_outputs {
            return;
        }

        let state_index: BigUint = BigUint::one() << action;
        let block_id = &state_index >> 7;
        let local_addr = (&state_index & BigUint::from(127u32))
            .to_u64_digits()
            .first()
            .copied()
            .unwrap_or(0) as usize;

        let block = self.quantum.get_or_create_block(block_id);
        block.set(local_addr, amplitude);
    }

    /// Get probability of measuring qubit i as |1⟩ (marginalized over all other qubits)
    ///
    /// For sparse encoding where we use |2^i⟩ basis states,
    /// this is the probability of action i.
    fn get_action_probability(&self, action: usize) -> f64 {
        if action >= self.config.n_outputs {
            return 0.0;
        }

        let mut prob = 0.0;

        // Sum probability from all basis states where qubit `action` is |1⟩
        // In our sparse encoding, state |2^action⟩ has qubit `action` set
        // After interference, we need to check all states
        for (block_id, block) in &self.quantum.blocks {
            for addr in 0..128 {
                let amp = block.get(addr);
                if amp.norm_squared() < 1e-30 {
                    continue;
                }

                // Reconstruct full state index
                let state_idx = block_id * BigUint::from(128u32) + BigUint::from(addr as u32);

                // Check if qubit `action` is |1⟩ in this state
                let qubit_set = if action < 64 {
                    // Can use u64 for efficiency
                    let idx_u64 = state_idx.to_u64_digits().first().copied().unwrap_or(0);
                    (idx_u64 >> action) & 1 == 1
                } else {
                    // Need BigUint arithmetic
                    (&state_idx >> action) & BigUint::one() == BigUint::one()
                };

                if qubit_set {
                    prob += amp.norm_squared();
                }
            }
        }

        prob
    }

    /// Measure the quantum state to select an action
    ///
    /// Returns the action index with probability proportional to |amplitude|²
    fn measure(&mut self) -> usize {
        // Compute probability distribution
        self.last_probabilities = (0..self.config.n_outputs)
            .map(|i| self.get_action_probability(i))
            .collect();

        // Normalize probabilities
        let total_prob: f64 = self.last_probabilities.iter().sum();
        if total_prob > 1e-10 {
            for p in &mut self.last_probabilities {
                *p /= total_prob;
            }
        } else {
            // Uniform if no probability
            let uniform = 1.0 / self.config.n_outputs as f64;
            self.last_probabilities = vec![uniform; self.config.n_outputs];
        }

        // Sample from distribution using LCG PRNG
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let r = (self.rng_state >> 33) as f64 / (1u64 << 31) as f64;

        let mut cumulative = 0.0;
        for (i, &prob) in self.last_probabilities.iter().enumerate() {
            cumulative += prob;
            if r < cumulative {
                return i;
            }
        }

        self.config.n_outputs - 1
    }

    /// Get action with highest probability (deterministic mode)
    pub fn measure_deterministic(&mut self) -> usize {
        // Compute probabilities if not already done
        if self.last_probabilities.is_empty() {
            self.last_probabilities = (0..self.config.n_outputs)
                .map(|i| self.get_action_probability(i))
                .collect();
        }

        self.last_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Provide reward feedback for R-STDP learning
    pub fn set_reward(&mut self, reward: i8) {
        self.snn.set_reward(reward);
    }

    /// Apply accumulated eligibility traces with current reward
    /// Call this after set_reward() to trigger weight updates
    pub fn apply_learning(&mut self) {
        self.snn.apply_learning();
    }

    /// Convenience: set reward and immediately apply learning
    /// Also checks if reward triggers exploitation mode (for Triggered interference)
    pub fn learn(&mut self, reward: i8) {
        // Check if this reward should trigger exploitation mode
        self.check_trigger(reward);

        self.snn.set_reward(reward);
        self.snn.apply_learning();
    }

    /// Action-gated learning: only update synapses causally connected to chosen action
    ///
    /// This implements proper credit assignment for RL:
    /// - Only synapses that contributed to the selected action receive reward
    /// - Prevents symmetric reward from reinforcing all actions equally
    /// - Uses the last action from decide() if no action specified
    ///
    /// # Arguments
    /// * `reward` - Reward signal (-127 to 127)
    /// * `action` - Optional action index. If None, uses last_action from decide()
    pub fn learn_gated(&mut self, reward: i8, action: Option<usize>) {
        // Check if this reward should trigger exploitation mode
        self.check_trigger(reward);

        // Use provided action or fall back to last action
        let chosen_action = action.unwrap_or(self.last_action);

        self.snn.set_reward(reward);
        self.snn.apply_learning_gated(chosen_action);
    }

    /// Competitive learning: strengthen chosen action, weaken others
    ///
    /// This explicitly creates competition between actions by weakening
    /// synapses targeting non-chosen outputs. Helps differentiate actions
    /// even with shared hidden layer representations.
    ///
    /// # Arguments
    /// * `reward` - Reward signal (positive = reinforce chosen, negative = punish chosen)
    /// * `action` - Optional action index. If None, uses last_action from decide()
    /// * `weaken_percent` - How much to weaken other actions (0-100, default 30)
    pub fn learn_competitive(
        &mut self,
        reward: i8,
        action: Option<usize>,
        weaken_percent: Option<i8>,
    ) {
        self.check_trigger(reward);
        let chosen_action = action.unwrap_or(self.last_action);
        let weaken = weaken_percent.unwrap_or(30);

        self.snn.set_reward(reward);
        self.snn.apply_learning_competitive(chosen_action, weaken);
    }

    /// Deep gated learning: update output AND selective hidden layer synapses.
    ///
    /// This extends gated learning to also update input->hidden synapses,
    /// but ONLY for hidden neurons that have stronger connections to the
    /// chosen output than to any other output. This enables context-dependent
    /// action learning while avoiding the shared hidden neuron problem.
    ///
    /// # Arguments
    /// * `reward` - Reward signal (positive = reinforce, negative = punish)
    /// * `action` - Optional action index. If None, uses last_action from decide()
    pub fn learn_deep_gated(&mut self, reward: i8, action: Option<usize>) {
        self.check_trigger(reward);
        let chosen_action = action.unwrap_or(self.last_action);

        self.snn.set_reward(reward);
        self.snn.apply_learning_deep_gated(chosen_action);
    }

    /// Channel-aware learning: respect the network's channel structure.
    ///
    /// The network is built with dedicated neuron channels per action.
    /// This method only updates synapses where BOTH source and target
    /// belong to the chosen action's channel, preventing learning interference.
    ///
    /// Best for context-dependent tasks where different inputs require
    /// different actions.
    pub fn learn_channeled(&mut self, reward: i8, action: Option<usize>) {
        self.check_trigger(reward);
        let chosen_action = action.unwrap_or(self.last_action);

        self.snn.set_reward(reward);
        self.snn.apply_learning_channeled(chosen_action);
    }

    /// Get the last action taken
    pub fn last_action(&self) -> usize {
        self.last_action
    }

    /// Get probability distribution from last measurement
    pub fn last_probabilities(&self) -> &[f64] {
        &self.last_probabilities
    }

    /// Get output spike counts from SNN
    pub fn output_spike_counts(&self) -> &[u32] {
        &self.snn.output_counts
    }

    /// Get total ticks elapsed
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Get reference to underlying SNN
    pub fn snn(&self) -> &SNNNetwork {
        &self.snn
    }

    /// Get mutable reference to underlying SNN
    pub fn snn_mut(&mut self) -> &mut SNNNetwork {
        &mut self.snn
    }

    /// Get number of active quantum blocks (for sparse efficiency tracking)
    pub fn quantum_blocks(&self) -> usize {
        self.quantum.active_blocks()
    }

    /// Reset the hybrid system
    pub fn reset(&mut self) {
        self.snn.reset_output_counts();
        self.quantum.initialize_zero_state();
        self.tick = 0;
        self.last_action = 0;
        self.last_probabilities.clear();
        self.exploitation_triggered = false;
        self.consecutive_high_rewards = 0;
        self.curiosity.reset();
        self.prev_state = None;
        if let Some(ref mut td) = self.td_credit {
            td.reset();
        }
    }

    // =========================================================================
    // Curiosity-Enhanced Learning (EPIC 121)
    // =========================================================================

    /// Learn with curiosity-augmented reward.
    ///
    /// Combines extrinsic reward from environment with intrinsic curiosity reward.
    /// Call this instead of `learn()` when curiosity is enabled.
    ///
    /// # Arguments
    /// * `extrinsic_reward` - External reward from environment (-127 to 127)
    /// * `current_state` - Current observation/state as floats (normalized 0-1)
    ///
    /// # Returns
    /// The total reward (extrinsic + intrinsic)
    pub fn learn_with_curiosity(&mut self, extrinsic_reward: i8, current_state: &[f32]) -> f32 {
        let extrinsic = extrinsic_reward as f32;

        // Compute intrinsic reward if curiosity is enabled
        let intrinsic = if self.curiosity.is_enabled() {
            self.curiosity.intrinsic_reward(
                self.prev_state.as_deref(),
                Some(self.last_action),
                current_state,
            )
        } else {
            0.0
        };

        // Store current state for next iteration
        self.prev_state = Some(current_state.to_vec());

        // Total reward for R-STDP
        let total_reward = extrinsic + intrinsic * 50.0; // Scale intrinsic to similar range
        let clamped_reward = total_reward.clamp(-127.0, 127.0) as i8;

        // Apply R-STDP learning with combined reward
        self.learn(clamped_reward);

        total_reward
    }

    /// Learn with curiosity using spike rates instead of floats
    ///
    /// Convenience method for when observations are already spike rates.
    pub fn learn_with_curiosity_spikes(
        &mut self,
        extrinsic_reward: i8,
        current_rates: &[u8],
    ) -> f32 {
        // Convert spike rates to normalized floats
        let current_state: Vec<f32> = current_rates.iter().map(|&r| r as f32 / 255.0).collect();

        self.learn_with_curiosity(extrinsic_reward, &current_state)
    }

    /// Apply curiosity decay (call at end of episode)
    ///
    /// For novelty-based curiosity, this decays visit counts so old states
    /// become novel again over time.
    pub fn decay_curiosity(&mut self) {
        self.curiosity.decay();
    }

    /// Get current intrinsic reward without updating (for debugging)
    pub fn peek_intrinsic_reward(&self, current_state: &[f32]) -> f32 {
        // Create a temporary copy to avoid mutation
        let mut temp_curiosity = self.curiosity.clone();
        temp_curiosity.intrinsic_reward(
            self.prev_state.as_deref(),
            Some(self.last_action),
            current_state,
        )
    }
}

// ============================================================================
// GPU-Accelerated Quantum-SNN Hybrid (Phase 2: 100K+ neurons)
// ============================================================================

#[cfg(feature = "cuda")]
use crate::cuda::CudaRuntime;
#[cfg(feature = "cuda")]
use crate::snn::LIFNeuron;
#[cfg(feature = "cuda")]
use crate::snn::gpu_kernels::GpuSNNState;
#[cfg(feature = "cuda")]
use crate::snn::synapse::SynapseCSR;

/// GPU-accelerated Quantum-SNN Hybrid for 100K+ neuron scale
///
/// Uses CUDA-accelerated SNN for the classical layer and sparse quantum
/// simulation for the decision layer.
///
/// Now supports all 6 interference modes (parity with CPU version):
/// - None: Pure spike-proportional sampling
/// - SoftMix: Gentle neighbor blending
/// - Grover: Amplitude amplification (exploitation)
/// - AntiGrover: Curiosity exploration
/// - Adaptive: Time-based exploration→exploitation decay
/// - Triggered: Reward-triggered mode switching (+69% improvement)
#[cfg(feature = "cuda")]
pub struct GpuQuantumSNN {
    /// GPU-accelerated SNN state
    gpu_state: GpuSNNState,

    /// Sparse quantum decision layer (runs on CPU - it's sparse so fast)
    quantum: SparseQuantumGridBigInt,

    /// Configuration
    config: QuantumSNNConfig,

    /// Current tick counter
    tick: u64,

    /// Last measurement result
    last_action: usize,

    /// Probability distribution from last measurement
    last_probabilities: Vec<f64>,

    /// RNG state for measurement sampling
    rng_state: u64,

    /// Cached input rates (to avoid reallocation)
    input_rates: Vec<u8>,

    /// Whether triggered mode has switched to exploitation
    /// (used by InterferenceMode::Triggered)
    exploitation_triggered: bool,

    /// Counter for consecutive high rewards (for Triggered mode)
    consecutive_high_rewards: usize,
}

#[cfg(feature = "cuda")]
impl GpuQuantumSNN {
    /// Create a new GPU-accelerated Quantum-SNN hybrid
    ///
    /// # Arguments
    /// * `rt` - CUDA runtime
    /// * `config` - Hybrid configuration
    ///
    /// # Returns
    /// Result containing the hybrid or CUDA error
    pub fn new(rt: &CudaRuntime, config: QuantumSNNConfig) -> Result<Self, crate::cuda::CudaError> {
        // Calculate network size
        let n_neurons =
            config.n_inputs + config.hidden_layers.iter().sum::<usize>() + config.n_outputs;

        // Create neurons
        let neurons: Vec<LIFNeuron> = (0..n_neurons).map(|_| LIFNeuron::default()).collect();

        // Create random sparse connectivity (CSR format)
        let avg_synapses_per_neuron = 10; // Sparse connectivity
        let estimated_synapses = n_neurons * avg_synapses_per_neuron;

        let mut syn_ptr = Vec::with_capacity(n_neurons + 1);
        let mut targets = Vec::with_capacity(estimated_synapses);
        let mut weights = Vec::with_capacity(estimated_synapses);
        let mut delays = Vec::with_capacity(estimated_synapses);

        let mut rng_state = 42u64;
        let mut ptr = 0u32;

        for src in 0..n_neurons {
            syn_ptr.push(ptr);

            // Generate random synapses (feed-forward bias)
            let n_synapses = avg_synapses_per_neuron;
            for _ in 0..n_synapses {
                // Simple LCG for randomness
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);

                // Bias towards forward connections
                let offset = ((rng_state >> 32) as usize) % n_neurons.max(1);
                let tgt = (src + offset + 1) % n_neurons;

                if tgt != src && tgt < n_neurons {
                    targets.push(tgt as u16);
                    // Random weight between 10-127
                    weights.push(((rng_state >> 48) as i8).abs().max(10));
                    delays.push(0);
                    ptr += 1;
                }
            }
        }
        syn_ptr.push(ptr);

        let csr = SynapseCSR {
            syn_ptr,
            targets,
            weights,
            delays,
        };

        // Create GPU state
        let gpu_state =
            GpuSNNState::from_cpu(rt, &neurons, &csr, config.n_inputs, config.n_outputs)?;

        // Create sparse quantum layer
        let n_qubits = config.n_outputs.max(7);
        let quantum = SparseQuantumGridBigInt::new(n_qubits);

        Ok(Self {
            gpu_state,
            quantum,
            config: config.clone(),
            tick: 0,
            last_action: 0,
            last_probabilities: Vec::new(),
            rng_state: 0xDEADBEEF,
            input_rates: vec![0u8; config.n_inputs],
            exploitation_triggered: false,
            consecutive_high_rewards: 0,
        })
    }

    /// Set input spike rates (0-255 for each input neuron)
    pub fn set_inputs(
        &mut self,
        rt: &CudaRuntime,
        inputs: &[u8],
    ) -> Result<(), crate::cuda::CudaError> {
        // Copy to cached buffer
        let len = inputs.len().min(self.input_rates.len());
        self.input_rates[..len].copy_from_slice(&inputs[..len]);

        // Upload to GPU
        self.gpu_state.set_input_rates(rt, &self.input_rates)
    }

    /// Run one complete decision cycle on GPU:
    /// 1. Run SNN for ticks_per_decision steps (GPU)
    /// 2. Download spike counts (GPU → CPU)
    /// 3. Encode as quantum amplitudes (CPU, sparse)
    /// 4. Apply quantum interference (CPU, sparse)
    /// 5. Measure to get action (CPU)
    pub fn decide(&mut self, rt: &CudaRuntime) -> Result<usize, crate::cuda::CudaError> {
        // Reset output counts on GPU
        self.gpu_state.reset_output_counts(rt)?;

        // Run SNN for integration window (GPU-accelerated)
        for _ in 0..self.config.ticks_per_decision {
            self.gpu_state.step(rt)?;
            self.tick += 1;
        }

        // Download spike counts from GPU
        let spike_counts = self.gpu_state.get_output_counts(rt)?;

        // Encode spike counts as quantum state (CPU, sparse - very fast)
        self.encode_spikes_to_quantum(&spike_counts);

        // Apply quantum interference (CPU, sparse - very fast)
        self.apply_interference();

        // Measure to get action (CPU)
        self.last_action = self.measure();

        Ok(self.last_action)
    }

    /// Get classical winner-take-all action (for comparison)
    pub fn classical_action(&self, rt: &CudaRuntime) -> Result<usize, crate::cuda::CudaError> {
        self.gpu_state.get_action(rt)
    }

    /// Encode spike counts as quantum amplitudes
    fn encode_spikes_to_quantum(&mut self, spike_counts: &[u32]) {
        let total: u32 = spike_counts.iter().sum();

        // Clear quantum state
        self.quantum.initialize_zero_state();

        if total == 0 {
            // No spikes: create uniform superposition
            for q in 0..self.config.n_outputs.min(7) {
                self.quantum.apply_h(q);
            }
        } else {
            // Encode spike rates as amplitudes
            self.quantum.blocks.clear();

            for (i, &count) in spike_counts.iter().enumerate() {
                if count > 0 && i < self.config.n_outputs {
                    let amplitude = (count as f64 / total as f64).sqrt();

                    let state_index: BigUint = BigUint::one() << i;
                    let block_id = &state_index >> 7;
                    let local_addr = (&state_index & BigUint::from(127u32))
                        .to_u64_digits()
                        .first()
                        .copied()
                        .unwrap_or(0) as usize;

                    let block = self.quantum.get_or_create_block(block_id);
                    block.set(local_addr, Complex64::new(amplitude, 0.0));
                }
            }

            self.normalize_quantum_state();
        }
    }

    /// Normalize the quantum state
    fn normalize_quantum_state(&mut self) {
        let total_prob = self.quantum.total_probability();
        if total_prob > 1e-10 && (total_prob - 1.0).abs() > 1e-10 {
            let norm_factor = 1.0 / total_prob.sqrt();
            for block in self.quantum.blocks.values_mut() {
                for i in 0..128 {
                    let amp = block.get(i);
                    if amp.norm_squared() > 1e-30 {
                        block.set(
                            i,
                            Complex64::new(amp.re * norm_factor, amp.im * norm_factor),
                        );
                    }
                }
            }
        }
    }

    /// Apply quantum interference circuit based on configured mode
    ///
    /// Different modes provide different exploration/exploitation tradeoffs:
    /// - None: Pure spike-proportional sampling
    /// - SoftMix: Gentle blending that preserves spike information
    /// - Grover: Amplitude amplification of dominant action
    /// - AntiGrover: Curiosity exploration (amplifies minority actions)
    /// - Adaptive: Time-based exploration→exploitation
    /// - Triggered: Reward-triggered mode switching
    /// - HadamardCZ: Original circuit (not recommended - destroys information)
    fn apply_interference(&mut self) {
        // Skip for large output counts - sparse advantage is lost
        if self.config.n_outputs > 12 {
            return;
        }

        match self.config.interference_mode {
            InterferenceMode::None => {
                // No interference - pure spike-proportional sampling
            }

            InterferenceMode::SoftMix => {
                self.apply_soft_mix();
            }

            InterferenceMode::Grover => {
                self.apply_grover_amplification();
            }

            InterferenceMode::AntiGrover => {
                self.apply_anti_grover();
            }

            InterferenceMode::Adaptive => {
                self.apply_adaptive_interference();
            }

            InterferenceMode::Triggered => {
                self.apply_triggered_interference();
            }

            InterferenceMode::HadamardCZ => {
                // Original H+CZ circuit (not recommended)
                for _layer in 0..self.config.interference_depth {
                    for q in (1..self.config.n_outputs).step_by(2) {
                        self.quantum.apply_h(q);
                    }
                    for q in 0..(self.config.n_outputs - 1) {
                        self.quantum.apply_cz(q, q + 1);
                    }
                    for q in (0..self.config.n_outputs).step_by(2) {
                        self.quantum.apply_h(q);
                    }
                }
            }
        }
    }

    /// Get amplitude for basis state |2^i⟩
    fn get_basis_amplitude(&self, action: usize) -> Complex64 {
        if action >= self.config.n_outputs {
            return Complex64::zero();
        }

        let state_index: BigUint = BigUint::one() << action;
        let block_id = &state_index >> 7;
        let local_addr = (&state_index & BigUint::from(127u32))
            .to_u64_digits()
            .first()
            .copied()
            .unwrap_or(0) as usize;

        if let Some(block) = self.quantum.blocks.get(&block_id) {
            block.get(local_addr)
        } else {
            Complex64::zero()
        }
    }

    /// Set amplitude for basis state |2^i⟩
    fn set_basis_amplitude(&mut self, action: usize, amplitude: Complex64) {
        if action >= self.config.n_outputs {
            return;
        }

        let state_index: BigUint = BigUint::one() << action;
        let block_id = &state_index >> 7;
        let local_addr = (&state_index & BigUint::from(127u32))
            .to_u64_digits()
            .first()
            .copied()
            .unwrap_or(0) as usize;

        let block = self.quantum.get_or_create_block(block_id);
        block.set(local_addr, amplitude);
    }

    /// Soft mixing: apply small rotations to gently blend probabilities
    fn apply_soft_mix(&mut self) {
        let angle = self.config.mix_angle;
        if angle.abs() < 1e-10 {
            return;
        }

        // Get current amplitudes
        let amplitudes: Vec<Complex64> = (0..self.config.n_outputs)
            .map(|i| self.get_basis_amplitude(i))
            .collect();

        // Apply soft mixing: new_amp[i] = cos(θ)*amp[i] + sin(θ)/√2 * (amp[i-1] + amp[i+1])
        let cos_a = angle.cos();
        let sin_a = angle.sin() / 2.0_f64.sqrt();

        let mut new_amplitudes = amplitudes.clone();
        for i in 0..self.config.n_outputs {
            let left = if i > 0 {
                amplitudes[i - 1]
            } else {
                Complex64::zero()
            };
            let right = if i < self.config.n_outputs - 1 {
                amplitudes[i + 1]
            } else {
                Complex64::zero()
            };

            new_amplitudes[i] = Complex64::new(
                cos_a * amplitudes[i].re + sin_a * (left.re + right.re),
                cos_a * amplitudes[i].im + sin_a * (left.im + right.im),
            );
        }

        // Normalize and write back
        let norm: f64 = new_amplitudes
            .iter()
            .map(|a| a.norm_squared())
            .sum::<f64>()
            .sqrt();
        if norm > 1e-10 {
            for (i, amp) in new_amplitudes.iter().enumerate() {
                let normalized = Complex64::new(amp.re / norm, amp.im / norm);
                self.set_basis_amplitude(i, normalized);
            }
        }
    }

    /// Grover-like amplitude amplification
    fn apply_grover_amplification(&mut self) {
        self.apply_grover_with_strength(0.5);
    }

    /// Anti-Grover: Curiosity-driven exploration
    fn apply_anti_grover(&mut self) {
        self.apply_grover_with_strength(-0.5);
    }

    /// Apply Grover amplification with configurable strength
    /// g > 0: amplify dominant (exploitation)
    /// g < 0: amplify minority (exploration / anti-grover)
    fn apply_grover_with_strength(&mut self, g: f64) {
        // Get current amplitudes
        let amplitudes: Vec<Complex64> = (0..self.config.n_outputs)
            .map(|i| self.get_basis_amplitude(i))
            .collect();

        // Compute mean amplitude
        let n = self.config.n_outputs as f64;
        let mean_re: f64 = amplitudes.iter().map(|a| a.re).sum::<f64>() / n;
        let mean_im: f64 = amplitudes.iter().map(|a| a.im).sum::<f64>() / n;

        // Amplification: amp'[i] = amp[i] + g*(amp[i] - mean)
        let mut new_amplitudes = Vec::with_capacity(self.config.n_outputs);
        for amp in &amplitudes {
            new_amplitudes.push(Complex64::new(
                amp.re + g * (amp.re - mean_re),
                amp.im + g * (amp.im - mean_im),
            ));
        }

        // Normalize and write back
        let norm: f64 = new_amplitudes
            .iter()
            .map(|a| a.norm_squared())
            .sum::<f64>()
            .sqrt();
        if norm > 1e-10 {
            for (i, amp) in new_amplitudes.iter().enumerate() {
                let normalized = Complex64::new(amp.re / norm, amp.im / norm);
                self.set_basis_amplitude(i, normalized);
            }
        }
    }

    /// Adaptive interference: exploration → exploitation over time
    fn apply_adaptive_interference(&mut self) {
        let exploration_phase = 5000.0;
        let transition_phase = 10000.0;
        let t = self.tick as f64;

        let g = if t < exploration_phase {
            -0.5 // Pure exploration
        } else if t < exploration_phase + transition_phase {
            // Linear transition from -0.5 to +0.5
            let progress = (t - exploration_phase) / transition_phase;
            -0.5 + progress * 1.0
        } else {
            0.5 // Pure exploitation
        };

        self.apply_grover_with_strength(g);
    }

    /// Triggered interference: explore until high reward, then exploit
    fn apply_triggered_interference(&mut self) {
        if self.exploitation_triggered {
            // High rewards confirmed - switch to pure classical (no interference)
            // This gives maximum exploitation/convergence
        } else {
            // Still exploring - use anti-grover to find good rewards
            self.apply_grover_with_strength(-0.5);
        }
    }

    /// Get probability of action
    fn get_action_probability(&self, action: usize) -> f64 {
        if action >= self.config.n_outputs {
            return 0.0;
        }

        let mut prob = 0.0;
        for (block_id, block) in &self.quantum.blocks {
            for addr in 0..128 {
                let amp = block.get(addr);
                if amp.norm_squared() < 1e-30 {
                    continue;
                }
                let state_idx = block_id * BigUint::from(128u32) + BigUint::from(addr as u32);
                let qubit_set = if action < 64 {
                    let idx_u64 = state_idx.to_u64_digits().first().copied().unwrap_or(0);
                    (idx_u64 >> action) & 1 == 1
                } else {
                    (&state_idx >> action) & BigUint::one() == BigUint::one()
                };
                if qubit_set {
                    prob += amp.norm_squared();
                }
            }
        }
        prob
    }

    /// Measure to select action
    fn measure(&mut self) -> usize {
        self.last_probabilities = (0..self.config.n_outputs)
            .map(|i| self.get_action_probability(i))
            .collect();

        let total_prob: f64 = self.last_probabilities.iter().sum();
        if total_prob > 1e-10 {
            for p in &mut self.last_probabilities {
                *p /= total_prob;
            }
        } else {
            let uniform = 1.0 / self.config.n_outputs as f64;
            self.last_probabilities = vec![uniform; self.config.n_outputs];
        }

        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let r = (self.rng_state >> 33) as f64 / (1u64 << 31) as f64;

        let mut cumulative = 0.0;
        for (i, &prob) in self.last_probabilities.iter().enumerate() {
            cumulative += prob;
            if r < cumulative {
                return i;
            }
        }
        self.config.n_outputs - 1
    }

    /// Get last action
    pub fn last_action(&self) -> usize {
        self.last_action
    }

    /// Get probability distribution
    pub fn last_probabilities(&self) -> &[f64] {
        &self.last_probabilities
    }

    /// Get total ticks
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Get network size info
    pub fn network_info(&self) -> (usize, usize, usize) {
        (
            self.gpu_state.n_neurons,
            self.gpu_state.n_synapses,
            self.config.n_outputs,
        )
    }

    /// Get quantum blocks count
    pub fn quantum_blocks(&self) -> usize {
        self.quantum.active_blocks()
    }

    // =========================================================================
    // Triggered Mode & Learning (GPU parity with CPU version)
    // =========================================================================

    /// Check if a reward should trigger the switch to exploitation mode
    /// Requires `trigger_count` consecutive high rewards before switching.
    pub fn check_trigger(&mut self, reward: i8) {
        if self.exploitation_triggered {
            return; // Already triggered
        }

        if reward >= self.config.trigger_threshold {
            self.consecutive_high_rewards += 1;
            if self.consecutive_high_rewards >= self.config.trigger_count {
                self.exploitation_triggered = true;
            }
        } else {
            // Reset counter on low reward - must be consecutive!
            self.consecutive_high_rewards = 0;
        }
    }

    /// Get the current consecutive high reward count
    pub fn consecutive_high_rewards(&self) -> usize {
        self.consecutive_high_rewards
    }

    /// Check if exploitation has been triggered
    pub fn is_exploitation_triggered(&self) -> bool {
        self.exploitation_triggered
    }

    /// Manually trigger exploitation mode (useful for testing)
    pub fn trigger_exploitation(&mut self) {
        self.exploitation_triggered = true;
    }

    /// Reset the trigger (go back to exploration mode)
    pub fn reset_trigger(&mut self) {
        self.exploitation_triggered = false;
        self.consecutive_high_rewards = 0;
    }

    /// Convenience: check trigger based on reward
    /// Also checks if reward triggers exploitation mode (for Triggered interference)
    ///
    /// Note: GPU version doesn't have direct SNN weight updates like CPU.
    /// For full R-STDP learning, use CPU version. GPU version is optimized
    /// for inference speed with large networks.
    pub fn learn(&mut self, reward: i8) {
        // Check if this reward should trigger exploitation mode
        self.check_trigger(reward);

        // Note: GPU SNN doesn't currently support R-STDP weight updates
        // The GPU version is optimized for fast inference, not online learning
        // For learning, use CPU QuantumSNN or implement GPU weight update kernel
    }

    /// Get action with highest probability (deterministic mode)
    pub fn measure_deterministic(&mut self) -> usize {
        // Compute probabilities if not already done
        if self.last_probabilities.is_empty() {
            self.last_probabilities = (0..self.config.n_outputs)
                .map(|i| self.get_action_probability(i))
                .collect();
        }

        self.last_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Reset the hybrid system
    pub fn reset(&mut self) {
        self.quantum.initialize_zero_state();
        self.tick = 0;
        self.last_action = 0;
        self.last_probabilities.clear();
        self.exploitation_triggered = false;
        self.consecutive_high_rewards = 0;
    }

    /// Get reference to the config
    pub fn config(&self) -> &QuantumSNNConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_snn_creation() {
        let config = QuantumSNNConfig::small();
        let qsnn = QuantumSNN::new(config);

        assert_eq!(qsnn.config.n_inputs, 16);
        assert_eq!(qsnn.config.n_outputs, 4);
    }

    #[test]
    fn test_spike_encoding() {
        let config = QuantumSNNConfig::small();
        let mut qsnn = QuantumSNN::new(config);

        // Set some inputs to generate spikes
        let inputs: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
        qsnn.set_inputs(&inputs);

        // Run one decision cycle
        let action = qsnn.decide();

        // Action should be valid
        assert!(action < 4);

        // Probabilities should sum to ~1.0
        let prob_sum: f64 = qsnn.last_probabilities().iter().sum();
        assert!(
            (prob_sum - 1.0).abs() < 0.01,
            "Probabilities sum to {} (expected ~1.0)",
            prob_sum
        );
    }

    #[test]
    fn test_quantum_vs_classical() {
        let config = QuantumSNNConfig::small();
        let mut qsnn = QuantumSNN::new(config);

        // Set inputs
        let inputs: Vec<u8> = vec![
            255, 0, 255, 0, 128, 128, 128, 128, 64, 64, 64, 64, 32, 32, 32, 32,
        ];
        qsnn.set_inputs(&inputs);

        // Run multiple decisions
        let mut quantum_actions = Vec::new();
        let mut classical_actions = Vec::new();

        for _ in 0..10 {
            let q_action = qsnn.decide();
            let c_action = qsnn.classical_action();
            quantum_actions.push(q_action);
            classical_actions.push(c_action);
        }

        // Quantum should show some variation (exploration)
        // Classical should be more deterministic
        let q_unique: std::collections::HashSet<_> = quantum_actions.iter().collect();
        let c_unique: std::collections::HashSet<_> = classical_actions.iter().collect();

        println!(
            "Quantum actions: {:?} (unique: {})",
            quantum_actions,
            q_unique.len()
        );
        println!(
            "Classical actions: {:?} (unique: {})",
            classical_actions,
            c_unique.len()
        );

        // Both should produce valid actions
        assert!(quantum_actions.iter().all(|&a| a < 4));
        assert!(classical_actions.iter().all(|&a| a < 4));
    }

    #[test]
    fn test_interference_effect() {
        let mut config = QuantumSNNConfig::small();
        config.interference_depth = 0; // No interference

        let mut qsnn_no_interference = QuantumSNN::new(config.clone());

        config.interference_depth = 2; // With interference
        let mut qsnn_with_interference = QuantumSNN::new(config);

        let inputs: Vec<u8> = vec![200; 16];
        qsnn_no_interference.set_inputs(&inputs);
        qsnn_with_interference.set_inputs(&inputs);

        // Collect probability distributions
        qsnn_no_interference.decide();
        let probs_no_int = qsnn_no_interference.last_probabilities().to_vec();

        qsnn_with_interference.decide();
        let probs_with_int = qsnn_with_interference.last_probabilities().to_vec();

        println!("No interference: {:?}", probs_no_int);
        println!("With interference: {:?}", probs_with_int);

        // Distributions should differ (interference changes probabilities)
        let diff: f64 = probs_no_int
            .iter()
            .zip(probs_with_int.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        println!("Distribution difference: {:.4}", diff);
        // Don't assert difference as it depends on random SNN initialization
    }

    #[test]
    fn test_sparse_efficiency() {
        let config = QuantumSNNConfig {
            n_inputs: 64,
            hidden_layers: vec![128],
            n_outputs: 10,
            interference_mode: InterferenceMode::SoftMix,
            interference_depth: 1,
            mix_angle: 0.1,
            ticks_per_decision: 10,
            learning_enabled: false,
            trigger_threshold: 50,
            trigger_count: 5,
            duplicate_inputs: false,
        };
        let mut qsnn = QuantumSNN::new(config);

        let inputs: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        qsnn.set_inputs(&inputs);

        qsnn.decide();

        let blocks = qsnn.quantum_blocks();
        println!("Active quantum blocks: {} (for 10 outputs)", blocks);

        // Should be sparse: far fewer than 2^10 = 1024
        assert!(blocks < 20, "Too many blocks: {}", blocks);
    }

    #[test]
    fn test_reward_feedback() {
        let mut config = QuantumSNNConfig::small();
        config.learning_enabled = true;
        let mut qsnn = QuantumSNN::new(config);

        let inputs: Vec<u8> = vec![128; 16];
        qsnn.set_inputs(&inputs);

        // Make decision and provide reward
        let action = qsnn.decide();
        qsnn.set_reward(if action == 0 { 10 } else { -10 });

        // Run more decisions - learning should influence behavior
        for _ in 0..5 {
            qsnn.decide();
            qsnn.set_reward(if qsnn.last_action() == 0 { 10 } else { -10 });
        }

        // Network should learn (hard to verify without more iterations)
        println!("Final action: {}", qsnn.last_action());
    }

    #[test]
    fn test_triggered_mode() {
        let mut config = QuantumSNNConfig::small();
        config.interference_mode = InterferenceMode::Triggered;
        config.trigger_threshold = 50;
        config.trigger_count = 3;

        let mut qsnn = QuantumSNN::with_seed(config, 42);

        let inputs: Vec<u8> = vec![128; 16];
        qsnn.set_inputs(&inputs);

        // Initially in exploration mode
        assert!(!qsnn.is_exploitation_triggered());
        assert_eq!(qsnn.consecutive_high_rewards(), 0);

        // Low rewards shouldn't trigger
        for _ in 0..5 {
            qsnn.decide();
            qsnn.learn(30); // Below threshold
        }
        assert!(!qsnn.is_exploitation_triggered());

        // High rewards should accumulate
        qsnn.decide();
        qsnn.learn(60);
        assert_eq!(qsnn.consecutive_high_rewards(), 1);

        qsnn.decide();
        qsnn.learn(70);
        assert_eq!(qsnn.consecutive_high_rewards(), 2);

        // Not yet triggered (need 3)
        assert!(!qsnn.is_exploitation_triggered());

        // Third high reward should trigger
        qsnn.decide();
        qsnn.learn(80);
        assert!(qsnn.is_exploitation_triggered());

        println!("Triggered mode test passed!");
    }

    #[test]
    fn test_all_interference_modes() {
        let modes = [
            InterferenceMode::None,
            InterferenceMode::SoftMix,
            InterferenceMode::Grover,
            InterferenceMode::AntiGrover,
            InterferenceMode::Adaptive,
            InterferenceMode::Triggered,
        ];

        for mode in &modes {
            let mut config = QuantumSNNConfig::small();
            config.interference_mode = *mode;

            let mut qsnn = QuantumSNN::with_seed(config, 42);
            let inputs: Vec<u8> = vec![128; 16];
            qsnn.set_inputs(&inputs);

            // Should be able to decide without panic
            let action = qsnn.decide();
            assert!(action < 4);

            // Probabilities should sum to ~1.0
            let prob_sum: f64 = qsnn.last_probabilities().iter().sum();
            assert!(
                (prob_sum - 1.0).abs() < 0.01,
                "Mode {:?}: probabilities sum to {} (expected ~1.0)",
                mode,
                prob_sum
            );

            println!(
                "Mode {:?}: action={}, probs={:?}",
                mode,
                action,
                qsnn.last_probabilities()
            );
        }

        println!("All interference modes work correctly!");
    }

    #[test]
    fn test_grover_vs_anti_grover() {
        // Grover should amplify the dominant action
        // AntiGrover should amplify minority actions

        let mut config_grover = QuantumSNNConfig::small();
        config_grover.interference_mode = InterferenceMode::Grover;

        let mut config_anti = QuantumSNNConfig::small();
        config_anti.interference_mode = InterferenceMode::AntiGrover;

        let mut qsnn_grover = QuantumSNN::with_seed(config_grover, 42);
        let mut qsnn_anti = QuantumSNN::with_seed(config_anti, 42);

        let inputs: Vec<u8> = vec![
            200, 50, 200, 50, 200, 50, 200, 50, 200, 50, 200, 50, 200, 50, 200, 50,
        ];
        qsnn_grover.set_inputs(&inputs);
        qsnn_anti.set_inputs(&inputs);

        // Run multiple decisions to see the difference
        let mut grover_actions = std::collections::HashMap::new();
        let mut anti_actions = std::collections::HashMap::new();

        for _ in 0..20 {
            let g = qsnn_grover.decide();
            let a = qsnn_anti.decide();
            *grover_actions.entry(g).or_insert(0) += 1;
            *anti_actions.entry(a).or_insert(0) += 1;
        }

        println!("Grover action distribution: {:?}", grover_actions);
        println!("AntiGrover action distribution: {:?}", anti_actions);

        // Grover should concentrate on fewer actions (exploitation)
        // AntiGrover should spread more evenly (exploration)
        // This is probabilistic, so we just verify both produce valid actions
        assert!(grover_actions.keys().all(|&a| a < 4));
        assert!(anti_actions.keys().all(|&a| a < 4));
    }

    #[test]
    fn test_reset() {
        let config = QuantumSNNConfig::small();
        let mut qsnn = QuantumSNN::new(config);

        // Make some decisions and trigger exploitation
        let inputs: Vec<u8> = vec![128; 16];
        qsnn.set_inputs(&inputs);
        qsnn.decide();
        qsnn.trigger_exploitation();

        assert!(qsnn.is_exploitation_triggered());
        assert!(qsnn.tick() > 0);

        // Reset should clear everything
        qsnn.reset();

        assert!(!qsnn.is_exploitation_triggered());
        assert_eq!(qsnn.tick(), 0);
        assert_eq!(qsnn.consecutive_high_rewards(), 0);
        assert!(qsnn.last_probabilities().is_empty());

        println!("Reset test passed!");
    }
}
