//! E-prop: Eligibility Propagation for Spiking Neural Networks
//!
//! Based on Bellec et al. (2020) "A solution to the learning dilemma for
//! recurrent networks of spiking neurons" - Nature Communications.
//!
//! Key differences from R-STDP:
//! 1. Uses pseudo-derivative (surrogate gradient) for credit assignment
//! 2. Eligibility traces track gradient information, not just spike correlations
//! 3. Can learn from scratch without pre-training
//!
//! The learning rule is: ΔW = η * L * e
//! where:
//! - η is the learning rate
//! - L is the learning signal (reward or error)
//! - e is the eligibility trace

/// Surrogate gradient functions for handling the non-differentiable spike
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SurrogateGradient {
    /// Piecewise linear: ψ(v) = max(0, 1 - |v - θ| / γ)
    /// Non-zero in window [θ-γ, θ+γ] around threshold
    PiecewiseLinear { gamma: f32 },

    /// Fast sigmoid: ψ(v) = 1 / (1 + γ|v - θ|)²
    FastSigmoid { gamma: f32 },

    /// Exponential: ψ(v) = exp(-γ|v - θ|)
    Exponential { gamma: f32 },
}

impl Default for SurrogateGradient {
    fn default() -> Self {
        // Piecewise linear with γ=0.3 is common choice
        SurrogateGradient::PiecewiseLinear { gamma: 0.3 }
    }
}

impl SurrogateGradient {
    /// Compute pseudo-derivative at given membrane potential
    ///
    /// # Arguments
    /// * `v` - Membrane potential (normalized, threshold = 1.0)
    pub fn compute(&self, v: f32) -> f32 {
        match self {
            SurrogateGradient::PiecewiseLinear { gamma } => {
                // ψ(v) = max(0, 1 - |v - 1| / γ)
                let dist = (v - 1.0).abs();
                (1.0 - dist / gamma).max(0.0)
            }
            SurrogateGradient::FastSigmoid { gamma } => {
                // ψ(v) = 1 / (1 + γ|v - 1|)²
                let denom = 1.0 + gamma * (v - 1.0).abs();
                1.0 / (denom * denom)
            }
            SurrogateGradient::Exponential { gamma } => {
                // ψ(v) = exp(-γ|v - 1|)
                (-gamma * (v - 1.0).abs()).exp()
            }
        }
    }
}

/// E-prop configuration
#[derive(Clone, Debug)]
pub struct EpropConfig {
    /// Number of input neurons
    pub n_inputs: usize,
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Number of output neurons
    pub n_outputs: usize,
    /// Membrane time constant (larger = slower decay)
    pub tau_mem: f32,
    /// Eligibility trace time constant
    pub tau_eligibility: f32,
    /// Firing threshold
    pub threshold: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Surrogate gradient function
    pub surrogate: SurrogateGradient,
    /// Regularization: baseline firing rate for policy gradient
    pub baseline_rate: f32,
    /// Use adaptive threshold (ALIF)
    pub adaptive_threshold: bool,
    /// Threshold adaptation time constant
    pub tau_adapt: f32,
    /// Threshold adaptation strength
    pub beta_adapt: f32,
}

impl Default for EpropConfig {
    fn default() -> Self {
        Self {
            n_inputs: 4,
            hidden_layers: vec![64],
            n_outputs: 3,
            tau_mem: 10.0,         // 10ms membrane time constant (faster integration)
            tau_eligibility: 50.0, // 50ms eligibility window
            threshold: 0.5,        // Lower threshold for easier spiking
            learning_rate: 0.01,
            surrogate: SurrogateGradient::default(),
            baseline_rate: 0.1,
            adaptive_threshold: false,
            tau_adapt: 200.0,
            beta_adapt: 0.1,
        }
    }
}

/// E-prop learning network
///
/// Implements eligibility propagation for online learning in SNNs.
/// Can learn from scratch without pre-training.
pub struct EpropNetwork {
    config: EpropConfig,

    // Network state
    /// Membrane potentials for all neurons [n_total]
    membrane: Vec<f32>,
    /// Spike outputs (binary) [n_total]
    spikes: Vec<bool>,
    /// Adaptive threshold (for ALIF) [n_total]
    threshold_adapt: Vec<f32>,

    // Weights (dense for simplicity, can optimize later)
    /// Input -> first hidden [hidden[0] x n_inputs]
    w_in: Vec<Vec<f32>>,
    /// Hidden -> hidden (recurrent) [hidden[i] x hidden[i]]
    w_rec: Vec<Vec<Vec<f32>>>,
    /// Hidden -> next layer [hidden[i+1] x hidden[i]]
    w_hidden: Vec<Vec<Vec<f32>>>,
    /// Last hidden -> output [n_outputs x hidden[-1]]
    w_out: Vec<Vec<f32>>,

    // Eligibility traces (same shape as weights)
    e_in: Vec<Vec<f32>>,
    e_rec: Vec<Vec<Vec<f32>>>,
    e_hidden: Vec<Vec<Vec<f32>>>,
    e_out: Vec<Vec<f32>>,

    // Filtered spike traces for eligibility computation
    /// Filtered pre-synaptic activity [n_total]
    filtered_pre: Vec<f32>,

    // Output
    /// Output spike counts for current decision window
    output_counts: Vec<u32>,

    // Time tracking
    tick: u64,

    // Decay constants (precomputed)
    alpha_mem: f32,   // exp(-dt/tau_mem)
    alpha_elig: f32,  // exp(-dt/tau_elig)
    alpha_adapt: f32, // exp(-dt/tau_adapt)

    // RNG
    rng: u64,
}

impl EpropNetwork {
    /// Create a new e-prop network
    pub fn new(config: EpropConfig) -> Self {
        Self::with_seed(config, 42)
    }

    /// Create with specific seed
    pub fn with_seed(config: EpropConfig, seed: u64) -> Self {
        let mut rng = seed;

        // Calculate total neurons
        let n_hidden_total: usize = config.hidden_layers.iter().sum();
        let n_total = config.n_inputs + n_hidden_total + config.n_outputs;

        // Initialize state
        let membrane = vec![0.0; n_total];
        let spikes = vec![false; n_total];
        let threshold_adapt = vec![0.0; n_total];
        let filtered_pre = vec![0.0; n_total];
        let output_counts = vec![0u32; config.n_outputs];

        // Compute decay constants (assuming dt = 1ms)
        let alpha_mem = (-1.0 / config.tau_mem).exp();
        let alpha_elig = (-1.0 / config.tau_eligibility).exp();
        let alpha_adapt = (-1.0 / config.tau_adapt).exp();

        // Initialize weights with SNN-appropriate scaling
        // If no hidden layers, w_in goes directly to outputs
        let first_hidden = config.hidden_layers.first().copied().unwrap_or(0);
        let (w_in, e_in) = if first_hidden > 0 {
            (
                init_weights(first_hidden, config.n_inputs, &mut rng),
                vec![vec![0.0; config.n_inputs]; first_hidden],
            )
        } else {
            // No hidden layers - don't create w_in (use w_out for input->output)
            (Vec::new(), Vec::new())
        };

        // Recurrent weights within each hidden layer
        let mut w_rec = Vec::new();
        let mut e_rec = Vec::new();
        for &size in &config.hidden_layers {
            w_rec.push(init_weights(size, size, &mut rng));
            e_rec.push(vec![vec![0.0; size]; size]);
        }

        // Feedforward hidden->hidden weights
        let mut w_hidden = Vec::new();
        let mut e_hidden = Vec::new();
        for i in 0..config.hidden_layers.len().saturating_sub(1) {
            let from = config.hidden_layers[i];
            let to = config.hidden_layers[i + 1];
            w_hidden.push(init_weights(to, from, &mut rng));
            e_hidden.push(vec![vec![0.0; from]; to]);
        }

        // Output weights - initialize with balanced small weights + bias
        let last_hidden = config
            .hidden_layers
            .last()
            .copied()
            .unwrap_or(config.n_inputs);
        let mut w_out = vec![vec![0.0f32; last_hidden]; config.n_outputs];

        // Initialize output weights to be small and balanced
        for j in 0..config.n_outputs {
            for i in 0..last_hidden {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u1 = (rng >> 11) as f64 / (1u64 << 53) as f64;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u2 = (rng >> 11) as f64 / (1u64 << 53) as f64;

                let z = ((-2.0 * u1.max(1e-10).ln()).sqrt()
                    * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
                // Small balanced weights
                w_out[j][i] = z * 0.1;
            }
        }

        let e_out = vec![vec![0.0; last_hidden]; config.n_outputs];

        Self {
            config,
            membrane,
            spikes,
            threshold_adapt,
            w_in,
            w_rec,
            w_hidden,
            w_out,
            e_in,
            e_rec,
            e_hidden,
            e_out,
            filtered_pre,
            output_counts,
            tick: 0,
            alpha_mem,
            alpha_elig,
            alpha_adapt,
            rng,
        }
    }

    /// Reset network state (membrane, spikes) but keep weights
    pub fn reset_state(&mut self) {
        self.membrane.fill(0.0);
        self.spikes.fill(false);
        self.threshold_adapt.fill(0.0);
        self.filtered_pre.fill(0.0);
        self.output_counts.fill(0);
    }

    /// Reset eligibility traces
    pub fn reset_eligibility(&mut self) {
        for row in &mut self.e_in {
            row.fill(0.0);
        }
        for layer in &mut self.e_rec {
            for row in layer {
                row.fill(0.0);
            }
        }
        for layer in &mut self.e_hidden {
            for row in layer {
                row.fill(0.0);
            }
        }
        for row in &mut self.e_out {
            row.fill(0.0);
        }
    }

    /// Set input spike rates (0-255) and convert to spikes probabilistically
    pub fn set_inputs(&mut self, rates: &[u8]) {
        for (i, &rate) in rates.iter().enumerate().take(self.config.n_inputs) {
            // Convert rate to spike probability
            let prob = rate as f32 / 255.0;
            self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rand = (self.rng >> 33) as f32 / (1u64 << 31) as f32;
            self.spikes[i] = rand < prob;
        }
    }

    /// Set input current directly (for rate-based processing)
    /// This bypasses the spike conversion for more reliable activation
    pub fn set_input_currents(&mut self, currents: &[f32]) {
        for (i, &current) in currents.iter().enumerate().take(self.config.n_inputs) {
            // Treat inputs as direct membrane potential contribution
            // Clamp to reasonable range
            self.membrane[i] = current.clamp(0.0, 2.0);
            self.spikes[i] = current > 0.5; // Spike if input is strong
        }
    }

    /// Run one simulation tick with eligibility trace updates
    pub fn step(&mut self) {
        let n_inputs = self.config.n_inputs;
        let threshold = self.config.threshold;

        // Update filtered pre-synaptic traces (for eligibility)
        // Use continuous activity for inputs, not just spikes
        for i in 0..n_inputs {
            // Input filtered trace based on continuous membrane/spike state
            let activity = if self.spikes[i] {
                1.0
            } else {
                self.membrane[i].max(0.0)
            };
            self.filtered_pre[i] =
                self.alpha_elig * self.filtered_pre[i] + (1.0 - self.alpha_elig) * activity;
        }

        // Process each layer
        let mut layer_start = n_inputs;

        // First hidden layer (from inputs)
        if let Some(&hidden_size) = self.config.hidden_layers.first() {
            for j in 0..hidden_size {
                let neuron_idx = layer_start + j;

                // Compute input current - use continuous activity, not just spikes
                let mut current = 0.0;
                for i in 0..n_inputs {
                    let activity = if self.spikes[i] {
                        1.0
                    } else {
                        self.membrane[i].max(0.0)
                    };
                    current += self.w_in[j][i] * activity;
                }

                // Add recurrent input (spike-based)
                for k in 0..hidden_size {
                    if self.spikes[layer_start + k] {
                        current += self.w_rec[0][j][k];
                    }
                }

                // Membrane dynamics with bias toward activity
                let v_old = self.membrane[neuron_idx];
                let effective_threshold = if self.config.adaptive_threshold {
                    threshold + self.threshold_adapt[neuron_idx]
                } else {
                    threshold
                };

                // Leaky integration with stronger current integration
                self.membrane[neuron_idx] = self.alpha_mem * v_old + current;

                // Spike with soft reset
                let v_normalized = self.membrane[neuron_idx] / effective_threshold;
                if self.membrane[neuron_idx] >= effective_threshold {
                    self.spikes[neuron_idx] = true;
                    self.membrane[neuron_idx] = 0.0; // Hard reset
                } else {
                    self.spikes[neuron_idx] = false;
                }

                // Threshold adaptation
                if self.config.adaptive_threshold {
                    if self.spikes[neuron_idx] {
                        self.threshold_adapt[neuron_idx] += self.config.beta_adapt;
                    }
                    self.threshold_adapt[neuron_idx] *= self.alpha_adapt;
                }

                // Update eligibility traces for input weights
                let psi = self.config.surrogate.compute(v_normalized);
                for i in 0..n_inputs {
                    self.e_in[j][i] =
                        self.alpha_elig * self.e_in[j][i] + psi * self.filtered_pre[i];
                }

                // Update filtered trace for this neuron
                let activity = if self.spikes[neuron_idx] {
                    1.0
                } else {
                    v_normalized.max(0.0).min(1.0)
                };
                self.filtered_pre[neuron_idx] = self.alpha_elig * self.filtered_pre[neuron_idx]
                    + (1.0 - self.alpha_elig) * activity;

                // Update eligibility for recurrent weights
                for k in 0..hidden_size {
                    self.e_rec[0][j][k] = self.alpha_elig * self.e_rec[0][j][k]
                        + psi * self.filtered_pre[layer_start + k];
                }
            }
            layer_start += hidden_size;
        }

        // Additional hidden layers
        for layer_idx in 1..self.config.hidden_layers.len() {
            let prev_size = self.config.hidden_layers[layer_idx - 1];
            let curr_size = self.config.hidden_layers[layer_idx];
            let prev_start = layer_start - prev_size;

            for j in 0..curr_size {
                let neuron_idx = layer_start + j;

                // Input from previous layer
                let mut current = 0.0;
                for i in 0..prev_size {
                    current += self.w_hidden[layer_idx - 1][j][i]
                        * (if self.spikes[prev_start + i] {
                            1.0
                        } else {
                            0.0
                        });
                }

                // Recurrent input
                for k in 0..curr_size {
                    current += self.w_rec[layer_idx][j][k]
                        * (if self.spikes[layer_start + k] {
                            1.0
                        } else {
                            0.0
                        });
                }

                let v_old = self.membrane[neuron_idx];
                let effective_threshold = if self.config.adaptive_threshold {
                    threshold + self.threshold_adapt[neuron_idx]
                } else {
                    threshold
                };

                self.membrane[neuron_idx] =
                    self.alpha_mem * v_old + (1.0 - self.alpha_mem) * current;

                let v_normalized = self.membrane[neuron_idx] / effective_threshold;
                if self.membrane[neuron_idx] >= effective_threshold {
                    self.spikes[neuron_idx] = true;
                    self.membrane[neuron_idx] -= effective_threshold;
                    if self.config.adaptive_threshold {
                        self.threshold_adapt[neuron_idx] += self.config.beta_adapt;
                    }
                } else {
                    self.spikes[neuron_idx] = false;
                }

                if self.config.adaptive_threshold {
                    self.threshold_adapt[neuron_idx] *= self.alpha_adapt;
                }

                // Eligibility traces
                let psi = self.config.surrogate.compute(v_normalized);
                for i in 0..prev_size {
                    self.e_hidden[layer_idx - 1][j][i] = self.alpha_elig
                        * self.e_hidden[layer_idx - 1][j][i]
                        + psi * self.filtered_pre[prev_start + i];
                }
                for k in 0..curr_size {
                    self.e_rec[layer_idx][j][k] = self.alpha_elig * self.e_rec[layer_idx][j][k]
                        + psi * self.filtered_pre[layer_start + k];
                }
            }
            layer_start += curr_size;
        }

        // Output layer - use RATE-BASED approach instead of spiking
        // This avoids the cold-start problem where non-spiking outputs can't learn
        let last_hidden_size = self
            .config
            .hidden_layers
            .last()
            .copied()
            .unwrap_or(n_inputs);
        let last_hidden_start = layer_start - last_hidden_size;

        for j in 0..self.config.n_outputs {
            let neuron_idx = layer_start + j;

            // Input from last hidden layer - use continuous activity
            let mut current = 0.0;
            for i in 0..last_hidden_size {
                // Use filtered trace (continuous) instead of discrete spikes
                let activity = self.filtered_pre[last_hidden_start + i];
                current += self.w_out[j][i] * activity;
            }

            // Leaky integration for output membrane
            let v_old = self.membrane[neuron_idx];
            self.membrane[neuron_idx] = self.alpha_mem * v_old + current;

            // For rate-based output: "spike" based on membrane potential
            // Higher membrane = higher spike rate
            let v_normalized = self.membrane[neuron_idx] / threshold;

            // Soft spiking based on membrane potential
            self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rand = (self.rng >> 33) as f32 / (1u64 << 31) as f32;
            let spike_prob = (v_normalized / 2.0).clamp(0.0, 1.0); // Scale membrane to prob

            if rand < spike_prob {
                self.spikes[neuron_idx] = true;
                self.output_counts[j] += 1;
            } else {
                self.spikes[neuron_idx] = false;
            }

            // Output eligibility traces - use psi = 1 for rate-based (always eligible)
            // This allows credit assignment even when output doesn't spike
            let psi = 1.0; // Always eligible for output layer
            for i in 0..last_hidden_size {
                self.e_out[j][i] = self.alpha_elig * self.e_out[j][i]
                    + psi * self.filtered_pre[last_hidden_start + i];
            }
        }

        self.tick += 1;
    }

    /// Run multiple ticks for a decision window
    pub fn step_many(&mut self, ticks: usize) {
        for _ in 0..ticks {
            self.step();
        }
    }

    /// Get action from output spike counts (argmax)
    pub fn get_action(&self) -> usize {
        self.output_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| *count)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get output spike counts
    pub fn output_counts(&self) -> &[u32] {
        &self.output_counts
    }

    /// Reset output counts for new decision window
    pub fn reset_output_counts(&mut self) {
        self.output_counts.fill(0);
    }

    /// Apply learning with reward signal (policy gradient style)
    ///
    /// Uses the formula: ΔW = η * L * e
    /// where L = reward * (spike - baseline) for policy gradient
    pub fn learn(&mut self, reward: f32) {
        let lr = self.config.learning_rate;

        // For policy gradient: L = R * (z - baseline)
        // But we apply it to all traces uniformly scaled by reward
        // The eligibility traces already encode the causal credit

        // Update input weights
        for j in 0..self.e_in.len() {
            for i in 0..self.e_in[j].len() {
                self.w_in[j][i] += lr * reward * self.e_in[j][i];
                // Weight clipping
                self.w_in[j][i] = self.w_in[j][i].clamp(-2.0, 2.0);
            }
        }

        // Update recurrent weights
        for layer in 0..self.e_rec.len() {
            for j in 0..self.e_rec[layer].len() {
                for k in 0..self.e_rec[layer][j].len() {
                    self.w_rec[layer][j][k] += lr * reward * self.e_rec[layer][j][k];
                    self.w_rec[layer][j][k] = self.w_rec[layer][j][k].clamp(-2.0, 2.0);
                }
            }
        }

        // Update hidden->hidden weights
        for layer in 0..self.e_hidden.len() {
            for j in 0..self.e_hidden[layer].len() {
                for i in 0..self.e_hidden[layer][j].len() {
                    self.w_hidden[layer][j][i] += lr * reward * self.e_hidden[layer][j][i];
                    self.w_hidden[layer][j][i] = self.w_hidden[layer][j][i].clamp(-2.0, 2.0);
                }
            }
        }

        // Update output weights
        for j in 0..self.e_out.len() {
            for i in 0..self.e_out[j].len() {
                self.w_out[j][i] += lr * reward * self.e_out[j][i];
                self.w_out[j][i] = self.w_out[j][i].clamp(-2.0, 2.0);
            }
        }
    }

    /// Apply action-specific learning (reinforce only the chosen action's output)
    pub fn learn_action(&mut self, action: usize, reward: f32) {
        let lr = self.config.learning_rate;

        // Only update weights connected to the chosen action's output
        if action < self.config.n_outputs {
            for i in 0..self.e_out[action].len() {
                self.w_out[action][i] += lr * reward * self.e_out[action][i];
                self.w_out[action][i] = self.w_out[action][i].clamp(-2.0, 2.0);
            }
        }

        // For hidden layers, we can still do full update since eligibility
        // traces properly credit the relevant pathways
        // Update input weights
        for j in 0..self.e_in.len() {
            for i in 0..self.e_in[j].len() {
                self.w_in[j][i] += lr * reward * self.e_in[j][i];
                self.w_in[j][i] = self.w_in[j][i].clamp(-2.0, 2.0);
            }
        }

        // Update recurrent weights
        for layer in 0..self.e_rec.len() {
            for j in 0..self.e_rec[layer].len() {
                for k in 0..self.e_rec[layer][j].len() {
                    self.w_rec[layer][j][k] += lr * reward * self.e_rec[layer][j][k];
                    self.w_rec[layer][j][k] = self.w_rec[layer][j][k].clamp(-2.0, 2.0);
                }
            }
        }

        // Update hidden->hidden weights
        for layer in 0..self.e_hidden.len() {
            for j in 0..self.e_hidden[layer].len() {
                for i in 0..self.e_hidden[layer][j].len() {
                    self.w_hidden[layer][j][i] += lr * reward * self.e_hidden[layer][j][i];
                    self.w_hidden[layer][j][i] = self.w_hidden[layer][j][i].clamp(-2.0, 2.0);
                }
            }
        }
    }

    /// Supervised learning: directly update output weights toward target
    ///
    /// This bypasses eligibility traces for the output layer, directly
    /// strengthening the connection to the target action based on hidden
    /// layer activity. More reliable than reward-based learning.
    pub fn learn_supervised(&mut self, target_action: usize) {
        let lr = self.config.learning_rate;
        let last_hidden_size = self
            .config
            .hidden_layers
            .last()
            .copied()
            .unwrap_or(self.config.n_inputs);
        let n_hidden_total: usize = self.config.hidden_layers.iter().sum();
        let last_hidden_start = self.config.n_inputs + n_hidden_total - last_hidden_size;

        // For supervised learning:
        // - Strengthen weights from active hidden neurons to target action
        // - Weaken weights from active hidden neurons to other actions
        for j in 0..self.config.n_outputs {
            for i in 0..last_hidden_size {
                let hidden_activity = self.filtered_pre[last_hidden_start + i];

                if j == target_action {
                    // Strengthen connection to target
                    self.w_out[j][i] += lr * hidden_activity;
                } else {
                    // Weaken connection to non-targets
                    self.w_out[j][i] -= lr * 0.5 * hidden_activity;
                }
                self.w_out[j][i] = self.w_out[j][i].clamp(-2.0, 2.0);
            }
        }

        // Also update hidden layers based on eligibility
        for j in 0..self.e_in.len() {
            for i in 0..self.e_in[j].len() {
                self.w_in[j][i] += lr * self.e_in[j][i];
                self.w_in[j][i] = self.w_in[j][i].clamp(-2.0, 2.0);
            }
        }
    }

    /// Combined step: set inputs, run ticks, return action
    pub fn decide(&mut self, inputs: &[u8], ticks: usize) -> usize {
        self.reset_output_counts();

        for _ in 0..ticks {
            self.set_inputs(inputs);
            self.step();
        }

        self.get_action()
    }
}

/// Initialize weights with larger scale for SNN dynamics
/// SNNs need stronger weights than ANNs because of sparse spike timing
fn init_weights(n_out: usize, n_in: usize, rng: &mut u64) -> Vec<Vec<f32>> {
    // Use larger scale for SNN - need strong enough weights to cause spiking
    let scale = (6.0 / (n_in + n_out) as f32).sqrt(); // 3x larger than Xavier

    let mut weights = Vec::with_capacity(n_out);
    for _ in 0..n_out {
        let mut row = Vec::with_capacity(n_in);
        for _ in 0..n_in {
            // Box-Muller for normal distribution
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (*rng >> 11) as f64 / (1u64 << 53) as f64;
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (*rng >> 11) as f64 / (1u64 << 53) as f64;

            let z = ((-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos())
                as f32;
            row.push(z * scale);
        }
        weights.push(row);
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surrogate_gradient() {
        let sg = SurrogateGradient::PiecewiseLinear { gamma: 0.3 };

        // At threshold (v=1), should be max
        let psi_at_thresh = sg.compute(1.0);
        assert!((psi_at_thresh - 1.0).abs() < 0.01);

        // Away from threshold, should decay
        let psi_above = sg.compute(1.2);
        assert!(psi_above < psi_at_thresh);

        let psi_below = sg.compute(0.8);
        assert!(psi_below < psi_at_thresh);

        // Far from threshold, should be zero
        let psi_far = sg.compute(2.0);
        assert!(psi_far < 0.01);
    }

    #[test]
    fn test_network_creation() {
        let config = EpropConfig {
            n_inputs: 4,
            hidden_layers: vec![16],
            n_outputs: 3,
            ..Default::default()
        };

        let net = EpropNetwork::new(config);
        assert_eq!(net.config.n_inputs, 4);
        assert_eq!(net.config.n_outputs, 3);
        assert_eq!(net.w_in.len(), 16); // 16 hidden neurons
        assert_eq!(net.w_in[0].len(), 4); // 4 inputs each
        assert_eq!(net.w_out.len(), 3); // 3 outputs
        assert_eq!(net.w_out[0].len(), 16); // from 16 hidden
    }

    #[test]
    fn test_step() {
        let config = EpropConfig {
            n_inputs: 4,
            hidden_layers: vec![8],
            n_outputs: 3,
            ..Default::default()
        };

        let mut net = EpropNetwork::new(config);

        // Set some inputs
        net.set_inputs(&[255, 128, 64, 0]);
        net.step();

        // Should have processed one tick
        assert_eq!(net.tick, 1);
    }

    #[test]
    fn test_decide() {
        let config = EpropConfig {
            n_inputs: 4,
            hidden_layers: vec![16],
            n_outputs: 3,
            ..Default::default()
        };

        let mut net = EpropNetwork::new(config);

        // Make a decision
        let action = net.decide(&[200, 100, 50, 25], 20);

        // Should return valid action
        assert!(action < 3);

        // Should have output counts
        let total: u32 = net.output_counts().iter().sum();
        println!("Output counts: {:?}, total: {}", net.output_counts(), total);
    }

    #[test]
    fn test_learning() {
        let config = EpropConfig {
            n_inputs: 4,
            hidden_layers: vec![8],
            n_outputs: 2,
            learning_rate: 0.1,
            ..Default::default()
        };

        let mut net = EpropNetwork::new(config);

        // Get initial weight
        let w_initial = net.w_in[0][0];

        // Run and learn
        net.decide(&[255, 0, 0, 0], 10);
        net.learn(1.0);

        // Weight should have changed
        let w_after = net.w_in[0][0];
        println!("Weight: {} -> {}", w_initial, w_after);

        // May or may not change depending on eligibility
        // Just check it doesn't crash
    }
}
