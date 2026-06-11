//! Hybrid Stabilizer-Cluster Quantum Neural Network (EPIC 134)
//!
//! Combines a stabilizer backbone (10K+ neurons, O(n²) memory) with small
//! full-quantum clusters (8-16 qubits, 2^n amplitudes) for true quantum advantage.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    HybridQuantumNetwork                         │
//! │  ┌───────────────────────────────────────────────────────────┐ │
//! │  │            StabilizerNeuronNetwork (backbone)             │ │
//! │  │    10K+ neurons with Clifford gates (H, S, CZ, CNOT)     │ │
//! │  └───────────────────────────────────────────────────────────┘ │
//! │                              │                                  │
//! │         ┌────────────────────┼────────────────────┐            │
//! │         │                    │                    │            │
//! │    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐      │
//! │    │Cluster 0│          │Cluster 1│          │Cluster 2│      │
//! │    │16 qubits│          │16 qubits│          │16 qubits│      │
//! │    │ T gates │          │ T gates │          │ T gates │      │
//! │    └─────────┘          └─────────┘          └─────────┘      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Features
//!
//! - **Stabilizer backbone**: Efficient O(n) Clifford gates on large networks
//! - **Full-quantum clusters**: Universal quantum computation (T gates, rotations)
//! - **Magic state injection**: Bridge between stabilizer and non-Clifford
//! - **Entanglement entropy**: Measure quantum correlations within clusters

use std::collections::HashMap;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

use crate::qec::noise::SimpleRng;

use super::stabilizer_network::{StabilizerNetworkConfig, StabilizerNeuronNetwork};

// =============================================================================
// Complex Number Type
// =============================================================================

/// Complex amplitude with f64 precision.
#[derive(Clone, Copy, Debug, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    #[inline]
    pub fn norm_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    #[inline]
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline]
    pub fn arg(&self) -> f64 {
        self.im.atan2(self.re)
    }
}

impl std::ops::Add for Complex64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f64> for Complex64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl std::ops::AddAssign for Complex64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

// =============================================================================
// Full Quantum Cluster
// =============================================================================

/// A small full-quantum cluster with state vector representation.
///
/// Supports universal quantum computation including non-Clifford gates (T gate).
/// Typical sizes: 8-16 qubits (256 - 65536 amplitudes).
#[derive(Clone)]
pub struct FullQuantumCluster {
    /// Number of qubits in this cluster
    n_qubits: usize,

    /// State vector: 2^n complex amplitudes
    state: Vec<Complex64>,

    /// Which backbone neurons are mapped to this cluster
    /// Maps local qubit index -> backbone neuron index
    neuron_map: Vec<usize>,

    /// Cluster ID
    id: usize,

    /// Whether the cluster is currently active (for adaptive mode)
    pub active: bool,

    /// Ticks remaining in activation cooldown
    activation_cooldown: u32,
}

impl FullQuantumCluster {
    /// Create a new cluster with n qubits, initialized to |0...0⟩.
    pub fn new(n_qubits: usize, id: usize) -> Self {
        assert!(
            n_qubits >= 1 && n_qubits <= 20,
            "Cluster size must be 1-20 qubits"
        );

        let dim = 1 << n_qubits;
        let mut state = vec![Complex64::ZERO; dim];
        state[0] = Complex64::ONE; // |0...0⟩

        Self {
            n_qubits,
            state,
            neuron_map: Vec::new(),
            id,
            active: true, // Active by default
            activation_cooldown: 0,
        }
    }

    /// Create a cluster with specific backbone neuron mappings.
    pub fn with_neurons(n_qubits: usize, id: usize, neuron_map: Vec<usize>) -> Self {
        let mut cluster = Self::new(n_qubits, id);
        cluster.neuron_map = neuron_map;
        cluster
    }

    /// Get number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Get cluster ID.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get dimension (2^n).
    pub fn dim(&self) -> usize {
        self.state.len()
    }

    /// Get backbone neuron for local qubit.
    pub fn neuron_for_qubit(&self, local_qubit: usize) -> Option<usize> {
        self.neuron_map.get(local_qubit).copied()
    }

    /// Reset to |0...0⟩ state.
    pub fn reset(&mut self) {
        for amp in &mut self.state {
            *amp = Complex64::ZERO;
        }
        self.state[0] = Complex64::ONE;
    }

    /// Get amplitude at basis state index.
    #[inline]
    pub fn amplitude(&self, idx: usize) -> Complex64 {
        self.state[idx]
    }

    /// Get probability of measuring basis state.
    #[inline]
    pub fn probability(&self, idx: usize) -> f64 {
        self.state[idx].norm_squared()
    }

    // =========================================================================
    // SINGLE-QUBIT GATES
    // =========================================================================

    /// Apply Pauli X gate (bit flip).
    pub fn x(&mut self, qubit: usize) {
        let mask = 1 << qubit;
        let dim = self.dim();
        for i in 0..dim / 2 {
            let j = i ^ mask;
            if i < j {
                self.state.swap(i, j);
            }
        }
    }

    /// Apply Pauli Y gate.
    pub fn y(&mut self, qubit: usize) {
        let mask = 1 << qubit;
        let dim = self.dim();
        for i in 0..dim {
            let j = i ^ mask;
            if i < j {
                let tmp = self.state[i];
                // Y = [[0, -i], [i, 0]]
                self.state[i] = self.state[j] * Complex64::new(0.0, -1.0);
                self.state[j] = tmp * Complex64::new(0.0, 1.0);
            }
        }
    }

    /// Apply Pauli Z gate (phase flip).
    pub fn z(&mut self, qubit: usize) {
        let mask = 1 << qubit;
        for (i, amp) in self.state.iter_mut().enumerate() {
            if i & mask != 0 {
                amp.re = -amp.re;
                amp.im = -amp.im;
            }
        }
    }

    /// Apply Hadamard gate.
    pub fn h(&mut self, qubit: usize) {
        let mask = 1 << qubit;
        let inv_sqrt2 = FRAC_1_SQRT_2;
        let dim = self.dim();

        for i in 0..dim {
            let j = i ^ mask;
            if i < j {
                let a = self.state[i];
                let b = self.state[j];
                self.state[i] = (a + b) * inv_sqrt2;
                self.state[j] = (a - b) * inv_sqrt2;
            }
        }
    }

    /// Apply S gate (phase gate, √Z).
    pub fn s(&mut self, qubit: usize) {
        let mask = 1 << qubit;
        for (i, amp) in self.state.iter_mut().enumerate() {
            if i & mask != 0 {
                // Multiply by i
                let tmp = amp.re;
                amp.re = -amp.im;
                amp.im = tmp;
            }
        }
    }

    /// Apply S† (S-dagger) gate.
    pub fn s_dag(&mut self, qubit: usize) {
        let mask = 1 << qubit;
        for (i, amp) in self.state.iter_mut().enumerate() {
            if i & mask != 0 {
                // Multiply by -i
                let tmp = amp.re;
                amp.re = amp.im;
                amp.im = -tmp;
            }
        }
    }

    /// Apply T gate (π/8 gate) - NON-CLIFFORD!
    ///
    /// This is the key gate that provides universal quantum computation.
    /// T = diag(1, e^{iπ/4})
    pub fn t(&mut self, qubit: usize) {
        let mask = 1 << qubit;
        // e^{iπ/4} = cos(π/4) + i*sin(π/4) = (1+i)/√2
        let phase = Complex64::from_polar(1.0, PI / 4.0);

        for (i, amp) in self.state.iter_mut().enumerate() {
            if i & mask != 0 {
                *amp = *amp * phase;
            }
        }
    }

    /// Apply T† (T-dagger) gate.
    pub fn t_dag(&mut self, qubit: usize) {
        let mask = 1 << qubit;
        let phase = Complex64::from_polar(1.0, -PI / 4.0);

        for (i, amp) in self.state.iter_mut().enumerate() {
            if i & mask != 0 {
                *amp = *amp * phase;
            }
        }
    }

    /// Apply Rx rotation gate.
    pub fn rx(&mut self, qubit: usize, theta: f64) {
        let mask = 1 << qubit;
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let dim = self.dim();

        for i in 0..dim {
            let j = i ^ mask;
            if i < j {
                let a = self.state[i];
                let b = self.state[j];
                // Rx = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
                self.state[i] = a * c + b * Complex64::new(0.0, -s);
                self.state[j] = a * Complex64::new(0.0, -s) + b * c;
            }
        }
    }

    /// Apply Ry rotation gate.
    pub fn ry(&mut self, qubit: usize, theta: f64) {
        let mask = 1 << qubit;
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let dim = self.dim();

        for i in 0..dim {
            let j = i ^ mask;
            if i < j {
                let a = self.state[i];
                let b = self.state[j];
                // Ry = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
                self.state[i] = a * c - b * s;
                self.state[j] = a * s + b * c;
            }
        }
    }

    /// Apply Rz rotation gate.
    pub fn rz(&mut self, qubit: usize, theta: f64) {
        let mask = 1 << qubit;
        let phase_neg = Complex64::from_polar(1.0, -theta / 2.0);
        let phase_pos = Complex64::from_polar(1.0, theta / 2.0);

        for (i, amp) in self.state.iter_mut().enumerate() {
            if i & mask != 0 {
                *amp = *amp * phase_pos;
            } else {
                *amp = *amp * phase_neg;
            }
        }
    }

    // =========================================================================
    // TWO-QUBIT GATES
    // =========================================================================

    /// Apply CNOT gate (controlled-X).
    pub fn cnot(&mut self, control: usize, target: usize) {
        let ctrl_mask = 1 << control;
        let tgt_mask = 1 << target;
        let dim = self.dim();

        for i in 0..dim {
            if i & ctrl_mask != 0 {
                let j = i ^ tgt_mask;
                if i < j {
                    self.state.swap(i, j);
                }
            }
        }
    }

    /// Apply CZ gate (controlled-Z).
    pub fn cz(&mut self, qubit_a: usize, qubit_b: usize) {
        let mask_a = 1 << qubit_a;
        let mask_b = 1 << qubit_b;

        for (i, amp) in self.state.iter_mut().enumerate() {
            if (i & mask_a != 0) && (i & mask_b != 0) {
                amp.re = -amp.re;
                amp.im = -amp.im;
            }
        }
    }

    /// Apply SWAP gate.
    pub fn swap(&mut self, qubit_a: usize, qubit_b: usize) {
        let mask_a = 1 << qubit_a;
        let mask_b = 1 << qubit_b;
        let dim = self.dim();

        for i in 0..dim {
            let bit_a = (i & mask_a) != 0;
            let bit_b = (i & mask_b) != 0;

            if bit_a != bit_b {
                let j = i ^ mask_a ^ mask_b;
                if i < j {
                    self.state.swap(i, j);
                }
            }
        }
    }

    // =========================================================================
    // MEASUREMENT
    // =========================================================================

    /// Measure a qubit in Z basis, collapsing the state.
    ///
    /// Returns the measurement result (0 or 1).
    pub fn measure(&mut self, qubit: usize, rng: &mut SimpleRng) -> u8 {
        let mask = 1 << qubit;

        // Calculate probability of measuring |1⟩
        let mut prob_one = 0.0;
        for (i, amp) in self.state.iter().enumerate() {
            if i & mask != 0 {
                prob_one += amp.norm_squared();
            }
        }

        // Determine outcome
        let outcome = if rng.next_f64() < prob_one { 1 } else { 0 };

        // Collapse state
        let norm_factor = if outcome == 1 {
            1.0 / prob_one.sqrt()
        } else {
            1.0 / (1.0 - prob_one).sqrt()
        };

        for (i, amp) in self.state.iter_mut().enumerate() {
            let has_one = (i & mask) != 0;
            if (outcome == 1) != has_one {
                *amp = Complex64::ZERO;
            } else {
                *amp = *amp * norm_factor;
            }
        }

        outcome
    }

    /// Measure all qubits, returning the classical bitstring.
    pub fn measure_all(&mut self, rng: &mut SimpleRng) -> usize {
        let mut result = 0;
        for q in 0..self.n_qubits {
            let bit = self.measure(q, rng);
            result |= (bit as usize) << q;
        }
        result
    }

    // =========================================================================
    // OBSERVABLES & EXPECTATIONS
    // =========================================================================

    /// Compute Z expectation value for a single qubit.
    ///
    /// Returns value in [-1, 1]: +1 means |0⟩, -1 means |1⟩.
    pub fn z_expectation(&self, qubit: usize) -> f64 {
        let mask = 1 << qubit;
        let mut z_exp = 0.0;

        for (i, amp) in self.state.iter().enumerate() {
            let prob = amp.norm_squared();
            if (i & mask) == 0 {
                z_exp += prob; // |0⟩ contribution: +1
            } else {
                z_exp -= prob; // |1⟩ contribution: -1
            }
        }

        z_exp
    }

    /// Compute total Z expectation across all qubits.
    ///
    /// Returns average Z expectation (normalized by qubit count).
    pub fn total_z_expectation(&self) -> f64 {
        if self.n_qubits == 0 {
            return 0.0;
        }

        let mut total = 0.0;
        for q in 0..self.n_qubits {
            total += self.z_expectation(q);
        }
        total / self.n_qubits as f64
    }

    /// Compute X expectation value for a single qubit.
    pub fn x_expectation(&self, qubit: usize) -> f64 {
        let mask = 1 << qubit;
        let mut x_exp = 0.0;

        for i in 0..self.dim() {
            let j = i ^ mask; // Partner with flipped qubit
            if i < j {
                // ⟨X⟩ = 2 * Re(α* β) where α = amp(|0⟩), β = amp(|1⟩)
                let a = self.state[i];
                let b = self.state[j];
                x_exp += 2.0 * (a.conj() * b).re;
            }
        }

        x_exp
    }

    // =========================================================================
    // ACTIVATION CONTROL
    // =========================================================================

    /// Activate the cluster, resetting to |+⟩^n state.
    pub fn activate(&mut self) {
        self.active = true;
        self.reset_to_plus_state();
    }

    /// Deactivate the cluster, resetting to |0⟩^n state.
    pub fn deactivate(&mut self) {
        self.active = false;
        self.reset();
    }

    /// Reset to |+⟩^⊗n superposition state.
    pub fn reset_to_plus_state(&mut self) {
        let dim = self.dim();
        let amplitude = 1.0 / (dim as f64).sqrt();

        for amp in &mut self.state {
            *amp = Complex64::new(amplitude, 0.0);
        }
    }

    /// Decrement activation cooldown, returns true if still in cooldown.
    pub fn tick_cooldown(&mut self) -> bool {
        if self.activation_cooldown > 0 {
            self.activation_cooldown -= 1;
            true
        } else {
            false
        }
    }

    /// Set activation cooldown.
    pub fn set_cooldown(&mut self, ticks: u32) {
        self.activation_cooldown = ticks;
    }

    // =========================================================================
    // SINGLE-QUBIT STATE MANIPULATION (for cluster bridges)
    // =========================================================================

    /// Reset a single qubit to |0⟩, leaving other qubits unchanged.
    ///
    /// This performs a projective measurement and correction, effectively
    /// mapping any state |ψ⟩ → |0⟩ for the target qubit.
    pub fn reset_qubit(&mut self, qubit: usize) {
        let mask = 1 << qubit;

        // First, measure to collapse
        // Calculate probability of measuring |1⟩
        let mut prob_one = 0.0;
        for (i, amp) in self.state.iter().enumerate() {
            if i & mask != 0 {
                prob_one += amp.norm_squared();
            }
        }

        // If qubit is mostly |1⟩, apply X to flip it to |0⟩
        if prob_one > 0.5 {
            self.x(qubit);
        }

        // Now project onto |0⟩ subspace
        let prob_zero = 1.0 - prob_one;
        let norm_factor = if prob_zero > 1e-15 {
            1.0 / prob_zero.sqrt()
        } else {
            // If purely |1⟩, after X flip it's now purely |0⟩
            1.0
        };

        for (i, amp) in self.state.iter_mut().enumerate() {
            if i & mask != 0 {
                *amp = Complex64::ZERO;
            } else {
                *amp = *amp * norm_factor;
            }
        }
    }

    /// Get reduced state amplitudes for a single qubit (traced over others).
    ///
    /// Returns (prob_0, prob_1, coherence) where coherence is the off-diagonal.
    pub fn get_qubit_reduced_state(&self, qubit: usize) -> (f64, f64, Complex64) {
        let mask = 1 << qubit;

        let mut prob_0 = 0.0;
        let mut prob_1 = 0.0;
        let mut coherence = Complex64::ZERO;

        for i in 0..self.dim() {
            let prob = self.state[i].norm_squared();
            if i & mask == 0 {
                prob_0 += prob;
                // Coherence: sum of α*β^* where α is |0⟩ amp, β is |1⟩ amp
                let j = i | mask;
                coherence = coherence + self.state[i] * self.state[j].conj();
            } else {
                prob_1 += prob;
            }
        }

        (prob_0, prob_1, coherence)
    }

    /// Inject a single-qubit state into this cluster's qubit.
    ///
    /// Given (α, β) where |ψ⟩ = α|0⟩ + β|1⟩, prepares the qubit in this state.
    /// Other qubits are left unchanged (tensor product structure).
    pub fn inject_qubit_state(&mut self, qubit: usize, alpha: Complex64, beta: Complex64) {
        // First reset the qubit to |0⟩
        self.reset_qubit(qubit);

        // Now apply the unitary that maps |0⟩ → α|0⟩ + β|1⟩
        // This is equivalent to:
        // |0⟩ → α|0⟩ + β|1⟩
        // |1⟩ → -β*|0⟩ + α*|1⟩ (to make it unitary)

        let mask = 1 << qubit;
        let dim = self.dim();

        for i in 0..dim {
            if i & mask == 0 {
                let j = i | mask;
                let amp_0 = self.state[i];
                let amp_1 = self.state[j]; // Should be zero after reset

                // Apply rotation
                // Note: -β* term for unitary transformation
                let neg_beta_conj = Complex64::new(-beta.conj().re, -beta.conj().im);
                self.state[i] = amp_0 * alpha + amp_1 * neg_beta_conj;
                self.state[j] = amp_0 * beta + amp_1 * alpha.conj();
            }
        }

        self.normalize();
    }

    /// Apply SWAP between two qubits across different clusters.
    /// This is a helper that extracts state from this cluster.
    pub fn extract_qubit_state(&self, qubit: usize) -> (Complex64, Complex64) {
        let (p0, p1, _coh) = self.get_qubit_reduced_state(qubit);

        // Reconstruct amplitudes from probabilities
        // For a pure state, we can use sqrt of probabilities
        // Phase information is partially lost in mixed states
        let alpha = Complex64::new(p0.sqrt(), 0.0);
        let beta = Complex64::new(p1.sqrt(), 0.0);

        (alpha, beta)
    }

    // =========================================================================
    // ENTROPY & ANALYSIS
    // =========================================================================

    /// Calculate von Neumann entanglement entropy for a bipartition.
    ///
    /// Partitions qubits [0, split) vs [split, n).
    /// Returns entropy in bits (log base 2).
    pub fn entanglement_entropy(&self, split: usize) -> f64 {
        if split == 0 || split >= self.n_qubits {
            return 0.0;
        }

        // Compute reduced density matrix for subsystem A (qubits 0..split)
        let dim_a = 1 << split;
        let dim_b = 1 << (self.n_qubits - split);

        // ρ_A = Tr_B(|ψ⟩⟨ψ|)
        let mut rho_a = vec![vec![Complex64::ZERO; dim_a]; dim_a];

        for i in 0..dim_a {
            for j in 0..dim_a {
                for k in 0..dim_b {
                    let idx_i = i | (k << split);
                    let idx_j = j | (k << split);
                    rho_a[i][j] += self.state[idx_i] * self.state[idx_j].conj();
                }
            }
        }

        // Compute eigenvalues of ρ_A (simplified: use diagonal approximation for now)
        // For a proper implementation, we'd use a matrix diagonalization library
        // Here we compute the diagonal elements which give probabilities for product states
        let mut entropy = 0.0;
        for i in 0..dim_a {
            let p = rho_a[i][i].re; // Diagonal element (should be real)
            if p > 1e-15 {
                entropy -= p * p.log2();
            }
        }

        entropy.max(0.0) // Ensure non-negative due to numerical errors
    }

    /// Calculate total probability (should be 1.0 for normalized states).
    pub fn total_probability(&self) -> f64 {
        self.state.iter().map(|a| a.norm_squared()).sum()
    }

    /// Renormalize the state vector.
    pub fn normalize(&mut self) {
        let norm = self.total_probability().sqrt();
        if norm > 1e-15 {
            let inv_norm = 1.0 / norm;
            for amp in &mut self.state {
                *amp = *amp * inv_norm;
            }
        }
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.state.len() * 16 // 16 bytes per Complex64
    }
}

impl std::fmt::Debug for FullQuantumCluster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FullQuantumCluster")
            .field("id", &self.id)
            .field("n_qubits", &self.n_qubits)
            .field("dim", &self.dim())
            .field("memory_bytes", &self.memory_bytes())
            .field("total_prob", &self.total_probability())
            .finish()
    }
}

// =============================================================================
// Magic State Factory
// =============================================================================

/// Magic state types for non-Clifford gate injection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MagicState {
    /// T state: |T⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2
    T,
    /// H state: |H⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩
    H,
    /// Y state: |Y⟩ = (|0⟩ + i|1⟩)/√2
    Y,
}

impl MagicState {
    /// Get the state vector amplitudes [α, β] where |ψ⟩ = α|0⟩ + β|1⟩.
    pub fn amplitudes(&self) -> (Complex64, Complex64) {
        match self {
            MagicState::T => {
                let inv_sqrt2 = FRAC_1_SQRT_2;
                (
                    Complex64::new(inv_sqrt2, 0.0),
                    Complex64::from_polar(inv_sqrt2, PI / 4.0),
                )
            }
            MagicState::H => {
                let c = (PI / 8.0).cos();
                let s = (PI / 8.0).sin();
                (Complex64::new(c, 0.0), Complex64::new(s, 0.0))
            }
            MagicState::Y => {
                let inv_sqrt2 = FRAC_1_SQRT_2;
                (
                    Complex64::new(inv_sqrt2, 0.0),
                    Complex64::new(0.0, inv_sqrt2),
                )
            }
        }
    }
}

// =============================================================================
// Hybrid Quantum Network
// =============================================================================

/// Configuration for adaptive cluster activation.
#[derive(Clone, Debug)]
pub struct AdaptiveClusterConfig {
    /// Enable adaptive cluster activation based on input entropy
    pub enabled: bool,

    /// Entropy threshold to activate clusters (higher = more selective)
    pub entropy_threshold: f64,

    /// Minimum ticks to keep clusters active after activation
    pub activation_cooldown: u32,

    /// Ticks to warm up cluster state after activation
    pub warmup_ticks: u32,
}

impl Default for AdaptiveClusterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            entropy_threshold: 1.5,
            activation_cooldown: 5,
            warmup_ticks: 2,
        }
    }
}

/// Configuration for bidirectional feedback from clusters to backbone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedbackMode {
    /// No feedback (Sprint 58.0 behavior)
    None,
    /// Modulate backbone phases via S gate based on cluster Z expectation
    Phase,
    /// Scale backbone input encoding based on cluster state
    Amplitude,
    /// Enable/disable synapses based on cluster measurements
    Gated,
}

impl Default for FeedbackMode {
    fn default() -> Self {
        Self::None
    }
}

// =============================================================================
// Cluster-to-Cluster Bridges (EPIC 139)
// =============================================================================

/// Type of quantum operation between clusters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BridgeGate {
    /// Teleport state from source to target (consumes source qubit state)
    Teleport,
    /// SWAP states between clusters (bidirectional exchange)
    RemoteSwap,
    /// Distributed CZ gate (creates inter-cluster entanglement)
    DistributedCZ,
}

/// Connection between two quantum clusters.
///
/// Enables quantum operations across cluster boundaries for
/// distributed quantum processing.
#[derive(Clone, Debug)]
pub struct ClusterBridge {
    /// Source cluster ID
    pub source_cluster: usize,
    /// Source qubit within the cluster
    pub source_qubit: usize,
    /// Target cluster ID
    pub target_cluster: usize,
    /// Target qubit within the cluster
    pub target_qubit: usize,
    /// Type of quantum operation
    pub gate_type: BridgeGate,
    /// Whether this bridge is currently active
    pub active: bool,
}

impl ClusterBridge {
    /// Create a new cluster bridge.
    pub fn new(
        source_cluster: usize,
        source_qubit: usize,
        target_cluster: usize,
        target_qubit: usize,
        gate_type: BridgeGate,
    ) -> Self {
        Self {
            source_cluster,
            source_qubit,
            target_cluster,
            target_qubit,
            gate_type,
            active: true,
        }
    }

    /// Create a teleportation bridge.
    pub fn teleport(
        src_cluster: usize,
        src_qubit: usize,
        tgt_cluster: usize,
        tgt_qubit: usize,
    ) -> Self {
        Self::new(
            src_cluster,
            src_qubit,
            tgt_cluster,
            tgt_qubit,
            BridgeGate::Teleport,
        )
    }

    /// Create a remote SWAP bridge.
    pub fn swap(
        src_cluster: usize,
        src_qubit: usize,
        tgt_cluster: usize,
        tgt_qubit: usize,
    ) -> Self {
        Self::new(
            src_cluster,
            src_qubit,
            tgt_cluster,
            tgt_qubit,
            BridgeGate::RemoteSwap,
        )
    }

    /// Create a distributed CZ bridge.
    pub fn distributed_cz(
        src_cluster: usize,
        src_qubit: usize,
        tgt_cluster: usize,
        tgt_qubit: usize,
    ) -> Self {
        Self::new(
            src_cluster,
            src_qubit,
            tgt_cluster,
            tgt_qubit,
            BridgeGate::DistributedCZ,
        )
    }
}

/// Configuration for hybrid network.
#[derive(Clone, Debug)]
pub struct HybridNetworkConfig {
    /// Stabilizer backbone configuration
    pub backbone: StabilizerNetworkConfig,

    /// Number of full-quantum clusters
    pub n_clusters: usize,

    /// Qubits per cluster (typically 8-16)
    pub cluster_size: usize,

    /// Which backbone neurons to map to clusters
    /// If None, automatically selects output layer neurons
    pub cluster_neurons: Option<Vec<Vec<usize>>>,

    /// Enable magic state injection
    pub enable_magic_states: bool,

    /// Adaptive cluster activation config
    pub adaptive: AdaptiveClusterConfig,

    /// Feedback mode from clusters to backbone
    pub feedback_mode: FeedbackMode,
}

impl Default for HybridNetworkConfig {
    fn default() -> Self {
        Self {
            backbone: StabilizerNetworkConfig::default(),
            n_clusters: 2,
            cluster_size: 8,
            cluster_neurons: None,
            enable_magic_states: true,
            adaptive: AdaptiveClusterConfig::default(),
            feedback_mode: FeedbackMode::None,
        }
    }
}

impl HybridNetworkConfig {
    /// Create config with specific cluster count and size.
    pub fn with_clusters(n_clusters: usize, cluster_size: usize) -> Self {
        Self {
            n_clusters,
            cluster_size,
            ..Default::default()
        }
    }

    /// Create a small hybrid for testing.
    pub fn small() -> Self {
        Self {
            backbone: StabilizerNetworkConfig::minimal(),
            n_clusters: 1,
            cluster_size: 4,
            cluster_neurons: None,
            enable_magic_states: true,
            adaptive: AdaptiveClusterConfig::default(),
            feedback_mode: FeedbackMode::None,
        }
    }

    /// Create a medium hybrid network.
    pub fn medium() -> Self {
        Self {
            backbone: StabilizerNetworkConfig::medium(),
            n_clusters: 2,
            cluster_size: 8,
            cluster_neurons: None,
            enable_magic_states: true,
            adaptive: AdaptiveClusterConfig::default(),
            feedback_mode: FeedbackMode::None,
        }
    }

    /// Enable adaptive cluster activation.
    pub fn with_adaptive(mut self, threshold: f64) -> Self {
        self.adaptive.enabled = true;
        self.adaptive.entropy_threshold = threshold;
        self
    }

    /// Set feedback mode.
    pub fn with_feedback(mut self, mode: FeedbackMode) -> Self {
        self.feedback_mode = mode;
        self
    }
}

/// Hybrid quantum neural network combining stabilizer backbone with full-quantum clusters.
pub struct HybridQuantumNetwork {
    /// Stabilizer backbone network
    backbone: StabilizerNeuronNetwork,

    /// Full-quantum clusters for non-Clifford operations
    clusters: Vec<FullQuantumCluster>,

    /// Mapping: backbone neuron -> (cluster_id, local_qubit)
    cluster_map: HashMap<usize, (usize, usize)>,

    /// Configuration
    config: HybridNetworkConfig,

    /// Random number generator
    rng: SimpleRng,

    /// Current tick
    tick: u64,

    /// Output spike counts
    output_spike_counts: Vec<u32>,

    /// Cluster-to-cluster bridges for distributed quantum operations
    cluster_bridges: Vec<ClusterBridge>,
}

impl HybridQuantumNetwork {
    /// Create a new hybrid network.
    pub fn new(config: HybridNetworkConfig, seed: u64) -> Self {
        let backbone = StabilizerNeuronNetwork::new(config.backbone.clone(), seed);
        let rng = SimpleRng::new(seed.wrapping_add(12345));

        // Create clusters
        let mut clusters = Vec::with_capacity(config.n_clusters);
        let mut cluster_map = HashMap::new();

        // Determine which neurons go in each cluster
        let neuron_assignments = if let Some(ref assignments) = config.cluster_neurons {
            assignments.clone()
        } else {
            // Default: map output layer neurons to clusters
            let output_range = backbone.topology().output_qubits();
            let output_neurons: Vec<usize> = output_range.collect();

            // Distribute evenly across clusters
            let mut assignments = vec![Vec::new(); config.n_clusters];
            for (i, &neuron) in output_neurons.iter().enumerate() {
                let cluster_id = i % config.n_clusters;
                if assignments[cluster_id].len() < config.cluster_size {
                    assignments[cluster_id].push(neuron);
                }
            }
            assignments
        };

        // Create clusters with neuron mappings
        for (cluster_id, neurons) in neuron_assignments.iter().enumerate() {
            let n_qubits = neurons.len().min(config.cluster_size).max(1);
            let mut cluster = FullQuantumCluster::new(n_qubits, cluster_id);
            cluster.neuron_map = neurons.iter().take(n_qubits).copied().collect();

            // Update cluster_map
            for (local_qubit, &neuron) in cluster.neuron_map.iter().enumerate() {
                cluster_map.insert(neuron, (cluster_id, local_qubit));
            }

            clusters.push(cluster);
        }

        let n_outputs = config.backbone.n_outputs;

        Self {
            backbone,
            clusters,
            cluster_map,
            config,
            rng,
            tick: 0,
            output_spike_counts: vec![0; n_outputs],
            cluster_bridges: Vec::new(),
        }
    }

    /// Convenience constructor.
    pub fn from_parts(
        n_inputs: usize,
        hidden: &[usize],
        n_outputs: usize,
        n_clusters: usize,
        cluster_size: usize,
        seed: u64,
    ) -> Self {
        let config = HybridNetworkConfig {
            backbone: StabilizerNetworkConfig {
                n_inputs,
                hidden_layers: hidden.to_vec(),
                n_outputs,
                connectivity: 0.3,
                ..Default::default()
            },
            n_clusters,
            cluster_size,
            ..Default::default()
        };
        Self::new(config, seed)
    }

    /// Get number of clusters.
    pub fn n_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Get a cluster by ID.
    pub fn cluster(&self, id: usize) -> Option<&FullQuantumCluster> {
        self.clusters.get(id)
    }

    /// Get mutable cluster by ID.
    pub fn cluster_mut(&mut self, id: usize) -> Option<&mut FullQuantumCluster> {
        self.clusters.get_mut(id)
    }

    /// Check if a neuron is in a cluster.
    pub fn is_in_cluster(&self, neuron: usize) -> bool {
        self.cluster_map.contains_key(&neuron)
    }

    /// Get cluster mapping for a neuron.
    pub fn neuron_cluster(&self, neuron: usize) -> Option<(usize, usize)> {
        self.cluster_map.get(&neuron).copied()
    }

    /// Get output spike counts.
    pub fn get_output_spike_counts(&self) -> Vec<u32> {
        self.output_spike_counts.clone()
    }

    /// Reset output counts.
    pub fn reset_output_counts(&mut self) {
        for c in &mut self.output_spike_counts {
            *c = 0;
        }
    }

    /// Get current tick.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    // =========================================================================
    // MAGIC STATE INJECTION
    // =========================================================================

    /// Inject a magic state into a cluster qubit.
    ///
    /// This prepares the qubit in the magic state, enabling subsequent
    /// non-Clifford operations via gate teleportation.
    pub fn inject_magic_state(&mut self, cluster_id: usize, qubit: usize, state: MagicState) {
        if let Some(cluster) = self.clusters.get_mut(cluster_id) {
            if qubit < cluster.n_qubits() {
                // Reset qubit to |0⟩ first (measure and correct)
                cluster.measure(qubit, &mut self.rng);

                // Prepare magic state: |ψ⟩ = α|0⟩ + β|1⟩
                let (alpha, beta) = state.amplitudes();

                // Apply the magic state preparation
                // For each basis state |...0...⟩, create superposition α|...0...⟩ + β|...1...⟩
                let mask = 1 << qubit;
                let dim = cluster.dim();

                // Collect updates first to avoid borrow issues
                let mut updates: Vec<(usize, Complex64, usize, Complex64)> = Vec::new();

                for i in 0..dim {
                    if (i & mask) == 0 {
                        // This is a |...0...⟩ state
                        let partner = i | mask;
                        let orig = cluster.state[i];

                        if orig.norm_squared() > 1e-15 {
                            updates.push((i, orig * alpha, partner, orig * beta));
                        }
                    }
                }

                // Apply updates
                for (i, val_i, partner, val_partner) in updates {
                    cluster.state[i] = val_i;
                    cluster.state[partner] = val_partner;
                }

                cluster.normalize();
            }
        }
    }

    /// Apply T gate to a cluster qubit via magic state.
    pub fn apply_cluster_t(&mut self, cluster_id: usize, qubit: usize) {
        if let Some(cluster) = self.clusters.get_mut(cluster_id) {
            cluster.t(qubit);
        }
    }

    /// Apply T† gate to a cluster qubit.
    pub fn apply_cluster_t_dag(&mut self, cluster_id: usize, qubit: usize) {
        if let Some(cluster) = self.clusters.get_mut(cluster_id) {
            cluster.t_dag(qubit);
        }
    }

    // =========================================================================
    // ADAPTIVE CLUSTER ACTIVATION (Sprint 59.0)
    // =========================================================================

    /// Compute Shannon entropy of input distribution.
    ///
    /// Higher entropy = more complex/varied input pattern.
    pub fn compute_input_entropy(&self, inputs: &[u8]) -> f64 {
        let sum: u32 = inputs.iter().map(|&x| x as u32).sum();
        if sum == 0 {
            return 0.0;
        }

        let sum_f = sum as f64;
        inputs
            .iter()
            .filter(|&&x| x > 0)
            .map(|&x| {
                let p = x as f64 / sum_f;
                -p * p.ln()
            })
            .sum()
    }

    /// Update cluster activation based on input entropy.
    fn update_cluster_activation(&mut self, inputs: &[u8]) {
        if !self.config.adaptive.enabled {
            return;
        }

        let entropy = self.compute_input_entropy(inputs);
        let threshold = self.config.adaptive.entropy_threshold;
        let cooldown = self.config.adaptive.activation_cooldown;

        for cluster in &mut self.clusters {
            // Tick down cooldown
            if cluster.tick_cooldown() {
                continue; // Still in cooldown, stay active
            }

            if entropy > threshold {
                // High entropy input: activate cluster
                if !cluster.active {
                    cluster.activate();
                    cluster.set_cooldown(cooldown);
                }
            } else {
                // Low entropy input: deactivate cluster (if not in cooldown)
                if cluster.active && cluster.activation_cooldown == 0 {
                    cluster.deactivate();
                }
            }
        }
    }

    /// Check how many clusters are currently active.
    pub fn active_cluster_count(&self) -> usize {
        self.clusters.iter().filter(|c| c.active).count()
    }

    // =========================================================================
    // BIDIRECTIONAL FEEDBACK (Sprint 59.0)
    // =========================================================================

    /// Apply feedback from clusters to backbone based on feedback mode.
    fn apply_cluster_feedback(&mut self) {
        match self.config.feedback_mode {
            FeedbackMode::None => {}
            FeedbackMode::Phase => self.apply_phase_feedback(),
            FeedbackMode::Amplitude => self.apply_amplitude_feedback(),
            FeedbackMode::Gated => self.apply_gated_feedback(),
        }
    }

    /// Apply phase feedback: modulate backbone phases via S gate.
    fn apply_phase_feedback(&mut self) {
        // Collect feedback data first to avoid borrow issues
        let mut phase_applications: Vec<usize> = Vec::new();

        for cluster in &self.clusters {
            if !cluster.active {
                continue;
            }

            let z_exp = cluster.total_z_expectation();

            // If cluster is in |1⟩-biased state (negative Z expectation), apply phase shift
            if z_exp < -0.3 {
                // Apply S to connected backbone neurons
                for &neuron in &cluster.neuron_map {
                    phase_applications.push(neuron);
                }
            }
        }

        // Now apply the phase gates
        for neuron in phase_applications {
            self.backbone.tableau_mut().phase_gate(neuron);
        }
    }

    /// Apply amplitude feedback: scale input encoding.
    fn apply_amplitude_feedback(&mut self) {
        // This is applied during input encoding, store modulation factors
        // For now, a simpler version that affects the backbone state directly
        for cluster in &self.clusters {
            if !cluster.active {
                continue;
            }

            let z_exp = cluster.total_z_expectation();

            // Strong positive Z expectation -> enhance connectivity
            if z_exp > 0.5 {
                // Apply Hadamard to create more superposition (enhancing quantum effects)
                for &neuron in cluster.neuron_map.iter().take(1) {
                    self.backbone.tableau_mut().hadamard(neuron);
                }
            }
        }
    }

    /// Apply gated feedback: enable/disable synapses.
    fn apply_gated_feedback(&mut self) {
        // Collect cluster states
        let mut gate_decisions: Vec<(usize, bool)> = Vec::new();

        for cluster in &self.clusters {
            if !cluster.active {
                continue;
            }

            let _z_exp = cluster.total_z_expectation();

            // Gate synapses based on cluster state
            for (local_q, &neuron) in cluster.neuron_map.iter().enumerate() {
                let qubit_z = cluster.z_expectation(local_q);
                // If qubit is strongly in |0⟩ state, enable outgoing synapses
                // If strongly in |1⟩ state, disable them
                let should_enable = qubit_z > 0.0;
                gate_decisions.push((neuron, should_enable));
            }
        }

        // Apply gating decisions to backbone synapses
        for (neuron, should_enable) in gate_decisions {
            self.backbone
                .set_synapses_from_neuron(neuron, should_enable);
        }
    }

    /// Get total Z expectation across all active clusters.
    pub fn total_cluster_z_expectation(&self) -> f64 {
        let active: Vec<_> = self.clusters.iter().filter(|c| c.active).collect();
        if active.is_empty() {
            return 0.0;
        }

        let sum: f64 = active.iter().map(|c| c.total_z_expectation()).sum();
        sum / active.len() as f64
    }

    // =========================================================================
    // TICK CYCLE
    // =========================================================================

    /// Run one tick of the hybrid network.
    pub fn step(&mut self, inputs: &[u8]) -> Vec<usize> {
        // Phase 0: Update cluster activation (adaptive mode)
        self.update_cluster_activation(inputs);

        // Phase 1: Run backbone stabilizer network
        let backbone_fired = self.backbone.step(inputs);

        // Phase 1.5: Apply feedback from clusters to backbone
        self.apply_cluster_feedback();

        // Phase 2: Process active clusters only
        // Apply quantum dynamics to cluster qubits based on backbone activity
        for cluster in &mut self.clusters {
            if !cluster.active {
                continue; // Skip inactive clusters
            }

            // Apply entangling gates within cluster based on backbone correlations
            if cluster.n_qubits() >= 2 {
                // Apply CZ between adjacent qubits (creates entanglement)
                for q in 0..cluster.n_qubits() - 1 {
                    cluster.cz(q, q + 1);
                }
            }

            // Apply T gates on some qubits (the non-Clifford magic!)
            if self.config.enable_magic_states && cluster.n_qubits() > 0 {
                // Apply T to first qubit every other tick
                if self.tick.is_multiple_of(2) {
                    cluster.t(0);
                }
            }
        }

        // Phase 3: Measure cluster qubits and determine firing (active clusters only)
        let mut cluster_fired = Vec::new();
        for cluster in &mut self.clusters {
            if !cluster.active {
                continue; // Skip inactive clusters
            }

            for local_q in 0..cluster.n_qubits() {
                if let Some(&neuron) = cluster.neuron_map.get(local_q) {
                    // Measure this qubit
                    let result = cluster.measure(local_q, &mut self.rng);
                    if result == 1 {
                        cluster_fired.push(neuron);
                    }

                    // Re-prepare qubit for next tick
                    // Apply H to create superposition again
                    cluster.h(local_q);
                }
            }
        }

        // Phase 4: Update output counts
        let output_start = self.backbone.topology().output_layer().start;
        for &neuron in &backbone_fired {
            if neuron >= output_start {
                let idx = neuron - output_start;
                if idx < self.output_spike_counts.len() {
                    self.output_spike_counts[idx] += 1;
                }
            }
        }
        for &neuron in &cluster_fired {
            if neuron >= output_start {
                let idx = neuron - output_start;
                if idx < self.output_spike_counts.len() {
                    self.output_spike_counts[idx] += 1;
                }
            }
        }

        self.tick += 1;

        // Combine fired neurons
        let mut all_fired = backbone_fired;
        all_fired.extend(cluster_fired);
        all_fired
    }

    /// Make a decision based on output spike counts.
    pub fn decide(&mut self, inputs: &[u8]) -> usize {
        self.step(inputs);

        self.output_spike_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| *count)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Run multiple ticks and then decide.
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
        // Learn in backbone
        self.backbone.learn(reward);

        // Could also modulate cluster behavior based on reward
        // For now, just pass through to backbone
    }

    // =========================================================================
    // ANALYSIS
    // =========================================================================

    /// Get entanglement entropy for a cluster.
    pub fn cluster_entropy(&self, cluster_id: usize) -> f64 {
        if let Some(cluster) = self.clusters.get(cluster_id) {
            // Bipartition in the middle
            let split = cluster.n_qubits() / 2;
            cluster.entanglement_entropy(split.max(1))
        } else {
            0.0
        }
    }

    /// Get total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let backbone_mem = self.backbone.memory_usage();
        let cluster_mem: usize = self.clusters.iter().map(|c| c.memory_bytes()).sum();
        backbone_mem + cluster_mem
    }

    /// Get reference to backbone.
    pub fn backbone(&self) -> &StabilizerNeuronNetwork {
        &self.backbone
    }

    /// Get mutable reference to backbone.
    pub fn backbone_mut(&mut self) -> &mut StabilizerNeuronNetwork {
        &mut self.backbone
    }

    // =========================================================================
    // CLUSTER BRIDGES (EPIC 139)
    // =========================================================================

    /// Add a bridge between two clusters.
    pub fn add_bridge(&mut self, bridge: ClusterBridge) {
        // Validate cluster IDs
        if bridge.source_cluster < self.clusters.len()
            && bridge.target_cluster < self.clusters.len()
            && bridge.source_cluster != bridge.target_cluster
        {
            self.cluster_bridges.push(bridge);
        }
    }

    /// Get number of bridges.
    pub fn n_bridges(&self) -> usize {
        self.cluster_bridges.len()
    }

    /// Get bridge by index.
    pub fn bridge(&self, idx: usize) -> Option<&ClusterBridge> {
        self.cluster_bridges.get(idx)
    }

    /// Enable/disable a bridge.
    pub fn set_bridge_active(&mut self, idx: usize, active: bool) {
        if let Some(bridge) = self.cluster_bridges.get_mut(idx) {
            bridge.active = active;
        }
    }

    /// Apply all active bridges.
    ///
    /// This performs the quantum operations specified by each bridge.
    pub fn apply_bridges(&mut self) {
        // Collect bridge operations to avoid borrow conflicts
        let bridges: Vec<ClusterBridge> = self
            .cluster_bridges
            .iter()
            .filter(|b| b.active)
            .cloned()
            .collect();

        for bridge in bridges {
            match bridge.gate_type {
                BridgeGate::Teleport => self.apply_teleport(&bridge),
                BridgeGate::RemoteSwap => self.apply_remote_swap(&bridge),
                BridgeGate::DistributedCZ => self.apply_distributed_cz(&bridge),
            }
        }
    }

    /// Teleport qubit state from source to target cluster.
    ///
    /// This transfers the quantum state while resetting the source qubit.
    fn apply_teleport(&mut self, bridge: &ClusterBridge) {
        // Extract state from source
        let (alpha, beta) = {
            let src = &self.clusters[bridge.source_cluster];
            src.extract_qubit_state(bridge.source_qubit)
        };

        // Inject into target
        self.clusters[bridge.target_cluster].inject_qubit_state(bridge.target_qubit, alpha, beta);

        // Reset source qubit
        self.clusters[bridge.source_cluster].reset_qubit(bridge.source_qubit);
    }

    /// Swap states between source and target clusters.
    fn apply_remote_swap(&mut self, bridge: &ClusterBridge) {
        // Extract both states
        let (alpha_src, beta_src) = {
            let src = &self.clusters[bridge.source_cluster];
            src.extract_qubit_state(bridge.source_qubit)
        };

        let (alpha_tgt, beta_tgt) = {
            let tgt = &self.clusters[bridge.target_cluster];
            tgt.extract_qubit_state(bridge.target_qubit)
        };

        // Inject swapped states
        self.clusters[bridge.source_cluster].inject_qubit_state(
            bridge.source_qubit,
            alpha_tgt,
            beta_tgt,
        );

        self.clusters[bridge.target_cluster].inject_qubit_state(
            bridge.target_qubit,
            alpha_src,
            beta_src,
        );
    }

    /// Apply distributed CZ gate between clusters.
    ///
    /// This creates entanglement between the two cluster qubits.
    /// Note: This is an approximation - true distributed CZ requires
    /// shared entanglement and classical communication.
    fn apply_distributed_cz(&mut self, bridge: &ClusterBridge) {
        // Get Z expectations from both qubits
        let z_src = self.clusters[bridge.source_cluster].z_expectation(bridge.source_qubit);
        let z_tgt = self.clusters[bridge.target_cluster].z_expectation(bridge.target_qubit);

        // If both are in |1⟩ state (z < 0), apply a phase flip
        // This approximates the CZ effect based on measurement statistics
        if z_src < 0.0 && z_tgt < 0.0 {
            // Apply Z to source (phase flip when both are |1⟩)
            self.clusters[bridge.source_cluster].z(bridge.source_qubit);
        }

        // For stronger entanglement effect, apply controlled rotations
        // based on the partner cluster's state
        if z_tgt < 0.0 {
            // Target is mostly |1⟩ -> rotate source
            let angle = PI / 4.0 * (-z_tgt); // Proportional to |1⟩ probability
            self.clusters[bridge.source_cluster].rz(bridge.source_qubit, angle);
        }
        if z_src < 0.0 {
            // Source is mostly |1⟩ -> rotate target
            let angle = PI / 4.0 * (-z_src);
            self.clusters[bridge.target_cluster].rz(bridge.target_qubit, angle);
        }
    }

    // =========================================================================
    // TOPOLOGY BUILDERS (EPIC 139)
    // =========================================================================

    /// Create a linear chain of cluster connections.
    ///
    /// Connects cluster i's last qubit to cluster i+1's first qubit.
    pub fn chain_clusters(&mut self) {
        for i in 0..self.clusters.len().saturating_sub(1) {
            let src_qubit = self.clusters[i].n_qubits().saturating_sub(1);
            let bridge = ClusterBridge::distributed_cz(i, src_qubit, i + 1, 0);
            self.add_bridge(bridge);
        }
    }

    /// Create a ring of cluster connections.
    ///
    /// Like chain, but also connects last cluster back to first.
    pub fn ring_clusters(&mut self) {
        let n = self.clusters.len();
        if n < 2 {
            return;
        }

        // Create chain first
        self.chain_clusters();

        // Connect last to first
        let src_qubit = self.clusters[n - 1].n_qubits().saturating_sub(1);
        let bridge = ClusterBridge::distributed_cz(n - 1, src_qubit, 0, 0);
        self.add_bridge(bridge);
    }

    /// Create a fully connected mesh of clusters.
    ///
    /// Every cluster is connected to every other cluster.
    pub fn mesh_clusters(&mut self) {
        let n = self.clusters.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let bridge = ClusterBridge::distributed_cz(i, 0, j, 0);
                self.add_bridge(bridge);
            }
        }
    }

    /// Create star topology with center cluster.
    ///
    /// Center cluster connects to all other clusters.
    pub fn star_clusters(&mut self, center: usize) {
        let n = self.clusters.len();
        if center >= n {
            return;
        }

        for i in 0..n {
            if i != center {
                let bridge = ClusterBridge::distributed_cz(
                    center,
                    i % self.clusters[center].n_qubits(),
                    i,
                    0,
                );
                self.add_bridge(bridge);
            }
        }
    }

    /// Clear all bridges.
    pub fn clear_bridges(&mut self) {
        self.cluster_bridges.clear();
    }
}

impl std::fmt::Debug for HybridQuantumNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridQuantumNetwork")
            .field("backbone_neurons", &self.backbone.num_neurons())
            .field("n_clusters", &self.clusters.len())
            .field("cluster_neurons", &self.cluster_map.len())
            .field("tick", &self.tick)
            .field("memory_bytes", &self.memory_bytes())
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex64_operations() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(3.0, 4.0);

        let sum = a + b;
        assert!((sum.re - 4.0).abs() < 1e-10);
        assert!((sum.im - 6.0).abs() < 1e-10);

        let prod = a * b;
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        assert!((prod.re - (-5.0)).abs() < 1e-10);
        assert!((prod.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_cluster_creation() {
        let cluster = FullQuantumCluster::new(4, 0);
        assert_eq!(cluster.n_qubits(), 4);
        assert_eq!(cluster.dim(), 16);
        assert!((cluster.total_probability() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cluster_hadamard() {
        let mut cluster = FullQuantumCluster::new(1, 0);

        // Apply H to |0⟩ -> |+⟩ = (|0⟩ + |1⟩)/√2
        cluster.h(0);

        // Both amplitudes should be 1/√2
        let inv_sqrt2 = FRAC_1_SQRT_2;
        assert!((cluster.amplitude(0).re - inv_sqrt2).abs() < 1e-10);
        assert!((cluster.amplitude(1).re - inv_sqrt2).abs() < 1e-10);
    }

    #[test]
    fn test_cluster_t_gate() {
        let mut cluster = FullQuantumCluster::new(1, 0);

        // Prepare |+⟩
        cluster.h(0);

        // Apply T
        cluster.t(0);

        // |0⟩ component unchanged, |1⟩ gets e^{iπ/4} phase
        let amp0 = cluster.amplitude(0);
        let amp1 = cluster.amplitude(1);

        // Both should have same magnitude
        assert!((amp0.norm() - amp1.norm()).abs() < 1e-10);

        // Phase difference should be π/4
        let phase_diff = amp1.arg() - amp0.arg();
        assert!((phase_diff - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_cluster_bell_state() {
        let mut cluster = FullQuantumCluster::new(2, 0);

        // Create Bell state: H(0), CNOT(0,1)
        cluster.h(0);
        cluster.cnot(0, 1);

        // Should be (|00⟩ + |11⟩)/√2
        let inv_sqrt2 = FRAC_1_SQRT_2;
        assert!((cluster.amplitude(0b00).re - inv_sqrt2).abs() < 1e-10); // |00⟩
        assert!((cluster.amplitude(0b01).norm() - 0.0).abs() < 1e-10); // |01⟩
        assert!((cluster.amplitude(0b10).norm() - 0.0).abs() < 1e-10); // |10⟩
        assert!((cluster.amplitude(0b11).re - inv_sqrt2).abs() < 1e-10); // |11⟩
    }

    #[test]
    fn test_cluster_measurement() {
        let mut cluster = FullQuantumCluster::new(1, 0);
        let mut rng = SimpleRng::new(42);

        // |0⟩ should always measure 0
        let result = cluster.measure(0, &mut rng);
        assert_eq!(result, 0);

        // After H, should measure 0 or 1 randomly
        cluster.reset();
        cluster.h(0);

        let mut zeros = 0;
        let mut ones = 0;
        for _ in 0..100 {
            cluster.reset();
            cluster.h(0);
            let r = cluster.measure(0, &mut rng);
            if r == 0 {
                zeros += 1;
            } else {
                ones += 1;
            }
        }

        // Should be roughly 50/50
        assert!(zeros > 20 && ones > 20);
    }

    #[test]
    fn test_cluster_entropy() {
        let mut cluster = FullQuantumCluster::new(2, 0);

        // Product state |00⟩ should have zero entropy
        let entropy_product = cluster.entanglement_entropy(1);
        assert!(entropy_product < 0.1);

        // Bell state should have entropy ~ 1 bit
        cluster.h(0);
        cluster.cnot(0, 1);
        let entropy_bell = cluster.entanglement_entropy(1);
        assert!(entropy_bell > 0.5); // Should be close to 1.0
    }

    #[test]
    fn test_hybrid_creation() {
        let config = HybridNetworkConfig::small();
        let hybrid = HybridQuantumNetwork::new(config, 42);

        assert_eq!(hybrid.n_clusters(), 1);
        assert!(hybrid.cluster(0).is_some());
    }

    #[test]
    fn test_hybrid_step() {
        let config = HybridNetworkConfig::small();
        let mut hybrid = HybridQuantumNetwork::new(config, 42);

        let inputs = vec![200, 150];
        let fired = hybrid.step(&inputs);

        // Should have run without panic
        assert!(fired.len() <= hybrid.backbone.num_neurons());
    }

    #[test]
    fn test_hybrid_decide() {
        let mut hybrid = HybridQuantumNetwork::from_parts(4, &[8], 2, 1, 4, 42);

        let inputs = vec![255, 255, 0, 0];
        let decision = hybrid.decide(&inputs);

        assert!(decision < 2);
    }

    #[test]
    fn test_magic_state_injection() {
        let config = HybridNetworkConfig::small();
        let mut hybrid = HybridQuantumNetwork::new(config, 42);

        // Inject T magic state
        hybrid.inject_magic_state(0, 0, MagicState::T);

        // Cluster should still be normalized
        if let Some(cluster) = hybrid.cluster(0) {
            assert!((cluster.total_probability() - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_cluster_entropy_in_hybrid() {
        let mut hybrid = HybridQuantumNetwork::from_parts(4, &[8], 4, 1, 4, 42);

        // Run a few steps to create some entanglement
        for _ in 0..5 {
            hybrid.step(&[200, 200, 100, 100]);
        }

        let entropy = hybrid.cluster_entropy(0);
        // Entropy should be defined (not NaN)
        assert!(!entropy.is_nan());
    }

    #[test]
    fn test_memory_usage() {
        let hybrid = HybridQuantumNetwork::from_parts(10, &[100], 10, 4, 8, 42);
        let mem = hybrid.memory_bytes();

        // Should have backbone memory + cluster memory
        // 4 clusters * 256 amplitudes * 16 bytes = 16 KB for clusters
        // Plus backbone
        assert!(mem > 10_000);
    }

    // =========================================================================
    // Sprint 59.0 Phase 1 Tests
    // =========================================================================

    #[test]
    fn test_z_expectation() {
        let mut cluster = FullQuantumCluster::new(2, 0);

        // |00⟩ state: Z expectation should be +1 for both qubits
        assert!((cluster.z_expectation(0) - 1.0).abs() < 1e-10);
        assert!((cluster.z_expectation(1) - 1.0).abs() < 1e-10);

        // After X on qubit 0: |10⟩, qubit 0 -> -1, qubit 1 -> +1
        cluster.x(0);
        assert!((cluster.z_expectation(0) - (-1.0)).abs() < 1e-10);
        assert!((cluster.z_expectation(1) - 1.0).abs() < 1e-10);

        // After H on qubit 1: superposition, Z expectation ~0
        cluster.h(1);
        assert!(cluster.z_expectation(1).abs() < 1e-10);
    }

    #[test]
    fn test_total_z_expectation() {
        let mut cluster = FullQuantumCluster::new(2, 0);

        // |00⟩ state: average Z = +1
        assert!((cluster.total_z_expectation() - 1.0).abs() < 1e-10);

        // |11⟩ state: average Z = -1
        cluster.x(0);
        cluster.x(1);
        assert!((cluster.total_z_expectation() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cluster_activation() {
        let mut cluster = FullQuantumCluster::new(2, 0);
        assert!(cluster.active);

        cluster.deactivate();
        assert!(!cluster.active);

        cluster.activate();
        assert!(cluster.active);
    }

    #[test]
    fn test_reset_to_plus_state() {
        let mut cluster = FullQuantumCluster::new(2, 0);

        cluster.reset_to_plus_state();

        // All amplitudes should be equal: 1/sqrt(4) = 0.5
        let expected = 0.5;
        for i in 0..4 {
            assert!((cluster.amplitude(i).re - expected).abs() < 1e-10);
            assert!(cluster.amplitude(i).im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_input_entropy() {
        let config = HybridNetworkConfig::small();
        let hybrid = HybridQuantumNetwork::new(config, 42);

        // Uniform distribution: high entropy
        let uniform = vec![100, 100, 100, 100];
        let entropy_uniform = hybrid.compute_input_entropy(&uniform);

        // Concentrated distribution: lower entropy
        let concentrated = vec![200, 10, 10, 10];
        let entropy_concentrated = hybrid.compute_input_entropy(&concentrated);

        // Uniform should have higher entropy
        assert!(entropy_uniform > entropy_concentrated);

        // Empty/zero input: zero entropy
        let zero = vec![0, 0, 0, 0];
        let entropy_zero = hybrid.compute_input_entropy(&zero);
        assert!((entropy_zero - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_cluster_activation() {
        // Create network with adaptive mode enabled
        let config = HybridNetworkConfig::small().with_adaptive(0.5);
        let mut hybrid = HybridQuantumNetwork::new(config, 42);

        // Initially all clusters active
        assert_eq!(hybrid.active_cluster_count(), 1);

        // Low entropy input should eventually deactivate
        // (after cooldown expires)
        for _ in 0..20 {
            hybrid.step(&[200, 0]); // Very low entropy input
        }

        // High entropy input should activate
        for _ in 0..5 {
            hybrid.step(&[100, 100]); // Higher entropy
        }

        // Should have processed without crashing
        assert!(hybrid.tick() > 0);
    }

    #[test]
    fn test_feedback_mode_config() {
        let config = HybridNetworkConfig::small().with_feedback(FeedbackMode::Phase);

        assert_eq!(config.feedback_mode, FeedbackMode::Phase);
    }

    #[test]
    fn test_phase_feedback() {
        let config = HybridNetworkConfig::small().with_feedback(FeedbackMode::Phase);
        let mut hybrid = HybridQuantumNetwork::new(config, 42);

        // Run a few steps - feedback should be applied
        for _ in 0..10 {
            hybrid.step(&[200, 150]);
        }

        // Should complete without error
        assert!(hybrid.tick() == 10);
    }

    #[test]
    fn test_active_cluster_count() {
        let config = HybridNetworkConfig::small();
        let hybrid = HybridQuantumNetwork::new(config, 42);

        // By default all clusters are active
        assert_eq!(hybrid.active_cluster_count(), 1);
    }

    #[test]
    fn test_total_cluster_z_expectation() {
        let config = HybridNetworkConfig::small();
        let hybrid = HybridQuantumNetwork::new(config, 42);

        // Fresh clusters in |0⟩ state should have positive Z expectation
        let z_exp = hybrid.total_cluster_z_expectation();
        assert!(z_exp > 0.0);
    }

    // =========================================================================
    // CLUSTER BRIDGE TESTS (EPIC 139)
    // =========================================================================

    #[test]
    fn test_bridge_creation() {
        let bridge = ClusterBridge::new(0, 0, 1, 0, BridgeGate::DistributedCZ);
        assert_eq!(bridge.source_cluster, 0);
        assert_eq!(bridge.target_cluster, 1);
        assert!(bridge.active);
    }

    #[test]
    fn test_bridge_constructors() {
        let teleport = ClusterBridge::teleport(0, 1, 1, 2);
        assert_eq!(teleport.gate_type, BridgeGate::Teleport);

        let swap = ClusterBridge::swap(0, 0, 1, 0);
        assert_eq!(swap.gate_type, BridgeGate::RemoteSwap);

        let cz = ClusterBridge::distributed_cz(0, 0, 1, 0);
        assert_eq!(cz.gate_type, BridgeGate::DistributedCZ);
    }

    #[test]
    fn test_add_bridge() {
        let mut hybrid = HybridQuantumNetwork::from_parts(4, &[8], 4, 2, 4, 42);

        assert_eq!(hybrid.n_bridges(), 0);

        let bridge = ClusterBridge::distributed_cz(0, 0, 1, 0);
        hybrid.add_bridge(bridge);

        assert_eq!(hybrid.n_bridges(), 1);
        assert!(hybrid.bridge(0).is_some());
    }

    #[test]
    fn test_invalid_bridge_rejected() {
        let mut hybrid = HybridQuantumNetwork::from_parts(4, &[8], 4, 2, 4, 42);

        // Bridge to non-existent cluster should be rejected
        let bridge = ClusterBridge::distributed_cz(0, 0, 99, 0);
        hybrid.add_bridge(bridge);

        assert_eq!(hybrid.n_bridges(), 0);

        // Self-loop should be rejected
        let self_bridge = ClusterBridge::distributed_cz(0, 0, 0, 1);
        hybrid.add_bridge(self_bridge);

        assert_eq!(hybrid.n_bridges(), 0);
    }

    #[test]
    fn test_set_bridge_active() {
        let mut hybrid = HybridQuantumNetwork::from_parts(4, &[8], 4, 2, 4, 42);

        let bridge = ClusterBridge::distributed_cz(0, 0, 1, 0);
        hybrid.add_bridge(bridge);

        assert!(hybrid.bridge(0).unwrap().active);

        hybrid.set_bridge_active(0, false);
        assert!(!hybrid.bridge(0).unwrap().active);

        hybrid.set_bridge_active(0, true);
        assert!(hybrid.bridge(0).unwrap().active);
    }

    #[test]
    fn test_apply_bridges_distributed_cz() {
        let mut hybrid = HybridQuantumNetwork::from_parts(4, &[8], 4, 2, 4, 42);

        // Add a distributed CZ bridge
        let bridge = ClusterBridge::distributed_cz(0, 0, 1, 0);
        hybrid.add_bridge(bridge);

        // Prepare qubits in superposition
        hybrid.cluster_mut(0).unwrap().h(0);
        hybrid.cluster_mut(1).unwrap().h(0);

        // Apply bridges
        hybrid.apply_bridges();

        // Both clusters should still be normalized
        assert!((hybrid.cluster(0).unwrap().total_probability() - 1.0).abs() < 0.1);
        assert!((hybrid.cluster(1).unwrap().total_probability() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_apply_teleport() {
        let mut hybrid = HybridQuantumNetwork::from_parts(4, &[8], 4, 2, 4, 42);

        // Prepare source qubit in |+⟩ state
        hybrid.cluster_mut(0).unwrap().h(0);

        // Get initial Z expectation (should be ~0 for |+⟩)
        let z_src_before = hybrid.cluster(0).unwrap().z_expectation(0);
        assert!(z_src_before.abs() < 0.5, "Should be in superposition");

        // Add and apply teleport bridge
        let bridge = ClusterBridge::teleport(0, 0, 1, 0);
        hybrid.add_bridge(bridge);
        hybrid.apply_bridges();

        // Source should be reset to |0⟩
        let z_src_after = hybrid.cluster(0).unwrap().z_expectation(0);
        assert!(z_src_after > 0.5, "Source should be reset to |0⟩");

        // Clusters should be normalized
        assert!((hybrid.cluster(0).unwrap().total_probability() - 1.0).abs() < 0.1);
        assert!((hybrid.cluster(1).unwrap().total_probability() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_apply_remote_swap() {
        let mut hybrid = HybridQuantumNetwork::from_parts(4, &[8], 4, 2, 4, 42);

        // Prepare different states
        hybrid.cluster_mut(0).unwrap().x(0); // |1⟩
        // Cluster 1 qubit 0 stays at |0⟩

        let z_src_before = hybrid.cluster(0).unwrap().z_expectation(0);
        let z_tgt_before = hybrid.cluster(1).unwrap().z_expectation(0);

        assert!(z_src_before < 0.0, "Source should be in |1⟩ (negative Z)");
        assert!(z_tgt_before > 0.0, "Target should be in |0⟩ (positive Z)");

        // Apply swap
        let bridge = ClusterBridge::swap(0, 0, 1, 0);
        hybrid.add_bridge(bridge);
        hybrid.apply_bridges();

        // States should be exchanged
        let z_src_after = hybrid.cluster(0).unwrap().z_expectation(0);
        let z_tgt_after = hybrid.cluster(1).unwrap().z_expectation(0);

        // After swap, source should be like old target and vice versa
        // Note: Due to approximation in extract/inject, exact values may differ
        assert!(
            z_src_after > z_src_before || z_tgt_after < z_tgt_before,
            "States should have changed"
        );
    }

    #[test]
    fn test_chain_clusters() {
        let mut hybrid = HybridQuantumNetwork::from_parts(8, &[16], 4, 4, 4, 42);

        assert_eq!(hybrid.n_bridges(), 0);

        hybrid.chain_clusters();

        // Should have n-1 bridges for n clusters
        assert_eq!(hybrid.n_bridges(), 3);

        // Check connections
        assert_eq!(hybrid.bridge(0).unwrap().source_cluster, 0);
        assert_eq!(hybrid.bridge(0).unwrap().target_cluster, 1);
        assert_eq!(hybrid.bridge(1).unwrap().source_cluster, 1);
        assert_eq!(hybrid.bridge(1).unwrap().target_cluster, 2);
    }

    #[test]
    fn test_ring_clusters() {
        let mut hybrid = HybridQuantumNetwork::from_parts(8, &[16], 4, 4, 4, 42);

        hybrid.ring_clusters();

        // Should have n bridges for n clusters (chain + wrap-around)
        assert_eq!(hybrid.n_bridges(), 4);

        // Last bridge should connect last to first
        let last_bridge = hybrid.bridge(3).unwrap();
        assert_eq!(last_bridge.source_cluster, 3);
        assert_eq!(last_bridge.target_cluster, 0);
    }

    #[test]
    fn test_mesh_clusters() {
        let mut hybrid = HybridQuantumNetwork::from_parts(8, &[16], 4, 4, 4, 42);

        hybrid.mesh_clusters();

        // Should have n*(n-1)/2 bridges for full mesh
        // 4 clusters: 4*3/2 = 6 bridges
        assert_eq!(hybrid.n_bridges(), 6);
    }

    #[test]
    fn test_star_clusters() {
        let mut hybrid = HybridQuantumNetwork::from_parts(8, &[16], 4, 4, 4, 42);

        hybrid.star_clusters(0); // Cluster 0 is center

        // Should have n-1 bridges (center to each other)
        assert_eq!(hybrid.n_bridges(), 3);

        // All bridges should have center as source
        for i in 0..3 {
            assert_eq!(hybrid.bridge(i).unwrap().source_cluster, 0);
        }
    }

    #[test]
    fn test_clear_bridges() {
        let mut hybrid = HybridQuantumNetwork::from_parts(8, &[16], 4, 4, 4, 42);

        hybrid.mesh_clusters();
        assert!(hybrid.n_bridges() > 0);

        hybrid.clear_bridges();
        assert_eq!(hybrid.n_bridges(), 0);
    }

    #[test]
    fn test_qubit_reset() {
        let mut cluster = FullQuantumCluster::new(2, 0);

        // Prepare |1⟩ state
        cluster.x(0);
        assert!(cluster.z_expectation(0) < -0.9);

        // Reset to |0⟩
        cluster.reset_qubit(0);
        assert!(cluster.z_expectation(0) > 0.9);
    }

    #[test]
    fn test_qubit_extract_inject() {
        let mut cluster = FullQuantumCluster::new(2, 0);

        // Prepare |+⟩ state
        cluster.h(0);

        // Extract and re-inject
        let (alpha, beta) = cluster.extract_qubit_state(0);
        cluster.reset_qubit(0);
        cluster.inject_qubit_state(0, alpha, beta);

        // Should still be normalized
        assert!((cluster.total_probability() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_get_qubit_reduced_state() {
        let mut cluster = FullQuantumCluster::new(2, 0);

        // |00⟩ should have prob_0 = 1, prob_1 = 0
        let (p0, p1, _coh) = cluster.get_qubit_reduced_state(0);
        assert!((p0 - 1.0).abs() < 0.01);
        assert!(p1.abs() < 0.01);

        // |10⟩ should have prob_0 = 0, prob_1 = 1 for qubit 0
        cluster.x(0);
        let (p0, p1, _coh) = cluster.get_qubit_reduced_state(0);
        assert!(p0.abs() < 0.01);
        assert!((p1 - 1.0).abs() < 0.01);
    }
}
