//! Sparse Noise Model for Quantum Error Simulation
//!
//! This module implements a sparse noise model that efficiently tracks error branches,
//! leveraging the insight that **Pauli errors (X, Y, Z) are sparsity-preserving**.
//!
//! # Key Insight
//!
//! Traditional density matrix simulation requires O(4^n) elements. Our approach:
//! - Track a **sparse set of error branches**, each with its own sparse quantum state
//! - Pauli errors don't increase amplitude count k, so each branch stays sparse
//! - Prune low-probability branches to keep simulation tractable
//!
//! # Sparsity Preservation
//!
//! For state |ψ⟩ with k non-zero amplitudes:
//! - X|ψ⟩ has k amplitudes (bit flip permutes basis states)
//! - Y|ψ⟩ has k amplitudes (bit flip + phase)
//! - Z|ψ⟩ has k amplitudes (phase flip, no new amplitudes)
//!
//! # Usage
//!
//! ```ignore
//! let mut noisy = SparseNoisyState::new(50);
//! noisy.apply_h(0);
//! noisy.apply_noise(0, 0.001);  // Depolarizing noise
//!
//! for i in 1..50 {
//!     noisy.apply_cnot(0, i);
//!     noisy.apply_noise(i, 0.001);
//! }
//!
//! println!("Branches: {}", noisy.num_branches());
//! println!("Fidelity: {:.6}", noisy.fidelity_with_ideal(&ideal_state));
//! ```

use super::sparse_quantum_hybrid::{Complex64, SparseBlock, SparseQuantumGridSmall};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;

// ============================================================================
// Pauli Operators
// ============================================================================

/// Single-qubit Pauli operator
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Pauli {
    I, // Identity
    X, // Bit flip
    Y, // Bit + phase flip
    Z, // Phase flip
}

impl Pauli {
    /// Multiply two Pauli operators (with phase absorbed into state)
    /// Returns (result, phase) where phase ∈ {1, -1, i, -i} encoded as (re, im)
    pub fn multiply(self, other: Pauli) -> (Pauli, (f64, f64)) {
        use Pauli::*;
        match (self, other) {
            // Identity cases
            (I, p) | (p, I) => (p, (1.0, 0.0)),

            // Same Pauli = Identity
            (X, X) | (Y, Y) | (Z, Z) => (I, (1.0, 0.0)),

            // XY = iZ, YX = -iZ
            (X, Y) => (Z, (0.0, 1.0)),
            (Y, X) => (Z, (0.0, -1.0)),

            // YZ = iX, ZY = -iX
            (Y, Z) => (X, (0.0, 1.0)),
            (Z, Y) => (X, (0.0, -1.0)),

            // ZX = iY, XZ = -iY
            (Z, X) => (Y, (0.0, 1.0)),
            (X, Z) => (Y, (0.0, -1.0)),
        }
    }
}

/// Sparse Pauli string - only stores non-identity terms
///
/// Represents a tensor product of Pauli operators: P = P_{i_1} ⊗ P_{i_2} ⊗ ...
/// where only non-identity terms are stored.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PauliString {
    /// Maps qubit index to Pauli operator (only non-I terms stored)
    terms: BTreeMap<usize, Pauli>,
}

impl PauliString {
    /// Create an identity Pauli string (no errors)
    pub fn identity() -> Self {
        Self {
            terms: BTreeMap::new(),
        }
    }

    /// Create a single-qubit Pauli string
    pub fn single(qubit: usize, pauli: Pauli) -> Self {
        let mut terms = BTreeMap::new();
        if pauli != Pauli::I {
            terms.insert(qubit, pauli);
        }
        Self { terms }
    }

    /// Get the Pauli operator on a specific qubit (I if not in map)
    pub fn get(&self, qubit: usize) -> Pauli {
        self.terms.get(&qubit).copied().unwrap_or(Pauli::I)
    }

    /// Set the Pauli operator on a specific qubit
    pub fn set(&mut self, qubit: usize, pauli: Pauli) {
        if pauli == Pauli::I {
            self.terms.remove(&qubit);
        } else {
            self.terms.insert(qubit, pauli);
        }
    }

    /// Multiply this Pauli string with another (tensor product composition)
    /// Returns (result, global_phase) where phase is (re, im)
    pub fn multiply(&self, other: &PauliString) -> (PauliString, (f64, f64)) {
        let mut result = self.clone();
        let mut phase = (1.0f64, 0.0f64);

        for (&qubit, &pauli) in &other.terms {
            let current = result.get(qubit);
            let (new_pauli, (pre, pim)) = current.multiply(pauli);

            // Multiply phases: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            let new_re = phase.0 * pre - phase.1 * pim;
            let new_im = phase.0 * pim + phase.1 * pre;
            phase = (new_re, new_im);

            result.set(qubit, new_pauli);
        }

        (result, phase)
    }

    /// Check if this is the identity (no errors)
    pub fn is_identity(&self) -> bool {
        self.terms.is_empty()
    }

    /// Get the weight (number of non-identity terms)
    pub fn weight(&self) -> usize {
        self.terms.len()
    }

    /// Iterate over non-identity terms
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &Pauli)> {
        self.terms.iter()
    }
}

// ============================================================================
// Error Branch
// ============================================================================

/// A single error branch in the noisy simulation
///
/// Each branch represents a possible error trajectory with:
/// - The accumulated Pauli error pattern
/// - The quantum state after applying that error
/// - The probability of this branch
#[derive(Clone)]
pub struct ErrorBranch {
    /// Accumulated error pattern (product of all Pauli errors)
    pub error: PauliString,
    /// Quantum state after error application
    pub state: SparseQuantumGridSmall,
    /// Probability of this branch
    pub probability: f64,
}

impl ErrorBranch {
    /// Create a new error branch
    pub fn new(state: SparseQuantumGridSmall, probability: f64) -> Self {
        Self {
            error: PauliString::identity(),
            state,
            probability,
        }
    }

    /// Clone state and apply a Pauli error, updating probability
    pub fn with_error(&self, qubit: usize, pauli: Pauli, prob_factor: f64) -> Self {
        let mut new_state = self.state.clone();

        // Apply the Pauli operator to the state
        match pauli {
            Pauli::I => {} // No change
            Pauli::X => apply_pauli_x(&mut new_state, qubit),
            Pauli::Y => apply_pauli_y(&mut new_state, qubit),
            Pauli::Z => apply_pauli_z(&mut new_state, qubit),
        }

        // Update error pattern
        let (new_error, _phase) = self.error.multiply(&PauliString::single(qubit, pauli));

        Self {
            error: new_error,
            state: new_state,
            probability: self.probability * prob_factor,
        }
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        // State: blocks * 2KB each
        // Error: ~8 bytes per term
        // Probability: 8 bytes
        self.state.active_blocks() * 2048 + self.error.weight() * 8 + 8
    }
}

// ============================================================================
// Pauli Gate Implementations
// ============================================================================

/// Apply Pauli X (bit flip) to a qubit
fn apply_pauli_x(state: &mut SparseQuantumGridSmall, qubit: usize) {
    if qubit < 7 {
        // Local X gate - swap amplitudes within each block
        let qubit_mask = 1usize << qubit;

        // Collect block ids first to avoid borrow issues
        let block_ids: Vec<u128> = state.blocks.keys().copied().collect();

        for block_id in block_ids {
            if let Some(block) = state.blocks.get_mut(&block_id) {
                for addr in 0..128 {
                    if addr < (addr ^ qubit_mask) {
                        let partner = addr ^ qubit_mask;
                        let tmp = block.get(addr);
                        block.set(addr, block.get(partner));
                        block.set(partner, tmp);
                    }
                }
            }
        }
    } else {
        // Cross-block X gate - swap amplitudes between blocks
        let block_bit = qubit - 7;
        let bit_mask: u128 = 1u128 << block_bit;

        let block_ids: Vec<u128> = state.blocks.keys().copied().collect();

        for src_id in &block_ids {
            let partner_id = src_id ^ bit_mask;
            if partner_id < *src_id {
                continue; // Process each pair once
            }

            // Swap entire block contents
            let src_amps: Option<Vec<Complex64>> = state
                .blocks
                .get(src_id)
                .map(|b| (0..128).map(|i| b.get(i)).collect());
            let partner_amps: Option<Vec<Complex64>> = state
                .blocks
                .get(&partner_id)
                .map(|b| (0..128).map(|i| b.get(i)).collect());

            // Update blocks
            if let Some(amps) = partner_amps {
                if !state.blocks.contains_key(src_id) {
                    state.blocks.insert(*src_id, SparseBlock::new());
                }
                if let Some(block) = state.blocks.get_mut(src_id) {
                    for (i, amp) in amps.iter().enumerate() {
                        block.set(i, *amp);
                    }
                }
            } else {
                state.blocks.remove(src_id);
            }

            if let Some(amps) = src_amps {
                state.blocks.entry(partner_id).or_default();
                if let Some(block) = state.blocks.get_mut(&partner_id) {
                    for (i, amp) in amps.iter().enumerate() {
                        block.set(i, *amp);
                    }
                }
            } else {
                state.blocks.remove(&partner_id);
            }
        }
    }
}

/// Apply Pauli Y (bit + phase flip) to a qubit
/// Y = iXZ, so Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
fn apply_pauli_y(state: &mut SparseQuantumGridSmall, qubit: usize) {
    if qubit < 7 {
        // Local Y gate
        let qubit_mask = 1usize << qubit;

        let block_ids: Vec<u128> = state.blocks.keys().copied().collect();

        for block_id in block_ids {
            if let Some(block) = state.blocks.get_mut(&block_id) {
                for addr in 0..128 {
                    let partner = addr ^ qubit_mask;
                    if addr < partner {
                        let amp0 = block.get(addr);
                        let amp1 = block.get(partner);

                        // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                        // new_amp0 = -i * amp1
                        // new_amp1 = i * amp0
                        let bit_set = (addr & qubit_mask) != 0;
                        if bit_set {
                            // addr has bit set: swap roles
                            block.set(addr, Complex64::new(amp1.im, -amp1.re)); // -i * amp1
                            block.set(partner, Complex64::new(-amp0.im, amp0.re)); // i * amp0
                        } else {
                            // addr has bit clear
                            block.set(addr, Complex64::new(amp1.im, -amp1.re)); // -i * amp1
                            block.set(partner, Complex64::new(-amp0.im, amp0.re)); // i * amp0
                        }
                    }
                }
            }
        }
    } else {
        // Cross-block Y gate
        apply_pauli_x(state, qubit);
        apply_pauli_z(state, qubit);
        // Apply global phase of i (handled implicitly in error tracking)
    }
}

/// Apply Pauli Z (phase flip) to a qubit
fn apply_pauli_z(state: &mut SparseQuantumGridSmall, qubit: usize) {
    if qubit < 7 {
        // Local Z gate - negate amplitudes where qubit is |1⟩
        let qubit_mask = 1usize << qubit;

        let block_ids: Vec<u128> = state.blocks.keys().copied().collect();

        for block_id in block_ids {
            if let Some(block) = state.blocks.get_mut(&block_id) {
                for addr in 0..128 {
                    if (addr & qubit_mask) != 0 {
                        let amp = block.get(addr);
                        block.set(addr, Complex64::new(-amp.re, -amp.im));
                    }
                }
            }
        }
    } else {
        // Cross-block Z gate - negate all amplitudes in blocks where bit is set
        let block_bit = qubit - 7;
        let bit_mask: u128 = 1u128 << block_bit;

        let block_ids: Vec<u128> = state
            .blocks
            .keys()
            .filter(|&&id| (id & bit_mask) != 0)
            .copied()
            .collect();

        for block_id in block_ids {
            if let Some(block) = state.blocks.get_mut(&block_id) {
                for addr in 0..128 {
                    let amp = block.get(addr);
                    if amp.norm_squared() > 1e-30 {
                        block.set(addr, Complex64::new(-amp.re, -amp.im));
                    }
                }
            }
        }
    }
}

// ============================================================================
// Adaptive Pruning
// ============================================================================

/// Adaptive pruning configuration
#[derive(Clone, Debug)]
pub struct AdaptivePruning {
    /// Current probability threshold
    pub threshold: f64,
    /// Minimum threshold (never go below)
    pub min_threshold: f64,
    /// Maximum threshold (never go above)
    pub max_threshold: f64,
    /// Target maximum number of branches
    pub max_branches: usize,
}

impl Default for AdaptivePruning {
    fn default() -> Self {
        Self {
            threshold: 1e-10,
            min_threshold: 1e-15,
            max_threshold: 1e-6,
            max_branches: 10000,
        }
    }
}

impl AdaptivePruning {
    /// Create with custom settings
    pub fn new(threshold: f64, max_branches: usize) -> Self {
        Self {
            threshold,
            min_threshold: threshold / 100000.0,
            max_threshold: threshold * 10000.0,
            max_branches,
        }
    }

    /// Adapt threshold based on current branch count
    pub fn adapt(&mut self, current_branches: usize) {
        if current_branches > self.max_branches {
            // Too many branches - increase threshold (more aggressive pruning)
            self.threshold = (self.threshold * 10.0).min(self.max_threshold);
        } else if current_branches < self.max_branches / 10 {
            // Few branches - decrease threshold (more conservative)
            self.threshold = (self.threshold / 10.0).max(self.min_threshold);
        }
    }
}

// ============================================================================
// Noise Statistics
// ============================================================================

/// Statistics for noise simulation
#[derive(Clone, Debug, Default)]
pub struct NoiseStats {
    /// Total branches created
    pub branches_created: usize,
    /// Total branches pruned
    pub branches_pruned: usize,
    /// Total branches merged
    pub branches_merged: usize,
    /// Total probability lost to pruning
    pub probability_pruned: f64,
    /// Maximum branches observed
    pub peak_branches: usize,
    /// Number of noise applications
    pub noise_applications: usize,
}

// ============================================================================
// Sparse Noisy State
// ============================================================================

/// Main sparse noisy state tracker
///
/// Tracks multiple error branches, each with its own sparse quantum state.
/// Uses adaptive pruning to manage branch explosion while preserving accuracy.
pub struct SparseNoisyState {
    /// Number of qubits
    n_qubits: usize,
    /// Error branches
    branches: Vec<ErrorBranch>,
    /// Adaptive pruning configuration
    pruning: AdaptivePruning,
    /// Statistics
    stats: NoiseStats,
}

impl SparseNoisyState {
    /// Create a new noisy state tracker
    ///
    /// Initializes with a single branch containing |0...0⟩ state with probability 1.0
    pub fn new(n_qubits: usize) -> Self {
        assert!(n_qubits >= 7, "Minimum 7 qubits (1 block)");
        assert!(n_qubits <= 135, "Maximum 135 qubits for u128 mode");

        let mut state = SparseQuantumGridSmall::new(n_qubits);
        state.initialize_zero_state();

        let branch = ErrorBranch::new(state, 1.0);

        Self {
            n_qubits,
            branches: vec![branch],
            pruning: AdaptivePruning::default(),
            stats: NoiseStats::default(),
        }
    }

    /// Create from an existing state
    pub fn from_state(state: SparseQuantumGridSmall) -> Self {
        let n_qubits = state.n_qubits();
        let branch = ErrorBranch::new(state, 1.0);

        Self {
            n_qubits,
            branches: vec![branch],
            pruning: AdaptivePruning::default(),
            stats: NoiseStats::default(),
        }
    }

    /// Get number of qubits
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Get number of branches
    pub fn num_branches(&self) -> usize {
        self.branches.len()
    }

    /// Get total probability (should be ~1.0)
    pub fn total_probability(&self) -> f64 {
        self.branches.iter().map(|b| b.probability).sum()
    }

    /// Get statistics
    pub fn stats(&self) -> &NoiseStats {
        &self.stats
    }

    /// Set probability threshold for pruning
    pub fn set_prob_threshold(&mut self, threshold: f64) {
        self.pruning.threshold = threshold;
    }

    /// Set maximum branches before aggressive pruning
    pub fn set_max_branches(&mut self, max: usize) {
        self.pruning.max_branches = max;
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.branches.iter().map(|b| b.memory_bytes()).sum()
    }

    // ========================================================================
    // Gate Operations (apply to all branches)
    // ========================================================================

    /// Apply Hadamard gate to all branches
    pub fn apply_h(&mut self, qubit: usize) {
        for branch in &mut self.branches {
            branch.state.apply_h(qubit);
        }
    }

    /// Apply CZ gate to all branches
    pub fn apply_cz(&mut self, qubit_a: usize, qubit_b: usize) {
        for branch in &mut self.branches {
            branch.state.apply_cz(qubit_a, qubit_b);
        }
    }

    /// Apply CNOT gate (control, target) to all branches
    /// CNOT = H_target * CZ * H_target
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        self.apply_h(target);
        self.apply_cz(control, target);
        self.apply_h(target);
    }

    /// Apply X gate to all branches
    pub fn apply_x(&mut self, qubit: usize) {
        for branch in &mut self.branches {
            apply_pauli_x(&mut branch.state, qubit);
        }
    }

    /// Apply Z gate to all branches
    pub fn apply_z(&mut self, qubit: usize) {
        for branch in &mut self.branches {
            apply_pauli_z(&mut branch.state, qubit);
        }
    }

    // ========================================================================
    // Noise Application
    // ========================================================================

    /// Apply depolarizing noise to a qubit
    ///
    /// Depolarizing channel: applies I with prob (1-p), X/Y/Z each with prob p/3
    ///
    /// # Arguments
    /// * `qubit` - The qubit to apply noise to
    /// * `p` - Total error probability (X, Y, Z each have probability p/3)
    pub fn apply_noise(&mut self, qubit: usize, p: f64) {
        assert!((0.0..=1.0).contains(&p), "Probability must be in [0, 1]");

        self.stats.noise_applications += 1;

        let p_i = 1.0 - p; // Probability of no error
        let p_xyz = p / 3.0; // Probability of each Pauli error

        let mut new_branches = Vec::with_capacity(self.branches.len() * 4);

        for branch in &self.branches {
            // Identity branch (no error) - most likely
            if p_i > self.pruning.threshold {
                let mut identity_branch = branch.clone();
                identity_branch.probability *= p_i;
                new_branches.push(identity_branch);
            }

            // X error branch
            if p_xyz > self.pruning.threshold {
                new_branches.push(branch.with_error(qubit, Pauli::X, p_xyz));
                self.stats.branches_created += 1;
            }

            // Y error branch
            if p_xyz > self.pruning.threshold {
                new_branches.push(branch.with_error(qubit, Pauli::Y, p_xyz));
                self.stats.branches_created += 1;
            }

            // Z error branch
            if p_xyz > self.pruning.threshold {
                new_branches.push(branch.with_error(qubit, Pauli::Z, p_xyz));
                self.stats.branches_created += 1;
            }
        }

        self.branches = new_branches;

        // Update peak branches
        if self.branches.len() > self.stats.peak_branches {
            self.stats.peak_branches = self.branches.len();
        }

        // Adapt pruning threshold if needed
        self.pruning.adapt(self.branches.len());

        // Prune low-probability branches
        self.prune();
    }

    /// Prune branches below threshold and renormalize
    pub fn prune(&mut self) {
        let threshold = self.pruning.threshold;
        let _before_count = self.branches.len();

        // Separate high and low probability branches
        let (keep, prune): (Vec<_>, Vec<_>) = self
            .branches
            .drain(..)
            .partition(|b| b.probability >= threshold);

        // Track pruned probability
        let pruned_prob: f64 = prune.iter().map(|b| b.probability).sum();
        self.stats.probability_pruned += pruned_prob;
        self.stats.branches_pruned += prune.len();

        self.branches = keep;

        // Renormalize remaining branches
        if !self.branches.is_empty() && pruned_prob > 0.0 {
            let total: f64 = self.branches.iter().map(|b| b.probability).sum();
            if total > 0.0 {
                let scale = 1.0 / total;
                for branch in &mut self.branches {
                    branch.probability *= scale;
                }
            }
        }
    }

    /// Merge branches with identical error patterns (optional optimization)
    pub fn merge_identical(&mut self) {
        // Group branches by error pattern
        let mut groups: FxHashMap<Vec<(usize, Pauli)>, Vec<usize>> = FxHashMap::default();

        for (i, branch) in self.branches.iter().enumerate() {
            let key: Vec<(usize, Pauli)> = branch.error.iter().map(|(&q, &p)| (q, p)).collect();
            groups.entry(key).or_default().push(i);
        }

        // Find groups with multiple branches
        let merge_groups: Vec<Vec<usize>> = groups
            .into_values()
            .filter(|indices| indices.len() > 1)
            .collect();

        if merge_groups.is_empty() {
            return;
        }

        // Mark indices to remove
        let mut to_remove: Vec<usize> = Vec::new();

        for indices in merge_groups {
            // Keep the first, merge others into it
            let keep_idx = indices[0];
            let mut combined_prob = self.branches[keep_idx].probability;

            for &idx in &indices[1..] {
                combined_prob += self.branches[idx].probability;
                to_remove.push(idx);
                self.stats.branches_merged += 1;
            }

            self.branches[keep_idx].probability = combined_prob;
        }

        // Remove merged branches (in reverse order to preserve indices)
        to_remove.sort_unstable();
        to_remove.reverse();
        for idx in to_remove {
            self.branches.remove(idx);
        }
    }

    // ========================================================================
    // Analysis Methods
    // ========================================================================

    /// Calculate fidelity with an ideal (noiseless) state
    ///
    /// Fidelity = probability of the identity error branch
    /// (branches where no errors occurred)
    pub fn fidelity_no_error(&self) -> f64 {
        self.branches
            .iter()
            .filter(|b| b.error.is_identity())
            .map(|b| b.probability)
            .sum()
    }

    /// Get the error distribution (probability of each error pattern)
    pub fn error_distribution(&self) -> Vec<(PauliString, f64)> {
        let mut dist: FxHashMap<Vec<(usize, Pauli)>, f64> = FxHashMap::default();

        for branch in &self.branches {
            let key: Vec<(usize, Pauli)> = branch.error.iter().map(|(&q, &p)| (q, p)).collect();
            *dist.entry(key).or_default() += branch.probability;
        }

        dist.into_iter()
            .map(|(key, prob)| {
                let mut ps = PauliString::identity();
                for (q, p) in key {
                    ps.set(q, p);
                }
                (ps, prob)
            })
            .collect()
    }

    /// Get syndrome distribution for QEC
    ///
    /// Returns a map from syndrome (as a vector of measurement outcomes) to probability.
    /// The syndrome is computed based on which qubits have errors.
    pub fn syndrome_distribution(&self) -> FxHashMap<Vec<bool>, f64> {
        let mut dist: FxHashMap<Vec<bool>, f64> = FxHashMap::default();

        for branch in &self.branches {
            // Syndrome: which qubits have non-identity errors
            let syndrome: Vec<bool> = (0..self.n_qubits)
                .map(|q| branch.error.get(q) != Pauli::I)
                .collect();

            *dist.entry(syndrome).or_default() += branch.probability;
        }

        dist
    }

    /// Calculate average error weight across branches
    pub fn average_error_weight(&self) -> f64 {
        let total_prob: f64 = self.branches.iter().map(|b| b.probability).sum();
        if total_prob == 0.0 {
            return 0.0;
        }

        let weighted_weight: f64 = self
            .branches
            .iter()
            .map(|b| b.error.weight() as f64 * b.probability)
            .sum();

        weighted_weight / total_prob
    }

    /// Get the ideal state (first branch with identity error, if it exists)
    pub fn ideal_branch(&self) -> Option<&ErrorBranch> {
        self.branches.iter().find(|b| b.error.is_identity())
    }

    // ========================================================================
    // State Preparation Helpers
    // ========================================================================

    /// Prepare a noisy GHZ state with depolarizing noise after each gate
    pub fn prepare_ghz_noisy(&mut self, error_rate: f64) {
        // H on qubit 0
        self.apply_h(0);
        self.apply_noise(0, error_rate);

        // CNOTs from qubit 0 to all others
        for i in 1..self.n_qubits {
            self.apply_cnot(0, i);
            self.apply_noise(i, error_rate);
        }
    }

    /// Verify GHZ fidelity against ideal GHZ state
    pub fn verify_ghz_fidelity(&self) -> f64 {
        // Fidelity is approximately the probability of no errors
        // For exact fidelity, we'd need to compute overlap with ideal state
        self.fidelity_no_error()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_multiply() {
        assert_eq!(Pauli::X.multiply(Pauli::X), (Pauli::I, (1.0, 0.0)));
        assert_eq!(Pauli::Y.multiply(Pauli::Y), (Pauli::I, (1.0, 0.0)));
        assert_eq!(Pauli::Z.multiply(Pauli::Z), (Pauli::I, (1.0, 0.0)));
        assert_eq!(Pauli::X.multiply(Pauli::Y), (Pauli::Z, (0.0, 1.0))); // XY = iZ
        assert_eq!(Pauli::Y.multiply(Pauli::X), (Pauli::Z, (0.0, -1.0))); // YX = -iZ
    }

    #[test]
    fn test_pauli_string() {
        let mut ps = PauliString::identity();
        assert!(ps.is_identity());

        ps.set(0, Pauli::X);
        ps.set(5, Pauli::Z);
        assert!(!ps.is_identity());
        assert_eq!(ps.weight(), 2);
        assert_eq!(ps.get(0), Pauli::X);
        assert_eq!(ps.get(5), Pauli::Z);
        assert_eq!(ps.get(3), Pauli::I);
    }

    #[test]
    fn test_sparse_noisy_state_new() {
        let noisy = SparseNoisyState::new(10);
        assert_eq!(noisy.n_qubits(), 10);
        assert_eq!(noisy.num_branches(), 1);
        assert!((noisy.total_probability() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_noise_splits_branches() {
        let mut noisy = SparseNoisyState::new(10);
        noisy.set_prob_threshold(1e-15); // Very low to keep all branches

        // Apply noise with p=0.01 (1% error rate)
        noisy.apply_noise(0, 0.01);

        // Should have 4 branches: I, X, Y, Z
        assert_eq!(noisy.num_branches(), 4);

        // Total probability should be ~1.0
        assert!((noisy.total_probability() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fidelity_no_error() {
        let mut noisy = SparseNoisyState::new(10);
        noisy.set_prob_threshold(1e-15);

        // Apply very low noise
        noisy.apply_noise(0, 0.001);

        // Fidelity should be ~0.999 (probability of no error)
        let fidelity = noisy.fidelity_no_error();
        assert!((fidelity - 0.999).abs() < 0.001);
    }

    #[test]
    fn test_ghz_noisy() {
        let mut noisy = SparseNoisyState::new(10);
        noisy.set_prob_threshold(1e-8);
        noisy.set_max_branches(1000);

        // Prepare noisy GHZ
        noisy.prepare_ghz_noisy(0.001);

        // Should have multiple branches
        assert!(noisy.num_branches() > 1);

        // Fidelity should be reasonable (> 0.9 for 1% error rate)
        let fidelity = noisy.verify_ghz_fidelity();
        assert!(fidelity > 0.5, "Fidelity should be > 0.5, got {}", fidelity);
    }

    #[test]
    fn test_adaptive_pruning() {
        let mut pruning = AdaptivePruning::new(1e-10, 100);

        // Simulate too many branches
        pruning.adapt(200);
        assert!(pruning.threshold > 1e-10);

        // Simulate few branches
        pruning.adapt(5);
        assert!(pruning.threshold < 1e-9);
    }
}
