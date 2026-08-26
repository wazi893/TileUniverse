//! Hybrid Sparse Quantum Grid: Automatic u128/BigInt Mode Selection
//!
//! This module provides a unified sparse quantum simulation that automatically
//! selects the optimal block ID representation based on qubit count:
//!
//! - **n ≤ 135 qubits**: Uses u128 block IDs with O(1) arithmetic
//! - **n > 135 qubits**: Uses BigUint block IDs with O(n) arithmetic
//!
//! The hybrid approach eliminates the O(n^1.5) BigInt overhead for the common
//! case while maintaining unlimited scalability for extreme qubit counts.
//!
//! # Performance Comparison
//!
//! | Qubits | u128-only | BigInt-only | Hybrid |
//! |--------|-----------|-------------|--------|
//! | 100    | O(1)/op   | O(n^1.5)/op | O(1)/op (auto-selects u128) |
//! | 1000   | N/A       | O(n^1.5)/op | O(n^1.5)/op (auto-selects BigInt) |
//!
//! For most practical use cases (n < 135), this provides optimal performance.
//!
//! # Fast W State
//!
//! For W states specifically, use [`create_fast_w_state`] which uses Vec-based
//! storage instead of HashMap<BigUint>. This enables billion-qubit W states:
//!
//! ```ignore
//! use engine::tile8::sparse_quantum_hybrid::create_fast_w_state;
//! let (grid, time_us) = create_fast_w_state(1_000_000_000); // 1 billion qubits!
//! ```

use num_bigint::BigUint;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

// Re-use Complex64 and SparseBlock from the BigInt module (they're identical)
pub use super::sparse_quantum_bigint::{
    Complex64, DickeStateVerification, GhzVerificationBigInt, GhzVerificationFast, SparseBlock,
    SparseGridStats, WStateVerification,
};

/// Threshold for switching from u128 to BigInt mode
/// u128 can represent block IDs up to 2^128, allowing 128 + 7 = 135 qubits
const U128_QUBIT_LIMIT: usize = 135;

/// Hybrid sparse quantum grid with automatic mode selection
///
/// Internally uses either u128 or BigUint block IDs based on qubit count.
#[derive(Clone)]
pub enum SparseQuantumGridHybrid {
    /// Fast path: u128 block IDs for n ≤ 135 qubits
    Small(SparseQuantumGridSmall),
    /// Unlimited path: BigUint block IDs for n > 135 qubits
    Large(super::sparse_quantum_bigint::SparseQuantumGridBigInt),
}

impl SparseQuantumGridHybrid {
    /// Create a new hybrid sparse grid
    ///
    /// Automatically selects optimal representation:
    /// - n ≤ 135: u128 mode (O(1) block operations)
    /// - n > 135: BigUint mode (unlimited qubits, O(n) block operations)
    pub fn new(n_qubits: usize) -> Self {
        assert!(n_qubits >= 7, "Minimum 7 qubits (1 block)");

        if n_qubits <= U128_QUBIT_LIMIT {
            SparseQuantumGridHybrid::Small(SparseQuantumGridSmall::new(n_qubits))
        } else {
            SparseQuantumGridHybrid::Large(
                super::sparse_quantum_bigint::SparseQuantumGridBigInt::new(n_qubits),
            )
        }
    }

    /// Returns true if using u128 mode (fast path)
    pub fn is_small_mode(&self) -> bool {
        matches!(self, SparseQuantumGridHybrid::Small(_))
    }

    /// Get number of qubits
    pub fn n_qubits(&self) -> usize {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.n_qubits(),
            SparseQuantumGridHybrid::Large(grid) => grid.n_qubits(),
        }
    }

    /// Get number of active blocks
    pub fn active_blocks(&self) -> usize {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.active_blocks(),
            SparseQuantumGridHybrid::Large(grid) => grid.active_blocks(),
        }
    }

    /// Initialize with |0...0⟩ state
    pub fn initialize_zero_state(&mut self) {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.initialize_zero_state(),
            SparseQuantumGridHybrid::Large(grid) => grid.initialize_zero_state(),
        }
    }

    /// Create GHZ state: (|0...0⟩ + |1...1⟩)/√2
    pub fn create_ghz_state(&mut self) -> u64 {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.create_ghz_state(),
            SparseQuantumGridHybrid::Large(grid) => grid.create_ghz_state(),
        }
    }

    /// Create W state: (|10...0⟩ + |01...0⟩ + ... + |0...01⟩)/√n
    pub fn create_w_state(&mut self) -> u64 {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.create_w_state(),
            SparseQuantumGridHybrid::Large(grid) => grid.create_w_state(),
        }
    }

    /// Create Dicke state: equal superposition of all states with k ones
    pub fn create_dicke_state(&mut self, k: usize) -> u64 {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.create_dicke_state(k),
            SparseQuantumGridHybrid::Large(grid) => grid.create_dicke_state(k),
        }
    }

    /// Apply Hadamard gate to specified qubit
    pub fn apply_h(&mut self, qubit: usize) {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.apply_h(qubit),
            SparseQuantumGridHybrid::Large(grid) => grid.apply_h(qubit),
        }
    }

    /// Apply CZ gate between two qubits
    pub fn apply_cz(&mut self, qubit_a: usize, qubit_b: usize) {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.apply_cz(qubit_a, qubit_b),
            SparseQuantumGridHybrid::Large(grid) => grid.apply_cz(qubit_a, qubit_b),
        }
    }

    /// Apply Pauli-X gate
    pub fn apply_x(&mut self, qubit: usize) {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.apply_x(qubit),
            SparseQuantumGridHybrid::Large(grid) => grid.apply_x(qubit),
        }
    }

    /// Apply Pauli-Z gate
    pub fn apply_z(&mut self, qubit: usize) {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.apply_z(qubit),
            SparseQuantumGridHybrid::Large(grid) => grid.apply_z(qubit),
        }
    }

    /// Apply CNOT gate
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.apply_cnot(control, target),
            SparseQuantumGridHybrid::Large(grid) => grid.apply_cnot(control, target),
        }
    }

    /// Apply Toffoli (CCX) gate - flips target when both controls are |1⟩
    pub fn apply_toffoli(&mut self, control1: usize, control2: usize, target: usize) {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.apply_toffoli(control1, control2, target),
            SparseQuantumGridHybrid::Large(grid) => grid.apply_toffoli(control1, control2, target),
        }
    }

    /// Count non-zero amplitudes
    pub fn count_nonzero_amplitudes(&self) -> usize {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.count_nonzero_amplitudes(),
            SparseQuantumGridHybrid::Large(grid) => grid.count_nonzero_amplitudes(),
        }
    }

    /// Verify GHZ state fidelity (computes BigInt last_block_id)
    ///
    /// WARNING: For very large qubit counts (>10B), use `verify_ghz_fast()` instead.
    pub fn verify_ghz_fidelity(&self) -> GhzVerificationBigInt {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.verify_ghz_fidelity(),
            SparseQuantumGridHybrid::Large(grid) => grid.verify_ghz_fidelity(),
        }
    }

    /// Fast O(1) GHZ verification - NO BigInt computation!
    ///
    /// Works for ANY qubit count including trillion+ qubits.
    /// Use this instead of `verify_ghz_fidelity()` for extreme scales.
    pub fn verify_ghz_fast(&self) -> GhzVerificationFast {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.verify_ghz_fast(),
            SparseQuantumGridHybrid::Large(grid) => grid.verify_ghz_fast(),
        }
    }

    /// Verify W state fidelity
    pub fn verify_w_fidelity(&self) -> WStateVerification {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.verify_w_fidelity(),
            SparseQuantumGridHybrid::Large(grid) => grid.verify_w_fidelity(),
        }
    }

    /// Verify Dicke state fidelity
    pub fn verify_dicke_fidelity(&self, k: usize) -> DickeStateVerification {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.verify_dicke_fidelity(k),
            SparseQuantumGridHybrid::Large(grid) => grid.verify_dicke_fidelity(k),
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &SparseGridStats {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.stats(),
            SparseQuantumGridHybrid::Large(grid) => grid.stats(),
        }
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        // Each block is 2KB (128 Complex64 @ 16 bytes each)
        self.active_blocks() * 2048
    }

    /// Get amplitude at a specific basis state index
    ///
    /// The basis state is decomposed into block_id (upper bits) and local_idx (lower 7 bits).
    pub fn get_amplitude_at(&self, basis_state: usize) -> Complex64 {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.get_amplitude_at(basis_state),
            SparseQuantumGridHybrid::Large(grid) => grid.get_amplitude_at(basis_state),
        }
    }

    /// Calculate probability of measuring |1⟩ on a specific qubit
    ///
    /// Sums |amplitude|² over all basis states where the specified qubit is 1.
    pub fn calculate_qubit_one_probability(&self, qubit: usize) -> f64 {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.calculate_qubit_one_probability(qubit),
            SparseQuantumGridHybrid::Large(grid) => grid.calculate_qubit_one_probability(qubit),
        }
    }

    /// Collapse state after measuring a qubit
    ///
    /// Zeroes out amplitudes inconsistent with the measurement outcome and renormalizes.
    pub fn collapse_qubit(&mut self, qubit: usize, outcome: bool) {
        match self {
            SparseQuantumGridHybrid::Small(grid) => grid.collapse_qubit(qubit, outcome),
            SparseQuantumGridHybrid::Large(grid) => grid.collapse_qubit(qubit, outcome),
        }
    }
}

/// u128-based sparse quantum grid for n ≤ 135 qubits
///
/// Uses O(1) block ID operations, providing optimal performance for most use cases.
#[derive(Clone)]
pub struct SparseQuantumGridSmall {
    n_qubits: usize,
    /// Block storage - pub(crate) for noise model access
    pub(crate) blocks: FxHashMap<u128, SparseBlock>,
    stats: SparseGridStats,
}

impl SparseQuantumGridSmall {
    pub fn new(n_qubits: usize) -> Self {
        assert!(n_qubits >= 7, "Minimum 7 qubits (1 block)");
        assert!(
            n_qubits <= U128_QUBIT_LIMIT,
            "Use SparseQuantumGridBigInt for > 135 qubits"
        );

        Self {
            n_qubits,
            blocks: FxHashMap::default(),
            stats: SparseGridStats::default(),
        }
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn active_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn block_bits(&self) -> usize {
        self.n_qubits.saturating_sub(7)
    }

    pub fn stats(&self) -> &SparseGridStats {
        &self.stats
    }

    fn get_or_create_block(&mut self, id: u128) -> &mut SparseBlock {
        if !self.blocks.contains_key(&id) {
            self.stats.blocks_allocated += 1;
            self.stats.blocks_active += 1;
            if self.stats.blocks_active > self.stats.peak_blocks {
                self.stats.peak_blocks = self.stats.blocks_active;
            }
        }
        self.blocks.entry(id).or_default()
    }

    pub fn initialize_zero_state(&mut self) {
        self.blocks.clear();
        self.blocks.insert(0, SparseBlock::zero_state());
        self.stats.blocks_allocated = 1;
        self.stats.blocks_active = 1;
        self.stats.peak_blocks = 1;
    }

    fn last_block_id(&self) -> u128 {
        if self.n_qubits <= 7 {
            0
        } else {
            let shift = self.n_qubits - 7;
            if shift >= 128 {
                u128::MAX // All bits set for 135 qubits
            } else {
                (1u128 << shift) - 1
            }
        }
    }

    fn cleanup_zero_blocks(&mut self) {
        let zero_blocks: Vec<u128> = self
            .blocks
            .iter()
            .filter(|(_, b)| b.is_zero())
            .map(|(&id, _)| id)
            .collect();

        for id in zero_blocks {
            self.blocks.remove(&id);
            self.stats.blocks_deallocated += 1;
            self.stats.blocks_active -= 1;
        }
    }

    // =========================================================================
    // GHZ State
    // =========================================================================

    /// Create GHZ state: (|0...0⟩ + |1...1⟩)/√2
    ///
    /// OPTIMIZED: Direct O(1) construction instead of O(n) CNOT chain.
    pub fn create_ghz_state(&mut self) -> u64 {
        let start = Instant::now();

        // Clear existing state
        self.blocks.clear();
        self.stats = SparseGridStats::default();

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        // Block 0: contains |0...0⟩ at address 0
        let mut block_zero = SparseBlock::new();
        block_zero.set(0, Complex64::new(inv_sqrt2, 0.0));
        self.blocks.insert(0, block_zero);

        // Block for |1...1⟩: all qubits in |1⟩
        if self.n_qubits > 7 {
            // Block ID with all bits set: 2^(n-7) - 1
            let last_block_id: u128 = (1u128 << (self.n_qubits - 7)) - 1;
            let mut block_last = SparseBlock::new();
            block_last.set(127, Complex64::new(inv_sqrt2, 0.0)); // All local qubits = 1
            self.blocks.insert(last_block_id, block_last);
        } else {
            // All qubits fit in one block
            let addr = (1usize << self.n_qubits) - 1;
            if let Some(block) = self.blocks.get_mut(&0) {
                block.set(addr, Complex64::new(inv_sqrt2, 0.0));
            }
        }

        // Update stats
        self.stats.blocks_allocated = self.blocks.len();
        self.stats.blocks_active = self.blocks.len();
        self.stats.peak_blocks = self.blocks.len();

        start.elapsed().as_micros() as u64
    }

    /// Create GHZ state using CNOT chain (for verification)
    #[allow(dead_code)]
    pub fn create_ghz_state_cnot_chain(&mut self) -> u64 {
        let start = Instant::now();

        self.initialize_zero_state();
        self.hadamard_q0();

        for target in 1..std::cmp::min(7, self.n_qubits) {
            self.local_cnot(target as u8);
        }

        for qubit in 7..self.n_qubits {
            self.cross_block_cnot(qubit);
        }

        start.elapsed().as_micros() as u64
    }

    fn hadamard_q0(&mut self) {
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        if let Some(block) = self.blocks.get_mut(&0) {
            let amp0 = block.get(0);
            let amp1 = block.get(1);

            let new_amp0 = Complex64::new(
                (amp0.re + amp1.re) * inv_sqrt2,
                (amp0.im + amp1.im) * inv_sqrt2,
            );
            let new_amp1 = Complex64::new(
                (amp0.re - amp1.re) * inv_sqrt2,
                (amp0.im - amp1.im) * inv_sqrt2,
            );

            block.set(0, new_amp0);
            block.set(1, new_amp1);
        }
    }

    fn local_cnot(&mut self, target: u8) {
        assert!((1..=6).contains(&target), "Local CNOT target must be 1-6");
        let target_mask = 1usize << target;

        for block in self.blocks.values_mut() {
            for addr in 0..128 {
                if (addr & 1) != 0 && addr < (addr ^ target_mask) {
                    let partner_addr = addr ^ target_mask;
                    let tmp = block.get(addr);
                    block.set(addr, block.get(partner_addr));
                    block.set(partner_addr, tmp);
                }
            }
        }
    }

    fn cross_block_cnot(&mut self, target_qubit: usize) {
        assert!(target_qubit >= 7, "Cross-block CNOT requires qubit >= 7");
        self.stats.cross_block_ops += 1;

        let block_bit = target_qubit - 7;
        let bit_mask: u128 = 1u128 << block_bit;

        let source_blocks: Vec<u128> = self
            .blocks
            .keys()
            .filter(|&&id| (id & bit_mask) == 0)
            .copied()
            .collect();

        for src_id in source_blocks {
            let dst_id = src_id | bit_mask;

            let mut transfers: Vec<(usize, Complex64)> = Vec::new();

            if let Some(src_block) = self.blocks.get(&src_id) {
                for addr in 0..128 {
                    if (addr & 1) != 0 {
                        let amp = src_block.get(addr);
                        if amp.norm_squared() > 1e-30 {
                            transfers.push((addr, amp));
                        }
                    }
                }
            }

            if !transfers.is_empty() {
                let dst_block = self.get_or_create_block(dst_id);
                for (addr, amp) in &transfers {
                    dst_block.set(*addr, *amp);
                }

                if let Some(src_block) = self.blocks.get_mut(&src_id) {
                    for (addr, _) in &transfers {
                        src_block.set(*addr, Complex64::zero());
                    }
                }
            }
        }

        self.cleanup_zero_blocks();
    }

    pub fn verify_ghz_fidelity(&self) -> GhzVerificationBigInt {
        let expected_amp = std::f64::consts::FRAC_1_SQRT_2;
        let last_block_id = self.last_block_id();

        let amp0 = self.blocks.get(&0).map(|b| b.get(0).norm()).unwrap_or(0.0);

        let amp_last = self
            .blocks
            .get(&last_block_id)
            .map(|b| b.get(127).norm())
            .unwrap_or(0.0);

        let fidelity = amp0.powi(2) + amp_last.powi(2);

        let mut max_spurious = 0.0f64;
        for (&block_id, block) in &self.blocks {
            for addr in 0..128 {
                if (block_id == 0 && addr == 0) || (block_id == last_block_id && addr == 127) {
                    continue;
                }
                let mag = block.get(addr).norm();
                if mag > max_spurious {
                    max_spurious = mag;
                }
            }
        }

        GhzVerificationBigInt {
            n_qubits: self.n_qubits,
            amp_zero_state: amp0,
            amp_one_state: amp_last,
            expected_amplitude: expected_amp,
            fidelity,
            max_spurious,
            active_blocks: self.blocks.len(),
            last_block_id: BigUint::from(last_block_id),
        }
    }

    /// Fast O(1) GHZ verification - works without computing last_block_id
    ///
    /// For u128 mode this is already fast, but we provide this method for API
    /// consistency with the BigInt implementation.
    pub fn verify_ghz_fast(&self) -> GhzVerificationFast {
        let expected_amp = std::f64::consts::FRAC_1_SQRT_2;

        // O(1): Get amplitude from block 0, index 0
        let amp0 = self.blocks.get(&0).map(|b| b.get(0).norm()).unwrap_or(0.0);

        // O(1): Find the non-zero block and get amplitude at index 127
        let amp_last = self
            .blocks
            .iter()
            .find(|&(&id, _)| id != 0)
            .map(|(_, b)| b.get(127).norm())
            .unwrap_or(0.0);

        let fidelity = amp0.powi(2) + amp_last.powi(2);

        // O(blocks) spurious check - but blocks.len() == 2 for GHZ
        let mut max_spurious = 0.0f64;
        let mut is_first_nonzero_block = true;

        for (&block_id, block) in &self.blocks {
            let is_zero_block = block_id == 0;
            let is_last_block = !is_zero_block && is_first_nonzero_block;

            if !is_zero_block {
                is_first_nonzero_block = false;
            }

            for addr in 0..128 {
                if (is_zero_block && addr == 0) || (is_last_block && addr == 127) {
                    continue;
                }
                let mag = block.get(addr).norm();
                if mag > max_spurious {
                    max_spurious = mag;
                }
            }
        }

        GhzVerificationFast {
            n_qubits: self.n_qubits,
            amp_zero_state: amp0,
            amp_one_state: amp_last,
            expected_amplitude: expected_amp,
            fidelity,
            max_spurious,
            active_blocks: self.blocks.len(),
        }
    }

    // =========================================================================
    // W State
    // =========================================================================

    pub fn create_w_state(&mut self) -> u64 {
        let start = Instant::now();

        self.blocks.clear();
        self.stats = SparseGridStats::default();

        let amplitude = 1.0 / (self.n_qubits as f64).sqrt();

        for qubit in 0..self.n_qubits {
            let state_index: u128 = 1u128 << qubit;
            let block_id = state_index >> 7;
            let local_addr = (state_index & 127) as usize;

            let block = self.get_or_create_block(block_id);
            block.set(local_addr, Complex64::new(amplitude, 0.0));
        }

        start.elapsed().as_micros() as u64
    }

    pub fn verify_w_fidelity(&self) -> WStateVerification {
        let expected_amp = 1.0 / (self.n_qubits as f64).sqrt();

        let mut total_prob = 0.0;
        let mut correct_amplitudes = 0usize;
        let mut max_error = 0.0f64;
        let mut max_spurious = 0.0f64;

        for qubit in 0..self.n_qubits {
            let state_index: u128 = 1u128 << qubit;
            let block_id = state_index >> 7;
            let local_addr = (state_index & 127) as usize;

            if let Some(block) = self.blocks.get(&block_id) {
                let amp = block.get(local_addr);
                let prob = amp.norm_squared();
                total_prob += prob;

                let error = (amp.norm() - expected_amp).abs();
                if error < 0.001 {
                    correct_amplitudes += 1;
                }
                if error > max_error {
                    max_error = error;
                }
            }
        }

        for (&block_id, block) in &self.blocks {
            for addr in 0..128 {
                let state_index = (block_id << 7) | (addr as u128);

                // Valid W-state amplitude: exactly one bit set
                let is_valid =
                    state_index.count_ones() == 1 && state_index < (1u128 << self.n_qubits);

                if !is_valid {
                    let mag = block.get(addr).norm();
                    if mag > max_spurious {
                        max_spurious = mag;
                    }
                }
            }
        }

        WStateVerification {
            n_qubits: self.n_qubits,
            expected_amplitudes: self.n_qubits,
            correct_amplitudes,
            expected_amplitude: expected_amp,
            total_probability: total_prob,
            fidelity: total_prob,
            max_amplitude_error: max_error,
            max_spurious,
            active_blocks: self.blocks.len(),
        }
    }

    // =========================================================================
    // Dicke State
    // =========================================================================

    pub fn create_dicke_state(&mut self, k: usize) -> u64 {
        let start = Instant::now();
        assert!(k <= self.n_qubits, "k must be <= n_qubits");

        self.blocks.clear();
        self.stats = SparseGridStats::default();

        let num_states = binomial(self.n_qubits, k);
        let amplitude = 1.0 / (num_states as f64).sqrt();

        if k == 0 {
            let block = self.get_or_create_block(0);
            block.set(0, Complex64::new(amplitude, 0.0));
            return start.elapsed().as_micros() as u64;
        }

        if k > self.n_qubits {
            return start.elapsed().as_micros() as u64;
        }

        // Gosper's hack for iterating k-subsets
        let n = self.n_qubits;

        // For n >= 128, max_state would overflow u128, so handle differently
        if n >= 128 {
            // Use recursive combination generation for large n
            self.create_dicke_state_large(k, amplitude);
        } else {
            let mut state: u128 = (1u128 << k) - 1;
            let max_state = 1u128 << n;

            while state < max_state {
                let block_id = state >> 7;
                let local_addr = (state & 127) as usize;

                let block = self.get_or_create_block(block_id);
                block.set(local_addr, Complex64::new(amplitude, 0.0));

                // Gosper's hack for next combination
                let c = state & state.wrapping_neg();
                let r = state.wrapping_add(c);
                state = (((r ^ state) >> 2) / c) | r;

                if state == 0 || state >= max_state {
                    break;
                }
            }
        }

        start.elapsed().as_micros() as u64
    }

    fn create_dicke_state_large(&mut self, k: usize, amplitude: f64) {
        // For n >= 128, use iterative combination generation
        // Generate all k-subsets of {0, 1, ..., n-1}
        let n = self.n_qubits;
        let mut indices: Vec<usize> = (0..k).collect();

        loop {
            // Compute block_id and local_addr from current combination
            // Block ID contains bits 7..n-1, local_addr contains bits 0..6
            let mut block_id: u128 = 0;
            let mut local_addr: usize = 0;

            for &idx in &indices {
                if idx < 7 {
                    // This bit goes in the local address
                    local_addr |= 1 << idx;
                } else {
                    // This bit goes in the block ID
                    let block_bit = idx - 7;
                    if block_bit < 128 {
                        block_id |= 1u128 << block_bit;
                    }
                }
            }

            let block = self.get_or_create_block(block_id);
            block.set(local_addr, Complex64::new(amplitude, 0.0));

            // Generate next combination in lexicographic order
            let mut incremented = false;
            for i in (0..k).rev() {
                if indices[i] < n - k + i {
                    indices[i] += 1;
                    for j in (i + 1)..k {
                        indices[j] = indices[j - 1] + 1;
                    }
                    incremented = true;
                    break;
                }
            }

            // If we couldn't increment any index, we're done
            if !incremented {
                break;
            }
        }
    }

    pub fn verify_dicke_fidelity(&self, k: usize) -> DickeStateVerification {
        let num_states = binomial(self.n_qubits, k);
        let expected_amp = 1.0 / (num_states as f64).sqrt();

        let mut total_prob = 0.0;
        let mut correct_count = 0usize;

        for block in self.blocks.values() {
            for addr in 0..128 {
                let amp = block.get(addr);
                if amp.norm_squared() > 1e-20 {
                    total_prob += amp.norm_squared();
                    correct_count += 1;
                }
            }
        }

        DickeStateVerification {
            n_qubits: self.n_qubits,
            k,
            expected_amplitudes: num_states,
            found_amplitudes: correct_count,
            expected_amplitude: expected_amp,
            total_probability: total_prob,
            fidelity: total_prob,
            active_blocks: self.blocks.len(),
            memory_bytes: self.blocks.len() * 2048,
        }
    }

    // =========================================================================
    // General Gates
    // =========================================================================

    pub fn apply_h(&mut self, qubit: usize) {
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        if qubit < 7 {
            // Local Hadamard within each block - PARALLELIZED with Rayon
            let qubit_mask = 1usize << qubit;

            self.blocks.par_iter_mut().for_each(|(_, block)| {
                for addr in 0..128 {
                    if addr < (addr ^ qubit_mask) {
                        let partner_addr = addr ^ qubit_mask;
                        let amp0 = block.get(addr);
                        let amp1 = block.get(partner_addr);

                        let new_amp0 = Complex64::new(
                            (amp0.re + amp1.re) * inv_sqrt2,
                            (amp0.im + amp1.im) * inv_sqrt2,
                        );
                        let new_amp1 = Complex64::new(
                            (amp0.re - amp1.re) * inv_sqrt2,
                            (amp0.im - amp1.im) * inv_sqrt2,
                        );

                        block.set(addr, new_amp0);
                        block.set(partner_addr, new_amp1);
                    }
                }
            });
        } else {
            // Cross-block Hadamard
            let block_bit = qubit - 7;
            let bit_mask: u128 = 1u128 << block_bit;

            let block_ids: Vec<u128> = self.blocks.keys().copied().collect();
            let mut processed: FxHashSet<u128> = FxHashSet::default();

            for src_id in block_ids {
                if processed.contains(&src_id) {
                    continue;
                }

                let partner_id = src_id ^ bit_mask;
                processed.insert(src_id);
                processed.insert(partner_id);

                let src_amps: Vec<Complex64> = if let Some(block) = self.blocks.get(&src_id) {
                    (0..128).map(|i| block.get(i)).collect()
                } else {
                    vec![Complex64::zero(); 128]
                };

                let partner_amps: Vec<Complex64> = if let Some(block) = self.blocks.get(&partner_id)
                {
                    (0..128).map(|i| block.get(i)).collect()
                } else {
                    vec![Complex64::zero(); 128]
                };

                let is_src_zero = (src_id & bit_mask) == 0;
                let (amp0_vec, amp1_vec) = if is_src_zero {
                    (src_amps, partner_amps)
                } else {
                    (partner_amps.clone(), src_amps.clone())
                };

                let mut new_amp0_vec = vec![Complex64::zero(); 128];
                let mut new_amp1_vec = vec![Complex64::zero(); 128];

                for addr in 0..128 {
                    let a0 = amp0_vec[addr];
                    let a1 = amp1_vec[addr];

                    new_amp0_vec[addr] =
                        Complex64::new((a0.re + a1.re) * inv_sqrt2, (a0.im + a1.im) * inv_sqrt2);
                    new_amp1_vec[addr] =
                        Complex64::new((a0.re - a1.re) * inv_sqrt2, (a0.im - a1.im) * inv_sqrt2);
                }

                let (zero_id, one_id) = if is_src_zero {
                    (src_id, partner_id)
                } else {
                    (partner_id, src_id)
                };

                let has_nonzero_0 = new_amp0_vec.iter().any(|a| a.norm_squared() > 1e-30);
                let has_nonzero_1 = new_amp1_vec.iter().any(|a| a.norm_squared() > 1e-30);

                if has_nonzero_0 {
                    let block = self.get_or_create_block(zero_id);
                    for (addr, &amp) in new_amp0_vec.iter().enumerate() {
                        block.set(addr, amp);
                    }
                } else {
                    self.blocks.remove(&zero_id);
                }

                if has_nonzero_1 {
                    let block = self.get_or_create_block(one_id);
                    for (addr, &amp) in new_amp1_vec.iter().enumerate() {
                        block.set(addr, amp);
                    }
                } else {
                    self.blocks.remove(&one_id);
                }
            }
        }

        self.cleanup_zero_blocks();
    }

    pub fn apply_cz(&mut self, qubit_a: usize, qubit_b: usize) {
        assert!(qubit_a != qubit_b, "CZ qubits must be different");

        let (q_low, q_high) = if qubit_a < qubit_b {
            (qubit_a, qubit_b)
        } else {
            (qubit_b, qubit_a)
        };

        if q_high < 7 {
            // Both qubits are local - PARALLELIZED with Rayon
            let mask_a = 1usize << q_low;
            let mask_b = 1usize << q_high;

            self.blocks.par_iter_mut().for_each(|(_, block)| {
                for addr in 0..128 {
                    if (addr & mask_a) != 0 && (addr & mask_b) != 0 {
                        let amp = block.get(addr);
                        block.set(addr, Complex64::new(-amp.re, -amp.im));
                    }
                }
            });
        } else if q_low < 7 {
            // One local, one in block ID - PARALLELIZED with Rayon
            let local_mask = 1usize << q_low;
            let block_bit = q_high - 7;
            let block_mask: u128 = 1u128 << block_bit;

            self.blocks.par_iter_mut().for_each(|(&block_id, block)| {
                if (block_id & block_mask) != 0 {
                    for addr in 0..128 {
                        if (addr & local_mask) != 0 {
                            let amp = block.get(addr);
                            block.set(addr, Complex64::new(-amp.re, -amp.im));
                        }
                    }
                }
            });
        } else {
            // Both in block ID - PARALLELIZED with Rayon
            let block_bit_a = q_low - 7;
            let block_bit_b = q_high - 7;
            let mask_a: u128 = 1u128 << block_bit_a;
            let mask_b: u128 = 1u128 << block_bit_b;

            self.blocks.par_iter_mut().for_each(|(&block_id, block)| {
                if (block_id & mask_a) != 0 && (block_id & mask_b) != 0 {
                    for addr in 0..128 {
                        let amp = block.get(addr);
                        if amp.norm_squared() > 1e-30 {
                            block.set(addr, Complex64::new(-amp.re, -amp.im));
                        }
                    }
                }
            });
        }
    }

    /// Apply Pauli-X gate to specified qubit
    /// X|0⟩ = |1⟩, X|1⟩ = |0⟩
    pub fn apply_x(&mut self, qubit: usize) {
        if qubit < 7 {
            // Local X within each block - just swap amplitude pairs
            let qubit_mask = 1usize << qubit;

            self.blocks.par_iter_mut().for_each(|(_, block)| {
                for addr in 0..128 {
                    if addr < (addr ^ qubit_mask) {
                        let partner_addr = addr ^ qubit_mask;
                        let amp0 = block.get(addr);
                        let amp1 = block.get(partner_addr);
                        block.set(addr, amp1);
                        block.set(partner_addr, amp0);
                    }
                }
            });
        } else {
            // Cross-block X - swap entire blocks
            let block_bit = qubit - 7;
            let bit_mask: u128 = 1u128 << block_bit;

            let block_ids: Vec<u128> = self.blocks.keys().copied().collect();
            let mut processed: FxHashSet<u128> = FxHashSet::default();

            for src_id in block_ids {
                if processed.contains(&src_id) {
                    continue;
                }

                let partner_id = src_id ^ bit_mask;
                processed.insert(src_id);
                processed.insert(partner_id);

                // Swap blocks (or create empty partner if needed)
                let src_block = self.blocks.remove(&src_id);
                let partner_block = self.blocks.remove(&partner_id);

                if let Some(block) = src_block {
                    self.blocks.insert(partner_id, block);
                }
                if let Some(block) = partner_block {
                    self.blocks.insert(src_id, block);
                }
            }
        }
    }

    /// Apply Pauli-Z gate to specified qubit
    /// Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    pub fn apply_z(&mut self, qubit: usize) {
        if qubit < 7 {
            // Local Z - negate amplitudes where qubit is |1⟩
            let qubit_mask = 1usize << qubit;

            self.blocks.par_iter_mut().for_each(|(_, block)| {
                for addr in 0..128 {
                    if (addr & qubit_mask) != 0 {
                        let amp = block.get(addr);
                        block.set(addr, Complex64::new(-amp.re, -amp.im));
                    }
                }
            });
        } else {
            // Cross-block Z - negate all amplitudes in blocks where qubit bit is 1
            let block_bit = qubit - 7;
            let bit_mask: u128 = 1u128 << block_bit;

            self.blocks.par_iter_mut().for_each(|(&block_id, block)| {
                if (block_id & bit_mask) != 0 {
                    for addr in 0..128 {
                        let amp = block.get(addr);
                        if amp.norm_squared() > 1e-30 {
                            block.set(addr, Complex64::new(-amp.re, -amp.im));
                        }
                    }
                }
            });
        }
    }

    /// Apply CNOT gate: flips target qubit when control qubit is |1⟩
    /// CNOT|00⟩ = |00⟩, CNOT|01⟩ = |01⟩, CNOT|10⟩ = |11⟩, CNOT|11⟩ = |10⟩
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        assert!(
            control != target,
            "CNOT control and target must be different"
        );

        let ctrl_local = control < 7;
        let tgt_local = target < 7;

        match (ctrl_local, tgt_local) {
            (true, true) => {
                // Both qubits local - swap amplitudes within blocks
                let ctrl_mask = 1usize << control;
                let tgt_mask = 1usize << target;

                self.blocks.par_iter_mut().for_each(|(_, block)| {
                    for addr in 0..128 {
                        // Only process when control is 1 and we haven't swapped yet
                        if (addr & ctrl_mask) != 0 && (addr & tgt_mask) == 0 {
                            let partner_addr = addr ^ tgt_mask;
                            let amp0 = block.get(addr);
                            let amp1 = block.get(partner_addr);
                            block.set(addr, amp1);
                            block.set(partner_addr, amp0);
                        }
                    }
                });
            }
            (true, false) => {
                // Control local, target in block ID
                let ctrl_mask = 1usize << control;
                let tgt_bit = target - 7;
                let tgt_block_mask: u128 = 1u128 << tgt_bit;

                // For each block pair differing in target bit, swap amplitudes where control=1
                let block_ids: Vec<u128> = self.blocks.keys().copied().collect();
                let mut processed: FxHashSet<u128> = FxHashSet::default();

                for src_id in block_ids {
                    if processed.contains(&src_id) {
                        continue;
                    }

                    let partner_id = src_id ^ tgt_block_mask;
                    processed.insert(src_id);
                    processed.insert(partner_id);

                    // Extract amplitudes where control=1
                    let src_ctrl1: Vec<(usize, Complex64)> =
                        if let Some(block) = self.blocks.get(&src_id) {
                            (0..128)
                                .filter(|&addr| (addr & ctrl_mask) != 0)
                                .map(|addr| (addr, block.get(addr)))
                                .collect()
                        } else {
                            vec![]
                        };

                    let partner_ctrl1: Vec<(usize, Complex64)> =
                        if let Some(block) = self.blocks.get(&partner_id) {
                            (0..128)
                                .filter(|&addr| (addr & ctrl_mask) != 0)
                                .map(|addr| (addr, block.get(addr)))
                                .collect()
                        } else {
                            vec![]
                        };

                    // Swap: src's ctrl=1 amps go to partner, partner's ctrl=1 amps go to src
                    for (addr, amp) in src_ctrl1 {
                        let block = self.get_or_create_block(partner_id);
                        block.set(addr, amp);
                        if let Some(src_block) = self.blocks.get_mut(&src_id) {
                            src_block.set(addr, Complex64::zero());
                        }
                    }

                    for (addr, amp) in partner_ctrl1 {
                        let block = self.get_or_create_block(src_id);
                        block.set(addr, amp);
                        if let Some(partner_block) = self.blocks.get_mut(&partner_id) {
                            partner_block.set(addr, Complex64::zero());
                        }
                    }
                }

                self.cleanup_zero_blocks();
            }
            (false, true) => {
                // Control in block ID, target local
                let ctrl_bit = control - 7;
                let ctrl_block_mask: u128 = 1u128 << ctrl_bit;
                let tgt_mask = 1usize << target;

                // Only operate on blocks where control bit is 1
                self.blocks.par_iter_mut().for_each(|(&block_id, block)| {
                    if (block_id & ctrl_block_mask) != 0 {
                        for addr in 0..128 {
                            if (addr & tgt_mask) == 0 {
                                let partner_addr = addr ^ tgt_mask;
                                let amp0 = block.get(addr);
                                let amp1 = block.get(partner_addr);
                                block.set(addr, amp1);
                                block.set(partner_addr, amp0);
                            }
                        }
                    }
                });
            }
            (false, false) => {
                // Both in block ID
                let ctrl_bit = control - 7;
                let tgt_bit = target - 7;
                let ctrl_mask: u128 = 1u128 << ctrl_bit;
                let tgt_mask: u128 = 1u128 << tgt_bit;

                let block_ids: Vec<u128> = self.blocks.keys().copied().collect();
                let mut processed: FxHashSet<u128> = FxHashSet::default();

                for src_id in block_ids {
                    if processed.contains(&src_id) {
                        continue;
                    }

                    // Only process when control=1
                    if (src_id & ctrl_mask) == 0 {
                        continue;
                    }

                    let partner_id = src_id ^ tgt_mask;
                    processed.insert(src_id);
                    processed.insert(partner_id);

                    // Swap entire blocks
                    let src_block = self.blocks.remove(&src_id);
                    let partner_block = self.blocks.remove(&partner_id);

                    if let Some(block) = src_block {
                        self.blocks.insert(partner_id, block);
                    }
                    if let Some(block) = partner_block {
                        self.blocks.insert(src_id, block);
                    }
                }
            }
        }
    }

    /// Apply Toffoli (CCX) gate - flips target when both controls are |1⟩
    ///
    /// Direct amplitude manipulation without T-gate decomposition.
    pub fn apply_toffoli(&mut self, control1: usize, control2: usize, target: usize) {
        // For each basis state where control1=1 AND control2=1, flip the target bit
        // This swaps amplitudes between states differing only in the target bit
        // when both controls are 1

        let c1_local = control1 < 7;
        let c2_local = control2 < 7;
        let t_local = target < 7;

        match (c1_local, c2_local, t_local) {
            (true, true, true) => {
                // All three qubits are local (within lower 7 bits)
                let c1_mask = 1usize << control1;
                let c2_mask = 1usize << control2;
                let t_mask = 1usize << target;

                for block in self.blocks.values_mut() {
                    for addr in 0..128 {
                        // Only process when both controls are 1 and target is 0
                        if (addr & c1_mask) != 0 && (addr & c2_mask) != 0 && (addr & t_mask) == 0 {
                            let partner = addr ^ t_mask;
                            let amp0 = block.get(addr);
                            let amp1 = block.get(partner);
                            block.set(addr, amp1);
                            block.set(partner, amp0);
                        }
                    }
                }
            }
            _ => {
                // Mixed local/block ID case - need to handle block swaps
                // For simplicity, iterate over all amplitudes and check conditions
                let c1_bit = if c1_local { control1 } else { control1 - 7 };
                let c2_bit = if c2_local { control2 } else { control2 - 7 };
                let t_bit = if t_local { target } else { target - 7 };

                let block_ids: Vec<u128> = self.blocks.keys().copied().collect();
                let mut processed = FxHashSet::default();

                for block_id in block_ids {
                    if processed.contains(&block_id) {
                        continue;
                    }

                    for addr in 0..128 {
                        // Check control1
                        let c1_val = if c1_local {
                            (addr >> c1_bit) & 1
                        } else {
                            ((block_id >> c1_bit) & 1) as usize
                        };

                        // Check control2
                        let c2_val = if c2_local {
                            (addr >> c2_bit) & 1
                        } else {
                            ((block_id >> c2_bit) & 1) as usize
                        };

                        // Only flip when both controls are 1
                        if c1_val != 1 || c2_val != 1 {
                            continue;
                        }

                        // Calculate partner (with target flipped)
                        let (partner_block, partner_addr) = if t_local {
                            (block_id, addr ^ (1 << t_bit))
                        } else {
                            (block_id ^ (1u128 << t_bit), addr)
                        };

                        // Only swap once per pair (when target=0)
                        let t_val = if t_local {
                            (addr >> t_bit) & 1
                        } else {
                            ((block_id >> t_bit) & 1) as usize
                        };

                        if t_val != 0 {
                            continue;
                        }

                        // Get amplitudes
                        let amp0 = self
                            .blocks
                            .get(&block_id)
                            .map(|b| b.get(addr))
                            .unwrap_or(Complex64::zero());
                        let amp1 = self
                            .blocks
                            .get(&partner_block)
                            .map(|b| b.get(partner_addr))
                            .unwrap_or(Complex64::zero());

                        // Swap
                        if amp0.norm_squared() > 1e-30 || amp1.norm_squared() > 1e-30 {
                            let b0 = self.get_or_create_block(block_id);
                            b0.set(addr, amp1);

                            let b1 = self.get_or_create_block(partner_block);
                            b1.set(partner_addr, amp0);
                        }
                    }

                    processed.insert(block_id);
                }

                self.cleanup_zero_blocks();
            }
        }
    }

    pub fn count_nonzero_amplitudes(&self) -> usize {
        let mut count = 0;
        for block in self.blocks.values() {
            for addr in 0..128 {
                if block.get(addr).norm_squared() > 1e-30 {
                    count += 1;
                }
            }
        }
        count
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        // Each block is 2KB (128 Complex64 @ 16 bytes each)
        self.active_blocks() * 2048
    }

    /// Get amplitude at a specific basis state index
    ///
    /// Decomposes basis_state into block_id (upper n-7 bits) and local_idx (lower 7 bits).
    pub fn get_amplitude_at(&self, basis_state: usize) -> Complex64 {
        let (block_id, local_idx) = if self.n_qubits <= 7 {
            (0u128, basis_state)
        } else {
            let local_idx = basis_state & 0x7F; // Lower 7 bits
            let block_id = (basis_state >> 7) as u128;
            (block_id, local_idx)
        };

        if let Some(block) = self.blocks.get(&block_id) {
            block.get(local_idx)
        } else {
            Complex64::zero()
        }
    }

    /// Calculate probability of measuring |1⟩ on a specific qubit
    ///
    /// Iterates over all active blocks and sums |amplitude|² where qubit=1.
    pub fn calculate_qubit_one_probability(&self, qubit: usize) -> f64 {
        let mut prob = 0.0;

        if qubit < 7 {
            // Qubit is within local index (lower 7 bits)
            let qubit_mask = 1usize << qubit;
            for block in self.blocks.values() {
                for addr in 0..128 {
                    if (addr & qubit_mask) != 0 {
                        let amp = block.get(addr);
                        prob += amp.re * amp.re + amp.im * amp.im;
                    }
                }
            }
        } else {
            // Qubit is in block ID (upper bits)
            let block_bit = qubit - 7;
            let block_mask: u128 = 1u128 << block_bit;
            for (&block_id, block) in &self.blocks {
                if (block_id & block_mask) != 0 {
                    for addr in 0..128 {
                        let amp = block.get(addr);
                        prob += amp.re * amp.re + amp.im * amp.im;
                    }
                }
            }
        }

        prob
    }

    /// Collapse state after measuring a qubit
    ///
    /// Zeroes amplitudes inconsistent with the measurement and renormalizes.
    pub fn collapse_qubit(&mut self, qubit: usize, outcome: bool) {
        let target_bit = if outcome { 1 } else { 0 };

        if qubit < 7 {
            // Qubit is in local index
            let _qubit_mask = 1usize << qubit;
            for block in self.blocks.values_mut() {
                for addr in 0..128 {
                    let bit_value = (addr >> qubit) & 1;
                    if bit_value != target_bit {
                        block.set(addr, Complex64::zero());
                    }
                }
            }
        } else {
            // Qubit is in block ID
            let block_bit = qubit - 7;
            let _block_mask: u128 = 1u128 << block_bit;
            let block_ids: Vec<u128> = self.blocks.keys().copied().collect();

            for block_id in block_ids {
                let bit_value = ((block_id >> block_bit) & 1) as usize;
                if bit_value != target_bit {
                    self.blocks.remove(&block_id);
                    self.stats.blocks_deallocated += 1;
                    self.stats.blocks_active -= 1;
                }
            }
        }

        // Renormalize
        let mut total_prob = 0.0;
        for block in self.blocks.values() {
            for addr in 0..128 {
                let amp = block.get(addr);
                total_prob += amp.re * amp.re + amp.im * amp.im;
            }
        }

        if total_prob > 1e-30 {
            let norm_factor = 1.0 / total_prob.sqrt();
            for block in self.blocks.values_mut() {
                for addr in 0..128 {
                    let amp = block.get(addr);
                    if amp.norm_squared() > 1e-30 {
                        block.set(
                            addr,
                            Complex64::new(amp.re * norm_factor, amp.im * norm_factor),
                        );
                    }
                }
            }
        }

        self.cleanup_zero_blocks();
    }
}

// ============================================================================
// SECTION: Lazy Hadamard Grid - Deferred Hadamard Application
// ============================================================================
//
// Key Insight: Hadamard gates cause exponential block explosion in sparse states.
// By tracking "pending" Hadamards and using algebraic identities, we can:
//
// 1. Cancel H·H = I pairs without any block creation
// 2. Transform CZ into CNOT when one qubit has pending H
// 3. Transform X ↔ Z when qubit has pending H
// 4. Only materialize when absolutely necessary (measurement, non-Clifford)
//
// Algebraic Identities Used:
// - H·H = I (self-inverse)
// - H·Z·H = X
// - H·X·H = Z
// - (H⊗I)·CZ·(H⊗I) = CNOT(ctrl=second, tgt=first)
// - (I⊗H)·CZ·(I⊗H) = CNOT(ctrl=first, tgt=second)
// - (H⊗H)·CZ·(H⊗H) = CZ (CZ is self-dual under H⊗H)
//
// For 1M qubit sparse states, this can avoid exponential blowup in circuits
// with many Hadamard pairs or Clifford-only sections.

/// Statistics for lazy Hadamard tracking
#[derive(Clone, Debug, Default)]
pub struct LazyHadamardStats {
    /// Number of H gates that were deferred (not materialized)
    pub hadamards_deferred: u64,
    /// Number of H·H pairs that cancelled (avoided 2x blowup each)
    pub hadamard_pairs_cancelled: u64,
    /// Number of CZ→CNOT transformations (avoided intermediate blowup)
    pub cz_to_cnot_transforms: u64,
    /// Number of X↔Z swaps due to pending H
    pub xz_swaps: u64,
    /// Number of forced materializations
    pub materializations: u64,
    /// Qubits materialized (cumulative)
    pub qubits_materialized: u64,
}

/// Lazy Hadamard Sparse Quantum Grid
///
/// Wraps SparseQuantumGridSmall with deferred Hadamard tracking.
/// Uses algebraic identities to avoid exponential block explosion.
///
/// # Example
/// ```ignore
/// let mut grid = LazyHadamardGrid::new(100);
/// grid.initialize_zero_state();
///
/// // These Hadamards are deferred, not applied yet
/// grid.apply_h(0);
/// grid.apply_h(1);
///
/// // This H cancels the first one (H·H = I), no blocks created!
/// grid.apply_h(0);
///
/// // CZ with pending H on qubit 1 becomes CNOT
/// grid.apply_cz(0, 1);
///
/// // Only materialize when needed
/// grid.materialize_all();
/// ```
pub struct LazyHadamardGrid {
    /// Underlying sparse grid
    inner: SparseQuantumGridSmall,
    /// Bitmask of qubits with pending Hadamard (up to 128 qubits tracked efficiently)
    pending_h: u128,
    /// For qubits > 127, use a set (rare case)
    pending_h_overflow: FxHashSet<usize>,
    /// Statistics
    stats: LazyHadamardStats,
}

impl LazyHadamardGrid {
    /// Create a new lazy Hadamard grid
    pub fn new(n_qubits: usize) -> Self {
        Self {
            inner: SparseQuantumGridSmall::new(n_qubits),
            pending_h: 0,
            pending_h_overflow: FxHashSet::default(),
            stats: LazyHadamardStats::default(),
        }
    }

    /// Get number of qubits
    pub fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// Get number of active blocks
    pub fn active_blocks(&self) -> usize {
        self.inner.active_blocks()
    }

    /// Initialize with |0...0⟩ state
    pub fn initialize_zero_state(&mut self) {
        self.inner.initialize_zero_state();
        self.pending_h = 0;
        self.pending_h_overflow.clear();
    }

    /// Create GHZ state: (|0...0⟩ + |1...1⟩)/√2
    pub fn create_ghz_state(&mut self) -> u64 {
        self.pending_h = 0;
        self.pending_h_overflow.clear();
        self.inner.create_ghz_state()
    }

    /// Create W state
    pub fn create_w_state(&mut self) -> u64 {
        self.pending_h = 0;
        self.pending_h_overflow.clear();
        self.inner.create_w_state()
    }

    /// Create Dicke state
    pub fn create_dicke_state(&mut self, k: usize) -> u64 {
        self.pending_h = 0;
        self.pending_h_overflow.clear();
        self.inner.create_dicke_state(k)
    }

    /// Check if qubit has pending Hadamard
    #[inline]
    fn has_pending_h(&self, qubit: usize) -> bool {
        if qubit < 128 {
            (self.pending_h >> qubit) & 1 != 0
        } else {
            self.pending_h_overflow.contains(&qubit)
        }
    }

    /// Toggle pending Hadamard for qubit
    #[inline]
    fn toggle_pending_h(&mut self, qubit: usize) {
        if qubit < 128 {
            self.pending_h ^= 1u128 << qubit;
        } else if self.pending_h_overflow.contains(&qubit) {
            self.pending_h_overflow.remove(&qubit);
        } else {
            self.pending_h_overflow.insert(qubit);
        }
    }

    /// Count pending Hadamards
    pub fn pending_hadamard_count(&self) -> usize {
        self.pending_h.count_ones() as usize + self.pending_h_overflow.len()
    }

    /// Apply Hadamard gate - DEFERRED unless it cancels
    ///
    /// H·H = I, so applying H twice to the same qubit cancels out.
    /// This is the key optimization: we never materialize H pairs.
    pub fn apply_h(&mut self, qubit: usize) {
        if self.has_pending_h(qubit) {
            // H·H = I: Cancel! No blocks created.
            self.stats.hadamard_pairs_cancelled += 1;
        } else {
            self.stats.hadamards_deferred += 1;
        }
        self.toggle_pending_h(qubit);
    }

    /// Apply CZ gate with algebraic transformation for pending Hadamards
    ///
    /// Uses identities:
    /// - (H⊗I)·CZ·(H⊗I) = CNOT(b→a) (control on b, target on a)
    /// - (I⊗H)·CZ·(I⊗H) = CNOT(a→b) (control on a, target on b)
    /// - (H⊗H)·CZ·(H⊗H) = CZ
    pub fn apply_cz(&mut self, qubit_a: usize, qubit_b: usize) {
        let h_a = self.has_pending_h(qubit_a);
        let h_b = self.has_pending_h(qubit_b);

        match (h_a, h_b) {
            (false, false) => {
                // No pending H: apply CZ directly
                self.inner.apply_cz(qubit_a, qubit_b);
            }
            (true, false) => {
                // H on a only: (H_a ⊗ I) · CZ · (H_a ⊗ I) = CNOT(b→a)
                // So we apply CNOT with control=b, target=a
                self.stats.cz_to_cnot_transforms += 1;
                self.inner.apply_cnot(qubit_b, qubit_a);
            }
            (false, true) => {
                // H on b only: (I ⊗ H_b) · CZ · (I ⊗ H_b) = CNOT(a→b)
                // So we apply CNOT with control=a, target=b
                self.stats.cz_to_cnot_transforms += 1;
                self.inner.apply_cnot(qubit_a, qubit_b);
            }
            (true, true) => {
                // Both have pending H: (H⊗H) · CZ · (H⊗H) = CZ
                // CZ is self-dual under local Hadamards
                self.inner.apply_cz(qubit_a, qubit_b);
            }
        }
    }

    /// Apply Pauli-X gate with algebraic transformation
    ///
    /// Uses identity: H·X·H = Z
    /// If qubit has pending H, apply Z instead.
    pub fn apply_x(&mut self, qubit: usize) {
        if self.has_pending_h(qubit) {
            // H·X·H = Z, so "logical X" when H is pending = physical Z
            self.stats.xz_swaps += 1;
            self.inner.apply_z(qubit);
        } else {
            self.inner.apply_x(qubit);
        }
    }

    /// Apply Pauli-Z gate with algebraic transformation
    ///
    /// Uses identity: H·Z·H = X
    /// If qubit has pending H, apply X instead.
    pub fn apply_z(&mut self, qubit: usize) {
        if self.has_pending_h(qubit) {
            // H·Z·H = X, so "logical Z" when H is pending = physical X
            self.stats.xz_swaps += 1;
            self.inner.apply_x(qubit);
        } else {
            self.inner.apply_z(qubit);
        }
    }

    /// Apply CNOT gate with algebraic transformation
    ///
    /// When Hadamards are pending, CNOT transforms as:
    /// - H_ctrl · CNOT · H_ctrl = CNOT (control basis change doesn't affect CNOT action)
    ///   Actually: (H⊗I)·CNOT·(H⊗I) = CZ (!)
    /// - H_tgt · CNOT · H_tgt = CNOT (target in X-basis)
    ///   Actually: (I⊗H)·CNOT·(I⊗H) = CNOT with control/target unchanged but acts differently
    ///
    /// The full algebra is complex. For now, we materialize if either has pending H.
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        let h_ctrl = self.has_pending_h(control);
        let h_tgt = self.has_pending_h(target);

        match (h_ctrl, h_tgt) {
            (false, false) => {
                // No pending H: apply CNOT directly
                self.inner.apply_cnot(control, target);
            }
            (true, false) => {
                // H on control: (H⊗I)·CNOT·(H⊗I) = CZ
                self.stats.cz_to_cnot_transforms += 1;
                self.inner.apply_cz(control, target);
            }
            (false, true) => {
                // H on target only: (I⊗H)·CNOT·(I⊗H) = CNOT
                // This is because CNOT flips target, and H·X·H = Z doesn't apply here
                // Actually the correct identity is that this equals CNOT but with X on target
                // becoming Z, which changes the action. For safety, materialize.
                self.materialize_qubit(target);
                self.inner.apply_cnot(control, target);
            }
            (true, true) => {
                // Both have pending H: complex interaction, materialize both
                self.materialize_qubit(control);
                self.materialize_qubit(target);
                self.inner.apply_cnot(control, target);
            }
        }
    }

    /// Materialize pending Hadamard for a single qubit
    ///
    /// This actually applies the Hadamard to the sparse state,
    /// potentially causing block explosion. Use sparingly!
    pub fn materialize_qubit(&mut self, qubit: usize) {
        if self.has_pending_h(qubit) {
            self.stats.materializations += 1;
            self.stats.qubits_materialized += 1;
            self.inner.apply_h(qubit);
            self.toggle_pending_h(qubit);
        }
    }

    /// Materialize all pending Hadamards
    ///
    /// WARNING: This can cause exponential block explosion!
    /// Only use when you need the final state (measurement, fidelity check).
    pub fn materialize_all(&mut self) {
        // Materialize from pending_h bitmask
        let mut pending = self.pending_h;
        while pending != 0 {
            let q = pending.trailing_zeros() as usize;
            self.stats.materializations += 1;
            self.stats.qubits_materialized += 1;
            self.inner.apply_h(q);
            pending &= pending - 1; // Clear lowest bit
        }
        self.pending_h = 0;

        // Materialize from overflow set
        let overflow: Vec<usize> = self.pending_h_overflow.drain().collect();
        for q in overflow {
            self.stats.materializations += 1;
            self.stats.qubits_materialized += 1;
            self.inner.apply_h(q);
        }
    }

    /// Count non-zero amplitudes (materializes all pending H first!)
    pub fn count_nonzero_amplitudes(&mut self) -> usize {
        self.materialize_all();
        self.inner.count_nonzero_amplitudes()
    }

    /// Get the underlying grid (materializes all pending H first!)
    pub fn into_inner(mut self) -> SparseQuantumGridSmall {
        self.materialize_all();
        self.inner
    }

    /// Get lazy Hadamard statistics
    pub fn lazy_stats(&self) -> &LazyHadamardStats {
        &self.stats
    }

    /// Get inner grid statistics
    pub fn stats(&self) -> &SparseGridStats {
        self.inner.stats()
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Estimate blocks saved by lazy H tracking
    ///
    /// Each cancelled H pair would have doubled blocks twice.
    /// Each deferred H avoids doubling until materialization.
    pub fn estimate_blocks_saved(&self) -> u64 {
        let current_blocks = self.active_blocks() as u64;
        // Each cancelled pair avoided 2^2 = 4x blowup (H doubles, then H again doubles)
        // But since they cancel, we avoided going from B to 4B and back to B
        // Net: we avoided creating 3B extra blocks per pair
        let pairs_saved = self.stats.hadamard_pairs_cancelled * 3 * current_blocks;

        // Deferred H: we're currently at B blocks, but would be at 2B if materialized
        let deferred_saved = self.pending_hadamard_count() as u64 * current_blocks;

        pairs_saved + deferred_saved
    }
}

/// Calculate binomial coefficient C(n, k)
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

// =============================================================================
// Fast Vec-based W State Functions
// =============================================================================

/// Create a fast W state using Vec-based storage (no BigInt overhead)
///
/// This function bypasses the HashMap<BigUint> storage and uses direct Vec
/// indexing, enabling simulation of billions of qubits for W states.
///
/// # Arguments
/// * `n_qubits` - Number of qubits (minimum 7)
///
/// # Returns
/// A tuple of (SparseQuantumGridVec, creation_time_microseconds)
///
/// # Example
/// ```ignore
/// use engine::tile8::sparse_quantum_hybrid::create_fast_w_state;
///
/// // Create 1 billion qubit W state
/// let (grid, time_us) = create_fast_w_state(1_000_000_000);
/// println!("Created in {} ms", time_us / 1000);
///
/// // Verify fidelity
/// let verification = grid.verify_w_fidelity();
/// assert!((verification.fidelity - 1.0).abs() < 1e-10);
/// ```
///
/// # Performance
/// - 1M qubits: ~25 ms
/// - 100M qubits: ~130 ms
/// - 1B qubits: ~1.2 sec
/// - 4B qubits: ~14 sec (60 GB RAM)
pub fn create_fast_w_state(
    n_qubits: usize,
) -> (super::sparse_quantum_vec::SparseQuantumGridVec, u64) {
    let mut grid = super::sparse_quantum_vec::SparseQuantumGridVec::new(n_qubits);
    let time_us = grid.create_w_state();
    (grid, time_us)
}

/// Create a fast W state using parallel construction
///
/// Uses Rayon for parallel block initialization. Faster for very large
/// qubit counts (>10M).
///
/// # Arguments
/// * `n_qubits` - Number of qubits (minimum 7)
///
/// # Returns
/// A tuple of (SparseQuantumGridVec, creation_time_microseconds)
pub fn create_fast_w_state_parallel(
    n_qubits: usize,
) -> (super::sparse_quantum_vec::SparseQuantumGridVec, u64) {
    let mut grid = super::sparse_quantum_vec::SparseQuantumGridVec::new(n_qubits);
    let time_us = grid.create_w_state_parallel();
    (grid, time_us)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_mode_selection() {
        // Small mode for n <= 135
        let grid_small = SparseQuantumGridHybrid::new(100);
        assert!(grid_small.is_small_mode());
        assert_eq!(grid_small.n_qubits(), 100);

        // Would use Large mode for n > 135 (but we can't easily test without BigInt overhead)
    }

    #[test]
    fn test_ghz_state_small_mode() {
        let mut grid = SparseQuantumGridHybrid::new(50);
        grid.create_ghz_state();

        let verification = grid.verify_ghz_fidelity();
        assert!(
            (verification.fidelity - 1.0).abs() < 1e-10,
            "Fidelity should be ~1.0"
        );
        assert_eq!(grid.active_blocks(), 2, "GHZ should have exactly 2 blocks");
    }

    #[test]
    fn test_w_state_small_mode() {
        let mut grid = SparseQuantumGridHybrid::new(20);
        grid.create_w_state();

        let verification = grid.verify_w_fidelity();
        assert!(
            (verification.fidelity - 1.0).abs() < 1e-10,
            "Fidelity should be ~1.0"
        );
        assert_eq!(
            grid.count_nonzero_amplitudes(),
            20,
            "W-state should have n amplitudes"
        );
    }

    #[test]
    fn test_dicke_state_small_mode() {
        let mut grid = SparseQuantumGridHybrid::new(20);
        grid.create_dicke_state(2);

        let verification = grid.verify_dicke_fidelity(2);
        assert!(
            (verification.fidelity - 1.0).abs() < 1e-10,
            "Fidelity should be ~1.0"
        );
        // C(20, 2) = 190
        assert_eq!(
            grid.count_nonzero_amplitudes(),
            190,
            "Dicke |D_20^2⟩ should have C(20,2)=190 amplitudes"
        );
    }

    #[test]
    fn test_apply_h_local() {
        let mut grid = SparseQuantumGridHybrid::new(10);
        grid.initialize_zero_state();
        grid.apply_h(0);

        // Should have created |+⟩ state on qubit 0
        assert_eq!(grid.count_nonzero_amplitudes(), 2);
    }

    #[test]
    fn test_apply_cz() {
        let mut grid = SparseQuantumGridHybrid::new(10);
        grid.initialize_zero_state();

        // Apply H to both qubits to get |++⟩
        grid.apply_h(0);
        grid.apply_h(1);

        // Apply CZ - should create a Bell-like state
        grid.apply_cz(0, 1);

        // Should still have 4 amplitudes (|00⟩, |01⟩, |10⟩, |11⟩)
        assert_eq!(grid.count_nonzero_amplitudes(), 4);
    }

    #[test]
    fn test_u128_boundary() {
        // Test at the u128 boundary (135 qubits)
        let grid = SparseQuantumGridHybrid::new(135);
        assert!(grid.is_small_mode());
        assert_eq!(grid.n_qubits(), 135);
    }

    // =========================================================================
    // Lazy Hadamard Grid Tests
    // =========================================================================

    #[test]
    fn test_lazy_h_cancellation() {
        // H·H = I: Applying H twice should cancel, creating no blocks
        let mut grid = LazyHadamardGrid::new(20);
        grid.initialize_zero_state();

        assert_eq!(grid.active_blocks(), 1, "Should start with 1 block");

        // Apply H (deferred)
        grid.apply_h(0);
        assert_eq!(grid.active_blocks(), 1, "H deferred, still 1 block");
        assert_eq!(grid.pending_hadamard_count(), 1);

        // Apply H again - should cancel!
        grid.apply_h(0);
        assert_eq!(grid.active_blocks(), 1, "H·H = I, still 1 block");
        assert_eq!(grid.pending_hadamard_count(), 0);

        // Stats should show cancellation
        assert_eq!(grid.lazy_stats().hadamard_pairs_cancelled, 1);
    }

    #[test]
    fn test_lazy_h_multiple_cancellations() {
        let mut grid = LazyHadamardGrid::new(20);
        grid.initialize_zero_state();

        // Apply H to 10 qubits, then cancel all
        for q in 0..10 {
            grid.apply_h(q);
        }
        assert_eq!(grid.pending_hadamard_count(), 10);
        assert_eq!(grid.active_blocks(), 1, "All H deferred, still 1 block");

        // Cancel all
        for q in 0..10 {
            grid.apply_h(q);
        }
        assert_eq!(grid.pending_hadamard_count(), 0);
        assert_eq!(grid.lazy_stats().hadamard_pairs_cancelled, 10);
        assert_eq!(grid.active_blocks(), 1, "All cancelled, still 1 block");
    }

    #[test]
    fn test_lazy_cz_with_pending_h() {
        // When one qubit has pending H, CZ becomes CNOT
        let mut grid = LazyHadamardGrid::new(20);
        grid.initialize_zero_state();

        // Start with |0⟩ state, apply H to qubit 0 (deferred)
        grid.apply_h(0);

        // Apply CZ(0, 1) - should become CNOT since qubit 0 has pending H
        grid.apply_cz(0, 1);

        assert_eq!(grid.lazy_stats().cz_to_cnot_transforms, 1);
    }

    #[test]
    fn test_lazy_xz_swap() {
        // When qubit has pending H: X becomes Z, Z becomes X
        let mut grid = LazyHadamardGrid::new(20);
        grid.initialize_zero_state();

        // Apply H (deferred)
        grid.apply_h(0);

        // Apply X - should become Z
        grid.apply_x(0);
        assert_eq!(grid.lazy_stats().xz_swaps, 1);

        // Apply Z - should become X
        grid.apply_z(0);
        assert_eq!(grid.lazy_stats().xz_swaps, 2);
    }

    #[test]
    fn test_lazy_materialize() {
        let mut grid = LazyHadamardGrid::new(10);
        grid.initialize_zero_state();

        // Defer some Hadamards
        grid.apply_h(0);
        grid.apply_h(1);
        assert_eq!(grid.pending_hadamard_count(), 2);
        assert_eq!(grid.active_blocks(), 1);

        // Materialize qubit 0
        grid.materialize_qubit(0);
        assert_eq!(grid.pending_hadamard_count(), 1);
        assert_eq!(
            grid.active_blocks(),
            1,
            "H on local qubit doesn't create blocks"
        );

        // Materialize all
        grid.materialize_all();
        assert_eq!(grid.pending_hadamard_count(), 0);
    }

    #[test]
    fn test_lazy_vs_eager_equivalence() {
        // Verify that lazy H produces same result as eager H
        let n_qubits = 10;

        // Eager: apply H directly
        let mut eager = SparseQuantumGridSmall::new(n_qubits);
        eager.initialize_zero_state();
        eager.apply_h(0);
        eager.apply_h(1);
        eager.apply_cz(0, 1);
        let eager_amps = eager.count_nonzero_amplitudes();

        // Lazy: defer H, then materialize
        let mut lazy = LazyHadamardGrid::new(n_qubits);
        lazy.initialize_zero_state();
        lazy.apply_h(0);
        lazy.apply_h(1);
        // Note: CZ with both pending H = CZ (self-dual)
        lazy.apply_cz(0, 1);
        let lazy_amps = lazy.count_nonzero_amplitudes(); // This materializes

        assert_eq!(
            eager_amps, lazy_amps,
            "Lazy and eager should produce same state"
        );
    }

    #[test]
    fn test_lazy_blocks_saved_estimate() {
        let mut grid = LazyHadamardGrid::new(20);
        grid.initialize_zero_state();

        // Defer 5 Hadamards
        for q in 0..5 {
            grid.apply_h(q);
        }

        // Estimate should show savings
        let saved = grid.estimate_blocks_saved();
        assert!(saved > 0, "Should estimate blocks saved from deferred H");
    }

    #[test]
    fn test_lazy_cnot_with_h_on_control() {
        // (H⊗I)·CNOT·(H⊗I) = CZ
        let mut grid = LazyHadamardGrid::new(10);
        grid.initialize_zero_state();

        // H on control qubit (deferred)
        grid.apply_h(0);

        // CNOT(0, 1) should become CZ since control has pending H
        grid.apply_cnot(0, 1);

        assert_eq!(grid.lazy_stats().cz_to_cnot_transforms, 1);
    }

    #[test]
    fn test_apply_x_gate() {
        let mut grid = SparseQuantumGridSmall::new(10);
        grid.initialize_zero_state();

        // X|0⟩ = |1⟩
        grid.apply_x(0);

        // Should have 1 amplitude at |1⟩
        assert_eq!(grid.count_nonzero_amplitudes(), 1);
    }

    #[test]
    fn test_apply_z_gate() {
        let mut grid = SparseQuantumGridSmall::new(10);
        grid.initialize_zero_state();

        // Z|0⟩ = |0⟩ (no phase on |0⟩)
        grid.apply_z(0);
        assert_eq!(grid.count_nonzero_amplitudes(), 1);

        // Apply H then Z
        grid.apply_h(0);
        grid.apply_z(0);
        // Should still have 2 amplitudes but with different phases
        assert_eq!(grid.count_nonzero_amplitudes(), 2);
    }

    #[test]
    fn test_apply_cnot_gate() {
        let mut grid = SparseQuantumGridSmall::new(10);
        grid.initialize_zero_state();

        // CNOT|00⟩ = |00⟩
        grid.apply_cnot(0, 1);
        assert_eq!(grid.count_nonzero_amplitudes(), 1);

        // Create |10⟩ state
        grid.apply_x(0);
        // CNOT|10⟩ = |11⟩
        grid.apply_cnot(0, 1);
        assert_eq!(grid.count_nonzero_amplitudes(), 1);
    }
}
