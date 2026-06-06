//! Quantum Router F64: Full-Precision Quantum Operations
//!
//! This module provides f64-precision quantum operations for 100% fidelity.
//! Unlike the Fixed8-based quantum_ops.rs (98.88% fidelity), this uses
//! double-precision floating point throughout.
//!
//! Parallelized with rayon for multi-core performance.

use rayon::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

/// Wrapper to send raw pointer across threads safely
/// Safety: Caller must ensure non-overlapping access to pointed memory
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    #[inline]
    unsafe fn get_mut(&self, offset: usize) -> &mut T {
        unsafe { &mut *self.0.add(offset) }
    }
}

/// Complex amplitude with f64 precision
#[derive(Clone, Copy, Debug, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn norm_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }
}

impl std::ops::Add for Complex64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul<f64> for Complex64 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

/// Block of 128 complex amplitudes with f64 precision
#[derive(Clone)]
pub struct QuantumBlock {
    pub amplitudes: [Complex64; 128],
}

impl Default for QuantumBlock {
    fn default() -> Self {
        Self {
            amplitudes: [Complex64::zero(); 128],
        }
    }
}

impl QuantumBlock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize to |0⟩ state (amplitude[0] = 1.0)
    pub fn init_zero_state(&mut self) {
        self.amplitudes = [Complex64::zero(); 128];
        self.amplitudes[0] = Complex64::new(1.0, 0.0);
    }
}

/// Grid of quantum blocks with f64 precision
pub struct QuantumGridF64 {
    /// Quantum blocks (128 amplitudes each)
    pub blocks: Vec<QuantumBlock>,
    /// Number of qubits in the system
    pub n_qubits: usize,
}

impl QuantumGridF64 {
    /// Create a new quantum grid for n qubits
    ///
    /// For n >= 7, uses 2^(n-7) blocks of 128 amplitudes each
    pub fn new(n_qubits: usize) -> Self {
        assert!(n_qubits >= 7, "Minimum 7 qubits for block-based simulation");

        let num_blocks = 1usize << (n_qubits - 7);
        let mut blocks = Vec::with_capacity(num_blocks);

        for _ in 0..num_blocks {
            blocks.push(QuantumBlock::new());
        }

        // Initialize block 0, amplitude 0 to 1.0 (|0...0⟩ state)
        blocks[0].amplitudes[0] = Complex64::new(1.0, 0.0);

        Self { blocks, n_qubits }
    }

    /// Get the number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Apply Hadamard gate to qubit 0 (local within each block)
    /// Parallelized across blocks with rayon
    pub fn apply_hadamard_q0(&mut self) {
        self.blocks.par_iter_mut().for_each(|block| {
            for pair_idx in 0..64 {
                let i = pair_idx * 2;
                let j = i + 1;

                let amp_i = block.amplitudes[i];
                let amp_j = block.amplitudes[j];

                // H|0⟩ = (|0⟩ + |1⟩)/√2
                // H|1⟩ = (|0⟩ - |1⟩)/√2
                block.amplitudes[i] = (amp_i + amp_j) * FRAC_1_SQRT_2;
                block.amplitudes[j] = (amp_i - amp_j) * FRAC_1_SQRT_2;
            }
        });
    }

    /// Apply Hadamard gate to any local qubit (0-6)
    /// Parallelized across blocks with rayon
    pub fn apply_hadamard(&mut self, qubit: u8) {
        assert!(
            qubit < 7,
            "Use apply_hadamard_q0 for qubit 0, or qubit must be < 7"
        );

        let mask = 1usize << qubit;

        self.blocks.par_iter_mut().for_each(|block| {
            for state in 0..128 {
                if state & mask == 0 {
                    let partner = state | mask;
                    let amp_0 = block.amplitudes[state];
                    let amp_1 = block.amplitudes[partner];

                    block.amplitudes[state] = (amp_0 + amp_1) * FRAC_1_SQRT_2;
                    block.amplitudes[partner] = (amp_0 - amp_1) * FRAC_1_SQRT_2;
                }
            }
        });
    }

    /// Apply CNOT gate for local qubits (0-6)
    /// Parallelized across blocks with rayon
    pub fn apply_cnot_local(&mut self, control: u8, target: u8) {
        assert!(
            control < 7,
            "Control qubit must be 0-6 for local operations"
        );
        assert!(target < 7, "Target qubit must be 0-6 for local operations");
        assert_ne!(control, target, "Control and target must differ");

        let control_mask = 1usize << control;
        let target_mask = 1usize << target;

        self.blocks.par_iter_mut().for_each(|block| {
            for state in 0..128 {
                // Only process when control bit is set and target bit is 0
                if (state & control_mask != 0) && (state & target_mask == 0) {
                    let partner = state | target_mask;
                    block.amplitudes.swap(state, partner);
                }
            }
        });
    }

    /// Apply cross-block CNOT for qubit >= 7
    ///
    /// For qubit q >= 7, swaps amplitudes between blocks that differ in bit (q-7).
    /// Parallelized: independent block pairs are processed concurrently.
    pub fn apply_cnot_cross_block(&mut self, qubit: u8) {
        assert!(qubit >= 7, "Use apply_cnot_local for qubits 0-6");
        assert!((qubit as usize) < self.n_qubits, "Qubit out of range");

        let block_bit = (qubit - 7) as usize;
        let block_mask = 1usize << block_bit;
        let num_blocks = self.blocks.len();
        let num_pairs = num_blocks >> 1; // Half the blocks are "lower" blocks

        // Wrap raw pointer for thread safety
        // Safety: Each pair (block_id, partner_block) is disjoint from all other pairs
        let blocks_ptr = SendPtr(self.blocks.as_mut_ptr());

        // Process all independent pairs in parallel using pair index
        // Pair i corresponds to blocks: lower = (i / stride) * (stride * 2) + (i % stride)
        // where stride = block_mask
        (0..num_pairs).into_par_iter().for_each(move |pair_idx| {
            // Convert pair index to actual block ID
            let stride = block_mask;
            let group = pair_idx / stride;
            let offset = pair_idx % stride;
            let block_id = group * (stride * 2) + offset;
            let partner_block = block_id | block_mask;

            if partner_block < num_blocks {
                unsafe {
                    let block_a = blocks_ptr.get_mut(block_id);
                    let block_b = blocks_ptr.get_mut(partner_block);
                    for local_addr in 0..128 {
                        // Swap if control bit (qubit 0) is set in local address
                        if local_addr & 1 != 0 {
                            std::mem::swap(
                                &mut block_a.amplitudes[local_addr],
                                &mut block_b.amplitudes[local_addr],
                            );
                        }
                    }
                }
            }
        });
    }

    /// Create GHZ state: (|0...0⟩ + |1...1⟩)/√2
    /// Parallelized with rayon
    pub fn create_ghz_state(&mut self) -> u64 {
        let start = std::time::Instant::now();

        // Reset to |0...0⟩ (parallel)
        self.blocks.par_iter_mut().for_each(|block| {
            block.amplitudes = [Complex64::zero(); 128];
        });
        self.blocks[0].amplitudes[0] = Complex64::new(1.0, 0.0);

        // Apply H to qubit 0
        self.apply_hadamard_q0();

        // Apply CNOT chain: CNOT(0,1), CNOT(0,2), ..., CNOT(0,n-1)
        // For qubits 1-6: local CNOTs
        for target in 1..7.min(self.n_qubits) {
            self.apply_cnot_local(0, target as u8);
        }

        // For qubits 7+: cross-block CNOTs
        for qubit in 7..self.n_qubits {
            self.apply_cnot_cross_block(qubit as u8);
        }

        start.elapsed().as_micros() as u64
    }

    /// Verify GHZ fidelity
    /// Parallelized with rayon for max_spurious calculation
    pub fn verify_ghz_fidelity(&self) -> GhzVerification {
        // For GHZ state: only amplitudes at |0...0⟩ and |1...1⟩ should be non-zero
        // |0...0⟩ is block 0, local address 0
        // |1...1⟩ is last block, local address 127

        let amp_zero = self.blocks[0].amplitudes[0];
        let last_block = self.blocks.len() - 1;
        let amp_one = self.blocks[last_block].amplitudes[127];

        let prob_zero = amp_zero.norm_squared();
        let prob_one = amp_one.norm_squared();
        let total_prob = prob_zero + prob_one;

        // Check for spurious amplitudes (parallel reduction)
        let max_spurious = self
            .blocks
            .par_iter()
            .enumerate()
            .map(|(block_id, block)| {
                let mut local_max = 0.0f64;
                for (addr, amp) in block.amplitudes.iter().enumerate() {
                    // Skip the two expected non-zero amplitudes
                    if (block_id == 0 && addr == 0) || (block_id == last_block && addr == 127) {
                        continue;
                    }
                    let norm = amp.norm();
                    if norm > local_max {
                        local_max = norm;
                    }
                }
                local_max
            })
            .reduce(|| 0.0f64, |a, b| a.max(b));

        // Calculate fidelity with ideal GHZ
        // |ψ_ideal⟩ = (|0...0⟩ + |1...1⟩)/√2
        // Fidelity = |⟨ψ_ideal|ψ_actual⟩|²
        let ideal_amp = FRAC_1_SQRT_2;
        // overlap = ideal_amp * (amp_zero.re + amp_one.re) for real amplitudes
        let overlap = amp_zero * ideal_amp + amp_one * ideal_amp;
        let fidelity = overlap.norm_squared();

        GhzVerification {
            amp_zero_state: amp_zero.norm(),
            amp_one_state: amp_one.norm(),
            prob_zero,
            prob_one,
            total_prob,
            fidelity,
            max_spurious,
            active_blocks: self.blocks.len(),
        }
    }

    /// Get probability data for heatmap visualization
    /// Returns (block_probs, max_prob, total_prob)
    /// block_probs[i] = sum of |amplitude|^2 for block i
    pub fn get_block_probabilities(&self) -> (Vec<f64>, f64, f64) {
        let block_probs: Vec<f64> = self
            .blocks
            .par_iter()
            .map(|block| {
                block
                    .amplitudes
                    .iter()
                    .map(|amp| amp.norm_squared())
                    .sum::<f64>()
            })
            .collect();

        let max_prob = block_probs
            .iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        let total_prob = block_probs.iter().sum();

        (block_probs, max_prob, total_prob)
    }

    /// Get top N blocks by probability (for sparse visualization)
    /// Returns Vec<(block_index, probability)> sorted by probability descending
    pub fn get_top_blocks(&self, n: usize) -> Vec<(usize, f64)> {
        let (block_probs, _, _) = self.get_block_probabilities();
        let mut indexed: Vec<(usize, f64)> = block_probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(n);
        indexed
    }

    /// Get amplitude at specific basis state
    /// basis_state is the computational basis index (e.g., 0 for |0...0⟩)
    pub fn get_amplitude(&self, basis_state: usize) -> Complex64 {
        let block_idx = basis_state >> 7; // divide by 128
        let local_idx = basis_state & 127; // mod 128
        if block_idx < self.blocks.len() {
            self.blocks[block_idx].amplitudes[local_idx]
        } else {
            Complex64::zero()
        }
    }

    /// Get probability at specific basis state
    pub fn get_probability(&self, basis_state: usize) -> f64 {
        self.get_amplitude(basis_state).norm_squared()
    }
}

/// GHZ state verification results
#[derive(Debug)]
pub struct GhzVerification {
    pub amp_zero_state: f64,
    pub amp_one_state: f64,
    pub prob_zero: f64,
    pub prob_one: f64,
    pub total_prob: f64,
    pub fidelity: f64,
    pub max_spurious: f64,
    pub active_blocks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_7_qubit_ghz_100_fidelity() {
        let mut grid = QuantumGridF64::new(7);
        let time_us = grid.create_ghz_state();
        let verification = grid.verify_ghz_fidelity();

        println!("\n7-Qubit GHZ State (F64 Precision):");
        println!("  Time: {}μs", time_us);
        println!("  |0000000⟩ amplitude: {:.10}", verification.amp_zero_state);
        println!("  |1111111⟩ amplitude: {:.10}", verification.amp_one_state);
        println!("  Fidelity: {:.10}%", verification.fidelity * 100.0);
        println!("  Max spurious: {:.2e}", verification.max_spurious);

        assert!(
            verification.fidelity > 0.9999999,
            "Fidelity {:.6}% not 100%",
            verification.fidelity * 100.0
        );
    }

    #[test]
    fn test_14_qubit_ghz_100_fidelity() {
        // 14 qubits = 128 blocks
        let mut grid = QuantumGridF64::new(14);
        let time_us = grid.create_ghz_state();
        let verification = grid.verify_ghz_fidelity();

        println!("\n14-Qubit GHZ State (F64 Precision):");
        println!("  Blocks: {}", grid.num_blocks());
        println!("  Time: {}μs", time_us);
        println!(
            "  |00000000000000⟩ amplitude: {:.10}",
            verification.amp_zero_state
        );
        println!(
            "  |11111111111111⟩ amplitude: {:.10}",
            verification.amp_one_state
        );
        println!("  Fidelity: {:.10}%", verification.fidelity * 100.0);
        println!("  Max spurious: {:.2e}", verification.max_spurious);

        assert!(
            verification.fidelity > 0.9999999,
            "Fidelity {:.6}% not 100%",
            verification.fidelity * 100.0
        );
    }

    #[test]
    fn test_20_qubit_ghz_100_fidelity() {
        // 20 qubits = 8,192 blocks
        let mut grid = QuantumGridF64::new(20);
        let time_us = grid.create_ghz_state();
        let verification = grid.verify_ghz_fidelity();

        println!("\n20-Qubit GHZ State (F64 Precision):");
        println!("  Blocks: {}", grid.num_blocks());
        println!("  Time: {:.2}ms", time_us as f64 / 1000.0);
        println!("  Fidelity: {:.10}%", verification.fidelity * 100.0);
        println!("  Max spurious: {:.2e}", verification.max_spurious);

        assert!(
            verification.fidelity > 0.9999999,
            "Fidelity {:.6}% not 100%",
            verification.fidelity * 100.0
        );
    }

    #[test]
    fn test_27_qubit_ghz_100_fidelity() {
        // 27 qubits = 1,048,576 blocks (1M CPUs equivalent)
        let mut grid = QuantumGridF64::new(27);

        println!("\n27-Qubit GHZ State (F64 Precision):");
        println!(
            "  Blocks: {} ({:.2}M)",
            grid.num_blocks(),
            grid.num_blocks() as f64 / 1e6
        );

        let time_us = grid.create_ghz_state();
        let verification = grid.verify_ghz_fidelity();

        println!("  Time: {:.2}ms", time_us as f64 / 1000.0);
        println!("  |0...0⟩ amplitude: {:.15}", verification.amp_zero_state);
        println!("  |1...1⟩ amplitude: {:.15}", verification.amp_one_state);
        println!("  Fidelity: {:.15}%", verification.fidelity * 100.0);
        println!("  Max spurious: {:.2e}", verification.max_spurious);

        // Should be essentially 100% with f64 precision
        assert!(
            verification.fidelity > 0.999999999999,
            "Fidelity {:.10}% not ~100%",
            verification.fidelity * 100.0
        );

        println!("\n  100% FIDELITY ACHIEVED WITH F64 PRECISION!");
    }

    #[test]
    fn compare_fixed8_vs_f64() {
        println!("\n=== Fixed8 vs F64 Precision Comparison ===\n");

        // F64 precision constants
        let f64_inv_sqrt2 = FRAC_1_SQRT_2;
        println!("F64 1/√2 = {:.15}", f64_inv_sqrt2);

        // Fixed8 approximation (0x5A = 90)
        let fixed8_inv_sqrt2 = 90.0 / 128.0;
        println!("Fixed8 1/√2 = {:.15} (90/128)", fixed8_inv_sqrt2);

        let error = (f64_inv_sqrt2 - fixed8_inv_sqrt2).abs();
        let rel_error = error / f64_inv_sqrt2;
        println!("Absolute error: {:.15}", error);
        println!("Relative error: {:.6}%", rel_error * 100.0);

        // GHZ fidelity calculation
        // For GHZ: |amp_0|² + |amp_1|² should = 1
        // With Fixed8: (90/128)² + (90/128)² = 2 * (90/128)² = 0.9844...
        let fixed8_fidelity = 2.0 * fixed8_inv_sqrt2.powi(2);
        let f64_fidelity = 2.0 * f64_inv_sqrt2.powi(2);

        println!("\nGHZ state fidelity:");
        println!("  Fixed8: {:.6}%", fixed8_fidelity * 100.0);
        println!("  F64: {:.15}%", f64_fidelity * 100.0);

        // This explains the ~98.88% fidelity with Fixed8
        println!(
            "\nFixed8 fidelity loss: {:.2}%",
            (1.0 - fixed8_fidelity) * 100.0
        );
    }

    #[test]
    #[ignore] // Requires ~33GB RAM
    fn test_31_qubit_ghz_dense() {
        // 31 qubits = 16,777,216 blocks = 33.5 GB
        println!("\n31-Qubit Dense GHZ State:");
        println!(
            "  Allocating {} blocks ({:.1} GB)...",
            1 << 24,
            (1usize << 24) * 2 / 1024 / 1024
        );

        let mut grid = QuantumGridF64::new(31);
        println!("  Allocation complete. Running GHZ creation...");

        let time_us = grid.create_ghz_state();
        let verification = grid.verify_ghz_fidelity();

        println!("  Time: {:.2}s", time_us as f64 / 1_000_000.0);
        println!("  Fidelity: {:.15}%", verification.fidelity * 100.0);
        println!("  Max spurious: {:.2e}", verification.max_spurious);

        assert!(verification.fidelity > 0.999999999999);
    }

    #[test]
    #[ignore] // Requires ~67GB RAM - will likely fail on most systems
    fn test_32_qubit_ghz_dense() {
        // 32 qubits = 33,554,432 blocks = 67 GB
        println!("\n32-Qubit Dense GHZ State:");
        println!(
            "  Allocating {} blocks ({:.1} GB)...",
            1 << 25,
            (1usize << 25) * 2 / 1024 / 1024
        );

        let mut grid = QuantumGridF64::new(32);
        println!("  Allocation complete. Running GHZ creation...");

        let time_us = grid.create_ghz_state();
        let verification = grid.verify_ghz_fidelity();

        println!("  Time: {:.2}s", time_us as f64 / 1_000_000.0);
        println!("  Fidelity: {:.15}%", verification.fidelity * 100.0);

        assert!(verification.fidelity > 0.999999999999);
    }
}
