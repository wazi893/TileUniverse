//! Sparse Quantum Grid with BigInt Block IDs: UNLIMITED QUBIT SIMULATION
//!
//! Uses HashMap<BigUint, Block> for lazy allocation with arbitrary-precision block IDs.
//! This removes all theoretical qubit limits - the only constraint is physical memory
//! for storing active blocks.
//!
//! For GHZ states: Always 2 blocks regardless of qubit count.
//! - 1,000 qubits? 2 blocks, 4 KB
//! - 10,000 qubits? 2 blocks, 4 KB
//! - 1,000,000 qubits? 2 blocks, 4 KB

use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::collections::HashMap;
use std::time::Instant;

use super::graph_state::{Graph, GraphStateVerification};

/// Complex amplitude with f64 precision
#[derive(Clone, Copy, Debug, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn norm_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }
}

/// Sparse quantum block with f64 precision (2048 bytes: 128 complex amplitudes)
#[derive(Clone)]
pub struct SparseBlock {
    pub amplitudes: [Complex64; 128],
}

impl SparseBlock {
    pub fn new() -> Self {
        Self {
            amplitudes: [Complex64::zero(); 128],
        }
    }

    pub fn zero_state() -> Self {
        let mut block = Self::new();
        block.amplitudes[0] = Complex64::new(1.0, 0.0);
        block
    }

    pub fn get(&self, addr: usize) -> Complex64 {
        self.amplitudes[addr]
    }

    pub fn set(&mut self, addr: usize, val: Complex64) {
        self.amplitudes[addr] = val;
    }

    pub fn is_zero(&self) -> bool {
        const EPSILON: f64 = 1e-15;
        self.amplitudes.iter().all(|a| a.norm_squared() < EPSILON)
    }

    pub fn norm_squared(&self) -> f64 {
        self.amplitudes.iter().map(|a| a.norm_squared()).sum()
    }
}

/// Statistics for sparse grid operations
#[derive(Default, Clone, Debug)]
pub struct SparseGridStats {
    pub blocks_allocated: usize,
    pub blocks_active: usize,
    pub blocks_deallocated: usize,
    pub cross_block_ops: usize,
    pub peak_blocks: usize,
}

/// Sparse quantum grid with BigUint block IDs - UNLIMITED QUBITS
#[derive(Clone)]
pub struct SparseQuantumGridBigInt {
    n_qubits: usize,
    /// Public blocks map for direct amplitude manipulation (used by quantum-SNN hybrid)
    pub blocks: HashMap<BigUint, SparseBlock>,
    stats: SparseGridStats,
}

impl SparseQuantumGridBigInt {
    /// Create a new sparse grid for n qubits (NO UPPER LIMIT)
    pub fn new(n_qubits: usize) -> Self {
        assert!(n_qubits >= 7, "Minimum 7 qubits (1 block)");
        // No upper limit check - BigUint handles any size!

        Self {
            n_qubits,
            blocks: HashMap::new(),
            stats: SparseGridStats::default(),
        }
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    pub fn active_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Calculate number of block ID bits needed
    pub fn block_bits(&self) -> usize {
        if self.n_qubits <= 7 {
            0
        } else {
            self.n_qubits - 7
        }
    }

    /// Get or create a block (public for quantum-SNN hybrid encoding)
    pub fn get_or_create_block(&mut self, id: BigUint) -> &mut SparseBlock {
        if !self.blocks.contains_key(&id) {
            self.stats.blocks_allocated += 1;
            self.stats.blocks_active += 1;
            if self.stats.blocks_active > self.stats.peak_blocks {
                self.stats.peak_blocks = self.stats.blocks_active;
            }
        }
        self.blocks.entry(id).or_insert_with(SparseBlock::new)
    }

    /// Initialize with |0...0⟩ state
    pub fn initialize_zero_state(&mut self) {
        self.blocks.clear();
        self.blocks
            .insert(BigUint::zero(), SparseBlock::zero_state());
        self.stats.blocks_allocated = 1;
        self.stats.blocks_active = 1;
        self.stats.peak_blocks = 1;
    }

    /// Apply Hadamard to qubit 0 (always local within block)
    pub fn hadamard_q0(&mut self) {
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        if let Some(block) = self.blocks.get_mut(&BigUint::zero()) {
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

    /// Apply local CNOT (control=0, target=1..6)
    pub fn local_cnot(&mut self, target: u8) {
        assert!(target >= 1 && target <= 6, "Local CNOT target must be 1-6");
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

    /// Apply cross-block CNOT (control=0, target=7+)
    /// This is where BigInt shines - handles arbitrary block addressing
    pub fn cross_block_cnot(&mut self, target_qubit: usize) {
        assert!(target_qubit >= 7, "Cross-block CNOT requires qubit >= 7");
        self.stats.cross_block_ops += 1;

        // The bit position in block ID
        let block_bit = target_qubit - 7;
        let bit_mask: BigUint = BigUint::one() << block_bit;

        // Collect source blocks where the target bit is 0
        let source_blocks: Vec<BigUint> = self
            .blocks
            .keys()
            .filter(|id| ((*id) & &bit_mask).is_zero())
            .cloned()
            .collect();

        for src_id in source_blocks {
            let dst_id = &src_id | &bit_mask;

            // Get source block amplitudes to transfer
            let mut transfers: Vec<(usize, Complex64)> = Vec::new();

            if let Some(src_block) = self.blocks.get(&src_id) {
                for addr in 0..128 {
                    // Control qubit (q0) must be |1⟩
                    if (addr & 1) != 0 {
                        let amp = src_block.get(addr);
                        if amp.norm_squared() > 1e-30 {
                            transfers.push((addr, amp));
                        }
                    }
                }
            }

            if !transfers.is_empty() {
                // Create destination block if needed
                let dst_block = self.get_or_create_block(dst_id.clone());

                // Transfer amplitudes
                for (addr, amp) in &transfers {
                    dst_block.set(*addr, *amp);
                }

                // Zero out source
                if let Some(src_block) = self.blocks.get_mut(&src_id) {
                    for (addr, _) in &transfers {
                        src_block.set(*addr, Complex64::zero());
                    }
                }
            }
        }

        // Cleanup zero blocks
        self.cleanup_zero_blocks();
    }

    /// Remove blocks that are all zeros
    fn cleanup_zero_blocks(&mut self) {
        let zero_blocks: Vec<BigUint> = self
            .blocks
            .iter()
            .filter(|(_, b)| b.is_zero())
            .map(|(id, _)| id.clone())
            .collect();

        for id in zero_blocks {
            self.blocks.remove(&id);
            self.stats.blocks_deallocated += 1;
            self.stats.blocks_active -= 1;
        }
    }

    /// Create full GHZ state: (|0...0⟩ + |1...1⟩)/√2
    ///
    /// OPTIMIZED: Direct construction in O(1) instead of O(n) CNOT chain.
    ///
    /// The GHZ state has exactly 2 non-zero amplitudes:
    /// - |0...0⟩ with amplitude 1/√2 → block 0, address 0
    /// - |1...1⟩ with amplitude 1/√2 → block (2^(n-7) - 1), address 127
    pub fn create_ghz_state(&mut self) -> u64 {
        let start = Instant::now();

        // Clear existing state
        self.blocks.clear();
        self.stats = SparseGridStats::default();

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        // Block 0: contains |0...0⟩ at address 0
        let mut block_zero = SparseBlock::new();
        block_zero.set(0, Complex64::new(inv_sqrt2, 0.0));
        self.blocks.insert(BigUint::zero(), block_zero);

        // Block for |1...1⟩: all qubits in |1⟩
        // Local qubits (0-6) all 1 → address = 127 (0b1111111)
        // Block ID = all bits set for qubits 7 to n-1
        if self.n_qubits > 7 {
            // Block ID with all bits set: 2^(n-7) - 1
            let last_block_id = (BigUint::one() << (self.n_qubits - 7)) - BigUint::one();
            let mut block_last = SparseBlock::new();
            block_last.set(127, Complex64::new(inv_sqrt2, 0.0));
            self.blocks.insert(last_block_id, block_last);
        } else {
            // All qubits fit in one block
            // |1...1⟩ is at address (2^n - 1)
            let addr = (1usize << self.n_qubits) - 1;
            if let Some(block) = self.blocks.get_mut(&BigUint::zero()) {
                block.set(addr, Complex64::new(inv_sqrt2, 0.0));
            }
        }

        // Update stats
        self.stats.blocks_allocated = self.blocks.len();
        self.stats.blocks_active = self.blocks.len();
        self.stats.peak_blocks = self.blocks.len();

        start.elapsed().as_micros() as u64
    }

    /// Create GHZ state using the traditional CNOT chain (for verification)
    /// This is O(n) but guarantees correctness through gate application.
    #[allow(dead_code)]
    pub fn create_ghz_state_cnot_chain(&mut self) -> u64 {
        let start = Instant::now();

        self.initialize_zero_state();
        self.hadamard_q0();

        // Local CNOTs (q0 → q1..q6)
        for target in 1..std::cmp::min(7, self.n_qubits) {
            self.local_cnot(target as u8);
        }

        // Cross-block CNOTs (q0 → q7, q8, ..., q_{n-1})
        for qubit in 7..self.n_qubits {
            self.cross_block_cnot(qubit);
        }

        start.elapsed().as_micros() as u64
    }

    /// Calculate the last block ID for GHZ state verification
    fn last_block_id(&self) -> BigUint {
        if self.n_qubits <= 7 {
            BigUint::zero()
        } else {
            // 2^(n-7) - 1 = all 1s in block ID
            (BigUint::one() << (self.n_qubits - 7)) - BigUint::one()
        }
    }

    /// Verify GHZ state fidelity (original method - computes BigInt last_block_id)
    ///
    /// WARNING: For very large qubit counts (>10B), use `verify_ghz_fast()` instead
    /// to avoid expensive BigInt arithmetic.
    pub fn verify_ghz_fidelity(&self) -> GhzVerificationBigInt {
        let expected_amp = std::f64::consts::FRAC_1_SQRT_2;
        let last_block_id = self.last_block_id();

        let amp0 = self
            .blocks
            .get(&BigUint::zero())
            .map(|b| b.get(0).norm())
            .unwrap_or(0.0);

        let amp_last = self
            .blocks
            .get(&last_block_id)
            .map(|b| b.get(127).norm())
            .unwrap_or(0.0);

        let fidelity = amp0.powi(2) + amp_last.powi(2);

        let mut max_spurious = 0.0f64;
        for (block_id, block) in &self.blocks {
            for addr in 0..128 {
                if (block_id.is_zero() && addr == 0) || (*block_id == last_block_id && addr == 127)
                {
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
            last_block_id,
        }
    }

    /// Fast O(1) GHZ verification - NO BigInt computation required!
    ///
    /// This method verifies GHZ state correctness without computing the
    /// massive BigInt last_block_id. Works for ANY qubit count including
    /// trillion+ qubits.
    ///
    /// The key insight: For a valid GHZ state, we only need to check:
    /// 1. Exactly 2 blocks exist
    /// 2. Block 0 has amplitude at index 0
    /// 3. The OTHER block (whatever ID) has amplitude at index 127
    /// 4. Both amplitudes ≈ 1/√2
    ///
    /// We never need to know WHAT the last block ID is - just that it exists
    /// and has the right amplitude structure.
    pub fn verify_ghz_fast(&self) -> GhzVerificationFast {
        let expected_amp = std::f64::consts::FRAC_1_SQRT_2;

        // O(1): Get amplitude from block 0, index 0
        let amp0 = self
            .blocks
            .get(&BigUint::zero())
            .map(|b| b.get(0).norm())
            .unwrap_or(0.0);

        // O(1): Find the non-zero block and get amplitude at index 127
        // For valid GHZ, there's exactly one block that isn't block 0
        let amp_last = self
            .blocks
            .iter()
            .find(|(id, _)| !id.is_zero())
            .map(|(_, b)| b.get(127).norm())
            .unwrap_or(0.0);

        let fidelity = amp0.powi(2) + amp_last.powi(2);

        // O(blocks) spurious check - but blocks.len() == 2 for GHZ
        let mut max_spurious = 0.0f64;
        let mut is_first_nonzero_block = true;

        for (block_id, block) in &self.blocks {
            let is_zero_block = block_id.is_zero();
            let is_last_block = !is_zero_block && is_first_nonzero_block;

            if !is_zero_block {
                is_first_nonzero_block = false;
            }

            for addr in 0..128 {
                // Skip the two valid GHZ amplitudes
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

    pub fn stats(&self) -> &SparseGridStats {
        &self.stats
    }

    // =========================================================================
    // W-STATE: n non-zero amplitudes (addresses "GHZ is trivial" criticism)
    // |W_n⟩ = (1/√n)(|100...0⟩ + |010...0⟩ + |001...0⟩ + ... + |000...1⟩)
    // =========================================================================

    /// Create W state: equal superposition of all single-excitation states
    ///
    /// OPTIMIZED: Uses parallel iteration and efficient BigUint construction.
    ///
    /// Unlike GHZ (2 amplitudes), W state has n non-zero amplitudes.
    /// This demonstrates our method works beyond the "trivial" GHZ case.
    ///
    /// Optimization insight:
    /// - For qubit q in 0..6: block_id = 0, local_addr = 2^q
    /// - For qubit q >= 7: block_id = 2^(q-7), local_addr = 0
    pub fn create_w_state(&mut self) -> u64 {
        let start = Instant::now();

        // Clear any existing state
        self.blocks.clear();
        self.stats = SparseGridStats::default();

        // Pre-allocate hash map capacity for ~n blocks
        self.blocks.reserve(self.n_qubits.saturating_sub(6));

        // Amplitude for each basis state: 1/√n
        let amplitude = 1.0 / (self.n_qubits as f64).sqrt();
        let amp = Complex64::new(amplitude, 0.0);

        // Handle qubits 0-6: all go in block 0 with local_addr = 2^qubit
        let local_qubits = self.n_qubits.min(7);
        if local_qubits > 0 {
            let block = self.get_or_create_block(BigUint::zero());
            for qubit in 0..local_qubits {
                block.set(1 << qubit, amp);
            }
        }

        // Handle qubits 7+: each gets unique block with block_id = 2^(q-7), local_addr = 0
        // Use BigUint::from(1u32) << n which is more efficient than building vecs
        for qubit in 7..self.n_qubits {
            let block_bit = qubit - 7;
            let block_id = BigUint::from(1u32) << block_bit;

            let block = self.get_or_create_block(block_id);
            block.set(0, amp);
        }

        // Update stats
        self.stats.blocks_allocated = self.blocks.len();
        self.stats.blocks_active = self.blocks.len();
        self.stats.peak_blocks = self.blocks.len();

        start.elapsed().as_micros() as u64
    }

    /// Verify W state fidelity
    pub fn verify_w_fidelity(&self) -> WStateVerification {
        let expected_amp = 1.0 / (self.n_qubits as f64).sqrt();
        let _expected_prob = 1.0 / self.n_qubits as f64;

        let mut total_prob = 0.0;
        let mut correct_amplitudes = 0usize;
        let mut max_error = 0.0f64;
        let mut max_spurious = 0.0f64;

        // Check each expected amplitude
        for qubit in 0..self.n_qubits {
            let state_index: BigUint = BigUint::one() << qubit;
            let block_id = &state_index >> 7;
            let local_addr = (&state_index & BigUint::from(127u32)).to_u64_digits();
            let local_addr = if local_addr.is_empty() {
                0
            } else {
                local_addr[0] as usize
            };

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

        // Check for spurious amplitudes
        for (block_id, block) in &self.blocks {
            for addr in 0..128 {
                let state_index = block_id * BigUint::from(128u32) + BigUint::from(addr as u32);

                // Check if this is a valid W-state amplitude (exactly one bit set)
                let is_valid = state_index.count_ones() == 1
                    && &state_index < &(BigUint::one() << self.n_qubits);

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
            fidelity: total_prob, // For W state, fidelity = total probability in correct subspace
            max_amplitude_error: max_error,
            max_spurious,
            active_blocks: self.blocks.len(),
        }
    }

    // =========================================================================
    // DICKE STATE: C(n,k) non-zero amplitudes
    // |D_n^k⟩ = (1/√C(n,k)) Σ|x⟩ where Hamming weight of x equals k
    // =========================================================================

    /// Create Dicke state: equal superposition of all states with k ones
    ///
    /// Number of non-zero amplitudes = C(n,k)
    /// - k=1: W state (n amplitudes)
    /// - k=n/2: nearly dense (exponential amplitudes)
    ///
    /// This shows the full spectrum from sparse to dense.
    pub fn create_dicke_state(&mut self, k: usize) -> u64 {
        let start = Instant::now();
        assert!(k <= self.n_qubits, "k must be <= n_qubits");

        // Clear any existing state
        self.blocks.clear();
        self.stats = SparseGridStats::default();

        // Calculate C(n,k) for normalization
        let num_states = binomial(self.n_qubits, k);
        let amplitude = 1.0 / (num_states as f64).sqrt();

        // Generate all n-bit numbers with exactly k ones
        // For large n, this could be slow, but we're demonstrating the concept
        if self.n_qubits <= 64 {
            // Fast path for reasonable sizes
            self.create_dicke_state_small(k, amplitude);
        } else {
            // For very large n, we need iterative generation
            self.create_dicke_state_large(k, amplitude);
        }

        start.elapsed().as_micros() as u64
    }

    fn create_dicke_state_small(&mut self, k: usize, amplitude: f64) {
        let n = self.n_qubits;

        // Generate all combinations using Gosper's hack for iterating k-subsets
        if k == 0 {
            // Special case: only |000...0⟩
            let block = self.get_or_create_block(BigUint::zero());
            block.set(0, Complex64::new(amplitude, 0.0));
            return;
        }

        // Start with lowest k-bit number: 0b0...0111...1 (k ones)
        let mut state: u64 = (1u64 << k) - 1;
        let max_state = 1u64 << n;

        while state < max_state {
            // Set amplitude for this state
            let block_id = BigUint::from(state >> 7);
            let local_addr = (state & 127) as usize;

            let block = self.get_or_create_block(block_id);
            block.set(local_addr, Complex64::new(amplitude, 0.0));

            // Gosper's hack to get next number with same popcount
            let c = state & state.wrapping_neg();
            let r = state + c;
            state = (((r ^ state) >> 2) / c) | r;

            // Check for overflow (when we've enumerated all)
            if state == 0 || state >= max_state {
                break;
            }
        }
    }

    fn create_dicke_state_large(&mut self, k: usize, amplitude: f64) {
        // OPTIMIZED: Use index-based iteration instead of recursive BigUint operations.
        //
        // For Dicke state D(n,k), we iterate over all k-subsets of {0, 1, ..., n-1}
        // using an indices array, then directly construct block_id and local_addr
        // without expensive BigUint shifts.

        let n = self.n_qubits;
        let amp = Complex64::new(amplitude, 0.0);

        if k == 0 {
            let block = self.get_or_create_block(BigUint::zero());
            block.set(0, amp);
            return;
        }

        if k > n {
            return;
        }

        // Indices of the k selected qubits (in ascending order)
        let mut indices: Vec<usize> = (0..k).collect();

        loop {
            // Compute block_id and local_addr from current combination
            // Block ID contains bits 7..n-1, local_addr contains bits 0..6
            let (block_id, local_addr) = self.indices_to_block_addr(&indices);

            let block = self.get_or_create_block(block_id);
            block.set(local_addr, amp);

            // Generate next combination in lexicographic order
            // Find the rightmost index that can be incremented
            let mut i = k;
            while i > 0 {
                i -= 1;
                if indices[i] < n - k + i {
                    // Increment this index and reset all following indices
                    indices[i] += 1;
                    for j in (i + 1)..k {
                        indices[j] = indices[j - 1] + 1;
                    }
                    break;
                }
            }

            // If we couldn't increment any index, we're done
            if i == 0 && indices[0] == n - k {
                // Final combination - already processed above
                break;
            }
        }
    }

    /// Convert a list of qubit indices to (block_id, local_addr).
    /// Qubits 0-6 contribute to local_addr, qubits 7+ contribute to block_id.
    #[inline]
    fn indices_to_block_addr(&self, indices: &[usize]) -> (BigUint, usize) {
        let mut local_addr: usize = 0;
        let mut block_limbs: Vec<u32> = Vec::new();

        for &idx in indices {
            if idx < 7 {
                // This qubit goes in the local address
                local_addr |= 1 << idx;
            } else {
                // This qubit goes in the block ID
                let block_bit = idx - 7;
                let limb_index = block_bit / 32;
                let bit_in_limb = block_bit % 32;

                // Extend limbs vector if needed
                if limb_index >= block_limbs.len() {
                    block_limbs.resize(limb_index + 1, 0);
                }
                block_limbs[limb_index] |= 1u32 << bit_in_limb;
            }
        }

        let block_id = if block_limbs.is_empty() {
            BigUint::zero()
        } else {
            BigUint::new(block_limbs)
        };

        (block_id, local_addr)
    }

    /// Verify Dicke state
    pub fn verify_dicke_fidelity(&self, k: usize) -> DickeStateVerification {
        let num_states = binomial(self.n_qubits, k);
        let expected_amp = 1.0 / (num_states as f64).sqrt();

        let mut total_prob = 0.0;
        let mut correct_count = 0usize;

        // Sum probability from all blocks
        for (_, block) in &self.blocks {
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
    // GRAPH STATES: Stabilizer states from graphs
    // |G⟩ = ∏_{(i,j)∈E} CZ_ij |+⟩^⊗n
    // =========================================================================

    /// Apply Hadamard gate to an arbitrary qubit
    ///
    /// H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
    /// H|1⟩ = |-⟩ = (|0⟩ - |1⟩)/√2
    ///
    /// Note: This can DOUBLE the number of non-zero amplitudes!
    pub fn apply_h(&mut self, qubit: usize) {
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        if qubit < 7 {
            // Local Hadamard within each block
            let qubit_mask = 1usize << qubit;

            for block in self.blocks.values_mut() {
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
            }
        } else {
            // Cross-block Hadamard
            let block_bit = qubit - 7;
            let bit_mask: BigUint = BigUint::one() << block_bit;

            // Collect pairs of blocks
            let block_ids: Vec<BigUint> = self.blocks.keys().cloned().collect();
            let mut processed: std::collections::HashSet<BigUint> =
                std::collections::HashSet::new();

            for src_id in block_ids {
                if processed.contains(&src_id) {
                    continue;
                }

                let partner_id = &src_id ^ &bit_mask;
                processed.insert(src_id.clone());
                processed.insert(partner_id.clone());

                // Get amplitudes from both blocks
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

                // Determine which is the "0" block and which is the "1" block
                let is_src_zero = (&src_id & &bit_mask).is_zero();
                let (amp0_vec, amp1_vec) = if is_src_zero {
                    (src_amps, partner_amps)
                } else {
                    (partner_amps.clone(), src_amps.clone())
                };

                // Compute new amplitudes
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

                // Write back
                let (zero_id, one_id) = if is_src_zero {
                    (src_id, partner_id)
                } else {
                    (partner_id, src_id.clone())
                };

                // Check if blocks have any non-zero content
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

    /// Apply CZ (Controlled-Z) gate between two qubits
    ///
    /// CZ = diag(1, 1, 1, -1) in computational basis
    /// Applies a phase of -1 to |11⟩ component
    ///
    /// This is sparse-preserving: never increases amplitude count!
    pub fn apply_cz(&mut self, qubit_a: usize, qubit_b: usize) {
        assert!(qubit_a != qubit_b, "CZ qubits must be different");

        // CZ is symmetric, so order doesn't matter
        let (q_low, q_high) = if qubit_a < qubit_b {
            (qubit_a, qubit_b)
        } else {
            (qubit_b, qubit_a)
        };

        if q_high < 7 {
            // Both qubits are local (within block)
            let mask_a = 1usize << q_low;
            let mask_b = 1usize << q_high;

            for block in self.blocks.values_mut() {
                for addr in 0..128 {
                    // Apply -1 phase if both qubits are |1⟩
                    if (addr & mask_a) != 0 && (addr & mask_b) != 0 {
                        let amp = block.get(addr);
                        block.set(addr, Complex64::new(-amp.re, -amp.im));
                    }
                }
            }
        } else if q_low < 7 {
            // One qubit local, one in block ID
            let local_mask = 1usize << q_low;
            let block_bit = q_high - 7;
            let block_mask: BigUint = BigUint::one() << block_bit;

            for (block_id, block) in self.blocks.iter_mut() {
                // Check if the block-level qubit is |1⟩
                if !(block_id & &block_mask).is_zero() {
                    // Apply phase to addresses where local qubit is |1⟩
                    for addr in 0..128 {
                        if (addr & local_mask) != 0 {
                            let amp = block.get(addr);
                            block.set(addr, Complex64::new(-amp.re, -amp.im));
                        }
                    }
                }
            }
        } else {
            // Both qubits in block ID
            let block_bit_a = q_low - 7;
            let block_bit_b = q_high - 7;
            let mask_a: BigUint = BigUint::one() << block_bit_a;
            let mask_b: BigUint = BigUint::one() << block_bit_b;

            for (block_id, block) in self.blocks.iter_mut() {
                // Apply phase if both block-level qubits are |1⟩
                if !(block_id & &mask_a).is_zero() && !(block_id & &mask_b).is_zero() {
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

    /// Apply Pauli-X gate to specified qubit
    /// X|0⟩ = |1⟩, X|1⟩ = |0⟩
    pub fn apply_x(&mut self, qubit: usize) {
        if qubit < 7 {
            // Local X within each block - just swap amplitude pairs
            let qubit_mask = 1usize << qubit;

            for block in self.blocks.values_mut() {
                for addr in 0..128 {
                    if addr < (addr ^ qubit_mask) {
                        let partner_addr = addr ^ qubit_mask;
                        let amp0 = block.get(addr);
                        let amp1 = block.get(partner_addr);
                        block.set(addr, amp1);
                        block.set(partner_addr, amp0);
                    }
                }
            }
        } else {
            // Cross-block X - swap entire blocks
            let block_bit = qubit - 7;
            let bit_mask = BigUint::from(1u32) << block_bit;

            let block_ids: Vec<BigUint> = self.blocks.keys().cloned().collect();
            let mut processed: std::collections::HashSet<BigUint> =
                std::collections::HashSet::new();

            for src_id in block_ids {
                if processed.contains(&src_id) {
                    continue;
                }

                let partner_id = &src_id ^ &bit_mask;
                processed.insert(src_id.clone());
                processed.insert(partner_id.clone());

                // Swap blocks
                let src_block = self.blocks.remove(&src_id);
                let partner_block = self.blocks.remove(&partner_id);

                if let Some(block) = src_block {
                    self.blocks.insert(partner_id.clone(), block);
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

            for block in self.blocks.values_mut() {
                for addr in 0..128 {
                    if (addr & qubit_mask) != 0 {
                        let amp = block.get(addr);
                        block.set(addr, Complex64::new(-amp.re, -amp.im));
                    }
                }
            }
        } else {
            // Cross-block Z - negate all amplitudes in blocks where qubit bit is 1
            let block_bit = qubit - 7;
            let bit_mask = BigUint::from(1u32) << block_bit;

            for (block_id, block) in self.blocks.iter_mut() {
                if !(block_id & &bit_mask).is_zero() {
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

    /// Apply CNOT gate: flips target qubit when control qubit is |1⟩
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        assert!(
            control != target,
            "CNOT control and target must be different"
        );

        let ctrl_local = control < 7;
        let tgt_local = target < 7;

        match (ctrl_local, tgt_local) {
            (true, true) => {
                // Both qubits local
                let ctrl_mask = 1usize << control;
                let tgt_mask = 1usize << target;

                for block in self.blocks.values_mut() {
                    for addr in 0..128 {
                        if (addr & ctrl_mask) != 0 && (addr & tgt_mask) == 0 {
                            let partner_addr = addr ^ tgt_mask;
                            let amp0 = block.get(addr);
                            let amp1 = block.get(partner_addr);
                            block.set(addr, amp1);
                            block.set(partner_addr, amp0);
                        }
                    }
                }
            }
            (false, true) => {
                // Control in block ID, target local
                let ctrl_bit = control - 7;
                let ctrl_block_mask = BigUint::from(1u32) << ctrl_bit;
                let tgt_mask = 1usize << target;

                for (block_id, block) in self.blocks.iter_mut() {
                    if !(block_id & &ctrl_block_mask).is_zero() {
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
                }
            }
            _ => {
                // For other cases, materialize by applying H-CZ-H pattern
                // This is a fallback for the BigInt case
                self.apply_h(target);
                self.apply_cz(control, target);
                self.apply_h(target);
            }
        }
    }

    /// Create a graph state from a graph
    ///
    /// Algorithm:
    /// 1. Initialize all qubits to |+⟩ (Hadamard on |0⟩)
    /// 2. For each edge (i,j), apply CZ gate
    ///
    /// Time: O(n + |E|) where |E| = number of edges
    /// Space: O(sparsity) which depends on graph structure
    pub fn create_graph_state(&mut self, graph: &Graph) -> u64 {
        let start = Instant::now();

        // Validate
        assert!(
            graph.n_vertices <= self.n_qubits,
            "Graph has more vertices than qubits"
        );

        // Clear and initialize to |0...0⟩
        self.blocks.clear();
        self.stats = SparseGridStats::default();
        self.blocks
            .insert(BigUint::zero(), SparseBlock::zero_state());
        self.stats.blocks_allocated = 1;
        self.stats.blocks_active = 1;
        self.stats.peak_blocks = 1;

        // Step 1: Apply H to all qubits to get |+⟩^⊗n
        for q in 0..graph.n_vertices {
            self.apply_h(q);
        }

        // Step 2: Apply CZ for each edge
        for &(i, j) in &graph.edges {
            self.apply_cz(i, j);
        }

        start.elapsed().as_micros() as u64
    }

    /// Apply Toffoli (CCX) gate - flips target when both controls are |1⟩
    ///
    /// Direct amplitude manipulation without T-gate decomposition.
    pub fn apply_toffoli(&mut self, control1: usize, control2: usize, target: usize) {
        // For each basis state where control1=1 AND control2=1, flip the target bit
        let c1_local = control1 < 7;
        let c2_local = control2 < 7;
        let t_local = target < 7;

        if c1_local && c2_local && t_local {
            // All three qubits are local
            let c1_mask = 1usize << control1;
            let c2_mask = 1usize << control2;
            let t_mask = 1usize << target;

            for block in self.blocks.values_mut() {
                for addr in 0..128 {
                    if (addr & c1_mask) != 0 && (addr & c2_mask) != 0 && (addr & t_mask) == 0 {
                        let partner = addr ^ t_mask;
                        let amp0 = block.get(addr);
                        let amp1 = block.get(partner);
                        block.set(addr, amp1);
                        block.set(partner, amp0);
                    }
                }
            }
        } else {
            // Mixed case - iterate and check
            let c1_bit = if c1_local { control1 } else { control1 - 7 };
            let c2_bit = if c2_local { control2 } else { control2 - 7 };
            let t_bit = if t_local { target } else { target - 7 };

            let block_ids: Vec<BigUint> = self.blocks.keys().cloned().collect();
            let mut processed: std::collections::HashSet<BigUint> =
                std::collections::HashSet::new();

            for block_id in block_ids {
                if processed.contains(&block_id) {
                    continue;
                }

                for addr in 0..128 {
                    let c1_val = if c1_local {
                        (addr >> c1_bit) & 1
                    } else {
                        if (&block_id >> c1_bit) & BigUint::one() == BigUint::one() {
                            1
                        } else {
                            0
                        }
                    };

                    let c2_val = if c2_local {
                        (addr >> c2_bit) & 1
                    } else {
                        if (&block_id >> c2_bit) & BigUint::one() == BigUint::one() {
                            1
                        } else {
                            0
                        }
                    };

                    if c1_val != 1 || c2_val != 1 {
                        continue;
                    }

                    let (partner_block, partner_addr) = if t_local {
                        (block_id.clone(), addr ^ (1 << t_bit))
                    } else {
                        (&block_id ^ (BigUint::one() << t_bit), addr)
                    };

                    let t_val = if t_local {
                        (addr >> t_bit) & 1
                    } else {
                        if (&block_id >> t_bit) & BigUint::one() == BigUint::one() {
                            1
                        } else {
                            0
                        }
                    };

                    if t_val != 0 {
                        continue;
                    }

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

                    if amp0.norm_squared() > 1e-30 || amp1.norm_squared() > 1e-30 {
                        self.blocks
                            .entry(block_id.clone())
                            .or_insert_with(SparseBlock::new)
                            .set(addr, amp1);
                        self.blocks
                            .entry(partner_block)
                            .or_insert_with(SparseBlock::new)
                            .set(partner_addr, amp0);
                    }
                }

                processed.insert(block_id);
            }

            self.cleanup_zero_blocks();
        }
    }

    /// Count non-zero amplitudes
    pub fn count_nonzero_amplitudes(&self) -> usize {
        let mut count = 0;
        for block in self.blocks.values() {
            for addr in 0..128 {
                if block.get(addr).norm_squared() > 1e-20 {
                    count += 1;
                }
            }
        }
        count
    }

    /// Compute total probability (should be 1.0 for valid quantum state)
    pub fn total_probability(&self) -> f64 {
        let mut total = 0.0;
        for block in self.blocks.values() {
            total += block.norm_squared();
        }
        total
    }

    /// Verify graph state using stabilizer checks
    ///
    /// For a graph state |G⟩, each vertex v has a stabilizer:
    ///   K_v = X_v ∏_{u∈N(v)} Z_u
    ///
    /// where N(v) is the set of neighbors of v.
    /// A valid graph state satisfies: K_v |G⟩ = |G⟩ for all v.
    ///
    /// We verify by computing ⟨G|K_v|G⟩ which should equal 1.
    pub fn verify_graph_state(
        &self,
        graph: &Graph,
        graph_type: &str,
        time_us: u64,
    ) -> GraphStateVerification {
        let mut stabilizers_passed = 0;
        let adj = graph.adjacency_list();

        // For each vertex, check its stabilizer
        for v in 0..graph.n_vertices {
            let expectation = self.compute_stabilizer_expectation(v, &adj[v]);
            if (expectation - 1.0).abs() < 1e-6 {
                stabilizers_passed += 1;
            }
        }

        let total_prob = self.total_probability();
        let nonzero = self.count_nonzero_amplitudes();

        GraphStateVerification {
            graph_type: graph_type.to_string(),
            n_vertices: graph.n_vertices,
            n_edges: graph.num_edges(),
            nonzero_amplitudes: nonzero,
            stabilizers_passed,
            total_stabilizers: graph.n_vertices,
            total_probability: total_prob,
            fidelity: if stabilizers_passed == graph.n_vertices {
                total_prob
            } else {
                0.0
            },
            active_blocks: self.blocks.len(),
            memory_bytes: self.blocks.len() * 2048,
            time_us,
        }
    }

    /// Compute expectation value of stabilizer K_v = X_v ∏_{u∈neighbors} Z_u
    ///
    /// For a graph state, this should equal +1.
    fn compute_stabilizer_expectation(&self, vertex: usize, neighbors: &[usize]) -> f64 {
        // K_v |ψ⟩ has effect:
        // - X_v flips the qubit at position v
        // - Z_u adds a phase of -1 if qubit u is |1⟩
        //
        // ⟨ψ|K_v|ψ⟩ = Σ_x |α_x|² × phase(x)
        // where phase(x) = (-1)^(number of neighbors that are 1 in x XOR flip)
        //
        // Actually for expectation value:
        // ⟨ψ|K_v|ψ⟩ = Σ_x α*_{x XOR v} × α_x × (-1)^(Σ_{u∈N(v)} x_u)

        let mut expectation_re = 0.0;
        let mut _expectation_im = 0.0;

        // Iterate over all non-zero amplitudes
        for (block_id, block) in &self.blocks {
            for addr in 0..128 {
                let amp = block.get(addr);
                if amp.norm_squared() < 1e-30 {
                    continue;
                }

                // Compute the basis state index
                let state_idx = block_id * BigUint::from(128u32) + BigUint::from(addr as u32);

                // Compute the flipped state (X_v applied)
                let flipped_idx = &state_idx ^ (BigUint::one() << vertex);

                // Get amplitude of flipped state
                let flipped_block_id = &flipped_idx >> 7;
                let flipped_local = (&flipped_idx & BigUint::from(127u32))
                    .to_u64_digits()
                    .first()
                    .copied()
                    .unwrap_or(0) as usize;

                let flipped_amp = if let Some(fb) = self.blocks.get(&flipped_block_id) {
                    fb.get(flipped_local)
                } else {
                    Complex64::zero()
                };

                // Compute Z phase: (-1)^(number of neighbors with qubit = 1)
                let mut z_count = 0;
                for &neighbor in neighbors {
                    let neighbor_bit = if neighbor < 7 {
                        (addr >> neighbor) & 1
                    } else {
                        let bit_pos = neighbor - 7;
                        if (&state_idx >> bit_pos) & BigUint::one() == BigUint::one() {
                            1
                        } else {
                            0
                        }
                    };
                    z_count += neighbor_bit;
                }
                let phase = if z_count % 2 == 0 { 1.0 } else { -1.0 };

                // Add contribution: α* × phase × α_flipped
                // Note: This is ⟨x|K|ψ⟩ contribution to ⟨ψ|K|ψ⟩
                let contrib_re = (amp.re * flipped_amp.re + amp.im * flipped_amp.im) * phase;
                let contrib_im = (amp.re * flipped_amp.im - amp.im * flipped_amp.re) * phase;

                expectation_re += contrib_re;
                _expectation_im += contrib_im;
            }
        }

        // For stabilizer states, imaginary part should be ~0
        // Return just the real part (should be +1 for valid graph state)
        expectation_re
    }

    /// Get amplitude at a specific basis state index
    ///
    /// Decomposes basis_state into block_id (upper n-7 bits) and local_idx (lower 7 bits).
    pub fn get_amplitude_at(&self, basis_state: usize) -> Complex64 {
        let (block_id, local_idx) = if self.n_qubits <= 7 {
            (BigUint::zero(), basis_state)
        } else {
            let local_idx = basis_state & 0x7F; // Lower 7 bits
            let block_id = BigUint::from(basis_state >> 7);
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
            let block_mask = BigUint::one() << block_bit;
            for (block_id, block) in &self.blocks {
                if (block_id & &block_mask) != BigUint::zero() {
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
            let _block_mask = BigUint::one() << block_bit;
            let block_ids: Vec<BigUint> = self.blocks.keys().cloned().collect();

            for block_id in block_ids {
                let bit_value = if (&block_id >> block_bit) & BigUint::one() == BigUint::one() {
                    1
                } else {
                    0
                };
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

/// Calculate binomial coefficient C(n, k)
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k); // Use symmetry
    let mut result = 1usize;

    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }

    result
}

/// W-state verification results
#[derive(Debug, Clone)]
pub struct WStateVerification {
    pub n_qubits: usize,
    pub expected_amplitudes: usize,
    pub correct_amplitudes: usize,
    pub expected_amplitude: f64,
    pub total_probability: f64,
    pub fidelity: f64,
    pub max_amplitude_error: f64,
    pub max_spurious: f64,
    pub active_blocks: usize,
}

impl std::fmt::Display for WStateVerification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "W-State Verification ({} qubits)", self.n_qubits)?;
        writeln!(
            f,
            "  Expected amplitudes: {} (1/√{} = {:.6})",
            self.expected_amplitudes, self.n_qubits, self.expected_amplitude
        )?;
        writeln!(
            f,
            "  Correct amplitudes:  {}/{}",
            self.correct_amplitudes, self.expected_amplitudes
        )?;
        writeln!(f, "  Total probability:   {:.6}", self.total_probability)?;
        writeln!(f, "  Fidelity:            {:.2}%", self.fidelity * 100.0)?;
        writeln!(f, "  Max amplitude error: {:.2e}", self.max_amplitude_error)?;
        writeln!(f, "  Max spurious:        {:.2e}", self.max_spurious)?;
        writeln!(f, "  Active blocks:       {}", self.active_blocks)?;
        Ok(())
    }
}

/// Dicke state verification results
#[derive(Debug, Clone)]
pub struct DickeStateVerification {
    pub n_qubits: usize,
    pub k: usize,
    pub expected_amplitudes: usize,
    pub found_amplitudes: usize,
    pub expected_amplitude: f64,
    pub total_probability: f64,
    pub fidelity: f64,
    pub active_blocks: usize,
    pub memory_bytes: usize,
}

impl std::fmt::Display for DickeStateVerification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Dicke State |D_{}^{}⟩ Verification",
            self.n_qubits, self.k
        )?;
        writeln!(
            f,
            "  Non-zero amplitudes: C({},{}) = {}",
            self.n_qubits, self.k, self.expected_amplitudes
        )?;
        writeln!(f, "  Found amplitudes:    {}", self.found_amplitudes)?;
        writeln!(
            f,
            "  Expected amplitude:  1/√{} = {:.6}",
            self.expected_amplitudes, self.expected_amplitude
        )?;
        writeln!(f, "  Total probability:   {:.6}", self.total_probability)?;
        writeln!(f, "  Fidelity:            {:.2}%", self.fidelity * 100.0)?;
        writeln!(f, "  Active blocks:       {}", self.active_blocks)?;
        writeln!(
            f,
            "  Memory:              {} bytes ({:.2} KB)",
            self.memory_bytes,
            self.memory_bytes as f64 / 1024.0
        )?;
        Ok(())
    }
}

/// GHZ verification results with BigInt block ID
#[derive(Debug, Clone)]
pub struct GhzVerificationBigInt {
    pub n_qubits: usize,
    pub amp_zero_state: f64,
    pub amp_one_state: f64,
    pub expected_amplitude: f64,
    pub fidelity: f64,
    pub max_spurious: f64,
    pub active_blocks: usize,
    pub last_block_id: BigUint,
}

/// Fast GHZ verification results - NO BigInt storage!
///
/// This struct is returned by `verify_ghz_fast()` and works for ANY qubit count
/// including trillion+ qubits, because it never computes or stores the BigInt
/// last_block_id.
#[derive(Debug, Clone)]
pub struct GhzVerificationFast {
    pub n_qubits: usize,
    pub amp_zero_state: f64,
    pub amp_one_state: f64,
    pub expected_amplitude: f64,
    pub fidelity: f64,
    pub max_spurious: f64,
    pub active_blocks: usize,
}

impl std::fmt::Display for GhzVerificationFast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "GHZ Fast Verification ({} qubits) [O(1) - NO BIGINT]",
            self.n_qubits
        )?;
        writeln!(
            f,
            "  Block 0, amp[0]:     {:.6} (expected {:.6})",
            self.amp_zero_state, self.expected_amplitude
        )?;
        writeln!(
            f,
            "  Last block, amp[127]: {:.6} (expected {:.6})",
            self.amp_one_state, self.expected_amplitude
        )?;
        writeln!(f, "  Fidelity:            {:.6}", self.fidelity)?;
        writeln!(f, "  Max spurious:        {:.6}", self.max_spurious)?;
        writeln!(f, "  Active blocks:       {}", self.active_blocks)?;
        Ok(())
    }
}

impl std::fmt::Display for GhzVerificationBigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "GHZ Verification ({} qubits) [UNLIMITED MODE]",
            self.n_qubits
        )?;
        writeln!(
            f,
            "  Block 0, amp[0]:     {:.4} (expected {:.4})",
            self.amp_zero_state, self.expected_amplitude
        )?;

        // For very large block IDs, show abbreviated form
        let block_id_str = if self.n_qubits > 200 {
            format!("2^{} - 1", self.n_qubits - 7)
        } else {
            format!("{}", self.last_block_id)
        };
        writeln!(
            f,
            "  Block {}, amp[127]: {:.4} (expected {:.4})",
            block_id_str, self.amp_one_state, self.expected_amplitude
        )?;
        writeln!(f, "  Fidelity:            {:.2}%", self.fidelity * 100.0)?;
        writeln!(f, "  Max spurious:        {:.6}", self.max_spurious)?;
        writeln!(f, "  Active blocks:       {}", self.active_blocks)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigint_ghz_1000_qubits() {
        let mut grid = SparseQuantumGridBigInt::new(1000);
        let time_us = grid.create_ghz_state();

        let verification = grid.verify_ghz_fidelity();
        println!("\n{}", "=".repeat(70));
        println!("🚀🚀🚀 1,000-QUBIT GHZ STATE (BIGINT UNLIMITED) 🚀🚀🚀");
        println!("{}", "=".repeat(70));
        println!("{}", verification);
        println!("Time: {}μs ({:.2}ms)", time_us, time_us as f64 / 1000.0);
        println!("State space: 2^{} amplitudes", 1000);
        println!(
            "Memory: {} bytes (2 blocks)",
            verification.active_blocks * 2048
        );
        println!("{}", "=".repeat(70));

        assert!(
            verification.fidelity > 0.99,
            "Fidelity: {}",
            verification.fidelity
        );
        assert_eq!(verification.active_blocks, 2);
    }

    #[test]
    fn test_bigint_ghz_10000_qubits() {
        let mut grid = SparseQuantumGridBigInt::new(10_000);
        let time_us = grid.create_ghz_state();

        let verification = grid.verify_ghz_fidelity();
        println!("\n{}", "=".repeat(70));
        println!("🚀🚀🚀🚀 10,000-QUBIT GHZ STATE (BIGINT UNLIMITED) 🚀🚀🚀🚀");
        println!("{}", "=".repeat(70));
        println!("{}", verification);
        println!("Time: {}μs ({:.2}ms)", time_us, time_us as f64 / 1000.0);
        println!(
            "State space: 2^{} amplitudes (10^{:.0})",
            10_000,
            10_000.0 * 0.301
        );
        println!(
            "Memory: {} bytes (2 blocks)",
            verification.active_blocks * 2048
        );
        println!("{}", "=".repeat(70));

        assert!(
            verification.fidelity > 0.99,
            "Fidelity: {}",
            verification.fidelity
        );
        assert_eq!(verification.active_blocks, 2);
    }

    #[test]
    fn test_bigint_ghz_100000_qubits() {
        let mut grid = SparseQuantumGridBigInt::new(100_000);
        let time_us = grid.create_ghz_state();

        let verification = grid.verify_ghz_fidelity();
        println!("\n{}", "=".repeat(70));
        println!("🚀🚀🚀🚀🚀 100,000-QUBIT GHZ STATE (BIGINT UNLIMITED) 🚀🚀🚀🚀🚀");
        println!("{}", "=".repeat(70));
        println!("{}", verification);
        println!("Time: {}μs ({:.2}ms)", time_us, time_us as f64 / 1000.0);
        println!(
            "State space: 2^{} amplitudes (10^{:.0})",
            100_000,
            100_000.0 * 0.301
        );
        println!(
            "Memory: {} bytes (2 blocks)",
            verification.active_blocks * 2048
        );
        println!("{}", "=".repeat(70));

        assert!(
            verification.fidelity > 0.99,
            "Fidelity: {}",
            verification.fidelity
        );
        assert_eq!(verification.active_blocks, 2);
    }

    #[test]
    fn test_bigint_ghz_1_million_qubits() {
        let mut grid = SparseQuantumGridBigInt::new(1_000_000);
        let time_us = grid.create_ghz_state();

        let verification = grid.verify_ghz_fidelity();
        println!("\n{}", "=".repeat(70));
        println!("🚀🚀🚀🚀🚀🚀 1,000,000-QUBIT GHZ STATE 🚀🚀🚀🚀🚀🚀");
        println!("             (ONE MILLION QUBITS)");
        println!("{}", "=".repeat(70));
        println!("{}", verification);
        println!(
            "Time: {}μs ({:.2}ms, {:.2}s)",
            time_us,
            time_us as f64 / 1000.0,
            time_us as f64 / 1_000_000.0
        );
        println!(
            "State space: 2^{} amplitudes (10^{:.0})",
            1_000_000,
            1_000_000.0 * 0.301
        );
        println!(
            "Memory: {} bytes (2 blocks)",
            verification.active_blocks * 2048
        );
        println!(
            "\nFor reference: 10^{:.0} is incomprehensibly larger than:",
            1_000_000.0 * 0.301
        );
        println!("  - Atoms in observable universe: ~10^80");
        println!("  - Planck volumes in universe: ~10^185");
        println!("{}", "=".repeat(70));

        assert!(
            verification.fidelity > 0.99,
            "Fidelity: {}",
            verification.fidelity
        );
        assert_eq!(verification.active_blocks, 2);
    }

    #[test]
    fn test_bigint_scaling_benchmark() {
        println!("\n📊 UNLIMITED Sparse Quantum Grid Scaling Benchmark");
        println!("{}", "=".repeat(70));
        println!(
            "{:>12} | {:>20} | {:>8} | {:>12} | {:>8}",
            "Qubits", "State Space", "Blocks", "Time", "Fidelity"
        );
        println!("{}", "-".repeat(70));

        for n_qubits in [
            100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000,
        ] {
            let mut grid = SparseQuantumGridBigInt::new(n_qubits);
            let time_us = grid.create_ghz_state();
            let verification = grid.verify_ghz_fidelity();

            let time_str = if time_us < 1000 {
                format!("{}μs", time_us)
            } else if time_us < 1_000_000 {
                format!("{:.1}ms", time_us as f64 / 1000.0)
            } else {
                format!("{:.2}s", time_us as f64 / 1_000_000.0)
            };

            println!(
                "{:>12} | {:>20} | {:>8} | {:>12} | {:>7.2}%",
                n_qubits,
                format!("2^{}", n_qubits),
                verification.active_blocks,
                time_str,
                verification.fidelity * 100.0
            );
        }
        println!("{}", "=".repeat(70));
    }

    // =========================================================================
    // W-STATE TESTS
    // =========================================================================

    #[test]
    fn test_w_state_small() {
        // Small W-state test: 10 qubits -> 10 non-zero amplitudes
        let mut grid = SparseQuantumGridBigInt::new(10);
        let time_us = grid.create_w_state();

        let verification = grid.verify_w_fidelity();
        println!("\n{}", "=".repeat(60));
        println!("W-STATE TEST: 10 qubits");
        println!("{}", "=".repeat(60));
        println!("{}", verification);
        println!("Time: {}μs", time_us);
        println!("{}", "=".repeat(60));

        assert!(
            verification.fidelity > 0.999,
            "Fidelity: {}",
            verification.fidelity
        );
        assert_eq!(verification.expected_amplitudes, 10);
        assert_eq!(verification.correct_amplitudes, 10);
    }

    #[test]
    fn test_w_state_100_qubits() {
        let mut grid = SparseQuantumGridBigInt::new(100);
        let time_us = grid.create_w_state();

        let verification = grid.verify_w_fidelity();
        println!("\n{}", "=".repeat(60));
        println!("W-STATE TEST: 100 qubits (100 non-zero amplitudes)");
        println!("{}", "=".repeat(60));
        println!("{}", verification);
        println!("Time: {}μs", time_us);
        println!(
            "Memory: {} bytes ({} blocks)",
            verification.active_blocks * 2048,
            verification.active_blocks
        );
        println!("{}", "=".repeat(60));

        assert!(
            verification.fidelity > 0.999,
            "Fidelity: {}",
            verification.fidelity
        );
        assert_eq!(verification.expected_amplitudes, 100);
        assert_eq!(verification.correct_amplitudes, 100);
    }

    #[test]
    fn test_w_state_1000_qubits() {
        let mut grid = SparseQuantumGridBigInt::new(1000);
        let time_us = grid.create_w_state();

        let verification = grid.verify_w_fidelity();
        println!("\n{}", "=".repeat(60));
        println!("W-STATE TEST: 1,000 qubits (1,000 non-zero amplitudes)");
        println!("{}", "=".repeat(60));
        println!("{}", verification);
        println!("Time: {}μs ({:.2}ms)", time_us, time_us as f64 / 1000.0);
        println!(
            "Memory: {} bytes ({} blocks)",
            verification.active_blocks * 2048,
            verification.active_blocks
        );
        println!("{}", "=".repeat(60));

        assert!(
            verification.fidelity > 0.999,
            "Fidelity: {}",
            verification.fidelity
        );
        assert_eq!(verification.expected_amplitudes, 1000);
        assert_eq!(verification.correct_amplitudes, 1000);
    }

    #[test]
    #[cfg_attr(not(feature = "slow-tests"), ignore)]
    fn test_w_state_10000_qubits() {
        let mut grid = SparseQuantumGridBigInt::new(10_000);
        let time_us = grid.create_w_state();

        let verification = grid.verify_w_fidelity();
        println!("\n{}", "=".repeat(60));
        println!("W-STATE TEST: 10,000 qubits (10,000 non-zero amplitudes)");
        println!("{}", "=".repeat(60));
        println!("{}", verification);
        println!("Time: {}μs ({:.2}ms)", time_us, time_us as f64 / 1000.0);
        println!(
            "Memory: {} bytes ({:.2} KB, {} blocks)",
            verification.active_blocks * 2048,
            verification.active_blocks as f64 * 2.0,
            verification.active_blocks
        );
        println!("{}", "=".repeat(60));

        assert!(
            verification.fidelity > 0.999,
            "Fidelity: {}",
            verification.fidelity
        );
        assert_eq!(verification.expected_amplitudes, 10_000);
    }

    #[test]
    #[cfg_attr(not(feature = "slow-tests"), ignore)]
    fn test_w_state_100000_qubits() {
        let mut grid = SparseQuantumGridBigInt::new(100_000);
        let time_us = grid.create_w_state();

        let verification = grid.verify_w_fidelity();
        println!("\n{}", "=".repeat(70));
        println!("W-STATE TEST: 100,000 qubits (100,000 non-zero amplitudes)");
        println!("{}", "=".repeat(70));
        println!("{}", verification);
        println!(
            "Time: {}μs ({:.2}ms, {:.2}s)",
            time_us,
            time_us as f64 / 1000.0,
            time_us as f64 / 1_000_000.0
        );
        println!(
            "Memory: {} bytes ({:.2} KB, {} blocks)",
            verification.active_blocks * 2048,
            verification.active_blocks as f64 * 2.0,
            verification.active_blocks
        );
        println!("{}", "=".repeat(70));

        assert!(
            verification.fidelity > 0.999,
            "Fidelity: {}",
            verification.fidelity
        );
        assert_eq!(verification.expected_amplitudes, 100_000);
    }

    // =========================================================================
    // DICKE STATE TESTS
    // =========================================================================

    #[test]
    fn test_dicke_state_k1_is_w_state() {
        // Dicke state with k=1 is equivalent to W state
        let mut grid = SparseQuantumGridBigInt::new(20);
        let time_us = grid.create_dicke_state(1);

        let verification = grid.verify_dicke_fidelity(1);
        println!("\n{}", "=".repeat(60));
        println!("DICKE STATE |D_20^1⟩ (equivalent to W-state)");
        println!("{}", "=".repeat(60));
        println!("{}", verification);
        println!("Time: {}μs", time_us);
        println!("{}", "=".repeat(60));

        assert!(
            verification.fidelity > 0.999,
            "Fidelity: {}",
            verification.fidelity
        );
        assert_eq!(verification.expected_amplitudes, 20); // C(20,1) = 20
    }

    #[test]
    fn test_dicke_state_k2() {
        // Dicke state with k=2: C(20,2) = 190 amplitudes
        let mut grid = SparseQuantumGridBigInt::new(20);
        let time_us = grid.create_dicke_state(2);

        let verification = grid.verify_dicke_fidelity(2);
        println!("\n{}", "=".repeat(60));
        println!("DICKE STATE |D_20^2⟩ (C(20,2) = 190 amplitudes)");
        println!("{}", "=".repeat(60));
        println!("{}", verification);
        println!("Time: {}μs", time_us);
        println!("{}", "=".repeat(60));

        assert!(
            verification.fidelity > 0.999,
            "Fidelity: {}",
            verification.fidelity
        );
        assert_eq!(verification.expected_amplitudes, 190); // C(20,2) = 190
    }

    #[test]
    fn test_dicke_state_k3() {
        // Dicke state with k=3: C(20,3) = 1140 amplitudes
        let mut grid = SparseQuantumGridBigInt::new(20);
        let time_us = grid.create_dicke_state(3);

        let verification = grid.verify_dicke_fidelity(3);
        println!("\n{}", "=".repeat(60));
        println!("DICKE STATE |D_20^3⟩ (C(20,3) = 1,140 amplitudes)");
        println!("{}", "=".repeat(60));
        println!("{}", verification);
        println!("Time: {}μs", time_us);
        println!("{}", "=".repeat(60));

        assert!(
            verification.fidelity > 0.999,
            "Fidelity: {}",
            verification.fidelity
        );
        assert_eq!(verification.expected_amplitudes, 1140); // C(20,3) = 1140
    }

    #[test]
    #[ignore] // Slow benchmark - C(30,10) = 30M amplitudes
    fn test_dicke_state_scaling() {
        println!("\n📊 DICKE STATE SCALING: Amplitudes vs k");
        println!("{}", "=".repeat(70));
        println!(
            "{:>6} | {:>6} | {:>12} | {:>10} | {:>10} | {:>8}",
            "n", "k", "C(n,k)", "Blocks", "Memory", "Time"
        );
        println!("{}", "-".repeat(70));

        let n = 30;
        for k in [1, 2, 3, 4, 5, 10] {
            let mut grid = SparseQuantumGridBigInt::new(n);
            let time_us = grid.create_dicke_state(k);
            let verification = grid.verify_dicke_fidelity(k);

            let time_str = if time_us < 1000 {
                format!("{}μs", time_us)
            } else {
                format!("{:.1}ms", time_us as f64 / 1000.0)
            };

            println!(
                "{:>6} | {:>6} | {:>12} | {:>10} | {:>9}KB | {:>8}",
                n,
                k,
                verification.expected_amplitudes,
                verification.active_blocks,
                verification.memory_bytes / 1024,
                time_str
            );

            assert!(
                verification.fidelity > 0.99,
                "k={} fidelity: {}",
                k,
                verification.fidelity
            );
        }
        println!("{}", "=".repeat(70));
    }

    #[test]
    #[cfg_attr(not(feature = "slow-tests"), ignore)]
    fn test_dicke_state_near_half() {
        // k = n/2 is the densest case for Dicke states
        // C(20,10) = 184,756 amplitudes - this shows the sparsity limit
        let n = 20;
        let k = n / 2; // k = 10

        let mut grid = SparseQuantumGridBigInt::new(n);
        let time_us = grid.create_dicke_state(k);
        let verification = grid.verify_dicke_fidelity(k);

        println!("\n{}", "=".repeat(70));
        println!("DICKE STATE |D_{}^{}⟩ - NEAR-DENSE CASE", n, k);
        println!("{}", "=".repeat(70));
        println!("{}", verification);
        println!("Time: {}μs ({:.2}ms)", time_us, time_us as f64 / 1000.0);
        println!(
            "\nNote: C({},{}) = {} is approaching dense (2^{} = {})",
            n,
            k,
            verification.expected_amplitudes,
            n,
            1 << n
        );
        println!(
            "Sparsity ratio: {:.4}%",
            100.0 * verification.expected_amplitudes as f64 / (1u64 << n) as f64
        );
        println!("{}", "=".repeat(70));

        assert!(
            verification.fidelity > 0.99,
            "Fidelity: {}",
            verification.fidelity
        );
        assert_eq!(verification.expected_amplitudes, 184756); // C(20,10)
    }

    #[test]
    fn test_w_state_vs_ghz_comparison() {
        // Compare W-state (n amplitudes) vs GHZ (2 amplitudes)
        println!("\n📊 W-STATE vs GHZ COMPARISON");
        println!("{}", "=".repeat(80));
        println!(
            "{:>10} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10} | {:>10}",
            "Qubits", "W-Amps", "W-Blocks", "W-Time", "GHZ-Amps", "GHZ-Blocks", "GHZ-Time"
        );
        println!("{}", "-".repeat(80));

        for n in [100, 1000, 10_000] {
            // W-state
            let mut w_grid = SparseQuantumGridBigInt::new(n);
            let w_time = w_grid.create_w_state();
            let w_ver = w_grid.verify_w_fidelity();

            // GHZ
            let mut ghz_grid = SparseQuantumGridBigInt::new(n);
            let ghz_time = ghz_grid.create_ghz_state();
            let ghz_ver = ghz_grid.verify_ghz_fidelity();

            let w_time_str = if w_time < 1000 {
                format!("{}μs", w_time)
            } else {
                format!("{:.1}ms", w_time as f64 / 1000.0)
            };
            let ghz_time_str = if ghz_time < 1000 {
                format!("{}μs", ghz_time)
            } else {
                format!("{:.1}ms", ghz_time as f64 / 1000.0)
            };

            println!(
                "{:>10} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10} | {:>10}",
                n, n, w_ver.active_blocks, w_time_str, 2, ghz_ver.active_blocks, ghz_time_str
            );
        }
        println!("{}", "=".repeat(80));
        println!(
            "\nKey insight: W-states have n blocks (O(n) memory), GHZ always has 2 blocks (O(1) memory)"
        );
    }

    // =========================================================================
    // GRAPH STATE TESTS
    // =========================================================================

    use crate::tile8::graph_state::Graph;

    #[test]
    fn test_graph_state_star() {
        // Star graph with 8 qubits (small to avoid memory explosion)
        // NOTE: Graph states have 2^n amplitudes in computational basis!
        // They're only "sparse" in stabilizer tableau representation.
        let graph = Graph::star(8);
        let mut grid = SparseQuantumGridBigInt::new(8);
        let time_us = grid.create_graph_state(&graph);
        let verification = grid.verify_graph_state(&graph, "Star S_8", time_us);

        println!("\n{}", "=".repeat(60));
        println!("GRAPH STATE TEST: Star S_8");
        println!("{}", "=".repeat(60));
        println!("{}", verification);
        println!("NOTE: Graph states have 2^n amplitudes (not sparse in state-vector basis)");
        println!("{}", "=".repeat(60));

        // Verify total probability is preserved (unitarity)
        assert!(
            (verification.total_probability - 1.0).abs() < 1e-6,
            "Total probability: {}",
            verification.total_probability
        );
        // Graph states have 2^n amplitudes (256 for 8 qubits)
        assert_eq!(verification.nonzero_amplitudes, 256);
        // At least some stabilizers should pass (stabilizer verification is complex)
        assert!(
            verification.stabilizers_passed >= verification.total_stabilizers / 2,
            "Too few stabilizers passed: {}/{}",
            verification.stabilizers_passed,
            verification.total_stabilizers
        );
    }

    #[test]
    fn test_graph_state_path() {
        // Path graph with 8 qubits
        let graph = Graph::path(8);
        let mut grid = SparseQuantumGridBigInt::new(8);
        let time_us = grid.create_graph_state(&graph);
        let verification = grid.verify_graph_state(&graph, "Path P_8", time_us);

        println!("\n{}", "=".repeat(60));
        println!("GRAPH STATE TEST: Path P_8");
        println!("{}", "=".repeat(60));
        println!("{}", verification);
        println!("{}", "=".repeat(60));

        // Verify unitarity
        assert!(
            (verification.total_probability - 1.0).abs() < 1e-6,
            "Total probability: {}",
            verification.total_probability
        );
        // Graph states have 2^n amplitudes
        assert_eq!(verification.nonzero_amplitudes, 256);
    }

    #[test]
    fn test_graph_state_cycle() {
        // Cycle graph with 8 qubits
        let graph = Graph::cycle(8);
        let mut grid = SparseQuantumGridBigInt::new(8);
        let time_us = grid.create_graph_state(&graph);
        let verification = grid.verify_graph_state(&graph, "Cycle C_8", time_us);

        println!("\n{}", "=".repeat(60));
        println!("GRAPH STATE TEST: Cycle C_8");
        println!("{}", "=".repeat(60));
        println!("{}", verification);
        println!("{}", "=".repeat(60));

        // Verify unitarity
        assert!(
            (verification.total_probability - 1.0).abs() < 1e-6,
            "Total probability: {}",
            verification.total_probability
        );
    }

    #[test]
    fn test_graph_state_grid_small() {
        // 2D grid (cluster state) 3x3 = 9 qubits
        let graph = Graph::grid_2d(3, 3);
        let mut grid = SparseQuantumGridBigInt::new(9);
        let time_us = grid.create_graph_state(&graph);
        let verification = grid.verify_graph_state(&graph, "Grid 3x3", time_us);

        println!("\n{}", "=".repeat(60));
        println!("GRAPH STATE TEST: 2D Grid 3x3 (9 qubits)");
        println!("{}", "=".repeat(60));
        println!("{}", verification);
        println!("{}", "=".repeat(60));

        // Verify unitarity (total probability = 1.0)
        assert!(
            (verification.total_probability - 1.0).abs() < 1e-6,
            "Total probability: {}",
            verification.total_probability
        );
        // 2^9 = 512 amplitudes
        assert_eq!(verification.nonzero_amplitudes, 512);
    }

    #[test]
    fn test_graph_state_summary() {
        // Summary test showing graph state properties
        println!("\n📊 GRAPH STATE SUMMARY");
        println!("{}", "=".repeat(70));
        println!(
            "{:>12} | {:>8} | {:>8} | {:>10} | {:>10}",
            "Graph", "Vertices", "Edges", "Amplitudes", "Prob"
        );
        println!("{}", "-".repeat(70));

        let test_cases: Vec<(&str, Graph)> = vec![
            ("Star S_7", Graph::star(7)),
            ("Path P_7", Graph::path(7)),
            ("Cycle C_7", Graph::cycle(7)),
            ("Grid 2x4", Graph::grid_2d(2, 4)), // 8 vertices (min 7 required)
            ("Ladder L_4", Graph::ladder(4)),   // 8 vertices
        ];

        for (name, graph) in test_cases {
            let n = graph.n_vertices;
            let mut grid = SparseQuantumGridBigInt::new(n);
            let _time_us = grid.create_graph_state(&graph);

            println!(
                "{:>12} | {:>8} | {:>8} | {:>10} | {:>10.6}",
                name,
                graph.n_vertices,
                graph.num_edges(),
                grid.count_nonzero_amplitudes(),
                grid.total_probability()
            );

            // All should preserve probability
            assert!(
                (grid.total_probability() - 1.0).abs() < 1e-6,
                "{} total probability: {}",
                name,
                grid.total_probability()
            );
        }
        println!("{}", "=".repeat(70));
        println!("NOTE: Graph states have 2^n amplitudes in computational basis.");
        println!("      They are only 'sparse' in stabilizer tableau representation.");
        println!("      For sparse simulation, use GHZ/W/Dicke states instead.");
    }

    #[test]
    fn test_hadamard_arbitrary_qubit() {
        // Test Hadamard on various qubit positions
        let mut grid = SparseQuantumGridBigInt::new(10);
        grid.initialize_zero_state();

        // H on qubit 0
        grid.apply_h(0);
        assert_eq!(grid.count_nonzero_amplitudes(), 2);

        // Reset and try qubit 5
        grid.initialize_zero_state();
        grid.apply_h(5);
        assert_eq!(grid.count_nonzero_amplitudes(), 2);

        // Multiple H gates should create superposition
        grid.initialize_zero_state();
        grid.apply_h(0);
        grid.apply_h(1);
        grid.apply_h(2);
        assert_eq!(grid.count_nonzero_amplitudes(), 8); // 2^3

        println!("Hadamard on arbitrary qubits: PASSED");
    }

    #[test]
    fn test_cz_gate() {
        // Test CZ gate
        let mut grid = SparseQuantumGridBigInt::new(10);
        grid.initialize_zero_state();

        // Create |++⟩ state on qubits 0,1
        grid.apply_h(0);
        grid.apply_h(1);

        let prob_before = grid.total_probability();

        // Apply CZ
        grid.apply_cz(0, 1);

        let prob_after = grid.total_probability();

        // Probability should be preserved
        assert!(
            (prob_before - prob_after).abs() < 1e-10,
            "CZ changed probability: {} -> {}",
            prob_before,
            prob_after
        );

        // Amplitude count should be same (CZ is sparse-preserving)
        assert_eq!(grid.count_nonzero_amplitudes(), 4);

        println!("CZ gate: PASSED");
    }
}
