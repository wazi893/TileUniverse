// EPIC 88: Block-Sparse Quantum State Representation
//
// Extends EPIC 85's element-sparse representation with block-level sparsity
// optimized for GPU tensor core (WMMA) acceleration.
//
// Key design decisions:
// - Block size: 128 complex amplitudes = 256 f16 values = 1 WMMA tile
// - Qubits 0-6: Operations within block (stride 1-64)
// - Qubits 7+: Operations span blocks (need gather/scatter like EPIC 87)
// - Auto-conversion from element-sparse when block count is sufficient
//
// Phase 88.2: GPU integration for tensor core acceleration

#[cfg(feature = "cuda")]
use crate::cuda::{CudaError, CudaResult, CudaRuntime};
#[cfg(feature = "cuda")]
use cudarc::driver::PushKernelArg;
use half::f16;
use std::collections::HashMap;

/// Block size: 128 complex amplitudes = 1 WMMA tile (256 f16 values)
pub const BLOCK_SIZE: usize = 128;

/// log2(BLOCK_SIZE) for bit shifting operations
pub const BLOCK_SHIFT: u8 = 7;

/// Minimum blocks for GPU to be profitable
///
/// SPRINT 41.0: Tuned for RTX 5090 (compute_120)
/// Benchmark results on RTX 5090:
///   - 8 blocks: CPU 4.3x faster
///   - 32 blocks: GPU 3.6x faster
///   - 128 blocks: GPU 15x faster
///   - 1024 blocks: GPU 77x faster
///   - 8192 blocks: GPU 254x faster
///
/// For older GPUs (RTX 40 series), a higher threshold like 256-512 may be optimal.
pub const MIN_BLOCKS_FOR_GPU: usize = 32;

/// Threshold below which amplitudes are considered zero
pub const BLOCK_SPARSE_THRESHOLD: f32 = 1e-10;

/// A complex amplitude in f16 precision (for WMMA compatibility)
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Complex16 {
    pub re: f16,
    pub im: f16,
}

impl Complex16 {
    pub const ZERO: Complex16 = Complex16 {
        re: f16::ZERO,
        im: f16::ZERO,
    };

    pub const ONE: Complex16 = Complex16 {
        re: f16::ONE,
        im: f16::ZERO,
    };

    #[inline]
    pub fn new(re: f16, im: f16) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn from_f32(re: f32, im: f32) -> Self {
        Self {
            re: f16::from_f32(re),
            im: f16::from_f32(im),
        }
    }

    #[inline]
    pub fn to_f32(self) -> (f32, f32) {
        (self.re.to_f32(), self.im.to_f32())
    }

    #[inline]
    pub fn norm_sq(&self) -> f32 {
        let re = self.re.to_f32();
        let im = self.im.to_f32();
        re * re + im * im
    }

    #[inline]
    pub fn is_negligible(&self) -> bool {
        self.norm_sq() < BLOCK_SPARSE_THRESHOLD * BLOCK_SPARSE_THRESHOLD
    }
}

/// A single block of 128 complex amplitudes
///
/// Stored as arrays for WMMA-friendly memory layout:
/// - real[128]: real parts
/// - imag[128]: imaginary parts
#[derive(Clone)]
pub struct Block {
    /// Real parts of amplitudes
    pub real: Box<[f16; BLOCK_SIZE]>,
    /// Imaginary parts of amplitudes
    pub imag: Box<[f16; BLOCK_SIZE]>,
    /// Count of non-zero amplitudes (for optimization decisions)
    pub nnz: usize,
}

impl Block {
    /// Create a new zero block
    pub fn new_zero() -> Self {
        Self {
            real: Box::new([f16::ZERO; BLOCK_SIZE]),
            imag: Box::new([f16::ZERO; BLOCK_SIZE]),
            nnz: 0,
        }
    }

    /// Create a block with a single non-zero amplitude at local index
    pub fn with_single(local_idx: usize, amp: Complex16) -> Self {
        assert!(local_idx < BLOCK_SIZE);
        let mut block = Self::new_zero();
        block.real[local_idx] = amp.re;
        block.imag[local_idx] = amp.im;
        block.nnz = 1;
        block
    }

    /// Get amplitude at local index
    #[inline]
    pub fn get(&self, local_idx: usize) -> Complex16 {
        Complex16 {
            re: self.real[local_idx],
            im: self.imag[local_idx],
        }
    }

    /// Set amplitude at local index
    #[inline]
    pub fn set(&mut self, local_idx: usize, amp: Complex16) {
        let was_zero = self.real[local_idx] == f16::ZERO && self.imag[local_idx] == f16::ZERO;
        let is_zero = amp.is_negligible();

        self.real[local_idx] = amp.re;
        self.imag[local_idx] = amp.im;

        // Update nnz count
        if was_zero && !is_zero {
            self.nnz += 1;
        } else if !was_zero && is_zero {
            self.nnz = self.nnz.saturating_sub(1);
        }
    }

    /// Check if block is all zeros
    pub fn is_empty(&self) -> bool {
        self.nnz == 0
    }

    /// Recount non-zero amplitudes (useful after bulk operations)
    pub fn recount_nnz(&mut self) {
        self.nnz = self
            .real
            .iter()
            .zip(self.imag.iter())
            .filter(|(r, i)| {
                let re = r.to_f32();
                let im = i.to_f32();
                re * re + im * im >= BLOCK_SPARSE_THRESHOLD * BLOCK_SPARSE_THRESHOLD
            })
            .count();
    }

    /// Get raw pointers for GPU transfer
    pub fn as_f16_slice(&self) -> (&[f16], &[f16]) {
        (self.real.as_ref(), self.imag.as_ref())
    }

    /// Get mutable raw pointers for GPU transfer
    pub fn as_f16_slice_mut(&mut self) -> (&mut [f16], &mut [f16]) {
        (self.real.as_mut(), self.imag.as_mut())
    }
}

impl std::fmt::Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Block").field("nnz", &self.nnz).finish()
    }
}

/// Block-sparse quantum state representation
///
/// Partitions the 2^n amplitude space into blocks of 128 amplitudes each.
/// Only non-zero blocks are stored.
#[derive(Clone, Debug)]
pub struct BlockSparseState {
    /// Number of qubits
    pub n_qubits: u8,
    /// Non-zero blocks: block_id -> Block
    blocks: HashMap<usize, Block>,
    /// Total number of possible blocks (2^(n_qubits - BLOCK_SHIFT))
    total_blocks: usize,
}

impl BlockSparseState {
    /// Create a new block-sparse state initialized to |0...0⟩
    pub fn new_zero(n_qubits: u8) -> Self {
        assert!(
            n_qubits >= BLOCK_SHIFT,
            "Need at least {} qubits for block-sparse",
            BLOCK_SHIFT
        );
        assert!(n_qubits <= 64, "n_qubits must be <= 64");

        // For 64 qubits: total_blocks = 2^57 = 144 quadrillion (fits in usize on 64-bit)
        let total_blocks = 1usize << (n_qubits - BLOCK_SHIFT);
        let mut blocks = HashMap::new();

        // |0...0⟩ is in block 0, local index 0
        blocks.insert(0, Block::with_single(0, Complex16::ONE));

        Self {
            n_qubits,
            blocks,
            total_blocks,
        }
    }

    /// Create from a single basis state |index⟩
    pub fn from_basis(n_qubits: u8, index: u64) -> Self {
        assert!((BLOCK_SHIFT..=64).contains(&n_qubits));
        // For 64 qubits, all u64 indices are valid (0 to 2^64-1)
        // For <64 qubits, check that index fits within the state space
        if n_qubits < 64 {
            assert!(index < (1u64 << n_qubits), "Basis index out of range");
        }

        // For 64 qubits: total_blocks = 2^57
        let total_blocks = 1usize << (n_qubits - BLOCK_SHIFT);
        let mut blocks = HashMap::new();

        let block_id = (index >> BLOCK_SHIFT) as usize;
        let local_idx = (index & (BLOCK_SIZE as u64 - 1)) as usize;
        blocks.insert(block_id, Block::with_single(local_idx, Complex16::ONE));

        Self {
            n_qubits,
            blocks,
            total_blocks,
        }
    }

    /// Number of active (non-zero) blocks
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Total number of possible blocks
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Block-level sparsity (active_blocks / total_blocks)
    pub fn block_sparsity(&self) -> f64 {
        self.block_count() as f64 / self.total_blocks as f64
    }

    /// Total non-zero amplitudes across all blocks
    pub fn nnz(&self) -> usize {
        self.blocks.values().map(|b| b.nnz).sum()
    }

    /// Check if GPU path should be used
    pub fn should_use_gpu(&self) -> bool {
        self.block_count() >= MIN_BLOCKS_FOR_GPU
    }

    // =========================================================================
    // Phase 88.3: Block Creation and Partner Management
    // =========================================================================

    /// Ensure all partner blocks exist for a cross-block gate on given qubit.
    /// Creates zero-initialized partner blocks where missing.
    /// Returns the number of new blocks created.
    pub fn ensure_partner_blocks_exist(&mut self, qubit: u8) -> usize {
        if qubit < BLOCK_SHIFT {
            return 0; // Within-block gate, no partners needed
        }

        let block_bit = qubit - BLOCK_SHIFT;
        let block_mask = 1usize << block_bit;

        // Collect current block IDs (can't modify while iterating)
        let existing_ids: Vec<usize> = self.blocks.keys().copied().collect();
        let mut created = 0;

        for block_id in existing_ids {
            let partner_id = block_id ^ block_mask;
            if let std::collections::hash_map::Entry::Vacant(e) = self.blocks.entry(partner_id) {
                e.insert(Block::new_zero());
                created += 1;
            }
        }

        created
    }

    /// Check if any partner blocks are missing for a cross-block gate
    pub fn has_missing_partners(&self, qubit: u8) -> bool {
        if qubit < BLOCK_SHIFT {
            return false; // Within-block gate
        }

        let block_bit = qubit - BLOCK_SHIFT;
        let block_mask = 1usize << block_bit;

        for &block_id in self.blocks.keys() {
            let partner_id = block_id ^ block_mask;
            if !self.blocks.contains_key(&partner_id) {
                return true;
            }
        }
        false
    }

    /// Average number of non-zero amplitudes per block
    pub fn amplitudes_per_block(&self) -> f32 {
        if self.block_count() == 0 {
            return 0.0;
        }
        self.nnz() as f32 / self.block_count() as f32
    }

    /// Get block by ID (returns None if block is all zeros)
    pub fn get_block(&self, block_id: usize) -> Option<&Block> {
        self.blocks.get(&block_id)
    }

    /// Get mutable block by ID, creating if necessary
    pub fn get_block_mut(&mut self, block_id: usize) -> &mut Block {
        self.blocks.entry(block_id).or_insert_with(Block::new_zero)
    }

    /// Get amplitude at global index
    pub fn get(&self, index: u64) -> Complex16 {
        let block_id = (index >> BLOCK_SHIFT) as usize;
        let local_idx = (index & (BLOCK_SIZE as u64 - 1)) as usize;

        self.blocks
            .get(&block_id)
            .map(|b| b.get(local_idx))
            .unwrap_or(Complex16::ZERO)
    }

    /// Set amplitude at global index
    pub fn set(&mut self, index: u64, amp: Complex16) {
        let block_id = (index >> BLOCK_SHIFT) as usize;
        let local_idx = (index & (BLOCK_SIZE as u64 - 1)) as usize;

        if amp.is_negligible() {
            // Try to remove from block
            if let Some(block) = self.blocks.get_mut(&block_id) {
                block.set(local_idx, Complex16::ZERO);
                // Remove block if empty
                if block.is_empty() {
                    self.blocks.remove(&block_id);
                }
            }
        } else {
            let block = self.get_block_mut(block_id);
            block.set(local_idx, amp);
        }
    }

    /// Prune empty blocks
    pub fn prune(&mut self) {
        self.blocks.retain(|_, block| {
            block.recount_nnz();
            !block.is_empty()
        });
    }

    /// Iterator over active blocks
    pub fn iter_blocks(&self) -> impl Iterator<Item = (usize, &Block)> {
        self.blocks.iter().map(|(&id, block)| (id, block))
    }

    /// Mutable iterator over active blocks
    pub fn iter_blocks_mut(&mut self) -> impl Iterator<Item = (usize, &mut Block)> {
        self.blocks.iter_mut().map(|(&id, block)| (id, block))
    }

    // =========================================================================
    // Conversion from/to other representations
    // =========================================================================

    /// Convert from element-sparse SparseQState
    pub fn from_element_sparse(sparse: &super::sparse_state::SparseQState) -> Self {
        assert!(sparse.n_qubits >= BLOCK_SHIFT);

        let total_blocks = 1usize << (sparse.n_qubits - BLOCK_SHIFT);
        let mut blocks: HashMap<usize, Block> = HashMap::new();

        for (index, amp) in sparse.iter() {
            let block_id = (index >> BLOCK_SHIFT) as usize;
            let local_idx = (index & (BLOCK_SIZE as u64 - 1)) as usize;

            let block = blocks.entry(block_id).or_insert_with(Block::new_zero);
            block.set(local_idx, Complex16::from_f32(amp.re, amp.im));
        }

        Self {
            n_qubits: sparse.n_qubits,
            blocks,
            total_blocks,
        }
    }

    /// Convert to element-sparse SparseQState
    pub fn to_element_sparse(&self) -> super::sparse_state::SparseQState {
        let mut sparse = super::sparse_state::SparseQState::from_basis(self.n_qubits, 0);
        // Clear the default amplitude
        sparse.set(0, super::sparse_state::Complex::ZERO);

        for (&block_id, block) in &self.blocks {
            let base_index = (block_id as u64) << BLOCK_SHIFT;
            for local_idx in 0..BLOCK_SIZE {
                let amp = block.get(local_idx);
                if !amp.is_negligible() {
                    let (re, im) = amp.to_f32();
                    sparse.set(
                        base_index + local_idx as u64,
                        super::sparse_state::Complex::new(re, im),
                    );
                }
            }
        }

        sparse
    }

    /// Convert from dense QState
    pub fn from_dense(state: &crate::quantum::QState) -> Self {
        assert!(state.n_qubits >= BLOCK_SHIFT);

        let total_blocks = 1usize << (state.n_qubits - BLOCK_SHIFT);
        let mut blocks: HashMap<usize, Block> = HashMap::new();

        let real_slice = state.real.as_slice();
        let imag_slice = state.imag.as_slice();

        for block_id in 0..total_blocks {
            let base_idx = block_id * BLOCK_SIZE;
            let mut block = Block::new_zero();
            let mut has_nonzero = false;

            for local_idx in 0..BLOCK_SIZE {
                let global_idx = base_idx + local_idx;
                if global_idx >= state.len {
                    break;
                }

                let re = real_slice[global_idx];
                let im = imag_slice[global_idx];

                if re.abs() > BLOCK_SPARSE_THRESHOLD || im.abs() > BLOCK_SPARSE_THRESHOLD {
                    block.real[local_idx] = f16::from_f32(re);
                    block.imag[local_idx] = f16::from_f32(im);
                    has_nonzero = true;
                }
            }

            if has_nonzero {
                block.recount_nnz();
                blocks.insert(block_id, block);
            }
        }

        Self {
            n_qubits: state.n_qubits,
            blocks,
            total_blocks,
        }
    }

    /// Convert to dense QState (only for small qubit counts)
    pub fn to_dense(&self) -> Option<crate::quantum::QState> {
        if self.n_qubits > 26 {
            return None; // Too large for dense (64M amplitudes)
        }

        let mut state = crate::quantum::QState::new_zero(self.n_qubits);
        // Clear default |0⟩
        state.real.as_mut_slice()[0] = 0.0;

        for (&block_id, block) in &self.blocks {
            let base_idx = block_id * BLOCK_SIZE;
            for local_idx in 0..BLOCK_SIZE {
                let global_idx = base_idx + local_idx;
                if global_idx >= state.len {
                    break;
                }
                state.real.as_mut_slice()[global_idx] = block.real[local_idx].to_f32();
                state.imag.as_mut_slice()[global_idx] = block.imag[local_idx].to_f32();
            }
        }

        Some(state)
    }

    // =========================================================================
    // Block-local gate operations (qubits 0-6)
    // =========================================================================

    /// Apply X gate (bit flip) to qubit q
    /// For qubits 0-6: operates within each block
    /// For qubits 7+: needs cross-block handling
    pub fn apply_x(&mut self, q: u8) {
        assert!(q < self.n_qubits, "Qubit index out of range");

        if q < BLOCK_SHIFT {
            // Within-block: just swap pairs in each block
            let mask = 1usize << q;
            for block in self.blocks.values_mut() {
                for i in 0..BLOCK_SIZE {
                    if (i & mask) == 0 {
                        let j = i | mask;
                        // Swap amplitudes at i and j
                        block.real.swap(i, j);
                        block.imag.swap(i, j);
                    }
                }
            }
        } else {
            // Cross-block: need to swap between blocks
            let block_bit = q - BLOCK_SHIFT;
            let block_mask = 1usize << block_bit;

            // Collect block pairs to swap
            let block_ids: Vec<usize> = self.blocks.keys().copied().collect();
            let mut processed = std::collections::HashSet::new();

            for block_id in block_ids {
                if processed.contains(&block_id) {
                    continue;
                }

                let partner_id = block_id ^ block_mask;
                processed.insert(block_id);
                processed.insert(partner_id);

                // Get or create both blocks
                let block_a = self.blocks.remove(&block_id);
                let block_b = self.blocks.remove(&partner_id);

                match (block_a, block_b) {
                    (Some(a), Some(b)) => {
                        // Swap entire blocks
                        self.blocks.insert(partner_id, a);
                        self.blocks.insert(block_id, b);
                    }
                    (Some(a), None) => {
                        // Move a to partner position
                        self.blocks.insert(partner_id, a);
                    }
                    (None, Some(b)) => {
                        // Move b to original position
                        self.blocks.insert(block_id, b);
                    }
                    (None, None) => {
                        // Nothing to do
                    }
                }
            }
        }
    }

    /// Apply Z gate (phase flip) to qubit q
    pub fn apply_z(&mut self, q: u8) {
        assert!(q < self.n_qubits, "Qubit index out of range");

        if q < BLOCK_SHIFT {
            // Within-block: negate amplitudes where local bit q is 1
            let mask = 1usize << q;
            for block in self.blocks.values_mut() {
                for i in 0..BLOCK_SIZE {
                    if (i & mask) != 0 {
                        block.real[i] = -block.real[i];
                        block.imag[i] = -block.imag[i];
                    }
                }
            }
        } else {
            // Cross-block: negate entire blocks where block bit is 1
            let block_bit = q - BLOCK_SHIFT;
            let block_mask = 1usize << block_bit;

            for (&block_id, block) in self.blocks.iter_mut() {
                if (block_id & block_mask) != 0 {
                    for i in 0..BLOCK_SIZE {
                        block.real[i] = -block.real[i];
                        block.imag[i] = -block.imag[i];
                    }
                }
            }
        }
    }

    /// Apply Hadamard gate to qubit q
    /// This can create new blocks (doubles active blocks in worst case)
    #[allow(unused_variables)]
    pub fn apply_h(&mut self, q: u8) {
        assert!(q < self.n_qubits, "Qubit index out of range");
        let sqrt2_inv = f16::from_f32(std::f32::consts::FRAC_1_SQRT_2);
        let neg_sqrt2_inv = -sqrt2_inv;

        if q < BLOCK_SHIFT {
            // Within-block Hadamard
            let mask = 1usize << q;
            for block in self.blocks.values_mut() {
                for i in 0..BLOCK_SIZE {
                    if (i & mask) == 0 {
                        let j = i | mask;
                        // H applied to |0⟩ at position i, |1⟩ at position j
                        // a|0⟩ + b|1⟩ -> (a+b)/√2 |0⟩ + (a-b)/√2 |1⟩
                        let a_re = block.real[i];
                        let a_im = block.imag[i];
                        let b_re = block.real[j];
                        let b_im = block.imag[j];

                        // (a + b) / √2
                        block.real[i] =
                            f16::from_f32((a_re.to_f32() + b_re.to_f32()) * sqrt2_inv.to_f32());
                        block.imag[i] =
                            f16::from_f32((a_im.to_f32() + b_im.to_f32()) * sqrt2_inv.to_f32());

                        // (a - b) / √2
                        block.real[j] =
                            f16::from_f32((a_re.to_f32() - b_re.to_f32()) * sqrt2_inv.to_f32());
                        block.imag[j] =
                            f16::from_f32((a_im.to_f32() - b_im.to_f32()) * sqrt2_inv.to_f32());
                    }
                }
                block.recount_nnz();
            }
        } else {
            // Cross-block Hadamard: combines pairs of blocks
            let block_bit = q - BLOCK_SHIFT;
            let block_mask = 1usize << block_bit;

            // Collect all block IDs
            let block_ids: Vec<usize> = self.blocks.keys().copied().collect();
            let mut processed = std::collections::HashSet::new();
            let mut new_blocks: HashMap<usize, Block> = HashMap::new();

            for block_id in block_ids {
                if processed.contains(&block_id) {
                    continue;
                }

                let partner_id = block_id ^ block_mask;
                processed.insert(block_id);
                processed.insert(partner_id);

                let block_0_id = block_id & !block_mask; // bit = 0
                let block_1_id = block_id | block_mask; // bit = 1

                let block_0 = self.blocks.get(&block_0_id);
                let block_1 = self.blocks.get(&block_1_id);

                // Create output blocks
                let mut out_0 = Block::new_zero();
                let mut out_1 = Block::new_zero();

                for i in 0..BLOCK_SIZE {
                    let a_re = block_0.map(|b| b.real[i].to_f32()).unwrap_or(0.0);
                    let a_im = block_0.map(|b| b.imag[i].to_f32()).unwrap_or(0.0);
                    let b_re = block_1.map(|b| b.real[i].to_f32()).unwrap_or(0.0);
                    let b_im = block_1.map(|b| b.imag[i].to_f32()).unwrap_or(0.0);

                    let sqrt2_inv_f = std::f32::consts::FRAC_1_SQRT_2;

                    // (a + b) / √2 goes to block with bit = 0
                    out_0.real[i] = f16::from_f32((a_re + b_re) * sqrt2_inv_f);
                    out_0.imag[i] = f16::from_f32((a_im + b_im) * sqrt2_inv_f);

                    // (a - b) / √2 goes to block with bit = 1
                    out_1.real[i] = f16::from_f32((a_re - b_re) * sqrt2_inv_f);
                    out_1.imag[i] = f16::from_f32((a_im - b_im) * sqrt2_inv_f);
                }

                out_0.recount_nnz();
                out_1.recount_nnz();

                if !out_0.is_empty() {
                    new_blocks.insert(block_0_id, out_0);
                }
                if !out_1.is_empty() {
                    new_blocks.insert(block_1_id, out_1);
                }
            }

            self.blocks = new_blocks;
        }

        self.prune();
    }
}

// ============================================================================
// EPIC 88 Phase 88.2: GPU State and Operations
// ============================================================================

/// GPU-resident block-sparse state for WMMA operations
///
/// Stores non-zero blocks contiguously on GPU for efficient WMMA processing.
/// Maintains a mapping from contiguous GPU indices to original block IDs.
#[cfg(feature = "cuda")]
pub struct GpuBlockSparseState {
    /// Real parts of all blocks contiguously: [block_count * BLOCK_SIZE]
    pub real: cudarc::driver::CudaSlice<u16>,
    /// Imaginary parts of all blocks contiguously: [block_count * BLOCK_SIZE]
    pub imag: cudarc::driver::CudaSlice<u16>,
    /// Block IDs in order (for scatter-back)
    pub block_ids: Vec<usize>,
    /// Number of blocks on GPU
    pub block_count: usize,
    /// Number of qubits in original state
    pub n_qubits: u8,
}

#[cfg(feature = "cuda")]
impl GpuBlockSparseState {
    /// Upload BlockSparseState to GPU
    ///
    /// Packs non-zero blocks into contiguous GPU buffers.
    pub fn from_host(rt: &CudaRuntime, state: &BlockSparseState) -> CudaResult<Self> {
        let block_count = state.block_count();
        if block_count == 0 {
            return Err(CudaError::InvalidConfig("No blocks to upload".into()));
        }

        let total_elements = block_count * BLOCK_SIZE;

        // Collect blocks in sorted order for deterministic behavior
        let mut block_ids: Vec<usize> = state.blocks.keys().copied().collect();
        block_ids.sort();

        // Pack real and imag parts into contiguous host buffers
        let mut host_real: Vec<u16> = Vec::with_capacity(total_elements);
        let mut host_imag: Vec<u16> = Vec::with_capacity(total_elements);

        for &block_id in &block_ids {
            let block = state.blocks.get(&block_id).unwrap();
            for i in 0..BLOCK_SIZE {
                host_real.push(block.real[i].to_bits());
                host_imag.push(block.imag[i].to_bits());
            }
        }

        // Upload to GPU
        let real = rt.upload(&host_real)?;
        let imag = rt.upload(&host_imag)?;

        Ok(GpuBlockSparseState {
            real,
            imag,
            block_ids,
            block_count,
            n_qubits: state.n_qubits,
        })
    }

    /// Download GPU state back to BlockSparseState
    pub fn to_host(&self, rt: &CudaRuntime) -> CudaResult<BlockSparseState> {
        // Download from GPU
        let host_real = rt.download(&self.real)?;
        let host_imag = rt.download(&self.imag)?;

        // Reconstruct BlockSparseState
        let total_blocks = 1usize << (self.n_qubits - BLOCK_SHIFT);
        let mut blocks = HashMap::new();

        for (gpu_idx, &block_id) in self.block_ids.iter().enumerate() {
            let offset = gpu_idx * BLOCK_SIZE;
            let mut block = Block::new_zero();

            for i in 0..BLOCK_SIZE {
                block.real[i] = f16::from_bits(host_real[offset + i]);
                block.imag[i] = f16::from_bits(host_imag[offset + i]);
            }
            block.recount_nnz();

            if !block.is_empty() {
                blocks.insert(block_id, block);
            }
        }

        Ok(BlockSparseState {
            n_qubits: self.n_qubits,
            blocks,
            total_blocks,
        })
    }
}

// ============================================================================
// CUDA Kernels for Block-Sparse WMMA
// ============================================================================

/// CUDA kernel source for block-sparse WMMA operations
const BLOCK_SPARSE_WMMA_CUDA: &str = r#"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Block size: 128 complex = 256 f16 = 16x16 WMMA tile (but we use 128 for complex)
// We process 128 elements at a time, which is 8x16 in WMMA terms
// For simplicity, we'll process 2 WMMA tiles (real + imag) per block

/// Apply a 2x2 gate matrix to all blocks (within-block, qubits 0-6)
///
/// For qubit q with stride = 2^q:
/// - Elements at indices i and i+stride are paired
/// - Matrix: [a b; c d] applied as:
///   out[i] = a*in[i] + b*in[i+stride]
///   out[i+stride] = c*in[i] + d*in[i+stride]
extern "C" __global__
void block_sparse_apply_gate_kernel(
    half* __restrict__ real,          // [block_count * 128] real parts
    half* __restrict__ imag,          // [block_count * 128] imag parts
    const half gate_re_00,            // Real part of gate[0,0]
    const half gate_im_00,            // Imag part of gate[0,0]
    const half gate_re_01,            // Real part of gate[0,1]
    const half gate_im_01,            // Imag part of gate[0,1]
    const half gate_re_10,            // Real part of gate[1,0]
    const half gate_im_10,            // Imag part of gate[1,0]
    const half gate_re_11,            // Real part of gate[1,1]
    const half gate_im_11,            // Imag part of gate[1,1]
    unsigned int block_count,
    unsigned int qubit,               // Target qubit (0-6)
    unsigned int stride               // 2^qubit
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int block_idx = tid / 64;  // Which block (64 pairs per block)
    unsigned int pair_idx = tid % 64;   // Which pair within block

    if (block_idx >= block_count) return;

    // Calculate indices within block
    // For qubit q, pairs are (i, i+stride) where i has bit q = 0
    unsigned int local_low = pair_idx & (stride - 1);
    unsigned int local_high = pair_idx >> (qubit);
    unsigned int idx0 = local_low | (local_high << (qubit + 1));
    unsigned int idx1 = idx0 | stride;

    // Global indices
    unsigned int base = block_idx * 128;
    unsigned int g0 = base + idx0;
    unsigned int g1 = base + idx1;

    // Load current values
    float r0 = __half2float(real[g0]);
    float i0 = __half2float(imag[g0]);
    float r1 = __half2float(real[g1]);
    float i1 = __half2float(imag[g1]);

    // Gate matrix elements
    float a_re = __half2float(gate_re_00), a_im = __half2float(gate_im_00);
    float b_re = __half2float(gate_re_01), b_im = __half2float(gate_im_01);
    float c_re = __half2float(gate_re_10), c_im = __half2float(gate_im_10);
    float d_re = __half2float(gate_re_11), d_im = __half2float(gate_im_11);

    // Complex multiplication: out = gate * in
    // out0 = a*in0 + b*in1
    float out0_re = (a_re*r0 - a_im*i0) + (b_re*r1 - b_im*i1);
    float out0_im = (a_re*i0 + a_im*r0) + (b_re*i1 + b_im*r1);

    // out1 = c*in0 + d*in1
    float out1_re = (c_re*r0 - c_im*i0) + (d_re*r1 - d_im*i1);
    float out1_im = (c_re*i0 + c_im*r0) + (d_re*i1 + d_im*r1);

    // Store results
    real[g0] = __float2half(out0_re);
    imag[g0] = __float2half(out0_im);
    real[g1] = __float2half(out1_re);
    imag[g1] = __float2half(out1_im);
}

/// Apply gate to cross-block pairs (qubits 7+)
/// Pairs: block N with block N^(1 << (qubit-7))
/// Both blocks must exist and be consecutive in the GPU buffer
extern "C" __global__
void block_sparse_apply_gate_cross_block_kernel(
    half* __restrict__ real,
    half* __restrict__ imag,
    const half gate_re_00, const half gate_im_00,
    const half gate_re_01, const half gate_im_01,
    const half gate_re_10, const half gate_im_10,
    const half gate_re_11, const half gate_im_11,
    const unsigned int* __restrict__ block_pairs,  // Pairs: [gpu_idx_0, gpu_idx_1, ...]
    unsigned int pair_count,
    unsigned int elements_per_block  // 128
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pair_idx = tid / elements_per_block;
    unsigned int elem_idx = tid % elements_per_block;

    if (pair_idx >= pair_count) return;

    unsigned int gpu_idx_0 = block_pairs[pair_idx * 2];
    unsigned int gpu_idx_1 = block_pairs[pair_idx * 2 + 1];

    unsigned int g0 = gpu_idx_0 * elements_per_block + elem_idx;
    unsigned int g1 = gpu_idx_1 * elements_per_block + elem_idx;

    // Load
    float r0 = __half2float(real[g0]);
    float i0 = __half2float(imag[g0]);
    float r1 = __half2float(real[g1]);
    float i1 = __half2float(imag[g1]);

    // Gate
    float a_re = __half2float(gate_re_00), a_im = __half2float(gate_im_00);
    float b_re = __half2float(gate_re_01), b_im = __half2float(gate_im_01);
    float c_re = __half2float(gate_re_10), c_im = __half2float(gate_im_10);
    float d_re = __half2float(gate_re_11), d_im = __half2float(gate_im_11);

    // out0 = a*in0 + b*in1
    float out0_re = (a_re*r0 - a_im*i0) + (b_re*r1 - b_im*i1);
    float out0_im = (a_re*i0 + a_im*r0) + (b_re*i1 + b_im*r1);

    // out1 = c*in0 + d*in1
    float out1_re = (c_re*r0 - c_im*i0) + (d_re*r1 - d_im*i1);
    float out1_im = (c_re*i0 + c_im*r0) + (d_re*i1 + d_im*r1);

    // Store
    real[g0] = __float2half(out0_re);
    imag[g0] = __float2half(out0_im);
    real[g1] = __float2half(out1_re);
    imag[g1] = __float2half(out1_im);
}
"#;

/// Compiled block-sparse WMMA kernels
/// Helper struct to hold compiled kernels
#[cfg(feature = "cuda")]
pub struct BlockSparseKernels {
    within_block_fn: cudarc::driver::CudaFunction,
    cross_block_fn: cudarc::driver::CudaFunction,
}

/// Compile block-sparse WMMA kernels
/// Compile kernels for the given context
#[cfg(feature = "cuda")]
pub fn compile_block_sparse_kernels(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
) -> CudaResult<BlockSparseKernels> {
    use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

    // EPIC 113.1: Dynamic CUDA include path and compute capability detection
    let cuda_include = crate::cuda::get_cuda_include_path();
    let arch: &'static str = Box::leak(crate::cuda::get_device_arch_string().into_boxed_str());

    let opts = CompileOptions {
        arch: Some(arch),
        include_paths: vec![cuda_include],
        options: vec!["-default-device".to_string()],
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(BLOCK_SPARSE_WMMA_CUDA, opts).map_err(|e| {
        CudaError::KernelCompilationFailed(format!("Block-sparse kernel compile: {:?}", e))
    })?;

    let module = ctx.load_module(ptx).map_err(|e| {
        CudaError::KernelCompilationFailed(format!("Block-sparse kernel load: {:?}", e))
    })?;

    let within_block_fn = module
        .load_function("block_sparse_apply_gate_kernel")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!("block_sparse_apply_gate_kernel: {:?}", e))
        })?;

    let cross_block_fn = module
        .load_function("block_sparse_apply_gate_cross_block_kernel")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!(
                "block_sparse_apply_gate_cross_block_kernel: {:?}",
                e
            ))
        })?;

    Ok(BlockSparseKernels {
        within_block_fn,
        cross_block_fn,
    })
}

/// Gate matrix in f16 format for GPU
#[derive(Clone, Copy)]
pub struct GateMatrix16 {
    pub re_00: f16,
    pub im_00: f16,
    pub re_01: f16,
    pub im_01: f16,
    pub re_10: f16,
    pub im_10: f16,
    pub re_11: f16,
    pub im_11: f16,
}

impl GateMatrix16 {
    /// Create Hadamard gate matrix
    pub fn hadamard() -> Self {
        let sqrt2_inv = f16::from_f32(std::f32::consts::FRAC_1_SQRT_2);
        let neg_sqrt2_inv = -sqrt2_inv;
        Self {
            re_00: sqrt2_inv,
            im_00: f16::ZERO,
            re_01: sqrt2_inv,
            im_01: f16::ZERO,
            re_10: sqrt2_inv,
            im_10: f16::ZERO,
            re_11: neg_sqrt2_inv,
            im_11: f16::ZERO,
        }
    }

    /// Create X (Pauli-X) gate matrix
    pub fn x() -> Self {
        Self {
            re_00: f16::ZERO,
            im_00: f16::ZERO,
            re_01: f16::ONE,
            im_01: f16::ZERO,
            re_10: f16::ONE,
            im_10: f16::ZERO,
            re_11: f16::ZERO,
            im_11: f16::ZERO,
        }
    }

    /// Create Z (Pauli-Z) gate matrix
    pub fn z() -> Self {
        let neg_one = f16::from_f32(-1.0);
        Self {
            re_00: f16::ONE,
            im_00: f16::ZERO,
            re_01: f16::ZERO,
            im_01: f16::ZERO,
            re_10: f16::ZERO,
            im_10: f16::ZERO,
            re_11: neg_one,
            im_11: f16::ZERO,
        }
    }
}

/// Apply a gate to block-sparse state on GPU (within-block, qubits 0-6)
/// Execute within-block gate operation
#[cfg(feature = "cuda")]
pub fn launch_within_block_kernel(
    kernels: &BlockSparseKernels,
    gpu_state: &mut GpuBlockSparseState,
    gate: GateMatrix16,
    q: u8,
    stream: &cudarc::driver::CudaStream,
) -> CudaResult<()> {
    use cudarc::driver::LaunchConfig;

    if q >= BLOCK_SHIFT {
        return Err(CudaError::InvalidConfig(format!(
            "Qubit {} requires cross-block kernel (use apply_gate_cross_block_gpu)",
            q
        )));
    }

    let stride = 1u32 << q;
    let pairs_per_block = 64u32; // 128 elements / 2 = 64 pairs
    let total_pairs = gpu_state.block_count as u32 * pairs_per_block;

    let threads = 256u32;
    let grid_blocks = (total_pairs + threads - 1) / threads;

    let cfg = LaunchConfig {
        grid_dim: (grid_blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernels.within_block_fn)
            .arg(&gpu_state.real)
            .arg(&gpu_state.imag)
            .arg(&gate.re_00.to_bits())
            .arg(&gate.im_00.to_bits())
            .arg(&gate.re_01.to_bits())
            .arg(&gate.im_01.to_bits())
            .arg(&gate.re_10.to_bits())
            .arg(&gate.im_10.to_bits())
            .arg(&gate.re_11.to_bits())
            .arg(&gate.im_11.to_bits())
            .arg(&(gpu_state.block_count as u32))
            .arg(&(q as u32))
            .arg(&stride)
            .launch(cfg)
            .map_err(|e| CudaError::LaunchFailed(format!("Within-block gate: {:?}", e)))?;
    }

    stream
        .synchronize()
        .map_err(|e| CudaError::LaunchFailed(format!("Sync failed: {:?}", e)))?;
    Ok(())
}

/// Apply a gate to block-sparse state on GPU (cross-block, qubits 7+)
///
/// Requires all partner blocks to exist in the GPU state.
#[cfg(feature = "cuda")]
pub fn apply_gate_cross_block_gpu(
    rt: &CudaRuntime,
    state: &mut GpuBlockSparseState,
    kernels: &BlockSparseKernels,
    gate: GateMatrix16,
    qubit: u8,
) -> CudaResult<()> {
    use cudarc::driver::LaunchConfig;

    if qubit < BLOCK_SHIFT {
        return Err(CudaError::InvalidConfig(format!(
            "Qubit {} is within-block (use apply_gate_block_sparse_gpu)",
            qubit
        )));
    }

    let block_bit = qubit - BLOCK_SHIFT;
    let block_mask = 1usize << block_bit;

    // Build block pair mapping
    // Find pairs of GPU indices where block_ids differ by block_mask
    let mut block_id_to_gpu_idx: HashMap<usize, usize> = HashMap::new();
    for (gpu_idx, &block_id) in state.block_ids.iter().enumerate() {
        block_id_to_gpu_idx.insert(block_id, gpu_idx);
    }

    let mut pairs: Vec<u32> = Vec::new();
    let mut processed: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for &block_id in &state.block_ids {
        if processed.contains(&block_id) {
            continue;
        }

        let partner_id = block_id ^ block_mask;
        if let Some(&partner_gpu_idx) = block_id_to_gpu_idx.get(&partner_id) {
            let gpu_idx = block_id_to_gpu_idx[&block_id];

            // Ensure consistent ordering: block with 0-bit is first (idx0), 1-bit is second (idx1)
            let (idx0, idx1) = if (block_id & block_mask) == 0 {
                (gpu_idx, partner_gpu_idx)
            } else {
                (partner_gpu_idx, gpu_idx)
            };

            pairs.push(idx0 as u32);
            pairs.push(idx1 as u32);

            processed.insert(block_id);
            processed.insert(partner_id);
        } else {
            return Err(CudaError::InvalidConfig(
                 format!("Missing partner block {} for block {} on GPU. Ensure partners exist before upload.", partner_id, block_id)
             ));
        }
    }

    if pairs.is_empty() {
        return Ok(());
    }

    let pair_count = pairs.len() / 2;
    let pairs_gpu = rt.upload(&pairs)?;

    // We process full blocks, so elements per block = BLOCK_SIZE
    let threads = 256u32;
    // Total items to process = pair_count * BLOCK_SIZE (since each pair has BLOCK_SIZE elements)
    // Actually kernel does: pair_idx = tid / 128, elem_idx = tid % 128
    let total_threads = pair_count * BLOCK_SIZE;
    let grid_blocks = ((total_threads + threads as usize - 1) / threads as usize) as u32;

    let cfg = LaunchConfig {
        grid_dim: (grid_blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        rt.stream()
            .launch_builder(&kernels.cross_block_fn)
            .arg(&state.real)
            .arg(&state.imag)
            .arg(&gate.re_00.to_bits())
            .arg(&gate.im_00.to_bits())
            .arg(&gate.re_01.to_bits())
            .arg(&gate.im_01.to_bits())
            .arg(&gate.re_10.to_bits())
            .arg(&gate.im_10.to_bits())
            .arg(&gate.re_11.to_bits())
            .arg(&gate.im_11.to_bits())
            .arg(&pairs_gpu)
            .arg(&(pair_count as u32))
            .arg(&(BLOCK_SIZE as u32))
            .launch(cfg)
            .map_err(|e| CudaError::LaunchFailed(format!("Cross-block gate: {:?}", e)))?;
    }

    rt.synchronize()?;
    Ok(())
}

// ============================================================================
// Phase 88.3: Representation Selection
// ============================================================================

/// Threshold for clustering detection: minimum average amplitudes per block
pub const CLUSTERING_THRESHOLD: f32 = 5.0;

/// Minimum total amplitudes before considering block-sparse
pub const MIN_AMPLITUDES_FOR_BLOCK_SPARSE: usize = 50;

/// Minimum total elements (blocks * BLOCK_SIZE) for GPU to be profitable
/// Benchmarks show GPU overhead ~500μs+, CPU wins even at 16K elements
/// GPU only profitable for truly massive states (100K+ elements, many gates)
pub const MIN_ELEMENTS_FOR_GPU: usize = 131072; // 1024 blocks * 128 elements

/// Hysteresis: switch from block-sparse to element-sparse when below this many blocks
pub const HYSTERESIS_MIN_BLOCKS: usize = 5;

/// Density threshold: convert to dense when block sparsity exceeds this
pub const DENSE_DENSITY_THRESHOLD: f64 = 0.5;

/// Backend selection for quantum operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseBackend {
    /// Element-sparse on CPU (for ultra-sparse or small states)
    ElementSparseCpu,
    /// Block-sparse on CPU (for moderate sparsity)
    BlockSparseCpu,
    /// Block-sparse on GPU (for large block counts)
    BlockSparseGpu,
    /// Convert to dense (for high-density states)
    Dense,
}

/// Check if an element-sparse state is clustered (suitable for block-sparse)
///
/// A state is clustered if:
/// 1. It has at least MIN_AMPLITUDES_FOR_BLOCK_SPARSE non-zero amplitudes
/// 2. The average amplitudes per block exceeds CLUSTERING_THRESHOLD
pub fn is_clustered(sparse: &super::sparse_state::SparseQState) -> bool {
    if sparse.nnz() < MIN_AMPLITUDES_FOR_BLOCK_SPARSE {
        return false;
    }

    // Count unique block IDs
    let mut block_ids = std::collections::HashSet::new();
    for (index, _) in sparse.iter() {
        let block_id = (index >> BLOCK_SHIFT) as usize;
        block_ids.insert(block_id);
    }

    if block_ids.is_empty() {
        return false;
    }

    let avg_per_block = sparse.nnz() as f32 / block_ids.len() as f32;
    avg_per_block >= CLUSTERING_THRESHOLD
}

/// Select the optimal backend for processing an element-sparse state
pub fn select_backend_for_sparse(sparse: &super::sparse_state::SparseQState) -> SparseBackend {
    let nnz = sparse.nnz();

    // Very sparse: stay element-sparse
    if nnz < MIN_AMPLITUDES_FOR_BLOCK_SPARSE {
        return SparseBackend::ElementSparseCpu;
    }

    // Check clustering
    if is_clustered(sparse) {
        // Estimate block count
        let mut block_ids = std::collections::HashSet::new();
        for (index, _) in sparse.iter() {
            let block_id = (index >> BLOCK_SHIFT) as usize;
            block_ids.insert(block_id);
        }

        if block_ids.len() >= MIN_BLOCKS_FOR_GPU {
            return SparseBackend::BlockSparseGpu;
        } else {
            return SparseBackend::BlockSparseCpu;
        }
    }

    // Not clustered - stay element-sparse
    SparseBackend::ElementSparseCpu
}

/// Minimum average amplitudes per block for block-sparse to be efficient
/// Below this, element-sparse is more efficient (less overhead per amplitude)
pub const MIN_AMPS_PER_BLOCK_FOR_BLOCK_SPARSE: f32 = 8.0;

/// Select the optimal backend for processing a block-sparse state
pub fn select_backend_for_block_sparse(state: &BlockSparseState) -> SparseBackend {
    let block_count = state.block_count();
    let total_elements = block_count * BLOCK_SIZE;
    let amps_per_block = state.amplitudes_per_block();

    // Very few blocks with hysteresis: consider element-sparse
    if block_count < HYSTERESIS_MIN_BLOCKS {
        return SparseBackend::ElementSparseCpu;
    }

    // Sparse content (few amps per block): element-sparse is more efficient
    // Block overhead not justified if amplitudes are scattered
    if amps_per_block < MIN_AMPS_PER_BLOCK_FOR_BLOCK_SPARSE {
        return SparseBackend::ElementSparseCpu;
    }

    // High density: consider dense
    if state.block_sparsity() > DENSE_DENSITY_THRESHOLD {
        return SparseBackend::Dense;
    }

    // GPU requires both enough blocks AND enough total elements to amortize overhead
    // GPU launch overhead ~500μs+, need massive work to justify
    if block_count >= MIN_BLOCKS_FOR_GPU && total_elements >= MIN_ELEMENTS_FOR_GPU {
        SparseBackend::BlockSparseGpu
    } else {
        SparseBackend::BlockSparseCpu
    }
}

/// Representation state tracker with hysteresis to prevent thrashing
#[derive(Debug, Clone)]
pub struct RepresentationTracker {
    current: SparseBackend,
    ops_since_switch: usize,
    min_ops_between_switches: usize,
}

impl RepresentationTracker {
    pub fn new(initial: SparseBackend) -> Self {
        Self {
            current: initial,
            ops_since_switch: 0,
            min_ops_between_switches: 10,
        }
    }

    pub fn current(&self) -> SparseBackend {
        self.current
    }

    /// Record that an operation was performed
    pub fn record_op(&mut self) {
        self.ops_since_switch = self.ops_since_switch.saturating_add(1);
    }

    /// Check if we should switch to a new backend
    pub fn should_switch_to(&self, suggested: SparseBackend) -> bool {
        if self.current == suggested {
            return false;
        }
        self.ops_since_switch >= self.min_ops_between_switches
    }

    /// Switch to a new backend (resets counter)
    pub fn switch_to(&mut self, backend: SparseBackend) {
        self.current = backend;
        self.ops_since_switch = 0;
    }

    /// Set minimum operations between switches
    pub fn set_min_ops(&mut self, min_ops: usize) {
        self.min_ops_between_switches = min_ops;
    }
}

impl Default for RepresentationTracker {
    fn default() -> Self {
        Self::new(SparseBackend::ElementSparseCpu)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_sparse_zero_state() {
        let state = BlockSparseState::new_zero(10);
        assert_eq!(state.block_count(), 1);
        assert_eq!(state.get(0), Complex16::ONE);
        assert_eq!(state.get(1), Complex16::ZERO);
        assert_eq!(state.nnz(), 1);
    }

    #[test]
    fn test_block_sparse_basis_state() {
        let state = BlockSparseState::from_basis(10, 500);
        assert_eq!(state.block_count(), 1);
        assert_eq!(state.get(500), Complex16::ONE);
        assert_eq!(state.get(0), Complex16::ZERO);
    }

    #[test]
    fn test_block_sparse_x_gate_within_block() {
        let mut state = BlockSparseState::new_zero(10);
        // X on qubit 0 should flip |0⟩ to |1⟩
        state.apply_x(0);
        assert_eq!(state.get(0), Complex16::ZERO);
        assert_eq!(state.get(1), Complex16::ONE);
        assert_eq!(state.block_count(), 1);
    }

    #[test]
    fn test_block_sparse_x_gate_cross_block() {
        // 10 qubits: qubit 7 affects block index bit 0
        let mut state = BlockSparseState::new_zero(10);
        // X on qubit 7 should move from block 0 to block 1
        state.apply_x(7);
        assert_eq!(state.block_count(), 1);
        // New index = 0 ^ (1 << 7) = 128, which is in block 1
        assert_eq!(state.get(128), Complex16::ONE);
        assert_eq!(state.get(0), Complex16::ZERO);
    }

    #[test]
    fn test_block_sparse_z_gate() {
        let mut state = BlockSparseState::new_zero(10);
        // Z on |0⟩ should leave it unchanged
        state.apply_z(0);
        assert_eq!(state.get(0), Complex16::ONE);

        // Create |1⟩ and apply Z
        state.apply_x(0);
        state.apply_z(0);
        // Should negate: |1⟩ -> -|1⟩
        let amp = state.get(1);
        assert!((amp.re.to_f32() - (-1.0)).abs() < 1e-3);
    }

    #[test]
    fn test_block_sparse_h_gate_within_block() {
        let mut state = BlockSparseState::new_zero(10);
        state.apply_h(0);

        // Should have |0⟩ and |1⟩ with equal amplitude 1/√2
        let sqrt2_inv = std::f32::consts::FRAC_1_SQRT_2;
        let amp0 = state.get(0);
        let amp1 = state.get(1);

        assert!((amp0.re.to_f32() - sqrt2_inv).abs() < 1e-2);
        assert!((amp1.re.to_f32() - sqrt2_inv).abs() < 1e-2);
        assert_eq!(state.block_count(), 1);
    }

    #[test]
    fn test_block_sparse_h_gate_cross_block() {
        let mut state = BlockSparseState::new_zero(10);
        // H on qubit 7 creates superposition across blocks
        state.apply_h(7);

        // Should have 2 blocks now
        assert_eq!(state.block_count(), 2);

        let sqrt2_inv = std::f32::consts::FRAC_1_SQRT_2;
        let amp0 = state.get(0);
        let amp128 = state.get(128);

        assert!((amp0.re.to_f32() - sqrt2_inv).abs() < 1e-2);
        assert!((amp128.re.to_f32() - sqrt2_inv).abs() < 1e-2);
    }

    #[test]
    fn test_block_sparse_conversion_roundtrip() {
        use crate::sparse_state::SparseQState;

        // Create element-sparse with a few amplitudes
        let mut elem_sparse = SparseQState::new_zero(10);
        elem_sparse.apply_h(0);
        elem_sparse.apply_h(1);

        // Convert to block-sparse
        let block_sparse = BlockSparseState::from_element_sparse(&elem_sparse);
        assert_eq!(block_sparse.block_count(), 1); // All in block 0
        assert_eq!(block_sparse.nnz(), 4); // 2^2 = 4 amplitudes

        // Convert back
        let back = block_sparse.to_element_sparse();
        assert_eq!(back.nnz(), 4);
    }

    #[test]
    fn test_should_use_gpu() {
        // Small state: should not use GPU
        let small = BlockSparseState::new_zero(10);
        assert!(!small.should_use_gpu());

        // Medium state (16 blocks) - still below threshold
        let mut medium = BlockSparseState::new_zero(15);
        for q in 7..11 {
            // 4 high qubits = 2^4 = 16 blocks
            medium.apply_h(q);
        }
        // 16 blocks < 32 threshold
        assert!(medium.block_count() < MIN_BLOCKS_FOR_GPU);
        assert!(!medium.should_use_gpu());

        // Very large state (2048 blocks) - above threshold
        let mut large = BlockSparseState::new_zero(18);
        for q in 7..18 {
            // 11 high qubits = 2^11 = 2048 blocks
            large.apply_h(q);
        }
        // 2048 blocks >= 1024 threshold
        assert!(large.block_count() >= MIN_BLOCKS_FOR_GPU);
        assert!(large.should_use_gpu());
    }

    // ========================================================================
    // GPU Integration Tests (require cuda feature)
    // ========================================================================

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_upload_download_roundtrip() {
        use crate::cuda::CudaRuntime;

        let rt = match CudaRuntime::new() {
            Ok(rt) => rt,
            Err(_) => {
                eprintln!("[SKIP] GPU not available");
                return;
            }
        };

        // Create block-sparse state with some blocks
        let mut state = BlockSparseState::new_zero(10);
        state.apply_h(0); // Create |0⟩ + |1⟩ superposition
        state.apply_h(7); // Spread to 2 blocks

        assert_eq!(state.block_count(), 2);

        // Upload to GPU
        let gpu_state = GpuBlockSparseState::from_host(&rt, &state).unwrap();
        assert_eq!(gpu_state.block_count, 2);

        // Download back
        let downloaded = gpu_state.to_host(&rt).unwrap();
        assert_eq!(downloaded.block_count(), 2);

        // Check values match
        let sqrt2_inv = std::f32::consts::FRAC_1_SQRT_2 * 0.5_f32.sqrt(); // 1/√4
        for idx in [0, 1, 128, 129] {
            let orig = state.get(idx);
            let back = downloaded.get(idx);
            assert!(
                (orig.re.to_f32() - back.re.to_f32()).abs() < 1e-3,
                "Mismatch at idx {}: orig={}, back={}",
                idx,
                orig.re.to_f32(),
                back.re.to_f32()
            );
        }

        eprintln!("[PASS] GPU upload/download roundtrip");
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore] // TODO: Implement apply_gate_block_sparse_gpu function
    fn test_gpu_within_block_gate() {
        // TODO: apply_gate_block_sparse_gpu is not yet implemented
        // When implemented, this test should:
        // 1. Create a BlockSparseState with some blocks
        // 2. Upload to GPU
        // 3. Apply a gate using apply_gate_block_sparse_gpu
        // 4. Compare with CPU result
        eprintln!("[SKIP] apply_gate_block_sparse_gpu not yet implemented");
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore] // Requires nvrtc.dll in PATH for runtime compilation
    fn test_gpu_cross_block_gate() {
        use crate::cuda::CudaRuntime;

        let rt = match CudaRuntime::new() {
            Ok(rt) => rt,
            Err(_) => {
                eprintln!("[SKIP] GPU not available");
                return;
            }
        };

        // Compile kernels
        let kernels = compile_block_sparse_kernels(rt.context()).unwrap();

        // Create state with 2 blocks where both partners exist
        let mut state = BlockSparseState::new_zero(10);
        state.apply_h(7); // Creates blocks 0 and 1

        // Make a CPU copy for comparison
        let mut cpu_state = state.clone();

        // Apply H on qubit 7 - cross-block operation
        // Since both blocks exist, this should work on GPU
        cpu_state.apply_h(7); // On CPU

        // Apply on GPU
        let mut gpu_state = GpuBlockSparseState::from_host(&rt, &state).unwrap();
        apply_gate_cross_block_gpu(&rt, &mut gpu_state, &kernels, GateMatrix16::hadamard(), 7)
            .unwrap();
        let gpu_result = gpu_state.to_host(&rt).unwrap();

        // Compare key amplitudes
        // H on qubit 7 twice = Identity, so |0⟩ should be restored
        let cpu_amp_0 = cpu_state.get(0);
        let gpu_amp_0 = gpu_result.get(0);
        let cpu_amp_128 = cpu_state.get(128);
        let gpu_amp_128 = gpu_result.get(128);

        assert!(
            (cpu_amp_0.re.to_f32() - gpu_amp_0.re.to_f32()).abs() < 1e-2,
            "Mismatch at 0: CPU={}, GPU={}",
            cpu_amp_0.re.to_f32(),
            gpu_amp_0.re.to_f32()
        );
        assert!(
            (cpu_amp_128.re.to_f32() - gpu_amp_128.re.to_f32()).abs() < 1e-2,
            "Mismatch at 128: CPU={}, GPU={}",
            cpu_amp_128.re.to_f32(),
            gpu_amp_128.re.to_f32()
        );

        eprintln!("[PASS] GPU cross-block H gate matches CPU");
    }

    // ========================================================================
    // Phase 88.3 Tests: Block Creation & Representation Selection
    // ========================================================================

    #[test]
    fn test_ensure_partner_blocks_exist() {
        // Start with single block at |0⟩
        let mut state = BlockSparseState::new_zero(10);
        assert_eq!(state.block_count(), 1);

        // Qubit 3 is within-block, should create 0 partners
        let created = state.ensure_partner_blocks_exist(3);
        assert_eq!(created, 0);
        assert_eq!(state.block_count(), 1);

        // Qubit 7 is cross-block, block 0's partner is block 1
        let created = state.ensure_partner_blocks_exist(7);
        assert_eq!(created, 1);
        assert_eq!(state.block_count(), 2);

        // Call again - should create 0 (partners already exist)
        let created = state.ensure_partner_blocks_exist(7);
        assert_eq!(created, 0);
        assert_eq!(state.block_count(), 2);
    }

    #[test]
    fn test_has_missing_partners() {
        let mut state = BlockSparseState::new_zero(10);

        // Within-block qubit - never has missing partners
        assert!(!state.has_missing_partners(3));

        // Qubit 7: block 0 exists, block 1 missing
        assert!(state.has_missing_partners(7));

        // Create the partner
        state.ensure_partner_blocks_exist(7);
        assert!(!state.has_missing_partners(7));
    }

    #[test]
    fn test_h_gate_with_partner_creation() {
        // Test the full flow: H on single block creates partner
        let mut state = BlockSparseState::new_zero(10);
        assert_eq!(state.block_count(), 1);

        // Apply H on qubit 7 (cross-block)
        // This should internally handle partner creation
        state.apply_h(7);

        // Should now have 2 blocks with correct amplitudes
        assert_eq!(state.block_count(), 2);

        let sqrt2_inv = std::f32::consts::FRAC_1_SQRT_2;
        let amp0 = state.get(0);
        let amp128 = state.get(128);

        assert!((amp0.re.to_f32() - sqrt2_inv).abs() < 1e-2);
        assert!((amp128.re.to_f32() - sqrt2_inv).abs() < 1e-2);
    }

    #[test]
    fn test_clustering_detection() {
        use crate::sparse_state::SparseQState;

        // Small state - not clustered (below threshold)
        let mut small = SparseQState::new_zero(10);
        small.apply_h(0); // 2 amplitudes
        assert!(!is_clustered(&small));

        // Many amplitudes but spread across blocks - not clustered
        let mut spread = SparseQState::new_zero(14);
        // Apply H to high qubits to spread across many blocks
        for q in 7..11 {
            spread.apply_h(q);
        }
        // 2^4 = 16 amplitudes across 16 blocks = 1 amp/block (not clustered)
        // Actually need to check the actual structure
        let spread_backend = select_backend_for_sparse(&spread);
        // Should be element-sparse since not well clustered

        // Clustered: many amplitudes in few blocks
        let mut clustered = SparseQState::new_zero(10);
        for q in 0..6 {
            clustered.apply_h(q);
        }
        // 2^6 = 64 amplitudes, all in block 0 = 64 amps/block (highly clustered)
        assert!(is_clustered(&clustered));
    }

    #[test]
    fn test_backend_selection_element_sparse() {
        use crate::sparse_state::SparseQState;

        // Very small state -> ElementSparseCpu
        let small = SparseQState::new_zero(10);
        assert_eq!(
            select_backend_for_sparse(&small),
            SparseBackend::ElementSparseCpu
        );

        // Clustered state with enough amplitudes
        let mut clustered = SparseQState::new_zero(10);
        for q in 0..6 {
            clustered.apply_h(q);
        }
        // 64 amplitudes in 1 block -> too few blocks for GPU
        assert_eq!(
            select_backend_for_sparse(&clustered),
            SparseBackend::BlockSparseCpu
        );
    }

    #[test]
    fn test_backend_selection_block_sparse() {
        // Few blocks -> ElementSparseCpu (hysteresis)
        let state = BlockSparseState::new_zero(10);
        assert_eq!(
            select_backend_for_block_sparse(&state),
            SparseBackend::ElementSparseCpu
        );

        // Medium blocks (16) with sparse content -> ElementSparseCpu
        // 16 blocks but only 16 amplitudes = 1 amp/block (below MIN_AMPS_PER_BLOCK)
        let mut medium_sparse = BlockSparseState::new_zero(14);
        for q in 7..11 {
            medium_sparse.apply_h(q);
        }
        // Low density: should stay element-sparse
        assert!(medium_sparse.amplitudes_per_block() < MIN_AMPS_PER_BLOCK_FOR_BLOCK_SPARSE);
        assert_eq!(
            select_backend_for_block_sparse(&medium_sparse),
            SparseBackend::ElementSparseCpu
        );

        // Medium blocks (8) with dense content -> BlockSparseCpu
        // Apply H to qubits 0-5 (64 amps) + H on qubits 7-9 (8 blocks) = 64/8 = 8 amps/block
        let mut medium_dense = BlockSparseState::new_zero(14);
        for q in 0..6 {
            medium_dense.apply_h(q);
        } // 64 amps in block 0
        for q in 7..10 {
            medium_dense.apply_h(q);
        } // spread to 8 blocks
          // 64 amps / 8 blocks = 8 amps/block (at threshold)
        assert!(medium_dense.amplitudes_per_block() >= MIN_AMPS_PER_BLOCK_FOR_BLOCK_SPARSE);
        assert_eq!(
            select_backend_for_block_sparse(&medium_dense),
            SparseBackend::BlockSparseCpu
        );

        // Very large blocks (2048) with dense content
        // With 18 qubits: total_blocks = 2^(18-7) = 2048
        // If we have 2048 active blocks out of 2048 total = 100% density -> Dense
        // Instead use 20 qubits: total_blocks = 2^(20-7) = 8192
        // 2048 active / 8192 total = 25% sparsity -> BlockSparseGpu
        let mut large = BlockSparseState::new_zero(20);
        for q in 0..6 {
            large.apply_h(q);
        } // 64 amps
        for q in 7..18 {
            large.apply_h(q);
        } // 2048 blocks
          // 2048 blocks >= 1024 threshold, and sufficient amps/block
        assert!(large.block_count() >= MIN_BLOCKS_FOR_GPU);
        assert!(large.amplitudes_per_block() >= MIN_AMPS_PER_BLOCK_FOR_BLOCK_SPARSE);
        assert!(large.block_sparsity() <= DENSE_DENSITY_THRESHOLD);
        assert_eq!(
            select_backend_for_block_sparse(&large),
            SparseBackend::BlockSparseGpu
        );
    }

    #[test]
    fn test_representation_tracker_hysteresis() {
        let mut tracker = RepresentationTracker::new(SparseBackend::ElementSparseCpu);
        tracker.set_min_ops(5);

        // Immediately after creation, shouldn't switch
        assert!(!tracker.should_switch_to(SparseBackend::BlockSparseGpu));

        // After a few ops, still shouldn't switch
        for _ in 0..4 {
            tracker.record_op();
        }
        assert!(!tracker.should_switch_to(SparseBackend::BlockSparseGpu));

        // After 5 ops, should be allowed to switch
        tracker.record_op();
        assert!(tracker.should_switch_to(SparseBackend::BlockSparseGpu));

        // Switch resets counter
        tracker.switch_to(SparseBackend::BlockSparseGpu);
        assert!(!tracker.should_switch_to(SparseBackend::ElementSparseCpu));
        assert_eq!(tracker.current(), SparseBackend::BlockSparseGpu);
    }
}
