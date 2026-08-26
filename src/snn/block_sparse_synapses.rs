//! Block Sparse Synapse Map for GPU-Accelerated Stabilizer Networks
//!
//! This module implements structured sparsity patterns (2:4 and 4:8) for
//! efficient GPU execution of entangling synapses.
//!
//! # Sparsity Patterns
//!
//! - **2:4 Sparsity**: 2 active elements per 4-element group (50% density)
//!   - NVIDIA Ampere+ GPUs have native hardware support
//!   - Optimal for small block sizes (4x4 neuron blocks)
//!
//! - **4:8 Sparsity**: 4 active elements per 8-element group (50% density)
//!   - Better for larger block sizes (8x8 neuron blocks)
//!   - Maps well to GPU warp operations
//!
//! # Memory Layout
//!
//! ```text
//! BlockSparseSynapseMap:
//!   block_size: 8 (neurons per block)
//!   n_blocks: ceil(n_neurons / block_size)
//!
//!   block_pairs: [(src_block, dst_block), ...]  // Active block pairs
//!   synapse_masks: [u64, ...]                   // 8x8=64 bits per block pair
//!
//! For 2:4 pattern within 8x8 block:
//!   - Divide into 16 groups of 4 synapses
//!   - Each group has exactly 2 active (enforced on construction)
//!   - Total: 32 active synapses per 64 possible = 50%
//!
//! For 4:8 pattern within 8x8 block:
//!   - Divide into 8 groups of 8 synapses
//!   - Each group has exactly 4 active
//!   - Total: 32 active synapses per 64 possible = 50%
//! ```
//!
//! # GPU Kernel Strategy
//!
//! 1. Each CUDA thread block processes one block pair
//! 2. Load source/target neuron X/Z bits into shared memory
//! 3. Unpack synapse mask and apply CZ gates
//! 4. Write back updated Z bits with atomic ops if needed

use crate::qec::noise::SimpleRng;

/// Sparsity pattern for synapse blocks
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparsityPattern {
    /// 2 active per 4 elements (NVIDIA native support)
    TwoFour,
    /// 4 active per 8 elements
    FourEight,
    /// Custom density (0.0-1.0)
    Custom(u8), // density * 255
}

impl SparsityPattern {
    /// Get the density (fraction of active synapses)
    pub fn density(&self) -> f32 {
        match self {
            SparsityPattern::TwoFour => 0.5,
            SparsityPattern::FourEight => 0.5,
            SparsityPattern::Custom(d) => *d as f32 / 255.0,
        }
    }

    /// Get the group size for this pattern
    pub fn group_size(&self) -> usize {
        match self {
            SparsityPattern::TwoFour => 4,
            SparsityPattern::FourEight => 8,
            SparsityPattern::Custom(_) => 1, // No grouping for custom
        }
    }

    /// Get active count per group
    pub fn active_per_group(&self) -> usize {
        match self {
            SparsityPattern::TwoFour => 2,
            SparsityPattern::FourEight => 4,
            SparsityPattern::Custom(d) => (*d as usize).min(255),
        }
    }
}

/// A block pair representing connections between two neuron blocks
#[derive(Clone, Copy, Debug)]
pub struct BlockPair {
    /// Source block index
    pub src_block: u16,
    /// Destination block index
    pub dst_block: u16,
}

impl BlockPair {
    pub fn new(src: usize, dst: usize) -> Self {
        Self {
            src_block: src as u16,
            dst_block: dst as u16,
        }
    }
}

/// Block sparse synapse map
///
/// Organizes synapses into blocks for efficient GPU processing.
/// Each block pair has a fixed sparsity pattern.
#[derive(Clone, Debug)]
pub struct BlockSparseSynapseMap {
    /// Number of neurons per block (typically 8)
    pub block_size: usize,

    /// Total number of neurons
    pub n_neurons: usize,

    /// Number of blocks
    pub n_blocks: usize,

    /// Sparsity pattern used
    pub pattern: SparsityPattern,

    /// Active block pairs (src_block, dst_block)
    pub block_pairs: Vec<BlockPair>,

    /// Synapse masks for each block pair (64 bits for 8x8 block)
    /// Bit i*8+j = 1 means synapse from src[i] to dst[j] is active
    pub synapse_masks: Vec<u64>,

    /// Total number of active synapses
    pub n_synapses: usize,
}

impl BlockSparseSynapseMap {
    /// Create a new block sparse synapse map
    pub fn new(n_neurons: usize, block_size: usize, pattern: SparsityPattern) -> Self {
        let n_blocks = n_neurons.div_ceil(block_size);

        Self {
            block_size,
            n_neurons,
            n_blocks,
            pattern,
            block_pairs: Vec::new(),
            synapse_masks: Vec::new(),
            n_synapses: 0,
        }
    }

    /// Create block sparse map from layer connectivity specification
    ///
    /// # Arguments
    /// * `layers` - List of (start, end) ranges for each layer
    /// * `connectivity` - Target connectivity between adjacent layers
    /// * `pattern` - Sparsity pattern to enforce
    /// * `seed` - Random seed for synapse selection
    pub fn from_layers(
        n_neurons: usize,
        layers: &[(usize, usize)],
        connectivity: f32,
        pattern: SparsityPattern,
        seed: u64,
    ) -> Self {
        let block_size = 8; // Optimal for GPU
        let mut map = Self::new(n_neurons, block_size, pattern);
        let mut rng = SimpleRng::new(seed);

        // Connect adjacent layers
        for i in 0..layers.len().saturating_sub(1) {
            let (src_start, src_end) = layers[i];
            let (dst_start, dst_end) = layers[i + 1];

            map.add_layer_connections(
                src_start,
                src_end,
                dst_start,
                dst_end,
                connectivity,
                &mut rng,
            );
        }

        map
    }

    /// Add connections between two layer ranges
    fn add_layer_connections(
        &mut self,
        src_start: usize,
        src_end: usize,
        dst_start: usize,
        dst_end: usize,
        connectivity: f32,
        rng: &mut SimpleRng,
    ) {
        let src_block_start = src_start / self.block_size;
        let src_block_end = src_end.div_ceil(self.block_size);
        let dst_block_start = dst_start / self.block_size;
        let dst_block_end = dst_end.div_ceil(self.block_size);

        // Determine which block pairs should be active
        // Use connectivity to decide block-level sparsity
        let block_connectivity = (connectivity * 2.0).min(1.0); // Higher than synapse connectivity

        for src_block in src_block_start..src_block_end {
            for dst_block in dst_block_start..dst_block_end {
                if rng.next_f64() < block_connectivity as f64 {
                    let mask = self.generate_sparse_mask(
                        src_block,
                        dst_block,
                        src_start,
                        src_end,
                        dst_start,
                        dst_end,
                        connectivity,
                        rng,
                    );

                    if mask != 0 {
                        self.block_pairs.push(BlockPair::new(src_block, dst_block));
                        self.synapse_masks.push(mask);
                        self.n_synapses += mask.count_ones() as usize;
                    }
                }
            }
        }
    }

    /// Generate a sparse mask for a block pair following the sparsity pattern
    fn generate_sparse_mask(
        &self,
        src_block: usize,
        dst_block: usize,
        src_start: usize,
        src_end: usize,
        dst_start: usize,
        dst_end: usize,
        connectivity: f32,
        rng: &mut SimpleRng,
    ) -> u64 {
        let block_src_start = src_block * self.block_size;
        let block_src_end = ((src_block + 1) * self.block_size).min(self.n_neurons);
        let block_dst_start = dst_block * self.block_size;
        let block_dst_end = ((dst_block + 1) * self.block_size).min(self.n_neurons);

        match self.pattern {
            SparsityPattern::TwoFour => self.generate_2_4_mask(
                block_src_start,
                block_src_end,
                block_dst_start,
                block_dst_end,
                src_start,
                src_end,
                dst_start,
                dst_end,
                connectivity,
                rng,
            ),
            SparsityPattern::FourEight => self.generate_4_8_mask(
                block_src_start,
                block_src_end,
                block_dst_start,
                block_dst_end,
                src_start,
                src_end,
                dst_start,
                dst_end,
                connectivity,
                rng,
            ),
            SparsityPattern::Custom(density) => self.generate_custom_mask(
                block_src_start,
                block_src_end,
                block_dst_start,
                block_dst_end,
                src_start,
                src_end,
                dst_start,
                dst_end,
                density as f32 / 255.0,
                rng,
            ),
        }
    }

    /// Generate 2:4 sparse mask (2 active per 4 elements)
    fn generate_2_4_mask(
        &self,
        block_src_start: usize,
        block_src_end: usize,
        block_dst_start: usize,
        block_dst_end: usize,
        layer_src_start: usize,
        layer_src_end: usize,
        layer_dst_start: usize,
        layer_dst_end: usize,
        _connectivity: f32,
        rng: &mut SimpleRng,
    ) -> u64 {
        let mut mask = 0u64;

        // Process in groups of 4 synapses
        // For 8x8 block, we have 64 synapses = 16 groups of 4
        for group in 0..16 {
            let group_start = group * 4;

            // Check which synapses in this group are valid (within layer bounds)
            let mut valid_indices = Vec::new();
            for offset in 0..4 {
                let bit_idx = group_start + offset;
                let src_local = bit_idx / 8;
                let dst_local = bit_idx % 8;
                let src_global = block_src_start + src_local;
                let dst_global = block_dst_start + dst_local;

                // Check bounds
                if src_global >= block_src_end || dst_global >= block_dst_end {
                    continue;
                }
                if src_global < layer_src_start || src_global >= layer_src_end {
                    continue;
                }
                if dst_global < layer_dst_start || dst_global >= layer_dst_end {
                    continue;
                }

                valid_indices.push(offset);
            }

            // Select 2 random indices from valid ones (or all if < 2)
            let n_active = 2.min(valid_indices.len());
            if n_active == 0 {
                continue;
            }

            // Fisher-Yates partial shuffle to select n_active
            let mut selected = valid_indices.clone();
            for i in 0..n_active {
                let j = i + (rng.next_u64() as usize % (selected.len() - i));
                selected.swap(i, j);
            }

            // Set the selected bits
            for i in 0..n_active {
                let bit_idx = group_start + selected[i];
                mask |= 1u64 << bit_idx;
            }
        }

        mask
    }

    /// Generate 4:8 sparse mask (4 active per 8 elements)
    fn generate_4_8_mask(
        &self,
        block_src_start: usize,
        block_src_end: usize,
        block_dst_start: usize,
        block_dst_end: usize,
        layer_src_start: usize,
        layer_src_end: usize,
        layer_dst_start: usize,
        layer_dst_end: usize,
        _connectivity: f32,
        rng: &mut SimpleRng,
    ) -> u64 {
        let mut mask = 0u64;

        // Process in groups of 8 synapses
        // For 8x8 block, we have 64 synapses = 8 groups of 8
        for group in 0..8 {
            let group_start = group * 8;

            // Check which synapses in this group are valid
            let mut valid_indices = Vec::new();
            for offset in 0..8 {
                let bit_idx = group_start + offset;
                let src_local = bit_idx / 8;
                let dst_local = bit_idx % 8;
                let src_global = block_src_start + src_local;
                let dst_global = block_dst_start + dst_local;

                if src_global >= block_src_end || dst_global >= block_dst_end {
                    continue;
                }
                if src_global < layer_src_start || src_global >= layer_src_end {
                    continue;
                }
                if dst_global < layer_dst_start || dst_global >= layer_dst_end {
                    continue;
                }

                valid_indices.push(offset);
            }

            // Select 4 random indices from valid ones
            let n_active = 4.min(valid_indices.len());
            if n_active == 0 {
                continue;
            }

            let mut selected = valid_indices.clone();
            for i in 0..n_active {
                let j = i + (rng.next_u64() as usize % (selected.len() - i));
                selected.swap(i, j);
            }

            for i in 0..n_active {
                let bit_idx = group_start + selected[i];
                mask |= 1u64 << bit_idx;
            }
        }

        mask
    }

    /// Generate custom density mask
    fn generate_custom_mask(
        &self,
        block_src_start: usize,
        block_src_end: usize,
        block_dst_start: usize,
        block_dst_end: usize,
        layer_src_start: usize,
        layer_src_end: usize,
        layer_dst_start: usize,
        layer_dst_end: usize,
        density: f32,
        rng: &mut SimpleRng,
    ) -> u64 {
        let mut mask = 0u64;

        for src_local in 0..self.block_size {
            for dst_local in 0..self.block_size {
                let src_global = block_src_start + src_local;
                let dst_global = block_dst_start + dst_local;

                // Check bounds
                if src_global >= block_src_end || dst_global >= block_dst_end {
                    continue;
                }
                if src_global < layer_src_start || src_global >= layer_src_end {
                    continue;
                }
                if dst_global < layer_dst_start || dst_global >= layer_dst_end {
                    continue;
                }

                // Random selection based on density
                if rng.next_f64() < density as f64 {
                    let bit_idx = src_local * 8 + dst_local;
                    mask |= 1u64 << bit_idx;
                }
            }
        }

        mask
    }

    /// Get the list of active synapse pairs (source_neuron, target_neuron)
    pub fn iter_synapses(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.block_pairs
            .iter()
            .zip(self.synapse_masks.iter())
            .flat_map(move |(pair, &mask)| {
                let src_base = pair.src_block as usize * self.block_size;
                let dst_base = pair.dst_block as usize * self.block_size;
                let block_size = self.block_size;

                (0..64).filter_map(move |bit| {
                    if mask & (1u64 << bit) != 0 {
                        let src_local = bit / 8;
                        let dst_local = bit % 8;
                        let src = src_base + src_local;
                        let dst = dst_base + dst_local;
                        if src_local < block_size && dst_local < block_size {
                            Some((src, dst))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
            })
    }

    /// Get statistics about the sparse map
    pub fn stats(&self) -> BlockSparseStats {
        let total_possible = self.block_pairs.len() * 64;
        let actual_density = if total_possible > 0 {
            self.n_synapses as f32 / total_possible as f32
        } else {
            0.0
        };

        BlockSparseStats {
            n_neurons: self.n_neurons,
            n_blocks: self.n_blocks,
            block_size: self.block_size,
            n_block_pairs: self.block_pairs.len(),
            n_synapses: self.n_synapses,
            pattern: self.pattern,
            actual_density,
            memory_bytes: self.memory_bytes(),
        }
    }

    /// Estimate memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        // block_pairs: 4 bytes each (2x u16)
        // synapse_masks: 8 bytes each (u64)
        self.block_pairs.len() * 4 + self.synapse_masks.len() * 8 + 32 // struct overhead
    }

    /// Get data ready for GPU upload
    pub fn gpu_data(&self) -> BlockSparseGpuData {
        // Flatten block pairs into separate arrays for coalesced GPU access
        let src_blocks: Vec<u16> = self.block_pairs.iter().map(|p| p.src_block).collect();
        let dst_blocks: Vec<u16> = self.block_pairs.iter().map(|p| p.dst_block).collect();

        BlockSparseGpuData {
            block_size: self.block_size as u32,
            n_block_pairs: self.block_pairs.len() as u32,
            src_blocks,
            dst_blocks,
            synapse_masks: self.synapse_masks.clone(),
        }
    }
}

/// Statistics about a block sparse synapse map
#[derive(Clone, Debug)]
pub struct BlockSparseStats {
    pub n_neurons: usize,
    pub n_blocks: usize,
    pub block_size: usize,
    pub n_block_pairs: usize,
    pub n_synapses: usize,
    pub pattern: SparsityPattern,
    pub actual_density: f32,
    pub memory_bytes: usize,
}

/// GPU-ready data for block sparse synapses
#[derive(Clone, Debug)]
pub struct BlockSparseGpuData {
    pub block_size: u32,
    pub n_block_pairs: u32,
    pub src_blocks: Vec<u16>,
    pub dst_blocks: Vec<u16>,
    pub synapse_masks: Vec<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2_4_sparsity_pattern() {
        let layers = vec![(0, 16), (16, 32)];
        let map =
            BlockSparseSynapseMap::from_layers(32, &layers, 0.5, SparsityPattern::TwoFour, 42);

        let stats = map.stats();
        println!("2:4 Pattern Stats: {:?}", stats);

        // Check that density is approximately 50%
        assert!(stats.actual_density > 0.3 && stats.actual_density < 0.7);

        // Verify 2:4 constraint: each group of 4 should have at most 2 active
        for &mask in &map.synapse_masks {
            for group in 0..16 {
                let group_bits = (mask >> (group * 4)) & 0xF;
                let count = group_bits.count_ones();
                assert!(
                    count <= 2,
                    "Group {} has {} active, expected <= 2",
                    group,
                    count
                );
            }
        }
    }

    #[test]
    fn test_4_8_sparsity_pattern() {
        let layers = vec![(0, 16), (16, 32)];
        let map =
            BlockSparseSynapseMap::from_layers(32, &layers, 0.5, SparsityPattern::FourEight, 42);

        let stats = map.stats();
        println!("4:8 Pattern Stats: {:?}", stats);

        // Check density
        assert!(stats.actual_density > 0.3 && stats.actual_density < 0.7);

        // Verify 4:8 constraint
        for &mask in &map.synapse_masks {
            for group in 0..8 {
                let group_bits = (mask >> (group * 8)) & 0xFF;
                let count = group_bits.count_ones();
                assert!(
                    count <= 4,
                    "Group {} has {} active, expected <= 4",
                    group,
                    count
                );
            }
        }
    }

    #[test]
    fn test_synapse_iteration() {
        let layers = vec![(0, 8), (8, 16)];
        let map =
            BlockSparseSynapseMap::from_layers(16, &layers, 0.5, SparsityPattern::TwoFour, 42);

        let synapses: Vec<(usize, usize)> = map.iter_synapses().collect();

        println!("Synapses: {:?}", synapses);
        assert_eq!(synapses.len(), map.n_synapses);

        // All synapses should be from layer 0 to layer 1
        for (src, dst) in synapses {
            assert!(src < 8, "Source {} should be in layer 0", src);
            assert!((8..16).contains(&dst), "Dest {} should be in layer 1", dst);
        }
    }

    #[test]
    fn test_large_network() {
        // 10K neuron network
        let layers = vec![(0, 100), (100, 5000), (5000, 10000), (10000, 10100)];
        let map =
            BlockSparseSynapseMap::from_layers(10100, &layers, 0.1, SparsityPattern::FourEight, 42);

        let stats = map.stats();
        println!("Large Network Stats: {:?}", stats);

        // Should have reasonable synapse count
        assert!(stats.n_synapses > 10000);
        // Block sparse still has many synapses, but memory is compact
        assert!(stats.n_synapses < 10_000_000);

        // Memory should be manageable
        assert!(stats.memory_bytes < 10 * 1024 * 1024); // < 10MB
    }

    #[test]
    fn test_gpu_data_preparation() {
        let layers = vec![(0, 16), (16, 32)];
        let map =
            BlockSparseSynapseMap::from_layers(32, &layers, 0.5, SparsityPattern::TwoFour, 42);

        let gpu_data = map.gpu_data();

        assert_eq!(gpu_data.n_block_pairs as usize, map.block_pairs.len());
        assert_eq!(gpu_data.src_blocks.len(), gpu_data.dst_blocks.len());
        assert_eq!(gpu_data.synapse_masks.len(), map.synapse_masks.len());
    }
}
