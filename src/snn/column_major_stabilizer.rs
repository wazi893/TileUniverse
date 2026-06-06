//! Column-Major Stabilizer Tableau with CSR Synapses
//!
//! This implementation uses a GPU-optimal memory layout:
//! - Column-major storage: each qubit's X/Z bits across all rows are contiguous
//! - CSR synapse format: compressed sparse row for efficient batch operations
//!
//! Key insight: CZ gates operate on COLUMNS (same qubit, all rows).
//! Column-major layout makes these operations coalesced memory accesses.

use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::cuda::{CudaError, CudaResult, CudaRuntime};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
#[cfg(feature = "cuda")]
use logic_fabric_core::cuda::get_cuda_include_path;

/// Memory allocation mode for GPU stabilizer tableau
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMemoryMode {
    /// Device memory only (fast, limited to VRAM)
    DeviceOnly,
    /// Unified memory (slower, can use system RAM)
    Unified,
}

/// Synapse connectivity in CSR (Compressed Sparse Row) format
///
/// For each source qubit, stores the list of destination qubits it connects to.
/// This is the standard sparse matrix format that GPUs are optimized for.
#[derive(Debug, Clone)]
pub struct SynapseCSR {
    /// Row pointers: row_ptrs[i] is the start index in `destinations` for qubit i
    /// row_ptrs[n_qubits] is the total number of directed edges
    pub row_ptrs: Vec<u32>,
    /// Destination qubits for each edge
    pub destinations: Vec<u32>,
    /// Number of qubits
    pub n_qubits: usize,
    /// Number of undirected synapses (directed edges = 2x this)
    pub n_synapses: usize,
}

impl SynapseCSR {
    /// Create CSR from a list of (source, destination) pairs
    ///
    /// Note: CZ is symmetric, so we automatically add both directions
    pub fn from_pairs(n_qubits: usize, pairs: &[(usize, usize)]) -> Self {
        // Count edges per source (both directions for symmetric CZ)
        let mut counts = vec![0u32; n_qubits];
        for &(src, dst) in pairs {
            if src < n_qubits && dst < n_qubits && src != dst {
                counts[src] += 1;
                counts[dst] += 1; // Symmetric
            }
        }

        // Build row_ptrs (prefix sum)
        let mut row_ptrs = vec![0u32; n_qubits + 1];
        for i in 0..n_qubits {
            row_ptrs[i + 1] = row_ptrs[i] + counts[i];
        }

        // Build destinations
        let total_edges = row_ptrs[n_qubits] as usize;
        let mut destinations = vec![0u32; total_edges];
        let mut current_pos = row_ptrs.clone();

        for &(src, dst) in pairs {
            if src < n_qubits && dst < n_qubits && src != dst {
                // Forward edge: src -> dst
                let pos = current_pos[src] as usize;
                destinations[pos] = dst as u32;
                current_pos[src] += 1;

                // Backward edge: dst -> src (CZ is symmetric)
                let pos = current_pos[dst] as usize;
                destinations[pos] = src as u32;
                current_pos[dst] += 1;
            }
        }

        Self {
            row_ptrs,
            destinations,
            n_qubits,
            n_synapses: pairs.len(),
        }
    }

    /// Create CSR from layer connectivity (feedforward network)
    pub fn from_layers(
        n_neurons: usize,
        layers: &[(usize, usize)],
        connectivity: f32,
        seed: u64,
    ) -> Self {
        use crate::qec::noise::SimpleRng;
        let mut rng = SimpleRng::new(seed);
        let mut pairs = Vec::new();

        // Connect consecutive layers
        for window in layers.windows(2) {
            let (start1, end1) = window[0];
            let (start2, end2) = window[1];

            for src in start1..end1 {
                for dst in start2..end2 {
                    if rng.next_f64() < connectivity as f64 {
                        pairs.push((src, dst));
                    }
                }
            }
        }

        Self::from_pairs(n_neurons, &pairs)
    }

    /// Get statistics
    pub fn stats(&self) -> SynapseCSRStats {
        let directed_edges = self.destinations.len();
        let max_degree = (0..self.n_qubits)
            .map(|i| self.row_ptrs[i + 1] - self.row_ptrs[i])
            .max()
            .unwrap_or(0);
        let avg_degree = if self.n_qubits > 0 {
            directed_edges as f32 / self.n_qubits as f32
        } else {
            0.0
        };

        SynapseCSRStats {
            n_qubits: self.n_qubits,
            n_synapses: self.n_synapses,
            directed_edges,
            max_degree: max_degree as usize,
            avg_degree,
            memory_bytes: self.row_ptrs.len() * 4 + self.destinations.len() * 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynapseCSRStats {
    pub n_qubits: usize,
    pub n_synapses: usize,
    pub directed_edges: usize,
    pub max_degree: usize,
    pub avg_degree: f32,
    pub memory_bytes: usize,
}

// =============================================================================
// CUDA Implementation
// =============================================================================

#[cfg(feature = "cuda")]
const COLUMN_MAJOR_CUDA_SRC: &str = r#"
extern "C" {

// =============================================================================
// BATCH CZ KERNEL - Column-Major Layout
//
// Each thread block handles one source qubit.
// Threads within block parallelize over row_words.
// Memory access is fully coalesced!
// =============================================================================
__global__ void batch_cz_column_major(
    unsigned long long* __restrict__ z_cols,    // [n_qubits × n_row_words] column-major
    const unsigned long long* __restrict__ x_cols, // [n_qubits × n_row_words] column-major
    const unsigned int* __restrict__ row_ptrs,  // [n_qubits + 1] CSR row pointers
    const unsigned int* __restrict__ destinations, // [n_edges] destination qubits
    int n_qubits,
    int n_row_words
) {
    int src_qubit = blockIdx.x;
    if (src_qubit >= n_qubits) return;

    int edge_start = row_ptrs[src_qubit];
    int edge_end = row_ptrs[src_qubit + 1];

    // Base index for source qubit's Z column
    int src_base = src_qubit * n_row_words;

    // Process all destinations for this source
    for (int e = edge_start; e < edge_end; e++) {
        int dst_qubit = destinations[e];
        int dst_base = dst_qubit * n_row_words;

        // XOR X column of dst into Z column of src
        // Each thread handles different row_words - COALESCED ACCESS!
        for (int w = threadIdx.x; w < n_row_words; w += blockDim.x) {
            z_cols[src_base + w] ^= x_cols[dst_base + w];
        }
    }
}

// =============================================================================
// OPTIMIZED BATCH CZ - Shared Memory Caching
//
// Cache destination X columns in shared memory for reuse
// Better for high-degree nodes
// =============================================================================
#define SMEM_WORDS 64

__global__ void batch_cz_column_major_v2(
    unsigned long long* __restrict__ z_cols,
    const unsigned long long* __restrict__ x_cols,
    const unsigned int* __restrict__ row_ptrs,
    const unsigned int* __restrict__ destinations,
    int n_qubits,
    int n_row_words
) {
    __shared__ unsigned long long s_x_cache[SMEM_WORDS];

    int src_qubit = blockIdx.x;
    if (src_qubit >= n_qubits) return;

    int edge_start = row_ptrs[src_qubit];
    int edge_end = row_ptrs[src_qubit + 1];
    int src_base = src_qubit * n_row_words;

    // Process row_words in tiles
    for (int w_start = 0; w_start < n_row_words; w_start += SMEM_WORDS) {
        int w_end = min(w_start + SMEM_WORDS, n_row_words);
        int tile_size = w_end - w_start;

        // Load src Z column tile into registers
        unsigned long long z_local = 0;
        int w_local = threadIdx.x;
        if (w_local < tile_size) {
            z_local = z_cols[src_base + w_start + w_local];
        }

        // Process all destinations
        for (int e = edge_start; e < edge_end; e++) {
            int dst_qubit = destinations[e];
            int dst_base = dst_qubit * n_row_words;

            // Cooperatively load dst X column tile into shared memory
            if (threadIdx.x < tile_size) {
                s_x_cache[threadIdx.x] = x_cols[dst_base + w_start + threadIdx.x];
            }
            __syncthreads();

            // XOR into local Z
            if (w_local < tile_size) {
                z_local ^= s_x_cache[w_local];
            }
            __syncthreads();
        }

        // Write back
        if (w_local < tile_size) {
            z_cols[src_base + w_start + w_local] = z_local;
        }
    }
}

// =============================================================================
// PHASE UPDATE KERNEL
//
// After CZ application, update phases based on X bit products
// This is done separately for better memory access patterns
// =============================================================================
__global__ void update_phases_column_major(
    unsigned char* __restrict__ phases,         // [n_rows]
    const unsigned long long* __restrict__ x_cols, // [n_qubits × n_row_words]
    const unsigned int* __restrict__ row_ptrs,
    const unsigned int* __restrict__ destinations,
    int n_qubits,
    int n_rows,
    int n_row_words
) {
    // Each thread handles one row
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int word_idx = row / 64;
    int bit_idx = row % 64;
    unsigned long long bit_mask = 1ULL << bit_idx;

    unsigned int phase_delta = 0;

    // For each source qubit
    for (int src = 0; src < n_qubits; src++) {
        int edge_start = row_ptrs[src];
        int edge_end = row_ptrs[src + 1];

        // Get X bit for source
        unsigned long long x_src_word = x_cols[src * n_row_words + word_idx];
        int x_src = (x_src_word >> bit_idx) & 1;

        if (x_src == 0) continue;  // Optimization: skip if X[src] = 0

        // Check all destinations
        for (int e = edge_start; e < edge_end; e++) {
            int dst = destinations[e];

            // Only count once per pair (src < dst)
            if (dst <= src) continue;

            unsigned long long x_dst_word = x_cols[dst * n_row_words + word_idx];
            int x_dst = (x_dst_word >> bit_idx) & 1;

            if (x_dst) {
                phase_delta += 2;
            }
        }
    }

    // Update phase (mod 4)
    if (phase_delta > 0) {
        phases[row] = (phases[row] + (phase_delta & 3)) & 3;
    }
}

// =============================================================================
// HADAMARD KERNEL - Column-Major
// Swap X and Z columns for specified qubits
// =============================================================================
__global__ void hadamard_column_major(
    unsigned long long* __restrict__ x_cols,
    unsigned long long* __restrict__ z_cols,
    const unsigned int* __restrict__ qubits,
    int n_qubits_to_apply,
    int n_row_words
) {
    int qubit_idx = blockIdx.x;
    if (qubit_idx >= n_qubits_to_apply) return;

    int qubit = qubits[qubit_idx];
    int base = qubit * n_row_words;

    // Swap X and Z columns
    for (int w = threadIdx.x; w < n_row_words; w += blockDim.x) {
        unsigned long long tmp = x_cols[base + w];
        x_cols[base + w] = z_cols[base + w];
        z_cols[base + w] = tmp;
    }
}

// =============================================================================
// INITIALIZATION KERNEL - Set to |+⟩^n state
// X columns = 0, Z columns = identity-like pattern
// =============================================================================
__global__ void init_plus_state_column_major(
    unsigned long long* __restrict__ x_cols,
    unsigned long long* __restrict__ z_cols,
    unsigned char* __restrict__ phases,
    int n_qubits,
    int n_rows,
    int n_row_words
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_words = n_qubits * n_row_words;

    // Initialize X columns (destabilizers have X=1 on diagonal)
    if (idx < total_words) {
        int qubit = idx / n_row_words;
        int word = idx % n_row_words;

        // Destabilizer rows [0, n) have X[i,i] = 1
        // Stabilizer rows [n, 2n) have X = 0
        int row_in_word_start = word * 64;
        unsigned long long x_val = 0;

        // Check if diagonal element falls in this word
        if (qubit >= row_in_word_start && qubit < row_in_word_start + 64) {
            x_val = 1ULL << (qubit - row_in_word_start);
        }
        x_cols[idx] = x_val;
    }

    // Initialize Z columns (stabilizers have Z=1 on diagonal)
    if (idx < total_words) {
        int qubit = idx / n_row_words;
        int word = idx % n_row_words;

        // Stabilizer rows [n, 2n) have Z[i,i-n] = 1
        int row_in_word_start = word * 64;
        unsigned long long z_val = 0;

        int stab_row = qubit + n_qubits;  // Row in stabilizer section
        if (stab_row >= row_in_word_start && stab_row < row_in_word_start + 64) {
            z_val = 1ULL << (stab_row - row_in_word_start);
        }
        z_cols[idx] = z_val;
    }

    // Initialize phases to 0
    if (idx < n_rows) {
        phases[idx] = 0;
    }
}

} // extern "C"
"#;

#[cfg(feature = "cuda")]
struct ColumnMajorKernels {
    batch_cz_fn: cudarc::driver::CudaFunction,
    batch_cz_v2_fn: cudarc::driver::CudaFunction,
    update_phases_fn: cudarc::driver::CudaFunction,
    hadamard_fn: cudarc::driver::CudaFunction,
    init_plus_state_fn: cudarc::driver::CudaFunction,
}

#[cfg(feature = "cuda")]
fn load_column_major_kernels(rt: &CudaRuntime) -> CudaResult<ColumnMajorKernels> {
    // Get CUDA include path
    let cuda_include = get_cuda_include_path();
    let arch: &'static str = "compute_89";

    let opts = CompileOptions {
        arch: Some(arch),
        include_paths: vec![cuda_include],
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(COLUMN_MAJOR_CUDA_SRC, opts).map_err(|e| {
        CudaError::KernelCompilationFailed(format!("Column-major kernel compile error: {:?}", e))
    })?;

    let ctx = rt.ctx();
    let module = ctx.load_module(ptx).map_err(|e| {
        CudaError::KernelCompilationFailed(format!("Column-major kernel load error: {:?}", e))
    })?;

    let batch_cz_fn = module
        .load_function("batch_cz_column_major")
        .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

    let batch_cz_v2_fn = module
        .load_function("batch_cz_column_major_v2")
        .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

    let update_phases_fn = module
        .load_function("update_phases_column_major")
        .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

    let hadamard_fn = module
        .load_function("hadamard_column_major")
        .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

    let init_plus_state_fn = module
        .load_function("init_plus_state_column_major")
        .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

    Ok(ColumnMajorKernels {
        batch_cz_fn,
        batch_cz_v2_fn,
        update_phases_fn,
        hadamard_fn,
        init_plus_state_fn,
    })
}

/// GPU-accelerated stabilizer tableau with column-major layout
#[cfg(feature = "cuda")]
pub struct GpuColumnMajorTableau {
    rt: Arc<CudaRuntime>,
    kernels: ColumnMajorKernels,

    // GPU buffers - column major layout
    x_cols_gpu: CudaSlice<u64>,
    z_cols_gpu: CudaSlice<u64>,
    phases_gpu: CudaSlice<u8>,

    // CSR synapse data on GPU
    row_ptrs_gpu: CudaSlice<u32>,
    destinations_gpu: CudaSlice<u32>,

    // Dimensions
    n_qubits: usize,
    n_rows: usize,      // = 2 * n_qubits
    n_row_words: usize, // = (n_rows + 63) / 64
    n_edges: usize,

    // Memory mode
    #[allow(dead_code)]
    memory_mode: GpuMemoryMode,
}

#[cfg(feature = "cuda")]
impl GpuColumnMajorTableau {
    /// Create a new column-major tableau with CSR synapses (device memory only)
    pub fn new(rt: Arc<CudaRuntime>, n_qubits: usize, synapses: &SynapseCSR) -> CudaResult<Self> {
        Self::with_memory_mode(rt, n_qubits, synapses, GpuMemoryMode::DeviceOnly)
    }

    /// Create with unified memory (can use system RAM, slower but unlimited size)
    ///
    /// NOTE: Currently falls back to device memory. Full unified memory support
    /// requires refactoring to use cudarc's UnifiedSlice type.
    pub fn new_unified(
        rt: Arc<CudaRuntime>,
        n_qubits: usize,
        synapses: &SynapseCSR,
    ) -> CudaResult<Self> {
        // For now, use device memory with a warning
        // TODO: Implement proper UnifiedSlice support
        let n_rows = 2 * n_qubits;
        let n_row_words = (n_rows + 63) / 64;
        let total_words = n_qubits * n_row_words;
        let tableau_bytes = total_words * 8 * 2 + n_rows;
        let tableau_gb = tableau_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

        eprintln!(
            "[Unified Memory] Requesting {:.2} GB for {} qubits - attempting device allocation",
            tableau_gb, n_qubits
        );

        Self::with_memory_mode(rt, n_qubits, synapses, GpuMemoryMode::Unified)
    }

    /// Create with specified memory mode
    pub fn with_memory_mode(
        rt: Arc<CudaRuntime>,
        n_qubits: usize,
        synapses: &SynapseCSR,
        memory_mode: GpuMemoryMode,
    ) -> CudaResult<Self> {
        let kernels = load_column_major_kernels(&rt)?;

        let n_rows = 2 * n_qubits;
        let n_row_words = (n_rows + 63) / 64;
        let total_words = n_qubits * n_row_words;

        // Allocate GPU buffers (device memory for now, unified memory TODO)
        let x_cols_gpu = rt.alloc_zeros(total_words)?;
        let z_cols_gpu = rt.alloc_zeros(total_words)?;
        let phases_gpu = rt.alloc_zeros(n_rows)?;

        // Upload CSR data
        let row_ptrs_gpu = rt.upload(&synapses.row_ptrs)?;
        let destinations_gpu = rt.upload(&synapses.destinations)?;

        let mut tableau = Self {
            rt,
            kernels,
            x_cols_gpu,
            z_cols_gpu,
            phases_gpu,
            row_ptrs_gpu,
            destinations_gpu,
            n_qubits,
            n_rows,
            n_row_words,
            n_edges: synapses.destinations.len(),
            memory_mode,
        };

        // Initialize to |+⟩^n state
        tableau.init_plus_state()?;

        Ok(tableau)
    }

    /// Initialize to |+⟩^n state
    pub fn init_plus_state(&mut self) -> CudaResult<()> {
        let total_words = self.n_qubits * self.n_row_words;
        let threads = 256u32;
        let blocks =
            ((total_words.max(self.n_rows) + threads as usize - 1) / threads as usize) as u32;

        let n_qubits_i32 = self.n_qubits as i32;
        let n_rows_i32 = self.n_rows as i32;
        let n_row_words_i32 = self.n_row_words as i32;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernels.init_plus_state_fn)
                .arg(&self.x_cols_gpu)
                .arg(&self.z_cols_gpu)
                .arg(&self.phases_gpu)
                .arg(&n_qubits_i32)
                .arg(&n_rows_i32)
                .arg(&n_row_words_i32)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Apply all CZ gates defined by the CSR synapses
    ///
    /// This is the main operation - applies CZ to all synapse pairs in parallel
    pub fn apply_all_cz(&mut self) -> CudaResult<()> {
        if self.n_edges == 0 {
            return Ok(());
        }

        let n_qubits_i32 = self.n_qubits as i32;
        let n_row_words_i32 = self.n_row_words as i32;

        // One block per source qubit, threads parallelize over row_words
        let threads = 256u32.min(self.n_row_words as u32).max(32);
        let blocks = self.n_qubits as u32;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernels.batch_cz_fn)
                .arg(&self.z_cols_gpu)
                .arg(&self.x_cols_gpu)
                .arg(&self.row_ptrs_gpu)
                .arg(&self.destinations_gpu)
                .arg(&n_qubits_i32)
                .arg(&n_row_words_i32)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Apply all CZ gates using optimized kernel with shared memory
    pub fn apply_all_cz_v2(&mut self) -> CudaResult<()> {
        if self.n_edges == 0 {
            return Ok(());
        }

        let n_qubits_i32 = self.n_qubits as i32;
        let n_row_words_i32 = self.n_row_words as i32;

        let threads = 64u32; // Match SMEM_WORDS
        let blocks = self.n_qubits as u32;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 64 * 8, // 64 u64 words
        };

        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernels.batch_cz_v2_fn)
                .arg(&self.z_cols_gpu)
                .arg(&self.x_cols_gpu)
                .arg(&self.row_ptrs_gpu)
                .arg(&self.destinations_gpu)
                .arg(&n_qubits_i32)
                .arg(&n_row_words_i32)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Update phases after CZ application
    ///
    /// Note: For many use cases, phase tracking can be deferred or skipped
    pub fn update_phases(&mut self) -> CudaResult<()> {
        let threads = 256u32;
        let blocks = ((self.n_rows + threads as usize - 1) / threads as usize) as u32;

        let n_qubits_i32 = self.n_qubits as i32;
        let n_rows_i32 = self.n_rows as i32;
        let n_row_words_i32 = self.n_row_words as i32;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernels.update_phases_fn)
                .arg(&self.phases_gpu)
                .arg(&self.x_cols_gpu)
                .arg(&self.row_ptrs_gpu)
                .arg(&self.destinations_gpu)
                .arg(&n_qubits_i32)
                .arg(&n_rows_i32)
                .arg(&n_row_words_i32)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Apply Hadamard to specified qubits
    pub fn hadamard(&mut self, qubits: &[u32]) -> CudaResult<()> {
        if qubits.is_empty() {
            return Ok(());
        }

        let qubits_gpu = self.rt.upload(qubits)?;
        let n_qubits_to_apply = qubits.len() as i32;
        let n_row_words_i32 = self.n_row_words as i32;

        let threads = 256u32.min(self.n_row_words as u32).max(32);
        let blocks = qubits.len() as u32;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernels.hadamard_fn)
                .arg(&self.x_cols_gpu)
                .arg(&self.z_cols_gpu)
                .arg(&qubits_gpu)
                .arg(&n_qubits_to_apply)
                .arg(&n_row_words_i32)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Synchronize GPU operations
    pub fn sync(&self) -> CudaResult<()> {
        self.rt
            .stream()
            .synchronize()
            .map_err(|e| CudaError::DeviceError(format!("Sync failed: {:?}", e)))
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let words = self.n_qubits * self.n_row_words;
        // X + Z columns + phases + CSR
        words * 8 * 2 + self.n_rows + self.n_qubits * 4 + self.n_edges * 4
    }

    /// Get tableau dimensions
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.n_qubits, self.n_rows, self.n_row_words)
    }
}

// CPU fallback when CUDA not available
#[cfg(not(feature = "cuda"))]
pub struct GpuColumnMajorTableau {
    _phantom: std::marker::PhantomData<()>,
}

#[cfg(not(feature = "cuda"))]
impl GpuColumnMajorTableau {
    pub fn new(_rt: Arc<()>, _n_qubits: usize, _synapses: &SynapseCSR) -> Result<Self, String> {
        Err("CUDA not available".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_from_pairs() {
        // Simple 4-qubit network: 0-1, 1-2, 2-3
        let pairs = vec![(0, 1), (1, 2), (2, 3)];
        let csr = SynapseCSR::from_pairs(4, &pairs);

        assert_eq!(csr.n_qubits, 4);
        assert_eq!(csr.n_synapses, 3);
        // Each edge is bidirectional, so 6 directed edges
        assert_eq!(csr.destinations.len(), 6);

        let stats = csr.stats();
        println!("CSR stats: {:?}", stats);
        assert_eq!(stats.directed_edges, 6);
    }

    #[test]
    fn test_csr_from_layers() {
        let csr = SynapseCSR::from_layers(100, &[(0, 10), (10, 50), (50, 100)], 0.1, 42);

        let stats = csr.stats();
        println!("Layer CSR stats: {:?}", stats);
        assert!(stats.n_synapses > 0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_column_major_creation() {
        use crate::cuda::CudaRuntime;

        let rt = match CudaRuntime::new() {
            Ok(rt) => Arc::new(rt),
            Err(_) => return, // Skip if no GPU
        };

        let csr = SynapseCSR::from_pairs(64, &[(0, 1), (1, 2), (2, 3)]);
        let tableau = GpuColumnMajorTableau::new(rt, 64, &csr);

        assert!(tableau.is_ok());
        let tableau = tableau.unwrap();
        println!("Memory: {} bytes", tableau.memory_bytes());
    }
}
