//! CUDA GPU Acceleration for P-bit Networks
//!
//! Implements GPU-accelerated Ising model simulation using p-bits.
//! Expected speedup: 20-100x for networks with 1K+ p-bits.
//!
//! # Architecture
//!
//! - Dense coupling matrix J stored in row-major order
//! - One thread per p-bit for local field computation and updates
//! - Simple LCG PRNG per thread for stochastic updates
//! - All state GPU-resident for zero-transfer-per-tick operation

use crate::cuda::{CudaError, CudaResult, CudaRuntime};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use super::ising::IsingProblem;
use super::network::UpdateOrder;

/// CUDA kernel source code for p-bit operations
const PBIT_KERNEL_CUDA: &str = r#"
extern "C" {

// Fast sigmoid approximation using tanh
// sigmoid(x) = 0.5 * (1 + tanh(x/2))
__device__ __forceinline__ float fast_sigmoid(float x) {
    // Use CUDA's fast tanh
    return 0.5f * (1.0f + tanhf(0.5f * x));
}

// Kernel 1: Compute local fields for all p-bits
// I_i = h_i + Σⱼ J_ij * s_j
// Dense matrix-vector multiply: one thread per p-bit (row)
__global__ void pbit_compute_fields(
    const float* __restrict__ J,      // [n*n] coupling matrix (row-major)
    const float* __restrict__ h,      // [n] bias vector
    const signed char* __restrict__ spins,  // [n] current spins (-1 or +1)
    float* __restrict__ fields,       // [n] output local fields
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float field = h[i];
    const float* J_row = J + i * n;

    // Unrolled dot product for better performance
    float sum = 0.0f;
    unsigned int j = 0;

    // Process 4 elements at a time
    for (; j + 3 < n; j += 4) {
        sum += J_row[j]     * (float)spins[j];
        sum += J_row[j + 1] * (float)spins[j + 1];
        sum += J_row[j + 2] * (float)spins[j + 2];
        sum += J_row[j + 3] * (float)spins[j + 3];
    }

    // Handle remainder
    for (; j < n; j++) {
        sum += J_row[j] * (float)spins[j];
    }

    fields[i] = field + sum;
}

// Kernel 2: Update p-bits using computed fields
// P(s_i = +1) = sigmoid(beta * I_i)
__global__ void pbit_update(
    const float* __restrict__ fields, // [n] local fields
    signed char* __restrict__ spins,  // [n] spins to update
    unsigned char* __restrict__ states, // [n] binary states (0 or 1)
    float beta,                       // inverse temperature
    unsigned int seed,                // random seed for this step
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Simple LCG PRNG per thread (from SNN kernels)
    unsigned int rng = seed ^ (i * 2654435761u);
    rng = rng * 1664525u + 1013904223u;
    float rand_val = (float)(rng >> 8) / 16777216.0f;  // [0, 1)

    // Sigmoid activation
    float prob = fast_sigmoid(beta * fields[i]);

    // Stochastic update
    if (rand_val < prob) {
        spins[i] = 1;
        states[i] = 1;
    } else {
        spins[i] = -1;
        states[i] = 0;
    }
}

// Kernel 3: Compute Ising energy
// E = -Σᵢ<ⱼ J_ij s_i s_j - Σᵢ h_i s_i
// Uses parallel reduction
__global__ void pbit_compute_energy(
    const float* __restrict__ J,      // [n*n] coupling matrix
    const float* __restrict__ h,      // [n] bias vector
    const signed char* __restrict__ spins, // [n] current spins
    float* __restrict__ partial_energy,   // [gridDim.x] partial sums
    unsigned int n
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float my_energy = 0.0f;

    if (i < n) {
        // Bias term: -h_i * s_i
        my_energy -= h[i] * (float)spins[i];

        // Coupling term: -Σⱼ>ᵢ J_ij s_i s_j (only upper triangle)
        float si = (float)spins[i];
        const float* J_row = J + i * n;

        for (unsigned int j = i + 1; j < n; j++) {
            my_energy -= J_row[j] * si * (float)spins[j];
        }
    }

    sdata[tid] = my_energy;
    __syncthreads();

    // Parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_energy[blockIdx.x] = sdata[0];
    }
}

// Kernel 4: Final reduction for energy
__global__ void pbit_reduce_energy(
    const float* __restrict__ partial_energy,
    float* __restrict__ total_energy,
    unsigned int n_partials
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    float sum = 0.0f;

    for (unsigned int i = tid; i < n_partials; i += blockDim.x) {
        sum += partial_energy[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        total_energy[0] = sdata[0];
    }
}

// Kernel 5: Random update order (shuffle indices)
// Fisher-Yates shuffle on GPU
__global__ void pbit_shuffle_indices(
    unsigned int* __restrict__ indices,
    unsigned int seed,
    unsigned int n
) {
    // Single thread for now (small n)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int rng = seed;
    for (unsigned int i = n - 1; i > 0; i--) {
        rng = rng * 1664525u + 1013904223u;
        unsigned int j = rng % (i + 1);

        // Swap
        unsigned int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

// Kernel 6: Sequential update with random order
// Updates p-bits one at a time in the given order
__global__ void pbit_update_sequential(
    const float* __restrict__ J,      // [n*n] coupling matrix
    const float* __restrict__ h,      // [n] bias vector
    signed char* __restrict__ spins,  // [n] spins (modified in place)
    unsigned char* __restrict__ states, // [n] binary states
    const unsigned int* __restrict__ order, // [n] update order
    float beta,
    unsigned int seed,
    unsigned int n
) {
    // Single thread executes sequential updates
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int rng = seed;

    for (unsigned int k = 0; k < n; k++) {
        unsigned int i = order[k];

        // Compute local field
        float field = h[i];
        const float* J_row = J + i * n;
        for (unsigned int j = 0; j < n; j++) {
            field += J_row[j] * (float)spins[j];
        }

        // Random number
        rng = rng * 1664525u + 1013904223u;
        float rand_val = (float)(rng >> 8) / 16777216.0f;

        // Update
        float prob = fast_sigmoid(beta * field);
        if (rand_val < prob) {
            spins[i] = 1;
            states[i] = 1;
        } else {
            spins[i] = -1;
            states[i] = 0;
        }
    }
}

} // extern "C"
"#;

/// Compiled p-bit kernel functions
struct PBitKernels {
    compute_fields_fn: cudarc::driver::CudaFunction,
    update_fn: cudarc::driver::CudaFunction,
    compute_energy_fn: cudarc::driver::CudaFunction,
    reduce_energy_fn: cudarc::driver::CudaFunction,
    shuffle_fn: cudarc::driver::CudaFunction,
    update_sequential_fn: cudarc::driver::CudaFunction,
}

/// Compile p-bit kernels from CUDA source
fn compile_pbit_kernels(ctx: &Arc<cudarc::driver::CudaContext>) -> CudaResult<PBitKernels> {
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use logic_fabric_core::cuda::get_cuda_include_path;

    let cuda_include = get_cuda_include_path();
    let arch = get_device_arch_string();

    let opts = CompileOptions {
        arch: Some(Box::leak(arch.into_boxed_str())),
        include_paths: vec![cuda_include],
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(PBIT_KERNEL_CUDA, opts).map_err(|e| {
        CudaError::KernelCompilationFailed(format!("P-bit kernel compile error: {:?}", e))
    })?;

    let module = ctx.load_module(ptx).map_err(|e| {
        CudaError::KernelCompilationFailed(format!("P-bit kernel load error: {:?}", e))
    })?;

    let load_fn = |name: &str| {
        module
            .load_function(name)
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{}: {:?}", name, e)))
    };

    Ok(PBitKernels {
        compute_fields_fn: load_fn("pbit_compute_fields")?,
        update_fn: load_fn("pbit_update")?,
        compute_energy_fn: load_fn("pbit_compute_energy")?,
        reduce_energy_fn: load_fn("pbit_reduce_energy")?,
        shuffle_fn: load_fn("pbit_shuffle_indices")?,
        update_sequential_fn: load_fn("pbit_update_sequential")?,
    })
}

/// Get device architecture string for kernel compilation
fn get_device_arch_string() -> String {
    "compute_89".to_string() // Ada Lovelace / RTX 40 series
}

/// GPU-accelerated p-bit network
///
/// All state is GPU-resident for maximum performance.
pub struct GpuPBitNetwork {
    /// CUDA runtime
    rt: CudaRuntime,
    /// Compiled kernels
    kernels: PBitKernels,
    /// Number of p-bits
    n: usize,
    /// Inverse temperature
    beta: f32,
    /// Update order
    update_order: UpdateOrder,
    /// RNG seed counter
    seed: u32,

    // GPU buffers
    /// Coupling matrix J [n*n]
    d_j: CudaSlice<f32>,
    /// Bias vector h [n]
    d_h: CudaSlice<f32>,
    /// Current spins (-1 or +1) [n]
    d_spins: CudaSlice<i8>,
    /// Binary states (0 or 1) [n]
    d_states: CudaSlice<u8>,
    /// Local fields buffer [n]
    d_fields: CudaSlice<f32>,
    /// Update order indices [n]
    d_order: CudaSlice<u32>,
    /// Partial energy sums [blocks]
    d_partial_energy: CudaSlice<f32>,
    /// Total energy [1]
    d_energy: CudaSlice<f32>,

    // Host-side tracking
    /// Best energy found
    best_energy: f32,
    /// Best state found
    best_state: Vec<u8>,
    /// Number of steps taken
    steps: u64,
}

impl GpuPBitNetwork {
    /// Create a new GPU p-bit network
    pub fn new(n: usize, beta: f32) -> CudaResult<Self> {
        Self::new_with_seed(n, beta, 42)
    }

    /// Create with specific seed
    pub fn new_with_seed(n: usize, beta: f32, seed: u64) -> CudaResult<Self> {
        let rt = CudaRuntime::new()?;
        let kernels = compile_pbit_kernels(rt.ctx())?;

        // Allocate GPU buffers
        let d_j = rt.alloc_zeros::<f32>(n * n)?;
        let d_h = rt.alloc_zeros::<f32>(n)?;
        let d_spins = rt.alloc_zeros::<i8>(n)?;
        let d_states = rt.alloc_zeros::<u8>(n)?;
        let d_fields = rt.alloc_zeros::<f32>(n)?;

        // Initialize order indices
        let order: Vec<u32> = (0..n as u32).collect();
        let d_order = rt.upload(&order)?;

        // Energy reduction buffers
        let n_blocks = (n + 255) / 256;
        let d_partial_energy = rt.alloc_zeros::<f32>(n_blocks)?;
        let d_energy = rt.alloc_zeros::<f32>(1)?;

        // Initialize spins to -1 (state 0)
        let init_spins: Vec<i8> = vec![-1; n];
        rt.upload_to_existing(&init_spins, &mut rt.alloc_zeros::<i8>(n)?)?;

        Ok(Self {
            rt,
            kernels,
            n,
            beta,
            update_order: UpdateOrder::Parallel,
            seed: seed as u32,
            d_j,
            d_h,
            d_spins,
            d_states,
            d_fields,
            d_order,
            d_partial_energy,
            d_energy,
            best_energy: f32::INFINITY,
            best_state: vec![0; n],
            steps: 0,
        })
    }

    /// Load an Ising problem
    pub fn set_problem(&mut self, problem: &IsingProblem) -> CudaResult<()> {
        assert_eq!(problem.n, self.n);

        // Convert to f32 and upload
        let j_f32: Vec<f32> = problem.j.iter().map(|&x| x as f32).collect();
        let h_f32: Vec<f32> = problem.h.iter().map(|&x| x as f32).collect();

        self.rt.upload_to_existing(&j_f32, &mut self.d_j)?;
        self.rt.upload_to_existing(&h_f32, &mut self.d_h)?;

        // Reset state
        self.randomize()?;
        self.best_energy = f32::INFINITY;
        self.best_state = vec![0; self.n];
        self.steps = 0;

        Ok(())
    }

    /// Set inverse temperature
    pub fn set_beta(&mut self, beta: f32) {
        self.beta = beta;
    }

    /// Set update order
    pub fn set_update_order(&mut self, order: UpdateOrder) {
        self.update_order = order;
    }

    /// Randomize p-bit states
    pub fn randomize(&mut self) -> CudaResult<()> {
        self.seed = self.seed.wrapping_add(1);

        // Use update kernel with beta=0 (uniform random)
        let threads = 256u32;
        let blocks = ((self.n as u32) + threads - 1) / threads;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernels.update_fn)
                .arg(&self.d_fields) // Will be ignored since beta=0
                .arg(&self.d_spins)
                .arg(&self.d_states)
                .arg(&0.0f32) // beta = 0 for uniform random
                .arg(&self.seed)
                .arg(&(self.n as u32))
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("randomize: {:?}", e)))?;
        }

        self.rt.synchronize()?;
        Ok(())
    }

    /// Perform one simulation step
    pub fn step(&mut self) -> CudaResult<()> {
        self.seed = self.seed.wrapping_add(1);

        match self.update_order {
            UpdateOrder::Parallel => self.step_parallel()?,
            UpdateOrder::Sequential => self.step_sequential()?,
            UpdateOrder::Random => {
                self.shuffle_order()?;
                self.step_sequential()?;
            }
        }

        self.steps += 1;

        // Periodically update best (every 100 steps to reduce overhead)
        if self.steps % 100 == 0 {
            self.update_best()?;
        }

        Ok(())
    }

    /// Parallel update (all p-bits simultaneously)
    fn step_parallel(&mut self) -> CudaResult<()> {
        let threads = 256u32;
        let blocks = ((self.n as u32) + threads - 1) / threads;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        // Compute local fields
        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernels.compute_fields_fn)
                .arg(&self.d_j)
                .arg(&self.d_h)
                .arg(&self.d_spins)
                .arg(&self.d_fields)
                .arg(&(self.n as u32))
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("compute_fields: {:?}", e)))?;
        }

        // Update p-bits
        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernels.update_fn)
                .arg(&self.d_fields)
                .arg(&self.d_spins)
                .arg(&self.d_states)
                .arg(&self.beta)
                .arg(&self.seed)
                .arg(&(self.n as u32))
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("update: {:?}", e)))?;
        }

        Ok(())
    }

    /// Sequential update (one p-bit at a time)
    fn step_sequential(&mut self) -> CudaResult<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernels.update_sequential_fn)
                .arg(&self.d_j)
                .arg(&self.d_h)
                .arg(&self.d_spins)
                .arg(&self.d_states)
                .arg(&self.d_order)
                .arg(&self.beta)
                .arg(&self.seed)
                .arg(&(self.n as u32))
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("update_sequential: {:?}", e)))?;
        }

        Ok(())
    }

    /// Shuffle update order
    fn shuffle_order(&mut self) -> CudaResult<()> {
        self.seed = self.seed.wrapping_add(7919); // Different sequence

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernels.shuffle_fn)
                .arg(&self.d_order)
                .arg(&self.seed)
                .arg(&(self.n as u32))
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("shuffle: {:?}", e)))?;
        }

        Ok(())
    }

    /// Run multiple steps
    pub fn run(&mut self, steps: u64) -> CudaResult<()> {
        for _ in 0..steps {
            self.step()?;
        }
        self.update_best()?;
        Ok(())
    }

    /// Compute current energy
    pub fn compute_energy(&mut self) -> CudaResult<f32> {
        let threads = 256u32;
        let blocks = ((self.n as u32) + threads - 1) / threads;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * std::mem::size_of::<f32>() as u32,
        };

        // Compute partial energies
        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernels.compute_energy_fn)
                .arg(&self.d_j)
                .arg(&self.d_h)
                .arg(&self.d_spins)
                .arg(&self.d_partial_energy)
                .arg(&(self.n as u32))
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("compute_energy: {:?}", e)))?;
        }

        // Final reduction
        let reduce_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * std::mem::size_of::<f32>() as u32,
        };

        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernels.reduce_energy_fn)
                .arg(&self.d_partial_energy)
                .arg(&self.d_energy)
                .arg(&blocks)
                .launch(reduce_cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("reduce_energy: {:?}", e)))?;
        }

        // Download result
        let energy = self.rt.download(&self.d_energy)?;
        Ok(energy[0])
    }

    /// Update best solution if current is better
    fn update_best(&mut self) -> CudaResult<()> {
        let energy = self.compute_energy()?;
        if energy < self.best_energy {
            self.best_energy = energy;
            self.best_state = self.rt.download(&self.d_states)?;
        }
        Ok(())
    }

    /// Get current state (downloads from GPU)
    pub fn state(&self) -> CudaResult<Vec<u8>> {
        self.rt.download(&self.d_states)
    }

    /// Get current spins (downloads from GPU)
    pub fn spins(&self) -> CudaResult<Vec<i8>> {
        self.rt.download(&self.d_spins)
    }

    /// Get best energy found
    pub fn best_energy(&self) -> f32 {
        self.best_energy
    }

    /// Get best state found
    pub fn best_state(&self) -> &[u8] {
        &self.best_state
    }

    /// Get number of steps taken
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Get number of p-bits
    pub fn n_pbits(&self) -> usize {
        self.n
    }

    /// Synchronize GPU operations
    pub fn sync(&self) -> CudaResult<()> {
        self.rt.synchronize()
    }
}

/// GPU-accelerated simulated annealing
pub struct GpuSimulatedAnnealing<'a> {
    network: &'a mut GpuPBitNetwork,
    beta_min: f32,
    beta_max: f32,
    total_steps: u64,
}

impl<'a> GpuSimulatedAnnealing<'a> {
    pub fn new(network: &'a mut GpuPBitNetwork) -> Self {
        Self {
            network,
            beta_min: 0.1,
            beta_max: 10.0,
            total_steps: 10000,
        }
    }

    pub fn with_beta_range(mut self, beta_min: f32, beta_max: f32) -> Self {
        self.beta_min = beta_min;
        self.beta_max = beta_max;
        self
    }

    pub fn with_steps(mut self, steps: u64) -> Self {
        self.total_steps = steps;
        self
    }

    /// Run annealing with exponential schedule
    pub fn run(&mut self) -> CudaResult<GpuAnnealingResult> {
        self.network.randomize()?;
        self.network.sync()?;

        // Initialize best from random state
        self.network.best_energy = f32::INFINITY;
        self.network.update_best()?;

        // Use slower annealing to avoid getting stuck
        let check_interval = (self.total_steps / 20).max(1);

        for t in 0..self.total_steps {
            // Exponential schedule with slower ramp
            let progress = t as f32 / self.total_steps as f32;
            // Use sqrt for slower beta increase
            let beta = self.beta_min + (self.beta_max - self.beta_min) * progress.sqrt();
            self.network.set_beta(beta);
            self.network.step()?;

            // Check best more frequently
            if t % check_interval == 0 {
                self.network.sync()?;
                self.network.update_best()?;
            }
        }

        self.network.sync()?;
        self.network.update_best()?;

        Ok(GpuAnnealingResult {
            best_state: self.network.best_state.clone(),
            best_energy: self.network.best_energy,
            steps: self.network.steps,
        })
    }
}

/// Result from GPU simulated annealing
#[derive(Debug, Clone)]
pub struct GpuAnnealingResult {
    pub best_state: Vec<u8>,
    pub best_energy: f32,
    pub steps: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbit::ising::MaxCutGraph;

    #[test]
    fn test_gpu_network_creation() {
        if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
            println!("Skipping GPU test - no CUDA device");
            return;
        }

        let result = GpuPBitNetwork::new(100, 1.0);
        assert!(
            result.is_ok(),
            "Failed to create GPU network: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_gpu_maxcut() {
        if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
            println!("Skipping GPU test - no CUDA device");
            return;
        }

        let graph = MaxCutGraph::complete(10);
        let problem = graph.to_ising();

        let mut network = GpuPBitNetwork::new(10, 5.0).unwrap();
        network.set_problem(&problem).unwrap();
        network.run(1000).unwrap();

        let cut = graph.cut_value(&network.best_state);
        assert!(cut >= 20.0, "Expected good cut, got {}", cut);
    }
}
