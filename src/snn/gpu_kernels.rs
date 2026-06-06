//! CUDA GPU Kernels for SNN Acceleration
//!
//! Implements GPU-accelerated SNN operations:
//! - LIF membrane dynamics (element-wise kernel)
//! - Current accumulation via CSR SpMV
//! - Output spike counting
//! - Input spike generation
//!
//! Expected speedup: 10-50x for networks with 10K+ neurons.

use crate::cuda::{CudaError, CudaResult, CudaRuntime};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::neuron::LIFNeuron;
use super::synapse::SynapseCSR;

/// CUDA kernel source code for SNN operations
const SNN_KERNEL_CUDA: &str = r#"
extern "C" {

// Kernel 1: LIF Membrane Dynamics
// One thread per neuron - updates membrane potential and detects spikes
__global__ void snn_lif_step(
    short* v_mem,           // [n] Q8.8 membrane potential
    const short* threshold, // [n] Q8.8 firing threshold
    const unsigned char* leak,     // [n] Q0.8 leak factor
    unsigned char* refract, // [n] refractory counter
    unsigned char* spiked,  // [n] output: did neuron spike?
    const int* currents,    // [n] input currents this tick (int for atomicAdd)
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Handle refractory period
    if (refract[i] > 0) {
        refract[i]--;
        spiked[i] = 0;
        return;
    }

    // Leak: v = v * leak >> 8 (Q8.8 * Q0.8 >> 8 = Q8.8)
    int v = ((int)v_mem[i] * (int)leak[i]) >> 8;

    // Integrate current
    v += currents[i];

    // Clamp to i16 range
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;

    // Fire?
    if (v >= (int)threshold[i]) {
        v_mem[i] = -128;  // V_RESET
        refract[i] = 2;
        spiked[i] = 1;
    } else {
        v_mem[i] = (short)v;
        spiked[i] = 0;
    }
}

// Kernel 2: Current Accumulation (SpMV with CSR format)
// Each thread handles one source neuron's outgoing synapses
__global__ void snn_accumulate_currents(
    const unsigned int* syn_ptr,    // [n+1] CSR row pointers
    const unsigned short* targets,  // [nnz] target neuron indices
    const signed char* weights,     // [nnz] Q1.7 weights
    const unsigned char* spiked,    // [n] which neurons fired
    int* currents,                  // [n] output: accumulated currents (must be zeroed first)
    unsigned int n
) {
    unsigned int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= n) return;

    // Only process if this neuron spiked
    if (spiked[src] == 0) return;

    // Iterate over outgoing synapses
    unsigned int start = syn_ptr[src];
    unsigned int end = syn_ptr[src + 1];

    for (unsigned int j = start; j < end; j++) {
        unsigned short tgt = targets[j];
        int current = (int)weights[j] * 2;  // Scale weight to current
        atomicAdd(&currents[tgt], current);
    }
}

// Kernel 3: Clear currents buffer (memset alternative)
__global__ void snn_clear_currents(
    int* currents,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    currents[i] = 0;
}

// Kernel 4: Output Spike Counting
// Count spikes in output layer neurons
__global__ void snn_count_output_spikes(
    const unsigned char* spiked,    // [n] all neuron spikes
    unsigned int* output_counts,    // [n_outputs] spike counts
    unsigned int output_start,      // first output neuron index
    unsigned int n_outputs
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_outputs) return;

    unsigned int neuron_idx = output_start + i;
    if (spiked[neuron_idx]) {
        atomicAdd(&output_counts[i], 1);
    }
}

// Kernel 5: Input Spike Generation
// Generate input spikes from rates using simple PRNG
__global__ void snn_generate_input_spikes(
    unsigned char* spiked,          // [n] neuron spike flags
    short* v_mem,                   // [n] membrane potentials
    unsigned char* refract,         // [n] refractory counters
    const unsigned char* rates,     // [n_inputs] input spike rates
    unsigned int n_inputs,
    unsigned int seed               // random seed for this tick
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_inputs) return;

    // Don't generate if in refractory
    if (refract[i] > 0) return;

    // Simple LCG PRNG per thread
    unsigned int rng = seed ^ (i * 2654435761u);
    rng = rng * 1664525u + 1013904223u;
    unsigned char rand_val = (unsigned char)(rng >> 24);

    if (rand_val < rates[i]) {
        spiked[i] = 1;
        v_mem[i] = -128;  // Reset
        refract[i] = 2;
    }
}

// Kernel 6: Reset output counts to zero
__global__ void snn_reset_output_counts(
    unsigned int* output_counts,
    unsigned int n_outputs
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_outputs) return;
    output_counts[i] = 0;
}

} // extern "C"
"#;

/// Compiled SNN kernel functions
struct SNNKernels {
    lif_step_fn: cudarc::driver::CudaFunction,
    accumulate_fn: cudarc::driver::CudaFunction,
    clear_currents_fn: cudarc::driver::CudaFunction,
    count_outputs_fn: cudarc::driver::CudaFunction,
    gen_inputs_fn: cudarc::driver::CudaFunction,
    reset_outputs_fn: cudarc::driver::CudaFunction,
}

/// Compile SNN kernels from CUDA source
fn compile_snn_kernels(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
) -> CudaResult<SNNKernels> {
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use logic_fabric_core::cuda::get_cuda_include_path;

    let cuda_include = get_cuda_include_path();

    // Get device architecture string
    let arch: &'static str = Box::leak(get_device_arch_string().into_boxed_str());

    let opts = CompileOptions {
        arch: Some(arch),
        include_paths: vec![cuda_include],
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(SNN_KERNEL_CUDA, opts).map_err(|e| {
        CudaError::KernelCompilationFailed(format!("SNN kernel compile error: {:?}", e))
    })?;

    let module = ctx.load_module(ptx).map_err(|e| {
        CudaError::KernelCompilationFailed(format!("SNN kernel load error: {:?}", e))
    })?;

    let lif_step_fn = module
        .load_function("snn_lif_step")
        .map_err(|e| CudaError::KernelCompilationFailed(format!("snn_lif_step: {:?}", e)))?;

    let accumulate_fn = module
        .load_function("snn_accumulate_currents")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!("snn_accumulate_currents: {:?}", e))
        })?;

    let clear_currents_fn = module
        .load_function("snn_clear_currents")
        .map_err(|e| CudaError::KernelCompilationFailed(format!("snn_clear_currents: {:?}", e)))?;

    let count_outputs_fn = module
        .load_function("snn_count_output_spikes")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!("snn_count_output_spikes: {:?}", e))
        })?;

    let gen_inputs_fn = module
        .load_function("snn_generate_input_spikes")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!("snn_generate_input_spikes: {:?}", e))
        })?;

    let reset_outputs_fn = module
        .load_function("snn_reset_output_counts")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!("snn_reset_output_counts: {:?}", e))
        })?;

    Ok(SNNKernels {
        lif_step_fn,
        accumulate_fn,
        clear_currents_fn,
        count_outputs_fn,
        gen_inputs_fn,
        reset_outputs_fn,
    })
}

/// Get device architecture string for kernel compilation
fn get_device_arch_string() -> String {
    // Default to compute_89 (Ada Lovelace / RTX 40 series)
    // The actual arch is detected at runtime by logic_fabric_core::cuda
    "compute_89".to_string()
}

/// Cached SNN kernels for reuse
pub struct SNNKernelCache {
    kernels: SNNKernels,
}

impl SNNKernelCache {
    /// Initialize the SNN kernel cache
    pub fn new(rt: &CudaRuntime) -> CudaResult<Self> {
        let kernels = compile_snn_kernels(rt.context())?;
        Ok(Self { kernels })
    }
}

/// GPU-resident SNN state buffers
pub struct GpuSNNState {
    // Neuron state buffers
    pub v_mem: CudaSlice<i16>,
    pub threshold: CudaSlice<i16>,
    pub leak: CudaSlice<u8>,
    pub refract: CudaSlice<u8>,
    pub spiked: CudaSlice<u8>,
    pub currents: CudaSlice<i32>, // i32 for atomicAdd compatibility

    // Synapse CSR buffers
    pub syn_ptr: CudaSlice<u32>,
    pub targets: CudaSlice<u16>,
    pub weights: CudaSlice<i8>,

    // Input/output buffers
    pub input_rates: CudaSlice<u8>,
    pub output_counts: CudaSlice<u32>,

    // Metadata
    pub n_neurons: usize,
    pub n_synapses: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub output_start: usize,

    // Kernel cache
    kernel_cache: SNNKernelCache,

    // Tick counter for RNG seeding
    tick: u64,
}

impl GpuSNNState {
    /// Create GPU state from CPU neuron and synapse data
    pub fn from_cpu(
        rt: &CudaRuntime,
        neurons: &[LIFNeuron],
        csr: &SynapseCSR,
        n_inputs: usize,
        n_outputs: usize,
    ) -> CudaResult<Self> {
        let n_neurons = neurons.len();
        let n_synapses = csr.n_synapses();
        let output_start = n_neurons - n_outputs;

        // Extract neuron fields into separate arrays
        let v_mem_data: Vec<i16> = neurons.iter().map(|n| n.v_mem).collect();
        let threshold_data: Vec<i16> = neurons.iter().map(|n| n.threshold).collect();
        let leak_data: Vec<u8> = neurons.iter().map(|n| n.leak).collect();
        let refract_data: Vec<u8> = neurons.iter().map(|n| n.refractory).collect();
        let spiked_data: Vec<u8> = neurons.iter().map(|n| n.spiked).collect();

        // Upload neuron state
        let v_mem = rt.upload(&v_mem_data)?;
        let threshold = rt.upload(&threshold_data)?;
        let leak = rt.upload(&leak_data)?;
        let refract = rt.upload(&refract_data)?;
        let spiked = rt.upload(&spiked_data)?;
        let currents = rt.alloc_zeros::<i32>(n_neurons)?;

        // Upload synapse CSR
        let syn_ptr = rt.upload(&csr.syn_ptr)?;
        let targets = rt.upload(&csr.targets)?;
        let weights = rt.upload(&csr.weights)?;

        // Allocate input/output buffers
        let input_rates = rt.alloc_zeros::<u8>(n_inputs)?;
        let output_counts = rt.alloc_zeros::<u32>(n_outputs)?;

        // Initialize kernel cache
        let kernel_cache = SNNKernelCache::new(rt)?;

        Ok(Self {
            v_mem,
            threshold,
            leak,
            refract,
            spiked,
            currents,
            syn_ptr,
            targets,
            weights,
            input_rates,
            output_counts,
            n_neurons,
            n_synapses,
            n_inputs,
            n_outputs,
            output_start,
            kernel_cache,
            tick: 0,
        })
    }

    /// Upload input spike rates from CPU
    pub fn set_input_rates(&mut self, rt: &CudaRuntime, rates: &[u8]) -> CudaResult<()> {
        assert!(rates.len() <= self.n_inputs);
        rt.upload_to_existing(rates, &mut self.input_rates)
    }

    /// Download output spike counts to CPU
    pub fn get_output_counts(&self, rt: &CudaRuntime) -> CudaResult<Vec<u32>> {
        rt.download(&self.output_counts)
    }

    /// Execute one SNN tick on GPU
    pub fn step(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        let threads_per_block = 256u32;
        let n = self.n_neurons as u32;
        let num_blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernels = &self.kernel_cache.kernels;

        // Step 1: Clear currents
        unsafe {
            rt.stream()
                .launch_builder(&kernels.clear_currents_fn)
                .arg(&self.currents)
                .arg(&n)
                .launch(cfg.clone())
                .map_err(|e| CudaError::LaunchFailed(format!("clear_currents: {:?}", e)))?;
        }

        // Step 2: Generate input spikes
        let input_blocks = (self.n_inputs as u32 + threads_per_block - 1) / threads_per_block;
        let input_cfg = LaunchConfig {
            grid_dim: (input_blocks.max(1), 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_inputs = self.n_inputs as u32;
        let seed = self.tick as u32;

        unsafe {
            rt.stream()
                .launch_builder(&kernels.gen_inputs_fn)
                .arg(&self.spiked)
                .arg(&self.v_mem)
                .arg(&self.refract)
                .arg(&self.input_rates)
                .arg(&n_inputs)
                .arg(&seed)
                .launch(input_cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("gen_inputs: {:?}", e)))?;
        }

        // Step 3: Accumulate currents from spiked neurons
        unsafe {
            rt.stream()
                .launch_builder(&kernels.accumulate_fn)
                .arg(&self.syn_ptr)
                .arg(&self.targets)
                .arg(&self.weights)
                .arg(&self.spiked)
                .arg(&self.currents)
                .arg(&n)
                .launch(cfg.clone())
                .map_err(|e| CudaError::LaunchFailed(format!("accumulate: {:?}", e)))?;
        }

        // Step 4: LIF membrane dynamics
        unsafe {
            rt.stream()
                .launch_builder(&kernels.lif_step_fn)
                .arg(&self.v_mem)
                .arg(&self.threshold)
                .arg(&self.leak)
                .arg(&self.refract)
                .arg(&self.spiked)
                .arg(&self.currents)
                .arg(&n)
                .launch(cfg.clone())
                .map_err(|e| CudaError::LaunchFailed(format!("lif_step: {:?}", e)))?;
        }

        // Step 5: Count output spikes
        let output_blocks = (self.n_outputs as u32 + threads_per_block - 1) / threads_per_block;
        let output_cfg = LaunchConfig {
            grid_dim: (output_blocks.max(1), 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        let output_start = self.output_start as u32;
        let n_outputs = self.n_outputs as u32;

        unsafe {
            rt.stream()
                .launch_builder(&kernels.count_outputs_fn)
                .arg(&self.spiked)
                .arg(&self.output_counts)
                .arg(&output_start)
                .arg(&n_outputs)
                .launch(output_cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("count_outputs: {:?}", e)))?;
        }

        self.tick += 1;

        // Synchronize to ensure all kernels complete
        rt.synchronize()?;

        Ok(())
    }

    /// Reset output counts to zero (call before decision period)
    pub fn reset_output_counts(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        let threads_per_block = 256u32;
        let output_blocks = (self.n_outputs as u32 + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (output_blocks.max(1), 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_outputs = self.n_outputs as u32;

        unsafe {
            rt.stream()
                .launch_builder(&self.kernel_cache.kernels.reset_outputs_fn)
                .arg(&self.output_counts)
                .arg(&n_outputs)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("reset_outputs: {:?}", e)))?;
        }

        rt.synchronize()?;
        Ok(())
    }

    /// Get the winning output neuron (argmax of spike counts)
    pub fn get_action(&self, rt: &CudaRuntime) -> CudaResult<usize> {
        let counts = self.get_output_counts(rt)?;
        let (max_idx, _) = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .unwrap_or((0, &0));
        Ok(max_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_gpu_snn_creation() {
        let rt = CudaRuntime::new().expect("CUDA runtime");

        // Create minimal test data
        let neurons = vec![LIFNeuron::default(); 100];
        let mut csr = super::super::synapse::SynapseCSR {
            syn_ptr: vec![0; 101],
            targets: vec![],
            weights: vec![],
            delays: vec![],
        };
        // Initialize syn_ptr properly
        for i in 0..=100 {
            csr.syn_ptr[i] = 0;
        }

        let gpu_state = GpuSNNState::from_cpu(&rt, &neurons, &csr, 10, 5);
        assert!(gpu_state.is_ok());
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_gpu_snn_step() {
        let rt = CudaRuntime::new().expect("CUDA runtime");

        let neurons = vec![LIFNeuron::default(); 100];
        let csr = super::super::synapse::SynapseCSR {
            syn_ptr: vec![0; 101],
            targets: vec![],
            weights: vec![],
            delays: vec![],
        };

        let mut gpu_state = GpuSNNState::from_cpu(&rt, &neurons, &csr, 10, 5).expect("GPU state");

        // Set some input rates
        let rates = vec![128u8; 10]; // 50% spike probability
        gpu_state.set_input_rates(&rt, &rates).expect("set rates");

        // Run a step
        gpu_state.step(&rt).expect("step");

        // Get output counts
        let counts = gpu_state.get_output_counts(&rt).expect("counts");
        assert_eq!(counts.len(), 5);
    }
}
