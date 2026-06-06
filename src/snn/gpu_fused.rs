//! EPIC 97-Style Fused GPU SNN Kernel
//!
//! Zero-transfer-per-tick SNN simulation with all state GPU-resident.
//!
//! Architecture:
//! - Phase A: Input spike generation (rate-coded Poisson)
//! - Phase B: Current accumulation (CSR SpMV with atomicAdd)
//! - Phase C: LIF dynamics + spike detection
//!
//! All phases execute on GPU with no PCIe transfers between ticks.
//! Fire-and-forget: call tick_many(N) and sync only when needed.
//!
//! Expected improvement: 1.5-2x over separate kernel approach.

use crate::cuda::{CudaError, CudaResult, CudaRuntime};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

/// Fused SNN CUDA kernels
///
/// The key insight from EPIC 97: minimize kernel launches and eliminate
/// PCIe transfers between ticks. We use three kernels that execute in
/// rapid succession with no CPU involvement.
const FUSED_SNN_CUDA: &str = r#"
extern "C" {

// =============================================================================
// KERNEL 1: Input + LIF Dynamics (Fused)
// =============================================================================
// This kernel handles both input spike generation AND LIF dynamics in one pass.
// For input neurons (i < n_inputs): generate spikes from rates
// For all neurons: apply leak and integrate currents
//
// By fusing these, we eliminate one kernel launch and its sync overhead.
__global__ void snn_fused_input_dynamics(
    short* v_mem,                   // [n] Q8.8 membrane potential
    const short* threshold,         // [n] Q8.8 firing threshold
    const unsigned char* leak,      // [n] Q0.8 leak factor
    unsigned char* refract,         // [n] refractory counter
    unsigned char* spiked,          // [n] output: spike flags
    int* currents,                  // [n] accumulated currents (cleared after use)
    const unsigned char* input_rates, // [n_inputs] spike rates for input layer
    unsigned int n_neurons,
    unsigned int n_inputs,
    unsigned int seed               // random seed for this tick
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;

    // ========== Phase A: Input Generation (for input neurons only) ==========
    if (i < n_inputs && refract[i] == 0) {
        // Simple LCG PRNG per thread
        unsigned int rng = seed ^ (i * 2654435761u);
        rng = rng * 1664525u + 1013904223u;
        unsigned char rand_val = (unsigned char)(rng >> 24);

        if (rand_val < input_rates[i]) {
            // Input neuron fires - set spike and reset
            spiked[i] = 1;
            v_mem[i] = -128;  // V_RESET
            refract[i] = 2;
            currents[i] = 0;  // Clear current for this neuron
            return;
        }
    }

    // ========== Phase C: LIF Dynamics (for all neurons) ==========

    // Handle refractory period
    if (refract[i] > 0) {
        refract[i]--;
        spiked[i] = 0;
        currents[i] = 0;  // Clear current
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

    // Clear current for next tick
    currents[i] = 0;
}

// =============================================================================
// KERNEL 2: Current Accumulation (SpMV)
// =============================================================================
// Each thread handles one source neuron's outgoing synapses.
// This MUST run BEFORE the dynamics kernel in the next tick.
__global__ void snn_fused_accumulate(
    const unsigned int* syn_ptr,    // [n+1] CSR row pointers
    const unsigned int* targets,    // [nnz] target neuron indices (u32 for >65K neurons)
    const signed char* weights,     // [nnz] Q1.7 weights
    const unsigned char* spiked,    // [n] which neurons fired
    int* currents,                  // [n] output: accumulated currents
    unsigned int n_neurons
) {
    unsigned int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= n_neurons) return;

    // Only process if this neuron spiked
    if (spiked[src] == 0) return;

    // Iterate over outgoing synapses
    unsigned int start = syn_ptr[src];
    unsigned int end = syn_ptr[src + 1];

    for (unsigned int j = start; j < end; j++) {
        unsigned int tgt = targets[j];
        int current = (int)weights[j] * 2;  // Scale weight to current
        atomicAdd(&currents[tgt], current);
    }
}

// =============================================================================
// KERNEL 3: Output Spike Counting
// =============================================================================
// Count spikes in output layer neurons. Only called when results are needed.
__global__ void snn_fused_count_outputs(
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

// =============================================================================
// KERNEL 4: Reset Output Counts
// =============================================================================
__global__ void snn_fused_reset_outputs(
    unsigned int* output_counts,
    unsigned int n_outputs
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_outputs) return;
    output_counts[i] = 0;
}

// =============================================================================
// KERNEL 5: Update Spike Times
// =============================================================================
// Records the current tick as the last spike time for any neuron that spiked.
// Must be called BEFORE the eligibility update to capture this tick's spikes.
__global__ void snn_fused_update_spike_times(
    const unsigned char* spiked,    // [n] current spike flags
    unsigned int* last_spike_time,  // [n] tick when each neuron last spiked
    unsigned int n_neurons,
    unsigned int current_tick
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;

    if (spiked[i]) {
        last_spike_time[i] = current_tick;
    }
}

// =============================================================================
// KERNEL 6: Timing-Based Eligibility Trace Update (Three-Factor STDP)
// =============================================================================
// Updates eligibility traces based on SPIKE TIMING, not just co-occurrence.
//
// Classic STDP curve:
// - If post fires AFTER pre (delta_t > 0): LTP with strength ~ exp(-delta_t / tau_plus)
// - If post fires BEFORE pre (delta_t < 0): LTD with strength ~ exp(delta_t / tau_minus)
// - Outside timing window: no update (except decay)
//
// This creates DIFFERENTIATED eligibility traces based on causal relationships.
__global__ void snn_fused_update_eligibility_timing(
    const unsigned int* syn_ptr,       // [n+1] CSR row pointers
    const unsigned int* targets,       // [nnz] target neuron indices
    const unsigned int* last_spike_time, // [n] tick when each neuron last spiked
    float* eligibility,                // [nnz] eligibility traces
    unsigned int n_neurons,
    unsigned int current_tick,
    float tau_decay,                   // eligibility decay factor (e.g., 0.95)
    float ltp_amplitude,               // A+ amplitude (e.g., 0.1)
    float ltd_amplitude,               // A- amplitude (e.g., 0.05)
    float tau_plus,                    // LTP time constant in ticks (e.g., 20.0)
    float tau_minus,                   // LTD time constant in ticks (e.g., 20.0)
    unsigned int stdp_window           // max timing window (e.g., 30 ticks)
) {
    unsigned int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= n_neurons) return;

    unsigned int start = syn_ptr[src];
    unsigned int end = syn_ptr[src + 1];

    unsigned int t_pre = last_spike_time[src];

    for (unsigned int j = start; j < end; j++) {
        // Decay existing eligibility
        float e = eligibility[j] * tau_decay;

        unsigned int tgt = targets[j];
        unsigned int t_post = last_spike_time[tgt];

        // Only compute STDP if both neurons have spiked recently
        // (last_spike_time == 0 means never spiked)
        if (t_pre > 0 && t_post > 0) {
            // Compute timing difference: positive means post fired after pre (causal)
            int delta_t = (int)t_post - (int)t_pre;

            if (delta_t > 0 && delta_t <= (int)stdp_window) {
                // LTP: post fired AFTER pre (causal relationship)
                // Strength decays exponentially with time difference
                float strength = expf(-(float)delta_t / tau_plus);
                e += ltp_amplitude * strength;
            } else if (delta_t < 0 && delta_t >= -(int)stdp_window) {
                // LTD: post fired BEFORE pre (anti-causal)
                float strength = expf((float)delta_t / tau_minus);
                e -= ltd_amplitude * strength;
            }
            // If delta_t == 0 (same tick), we can choose either:
            // - Treat as weak LTP (common choice)
            // - No update
            // Here we treat same-tick as weak LTP
            else if (delta_t == 0) {
                e += ltp_amplitude * 0.5f;
            }
        }

        // Clamp eligibility to reasonable range
        if (e > 1.0f) e = 1.0f;
        if (e < -1.0f) e = -1.0f;

        eligibility[j] = e;
    }
}

// Legacy kernel for backwards compatibility
__global__ void snn_fused_update_eligibility(
    const unsigned int* syn_ptr,    // [n+1] CSR row pointers
    const unsigned int* targets,    // [nnz] target neuron indices
    const unsigned char* spiked,    // [n] current spike flags
    float* eligibility,             // [nnz] eligibility traces
    unsigned int n_neurons,
    float tau_decay,                // eligibility decay factor (e.g., 0.95)
    float ltp_rate,                 // LTP increment (e.g., 0.1)
    float ltd_rate                  // LTD decrement (e.g., 0.05)
) {
    unsigned int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= n_neurons) return;

    unsigned int start = syn_ptr[src];
    unsigned int end = syn_ptr[src + 1];

    unsigned char pre_spike = spiked[src];

    for (unsigned int j = start; j < end; j++) {
        // Decay existing eligibility
        float e = eligibility[j] * tau_decay;

        if (pre_spike) {
            unsigned int tgt = targets[j];
            unsigned char post_spike = spiked[tgt];

            if (post_spike) {
                // Pre and post both spiked: LTP (positive eligibility)
                e += ltp_rate;
            } else {
                // Pre spiked but post didn't: LTD (negative eligibility)
                e -= ltd_rate;
            }
        }

        // Clamp eligibility to reasonable range
        if (e > 1.0f) e = 1.0f;
        if (e < -1.0f) e = -1.0f;

        eligibility[j] = e;
    }
}

// =============================================================================
// KERNEL 6: Apply Reward (R-STDP Weight Update)
// =============================================================================
// Applies reward-modulated weight update using accumulated eligibility traces.
//
// Weight update rule: Δw = learning_rate * eligibility * reward
// Called at decision boundaries (after step() returns).
__global__ void snn_fused_apply_reward(
    signed char* weights,           // [nnz] Q1.7 weights (modified in place)
    const float* eligibility,       // [nnz] eligibility traces
    unsigned int n_synapses,
    float reward,                   // scalar reward signal [-1, 1]
    float learning_rate             // learning rate (e.g., 0.01)
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_synapses) return;

    // Compute weight change
    float delta = learning_rate * eligibility[j] * reward;

    // Convert to Q1.7 format and apply
    int new_weight = (int)weights[j] + (int)(delta * 127.0f);

    // Clamp to i8 range
    if (new_weight > 127) new_weight = 127;
    if (new_weight < -128) new_weight = -128;

    weights[j] = (signed char)new_weight;
}

// =============================================================================
// KERNEL 7: Reset Eligibility Traces
// =============================================================================
__global__ void snn_fused_reset_eligibility(
    float* eligibility,
    unsigned int n_synapses
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_synapses) return;
    eligibility[j] = 0.0f;
}

// =============================================================================
// KERNEL 8: Action-Specific Reward (Winner-Takes-All R-STDP)
// =============================================================================
// Only updates weights for synapses targeting the chosen action's output neuron.
// Optionally applies opposite reward to synapses targeting other outputs.
//
// This is the key for RL: credit assignment to the pathway that led to the action.
__global__ void snn_fused_apply_reward_action_specific(
    signed char* weights,           // [nnz] Q1.7 weights
    const float* eligibility,       // [nnz] eligibility traces
    const unsigned int* targets,    // [nnz] target neuron indices
    unsigned int n_synapses,
    unsigned int output_start,      // first output neuron index
    unsigned int n_outputs,
    unsigned int chosen_action,     // which output won (0 to n_outputs-1)
    float reward,                   // reward for chosen action
    float anti_reward_factor,       // multiply reward by this for non-chosen (e.g., -0.5)
    float learning_rate
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_synapses) return;

    unsigned int tgt = targets[j];

    // Check if this synapse targets an output neuron
    if (tgt >= output_start && tgt < output_start + n_outputs) {
        unsigned int output_idx = tgt - output_start;

        float effective_reward;
        if (output_idx == chosen_action) {
            // This synapse contributed to the chosen action - full reward
            effective_reward = reward;
        } else {
            // This synapse contributed to a non-chosen action - anti-reward
            effective_reward = reward * anti_reward_factor;
        }

        // Apply reward-modulated weight update
        float delta = learning_rate * eligibility[j] * effective_reward;
        int new_weight = (int)weights[j] + (int)(delta * 127.0f);

        if (new_weight > 127) new_weight = 127;
        if (new_weight < -128) new_weight = -128;

        weights[j] = (signed char)new_weight;
    }
    // Non-output synapses: no update (could optionally add decay here)
}

// =============================================================================
// KERNEL 9: Pathway-Traced Action-Specific Reward
// =============================================================================
// Updates ALL synapses, but scales reward by how much the synapse contributed
// to the winning vs losing outputs. Uses a softer credit assignment.
__global__ void snn_fused_apply_reward_pathway(
    signed char* weights,
    const float* eligibility,
    const unsigned int* targets,
    const unsigned char* spiked,    // [n] spike flags (for pathway tracing)
    unsigned int n_synapses,
    unsigned int output_start,
    unsigned int n_outputs,
    unsigned int chosen_action,
    float reward,
    float learning_rate
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_synapses) return;

    unsigned int tgt = targets[j];
    float e = eligibility[j];

    // Skip if no eligibility
    if (e == 0.0f) return;

    float effective_reward = reward;

    // Check if target is an output neuron
    if (tgt >= output_start && tgt < output_start + n_outputs) {
        unsigned int output_idx = tgt - output_start;
        if (output_idx == chosen_action) {
            // Direct connection to winner: full positive reward
            effective_reward = reward;
        } else {
            // Direct connection to loser: negative reward
            effective_reward = -reward * 0.3f;
        }
    } else {
        // Hidden layer synapse: scale down reward (indirect contribution)
        effective_reward = reward * 0.1f;
    }

    float delta = learning_rate * e * effective_reward;
    int new_weight = (int)weights[j] + (int)(delta * 127.0f);

    if (new_weight > 127) new_weight = 127;
    if (new_weight < -128) new_weight = -128;

    weights[j] = (signed char)new_weight;
}

} // extern "C"
"#;

/// Compiled fused SNN kernels
struct FusedSNNKernels {
    input_dynamics_fn: CudaFunction,
    accumulate_fn: CudaFunction,
    count_outputs_fn: CudaFunction,
    reset_outputs_fn: CudaFunction,
    // R-STDP learning kernels
    update_eligibility_fn: CudaFunction,
    apply_reward_fn: CudaFunction,
    reset_eligibility_fn: CudaFunction,
    // Action-specific learning kernels
    apply_reward_action_specific_fn: CudaFunction,
    apply_reward_pathway_fn: CudaFunction,
    // Timing-based three-factor STDP kernels
    update_spike_times_fn: CudaFunction,
    update_eligibility_timing_fn: CudaFunction,
}

/// Compile fused SNN kernels
fn compile_fused_kernels(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
) -> CudaResult<FusedSNNKernels> {
    use logic_fabric_core::cuda::get_cuda_include_path;

    let cuda_include = get_cuda_include_path();

    // Get device compute capability
    let arch = get_device_arch();

    let opts = CompileOptions {
        arch: Some(arch),
        include_paths: vec![cuda_include],
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(FUSED_SNN_CUDA, opts).map_err(|e| {
        CudaError::KernelCompilationFailed(format!("Fused SNN kernel compile error: {:?}", e))
    })?;

    let module = ctx.load_module(ptx).map_err(|e| {
        CudaError::KernelCompilationFailed(format!("Fused SNN kernel load error: {:?}", e))
    })?;

    let input_dynamics_fn = module
        .load_function("snn_fused_input_dynamics")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!("snn_fused_input_dynamics: {:?}", e))
        })?;

    let accumulate_fn = module.load_function("snn_fused_accumulate").map_err(|e| {
        CudaError::KernelCompilationFailed(format!("snn_fused_accumulate: {:?}", e))
    })?;

    let count_outputs_fn = module
        .load_function("snn_fused_count_outputs")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!("snn_fused_count_outputs: {:?}", e))
        })?;

    let reset_outputs_fn = module
        .load_function("snn_fused_reset_outputs")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!("snn_fused_reset_outputs: {:?}", e))
        })?;

    // R-STDP learning kernels
    let update_eligibility_fn = module
        .load_function("snn_fused_update_eligibility")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!("snn_fused_update_eligibility: {:?}", e))
        })?;

    let apply_reward_fn = module
        .load_function("snn_fused_apply_reward")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!("snn_fused_apply_reward: {:?}", e))
        })?;

    let reset_eligibility_fn = module
        .load_function("snn_fused_reset_eligibility")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!("snn_fused_reset_eligibility: {:?}", e))
        })?;

    // Action-specific learning kernels
    let apply_reward_action_specific_fn = module
        .load_function("snn_fused_apply_reward_action_specific")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!(
                "snn_fused_apply_reward_action_specific: {:?}",
                e
            ))
        })?;

    let apply_reward_pathway_fn = module
        .load_function("snn_fused_apply_reward_pathway")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!("snn_fused_apply_reward_pathway: {:?}", e))
        })?;

    // Timing-based three-factor STDP kernels
    let update_spike_times_fn = module
        .load_function("snn_fused_update_spike_times")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!("snn_fused_update_spike_times: {:?}", e))
        })?;

    let update_eligibility_timing_fn = module
        .load_function("snn_fused_update_eligibility_timing")
        .map_err(|e| {
            CudaError::KernelCompilationFailed(format!(
                "snn_fused_update_eligibility_timing: {:?}",
                e
            ))
        })?;

    Ok(FusedSNNKernels {
        input_dynamics_fn,
        accumulate_fn,
        count_outputs_fn,
        reset_outputs_fn,
        update_eligibility_fn,
        apply_reward_fn,
        reset_eligibility_fn,
        apply_reward_action_specific_fn,
        apply_reward_pathway_fn,
        update_spike_times_fn,
        update_eligibility_timing_fn,
    })
}

/// Get device architecture string for kernel compilation
fn get_device_arch() -> &'static str {
    // Default to compute_89 (Ada Lovelace / RTX 40 series)
    // RTX 5090 (Blackwell) is compute_120 (SM120), but 89 should work via JIT
    "compute_89"
}

/// GPU-resident fused SNN state
///
/// All buffers are GPU-resident. No PCIe transfers between ticks.
/// Call tick() or tick_many() for fire-and-forget execution.
pub struct GpuFusedSNN {
    // Neuron state (GPU-resident)
    d_v_mem: CudaSlice<i16>,
    d_threshold: CudaSlice<i16>,
    d_leak: CudaSlice<u8>,
    d_refract: CudaSlice<u8>,
    d_spiked: CudaSlice<u8>,
    d_currents: CudaSlice<i32>,

    // Synapse CSR (GPU-resident)
    d_syn_ptr: CudaSlice<u32>,
    d_targets: CudaSlice<u32>,
    d_weights: CudaSlice<i8>,

    // Input/Output (GPU-resident)
    d_input_rates: CudaSlice<u8>,
    d_output_counts: CudaSlice<u32>,

    // R-STDP Learning (GPU-resident)
    d_eligibility: CudaSlice<f32>, // [n_synapses] eligibility traces
    d_last_spike_time: CudaSlice<u32>, // [n_neurons] tick when each neuron last spiked
    n_synapses: u32,
    learning_enabled: bool,
    use_timing_stdp: bool, // Use timing-based STDP (true) or legacy co-occurrence (false)
    tau_decay: f32,        // eligibility decay factor (default: 0.95)
    ltp_rate: f32,         // LTP amplitude (default: 0.1)
    ltd_rate: f32,         // LTD amplitude (default: 0.05)
    learning_rate: f32,    // weight update rate (default: 0.01)
    stdp_tau_plus: f32,    // LTP time constant in ticks (default: 20.0)
    stdp_tau_minus: f32,   // LTD time constant in ticks (default: 20.0)
    stdp_window: u32,      // max timing window in ticks (default: 30)

    // Kernels
    kernels: FusedSNNKernels,

    // Configuration
    n_neurons: u32,
    n_inputs: u32,
    n_outputs: u32,
    output_start: u32,

    // Tick counter (for RNG seeding)
    tick: u64,
}

/// Network state snapshot (for CPU inspection)
#[derive(Clone, Debug)]
pub struct SNNSnapshot {
    pub v_mem: Vec<i16>,
    pub spiked: Vec<u8>,
    pub output_counts: Vec<u32>,
    pub tick: u64,
}

/// CSR synapse format for upload
pub struct SynapseCSRData {
    pub syn_ptr: Vec<u32>,
    pub targets: Vec<u32>, // u32 to support >65K neurons
    pub weights: Vec<i8>,
}

impl SynapseCSRData {
    pub fn n_synapses(&self) -> usize {
        self.targets.len()
    }
}

impl GpuFusedSNN {
    /// Create new fused SNN with given configuration
    ///
    /// # Arguments
    /// * `rt` - CUDA runtime
    /// * `n_neurons` - Total number of neurons
    /// * `n_inputs` - Number of input neurons (first n_inputs neurons)
    /// * `n_outputs` - Number of output neurons (last n_outputs neurons)
    /// * `threshold` - Firing threshold for all neurons (Q8.8, default 256 = 1.0)
    /// * `leak` - Leak factor for all neurons (Q0.8, default 230 ≈ 0.9)
    /// * `synapses` - CSR-format synapse connectivity
    pub fn new(
        rt: &CudaRuntime,
        n_neurons: usize,
        n_inputs: usize,
        n_outputs: usize,
        threshold: i16,
        leak: u8,
        synapses: &SynapseCSRData,
    ) -> CudaResult<Self> {
        // Compile kernels
        let kernels = compile_fused_kernels(rt.context())?;

        // Allocate neuron state buffers
        let d_v_mem = rt.alloc_zeros::<i16>(n_neurons)?;
        let d_threshold = {
            let data = vec![threshold; n_neurons];
            rt.upload(&data)?
        };
        let d_leak = {
            let data = vec![leak; n_neurons];
            rt.upload(&data)?
        };
        let d_refract = rt.alloc_zeros::<u8>(n_neurons)?;
        let d_spiked = rt.alloc_zeros::<u8>(n_neurons)?;
        let d_currents = rt.alloc_zeros::<i32>(n_neurons)?;

        // Upload synapse CSR
        let d_syn_ptr = rt.upload(&synapses.syn_ptr)?;
        let d_targets = rt.upload(&synapses.targets)?;
        let d_weights = rt.upload(&synapses.weights)?;

        // Allocate input/output buffers
        let d_input_rates = rt.alloc_zeros::<u8>(n_inputs)?;
        let d_output_counts = rt.alloc_zeros::<u32>(n_outputs)?;

        // Allocate eligibility trace buffer for R-STDP learning
        let n_synapses = synapses.n_synapses();
        let d_eligibility = rt.alloc_zeros::<f32>(n_synapses)?;

        // Allocate spike time tracking buffer for timing-based STDP
        let d_last_spike_time = rt.alloc_zeros::<u32>(n_neurons)?;

        let output_start = n_neurons - n_outputs;

        Ok(Self {
            d_v_mem,
            d_threshold,
            d_leak,
            d_refract,
            d_spiked,
            d_currents,
            d_syn_ptr,
            d_targets,
            d_weights,
            d_input_rates,
            d_output_counts,
            // R-STDP learning
            d_eligibility,
            d_last_spike_time,
            n_synapses: n_synapses as u32,
            learning_enabled: false, // disabled by default
            use_timing_stdp: true,   // use timing-based STDP by default
            tau_decay: 0.95,
            ltp_rate: 0.1,
            ltd_rate: 0.05,
            learning_rate: 0.01,
            stdp_tau_plus: 20.0,  // 20ms time constant for LTP
            stdp_tau_minus: 20.0, // 20ms time constant for LTD
            stdp_window: 30,      // 30 tick timing window
            kernels,
            n_neurons: n_neurons as u32,
            n_inputs: n_inputs as u32,
            n_outputs: n_outputs as u32,
            output_start: output_start as u32,
            tick: 0,
        })
    }

    /// Set input spike rates (upload once, then fire-and-forget)
    pub fn set_input_rates(&mut self, rt: &CudaRuntime, rates: &[u8]) -> CudaResult<()> {
        assert_eq!(rates.len(), self.n_inputs as usize);
        self.d_input_rates = rt.upload(rates)?;
        Ok(())
    }

    /// Run one tick on GPU (minimal kernel launches)
    ///
    /// Execution order:
    /// 1. Accumulate currents from previous tick's spikes
    /// 2. Input generation + LIF dynamics (fused)
    ///
    /// No synchronization between kernels - they queue up for execution.
    pub fn tick(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        let block_size = 256u32;
        let grid_neurons = (self.n_neurons + block_size - 1) / block_size;

        let cfg_neurons = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_neurons, 1, 1),
            shared_mem_bytes: 0,
        };

        // Kernel 1: Accumulate currents from previous tick's spikes
        // (Must run first so currents are ready for dynamics)
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.accumulate_fn)
                .arg(&self.d_syn_ptr)
                .arg(&self.d_targets)
                .arg(&self.d_weights)
                .arg(&self.d_spiked)
                .arg(&self.d_currents)
                .arg(&self.n_neurons)
                .launch(cfg_neurons)
                .map_err(|e| CudaError::LaunchFailed(format!("accumulate: {:?}", e)))?;
        }

        // Kernel 2: Fused input generation + LIF dynamics
        let seed = self.tick.wrapping_mul(2654435761) as u32;
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.input_dynamics_fn)
                .arg(&self.d_v_mem)
                .arg(&self.d_threshold)
                .arg(&self.d_leak)
                .arg(&self.d_refract)
                .arg(&self.d_spiked)
                .arg(&self.d_currents)
                .arg(&self.d_input_rates)
                .arg(&self.n_neurons)
                .arg(&self.n_inputs)
                .arg(&seed)
                .launch(cfg_neurons)
                .map_err(|e| CudaError::LaunchFailed(format!("input_dynamics: {:?}", e)))?;
        }

        self.tick += 1;
        Ok(())
    }

    /// Run N ticks without any synchronization (fire-and-forget!)
    ///
    /// This is the key EPIC 97-style optimization: queue up many ticks
    /// and let the GPU execute them without CPU involvement.
    pub fn tick_many(&mut self, rt: &CudaRuntime, n_ticks: usize) -> CudaResult<()> {
        for _ in 0..n_ticks {
            self.tick(rt)?;
        }
        Ok(())
    }

    /// Run N ticks WITH output counting each tick (for RL/benchmarks)
    ///
    /// This accumulates output spike counts across all ticks.
    /// Slightly less efficient than tick_many() due to extra kernel,
    /// but necessary when you need total output spike counts.
    pub fn tick_many_with_counting(&mut self, rt: &CudaRuntime, n_ticks: usize) -> CudaResult<()> {
        for _ in 0..n_ticks {
            self.tick(rt)?;
            self.count_outputs(rt)?;
        }
        Ok(())
    }

    /// Count output spikes (call after tick/tick_many, before sync)
    pub fn count_outputs(&self, rt: &CudaRuntime) -> CudaResult<()> {
        let block_size = 256u32;
        let grid_outputs = (self.n_outputs + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_outputs.max(1), 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.count_outputs_fn)
                .arg(&self.d_spiked)
                .arg(&self.d_output_counts)
                .arg(&self.output_start)
                .arg(&self.n_outputs)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("count_outputs: {:?}", e)))?;
        }

        Ok(())
    }

    /// Reset output spike counts
    pub fn reset_output_counts(&self, rt: &CudaRuntime) -> CudaResult<()> {
        let block_size = 256u32;
        let grid_outputs = (self.n_outputs + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_outputs.max(1), 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.reset_outputs_fn)
                .arg(&self.d_output_counts)
                .arg(&self.n_outputs)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("reset_outputs: {:?}", e)))?;
        }

        Ok(())
    }

    /// Reset neuron state for new episode.
    ///
    /// Resets membrane potentials, spike flags, currents, and refractory counters
    /// to their initial zero values. Used for Gymnasium episode reset.
    pub fn reset_state(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        // Reset all neuron state to zeros
        self.d_v_mem = rt.alloc_zeros::<i16>(self.n_neurons as usize)?;
        self.d_spiked = rt.alloc_zeros::<u8>(self.n_neurons as usize)?;
        self.d_currents = rt.alloc_zeros::<i32>(self.n_neurons as usize)?;
        self.d_refract = rt.alloc_zeros::<u8>(self.n_neurons as usize)?;
        Ok(())
    }

    /// Set per-neuron thresholds (upload new threshold array)
    ///
    /// Feedback insight: "threshold is a numerical gain knob, not a membrane voltage"
    /// Per-neuron thresholds enable:
    /// - Load balancing spike rates (avoid dead neurons and atomic storms)
    /// - Breaking output symmetry for varied decisions
    /// - Dynamic range control
    pub fn set_thresholds(&mut self, rt: &CudaRuntime, thresholds: &[i16]) -> CudaResult<()> {
        assert_eq!(thresholds.len(), self.n_neurons as usize);
        self.d_threshold = rt.upload(thresholds)?;
        Ok(())
    }

    /// Apply asymmetric thresholds to output neurons for symmetry breaking
    ///
    /// Creates a spread of thresholds around base_threshold for output neurons.
    /// This breaks the uniform output distribution that results from symmetric
    /// random connectivity with large networks.
    ///
    /// # Arguments
    /// * `base_threshold` - Center threshold value (Q8.8 format)
    /// * `spread` - Range of variation (±spread around base)
    /// * `seed` - Random seed for asymmetry pattern
    pub fn apply_output_threshold_asymmetry(
        &mut self,
        rt: &CudaRuntime,
        base_threshold: i16,
        spread: i16,
        seed: u64,
    ) -> CudaResult<()> {
        // Download current thresholds
        let mut thresholds = rt.download(&self.d_threshold)?;

        // Apply asymmetric thresholds to output neurons only
        let output_start = self.n_neurons as usize - self.n_outputs as usize;
        let mut rng = seed;

        for i in 0..self.n_outputs as usize {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Deterministic but varied: each output gets different threshold
            let offset = ((rng >> 32) as i32 % (spread as i32 * 2 + 1)) - spread as i32;
            thresholds[output_start + i] = (base_threshold as i32 + offset)
                .max(1) // Never allow threshold ≤ 0
                .min(32767) as i16;
        }

        // Upload modified thresholds
        self.d_threshold = rt.upload(&thresholds)?;
        Ok(())
    }

    /// Synchronize GPU (wait for all queued operations to complete)
    pub fn synchronize(&self, rt: &CudaRuntime) -> CudaResult<()> {
        rt.synchronize()
    }

    /// Download state snapshot (expensive, do sparingly)
    pub fn snapshot(&self, rt: &CudaRuntime) -> CudaResult<SNNSnapshot> {
        let v_mem = rt.download(&self.d_v_mem)?;
        let spiked = rt.download(&self.d_spiked)?;
        let output_counts = rt.download(&self.d_output_counts)?;

        Ok(SNNSnapshot {
            v_mem,
            spiked,
            output_counts,
            tick: self.tick,
        })
    }

    /// Get output spike counts (download only output counts)
    pub fn get_output_counts(&self, rt: &CudaRuntime) -> CudaResult<Vec<u32>> {
        rt.download(&self.d_output_counts)
    }

    /// Get winner-takes-all decision (most active output neuron)
    pub fn get_decision(&self, rt: &CudaRuntime) -> CudaResult<usize> {
        let counts = self.get_output_counts(rt)?;
        let (best_idx, _) = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .unwrap_or((0, &0));
        Ok(best_idx)
    }

    // Getters
    pub fn n_neurons(&self) -> usize {
        self.n_neurons as usize
    }
    pub fn n_inputs(&self) -> usize {
        self.n_inputs as usize
    }
    pub fn n_outputs(&self) -> usize {
        self.n_outputs as usize
    }
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    // ========================================================================
    // R-STDP Learning Methods
    // ========================================================================

    /// Enable R-STDP learning
    pub fn enable_learning(&mut self) {
        self.learning_enabled = true;
    }

    /// Disable R-STDP learning
    pub fn disable_learning(&mut self) {
        self.learning_enabled = false;
    }

    /// Check if learning is enabled
    pub fn is_learning_enabled(&self) -> bool {
        self.learning_enabled
    }

    /// Set R-STDP learning parameters
    ///
    /// # Arguments
    /// * `tau_decay` - Eligibility decay factor per tick (default: 0.95)
    /// * `ltp_rate` - LTP increment when pre+post spike (default: 0.1)
    /// * `ltd_rate` - LTD decrement when pre without post (default: 0.05)
    /// * `learning_rate` - Weight update rate (default: 0.01)
    pub fn set_learning_params(
        &mut self,
        tau_decay: f32,
        ltp_rate: f32,
        ltd_rate: f32,
        learning_rate: f32,
    ) {
        self.tau_decay = tau_decay;
        self.ltp_rate = ltp_rate;
        self.ltd_rate = ltd_rate;
        self.learning_rate = learning_rate;
    }

    /// Enable timing-based STDP (three-factor learning)
    ///
    /// When enabled, eligibility traces depend on the relative timing of
    /// pre and post spikes, not just co-occurrence. This creates differentiated
    /// eligibility traces that can solve the "cold start" problem.
    pub fn enable_timing_stdp(&mut self) {
        self.use_timing_stdp = true;
    }

    /// Disable timing-based STDP (use legacy co-occurrence rule)
    pub fn disable_timing_stdp(&mut self) {
        self.use_timing_stdp = false;
    }

    /// Check if timing-based STDP is enabled
    pub fn is_timing_stdp_enabled(&self) -> bool {
        self.use_timing_stdp
    }

    /// Set STDP timing parameters
    ///
    /// # Arguments
    /// * `tau_plus` - LTP time constant in ticks (how fast LTP decays with timing)
    /// * `tau_minus` - LTD time constant in ticks (how fast LTD decays with timing)
    /// * `window` - Maximum timing window in ticks (spikes further apart are ignored)
    pub fn set_stdp_timing_params(&mut self, tau_plus: f32, tau_minus: f32, window: u32) {
        self.stdp_tau_plus = tau_plus;
        self.stdp_tau_minus = tau_minus;
        self.stdp_window = window;
    }

    /// Reset spike times (call at episode start)
    pub fn reset_spike_times(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        // Zero out the spike time buffer
        let zeros = vec![0u32; self.n_neurons as usize];
        self.d_last_spike_time = rt.upload(&zeros)?;
        Ok(())
    }

    /// Update eligibility traces based on current spike pattern
    ///
    /// Call this after each tick to accumulate eligibility.
    /// If use_timing_stdp is true: Uses timing-based STDP with proper timing curves
    /// If use_timing_stdp is false: Uses legacy co-occurrence rule
    pub fn update_eligibility(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        if !self.learning_enabled {
            return Ok(());
        }

        let block_size = 256u32;
        let grid_neurons = (self.n_neurons + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_neurons.max(1), 1, 1),
            shared_mem_bytes: 0,
        };

        let current_tick = self.tick as u32;

        if self.use_timing_stdp {
            // First, update spike times for neurons that spiked this tick
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.update_spike_times_fn)
                    .arg(&self.d_spiked)
                    .arg(&self.d_last_spike_time)
                    .arg(&self.n_neurons)
                    .arg(&current_tick)
                    .launch(cfg.clone())
                    .map_err(|e| CudaError::LaunchFailed(format!("update_spike_times: {:?}", e)))?;
            }

            // Then, update eligibility using timing-based STDP
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.update_eligibility_timing_fn)
                    .arg(&self.d_syn_ptr)
                    .arg(&self.d_targets)
                    .arg(&self.d_last_spike_time)
                    .arg(&self.d_eligibility)
                    .arg(&self.n_neurons)
                    .arg(&current_tick)
                    .arg(&self.tau_decay)
                    .arg(&self.ltp_rate)
                    .arg(&self.ltd_rate)
                    .arg(&self.stdp_tau_plus)
                    .arg(&self.stdp_tau_minus)
                    .arg(&self.stdp_window)
                    .launch(cfg)
                    .map_err(|e| {
                        CudaError::LaunchFailed(format!("update_eligibility_timing: {:?}", e))
                    })?;
            }
        } else {
            // Legacy co-occurrence based eligibility
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.update_eligibility_fn)
                    .arg(&self.d_syn_ptr)
                    .arg(&self.d_targets)
                    .arg(&self.d_spiked)
                    .arg(&self.d_eligibility)
                    .arg(&self.n_neurons)
                    .arg(&self.tau_decay)
                    .arg(&self.ltp_rate)
                    .arg(&self.ltd_rate)
                    .launch(cfg)
                    .map_err(|e| CudaError::LaunchFailed(format!("update_eligibility: {:?}", e)))?;
            }
        }

        Ok(())
    }

    /// Apply reward signal to update weights (R-STDP)
    ///
    /// Weight update: Δw = learning_rate * eligibility * reward
    /// Call at decision boundaries (after each step).
    ///
    /// # Arguments
    /// * `reward` - Scalar reward signal, typically in [-1, 1]
    pub fn apply_reward(&mut self, rt: &CudaRuntime, reward: f32) -> CudaResult<()> {
        if !self.learning_enabled {
            return Ok(());
        }

        let block_size = 256u32;
        let grid_synapses = (self.n_synapses + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_synapses.max(1), 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.apply_reward_fn)
                .arg(&self.d_weights)
                .arg(&self.d_eligibility)
                .arg(&self.n_synapses)
                .arg(&reward)
                .arg(&self.learning_rate)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("apply_reward: {:?}", e)))?;
        }

        Ok(())
    }

    /// Apply reward only to synapses targeting the chosen action (Winner-Takes-All)
    ///
    /// This is the key for proper RL credit assignment:
    /// - Synapses to the chosen output get full reward
    /// - Synapses to other outputs get anti_reward_factor * reward
    /// - Non-output synapses are not updated
    ///
    /// # Arguments
    /// * `chosen_action` - Which output neuron won (0 to n_outputs-1)
    /// * `reward` - Reward signal for the chosen action
    /// * `anti_reward_factor` - Factor for non-chosen outputs (e.g., -0.5 means losers get opposite reward)
    pub fn apply_reward_action_specific(
        &mut self,
        rt: &CudaRuntime,
        chosen_action: usize,
        reward: f32,
        anti_reward_factor: f32,
    ) -> CudaResult<()> {
        if !self.learning_enabled {
            return Ok(());
        }

        assert!(chosen_action < self.n_outputs as usize);

        let block_size = 256u32;
        let grid_synapses = (self.n_synapses + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_synapses.max(1), 1, 1),
            shared_mem_bytes: 0,
        };

        let chosen = chosen_action as u32;

        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.apply_reward_action_specific_fn)
                .arg(&self.d_weights)
                .arg(&self.d_eligibility)
                .arg(&self.d_targets)
                .arg(&self.n_synapses)
                .arg(&self.output_start)
                .arg(&self.n_outputs)
                .arg(&chosen)
                .arg(&reward)
                .arg(&anti_reward_factor)
                .arg(&self.learning_rate)
                .launch(cfg)
                .map_err(|e| {
                    CudaError::LaunchFailed(format!("apply_reward_action_specific: {:?}", e))
                })?;
        }

        Ok(())
    }

    /// Apply reward with pathway tracing (softer credit assignment)
    ///
    /// Updates all synapses but scales reward based on whether they
    /// contributed to the winner or loser outputs:
    /// - Direct to winner: full reward
    /// - Direct to loser: -0.3 * reward
    /// - Hidden layer: 0.1 * reward
    pub fn apply_reward_pathway(
        &mut self,
        rt: &CudaRuntime,
        chosen_action: usize,
        reward: f32,
    ) -> CudaResult<()> {
        if !self.learning_enabled {
            return Ok(());
        }

        assert!(chosen_action < self.n_outputs as usize);

        let block_size = 256u32;
        let grid_synapses = (self.n_synapses + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_synapses.max(1), 1, 1),
            shared_mem_bytes: 0,
        };

        let chosen = chosen_action as u32;

        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.apply_reward_pathway_fn)
                .arg(&self.d_weights)
                .arg(&self.d_eligibility)
                .arg(&self.d_targets)
                .arg(&self.d_spiked)
                .arg(&self.n_synapses)
                .arg(&self.output_start)
                .arg(&self.n_outputs)
                .arg(&chosen)
                .arg(&reward)
                .arg(&self.learning_rate)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("apply_reward_pathway: {:?}", e)))?;
        }

        Ok(())
    }

    /// Reset eligibility traces to zero
    ///
    /// Call at episode boundaries.
    pub fn reset_eligibility(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        let block_size = 256u32;
        let grid_synapses = (self.n_synapses + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_synapses.max(1), 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.reset_eligibility_fn)
                .arg(&self.d_eligibility)
                .arg(&self.n_synapses)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("reset_eligibility: {:?}", e)))?;
        }

        Ok(())
    }

    /// Run ticks with learning (eligibility update after each tick)
    ///
    /// This combines tick execution with eligibility trace updates.
    /// More efficient than calling tick() + update_eligibility() separately.
    pub fn tick_many_with_learning(&mut self, rt: &CudaRuntime, n_ticks: usize) -> CudaResult<()> {
        for _ in 0..n_ticks {
            self.tick(rt)?;
            if self.learning_enabled {
                self.update_eligibility(rt)?;
            }
        }
        Ok(())
    }

    /// Complete learning step: run ticks, count outputs, update eligibility
    ///
    /// Combines tick_many_with_counting behavior with eligibility updates.
    pub fn tick_many_with_counting_and_learning(
        &mut self,
        rt: &CudaRuntime,
        n_ticks: usize,
    ) -> CudaResult<()> {
        for _ in 0..n_ticks {
            self.tick(rt)?;
            self.count_outputs(rt)?;
            if self.learning_enabled {
                self.update_eligibility(rt)?;
            }
        }
        Ok(())
    }
}

/// Generate random sparse CSR connectivity (uniform random, may not propagate to outputs)
pub fn generate_random_csr(
    n_neurons: usize,
    synapses_per_neuron: usize,
    seed: u64,
) -> SynapseCSRData {
    let mut syn_ptr = Vec::with_capacity(n_neurons + 1);
    let mut targets = Vec::with_capacity(n_neurons * synapses_per_neuron);
    let mut weights = Vec::with_capacity(n_neurons * synapses_per_neuron);

    let mut rng = seed;
    let mut ptr = 0u32;

    for src in 0..n_neurons {
        syn_ptr.push(ptr);

        for _ in 0..synapses_per_neuron {
            // Simple LCG
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let tgt = ((rng >> 32) as usize) % n_neurons;

            if tgt != src {
                targets.push(tgt as u32);
                // Random weight in [-64, 63] range
                weights.push(((rng >> 48) as i8).max(-64).min(63));
                ptr += 1;
            }
        }
    }
    syn_ptr.push(ptr);

    SynapseCSRData {
        syn_ptr,
        targets,
        weights,
    }
}

/// Generate feedforward layered CSR connectivity
///
/// Creates proper input → hidden → output pathways:
/// - Input neurons connect to hidden layer
/// - Hidden neurons connect to other hidden neurons and outputs
/// - Uses positive weights to ensure excitation propagates
pub fn generate_layered_csr(
    n_neurons: usize,
    n_inputs: usize,
    n_outputs: usize,
    synapses_per_neuron: usize,
    seed: u64,
) -> SynapseCSRData {
    let mut syn_ptr = Vec::with_capacity(n_neurons + 1);
    let mut targets = Vec::with_capacity(n_neurons * synapses_per_neuron);
    let mut weights = Vec::with_capacity(n_neurons * synapses_per_neuron);

    let mut rng = seed;
    let mut ptr = 0u32;

    let n_hidden = n_neurons - n_inputs - n_outputs;
    let hidden_start = n_inputs;
    let output_start = n_neurons - n_outputs;

    for src in 0..n_neurons {
        syn_ptr.push(ptr);

        if src < n_inputs {
            // Input neurons → hidden neurons (feedforward)
            for _ in 0..synapses_per_neuron {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let tgt = hidden_start + ((rng >> 32) as usize % n_hidden);

                targets.push(tgt as u32);
                // Positive weights for excitation (20-60 range)
                weights.push(20 + ((rng >> 48) as i8).unsigned_abs() as i8 % 40);
                ptr += 1;
            }
        } else if src < output_start {
            // Hidden neurons → hidden + output neurons
            let half = synapses_per_neuron / 2;

            // Half to other hidden neurons (recurrent within hidden layer)
            for _ in 0..half {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let tgt = hidden_start + ((rng >> 32) as usize % n_hidden);

                if tgt != src {
                    targets.push(tgt as u32);
                    // Mix of excitatory and inhibitory
                    let w = ((rng >> 48) as i8).max(-40).min(50);
                    weights.push(w);
                    ptr += 1;
                }
            }

            // Half to output neurons (feedforward to outputs)
            for _ in 0..(synapses_per_neuron - half) {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let tgt = output_start + ((rng >> 32) as usize % n_outputs);

                targets.push(tgt as u32);
                // Strong positive weights to outputs
                weights.push(30 + ((rng >> 48) as i8).unsigned_abs() as i8 % 30);
                ptr += 1;
            }
        }
        // Output neurons have no outgoing synapses (they're the final layer)
    }
    syn_ptr.push(ptr);

    SynapseCSRData {
        syn_ptr,
        targets,
        weights,
    }
}

/// Generate dense feedforward CSR for maximum output activity
///
/// Creates high-connectivity feedforward network optimized for spike propagation
pub fn generate_dense_feedforward_csr(
    n_neurons: usize,
    n_inputs: usize,
    n_outputs: usize,
    seed: u64,
) -> SynapseCSRData {
    let mut syn_ptr = Vec::with_capacity(n_neurons + 1);
    let mut targets = Vec::new();
    let mut weights = Vec::new();

    let mut rng = seed;
    let mut ptr = 0u32;

    let n_hidden = n_neurons - n_inputs - n_outputs;
    let hidden_start = n_inputs;
    let output_start = n_neurons - n_outputs;

    // Target ~20 synapses per neuron for dense connectivity
    let input_to_hidden = 20.min(n_hidden);
    let hidden_to_hidden = 10.min(n_hidden);
    let hidden_to_output = 10.min(n_outputs);

    for src in 0..n_neurons {
        syn_ptr.push(ptr);

        if src < n_inputs {
            // Input → Hidden (dense, all positive)
            for i in 0..input_to_hidden {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let tgt = hidden_start + ((src * input_to_hidden + i) % n_hidden);

                targets.push(tgt as u32);
                weights.push(40 + (rng >> 56) as i8 % 20); // Strong positive [40-59]
                ptr += 1;
            }
        } else if src < output_start {
            // Hidden → Hidden (sparse recurrent)
            for _ in 0..hidden_to_hidden {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let tgt = hidden_start + ((rng >> 32) as usize % n_hidden);

                if tgt != src {
                    targets.push(tgt as u32);
                    // Mostly positive with some inhibition
                    let w = if (rng >> 16) % 4 == 0 {
                        -(((rng >> 48) as u8 % 30) as i8) // 25% inhibitory [-29, 0]
                    } else {
                        20 + ((rng >> 48) as u8 % 30) as i8 // 75% excitatory [20-49]
                    };
                    weights.push(w);
                    ptr += 1;
                }
            }

            // Hidden → Output (random selection for varied decisions)
            for _ in 0..hidden_to_output {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let tgt = output_start + ((rng >> 32) as usize % n_outputs);

                targets.push(tgt as u32);
                weights.push(50 + (rng >> 56) as i8 % 20); // Very strong [50-69]
                ptr += 1;
            }
        }
        // Output neurons: no outgoing synapses
    }
    syn_ptr.push(ptr);

    SynapseCSRData {
        syn_ptr,
        targets,
        weights,
    }
}

/// Generate columnar connectivity optimized for GPU block structure
///
/// Key insight from feedback: GPUs reward structure. Columnar organization gives:
/// - Predictable memory access (neurons in same column have nearby targets)
/// - Shared targets per block (natural for shared memory staging)
/// - Easy alignment: block_size = column_size
///
/// Architecture:
/// - Neurons organized into columns of `column_size` neurons
/// - Intra-column: dense local connections (high locality)
/// - Inter-column: sparse lateral connections (structured fan-out)
/// - Output neurons receive from dedicated "output columns"
pub fn generate_columnar_csr(
    n_neurons: usize,
    n_inputs: usize,
    n_outputs: usize,
    column_size: usize,           // Align with GPU block size (e.g., 256)
    intra_column_density: f32,    // 0.0-1.0, connections within column
    inter_column_synapses: usize, // connections to other columns
    seed: u64,
) -> SynapseCSRData {
    let mut syn_ptr = Vec::with_capacity(n_neurons + 1);
    let mut targets = Vec::new();
    let mut weights = Vec::new();

    let mut rng = seed;
    let mut ptr = 0u32;

    let n_hidden = n_neurons - n_inputs - n_outputs;
    let hidden_start = n_inputs;
    let output_start = n_neurons - n_outputs;

    // Calculate column assignments
    let n_hidden_columns = (n_hidden + column_size - 1) / column_size;
    let n_output_columns = (n_outputs + column_size - 1) / column_size;

    // Helper: get column index for a hidden neuron
    let get_column = |neuron_idx: usize| -> usize {
        if neuron_idx < hidden_start || neuron_idx >= output_start {
            return 0; // Not a hidden neuron
        }
        (neuron_idx - hidden_start) / column_size
    };

    // Helper: get neurons in a column
    let column_range = |col: usize| -> (usize, usize) {
        let start = hidden_start + col * column_size;
        let end = (start + column_size).min(output_start);
        (start, end)
    };

    for src in 0..n_neurons {
        syn_ptr.push(ptr);

        if src < n_inputs {
            // Input neurons → first few hidden columns (structured fan-in)
            // Each input targets neurons in a specific column range
            let target_column = (src * n_hidden_columns) / n_inputs;
            let (col_start, col_end) = column_range(target_column);
            let col_neurons = col_end - col_start;

            // Connect to ~20 neurons in target column
            let n_targets = 20.min(col_neurons);
            for i in 0..n_targets {
                let tgt = col_start + (i * col_neurons / n_targets);
                targets.push(tgt as u32);
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                weights.push(40 + (rng >> 56) as i8 % 20); // [40-59]
                ptr += 1;
            }
        } else if src < output_start {
            // Hidden neurons: intra-column + inter-column connections
            let my_column = get_column(src);
            let (my_col_start, my_col_end) = column_range(my_column);
            let my_col_size = my_col_end - my_col_start;

            // Intra-column connections (dense, local)
            let intra_targets = (my_col_size as f32 * intra_column_density) as usize;
            for _ in 0..intra_targets {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let offset = (rng >> 32) as usize % my_col_size;
                let tgt = my_col_start + offset;

                if tgt != src {
                    targets.push(tgt as u32);
                    // Intra-column: mix of excitatory/inhibitory
                    let w = if (rng >> 16) % 4 == 0 {
                        -(((rng >> 48) as u8 % 30) as i8)
                    } else {
                        20 + ((rng >> 48) as u8 % 30) as i8
                    };
                    weights.push(w);
                    ptr += 1;
                }
            }

            // Inter-column connections (sparse, to nearby columns)
            for _ in 0..inter_column_synapses {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Prefer nearby columns (locality)
                let col_offset = ((rng >> 32) as usize % 5) as isize - 2; // -2 to +2
                let target_col =
                    ((my_column as isize + col_offset).max(0) as usize).min(n_hidden_columns - 1);

                if target_col != my_column {
                    let (tc_start, tc_end) = column_range(target_col);
                    let tc_size = tc_end - tc_start;
                    if tc_size > 0 {
                        let tgt = tc_start + ((rng >> 48) as usize % tc_size);
                        targets.push(tgt as u32);
                        weights.push(30 + (rng >> 56) as i8 % 20); // [30-49]
                        ptr += 1;
                    }
                }
            }

            // Output connections: last few columns connect to outputs
            // This creates dedicated "output columns" in the hidden layer
            let output_column_threshold = n_hidden_columns.saturating_sub(n_output_columns);
            if my_column >= output_column_threshold {
                // This column feeds outputs
                let output_idx = my_column - output_column_threshold;
                let outputs_per_column = (n_outputs + n_output_columns - 1) / n_output_columns;
                let out_start = output_idx * outputs_per_column;
                let out_end = ((output_idx + 1) * outputs_per_column).min(n_outputs);

                for out_i in out_start..out_end {
                    let tgt = output_start + out_i;
                    targets.push(tgt as u32);
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    weights.push(50 + (rng >> 56) as i8 % 20); // Strong [50-69]
                    ptr += 1;
                }
            }
        }
        // Output neurons: no outgoing synapses
    }
    syn_ptr.push(ptr);

    SynapseCSRData {
        syn_ptr,
        targets,
        weights,
    }
}

/// Generate direct input→output connectivity for simple classification
///
/// This creates a minimal network where inputs connect directly to outputs,
/// allowing R-STDP to learn simple input→output mappings without signal
/// dilution through hidden layers.
///
/// Architecture:
/// - Each input connects to ALL outputs (fully connected input→output)
/// - Small hidden layer for capacity (optional)
/// - Hidden neurons receive from inputs and project to outputs
///
/// Best for: Simple classification tasks, debugging R-STDP learning
pub fn generate_direct_csr(
    n_neurons: usize,
    n_inputs: usize,
    n_outputs: usize,
    seed: u64,
) -> SynapseCSRData {
    let mut syn_ptr = Vec::with_capacity(n_neurons + 1);
    let mut targets = Vec::new();
    let mut weights = Vec::new();

    let mut rng = seed;
    let mut ptr = 0u32;

    let n_hidden = n_neurons.saturating_sub(n_inputs + n_outputs);
    let hidden_start = n_inputs;
    let output_start = n_neurons - n_outputs;

    for src in 0..n_neurons {
        syn_ptr.push(ptr);

        if src < n_inputs {
            // Input neurons connect to ALL outputs
            //
            // POPULATION CODING MAPPING:
            // For typical RL setups with population-coded velocity:
            // - Input 1 (left indicator) -> strong to output 0 (left action)
            // - Input 2 (right indicator) -> strong to output n-1 (right action)
            // Other inputs have uniform random weights.
            //
            // This "pre-training" gives R-STDP a starting point to refine.
            for out_idx in 0..n_outputs {
                let tgt = output_start + out_idx;
                targets.push(tgt as u32);

                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);

                // Determine weight based on input-output mapping
                let w = if src == 1 && out_idx == 0 {
                    // Input 1 (left_vel) -> Output 0 (left action): STRONG
                    70 + ((rng >> 48) as u8 % 10) as i8
                } else if src == 2 && out_idx == n_outputs - 1 {
                    // Input 2 (right_vel) -> Output n-1 (right action): STRONG
                    70 + ((rng >> 48) as u8 % 10) as i8
                } else if (src == 1 && out_idx == n_outputs - 1) || (src == 2 && out_idx == 0) {
                    // Cross connections (left_vel->right, right_vel->left): WEAK
                    5 + ((rng >> 48) as u8 % 5) as i8
                } else {
                    // All other connections: moderate random
                    20 + ((rng >> 48) as u8 % 30) as i8
                };
                weights.push(w);
                ptr += 1;
            }

            // Also connect to some hidden neurons for capacity
            if n_hidden > 0 {
                let synapses_to_hidden = 5.min(n_hidden);
                for i in 0..synapses_to_hidden {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let tgt = hidden_start + ((src * synapses_to_hidden + i) % n_hidden);
                    targets.push(tgt as u32);
                    weights.push(30 + (rng >> 56) as i8 % 20); // [30-49]
                    ptr += 1;
                }
            }
        } else if src < output_start {
            // Hidden neurons connect to outputs
            for out_idx in 0..n_outputs {
                let tgt = output_start + out_idx;
                targets.push(tgt as u32);

                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let w = 20 + ((rng >> 48) as u8 % 30) as i8; // [20-49]
                weights.push(w);
                ptr += 1;
            }
        } else {
            // OUTPUT NEURONS: Add LATERAL INHIBITION (WTA dynamics)
            // Each output neuron inhibits all OTHER output neurons
            // This creates winner-take-all: when one fires, it suppresses others
            let my_output_idx = src - output_start;

            for other_idx in 0..n_outputs {
                if other_idx != my_output_idx {
                    let tgt = output_start + other_idx;
                    targets.push(tgt as u32);

                    // VERY strong inhibitory weight (near maximum negative)
                    // This ensures winner suppresses all others completely
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let w = -120 - ((rng >> 48) as u8 % 8) as i8; // [-120, -128]
                    weights.push(w);
                    ptr += 1;
                }
            }
        }
    }
    syn_ptr.push(ptr);

    SynapseCSRData {
        syn_ptr,
        targets,
        weights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::is_cuda_available;

    #[test]
    fn test_fused_snn_creation() {
        if !is_cuda_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        let rt = CudaRuntime::new().unwrap();
        let synapses = generate_random_csr(1000, 10, 42);

        let snn = GpuFusedSNN::new(&rt, 1000, 100, 10, 256, 230, &synapses);
        assert!(snn.is_ok());

        let snn = snn.unwrap();
        assert_eq!(snn.n_neurons(), 1000);
        assert_eq!(snn.n_inputs(), 100);
        assert_eq!(snn.n_outputs(), 10);
    }

    #[test]
    fn test_fused_snn_tick() {
        if !is_cuda_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        let rt = CudaRuntime::new().unwrap();
        let synapses = generate_random_csr(1000, 10, 42);

        let mut snn = GpuFusedSNN::new(&rt, 1000, 100, 10, 256, 230, &synapses).unwrap();

        // Set input rates
        let rates: Vec<u8> = (0..100).map(|i| ((i * 2) % 256) as u8).collect();
        snn.set_input_rates(&rt, &rates).unwrap();

        // Run 10 ticks
        snn.tick_many(&rt, 10).unwrap();
        snn.count_outputs(&rt).unwrap();
        snn.synchronize(&rt).unwrap();

        // Check output counts
        let counts = snn.get_output_counts(&rt).unwrap();
        assert_eq!(counts.len(), 10);

        // Should have some activity
        let total: u32 = counts.iter().sum();
        println!("Total output spikes after 10 ticks: {}", total);
    }

    #[test]
    fn test_fused_snn_fire_and_forget() {
        if !is_cuda_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        let rt = CudaRuntime::new().unwrap();
        let synapses = generate_random_csr(10_000, 10, 42);

        let mut snn = GpuFusedSNN::new(&rt, 10_000, 1000, 10, 256, 230, &synapses).unwrap();

        // Set input rates
        let rates: Vec<u8> = (0..1000).map(|i| ((i * 3) % 256) as u8).collect();
        snn.set_input_rates(&rt, &rates).unwrap();

        // Fire-and-forget: 1000 ticks
        snn.tick_many(&rt, 1000).unwrap();
        snn.count_outputs(&rt).unwrap();
        snn.synchronize(&rt).unwrap();

        let snapshot = snn.snapshot(&rt).unwrap();
        assert_eq!(snapshot.tick, 1000);
    }
}

// =============================================================================
// Milestone 3: Discriminative Channel CSR Generator
// =============================================================================

/// Generate CSR connectivity for a multi-class discriminative channel SNN.
///
/// Mirrors the CPU `build_excitatory_channels` topology with explicit channel
/// isolation: each input channel exclusively drives its matching hidden channel,
/// and hidden neurons connect only to their class output neuron.
///
/// ## Network layout
///
/// ```text
/// Input:   [0,           n_inputs)             n_inputs = k_per_class * n_classes
/// Hidden:  [n_inputs,    n_inputs + n_hidden)   n_hidden = hidden_per_class * n_classes
/// Output:  [n_inputs+n_hidden, n_total)         n_total  = n_inputs + n_hidden + n_classes
/// ```
///
/// ## Connectivity
///
/// - Input neuron `i` (channel `c = i / k_per_class`) connects to hidden neurons
///   in channel `c` with probability `connection_prob`.
/// - Hidden neuron `h` (channel `c = h_idx / hidden_per_class`) connects to
///   output neuron `c` with probability `hidden_out_prob`.
///
/// ## Arguments
///
/// * `k_per_class`       — K discriminative pixels per class (= input neurons per channel).
/// * `hidden_per_class`  — H hidden neurons per class.
/// * `n_classes`         — Number of output classes.
/// * `connection_prob`   — Probability of input→hidden connection within a channel.
/// * `hidden_out_prob`   — Probability of hidden→output connection within a channel.
/// * `w_in_lo/hi`        — Weight range (i8) for input→hidden synapses.
/// * `w_out_lo/hi`       — Weight range (i8) for hidden→output synapses.
/// * `seed`              — RNG seed for reproducibility.
///
/// ## Returns
///
/// `SynapseCSRData` for use with `GpuFusedSNN::new` or `GpuBatchSNN::new`.
pub fn generate_discriminative_channel_csr(
    k_per_class: usize,
    hidden_per_class: usize,
    n_classes: usize,
    connection_prob: f32,
    hidden_out_prob: f32,
    w_in_lo: i8,
    w_in_hi: i8,
    w_out_lo: i8,
    w_out_hi: i8,
    seed: u64,
) -> SynapseCSRData {
    let n_inputs = k_per_class * n_classes;
    let n_hidden = hidden_per_class * n_classes;
    let n_neurons = n_inputs + n_hidden + n_classes;

    let hidden_start = n_inputs;
    let output_start = n_inputs + n_hidden;

    let mut syn_ptr = vec![0u32; n_neurons + 1];
    let mut targets: Vec<u32> = Vec::new();
    let mut weights: Vec<i8> = Vec::new();

    // LCG random float [0, 1)
    let mut rng = seed;
    let mut rng_f = |r: &mut u64| -> f32 {
        *r = r
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*r >> 33) as f32) / (u32::MAX as f32)
    };

    // ── Input → Hidden (same channel only) ────────────────────────────────
    for c in 0..n_classes {
        let inp_start = c * k_per_class;
        let inp_end = inp_start + k_per_class;
        let hid_start = hidden_start + c * hidden_per_class;
        let hid_end = hid_start + hidden_per_class;

        for src in inp_start..inp_end {
            for dst in hid_start..hid_end {
                if rng_f(&mut rng) < connection_prob {
                    targets.push(dst as u32);
                    let w = w_in_lo as f32 + rng_f(&mut rng) * (w_in_hi - w_in_lo) as f32;
                    weights.push(w.clamp(-128.0, 127.0) as i8);
                }
            }
            syn_ptr[src + 1] = targets.len() as u32;
        }
    }

    // ── Hidden → Output (same channel only) ───────────────────────────────
    for h_idx in 0..n_hidden {
        let src = hidden_start + h_idx;
        let c = h_idx / hidden_per_class;
        let out_neuron = output_start + c;

        if rng_f(&mut rng) < hidden_out_prob {
            targets.push(out_neuron as u32);
            let w = w_out_lo as f32 + rng_f(&mut rng) * (w_out_hi - w_out_lo) as f32;
            weights.push(w.clamp(-128.0, 127.0) as i8);
        }
        syn_ptr[src + 1] = targets.len() as u32;
    }

    // ── Output neurons: no outgoing synapses ──────────────────────────────
    for o in 0..n_classes {
        syn_ptr[output_start + o + 1] = targets.len() as u32;
    }

    SynapseCSRData {
        syn_ptr,
        targets,
        weights,
    }
}

/// Generate a **2-layer** discriminative channel CSR (input → hidden only).
///
/// No output neuron layer — the hidden layer IS the readout.
/// Compute Fisher LDA discriminative scores for pixel selection.
///
/// For each class `c` and pixel `i`, returns the Fisher ratio:
///
/// ```text
/// score[c][i] = (avg[c][i] − mean_rest[i])²  /  (var[c][i] + var_rest[i])
/// ```
///
/// where `mean_rest` and `var_rest` are the mean and pooled variance across all
/// other classes.  Compared to the raw mean-difference score used in Phase 3a,
/// the Fisher ratio down-weights pixels that are noisy (high within-class variance)
/// even when their mean difference is large.
///
/// # Arguments
/// - `avgs`      : per-class pixel means `[n_classes][n_pixels]`
/// - `vars`      : per-class pixel variances `[n_classes][n_pixels]` (unbiased, N-1 denominator)
/// - `n_classes` : number of classes
/// - `n_pixels`  : number of pixels (784 for MNIST)
///
/// # Returns
/// Fisher scores `[n_classes][n_pixels]`.  Use `.iter().enumerate().sort_by…` to rank
/// pixels by score and take the top-K for each class.
pub fn fisher_discriminative_scores(
    avgs: &[[f32; 784]],
    vars: &[[f32; 784]],
    n_classes: usize,
    n_pixels: usize,
) -> Vec<Vec<f32>> {
    assert_eq!(avgs.len(), n_classes);
    assert_eq!(vars.len(), n_classes);

    let mut scores = vec![vec![0.0f32; n_pixels]; n_classes];

    for c in 0..n_classes {
        let n_others = (n_classes - 1) as f32;
        for i in 0..n_pixels {
            // Mean and variance pooled across all *other* classes
            let mean_rest: f32 = avgs
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != c)
                .map(|(_, a)| a[i])
                .sum::<f32>()
                / n_others;

            let var_rest: f32 = vars
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != c)
                .map(|(_, v)| v[i])
                .sum::<f32>()
                / n_others;

            let diff = avgs[c][i] - mean_rest;
            let denom = (vars[c][i] + var_rest).max(1e-6);
            scores[c][i] = diff * diff / denom;
        }
    }

    scores
}

/// `n_neurons = k_per_class * n_classes + hidden_per_class * n_classes`.
///
/// Use with `GpuBatchSNN::new` where:
/// - `n_outputs = hidden_per_class * n_classes`  (entire hidden layer)
/// - `output_start` is computed as `n_neurons - n_outputs = n_inputs`
///
/// Prediction: for batch instance `b` and class `c`,
/// `channel_score[c] = sum(output_counts[b][c*H .. (c+1)*H])`;
/// predicted class = `argmax(channel_score)`.
/// Build a 2-layer discriminative channel CSR with **d_norm-weighted** input synapses.
///
/// Input synapse weight for pixel k in class c = `round(w_max × d_norms[c][k])`.
/// Pixels with high discriminativeness drive stronger currents, directly
/// matching the CPU baseline's `scale_input_weights` approach.
///
/// `d_norms` is a flat `n_classes × k_per_class` slice in row-major order
/// (`d_norms[c * k_per_class + k]`).
///
/// With d_norm-weighted synapses, use:
/// ```text
/// V_ss(cc, cj) = CONN_PROB × w_max × 2 × MAX_RATE
///                ──────────────────────────────────── × Σ_k avg[cj][pix_k] × d_norm[cc][k]
///                     255 × (256 − LEAK)
/// ```
pub fn generate_discriminative_2layer_csr_weighted(
    k_per_class: usize,
    hidden_per_class: usize,
    n_classes: usize,
    connection_prob: f32,
    w_max: i8,
    d_norms: &[f32], // flat [n_classes × k_per_class], row-major
    seed: u64,
) -> SynapseCSRData {
    assert_eq!(d_norms.len(), k_per_class * n_classes);
    let n_inputs = k_per_class * n_classes;
    let n_hidden = hidden_per_class * n_classes;
    let n_neurons = n_inputs + n_hidden;

    let mut syn_ptr = vec![0u32; n_neurons + 1];
    let mut targets: Vec<u32> = Vec::new();
    let mut weights: Vec<i8> = Vec::new();

    let mut rng = seed;
    let mut rng_f = |r: &mut u64| -> f32 {
        *r = r
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*r >> 33) as f32) / (u32::MAX as f32)
    };

    for c in 0..n_classes {
        let inp_start = c * k_per_class;
        let inp_end = inp_start + k_per_class;
        let hid_start = n_inputs + c * hidden_per_class;
        let hid_end = hid_start + hidden_per_class;

        for (local_k, src) in (inp_start..inp_end).enumerate() {
            let d = d_norms[c * k_per_class + local_k];
            let w = (w_max as f32 * d).clamp(1.0, 127.0) as i8;

            for dst in hid_start..hid_end {
                if rng_f(&mut rng) < connection_prob {
                    targets.push(dst as u32);
                    weights.push(w);
                }
            }
            syn_ptr[src + 1] = targets.len() as u32;
        }
    }

    for h in 0..n_hidden {
        syn_ptr[n_inputs + h + 1] = targets.len() as u32;
    }

    SynapseCSRData {
        syn_ptr,
        targets,
        weights,
    }
}

pub fn generate_discriminative_2layer_csr(
    k_per_class: usize,
    hidden_per_class: usize,
    n_classes: usize,
    connection_prob: f32,
    w_lo: i8,
    w_hi: i8,
    seed: u64,
) -> SynapseCSRData {
    let n_inputs = k_per_class * n_classes;
    let n_hidden = hidden_per_class * n_classes;
    let n_neurons = n_inputs + n_hidden;

    let mut syn_ptr = vec![0u32; n_neurons + 1];
    let mut targets: Vec<u32> = Vec::new();
    let mut weights: Vec<i8> = Vec::new();

    let mut rng = seed;
    let mut rng_f = |r: &mut u64| -> f32 {
        *r = r
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*r >> 33) as f32) / (u32::MAX as f32)
    };

    // ── Input → Hidden (same channel only) ────────────────────────────────
    for c in 0..n_classes {
        let inp_start = c * k_per_class;
        let inp_end = inp_start + k_per_class;
        let hid_start = n_inputs + c * hidden_per_class;
        let hid_end = hid_start + hidden_per_class;

        for src in inp_start..inp_end {
            for dst in hid_start..hid_end {
                if rng_f(&mut rng) < connection_prob {
                    targets.push(dst as u32);
                    let w = w_lo as f32 + rng_f(&mut rng) * (w_hi - w_lo) as f32;
                    weights.push(w.clamp(-128.0, 127.0) as i8);
                }
            }
            syn_ptr[src + 1] = targets.len() as u32;
        }
    }

    // ── Hidden neurons: no outgoing synapses ──────────────────────────────
    for h in 0..n_hidden {
        syn_ptr[n_inputs + h + 1] = targets.len() as u32;
    }

    SynapseCSRData {
        syn_ptr,
        targets,
        weights,
    }
}

/// Build a per-neuron threshold array for a **2-layer** discriminative channel SNN.
///
/// Layout:
/// * Input neurons `[0, n_inputs)`:     `input_threshold` (large).
/// * Hidden neurons per class `c`:      `t_per_class[c]`.
///
/// No output neuron layer — hidden IS the readout.
pub fn discriminative_2layer_thresholds(
    k_per_class: usize,
    hidden_per_class: usize,
    n_classes: usize,
    t_per_class: &[i16],
    input_threshold: i16,
) -> Vec<i16> {
    assert_eq!(t_per_class.len(), n_classes);
    let n_inputs = k_per_class * n_classes;
    let n_hidden = hidden_per_class * n_classes;
    let n_total = n_inputs + n_hidden;

    let mut thresholds = vec![0i16; n_total];
    for i in 0..n_inputs {
        thresholds[i] = input_threshold;
    }
    for c in 0..n_classes {
        let hs = n_inputs + c * hidden_per_class;
        let he = hs + hidden_per_class;
        for h in hs..he {
            thresholds[h] = t_per_class[c];
        }
    }
    thresholds
}

/// Build a 2-layer discriminative channel CSR with **d_norm-weighted** excitatory
/// input synapses **and** sparse cross-channel inhibitory connections.
///
/// Extends `generate_discriminative_2layer_csr_weighted` by adding inhibitory
/// synapses from each class's hidden pool to every other class's hidden pool.
/// This implements soft WTA competition: when the correct class fires, it
/// suppresses rival channels.
///
/// # Additional parameter vs `generate_discriminative_2layer_csr_weighted`
/// - `inhibit_cross`: inhibitory weight as a fraction of `w_max` (e.g. 0.3).
///   Synapses are added with weight `-(w_max × inhibit_cross).round()`.
///   A value of 0.0 produces identical output to `…_weighted`.
/// - `inhibit_prob`: connection probability for cross-channel inhibitory synapses
///   (recommended: 0.1 — sparse to avoid silencing correct class).
///
/// # Network size
/// Same as `generate_discriminative_2layer_csr_weighted`:
/// `n_neurons = k_per_class × n_classes + hidden_per_class × n_classes`
pub fn generate_discriminative_2layer_csr_inhibitory(
    k_per_class: usize,
    hidden_per_class: usize,
    n_classes: usize,
    connection_prob: f32,
    w_max: i8,
    d_norms: &[f32],    // flat [n_classes × k_per_class], row-major
    inhibit_cross: f32, // fraction of w_max for cross-channel inhibition
    inhibit_prob: f32,  // connection probability for inhibitory synapses
    seed: u64,
) -> SynapseCSRData {
    assert_eq!(d_norms.len(), k_per_class * n_classes);
    let n_inputs = k_per_class * n_classes;
    let n_hidden = hidden_per_class * n_classes;
    let n_neurons = n_inputs + n_hidden;
    let w_inhib = -((w_max as f32 * inhibit_cross).round().clamp(1.0, 127.0) as i8);

    let mut syn_ptr = vec![0u32; n_neurons + 1];
    let mut targets: Vec<u32> = Vec::new();
    let mut weights: Vec<i8> = Vec::new();

    let mut rng = seed;
    let mut rng_f = |r: &mut u64| -> f32 {
        *r = r
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*r >> 33) as f32) / (u32::MAX as f32)
    };

    // ── Input → Hidden (same channel, excitatory, d_norm-weighted) ────────────
    for c in 0..n_classes {
        let inp_start = c * k_per_class;
        let inp_end = inp_start + k_per_class;
        let hid_start = n_inputs + c * hidden_per_class;
        let hid_end = hid_start + hidden_per_class;

        for (local_k, src) in (inp_start..inp_end).enumerate() {
            let d = d_norms[c * k_per_class + local_k];
            let w = (w_max as f32 * d).clamp(1.0, 127.0) as i8;

            for dst in hid_start..hid_end {
                if rng_f(&mut rng) < connection_prob {
                    targets.push(dst as u32);
                    weights.push(w);
                }
            }
            syn_ptr[src + 1] = targets.len() as u32;
        }
    }

    // ── Hidden → Hidden (cross-channel, inhibitory) ────────────────────────────
    // For each class c: hidden neurons of c send inhibitory synapses to hidden
    // neurons of every other class j ≠ c.
    for c in 0..n_classes {
        let hid_src_start = n_inputs + c * hidden_per_class;
        let hid_src_end = hid_src_start + hidden_per_class;

        for src in hid_src_start..hid_src_end {
            for j in 0..n_classes {
                if j == c {
                    continue;
                }
                let hid_dst_start = n_inputs + j * hidden_per_class;
                let hid_dst_end = hid_dst_start + hidden_per_class;
                for dst in hid_dst_start..hid_dst_end {
                    if rng_f(&mut rng) < inhibit_prob {
                        targets.push(dst as u32);
                        weights.push(w_inhib);
                    }
                }
            }
            syn_ptr[src + 1] = targets.len() as u32;
        }
    }

    SynapseCSRData {
        syn_ptr,
        targets,
        weights,
    }
}

// ── Milestone 6: 3-layer discriminative CSR ───────────────────────────────────

/// Build a **3-layer** discriminative SNN synapse matrix in CSR format.
///
/// ## Architecture
///
/// ```text
/// Input  [k×C]   →  Hidden  [h×C]   →  Readout  [r×C]
///  (Fisher prior)   (discriminative)    (E-prop trains from scratch)
/// ```
///
/// Neuron layout (contiguous):
/// * `[0,             k×C)` — input  neurons (k per class, C classes)
/// * `[k×C,       k×C+h×C)` — hidden neurons (h per class)
/// * `[k×C+h×C, k×C+h×C+r×C)` — readout neurons (r per class)
///
/// ## Synapse layers
///
/// * **Input → Hidden** (Fisher-weighted, same class only): each input pixel `i`
///   of class `c` connects to hidden neurons of class `c` with probability
///   `conn_prob_ih`. Weight = `clamp(w_max_ih × d_norms[c][local_i], 1, 127)`.
/// * **Hidden → Readout** (class-aligned, uniform): each hidden neuron of class `c`
///   connects to readout neurons of class `c` with probability `conn_prob_hr`.
///   Weight = `w_max_hr` (uniform positive). E-prop trains these from scratch.
/// * **Readout**: no outgoing synapses.
///
/// ## Arguments
///
/// - `k_per_class`       — discriminative pixels per class (e.g. 150)
/// - `hidden_per_class`  — hidden neurons per class (e.g. 32)
/// - `readout_per_class` — readout neurons per class (e.g. 8)
/// - `n_classes`         — number of classes (e.g. 10)
/// - `conn_prob_ih`      — input→hidden connection probability (e.g. 0.5)
/// - `conn_prob_hr`      — hidden→readout connection probability (e.g. 0.8)
/// - `w_max_ih`          — max weight for Fisher-scaled input→hidden (e.g. 120)
/// - `w_max_hr`          — uniform weight for hidden→readout (e.g. 32)
/// - `d_norms`           — flat `[n_classes × k_per_class]` Fisher norms, row-major
/// - `seed`              — PRNG seed for reproducibility
pub fn generate_discriminative_3layer_csr(
    k_per_class: usize,
    hidden_per_class: usize,
    readout_per_class: usize,
    n_classes: usize,
    conn_prob_ih: f32,
    conn_prob_hr: f32,
    w_max_ih: i8,
    w_max_hr: i8,
    d_norms: &[f32], // flat [n_classes × k_per_class], row-major
    seed: u64,
) -> SynapseCSRData {
    assert_eq!(
        d_norms.len(),
        k_per_class * n_classes,
        "d_norms must have length n_classes × k_per_class = {}",
        n_classes * k_per_class
    );

    let n_inp = k_per_class * n_classes;
    let n_hid = hidden_per_class * n_classes;
    let n_out = readout_per_class * n_classes;
    let n_neurons = n_inp + n_hid + n_out;

    let mut syn_ptr = vec![0u32; n_neurons + 1];
    let mut targets: Vec<u32> = Vec::new();
    let mut weights: Vec<i8> = Vec::new();

    let mut rng = seed;
    let rng_f = |r: &mut u64| -> f32 {
        *r = r
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*r >> 33) as f32) / (u32::MAX as f32)
    };

    // ── Pass 1: Input → Hidden (Fisher-weighted, same class only) ─────────────
    for c in 0..n_classes {
        let inp_start = c * k_per_class;
        let hid_start = n_inp + c * hidden_per_class;
        let hid_end = hid_start + hidden_per_class;

        for local_i in 0..k_per_class {
            let src = inp_start + local_i;
            let d = d_norms[c * k_per_class + local_i];
            let w = (w_max_ih as f32 * d).clamp(1.0, 127.0) as i8;

            for dst in hid_start..hid_end {
                if rng_f(&mut rng) < conn_prob_ih {
                    targets.push(dst as u32);
                    weights.push(w);
                }
            }
            syn_ptr[src + 1] = targets.len() as u32;
        }
    }

    // ── Pass 2: Hidden → Readout (class-aligned, uniform small weight) ────────
    for c in 0..n_classes {
        let hid_start = n_inp + c * hidden_per_class;
        let out_start = n_inp + n_hid + c * readout_per_class;
        let out_end = out_start + readout_per_class;

        for local_h in 0..hidden_per_class {
            let src = hid_start + local_h;
            for dst in out_start..out_end {
                if rng_f(&mut rng) < conn_prob_hr {
                    targets.push(dst as u32);
                    weights.push(w_max_hr);
                }
            }
            syn_ptr[src + 1] = targets.len() as u32;
        }
    }

    // ── Pass 3: Readout has no outgoing synapses ───────────────────────────────
    for r in 0..n_out {
        syn_ptr[n_inp + n_hid + r + 1] = targets.len() as u32;
    }

    SynapseCSRData {
        syn_ptr,
        targets,
        weights,
    }
}

/// Build a per-neuron threshold array for a **3-layer** discriminative SNN.
///
/// ## Zones
///
/// * **Input**   `[0,           k×C)`:         `input_threshold` (large — prevents spurious fire)
/// * **Hidden**  `[k×C,     k×C+h×C)`:         `t_hidden[c]` per class (from alpha-sweep)
/// * **Readout** `[k×C+h×C, k×C+h×C+r×C)`:    `t_readout` (uniform — calibrated to hidden activity)
///
/// ## Arguments
///
/// - `k_per_class`       — discriminative pixels per class
/// - `hidden_per_class`  — hidden neurons per class
/// - `readout_per_class` — readout neurons per class
/// - `n_classes`         — number of classes
/// - `t_hidden`          — per-class hidden threshold, length = `n_classes`
/// - `t_readout`         — uniform threshold for all readout neurons (e.g. 40)
/// - `input_threshold`   — uniform threshold for all input neurons (e.g. 32000)
pub fn discriminative_3layer_thresholds(
    k_per_class: usize,
    hidden_per_class: usize,
    readout_per_class: usize,
    n_classes: usize,
    t_hidden: &[i16], // per-class, len = n_classes
    t_readout: i16,
    input_threshold: i16,
) -> Vec<i16> {
    assert_eq!(
        t_hidden.len(),
        n_classes,
        "t_hidden must have length n_classes = {}",
        n_classes
    );

    let n_inp = k_per_class * n_classes;
    let n_hid = hidden_per_class * n_classes;
    let n_out = readout_per_class * n_classes;
    let n_total = n_inp + n_hid + n_out;

    let mut thresholds = vec![0i16; n_total];

    // Input zone
    for i in 0..n_inp {
        thresholds[i] = input_threshold;
    }

    // Hidden zone: per-class
    for c in 0..n_classes {
        let hs = n_inp + c * hidden_per_class;
        let he = hs + hidden_per_class;
        for h in hs..he {
            thresholds[h] = t_hidden[c];
        }
    }

    // Readout zone: uniform
    for r in 0..n_out {
        thresholds[n_inp + n_hid + r] = t_readout;
    }

    thresholds
}

/// Build a 3-layer discriminative CSR with **cross-class lateral inhibition** in the readout.
///
/// This is identical to [`generate_discriminative_3layer_csr`] except that Pass 2
/// adds a second sub-pass: each hidden neuron for class `c` also sends **inhibitory**
/// synapses to the readout neurons of every OTHER class `c' ≠ c`.  This creates
/// a winner-take-all dynamic in the readout layer:
///
/// * On a class-c input hidden[c] fires strongly → excites readout[c] AND inhibits readout[c'].
/// * On a wrong-class input hidden[c] fires weakly → excitation < T_readout; inhibition is small.
///
/// V_ss estimates (K=150, H=32, R=8, W_MAX_HR=32, W_INH=-8, CONN_PROB_HR=0.8,
/// CONN_PROB_CROSS=0.15, LEAK=230, MAX_RATE=100):
/// * Readout[c] on correct-class input:  V_ss ≈ 1800  >>  T_READOUT=400  → fires frequently
/// * Readout[c] on wrong-class input:    V_ss ≈ 232   <   T_READOUT=400  → sub-threshold
///
/// Consequence: raw argmax (no T-norm needed) directly gives the correct class in most cases.
/// E-prop can then fine-tune the inhibitory weights to suppress residual confuser activity.
///
/// ## Parameters
///
/// Same as [`generate_discriminative_3layer_csr`] plus:
/// - `conn_prob_cross` — probability of inhibitory hidden[c'] → readout[c] connection (e.g. 0.15)
/// - `w_inh`           — inhibitory weight (must be negative, e.g. `-8`)
pub fn generate_discriminative_3layer_csr_inhibitory(
    k_per_class: usize,
    hidden_per_class: usize,
    readout_per_class: usize,
    n_classes: usize,
    conn_prob_ih: f32,
    conn_prob_hr: f32,
    conn_prob_cross: f32, // cross-class inhibitory connection probability
    w_max_ih: i8,
    w_max_hr: i8,
    w_inh: i8, // inhibitory weight (negative, e.g. -8)
    d_norms: &[f32],
    seed: u64,
) -> SynapseCSRData {
    assert_eq!(
        d_norms.len(),
        k_per_class * n_classes,
        "d_norms must have length n_classes × k_per_class = {}",
        n_classes * k_per_class
    );
    assert!(
        w_inh <= 0,
        "w_inh must be non-positive (inhibitory); got {}",
        w_inh
    );

    let n_inp = k_per_class * n_classes;
    let n_hid = hidden_per_class * n_classes;
    let n_out = readout_per_class * n_classes;
    let n_neurons = n_inp + n_hid + n_out;

    let mut syn_ptr = vec![0u32; n_neurons + 1];
    let mut targets: Vec<u32> = Vec::new();
    let mut weights: Vec<i8> = Vec::new();

    let mut rng = seed;
    let rng_f = |r: &mut u64| -> f32 {
        *r = r
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*r >> 33) as f32) / (u32::MAX as f32)
    };

    // ── Pass 1: Input → Hidden (Fisher-weighted, same class only) ─────────────
    for c in 0..n_classes {
        let inp_start = c * k_per_class;
        let hid_start = n_inp + c * hidden_per_class;
        let hid_end = hid_start + hidden_per_class;

        for local_i in 0..k_per_class {
            let src = inp_start + local_i;
            let d = d_norms[c * k_per_class + local_i];
            let w = (w_max_ih as f32 * d).clamp(1.0, 127.0) as i8;

            for dst in hid_start..hid_end {
                if rng_f(&mut rng) < conn_prob_ih {
                    targets.push(dst as u32);
                    weights.push(w);
                }
            }
            syn_ptr[src + 1] = targets.len() as u32;
        }
    }

    // ── Pass 2: Hidden → Readout ────────────────────────────────────────────
    // Sub-pass 2a: same-class excitatory (hidden[c] → readout[c], w = w_max_hr)
    // Sub-pass 2b: cross-class inhibitory (hidden[c] → readout[c'], c'≠c, w = w_inh)
    // Both are emitted per source neuron so syn_ptr is set correctly after both.
    for c in 0..n_classes {
        let hid_start = n_inp + c * hidden_per_class;
        let out_start = n_inp + n_hid + c * readout_per_class;
        let out_end = out_start + readout_per_class;

        for local_h in 0..hidden_per_class {
            let src = hid_start + local_h;

            // 2a: same-class excitatory
            for dst in out_start..out_end {
                if rng_f(&mut rng) < conn_prob_hr {
                    targets.push(dst as u32);
                    weights.push(w_max_hr);
                }
            }

            // 2b: cross-class inhibitory — hidden[c] inhibits readout[c'] for all c' ≠ c
            for c2 in 0..n_classes {
                if c2 == c {
                    continue;
                }
                let inh_start = n_inp + n_hid + c2 * readout_per_class;
                let inh_end = inh_start + readout_per_class;
                for dst in inh_start..inh_end {
                    if rng_f(&mut rng) < conn_prob_cross {
                        targets.push(dst as u32);
                        weights.push(w_inh);
                    }
                }
            }

            syn_ptr[src + 1] = targets.len() as u32;
        }
    }

    // ── Pass 3: Readout has no outgoing synapses ───────────────────────────────
    for r in 0..n_out {
        syn_ptr[n_inp + n_hid + r + 1] = targets.len() as u32;
    }

    SynapseCSRData {
        syn_ptr,
        targets,
        weights,
    }
}

/// Build a per-neuron threshold array for a discriminative channel SNN.
///
/// ## Layout
///
/// * Input neurons `[0, n_inputs)`:                `input_threshold` (large, prevents spurious LIF fire).
/// * Hidden neurons per class `c`:                 `t_per_class[c]`.
/// * Output neurons `[n_inputs+n_hidden, total)`:  `output_threshold`.
///
/// The returned `Vec<i16>` is ready to pass as the `thresholds` argument to
/// `GpuBatchSNN::new` or `GpuFusedSNN::set_thresholds`.
pub fn discriminative_channel_thresholds(
    k_per_class: usize,
    hidden_per_class: usize,
    n_classes: usize,
    t_per_class: &[i16],
    input_threshold: i16,
    output_threshold: i16,
) -> Vec<i16> {
    assert_eq!(
        t_per_class.len(),
        n_classes,
        "t_per_class.len() must equal n_classes"
    );
    let n_inputs = k_per_class * n_classes;
    let n_hidden = hidden_per_class * n_classes;
    let n_total = n_inputs + n_hidden + n_classes;

    let mut thresholds = vec![0i16; n_total];

    // Input neurons: high threshold (they spike from Poisson, not LIF)
    for i in 0..n_inputs {
        thresholds[i] = input_threshold;
    }

    // Hidden neurons: per-class analytical threshold
    for c in 0..n_classes {
        let h_start = n_inputs + c * hidden_per_class;
        let h_end = h_start + hidden_per_class;
        for h in h_start..h_end {
            thresholds[h] = t_per_class[c];
        }
    }

    // Output neurons: output threshold
    for o in 0..n_classes {
        thresholds[n_inputs + n_hidden + o] = output_threshold;
    }

    thresholds
}

/// Generate a 3-layer discriminative CSR with **within-class hidden recurrence**.
///
/// Same as `generate_discriminative_3layer_csr` but adds hidden→hidden synapses
/// within each class. No self-loops (neuron cannot connect to itself).
///
/// ## HH connectivity
///
/// For each hidden neuron `src` in class `c`, connects to each OTHER hidden neuron
/// `dst` in class `c` with probability `conn_prob_hh`. Weight is `w_hh_exc` with
/// probability `hh_exc_ratio`, else `w_hh_inh`. This creates mixed excitatory/inhibitory
/// recurrence for stability.
///
/// ## CSR order
///
/// Pass 1: Input → Hidden (Fisher-weighted)  — rows 0..n_inp
/// Pass 2: Hidden → Hidden + Hidden → Readout — rows n_inp..n_inp+n_hid
/// Pass 3: Readout (no outgoing)              — rows n_inp+n_hid..
///
/// The IH synapse boundary (`syn_ptr[n_inp]`) is identical to the non-recurrent variant.
pub fn generate_discriminative_3layer_csr_recurrent(
    k_per_class: usize,
    hidden_per_class: usize,
    readout_per_class: usize,
    n_classes: usize,
    conn_prob_ih: f32,
    conn_prob_hr: f32,
    conn_prob_hh: f32,
    w_max_ih: i8,
    w_max_hr: i8,
    w_hh_exc: i8,
    w_hh_inh: i8,
    hh_exc_ratio: f32,
    d_norms: &[f32],
    seed: u64,
) -> SynapseCSRData {
    assert_eq!(
        d_norms.len(),
        k_per_class * n_classes,
        "d_norms must have length n_classes × k_per_class = {}",
        n_classes * k_per_class
    );

    let n_inp = k_per_class * n_classes;
    let n_hid = hidden_per_class * n_classes;
    let n_out = readout_per_class * n_classes;
    let n_neurons = n_inp + n_hid + n_out;

    let mut syn_ptr = vec![0u32; n_neurons + 1];
    let mut targets: Vec<u32> = Vec::new();
    let mut weights: Vec<i8> = Vec::new();

    let mut rng = seed;
    // >> 33 gives [0, ~0.5) — matches other CSR generators, keeps IH boundary identical
    let rng_f = |r: &mut u64| -> f32 {
        *r = r
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*r >> 33) as f32) / (u32::MAX as f32)
    };
    // >> 32 gives [0, ~1.0) — full unit range for exc/inh ratio test
    let rng_unit = |r: &mut u64| -> f32 {
        *r = r
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*r >> 32) as f32) / (u32::MAX as f32)
    };

    // ── Pass 1: Input → Hidden (Fisher-weighted, same class only) ─────────────
    for c in 0..n_classes {
        let inp_start = c * k_per_class;
        let hid_start = n_inp + c * hidden_per_class;
        let hid_end = hid_start + hidden_per_class;

        for local_i in 0..k_per_class {
            let src = inp_start + local_i;
            let d = d_norms[c * k_per_class + local_i];
            let w = (w_max_ih as f32 * d).clamp(1.0, 127.0) as i8;

            for dst in hid_start..hid_end {
                if rng_f(&mut rng) < conn_prob_ih {
                    targets.push(dst as u32);
                    weights.push(w);
                }
            }
            syn_ptr[src + 1] = targets.len() as u32;
        }
    }

    // ── Pass 2: Hidden → Hidden (within-class, mixed exc/inh) + Hidden → Readout
    for c in 0..n_classes {
        let hid_start = n_inp + c * hidden_per_class;
        let hid_end = hid_start + hidden_per_class;
        let out_start = n_inp + n_hid + c * readout_per_class;
        let out_end = out_start + readout_per_class;

        for local_h in 0..hidden_per_class {
            let src = hid_start + local_h;

            // (a) HH synapses: within-class, no self-loops
            for dst in hid_start..hid_end {
                if dst == src {
                    continue;
                } // no self-loop
                if rng_f(&mut rng) < conn_prob_hh {
                    let w = if rng_unit(&mut rng) < hh_exc_ratio {
                        w_hh_exc
                    } else {
                        w_hh_inh
                    };
                    targets.push(dst as u32);
                    weights.push(w);
                }
            }

            // (b) HR synapses: class-aligned readout (unchanged)
            for dst in out_start..out_end {
                if rng_f(&mut rng) < conn_prob_hr {
                    targets.push(dst as u32);
                    weights.push(w_max_hr);
                }
            }

            syn_ptr[src + 1] = targets.len() as u32;
        }
    }

    // ── Pass 3: Readout has no outgoing synapses ───────────────────────────────
    for r in 0..n_out {
        syn_ptr[n_inp + n_hid + r + 1] = targets.len() as u32;
    }

    SynapseCSRData {
        syn_ptr,
        targets,
        weights,
    }
}

#[cfg(test)]
mod discrim_tests {
    use super::*;

    /// Channel isolation: input in class c should only connect to hidden in class c.
    #[test]
    fn test_discrim_csr_channel_isolation() {
        let k = 5;
        let h = 10;
        let c = 3;
        let csr = generate_discriminative_channel_csr(k, h, c, 1.0, 1.0, 80, 120, 50, 80, 42);

        let n_inputs = k * c;
        let hidden_start = n_inputs;

        // For each input neuron i, all its targets must be in channel (i / k)
        for src in 0..n_inputs {
            let expected_channel = src / k;
            let ch_hid_start = (hidden_start + expected_channel * h) as u32;
            let ch_hid_end = (hidden_start + (expected_channel + 1) * h) as u32;

            let s = csr.syn_ptr[src] as usize;
            let e = csr.syn_ptr[src + 1] as usize;
            for &tgt in &csr.targets[s..e] {
                assert!(
                    tgt >= ch_hid_start && tgt < ch_hid_end,
                    "input {} (channel {}) connected to hidden {} outside channel [{}, {})",
                    src,
                    expected_channel,
                    tgt,
                    ch_hid_start,
                    ch_hid_end
                );
            }
        }
    }

    /// With connection_prob=1.0, every input connects to every same-channel hidden neuron.
    #[test]
    fn test_discrim_csr_full_density() {
        let k = 4;
        let h = 8;
        let c = 2;
        let csr = generate_discriminative_channel_csr(k, h, c, 1.0, 1.0, 80, 120, 50, 80, 0);

        let n_inputs = k * c;

        // Each input neuron should have exactly h connections
        for src in 0..n_inputs {
            let degree = (csr.syn_ptr[src + 1] - csr.syn_ptr[src]) as usize;
            assert_eq!(
                degree, h,
                "input neuron {} should have {} connections at prob=1.0, got {}",
                src, h, degree
            );
        }
    }

    /// Threshold array shape and values are correct.
    #[test]
    fn test_discriminative_thresholds_layout() {
        let k = 5;
        let h = 10;
        let c = 3;
        let t_per_class = vec![200i16, 250i16, 180i16];
        let thresholds = discriminative_channel_thresholds(k, h, c, &t_per_class, 32767, 100);

        let n_inputs = k * c;
        let n_hidden = h * c;
        let n_total = n_inputs + n_hidden + c;
        assert_eq!(thresholds.len(), n_total);

        // Input thresholds
        assert!(thresholds[0..n_inputs].iter().all(|&t| t == 32767));

        // Per-class hidden thresholds
        for cls in 0..c {
            let hs = n_inputs + cls * h;
            let he = hs + h;
            assert!(
                thresholds[hs..he].iter().all(|&t| t == t_per_class[cls]),
                "class {} hidden threshold mismatch",
                cls
            );
        }

        // Output thresholds
        assert!(thresholds[n_inputs + n_hidden..].iter().all(|&t| t == 100));
    }

    // ── Milestone 6: 3-layer tests ────────────────────────────────────────────

    /// syn_ptr has length n_inp + n_hid + n_out + 1.
    #[test]
    fn test_3layer_csr_neuron_count() {
        let k = 5;
        let h = 4;
        let r = 2;
        let c = 3;
        let d_norms: Vec<f32> = (0..k * c)
            .map(|i| (i as f32 + 1.0) / (k * c) as f32)
            .collect();
        let csr = generate_discriminative_3layer_csr(k, h, r, c, 1.0, 1.0, 120, 32, &d_norms, 42);
        let expected_len = (k + h + r) * c + 1;
        assert_eq!(
            csr.syn_ptr.len(),
            expected_len,
            "syn_ptr len: expected {} (k+h+r)*c+1, got {}",
            expected_len,
            csr.syn_ptr.len()
        );
    }

    /// No hidden neuron in class c connects to readout neurons of class c' ≠ c.
    #[test]
    fn test_3layer_csr_no_cross_class_hr() {
        let k = 4;
        let h = 6;
        let r = 3;
        let c = 4;
        let d_norms: Vec<f32> = vec![1.0; k * c];
        let csr = generate_discriminative_3layer_csr(k, h, r, c, 0.5, 1.0, 120, 32, &d_norms, 99);
        let n_inp = k * c;
        let n_hid = h * c;
        // For each hidden neuron in class ch, all its targets must be readout neurons of class ch
        for ch in 0..c {
            let hid_start = n_inp + ch * h;
            let out_start = (n_inp + n_hid + ch * r) as u32;
            let out_end = (n_inp + n_hid + (ch + 1) * r) as u32;
            for local_h in 0..h {
                let src = hid_start + local_h;
                let s = csr.syn_ptr[src] as usize;
                let e = csr.syn_ptr[src + 1] as usize;
                for &tgt in &csr.targets[s..e] {
                    assert!(
                        tgt >= out_start && tgt < out_end,
                        "hidden {} (class {}) connected to wrong readout {} (expected [{}, {}))",
                        src,
                        ch,
                        tgt,
                        out_start,
                        out_end
                    );
                }
            }
        }
    }

    /// IH weights are in [1, 127]; HR weights == w_max_hr.
    #[test]
    fn test_3layer_csr_weight_ranges() {
        let k = 5;
        let h = 4;
        let r = 2;
        let c = 3;
        let d_norms: Vec<f32> = (0..k * c)
            .map(|i| (i as f32) / (k * c - 1) as f32)
            .collect();
        let w_max_ih: i8 = 120;
        let w_max_hr: i8 = 32;
        let csr = generate_discriminative_3layer_csr(
            k, h, r, c, 1.0, 1.0, w_max_ih, w_max_hr, &d_norms, 7,
        );
        let n_inp = k * c;
        let n_hid = h * c;
        // Input→Hidden weights: [1, 127]
        for src in 0..n_inp {
            let s = csr.syn_ptr[src] as usize;
            let e = csr.syn_ptr[src + 1] as usize;
            for &w in &csr.weights[s..e] {
                assert!(
                    w >= 1,
                    "IH weight below minimum (1): w={} for src={}",
                    w,
                    src
                );
            }
        }
        // Hidden→Readout weights: exactly w_max_hr
        for src in n_inp..n_inp + n_hid {
            let s = csr.syn_ptr[src] as usize;
            let e = csr.syn_ptr[src + 1] as usize;
            for &w in &csr.weights[s..e] {
                assert_eq!(
                    w, w_max_hr,
                    "HR weight should be w_max_hr={}, got {} for src={}",
                    w_max_hr, w, src
                );
            }
        }
    }

    /// Three threshold zones are correctly assigned.
    #[test]
    fn test_3layer_thresholds_zones() {
        let k = 5;
        let h = 4;
        let r = 2;
        let c = 3;
        let t_hidden = vec![100i16, 200i16, 300i16];
        let t_readout = 40i16;
        let t_input = 32000i16;
        let thresholds =
            discriminative_3layer_thresholds(k, h, r, c, &t_hidden, t_readout, t_input);
        let n_inp = k * c;
        let n_hid = h * c;
        let n_out = r * c;
        assert_eq!(thresholds.len(), n_inp + n_hid + n_out);
        // Input zone
        assert!(
            thresholds[..n_inp].iter().all(|&t| t == t_input),
            "input zone mismatch"
        );
        // Hidden zone per class
        for cls in 0..c {
            let hs = n_inp + cls * h;
            let he = hs + h;
            assert!(
                thresholds[hs..he].iter().all(|&t| t == t_hidden[cls]),
                "hidden zone class {} mismatch",
                cls
            );
        }
        // Readout zone
        assert!(
            thresholds[n_inp + n_hid..].iter().all(|&t| t == t_readout),
            "readout zone mismatch"
        );
    }

    // =========================================================================
    // Tests for generate_discriminative_3layer_csr_recurrent
    // =========================================================================

    fn make_recurrent_csr() -> (SynapseCSRData, usize, usize, usize, usize, usize) {
        let k = 10;
        let h = 8;
        let r = 4;
        let c = 3;
        let d_norms = vec![0.5f32; k * c];
        let csr = generate_discriminative_3layer_csr_recurrent(
            k, h, r, c, 0.5, // conn_prob_ih
            0.5, // conn_prob_hr
            0.4, // conn_prob_hh
            64,  // w_max_ih
            32,  // w_max_hr
            12,  // w_hh_exc
            -8,  // w_hh_inh
            0.8, // hh_exc_ratio
            &d_norms, 42,
        );
        let n_inp = k * c;
        let n_hid = h * c;
        let n_out = r * c;
        (csr, k, h, r, c, n_inp)
    }

    /// No HH self-loops: no synapse where src == dst.
    #[test]
    fn test_recurrent_csr_no_self_loops() {
        let (csr, k, h, _r, c, n_inp) = make_recurrent_csr();
        let n_hid = h * c;
        for src_idx in n_inp..(n_inp + n_hid) {
            let start = csr.syn_ptr[src_idx] as usize;
            let end = csr.syn_ptr[src_idx + 1] as usize;
            for j in start..end {
                assert_ne!(
                    csr.targets[j] as usize, src_idx,
                    "self-loop detected: neuron {src_idx} -> {src_idx}"
                );
            }
        }
    }

    /// HH synapses stay within class: hidden neuron of class c only targets
    /// other hidden neurons of class c (not readout — those are HR).
    #[test]
    fn test_recurrent_csr_hh_within_class() {
        let (csr, k, h, _r, c, n_inp) = make_recurrent_csr();
        let n_hid = h * c;
        for cls in 0..c {
            let hid_start = n_inp + cls * h;
            let hid_end = hid_start + h;
            for src_idx in hid_start..hid_end {
                let start = csr.syn_ptr[src_idx] as usize;
                let end = csr.syn_ptr[src_idx + 1] as usize;
                for j in start..end {
                    let tgt = csr.targets[j] as usize;
                    // Target is either a hidden neuron in the SAME class or a readout neuron
                    let is_same_class_hid = tgt >= hid_start && tgt < hid_end;
                    let is_readout = tgt >= n_inp + n_hid;
                    assert!(
                        is_same_class_hid || is_readout,
                        "cross-class HH: src={src_idx} (class {cls}) -> tgt={tgt}"
                    );
                }
            }
        }
    }

    /// IH boundary (syn_ptr[n_inp]) matches non-recurrent variant.
    #[test]
    fn test_recurrent_csr_ih_boundary_unchanged() {
        let k = 10;
        let h = 8;
        let r = 4;
        let c = 3;
        let d_norms = vec![0.5f32; k * c];
        let n_inp = k * c;

        let baseline =
            generate_discriminative_3layer_csr(k, h, r, c, 0.5, 0.5, 64, 32, &d_norms, 42);
        let recurrent = generate_discriminative_3layer_csr_recurrent(
            k, h, r, c, 0.5, 0.5, 0.4, 64, 32, 12, -8, 0.8, &d_norms, 42,
        );

        assert_eq!(
            baseline.syn_ptr[n_inp], recurrent.syn_ptr[n_inp],
            "IH boundary should be identical: baseline={} recurrent={}",
            baseline.syn_ptr[n_inp], recurrent.syn_ptr[n_inp]
        );
    }

    /// HH sign mix roughly matches hh_exc_ratio (±0.10 tolerance for stochastic sampling).
    #[test]
    fn test_recurrent_csr_hh_sign_mix() {
        let (csr, k, h, _r, c, n_inp) = make_recurrent_csr();
        let n_hid = h * c;
        let n_hid_end = n_inp + n_hid;
        let mut pos_count = 0usize;
        let mut neg_count = 0usize;

        for src_idx in n_inp..n_hid_end {
            let start = csr.syn_ptr[src_idx] as usize;
            let end = csr.syn_ptr[src_idx + 1] as usize;
            for j in start..end {
                let tgt = csr.targets[j] as usize;
                // Only count HH synapses (target is hidden, not readout)
                if tgt >= n_inp && tgt < n_hid_end {
                    if csr.weights[j] > 0 {
                        pos_count += 1;
                    } else {
                        neg_count += 1;
                    }
                }
            }
        }

        let total_hh = pos_count + neg_count;
        assert!(total_hh > 0, "no HH synapses found");
        let exc_ratio = pos_count as f32 / total_hh as f32;
        assert!(
            (exc_ratio - 0.8).abs() < 0.10,
            "exc_ratio {exc_ratio:.3} deviates >0.10 from target 0.8 (pos={pos_count}, neg={neg_count})"
        );
    }
}
