//! Milestone 3: GPU-Batched SNN for RTX 5090
//!
//! `GpuBatchSNN` processes B independent samples simultaneously:
//! - **Shared weights**: One CSR graph, same for all B instances.
//! - **Per-instance state**: `[B × N]` for v_mem, refract, spiked, currents.
//! - **Per-instance inputs**: `[B × n_inputs]` Poisson rates.
//! - **Per-instance outputs**: `[B × n_outputs]` spike counts.
//!
//! Design goal: 256 samples × 3,354 neurons = 859K parallel LIF updates per tick
//! on RTX 5090 (21,760 CUDA cores). Target ≥10,000 MNIST samples/sec.
//!
//! ## Usage
//!
//! ```ignore
//! let mut snn = GpuBatchSNN::new(
//!     &rt, n_neurons, n_inputs, n_outputs,
//!     threshold_per_neuron, leak_per_neuron, &synapses, batch_size,
//! )?;
//!
//! // Per batch:
//! snn.upload_input_batch(&rt, &rates)?;   // rates: &[Vec<u8>] len=batch_size
//! snn.reset_state(&rt)?;
//! snn.reset_output_counts(&rt)?;
//! snn.tick_many_with_counting(&rt, 100)?;
//! rt.synchronize()?;
//! let counts = snn.get_output_counts(&rt)?; // Vec<Vec<u32>> [batch][class]
//! ```

use crate::cuda::{CudaError, CudaResult, CudaRuntime};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

use super::gpu_fused::SynapseCSRData;

// ─────────────────────────────────────────────────────────────────────────────
// CUDA kernels
// ─────────────────────────────────────────────────────────────────────────────

const BATCH_SNN_CUDA: &str = r#"
extern "C" {

// =============================================================================
// KERNEL 1: Batched LIF Dynamics + Input Generation (fused)
// =============================================================================
// Thread layout: tid = batch_id * n_neurons + neuron_id
// Total threads: batch_size * n_neurons
//
// Input neurons (neuron_id < n_inputs):  Poisson spike generation from rates.
// All neurons:                           leak + integrate + fire LIF dynamics.
//
// RNG is seeded per (batch_id, neuron_id, tick) to prevent cross-correlation.
__global__ void snn_batch_lif_step(
    short*               v_mem,         // [B * N] Q8.8 membrane potentials
    const short*         threshold,     // [N]    per-neuron thresholds (shared)
    const unsigned char* leak,          // [N]    per-neuron leak       (shared)
    unsigned char*       refract,       // [B * N] refractory counters
    unsigned char*       spiked,        // [B * N] spike flags (output)
    int*                 currents,      // [B * N] accumulated currents (cleared here)
    const unsigned char* input_rates,   // [B * n_inputs] per-sample Poisson rates
    unsigned int         n_neurons,
    unsigned int         n_inputs,
    unsigned int         batch_size,
    unsigned int         seed
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons * batch_size) return;

    unsigned int batch_id  = tid / n_neurons;
    unsigned int neuron_id = tid % n_neurons;

    // ── Input generation (first n_inputs neurons, Poisson) ─────────────────
    if (neuron_id < n_inputs && refract[tid] == 0) {
        // XOR batch/neuron into seed to decorrelate across samples and neurons
        unsigned int rng = seed
            ^ (batch_id  * 2654435761u)
            ^ (neuron_id * 1234567891u);
        rng = rng * 1664525u + 1013904223u;
        unsigned char rand_val = (unsigned char)(rng >> 24);

        unsigned int rate_idx = batch_id * n_inputs + neuron_id;
        if (rand_val < input_rates[rate_idx]) {
            spiked[tid]   = 1;
            v_mem[tid]    = -128;   // V_RESET
            refract[tid]  = 2;
            currents[tid] = 0;
            return;
        }
    }

    // ── Refractory period ──────────────────────────────────────────────────
    if (refract[tid] > 0) {
        refract[tid]--;
        spiked[tid]   = 0;
        currents[tid] = 0;
        return;
    }

    // ── LIF dynamics ──────────────────────────────────────────────────────
    // Threshold and leak are neuron-specific but shared across all batch instances.
    int v = ((int)v_mem[tid] * (int)leak[neuron_id]) >> 8;
    v += currents[tid];

    if (v >  32767) v =  32767;
    if (v < -32768) v = -32768;

    if (v >= (int)threshold[neuron_id]) {
        v_mem[tid]   = -128;   // V_RESET
        refract[tid] = 2;
        spiked[tid]  = 1;
    } else {
        v_mem[tid]   = (short)v;
        spiked[tid]  = 0;
    }

    currents[tid] = 0;   // clear for next tick
}

// =============================================================================
// KERNEL 2: Batched Current Accumulation (CSR SpMV, shared weights)
// =============================================================================
// Thread layout: tid = batch_id * n_neurons + src
// Shared CSR (syn_ptr, targets, weights) is identical for all B instances.
// Each batch instance writes to its own current slice: base = batch_id * n_neurons.
// No cross-batch atomicAdd contention — each batch uses a disjoint target range.
__global__ void snn_batch_accumulate(
    const unsigned int*  syn_ptr,    // [N+1] CSR row pointers (shared)
    const unsigned int*  targets,    // [nnz] target indices in [0, N)  (shared)
    const signed char*   weights,    // [nnz] Q1.7 weights              (shared)
    const unsigned char* spiked,     // [B * N] spike flags
    int*                 currents,   // [B * N] current accumulators
    unsigned int         n_neurons,
    unsigned int         batch_size
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons * batch_size) return;

    if (spiked[tid] == 0) return;

    unsigned int batch_id = tid / n_neurons;
    unsigned int src      = tid % n_neurons;
    unsigned int base     = batch_id * n_neurons;   // current-slice offset

    unsigned int start = syn_ptr[src];
    unsigned int end   = syn_ptr[src + 1];

    for (unsigned int j = start; j < end; j++) {
        unsigned int tgt  = targets[j];
        int          curr = (int)weights[j] * 2;     // scale to current units
        atomicAdd(&currents[base + tgt], curr);
    }
}

// =============================================================================
// KERNEL 3: Batched Output Spike Counting
// =============================================================================
// Thread layout: tid = batch_id * n_outputs + output_id
// Accumulates into output_counts[batch_id * n_outputs + output_id].
// Call once per tick (from tick_many_with_counting) or at end of episode.
__global__ void snn_batch_count_outputs(
    const unsigned char* spiked,         // [B * N]
    unsigned int*        output_counts,  // [B * n_outputs]
    unsigned int         output_start,   // first output neuron index
    unsigned int         n_outputs,
    unsigned int         n_neurons,
    unsigned int         batch_size
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_outputs * batch_size) return;

    unsigned int batch_id = tid / n_outputs;
    unsigned int out_id   = tid % n_outputs;

    unsigned int state_idx = batch_id * n_neurons + output_start + out_id;
    if (spiked[state_idx]) {
        atomicAdd(&output_counts[batch_id * n_outputs + out_id], 1);
    }
}

// =============================================================================
// KERNEL 4: Batched State Reset
// =============================================================================
// Zero v_mem, refract, spiked, currents for ALL B × N entries.
// Called between samples to start each inference from rest.
__global__ void snn_batch_reset_state(
    short*         v_mem,
    unsigned char* refract,
    unsigned char* spiked,
    int*           currents,
    unsigned int   total    // batch_size * n_neurons
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    v_mem[i]    = 0;
    refract[i]  = 0;
    spiked[i]   = 0;
    currents[i] = 0;
}

// =============================================================================
// KERNEL 5: Reset Output Counts
// =============================================================================
__global__ void snn_batch_reset_outputs(
    unsigned int* output_counts,
    unsigned int  total    // batch_size * n_outputs
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    output_counts[i] = 0;
}

// =============================================================================
// KERNEL 6: Count spikes from ALL neurons (Phase 4 STDP)
// =============================================================================
// Thread layout: tid = batch_id * n_neurons + neuron_id
// Accumulates spiked[tid] into all_counts[tid] (same flat layout).
__global__ void snn_batch_count_all(
    const unsigned char* spiked,      // [B * N]
    unsigned int*        all_counts,  // [B * N] cumulative
    unsigned int         n_neurons,
    unsigned int         batch_size
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons * batch_size) return;
    if (spiked[tid]) atomicAdd(&all_counts[tid], 1u);
}

// =============================================================================
// KERNEL 7: Reset all-neuron spike count buffer (Phase 4 STDP)
// =============================================================================
__global__ void snn_batch_reset_all_counts(
    unsigned int* all_counts,
    unsigned int  total    // batch_size * n_neurons
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    all_counts[i] = 0;
}

// =============================================================================
// KERNEL 8: Batch reward-modulated Hebbian weight update (Phase 4 STDP)
// =============================================================================
// One thread per synapse.  Reads pre- and post-synaptic spike counts accumulated
// over a forward pass (all_counts[B × N]), weights by per-sample reward, and
// applies the update to the shared weight array.
//
// delta_w_j = learning_rate * inv_norm * (1/B) * Σ_b counts[b][src] * counts[b][dst] * reward[b]
// inv_norm   = 1.0 / (n_ticks * n_ticks)  — normalises for episode length
__global__ void snn_batch_apply_hebbian(
    const unsigned int* src_of_syn,  // [nnz] source neuron for each synapse
    const unsigned int* targets,     // [nnz] target neuron index
    signed char*        weights,     // [nnz] Q1.7 weights — updated in place
    const unsigned int* all_counts,  // [B * N]
    const float*        rewards,     // [B]
    unsigned int        n_synapses,
    unsigned int        n_neurons,
    unsigned int        batch_size,
    float               learning_rate,
    float               inv_norm     // 1.0 / (n_ticks * n_ticks * batch_size)
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_synapses) return;

    unsigned int src = src_of_syn[j];
    unsigned int dst = targets[j];

    float delta = 0.0f;
    for (unsigned int b = 0; b < batch_size; b++) {
        float pre  = (float)all_counts[b * n_neurons + src];
        float post = (float)all_counts[b * n_neurons + dst];
        delta += pre * post * rewards[b];
    }
    delta *= learning_rate * inv_norm;

    int new_w = (int)weights[j] + (int)(delta * 127.0f);
    if (new_w >  127) new_w =  127;
    if (new_w < -128) new_w = -128;
    weights[j] = (signed char)new_w;
}

// =============================================================================
// KERNEL 9: ALIF LIF step — adds adaptive threshold to kernel 1
// =============================================================================
// adapt_state[tid] stores a_i in Q8.8 (256 units = 1 threshold unit).
// theta_eff = threshold[neuron_id] + (int)(beta_adapt * adapt_state[tid])
// On fire: adapt_state decays then gets +256 (= +1 threshold unit).
// Each tick: adapt_state *= alpha_adapt.
__global__ void snn_batch_lif_step_alif(
    short*               v_mem,
    const short*         threshold,
    const unsigned char* leak,
    unsigned char*       refract,
    unsigned char*       spiked,
    int*                 currents,
    const unsigned char* input_rates,
    short*               adapt_state,    // [B * N] adaptation accumulator (Q8.8)
    float                alpha_adapt,    // decay coefficient, e.g. 0.967 = exp(-1/30)
    float                beta_adapt,     // threshold scale, e.g. 0.1
    unsigned int         n_neurons,
    unsigned int         n_inputs,
    unsigned int         batch_size,
    unsigned int         seed
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_neurons * batch_size) return;

    unsigned int batch_id  = tid / n_neurons;
    unsigned int neuron_id = tid % n_neurons;

    // ── Input generation (identical to kernel 1) ──────────────────────────
    if (neuron_id < n_inputs && refract[tid] == 0) {
        unsigned int rng = seed
            ^ (batch_id  * 2654435761u)
            ^ (neuron_id * 1234567891u);
        rng = rng * 1664525u + 1013904223u;
        unsigned char rand_val = (unsigned char)(rng >> 24);
        unsigned int rate_idx = batch_id * n_inputs + neuron_id;
        if (rand_val < input_rates[rate_idx]) {
            adapt_state[tid] = (short)((float)adapt_state[tid] * alpha_adapt);
            spiked[tid] = 1; v_mem[tid] = -128; refract[tid] = 2; currents[tid] = 0;
            return;
        }
    }

    if (refract[tid] > 0) {
        adapt_state[tid] = (short)((float)adapt_state[tid] * alpha_adapt);
        refract[tid]--; spiked[tid] = 0; currents[tid] = 0; return;
    }

    // ── LIF dynamics with adaptive threshold ──────────────────────────────
    int v = ((int)v_mem[tid] * (int)leak[neuron_id]) >> 8;
    v += currents[tid];
    if (v >  32767) v =  32767;
    if (v < -32768) v = -32768;

    // Effective threshold: base + beta * adaptation (adapt_state in Q8.8)
    int theta_eff = (int)threshold[neuron_id]
                  + (int)((float)adapt_state[tid] * beta_adapt);

    if (v >= theta_eff) {
        // Decay then add +1 threshold unit (256 in Q8.8) for the spike
        adapt_state[tid] = (short)((float)adapt_state[tid] * alpha_adapt + 256.0f);
        v_mem[tid] = -128; refract[tid] = 2; spiked[tid] = 1;
    } else {
        adapt_state[tid] = (short)((float)adapt_state[tid] * alpha_adapt);
        v_mem[tid] = (short)v; spiked[tid] = 0;
    }
    currents[tid] = 0;
}

// =============================================================================
// KERNEL 10: Update pre-synaptic low-pass trace (Phase 5 E-prop)
// =============================================================================
// pre_trace[tid] = alpha_pre * pre_trace[tid] + spiked[tid]
// Decaying exponential filter of spike train — used as pre-synaptic signal
// in eligibility trace computation.
__global__ void snn_update_pre_trace(
    const unsigned char* spiked,    // [B * N]
    float*               pre_trace, // [B * N]
    float                alpha_pre,
    unsigned int         total      // B * N
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    pre_trace[tid] = alpha_pre * pre_trace[tid] + (float)spiked[tid];
}

// =============================================================================
// KERNEL 11: Update per-synapse eligibility traces (Phase 5 E-prop)
// =============================================================================
// One thread per (batch × synapse) pair.
// Surrogate gradient: fast sigmoid  psi = 1 / (1 + gamma * |v/theta - 1|)^2
// Eligibility update: e[b][j] = alpha_elig * e[b][j] + psi_post * pre_trace_src
__global__ void snn_update_eligibility(
    const float*         pre_trace,    // [B * N]
    const short*         v_mem,        // [B * N]
    const short*         threshold,    // [N] (shared)
    const unsigned int*  src_of_syn,   // [nnz]
    const unsigned int*  targets,      // [nnz]
    float*               eligibility,  // [B * nnz]
    float                alpha_elig,
    float                gamma_surr,   // surrogate gradient sharpness
    unsigned int         n_neurons,
    unsigned int         n_synapses,
    unsigned int         batch_size
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * n_synapses) return;

    unsigned int b   = tid / n_synapses;
    unsigned int j   = tid % n_synapses;
    unsigned int src = src_of_syn[j];
    unsigned int dst = targets[j];

    // Fast-sigmoid surrogate gradient at post-synaptic neuron
    float v    = (float)v_mem[b * n_neurons + dst];
    float thr  = (float)threshold[dst];
    float ratio = (thr != 0.0f) ? (v / thr) : 0.0f;
    float denom = 1.0f + gamma_surr * fabsf(ratio - 1.0f);
    float psi   = 1.0f / (denom * denom);

    float pre = pre_trace[b * n_neurons + src];
    eligibility[tid] = alpha_elig * eligibility[tid] + psi * pre;
}

// =============================================================================
// KERNEL 12: Apply e-prop weight update (Phase 5 E-prop)
// =============================================================================
// One thread per synapse. Loops over B, accumulates eligibility * learning_signal.
// delta_w_j = lr * (1/B) * Σ_b eligibility[b][j] * learning_signal[b]
__global__ void snn_apply_eprop(
    const float*    eligibility,       // [B * nnz]
    const float*    learning_signals,  // [B]
    signed char*    weights,           // [nnz] Q1.7 — updated in place
    unsigned int    n_synapses,
    unsigned int    batch_size,
    float           learning_rate
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_synapses) return;

    float delta = 0.0f;
    for (unsigned int b = 0; b < batch_size; b++) {
        delta += eligibility[b * n_synapses + j] * learning_signals[b];
    }
    delta *= learning_rate / (float)batch_size;

    int new_w = (int)weights[j] + (int)(delta * 127.0f);
    if (new_w >  127) new_w =  127;
    if (new_w < -128) new_w = -128;
    weights[j] = (signed char)new_w;
}

// =============================================================================
// KERNEL 13: Initialise float32 weight shadow from i8 weights
// =============================================================================
__global__ void snn_init_weight_shadow(
    const signed char* weights,   // [n_synapses] i8 source
    float*             weights_f32, // [n_synapses] f32 destination
    unsigned int       n_synapses
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_synapses) return;
    weights_f32[j] = (float)weights[j];
}

// =============================================================================
// KERNEL 14: Apply e-prop weight update to float32 shadow (no i8 rounding)
// =============================================================================
// Identical formula to snn_apply_eprop but accumulates in float32.
// Avoids the precision dead-zone where delta * 127 < 1 and always truncates to 0.
// delta_w_j = lr * (1/B) * Σ_b eligibility[b][j] * learning_signal[b]
__global__ void snn_apply_eprop_f32(
    const float*    eligibility,       // [B * nnz]
    const float*    learning_signals,  // [B]
    float*          weights_f32,       // [nnz] — updated in place
    unsigned int    n_synapses,
    unsigned int    batch_size,
    float           learning_rate
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_synapses) return;

    float delta = 0.0f;
    for (unsigned int b = 0; b < batch_size; b++) {
        delta += eligibility[b * n_synapses + j] * learning_signals[b];
    }
    delta *= learning_rate / (float)batch_size;

    float new_w = weights_f32[j] + delta;
    if (new_w >  127.0f) new_w =  127.0f;
    if (new_w < -128.0f) new_w = -128.0f;
    weights_f32[j] = new_w;
}

// =============================================================================
// KERNEL 15: Project float32 weight shadow to i8 (for inference kernels)
// =============================================================================
__global__ void snn_project_weights_f32(
    const float*    weights_f32,  // [n_synapses] source
    signed char*    weights,      // [n_synapses] destination — updated in place
    unsigned int    n_synapses
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_synapses) return;
    float w = weights_f32[j];
    if (w >  127.0f) w =  127.0f;
    if (w < -128.0f) w = -128.0f;
    weights[j] = (signed char)(int)w;
}

// =============================================================================
// KERNEL 16: Apply e-prop update with per-layer learning rates (float32 shadow)
// =============================================================================
// Identical to kernel 14 except the learning rate depends on whether synapse j
// falls in the input→hidden range [0, boundary) or hidden→readout [boundary, n).
// This lets Phase 4 train the readout layer at a higher LR without disturbing the
// Fisher-calibrated input→hidden weights.
__global__ void snn_apply_eprop_f32_layered(
    const float*    eligibility,       // [B * nnz]
    const float*    learning_signals,  // [B]
    float*          weights_f32,       // [nnz] — updated in place
    unsigned int    n_synapses,
    unsigned int    batch_size,
    float           lr_a,              // LR for synapses [0, boundary)
    float           lr_b,              // LR for synapses [boundary, n_synapses)
    unsigned int    boundary           // first HR synapse index
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_synapses) return;

    float lr = (j < boundary) ? lr_a : lr_b;

    float delta = 0.0f;
    for (unsigned int b = 0; b < batch_size; b++) {
        delta += eligibility[b * n_synapses + j] * learning_signals[b];
    }
    delta *= lr / (float)batch_size;

    float new_w = weights_f32[j] + delta;
    if (new_w >  127.0f) new_w =  127.0f;
    if (new_w < -128.0f) new_w = -128.0f;
    weights_f32[j] = new_w;
}

// =============================================================================
// KERNEL 17: Apply e-prop update with per-class learning signals (M8)
// =============================================================================
// Replaces kernel 16's scalar learning_signals[B] with per-class
// learning_signals[B * n_classes]. Each synapse j looks up its class from
// synapse_class[j] and uses the corresponding per-class signal.
// Enables class-specific weight specialization (vs M7P2 uniform drift).
__global__ void snn_apply_eprop_f32_per_class(
    const float*        eligibility,       // [B * nnz]
    const float*        learning_signals,  // [B * n_classes]
    const unsigned int* synapse_class,     // [nnz] class index per synapse
    float*              weights_f32,       // [nnz] — updated in place
    unsigned int        n_synapses,
    unsigned int        batch_size,
    unsigned int        n_classes,
    float               lr_a,             // LR for synapses [0, boundary)
    float               lr_b,             // LR for synapses [boundary, n_synapses)
    unsigned int        boundary           // first HR synapse index
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_synapses) return;

    float lr = (j < boundary) ? lr_a : lr_b;
    unsigned int cls = synapse_class[j];

    float delta = 0.0f;
    for (unsigned int b = 0; b < batch_size; b++) {
        delta += eligibility[b * n_synapses + j]
               * learning_signals[b * n_classes + cls];
    }
    delta *= lr / (float)batch_size;

    float new_w = weights_f32[j] + delta;
    if (new_w >  127.0f) new_w =  127.0f;
    if (new_w < -128.0f) new_w = -128.0f;
    weights_f32[j] = new_w;
}

// =============================================================================
// KERNEL: Block-Sparse Accumulate (Sprint 348)
// =============================================================================
// Thread layout: tid = batch_id * n_block_pairs + pair_id
// Each thread processes one 8x8 block pair for one batch instance.
// Iterates the 64-bit synapse mask, checking source spikes and accumulating
// fixed-weight current into targets via atomicAdd.
__global__ void snn_batch_accumulate_blocksparse(
    const unsigned short*   src_blocks,       // [n_block_pairs] source block indices
    const unsigned short*   dst_blocks,       // [n_block_pairs] dest block indices
    const unsigned long long* synapse_masks,  // [n_block_pairs] 64-bit masks
    const unsigned char*    spiked,           // [B * N] spike flags
    int*                    currents,         // [B * N] current accumulators
    unsigned int            n_neurons,
    unsigned int            n_block_pairs,
    unsigned int            batch_size,
    int                     fixed_weight      // weight * 2 (scaled)
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_block_pairs * batch_size) return;

    unsigned int batch_id = tid / n_block_pairs;
    unsigned int pair_id  = tid % n_block_pairs;

    unsigned int src_base = (unsigned int)src_blocks[pair_id] * 8;
    unsigned int dst_base = (unsigned int)dst_blocks[pair_id] * 8;
    unsigned long long mask = synapse_masks[pair_id];

    unsigned int b_offset = batch_id * n_neurons;

    // Check if any source neuron in this block spiked.
    unsigned char any_spike = 0;
    for (int si = 0; si < 8; si++) {
        if (src_base + si < n_neurons && spiked[b_offset + src_base + si]) {
            any_spike = 1;
            break;
        }
    }
    if (!any_spike) return;

    // Iterate set bits in 64-bit mask.
    while (mask != 0) {
        int bit = __ffsll(mask) - 1;  // find first set bit (0-based)
        mask &= mask - 1;             // clear lowest set bit

        int si = bit / 8;  // source index within block
        int di = bit % 8;  // dest index within block

        unsigned int src_idx = src_base + si;
        unsigned int dst_idx = dst_base + di;

        if (src_idx < n_neurons && dst_idx < n_neurons
            && spiked[b_offset + src_idx]) {
            atomicAdd(&currents[b_offset + dst_idx], fixed_weight);
        }
    }
}

// =============================================================================
// KERNEL: Weighted Block-Sparse Accumulate (Sprint 349)
// =============================================================================
// Like snn_batch_accumulate_blocksparse but with per-synapse weights from a
// packed weight array. weight_offsets[pair_id] gives the start index into
// packed_weights for this block pair's active synapses (ordered by bit position).
__global__ void snn_batch_accumulate_blocksparse_weighted(
    const unsigned short*     src_blocks,
    const unsigned short*     dst_blocks,
    const unsigned long long* synapse_masks,
    const unsigned int*       weight_offsets,    // [n_block_pairs + 1]
    const signed char*        packed_weights,    // [total_active_synapses]
    const unsigned char*      spiked,
    int*                      currents,
    unsigned int              n_neurons,
    unsigned int              n_block_pairs,
    unsigned int              batch_size
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_block_pairs * batch_size) return;

    unsigned int batch_id = tid / n_block_pairs;
    unsigned int pair_id  = tid % n_block_pairs;

    unsigned int src_base = (unsigned int)src_blocks[pair_id] * 8;
    unsigned long long mask = synapse_masks[pair_id];
    unsigned int b_offset = batch_id * n_neurons;

    // Early exit if no source neuron spiked.
    unsigned char any_spike = 0;
    for (int si = 0; si < 8; si++) {
        if (src_base + si < n_neurons && spiked[b_offset + src_base + si]) {
            any_spike = 1;
            break;
        }
    }
    if (!any_spike) return;

    unsigned int dst_base = (unsigned int)dst_blocks[pair_id] * 8;
    unsigned int w_base = weight_offsets[pair_id];
    unsigned long long m = mask;
    unsigned int w_idx = 0;

    while (m != 0) {
        int bit = __ffsll(m) - 1;
        m &= m - 1;

        int si = bit / 8;
        int di = bit % 8;
        unsigned int src_idx = src_base + si;
        unsigned int dst_idx = dst_base + di;

        if (src_idx < n_neurons && dst_idx < n_neurons
            && spiked[b_offset + src_idx]) {
            int curr = (int)packed_weights[w_base + w_idx] * 2;
            atomicAdd(&currents[b_offset + dst_idx], curr);
        }
        w_idx++;
    }
}

// =============================================================================
// KERNEL 18: Sync CSR weights → block-sparse packed weights via remap
// =============================================================================
// After E-prop updates d_weights (CSR order), this kernel copies them into
// d_bs_packed_weights (block-sparse packed order) using a precomputed remap.
// remap[i] = CSR synapse index for packed weight position i.
__global__ void snn_sync_blocksparse_weights(
    const signed char*  csr_weights,     // [n_synapses] CSR i8 weights
    signed char*        packed_weights,  // [n_packed] block-sparse packed weights
    const unsigned int* remap,           // [n_packed] CSR index per packed position
    unsigned int        n_packed
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_packed) return;
    packed_weights[tid] = csr_weights[remap[tid]];
}

} // extern "C"
"#;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel handles
// ─────────────────────────────────────────────────────────────────────────────

struct BatchSNNKernels {
    lif_step_fn: CudaFunction,
    accumulate_fn: CudaFunction,
    count_outputs_fn: CudaFunction,
    reset_state_fn: CudaFunction,
    reset_outputs_fn: CudaFunction,
    // Phase 4: STDP kernels
    count_all_fn: CudaFunction,
    reset_all_counts_fn: CudaFunction,
    apply_hebbian_fn: CudaFunction,
    // Phase 5: ALIF + E-prop kernels
    lif_step_alif_fn: CudaFunction,
    update_pre_trace_fn: CudaFunction,
    update_eligibility_fn: CudaFunction,
    apply_eprop_fn: CudaFunction,
    // Phase 5b: float32 weight shadow kernels
    init_weight_shadow_fn: CudaFunction,
    apply_eprop_f32_fn: CudaFunction,
    project_weights_f32_fn: CudaFunction,
    // Phase 4 (M6): layered learning rate
    apply_eprop_f32_layered_fn: CudaFunction,
    // M8: per-class learning signals
    apply_eprop_f32_per_class_fn: CudaFunction,
    // Sprint 348: Block-sparse accumulate
    accumulate_blocksparse_fn: CudaFunction,
    // Sprint 349: Weighted block-sparse accumulate
    accumulate_blocksparse_weighted_fn: CudaFunction,
    // Sprint 353: Sync CSR weights → block-sparse packed weights
    sync_blocksparse_weights_fn: CudaFunction,
}

fn compile_batch_kernels(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
) -> CudaResult<BatchSNNKernels> {
    use logic_fabric_core::cuda::get_cuda_include_path;

    let cuda_include = get_cuda_include_path();
    let arch = batch_device_arch();

    let opts = CompileOptions {
        arch: Some(arch),
        include_paths: vec![cuda_include],
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(BATCH_SNN_CUDA, opts).map_err(|e| {
        CudaError::KernelCompilationFailed(format!("BatchSNN compile error: {:?}", e))
    })?;

    let module = ctx.load_module(ptx).map_err(|e| {
        CudaError::KernelCompilationFailed(format!("BatchSNN module load error: {:?}", e))
    })?;

    macro_rules! load {
        ($name:expr) => {
            module
                .load_function($name)
                .map_err(|e| CudaError::KernelCompilationFailed(format!("{}: {:?}", $name, e)))?
        };
    }

    Ok(BatchSNNKernels {
        lif_step_fn: load!("snn_batch_lif_step"),
        accumulate_fn: load!("snn_batch_accumulate"),
        count_outputs_fn: load!("snn_batch_count_outputs"),
        reset_state_fn: load!("snn_batch_reset_state"),
        reset_outputs_fn: load!("snn_batch_reset_outputs"),
        count_all_fn: load!("snn_batch_count_all"),
        reset_all_counts_fn: load!("snn_batch_reset_all_counts"),
        apply_hebbian_fn: load!("snn_batch_apply_hebbian"),
        // Phase 5: ALIF + E-prop
        lif_step_alif_fn: load!("snn_batch_lif_step_alif"),
        update_pre_trace_fn: load!("snn_update_pre_trace"),
        update_eligibility_fn: load!("snn_update_eligibility"),
        apply_eprop_fn: load!("snn_apply_eprop"),
        // Phase 5b: float32 weight shadow
        init_weight_shadow_fn: load!("snn_init_weight_shadow"),
        apply_eprop_f32_fn: load!("snn_apply_eprop_f32"),
        project_weights_f32_fn: load!("snn_project_weights_f32"),
        // Phase 4 (M6): layered LR
        apply_eprop_f32_layered_fn: load!("snn_apply_eprop_f32_layered"),
        // M8: per-class signals
        apply_eprop_f32_per_class_fn: load!("snn_apply_eprop_f32_per_class"),
        // Sprint 348: Block-sparse accumulate
        accumulate_blocksparse_fn: load!("snn_batch_accumulate_blocksparse"),
        accumulate_blocksparse_weighted_fn: load!("snn_batch_accumulate_blocksparse_weighted"),
        sync_blocksparse_weights_fn: load!("snn_sync_blocksparse_weights"),
    })
}

fn batch_device_arch() -> &'static str {
    // RTX 5090 (Blackwell SM120). compute_89 (Ada) works via PTX JIT.
    "compute_89"
}

// ─────────────────────────────────────────────────────────────────────────────
// Public struct
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-batched SNN: B independent samples share one weight matrix.
///
/// Memory layout (all arrays are flat row-major):
/// - Neuron state:  `[batch_size × n_neurons]`
/// - Input rates:   `[batch_size × n_inputs]`
/// - Output counts: `[batch_size × n_outputs]`
/// - Weights (CSR): `[n_synapses]` — shared across all B instances.
pub struct GpuBatchSNN {
    // ── Shared network topology (GPU-resident) ─────────────────────────────
    d_threshold: CudaSlice<i16>, // [N] per-neuron firing thresholds
    d_leak: CudaSlice<u8>,       // [N] per-neuron leak factors
    d_syn_ptr: CudaSlice<u32>,   // [N+1] CSR row pointers
    d_targets: CudaSlice<u32>,   // [nnz] CSR column indices
    d_weights: CudaSlice<i8>,    // [nnz] Q1.7 weights

    // ── Sprint 351: Optional block-sparse topology (GPU-resident) ─────────
    d_bs_src_blocks: Option<CudaSlice<u16>>,
    d_bs_dst_blocks: Option<CudaSlice<u16>>,
    d_bs_synapse_masks: Option<CudaSlice<u64>>,
    d_bs_weight_offsets: Option<CudaSlice<u32>>,
    d_bs_packed_weights: Option<CudaSlice<i8>>,
    /// Sprint 353: Maps packed weight index → CSR synapse index for weight sync.
    d_bs_weight_remap: Option<CudaSlice<u32>>,
    /// Sprint 353: Number of packed weights in block-sparse format.
    bs_n_packed_weights: u32,
    bs_n_block_pairs: u32,
    use_blocksparse: bool,

    // ── Per-instance state [B × N] (GPU-resident) ─────────────────────────
    d_v_mem: CudaSlice<i16>,
    d_refract: CudaSlice<u8>,
    d_spiked: CudaSlice<u8>,
    d_currents: CudaSlice<i32>,

    // ── Per-instance I/O ──────────────────────────────────────────────────
    d_input_rates: CudaSlice<u8>,    // [B × n_inputs]
    d_output_counts: CudaSlice<u32>, // [B × n_outputs]

    // ── Kernels ───────────────────────────────────────────────────────────
    kernels: BatchSNNKernels,

    // ── Dimensions ───────────────────────────────────────────────────────
    n_neurons: u32,
    n_inputs: u32,
    n_outputs: u32,
    n_synapses: u32,
    output_start: u32,
    batch_size: u32,

    // ── Tick counter (seeds per-tick RNG) ─────────────────────────────────
    tick: u64,

    // ── Phase 4: STDP support (allocated on demand via enable_stdp()) ──────
    /// Source neuron index for each synapse `j`: inverse of CSR syn_ptr.
    /// Built at construction time (cheap single pass), stored on GPU.
    d_src_of_syn: CudaSlice<u32>,
    /// [B × N] spike count buffer for ALL neurons (including inputs).
    /// `None` until `enable_stdp()` is called.
    d_all_counts: Option<CudaSlice<u32>>,
    stdp_enabled: bool,

    // ── Phase 5: ALIF adaptive threshold (allocated on demand via enable_alif()) ──
    /// [B × N] adaptation accumulator in Q8.8 (256 units = 1 threshold unit).
    /// `None` until `enable_alif()` is called.
    d_adapt_state: Option<CudaSlice<i16>>,
    alif_alpha: f32, // decay coefficient, e.g. 0.967 = exp(-1/30)
    alif_beta: f32,  // threshold scale, e.g. 0.1
    alif_enabled: bool,

    // ── Phase 5: E-prop eligibility traces (allocated via enable_eprop()) ──
    /// [B × N] pre-synaptic low-pass filtered spike train.
    d_pre_trace: Option<CudaSlice<f32>>,
    /// [B × nnz] per-synapse eligibility traces.
    d_eligibility: Option<CudaSlice<f32>>,
    eprop_alpha_pre: f32,  // pre-trace decay, e.g. 0.95
    eprop_alpha_elig: f32, // eligibility decay, e.g. 0.95
    eprop_gamma: f32,      // surrogate gradient sharpness, e.g. 0.3
    eprop_enabled: bool,

    // ── Phase 5b: float32 weight shadow (allocated via init_weight_shadow()) ─
    /// [nnz] float32 weight accumulation buffer — avoids i8 rounding dead-zone.
    /// Projected back to d_weights (i8) for inference via project_weights_f32().
    d_weights_f32: Option<CudaSlice<f32>>,

    // ── M8: per-class synapse mapping (allocated via init_synapse_class_map()) ─
    /// [nnz] u32 mapping each synapse j to its class index.
    d_synapse_class: Option<CudaSlice<u32>>,
}

impl GpuBatchSNN {
    /// Create a batched SNN.
    ///
    /// # Arguments
    /// * `rt`           – CUDA runtime.
    /// * `n_neurons`    – Total neuron count (inputs + hidden + outputs).
    /// * `n_inputs`     – First `n_inputs` neurons are input-driven.
    /// * `n_outputs`    – Last `n_outputs` neurons are output readout.
    /// * `thresholds`   – Per-neuron Q8.8 firing thresholds (`[N]`). Uniform: pass `vec![t; N]`.
    /// * `leaks`        – Per-neuron Q0.8 leak factors (`[N]`). Uniform: pass `vec![l; N]`.
    /// * `synapses`     – CSR connectivity (shared across all batch instances).
    /// * `batch_size`   – Number of samples to process simultaneously.
    pub fn new(
        rt: &CudaRuntime,
        n_neurons: usize,
        n_inputs: usize,
        n_outputs: usize,
        thresholds: &[i16],
        leaks: &[u8],
        synapses: &SynapseCSRData,
        batch_size: usize,
    ) -> CudaResult<Self> {
        assert_eq!(
            thresholds.len(),
            n_neurons,
            "thresholds.len() must equal n_neurons"
        );
        assert_eq!(leaks.len(), n_neurons, "leaks.len() must equal n_neurons");
        assert!(
            n_inputs + n_outputs <= n_neurons,
            "n_inputs + n_outputs must be ≤ n_neurons"
        );
        assert!(batch_size > 0, "batch_size must be > 0");

        let bn = batch_size * n_neurons;
        let output_start = n_neurons - n_outputs;

        // Shared topology
        let d_threshold = rt.upload(thresholds)?;
        let d_leak = rt.upload(leaks)?;
        let d_syn_ptr = rt.upload(&synapses.syn_ptr)?;
        let d_targets = rt.upload(&synapses.targets)?;
        let d_weights = rt.upload(&synapses.weights)?;

        // Per-instance state (zero-initialised)
        let d_v_mem = rt.alloc_zeros::<i16>(bn)?;
        let d_refract = rt.alloc_zeros::<u8>(bn)?;
        let d_spiked = rt.alloc_zeros::<u8>(bn)?;
        let d_currents = rt.alloc_zeros::<i32>(bn)?;

        // Per-instance I/O (zero-initialised)
        let d_input_rates = rt.alloc_zeros::<u8>(batch_size * n_inputs)?;
        let d_output_counts = rt.alloc_zeros::<u32>(batch_size * n_outputs)?;

        // Phase 4: Build src_of_syn[j] = source neuron for synapse j (CSR inverse).
        // Single pass over syn_ptr — cheap but needed on GPU for apply_hebbian.
        let n_synapses = synapses.targets.len();
        let mut src_of_syn = vec![0u32; n_synapses];
        for src in 0..n_neurons {
            let start = synapses.syn_ptr[src] as usize;
            let end = synapses.syn_ptr[src + 1] as usize;
            for j in start..end {
                src_of_syn[j] = src as u32;
            }
        }
        let d_src_of_syn = rt.upload(&src_of_syn)?;

        // Compile kernels
        let kernels = compile_batch_kernels(rt.context())?;

        Ok(Self {
            d_threshold,
            d_leak,
            d_syn_ptr,
            d_targets,
            d_weights,
            // Sprint 351: block-sparse fields (None until enable_blocksparse).
            d_bs_src_blocks: None,
            d_bs_dst_blocks: None,
            d_bs_synapse_masks: None,
            d_bs_weight_offsets: None,
            d_bs_packed_weights: None,
            d_bs_weight_remap: None,
            bs_n_packed_weights: 0,
            bs_n_block_pairs: 0,
            use_blocksparse: false,
            d_v_mem,
            d_refract,
            d_spiked,
            d_currents,
            d_input_rates,
            d_output_counts,
            kernels,
            n_neurons: n_neurons as u32,
            n_inputs: n_inputs as u32,
            n_outputs: n_outputs as u32,
            n_synapses: n_synapses as u32,
            output_start: output_start as u32,
            batch_size: batch_size as u32,
            tick: 0,
            d_src_of_syn,
            d_all_counts: None,
            stdp_enabled: false,
            // Phase 5: ALIF
            d_adapt_state: None,
            alif_alpha: 0.0,
            alif_beta: 0.0,
            alif_enabled: false,
            // Phase 5: E-prop
            d_pre_trace: None,
            d_eligibility: None,
            eprop_alpha_pre: 0.0,
            eprop_alpha_elig: 0.0,
            eprop_gamma: 0.0,
            eprop_enabled: false,
            // Phase 5b: float32 weight shadow
            d_weights_f32: None,
            // M8: per-class synapse mapping
            d_synapse_class: None,
        })
    }

    // ─────────────────────────────────────────────────────────────────────
    // Input / Output
    // ─────────────────────────────────────────────────────────────────────

    /// Upload input spike rates for an entire batch.
    ///
    /// `rates` must be a flat `[batch_size × n_inputs]` slice in row-major
    /// order (sample 0 first, then sample 1, …).
    pub fn upload_input_batch(&mut self, rt: &CudaRuntime, rates: &[u8]) -> CudaResult<()> {
        assert_eq!(
            rates.len(),
            self.batch_size as usize * self.n_inputs as usize,
            "rates.len() must equal batch_size * n_inputs"
        );
        self.d_input_rates = rt.upload(rates)?;
        Ok(())
    }

    /// Download output spike counts: returns `Vec<Vec<u32>>` shaped `[batch_size][n_outputs]`.
    pub fn get_output_counts(&self, rt: &CudaRuntime) -> CudaResult<Vec<Vec<u32>>> {
        let flat = rt.download(&self.d_output_counts)?;
        let n_out = self.n_outputs as usize;
        Ok(flat.chunks(n_out).map(|c| c.to_vec()).collect())
    }

    /// Download output counts as a flat `[batch_size × n_outputs]` vector.
    pub fn get_output_counts_flat(&self, rt: &CudaRuntime) -> CudaResult<Vec<u32>> {
        rt.download(&self.d_output_counts)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Reset helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Zero v_mem, refract, spiked, currents for all B instances.
    /// Call between samples to start each inference from rest.
    pub fn reset_state(&self, rt: &CudaRuntime) -> CudaResult<()> {
        let total = (self.batch_size * self.n_neurons) as u32;
        let cfg = launch_cfg_1d(total);
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.reset_state_fn)
                .arg(&self.d_v_mem)
                .arg(&self.d_refract)
                .arg(&self.d_spiked)
                .arg(&self.d_currents)
                .arg(&total)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("reset_state: {:?}", e)))?;
        }
        Ok(())
    }

    /// Zero output spike counts for all B instances.
    pub fn reset_output_counts(&self, rt: &CudaRuntime) -> CudaResult<()> {
        let total = (self.batch_size * self.n_outputs) as u32;
        let cfg = launch_cfg_1d(total);
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.reset_outputs_fn)
                .arg(&self.d_output_counts)
                .arg(&total)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("reset_outputs: {:?}", e)))?;
        }
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tick
    // ─────────────────────────────────────────────────────────────────────

    /// Run one simulation tick across all B instances.
    ///
    /// Execution order (no sync between kernels — they queue on the CUDA stream):
    /// 1. `snn_batch_accumulate` — propagate last tick's spikes into currents.
    /// 2. `snn_batch_lif_step`   — LIF dynamics + input generation + clear currents.
    pub fn tick(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        // 1. Accumulate currents — block-sparse or CSR.
        // Sprint 351: Automatically use block-sparse when enabled.
        if self.use_blocksparse {
            let total_threads = self.batch_size * self.bs_n_block_pairs;
            let lc = launch_cfg_1d(total_threads);
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.accumulate_blocksparse_weighted_fn)
                    .arg(self.d_bs_src_blocks.as_ref().unwrap())
                    .arg(self.d_bs_dst_blocks.as_ref().unwrap())
                    .arg(self.d_bs_synapse_masks.as_ref().unwrap())
                    .arg(self.d_bs_weight_offsets.as_ref().unwrap())
                    .arg(self.d_bs_packed_weights.as_ref().unwrap())
                    .arg(&self.d_spiked)
                    .arg(&self.d_currents)
                    .arg(&self.n_neurons)
                    .arg(&self.bs_n_block_pairs)
                    .arg(&self.batch_size)
                    .launch(lc)
                    .map_err(|e| CudaError::LaunchFailed(format!("accumulate_bs: {:?}", e)))?;
            }
        } else {
            let csr_total = (self.batch_size * self.n_neurons) as u32;
            let csr_cfg = launch_cfg_1d(csr_total);
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.accumulate_fn)
                    .arg(&self.d_syn_ptr)
                    .arg(&self.d_targets)
                    .arg(&self.d_weights)
                    .arg(&self.d_spiked)
                    .arg(&self.d_currents)
                    .arg(&self.n_neurons)
                    .arg(&self.batch_size)
                    .launch(csr_cfg)
                    .map_err(|e| CudaError::LaunchFailed(format!("batch_accumulate: {:?}", e)))?;
            }
        }

        // 2. LIF step + input generation (standard or ALIF)
        let total_neurons = (self.batch_size * self.n_neurons) as u32;
        let cfg = launch_cfg_1d(total_neurons);
        let seed = self.tick.wrapping_mul(2654435761) as u32;
        if self.alif_enabled {
            let adapt = self.d_adapt_state.as_ref().unwrap();
            let alpha = self.alif_alpha;
            let beta = self.alif_beta;
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.lif_step_alif_fn)
                    .arg(&self.d_v_mem)
                    .arg(&self.d_threshold)
                    .arg(&self.d_leak)
                    .arg(&self.d_refract)
                    .arg(&self.d_spiked)
                    .arg(&self.d_currents)
                    .arg(&self.d_input_rates)
                    .arg(adapt)
                    .arg(&alpha)
                    .arg(&beta)
                    .arg(&self.n_neurons)
                    .arg(&self.n_inputs)
                    .arg(&self.batch_size)
                    .arg(&seed)
                    .launch(cfg)
                    .map_err(|e| {
                        CudaError::LaunchFailed(format!("batch_lif_step_alif: {:?}", e))
                    })?;
            }
        } else {
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.lif_step_fn)
                    .arg(&self.d_v_mem)
                    .arg(&self.d_threshold)
                    .arg(&self.d_leak)
                    .arg(&self.d_refract)
                    .arg(&self.d_spiked)
                    .arg(&self.d_currents)
                    .arg(&self.d_input_rates)
                    .arg(&self.n_neurons)
                    .arg(&self.n_inputs)
                    .arg(&self.batch_size)
                    .arg(&seed)
                    .launch(cfg)
                    .map_err(|e| CudaError::LaunchFailed(format!("batch_lif_step: {:?}", e)))?;
            }
        }

        self.tick += 1;
        Ok(())
    }

    /// Accumulate output spike counts into `d_output_counts` for this tick.
    /// Call once per tick if you need per-tick counts; or call once after
    /// `tick_many` if you only need totals (slightly more efficient).
    pub fn count_outputs(&self, rt: &CudaRuntime) -> CudaResult<()> {
        let total_outputs = (self.batch_size * self.n_outputs) as u32;
        let cfg = launch_cfg_1d(total_outputs);
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.count_outputs_fn)
                .arg(&self.d_spiked)
                .arg(&self.d_output_counts)
                .arg(&self.output_start)
                .arg(&self.n_outputs)
                .arg(&self.n_neurons)
                .arg(&self.batch_size)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("batch_count_outputs: {:?}", e)))?;
        }
        Ok(())
    }

    /// Run `n_ticks` ticks, accumulating output spike counts every tick.
    ///
    /// This is the standard MNIST inference loop:
    /// ```ignore
    /// snn.tick_many_with_counting(&rt, 100)?;
    /// ```
    pub fn tick_many_with_counting(&mut self, rt: &CudaRuntime, n_ticks: usize) -> CudaResult<()> {
        for _ in 0..n_ticks {
            self.tick(rt)?;
            self.count_outputs(rt)?;
        }
        Ok(())
    }

    /// Run `n_ticks` without output counting (faster, use when counts not needed).
    pub fn tick_many(&mut self, rt: &CudaRuntime, n_ticks: usize) -> CudaResult<()> {
        for _ in 0..n_ticks {
            self.tick(rt)?;
        }
        Ok(())
    }

    /// Sprint 348: One tick using block-sparse accumulate instead of CSR.
    /// Requires GPU-uploaded block-sparse data.
    pub fn tick_blocksparse(
        &mut self,
        rt: &CudaRuntime,
        d_src_blocks: &CudaSlice<u16>,
        d_dst_blocks: &CudaSlice<u16>,
        d_synapse_masks: &CudaSlice<u64>,
        n_block_pairs: u32,
        fixed_weight: i32,
    ) -> CudaResult<()> {
        let total_threads = (self.batch_size as u32) * n_block_pairs;
        let cfg = launch_cfg_1d(total_threads);

        // Block-sparse accumulate.
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.accumulate_blocksparse_fn)
                .arg(d_src_blocks)
                .arg(d_dst_blocks)
                .arg(d_synapse_masks)
                .arg(&self.d_spiked)
                .arg(&self.d_currents)
                .arg(&self.n_neurons)
                .arg(&n_block_pairs)
                .arg(&self.batch_size)
                .arg(&fixed_weight)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("accumulate_blocksparse: {:?}", e)))?;
        }

        // LIF step (same as regular tick).
        let total_neurons = (self.batch_size * self.n_neurons) as u32;
        let lif_cfg = launch_cfg_1d(total_neurons);
        let seed = self.tick.wrapping_mul(2654435761) as u32;
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.lif_step_fn)
                .arg(&self.d_v_mem)
                .arg(&self.d_threshold)
                .arg(&self.d_leak)
                .arg(&self.d_refract)
                .arg(&self.d_spiked)
                .arg(&self.d_currents)
                .arg(&self.d_input_rates)
                .arg(&self.n_neurons)
                .arg(&self.n_inputs)
                .arg(&self.batch_size)
                .arg(&seed)
                .launch(lif_cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("lif_step (blocksparse): {:?}", e)))?;
        }
        self.tick += 1;
        Ok(())
    }

    /// Sprint 351: Enable block-sparse inference from CSR topology.
    /// Converts CSR edges to block-sparse format, uploads to GPU, and switches
    /// `tick()` to use the weighted block-sparse accumulate kernel.
    pub fn enable_blocksparse(
        &mut self,
        rt: &CudaRuntime,
        synapses: &SynapseCSRData,
    ) -> CudaResult<()> {
        let n = self.n_neurons as usize;
        let (src_blocks, dst_blocks, masks, weight_offsets, packed_weights, weight_remap) =
            Self::csr_to_blocksparse_static(n, synapses);
        self.d_bs_src_blocks = Some(rt.upload(&src_blocks)?);
        self.d_bs_dst_blocks = Some(rt.upload(&dst_blocks)?);
        self.d_bs_synapse_masks = Some(rt.upload(&masks)?);
        self.d_bs_weight_offsets = Some(rt.upload(&weight_offsets)?);
        self.d_bs_packed_weights = Some(rt.upload(&packed_weights)?);
        // Sprint 353: Upload remap for E-prop weight sync.
        self.d_bs_weight_remap = Some(rt.upload(&weight_remap)?);
        self.bs_n_packed_weights = packed_weights.len() as u32;
        self.bs_n_block_pairs = src_blocks.len() as u32;
        self.use_blocksparse = true;
        Ok(())
    }

    /// Sprint 351/353: Build block-sparse from CSR (static, no &self).
    /// Returns (src_blocks, dst_blocks, masks, weight_offsets, packed_weights, weight_remap).
    /// weight_remap[i] = CSR synapse index for packed weight position i.
    fn csr_to_blocksparse_static(
        n_neurons: usize,
        csr: &SynapseCSRData,
    ) -> (Vec<u16>, Vec<u16>, Vec<u64>, Vec<u32>, Vec<i8>, Vec<u32>) {
        use std::collections::BTreeMap;
        let block_size = 8usize;
        // Sprint 353: Track CSR index alongside bit+weight for remap.
        let mut block_map: BTreeMap<(u16, u16), (u64, Vec<(u8, i8, u32)>)> = BTreeMap::new();

        for src in 0..n_neurons {
            let start = csr.syn_ptr[src] as usize;
            let end = csr.syn_ptr[src + 1] as usize;
            let sb = (src / block_size) as u16;
            let si = (src % block_size) as u8;
            for j in start..end {
                let dst = csr.targets[j] as usize;
                let w = csr.weights[j];
                let db = (dst / block_size) as u16;
                let di = (dst % block_size) as u8;
                let bit = (si as u32) * 8 + (di as u32);
                let entry = block_map
                    .entry((sb, db))
                    .or_insert_with(|| (0u64, Vec::new()));
                entry.0 |= 1u64 << bit;
                entry.1.push((bit as u8, w, j as u32));
            }
        }

        let n_pairs = block_map.len();
        let mut src_blocks = Vec::with_capacity(n_pairs);
        let mut dst_blocks = Vec::with_capacity(n_pairs);
        let mut masks = Vec::with_capacity(n_pairs);
        let mut weight_offsets = Vec::with_capacity(n_pairs + 1);
        let mut packed_weights: Vec<i8> = Vec::new();
        let mut weight_remap: Vec<u32> = Vec::new();

        for ((sb, db), (mask, mut bits)) in block_map {
            src_blocks.push(sb);
            dst_blocks.push(db);
            masks.push(mask);
            weight_offsets.push(packed_weights.len() as u32);
            bits.sort_by_key(|(b, _, _)| *b);
            for (_, w, csr_idx) in bits {
                packed_weights.push(w);
                weight_remap.push(csr_idx);
            }
        }
        weight_offsets.push(packed_weights.len() as u32);

        (
            src_blocks,
            dst_blocks,
            masks,
            weight_offsets,
            packed_weights,
            weight_remap,
        )
    }

    /// Sprint 349: One tick using weighted block-sparse accumulate.
    pub fn tick_blocksparse_weighted(
        &mut self,
        rt: &CudaRuntime,
        d_src_blocks: &CudaSlice<u16>,
        d_dst_blocks: &CudaSlice<u16>,
        d_synapse_masks: &CudaSlice<u64>,
        d_weight_offsets: &CudaSlice<u32>,
        d_packed_weights: &CudaSlice<i8>,
        n_block_pairs: u32,
    ) -> CudaResult<()> {
        let total_threads = (self.batch_size as u32) * n_block_pairs;
        let cfg = launch_cfg_1d(total_threads);

        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.accumulate_blocksparse_weighted_fn)
                .arg(d_src_blocks)
                .arg(d_dst_blocks)
                .arg(d_synapse_masks)
                .arg(d_weight_offsets)
                .arg(d_packed_weights)
                .arg(&self.d_spiked)
                .arg(&self.d_currents)
                .arg(&self.n_neurons)
                .arg(&n_block_pairs)
                .arg(&self.batch_size)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("accumulate_bs_weighted: {:?}", e)))?;
        }

        let total_neurons = (self.batch_size * self.n_neurons) as u32;
        let lif_cfg = launch_cfg_1d(total_neurons);
        let seed = self.tick.wrapping_mul(2654435761) as u32;
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.lif_step_fn)
                .arg(&self.d_v_mem)
                .arg(&self.d_threshold)
                .arg(&self.d_leak)
                .arg(&self.d_refract)
                .arg(&self.d_spiked)
                .arg(&self.d_currents)
                .arg(&self.d_input_rates)
                .arg(&self.n_neurons)
                .arg(&self.n_inputs)
                .arg(&self.batch_size)
                .arg(&seed)
                .launch(lif_cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("lif_step (bs_weighted): {:?}", e)))?;
        }
        self.tick += 1;
        Ok(())
    }

    /// Sprint 349: tick_many with weighted block-sparse + output counting.
    pub fn tick_many_blocksparse_weighted_with_counting(
        &mut self,
        rt: &CudaRuntime,
        n_ticks: usize,
        d_src_blocks: &CudaSlice<u16>,
        d_dst_blocks: &CudaSlice<u16>,
        d_synapse_masks: &CudaSlice<u64>,
        d_weight_offsets: &CudaSlice<u32>,
        d_packed_weights: &CudaSlice<i8>,
        n_block_pairs: u32,
    ) -> CudaResult<()> {
        for _ in 0..n_ticks {
            self.tick_blocksparse_weighted(
                rt,
                d_src_blocks,
                d_dst_blocks,
                d_synapse_masks,
                d_weight_offsets,
                d_packed_weights,
                n_block_pairs,
            )?;
            self.count_outputs(rt)?;
        }
        Ok(())
    }

    /// Sprint 348: Run n_ticks using block-sparse accumulate + output counting.
    pub fn tick_many_blocksparse_with_counting(
        &mut self,
        rt: &CudaRuntime,
        n_ticks: usize,
        d_src_blocks: &CudaSlice<u16>,
        d_dst_blocks: &CudaSlice<u16>,
        d_synapse_masks: &CudaSlice<u64>,
        n_block_pairs: u32,
        fixed_weight: i32,
    ) -> CudaResult<()> {
        for _ in 0..n_ticks {
            self.tick_blocksparse(
                rt,
                d_src_blocks,
                d_dst_blocks,
                d_synapse_masks,
                n_block_pairs,
                fixed_weight,
            )?;
            self.count_outputs(rt)?;
        }
        Ok(())
    }

    /// Wait for all queued GPU operations to complete.
    pub fn synchronize(&self, rt: &CudaRuntime) -> CudaResult<()> {
        rt.synchronize()
    }

    // ─────────────────────────────────────────────────────────────────────
    // Phase 4: STDP — train→deploy path
    // ─────────────────────────────────────────────────────────────────────

    /// Allocate the `[B × N]` all-neuron spike count buffer and enable STDP.
    ///
    /// Must be called once before `reset_all_counts`, `tick_many_full_counting`,
    /// `get_all_counts`, or `apply_hebbian_reward`.
    pub fn enable_stdp(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        let total = (self.batch_size * self.n_neurons) as usize;
        self.d_all_counts = Some(rt.alloc_zeros::<u32>(total)?);
        self.stdp_enabled = true;
        Ok(())
    }

    /// Zero the full [B × N] spike count buffer.  Call before each forward pass.
    ///
    /// Requires `enable_stdp()` to have been called first.
    pub fn reset_all_counts(&self, rt: &CudaRuntime) -> CudaResult<()> {
        let buf = self.d_all_counts.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed("reset_all_counts: enable_stdp() not called".into())
        })?;
        let total = (self.batch_size * self.n_neurons) as u32;
        let cfg = launch_cfg_1d(total);
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.reset_all_counts_fn)
                .arg(buf)
                .arg(&total)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("reset_all_counts: {:?}", e)))?;
        }
        Ok(())
    }

    /// Run `n_ticks` ticks, counting spikes from **all** neurons every tick.
    ///
    /// After this call `get_all_counts()` returns cumulative spike counts for
    /// every neuron in every batch instance.  Also accumulates `d_output_counts`
    /// as usual (backward-compatible with `tick_many_with_counting`).
    ///
    /// Requires `enable_stdp()` to have been called first.
    pub fn tick_many_full_counting(&mut self, rt: &CudaRuntime, n_ticks: usize) -> CudaResult<()> {
        let buf = self.d_all_counts.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed("tick_many_full_counting: enable_stdp() not called".into())
        })? as *const CudaSlice<u32>;

        let total_neurons = (self.batch_size * self.n_neurons) as u32;
        let cfg_all = launch_cfg_1d(total_neurons);

        for _ in 0..n_ticks {
            self.tick(rt)?;
            self.count_outputs(rt)?;

            // Count all neurons (including inputs and hidden)
            let buf_ref = unsafe { &*buf };
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.count_all_fn)
                    .arg(&self.d_spiked)
                    .arg(buf_ref)
                    .arg(&self.n_neurons)
                    .arg(&self.batch_size)
                    .launch(cfg_all)
                    .map_err(|e| CudaError::LaunchFailed(format!("count_all: {:?}", e)))?;
            }
        }
        Ok(())
    }

    /// Run `n_ticks` ticks accumulating hidden spike counts AND building E-prop eligibility traces.
    ///
    /// Combines `tick_many_full_counting` (kernel 6: count all neurons) with
    /// `tick_many_with_eprop` (kernels 10-11: pre-trace + eligibility) in a single pass.
    ///
    /// Kernel order per tick:
    ///   1. LIF step (accumulate + snn_batch_lif_step)
    ///   2. count_outputs (kernel 3)
    ///   3. count_all     (kernel 6) → d_all_counts  (for hidden rate extraction)
    ///   4. update_pre_trace (kernel 10)
    ///   5. update_eligibility (kernel 11)
    ///
    /// Requires `enable_stdp()` (for `d_all_counts`) and `enable_eprop()` (for traces).
    /// After ticking, call `get_all_counts()` for hidden features and
    /// `apply_eprop_update_layered()` to update IH weights.
    pub fn tick_many_with_eprop_and_hidden_counting(
        &mut self,
        rt: &CudaRuntime,
        n_ticks: usize,
    ) -> CudaResult<()> {
        let buf_ptr = self.d_all_counts.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed(
                "tick_many_with_eprop_and_hidden_counting: enable_stdp() not called".into(),
            )
        })? as *const CudaSlice<u32>;
        let pre_ptr = self.d_pre_trace.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed(
                "tick_many_with_eprop_and_hidden_counting: enable_eprop() not called".into(),
            )
        })? as *const CudaSlice<f32>;
        let elig_ptr = self.d_eligibility.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed(
                "tick_many_with_eprop_and_hidden_counting: enable_eprop() not called".into(),
            )
        })? as *const CudaSlice<f32>;

        let total_neurons = (self.batch_size * self.n_neurons) as u32;
        let total_synapses = (self.batch_size * self.n_synapses) as u32;
        let cfg_n = launch_cfg_1d(total_neurons);
        let cfg_nnz = launch_cfg_1d(total_synapses);

        let alpha_pre = self.eprop_alpha_pre;
        let alpha_elig = self.eprop_alpha_elig;
        let gamma = self.eprop_gamma;

        for _ in 0..n_ticks {
            // 1. LIF step
            self.tick(rt)?;
            // 2. Output counting
            self.count_outputs(rt)?;

            // 3. Count all neurons (for hidden rate extraction via get_all_counts)
            let buf = unsafe { &*buf_ptr };
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.count_all_fn)
                    .arg(&self.d_spiked)
                    .arg(buf)
                    .arg(&self.n_neurons)
                    .arg(&self.batch_size)
                    .launch(cfg_n)
                    .map_err(|e| CudaError::LaunchFailed(format!("count_all: {:?}", e)))?;
            }

            // 4. Pre-synaptic trace update
            let pre = unsafe { &*pre_ptr };
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.update_pre_trace_fn)
                    .arg(&self.d_spiked)
                    .arg(pre)
                    .arg(&alpha_pre)
                    .arg(&total_neurons)
                    .launch(cfg_n)
                    .map_err(|e| CudaError::LaunchFailed(format!("update_pre_trace: {:?}", e)))?;
            }

            // 5. Eligibility trace update
            let elig = unsafe { &*elig_ptr };
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.update_eligibility_fn)
                    .arg(pre)
                    .arg(&self.d_v_mem)
                    .arg(&self.d_threshold)
                    .arg(&self.d_src_of_syn)
                    .arg(&self.d_targets)
                    .arg(elig)
                    .arg(&alpha_elig)
                    .arg(&gamma)
                    .arg(&self.n_neurons)
                    .arg(&self.n_synapses)
                    .arg(&self.batch_size)
                    .launch(cfg_nnz)
                    .map_err(|e| CudaError::LaunchFailed(format!("update_eligibility: {:?}", e)))?;
            }
        }
        Ok(())
    }

    /// Download all-neuron spike counts as `[batch_size][n_neurons]`.
    ///
    /// Requires `enable_stdp()` and at least one `tick_many_full_counting` call.
    pub fn get_all_counts(&self, rt: &CudaRuntime) -> CudaResult<Vec<Vec<u32>>> {
        let buf = self.d_all_counts.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed("get_all_counts: enable_stdp() not called".into())
        })?;
        let flat = rt.download(buf)?;
        let n = self.n_neurons as usize;
        Ok(flat.chunks(n).map(|c| c.to_vec()).collect())
    }

    /// Apply a batch reward-modulated Hebbian weight update to the shared weights.
    ///
    /// For each synapse j (src→dst):
    /// ```text
    /// Δw_j = lr × (1/B) × Σ_b counts[b][src] × counts[b][dst] × reward[b] / n_ticks²
    /// ```
    ///
    /// # Arguments
    /// * `rewards`       – Per-sample reward signal `[batch_size]`.
    ///                     Typical values: +1.0 (correct), -1.0 (wrong).
    /// * `n_ticks`       – Number of ticks in the forward pass (for normalisation).
    /// * `learning_rate` – Step size.
    ///
    /// Requires `enable_stdp()` and at least one `tick_many_full_counting` call.
    pub fn apply_hebbian_reward(
        &mut self,
        rt: &CudaRuntime,
        rewards: &[f32],
        n_ticks: usize,
        learning_rate: f32,
    ) -> CudaResult<()> {
        assert_eq!(
            rewards.len(),
            self.batch_size as usize,
            "rewards.len() must equal batch_size"
        );
        let buf = self.d_all_counts.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed("apply_hebbian_reward: enable_stdp() not called".into())
        })?;

        let d_rewards = rt.upload(rewards)?;
        let inv_norm = 1.0f32 / (n_ticks as f32 * n_ticks as f32 * self.batch_size as f32);
        let cfg = launch_cfg_1d(self.n_synapses);
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.apply_hebbian_fn)
                .arg(&self.d_src_of_syn)
                .arg(&self.d_targets)
                .arg(&self.d_weights)
                .arg(buf)
                .arg(&d_rewards)
                .arg(&self.n_synapses)
                .arg(&self.n_neurons)
                .arg(&self.batch_size)
                .arg(&learning_rate)
                .arg(&inv_norm)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("apply_hebbian: {:?}", e)))?;
        }
        // Sprint 353: Auto-sync block-sparse packed weights after direct i8 update.
        self.sync_blocksparse_weights(rt)?;
        Ok(())
    }

    /// Download a copy of the current weight vector from GPU.
    pub fn get_weights(&self, rt: &CudaRuntime) -> CudaResult<Vec<i8>> {
        rt.download(&self.d_weights)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Phase 5: ALIF — Adaptive LIF neurons
    // ─────────────────────────────────────────────────────────────────────

    /// Enable ALIF adaptive thresholds.
    ///
    /// Allocates `[B × N]` adaptation state buffer (zero-initialized).
    /// After this call, `tick()` uses the ALIF kernel instead of standard LIF.
    ///
    /// # Arguments
    /// * `alpha_adapt` – Decay coefficient for adaptation state. e.g. `0.967 ≈ exp(-1/30)`
    ///   (time constant of ~30 ticks). Larger = slower decay = stronger adaptation memory.
    /// * `beta_adapt`  – Threshold scale factor, e.g. `0.1`. Each spike raises the effective
    ///   threshold by `beta_adapt` threshold units; decays with `alpha_adapt`.
    pub fn enable_alif(
        &mut self,
        rt: &CudaRuntime,
        alpha_adapt: f32,
        beta_adapt: f32,
    ) -> CudaResult<()> {
        let total = (self.batch_size * self.n_neurons) as usize;
        self.d_adapt_state = Some(rt.alloc_zeros::<i16>(total)?);
        self.alif_alpha = alpha_adapt;
        self.alif_beta = beta_adapt;
        self.alif_enabled = true;
        Ok(())
    }

    /// Zero the adaptation state buffer. Call between episodes during training.
    ///
    /// No-op if `enable_alif()` has not been called.
    pub fn reset_adapt_state(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        if let Some(ref mut buf) = self.d_adapt_state {
            rt.memset_zeros(buf)?;
        }
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // Phase 5: E-prop — Online eligibility traces
    // ─────────────────────────────────────────────────────────────────────

    /// Enable e-prop eligibility trace buffers.
    ///
    /// Allocates:
    /// - `d_pre_trace [B × N]` — pre-synaptic low-pass filtered spike train
    /// - `d_eligibility [B × nnz]` — per-synapse eligibility traces
    ///
    /// # Arguments
    /// * `alpha_pre`  – Pre-trace decay (e.g. `0.95`).
    /// * `alpha_elig` – Eligibility trace decay (e.g. `0.95`).
    /// * `gamma`      – Surrogate gradient sharpness for fast-sigmoid (e.g. `0.3`).
    pub fn enable_eprop(
        &mut self,
        rt: &CudaRuntime,
        alpha_pre: f32,
        alpha_elig: f32,
        gamma: f32,
    ) -> CudaResult<()> {
        let bn = (self.batch_size * self.n_neurons) as usize;
        let bnnz = (self.batch_size * self.n_synapses) as usize;
        self.d_pre_trace = Some(rt.alloc_zeros::<f32>(bn)?);
        self.d_eligibility = Some(rt.alloc_zeros::<f32>(bnnz)?);
        self.eprop_alpha_pre = alpha_pre;
        self.eprop_alpha_elig = alpha_elig;
        self.eprop_gamma = gamma;
        self.eprop_enabled = true;
        Ok(())
    }

    /// Zero pre_trace and eligibility buffers. Call between episodes.
    ///
    /// Requires `enable_eprop()` to have been called first.
    pub fn reset_eprop_state(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        rt.memset_zeros(self.d_pre_trace.as_mut().ok_or_else(|| {
            CudaError::LaunchFailed("reset_eprop_state: enable_eprop() not called".into())
        })?)?;
        rt.memset_zeros(self.d_eligibility.as_mut().ok_or_else(|| {
            CudaError::LaunchFailed("reset_eprop_state: enable_eprop() not called".into())
        })?)?;
        Ok(())
    }

    /// Run `n_ticks` ticks with full e-prop accounting.
    ///
    /// Each tick runs (in order):
    /// 1. LIF dynamics (ALIF kernel if `enable_alif()` was called, else standard)
    /// 2. Output spike counting (`d_output_counts`)
    /// 3. Pre-trace update (kernel 10)
    /// 4. Eligibility trace update (kernel 11)
    ///
    /// Requires `enable_eprop()`.
    pub fn tick_many_with_eprop(&mut self, rt: &CudaRuntime, n_ticks: usize) -> CudaResult<()> {
        // Borrow raw pointers to avoid simultaneous &self / &mut self conflicts.
        let pre_ptr = self.d_pre_trace.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed("tick_many_with_eprop: enable_eprop() not called".into())
        })? as *const CudaSlice<f32>;
        let elig_ptr = self.d_eligibility.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed("tick_many_with_eprop: enable_eprop() not called".into())
        })? as *const CudaSlice<f32>;

        let total_neurons = (self.batch_size * self.n_neurons) as u32;
        let total_synapses = (self.batch_size * self.n_synapses) as u32;
        let cfg_n = launch_cfg_1d(total_neurons);
        let cfg_nnz = launch_cfg_1d(total_synapses);

        let alpha_pre = self.eprop_alpha_pre;
        let alpha_elig = self.eprop_alpha_elig;
        let gamma = self.eprop_gamma;

        for _ in 0..n_ticks {
            // 1. LIF step (dispatches ALIF or standard)
            self.tick(rt)?;
            // 2. Output counting
            self.count_outputs(rt)?;

            // 3. Pre-trace update
            let pre = unsafe { &*pre_ptr };
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.update_pre_trace_fn)
                    .arg(&self.d_spiked)
                    .arg(pre)
                    .arg(&alpha_pre)
                    .arg(&total_neurons)
                    .launch(cfg_n)
                    .map_err(|e| CudaError::LaunchFailed(format!("update_pre_trace: {:?}", e)))?;
            }

            // 4. Eligibility trace update
            let elig = unsafe { &*elig_ptr };
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.update_eligibility_fn)
                    .arg(pre)
                    .arg(&self.d_v_mem)
                    .arg(&self.d_threshold)
                    .arg(&self.d_src_of_syn)
                    .arg(&self.d_targets)
                    .arg(elig)
                    .arg(&alpha_elig)
                    .arg(&gamma)
                    .arg(&self.n_neurons)
                    .arg(&self.n_synapses)
                    .arg(&self.batch_size)
                    .launch(cfg_nnz)
                    .map_err(|e| CudaError::LaunchFailed(format!("update_eligibility: {:?}", e)))?;
            }
        }
        Ok(())
    }

    /// Apply e-prop weight update at end of episode.
    ///
    /// `Δw_j = lr × (1/B) × Σ_b eligibility[b][j] × learning_signal[b]`
    ///
    /// # Arguments
    /// * `learning_signals` – `[batch_size]` values. Typical: `+1.0` correct, `-1.0` wrong.
    /// * `learning_rate`    – Step size (recommended: 0.0001–0.001).
    ///
    /// Requires `enable_eprop()`.
    pub fn apply_eprop_update(
        &mut self,
        rt: &CudaRuntime,
        learning_signals: &[f32],
        learning_rate: f32,
    ) -> CudaResult<()> {
        assert_eq!(
            learning_signals.len(),
            self.batch_size as usize,
            "learning_signals.len() must equal batch_size"
        );
        let elig = self.d_eligibility.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed("apply_eprop_update: enable_eprop() not called".into())
        })?;

        let d_signals = rt.upload(learning_signals)?;
        let cfg = launch_cfg_1d(self.n_synapses);
        if let Some(ref wf32) = self.d_weights_f32 {
            // Float32 shadow path: no i8 precision dead-zone.
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.apply_eprop_f32_fn)
                    .arg(elig)
                    .arg(&d_signals)
                    .arg(wf32)
                    .arg(&self.n_synapses)
                    .arg(&self.batch_size)
                    .arg(&learning_rate)
                    .launch(cfg)
                    .map_err(|e| CudaError::LaunchFailed(format!("apply_eprop_f32: {:?}", e)))?;
            }
        } else {
            // Legacy i8 path: kept for backward compatibility.
            unsafe {
                rt.stream()
                    .launch_builder(&self.kernels.apply_eprop_fn)
                    .arg(elig)
                    .arg(&d_signals)
                    .arg(&self.d_weights)
                    .arg(&self.n_synapses)
                    .arg(&self.batch_size)
                    .arg(&learning_rate)
                    .launch(cfg)
                    .map_err(|e| CudaError::LaunchFailed(format!("apply_eprop: {:?}", e)))?;
            }
            // Sprint 353: Auto-sync block-sparse packed weights after direct i8 update.
            self.sync_blocksparse_weights(rt)?;
        }
        Ok(())
    }

    /// Apply e-prop weight update with separate learning rates for two synapse layers.
    ///
    /// Synapses `[0, boundary)` use `lr_a`; synapses `[boundary, n_synapses)` use `lr_b`.
    /// Intended for 3-layer networks where the readout (HR) layer needs a higher LR than
    /// the Fisher-calibrated input→hidden (IH) layer.
    ///
    /// Requires `init_weight_shadow()` (uses f32 shadow path only).
    ///
    /// # Arguments
    /// * `learning_signals` – `[batch_size]` values (+1.0/−1.0/0.0).
    /// * `lr_a`             – Learning rate for IH synapses `[0, boundary)`.
    /// * `lr_b`             – Learning rate for HR synapses `[boundary, n_synapses)`.
    /// * `boundary`         – Index of the first HR synapse (`n_ih_synapses`).
    pub fn apply_eprop_update_layered(
        &mut self,
        rt: &CudaRuntime,
        learning_signals: &[f32],
        lr_a: f32,
        lr_b: f32,
        boundary: usize,
    ) -> CudaResult<()> {
        assert_eq!(
            learning_signals.len(),
            self.batch_size as usize,
            "learning_signals.len() must equal batch_size"
        );
        let elig = self.d_eligibility.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed("apply_eprop_update_layered: enable_eprop() not called".into())
        })?;
        let wf32 = self.d_weights_f32.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed(
                "apply_eprop_update_layered: init_weight_shadow() not called".into(),
            )
        })?;

        let d_signals = rt.upload(learning_signals)?;
        let bd = boundary as u32;
        let cfg = launch_cfg_1d(self.n_synapses);
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.apply_eprop_f32_layered_fn)
                .arg(elig)
                .arg(&d_signals)
                .arg(wf32)
                .arg(&self.n_synapses)
                .arg(&self.batch_size)
                .arg(&lr_a)
                .arg(&lr_b)
                .arg(&bd)
                .launch(cfg)
                .map_err(|e| {
                    CudaError::LaunchFailed(format!("apply_eprop_f32_layered: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Apply e-prop weight update with per-class learning signals (M8).
    ///
    /// Each synapse `j` is mapped to its class via `d_synapse_class[j]` and uses
    /// `learning_signals[b * n_classes + class]` instead of a scalar per sample.
    /// This enables class-specific weight specialization.
    ///
    /// Requires `enable_eprop()`, `init_weight_shadow()`, `init_synapse_class_map()`.
    ///
    /// # Arguments
    /// * `learning_signals` – `[batch_size * n_classes]` flat row-major.
    ///   Typically `y_c - softmax_c` (CE gradient per class).
    /// * `lr_a`             – Learning rate for IH synapses `[0, boundary)`.
    /// * `lr_b`             – Learning rate for HR synapses `[boundary, n_synapses)`.
    /// * `n_classes`        – Number of output classes.
    /// * `boundary`         – Index of the first HR synapse (`n_ih_synapses`).
    pub fn apply_eprop_update_per_class(
        &mut self,
        rt: &CudaRuntime,
        learning_signals: &[f32],
        lr_a: f32,
        lr_b: f32,
        n_classes: usize,
        boundary: usize,
    ) -> CudaResult<()> {
        assert_eq!(
            learning_signals.len(),
            self.batch_size as usize * n_classes,
            "learning_signals.len() must equal batch_size * n_classes"
        );
        let elig = self.d_eligibility.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed(
                "apply_eprop_update_per_class: enable_eprop() not called".into(),
            )
        })?;
        let wf32 = self.d_weights_f32.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed(
                "apply_eprop_update_per_class: init_weight_shadow() not called".into(),
            )
        })?;
        let syn_class = self.d_synapse_class.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed(
                "apply_eprop_update_per_class: init_synapse_class_map() not called".into(),
            )
        })?;

        let d_signals = rt.upload(learning_signals)?;
        let nc = n_classes as u32;
        let bd = boundary as u32;
        let cfg = launch_cfg_1d(self.n_synapses);
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.apply_eprop_f32_per_class_fn)
                .arg(elig)
                .arg(&d_signals)
                .arg(syn_class)
                .arg(wf32)
                .arg(&self.n_synapses)
                .arg(&self.batch_size)
                .arg(&nc)
                .arg(&lr_a)
                .arg(&lr_b)
                .arg(&bd)
                .launch(cfg)
                .map_err(|e| {
                    CudaError::LaunchFailed(format!("apply_eprop_f32_per_class: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Apply E-prop update using device-resident learning signals.
    ///
    /// Identical to `apply_eprop_update_per_class` but takes a `&CudaSlice<f32>`
    /// directly, avoiding the CPU→GPU upload of learning signals.
    /// Used by `GpuMLP` which computes signals on GPU.
    pub fn apply_eprop_update_per_class_device(
        &mut self,
        rt: &CudaRuntime,
        d_learning_signals: &CudaSlice<f32>,
        lr_a: f32,
        lr_b: f32,
        n_classes: usize,
        boundary: usize,
    ) -> CudaResult<()> {
        let elig = self.d_eligibility.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed(
                "apply_eprop_update_per_class_device: enable_eprop() not called".into(),
            )
        })?;
        let wf32 = self.d_weights_f32.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed(
                "apply_eprop_update_per_class_device: init_weight_shadow() not called".into(),
            )
        })?;
        let syn_class = self.d_synapse_class.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed(
                "apply_eprop_update_per_class_device: init_synapse_class_map() not called".into(),
            )
        })?;

        let nc = n_classes as u32;
        let bd = boundary as u32;
        let cfg = launch_cfg_1d(self.n_synapses);
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.apply_eprop_f32_per_class_fn)
                .arg(elig)
                .arg(d_learning_signals)
                .arg(syn_class)
                .arg(wf32)
                .arg(&self.n_synapses)
                .arg(&self.batch_size)
                .arg(&nc)
                .arg(&lr_a)
                .arg(&lr_b)
                .arg(&bd)
                .launch(cfg)
                .map_err(|e| {
                    CudaError::LaunchFailed(format!("apply_eprop_f32_per_class_device: {:?}", e))
                })?;
        }
        Ok(())
    }

    /// Return a reference to the device-resident all-neuron spike count buffer.
    ///
    /// Used by `GpuMLP` to read hidden counts without D2H transfer.
    /// Requires `enable_stdp()` to have been called first.
    pub fn all_counts_device(&self) -> CudaResult<&CudaSlice<u32>> {
        self.d_all_counts.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed("all_counts_device: enable_stdp() not called".into())
        })
    }

    /// Initialise the float32 weight shadow from the current i8 weights.
    ///
    /// Call once after `enable_eprop()` to switch `apply_eprop_update` from the
    /// i8-truncating kernel to the float32-accumulating kernel, which avoids the
    /// precision dead-zone where `delta * 127 < 1` always truncates to 0.
    ///
    /// After calling this, `apply_eprop_update` accumulates in f32.
    /// Call `project_weights_f32` before each test/inference evaluation to sync
    /// the i8 weight buffer used by the forward-pass kernels.
    pub fn init_weight_shadow(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        let mut buf = rt.alloc_zeros::<f32>(self.n_synapses as usize)?;
        let cfg = launch_cfg_1d(self.n_synapses);
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.init_weight_shadow_fn)
                .arg(&self.d_weights)
                .arg(&mut buf)
                .arg(&self.n_synapses)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("init_weight_shadow: {:?}", e)))?;
        }
        self.d_weights_f32 = Some(buf);
        Ok(())
    }

    /// Project the float32 weight shadow back to the i8 weight buffer.
    ///
    /// Must be called before any inference (`tick_many_with_counting`) after
    /// training updates, so the forward-pass kernels see the latest weights.
    /// Requires `init_weight_shadow()` to have been called.
    pub fn project_weights_f32(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        let buf = self.d_weights_f32.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed("project_weights_f32: init_weight_shadow() not called".into())
        })?;
        let cfg = launch_cfg_1d(self.n_synapses);
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.project_weights_f32_fn)
                .arg(buf)
                .arg(&self.d_weights)
                .arg(&self.n_synapses)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("project_weights_f32: {:?}", e)))?;
        }
        // Sprint 353: Auto-sync block-sparse packed weights after CSR i8 update.
        self.sync_blocksparse_weights(rt)?;
        Ok(())
    }

    /// Sprint 353: Sync CSR i8 weights → block-sparse packed weights.
    ///
    /// After `project_weights_f32()` updates `d_weights` (CSR order), this copies
    /// the updated i8 weights into `d_bs_packed_weights` (block-sparse packed order)
    /// using the precomputed remap from `enable_blocksparse()`.
    ///
    /// Call this after `project_weights_f32()` when block-sparse is enabled,
    /// so the next forward pass sees the trained weights.
    pub fn sync_blocksparse_weights(&mut self, rt: &CudaRuntime) -> CudaResult<()> {
        if !self.use_blocksparse {
            return Ok(()); // no-op when block-sparse not active
        }
        let remap = self.d_bs_weight_remap.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed(
                "sync_blocksparse_weights: enable_blocksparse() not called".into(),
            )
        })?;
        let packed = self.d_bs_packed_weights.as_mut().ok_or_else(|| {
            CudaError::LaunchFailed("sync_blocksparse_weights: no packed weights".into())
        })?;
        let n_packed = self.bs_n_packed_weights;
        let cfg = launch_cfg_1d(n_packed);
        unsafe {
            rt.stream()
                .launch_builder(&self.kernels.sync_blocksparse_weights_fn)
                .arg(&self.d_weights)
                .arg(packed)
                .arg(remap)
                .arg(&n_packed)
                .launch(cfg)
                .map_err(|e| {
                    CudaError::LaunchFailed(format!("sync_blocksparse_weights: {:?}", e))
                })?;
        }
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // M8: Per-class synapse mapping
    // ─────────────────────────────────────────────────────────────────────

    /// Build and upload a per-synapse class map for per-class E-prop signals.
    ///
    /// For each synapse `j`, determines which class it belongs to based on the
    /// source neuron:
    /// - IH synapse (`src < n_inp`): `class = src / k_per_class`
    /// - HR synapse (`src >= n_inp`): `class = (src - n_inp) / h_per_class`
    ///
    /// Computed once from `d_src_of_syn` (already GPU-resident); ~192 KB for
    /// typical networks. Required by `apply_eprop_update_per_class()`.
    pub fn init_synapse_class_map(
        &mut self,
        rt: &CudaRuntime,
        k_per_class: usize,
        h_per_class: usize,
        n_inp: usize,
    ) -> CudaResult<()> {
        let src_of_syn = rt.download(&self.d_src_of_syn)?;
        let mut class_map = vec![0u32; src_of_syn.len()];
        for (j, &src) in src_of_syn.iter().enumerate() {
            let src = src as usize;
            class_map[j] = if src < n_inp {
                (src / k_per_class) as u32
            } else {
                ((src - n_inp) / h_per_class) as u32
            };
        }
        self.d_synapse_class = Some(rt.upload(&class_map)?);
        Ok(())
    }

    /// Build per-synapse target-hidden-neuron map for per-neuron E-prop signals.
    ///
    /// Maps each synapse `j` to its target hidden neuron index (0-based within
    /// the hidden layer): `target_map[j] = targets[j] - n_inp` for hidden targets.
    /// Non-hidden targets (HR synapses targeting readout neurons) get sentinel
    /// index `n_hid`, which should point to a zero-padded signal channel.
    ///
    /// Used with `apply_eprop_update_per_class()` where `n_classes = n_hid + 1`
    /// and `learning_signals[B * (n_hid + 1)]` has channels 0..n_hid-1 for hidden
    /// neurons and channel n_hid always zero (sentinel). This keeps HR synapses
    /// frozen regardless of lr_b.
    pub fn init_synapse_target_hidden_map(
        &mut self,
        rt: &CudaRuntime,
        n_inp: usize,
        n_hid: usize,
    ) -> CudaResult<()> {
        let targets = rt.download(&self.d_targets)?;
        let mut target_map = vec![0u32; targets.len()];
        for (j, &tgt) in targets.iter().enumerate() {
            let tgt = tgt as usize;
            target_map[j] = if tgt >= n_inp && tgt < n_inp + n_hid {
                (tgt - n_inp) as u32
            } else {
                n_hid as u32 // sentinel → zero-padded signal channel
            };
        }
        self.d_synapse_class = Some(rt.upload(&target_map)?);
        Ok(())
    }

    /// Download eligibility stats: `(mean_abs, max_abs)`.
    ///
    /// Useful for diagnosing convergence: mean_abs should be non-zero and
    /// growing over the first few epochs if learning is proceeding correctly.
    ///
    /// Requires `enable_eprop()`.
    pub fn get_eligibility_stats(&self, rt: &CudaRuntime) -> CudaResult<(f32, f32)> {
        let elig = self.d_eligibility.as_ref().ok_or_else(|| {
            CudaError::LaunchFailed("get_eligibility_stats: enable_eprop() not called".into())
        })?;
        rt.synchronize()?;
        let flat = rt.download(elig)?;
        if flat.is_empty() {
            return Ok((0.0, 0.0));
        }
        let mean_abs = flat.iter().map(|x| x.abs()).sum::<f32>() / flat.len() as f32;
        let max_abs = flat.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        Ok((mean_abs, max_abs))
    }

    // ─────────────────────────────────────────────────────────────────────
    // Getters
    // ─────────────────────────────────────────────────────────────────────

    pub fn n_neurons(&self) -> usize {
        self.n_neurons as usize
    }
    pub fn n_inputs(&self) -> usize {
        self.n_inputs as usize
    }
    pub fn n_outputs(&self) -> usize {
        self.n_outputs as usize
    }
    pub fn batch_size(&self) -> usize {
        self.batch_size as usize
    }
    pub fn current_tick(&self) -> u64 {
        self.tick
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn launch_cfg_1d(n: u32) -> LaunchConfig {
    const BLOCK: u32 = 256;
    LaunchConfig {
        block_dim: (BLOCK, 1, 1),
        grid_dim: ((n + BLOCK - 1) / BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::*;

    /// Ring topology: neuron i → neuron (i+1) % n
    #[cfg(feature = "cuda")]
    fn make_ring_synapses(n: usize) -> SynapseCSRData {
        let mut syn_ptr = vec![0u32; n + 1];
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        for i in 0..n {
            syn_ptr[i + 1] = syn_ptr[i] + 1;
            targets.push(((i + 1) % n) as u32);
            weights.push(50i8);
        }
        SynapseCSRData {
            syn_ptr,
            targets,
            weights,
        }
    }

    /// Direct topology: each input neuron connects to every output neuron.
    /// Input neurons: [0, n_inputs). Output neurons: [n_neurons-n_outputs, n_neurons).
    #[cfg(feature = "cuda")]
    fn make_direct_synapses(n_neurons: usize, n_inputs: usize, n_outputs: usize) -> SynapseCSRData {
        let output_start = n_neurons - n_outputs;
        let mut syn_ptr = vec![0u32; n_neurons + 1];
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        for i in 0..n_neurons {
            if i < n_inputs {
                for j in output_start..n_neurons {
                    targets.push(j as u32);
                    weights.push(50i8);
                }
            }
            syn_ptr[i + 1] = targets.len() as u32;
        }
        SynapseCSRData {
            syn_ptr,
            targets,
            weights,
        }
    }

    /// Batch of 1: direct input→output topology, high rate → output must fire within 50 ticks.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_batch_basic_fires() {
        let rt = CudaRuntime::new().expect("CUDA init");
        let n = 8;
        let n_inputs = 2;
        let n_outputs = 2;
        // Direct synapses: each input neuron drives every output neuron.
        // With weight=50 (current=100) and threshold=256, two simultaneous
        // input spikes give 200 current — fires in ~2 ticks.
        let synapses = make_direct_synapses(n, n_inputs, n_outputs);
        let thresholds = vec![256i16; n];
        let leaks = vec![230u8; n];

        let mut snn = GpuBatchSNN::new(
            &rt,
            n,
            n_inputs,
            n_outputs,
            &thresholds,
            &leaks,
            &synapses,
            1,
        )
        .expect("GpuBatchSNN::new");

        // Input: rate=200/255 ≈ 78% fire probability per tick.
        let rates = vec![200u8, 200u8];
        snn.upload_input_batch(&rt, &rates).unwrap();
        snn.reset_state(&rt).unwrap();
        snn.reset_output_counts(&rt).unwrap();
        snn.tick_many_with_counting(&rt, 50).unwrap();
        snn.synchronize(&rt).unwrap();

        let counts = snn.get_output_counts(&rt).unwrap();
        assert_eq!(counts.len(), 1);
        let total_spikes: u32 = counts[0].iter().sum();
        assert!(
            total_spikes > 0,
            "expected output spikes with direct topology + rate=200, got {:?}",
            counts[0]
        );
    }

    /// Batch instances are independent: higher rates produce strictly more output spikes.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_batch_isolation() {
        let rt = CudaRuntime::new().expect("CUDA init");
        let n = 8;
        let n_inputs = 2;
        let n_outputs = 2;
        let batch = 4;
        let synapses = make_direct_synapses(n, n_inputs, n_outputs);
        let thresholds = vec![256i16; n];
        let leaks = vec![230u8; n];

        let mut snn = GpuBatchSNN::new(
            &rt,
            n,
            n_inputs,
            n_outputs,
            &thresholds,
            &leaks,
            &synapses,
            batch,
        )
        .expect("GpuBatchSNN::new");

        // Rates: sample b has rate = 50*(b+1) per input: 50, 100, 150, 200
        let rates: Vec<u8> = (0..batch)
            .flat_map(|b| vec![50u8 * (b as u8 + 1); n_inputs])
            .collect();

        snn.upload_input_batch(&rt, &rates).unwrap();
        snn.reset_state(&rt).unwrap();
        snn.reset_output_counts(&rt).unwrap();
        snn.tick_many_with_counting(&rt, 100).unwrap();
        snn.synchronize(&rt).unwrap();

        let counts = snn.get_output_counts(&rt).unwrap();
        assert_eq!(counts.len(), batch);

        let totals: Vec<u32> = counts.iter().map(|c| c.iter().sum()).collect();
        // Highest-rate sample (200) should fire more than lowest-rate sample (50).
        assert!(
            totals[3] >= totals[0],
            "expected more spikes from higher-rate sample: {:?}",
            totals
        );
        // All samples with rate > 0 should produce at least some spikes over 100 ticks.
        for (i, &t) in totals.iter().enumerate() {
            assert!(
                t > 0,
                "sample {} with rate {} should fire, got 0 spikes",
                i,
                50 * (i + 1)
            );
        }
    }

    /// reset_state zeroes all B instances; output counts are zero after reset.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_batch_reset_state() {
        let rt = CudaRuntime::new().expect("CUDA init");
        let n = 8;
        let n_inputs = 2;
        let n_outputs = 2;
        let synapses = make_direct_synapses(n, n_inputs, n_outputs);
        let thresholds = vec![256i16; n];
        let leaks = vec![230u8; n];

        let mut snn = GpuBatchSNN::new(
            &rt,
            n,
            n_inputs,
            n_outputs,
            &thresholds,
            &leaks,
            &synapses,
            2,
        )
        .expect("GpuBatchSNN::new");

        let rates = vec![200u8; 2 * n_inputs];
        snn.upload_input_batch(&rt, &rates).unwrap();

        // Run 50 ticks to build up state
        snn.reset_state(&rt).unwrap();
        snn.tick_many(&rt, 50).unwrap();
        snn.synchronize(&rt).unwrap();

        // After reset, output counts must be zero
        snn.reset_state(&rt).unwrap();
        snn.reset_output_counts(&rt).unwrap();
        snn.synchronize(&rt).unwrap();

        let counts = snn.get_output_counts_flat(&rt).unwrap();
        assert!(
            counts.iter().all(|&c| c == 0),
            "expected all zero after reset, got {:?}",
            counts
        );
    }

    /// Verify dimension getters match construction arguments.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_batch_dimensions() {
        let rt = CudaRuntime::new().expect("CUDA init");
        let n = 16;
        let synapses = make_ring_synapses(n);
        let thresholds = vec![256i16; n];
        let leaks = vec![230u8; n];

        let snn = GpuBatchSNN::new(&rt, n, 4, 2, &thresholds, &leaks, &synapses, 32)
            .expect("GpuBatchSNN::new");

        assert_eq!(snn.n_neurons(), n);
        assert_eq!(snn.n_inputs(), 4);
        assert_eq!(snn.n_outputs(), 2);
        assert_eq!(snn.batch_size(), 32);
    }

    /// Sprint 351: Productized block-sparse API test.
    /// Uses enable_blocksparse() on a real CSR topology, verifies parity with CSR tick().
    #[test]
    #[cfg(feature = "cuda")]
    fn test_snn_s351_productized_blocksparse() {
        let rt = CudaRuntime::new().expect("CUDA runtime");
        let n_inputs = 1568;
        let n_hidden = 320;
        let n_outputs = 10;
        let n_neurons = n_inputs + n_hidden + n_outputs;
        let hidden_start = n_inputs;
        let output_start = n_neurons - n_outputs;
        let batch_size = 128;
        let ticks = 100;

        // Build CSR.
        let mut syn_ptr = vec![0u32; n_neurons + 1];
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        for i in 0..n_neurons {
            if i < n_inputs {
                for h in hidden_start..(hidden_start + n_hidden) {
                    if (i * 7 + h * 13) % 10 == 0 {
                        targets.push(h as u32);
                        weights.push(30i8);
                    }
                }
            } else if i >= hidden_start && i < output_start {
                for o in output_start..n_neurons {
                    targets.push(o as u32);
                    weights.push(50i8);
                }
            }
            syn_ptr[i + 1] = targets.len() as u32;
        }
        let csr = SynapseCSRData {
            syn_ptr,
            targets,
            weights,
        };
        let thresholds = vec![256i16; n_neurons];
        let leaks = vec![230u8; n_neurons];
        let rates: Vec<u8> = (0..batch_size * n_inputs)
            .map(|i| ((i * 37 + 13) % 100) as u8)
            .collect();

        // CSR reference run.
        let mut snn_csr = GpuBatchSNN::new(
            &rt,
            n_neurons,
            n_inputs,
            n_outputs,
            &thresholds,
            &leaks,
            &csr,
            batch_size,
        )
        .unwrap();
        snn_csr.upload_input_batch(&rt, &rates).unwrap();
        snn_csr.tick_many_with_counting(&rt, ticks).unwrap();
        rt.synchronize().unwrap();
        let csr_counts = snn_csr.get_output_counts_flat(&rt).unwrap();

        // Block-sparse run via productized API.
        let mut snn_bs = GpuBatchSNN::new(
            &rt,
            n_neurons,
            n_inputs,
            n_outputs,
            &thresholds,
            &leaks,
            &csr,
            batch_size,
        )
        .unwrap();
        snn_bs.enable_blocksparse(&rt, &csr).unwrap();
        assert!(snn_bs.use_blocksparse, "block-sparse should be enabled");
        snn_bs.upload_input_batch(&rt, &rates).unwrap();
        // tick() should now automatically use block-sparse accumulate.
        snn_bs.tick_many_with_counting(&rt, ticks).unwrap();
        rt.synchronize().unwrap();
        let bs_counts = snn_bs.get_output_counts_flat(&rt).unwrap();

        // Parity.
        let mismatches: usize = csr_counts
            .iter()
            .zip(bs_counts.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches,
            0,
            "S351: enable_blocksparse() output diverged from CSR ({} mismatches / {})",
            mismatches,
            csr_counts.len(),
        );
        eprintln!(
            "\n  S351: Productized block-sparse parity PASS (0/{} mismatches)\n",
            csr_counts.len(),
        );
    }

    /// Sprint 350: Sparse-network scaling matrix.
    /// Tests block-sparse at biologically realistic connectivity (1-5%)
    /// across 10K-100K neurons. Measures throughput and synapse counts.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_snn_s350_sparse_network_scaling() {
        use std::time::Instant;

        let rt = CudaRuntime::new().expect("CUDA runtime");
        let ticks = 100usize;
        let batch_size = 256usize;

        // Scale points: (n_inputs, n_hidden, n_outputs)
        let scales: Vec<(usize, usize, usize, &str)> = vec![
            (2000, 8000, 10, "10K"),
            (2000, 23000, 10, "25K"),
            (2000, 98000, 10, "100K"),
        ];
        let densities: Vec<f32> = vec![0.01, 0.02, 0.05];

        eprintln!("\n=== Sprint 350: Sparse Network Scaling Matrix ===\n");
        eprintln!(
            "  {:<6} {:>6} {:>10} {:>10} {:>10} {:>10}",
            "Scale", "Dens%", "Synapses", "BS pairs", "ms/batch", "Samp/sec"
        );
        eprintln!("  {}", "-".repeat(58));

        for (n_inputs, n_hidden, n_outputs, label) in &scales {
            let n_neurons = n_inputs + n_hidden + n_outputs;
            let hidden_start = *n_inputs;
            let output_start = n_neurons - n_outputs;

            let thresholds = vec![256i16; n_neurons];
            let leaks = vec![230u8; n_neurons];
            let rates: Vec<u8> = (0..batch_size * n_inputs)
                .map(|i| ((i * 37 + 13) % 100) as u8)
                .collect();

            // Dummy CSR for GpuBatchSNN construction.
            let dummy_csr = SynapseCSRData {
                syn_ptr: vec![0u32; n_neurons + 1],
                targets: Vec::new(),
                weights: Vec::new(),
            };

            for &density in &densities {
                // Build CSR with target density.
                let mut syn_ptr = vec![0u32; n_neurons + 1];
                let mut targets = Vec::new();
                let mut weights = Vec::new();
                let inv = (1.0 / density).round() as usize;

                for i in 0..n_neurons {
                    if i < *n_inputs {
                        for h in hidden_start..(hidden_start + n_hidden) {
                            if (i * 7 + h * 13) % inv == 0 {
                                targets.push(h as u32);
                                weights.push(30i8);
                            }
                        }
                    } else if i >= hidden_start && i < output_start {
                        // Hidden→output: full connectivity (small output layer).
                        for o in output_start..n_neurons {
                            targets.push(o as u32);
                            weights.push(50i8);
                        }
                    }
                    syn_ptr[i + 1] = targets.len() as u32;
                }
                let csr = SynapseCSRData {
                    syn_ptr,
                    targets,
                    weights,
                };

                // Convert to block-sparse.
                let (bs_src, bs_dst, bs_masks, bs_woff, bs_weights) =
                    csr_to_blocksparse(n_neurons, &csr);
                let n_bp = bs_src.len() as u32;

                // Try to build + run. Skip on OOM.
                let snn_result = GpuBatchSNN::new(
                    &rt,
                    n_neurons,
                    *n_inputs,
                    *n_outputs,
                    &thresholds,
                    &leaks,
                    &dummy_csr,
                    batch_size,
                );
                let mut snn = match snn_result {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!(
                            "  {:<6} {:>5}%  (build failed: {})",
                            label,
                            (density * 100.0) as u32,
                            e
                        );
                        continue;
                    }
                };

                let d_src = rt.upload(&bs_src).unwrap();
                let d_dst = rt.upload(&bs_dst).unwrap();
                let d_masks = rt.upload(&bs_masks).unwrap();
                let d_woff = rt.upload(&bs_woff).unwrap();
                let d_pw = rt.upload(&bs_weights).unwrap();

                snn.upload_input_batch(&rt, &rates).unwrap();

                // Warmup.
                snn.tick_many_blocksparse_weighted_with_counting(
                    &rt, ticks, &d_src, &d_dst, &d_masks, &d_woff, &d_pw, n_bp,
                )
                .unwrap();
                rt.synchronize().unwrap();

                // Measure.
                let n_batches = 3usize;
                let start = Instant::now();
                for _ in 0..n_batches {
                    snn.reset_state(&rt).unwrap();
                    snn.reset_output_counts(&rt).unwrap();
                    snn.upload_input_batch(&rt, &rates).unwrap();
                    snn.tick_many_blocksparse_weighted_with_counting(
                        &rt, ticks, &d_src, &d_dst, &d_masks, &d_woff, &d_pw, n_bp,
                    )
                    .unwrap();
                }
                rt.synchronize().unwrap();
                let elapsed = start.elapsed();

                let total_samples = n_batches * batch_size;
                let ms_per_batch = elapsed.as_secs_f64() * 1000.0 / n_batches as f64;
                let samp_per_sec = total_samples as f64 / elapsed.as_secs_f64();

                eprintln!(
                    "  {:<6} {:>5}% {:>10} {:>10} {:>9.1} {:>9.0}",
                    label,
                    (density * 100.0) as u32,
                    csr.n_synapses(),
                    n_bp,
                    ms_per_batch,
                    samp_per_sec,
                );
            }
        }
        eprintln!("\n=== End Sprint 350 ===\n");
    }

    /// Sprint 349: Convert CSR to weighted block-sparse (same edges, same weights).
    #[cfg(feature = "cuda")]
    fn csr_to_blocksparse(
        n_neurons: usize,
        csr: &SynapseCSRData,
    ) -> (Vec<u16>, Vec<u16>, Vec<u64>, Vec<u32>, Vec<i8>) {
        use std::collections::BTreeMap;

        let block_size = 8usize;
        // Group edges by (src_block, dst_block).
        let mut block_map: BTreeMap<(u16, u16), (u64, Vec<(u8, i8)>)> = BTreeMap::new();

        for src in 0..n_neurons {
            let start = csr.syn_ptr[src] as usize;
            let end = csr.syn_ptr[src + 1] as usize;
            let sb = (src / block_size) as u16;
            let si = (src % block_size) as u8;

            for j in start..end {
                let dst = csr.targets[j] as usize;
                let w = csr.weights[j];
                let db = (dst / block_size) as u16;
                let di = (dst % block_size) as u8;
                let bit = (si as u32) * 8 + (di as u32);

                let entry = block_map
                    .entry((sb, db))
                    .or_insert_with(|| (0u64, Vec::new()));
                entry.0 |= 1u64 << bit;
                entry.1.push((bit as u8, w));
            }
        }

        let n_pairs = block_map.len();
        let mut src_blocks = Vec::with_capacity(n_pairs);
        let mut dst_blocks = Vec::with_capacity(n_pairs);
        let mut masks = Vec::with_capacity(n_pairs);
        let mut weight_offsets = Vec::with_capacity(n_pairs + 1);
        let mut packed_weights: Vec<i8> = Vec::new();

        for ((sb, db), (mask, mut bits)) in block_map {
            src_blocks.push(sb);
            dst_blocks.push(db);
            masks.push(mask);
            weight_offsets.push(packed_weights.len() as u32);
            // Sort by bit position to match kernel iteration order.
            bits.sort_by_key(|(b, _)| *b);
            for (_, w) in bits {
                packed_weights.push(w);
            }
        }
        weight_offsets.push(packed_weights.len() as u32);

        (
            src_blocks,
            dst_blocks,
            masks,
            weight_offsets,
            packed_weights,
        )
    }

    /// Sprint 349: Same-topology weighted block-sparse parity + A/B.
    /// Converts CSR to block-sparse (preserving all edges and weights),
    /// verifies output parity, then measures performance.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_snn_s349_same_topology_parity() {
        use std::time::Instant;

        let rt = CudaRuntime::new().expect("CUDA runtime");
        let ticks = 100usize;
        let batch_size = 256usize;

        let scales: Vec<(usize, usize, usize, &str)> =
            vec![(1568, 320, 10, "2K"), (2000, 8000, 10, "10K")];

        eprintln!("\n=== Sprint 349: Same-Topology Parity + A/B ===\n");

        for (n_inputs, n_hidden, n_outputs, label) in &scales {
            let n_neurons = n_inputs + n_hidden + n_outputs;
            let hidden_start = *n_inputs;
            let output_start = n_neurons - n_outputs;

            // Build CSR.
            let mut csr_syn_ptr = vec![0u32; n_neurons + 1];
            let mut csr_targets = Vec::new();
            let mut csr_weights = Vec::new();
            for i in 0..n_neurons {
                if i < *n_inputs {
                    for h in hidden_start..(hidden_start + n_hidden) {
                        if (i * 7 + h * 13) % 10 == 0 {
                            csr_targets.push(h as u32);
                            csr_weights.push(30i8);
                        }
                    }
                } else if i >= hidden_start && i < output_start {
                    for o in output_start..n_neurons {
                        csr_targets.push(o as u32);
                        csr_weights.push(50i8);
                    }
                }
                csr_syn_ptr[i + 1] = csr_targets.len() as u32;
            }
            let csr = SynapseCSRData {
                syn_ptr: csr_syn_ptr,
                targets: csr_targets,
                weights: csr_weights,
            };

            // Convert CSR → block-sparse (same edges, same weights).
            let (bs_src, bs_dst, bs_masks, bs_woff, bs_weights) =
                csr_to_blocksparse(n_neurons, &csr);
            let n_bp = bs_src.len() as u32;
            let bs_nnz: u32 = bs_woff.last().copied().unwrap_or(0);

            eprintln!(
                "  {} ({}): CSR {} syn, BS {} pairs / {} syn",
                label,
                n_neurons,
                csr.n_synapses(),
                n_bp,
                bs_nnz,
            );

            // Verify synapse count matches.
            assert_eq!(
                csr.n_synapses(),
                bs_nnz as usize,
                "synapse count mismatch: CSR {} vs BS {}",
                csr.n_synapses(),
                bs_nnz,
            );

            let thresholds = vec![256i16; n_neurons];
            let leaks = vec![230u8; n_neurons];
            let rates: Vec<u8> = (0..batch_size * n_inputs)
                .map(|i| ((i * 37 + 13) % 100) as u8)
                .collect();

            // --- CSR run ---
            let mut snn_csr = GpuBatchSNN::new(
                &rt,
                n_neurons,
                *n_inputs,
                *n_outputs,
                &thresholds,
                &leaks,
                &csr,
                batch_size,
            )
            .unwrap();
            snn_csr.upload_input_batch(&rt, &rates).unwrap();
            snn_csr.tick_many_with_counting(&rt, ticks).unwrap();
            rt.synchronize().unwrap();
            let csr_counts = snn_csr.get_output_counts_flat(&rt).unwrap();

            // --- Block-sparse run (same topology, same weights) ---
            let dummy_csr = SynapseCSRData {
                syn_ptr: vec![0u32; n_neurons + 1],
                targets: Vec::new(),
                weights: Vec::new(),
            };
            let mut snn_bs = GpuBatchSNN::new(
                &rt,
                n_neurons,
                *n_inputs,
                *n_outputs,
                &thresholds,
                &leaks,
                &dummy_csr,
                batch_size,
            )
            .unwrap();
            let d_src = rt.upload(&bs_src).unwrap();
            let d_dst = rt.upload(&bs_dst).unwrap();
            let d_masks = rt.upload(&bs_masks).unwrap();
            let d_woff = rt.upload(&bs_woff).unwrap();
            let d_pw = rt.upload(&bs_weights).unwrap();
            snn_bs.upload_input_batch(&rt, &rates).unwrap();
            snn_bs
                .tick_many_blocksparse_weighted_with_counting(
                    &rt, ticks, &d_src, &d_dst, &d_masks, &d_woff, &d_pw, n_bp,
                )
                .unwrap();
            rt.synchronize().unwrap();
            let bs_counts = snn_bs.get_output_counts_flat(&rt).unwrap();

            // --- Parity check ---
            let mut mismatches = 0usize;
            for (i, (c, b)) in csr_counts.iter().zip(bs_counts.iter()).enumerate() {
                if c != b {
                    mismatches += 1;
                    if mismatches <= 5 {
                        eprintln!("    MISMATCH at output[{}]: CSR={}, BS={}", i, c, b,);
                    }
                }
            }
            let parity = if mismatches == 0 { "PASS" } else { "FAIL" };
            eprintln!(
                "    Output parity: {} ({} mismatches / {})",
                parity,
                mismatches,
                csr_counts.len()
            );
            assert_eq!(
                mismatches, 0,
                "{}: block-sparse output diverged from CSR ({} mismatches)",
                label, mismatches
            );

            // --- A/B timing ---
            {
                // CSR timing.
                snn_csr.reset_state(&rt).unwrap();
                snn_csr.reset_output_counts(&rt).unwrap();
                snn_csr.upload_input_batch(&rt, &rates).unwrap();
                snn_csr.tick_many_with_counting(&rt, ticks).unwrap();
                rt.synchronize().unwrap();
                let csr_start = Instant::now();
                for _ in 0..3 {
                    snn_csr.reset_state(&rt).unwrap();
                    snn_csr.reset_output_counts(&rt).unwrap();
                    snn_csr.upload_input_batch(&rt, &rates).unwrap();
                    snn_csr.tick_many_with_counting(&rt, ticks).unwrap();
                }
                rt.synchronize().unwrap();
                let csr_ms = csr_start.elapsed().as_secs_f64() * 1000.0 / 3.0;

                // BS timing.
                snn_bs.reset_state(&rt).unwrap();
                snn_bs.reset_output_counts(&rt).unwrap();
                snn_bs.upload_input_batch(&rt, &rates).unwrap();
                snn_bs
                    .tick_many_blocksparse_weighted_with_counting(
                        &rt, ticks, &d_src, &d_dst, &d_masks, &d_woff, &d_pw, n_bp,
                    )
                    .unwrap();
                rt.synchronize().unwrap();
                let bs_start = Instant::now();
                for _ in 0..3 {
                    snn_bs.reset_state(&rt).unwrap();
                    snn_bs.reset_output_counts(&rt).unwrap();
                    snn_bs.upload_input_batch(&rt, &rates).unwrap();
                    snn_bs
                        .tick_many_blocksparse_weighted_with_counting(
                            &rt, ticks, &d_src, &d_dst, &d_masks, &d_woff, &d_pw, n_bp,
                        )
                        .unwrap();
                }
                rt.synchronize().unwrap();
                let bs_ms = bs_start.elapsed().as_secs_f64() * 1000.0 / 3.0;

                eprintln!(
                    "    CSR: {:.1} ms, BS: {:.1} ms, speedup: {:.2}x",
                    csr_ms,
                    bs_ms,
                    csr_ms / bs_ms,
                );
            }
            eprintln!();
        }
        eprintln!("=== End Sprint 349 ===\n");
    }

    /// Sprint 348: CSR vs Block-Sparse A/B benchmark at 2K/10K/25K neurons.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_snn_s348_csr_vs_blocksparse() {
        use super::super::block_sparse_synapses::{BlockSparseSynapseMap, SparsityPattern};
        use std::time::Instant;

        let rt = CudaRuntime::new().expect("CUDA runtime");
        let ticks = 100usize;
        let n_batches = 3usize;
        let batch_size = 256usize;
        let fixed_weight = 60i32; // 30 * 2

        let scales: Vec<(usize, usize, usize, &str)> = vec![
            (1568, 320, 10, "2K"),
            (2000, 8000, 10, "10K"),
            (2000, 23000, 10, "25K"),
        ];

        eprintln!("\n=== Sprint 348: CSR vs Block-Sparse A/B ===\n");
        eprintln!(
            "  {:<6} {:>8} {:>8} {:>10} {:>10} {:>8}",
            "Scale", "CSR syn", "BS syn", "CSR ms", "BS ms", "Speedup"
        );
        eprintln!("  {}", "-".repeat(56));

        for (n_inputs, n_hidden, n_outputs, label) in &scales {
            let n_neurons = n_inputs + n_hidden + n_outputs;
            let hidden_start = *n_inputs;
            let output_start = n_neurons - n_outputs;

            // Build CSR synapses (10% input→hidden + full hidden→output).
            let mut csr_syn_ptr = vec![0u32; n_neurons + 1];
            let mut csr_targets = Vec::new();
            let mut csr_weights = Vec::new();
            for i in 0..n_neurons {
                if i < *n_inputs {
                    for h in hidden_start..(hidden_start + n_hidden) {
                        if (i * 7 + h * 13) % 10 == 0 {
                            csr_targets.push(h as u32);
                            csr_weights.push(30i8);
                        }
                    }
                } else if i >= hidden_start && i < output_start {
                    for o in output_start..n_neurons {
                        csr_targets.push(o as u32);
                        csr_weights.push(50i8);
                    }
                }
                csr_syn_ptr[i + 1] = csr_targets.len() as u32;
            }
            let csr_synapses = SynapseCSRData {
                syn_ptr: csr_syn_ptr,
                targets: csr_targets,
                weights: csr_weights,
            };
            let csr_nnz = csr_synapses.n_synapses();

            // Build equivalent block-sparse (~10% density).
            let bs_map = BlockSparseSynapseMap::from_layers(
                n_neurons,
                &[
                    (0, *n_inputs),
                    (hidden_start, output_start),
                    (output_start, n_neurons),
                ],
                0.10,
                SparsityPattern::Custom(25),
                42,
            );
            let bs_gpu = bs_map.gpu_data();
            let bs_nnz = bs_map.n_synapses;

            let thresholds = vec![256i16; n_neurons];
            let leaks = vec![230u8; n_neurons];
            let rates: Vec<u8> = (0..batch_size * n_inputs)
                .map(|i| ((i * 37 + 13) % 100) as u8)
                .collect();

            // --- CSR benchmark ---
            let mut snn_csr = GpuBatchSNN::new(
                &rt,
                n_neurons,
                *n_inputs,
                *n_outputs,
                &thresholds,
                &leaks,
                &csr_synapses,
                batch_size,
            )
            .expect("CSR build");
            snn_csr.upload_input_batch(&rt, &rates).unwrap();
            snn_csr.tick_many(&rt, ticks).unwrap();
            rt.synchronize().unwrap();

            let csr_start = Instant::now();
            for _ in 0..n_batches {
                snn_csr.reset_state(&rt).unwrap();
                snn_csr.reset_output_counts(&rt).unwrap();
                snn_csr.upload_input_batch(&rt, &rates).unwrap();
                snn_csr.tick_many_with_counting(&rt, ticks).unwrap();
            }
            rt.synchronize().unwrap();
            let csr_ms = csr_start.elapsed().as_secs_f64() * 1000.0 / n_batches as f64;

            // --- Block-sparse benchmark ---
            // Reuse same SNN for neuron state; swap accumulate path.
            let d_src = rt.upload(&bs_gpu.src_blocks).unwrap();
            let d_dst = rt.upload(&bs_gpu.dst_blocks).unwrap();
            let d_masks = rt.upload(&bs_gpu.synapse_masks).unwrap();
            let n_bp = bs_gpu.n_block_pairs;

            // Build a fresh SNN (needs dummy CSR for construction).
            let dummy_csr = SynapseCSRData {
                syn_ptr: vec![0u32; n_neurons + 1],
                targets: Vec::new(),
                weights: Vec::new(),
            };
            let mut snn_bs = GpuBatchSNN::new(
                &rt,
                n_neurons,
                *n_inputs,
                *n_outputs,
                &thresholds,
                &leaks,
                &dummy_csr,
                batch_size,
            )
            .expect("BS build");
            snn_bs.upload_input_batch(&rt, &rates).unwrap();
            snn_bs
                .tick_many_blocksparse_with_counting(
                    &rt,
                    ticks,
                    &d_src,
                    &d_dst,
                    &d_masks,
                    n_bp,
                    fixed_weight,
                )
                .unwrap();
            rt.synchronize().unwrap();

            let bs_start = Instant::now();
            for _ in 0..n_batches {
                snn_bs.reset_state(&rt).unwrap();
                snn_bs.reset_output_counts(&rt).unwrap();
                snn_bs.upload_input_batch(&rt, &rates).unwrap();
                snn_bs
                    .tick_many_blocksparse_with_counting(
                        &rt,
                        ticks,
                        &d_src,
                        &d_dst,
                        &d_masks,
                        n_bp,
                        fixed_weight,
                    )
                    .unwrap();
            }
            rt.synchronize().unwrap();
            let bs_ms = bs_start.elapsed().as_secs_f64() * 1000.0 / n_batches as f64;

            let speedup = csr_ms / bs_ms;
            eprintln!(
                "  {:<6} {:>8} {:>8} {:>9.1} {:>9.1} {:>7.2}x",
                label, csr_nnz, bs_nnz, csr_ms, bs_ms, speedup,
            );
        }
        eprintln!("\n=== End Sprint 348 ===\n");
    }

    /// Sprint 347: SNN GPU scaling matrix — split timing + neuron/batch sweep.
    /// Separates upload, reset, tick, and sync costs. Tests neuron scaling.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_snn_s347_gpu_scaling_matrix() {
        use std::time::Instant;

        let rt = CudaRuntime::new().expect("CUDA runtime");
        let ticks_per_sample = 100usize;
        let n_batches = 3usize; // measure 3 batches, take average

        // Network scale points: (n_inputs, n_hidden, n_outputs, label)
        let scales: Vec<(usize, usize, usize, &str)> = vec![
            (1568, 320, 10, "2K (MNIST)"),
            (2000, 8000, 10, "10K"),
            (2000, 23000, 10, "25K"),
        ];
        let batch_sizes: Vec<usize> = vec![32, 128, 256, 512, 1024];

        eprintln!("\n=== Sprint 347: SNN GPU Scaling Matrix ===\n");

        for (n_inputs, n_hidden, n_outputs, label) in &scales {
            let n_neurons = n_inputs + n_hidden + n_outputs;
            let hidden_start = *n_inputs;
            let output_start = n_neurons - n_outputs;

            // Build feedforward synapses: inputs→hidden (10% sparse) + hidden→outputs (full).
            let mut syn_ptr = vec![0u32; n_neurons + 1];
            let mut targets = Vec::new();
            let mut weights = Vec::new();
            for i in 0..n_neurons {
                if i < *n_inputs {
                    for h in hidden_start..(hidden_start + n_hidden) {
                        if (i * 7 + h * 13) % 10 == 0 {
                            targets.push(h as u32);
                            weights.push(30i8);
                        }
                    }
                } else if i >= hidden_start && i < output_start {
                    for o in output_start..n_neurons {
                        targets.push(o as u32);
                        weights.push(50i8);
                    }
                }
                syn_ptr[i + 1] = targets.len() as u32;
            }
            let synapses = SynapseCSRData {
                syn_ptr,
                targets,
                weights,
            };
            let thresholds = vec![256i16; n_neurons];
            let leaks = vec![230u8; n_neurons];

            eprintln!(
                "  --- {} neurons ({}) | {} synapses ---",
                n_neurons,
                label,
                synapses.n_synapses()
            );
            eprintln!(
                "  {:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
                "Batch", "Upload", "Reset", "Tick", "Sync", "Total", "Samp/sec"
            );
            eprintln!("  {}", "-".repeat(68));

            for &batch_size in &batch_sizes {
                // Skip if likely OOM (rough estimate: 10 bytes/neuron × batch).
                let est_mb = (batch_size * n_neurons * 10) / (1024 * 1024);
                if est_mb > 4000 {
                    eprintln!(
                        "  {:<8} (skipped — est {} MB GPU memory)",
                        batch_size, est_mb
                    );
                    continue;
                }

                let snn_result = GpuBatchSNN::new(
                    &rt,
                    n_neurons,
                    *n_inputs,
                    *n_outputs,
                    &thresholds,
                    &leaks,
                    &synapses,
                    batch_size,
                );
                let mut snn = match snn_result {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("  {:<8} (build failed: {})", batch_size, e);
                        continue;
                    }
                };

                let rates: Vec<u8> = (0..batch_size * n_inputs)
                    .map(|i| ((i * 37 + 13) % 100) as u8)
                    .collect();

                // Warmup.
                snn.upload_input_batch(&rt, &rates).unwrap();
                snn.tick_many(&rt, ticks_per_sample).unwrap();
                rt.synchronize().unwrap();

                // Measure with split timing.
                let mut upload_ns = 0u64;
                let mut reset_ns = 0u64;
                let mut tick_ns = 0u64;
                let mut sync_ns = 0u64;

                for _ in 0..n_batches {
                    let t0 = Instant::now();
                    snn.reset_state(&rt).unwrap();
                    snn.reset_output_counts(&rt).unwrap();
                    rt.synchronize().unwrap();
                    reset_ns += t0.elapsed().as_nanos() as u64;

                    let t1 = Instant::now();
                    snn.upload_input_batch(&rt, &rates).unwrap();
                    rt.synchronize().unwrap();
                    upload_ns += t1.elapsed().as_nanos() as u64;

                    let t2 = Instant::now();
                    snn.tick_many_with_counting(&rt, ticks_per_sample).unwrap();
                    tick_ns += t2.elapsed().as_nanos() as u64;

                    let t3 = Instant::now();
                    rt.synchronize().unwrap();
                    sync_ns += t3.elapsed().as_nanos() as u64;
                }

                let total_ns = upload_ns + reset_ns + tick_ns + sync_ns;
                let total_samples = n_batches * batch_size;
                let samp_per_sec = total_samples as f64 / (total_ns as f64 / 1e9);

                let to_us = |ns: u64| ns as f64 / 1000.0 / n_batches as f64;
                eprintln!(
                    "  {:<8} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>9.0}",
                    batch_size,
                    to_us(upload_ns),
                    to_us(reset_ns),
                    to_us(tick_ns),
                    to_us(sync_ns),
                    to_us(total_ns),
                    samp_per_sec,
                );
            }
            eprintln!();
        }
        eprintln!("=== End Sprint 347 ===\n");
    }

    /// Sprint 346: SNN GPU measurement — throughput, kernel launches, batch scaling.
    /// Profiles inference at different batch sizes and network scales.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_snn_s346_gpu_measurement() {
        use std::time::Instant;

        let rt = CudaRuntime::new().expect("CUDA runtime");

        // MNIST-scale feedforward network: 1568 inputs → 320 hidden → 10 outputs.
        let n_inputs = 1568;
        let n_hidden = 320;
        let n_outputs = 10;
        let n_neurons = n_inputs + n_hidden + n_outputs;

        // Feedforward synapses: inputs→hidden (50% connectivity) + hidden→outputs (full).
        let output_start = n_neurons - n_outputs;
        let hidden_start = n_inputs;
        let mut syn_ptr = vec![0u32; n_neurons + 1];
        let mut targets = Vec::new();
        let mut weights = Vec::new();

        for i in 0..n_neurons {
            if i < n_inputs {
                // Input → hidden (50% sparse).
                for h in hidden_start..(hidden_start + n_hidden) {
                    if (i * 7 + h * 13) % 2 == 0 {
                        targets.push(h as u32);
                        weights.push(30i8);
                    }
                }
            } else if i >= hidden_start && i < output_start {
                // Hidden → output (full connectivity).
                for o in output_start..n_neurons {
                    targets.push(o as u32);
                    weights.push(50i8);
                }
            }
            syn_ptr[i + 1] = targets.len() as u32;
        }
        let synapses = SynapseCSRData {
            syn_ptr,
            targets,
            weights,
        };

        let thresholds = vec![256i16; n_neurons];
        let leaks = vec![230u8; n_neurons];
        let ticks_per_sample = 100usize;

        eprintln!("\n=== Sprint 346: SNN GPU Measurement ===\n");
        eprintln!(
            "  Network: {} inputs, {} hidden, {} outputs ({} total)",
            n_inputs, n_hidden, n_outputs, n_neurons
        );
        eprintln!("  Synapses: {}", synapses.n_synapses());
        eprintln!("  Ticks/sample: {}", ticks_per_sample);
        eprintln!("  Kernels/tick: 2 (accumulate + LIF step)");
        eprintln!();
        eprintln!(
            "  {:<10} {:>8} {:>10} {:>10} {:>10} {:>10}",
            "Batch", "Samples", "Wall ms", "Samp/sec", "Kern/sec", "ns/kern"
        );
        eprintln!("  {}", "-".repeat(62));

        for batch_size in [32, 64, 128, 256, 512] {
            let mut snn = GpuBatchSNN::new(
                &rt,
                n_neurons,
                n_inputs,
                n_outputs,
                &thresholds,
                &leaks,
                &synapses,
                batch_size,
            )
            .expect("GpuBatchSnn::new");

            // Upload random-ish input rates.
            let rates: Vec<u8> = (0..batch_size * n_inputs)
                .map(|i| ((i * 37 + 13) % 100) as u8)
                .collect();
            snn.upload_input_batch(&rt, &rates).unwrap();

            // Warmup: 1 batch.
            snn.tick_many(&rt, ticks_per_sample).unwrap();
            snn.synchronize(&rt).unwrap();
            snn.reset_state(&rt).unwrap();
            snn.upload_input_batch(&rt, &rates).unwrap();

            // Measure: 5 batches.
            let n_batches = 5usize;
            let start = Instant::now();
            for _ in 0..n_batches {
                snn.reset_state(&rt).unwrap();
                snn.upload_input_batch(&rt, &rates).unwrap();
                snn.tick_many_with_counting(&rt, ticks_per_sample).unwrap();
            }
            snn.synchronize(&rt).unwrap();
            let elapsed = start.elapsed();

            let total_samples = n_batches * batch_size;
            let total_kernels = n_batches * ticks_per_sample * 3; // accum + lif + count_outputs
            let samp_per_sec = total_samples as f64 / elapsed.as_secs_f64();
            let kern_per_sec = total_kernels as f64 / elapsed.as_secs_f64();
            let ns_per_kern = elapsed.as_nanos() as f64 / total_kernels as f64;

            eprintln!(
                "  {:<10} {:>8} {:>9.1} {:>9.0} {:>9.0} {:>9.0}",
                batch_size,
                total_samples,
                elapsed.as_secs_f64() * 1000.0,
                samp_per_sec,
                kern_per_sec,
                ns_per_kern,
            );
        }

        eprintln!("\n=== End Sprint 346 ===\n");
    }

    /// Sprint 353: Block-sparse E-prop training parity.
    /// Trains the same network with CSR-only and block-sparse+sync paths,
    /// verifies that final i8 weights match exactly after N batches.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_snn_s353_blocksparse_eprop_training_parity() {
        let rt = CudaRuntime::new().expect("CUDA runtime");
        let n_inputs = 200;
        let n_hidden = 100;
        let n_outputs = 10;
        let n_neurons = n_inputs + n_hidden + n_outputs;
        let hidden_start = n_inputs;
        let output_start = n_neurons - n_outputs;
        let batch_size = 32usize;
        let ticks = 50;
        let n_batches = 5;
        let lr = 0.01f32;

        // Build sparse CSR (~10% IH connectivity + full HR).
        let mut syn_ptr = vec![0u32; n_neurons + 1];
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        for i in 0..n_neurons {
            if i < n_inputs {
                for h in hidden_start..(hidden_start + n_hidden) {
                    if (i * 7 + h * 13) % 10 == 0 {
                        targets.push(h as u32);
                        weights.push(((i * 3 + h * 5) % 60 - 30) as i8);
                    }
                }
            } else if i >= hidden_start && i < output_start {
                for o in output_start..n_neurons {
                    targets.push(o as u32);
                    weights.push(((i * 11 + o * 7) % 80 - 40) as i8);
                }
            }
            syn_ptr[i + 1] = targets.len() as u32;
        }
        let csr = SynapseCSRData {
            syn_ptr,
            targets,
            weights,
        };
        let thresholds = vec![256i16; n_neurons];
        let leaks = vec![230u8; n_neurons];

        // Deterministic input rates (same for both runs).
        let rates: Vec<u8> = (0..batch_size * n_inputs)
            .map(|i| ((i * 37 + 13) % 80) as u8)
            .collect();
        // Deterministic learning signals.
        let signals: Vec<f32> = (0..batch_size)
            .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
            .collect();

        // ── Run A: CSR-only E-prop training ──
        let mut snn_csr = GpuBatchSNN::new(
            &rt,
            n_neurons,
            n_inputs,
            n_outputs,
            &thresholds,
            &leaks,
            &csr,
            batch_size,
        )
        .unwrap();
        snn_csr.enable_eprop(&rt, 0.95, 0.95, 0.3).unwrap();
        snn_csr.init_weight_shadow(&rt).unwrap();

        for _batch in 0..n_batches {
            snn_csr.upload_input_batch(&rt, &rates).unwrap();
            snn_csr.reset_state(&rt).unwrap();
            snn_csr.reset_eprop_state(&rt).unwrap();
            snn_csr.tick_many_with_eprop(&rt, ticks).unwrap();
            snn_csr.apply_eprop_update(&rt, &signals, lr).unwrap();
            snn_csr.project_weights_f32(&rt).unwrap();
        }
        rt.synchronize().unwrap();
        let csr_weights = rt.download(&snn_csr.d_weights).unwrap();

        // ── Run B: Block-sparse E-prop training ──
        let mut snn_bs = GpuBatchSNN::new(
            &rt,
            n_neurons,
            n_inputs,
            n_outputs,
            &thresholds,
            &leaks,
            &csr,
            batch_size,
        )
        .unwrap();
        snn_bs.enable_blocksparse(&rt, &csr).unwrap();
        snn_bs.enable_eprop(&rt, 0.95, 0.95, 0.3).unwrap();
        snn_bs.init_weight_shadow(&rt).unwrap();

        for _batch in 0..n_batches {
            snn_bs.upload_input_batch(&rt, &rates).unwrap();
            snn_bs.reset_state(&rt).unwrap();
            snn_bs.reset_eprop_state(&rt).unwrap();
            // tick_many_with_eprop → self.tick() → block-sparse accumulate.
            snn_bs.tick_many_with_eprop(&rt, ticks).unwrap();
            snn_bs.apply_eprop_update(&rt, &signals, lr).unwrap();
            // project_weights_f32 auto-syncs block-sparse weights (Sprint 353).
            snn_bs.project_weights_f32(&rt).unwrap();
        }
        rt.synchronize().unwrap();
        let bs_weights = rt.download(&snn_bs.d_weights).unwrap();

        // Compare final CSR weights (both paths update d_weights the same way).
        let weight_mismatches: usize = csr_weights
            .iter()
            .zip(bs_weights.iter())
            .filter(|(a, b)| a != b)
            .count();

        eprintln!("\n=== Sprint 353: Block-Sparse E-prop Training Parity ===");
        eprintln!("  Network: {}→{}→{}", n_inputs, n_hidden, n_outputs);
        eprintln!("  Synapses: {}", csr.weights.len());
        eprintln!("  Batches: {}, ticks: {}, lr: {}", n_batches, ticks, lr);
        eprintln!(
            "  Weight mismatches: {} / {}",
            weight_mismatches,
            csr_weights.len()
        );

        // Weights should be IDENTICAL since both paths use the same E-prop
        // kernels (CSR-indexed), same CSR d_weights buffer, same learning
        // signals. The only difference is the accumulate kernel (CSR vs block-sparse).
        assert_eq!(
            weight_mismatches,
            0,
            "S353: block-sparse training diverged from CSR ({} / {} mismatches)",
            weight_mismatches,
            csr_weights.len(),
        );
        eprintln!("  Result: PASS (0 weight mismatches)");
        eprintln!("=== End Sprint 353 ===\n");
    }
}
