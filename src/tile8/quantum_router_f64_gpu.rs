//! CUDA-Accelerated F64 Quantum Router
//!
//! This module extends quantum_router_f64.rs with GPU acceleration using CUDA.
//! Provides 10-50x speedup for dense quantum operations on supported GPUs.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    QuantumGridF64Gpu                            │
//! │  ┌─────────────────────┐  ┌─────────────────────┐               │
//! │  │ GPU real[2^(n-7)*128]│  │ GPU imag[2^(n-7)*128]│             │
//! │  │ (f64 in VRAM)        │  │ (f64 in VRAM)        │             │
//! │  └─────────────────────┘  └─────────────────────┘               │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use engine::tile8::quantum_router_f64_gpu::QuantumGridF64Gpu;
//!
//! let rt = CudaRuntime::new()?;
//! let mut grid = QuantumGridF64Gpu::new(&rt, 25)?; // 25 qubits on GPU
//! let time_us = grid.create_ghz_state()?;
//! let verification = grid.verify_ghz_fidelity()?;
//! ```

#[cfg(feature = "cuda")]
use logic_fabric_core::cuda::{CudaError, CudaResult, CudaRuntime};

#[cfg(feature = "cuda")]
use cudarc::driver::{
    CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};

#[cfg(feature = "cuda")]
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, result as cublas_result, sys::cublasOperation_t};

#[cfg(feature = "cuda")]
use half::f16;

#[cfg(feature = "cuda")]
use std::os::raw::c_int;

#[cfg(feature = "cuda")]
use std::sync::Arc;

/// CUDA C source for f64 quantum kernels
///
/// Uses compile_ptx_with_opts for runtime compilation, ensuring compatibility
/// with any GPU architecture including the latest RTX 5090 (Blackwell).
#[cfg(feature = "cuda")]
const QUANTUM_F64_CUDA: &str = r#"
#include <cuda_fp16.h>
#define RSQRT2 0.7071067811865475244

// Hadamard kernel for f64 amplitudes on qubit 0
// Each thread processes one amplitude pair (i, i+1)
extern "C" __global__ void hadamard_f64_q0(
    double* real,
    double* imag,
    unsigned long long num_pairs
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;

    unsigned long long i = tid * 2;
    unsigned long long j = i + 1;

    double r0 = real[i];
    double r1 = real[j];
    double im0 = imag[i];
    double im1 = imag[j];

    real[i] = (r0 + r1) * RSQRT2;
    real[j] = (r0 - r1) * RSQRT2;
    imag[i] = (im0 + im1) * RSQRT2;
    imag[j] = (im0 - im1) * RSQRT2;
}

// Hadamard kernel for arbitrary local qubit (0-6)
extern "C" __global__ void hadamard_f64_local(
    double* real,
    double* imag,
    unsigned long long num_pairs,
    unsigned int qubit_mask
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;

    unsigned long long mask = qubit_mask;
    unsigned long long lower = tid & (mask - 1);
    unsigned long long upper = tid & ~(mask - 1);
    unsigned long long idx0 = lower | (upper << 1);
    unsigned long long idx1 = idx0 | mask;

    double r0 = real[idx0];
    double r1 = real[idx1];
    double im0 = imag[idx0];
    double im1 = imag[idx1];

    real[idx0] = (r0 + r1) * RSQRT2;
    real[idx1] = (r0 - r1) * RSQRT2;
    imag[idx0] = (im0 + im1) * RSQRT2;
    imag[idx1] = (im0 - im1) * RSQRT2;
}

// CNOT kernel for local qubits (control=0, target=1-6)
extern "C" __global__ void cnot_f64_local(
    double* real,
    double* imag,
    unsigned long long num_amps,
    unsigned int target_mask
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_amps) return;

    // Only process if control bit (bit 0) is set AND target bit is 0
    unsigned int idx_low = (unsigned int)tid;
    if ((idx_low & 1) == 0) return;  // control=0, skip
    if ((idx_low & target_mask) != 0) return;  // target=1, skip

    unsigned long long partner = tid ^ target_mask;

    double tmp_r = real[tid];
    double tmp_i = imag[tid];
    real[tid] = real[partner];
    imag[tid] = imag[partner];
    real[partner] = tmp_r;
    imag[partner] = tmp_i;
}

// Cross-block CNOT: swaps amplitudes between blocks (ORIGINAL - kept for reference)
// Block size = 128 = 2^7
// PROBLEM: Only 25% of threads do useful work (75% waste)
extern "C" __global__ void cnot_f64_cross_block(
    double* real,
    double* imag,
    unsigned long long total_amps,
    unsigned long long block_mask
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_amps) return;

    unsigned long long block_id = tid >> 7;     // block_id = tid / 128
    unsigned long long local_addr = tid & 127;  // local_addr = tid % 128

    // Skip if block bit is set (this is a "destination" block)
    if ((block_id & block_mask) != 0) return;

    // Skip if control qubit (local bit 0) is not set
    if ((local_addr & 1) == 0) return;

    unsigned long long partner_block = block_id | block_mask;
    unsigned long long partner_idx = (partner_block << 7) | local_addr;

    double tmp_r = real[tid];
    double tmp_i = imag[tid];
    real[tid] = real[partner_idx];
    imag[tid] = imag[partner_idx];
    real[partner_idx] = tmp_r;
    imag[partner_idx] = tmp_i;
}

// ============================================================================
// OPTIMIZED CROSS-BLOCK CNOT: Compacted indexing + vectorized access
// ============================================================================
// Key optimizations:
// 1. Compacted indexing: Launch only threads that do work (4x fewer threads)
// 2. Coalesced access: Adjacent threads access adjacent memory within blocks
// 3. Reduced divergence: No early-exit branches
//
// For CNOT(control=0, target>=7):
// - Only source blocks (block_mask bit = 0) participate
// - Only odd local addresses (control bit = 1) are swapped
// - Total swaps = (num_blocks / 2) * 64

// Helper: Insert a 0 bit at position 'bit_pos' in value 'x'
// Example: insert_zero_bit(0b1111, 2) = 0b11011 (insert 0 at position 2)
__device__ __forceinline__ unsigned long long insert_zero_bit(
    unsigned long long x,
    unsigned int bit_pos
) {
    unsigned long long lower_mask = (1ULL << bit_pos) - 1;
    unsigned long long lower = x & lower_mask;
    unsigned long long upper = x & ~lower_mask;
    return lower | (upper << 1);
}

extern "C" __global__ void cnot_f64_cross_block_v2(
    double* real,
    double* imag,
    unsigned long long num_source_blocks,  // = num_blocks / 2
    unsigned int target_block_bit          // Which bit in block_id corresponds to target qubit
) {
    // Each thread handles exactly one swap operation
    // Total swaps = num_source_blocks * 64 (64 odd addresses per 128-amplitude block)
    unsigned long long swap_id = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total_swaps = num_source_blocks << 6;  // * 64

    if (swap_id >= total_swaps) return;

    // Decompose swap_id:
    // - Upper bits: which source block (linear index)
    // - Lower 6 bits: which of the 64 odd addresses (0-63 maps to 1,3,5,...,127)
    unsigned long long source_block_linear = swap_id >> 6;
    unsigned int odd_idx = (unsigned int)(swap_id & 63);

    // Map linear source block index to actual block index
    // Source blocks have 0 at target_block_bit position
    // So we insert a 0 at that bit position
    unsigned long long source_block = insert_zero_bit(source_block_linear, target_block_bit);

    // Destination block has 1 at target_block_bit position
    unsigned long long dest_block = source_block | (1ULL << target_block_bit);

    // Local address: odd numbers only (control bit = 1)
    // odd_idx in [0,63] -> local_addr in {1,3,5,...,127}
    unsigned long long local_addr = ((unsigned long long)odd_idx << 1) | 1;

    // Compute global indices
    unsigned long long src_idx = (source_block << 7) | local_addr;
    unsigned long long dst_idx = (dest_block << 7) | local_addr;

    // Swap amplitudes
    double src_r = real[src_idx];
    double src_i = imag[src_idx];
    double dst_r = real[dst_idx];
    double dst_i = imag[dst_idx];

    real[src_idx] = dst_r;
    imag[src_idx] = dst_i;
    real[dst_idx] = src_r;
    imag[dst_idx] = src_i;
}

// ============================================================================
// ULTRA-OPTIMIZED CROSS-BLOCK CNOT: Shared memory + coalesced loads
// ============================================================================
// Further optimizations:
// 1. Load full cache lines (16 doubles = 128 bytes) for coalesced access
// 2. Use shared memory to stage swaps
// 3. Process blocks in pairs for better memory locality
//
// Each thread block processes one (source_block, dest_block) pair
// 128 threads per block, each handling one amplitude

extern "C" __global__ void cnot_f64_cross_block_v3(
    double* real,
    double* imag,
    unsigned long long num_source_blocks,
    unsigned int target_block_bit
) {
    // Shared memory for staging
    __shared__ double src_real[128];
    __shared__ double src_imag[128];
    __shared__ double dst_real[128];
    __shared__ double dst_imag[128];

    // Each block handles one (source, dest) block pair
    unsigned long long pair_id = blockIdx.x;
    if (pair_id >= num_source_blocks) return;

    unsigned int local_id = threadIdx.x;

    // Compute actual block indices
    unsigned long long source_block = insert_zero_bit(pair_id, target_block_bit);
    unsigned long long dest_block = source_block | (1ULL << target_block_bit);

    // Global base addresses
    unsigned long long src_base = source_block << 7;
    unsigned long long dst_base = dest_block << 7;

    // Phase 1: Coalesced load of both blocks into shared memory
    src_real[local_id] = real[src_base + local_id];
    src_imag[local_id] = imag[src_base + local_id];
    dst_real[local_id] = real[dst_base + local_id];
    dst_imag[local_id] = imag[dst_base + local_id];

    __syncthreads();

    // Phase 2: Swap in shared memory (only odd indices where control=1)
    if (local_id & 1) {
        double tmp_r = src_real[local_id];
        double tmp_i = src_imag[local_id];
        src_real[local_id] = dst_real[local_id];
        src_imag[local_id] = dst_imag[local_id];
        dst_real[local_id] = tmp_r;
        dst_imag[local_id] = tmp_i;
    }

    __syncthreads();

    // Phase 3: Coalesced store back to global memory
    real[src_base + local_id] = src_real[local_id];
    imag[src_base + local_id] = src_imag[local_id];
    real[dst_base + local_id] = dst_real[local_id];
    imag[dst_base + local_id] = dst_imag[local_id];
}

// ============================================================================
// CASCADE CNOT KERNELS: For qubit-to-adjacent-qubit CNOT chains
// ============================================================================
// These kernels enable the "cascade" GHZ pattern:
//   H(0) -> CNOT(0,1) -> CNOT(1,2) -> ... -> CNOT(n-2,n-1)
//
// Key insight: Adjacent qubit CNOTs have MUCH better memory locality!
// - CNOT(0,29) in fan-out pattern: source/dest blocks are 4GB apart
// - CNOT(28,29) in cascade pattern: source/dest blocks are 2GB apart
// - CNOT(7,8) in cascade pattern: source/dest blocks are only 2KB apart!

// CNOT where control is any local qubit (0-6), target is cross-block (>=7)
// Used for CNOT(6,7) in cascade pattern
extern "C" __global__ void cnot_cross_block_ctrl_local(
    double* real,
    double* imag,
    unsigned long long num_source_blocks,
    unsigned int ctrl_local_mask,    // 1 << ctrl_qubit (ctrl in 0-6)
    unsigned int target_block_bit    // target_qubit - 7
) {
    // Each thread block handles one (source, dest) quantum block pair
    unsigned long long pair_id = blockIdx.x;
    if (pair_id >= num_source_blocks) return;

    unsigned int local_id = threadIdx.x;

    // Compute actual block indices
    unsigned long long source_block = insert_zero_bit(pair_id, target_block_bit);
    unsigned long long dest_block = source_block | (1ULL << target_block_bit);

    unsigned long long src_base = source_block << 7;
    unsigned long long dst_base = dest_block << 7;

    // Only swap if control qubit (local) is 1
    if ((local_id & ctrl_local_mask) == 0) return;

    unsigned long long src_idx = src_base + local_id;
    unsigned long long dst_idx = dst_base + local_id;

    double tmp_r = real[src_idx];
    double tmp_i = imag[src_idx];
    real[src_idx] = real[dst_idx];
    imag[src_idx] = imag[dst_idx];
    real[dst_idx] = tmp_r;
    imag[dst_idx] = tmp_i;
}

// CNOT where BOTH control and target are in block space (both >= 7)
// Used for CNOT(7,8), CNOT(8,9), ..., CNOT(n-2,n-1) in cascade pattern
//
// This is the KEY optimization for cascade GHZ!
// The distance between source and dest blocks is only 2^(tgt-ctrl) * block_size
// For CNOT(7,8): 2 * 1KB = 2KB (fits in L1 cache!)
// For CNOT(28,29): 2^21 * 1KB = 2GB (still better than 4GB for fan-out)
extern "C" __global__ void cnot_block_to_block(
    double* real,
    double* imag,
    unsigned long long num_source_blocks,  // Blocks where ctrl=1 AND tgt=0
    unsigned int ctrl_block_bit,           // control_qubit - 7
    unsigned int tgt_block_bit             // target_qubit - 7
) {
    // Each thread block handles one (source, dest) quantum block pair
    unsigned long long pair_id = blockIdx.x;
    if (pair_id >= num_source_blocks) return;

    unsigned int local_id = threadIdx.x;

    // Map pair_id to actual block indices
    // Source blocks have: ctrl_block_bit=1, tgt_block_bit=0
    // We need to insert TWO bits: 0 at tgt_block_bit, 1 at ctrl_block_bit

    // Strategy:
    // 1. Insert 0 at the lower bit position
    // 2. Insert the other bit at its position (adjusted if needed)

    unsigned long long source_block;
    if (ctrl_block_bit < tgt_block_bit) {
        // ctrl bit is lower, insert it first (as 1), then insert tgt (as 0)
        unsigned long long tmp = insert_zero_bit(pair_id, ctrl_block_bit);
        tmp |= (1ULL << ctrl_block_bit);  // Set ctrl bit to 1
        source_block = insert_zero_bit(tmp, tgt_block_bit);
        // tgt bit is already 0 from insert_zero_bit
    } else {
        // tgt bit is lower, insert it first (as 0), then insert ctrl (as 1)
        unsigned long long tmp = insert_zero_bit(pair_id, tgt_block_bit);
        // tgt bit is already 0
        source_block = insert_zero_bit(tmp, ctrl_block_bit);
        source_block |= (1ULL << ctrl_block_bit);  // Set ctrl bit to 1
    }

    // Dest block: flip the target bit
    unsigned long long dest_block = source_block | (1ULL << tgt_block_bit);

    unsigned long long src_base = source_block << 7;
    unsigned long long dst_base = dest_block << 7;

    // ALL 128 amplitudes are swapped (control is in block space, always "1" for source blocks)
    unsigned long long src_idx = src_base + local_id;
    unsigned long long dst_idx = dst_base + local_id;

    double tmp_r = real[src_idx];
    double tmp_i = imag[src_idx];
    real[src_idx] = real[dst_idx];
    imag[src_idx] = imag[dst_idx];
    real[dst_idx] = tmp_r;
    imag[dst_idx] = tmp_i;
}

// Optimized version using shared memory for better coalescing
extern "C" __global__ void cnot_block_to_block_shared(
    double* real,
    double* imag,
    unsigned long long num_source_blocks,
    unsigned int ctrl_block_bit,
    unsigned int tgt_block_bit
) {
    __shared__ double src_real[128];
    __shared__ double src_imag[128];
    __shared__ double dst_real[128];
    __shared__ double dst_imag[128];

    unsigned long long pair_id = blockIdx.x;
    if (pair_id >= num_source_blocks) return;

    unsigned int local_id = threadIdx.x;

    // Compute block indices (same logic as above)
    unsigned long long source_block;
    if (ctrl_block_bit < tgt_block_bit) {
        unsigned long long tmp = insert_zero_bit(pair_id, ctrl_block_bit);
        tmp |= (1ULL << ctrl_block_bit);
        source_block = insert_zero_bit(tmp, tgt_block_bit);
    } else {
        unsigned long long tmp = insert_zero_bit(pair_id, tgt_block_bit);
        source_block = insert_zero_bit(tmp, ctrl_block_bit);
        source_block |= (1ULL << ctrl_block_bit);
    }
    unsigned long long dest_block = source_block | (1ULL << tgt_block_bit);

    unsigned long long src_base = source_block << 7;
    unsigned long long dst_base = dest_block << 7;

    // Phase 1: Coalesced load
    src_real[local_id] = real[src_base + local_id];
    src_imag[local_id] = imag[src_base + local_id];
    dst_real[local_id] = real[dst_base + local_id];
    dst_imag[local_id] = imag[dst_base + local_id];

    __syncthreads();

    // Phase 2: Swap in shared memory (all amplitudes)
    double tmp_r = src_real[local_id];
    double tmp_i = src_imag[local_id];
    src_real[local_id] = dst_real[local_id];
    src_imag[local_id] = dst_imag[local_id];
    dst_real[local_id] = tmp_r;
    dst_imag[local_id] = tmp_i;

    __syncthreads();

    // Phase 3: Coalesced store
    real[src_base + local_id] = src_real[local_id];
    imag[src_base + local_id] = src_imag[local_id];
    real[dst_base + local_id] = dst_real[local_id];
    imag[dst_base + local_id] = dst_imag[local_id];
}

// ============================================================================
// BATCHED GATE KERNEL: Apply multiple gates in a single kernel launch
// ============================================================================
// Gate types for batched execution:
#define GATE_H      0   // Hadamard: param1 = qubit (0-6)
#define GATE_X      1   // Pauli-X:  param1 = qubit (0-6)
#define GATE_Z      2   // Pauli-Z:  param1 = qubit (0-6)
#define GATE_CNOT   3   // CNOT:     param1 = control, param2 = target (both 0-6)
#define GATE_CZ     4   // CZ:       param1 = qubit_a, param2 = qubit_b (both 0-6)
#define GATE_PHASE  5   // Phase:    param1 = qubit, phase_re/phase_im = e^(i*theta)

// Gate operation descriptor (passed as array to kernel)
struct GateOp {
    unsigned int type;
    unsigned int param1;
    unsigned int param2;
    double phase_re;  // For phase gates: real part of e^(i*theta)
    double phase_im;  // For phase gates: imag part of e^(i*theta)
};

// Apply Hadamard to local qubit within a block
__device__ void apply_h_local(
    double* real, double* imag,
    unsigned long long block_start,
    unsigned int qubit
) {
    unsigned int local_id = threadIdx.x;
    unsigned int mask = 1u << qubit;

    // Each thread in the block handles amplitudes that differ only in the target qubit
    // We need to pair up threads: one handles idx0, partner handles idx1
    if ((local_id & mask) != 0) return;  // Only process if target bit is 0

    unsigned long long idx0 = block_start + local_id;
    unsigned long long idx1 = idx0 + mask;

    double r0 = real[idx0], r1 = real[idx1];
    double i0 = imag[idx0], i1 = imag[idx1];

    real[idx0] = (r0 + r1) * RSQRT2;
    real[idx1] = (r0 - r1) * RSQRT2;
    imag[idx0] = (i0 + i1) * RSQRT2;
    imag[idx1] = (i0 - i1) * RSQRT2;
}

// Apply Pauli-X to local qubit within a block
__device__ void apply_x_local(
    double* real, double* imag,
    unsigned long long block_start,
    unsigned int qubit
) {
    unsigned int local_id = threadIdx.x;
    unsigned int mask = 1u << qubit;

    if ((local_id & mask) != 0) return;

    unsigned long long idx0 = block_start + local_id;
    unsigned long long idx1 = idx0 + mask;

    double tmp_r = real[idx0], tmp_i = imag[idx0];
    real[idx0] = real[idx1];  imag[idx0] = imag[idx1];
    real[idx1] = tmp_r;       imag[idx1] = tmp_i;
}

// Apply Pauli-Z to local qubit within a block
__device__ void apply_z_local(
    double* real, double* imag,
    unsigned long long block_start,
    unsigned int qubit
) {
    unsigned int local_id = threadIdx.x;
    unsigned int mask = 1u << qubit;

    // Z gate: |0⟩ → |0⟩, |1⟩ → -|1⟩
    // Only negate if qubit is 1
    if ((local_id & mask) == 0) return;

    unsigned long long idx = block_start + local_id;
    real[idx] = -real[idx];
    imag[idx] = -imag[idx];
}

// Apply CNOT to local qubits within a block
__device__ void apply_cnot_local(
    double* real, double* imag,
    unsigned long long block_start,
    unsigned int control,
    unsigned int target
) {
    unsigned int local_id = threadIdx.x;
    unsigned int ctrl_mask = 1u << control;
    unsigned int tgt_mask = 1u << target;

    // Only process if control=1 and target=0
    if ((local_id & ctrl_mask) == 0) return;
    if ((local_id & tgt_mask) != 0) return;

    unsigned long long idx0 = block_start + local_id;
    unsigned long long idx1 = idx0 ^ tgt_mask;

    double tmp_r = real[idx0], tmp_i = imag[idx0];
    real[idx0] = real[idx1];  imag[idx0] = imag[idx1];
    real[idx1] = tmp_r;       imag[idx1] = tmp_i;
}

// Apply CZ to local qubits within a block
__device__ void apply_cz_local(
    double* real, double* imag,
    unsigned long long block_start,
    unsigned int qubit_a,
    unsigned int qubit_b
) {
    unsigned int local_id = threadIdx.x;
    unsigned int mask_a = 1u << qubit_a;
    unsigned int mask_b = 1u << qubit_b;

    // CZ: phase flip only when both qubits are |1⟩
    if ((local_id & mask_a) == 0) return;
    if ((local_id & mask_b) == 0) return;

    unsigned long long idx = block_start + local_id;
    real[idx] = -real[idx];
    imag[idx] = -imag[idx];
}

// Apply Phase gate to local qubit within a block
__device__ void apply_phase_local(
    double* real, double* imag,
    unsigned long long block_start,
    unsigned int qubit,
    double phase_re,
    double phase_im
) {
    unsigned int local_id = threadIdx.x;
    unsigned int mask = 1u << qubit;

    // Phase: |0⟩ → |0⟩, |1⟩ → e^(iθ)|1⟩
    if ((local_id & mask) == 0) return;

    unsigned long long idx = block_start + local_id;
    double r = real[idx], i = imag[idx];
    // (r + i*i) * (phase_re + i*phase_im) = r*phase_re - i*phase_im + i*(r*phase_im + i*phase_re)
    real[idx] = r * phase_re - i * phase_im;
    imag[idx] = r * phase_im + i * phase_re;
}

// Main batched gate kernel for local operations (qubits 0-6)
// Processes 128 amplitudes per thread block (one quantum block)
extern "C" __global__ void batched_gates_local(
    double* real,
    double* imag,
    const GateOp* gates,
    unsigned int num_gates,
    unsigned long long num_blocks  // Number of 128-amplitude blocks
) {
    unsigned long long block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    unsigned long long block_start = block_id * 128;

    // Apply each gate in sequence with sync between gates
    for (unsigned int g = 0; g < num_gates; g++) {
        GateOp op = gates[g];

        switch (op.type) {
            case GATE_H:
                apply_h_local(real, imag, block_start, op.param1);
                break;
            case GATE_X:
                apply_x_local(real, imag, block_start, op.param1);
                break;
            case GATE_Z:
                apply_z_local(real, imag, block_start, op.param1);
                break;
            case GATE_CNOT:
                apply_cnot_local(real, imag, block_start, op.param1, op.param2);
                break;
            case GATE_CZ:
                apply_cz_local(real, imag, block_start, op.param1, op.param2);
                break;
            case GATE_PHASE:
                apply_phase_local(real, imag, block_start, op.param1, op.phase_re, op.phase_im);
                break;
        }

        // Sync all threads in block before next gate
        __syncthreads();
    }
}

// ============================================================================
// FUSED GHZ KERNEL: Creates (|0...0⟩ + |1...1⟩)/√2 in ONE pass
// ============================================================================
// This kernel bypasses individual gate simulation by directly computing the
// final GHZ state. Each thread sets its amplitude based on whether the index
// is 0 (all zeros state) or 2^n-1 (all ones state).
//
// Mathematically equivalent to H(0) + CNOT(0,1) + ... + CNOT(0,n-1)
// but executes in a SINGLE kernel launch instead of n launches.
//
// Performance: Eliminates kernel launch overhead (30 launches -> 1 launch)
// This is the GPU equivalent of CPU-side gate fusion (MegaGate).
extern "C" __global__ void create_ghz_fused(
    double* real,
    double* imag,
    unsigned long long total_amps,
    unsigned long long all_ones_idx  // 2^n - 1
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_amps) return;

    if (tid == 0 || tid == all_ones_idx) {
        // GHZ amplitudes: 1/sqrt(2) for |0...0⟩ and |1...1⟩
        real[tid] = RSQRT2;
        imag[tid] = 0.0;
    } else {
        // All other amplitudes are zero
        real[tid] = 0.0;
        imag[tid] = 0.0;
    }
}

// ============================================================================
// FP16 Conversion Kernels for Tensor Core GEMM
// ============================================================================

// Convert f64 array to f16 (half precision)
extern "C" __global__ void convert_f64_to_f16(
    const double* __restrict__ input,
    __half* __restrict__ output,
    unsigned long long n
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    output[tid] = __float2half((float)input[tid]);
}

// Convert f16 array back to f64
extern "C" __global__ void convert_f16_to_f64(
    const __half* __restrict__ input,
    double* __restrict__ output,
    unsigned long long n
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    output[tid] = (double)__half2float(input[tid]);
}
"#;

/// Cached CUDA kernels for f64 quantum operations
#[cfg(feature = "cuda")]
pub struct F64KernelCache {
    hadamard_q0: CudaFunction,
    hadamard_local: CudaFunction,
    cnot_local: CudaFunction,
    cnot_cross_block: CudaFunction,
    cnot_cross_block_v2: CudaFunction, // Compacted indexing
    cnot_cross_block_v3: CudaFunction, // Shared memory + coalesced
    // Cascade CNOT kernels for qubit-to-adjacent-qubit chains
    cnot_cross_block_ctrl_local: CudaFunction, // Control local (0-6), target cross-block (>=7)
    cnot_block_to_block: CudaFunction,         // Both control and target in block space
    cnot_block_to_block_shared: CudaFunction,  // Shared memory version
    ghz_fused: CudaFunction,
    batched_gates_local: CudaFunction,
    // FP16 conversion kernels for Tensor Core GEMM
    convert_f64_to_f16: CudaFunction,
    convert_f16_to_f64: CudaFunction,
}

// ============================================================================
// Gate Operation Types for Batched Execution
// ============================================================================

/// Gate type constants (must match CUDA #defines)
#[cfg(feature = "cuda")]
pub mod gate_type {
    pub const H: u32 = 0;
    pub const X: u32 = 1;
    pub const Z: u32 = 2;
    pub const CNOT: u32 = 3;
    pub const CZ: u32 = 4;
    pub const PHASE: u32 = 5;
}

/// Gate operation descriptor for batched execution
/// Must match the CUDA GateOp struct layout exactly
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GateOp {
    pub gate_type: u32,
    pub param1: u32,
    pub param2: u32,
    pub phase_re: f64,
    pub phase_im: f64,
}

// Safety: GateOp is repr(C) with only primitive types, suitable for GPU transfer
#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::DeviceRepr for GateOp {}

#[cfg(feature = "cuda")]
impl GateOp {
    /// Create Hadamard gate on qubit (0-6)
    pub fn h(qubit: u8) -> Self {
        assert!(qubit < 7, "Batched H requires qubit < 7");
        Self {
            gate_type: gate_type::H,
            param1: qubit as u32,
            param2: 0,
            phase_re: 0.0,
            phase_im: 0.0,
        }
    }

    /// Create Pauli-X gate on qubit (0-6)
    pub fn x(qubit: u8) -> Self {
        assert!(qubit < 7, "Batched X requires qubit < 7");
        Self {
            gate_type: gate_type::X,
            param1: qubit as u32,
            param2: 0,
            phase_re: 0.0,
            phase_im: 0.0,
        }
    }

    /// Create Pauli-Z gate on qubit (0-6)
    pub fn z(qubit: u8) -> Self {
        assert!(qubit < 7, "Batched Z requires qubit < 7");
        Self {
            gate_type: gate_type::Z,
            param1: qubit as u32,
            param2: 0,
            phase_re: 0.0,
            phase_im: 0.0,
        }
    }

    /// Create CNOT gate with control and target (both 0-6)
    pub fn cnot(control: u8, target: u8) -> Self {
        assert!(
            control < 7 && target < 7,
            "Batched CNOT requires both qubits < 7"
        );
        Self {
            gate_type: gate_type::CNOT,
            param1: control as u32,
            param2: target as u32,
            phase_re: 0.0,
            phase_im: 0.0,
        }
    }

    /// Create CZ gate on two qubits (both 0-6)
    pub fn cz(qubit_a: u8, qubit_b: u8) -> Self {
        assert!(
            qubit_a < 7 && qubit_b < 7,
            "Batched CZ requires both qubits < 7"
        );
        Self {
            gate_type: gate_type::CZ,
            param1: qubit_a as u32,
            param2: qubit_b as u32,
            phase_re: 0.0,
            phase_im: 0.0,
        }
    }

    /// Create Phase gate on qubit (0-6) with angle theta
    pub fn phase(qubit: u8, theta: f64) -> Self {
        assert!(qubit < 7, "Batched Phase requires qubit < 7");
        Self {
            gate_type: gate_type::PHASE,
            param1: qubit as u32,
            param2: 0,
            phase_re: theta.cos(),
            phase_im: theta.sin(),
        }
    }

    /// Create S gate (Phase with theta = pi/2)
    pub fn s(qubit: u8) -> Self {
        Self::phase(qubit, std::f64::consts::FRAC_PI_2)
    }

    /// Create T gate (Phase with theta = pi/4)
    pub fn t(qubit: u8) -> Self {
        Self::phase(qubit, std::f64::consts::FRAC_PI_4)
    }
}

/// GPU-resident quantum grid with f64 precision
///
/// Stores 2^n amplitudes across 2^(n-7) blocks of 128 amplitudes each.
/// Uses SoA (Structure of Arrays) layout for coalesced memory access.
#[cfg(feature = "cuda")]
pub struct QuantumGridF64Gpu {
    /// Real components (f64 in VRAM)
    pub real: CudaSlice<f64>,
    /// Imaginary components (f64 in VRAM)
    pub imag: CudaSlice<f64>,
    /// Number of qubits
    pub n_qubits: usize,
    /// Number of blocks (2^(n-7))
    pub num_blocks: usize,
    /// Total number of amplitudes (num_blocks * 128)
    pub num_amplitudes: usize,
    /// Cached kernels
    kernels: F64KernelCache,
    /// Reference to runtime for kernel launches
    stream: Arc<cudarc::driver::CudaStream>,
}

#[cfg(feature = "cuda")]
impl QuantumGridF64Gpu {
    /// Create a new GPU-resident quantum grid for n qubits
    ///
    /// Allocates 2^n * 16 bytes of VRAM (real + imag, each f64).
    ///
    /// # Memory Requirements
    /// - 20 qubits: 16 MB
    /// - 25 qubits: 512 MB
    /// - 27 qubits: 2 GB
    /// - 30 qubits: 16 GB
    pub fn new(rt: &CudaRuntime, n_qubits: usize) -> CudaResult<Self> {
        assert!(n_qubits >= 7, "Minimum 7 qubits for block-based simulation");

        let num_blocks = 1usize << (n_qubits - 7);
        let num_amplitudes = num_blocks * 128;

        // Allocate GPU memory
        let mut real: CudaSlice<f64> = rt.alloc_zeros(num_amplitudes)?;
        let imag: CudaSlice<f64> = rt.alloc_zeros(num_amplitudes)?;

        // Compile kernels
        let kernels = Self::compile_kernels(rt)?;

        // Initialize |0...0⟩ state: amplitude[0] = 1.0
        let mut init_real = vec![0.0f64; num_amplitudes];
        init_real[0] = 1.0;
        rt.stream()
            .memcpy_htod(&init_real, &mut real)
            .map_err(|e| CudaError::TransferFailed(format!("{:?}", e)))?;

        Ok(Self {
            real,
            imag,
            n_qubits,
            num_blocks,
            num_amplitudes,
            kernels,
            stream: rt.stream().clone(),
        })
    }

    /// Compile f64 quantum kernels using NVRTC
    ///
    /// Uses compile_ptx_with_opts to compile CUDA C to PTX at runtime,
    /// ensuring compatibility with any GPU architecture.
    fn compile_kernels(rt: &CudaRuntime) -> CudaResult<F64KernelCache> {
        use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
        use logic_fabric_core::cuda::get_cuda_include_path;

        // Get CUDA include path for headers
        let cuda_include = get_cuda_include_path();

        // Get the compute capability for this GPU
        // Leak the string to satisfy 'static lifetime requirement (only done once per process)
        let arch: &'static str = Box::leak(
            rt.get_arch_string()
                .unwrap_or_else(|_| "compute_80".to_string())
                .into_boxed_str(),
        );

        // Compile options for the target GPU
        let opts = CompileOptions {
            include_paths: vec![cuda_include],
            arch: Some(arch),
            ..Default::default()
        };

        // Compile CUDA C to PTX
        let ptx = compile_ptx_with_opts(QUANTUM_F64_CUDA, opts).map_err(|e| {
            CudaError::KernelCompilationFailed(format!("F64 quantum kernel compile error: {:?}", e))
        })?;

        // Load the compiled module
        let module = rt
            .ctx()
            .load_module(ptx)
            .map_err(|e| CudaError::KernelCompilationFailed(format!("Module load: {:?}", e)))?;

        // Load kernel functions
        let hadamard_q0 = module
            .load_function("hadamard_f64_q0")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

        let hadamard_local = module
            .load_function("hadamard_f64_local")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

        let cnot_local = module
            .load_function("cnot_f64_local")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

        let cnot_cross_block = module
            .load_function("cnot_f64_cross_block")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

        let cnot_cross_block_v2 = module
            .load_function("cnot_f64_cross_block_v2")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

        let cnot_cross_block_v3 = module
            .load_function("cnot_f64_cross_block_v3")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

        let ghz_fused = module
            .load_function("create_ghz_fused")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

        let batched_gates_local = module
            .load_function("batched_gates_local")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

        // Cascade CNOT kernels for qubit-to-adjacent-qubit chains
        let cnot_cross_block_ctrl_local = module
            .load_function("cnot_cross_block_ctrl_local")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

        let cnot_block_to_block = module
            .load_function("cnot_block_to_block")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

        let cnot_block_to_block_shared = module
            .load_function("cnot_block_to_block_shared")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

        // FP16 conversion kernels for Tensor Core GEMM
        let convert_f64_to_f16 = module
            .load_function("convert_f64_to_f16")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

        let convert_f16_to_f64 = module
            .load_function("convert_f16_to_f64")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{:?}", e)))?;

        Ok(F64KernelCache {
            hadamard_q0,
            hadamard_local,
            cnot_local,
            cnot_cross_block,
            cnot_cross_block_v2,
            cnot_cross_block_v3,
            cnot_cross_block_ctrl_local,
            cnot_block_to_block,
            cnot_block_to_block_shared,
            ghz_fused,
            batched_gates_local,
            convert_f64_to_f16,
            convert_f16_to_f64,
        })
    }

    /// Apply Hadamard to qubit 0 (optimized kernel)
    pub fn apply_hadamard_q0(&mut self) -> CudaResult<()> {
        let num_pairs = (self.num_amplitudes / 2) as u64;
        let threads_per_block = 256u32;
        let num_blocks = ((num_pairs as u32) + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.hadamard_q0)
                .arg(&self.real)
                .arg(&self.imag)
                .arg(&num_pairs)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Apply Hadamard to any local qubit (0-6)
    pub fn apply_hadamard(&mut self, qubit: u8) -> CudaResult<()> {
        assert!(qubit < 7, "Use cross-block Hadamard for qubit >= 7");

        if qubit == 0 {
            return self.apply_hadamard_q0();
        }

        let qubit_mask = 1u32 << qubit;
        let num_pairs = (self.num_amplitudes / 2) as u64;
        let threads_per_block = 256u32;
        let num_blocks = ((num_pairs as u32) + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.hadamard_local)
                .arg(&self.real)
                .arg(&self.imag)
                .arg(&num_pairs)
                .arg(&qubit_mask)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Apply CNOT with control=0 and target in range 1-6
    pub fn apply_cnot_local(&mut self, target: u8) -> CudaResult<()> {
        assert!(
            target >= 1 && target < 7,
            "Target must be 1-6 for local CNOT"
        );

        let target_mask = 1u32 << target;
        let num_amps = self.num_amplitudes as u64;
        let threads_per_block = 256u32;
        let num_blocks_launch = ((num_amps as u32) + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks_launch, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.cnot_local)
                .arg(&self.real)
                .arg(&self.imag)
                .arg(&num_amps)
                .arg(&target_mask)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Apply cross-block CNOT (control=0, target >= 7)
    pub fn apply_cnot_cross_block(&mut self, target_qubit: usize) -> CudaResult<()> {
        assert!(
            target_qubit >= 7 && target_qubit < self.n_qubits,
            "Target must be >= 7 and < n_qubits"
        );

        let block_bit = target_qubit - 7;
        let block_mask = 1u64 << block_bit;
        let total_amps = self.num_amplitudes as u64;

        let threads_per_block = 256u32;
        let num_blocks_launch = ((total_amps as u32) + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks_launch, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.cnot_cross_block)
                .arg(&self.real)
                .arg(&self.imag)
                .arg(&total_amps)
                .arg(&block_mask)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Apply cross-block CNOT with compacted indexing (v2)
    /// Launches 4x fewer threads than v1 by only spawning threads that do work.
    pub fn apply_cnot_cross_block_v2(&mut self, target_qubit: usize) -> CudaResult<()> {
        assert!(
            target_qubit >= 7 && target_qubit < self.n_qubits,
            "Target must be >= 7 and < n_qubits"
        );

        let target_block_bit = (target_qubit - 7) as u32;
        let num_source_blocks = (self.num_blocks / 2) as u64;
        let total_swaps = num_source_blocks * 64; // 64 odd addresses per block

        let threads_per_block = 256u32;
        let num_blocks_launch = ((total_swaps as u32) + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks_launch, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.cnot_cross_block_v2)
                .arg(&self.real)
                .arg(&self.imag)
                .arg(&num_source_blocks)
                .arg(&target_block_bit)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Apply cross-block CNOT with shared memory + coalesced access (v3)
    /// Uses shared memory staging for fully coalesced global memory access.
    /// Each thread block processes one (source, dest) block pair.
    pub fn apply_cnot_cross_block_v3(&mut self, target_qubit: usize) -> CudaResult<()> {
        assert!(
            target_qubit >= 7 && target_qubit < self.n_qubits,
            "Target must be >= 7 and < n_qubits"
        );

        let target_block_bit = (target_qubit - 7) as u32;
        let num_source_blocks = (self.num_blocks / 2) as u64;

        // Each CUDA block handles one (source, dest) quantum block pair
        // 128 threads per block (one per amplitude in quantum block)
        let cfg = LaunchConfig {
            grid_dim: ((self.num_blocks / 2) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 128 * 8 * 4, // 4 arrays of 128 doubles
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.cnot_cross_block_v3)
                .arg(&self.real)
                .arg(&self.imag)
                .arg(&num_source_blocks)
                .arg(&target_block_bit)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    // ========================================================================
    // CASCADE CNOT METHODS: For adjacent qubit chains with better locality
    // ========================================================================

    /// Apply CNOT with control in local qubits (0-6) and target in block space (>=7)
    ///
    /// Used for CNOT(6,7) in the cascade GHZ pattern.
    /// Each thread block handles one (source, dest) quantum block pair.
    pub fn apply_cnot_ctrl_local_tgt_cross(
        &mut self,
        control: u8,
        target_qubit: usize,
    ) -> CudaResult<()> {
        assert!(control < 7, "Control must be local qubit (0-6)");
        assert!(
            target_qubit >= 7 && target_qubit < self.n_qubits,
            "Target must be >= 7 and < n_qubits"
        );

        let ctrl_local_mask = 1u32 << control;
        let target_block_bit = (target_qubit - 7) as u32;
        let num_source_blocks = (self.num_blocks / 2) as u64;

        // Each CUDA block handles one (source, dest) quantum block pair
        // 128 threads per block (one per amplitude in quantum block)
        let cfg = LaunchConfig {
            grid_dim: ((self.num_blocks / 2) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.cnot_cross_block_ctrl_local)
                .arg(&self.real)
                .arg(&self.imag)
                .arg(&num_source_blocks)
                .arg(&ctrl_local_mask)
                .arg(&target_block_bit)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Apply CNOT with both control and target in block space (both >= 7)
    ///
    /// Used for CNOT(7,8), CNOT(8,9), ..., CNOT(n-2,n-1) in cascade pattern.
    ///
    /// Key optimization: Adjacent qubit CNOTs have much better memory locality!
    /// - CNOT(7,8): source/dest blocks are 2KB apart (fits in L1 cache)
    /// - CNOT(28,29): source/dest blocks are 2GB apart (still 2x better than fan-out)
    pub fn apply_cnot_block_to_block(
        &mut self,
        control_qubit: usize,
        target_qubit: usize,
    ) -> CudaResult<()> {
        assert!(
            control_qubit >= 7 && control_qubit < self.n_qubits,
            "Control must be >= 7 and < n_qubits"
        );
        assert!(
            target_qubit >= 7 && target_qubit < self.n_qubits,
            "Target must be >= 7 and < n_qubits"
        );
        assert!(
            control_qubit != target_qubit,
            "Control and target must differ"
        );

        let ctrl_block_bit = (control_qubit - 7) as u32;
        let tgt_block_bit = (target_qubit - 7) as u32;

        // Number of source blocks: blocks where ctrl=1 AND tgt=0
        // = total_blocks / 4
        let num_source_blocks = (self.num_blocks / 4) as u64;

        // Each CUDA block handles one (source, dest) quantum block pair
        let cfg = LaunchConfig {
            grid_dim: ((self.num_blocks / 4) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.cnot_block_to_block)
                .arg(&self.real)
                .arg(&self.imag)
                .arg(&num_source_blocks)
                .arg(&ctrl_block_bit)
                .arg(&tgt_block_bit)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Apply CNOT with both control and target in block space (shared memory version)
    ///
    /// Uses shared memory staging for better memory coalescing.
    pub fn apply_cnot_block_to_block_shared(
        &mut self,
        control_qubit: usize,
        target_qubit: usize,
    ) -> CudaResult<()> {
        assert!(
            control_qubit >= 7 && control_qubit < self.n_qubits,
            "Control must be >= 7 and < n_qubits"
        );
        assert!(
            target_qubit >= 7 && target_qubit < self.n_qubits,
            "Target must be >= 7 and < n_qubits"
        );
        assert!(
            control_qubit != target_qubit,
            "Control and target must differ"
        );

        let ctrl_block_bit = (control_qubit - 7) as u32;
        let tgt_block_bit = (target_qubit - 7) as u32;
        let num_source_blocks = (self.num_blocks / 4) as u64;

        let cfg = LaunchConfig {
            grid_dim: ((self.num_blocks / 4) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 128 * 8 * 4, // 4 arrays of 128 doubles
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.cnot_block_to_block_shared)
                .arg(&self.real)
                .arg(&self.imag)
                .arg(&num_source_blocks)
                .arg(&ctrl_block_bit)
                .arg(&tgt_block_bit)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Create GHZ state using optimized v2 cross-block CNOT
    pub fn create_ghz_state_v2(&mut self) -> CudaResult<u64> {
        let start = std::time::Instant::now();

        self.reset_to_zero()?;
        self.apply_hadamard_q0()?;

        for target in 1..7.min(self.n_qubits) {
            self.apply_cnot_local(target as u8)?;
        }

        for target_qubit in 7..self.n_qubits {
            self.apply_cnot_cross_block_v2(target_qubit)?;
        }

        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    /// Create GHZ state using optimized v3 cross-block CNOT (shared memory)
    pub fn create_ghz_state_v3(&mut self) -> CudaResult<u64> {
        let start = std::time::Instant::now();

        self.reset_to_zero()?;
        self.apply_hadamard_q0()?;

        for target in 1..7.min(self.n_qubits) {
            self.apply_cnot_local(target as u8)?;
        }

        for target_qubit in 7..self.n_qubits {
            self.apply_cnot_cross_block_v3(target_qubit)?;
        }

        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    /// Create GHZ state using CASCADE pattern for better memory locality
    ///
    /// HYBRID APPROACH:
    /// - Local qubits (0-6): Fan-out pattern (already fast, within 1KB block)
    /// - Cross-block qubits (7+): Cascade pattern (much better locality!)
    ///
    /// Pattern: H(0) -> CNOT(0,1)...CNOT(0,6) -> CNOT(6,7) -> CNOT(7,8) -> ... -> CNOT(n-2,n-1)
    ///
    /// Memory locality improvement for cross-block operations:
    /// - Fan-out CNOT(0,29): source/dest blocks are 4GB apart!
    /// - Cascade CNOT(7,8): source/dest blocks are only 2KB apart (L1 cache!)
    /// - Cascade CNOT(28,29): source/dest blocks are 2GB apart (2x better than fan-out)
    ///
    /// Both patterns produce the same GHZ state (|0...0⟩ + |1...1⟩)/√2.
    ///
    /// Returns time in microseconds.
    pub fn create_ghz_state_cascade(&mut self) -> CudaResult<u64> {
        let start = std::time::Instant::now();

        // Reset to |0...0⟩
        self.reset_to_zero()?;

        // Apply H to qubit 0
        self.apply_hadamard_q0()?;

        // Local qubits: Fan-out pattern (CNOT(0,1), CNOT(0,2), ..., CNOT(0,6))
        // This is already fast because all amplitudes are within the same 1KB block
        for target in 1..7.min(self.n_qubits) {
            self.apply_cnot_local(target as u8)?;
        }

        // Transition from local to cross-block: CNOT(6,7)
        // Uses control qubit 6 (local) and target qubit 7 (cross-block)
        if self.n_qubits > 7 {
            self.apply_cnot_ctrl_local_tgt_cross(6, 7)?;
        }

        // Cross-block cascade: CNOT(7,8), CNOT(8,9), ..., CNOT(n-2,n-1)
        // Each CNOT operates on adjacent block-space qubits for better locality
        for i in 7..(self.n_qubits - 1) {
            self.apply_cnot_block_to_block(i, i + 1)?;
        }

        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    /// Create GHZ state using CASCADE pattern with shared memory optimization
    ///
    /// Same as create_ghz_state_cascade but uses shared memory for cross-block CNOTs.
    ///
    /// Returns time in microseconds.
    pub fn create_ghz_state_cascade_shared(&mut self) -> CudaResult<u64> {
        let start = std::time::Instant::now();

        self.reset_to_zero()?;
        self.apply_hadamard_q0()?;

        // Local fan-out (fast within block)
        for target in 1..7.min(self.n_qubits) {
            self.apply_cnot_local(target as u8)?;
        }

        // Transition: CNOT(6,7)
        if self.n_qubits > 7 {
            self.apply_cnot_ctrl_local_tgt_cross(6, 7)?;
        }

        // Cross-block cascade with shared memory
        for i in 7..(self.n_qubits - 1) {
            self.apply_cnot_block_to_block_shared(i, i + 1)?;
        }

        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    // ========================================================================
    // GEMM-BASED CROSS-BLOCK OPERATIONS: Using cuBLAS for Tensor Core acceleration
    // ========================================================================

    /// Apply CNOT on the two highest qubits using cuBLAS DGEMM
    ///
    /// This uses the matrix multiplication insight from tensor network contraction:
    /// - Reshape state: 2^n → (2^(n-2)) × 4 matrix
    /// - Apply 4×4 CNOT gate via DGEMM
    /// - Reshape back
    ///
    /// The CNOT matrix for control=bit2, target=bit3 (in 4-element indexing):
    /// ```text
    /// |00⟩ → |00⟩   [1 0 0 0]
    /// |01⟩ → |01⟩   [0 1 0 0]
    /// |10⟩ → |11⟩   [0 0 0 1]
    /// |11⟩ → |10⟩   [0 0 1 0]
    /// ```
    ///
    /// Since CNOT is real-valued and we have SoA layout, we do separate
    /// DGEMM for real and imaginary parts.
    pub fn apply_cnot_q0_q1_gemm(&mut self, rt: &CudaRuntime) -> CudaResult<u64> {
        let start = std::time::Instant::now();

        // CNOT matrix for control=bit0, target=bit1
        // Memory layout: offset 0=|00⟩, 1=|01⟩, 2=|10⟩, 3=|11⟩
        // CNOT swaps |01⟩ ↔ |11⟩ (swap offsets 1 and 3)
        // Column-major storage for cuBLAS
        let cnot_matrix: [f64; 16] = [
            1.0, 0.0, 0.0, 0.0, // column 0: |00⟩ → |00⟩
            0.0, 0.0, 0.0, 1.0, // column 1: |01⟩ → |11⟩
            0.0, 0.0, 1.0, 0.0, // column 2: |10⟩ → |10⟩
            0.0, 1.0, 0.0, 0.0, // column 3: |11⟩ → |01⟩
        ];

        // Upload CNOT matrix to GPU
        let gate_gpu: CudaSlice<f64> = rt.upload(&cnot_matrix)?;

        // Number of 4-element groups = 2^(n-2)
        let num_groups = self.num_amplitudes / 4;

        // Create cuBLAS handle using the stream
        let blas = CudaBlas::new(self.stream.clone())
            .map_err(|e| CudaError::LaunchFailed(format!("cuBLAS init: {:?}", e)))?;

        // Allocate single temp buffer and reuse it (saves 8GB VRAM at 30 qubits)
        let mut temp_out: CudaSlice<f64> = rt.alloc_zeros(self.num_amplitudes)?;

        // DGEMM: C = alpha * A * B + beta * C
        // We want: new_state = old_state × CNOT^T
        //
        // In cuBLAS column-major:
        // - A = CNOT (4×4), stored column-major
        // - B = state reshaped as (4 × num_groups), stored column-major
        // - C = output (4 × num_groups)
        //
        // Actually, cuBLAS gemm does: C = alpha * op(A) * op(B) + beta * C
        // We want: out = state × CNOT^T
        // Which is: out^T = CNOT × state^T
        //
        // With our row-major state interpreted as column-major:
        // - B = state is (num_groups × 4) row-major = (4 × num_groups) column-major
        // - We want C such that C[i,j] = sum_k B[i,k] * CNOT[j,k]
        //
        // This is: C = A × B where A=CNOT, B=state viewed as (4 × num_groups)
        // m=4, n=num_groups, k=4

        unsafe {
            // DGEMM for real part
            blas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: 4,
                    n: num_groups as i32,
                    k: 4,
                    alpha: 1.0f64,
                    lda: 4,
                    ldb: 4,
                    beta: 0.0f64,
                    ldc: 4,
                },
                &gate_gpu,
                &self.real,
                &mut temp_out,
            )
            .map_err(|e| CudaError::LaunchFailed(format!("DGEMM real: {:?}", e)))?;
        }

        // Copy real result back before reusing temp
        self.stream
            .memcpy_dtod(&temp_out, &mut self.real)
            .map_err(|e| CudaError::TransferFailed(format!("{:?}", e)))?;

        unsafe {
            // DGEMM for imaginary part (reuse temp buffer)
            blas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: 4,
                    n: num_groups as i32,
                    k: 4,
                    alpha: 1.0f64,
                    lda: 4,
                    ldb: 4,
                    beta: 0.0f64,
                    ldc: 4,
                },
                &gate_gpu,
                &self.imag,
                &mut temp_out,
            )
            .map_err(|e| CudaError::LaunchFailed(format!("DGEMM imag: {:?}", e)))?;
        }

        // Copy imag result back
        self.stream
            .memcpy_dtod(&temp_out, &mut self.imag)
            .map_err(|e| CudaError::TransferFailed(format!("{:?}", e)))?;

        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    /// Compute the 128×128 fused gate matrix for H(0) + CNOT(0,1) + ... + CNOT(0,6)
    ///
    /// This fuses 7 gate operations into a single matrix multiplication.
    /// The transformation for input state |j⟩ where j is a 7-bit index:
    /// - H on qubit 0 creates superposition
    /// - CNOTs propagate: when q0=1, flip all of q1-q6
    ///
    /// Result: |j⟩ → (|j'⟩ + (-1)^(j&1) * |(j' XOR 126) | 1⟩) / √2
    /// where j' = j & ~1 (j with bit 0 cleared)
    fn compute_fused_h_cnot_matrix_128() -> Vec<f64> {
        let mut matrix = vec![0.0f64; 128 * 128];
        let rsqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        for input_idx in 0..128 {
            let upper_bits = input_idx & !1; // Clear bit 0
            let sign = if (input_idx & 1) == 0 { 1.0 } else { -1.0 };

            // After H on q0, then CNOTs from q0 to q1-q6:
            // - Component with q0=0 goes to index upper_bits
            // - Component with q0=1 goes to index (upper_bits XOR 126) | 1
            //   (126 = 0b1111110, flips bits 1-6)
            let out_q0_zero = upper_bits;
            let out_q0_one = (upper_bits ^ 126) | 1;

            // Column-major storage for cuBLAS: element (row, col) at row + col * 128
            matrix[out_q0_zero + input_idx * 128] = rsqrt2;
            matrix[out_q0_one + input_idx * 128] = sign * rsqrt2;
        }

        matrix
    }

    /// Apply the fused 128×128 gate (H + 6 CNOTs) to qubits 0-6 using cuBLAS DGEMM
    ///
    /// Reshapes state as (128 × num_groups) and applies gate via matrix multiply:
    /// - State: 2^n amplitudes → (128 rows) × (2^(n-7) columns)
    /// - Gate: 128 × 128 fused matrix
    /// - Result: Single DGEMM replaces 7 kernel launches
    pub fn apply_fused_local_gemm(&mut self, rt: &CudaRuntime) -> CudaResult<u64> {
        let start = std::time::Instant::now();

        if self.n_qubits < 7 {
            return Err(CudaError::LaunchFailed(
                "Fused 128×128 GEMM requires at least 7 qubits".to_string(),
            ));
        }

        // Compute and upload the fused gate matrix
        let fused_matrix = Self::compute_fused_h_cnot_matrix_128();
        let gate_gpu: CudaSlice<f64> = rt.upload(&fused_matrix)?;

        // Number of 128-element groups = 2^(n-7)
        let num_groups = self.num_amplitudes / 128;

        // Create cuBLAS handle
        let blas = CudaBlas::new(self.stream.clone())
            .map_err(|e| CudaError::LaunchFailed(format!("cuBLAS init: {:?}", e)))?;

        // Allocate temp buffer (reuse for real and imag)
        let mut temp_out: CudaSlice<f64> = rt.alloc_zeros(self.num_amplitudes)?;

        // DGEMM: C = A × B
        // A = gate (128 × 128), B = state (128 × num_groups), C = output (128 × num_groups)
        // m=128, n=num_groups, k=128

        unsafe {
            // DGEMM for real part
            blas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: 128,
                    n: num_groups as i32,
                    k: 128,
                    alpha: 1.0f64,
                    lda: 128,
                    ldb: 128,
                    beta: 0.0f64,
                    ldc: 128,
                },
                &gate_gpu,
                &self.real,
                &mut temp_out,
            )
            .map_err(|e| CudaError::LaunchFailed(format!("DGEMM real: {:?}", e)))?;
        }

        // Copy real result back
        self.stream
            .memcpy_dtod(&temp_out, &mut self.real)
            .map_err(|e| CudaError::TransferFailed(format!("{:?}", e)))?;

        unsafe {
            // DGEMM for imaginary part
            blas.gemm(
                GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: 128,
                    n: num_groups as i32,
                    k: 128,
                    alpha: 1.0f64,
                    lda: 128,
                    ldb: 128,
                    beta: 0.0f64,
                    ldc: 128,
                },
                &gate_gpu,
                &self.imag,
                &mut temp_out,
            )
            .map_err(|e| CudaError::LaunchFailed(format!("DGEMM imag: {:?}", e)))?;
        }

        // Copy imag result back
        self.stream
            .memcpy_dtod(&temp_out, &mut self.imag)
            .map_err(|e| CudaError::TransferFailed(format!("{:?}", e)))?;

        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    /// Create GHZ state using fused 128×128 GEMM for local gates
    ///
    /// Fuses H(0) + CNOT(0,1) + ... + CNOT(0,6) into single 128×128 matrix multiply.
    /// This replaces 7 kernel launches with 1 GEMM call.
    pub fn create_ghz_state_fused_gemm(&mut self, rt: &CudaRuntime) -> CudaResult<u64> {
        let start = std::time::Instant::now();

        // Reset to |0...0⟩
        self.reset_to_zero()?;

        // Apply fused H + 6 CNOTs via single 128×128 GEMM
        self.apply_fused_local_gemm(rt)?;

        // Cross-block CNOTs with regular kernels
        for target_qubit in 7..self.n_qubits {
            self.apply_cnot_cross_block(target_qubit)?;
        }

        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    /// Compute the 128×128 fused gate matrix in FP16 for Tensor Core acceleration
    /// Returns Vec<u16> (bit-compatible with half::f16) for GPU compatibility
    fn compute_fused_h_cnot_matrix_128_u16() -> Vec<u16> {
        let mut matrix = vec![0u16; 128 * 128];
        let rsqrt2 = f16::from_f64(std::f64::consts::FRAC_1_SQRT_2);
        let neg_rsqrt2 = f16::from_f64(-std::f64::consts::FRAC_1_SQRT_2);

        for input_idx in 0..128 {
            let upper_bits = input_idx & !1;
            let sign = if (input_idx & 1) == 0 {
                rsqrt2
            } else {
                neg_rsqrt2
            };

            let out_q0_zero = upper_bits;
            let out_q0_one = (upper_bits ^ 126) | 1;

            matrix[out_q0_zero + input_idx * 128] = rsqrt2.to_bits();
            matrix[out_q0_one + input_idx * 128] = sign.to_bits();
        }

        matrix
    }

    /// Apply the fused 128×128 gate using FP16 Tensor Core GEMM
    ///
    /// This version:
    /// 1. Converts F64 state to FP16
    /// 2. Applies 128×128 HGEMM via Tensor Cores
    /// 3. Converts result back to F64
    ///
    /// Tensor Cores provide ~10-20x speedup for matrix multiply at reduced precision.
    pub fn apply_fused_local_gemm_fp16(&mut self, rt: &CudaRuntime) -> CudaResult<u64> {
        let start = std::time::Instant::now();

        if self.n_qubits < 7 {
            return Err(CudaError::LaunchFailed(
                "Fused 128×128 GEMM requires at least 7 qubits".to_string(),
            ));
        }

        // Compute and upload the FP16 fused gate matrix (as u16 for GPU compat)
        let fused_matrix = Self::compute_fused_h_cnot_matrix_128_u16();
        let gate_gpu: CudaSlice<u16> = rt.upload(&fused_matrix)?;

        let num_groups = self.num_amplitudes / 128;
        let threads_per_block = 256u32;
        let num_blocks = ((self.num_amplitudes as u32) + threads_per_block - 1) / threads_per_block;

        // Allocate FP16 buffers as u16 (bit-compatible with half::f16)
        let mut state_fp16: CudaSlice<u16> = rt.alloc_zeros(self.num_amplitudes)?;
        let mut output_fp16: CudaSlice<u16> = rt.alloc_zeros(self.num_amplitudes)?;

        // Create cuBLAS handle
        let blas = CudaBlas::new(self.stream.clone())
            .map_err(|e| CudaError::LaunchFailed(format!("cuBLAS init: {:?}", e)))?;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_amps = self.num_amplitudes as u64;

        // ===== Process REAL part =====
        // Convert F64 real to FP16
        unsafe {
            self.stream
                .launch_builder(&self.kernels.convert_f64_to_f16)
                .arg(&self.real)
                .arg(&mut state_fp16)
                .arg(&n_amps)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        // FP16 GEMM using raw cuBLAS hgemm (Tensor Core accelerated)
        let alpha = f16::ONE;
        let beta = f16::ZERO;
        unsafe {
            let (gate_ptr, _g) = gate_gpu.device_ptr(&self.stream);
            let (state_ptr, _s) = state_fp16.device_ptr(&self.stream);
            let (output_ptr, _o) = output_fp16.device_ptr_mut(&self.stream);
            cublas_result::hgemm(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                128 as c_int,
                num_groups as c_int,
                128 as c_int,
                &alpha as *const f16,
                gate_ptr as *const f16,
                128 as c_int,
                state_ptr as *const f16,
                128 as c_int,
                &beta as *const f16,
                output_ptr as *mut f16,
                128 as c_int,
            )
            .map_err(|e| CudaError::LaunchFailed(format!("HGEMM real: {:?}", e)))?;
        }

        // Convert FP16 result back to F64
        unsafe {
            self.stream
                .launch_builder(&self.kernels.convert_f16_to_f64)
                .arg(&output_fp16)
                .arg(&mut self.real)
                .arg(&n_amps)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        // ===== Process IMAGINARY part =====
        // Convert F64 imag to FP16
        unsafe {
            self.stream
                .launch_builder(&self.kernels.convert_f64_to_f16)
                .arg(&self.imag)
                .arg(&mut state_fp16)
                .arg(&n_amps)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        // FP16 GEMM (Tensor Core accelerated)
        unsafe {
            let (gate_ptr, _g) = gate_gpu.device_ptr(&self.stream);
            let (state_ptr, _s) = state_fp16.device_ptr(&self.stream);
            let (output_ptr, _o) = output_fp16.device_ptr_mut(&self.stream);
            cublas_result::hgemm(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                128 as c_int,
                num_groups as c_int,
                128 as c_int,
                &alpha as *const f16,
                gate_ptr as *const f16,
                128 as c_int,
                state_ptr as *const f16,
                128 as c_int,
                &beta as *const f16,
                output_ptr as *mut f16,
                128 as c_int,
            )
            .map_err(|e| CudaError::LaunchFailed(format!("HGEMM imag: {:?}", e)))?;
        }

        // Convert FP16 result back to F64
        unsafe {
            self.stream
                .launch_builder(&self.kernels.convert_f16_to_f64)
                .arg(&output_fp16)
                .arg(&mut self.imag)
                .arg(&n_amps)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    /// Create GHZ state using FP16 Tensor Core fused GEMM for local gates
    ///
    /// Uses FP16 Tensor Cores for the 128×128 fused gate matrix multiply.
    /// Trades some precision for ~10-20x faster matrix operations.
    pub fn create_ghz_state_fused_gemm_fp16(&mut self, rt: &CudaRuntime) -> CudaResult<u64> {
        let start = std::time::Instant::now();

        // Reset to |0...0⟩
        self.reset_to_zero()?;

        // Apply fused H + 6 CNOTs via FP16 Tensor Core GEMM
        self.apply_fused_local_gemm_fp16(rt)?;

        // Cross-block CNOTs with regular F64 kernels
        for target_qubit in 7..self.n_qubits {
            self.apply_cnot_cross_block(target_qubit)?;
        }

        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    // =========================================================================
    // 256x256 FUSED GEMM (H + 7 CNOTs = 8 local qubits)
    // =========================================================================

    fn compute_fused_h_cnot_matrix_256_u16() -> Vec<u16> {
        let mut matrix = vec![0u16; 256 * 256];
        let rsqrt2 = f16::from_f64(std::f64::consts::FRAC_1_SQRT_2);
        let neg_rsqrt2 = f16::from_f64(-std::f64::consts::FRAC_1_SQRT_2);
        for input_idx in 0..256 {
            let upper_bits = input_idx & !1;
            let sign = if (input_idx & 1) == 0 {
                rsqrt2
            } else {
                neg_rsqrt2
            };
            let out_q0_zero = upper_bits;
            let out_q0_one = (upper_bits ^ 254) | 1;
            matrix[out_q0_zero + input_idx * 256] = rsqrt2.to_bits();
            matrix[out_q0_one + input_idx * 256] = sign.to_bits();
        }
        matrix
    }

    pub fn apply_fused_local_gemm_fp16_256(&mut self, rt: &CudaRuntime) -> CudaResult<u64> {
        let start = std::time::Instant::now();
        if self.n_qubits < 8 {
            return Err(CudaError::LaunchFailed(
                "Fused 256x256 GEMM requires at least 8 qubits".to_string(),
            ));
        }
        let fused_matrix = Self::compute_fused_h_cnot_matrix_256_u16();
        let gate_gpu: CudaSlice<u16> = rt.upload(&fused_matrix)?;
        let num_groups = self.num_amplitudes / 256;
        let threads_per_block = 256u32;
        let num_blocks = ((self.num_amplitudes as u32) + threads_per_block - 1) / threads_per_block;
        let mut state_fp16: CudaSlice<u16> = rt.alloc_zeros(self.num_amplitudes)?;
        let mut output_fp16: CudaSlice<u16> = rt.alloc_zeros(self.num_amplitudes)?;
        let blas = CudaBlas::new(self.stream.clone())
            .map_err(|e| CudaError::LaunchFailed(format!("cuBLAS init: {:?}", e)))?;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_amps = self.num_amplitudes as u64;
        let alpha = f16::ONE;
        let beta = f16::ZERO;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.convert_f64_to_f16)
                .arg(&self.real)
                .arg(&mut state_fp16)
                .arg(&n_amps)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }
        unsafe {
            let (gate_ptr, _g) = gate_gpu.device_ptr(&self.stream);
            let (state_ptr, _s) = state_fp16.device_ptr(&self.stream);
            let (output_ptr, _o) = output_fp16.device_ptr_mut(&self.stream);
            cublas_result::hgemm(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                256,
                num_groups as c_int,
                256,
                &alpha as *const f16,
                gate_ptr as *const f16,
                256,
                state_ptr as *const f16,
                256,
                &beta as *const f16,
                output_ptr as *mut f16,
                256,
            )
            .map_err(|e| CudaError::LaunchFailed(format!("HGEMM 256 real: {:?}", e)))?;
        }
        unsafe {
            self.stream
                .launch_builder(&self.kernels.convert_f16_to_f64)
                .arg(&output_fp16)
                .arg(&mut self.real)
                .arg(&n_amps)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }
        unsafe {
            self.stream
                .launch_builder(&self.kernels.convert_f64_to_f16)
                .arg(&self.imag)
                .arg(&mut state_fp16)
                .arg(&n_amps)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }
        unsafe {
            let (gate_ptr, _g) = gate_gpu.device_ptr(&self.stream);
            let (state_ptr, _s) = state_fp16.device_ptr(&self.stream);
            let (output_ptr, _o) = output_fp16.device_ptr_mut(&self.stream);
            cublas_result::hgemm(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                256,
                num_groups as c_int,
                256,
                &alpha as *const f16,
                gate_ptr as *const f16,
                256,
                state_ptr as *const f16,
                256,
                &beta as *const f16,
                output_ptr as *mut f16,
                256,
            )
            .map_err(|e| CudaError::LaunchFailed(format!("HGEMM 256 imag: {:?}", e)))?;
        }
        unsafe {
            self.stream
                .launch_builder(&self.kernels.convert_f16_to_f64)
                .arg(&output_fp16)
                .arg(&mut self.imag)
                .arg(&n_amps)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }
        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    pub fn create_ghz_state_fused_gemm_fp16_256(&mut self, rt: &CudaRuntime) -> CudaResult<u64> {
        let start = std::time::Instant::now();
        self.reset_to_zero()?;
        self.apply_fused_local_gemm_fp16_256(rt)?;
        for target_qubit in 8..self.n_qubits {
            self.apply_cnot_cross_block(target_qubit)?;
        }
        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    // =========================================================================
    // 512x512 FUSED GEMM (H + 8 CNOTs = 9 local qubits)
    // =========================================================================

    fn compute_fused_h_cnot_matrix_512_u16() -> Vec<u16> {
        let mut matrix = vec![0u16; 512 * 512];
        let rsqrt2 = f16::from_f64(std::f64::consts::FRAC_1_SQRT_2);
        let neg_rsqrt2 = f16::from_f64(-std::f64::consts::FRAC_1_SQRT_2);
        for input_idx in 0..512 {
            let upper_bits = input_idx & !1;
            let sign = if (input_idx & 1) == 0 {
                rsqrt2
            } else {
                neg_rsqrt2
            };
            let out_q0_zero = upper_bits;
            let out_q0_one = (upper_bits ^ 510) | 1;
            matrix[out_q0_zero + input_idx * 512] = rsqrt2.to_bits();
            matrix[out_q0_one + input_idx * 512] = sign.to_bits();
        }
        matrix
    }

    pub fn apply_fused_local_gemm_fp16_512(&mut self, rt: &CudaRuntime) -> CudaResult<u64> {
        let start = std::time::Instant::now();
        if self.n_qubits < 9 {
            return Err(CudaError::LaunchFailed(
                "Fused 512x512 GEMM requires at least 9 qubits".to_string(),
            ));
        }
        let fused_matrix = Self::compute_fused_h_cnot_matrix_512_u16();
        let gate_gpu: CudaSlice<u16> = rt.upload(&fused_matrix)?;
        let num_groups = self.num_amplitudes / 512;
        let threads_per_block = 256u32;
        let num_blocks = ((self.num_amplitudes as u32) + threads_per_block - 1) / threads_per_block;
        let mut state_fp16: CudaSlice<u16> = rt.alloc_zeros(self.num_amplitudes)?;
        let mut output_fp16: CudaSlice<u16> = rt.alloc_zeros(self.num_amplitudes)?;
        let blas = CudaBlas::new(self.stream.clone())
            .map_err(|e| CudaError::LaunchFailed(format!("cuBLAS init: {:?}", e)))?;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_amps = self.num_amplitudes as u64;
        let alpha = f16::ONE;
        let beta = f16::ZERO;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.convert_f64_to_f16)
                .arg(&self.real)
                .arg(&mut state_fp16)
                .arg(&n_amps)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }
        unsafe {
            let (gate_ptr, _g) = gate_gpu.device_ptr(&self.stream);
            let (state_ptr, _s) = state_fp16.device_ptr(&self.stream);
            let (output_ptr, _o) = output_fp16.device_ptr_mut(&self.stream);
            cublas_result::hgemm(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                512,
                num_groups as c_int,
                512,
                &alpha as *const f16,
                gate_ptr as *const f16,
                512,
                state_ptr as *const f16,
                512,
                &beta as *const f16,
                output_ptr as *mut f16,
                512,
            )
            .map_err(|e| CudaError::LaunchFailed(format!("HGEMM 512 real: {:?}", e)))?;
        }
        unsafe {
            self.stream
                .launch_builder(&self.kernels.convert_f16_to_f64)
                .arg(&output_fp16)
                .arg(&mut self.real)
                .arg(&n_amps)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }
        unsafe {
            self.stream
                .launch_builder(&self.kernels.convert_f64_to_f16)
                .arg(&self.imag)
                .arg(&mut state_fp16)
                .arg(&n_amps)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }
        unsafe {
            let (gate_ptr, _g) = gate_gpu.device_ptr(&self.stream);
            let (state_ptr, _s) = state_fp16.device_ptr(&self.stream);
            let (output_ptr, _o) = output_fp16.device_ptr_mut(&self.stream);
            cublas_result::hgemm(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                512,
                num_groups as c_int,
                512,
                &alpha as *const f16,
                gate_ptr as *const f16,
                512,
                state_ptr as *const f16,
                512,
                &beta as *const f16,
                output_ptr as *mut f16,
                512,
            )
            .map_err(|e| CudaError::LaunchFailed(format!("HGEMM 512 imag: {:?}", e)))?;
        }
        unsafe {
            self.stream
                .launch_builder(&self.kernels.convert_f16_to_f64)
                .arg(&output_fp16)
                .arg(&mut self.imag)
                .arg(&n_amps)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }
        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    pub fn create_ghz_state_fused_gemm_fp16_512(&mut self, rt: &CudaRuntime) -> CudaResult<u64> {
        let start = std::time::Instant::now();
        self.reset_to_zero()?;
        self.apply_fused_local_gemm_fp16_512(rt)?;
        for target_qubit in 9..self.n_qubits {
            self.apply_cnot_cross_block(target_qubit)?;
        }
        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    /// Create GHZ state using GEMM for CNOT(0,1)
    ///
    /// GEMM with consecutive 4-element groups can only apply gates to qubits 0,1.
    /// Uses GEMM for CNOT(0,1), then regular kernels for the rest.
    pub fn create_ghz_state_gemm(&mut self, rt: &CudaRuntime) -> CudaResult<u64> {
        let start = std::time::Instant::now();

        self.reset_to_zero()?;
        self.apply_hadamard_q0()?;

        // Use GEMM for CNOT(0,1) - the only CNOT amenable to simple GEMM
        if self.n_qubits >= 2 {
            self.apply_cnot_q0_q1_gemm(rt)?;
        }

        // Remaining local CNOTs (2-6) with regular kernels
        for target in 2..7.min(self.n_qubits) {
            self.apply_cnot_local(target as u8)?;
        }

        // Cross-block CNOTs with regular kernels
        for target_qubit in 7..self.n_qubits {
            self.apply_cnot_cross_block(target_qubit)?;
        }

        self.synchronize()?;
        Ok(start.elapsed().as_micros() as u64)
    }

    /// Create GHZ state: (|0...0⟩ + |1...1⟩)/√2
    ///
    /// Returns time in microseconds.
    pub fn create_ghz_state(&mut self) -> CudaResult<u64> {
        let start = std::time::Instant::now();

        // Reset to |0...0⟩
        self.reset_to_zero()?;

        // Apply H to qubit 0
        self.apply_hadamard_q0()?;

        // Apply CNOT chain: CNOT(0,1), CNOT(0,2), ..., CNOT(0,n-1)
        for target in 1..7.min(self.n_qubits) {
            self.apply_cnot_local(target as u8)?;
        }

        for target_qubit in 7..self.n_qubits {
            self.apply_cnot_cross_block(target_qubit)?;
        }

        // Synchronize to measure time accurately
        self.synchronize()?;

        Ok(start.elapsed().as_micros() as u64)
    }

    /// Create GHZ state using FUSED kernel (single launch, no gate simulation)
    ///
    /// This is mathematically equivalent to create_ghz_state() but eliminates
    /// kernel launch overhead by computing the final state directly.
    ///
    /// Performance: 1 kernel launch instead of n launches (30x reduction for 30 qubits).
    ///
    /// Returns time in microseconds.
    pub fn create_ghz_state_fused(&mut self) -> CudaResult<u64> {
        let start = std::time::Instant::now();

        let total_amps = self.num_amplitudes as u64;
        let all_ones_idx = (self.num_amplitudes - 1) as u64;

        let threads_per_block = 256u32;
        let num_blocks_launch = ((total_amps as u32) + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks_launch, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.ghz_fused)
                .arg(&self.real)
                .arg(&self.imag)
                .arg(&total_amps)
                .arg(&all_ones_idx)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        // Synchronize to measure time accurately
        self.synchronize()?;

        Ok(start.elapsed().as_micros() as u64)
    }

    /// Execute a batch of local gates (qubits 0-6) in a single kernel launch
    ///
    /// This is the GPU equivalent of MegaGate fusion - multiple gates are applied
    /// in sequence within a single kernel, eliminating kernel launch overhead.
    ///
    /// # Arguments
    /// * `rt` - CUDA runtime for memory allocation
    /// * `gates` - Slice of gate operations to execute
    ///
    /// # Returns
    /// Time in microseconds for the batched execution
    ///
    /// # Example
    /// ```ignore
    /// let gates = vec![
    ///     GateOp::h(0),           // H on qubit 0
    ///     GateOp::cnot(0, 1),     // CNOT(0,1)
    ///     GateOp::cnot(0, 2),     // CNOT(0,2)
    ///     GateOp::t(1),           // T gate on qubit 1
    /// ];
    /// let time = grid.apply_batched_gates(&rt, &gates)?;
    /// ```
    pub fn apply_batched_gates(&mut self, rt: &CudaRuntime, gates: &[GateOp]) -> CudaResult<u64> {
        if gates.is_empty() {
            return Ok(0);
        }

        let start = std::time::Instant::now();

        // Upload gates to GPU
        let gates_gpu: CudaSlice<GateOp> = rt.upload(gates)?;

        let num_gates = gates.len() as u32;
        let num_blocks_val = self.num_blocks as u64;

        // Launch with 128 threads per block (one per amplitude in quantum block)
        let cfg = LaunchConfig {
            grid_dim: (self.num_blocks as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.batched_gates_local)
                .arg(&self.real)
                .arg(&self.imag)
                .arg(&gates_gpu)
                .arg(&num_gates)
                .arg(&num_blocks_val)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("{:?}", e)))?;
        }

        self.synchronize()?;

        Ok(start.elapsed().as_micros() as u64)
    }

    /// Create GHZ state using batched gates (for benchmarking comparison)
    ///
    /// Uses the batched gate kernel for local qubits (0-6), then individual
    /// kernels for cross-block operations (qubits 7+).
    ///
    /// Returns time in microseconds.
    pub fn create_ghz_state_batched(&mut self, rt: &CudaRuntime) -> CudaResult<u64> {
        let start = std::time::Instant::now();

        // Reset to |0...0⟩
        self.reset_to_zero()?;

        // Build batch of local gates: H(0) + CNOT(0,1-6)
        let mut local_gates = Vec::new();
        local_gates.push(GateOp::h(0));
        for target in 1..7.min(self.n_qubits) {
            local_gates.push(GateOp::cnot(0, target as u8));
        }

        // Execute local gates in one kernel
        self.apply_batched_gates(rt, &local_gates)?;

        // Apply remaining cross-block CNOTs individually
        for target_qubit in 7..self.n_qubits {
            self.apply_cnot_cross_block(target_qubit)?;
        }

        self.synchronize()?;

        Ok(start.elapsed().as_micros() as u64)
    }

    /// Reset state to |0...0⟩
    pub fn reset_to_zero(&mut self) -> CudaResult<()> {
        // Zero out all amplitudes
        let zeros = vec![0.0f64; self.num_amplitudes];
        self.stream
            .memcpy_htod(&zeros, &mut self.real)
            .map_err(|e| CudaError::TransferFailed(format!("{:?}", e)))?;
        self.stream
            .memcpy_htod(&zeros, &mut self.imag)
            .map_err(|e| CudaError::TransferFailed(format!("{:?}", e)))?;

        // Set amplitude[0] = 1.0
        let mut init = vec![0.0f64; self.num_amplitudes];
        init[0] = 1.0;
        self.stream
            .memcpy_htod(&init, &mut self.real)
            .map_err(|e| CudaError::TransferFailed(format!("{:?}", e)))?;

        Ok(())
    }

    /// Synchronize GPU operations
    pub fn synchronize(&self) -> CudaResult<()> {
        self.stream
            .synchronize()
            .map_err(|e| CudaError::LaunchFailed(format!("Sync: {:?}", e)))
    }

    /// Download state to CPU and verify GHZ fidelity
    pub fn verify_ghz_fidelity(&self) -> CudaResult<GhzVerificationGpu> {
        // Download amplitudes
        let real: Vec<f64> = self
            .stream
            .clone_dtoh(&self.real)
            .map_err(|e| CudaError::TransferFailed(format!("{:?}", e)))?;
        let imag: Vec<f64> = self
            .stream
            .clone_dtoh(&self.imag)
            .map_err(|e| CudaError::TransferFailed(format!("{:?}", e)))?;

        // GHZ state: |0...0⟩ at index 0, |1...1⟩ at last index
        let amp_zero_re = real[0];
        let amp_zero_im = imag[0];
        let amp_zero = (amp_zero_re * amp_zero_re + amp_zero_im * amp_zero_im).sqrt();

        let last_idx = self.num_amplitudes - 1;
        let amp_one_re = real[last_idx];
        let amp_one_im = imag[last_idx];
        let amp_one = (amp_one_re * amp_one_re + amp_one_im * amp_one_im).sqrt();

        let prob_zero = amp_zero * amp_zero;
        let prob_one = amp_one * amp_one;
        let total_prob = prob_zero + prob_one;

        // Find max spurious amplitude
        let mut max_spurious = 0.0f64;
        for i in 0..self.num_amplitudes {
            if i == 0 || i == last_idx {
                continue;
            }
            let norm = (real[i] * real[i] + imag[i] * imag[i]).sqrt();
            if norm > max_spurious {
                max_spurious = norm;
            }
        }

        // Fidelity with ideal GHZ
        let ideal_amp = std::f64::consts::FRAC_1_SQRT_2;
        let overlap_re = amp_zero_re * ideal_amp + amp_one_re * ideal_amp;
        let overlap_im = amp_zero_im * ideal_amp + amp_one_im * ideal_amp;
        let fidelity = overlap_re * overlap_re + overlap_im * overlap_im;

        Ok(GhzVerificationGpu {
            n_qubits: self.n_qubits,
            amp_zero_state: amp_zero,
            amp_one_state: amp_one,
            prob_zero,
            prob_one,
            total_prob,
            fidelity,
            max_spurious,
            num_blocks: self.num_blocks,
        })
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.num_amplitudes * 16 // 2 * f64 per amplitude
    }
}

/// GHZ state verification results from GPU
#[derive(Debug, Clone)]
pub struct GhzVerificationGpu {
    pub n_qubits: usize,
    pub amp_zero_state: f64,
    pub amp_one_state: f64,
    pub prob_zero: f64,
    pub prob_one: f64,
    pub total_prob: f64,
    pub fidelity: f64,
    pub max_spurious: f64,
    pub num_blocks: usize,
}

impl std::fmt::Display for GhzVerificationGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GPU GHZ Verification ({} qubits)", self.n_qubits)?;
        writeln!(f, "  |0...0⟩ amplitude: {:.15}", self.amp_zero_state)?;
        writeln!(f, "  |1...1⟩ amplitude: {:.15}", self.amp_one_state)?;
        writeln!(f, "  Fidelity: {:.15}%", self.fidelity * 100.0)?;
        writeln!(f, "  Max spurious: {:.2e}", self.max_spurious)?;
        writeln!(f, "  Blocks: {}", self.num_blocks)?;
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_ghz_20_qubits() {
        let rt = match CudaRuntime::new() {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("CUDA not available: {:?}", e);
                return;
            }
        };

        let mut grid = QuantumGridF64Gpu::new(&rt, 20).expect("Failed to create grid");
        println!("\n20-Qubit GPU GHZ State:");
        println!("  Blocks: {}", grid.num_blocks);
        println!("  Memory: {} MB", grid.memory_bytes() / 1024 / 1024);

        let time_us = grid.create_ghz_state().expect("Failed to create GHZ");
        let verification = grid.verify_ghz_fidelity().expect("Failed to verify");

        println!("  Time: {:.2}ms", time_us as f64 / 1000.0);
        println!("{}", verification);

        assert!(
            verification.fidelity > 0.9999999,
            "Fidelity: {}",
            verification.fidelity
        );
    }

    #[test]
    fn test_gpu_ghz_25_qubits() {
        let rt = match CudaRuntime::new() {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("CUDA not available: {:?}", e);
                return;
            }
        };

        let mut grid = QuantumGridF64Gpu::new(&rt, 25).expect("Failed to create grid");
        println!("\n25-Qubit GPU GHZ State:");
        println!("  Blocks: {}", grid.num_blocks);
        println!("  Memory: {} MB", grid.memory_bytes() / 1024 / 1024);

        let time_us = grid.create_ghz_state().expect("Failed to create GHZ");
        let verification = grid.verify_ghz_fidelity().expect("Failed to verify");

        println!("  Time: {:.2}ms", time_us as f64 / 1000.0);
        println!("{}", verification);

        assert!(
            verification.fidelity > 0.9999999,
            "Fidelity: {}",
            verification.fidelity
        );
    }

    #[test]
    #[ignore] // Requires ~2GB VRAM
    fn test_gpu_ghz_27_qubits() {
        let rt = match CudaRuntime::new() {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("CUDA not available: {:?}", e);
                return;
            }
        };

        let mut grid = QuantumGridF64Gpu::new(&rt, 27).expect("Failed to create grid");
        println!("\n27-Qubit GPU GHZ State:");
        println!(
            "  Blocks: {} ({:.2}M)",
            grid.num_blocks,
            grid.num_blocks as f64 / 1e6
        );
        println!("  Memory: {} MB", grid.memory_bytes() / 1024 / 1024);

        let time_us = grid.create_ghz_state().expect("Failed to create GHZ");
        let verification = grid.verify_ghz_fidelity().expect("Failed to verify");

        println!("  Time: {:.2}ms", time_us as f64 / 1000.0);
        println!("{}", verification);

        assert!(
            verification.fidelity > 0.999999999999,
            "Fidelity: {}",
            verification.fidelity
        );
    }
}
