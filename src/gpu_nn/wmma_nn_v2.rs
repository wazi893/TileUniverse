//! EPIC 96 V2: WMMA Tensor Core Neural Network - REBUILT FROM SCRATCH
//!
//! Previous version: 30.6% correctness, 0.7× speed (FAILED)
//! Root causes identified:
//! 1. Fragment element mapping was wrong
//! 2. B matrix wasn't column-major
//! 3. Stride was original width, not padded width
//! 4. Direct fragment indexing instead of store/load through smem
//!
//! This version: Built from first principles with verified WMMA patterns

use crate::batched_brain::BrainWeights;
use crate::classical_brain::Move;
use crate::cuda::{CudaError, CudaResult, CudaRuntime};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig};
use cudarc::nvrtc::compile_ptx_with_opts;

// WMMA NN kernel V2 - CORRECT implementation
const WMMA_NN_KERNEL_V2: &str = r#"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// Fast tanh approximation
__device__ __forceinline__ float tanh_approx(float x) {
    if (x > 3.0f) return 1.0f;
    if (x < -3.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// WMMA-based NN forward pass
// Process 16 organisms at once using 16×16 tensor cores
extern "C" __global__ void wmma_nn_forward_v2(
    const half* __restrict__ sensors,     // [n_batches × 16 × 16] row-major, padded from [n × 8]
    const half* __restrict__ weights1,    // [16 × 16] COLUMN-MAJOR (for WMMA B matrix!)
    const half* __restrict__ bias1,       // [16] padded from [8]
    const half* __restrict__ weights2,    // [16 × 16] COLUMN-MAJOR
    const half* __restrict__ bias2,       // [16] padded from [5]
    int* __restrict__ moves,              // [n]
    int n_organisms
) {
    // Each warp handles 16 organisms
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int batch_start = warp_id * 16;

    if (batch_start >= n_organisms) return;

    // Shared memory for intermediate results
    // Layout: [float results for all warps][half intermediates for all warps]
    extern __shared__ float smem[];
    const int warp_idx = threadIdx.x / 32;
    float* my_smem = smem + warp_idx * 256;  // 256 floats per warp
    half* my_hidden_half = (half*)(smem + blockDim.x / 32 * 256) + warp_idx * 256;  // 256 halves per warp

    // Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> sensor_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> weight_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> result_frag;

    // ============================================================
    // LAYER 1: sensors[16×16] × weights1[16×16] → hidden[16×16]
    // ============================================================

    // Load sensor batch (16 organisms × 16 padded sensors)
    const half* sensor_ptr = sensors + warp_id * 256;  // Each batch is 16×16 = 256
    wmma::load_matrix_sync(sensor_frag, sensor_ptr, 16);  // Stride = 16 (padded width!)

    // Load weights (COLUMN-MAJOR for standard A×B)
    wmma::load_matrix_sync(weight_frag, weights1, 16);

    // Matrix multiply
    wmma::fill_fragment(result_frag, 0.0f);
    wmma::mma_sync(result_frag, sensor_frag, weight_frag, result_frag);

    // Store to shared memory (this handles fragment→memory mapping correctly!)
    wmma::store_matrix_sync(my_smem, result_frag, 16, wmma::mem_row_major);
    __syncwarp();

    // Apply bias + tanh in shared memory (only first 8 outputs are valid!)
    for (int i = lane_id; i < 16 * 8; i += 32) {
        int row = i / 8;   // Organism index within batch
        int col = i % 8;   // Hidden neuron index
        float val = my_smem[row * 16 + col] + __half2float(bias1[col]);
        my_smem[row * 16 + col] = tanh_approx(val);
    }
    // Zero out padding (columns 8-15)
    for (int i = lane_id; i < 16 * 8; i += 32) {
        int row = i / 8;
        int col = i % 8;
        my_smem[row * 16 + col + 8] = 0.0f;
    }
    __syncwarp();

    // Convert back to half for next layer (using pre-declared my_hidden_half pointer)
    for (int i = lane_id; i < 256; i += 32) {
        my_hidden_half[i] = __float2half(my_smem[i]);
    }
    __syncwarp();

    // ============================================================
    // LAYER 2: hidden[16×16] × weights2[16×16] → output[16×16]
    // ============================================================

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> hidden_frag;

    // Load hidden from shared memory
    wmma::load_matrix_sync(hidden_frag, my_hidden_half, 16);

    // Load output weights (COLUMN-MAJOR)
    wmma::load_matrix_sync(weight_frag, weights2, 16);

    // Matrix multiply
    wmma::fill_fragment(result_frag, 0.0f);
    wmma::mma_sync(result_frag, hidden_frag, weight_frag, result_frag);

    // Store to shared memory
    wmma::store_matrix_sync(my_smem, result_frag, 16, wmma::mem_row_major);
    __syncwarp();

    // ============================================================
    // OUTPUT: Apply bias + tanh, then argmax (only first 5 outputs!)
    // ============================================================

    for (int org = lane_id; org < 16; org += 32) {
        if (batch_start + org >= n_organisms) continue;

        // Apply bias + tanh to this organism's outputs
        float outputs[5];
        for (int o = 0; o < 5; o++) {
            float val = my_smem[org * 16 + o] + __half2float(bias2[o]);
            outputs[o] = tanh_approx(val);
        }

        // Argmax
        int best = 0;
        for (int o = 1; o < 5; o++) {
            if (outputs[o] > outputs[best]) {
                best = o;
            }
        }

        moves[batch_start + org] = best;
    }
}
"#;

/// Compile WMMA NN V2 kernel
fn compile_wmma_nn_v2_kernel(rt: &CudaRuntime) -> CudaResult<CudaFunction> {
    let cuda_include = std::env::var("CUDA_PATH")
        .map(|p| format!("{}/include", p))
        .unwrap_or_else(|_| {
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include".to_string()
        });

    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("sm_75"),
        include_paths: vec![cuda_include],
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(WMMA_NN_KERNEL_V2, opts)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("WMMA NN V2 kernel: {:?}", e)))?;

    let module = rt
        .ctx()
        .load_module(ptx)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("Module load: {:?}", e)))?;

    let kernel = module
        .load_function("wmma_nn_forward_v2")
        .map_err(|e| CudaError::KernelCompilationFailed(format!("Function load: {:?}", e)))?;

    Ok(kernel)
}

/// Helper: Convert BrainWeights to FP16 with CORRECT LAYOUT
///
/// KEY INSIGHT from CPU code analysis:
/// CPU computes: hidden[h] = Σ_inp sensor[inp] * weights_ih[h][inp]
/// This is: hidden = sensor × weights_ih^T (transpose!)
///
/// For WMMA C = A × B where A is sensors (row-major):
/// We need B to represent weights_ih^T, stored in column-major.
///
/// weights_ih is [hidden][input] = [h][inp]
/// weights_ih^T is [input][hidden] = [inp][h] = weights_ih[h][inp]
///
/// Column-major storage of weights_ih^T[inp][h]:
/// element at [inp][h] goes to index h*stride + inp
fn weights_to_fp16_v2(weights: &BrainWeights) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    // weights_ih: [h][inp] → need weights_ih^T in column-major
    // weights_ih^T[inp][h] = weights_ih[h][inp]
    // Col-major: [inp][h] at position h*16 + inp
    let mut weights1 = vec![0u16; 256];
    for h in 0..8 {
        for inp in 0..8 {
            // Transposed, column-major: position = h * 16 + inp
            let idx = h * 16 + inp;
            weights1[idx] = half::f16::from_f32(weights.weights_ih[h][inp]).to_bits();
        }
    }

    // bias_h: 8 → 16
    let mut bias1 = vec![0u16; 16];
    for i in 0..8 {
        bias1[i] = half::f16::from_f32(weights.bias_h[i]).to_bits();
    }

    // weights_ho: [h][o] → need weights_ho^T in column-major
    // weights_ho^T[h][o] → column-major at position o*16 + h
    // Wait, CPU: output[o] = Σ_h hidden[h] * weights_ho[h][o]
    // This is: output = hidden × weights_ho (NOT transposed!)
    // So B = weights_ho, column-major: [h][o] at position o*16 + h
    let mut weights2 = vec![0u16; 256];
    for h in 0..8 {
        for o in 0..5 {
            // Column-major: [h][o] at position o*16 + h
            let idx = o * 16 + h;
            weights2[idx] = half::f16::from_f32(weights.weights_ho[h][o]).to_bits();
        }
    }

    // bias_o: 5 → 16
    let mut bias2 = vec![0u16; 16];
    for i in 0..5 {
        bias2[i] = half::f16::from_f32(weights.bias_o[i]).to_bits();
    }

    (weights1, bias1, weights2, bias2)
}

/// WMMA GPU NN V2 - Rebuilt from scratch
pub struct WmmaGpuNNV2 {
    rt: CudaRuntime,
    kernel: CudaFunction,

    // GPU buffers
    d_sensors: CudaSlice<u16>, // [capacity × 16] padded, stored as u16 (f16 bits)
    d_weights1: CudaSlice<u16>, // [16 × 16] column-major
    d_bias1: CudaSlice<u16>,   // [16]
    d_weights2: CudaSlice<u16>, // [16 × 16] column-major
    d_bias2: CudaSlice<u16>,   // [16]
    d_moves: CudaSlice<i32>,   // [capacity]

    capacity: usize,
    active_count: usize,
}

impl WmmaGpuNNV2 {
    /// Create new WMMA NN V2 accelerator
    pub fn new(capacity: usize) -> CudaResult<Self> {
        let rt = CudaRuntime::new()?;
        let kernel = compile_wmma_nn_v2_kernel(&rt)?;

        // Allocate buffers
        let d_sensors = rt.alloc_zeros::<u16>(capacity * 16)?; // Each organism: 16 padded sensors
        let d_weights1 = rt.alloc_zeros::<u16>(256)?; // 16×16
        let d_bias1 = rt.alloc_zeros::<u16>(16)?;
        let d_weights2 = rt.alloc_zeros::<u16>(256)?; // 16×16
        let d_bias2 = rt.alloc_zeros::<u16>(16)?;
        let d_moves = rt.alloc_zeros::<i32>(capacity)?;

        Ok(Self {
            rt,
            kernel,
            d_sensors,
            d_weights1,
            d_bias1,
            d_weights2,
            d_bias2,
            d_moves,
            capacity,
            active_count: 0,
        })
    }

    /// Upload weights (call once per weights change)
    pub fn upload_weights(&mut self, weights: &BrainWeights) -> CudaResult<()> {
        let (w1, b1, w2, b2) = weights_to_fp16_v2(weights);

        self.d_weights1 = self.rt.upload(&w1)?;
        self.d_bias1 = self.rt.upload(&b1)?;
        self.d_weights2 = self.rt.upload(&w2)?;
        self.d_bias2 = self.rt.upload(&b2)?;

        Ok(())
    }

    /// Upload all sensors (call once at initialization or when sensors change)
    pub fn upload_all_sensors(&mut self, sensors: &[[f32; 8]]) -> CudaResult<()> {
        self.active_count = sensors.len();

        // Pad sensors: each organism gets 16 slots, only first 8 are filled
        let n_batches = (sensors.len() + 15) / 16;
        let mut sensors_padded = vec![0u16; n_batches * 16 * 16];

        for (i, sensor) in sensors.iter().enumerate() {
            let batch = i / 16;
            let row = i % 16;
            let base = batch * 256 + row * 16; // Row-major: row * stride

            for (col, &val) in sensor.iter().enumerate() {
                sensors_padded[base + col] = half::f16::from_f32(val).to_bits();
            }
            // Columns 8-15 stay zero (padding)
        }

        self.d_sensors = self.rt.upload(&sensors_padded)?;

        Ok(())
    }

    /// Forward pass (kernel + no download)
    pub fn forward_persistent(&mut self) -> CudaResult<()> {
        let n_batches = (self.active_count + 15) / 16;
        let n_warps = n_batches;

        // Launch config: one warp per batch
        let threads_per_block = 128; // 4 warps per block
        let blocks = (n_warps + 3) / 4;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 4 * 256 * 4 + 4 * 256 * 2, // 4 warps × (256 floats × 4 bytes + 256 halves × 2 bytes)
        };

        let n_i32 = self.active_count as i32;

        use cudarc::driver::PushKernelArg;
        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernel)
                .arg(&self.d_sensors)
                .arg(&self.d_weights1)
                .arg(&self.d_bias1)
                .arg(&self.d_weights2)
                .arg(&self.d_bias2)
                .arg(&self.d_moves)
                .arg(&n_i32)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("WMMA V2: {:?}", e)))?;
        }

        Ok(())
    }

    /// Download moves
    pub fn download_moves(&self) -> CudaResult<Vec<Move>> {
        let moves_raw: Vec<i32> = self.rt.download(&self.d_moves)?;

        let moves: Vec<Move> = moves_raw[0..self.active_count]
            .iter()
            .map(|&m| match m {
                0 => Move::Up,
                1 => Move::Down,
                2 => Move::Left,
                3 => Move::Right,
                _ => Move::Stay,
            })
            .collect();

        Ok(moves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weights_column_major() {
        let weights = BrainWeights::random(12345);
        let (w1, _b1, _w2, _b2) = weights_to_fp16_v2(&weights);

        // Verify transposed column-major layout for WMMA B matrix
        // B[inp][h] = weights_ih[h][inp], stored at position h*16 + inp
        // So weights_ih[row][col] should be at w1[row*16 + col]
        for row in 0..8 {
            for col in 0..8 {
                let idx = row * 16 + col;
                let stored_bits = w1[idx];
                let stored_val = half::f16::from_bits(stored_bits).to_f32();
                let expected_val = weights.weights_ih[row][col];

                assert!(
                    (stored_val - expected_val).abs() < 0.01,
                    "Col-major check failed at [{},{}]: expected {}, got {}",
                    row,
                    col,
                    expected_val,
                    stored_val
                );
            }
        }
    }
}
