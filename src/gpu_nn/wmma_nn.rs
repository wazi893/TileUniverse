//! EPIC 96: WMMA Tensor Core Neural Network
//!
//! Uses FP16 tensor cores for 5-10× kernel speedup over EPIC 95.1.
//! Target: 3-4× batched workload speedup (realistic, with downloads).
//!
//! IMPORTANT: This is kernel optimization only. Download time (2.22ms) unchanged.

use crate::batched_brain::BrainWeights;
use crate::classical_brain::Move;
use crate::cuda::{CudaError, CudaResult, CudaRuntime};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig};
use cudarc::nvrtc::compile_ptx_with_opts;

// WMMA NN kernel - uses tensor cores for matrix multiplication
const WMMA_NN_KERNEL_SRC: &str = r#"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// Fast tanh approximation (FP32 version for after WMMA)
__device__ __forceinline__ float tanh_approx(float x) {
    if (x > 3.0f) return 1.0f;
    if (x < -3.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// WMMA-based NN forward pass
// Each warp processes 16 organisms using 16×16 tensor cores
extern "C" __global__ void wmma_nn_forward(
    const half* __restrict__ sensors,      // [n × 16] (padded from 8)
    const half* __restrict__ weights_ih,   // [16 × 16] (padded from 8×8)
    const half* __restrict__ bias_h,       // [16] (padded from 8)
    const half* __restrict__ weights_ho,   // [16 × 16] (padded from 8×5)
    const half* __restrict__ bias_o,       // [16] (padded from 5)
    int* __restrict__ moves,               // [n]
    int n_organisms
) {
    // Each warp handles 16 organisms
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int org_base = warp_id * 16;

    if (org_base >= n_organisms) return;

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> sensor_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> weight_ih_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> hidden_frag;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> hidden_half_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> weight_ho_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> output_frag;

    // ==================================================================
    // Layer 1: Input → Hidden (16×16 matmul via tensor cores)
    // ==================================================================

    // Load sensor tile (16 organisms × 16 inputs, row-major)
    wmma::load_matrix_sync(sensor_frag, sensors + org_base * 16, 16);

    // Load weight matrix (16×16, column-major for efficiency)
    wmma::load_matrix_sync(weight_ih_frag, weights_ih, 16);

    // Matrix multiply: hidden = sensors × weights_ih
    wmma::fill_fragment(hidden_frag, 0.0f);
    wmma::mma_sync(hidden_frag, sensor_frag, weight_ih_frag, hidden_frag);

    // Add bias and apply tanh (in FP32 accumulator)
    // Each thread in warp owns hidden_frag.num_elements values
    for (int i = 0; i < hidden_frag.num_elements; i++) {
        int col = (lane_id * hidden_frag.num_elements + i) % 16;
        float biased = hidden_frag.x[i] + __half2float(bias_h[col]);
        hidden_frag.x[i] = tanh_approx(biased);
    }

    // Convert hidden_frag (FP32) back to FP16 for next layer
    // Store to shared memory temporarily
    __shared__ half hidden_shared[32 * 16 * 16];  // Max 32 warps per block
    half* my_hidden = hidden_shared + (threadIdx.x / 32) * 256;

    // Convert and store
    for (int i = 0; i < hidden_frag.num_elements; i++) {
        int idx = lane_id * hidden_frag.num_elements + i;
        my_hidden[idx] = __float2half(hidden_frag.x[i]);
    }
    __syncwarp();

    // Load hidden as FP16 fragment
    wmma::load_matrix_sync(hidden_half_frag, my_hidden, 16);

    // ==================================================================
    // Layer 2: Hidden → Output (16×16 matmul via tensor cores)
    // ==================================================================

    // Load output weights (16×16, column-major)
    wmma::load_matrix_sync(weight_ho_frag, weights_ho, 16);

    // Matrix multiply: output = hidden × weights_ho
    wmma::fill_fragment(output_frag, 0.0f);
    wmma::mma_sync(output_frag, hidden_half_frag, weight_ho_frag, output_frag);

    // Add bias and apply tanh
    for (int i = 0; i < output_frag.num_elements; i++) {
        int col = (lane_id * output_frag.num_elements + i) % 16;
        float biased = output_frag.x[i] + __half2float(bias_o[col]);
        output_frag.x[i] = tanh_approx(biased);
    }

    // ==================================================================
    // Argmax: Find move index (use only first 5 outputs)
    // ==================================================================

    // Store output to shared memory for reduction
    __shared__ float output_shared[32 * 16 * 16];
    float* my_output = output_shared + (threadIdx.x / 32) * 256;

    for (int i = 0; i < output_frag.num_elements; i++) {
        int idx = lane_id * output_frag.num_elements + i;
        my_output[idx] = output_frag.x[i];
    }
    __syncwarp();

    // Each thread handles one organism's argmax
    if (lane_id < 16 && (org_base + lane_id) < n_organisms) {
        float* org_output = my_output + lane_id * 16;

        // Argmax over first 5 outputs only
        int max_idx = 0;
        float max_val = org_output[0];
        for (int i = 1; i < 5; i++) {
            if (org_output[i] > max_val) {
                max_val = org_output[i];
                max_idx = i;
            }
        }

        moves[org_base + lane_id] = max_idx;
    }
}
"#;

/// Compile WMMA NN kernel
fn compile_wmma_nn_kernel(rt: &CudaRuntime) -> CudaResult<CudaFunction> {
    // Get CUDA include path
    let cuda_include = std::env::var("CUDA_PATH")
        .map(|p| format!("{}/include", p))
        .unwrap_or_else(|_| {
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include".to_string()
        });

    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("sm_75"), // Turing+ for tensor cores
        include_paths: vec![cuda_include],
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(WMMA_NN_KERNEL_SRC, opts)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("WMMA NN kernel: {:?}", e)))?;

    let module = rt
        .ctx()
        .load_module(ptx)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("Module load: {:?}", e)))?;

    let kernel = module
        .load_function("wmma_nn_forward")
        .map_err(|e| CudaError::KernelCompilationFailed(format!("Function load: {:?}", e)))?;

    Ok(kernel)
}

/// Helper: Convert BrainWeights to FP16 and pad to 16×16
fn weights_to_fp16_padded(weights: &BrainWeights) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    // weights_ih: 8×8 → 16×16 (column-major for WMMA efficiency)
    let mut weights_ih_fp16 = vec![0u16; 256];
    for i in 0..8 {
        for j in 0..8 {
            let val = weights.weights_ih[i][j];
            let fp16_bits = half::f16::from_f32(val).to_bits();
            weights_ih_fp16[j * 16 + i] = fp16_bits; // Column-major
        }
    }

    // bias_h: 8 → 16
    let mut bias_h_fp16 = vec![0u16; 16];
    for i in 0..8 {
        bias_h_fp16[i] = half::f16::from_f32(weights.bias_h[i]).to_bits();
    }

    // weights_ho: 8×5 → 16×16 (column-major)
    let mut weights_ho_fp16 = vec![0u16; 256];
    for i in 0..8 {
        for j in 0..5 {
            let val = weights.weights_ho[i][j];
            let fp16_bits = half::f16::from_f32(val).to_bits();
            weights_ho_fp16[j * 16 + i] = fp16_bits; // Column-major
        }
    }

    // bias_o: 5 → 16
    let mut bias_o_fp16 = vec![0u16; 16];
    for i in 0..5 {
        bias_o_fp16[i] = half::f16::from_f32(weights.bias_o[i]).to_bits();
    }

    (weights_ih_fp16, bias_h_fp16, weights_ho_fp16, bias_o_fp16)
}

/// WMMA Tensor Core NN Accelerator
///
/// Uses FP16 tensor cores for matrix multiplication.
/// Target: 5-10× kernel speedup over EPIC 95.1.
pub struct WmmaGpuNN {
    rt: CudaRuntime,
    kernel: CudaFunction,

    // GPU buffers (FP16 for WMMA, padded to 16)
    d_sensors: CudaSlice<u16>,    // [capacity × 16] (padded from 8)
    d_weights_ih: CudaSlice<u16>, // [16 × 16]
    d_bias_h: CudaSlice<u16>,     // [16]
    d_weights_ho: CudaSlice<u16>, // [16 × 16]
    d_bias_o: CudaSlice<u16>,     // [16]
    d_moves: CudaSlice<i32>,      // [capacity]

    capacity: usize,
    active_count: usize,
}

impl WmmaGpuNN {
    /// Create new WMMA tensor core NN accelerator
    pub fn new(capacity: usize) -> CudaResult<Self> {
        let rt = CudaRuntime::new()?;
        let kernel = compile_wmma_nn_kernel(&rt)?;

        // Allocate buffers (padded to 16 for WMMA)
        let d_sensors = rt.alloc_zeros::<u16>(capacity * 16)?;
        let d_weights_ih = rt.alloc_zeros::<u16>(256)?; // 16×16
        let d_bias_h = rt.alloc_zeros::<u16>(16)?;
        let d_weights_ho = rt.alloc_zeros::<u16>(256)?; // 16×16
        let d_bias_o = rt.alloc_zeros::<u16>(16)?;
        let d_moves = rt.alloc_zeros::<i32>(capacity)?;

        Ok(Self {
            rt,
            kernel,
            d_sensors,
            d_weights_ih,
            d_bias_h,
            d_weights_ho,
            d_bias_o,
            d_moves,
            capacity,
            active_count: 0,
        })
    }

    /// Upload weights (FP16, padded to 16×16)
    pub fn upload_weights(&mut self, weights: &BrainWeights) -> CudaResult<()> {
        let (w_ih, b_h, w_ho, b_o) = weights_to_fp16_padded(weights);

        self.d_weights_ih = self.rt.upload(&w_ih)?;
        self.d_bias_h = self.rt.upload(&b_h)?;
        self.d_weights_ho = self.rt.upload(&w_ho)?;
        self.d_bias_o = self.rt.upload(&b_o)?;

        Ok(())
    }

    /// Upload sensors (convert FP32 → FP16, pad to 16)
    pub fn upload_all_sensors(&mut self, sensors: &[[f32; 8]]) -> CudaResult<()> {
        if sensors.len() > self.capacity {
            return Err(CudaError::InvalidConfig(format!(
                "Sensor count {} exceeds capacity {}",
                sensors.len(),
                self.capacity
            )));
        }

        // Convert and pad: 8 → 16
        let mut sensors_fp16 = Vec::with_capacity(sensors.len() * 16);
        for sensor in sensors {
            for &val in sensor {
                sensors_fp16.push(half::f16::from_f32(val).to_bits());
            }
            // Pad with zeros to 16
            sensors_fp16.extend_from_slice(&[0u16; 8]);
        }

        self.d_sensors = self.rt.upload(&sensors_fp16)?;
        self.active_count = sensors.len();

        Ok(())
    }

    /// Forward pass using WMMA tensor cores (zero transfers)
    pub fn forward_persistent(&mut self) -> CudaResult<()> {
        if self.active_count == 0 {
            return Ok(());
        }

        // Each warp handles 16 organisms
        let organisms_per_warp = 16;
        let num_warps = (self.active_count + organisms_per_warp - 1) / organisms_per_warp;

        // Launch config: 32 threads per warp, 8 warps per block
        let warps_per_block = 8;
        let threads_per_block = warps_per_block * 32;
        let num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = self.active_count as i32;

        use cudarc::driver::PushKernelArg;
        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernel)
                .arg(&self.d_sensors)
                .arg(&self.d_weights_ih)
                .arg(&self.d_bias_h)
                .arg(&self.d_weights_ho)
                .arg(&self.d_bias_o)
                .arg(&self.d_moves)
                .arg(&n_i32)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("WMMA kernel launch: {:?}", e)))?;
        }

        Ok(())
    }

    /// Download moves (lazy sync point)
    pub fn download_moves(&self) -> CudaResult<Vec<Move>> {
        let moves_raw = self.rt.download(&self.d_moves)?;

        let moves: Vec<Move> = moves_raw[..self.active_count]
            .iter()
            .map(|&idx| Move::from_index(idx as usize))
            .collect();

        Ok(moves)
    }

    pub fn active_count(&self) -> usize {
        self.active_count
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA + tensor cores
    fn test_wmma_matches_cpu() {
        use crate::batched_brain::forward_batch_shared_weights;

        let n = 1000;
        let sensors: Vec<[f32; 8]> = (0..n)
            .map(|i| {
                let val = (i as f32 / n as f32) * 2.0 - 1.0;
                [val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            })
            .collect();

        let weights = BrainWeights::random(12345);

        // CPU baseline
        let cpu_moves = forward_batch_shared_weights(&sensors, &weights);

        // WMMA
        let mut wmma = WmmaGpuNN::new(1000).expect("Failed to create WMMA GPU");
        wmma.upload_weights(&weights)
            .expect("Upload weights failed");
        wmma.upload_all_sensors(&sensors)
            .expect("Upload sensors failed");
        wmma.forward_persistent().expect("WMMA forward failed");
        let wmma_moves = wmma.download_moves().expect("Download failed");

        // Verify >99% agreement (FP16 precision may cause minor differences)
        let matches = cpu_moves
            .iter()
            .zip(wmma_moves.iter())
            .filter(|(cpu, wmma)| cpu == wmma)
            .count();

        let agreement = matches as f64 / cpu_moves.len() as f64;
        assert!(
            agreement > 0.99,
            "WMMA should match CPU >99%, got {:.2}%",
            agreement * 100.0
        );
    }
}
