//! EPIC 94: GPU Neural Network using cudarc NVRTC
//!
//! Runtime-compiled CUDA kernels - no build script needed!

use crate::batched_brain::BrainWeights;
use crate::classical_brain::Move;
use crate::cuda::{CudaError, CudaResult, CudaRuntime};
use cudarc::driver::{CudaFunction, CudaSlice};
use cudarc::nvrtc::compile_ptx_with_opts;

// CUDA kernel source (compiled at runtime via NVRTC)
const NN_KERNEL_SRC: &str = r#"
// Fast tanh approximation (matches CPU implementation)
__device__ __forceinline__ float tanh_approx(float x) {
    if (x > 3.0f) return 1.0f;
    if (x < -3.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// Batched NN forward pass kernel
extern "C" __global__ void nn_forward_kernel(
    const float* __restrict__ sensors,      // [n × 8]
    const float* __restrict__ weights_ih,   // [8 × 8] (row-major)
    const float* __restrict__ bias_h,       // [8]
    const float* __restrict__ weights_ho,   // [8 × 5] (row-major: weights_ho[h][o])
    const float* __restrict__ bias_o,       // [5]
    int* __restrict__ moves,                // [n] output
    int n_organisms
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_organisms) return;

    // Layer 1: Input → Hidden (with tanh activation)
    float hidden[8];
    for (int h = 0; h < 8; h++) {
        float sum = bias_h[h];
        for (int i = 0; i < 8; i++) {
            sum += sensors[tid * 8 + i] * weights_ih[h * 8 + i];
        }
        hidden[h] = tanh_approx(sum);
    }

    // Layer 2: Hidden → Output (with tanh activation)
    float output[5];
    for (int o = 0; o < 5; o++) {
        float sum = bias_o[o];
        for (int h = 0; h < 8; h++) {
            sum += hidden[h] * weights_ho[h * 5 + o];
        }
        output[o] = tanh_approx(sum);
    }

    // Argmax to get move index
    int max_idx = 0;
    float max_val = output[0];
    for (int i = 1; i < 5; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }

    moves[tid] = max_idx;
}
"#;

/// Compiled NN kernel function
struct NNKernels {
    forward_fn: CudaFunction,
}

/// Compile NN kernels using NVRTC
fn compile_nn_kernels(rt: &CudaRuntime) -> CudaResult<NNKernels> {
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("sm_75"), // Turing+ for best performance
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(NN_KERNEL_SRC, opts).map_err(|e| {
        CudaError::KernelCompilationFailed(format!("NN kernel compile error: {:?}", e))
    })?;

    // Load module and get function
    let module = rt
        .ctx()
        .load_module(ptx)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("Module load: {:?}", e)))?;

    let forward_fn = module
        .load_function("nn_forward_kernel")
        .map_err(|e| CudaError::KernelCompilationFailed(format!("Function load: {:?}", e)))?;

    Ok(NNKernels { forward_fn })
}

/// GPU neural network accelerator
pub struct GpuNNAccelerator {
    rt: CudaRuntime,
    kernels: NNKernels,

    // Device buffers (persistent)
    d_sensors: CudaSlice<f32>,
    d_weights_ih: CudaSlice<f32>,
    d_bias_h: CudaSlice<f32>,
    d_weights_ho: CudaSlice<f32>,
    d_bias_o: CudaSlice<f32>,
    d_moves: CudaSlice<i32>,

    // Host buffer for results
    h_moves: Vec<i32>,

    capacity: usize,
}

impl GpuNNAccelerator {
    /// Create new GPU NN accelerator
    pub fn new(capacity: usize) -> CudaResult<Self> {
        let rt = CudaRuntime::new()?;
        let kernels = compile_nn_kernels(&rt)?;

        // Allocate device memory
        let d_sensors = rt.alloc_zeros::<f32>(capacity * 8)?;
        let d_weights_ih = rt.alloc_zeros::<f32>(64)?;
        let d_bias_h = rt.alloc_zeros::<f32>(8)?;
        let d_weights_ho = rt.alloc_zeros::<f32>(40)?;
        let d_bias_o = rt.alloc_zeros::<f32>(5)?;
        let d_moves = rt.alloc_zeros::<i32>(capacity)?;

        Ok(Self {
            rt,
            kernels,
            d_sensors,
            d_weights_ih,
            d_bias_h,
            d_weights_ho,
            d_bias_o,
            d_moves,
            h_moves: vec![0; capacity],
            capacity,
        })
    }

    /// Upload weights to GPU (call when weights change)
    pub fn upload_weights(&mut self, weights: &BrainWeights) -> CudaResult<()> {
        // Flatten weights
        let mut weights_ih_flat = Vec::with_capacity(64);
        for i in 0..8 {
            for j in 0..8 {
                weights_ih_flat.push(weights.weights_ih[i][j]);
            }
        }

        let mut weights_ho_flat = Vec::with_capacity(40);
        for i in 0..8 {
            for j in 0..5 {
                weights_ho_flat.push(weights.weights_ho[i][j]);
            }
        }

        // Upload to device (using stream copy_into)
        self.d_weights_ih = self.rt.upload(&weights_ih_flat)?;
        self.d_bias_h = self.rt.upload(&weights.bias_h)?;
        self.d_weights_ho = self.rt.upload(&weights_ho_flat)?;
        self.d_bias_o = self.rt.upload(&weights.bias_o)?;

        Ok(())
    }

    /// Batched forward pass on GPU
    pub fn forward_batch(&mut self, sensors: &[[f32; 8]]) -> CudaResult<Vec<Move>> {
        let n = sensors.len();
        if n > self.capacity {
            return Err(CudaError::InvalidConfig(format!(
                "Batch size {} exceeds capacity {}",
                n, self.capacity
            )));
        }

        // Flatten sensors
        let mut sensors_flat = Vec::with_capacity(n * 8);
        for sensor in sensors {
            sensors_flat.extend_from_slice(sensor);
        }

        // Upload sensors
        self.d_sensors = self.rt.upload(&sensors_flat)?;

        // Launch kernel
        let threads = 256;
        let blocks = (n as u32 + threads - 1) / threads;
        let n_i32 = n as i32;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        use cudarc::driver::PushKernelArg;
        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernels.forward_fn)
                .arg(&self.d_sensors)
                .arg(&self.d_weights_ih)
                .arg(&self.d_bias_h)
                .arg(&self.d_weights_ho)
                .arg(&self.d_bias_o)
                .arg(&self.d_moves)
                .arg(&n_i32)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("Kernel launch failed: {:?}", e)))?;
        }

        // Download results
        self.h_moves = self.rt.download(&self.d_moves)?;

        // Convert to Move enum
        let moves: Vec<Move> = self.h_moves[..n]
            .iter()
            .map(|&idx| Move::from_index(idx as usize))
            .collect();

        Ok(moves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Only run with CUDA available
    fn test_gpu_forward_matches_cpu() {
        let n = 100;
        let mut sensors = Vec::with_capacity(n);
        for i in 0..n {
            sensors.push([
                (i as f32 / n as f32) * 2.0 - 1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]);
        }

        let weights = BrainWeights::random(12345);

        // CPU baseline
        let cpu_moves = crate::batched_brain::forward_batch_shared_weights(&sensors, &weights);

        // GPU
        let mut gpu_acc = GpuNNAccelerator::new(1000).expect("Failed to create GPU accelerator");
        gpu_acc
            .upload_weights(&weights)
            .expect("Failed to upload weights");
        let gpu_moves = gpu_acc.forward_batch(&sensors).expect("GPU forward failed");

        // Compare
        assert_eq!(cpu_moves.len(), gpu_moves.len());
        for (i, (cpu, gpu)) in cpu_moves.iter().zip(gpu_moves.iter()).enumerate() {
            assert_eq!(cpu, gpu, "Move mismatch at organism {}", i);
        }
    }
}
