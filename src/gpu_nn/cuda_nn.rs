//! EPIC 94: GPU WMMA Neural Network Forward Pass
//!
//! GPU-accelerated batched neural network inference using CUDA Tensor Cores.
//!
//! Target: 100-1000× speedup over CPU per-organism processing
//!
//! Architecture: 8 inputs → 8 hidden (tanh) → 5 outputs (argmax)

use crate::batched_brain::BrainWeights;
use crate::classical_brain::{ClassicalBrain, Move};
use crate::cuda::{CudaError, CudaResult};

/// GPU neural network forward pass accelerator
pub struct GpuNNAccelerator {
    // Device pointers (allocated once, reused)
    d_sensors: *mut f32,
    d_weights_ih: *mut f32,
    d_bias_h: *mut f32,
    d_weights_ho: *mut f32,
    d_bias_o: *mut f32,
    d_moves: *mut i32,

    // Host-side move buffer
    h_moves: Vec<i32>,

    // Capacity (max organisms that can be processed)
    capacity: usize,

    // CUDA stream for async operations
    stream: cudaStream_t,
}

// Opaque CUDA types
#[allow(non_camel_case_types)]
type cudaStream_t = *mut std::ffi::c_void;

// Link to CUDA runtime library
#[cfg(target_os = "windows")]
#[link(name = "cudart")]
extern "C" {}

// External C functions from cuda_nn.cu
unsafe extern "C" {
    fn cuda_nn_forward(
        sensors: *const f32,
        weights_ih: *const f32,
        bias_h: *const f32,
        weights_ho: *const f32,
        bias_o: *const f32,
        moves: *mut i32,
        n_organisms: i32,
        stream: cudaStream_t,
    );

    fn cuda_check_last_error() -> i32;

    fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
        stream: cudaStream_t,
    ) -> i32;
    fn cudaStreamCreate(pStream: *mut cudaStream_t) -> i32;
    fn cudaStreamDestroy(stream: cudaStream_t) -> i32;
    fn cudaStreamSynchronize(stream: cudaStream_t) -> i32;
}

// CUDA memory copy kinds
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

impl GpuNNAccelerator {
    /// Create a new GPU NN accelerator with given capacity
    pub fn new(capacity: usize) -> CudaResult<Self> {
        unsafe {
            // Allocate device memory
            let mut d_sensors: *mut f32 = std::ptr::null_mut();
            let mut d_weights_ih: *mut f32 = std::ptr::null_mut();
            let mut d_bias_h: *mut f32 = std::ptr::null_mut();
            let mut d_weights_ho: *mut f32 = std::ptr::null_mut();
            let mut d_bias_o: *mut f32 = std::ptr::null_mut();
            let mut d_moves: *mut i32 = std::ptr::null_mut();

            let sensors_bytes = capacity * 8 * std::mem::size_of::<f32>();
            let weights_ih_bytes = 8 * 8 * std::mem::size_of::<f32>();
            let bias_h_bytes = 8 * std::mem::size_of::<f32>();
            let weights_ho_bytes = 8 * 5 * std::mem::size_of::<f32>();
            let bias_o_bytes = 5 * std::mem::size_of::<f32>();
            let moves_bytes = capacity * std::mem::size_of::<i32>();

            if cudaMalloc(
                &mut d_sensors as *mut *mut f32 as *mut *mut std::ffi::c_void,
                sensors_bytes,
            ) != 0
            {
                return Err(CudaError::AllocationFailed("sensors".to_string()));
            }
            if cudaMalloc(
                &mut d_weights_ih as *mut *mut f32 as *mut *mut std::ffi::c_void,
                weights_ih_bytes,
            ) != 0
            {
                cudaFree(d_sensors as *mut std::ffi::c_void);
                return Err(CudaError::AllocationFailed("weights_ih".to_string()));
            }
            if cudaMalloc(
                &mut d_bias_h as *mut *mut f32 as *mut *mut std::ffi::c_void,
                bias_h_bytes,
            ) != 0
            {
                cudaFree(d_sensors as *mut std::ffi::c_void);
                cudaFree(d_weights_ih as *mut std::ffi::c_void);
                return Err(CudaError::AllocationFailed("bias_h".to_string()));
            }
            if cudaMalloc(
                &mut d_weights_ho as *mut *mut f32 as *mut *mut std::ffi::c_void,
                weights_ho_bytes,
            ) != 0
            {
                cudaFree(d_sensors as *mut std::ffi::c_void);
                cudaFree(d_weights_ih as *mut std::ffi::c_void);
                cudaFree(d_bias_h as *mut std::ffi::c_void);
                return Err(CudaError::AllocationFailed("weights_ho".to_string()));
            }
            if cudaMalloc(
                &mut d_bias_o as *mut *mut f32 as *mut *mut std::ffi::c_void,
                bias_o_bytes,
            ) != 0
            {
                cudaFree(d_sensors as *mut std::ffi::c_void);
                cudaFree(d_weights_ih as *mut std::ffi::c_void);
                cudaFree(d_bias_h as *mut std::ffi::c_void);
                cudaFree(d_weights_ho as *mut std::ffi::c_void);
                return Err(CudaError::AllocationFailed("bias_o".to_string()));
            }
            if cudaMalloc(
                &mut d_moves as *mut *mut i32 as *mut *mut std::ffi::c_void,
                moves_bytes,
            ) != 0
            {
                cudaFree(d_sensors as *mut std::ffi::c_void);
                cudaFree(d_weights_ih as *mut std::ffi::c_void);
                cudaFree(d_bias_h as *mut std::ffi::c_void);
                cudaFree(d_weights_ho as *mut std::ffi::c_void);
                cudaFree(d_bias_o as *mut std::ffi::c_void);
                return Err(CudaError::AllocationFailed("moves".to_string()));
            }

            // Create CUDA stream
            let mut stream: cudaStream_t = std::ptr::null_mut();
            if cudaStreamCreate(&mut stream) != 0 {
                cudaFree(d_sensors as *mut std::ffi::c_void);
                cudaFree(d_weights_ih as *mut std::ffi::c_void);
                cudaFree(d_bias_h as *mut std::ffi::c_void);
                cudaFree(d_weights_ho as *mut std::ffi::c_void);
                cudaFree(d_bias_o as *mut std::ffi::c_void);
                cudaFree(d_moves as *mut std::ffi::c_void);
                return Err(CudaError::InitializationFailed(
                    "stream creation".to_string(),
                ));
            }

            Ok(Self {
                d_sensors,
                d_weights_ih,
                d_bias_h,
                d_weights_ho,
                d_bias_o,
                d_moves,
                h_moves: vec![0; capacity],
                capacity,
                stream,
            })
        }
    }

    /// Upload weights to GPU (call once when weights change)
    pub fn upload_weights(&mut self, weights: &BrainWeights) -> CudaResult<()> {
        unsafe {
            // Flatten weights for transfer
            let mut weights_ih_flat = Vec::with_capacity(64);
            let mut weights_ho_flat = Vec::with_capacity(40);

            for i in 0..8 {
                for j in 0..8 {
                    weights_ih_flat.push(weights.weights_ih[i][j]);
                }
            }

            for i in 0..8 {
                for j in 0..5 {
                    weights_ho_flat.push(weights.weights_ho[i][j]);
                }
            }

            // Upload to device
            if cudaMemcpy(
                self.d_weights_ih as *mut std::ffi::c_void,
                weights_ih_flat.as_ptr() as *const std::ffi::c_void,
                64 * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ) != 0
            {
                return Err(CudaError::TransferFailed("weights_ih upload".to_string()));
            }

            if cudaMemcpy(
                self.d_bias_h as *mut std::ffi::c_void,
                weights.bias_h.as_ptr() as *const std::ffi::c_void,
                8 * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ) != 0
            {
                return Err(CudaError::TransferFailed("bias_h upload".to_string()));
            }

            if cudaMemcpy(
                self.d_weights_ho as *mut std::ffi::c_void,
                weights_ho_flat.as_ptr() as *const std::ffi::c_void,
                40 * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ) != 0
            {
                return Err(CudaError::TransferFailed("weights_ho upload".to_string()));
            }

            if cudaMemcpy(
                self.d_bias_o as *mut std::ffi::c_void,
                weights.bias_o.as_ptr() as *const std::ffi::c_void,
                5 * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ) != 0
            {
                return Err(CudaError::TransferFailed("bias_o upload".to_string()));
            }

            Ok(())
        }
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

        unsafe {
            // Flatten sensors for transfer
            let mut sensors_flat = Vec::with_capacity(n * 8);
            for sensor in sensors {
                sensors_flat.extend_from_slice(sensor);
            }

            // Upload sensors
            if cudaMemcpyAsync(
                self.d_sensors as *mut std::ffi::c_void,
                sensors_flat.as_ptr() as *const std::ffi::c_void,
                n * 8 * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                self.stream,
            ) != 0
            {
                return Err(CudaError::TransferFailed("sensors upload".to_string()));
            }

            // Launch kernel
            cuda_nn_forward(
                self.d_sensors,
                self.d_weights_ih,
                self.d_bias_h,
                self.d_weights_ho,
                self.d_bias_o,
                self.d_moves,
                n as i32,
                self.stream,
            );

            // Check for kernel launch errors
            if cuda_check_last_error() != 0 {
                return Err(CudaError::LaunchFailed("nn_forward_kernel".to_string()));
            }

            // Download results
            if cudaMemcpyAsync(
                self.h_moves.as_mut_ptr() as *mut std::ffi::c_void,
                self.d_moves as *const std::ffi::c_void,
                n * std::mem::size_of::<i32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                self.stream,
            ) != 0
            {
                return Err(CudaError::TransferFailed("moves download".to_string()));
            }

            // Wait for stream to complete
            if cudaStreamSynchronize(self.stream) != 0 {
                return Err(CudaError::LaunchFailed("stream sync".to_string()));
            }

            // Convert move indices to Move enum
            let moves: Vec<Move> = self.h_moves[..n]
                .iter()
                .map(|&idx| Move::from_index(idx as usize))
                .collect();

            Ok(moves)
        }
    }
}

impl Drop for GpuNNAccelerator {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.d_sensors as *mut std::ffi::c_void);
            cudaFree(self.d_weights_ih as *mut std::ffi::c_void);
            cudaFree(self.d_bias_h as *mut std::ffi::c_void);
            cudaFree(self.d_weights_ho as *mut std::ffi::c_void);
            cudaFree(self.d_bias_o as *mut std::ffi::c_void);
            cudaFree(self.d_moves as *mut std::ffi::c_void);
            cudaStreamDestroy(self.stream);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Only run with CUDA available
    fn test_gpu_forward_matches_cpu() {
        // Create test data
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

        // CPU forward
        let cpu_moves = crate::batched_brain::forward_batch_shared_weights(&sensors, &weights);

        // GPU forward
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
