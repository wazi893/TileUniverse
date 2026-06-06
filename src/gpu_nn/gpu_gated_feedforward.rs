//! EPIC 102: GPU-Accelerated GatedFeedforward Neural Network
//!
//! Winner of EPIC 101.5: GatedFeedforward beats LSTM by 39% on memory tasks.
//! This module provides GPU-accelerated inference for populations of organisms,
//! where each organism has its OWN weights (per-organism evolution).
//!
//! Architecture: input(8) → gated_hidden(24) → output(5)
//! Parameters per organism: 557
//!
//! Memory per organism: 557 × 4 bytes = 2,228 bytes (~2.2KB)
//! 1M organisms = 2.2GB (fits RTX 4070 12GB)
//!
//! This implementation uses a simple CUDA kernel (one thread per organism)
//! for correctness and debuggability before any WMMA optimization.

#[cfg(feature = "cuda")]
use crate::cuda::{CudaError, CudaResult, CudaRuntime};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx_with_opts;

use crate::brain::gated_feedforward::{GatedFeedforwardPopulation, GatedFeedforwardWeights};

/// Simple CUDA kernel for GatedFeedforward (one thread per organism)
///
/// Each organism has its OWN weights. Thread i processes organism i using:
/// - sensors[i * 8 .. i * 8 + 8]
/// - Wg[i * 8 * 24 .. i * 8 * 24 + 192], etc.
#[cfg(feature = "cuda")]
const GATED_FF_KERNEL: &str = r#"
// GatedFeedforward forward pass
// Architecture: 8 → 24 (gated) → 5
// Each thread processes ONE organism with its OWN weights

extern "C" __global__ void gated_ff_forward(
    const float* __restrict__ sensors,    // [N × 8] sensor inputs
    const float* __restrict__ Wg,         // [N × 8 × 24] gate weights
    const float* __restrict__ bg,         // [N × 24] gate biases
    const float* __restrict__ Wh,         // [N × 8 × 24] hidden weights
    const float* __restrict__ bh,         // [N × 24] hidden biases
    const float* __restrict__ Wo,         // [N × 24 × 5] output weights
    const float* __restrict__ bo,         // [N × 5] output biases
    int* __restrict__ actions,            // [N] output actions (argmax)
    int n_organisms
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_organisms) return;

    // Constants
    const int INPUT_SIZE = 8;
    const int HIDDEN_SIZE = 24;
    const int OUTPUT_SIZE = 5;

    // Per-organism weight offsets
    const int sensor_offset = idx * INPUT_SIZE;
    const int wg_offset = idx * INPUT_SIZE * HIDDEN_SIZE;
    const int bg_offset = idx * HIDDEN_SIZE;
    const int wh_offset = idx * INPUT_SIZE * HIDDEN_SIZE;
    const int bh_offset = idx * HIDDEN_SIZE;
    const int wo_offset = idx * HIDDEN_SIZE * OUTPUT_SIZE;
    const int bo_offset = idx * OUTPUT_SIZE;

    // Load sensors into registers
    float input[8];
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = sensors[sensor_offset + i];
    }

    // Compute hidden layer with gating
    float hidden[24];
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        // Gate: sigmoid(Wg · input + bg)
        float gate_sum = bg[bg_offset + h];
        for (int i = 0; i < INPUT_SIZE; i++) {
            gate_sum += input[i] * Wg[wg_offset + i * HIDDEN_SIZE + h];
        }
        float gate = 1.0f / (1.0f + expf(-gate_sum));

        // Candidate: tanh(Wh · input + bh)
        float candidate_sum = bh[bh_offset + h];
        for (int i = 0; i < INPUT_SIZE; i++) {
            candidate_sum += input[i] * Wh[wh_offset + i * HIDDEN_SIZE + h];
        }
        float candidate = tanhf(candidate_sum);

        // Gated hidden = gate ⊙ candidate
        hidden[h] = gate * candidate;
    }

    // Compute output layer: tanh(Wo · hidden + bo)
    float output[5];
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        float sum = bo[bo_offset + o];
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            sum += hidden[h] * Wo[wo_offset + h * OUTPUT_SIZE + o];
        }
        output[o] = tanhf(sum);
    }

    // Argmax for action selection
    int best_action = 0;
    float best_value = output[0];
    for (int o = 1; o < OUTPUT_SIZE; o++) {
        if (output[o] > best_value) {
            best_value = output[o];
            best_action = o;
        }
    }

    actions[idx] = best_action;
}
"#;

/// GPU-accelerated GatedFeedforward population
///
/// Supports per-organism weights for evolutionary algorithms.
#[cfg(feature = "cuda")]
pub struct GpuGatedFeedforward {
    rt: CudaRuntime,
    kernel: CudaFunction,

    // GPU buffers for weights (per-organism)
    d_wg: CudaSlice<f32>, // [N × 8 × 24]
    d_bg: CudaSlice<f32>, // [N × 24]
    d_wh: CudaSlice<f32>, // [N × 8 × 24]
    d_bh: CudaSlice<f32>, // [N × 24]
    d_wo: CudaSlice<f32>, // [N × 24 × 5]
    d_bo: CudaSlice<f32>, // [N × 5]

    // GPU buffers for I/O
    d_sensors: CudaSlice<f32>, // [N × 8]
    d_actions: CudaSlice<i32>, // [N]

    capacity: usize,
    active_count: usize,
}

#[cfg(feature = "cuda")]
impl GpuGatedFeedforward {
    /// Create new GPU GatedFeedforward accelerator
    pub fn new(capacity: usize) -> CudaResult<Self> {
        let rt = CudaRuntime::new()?;
        let kernel = Self::compile_kernel(&rt)?;

        let i = GatedFeedforwardWeights::INPUT_SIZE;
        let h = GatedFeedforwardWeights::HIDDEN_SIZE;
        let o = GatedFeedforwardWeights::OUTPUT_SIZE;

        // Allocate weight buffers (per-organism)
        let d_wg = rt.alloc_zeros::<f32>(capacity * i * h)?;
        let d_bg = rt.alloc_zeros::<f32>(capacity * h)?;
        let d_wh = rt.alloc_zeros::<f32>(capacity * i * h)?;
        let d_bh = rt.alloc_zeros::<f32>(capacity * h)?;
        let d_wo = rt.alloc_zeros::<f32>(capacity * h * o)?;
        let d_bo = rt.alloc_zeros::<f32>(capacity * o)?;

        // Allocate I/O buffers
        let d_sensors = rt.alloc_zeros::<f32>(capacity * i)?;
        let d_actions = rt.alloc_zeros::<i32>(capacity)?;

        Ok(Self {
            rt,
            kernel,
            d_wg,
            d_bg,
            d_wh,
            d_bh,
            d_wo,
            d_bo,
            d_sensors,
            d_actions,
            capacity,
            active_count: 0,
        })
    }

    fn compile_kernel(rt: &CudaRuntime) -> CudaResult<CudaFunction> {
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("sm_75"),
            ..Default::default()
        };

        let ptx = compile_ptx_with_opts(GATED_FF_KERNEL, opts)
            .map_err(|e| CudaError::KernelCompilationFailed(format!("GatedFF kernel: {:?}", e)))?;

        let module = rt
            .ctx()
            .load_module(ptx)
            .map_err(|e| CudaError::KernelCompilationFailed(format!("Module load: {:?}", e)))?;

        let kernel = module
            .load_function("gated_ff_forward")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("Function load: {:?}", e)))?;

        Ok(kernel)
    }

    /// Upload population weights to GPU
    pub fn upload_population(&mut self, pop: &GatedFeedforwardPopulation) -> CudaResult<()> {
        let n = pop.len();
        if n > self.capacity {
            return Err(CudaError::InvalidConfig(format!(
                "Population size {} exceeds capacity {}",
                n, self.capacity
            )));
        }

        let (wg, bg, wh, bh, wo, bo) = pop.to_gpu_arrays();

        self.d_wg = self.rt.upload(&wg)?;
        self.d_bg = self.rt.upload(&bg)?;
        self.d_wh = self.rt.upload(&wh)?;
        self.d_bh = self.rt.upload(&bh)?;
        self.d_wo = self.rt.upload(&wo)?;
        self.d_bo = self.rt.upload(&bo)?;

        self.active_count = n;

        Ok(())
    }

    /// Upload sensor data for all organisms
    pub fn upload_sensors(&mut self, sensors: &[f32]) -> CudaResult<()> {
        // sensors should be [N × 8] flattened
        let expected_len = self.active_count * GatedFeedforwardWeights::INPUT_SIZE;
        if sensors.len() != expected_len {
            return Err(CudaError::InvalidConfig(format!(
                "Expected {} sensor values, got {}",
                expected_len,
                sensors.len()
            )));
        }

        self.d_sensors = self.rt.upload(sensors)?;

        Ok(())
    }

    /// Run forward pass on GPU (inference only)
    pub fn forward(&mut self) -> CudaResult<()> {
        let threads_per_block = 256;
        let blocks = (self.active_count + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = self.active_count as i32;

        unsafe {
            self.rt
                .stream()
                .launch_builder(&self.kernel)
                .arg(&self.d_sensors)
                .arg(&self.d_wg)
                .arg(&self.d_bg)
                .arg(&self.d_wh)
                .arg(&self.d_bh)
                .arg(&self.d_wo)
                .arg(&self.d_bo)
                .arg(&self.d_actions)
                .arg(&n_i32)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("GatedFF forward: {:?}", e)))?;
        }

        Ok(())
    }

    /// Download actions from GPU
    pub fn download_actions(&self) -> CudaResult<Vec<i32>> {
        let actions = self.rt.download(&self.d_actions)?;
        Ok(actions[..self.active_count].to_vec())
    }

    /// Full forward pass: upload sensors → compute → download actions
    pub fn forward_batch(&mut self, sensors: &[f32]) -> CudaResult<Vec<i32>> {
        self.upload_sensors(sensors)?;
        self.forward()?;
        self.download_actions()
    }

    /// Get active organism count
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Synchronize GPU stream
    pub fn sync(&self) -> CudaResult<()> {
        self.rt.synchronize()
    }
}

/// CPU/GPU parity verification helper
#[cfg(feature = "cuda")]
pub fn verify_cpu_gpu_parity(
    pop: &GatedFeedforwardPopulation,
    sensors: &[Vec<f32>],
    tolerance: f32,
) -> CudaResult<(bool, Vec<(usize, i32, i32)>)> {
    let n = pop.len();
    assert_eq!(sensors.len(), n, "Sensor count must match population size");

    // CPU inference
    let cpu_actions: Vec<i32> = pop
        .organisms
        .iter()
        .zip(sensors.iter())
        .map(|(w, s): (&GatedFeedforwardWeights, &Vec<f32>)| w.decide(s) as i32)
        .collect();

    // GPU inference
    let mut gpu = GpuGatedFeedforward::new(n)?;
    gpu.upload_population(pop)?;

    let flat_sensors: Vec<f32> = sensors.iter().flatten().copied().collect();
    let gpu_actions = gpu.forward_batch(&flat_sensors)?;

    // Compare
    let mut mismatches = Vec::new();
    for i in 0..n {
        if cpu_actions[i] != gpu_actions[i] {
            mismatches.push((i, cpu_actions[i], gpu_actions[i]));
        }
    }

    let passed = mismatches.is_empty();

    // Note: tolerance parameter reserved for future output value comparison
    let _ = tolerance;

    Ok((passed, mismatches))
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_compile() {
        let gpu = GpuGatedFeedforward::new(1000);
        assert!(gpu.is_ok(), "GPU kernel should compile");
    }

    #[test]
    fn test_gpu_upload_population() {
        let pop = GatedFeedforwardPopulation::new_random(100, 42);
        let mut gpu = GpuGatedFeedforward::new(1000).unwrap();
        let result = gpu.upload_population(&pop);
        assert!(result.is_ok(), "Population upload should succeed");
        assert_eq!(gpu.active_count(), 100);
    }

    #[test]
    fn test_gpu_forward_basic() {
        let pop = GatedFeedforwardPopulation::new_random(100, 42);
        let mut gpu = GpuGatedFeedforward::new(1000).unwrap();
        gpu.upload_population(&pop).unwrap();

        // Create random sensors
        let sensors: Vec<f32> = (0..100 * 8)
            .map(|i| ((i * 17 + 31) % 100) as f32 / 100.0 - 0.5)
            .collect();

        let actions = gpu.forward_batch(&sensors).unwrap();
        assert_eq!(actions.len(), 100);

        // All actions should be valid (0-4)
        for &a in &actions {
            assert!(a >= 0 && a < 5, "Action {} out of range", a);
        }
    }

    #[test]
    fn test_cpu_gpu_parity() {
        let pop = GatedFeedforwardPopulation::new_random(50, 12345);

        // Create sensor data
        let sensors: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                (0..8)
                    .map(|j| ((i * 17 + j * 31) % 100) as f32 / 100.0 - 0.5)
                    .collect()
            })
            .collect();

        let (passed, mismatches) = verify_cpu_gpu_parity(&pop, &sensors, 1e-5).unwrap();

        if !passed {
            println!("Mismatches: {:?}", mismatches);
        }

        assert!(passed, "CPU and GPU should produce identical actions");
    }
}
