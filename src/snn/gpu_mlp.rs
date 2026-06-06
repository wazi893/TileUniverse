//! GPU-accelerated MLP readout for the LSM pipeline.
//!
//! Moves the entire MLP forward/backward/Adam/E-prop signal computation to GPU,
//! eliminating the CPU bottleneck that caused ~10% GPU utilization in M10.
//!
//! # Architecture
//!
//! ```text
//! GPU:  d_all_counts [B*N] u32  (from SNN tick_many)
//!   → extract_hidden_rates     → d_hidden_rates [B*n_hid] f32
//!   → matmul_bias_relu (L1)    → d_a1, d_z1
//!   → matmul_bias_relu (L2)    → d_a2, d_z2
//!   → matmul_bias_relu (L3)    → d_logits
//!   → softmax_ce               → d_sm, d_loss, d_preds, d_correct
//!   → backward_output          → d_dlogits, dW3, db3
//!   → backward_relu (L2)       → d_dz2, dW2, db2
//!   → backward_relu (L1)       → d_dz1, dW1, db1
//!   → adam_update × 6          → params updated in-place
//!   → eprop_signal             → d_signals [B*(n_hid+1)]
//! ```
//!
//! Only ~20 bytes (correct count + loss) are downloaded per batch for logging.

use crate::cuda::{CudaError, CudaResult, CudaRuntime};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

// ─── CUDA Kernel Source ──────────────────────────────────────────────────────

const GPU_MLP_CUDA: &str = r#"
extern "C" {

// ============================================================================
// KERNEL 1: Extract hidden firing rates from all_counts
// ============================================================================
// Thread: one per (batch_size * n_hid).
// Reads all_counts[b * n_neurons + n_inp + h], multiplies by inv_ticks.
__global__ void mlp_extract_hidden_rates(
    const unsigned int* all_counts,   // [B * n_neurons]
    float*              hidden_rates, // [B * n_hid]
    unsigned int        n_neurons,
    unsigned int        n_inp,
    unsigned int        n_hid,
    unsigned int        batch_size,
    float               inv_ticks     // 1.0f / TICKS
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * n_hid) return;
    unsigned int b = tid / n_hid;
    unsigned int h = tid % n_hid;
    hidden_rates[tid] = (float)all_counts[b * n_neurons + n_inp + h] * inv_ticks;
}

// ============================================================================
// KERNEL 2: Batch matmul with bias and optional ReLU
// ============================================================================
// C[b,m] = sum_k A[b,k] * W[k,m] + bias[m]
// If apply_relu: C = max(C, 0). Pre-ReLU stored in Z when store_z=1.
// Thread: one per (batch_size * out_dim).
__global__ void mlp_matmul_bias_relu(
    const float* A,          // [B * in_dim]
    const float* W,          // [in_dim * out_dim]
    const float* bias,       // [out_dim]
    float*       C,          // [B * out_dim] output (post-ReLU if apply_relu)
    float*       Z,          // [B * out_dim] pre-ReLU storage (only written if store_z)
    unsigned int batch_size,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int apply_relu,
    unsigned int store_z
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * out_dim) return;
    unsigned int b = tid / out_dim;
    unsigned int m = tid % out_dim;

    float sum = bias[m];
    const float* a_row = A + b * in_dim;
    for (unsigned int k = 0; k < in_dim; k++) {
        sum += a_row[k] * W[k * out_dim + m];
    }
    if (store_z) Z[tid] = sum;
    C[tid] = (apply_relu && sum < 0.0f) ? 0.0f : sum;
}

// ============================================================================
// KERNEL 3: Softmax + CE loss + argmax prediction
// ============================================================================
// Thread: one per batch sample.
__global__ void mlp_softmax_ce(
    const float*        logits,     // [B * n_classes]
    const unsigned int* labels,     // [B]
    float*              sm,         // [B * n_classes] softmax output
    float*              loss,       // [B] per-sample CE loss
    unsigned int*       preds,      // [B] argmax prediction
    unsigned int*       correct,    // [1] atomic correct count
    unsigned int        batch_size,
    unsigned int        n_classes
) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    const float* row = logits + b * n_classes;
    float* sm_row = sm + b * n_classes;

    // Find max for numerical stability
    float max_val = row[0];
    for (unsigned int c = 1; c < n_classes; c++) {
        if (row[c] > max_val) max_val = row[c];
    }

    // Exp + sum
    float sum_exp = 0.0f;
    for (unsigned int c = 0; c < n_classes; c++) {
        float e = expf(row[c] - max_val);
        sm_row[c] = e;
        sum_exp += e;
    }

    // Normalize + argmax
    unsigned int pred = 0;
    float max_sm = 0.0f;
    for (unsigned int c = 0; c < n_classes; c++) {
        sm_row[c] /= sum_exp;
        if (sm_row[c] > max_sm) { max_sm = sm_row[c]; pred = c; }
    }

    preds[b] = pred;
    float p = sm_row[labels[b]];
    loss[b] = -logf(p > 1e-20f ? p : 1e-20f);

    if (pred == labels[b]) atomicAdd(correct, 1u);
}

// ============================================================================
// KERNEL 4: Backward through output layer (CE gradient + weight/bias grads)
// ============================================================================
// dLogits[b,c] = sm[b,c] - one_hot(labels[b], c)
// Accumulates dW[k,c] += a_prev[b,k] * dLogits[b,c]  (atomicAdd)
// Accumulates db[c] += dLogits[b,c]  (atomicAdd)
// Thread: one per (batch_size * n_classes).
__global__ void mlp_backward_output(
    const float*        sm,        // [B * n_classes] softmax probabilities
    const unsigned int* labels,    // [B]
    const float*        a_prev,    // [B * prev_dim] activations from previous layer
    float*              dlogits,   // [B * n_classes] output: CE gradient
    float*              dw,        // [prev_dim * n_classes] weight grads (atomicAdd)
    float*              db,        // [n_classes] bias grads (atomicAdd)
    unsigned int        batch_size,
    unsigned int        prev_dim,
    unsigned int        n_classes
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * n_classes) return;
    unsigned int b = tid / n_classes;
    unsigned int c = tid % n_classes;

    float dl = sm[tid] - ((c == labels[b]) ? 1.0f : 0.0f);
    dlogits[tid] = dl;

    atomicAdd(&db[c], dl);
    const float* a_row = a_prev + b * prev_dim;
    for (unsigned int k = 0; k < prev_dim; k++) {
        atomicAdd(&dw[k * n_classes + c], a_row[k] * dl);
    }
}

// ============================================================================
// KERNEL 5: Backward through ReLU layer (gradient + weight/bias grads)
// ============================================================================
// dZ_prev[b,k] = (sum_m dZ_next[b,m] * W_next[k,m]) * relu'(Z_prev[b,k])
// Accumulates dW_prev[i,k] += a_input[b,i] * dZ_prev[b,k]  (atomicAdd)
// Accumulates db_prev[k] += dZ_prev[b,k]  (atomicAdd)
// Thread: one per (batch_size * prev_dim).
__global__ void mlp_backward_relu(
    const float* dz_next,      // [B * next_dim] gradient from next layer
    const float* w_next,       // [prev_dim * next_dim] weights of next layer
    const float* z_prev,       // [B * prev_dim] pre-ReLU activations
    const float* a_input,      // [B * input_dim] activations feeding this layer
    float*       dz_prev,      // [B * prev_dim] output gradient
    float*       dw_prev,      // [input_dim * prev_dim] weight grads (atomicAdd)
    float*       db_prev,      // [prev_dim] bias grads (atomicAdd)
    unsigned int batch_size,
    unsigned int input_dim,
    unsigned int prev_dim,
    unsigned int next_dim
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * prev_dim) return;
    unsigned int b = tid / prev_dim;
    unsigned int k = tid % prev_dim;

    // ReLU derivative gate
    if (z_prev[tid] <= 0.0f) {
        dz_prev[tid] = 0.0f;
        return;
    }

    // Sum over next_dim
    float grad = 0.0f;
    const float* dz_row = dz_next + b * next_dim;
    for (unsigned int m = 0; m < next_dim; m++) {
        grad += dz_row[m] * w_next[k * next_dim + m];
    }
    dz_prev[tid] = grad;

    // Accumulate bias gradient
    atomicAdd(&db_prev[k], grad);

    // Accumulate weight gradient
    const float* inp = a_input + b * input_dim;
    for (unsigned int i = 0; i < input_dim; i++) {
        atomicAdd(&dw_prev[i * prev_dim + k], inp[i] * grad);
    }
}

// ============================================================================
// KERNEL 6: Adam optimizer update
// ============================================================================
// Thread: one per parameter element.
__global__ void mlp_adam_update(
    float*       param,      // [n] parameters (updated in-place)
    const float* grad,       // [n] accumulated gradients
    float*       m_buf,      // [n] first moment
    float*       v_buf,      // [n] second moment
    unsigned int n,
    float        lr_eff,     // LR * sqrt(1-beta2^t) / (1-beta1^t)
    float        beta1,
    float        beta2,
    float        eps,
    float        inv_B       // 1.0 / actual_batch_size (gradient scaling)
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i] * inv_B;
    m_buf[i] = beta1 * m_buf[i] + (1.0f - beta1) * g;
    v_buf[i] = beta2 * v_buf[i] + (1.0f - beta2) * g * g;
    param[i] -= lr_eff * m_buf[i] / (sqrtf(v_buf[i]) + eps);
}

// ============================================================================
// KERNEL 7: E-prop signal computation
// ============================================================================
// signal[b,h] = -sum_j dz1[b,j] * W1[h,j]
// Sentinel channel (index n_hid) stays zero from memset.
// Thread: one per (batch_size * n_hid).
__global__ void mlp_eprop_signal(
    const float* dz1,              // [B * mlp_h1]
    const float* w1,               // [n_hid * mlp_h1]
    float*       signals,          // [B * n_signal_channels]
    unsigned int batch_size,
    unsigned int n_hid,
    unsigned int mlp_h1,
    unsigned int n_signal_channels  // n_hid + 1
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * n_hid) return;
    unsigned int b = tid / n_hid;
    unsigned int h = tid % n_hid;

    float s = 0.0f;
    const float* dz_row = dz1 + b * mlp_h1;
    const float* w_row  = w1  + h * mlp_h1;
    for (unsigned int j = 0; j < mlp_h1; j++) {
        s -= dz_row[j] * w_row[j];
    }
    signals[b * n_signal_channels + h] = s;
}

// ============================================================================
// KERNEL 8: Zero a float buffer
// ============================================================================
__global__ void mlp_zero_buf(float* buf, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    buf[i] = 0.0f;
}

} // extern "C"
"#;

// ─── Kernel Handles ──────────────────────────────────────────────────────────

struct GpuMlpKernels {
    extract_hidden_rates_fn: CudaFunction,
    matmul_bias_relu_fn: CudaFunction,
    softmax_ce_fn: CudaFunction,
    backward_output_fn: CudaFunction,
    backward_relu_fn: CudaFunction,
    adam_update_fn: CudaFunction,
    eprop_signal_fn: CudaFunction,
    zero_buf_fn: CudaFunction,
}

fn compile_mlp_kernels(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
) -> CudaResult<GpuMlpKernels> {
    use logic_fabric_core::cuda::get_cuda_include_path;

    let cuda_include = get_cuda_include_path();
    let arch = "compute_89"; // Ada; PTX JIT for Blackwell SM120

    let opts = CompileOptions {
        arch: Some(arch),
        include_paths: vec![cuda_include],
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(GPU_MLP_CUDA, opts)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("GPU MLP compile: {:?}", e)))?;

    let module = ctx
        .load_module(ptx)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("GPU MLP module load: {:?}", e)))?;

    let load = |name: &str| -> CudaResult<CudaFunction> {
        module
            .load_function(name)
            .map_err(|e| CudaError::KernelCompilationFailed(format!("{name}: {:?}", e)))
    };

    Ok(GpuMlpKernels {
        extract_hidden_rates_fn: load("mlp_extract_hidden_rates")?,
        matmul_bias_relu_fn: load("mlp_matmul_bias_relu")?,
        softmax_ce_fn: load("mlp_softmax_ce")?,
        backward_output_fn: load("mlp_backward_output")?,
        backward_relu_fn: load("mlp_backward_relu")?,
        adam_update_fn: load("mlp_adam_update")?,
        eprop_signal_fn: load("mlp_eprop_signal")?,
        zero_buf_fn: load("mlp_zero_buf")?,
    })
}

fn launch_cfg(n: u32) -> LaunchConfig {
    const BLOCK: u32 = 256;
    LaunchConfig {
        block_dim: (BLOCK, 1, 1),
        grid_dim: ((n + BLOCK - 1) / BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ─── Public Types ────────────────────────────────────────────────────────────

// Re-export from the non-cuda mlp_weights module so existing users see no change.
pub use crate::snn::mlp_weights::MlpWeights;

// ─── GpuMLP Struct ───────────────────────────────────────────────────────────

/// GPU-resident 3-layer MLP with Adam optimizer.
///
/// All forward/backward/optimizer computation runs on GPU.
/// Only tiny metric downloads (~20 bytes) for logging.
pub struct GpuMLP {
    // ── Parameters (device-resident) ─────────────────────────
    d_w1: CudaSlice<f32>,
    d_b1: CudaSlice<f32>,
    d_w2: CudaSlice<f32>,
    d_b2: CudaSlice<f32>,
    d_w3: CudaSlice<f32>,
    d_b3: CudaSlice<f32>,

    // ── Adam moments (device-resident) ───────────────────────
    d_m_w1: CudaSlice<f32>,
    d_v_w1: CudaSlice<f32>,
    d_m_b1: CudaSlice<f32>,
    d_v_b1: CudaSlice<f32>,
    d_m_w2: CudaSlice<f32>,
    d_v_w2: CudaSlice<f32>,
    d_m_b2: CudaSlice<f32>,
    d_v_b2: CudaSlice<f32>,
    d_m_w3: CudaSlice<f32>,
    d_v_w3: CudaSlice<f32>,
    d_m_b3: CudaSlice<f32>,
    d_v_b3: CudaSlice<f32>,

    // ── Forward pass activations (device-resident) ───────────
    d_hidden_rates: CudaSlice<f32>,
    d_z1: CudaSlice<f32>,
    d_a1: CudaSlice<f32>,
    d_z2: CudaSlice<f32>,
    d_a2: CudaSlice<f32>,
    d_logits: CudaSlice<f32>,
    d_sm: CudaSlice<f32>,

    // ── Backward pass gradients (device-resident) ────────────
    d_dlogits: CudaSlice<f32>,
    d_dz2: CudaSlice<f32>,
    d_dz1: CudaSlice<f32>,
    d_dw1: CudaSlice<f32>,
    d_db1: CudaSlice<f32>,
    d_dw2: CudaSlice<f32>,
    d_db2: CudaSlice<f32>,
    d_dw3: CudaSlice<f32>,
    d_db3: CudaSlice<f32>,

    // ── Training I/O (device-resident) ───────────────────────
    d_labels: CudaSlice<u32>,
    d_preds: CudaSlice<u32>,
    d_loss: CudaSlice<f32>,
    d_correct: CudaSlice<u32>,

    // ── E-prop signals (device-resident) ─────────────────────
    d_signals: CudaSlice<f32>,

    // ── Kernel functions ─────────────────────────────────────
    kernels: GpuMlpKernels,

    // ── Dimensions ───────────────────────────────────────────
    batch_size: u32,
    n_hid: u32,
    n_neurons: u32,
    n_inp: u32,
    mlp_h1: u32,
    mlp_h2: u32,
    n_classes: u32,
    n_signal_channels: u32,

    // ── Adam state ───────────────────────────────────────────
    adam_t: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl GpuMLP {
    /// Create and initialise GPU MLP.
    ///
    /// Compiles CUDA kernels, uploads initial weights, allocates all device buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rt: &CudaRuntime,
        n_hid: usize,
        n_neurons: usize,
        n_inp: usize,
        mlp_h1: usize,
        mlp_h2: usize,
        n_classes: usize,
        batch_size: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
        w3: &[f32],
        b3: &[f32],
    ) -> CudaResult<Self> {
        assert_eq!(w1.len(), n_hid * mlp_h1);
        assert_eq!(b1.len(), mlp_h1);
        assert_eq!(w2.len(), mlp_h1 * mlp_h2);
        assert_eq!(b2.len(), mlp_h2);
        assert_eq!(w3.len(), mlp_h2 * n_classes);
        assert_eq!(b3.len(), n_classes);

        let kernels = compile_mlp_kernels(rt.context())?;

        let b = batch_size;
        let n_sig = n_hid + 1;

        Ok(Self {
            // Parameters
            d_w1: rt.upload(w1)?,
            d_b1: rt.upload(b1)?,
            d_w2: rt.upload(w2)?,
            d_b2: rt.upload(b2)?,
            d_w3: rt.upload(w3)?,
            d_b3: rt.upload(b3)?,

            // Adam moments (zero-initialised)
            d_m_w1: rt.alloc_zeros(n_hid * mlp_h1)?,
            d_v_w1: rt.alloc_zeros(n_hid * mlp_h1)?,
            d_m_b1: rt.alloc_zeros(mlp_h1)?,
            d_v_b1: rt.alloc_zeros(mlp_h1)?,
            d_m_w2: rt.alloc_zeros(mlp_h1 * mlp_h2)?,
            d_v_w2: rt.alloc_zeros(mlp_h1 * mlp_h2)?,
            d_m_b2: rt.alloc_zeros(mlp_h2)?,
            d_v_b2: rt.alloc_zeros(mlp_h2)?,
            d_m_w3: rt.alloc_zeros(mlp_h2 * n_classes)?,
            d_v_w3: rt.alloc_zeros(mlp_h2 * n_classes)?,
            d_m_b3: rt.alloc_zeros(n_classes)?,
            d_v_b3: rt.alloc_zeros(n_classes)?,

            // Forward activations
            d_hidden_rates: rt.alloc_zeros(b * n_hid)?,
            d_z1: rt.alloc_zeros(b * mlp_h1)?,
            d_a1: rt.alloc_zeros(b * mlp_h1)?,
            d_z2: rt.alloc_zeros(b * mlp_h2)?,
            d_a2: rt.alloc_zeros(b * mlp_h2)?,
            d_logits: rt.alloc_zeros(b * n_classes)?,
            d_sm: rt.alloc_zeros(b * n_classes)?,

            // Backward gradients
            d_dlogits: rt.alloc_zeros(b * n_classes)?,
            d_dz2: rt.alloc_zeros(b * mlp_h2)?,
            d_dz1: rt.alloc_zeros(b * mlp_h1)?,
            d_dw1: rt.alloc_zeros(n_hid * mlp_h1)?,
            d_db1: rt.alloc_zeros(mlp_h1)?,
            d_dw2: rt.alloc_zeros(mlp_h1 * mlp_h2)?,
            d_db2: rt.alloc_zeros(mlp_h2)?,
            d_dw3: rt.alloc_zeros(mlp_h2 * n_classes)?,
            d_db3: rt.alloc_zeros(n_classes)?,

            // Training I/O
            d_labels: rt.alloc_zeros(b)?,
            d_preds: rt.alloc_zeros(b)?,
            d_loss: rt.alloc_zeros(b)?,
            d_correct: rt.alloc_zeros(1)?,

            // E-prop signals
            d_signals: rt.alloc_zeros(b * n_sig)?,

            kernels,

            batch_size: batch_size as u32,
            n_hid: n_hid as u32,
            n_neurons: n_neurons as u32,
            n_inp: n_inp as u32,
            mlp_h1: mlp_h1 as u32,
            mlp_h2: mlp_h2 as u32,
            n_classes: n_classes as u32,
            n_signal_channels: n_sig as u32,

            adam_t: 0,
            lr,
            beta1,
            beta2,
            eps,
        })
    }

    /// Run full MLP forward + backward + Adam update on GPU.
    ///
    /// After this call, E-prop signals are ready in `signals_device()`.
    /// Call `get_train_metrics()` to download accuracy/loss for logging.
    ///
    /// All kernel launches use `&CudaSlice` (shared refs). The GPU writes through
    /// the device pointer; this is the standard cudarc pattern (see gpu_batch.rs).
    pub fn train_step(
        &mut self,
        rt: &CudaRuntime,
        d_all_counts: &CudaSlice<u32>,
        labels: &[usize],
        ticks: usize,
        actual_bs: usize,
    ) -> CudaResult<()> {
        let bs = actual_bs as u32;
        let stream = rt.stream();

        // 1. Upload labels (small: ~1KB)
        let label_u32: Vec<u32> = labels.iter().map(|&l| l as u32).collect();
        rt.upload_to_existing(&label_u32, &mut self.d_labels)?;

        // 2. Zero correct counter
        rt.memset_zeros(&mut self.d_correct)?;

        // 3. Extract hidden rates from all_counts (no D2H transfer)
        let inv_ticks = 1.0f32 / ticks as f32;
        unsafe {
            stream
                .launch_builder(&self.kernels.extract_hidden_rates_fn)
                .arg(d_all_counts)
                .arg(&self.d_hidden_rates)
                .arg(&self.n_neurons)
                .arg(&self.n_inp)
                .arg(&self.n_hid)
                .arg(&bs)
                .arg(&inv_ticks)
                .launch(launch_cfg(bs * self.n_hid))
                .map_err(|e| {
                    CudaError::LaunchFailed(format!("mlp_extract_hidden_rates: {:?}", e))
                })?;
        }

        // 4. Forward layer 1: H @ W1 + b1, ReLU → d_a1, d_z1
        {
            let (relu_flag, z_flag) = (1u32, 1u32);
            unsafe {
                stream
                    .launch_builder(&self.kernels.matmul_bias_relu_fn)
                    .arg(&self.d_hidden_rates)
                    .arg(&self.d_w1)
                    .arg(&self.d_b1)
                    .arg(&self.d_a1)
                    .arg(&self.d_z1)
                    .arg(&bs)
                    .arg(&self.n_hid)
                    .arg(&self.mlp_h1)
                    .arg(&relu_flag)
                    .arg(&z_flag)
                    .launch(launch_cfg(bs * self.mlp_h1))
                    .map_err(|e| CudaError::LaunchFailed(format!("mlp_matmul L1: {:?}", e)))?;
            }
        }

        // 5. Forward layer 2: A1 @ W2 + b2, ReLU → d_a2, d_z2
        {
            let (relu_flag, z_flag) = (1u32, 1u32);
            unsafe {
                stream
                    .launch_builder(&self.kernels.matmul_bias_relu_fn)
                    .arg(&self.d_a1)
                    .arg(&self.d_w2)
                    .arg(&self.d_b2)
                    .arg(&self.d_a2)
                    .arg(&self.d_z2)
                    .arg(&bs)
                    .arg(&self.mlp_h1)
                    .arg(&self.mlp_h2)
                    .arg(&relu_flag)
                    .arg(&z_flag)
                    .launch(launch_cfg(bs * self.mlp_h2))
                    .map_err(|e| CudaError::LaunchFailed(format!("mlp_matmul L2: {:?}", e)))?;
            }
        }

        // 6. Forward layer 3: A2 @ W3 + b3, linear → d_logits
        {
            let (relu_flag, z_flag) = (0u32, 0u32);
            unsafe {
                stream
                    .launch_builder(&self.kernels.matmul_bias_relu_fn)
                    .arg(&self.d_a2)
                    .arg(&self.d_w3)
                    .arg(&self.d_b3)
                    .arg(&self.d_logits)
                    .arg(&self.d_logits) // Z not stored (z_flag=0), valid ptr required
                    .arg(&bs)
                    .arg(&self.mlp_h2)
                    .arg(&self.n_classes)
                    .arg(&relu_flag)
                    .arg(&z_flag)
                    .launch(launch_cfg(bs * self.n_classes))
                    .map_err(|e| CudaError::LaunchFailed(format!("mlp_matmul L3: {:?}", e)))?;
            }
        }

        // 7. Softmax + CE loss + predictions
        unsafe {
            stream
                .launch_builder(&self.kernels.softmax_ce_fn)
                .arg(&self.d_logits)
                .arg(&self.d_labels)
                .arg(&self.d_sm)
                .arg(&self.d_loss)
                .arg(&self.d_preds)
                .arg(&self.d_correct)
                .arg(&bs)
                .arg(&self.n_classes)
                .launch(launch_cfg(bs))
                .map_err(|e| CudaError::LaunchFailed(format!("mlp_softmax_ce: {:?}", e)))?;
        }

        // 8. Zero gradient buffers (atomicAdd accumulation targets)
        macro_rules! zero_buf {
            ($buf:expr, $n:expr, $label:expr) => {
                unsafe {
                    stream
                        .launch_builder(&self.kernels.zero_buf_fn)
                        .arg(&$buf)
                        .arg(&$n)
                        .launch(launch_cfg($n))
                        .map_err(|e| {
                            CudaError::LaunchFailed(format!("mlp_zero_buf {}: {:?}", $label, e))
                        })?;
                }
            };
        }
        let n_dw3 = self.mlp_h2 * self.n_classes;
        let n_dw2 = self.mlp_h1 * self.mlp_h2;
        let n_dw1 = self.n_hid * self.mlp_h1;
        zero_buf!(self.d_dw3, n_dw3, "dw3");
        zero_buf!(self.d_db3, self.n_classes, "db3");
        zero_buf!(self.d_dw2, n_dw2, "dw2");
        zero_buf!(self.d_db2, self.mlp_h2, "db2");
        zero_buf!(self.d_dw1, n_dw1, "dw1");
        zero_buf!(self.d_db1, self.mlp_h1, "db1");

        // 9. Backward output layer: dlogits = sm - onehot, accumulate dW3/db3
        unsafe {
            stream
                .launch_builder(&self.kernels.backward_output_fn)
                .arg(&self.d_sm)
                .arg(&self.d_labels)
                .arg(&self.d_a2)
                .arg(&self.d_dlogits)
                .arg(&self.d_dw3)
                .arg(&self.d_db3)
                .arg(&bs)
                .arg(&self.mlp_h2)
                .arg(&self.n_classes)
                .launch(launch_cfg(bs * self.n_classes))
                .map_err(|e| CudaError::LaunchFailed(format!("mlp_backward_output: {:?}", e)))?;
        }

        // 10. Backward layer 2: dz2 = (dlogits @ W3^T) * relu'(z2), accumulate dW2/db2
        unsafe {
            stream
                .launch_builder(&self.kernels.backward_relu_fn)
                .arg(&self.d_dlogits)
                .arg(&self.d_w3)
                .arg(&self.d_z2)
                .arg(&self.d_a1)
                .arg(&self.d_dz2)
                .arg(&self.d_dw2)
                .arg(&self.d_db2)
                .arg(&bs)
                .arg(&self.mlp_h1)
                .arg(&self.mlp_h2)
                .arg(&self.n_classes)
                .launch(launch_cfg(bs * self.mlp_h2))
                .map_err(|e| CudaError::LaunchFailed(format!("mlp_backward_relu L2: {:?}", e)))?;
        }

        // 11. Backward layer 1: dz1 = (dz2 @ W2^T) * relu'(z1), accumulate dW1/db1
        unsafe {
            stream
                .launch_builder(&self.kernels.backward_relu_fn)
                .arg(&self.d_dz2)
                .arg(&self.d_w2)
                .arg(&self.d_z1)
                .arg(&self.d_hidden_rates)
                .arg(&self.d_dz1)
                .arg(&self.d_dw1)
                .arg(&self.d_db1)
                .arg(&bs)
                .arg(&self.n_hid)
                .arg(&self.mlp_h1)
                .arg(&self.mlp_h2)
                .launch(launch_cfg(bs * self.mlp_h1))
                .map_err(|e| CudaError::LaunchFailed(format!("mlp_backward_relu L1: {:?}", e)))?;
        }

        // 12. Adam updates — 6 parameter sets, all inline
        self.adam_t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.adam_t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.adam_t as i32);
        let lr_eff = self.lr * bc2.sqrt() / bc1;
        let inv_b = 1.0f32 / actual_bs as f32;
        let b1_val = self.beta1;
        let b2_val = self.beta2;
        let eps_val = self.eps;

        macro_rules! adam {
            ($p:expr, $g:expr, $m:expr, $v:expr, $n:expr, $label:expr) => {
                unsafe {
                    stream
                        .launch_builder(&self.kernels.adam_update_fn)
                        .arg(&$p)
                        .arg(&$g)
                        .arg(&$m)
                        .arg(&$v)
                        .arg(&$n)
                        .arg(&lr_eff)
                        .arg(&b1_val)
                        .arg(&b2_val)
                        .arg(&eps_val)
                        .arg(&inv_b)
                        .launch(launch_cfg($n))
                        .map_err(|e| {
                            CudaError::LaunchFailed(format!("mlp_adam {}: {:?}", $label, e))
                        })?;
                }
            };
        }
        adam!(self.d_w1, self.d_dw1, self.d_m_w1, self.d_v_w1, n_dw1, "w1");
        adam!(
            self.d_b1,
            self.d_db1,
            self.d_m_b1,
            self.d_v_b1,
            self.mlp_h1,
            "b1"
        );
        adam!(self.d_w2, self.d_dw2, self.d_m_w2, self.d_v_w2, n_dw2, "w2");
        adam!(
            self.d_b2,
            self.d_db2,
            self.d_m_b2,
            self.d_v_b2,
            self.mlp_h2,
            "b2"
        );
        adam!(self.d_w3, self.d_dw3, self.d_m_w3, self.d_v_w3, n_dw3, "w3");
        adam!(
            self.d_b3,
            self.d_db3,
            self.d_m_b3,
            self.d_v_b3,
            self.n_classes,
            "b3"
        );

        // 13. Compute E-prop signals: signal[b,h] = -sum_j dz1[b,j] * W1[h,j]
        rt.memset_zeros(&mut self.d_signals)?;
        unsafe {
            stream
                .launch_builder(&self.kernels.eprop_signal_fn)
                .arg(&self.d_dz1)
                .arg(&self.d_w1)
                .arg(&self.d_signals)
                .arg(&bs)
                .arg(&self.n_hid)
                .arg(&self.mlp_h1)
                .arg(&self.n_signal_channels)
                .launch(launch_cfg(bs * self.n_hid))
                .map_err(|e| CudaError::LaunchFailed(format!("mlp_eprop_signal: {:?}", e)))?;
        }

        Ok(())
    }

    /// Device-resident E-prop signals for direct GPU-to-GPU transfer.
    pub fn signals_device(&self) -> &CudaSlice<f32> {
        &self.d_signals
    }

    /// Number of signal channels (n_hid + 1, including sentinel).
    pub fn n_signal_channels(&self) -> usize {
        self.n_signal_channels as usize
    }

    /// Download training metrics (~20 bytes D2H).
    /// Returns (correct_count, mean_loss).
    pub fn get_train_metrics(
        &self,
        rt: &CudaRuntime,
        actual_bs: usize,
    ) -> CudaResult<(usize, f32)> {
        let correct_vec = rt.download(&self.d_correct)?;
        let loss_vec = rt.download(&self.d_loss)?;
        let correct = correct_vec[0] as usize;
        let mean_loss = loss_vec[..actual_bs].iter().sum::<f32>() / actual_bs as f32;
        Ok((correct, mean_loss))
    }

    /// Forward-only evaluation (no backward, no Adam).
    /// Returns argmax predictions for the batch.
    pub fn forward_test(
        &mut self,
        rt: &CudaRuntime,
        d_all_counts: &CudaSlice<u32>,
        ticks: usize,
        actual_bs: usize,
    ) -> CudaResult<Vec<u32>> {
        let bs = actual_bs as u32;
        let inv_ticks = 1.0f32 / ticks as f32;
        let stream = rt.stream();

        // Extract hidden rates
        unsafe {
            stream
                .launch_builder(&self.kernels.extract_hidden_rates_fn)
                .arg(d_all_counts)
                .arg(&self.d_hidden_rates)
                .arg(&self.n_neurons)
                .arg(&self.n_inp)
                .arg(&self.n_hid)
                .arg(&bs)
                .arg(&inv_ticks)
                .launch(launch_cfg(bs * self.n_hid))
                .map_err(|e| {
                    CudaError::LaunchFailed(format!("mlp_extract_hidden_rates (test): {:?}", e))
                })?;
        }

        // Forward L1: H @ W1 + b1, ReLU → d_a1 (no Z needed for test)
        {
            let (relu_flag, z_flag) = (1u32, 0u32);
            unsafe {
                stream
                    .launch_builder(&self.kernels.matmul_bias_relu_fn)
                    .arg(&self.d_hidden_rates)
                    .arg(&self.d_w1)
                    .arg(&self.d_b1)
                    .arg(&self.d_a1)
                    .arg(&self.d_z1) // not stored, valid ptr
                    .arg(&bs)
                    .arg(&self.n_hid)
                    .arg(&self.mlp_h1)
                    .arg(&relu_flag)
                    .arg(&z_flag)
                    .launch(launch_cfg(bs * self.mlp_h1))
                    .map_err(|e| {
                        CudaError::LaunchFailed(format!("mlp_matmul L1 (test): {:?}", e))
                    })?;
            }
        }

        // Forward L2: A1 @ W2 + b2, ReLU → d_a2
        {
            let (relu_flag, z_flag) = (1u32, 0u32);
            unsafe {
                stream
                    .launch_builder(&self.kernels.matmul_bias_relu_fn)
                    .arg(&self.d_a1)
                    .arg(&self.d_w2)
                    .arg(&self.d_b2)
                    .arg(&self.d_a2)
                    .arg(&self.d_z2) // not stored, valid ptr
                    .arg(&bs)
                    .arg(&self.mlp_h1)
                    .arg(&self.mlp_h2)
                    .arg(&relu_flag)
                    .arg(&z_flag)
                    .launch(launch_cfg(bs * self.mlp_h2))
                    .map_err(|e| {
                        CudaError::LaunchFailed(format!("mlp_matmul L2 (test): {:?}", e))
                    })?;
            }
        }

        // Forward L3: A2 @ W3 + b3, linear → d_logits
        {
            let (relu_flag, z_flag) = (0u32, 0u32);
            unsafe {
                stream
                    .launch_builder(&self.kernels.matmul_bias_relu_fn)
                    .arg(&self.d_a2)
                    .arg(&self.d_w3)
                    .arg(&self.d_b3)
                    .arg(&self.d_logits)
                    .arg(&self.d_logits) // valid ptr, not stored
                    .arg(&bs)
                    .arg(&self.mlp_h2)
                    .arg(&self.n_classes)
                    .arg(&relu_flag)
                    .arg(&z_flag)
                    .launch(launch_cfg(bs * self.n_classes))
                    .map_err(|e| {
                        CudaError::LaunchFailed(format!("mlp_matmul L3 (test): {:?}", e))
                    })?;
            }
        }

        // Softmax → predictions (loss/correct ignored)
        rt.memset_zeros(&mut self.d_correct)?;
        unsafe {
            stream
                .launch_builder(&self.kernels.softmax_ce_fn)
                .arg(&self.d_logits)
                .arg(&self.d_labels) // dummy labels, loss ignored
                .arg(&self.d_sm)
                .arg(&self.d_loss) // ignored
                .arg(&self.d_preds)
                .arg(&self.d_correct)
                .arg(&bs)
                .arg(&self.n_classes)
                .launch(launch_cfg(bs))
                .map_err(|e| CudaError::LaunchFailed(format!("mlp_softmax_ce (test): {:?}", e)))?;
        }

        rt.synchronize()?;
        let preds = rt.download(&self.d_preds)?;
        Ok(preds[..actual_bs].to_vec())
    }

    /// Download the hidden firing rates for spike rate monitoring.
    pub fn get_mean_spike_rate(&self, rt: &CudaRuntime, actual_bs: usize) -> CudaResult<f32> {
        let rates = rt.download(&self.d_hidden_rates)?;
        let n = actual_bs * self.n_hid as usize;
        let sum: f32 = rates[..n].iter().sum();
        Ok(sum / n as f32)
    }

    /// Download hidden firing rates for a batch (after `forward_test` has been called).
    ///
    /// Returns `actual_bs * n_hid` f32 values in row-major order (one row per sample).
    pub fn download_hidden_rates(
        &self,
        rt: &CudaRuntime,
        actual_bs: usize,
    ) -> CudaResult<Vec<f32>> {
        let rates = rt.download(&self.d_hidden_rates)?;
        let n = actual_bs * self.n_hid as usize;
        Ok(rates[..n].to_vec())
    }

    /// Download MLP weights for diagnostic logging.
    pub fn download_weights(&self, rt: &CudaRuntime) -> CudaResult<MlpWeights> {
        Ok(MlpWeights {
            w1: rt.download(&self.d_w1)?,
            b1: rt.download(&self.d_b1)?,
            w2: rt.download(&self.d_w2)?,
            b2: rt.download(&self.d_b2)?,
            w3: rt.download(&self.d_w3)?,
            b3: rt.download(&self.d_b3)?,
        })
    }
}
