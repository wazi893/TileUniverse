//! EPIC 96 V2: WMMA Scaling Benchmark
//!
//! Tests WMMA performance across different NN sizes:
//! - 8×8 (current baseline, 75% padding waste)
//! - 16×16 (perfect WMMA fit, 0% waste)
//! - 32×32 (4 WMMA tiles, 0% waste)
//!
//! The question: Does bigger NN = bigger speedup?

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx_with_opts;
#[cfg(feature = "cuda")]
use engine::cuda::{CudaError, CudaResult, CudaRuntime};
#[cfg(feature = "cuda")]
use engine::quantum::QRng;
#[cfg(feature = "cuda")]
use std::time::Instant;

// Generalized WMMA NN kernel for any 16×16 tile size
#[cfg(feature = "cuda")]
const WMMA_SCALED_KERNEL: &str = r#"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

__device__ __forceinline__ float tanh_approx(float x) {
    if (x > 3.0f) return 1.0f;
    if (x < -3.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// 16×16 single-tile WMMA matmul + bias + tanh
// A: [16 × 16] row-major, B: [16 × 16] col-major, C: [16 × 16] output
extern "C" __global__ void wmma_16x16_forward(
    const half* __restrict__ sensors,     // [n_batches × 16 × 16]
    const half* __restrict__ weights1,    // [16 × 16] col-major (transposed)
    const half* __restrict__ bias1,       // [16]
    const half* __restrict__ weights2,    // [16 × 16] col-major
    const half* __restrict__ bias2,       // [16]
    int* __restrict__ moves,              // [n]
    int n_organisms,
    int n_outputs                         // How many outputs to argmax over
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int batch_start = warp_id * 16;

    if (batch_start >= n_organisms) return;

    extern __shared__ float smem[];
    const int warp_idx = threadIdx.x / 32;
    float* my_smem = smem + warp_idx * 256;
    half* my_hidden_half = (half*)(smem + blockDim.x / 32 * 256) + warp_idx * 256;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Layer 1
    const half* sensor_ptr = sensors + warp_id * 256;
    wmma::load_matrix_sync(a_frag, sensor_ptr, 16);
    wmma::load_matrix_sync(b_frag, weights1, 16);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(my_smem, c_frag, 16, wmma::mem_row_major);
    __syncwarp();

    // Bias + tanh for layer 1
    for (int i = lane_id; i < 256; i += 32) {
        int col = i % 16;
        float val = my_smem[i] + __half2float(bias1[col]);
        my_smem[i] = tanh_approx(val);
    }
    __syncwarp();

    // Convert to half
    for (int i = lane_id; i < 256; i += 32) {
        my_hidden_half[i] = __float2half(my_smem[i]);
    }
    __syncwarp();

    // Layer 2
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> hidden_frag;
    wmma::load_matrix_sync(hidden_frag, my_hidden_half, 16);
    wmma::load_matrix_sync(b_frag, weights2, 16);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, hidden_frag, b_frag, c_frag);
    wmma::store_matrix_sync(my_smem, c_frag, 16, wmma::mem_row_major);
    __syncwarp();

    // Output: bias + tanh + argmax
    for (int org = lane_id; org < 16; org += 32) {
        if (batch_start + org >= n_organisms) continue;

        int best = 0;
        float best_val = -1e9f;
        for (int o = 0; o < n_outputs; o++) {
            float val = my_smem[org * 16 + o] + __half2float(bias2[o]);
            val = tanh_approx(val);
            if (val > best_val) {
                best_val = val;
                best = o;
            }
        }
        moves[batch_start + org] = best;
    }
}

// 32×32 four-tile WMMA matmul (2×2 tiles)
extern "C" __global__ void wmma_32x32_forward(
    const half* __restrict__ sensors,     // [n_batches × 32 × 32]
    const half* __restrict__ weights1,    // [32 × 32] col-major
    const half* __restrict__ bias1,       // [32]
    const half* __restrict__ weights2,    // [32 × 32] col-major
    const half* __restrict__ bias2,       // [32]
    int* __restrict__ moves,
    int n_organisms,
    int n_outputs
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int batch_start = warp_id * 16;  // Still 16 organisms per warp

    if (batch_start >= n_organisms) return;

    extern __shared__ float smem[];
    const int warp_idx = threadIdx.x / 32;
    // For 32×32: need 32×32 = 1024 floats for intermediate, plus halves
    float* my_smem = smem + warp_idx * 1024;
    half* my_hidden_half = (half*)(smem + blockDim.x / 32 * 1024) + warp_idx * 1024;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Layer 1: 32×32 = 2×2 tiles
    // C[0:16, 0:16] = A[0:16, 0:16] × B[0:16, 0:16] + A[0:16, 16:32] × B[16:32, 0:16]
    // etc. for all 4 output tiles

    for (int tile_row = 0; tile_row < 2; tile_row++) {
        for (int tile_col = 0; tile_col < 2; tile_col++) {
            wmma::fill_fragment(c_frag, 0.0f);

            // Accumulate over K dimension (2 tiles)
            for (int k = 0; k < 2; k++) {
                const half* a_ptr = sensors + warp_id * 1024 + tile_row * 16 * 32 + k * 16;
                const half* b_ptr = weights1 + k * 16 + tile_col * 16 * 32;

                wmma::load_matrix_sync(a_frag, a_ptr, 32);
                wmma::load_matrix_sync(b_frag, b_ptr, 32);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            // Store this tile
            float* c_ptr = my_smem + tile_row * 16 * 32 + tile_col * 16;
            wmma::store_matrix_sync(c_ptr, c_frag, 32, wmma::mem_row_major);
        }
    }
    __syncwarp();

    // Bias + tanh
    for (int i = lane_id; i < 1024; i += 32) {
        int col = i % 32;
        float val = my_smem[i] + __half2float(bias1[col]);
        my_smem[i] = tanh_approx(val);
    }
    __syncwarp();

    // Convert to half
    for (int i = lane_id; i < 1024; i += 32) {
        my_hidden_half[i] = __float2half(my_smem[i]);
    }
    __syncwarp();

    // Layer 2: same pattern
    for (int tile_row = 0; tile_row < 2; tile_row++) {
        for (int tile_col = 0; tile_col < 2; tile_col++) {
            wmma::fill_fragment(c_frag, 0.0f);

            for (int k = 0; k < 2; k++) {
                half* a_ptr = my_hidden_half + tile_row * 16 * 32 + k * 16;
                const half* b_ptr = weights2 + k * 16 + tile_col * 16 * 32;

                wmma::load_matrix_sync(a_frag, a_ptr, 32);
                wmma::load_matrix_sync(b_frag, b_ptr, 32);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            float* c_ptr = my_smem + tile_row * 16 * 32 + tile_col * 16;
            wmma::store_matrix_sync(c_ptr, c_frag, 32, wmma::mem_row_major);
        }
    }
    __syncwarp();

    // Output
    for (int org = lane_id; org < 16; org += 32) {
        if (batch_start + org >= n_organisms) continue;

        int best = 0;
        float best_val = -1e9f;
        for (int o = 0; o < n_outputs && o < 32; o++) {
            float val = my_smem[org * 32 + o] + __half2float(bias2[o]);
            val = tanh_approx(val);
            if (val > best_val) {
                best_val = val;
                best = o;
            }
        }
        moves[batch_start + org] = best;
    }
}
"#;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║        EPIC 96 V2: WMMA Scaling Benchmark               ║");
    println!("║                                                          ║");
    println!("║  Does bigger NN = bigger WMMA speedup?                  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: Requires CUDA support!");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        run_scaling_bench();
    }
}

#[cfg(feature = "cuda")]
fn run_scaling_bench() {
    const N_ORGANISMS: usize = 10000;
    const N_ITERATIONS: usize = 100;
    const WARMUP: usize = 10;

    println!("Configuration:");
    println!("  Organisms:     {}", N_ORGANISMS);
    println!("  Iterations:    {}", N_ITERATIONS);
    println!();

    let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

    // Compile kernels
    println!("Compiling WMMA kernels...");
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

    let ptx = compile_ptx_with_opts(WMMA_SCALED_KERNEL, opts).expect("PTX compilation failed");

    let module = rt.ctx().load_module(ptx).expect("Module load failed");
    let kernel_16x16 = module
        .load_function("wmma_16x16_forward")
        .expect("16x16 kernel failed");
    let kernel_32x32 = module
        .load_function("wmma_32x32_forward")
        .expect("32x32 kernel failed");

    println!("Kernels compiled successfully!\n");

    // ========================================
    // 16×16 Benchmark
    // ========================================
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│              16×16 NN (Perfect WMMA Fit)                │");
    println!("│              0% padding waste, 256 params/layer         │");
    println!("└─────────────────────────────────────────────────────────┘");

    let result_16 = bench_size(&rt, &kernel_16x16, 16, N_ORGANISMS, N_ITERATIONS, WARMUP);
    println!("  Kernel time:   {:.2} us/iter", result_16.kernel_us);
    println!(
        "  Throughput:    {:.2}M organisms/sec",
        result_16.throughput_m
    );

    // ========================================
    // 32×32 Benchmark
    // ========================================
    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!("│              32×32 NN (4 WMMA Tiles)                    │");
    println!("│              0% padding waste, 1024 params/layer        │");
    println!("└─────────────────────────────────────────────────────────┘");

    let result_32 = bench_size(&rt, &kernel_32x32, 32, N_ORGANISMS, N_ITERATIONS, WARMUP);
    println!("  Kernel time:   {:.2} us/iter", result_32.kernel_us);
    println!(
        "  Throughput:    {:.2}M organisms/sec",
        result_32.throughput_m
    );

    // ========================================
    // CPU Baseline for comparison
    // ========================================
    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!("│                    CPU Baseline                         │");
    println!("│              (8×8 from previous benchmark)              │");
    println!("└─────────────────────────────────────────────────────────┘");
    println!("  Reference: ~405 us/iter, ~24M organisms/sec");

    // ========================================
    // Summary
    // ========================================
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║                    RESULTS SUMMARY                       ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("┌───────────────┬────────────────┬────────────────┬───────────┐");
    println!("│ NN Size       │ Kernel Time    │ Throughput     │ vs CPU 8×8│");
    println!("├───────────────┼────────────────┼────────────────┼───────────┤");
    println!("│ 8×8 (ref)     │ ~17 us         │ ~600M/s        │   24.3×   │");
    println!(
        "│ 16×16         │ {:>8.2} us     │ {:>8.2}M/s     │ {:>7.1}×  │",
        result_16.kernel_us,
        result_16.throughput_m,
        405.0 / result_16.kernel_us
    );
    println!(
        "│ 32×32         │ {:>8.2} us     │ {:>8.2}M/s     │ {:>7.1}×  │",
        result_32.kernel_us,
        result_32.throughput_m,
        405.0 / result_32.kernel_us
    );
    println!("└───────────────┴────────────────┴────────────────┴───────────┘");

    println!("\n✓ WMMA Scaling Benchmark Complete");
}

#[cfg(feature = "cuda")]
struct BenchResult {
    kernel_us: f64,
    throughput_m: f64,
}

#[cfg(feature = "cuda")]
fn bench_size(
    rt: &CudaRuntime,
    kernel: &CudaFunction,
    size: usize,
    n_organisms: usize,
    n_iterations: usize,
    warmup: usize,
) -> BenchResult {
    let mut rng = QRng::new(12345);

    // Allocate buffers
    let n_batches = (n_organisms + 15) / 16;
    let sensors_size = n_batches * 16 * size * size;
    let weights_size = size * size;

    let mut sensors_data = vec![0u16; sensors_size];
    for i in 0..sensors_size {
        sensors_data[i] = half::f16::from_f32((rng.next_f32() - 0.5) * 2.0).to_bits();
    }

    let weights1: Vec<u16> = (0..weights_size)
        .map(|_| half::f16::from_f32((rng.next_f32() - 0.5) * 2.0).to_bits())
        .collect();
    let bias1: Vec<u16> = (0..size)
        .map(|_| half::f16::from_f32((rng.next_f32() - 0.5) * 0.5).to_bits())
        .collect();
    let weights2: Vec<u16> = (0..weights_size)
        .map(|_| half::f16::from_f32((rng.next_f32() - 0.5) * 2.0).to_bits())
        .collect();
    let bias2: Vec<u16> = (0..size)
        .map(|_| half::f16::from_f32((rng.next_f32() - 0.5) * 0.5).to_bits())
        .collect();

    let d_sensors = rt.upload(&sensors_data).expect("Upload sensors");
    let d_weights1 = rt.upload(&weights1).expect("Upload weights1");
    let d_bias1 = rt.upload(&bias1).expect("Upload bias1");
    let d_weights2 = rt.upload(&weights2).expect("Upload weights2");
    let d_bias2 = rt.upload(&bias2).expect("Upload bias2");
    let d_moves = rt.alloc_zeros::<i32>(n_organisms).expect("Alloc moves");

    let n_warps = n_batches;
    let threads_per_block = 128u32;
    let blocks = ((n_warps + 3) / 4) as u32;

    let smem_per_warp = if size == 16 { 256 + 256 } else { 1024 + 1024 };
    let shared_mem_bytes = (4 * smem_per_warp * 4) as u32; // 4 warps, floats + halves

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes,
    };

    let n_i32 = n_organisms as i32;
    let n_outputs = 5i32;

    // Warmup
    for _ in 0..warmup {
        use cudarc::driver::PushKernelArg;
        unsafe {
            rt.stream()
                .launch_builder(kernel)
                .arg(&d_sensors)
                .arg(&d_weights1)
                .arg(&d_bias1)
                .arg(&d_weights2)
                .arg(&d_bias2)
                .arg(&d_moves)
                .arg(&n_i32)
                .arg(&n_outputs)
                .launch(cfg.clone())
                .expect("Launch failed");
        }
    }
    rt.stream().synchronize().expect("Sync failed");

    // Benchmark
    let start = Instant::now();
    for _ in 0..n_iterations {
        use cudarc::driver::PushKernelArg;
        unsafe {
            rt.stream()
                .launch_builder(kernel)
                .arg(&d_sensors)
                .arg(&d_weights1)
                .arg(&d_bias1)
                .arg(&d_weights2)
                .arg(&d_bias2)
                .arg(&d_moves)
                .arg(&n_i32)
                .arg(&n_outputs)
                .launch(cfg.clone())
                .expect("Launch failed");
        }
    }
    rt.stream().synchronize().expect("Sync failed");
    let elapsed = start.elapsed();

    let kernel_us = elapsed.as_micros() as f64 / n_iterations as f64;
    let throughput_m = n_organisms as f64 / (kernel_us / 1_000_000.0) / 1_000_000.0;

    BenchResult {
        kernel_us,
        throughput_m,
    }
}
