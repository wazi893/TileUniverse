//! EPIC 96 V2: Optimized 32×32 WMMA Kernel
//!
//! The previous 32×32 was slow because each warp did all 4 tiles sequentially.
//! This version tries a different approach: process fewer organisms per warp
//! but do the 32×32 matmul more efficiently.
//!
//! Strategy: Use tensor core MMA for the full 32×32 by treating it as
//! a single larger operation with multiple accumulates.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaFunction, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx_with_opts;
#[cfg(feature = "cuda")]
use engine::cuda::CudaRuntime;
#[cfg(feature = "cuda")]
use engine::quantum::QRng;
#[cfg(feature = "cuda")]
use std::time::Instant;

// Optimized 32×32 kernel - use multiple warps per organism batch
#[cfg(feature = "cuda")]
const WMMA_32X32_OPTIMIZED: &str = r#"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

__device__ __forceinline__ float tanh_approx(float x) {
    if (x > 3.0f) return 1.0f;
    if (x < -3.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// Strategy: Each warp handles ONE organism's 32×32 matmul
// This gives maximum parallelism at the cost of more warps needed
extern "C" __global__ void wmma_32x32_parallel(
    const half* __restrict__ sensors,     // [n × 32] padded to [n × 32]
    const half* __restrict__ weights1,    // [32 × 32] col-major
    const half* __restrict__ bias1,       // [32]
    const half* __restrict__ weights2,    // [32 × 32] col-major
    const half* __restrict__ bias2,       // [32]
    int* __restrict__ moves,
    int n_organisms,
    int n_outputs
) {
    // One warp per organism (more parallelism, less per-warp work)
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;

    if (warp_id >= n_organisms) return;

    // Each organism has 32 sensors, we need to compute 32 hidden activations
    // hidden[h] = Σ_i sensor[i] * weights[i][h]
    // This is a 1×32 × 32×32 → 1×32 operation
    // But WMMA needs 16×16, so we'll do it differently...

    // Actually, let's batch 16 organisms and use proper 16×32 × 32×32 → 16×32
    // But that requires reorganizing. Let's try a simpler approach:

    // Use registers for the small per-organism computation
    // Load weights into shared memory once, then each thread computes part of the result

    extern __shared__ half shared_weights[];

    // Cooperatively load weights1 into shared memory
    for (int i = lane_id; i < 1024; i += 32) {
        shared_weights[i] = weights1[i];
    }
    __syncwarp();

    // Each organism's sensors
    const half* my_sensors = sensors + warp_id * 32;

    // Compute hidden layer (32 outputs)
    float hidden[32];
    for (int h = 0; h < 32; h++) {
        float sum = __half2float(bias1[h]);
        for (int i = 0; i < 32; i++) {
            // weights1 is col-major: [i][h] at position h*32 + i
            sum += __half2float(my_sensors[i]) * __half2float(shared_weights[h * 32 + i]);
        }
        hidden[h] = tanh_approx(sum);
    }

    // Load weights2
    __syncwarp();
    for (int i = lane_id; i < 1024; i += 32) {
        shared_weights[i] = weights2[i];
    }
    __syncwarp();

    // Compute output layer
    float output[32];
    for (int o = 0; o < 32 && o < n_outputs; o++) {
        float sum = __half2float(bias2[o]);
        for (int h = 0; h < 32; h++) {
            // weights2 is col-major: [h][o] at position o*32 + h
            sum += hidden[h] * __half2float(shared_weights[o * 32 + h]);
        }
        output[o] = tanh_approx(sum);
    }

    // Argmax (only lane 0 writes)
    if (lane_id == 0) {
        int best = 0;
        float best_val = output[0];
        for (int o = 1; o < n_outputs && o < 32; o++) {
            if (output[o] > best_val) {
                best_val = output[o];
                best = o;
            }
        }
        moves[warp_id] = best;
    }
}

// Alternative: Proper WMMA-based 32×32 with 16-organism batching
// Use 4 warps cooperatively for the 4 tiles
extern "C" __global__ void wmma_32x32_cooperative(
    const half* __restrict__ sensors,     // [batches × 16 × 32]
    const half* __restrict__ weights1,    // [32 × 32] col-major
    const half* __restrict__ bias1,       // [32]
    const half* __restrict__ weights2,    // [32 × 32] col-major
    const half* __restrict__ bias2,       // [32]
    int* __restrict__ moves,
    int n_organisms,
    int n_outputs
) {
    // 4 warps work together on one batch of 16 organisms
    // Each warp handles one 16×16 output tile
    const int block_batch = blockIdx.x;  // Which batch of 16 organisms
    const int warp_in_block = threadIdx.x / 32;  // 0-3
    const int lane_id = threadIdx.x % 32;

    const int batch_start = block_batch * 16;
    if (batch_start >= n_organisms) return;

    // Warp assignment for 2×2 output tiles:
    // Warp 0: C[0:16, 0:16]
    // Warp 1: C[0:16, 16:32]
    // (We only need top row since we have 16 organisms with 32 outputs each)
    const int tile_col = warp_in_block;  // 0, 1, 2, 3 -> output columns 0-15, 16-31, ...

    if (tile_col >= 2) return;  // Only need 2 column tiles for 32 outputs

    extern __shared__ char shared_mem[];
    float* result_smem = (float*)shared_mem;  // 16 × 32 floats
    half* hidden_smem = (half*)(result_smem + 16 * 32);  // 16 × 32 halves

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Layer 1: [16×32] × [32×32] → [16×32]
    // Each warp computes one 16×16 output tile
    wmma::fill_fragment(c_frag, 0.0f);

    // Accumulate over K=32 (two 16×16 tiles)
    for (int k = 0; k < 2; k++) {
        // A tile: sensors[batch][0:16, k*16:(k+1)*16]
        const half* a_ptr = sensors + block_batch * 16 * 32 + k * 16;
        // B tile: weights1[k*16:(k+1)*16, tile_col*16:(tile_col+1)*16]
        const half* b_ptr = weights1 + tile_col * 16 * 32 + k * 16;

        wmma::load_matrix_sync(a_frag, a_ptr, 32);  // stride=32
        wmma::load_matrix_sync(b_frag, b_ptr, 32);  // stride=32
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store to shared memory
    float* my_result = result_smem + tile_col * 16;
    wmma::store_matrix_sync(my_result, c_frag, 32, wmma::mem_row_major);
    __syncthreads();

    // Apply bias + tanh (all warps cooperate)
    for (int i = threadIdx.x; i < 16 * 32; i += blockDim.x) {
        int col = i % 32;
        result_smem[i] = tanh_approx(result_smem[i] + __half2float(bias1[col]));
    }
    __syncthreads();

    // Convert to half
    for (int i = threadIdx.x; i < 16 * 32; i += blockDim.x) {
        hidden_smem[i] = __float2half(result_smem[i]);
    }
    __syncthreads();

    // Layer 2: [16×32] × [32×32] → [16×32]
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < 2; k++) {
        half* a_ptr = hidden_smem + k * 16;
        const half* b_ptr = weights2 + tile_col * 16 * 32 + k * 16;

        wmma::load_matrix_sync(a_frag, a_ptr, 32);
        wmma::load_matrix_sync(b_frag, b_ptr, 32);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(my_result, c_frag, 32, wmma::mem_row_major);
    __syncthreads();

    // Argmax (only warp 0, and only for valid organisms)
    if (warp_in_block == 0) {
        for (int org = lane_id; org < 16; org += 32) {
            if (batch_start + org >= n_organisms) continue;

            int best = 0;
            float best_val = -1e9f;
            for (int o = 0; o < n_outputs && o < 32; o++) {
                float val = result_smem[org * 32 + o] + __half2float(bias2[o]);
                val = tanh_approx(val);
                if (val > best_val) {
                    best_val = val;
                    best = o;
                }
            }
            moves[batch_start + org] = best;
        }
    }
}
"#;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║      EPIC 96 V2: Optimized 32×32 WMMA Exploration       ║");
    println!("║                                                          ║");
    println!("║  Can we make 32×32 competitive with 16×16?              ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("ERROR: Requires CUDA support!");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        run_optimized_bench();
    }
}

#[cfg(feature = "cuda")]
fn run_optimized_bench() {
    const N_ORGANISMS: usize = 10000;
    const N_ITERATIONS: usize = 100;
    const WARMUP: usize = 10;

    let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

    println!("Compiling optimized 32×32 kernels...");
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

    let ptx = compile_ptx_with_opts(WMMA_32X32_OPTIMIZED, opts).expect("PTX compilation failed");

    let module = rt.ctx().load_module(ptx).expect("Module load failed");
    let kernel_parallel = module
        .load_function("wmma_32x32_parallel")
        .expect("parallel kernel failed");
    let kernel_coop = module
        .load_function("wmma_32x32_cooperative")
        .expect("cooperative kernel failed");

    println!("Kernels compiled!\n");

    let mut rng = QRng::new(12345);

    // ========================================
    // Setup data
    // ========================================

    // For parallel kernel: each organism has 32 sensors
    let sensors_parallel: Vec<u16> = (0..N_ORGANISMS * 32)
        .map(|_| half::f16::from_f32((rng.next_f32() - 0.5) * 2.0).to_bits())
        .collect();

    // For cooperative kernel: batched as [batches × 16 × 32]
    let n_batches = (N_ORGANISMS + 15) / 16;
    let sensors_coop: Vec<u16> = (0..n_batches * 16 * 32)
        .map(|_| half::f16::from_f32((rng.next_f32() - 0.5) * 2.0).to_bits())
        .collect();

    let weights1: Vec<u16> = (0..32 * 32)
        .map(|_| half::f16::from_f32((rng.next_f32() - 0.5) * 2.0).to_bits())
        .collect();
    let bias1: Vec<u16> = (0..32)
        .map(|_| half::f16::from_f32((rng.next_f32() - 0.5) * 0.5).to_bits())
        .collect();
    let weights2: Vec<u16> = (0..32 * 32)
        .map(|_| half::f16::from_f32((rng.next_f32() - 0.5) * 2.0).to_bits())
        .collect();
    let bias2: Vec<u16> = (0..32)
        .map(|_| half::f16::from_f32((rng.next_f32() - 0.5) * 0.5).to_bits())
        .collect();

    let d_sensors_par = rt.upload(&sensors_parallel).expect("upload");
    let d_sensors_coop = rt.upload(&sensors_coop).expect("upload");
    let d_weights1 = rt.upload(&weights1).expect("upload");
    let d_bias1 = rt.upload(&bias1).expect("upload");
    let d_weights2 = rt.upload(&weights2).expect("upload");
    let d_bias2 = rt.upload(&bias2).expect("upload");
    let d_moves = rt.alloc_zeros::<i32>(N_ORGANISMS).expect("alloc");

    let n_i32 = N_ORGANISMS as i32;
    let n_outputs = 8i32;

    // ========================================
    // Benchmark: Parallel (one warp per organism)
    // ========================================
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│     32×32 Parallel (1 warp per organism, no WMMA)      │");
    println!("└─────────────────────────────────────────────────────────┘");

    let threads = 128u32;
    let blocks_par = ((N_ORGANISMS * 32 + threads as usize - 1) / threads as usize) as u32;
    let cfg_par = LaunchConfig {
        grid_dim: (blocks_par, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 2 * 1024, // weights
    };

    // Warmup
    for _ in 0..WARMUP {
        use cudarc::driver::PushKernelArg;
        unsafe {
            rt.stream()
                .launch_builder(&kernel_parallel)
                .arg(&d_sensors_par)
                .arg(&d_weights1)
                .arg(&d_bias1)
                .arg(&d_weights2)
                .arg(&d_bias2)
                .arg(&d_moves)
                .arg(&n_i32)
                .arg(&n_outputs)
                .launch(cfg_par.clone())
                .expect("launch");
        }
    }
    rt.stream().synchronize().expect("sync");

    let start = Instant::now();
    for _ in 0..N_ITERATIONS {
        use cudarc::driver::PushKernelArg;
        unsafe {
            rt.stream()
                .launch_builder(&kernel_parallel)
                .arg(&d_sensors_par)
                .arg(&d_weights1)
                .arg(&d_bias1)
                .arg(&d_weights2)
                .arg(&d_bias2)
                .arg(&d_moves)
                .arg(&n_i32)
                .arg(&n_outputs)
                .launch(cfg_par.clone())
                .expect("launch");
        }
    }
    rt.stream().synchronize().expect("sync");
    let elapsed_par = start.elapsed();
    let us_par = elapsed_par.as_micros() as f64 / N_ITERATIONS as f64;
    let throughput_par = N_ORGANISMS as f64 / (us_par / 1_000_000.0) / 1_000_000.0;

    println!("  Kernel time:   {:.2} us/iter", us_par);
    println!("  Throughput:    {:.2}M organisms/sec", throughput_par);

    // ========================================
    // Benchmark: Cooperative (4 warps per batch, WMMA)
    // ========================================
    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!("│     32×32 Cooperative (4 warps per 16 orgs, WMMA)      │");
    println!("└─────────────────────────────────────────────────────────┘");

    let cfg_coop = LaunchConfig {
        grid_dim: (n_batches as u32, 1, 1),
        block_dim: (128, 1, 1),                               // 4 warps
        shared_mem_bytes: (16 * 32 * 4 + 16 * 32 * 2) as u32, // floats + halves
    };

    // Warmup
    for _ in 0..WARMUP {
        use cudarc::driver::PushKernelArg;
        unsafe {
            rt.stream()
                .launch_builder(&kernel_coop)
                .arg(&d_sensors_coop)
                .arg(&d_weights1)
                .arg(&d_bias1)
                .arg(&d_weights2)
                .arg(&d_bias2)
                .arg(&d_moves)
                .arg(&n_i32)
                .arg(&n_outputs)
                .launch(cfg_coop.clone())
                .expect("launch");
        }
    }
    rt.stream().synchronize().expect("sync");

    let start = Instant::now();
    for _ in 0..N_ITERATIONS {
        use cudarc::driver::PushKernelArg;
        unsafe {
            rt.stream()
                .launch_builder(&kernel_coop)
                .arg(&d_sensors_coop)
                .arg(&d_weights1)
                .arg(&d_bias1)
                .arg(&d_weights2)
                .arg(&d_bias2)
                .arg(&d_moves)
                .arg(&n_i32)
                .arg(&n_outputs)
                .launch(cfg_coop.clone())
                .expect("launch");
        }
    }
    rt.stream().synchronize().expect("sync");
    let elapsed_coop = start.elapsed();
    let us_coop = elapsed_coop.as_micros() as f64 / N_ITERATIONS as f64;
    let throughput_coop = N_ORGANISMS as f64 / (us_coop / 1_000_000.0) / 1_000_000.0;

    println!("  Kernel time:   {:.2} us/iter", us_coop);
    println!("  Throughput:    {:.2}M organisms/sec", throughput_coop);

    // ========================================
    // Summary
    // ========================================
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║                    RESULTS SUMMARY                       ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("┌──────────────────────┬────────────┬─────────────┬──────────┐");
    println!("│ Approach             │ Time (us)  │ Throughput  │ vs 16×16 │");
    println!("├──────────────────────┼────────────┼─────────────┼──────────┤");
    println!("│ 16×16 WMMA (ref)     │       9.54 │   1048 M/s  │   1.00×  │");
    println!("│ 32×32 Original       │      20.86 │    479 M/s  │   0.46×  │");
    println!(
        "│ 32×32 Parallel       │ {:>10.2} │ {:>7.2} M/s  │ {:>6.2}×  │",
        us_par,
        throughput_par,
        throughput_par / 1048.0
    );
    println!(
        "│ 32×32 Cooperative    │ {:>10.2} │ {:>7.2} M/s  │ {:>6.2}×  │",
        us_coop,
        throughput_coop,
        throughput_coop / 1048.0
    );
    println!("└──────────────────────┴────────────┴─────────────┴──────────┘");

    if throughput_coop > 1000.0 || throughput_par > 1000.0 {
        println!("\n🎉 Found a 32×32 approach that matches or beats 16×16!");
    } else {
        println!("\n📊 16×16 remains the sweet spot for this workload.");
        println!("   32×32 has 4× more compute but overhead dominates.");
    }
}
