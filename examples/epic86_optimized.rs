// EPIC 86: Optimized Batched Gates - Double-Buffered Fragment Conversion
//
// Optimization: Overlap store_matrix_sync with load_matrix_sync using ping-pong buffers
// to hide shared memory latency.
//
// Run with: cargo run --example epic86_optimized --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use std::sync::Arc;
    use std::time::Instant;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       EPIC 86: OPTIMIZED BATCHED GATES                           ║");
    println!("║       Double-buffered fragment conversion                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let ctx = Arc::new(CudaContext::new(0).expect("Failed to create CUDA context"));
    let stream = ctx.new_stream().expect("Failed to create CUDA stream");

    println!("Compiling kernels...");

    let cuda_include = std::env::var("CUDA_PATH")
        .map(|p| format!("{}/include", p))
        .unwrap_or_else(|_| {
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include".to_string()
        });

    let opts = CompileOptions {
        arch: Some("compute_89"),
        include_paths: vec![cuda_include.clone()],
        options: vec!["-default-device".to_string()],
        ..Default::default()
    };

    let kernel_src = r#"
#include <mma.h>
using namespace nvcuda;

// ============================================================================
// ORIGINAL: Sequential store-then-load
// ============================================================================
extern "C" __global__
void batched_original(
    half* __restrict__ states,
    const half* __restrict__ gates,
    int tiles_per_state,
    int num_gates
) {
    int state_id = blockIdx.y;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= tiles_per_state) return;

    int state_offset = state_id * tiles_per_state * 256;
    half* tile = states + state_offset + warp_id * 256;

    __shared__ half shared_a[8][256];
    __shared__ half shared_b[8][256];
    int local_warp = threadIdx.x / 32;
    half* buf_a = shared_a[local_warp];
    half* buf_b = shared_b[local_warp];

    int lane = threadIdx.x % 32;

    for (int i = lane; i < 256; i += 32) {
        buf_a[i] = tile[i];
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> state_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> gate_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> result_frag;

    half* read_buf = buf_a;
    half* write_buf = buf_b;

    for (int g = 0; g < num_gates; g++) {
        wmma::load_matrix_sync(state_frag, read_buf, 16);
        wmma::load_matrix_sync(gate_frag, gates + g * 256, 16);

        #pragma unroll
        for (int i = 0; i < result_frag.num_elements; i++) {
            result_frag.x[i] = __float2half(0.0f);
        }

        wmma::mma_sync(result_frag, state_frag, gate_frag, result_frag);
        wmma::store_matrix_sync(write_buf, result_frag, 16, wmma::mem_row_major);
        __syncwarp();

        half* tmp = read_buf;
        read_buf = write_buf;
        write_buf = tmp;
    }

    half* result = (num_gates % 2 == 1) ? buf_b : buf_a;
    for (int i = lane; i < 256; i += 32) {
        tile[i] = result[i];
    }
}

// ============================================================================
// OPTIMIZED V1: Prefetch gate matrix while computing
// ============================================================================
extern "C" __global__
void batched_prefetch_gate(
    half* __restrict__ states,
    const half* __restrict__ gates,
    int tiles_per_state,
    int num_gates
) {
    int state_id = blockIdx.y;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= tiles_per_state) return;

    int state_offset = state_id * tiles_per_state * 256;
    half* tile = states + state_offset + warp_id * 256;

    __shared__ half shared_a[8][256];
    __shared__ half shared_b[8][256];
    __shared__ half gate_buf[8][256];  // Prefetch buffer for next gate
    int local_warp = threadIdx.x / 32;
    half* buf_a = shared_a[local_warp];
    half* buf_b = shared_b[local_warp];
    half* gbuf = gate_buf[local_warp];

    int lane = threadIdx.x % 32;

    for (int i = lane; i < 256; i += 32) {
        buf_a[i] = tile[i];
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> state_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> gate_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> next_gate_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> result_frag;

    half* read_buf = buf_a;
    half* write_buf = buf_b;

    // Prefetch first gate
    wmma::load_matrix_sync(gate_frag, gates, 16);

    for (int g = 0; g < num_gates; g++) {
        // Load state
        wmma::load_matrix_sync(state_frag, read_buf, 16);

        // Prefetch next gate while we compute (if not last)
        if (g + 1 < num_gates) {
            // Load next gate to shared memory with regular loads
            for (int i = lane; i < 256; i += 32) {
                gbuf[i] = gates[(g + 1) * 256 + i];
            }
        }

        // Zero and compute
        #pragma unroll
        for (int i = 0; i < result_frag.num_elements; i++) {
            result_frag.x[i] = __float2half(0.0f);
        }

        wmma::mma_sync(result_frag, state_frag, gate_frag, result_frag);
        wmma::store_matrix_sync(write_buf, result_frag, 16, wmma::mem_row_major);
        __syncwarp();

        // Load prefetched gate into fragment
        if (g + 1 < num_gates) {
            wmma::load_matrix_sync(gate_frag, gbuf, 16);
        }

        half* tmp = read_buf;
        read_buf = write_buf;
        write_buf = tmp;
    }

    half* result = (num_gates % 2 == 1) ? buf_b : buf_a;
    for (int i = lane; i < 256; i += 32) {
        tile[i] = result[i];
    }
}

// ============================================================================
// OPTIMIZED V2: Manual unroll by 2 - interleave two gate applications
// ============================================================================
extern "C" __global__
void batched_unroll2(
    half* __restrict__ states,
    const half* __restrict__ gates,
    int tiles_per_state,
    int num_gates
) {
    int state_id = blockIdx.y;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= tiles_per_state) return;

    int state_offset = state_id * tiles_per_state * 256;
    half* tile = states + state_offset + warp_id * 256;

    __shared__ half shared_a[8][256];
    __shared__ half shared_b[8][256];
    int local_warp = threadIdx.x / 32;
    half* buf_a = shared_a[local_warp];
    half* buf_b = shared_b[local_warp];

    int lane = threadIdx.x % 32;

    for (int i = lane; i < 256; i += 32) {
        buf_a[i] = tile[i];
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> state_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> gate_frag0;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> gate_frag1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> result_frag;

    half* read_buf = buf_a;
    half* write_buf = buf_b;

    int g = 0;
    // Process pairs of gates
    for (; g + 1 < num_gates; g += 2) {
        // Load both gates upfront
        wmma::load_matrix_sync(gate_frag0, gates + g * 256, 16);
        wmma::load_matrix_sync(gate_frag1, gates + (g + 1) * 256, 16);

        // First gate
        wmma::load_matrix_sync(state_frag, read_buf, 16);
        #pragma unroll
        for (int i = 0; i < result_frag.num_elements; i++) {
            result_frag.x[i] = __float2half(0.0f);
        }
        wmma::mma_sync(result_frag, state_frag, gate_frag0, result_frag);
        wmma::store_matrix_sync(write_buf, result_frag, 16, wmma::mem_row_major);
        __syncwarp();

        // Swap
        half* tmp = read_buf; read_buf = write_buf; write_buf = tmp;

        // Second gate
        wmma::load_matrix_sync(state_frag, read_buf, 16);
        #pragma unroll
        for (int i = 0; i < result_frag.num_elements; i++) {
            result_frag.x[i] = __float2half(0.0f);
        }
        wmma::mma_sync(result_frag, state_frag, gate_frag1, result_frag);
        wmma::store_matrix_sync(write_buf, result_frag, 16, wmma::mem_row_major);
        __syncwarp();

        // Swap
        tmp = read_buf; read_buf = write_buf; write_buf = tmp;
    }

    // Handle odd remaining gate
    if (g < num_gates) {
        wmma::load_matrix_sync(gate_frag0, gates + g * 256, 16);
        wmma::load_matrix_sync(state_frag, read_buf, 16);
        #pragma unroll
        for (int i = 0; i < result_frag.num_elements; i++) {
            result_frag.x[i] = __float2half(0.0f);
        }
        wmma::mma_sync(result_frag, state_frag, gate_frag0, result_frag);
        wmma::store_matrix_sync(write_buf, result_frag, 16, wmma::mem_row_major);
        __syncwarp();
        half* tmp = read_buf; read_buf = write_buf; write_buf = tmp;
    }

    half* result = read_buf;  // After all swaps, result is in read_buf
    for (int i = lane; i < 256; i += 32) {
        tile[i] = result[i];
    }
}

// ============================================================================
// OPTIMIZED V3: Use accumulator as passthrough (skip shared for intermediates)
// Chain: A × G0 = R0, R0 × G1 = R1, ...
// Problem: accumulator can't be used as matrix_a directly
// Workaround: Store to shared ONCE every N gates
// ============================================================================
extern "C" __global__
void batched_reduced_sync(
    half* __restrict__ states,
    const half* __restrict__ gates,
    int tiles_per_state,
    int num_gates
) {
    int state_id = blockIdx.y;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= tiles_per_state) return;

    int state_offset = state_id * tiles_per_state * 256;
    half* tile = states + state_offset + warp_id * 256;

    __shared__ half shared_a[8][256];
    __shared__ half shared_b[8][256];
    int local_warp = threadIdx.x / 32;
    half* buf_a = shared_a[local_warp];
    half* buf_b = shared_b[local_warp];

    int lane = threadIdx.x % 32;

    for (int i = lane; i < 256; i += 32) {
        buf_a[i] = tile[i];
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> state_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> gate_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> result_frag;

    half* read_buf = buf_a;
    half* write_buf = buf_b;

    // Process gates normally but try to reduce sync overhead
    for (int g = 0; g < num_gates; g++) {
        wmma::load_matrix_sync(state_frag, read_buf, 16);
        wmma::load_matrix_sync(gate_frag, gates + g * 256, 16);

        #pragma unroll
        for (int i = 0; i < result_frag.num_elements; i++) {
            result_frag.x[i] = __float2half(0.0f);
        }

        wmma::mma_sync(result_frag, state_frag, gate_frag, result_frag);
        wmma::store_matrix_sync(write_buf, result_frag, 16, wmma::mem_row_major);

        // Only sync every other iteration? NO - this breaks correctness
        // The store must complete before the next load from write_buf
        __syncwarp();

        half* tmp = read_buf;
        read_buf = write_buf;
        write_buf = tmp;
    }

    half* result = (num_gates % 2 == 1) ? buf_b : buf_a;
    for (int i = lane; i < 256; i += 32) {
        tile[i] = result[i];
    }
}

// ============================================================================
// OPTIMIZED V4: Manual 4x unroll with prefetched gates
// ============================================================================
extern "C" __global__
void batched_unroll4(
    half* __restrict__ states,
    const half* __restrict__ gates,
    int tiles_per_state,
    int num_gates
) {
    int state_id = blockIdx.y;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= tiles_per_state) return;

    int state_offset = state_id * tiles_per_state * 256;
    half* tile = states + state_offset + warp_id * 256;

    __shared__ half shared_a[8][256];
    __shared__ half shared_b[8][256];
    int local_warp = threadIdx.x / 32;
    half* buf_a = shared_a[local_warp];
    half* buf_b = shared_b[local_warp];

    int lane = threadIdx.x % 32;

    for (int i = lane; i < 256; i += 32) {
        buf_a[i] = tile[i];
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> state_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> g0, g1, g2, g3;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> result_frag;

    half* read_buf = buf_a;
    half* write_buf = buf_b;

    int g = 0;
    // Process quads
    for (; g + 3 < num_gates; g += 4) {
        // Prefetch all 4 gates
        wmma::load_matrix_sync(g0, gates + (g + 0) * 256, 16);
        wmma::load_matrix_sync(g1, gates + (g + 1) * 256, 16);
        wmma::load_matrix_sync(g2, gates + (g + 2) * 256, 16);
        wmma::load_matrix_sync(g3, gates + (g + 3) * 256, 16);

        // Gate 0
        wmma::load_matrix_sync(state_frag, read_buf, 16);
        for (int i = 0; i < result_frag.num_elements; i++) result_frag.x[i] = __float2half(0.0f);
        wmma::mma_sync(result_frag, state_frag, g0, result_frag);
        wmma::store_matrix_sync(write_buf, result_frag, 16, wmma::mem_row_major);
        __syncwarp();
        half* tmp = read_buf; read_buf = write_buf; write_buf = tmp;

        // Gate 1
        wmma::load_matrix_sync(state_frag, read_buf, 16);
        for (int i = 0; i < result_frag.num_elements; i++) result_frag.x[i] = __float2half(0.0f);
        wmma::mma_sync(result_frag, state_frag, g1, result_frag);
        wmma::store_matrix_sync(write_buf, result_frag, 16, wmma::mem_row_major);
        __syncwarp();
        tmp = read_buf; read_buf = write_buf; write_buf = tmp;

        // Gate 2
        wmma::load_matrix_sync(state_frag, read_buf, 16);
        for (int i = 0; i < result_frag.num_elements; i++) result_frag.x[i] = __float2half(0.0f);
        wmma::mma_sync(result_frag, state_frag, g2, result_frag);
        wmma::store_matrix_sync(write_buf, result_frag, 16, wmma::mem_row_major);
        __syncwarp();
        tmp = read_buf; read_buf = write_buf; write_buf = tmp;

        // Gate 3
        wmma::load_matrix_sync(state_frag, read_buf, 16);
        for (int i = 0; i < result_frag.num_elements; i++) result_frag.x[i] = __float2half(0.0f);
        wmma::mma_sync(result_frag, state_frag, g3, result_frag);
        wmma::store_matrix_sync(write_buf, result_frag, 16, wmma::mem_row_major);
        __syncwarp();
        tmp = read_buf; read_buf = write_buf; write_buf = tmp;
    }

    // Handle remaining gates
    for (; g < num_gates; g++) {
        wmma::load_matrix_sync(g0, gates + g * 256, 16);
        wmma::load_matrix_sync(state_frag, read_buf, 16);
        for (int i = 0; i < result_frag.num_elements; i++) result_frag.x[i] = __float2half(0.0f);
        wmma::mma_sync(result_frag, state_frag, g0, result_frag);
        wmma::store_matrix_sync(write_buf, result_frag, 16, wmma::mem_row_major);
        __syncwarp();
        half* tmp = read_buf; read_buf = write_buf; write_buf = tmp;
    }

    for (int i = lane; i < 256; i += 32) {
        tile[i] = read_buf[i];
    }
}
"#;

    let ptx = compile_ptx_with_opts(kernel_src, opts).expect("Failed to compile kernels");

    let module = ctx.load_module(ptx).expect("Failed to load PTX module");

    let original_fn = module
        .load_function("batched_original")
        .expect("batched_original");
    let prefetch_fn = module
        .load_function("batched_prefetch_gate")
        .expect("batched_prefetch_gate");
    let unroll2_fn = module
        .load_function("batched_unroll2")
        .expect("batched_unroll2");
    let unroll4_fn = module
        .load_function("batched_unroll4")
        .expect("batched_unroll4");

    println!("  All kernels compiled successfully\n");

    // Configuration
    let num_states = 1024;
    let tiles_per_state = 16;
    let total_tiles = num_states * tiles_per_state;
    let state_elements = total_tiles * 256;

    // Create unitary gate matrices
    let max_gates = 100;
    let mut gate_data: Vec<u16> = Vec::with_capacity(max_gates * 256);
    let norm_factor: f32 = 0.25; // 1/sqrt(16) for unitary Hadamard

    for g in 0..max_gates {
        for row in 0..16 {
            for col in 0..16 {
                let val = if g % 2 == 0 {
                    let sign = if ((row as u32) & (col as u32)).count_ones() % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    sign * norm_factor
                } else {
                    if row == col { 1.0 } else { 0.0 }
                };
                gate_data.push(half::f16::from_f32(val).to_bits());
            }
        }
    }
    let gates_gpu: CudaSlice<u16> = stream.memcpy_stod(&gate_data).expect("gates");

    // Initial state
    let mut initial_state: Vec<u16> = vec![half::f16::from_f32(0.0).to_bits(); state_elements];
    for tile in 0..total_tiles {
        initial_state[tile * 256] = half::f16::from_f32(1.0).to_bits();
    }

    let warps_per_block = 8u32;
    let threads_per_block = warps_per_block * 32;
    let blocks_x = (tiles_per_state as u32 + warps_per_block - 1) / warps_per_block;
    let blocks_y = num_states as u32;

    let cfg = LaunchConfig {
        grid_dim: (blocks_x, blocks_y, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 8 * 256 * 2 * 3, // Extra for prefetch buffer
    };

    let tiles_i32 = tiles_per_state as i32;

    let warmup = 10;
    let iters = 50;

    println!("═══════════════════════════════════════════════════════════════════");
    println!(" OPTIMIZATION COMPARISON");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let gate_counts = [10, 20, 50, 100];

    println!("┌──────────┬────────────────┬────────────────┬────────────────┬────────────────┐");
    println!("│  Gates   │  Original(ms)  │  Prefetch(ms)  │  Unroll2(ms)   │  Unroll4(ms)   │");
    println!("├──────────┼────────────────┼────────────────┼────────────────┼────────────────┤");

    for num_gates in gate_counts {
        let gates_i32 = num_gates as i32;

        // Warmup
        let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
        for _ in 0..warmup {
            unsafe {
                stream
                    .launch_builder(&original_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(cfg)
                    .ok();
                stream
                    .launch_builder(&prefetch_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(cfg)
                    .ok();
                stream
                    .launch_builder(&unroll2_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(cfg)
                    .ok();
                stream
                    .launch_builder(&unroll4_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(cfg)
                    .ok();
            }
        }
        stream.synchronize().expect("sync");

        // Benchmark original
        let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream
                    .launch_builder(&original_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(cfg)
                    .expect("original");
            }
        }
        stream.synchronize().expect("sync");
        let original_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Benchmark prefetch
        let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream
                    .launch_builder(&prefetch_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(cfg)
                    .expect("prefetch");
            }
        }
        stream.synchronize().expect("sync");
        let prefetch_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Benchmark unroll2
        let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream
                    .launch_builder(&unroll2_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(cfg)
                    .expect("unroll2");
            }
        }
        stream.synchronize().expect("sync");
        let unroll2_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Benchmark unroll4
        let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream
                    .launch_builder(&unroll4_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(cfg)
                    .expect("unroll4");
            }
        }
        stream.synchronize().expect("sync");
        let unroll4_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        println!(
            "│ {:>8} │ {:>12.3}ms │ {:>12.3}ms │ {:>12.3}ms │ {:>12.3}ms │",
            num_gates, original_ms, prefetch_ms, unroll2_ms, unroll4_ms
        );
    }
    println!("└──────────┴────────────────┴────────────────┴────────────────┴────────────────┘\n");

    // Verify correctness of best variant
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" CORRECTNESS VERIFICATION");
    println!("═══════════════════════════════════════════════════════════════════\n");

    for (name, kernel_fn) in [
        ("Original", &original_fn),
        ("Prefetch", &prefetch_fn),
        ("Unroll2", &unroll2_fn),
        ("Unroll4", &unroll4_fn),
    ] {
        let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
        let gates_i32 = 10i32;

        unsafe {
            stream
                .launch_builder(kernel_fn)
                .arg(&states)
                .arg(&gates_gpu)
                .arg(&tiles_i32)
                .arg(&gates_i32)
                .launch(cfg)
                .expect("kernel");
        }
        stream.synchronize().expect("sync");

        let mut result: Vec<u16> = vec![0u16; state_elements];
        stream.memcpy_dtoh(&states, &mut result).expect("dtoh");

        let mut norm_sq: f64 = 0.0;
        for i in 0..256 {
            let val = half::f16::from_bits(result[i]).to_f32() as f64;
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        let status = if (norm - 1.0).abs() < 0.1 {
            "✓"
        } else {
            "✗"
        };
        println!("  {}: norm = {:.6} {}", name, norm, status);
    }
    println!();
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires 'cuda' and 'perf-bench' features.");
}
