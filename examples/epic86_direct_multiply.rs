// EPIC 86: Direct Matrix Multiply vs WMMA
//
// Hypothesis: The 2,500× gap between theoretical and actual fragment conversion
// suggests WMMA intrinsics themselves are the bottleneck, not memory.
//
// Test: Compare direct 16×16 multiply (CUDA cores) vs WMMA batched gates.
// The direct path loses tensor cores but eliminates fragment conversion entirely.
//
// Run with: cargo run --example epic86_direct_multiply --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use std::sync::Arc;
    use std::time::Instant;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       EPIC 86: DIRECT MULTIPLY vs WMMA                           ║");
    println!("║       Testing if CUDA cores beat tensor cores for this use case  ║");
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
#include <cuda_fp16.h>
using namespace nvcuda;

// ============================================================================
// WMMA BATCHED: Our current best (for comparison)
// ============================================================================
extern "C" __global__
void wmma_batched(
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
// DIRECT MULTIPLY V1: One thread per output element (16 threads per row)
// Each warp handles 2 rows (32 threads / 16 cols = 2 rows)
// ============================================================================
extern "C" __global__
void direct_multiply_v1(
    half* __restrict__ states,
    const half* __restrict__ gates,
    int tiles_per_state,
    int num_gates
) {
    // 256 threads per block = 16 warps
    // Each warp handles 1 tile (16x16 = 256 elements, 32 threads per warp)
    // So we need 2 passes per row or different mapping

    int state_id = blockIdx.y;
    int tile_id = blockIdx.x;
    if (tile_id >= tiles_per_state) return;

    int state_offset = state_id * tiles_per_state * 256;
    half* tile_ptr = states + state_offset + tile_id * 256;

    // Each thread handles one element of the 16x16 matrix
    // threadIdx.x in [0, 255] maps to [row, col]
    int local_row = threadIdx.x / 16;
    int local_col = threadIdx.x % 16;

    __shared__ half state_tile[16][16];
    __shared__ half gate_tile[16][16];

    // Load state tile to shared memory
    state_tile[local_row][local_col] = tile_ptr[threadIdx.x];
    __syncthreads();

    // Apply each gate
    for (int g = 0; g < num_gates; g++) {
        // Load gate tile
        gate_tile[local_row][local_col] = gates[g * 256 + threadIdx.x];
        __syncthreads();

        // Compute one element: result[row][col] = sum_k(state[row][k] * gate[k][col])
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            sum += __half2float(state_tile[local_row][k]) * __half2float(gate_tile[k][local_col]);
        }

        __syncthreads();

        // Write result back to state_tile for next iteration
        state_tile[local_row][local_col] = __float2half(sum);
        __syncthreads();
    }

    // Store final result to DRAM
    tile_ptr[threadIdx.x] = state_tile[local_row][local_col];
}

// ============================================================================
// DIRECT MULTIPLY V2: Use half2 for vectorized loads (2x throughput)
// ============================================================================
extern "C" __global__
void direct_multiply_v2(
    half* __restrict__ states,
    const half* __restrict__ gates,
    int tiles_per_state,
    int num_gates
) {
    int state_id = blockIdx.y;
    int tile_id = blockIdx.x;
    if (tile_id >= tiles_per_state) return;

    int state_offset = state_id * tiles_per_state * 256;
    half* tile_ptr = states + state_offset + tile_id * 256;

    int local_row = threadIdx.x / 16;
    int local_col = threadIdx.x % 16;

    __shared__ half state_tile[16][16];
    __shared__ half gate_tile[16][16];

    // Load state tile
    state_tile[local_row][local_col] = tile_ptr[threadIdx.x];
    __syncthreads();

    for (int g = 0; g < num_gates; g++) {
        // Load gate tile
        gate_tile[local_row][local_col] = gates[g * 256 + threadIdx.x];
        __syncthreads();

        // Compute using half2 for partial vectorization
        half2 sum2 = __float2half2_rn(0.0f);

        #pragma unroll
        for (int k = 0; k < 16; k += 2) {
            half2 state_pair = __halves2half2(state_tile[local_row][k], state_tile[local_row][k+1]);
            half2 gate_pair = __halves2half2(gate_tile[k][local_col], gate_tile[k+1][local_col]);
            sum2 = __hfma2(state_pair, gate_pair, sum2);
        }

        // Reduce half2 to half
        float result = __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));

        __syncthreads();
        state_tile[local_row][local_col] = __float2half(result);
        __syncthreads();
    }

    tile_ptr[threadIdx.x] = state_tile[local_row][local_col];
}

// ============================================================================
// DIRECT MULTIPLY V3: Multiple tiles per block for better occupancy
// ============================================================================
extern "C" __global__
void direct_multiply_v3(
    half* __restrict__ states,
    const half* __restrict__ gates,
    int tiles_per_state,
    int num_gates
) {
    // Process 4 tiles per block (1024 threads = 4 × 256)
    int state_id = blockIdx.y;
    int base_tile = blockIdx.x * 4;
    int local_tile = threadIdx.x / 256;
    int thread_in_tile = threadIdx.x % 256;

    int tile_id = base_tile + local_tile;
    if (tile_id >= tiles_per_state) return;

    int state_offset = state_id * tiles_per_state * 256;
    half* tile_ptr = states + state_offset + tile_id * 256;

    int local_row = thread_in_tile / 16;
    int local_col = thread_in_tile % 16;

    // 4 tiles × 256 = 1024 half values = 2KB shared per component
    __shared__ half state_tiles[4][16][16];
    __shared__ half gate_tile[16][16];  // Gates are shared across all tiles in block

    // Load state tile
    state_tiles[local_tile][local_row][local_col] = tile_ptr[thread_in_tile];
    __syncthreads();

    for (int g = 0; g < num_gates; g++) {
        // Only first 256 threads load the gate (shared across tiles)
        if (threadIdx.x < 256) {
            int gr = threadIdx.x / 16;
            int gc = threadIdx.x % 16;
            gate_tile[gr][gc] = gates[g * 256 + threadIdx.x];
        }
        __syncthreads();

        // Compute
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            sum += __half2float(state_tiles[local_tile][local_row][k]) *
                   __half2float(gate_tile[k][local_col]);
        }

        __syncthreads();
        state_tiles[local_tile][local_row][local_col] = __float2half(sum);
        __syncthreads();
    }

    tile_ptr[thread_in_tile] = state_tiles[local_tile][local_row][local_col];
}

// ============================================================================
// DIRECT MULTIPLY V4: Warp-level with registers (minimal shared memory)
// Each warp processes one 16x16 tile
// 32 threads handle 16 rows × 2 cols each
// ============================================================================
extern "C" __global__
void direct_multiply_v4(
    half* __restrict__ states,
    const half* __restrict__ gates,
    int tiles_per_state,
    int num_gates
) {
    int state_id = blockIdx.y;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= tiles_per_state) return;

    int state_offset = state_id * tiles_per_state * 256;
    half* tile_ptr = states + state_offset + warp_id * 256;

    // Each thread in warp handles 8 elements (256/32 = 8)
    // Map: lane handles elements [lane*8 .. lane*8+7]
    // Or better: lane handles row (lane/2), cols based on (lane%2)

    int local_warp = threadIdx.x / 32;
    __shared__ half state_tile[8][16][16];  // 8 warps per block
    __shared__ half gate_tile[8][16][16];

    half* st = state_tile[local_warp][0];
    half* gt = gate_tile[local_warp][0];

    // Load state: each lane loads 8 consecutive elements
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        st[lane * 8 + i] = tile_ptr[lane * 8 + i];
    }
    __syncwarp();

    for (int g = 0; g < num_gates; g++) {
        // Load gate
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            gt[lane * 8 + i] = gates[g * 256 + lane * 8 + i];
        }
        __syncwarp();

        // Compute: each lane computes 8 output elements
        half results[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = lane * 8 + i;
            int row = idx / 16;
            int col = idx % 16;

            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                sum += __half2float(state_tile[local_warp][row][k]) *
                       __half2float(gate_tile[local_warp][k][col]);
            }
            results[i] = __float2half(sum);
        }
        __syncwarp();

        // Write back to shared
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            st[lane * 8 + i] = results[i];
        }
        __syncwarp();
    }

    // Store to DRAM
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        tile_ptr[lane * 8 + i] = st[lane * 8 + i];
    }
}
"#;

    let ptx = compile_ptx_with_opts(kernel_src, opts).expect("Failed to compile kernels");

    let module = ctx.load_module(ptx).expect("Failed to load PTX module");

    let wmma_fn = module.load_function("wmma_batched").expect("wmma_batched");
    let direct_v1_fn = module
        .load_function("direct_multiply_v1")
        .expect("direct_v1");
    let direct_v2_fn = module
        .load_function("direct_multiply_v2")
        .expect("direct_v2");
    let direct_v3_fn = module
        .load_function("direct_multiply_v3")
        .expect("direct_v3");
    let direct_v4_fn = module
        .load_function("direct_multiply_v4")
        .expect("direct_v4");

    println!("  All kernels compiled successfully\n");

    // Configuration
    let num_states = 1024;
    let tiles_per_state = 16;
    let total_tiles = num_states * tiles_per_state;
    let state_elements = total_tiles * 256;

    // Create identity gates for correctness verification
    let max_gates = 100;
    let mut gate_data: Vec<u16> = Vec::with_capacity(max_gates * 256);
    let norm_factor: f32 = 0.25;

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

    let tiles_i32 = tiles_per_state as i32;

    let warmup = 10;
    let iters = 50;

    // =========================================================================
    // CORRECTNESS CHECK FIRST
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" CORRECTNESS VERIFICATION (10 gates)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let test_gates = 10i32;

    // WMMA config
    let wmma_cfg = LaunchConfig {
        grid_dim: ((tiles_per_state as u32 + 7) / 8, num_states as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 8 * 256 * 2 * 2,
    };

    // Direct V1/V2 config: 1 block per tile, 256 threads per block
    let direct_cfg = LaunchConfig {
        grid_dim: (tiles_per_state as u32, num_states as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 16 * 16 * 2 * 2, // state + gate tiles
    };

    // Direct V3 config: 4 tiles per block
    let direct_v3_cfg = LaunchConfig {
        grid_dim: ((tiles_per_state as u32 + 3) / 4, num_states as u32, 1),
        block_dim: (1024, 1, 1),
        shared_mem_bytes: 4 * 16 * 16 * 2 + 16 * 16 * 2,
    };

    // Direct V4 config: same as WMMA (warp-based)
    let direct_v4_cfg = LaunchConfig {
        grid_dim: ((tiles_per_state as u32 + 7) / 8, num_states as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 8 * 16 * 16 * 2 * 2,
    };

    for (name, kernel_fn, cfg) in [
        ("WMMA Batched", &wmma_fn, wmma_cfg),
        ("Direct V1 (basic)", &direct_v1_fn, direct_cfg),
        ("Direct V2 (half2)", &direct_v2_fn, direct_cfg),
        ("Direct V3 (4 tiles/block)", &direct_v3_fn, direct_v3_cfg),
        ("Direct V4 (warp-level)", &direct_v4_fn, direct_v4_cfg),
    ] {
        let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");

        unsafe {
            stream
                .launch_builder(kernel_fn)
                .arg(&states)
                .arg(&gates_gpu)
                .arg(&tiles_i32)
                .arg(&test_gates)
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
        let status = if (norm - 1.0).abs() < 0.15 {
            "✓"
        } else {
            "✗"
        };
        println!("  {:25} norm = {:.6} {}", name, norm, status);
    }
    println!();

    // =========================================================================
    // PERFORMANCE BENCHMARK
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" PERFORMANCE COMPARISON: WMMA vs DIRECT");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let gate_counts = [10, 20, 50, 100];

    println!("┌────────┬────────────┬────────────┬────────────┬────────────┬────────────┐");
    println!("│ Gates  │ WMMA (ms)  │ Direct V1  │ Direct V2  │ Direct V3  │ Direct V4  │");
    println!("├────────┼────────────┼────────────┼────────────┼────────────┼────────────┤");

    for num_gates in gate_counts {
        let gates_i32 = num_gates as i32;

        // Warmup all kernels
        let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
        for _ in 0..warmup {
            unsafe {
                stream
                    .launch_builder(&wmma_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(wmma_cfg)
                    .ok();
                stream
                    .launch_builder(&direct_v1_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(direct_cfg)
                    .ok();
                stream
                    .launch_builder(&direct_v2_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(direct_cfg)
                    .ok();
                stream
                    .launch_builder(&direct_v3_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(direct_v3_cfg)
                    .ok();
                stream
                    .launch_builder(&direct_v4_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(direct_v4_cfg)
                    .ok();
            }
        }
        stream.synchronize().expect("sync");

        // Benchmark WMMA
        let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream
                    .launch_builder(&wmma_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(wmma_cfg)
                    .expect("wmma");
            }
        }
        stream.synchronize().expect("sync");
        let wmma_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Benchmark Direct V1
        let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream
                    .launch_builder(&direct_v1_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(direct_cfg)
                    .expect("direct_v1");
            }
        }
        stream.synchronize().expect("sync");
        let v1_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Benchmark Direct V2
        let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream
                    .launch_builder(&direct_v2_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(direct_cfg)
                    .expect("direct_v2");
            }
        }
        stream.synchronize().expect("sync");
        let v2_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Benchmark Direct V3
        let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream
                    .launch_builder(&direct_v3_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(direct_v3_cfg)
                    .expect("direct_v3");
            }
        }
        stream.synchronize().expect("sync");
        let v3_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Benchmark Direct V4
        let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream
                    .launch_builder(&direct_v4_fn)
                    .arg(&states)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(direct_v4_cfg)
                    .expect("direct_v4");
            }
        }
        stream.synchronize().expect("sync");
        let v4_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        println!(
            "│ {:>6} │ {:>8.3}ms │ {:>8.3}ms │ {:>8.3}ms │ {:>8.3}ms │ {:>8.3}ms │",
            num_gates, wmma_ms, v1_ms, v2_ms, v3_ms, v4_ms
        );
    }
    println!("└────────┴────────────┴────────────┴────────────┴────────────┴────────────┘\n");

    // =========================================================================
    // ANALYSIS
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Run one final benchmark to get numbers for analysis
    let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
    let gates_i32 = 50i32;

    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&wmma_fn)
                .arg(&states)
                .arg(&gates_gpu)
                .arg(&tiles_i32)
                .arg(&gates_i32)
                .launch(wmma_cfg)
                .expect("wmma");
        }
    }
    stream.synchronize().expect("sync");
    let wmma_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&direct_v4_fn)
                .arg(&states)
                .arg(&gates_gpu)
                .arg(&tiles_i32)
                .arg(&gates_i32)
                .launch(direct_v4_cfg)
                .expect("direct_v4");
        }
    }
    stream.synchronize().expect("sync");
    let best_direct_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let speedup = wmma_ms / best_direct_ms;

    println!("  For 50 gates on 1024 states × 16 tiles:");
    println!("    WMMA Batched:    {:.3}ms", wmma_ms);
    println!("    Best Direct:     {:.3}ms", best_direct_ms);
    println!("    Speedup:         {:.2}×", speedup);
    println!();

    if speedup > 1.0 {
        println!("  RESULT: Direct CUDA cores BEAT tensor cores for this use case!");
        println!("          The fragment conversion overhead makes WMMA slower.");
    } else {
        println!("  RESULT: WMMA still faster despite fragment overhead.");
        println!("          Tensor core raw throughput compensates for the overhead.");
    }
    println!();

    // Theoretical analysis
    let flops_per_gate = num_states as u64 * tiles_per_state as u64 * 16 * 16 * 16 * 2; // 16×16×16 FMA
    let total_flops = flops_per_gate * 50;
    let wmma_tflops = total_flops as f64 / (wmma_ms / 1000.0) / 1e12;
    let direct_tflops = total_flops as f64 / (best_direct_ms / 1000.0) / 1e12;

    println!("  Theoretical TFLOPS:");
    println!("    RTX 4070 Tensor:  116 TFLOPS");
    println!("    RTX 4070 FP16:    ~30 TFLOPS (CUDA cores)");
    println!();
    println!("  Achieved TFLOPS:");
    println!(
        "    WMMA:             {:.2} TFLOPS ({:.1}% of peak tensor)",
        wmma_tflops,
        wmma_tflops / 116.0 * 100.0
    );
    println!(
        "    Direct:           {:.2} TFLOPS ({:.1}% of peak CUDA)",
        direct_tflops,
        direct_tflops / 30.0 * 100.0
    );
    println!();
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires 'cuda' and 'perf-bench' features.");
}
