// EPIC 86: Diagnostic - Where is the time going?
//
// This benchmark isolates different phases to identify the bottleneck:
// 1. DRAM load time
// 2. Gate loop time (shared memory only)
// 3. DRAM store time
// 4. Sync overhead
//
// Run with: cargo run --example epic86_diagnostic --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use std::sync::Arc;
    use std::time::Instant;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       EPIC 86: DIAGNOSTIC - Where is the time going?            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let ctx = Arc::new(CudaContext::new(0).expect("Failed to create CUDA context"));
    let stream = ctx.new_stream().expect("Failed to create CUDA stream");

    println!("Compiling diagnostic kernels...");

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

    let diagnostic_kernels = r#"
#include <mma.h>
using namespace nvcuda;

// ============================================================================
// KERNEL 1: Full batched (baseline for comparison)
// ============================================================================
extern "C" __global__
void batched_full(
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

    // PHASE 1: DRAM Load
    for (int i = lane; i < 256; i += 32) {
        buf_a[i] = tile[i];
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> state_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> gate_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> result_frag;

    half* read_buf = buf_a;
    half* write_buf = buf_b;

    // PHASE 2: Gate Loop
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

    // PHASE 3: DRAM Store
    half* result = (num_gates % 2 == 1) ? buf_b : buf_a;
    for (int i = lane; i < 256; i += 32) {
        tile[i] = result[i];
    }
}

// ============================================================================
// KERNEL 2: DRAM Load Only - measures memory read bandwidth
// ============================================================================
extern "C" __global__
void dram_load_only(
    half* __restrict__ states,
    int tiles_per_state
) {
    int state_id = blockIdx.y;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= tiles_per_state) return;

    int state_offset = state_id * tiles_per_state * 256;
    half* tile = states + state_offset + warp_id * 256;

    __shared__ half shared_a[8][256];
    int local_warp = threadIdx.x / 32;
    half* buf_a = shared_a[local_warp];

    int lane = threadIdx.x % 32;

    // Only DRAM load
    for (int i = lane; i < 256; i += 32) {
        buf_a[i] = tile[i];
    }
    __syncwarp();

    // Prevent optimization - touch the data
    if (lane == 0 && buf_a[0] == __float2half(-999.0f)) {
        tile[0] = buf_a[0];
    }
}

// ============================================================================
// KERNEL 3: DRAM Store Only - measures memory write bandwidth
// ============================================================================
extern "C" __global__
void dram_store_only(
    half* __restrict__ states,
    int tiles_per_state
) {
    int state_id = blockIdx.y;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= tiles_per_state) return;

    int state_offset = state_id * tiles_per_state * 256;
    half* tile = states + state_offset + warp_id * 256;

    __shared__ half shared_a[8][256];
    int local_warp = threadIdx.x / 32;
    half* buf_a = shared_a[local_warp];

    int lane = threadIdx.x % 32;

    // Initialize shared (simulate having data to write)
    buf_a[lane] = __float2half((float)lane);
    buf_a[lane + 32] = __float2half((float)(lane + 32));
    buf_a[lane + 64] = __float2half((float)(lane + 64));
    buf_a[lane + 96] = __float2half((float)(lane + 96));
    buf_a[lane + 128] = __float2half((float)(lane + 128));
    buf_a[lane + 160] = __float2half((float)(lane + 160));
    buf_a[lane + 192] = __float2half((float)(lane + 192));
    buf_a[lane + 224] = __float2half((float)(lane + 224));
    __syncwarp();

    // Only DRAM store
    for (int i = lane; i < 256; i += 32) {
        tile[i] = buf_a[i];
    }
}

// ============================================================================
// KERNEL 4: Gate Loop Only - No DRAM, measures pure compute + shared mem
// State pre-loaded, result stays in shared
// ============================================================================
extern "C" __global__
void gate_loop_only(
    half* __restrict__ states,  // Only for initial load
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

    // Pre-load state (not timed separately here, but needed for correctness)
    for (int i = lane; i < 256; i += 32) {
        buf_a[i] = tile[i];
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> state_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> gate_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> result_frag;

    half* read_buf = buf_a;
    half* write_buf = buf_b;

    // ONLY the gate loop - this is what we want to measure
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

    // NO DRAM store - result stays in shared
    // Touch result to prevent optimization
    if (lane == 0) {
        half* result = (num_gates % 2 == 1) ? buf_b : buf_a;
        if (result[0] == __float2half(-999.0f)) {
            tile[0] = result[0];
        }
    }
}

// ============================================================================
// KERNEL 5: MMA Only - No shared memory round-trip, measures pure tensor core
// ============================================================================
extern "C" __global__
void mma_only(
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

    __shared__ half shared_buf[8][256];
    int local_warp = threadIdx.x / 32;
    half* buf = shared_buf[local_warp];

    int lane = threadIdx.x % 32;

    // Load state once
    for (int i = lane; i < 256; i += 32) {
        buf[i] = tile[i];
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> state_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> gate_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> result_frag;

    // Load state fragment ONCE
    wmma::load_matrix_sync(state_frag, buf, 16);

    // Run N MMAs with same state (measures pure MMA throughput)
    // This is NOT correct computation, just measuring MMA speed
    for (int g = 0; g < num_gates; g++) {
        wmma::load_matrix_sync(gate_frag, gates + g * 256, 16);

        #pragma unroll
        for (int i = 0; i < result_frag.num_elements; i++) {
            result_frag.x[i] = __float2half(0.0f);
        }

        wmma::mma_sync(result_frag, state_frag, gate_frag, result_frag);
        // NO store to shared, NO reload - just accumulate
    }

    // Store final result
    wmma::store_matrix_sync(buf, result_frag, 16, wmma::mem_row_major);
    __syncwarp();

    for (int i = lane; i < 256; i += 32) {
        tile[i] = buf[i];
    }
}

// ============================================================================
// KERNEL 6: Sync Only - measures barrier overhead
// ============================================================================
extern "C" __global__
void sync_only(
    int num_syncs
) {
    for (int i = 0; i < num_syncs; i++) {
        __syncwarp();
    }
}
"#;

    let ptx = compile_ptx_with_opts(diagnostic_kernels, opts)
        .expect("Failed to compile diagnostic kernels");

    let module = ctx.load_module(ptx).expect("Failed to load PTX module");

    let batched_full_fn = module.load_function("batched_full").expect("batched_full");
    let dram_load_fn = module
        .load_function("dram_load_only")
        .expect("dram_load_only");
    let dram_store_fn = module
        .load_function("dram_store_only")
        .expect("dram_store_only");
    let gate_loop_fn = module
        .load_function("gate_loop_only")
        .expect("gate_loop_only");
    let mma_only_fn = module.load_function("mma_only").expect("mma_only");
    let sync_only_fn = module.load_function("sync_only").expect("sync_only");

    println!("  All diagnostic kernels compiled successfully\n");

    // Configuration
    let num_states = 1024;
    let tiles_per_state = 16;
    let total_tiles = num_states * tiles_per_state;
    let state_elements = total_tiles * 256;
    let num_gates = 50;

    println!("Configuration:");
    println!("  States:          {}", num_states);
    println!("  Tiles/state:     {}", tiles_per_state);
    println!("  Gates:           {}", num_gates);
    println!(
        "  Total elements:  {} ({:.2} MB)\n",
        state_elements,
        state_elements as f64 * 2.0 / 1e6
    );

    // Create gate matrices (identity for simplicity)
    let mut gate_data: Vec<u16> = Vec::with_capacity(num_gates * 256);
    for _g in 0..num_gates {
        for row in 0..16 {
            for col in 0..16 {
                let val = if row == col { 1.0f32 } else { 0.0f32 };
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
        shared_mem_bytes: 8 * 256 * 2 * 2,
    };

    let tiles_i32 = tiles_per_state as i32;
    let gates_i32 = num_gates as i32;

    let warmup = 10;
    let iters = 50;

    println!("═══════════════════════════════════════════════════════════════════");
    println!(" PHASE TIMING BREAKDOWN ({}× {} gates)", iters, num_gates);
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Warmup all kernels
    let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
    for _ in 0..warmup {
        unsafe {
            stream
                .launch_builder(&batched_full_fn)
                .arg(&states)
                .arg(&gates_gpu)
                .arg(&tiles_i32)
                .arg(&gates_i32)
                .launch(cfg)
                .ok();
            stream
                .launch_builder(&dram_load_fn)
                .arg(&states)
                .arg(&tiles_i32)
                .launch(cfg)
                .ok();
            stream
                .launch_builder(&dram_store_fn)
                .arg(&states)
                .arg(&tiles_i32)
                .launch(cfg)
                .ok();
            stream
                .launch_builder(&gate_loop_fn)
                .arg(&states)
                .arg(&gates_gpu)
                .arg(&tiles_i32)
                .arg(&gates_i32)
                .launch(cfg)
                .ok();
            stream
                .launch_builder(&mma_only_fn)
                .arg(&states)
                .arg(&gates_gpu)
                .arg(&tiles_i32)
                .arg(&gates_i32)
                .launch(cfg)
                .ok();
        }
    }
    stream.synchronize().expect("sync");

    // Benchmark each phase
    let mut results: Vec<(&str, f64)> = Vec::new();

    // 1. Full batched kernel
    let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&batched_full_fn)
                .arg(&states)
                .arg(&gates_gpu)
                .arg(&tiles_i32)
                .arg(&gates_i32)
                .launch(cfg)
                .expect("batched_full");
        }
    }
    stream.synchronize().expect("sync");
    let full_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    results.push(("Full Batched (DRAM+Loop+DRAM)", full_ms));

    // 2. DRAM Load only
    let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&dram_load_fn)
                .arg(&states)
                .arg(&tiles_i32)
                .launch(cfg)
                .expect("dram_load");
        }
    }
    stream.synchronize().expect("sync");
    let load_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    results.push(("DRAM Load Only", load_ms));

    // 3. DRAM Store only
    let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&dram_store_fn)
                .arg(&states)
                .arg(&tiles_i32)
                .launch(cfg)
                .expect("dram_store");
        }
    }
    stream.synchronize().expect("sync");
    let store_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    results.push(("DRAM Store Only", store_ms));

    // 4. Gate loop only (includes pre-load but no final store)
    let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&gate_loop_fn)
                .arg(&states)
                .arg(&gates_gpu)
                .arg(&tiles_i32)
                .arg(&gates_i32)
                .launch(cfg)
                .expect("gate_loop");
        }
    }
    stream.synchronize().expect("sync");
    let loop_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    results.push(("Gate Loop (Load+Loop, no store)", loop_ms));

    // 5. MMA only (no shared memory round-trip per gate)
    let states: CudaSlice<u16> = stream.memcpy_stod(&initial_state).expect("states");
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&mma_only_fn)
                .arg(&states)
                .arg(&gates_gpu)
                .arg(&tiles_i32)
                .arg(&gates_i32)
                .launch(cfg)
                .expect("mma_only");
        }
    }
    stream.synchronize().expect("sync");
    let mma_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    results.push(("MMA Only (no shared round-trip)", mma_ms));

    // 6. Sync only
    let sync_cfg = LaunchConfig {
        grid_dim: (blocks_x, blocks_y, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_builder(&sync_only_fn)
                .arg(&gates_i32) // num_syncs = num_gates
                .launch(sync_cfg)
                .expect("sync_only");
        }
    }
    stream.synchronize().expect("sync");
    let sync_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    results.push(("Sync Only (50 __syncwarp)", sync_ms));

    // Print results
    println!("┌────────────────────────────────────────┬────────────────┬────────────────┐");
    println!("│              Phase                     │   Time (ms)    │   % of Full    │");
    println!("├────────────────────────────────────────┼────────────────┼────────────────┤");

    for (name, time) in &results {
        let pct = time / full_ms * 100.0;
        println!("│ {:38} │ {:>12.4}ms │ {:>12.1}% │", name, time, pct);
    }
    println!("└────────────────────────────────────────┴────────────────┴────────────────┘\n");

    // Analysis
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let dram_overhead = load_ms + store_ms;
    let loop_overhead = loop_ms - load_ms; // Loop without the pre-load
    let shared_roundtrip_overhead = loop_overhead - mma_ms + load_ms; // Difference between loop and MMA-only

    println!(
        "  DRAM overhead (load + store):     {:.4}ms ({:.1}%)",
        dram_overhead,
        dram_overhead / full_ms * 100.0
    );
    println!(
        "  Gate loop overhead:               {:.4}ms ({:.1}%)",
        loop_overhead,
        loop_overhead / full_ms * 100.0
    );
    println!(
        "  Pure MMA time:                    {:.4}ms ({:.1}%)",
        mma_ms - load_ms - store_ms,
        (mma_ms - load_ms - store_ms).max(0.0) / full_ms * 100.0
    );
    println!(
        "  Shared mem round-trip overhead:   {:.4}ms ({:.1}%)",
        shared_roundtrip_overhead.max(0.0),
        shared_roundtrip_overhead.max(0.0) / full_ms * 100.0
    );
    println!(
        "  Sync barrier overhead:            {:.4}ms ({:.1}%)",
        sync_ms,
        sync_ms / full_ms * 100.0
    );
    println!();

    // Theoretical analysis
    let state_bytes = state_elements * 2;
    let theoretical_dram_time = (state_bytes * 2) as f64 / 256e9 * 1000.0; // 256 GB/s
    let actual_dram_time = load_ms + store_ms;

    println!(
        "  Theoretical DRAM time (256 GB/s): {:.4}ms",
        theoretical_dram_time
    );
    println!(
        "  Actual DRAM time:                 {:.4}ms",
        actual_dram_time
    );
    println!(
        "  DRAM efficiency:                  {:.1}%",
        theoretical_dram_time / actual_dram_time * 100.0
    );
    println!();

    // MMA analysis
    let mma_ops = num_states as u64 * tiles_per_state as u64 * num_gates as u64;
    let mma_time_only = mma_ms - load_ms - store_ms;
    if mma_time_only > 0.0 {
        let mma_throughput = mma_ops as f64 / (mma_time_only / 1000.0) / 1e9;
        println!("  MMA operations:                   {}", mma_ops);
        println!(
            "  Pure MMA throughput:              {:.2} GMMA/s",
            mma_throughput
        );
    }
    println!();

    // Bottleneck identification
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" BOTTLENECK IDENTIFICATION");
    println!("═══════════════════════════════════════════════════════════════════\n");

    if loop_overhead > dram_overhead * 2.0 {
        println!("  → PRIMARY BOTTLENECK: Gate loop (shared memory round-trip)");
        println!("    The fragment store → load cycle per gate is the limiting factor.");
        println!("    Consider: double-buffered fragment conversion");
    } else if dram_overhead > loop_overhead {
        println!("  → PRIMARY BOTTLENECK: DRAM bandwidth");
        println!("    Memory load/store dominates. Already near optimal for this approach.");
    } else {
        println!("  → BALANCED: Both DRAM and loop contribute significantly");
        println!("    Optimization should target both.");
    }
    println!();

    // Expected vs actual speedup
    let expected_speedup = (num_gates as f64 * (load_ms + store_ms)) / full_ms;
    println!("  If DRAM was the only overhead:");
    println!(
        "    {} separate kernels would take: {:.3}ms",
        num_gates,
        num_gates as f64 * (load_ms + store_ms + 0.005)
    ); // 5μs launch overhead
    println!("    Batched kernel takes:           {:.3}ms", full_ms);
    println!(
        "    Expected speedup:               {:.1}×",
        expected_speedup
    );
    println!();
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires 'cuda' and 'perf-bench' features.");
}
