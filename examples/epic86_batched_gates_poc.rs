// EPIC 86: Batched Gate Application - Proof of Concept
//
// This benchmark proves the core hypothesis:
// Applying N different gates with state staying in shared memory
// beats applying gates one at a time with DRAM round-trips.
//
// Expected improvement: 5-10× for deep gate sequences
//
// Run with: cargo run --example epic86_batched_gates_poc --features cuda,perf-bench --release

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use std::sync::Arc;
    use std::time::Instant;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       EPIC 86: BATCHED GATE APPLICATION - PROOF OF CONCEPT       ║");
    println!("║       Testing register-resident state for N different gates      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Initialize CUDA
    let ctx = Arc::new(CudaContext::new(0).expect("Failed to create CUDA context"));
    let stream = ctx.new_stream().expect("Failed to create CUDA stream");

    println!("Compiling batched gates kernel...");

    // Compile the batched gates kernel
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

    // Batched gates kernel - applies N different gates with state in shared memory
    let batched_kernel_src = r#"
#include <mma.h>
using namespace nvcuda;

extern "C" __global__
void wmma_batched_gates(
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

    // Load state from DRAM ONCE
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

    // Apply ALL gates with state in shared memory
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

    // Write final state to DRAM ONCE
    half* result = (num_gates % 2 == 1) ? buf_b : buf_a;
    for (int i = lane; i < 256; i += 32) {
        tile[i] = result[i];
    }
}

// Baseline: apply a SINGLE gate (to be called N times for N different gates)
// This represents the current implementation pattern: one kernel launch per gate
extern "C" __global__
void wmma_single_gate(
    half* __restrict__ states,
    const half* __restrict__ gate,
    int tiles_per_state
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

    // Load state from DRAM
    for (int i = lane; i < 256; i += 32) {
        buf_a[i] = tile[i];
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> state_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> gate_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> result_frag;

    // Load state and gate
    wmma::load_matrix_sync(state_frag, buf_a, 16);
    wmma::load_matrix_sync(gate_frag, gate, 16);

    // Zero accumulator
    #pragma unroll
    for (int i = 0; i < result_frag.num_elements; i++) {
        result_frag.x[i] = __float2half(0.0f);
    }

    // Single MMA
    wmma::mma_sync(result_frag, state_frag, gate_frag, result_frag);

    // Store result to shared then DRAM
    wmma::store_matrix_sync(buf_b, result_frag, 16, wmma::mem_row_major);
    __syncwarp();

    for (int i = lane; i < 256; i += 32) {
        tile[i] = buf_b[i];
    }
}
"#;

    let ptx = compile_ptx_with_opts(batched_kernel_src, opts)
        .expect("Failed to compile batched gates kernel");

    // Check for mma.sync instructions in PTX
    let ptx_str = ptx.to_src();
    let mma_count = ptx_str.matches("mma.sync").count();
    println!("  PTX contains {} mma.sync instructions", mma_count);

    // Load module and get functions
    let module = ctx.load_module(ptx).expect("Failed to load PTX module");

    let batched_fn = module
        .load_function("wmma_batched_gates")
        .expect("Failed to get batched function");
    let single_gate_fn = module
        .load_function("wmma_single_gate")
        .expect("Failed to get single gate function");

    println!("  Kernels compiled and loaded successfully\n");

    // ==========================================================================
    // Test Configuration
    // ==========================================================================
    let num_states = 1024;
    let tiles_per_state = 16; // 16 tiles × 256 = 4096 amplitudes = 2^12 qubits
    let total_tiles = num_states * tiles_per_state;
    let state_elements = total_tiles * 256;

    println!("Configuration:");
    println!("  States:          {}", num_states);
    println!("  Tiles/state:     {}", tiles_per_state);
    println!(
        "  Total elements:  {} ({:.2} MB)",
        state_elements,
        state_elements as f64 * 2.0 / 1e6
    );
    println!();

    // ==========================================================================
    // Create gate matrices (16x16 Hadamard-like matrices in FP16, stored as u16)
    // ==========================================================================
    // For a unitary 16x16 Hadamard: H[i][j] = (1/4) * (-1)^(popcount(i & j))
    // 1/sqrt(16) = 1/4 = 0.25
    let norm_factor: f32 = 0.25;

    // Create a sequence of gate matrices
    let max_gates = 100;
    let mut gate_data: Vec<u16> = Vec::with_capacity(max_gates * 256);

    for g in 0..max_gates {
        // Alternate between Hadamard (unitary) and Identity matrices
        for row in 0..16 {
            for col in 0..16 {
                let val = if g % 2 == 0 {
                    // Hadamard 16x16 (unitary): H[i][j] = (1/4) * (-1)^(popcount(i & j))
                    let sign = if ((row as u32) & (col as u32)).count_ones() % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    sign * norm_factor
                } else {
                    // Identity (also unitary)
                    if row == col { 1.0 } else { 0.0 }
                };
                gate_data.push(half::f16::from_f32(val).to_bits());
            }
        }
    }

    // Upload gates to GPU (as u16, same binary representation as half)
    let gates_gpu: CudaSlice<u16> = stream
        .memcpy_stod(&gate_data)
        .expect("Failed to upload gates");

    // Single gate for baseline
    let single_gate: Vec<u16> = gate_data[0..256].to_vec();
    let single_gate_gpu: CudaSlice<u16> = stream
        .memcpy_stod(&single_gate)
        .expect("Failed to upload single gate");

    // Create initial state (all zeros except first element = 1)
    let mut initial_state: Vec<u16> = vec![half::f16::from_f32(0.0).to_bits(); state_elements];
    for tile in 0..total_tiles {
        initial_state[tile * 256] = half::f16::from_f32(1.0).to_bits(); // |0⟩ state
    }

    // ==========================================================================
    // Benchmark different gate counts
    // ==========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" BENCHMARK: N Kernel Launches vs 1 Batched Launch");
    println!(" (Fair comparison: both apply N DIFFERENT gates)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let gate_counts = [10, 20, 50, 100];
    let warmup_iters = 5;
    let bench_iters = 20;

    println!("┌──────────┬────────────────┬────────────────┬──────────────┬──────────────┐");
    println!("│  Gates   │ N Launches(ms) │ Batched (ms)   │   Speedup    │ Eff. TCOPS   │");
    println!("├──────────┼────────────────┼────────────────┼──────────────┼──────────────┤");

    // Upload individual gate slices for baseline
    let mut individual_gates: Vec<CudaSlice<u16>> = Vec::new();
    for g in 0..100 {
        let gate_slice: Vec<u16> = gate_data[g * 256..(g + 1) * 256].to_vec();
        individual_gates.push(
            stream
                .memcpy_stod(&gate_slice)
                .expect("Failed to upload gate"),
        );
    }

    for num_gates in gate_counts {
        // Launch configuration
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

        // =====================================================================
        // Warmup
        // =====================================================================
        let states_warmup: CudaSlice<u16> = stream
            .memcpy_stod(&initial_state)
            .expect("Failed to upload warmup state");
        for _ in 0..warmup_iters {
            // Warmup baseline: N separate kernel launches
            for g in 0..num_gates {
                unsafe {
                    stream
                        .launch_builder(&single_gate_fn)
                        .arg(&states_warmup)
                        .arg(&individual_gates[g])
                        .arg(&tiles_i32)
                        .launch(cfg)
                        .expect("Warmup single gate failed");
                }
            }
            // Warmup batched
            unsafe {
                stream
                    .launch_builder(&batched_fn)
                    .arg(&states_warmup)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(cfg)
                    .expect("Batched warmup failed");
            }
        }
        stream.synchronize().expect("Sync failed");

        // =====================================================================
        // Benchmark Baseline: N kernel launches, each applies one gate
        // This is the FAIR comparison - both methods apply N different gates
        // =====================================================================
        let states_baseline: CudaSlice<u16> = stream
            .memcpy_stod(&initial_state)
            .expect("Failed to upload baseline state");

        let baseline_start = Instant::now();
        for _ in 0..bench_iters {
            // Apply N different gates via N separate kernel launches
            for g in 0..num_gates {
                unsafe {
                    stream
                        .launch_builder(&single_gate_fn)
                        .arg(&states_baseline)
                        .arg(&individual_gates[g])
                        .arg(&tiles_i32)
                        .launch(cfg)
                        .expect("Baseline single gate failed");
                }
            }
        }
        stream.synchronize().expect("Sync failed");
        let baseline_time = baseline_start.elapsed();
        let baseline_ms = baseline_time.as_secs_f64() * 1000.0 / bench_iters as f64;

        // =====================================================================
        // Benchmark Batched: 1 kernel launch applies all N gates
        // State stays in shared memory between gates
        // =====================================================================
        let states_batched: CudaSlice<u16> = stream
            .memcpy_stod(&initial_state)
            .expect("Failed to upload batched state");

        let batched_start = Instant::now();
        for _ in 0..bench_iters {
            unsafe {
                stream
                    .launch_builder(&batched_fn)
                    .arg(&states_batched)
                    .arg(&gates_gpu)
                    .arg(&tiles_i32)
                    .arg(&gates_i32)
                    .launch(cfg)
                    .expect("Batched launch failed");
            }
        }
        stream.synchronize().expect("Sync failed");
        let batched_time = batched_start.elapsed();
        let batched_ms = batched_time.as_secs_f64() * 1000.0 / bench_iters as f64;

        // Calculate metrics
        let speedup = baseline_ms / batched_ms;

        // Effective TCOPS: total amplitude operations / time
        let amp_ops_per_iter = num_states as u64 * tiles_per_state as u64 * 256 * num_gates as u64;
        let eff_tcops = amp_ops_per_iter as f64 / (batched_ms / 1000.0) / 1e12;

        println!(
            "│ {:>8} │ {:>12.3}ms │ {:>12.3}ms │ {:>10.2}x │ {:>10.2} │",
            num_gates, baseline_ms, batched_ms, speedup, eff_tcops
        );
    }
    println!("└──────────┴────────────────┴────────────────┴──────────────┴──────────────┘\n");

    // ==========================================================================
    // Correctness Verification
    // ==========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" CORRECTNESS VERIFICATION");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let test_gates = 10;
    let mut states_test: CudaSlice<u16> = stream
        .memcpy_stod(&initial_state)
        .expect("Failed to upload test state");

    let cfg = LaunchConfig {
        grid_dim: ((tiles_per_state as u32 + 7) / 8, num_states as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 8 * 256 * 2 * 2,
    };

    let tiles_i32 = tiles_per_state as i32;
    let gates_i32 = test_gates as i32;

    unsafe {
        stream
            .launch_builder(&batched_fn)
            .arg(&states_test)
            .arg(&gates_gpu)
            .arg(&tiles_i32)
            .arg(&gates_i32)
            .launch(cfg)
            .expect("Test launch failed");
    }
    stream.synchronize().expect("Sync failed");

    let mut result_u16: Vec<u16> = vec![0u16; state_elements];
    stream
        .memcpy_dtoh(&states_test, &mut result_u16)
        .expect("Failed to download result");

    // Convert back to f16 for norm calculation
    let result: Vec<half::f16> = result_u16
        .iter()
        .map(|&u| half::f16::from_bits(u))
        .collect();

    // Check first tile's norm (should be ~1.0 for unitary gates)
    let mut norm_sq: f64 = 0.0;
    for i in 0..256 {
        let val = result[i].to_f32() as f64;
        norm_sq += val * val;
    }
    let norm = norm_sq.sqrt();

    println!("  Test configuration: {} gates applied", test_gates);
    println!("  First tile norm after {} gates: {:.6}", test_gates, norm);

    if (norm - 1.0).abs() < 0.1 {
        println!("  ✓ Norm preserved (within 10% tolerance)");
    } else {
        println!("  ⚠ Norm drift detected: {:.6} (expected ~1.0)", norm);
    }
    println!();

    // ==========================================================================
    // Memory Traffic Analysis
    // ==========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" MEMORY TRAFFIC ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let state_bytes = state_elements * 2; // FP16 = 2 bytes

    println!("  For {} gates applied to {} states:", 50, num_states);
    println!();
    println!("  BASELINE (gate by gate, if DRAM round-trip each):");
    println!(
        "    DRAM reads:  50 × {} bytes = {:.2} GB",
        state_bytes,
        50.0 * state_bytes as f64 / 1e9
    );
    println!(
        "    DRAM writes: 50 × {} bytes = {:.2} GB",
        state_bytes,
        50.0 * state_bytes as f64 / 1e9
    );
    println!(
        "    Total DRAM traffic: {:.2} GB",
        100.0 * state_bytes as f64 / 1e9
    );
    println!();
    println!("  BATCHED (state in shared memory):");
    println!(
        "    DRAM reads:  1 × {} bytes = {:.2} MB",
        state_bytes,
        state_bytes as f64 / 1e6
    );
    println!(
        "    DRAM writes: 1 × {} bytes = {:.2} MB",
        state_bytes,
        state_bytes as f64 / 1e6
    );
    println!(
        "    Gate loads:  50 × {} bytes = {:.2} KB (L2 cached)",
        256 * 2,
        50.0 * 256.0 * 2.0 / 1e3
    );
    println!(
        "    Total DRAM traffic: {:.2} MB",
        2.0 * state_bytes as f64 / 1e6
    );
    println!();
    println!(
        "  TRAFFIC REDUCTION: {:.0}×",
        100.0 * state_bytes as f64 / (2.0 * state_bytes as f64)
    );
    println!();

    // ==========================================================================
    // Summary
    // ==========================================================================
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    EPIC 86 POC SUMMARY                           ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║                                                                  ║");
    println!("║  KEY FINDING:                                                    ║");
    println!("║    Batched gate application keeps state in shared memory,        ║");
    println!("║    reducing DRAM traffic by 50× for 50 gates.                    ║");
    println!("║                                                                  ║");
    println!("║  NEXT STEPS:                                                     ║");
    println!("║    1. Integrate with algebraic fusion layer                      ║");
    println!("║    2. Upload gate sequences instead of mega-matrices             ║");
    println!("║    3. Full benchmarking with real quantum circuits               ║");
    println!("║                                                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    println!("This example requires 'cuda' and 'perf-bench' features.");
    println!(
        "Run with: cargo run --example epic86_batched_gates_poc --features cuda,perf-bench --release"
    );
}
