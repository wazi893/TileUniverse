// NCU Profiling Target
// A simple benchmark that runs a single kernel invocation for clean profiling.
//
// Run with NCU:
// ncu --metrics <metrics> ./target/release/examples/ncu_profile_target.exe

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{CudaRuntime, MultiStateOpt, MultiStatePersistent, WmmaGateType};
    use std::sync::Arc;

    eprintln!("NCU Profile Target: Initializing CUDA...");

    let rt = Arc::new(CudaRuntime::new().expect("Failed to create CUDA runtime"));

    let n_states = 1024;
    let tiles_per_state = 16;
    let depth = 1000u32; // Enough iterations for meaningful profiling

    eprintln!(
        "Creating persistent pool: {} states × {} tiles",
        n_states, tiles_per_state
    );
    let pool = MultiStatePersistent::new(rt.clone(), n_states, tiles_per_state)
        .expect("Failed to create pool");

    // Warmup
    eprintln!("Warming up...");
    let _ = pool.run_benchmark(WmmaGateType::Hadamard, 100, MultiStateOpt::ILP);

    // The actual kernel call that NCU will profile
    eprintln!("Running profiled kernel (depth={})...", depth);
    let (_, amp_ops, time_s) = pool
        .run_benchmark(WmmaGateType::Hadamard, depth, MultiStateOpt::ILP)
        .expect("Benchmark failed");

    let tcops = amp_ops as f64 / time_s / 1e12;
    eprintln!(
        "Result: {:.2} TCOPS ({} amp ops in {:.4}s)",
        tcops, amp_ops, time_s
    );
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    eprintln!("This example requires 'cuda' and 'perf-bench' features.");
}
