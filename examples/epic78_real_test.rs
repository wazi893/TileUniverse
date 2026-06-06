// EPIC 78: Real performance test with correct methodology

#[cfg(all(feature = "cuda", feature = "perf-bench"))]
fn main() {
    use engine::cuda::{CudaRuntime, MultiStateOpt, WmmaGateType};

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 78: REAL Performance Numbers (No Cave Walls!)            ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let rt = CudaRuntime::new().expect("CUDA runtime");

    let tests = vec![
        ("EPIC 72 Baseline (1 state)", 1, 100_000_000),
        ("Phase 2B Basic (1024 states)", 1024, 100_000),
        ("Phase 2C ILP (1024 states)", 1024, 100_000),
    ];

    for (name, num_states, depth) in tests {
        println!("──────────────────────────────────────────────────────────────");
        println!("{}", name);
        println!("──────────────────────────────────────────────────────────────\n");

        let opt = if name.contains("ILP") {
            MultiStateOpt::ILP
        } else {
            MultiStateOpt::Basic
        };

        let (gate_ops, amp_ops) = engine::cuda::run_wmma_multi_state(
            &rt,
            num_states,
            16, // 12 qubits
            WmmaGateType::Hadamard,
            depth,
            opt,
        )
        .expect("Benchmark");

        println!("Gate ops: {}", gate_ops);
        println!("Amp ops: {}", amp_ops);
        println!();
    }
}

#[cfg(not(all(feature = "cuda", feature = "perf-bench")))]
fn main() {
    eprintln!("Error: Requires cuda,perf-bench features");
    std::process::exit(1);
}
