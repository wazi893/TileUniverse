// HONEST QUANTUM BENCHMARK
//
// NO TRICKS. NO FUSION MULTIPLIERS. NO "EFFECTIVE OPS".
// Just raw throughput measurement of the actual quantum substrate.
//
// Run with: cargo run --example honest_quantum_bench --features cuda,perf-bench --release

use engine::quantum::{QBackend, QGate, QRng, QState, apply_gate_backend, apply_gate_scalar};
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  HONEST QUANTUM SUBSTRATE BENCHMARK                            ║");
    println!("║  No tricks. No inflation. Just real numbers.                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Test configurations
    let qubit_configs = [4u8, 6, 8, 10, 12];
    let gate_count = 10_000u32; // Fixed number of actual gate applications
    let warmup_gates = 1_000u32;

    println!("Configuration:");
    println!(
        "  Gates per test:  {} (actual applications, no multipliers)",
        gate_count
    );
    println!("  Warmup gates:    {}", warmup_gates);
    println!("  Backends:        Scalar, AVX2");
    println!();

    // Results storage
    println!("┌────────┬────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│ Qubits │ Amplitudes │ Scalar (M/s) │  AVX2 (M/s)  │   Speedup    │");
    println!("├────────┼────────────┼──────────────┼──────────────┼──────────────┤");

    for &n_qubits in &qubit_configs {
        let result = bench_qubit_count(n_qubits, gate_count, warmup_gates);

        let speedup = if result.scalar_gops > 0.0 {
            result.avx2_gops / result.scalar_gops
        } else {
            0.0
        };

        println!(
            "│ {:>6} │ {:>10} │ {:>12.2} │ {:>12.2} │ {:>11.2}x │",
            n_qubits,
            1usize << n_qubits,
            result.scalar_gops / 1e6,
            result.avx2_gops / 1e6,
            speedup
        );
    }

    println!("└────────┴────────────┴──────────────┴──────────────┴──────────────┘");
    println!();

    // Detailed breakdown for 12 qubits
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" DETAILED BREAKDOWN: 12 QUBITS (4096 amplitudes)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let detailed = bench_detailed(12, gate_count);

    println!("  Gate-by-gate throughput (gates/sec, NOT amplitude ops):\n");
    println!("  ┌──────────────┬──────────────┬──────────────┐");
    println!("  │ Gate         │ Scalar (K/s) │  AVX2 (K/s)  │");
    println!("  ├──────────────┼──────────────┼──────────────┤");

    for (gate_name, scalar_rate, avx2_rate) in &detailed {
        println!(
            "  │ {:>12} │ {:>12.1} │ {:>12.1} │",
            gate_name,
            scalar_rate / 1e3,
            avx2_rate / 1e3
        );
    }
    println!("  └──────────────┴──────────────┴──────────────┘");
    println!();

    // Raw amplitude throughput
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" RAW AMPLITUDE THROUGHPUT");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("  What 'amplitude ops/sec' means:");
    println!("  - Each gate touches ALL 2^n amplitudes");
    println!("  - Amplitude ops = gates/sec × 2^n");
    println!("  - This is the HONEST metric (no fusion multiplier)\n");

    for &n_qubits in &[8u8, 10, 12] {
        let result = bench_qubit_count(n_qubits, gate_count, warmup_gates);
        let amps = 1u64 << n_qubits;

        let scalar_amp_ops = result.scalar_gops * (amps as f64);
        let avx2_amp_ops = result.avx2_gops * (amps as f64);

        println!("  {} qubits ({} amplitudes):", n_qubits, amps);
        println!(
            "    Scalar: {:.2} M amplitude-ops/sec",
            scalar_amp_ops / 1e6
        );
        println!("    AVX2:   {:.2} M amplitude-ops/sec", avx2_amp_ops / 1e6);
        println!("    AVX2:   {:.3} G amplitude-ops/sec", avx2_amp_ops / 1e9);
        println!();
    }

    // Mixed circuit benchmark (realistic workload)
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" REALISTIC CIRCUIT BENCHMARK");
    println!("═══════════════════════════════════════════════════════════════════\n");

    bench_realistic_circuit(10, 1000);
    bench_realistic_circuit(12, 1000);

    // Summary
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" HONEST SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let final_result = bench_qubit_count(12, 10_000, 1_000);
    let amps_12q = 4096u64;
    let avx2_amp_throughput = final_result.avx2_gops * (amps_12q as f64);

    println!("  CPU Quantum Substrate (12 qubits, AVX2):");
    println!(
        "  ├─ Gate throughput:      {:.2} K gates/sec",
        final_result.avx2_gops / 1e3
    );
    println!(
        "  ├─ Amplitude throughput: {:.2} M amp-ops/sec",
        avx2_amp_throughput / 1e6
    );
    println!(
        "  └─ In TCOPS:             {:.6} TCOPS",
        avx2_amp_throughput / 1e12
    );
    println!();
    println!("  This is the REAL number. No fusion. No multipliers.");
    println!("  If someone claims higher, ask: 'raw or effective?'");
    println!();

    // CUDA section (if available)
    #[cfg(feature = "cuda")]
    {
        println!("═══════════════════════════════════════════════════════════════════");
        println!(" CUDA GPU BENCHMARK");
        println!("═══════════════════════════════════════════════════════════════════\n");

        bench_cuda_honest();
    }
}

struct BenchResult {
    scalar_gops: f64, // gates per second (NOT amplitude ops)
    avx2_gops: f64,
}

fn bench_qubit_count(n_qubits: u8, gate_count: u32, warmup: u32) -> BenchResult {
    let mut rng = QRng::new(12345);

    // Warmup scalar
    {
        let mut state = QState::new_zero(n_qubits);
        for _ in 0..warmup {
            apply_gate_scalar(&mut state, &QGate::H(0), &mut rng);
        }
    }

    // Benchmark scalar
    let scalar_gops = {
        let mut state = QState::new_zero(n_qubits);
        let start = Instant::now();
        for i in 0..gate_count {
            let target = (i as u8) % n_qubits;
            apply_gate_scalar(&mut state, &QGate::H(target), &mut rng);
        }
        let elapsed = start.elapsed().as_secs_f64();
        (gate_count as f64) / elapsed
    };

    // Warmup AVX2
    {
        let mut state = QState::new_zero(n_qubits);
        for _ in 0..warmup {
            apply_gate_backend(&mut state, &QGate::H(0), &mut rng, QBackend::Avx2);
        }
    }

    // Benchmark AVX2
    let avx2_gops = {
        let mut state = QState::new_zero(n_qubits);
        let start = Instant::now();
        for i in 0..gate_count {
            let target = (i as u8) % n_qubits;
            apply_gate_backend(&mut state, &QGate::H(target), &mut rng, QBackend::Avx2);
        }
        let elapsed = start.elapsed().as_secs_f64();
        (gate_count as f64) / elapsed
    };

    BenchResult {
        scalar_gops,
        avx2_gops,
    }
}

fn bench_detailed(n_qubits: u8, gate_count: u32) -> Vec<(&'static str, f64, f64)> {
    let mut results = Vec::new();
    let mut rng = QRng::new(12345);

    let gates: Vec<(&str, QGate)> = vec![
        ("H", QGate::H(0)),
        ("X", QGate::X(0)),
        ("Z", QGate::Z(0)),
        ("Phase(π/4)", QGate::Phase(0, std::f32::consts::FRAC_PI_4)),
        ("CNOT", QGate::CNot(0, 1)),
        ("CZ", QGate::CZ(0, 1)),
    ];

    for (name, gate) in gates {
        // Scalar
        let scalar_rate = {
            let mut state = QState::new_zero(n_qubits);
            let start = Instant::now();
            for _ in 0..gate_count {
                apply_gate_scalar(&mut state, &gate, &mut rng);
            }
            let elapsed = start.elapsed().as_secs_f64();
            (gate_count as f64) / elapsed
        };

        // AVX2
        let avx2_rate = {
            let mut state = QState::new_zero(n_qubits);
            let start = Instant::now();
            for _ in 0..gate_count {
                apply_gate_backend(&mut state, &gate, &mut rng, QBackend::Avx2);
            }
            let elapsed = start.elapsed().as_secs_f64();
            (gate_count as f64) / elapsed
        };

        results.push((name, scalar_rate, avx2_rate));
    }

    results
}

fn bench_realistic_circuit(n_qubits: u8, circuit_depth: u32) {
    let mut rng = QRng::new(42);

    // Build a realistic circuit: alternating layers of single-qubit and two-qubit gates
    let mut circuit = Vec::new();
    for layer in 0..circuit_depth {
        if layer % 2 == 0 {
            // Single-qubit layer: H on all qubits
            for q in 0..n_qubits {
                circuit.push(QGate::H(q));
            }
        } else {
            // Two-qubit layer: CNOTs between adjacent pairs
            for q in (0..n_qubits - 1).step_by(2) {
                circuit.push(QGate::CNot(q, q + 1));
            }
        }
    }

    let total_gates = circuit.len();

    // Warmup
    {
        let mut state = QState::new_zero(n_qubits);
        for gate in circuit.iter().take(100) {
            apply_gate_backend(&mut state, gate, &mut rng, QBackend::Avx2);
        }
    }

    // Benchmark
    let mut state = QState::new_zero(n_qubits);
    let start = Instant::now();
    for gate in &circuit {
        apply_gate_backend(&mut state, gate, &mut rng, QBackend::Avx2);
    }
    let elapsed = start.elapsed().as_secs_f64();

    let gates_per_sec = (total_gates as f64) / elapsed;
    let amps = 1u64 << n_qubits;
    let amp_ops_per_sec = gates_per_sec * (amps as f64);

    println!(
        "  {} qubits, depth {} ({} total gates):",
        n_qubits, circuit_depth, total_gates
    );
    println!("    Time:            {:.3} ms", elapsed * 1000.0);
    println!("    Gates/sec:       {:.2} K", gates_per_sec / 1e3);
    println!("    Amplitude ops/s: {:.2} M", amp_ops_per_sec / 1e6);
    println!();
}

#[cfg(feature = "cuda")]
fn bench_cuda_honest() {
    use engine::cuda::{CudaRuntime, MultiStateOpt, MultiStatePersistent, WmmaGateType};
    use std::sync::Arc;

    println!("  Testing CUDA GPU throughput (raw, no fusion multiplier)...\n");

    let rt = match CudaRuntime::new() {
        Ok(r) => Arc::new(r),
        Err(e) => {
            println!("  CUDA init failed: {:?}", e);
            return;
        }
    };

    // Configuration
    let num_states = 1024usize;
    let tiles_per_state = 16usize; // 4096 amplitudes per state
    let depths = [100u32, 1000, 10000];

    println!("  Configuration:");
    println!("    Parallel states:   {}", num_states);
    println!(
        "    Tiles per state:   {} ({} amplitudes)",
        tiles_per_state,
        tiles_per_state * 256
    );
    println!(
        "    Total amplitudes:  {}",
        num_states * tiles_per_state * 256
    );
    println!();

    let pool = match MultiStatePersistent::new(rt.clone(), num_states, tiles_per_state) {
        Ok(p) => p,
        Err(e) => {
            println!("  Pool creation failed: {:?}", e);
            return;
        }
    };

    // Warmup
    let _ = pool.run_benchmark(WmmaGateType::Hadamard, 1000, MultiStateOpt::ILP);

    println!("  ┌───────────┬────────────┬──────────────┬──────────────┐");
    println!("  │ Depth     │ Time (ms)  │ Gates/sec    │ Raw TCOPS    │");
    println!("  ├───────────┼────────────┼──────────────┼──────────────┤");

    for &depth in &depths {
        let result = pool.run_benchmark(WmmaGateType::Hadamard, depth, MultiStateOpt::ILP);

        match result {
            Ok((gate_ops, _inflated_amp_ops, elapsed)) => {
                // Calculate RAW throughput (no fusion multiplier)
                let gates_per_sec = (gate_ops as f64) / elapsed;
                let amps_per_gate = (tiles_per_state * 256) as f64;
                let raw_amp_ops = gates_per_sec * amps_per_gate;
                let raw_tcops = raw_amp_ops / 1e12;

                println!(
                    "  │ {:>9} │ {:>10.2} │ {:>12.0} │ {:>12.4} │",
                    depth,
                    elapsed * 1000.0,
                    gates_per_sec,
                    raw_tcops
                );
            }
            Err(e) => {
                println!("  │ {:>9} │ ERROR: {:?}", depth, e);
            }
        }
    }

    println!("  └───────────┴────────────┴──────────────┴──────────────┘");
    println!();

    // Final honest summary
    if let Ok((gate_ops, _, elapsed)) =
        pool.run_benchmark(WmmaGateType::Hadamard, 10000, MultiStateOpt::ILP)
    {
        let gates_per_sec = (gate_ops as f64) / elapsed;
        let amps_per_gate = (tiles_per_state * 256) as f64;
        let raw_amp_ops = gates_per_sec * amps_per_gate;
        let raw_tcops = raw_amp_ops / 1e12;

        println!("  HONEST GPU SUMMARY:");
        println!(
            "  ├─ Gate throughput:      {:.2} M gates/sec",
            gates_per_sec / 1e6
        );
        println!(
            "  ├─ Amplitude throughput: {:.2} G amp-ops/sec",
            raw_amp_ops / 1e9
        );
        println!(
            "  └─ In TCOPS:             {:.4} TCOPS (RAW, no fusion)",
            raw_tcops
        );
        println!();
        println!("  Compare to EPIC 81 claim: 26,000 TCOPS");
        println!("  That was 'effective' (×100,000 fusion multiplier)");
        println!("  This is RAW: {:.4} TCOPS", raw_tcops);
        println!();

        // Theoretical peak analysis
        let rtx4070_tensor_tflops = 116.0f64;
        let utilization = (raw_amp_ops * 8.0) / (rtx4070_tensor_tflops * 1e12) * 100.0; // 8 FLOPs per complex multiply-add
        println!(
            "  Tensor Core utilization: {:.1}% of 116 TFLOPS theoretical",
            utilization
        );
    }
}

#[cfg(not(feature = "cuda"))]
fn bench_cuda_honest() {
    // No-op when CUDA not available
}
