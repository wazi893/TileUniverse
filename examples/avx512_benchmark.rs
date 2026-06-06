// AVX512 vs AVX2 Benchmark
// Tests native 512-bit vector operations on AMD 9950X3D (Zen 4)

use logic_fabric_core::quantum::{
    QBackend, QGate, QRng, QState, apply_gate_backend, is_avx512f_available,
};
use std::time::Instant;

fn benchmark_backend(name: &str, backend: QBackend, qubits: u8, depth: usize) -> f64 {
    let mut state = QState::new_zero(qubits);
    let mut rng = QRng::new(42);

    // Warmup
    for i in 0..qubits {
        let _ = apply_gate_backend(&mut state, &QGate::H(i), &mut rng, backend);
    }

    // Reset state
    state = QState::new_zero(qubits);

    let start = Instant::now();

    // Apply alternating H and Phase gates
    for _ in 0..depth {
        for q in 0..qubits {
            let _ = apply_gate_backend(&mut state, &QGate::H(q), &mut rng, backend);
        }
        for q in 0..qubits {
            let _ = apply_gate_backend(&mut state, &QGate::Phase(q, 0.1), &mut rng, backend);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total_gates = (depth * qubits as usize * 2) as f64;
    let gates_per_sec = total_gates / elapsed;
    let amp_ops = total_gates * (1u64 << qubits) as f64;
    let gcops = amp_ops / elapsed / 1e9;

    println!(
        "  {}: {:.3}s, {:.2}M gates/s, {:.2} GCOPS",
        name,
        elapsed,
        gates_per_sec / 1e6,
        gcops
    );

    gcops
}

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  AVX512 vs AVX2 CPU QUANTUM BENCHMARK                          ║");
    println!("║  AMD 9950X3D (Zen 4) - Native 512-bit Vectors                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    if !is_avx512f_available() {
        println!("  ⚠ AVX512F not detected on this system!");
        return;
    }
    println!("  ✓ AVX512F detected\n");

    // Test configurations
    let configs = [
        (12, 10_000, "12 qubits (4K amps)"),
        (14, 5_000, "14 qubits (16K amps)"),
        (16, 2_000, "16 qubits (64K amps)"),
        (18, 1_000, "18 qubits (256K amps)"),
        (20, 500, "20 qubits (1M amps)"),
    ];

    println!("┌────────────────────────┬───────────────┬───────────────┬─────────┐");
    println!("│      Configuration     │  AVX2 GCOPS   │  AVX512 GCOPS │ Speedup │");
    println!("├────────────────────────┼───────────────┼───────────────┼─────────┤");

    for (qubits, depth, desc) in configs {
        println!("\n{}", desc);
        let gcops_avx2 = benchmark_backend("AVX2  ", QBackend::Avx2, qubits, depth);
        let gcops_avx512 = benchmark_backend("AVX512", QBackend::Avx512, qubits, depth);
        let speedup = gcops_avx512 / gcops_avx2;
        println!(
            "│ {:22} │ {:13.2} │ {:13.2} │ {:7.2}x│",
            desc, gcops_avx2, gcops_avx512, speedup
        );
    }

    println!("└────────────────────────┴───────────────┴───────────────┴─────────┘");

    // Extended test at 12 qubits for sustained performance
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!(" SUSTAINED THROUGHPUT TEST (12 qubits, 100K depth)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let gcops_avx2 = benchmark_backend("AVX2  ", QBackend::Avx2, 12, 100_000);
    let gcops_avx512 = benchmark_backend("AVX512", QBackend::Avx512, 12, 100_000);

    println!(
        "\n  🚀 AVX512 vs AVX2 speedup: {:.2}x",
        gcops_avx512 / gcops_avx2
    );
    println!("  📊 Expected speedup: ~2.0x (16 vs 8 lanes)");
    println!();
}
