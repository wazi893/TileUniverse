//! EPIC 103: Fusion Scaling Test
//!
//! Test how fusion speedup scales with circuit depth
//! Hypothesis: Deeper circuits should show more fusion benefit

use engine::fusion::{execute_fused_cpu, execute_unfused_cpu, fuse_program};
use engine::quantum::{QGate, QRng, QState};
use std::time::Instant;

fn build_circuit(depth: usize) -> Vec<QGate> {
    let mut gates = Vec::new();
    for i in 0..depth {
        let q = (i % 4) as u8; // Cycle through 4 qubits
        match i % 6 {
            0 => gates.push(QGate::H(q)),
            1 => gates.push(QGate::X(q)),
            2 => gates.push(QGate::Z(q)),
            3 => gates.push(QGate::Phase(q, 0.5)),
            4 => gates.push(QGate::Rx(q, 0.3)),
            5 => gates.push(QGate::Ry(q, 0.7)),
            _ => unreachable!(),
        }
    }
    gates
}

fn benchmark_circuit(circuit: &[QGate], iterations: usize) -> (f64, f64, f64) {
    let n_qubits = 4;

    // Unfused
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(42);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = execute_unfused_cpu(&mut state, circuit, &mut rng);
    }
    let unfused_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Fused
    let fused = fuse_program(circuit);
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(42);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = execute_fused_cpu(&mut state, &fused, &mut rng);
    }
    let fused_ms = start.elapsed().as_secs_f64() * 1000.0;

    let speedup = unfused_ms / fused_ms;
    (unfused_ms, fused_ms, speedup)
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 103: Fusion Scaling Test                                ║");
    println!("║  How does fusion speedup scale with circuit depth?            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!(
        "  {:>6} | {:>10} | {:>12} | {:>12} | {:>8}",
        "Depth", "Iterations", "Unfused (ms)", "Fused (ms)", "Speedup"
    );
    println!(
        "  {:-<6}-+-{:-<10}-+-{:-<12}-+-{:-<12}-+-{:-<8}",
        "", "", "", "", ""
    );

    let configs = vec![
        (4, 50_000),
        (8, 25_000),
        (16, 12_000),
        (32, 6_000),
        (64, 3_000),
        (128, 1_500),
        (256, 750),
    ];

    let mut max_speedup = 0.0f64;
    let mut max_depth = 0usize;

    for (depth, iterations) in configs {
        let circuit = build_circuit(depth);
        let (unfused_ms, fused_ms, speedup) = benchmark_circuit(&circuit, iterations);

        println!(
            "  {:>6} | {:>10} | {:>12.3} | {:>12.3} | {:>6.2}×",
            depth, iterations, unfused_ms, fused_ms, speedup
        );

        if speedup > max_speedup {
            max_speedup = speedup;
            max_depth = depth;
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!(" ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════\n");
    println!(
        "  Best speedup:     {:.2}× (at depth {})",
        max_speedup, max_depth
    );
    println!();

    if max_speedup > 10.0 {
        println!("  ✓ FUSION VALIDATED - Achieves >10× on deep circuits");
    } else if max_speedup > 5.0 {
        println!("  ⚠ FUSION MODERATE - Helps but <10× maximum");
    } else {
        println!("  ✗ FUSION MINIMAL - Never exceeds 5× speedup");
    }
    println!();
}
