//! EPIC 103: Simple Fusion Validation Test
//!
//! The Core Question: Does fusing N gates into one operation, then applying it M times,
//! beat applying N×M individual gates?
//!
//! If YES (>10× speedup): Fusion is validated, EPIC 81 benchmark is real
//! If NO (<2× speedup): Fusion doesn't help, pivot to creature game
//!
//! Time budget: 30 minutes
//! Run with: cargo run --release --example epic103_fusion_simple

use engine::fusion::{execute_fused_cpu, execute_unfused_cpu, fuse_program};
use engine::quantum::{QGate, QRng, QState};
use std::time::Instant;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 103: Fusion Validation Test (Simple)                    ║");
    println!("║  Does fusion beat naive execution on a real circuit?          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // A real circuit (NOT pathological H²=I case)
    // This is a genuine quantum circuit that might appear in real algorithms
    let circuit = vec![
        QGate::H(0),          // Hadamard on qubit 0
        QGate::X(0),          // NOT on qubit 0
        QGate::Z(0),          // Phase flip on qubit 0
        QGate::H(0),          // Hadamard on qubit 0
        QGate::Phase(0, 0.5), // Parametric phase rotation
        QGate::H(1),          // Hadamard on qubit 1
        QGate::CNot(0, 1),    // Controlled-NOT (entanglement)
        QGate::Z(1),          // Phase flip on qubit 1
    ];

    let iterations = 10_000usize;
    let n_qubits = 4;

    println!("Configuration:");
    println!("  Circuit depth:   {} gates", circuit.len());
    println!("  Iterations:      {}", iterations);
    println!(
        "  Total gate ops:  {} ({}×{})",
        circuit.len() * iterations,
        circuit.len(),
        iterations
    );
    println!("  Qubits:          {}", n_qubits);
    println!();

    // ========================================================================
    // TEST 1: UNFUSED (Naive execution - apply each gate individually)
    // ========================================================================

    println!("Running UNFUSED (naive)...");
    let mut state_unfused = QState::new_zero(n_qubits);
    let mut rng = QRng::new(42);

    let start = Instant::now();

    for _ in 0..iterations {
        let _ = execute_unfused_cpu(&mut state_unfused, &circuit, &mut rng);
    }

    let unfused_time = start.elapsed();
    let unfused_ms = unfused_time.as_secs_f64() * 1000.0;

    println!("  Time: {:.3} ms", unfused_ms);
    println!();

    // ========================================================================
    // TEST 2: FUSED (Optimized - fuse circuit once, apply many times)
    // ========================================================================

    println!("Running FUSED (optimized)...");
    let mut state_fused = QState::new_zero(n_qubits);
    let mut rng = QRng::new(42);

    // Precompute fused program (this is the optimization)
    let fused = fuse_program(&circuit);

    println!("  Fusion stats:");
    println!("    Original gates:  {}", fused.original_gates);
    println!("    Effective ops:   {}", fused.effective_gates());
    println!("    Eliminated:      {}", fused.eliminated);
    println!("    Batches:         {}", fused.batches);
    println!();

    let start = Instant::now();

    for _ in 0..iterations {
        let _ = execute_fused_cpu(&mut state_fused, &fused, &mut rng);
    }

    let fused_time = start.elapsed();
    let fused_ms = fused_time.as_secs_f64() * 1000.0;

    println!("  Time: {:.3} ms", fused_ms);
    println!();

    // ========================================================================
    // RESULTS
    // ========================================================================

    let speedup = unfused_time.as_secs_f64() / fused_time.as_secs_f64();

    println!("═══════════════════════════════════════════════════════════════");
    println!(" RESULTS");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("  Unfused (naive):  {:.3} ms", unfused_ms);
    println!("  Fused (optimized): {:.3} ms", fused_ms);
    println!();
    println!("  Speedup:          {:.2}×", speedup);
    println!();

    // ========================================================================
    // DECISION GATE
    // ========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!(" DECISION");
    println!("═══════════════════════════════════════════════════════════════\n");

    if speedup > 10.0 {
        println!("  ✓ FUSION VALIDATED (speedup > 10×)");
        println!();
        println!("  The fusion mechanism is REAL.");
        println!("  EPIC 81's 63-71 PCOPS benchmark is VALIDATED.");
        println!("  The 100,000× fusion factor is just scaling this mechanism.");
        println!();
        println!("  RECOMMENDATION: Proceed with publication path.");
        println!("                  The benchmark is honest engineering.");
    } else if speedup > 2.0 {
        println!("  ⚠ FUSION HELPS (speedup 2-10×)");
        println!();
        println!("  Fusion provides measurable speedup, but less than expected.");
        println!("  EPIC 81's claims may need recalibration.");
        println!();
        println!("  RECOMMENDATION: Recalibrate claims to match observed speedup.");
        println!("                  Decide: pursue quantum or pivot to game?");
    } else {
        println!("  ✗ FUSION MINIMAL (speedup < 2×)");
        println!();
        println!("  Fusion does not provide meaningful speedup.");
        println!("  EPIC 81 benchmark was measuring something else (likely H²=I).");
        println!();
        println!("  RECOMMENDATION: Pivot to creature game immediately.");
        println!("                  Archive quantum work, ship product in 3-7 days.");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
}
