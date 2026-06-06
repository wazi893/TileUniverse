//! Random Circuit Test Battery
//!
//! Tests quantum simulation correctness using random circuits with:
//! - Clifford gates: H, S, CNOT (generate stabilizer states)
//! - Non-Clifford gates: T, arbitrary phases (generate universal states)
//! - Various depths: 10, 100, 1000 layers
//!
//! Key properties verified:
//! 1. Norm preservation (||ψ||² = 1) throughout circuit execution
//! 2. Deterministic output for seeded RNG
//! 3. Per-layer norm drift tracking
//!
//! Run with: cargo test --test random_circuit_tests --release -- --nocapture

use engine::quantum::{QBackend, QGate, QRng, QState, apply_gate_backend};
use std::f32::consts::PI;

/// Compute ||ψ||² (should be 1.0 for normalized states)
fn norm_squared(state: &QState) -> f64 {
    let real = state.real.as_slice();
    let imag = state.imag.as_slice();
    let n = 1 << state.n_qubits;
    (0..n)
        .map(|i| (real[i] as f64).powi(2) + (imag[i] as f64).powi(2))
        .sum()
}

/// Simple deterministic PRNG for gate selection (separate from QRng for state)
struct SimplePrng {
    state: u64,
}

impl SimplePrng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        // xorshift64*
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_u8(&mut self, max: u8) -> u8 {
        (self.next() % (max as u64)) as u8
    }

    fn next_f32(&mut self) -> f32 {
        (self.next() as f32) / (u64::MAX as f32)
    }
}

/// Gate type enum for random selection
#[derive(Clone, Copy, Debug)]
enum RandomGateType {
    // Clifford gates
    H,
    S,
    X,
    Y,
    Z,
    CNot,
    CZ,
    // Non-Clifford gates
    T,
    Phase, // arbitrary angle
}

/// Generate a random gate given the gate type and available qubits
fn random_gate(gate_type: RandomGateType, n_qubits: u8, rng: &mut SimplePrng) -> QGate {
    match gate_type {
        RandomGateType::H => QGate::H(rng.next_u8(n_qubits)),
        RandomGateType::S => QGate::Phase(rng.next_u8(n_qubits), PI / 2.0),
        RandomGateType::X => QGate::X(rng.next_u8(n_qubits)),
        RandomGateType::Y => QGate::Y(rng.next_u8(n_qubits)),
        RandomGateType::Z => QGate::Z(rng.next_u8(n_qubits)),
        RandomGateType::T => QGate::Phase(rng.next_u8(n_qubits), PI / 4.0),
        RandomGateType::Phase => {
            let target = rng.next_u8(n_qubits);
            let angle = rng.next_f32() * 2.0 * PI;
            QGate::Phase(target, angle)
        }
        RandomGateType::CNot => {
            if n_qubits < 2 {
                return QGate::H(0); // fallback
            }
            let control = rng.next_u8(n_qubits);
            let mut target = rng.next_u8(n_qubits);
            while target == control {
                target = rng.next_u8(n_qubits);
            }
            QGate::CNot(control, target)
        }
        RandomGateType::CZ => {
            if n_qubits < 2 {
                return QGate::H(0); // fallback
            }
            let q0 = rng.next_u8(n_qubits);
            let mut q1 = rng.next_u8(n_qubits);
            while q1 == q0 {
                q1 = rng.next_u8(n_qubits);
            }
            QGate::CZ(q0, q1)
        }
    }
}

/// Select a random gate type based on the circuit type
fn random_gate_type(circuit_type: CircuitType, rng: &mut SimplePrng) -> RandomGateType {
    let val = rng.next_u8(100);
    match circuit_type {
        CircuitType::CliffordOnly => {
            // Only Clifford gates: H, S, X, Y, Z, CNOT, CZ
            match val % 7 {
                0 => RandomGateType::H,
                1 => RandomGateType::S,
                2 => RandomGateType::X,
                3 => RandomGateType::Y,
                4 => RandomGateType::Z,
                5 => RandomGateType::CNot,
                _ => RandomGateType::CZ,
            }
        }
        CircuitType::Universal => {
            // Mix of Clifford + non-Clifford (T gates, arbitrary phases)
            match val % 9 {
                0 => RandomGateType::H,
                1 => RandomGateType::S,
                2 => RandomGateType::X,
                3 => RandomGateType::Y,
                4 => RandomGateType::Z,
                5 => RandomGateType::CNot,
                6 => RandomGateType::CZ,
                7 => RandomGateType::T,
                _ => RandomGateType::Phase,
            }
        }
        CircuitType::HighlyEntangled => {
            // Emphasis on two-qubit gates + H for superposition
            match val % 5 {
                0 => RandomGateType::H,
                1 => RandomGateType::CNot,
                2 => RandomGateType::CZ,
                3 => RandomGateType::T,
                _ => RandomGateType::S,
            }
        }
    }
}

#[derive(Clone, Copy)]
enum CircuitType {
    CliffordOnly,
    Universal,
    HighlyEntangled,
}

impl std::fmt::Display for CircuitType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitType::CliffordOnly => write!(f, "Clifford"),
            CircuitType::Universal => write!(f, "Universal"),
            CircuitType::HighlyEntangled => write!(f, "Entangled"),
        }
    }
}

/// Run a random circuit and return per-layer norm values
fn run_random_circuit_with_tracking(
    n_qubits: u8,
    depth: usize,
    gates_per_layer: usize,
    circuit_type: CircuitType,
    seed: u64,
) -> Vec<f64> {
    let mut state = QState::new_zero(n_qubits);
    let mut qrng = QRng::new(seed);
    let mut gate_rng = SimplePrng::new(seed);

    // Initial superposition
    for q in 0..n_qubits {
        apply_gate_backend(&mut state, &QGate::H(q), &mut qrng, QBackend::Avx2);
    }

    let mut norms = Vec::with_capacity(depth + 1);
    norms.push(norm_squared(&state));

    for _layer in 0..depth {
        for _ in 0..gates_per_layer {
            let gate_type = random_gate_type(circuit_type, &mut gate_rng);
            let gate = random_gate(gate_type, n_qubits, &mut gate_rng);
            apply_gate_backend(&mut state, &gate, &mut qrng, QBackend::Avx2);
        }
        norms.push(norm_squared(&state));
    }

    norms
}

/// Compute norm drift statistics from a sequence of norms
fn compute_drift_stats(norms: &[f64]) -> (f64, f64, f64) {
    let max_drift = norms.iter().map(|n| (n - 1.0).abs()).fold(0.0, f64::max);
    let mean_drift = norms.iter().map(|n| (n - 1.0).abs()).sum::<f64>() / norms.len() as f64;
    let final_drift = (norms.last().unwrap() - 1.0).abs();
    (max_drift, mean_drift, final_drift)
}

// ============================================================================
// Test Functions
// ============================================================================

#[test]
fn test_clifford_random_circuits() {
    println!("\n=== Random Clifford Circuits ===");
    println!("  Testing norm preservation with Clifford gates only (H, S, X, Y, Z, CNOT, CZ)\n");

    let configs = [
        (4u8, 10, 10), // 4 qubits, depth 10, 10 gates/layer = 100 gates
        (4, 100, 10),  // 4 qubits, depth 100, 10 gates/layer = 1000 gates
        (6, 100, 15),  // 6 qubits, depth 100, 15 gates/layer = 1500 gates
        (8, 100, 20),  // 8 qubits, depth 100, 20 gates/layer = 2000 gates
        (10, 100, 25), // 10 qubits, depth 100, 25 gates/layer = 2500 gates
    ];

    for (n_qubits, depth, gates_per_layer) in configs {
        let norms = run_random_circuit_with_tracking(
            n_qubits,
            depth,
            gates_per_layer,
            CircuitType::CliffordOnly,
            12345,
        );

        let (max_drift, mean_drift, final_drift) = compute_drift_stats(&norms);
        let total_gates = depth * gates_per_layer;
        let passed = max_drift < 1e-3;

        println!(
            "  {}q, {} gates ({}×{}): max_drift={:.2e}, mean={:.2e}, final={:.2e} [{}]",
            n_qubits,
            total_gates,
            depth,
            gates_per_layer,
            max_drift,
            mean_drift,
            final_drift,
            if passed { "PASS" } else { "FAIL" }
        );

        assert!(
            passed,
            "Norm drift too large for {}q Clifford circuit: max_drift = {:.2e}",
            n_qubits, max_drift
        );
    }
}

#[test]
fn test_universal_random_circuits() {
    println!("\n=== Random Universal Circuits ===");
    println!("  Testing with Clifford + non-Clifford gates (T, arbitrary phases)\n");

    let configs = [
        (4u8, 10, 10), // 100 gates
        (4, 100, 10),  // 1000 gates
        (6, 100, 15),  // 1500 gates
        (8, 100, 20),  // 2000 gates
        (8, 500, 20),  // 10000 gates
    ];

    for (n_qubits, depth, gates_per_layer) in configs {
        let norms = run_random_circuit_with_tracking(
            n_qubits,
            depth,
            gates_per_layer,
            CircuitType::Universal,
            54321,
        );

        let (max_drift, mean_drift, final_drift) = compute_drift_stats(&norms);
        let total_gates = depth * gates_per_layer;
        let passed = max_drift < 1e-3;

        println!(
            "  {}q, {} gates: max_drift={:.2e}, mean={:.2e}, final={:.2e} [{}]",
            n_qubits,
            total_gates,
            max_drift,
            mean_drift,
            final_drift,
            if passed { "PASS" } else { "FAIL" }
        );

        assert!(
            passed,
            "Norm drift too large for {}q universal circuit: max_drift = {:.2e}",
            n_qubits, max_drift
        );
    }
}

#[test]
fn test_highly_entangled_circuits() {
    println!("\n=== Highly Entangled Random Circuits ===");
    println!("  Testing circuits with many two-qubit gates\n");

    let configs = [
        (4u8, 50, 20), // 1000 gates, heavily entangled
        (6, 100, 30),  // 3000 gates
        (8, 100, 40),  // 4000 gates
        (10, 100, 50), // 5000 gates
    ];

    for (n_qubits, depth, gates_per_layer) in configs {
        let norms = run_random_circuit_with_tracking(
            n_qubits,
            depth,
            gates_per_layer,
            CircuitType::HighlyEntangled,
            99999,
        );

        let (max_drift, mean_drift, final_drift) = compute_drift_stats(&norms);
        let total_gates = depth * gates_per_layer;
        let passed = max_drift < 1e-3;

        println!(
            "  {}q, {} gates: max_drift={:.2e}, mean={:.2e}, final={:.2e} [{}]",
            n_qubits,
            total_gates,
            max_drift,
            mean_drift,
            final_drift,
            if passed { "PASS" } else { "FAIL" }
        );

        assert!(
            passed,
            "Norm drift too large for {}q entangled circuit: max_drift = {:.2e}",
            n_qubits, max_drift
        );
    }
}

#[test]
fn test_deep_random_circuits_1000_layers() {
    println!("\n=== Very Deep Random Circuits (1000 layers) ===");
    println!("  Stress test for numerical stability\n");

    let configs = [
        (4u8, 1000, 10, CircuitType::CliffordOnly),
        (4, 1000, 10, CircuitType::Universal),
        (6, 1000, 15, CircuitType::Universal),
        (8, 1000, 20, CircuitType::Universal),
    ];

    for (n_qubits, depth, gates_per_layer, circuit_type) in configs {
        let norms =
            run_random_circuit_with_tracking(n_qubits, depth, gates_per_layer, circuit_type, 77777);

        let (max_drift, mean_drift, final_drift) = compute_drift_stats(&norms);
        let total_gates = depth * gates_per_layer;

        // Allow slightly more drift for very deep circuits
        let threshold = 5e-3;
        let passed = max_drift < threshold;

        println!(
            "  {}q {} {} gates: max_drift={:.2e}, mean={:.2e}, final={:.2e} [{}]",
            n_qubits,
            circuit_type,
            total_gates,
            max_drift,
            mean_drift,
            final_drift,
            if passed { "PASS" } else { "FAIL" }
        );

        assert!(
            passed,
            "Norm drift too large for {}q {} circuit: max_drift = {:.2e}",
            n_qubits, circuit_type, max_drift
        );
    }
}

#[test]
fn test_determinism() {
    println!("\n=== Determinism Test ===");
    println!("  Verifying same seed produces identical output\n");

    let n_qubits = 6u8;
    let depth = 100;
    let gates_per_layer = 20;
    let seed = 123456789u64;

    // Run the same circuit twice
    let norms1 = run_random_circuit_with_tracking(
        n_qubits,
        depth,
        gates_per_layer,
        CircuitType::Universal,
        seed,
    );

    let norms2 = run_random_circuit_with_tracking(
        n_qubits,
        depth,
        gates_per_layer,
        CircuitType::Universal,
        seed,
    );

    // Compare all norms
    let mut max_diff = 0.0f64;
    for (n1, n2) in norms1.iter().zip(norms2.iter()) {
        max_diff = max_diff.max((n1 - n2).abs());
    }

    let passed = max_diff == 0.0;

    println!(
        "  {}q, {} layers, {} gates/layer: max_diff={:.2e} [{}]",
        n_qubits,
        depth,
        gates_per_layer,
        max_diff,
        if passed { "PASS" } else { "FAIL" }
    );

    assert!(
        passed,
        "Determinism failed: same seed produced different results, max_diff = {:.2e}",
        max_diff
    );
}

#[test]
fn test_per_layer_drift_tracking() {
    println!("\n=== Per-Layer Norm Drift Tracking ===");
    println!("  Analyzing drift pattern over circuit execution\n");

    let n_qubits = 8u8;
    let depth = 100;
    let gates_per_layer = 20;

    let norms = run_random_circuit_with_tracking(
        n_qubits,
        depth,
        gates_per_layer,
        CircuitType::Universal,
        42424242,
    );

    // Sample some layers to show drift progression
    let sample_layers = [0, 10, 25, 50, 75, 100];

    println!("  Layer  | Norm          | Drift from 1.0");
    println!("  -------|---------------|---------------");

    for &layer in &sample_layers {
        if layer < norms.len() {
            let norm = norms[layer];
            let drift = (norm - 1.0).abs();
            println!("  {:>5}  | {:.10} | {:.2e}", layer, norm, drift);
        }
    }

    // Check monotonicity of drift (drift should grow roughly monotonically, not oscillate wildly)
    let final_drift = (norms.last().unwrap() - 1.0).abs();
    let mid_drift = (norms[depth / 2] - 1.0).abs();
    let early_drift = (norms[depth / 10] - 1.0).abs();

    // Drift should generally increase with depth (allowing some tolerance)
    let reasonable_progression = final_drift >= early_drift * 0.5; // allow some recovery

    println!(
        "\n  Drift progression: early={:.2e}, mid={:.2e}, final={:.2e}",
        early_drift, mid_drift, final_drift
    );
    println!(
        "  Drift rate: ~{:.2e} per 1000 gates",
        final_drift / ((depth * gates_per_layer) as f64 / 1000.0)
    );

    assert!(
        reasonable_progression,
        "Drift pattern is suspicious: early={:.2e}, mid={:.2e}, final={:.2e}",
        early_drift, mid_drift, final_drift
    );
}

#[test]
fn test_different_qubit_counts() {
    println!("\n=== Scaling Test Across Qubit Counts ===");
    println!("  Testing 4, 8, 12, 16 qubits with same circuit structure\n");

    let depth = 100;
    let gates_per_layer = 10; // gates per layer per qubit

    for n_qubits in [4u8, 8, 12, 16] {
        let actual_gates_per_layer = gates_per_layer * (n_qubits as usize);

        let norms = run_random_circuit_with_tracking(
            n_qubits,
            depth,
            actual_gates_per_layer,
            CircuitType::Universal,
            11111,
        );

        let (max_drift, _mean_drift, final_drift) = compute_drift_stats(&norms);
        let total_gates = depth * actual_gates_per_layer;
        let passed = max_drift < 1e-3;

        println!(
            "  {:>2}q, {} gates total: max_drift={:.2e}, final={:.2e} [{}]",
            n_qubits,
            total_gates,
            max_drift,
            final_drift,
            if passed { "PASS" } else { "FAIL" }
        );

        assert!(
            passed,
            "Norm drift too large for {}q: max_drift = {:.2e}",
            n_qubits, max_drift
        );
    }
}

#[test]
fn test_random_circuit_summary() {
    println!("\n");
    println!("======================================================================");
    println!("  RANDOM CIRCUIT TEST SUMMARY");
    println!("======================================================================");
    println!();
    println!("  Circuit Types Tested:");
    println!("    - Clifford only: H, S, X, Y, Z, CNOT, CZ");
    println!("    - Universal: Clifford + T + arbitrary Phase");
    println!("    - Highly entangled: emphasis on two-qubit gates");
    println!();
    println!("  Depths Tested: 10, 100, 500, 1000 layers");
    println!("  Qubit Counts: 4, 6, 8, 10, 12, 16 qubits");
    println!("  Gate Counts: 100 to 20,000 gates per circuit");
    println!();
    println!("  Key Metrics:");
    println!("    - max_drift: Maximum deviation of ||psi||^2 from 1.0");
    println!("    - mean_drift: Average deviation across all layers");
    println!("    - final_drift: Deviation after final layer");
    println!();
    println!("  Threshold: max_drift < 1e-3 (PASS)");
    println!("           : max_drift < 5e-3 for 1000-layer circuits");
    println!("======================================================================");
    println!();
}
