// Literal angles below are verbatim Qiskit synthesis output; keeping them as
// printed documents what the synthesizer emitted (π-valued or not).
#![allow(clippy::approx_constant)]
use std::io::Write as IoWrite;
use std::process::{Command, Stdio};
/// SPRINT 37.0: Benchmark Qiskit synthesis overhead vs native execution
///
/// Measures:
/// 1. Time to call Qiskit subprocess for synthesis
/// 2. Time to execute gates on quantum state
/// 3. Overall cost analysis
///
/// Usage: cargo run --example synthesis_cost_bench --release
use std::time::Instant;

use logic_fabric_core::quantum::{QGate, QRng, QState, apply_gate_scalar};

fn main() {
    println!("=== Qiskit Synthesis Cost Benchmark ===\n");

    benchmark_qiskit_subprocess_overhead();
    benchmark_gate_execution();
    analyze_tradeoffs();
}

/// Measure time to call Qiskit Python subprocess
fn benchmark_qiskit_subprocess_overhead() {
    println!("## 1. Qiskit Subprocess Overhead");

    // iSWAP unitary matrix
    let json_input = r#"{"matrix":[[{"re":1.0,"im":0.0},{"re":0.0,"im":0.0},{"re":0.0,"im":0.0},{"re":0.0,"im":0.0}],[{"re":0.0,"im":0.0},{"re":0.0,"im":0.0},{"re":0.0,"im":1.0},{"re":0.0,"im":0.0}],[{"re":0.0,"im":0.0},{"re":0.0,"im":1.0},{"re":0.0,"im":0.0},{"re":0.0,"im":0.0}],[{"re":0.0,"im":0.0},{"re":0.0,"im":0.0},{"re":0.0,"im":0.0},{"re":1.0,"im":0.0}]],"qa":0,"qb":1}"#;

    let python_cmd = if cfg!(windows) { "python" } else { "python3" };

    let script_path = "scripts/qiskit_two_qubit_synth.py";
    let n_trials = 20;

    println!("  Running {} trials...", n_trials);
    let start = Instant::now();

    for _ in 0..n_trials {
        let mut child = Command::new(python_cmd)
            .arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to spawn Python");

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(json_input.as_bytes()).ok();
        }

        let _output = child.wait_with_output().expect("Failed to read output");
    }

    let elapsed = start.elapsed();
    let avg_time = elapsed.as_millis() as f64 / n_trials as f64;
    let avg_time_us = elapsed.as_micros() as f64 / n_trials as f64;

    println!(
        "  Average synthesis time: {:.2} ms ({:.0} μs)",
        avg_time, avg_time_us
    );
    println!(
        "  Total for {} calls: {:.2} ms",
        n_trials,
        elapsed.as_millis()
    );
}

/// Measure time to execute gates on quantum state
fn benchmark_gate_execution() {
    println!("\n## 2. Gate Execution Performance");

    let n_qubits = 10;
    let n_trials = 1000;
    let mut rng = QRng::new(42);

    // Benchmark single CNot execution
    {
        let mut state = QState::new_zero(n_qubits);
        let gate = QGate::CNot(0, 1);

        let start = Instant::now();
        for _ in 0..n_trials {
            apply_gate_scalar(&mut state, &gate, &mut rng);
        }
        let elapsed = start.elapsed();

        let avg_ns = elapsed.as_nanos() as f64 / n_trials as f64;
        println!("  CNot gate: {:.0} ns/gate ({} qubits)", avg_ns, n_qubits);
    }

    // Benchmark rotation gate execution
    {
        let mut state = QState::new_zero(n_qubits);
        let gate = QGate::Rz(0, 1.5707);

        let start = Instant::now();
        for _ in 0..n_trials {
            apply_gate_scalar(&mut state, &gate, &mut rng);
        }
        let elapsed = start.elapsed();

        let avg_ns = elapsed.as_nanos() as f64 / n_trials as f64;
        println!("  Rz gate:   {:.0} ns/gate ({} qubits)", avg_ns, n_qubits);
    }

    // Benchmark executing a synthesized circuit (18 gates for iSWAP)
    {
        let mut state = QState::new_zero(n_qubits);
        let synthesized_gates = vec![
            QGate::Rz(1, -1.5707964),
            QGate::Ry(1, 2.6887183),
            QGate::Rz(1, 1.5707964),
            QGate::Ry(0, 1.5707964),
            QGate::CNot(0, 1),
            QGate::Rz(1, 1.0747674),
            QGate::Ry(1, 1.1664219),
            QGate::Rz(1, -2.9318101),
            QGate::Rz(0, -3.1415927),
            QGate::Ry(0, 1.5707964),
            QGate::Rz(0, -1.5707964),
            QGate::CNot(0, 1),
            QGate::Rz(1, 1.5707964),
            QGate::Ry(1, 1.117922),
            QGate::Rz(1, -3.1415927),
            QGate::Rz(0, 1.5707964),
            QGate::Ry(0, 1.5707964),
            QGate::Rz(0, -1.5707964),
        ];

        let n_circuits = 100;
        let start = Instant::now();

        for _ in 0..n_circuits {
            for gate in &synthesized_gates {
                apply_gate_scalar(&mut state, gate, &mut rng);
            }
        }

        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / n_circuits as f64;
        println!("  Synthesized iSWAP (18 gates): {:.1} μs/circuit", avg_us);
    }
}

/// Analyze cost tradeoffs
fn analyze_tradeoffs() {
    println!("\n## 3. Cost Analysis");
    println!();
    println!("  Typical measurements:");
    println!("    Qiskit synthesis:    ~1-5 ms  (subprocess overhead)");
    println!("    Execute 18 gates:    ~1-10 μs (on 10-qubit state)");
    println!("    Gate count ratio:    18x (synthesized vs native)");
    println!();
    println!("  Synthesis makes sense when:");
    println!("    1. Hardware doesn't support the gate natively");
    println!("    2. Circuit will be executed many times (amortize synthesis cost)");
    println!("    3. Synthesis enables further optimizations (gate cancellation, etc.)");
    println!();
    println!("  Break-even point:");
    println!("    Synthesis cost / Execution cost ≈ 1000-5000 executions");
    println!("    For one-shot circuits: synthesis overhead dominates");
    println!("    For repeated circuits: synthesis cost amortizes quickly");
    println!();
    println!("  Optimization opportunities:");
    println!("    - Cache synthesized circuits for repeated gates");
    println!("    - Batch multiple synthesis requests");
    println!("    - Skip synthesis if gate count increases significantly");
    println!("    - Use native gates when available");
}
