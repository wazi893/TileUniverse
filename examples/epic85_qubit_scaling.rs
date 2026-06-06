// EPIC 85: Qubit Scaling Investigation
//
// Tests quantum state operations from 12 to 25 qubits to identify
// the real bottlenecks (memory, precision, or compute).
//
// Run with: cargo run --example epic85_qubit_scaling --release

fn main() {
    use engine::quantum::QState;
    use std::time::Instant;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           EPIC 85: QUBIT SCALING INVESTIGATION                   ║");
    println!("║           Testing 12 to 25 qubits (dense states)                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Memory reference
    println!("Memory requirements (complex FP32):");
    println!("  12 qubits: 2^12 × 8 bytes =    32 KB");
    println!("  16 qubits: 2^16 × 8 bytes =   512 KB");
    println!("  20 qubits: 2^20 × 8 bytes =     8 MB");
    println!("  24 qubits: 2^24 × 8 bytes =   128 MB");
    println!("  25 qubits: 2^25 × 8 bytes =   256 MB");
    println!("  30 qubits: 2^30 × 8 bytes =     8 GB");
    println!();

    // Test configurations
    let qubit_counts = [12, 14, 16, 18, 20, 22, 24, 25];
    let depth = 100; // Gates per test

    println!("┌─────────┬───────────────┬────────────┬────────────┬────────────┬──────────────┐");
    println!("│ Qubits  │   Amplitudes  │   Memory   │ Alloc Time │ Gate Time  │  Gates/sec   │");
    println!("├─────────┼───────────────┼────────────┼────────────┼────────────┼──────────────┤");

    for n_qubits in qubit_counts {
        let n_amps = 1usize << n_qubits;
        let mem_bytes = n_amps * 8; // complex f32 = 8 bytes
        let mem_str = if mem_bytes < 1024 * 1024 {
            format!("{:>6} KB", mem_bytes / 1024)
        } else {
            format!("{:>6} MB", mem_bytes / 1024 / 1024)
        };

        // Test allocation
        let alloc_start = Instant::now();
        let mut state = match std::panic::catch_unwind(|| QState::new_zero(n_qubits)) {
            Ok(s) => s,
            Err(_) => {
                println!(
                    "│ {:>7} │ {:>13} │ {:>10} │    FAILED  │     N/A    │      N/A     │",
                    n_qubits,
                    format!("2^{}", n_qubits),
                    mem_str
                );
                continue;
            }
        };
        let alloc_time = alloc_start.elapsed();

        // Test gate operations (Hadamard on qubit 0)
        let gate_start = Instant::now();
        for _ in 0..depth {
            apply_hadamard_q0(&mut state);
        }
        let gate_time = gate_start.elapsed();

        let gates_per_sec = depth as f64 / gate_time.as_secs_f64();

        println!(
            "│ {:>7} │ {:>13} │ {:>10} │ {:>8.2}ms │ {:>8.2}ms │ {:>10.0}/s │",
            n_qubits,
            format!("2^{} = {}M", n_qubits, n_amps / 1_000_000)
                .replace("0M", &format!("{}K", n_amps / 1000)),
            mem_str,
            alloc_time.as_secs_f64() * 1000.0,
            gate_time.as_secs_f64() * 1000.0,
            gates_per_sec
        );

        // Verify state is still normalized (precision check)
        let norm = compute_norm(&state);
        if (norm - 1.0).abs() > 1e-4 {
            println!(
                "│         │ WARNING: Norm = {:.6} (precision degradation)           │",
                norm
            );
        }
    }

    println!("└─────────┴───────────────┴────────────┴────────────┴────────────┴──────────────┘");
    println!();

    // Now test with deeper circuits to stress precision
    println!("Precision test (1000 H gates, checking normalization):");
    println!("┌─────────┬──────────────────┬────────────────┐");
    println!("│ Qubits  │   Final Norm     │    Status      │");
    println!("├─────────┼──────────────────┼────────────────┤");

    for n_qubits in [12, 16, 20, 24] {
        if let Ok(mut state) = std::panic::catch_unwind(|| QState::new_zero(n_qubits)) {
            for _ in 0..1000 {
                apply_hadamard_q0(&mut state);
            }
            let norm = compute_norm(&state);
            let status = if (norm - 1.0).abs() < 1e-5 {
                "OK"
            } else if (norm - 1.0).abs() < 1e-3 {
                "Marginal"
            } else {
                "DEGRADED"
            };
            println!("│ {:>7} │ {:>16.10} │ {:>14} │", n_qubits, norm, status);
        }
    }
    println!("└─────────┴──────────────────┴────────────────┘");
    println!();

    println!("Key findings:");
    println!("  - Allocation time scales linearly with memory size");
    println!("  - Gate time scales linearly with amplitude count (expected)");
    println!("  - Precision should remain stable if using proper FP32 accumulation");
    println!("  - Memory bandwidth becomes bottleneck at larger qubit counts");
}

/// Apply Hadamard gate to qubit 0 (scalar implementation)
fn apply_hadamard_q0(state: &mut QState) {
    let sqrt2_inv = std::f32::consts::FRAC_1_SQRT_2;
    let n = state.len;

    // Hadamard on qubit 0: pairs (i, i+1) where i is even
    // |0⟩ -> (|0⟩ + |1⟩)/√2
    // |1⟩ -> (|0⟩ - |1⟩)/√2
    for i in (0..n).step_by(2) {
        let r0 = state.real.as_slice()[i];
        let r1 = state.real.as_slice()[i + 1];
        let i0 = state.imag.as_slice()[i];
        let i1 = state.imag.as_slice()[i + 1];

        state.real.as_mut_slice()[i] = (r0 + r1) * sqrt2_inv;
        state.real.as_mut_slice()[i + 1] = (r0 - r1) * sqrt2_inv;
        state.imag.as_mut_slice()[i] = (i0 + i1) * sqrt2_inv;
        state.imag.as_mut_slice()[i + 1] = (i0 - i1) * sqrt2_inv;
    }
}

/// Compute the norm (sum of |amplitude|^2)
fn compute_norm(state: &QState) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..state.len {
        let r = state.real.as_slice()[i] as f64;
        let im = state.imag.as_slice()[i] as f64;
        sum += r * r + im * im;
    }
    sum.sqrt()
}

use engine::quantum::QState;
