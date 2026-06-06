//! Path Optimizer Comparison Benchmark
//!
//! Compares greedy vs beam search path optimization.
//! Key metrics: peak intermediate size, total time, correctness.
//!
//! Run with: cargo run --release --example path_optimizer_bench

use engine::quantum::{QGate, QRng, QState, apply_gate_scalar};
use engine::tensor_network::{
    GreedyMetric, PathConfig, circuit_to_network, compute_amplitude,
    find_greedy_order, find_optimal_path, verify_path_peak_memory,
};
use std::time::Instant;

/// Simple xorshift RNG for benchmarking
struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

/// Generate a random circuit with given depth
fn random_circuit(n_qubits: u8, depth: usize, seed: u64) -> Vec<QGate> {
    let mut rng = SimpleRng::new(seed);
    let mut gates = Vec::with_capacity(depth);

    for _ in 0..depth {
        let gate_type = rng.next() % 6;
        let q1 = (rng.next() % n_qubits as u64) as u8;

        match gate_type {
            0 => gates.push(QGate::H(q1)),
            1 => gates.push(QGate::X(q1)),
            2 => gates.push(QGate::Z(q1)),
            3 => gates.push(QGate::T(q1)),
            4 if n_qubits > 1 => {
                let q2 = ((rng.next() % (n_qubits - 1) as u64) as u8 + q1 + 1) % n_qubits;
                gates.push(QGate::CNot(q1, q2));
            }
            5 if n_qubits > 1 => {
                let q2 = ((rng.next() % (n_qubits - 1) as u64) as u8 + q1 + 1) % n_qubits;
                gates.push(QGate::CZ(q1, q2));
            }
            _ => gates.push(QGate::H(q1)),
        }
    }

    gates
}

/// Compute amplitude using state vector simulation (ground truth)
fn statevector_amplitude(gates: &[QGate], n_qubits: u8, in_bits: u64, out_bits: u64) -> (f32, f32) {
    let mut state = QState::new_zero(n_qubits);
    {
        let real = state.real.as_mut_slice();
        let imag = state.imag.as_mut_slice();
        real.fill(0.0);
        imag.fill(0.0);
        real[in_bits as usize] = 1.0;
    }
    let mut rng = QRng::new(42);
    for gate in gates {
        apply_gate_scalar(&mut state, gate, &mut rng);
    }
    let real = state.real.as_slice();
    let imag = state.imag.as_slice();
    (real[out_bits as usize], imag[out_bits as usize])
}

fn main() {
    println!("=== Path Optimizer Comparison Benchmark ===\n");

    // Test 1: Compare path quality (peak size)
    println!("--- Test 1: Path Quality (Peak Intermediate Size) ---");
    println!(
        "{:>6} {:>6} {:>14} {:>14} {:>10} {:>8}",
        "Qubits", "Depth", "Greedy Peak", "Beam Peak", "Improvement", "Valid"
    );
    println!("{:-<70}", "");

    for n_qubits in [6, 8, 10, 12, 14] {
        for depth in [10, 20, 40] {
            let circuit = random_circuit(n_qubits, depth, 12345);
            let network = circuit_to_network(&circuit, n_qubits);

            // Greedy path
            let greedy_path = find_greedy_order(&network, GreedyMetric::MinMemory);
            let greedy_peak = greedy_path.peak_memory;

            // Beam search path
            let config = PathConfig::default();
            let beam_path = find_optimal_path(&network, &config);
            let beam_peak = beam_path.peak_memory;

            // Verify path peak memory matches single source of truth
            let valid = verify_path_peak_memory(&network, &beam_path).is_ok();

            let improvement = if beam_peak > 0 && greedy_peak > 0 {
                greedy_peak as f64 / beam_peak as f64
            } else {
                1.0
            };

            println!(
                "{:>6} {:>6} {:>14} {:>14} {:>9.1}x {:>8}",
                n_qubits,
                depth,
                greedy_peak,
                beam_peak,
                improvement,
                if valid { "OK" } else { "FAIL" }
            );
        }
    }

    // Test 2: Compare optimization time only (execution uses same compute_amplitude)
    println!("\n--- Test 2: Optimization Time Only ---");
    println!(
        "{:>6} {:>6} {:>14} {:>14} {:>10}",
        "Qubits", "Depth", "Greedy Opt", "Beam Opt", "Ratio"
    );
    println!("{:-<60}", "");

    for n_qubits in [8, 10, 12, 14] {
        let depth = 20;
        let circuit = random_circuit(n_qubits, depth, 12345);
        let network = circuit_to_network(&circuit, n_qubits);

        // Time greedy optimization only
        let greedy_start = Instant::now();
        let _greedy_path = find_greedy_order(&network, GreedyMetric::MinMemory);
        let greedy_time = greedy_start.elapsed();

        // Time beam search optimization only
        let beam_start = Instant::now();
        let config = PathConfig::default();
        let _beam_path = find_optimal_path(&network, &config);
        let beam_time = beam_start.elapsed();

        // Note: execution time is the same since compute_amplitude uses greedy internally
        // The benefit of beam search is the reduced peak memory, not execution time

        let ratio = if greedy_time.as_nanos() > 0 {
            beam_time.as_secs_f64() / greedy_time.as_secs_f64()
        } else {
            1.0
        };

        println!(
            "{:>6} {:>6} {:>12}us {:>12}us {:>9.1}x",
            n_qubits,
            depth,
            greedy_time.as_micros(),
            beam_time.as_micros(),
            ratio
        );
    }

    // Test 3: Correctness verification
    println!("\n--- Test 3: Correctness Verification ---");

    for n_qubits in [4, 6, 8, 10, 12] {
        let circuit = random_circuit(n_qubits, 20, 54321);
        let network = circuit_to_network(&circuit, n_qubits);

        let tn_amp = compute_amplitude(&network, 0, 0);
        let (sv_re, sv_im) = statevector_amplitude(&circuit, n_qubits, 0, 0);

        let err = (tn_amp.re - sv_re).abs().max((tn_amp.im - sv_im).abs());
        let status = if err < 1e-5 { "OK" } else { "FAIL" };

        println!(
            "{:>2} qubits: TN=({:.6}, {:.6}i) SV=({:.6}, {:.6}i) err={:.2e} [{}]",
            n_qubits, tn_amp.re, tn_amp.im, sv_re, sv_im, err, status
        );
    }

    // Test 4: Memory budget enforcement with slicing
    println!("\n--- Test 4: Memory Budget with Slicing ---");

    use engine::tensor_network::plan_slicing;

    // Test slicing on the raw network (no capping) - this represents computing
    // the full transfer matrix. Peak memory can be very large.

    let circuit = random_circuit(6, 30, 99999); // 6 qubits, depth 30
    let network = circuit_to_network(&circuit, 6);

    // Use the raw network - open_indices includes all 12 boundary indices
    // The internal contraction bonds create large intermediates
    let path = find_greedy_order(&network, GreedyMetric::MinMemory);
    let base_peak = path.peak_memory;

    println!("Raw network: 6 qubits, depth 30, peak={}", base_peak);
    println!(
        "{:>12} {:>12} {:>12} {:>10}",
        "Budget", "Est. Peak", "Slices", "Iterations"
    );
    println!("{:-<50}", "");

    for budget_log in [8, 10, 12, 14] {
        let budget = 1usize << budget_log;

        match plan_slicing(&network, &path, budget) {
            Some(sliced) => {
                let iters = sliced.slice_iterations();
                let slices = sliced.slices.len();
                println!(
                    "2^{:>2}={:>8} {:>12} {:>12} {:>10}",
                    budget_log, budget, sliced.estimated_peak, slices, iters
                );
            }
            None => {
                println!(
                    "2^{:>2}={:>8} {:>12} {:>12} {:>10}",
                    budget_log, budget, base_peak, 0, 1
                );
            }
        }
    }

    // Test 5: Scaling behavior
    println!("\n--- Test 5: Scaling (Fixed Depth=20) ---");
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>8}",
        "Qubits", "Greedy Peak", "Beam Peak", "Beam Time", "Valid"
    );
    println!("{:-<60}", "");

    for n_qubits in [6, 8, 10, 12, 14, 16] {
        let circuit = random_circuit(n_qubits, 20, 12345);
        let network = circuit_to_network(&circuit, n_qubits);

        let greedy_path = find_greedy_order(&network, GreedyMetric::MinMemory);

        let start = Instant::now();
        let config = PathConfig::default();
        let beam_path = find_optimal_path(&network, &config);
        let time = start.elapsed();

        // Verify path is valid
        let valid = verify_path_peak_memory(&network, &beam_path).is_ok();

        println!(
            "{:>6} {:>12} {:>12} {:>10}ms {:>8}",
            n_qubits,
            greedy_path.peak_memory,
            beam_path.peak_memory,
            time.as_millis(),
            if valid { "OK" } else { "FAIL" }
        );
    }

    println!("\n=== Benchmark Complete ===");
}
