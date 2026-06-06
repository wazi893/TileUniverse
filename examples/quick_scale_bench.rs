//! Quick scalability benchmark - TileUniverse at 12-20 qubits

use engine::quantum::{QBackend, QGate, QRng, QState, apply_gate_backend};
use std::time::Instant;

fn main() {
    println!("TileUniverse Scalability Benchmark");
    println!("===================================\n");

    let mut rng = QRng::new(42);

    for n_qubits in [12u8, 14, 16, 18, 20] {
        let depth = 500usize;

        // Create state
        let mut state = QState::new_zero(n_qubits);

        // Warmup
        let h_gate = QGate::H(0);
        for _ in 0..100 {
            apply_gate_backend(&mut state, &h_gate, &mut rng, QBackend::Avx2);
        }

        // Reset and measure
        state = QState::new_zero(n_qubits);
        let start = Instant::now();

        for _d in 0..depth {
            for q in 0..n_qubits {
                let gate = QGate::H(q);
                apply_gate_backend(&mut state, &gate, &mut rng, QBackend::Avx2);
            }
            for q in (0..n_qubits - 1).step_by(2) {
                let gate = QGate::CNot(q, q + 1);
                apply_gate_backend(&mut state, &gate, &mut rng, QBackend::Avx2);
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let gates = depth * (n_qubits as usize + n_qubits as usize / 2);
        let amp_ops = (1u64 << n_qubits) * gates as u64;
        let gops = amp_ops as f64 / elapsed / 1e9;

        println!(
            "{:2}q: {:8.1} ms | {:7.2} G amp-ops/s | {:6} gates",
            n_qubits,
            elapsed * 1000.0,
            gops,
            gates
        );
    }
}
