use engine::fusion::{FusionConfig, execute_fused_cpu, fuse_program_with_config};
use engine::quantum::{QGate, QRng, QState, apply_gate_scalar}; // Import apply_gate_scalar

// Constants for the torture test
const NUM_TEST_CIRCUITS: usize = 100;
const MAX_GATES: usize = 50;
const N_QUBITS: u8 = 4;

// Simple local RNG for test generation (since QRng internals are private)
struct TestRng {
    state: u64,
}

impl TestRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

// Helper to run ground truth scalar simulation
fn run_ground_truth(circuit: &[QGate], n_qubits: u8) -> QState {
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(0); // Dummy RNG for gates that might need it (Measurement)

    for gate in circuit {
        apply_gate_scalar(&mut state, gate, &mut rng);
    }
    state
}

#[test]
fn sanctity_torture_test() {
    println!(
        "Starting Sanctity Torture Test: {} random circuits...",
        NUM_TEST_CIRCUITS
    );

    let mut rng = TestRng::new(12345);
    let mut failures = 0;

    for i in 0..NUM_TEST_CIRCUITS {
        let circuit = generate_random_circuit(&mut rng, N_QUBITS, MAX_GATES);

        // 1. Run Ground Truth (Scalar Simulation)
        let ground_truth_state = run_ground_truth(&circuit, N_QUBITS);

        // 2. Run Algebraic Reduction Engine (Optimized)
        let mut config = FusionConfig::default();
        config.enable_algebraic_reduction = true;
        config.enable_mega_fusion = true;

        let fused = fuse_program_with_config(&circuit, &config);

        let mut optimized_state = QState::new_zero(N_QUBITS);
        let mut exec_rng = QRng::new(0);
        execute_fused_cpu(&mut optimized_state, &fused, &mut exec_rng);

        // 3. Compare Results
        if !states_are_equal(&ground_truth_state, &optimized_state) {
            println!("FAILURE on Circuit #{}:", i);
            println!("Circuit: {:?}", circuit);
            println!("Fused Ops: {:?}", fused.ops);
            failures += 1;
        } else {
            if i % 10 == 0 {
                println!(
                    "Circuit #{} PASS (Reduced {} gates -> {} ops)",
                    i,
                    circuit.len(),
                    fused.ops.len()
                );
            }
        }
    }

    if failures > 0 {
        panic!(
            "Sanctity Torture Test FAILED: {}/{} circuits mismatch!",
            failures, NUM_TEST_CIRCUITS
        );
    } else {
        println!(
            "Sanctity Torture Test PASSED: {}/{} circuits verified.",
            NUM_TEST_CIRCUITS, NUM_TEST_CIRCUITS
        );
    }
}

fn generate_random_circuit(rng: &mut TestRng, n_qubits: u8, max_gates: usize) -> Vec<QGate> {
    let mut gates = Vec::new();
    let num_gates = (rng.next_u64() as usize % max_gates) + 1;

    for _ in 0..num_gates {
        let q = (rng.next_u64() % n_qubits as u64) as u8;
        let gate_type = rng.next_u64() % 7; // Changed to 7 to include some randomness

        match gate_type {
            0 => gates.push(QGate::H(q)),
            1 => gates.push(QGate::X(q)),
            2 => gates.push(QGate::Z(q)),
            3 => gates.push(QGate::Phase(q, std::f32::consts::PI / 4.0)),
            4 => {
                let t = (rng.next_u64() % n_qubits as u64) as u8;
                if t != q {
                    gates.push(QGate::CNot(q, t));
                } else {
                    gates.push(QGate::H(q)); // Fallback
                }
            }
            5 => {
                gates.push(QGate::Phase(q, std::f32::consts::PI / 2.0));
            }
            6 => {
                // Identity-like sequence to test reduction (H-H)
                gates.push(QGate::H(q));
                gates.push(QGate::H(q));
            }
            _ => {}
        }
    }
    gates
}

fn states_are_equal(s1: &QState, s2: &QState) -> bool {
    if s1.real.len() != s2.real.len() {
        return false;
    }

    let epsilon = 1e-4;

    for i in 0..s1.real.len() {
        let r1 = s1.real.as_slice()[i];
        let i1 = s1.imag.as_slice()[i];
        let r2 = s2.real.as_slice()[i];
        let i2 = s2.imag.as_slice()[i];

        if (r1 - r2).abs() > epsilon || (i1 - i2).abs() > epsilon {
            return false;
        }
    }
    true
}
