/// Debug test for Toffoli synthesis
use logic_fabric_core::algebraic_fusion::{ThreeQubitGate, synthesize_three_qubit};
use logic_fabric_core::quantum::{QGate, QRng, QState, apply_gate_scalar};

fn main() {
    println!("=== Toffoli Debug Test ===\n");

    // Create |000⟩ state
    let mut state = QState::new_zero(3);
    println!("Initial state |000⟩:");
    for i in 0..8 {
        let re = state.real.as_slice()[i];
        let im = state.imag.as_slice()[i];
        if re.abs() > 1e-6 || im.abs() > 1e-6 {
            println!("  state[{}] = {:.6} + {:.6}i", i, re, im);
        }
    }

    // Apply a single X gate to qubit 0
    let mut rng = QRng::new(0);
    println!("\nApplying X(0)...");
    apply_gate_scalar(&mut state, &QGate::X(0), &mut rng);

    println!("State after X(0):");
    for i in 0..8 {
        let re = state.real.as_slice()[i];
        let im = state.imag.as_slice()[i];
        if re.abs() > 1e-6 || im.abs() > 1e-6 {
            println!("  state[{}] = {:.6} + {:.6}i", i, re, im);
        }
    }

    // Synthesize Toffoli
    println!("\nSynthesizing Toffoli...");
    let toffoli_gates = synthesize_three_qubit(ThreeQubitGate::Toffoli, 0, 1, 2)
        .expect("Failed to synthesize Toffoli");

    println!("Synthesized {} gates:", toffoli_gates.len());
    for (i, gate) in toffoli_gates.iter().enumerate() {
        println!("  {}: {:?}", i + 1, gate);
    }

    // Apply Toffoli to |100⟩
    println!("\nApplying Toffoli to current state |100⟩...");
    for gate in &toffoli_gates {
        apply_gate_scalar(&mut state, gate, &mut rng);
    }

    println!("Final state:");
    for i in 0..8 {
        let re = state.real.as_slice()[i];
        let im = state.imag.as_slice()[i];
        if re.abs() > 1e-6 || im.abs() > 1e-6 {
            println!("  state[{}] = {:.6} + {:.6}i (|{:03b}⟩)", i, re, im, i);
        }
    }

    // Test |110⟩ → |111⟩
    println!("\n\n=== Test |110⟩ → |111⟩ ===");
    let mut state2 = QState::new_zero(3);
    let mut rng = QRng::new(0);

    apply_gate_scalar(&mut state2, &QGate::X(0), &mut rng);
    apply_gate_scalar(&mut state2, &QGate::X(1), &mut rng);

    println!("Initial state |110⟩:");
    for i in 0..8 {
        let re = state2.real.as_slice()[i];
        let im = state2.imag.as_slice()[i];
        if re.abs() > 1e-6 || im.abs() > 1e-6 {
            println!("  state[{}] = {:.6} + {:.6}i (|{:03b}⟩)", i, re, im, i);
        }
    }

    println!("\nApplying Toffoli...");
    for gate in &toffoli_gates {
        apply_gate_scalar(&mut state2, gate, &mut rng);
    }

    println!("Final state (should be |111⟩):");
    for i in 0..8 {
        let re = state2.real.as_slice()[i];
        let im = state2.imag.as_slice()[i];
        if re.abs() > 1e-6 || im.abs() > 1e-6 {
            println!("  state[{}] = {:.6} + {:.6}i (|{:03b}⟩)", i, re, im, i);
        }
    }
}
