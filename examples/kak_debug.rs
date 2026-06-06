//! Debug test for KAK synthesis

use logic_fabric_core::algebraic_fusion::{
    TwoQubitBlock, analyze_two_qubit_block, compute_block_unitary, synthesize_two_qubit_block,
    unitary_fidelity,
};
use logic_fabric_core::quantum::QGate;

fn main() {
    // Create SWAP matrix
    let mut swap = [[num_complex::Complex32::new(0.0, 0.0); 4]; 4];
    swap[0][0] = num_complex::Complex32::new(1.0, 0.0);
    swap[1][2] = num_complex::Complex32::new(1.0, 0.0);
    swap[2][1] = num_complex::Complex32::new(1.0, 0.0);
    swap[3][3] = num_complex::Complex32::new(1.0, 0.0);

    println!("=== KAK Synthesis Debug ===\n");

    // Test SWAP
    let swap_gates = vec![QGate::CNot(0, 1), QGate::CNot(1, 0), QGate::CNot(0, 1)];

    let block = TwoQubitBlock {
        gates: swap_gates.clone(),
        qubit_a: 0,
        qubit_b: 1,
        start_idx: 0,
        end_idx: 3,
    };

    let swap_unitary = compute_block_unitary(&block);
    println!("SWAP unitary from CNOT-CNOT-CNOT:");
    for i in 0..4 {
        println!(
            "  [{:.2}+{:.2}i, {:.2}+{:.2}i, {:.2}+{:.2}i, {:.2}+{:.2}i]",
            swap_unitary.data[i][0].re,
            swap_unitary.data[i][0].im,
            swap_unitary.data[i][1].re,
            swap_unitary.data[i][1].im,
            swap_unitary.data[i][2].re,
            swap_unitary.data[i][2].im,
            swap_unitary.data[i][3].re,
            swap_unitary.data[i][3].im
        );
    }

    let kak = analyze_two_qubit_block(&swap_unitary);
    println!("\nKAK analysis:");
    println!(
        "  alpha={:.4}, beta={:.4}, gamma={:.4}",
        kak.alpha, kak.beta, kak.gamma
    );
    println!("  cnot_count={}", kak.cnot_count);
    println!(
        "  A1: [{:.4}, {:.4}, {:.4}]",
        kak.a1[0], kak.a1[1], kak.a1[2]
    );
    println!(
        "  A2: [{:.4}, {:.4}, {:.4}]",
        kak.a2[0], kak.a2[1], kak.a2[2]
    );
    println!(
        "  B1: [{:.4}, {:.4}, {:.4}]",
        kak.b1[0], kak.b1[1], kak.b1[2]
    );
    println!(
        "  B2: [{:.4}, {:.4}, {:.4}]",
        kak.b2[0], kak.b2[1], kak.b2[2]
    );

    let synthesized = synthesize_two_qubit_block(&swap_unitary, &kak, 0, 1);
    println!("\nSynthesized circuit ({} gates):", synthesized.len());
    for g in &synthesized {
        println!("  {:?}", g);
    }

    let synth_block = TwoQubitBlock {
        gates: synthesized.clone(),
        qubit_a: 0,
        qubit_b: 1,
        start_idx: 0,
        end_idx: synthesized.len(),
    };

    let result = compute_block_unitary(&synth_block);
    println!("\nSynthesized unitary:");
    for i in 0..4 {
        println!(
            "  [{:.2}+{:.2}i, {:.2}+{:.2}i, {:.2}+{:.2}i, {:.2}+{:.2}i]",
            result.data[i][0].re,
            result.data[i][0].im,
            result.data[i][1].re,
            result.data[i][1].im,
            result.data[i][2].re,
            result.data[i][2].im,
            result.data[i][3].re,
            result.data[i][3].im
        );
    }

    let fidelity = unitary_fidelity(&swap_unitary, &result);
    println!("\nFidelity: {:.6}", fidelity);
}
