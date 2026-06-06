//! EPIC 89: CNOT WMMA Test
//!
//! Tests Bell state creation via WMMA batched gates:
//! H(0) → CNOT(0,1) should produce (|00⟩+|11⟩)/√2
//!
//! Run with: cargo run --example epic89_cnot_wmma_test --features cuda --release

#[cfg(feature = "cuda")]
use engine::cuda::{CudaRuntime, WmmaState, is_cuda_available, run_wmma_batched_gates};
#[cfg(feature = "cuda")]
use engine::quantum::QGate;

#[cfg(feature = "cuda")]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              EPIC 89: CNOT WMMA Bell State Test                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    if !is_cuda_available() {
        println!("❌ CUDA not available - skipping test");
        return;
    }

    let rt = CudaRuntime::new().expect("Failed to create CUDA runtime");

    // Create |0000⟩ state for 4 qubits (16 amplitudes for WMMA tile)
    // WMMA format: 16 real amplitudes, padded to 256 for alignment
    let mut init_data = vec![half::f16::ZERO; 256];
    init_data[0] = half::f16::ONE; // amplitude[0] = 1.0 for |0000⟩

    let mut wmma_state = WmmaState::from_host(&rt, &init_data).expect("WmmaState creation");

    println!("Initial state: |0000⟩");
    println!("Applying: H(0) → CNOT(0,1)");
    println!();

    // Apply H(0) → CNOT(0,1) sequence
    let gates = vec![QGate::H(0), QGate::CNot(0, 1)];

    // Execute via WMMA batched gates
    let ops =
        run_wmma_batched_gates(&rt, &mut wmma_state, &gates).expect("WMMA batched gates failed");

    println!("Gates executed: {} operations", ops);

    // Download result
    let result_bits = rt.download(&wmma_state.data).expect("Download result");
    let result: Vec<half::f16> = result_bits
        .iter()
        .map(|b| half::f16::from_bits(*b))
        .collect();

    // WMMA format: 16 real amplitudes (no interleaving, real-only)
    // Extract amplitudes directly
    let amplitudes: Vec<f32> = (0..16).map(|i| result[i].to_f32()).collect();

    // Bell state |00⟩+|11⟩: expect non-zero at indices 0 (|0000⟩) and 3 (|0011⟩)
    let sqrt2_inv = 1.0 / 2.0f32.sqrt();
    let tolerance = 0.02; // FP16 has limited precision

    println!("\n## Results:");
    println!();
    println!("| Index | Binary | Amplitude | Expected |");
    println!("|-------|--------|-----------|----------|");
    for i in 0..16 {
        let expected = if i == 0 || i == 3 {
            format!("{:.4}", sqrt2_inv)
        } else {
            "0".to_string()
        };
        let mark = if i == 0 || i == 3 { " ←" } else { "" };
        println!(
            "| {:>5} | {:04b} | {:>9.4} | {:>8} |{}",
            i, i, amplitudes[i], expected, mark
        );
    }

    // Verify Bell state amplitudes
    println!("\n## Verification:");
    let mut passed = true;

    if (amplitudes[0] - sqrt2_inv).abs() < tolerance {
        println!(
            "✅ amplitude[0] = {:.4} ≈ 1/√2 = {:.4}",
            amplitudes[0], sqrt2_inv
        );
    } else {
        println!(
            "❌ amplitude[0] = {:.4}, expected ≈ {:.4}",
            amplitudes[0], sqrt2_inv
        );
        passed = false;
    }

    if (amplitudes[3] - sqrt2_inv).abs() < tolerance {
        println!(
            "✅ amplitude[3] = {:.4} ≈ 1/√2 = {:.4}",
            amplitudes[3], sqrt2_inv
        );
    } else {
        println!(
            "❌ amplitude[3] = {:.4}, expected ≈ {:.4}",
            amplitudes[3], sqrt2_inv
        );
        passed = false;
    }

    // Check other amplitudes are zero
    for i in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
        if amplitudes[i].abs() > tolerance {
            println!("❌ amplitude[{}] = {:.4} should be ≈ 0", i, amplitudes[i]);
            passed = false;
        }
    }

    if passed {
        println!();
        println!(
            "╔══════════════════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║                    ✅ EPIC 89 CNOT WMMA Test PASSED!                         ║"
        );
        println!(
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        );
    } else {
        println!();
        println!(
            "╔══════════════════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║                    ❌ EPIC 89 CNOT WMMA Test FAILED!                         ║"
        );
        println!(
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        );
        std::process::exit(1);
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Error: This example requires the 'cuda' feature.");
    eprintln!("Run with: cargo run --example epic89_cnot_wmma_test --features cuda --release");
    std::process::exit(1);
}
