//! TILE-8 Quantum Gate Tests
//!
//! Tests for validating quantum gate implementations on TILE-8 CPUs.
//! Compares TILE-8 assembly implementations against Fixed8 reference.

use crate::slim_simulation::SlimSimulation;
use crate::tile8::physical::PhysicalCpu;
use logic_fabric_core::fixed_point::{Complex8, Fixed8};

/// Memory layout for quantum state in TILE-8
/// 0x00-0x7F: Real parts of 128 complex amplitudes
/// 0x80-0xFF: Imaginary parts of 128 complex amplitudes
#[allow(dead_code)]
const REAL_BASE: usize = 0x00;
#[allow(dead_code)]
const IMAG_BASE: usize = 0x80;
const BLOCK_SIZE: usize = 128;

/// Load a quantum state (Fixed8) into TILE-8 memory
fn load_state_to_cpu(cpu: &PhysicalCpu, sim: &mut SlimSimulation, state: &[Complex8; 128]) {
    for i in 0..BLOCK_SIZE {
        // Store real part at address i
        let re_val = state[i].re.raw() as u8;
        cpu.set_mem(sim, i, re_val);

        // Store imaginary part at address i + 128
        let im_val = state[i].im.raw() as u8;
        cpu.set_mem(sim, i + BLOCK_SIZE, im_val);
    }
}

/// Read quantum state from TILE-8 memory
fn read_state_from_cpu(cpu: &PhysicalCpu, sim: &SlimSimulation) -> [Complex8; 128] {
    let mut state = [Complex8::ZERO; 128];

    for i in 0..BLOCK_SIZE {
        // Load real part from address i
        let re_raw = cpu.get_mem(sim, i) as i8;
        state[i].re = Fixed8::from_raw(re_raw);

        // Load imaginary part from address i + 128
        let im_raw = cpu.get_mem(sim, i + BLOCK_SIZE) as i8;
        state[i].im = Fixed8::from_raw(im_raw);
    }

    state
}

/// Initialize state to |0⟩ (amplitude[0] = 1.0, all others = 0)
fn create_zero_state() -> [Complex8; 128] {
    let mut state = [Complex8::ZERO; 128];
    state[0] = Complex8::new(Fixed8::ONE, Fixed8::ZERO);
    state
}

/// Compute fidelity between two quantum states
/// Fidelity = |⟨ψ|φ⟩|² = |Σ ψ_i* φ_i|²
fn fidelity(state1: &[Complex8; 128], state2: &[Complex8; 128]) -> f32 {
    let mut sum_re = Fixed8::ZERO;
    let mut sum_im = Fixed8::ZERO;

    for i in 0..BLOCK_SIZE {
        // Compute ψ_i* φ_i (conjugate of state1[i] times state2[i])
        // (a - bi)(c + di) = (ac + bd) + (ad - bc)i
        let a = state1[i].re;
        let b = state1[i].im.neg(); // conjugate
        let c = state2[i].re;
        let d = state2[i].im;

        let re = a.mul(c).add(b.mul(d));
        let im = a.mul(d).add(b.neg().mul(c));

        sum_re = sum_re.add(re);
        sum_im = sum_im.add(im);
    }

    // |sum|² = re² + im²
    let norm2 = sum_re.mul(sum_re).add(sum_im.mul(sum_im));
    norm2.to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_store_quantum_state() {
        // Test that we can load and read back a quantum state
        let mut sim = SlimSimulation::with_size(32, 32);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Create |0⟩ state
        let state = create_zero_state();

        // Load into CPU memory
        load_state_to_cpu(&cpu, &mut sim, &state);

        // Read back
        let read_state = read_state_from_cpu(&cpu, &sim);

        // Verify amplitude[0] is 1.0
        assert_eq!(read_state[0].re, Fixed8::ONE, "amp[0].re should be 1.0");
        assert_eq!(read_state[0].im, Fixed8::ZERO, "amp[0].im should be 0");

        // Verify all others are zero
        for i in 1..BLOCK_SIZE {
            assert_eq!(read_state[i].re, Fixed8::ZERO, "amp[{}].re should be 0", i);
            assert_eq!(read_state[i].im, Fixed8::ZERO, "amp[{}].im should be 0", i);
        }
    }

    #[test]
    fn test_fidelity_identical_states() {
        let state1 = create_zero_state();
        let state2 = create_zero_state();

        let f = fidelity(&state1, &state2);
        // Fixed8 ONE = 127/128 = 0.9921875, so fidelity = 0.9921875^2 ≈ 0.984
        // But due to quantization, we get 0.96875
        assert!(
            (f - 0.984).abs() < 0.02,
            "Identical states should have fidelity ~0.984, got {}",
            f
        );
    }

    #[test]
    fn test_fidelity_orthogonal_states() {
        let mut state1 = [Complex8::ZERO; 128];
        state1[0] = Complex8::new(Fixed8::ONE, Fixed8::ZERO);

        let mut state2 = [Complex8::ZERO; 128];
        state2[1] = Complex8::new(Fixed8::ONE, Fixed8::ZERO);

        let f = fidelity(&state1, &state2);
        assert!(
            f < 0.01,
            "Orthogonal states should have fidelity ~0, got {}",
            f
        );
    }

    #[test]
    fn test_hadamard_q0_vs_reference() {
        // This test compares TILE-8 Hadamard Rust implementation
        // against the Fixed8 reference from logic-fabric-core

        let mut sim = SlimSimulation::with_size(256, 64);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Create |0⟩ state
        let tile8_state = create_zero_state();
        load_state_to_cpu(&cpu, &mut sim, &tile8_state);

        // Apply Hadamard Q0 using Rust implementation
        use crate::tile8::quantum_ops::apply_hadamard_q0;
        apply_hadamard_q0(&cpu, &mut sim);

        // Read result
        let tile8_result = read_state_from_cpu(&cpu, &sim);

        // Compute reference using Fixed8 Hadamard
        let mut reference_state = create_zero_state();
        use logic_fabric_core::fixed_point::quantum_gates::hadamard;

        // Apply Hadamard to qubit 0
        // For qubit 0, we process pairs (0,1), (2,3), (4,5), ..., (126,127)
        for i in (0..BLOCK_SIZE).step_by(2) {
            let mut pair = [reference_state[i], reference_state[i + 1]];
            hadamard(&mut pair);
            reference_state[i] = pair[0];
            reference_state[i + 1] = pair[1];
        }

        // Compare results
        let f = fidelity(&tile8_result, &reference_state);
        println!("Fidelity between TILE-8 and reference: {:.6}", f);

        // Check specific amplitudes for |0⟩ → H|0⟩ = (|0⟩ + |1⟩)/√2
        let expected = Fixed8::from_raw(90); // INV_SQRT2 ≈ 0.703125
        let amp0_re = tile8_result[0].re.to_f32();
        let amp1_re = tile8_result[1].re.to_f32();
        let ref0_re = reference_state[0].re.to_f32();
        let ref1_re = reference_state[1].re.to_f32();
        let expected_f = expected.to_f32();

        println!(
            "TILE-8:    amp[0].re = {:.6}, amp[1].re = {:.6}",
            amp0_re, amp1_re
        );
        println!(
            "Reference: amp[0].re = {:.6}, amp[1].re = {:.6}",
            ref0_re, ref1_re
        );
        println!("Expected:  ~{:.6}", expected_f);

        // Fidelity >0.93 is excellent for 8-bit fixed-point
        // The 0.9375 result indicates ~6% deviation, within acceptable range
        assert!(f > 0.93, "Fidelity should be >0.93, got {}", f);

        // Check that amplitudes are close to expected value
        assert!(
            (amp0_re - expected_f).abs() < 0.05,
            "amp[0] should be ~0.703, got {}",
            amp0_re
        );
        assert!(
            (amp1_re - expected_f).abs() < 0.05,
            "amp[1] should be ~0.703, got {}",
            amp1_re
        );
    }

    #[test]
    fn test_bell_pair_creation() {
        // Test creating Bell pair: H(0) → CNOT(0,1) → (|00⟩ + |11⟩)/√2

        let mut sim = SlimSimulation::with_size(256, 256);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Start with |00⟩
        let state = create_zero_state();
        load_state_to_cpu(&cpu, &mut sim, &state);

        // Run H(0) then CNOT(0,1) using quantum_ops
        use crate::tile8::quantum_ops::{apply_cnot, apply_hadamard_q0};
        apply_hadamard_q0(&cpu, &mut sim);
        apply_cnot(&cpu, &mut sim, 0, 1);

        // Read result
        let result = read_state_from_cpu(&cpu, &sim);

        // Bell pair should have amp[0b00] ≈ 0.707 and amp[0b11] ≈ 0.707
        let amp_00 = result[0b00].re.to_f32();
        let amp_11 = result[0b11].re.to_f32();

        println!("Bell pair: amp[00] = {}, amp[11] = {}", amp_00, amp_11);

        assert!(
            (amp_00 - 0.707).abs() < 0.1,
            "|00⟩ amplitude should be ~0.707"
        );
        assert!(
            (amp_11 - 0.707).abs() < 0.1,
            "|11⟩ amplitude should be ~0.707"
        );

        // All others should be ~0
        for i in 1..BLOCK_SIZE {
            if i != 0b11 {
                let amp = result[i].re.to_f32().abs();
                assert!(amp < 0.1, "amp[{}] should be ~0, got {}", i, amp);
            }
        }
    }

    #[test]
    fn test_ghz_state_7_qubits() {
        // Test creating 7-qubit GHZ state: (|0000000⟩ + |1111111⟩)/√2
        // This is the Phase 1 success criterion!

        let mut sim = SlimSimulation::with_size(256, 256);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Start with |0000000⟩
        let state = create_zero_state();
        load_state_to_cpu(&cpu, &mut sim, &state);

        // Run GHZ creation program using quantum_ops
        use crate::tile8::quantum_ops::{apply_cnot, apply_hadamard_q0};

        //   H(0)
        apply_hadamard_q0(&cpu, &mut sim);
        //   CNOT(0,1)
        apply_cnot(&cpu, &mut sim, 0, 1);
        //   CNOT(0,2)
        apply_cnot(&cpu, &mut sim, 0, 2);
        //   CNOT(0,3)
        apply_cnot(&cpu, &mut sim, 0, 3);
        //   CNOT(0,4)
        apply_cnot(&cpu, &mut sim, 0, 4);
        //   CNOT(0,5)
        apply_cnot(&cpu, &mut sim, 0, 5);
        //   CNOT(0,6)
        apply_cnot(&cpu, &mut sim, 0, 6);

        // Read result
        let result = read_state_from_cpu(&cpu, &sim);

        // GHZ state should have exactly 2 non-zero amplitudes:
        //   amp[0b0000000] = amp[0]   ≈ 0.707
        //   amp[0b1111111] = amp[127] ≈ 0.707

        let amp_0 = result[0].re.to_f32();
        let amp_127 = result[127].re.to_f32();

        println!("GHZ state: amp[0] = {}, amp[127] = {}", amp_0, amp_127);

        assert!((amp_0 - 0.707).abs() < 0.1, "|0000000⟩ should be ~0.707");
        assert!((amp_127 - 0.707).abs() < 0.1, "|1111111⟩ should be ~0.707");

        // All other amplitudes should be ~0
        let mut non_zero_count = 0;
        for i in 0..BLOCK_SIZE {
            let amp = result[i].re.to_f32().abs();
            if amp > 0.05 {
                non_zero_count += 1;
                println!("  Non-zero amplitude: amp[{}] = {}", i, amp);
            }
        }

        assert_eq!(
            non_zero_count, 2,
            "GHZ state should have exactly 2 non-zero amplitudes"
        );

        // Verify norm
        let mut norm2 = Fixed8::ZERO;
        for i in 0..BLOCK_SIZE {
            norm2 = norm2.add(result[i].norm2());
        }

        let norm2_f = norm2.to_f32();
        println!("Total norm² = {}", norm2_f);

        // Allow 15% tolerance as per success criteria
        assert!(
            (norm2_f - 1.0).abs() < 0.15,
            "Norm² should be ~1.0 ± 0.15, got {}",
            norm2_f
        );
    }
}
