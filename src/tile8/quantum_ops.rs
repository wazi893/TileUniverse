//! Quantum Operations for TILE-8 CPUs
//!
//! Direct Rust implementations of quantum gates operating on TILE-8 memory.
//! These bypass assembly size limitations while using the same Fixed8 arithmetic.

use crate::slim_simulation::SlimSimulation;
use crate::tile8::physical::PhysicalCpu;
use logic_fabric_core::fixed_point::Fixed8;

/// Memory layout constants
pub const BLOCK_SIZE: usize = 128;
pub const REAL_BASE: usize = 0;
pub const IMAG_BASE: usize = 128;

/// Multiply a Fixed8 value by INV_SQRT2 = 90/128
///
/// This implements the same shift-add algorithm as the assembly version:
/// result = (x*64 + x*16 + x*8 + x*2) / 128
#[inline]
fn mul_inv_sqrt2(x: i8) -> i8 {
    // Convert to i32 for intermediate calculations to avoid overflow
    let x32 = x as i32;

    // Compute x * 90 using shift-add
    let x64 = x32 << 6; // x * 64
    let x16 = x32 << 4; // x * 16
    let x8 = x32 << 3; // x * 8
    let x2 = x32 << 1; // x * 2

    let sum = x64 + x16 + x8 + x2; // x * 90

    // Divide by 128 (arithmetic right shift to preserve sign)
    let result = sum >> 7;

    // Clamp to i8 range
    result.clamp(-128, 127) as i8
}

/// Apply Hadamard gate to qubit 0
///
/// Processes all 128 amplitudes in pairs (i, i+1) where i is even.
/// For each pair:
///   new_amp[i] = (amp[i] + amp[i+1]) * INV_SQRT2
///   new_amp[i+1] = (amp[i] - amp[i+1]) * INV_SQRT2
///
/// Memory layout:
///   Addresses 0-127: Real parts
///   Addresses 128-255: Imaginary parts
pub fn apply_hadamard_q0(cpu: &PhysicalCpu, sim: &mut SlimSimulation) {
    // Process real parts (addresses 0-127)
    for pair_idx in 0..64 {
        let i = pair_idx * 2;
        let j = i + 1;

        // Load original values
        let amp_i_re = cpu.get_mem(sim, i) as i8;
        let amp_j_re = cpu.get_mem(sim, j) as i8;

        // Compute sum and diff
        let sum = amp_i_re.wrapping_add(amp_j_re);
        let diff = amp_i_re.wrapping_sub(amp_j_re);

        // Multiply by INV_SQRT2 and store
        let new_i = mul_inv_sqrt2(sum);
        let new_j = mul_inv_sqrt2(diff);

        cpu.set_mem(sim, i, new_i as u8);
        cpu.set_mem(sim, j, new_j as u8);
    }

    // Process imaginary parts (addresses 128-255)
    for pair_idx in 0..64 {
        let i = IMAG_BASE + pair_idx * 2;
        let j = i + 1;

        // Load original values
        let amp_i_im = cpu.get_mem(sim, i) as i8;
        let amp_j_im = cpu.get_mem(sim, j) as i8;

        // Compute sum and diff
        let sum = amp_i_im.wrapping_add(amp_j_im);
        let diff = amp_i_im.wrapping_sub(amp_j_im);

        // Multiply by INV_SQRT2 and store
        let new_i = mul_inv_sqrt2(sum);
        let new_j = mul_inv_sqrt2(diff);

        cpu.set_mem(sim, i, new_i as u8);
        cpu.set_mem(sim, j, new_j as u8);
    }
}

/// Apply Pauli-X gate to qubit (bit flip)
///
/// For qubit q, swaps amplitudes where bit q differs.
/// Only supports qubits 0-6 (local to this CPU's 128-amplitude block).
pub fn apply_pauli_x(cpu: &PhysicalCpu, sim: &mut SlimSimulation, qubit: u8) {
    assert!(qubit < 7, "Qubit must be 0-6 for local operations");

    let mask = 1u8 << qubit;

    // Process real parts
    for state in 0..BLOCK_SIZE {
        // Only process each pair once (when the qubit bit is 0)
        if (state as u8) & mask == 0 {
            let partner = state ^ (mask as usize);

            let amp_state_re = cpu.get_mem(sim, state);
            let amp_partner_re = cpu.get_mem(sim, partner);

            cpu.set_mem(sim, state, amp_partner_re);
            cpu.set_mem(sim, partner, amp_state_re);
        }
    }

    // Process imaginary parts
    for state in 0..BLOCK_SIZE {
        if (state as u8) & mask == 0 {
            let partner = state ^ (mask as usize);

            let state_addr = IMAG_BASE + state;
            let partner_addr = IMAG_BASE + partner;

            let amp_state_im = cpu.get_mem(sim, state_addr);
            let amp_partner_im = cpu.get_mem(sim, partner_addr);

            cpu.set_mem(sim, state_addr, amp_partner_im);
            cpu.set_mem(sim, partner_addr, amp_state_im);
        }
    }
}

/// Apply Pauli-Z gate to qubit (phase flip)
///
/// For qubit q, negates amplitudes where bit q is set.
/// Only supports qubits 0-6 (local to this CPU's 128-amplitude block).
pub fn apply_pauli_z(cpu: &PhysicalCpu, sim: &mut SlimSimulation, qubit: u8) {
    assert!(qubit < 7, "Qubit must be 0-6 for local operations");

    let mask = 1u8 << qubit;

    // Process real parts
    for state in 0..BLOCK_SIZE {
        if (state as u8) & mask != 0 {
            let val = cpu.get_mem(sim, state) as i8;
            cpu.set_mem(sim, state, val.wrapping_neg() as u8);
        }
    }

    // Process imaginary parts
    for state in 0..BLOCK_SIZE {
        if (state as u8) & mask != 0 {
            let addr = IMAG_BASE + state;
            let val = cpu.get_mem(sim, addr) as i8;
            cpu.set_mem(sim, addr, val.wrapping_neg() as u8);
        }
    }
}

/// Apply CNOT gate (controlled-NOT)
///
/// Flips target qubit when control qubit is |1⟩.
/// For control=c, target=t:
///   If bit c is set in state index, swap amplitudes at state and state^(1<<t)
///
/// Only supports qubits 0-6 (local to this CPU's 128-amplitude block).
pub fn apply_cnot(cpu: &PhysicalCpu, sim: &mut SlimSimulation, control: u8, target: u8) {
    assert!(
        control < 7,
        "Control qubit must be 0-6 for local operations"
    );
    assert!(target < 7, "Target qubit must be 0-6 for local operations");
    assert_ne!(
        control, target,
        "Control and target must be different qubits"
    );

    let control_mask = 1u8 << control;
    let target_mask = 1u8 << target;

    // Process real parts
    for state in 0..BLOCK_SIZE {
        let state_bits = state as u8;

        // Only process if control bit is set AND we haven't swapped this pair yet
        if (state_bits & control_mask != 0) && (state_bits & target_mask == 0) {
            let partner = state ^ (target_mask as usize);

            let amp_state_re = cpu.get_mem(sim, state);
            let amp_partner_re = cpu.get_mem(sim, partner);

            cpu.set_mem(sim, state, amp_partner_re);
            cpu.set_mem(sim, partner, amp_state_re);
        }
    }

    // Process imaginary parts
    for state in 0..BLOCK_SIZE {
        let state_bits = state as u8;

        if (state_bits & control_mask != 0) && (state_bits & target_mask == 0) {
            let partner = state ^ (target_mask as usize);

            let state_addr = IMAG_BASE + state;
            let partner_addr = IMAG_BASE + partner;

            let amp_state_im = cpu.get_mem(sim, state_addr);
            let amp_partner_im = cpu.get_mem(sim, partner_addr);

            cpu.set_mem(sim, state_addr, amp_partner_im);
            cpu.set_mem(sim, partner_addr, amp_state_im);
        }
    }
}

/// Apply Phase gate (rotation around Z-axis by angle θ)
///
/// For qubit q, multiplies amplitudes where bit q is set by e^(iθ).
/// This is equivalent to applying the matrix [[1, 0], [0, e^(iθ)]].
///
/// Complex multiplication: (a + bi) * (cos θ + i sin θ) = (a·cos θ - b·sin θ) + i(a·sin θ + b·cos θ)
///
/// Only supports qubits 0-6 (local to this CPU's 128-amplitude block).
///
/// # Arguments
/// * `cpu` - The CPU to operate on
/// * `sim` - The simulation containing the CPU
/// * `qubit` - The qubit to apply the phase to (0-6)
/// * `angle` - The phase angle in radians
pub fn apply_phase(cpu: &PhysicalCpu, sim: &mut SlimSimulation, qubit: u8, angle: f64) {
    assert!(qubit < 7, "Qubit must be 0-6 for local operations");

    let mask = 1u8 << qubit;

    // Compute cos and sin in Fixed8 format
    let cos_theta = Fixed8::from_f32(angle.cos() as f32);
    let sin_theta = Fixed8::from_f32(angle.sin() as f32);

    // Process all amplitudes where the qubit bit is set
    for state in 0..BLOCK_SIZE {
        if (state as u8) & mask != 0 {
            // Load current amplitude (re, im)
            let re = cpu.get_mem(sim, state) as i8;
            let im = cpu.get_mem(sim, IMAG_BASE + state) as i8;

            // Convert to Fixed8
            let re_fixed = Fixed8::from_raw(re);
            let im_fixed = Fixed8::from_raw(im);

            // Complex multiplication: (re + i*im) * (cos + i*sin)
            // new_re = re * cos - im * sin
            // new_im = re * sin + im * cos
            let new_re = re_fixed * cos_theta - im_fixed * sin_theta;
            let new_im = re_fixed * sin_theta + im_fixed * cos_theta;

            // Store the results
            cpu.set_mem(sim, state, new_re.raw() as u8);
            cpu.set_mem(sim, IMAG_BASE + state, new_im.raw() as u8);
        }
    }
}

/// Apply S gate (π/2 phase gate, also called √Z)
///
/// This is a special case of the phase gate with angle = π/2.
/// S = [[1, 0], [0, i]]
///
/// Only supports qubits 0-6 (local to this CPU's 128-amplitude block).
pub fn apply_s_gate(cpu: &PhysicalCpu, sim: &mut SlimSimulation, qubit: u8) {
    assert!(qubit < 7, "Qubit must be 0-6 for local operations");

    let mask = 1u8 << qubit;

    // For S gate: cos(π/2) = 0, sin(π/2) = 1
    // So (re + i*im) * i = -im + i*re

    for state in 0..BLOCK_SIZE {
        if (state as u8) & mask != 0 {
            let re = cpu.get_mem(sim, state) as i8;
            let im = cpu.get_mem(sim, IMAG_BASE + state) as i8;

            // Multiply by i: (re + i*im) * i = -im + i*re
            let new_re = -im;
            let new_im = re;

            cpu.set_mem(sim, state, new_re as u8);
            cpu.set_mem(sim, IMAG_BASE + state, new_im as u8);
        }
    }
}

/// Apply T gate (π/4 phase gate)
///
/// This is a special case of the phase gate with angle = π/4.
/// T = [[1, 0], [0, e^(iπ/4)]]
///
/// Only supports qubits 0-6 (local to this CPU's 128-amplitude block).
pub fn apply_t_gate(cpu: &PhysicalCpu, sim: &mut SlimSimulation, qubit: u8) {
    // T gate: angle = π/4
    apply_phase(cpu, sim, qubit, std::f64::consts::FRAC_PI_4);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slim_simulation::SlimSimulation;
    use crate::tile8::physical::PhysicalCpu;

    #[test]
    fn test_mul_inv_sqrt2() {
        // Test the multiplication function

        // For x = 127 (≈1.0), result should be ≈ 0.707 * 127 ≈ 89.8 → 90
        let result = mul_inv_sqrt2(127);
        assert!(
            (result - 90).abs() <= 1,
            "127 * INV_SQRT2 should be ~90, got {}",
            result
        );

        // For x = 64 (0.5), result should be ≈ 0.707 * 64 ≈ 45.2 → 45
        let result = mul_inv_sqrt2(64);
        assert!(
            (result - 45).abs() <= 1,
            "64 * INV_SQRT2 should be ~45, got {}",
            result
        );

        // For x = 0, result should be 0
        assert_eq!(mul_inv_sqrt2(0), 0);

        // For negative values
        let result = mul_inv_sqrt2(-127);
        assert!(
            (result + 90).abs() <= 1,
            "-127 * INV_SQRT2 should be ~-90, got {}",
            result
        );
    }

    #[test]
    fn test_hadamard_on_zero_state() {
        // Apply H to |0⟩ should give (|0⟩ + |1⟩)/√2
        let mut sim = SlimSimulation::with_size(256, 64);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Initialize |0⟩ state: amp[0] = 127 (≈1.0), all others = 0
        cpu.set_mem(&mut sim, 0, 127); // amp[0].re = 1.0
        for i in 1..256 {
            cpu.set_mem(&mut sim, i, 0);
        }

        // Apply Hadamard
        apply_hadamard_q0(&cpu, &mut sim);

        // Check results
        let amp0_re = cpu.get_mem(&sim, 0) as i8;
        let amp1_re = cpu.get_mem(&sim, 1) as i8;

        println!(
            "After H|0⟩: amp[0].re = {}, amp[1].re = {}",
            amp0_re, amp1_re
        );

        // Both should be approximately 90 (≈0.707)
        assert!(
            (amp0_re - 90).abs() <= 2,
            "amp[0] should be ~90, got {}",
            amp0_re
        );
        assert!(
            (amp1_re - 90).abs() <= 2,
            "amp[1] should be ~90, got {}",
            amp1_re
        );

        // All others should still be 0
        for i in 2..128 {
            let val = cpu.get_mem(&sim, i);
            assert_eq!(val, 0, "amp[{}] should be 0", i);
        }
    }

    #[test]
    fn test_pauli_x() {
        let mut sim = SlimSimulation::with_size(256, 64);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Initialize |0⟩ state
        cpu.set_mem(&mut sim, 0, 127); // amp[0].re = 1.0

        // Apply X gate to qubit 0
        apply_pauli_x(&cpu, &mut sim, 0);

        // Should now be |1⟩: amp[1] = 1.0, amp[0] = 0
        assert_eq!(cpu.get_mem(&sim, 0), 0, "amp[0] should be 0 after X");
        assert_eq!(cpu.get_mem(&sim, 1), 127, "amp[1] should be 127 after X");
    }

    #[test]
    fn test_pauli_z() {
        let mut sim = SlimSimulation::with_size(256, 64);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Initialize |1⟩ state
        cpu.set_mem(&mut sim, 1, 127); // amp[1].re = 1.0

        // Apply Z gate to qubit 0
        apply_pauli_z(&cpu, &mut sim, 0);

        // Should now be -|1⟩: amp[1] = -127
        let amp1 = cpu.get_mem(&sim, 1) as i8;
        assert_eq!(amp1, -127, "amp[1] should be -127 after Z");
    }

    #[test]
    fn test_cnot_creates_bell_pair() {
        // Test CNOT(0,1) on (|0⟩ + |1⟩)/√2 ⊗ |0⟩ = (|00⟩ + |11⟩)/√2
        let mut sim = SlimSimulation::with_size(256, 64);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Initialize |0⟩ state
        cpu.set_mem(&mut sim, 0, 127); // amp[0].re = 1.0

        // Create (|0⟩ + |1⟩)/√2 on qubit 0
        apply_hadamard_q0(&cpu, &mut sim);

        // Apply CNOT(0, 1)
        apply_cnot(&cpu, &mut sim, 0, 1);

        // Check Bell pair: amp[0b00] and amp[0b11] should be equal (~90)
        let amp_00 = cpu.get_mem(&sim, 0b00) as i8;
        let amp_11 = cpu.get_mem(&sim, 0b11) as i8;

        println!("Bell pair: amp[00] = {}, amp[11] = {}", amp_00, amp_11);

        assert!(
            (amp_00 - 90).abs() <= 2,
            "amp[00] should be ~90, got {}",
            amp_00
        );
        assert!(
            (amp_11 - 90).abs() <= 2,
            "amp[11] should be ~90, got {}",
            amp_11
        );

        // Check that other amplitudes are ~0
        assert_eq!(cpu.get_mem(&sim, 0b01), 0, "amp[01] should be 0");
        assert_eq!(cpu.get_mem(&sim, 0b10), 0, "amp[10] should be 0");
    }

    #[test]
    fn test_cnot_on_zero_state() {
        // CNOT on |00⟩ should remain |00⟩
        let mut sim = SlimSimulation::with_size(256, 64);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Initialize |00⟩
        cpu.set_mem(&mut sim, 0, 127);

        // Apply CNOT(0, 1)
        apply_cnot(&cpu, &mut sim, 0, 1);

        // Should still be |00⟩
        assert_eq!(cpu.get_mem(&sim, 0b00), 127, "amp[00] should be 127");
        assert_eq!(cpu.get_mem(&sim, 0b01), 0, "amp[01] should be 0");
        assert_eq!(cpu.get_mem(&sim, 0b10), 0, "amp[10] should be 0");
        assert_eq!(cpu.get_mem(&sim, 0b11), 0, "amp[11] should be 0");
    }

    #[test]
    fn test_s_gate() {
        // Test S gate: S|1⟩ = i|1⟩
        let mut sim = SlimSimulation::with_size(256, 64);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Initialize |1⟩ state
        cpu.set_mem(&mut sim, 1, 127); // amp[1].re = 1.0

        // Apply S gate to qubit 0
        apply_s_gate(&cpu, &mut sim, 0);

        // S|1⟩ = i|1⟩ means: re = 0, im = 1.0
        let re = cpu.get_mem(&sim, 1) as i8;
        let im = cpu.get_mem(&sim, IMAG_BASE + 1) as i8;

        println!("After S|1⟩: re = {}, im = {}", re, im);

        assert_eq!(re, 0, "Real part should be 0 after S gate");
        assert_eq!(im, 127, "Imaginary part should be 127 after S gate");
    }

    #[test]
    fn test_s_gate_twice_is_z() {
        // Test that S² = Z (two S gates equal one Z gate)
        let mut sim = SlimSimulation::with_size(256, 64);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Initialize |1⟩ state
        cpu.set_mem(&mut sim, 1, 127);

        // Apply S gate twice
        apply_s_gate(&cpu, &mut sim, 0);
        apply_s_gate(&cpu, &mut sim, 0);

        // S²|1⟩ = i²|1⟩ = -|1⟩ (same as Z|1⟩)
        let re = cpu.get_mem(&sim, 1) as i8;
        let im = cpu.get_mem(&sim, IMAG_BASE + 1) as i8;

        println!("After S²|1⟩: re = {}, im = {}", re, im);

        assert_eq!(re, -127, "Real part should be -127 (S² = Z)");
        assert_eq!(im, 0, "Imaginary part should be 0");
    }

    #[test]
    fn test_t_gate() {
        // Test T gate: T|1⟩ = e^(iπ/4)|1⟩ = (1/√2 + i/√2)|1⟩
        let mut sim = SlimSimulation::with_size(256, 64);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Initialize |1⟩ state
        cpu.set_mem(&mut sim, 1, 127);

        // Apply T gate to qubit 0
        apply_t_gate(&cpu, &mut sim, 0);

        // T|1⟩ = e^(iπ/4)|1⟩
        // cos(π/4) = sin(π/4) = 1/√2 ≈ 0.707
        let re = cpu.get_mem(&sim, 1) as i8;
        let im = cpu.get_mem(&sim, IMAG_BASE + 1) as i8;

        println!("After T|1⟩: re = {}, im = {}", re, im);

        // Both should be approximately 90 (≈0.707 in Fixed8)
        assert!((re - 90).abs() <= 2, "Real part should be ~90, got {}", re);
        assert!(
            (im - 90).abs() <= 2,
            "Imaginary part should be ~90, got {}",
            im
        );
    }

    #[test]
    fn test_phase_gate_arbitrary_angle() {
        // Test phase gate with π/3 angle
        let mut sim = SlimSimulation::with_size(256, 64);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Initialize |1⟩ state
        cpu.set_mem(&mut sim, 1, 127);

        // Apply phase gate with angle π/3
        let angle = std::f64::consts::FRAC_PI_3;
        apply_phase(&cpu, &mut sim, 0, angle);

        // e^(iπ/3) = cos(π/3) + i*sin(π/3) = 0.5 + i*0.866
        let re = cpu.get_mem(&sim, 1) as i8;
        let im = cpu.get_mem(&sim, IMAG_BASE + 1) as i8;

        println!("After Phase(π/3)|1⟩: re = {}, im = {}", re, im);

        // cos(π/3) ≈ 0.5 → ~64 in Fixed8
        // sin(π/3) ≈ 0.866 → ~110 in Fixed8
        assert!(
            (re - 64).abs() <= 3,
            "Real part should be ~64 (0.5), got {}",
            re
        );
        assert!(
            (im - 110).abs() <= 3,
            "Imaginary part should be ~110 (0.866), got {}",
            im
        );
    }

    #[test]
    fn test_phase_gate_zero_angle() {
        // Test that phase(0) is identity
        let mut sim = SlimSimulation::with_size(256, 64);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Initialize |1⟩ state
        cpu.set_mem(&mut sim, 1, 127);

        // Apply phase gate with angle 0
        apply_phase(&cpu, &mut sim, 0, 0.0);

        // Should remain |1⟩ (allow ±1 for Fixed8 rounding)
        let re = cpu.get_mem(&sim, 1) as i8;
        let im = cpu.get_mem(&sim, IMAG_BASE + 1) as i8;

        assert!(
            (re - 127).abs() <= 1,
            "Real part should be ~127, got {}",
            re
        );
        assert!(im.abs() <= 1, "Imaginary part should be ~0, got {}", im);
    }

    #[test]
    fn test_phase_doesnt_affect_zero_state() {
        // Phase gate should not affect |0⟩ state
        let mut sim = SlimSimulation::with_size(256, 64);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Initialize |0⟩ state
        cpu.set_mem(&mut sim, 0, 127);

        // Apply phase gate to qubit 0
        apply_phase(&cpu, &mut sim, 0, std::f64::consts::PI);

        // |0⟩ should be unchanged
        let re = cpu.get_mem(&sim, 0) as i8;
        let im = cpu.get_mem(&sim, IMAG_BASE) as i8;

        assert_eq!(re, 127, "|0⟩ should be unchanged by phase gate");
        assert_eq!(im, 0, "Imaginary part should be 0");
    }

    #[test]
    fn test_xyz_gates_on_superposition() {
        // Test X, Z, and Phase gates on superposition state
        let mut sim = SlimSimulation::with_size(256, 64);
        let cpu = PhysicalCpu::new((4, 4));
        cpu.place(&mut sim);

        // Create (|0⟩ + |1⟩)/√2 using Hadamard
        cpu.set_mem(&mut sim, 0, 127);
        apply_hadamard_q0(&cpu, &mut sim);

        let amp0_initial = cpu.get_mem(&sim, 0) as i8;
        let amp1_initial = cpu.get_mem(&sim, 1) as i8;

        println!(
            "Initial superposition: amp[0] = {}, amp[1] = {}",
            amp0_initial, amp1_initial
        );

        // Apply X gate: (|0⟩ + |1⟩)/√2 → (|1⟩ + |0⟩)/√2
        apply_pauli_x(&cpu, &mut sim, 0);

        let amp0_after_x = cpu.get_mem(&sim, 0) as i8;
        let amp1_after_x = cpu.get_mem(&sim, 1) as i8;

        println!(
            "After X: amp[0] = {}, amp[1] = {}",
            amp0_after_x, amp1_after_x
        );

        // Amplitudes should be swapped
        assert_eq!(amp0_after_x, amp1_initial, "X should swap amplitudes");
        assert_eq!(amp1_after_x, amp0_initial, "X should swap amplitudes");
    }
}
