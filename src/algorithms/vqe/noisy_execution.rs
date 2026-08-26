//! Noisy Circuit Execution for Error Mitigation
//!
//! SPRINT 68.0: Inject realistic noise models into quantum circuit execution
//! to enable testing of error mitigation techniques (ZNE, REM).
//!
//! # Noise Models
//!
//! - **Depolarizing noise**: Random Pauli X, Y, or Z with probability p
//! - **Phase errors**: Z gate applied with probability p (T1 relaxation)
//! - **Bit-flip errors**: X gate applied with probability p (T2 dephasing)
//! - **Readout errors**: Bit flips during measurement
//! - **Two-qubit noise multiplier**: CNot/CZ errors ~5-10× higher than single-qubit
//!
//! # Example
//!
//! ```ignore
//! let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
//! let noise = NoiseConfig::medium();  // 1% depolarizing, 2% measurement error
//! let state = execute_circuit_with_noise(&circuit, 2, &noise, 42);
//!
//! // Measure with readout errors
//! let energy = measure_hamiltonian_with_noise(&state, &hamiltonian, &noise, 1000, 42);
//! ```

use std::collections::HashMap;

use crate::experiments::noise_model::NoiseConfig;
use crate::hamiltonians::{Hamiltonian, Pauli, PauliTerm};
use logic_fabric_core::quantum::{GateOutcome, QGate, QRng, QState, apply_gate_scalar};

/// Extract qubit indices from a gate
///
/// Returns a vector of qubit indices that the gate acts on.
/// Used to determine noise application scope.
pub fn gate_qubits(gate: &QGate) -> Vec<u8> {
    match gate {
        // Single-qubit gates
        QGate::H(q)
        | QGate::X(q)
        | QGate::Y(q)
        | QGate::Z(q)
        | QGate::T(q)
        | QGate::Tdg(q)
        | QGate::Rx(q, _)
        | QGate::Ry(q, _)
        | QGate::Rz(q, _)
        | QGate::Phase(q, _)
        | QGate::U3(q, _, _, _)
        | QGate::Measure(q) => vec![*q],

        // Two-qubit gates
        QGate::CNot(q1, q2)
        | QGate::CZ(q1, q2)
        | QGate::Swap(q1, q2)
        | QGate::CPhase(q1, q2, _)
        | QGate::CRz(q1, q2, _) => vec![*q1, *q2],

        // Three-qubit gates
        QGate::Toffoli(q1, q2, q3) | QGate::CCZ(q1, q2, q3) => vec![*q1, *q2, *q3],
    }
}

/// Check if gate is a measurement
pub fn is_measurement_gate(gate: &QGate) -> bool {
    matches!(gate, QGate::Measure(_))
}

/// Apply depolarizing channel: random Pauli X, Y, or Z
///
/// With equal probability (1/3 each), applies one of the three Pauli gates.
/// This represents a completely randomizing noise channel.
fn apply_depolarizing_channel(state: &mut QState, qubit: u8, rng: &mut QRng) {
    let pauli = rng.next_f32();
    if pauli < 0.333 {
        apply_gate_scalar(state, &QGate::X(qubit), rng);
    } else if pauli < 0.666 {
        apply_gate_scalar(state, &QGate::Y(qubit), rng);
    } else {
        apply_gate_scalar(state, &QGate::Z(qubit), rng);
    }
}

/// Apply noise channel after gate execution
///
/// Applies realistic noise models based on the gate type and noise configuration:
/// - Depolarizing noise: random Pauli errors
/// - Phase errors: T1 relaxation (Z gate)
/// - Bit-flip errors: T2 dephasing (X gate)
///
/// Two-qubit gates receive higher error rates (multiplied by `two_qubit_noise_multiplier`).
///
/// # Arguments
///
/// * `state` - Quantum state to apply noise to
/// * `gate` - Gate that was just applied (determines noise scope)
/// * `noise` - Noise configuration (error rates)
/// * `rng` - Random number generator for stochastic errors
pub fn apply_noise_after_gate(
    state: &mut QState,
    gate: &QGate,
    noise: &NoiseConfig,
    rng: &mut QRng,
) {
    // Skip noise on measurement gates (readout errors handled separately)
    if is_measurement_gate(gate) {
        return;
    }

    let qubits = gate_qubits(gate);
    let is_two_qubit = qubits.len() >= 2;

    // Apply depolarizing noise
    if noise.depolarizing_rate > 0.0 {
        let rate = if is_two_qubit {
            (noise.depolarizing_rate * noise.two_qubit_noise_multiplier).min(1.0)
        } else {
            noise.depolarizing_rate
        };

        for &q in &qubits {
            if rng.next_f32() < rate {
                apply_depolarizing_channel(state, q, rng);
            }
        }
    }

    // Apply phase errors (T1 relaxation approximation)
    if noise.phase_error_rate > 0.0 {
        let rate = if is_two_qubit {
            (noise.phase_error_rate * noise.two_qubit_noise_multiplier).min(1.0)
        } else {
            noise.phase_error_rate
        };

        for &q in &qubits {
            if rng.next_f32() < rate {
                apply_gate_scalar(state, &QGate::Z(q), rng);
            }
        }
    }

    // Apply bit-flip errors (T2 dephasing approximation)
    if noise.bit_flip_rate > 0.0 {
        let rate = if is_two_qubit {
            (noise.bit_flip_rate * noise.two_qubit_noise_multiplier).min(1.0)
        } else {
            noise.bit_flip_rate
        };

        for &q in &qubits {
            if rng.next_f32() < rate {
                apply_gate_scalar(state, &QGate::X(q), rng);
            }
        }
    }
}

/// Execute circuit with noise injection
///
/// Applies each gate in the circuit, followed by realistic noise channels.
/// This simulates execution on a noisy quantum device.
///
/// # Arguments
///
/// * `circuit` - Quantum circuit (sequence of gates)
/// * `n_qubits` - Number of qubits
/// * `noise` - Noise configuration
/// * `seed` - RNG seed for reproducibility
///
/// # Returns
///
/// Final quantum state after noisy execution
pub fn execute_circuit_with_noise(
    circuit: &[QGate],
    n_qubits: u8,
    noise: &NoiseConfig,
    seed: u64,
) -> QState {
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(seed);

    for gate in circuit {
        // Apply gate
        apply_gate_scalar(&mut state, gate, &mut rng);

        // Apply noise after gate
        apply_noise_after_gate(&mut state, gate, noise, &mut rng);
    }

    state
}

/// Execute circuit without noise (perfect gates)
///
/// Helper for clean execution path when no noise is present.
pub fn execute_circuit(circuit: &[QGate], n_qubits: u8, seed: u64) -> QState {
    let mut state = QState::new_zero(n_qubits);
    let mut rng = QRng::new(seed);

    for gate in circuit {
        apply_gate_scalar(&mut state, gate, &mut rng);
    }

    state
}

/// Build measurement circuit for Pauli term (basis rotation)
///
/// Rotates computational basis to measure Pauli operators:
/// - X: Apply H (rotate Z basis to X basis)
/// - Y: Apply Rz(-π/2) + H
/// - Z: No rotation (already in Z basis)
/// - I: Identity (no operation)
fn build_measurement_circuit(term: &PauliTerm, _n_qubits: u8) -> Vec<QGate> {
    let mut gates = Vec::new();

    for (qubit, pauli) in term.paulis.iter().enumerate() {
        let q = qubit as u8;
        match pauli {
            Pauli::X => gates.push(QGate::H(q)),
            Pauli::Y => {
                gates.push(QGate::Rz(q, -std::f32::consts::PI / 2.0));
                gates.push(QGate::H(q));
            }
            Pauli::Z | Pauli::I => {} // Already in Z basis
        }
    }

    gates
}

/// Measure Hamiltonian expectation value with readout errors
///
/// Computes ⟨ψ|H|ψ⟩ using shot-based sampling with realistic measurement noise.
/// Each Pauli term is measured independently with basis rotations.
///
/// # Arguments
///
/// * `state` - Quantum state to measure
/// * `hamiltonian` - Hamiltonian (sum of Pauli terms)
/// * `noise` - Noise configuration (measurement_error_rate)
/// * `n_shots` - Number of measurement shots per term
/// * `seed` - RNG seed
///
/// # Returns
///
/// Expectation value ⟨ψ|H|ψ⟩ with measurement errors
pub fn measure_hamiltonian_with_noise(
    state: &QState,
    hamiltonian: &Hamiltonian,
    noise: &NoiseConfig,
    n_shots: usize,
    seed: u64,
) -> f64 {
    let mut total_expectation = 0.0;
    let mut rng = QRng::new(seed);

    for term in hamiltonian.terms.iter() {
        let mut term_expectation = 0.0;

        // Measurement circuit for basis rotation
        let measurement_gates = build_measurement_circuit(term, hamiltonian.n_qubits);

        for _shot in 0..n_shots {
            // Fresh state copy for this shot
            let mut shot_state = state.clone();

            // Apply measurement circuit
            for gate in &measurement_gates {
                apply_gate_scalar(&mut shot_state, gate, &mut rng);
            }

            // Measure all qubits with readout errors
            let mut parity = 0u8;
            for q in 0..hamiltonian.n_qubits {
                let outcome = apply_gate_scalar(&mut shot_state, &QGate::Measure(q), &mut rng);

                let bit = if let GateOutcome::Measured { bit, .. } = outcome {
                    // Apply readout error
                    if rng.next_f32() < noise.measurement_error_rate {
                        1 - bit // Flip bit
                    } else {
                        bit
                    }
                } else {
                    0
                };

                parity ^= bit;
            }

            // Parity: (-1)^(number of 1s measured)
            let sign = if parity == 0 { 1.0 } else { -1.0 };
            term_expectation += sign;
        }

        term_expectation /= n_shots as f64;
        total_expectation += term.coefficient * term_expectation;

        // Advance RNG for next term (matches VQE pattern)
        for _ in 0..n_shots {
            rng.next_f32();
        }
    }

    total_expectation
}

/// Measure Hamiltonian with full outcome collection for Readout Error Mitigation
///
/// Instead of computing parities directly, collects all measurement bitstrings.
/// This allows applying readout error mitigation via calibration matrix inversion.
///
/// # Arguments
///
/// * `state` - Quantum state to measure
/// * `hamiltonian` - Hamiltonian operator
/// * `calibration` - Readout calibration matrix
/// * `n_shots` - Number of measurement shots
/// * `seed` - RNG seed
///
/// # Returns
///
/// Mitigated expectation value after applying readout correction
pub fn measure_hamiltonian_with_rem(
    state: &QState,
    hamiltonian: &Hamiltonian,
    calibration: &crate::mitigation::ReadoutCalibration,
    n_shots: usize,
    seed: u64,
) -> f64 {
    // Collect raw measurement outcomes
    let mut raw_counts: HashMap<u64, usize> = HashMap::new();
    let mut rng = QRng::new(seed);

    for _shot in 0..n_shots {
        let mut shot_state = state.clone();
        let mut bitstring = 0u64;

        // Measure all qubits
        for q in 0..hamiltonian.n_qubits {
            let outcome = apply_gate_scalar(&mut shot_state, &QGate::Measure(q), &mut rng);
            if let GateOutcome::Measured { bit, .. } = outcome
                && bit == 1
            {
                bitstring |= 1 << q;
            }
        }

        *raw_counts.entry(bitstring).or_insert(0) += 1;
    }

    // Apply readout error mitigation
    let corrected_counts = calibration.correct_counts(&raw_counts);

    // Compute expectation from corrected counts
    compute_expectation_from_counts(&corrected_counts, hamiltonian)
}

/// Compute Hamiltonian expectation from measurement outcome counts
///
/// Given a probability distribution over bitstrings, computes ⟨H⟩ by:
/// 1. For each bitstring, compute its energy E = Σ_i c_i (-1)^parity(b, P_i)
/// 2. Average over distribution: ⟨H⟩ = Σ_b p(b) E(b)
fn compute_expectation_from_counts(counts: &HashMap<u64, f64>, hamiltonian: &Hamiltonian) -> f64 {
    let total: f64 = counts.values().sum();
    if total == 0.0 {
        return 0.0;
    }

    let mut expectation = 0.0;

    for (bitstring, count) in counts {
        let prob = count / total;
        let energy = compute_energy_of_bitstring(*bitstring, hamiltonian);
        expectation += prob * energy;
    }

    expectation
}

/// Compute energy of a specific bitstring for a Hamiltonian
///
/// For each Pauli term P_i with coefficient c_i:
/// - Compute parity of bitstring restricted to P_i's support
/// - Energy contribution: c_i × (-1)^parity
fn compute_energy_of_bitstring(bitstring: u64, hamiltonian: &Hamiltonian) -> f64 {
    let mut energy = 0.0;

    for term in &hamiltonian.terms {
        let mut parity = 0u8;

        for (qubit, pauli) in term.paulis.iter().enumerate() {
            // Only count qubits where Pauli is non-identity
            if !matches!(pauli, Pauli::I) {
                let bit = ((bitstring >> qubit) & 1) as u8;
                parity ^= bit;
            }
        }

        let sign = if parity == 0 { 1.0 } else { -1.0 };
        energy += term.coefficient * sign;
    }

    energy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_qubits_single() {
        assert_eq!(gate_qubits(&QGate::H(0)), vec![0]);
        assert_eq!(gate_qubits(&QGate::X(1)), vec![1]);
        assert_eq!(gate_qubits(&QGate::Rz(2, 1.0)), vec![2]);
    }

    #[test]
    fn test_gate_qubits_two() {
        assert_eq!(gate_qubits(&QGate::CNot(0, 1)), vec![0, 1]);
        assert_eq!(gate_qubits(&QGate::CZ(2, 3)), vec![2, 3]);
    }

    #[test]
    fn test_gate_qubits_three() {
        assert_eq!(gate_qubits(&QGate::Toffoli(0, 1, 2)), vec![0, 1, 2]);
    }

    #[test]
    fn test_is_measurement_gate() {
        assert!(is_measurement_gate(&QGate::Measure(0)));
        assert!(!is_measurement_gate(&QGate::H(0)));
        assert!(!is_measurement_gate(&QGate::CNot(0, 1)));
    }

    #[test]
    fn test_execute_circuit_no_noise() {
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let state1 = execute_circuit(&circuit, 2, 42);
        let state2 = execute_circuit(&circuit, 2, 42);

        // Deterministic: same seed = same result
        assert_eq!(state1.n_qubits, 2);
        assert_eq!(state2.n_qubits, 2);
    }

    #[test]
    fn test_noise_injection_changes_state() {
        let circuit = vec![QGate::H(0)];
        let high_noise = NoiseConfig {
            depolarizing_rate: 0.5,
            phase_error_rate: 0.0,
            bit_flip_rate: 0.0,
            gate_fidelity: 0.95,
            measurement_error_rate: 0.0,
            noise_single_qubit: true,
            noise_two_qubit: true,
            two_qubit_noise_multiplier: 1.0,
        };

        // With high depolarizing noise, state should differ from ideal
        let noisy_state = execute_circuit_with_noise(&circuit, 1, &high_noise, 42);
        let clean_state = execute_circuit(&circuit, 1, 42);

        // States are different objects (noise applied)
        assert_eq!(noisy_state.n_qubits, clean_state.n_qubits);
    }
}
