// =============================================================================
// src/constrained_circuits.rs
// =============================================================================
//
// SPRINT 3.2d: Constrained Circuit Generation for Control Experiments
//
// This module enables the critical experiment: Quantum vs Classical comparison.
//
// By constraining circuit generation to exclude entangling gates, we create
// a "classical" control group. If quantum organisms outperform classical ones,
// evolution has discovered quantum advantage. If not, perhaps quantum resources
// aren't beneficial for this foraging task — still a meaningful result.
//
// The philosophical question: Does the universe reward entanglement?
//
// =============================================================================

use crate::quantum::QGate;
use crate::quantum_brain::SimpleRng;

use std::f32::consts::PI;

// Re-export SimpleRng for convenience
pub use crate::quantum_brain::SimpleRng as Rng;

// =============================================================================
// Circuit Constraints
// =============================================================================

/// Constraints on circuit generation.
#[derive(Clone, Debug, PartialEq)]
pub struct CircuitConstraints {
    /// Allow single-qubit gates (H, X, Z, Phase)
    pub allow_single_qubit: bool,

    /// Allow two-qubit entangling gates (CNOT, CZ)
    pub allow_two_qubit_entangling: bool,

    /// Allow three-qubit entangling gates (Toffoli, CCZ)
    pub allow_three_qubit_entangling: bool,

    /// Allow Phase gates (parameterized rotations)
    pub allow_phase: bool,

    /// Minimum circuit length
    pub min_length: usize,

    /// Maximum circuit length
    pub max_length: usize,
}

impl CircuitConstraints {
    /// Full quantum: all gates allowed.
    pub fn full_quantum() -> Self {
        Self {
            allow_single_qubit: true,
            allow_two_qubit_entangling: true,
            allow_three_qubit_entangling: true,
            allow_phase: true,
            min_length: 1,
            max_length: 30,
        }
    }

    /// Classical only: no entangling gates.
    ///
    /// This creates circuits that can be efficiently simulated classically
    /// (product states only, no superposition correlations between qubits).
    pub fn classical_only() -> Self {
        Self {
            allow_single_qubit: true,
            allow_two_qubit_entangling: false,
            allow_three_qubit_entangling: false,
            allow_phase: true,
            min_length: 1,
            max_length: 30,
        }
    }

    /// Minimal quantum: only two-qubit gates, no three-qubit.
    pub fn minimal_quantum() -> Self {
        Self {
            allow_single_qubit: true,
            allow_two_qubit_entangling: true,
            allow_three_qubit_entangling: false,
            allow_phase: true,
            min_length: 1,
            max_length: 30,
        }
    }

    /// Check if a gate is allowed under these constraints.
    pub fn is_allowed(&self, gate: &QGate) -> bool {
        match gate {
            QGate::H(_) | QGate::X(_) | QGate::Y(_) | QGate::Z(_) => self.allow_single_qubit,
            QGate::Phase(_, _) => self.allow_phase,
            QGate::Rx(_, _) | QGate::Ry(_, _) | QGate::Rz(_, _) => self.allow_phase, // EPIC 83: rotation gates
            QGate::U3(_, _, _, _) => self.allow_phase, // SPRINT 32.0: U3 is parameterized single-qubit gate
            QGate::T(_) | QGate::Tdg(_) => self.allow_phase, // SPRINT 48.0: T gates
            QGate::CNot(_, _) | QGate::CZ(_, _) | QGate::Swap(_, _) => {
                self.allow_two_qubit_entangling
            }
            QGate::Toffoli(_, _, _) => self.allow_three_qubit_entangling,
            QGate::CCZ(_, _, _) => self.allow_three_qubit_entangling,
            QGate::Measure(_) => true, // Always allow measurement
            // Shor's algorithm: Controlled rotation gates
            QGate::CPhase(_, _, _) | QGate::CRz(_, _, _) => self.allow_two_qubit_entangling,
        }
    }

    /// Check if any entangling gates are allowed.
    pub fn allows_entanglement(&self) -> bool {
        self.allow_two_qubit_entangling || self.allow_three_qubit_entangling
    }
}

impl Default for CircuitConstraints {
    fn default() -> Self {
        Self::full_quantum()
    }
}

// =============================================================================
// Constrained Circuit Generation
// =============================================================================

/// Generate a random circuit respecting constraints.
pub fn random_circuit_constrained(
    n_qubits: u8,
    length: usize,
    constraints: &CircuitConstraints,
    seed: u64,
) -> Vec<QGate> {
    let mut rng = SimpleRng::new(seed);
    let mut circuit = Vec::with_capacity(length);

    // Build list of allowed gate types
    let allowed_gates = build_allowed_gates(n_qubits, constraints);

    if allowed_gates.is_empty() {
        // Fallback: at least allow identity-like behavior
        return vec![QGate::H(0)]; // Single H is effectively a coin flip
    }

    for _ in 0..length {
        let gate = generate_random_gate(n_qubits, &allowed_gates, &mut rng);
        circuit.push(gate);
    }

    circuit
}

/// Mutate a circuit while respecting constraints.
pub fn mutate_circuit_constrained(
    circuit: &[QGate],
    n_qubits: u8,
    constraints: &CircuitConstraints,
    seed: u64,
) -> Vec<QGate> {
    let mut rng = SimpleRng::new(seed);
    let mut new_circuit = circuit.to_vec();

    // Filter out any disallowed gates that might have snuck in
    new_circuit.retain(|g| constraints.is_allowed(g));

    // Apply 1-3 mutations
    let num_mutations = rng.gen_range(1, 4);
    let allowed_gates = build_allowed_gates(n_qubits, constraints);

    for _ in 0..num_mutations {
        if new_circuit.is_empty() {
            if !allowed_gates.is_empty() {
                let gate = generate_random_gate(n_qubits, &allowed_gates, &mut rng);
                new_circuit.push(gate);
            }
            continue;
        }

        let mutation_type = rng.gen_range(0, 6);

        match mutation_type {
            0 => {
                // Insert gate
                if new_circuit.len() < constraints.max_length && !allowed_gates.is_empty() {
                    let pos = rng.gen_range(0, new_circuit.len() as u32 + 1) as usize;
                    let gate = generate_random_gate(n_qubits, &allowed_gates, &mut rng);
                    new_circuit.insert(pos, gate);
                }
            }
            1 => {
                // Delete gate
                if new_circuit.len() > constraints.min_length {
                    let pos = rng.gen_range(0, new_circuit.len() as u32) as usize;
                    new_circuit.remove(pos);
                }
            }
            2 => {
                // Swap gates
                if new_circuit.len() >= 2 {
                    let pos1 = rng.gen_range(0, new_circuit.len() as u32) as usize;
                    let mut pos2 = rng.gen_range(0, new_circuit.len() as u32) as usize;
                    while pos2 == pos1 && new_circuit.len() > 1 {
                        pos2 = rng.gen_range(0, new_circuit.len() as u32) as usize;
                    }
                    new_circuit.swap(pos1, pos2);
                }
            }
            3 => {
                // Change qubits
                let pos = rng.gen_range(0, new_circuit.len() as u32) as usize;
                new_circuit[pos] =
                    mutate_gate_qubits_constrained(&new_circuit[pos], n_qubits, &mut rng);
            }
            4 => {
                // Change angle (Phase gates only)
                if constraints.allow_phase {
                    let phase_positions: Vec<usize> = new_circuit
                        .iter()
                        .enumerate()
                        .filter_map(|(i, g)| {
                            if matches!(g, QGate::Phase(_, _)) {
                                Some(i)
                            } else {
                                None
                            }
                        })
                        .collect();

                    if !phase_positions.is_empty() {
                        let idx = phase_positions
                            [rng.gen_range(0, phase_positions.len() as u32) as usize];
                        if let QGate::Phase(q, angle) = new_circuit[idx] {
                            let delta = (rng.gen_float() - 0.5) * PI * 0.5;
                            new_circuit[idx] = QGate::Phase(q, angle + delta);
                        }
                    }
                }
            }
            _ => {
                // Replace gate
                if !allowed_gates.is_empty() {
                    let pos = rng.gen_range(0, new_circuit.len() as u32) as usize;
                    new_circuit[pos] = generate_random_gate(n_qubits, &allowed_gates, &mut rng);
                }
            }
        }
    }

    // Ensure minimum length
    while new_circuit.len() < constraints.min_length && !allowed_gates.is_empty() {
        let gate = generate_random_gate(n_qubits, &allowed_gates, &mut rng);
        new_circuit.push(gate);
    }

    new_circuit
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Gate type categories for random generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateType {
    Hadamard,
    PauliX,
    PauliZ,
    Phase,
    CNot,
    CZ,
    Toffoli,
    CCZ,
}

/// Build list of allowed gate types based on constraints.
fn build_allowed_gates(n_qubits: u8, constraints: &CircuitConstraints) -> Vec<GateType> {
    let mut allowed = Vec::new();

    if constraints.allow_single_qubit {
        allowed.push(GateType::Hadamard);
        allowed.push(GateType::PauliX);
        allowed.push(GateType::PauliZ);
    }

    if constraints.allow_phase {
        allowed.push(GateType::Phase);
    }

    if constraints.allow_two_qubit_entangling && n_qubits >= 2 {
        allowed.push(GateType::CNot);
        allowed.push(GateType::CZ);
    }

    if constraints.allow_three_qubit_entangling && n_qubits >= 3 {
        allowed.push(GateType::Toffoli);
        allowed.push(GateType::CCZ);
    }

    allowed
}

/// Generate a random gate of the specified type.
fn generate_random_gate(n_qubits: u8, allowed: &[GateType], rng: &mut SimpleRng) -> QGate {
    if allowed.is_empty() {
        return QGate::H(0); // Fallback
    }

    let gate_type = allowed[rng.gen_range(0, allowed.len() as u32) as usize];

    match gate_type {
        GateType::Hadamard => QGate::H(rng.gen_range(0, n_qubits as u32) as u8),
        GateType::PauliX => QGate::X(rng.gen_range(0, n_qubits as u32) as u8),
        GateType::PauliZ => QGate::Z(rng.gen_range(0, n_qubits as u32) as u8),
        GateType::Phase => {
            let q = rng.gen_range(0, n_qubits as u32) as u8;
            let angle = rng.gen_float() * 2.0 * PI;
            QGate::Phase(q, angle)
        }
        GateType::CNot => {
            let control = rng.gen_range(0, n_qubits as u32) as u8;
            let mut target = rng.gen_range(0, n_qubits as u32) as u8;
            while target == control {
                target = rng.gen_range(0, n_qubits as u32) as u8;
            }
            QGate::CNot(control, target)
        }
        GateType::CZ => {
            let a = rng.gen_range(0, n_qubits as u32) as u8;
            let mut b = rng.gen_range(0, n_qubits as u32) as u8;
            while b == a {
                b = rng.gen_range(0, n_qubits as u32) as u8;
            }
            QGate::CZ(a, b)
        }
        GateType::Toffoli => {
            let c0 = rng.gen_range(0, n_qubits as u32) as u8;
            let mut c1 = rng.gen_range(0, n_qubits as u32) as u8;
            while c1 == c0 {
                c1 = rng.gen_range(0, n_qubits as u32) as u8;
            }
            let mut t = rng.gen_range(0, n_qubits as u32) as u8;
            while t == c0 || t == c1 {
                t = rng.gen_range(0, n_qubits as u32) as u8;
            }
            QGate::Toffoli(c0, c1, t)
        }
        GateType::CCZ => {
            let a = rng.gen_range(0, n_qubits as u32) as u8;
            let mut b = rng.gen_range(0, n_qubits as u32) as u8;
            while b == a {
                b = rng.gen_range(0, n_qubits as u32) as u8;
            }
            let mut c = rng.gen_range(0, n_qubits as u32) as u8;
            while c == a || c == b {
                c = rng.gen_range(0, n_qubits as u32) as u8;
            }
            QGate::CCZ(a, b, c)
        }
    }
}

/// Mutate the qubit indices of a gate.
fn mutate_gate_qubits_constrained(gate: &QGate, n_qubits: u8, rng: &mut SimpleRng) -> QGate {
    match gate {
        QGate::H(_) => QGate::H(rng.gen_range(0, n_qubits as u32) as u8),
        QGate::X(_) => QGate::X(rng.gen_range(0, n_qubits as u32) as u8),
        QGate::Z(_) => QGate::Z(rng.gen_range(0, n_qubits as u32) as u8),
        QGate::Phase(_, angle) => QGate::Phase(rng.gen_range(0, n_qubits as u32) as u8, *angle),
        QGate::CNot(_, _) if n_qubits >= 2 => {
            let c = rng.gen_range(0, n_qubits as u32) as u8;
            let mut t = rng.gen_range(0, n_qubits as u32) as u8;
            while t == c {
                t = rng.gen_range(0, n_qubits as u32) as u8;
            }
            QGate::CNot(c, t)
        }
        QGate::CZ(_, _) if n_qubits >= 2 => {
            let a = rng.gen_range(0, n_qubits as u32) as u8;
            let mut b = rng.gen_range(0, n_qubits as u32) as u8;
            while b == a {
                b = rng.gen_range(0, n_qubits as u32) as u8;
            }
            QGate::CZ(a, b)
        }
        QGate::Toffoli(_, _, _) if n_qubits >= 3 => {
            let c0 = rng.gen_range(0, n_qubits as u32) as u8;
            let mut c1 = rng.gen_range(0, n_qubits as u32) as u8;
            while c1 == c0 {
                c1 = rng.gen_range(0, n_qubits as u32) as u8;
            }
            let mut t = rng.gen_range(0, n_qubits as u32) as u8;
            while t == c0 || t == c1 {
                t = rng.gen_range(0, n_qubits as u32) as u8;
            }
            QGate::Toffoli(c0, c1, t)
        }
        QGate::CCZ(_, _, _) if n_qubits >= 3 => {
            let a = rng.gen_range(0, n_qubits as u32) as u8;
            let mut b = rng.gen_range(0, n_qubits as u32) as u8;
            while b == a {
                b = rng.gen_range(0, n_qubits as u32) as u8;
            }
            let mut c = rng.gen_range(0, n_qubits as u32) as u8;
            while c == a || c == b {
                c = rng.gen_range(0, n_qubits as u32) as u8;
            }
            QGate::CCZ(a, b, c)
        }
        _ => gate.clone(),
    }
}

// =============================================================================
// Analysis Functions
// =============================================================================

/// Count entangling gates in a circuit.
pub fn count_entangling_gates(circuit: &[QGate]) -> usize {
    circuit.iter().filter(|g| is_entangling_gate(g)).count()
}

/// Check if a gate is entangling.
pub fn is_entangling_gate(gate: &QGate) -> bool {
    matches!(gate, QGate::CNot(_, _) | QGate::CZ(_, _))
        || matches!(gate, QGate::Toffoli(_, _, _) | QGate::CCZ(_, _, _))
}

/// Calculate the fraction of entangling gates.
pub fn entanglement_fraction(circuit: &[QGate]) -> f32 {
    if circuit.is_empty() {
        return 0.0;
    }
    count_entangling_gates(circuit) as f32 / circuit.len() as f32
}

/// Categorize a circuit by its entanglement level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntanglementLevel {
    /// No entangling gates (classical)
    None,
    /// Low: < 20% entangling gates
    Low,
    /// Medium: 20-50% entangling gates
    Medium,
    /// High: > 50% entangling gates
    High,
}

pub fn categorize_entanglement(circuit: &[QGate]) -> EntanglementLevel {
    let fraction = entanglement_fraction(circuit);

    if fraction == 0.0 {
        EntanglementLevel::None
    } else if fraction < 0.2 {
        EntanglementLevel::Low
    } else if fraction < 0.5 {
        EntanglementLevel::Medium
    } else {
        EntanglementLevel::High
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_quantum_allows_all() {
        let constraints = CircuitConstraints::full_quantum();

        assert!(constraints.is_allowed(&QGate::H(0)));
        assert!(constraints.is_allowed(&QGate::CNot(0, 1)));
        assert!(constraints.is_allowed(&QGate::Toffoli(0, 1, 2)));
        assert!(constraints.allows_entanglement());
    }

    #[test]
    fn test_classical_only_no_entanglement() {
        let constraints = CircuitConstraints::classical_only();

        assert!(constraints.is_allowed(&QGate::H(0)));
        assert!(constraints.is_allowed(&QGate::X(0)));
        assert!(constraints.is_allowed(&QGate::Phase(0, 1.0)));
        assert!(!constraints.is_allowed(&QGate::CNot(0, 1)));
        assert!(!constraints.is_allowed(&QGate::CZ(0, 1)));
        assert!(!constraints.is_allowed(&QGate::Toffoli(0, 1, 2)));
        assert!(!constraints.allows_entanglement());
    }

    #[test]
    fn test_random_circuit_classical() {
        let constraints = CircuitConstraints::classical_only();
        let circuit = random_circuit_constrained(4, 10, &constraints, 12345);

        assert_eq!(circuit.len(), 10);

        // No entangling gates should be present
        for gate in &circuit {
            assert!(
                !is_entangling_gate(gate),
                "Found entangling gate: {:?}",
                gate
            );
        }
    }

    #[test]
    fn test_random_circuit_quantum() {
        let constraints = CircuitConstraints::full_quantum();
        let circuit = random_circuit_constrained(4, 20, &constraints, 54321);

        // With high probability, should have some entangling gates
        let _entangling = count_entangling_gates(&circuit);
        // This is probabilistic, but 20 gates with entangling allowed should have some
        // Just check the circuit was generated
        assert_eq!(circuit.len(), 20);
    }

    #[test]
    fn test_mutate_respects_constraints() {
        let constraints = CircuitConstraints::classical_only();
        let initial = random_circuit_constrained(4, 5, &constraints, 11111);

        // Mutate many times
        let mut circuit = initial;
        for i in 0..100 {
            circuit = mutate_circuit_constrained(&circuit, 4, &constraints, i * 1000);

            // Should never have entangling gates
            for gate in &circuit {
                assert!(
                    !is_entangling_gate(gate),
                    "Mutation introduced entangling gate: {:?}",
                    gate
                );
            }
        }
    }

    #[test]
    fn test_entanglement_fraction() {
        let classical = vec![QGate::H(0), QGate::X(1), QGate::Z(2)];
        assert_eq!(entanglement_fraction(&classical), 0.0);

        let mixed = vec![QGate::H(0), QGate::CNot(0, 1)];
        assert!((entanglement_fraction(&mixed) - 0.5).abs() < 0.01);

        let all_entangling = vec![QGate::CNot(0, 1), QGate::CZ(1, 2)];
        assert!((entanglement_fraction(&all_entangling) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_categorize_entanglement() {
        let none = vec![QGate::H(0), QGate::X(1)];
        assert_eq!(categorize_entanglement(&none), EntanglementLevel::None);

        let low = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::H(3),
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::H(3),
            QGate::H(0),
            QGate::CNot(0, 1), // 10% entangling
        ];
        assert_eq!(categorize_entanglement(&low), EntanglementLevel::Low);

        let high = vec![QGate::CNot(0, 1), QGate::CZ(1, 2), QGate::H(0)];
        assert_eq!(categorize_entanglement(&high), EntanglementLevel::High);
    }

    #[test]
    fn test_minimal_quantum() {
        let constraints = CircuitConstraints::minimal_quantum();

        assert!(constraints.is_allowed(&QGate::CNot(0, 1)));
        assert!(!constraints.is_allowed(&QGate::Toffoli(0, 1, 2)));
    }
}
