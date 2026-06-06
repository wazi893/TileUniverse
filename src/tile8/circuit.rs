//! Quantum Circuit Representation and Compilation
//!
//! Provides a high-level API for defining quantum circuits and compiling them
//! into TILE-8 CPU operations (local vs cross-block gates).
//!
//! # Example
//!
//! ```ignore
//! use engine::tile8::circuit::Circuit;
//!
//! // Create 9-qubit GHZ state
//! let circuit = Circuit::new(9)
//!     .h(0)                  // Hadamard on qubit 0
//!     .cnot(0, 1)            // Local CNOT
//!     .cnot(0, 2)            // Local CNOT
//!     .cnot(0, 7)            // Cross-block CNOT (qubit 7 crosses CPU boundary)
//!     .cnot(0, 8);           // Cross-block CNOT
//!
//! // Circuit automatically classifies gates as local or cross-block
//! ```

use std::fmt;

/// Quantum gate types supported by TILE-8
#[derive(Debug, Clone, PartialEq)]
pub enum Gate {
    /// Hadamard gate - creates superposition
    /// H|0⟩ = (|0⟩ + |1⟩)/√2
    H { target: usize },

    /// Pauli-X gate - bit flip
    /// X|0⟩ = |1⟩, X|1⟩ = |0⟩
    X { target: usize },

    /// Pauli-Z gate - phase flip
    /// Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    Z { target: usize },

    /// CNOT gate - controlled NOT
    /// |control,target⟩ → |control, target ⊕ control⟩
    CNOT { control: usize, target: usize },

    /// Phase rotation gate (future)
    /// Rz(θ)|ψ⟩ = e^(iθ)|ψ⟩
    Rz { target: usize, angle: f64 },

    /// Measurement (collapses to classical bit)
    Measure { target: usize },
}

impl Gate {
    /// Returns the qubits affected by this gate
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            Gate::H { target }
            | Gate::X { target }
            | Gate::Z { target }
            | Gate::Rz { target, .. }
            | Gate::Measure { target } => vec![*target],
            Gate::CNOT { control, target } => vec![*control, *target],
        }
    }

    /// Returns the maximum qubit index used by this gate
    pub fn max_qubit(&self) -> usize {
        *self.qubits().iter().max().unwrap()
    }

    /// Check if this gate operates on a single qubit
    pub fn is_single_qubit(&self) -> bool {
        matches!(
            self,
            Gate::H { .. }
                | Gate::X { .. }
                | Gate::Z { .. }
                | Gate::Rz { .. }
                | Gate::Measure { .. }
        )
    }

    /// Check if this gate is a two-qubit gate
    pub fn is_two_qubit(&self) -> bool {
        matches!(self, Gate::CNOT { .. })
    }
}

impl fmt::Display for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Gate::H { target } => write!(f, "H(q{})", target),
            Gate::X { target } => write!(f, "X(q{})", target),
            Gate::Z { target } => write!(f, "Z(q{})", target),
            Gate::CNOT { control, target } => write!(f, "CNOT(q{}, q{})", control, target),
            Gate::Rz { target, angle } => write!(f, "Rz(q{}, {:.3})", target, angle),
            Gate::Measure { target } => write!(f, "M(q{})", target),
        }
    }
}

/// Gate classification based on CPU locality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateType {
    /// Gate operates entirely within a single CPU
    /// Examples: H(0), X(3), CNOT(0, 5) for 7-qubit local block
    Local,

    /// Gate spans multiple CPUs (requires cross-block communication)
    /// Examples: CNOT(0, 7), CNOT(0, 8) for 9-qubit system with 4 CPUs
    CrossBlock,
}

/// Compiled gate with CPU assignment and classification
#[derive(Debug, Clone)]
pub struct CompiledGate {
    /// The original gate
    pub gate: Gate,

    /// Gate classification (local or cross-block)
    pub gate_type: GateType,

    /// CPU ID(s) involved in this gate
    /// Local gates: single CPU
    /// Cross-block gates: multiple CPUs
    pub cpu_ids: Vec<usize>,
}

/// Quantum circuit - sequence of quantum gates
pub struct Circuit {
    /// Number of qubits in the circuit
    num_qubits: usize,

    /// Sequence of gates to apply
    gates: Vec<Gate>,
}

impl Circuit {
    /// Create a new quantum circuit with specified number of qubits
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits (must be > 0)
    ///
    /// # Example
    /// ```ignore
    /// let circuit = Circuit::new(9);  // 9-qubit circuit
    /// ```
    pub fn new(num_qubits: usize) -> Self {
        assert!(num_qubits > 0, "Circuit must have at least 1 qubit");
        Self {
            num_qubits,
            gates: Vec::new(),
        }
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the gates in this circuit
    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    /// Add a Hadamard gate
    ///
    /// # Arguments
    /// * `target` - Target qubit index (0-indexed)
    pub fn h(mut self, target: usize) -> Self {
        assert!(target < self.num_qubits, "Qubit {} out of range", target);
        self.gates.push(Gate::H { target });
        self
    }

    /// Add a Pauli-X gate
    pub fn x(mut self, target: usize) -> Self {
        assert!(target < self.num_qubits, "Qubit {} out of range", target);
        self.gates.push(Gate::X { target });
        self
    }

    /// Add a Pauli-Z gate
    pub fn z(mut self, target: usize) -> Self {
        assert!(target < self.num_qubits, "Qubit {} out of range", target);
        self.gates.push(Gate::Z { target });
        self
    }

    /// Add a CNOT gate
    ///
    /// # Arguments
    /// * `control` - Control qubit
    /// * `target` - Target qubit (flipped if control is |1⟩)
    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        assert!(
            control < self.num_qubits,
            "Control qubit {} out of range",
            control
        );
        assert!(
            target < self.num_qubits,
            "Target qubit {} out of range",
            target
        );
        assert_ne!(
            control, target,
            "Control and target must be different qubits"
        );
        self.gates.push(Gate::CNOT { control, target });
        self
    }

    /// Add a phase rotation gate
    pub fn rz(mut self, target: usize, angle: f64) -> Self {
        assert!(target < self.num_qubits, "Qubit {} out of range", target);
        self.gates.push(Gate::Rz { target, angle });
        self
    }

    /// Add a measurement
    pub fn measure(mut self, target: usize) -> Self {
        assert!(target < self.num_qubits, "Qubit {} out of range", target);
        self.gates.push(Gate::Measure { target });
        self
    }

    /// Measure all qubits
    pub fn measure_all(mut self) -> Self {
        for q in 0..self.num_qubits {
            self.gates.push(Gate::Measure { target: q });
        }
        self
    }

    /// Compile the circuit for execution on a distributed quantum grid
    ///
    /// Classifies gates as local or cross-block based on:
    /// - Number of qubits
    /// - Number of CPUs (derived from qubit count)
    /// - Qubit-to-CPU mapping
    ///
    /// # Returns
    /// Vector of compiled gates with CPU assignments
    pub fn compile(&self) -> Vec<CompiledGate> {
        let mut compiled = Vec::new();

        for gate in &self.gates {
            let (gate_type, cpu_ids) = self.classify_gate(gate);
            compiled.push(CompiledGate {
                gate: gate.clone(),
                gate_type,
                cpu_ids,
            });
        }

        compiled
    }

    /// Classify a gate as local or cross-block
    ///
    /// **CPU Assignment Rules**:
    /// - Each CPU handles 128 complex amplitudes (7 qubits worth)
    /// - For N qubits, we need 2^N / 128 = 2^(N-7) CPUs
    /// - Qubits 0-6 are local to each CPU
    /// - Qubit 7+ determine which CPU (cross-block operations)
    ///
    /// **Examples**:
    /// - 9 qubits: 4 CPUs, qubits 7-8 are cross-block
    /// - 15 qubits: 256 CPUs, qubits 7-14 are cross-block
    fn classify_gate(&self, gate: &Gate) -> (GateType, Vec<usize>) {
        match gate {
            // Single-qubit gates are always local
            Gate::H { target }
            | Gate::X { target }
            | Gate::Z { target }
            | Gate::Rz { target, .. }
            | Gate::Measure { target } => {
                let cpu_id = self.qubit_to_cpu(*target);
                (GateType::Local, vec![cpu_id])
            }

            // CNOT classification depends on target qubit
            Gate::CNOT { control, target } => {
                // If target is in qubits 0-6, it's local within a CPU
                // If target is qubit 7+, it crosses CPU boundaries
                if *target < 7 {
                    // Local CNOT within a CPU
                    let cpu_id = self.qubit_to_cpu(*control);
                    (GateType::Local, vec![cpu_id])
                } else {
                    // Cross-block CNOT
                    // For CNOT(0, 7): affects multiple CPUs based on control bit
                    // We need to identify which CPUs are involved
                    let cpu_ids = self.cross_block_cpu_ids(*control, *target);
                    (GateType::CrossBlock, cpu_ids)
                }
            }
        }
    }

    /// Map a qubit to its owning CPU
    ///
    /// For qubits 0-6: All CPUs have these (local qubits)
    /// For qubits 7+: These bits determine which CPU
    ///
    /// For now, return CPU 0 for local qubits (they're on all CPUs)
    fn qubit_to_cpu(&self, qubit: usize) -> usize {
        if qubit < 7 {
            0 // Local qubit (exists on all CPUs)
        } else {
            // For cross-block qubits, this is more complex
            // For now, return 0 as placeholder
            0
        }
    }

    /// Identify which CPUs are involved in a cross-block operation
    ///
    /// For CNOT(control, target) where target >= 7:
    /// - The operation swaps amplitudes between CPUs based on the target qubit
    /// - For target = 7 (qubit 7): CPUs differ in bit 0 of their ID (0↔1, 2↔3, etc.)
    /// - For target = 8 (qubit 8): CPUs differ in bit 1 of their ID (0↔2, 1↔3, etc.)
    ///
    /// For simplicity in Phase 3, we'll return all CPUs involved
    fn cross_block_cpu_ids(&self, _control: usize, _target: usize) -> Vec<usize> {
        // Calculate number of CPUs needed
        let num_cpus = 1 << (self.num_qubits - 7);

        // For cross-block operations, we need pairs of CPUs that differ in the target bit
        // For now, return range 0..num_cpus as a simplification
        // In actual execution, we'll call cross_block_cnot for specific pairs
        (0..num_cpus).collect()
    }

    /// Count gates by type
    pub fn gate_counts(&self) -> GateCounts {
        let compiled = self.compile();
        let mut counts = GateCounts::default();

        for cg in &compiled {
            match cg.gate_type {
                GateType::Local => counts.local += 1,
                GateType::CrossBlock => counts.cross_block += 1,
            }

            match &cg.gate {
                Gate::H { .. } => counts.hadamard += 1,
                Gate::X { .. } => counts.pauli_x += 1,
                Gate::Z { .. } => counts.pauli_z += 1,
                Gate::CNOT { .. } => counts.cnot += 1,
                Gate::Rz { .. } => counts.rz += 1,
                Gate::Measure { .. } => counts.measure += 1,
            }
        }

        counts
    }

    /// Estimate circuit depth (number of time steps required)
    ///
    /// Assumes perfect parallelization of independent gates
    /// Cross-block gates may need sequential execution
    pub fn depth(&self) -> usize {
        // Simplified: assume all gates are sequential for now
        // TODO: Implement proper dependency analysis
        self.gates.len()
    }
}

/// Statistics about gates in a circuit
#[derive(Debug, Default, Clone)]
pub struct GateCounts {
    pub hadamard: usize,
    pub pauli_x: usize,
    pub pauli_z: usize,
    pub cnot: usize,
    pub rz: usize,
    pub measure: usize,
    pub local: usize,
    pub cross_block: usize,
}

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Circuit ({} qubits, {} gates):",
            self.num_qubits,
            self.gates.len()
        )?;
        for (i, gate) in self.gates.iter().enumerate() {
            writeln!(f, "  {}: {}", i, gate)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_creation() {
        let circuit = Circuit::new(9);
        assert_eq!(circuit.num_qubits(), 9);
        assert_eq!(circuit.gates().len(), 0);
    }

    #[test]
    fn test_circuit_builder() {
        let circuit = Circuit::new(9).h(0).cnot(0, 1).cnot(0, 2).x(3).z(4);

        assert_eq!(circuit.gates().len(), 5);
        assert!(matches!(circuit.gates()[0], Gate::H { target: 0 }));
        assert!(matches!(
            circuit.gates()[1],
            Gate::CNOT {
                control: 0,
                target: 1
            }
        ));
    }

    #[test]
    fn test_ghz_circuit() {
        // 9-qubit GHZ state
        let circuit = Circuit::new(9)
            .h(0)
            .cnot(0, 1)
            .cnot(0, 2)
            .cnot(0, 3)
            .cnot(0, 4)
            .cnot(0, 5)
            .cnot(0, 6)
            .cnot(0, 7) // Cross-block
            .cnot(0, 8); // Cross-block

        assert_eq!(circuit.num_qubits(), 9);
        assert_eq!(circuit.gates().len(), 9); // 1 H + 8 CNOTs
    }

    #[test]
    fn test_gate_classification() {
        let circuit = Circuit::new(9)
            .h(0) // Local
            .cnot(0, 1) // Local (target < 7)
            .cnot(0, 7) // Cross-block (target >= 7)
            .cnot(0, 8); // Cross-block

        let compiled = circuit.compile();

        assert_eq!(compiled.len(), 4);

        // H(0) is local
        assert_eq!(compiled[0].gate_type, GateType::Local);

        // CNOT(0, 1) is local (both qubits < 7)
        assert_eq!(compiled[1].gate_type, GateType::Local);

        // CNOT(0, 7) is cross-block (target = 7)
        assert_eq!(compiled[2].gate_type, GateType::CrossBlock);

        // CNOT(0, 8) is cross-block (target = 8)
        assert_eq!(compiled[3].gate_type, GateType::CrossBlock);
    }

    #[test]
    fn test_gate_counts() {
        let circuit = Circuit::new(9)
            .h(0)
            .cnot(0, 1)
            .cnot(0, 2)
            .cnot(0, 7)
            .cnot(0, 8)
            .measure_all();

        let counts = circuit.gate_counts();

        assert_eq!(counts.hadamard, 1);
        assert_eq!(counts.cnot, 4);
        assert_eq!(counts.measure, 9);
        assert_eq!(counts.local, 1 + 2 + 9); // H + 2 local CNOTs + 9 measures
        assert_eq!(counts.cross_block, 2); // 2 cross-block CNOTs
    }

    #[test]
    fn test_circuit_display() {
        let circuit = Circuit::new(3).h(0).cnot(0, 1).cnot(0, 2);

        let display = format!("{}", circuit);
        assert!(display.contains("Circuit (3 qubits, 3 gates)"));
        assert!(display.contains("H(q0)"));
        assert!(display.contains("CNOT(q0, q1)"));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_invalid_qubit_index() {
        Circuit::new(5).h(10); // Qubit 10 doesn't exist
    }

    #[test]
    #[should_panic(expected = "Control and target must be different")]
    fn test_cnot_same_qubit() {
        Circuit::new(5).cnot(2, 2); // Can't CNOT qubit with itself
    }
}
