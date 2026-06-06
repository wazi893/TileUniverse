//! IonQ Transpiler
//!
//! SPRINT 53.0: Transpile abstract circuits to IonQ native gate set
//!
//! Converts QGate circuits to IonQ Aria native gates:
//! - GPI (single-qubit rotation)
//! - GPI2 (π/2 rotation)
//! - MS (Mølmer-Sørensen entangling gate)
//!
//! Note: IonQ has all-to-all connectivity, so no SWAP routing is needed.

use std::f64::consts::PI;

use super::native_gates::{IonQCircuit, IonQGate, ionq_decompositions};
use crate::hardware::{HardwareProfile, profiles};
use logic_fabric_core::quantum::QGate;

/// IonQ Transpiler configuration
#[derive(Clone, Debug)]
pub struct IonQTranspilerConfig {
    /// Optimize by merging single-qubit gates
    pub optimize_single_qubit: bool,
}

impl Default for IonQTranspilerConfig {
    fn default() -> Self {
        Self {
            optimize_single_qubit: true,
        }
    }
}

/// IonQ Transpiler
///
/// Converts abstract quantum gates to IonQ native gate set.
/// IonQ trapped-ion systems have all-to-all connectivity,
/// so no SWAP routing is required.
pub struct IonQTranspiler {
    /// Hardware profile
    profile: HardwareProfile,
    /// Configuration
    config: IonQTranspilerConfig,
}

impl IonQTranspiler {
    /// Create transpiler for IonQ Aria
    pub fn aria() -> Self {
        Self {
            profile: profiles::ionq_aria(),
            config: IonQTranspilerConfig::default(),
        }
    }

    /// Create transpiler with custom profile
    pub fn new(profile: HardwareProfile) -> Self {
        Self {
            profile,
            config: IonQTranspilerConfig::default(),
        }
    }

    /// Create transpiler with custom config
    pub fn with_config(mut self, config: IonQTranspilerConfig) -> Self {
        self.config = config;
        self
    }

    /// Transpile a single QGate to IonQ native gates
    pub fn transpile_gate(&self, gate: &QGate) -> Vec<IonQGate> {
        match gate {
            // Pauli gates
            QGate::X(q) => ionq_decompositions::x_gate(*q as usize),
            QGate::Y(q) => ionq_decompositions::y_gate(*q as usize),
            QGate::Z(q) => ionq_decompositions::z_gate(*q as usize),

            // Hadamard
            QGate::H(q) => ionq_decompositions::hadamard(*q as usize),

            // T gates
            QGate::T(q) => {
                // T = Rz(π/4) - approximate with GPI sequence
                ionq_decompositions::rz(*q as usize, PI / 4.0)
            }
            QGate::Tdg(q) => {
                // T† = Rz(-π/4)
                ionq_decompositions::rz(*q as usize, -PI / 4.0)
            }

            // Rotation gates
            QGate::Rx(q, angle) => {
                // Rx(θ) = GPI(0) when θ = π, else use GPI2 sequence
                let theta = *angle as f64;
                if (theta - PI).abs() < 1e-10 {
                    vec![IonQGate::GPI(*q as usize, 0.0)]
                } else {
                    // Rx(θ) = GPI2(-π/2) · GPI2(θ/π - π/2)
                    vec![
                        IonQGate::GPI2(*q as usize, -PI / 2.0),
                        IonQGate::GPI2(*q as usize, theta - PI / 2.0),
                    ]
                }
            }
            QGate::Ry(q, angle) => {
                // Ry(θ) using GPI sequence
                let theta = *angle as f64;
                if (theta - PI).abs() < 1e-10 {
                    vec![IonQGate::GPI(*q as usize, PI / 2.0)]
                } else {
                    vec![
                        IonQGate::GPI2(*q as usize, 0.0),
                        IonQGate::GPI2(*q as usize, theta),
                    ]
                }
            }
            QGate::Rz(q, angle) => ionq_decompositions::rz(*q as usize, *angle as f64),

            // Two-qubit gates
            QGate::CNot(c, t) => ionq_decompositions::cnot(*c as usize, *t as usize),

            QGate::CZ(c, t) => {
                // CZ = H(t) · CX(c,t) · H(t)
                let mut gates = ionq_decompositions::hadamard(*t as usize);
                gates.extend(ionq_decompositions::cnot(*c as usize, *t as usize));
                gates.extend(ionq_decompositions::hadamard(*t as usize));
                gates
            }

            // SWAP using 3 CNOTs (each CNOT uses 1 MS)
            QGate::Swap(a, b) => {
                let mut gates = ionq_decompositions::cnot(*a as usize, *b as usize);
                gates.extend(ionq_decompositions::cnot(*b as usize, *a as usize));
                gates.extend(ionq_decompositions::cnot(*a as usize, *b as usize));
                gates
            }

            // Measurement
            QGate::Measure(q) => vec![IonQGate::Measure(*q as usize)],

            // U3 gate decomposition
            QGate::U3(q, theta, phi, lambda) => {
                // U3(θ, φ, λ) = Rz(φ) · Ry(θ) · Rz(λ)
                let q_idx = *q as usize;
                let mut gates = ionq_decompositions::rz(q_idx, *lambda as f64);
                gates.extend(self.transpile_gate(&QGate::Ry(*q, *theta)));
                gates.extend(ionq_decompositions::rz(q_idx, *phi as f64));
                gates
            }

            // Toffoli decomposition
            QGate::Toffoli(a, b, c) => {
                self.decompose_toffoli(*a as usize, *b as usize, *c as usize)
            }

            // CCZ = H(c) · Toffoli · H(c)
            QGate::CCZ(a, b, c) => {
                let mut gates = ionq_decompositions::hadamard(*c as usize);
                gates.extend(self.decompose_toffoli(*a as usize, *b as usize, *c as usize));
                gates.extend(ionq_decompositions::hadamard(*c as usize));
                gates
            }

            // CRz decomposition
            QGate::CRz(c, t, angle) => {
                let half = *angle as f64 / 2.0;
                let mut gates = ionq_decompositions::rz(*t as usize, half);
                gates.extend(ionq_decompositions::cnot(*c as usize, *t as usize));
                gates.extend(ionq_decompositions::rz(*t as usize, -half));
                gates.extend(ionq_decompositions::cnot(*c as usize, *t as usize));
                gates
            }

            // CPhase decomposition
            QGate::CPhase(c, t, angle) => {
                let half = *angle as f64 / 2.0;
                let mut gates = ionq_decompositions::rz(*c as usize, half);
                gates.extend(ionq_decompositions::cnot(*c as usize, *t as usize));
                gates.extend(ionq_decompositions::rz(*t as usize, -half));
                gates.extend(ionq_decompositions::cnot(*c as usize, *t as usize));
                gates.extend(ionq_decompositions::rz(*t as usize, half));
                gates
            }

            // Phase gate
            QGate::Phase(q, angle) => ionq_decompositions::rz(*q as usize, *angle as f64),
        }
    }

    /// Decompose Toffoli to native gates
    fn decompose_toffoli(&self, a: usize, b: usize, c: usize) -> Vec<IonQGate> {
        let mut gates = Vec::new();

        // Standard Toffoli decomposition
        gates.extend(ionq_decompositions::hadamard(c));
        gates.extend(ionq_decompositions::cnot(b, c));
        gates.extend(ionq_decompositions::rz(c, -PI / 4.0)); // T†
        gates.extend(ionq_decompositions::cnot(a, c));
        gates.extend(ionq_decompositions::rz(c, PI / 4.0)); // T
        gates.extend(ionq_decompositions::cnot(b, c));
        gates.extend(ionq_decompositions::rz(c, -PI / 4.0)); // T†
        gates.extend(ionq_decompositions::cnot(a, c));
        gates.extend(ionq_decompositions::rz(b, PI / 4.0)); // T
        gates.extend(ionq_decompositions::rz(c, PI / 4.0)); // T
        gates.extend(ionq_decompositions::cnot(a, b));
        gates.extend(ionq_decompositions::hadamard(c));
        gates.extend(ionq_decompositions::rz(a, PI / 4.0)); // T
        gates.extend(ionq_decompositions::rz(b, -PI / 4.0)); // T†
        gates.extend(ionq_decompositions::cnot(a, b));

        gates
    }

    /// Transpile a full circuit
    pub fn transpile(&self, circuit: &[QGate], name: &str) -> IonQCircuit {
        // Find max qubit index
        let num_qubits = circuit
            .iter()
            .flat_map(|g| self.gate_qubits(g))
            .max()
            .map(|m| m + 1)
            .unwrap_or(1);

        let mut ionq_circuit = IonQCircuit::new(name, num_qubits);

        // Transpile each gate
        for gate in circuit {
            let ionq_gates = self.transpile_gate(gate);
            for g in ionq_gates {
                ionq_circuit.add(g);
            }
        }

        ionq_circuit
    }

    /// Get qubits involved in a QGate
    fn gate_qubits(&self, gate: &QGate) -> Vec<usize> {
        match gate {
            QGate::X(q)
            | QGate::Y(q)
            | QGate::Z(q)
            | QGate::H(q)
            | QGate::T(q)
            | QGate::Tdg(q)
            | QGate::Rx(q, _)
            | QGate::Ry(q, _)
            | QGate::Rz(q, _)
            | QGate::Phase(q, _)
            | QGate::Measure(q) => vec![*q as usize],

            QGate::CNot(c, t)
            | QGate::CZ(c, t)
            | QGate::Swap(c, t)
            | QGate::CRz(c, t, _)
            | QGate::CPhase(c, t, _) => vec![*c as usize, *t as usize],

            QGate::Toffoli(a, b, c) | QGate::CCZ(a, b, c) => {
                vec![*a as usize, *b as usize, *c as usize]
            }

            QGate::U3(q, _, _, _) => vec![*q as usize],
        }
    }

    /// Get hardware profile
    pub fn profile(&self) -> &HardwareProfile {
        &self.profile
    }
}

/// Transpilation statistics for IonQ
#[derive(Clone, Debug, Default)]
pub struct IonQTranspileStats {
    /// Original gate count
    pub original_gates: usize,
    /// Transpiled gate count
    pub transpiled_gates: usize,
    /// MS (entangling) gate count
    pub ms_count: usize,
    /// Single-qubit gate count (GPI + GPI2)
    pub single_qubit_count: usize,
    /// Circuit depth
    pub depth: usize,
    /// Estimated duration in milliseconds
    pub duration_ms: f64,
}

impl IonQTranspiler {
    /// Get transpilation statistics
    pub fn stats(&self, original: &[QGate], transpiled: &IonQCircuit) -> IonQTranspileStats {
        IonQTranspileStats {
            original_gates: original.len(),
            transpiled_gates: transpiled.gate_count(),
            ms_count: transpiled.count("ms"),
            single_qubit_count: transpiled.count("gpi") + transpiled.count("gpi2"),
            depth: transpiled.depth(),
            duration_ms: transpiled.total_duration_ns() / 1e6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpile_hadamard() {
        let transpiler = IonQTranspiler::aria();
        let circuit = vec![QGate::H(0)];
        let ionq = transpiler.transpile(&circuit, "test");

        // H = GPI2 + GPI
        assert!(ionq.gate_count() >= 2);
    }

    #[test]
    fn test_transpile_cnot() {
        let transpiler = IonQTranspiler::aria();
        let circuit = vec![QGate::CNot(0, 1)];
        let ionq = transpiler.transpile(&circuit, "test");

        // CNOT uses 1 MS gate
        assert_eq!(ionq.count("ms"), 1);
    }

    #[test]
    fn test_transpile_bell_state() {
        let transpiler = IonQTranspiler::aria();
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let ionq = transpiler.transpile(&circuit, "bell");

        assert_eq!(ionq.count("ms"), 1);
        assert_eq!(ionq.num_qubits, 2);
    }

    #[test]
    fn test_no_swap_needed() {
        // IonQ has all-to-all connectivity
        let transpiler = IonQTranspiler::aria();

        // CNOT between non-adjacent qubits (would need SWAP on IBM)
        let circuit = vec![QGate::CNot(0, 5), QGate::CNot(2, 7)];
        let ionq = transpiler.transpile(&circuit, "test");

        // Should only have 2 MS gates, no extra SWAPs
        assert_eq!(ionq.count("ms"), 2);
    }

    #[test]
    fn test_transpile_swap() {
        let transpiler = IonQTranspiler::aria();
        let circuit = vec![QGate::Swap(0, 1)];
        let ionq = transpiler.transpile(&circuit, "test");

        // SWAP = 3 CNOT = 3 MS
        assert_eq!(ionq.count("ms"), 3);
    }

    #[test]
    fn test_circuit_timing() {
        let transpiler = IonQTranspiler::aria();
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let ionq = transpiler.transpile(&circuit, "bell");

        // IonQ gates are slow (microseconds)
        let duration_us = ionq.total_duration_ns() / 1000.0;
        assert!(duration_us > 100.0); // Should be hundreds of microseconds
    }

    #[test]
    fn test_stats() {
        let transpiler = IonQTranspiler::aria();
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1), QGate::Measure(0)];
        let ionq = transpiler.transpile(&circuit, "test");
        let stats = transpiler.stats(&circuit, &ionq);

        assert_eq!(stats.original_gates, 3);
        assert!(stats.transpiled_gates > 3);
        assert_eq!(stats.ms_count, 1);
    }
}
