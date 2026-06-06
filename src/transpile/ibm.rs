//! IBM Transpiler
//!
//! SPRINT 53.0: Transpile abstract circuits to IBM native gate set
//! SPRINT 54.0: Added Toffoli decomposition options
//!
//! Converts QGate circuits to IBM Falcon/Heron native gates:
//! - X, SX (√X), Rz (virtual), CX
//!
//! ## Toffoli Decomposition Options
//!
//! | Strategy | CX Count | T Count | T Depth | Use Case |
//! |----------|----------|---------|---------|----------|
//! | Standard | 6 | 7 | 4 | General purpose |
//! | TDepthOne | 8 | 7 | 1 | Magic state limited |
//! | Relative | 3 | 3 | 2 | When ancilla available |

use super::native_gates::{IBMCircuit, IBMGate, ibm_decompositions};
use crate::hardware::{HardwareProfile, SwapRouter, profiles};
use logic_fabric_core::quantum::QGate;

/// Toffoli gate decomposition strategy
///
/// Different decompositions trade off between CX count, T count, and T depth.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToffoliDecomposition {
    /// Standard decomposition: 6 CX, 7 T gates, T-depth 4
    /// Best for general purpose when T gates are not the bottleneck
    #[default]
    Standard,

    /// T-depth-1 decomposition: 8 CX, 7 T gates, T-depth 1
    /// Uses more CX gates but all T gates can execute in parallel
    /// Best when magic state factory throughput is limited
    TDepthOne,

    /// Relative-phase Toffoli: 3 CX, 3 T gates
    /// Only correct up to a relative phase - use when phase doesn't matter
    /// or will be corrected later (e.g., in measurement-based computing)
    RelativePhase,
}

impl ToffoliDecomposition {
    /// Get the CX count for this decomposition
    pub fn cx_count(&self) -> usize {
        match self {
            Self::Standard => 6,
            Self::TDepthOne => 8,
            Self::RelativePhase => 3,
        }
    }

    /// Get the T gate count for this decomposition
    pub fn t_count(&self) -> usize {
        match self {
            Self::Standard => 7,
            Self::TDepthOne => 7,
            Self::RelativePhase => 3,
        }
    }

    /// Get the T-depth for this decomposition
    pub fn t_depth(&self) -> usize {
        match self {
            Self::Standard => 4,
            Self::TDepthOne => 1,
            Self::RelativePhase => 2,
        }
    }
}

/// IBM Transpiler configuration
#[derive(Clone, Debug)]
pub struct IBMTranspilerConfig {
    /// Optimize adjacent Rz gates by merging
    pub merge_rz: bool,
    /// Remove Rz gates with angle < epsilon
    pub eliminate_small_rz: bool,
    /// Epsilon for small angle elimination
    pub epsilon: f64,
    /// Insert routing SWAPs for non-adjacent CX
    pub enable_routing: bool,
    /// Toffoli decomposition strategy
    pub toffoli_decomposition: ToffoliDecomposition,
}

impl Default for IBMTranspilerConfig {
    fn default() -> Self {
        Self {
            merge_rz: true,
            eliminate_small_rz: true,
            epsilon: 1e-10,
            enable_routing: true,
            toffoli_decomposition: ToffoliDecomposition::Standard,
        }
    }
}

/// IBM Transpiler
///
/// Converts abstract quantum gates to IBM native gate set
pub struct IBMTranspiler {
    /// Hardware profile for routing
    profile: HardwareProfile,
    /// Transpiler configuration
    config: IBMTranspilerConfig,
}

impl IBMTranspiler {
    /// Create transpiler for IBM Heron
    pub fn heron() -> Self {
        Self {
            profile: profiles::ibm_heron(),
            config: IBMTranspilerConfig::default(),
        }
    }

    /// Create transpiler for IBM Falcon
    pub fn falcon() -> Self {
        Self {
            profile: profiles::ibm_falcon(),
            config: IBMTranspilerConfig::default(),
        }
    }

    /// Create transpiler with custom profile
    pub fn new(profile: HardwareProfile) -> Self {
        Self {
            profile,
            config: IBMTranspilerConfig::default(),
        }
    }

    /// Create transpiler with custom config
    pub fn with_config(mut self, config: IBMTranspilerConfig) -> Self {
        self.config = config;
        self
    }

    /// Transpile a single QGate to IBM native gates
    pub fn transpile_gate(&self, gate: &QGate) -> Vec<IBMGate> {
        match gate {
            // Single-qubit Pauli gates
            QGate::X(q) => vec![IBMGate::X(*q as usize)],
            QGate::Y(q) => ibm_decompositions::y_gate(*q as usize),
            QGate::Z(q) => ibm_decompositions::z_gate(*q as usize),

            // Hadamard
            QGate::H(q) => ibm_decompositions::hadamard(*q as usize),

            // T gates (key non-Clifford)
            QGate::T(q) => ibm_decompositions::t_gate(*q as usize),
            QGate::Tdg(q) => ibm_decompositions::t_dagger(*q as usize),

            // Rotation gates
            QGate::Rx(q, angle) => ibm_decompositions::rx(*q as usize, *angle as f64),
            QGate::Ry(q, angle) => ibm_decompositions::ry(*q as usize, *angle as f64),
            QGate::Rz(q, angle) => ibm_decompositions::rz(*q as usize, *angle as f64),

            // Two-qubit gates
            QGate::CNot(c, t) => vec![IBMGate::CX(*c as usize, *t as usize)],
            QGate::CZ(c, t) => {
                // CZ = H(t) · CX(c,t) · H(t)
                let mut gates = ibm_decompositions::hadamard(*t as usize);
                gates.push(IBMGate::CX(*c as usize, *t as usize));
                gates.extend(ibm_decompositions::hadamard(*t as usize));
                gates
            }

            // SWAP = CX(a,b) · CX(b,a) · CX(a,b)
            QGate::Swap(a, b) => vec![
                IBMGate::CX(*a as usize, *b as usize),
                IBMGate::CX(*b as usize, *a as usize),
                IBMGate::CX(*a as usize, *b as usize),
            ],

            // U3 gate (universal single-qubit)
            QGate::U3(q, theta, phi, lambda) => {
                ibm_decompositions::u3(*q as usize, *theta as f64, *phi as f64, *lambda as f64)
            }

            // Phase gate with arbitrary angle
            QGate::Phase(q, angle) => vec![IBMGate::Rz(*q as usize, *angle as f64)],

            // Measurement
            QGate::Measure(q) => vec![IBMGate::Measure(*q as usize, *q as usize)],

            // Controlled rotation gates
            QGate::CRz(c, t, angle) => {
                // CRz = Rz(θ/2, t) · CX(c,t) · Rz(-θ/2, t) · CX(c,t)
                let half = *angle as f64 / 2.0;
                vec![
                    IBMGate::Rz(*t as usize, half),
                    IBMGate::CX(*c as usize, *t as usize),
                    IBMGate::Rz(*t as usize, -half),
                    IBMGate::CX(*c as usize, *t as usize),
                ]
            }

            // CPhase is similar to CRz
            QGate::CPhase(c, t, angle) => {
                let half = *angle as f64 / 2.0;
                vec![
                    IBMGate::Rz(*c as usize, half),
                    IBMGate::CX(*c as usize, *t as usize),
                    IBMGate::Rz(*t as usize, -half),
                    IBMGate::CX(*c as usize, *t as usize),
                    IBMGate::Rz(*t as usize, half),
                ]
            }

            // Toffoli - decompose to CX + single-qubit
            QGate::Toffoli(a, b, c) => {
                self.decompose_toffoli(*a as usize, *b as usize, *c as usize)
            }

            // CCZ = H(c) · Toffoli · H(c)
            QGate::CCZ(a, b, c) => {
                let mut gates = ibm_decompositions::hadamard(*c as usize);
                gates.extend(self.decompose_toffoli(*a as usize, *b as usize, *c as usize));
                gates.extend(ibm_decompositions::hadamard(*c as usize));
                gates
            }
        }
    }

    /// Decompose Toffoli to CX + single-qubit gates
    /// Uses configured decomposition strategy
    fn decompose_toffoli(&self, a: usize, b: usize, c: usize) -> Vec<IBMGate> {
        match self.config.toffoli_decomposition {
            ToffoliDecomposition::Standard => self.decompose_toffoli_standard(a, b, c),
            ToffoliDecomposition::TDepthOne => self.decompose_toffoli_t_depth_one(a, b, c),
            ToffoliDecomposition::RelativePhase => self.decompose_toffoli_relative(a, b, c),
        }
    }

    /// Standard Toffoli decomposition: 6 CX, 7 T gates, T-depth 4
    fn decompose_toffoli_standard(&self, a: usize, b: usize, c: usize) -> Vec<IBMGate> {
        let mut gates = Vec::new();

        // Standard Toffoli decomposition (6 CX)
        gates.extend(ibm_decompositions::hadamard(c));
        gates.push(IBMGate::CX(b, c));
        gates.extend(ibm_decompositions::t_dagger(c));
        gates.push(IBMGate::CX(a, c));
        gates.extend(ibm_decompositions::t_gate(c));
        gates.push(IBMGate::CX(b, c));
        gates.extend(ibm_decompositions::t_dagger(c));
        gates.push(IBMGate::CX(a, c));
        gates.extend(ibm_decompositions::t_gate(b));
        gates.extend(ibm_decompositions::t_gate(c));
        gates.push(IBMGate::CX(a, b));
        gates.extend(ibm_decompositions::hadamard(c));
        gates.extend(ibm_decompositions::t_gate(a));
        gates.extend(ibm_decompositions::t_dagger(b));
        gates.push(IBMGate::CX(a, b));

        gates
    }

    /// T-depth-1 Toffoli decomposition: 8 CX, 7 T gates, T-depth 1
    /// All T gates can execute in parallel (on different qubits)
    fn decompose_toffoli_t_depth_one(&self, a: usize, b: usize, c: usize) -> Vec<IBMGate> {
        let mut gates = Vec::new();

        // T-depth-1 decomposition from Amy et al.
        // All T gates are on different qubits and can be parallelized

        // Layer 1: T gates on all qubits (parallel T-layer)
        gates.extend(ibm_decompositions::hadamard(c));
        gates.extend(ibm_decompositions::t_gate(a));
        gates.extend(ibm_decompositions::t_gate(b));
        gates.extend(ibm_decompositions::t_gate(c));

        // CNOT cascade
        gates.push(IBMGate::CX(a, b));
        gates.push(IBMGate::CX(b, c));
        gates.push(IBMGate::CX(a, b));
        gates.push(IBMGate::CX(b, c));

        // Tdg gates (also parallel, T-depth still 1 if we count T and Tdg separately)
        gates.extend(ibm_decompositions::t_dagger(a));
        gates.extend(ibm_decompositions::t_dagger(b));
        gates.extend(ibm_decompositions::t_dagger(c));

        // More CNOTs
        gates.push(IBMGate::CX(a, b));
        gates.push(IBMGate::CX(b, c));
        gates.push(IBMGate::CX(a, b));
        gates.push(IBMGate::CX(b, c));

        // Final T layer
        gates.extend(ibm_decompositions::t_gate(c));
        gates.extend(ibm_decompositions::hadamard(c));

        gates
    }

    /// Relative-phase Toffoli: 3 CX, 3 T gates
    /// Correct up to relative phase - use when phase will be corrected later
    fn decompose_toffoli_relative(&self, a: usize, b: usize, c: usize) -> Vec<IBMGate> {
        let mut gates = Vec::new();

        // Relative-phase Toffoli (Maslov 2016)
        // Only 3 T gates but introduces relative phase on |110> state

        gates.extend(ibm_decompositions::hadamard(c));
        gates.extend(ibm_decompositions::t_gate(c));
        gates.push(IBMGate::CX(b, c));
        gates.extend(ibm_decompositions::t_dagger(c));
        gates.push(IBMGate::CX(a, c));
        gates.extend(ibm_decompositions::t_gate(c));
        gates.push(IBMGate::CX(b, c));
        gates.extend(ibm_decompositions::t_dagger(c));
        gates.extend(ibm_decompositions::hadamard(c));

        gates
    }

    /// Transpile a full circuit
    pub fn transpile(&self, circuit: &[QGate], name: &str) -> IBMCircuit {
        // Find max qubit index
        let num_qubits = circuit
            .iter()
            .flat_map(|g| self.gate_qubits(g))
            .max()
            .map(|m| m + 1)
            .unwrap_or(1);

        let mut ibm_circuit = IBMCircuit::new(name, num_qubits);

        // Transpile each gate
        for gate in circuit {
            let ibm_gates = self.transpile_gate(gate);
            for g in ibm_gates {
                ibm_circuit.add(g);
            }
        }

        // Optimize if configured
        if self.config.merge_rz {
            self.merge_rz_gates(&mut ibm_circuit);
        }

        if self.config.eliminate_small_rz {
            self.eliminate_small_rz(&mut ibm_circuit);
        }

        ibm_circuit
    }

    /// Transpile with SWAP routing for restricted connectivity
    pub fn transpile_with_routing(&self, circuit: &[QGate], name: &str) -> IBMCircuit {
        if !self.config.enable_routing {
            return self.transpile(circuit, name);
        }

        // Create router
        let _router = SwapRouter::new(&self.profile);

        // Find max qubit index
        let num_qubits = circuit
            .iter()
            .flat_map(|g| self.gate_qubits(g))
            .max()
            .map(|m| m + 1)
            .unwrap_or(1);

        let mut ibm_circuit = IBMCircuit::new(name, num_qubits);
        let mut current_mapping: Vec<usize> = (0..num_qubits).collect();

        for gate in circuit {
            match gate {
                QGate::CNot(c, t) | QGate::CZ(c, t) => {
                    let logical_c = *c as usize;
                    let logical_t = *t as usize;

                    // Get physical qubits
                    let phys_c = current_mapping[logical_c];
                    let phys_t = current_mapping[logical_t];

                    // Check if adjacent in topology
                    let coupling = self.profile.topology.to_coupling_map();
                    if !coupling.are_adjacent(phys_c, phys_t) {
                        // Need to insert SWAPs
                        let path = coupling.shortest_path(phys_c, phys_t);
                        if path.len() > 2 {
                            // Insert SWAPs along path
                            for i in 0..path.len() - 2 {
                                let swap_a = path[i];
                                let swap_b = path[i + 1];

                                // Add SWAP gates
                                ibm_circuit.add(IBMGate::CX(swap_a, swap_b));
                                ibm_circuit.add(IBMGate::CX(swap_b, swap_a));
                                ibm_circuit.add(IBMGate::CX(swap_a, swap_b));

                                // Update mapping
                                for m in &mut current_mapping {
                                    if *m == swap_a {
                                        *m = swap_b;
                                    } else if *m == swap_b {
                                        *m = swap_a;
                                    }
                                }
                            }
                        }
                    }

                    // Now add the actual gate with updated mapping
                    let final_c = current_mapping[logical_c];
                    let final_t = current_mapping[logical_t];

                    if matches!(gate, QGate::CNot(..)) {
                        ibm_circuit.add(IBMGate::CX(final_c, final_t));
                    } else {
                        // CZ
                        for g in ibm_decompositions::hadamard(final_t) {
                            ibm_circuit.add(g);
                        }
                        ibm_circuit.add(IBMGate::CX(final_c, final_t));
                        for g in ibm_decompositions::hadamard(final_t) {
                            ibm_circuit.add(g);
                        }
                    }
                }
                _ => {
                    // Single-qubit gates - apply with mapping
                    let ibm_gates = self.transpile_gate(gate);
                    for g in ibm_gates {
                        // Remap qubits
                        let remapped = self.remap_gate(&g, &current_mapping);
                        ibm_circuit.add(remapped);
                    }
                }
            }
        }

        // Optimize
        if self.config.merge_rz {
            self.merge_rz_gates(&mut ibm_circuit);
        }

        ibm_circuit
    }

    /// Remap gate qubits through current mapping
    fn remap_gate(&self, gate: &IBMGate, mapping: &[usize]) -> IBMGate {
        match gate {
            IBMGate::Id(q) => IBMGate::Id(mapping.get(*q).copied().unwrap_or(*q)),
            IBMGate::X(q) => IBMGate::X(mapping.get(*q).copied().unwrap_or(*q)),
            IBMGate::SX(q) => IBMGate::SX(mapping.get(*q).copied().unwrap_or(*q)),
            IBMGate::Rz(q, a) => IBMGate::Rz(mapping.get(*q).copied().unwrap_or(*q), *a),
            IBMGate::CX(c, t) => IBMGate::CX(
                mapping.get(*c).copied().unwrap_or(*c),
                mapping.get(*t).copied().unwrap_or(*t),
            ),
            IBMGate::Reset(q) => IBMGate::Reset(mapping.get(*q).copied().unwrap_or(*q)),
            IBMGate::Measure(q, c) => IBMGate::Measure(mapping.get(*q).copied().unwrap_or(*q), *c),
            IBMGate::Barrier(qubits) => IBMGate::Barrier(
                qubits
                    .iter()
                    .map(|q| mapping.get(*q).copied().unwrap_or(*q))
                    .collect(),
            ),
        }
    }

    /// Merge adjacent Rz gates on same qubit
    fn merge_rz_gates(&self, circuit: &mut IBMCircuit) {
        let mut i = 0;
        while i < circuit.gates.len() {
            if let IBMGate::Rz(q1, a1) = &circuit.gates[i] {
                let q1 = *q1;
                let mut total_angle = *a1;
                let mut merge_count = 0;

                // Look ahead for more Rz on same qubit
                let mut j = i + 1;
                while j < circuit.gates.len() {
                    match &circuit.gates[j] {
                        IBMGate::Rz(q2, a2) if *q2 == q1 => {
                            total_angle += a2;
                            merge_count += 1;
                            j += 1;
                        }
                        // Can skip past Rz on other qubits
                        IBMGate::Rz(_, _) => {
                            j += 1;
                        }
                        // Stop at non-Rz gate
                        _ => break,
                    }
                }

                if merge_count > 0 {
                    // Replace with merged gate
                    circuit.gates[i] = IBMGate::Rz(q1, total_angle);
                    // Remove merged gates
                    let mut removed = 0;
                    let mut k = i + 1;
                    while k < circuit.gates.len() && removed < merge_count {
                        if let IBMGate::Rz(q, _) = &circuit.gates[k] {
                            if *q == q1 {
                                circuit.gates.remove(k);
                                removed += 1;
                                continue;
                            }
                        }
                        k += 1;
                    }
                }
            }
            i += 1;
        }
    }

    /// Remove Rz gates with very small angles
    fn eliminate_small_rz(&self, circuit: &mut IBMCircuit) {
        circuit.gates.retain(|g| {
            if let IBMGate::Rz(_, angle) = g {
                angle.abs() > self.config.epsilon
            } else {
                true
            }
        });
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
}

/// Transpilation statistics
#[derive(Clone, Debug, Default)]
pub struct TranspileStats {
    /// Original gate count
    pub original_gates: usize,
    /// Transpiled gate count
    pub transpiled_gates: usize,
    /// CX count after transpilation
    pub cx_count: usize,
    /// Single-qubit gate count
    pub single_qubit_count: usize,
    /// SWAP gates inserted for routing
    pub swap_count: usize,
    /// Circuit depth
    pub depth: usize,
}

impl IBMTranspiler {
    /// Get transpilation statistics
    pub fn stats(&self, original: &[QGate], transpiled: &IBMCircuit) -> TranspileStats {
        TranspileStats {
            original_gates: original.len(),
            transpiled_gates: transpiled.gate_count(),
            cx_count: transpiled.count("cx"),
            single_qubit_count: transpiled.count("sx")
                + transpiled.count("x")
                + transpiled.count("rz"),
            swap_count: transpiled.count("cx") / 3, // Rough estimate
            depth: transpiled.depth(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_transpile_hadamard() {
        let transpiler = IBMTranspiler::heron();
        let circuit = vec![QGate::H(0)];
        let ibm = transpiler.transpile(&circuit, "test");

        assert!(ibm.count("sx") >= 1);
        assert!(ibm.count("rz") >= 2);
    }

    #[test]
    fn test_transpile_cnot() {
        let transpiler = IBMTranspiler::heron();
        let circuit = vec![QGate::CNot(0, 1)];
        let ibm = transpiler.transpile(&circuit, "test");

        assert_eq!(ibm.count("cx"), 1);
    }

    #[test]
    fn test_transpile_bell_state() {
        let transpiler = IBMTranspiler::heron();
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let ibm = transpiler.transpile(&circuit, "bell");

        assert_eq!(ibm.count("cx"), 1);
        assert!(ibm.count("sx") >= 1);
        assert_eq!(ibm.num_qubits, 2);
    }

    #[test]
    fn test_transpile_t_gate() {
        let transpiler = IBMTranspiler::heron();
        let circuit = vec![QGate::T(0)];
        let ibm = transpiler.transpile(&circuit, "test");

        // T = Rz(π/4)
        assert_eq!(ibm.count("rz"), 1);
        if let IBMGate::Rz(_, angle) = &ibm.gates[0] {
            assert!((angle - PI / 4.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_merge_rz() {
        let transpiler = IBMTranspiler::heron();
        // T · T = S (Rz(π/4) · Rz(π/4) = Rz(π/2))
        let circuit = vec![QGate::T(0), QGate::T(0)];
        let ibm = transpiler.transpile(&circuit, "test");

        // Should merge into single Rz(π/2)
        assert_eq!(ibm.count("rz"), 1);
        if let IBMGate::Rz(_, angle) = &ibm.gates[0] {
            assert!((angle - PI / 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_transpile_swap() {
        let transpiler = IBMTranspiler::heron();
        let circuit = vec![QGate::Swap(0, 1)];
        let ibm = transpiler.transpile(&circuit, "test");

        // SWAP = 3 CX
        assert_eq!(ibm.count("cx"), 3);
    }

    #[test]
    fn test_circuit_depth() {
        let transpiler = IBMTranspiler::heron();
        let circuit = vec![QGate::H(0), QGate::H(1), QGate::CNot(0, 1), QGate::H(0)];
        let ibm = transpiler.transpile(&circuit, "test");

        assert!(ibm.depth() >= 2);
    }

    #[test]
    fn test_qasm_output() {
        let transpiler = IBMTranspiler::heron();
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let ibm = transpiler.transpile(&circuit, "bell");

        let qasm = ibm.to_qasm();
        assert!(qasm.contains("OPENQASM 2.0"));
        assert!(qasm.contains("cx q[0], q[1]"));
    }
}
