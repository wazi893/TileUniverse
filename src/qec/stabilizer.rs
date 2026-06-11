//! Stabilizer Tableau: Aaronson-Gottesman Representation
//!
//! Represents stabilizer states using a tableau of Pauli operators.
//! This enables O(n²) simulation of Clifford circuits on n qubits.
//!
//! Reference: Aaronson & Gottesman, "Improved Simulation of Stabilizer Circuits"
//! https://arxiv.org/abs/quant-ph/0406196

use std::f32::consts::PI;
use std::fmt;

use logic_fabric_core::quantum::QGate;

use super::gate_noise::{
    GateError, GateErrorModel, Pauli, sample_measurement_error, sample_single_qubit_error,
    sample_two_qubit_error,
};
use super::noise::SimpleRng;

// =========================================================================
// ERROR TYPES (Phase 77.2)
// =========================================================================

/// Errors that can occur during stabilizer circuit execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StabilizerError {
    /// The circuit contains a non-Clifford gate that cannot be simulated.
    NonCliffordGate(String),
    /// A qubit index is out of range for the tableau.
    QubitOutOfRange { qubit: usize, n_qubits: usize },
}

/// A single Pauli operator on n qubits: ±i^r X^x Z^z
/// where x, z are n-bit vectors and r ∈ {0,1,2,3} tracks the phase.
#[derive(Clone, Debug)]
pub struct PauliOperator {
    /// X component: bit i is 1 if X acts on qubit i
    pub x: Vec<u64>,
    /// Z component: bit i is 1 if Z acts on qubit i
    pub z: Vec<u64>,
    /// Phase: 0 = +1, 1 = +i, 2 = -1, 3 = -i
    pub phase: u8,
    /// Number of qubits
    n_qubits: usize,
}

impl PauliOperator {
    /// Create identity operator on n qubits
    pub fn identity(n_qubits: usize) -> Self {
        let n_words = n_qubits.div_ceil(64);
        Self {
            x: vec![0u64; n_words],
            z: vec![0u64; n_words],
            phase: 0,
            n_qubits,
        }
    }

    /// Create X operator on single qubit
    pub fn x(n_qubits: usize, qubit: usize) -> Self {
        let mut op = Self::identity(n_qubits);
        op.set_x(qubit, true);
        op
    }

    /// Create Z operator on single qubit
    pub fn z(n_qubits: usize, qubit: usize) -> Self {
        let mut op = Self::identity(n_qubits);
        op.set_z(qubit, true);
        op
    }

    /// Create Y operator on single qubit (Y = iXZ)
    pub fn y(n_qubits: usize, qubit: usize) -> Self {
        let mut op = Self::identity(n_qubits);
        op.set_x(qubit, true);
        op.set_z(qubit, true);
        op.phase = 1; // i factor
        op
    }

    #[inline]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    #[inline]
    pub fn get_x(&self, qubit: usize) -> bool {
        let word = qubit / 64;
        let bit = qubit % 64;
        (self.x[word] >> bit) & 1 != 0
    }

    #[inline]
    pub fn get_z(&self, qubit: usize) -> bool {
        let word = qubit / 64;
        let bit = qubit % 64;
        (self.z[word] >> bit) & 1 != 0
    }

    #[inline]
    pub fn set_x(&mut self, qubit: usize, val: bool) {
        let word = qubit / 64;
        let bit = qubit % 64;
        if val {
            self.x[word] |= 1u64 << bit;
        } else {
            self.x[word] &= !(1u64 << bit);
        }
    }

    #[inline]
    pub fn set_z(&mut self, qubit: usize, val: bool) {
        let word = qubit / 64;
        let bit = qubit % 64;
        if val {
            self.z[word] |= 1u64 << bit;
        } else {
            self.z[word] &= !(1u64 << bit);
        }
    }

    /// Get Pauli type on qubit: I=0, X=1, Z=2, Y=3
    #[inline]
    pub fn pauli_at(&self, qubit: usize) -> u8 {
        let x = self.get_x(qubit) as u8;
        let z = self.get_z(qubit) as u8;
        x | (z << 1)
    }

    /// Multiply two Pauli operators: self * other
    /// Updates phase according to Pauli multiplication rules
    pub fn multiply(&self, other: &PauliOperator) -> PauliOperator {
        let mut result = self.clone();
        result.multiply_assign(other);
        result
    }

    /// In-place multiply: self *= other (SIMD-optimized)
    /// Avoids allocation and uses fast bit-parallel phase calculation
    #[inline]
    pub fn multiply_assign(&mut self, other: &PauliOperator) {
        debug_assert_eq!(self.n_qubits, other.n_qubits);

        // Fast phase calculation using bit operations
        // Phase contribution = 2 * popcount(x1 & z2) + 2 * popcount(x1 & x2 & (z1 ^ z2))
        // Simplified: count where self has X and other has Z (each gives +i = +1 phase)
        //            count where self has Z and other has X (each gives -i = +3 phase)
        let mut phase_add = 0u32;
        let mut phase_sub = 0u32;

        for i in 0..self.x.len() {
            // X1 * Z2 -> +i contribution
            phase_add += (self.x[i] & other.z[i]).count_ones();
            // Z1 * X2 -> -i contribution
            phase_sub += (self.z[i] & other.x[i]).count_ones();
            // XOR the components
            self.x[i] ^= other.x[i];
            self.z[i] ^= other.z[i];
        }

        // Net phase: each X*Z gives +1, each Z*X gives +3 (= -1 mod 4)
        let phase_contrib = (phase_add + 3 * phase_sub) % 4;
        self.phase = ((self.phase as u32 + other.phase as u32 + phase_contrib) % 4) as u8;
    }

    /// Check if this operator commutes with another
    pub fn commutes_with(&self, other: &PauliOperator) -> bool {
        assert_eq!(self.n_qubits, other.n_qubits);

        // Count anti-commuting positions
        let mut anticommute_count = 0u32;
        for i in 0..self.x.len() {
            // Anti-commutation: X_a Z_b + Z_a X_b (mod 2)
            let ac = (self.x[i] & other.z[i]) ^ (self.z[i] & other.x[i]);
            anticommute_count += ac.count_ones();
        }

        // Commutes if even number of anti-commuting positions
        anticommute_count.is_multiple_of(2)
    }

    /// Check if operator is identity
    pub fn is_identity(&self) -> bool {
        self.x.iter().all(|&w| w == 0) && self.z.iter().all(|&w| w == 0) && self.phase == 0
    }

    /// Get the weight (number of non-identity Paulis)
    pub fn weight(&self) -> usize {
        let mut count = 0;
        for q in 0..self.n_qubits {
            if self.get_x(q) || self.get_z(q) {
                count += 1;
            }
        }
        count
    }
}

/// Phase contribution from multiplying two single-qubit Paulis
/// I=0, X=1, Z=2, Y=3
#[allow(dead_code)]
#[inline]
fn pauli_phase(p1: u8, p2: u8) -> i32 {
    // Multiplication table phases (in units of i)
    // Rows: first Pauli, Cols: second Pauli
    // Result phase when computing p1 * p2
    const PHASE_TABLE: [[i32; 4]; 4] = [
        [0, 0, 0, 0], // I * {I,X,Z,Y}
        [0, 0, 1, 3], // X * {I,X,Z,Y}: XZ = iY, XY = -iZ
        [0, 3, 0, 1], // Z * {I,X,Z,Y}: ZX = -iY, ZY = iX
        [0, 1, 3, 0], // Y * {I,X,Z,Y}: YX = iZ, YZ = -iX
    ];
    PHASE_TABLE[p1 as usize][p2 as usize]
}

impl fmt::Display for PauliOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let phase_str = match self.phase {
            0 => "+",
            1 => "+i",
            2 => "-",
            3 => "-i",
            _ => "?",
        };
        write!(f, "{}", phase_str)?;

        for q in 0..self.n_qubits {
            let c = match self.pauli_at(q) {
                0 => 'I',
                1 => 'X',
                2 => 'Z',
                3 => 'Y',
                _ => '?',
            };
            write!(f, "{}", c)?;
        }
        Ok(())
    }
}

/// Stabilizer Tableau: Aaronson-Gottesman representation
///
/// Represents an n-qubit stabilizer state using:
/// - n stabilizer generators (rows 0 to n-1)
/// - n destabilizer generators (rows n to 2n-1)
///
/// Each generator is a Pauli operator stored as x, z bit vectors plus phase.
/// Total memory: O(n²) bits
#[derive(Clone)]
pub struct StabilizerTableau {
    /// Number of qubits
    n_qubits: usize,
    /// Stabilizer generators (first n) and destabilizers (next n)
    /// Total 2n generators
    generators: Vec<PauliOperator>,
}

impl StabilizerTableau {
    /// Create tableau for |0⟩^⊗n state
    ///
    /// Stabilizers: Z_i for each qubit (eigenvalue +1)
    /// Destabilizers: X_i for each qubit
    pub fn new(n_qubits: usize) -> Self {
        let mut generators = Vec::with_capacity(2 * n_qubits);

        // Stabilizers: Z_0, Z_1, ..., Z_{n-1}
        for i in 0..n_qubits {
            generators.push(PauliOperator::z(n_qubits, i));
        }

        // Destabilizers: X_0, X_1, ..., X_{n-1}
        for i in 0..n_qubits {
            generators.push(PauliOperator::x(n_qubits, i));
        }

        Self {
            n_qubits,
            generators,
        }
    }

    /// Create tableau for |+⟩^⊗n state
    ///
    /// Stabilizers: X_i for each qubit
    /// Destabilizers: Z_i for each qubit
    pub fn new_plus_state(n_qubits: usize) -> Self {
        let mut generators = Vec::with_capacity(2 * n_qubits);

        // Stabilizers: X_0, X_1, ..., X_{n-1}
        for i in 0..n_qubits {
            generators.push(PauliOperator::x(n_qubits, i));
        }

        // Destabilizers: Z_0, Z_1, ..., Z_{n-1}
        for i in 0..n_qubits {
            generators.push(PauliOperator::z(n_qubits, i));
        }

        Self {
            n_qubits,
            generators,
        }
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Get stabilizer generator i (0 <= i < n)
    pub fn stabilizer(&self, i: usize) -> &PauliOperator {
        &self.generators[i]
    }

    /// Get destabilizer generator i (0 <= i < n)
    pub fn destabilizer(&self, i: usize) -> &PauliOperator {
        &self.generators[self.n_qubits + i]
    }

    // =========================================================================
    // CLIFFORD GATES
    // =========================================================================

    /// Apply Hadamard gate to qubit
    /// H: X ↔ Z
    pub fn hadamard(&mut self, qubit: usize) {
        for row in &mut self.generators {
            let x = row.get_x(qubit);
            let z = row.get_z(qubit);
            row.set_x(qubit, z);
            row.set_z(qubit, x);

            // Phase update: if both X and Z, flip phase by 2 (multiply by -1)
            if x && z {
                row.phase = (row.phase + 2) % 4;
            }
        }
    }

    /// Apply S gate (phase gate) to qubit
    /// S: X → Y, Z → Z
    pub fn phase_gate(&mut self, qubit: usize) {
        for row in &mut self.generators {
            let x = row.get_x(qubit);
            let z = row.get_z(qubit);

            if x {
                // X → Y = iXZ, so add Z and phase += 1
                row.set_z(qubit, !z);
                if !z {
                    row.phase = (row.phase + 1) % 4;
                } else {
                    row.phase = (row.phase + 3) % 4; // -i from XZ→-iY
                }
            }
        }
    }

    /// Apply S† (S-dagger) gate to qubit
    /// S†: X → -Y, Z → Z
    pub fn phase_dag(&mut self, qubit: usize) {
        for row in &mut self.generators {
            let x = row.get_x(qubit);
            let z = row.get_z(qubit);

            if x {
                row.set_z(qubit, !z);
                if !z {
                    row.phase = (row.phase + 3) % 4; // -i
                } else {
                    row.phase = (row.phase + 1) % 4;
                }
            }
        }
    }

    /// Apply CNOT gate (control, target)
    /// CNOT: X_c → X_c X_t, Z_t → Z_c Z_t
    pub fn cnot(&mut self, control: usize, target: usize) {
        for row in &mut self.generators {
            let xc = row.get_x(control);
            let zc = row.get_z(control);
            let xt = row.get_x(target);
            let zt = row.get_z(target);

            // X_control propagates to target
            if xc {
                row.set_x(target, xt ^ true);
            }

            // Z_target propagates to control
            if zt {
                row.set_z(control, zc ^ true);
            }

            // Phase update for Y terms
            if xc && zt && (xt == zc) {
                row.phase = (row.phase + 2) % 4;
            }
        }
    }

    /// Apply CZ gate (controlled-Z)
    /// CZ: X_a → X_a Z_b, X_b → Z_a X_b
    pub fn cz(&mut self, qubit_a: usize, qubit_b: usize) {
        for row in &mut self.generators {
            let xa = row.get_x(qubit_a);
            let za = row.get_z(qubit_a);
            let xb = row.get_x(qubit_b);
            let zb = row.get_z(qubit_b);

            // X_a picks up Z_b
            if xa {
                row.set_z(qubit_b, zb ^ true);
            }

            // X_b picks up Z_a
            if xb {
                row.set_z(qubit_a, za ^ true);
            }

            // Phase correction
            if xa && xb {
                row.phase = (row.phase + 2) % 4;
            }
        }
    }

    /// Apply Pauli X gate
    pub fn pauli_x(&mut self, qubit: usize) {
        for row in &mut self.generators {
            if row.get_z(qubit) {
                row.phase = (row.phase + 2) % 4;
            }
        }
    }

    /// Apply Pauli Y gate
    pub fn pauli_y(&mut self, qubit: usize) {
        for row in &mut self.generators {
            let x = row.get_x(qubit);
            let z = row.get_z(qubit);
            if x != z {
                row.phase = (row.phase + 2) % 4;
            }
        }
    }

    /// Apply Pauli Z gate
    pub fn pauli_z(&mut self, qubit: usize) {
        for row in &mut self.generators {
            if row.get_x(qubit) {
                row.phase = (row.phase + 2) % 4;
            }
        }
    }

    // =========================================================================
    // MEASUREMENT
    // =========================================================================

    /// Measure qubit in Z basis
    /// Returns (measurement_result, random_flag)
    /// - measurement_result: 0 or 1
    /// - random_flag: true if outcome was random, false if deterministic
    pub fn measure_z(&mut self, qubit: usize, random_bit: bool) -> (u8, bool) {
        // Find a stabilizer that anti-commutes with Z_qubit (i.e., has X on qubit)
        let mut anticommuting_idx = None;
        for i in 0..self.n_qubits {
            if self.generators[i].get_x(qubit) {
                anticommuting_idx = Some(i);
                break;
            }
        }

        match anticommuting_idx {
            Some(p) => {
                // Random outcome case
                // Row-reduce: multiply all other anticommuting rows by row p
                for i in 0..2 * self.n_qubits {
                    if i != p && self.generators[i].get_x(qubit) {
                        self.rowmult(i, p);
                    }
                }

                // Move destabilizer to stabilizer position
                self.generators.swap(p, self.n_qubits + p);

                // Set stabilizer p to ±Z_qubit based on measurement outcome
                let result = if random_bit { 1 } else { 0 };
                self.generators[p] = PauliOperator::z(self.n_qubits, qubit);
                if result == 1 {
                    self.generators[p].phase = 2; // -Z for |1⟩ eigenstate
                }

                (result, true)
            }
            None => {
                // Deterministic outcome case - optimized phase-only calculation
                // We only need the phase, not the full Pauli operator
                let mut total_phase = 0u32;
                let n_words = self.n_qubits.div_ceil(64);

                // Track accumulated x,z for phase calculation
                let mut acc_x = vec![0u64; n_words];
                let mut acc_z = vec![0u64; n_words];

                for i in 0..self.n_qubits {
                    if self.generators[self.n_qubits + i].get_x(qubit) {
                        // Multiply accumulator by stabilizer i
                        let stab = &self.generators[i];
                        let mut phase_add = 0u32;
                        let mut phase_sub = 0u32;

                        for w in 0..n_words {
                            phase_add += (acc_x[w] & stab.z[w]).count_ones();
                            phase_sub += (acc_z[w] & stab.x[w]).count_ones();
                            acc_x[w] ^= stab.x[w];
                            acc_z[w] ^= stab.z[w];
                        }

                        let phase_contrib = (phase_add + 3 * phase_sub) % 4;
                        total_phase = (total_phase + stab.phase as u32 + phase_contrib) % 4;
                    }
                }

                let result = if total_phase == 0 || total_phase == 1 {
                    0
                } else {
                    1
                };
                (result, false)
            }
        }
    }

    /// Multiply row i by row j: generators[i] = generators[i] * generators[j]
    /// Optimized: in-place phase calculation, no allocation
    #[inline]
    fn rowmult(&mut self, i: usize, j: usize) {
        let n_words = self.generators[j].x.len();

        // Compute phase contribution and XOR in place
        let mut phase_add = 0u32;
        let mut phase_sub = 0u32;

        for w in 0..n_words {
            let xi = self.generators[i].x[w];
            let zi = self.generators[i].z[w];
            let xj = self.generators[j].x[w];
            let zj = self.generators[j].z[w];

            phase_add += (xi & zj).count_ones();
            phase_sub += (zi & xj).count_ones();

            self.generators[i].x[w] = xi ^ xj;
            self.generators[i].z[w] = zi ^ zj;
        }

        let phase_contrib = (phase_add + 3 * phase_sub) % 4;
        let new_phase =
            (self.generators[i].phase as u32 + self.generators[j].phase as u32 + phase_contrib) % 4;
        self.generators[i].phase = new_phase as u8;
    }

    // =========================================================================
    // STATE INSPECTION
    // =========================================================================

    /// Check if a Pauli operator is in the stabilizer group
    pub fn is_stabilized_by(&self, op: &PauliOperator) -> bool {
        // The operator must commute with all stabilizers
        for i in 0..self.n_qubits {
            if !op.commutes_with(&self.generators[i]) {
                return false;
            }
        }
        true
    }

    /// Get the expectation value of a Pauli operator: ⟨ψ|P|ψ⟩
    /// Returns +1, -1, or 0
    pub fn expectation(&self, op: &PauliOperator) -> i8 {
        // Check if op is in stabilizer group
        if !self.is_stabilized_by(op) {
            return 0; // Anti-commutes with some stabilizer
        }

        // Compute the phase by expressing op in terms of stabilizers
        // This is simplified - full implementation needs Gaussian elimination
        // For now, return 1 if identity, else 0
        if op.is_identity() {
            return 1;
        }

        // TODO: Full implementation with Gaussian elimination
        0
    }

    /// Create GHZ state: (|00...0⟩ + |11...1⟩)/√2
    pub fn create_ghz(&mut self) {
        // Start from |0...0⟩
        *self = Self::new(self.n_qubits);

        // H on qubit 0
        self.hadamard(0);

        // CNOT chain
        for i in 1..self.n_qubits {
            self.cnot(0, i);
        }
    }

    /// Create graph state from adjacency matrix
    /// Graph state: H on all qubits, then CZ for each edge
    pub fn create_graph_state(&mut self, edges: &[(usize, usize)]) {
        // Start from |+⟩^⊗n
        *self = Self::new_plus_state(self.n_qubits);

        // Apply CZ for each edge
        for &(a, b) in edges {
            self.cz(a, b);
        }
    }

    // =========================================================================
    // BATCH OPERATIONS (Performance Optimization)
    // =========================================================================

    /// Apply multiple CZ gates in a single pass through generators.
    /// This is O(n * k) instead of O(n * k) individual calls, but with better cache locality.
    /// For large numbers of CZ gates, this significantly reduces overhead.
    #[inline]
    pub fn batch_cz(&mut self, pairs: &[(usize, usize)]) {
        if pairs.is_empty() {
            return;
        }

        let _n_words = self.n_qubits.div_ceil(64);

        for row in &mut self.generators {
            let mut phase_delta = 0u8;

            for &(qubit_a, qubit_b) in pairs {
                let word_a = qubit_a / 64;
                let bit_a = qubit_a % 64;
                let word_b = qubit_b / 64;
                let bit_b = qubit_b % 64;

                let xa = (row.x[word_a] >> bit_a) & 1 != 0;
                let xb = (row.x[word_b] >> bit_b) & 1 != 0;

                // X_a picks up Z_b
                if xa {
                    row.z[word_b] ^= 1u64 << bit_b;
                }

                // X_b picks up Z_a
                if xb {
                    row.z[word_a] ^= 1u64 << bit_a;
                }

                // Phase correction
                if xa && xb {
                    phase_delta = (phase_delta + 2) % 4;
                }
            }

            row.phase = (row.phase + phase_delta) % 4;
        }
    }

    /// Apply Hadamard gates to multiple qubits in a single pass.
    #[inline]
    pub fn batch_hadamard(&mut self, qubits: &[usize]) {
        if qubits.is_empty() {
            return;
        }

        for row in &mut self.generators {
            let mut phase_delta = 0u8;

            for &qubit in qubits {
                let word = qubit / 64;
                let bit = qubit % 64;

                let x = (row.x[word] >> bit) & 1 != 0;
                let z = (row.z[word] >> bit) & 1 != 0;

                // Swap X and Z
                if x && !z {
                    row.x[word] &= !(1u64 << bit);
                    row.z[word] |= 1u64 << bit;
                } else if !x && z {
                    row.x[word] |= 1u64 << bit;
                    row.z[word] &= !(1u64 << bit);
                }
                // If both or neither, no change needed (XOR would be same)

                // Phase update: if both X and Z, flip phase by 2
                if x && z {
                    phase_delta = (phase_delta + 2) % 4;
                }
            }

            row.phase = (row.phase + phase_delta) % 4;
        }
    }

    /// Apply phase gates to multiple qubits in a single pass.
    #[inline]
    pub fn batch_phase(&mut self, qubits: &[usize]) {
        if qubits.is_empty() {
            return;
        }

        for row in &mut self.generators {
            let mut phase_delta = 0u8;

            for &qubit in qubits {
                let word = qubit / 64;
                let bit = qubit % 64;

                let x = (row.x[word] >> bit) & 1 != 0;
                let z = (row.z[word] >> bit) & 1 != 0;

                // S gate: X → Y = XZ, so X picks up Z
                if x {
                    row.z[word] ^= 1u64 << bit;
                }

                // Phase: if X but not Z, add phase
                if x && !z {
                    phase_delta = (phase_delta + 1) % 4;
                }
            }

            row.phase = (row.phase + phase_delta) % 4;
        }
    }

    // =========================================================================
    // CIRCUIT EXECUTION (Phase 77.2)
    // =========================================================================

    /// Check if a qubit index is valid for this tableau.
    #[inline]
    fn check_qubit(&self, qubit: usize) -> Result<(), StabilizerError> {
        if qubit >= self.n_qubits {
            Err(StabilizerError::QubitOutOfRange {
                qubit,
                n_qubits: self.n_qubits,
            })
        } else {
            Ok(())
        }
    }

    /// Check if an angle is approximately a Clifford angle (multiple of pi/2).
    #[inline]
    fn is_clifford_angle(angle: f32) -> bool {
        let normalized = angle.abs() % (PI / 2.0);
        normalized < 1e-5 || (PI / 2.0 - normalized).abs() < 1e-5
    }

    /// Classify an angle to determine which Clifford rotation it represents.
    /// Returns the number of pi/2 rotations (0, 1, 2, or 3) mod 4.
    #[inline]
    fn classify_clifford_angle(angle: f32) -> Option<u8> {
        // Normalize angle to [0, 2*pi)
        let normalized = angle.rem_euclid(2.0 * PI);

        // Check each Clifford angle
        const TOL: f32 = 1e-4;
        if normalized < TOL || (2.0 * PI - normalized) < TOL {
            Some(0) // 0 or 2*pi -> identity
        } else if (normalized - PI / 2.0).abs() < TOL {
            Some(1) // pi/2 -> S gate
        } else if (normalized - PI).abs() < TOL {
            Some(2) // pi -> Z gate (S^2)
        } else if (normalized - 3.0 * PI / 2.0).abs() < TOL {
            Some(3) // 3*pi/2 -> S^dagger
        } else {
            None // Not a Clifford angle
        }
    }

    /// Execute a quantum circuit on the stabilizer tableau.
    ///
    /// This method applies each gate in the circuit to the stabilizer state.
    /// Only Clifford gates are supported; non-Clifford gates will return an error.
    ///
    /// # Supported Gates
    /// - H(q) -> hadamard(q)
    /// - X(q) -> pauli_x(q)
    /// - Y(q) -> pauli_y(q)
    /// - Z(q) -> pauli_z(q)
    /// - Phase(q, angle) where angle is Clifford -> phase_gate or phase_dag
    /// - CNot(c, t) -> cnot(c, t)
    /// - CZ(a, b) -> cz(a, b)
    /// - Swap(a, b) -> 3 CNOTs decomposition
    /// - Rz/Rx/Ry with Clifford angles
    ///
    /// # Arguments
    /// * `circuit` - A slice of QGate representing the circuit to execute
    ///
    /// # Returns
    /// `Ok(())` if all gates were successfully applied, or `Err(StabilizerError)` if
    /// a non-Clifford gate was encountered or a qubit index was out of range.
    ///
    /// # Example
    /// ```ignore
    /// use engine::qec::StabilizerTableau;
    /// use logic_fabric_core::quantum::QGate;
    ///
    /// let mut tableau = StabilizerTableau::new(2);
    /// let bell_circuit = vec![
    ///     QGate::H(0),
    ///     QGate::CNot(0, 1),
    /// ];
    /// tableau.execute_circuit(&bell_circuit).expect("Bell state preparation");
    /// ```
    pub fn execute_circuit(&mut self, circuit: &[QGate]) -> Result<(), StabilizerError> {
        for gate in circuit {
            self.execute_gate(gate)?;
        }
        Ok(())
    }

    /// Execute a single quantum gate on the stabilizer tableau.
    fn execute_gate(&mut self, gate: &QGate) -> Result<(), StabilizerError> {
        match gate {
            // Single-qubit Clifford gates
            QGate::H(q) => {
                self.check_qubit(*q as usize)?;
                self.hadamard(*q as usize);
            }
            QGate::X(q) => {
                self.check_qubit(*q as usize)?;
                self.pauli_x(*q as usize);
            }
            QGate::Y(q) => {
                self.check_qubit(*q as usize)?;
                self.pauli_y(*q as usize);
            }
            QGate::Z(q) => {
                self.check_qubit(*q as usize)?;
                self.pauli_z(*q as usize);
            }

            // Phase gate with Clifford angle
            QGate::Phase(q, angle) => {
                self.check_qubit(*q as usize)?;
                match Self::classify_clifford_angle(*angle) {
                    Some(0) => {}                            // Identity, do nothing
                    Some(1) => self.phase_gate(*q as usize), // S gate (pi/2)
                    Some(2) => self.pauli_z(*q as usize),    // Z gate (pi)
                    Some(3) => self.phase_dag(*q as usize),  // S^dag (3*pi/2 or -pi/2)
                    Some(_) | None => {
                        return Err(StabilizerError::NonCliffordGate(format!(
                            "Phase({}, {}) is not a Clifford gate",
                            q, angle
                        )));
                    }
                }
            }

            // Two-qubit Clifford gates
            QGate::CNot(c, t) => {
                self.check_qubit(*c as usize)?;
                self.check_qubit(*t as usize)?;
                self.cnot(*c as usize, *t as usize);
            }
            QGate::CZ(a, b) => {
                self.check_qubit(*a as usize)?;
                self.check_qubit(*b as usize)?;
                self.cz(*a as usize, *b as usize);
            }

            // Swap decomposed as 3 CNOTs: SWAP(a,b) = CNOT(a,b) CNOT(b,a) CNOT(a,b)
            QGate::Swap(a, b) => {
                self.check_qubit(*a as usize)?;
                self.check_qubit(*b as usize)?;
                self.cnot(*a as usize, *b as usize);
                self.cnot(*b as usize, *a as usize);
                self.cnot(*a as usize, *b as usize);
            }

            // Rotation gates with Clifford angles
            QGate::Rz(q, angle) => {
                self.check_qubit(*q as usize)?;
                // Rz(theta) = diag(e^{-i*theta/2}, e^{i*theta/2})
                // At Clifford angles, Rz acts like: 0->I, pi/2->S, pi->Z, 3pi/2->S^dag
                // But Rz has different global phase than Phase gate
                // For stabilizer simulation, global phase is irrelevant
                match Self::classify_clifford_angle(*angle) {
                    Some(0) => {}                            // Identity
                    Some(1) => self.phase_gate(*q as usize), // S
                    Some(2) => self.pauli_z(*q as usize),    // Z
                    Some(3) => self.phase_dag(*q as usize),  // S^dag
                    Some(_) | None => {
                        return Err(StabilizerError::NonCliffordGate(format!(
                            "Rz({}, {}) is not a Clifford gate",
                            q, angle
                        )));
                    }
                }
            }

            QGate::Rx(q, angle) => {
                self.check_qubit(*q as usize)?;
                // Rx(theta) = H Rz(theta) H
                // At Clifford angles: 0->I, pi/2->SX (sqrt X), pi->X, 3pi/2->SX^dag
                // We can implement using H-Rz-H conjugation
                match Self::classify_clifford_angle(*angle) {
                    Some(0) => {} // Identity
                    Some(1) => {
                        // Rx(pi/2) = sqrt(X): H S H
                        self.hadamard(*q as usize);
                        self.phase_gate(*q as usize);
                        self.hadamard(*q as usize);
                    }
                    Some(2) => self.pauli_x(*q as usize), // X
                    Some(3) => {
                        // Rx(-pi/2) = sqrt(X)^dag: H S^dag H
                        self.hadamard(*q as usize);
                        self.phase_dag(*q as usize);
                        self.hadamard(*q as usize);
                    }
                    Some(_) | None => {
                        return Err(StabilizerError::NonCliffordGate(format!(
                            "Rx({}, {}) is not a Clifford gate",
                            q, angle
                        )));
                    }
                }
            }

            QGate::Ry(q, angle) => {
                self.check_qubit(*q as usize)?;
                // Ry(theta) at Clifford angles
                match Self::classify_clifford_angle(*angle) {
                    Some(0) => {} // Identity
                    Some(1) => {
                        // Ry(pi/2): S^dag H S
                        self.phase_dag(*q as usize);
                        self.hadamard(*q as usize);
                        self.phase_gate(*q as usize);
                    }
                    Some(2) => self.pauli_y(*q as usize), // Y (up to global phase)
                    Some(3) => {
                        // Ry(-pi/2): S^dag H S^dag = Ry(3pi/2)
                        self.phase_dag(*q as usize);
                        self.hadamard(*q as usize);
                        self.phase_dag(*q as usize);
                    }
                    Some(_) | None => {
                        return Err(StabilizerError::NonCliffordGate(format!(
                            "Ry({}, {}) is not a Clifford gate",
                            q, angle
                        )));
                    }
                }
            }

            // U3 gate: only Clifford if all angles are Clifford
            QGate::U3(q, theta, phi, lambda) => {
                self.check_qubit(*q as usize)?;
                if !Self::is_clifford_angle(*theta)
                    || !Self::is_clifford_angle(*phi)
                    || !Self::is_clifford_angle(*lambda)
                {
                    return Err(StabilizerError::NonCliffordGate(format!(
                        "U3({}, {}, {}, {}) contains non-Clifford angles",
                        q, theta, phi, lambda
                    )));
                }
                // U3(theta, phi, lambda) = Rz(phi) Ry(theta) Rz(lambda)
                // Execute as decomposition
                self.execute_gate(&QGate::Rz(*q, *lambda))?;
                self.execute_gate(&QGate::Ry(*q, *theta))?;
                self.execute_gate(&QGate::Rz(*q, *phi))?;
            }

            // Controlled rotations with Clifford angles
            QGate::CPhase(c, t, angle) => {
                self.check_qubit(*c as usize)?;
                self.check_qubit(*t as usize)?;
                match Self::classify_clifford_angle(*angle) {
                    Some(0) => {} // Identity
                    Some(2) => {
                        // CPhase(pi) = CZ
                        self.cz(*c as usize, *t as usize);
                    }
                    // CPhase(pi/2) and CPhase(3pi/2) are non-Clifford (CS and CS^dag gates)
                    Some(_) | None => {
                        return Err(StabilizerError::NonCliffordGate(format!(
                            "CPhase({}, {}, {}) is not a Clifford gate",
                            c, t, angle
                        )));
                    }
                }
            }

            QGate::CRz(c, t, angle) => {
                self.check_qubit(*c as usize)?;
                self.check_qubit(*t as usize)?;
                // Only CRz(pi) = CZ is Clifford
                match Self::classify_clifford_angle(*angle) {
                    Some(0) => {}                                 // Identity
                    Some(2) => self.cz(*c as usize, *t as usize), // CZ
                    _ => {
                        return Err(StabilizerError::NonCliffordGate(format!(
                            "CRz({}, {}, {}) is not a Clifford gate",
                            c, t, angle
                        )));
                    }
                }
            }

            // Non-Clifford gates
            QGate::T(q) => {
                return Err(StabilizerError::NonCliffordGate(format!(
                    "T({}) is not a Clifford gate",
                    q
                )));
            }
            QGate::Tdg(q) => {
                return Err(StabilizerError::NonCliffordGate(format!(
                    "Tdg({}) is not a Clifford gate",
                    q
                )));
            }

            // Multi-qubit gates that would need decomposition
            QGate::Toffoli(a, b, c) => {
                return Err(StabilizerError::NonCliffordGate(format!(
                    "Toffoli({}, {}, {}) is not a Clifford gate (requires T gates in decomposition)",
                    a, b, c
                )));
            }
            QGate::CCZ(a, b, c) => {
                return Err(StabilizerError::NonCliffordGate(format!(
                    "CCZ({}, {}, {}) is not a Clifford gate (requires T gates in decomposition)",
                    a, b, c
                )));
            }

            // Measurement - not handled by execute_circuit (use measure_z instead)
            QGate::Measure(q) => {
                // For now, we skip measurements in circuit execution
                // Users should use measure_z() for measurement
                self.check_qubit(*q as usize)?;
                // Do nothing - measurement is handled separately
            }
        }
        Ok(())
    }

    // =========================================================================
    // NOISY CIRCUIT EXECUTION (Phase 78.2)
    // =========================================================================

    /// Apply a Pauli error to a qubit.
    ///
    /// This is used internally by noisy execution methods to inject errors.
    fn apply_pauli(&mut self, qubit: usize, pauli: Pauli) {
        match pauli {
            Pauli::I => {} // Identity - no operation
            Pauli::X => self.pauli_x(qubit),
            Pauli::Y => self.pauli_y(qubit),
            Pauli::Z => self.pauli_z(qubit),
        }
    }

    /// Execute a single gate with optional noise injection.
    ///
    /// This method first applies the ideal gate, then samples and applies
    /// errors according to the provided error model.
    ///
    /// # Noise Model
    ///
    /// - Single-qubit gates: After execution, with probability p_single,
    ///   a random Pauli (X, Y, or Z) is applied to the qubit.
    /// - Two-qubit gates: After execution, with probability p_two,
    ///   a random non-II Pauli pair is applied to both qubits.
    ///
    /// # Arguments
    /// * `gate` - The quantum gate to execute
    /// * `error_model` - Optional error model for noise injection
    /// * `rng` - Random number generator for sampling errors
    ///
    /// # Returns
    /// `Ok(Some(GateError))` describing any error that was injected,
    /// or `Err(StabilizerError)` if the gate is non-Clifford.
    ///
    /// # Example
    /// ```ignore
    /// use engine::qec::{StabilizerTableau, GateErrorModel, SimpleRng};
    /// use logic_fabric_core::quantum::QGate;
    ///
    /// let mut tableau = StabilizerTableau::new(2);
    /// let mut rng = SimpleRng::new(42);
    /// let model = GateErrorModel::superconducting_ballpark();
    ///
    /// let error = tableau.execute_gate_noisy(
    ///     &QGate::H(0),
    ///     Some(&model),
    ///     &mut rng
    /// ).expect("Clifford gate");
    /// ```
    pub fn execute_gate_noisy(
        &mut self,
        gate: &QGate,
        error_model: Option<&GateErrorModel>,
        rng: &mut SimpleRng,
    ) -> Result<Option<GateError>, StabilizerError> {
        // 1. Execute the ideal gate
        self.execute_gate(gate)?;

        // 2. Sample and apply error if model provided
        if let Some(model) = error_model {
            let error = match gate {
                // Single-qubit gates
                QGate::H(q)
                | QGate::X(q)
                | QGate::Y(q)
                | QGate::Z(q)
                | QGate::T(q)
                | QGate::Tdg(q) => {
                    // Note: T/Tdg will have failed in execute_gate, but we handle for completeness
                    if let Some(pauli) = sample_single_qubit_error(model.p_single, rng) {
                        self.apply_pauli(*q as usize, pauli);
                        GateError::Single {
                            qubit: *q as usize,
                            pauli,
                        }
                    } else {
                        GateError::None
                    }
                }
                QGate::Phase(q, _) | QGate::Rx(q, _) | QGate::Ry(q, _) | QGate::Rz(q, _) => {
                    if let Some(pauli) = sample_single_qubit_error(model.p_single, rng) {
                        self.apply_pauli(*q as usize, pauli);
                        GateError::Single {
                            qubit: *q as usize,
                            pauli,
                        }
                    } else {
                        GateError::None
                    }
                }
                QGate::U3(q, _, _, _) => {
                    if let Some(pauli) = sample_single_qubit_error(model.p_single, rng) {
                        self.apply_pauli(*q as usize, pauli);
                        GateError::Single {
                            qubit: *q as usize,
                            pauli,
                        }
                    } else {
                        GateError::None
                    }
                }
                // Two-qubit gates
                QGate::CNot(c, t) | QGate::CZ(c, t) => {
                    if let Some((p1, p2)) = sample_two_qubit_error(model.p_two, rng) {
                        self.apply_pauli(*c as usize, p1);
                        self.apply_pauli(*t as usize, p2);
                        GateError::TwoQubit {
                            q1: *c as usize,
                            p1,
                            q2: *t as usize,
                            p2,
                        }
                    } else {
                        GateError::None
                    }
                }
                QGate::CPhase(c, t, _) | QGate::CRz(c, t, _) => {
                    if let Some((p1, p2)) = sample_two_qubit_error(model.p_two, rng) {
                        self.apply_pauli(*c as usize, p1);
                        self.apply_pauli(*t as usize, p2);
                        GateError::TwoQubit {
                            q1: *c as usize,
                            p1,
                            q2: *t as usize,
                            p2,
                        }
                    } else {
                        GateError::None
                    }
                }
                QGate::Swap(a, b) => {
                    // SWAP is decomposed into 3 CNOTs, so apply 2Q error
                    if let Some((p1, p2)) = sample_two_qubit_error(model.p_two, rng) {
                        self.apply_pauli(*a as usize, p1);
                        self.apply_pauli(*b as usize, p2);
                        GateError::TwoQubit {
                            q1: *a as usize,
                            p1,
                            q2: *b as usize,
                            p2,
                        }
                    } else {
                        GateError::None
                    }
                }
                // Three-qubit gates (will have failed in execute_gate, but handle for completeness)
                QGate::Toffoli(a, b, c) | QGate::CCZ(a, b, c) => {
                    // Apply depolarizing to all three qubits independently
                    let mut errors = Vec::new();
                    for &q in &[*a, *b, *c] {
                        if let Some(pauli) = sample_single_qubit_error(model.p_single, rng) {
                            self.apply_pauli(q as usize, pauli);
                            errors.push((q as usize, pauli));
                        }
                    }
                    if !errors.is_empty() {
                        // Return first error for simplicity (full tracking would need a different type)
                        GateError::Single {
                            qubit: errors[0].0,
                            pauli: errors[0].1,
                        }
                    } else {
                        GateError::None
                    }
                }
                // Measurement - no gate error, measurement error handled separately
                QGate::Measure(_) => GateError::None,
            };
            return Ok(Some(error));
        }

        Ok(None)
    }

    /// Execute a circuit with noise injection.
    ///
    /// Applies each gate in the circuit with optional noise, collecting all
    /// errors that occur during execution.
    ///
    /// # Arguments
    /// * `circuit` - The quantum circuit to execute
    /// * `error_model` - Optional error model for noise injection
    /// * `rng` - Random number generator for sampling errors
    ///
    /// # Returns
    /// A vector of all non-trivial errors that occurred during execution.
    ///
    /// # Example
    /// ```ignore
    /// use engine::qec::{StabilizerTableau, GateErrorModel, SimpleRng};
    /// use logic_fabric_core::quantum::QGate;
    ///
    /// let mut tableau = StabilizerTableau::new(3);
    /// let mut rng = SimpleRng::new(42);
    /// let model = GateErrorModel::superconducting_ballpark();
    ///
    /// let circuit = vec![
    ///     QGate::H(0),
    ///     QGate::CNot(0, 1),
    ///     QGate::CNot(1, 2),
    /// ];
    ///
    /// let errors = tableau.execute_circuit_noisy(
    ///     &circuit,
    ///     Some(&model),
    ///     &mut rng
    /// ).expect("All Clifford gates");
    ///
    /// println!("Injected {} errors", errors.len());
    /// ```
    pub fn execute_circuit_noisy(
        &mut self,
        circuit: &[QGate],
        error_model: Option<&GateErrorModel>,
        rng: &mut SimpleRng,
    ) -> Result<Vec<GateError>, StabilizerError> {
        let mut errors = Vec::new();
        for gate in circuit {
            if let Some(error) = self.execute_gate_noisy(gate, error_model, rng)? {
                if !matches!(error, GateError::None) {
                    errors.push(error);
                }
            }
        }
        Ok(errors)
    }

    /// Measure a qubit in the Z basis with optional measurement error.
    ///
    /// This extends the standard measure_z with measurement error injection.
    /// With probability p_meas, the classical measurement outcome is flipped.
    ///
    /// # Arguments
    /// * `qubit` - The qubit to measure
    /// * `random_bit` - The random bit for measurement outcome (if random)
    /// * `error_model` - Optional error model for measurement error
    /// * `rng` - Random number generator for error sampling
    ///
    /// # Returns
    /// A tuple of (measurement_result, was_random, had_error)
    pub fn measure_z_noisy(
        &mut self,
        qubit: usize,
        random_bit: bool,
        error_model: Option<&GateErrorModel>,
        rng: &mut SimpleRng,
    ) -> (u8, bool, bool) {
        // Perform the ideal measurement
        let (mut result, random) = self.measure_z(qubit, random_bit);

        // Apply measurement error if model provided
        let had_error = if let Some(model) = error_model {
            if sample_measurement_error(model.p_meas, rng) {
                result = 1 - result;
                true
            } else {
                false
            }
        } else {
            false
        };

        (result, random, had_error)
    }
}

impl fmt::Display for StabilizerTableau {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "StabilizerTableau ({} qubits):", self.n_qubits)?;
        writeln!(f, "Stabilizers:")?;
        for i in 0..self.n_qubits {
            writeln!(f, "  S{}: {}", i, self.generators[i])?;
        }
        writeln!(f, "Destabilizers:")?;
        for i in 0..self.n_qubits {
            writeln!(f, "  D{}: {}", i, self.generators[self.n_qubits + i])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_state() {
        let tab = StabilizerTableau::new(3);
        // Stabilizers should be Z0, Z1, Z2
        assert!(tab.stabilizer(0).get_z(0));
        assert!(tab.stabilizer(1).get_z(1));
        assert!(tab.stabilizer(2).get_z(2));
    }

    #[test]
    fn test_hadamard() {
        let mut tab = StabilizerTableau::new(1);
        // |0⟩ stabilized by Z
        assert!(tab.stabilizer(0).get_z(0));
        assert!(!tab.stabilizer(0).get_x(0));

        // After H, |+⟩ stabilized by X
        tab.hadamard(0);
        assert!(!tab.stabilizer(0).get_z(0));
        assert!(tab.stabilizer(0).get_x(0));
    }

    #[test]
    fn test_cnot() {
        let mut tab = StabilizerTableau::new(2);
        tab.hadamard(0);
        tab.cnot(0, 1);
        // Should create Bell state (|00⟩ + |11⟩)/√2
        // Stabilizers: XX, ZZ
        let s0 = tab.stabilizer(0);
        let s1 = tab.stabilizer(1);

        // One stabilizer should be XX, one should be ZZ
        let is_xx = s0.get_x(0) && s0.get_x(1) && !s0.get_z(0) && !s0.get_z(1);
        let is_zz = s1.get_z(0) && s1.get_z(1) && !s1.get_x(0) && !s1.get_x(1);

        assert!(is_xx || is_zz);
    }

    #[test]
    fn test_measurement_deterministic() {
        let mut tab = StabilizerTableau::new(1);
        // |0⟩ measured in Z gives 0 deterministically
        let (result, random) = tab.measure_z(0, false);
        assert_eq!(result, 0);
        assert!(!random);
    }

    #[test]
    fn test_measurement_random() {
        let mut tab = StabilizerTableau::new(1);
        tab.hadamard(0);
        // |+⟩ measured in Z gives random result
        let (_, random) = tab.measure_z(0, false);
        assert!(random);
    }

    #[test]
    fn test_ghz() {
        let mut tab = StabilizerTableau::new(4);
        tab.create_ghz();

        // GHZ stabilizers: XXXX, ZZII, IZZI, IIZZ
        // Check that X⊗X⊗X⊗X is a stabilizer
        let xxxx = PauliOperator {
            x: vec![0b1111],
            z: vec![0],
            phase: 0,
            n_qubits: 4,
        };
        assert!(tab.is_stabilized_by(&xxxx));
    }

    #[test]
    #[ignore] // Uses ~250GB+ RAM - run explicitly with --ignored
    fn test_million_qubits() {
        // This should complete in milliseconds!
        let n = 1_000_000;
        let start = std::time::Instant::now();
        let tab = StabilizerTableau::new(n);
        let create_time = start.elapsed();

        assert_eq!(tab.n_qubits(), n);
        println!("Created {}-qubit stabilizer state in {:?}", n, create_time);

        // Memory: ~2n² bits = 2 * 1M * 1M / 8 bytes = 250 GB for full tableau
        // But we use sparse: 2n generators * n bits each = 2 * 1M * 125KB = 250 MB
        // Actually our impl: 2n generators * (n/64) u64s = much less
    }

    // =========================================================================
    // CIRCUIT EXECUTION TESTS (Phase 77.2)
    // =========================================================================

    #[test]
    fn test_execute_circuit_bell_state() {
        // Bell state: H(0), CNOT(0,1)
        let mut tab = StabilizerTableau::new(2);
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];

        let result = tab.execute_circuit(&circuit);
        assert!(result.is_ok());

        // Verify Bell state stabilizers: XX and ZZ
        let s0 = tab.stabilizer(0);
        let s1 = tab.stabilizer(1);

        // One stabilizer should be XX, one should be ZZ
        let has_xx = (s0.get_x(0) && s0.get_x(1) && !s0.get_z(0) && !s0.get_z(1))
            || (s1.get_x(0) && s1.get_x(1) && !s1.get_z(0) && !s1.get_z(1));
        let has_zz = (s0.get_z(0) && s0.get_z(1) && !s0.get_x(0) && !s0.get_x(1))
            || (s1.get_z(0) && s1.get_z(1) && !s1.get_x(0) && !s1.get_x(1));

        assert!(has_xx, "Bell state should have XX stabilizer");
        assert!(has_zz, "Bell state should have ZZ stabilizer");
    }

    #[test]
    fn test_execute_circuit_clifford_only() {
        // H-CNOT-CZ circuit
        let mut tab = StabilizerTableau::new(3);
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1), QGate::CZ(1, 2)];

        let result = tab.execute_circuit(&circuit);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_circuit_rejects_t_gate() {
        let mut tab = StabilizerTableau::new(2);
        let circuit = vec![QGate::H(0), QGate::T(0), QGate::CNot(0, 1)];

        let result = tab.execute_circuit(&circuit);
        assert!(result.is_err());
        match result {
            Err(StabilizerError::NonCliffordGate(msg)) => {
                assert!(msg.contains("T(0)"));
            }
            _ => panic!("Expected NonCliffordGate error"),
        }
    }

    #[test]
    fn test_execute_circuit_rejects_non_clifford_rotation() {
        let mut tab = StabilizerTableau::new(2);
        let circuit = vec![
            QGate::H(0),
            QGate::Rz(0, PI / 4.0), // T gate angle - non-Clifford
            QGate::CNot(0, 1),
        ];

        let result = tab.execute_circuit(&circuit);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_circuit_accepts_clifford_rotations() {
        let mut tab = StabilizerTableau::new(2);
        let circuit = vec![
            QGate::Rz(0, PI / 2.0),    // S gate
            QGate::Rz(0, PI),          // Z gate
            QGate::Phase(0, PI / 2.0), // S gate via Phase
        ];

        let result = tab.execute_circuit(&circuit);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_circuit_swap_gate() {
        let mut tab = StabilizerTableau::new(2);

        // Prepare |10⟩ state: X on qubit 0
        tab.pauli_x(0);

        // Apply SWAP
        let circuit = vec![QGate::Swap(0, 1)];
        let result = tab.execute_circuit(&circuit);
        assert!(result.is_ok());

        // After SWAP on |10⟩, should be |01⟩
        // The state |01⟩ is stabilized by Z0 (qubit 0 is |0⟩) and -Z1 (qubit 1 is |1⟩)
        //
        // The stabilizer tableau works differently than the test originally assumed.
        // After the SWAP, we verify by measuring: qubit 0 should be |0⟩, qubit 1 should be |1⟩
        // We can verify this by checking measurement outcomes.

        // Verify via deterministic measurement:
        // Qubit 0 should measure 0 (was |1⟩, now |0⟩ after swap)
        let (result0, _) = tab.measure_z(0, false);
        assert_eq!(result0, 0, "Qubit 0 should be in |0⟩ state after SWAP");

        // After measuring qubit 0, create fresh tableau and test qubit 1
        let mut tab2 = StabilizerTableau::new(2);
        tab2.pauli_x(0); // |10⟩
        tab2.execute_circuit(&[QGate::Swap(0, 1)]).unwrap();
        let (result1, _) = tab2.measure_z(1, false);
        assert_eq!(result1, 1, "Qubit 1 should be in |1⟩ state after SWAP");
    }

    #[test]
    fn test_execute_circuit_qubit_out_of_range() {
        let mut tab = StabilizerTableau::new(2);
        let circuit = vec![QGate::H(5)]; // Qubit 5 doesn't exist

        let result = tab.execute_circuit(&circuit);
        assert!(result.is_err());
        match result {
            Err(StabilizerError::QubitOutOfRange { qubit, n_qubits }) => {
                assert_eq!(qubit, 5);
                assert_eq!(n_qubits, 2);
            }
            _ => panic!("Expected QubitOutOfRange error"),
        }
    }

    #[test]
    fn test_execute_circuit_empty() {
        let mut tab = StabilizerTableau::new(2);
        let circuit: Vec<QGate> = vec![];

        let result = tab.execute_circuit(&circuit);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_circuit_pauli_gates() {
        let mut tab = StabilizerTableau::new(3);
        let circuit = vec![QGate::X(0), QGate::Y(1), QGate::Z(2)];

        let result = tab.execute_circuit(&circuit);
        assert!(result.is_ok());
    }

    #[test]
    fn test_classify_clifford_angle() {
        // Test the angle classification helper
        assert_eq!(StabilizerTableau::classify_clifford_angle(0.0), Some(0));
        assert_eq!(
            StabilizerTableau::classify_clifford_angle(PI / 2.0),
            Some(1)
        );
        assert_eq!(StabilizerTableau::classify_clifford_angle(PI), Some(2));
        assert_eq!(
            StabilizerTableau::classify_clifford_angle(3.0 * PI / 2.0),
            Some(3)
        );
        assert_eq!(
            StabilizerTableau::classify_clifford_angle(2.0 * PI),
            Some(0)
        );

        // Non-Clifford angles
        assert_eq!(StabilizerTableau::classify_clifford_angle(PI / 4.0), None);
        assert_eq!(StabilizerTableau::classify_clifford_angle(PI / 3.0), None);
        assert_eq!(StabilizerTableau::classify_clifford_angle(0.1), None);
    }

    // =========================================================================
    // NOISY CIRCUIT EXECUTION TESTS (Phase 78.2)
    // =========================================================================

    #[test]
    fn test_ideal_model_no_errors_in_circuit() {
        // GateErrorModel::ideal() should produce zero errors
        let mut tab = StabilizerTableau::new(5);
        let mut rng = SimpleRng::new(42);
        let model = GateErrorModel::ideal();

        // Large circuit with many gates
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::CNot(0, 1),
            QGate::CNot(1, 2),
            QGate::CNot(2, 3),
            QGate::CZ(3, 4),
            QGate::X(0),
            QGate::Y(1),
            QGate::Z(2),
        ];

        let errors = tab
            .execute_circuit_noisy(&circuit, Some(&model), &mut rng)
            .expect("All Clifford gates");

        // No errors with ideal model
        assert!(
            errors.is_empty(),
            "Ideal model should produce no errors, got {} errors",
            errors.len()
        );
    }

    #[test]
    fn test_noisy_circuit_accumulates_errors() {
        // Circuit with N gates at high error rate should have some errors
        let mut tab = StabilizerTableau::new(4);
        let mut rng = SimpleRng::new(42);

        // Use high error rate for testing
        let model = GateErrorModel::custom(0.5, 0.5, 0.5, 0.5, 0.0);

        // Circuit with multiple gates
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::H(3),
            QGate::CNot(0, 1),
            QGate::CNot(2, 3),
        ];

        let errors = tab
            .execute_circuit_noisy(&circuit, Some(&model), &mut rng)
            .expect("All Clifford gates");

        // With 50% error rate on 6 gates, we expect ~3 errors (but could be 0-6)
        // Just verify we get some errors (statistically very likely with 50% rate)
        // Run multiple times to reduce flakiness
        let mut total_errors = errors.len();
        for seed in 100..110 {
            let mut tab2 = StabilizerTableau::new(4);
            let mut rng2 = SimpleRng::new(seed);
            let errors2 = tab2
                .execute_circuit_noisy(&circuit, Some(&model), &mut rng2)
                .expect("All Clifford gates");
            total_errors += errors2.len();
        }

        // Over 11 runs with 6 gates each at 50% rate, we expect ~33 errors
        // Probability of getting 0 errors in all runs is (0.5^6)^11 = essentially zero
        assert!(
            total_errors > 0,
            "Should have some errors with 50% error rate over multiple runs"
        );
    }

    #[test]
    fn test_noisy_single_gate_error_types() {
        // Verify that single-qubit gates produce single-qubit errors
        let model = GateErrorModel::custom(1.0, 0.0, 0.0, 0.0, 0.0); // 100% single-qubit error

        for _ in 0..10 {
            let mut tab = StabilizerTableau::new(2);
            let mut rng = SimpleRng::new(42);

            let error = tab
                .execute_gate_noisy(&QGate::H(0), Some(&model), &mut rng)
                .expect("H is Clifford")
                .expect("Should have error result");

            match error {
                GateError::Single { qubit, pauli } => {
                    assert_eq!(qubit, 0);
                    assert!(matches!(pauli, Pauli::X | Pauli::Y | Pauli::Z));
                }
                GateError::None => {} // Can happen due to RNG
                _ => panic!("Expected single-qubit error for H gate"),
            }
        }
    }

    #[test]
    fn test_noisy_two_qubit_gate_error_types() {
        // Verify that two-qubit gates produce two-qubit errors
        let model = GateErrorModel::custom(0.0, 1.0, 0.0, 0.0, 0.0); // 100% two-qubit error

        for seed in 0..10 {
            let mut tab = StabilizerTableau::new(3);
            let mut rng = SimpleRng::new(seed);

            let error = tab
                .execute_gate_noisy(&QGate::CNot(0, 1), Some(&model), &mut rng)
                .expect("CNOT is Clifford")
                .expect("Should have error result");

            match error {
                GateError::TwoQubit { q1, p1, q2, p2 } => {
                    assert_eq!(q1, 0);
                    assert_eq!(q2, 1);
                    // At least one Pauli should be non-identity (since we exclude II)
                    assert!(
                        !(p1 == Pauli::I && p2 == Pauli::I),
                        "Two-qubit error should not be II"
                    );
                }
                GateError::None => {} // Can happen due to RNG
                _ => panic!("Expected two-qubit error for CNOT gate"),
            }
        }
    }

    #[test]
    fn test_noisy_measurement() {
        // Test measurement error injection
        let model = GateErrorModel::custom(0.0, 0.0, 0.5, 0.0, 0.0); // 50% measurement error

        let mut flip_count = 0;
        let n = 100;

        for seed in 0..n {
            let mut tab = StabilizerTableau::new(1);
            let mut rng = SimpleRng::new(seed);

            // |0⟩ state - deterministic measurement should give 0
            let (result, _random, had_error) =
                tab.measure_z_noisy(0, false, Some(&model), &mut rng);

            if had_error {
                assert_eq!(result, 1, "Error should flip 0 to 1");
                flip_count += 1;
            } else {
                assert_eq!(result, 0, "No error should give 0");
            }
        }

        // With 50% error rate, should have roughly half flipped
        let flip_frac = flip_count as f64 / n as f64;
        assert!(
            (flip_frac - 0.5).abs() < 0.2,
            "Flip fraction {} should be near 0.5",
            flip_frac
        );
    }

    #[test]
    fn test_noisy_execution_no_model() {
        // Without error model, should execute without errors
        let mut tab = StabilizerTableau::new(3);
        let mut rng = SimpleRng::new(42);

        let result = tab.execute_gate_noisy(&QGate::H(0), None, &mut rng);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // No error info when no model
    }

    #[test]
    fn test_apply_pauli_helper() {
        // Test the internal apply_pauli method indirectly through noisy execution
        let mut tab = StabilizerTableau::new(2);

        // Start with |00⟩, apply X manually
        tab.apply_pauli(0, Pauli::X);

        // Now qubit 0 should be |1⟩
        let (result, _) = tab.measure_z(0, false);
        assert_eq!(result, 1, "X on |0⟩ should give |1⟩");

        // Test identity does nothing
        let mut tab2 = StabilizerTableau::new(1);
        tab2.apply_pauli(0, Pauli::I);
        let (result2, _) = tab2.measure_z(0, false);
        assert_eq!(result2, 0, "Identity should not change state");
    }
}
