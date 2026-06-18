//! Quantum Error Correction Codes
//!
//! Implementations of common QEC codes using the stabilizer formalism.

use super::stabilizer::{PauliOperator, StabilizerTableau};

/// Bit-flip repetition code [[n, 1, n]]
///
/// Encodes 1 logical qubit into n physical qubits.
/// Protects against up to (n-1)/2 bit-flip (X) errors.
///
/// Logical states:
/// - |0_L⟩ = |00...0⟩
/// - |1_L⟩ = |11...1⟩
///
/// Stabilizers: Z_i Z_{i+1} for i = 0..n-2
#[derive(Clone)]
pub struct RepetitionCode {
    /// Number of physical qubits
    pub n: usize,
    /// Stabilizer tableau for current state
    pub tableau: StabilizerTableau,
}

impl RepetitionCode {
    /// Create repetition code with n physical qubits in |0_L⟩ state
    pub fn new(n: usize) -> Self {
        assert!(n >= 3, "Need at least 3 qubits for error correction");

        let code = Self {
            n,
            tableau: StabilizerTableau::new(n),
        };

        // The |0...0⟩ state is already in the code space
        // Stabilizers are ZZ pairs: Z0Z1, Z1Z2, ..., Z_{n-2}Z_{n-1}
        // Plus one logical Z: Z0Z1...Z_{n-1}

        code
    }

    /// Get the i-th stabilizer generator: Z_i Z_{i+1}
    pub fn stabilizer_generator(&self, i: usize) -> PauliOperator {
        assert!(i < self.n - 1);
        let mut op = PauliOperator::identity(self.n);
        op.set_z(i, true);
        op.set_z(i + 1, true);
        op
    }

    /// Get logical X operator: X on all qubits
    pub fn logical_x(&self) -> PauliOperator {
        let mut op = PauliOperator::identity(self.n);
        for i in 0..self.n {
            op.set_x(i, true);
        }
        op
    }

    /// Get logical Z operator: Z on any single qubit (all equivalent)
    pub fn logical_z(&self) -> PauliOperator {
        PauliOperator::z(self.n, 0)
    }

    /// Apply X error to qubit
    pub fn apply_x_error(&mut self, qubit: usize) {
        self.tableau.pauli_x(qubit);
    }

    /// Measure all stabilizers, return syndrome
    ///
    /// Syndrome is a bit vector where bit i = 1 indicates
    /// stabilizer i has eigenvalue -1 (error detected)
    ///
    /// For the repetition code starting from |0...0⟩:
    /// - Each qubit has stabilizer Zᵢ with phase 0 (+1) or 2 (-1)
    /// - An X error on qubit j flips Zⱼ's phase from 0 to 2
    /// - Syndrome bit i = 1 when ZᵢZᵢ₊₁ has eigenvalue -1
    ///   (i.e., when exactly one of Zᵢ, Zᵢ₊₁ has flipped phase)
    pub fn measure_syndrome(&self) -> Vec<u8> {
        let mut syndrome = Vec::with_capacity(self.n - 1);

        for i in 0..(self.n - 1) {
            // Get phases of Z_i and Z_{i+1} stabilizers
            // Phase 0 = +1 eigenvalue, Phase 2 = -1 eigenvalue
            let phase_i = self.tableau.stabilizer(i).phase;
            let phase_i1 = self.tableau.stabilizer(i + 1).phase;

            // Check if phases indicate negative eigenvalues
            let neg_i = phase_i == 2;
            let neg_i1 = phase_i1 == 2;

            // XOR: syndrome=1 when there's an error boundary
            // (exactly one of the two qubits has an X error)
            if neg_i != neg_i1 {
                syndrome.push(1);
            } else {
                syndrome.push(0);
            }
        }

        syndrome
    }

    /// Decode syndrome to identify all error locations
    ///
    /// For repetition code syndrome decoding:
    /// - Syndrome bit i = 1 means ZᵢZᵢ₊₁ has eigenvalue -1
    /// - A single X error at qubit j flips syndrome bits j-1 and j
    /// - We find "runs" of consecutive 1s; each run indicates one error
    ///
    /// Returns Vec of qubit indices where X errors likely occurred
    pub fn decode_syndrome(&self, syndrome: &[u8]) -> Vec<usize> {
        if syndrome.iter().all(|&s| s == 0) {
            return Vec::new(); // No errors
        }

        let mut errors = Vec::new();
        let n_syndrome = syndrome.len(); // = n_qubits - 1

        // Find runs of consecutive 1s
        // Each run [start, end] indicates an error
        let mut i = 0;
        while i < n_syndrome {
            if syndrome[i] == 1 {
                // Start of a run
                let run_start = i;
                while i < n_syndrome && syndrome[i] == 1 {
                    i += 1;
                }
                let run_end = i - 1; // Last 1 in the run

                // Decode this run to find the error qubit
                //
                // For a single X error at qubit q:
                // - q=0: only syndrome[0]=1 (run is just [0])
                // - q=n-1: only syndrome[n-2]=1 (run is just [n-2])
                // - 0<q<n-1: syndrome[q-1]=1 and syndrome[q]=1 (run is [q-1, q])
                //
                // So for a run [start, end]:
                // - If start == end == 0: error at qubit 0
                // - If start == end == n-2: error at qubit n-1
                // - If start == end (middle): ambiguous, guess start+1
                // - If end == start + 1: error at qubit end (= start + 1)
                // - Longer runs indicate multiple adjacent errors

                if run_start == run_end {
                    // Single syndrome bit flipped
                    if run_start == 0 {
                        errors.push(0); // Edge: error at qubit 0
                    } else if run_start == n_syndrome - 1 {
                        errors.push(self.n - 1); // Edge: error at last qubit
                    } else {
                        // Middle single flip is unusual for single errors
                        // Could be qubit start or start+1; guess start+1
                        errors.push(run_start + 1);
                    }
                } else if run_end == run_start + 1 {
                    // Two consecutive 1s: classic single-error pattern
                    // Error at qubit run_end (= run_start + 1)
                    errors.push(run_end);
                } else {
                    // Longer run: multiple adjacent errors or odd error pattern
                    // Decode as error at the center of the run
                    let center = (run_start + run_end).div_ceil(2);
                    errors.push(center);
                }
            } else {
                i += 1;
            }
        }

        errors
    }

    /// Decode syndrome (legacy single-error interface)
    pub fn decode_syndrome_single(&self, syndrome: &[u8]) -> Option<usize> {
        let errors = self.decode_syndrome(syndrome);
        errors.first().copied()
    }

    /// Apply correction based on decoded error locations
    pub fn correct(&mut self, error_qubits: &[usize]) {
        for &q in error_qubits {
            self.tableau.pauli_x(q);
        }
    }

    /// Apply correction for single error (legacy interface)
    pub fn correct_single(&mut self, error_qubit: Option<usize>) {
        if let Some(q) = error_qubit {
            self.tableau.pauli_x(q);
        }
    }
}

/// Steane [[7, 1, 3]] code
///
/// Encodes 1 logical qubit into 7 physical qubits.
/// Can correct any single-qubit error (X, Y, or Z).
///
/// Based on classical [7,4,3] Hamming code.
#[derive(Clone)]
pub struct SteaneCode {
    pub tableau: StabilizerTableau,
    /// Track accumulated X errors (for Z syndrome)
    x_errors: [bool; 7],
    /// Track accumulated Z errors (for X syndrome)
    z_errors: [bool; 7],
}

impl SteaneCode {
    /// Create Steane code in |0_L⟩ state
    pub fn new() -> Self {
        Self {
            tableau: StabilizerTableau::new(7),
            x_errors: [false; 7],
            z_errors: [false; 7],
        }
    }

    /// X-type stabilizers (from Hamming code parity checks)
    pub fn x_stabilizers() -> [PauliOperator; 3] {
        let mut s1 = PauliOperator::identity(7);
        let mut s2 = PauliOperator::identity(7);
        let mut s3 = PauliOperator::identity(7);

        // Steane code X stabilizers based on Hamming [7,4,3]
        // S1: X on qubits 0,2,4,6
        for i in [0, 2, 4, 6] {
            s1.set_x(i, true);
        }
        // S2: X on qubits 1,2,5,6
        for i in [1, 2, 5, 6] {
            s2.set_x(i, true);
        }
        // S3: X on qubits 3,4,5,6
        for i in [3, 4, 5, 6] {
            s3.set_x(i, true);
        }

        [s1, s2, s3]
    }

    /// Z-type stabilizers
    pub fn z_stabilizers() -> [PauliOperator; 3] {
        let mut s1 = PauliOperator::identity(7);
        let mut s2 = PauliOperator::identity(7);
        let mut s3 = PauliOperator::identity(7);

        // Same pattern as X but with Z
        for i in [0, 2, 4, 6] {
            s1.set_z(i, true);
        }
        for i in [1, 2, 5, 6] {
            s2.set_z(i, true);
        }
        for i in [3, 4, 5, 6] {
            s3.set_z(i, true);
        }

        [s1, s2, s3]
    }

    /// Logical X: X on all qubits
    pub fn logical_x() -> PauliOperator {
        let mut op = PauliOperator::identity(7);
        for i in 0..7 {
            op.set_x(i, true);
        }
        op
    }

    /// Logical Z: Z on all qubits
    pub fn logical_z() -> PauliOperator {
        let mut op = PauliOperator::identity(7);
        for i in 0..7 {
            op.set_z(i, true);
        }
        op
    }

    /// Apply single-qubit error
    pub fn apply_error(&mut self, qubit: usize, error_type: char) {
        match error_type {
            'X' => {
                self.tableau.pauli_x(qubit);
                self.x_errors[qubit] ^= true;
            }
            'Y' => {
                self.tableau.pauli_y(qubit);
                self.x_errors[qubit] ^= true;
                self.z_errors[qubit] ^= true;
            }
            'Z' => {
                self.tableau.pauli_z(qubit);
                self.z_errors[qubit] ^= true;
            }
            _ => {}
        }
    }

    /// Measure X-type syndrome (detects Z errors)
    ///
    /// Returns 3-bit syndrome where each bit corresponds to one X stabilizer.
    /// The syndrome value directly encodes the error location (Hamming code property).
    ///
    /// X stabilizers anticommute with Z errors:
    /// - S1 = X₀X₂X₄X₆ anticommutes with Z on qubits {0,2,4,6}
    /// - S2 = X₁X₂X₅X₆ anticommutes with Z on qubits {1,2,5,6}
    /// - S3 = X₃X₄X₅X₆ anticommutes with Z on qubits {3,4,5,6}
    pub fn measure_x_syndrome(&self) -> [u8; 3] {
        let mut syndrome = [0u8; 3];

        // Stabilizer patterns (which qubits each X stabilizer covers)
        let patterns: [[usize; 4]; 3] = [[0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6]];

        for (i, pattern) in patterns.iter().enumerate() {
            let mut parity = 0u8;
            for &qubit in pattern {
                // Z error on qubit j anticommutes with X on qubit j
                // Syndrome bit = XOR of Z errors in the stabilizer's support
                if self.z_errors[qubit] {
                    parity ^= 1;
                }
            }
            syndrome[i] = parity;
        }

        syndrome
    }

    /// Measure Z-type syndrome (detects X errors)
    ///
    /// Returns 3-bit syndrome where each bit corresponds to one Z stabilizer.
    ///
    /// Z stabilizers anticommute with X errors:
    /// - S1 = Z₀Z₂Z₄Z₆ anticommutes with X on qubits {0,2,4,6}
    /// - S2 = Z₁Z₂Z₅Z₆ anticommutes with X on qubits {1,2,5,6}
    /// - S3 = Z₃Z₄Z₅Z₆ anticommutes with X on qubits {3,4,5,6}
    pub fn measure_z_syndrome(&self) -> [u8; 3] {
        let mut syndrome = [0u8; 3];

        // Same patterns as X stabilizers (Steane code is CSS)
        let patterns: [[usize; 4]; 3] = [[0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6]];

        for (i, pattern) in patterns.iter().enumerate() {
            let mut parity = 0u8;
            for &qubit in pattern {
                // X error on qubit j anticommutes with Z on qubit j
                // Syndrome bit = XOR of X errors in the stabilizer's support
                if self.x_errors[qubit] {
                    parity ^= 1;
                }
            }
            syndrome[i] = parity;
        }

        syndrome
    }

    /// Measure full syndrome (both X and Z)
    ///
    /// Returns (x_syndrome, z_syndrome) where each is a 3-bit array.
    /// - x_syndrome detects Z errors (phase flips)
    /// - z_syndrome detects X errors (bit flips)
    pub fn measure_syndrome(&self) -> ([u8; 3], [u8; 3]) {
        (self.measure_x_syndrome(), self.measure_z_syndrome())
    }

    /// Decode syndrome to identify error
    ///
    /// Returns (error_qubit, error_type) where error_type is 'X', 'Z', 'Y', or 'I'.
    /// Uses Hamming code property: syndrome bits form binary error location.
    pub fn decode_syndrome(&self, x_syn: &[u8; 3], z_syn: &[u8; 3]) -> (Option<usize>, char) {
        // X syndrome (from X stabilizers) detects Z errors
        // Z syndrome (from Z stabilizers) detects X errors

        // Convert syndrome bits to error location (1-indexed in Hamming, 0-indexed here)
        let z_error_loc =
            (x_syn[0] as usize) | ((x_syn[1] as usize) << 1) | ((x_syn[2] as usize) << 2);
        let x_error_loc =
            (z_syn[0] as usize) | ((z_syn[1] as usize) << 1) | ((z_syn[2] as usize) << 2);

        // Determine error type
        match (x_error_loc, z_error_loc) {
            (0, 0) => (None, 'I'),                  // No error
            (0, z) => (Some(z - 1), 'Z'),           // Z error only (Hamming is 1-indexed)
            (x, 0) => (Some(x - 1), 'X'),           // X error only
            (x, z) if x == z => (Some(x - 1), 'Y'), // Both X and Z = Y error
            (x, _z) => {
                // Different locations - likely two errors (beyond correction capability)
                // Return the X error location as primary
                (Some(x - 1), 'X')
            }
        }
    }

    /// Apply correction based on decoded error
    ///
    /// Correction is the same as the error (Pauli operators are self-inverse)
    pub fn correct(&mut self, qubit: Option<usize>, error_type: char) {
        if let Some(q) = qubit {
            if q < 7 {
                match error_type {
                    'X' => {
                        self.tableau.pauli_x(q);
                        self.x_errors[q] ^= true;
                    }
                    'Y' => {
                        self.tableau.pauli_y(q);
                        self.x_errors[q] ^= true;
                        self.z_errors[q] ^= true;
                    }
                    'Z' => {
                        self.tableau.pauli_z(q);
                        self.z_errors[q] ^= true;
                    }
                    _ => {}
                }
            }
        }
    }

    /// Full error correction cycle: measure, decode, correct
    ///
    /// Returns (detected_qubit, detected_error_type, success)
    pub fn error_correction_cycle(&mut self) -> (Option<usize>, char, bool) {
        let (x_syn, z_syn) = self.measure_syndrome();
        let (qubit, error_type) = self.decode_syndrome(&x_syn, &z_syn);

        // Apply correction
        self.correct(qubit, error_type);

        // Verify by re-measuring syndrome
        let (x_syn_after, z_syn_after) = self.measure_syndrome();
        let success = x_syn_after.iter().all(|&s| s == 0) && z_syn_after.iter().all(|&s| s == 0);

        (qubit, error_type, success)
    }
}

/// Surface Code lattice
///
/// The surface code is the leading candidate for fault-tolerant QC.
/// Uses a 2D grid of data qubits with X and Z stabilizers.
///
/// Layout (rotated surface code for distance d):
/// ```text
///   For d=3:
///     0 - 1 - 2      Data qubits at grid positions
///     |   |   |
///     3 - 4 - 5      X stabilizers (plaquettes) on faces
///     |   |   |      Z stabilizers (vertices) between faces
///     6 - 7 - 8
/// ```
///
/// For a distance-d code:
/// - d × d data qubits
/// - (d-1)² / 2 X stabilizers (approximately)
/// - (d-1)² / 2 Z stabilizers (approximately)
#[derive(Clone)]
pub struct SurfaceCode {
    /// Code distance
    pub distance: usize,
    /// Grid width (number of data qubits per row)
    pub width: usize,
    /// Grid height
    pub height: usize,
    /// Total number of data qubits
    pub n_data: usize,
    /// Stabilizer tableau
    pub tableau: StabilizerTableau,
    /// Track X errors on each qubit (for Z syndrome)
    x_errors: Vec<bool>,
    /// Track Z errors on each qubit (for X syndrome)
    z_errors: Vec<bool>,
    /// Cached X stabilizer definitions: (stabilizer_id, qubits)
    x_stabilizers: Vec<Vec<usize>>,
    /// Cached Z stabilizer definitions: (stabilizer_id, qubits)
    z_stabilizers: Vec<Vec<usize>>,
}

impl SurfaceCode {
    /// Create surface code with given distance
    pub fn new(distance: usize) -> Self {
        assert!(distance >= 2, "Distance must be at least 2");

        // Rotated surface code layout
        let width = distance;
        let height = distance;
        let n_data = width * height;

        // Build stabilizer lists using standard rotated surface code layout
        // X stabilizers (plaquettes) and Z stabilizers (stars) alternate in checkerboard
        let mut x_stabilizers = Vec::new();
        let mut z_stabilizers = Vec::new();

        // Interior 4-qubit stabilizers
        // For each 2x2 block at (r, c), we have a stabilizer on qubits:
        // (r,c), (r,c+1), (r+1,c), (r+1,c+1)
        for r in 0..(height - 1) {
            for c in 0..(width - 1) {
                let qubits = vec![
                    r * width + c,           // top-left
                    r * width + c + 1,       // top-right
                    (r + 1) * width + c,     // bottom-left
                    (r + 1) * width + c + 1, // bottom-right
                ];

                // Checkerboard pattern: X on even parity, Z on odd parity
                if (r + c) % 2 == 0 {
                    x_stabilizers.push(qubits);
                } else {
                    z_stabilizers.push(qubits);
                }
            }
        }

        // Boundary stabilizers (2-qubit) for rotated surface code
        // These complete the stabilizer set at the edges

        // Top edge: horizontal 2-qubit stabilizers at certain positions
        for c in 0..(width - 1) {
            if c % 2 == 1 {
                // X stabilizer on top boundary at odd columns
                x_stabilizers.push(vec![c, c + 1]);
            }
        }

        // Bottom edge: horizontal 2-qubit stabilizers
        for c in 0..(width - 1) {
            if c % 2 == 0 {
                // X stabilizer on bottom boundary at even columns
                let base = (height - 1) * width;
                x_stabilizers.push(vec![base + c, base + c + 1]);
            }
        }

        // Left edge: vertical 2-qubit stabilizers
        for r in 0..(height - 1) {
            if r % 2 == 1 {
                // Z stabilizer on left boundary at odd rows
                z_stabilizers.push(vec![r * width, (r + 1) * width]);
            }
        }

        // Right edge: vertical 2-qubit stabilizers
        for r in 0..(height - 1) {
            if r % 2 == 0 {
                // Z stabilizer on right boundary at even rows
                z_stabilizers.push(vec![r * width + (width - 1), (r + 1) * width + (width - 1)]);
            }
        }

        Self {
            distance,
            width,
            height,
            n_data,
            tableau: StabilizerTableau::new(n_data),
            x_errors: vec![false; n_data],
            z_errors: vec![false; n_data],
            x_stabilizers,
            z_stabilizers,
        }
    }

    /// Convert (row, col) to qubit index
    #[inline]
    pub fn qubit_index(&self, row: usize, col: usize) -> usize {
        row * self.width + col
    }

    /// Get X stabilizer (plaquette) at position
    /// Returns None if position is outside the lattice
    pub fn x_stabilizer(&self, row: usize, col: usize) -> Option<PauliOperator> {
        if row >= self.height - 1 || col >= self.width {
            return None;
        }

        let mut op = PauliOperator::identity(self.n_data);

        // X stabilizer acts on 4 qubits (or 2 at boundaries)
        // Standard plaquette: (row,col), (row,col+1), (row+1,col), (row+1,col+1)
        // Adjust for boundaries

        if col > 0 {
            op.set_x(self.qubit_index(row, col - 1), true);
        }
        if col < self.width {
            op.set_x(self.qubit_index(row, col), true);
        }
        if row + 1 < self.height && col > 0 {
            op.set_x(self.qubit_index(row + 1, col - 1), true);
        }
        if row + 1 < self.height && col < self.width {
            op.set_x(self.qubit_index(row + 1, col), true);
        }

        Some(op)
    }

    /// Get Z stabilizer (vertex) at position
    pub fn z_stabilizer(&self, row: usize, col: usize) -> Option<PauliOperator> {
        if row >= self.height || col >= self.width - 1 {
            return None;
        }

        let mut op = PauliOperator::identity(self.n_data);

        // Z stabilizer on 4 qubits
        if row > 0 {
            op.set_z(self.qubit_index(row - 1, col), true);
            op.set_z(self.qubit_index(row - 1, col + 1), true);
        }
        op.set_z(self.qubit_index(row, col), true);
        op.set_z(self.qubit_index(row, col + 1), true);

        Some(op)
    }

    /// Apply X error to qubit by (row, col)
    pub fn apply_x_error(&mut self, row: usize, col: usize) {
        let q = self.qubit_index(row, col);
        self.apply_x_error_at(q);
    }

    /// Apply Z error to qubit by (row, col)
    pub fn apply_z_error(&mut self, row: usize, col: usize) {
        let q = self.qubit_index(row, col);
        self.apply_z_error_at(q);
    }

    /// Apply X error to qubit by index
    pub fn apply_x_error_at(&mut self, qubit: usize) {
        self.tableau.pauli_x(qubit);
        self.x_errors[qubit] ^= true;
    }

    /// Apply Z error to qubit by index
    pub fn apply_z_error_at(&mut self, qubit: usize) {
        self.tableau.pauli_z(qubit);
        self.z_errors[qubit] ^= true;
    }

    /// Apply Y error to qubit by index
    pub fn apply_y_error_at(&mut self, qubit: usize) {
        self.tableau.pauli_y(qubit);
        self.x_errors[qubit] ^= true;
        self.z_errors[qubit] ^= true;
    }

    /// Apply general error to qubit
    pub fn apply_error(&mut self, qubit: usize, error_type: char) {
        match error_type {
            'X' => self.apply_x_error_at(qubit),
            'Y' => self.apply_y_error_at(qubit),
            'Z' => self.apply_z_error_at(qubit),
            _ => {}
        }
    }

    /// Get logical X operator (horizontal chain)
    pub fn logical_x(&self) -> PauliOperator {
        let mut op = PauliOperator::identity(self.n_data);
        // X chain along top row
        for col in 0..self.width {
            op.set_x(self.qubit_index(0, col), true);
        }
        op
    }

    /// Get logical Z operator (vertical chain)
    pub fn logical_z(&self) -> PauliOperator {
        let mut op = PauliOperator::identity(self.n_data);
        // Z chain along left column
        for row in 0..self.height {
            op.set_z(self.qubit_index(row, 0), true);
        }
        op
    }

    /// Get number of X stabilizers
    pub fn num_x_stabilizers(&self) -> usize {
        self.x_stabilizers.len()
    }

    /// Get number of Z stabilizers
    pub fn num_z_stabilizers(&self) -> usize {
        self.z_stabilizers.len()
    }

    /// Data-qubit supports of each X stabilizer, in X-syndrome order.
    /// Stabilizer `i` corresponds to `measure_x_syndrome()[i]`; its lattice
    /// position is the centroid of these qubits' `(row, col)` coordinates.
    pub fn x_stabilizer_supports(&self) -> &[Vec<usize>] {
        &self.x_stabilizers
    }

    /// Data-qubit supports of each Z stabilizer, in Z-syndrome order.
    pub fn z_stabilizer_supports(&self) -> &[Vec<usize>] {
        &self.z_stabilizers
    }

    /// Measure X-type syndrome (detects Z errors)
    ///
    /// Returns a vector of syndrome bits, one per X stabilizer.
    /// Syndrome bit is 1 if the stabilizer has eigenvalue -1.
    ///
    /// X stabilizers anticommute with Z errors on their support qubits.
    pub fn measure_x_syndrome(&self) -> Vec<u8> {
        let mut syndrome = Vec::with_capacity(self.x_stabilizers.len());

        for stab_qubits in &self.x_stabilizers {
            let mut parity = 0u8;
            for &q in stab_qubits {
                // Z error on qubit q anticommutes with X on qubit q
                if self.z_errors[q] {
                    parity ^= 1;
                }
            }
            syndrome.push(parity);
        }

        syndrome
    }

    /// Measure Z-type syndrome (detects X errors)
    ///
    /// Returns a vector of syndrome bits, one per Z stabilizer.
    /// Z stabilizers anticommute with X errors on their support qubits.
    pub fn measure_z_syndrome(&self) -> Vec<u8> {
        let mut syndrome = Vec::with_capacity(self.z_stabilizers.len());

        for stab_qubits in &self.z_stabilizers {
            let mut parity = 0u8;
            for &q in stab_qubits {
                // X error on qubit q anticommutes with Z on qubit q
                if self.x_errors[q] {
                    parity ^= 1;
                }
            }
            syndrome.push(parity);
        }

        syndrome
    }

    /// Measure full syndrome (both X and Z)
    ///
    /// Returns (x_syndrome, z_syndrome) where:
    /// - x_syndrome detects Z errors
    /// - z_syndrome detects X errors
    pub fn measure_syndrome(&self) -> (Vec<u8>, Vec<u8>) {
        (self.measure_x_syndrome(), self.measure_z_syndrome())
    }

    /// Simple greedy decoder for surface code
    ///
    /// This is a basic decoder that finds the minimum-weight correction
    /// for each triggered stabilizer independently. For production use,
    /// MWPM (Minimum Weight Perfect Matching) would be preferred.
    ///
    /// Returns (x_corrections, z_corrections) as lists of qubit indices.
    pub fn decode_syndrome(&self, x_syn: &[u8], z_syn: &[u8]) -> (Vec<usize>, Vec<usize>) {
        let mut x_corrections = Vec::new(); // X errors to apply (correct Z syndrome)
        let mut z_corrections = Vec::new(); // Z errors to apply (correct X syndrome)

        // Decode Z syndrome → X errors
        // For each triggered Z stabilizer, apply X correction to one qubit
        for (stab_idx, &syn_bit) in z_syn.iter().enumerate() {
            if syn_bit == 1 {
                // Find a qubit in this stabilizer that hasn't been corrected yet
                if let Some(&q) = self.z_stabilizers.get(stab_idx).and_then(|qs| qs.first()) {
                    // Simple strategy: correct the first qubit of this stabilizer
                    // Check if already in corrections to avoid double-correction
                    if !x_corrections.contains(&q) {
                        x_corrections.push(q);
                    }
                }
            }
        }

        // Decode X syndrome → Z errors
        for (stab_idx, &syn_bit) in x_syn.iter().enumerate() {
            if syn_bit == 1 {
                if let Some(&q) = self.x_stabilizers.get(stab_idx).and_then(|qs| qs.first()) {
                    if !z_corrections.contains(&q) {
                        z_corrections.push(q);
                    }
                }
            }
        }

        (x_corrections, z_corrections)
    }

    /// Apply corrections to the code
    ///
    /// x_corrections: qubits to apply X (corrects Z errors)
    /// z_corrections: qubits to apply Z (corrects X errors)
    pub fn correct(&mut self, x_corrections: &[usize], z_corrections: &[usize]) {
        for &q in x_corrections {
            self.apply_x_error_at(q);
        }
        for &q in z_corrections {
            self.apply_z_error_at(q);
        }
    }

    /// Full error correction cycle: measure, decode, correct
    ///
    /// Returns (success, x_corrections, z_corrections)
    pub fn error_correction_cycle(&mut self) -> (bool, Vec<usize>, Vec<usize>) {
        let (x_syn, z_syn) = self.measure_syndrome();
        let (x_corrections, z_corrections) = self.decode_syndrome(&x_syn, &z_syn);

        // Apply corrections
        self.correct(&x_corrections, &z_corrections);

        // Verify by re-measuring syndrome
        let (x_syn_after, z_syn_after) = self.measure_syndrome();
        let success = x_syn_after.iter().all(|&s| s == 0) && z_syn_after.iter().all(|&s| s == 0);

        (success, x_corrections, z_corrections)
    }

    /// Reset the code to initial |0_L⟩ state
    pub fn reset(&mut self) {
        self.tableau = StabilizerTableau::new(self.n_data);
        self.x_errors = vec![false; self.n_data];
        self.z_errors = vec![false; self.n_data];
    }

    /// Check if a logical X error has occurred
    ///
    /// A logical X error flips the logical qubit (|0_L⟩ ↔ |1_L⟩).
    /// For the surface code, this happens when an X error chain
    /// connects the top and bottom rough boundaries.
    pub fn has_logical_x_error(&self) -> bool {
        // Check if there's an odd number of X errors along the left column
        // (which is the support of logical Z)
        let mut parity = false;
        for row in 0..self.height {
            let q = self.qubit_index(row, 0);
            if self.x_errors[q] {
                parity = !parity;
            }
        }
        parity
    }

    /// Check if a logical Z error has occurred
    ///
    /// A logical Z error adds a phase to the logical qubit.
    /// For the surface code, this happens when a Z error chain
    /// connects the left and right smooth boundaries.
    pub fn has_logical_z_error(&self) -> bool {
        // Check if there's an odd number of Z errors along the top row
        // (which is the support of logical X)
        let mut parity = false;
        for col in 0..self.width {
            let q = self.qubit_index(0, col);
            if self.z_errors[q] {
                parity = !parity;
            }
        }
        parity
    }

    /// Error correction cycle using Union-Find decoder
    ///
    /// This uses the more sophisticated Union-Find decoder instead of
    /// the simple greedy decoder. Achieves better performance, especially
    /// for multiple errors.
    ///
    /// Returns (success, x_corrections, z_corrections)
    pub fn error_correction_cycle_union_find(
        &mut self,
        decoder: &super::decoder::UnionFindDecoder,
    ) -> (bool, Vec<usize>, Vec<usize>) {
        let (x_syn, z_syn) = self.measure_syndrome();
        let (x_corrections, z_corrections) = decoder.decode(&x_syn, &z_syn);

        // Apply corrections
        self.correct(&x_corrections, &z_corrections);

        // Verify by re-measuring syndrome
        let (x_syn_after, z_syn_after) = self.measure_syndrome();
        let success = x_syn_after.iter().all(|&s| s == 0) && z_syn_after.iter().all(|&s| s == 0);

        (success, x_corrections, z_corrections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_code() {
        let code = RepetitionCode::new(5);
        assert_eq!(code.n, 5);
    }

    #[test]
    fn test_repetition_error_correction() {
        let mut code = RepetitionCode::new(5);

        // Apply X error on qubit 2
        code.apply_x_error(2);

        // Syndrome should detect error
        let syndrome = code.measure_syndrome();

        // Decode
        let errors = code.decode_syndrome(&syndrome);
        assert!(!errors.is_empty());
        assert_eq!(errors[0], 2); // Should identify qubit 2

        println!("Syndrome: {:?}, Errors: {:?}", syndrome, errors);

        // Apply correction and verify syndrome clears
        code.correct(&errors);
        let syndrome_after = code.measure_syndrome();
        assert!(
            syndrome_after.iter().all(|&s| s == 0),
            "Syndrome should clear after correction"
        );
        println!("After correction: {:?}", syndrome_after);
    }

    #[test]
    fn test_repetition_multi_error() {
        let mut code = RepetitionCode::new(11);

        // Apply X errors on qubits 3 and 7
        code.apply_x_error(3);
        code.apply_x_error(7);

        let syndrome = code.measure_syndrome();
        println!("Multi-error syndrome: {:?}", syndrome);

        // Should be [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        assert_eq!(syndrome, vec![0, 0, 1, 1, 0, 0, 1, 1, 0, 0]);

        // Decode should find both errors
        let errors = code.decode_syndrome(&syndrome);
        println!("Decoded errors: {:?}", errors);
        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0], 3);
        assert_eq!(errors[1], 7);

        // Correct and verify
        code.correct(&errors);
        let syndrome_after = code.measure_syndrome();
        assert!(syndrome_after.iter().all(|&s| s == 0));
        println!("After correction: {:?}", syndrome_after);
    }

    #[test]
    fn test_repetition_edge_errors() {
        // Test error at qubit 0 (edge case)
        let mut code = RepetitionCode::new(5);
        code.apply_x_error(0);
        let syndrome = code.measure_syndrome();
        println!("Edge error at 0: syndrome={:?}", syndrome);
        assert_eq!(syndrome[0], 1);
        assert!(syndrome[1..].iter().all(|&s| s == 0));

        let errors = code.decode_syndrome(&syndrome);
        assert_eq!(errors, vec![0]);

        // Test error at last qubit (edge case)
        let mut code2 = RepetitionCode::new(5);
        code2.apply_x_error(4);
        let syndrome2 = code2.measure_syndrome();
        println!("Edge error at 4: syndrome={:?}", syndrome2);
        assert!(syndrome2[..3].iter().all(|&s| s == 0));
        assert_eq!(syndrome2[3], 1);

        let errors2 = code2.decode_syndrome(&syndrome2);
        assert_eq!(errors2, vec![4]);
    }

    #[test]
    fn test_surface_code_creation() {
        let code = SurfaceCode::new(3);
        assert_eq!(code.distance, 3);
        assert_eq!(code.n_data, 9);
    }

    #[test]
    fn test_surface_code_logical_operators() {
        let code = SurfaceCode::new(3);

        let lx = code.logical_x();
        let lz = code.logical_z();

        // Logical X and Z should anti-commute
        assert!(!lx.commutes_with(&lz));
    }

    #[test]
    fn test_steane_code() {
        let _code = SteaneCode::new();

        // X and Z stabilizers should commute
        let x_stabs = SteaneCode::x_stabilizers();
        let z_stabs = SteaneCode::z_stabilizers();

        for xs in &x_stabs {
            for zs in &z_stabs {
                assert!(xs.commutes_with(zs));
            }
        }
    }

    #[test]
    fn test_steane_x_error_correction() {
        println!("\n=== Steane Code X Error Correction ===");

        // Test X error on each qubit
        for error_qubit in 0..7 {
            let mut code = SteaneCode::new();

            // Apply X error
            code.apply_error(error_qubit, 'X');

            // Measure and decode
            let (x_syn, z_syn) = code.measure_syndrome();
            let (detected_qubit, error_type) = code.decode_syndrome(&x_syn, &z_syn);

            println!(
                "X error on qubit {}: x_syn={:?}, z_syn={:?}, detected=({:?}, {})",
                error_qubit, x_syn, z_syn, detected_qubit, error_type
            );

            // Apply correction
            code.correct(detected_qubit, error_type);

            // Verify syndrome clears
            let (x_after, z_after) = code.measure_syndrome();
            let success = x_after.iter().all(|&s| s == 0) && z_after.iter().all(|&s| s == 0);

            println!(
                "  After correction: x={:?}, z={:?}, success={}",
                x_after, z_after, success
            );
        }
    }

    #[test]
    fn test_steane_z_error_correction() {
        println!("\n=== Steane Code Z Error Correction ===");

        // Test Z error on each qubit
        for error_qubit in 0..7 {
            let mut code = SteaneCode::new();

            // Apply Z error
            code.apply_error(error_qubit, 'Z');

            // Measure and decode
            let (x_syn, z_syn) = code.measure_syndrome();
            let (detected_qubit, error_type) = code.decode_syndrome(&x_syn, &z_syn);

            println!(
                "Z error on qubit {}: x_syn={:?}, z_syn={:?}, detected=({:?}, {})",
                error_qubit, x_syn, z_syn, detected_qubit, error_type
            );

            // Apply correction
            code.correct(detected_qubit, error_type);

            // Verify syndrome clears
            let (x_after, z_after) = code.measure_syndrome();
            println!("  After correction: x={:?}, z={:?}", x_after, z_after);
        }
    }

    #[test]
    fn test_steane_y_error_correction() {
        println!("\n=== Steane Code Y Error Correction ===");

        // Test Y error on each qubit (Y = iXZ, so both syndromes should trigger)
        for error_qubit in 0..7 {
            let mut code = SteaneCode::new();

            // Apply Y error
            code.apply_error(error_qubit, 'Y');

            // Measure and decode
            let (x_syn, z_syn) = code.measure_syndrome();
            let (detected_qubit, error_type) = code.decode_syndrome(&x_syn, &z_syn);

            println!(
                "Y error on qubit {}: x_syn={:?}, z_syn={:?}, detected=({:?}, {})",
                error_qubit, x_syn, z_syn, detected_qubit, error_type
            );
        }
    }

    #[test]
    fn test_steane_full_correction_cycle() {
        use crate::qec::noise::SimpleRng;

        println!("\n=== Steane Code Full Correction Cycle ===");

        let mut rng = SimpleRng::new(42);
        let n_trials = 100;
        let error_types = ['X', 'Y', 'Z'];

        let mut success_count = 0;

        for _ in 0..n_trials {
            let mut code = SteaneCode::new();

            // Random single error
            let error_qubit = rng.next_usize(7);
            let error_type = error_types[rng.next_usize(3)];

            code.apply_error(error_qubit, error_type);

            // Full correction cycle
            let (detected, detected_type, success) = code.error_correction_cycle();

            if success {
                success_count += 1;
            } else {
                println!(
                    "Failed: applied {}@{}, detected {:?}@{}",
                    error_type, error_qubit, detected, detected_type
                );
            }
        }

        let success_rate = success_count as f64 / n_trials as f64 * 100.0;
        println!(
            "Steane code correction: {}/{} ({:.1}%) successful",
            success_count, n_trials, success_rate
        );

        // Single-qubit errors should always be correctable
        assert!(
            success_rate >= 90.0,
            "Steane code should correct single-qubit errors reliably"
        );
    }

    #[test]
    fn test_steane_noise_simulation() {
        use crate::qec::noise::SimpleRng;

        println!("\n=== Steane [[7,1,3]] Code Noise Simulation ===");
        println!("Can correct 1 error of any type (X, Y, or Z)");

        let n_trials = 1000;
        let mut rng = SimpleRng::new(77777);
        let error_types = ['X', 'Y', 'Z'];

        // Test various physical error rates
        for &error_prob in &[0.01, 0.02, 0.05, 0.10, 0.15] {
            let mut success_count = 0;
            let mut total_errors = 0;
            let mut logical_errors = 0;

            for _ in 0..n_trials {
                let mut code = SteaneCode::new();
                let mut n_errors = 0;

                // Apply random Pauli errors to each qubit
                for q in 0..7 {
                    if rng.next_f64() < error_prob {
                        let error_type = error_types[rng.next_usize(3)];
                        code.apply_error(q, error_type);
                        n_errors += 1;
                    }
                }

                total_errors += n_errors;

                // Measure, decode, correct
                let (x_syn, z_syn) = code.measure_syndrome();
                let (qubit, error_type) = code.decode_syndrome(&x_syn, &z_syn);
                code.correct(qubit, error_type);

                // Check if syndrome cleared
                let (x_after, z_after) = code.measure_syndrome();
                let syndrome_clear =
                    x_after.iter().all(|&s| s == 0) && z_after.iter().all(|&s| s == 0);

                if syndrome_clear {
                    success_count += 1;
                } else {
                    // Residual error = logical error
                    logical_errors += 1;
                }
            }

            let success_rate = success_count as f64 / n_trials as f64 * 100.0;
            let avg_errors = total_errors as f64 / n_trials as f64;
            let logical_rate = logical_errors as f64 / n_trials as f64 * 100.0;

            println!(
                "p={:.2}: success={:5.1}%, logical_err={:5.2}%, avg_errors={:.2}",
                error_prob, success_rate, logical_rate, avg_errors
            );

            // At very low error rates, should almost always succeed
            if error_prob <= 0.02 {
                assert!(
                    success_rate > 85.0,
                    "Low error rate should have high success"
                );
            }
        }
    }

    #[test]
    fn test_steane_threshold_behavior() {
        use crate::qec::noise::SimpleRng;

        println!("\n=== Steane Code Threshold Analysis ===");
        println!("Testing how error rate affects logical error rate\n");

        let n_trials = 500;
        let mut rng = SimpleRng::new(88888);
        let error_types = ['X', 'Y', 'Z'];

        println!(
            "{:>8} {:>12} {:>12} {:>15}",
            "p_phys", "p_logical", "Syndrome OK", "Avg Errors"
        );
        println!("{:-<8} {:-<12} {:-<12} {:-<15}", "", "", "", "");

        for &p_phys in &[0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20] {
            let mut logical_errors = 0;
            let mut syndrome_ok = 0;
            let mut total_errors = 0;

            for _ in 0..n_trials {
                let mut code = SteaneCode::new();
                let mut n_errors = 0;

                // Apply depolarizing-like noise (random X, Y, Z with prob p/3 each)
                for q in 0..7 {
                    if rng.next_f64() < p_phys {
                        let error_type = error_types[rng.next_usize(3)];
                        code.apply_error(q, error_type);
                        n_errors += 1;
                    }
                }

                total_errors += n_errors;

                // Error correction cycle
                let (_, _, success) = code.error_correction_cycle();

                if success {
                    syndrome_ok += 1;
                } else {
                    logical_errors += 1;
                }
            }

            let p_logical = logical_errors as f64 / n_trials as f64;
            let syndrome_rate = syndrome_ok as f64 / n_trials as f64 * 100.0;
            let avg_errors = total_errors as f64 / n_trials as f64;

            println!(
                "{:>8.3} {:>11.4} {:>11.1}% {:>14.2}",
                p_phys, p_logical, syndrome_rate, avg_errors
            );
        }

        println!("\nNote: Steane code corrects 1 error. At ~14% physical error rate,");
        println!("      probability of 2+ errors becomes significant → threshold.");
    }

    #[test]
    fn test_steane_error_weight_distribution() {
        use crate::qec::noise::SimpleRng;

        println!("\n=== Steane Code: Success vs Error Weight ===");

        let n_trials = 200;
        let mut rng = SimpleRng::new(99999);
        let error_types = ['X', 'Y', 'Z'];

        println!(
            "{:>12} {:>12} {:>12}",
            "Num Errors", "Success Rate", "Trials"
        );
        println!("{:-<12} {:-<12} {:-<12}", "", "", "");

        // Test with exactly 0, 1, 2, 3 errors
        for target_errors in 0..=4 {
            let mut success_count = 0;
            let mut trial_count = 0;

            for _ in 0..n_trials * 10 {
                // Generate random error pattern
                let mut code = SteaneCode::new();
                let mut error_positions: Vec<usize> = Vec::new();

                // Randomly select target_errors distinct qubits
                while error_positions.len() < target_errors {
                    let q = rng.next_usize(7);
                    if !error_positions.contains(&q) {
                        error_positions.push(q);
                    }
                }

                // Apply errors
                for &q in &error_positions {
                    let error_type = error_types[rng.next_usize(3)];
                    code.apply_error(q, error_type);
                }

                // Correct
                let (_, _, success) = code.error_correction_cycle();

                if success {
                    success_count += 1;
                }
                trial_count += 1;

                if trial_count >= n_trials {
                    break;
                }
            }

            let success_rate = success_count as f64 / trial_count as f64 * 100.0;
            println!(
                "{:>12} {:>11.1}% {:>12}",
                target_errors, success_rate, trial_count
            );
        }

        println!("\nExpected: 100% for 0-1 errors, <100% for 2+ errors");
    }

    #[test]
    fn test_noise_simulation() {
        use crate::qec::noise::SimpleRng;

        let distance = 11; // [[11, 1, 11]] code - corrects up to 5 errors
        let max_correctable = (distance - 1) / 2;
        let n_trials = 1000;
        let mut rng = SimpleRng::new(12345);

        println!(
            "\n=== Noise Simulation: [[{}, 1, {}]] Repetition Code ===",
            distance, distance
        );
        println!("Max correctable errors: {}", max_correctable);

        // Test various error rates
        for &error_prob in &[0.01, 0.05, 0.10, 0.20, 0.30] {
            let mut success_count = 0;
            let mut total_errors_applied = 0;
            let mut total_errors_decoded = 0;

            for _ in 0..n_trials {
                let mut code = RepetitionCode::new(distance);
                let mut actual_errors = Vec::new();

                // Apply random X errors
                for q in 0..distance {
                    if rng.next_f64() < error_prob {
                        code.apply_x_error(q);
                        actual_errors.push(q);
                    }
                }

                total_errors_applied += actual_errors.len();

                // Measure and decode
                let syndrome = code.measure_syndrome();
                let decoded_errors = code.decode_syndrome(&syndrome);
                total_errors_decoded += decoded_errors.len();

                // Apply correction
                code.correct(&decoded_errors);

                // Check if syndrome is cleared
                let final_syndrome = code.measure_syndrome();
                if final_syndrome.iter().all(|&s| s == 0) {
                    success_count += 1;
                }
            }

            let success_rate = success_count as f64 / n_trials as f64 * 100.0;
            let avg_errors = total_errors_applied as f64 / n_trials as f64;
            let avg_decoded = total_errors_decoded as f64 / n_trials as f64;

            println!(
                "p={:.2}: success={:5.1}%, avg_errors={:.2}, avg_decoded={:.2}",
                error_prob, success_rate, avg_errors, avg_decoded
            );

            // At low error rates, should succeed most of the time
            if error_prob <= 0.05 {
                assert!(
                    success_rate > 80.0,
                    "Low error rate should have high success"
                );
            }
        }
    }

    #[test]
    fn test_threshold_estimation() {
        use crate::qec::noise::SimpleRng;

        // Test error correction threshold
        // For repetition code, threshold is around p = 0.5 for X errors
        // But with our decoder, effective threshold is lower due to multi-error patterns

        let n_trials = 500;
        let mut rng = SimpleRng::new(54321);

        println!("\n=== Threshold Estimation ===");
        println!("Distance | p=0.05 | p=0.10 | p=0.15 | p=0.20");
        println!("---------+--------+--------+--------+--------");

        for distance in [5, 9, 13, 17, 21] {
            let _max_correctable = (distance - 1) / 2;
            let mut results = Vec::new();

            for &error_prob in &[0.05, 0.10, 0.15, 0.20] {
                let mut success_count = 0;

                for _ in 0..n_trials {
                    let mut code = RepetitionCode::new(distance);

                    // Apply random X errors
                    for q in 0..distance {
                        if rng.next_f64() < error_prob {
                            code.apply_x_error(q);
                        }
                    }

                    // Decode and correct
                    let syndrome = code.measure_syndrome();
                    let decoded_errors = code.decode_syndrome(&syndrome);
                    code.correct(&decoded_errors);

                    // Check success
                    let final_syndrome = code.measure_syndrome();
                    if final_syndrome.iter().all(|&s| s == 0) {
                        success_count += 1;
                    }
                }

                results.push(success_count as f64 / n_trials as f64 * 100.0);
            }

            println!(
                "   {:2}    | {:5.1}% | {:5.1}% | {:5.1}% | {:5.1}%",
                distance, results[0], results[1], results[2], results[3]
            );
        }

        println!("\nExpected: Success rate should increase with distance at low p");
    }

    #[test]
    fn test_logical_error_rate() {
        use crate::qec::noise::SimpleRng;

        // After correction, check if the logical state is preserved
        // For repetition code: logical Z = Z on any qubit
        // Logical error occurs if odd number of residual errors

        let distance = 11;
        let n_trials = 1000;
        let mut rng = SimpleRng::new(99999);

        println!("\n=== Logical Error Rate Analysis ===");

        for &error_prob in &[0.02, 0.05, 0.08, 0.10] {
            let mut logical_errors = 0;
            let mut correction_failures = 0;

            for _ in 0..n_trials {
                let mut code = RepetitionCode::new(distance);
                let mut _actual_error_count = 0;

                // Apply random X errors
                for q in 0..distance {
                    if rng.next_f64() < error_prob {
                        code.apply_x_error(q);
                        _actual_error_count += 1;
                    }
                }

                // Decode and correct
                let syndrome = code.measure_syndrome();
                let decoded_errors = code.decode_syndrome(&syndrome);
                code.correct(&decoded_errors);

                // Check syndrome
                let final_syndrome = code.measure_syndrome();
                if !final_syndrome.iter().all(|&s| s == 0) {
                    correction_failures += 1;
                }

                // Check logical error: count residual errors
                // by re-measuring the state
                // Logical error = odd number of uncorrected X errors
                // This is approximated by checking if Z₀ eigenvalue flipped
                let z0_phase = code.tableau.stabilizer(0).phase;
                if z0_phase == 2 {
                    // Logical bit flipped
                    logical_errors += 1;
                }
            }

            let logical_rate = logical_errors as f64 / n_trials as f64 * 100.0;
            let correction_rate = (n_trials - correction_failures) as f64 / n_trials as f64 * 100.0;

            println!(
                "p={:.2}: logical_error_rate={:5.2}%, syndrome_clear_rate={:5.1}%",
                error_prob, logical_rate, correction_rate
            );
        }
    }

    // =========================================================================
    // Surface Code Tests
    // =========================================================================

    #[test]
    fn test_surface_code_stabilizer_structure() {
        println!("\n=== Surface Code Stabilizer Structure ===");

        for distance in [3, 5, 7] {
            let code = SurfaceCode::new(distance);
            println!(
                "Distance {}: {} data qubits, {} X stabilizers, {} Z stabilizers",
                distance,
                code.n_data,
                code.num_x_stabilizers(),
                code.num_z_stabilizers()
            );

            // Total stabilizers should be n_data - 1 (one logical qubit)
            let total_stabs = code.num_x_stabilizers() + code.num_z_stabilizers();
            println!(
                "  Total stabilizers: {} (expected {})",
                total_stabs,
                code.n_data - 1
            );
        }
    }

    #[test]
    fn test_surface_code_single_x_error() {
        println!("\n=== Surface Code Single X Error ===");

        let mut code = SurfaceCode::new(3);
        println!("Distance 3 surface code: {} qubits", code.n_data);
        println!(
            "X stabilizers: {}, Z stabilizers: {}",
            code.num_x_stabilizers(),
            code.num_z_stabilizers()
        );

        // Apply X error at center qubit (1,1) = qubit 4
        let error_qubit = 4;
        code.apply_x_error_at(error_qubit);
        println!("Applied X error at qubit {}", error_qubit);

        // Measure syndrome
        let (x_syn, z_syn) = code.measure_syndrome();
        println!("X syndrome (detects Z): {:?}", x_syn);
        println!("Z syndrome (detects X): {:?}", z_syn);

        // Z syndrome should be non-trivial
        assert!(
            z_syn.iter().any(|&s| s == 1),
            "Z syndrome should detect X error"
        );
        assert!(
            x_syn.iter().all(|&s| s == 0),
            "X syndrome should be trivial for X error"
        );

        // Apply correction
        let (success, x_corr, z_corr) = code.error_correction_cycle();
        println!(
            "Correction: X={:?}, Z={:?}, success={}",
            x_corr, z_corr, success
        );
    }

    #[test]
    fn test_surface_code_single_z_error() {
        println!("\n=== Surface Code Single Z Error ===");

        let mut code = SurfaceCode::new(3);

        // Apply Z error at center qubit
        let error_qubit = 4;
        code.apply_z_error_at(error_qubit);
        println!("Applied Z error at qubit {}", error_qubit);

        // Measure syndrome
        let (x_syn, z_syn) = code.measure_syndrome();
        println!("X syndrome (detects Z): {:?}", x_syn);
        println!("Z syndrome (detects X): {:?}", z_syn);

        // X syndrome should be non-trivial
        assert!(
            x_syn.iter().any(|&s| s == 1),
            "X syndrome should detect Z error"
        );
        assert!(
            z_syn.iter().all(|&s| s == 0),
            "Z syndrome should be trivial for Z error"
        );

        // Apply correction
        let (success, x_corr, z_corr) = code.error_correction_cycle();
        println!(
            "Correction: X={:?}, Z={:?}, success={}",
            x_corr, z_corr, success
        );
    }

    #[test]
    fn test_surface_code_all_single_errors() {
        println!("\n=== Surface Code: All Single Qubit Errors ===");

        let distance = 3;
        let n_data = distance * distance;

        for error_type in ['X', 'Y', 'Z'] {
            let mut successes = 0;

            for q in 0..n_data {
                let mut code = SurfaceCode::new(distance);
                code.apply_error(q, error_type);

                let (success, _, _) = code.error_correction_cycle();
                if success {
                    successes += 1;
                }
            }

            println!(
                "{} errors: {}/{} syndrome-cleared ({:.1}%)",
                error_type,
                successes,
                n_data,
                successes as f64 / n_data as f64 * 100.0
            );
        }
    }

    #[test]
    fn test_surface_code_noise_simulation() {
        use crate::qec::noise::SimpleRng;

        println!("\n=== Surface Code Noise Simulation ===");

        let n_trials = 500;
        let mut rng = SimpleRng::new(12345);
        let error_types = ['X', 'Y', 'Z'];

        // Test different distances
        for distance in [3, 5] {
            println!(
                "\nDistance {} surface code ({} qubits):",
                distance,
                distance * distance
            );

            for &error_prob in &[0.01, 0.02, 0.05, 0.10] {
                let mut success_count = 0;
                let mut logical_x_errors = 0;
                let mut logical_z_errors = 0;
                let mut total_errors = 0;

                for _ in 0..n_trials {
                    let mut code = SurfaceCode::new(distance);
                    let mut n_errors = 0;

                    // Apply random Pauli errors
                    for q in 0..(distance * distance) {
                        if rng.next_f64() < error_prob {
                            let error_type = error_types[rng.next_usize(3)];
                            code.apply_error(q, error_type);
                            n_errors += 1;
                        }
                    }

                    total_errors += n_errors;

                    // Error correction
                    let (success, _, _) = code.error_correction_cycle();

                    if success {
                        success_count += 1;
                    }

                    // Check for logical errors
                    if code.has_logical_x_error() {
                        logical_x_errors += 1;
                    }
                    if code.has_logical_z_error() {
                        logical_z_errors += 1;
                    }
                }

                let success_rate = success_count as f64 / n_trials as f64 * 100.0;
                let avg_errors = total_errors as f64 / n_trials as f64;
                let logical_x_rate = logical_x_errors as f64 / n_trials as f64 * 100.0;
                let logical_z_rate = logical_z_errors as f64 / n_trials as f64 * 100.0;

                println!(
                    "  p={:.2}: syndrome_ok={:5.1}%, logical_X={:4.1}%, logical_Z={:4.1}%, avg_err={:.1}",
                    error_prob, success_rate, logical_x_rate, logical_z_rate, avg_errors
                );

                // At low error rates, should have high success
                if error_prob <= 0.02 && distance >= 3 {
                    assert!(
                        success_rate > 60.0,
                        "Low error rate should have reasonable success"
                    );
                }
            }
        }
    }

    #[test]
    fn test_surface_code_distance_scaling() {
        use crate::qec::noise::SimpleRng;

        println!("\n=== Surface Code Distance Scaling ===");
        println!("Testing how larger codes improve error correction\n");

        let n_trials = 300;
        let mut rng = SimpleRng::new(54321);
        let error_types = ['X', 'Y', 'Z'];
        let error_prob = 0.05;

        println!(
            "{:>10} {:>10} {:>12} {:>12}",
            "Distance", "Qubits", "Syndrome OK", "Logical Err"
        );
        println!("{:-<10} {:-<10} {:-<12} {:-<12}", "", "", "", "");

        for distance in [3, 5, 7] {
            let n_data = distance * distance;
            let mut syndrome_ok = 0;
            let mut logical_errors = 0;

            for _ in 0..n_trials {
                let mut code = SurfaceCode::new(distance);

                // Apply random errors
                for q in 0..n_data {
                    if rng.next_f64() < error_prob {
                        let error_type = error_types[rng.next_usize(3)];
                        code.apply_error(q, error_type);
                    }
                }

                // Correct
                let (success, _, _) = code.error_correction_cycle();

                if success {
                    syndrome_ok += 1;
                }

                // Check logical errors
                if code.has_logical_x_error() || code.has_logical_z_error() {
                    logical_errors += 1;
                }
            }

            let syndrome_rate = syndrome_ok as f64 / n_trials as f64 * 100.0;
            let logical_rate = logical_errors as f64 / n_trials as f64 * 100.0;

            println!(
                "{:>10} {:>10} {:>11.1}% {:>11.1}%",
                distance, n_data, syndrome_rate, logical_rate
            );
        }

        println!("\nExpected: Larger distance should reduce logical error rate");
    }

    #[test]
    fn test_surface_code_error_weight() {
        use crate::qec::noise::SimpleRng;

        println!("\n=== Surface Code: Success vs Error Weight ===");

        let distance = 5;
        let n_trials = 100;
        let mut rng = SimpleRng::new(99999);
        let error_types = ['X', 'Y', 'Z'];
        let n_data = distance * distance;
        let max_errors = (distance - 1) / 2; // Guaranteed correctable

        println!(
            "Distance {} code can correct up to {} errors",
            distance, max_errors
        );
        println!(
            "{:>12} {:>15} {:>12}",
            "Num Errors", "Syndrome OK", "Trials"
        );
        println!("{:-<12} {:-<15} {:-<12}", "", "", "");

        for target_errors in 0..=3.min(max_errors + 1) {
            let mut success_count = 0;
            let mut trial_count = 0;

            for _ in 0..(n_trials * 5) {
                if trial_count >= n_trials {
                    break;
                }

                // Generate random error pattern with exactly target_errors errors
                let mut code = SurfaceCode::new(distance);
                let mut error_positions: Vec<usize> = Vec::new();

                while error_positions.len() < target_errors {
                    let q = rng.next_usize(n_data);
                    if !error_positions.contains(&q) {
                        error_positions.push(q);
                    }
                }

                for &q in &error_positions {
                    let error_type = error_types[rng.next_usize(3)];
                    code.apply_error(q, error_type);
                }

                // Correct
                let (success, _, _) = code.error_correction_cycle();

                if success {
                    success_count += 1;
                }
                trial_count += 1;
            }

            let success_rate = success_count as f64 / trial_count as f64 * 100.0;
            println!(
                "{:>12} {:>14.1}% {:>12}",
                target_errors, success_rate, trial_count
            );
        }
    }
}
