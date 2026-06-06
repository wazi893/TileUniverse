//! Bacon-Shor Subsystem Code
//!
//! The Bacon-Shor code is a subsystem code that encodes 1 logical qubit into d² physical
//! qubits arranged in a d×d lattice. It achieves distance d error correction using only
//! weight-2 gauge measurements.
//!
//! Key properties:
//! - [[d², 1, d]] parameters
//! - Gauge operators: XX on horizontal pairs, ZZ on vertical pairs
//! - Stabilizers are products of gauge operators (row/column products)
//! - Decoding reduces to independent 1D repetition code decoders
//!
//! Reference: Bacon, "Operator quantum error-correcting subsystems for self-correcting
//! quantum memories" (2006)

/// Gauge operator type for Bacon-Shor code
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaugePauliType {
    /// XX gauge operator (horizontal pairs, detects Z errors)
    XX,
    /// ZZ gauge operator (vertical pairs, detects X errors)
    ZZ,
}

/// A weight-2 gauge operator acting on two adjacent qubits
#[derive(Clone, Debug)]
pub struct GaugeOperator {
    /// First qubit index
    pub q1: usize,
    /// Second qubit index
    pub q2: usize,
    /// Type of gauge operator (XX or ZZ)
    pub pauli_type: GaugePauliType,
    /// Row index for this gauge (for XX gauges) or column index (for ZZ gauges)
    pub major_index: usize,
    /// Position within the row/column (0 to d-2)
    pub minor_index: usize,
}

/// Bacon-Shor [[d², 1, d]] subsystem code on a d×d lattice
///
/// The code encodes 1 logical qubit into d² physical qubits.
/// Distance d means it can correct up to floor((d-1)/2) errors.
///
/// Layout for d=3:
/// ```text
///   0 - 1 - 2    (row 0)
///   |   |   |
///   3 - 4 - 5    (row 1)
///   |   |   |
///   6 - 7 - 8    (row 2)
///
///   Horizontal pairs have XX gauge operators
///   Vertical pairs have ZZ gauge operators
/// ```
///
/// Gauge operators (weight-2):
/// - XX gauges: d(d-1) operators on horizontal pairs
/// - ZZ gauges: d(d-1) operators on vertical pairs
///
/// Stabilizers (derived from gauge products):
/// - X stabilizer for row i = product of all XX gauges in row i → X on entire row
/// - Z stabilizer for column j = product of all ZZ gauges in column j → Z on entire column
///
/// Logical operators:
/// - Logical X = X on any complete row (e.g., X₀X₁X₂ for row 0)
/// - Logical Z = Z on any complete column (e.g., Z₀Z₃Z₆ for column 0)
#[derive(Clone, Debug)]
pub struct BaconShorCode {
    /// Code distance (must be odd >= 3)
    pub distance: usize,
    /// Total number of physical qubits (d²)
    pub n_qubits: usize,
    /// X error tracking (true = X error present on qubit)
    x_errors: Vec<bool>,
    /// Z error tracking (true = Z error present on qubit)
    z_errors: Vec<bool>,
    /// Horizontal XX gauge operators (d rows × (d-1) gauges per row)
    xx_gauges: Vec<GaugeOperator>,
    /// Vertical ZZ gauge operators (d columns × (d-1) gauges per column)
    zz_gauges: Vec<GaugeOperator>,
}

impl BaconShorCode {
    /// Create a new Bacon-Shor code with the given distance
    ///
    /// # Arguments
    /// * `distance` - Code distance, must be odd and >= 3
    ///
    /// # Panics
    /// Panics if distance is even or less than 3
    ///
    /// # Example
    /// ```
    /// use engine::qec::bacon_shor::BaconShorCode;
    /// let code = BaconShorCode::new(3);
    /// assert_eq!(code.n_qubits, 9);
    /// assert_eq!(code.distance, 3);
    /// ```
    pub fn new(distance: usize) -> Self {
        assert!(
            distance >= 3,
            "Bacon-Shor code distance must be at least 3, got {}",
            distance
        );
        assert!(
            distance % 2 == 1,
            "Bacon-Shor code distance must be odd, got {}",
            distance
        );

        let n_qubits = distance * distance;

        // Build XX gauges (horizontal pairs)
        // For each row, we have (d-1) horizontal pairs
        let mut xx_gauges = Vec::with_capacity(distance * (distance - 1));
        for row in 0..distance {
            for col in 0..(distance - 1) {
                let q1 = row * distance + col;
                let q2 = row * distance + col + 1;
                xx_gauges.push(GaugeOperator {
                    q1,
                    q2,
                    pauli_type: GaugePauliType::XX,
                    major_index: row,
                    minor_index: col,
                });
            }
        }

        // Build ZZ gauges (vertical pairs)
        // For each column, we have (d-1) vertical pairs
        let mut zz_gauges = Vec::with_capacity(distance * (distance - 1));
        for col in 0..distance {
            for row in 0..(distance - 1) {
                let q1 = row * distance + col;
                let q2 = (row + 1) * distance + col;
                zz_gauges.push(GaugeOperator {
                    q1,
                    q2,
                    pauli_type: GaugePauliType::ZZ,
                    major_index: col,
                    minor_index: row,
                });
            }
        }

        Self {
            distance,
            n_qubits,
            x_errors: vec![false; n_qubits],
            z_errors: vec![false; n_qubits],
            xx_gauges,
            zz_gauges,
        }
    }

    /// Get qubit index from (row, col) coordinates
    ///
    /// # Arguments
    /// * `row` - Row index (0 to d-1)
    /// * `col` - Column index (0 to d-1)
    #[inline]
    pub fn qubit_index(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.distance);
        debug_assert!(col < self.distance);
        row * self.distance + col
    }

    /// Get (row, col) coordinates from qubit index
    #[inline]
    pub fn qubit_coords(&self, qubit: usize) -> (usize, usize) {
        debug_assert!(qubit < self.n_qubits);
        (qubit / self.distance, qubit % self.distance)
    }

    /// Apply X error to a specific qubit
    #[inline]
    pub fn apply_x_error(&mut self, qubit: usize) {
        debug_assert!(qubit < self.n_qubits);
        self.x_errors[qubit] ^= true;
    }

    /// Apply Z error to a specific qubit
    #[inline]
    pub fn apply_z_error(&mut self, qubit: usize) {
        debug_assert!(qubit < self.n_qubits);
        self.z_errors[qubit] ^= true;
    }

    /// Apply an error specified by Pauli character ('X', 'Y', 'Z', or 'I')
    ///
    /// Y error = XZ (applies both X and Z errors)
    pub fn apply_error(&mut self, qubit: usize, pauli: char) {
        assert!(qubit < self.n_qubits, "Qubit {} out of range", qubit);
        match pauli {
            'X' | 'x' => self.apply_x_error(qubit),
            'Z' | 'z' => self.apply_z_error(qubit),
            'Y' | 'y' => {
                self.apply_x_error(qubit);
                self.apply_z_error(qubit);
            }
            'I' | 'i' => {} // Identity, do nothing
            _ => panic!(
                "Invalid Pauli operator: '{}'. Use 'X', 'Y', 'Z', or 'I'",
                pauli
            ),
        }
    }

    /// Measure all XX gauge operators
    ///
    /// XX gauge on (q1, q2) detects Z errors: outcome = (-1)^(z_errors[q1] XOR z_errors[q2])
    ///
    /// Returns a vector of outcomes:
    /// - +1 if no Z error boundary between q1 and q2
    /// - -1 if Z error on exactly one of q1, q2
    pub fn measure_xx_gauges(&self) -> Vec<i8> {
        self.xx_gauges
            .iter()
            .map(|gauge| {
                let z1 = self.z_errors[gauge.q1];
                let z2 = self.z_errors[gauge.q2];
                if z1 ^ z2 { -1 } else { 1 }
            })
            .collect()
    }

    /// Measure all ZZ gauge operators
    ///
    /// ZZ gauge on (q1, q2) detects X errors: outcome = (-1)^(x_errors[q1] XOR x_errors[q2])
    ///
    /// Returns a vector of outcomes:
    /// - +1 if no X error boundary between q1 and q2
    /// - -1 if X error on exactly one of q1, q2
    pub fn measure_zz_gauges(&self) -> Vec<i8> {
        self.zz_gauges
            .iter()
            .map(|gauge| {
                let x1 = self.x_errors[gauge.q1];
                let x2 = self.x_errors[gauge.q2];
                if x1 ^ x2 { -1 } else { 1 }
            })
            .collect()
    }

    /// Get the indices of XX gauges in a specific row
    fn xx_gauge_indices_for_row(&self, row: usize) -> std::ops::Range<usize> {
        let start = row * (self.distance - 1);
        let end = start + (self.distance - 1);
        start..end
    }

    /// Get the indices of ZZ gauges in a specific column
    fn zz_gauge_indices_for_col(&self, col: usize) -> std::ops::Range<usize> {
        let start = col * (self.distance - 1);
        let end = start + (self.distance - 1);
        start..end
    }

    /// Compute X stabilizer syndrome for a specific row from XX gauge outcomes
    ///
    /// The X stabilizer for row i is the product of all XX gauges in that row.
    /// If all XX gauges in the row have outcome +1, the stabilizer has eigenvalue +1.
    /// Otherwise, the eigenvalue is the product of all gauge outcomes.
    ///
    /// # Arguments
    /// * `xx_outcomes` - Results from measure_xx_gauges()
    /// * `row` - Row index (0 to d-1)
    ///
    /// # Returns
    /// +1 or -1 for the stabilizer eigenvalue
    pub fn x_stabilizer_from_gauges(&self, xx_outcomes: &[i8], row: usize) -> i8 {
        let mut product = 1i8;
        for idx in self.xx_gauge_indices_for_row(row) {
            product *= xx_outcomes[idx];
        }
        product
    }

    /// Compute Z stabilizer syndrome for a specific column from ZZ gauge outcomes
    ///
    /// The Z stabilizer for column j is the product of all ZZ gauges in that column.
    ///
    /// # Arguments
    /// * `zz_outcomes` - Results from measure_zz_gauges()
    /// * `col` - Column index (0 to d-1)
    ///
    /// # Returns
    /// +1 or -1 for the stabilizer eigenvalue
    pub fn z_stabilizer_from_gauges(&self, zz_outcomes: &[i8], col: usize) -> i8 {
        let mut product = 1i8;
        for idx in self.zz_gauge_indices_for_col(col) {
            product *= zz_outcomes[idx];
        }
        product
    }

    /// Measure the full X stabilizer syndrome (one value per row)
    ///
    /// Returns d values, one for each row. Each value is +1 or -1.
    /// These detect Z errors that span partial rows.
    pub fn measure_x_syndrome(&self) -> Vec<i8> {
        let xx = self.measure_xx_gauges();
        (0..self.distance)
            .map(|row| self.x_stabilizer_from_gauges(&xx, row))
            .collect()
    }

    /// Measure the full Z stabilizer syndrome (one value per column)
    ///
    /// Returns d values, one for each column. Each value is +1 or -1.
    /// These detect X errors that span partial columns.
    pub fn measure_z_syndrome(&self) -> Vec<i8> {
        let zz = self.measure_zz_gauges();
        (0..self.distance)
            .map(|col| self.z_stabilizer_from_gauges(&zz, col))
            .collect()
    }

    /// Apply logical X operator (X on entire row 0)
    ///
    /// The logical X operator is X applied to all qubits in any row.
    /// We use row 0 by convention.
    pub fn apply_logical_x(&mut self) {
        for col in 0..self.distance {
            let q = self.qubit_index(0, col);
            self.apply_x_error(q);
        }
    }

    /// Apply logical Z operator (Z on entire column 0)
    ///
    /// The logical Z operator is Z applied to all qubits in any column.
    /// We use column 0 by convention.
    pub fn apply_logical_z(&mut self) {
        for row in 0..self.distance {
            let q = self.qubit_index(row, 0);
            self.apply_z_error(q);
        }
    }

    /// Check if there's a logical X error
    ///
    /// A logical X error occurs when X errors span from top to bottom of the lattice
    /// in any column. This is equivalent to having an odd number of X errors in
    /// any complete column.
    pub fn has_logical_x_error(&self) -> bool {
        // Check each column for odd parity of X errors
        for col in 0..self.distance {
            let mut parity = false;
            for row in 0..self.distance {
                let q = self.qubit_index(row, col);
                if self.x_errors[q] {
                    parity = !parity;
                }
            }
            if parity {
                return true;
            }
        }
        false
    }

    /// Check if there's a logical Z error
    ///
    /// A logical Z error occurs when Z errors span from left to right of the lattice
    /// in any row. This is equivalent to having an odd number of Z errors in
    /// any complete row.
    pub fn has_logical_z_error(&self) -> bool {
        // Check each row for odd parity of Z errors
        for row in 0..self.distance {
            let mut parity = false;
            for col in 0..self.distance {
                let q = self.qubit_index(row, col);
                if self.z_errors[q] {
                    parity = !parity;
                }
            }
            if parity {
                return true;
            }
        }
        false
    }

    /// Perform one round of error correction
    ///
    /// This measures all gauge operators, decodes using 1D repetition code decoders
    /// for each row (X errors) and column (Z errors), and applies corrections.
    ///
    /// # Returns
    /// `true` if no logical error occurred after correction, `false` otherwise
    pub fn error_correction_round(&mut self) -> bool {
        let decoder = BaconShorDecoder::new(self.distance);

        // Measure gauge operators
        let xx_outcomes = self.measure_xx_gauges();
        let zz_outcomes = self.measure_zz_gauges();

        // Decode X errors (detected by ZZ gauges) - one 1D decoder per column
        for col in 0..self.distance {
            // Extract the (d-1) ZZ gauge outcomes for this column
            let mut col_syndrome = Vec::with_capacity(self.distance - 1);
            for idx in self.zz_gauge_indices_for_col(col) {
                col_syndrome.push(zz_outcomes[idx]);
            }

            // Decode to find error positions within the column
            let error_rows = decoder.decode_1d(&col_syndrome);

            // Apply X corrections
            for row in error_rows {
                let q = self.qubit_index(row, col);
                self.apply_x_error(q);
            }
        }

        // Decode Z errors (detected by XX gauges) - one 1D decoder per row
        for row in 0..self.distance {
            // Extract the (d-1) XX gauge outcomes for this row
            let mut row_syndrome = Vec::with_capacity(self.distance - 1);
            for idx in self.xx_gauge_indices_for_row(row) {
                row_syndrome.push(xx_outcomes[idx]);
            }

            // Decode to find error positions within the row
            let error_cols = decoder.decode_1d(&row_syndrome);

            // Apply Z corrections
            for col in error_cols {
                let q = self.qubit_index(row, col);
                self.apply_z_error(q);
            }
        }

        // Check for logical errors
        !self.has_logical_x_error() && !self.has_logical_z_error()
    }

    /// Reset the code to error-free state (logical |0⟩)
    pub fn reset(&mut self) {
        self.x_errors.fill(false);
        self.z_errors.fill(false);
    }

    /// Get current X error pattern (for debugging/testing)
    pub fn x_error_pattern(&self) -> &[bool] {
        &self.x_errors
    }

    /// Get current Z error pattern (for debugging/testing)
    pub fn z_error_pattern(&self) -> &[bool] {
        &self.z_errors
    }

    /// Get reference to XX gauge operators
    pub fn xx_gauges(&self) -> &[GaugeOperator] {
        &self.xx_gauges
    }

    /// Get reference to ZZ gauge operators
    pub fn zz_gauges(&self) -> &[GaugeOperator] {
        &self.zz_gauges
    }

    /// Get the number of gauge operators
    pub fn num_gauges(&self) -> usize {
        self.xx_gauges.len() + self.zz_gauges.len()
    }

    /// Get the maximum number of errors that can be corrected
    ///
    /// For a distance-d code, this is floor((d-1)/2)
    pub fn max_correctable_errors(&self) -> usize {
        (self.distance - 1) / 2
    }
}

/// Decoder for Bacon-Shor code using 1D repetition code strategy
///
/// The Bacon-Shor code has a unique property: decoding reduces to independent
/// 1D repetition code decoding for each row (Z errors) and column (X errors).
///
/// This decoder implements the simple 1D strategy:
/// 1. Find positions where syndrome indicates error boundaries
/// 2. Pair up boundaries and correct qubits between pairs
#[derive(Clone, Debug)]
pub struct BaconShorDecoder {
    distance: usize,
}

impl BaconShorDecoder {
    /// Create a new decoder for a Bacon-Shor code with given distance
    pub fn new(distance: usize) -> Self {
        Self { distance }
    }

    /// Decode a 1D syndrome to find error positions
    ///
    /// The syndrome has (d-1) entries, where syndrome[i] = -1 indicates an error
    /// boundary between positions i and i+1.
    ///
    /// # Algorithm
    /// 1. Find all positions where syndrome is -1 (error boundaries)
    /// 2. Pair up consecutive boundaries
    /// 3. For each pair (i, j), there's an error at some position in [i+1, j]
    ///    We choose the position closest to i+1 (minimum weight correction)
    ///
    /// # Arguments
    /// * `syndrome` - Array of +1/-1 values, length d-1
    ///
    /// # Returns
    /// Vector of qubit positions (0 to d-1) where corrections should be applied
    pub fn decode_1d(&self, syndrome: &[i8]) -> Vec<usize> {
        assert_eq!(
            syndrome.len(),
            self.distance - 1,
            "Syndrome length must be d-1"
        );

        // Find all error boundaries (positions where syndrome is -1)
        let mut boundaries: Vec<usize> = syndrome
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if s == -1 { Some(i) } else { None })
            .collect();

        // If no boundaries, no errors
        if boundaries.is_empty() {
            return Vec::new();
        }

        let mut corrections = Vec::new();

        // Pair up boundaries
        // If odd number of boundaries, the last one extends to a virtual boundary at -1 or d-1
        // This handles errors at the edges of the lattice
        if boundaries.len() % 2 == 1 {
            // Check if it's closer to the start or end
            let last = *boundaries.last().unwrap();
            if last < (self.distance - 1) / 2 {
                // Closer to start - prepend virtual boundary at position -1
                // Error is at position 0
                corrections.push(0);
                // Remove the first real boundary since we've handled it
                boundaries.remove(0);
            } else {
                // Closer to end - error is at position d-1
                corrections.push(self.distance - 1);
                // Remove the last boundary since we've handled it
                boundaries.pop();
            }
        }

        // Now we have an even number of boundaries - pair them up
        for pair in boundaries.chunks(2) {
            if pair.len() == 2 {
                let start = pair[0];
                let _end = pair[1];
                // Error is between positions start and end (exclusive of boundaries)
                // Apply correction at position start + 1 (minimum weight)
                corrections.push(start + 1);
            }
        }

        corrections
    }

    /// Alternative decoder: decode using majority logic
    ///
    /// For each position, count how many syndrome bits indicate it could be the error.
    /// Choose the position with the most evidence.
    ///
    /// This is more robust for multiple errors but slower.
    #[allow(dead_code)]
    pub fn decode_1d_majority(&self, syndrome: &[i8]) -> Vec<usize> {
        // Count votes for each position
        let mut votes = vec![0i32; self.distance];

        for (i, &s) in syndrome.iter().enumerate() {
            if s == -1 {
                // Boundary at position i means error could be at i or i+1
                votes[i] += 1;
                votes[i + 1] += 1;
            }
        }

        // Find positions with maximum votes (odd count indicates error)
        let mut corrections = Vec::new();
        for (pos, &v) in votes.iter().enumerate() {
            if v % 2 == 1 {
                corrections.push(pos);
            }
        }

        corrections
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bacon_shor_d3_creation() {
        let code = BaconShorCode::new(3);
        assert_eq!(code.distance, 3);
        assert_eq!(code.n_qubits, 9);
        // d=3: 3 rows * 2 XX gauges per row = 6 XX gauges
        assert_eq!(code.xx_gauges.len(), 6);
        // d=3: 3 cols * 2 ZZ gauges per col = 6 ZZ gauges
        assert_eq!(code.zz_gauges.len(), 6);
    }

    #[test]
    fn test_bacon_shor_d5_creation() {
        let code = BaconShorCode::new(5);
        assert_eq!(code.distance, 5);
        assert_eq!(code.n_qubits, 25);
        assert_eq!(code.xx_gauges.len(), 20); // 5 * 4
        assert_eq!(code.zz_gauges.len(), 20); // 5 * 4
    }

    #[test]
    #[should_panic(expected = "odd")]
    fn test_bacon_shor_even_distance_panics() {
        BaconShorCode::new(4);
    }

    #[test]
    #[should_panic(expected = "at least 3")]
    fn test_bacon_shor_small_distance_panics() {
        BaconShorCode::new(1);
    }

    #[test]
    fn test_qubit_coords() {
        let code = BaconShorCode::new(3);
        assert_eq!(code.qubit_index(0, 0), 0);
        assert_eq!(code.qubit_index(0, 2), 2);
        assert_eq!(code.qubit_index(1, 0), 3);
        assert_eq!(code.qubit_index(2, 2), 8);

        assert_eq!(code.qubit_coords(0), (0, 0));
        assert_eq!(code.qubit_coords(4), (1, 1));
        assert_eq!(code.qubit_coords(8), (2, 2));
    }

    #[test]
    fn test_gauge_structure() {
        let code = BaconShorCode::new(3);

        // Check XX gauges are horizontal pairs
        for gauge in &code.xx_gauges {
            assert_eq!(gauge.pauli_type, GaugePauliType::XX);
            // q2 should be q1 + 1 (horizontal neighbor)
            assert_eq!(gauge.q2, gauge.q1 + 1);
        }

        // Check ZZ gauges are vertical pairs
        for gauge in &code.zz_gauges {
            assert_eq!(gauge.pauli_type, GaugePauliType::ZZ);
            // q2 should be q1 + d (vertical neighbor)
            assert_eq!(gauge.q2, gauge.q1 + code.distance);
        }
    }

    #[test]
    fn test_no_errors_trivial_syndrome() {
        let code = BaconShorCode::new(3);

        // No errors should give all +1 syndrome
        let xx = code.measure_xx_gauges();
        let zz = code.measure_zz_gauges();

        assert!(xx.iter().all(|&s| s == 1));
        assert!(zz.iter().all(|&s| s == 1));

        let x_syn = code.measure_x_syndrome();
        let z_syn = code.measure_z_syndrome();

        assert!(x_syn.iter().all(|&s| s == 1));
        assert!(z_syn.iter().all(|&s| s == 1));
    }

    #[test]
    fn test_single_z_error_detection() {
        let mut code = BaconShorCode::new(3);

        // Apply Z error at center (qubit 4)
        code.apply_error(4, 'Z');

        // Z error should be detected by XX gauges
        let xx = code.measure_xx_gauges();

        // Qubit 4 is at (row=1, col=1)
        // It's part of XX gauges: (3,4) and (4,5) in row 1
        // Gauge for (3,4): index = 1*2 + 0 = 2
        // Gauge for (4,5): index = 1*2 + 1 = 3

        // The Z error at qubit 4 should flip gauges involving qubit 4
        let mut flipped_count = 0;
        for &s in &xx {
            if s == -1 {
                flipped_count += 1;
            }
        }
        assert!(flipped_count > 0, "Z error should flip some XX gauges");
    }

    #[test]
    fn test_single_x_error_detection() {
        let mut code = BaconShorCode::new(3);

        // Apply X error at center (qubit 4)
        code.apply_error(4, 'X');

        // X error should be detected by ZZ gauges
        let zz = code.measure_zz_gauges();

        let mut flipped_count = 0;
        for &s in &zz {
            if s == -1 {
                flipped_count += 1;
            }
        }
        assert!(flipped_count > 0, "X error should flip some ZZ gauges");
    }

    #[test]
    fn test_gauge_products_equal_stabilizers() {
        // This is a key algebraic consistency check:
        // The X stabilizer for a row should equal the product of all XX gauges in that row
        // The Z stabilizer for a column should equal the product of all ZZ gauges in that column

        let mut code = BaconShorCode::new(3);

        // Apply some errors
        code.apply_error(4, 'X');
        code.apply_error(2, 'Z');

        let xx = code.measure_xx_gauges();
        let zz = code.measure_zz_gauges();

        // Check that stabilizer syndrome matches gauge product
        for row in 0..code.distance {
            let stab = code.x_stabilizer_from_gauges(&xx, row);

            // Manual product of gauges in this row
            let mut manual_product = 1i8;
            for idx in code.xx_gauge_indices_for_row(row) {
                manual_product *= xx[idx];
            }

            assert_eq!(
                stab, manual_product,
                "X stabilizer should equal product of XX gauges"
            );
        }

        for col in 0..code.distance {
            let stab = code.z_stabilizer_from_gauges(&zz, col);

            let mut manual_product = 1i8;
            for idx in code.zz_gauge_indices_for_col(col) {
                manual_product *= zz[idx];
            }

            assert_eq!(
                stab, manual_product,
                "Z stabilizer should equal product of ZZ gauges"
            );
        }
    }

    #[test]
    fn test_logical_operators() {
        let mut code = BaconShorCode::new(3);

        // Initially no logical errors
        assert!(!code.has_logical_x_error());
        assert!(!code.has_logical_z_error());

        // Apply logical X (X on row 0)
        code.apply_logical_x();

        // Should show as logical X error (odd X parity in column)
        assert!(code.has_logical_x_error());
        assert!(!code.has_logical_z_error());

        // Reset
        code.reset();

        // Apply logical Z (Z on column 0)
        code.apply_logical_z();

        // Should show as logical Z error (odd Z parity in row)
        assert!(!code.has_logical_x_error());
        assert!(code.has_logical_z_error());
    }

    #[test]
    fn test_decoder_trivial() {
        let decoder = BaconShorDecoder::new(3);

        // No errors
        let syndrome = vec![1, 1];
        let corrections = decoder.decode_1d(&syndrome);
        assert!(corrections.is_empty());
    }

    #[test]
    fn test_decoder_single_interior_error() {
        let decoder = BaconShorDecoder::new(3);

        // Error at position 1 (middle) gives boundaries at 0 and 1
        // syndrome[0] = -1 (boundary between 0 and 1)
        // syndrome[1] = -1 (boundary between 1 and 2)
        let syndrome = vec![-1, -1];
        let corrections = decoder.decode_1d(&syndrome);

        // Should correct at position 1
        assert_eq!(corrections, vec![1]);
    }

    #[test]
    fn test_decoder_edge_error_left() {
        let decoder = BaconShorDecoder::new(3);

        // Error at position 0 gives boundary only at 0
        let syndrome = vec![-1, 1];
        let corrections = decoder.decode_1d(&syndrome);

        // Should correct at position 0
        assert_eq!(corrections, vec![0]);
    }

    #[test]
    fn test_decoder_edge_error_right() {
        let decoder = BaconShorDecoder::new(3);

        // Error at position 2 (last) gives boundary only at 1
        let syndrome = vec![1, -1];
        let corrections = decoder.decode_1d(&syndrome);

        // Should correct at position 2
        assert_eq!(corrections, vec![2]);
    }

    #[test]
    fn test_max_correctable_errors() {
        let code3 = BaconShorCode::new(3);
        assert_eq!(code3.max_correctable_errors(), 1);

        let code5 = BaconShorCode::new(5);
        assert_eq!(code5.max_correctable_errors(), 2);

        let code7 = BaconShorCode::new(7);
        assert_eq!(code7.max_correctable_errors(), 3);
    }
}
