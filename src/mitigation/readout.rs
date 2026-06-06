//! Readout Error Mitigation (REM)
//!
//! SPRINT 57.0: Correct measurement errors via calibration matrix inversion.
//!
//! # Algorithm
//!
//! 1. **Calibration**: For each basis state |j⟩ (j = 0..2^n):
//!    - Prepare state |j⟩
//!    - Measure many times with realistic noise
//!    - Record P(measure i | prepared j) in calibration matrix A
//!
//! 2. **Correction**: Given raw measurement counts:
//!    - Convert to probability vector b
//!    - Solve A·x = b for corrected probabilities x
//!    - Convert back to corrected counts
//!
//! # Complexity
//!
//! - **Calibration**: O(2^n × n_shots) measurements
//! - **Correction**: O(2^2n) matrix operations
//! - **Memory**: O(4^n) for n-qubit calibration matrix
//!
//! Practical limit: n ≤ 10 qubits for full calibration
//!
//! # References
//!
//! - Kandala et al., "Error mitigation extends the computational reach of a noisy quantum processor" (2019)
//! - Bravyi et al., "Mitigating measurement errors in multiqubit experiments" (2020)

use std::collections::HashMap;

use crate::hardware::HardwareProfile;
use logic_fabric_core::quantum::{GateOutcome, QGate, QRng, QState, apply_gate_scalar};

/// Calibration matrix: A[i][j] = P(measure i | prepared j)
///
/// For n qubits, this is a 2^n × 2^n matrix where:
/// - Column j represents preparing basis state |j⟩
/// - Row i represents measuring bitstring i
/// - Entry A[i][j] is the probability of measuring i when j was prepared
#[derive(Clone, Debug)]
pub struct CalibrationMatrix {
    /// Matrix entries: matrix[measured_state][prepared_state]
    pub matrix: Vec<Vec<f64>>,
    /// Number of qubits
    pub n_qubits: usize,
}

impl CalibrationMatrix {
    /// Get matrix dimension (2^n)
    pub fn dimension(&self) -> usize {
        self.matrix.len()
    }

    /// Check if matrix is approximately identity (no measurement errors)
    pub fn is_identity(&self, tolerance: f64) -> bool {
        let dim = self.dimension();
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (self.matrix[i][j] - expected).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }
}

/// Readout calibration for error mitigation
///
/// Contains the calibration matrix and provides methods to correct
/// measurement statistics.
#[derive(Clone, Debug)]
pub struct ReadoutCalibration {
    /// Calibration matrix
    pub calibration_matrix: CalibrationMatrix,
}

impl ReadoutCalibration {
    /// Build calibration matrix by measuring all basis states
    ///
    /// This is the calibration phase that must be run once before using
    /// readout error mitigation.
    ///
    /// # Arguments
    ///
    /// * `profile` - Hardware profile with measurement fidelity
    /// * `n_qubits` - Number of qubits to calibrate
    /// * `n_shots` - Number of shots per basis state (more = better accuracy)
    /// * `seed` - RNG seed for reproducibility
    ///
    /// # Complexity
    ///
    /// - Time: O(2^n × n_shots) quantum measurements
    /// - Memory: O(4^n) for calibration matrix
    ///
    /// # Example
    ///
    /// ```ignore
    /// let profile = HardwareProfile::ibm_falcon();
    /// let calibration = ReadoutCalibration::build(&profile, 4, 5000, 42);
    /// // Takes ~5 seconds for 4 qubits, 16 basis states × 5000 shots each
    /// ```
    pub fn build(profile: &HardwareProfile, n_qubits: usize, n_shots: usize, seed: u64) -> Self {
        assert!(
            n_qubits <= 12,
            "Readout calibration limited to 12 qubits (4096×4096 matrix)"
        );

        let dim = 1 << n_qubits; // 2^n basis states
        let mut matrix = vec![vec![0.0; dim]; dim];

        // For each basis state |j⟩
        for j in 0..dim {
            let mut counts: HashMap<usize, usize> = HashMap::new();

            // Prepare |j⟩ and measure n_shots times
            for shot in 0..n_shots {
                let state = prepare_basis_state(j, n_qubits);
                let outcome =
                    measure_with_noise(&state, profile, seed + (j * n_shots + shot) as u64);
                *counts.entry(outcome).or_insert(0) += 1;
            }

            // Fill column j of calibration matrix
            // matrix[i][j] = P(measure i | prepared j)
            for (measured_state, count) in &counts {
                matrix[*measured_state][j] = *count as f64 / n_shots as f64;
            }

            // Fill any missing entries with 0 (states never observed)
            for i in 0..dim {
                if !counts.contains_key(&i) {
                    matrix[i][j] = 0.0;
                }
            }
        }

        Self {
            calibration_matrix: CalibrationMatrix { matrix, n_qubits },
        }
    }

    /// Correct measurement counts using calibration matrix
    ///
    /// Solves the linear system A·x = b where:
    /// - A is the calibration matrix
    /// - b is the measured probability distribution
    /// - x is the corrected probability distribution
    ///
    /// # Arguments
    ///
    /// * `raw_counts` - Raw measurement counts from circuit execution
    ///
    /// # Returns
    ///
    /// Corrected counts as floating-point numbers (may not be integers)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut raw_counts = HashMap::new();
    /// raw_counts.insert(0b00, 450);
    /// raw_counts.insert(0b01, 50);
    /// raw_counts.insert(0b10, 50);
    /// raw_counts.insert(0b11, 450);
    ///
    /// let corrected = calibration.correct_counts(&raw_counts);
    /// // corrected[0b00] ≈ 500, corrected[0b11] ≈ 500 (closer to ideal Bell state)
    /// ```
    pub fn correct_counts(&self, raw_counts: &HashMap<u64, usize>) -> HashMap<u64, f64> {
        let dim = self.calibration_matrix.dimension();
        let total: usize = raw_counts.values().sum();

        if total == 0 {
            return HashMap::new();
        }

        // Convert counts to probability vector
        let mut prob_vec = vec![0.0; dim];
        for (state, count) in raw_counts {
            let idx = *state as usize;
            if idx < dim {
                prob_vec[idx] = *count as f64 / total as f64;
            }
        }

        // Solve A·x = b via least-squares: x = (A^T A)^-1 A^T b
        let corrected_probs = solve_linear_system_ls(&self.calibration_matrix.matrix, &prob_vec);

        // Convert back to counts (preserving total)
        let mut result = HashMap::new();
        for (state, &prob) in corrected_probs.iter().enumerate() {
            // Negative probabilities can occur due to numerical issues
            // Clamp to [0, 1] for physical validity
            let clamped_prob = prob.max(0.0).min(1.0);

            if clamped_prob > 1e-10 {
                result.insert(state as u64, clamped_prob * total as f64);
            }
        }

        result
    }

    /// Get expected improvement factor
    ///
    /// Estimates how much measurement fidelity improves with mitigation.
    /// Based on the condition number of the calibration matrix.
    pub fn expected_improvement(&self) -> f64 {
        // Rough estimate: improvement scales with measurement fidelity
        // For identity matrix (no errors): 1.0 (no improvement needed)
        // For noisy measurements: 1.2-2.0× improvement typical
        let dim = self.calibration_matrix.dimension();

        let mut off_diagonal_sum = 0.0;
        for i in 0..dim {
            for j in 0..dim {
                if i != j {
                    off_diagonal_sum += self.calibration_matrix.matrix[i][j];
                }
            }
        }

        let avg_error = off_diagonal_sum / (dim * dim) as f64;
        1.0 + avg_error * 5.0 // Rough heuristic
    }
}

/// Prepare computational basis state |j⟩
///
/// For j = 5 = 0b101, prepares |101⟩ by applying X gates to qubits 0 and 2.
fn prepare_basis_state(j: usize, n_qubits: usize) -> QState {
    let mut state = QState::new_zero(n_qubits as u8);
    let mut rng = QRng::new(0); // Deterministic for state preparation

    // Apply X gate to qubits where j has a 1 bit
    for qubit in 0..n_qubits {
        if (j >> qubit) & 1 == 1 {
            apply_gate_scalar(&mut state, &QGate::X(qubit as u8), &mut rng);
        }
    }

    state
}

/// Measure all qubits with realistic measurement noise
///
/// Applies bit-flip errors according to `profile.measure_fidelity`.
///
/// # Returns
///
/// Integer representing measured bitstring (qubit 0 is LSB)
fn measure_with_noise(state: &QState, profile: &HardwareProfile, seed: u64) -> usize {
    let mut rng = QRng::new(seed);
    let mut outcome = 0;
    let mut measurement_state = state.clone();

    // Measure each qubit
    for q in 0..state.n_qubits {
        let result = apply_gate_scalar(&mut measurement_state, &QGate::Measure(q), &mut rng);

        if let GateOutcome::Measured { bit, .. } = result {
            // Apply measurement error: flip bit with probability (1 - measure_fidelity)
            let error_prob = (1.0 - profile.measure_fidelity) as f32;
            let flip = rng.next_f32() < error_prob;
            let final_bit = if flip { 1 - bit } else { bit };

            if final_bit == 1 {
                outcome |= 1 << q;
            }
        }
    }

    outcome
}

/// Solve linear system via least-squares: x = (A^T A)^-1 A^T b
///
/// For overdetermined or ill-conditioned systems, this is more stable
/// than direct Gaussian elimination.
fn solve_linear_system_ls(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let m = a.len(); // Number of rows
    let n = a[0].len(); // Number of columns

    // Compute A^T A
    let mut ata = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += a[k][i] * a[k][j];
            }
            ata[i][j] = sum;
        }
    }

    // Compute A^T b
    let mut atb = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..m {
            sum += a[k][i] * b[k];
        }
        atb[i] = sum;
    }

    // Solve A^T A x = A^T b via Gaussian elimination
    solve_gauss(&ata, &atb)
}

/// Solve linear system Ax = b via Gaussian elimination with partial pivoting
fn solve_gauss(a_orig: &[Vec<f64>], b_orig: &[f64]) -> Vec<f64> {
    let n = a_orig.len();
    let mut a = a_orig.to_vec();
    let mut b = b_orig.to_vec();

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = a[k][k].abs();
        for i in (k + 1)..n {
            let val = a[i][k].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Swap rows
        if max_idx != k {
            a.swap(k, max_idx);
            b.swap(k, max_idx);
        }

        // Check for singular matrix
        if a[k][k].abs() < 1e-12 {
            // Matrix is singular or nearly singular
            // Return best-effort solution (unchanged b)
            return b;
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = a[i][k] / a[k][k];
            b[i] -= factor * b[k];
            for j in k..n {
                a[i][j] -= factor * a[k][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::profiles;

    #[test]
    fn test_prepare_basis_state() {
        // Test |0⟩
        let state = prepare_basis_state(0, 2);
        assert_eq!(state.n_qubits, 2);

        // Test |3⟩ = |11⟩
        let state = prepare_basis_state(3, 2);
        assert_eq!(state.n_qubits, 2);
    }

    #[test]
    fn test_calibration_matrix_identity_for_perfect_hardware() {
        let mut profile = profiles::ibm_falcon();
        profile.measure_fidelity = 1.0; // Perfect measurements

        let calibration = ReadoutCalibration::build(&profile, 2, 1000, 42);

        // With perfect measurements, calibration matrix should be ~identity
        assert!(
            calibration.calibration_matrix.is_identity(0.1),
            "Calibration matrix should be identity for perfect hardware"
        );
    }

    #[test]
    fn test_calibration_matrix_noisy_hardware() {
        let mut profile = profiles::ibm_falcon();
        profile.measure_fidelity = 0.95; // 5% error rate

        let calibration = ReadoutCalibration::build(&profile, 2, 5000, 42);

        // Should NOT be identity
        assert!(
            !calibration.calibration_matrix.is_identity(0.01),
            "Calibration matrix should not be identity for noisy hardware"
        );

        // Diagonal elements should be larger than off-diagonal
        let dim = calibration.calibration_matrix.dimension();
        for i in 0..dim {
            let diagonal = calibration.calibration_matrix.matrix[i][i];
            assert!(
                diagonal > 0.8,
                "Diagonal entry {} should be high, got {}",
                i,
                diagonal
            );
        }
    }

    #[test]
    fn test_correct_counts_no_errors() {
        let mut profile = profiles::ibm_falcon();
        profile.measure_fidelity = 1.0;

        let calibration = ReadoutCalibration::build(&profile, 2, 1000, 42);

        // Perfect measurements: correction should be identity operation
        let mut raw_counts = HashMap::new();
        raw_counts.insert(0, 500);
        raw_counts.insert(3, 500);

        let corrected = calibration.correct_counts(&raw_counts);

        // Should be very close to input
        assert!((corrected[&0] - 500.0).abs() < 50.0);
        assert!((corrected[&3] - 500.0).abs() < 50.0);
    }

    #[test]
    fn test_correct_counts_with_errors() {
        let mut profile = profiles::ibm_falcon();
        profile.measure_fidelity = 0.9; // 10% error rate

        let calibration = ReadoutCalibration::build(&profile, 2, 5000, 42);

        // Simulate measuring |00⟩ with bit-flip errors
        let mut raw_counts = HashMap::new();
        raw_counts.insert(0b00, 810); // Mostly correct
        raw_counts.insert(0b01, 90); // Some single-bit flips
        raw_counts.insert(0b10, 90);
        raw_counts.insert(0b11, 10); // Rare double-bit flips

        let corrected = calibration.correct_counts(&raw_counts);

        // After correction, |00⟩ count should increase
        let corrected_00 = corrected.get(&0b00).copied().unwrap_or(0.0);
        assert!(
            corrected_00 > 810.0,
            "Correction should increase |00⟩ count from 810 to ~900"
        );
    }

    #[test]
    fn test_expected_improvement() {
        let mut profile = profiles::ibm_falcon();
        profile.measure_fidelity = 0.95;

        let calibration = ReadoutCalibration::build(&profile, 2, 1000, 42);
        let improvement = calibration.expected_improvement();

        // Should expect some improvement
        assert!(
            improvement > 1.0 && improvement < 3.0,
            "Expected improvement 1-3×, got {}",
            improvement
        );
    }
}
