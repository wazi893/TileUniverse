//! Sprint 67.0: Expectation Value Computation
//!
//! Efficient computation of Hamiltonian expectation values from measurement counts.
//!
//! # Example
//!
//! ```ignore
//! use engine::hybrid::{ExpectationEvaluator, PauliTerm};
//!
//! let terms = vec![
//!     PauliTerm::z(0.5, 0),      // 0.5 * Z_0
//!     PauliTerm::zz(-1.0, 0, 1), // -1.0 * Z_0 * Z_1
//! ];
//!
//! let evaluator = ExpectationEvaluator::new(terms);
//! let expectation = evaluator.evaluate(&counts);
//! ```

use std::collections::HashMap;

use super::job::{CostFunction, PauliTerm};

/// Evaluator for Hamiltonian expectation values.
#[derive(Clone, Debug)]
pub struct ExpectationEvaluator {
    /// Pauli terms in the Hamiltonian
    terms: Vec<PauliTerm>,

    /// Grouped terms for efficient evaluation
    groups: Vec<PauliGroup>,

    /// Grouping strategy used
    grouping: PauliGrouping,
}

/// A group of commuting Pauli terms.
#[derive(Clone, Debug)]
pub struct PauliGroup {
    /// Terms in this group
    pub terms: Vec<PauliTerm>,

    /// Measurement basis for this group (per qubit)
    pub basis: Vec<MeasurementBasisType>,
}

/// Type of measurement basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasurementBasisType {
    /// Computational (Z) basis
    Z,
    /// X basis
    X,
    /// Y basis
    Y,
    /// Identity (no measurement needed)
    I,
}

/// Grouping strategy for Pauli terms.
#[derive(Clone, Copy, Debug, Default)]
pub enum PauliGrouping {
    /// No grouping - evaluate each term separately
    #[default]
    None,

    /// Group terms that are qubit-wise commuting
    QubitWise,

    /// General commuting groups (more efficient but complex)
    FullCommuting,
}

impl ExpectationEvaluator {
    /// Create a new evaluator with the given Pauli terms.
    pub fn new(terms: Vec<PauliTerm>) -> Self {
        Self {
            terms,
            groups: Vec::new(),
            grouping: PauliGrouping::None,
        }
    }

    /// Create evaluator from a cost function.
    pub fn from_cost_function(cost_fn: &CostFunction) -> Self {
        match cost_fn {
            CostFunction::ExpectationValue(terms) => Self::new(terms.clone()),
            CostFunction::MaxCut { edges } => {
                // MaxCut Hamiltonian: H = -sum_{ij} w_ij (1 - Z_i Z_j) / 2
                //                      = -sum_{ij} w_ij/2 + sum_{ij} w_ij Z_i Z_j / 2
                let mut terms = Vec::new();

                // Constant term (identity)
                let const_coeff: f64 = -edges.iter().map(|(_, _, w)| w / 2.0).sum::<f64>();
                if const_coeff.abs() > 1e-10 {
                    terms.push(PauliTerm::identity(const_coeff));
                }

                // ZZ terms
                for &(i, j, w) in edges {
                    terms.push(PauliTerm::zz(w / 2.0, i, j));
                }

                Self::new(terms)
            }
            _ => Self::new(Vec::new()),
        }
    }

    /// Set grouping strategy.
    pub fn with_grouping(mut self, grouping: PauliGrouping) -> Self {
        self.grouping = grouping;
        if !matches!(grouping, PauliGrouping::None) {
            self.compute_groups();
        }
        self
    }

    /// Compute groups based on grouping strategy.
    fn compute_groups(&mut self) {
        match self.grouping {
            PauliGrouping::None => {
                // Each term in its own group
                self.groups = self
                    .terms
                    .iter()
                    .map(|t| PauliGroup {
                        terms: vec![t.clone()],
                        basis: Self::term_to_basis(t),
                    })
                    .collect();
            }
            PauliGrouping::QubitWise => {
                self.groups = Self::group_qubitwise(&self.terms);
            }
            PauliGrouping::FullCommuting => {
                // For now, use qubitwise as a simpler approximation
                self.groups = Self::group_qubitwise(&self.terms);
            }
        }
    }

    /// Convert a Pauli term to measurement basis per qubit.
    fn term_to_basis(term: &PauliTerm) -> Vec<MeasurementBasisType> {
        let max_qubit = term.paulis.iter().map(|(q, _)| *q).max().unwrap_or(0);
        let mut basis = vec![MeasurementBasisType::I; max_qubit + 1];

        for &(qubit, pauli) in &term.paulis {
            basis[qubit] = match pauli {
                'X' | 'x' => MeasurementBasisType::X,
                'Y' | 'y' => MeasurementBasisType::Y,
                'Z' | 'z' => MeasurementBasisType::Z,
                _ => MeasurementBasisType::I,
            };
        }

        basis
    }

    /// Group terms that are qubit-wise commuting.
    fn group_qubitwise(terms: &[PauliTerm]) -> Vec<PauliGroup> {
        let mut groups: Vec<PauliGroup> = Vec::new();

        for term in terms {
            let term_basis = Self::term_to_basis(term);

            // Find compatible group
            let mut found = false;
            for group in &mut groups {
                if Self::basis_compatible(&group.basis, &term_basis) {
                    group.terms.push(term.clone());
                    // Merge basis
                    for (i, &b) in term_basis.iter().enumerate() {
                        if i < group.basis.len() {
                            if group.basis[i] == MeasurementBasisType::I {
                                group.basis[i] = b;
                            }
                        }
                    }
                    // Extend if needed
                    if term_basis.len() > group.basis.len() {
                        group
                            .basis
                            .extend_from_slice(&term_basis[group.basis.len()..]);
                    }
                    found = true;
                    break;
                }
            }

            if !found {
                groups.push(PauliGroup {
                    terms: vec![term.clone()],
                    basis: term_basis,
                });
            }
        }

        groups
    }

    /// Check if two measurement bases are compatible.
    fn basis_compatible(b1: &[MeasurementBasisType], b2: &[MeasurementBasisType]) -> bool {
        let len = b1.len().min(b2.len());
        for i in 0..len {
            if b1[i] != MeasurementBasisType::I
                && b2[i] != MeasurementBasisType::I
                && b1[i] != b2[i]
            {
                return false;
            }
        }
        true
    }

    /// Evaluate expectation value from measurement counts.
    pub fn evaluate(&self, counts: &HashMap<String, usize>) -> f64 {
        Self::compute_pauli_expectation(counts, &self.terms)
    }

    /// Evaluate expectation value from grouped measurements.
    pub fn evaluate_grouped(&self, measurements: &[GroupedMeasurement]) -> f64 {
        let mut expectation = 0.0;

        for (group, measurement) in self.groups.iter().zip(measurements.iter()) {
            for term in &group.terms {
                expectation += Self::evaluate_term(&measurement.counts, term);
            }
        }

        expectation
    }

    /// Compute expectation value of Pauli Hamiltonian from counts.
    pub fn compute_pauli_expectation(counts: &HashMap<String, usize>, terms: &[PauliTerm]) -> f64 {
        let mut expectation = 0.0;

        for term in terms {
            expectation += Self::evaluate_term(counts, term);
        }

        expectation
    }

    /// Evaluate a single Pauli term.
    fn evaluate_term(counts: &HashMap<String, usize>, term: &PauliTerm) -> f64 {
        // Identity term
        if term.paulis.is_empty() {
            return term.coeff;
        }

        // Only Z terms can be evaluated from computational basis measurements
        // For X and Y, we would need different measurement bases
        let all_z = term.paulis.iter().all(|(_, p)| *p == 'Z' || *p == 'z');
        if !all_z {
            // TODO: Handle X and Y measurements
            return 0.0;
        }

        let total: usize = counts.values().sum();
        if total == 0 {
            return 0.0;
        }

        let mut term_expectation = 0.0;

        for (bitstring, &count) in counts {
            // Compute eigenvalue for this bitstring
            let mut eigenvalue = 1.0;

            for &(qubit, _pauli) in &term.paulis {
                // Get bit value (bitstring is in big-endian, qubit 0 is rightmost)
                let bit = bitstring
                    .chars()
                    .rev()
                    .nth(qubit)
                    .map(|c| c == '1')
                    .unwrap_or(false);

                // Z eigenvalue: +1 for |0⟩, -1 for |1⟩
                eigenvalue *= if bit { -1.0 } else { 1.0 };
            }

            term_expectation += eigenvalue * (count as f64 / total as f64);
        }

        term.coeff * term_expectation
    }

    /// Get number of groups.
    pub fn num_groups(&self) -> usize {
        if self.groups.is_empty() {
            self.terms.len()
        } else {
            self.groups.len()
        }
    }

    /// Get number of terms.
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Get groups (for measurement planning).
    pub fn groups(&self) -> &[PauliGroup] {
        &self.groups
    }
}

/// Measurement result for a group.
#[derive(Clone, Debug)]
pub struct GroupedMeasurement {
    /// Group index
    pub group_idx: usize,

    /// Measurement counts
    pub counts: HashMap<String, usize>,
}

/// Create H₂ molecule Hamiltonian at given bond length (in Bohr radii).
///
/// Simplified two-qubit Hamiltonian from Jordan-Wigner transformation.
pub fn h2_hamiltonian(bond_length: f64) -> Vec<PauliTerm> {
    // Coefficients from VQE literature (approximate, for bond_length ≈ 1.4 Bohr)
    // These are simplified - real implementation would compute from integrals

    // Scale factor based on bond length (very approximate)
    let scale = 1.0 / (1.0 + (bond_length - 1.4).abs() * 0.5);

    vec![
        // Identity (nuclear repulsion + constant)
        PauliTerm::identity(-0.8124 * scale),
        // Z terms
        PauliTerm::z(0.1712 * scale, 0),
        PauliTerm::z(0.1712 * scale, 1),
        // ZZ term
        PauliTerm::zz(0.1686 * scale, 0, 1),
        // XX term (requires X-basis measurement)
        PauliTerm::new(0.0454 * scale, vec![(0, 'X'), (1, 'X')]),
        // YY term (requires Y-basis measurement)
        PauliTerm::new(0.0454 * scale, vec![(0, 'Y'), (1, 'Y')]),
    ]
}

/// Exact ground state energy of H₂ at equilibrium (0.735 Å ≈ 1.4 Bohr).
pub const H2_GROUND_STATE_ENERGY: f64 = -1.137;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expectation_evaluator_new() {
        let terms = vec![PauliTerm::z(1.0, 0), PauliTerm::zz(-0.5, 0, 1)];
        let evaluator = ExpectationEvaluator::new(terms);

        assert_eq!(evaluator.num_terms(), 2);
    }

    #[test]
    fn test_identity_term() {
        let terms = vec![PauliTerm::identity(2.5)];
        let counts: HashMap<String, usize> = [("00".to_string(), 500), ("11".to_string(), 500)]
            .into_iter()
            .collect();

        let expectation = ExpectationEvaluator::compute_pauli_expectation(&counts, &terms);
        assert!((expectation - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_z_term_all_zeros() {
        let terms = vec![PauliTerm::z(1.0, 0)];
        let counts: HashMap<String, usize> = [("00".to_string(), 1000)].into_iter().collect();

        let expectation = ExpectationEvaluator::compute_pauli_expectation(&counts, &terms);
        // All |0⟩ on qubit 0 → eigenvalue +1
        assert!((expectation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_z_term_all_ones() {
        let terms = vec![PauliTerm::z(1.0, 0)];
        let counts: HashMap<String, usize> = [("11".to_string(), 1000)].into_iter().collect();

        let expectation = ExpectationEvaluator::compute_pauli_expectation(&counts, &terms);
        // All |1⟩ on qubit 0 → eigenvalue -1
        assert!((expectation - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_z_term_equal_superposition() {
        let terms = vec![PauliTerm::z(1.0, 0)];
        let counts: HashMap<String, usize> = [("00".to_string(), 500), ("01".to_string(), 500)]
            .into_iter()
            .collect();

        let expectation = ExpectationEvaluator::compute_pauli_expectation(&counts, &terms);
        // 50% |0⟩ (+1) and 50% |1⟩ (-1) → expectation ≈ 0
        assert!(expectation.abs() < 0.1);
    }

    #[test]
    fn test_zz_term() {
        let terms = vec![PauliTerm::zz(1.0, 0, 1)];

        // |00⟩: Z₀=+1, Z₁=+1 → ZZ=+1
        let counts: HashMap<String, usize> = [("00".to_string(), 1000)].into_iter().collect();
        let exp = ExpectationEvaluator::compute_pauli_expectation(&counts, &terms);
        assert!((exp - 1.0).abs() < 1e-10);

        // |01⟩: Z₀=-1, Z₁=+1 → ZZ=-1
        let counts: HashMap<String, usize> = [("01".to_string(), 1000)].into_iter().collect();
        let exp = ExpectationEvaluator::compute_pauli_expectation(&counts, &terms);
        assert!((exp - (-1.0)).abs() < 1e-10);

        // |11⟩: Z₀=-1, Z₁=-1 → ZZ=+1
        let counts: HashMap<String, usize> = [("11".to_string(), 1000)].into_iter().collect();
        let exp = ExpectationEvaluator::compute_pauli_expectation(&counts, &terms);
        assert!((exp - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_basis_compatible() {
        use MeasurementBasisType::*;

        // Same basis - compatible
        assert!(ExpectationEvaluator::basis_compatible(&[Z, Z], &[Z, Z]));

        // Different basis on same qubit - not compatible
        assert!(!ExpectationEvaluator::basis_compatible(&[Z, Z], &[X, Z]));

        // Identity is compatible with anything
        assert!(ExpectationEvaluator::basis_compatible(&[I, Z], &[X, Z]));
        assert!(ExpectationEvaluator::basis_compatible(&[Z, I], &[Z, Y]));
    }

    #[test]
    fn test_qubitwise_grouping() {
        let terms = vec![
            PauliTerm::z(1.0, 0),
            PauliTerm::z(1.0, 1),
            PauliTerm::zz(1.0, 0, 1),
        ];

        let groups = ExpectationEvaluator::group_qubitwise(&terms);

        // All Z terms should be in one group
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].terms.len(), 3);
    }

    #[test]
    fn test_h2_hamiltonian() {
        let terms = h2_hamiltonian(1.4);

        // Should have 6 terms
        assert_eq!(terms.len(), 6);

        // Check for identity term
        assert!(terms.iter().any(|t| t.paulis.is_empty()));

        // Check for ZZ term
        assert!(
            terms
                .iter()
                .any(|t| { t.paulis.len() == 2 && t.paulis.iter().all(|(_, p)| *p == 'Z') })
        );
    }

    #[test]
    fn test_maxcut_to_hamiltonian() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];
        let cost_fn = CostFunction::MaxCut { edges };

        let evaluator = ExpectationEvaluator::from_cost_function(&cost_fn);

        // Should have identity + 2 ZZ terms
        assert!(evaluator.num_terms() >= 2);
    }
}
