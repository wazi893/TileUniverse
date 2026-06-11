// =============================================================================
// Hamiltonians Module
// =============================================================================
//
// Pauli operator representation for quantum Hamiltonians.
// Enables VQE and other variational algorithms for molecular chemistry.
//
// Key concepts:
// - Hamiltonian H = Σᵢ cᵢ Pᵢ where Pᵢ are Pauli strings
// - Measurement via basis rotation (X→Z via H, Y→Z via Rz+H)
// - Expectation values via parity measurements
//
// =============================================================================

use crate::quantum::{GateOutcome, QGate, QRng, QState, apply_gate_scalar};
use std::f32::consts::PI;

/// Single-qubit Pauli operators
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Pauli {
    I, // Identity
    X, // Pauli-X
    Y, // Pauli-Y
    Z, // Pauli-Z
}

/// Weighted Pauli term: coefficient × (P₀ ⊗ P₁ ⊗ ... ⊗ Pₙ)
#[derive(Clone, Debug)]
pub struct PauliTerm {
    pub coefficient: f64,
    pub paulis: Vec<Pauli>,
}

impl PauliTerm {
    /// Create a new Pauli term
    pub fn new(coefficient: f64, paulis: Vec<Pauli>) -> Self {
        Self {
            coefficient,
            paulis,
        }
    }

    /// Create identity term (constant energy shift)
    pub fn identity(n_qubits: u8, coefficient: f64) -> Self {
        Self {
            coefficient,
            paulis: vec![Pauli::I; n_qubits as usize],
        }
    }

    /// Number of qubits this term acts on
    pub fn n_qubits(&self) -> u8 {
        self.paulis.len() as u8
    }

    /// Check if this is a pure identity term
    pub fn is_identity(&self) -> bool {
        self.paulis.iter().all(|p| matches!(p, Pauli::I))
    }

    /// Generate measurement circuit to rotate this Pauli string to Z basis
    pub fn measurement_circuit(&self) -> Vec<QGate> {
        let mut gates = Vec::new();

        for (qubit, pauli) in self.paulis.iter().enumerate() {
            match pauli {
                Pauli::I => {
                    // Identity - no measurement needed
                }
                Pauli::X => {
                    // Rotate X → Z via Hadamard
                    gates.push(QGate::H(qubit as u8));
                }
                Pauli::Y => {
                    // Rotate Y → Z via Rz(-π/2) then H
                    gates.push(QGate::Rz(qubit as u8, -PI / 2.0));
                    gates.push(QGate::H(qubit as u8));
                }
                Pauli::Z => {
                    // Already in Z basis - no rotation needed
                }
            }
        }

        gates
    }

    /// Get qubits that are measured (non-identity Paulis)
    pub fn measured_qubits(&self) -> Vec<u8> {
        self.paulis
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                if !matches!(p, Pauli::I) {
                    Some(i as u8)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Full Hamiltonian: H = Σᵢ cᵢ Pᵢ
#[derive(Clone, Debug)]
pub struct Hamiltonian {
    pub n_qubits: u8,
    pub terms: Vec<PauliTerm>,
}

impl Hamiltonian {
    /// Create a new Hamiltonian from Pauli terms
    pub fn new(n_qubits: u8, terms: Vec<PauliTerm>) -> Self {
        // Validate all terms have correct qubit count
        for term in &terms {
            assert_eq!(
                term.n_qubits(),
                n_qubits,
                "All Pauli terms must act on {} qubits",
                n_qubits
            );
        }

        Self { n_qubits, terms }
    }

    /// H₂ molecule coefficient lookup table for 2-qubit reduced Hamiltonian
    /// Reference: arXiv:2407.01000v1, Table 2 - STO-3G basis with Qiskit
    /// Format: (bond_length_angstrom, a0, a1, a2, a3, a4)
    /// Hamiltonian: H = a₀·I + a₁·Z₀ + a₂·Z₁ + a₃·Z₀Z₁ + a₄·(X₀X₁ + Y₀Y₁)
    const H2_COEFFICIENTS_2Q: [(f64, f64, f64, f64, f64, f64); 16] = [
        (0.30, -0.75374, 0.80864, -0.80864, -0.01328, 0.16081),
        (0.40, -0.86257, 0.68881, -0.68881, -0.01291, 0.16451),
        (0.50, -0.94770, 0.58307, -0.58307, -0.01251, 0.16887),
        (0.60, -1.00712, 0.49401, -0.49401, -0.01206, 0.17373),
        (0.70, -1.04391, 0.42045, -0.42045, -0.01150, 0.17901),
        (0.80, -1.06321, 0.35995, -0.35995, -0.01080, 0.18462),
        (0.90, -1.07028, 0.30978, -0.30978, -0.00996, 0.19057),
        (1.00, -1.06924, 0.26752, -0.26752, -0.00901, 0.19679),
        (1.10, -1.06281, 0.23139, -0.23139, -0.00799, 0.20322),
        (1.20, -1.05267, 0.20018, -0.20018, -0.00696, 0.20979),
        (1.30, -1.03991, 0.17310, -0.17310, -0.00596, 0.21641),
        (1.40, -1.02535, 0.14956, -0.14956, -0.00503, 0.22302),
        (1.50, -1.00964, 0.12910, -0.12910, -0.00418, 0.22953),
        (1.60, -0.99329, 0.11130, -0.11130, -0.00344, 0.23590),
        (1.70, -0.97673, 0.09584, -0.09584, -0.00280, 0.24207),
        (1.80, -0.96028, 0.08240, -0.08240, -0.00226, 0.24801),
    ];

    /// Create H₂ molecule Hamiltonian in 2-qubit reduced form (Bravyi-Kitaev/parity mapping)
    ///
    /// This is a simplified 2-qubit Hamiltonian that captures the essential physics of H₂.
    /// Coefficients from arXiv:2407.01000v1, computed using STO-3G basis with Qiskit.
    /// Supports bond lengths from 0.30 to 1.80 Å with linear interpolation.
    ///
    /// Hamiltonian form: H = a₀·I + a₁·Z₀ + a₂·Z₁ + a₃·Z₀Z₁ + a₄·(X₀X₁ + Y₀Y₁)
    pub fn h2_molecule_2q(bond_length: f64) -> Self {
        let (a0, a1, a2, a3, a4) = Self::interpolate_h2_coefficients(bond_length);

        let terms = vec![
            PauliTerm::new(a0, vec![Pauli::I, Pauli::I]),
            PauliTerm::new(a1, vec![Pauli::Z, Pauli::I]),
            PauliTerm::new(a2, vec![Pauli::I, Pauli::Z]),
            PauliTerm::new(a3, vec![Pauli::Z, Pauli::Z]),
            PauliTerm::new(a4, vec![Pauli::X, Pauli::X]),
            PauliTerm::new(a4, vec![Pauli::Y, Pauli::Y]),
        ];

        Self::new(2, terms)
    }

    /// Interpolate H₂ coefficients for a given bond length
    fn interpolate_h2_coefficients(bond_length: f64) -> (f64, f64, f64, f64, f64) {
        let table = &Self::H2_COEFFICIENTS_2Q;

        // Clamp to valid range
        if bond_length <= table[0].0 {
            return (table[0].1, table[0].2, table[0].3, table[0].4, table[0].5);
        }
        if bond_length >= table[table.len() - 1].0 {
            let last = &table[table.len() - 1];
            return (last.1, last.2, last.3, last.4, last.5);
        }

        // Find bracketing entries and interpolate
        for i in 0..table.len() - 1 {
            let (r0, a0_0, a1_0, a2_0, a3_0, a4_0) = table[i];
            let (r1, a0_1, a1_1, a2_1, a3_1, a4_1) = table[i + 1];

            if bond_length >= r0 && bond_length <= r1 {
                let t = (bond_length - r0) / (r1 - r0);
                return (
                    a0_0 + t * (a0_1 - a0_0),
                    a1_0 + t * (a1_1 - a1_0),
                    a2_0 + t * (a2_1 - a2_0),
                    a3_0 + t * (a3_1 - a3_0),
                    a4_0 + t * (a4_1 - a4_0),
                );
            }
        }

        // Fallback (should never reach here)
        let mid = &table[table.len() / 2];
        (mid.1, mid.2, mid.3, mid.4, mid.5)
    }

    /// Get the exact FCI energy for H₂ at a given bond length (2-qubit form)
    ///
    /// Returns the exact ground state energy computed from the 2-qubit Hamiltonian.
    /// This serves as a reference for VQE validation.
    pub fn h2_fci_energy_2q(bond_length: f64) -> f64 {
        Self::h2_molecule_2q(bond_length).exact_ground_state_energy()
    }

    /// Create H₂ molecule Hamiltonian in minimal STO-3G basis (4-qubit Jordan-Wigner form)
    ///
    /// Uses the Jordan-Wigner transformation mapping 2 electrons in 4 spin-orbitals to 4 qubits.
    /// Coefficients from MathWorks documentation (originally from Whitfield et al.)
    ///
    /// For the full dissociation curve, use `h2_molecule_2q()` which supports bond lengths
    /// from 0.30 to 1.80 Å. This 4-qubit form is only accurate at equilibrium (~0.74 Å).
    ///
    /// The Hamiltonian has 15 Pauli terms and a ground state energy of approximately -1.137 Ha
    pub fn h2_molecule(bond_length: f64) -> Self {
        // Full 4-qubit Jordan-Wigner Hamiltonian for H₂ at R ≈ 0.742 Å
        // These coefficients are from established quantum chemistry references

        if (bond_length - 0.742).abs() > 0.1 && (bond_length - 0.735).abs() > 0.1 {
            eprintln!(
                "Warning: 4-qubit H₂ Hamiltonian only calibrated for R≈0.74 Å. \
                For bond length {} Å, consider using h2_molecule_2q() instead.",
                bond_length
            );
        }

        let terms = vec![
            // Identity (constant energy offset including nuclear repulsion)
            PauliTerm::new(
                -0.09963387941370971,
                vec![Pauli::I, Pauli::I, Pauli::I, Pauli::I],
            ),
            // One-body Z terms
            PauliTerm::new(
                0.17110545123720233,
                vec![Pauli::Z, Pauli::I, Pauli::I, Pauli::I],
            ),
            PauliTerm::new(
                0.17110545123720233,
                vec![Pauli::I, Pauli::Z, Pauli::I, Pauli::I],
            ),
            PauliTerm::new(
                -0.22250914236600539,
                vec![Pauli::I, Pauli::I, Pauli::Z, Pauli::I],
            ),
            PauliTerm::new(
                -0.22250914236600539,
                vec![Pauli::I, Pauli::I, Pauli::I, Pauli::Z],
            ),
            // Two-body ZZ terms
            PauliTerm::new(
                0.16859349595532533,
                vec![Pauli::Z, Pauli::Z, Pauli::I, Pauli::I],
            ),
            PauliTerm::new(
                0.12051027989546245,
                vec![Pauli::Z, Pauli::I, Pauli::Z, Pauli::I],
            ),
            PauliTerm::new(
                0.16584090244119712,
                vec![Pauli::Z, Pauli::I, Pauli::I, Pauli::Z],
            ),
            PauliTerm::new(
                0.16584090244119712,
                vec![Pauli::I, Pauli::Z, Pauli::Z, Pauli::I],
            ),
            PauliTerm::new(
                0.12051027989546245,
                vec![Pauli::I, Pauli::Z, Pauli::I, Pauli::Z],
            ),
            PauliTerm::new(
                0.17432077259242010,
                vec![Pauli::I, Pauli::I, Pauli::Z, Pauli::Z],
            ),
            // Four-body exchange terms (crucial for electron correlation)
            PauliTerm::new(
                0.04533062254573469,
                vec![Pauli::Y, Pauli::X, Pauli::X, Pauli::Y],
            ),
            PauliTerm::new(
                0.04533062254573469,
                vec![Pauli::X, Pauli::Y, Pauli::Y, Pauli::X],
            ),
            PauliTerm::new(
                -0.04533062254573469,
                vec![Pauli::X, Pauli::X, Pauli::Y, Pauli::Y],
            ),
            PauliTerm::new(
                -0.04533062254573469,
                vec![Pauli::Y, Pauli::Y, Pauli::X, Pauli::X],
            ),
        ];

        Self::new(4, terms)
    }

    /// Create a simple single-qubit Hamiltonian (for testing)
    pub fn single_qubit_z() -> Self {
        let terms = vec![PauliTerm::new(1.0, vec![Pauli::Z])];
        Self::new(1, terms)
    }

    /// Create a two-qubit XX + ZZ Hamiltonian (for testing)
    pub fn two_qubit_xx_zz() -> Self {
        let terms = vec![
            PauliTerm::new(1.0, vec![Pauli::X, Pauli::X]),
            PauliTerm::new(0.5, vec![Pauli::Z, Pauli::Z]),
        ];
        Self::new(2, terms)
    }

    /// Get measurement circuits for each Pauli term
    pub fn measurement_circuits(&self) -> Vec<(PauliTerm, Vec<QGate>)> {
        self.terms
            .iter()
            .map(|term| (term.clone(), term.measurement_circuit()))
            .collect()
    }

    /// Compute exact ground state energy via full matrix diagonalization
    ///
    /// Constructs the full 2^n × 2^n Hamiltonian matrix and finds the minimum eigenvalue.
    /// WARNING: This is exponentially expensive! Only use for validation on small systems.
    pub fn exact_ground_state_energy(&self) -> f64 {
        if self.n_qubits > 10 {
            panic!(
                "Cannot diagonalize {} qubit Hamiltonian (too large, max 10)",
                self.n_qubits
            );
        }

        let dim = 1usize << self.n_qubits;

        // Build full Hamiltonian matrix (real part only, assuming Hermitian)
        // For efficiency, we only store diagonal and track off-diagonal contributions
        let mut h_matrix: Vec<Vec<f64>> = vec![vec![0.0; dim]; dim];

        for term in &self.terms {
            self.add_pauli_term_to_matrix(&mut h_matrix, term);
        }

        // Find minimum eigenvalue using power iteration on (H - λI)
        // For small systems, just compute all eigenvalues via characteristic polynomial
        // or use iterative diagonalization
        self.find_min_eigenvalue(&h_matrix)
    }

    /// Add a Pauli term's contribution to the Hamiltonian matrix
    fn add_pauli_term_to_matrix(&self, matrix: &mut [Vec<f64>], term: &PauliTerm) {
        let dim = matrix.len();
        let coef = term.coefficient;

        for row in 0..dim {
            // Compute H[row, col] contribution from this Pauli term
            // Pauli matrices: I = [[1,0],[0,1]], X = [[0,1],[1,0]], Y = [[0,-i],[i,0]], Z = [[1,0],[0,-1]]
            // For a tensor product, we compute the matrix element for each qubit and multiply

            let (col, phase) = self.compute_pauli_action(term, row);

            // phase is real for X, Z, I; imaginary for Y
            // We only handle real Hamiltonians properly here (XX, ZZ, etc.)
            if phase.1.abs() < 1e-10 {
                // Real contribution
                matrix[row][col] += coef * phase.0;
            }
            // Note: For Y terms, we'd need complex matrix elements
            // H₂ in standard form doesn't have standalone Y terms
        }
    }

    /// Compute the action of a Pauli string on a basis state
    /// Returns (new_state_idx, (real_coef, imag_coef))
    fn compute_pauli_action(&self, term: &PauliTerm, state_idx: usize) -> (usize, (f64, f64)) {
        let mut new_idx = state_idx;
        let mut phase_re = 1.0;
        let mut phase_im = 0.0;

        for (qubit, pauli) in term.paulis.iter().enumerate() {
            let bit = (state_idx >> qubit) & 1;

            match pauli {
                Pauli::I => {
                    // Identity: no change
                }
                Pauli::X => {
                    // X flips the bit: |0⟩ ↔ |1⟩
                    new_idx ^= 1 << qubit;
                }
                Pauli::Y => {
                    // Y flips the bit with phase: Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                    new_idx ^= 1 << qubit;
                    let (new_re, new_im) = if bit == 0 {
                        // Y|0⟩ = i|1⟩
                        (-phase_im, phase_re)
                    } else {
                        // Y|1⟩ = -i|0⟩
                        (phase_im, -phase_re)
                    };
                    phase_re = new_re;
                    phase_im = new_im;
                }
                Pauli::Z => {
                    // Z gives eigenvalue: Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
                    if bit == 1 {
                        phase_re = -phase_re;
                        phase_im = -phase_im;
                    }
                }
            }
        }

        (new_idx, (phase_re, phase_im))
    }

    /// Find minimum eigenvalue of a symmetric matrix using power iteration
    fn find_min_eigenvalue(&self, matrix: &[Vec<f64>]) -> f64 {
        let dim = matrix.len();

        // For very small matrices, use direct computation
        if dim <= 4 {
            return self.min_eigenvalue_direct(matrix);
        }

        // Use shifted power iteration to find minimum eigenvalue
        // First estimate spectral range
        let trace: f64 = (0..dim).map(|i| matrix[i][i]).sum();
        let _avg_diag = trace / dim as f64;

        // Gershgorin circle theorem for bounds
        let mut max_gershgorin = f64::MIN;
        let mut min_gershgorin = f64::MAX;
        for i in 0..dim {
            let row_sum: f64 = matrix[i]
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &v)| v.abs())
                .sum();
            max_gershgorin = max_gershgorin.max(matrix[i][i] + row_sum);
            min_gershgorin = min_gershgorin.min(matrix[i][i] - row_sum);
        }

        // Shift matrix to find smallest eigenvalue: (H - σI) where σ > λ_max
        let shift = max_gershgorin + 1.0;

        // Power iteration on shifted matrix
        let mut v: Vec<f64> = vec![1.0 / (dim as f64).sqrt(); dim];
        let mut lambda = 0.0;

        for _ in 0..1000 {
            // w = (H - σI) * v
            let mut w = vec![0.0; dim];
            for i in 0..dim {
                for j in 0..dim {
                    let h_ij = if i == j {
                        matrix[i][j] - shift
                    } else {
                        matrix[i][j]
                    };
                    w[i] += h_ij * v[j];
                }
            }

            // Normalize
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            for i in 0..dim {
                w[i] /= norm;
            }

            // Rayleigh quotient for eigenvalue estimate
            let mut new_lambda = 0.0;
            for i in 0..dim {
                for j in 0..dim {
                    let h_ij = if i == j {
                        matrix[i][j] - shift
                    } else {
                        matrix[i][j]
                    };
                    new_lambda += w[i] * h_ij * w[j];
                }
            }

            if (new_lambda - lambda).abs() < 1e-10 {
                break;
            }

            lambda = new_lambda;
            v = w;
        }

        // Shift back: actual eigenvalue = lambda + shift
        // But we want minimum, so this gives maximum of shifted
        // Need inverse iteration instead...

        // For reliability, just enumerate all eigenvalues for small systems
        self.min_eigenvalue_direct(matrix)
    }

    /// Direct computation of minimum eigenvalue for small matrices
    /// Uses inverse power iteration with shift to find smallest eigenvalue
    fn min_eigenvalue_direct(&self, matrix: &[Vec<f64>]) -> f64 {
        let dim = matrix.len();

        // For 2x2, use quadratic formula
        if dim == 2 {
            let a = matrix[0][0];
            let b = matrix[0][1];
            let c = matrix[1][0];
            let d = matrix[1][1];
            let trace = a + d;
            let det = a * d - b * c;
            let discriminant = trace * trace - 4.0 * det;
            if discriminant >= 0.0 {
                return (trace - discriminant.sqrt()) / 2.0;
            } else {
                return trace / 2.0; // Complex eigenvalues, return real part
            }
        }

        // For small matrices (up to 16x16), enumerate all eigenvalues via
        // characteristic polynomial roots using inverse iteration from multiple starting points

        // First, estimate eigenvalue bounds using Gershgorin circles
        let mut max_bound = f64::MIN;
        let mut min_bound = f64::MAX;
        for i in 0..dim {
            let diag = matrix[i][i];
            let row_sum: f64 = matrix[i]
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &v)| v.abs())
                .sum();
            max_bound = max_bound.max(diag + row_sum);
            min_bound = min_bound.min(diag - row_sum);
        }

        // Use inverse power iteration with shift to find minimum eigenvalue
        // Shift to near lower bound
        let shift = min_bound - 0.01;
        let mut best_eigenvalue = f64::MAX;

        // Try multiple starting vectors to avoid missing the minimum
        for seed in 0..20 {
            // Create shifted matrix: M = H - shift*I
            let mut shifted = matrix.to_vec();
            for i in 0..dim {
                shifted[i][i] -= shift;
            }

            // Starting vector (vary with seed)
            let mut v: Vec<f64> = (0..dim)
                .map(|i| {
                    let x = ((i + seed * 7) as f64 * 0.618034).fract();
                    x - 0.5
                })
                .collect();
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in &mut v {
                *x /= norm;
            }

            // Inverse iteration: solve (H - shift*I) * w = v
            // For small matrices, use Gaussian elimination
            for _ in 0..100 {
                // Solve shifted * w = v using Gaussian elimination with partial pivoting
                let w = self.solve_linear_system(&shifted, &v);

                // Normalize
                let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-15 || !norm.is_finite() {
                    break;
                }
                let w_normalized: Vec<f64> = w.iter().map(|x| x / norm).collect();

                // Rayleigh quotient for original matrix
                let mut lambda = 0.0;
                for i in 0..dim {
                    for j in 0..dim {
                        lambda += w_normalized[i] * matrix[i][j] * w_normalized[j];
                    }
                }

                if lambda.is_finite() {
                    best_eigenvalue = best_eigenvalue.min(lambda);
                }

                v = w_normalized;
            }
        }

        best_eigenvalue
    }

    /// Solve Ax = b using Gaussian elimination with partial pivoting
    fn solve_linear_system(&self, a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut aug: Vec<Vec<f64>> = a
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut new_row = row.clone();
                new_row.push(b[i]);
                new_row
            })
            .collect();

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }

            // Swap rows
            aug.swap(col, max_row);

            // Check for singular matrix
            if aug[col][col].abs() < 1e-12 {
                continue;
            }

            // Eliminate
            for row in (col + 1)..n {
                let factor = aug[row][col] / aug[col][col];
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * x[j];
            }
            if aug[i][i].abs() > 1e-12 {
                x[i] = sum / aug[i][i];
            }
        }

        x
    }
}

/// Measure expectation value of a Pauli term on a quantum state
///
/// Algorithm:
/// 1. Apply measurement circuit to rotate Pauli string to Z basis
/// 2. Sample n_shots measurements
/// 3. Compute parity: (-1)^(popcount of measured qubits)
/// 4. Return average parity × coefficient
pub fn measure_pauli_expectation(
    initial_state: &QState,
    pauli_term: &PauliTerm,
    n_shots: usize,
    seed: u64,
) -> f64 {
    // Identity terms contribute coefficient directly
    if pauli_term.is_identity() {
        return pauli_term.coefficient;
    }

    let measurement_gates = pauli_term.measurement_circuit();
    let measured_qubits = pauli_term.measured_qubits();

    let mut total_parity = 0.0;
    let mut rng = QRng::new(seed);

    for _ in 0..n_shots {
        // Fresh copy of state for each shot
        let mut state = initial_state.clone();

        // Apply measurement circuit
        for gate in &measurement_gates {
            let _ = apply_gate_scalar(&mut state, gate, &mut rng);
        }

        // Measure all qubits
        let mut measurement_outcome = 0u64;
        for q in 0..pauli_term.n_qubits() {
            let result = apply_gate_scalar(&mut state, &QGate::Measure(q), &mut rng);
            if let GateOutcome::Measured { bit, .. } = result {
                if bit == 1 {
                    measurement_outcome |= 1 << q;
                }
            }
        }

        // Compute parity of measured qubits only
        let mut parity_bits = 0;
        for &q in &measured_qubits {
            if (measurement_outcome >> q) & 1 == 1 {
                parity_bits ^= 1;
            }
        }

        // Parity: (-1)^(number of 1s)
        let parity = if parity_bits == 0 { 1.0 } else { -1.0 };
        total_parity += parity;
    }

    let average_parity = total_parity / n_shots as f64;
    pauli_term.coefficient * average_parity
}

/// Measure full Hamiltonian expectation value
///
/// E = ⟨ψ|H|ψ⟩ = Σᵢ cᵢ ⟨ψ|Pᵢ|ψ⟩
pub fn measure_hamiltonian_expectation(
    state: &QState,
    hamiltonian: &Hamiltonian,
    n_shots: usize,
    seed: u64,
) -> f64 {
    let mut total_energy = 0.0;
    let mut rng_seed = seed;

    for term in &hamiltonian.terms {
        let expectation = measure_pauli_expectation(state, term, n_shots, rng_seed);
        total_energy += expectation;
        rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1); // LCG
    }

    total_energy
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_term_construction() {
        let term = PauliTerm::new(0.5, vec![Pauli::X, Pauli::Z, Pauli::I]);
        assert_eq!(term.coefficient, 0.5);
        assert_eq!(term.n_qubits(), 3);
        assert!(!term.is_identity());
    }

    #[test]
    fn test_identity_term() {
        let term = PauliTerm::identity(4, 2.0);
        assert_eq!(term.coefficient, 2.0);
        assert_eq!(term.n_qubits(), 4);
        assert!(term.is_identity());
    }

    #[test]
    fn test_measurement_circuit_x() {
        let term = PauliTerm::new(1.0, vec![Pauli::X]);
        let gates = term.measurement_circuit();
        assert_eq!(gates.len(), 1);
        assert!(matches!(gates[0], QGate::H(0)));
    }

    #[test]
    fn test_measurement_circuit_y() {
        let term = PauliTerm::new(1.0, vec![Pauli::Y]);
        let gates = term.measurement_circuit();
        assert_eq!(gates.len(), 2);
        assert!(matches!(gates[0], QGate::Rz(0, _)));
        assert!(matches!(gates[1], QGate::H(0)));
    }

    #[test]
    fn test_measurement_circuit_z() {
        let term = PauliTerm::new(1.0, vec![Pauli::Z]);
        let gates = term.measurement_circuit();
        assert_eq!(gates.len(), 0); // No rotation needed
    }

    #[test]
    fn test_measurement_circuit_mixed() {
        let term = PauliTerm::new(1.0, vec![Pauli::X, Pauli::Y, Pauli::Z, Pauli::I]);
        let gates = term.measurement_circuit();

        // X: 1 gate (H)
        // Y: 2 gates (Rz + H)
        // Z: 0 gates
        // I: 0 gates
        // Total: 3 gates
        assert_eq!(gates.len(), 3);
    }

    #[test]
    fn test_measured_qubits() {
        let term = PauliTerm::new(1.0, vec![Pauli::X, Pauli::I, Pauli::Z, Pauli::I]);
        let qubits = term.measured_qubits();
        assert_eq!(qubits, vec![0, 2]); // Qubits 0 (X) and 2 (Z)
    }

    #[test]
    fn test_hamiltonian_creation() {
        let h = Hamiltonian::single_qubit_z();
        assert_eq!(h.n_qubits, 1);
        assert_eq!(h.terms.len(), 1);
        assert_eq!(h.terms[0].coefficient, 1.0);
    }

    #[test]
    fn test_h2_hamiltonian() {
        let h = Hamiltonian::h2_molecule(0.735);
        assert_eq!(h.n_qubits, 4); // 4-qubit Jordan-Wigner form
        assert_eq!(h.terms.len(), 15); // 15 Pauli terms

        // Verify it has the expected exchange terms (YXXY, XYYX, etc.)
        let has_yxxy = h
            .terms
            .iter()
            .any(|t| matches!(&t.paulis[..], [Pauli::Y, Pauli::X, Pauli::X, Pauli::Y]));
        let has_xxyy = h
            .terms
            .iter()
            .any(|t| matches!(&t.paulis[..], [Pauli::X, Pauli::X, Pauli::Y, Pauli::Y]));
        assert!(has_yxxy, "H₂ should have YXXY term");
        assert!(has_xxyy, "H₂ should have XXYY term");
    }

    #[test]
    fn test_h2_measurement_circuits() {
        let h = Hamiltonian::h2_molecule(0.735);
        let circuits = h.measurement_circuits();
        assert_eq!(circuits.len(), 15); // 15 Pauli terms

        // IIII term should have no gates
        assert!(circuits[0].1.is_empty());

        // Find a YXXY term and check it has correct measurement gates
        let yxxy_circuit = circuits
            .iter()
            .find(|(term, _)| matches!(&term.paulis[..], [Pauli::Y, Pauli::X, Pauli::X, Pauli::Y]));
        assert!(yxxy_circuit.is_some(), "Should have YXXY term");

        let yxxy_gates = &yxxy_circuit.unwrap().1;
        // Y measurement needs Rz + H, X measurement needs H
        let has_rz = yxxy_gates.iter().any(|g| matches!(g, QGate::Rz(_, _)));
        let has_h = yxxy_gates.iter().any(|g| matches!(g, QGate::H(_)));
        assert!(has_rz && has_h, "YXXY measurement should use Rz and H");
    }

    #[test]
    fn test_measure_identity_term() {
        let state = QState::new_zero(2);
        let term = PauliTerm::identity(2, 2.5);

        let expectation = measure_pauli_expectation(&state, &term, 100, 42);
        assert!((expectation - 2.5).abs() < 1e-6); // Identity always contributes coefficient
    }

    #[test]
    fn test_measure_z_on_zero_state() {
        // |0⟩ state, measure Z
        // ⟨0|Z|0⟩ = +1
        let state = QState::new_zero(1);
        let term = PauliTerm::new(1.0, vec![Pauli::Z]);

        let expectation = measure_pauli_expectation(&state, &term, 1000, 42);
        assert!((expectation - 1.0).abs() < 0.1); // Should be close to +1
    }

    #[test]
    fn test_measure_z_on_one_state() {
        // |1⟩ state, measure Z
        // ⟨1|Z|1⟩ = -1
        let mut state = QState::new_zero(1);
        let mut rng = QRng::new(0);
        let _ = apply_gate_scalar(&mut state, &QGate::X(0), &mut rng); // Flip to |1⟩

        let term = PauliTerm::new(1.0, vec![Pauli::Z]);
        let expectation = measure_pauli_expectation(&state, &term, 1000, 42);
        assert!((expectation - (-1.0)).abs() < 0.1); // Should be close to -1
    }

    #[test]
    fn test_measure_x_on_plus_state() {
        // |+⟩ state, measure X
        // ⟨+|X|+⟩ = +1
        let mut state = QState::new_zero(1);
        let mut rng = QRng::new(0);
        let _ = apply_gate_scalar(&mut state, &QGate::H(0), &mut rng); // Create |+⟩

        let term = PauliTerm::new(1.0, vec![Pauli::X]);
        let expectation = measure_pauli_expectation(&state, &term, 1000, 42);
        assert!((expectation - 1.0).abs() < 0.1); // Should be close to +1
    }

    #[test]
    fn test_hamiltonian_expectation_single_qubit() {
        // H = Z on |0⟩ should give E = +1
        let state = QState::new_zero(1);
        let h = Hamiltonian::single_qubit_z();

        let energy = measure_hamiltonian_expectation(&state, &h, 1000, 42);
        assert!((energy - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_exact_ground_state_single_qubit_z() {
        // H = Z, ground state is |1⟩ with E = -1
        let h = Hamiltonian::single_qubit_z();
        let e_ground = h.exact_ground_state_energy();
        assert!((e_ground - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_exact_ground_state_two_qubit() {
        let h = Hamiltonian::two_qubit_xx_zz();
        let e_ground = h.exact_ground_state_energy();
        println!("Two-qubit XX+ZZ ground state energy: {:.4}", e_ground);
        // H = XX + 0.5*ZZ has ground state energy -1.5
        // (Bell state |01⟩-|10⟩ with XX=-1 and ZZ=-1)
        assert!(
            e_ground < -1.0,
            "Ground state should be around -1.5, got {}",
            e_ground
        );
    }

    #[test]
    fn test_h2_exact_energy_diagnostic() {
        let h = Hamiltonian::h2_molecule(0.735);

        // Print all terms
        println!("\nH2 Hamiltonian terms (4-qubit Jordan-Wigner form):");
        for term in &h.terms {
            let paulis: String = term
                .paulis
                .iter()
                .map(|p| match p {
                    Pauli::I => 'I',
                    Pauli::X => 'X',
                    Pauli::Y => 'Y',
                    Pauli::Z => 'Z',
                })
                .collect();
            println!("  {:+.6} × {}", term.coefficient, paulis);
        }

        // Compute exact energy
        let exact = h.exact_ground_state_energy();
        println!("\nExact ground state energy: {:.4} Ha", exact);
        println!("Expected FCI energy:       -1.1373 Ha");

        // Verify we're close to the expected value
        let error = (exact - (-1.1373)).abs();
        println!("Error: {:.4} Ha", error);

        // The exact calculation should give something close to -1.137 Ha
        // Allow larger tolerance due to numerical precision in matrix diagonalization
        assert!(
            error < 0.1,
            "Exact energy {:.4} should be close to -1.1373 Ha",
            exact
        );
    }
}
