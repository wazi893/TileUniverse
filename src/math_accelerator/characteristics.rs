//! Circuit characteristics data structures for mathematical analysis.
//!
//! These structures capture the mathematical properties of quantum circuits
//! that inform backend routing decisions.

use logic_fabric_core::algebraic_fusion::AlgebraicStats;

/// Mathematical characteristics of a quantum circuit.
///
/// This struct captures both quick metrics (always computed in O(n))
/// and optional algebraic stats (computed on demand for larger circuits).
#[derive(Debug, Clone)]
pub struct CircuitCharacteristics {
    /// Algebraic fusion statistics (when computed via `optimize_algebraic`).
    /// This is `None` for small circuits where analysis overhead > execution time.
    pub algebraic_stats: Option<AlgebraicStats>,

    /// Total number of gates in the circuit.
    pub gate_count: usize,

    /// Number of two-qubit gates (CNot, CZ, Swap, etc.).
    /// High counts indicate entanglement spreading.
    pub two_qubit_gate_count: usize,

    /// Number of distinct qubits that have Hadamard gates applied.
    /// This bounds the base sparsity: 2^h amplitudes.
    pub distinct_hadamard_qubits: usize,

    /// Predicted sparsity of the final state vector.
    pub predicted_sparsity: SparsityPrediction,

    /// Detected circuit pattern (if any).
    pub detected_pattern: Option<CircuitPattern>,
}

impl CircuitCharacteristics {
    /// Create characteristics for a small circuit (skip detailed analysis).
    pub fn for_small_circuit(gate_count: usize) -> Self {
        Self {
            algebraic_stats: None,
            gate_count,
            two_qubit_gate_count: 0,
            distinct_hadamard_qubits: 0,
            predicted_sparsity: SparsityPrediction::Dense,
            detected_pattern: None,
        }
    }

    /// Check if the circuit reduced to identity (no gates remaining).
    pub fn is_identity(&self) -> bool {
        self.algebraic_stats
            .as_ref()
            .map(|s| s.gates_remaining == 0)
            .unwrap_or(false)
    }

    /// Check if this circuit has high algebraic reduction (>80%).
    /// High reduction means CPU may be faster than GPU due to transfer overhead.
    pub fn has_high_reduction(&self) -> bool {
        self.algebraic_stats
            .as_ref()
            .map(|s| s.reduction_rate() > 0.8)
            .unwrap_or(false)
    }
}

/// Prediction of state vector sparsity after circuit execution.
///
/// Sparse predictions enable routing to hash-based backends that
/// avoid allocating full 2^n state vectors.
#[derive(Debug, Clone, PartialEq)]
pub enum SparsityPrediction {
    /// State is expected to be dense (many non-zero amplitudes).
    Dense,

    /// State is expected to be sparse with approximately `expected_nnz` non-zero entries.
    Sparse {
        /// Expected number of non-zero amplitudes.
        expected_nnz: usize,
    },

    /// GHZ state pattern detected: exactly 2 amplitudes (|00...0⟩ + |11...1⟩).
    GHZ,

    /// W state pattern detected: exactly n amplitudes (superposition of single |1⟩ states).
    W {
        /// Number of qubits (equals number of non-zero amplitudes).
        n_qubits: usize,
    },
}

impl SparsityPrediction {
    /// Check if this prediction indicates a sparse state.
    pub fn is_sparse(&self) -> bool {
        !matches!(self, SparsityPrediction::Dense)
    }

    /// Get the expected number of non-zero entries.
    pub fn expected_nnz(&self) -> Option<usize> {
        match self {
            SparsityPrediction::Dense => None,
            SparsityPrediction::Sparse { expected_nnz } => Some(*expected_nnz),
            SparsityPrediction::GHZ => Some(2),
            SparsityPrediction::W { n_qubits } => Some(*n_qubits),
        }
    }
}

/// Recognized circuit patterns that enable optimized execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitPattern {
    /// GHZ state preparation: H(0) followed by CNOT chain.
    /// Results in exactly 2 non-zero amplitudes regardless of qubit count.
    GHZPreparation,

    /// W state preparation: Creates superposition of single-excitation states.
    /// Results in exactly n non-zero amplitudes.
    WStatePreparation,

    /// Grover iteration pattern: Oracle + Diffusion operator.
    /// High algebraic reduction potential.
    GroverIteration,

    /// VQE ansatz: Parameterized rotation layers with entanglement.
    VQEAnsatz,

    /// Unknown or no specific pattern detected.
    Unknown,
}

/// Quick classification result from O(n) pre-analysis.
///
/// This determines whether full algebraic analysis is needed
/// or if a fast-path decision can be made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuickClassification {
    /// Small circuit (< 50 gates): skip analysis, just execute.
    /// Analysis overhead would exceed execution time.
    Small,

    /// GHZ preparation pattern detected: route to sparse backend.
    GHZ,

    /// W state preparation pattern detected: route to sparse backend.
    WState,

    /// Large circuit requiring full algebraic analysis.
    NeedsFullAnalysis,
}
