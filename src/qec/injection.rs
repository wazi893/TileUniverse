//! T-Gate Injection Models
//!
//! Implements two execution modes for T-gate injection:
//!
//! 1. **Abstract (Estimator)**: Fast resource estimation without simulation.
//!    Tracks error probability and resource cost but doesn't simulate
//!    measurement outcomes.
//!
//! 2. **Sampled (Pauli Frame)**: Simulates the classical control flow.
//!    Samples measurement outcomes and updates the Pauli frame accordingly.
//!    This is closer to how real FT execution works.
//!
//! # T-Gate Injection Physics
//!
//! A T-gate on a logical qubit is implemented via state injection:
//!
//! 1. Prepare magic state |T⟩ = (|0⟩ + e^(iπ/4)|1⟩)/√2
//! 2. Apply CNOT from data qubit to magic state
//! 3. Measure magic state in X basis
//! 4. Apply S correction if measurement is |−⟩
//!
//! The measurement outcome is random (50/50), and determines whether
//! an S correction is needed. This S correction is tracked in the Pauli frame.

use super::ft_types::{KnownBasis, LogicalErrorBudget, LogicalQubit, PauliFrame, UnknownReason};
use crate::qec::noise::SimpleRng;
use std::fmt;

// ============================================================================
// Injection Result
// ============================================================================

/// Result of a T-gate injection operation.
#[derive(Clone, Debug)]
pub struct InjectionResult {
    /// Pauli frame update from this injection
    pub frame_update: PauliFrame,
    /// Cycles consumed by this injection
    pub cycles: usize,
    /// Measurement outcome (for sampled mode)
    pub measurement_outcome: Option<bool>,
    /// Whether an error occurred during injection
    pub error_occurred: bool,
}

impl InjectionResult {
    /// Create result for abstract mode (no measurement sampling)
    pub fn abstract_result(cycles: usize) -> Self {
        Self {
            frame_update: PauliFrame::identity(),
            cycles,
            measurement_outcome: None,
            error_occurred: false,
        }
    }

    /// Create result for sampled mode
    pub fn sampled_result(measurement: bool, error: bool, cycles: usize) -> Self {
        // If measurement is true (|−⟩ outcome), we need S correction
        // S gate adds Z to the frame
        let frame_update = if measurement {
            PauliFrame::z()
        } else {
            PauliFrame::identity()
        };

        Self {
            frame_update,
            cycles,
            measurement_outcome: Some(measurement),
            error_occurred: error,
        }
    }
}

// ============================================================================
// Injection Model
// ============================================================================

/// T-gate injection execution model.
///
/// Determines how T-gates are handled during circuit execution.
#[derive(Clone, Debug)]
pub enum InjectionModel {
    /// Fast estimator: track error rate and cost only.
    ///
    /// This mode doesn't simulate individual gate outcomes.
    /// Use for resource estimation at scale.
    Abstract {
        /// Error rate per T-gate (from distillation)
        error_rate: f64,
        /// Cycle cost per T-gate
        cycle_cost: usize,
    },

    /// Sampled mode: simulate measurement outcomes.
    ///
    /// Actually samples the injection measurement and updates
    /// the Pauli frame. Use for verifying classical control logic.
    Sampled {
        /// Measurement error probability
        measurement_error: f64,
        /// Cycle cost per T-gate
        cycle_cost: usize,
    },

    /// Deterministic mode for testing.
    ///
    /// Uses fixed outcomes for reproducible tests.
    Deterministic {
        /// Fixed measurement outcomes (cycles through this list)
        outcomes: Vec<bool>,
        /// Current index in outcome list
        index: usize,
        /// Cycle cost per T-gate
        cycle_cost: usize,
    },
}

impl InjectionModel {
    /// Create abstract model with specified error rate.
    pub fn abstract_mode(error_rate: f64, cycle_cost: usize) -> Self {
        Self::Abstract {
            error_rate,
            cycle_cost,
        }
    }

    /// Create abstract model with typical surface code parameters.
    pub fn abstract_default() -> Self {
        Self::Abstract {
            error_rate: 1e-10,
            cycle_cost: 5,
        }
    }

    /// Create sampled model with specified measurement error.
    pub fn sampled_mode(measurement_error: f64, cycle_cost: usize) -> Self {
        Self::Sampled {
            measurement_error,
            cycle_cost,
        }
    }

    /// Create sampled model with typical parameters.
    pub fn sampled_default() -> Self {
        Self::Sampled {
            measurement_error: 1e-10,
            cycle_cost: 5,
        }
    }

    /// Create deterministic model for testing.
    pub fn deterministic(outcomes: Vec<bool>, cycle_cost: usize) -> Self {
        Self::Deterministic {
            outcomes,
            index: 0,
            cycle_cost,
        }
    }

    /// Get the error rate for this model.
    pub fn error_rate(&self) -> f64 {
        match self {
            Self::Abstract { error_rate, .. } => *error_rate,
            Self::Sampled {
                measurement_error, ..
            } => *measurement_error,
            Self::Deterministic { .. } => 0.0,
        }
    }

    /// Get the cycle cost for this model.
    pub fn cycle_cost(&self) -> usize {
        match self {
            Self::Abstract { cycle_cost, .. } => *cycle_cost,
            Self::Sampled { cycle_cost, .. } => *cycle_cost,
            Self::Deterministic { cycle_cost, .. } => *cycle_cost,
        }
    }

    /// Perform T-gate injection on a logical qubit.
    ///
    /// Updates the qubit's Pauli frame and the error budget.
    /// Returns the injection result.
    pub fn inject(
        &mut self,
        qubit: &mut LogicalQubit,
        rng: &mut SimpleRng,
        budget: &mut LogicalErrorBudget,
    ) -> InjectionResult {
        match self {
            Self::Abstract {
                error_rate,
                cycle_cost,
            } => {
                // Just track error accumulation
                budget.add_injection(*error_rate);
                InjectionResult::abstract_result(*cycle_cost)
            }

            Self::Sampled {
                measurement_error,
                cycle_cost,
            } => {
                // Sample measurement outcome (50/50 for |+⟩/|−⟩)
                let measurement = rng.next_f64() < 0.5;

                // Sample whether measurement error occurred
                let error = rng.next_f64() < *measurement_error;
                let actual_outcome = measurement ^ error;

                // Update qubit's Pauli frame based on outcome
                // |−⟩ outcome requires S correction (Z in frame)
                if actual_outcome {
                    qubit.frame.compose(&PauliFrame::z());
                }

                // T-gate basis tracking:
                // T = diag(1, e^{iπ/4}) is diagonal, so:
                // - Z-basis states are PRESERVED (T|0⟩=|0⟩, T|1⟩=e^{iπ/4}|1⟩)
                // - X-basis states become superpositions (unknown)
                // Only mark unknown if NOT already in Z-basis
                match qubit.basis {
                    KnownBasis::Z(_) => {
                        // Z-basis preserved - T only adds phase
                    }
                    KnownBasis::X(_) => {
                        // X-basis (superposition) -> T creates non-stabilizer state
                        qubit.mark_unknown_because(UnknownReason::NonCliffordOnSuperposition);
                    }
                    KnownBasis::Unknown(_) => {
                        // Already unknown -> stays unknown
                        qubit.mark_unknown_because(UnknownReason::NonCliffordOnUnknown);
                    }
                }

                // Track error in budget
                budget.add_injection(*measurement_error);

                InjectionResult::sampled_result(actual_outcome, error, *cycle_cost)
            }

            Self::Deterministic {
                outcomes,
                index,
                cycle_cost,
            } => {
                // Use predetermined outcome
                let measurement = if outcomes.is_empty() {
                    false
                } else {
                    let outcome = outcomes[*index % outcomes.len()];
                    *index += 1;
                    outcome
                };

                if measurement {
                    qubit.frame.compose(&PauliFrame::z());
                }

                // T-gate preserves Z-basis states (same logic as Sampled)
                match qubit.basis {
                    KnownBasis::Z(_) => {
                        // Z-basis preserved
                    }
                    KnownBasis::X(_) => {
                        qubit.mark_unknown_because(UnknownReason::NonCliffordOnSuperposition);
                    }
                    KnownBasis::Unknown(_) => {
                        qubit.mark_unknown_because(UnknownReason::NonCliffordOnUnknown);
                    }
                }

                InjectionResult::sampled_result(measurement, false, *cycle_cost)
            }
        }
    }

    /// Check if this is abstract mode.
    pub fn is_abstract(&self) -> bool {
        matches!(self, Self::Abstract { .. })
    }

    /// Check if this is sampled mode.
    pub fn is_sampled(&self) -> bool {
        matches!(self, Self::Sampled { .. })
    }
}

impl fmt::Display for InjectionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Abstract {
                error_rate,
                cycle_cost,
            } => {
                write!(f, "Abstract (p={:.2e}, {} cycles)", error_rate, cycle_cost)
            }
            Self::Sampled {
                measurement_error,
                cycle_cost,
            } => {
                write!(
                    f,
                    "Sampled (p_meas={:.2e}, {} cycles)",
                    measurement_error, cycle_cost
                )
            }
            Self::Deterministic {
                outcomes,
                cycle_cost,
                ..
            } => {
                write!(
                    f,
                    "Deterministic ({} outcomes, {} cycles)",
                    outcomes.len(),
                    cycle_cost
                )
            }
        }
    }
}

// ============================================================================
// T-Dagger Injection
// ============================================================================

/// Perform T† (T-dagger) injection.
///
/// T† = T^(-1), which in terms of injection means we need
/// a different magic state or conjugate the correction.
pub fn inject_tdg(
    model: &mut InjectionModel,
    qubit: &mut LogicalQubit,
    rng: &mut SimpleRng,
    budget: &mut LogicalErrorBudget,
) -> InjectionResult {
    // T† injection is similar to T but with conjugated correction
    let result = model.inject(qubit, rng, budget);

    // T† correction: S† instead of S
    // S† adds -Z to frame (which is same as Z in mod-2 arithmetic)
    // So the frame update is the same

    result
}

// ============================================================================
// Toffoli Injection
// ============================================================================

/// Perform Toffoli gate via T-gate decomposition.
///
/// Standard Toffoli decomposes into 7 T-gates (or 4 with magic state injection).
/// This function handles the full decomposition.
pub fn inject_toffoli(
    model: &mut InjectionModel,
    control1: &mut LogicalQubit,
    control2: &mut LogicalQubit,
    target: &mut LogicalQubit,
    rng: &mut SimpleRng,
    budget: &mut LogicalErrorBudget,
) -> Vec<InjectionResult> {
    // Toffoli = 7 T gates in standard decomposition
    // We apply them to different qubits in the decomposition
    // For simplicity, we just track 7 T-gate injections

    let mut results = Vec::with_capacity(7);

    // The actual decomposition pattern matters for frame tracking
    // but for resource estimation, we just need 7 T-gates
    // A more accurate simulation would track which qubit each T is on

    // Pattern: T on each qubit, then interactions
    results.push(model.inject(target, rng, budget));
    results.push(model.inject(control1, rng, budget));
    results.push(model.inject(control2, rng, budget));
    results.push(model.inject(target, rng, budget));
    results.push(model.inject(control1, rng, budget));
    results.push(model.inject(control2, rng, budget));
    results.push(model.inject(target, rng, budget));

    results
}

// ============================================================================
// Batch Injection
// ============================================================================

/// Batch injection for parallel T-gates.
///
/// When multiple T-gates are on different qubits in the same layer,
/// they can be executed in parallel with shared cycle cost.
pub fn inject_batch(
    model: &mut InjectionModel,
    qubits: &mut [&mut LogicalQubit],
    rng: &mut SimpleRng,
    budget: &mut LogicalErrorBudget,
) -> BatchInjectionResult {
    let mut results = Vec::with_capacity(qubits.len());

    for qubit in qubits {
        results.push(model.inject(*qubit, rng, budget));
    }

    // Parallel execution: max cycle cost, not sum
    let cycles = results.iter().map(|r| r.cycles).max().unwrap_or(0);
    let errors = results.iter().filter(|r| r.error_occurred).count();

    BatchInjectionResult {
        individual: results,
        total_cycles: cycles, // Parallel, not sequential
        error_count: errors,
    }
}

/// Result of batch T-gate injection.
#[derive(Clone, Debug)]
pub struct BatchInjectionResult {
    /// Individual injection results
    pub individual: Vec<InjectionResult>,
    /// Total cycles (for parallel execution)
    pub total_cycles: usize,
    /// Number of errors that occurred
    pub error_count: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abstract_injection() {
        let mut model = InjectionModel::abstract_mode(1e-10, 5);
        let mut qubit = LogicalQubit::new(0);
        let mut rng = SimpleRng::new(42);
        let mut budget = LogicalErrorBudget::new();

        let result = model.inject(&mut qubit, &mut rng, &mut budget);

        assert_eq!(result.cycles, 5);
        assert!(result.measurement_outcome.is_none());
        assert!(budget.from_injection > 0.0);
    }

    #[test]
    fn test_sampled_injection() {
        let mut model = InjectionModel::sampled_mode(0.0, 5);
        let mut qubit = LogicalQubit::new(0); // Starts in Z(false)
        let mut rng = SimpleRng::new(42);
        let mut budget = LogicalErrorBudget::new();

        let result = model.inject(&mut qubit, &mut rng, &mut budget);

        assert_eq!(result.cycles, 5);
        assert!(result.measurement_outcome.is_some());
        // T-gate preserves Z-basis states (diagonal gate)
        assert!(
            qubit.basis.is_known(),
            "Z-basis should be preserved after T"
        );
    }

    #[test]
    fn test_sampled_injection_x_basis_becomes_unknown() {
        let mut model = InjectionModel::sampled_mode(0.0, 5);
        let mut qubit = LogicalQubit::new(0);
        qubit.basis = KnownBasis::X(false); // |+⟩ state
        let mut rng = SimpleRng::new(42);
        let mut budget = LogicalErrorBudget::new();

        let result = model.inject(&mut qubit, &mut rng, &mut budget);

        assert_eq!(result.cycles, 5);
        assert!(result.measurement_outcome.is_some());
        // T-gate on X-basis creates superposition -> unknown
        assert!(
            !qubit.basis.is_known(),
            "X-basis should become unknown after T"
        );
    }

    #[test]
    fn test_deterministic_injection() {
        let mut model = InjectionModel::deterministic(vec![true, false, true], 5);
        let mut qubit = LogicalQubit::new(0);
        let mut rng = SimpleRng::new(42);
        let mut budget = LogicalErrorBudget::new();

        // First injection: outcome = true → Z frame update
        let r1 = model.inject(&mut qubit, &mut rng, &mut budget);
        assert_eq!(r1.measurement_outcome, Some(true));
        assert!(qubit.frame.z);

        // Reset qubit frame
        qubit.frame = PauliFrame::identity();

        // Second injection: outcome = false → no frame update
        let r2 = model.inject(&mut qubit, &mut rng, &mut budget);
        assert_eq!(r2.measurement_outcome, Some(false));
        assert!(!qubit.frame.z);
    }

    #[test]
    fn test_frame_accumulation() {
        let mut model = InjectionModel::deterministic(vec![true, true], 5);
        let mut qubit = LogicalQubit::new(0);
        let mut rng = SimpleRng::new(42);
        let mut budget = LogicalErrorBudget::new();

        // Two T gates with |−⟩ outcomes
        model.inject(&mut qubit, &mut rng, &mut budget);
        model.inject(&mut qubit, &mut rng, &mut budget);

        // Z + Z = I (mod 2)
        assert!(!qubit.frame.z); // Should cancel
    }

    #[test]
    fn test_batch_injection() {
        let mut model = InjectionModel::abstract_mode(1e-10, 5);
        let mut q0 = LogicalQubit::new(0);
        let mut q1 = LogicalQubit::new(1);
        let mut q2 = LogicalQubit::new(2);
        let mut rng = SimpleRng::new(42);
        let mut budget = LogicalErrorBudget::new();

        let result = inject_batch(
            &mut model,
            &mut [&mut q0, &mut q1, &mut q2],
            &mut rng,
            &mut budget,
        );

        assert_eq!(result.individual.len(), 3);
        assert_eq!(result.total_cycles, 5); // Parallel, not 15
        assert_eq!(budget.t_gates_processed, 3);
    }

    #[test]
    fn test_toffoli_injection() {
        let mut model = InjectionModel::abstract_mode(1e-10, 5);
        let mut c1 = LogicalQubit::new(0);
        let mut c2 = LogicalQubit::new(1);
        let mut t = LogicalQubit::new(2);
        let mut rng = SimpleRng::new(42);
        let mut budget = LogicalErrorBudget::new();

        let results = inject_toffoli(&mut model, &mut c1, &mut c2, &mut t, &mut rng, &mut budget);

        assert_eq!(results.len(), 7);
        assert_eq!(budget.t_gates_processed, 7);
    }

    #[test]
    fn test_error_rate_access() {
        let abstract_model = InjectionModel::abstract_mode(1e-8, 5);
        assert_eq!(abstract_model.error_rate(), 1e-8);

        let sampled_model = InjectionModel::sampled_mode(1e-9, 5);
        assert_eq!(sampled_model.error_rate(), 1e-9);
    }
}
