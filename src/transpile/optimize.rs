//! SPRINT 54.0: Pre-Transpile Optimization Integration
//!
//! This module connects the logic-fabric-core optimization passes to the
//! hardware transpilation pipeline. It provides a unified optimization
//! interface that applies algebraic simplifications before hardware mapping.
//!
//! ## Optimization Pipeline
//!
//! ```text
//! Input Circuit
//!     │
//!     ▼
//! ┌─────────────────────────────┐
//! │ 1. Adjacent Cancellation    │  H·H=I, X·X=I, CNOT·CNOT=I
//! └─────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────┐
//! │ 2. Rotation Composition     │  Rz(θ₁)·Rz(θ₂) = Rz(θ₁+θ₂)
//! └─────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────┐
//! │ 3. Commutation Reordering   │  Move gates to enable cancellation
//! └─────────────────────────────┘
//!     │
//!     ▼
//! ┌─────────────────────────────┐
//! │ 4. Long-Range Cancellation  │  Cancel via commutation
//! └─────────────────────────────┘
//!     │
//!     ▼
//! Optimized Circuit
//! ```

use logic_fabric_core::algebraic_fusion::{
    cancel_inverse_fixpoint, fuse_general_single_qubit, optimize_rotation_chain,
};
use logic_fabric_core::commutation::{
    commutation_cancellation_fixpoint, commute_for_fusion_fixpoint,
    comprehensive_commutation_optimize,
};
use logic_fabric_core::quantum::QGate;

/// Optimization level for pre-transpile passes
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationLevel {
    /// No optimization - pass through as-is
    None = 0,
    /// Basic optimization - adjacent cancellation only
    Basic = 1,
    /// Standard optimization - cancellation + rotation composition
    #[default]
    Standard = 2,
    /// Aggressive optimization - full commutation analysis
    Aggressive = 3,
}

/// Configuration for pre-transpile optimization
#[derive(Clone, Debug)]
pub struct OptimizationConfig {
    /// Overall optimization level
    pub level: OptimizationLevel,
    /// Enable adjacent gate cancellation (H·H, X·X, etc.)
    pub enable_cancellation: bool,
    /// Enable rotation composition (Rz(θ₁)·Rz(θ₂) = Rz(θ₁+θ₂))
    pub enable_rotation_merge: bool,
    /// Enable commutation-based reordering
    pub enable_commutation: bool,
    /// Enable single-qubit gate fusion
    pub enable_single_qubit_fusion: bool,
    /// Maximum iterations for fixpoint algorithms
    pub max_iterations: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            level: OptimizationLevel::Standard,
            enable_cancellation: true,
            enable_rotation_merge: true,
            enable_commutation: true,
            enable_single_qubit_fusion: true,
            max_iterations: 100,
        }
    }
}

impl OptimizationConfig {
    /// Create config for no optimization
    pub fn none() -> Self {
        Self {
            level: OptimizationLevel::None,
            enable_cancellation: false,
            enable_rotation_merge: false,
            enable_commutation: false,
            enable_single_qubit_fusion: false,
            max_iterations: 0,
        }
    }

    /// Create config for basic optimization
    pub fn basic() -> Self {
        Self {
            level: OptimizationLevel::Basic,
            enable_cancellation: true,
            enable_rotation_merge: false,
            enable_commutation: false,
            enable_single_qubit_fusion: false,
            max_iterations: 10,
        }
    }

    /// Create config for standard optimization
    pub fn standard() -> Self {
        Self::default()
    }

    /// Create config for aggressive optimization
    pub fn aggressive() -> Self {
        Self {
            level: OptimizationLevel::Aggressive,
            enable_cancellation: true,
            enable_rotation_merge: true,
            enable_commutation: true,
            enable_single_qubit_fusion: true,
            max_iterations: 100,
        }
    }
}

/// Statistics from optimization pass
#[derive(Clone, Debug, Default)]
pub struct OptimizationStats {
    /// Original gate count
    pub original_gates: usize,
    /// Final gate count after optimization
    pub final_gates: usize,
    /// Gates eliminated by cancellation
    pub cancelled_gates: usize,
    /// Gates merged by rotation composition
    pub merged_rotations: usize,
    /// Gates eliminated by commutation
    pub commutation_eliminations: usize,
    /// Single-qubit fusion compressions
    pub single_qubit_fusions: usize,
    /// Number of optimization iterations
    pub iterations: usize,
}

impl OptimizationStats {
    /// Calculate gate reduction percentage
    pub fn reduction_percent(&self) -> f64 {
        if self.original_gates == 0 {
            return 0.0;
        }
        let reduced = self.original_gates.saturating_sub(self.final_gates);
        (reduced as f64 / self.original_gates as f64) * 100.0
    }

    /// Calculate overall speedup factor
    pub fn speedup_factor(&self) -> f64 {
        if self.final_gates == 0 {
            return f64::INFINITY;
        }
        self.original_gates as f64 / self.final_gates as f64
    }
}

/// Result of pre-transpile optimization
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    /// Optimized circuit
    pub circuit: Vec<QGate>,
    /// Optimization statistics
    pub stats: OptimizationStats,
}

/// Apply pre-transpile optimization to a circuit
///
/// This is the main entry point for optimization before hardware transpilation.
/// It applies the optimization passes according to the configuration.
pub fn optimize_circuit(circuit: &[QGate], config: &OptimizationConfig) -> OptimizationResult {
    let original_gates = circuit.len();

    if config.level == OptimizationLevel::None || circuit.is_empty() {
        return OptimizationResult {
            circuit: circuit.to_vec(),
            stats: OptimizationStats {
                original_gates,
                final_gates: original_gates,
                ..Default::default()
            },
        };
    }

    let mut current = circuit.to_vec();
    let mut stats = OptimizationStats {
        original_gates,
        ..Default::default()
    };

    // Track intermediate gate counts for statistics
    let mut prev_len = current.len();

    // Pass 1: Adjacent cancellation
    if config.enable_cancellation {
        current = cancel_inverse_fixpoint(current);
        let cancelled = prev_len.saturating_sub(current.len());
        stats.cancelled_gates += cancelled;
        prev_len = current.len();
    }

    // Pass 2: Rotation composition
    if config.enable_rotation_merge {
        current = optimize_rotation_chain(&current);
        let merged = prev_len.saturating_sub(current.len());
        stats.merged_rotations += merged;
        prev_len = current.len();
    }

    // Pass 3: Commutation-based optimization (for Standard and Aggressive)
    if config.enable_commutation && config.level >= OptimizationLevel::Standard {
        // First, reorder for fusion
        current = commute_for_fusion_fixpoint(current);

        // Then, long-range cancellation
        current = commutation_cancellation_fixpoint(current);
        let eliminated = prev_len.saturating_sub(current.len());
        stats.commutation_eliminations += eliminated;
        prev_len = current.len();
    }

    // Pass 4: Single-qubit gate fusion (for Aggressive)
    if config.enable_single_qubit_fusion && config.level == OptimizationLevel::Aggressive {
        current = fuse_general_single_qubit(&current);
        let fused = prev_len.saturating_sub(current.len());
        stats.single_qubit_fusions += fused;
        prev_len = current.len();
    }

    // Pass 5: Final cleanup - another round of cancellation
    if config.enable_cancellation {
        current = cancel_inverse_fixpoint(current);
        let cancelled = prev_len.saturating_sub(current.len());
        stats.cancelled_gates += cancelled;
    }

    // For Aggressive: run comprehensive optimization (fixpoint loop)
    if config.level == OptimizationLevel::Aggressive {
        let before_comprehensive = current.len();
        current = comprehensive_commutation_optimize(current);
        let comprehensive_reduction = before_comprehensive.saturating_sub(current.len());
        stats.commutation_eliminations += comprehensive_reduction;
    }

    stats.final_gates = current.len();
    stats.iterations = 1; // Simplified - actual iterations tracked internally

    OptimizationResult {
        circuit: current,
        stats,
    }
}

/// Quick optimization with default settings
pub fn optimize(circuit: &[QGate]) -> Vec<QGate> {
    optimize_circuit(circuit, &OptimizationConfig::default()).circuit
}

/// Aggressive optimization for maximum gate reduction
pub fn optimize_aggressive(circuit: &[QGate]) -> Vec<QGate> {
    optimize_circuit(circuit, &OptimizationConfig::aggressive()).circuit
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_no_optimization() {
        let circuit = vec![QGate::H(0), QGate::X(0), QGate::H(0)];
        let config = OptimizationConfig::none();
        let result = optimize_circuit(&circuit, &config);

        assert_eq!(result.circuit.len(), 3);
        assert_eq!(result.stats.reduction_percent(), 0.0);
    }

    #[test]
    fn test_basic_cancellation() {
        // H·H = I
        let circuit = vec![QGate::H(0), QGate::H(0)];
        let config = OptimizationConfig::basic();
        let result = optimize_circuit(&circuit, &config);

        assert_eq!(result.circuit.len(), 0);
        assert_eq!(result.stats.cancelled_gates, 2);
        assert_eq!(result.stats.reduction_percent(), 100.0);
    }

    #[test]
    fn test_rotation_merge() {
        // Rz(π/4)·Rz(π/4) = Rz(π/2)
        let circuit = vec![QGate::Rz(0, PI / 4.0), QGate::Rz(0, PI / 4.0)];
        let config = OptimizationConfig::standard();
        let result = optimize_circuit(&circuit, &config);

        assert_eq!(result.circuit.len(), 1);
        if let QGate::Rz(_, angle) = result.circuit[0] {
            assert!((angle - PI / 2.0).abs() < 1e-6);
        } else {
            panic!("Expected Rz gate");
        }
    }

    #[test]
    fn test_commutation_cancellation() {
        // H(0) - X(1) - H(0) should cancel via commutation
        let circuit = vec![QGate::H(0), QGate::X(1), QGate::H(0)];
        let config = OptimizationConfig::standard();
        let result = optimize_circuit(&circuit, &config);

        // H gates should cancel, leaving X(1)
        assert_eq!(result.circuit.len(), 1);
        assert!(matches!(result.circuit[0], QGate::X(1)));
    }

    #[test]
    fn test_cnot_cancellation() {
        // CNOT·CNOT = I
        let circuit = vec![QGate::CNot(0, 1), QGate::CNot(0, 1)];
        let result = optimize(&circuit);

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_aggressive_optimization() {
        // Complex circuit that benefits from aggressive optimization
        let circuit = vec![
            QGate::H(0),
            QGate::Rz(0, PI / 4.0),
            QGate::X(1), // Different qubit - can commute
            QGate::Rz(0, PI / 4.0),
            QGate::H(0),
        ];
        let result = optimize_aggressive(&circuit);

        // Aggressive should:
        // 1. Merge Rz gates -> Rz(π/2)
        // 2. Possibly simplify H·Rz·H pattern
        assert!(result.len() < circuit.len());
    }

    #[test]
    fn test_optimization_stats() {
        let circuit = vec![
            QGate::H(0),
            QGate::H(0), // Cancels
            QGate::X(0),
            QGate::X(0), // Cancels
            QGate::Rz(0, PI / 4.0),
            QGate::Rz(0, PI / 4.0), // Merges
        ];
        let config = OptimizationConfig::standard();
        let result = optimize_circuit(&circuit, &config);

        assert!(result.stats.cancelled_gates >= 4);
        assert!(result.stats.reduction_percent() > 50.0);
    }

    #[test]
    fn test_empty_circuit() {
        let circuit: Vec<QGate> = vec![];
        let result = optimize(&circuit);
        assert!(result.is_empty());
    }

    #[test]
    fn test_speedup_factor() {
        let stats = OptimizationStats {
            original_gates: 100,
            final_gates: 25,
            ..Default::default()
        };
        assert!((stats.speedup_factor() - 4.0).abs() < 1e-6);
        assert!((stats.reduction_percent() - 75.0).abs() < 1e-6);
    }
}
