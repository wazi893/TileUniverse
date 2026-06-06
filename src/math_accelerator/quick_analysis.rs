//! Fast O(n) pre-analysis for routing decisions.
//!
//! This module provides quick classification of circuits to determine
//! if full algebraic analysis is needed or if a fast-path decision can be made.
//!
//! # Performance Considerations
//!
//! For small circuits, the overhead of algebraic analysis exceeds execution time.
//! Quick classification runs in O(n) where n is the gate count, enabling
//! fast-path decisions for:
//! - Small circuits (< 50 gates): skip analysis, just execute
//! - GHZ patterns: route directly to sparse backend
//! - W state patterns: route directly to sparse backend

use logic_fabric_core::quantum::QGate;

use super::QuickClassification;
use super::metrics::{detect_ghz_pattern, detect_w_state_pattern};

/// Threshold for "small" circuits that don't need analysis.
/// Analysis overhead exceeds execution time below this threshold.
const SMALL_CIRCUIT_THRESHOLD: usize = 50;

/// Quickly classify a circuit to determine the routing fast-path.
///
/// This function runs in O(n) time and determines whether:
/// 1. The circuit is small enough to skip analysis
/// 2. A special pattern (GHZ, W) is detected
/// 3. Full algebraic analysis is needed
///
/// # Arguments
/// * `circuit` - The quantum circuit to classify
/// * `_n_qubits` - Total number of qubits (reserved for future use)
///
/// # Returns
/// [`QuickClassification`] indicating the recommended routing approach
///
/// # Performance
/// Target: < 100 microseconds for 1000-gate circuits
///
/// # Example
/// ```ignore
/// use logic_fabric_core::quantum::QGate;
/// use engine::math_accelerator::{quick_classify, QuickClassification};
///
/// // Small circuit: skip analysis
/// let small = vec![QGate::H(0), QGate::CNot(0, 1)];
/// assert_eq!(quick_classify(&small, 2), QuickClassification::Small);
///
/// // GHZ pattern: fast-path to sparse
/// let ghz = vec![
///     QGate::H(0),
///     QGate::CNot(0, 1),
///     QGate::CNot(1, 2),
/// ];
/// // Note: GHZ detection requires circuit.len() >= SMALL_CIRCUIT_THRESHOLD
/// // or special handling for known patterns
/// ```
pub fn quick_classify(circuit: &[QGate], _n_qubits: u8) -> QuickClassification {
    // Small circuits: skip analysis, execution is cheaper
    if circuit.len() < SMALL_CIRCUIT_THRESHOLD {
        // Even for small circuits, check for known sparse patterns
        // GHZ and W states are always sparse regardless of size
        if detect_ghz_pattern(circuit) {
            return QuickClassification::GHZ;
        }
        if detect_w_state_pattern(circuit) {
            return QuickClassification::WState;
        }
        return QuickClassification::Small;
    }

    // For larger circuits, check for patterns first
    // Pattern detection is O(n) but can avoid O(n²) algebraic analysis
    if detect_ghz_pattern(circuit) {
        return QuickClassification::GHZ;
    }

    if detect_w_state_pattern(circuit) {
        return QuickClassification::WState;
    }

    // Large circuit without obvious pattern: needs full analysis
    QuickClassification::NeedsFullAnalysis
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_circuit_classification() {
        let small = vec![QGate::H(0), QGate::X(1), QGate::CNot(0, 1)];
        assert_eq!(quick_classify(&small, 2), QuickClassification::Small);
    }

    #[test]
    fn test_small_ghz_detected() {
        // Even small GHZ circuits should be detected
        let ghz = vec![QGate::H(0), QGate::CNot(0, 1), QGate::CNot(1, 2)];
        assert_eq!(quick_classify(&ghz, 3), QuickClassification::GHZ);
    }

    #[test]
    fn test_large_circuit_needs_analysis() {
        // Generate a large random-ish circuit
        let large: Vec<QGate> = (0..100)
            .map(|i| {
                if i % 3 == 0 {
                    QGate::H((i % 8) as u8)
                } else if i % 3 == 1 {
                    QGate::X((i % 8) as u8)
                } else {
                    QGate::CNot((i % 7) as u8, ((i + 1) % 7) as u8)
                }
            })
            .collect();

        assert_eq!(
            quick_classify(&large, 8),
            QuickClassification::NeedsFullAnalysis
        );
    }

    #[test]
    fn test_large_ghz_detected() {
        // Large GHZ circuit should still be detected
        let mut ghz = vec![QGate::H(0)];
        for i in 0..60 {
            ghz.push(QGate::CNot(i, i + 1));
        }
        assert_eq!(quick_classify(&ghz, 61), QuickClassification::GHZ);
    }
}
