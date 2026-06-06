//! Toffoli Gate Decomposition Tracking
//!
//! SPRINT 51.0 Phase 3: Toffoli Decomposition
//!
//! This module tracks different Toffoli gate decomposition variants and
//! their resource costs. Different decompositions trade off between
//! T-gate count, T-depth, and ancilla requirements.
//!
//! # Decomposition Variants
//!
//! | Variant | T-gates | T-depth | CNOTs | Ancillas |
//! |---------|---------|---------|-------|----------|
//! | Standard | 7 | 4 | 6 | 0 |
//! | CCZ-based | 7 | 3 | 4 | 0 |
//! | Relative-phase | 4 | 2 | 4 | 0 |
//! | Magic injection | 4 | 1 | 2 | 1 |
//!
//! # Example
//!
//! ```ignore
//! use engine::qram::toffoli_decomp::{ToffoliDecomposition, ToffoliTracker};
//!
//! let mut tracker = ToffoliTracker::new();
//! tracker.track_routing(0, 1, 2, 0, 0); // QRAM routing Toffoli
//!
//! let budget = tracker.to_budget();
//! println!("Total T-gates: {}", budget.total_t_count_bare);
//! ```

use std::fmt;

use super::magic_budget::MagicStateBudget;

/// Toffoli gate decomposition variants
///
/// Each variant trades off between different resource costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToffoliDecomposition {
    /// Standard decomposition: 6 CNOTs + 7 T gates (Nielsen-Chuang)
    /// T-depth: 4, no ancillas
    Standard,

    /// CCZ-based decomposition: convert to CCZ first
    /// 4 CNOTs + 7 T gates, T-depth: 3
    CCZBased,

    /// Relative-phase Toffoli: only correct up to phase
    /// 4 CNOTs + 4 T gates, T-depth: 2
    /// Valid when overall phase doesn't matter
    RelativePhase,

    /// Magic state injection: use pre-distilled states
    /// 2 CNOTs + 1 magic state (equivalent to 4 T gates)
    /// T-depth: 1, requires 1 ancilla
    MagicStateInjection,

    /// Measurement-based: uses ancilla and measurement
    /// 3 CNOTs + 4 T gates + measurement
    MeasurementBased,
}

impl ToffoliDecomposition {
    /// T-gate count for this decomposition
    pub fn t_count(&self) -> usize {
        match self {
            Self::Standard => 7,
            Self::CCZBased => 7,
            Self::RelativePhase => 4,
            Self::MagicStateInjection => 4, // Magic state = 4 T gates equivalent
            Self::MeasurementBased => 4,
        }
    }

    /// CNOT count for this decomposition
    pub fn cnot_count(&self) -> usize {
        match self {
            Self::Standard => 6,
            Self::CCZBased => 4,
            Self::RelativePhase => 4,
            Self::MagicStateInjection => 2,
            Self::MeasurementBased => 3,
        }
    }

    /// T-depth (sequential T gates) for this decomposition
    pub fn t_depth(&self) -> usize {
        match self {
            Self::Standard => 4,
            Self::CCZBased => 3,
            Self::RelativePhase => 2,
            Self::MagicStateInjection => 1,
            Self::MeasurementBased => 2,
        }
    }

    /// Ancilla qubit count
    pub fn ancilla_count(&self) -> usize {
        match self {
            Self::Standard => 0,
            Self::CCZBased => 0,
            Self::RelativePhase => 0,
            Self::MagicStateInjection => 1,
            Self::MeasurementBased => 1,
        }
    }

    /// Whether this decomposition requires measurement
    pub fn requires_measurement(&self) -> bool {
        matches!(self, Self::MeasurementBased)
    }

    /// Whether this decomposition is exact (preserves global phase)
    pub fn is_exact(&self) -> bool {
        !matches!(self, Self::RelativePhase)
    }

    /// Total two-qubit gate count
    pub fn two_qubit_count(&self) -> usize {
        self.cnot_count()
    }

    /// Default decomposition (standard)
    pub fn default_decomposition() -> Self {
        Self::Standard
    }

    /// Best decomposition for minimizing T-count
    pub fn min_t_count() -> Self {
        Self::RelativePhase // 4 T gates
    }

    /// Best decomposition for minimizing T-depth
    pub fn min_t_depth() -> Self {
        Self::MagicStateInjection // T-depth 1
    }
}

impl fmt::Display for ToffoliDecomposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard(7T, 6CNOT, depth=4)"),
            Self::CCZBased => write!(f, "CCZ(7T, 4CNOT, depth=3)"),
            Self::RelativePhase => write!(f, "RelPhase(4T, 4CNOT, depth=2)"),
            Self::MagicStateInjection => write!(f, "MagicInj(4T, 2CNOT, depth=1)"),
            Self::MeasurementBased => write!(f, "Measure(4T, 3CNOT, depth=2)"),
        }
    }
}

/// Context where a Toffoli gate appears
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToffoliContext {
    /// Part of QRAM routing (Fredkin gate decomposition)
    QRAMRouting {
        /// Level in routing tree
        level: usize,
        /// Router index at that level
        router: usize,
    },
    /// Part of error correction circuit
    ErrorCorrection {
        /// Syndrome bit being computed
        syndrome: usize,
    },
    /// Part of data manipulation (e.g., arithmetic)
    DataOperation {
        /// Operation name
        operation: String,
    },
    /// General/unknown context
    General,
}

impl fmt::Display for ToffoliContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QRAMRouting { level, router } => {
                write!(f, "QRAM(L{},R{})", level, router)
            }
            Self::ErrorCorrection { syndrome } => {
                write!(f, "EC(S{})", syndrome)
            }
            Self::DataOperation { operation } => {
                write!(f, "Data({})", operation)
            }
            Self::General => write!(f, "General"),
        }
    }
}

/// A tracked Toffoli gate with full metadata
#[derive(Clone, Debug)]
pub struct TrackedToffoli {
    /// First control qubit
    pub control1: usize,
    /// Second control qubit
    pub control2: usize,
    /// Target qubit
    pub target: usize,
    /// Decomposition method
    pub decomposition: ToffoliDecomposition,
    /// Context where this Toffoli appears
    pub context: ToffoliContext,
    /// Sequential index (order of application)
    pub index: usize,
}

impl TrackedToffoli {
    /// Create new tracked Toffoli
    pub fn new(
        control1: usize,
        control2: usize,
        target: usize,
        decomposition: ToffoliDecomposition,
        context: ToffoliContext,
        index: usize,
    ) -> Self {
        Self {
            control1,
            control2,
            target,
            decomposition,
            context,
            index,
        }
    }

    /// Get T-gate count for this Toffoli
    pub fn t_count(&self) -> usize {
        self.decomposition.t_count()
    }

    /// Get T-depth for this Toffoli
    pub fn t_depth(&self) -> usize {
        self.decomposition.t_depth()
    }
}

impl fmt::Display for TrackedToffoli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Toffoli({},{},{}) {} [{}]",
            self.control1, self.control2, self.target, self.decomposition, self.context
        )
    }
}

/// Tracker for Toffoli decompositions and resources
#[derive(Clone, Debug, Default)]
pub struct ToffoliTracker {
    /// Tracked Toffoli gates
    toffolis: Vec<TrackedToffoli>,
    /// Default decomposition to use
    default_decomposition: Option<ToffoliDecomposition>,
    /// Running T-count
    total_t_count: usize,
    /// Running T-depth (conservative: sum of depths)
    total_t_depth: usize,
}

impl ToffoliTracker {
    /// Create new tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Create tracker with specific default decomposition
    pub fn with_decomposition(decomposition: ToffoliDecomposition) -> Self {
        Self {
            default_decomposition: Some(decomposition),
            ..Default::default()
        }
    }

    /// Track a general Toffoli gate
    pub fn track(
        &mut self,
        control1: usize,
        control2: usize,
        target: usize,
        context: ToffoliContext,
    ) {
        let decomposition = self
            .default_decomposition
            .unwrap_or(ToffoliDecomposition::Standard);
        let index = self.toffolis.len();

        let toffoli =
            TrackedToffoli::new(control1, control2, target, decomposition, context, index);

        self.total_t_count += toffoli.t_count();
        self.total_t_depth += toffoli.t_depth();
        self.toffolis.push(toffoli);
    }

    /// Track a QRAM routing Toffoli
    pub fn track_routing(
        &mut self,
        control1: usize,
        control2: usize,
        target: usize,
        level: usize,
        router: usize,
    ) {
        self.track(
            control1,
            control2,
            target,
            ToffoliContext::QRAMRouting { level, router },
        );
    }

    /// Track an error correction Toffoli
    pub fn track_error_correction(
        &mut self,
        control1: usize,
        control2: usize,
        target: usize,
        syndrome: usize,
    ) {
        self.track(
            control1,
            control2,
            target,
            ToffoliContext::ErrorCorrection { syndrome },
        );
    }

    /// Track a data operation Toffoli
    pub fn track_data_operation(
        &mut self,
        control1: usize,
        control2: usize,
        target: usize,
        operation: &str,
    ) {
        self.track(
            control1,
            control2,
            target,
            ToffoliContext::DataOperation {
                operation: operation.to_string(),
            },
        );
    }

    /// Get total Toffoli count
    pub fn count(&self) -> usize {
        self.toffolis.len()
    }

    /// Get total T-count
    pub fn t_count(&self) -> usize {
        self.total_t_count
    }

    /// Get total T-depth (conservative estimate)
    pub fn t_depth(&self) -> usize {
        self.total_t_depth
    }

    /// Get Toffolis by context
    pub fn by_context(&self, context_type: &str) -> Vec<&TrackedToffoli> {
        self.toffolis
            .iter()
            .filter(|t| match (&t.context, context_type) {
                (ToffoliContext::QRAMRouting { .. }, "routing") => true,
                (ToffoliContext::ErrorCorrection { .. }, "correction") => true,
                (ToffoliContext::DataOperation { .. }, "data") => true,
                (ToffoliContext::General, "general") => true,
                _ => false,
            })
            .collect()
    }

    /// Count Toffolis in QRAM routing
    pub fn routing_count(&self) -> usize {
        self.toffolis
            .iter()
            .filter(|t| matches!(t.context, ToffoliContext::QRAMRouting { .. }))
            .count()
    }

    /// Count Toffolis in error correction
    pub fn correction_count(&self) -> usize {
        self.toffolis
            .iter()
            .filter(|t| matches!(t.context, ToffoliContext::ErrorCorrection { .. }))
            .count()
    }

    /// Convert to magic state budget
    pub fn to_budget(&self) -> MagicStateBudget {
        let mut budget = MagicStateBudget::from_gate_counts(self.count(), 0, 0);

        // Break down by context
        budget.routing_t_gates = self
            .toffolis
            .iter()
            .filter(|t| matches!(t.context, ToffoliContext::QRAMRouting { .. }))
            .map(|t| t.t_count())
            .sum();

        budget.correction_t_gates = self
            .toffolis
            .iter()
            .filter(|t| matches!(t.context, ToffoliContext::ErrorCorrection { .. }))
            .map(|t| t.t_count())
            .sum();

        budget.data_t_gates = self
            .toffolis
            .iter()
            .filter(|t| matches!(t.context, ToffoliContext::DataOperation { .. }))
            .map(|t| t.t_count())
            .sum();

        budget.t_depth = self.t_depth();

        budget
    }

    /// Get all tracked Toffolis
    pub fn toffolis(&self) -> &[TrackedToffoli] {
        &self.toffolis
    }

    /// Clear all tracked gates
    pub fn clear(&mut self) {
        self.toffolis.clear();
        self.total_t_count = 0;
        self.total_t_depth = 0;
    }

    /// Get decomposition statistics
    pub fn decomposition_stats(&self) -> DecompositionStats {
        let mut stats = DecompositionStats::default();

        for toffoli in &self.toffolis {
            match toffoli.decomposition {
                ToffoliDecomposition::Standard => stats.standard += 1,
                ToffoliDecomposition::CCZBased => stats.ccz_based += 1,
                ToffoliDecomposition::RelativePhase => stats.relative_phase += 1,
                ToffoliDecomposition::MagicStateInjection => stats.magic_injection += 1,
                ToffoliDecomposition::MeasurementBased => stats.measurement_based += 1,
            }
        }

        stats
    }
}

impl fmt::Display for ToffoliTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ToffoliTracker({} gates, {} T-gates, {} T-depth)",
            self.count(),
            self.t_count(),
            self.t_depth()
        )
    }
}

/// Statistics about decomposition types used
#[derive(Clone, Debug, Default)]
pub struct DecompositionStats {
    /// Standard decomposition count
    pub standard: usize,
    /// CCZ-based count
    pub ccz_based: usize,
    /// Relative-phase count
    pub relative_phase: usize,
    /// Magic state injection count
    pub magic_injection: usize,
    /// Measurement-based count
    pub measurement_based: usize,
}

impl DecompositionStats {
    /// Total count
    pub fn total(&self) -> usize {
        self.standard
            + self.ccz_based
            + self.relative_phase
            + self.magic_injection
            + self.measurement_based
    }
}

/// Select optimal decomposition based on constraints
pub fn select_decomposition(
    minimize: &str,
    allow_ancilla: bool,
    require_exact: bool,
) -> ToffoliDecomposition {
    match minimize {
        "t_depth" => {
            if allow_ancilla {
                ToffoliDecomposition::MagicStateInjection
            } else {
                ToffoliDecomposition::CCZBased
            }
        }
        "t_count" => {
            if require_exact {
                ToffoliDecomposition::CCZBased
            } else {
                ToffoliDecomposition::RelativePhase
            }
        }
        "cnot" => ToffoliDecomposition::MagicStateInjection,
        _ => ToffoliDecomposition::Standard,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposition_counts() {
        let std = ToffoliDecomposition::Standard;
        assert_eq!(std.t_count(), 7);
        assert_eq!(std.cnot_count(), 6);
        assert_eq!(std.t_depth(), 4);

        let rel = ToffoliDecomposition::RelativePhase;
        assert_eq!(rel.t_count(), 4);
        assert!(!rel.is_exact());
    }

    #[test]
    fn test_tracker_basic() {
        let mut tracker = ToffoliTracker::new();
        tracker.track(0, 1, 2, ToffoliContext::General);

        assert_eq!(tracker.count(), 1);
        assert_eq!(tracker.t_count(), 7);
    }

    #[test]
    fn test_tracker_routing() {
        let mut tracker = ToffoliTracker::new();
        tracker.track_routing(0, 1, 2, 0, 0);
        tracker.track_routing(0, 3, 4, 1, 0);
        tracker.track_routing(0, 3, 5, 1, 1);

        assert_eq!(tracker.routing_count(), 3);
        assert_eq!(tracker.t_count(), 21);
    }

    #[test]
    fn test_tracker_to_budget() {
        let mut tracker = ToffoliTracker::new();
        tracker.track_routing(0, 1, 2, 0, 0);
        tracker.track_error_correction(0, 1, 3, 0);

        let budget = tracker.to_budget();
        assert_eq!(budget.toffoli_count, 2);
        assert_eq!(budget.routing_t_gates, 7);
        assert_eq!(budget.correction_t_gates, 7);
    }

    #[test]
    fn test_different_decompositions() {
        let tracker_std = ToffoliTracker::with_decomposition(ToffoliDecomposition::Standard);
        let tracker_rel = ToffoliTracker::with_decomposition(ToffoliDecomposition::RelativePhase);

        let mut t1 = tracker_std;
        let mut t2 = tracker_rel;

        t1.track(0, 1, 2, ToffoliContext::General);
        t2.track(0, 1, 2, ToffoliContext::General);

        assert_eq!(t1.t_count(), 7);
        assert_eq!(t2.t_count(), 4);
    }

    #[test]
    fn test_decomposition_stats() {
        let mut tracker = ToffoliTracker::new();
        tracker.track(0, 1, 2, ToffoliContext::General);
        tracker.track(0, 1, 3, ToffoliContext::General);

        let stats = tracker.decomposition_stats();
        assert_eq!(stats.standard, 2);
        assert_eq!(stats.total(), 2);
    }

    #[test]
    fn test_select_decomposition() {
        let min_depth = select_decomposition("t_depth", true, true);
        assert_eq!(min_depth, ToffoliDecomposition::MagicStateInjection);

        let min_t = select_decomposition("t_count", false, false);
        assert_eq!(min_t, ToffoliDecomposition::RelativePhase);
    }

    #[test]
    fn test_display() {
        let decomp = ToffoliDecomposition::Standard;
        let s = format!("{}", decomp);
        assert!(s.contains("7T"));

        let context = ToffoliContext::QRAMRouting {
            level: 2,
            router: 3,
        };
        let s = format!("{}", context);
        assert!(s.contains("QRAM"));
    }
}
