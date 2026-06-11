//! SPRINT 54.0: T-Depth Minimization
//!
//! This module implements T-gate scheduling optimizations to minimize T-depth.
//! T-depth is the number of sequential T-gate layers, which directly impacts
//! the number of magic state factory cycles needed in fault-tolerant execution.
//!
//! ## Key Insight
//!
//! T gates on different qubits can execute in parallel. By commuting T gates
//! past compatible gates, we can group them into layers and reduce T-depth.
//!
//! ## Commutation Rules for T Gates
//!
//! | Gate A | Gate B | Commutes? | Reason |
//! |--------|--------|-----------|--------|
//! | T(q) | X(p) | Yes if q≠p | Disjoint qubits |
//! | T(q) | CNOT(c,t) | Yes if q=c | T is Z-axis, commutes on control |
//! | T(q) | CZ(a,b) | Yes | CZ is diagonal, commutes with T |
//! | T(q) | Rz(p,θ) | Yes | Same axis (Z) on any qubit |
//! | T(q) | Phase(p,θ) | Yes | Same axis (Z) |
//! | T(q) | H(p) | Yes if q≠p | Disjoint qubits |
//!
//! ## T-Depth Reduction Strategy
//!
//! 1. Identify all T/Tdg gates in the circuit
//! 2. Build a dependency graph based on non-commuting gates
//! 3. Schedule T gates into minimal layers respecting dependencies
//! 4. Reorder circuit to realize the optimized schedule

use logic_fabric_core::quantum::QGate;
use std::collections::{HashMap, HashSet};

/// T-depth analysis result
#[derive(Clone, Debug)]
pub struct TDepthAnalysis {
    /// Original T-depth before optimization
    pub original_t_depth: usize,
    /// Optimized T-depth after scheduling
    pub optimized_t_depth: usize,
    /// Total T-gate count (unchanged by scheduling)
    pub t_count: usize,
    /// T-gate layers (indices into original circuit)
    pub t_layers: Vec<Vec<usize>>,
    /// Number of T gates moved
    pub gates_moved: usize,
}

impl TDepthAnalysis {
    /// Calculate T-depth reduction percentage
    pub fn reduction_percent(&self) -> f64 {
        if self.original_t_depth == 0 {
            return 0.0;
        }
        let reduced = self.original_t_depth.saturating_sub(self.optimized_t_depth);
        (reduced as f64 / self.original_t_depth as f64) * 100.0
    }

    /// Calculate parallelization factor
    pub fn parallelization_factor(&self) -> f64 {
        if self.optimized_t_depth == 0 {
            return 1.0;
        }
        self.t_count as f64 / self.optimized_t_depth as f64
    }
}

/// Check if a gate is a T or Tdg gate
fn is_t_gate(gate: &QGate) -> bool {
    matches!(gate, QGate::T(_) | QGate::Tdg(_))
}

/// Get the qubit a T gate operates on
fn t_gate_qubit(gate: &QGate) -> Option<u8> {
    match gate {
        QGate::T(q) | QGate::Tdg(q) => Some(*q),
        _ => None,
    }
}

/// Check if a T gate can commute past another gate
///
/// T gates commute with:
/// - Gates on disjoint qubits
/// - Z-axis gates (Rz, Phase, Z, T, Tdg) on any qubit
/// - CNOT where T is on the control qubit
/// - CZ (diagonal, always commutes)
fn t_commutes_with(t_qubit: u8, other: &QGate) -> bool {
    match other {
        // Disjoint single-qubit gates
        QGate::H(q) | QGate::X(q) | QGate::Y(q) => *q != t_qubit,

        // Z-axis gates always commute with T
        QGate::Z(_) | QGate::Rz(_, _) | QGate::Phase(_, _) | QGate::T(_) | QGate::Tdg(_) => true,

        // CNOT: T on control commutes, T on target does not
        QGate::CNot(ctrl, target) => {
            if t_qubit == *ctrl {
                true // T on control commutes
            } else if t_qubit == *target {
                false // T on target does NOT commute
            } else {
                true // Disjoint qubits
            }
        }

        // CZ is diagonal, always commutes with T
        QGate::CZ(_, _) => true,

        // Swap: T commutes only if on disjoint qubits
        QGate::Swap(a, b) => t_qubit != *a && t_qubit != *b,

        // Rotation gates on different axes
        QGate::Rx(q, _) | QGate::Ry(q, _) => *q != t_qubit,

        // U3 is general, doesn't commute on same qubit
        QGate::U3(q, _, _, _) => *q != t_qubit,

        // Toffoli/CCZ: T commutes if not on target
        QGate::Toffoli(_, _, t) => t_qubit != *t,
        QGate::CCZ(_, _, _) => true, // CCZ is diagonal

        // Controlled rotations
        QGate::CPhase(_, _, _) | QGate::CRz(_, _, _) => true, // Diagonal

        // Measure: cannot commute past measurement
        QGate::Measure(q) => *q != t_qubit,
    }
}

/// Build dependency graph for T gates
///
/// Returns: For each T gate index, the set of T gate indices it depends on
///
/// Two T gates have a dependency if:
/// 1. They are on the same qubit (must be sequential), OR
/// 2. There's a gate between them that doesn't commute with both
fn build_t_dependency_graph(circuit: &[QGate]) -> HashMap<usize, HashSet<usize>> {
    let t_indices: Vec<usize> = circuit
        .iter()
        .enumerate()
        .filter(|(_, g)| is_t_gate(g))
        .map(|(i, _)| i)
        .collect();

    let mut deps: HashMap<usize, HashSet<usize>> = HashMap::new();

    for &t_idx in &t_indices {
        deps.insert(t_idx, HashSet::new());
    }

    // For each pair of T gates, check if there's a blocking gate between them
    for (i, &t1_idx) in t_indices.iter().enumerate() {
        let t1_qubit = t_gate_qubit(&circuit[t1_idx]).unwrap();

        for &t2_idx in t_indices.iter().skip(i + 1) {
            let t2_qubit = t_gate_qubit(&circuit[t2_idx]).unwrap();

            // T gates on the same qubit always have a dependency
            if t1_qubit == t2_qubit {
                deps.get_mut(&t2_idx).unwrap().insert(t1_idx);
                continue;
            }

            // T gates on different qubits: check intervening gates
            // They only have a dependency if an intervening gate creates one
            let mut blocked = false;
            for gate_idx in (t1_idx + 1)..t2_idx {
                let gate = &circuit[gate_idx];
                let gate_qubits = get_gate_qubits(gate);

                // If the gate involves both T qubits and doesn't commute with T
                // on one of those qubits, there's a dependency
                let involves_t1 = gate_qubits.contains(&t1_qubit);
                let involves_t2 = gate_qubits.contains(&t2_qubit);

                if involves_t1 && involves_t2 {
                    // Two-qubit gate on both T qubits - check commutation
                    if !t_commutes_with(t1_qubit, gate) || !t_commutes_with(t2_qubit, gate) {
                        blocked = true;
                        break;
                    }
                }
            }

            if blocked {
                deps.get_mut(&t2_idx).unwrap().insert(t1_idx);
            }
        }
    }

    deps
}

/// Get qubits involved in a gate
fn get_gate_qubits(gate: &QGate) -> Vec<u8> {
    match gate {
        QGate::H(q)
        | QGate::X(q)
        | QGate::Y(q)
        | QGate::Z(q)
        | QGate::T(q)
        | QGate::Tdg(q)
        | QGate::Rx(q, _)
        | QGate::Ry(q, _)
        | QGate::Rz(q, _)
        | QGate::Phase(q, _)
        | QGate::U3(q, _, _, _)
        | QGate::Measure(q) => vec![*q],

        QGate::CNot(c, t) | QGate::CZ(c, t) | QGate::Swap(c, t) => vec![*c, *t],

        QGate::CPhase(c, t, _) | QGate::CRz(c, t, _) => vec![*c, *t],

        QGate::Toffoli(c1, c2, t) | QGate::CCZ(c1, c2, t) => vec![*c1, *c2, *t],
    }
}

/// Schedule T gates into layers using ASAP (As Soon As Possible) algorithm
fn schedule_t_gates(deps: &HashMap<usize, HashSet<usize>>) -> Vec<Vec<usize>> {
    if deps.is_empty() {
        return vec![];
    }

    let mut layers: Vec<Vec<usize>> = vec![];
    let mut scheduled: HashSet<usize> = HashSet::new();
    let mut layer_assignment: HashMap<usize, usize> = HashMap::new();

    // ASAP scheduling: each T gate goes in the earliest layer where all deps are satisfied
    let mut t_indices: Vec<usize> = deps.keys().cloned().collect();
    t_indices.sort(); // Process in circuit order

    for &t_idx in &t_indices {
        let t_deps = &deps[&t_idx];

        // Find the earliest layer after all dependencies
        let min_layer = if t_deps.is_empty() {
            0
        } else {
            t_deps
                .iter()
                .map(|&dep| layer_assignment.get(&dep).unwrap_or(&0) + 1)
                .max()
                .unwrap_or(0)
        };

        // Assign to layer
        layer_assignment.insert(t_idx, min_layer);
        scheduled.insert(t_idx);

        // Ensure layer exists
        while layers.len() <= min_layer {
            layers.push(vec![]);
        }
        layers[min_layer].push(t_idx);
    }

    layers
}

/// Analyze T-depth and compute optimal scheduling
pub fn analyze_t_depth(circuit: &[QGate]) -> TDepthAnalysis {
    // Count T gates and find original T-depth (sequential)
    let t_positions: Vec<(usize, u8)> = circuit
        .iter()
        .enumerate()
        .filter_map(|(i, g)| t_gate_qubit(g).map(|q| (i, q)))
        .collect();

    let t_count = t_positions.len();

    if t_count == 0 {
        return TDepthAnalysis {
            original_t_depth: 0,
            optimized_t_depth: 0,
            t_count: 0,
            t_layers: vec![],
            gates_moved: 0,
        };
    }

    // Original T-depth: count sequential T-gate "waves"
    // A new wave starts when we see a T on a qubit that had a T earlier with blocking gates between
    let original_t_depth = compute_original_t_depth(circuit);

    // Build dependency graph and schedule
    let deps = build_t_dependency_graph(circuit);
    let t_layers = schedule_t_gates(&deps);
    let optimized_t_depth = t_layers.len();

    // Count gates that would need to move
    let gates_moved = count_gates_moved(&t_layers, circuit);

    TDepthAnalysis {
        original_t_depth,
        optimized_t_depth,
        t_count,
        t_layers,
        gates_moved,
    }
}

/// Compute original T-depth by simulating sequential execution
fn compute_original_t_depth(circuit: &[QGate]) -> usize {
    let mut qubit_last_t_layer: HashMap<u8, usize> = HashMap::new();
    let mut max_layer = 0;

    for gate in circuit {
        if let Some(q) = t_gate_qubit(gate) {
            let layer = qubit_last_t_layer.get(&q).map(|l| l + 1).unwrap_or(0);
            qubit_last_t_layer.insert(q, layer);
            max_layer = max_layer.max(layer + 1);
        }
    }

    max_layer
}

/// Count how many gates would need to be moved for optimal scheduling
fn count_gates_moved(t_layers: &[Vec<usize>], circuit: &[QGate]) -> usize {
    if t_layers.is_empty() {
        return 0;
    }

    // For each layer, check if T gates are already consecutive
    let mut moved = 0;

    for layer in t_layers {
        if layer.len() < 2 {
            continue;
        }

        // Check if indices are consecutive (modulo intervening non-T gates)
        let mut sorted_layer = layer.clone();
        sorted_layer.sort();

        for i in 0..sorted_layer.len() - 1 {
            let start = sorted_layer[i];
            let end = sorted_layer[i + 1];

            // Count non-T gates between consecutive T gates in the layer
            for idx in (start + 1)..end {
                if !is_t_gate(&circuit[idx]) {
                    moved += 1;
                }
            }
        }
    }

    moved
}

/// Minimize T-depth by reordering the circuit
///
/// This produces an equivalent circuit with minimized T-depth.
pub fn minimize_t_depth(circuit: &[QGate]) -> (Vec<QGate>, TDepthAnalysis) {
    let analysis = analyze_t_depth(circuit);

    if analysis.t_count == 0 || analysis.original_t_depth <= analysis.optimized_t_depth {
        return (circuit.to_vec(), analysis);
    }

    // Build optimized circuit by respecting T-layer schedule
    let mut result = Vec::with_capacity(circuit.len());
    let mut used: HashSet<usize> = HashSet::new();

    // Collect T gate indices by layer
    let t_in_layer: HashSet<usize> = analysis.t_layers.iter().flatten().cloned().collect();

    // Process circuit, pulling T gates forward when possible
    let mut pending_t_gates: Vec<(usize, QGate)> = vec![];
    let mut current_layer = 0;

    for (idx, gate) in circuit.iter().enumerate() {
        if t_in_layer.contains(&idx) {
            // This is a T gate - check if it belongs to current layer
            let gate_layer = analysis
                .t_layers
                .iter()
                .position(|layer| layer.contains(&idx))
                .unwrap_or(0);

            if gate_layer == current_layer {
                result.push(gate.clone());
                used.insert(idx);
            } else {
                pending_t_gates.push((idx, gate.clone()));
            }
        } else {
            // Non-T gate - emit pending T gates for current layer first
            let ready: Vec<_> = pending_t_gates
                .iter()
                .filter(|(i, _)| {
                    analysis
                        .t_layers
                        .get(current_layer)
                        .map(|l| l.contains(i))
                        .unwrap_or(false)
                })
                .cloned()
                .collect();

            for (i, g) in ready {
                if !used.contains(&i) {
                    result.push(g);
                    used.insert(i);
                }
            }
            pending_t_gates.retain(|(i, _)| !used.contains(i));

            result.push(gate.clone());
            used.insert(idx);

            // Check if we should advance to next layer
            if analysis
                .t_layers
                .get(current_layer)
                .map(|l| l.iter().all(|i| used.contains(i)))
                .unwrap_or(true)
            {
                current_layer += 1;
            }
        }
    }

    // Emit any remaining T gates
    for (_, g) in pending_t_gates {
        result.push(g);
    }

    (result, analysis)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_t_gates() {
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let analysis = analyze_t_depth(&circuit);

        assert_eq!(analysis.t_count, 0);
        assert_eq!(analysis.original_t_depth, 0);
        assert_eq!(analysis.optimized_t_depth, 0);
    }

    #[test]
    fn test_single_t_gate() {
        let circuit = vec![QGate::H(0), QGate::T(0), QGate::H(0)];
        let analysis = analyze_t_depth(&circuit);

        assert_eq!(analysis.t_count, 1);
        assert_eq!(analysis.original_t_depth, 1);
        assert_eq!(analysis.optimized_t_depth, 1);
    }

    #[test]
    fn test_parallel_t_gates() {
        // T gates on different qubits can be parallel
        let circuit = vec![QGate::T(0), QGate::T(1), QGate::T(2)];
        let analysis = analyze_t_depth(&circuit);

        assert_eq!(analysis.t_count, 3);
        // These should all be in one layer (T-depth 1)
        assert_eq!(analysis.optimized_t_depth, 1);
    }

    #[test]
    fn test_sequential_t_gates_same_qubit() {
        // T gates on same qubit must be sequential
        let circuit = vec![QGate::T(0), QGate::T(0), QGate::T(0)];
        let analysis = analyze_t_depth(&circuit);

        assert_eq!(analysis.t_count, 3);
        assert_eq!(analysis.optimized_t_depth, 3);
    }

    #[test]
    fn test_t_commutes_with_cnot_control() {
        // T(0) on control of CNOT(0,1), T(1) on target
        // CNOT creates a dependency because T on target doesn't commute
        let circuit = vec![QGate::T(0), QGate::CNot(0, 1), QGate::T(1)];
        let analysis = analyze_t_depth(&circuit);

        assert_eq!(analysis.t_count, 2);
        // With CNOT between them, T gates are in sequence
        assert_eq!(analysis.optimized_t_depth, 2);
    }

    #[test]
    fn test_t_blocked_by_cnot_target() {
        // T(1) cannot commute past CNOT(0,1) if another T(1) follows
        let circuit = vec![QGate::T(1), QGate::CNot(0, 1), QGate::T(1)];
        let analysis = analyze_t_depth(&circuit);

        // Same qubit, must be sequential
        assert_eq!(analysis.t_count, 2);
        assert_eq!(analysis.optimized_t_depth, 2);
    }

    #[test]
    fn test_t_commutes_with_cz() {
        // CZ is diagonal, T always commutes
        let circuit = vec![QGate::T(0), QGate::CZ(0, 1), QGate::T(1)];
        let analysis = analyze_t_depth(&circuit);

        assert_eq!(analysis.t_count, 2);
        assert_eq!(analysis.optimized_t_depth, 1);
    }

    #[test]
    fn test_complex_circuit() {
        // A circuit with T gates interspersed with CNOTs
        // T(0) -> CNOT(0,1) -> T(1) -> CNOT(1,2) -> T(2)
        // Each T gate has a CNOT dependency with the next
        let circuit = vec![
            QGate::H(0),
            QGate::T(0),
            QGate::CNot(0, 1),
            QGate::T(1),
            QGate::CNot(1, 2),
            QGate::T(2),
            QGate::H(2),
        ];
        let analysis = analyze_t_depth(&circuit);

        assert_eq!(analysis.t_count, 3);
        // With CNOT chains creating dependencies, T-depth is 3
        // (each T depends on the previous via CNOT)
        assert_eq!(analysis.optimized_t_depth, 3);
    }

    #[test]
    fn test_minimize_t_depth() {
        let circuit = vec![QGate::T(0), QGate::X(1), QGate::T(1)];
        let (optimized, analysis) = minimize_t_depth(&circuit);

        assert_eq!(optimized.len(), circuit.len());
        assert_eq!(analysis.t_count, 2);
        // T(0) and T(1) should be parallel after optimization
        assert_eq!(analysis.optimized_t_depth, 1);
    }

    #[test]
    fn test_parallelization_factor() {
        let analysis = TDepthAnalysis {
            original_t_depth: 10,
            optimized_t_depth: 2,
            t_count: 10,
            t_layers: vec![],
            gates_moved: 0,
        };

        assert!((analysis.parallelization_factor() - 5.0).abs() < 1e-6);
        assert!((analysis.reduction_percent() - 80.0).abs() < 1e-6);
    }
}
