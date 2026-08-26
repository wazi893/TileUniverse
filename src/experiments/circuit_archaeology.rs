// =============================================================================
// src/circuit_archaeology.rs
// =============================================================================
//
// SPRINT 3.3: Circuit Archaeology
//
// Extract and analyze circuits from successful organisms.
// Identify patterns, motifs, and structures that evolution discovered.
//
// Key questions:
// - Do quantum winners use actual quantum structure (entanglement, interference)?
// - Or just "good randomness" from Hadamard gates?
// - Do classical organisms converge on simpler circuits?
// - What gate patterns appear repeatedly in successful lineages?
//
// =============================================================================

use crate::constrained_circuits::{count_entangling_gates, entanglement_fraction};
use crate::quantum::QGate;

use std::collections::HashMap;

// =============================================================================
// Circuit Patterns
// =============================================================================

/// Known quantum circuit patterns that indicate structured quantum computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantumPattern {
    /// H → CNOT: Creates Bell pair / entanglement
    BellPair,

    /// H⊗n → cascade of CNOTs: Creates GHZ state
    GHZSetup,

    /// H → gates → H: Interference pattern (used in DJ, BV)
    HadamardSandwich,

    /// Phase → CNOT → Phase: Phase kickback (used in phase estimation)
    PhaseKickback,

    /// Multiple CNOTs in sequence on same qubits: Entanglement layer
    EntanglementLayer,

    /// H on all qubits: Uniform superposition / randomness
    UniformSuperposition,

    /// CNOT chain: q0→q1→q2→q3
    CNOTCascade,

    /// Controlled-Z network
    CZNetwork,

    /// No discernible quantum structure
    Unstructured,
}

/// Result of pattern detection on a circuit.
#[derive(Clone, Debug)]
pub struct PatternAnalysis {
    /// Patterns found in the circuit
    pub patterns: Vec<(QuantumPattern, usize)>, // (pattern, count)

    /// Gate type distribution
    pub gate_distribution: HashMap<String, usize>,

    /// Circuit metrics
    pub total_gates: usize,
    pub entangling_gates: usize,
    pub single_qubit_gates: usize,
    pub quantum_ness: f32,

    /// Structural metrics
    pub depth_estimate: usize,
    pub qubit_utilization: Vec<usize>, // gates per qubit

    /// Pattern scores
    pub interference_score: f32, // How much H-sandwich structure
    pub entanglement_score: f32, // How structured is entanglement
    pub randomness_score: f32,   // Pure randomness vs structure
}

// =============================================================================
// Pattern Detection
// =============================================================================

/// Detect quantum patterns in a circuit.
pub fn detect_patterns(circuit: &[QGate], n_qubits: u8) -> PatternAnalysis {
    let mut patterns = Vec::new();
    let mut gate_dist = HashMap::new();

    // Count gate types
    for gate in circuit {
        let name = gate_type_name(gate);
        *gate_dist.entry(name.to_string()).or_insert(0) += 1;
    }

    let total = circuit.len();
    let entangling = count_entangling_gates(circuit);
    let single = total - entangling;
    let qness = entanglement_fraction(circuit);

    // Detect specific patterns
    let bell_count = detect_bell_pairs(circuit);
    if bell_count > 0 {
        patterns.push((QuantumPattern::BellPair, bell_count));
    }

    let sandwich_count = detect_hadamard_sandwich(circuit);
    if sandwich_count > 0 {
        patterns.push((QuantumPattern::HadamardSandwich, sandwich_count));
    }

    let superposition_count = detect_uniform_superposition(circuit, n_qubits);
    if superposition_count > 0 {
        patterns.push((QuantumPattern::UniformSuperposition, superposition_count));
    }

    let cascade_count = detect_cnot_cascade(circuit);
    if cascade_count > 0 {
        patterns.push((QuantumPattern::CNOTCascade, cascade_count));
    }

    let layer_count = detect_entanglement_layer(circuit);
    if layer_count > 0 {
        patterns.push((QuantumPattern::EntanglementLayer, layer_count));
    }

    if patterns.is_empty() && total > 0 {
        patterns.push((QuantumPattern::Unstructured, 1));
    }

    // Compute structural metrics
    let depth = estimate_depth(circuit, n_qubits);
    let qubit_util = qubit_utilization(circuit, n_qubits);

    // Compute pattern scores
    let interference_score = sandwich_count as f32 / (total.max(1) as f32 / 3.0).max(1.0);
    let entanglement_score = if entangling > 0 {
        (bell_count + cascade_count + layer_count) as f32 / entangling as f32
    } else {
        0.0
    };
    let randomness_score = if total > 0 {
        *gate_dist.get("H").unwrap_or(&0) as f32 / total as f32
    } else {
        0.0
    };

    PatternAnalysis {
        patterns,
        gate_distribution: gate_dist,
        total_gates: total,
        entangling_gates: entangling,
        single_qubit_gates: single,
        quantum_ness: qness,
        depth_estimate: depth,
        qubit_utilization: qubit_util,
        interference_score: interference_score.min(1.0),
        entanglement_score: entanglement_score.min(1.0),
        randomness_score,
    }
}

/// Detect Bell pair creation patterns (H followed by CNOT on same qubit).
fn detect_bell_pairs(circuit: &[QGate]) -> usize {
    let mut count = 0;

    for i in 0..circuit.len().saturating_sub(1) {
        if let QGate::H(q) = circuit[i] {
            // Look for CNOT with this qubit as control
            for j in (i + 1)..circuit.len().min(i + 4) {
                if let QGate::CNot(ctrl, _) = circuit[j]
                    && ctrl == q
                {
                    count += 1;
                    break;
                }
            }
        }
    }

    count
}

/// Detect Hadamard sandwich patterns (H...gates...H on same qubit).
fn detect_hadamard_sandwich(circuit: &[QGate]) -> usize {
    let mut count = 0;
    let mut h_positions: HashMap<u8, Vec<usize>> = HashMap::new();

    // Find all H gate positions per qubit
    for (i, gate) in circuit.iter().enumerate() {
        if let QGate::H(q) = gate {
            h_positions.entry(*q).or_default().push(i);
        }
    }

    // Count sandwiches (pairs of H with gates between)
    for positions in h_positions.values() {
        if positions.len() >= 2 {
            for i in 0..positions.len() - 1 {
                let gap = positions[i + 1] - positions[i];
                if (2..=10).contains(&gap) {
                    // Check if there are non-H gates between
                    let has_gates = (positions[i] + 1..positions[i + 1])
                        .any(|j| !matches!(circuit[j], QGate::H(_)));
                    if has_gates {
                        count += 1;
                    }
                }
            }
        }
    }

    count
}

/// Detect uniform superposition (H on all qubits at start).
fn detect_uniform_superposition(circuit: &[QGate], n_qubits: u8) -> usize {
    let mut h_qubits = std::collections::HashSet::new();

    // Check first n_qubits gates
    for gate in circuit.iter().take(n_qubits as usize * 2) {
        if let QGate::H(q) = gate
            && *q < n_qubits
        {
            h_qubits.insert(*q);
        }
    }

    if h_qubits.len() == n_qubits as usize {
        1
    } else {
        0
    }
}

/// Detect CNOT cascade (chain of CNOTs: q0→q1→q2→...).
fn detect_cnot_cascade(circuit: &[QGate]) -> usize {
    let mut count = 0;
    let mut i = 0;

    while i < circuit.len() {
        if let QGate::CNot(_ctrl, tgt) = circuit[i] {
            let mut chain_len = 1;
            let mut last_tgt = tgt;

            // Look for continuing chain
            for j in (i + 1)..circuit.len().min(i + 8) {
                if let QGate::CNot(next_ctrl, next_tgt) = circuit[j] {
                    if next_ctrl == last_tgt {
                        chain_len += 1;
                        last_tgt = next_tgt;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            if chain_len >= 2 {
                count += 1;
                i += chain_len;
                continue;
            }
        }
        i += 1;
    }

    count
}

/// Detect entanglement layers (multiple CNOTs close together).
fn detect_entanglement_layer(circuit: &[QGate]) -> usize {
    let mut count = 0;
    let mut i = 0;

    while i < circuit.len() {
        if matches!(circuit[i], QGate::CNot(_, _) | QGate::CZ(_, _)) {
            let mut layer_size = 1;

            // Count consecutive entangling gates
            for j in (i + 1)..circuit.len().min(i + 6) {
                if matches!(circuit[j], QGate::CNot(_, _) | QGate::CZ(_, _)) {
                    layer_size += 1;
                } else {
                    break;
                }
            }

            if layer_size >= 2 {
                count += 1;
                i += layer_size;
                continue;
            }
        }
        i += 1;
    }

    count
}

/// Estimate circuit depth (parallel layers).
fn estimate_depth(circuit: &[QGate], n_qubits: u8) -> usize {
    if circuit.is_empty() {
        return 0;
    }

    // Simple estimate: track when each qubit was last used
    let mut qubit_depth = vec![0usize; n_qubits as usize];

    for gate in circuit {
        match gate {
            QGate::H(q)
            | QGate::X(q)
            | QGate::Y(q)
            | QGate::Z(q)
            | QGate::Phase(q, _)
            | QGate::Measure(q)
            | QGate::Rx(q, _)
            | QGate::Ry(q, _)
            | QGate::Rz(q, _)
            | QGate::U3(q, _, _, _)
            | QGate::T(q)
            | QGate::Tdg(q) => {
                if (*q as usize) < qubit_depth.len() {
                    qubit_depth[*q as usize] += 1;
                }
            }
            QGate::CNot(c, t)
            | QGate::CZ(c, t)
            | QGate::Swap(c, t)
            | QGate::CPhase(c, t, _)
            | QGate::CRz(c, t, _) => {
                let c_idx = *c as usize;
                let t_idx = *t as usize;
                if c_idx < qubit_depth.len() && t_idx < qubit_depth.len() {
                    let max_depth = qubit_depth[c_idx].max(qubit_depth[t_idx]) + 1;
                    qubit_depth[c_idx] = max_depth;
                    qubit_depth[t_idx] = max_depth;
                }
            }
            QGate::Toffoli(a, b, c) => {
                let a_idx = *a as usize;
                let b_idx = *b as usize;
                let c_idx = *c as usize;
                if a_idx < qubit_depth.len()
                    && b_idx < qubit_depth.len()
                    && c_idx < qubit_depth.len()
                {
                    let max_depth = qubit_depth[a_idx]
                        .max(qubit_depth[b_idx])
                        .max(qubit_depth[c_idx])
                        + 1;
                    qubit_depth[a_idx] = max_depth;
                    qubit_depth[b_idx] = max_depth;
                    qubit_depth[c_idx] = max_depth;
                }
            }
            QGate::CCZ(a, b, c) => {
                let a_idx = *a as usize;
                let b_idx = *b as usize;
                let c_idx = *c as usize;
                if a_idx < qubit_depth.len()
                    && b_idx < qubit_depth.len()
                    && c_idx < qubit_depth.len()
                {
                    let max_depth = qubit_depth[a_idx]
                        .max(qubit_depth[b_idx])
                        .max(qubit_depth[c_idx])
                        + 1;
                    qubit_depth[a_idx] = max_depth;
                    qubit_depth[b_idx] = max_depth;
                    qubit_depth[c_idx] = max_depth;
                }
            }
        }
    }

    qubit_depth.into_iter().max().unwrap_or(0)
}

/// Count gates per qubit.
fn qubit_utilization(circuit: &[QGate], n_qubits: u8) -> Vec<usize> {
    let mut util = vec![0usize; n_qubits as usize];

    for gate in circuit {
        let qubits = gate_qubits(gate);
        for q in qubits {
            if (q as usize) < util.len() {
                util[q as usize] += 1;
            }
        }
    }

    util
}

/// Get qubits used by a gate.
fn gate_qubits(gate: &QGate) -> Vec<u8> {
    match gate {
        QGate::H(q)
        | QGate::X(q)
        | QGate::Y(q)
        | QGate::Z(q)
        | QGate::Phase(q, _)
        | QGate::Measure(q)
        | QGate::Rx(q, _)
        | QGate::Ry(q, _)
        | QGate::Rz(q, _)
        | QGate::U3(q, _, _, _)
        | QGate::T(q)
        | QGate::Tdg(q) => {
            vec![*q]
        }
        QGate::CNot(a, b)
        | QGate::CZ(a, b)
        | QGate::Swap(a, b)
        | QGate::CPhase(a, b, _)
        | QGate::CRz(a, b, _) => vec![*a, *b],
        QGate::Toffoli(a, b, c) => vec![*a, *b, *c],
        QGate::CCZ(a, b, c) => vec![*a, *b, *c],
    }
}

/// Get human-readable gate type name.
fn gate_type_name(gate: &QGate) -> &'static str {
    match gate {
        QGate::H(_) => "H",
        QGate::X(_) => "X",
        QGate::Z(_) => "Z",
        QGate::Phase(_, _) => "Phase",
        QGate::Rx(_, _) => "Rx",
        QGate::Ry(_, _) => "Ry",
        QGate::Rz(_, _) => "Rz",
        QGate::U3(_, _, _, _) => "U3", // SPRINT 32.0
        QGate::T(_) => "T",            // SPRINT 48.0
        QGate::Tdg(_) => "Tdg",        // SPRINT 48.0
        QGate::Y(_) => "Y",
        QGate::Swap(_, _) => "Swap",
        QGate::CNot(_, _) => "CNOT",
        QGate::CZ(_, _) => "CZ",
        QGate::CPhase(_, _, _) => "CPhase",
        QGate::CRz(_, _, _) => "CRz",
        QGate::Toffoli(_, _, _) => "Toffoli",
        QGate::CCZ(_, _, _) => "CCZ",
        QGate::Measure(_) => "Measure",
    }
}

// =============================================================================
// Circuit Comparison
// =============================================================================

/// Compare two circuits structurally.
#[derive(Clone, Debug)]
pub struct CircuitComparison {
    pub length_diff: i32,
    pub entangling_diff: i32,
    pub quantum_ness_diff: f32,
    pub gate_overlap: f32,          // Jaccard similarity of gate types
    pub pattern_overlap: f32,       // Shared patterns
    pub structural_similarity: f32, // Overall similarity score
}

/// Compare two circuits.
pub fn compare_circuits(a: &[QGate], b: &[QGate], n_qubits: u8) -> CircuitComparison {
    let analysis_a = detect_patterns(a, n_qubits);
    let analysis_b = detect_patterns(b, n_qubits);

    let length_diff = b.len() as i32 - a.len() as i32;
    let entangling_diff = analysis_b.entangling_gates as i32 - analysis_a.entangling_gates as i32;
    let qness_diff = analysis_b.quantum_ness - analysis_a.quantum_ness;

    // Gate type overlap (Jaccard)
    let gate_overlap = jaccard_similarity(
        &analysis_a.gate_distribution.keys().cloned().collect(),
        &analysis_b.gate_distribution.keys().cloned().collect(),
    );

    // Pattern overlap
    let patterns_a: std::collections::HashSet<_> =
        analysis_a.patterns.iter().map(|(p, _)| p).collect();
    let patterns_b: std::collections::HashSet<_> =
        analysis_b.patterns.iter().map(|(p, _)| p).collect();
    let pattern_overlap = if patterns_a.is_empty() && patterns_b.is_empty() {
        1.0
    } else {
        let intersection = patterns_a.intersection(&patterns_b).count();
        let union = patterns_a.union(&patterns_b).count();
        intersection as f32 / union.max(1) as f32
    };

    // Overall similarity
    let len_sim = 1.0 - (length_diff.abs() as f32 / (a.len().max(b.len()).max(1) as f32));
    let structural_similarity = (len_sim + gate_overlap + pattern_overlap) / 3.0;

    CircuitComparison {
        length_diff,
        entangling_diff,
        quantum_ness_diff: qness_diff,
        gate_overlap,
        pattern_overlap,
        structural_similarity,
    }
}

fn jaccard_similarity(a: &Vec<String>, b: &Vec<String>) -> f32 {
    let set_a: std::collections::HashSet<_> = a.iter().collect();
    let set_b: std::collections::HashSet<_> = b.iter().collect();

    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    intersection as f32 / union.max(1) as f32
}

// =============================================================================
// Population-Level Analysis
// =============================================================================

/// Analyze patterns across a population of circuits.
#[derive(Clone, Debug)]
pub struct PopulationPatternAnalysis {
    pub sample_size: usize,
    pub pattern_frequencies: HashMap<QuantumPattern, usize>,
    pub avg_quantum_ness: f32,
    pub avg_circuit_length: f32,
    pub avg_entangling_gates: f32,
    pub avg_interference_score: f32,
    pub avg_randomness_score: f32,
    pub dominant_pattern: Option<QuantumPattern>,
}

/// Analyze patterns across multiple circuits.
pub fn analyze_population_patterns(
    circuits: &[Vec<QGate>],
    n_qubits: u8,
) -> PopulationPatternAnalysis {
    if circuits.is_empty() {
        return PopulationPatternAnalysis {
            sample_size: 0,
            pattern_frequencies: HashMap::new(),
            avg_quantum_ness: 0.0,
            avg_circuit_length: 0.0,
            avg_entangling_gates: 0.0,
            avg_interference_score: 0.0,
            avg_randomness_score: 0.0,
            dominant_pattern: None,
        };
    }

    let mut pattern_freq: HashMap<QuantumPattern, usize> = HashMap::new();
    let mut total_qness = 0.0f32;
    let mut total_len = 0.0f32;
    let mut total_entangling = 0.0f32;
    let mut total_interference = 0.0f32;
    let mut total_randomness = 0.0f32;

    for circuit in circuits {
        let analysis = detect_patterns(circuit, n_qubits);

        for (pattern, count) in &analysis.patterns {
            *pattern_freq.entry(*pattern).or_insert(0) += count;
        }

        total_qness += analysis.quantum_ness;
        total_len += analysis.total_gates as f32;
        total_entangling += analysis.entangling_gates as f32;
        total_interference += analysis.interference_score;
        total_randomness += analysis.randomness_score;
    }

    let n = circuits.len() as f32;

    let dominant = pattern_freq
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(pattern, _)| *pattern);

    PopulationPatternAnalysis {
        sample_size: circuits.len(),
        pattern_frequencies: pattern_freq,
        avg_quantum_ness: total_qness / n,
        avg_circuit_length: total_len / n,
        avg_entangling_gates: total_entangling / n,
        avg_interference_score: total_interference / n,
        avg_randomness_score: total_randomness / n,
        dominant_pattern: dominant,
    }
}

// =============================================================================
// Circuit Serialization (for export/comparison)
// =============================================================================

/// Convert circuit to a canonical string representation.
pub fn circuit_to_string(circuit: &[QGate]) -> String {
    circuit
        .iter()
        .map(gate_to_string)
        .collect::<Vec<_>>()
        .join("-")
}

fn gate_to_string(gate: &QGate) -> String {
    match gate {
        QGate::H(q) => format!("H{}", q),
        QGate::X(q) => format!("X{}", q),
        QGate::Z(q) => format!("Z{}", q),
        QGate::Phase(q, angle) => format!("P{}:{:.2}", q, angle),
        QGate::Rx(q, angle) => format!("Rx{}:{:.2}", q, angle),
        QGate::Ry(q, angle) => format!("Ry{}:{:.2}", q, angle),
        QGate::Rz(q, angle) => format!("Rz{}:{:.2}", q, angle),
        QGate::U3(q, theta, phi, lambda) => {
            format!("U3{}:{:.2},{:.2},{:.2}", q, theta, phi, lambda)
        } // SPRINT 32.0
        QGate::T(q) => format!("T{}", q),     // SPRINT 48.0
        QGate::Tdg(q) => format!("Tdg{}", q), // SPRINT 48.0
        QGate::Y(q) => format!("Y{}", q),
        QGate::Swap(a, b) => format!("Swap{}{}", a, b),
        QGate::CNot(c, t) => format!("CX{}{}", c, t),
        QGate::CZ(a, b) => format!("CZ{}{}", a, b),
        QGate::CPhase(a, b, angle) => format!("CP{}{}:{:.2}", a, b, angle),
        QGate::CRz(a, b, angle) => format!("CRz{}{}:{:.2}", a, b, angle),
        QGate::Toffoli(a, b, c) => format!("CCX{}{}{}", a, b, c),
        QGate::CCZ(a, b, c) => format!("CCZ{}{}{}", a, b, c),
        QGate::Measure(q) => format!("M{}", q),
    }
}

/// Print a human-readable circuit diagram.
pub fn circuit_diagram(circuit: &[QGate], n_qubits: u8) -> String {
    let mut lines = vec![String::new(); n_qubits as usize];

    for gate in circuit {
        let width = gate_diagram_width(gate);
        let diagram = gate_to_diagram(gate);

        // Pad all lines to same length
        let max_len = lines.iter().map(|l| l.len()).max().unwrap_or(0);
        for line in &mut lines {
            while line.len() < max_len {
                line.push('─');
            }
        }

        // Add gate
        match gate {
            QGate::H(q) | QGate::X(q) | QGate::Z(q) | QGate::Phase(q, _) | QGate::Measure(q) => {
                if (*q as usize) < lines.len() {
                    lines[*q as usize].push_str(&diagram);
                    for i in 0..n_qubits {
                        if i != *q {
                            lines[i as usize].push_str(&"─".repeat(width));
                        }
                    }
                }
            }
            QGate::CNot(c, t) => {
                if (*c as usize) < lines.len() && (*t as usize) < lines.len() {
                    lines[*c as usize].push('●');
                    lines[*t as usize].push('⊕');
                    for i in 0..n_qubits {
                        if i != *c && i != *t {
                            if (i > *c && i < *t) || (i < *c && i > *t) {
                                lines[i as usize].push('│');
                            } else {
                                lines[i as usize].push('─');
                            }
                        }
                    }
                }
            }
            _ => {
                // Simplified for other gates
                for i in 0..n_qubits {
                    lines[i as usize].push('─');
                }
            }
        }
    }

    // Format with qubit labels
    let mut result = String::new();
    for (i, line) in lines.iter().enumerate() {
        result.push_str(&format!("q{}: {}\n", i, line));
    }
    result
}

fn gate_diagram_width(gate: &QGate) -> usize {
    match gate {
        QGate::H(_) => 3,
        QGate::X(_) => 3,
        QGate::Z(_) => 3,
        QGate::Phase(_, _) => 3,
        QGate::Rx(_, _) | QGate::Ry(_, _) | QGate::Rz(_, _) => 4,
        QGate::U3(_, _, _, _) => 5, // SPRINT 32.0: U3 has 3 params, wider display
        QGate::T(_) | QGate::Tdg(_) => 3, // SPRINT 48.0
        QGate::Y(_) => 3,
        QGate::Swap(_, _) => 1,
        QGate::CNot(_, _) => 1,
        QGate::CZ(_, _) => 1,
        QGate::CPhase(_, _, _) => 1,
        QGate::CRz(_, _, _) => 1,
        QGate::Toffoli(_, _, _) => 1,
        QGate::CCZ(_, _, _) => 1,
        QGate::Measure(_) => 3,
    }
}

fn gate_to_diagram(gate: &QGate) -> String {
    match gate {
        QGate::H(_) => "[H]".to_string(),
        QGate::X(_) => "[X]".to_string(),
        QGate::Z(_) => "[Z]".to_string(),
        QGate::Phase(_, _) => "[P]".to_string(),
        QGate::Rx(_, _) => "[Rx]".to_string(),
        QGate::Ry(_, _) => "[Ry]".to_string(),
        QGate::Rz(_, _) => "[Rz]".to_string(),
        QGate::Y(_) => "[Y]".to_string(),
        QGate::Measure(_) => "[M]".to_string(),
        _ => "─".to_string(),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_bell_pair() {
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let analysis = detect_patterns(&circuit, 4);

        assert!(
            analysis
                .patterns
                .iter()
                .any(|(p, _)| *p == QuantumPattern::BellPair)
        );
    }

    #[test]
    fn test_detect_hadamard_sandwich() {
        let circuit = vec![QGate::H(0), QGate::X(0), QGate::Z(0), QGate::H(0)];
        let analysis = detect_patterns(&circuit, 4);

        assert!(
            analysis
                .patterns
                .iter()
                .any(|(p, _)| *p == QuantumPattern::HadamardSandwich)
        );
    }

    #[test]
    fn test_detect_uniform_superposition() {
        let circuit = vec![
            QGate::H(0),
            QGate::H(1),
            QGate::H(2),
            QGate::H(3),
            QGate::CNot(0, 1),
        ];
        let analysis = detect_patterns(&circuit, 4);

        assert!(
            analysis
                .patterns
                .iter()
                .any(|(p, _)| *p == QuantumPattern::UniformSuperposition)
        );
    }

    #[test]
    fn test_detect_cnot_cascade() {
        let circuit = vec![QGate::CNot(0, 1), QGate::CNot(1, 2), QGate::CNot(2, 3)];
        let analysis = detect_patterns(&circuit, 4);

        assert!(
            analysis
                .patterns
                .iter()
                .any(|(p, _)| *p == QuantumPattern::CNOTCascade)
        );
    }

    #[test]
    fn test_circuit_comparison() {
        let a = vec![QGate::H(0), QGate::CNot(0, 1)];
        let b = vec![QGate::H(0), QGate::CNot(0, 1), QGate::H(1)];

        let comparison = compare_circuits(&a, &b, 4);

        assert_eq!(comparison.length_diff, 1);
        assert!(comparison.structural_similarity > 0.5);
    }

    #[test]
    fn test_circuit_to_string() {
        let circuit = vec![QGate::H(0), QGate::CNot(0, 1)];
        let s = circuit_to_string(&circuit);

        assert_eq!(s, "H0-CX01");
    }

    #[test]
    fn test_population_analysis() {
        let circuits = vec![
            vec![QGate::H(0), QGate::CNot(0, 1)],
            vec![QGate::H(0), QGate::H(1), QGate::CNot(0, 1)],
            vec![QGate::X(0), QGate::Z(1)],
        ];

        let analysis = analyze_population_patterns(&circuits, 4);

        assert_eq!(analysis.sample_size, 3);
        assert!(analysis.avg_quantum_ness > 0.0);
    }

    #[test]
    fn test_estimate_depth() {
        // Parallel gates: depth 1
        let parallel = vec![QGate::H(0), QGate::H(1), QGate::H(2)];
        assert_eq!(estimate_depth(&parallel, 4), 1);

        // Sequential on same qubit: depth 3
        let sequential = vec![QGate::H(0), QGate::X(0), QGate::Z(0)];
        assert_eq!(estimate_depth(&sequential, 4), 3);
    }
}
