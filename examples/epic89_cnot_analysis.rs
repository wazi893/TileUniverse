//! EPIC 89 Phase 89.0.5: CNOT Pattern Analysis
//!
//! Analyzes CNOT distribution in VQE/QAOA circuits to understand:
//! 1. How many CNOTs can be eliminated (adjacent pairs)
//! 2. How many CNOTs can be batched (same qubit sequences)
//! 3. Qubit pair distribution

use engine::algebraic_fusion::{generate_qaoa_pattern, generate_vqe_pattern};
use engine::quantum::QGate;
use std::collections::HashMap;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              EPIC 89 Phase 89.0.5: CNOT Pattern Analysis                    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // Analyze VQE circuits
    println!("## VQE Circuit Analysis\n");
    println!("| Qubits | Layers | Total Gates | CNOTs | Adjacent Pairs | Qubit Dist |");
    println!("|--------|--------|-------------|-------|----------------|------------|");

    for &(n_qubits, layers) in &[(4, 10), (8, 20), (12, 30)] {
        let circuit = generate_vqe_pattern(n_qubits, layers);
        let analysis = analyze_cnot_patterns(&circuit);

        let dist_str = analysis
            .qubit_pairs
            .iter()
            .take(3)
            .map(|((c, t), cnt)| format!("({},{}):{}", c, t, cnt))
            .collect::<Vec<_>>()
            .join(" ");

        println!(
            "| {} | {} | {} | {} | {} | {} |",
            n_qubits,
            layers,
            circuit.len(),
            analysis.total_cnots,
            analysis.adjacent_eliminable,
            dist_str
        );
    }

    // Analyze QAOA circuits
    println!("\n## QAOA Circuit Analysis\n");
    println!("| Qubits | p | Total Gates | CNOTs | Adjacent Pairs | Batchable Seq |");
    println!("|--------|---|-------------|-------|----------------|---------------|");

    for &(n_qubits, p) in &[(4, 10), (8, 20), (12, 30)] {
        let circuit = generate_qaoa_pattern(n_qubits, p);
        let analysis = analyze_cnot_patterns(&circuit);

        println!(
            "| {} | {} | {} | {} | {} | {} |",
            n_qubits,
            p,
            circuit.len(),
            analysis.total_cnots,
            analysis.adjacent_eliminable,
            analysis.batchable_sequences
        );
    }

    // Detailed analysis for VQE 8q 20L (target circuit)
    println!("\n## Detailed Analysis: VQE 8q 20L\n");
    let vqe_8q = generate_vqe_pattern(8, 20);
    let analysis = analyze_cnot_patterns(&vqe_8q);

    println!("Total gates: {}", vqe_8q.len());
    println!("Total CNOTs: {}", analysis.total_cnots);
    println!(
        "Single-qubit gates: {}",
        vqe_8q.len() - analysis.total_cnots
    );
    println!(
        "CNOT percentage: {:.1}%",
        100.0 * analysis.total_cnots as f64 / vqe_8q.len() as f64
    );
    println!();

    println!("### Elimination Potential");
    println!(
        "- Adjacent CNOT pairs (eliminable): {}",
        analysis.adjacent_eliminable
    );
    println!(
        "- Elimination rate: {:.1}%",
        100.0 * analysis.adjacent_eliminable as f64 / analysis.total_cnots as f64
    );
    println!();

    println!("### Batching Potential");
    println!(
        "- Same-qubit sequences (batchable): {}",
        analysis.batchable_sequences
    );
    println!(
        "- Average sequence length: {:.1}",
        analysis.avg_sequence_length
    );
    println!("- Max sequence length: {}", analysis.max_sequence_length);
    println!();

    println!("### Qubit Pair Distribution");
    println!("| Control | Target | Count | Percentage |");
    println!("|---------|--------|-------|------------|");
    for ((ctrl, tgt), count) in &analysis.qubit_pairs {
        let pct = 100.0 * *count as f64 / analysis.total_cnots as f64;
        println!("| {} | {} | {} | {:.1}% |", ctrl, tgt, count, pct);
    }

    // Check for non-consecutive elimination opportunities
    println!("\n### Non-Consecutive Elimination Opportunities");
    let non_consec = find_non_consecutive_pairs(&vqe_8q);
    println!(
        "- CNOT pairs separated by single-qubit gates: {}",
        non_consec
    );
    println!(
        "- Additional elimination potential: {:.1}%",
        100.0 * non_consec as f64 / analysis.total_cnots as f64
    );

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                          Analysis Complete                                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}

struct CnotAnalysis {
    total_cnots: usize,
    adjacent_eliminable: usize,
    batchable_sequences: usize,
    avg_sequence_length: f64,
    max_sequence_length: usize,
    qubit_pairs: Vec<((u8, u8), usize)>,
}

fn analyze_cnot_patterns(circuit: &[QGate]) -> CnotAnalysis {
    let mut total_cnots = 0;
    let mut adjacent_eliminable = 0;
    let mut qubit_counts: HashMap<(u8, u8), usize> = HashMap::new();

    // Count CNOTs and check for adjacent pairs
    let mut prev_cnot: Option<(u8, u8)> = None;
    for gate in circuit {
        if let QGate::CNot(ctrl, tgt) = gate {
            total_cnots += 1;
            *qubit_counts.entry((*ctrl, *tgt)).or_insert(0) += 1;

            if prev_cnot == Some((*ctrl, *tgt)) {
                adjacent_eliminable += 2; // Both gates eliminated
                prev_cnot = None; // Reset after elimination
            } else {
                prev_cnot = Some((*ctrl, *tgt));
            }
        } else {
            prev_cnot = None; // Non-CNOT breaks the sequence
        }
    }

    // Calculate batchable sequences (consecutive CNOTs on same qubit pair)
    let mut sequences: Vec<usize> = Vec::new();
    let mut current_seq_len = 0;
    let mut current_pair: Option<(u8, u8)> = None;

    for gate in circuit {
        if let QGate::CNot(ctrl, tgt) = gate {
            if current_pair == Some((*ctrl, *tgt)) {
                current_seq_len += 1;
            } else {
                if current_seq_len > 0 {
                    sequences.push(current_seq_len);
                }
                current_pair = Some((*ctrl, *tgt));
                current_seq_len = 1;
            }
        } else {
            if current_seq_len > 0 {
                sequences.push(current_seq_len);
                current_seq_len = 0;
            }
            current_pair = None;
        }
    }
    if current_seq_len > 0 {
        sequences.push(current_seq_len);
    }

    let batchable_sequences = sequences.iter().filter(|&&len| len >= 2).count();
    let avg_sequence_length = if sequences.is_empty() {
        0.0
    } else {
        sequences.iter().sum::<usize>() as f64 / sequences.len() as f64
    };
    let max_sequence_length = sequences.iter().copied().max().unwrap_or(0);

    // Sort qubit pairs by count
    let mut qubit_pairs: Vec<_> = qubit_counts.into_iter().collect();
    qubit_pairs.sort_by(|a, b| b.1.cmp(&a.1));

    CnotAnalysis {
        total_cnots,
        adjacent_eliminable,
        batchable_sequences,
        avg_sequence_length,
        max_sequence_length,
        qubit_pairs,
    }
}

/// Find CNOT pairs that are separated only by single-qubit gates on different qubits
fn find_non_consecutive_pairs(circuit: &[QGate]) -> usize {
    let mut count = 0;
    let mut i = 0;

    while i < circuit.len() {
        if let QGate::CNot(c1, t1) = &circuit[i] {
            // Look ahead for matching CNOT separated only by non-interfering gates
            let mut j = i + 1;
            let mut all_non_interfering = true;

            while j < circuit.len() && j < i + 10 {
                // Look ahead max 10 gates
                match &circuit[j] {
                    QGate::CNot(c2, t2) => {
                        if c1 == c2 && t1 == t2 && all_non_interfering {
                            // Found matching CNOT with only non-interfering gates between
                            count += 1;
                        }
                        break; // Stop at any CNOT (matching or not)
                    }
                    // Single-qubit gates are non-interfering if they don't touch ctrl or tgt
                    QGate::H(q)
                    | QGate::X(q)
                    | QGate::Z(q)
                    | QGate::Phase(q, _)
                    | QGate::Rx(q, _)
                    | QGate::Ry(q, _)
                    | QGate::Rz(q, _) => {
                        if q == c1 || q == t1 {
                            all_non_interfering = false;
                        }
                    }
                    _ => {
                        all_non_interfering = false;
                    }
                }
                j += 1;
            }
        }
        i += 1;
    }

    count
}
