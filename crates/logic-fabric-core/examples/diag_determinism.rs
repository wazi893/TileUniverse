//! Characterize ultimate_optimize nondeterminism: run the same circuit many
//! times in ONE process. If counts vary within a process -> algorithmic
//! ordering (e.g. HashMap iteration). If stable here but differs across
//! processes -> per-process HashMap seed or global cache state.

use logic_fabric_core::algebraic_fusion::{ultimate_optimize, AlgebraicOp};
use logic_fabric_core::qasm::parse_qasm;
use std::fs;

fn count(ops: &[AlgebraicOp]) -> usize {
    ops.iter()
        .map(|op| match op {
            AlgebraicOp::Single(_) => 1,
            AlgebraicOp::Power { power, .. } => *power as usize,
            AlgebraicOp::Skip { .. } => 0,
            _ => 0,
        })
        .sum()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: diag_determinism <qasm>");
    let src = fs::read_to_string(&path).unwrap();
    let (gates, nq) = parse_qasm(&src).unwrap();
    println!("input {} gates, {} qubits", gates.len(), nq);
    println!("-- without clearing cache --");
    for run in 0..8 {
        let (ops, _) = ultimate_optimize(gates.clone(), nq as u8);
        println!("  run {run}: {} gates", count(&ops));
    }
    println!("-- clearing mega cache before each run --");
    for run in 0..8 {
        logic_fabric_core::algebraic_fusion::mega_cache_clear();
        let (ops, _) = ultimate_optimize(gates.clone(), nq as u8);
        println!("  run {run}: {} gates", count(&ops));
    }
}
