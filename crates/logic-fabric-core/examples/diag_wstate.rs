//! Localize the ultimate_optimize correctness bug on wstate_n3 by running each
//! public optimization phase in isolation and checking state-equivalence.

use logic_fabric_core::algebraic_fusion as af;
use logic_fabric_core::commutation;
use logic_fabric_core::qasm::parse_qasm;
use logic_fabric_core::quantum::{apply_gate_scalar, Complex32, QGate, QRng, QState};
use std::fs;

fn simulate_from(gates: &[QGate], n: u8, basis: usize) -> Vec<Complex32> {
    let mut s = QState::new_zero(n);
    let mut rng = QRng::new(1);
    // Prepare basis state |basis> by flipping the set bits.
    for q in 0..n {
        if basis & (1 << q) != 0 {
            apply_gate_scalar(&mut s, &QGate::X(q), &mut rng);
        }
    }
    for g in gates {
        apply_gate_scalar(&mut s, g, &mut rng);
    }
    s.to_interleaved()
}

fn simulate(gates: &[QGate], n: u8) -> Vec<Complex32> {
    simulate_from(gates, n, 0)
}

fn fid(a: &[Complex32], b: &[Complex32]) -> f32 {
    let (mut re, mut im) = (0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        re += x.re * y.re + x.im * y.im;
        im += x.re * y.im - x.im * y.re;
    }
    re * re + im * im
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: diag_wstate <qasm>");
    let src = fs::read_to_string(&path).unwrap();
    let (gates, nq) = parse_qasm(&src).unwrap();
    let n = nq as u8;
    println!("input: {} gates on {} qubits", gates.len(), nq);

    // True unitary equivalence up to ONE global phase: build both unitaries
    // column-by-column (action on each basis state), fix a single global phase
    // from the largest amplitude of column 0, then require ALL columns match.
    let cols: Vec<Vec<Complex32>> = (0..(1usize << n))
        .map(|k| simulate_from(&gates, n, k))
        .collect();
    let check = |label: &str, out: &[QGate]| {
        let ocols: Vec<Vec<Complex32>> = (0..(1usize << n))
            .map(|k| simulate_from(out, n, k))
            .collect();
        // Find the largest-magnitude entry in original column 0 to fix phase.
        let mut bi = 0usize;
        let mut best = 0.0f32;
        for (idx, a) in cols[0].iter().enumerate() {
            if a.norm2() > best {
                best = a.norm2();
                bi = idx;
            }
        }
        // global phase factor p such that ocols[0][bi] = p * cols[0][bi]
        let r = cols[0][bi];
        let o = ocols[0][bi];
        let denom = r.norm2().max(1e-12);
        let pre = (o.re * r.re + o.im * r.im) / denom;
        let pim = (o.im * r.re - o.re * r.im) / denom;
        let mut max_err = 0.0f32;
        for k in 0..cols.len() {
            for a in 0..cols[k].len() {
                let r = cols[k][a];
                // expected = p * r
                let er = pre * r.re - pim * r.im;
                let ei = pre * r.im + pim * r.re;
                let g = ocols[k][a];
                let e = ((g.re - er).powi(2) + (g.im - ei).powi(2)).sqrt();
                if e > max_err {
                    max_err = e;
                }
            }
        }
        let mark = if max_err < 1e-3 { "OK  " } else { "FAIL" };
        println!(
            "  {mark} max_err={max_err:.4}  {:>4} gates  {label}",
            out.len()
        );
    };
    let _ = fid; // keep helper

    check(
        "optimize_rotation_chain",
        &af::optimize_rotation_chain(&gates),
    );
    check("reorder_for_fusion", &af::reorder_for_fusion(gates.clone()));
    check(
        "fuse_general_single_qubit",
        &af::fuse_general_single_qubit(&gates),
    );
    check(
        "cancel_inverse_fixpoint",
        &af::cancel_inverse_fixpoint(gates.clone()),
    );
    check(
        "comprehensive_commutation_optimize",
        &commutation::comprehensive_commutation_optimize(gates.clone()),
    );
    check(
        "optimize_pauli_fixpoint",
        &af::optimize_pauli_fixpoint(gates.clone()),
    );
    check(
        "optimize_two_qubit_blocks",
        &af::optimize_two_qubit_blocks(&gates),
    );
    check(
        "commute_for_fusion_fixpoint",
        &commutation::commute_for_fusion_fixpoint(gates.clone()),
    );
    check(
        "commutation_cancellation_fixpoint",
        &commutation::commutation_cancellation_fixpoint(gates.clone()),
    );

    // Dump the exact sequences around the breaking transition.
    let dump = |label: &str, gs: &[QGate]| {
        println!("--- {label} ({} gates) ---", gs.len());
        for (idx, g) in gs.iter().enumerate() {
            println!("  {idx:>2}: {g:?}");
        }
    };
    dump("ORIGINAL", &gates);
    let reordered = commutation::commute_for_fusion_fixpoint(gates.clone());
    dump("AFTER commute_for_fusion", &reordered);
    let fused = af::fuse_general_single_qubit(&reordered);
    dump("AFTER fuse_general (BROKEN)", &fused);

    // Replicate the comprehensive_commutation_optimize loop step-by-step.
    println!("--- step-by-step comprehensive loop ---");
    let mut cur = gates.clone();
    let mut prev_len = usize::MAX;
    let mut it = 0;
    while cur.len() < prev_len && it < 20 {
        prev_len = cur.len();
        it += 1;
        cur = commutation::commute_for_fusion_fixpoint(cur);
        check(&format!("[{it}] after commute_for_fusion"), &cur);
        cur = af::fuse_general_single_qubit(&cur);
        check(&format!("[{it}] after fuse_general"), &cur);
        cur = commutation::commutation_cancellation_fixpoint(cur);
        check(&format!("[{it}] after commutation_cancel"), &cur);
        cur = af::cancel_inverse_fixpoint(cur);
        check(&format!("[{it}] after cancel_inverse"), &cur);
    }

    // Full pipeline for reference.
    let (ops, _) = af::ultimate_optimize(gates.clone(), n);
    let mut full = Vec::new();
    for op in ops {
        match op {
            af::AlgebraicOp::Single(g) => full.push(g),
            af::AlgebraicOp::Power { gate, power } => {
                for _ in 0..power {
                    full.push(gate.clone());
                }
            }
            _ => {}
        }
    }
    check("ultimate_optimize (full)", &full);
}
