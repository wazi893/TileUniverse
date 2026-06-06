//! Verified gate-count benchmark for the algebraic fusion engine.
//!
//! For each input QASM file: parse -> count -> `ultimate_optimize` -> count.
//! For circuits small enough to simulate (<= MAX_VERIFY_QUBITS), it also
//! verifies the optimized circuit is *state-equivalent* to the original by
//! computing |<psi_orig | psi_opt>|^2 and asserting it is ~1. This makes the
//! reductions trustworthy: smaller AND provably the same circuit.
//!
//! Usage: cargo run --release --example bench_fusion -- <file_or_dir> [more...]

use logic_fabric_core::algebraic_fusion::{ultimate_optimize, AlgebraicOp};
use logic_fabric_core::qasm::parse_qasm;
use logic_fabric_core::quantum::{apply_gate_scalar, Complex32, QGate, QRng, QState};
use std::fs;
use std::path::{Path, PathBuf};

const MAX_VERIFY_QUBITS: u8 = 16;
/// Number of random input states used per circuit for equivalence testing.
/// Each random state has full support, so a single one already exposes any
/// relative-phase error with probability 1; a handful makes false positives
/// astronomically unlikely.
const RANDOM_TRIALS: usize = 5;

fn ops_to_gates(ops: Vec<AlgebraicOp>) -> Vec<QGate> {
    let mut out = Vec::new();
    for op in ops {
        match op {
            AlgebraicOp::Single(g) => out.push(g),
            AlgebraicOp::Power { gate, power } => {
                for _ in 0..power {
                    out.push(gate.clone());
                }
            }
            AlgebraicOp::Skip { .. } => {}
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }
    out
}

/// A simple seeded xorshift used to build reproducible random input states.
fn rand_unit(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // Map to (-1, 1).
        (s as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
    }
}

/// Build a reproducible random normalized statevector of `2^n` amplitudes.
fn random_state(n_qubits: u8, seed: u64) -> Vec<Complex32> {
    let dim = 1usize << n_qubits;
    let mut next = rand_unit(seed);
    let mut amps: Vec<Complex32> = (0..dim).map(|_| Complex32::new(next(), next())).collect();
    let norm: f32 = amps.iter().map(|c| c.norm2()).sum::<f32>().sqrt();
    if norm > 0.0 {
        for a in &mut amps {
            a.re /= norm;
            a.im /= norm;
        }
    }
    amps
}

/// Apply a unitary circuit (no measurements) to an arbitrary input state.
fn apply_circuit(gates: &[QGate], n_qubits: u8, input: &[Complex32]) -> Vec<Complex32> {
    let mut state = QState::from_interleaved(n_qubits, input);
    let mut rng = QRng::new(1);
    for g in gates {
        apply_gate_scalar(&mut state, g, &mut rng);
    }
    state.to_interleaved()
}

/// Fidelity |<a|b>|^2 between two statevectors (phase-invariant).
fn fidelity(a: &[Complex32], b: &[Complex32]) -> f32 {
    let mut re = 0.0f32;
    let mut im = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        // <a|b> = conj(a) * b
        re += x.re * y.re + x.im * y.im;
        im += x.re * y.im - x.im * y.re;
    }
    re * re + im * im
}

/// True up-to-global-phase equivalence: apply both circuits to several random
/// input states and require |<orig|opt>|^2 ~ 1 on every one. Returns the worst
/// fidelity observed.
fn worst_equivalence(orig: &[QGate], opt: &[QGate], n_qubits: u8) -> f32 {
    let mut worst = 1.0f32;
    for t in 0..RANDOM_TRIALS {
        let input = random_state(n_qubits, 0x9E3779B97F4A7C15u64.wrapping_mul(t as u64 + 1));
        let a = apply_circuit(orig, n_qubits, &input);
        let b = apply_circuit(opt, n_qubits, &input);
        let f = fidelity(&a, &b);
        if f < worst {
            worst = f;
        }
    }
    worst
}

fn collect_qasm(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        if let Ok(entries) = fs::read_dir(path) {
            let mut v: Vec<_> = entries.flatten().map(|e| e.path()).collect();
            v.sort();
            for p in v {
                collect_qasm(&p, out);
            }
        }
    } else if path.extension().and_then(|s| s.to_str()) == Some("qasm") {
        out.push(path.to_path_buf());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: bench_fusion <file_or_dir> [more...]");
        std::process::exit(1);
    }

    let mut files = Vec::new();
    for a in &args {
        collect_qasm(Path::new(a), &mut files);
    }

    println!(
        "{:<28} {:>6} {:>6} {:>8} {:>10}",
        "circuit", "orig", "opt", "reduct", "verified"
    );
    println!("{}", "-".repeat(64));

    let mut total_orig = 0usize;
    let mut total_opt = 0usize;
    let mut verified_ok = 0usize;
    let mut verified_total = 0usize;
    let mut parse_err = 0usize;

    for f in &files {
        let name = f
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .replace("_transpiled", "");
        let src = match fs::read_to_string(f) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let (gates, n_qubits) = match parse_qasm(&src) {
            Ok(r) => r,
            Err(_) => {
                parse_err += 1;
                println!(
                    "{:<28} {:>6} {:>6} {:>8} {:>10}",
                    name, "-", "-", "PARSE", "-"
                );
                continue;
            }
        };
        if n_qubits == 0 || n_qubits > 255 {
            continue;
        }
        let orig_n = gates.len();
        let (ops, _stats) = ultimate_optimize(gates.clone(), n_qubits as u8);
        let opt_gates = ops_to_gates(ops);
        let opt_n = opt_gates.len();

        let reduct = if orig_n > 0 {
            100.0 * (1.0 - opt_n as f32 / orig_n as f32)
        } else {
            0.0
        };

        let verified = if (n_qubits as u8) <= MAX_VERIFY_QUBITS {
            verified_total += 1;
            let worst = worst_equivalence(&gates, &opt_gates, n_qubits as u8);
            if worst > 0.999 {
                verified_ok += 1;
                "OK".to_string()
            } else {
                format!("FAIL {:.3}", worst)
            }
        } else {
            "skip(>16q)".to_string()
        };

        total_orig += orig_n;
        total_opt += opt_n;
        println!(
            "{:<28} {:>6} {:>6} {:>7.1}% {:>10}",
            name, orig_n, opt_n, reduct, verified
        );
    }

    println!("{}", "-".repeat(64));
    let overall = if total_orig > 0 {
        100.0 * (1.0 - total_opt as f32 / total_orig as f32)
    } else {
        0.0
    };
    println!(
        "TOTAL gates {} -> {} ({:.1}% reduction) | verified {}/{} equivalent | {} parse-err",
        total_orig, total_opt, overall, verified_ok, verified_total, parse_err
    );
}
