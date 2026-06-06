//! Reproducible claim checks for the public launch post.
//!
//! This example is intentionally boring: it asserts the structural claims and
//! prints machine-local timings. Runtime numbers are hardware-dependent; the
//! checked claims are fidelity, fixed GHZ memory, and linear W-state memory.

use std::env;
use std::time::Instant;

use tileuniverse_quantum::*;

const DEFAULT_W_QUBITS: usize = 1_000_000;

fn main() {
    let options = Options::from_args();

    println!("tileuniverse-quantum launch claim check");
    println!(
        "release build recommended: cargo run --release -p tileuniverse-quantum --example launch_claims"
    );
    println!();

    check_minimal_ghz();
    check_unlimited_and_symbolic_ghz();
    check_symbolic_w();

    if let Some(w_qubits) = options.w_qubits {
        check_w_state(w_qubits);
    } else {
        println!("W-state materialized check skipped.");
    }
}

fn check_minimal_ghz() {
    println!("GHZ fast path");
    println!(
        "{:>18} {:>12} {:>12} {:>12} {:>10}",
        "qubits", "memory_B", "create_us", "verify_us", "fidelity"
    );

    let baseline_memory = MinimalGhzState::new(5).memory_bytes();

    for n in [5usize, 1_000_000_000usize] {
        let start_create = Instant::now();
        let ghz = MinimalGhzState::new(n);
        let create_us = start_create.elapsed().as_secs_f64() * 1_000_000.0;

        let start_verify = Instant::now();
        let verification = ghz.verify();
        let verify_us = start_verify.elapsed().as_secs_f64() * 1_000_000.0;

        assert_close(verification.fidelity, 1.0, 1e-10);
        assert_close(
            verification.amp_zero_state,
            std::f64::consts::FRAC_1_SQRT_2,
            1e-10,
        );
        assert_close(
            verification.amp_one_state,
            std::f64::consts::FRAC_1_SQRT_2,
            1e-10,
        );
        assert_eq!(verification.max_spurious, 0.0);
        assert_eq!(ghz.memory_bytes(), baseline_memory);

        println!(
            "{:>18} {:>12} {:>12.3} {:>12.3} {:>10.6}",
            n,
            ghz.memory_bytes(),
            create_us,
            verify_us,
            verification.fidelity
        );
    }

    println!();
}

fn check_unlimited_and_symbolic_ghz() {
    println!("BigUint and symbolic GHZ");
    println!(
        "{:>18} {:>12} {:>12} {:>12} {:>14}",
        "label", "memory_B", "create_us", "verify_us", "size_class"
    );

    let googol = num_bigint::BigUint::from(10u32).pow(100);
    let start_create = Instant::now();
    let googol_ghz = UnlimitedGhzState::new(googol);
    let create_us = start_create.elapsed().as_secs_f64() * 1_000_000.0;
    let start_verify = Instant::now();
    let googol_v = googol_ghz.verify();
    let verify_us = start_verify.elapsed().as_secs_f64() * 1_000_000.0;
    assert_close(googol_v.fidelity, 1.0, 1e-10);
    assert_eq!(googol_v.max_spurious, 0.0);
    println!(
        "{:>18} {:>12} {:>12.3} {:>12.3} {:>14}",
        "10^100",
        googol_ghz.memory_bytes(),
        create_us,
        verify_us,
        "BigUint"
    );

    let start_create = Instant::now();
    let graham_ghz = SymbolicGhzState::graham();
    let create_us = start_create.elapsed().as_secs_f64() * 1_000_000.0;
    let start_verify = Instant::now();
    let graham_v = graham_ghz.verify();
    let verify_us = start_verify.elapsed().as_secs_f64() * 1_000_000.0;
    assert_close(graham_v.fidelity, 1.0, 1e-10);
    assert_eq!(graham_v.max_spurious, 0.0);
    assert_eq!(graham_v.size_class, "Graham-class");
    println!(
        "{:>18} {:>12} {:>12.3} {:>12.3} {:>14}",
        "Graham",
        graham_ghz.memory_bytes(),
        create_us,
        verify_us,
        graham_v.size_class
    );

    println!();
}

fn check_symbolic_w() {
    let w = create_graham_w();
    let verification = w.verify();

    assert!(verification.is_valid);
    assert!(verification.total_probability.is_one());

    println!("Symbolic W-state");
    println!(
        "label=Graham memory_B={} size_class={} total_probability={}",
        w.memory_bytes(),
        verification.size_class,
        verification.total_probability
    );
    println!();
}

fn check_w_state(n_qubits: usize) {
    assert!(
        n_qubits >= 7,
        "W-state materialized check needs at least 7 qubits"
    );

    println!("Materialized W-state");
    println!("qubits={}", n_qubits);

    let start_alloc = Instant::now();
    let mut grid = SparseQuantumGridVec::new(n_qubits);
    let alloc_ms = start_alloc.elapsed().as_secs_f64() * 1_000.0;

    let create_us = grid.create_w_state_parallel();

    let start_verify = Instant::now();
    let verification = grid.verify_w_fidelity_parallel();
    let verify_ms = start_verify.elapsed().as_secs_f64() * 1_000.0;

    assert_close(verification.fidelity, 1.0, 1e-8);
    assert_eq!(verification.correct_amplitudes, n_qubits);
    assert_eq!(verification.expected_amplitudes, n_qubits);
    assert!(
        verification.max_amplitude_error <= 1e-12,
        "max W amplitude error too large: {}",
        verification.max_amplitude_error
    );
    assert_eq!(verification.max_spurious_amplitude, 0.0);
    assert_eq!(verification.memory_bytes, grid.memory_bytes());

    println!(
        "memory_B={} active_blocks={} alloc_ms={:.3} create_ms={:.3} verify_ms={:.3} fidelity={:.9} max_amp_error={:.3e} max_spurious={:.3e}",
        verification.memory_bytes,
        verification.active_blocks,
        alloc_ms,
        create_us as f64 / 1000.0,
        verify_ms,
        verification.fidelity,
        verification.max_amplitude_error,
        verification.max_spurious_amplitude
    );
    println!();
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[derive(Debug)]
struct Options {
    w_qubits: Option<usize>,
}

impl Options {
    fn from_args() -> Self {
        let mut w_qubits = Some(DEFAULT_W_QUBITS);
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--w" | "--w-qubits" => {
                    let raw = args
                        .next()
                        .unwrap_or_else(|| panic!("{arg} requires a qubit count"));
                    w_qubits = Some(
                        raw.parse::<usize>()
                            .unwrap_or_else(|_| panic!("invalid qubit count: {raw}")),
                    );
                }
                "--skip-w" => {
                    w_qubits = None;
                }
                "-h" | "--help" => {
                    print_usage_and_exit();
                }
                other => {
                    panic!("unknown argument: {other}");
                }
            }
        }

        Self { w_qubits }
    }
}

fn print_usage_and_exit() -> ! {
    println!("Usage:");
    println!("  cargo run --release -p tileuniverse-quantum --example launch_claims");
    println!(
        "  cargo run --release -p tileuniverse-quantum --example launch_claims -- --w 100000000"
    );
    println!("  cargo run --release -p tileuniverse-quantum --example launch_claims -- --skip-w");
    std::process::exit(0);
}
