//! SIMT Throughput Roadmap — **Tier 0: scalar baseline + golden oracle**.
//!
//! This module introduces **no engine change**. It builds K *independent* scalar
//! physical V2 Tile CPUs (one `Simulation` each, no shared fabric — independence is
//! the whole point of the SIMT instance model) and runs each to completion.
//!
//! It produces the two artifacts every later tier of the roadmap depends on:
//!
//!  1. **The golden oracle** — `hash_v2_final_state` for each program. Lane 0 of any
//!     future lane-packed evaluator, running the same program, must reproduce these
//!     hashes byte-for-byte. This is the built-in correctness check that makes the
//!     whole track golden-safe.
//!
//!  2. **The throughput baseline** — CPU-cycles advanced per real second on the
//!     current scalar evaluator, when K CPUs are run as K independent simulations.
//!     This is the honest number SIMT must beat: it already pays K full topology
//!     builds and K independent (non-amortized) tick streams. Build time and step
//!     time are reported separately, because SIMT's win includes amortizing the
//!     one-time topology build across N lanes.
//!
//! The corpus is the existing compiled benchmark set (`V2_COMPILED_BENCHMARKS`), so
//! the oracle stays aligned with goldens already trusted elsewhere in the tree.
//!
//! Run: `cargo run --release --example v2_simt_baseline`

use std::error::Error;
use std::rc::Rc;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::simulation::Simulation;
use crate::tile_cpu::hash_v2_final_state;
use crate::tile_cpu::v2_compiler_bench::{V2_COMPILED_BENCHMARKS, V2CompiledBenchCase};
use crate::tile_cpu::v2_mmio::V2MmioHandle;
use crate::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
use crate::tile_cpu::v2_parser::compile_source;
use crate::tile_cpu::{V2Builder, V2SynthConfig};

/// Per-CPU result of one scalar baseline run.
#[derive(Debug, Clone)]
pub struct SimtBaselineResult {
    pub name: &'static str,
    pub program_words: usize,
    /// Cycles this CPU retired before halting (or hitting `max_cycles`).
    pub cycles: u64,
    /// `hash_v2_final_state` — **the oracle**. Future lane-0 must match this.
    pub golden_hash: u64,
    /// Value left in R0 at halt (sanity check vs the corpus's `expected_r0`).
    pub r0: u64,
    pub halted: bool,
    /// One-time topology build cost (the part SIMT amortizes across lanes).
    pub build_time: Duration,
    /// Steady-state stepping cost — the throughput-relevant number.
    pub step_time: Duration,
}

/// Aggregate report over a baseline run of K CPUs.
#[derive(Debug, Clone)]
pub struct SimtBaselineReport {
    pub per_cpu: Vec<SimtBaselineResult>,
    pub total_cycles: u64,
    pub total_build_time: Duration,
    pub total_step_time: Duration,
}

impl SimtBaselineReport {
    /// The headline baseline: CPU-cycles advanced per real second, **stepping only**
    /// (excludes the one-time topology build, which SIMT amortizes).
    pub fn cycles_per_sec(&self) -> f64 {
        rate(self.total_cycles, self.total_step_time)
    }

    /// End-to-end rate including build cost — the "cold" number.
    pub fn cycles_per_sec_with_build(&self) -> f64 {
        rate(
            self.total_cycles,
            self.total_build_time + self.total_step_time,
        )
    }
}

/// Build one scalar physical V2 CPU configured **identically** to
/// `run_v2_compiled_benchmark` (so its golden hash is directly comparable), run it
/// to completion, and return its result.
pub fn run_scalar_baseline_case(
    case: &V2CompiledBenchCase,
) -> Result<SimtBaselineResult, Box<dyn Error>> {
    let program = compile_source(case.source).map_err(|e| e.to_string())?;

    // --- one-time topology build (amortized away by SIMT) ---
    let build_start = Instant::now();
    let console = Rc::new(V2MmioCombinedDevice::new(0x000B_E5C0));
    let mut sim = Simulation::with_size_layered(128, 640, 16);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(128)
        .with_ram_size(128)
        .with_extended_pc()
        .with_mmio(V2MmioHandle::from_rc(console.clone()))
        .with_synth_blocks(V2SynthConfig::max_authority())
        .build(&mut sim);
    let build_time = build_start.elapsed();

    // --- steady-state stepping (the throughput-relevant work) ---
    let step_start = Instant::now();
    let mut cycles = 0u64;
    for _ in 0..case.max_cycles {
        if cpu.is_halted() {
            break;
        }
        cpu.step(&mut sim);
        cycles += 1;
    }
    let step_time = step_start.elapsed();

    Ok(SimtBaselineResult {
        name: case.name,
        program_words: program.len(),
        cycles,
        golden_hash: hash_v2_final_state(&cpu, &sim),
        r0: cpu.read_reg(&sim, 0),
        halted: cpu.is_halted(),
        build_time,
        step_time,
    })
}

/// Run a set of cases as K independent scalar CPUs and aggregate the report.
pub fn run_scalar_baseline(
    cases: &[V2CompiledBenchCase],
) -> Result<SimtBaselineReport, Box<dyn Error>> {
    let per_cpu_results: Vec<_> = cases
        .par_iter()
        .map(|case| run_scalar_baseline_case(case).map_err(|e| e.to_string()))
        .collect();
    let per_cpu = per_cpu_results
        .into_iter()
        .collect::<Result<Vec<_>, String>>()?;

    let total_cycles = per_cpu.iter().map(|r| r.cycles).sum();
    let total_build_time = per_cpu.iter().map(|r| r.build_time).sum();
    let total_step_time = per_cpu.iter().map(|r| r.step_time).sum();

    Ok(SimtBaselineReport {
        per_cpu,
        total_cycles,
        total_build_time,
        total_step_time,
    })
}

/// Convenience: run the full standard corpus.
pub fn run_scalar_baseline_corpus() -> Result<SimtBaselineReport, Box<dyn Error>> {
    run_scalar_baseline(V2_COMPILED_BENCHMARKS)
}

fn rate(total: u64, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs == 0.0 {
        total as f64
    } else {
        total as f64 / secs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case_named(name: &str) -> &'static V2CompiledBenchCase {
        V2_COMPILED_BENCHMARKS
            .iter()
            .find(|case| case.name == name)
            .unwrap_or_else(|| panic!("missing benchmark case '{name}'"))
    }

    #[test]
    fn report_rates_use_aggregate_times() {
        let report = SimtBaselineReport {
            per_cpu: Vec::new(),
            total_cycles: 2_000,
            total_build_time: Duration::from_millis(1_500),
            total_step_time: Duration::from_millis(500),
        };
        assert_eq!(report.cycles_per_sec(), 4_000.0);
        assert_eq!(report.cycles_per_sec_with_build(), 1_000.0);
    }

    /// Even one physical V2 CPU build/run is slow in debug, so keep this out of
    /// default unit tests. It is the quick manual/CI-gated smoke for the oracle
    /// path when `slow-tests` is explicitly enabled.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow physical-CPU oracle smoke; run with --features slow-tests"
    )]
    fn scalar_baseline_fast_case_matches_expected_r0() {
        let case = case_named("fact_iterative");
        let res = run_scalar_baseline_case(case)
            .unwrap_or_else(|e| panic!("baseline '{}' failed: {e}", case.name));
        assert!(res.halted, "case '{}' did not halt", case.name);
        assert_eq!(
            res.r0, case.expected_r0,
            "case '{}' R0 mismatch (got {}, want {})",
            case.name, res.r0, case.expected_r0
        );
    }

    /// The full baseline must reproduce the corpus's expected R0 for every case —
    /// i.e. these scalar physical CPUs compute the right answers. This pins the
    /// oracle: `golden_hash` is only meaningful if the run is correct.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow full physical-CPU corpus; run with --features slow-tests"
    )]
    fn scalar_baseline_full_corpus_matches_expected_r0() {
        for case in V2_COMPILED_BENCHMARKS {
            let res = run_scalar_baseline_case(case)
                .unwrap_or_else(|e| panic!("baseline '{}' failed: {e}", case.name));
            assert!(res.halted, "case '{}' did not halt", case.name);
            assert_eq!(
                res.r0, case.expected_r0,
                "case '{}' R0 mismatch (got {}, want {})",
                case.name, res.r0, case.expected_r0
            );
        }
    }

    /// The oracle must be deterministic: the same program built and run twice yields
    /// the identical golden hash. Every later tier's lane-0 check relies on this.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow physical-CPU oracle determinism; run with --features slow-tests"
    )]
    fn golden_hash_is_deterministic() {
        let case = case_named("fact_iterative");
        let a = run_scalar_baseline_case(case).unwrap();
        let b = run_scalar_baseline_case(case).unwrap();
        assert_eq!(
            a.golden_hash, b.golden_hash,
            "golden hash for '{}' is not deterministic",
            case.name
        );
    }
}
