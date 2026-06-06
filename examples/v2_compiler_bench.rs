//! Sprint 377 (Perf Track P1): the compiled-program performance dashboard.
//!
//! Compiles each program in the benchmark corpus, runs it on the physical tile CPU,
//! and prints size / instructions / ticks / ticks-per-instruction / tiles-evaluated.
//! This is the yardstick for the perf-track sprints (register allocation, DIV, the
//! simulation engine) — every change is judged against these numbers.
//!
//! Run: `cargo run --release --example v2_compiler_bench`

use std::error::Error;

use engine::tile_cpu::v2_compiler_bench::{
    bench_report_header, format_bench_row, format_phase_row, phase_report_header,
};
use engine::tile_cpu::{V2_COMPILED_BENCHMARKS, V2CompiledBenchOutcome, run_v2_compiled_benchmark};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Compiled-program benchmark dashboard (physical tile CPU)\n");
    println!("{}", bench_report_header());
    println!("{}", "-".repeat(80));

    let mut total_ticks = 0u64;
    let mut outcomes: Vec<V2CompiledBenchOutcome> = Vec::new();
    for case in V2_COMPILED_BENCHMARKS {
        let outcome = run_v2_compiled_benchmark(case)?;
        println!("{}", format_bench_row(&outcome));
        total_ticks += outcome.metrics.cycles;
        outcomes.push(outcome);
    }
    println!("{}", "-".repeat(80));
    println!("total ticks across corpus: {total_ticks}");

    // Sprint 381 (Perf L3 profiling): where do the ~7,000 tile-evals/instruction go?
    println!("\nPer-phase tile-eval breakdown (lever 3 = simulation engine):\n");
    println!("{}", phase_report_header());
    println!("{}", "-".repeat(80));
    let mut agg = [0u64; 5];
    let mut agg_instr = 0u64;
    for o in &outcomes {
        println!("{}", format_phase_row(o));
        for (i, (_, e)) in o.per_path.iter().enumerate() {
            agg[i] += e;
        }
        agg_instr += o.metrics.instructions_executed;
    }
    println!("{}", "-".repeat(80));
    let agg_total: u64 = agg.iter().sum();
    println!("corpus aggregate (share of total tile-evals):");
    for (i, name) in ["cone", "settle", "constswap", "branch", "commit"]
        .iter()
        .enumerate()
    {
        let pct = if agg_total > 0 {
            agg[i] as f64 / agg_total as f64 * 100.0
        } else {
            0.0
        };
        let per_instr = if agg_instr > 0 {
            agg[i] as f64 / agg_instr as f64
        } else {
            0.0
        };
        println!("  {name:<10} {pct:>5.1}%   {per_instr:>8.0} evals/instr");
    }
    Ok(())
}
