//! Sprint 388 (HW/SW co-design, Phase 4): one source, one SoC — the showcase.
//!
//! ONE C-like source file. `main()` calls `madd(a, b, c) = a*b + c` several times.
//! Compile it TWO ways from the SAME source, changing nothing but one keyword:
//!
//!   1. **all-CPU**     — `int madd(...)`: `madd` compiles to instructions and runs on
//!                        the tile CPU's single ALU, reused every cycle (the multiply is
//!                        software-assisted — the lone authority gap).
//!   2. **accelerated** — `accel int madd(...)`: `madd` is synthesized by the HLS middle
//!                        layer into a spatial tile datapath behind an MMIO device; its
//!                        call sites lower to the indexed-operand protocol
//!                        (ARG_SELECT / ARG_DATA / RESULT).
//!
//! Both builds run on the **physical tile CPU** and must return the identical result.
//! The table is the classic latency–area trade-off, now *inside a running SoC*: the
//! accelerated build retires fewer CPU ticks / re-evaluates less of the fabric per
//! `madd` (each call is one tile-datapath evaluation) at the cost of a fixed datapath
//! footprint (cells / critical path / placed tiles).
//!
//! Run: `cargo run --release --example v2_soc_codesign`                 (width 5, seconds)
//!      `cargo run --release --example v2_soc_codesign -- 8`            (width 8)
//!      `cargo run --release --example v2_soc_codesign -- 8 --compare-sa`  (adds the
//!          AlphaFabric SA area comparison; the SA place+verify takes ~2 min)
//!
//! The compiler's placement policy is row-major bridge ladder first (fast,
//! deterministic), AlphaFabric SA fallback only when the ladder fails to route.
//! Measured: the ladder routes madd up to width 8 — at a large footprint (the
//! escalated halo spreads 270 cells over ~56k tiles). `--compare-sa` places the
//! SAME datapath with the SA placer (verify-gated) and prints the area ratio,
//! showing what the connectivity-aware placer buys over topological order.

use std::error::Error;

use engine::synth::alphafabric::{AnnealConfig, Circuit, sa_place_escalating};
use engine::synth::cell_lib::CellLibrary;
use engine::synth::hls::synthesize_func;
use engine::synth::timing::{TimingConstraint, analyze_timing};
use engine::synth::{HlsConfig, SynthConfig, materialize_inversions, synthesize};
use engine::tile_cpu::v2_hls_accel::{V2SocBenchOutcome, run_v2_soc_benchmark};
use engine::tile_cpu::v2_parser::parse_program;

/// The one source. Only the `accel` keyword in front of `madd` differs between builds.
/// Each individual `madd` result fits the width-5 datapath (mask 31), so the masked
/// tile datapath and the 64-bit CPU agree exactly; the accumulator `s` is the CPU's.
const ACCEL_SRC: &str = r#"
    accel int madd(int a, int b, int c) { return a * b + c; }
    int main() {
        int s = 0;
        s = s + madd(2, 3, 4);
        s = s + madd(3, 5, 2);
        s = s + madd(4, 4, 1);
        s = s + madd(6, 2, 3);
        return s;
    }
"#;

/// `madd(2,3,4)+madd(3,5,2)+madd(4,4,1)+madd(6,2,3) = 10+17+17+15`.
const EXPECTED: u64 = 59;

/// Default datapath bit width. Width 5 keeps the multiplier's place & route inside a
/// few seconds (row-major fast path); the protocol and lowering are width-independent.
/// Pass a width as the first CLI arg (e.g. `-- 8`) to synthesize a wider datapath —
/// width 8 exercises the AlphaFabric SA-fallback placement (takes minutes).
const DEFAULT_WIDTH: u32 = 5;

const MAX_CYCLES: u64 = 100_000;

fn main() -> Result<(), Box<dyn Error>> {
    let width: u32 = std::env::args()
        .skip(1)
        .find_map(|s| s.parse().ok())
        .unwrap_or(DEFAULT_WIDTH);
    let compare_sa = std::env::args().any(|a| a == "--compare-sa");
    assert!(
        (5..=64).contains(&width),
        "width must be 5..=64 (madd results need 5 bits)"
    );

    println!("One source, one SoC: madd(a, b, c) = a * b + c  (width {width})");
    println!("main() calls madd 4 times; the result must match across builds.\n");

    // --- Build 1: the SAME source with the `accel` marker stripped (pure-CPU). ---
    let cpu_src = ACCEL_SRC.replace("accel int madd", "int madd");
    let cpu = run_v2_soc_benchmark(&cpu_src, &HlsConfig { width }, MAX_CYCLES)
        .map_err(|e| format!("all-CPU build: {e:?}"))?;
    assert_eq!(cpu.r0, EXPECTED, "all-CPU build result");
    assert!(cpu.accel_kind.is_none(), "all-CPU build has no accelerator");

    // --- Build 2: madd marked `accel` → synthesized tile datapath behind MMIO. ---
    let acc = run_v2_soc_benchmark(ACCEL_SRC, &HlsConfig { width }, MAX_CYCLES)
        .map_err(|e| format!("accelerated build: {e:?}"))?;
    assert_eq!(acc.r0, EXPECTED, "accelerated build result");
    assert_eq!(
        acc.accel_evals, 4,
        "all 4 madd calls must have been computed by the tile datapath"
    );

    // --- The whole-program table. ---
    println!(
        "{:<14} {:>6} {:>8} {:>8} {:>14} {:>12} {:>6}",
        "build", "words", "instrs", "ticks", "tiles_eval", "accel_evals", "R0"
    );
    print_row("all-CPU", &cpu);
    print_row("accelerated", &acc);

    // --- The accelerator's datapath footprint (the area side of the trade-off). ---
    // Netlist-level stats come from the synth front half only (AIG → map → STA);
    // the *placed* footprint is reported straight from the device the benchmark
    // actually ran (`accel_footprint`/`accel_placement`), so it is truthful for
    // both the row-major fast path and the SA fallback — and costs no second P&R.
    let program = parse_program(ACCEL_SRC).map_err(|e| format!("parse error: {e:?}"))?;
    let madd = program
        .accel_funcs
        .iter()
        .find(|f| f.name == "madd")
        .ok_or("accel function `madd` not found")?;
    let aig = synthesize_func(madd, &HlsConfig { width })?;
    let stats = aig.stats();
    let lib = CellLibrary::tile_native();
    let synth_result = synthesize(&aig, &lib, &SynthConfig::default());
    let materialized = materialize_inversions(&synth_result.netlist, &lib);
    let timing = analyze_timing(&materialized, &lib, &TimingConstraint::intrinsic());
    let placed = acc
        .accel_footprint
        .map(|(w, h, l)| format!("{w}x{h}x{l} ({} tiles)", w * h))
        .unwrap_or_else(|| "?".to_string());
    println!(
        "\nAccelerator datapath ({}, placement: {}, HLS width {width}):",
        acc.accel_kind.unwrap_or("?"),
        acc.accel_placement.as_deref().unwrap_or("?"),
    );
    println!(
        "  {} ANDs, depth {} -> {} cells, critical path {:.0} tile-delays, placed {placed}",
        stats.num_and_nodes,
        stats.depth,
        materialized.num_gates(),
        timing.circuit_delay,
    );

    // --- The trade-off, in honest deltas computed from the run above. ---
    let d_instrs =
        cpu.metrics.instructions_executed as i64 - acc.metrics.instructions_executed as i64;
    let d_ticks = cpu.metrics.cycles as i64 - acc.metrics.cycles as i64;
    let d_tiles =
        cpu.metrics.total_tiles_evaluated as i64 - acc.metrics.total_tiles_evaluated as i64;
    let placed_tiles = acc.accel_footprint.map(|(w, h, _)| w * h).unwrap_or(0);
    println!("\nTrade-off (all-CPU vs accelerated, whole program):");
    println!(
        "  CPU instructions retired : {} -> {}  ({})",
        cpu.metrics.instructions_executed,
        acc.metrics.instructions_executed,
        signed(d_instrs),
    );
    println!(
        "  CPU ticks                : {} -> {}  ({})",
        cpu.metrics.cycles,
        acc.metrics.cycles,
        signed(d_ticks),
    );
    println!(
        "  fabric re-evaluated      : {} -> {} tile-evals  ({})",
        cpu.metrics.total_tiles_evaluated,
        acc.metrics.total_tiles_evaluated,
        signed(d_tiles),
    );
    println!(
        "  accelerator fabric spent : {placed_tiles} placed tiles, {} datapath evals",
        acc.accel_evals,
    );
    println!(
        "\nBoth builds returned R0 = {EXPECTED}. One source — program or circuit, per function."
    );

    // --- Optional: AlphaFabric SA placement of the SAME datapath (area comparison). ---
    // The compiler chose row-major-first for build speed; this shows what the
    // connectivity-aware SA placer buys in area on the same AIG, verify-gated
    // (the layout must still compute the exact function on real tiles).
    if compare_sa {
        println!("\nAlphaFabric SA placement of the same {width}-bit madd datapath:");
        let circuit = Circuit::from_aig("madd_datapath", aig);
        let sa_cfg = AnnealConfig {
            iterations: 30_000,
            ..AnnealConfig::default()
        };
        match sa_place_escalating(&circuit, &[3, 4, 5], &sa_cfg) {
            Some(sa) => {
                let (w, h) = (sa.export.sim.width(), sa.export.sim.height());
                let sa_tiles = w * h;
                println!(
                    "  SA-placed (halo {}): {w}x{h} ({sa_tiles} tiles), physically verified: {}",
                    sa.halo, sa.verified,
                );
                if placed_tiles > 0 && sa_tiles > 0 {
                    println!(
                        "  compiled build used {placed_tiles} tiles ({}) -> SA layout is {:.1}x denser",
                        acc.accel_placement.as_deref().unwrap_or("?"),
                        placed_tiles as f64 / sa_tiles as f64,
                    );
                }
            }
            None => println!("  SA failed to route+verify at halos 3-5"),
        }
    }
    Ok(())
}

fn print_row(name: &str, o: &V2SocBenchOutcome) {
    println!(
        "{:<14} {:>6} {:>8} {:>8} {:>14} {:>12} {:>6}",
        name,
        o.program_words,
        o.metrics.instructions_executed,
        o.metrics.cycles,
        o.metrics.total_tiles_evaluated,
        o.accel_evals,
        o.r0,
    );
}

/// Format a signed delta with an explicit sign and "fewer/more" sense for the table.
fn signed(d: i64) -> String {
    match d.cmp(&0) {
        std::cmp::Ordering::Greater => format!("-{d} fewer on the accelerated build"),
        std::cmp::Ordering::Less => format!("+{} more on the accelerated build", -d),
        std::cmp::Ordering::Equal => "no change".to_string(),
    }
}
