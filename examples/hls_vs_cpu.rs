//! One source, two targets — the dual-target cost comparison (Sprint 383, Phase 2).
//!
//! The same C-like function is lowered both ways and the costs are printed side
//! by side:
//!
//! 1. **CPU target** (`v2_compiler` + `v2_compiler_bench`): the function compiles to
//!    instructions and runs sequentially on the physical tile CPU — one ALU reused
//!    over many cycles.
//! 2. **HLS target** (`synth::hls`): the same AST bit-blasts into a spatial datapath —
//!    area is spent so the answer settles combinationally in one critical-path delay.
//!
//! This is the classic latency–area trade-off, measured on real infrastructure at
//! both ends. The CPU numbers come from the compiled-benchmark harness; the HLS
//! numbers from AIG stats, the technology mapper, gate-level STA, and the placed
//! export grid. As a sanity anchor, the 8-bit datapath is also *run as tiles* on the
//! showcase inputs and must agree with the CPU's R0.
//!
//! Run: `cargo run --release --example hls_vs_cpu`

use std::error::Error;

use engine::simulation::Simulation;
use engine::synth::bridge::{compile_and_inject, evaluate_block};
use engine::synth::hls::{report_func, HlsReport};
use engine::synth::{synthesize_func, CellLibrary, HlsConfig};
use engine::tile_cpu::v2_compiler_bench::{run_v2_compiled_benchmark, V2CompiledBenchCase};
use engine::tile_cpu::v2_parser::parse_program;

/// The one source. `madd` is the function under comparison; `main` gives the CPU
/// target a concrete invocation: madd(2, 3, 7) = 13 (fits in 4 bits, so the 4-bit
/// tile datapath and the 64-bit CPU agree exactly).
const SOURCE: &str = r#"
    int madd(int a, int b, int c) {
        return a * b + c;
    }
    int main() { return madd(2, 3, 7); }
"#;

const SHOWCASE_ARGS: (u64, u64, u64) = (2, 3, 7);
const EXPECTED: u64 = 13;

fn main() -> Result<(), Box<dyn Error>> {
    println!("One source, two targets: madd(a, b, c) = a * b + c\n");

    // --- Target 1: the CPU (sequential program) ---
    let case = V2CompiledBenchCase {
        name: "madd",
        source: SOURCE,
        max_cycles: 100_000,
        expected_console: "",
        expected_r0: EXPECTED,
    };
    let cpu = run_v2_compiled_benchmark(&case)?;
    assert_eq!(cpu.r0, EXPECTED, "CPU target result");

    println!("CPU target (physical tile CPU, 64-bit ALU reused every cycle):");
    println!("  program size        {:>8} words", cpu.program_words);
    println!(
        "  instructions        {:>8}",
        cpu.metrics.instructions_executed
    );
    println!("  cycles (ticks)      {:>8}", cpu.metrics.cycles);
    println!(
        "  tiles evaluated     {:>8}  ({:.0} per instruction)",
        cpu.metrics.total_tiles_evaluated,
        cpu.tiles_per_instruction()
    );
    println!("  result (R0)         {:>8}\n", cpu.r0);

    // --- Target 2: the HLS datapath (spatial circuit) ---
    let program = parse_program(SOURCE).map_err(|e| format!("parse error: {e:?}"))?;
    let func = program
        .funcs
        .iter()
        .find(|f| f.name == "madd")
        .ok_or("function `madd` not found")?;

    println!("HLS target (combinational datapath, one answer per critical path):");
    println!(
        "  {:>5} {:>6} {:>6} {:>7} {:>10} {:>14}",
        "width", "ANDs", "depth", "cells", "crit.path", "placed tiles"
    );
    let mut reports: Vec<HlsReport> = Vec::new();
    for width in [4u32, 8] {
        let r = report_func(func, &HlsConfig { width })?;
        let placed = match &r.placed {
            Some(p) => format!("{:>9} ({}x{})", p.area_tiles(), p.width, p.height),
            None => "        — (unroutable)".to_string(),
        };
        println!(
            "  {:>5} {:>6} {:>6} {:>7} {:>10.0} {placed}",
            r.width, r.aig_and_nodes, r.aig_depth, r.mapped_cells, r.critical_path_delay,
        );
        reports.push(r);
    }

    // --- The trade-off, in one sentence ---
    let r8 = &reports[1];
    let r8_area = r8.placed.map(|p| p.area_tiles()).unwrap_or(0);
    println!();
    println!("Trade-off at 8 bits:");
    println!(
        "  CPU:  1 ALU, {} cycles for one madd call (whole-program run: {} instrs).",
        cpu.metrics.cycles, cpu.metrics.instructions_executed
    );
    println!(
        "  HLS:  {} cells over {} tiles, answer settles in ~{:.0} tile-delays — every {:.0} delays another answer.",
        r8.mapped_cells, r8_area, r8.critical_path_delay, r8.critical_path_delay
    );

    // --- Physical anchor: run the 4-bit datapath as tiles on the showcase inputs ---
    let width = 4u32;
    let w = width as usize;
    let aig = synthesize_func(func, &HlsConfig { width })?;
    let lib = CellLibrary::tile_native();
    let mut sim = Simulation::with_size_layered(256, 1280, 16);
    let (block, _export) = compile_and_inject(&aig, &lib, &mut sim, 64, 100, 0)?;

    let (a, b, c) = SHOWCASE_ARGS;
    let mut inputs: Vec<u64> = Vec::with_capacity(3 * w);
    for &val in &[a, b, c] {
        for k in 0..w {
            inputs.push(if (val >> k) & 1 == 1 { u64::MAX } else { 0 });
        }
    }
    let outs = evaluate_block(&mut sim, &block, &inputs)?;
    let tiles = outs
        .iter()
        .enumerate()
        .fold(0u64, |acc, (k, &v)| acc | (((v != 0) as u64) << k));

    println!();
    println!(
        "Cross-check: madd({a},{b},{c}) — CPU R0 = {}, tile datapath = {}",
        cpu.r0, tiles
    );
    if tiles == cpu.r0 {
        println!("Both targets agree. Same source, program or circuit — pick your trade-off.");
        Ok(())
    } else {
        Err("HLS tile datapath disagreed with the CPU target".into())
    }
}
