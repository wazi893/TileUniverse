//! V2 build-cost amortization — Stage 1: build once, clone N, prove the oracle.
//!
//! The SIMT Tier-0 baseline surfaced that topology BUILD (synth + placement +
//! scope-closure BFS, ~5 s/CPU under `max_authority`) is ~97% of wall-clock when
//! running K independent physical V2 CPUs — stepping is ~3%. Every independent
//! CPU re-runs that whole build from scratch.
//!
//! But the build is program-INDEPENDENT topology: the same tile types, neighbor
//! structure, and scope closures regardless of the program (only ROM *values*
//! differ). `Simulation` already has a deep `Clone` (it copies each tile's atomic
//! logic value + every field array), and `TileCpuV2` derives `Clone` with all
//! mutable run state behind `Cell`. So we can pay the build ONCE and clone the
//! topology per lane — a memcpy instead of a re-synthesis.
//!
//! This Stage-1 demo proves the win is real AND correctness-preserving:
//!   1. Build a template CPU once (measured).
//!   2. Clone (cpu + sim) per lane and run each to completion (measured).
//!   3. ORACLE: every lane's `hash_v2_final_state` is byte-identical to an
//!      INDEPENDENT fresh build+run of the same program. If the clone diverged,
//!      this fails. (Same program across lanes ⇒ all lanes identical.)
//!
//! Stage 2 (separate) factors ROM injection out of `wire_v2_cpu` so different
//! programs can share one build — the full corpus win. Here every lane runs the
//! same program (the SPMD-lockstep case the SIMT roadmap targets first).
//!
//! Run: `cargo run --release --example v2_build_amortization`

use std::error::Error;
use std::rc::Rc;
use std::time::{Duration, Instant};

use engine::simulation::Simulation;
use engine::tile_cpu::hash_v2_final_state;
use engine::tile_cpu::v2_compiler_bench::{V2CompiledBenchCase, V2_COMPILED_BENCHMARKS};
use engine::tile_cpu::v2_mmio::V2MmioHandle;
use engine::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
use engine::tile_cpu::v2_parser::compile_source;
use engine::tile_cpu::{TileCpuV2, V2Builder, V2SynthConfig};

/// Number of independent lanes (CPUs) to run on one built topology.
const LANES: usize = 8;

/// Build one physical V2 CPU configured identically to the SIMT baseline / the
/// compiled-benchmark harness, so its golden hash is directly comparable.
fn build_cpu(program: &[u32]) -> (TileCpuV2, Simulation) {
    let console = Rc::new(V2MmioCombinedDevice::new(0x000B_E5C0));
    let mut sim = Simulation::with_size_layered(128, 640, 16);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(program)
        .with_rom_size(128)
        .with_ram_size(128)
        .with_extended_pc()
        .with_mmio(V2MmioHandle::from_rc(console.clone()))
        .with_synth_blocks(V2SynthConfig::max_authority())
        .build(&mut sim);
    (cpu, sim)
}

/// Run a CPU to halt (or `max_cycles`), returning (cycles, golden_hash, r0).
fn run_to_halt(cpu: &TileCpuV2, sim: &mut Simulation, max_cycles: u64) -> (u64, u64, u64) {
    let mut cycles = 0u64;
    for _ in 0..max_cycles {
        if cpu.is_halted() {
            break;
        }
        cpu.step(sim);
        cycles += 1;
    }
    (cycles, hash_v2_final_state(cpu, sim), cpu.read_reg(sim, 0))
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("V2 build-cost amortization — Stage 1 (build once, clone {LANES})\n");

    let case: &V2CompiledBenchCase = V2_COMPILED_BENCHMARKS
        .iter()
        .find(|c| c.name == "fact_iterative")
        .expect("fact_iterative in corpus");
    let program = compile_source(case.source).map_err(|e| e.to_string())?;

    // --- Oracle: an INDEPENDENT fresh build+run (the scalar reference). ---
    let (ref_cpu, mut ref_sim) = build_cpu(&program);
    let (_, golden, ref_r0) = run_to_halt(&ref_cpu, &mut ref_sim, case.max_cycles);
    assert_eq!(ref_r0, case.expected_r0, "reference R0 wrong — bad oracle");

    // --- Build the template ONCE (the cost we amortize). ---
    let build_start = Instant::now();
    let (template_cpu, template_sim) = build_cpu(&program);
    let build_time = build_start.elapsed();

    // --- Amortized: clone (cpu + sim) per lane, run each. ---
    let mut clone_total = Duration::ZERO;
    let mut step_total = Duration::ZERO;
    let mut total_cycles = 0u64;
    for lane in 0..LANES {
        let clone_start = Instant::now();
        let cpu = template_cpu.clone();
        let mut sim = template_sim.clone();
        clone_total += clone_start.elapsed();

        let step_start = Instant::now();
        let (cycles, hash, r0) = run_to_halt(&cpu, &mut sim, case.max_cycles);
        step_total += step_start.elapsed();
        total_cycles += cycles;

        // ORACLE: a cloned topology must compute byte-identically to a fresh build.
        assert_eq!(hash, golden, "lane {lane} golden hash diverged from fresh build");
        assert_eq!(r0, case.expected_r0, "lane {lane} R0 wrong");
    }

    // --- Report. Build cost is what each INDEPENDENT CPU pays in the baseline. ---
    let per_build = build_time.as_secs_f64();
    let per_clone = clone_total.as_secs_f64() / LANES as f64;
    let per_step = step_total.as_secs_f64() / LANES as f64;

    // Baseline pays a full build per CPU; amortized pays one build for all lanes.
    let baseline_wall = LANES as f64 * per_build + step_total.as_secs_f64();
    let amortized_wall = per_build + clone_total.as_secs_f64() + step_total.as_secs_f64();

    println!("program: {} ({} words), {LANES} lanes, all identical\n", case.name, program.len());
    println!("  topology build (1×):     {:>9.1} ms   ← paid ONCE when amortized", per_build * 1000.0);
    println!("  clone per lane:          {:>9.1} ms   ← the amortized cost", per_clone * 1000.0);
    println!("  step per lane:           {:>9.1} ms", per_step * 1000.0);
    println!();
    println!("  baseline wall ({LANES} builds): {:>9.2} s", baseline_wall);
    println!("  amortized wall (1 build): {:>9.2} s", amortized_wall);
    println!("  speedup:                 {:>9.2}×", baseline_wall / amortized_wall);
    println!();
    println!(
        "  throughput (with build):  baseline {:>10.1}  →  amortized {:>10.1}  CPU-cycles/sec",
        total_cycles as f64 / baseline_wall,
        total_cycles as f64 / amortized_wall,
    );
    println!(
        "\nOracle PASSED: all {LANES} cloned lanes are byte-identical to an independent\n\
         fresh build (golden hash {golden:#018x}). Build amortization is correctness-preserving.\n\
         Clone is a memcpy of the built topology — no re-synthesis. Stage 2 adds ROM\n\
         injection so different programs share one build (the full corpus win)."
    );

    Ok(())
}
