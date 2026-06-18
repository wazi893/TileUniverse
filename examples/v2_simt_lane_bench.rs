//! SIMT Tier 2a — lane-packed dense kernel GO/NO-GO microbenchmark.
//!
//! Measures the lane-packed dense compact-op kernel (`LaneKernel`) against the
//! scalar `propagate_compact` interpreter on the REAL settle op array of a
//! max-authority compiled V2 CPU (the settle phase is ~72% of per-step tile
//! evals). Both sides are dense (evaluate every op), so this isolates the
//! question the roadmap's Tier-2 gate asks: does packing N instances into the
//! value dimension amortize the interpreter's per-op overhead across lanes?
//!
//! Honest-reporting rules: every number comes from a run that executed; the
//! lane-0 oracle is checked in-process against the scalar evaluator before any
//! timing is reported. No ×N extrapolation.
//!
//! Run: cargo run --release --example v2_simt_lane_bench

use std::error::Error;
use std::rc::Rc;
use std::time::Instant;

use engine::simulation::Simulation;
use engine::tile_cpu::v2_compiler_bench::V2_COMPILED_BENCHMARKS;
use engine::tile_cpu::v2_mmio::V2MmioHandle;
use engine::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
use engine::tile_cpu::v2_parser::compile_source;
use engine::tile_cpu::{LaneKernel, LanePack, V2Builder, V2SynthConfig};

const WARMUP_ITERS: usize = 200;
const TIMED_ITERS: usize = 2_000;
const LANE_CONFIGS: [usize; 6] = [1, 2, 4, 8, 16, 32];

fn main() -> Result<(), Box<dyn Error>> {
    println!("SIMT Tier 2a: lane-packed dense kernel vs scalar propagate_compact\n");

    // Build one max-authority compiled CPU — same config as the Tier-0
    // baseline — and take its settle compact-op array.
    let case = V2_COMPILED_BENCHMARKS
        .iter()
        .find(|c| c.name == "fact_iterative")
        .expect("fact_iterative in corpus");
    let program = compile_source(case.source).map_err(|e| e.to_string())?;

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
    println!(
        "topology build: {:.1} ms (once — amortized across all lanes)",
        build_start.elapsed().as_secs_f64() * 1000.0
    );

    // Step a few cycles so the settle scope holds realistic mid-execution values.
    for _ in 0..8 {
        cpu.step(&mut sim);
    }

    let (ops, wvia) = cpu.settle_compact_view();
    let ops = ops.to_vec();
    let wvia = wvia.to_vec();
    println!(
        "settle compact ops: {} (real CPU settle scope)\n",
        ops.len()
    );

    let kernel = match LaneKernel::new_from_sim(&ops, &wvia, &sim) {
        Ok(k) => k,
        Err(e) => {
            println!("preflight failed — settle ops not lane-kernel compatible: {e}");
            return Ok(());
        }
    };

    // ---- Oracle: lane 0 (and every other lane) must match the scalar
    // evaluator after the same number of dense iterations. ----
    let oracle_iters = 5;
    let mut scalar_sim = sim.clone();
    for _ in 0..oracle_iters {
        scalar_sim.propagate_compact(&ops, &wvia);
    }
    let mut lp = LanePack::from_sim(&sim, 8);
    for _ in 0..oracle_iters {
        kernel.eval(&mut lp);
    }
    let mut checked = 0usize;
    for op in &ops {
        let idx = op.idx as usize;
        let want = scalar_sim.tilemap.value(idx);
        for lane in 0..lp.lanes() {
            assert_eq!(
                lp.get(idx, lane),
                want,
                "ORACLE FAILED: tile {idx} lane {lane}"
            );
        }
        checked += 1;
    }
    println!(
        "oracle PASSED: {} op-output tiles byte-identical to scalar propagate_compact (8 lanes)\n",
        checked
    );

    // ---- Scalar reference timing (per-instance dense interpreter). ----
    let mut bench_sim = sim.clone();
    for _ in 0..WARMUP_ITERS {
        bench_sim.propagate_compact(&ops, &wvia);
    }
    let t0 = Instant::now();
    for _ in 0..TIMED_ITERS {
        bench_sim.propagate_compact(&ops, &wvia);
    }
    let scalar_per_iter = t0.elapsed().as_secs_f64() / TIMED_ITERS as f64;
    println!(
        "scalar propagate_compact: {:8.2} µs/iter  ({:.1} M op-evals/s)\n",
        scalar_per_iter * 1e6,
        ops.len() as f64 / scalar_per_iter / 1e6
    );

    // ---- Lane-packed kernel timings. ----
    println!("lanes   µs/iter   µs/iter/lane   M op-evals/s   per-CPU speedup vs scalar");
    println!("{}", "-".repeat(78));
    let mut go_speedup_at_8 = 0.0f64;
    for &lanes in &LANE_CONFIGS {
        let mut lp = LanePack::from_sim(&sim, lanes);
        for _ in 0..WARMUP_ITERS {
            kernel.eval(&mut lp);
        }
        let t0 = Instant::now();
        for _ in 0..TIMED_ITERS {
            kernel.eval(&mut lp);
        }
        let per_iter = t0.elapsed().as_secs_f64() / TIMED_ITERS as f64;
        let per_lane = per_iter / lanes as f64;
        let speedup = scalar_per_iter / per_lane;
        if lanes == 8 {
            go_speedup_at_8 = speedup;
        }
        println!(
            "{:>5} {:>9.2} {:>14.3} {:>14.1} {:>12.2}x",
            lanes,
            per_iter * 1e6,
            per_lane * 1e6,
            (ops.len() * lanes) as f64 / per_iter / 1e6,
            speedup
        );
    }

    println!(
        "\nGate-2 kernel signal (target >=2x per-CPU at 8 lanes): {:.2}x — {}",
        go_speedup_at_8,
        if go_speedup_at_8 >= 2.0 {
            "GO"
        } else {
            "NO-GO"
        }
    );
    println!(
        "Scope note: this is the settle-phase kernel (~72% of per-step tile evals),\n\
         not yet the full CPU step; full-step integration is Tier 2b."
    );

    Ok(())
}
