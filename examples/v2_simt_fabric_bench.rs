//! SIMT Tier 2b — Gate 2b: K full V2 CPUs in lane-lockstep, measured honestly.
//!
//! Three measured configurations, same program (SPMD), same build flags:
//!   A. Tier-0 style scalar: K independent builds        (the original baseline)
//!      — quoted from the Tier-0 report (12,026 cyc/s step-only), not re-run.
//!   B. Build-amortized scalar: ONE build, K (cpu+sim) clones stepped
//!      sequentially — the honest stronger baseline (Stage-1 amortization).
//!   C. V2SimtFabric: ONE build, K lanes lockstepped with the wide settle.
//!
//! ORACLE before any timing is reported: every fabric lane's final state hash
//! must equal the scalar reference's hash for the same program (byte-identical
//! `hash_v2_final_state`), and R0 must match the corpus expectation.
//!
//! Run: cargo run --release --example v2_simt_fabric_bench [lanes]

use std::error::Error;
use std::rc::Rc;
use std::time::Instant;

use engine::simulation::Simulation;
use engine::tile_cpu::v2_compiler_bench::V2_COMPILED_BENCHMARKS;
use engine::tile_cpu::v2_mmio::V2MmioHandle;
use engine::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
use engine::tile_cpu::v2_parser::compile_source;
use engine::tile_cpu::{TileCpuV2, V2Builder, V2SimtFabric, V2SynthConfig, hash_v2_final_state};

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

fn run_to_halt(cpu: &TileCpuV2, sim: &mut Simulation, max_cycles: u64) -> u64 {
    let mut cycles = 0u64;
    for _ in 0..max_cycles {
        if cpu.is_halted() {
            break;
        }
        cpu.step(sim);
        cycles += 1;
    }
    cycles
}

fn main() -> Result<(), Box<dyn Error>> {
    let lanes: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    println!("SIMT Tier 2b — Gate 2b fabric benchmark ({lanes} lanes)\n");

    let case = V2_COMPILED_BENCHMARKS
        .iter()
        .find(|c| c.name == "fact_iterative")
        .expect("fact_iterative in corpus");
    let program = compile_source(case.source).map_err(|e| e.to_string())?;

    // --- Template build (paid once for B and C). ---
    let t0 = Instant::now();
    let (template_cpu, template_sim) = build_cpu(&program);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("topology build: {build_ms:.0} ms (once)\n");

    // --- Scalar reference + oracle hash. ---
    let ref_cpu = template_cpu.clone();
    let mut ref_sim = template_sim.clone();
    let ref_cycles = run_to_halt(&ref_cpu, &mut ref_sim, case.max_cycles);
    let golden = hash_v2_final_state(&ref_cpu, &ref_sim);
    assert_eq!(
        ref_cpu.read_reg(&ref_sim, 0),
        case.expected_r0,
        "scalar reference wrong — bad oracle"
    );
    println!(
        "scalar reference: {} cycles, golden hash {:#018x}",
        ref_cycles, golden
    );

    // --- B: build-amortized scalar clones, stepped sequentially. ---
    let mut clone_pairs: Vec<(TileCpuV2, Simulation)> = (0..lanes)
        .map(|_| (template_cpu.clone(), template_sim.clone()))
        .collect();
    let t0 = Instant::now();
    let mut b_cycles = 0u64;
    for (cpu, sim) in clone_pairs.iter_mut() {
        b_cycles += run_to_halt(cpu, sim, case.max_cycles);
    }
    let b_secs = t0.elapsed().as_secs_f64();
    for (cpu, sim) in clone_pairs.iter() {
        assert_eq!(hash_v2_final_state(cpu, sim), golden, "clone diverged");
    }
    let b_rate = b_cycles as f64 / b_secs;
    println!(
        "B amortized scalar clones: {} lanes x {} cycles in {:.3} s = {:>8.0} cyc/s",
        lanes, ref_cycles, b_secs, b_rate
    );

    // --- C: the fabric. ---
    let mut fabric = V2SimtFabric::new(template_cpu, template_sim, lanes)?;
    let t0 = Instant::now();
    let c_cycles = fabric.run_to_halt(case.max_cycles);
    let c_secs = t0.elapsed().as_secs_f64();

    // ORACLE: every lane byte-identical to the scalar golden.
    for lane in 0..lanes {
        let h = fabric.lane_state_hash(lane);
        assert_eq!(
            h, golden,
            "ORACLE FAILED: lane {lane} hash {h:#018x} != scalar golden"
        );
        assert_eq!(fabric.lane_reg(lane, 0), case.expected_r0, "lane {lane} R0");
    }
    println!("C fabric ORACLE PASSED: all {lanes} lanes byte-identical to scalar golden");
    let c_rate = c_cycles as f64 / c_secs;
    println!(
        "C fabric lockstep:        {} lanes x {} cycles in {:.3} s = {:>8.0} cyc/s  (wide settles: {}, fallback: {})",
        lanes, ref_cycles, c_secs, c_rate, fabric.wide_settles, fabric.fallback_settles
    );

    println!("\nGate 2b (CPUs-advanced/sec, step-only):");
    println!("  Tier-0 baseline (K independent builds, quoted): 12,026 cyc/s");
    println!("  B amortized scalar:  {b_rate:>9.0} cyc/s");
    println!(
        "  C fabric:            {c_rate:>9.0} cyc/s   ({:.2}x vs B)",
        c_rate / b_rate
    );

    Ok(())
}
