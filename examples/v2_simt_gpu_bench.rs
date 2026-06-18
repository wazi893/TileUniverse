//! SIMT Tier 3a — Gate 3: GPU lane-packed settle kernel, measured honestly.
//!
//! One CUDA thread per lane over the SAME preflighted op program the CPU
//! `LaneKernel` runs. Oracle before any timing: EVERY lane downloaded from the
//! device must equal the scalar `propagate_compact` result after the same
//! number of iterations (byte-identical). Throughput is measured
//! device-resident at K = 256 / 1024 / 4096 / 16384 — no extrapolation.
//!
//! Run: cargo run --release --features cuda --example v2_simt_gpu_bench

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::rc::Rc;
    use std::time::Instant;

    use engine::simulation::Simulation;
    use engine::tile_cpu::v2_compiler_bench::V2_COMPILED_BENCHMARKS;
    use engine::tile_cpu::v2_mmio::V2MmioHandle;
    use engine::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
    use engine::tile_cpu::v2_parser::compile_source;
    use engine::tile_cpu::v2_simt_gpu::{GpuLaneProgram, GpuLaneSettle};
    use engine::tile_cpu::{LaneKernel, V2Builder, V2SynthConfig};
    use logic_fabric_core::cuda::CudaRuntime;

    println!("SIMT Tier 3a — GPU lane-packed settle kernel (Gate 3)\n");
    let rt = CudaRuntime::new()?;

    // Real settle op program from a max-authority compiled CPU.
    let case = V2_COMPILED_BENCHMARKS
        .iter()
        .find(|c| c.name == "fact_iterative")
        .expect("fact_iterative");
    let program = compile_source(case.source).map_err(|e| e.to_string())?;
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
    for _ in 0..8 {
        cpu.step(&mut sim);
    }
    let (ops, wvia) = cpu.settle_compact_view();
    let ops = ops.to_vec();
    let wvia = wvia.to_vec();
    let kernel = LaneKernel::new_from_sim(&ops, &wvia, &sim)?;
    let gpu_prog = GpuLaneProgram::from_kernel(&kernel);
    println!(
        "program: {} ops over {} compact tile slots (full grid: {})\n",
        gpu_prog.op_count(),
        gpu_prog.n_slots,
        sim.tilemap.tile_count()
    );

    // --- ORACLE at K=256: N iterations on device == N scalar iterations. ---
    const ORACLE_ITERS: usize = 5;
    let mut scalar_sim = sim.clone();
    for _ in 0..ORACLE_ITERS {
        scalar_sim.propagate_compact(&ops, &wvia);
    }
    {
        let lanes = 256usize;
        let host = gpu_prog.lanes_from_sim(&sim, lanes);
        let mut dev = GpuLaneSettle::new(&rt, &gpu_prog, &host, lanes)?;
        dev.run(&rt, ORACLE_ITERS as i32)?;
        let out = dev.download(&rt)?;
        let mut checked = 0usize;
        for (s, &tile) in gpu_prog.touched.iter().enumerate() {
            let want = scalar_sim.tilemap.value(tile as usize);
            for lane in 0..lanes {
                assert_eq!(
                    out[s * lanes + lane],
                    want,
                    "ORACLE FAILED: tile {tile} (slot {s}) lane {lane}"
                );
            }
            checked += 1;
        }
        println!(
            "oracle PASSED: {checked} tiles x {lanes} lanes byte-identical to scalar propagate_compact\n"
        );
    }

    // --- CPU kernel reference (same box, for the honest comparison). ---
    // From Tier 2a measurements on this op array: 2.70 µs/iter/lane @ 8 lanes,
    // 1.55 @ 16 (re-quoted; CPU kernel unchanged).

    // --- Throughput sweep, device-resident. ---
    println!("lanes     iters   wall_ms   µs/iter   ns/iter/lane   G op-evals/s");
    println!("{}", "-".repeat(72));
    for &lanes in &[256usize, 1024, 4096, 16384] {
        let host = gpu_prog.lanes_from_sim(&sim, lanes);
        let mut dev = GpuLaneSettle::new(&rt, &gpu_prog, &host, lanes)?;
        // Warmup
        dev.run(&rt, 50)?;
        let iters = 2_000i32;
        let t0 = Instant::now();
        dev.run(&rt, iters)?;
        let secs = t0.elapsed().as_secs_f64();
        let per_iter = secs / iters as f64;
        println!(
            "{:>5} {:>9} {:>9.1} {:>9.2} {:>14.2} {:>14.1}",
            lanes,
            iters,
            secs * 1000.0,
            per_iter * 1e6,
            per_iter * 1e9 / lanes as f64,
            (gpu_prog.op_count() as f64 * lanes as f64) / per_iter / 1e9
        );
    }

    println!(
        "\nCPU kernel reference (Tier 2a, same op array): 2.70 µs/iter/lane @ 8 lanes,\n\
         1.55 µs/iter/lane @ 16 (= 3.55 G op-evals/s aggregate peak)."
    );
    Ok(())
}
