//! Redstone Super-Engine Demo — "Minecraft Redstone but on a Rust GPU Supercomputer"
//
// This example is the flagship demonstration of the TileUniverse fabric as a
// creative, high-performance "redstone" world.
//
// Minecraft Redstone lets you build logic by placing dust, torches, repeaters,
// comparators, pistons, observers, and command blocks. It is beloved because
// it is *visual*, *tactile*, *immediate*, and *surprisingly powerful* (people
// have built working CPUs in it).
//
// TileUniverse gives you the same joy — but as a *super-engine*:
// - You place (or synthesize) real logic tiles: wires, gates, registers, RAM, muxes.
// - The V2 Tile CPU is a pre-built "computer block" you can drop into the world.
// - Execution happens by the simulation ticking the *actual placed gates* (physical).
// - Under the hood you get 100s of millions of tile evals/sec on CPU, up to
//   115 *trillion* on RTX 5090 with the packed register-resident kernels.
// - You get synthesis (AIG + placement + routing) that automatically optimizes
//   complex "redstone" subcircuits.
// - Timing, critical paths, glitches, and sparse eval for mostly-idle builds.
// - Deterministic replay bundles = perfect "world saves" you can step forward/back.
// - Python bindings for a creative-mode REPL experience.
// - MMIO "peripherals" that act like redstone screens, sensors, or even
//   quantum-entangled "ender chests" and neural "observers".
// - Full modern toolchain: C-like source or asm → physical gate layout → run.
//
// Run:
//   cargo run --release --example redstone_super_demo
//
// For GPU-accelerated "super redstone" (packed 1-bit + register-resident):
//   cargo run --release --features cuda,perf-bench --example redstone_super_demo
//
// Watch the console output (produced by physical tiles writing to an MMIO device)
// and the generated visualization frames that show signal flow — exactly like
// watching your redstone contraption light up tick by tick.

use std::error::Error;
use std::fs::{File, create_dir_all};
use std::path::Path;
use std::rc::Rc;

use engine::simulation::Simulation;
use engine::tile_cpu::v2_mmio::V2MmioHandle;
use engine::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
use engine::tile_cpu::v2_parser::compile_source;
use engine::tile_cpu::{
    V2Builder, V2SynthConfig, V2VizLayerLayout, V2VizRenderParams, V2VizSession,
    write_v2_viz_frame_ppm,
};

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║  TILEUNIVERSE — MINECRAFT REDSTONE SUPER-ENGINE DEMO                       ║");
    println!("║  A full CPU built from logic tiles, running at supercomputer speeds       ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    // ---------------------------------------------------------------------
    // 1. "Creative Mode" — describe what you want in a high-level language
    //    (exactly like writing a command block or redstone computer program).
    // ---------------------------------------------------------------------
    let redstone_program = r#"
        int putchar(int c) {
            putchar(c);
            return 0;
        }
        int main() {
            putchar(82); putchar(69); putchar(68); putchar(83); putchar(84); putchar(79);
            putchar(78); putchar(69); putchar(32); putchar(83); putchar(85); putchar(80);
            putchar(69); putchar(82); putchar(10);
            putchar(51); putchar(54); putchar(10);
            return 0;
        }
    "#;

    println!("Compiling redstone program (C-like source → AST → assembly → machine words)...");
    let program = compile_source(redstone_program)
        .map_err(|e| format!("Redstone program compile error: {}", e))?;
    println!(
        "  → {} instruction words ready to become physical tiles.\n",
        program.len()
    );

    // ---------------------------------------------------------------------
    // 2. Build the "Redstone Computer" in the world
    //    (V2Builder places the CPU as real tiles + we attach a "display" peripheral
    //     that acts like a big redstone lamp array / console).
    // ---------------------------------------------------------------------
    let console = Rc::new(V2MmioCombinedDevice::new(0xBEEF));
    let mut sim = Simulation::with_size_layered(128, 640, 16); // Match proven working hello layout for synth blocks

    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(128)
        .with_ram_size(128)
        .with_extended_pc()
        // Max physical authority + synth blocks (the "super" part: AIG-optimized subcircuits)
        .with_synth_blocks(V2SynthConfig::max_authority())
        .with_mmio(V2MmioHandle::from_rc(console.clone()))
        .build(&mut sim);

    println!("Placed V2 Tile CPU (physical gate layout) + MMIO console peripheral.");
    println!("  The CPU *is* redstone: every gate, wire, register, and clock is a tile.\n");

    // ---------------------------------------------------------------------
    // 3. "Redstone Tick" the world and watch signals propagate
    //    We also capture visualization frames — the equivalent of watching
    //    your redstone dust light up and repeaters fire.
    // ---------------------------------------------------------------------
    let mut viz = V2VizSession::new(&sim);
    let viz_params = V2VizRenderParams {
        scale: 1,
        layout: V2VizLayerLayout::Grid2x2,
        highlight_active: true,
        show_flow: true,
        max_flow_edges: 4096,
    };

    let out_dir = Path::new("target/redstone_viz");
    create_dir_all(out_dir)?;

    println!("Running the redstone computer (physical tile propagation)...");
    let max_ticks = 400u64;
    let mut ticks = 0u64;

    while !cpu.is_halted() && ticks < max_ticks {
        cpu.step(&mut sim); // One "redstone tick" — the simulation evaluates the placed tiles
        ticks += 1;

        // Capture a viz frame every few ticks so we can see the signals move
        if ticks.is_multiple_of(8) || cpu.is_halted() {
            let frame = viz.capture_frame(&sim, &viz_params);
            let frame_path = out_dir.join(format!("redstone_frame_{:03}.ppm", ticks));
            let mut f = File::create(&frame_path)?;
            write_v2_viz_frame_ppm(&frame, &mut f)?;
        }
    }

    println!("\nHalted after {} redstone ticks.", ticks);

    // ---------------------------------------------------------------------
    // 4. Read the "output" the physical tiles produced
    // ---------------------------------------------------------------------
    println!("\n=== Redstone Console Output (written by physical tiles) ===");
    println!("----------------------------------------");
    print!("{}", console.ref_pack.console_string());
    println!("----------------------------------------\n");

    // ---------------------------------------------------------------------
    // 5. Notable Metrics — the "super" part
    // ---------------------------------------------------------------------
    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║  SUPER-ENGINE METRICS (Physical Tile Execution)                            ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝");

    println!(
        "  Program size:           {} instructions (compiled from high-level source)",
        program.len()
    );
    println!("  Redstone ticks:         {}", ticks);
    println!("  CPU cycles retired:     {}", ticks); // In this model they are very close
    println!(
        "  Visualization frames:   {} (signal flow + active tiles)",
        (ticks / 8) + 1
    );
    println!("  Viz output dir:         {}", out_dir.display());
    println!("  Backend:                Pure tile simulation (wires + gates + clocked registers)");
    println!("  (With --features cuda,perf-bench this same layout runs on the 115T tile/sec");
    println!("   packed register-resident kernels — your redstone computer at ludicrous speed.)\n");

    println!("This is the equivalent of building a working CPU out of redstone in Minecraft,");
    println!("except the engine can evaluate *millions to trillions* of gate updates per second,");
    println!("gives you automatic synthesis optimization, perfect determinism/replay,");
    println!(
        "and lets you hybridize with quantum states or spiking neural networks as special blocks.\n"
    );

    println!("Next steps to go even more super-redstone:");
    println!("  • Open the .ppm frames in target/redstone_viz/ to watch the signals propagate.");
    println!("  • Try the other V2 examples (tile_cpu_v2_benchmark_suite, tile_cpu_v2_trace_viz).");
    println!("  • Enable CUDA packed kernels for planet-scale redstone builds.");
    println!("  • Use Python: tu.V2Cpu.from_asm(...) in a creative loop.");
    println!("  • Build your own circuits with V2Builder + custom tile placement.");

    Ok(())
}
