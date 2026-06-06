//! TileFabric CPU Demo
//!
//! Demonstrates a CPU that executes by propagating signals through tiles.
//! This is NOT a software emulator - every gate, register, and wire is a
//! discrete tile evaluated by the simulation engine.
//!
//! Run with: cargo run --example tile_cpu_demo --release

use engine::simulation::Simulation;
use engine::tile_cpu::{TileCpuBuilder, TileCpuDatapath};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║          TileFabric CPU - Tile-Based Processor Demo           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // Part 1: Architecture Analysis
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ PART 1: Datapath Analysis                                       │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let datapath = TileCpuDatapath::single_cycle();
    let analysis = datapath.analyze();
    println!("{}", analysis);

    let pipelined = TileCpuDatapath::two_stage_pipeline();
    println!(
        "Pipeline stages: {} → critical path {} deltas",
        pipelined.pipeline_stages,
        pipelined.total_critical_path()
    );
    println!();

    // =========================================================================
    // Part 2: CPU Construction
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ PART 2: CPU Construction                                        │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    // Create a simulation grid large enough for the CPU
    let mut sim = Simulation::with_size(64, 64);

    // Program: Simple addition
    // LDI R0, #2    ; R0 = 2
    // LDI R1, #3    ; R1 = 3
    // ADD R0, R1    ; R0 = R0 + R1 = 5
    // NOP           ; (halt - infinite NOPs)
    let program = vec![
        0x12, // LDI R0, #2 (opcode 1, rd=0, imm=2)
        0x17, // LDI R1, #3 (opcode 1, rd=1, imm=3)
        0x31, // ADD R0, R1 (opcode 3, rd=0, rs=1)
        0x00, // NOP
    ];

    println!("Program:");
    println!("  0x00: LDI R0, #2    ; Load immediate 2 into R0");
    println!("  0x01: LDI R1, #3    ; Load immediate 3 into R1");
    println!("  0x02: ADD R0, R1    ; R0 = R0 + R1");
    println!("  0x03: NOP           ; No operation");
    println!();

    // Build the CPU with full wiring
    let cpu = TileCpuBuilder::new()
        .with_origin(4, 4)
        .with_program(&program)
        .with_rom_size(16)
        .with_ram_size(16)
        .build(&mut sim); // Full wired datapath

    println!("CPU constructed:");
    println!("  Origin: ({}, {})", cpu.origin.0, cpu.origin.1);
    println!("  Tiles placed: {}", cpu.tile_count);
    println!("  ROM size: {} bytes", 16);
    println!("  RAM size: {} bytes", 16);
    println!();

    // Initialize: evaluate all tiles to establish signal values
    println!("Initializing (evaluating all tiles)...");
    let init_stats = sim.initialize();
    println!(
        "  Initialization: {} deltas, critical path {} deltas",
        init_stats.total_deltas, init_stats.critical_path_deltas
    );
    println!();

    // =========================================================================
    // Part 3: Execution via Tile Simulation
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ PART 3: Execution via Tile Simulation                           │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!("After initialization:");
    println!("  {}", cpu.dump_state(&sim));
    println!();

    // Execute cycles and observe
    println!("Executing cycles (calling sim.tick_with_delays()):");
    println!();

    for cycle in 0..6 {
        let stats = cpu.tick(&mut sim);

        println!(
            "Cycle {}: {} deltas, critical path {} deltas",
            cycle, stats.total_deltas, stats.critical_path_deltas
        );
        println!("  State: {}", cpu.dump_state(&sim));

        if !stats.converged {
            println!("  WARNING: Timing did not converge!");
        }
    }
    println!();

    // =========================================================================
    // Part 4: Timing Analysis
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ PART 4: Timing Analysis                                         │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    // Run for more cycles to gather metrics
    let metrics = cpu.run(&mut sim, 100);
    println!("{}", metrics);

    // Check timing at various frequencies
    for target_mhz in &[100.0, 50.0, 25.0, 10.0] {
        let meets = cpu.check_timing(&sim, *target_mhz);
        println!(
            "  {} MHz: {}",
            target_mhz,
            if meets { "PASS" } else { "FAIL" }
        );
    }
    println!();

    // =========================================================================
    // Part 5: Critical Path
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ PART 5: Critical Path Analysis                                  │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let critical_path = cpu.analyze_critical_path(&sim);
    if critical_path.is_empty() {
        println!("No critical path recorded (circuit stable)");
    } else {
        println!("Critical path tiles: {} tiles", critical_path.len());
        for (i, idx) in critical_path.iter().take(10).enumerate() {
            let x = idx % sim.width();
            let y = idx / sim.width();
            let tile_type = sim.tile_type_xy(x, y);
            println!("  {}: ({}, {}) - {:?}", i, x, y, tile_type);
        }
        if critical_path.len() > 10 {
            println!("  ... and {} more", critical_path.len() - 10);
        }
    }
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ SUMMARY                                                         │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("The TileFabric CPU differs from software emulators:");
    println!();
    println!("  SOFTWARE EMULATOR          TILEFABRIC CPU");
    println!("  ─────────────────          ──────────────");
    println!("  match opcode {{             sim.tick_with_delays()");
    println!("    ADD => a + b             // Signals propagate through");
    println!("  }}                          // actual Add/Sub/And tiles");
    println!();
    println!("  No timing model            Gate-accurate timing");
    println!(
        "  Instant execution          {} deltas per cycle",
        metrics.max_critical_path
    );
    println!(
        "  N/A                        {:.1} MHz estimated max",
        metrics.estimated_max_mhz
    );
    println!();
    println!("This is the foundation for:");
    println!("  • AI-guided tile placement optimization (AlphaChip-style)");
    println!("  • Visual debugging of signal propagation");
    println!("  • Hardware timing verification");
    println!("  • Export to Verilog/FPGA");
}
