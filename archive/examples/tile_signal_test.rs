//! Tile Signal Propagation Test
//!
//! Tests basic signal flow through tiles to verify the simulation works.
//! This helps debug why the CPU tiles aren't propagating signals.

use engine::simulation::Simulation;
use engine::tile_meta::TileType;

fn main() {
    println!("Tile Signal Propagation Test\n");

    // =========================================================================
    // Test 1: Simple wire propagation
    // =========================================================================
    println!("Test 1: Wire chain propagation");
    println!("─────────────────────────────────");

    let mut sim = Simulation::with_size(16, 16);

    // Create a chain: Const -> Wire -> Wire -> Wire
    //                 (5,5)   (6,5)   (7,5)   (8,5)
    sim.set_tile(5, 5, TileType::Const);
    sim.set_logic_value(5, 5, 42);  // Set constant to 42

    sim.set_tile(6, 5, TileType::Wire);
    sim.set_tile(7, 5, TileType::Wire);
    sim.set_tile(8, 5, TileType::Wire);

    println!("Before initialization:");
    println!("  Const(5,5) = {}", sim.get_logic_at(5, 5));
    println!("  Wire(6,5)  = {}", sim.get_logic_at(6, 5));
    println!("  Wire(7,5)  = {}", sim.get_logic_at(7, 5));
    println!("  Wire(8,5)  = {}", sim.get_logic_at(8, 5));

    // Initialize: mark all tiles dirty and evaluate
    let stats = sim.initialize();
    println!("\nAfter tick ({} deltas, critical path {}):",
        stats.total_deltas, stats.critical_path_deltas);
    println!("  Const(5,5) = {}", sim.get_logic_at(5, 5));
    println!("  Wire(6,5)  = {}", sim.get_logic_at(6, 5));
    println!("  Wire(7,5)  = {}", sim.get_logic_at(7, 5));
    println!("  Wire(8,5)  = {}", sim.get_logic_at(8, 5));
    println!();

    // =========================================================================
    // Test 2: Add tile
    // =========================================================================
    println!("Test 2: Add tile operation");
    println!("─────────────────────────────────");

    let mut sim2 = Simulation::with_size(16, 16);

    // Create: Const(10) -> Add <- Const(20)
    //                (5,5)  (6,5)  (7,5)
    sim2.set_tile(5, 5, TileType::Const);
    sim2.set_logic_value(5, 5, 10);

    sim2.set_tile(6, 5, TileType::Add);

    sim2.set_tile(7, 5, TileType::Const);
    sim2.set_logic_value(7, 5, 20);

    println!("Before initialization:");
    println!("  Left(5,5)  = {}", sim2.get_logic_at(5, 5));
    println!("  Add(6,5)   = {}", sim2.get_logic_at(6, 5));
    println!("  Right(7,5) = {}", sim2.get_logic_at(7, 5));

    let stats2 = sim2.initialize();
    println!("\nAfter tick ({} deltas):", stats2.total_deltas);
    println!("  Left(5,5)  = {}", sim2.get_logic_at(5, 5));
    println!("  Add(6,5)   = {} (expected 30)", sim2.get_logic_at(6, 5));
    println!("  Right(7,5) = {}", sim2.get_logic_at(7, 5));
    println!();

    // =========================================================================
    // Test 3: Register capture on clock edge
    // =========================================================================
    println!("Test 3: Register8 clock edge capture");
    println!("─────────────────────────────────────");

    let mut sim3 = Simulation::with_size(16, 16);

    // Create: Const -> Register8
    sim3.set_tile(5, 5, TileType::Const);
    sim3.set_logic_value(5, 5, 99);

    sim3.set_tile(6, 5, TileType::Register8);

    println!("Before tick:");
    println!("  Input(5,5)    = {}", sim3.get_logic_at(5, 5));
    println!("  Register(6,5) = {}", sim3.get_logic_at(6, 5));
    println!("  Clock: {}", sim3.global_clock);

    let stats3 = sim3.tick_with_delays();
    println!("\nAfter tick 1 ({} deltas):", stats3.total_deltas);
    println!("  Input(5,5)    = {}", sim3.get_logic_at(5, 5));
    println!("  Register(6,5) = {} (captures on rising edge)", sim3.get_logic_at(6, 5));
    println!("  Clock: {}", sim3.global_clock);

    let stats4 = sim3.tick_with_delays();
    println!("\nAfter tick 2 ({} deltas):", stats4.total_deltas);
    println!("  Register(6,5) = {}", sim3.get_logic_at(6, 5));
    println!("  Clock: {}", sim3.global_clock);
    println!();

    // =========================================================================
    // Test 4: Program Counter
    // =========================================================================
    println!("Test 4: ProgramCounter increment");
    println!("─────────────────────────────────────");

    let mut sim4 = Simulation::with_size(16, 16);

    // Clock signal above PC (up input)
    sim4.set_tile(5, 4, TileType::ClockGlobal);

    // Program Counter
    sim4.set_tile(5, 5, TileType::ProgramCounter);
    sim4.set_logic_value(5, 5, 0);

    // Right input (jump enable) = 0, so PC should increment
    sim4.set_tile(6, 5, TileType::Const);
    sim4.set_logic_value(6, 5, 0);

    // Initialize
    sim4.initialize();

    println!("After initialization:");
    println!("  Clock(5,4) = {}", sim4.get_logic_at(5, 4));
    println!("  PC(5,5)    = {}", sim4.get_logic_at(5, 5));

    for i in 0..5 {
        sim4.tick_with_delays();
        println!("  After tick {}: Clock={} PC = {}",
            i + 1,
            if sim4.global_clock { "HIGH" } else { "LOW " },
            sim4.get_logic_at(5, 5));
    }
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("Summary");
    println!("─────────────────────────────────────");
    println!("If tests show expected values, tile simulation is working.");
    println!("If values stay 0, there's an issue with tile adjacency or dirty marking.");
}
