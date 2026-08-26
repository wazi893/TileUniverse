//! EPIC 123: Propagation Delay Timing Analysis Demo
//!
//! Demonstrates the new timing-aware simulation capabilities:
//! - Critical path analysis
//! - Timing verification
//! - Race condition detection
//!
//! Run with:
//!   cargo run --example timing_analysis_demo --release

use engine::simulation::Simulation;
use engine::tile_meta::TileType;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 123: PROPAGATION DELAY TIMING ANALYSIS                  ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // Demo 1: Simple Chain - Critical Path Analysis
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEMO 1: Critical Path in a Logic Chain");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut sim = Simulation::with_size(32, 8);

    // Build: ClockGlobal -> Wire -> Wire -> And -> Wire -> Or -> Wire -> Register
    // Delays:     0          1       1      2       1      2       1        0
    // Cumulative: 0          1       2      4       5      7       8        8

    sim.set_tile(0, 0, TileType::ClockGlobal);
    sim.set_tile(1, 0, TileType::Wire);
    sim.set_tile(2, 0, TileType::Wire);
    sim.set_tile(3, 0, TileType::And);
    sim.set_tile(4, 0, TileType::Wire);
    sim.set_tile(5, 0, TileType::Or);
    sim.set_tile(6, 0, TileType::Wire);
    sim.set_tile(7, 0, TileType::Register8);

    // Add second input to And gate
    sim.set_tile(3, 1, TileType::Wire);
    sim.tilemap.set_value_at(3, 1, u64::MAX);

    println!("Circuit: Clock -> Wire -> Wire -> AND -> Wire -> OR -> Wire -> Register");
    println!("Delays:    0       1       1       2       1      2       1        0");
    println!();

    // Run timing-aware simulation
    let stats = sim.tick_with_delays();

    println!("{}", stats);

    // Trace critical path
    let path = sim.trace_critical_path();
    if !path.is_empty() {
        println!("Critical Path Trace:");
        for (i, &idx) in path.iter().enumerate() {
            let x = idx % sim.width();
            let y = idx / sim.width();
            let tile_type = sim.tilemap.tiles[idx].meta.tile_type;
            let delay = tile_type.propagation_delay();
            println!(
                "  {}. ({:2}, {:2}) {:?} +{} deltas",
                i + 1,
                x,
                y,
                tile_type,
                delay
            );
        }
    }
    println!();

    // =========================================================================
    // Demo 2: Complex Arithmetic - Longer Critical Path
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEMO 2: Arithmetic Circuit (MUL + DIV)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut sim2 = Simulation::with_size(32, 8);

    // Clock -> Wire -> MUL -> Wire -> DIV -> Wire -> Register
    // Delays: 0      1      8       1      12      1       0
    sim2.set_tile(0, 0, TileType::ClockGlobal);
    sim2.set_tile(1, 0, TileType::Wire);
    sim2.set_tile(2, 0, TileType::Mul);
    sim2.set_tile(3, 0, TileType::Wire);
    sim2.set_tile(4, 0, TileType::Div);
    sim2.set_tile(5, 0, TileType::Wire);
    sim2.set_tile(6, 0, TileType::Register8);

    // Inputs for MUL and DIV
    sim2.set_tile(2, 1, TileType::Const);
    sim2.set_tile(4, 1, TileType::Const);
    sim2.tilemap.set_value_at(2, 1, 7);
    sim2.tilemap.set_value_at(4, 1, 3);

    println!("Circuit: Clock -> Wire -> MUL -> Wire -> DIV -> Wire -> Register");
    println!("Delays:    0       1       8       1      12       1        0");
    println!("Expected critical path: ~23 deltas");
    println!();

    let stats2 = sim2.tick_with_delays();
    println!("{}", stats2);

    // Check timing against different targets
    println!("Timing Checks:");
    for target in [10, 20, 25, 30, 50] {
        let result = sim2.check_timing(target);
        let status = if result.meets_timing { "PASS" } else { "FAIL" };
        println!(
            "  Target {:2} deltas: {} (slack: {:+3})",
            target, status, result.slack
        );
    }
    println!();

    // =========================================================================
    // Demo 3: Parallel Paths - Race Detection
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEMO 3: Race Condition Detection");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut sim3 = Simulation::with_size(16, 8);

    // Create two paths of different lengths converging at an AND gate
    //
    // Short path: Clock(0,0) -> Wire(1,0) -> Wire(2,0) -> AND(3,0)
    // Long path:  Clock(0,0) -> Wire(0,1) -> Wire(1,1) -> Wire(2,1) -> Wire(3,1) -> AND(3,0)
    //
    // Short: 0 + 1 + 1 = 2 deltas
    // Long:  0 + 1 + 1 + 1 + 1 = 4 deltas
    // Race window: 2 deltas

    sim3.set_tile(0, 0, TileType::ClockGlobal);

    // Short path
    sim3.set_tile(1, 0, TileType::Wire);
    sim3.set_tile(2, 0, TileType::Wire);

    // Long path
    sim3.set_tile(0, 1, TileType::Wire);
    sim3.set_tile(1, 1, TileType::Wire);
    sim3.set_tile(2, 1, TileType::Wire);
    sim3.set_tile(3, 1, TileType::Wire);

    // Convergence point
    sim3.set_tile(3, 0, TileType::And);

    println!("Circuit with two paths to AND gate:");
    println!("  Short: Clock -> Wire -> Wire -> AND  (2 deltas)");
    println!("  Long:  Clock -> Wire -> Wire -> Wire -> Wire -> AND  (4 deltas)");
    println!();

    let stats3 = sim3.tick_with_delays();
    println!("{}", stats3);

    // Detect races
    let races = sim3.detect_races(1);
    if races.is_empty() {
        println!("No race conditions detected with window >= 1 delta");
    } else {
        println!("Race Conditions Detected:");
        for race in &races {
            println!(
                "  Tile ({}, {}): early={}, late={}, window={} deltas",
                race.x, race.y, race.early_arrival, race.late_arrival, race.race_window
            );
        }
    }
    println!();

    // =========================================================================
    // Demo 4: Delay Value Reference
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("  REFERENCE: Propagation Delays by Tile Type");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let tile_types = [
        TileType::Wire,
        TileType::WireH,
        TileType::WireV,
        TileType::And,
        TileType::Or,
        TileType::Not,
        TileType::Xor,
        TileType::Add,
        TileType::Sub,
        TileType::Mul,
        TileType::Div,
        TileType::Lt,
        TileType::Eq,
        TileType::Mux,
        TileType::Register8,
        TileType::Latch,
        TileType::ClockGlobal,
    ];

    println!("  {:20} {:>8}  {:>10}", "Tile Type", "Delay", "Sequential");
    println!(
        "  {:20} {:>8}  {:>10}",
        "-".repeat(20),
        "-".repeat(8),
        "-".repeat(10)
    );
    for tt in tile_types {
        println!(
            "  {:20} {:>8}  {:>10}",
            format!("{:?}", tt),
            tt.propagation_delay(),
            if tt.is_sequential() { "yes" } else { "no" }
        );
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
}
