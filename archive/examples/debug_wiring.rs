//! Debug Wiring - Find combinational loops in CPU wiring

use engine::simulation::Simulation;
use engine::tile_cpu::TileCpuBuilder;
use engine::tile_meta::TileType;

fn main() {
    println!("Debugging CPU Wiring\n");

    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(4, 4)
        .with_program(&[0x12, 0x17, 0x31, 0x00])
        .with_rom_size(16)
        .with_ram_size(16)
        .build(&mut sim);

    println!("CPU tiles placed: {}\n", cpu.tile_count);

    // Key positions in the CPU (origin = 4, 4)
    let origin = (4usize, 4usize);

    println!("Checking key component positions:");

    // Check PC area
    println!("\nPC area (y = origin):");
    for x in origin.0..(origin.0 + 12) {
        let tt = sim.tile_type_xy(x, origin.1);
        if tt != TileType::Wire {
            println!("  ({}, {}): {:?}", x, origin.1, tt);
        }
    }

    // Check ALU area (y = origin + 8)
    println!("\nALU area (y = origin + 8):");
    let alu_y = origin.1 + 8;
    for x in origin.0..(origin.0 + 24) {
        let tt = sim.tile_type_xy(x, alu_y);
        if tt != TileType::Wire {
            println!("  ({}, {}): {:?}", x, alu_y, tt);
        }
    }

    // Check register area (y = origin + 4)
    println!("\nRegister area (y = origin + 4):");
    let reg_y = origin.1 + 4;
    for x in origin.0..(origin.0 + 24) {
        let tt = sim.tile_type_xy(x, reg_y);
        if tt != TileType::Wire {
            println!("  ({}, {}): {:?}", x, reg_y, tt);
        }
    }

    // Check branch logic area (y = origin + 11)
    println!("\nBranch logic area (y = origin + 11):");
    let branch_y = origin.1 + 11;
    for x in origin.0..(origin.0 + 28) {
        let tt = sim.tile_type_xy(x, branch_y);
        if tt != TileType::Wire {
            println!("  ({}, {}): {:?}", x, branch_y, tt);
        }
    }

    // Initialize and check propagation
    println!("\n\nInitializing simulation...");
    let init_stats = sim.initialize();
    println!("  Init: {} deltas, critical path {}", init_stats.total_deltas, init_stats.critical_path_deltas);

    // Run a few ticks
    println!("\nRunning ticks:");
    for i in 0..5 {
        let stats = sim.tick_with_delays();
        println!("  Tick {}: {} deltas, converged={}", i, stats.total_deltas, stats.converged);
        if !stats.converged {
            println!("    WARNING: Non-convergence detected!");
            break;
        }
    }

    // Simple test: minimal circuit
    println!("\n\n=== Minimal Circuit Test ===");
    let mut sim2 = Simulation::with_size(32, 32);

    // Just PC with clock - this should work
    sim2.set_tile(10, 9, TileType::ClockGlobal);  // Clock above
    sim2.set_tile(10, 10, TileType::ProgramCounter);
    sim2.set_tile(11, 10, TileType::Const);  // Right input = 0 (no jump)
    sim2.set_logic_value(11, 10, 0);

    sim2.initialize();
    println!("PC + Clock initialized, PC = {}", sim2.get_logic_at(10, 10));

    for i in 0..5 {
        let stats = sim2.tick_with_delays();
        println!("  Tick {}: {} deltas, PC = {}", i, stats.total_deltas, sim2.get_logic_at(10, 10));
    }

    // Test with feedback loop (intentional)
    println!("\n=== Intentional Feedback Loop Test ===");
    let mut sim3 = Simulation::with_size(32, 32);

    // Create a ring of wires - this WILL oscillate
    sim3.set_tile(10, 10, TileType::Wire);
    sim3.set_tile(11, 10, TileType::Wire);
    sim3.set_tile(12, 10, TileType::Wire);
    sim3.set_tile(12, 11, TileType::Wire);
    sim3.set_tile(11, 11, TileType::Wire);
    sim3.set_tile(10, 11, TileType::Wire);

    // Inject a signal
    sim3.set_logic_value(10, 10, 1);

    sim3.mark_all_dirty();
    let stats = sim3.tick_with_delays();
    println!("Wire ring: {} deltas, converged={}", stats.total_deltas, stats.converged);
    println!("  (This should converge because wires OR together and stabilize)");

    // Test with NOT gate in the loop - this WILL oscillate
    println!("\n=== NOT Gate Oscillator Test ===");
    let mut sim4 = Simulation::with_size(32, 32);

    // NOT -> Wire -> feeds back to NOT
    sim4.set_tile(10, 10, TileType::Not);
    sim4.set_tile(11, 10, TileType::Wire);
    sim4.set_tile(11, 11, TileType::Wire);
    sim4.set_tile(10, 11, TileType::Wire);

    sim4.mark_all_dirty();
    let stats = sim4.tick_with_delays();
    println!("NOT oscillator: {} deltas, converged={}", stats.total_deltas, stats.converged);
    if !stats.converged {
        println!("  As expected - NOT gate feedback creates oscillation!");
    }

    println!("\n\nDiagnosis:");
    println!("The CPU wiring likely has an unintended NOT or comparison tile");
    println!("that creates an inverting feedback loop. Check:");
    println!("  1. Branch logic (has NOT for JNZ condition)");
    println!("  2. Zero flag feeding back to ALU area");
    println!("  3. Wire overlaps connecting incompatible signals");
}
