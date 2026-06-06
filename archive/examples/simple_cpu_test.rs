//! Simple CPU Test - Step by step component verification
//!
//! Run with: cargo run --example simple_cpu_test --release

use engine::simulation::Simulation;
use engine::tile_meta::TileType;

fn main() {
    println!("Simple CPU Component Tests\n");

    // =========================================================================
    // Test 1: Basic wire propagation
    // =========================================================================
    println!("Test 1: Wire propagation");
    println!("────────────────────────");
    {
        let mut sim = Simulation::with_size(16, 16);

        // Const -> Wire -> Wire -> destination
        // All on same row, using omnidirectional Wire
        sim.set_tile(2, 5, TileType::Const);
        sim.set_logic_value(2, 5, 42);
        sim.set_tile(3, 5, TileType::Wire);
        sim.set_tile(4, 5, TileType::Wire);
        sim.set_tile(5, 5, TileType::Wire);

        let stats = sim.initialize();
        println!("  Init: {} deltas", stats.total_deltas);
        println!("  Const(2,5) = {}", sim.get_logic_at(2, 5));
        println!("  Wire(3,5)  = {}", sim.get_logic_at(3, 5));
        println!("  Wire(4,5)  = {}", sim.get_logic_at(4, 5));
        println!("  Wire(5,5)  = {}", sim.get_logic_at(5, 5));

        if sim.get_logic_at(5, 5) == 42 {
            println!("  PASS: Signal propagated!");
        } else {
            println!("  FAIL: Signal didn't propagate!");
        }
    }
    println!();

    // =========================================================================
    // Test 2: Register8 capture
    // =========================================================================
    println!("Test 2: Register8 capture");
    println!("─────────────────────────");
    {
        let mut sim = Simulation::with_size(16, 16);

        // Layout:
        //   Clock at (5, 4) - above register
        //   Const(99) at (4, 5) - left of register (data input)
        //   Register8 at (5, 5)

        sim.set_tile(5, 4, TileType::ClockGlobal);
        sim.set_tile(4, 5, TileType::Const);
        sim.set_logic_value(4, 5, 99);
        sim.set_tile(5, 5, TileType::Register8);

        let stats = sim.initialize();
        println!("  Init: {} deltas, Register = {}", stats.total_deltas, sim.get_logic_at(5, 5));

        // Run ticks - register should capture on rising clock edge
        for i in 0..4 {
            let stats = sim.tick_with_delays();
            println!(
                "  Tick {}: clock={} Register={} ({} deltas)",
                i,
                if sim.global_clock { "HIGH" } else { "LOW" },
                sim.get_logic_at(5, 5),
                stats.total_deltas
            );
        }
    }
    println!();

    // =========================================================================
    // Test 3: ProgramCounter behavior
    // =========================================================================
    println!("Test 3: ProgramCounter increment");
    println!("─────────────────────────────────");
    {
        let mut sim = Simulation::with_size(16, 16);

        // Layout:
        //   Clock at (5, 4) - above PC (up input is clock)
        //   PC at (5, 5)
        //   Const(0) at (6, 5) - right of PC (jump enable = 0, so increment)

        sim.set_tile(5, 4, TileType::ClockGlobal);
        sim.set_tile(5, 5, TileType::ProgramCounter);
        sim.set_logic_value(5, 5, 0); // Start at 0
        sim.set_tile(6, 5, TileType::Const);
        sim.set_logic_value(6, 5, 0); // Jump enable = 0

        let stats = sim.initialize();
        println!("  Init: {} deltas, PC = {}", stats.total_deltas, sim.get_logic_at(5, 5));

        for i in 0..8 {
            let stats = sim.tick_with_delays();
            println!(
                "  Tick {}: clock={} PC={} ({} deltas)",
                i,
                if sim.global_clock { "HIGH" } else { "LOW" },
                sim.get_logic_at(5, 5),
                stats.total_deltas
            );
        }
    }
    println!();

    // =========================================================================
    // Test 4: ROM selection with Mux8to1
    // =========================================================================
    println!("Test 4: ROM with Mux8to1");
    println!("────────────────────────");
    {
        let mut sim = Simulation::with_size(16, 16);

        // Layout:
        //   Mux8to1 at (5, 5)
        //   - left input: packed ROM data (8 bytes in one u64)
        //   - right input: selector (PC value)

        // Pack 8 ROM bytes: addr 0 = 0x12, addr 1 = 0x17, addr 2 = 0x31, etc.
        let rom_packed: u64 = 0x0000_0000_0031_1712; // [0x12, 0x17, 0x31, 0, 0, 0, 0, 0]

        sim.set_tile(4, 5, TileType::Const);
        sim.set_logic_value(4, 5, rom_packed);
        sim.set_tile(5, 5, TileType::Mux8to1);
        sim.set_tile(6, 5, TileType::Const); // Selector

        for addr in 0..4 {
            sim.set_logic_value(6, 5, addr);
            sim.mark_all_dirty();
            let stats = sim.tick_with_delays();
            let result = sim.get_logic_at(5, 5);
            let expected = (rom_packed >> (addr * 8)) & 0xFF;
            println!(
                "  Select {}: Mux output = 0x{:02X} (expected 0x{:02X}) {} ({} deltas)",
                addr,
                result,
                expected,
                if result == expected { "PASS" } else { "FAIL" },
                stats.total_deltas
            );
        }
    }
    println!();

    // =========================================================================
    // Test 5: RegEnable with write control
    // =========================================================================
    println!("Test 5: RegEnable with enable signal");
    println!("─────────────────────────────────────");
    {
        let mut sim = Simulation::with_size(16, 16);

        // Layout:
        //   Clock at (5, 4) - above (up input)
        //   Data at (4, 5) - left input
        //   RegEnable at (5, 5)
        //   Enable at (6, 5) - right input

        sim.set_tile(5, 4, TileType::ClockGlobal);
        sim.set_tile(4, 5, TileType::Const);
        sim.set_tile(5, 5, TileType::RegEnable);
        sim.set_tile(6, 5, TileType::Const);

        // Initial: data=42, enable=0 (disabled)
        sim.set_logic_value(4, 5, 42);
        sim.set_logic_value(6, 5, 0); // Disabled

        sim.initialize();
        println!("  Initial: data=42, enable=0, reg={}", sim.get_logic_at(5, 5));

        // Tick with enable=0 - should NOT capture
        sim.tick_with_delays();
        sim.tick_with_delays();
        println!("  After 2 ticks (enable=0): reg={}", sim.get_logic_at(5, 5));

        // Enable and tick - should capture
        sim.set_logic_value(6, 5, 1); // Enable
        sim.mark_all_dirty();
        sim.tick_with_delays();
        sim.tick_with_delays();
        println!("  After 2 ticks (enable=1): reg={}", sim.get_logic_at(5, 5));

        if sim.get_logic_at(5, 5) == 42 {
            println!("  PASS: RegEnable captured value!");
        } else {
            println!("  FAIL: RegEnable didn't capture!");
        }
    }
    println!();

    // =========================================================================
    // Test 6: Simple CPU - PC -> ROM -> IR
    // =========================================================================
    println!("Test 6: PC -> ROM -> IR chain");
    println!("──────────────────────────────");
    {
        let mut sim = Simulation::with_size(32, 32);

        // Layout (all directly adjacent):
        //   Row 4: [Clock]
        //   Row 5: [PC] [Jump=0]
        //   Row 6: [Wire] (PC output goes down)
        //   Row 7: [ROM packed] [Mux8to1] [PC wire]
        //   Row 8: [Wire] (Mux output goes down)
        //   Row 9: [Clock2] (for IR)
        //   Row 10: [ROM wire] [IR]

        let ox = 10;
        let oy = 4;

        // Clock for PC
        sim.set_tile(ox, oy, TileType::ClockGlobal);

        // PC
        sim.set_tile(ox, oy + 1, TileType::ProgramCounter);
        sim.set_logic_value(ox, oy + 1, 0);
        sim.set_tile(ox + 1, oy + 1, TileType::Const); // Jump enable = 0
        sim.set_logic_value(ox + 1, oy + 1, 0);

        // PC output wire (down)
        sim.set_tile(ox, oy + 2, TileType::Wire);
        sim.set_tile(ox, oy + 3, TileType::Wire);

        // ROM packed data (at left of Mux)
        let rom_packed: u64 = 0x0000_0000_0031_1712; // [0x12, 0x17, 0x31, 0, ...]
        sim.set_tile(ox - 1, oy + 3, TileType::Const);
        sim.set_logic_value(ox - 1, oy + 3, rom_packed);

        // Mux8to1 (select = PC, data = ROM)
        sim.set_tile(ox, oy + 4, TileType::Mux8to1);

        // Wire from PC to Mux select (right side)
        sim.set_tile(ox + 1, oy + 3, TileType::Wire);
        sim.set_tile(ox + 1, oy + 4, TileType::Wire);

        // Mux output wire (down)
        sim.set_tile(ox, oy + 5, TileType::Wire);

        // Clock for IR (separate clock line)
        sim.set_tile(ox - 1, oy + 5, TileType::Wire);
        sim.set_tile(ox - 1, oy + 4, TileType::Wire);
        sim.set_tile(ox - 1, oy + 3, TileType::Wire);
        sim.set_tile(ox - 1, oy + 2, TileType::Wire);
        sim.set_tile(ox - 1, oy + 1, TileType::Wire);
        sim.set_tile(ox - 1, oy, TileType::Wire);

        // IR
        sim.set_tile(ox, oy + 6, TileType::Register8);

        let stats = sim.initialize();
        println!("  Init: {} deltas", stats.total_deltas);

        let pc_pos = (ox, oy + 1);
        let mux_pos = (ox, oy + 4);
        let ir_pos = (ox, oy + 6);

        println!("  PC at ({}, {})", pc_pos.0, pc_pos.1);
        println!("  Mux at ({}, {})", mux_pos.0, mux_pos.1);
        println!("  IR at ({}, {})", ir_pos.0, ir_pos.1);

        println!("\n  Running cycles:");
        for cycle in 0..8 {
            let stats = sim.tick_with_delays();
            let pc = sim.get_logic_at(pc_pos.0, pc_pos.1);
            let mux = sim.get_logic_at(mux_pos.0, mux_pos.1);
            let ir = sim.get_logic_at(ir_pos.0, ir_pos.1);
            println!(
                "  Cycle {}: clock={} PC={} Mux=0x{:02X} IR=0x{:02X} ({} deltas)",
                cycle,
                if sim.global_clock { "H" } else { "L" },
                pc,
                mux & 0xFF,
                ir & 0xFF,
                stats.total_deltas
            );
        }
    }
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("Summary");
    println!("═══════");
    println!("These tests verify individual components work correctly.");
    println!("The full CPU needs proper layout with directly adjacent tiles");
    println!("and omnidirectional Wire tiles for any routing.");
}
