//! CPU Chain Test - Proper PC -> ROM -> IR wiring
//!
//! Run with: cargo run --example cpu_chain_test --release

use engine::simulation::Simulation;
use engine::tile_meta::TileType;

fn main() {
    println!("CPU Chain Test: PC -> ROM -> IR\n");

    // =========================================================================
    // Test 1: Minimal Working Chain
    // =========================================================================
    println!("Test 1: Minimal PC -> ROM -> IR");
    println!("────────────────────────────────");

    let mut sim = Simulation::with_size(32, 32);

    // Layout - all tiles directly adjacent:
    //
    //   Y=4: [Clock] at (10, 4)
    //   Y=5: [PC] at (10, 5), [Jump=0] at (11, 5)
    //   Y=6: (PC output goes down via Wire)
    //   Y=7: [ROM data] at (9, 7), [Mux8to1] at (10, 7), [PC wire] at (11, 7)
    //   Y=8: (Mux output goes down via Wire)
    //   Y=9: [ROM wire] at (9, 9), [IR] at (10, 9)
    //
    // Clock routing: Clock -> PC (direct)
    //                Clock -> IR (via wire chain down left side)

    // Pack ROM: 4 instructions
    // [0x12, 0x17, 0x31, 0x00] = LDI R0,#2; LDI R1,#3; ADD R0,R1; NOP
    let rom_packed: u64 = 0x0000_0000_0031_1712;

    // Row 4: Clock
    sim.set_tile(10, 4, TileType::ClockGlobal);

    // Row 5: PC + jump disable
    sim.set_tile(10, 5, TileType::ProgramCounter);
    sim.set_logic_value(10, 5, 0); // Start at address 0
    sim.set_tile(11, 5, TileType::Const);
    sim.set_logic_value(11, 5, 0); // Jump enable = 0

    // Row 6: Wire from PC output (down)
    sim.set_tile(10, 6, TileType::Wire);

    // Row 7: ROM data, Mux, PC selector
    sim.set_tile(9, 7, TileType::Const);
    sim.set_logic_value(9, 7, rom_packed); // ROM packed data
    sim.set_tile(10, 7, TileType::Mux8to1); // Selects ROM[PC]
    sim.set_tile(11, 7, TileType::Wire);    // Routes PC to Mux right input

    // Row 8: Wire from Mux output (down)
    sim.set_tile(10, 8, TileType::Wire);

    // Row 9: Clock wire + IR
    // Clock needs to reach IR - route down left side
    sim.set_tile(9, 5, TileType::Wire);
    sim.set_tile(9, 6, TileType::Wire);
    sim.set_tile(9, 7, TileType::Wire); // Crosses ROM data - problem!

    // Actually, the Clock wire can't cross the ROM without a Cross tile
    // Let me use a different layout where clock is direct

    println!("  Initial layout has wiring conflicts. Trying cleaner layout...\n");

    // =========================================================================
    // Test 2: Clean Layout with proper separation
    // =========================================================================
    println!("Test 2: Clean layout");
    println!("────────────────────");

    let mut sim2 = Simulation::with_size(32, 32);

    // Layout:
    //   Col 5: Clock routing (vertical)
    //   Col 7: Data path (PC -> Mux -> IR)
    //
    //   Row 4: [Clock] at (5, 4)
    //   Row 5: [Clock wire] at (5, 5)   [PC] at (7, 5)   [Jump=0] at (8, 5)
    //   Row 6: [Clock wire] at (5, 6)   [Wire] at (7, 6)
    //   Row 7: [Clock wire] at (5, 7)   [Mux8to1] at (7, 7)   [Wire sel] at (8, 7)
    //          [ROM] at (6, 7)
    //   Row 8: [Clock wire] at (5, 8)   [Wire] at (7, 8)
    //   Row 9: [Clock -> IR] at (6, 9)  [IR] at (7, 9)

    // Clock column (5)
    sim2.set_tile(5, 4, TileType::ClockGlobal);
    sim2.set_tile(5, 5, TileType::Wire);
    sim2.set_tile(5, 6, TileType::Wire);
    sim2.set_tile(5, 7, TileType::Wire);
    sim2.set_tile(5, 8, TileType::Wire);
    sim2.set_tile(5, 9, TileType::Wire);
    sim2.set_tile(6, 9, TileType::Wire); // Clock to IR

    // PC at (7, 5)
    sim2.set_tile(7, 5, TileType::ProgramCounter);
    sim2.set_logic_value(7, 5, 0);
    sim2.set_tile(8, 5, TileType::Const); // Jump = 0
    sim2.set_logic_value(8, 5, 0);
    // Clock to PC: need wire from (5,5) to (6,5) to PC
    // But PC needs clock from ABOVE (up input)
    // Let me re-think...

    println!("  PC needs clock from UP input. Reorganizing...\n");

    // =========================================================================
    // Test 3: Correct input directions
    // =========================================================================
    println!("Test 3: Correct input directions");
    println!("─────────────────────────────────");

    // PC inputs:
    //   - up: clock
    //   - left: jump target
    //   - right: jump enable
    //
    // Mux8to1 inputs:
    //   - left: packed data (ROM)
    //   - right: selector (PC value)
    //
    // IR (Register8) inputs:
    //   - up: clock
    //   - left: data

    let mut sim3 = Simulation::with_size(32, 32);

    // Layout with correct input directions:
    //
    //   Row 4: [Clock] at (7, 4)       -- above PC
    //   Row 5: [PC] at (7, 5)  [Jump=0] at (8, 5)
    //   Row 6: [Wire] at (7, 6) -- PC output down
    //   Row 7: [ROM] at (6, 7)  [Mux] at (7, 7)  [Wire] at (8, 7)
    //   Row 8: [Wire] at (7, 8) -- Mux output down (also clock path)
    //   Row 9: [Clock2] at (7, 8) -- reuse Wire as clock
    //          Wait, IR needs separate clock input from above

    // Simpler: use separate clocks for PC and IR
    // In real hardware we'd route clock properly, but for test just use two ClockGlobal

    // Row 4: Clock for PC
    sim3.set_tile(7, 4, TileType::ClockGlobal);

    // Row 5: PC with jump disable
    sim3.set_tile(7, 5, TileType::ProgramCounter);
    sim3.set_logic_value(7, 5, 0);
    sim3.set_tile(8, 5, TileType::Const);
    sim3.set_logic_value(8, 5, 0);

    // Row 6: Wire from PC (down)
    sim3.set_tile(7, 6, TileType::Wire);

    // Row 7: ROM, Mux, PC-to-selector wire
    let rom_packed: u64 = 0x0000_0000_0031_1712;
    sim3.set_tile(6, 7, TileType::Const);
    sim3.set_logic_value(6, 7, rom_packed);
    sim3.set_tile(7, 7, TileType::Mux8to1);
    sim3.set_tile(8, 7, TileType::Wire); // Connects to (7,6) via adjacency

    // Wait, (8,7) is not adjacent to (7,6). Let me trace the path:
    // PC at (7,5) outputs at (7,5)
    // Wire at (7,6) receives from above (7,5) - OK
    // Mux at (7,7):
    //   - left = (6,7) = ROM data - OK
    //   - right = (8,7) = needs PC value

    // To get PC value to (8,7), we need: (7,6) -> (8,6) -> (8,7)
    sim3.set_tile(8, 6, TileType::Wire);
    sim3.set_tile(8, 7, TileType::Wire);

    // Row 8: Mux output (down)
    sim3.set_tile(7, 8, TileType::Wire);

    // Row 9: Clock for IR + IR
    // IR at (8, 9) with clock from above at (8, 8)
    // Actually, let's use (7, 9) for IR with clock routed there

    // Clock routing to IR: we need clock signal at (7, 8) which is Mux output wire
    // That won't work as clock. Let's use a separate clock column.

    // Column 5: Clock line for IR
    sim3.set_tile(5, 4, TileType::Wire); // Connect to main clock
    // Actually this gets complex. Let's just use two separate clocks for simplicity.

    // Row 8: Second clock for IR
    sim3.set_tile(6, 8, TileType::ClockGlobal);

    // Row 9: IR receives clock from (6,8) via left, data from (7,8) above...
    // Wait, IR clock input is UP, data is LEFT
    // IR at (7, 9):
    //   - up = (7, 8) = Mux output wire - this should be DATA not clock!
    //   - left = (6, 9) = would be clock

    // Let me reconsider IR position
    // IR at (6, 9):
    //   - up = (6, 8) = Clock - YES
    //   - left = (5, 9) = need Mux output here

    // Mux output route to IR:
    // Mux at (7, 7) outputs at (7, 7)
    // Wire at (7, 8) receives and propagates
    // Wire at (6, 8)... no that's the clock!

    // This is getting messy. Let me try a completely different layout.

    println!("  Wiring is complex. Creating simplified direct-connect version...\n");

    // =========================================================================
    // Test 4: Super simple - just verify each component works
    // =========================================================================
    println!("Test 4: Verified component chain");
    println!("─────────────────────────────────");

    let mut sim4 = Simulation::with_size(32, 32);

    // Step-by-step verified layout:
    //
    // 1. PC increments correctly (verified in earlier tests)
    // 2. Mux selects ROM bytes correctly (verified in earlier tests)
    // 3. Register8 captures data (verified in earlier tests)
    //
    // Now chain them:
    //
    //   (5,5) Clock    (6,5) Wire    (7,5) PC    (8,5) Const=0 (jump disable)
    //                                (7,6) Wire (PC output)
    //   (5,7) ROM      (6,7) Mux     (7,7) Wire (PC to Mux select)
    //                  (6,8) Wire (Mux output)
    //   (5,9) Clock2   (6,9) IR

    // PC setup
    sim4.set_tile(5, 5, TileType::ClockGlobal);
    sim4.set_tile(6, 5, TileType::Wire);
    sim4.set_tile(7, 5, TileType::ProgramCounter);
    sim4.set_logic_value(7, 5, 0);
    sim4.set_tile(8, 5, TileType::Const);
    sim4.set_logic_value(8, 5, 0);

    // PC output wire
    sim4.set_tile(7, 6, TileType::Wire);

    // Mux with ROM
    let rom_packed: u64 = 0x0000_0000_0031_1712;
    sim4.set_tile(5, 7, TileType::Const);
    sim4.set_logic_value(5, 7, rom_packed);
    sim4.set_tile(6, 7, TileType::Mux8to1);
    // Mux left = (5,7) = ROM
    // Mux right = (7,7) = need PC value

    // Route PC to Mux right input
    sim4.set_tile(7, 7, TileType::Wire);

    // But wait, (7,6) is not adjacent to (7,7)? Yes it is - vertically
    // Wire at (7,6) -> Wire at (7,7) -> propagates to Mux right

    // Mux output
    sim4.set_tile(6, 8, TileType::Wire);

    // IR with separate clock
    sim4.set_tile(5, 9, TileType::ClockGlobal);
    sim4.set_tile(6, 9, TileType::Register8);
    // IR up = (6, 8) = Mux output wire - NO, IR clock is from UP, data from LEFT
    // Register8: up=clock, left=data

    // Correct IR position: needs clock from above, data from left
    // Let's put IR at (7, 9):
    //   - up = (7, 8) = need clock here
    //   - left = (6, 9) = need Mux output here

    // Add wires to route:
    sim4.set_tile(6, 9, TileType::Wire); // Mux output to IR left
    sim4.set_tile(7, 8, TileType::Wire); // Extend Mux output down
    // Wait, (6,8) is Mux output, need to route to (6,9)
    // (6,8) Wire is already there, (6,9) Wire will receive from (6,8) - OK

    // Clock for IR at (7,8)? That's a wire from Mux, not clock
    // Need to rethink...

    // Actually, let me just use a clock at (7,8) directly for simplicity
    sim4.set_tile(7, 8, TileType::ClockGlobal);
    sim4.set_tile(7, 9, TileType::Register8);
    // IR at (7,9): up=(7,8)=Clock2, left=(6,9)=Wire from Mux

    let stats = sim4.initialize();
    println!("  Layout tiles placed");
    println!("  Init: {} deltas\n", stats.total_deltas);

    let pc_pos = (7, 5);
    let mux_pos = (6, 7);
    let ir_pos = (7, 9);

    println!("  PC at {:?}", pc_pos);
    println!("  Mux at {:?}", mux_pos);
    println!("  IR at {:?}\n", ir_pos);

    // Check initial values
    println!("  After init:");
    println!("    PC = {}", sim4.get_logic_at(pc_pos.0, pc_pos.1));
    println!("    Mux = 0x{:02X}", sim4.get_logic_at(mux_pos.0, mux_pos.1) & 0xFF);
    println!("    IR = 0x{:02X}\n", sim4.get_logic_at(ir_pos.0, ir_pos.1) & 0xFF);

    println!("  Running cycles:");
    for cycle in 0..8 {
        let stats = sim4.tick_with_delays();
        let pc = sim4.get_logic_at(pc_pos.0, pc_pos.1);
        let mux = sim4.get_logic_at(mux_pos.0, mux_pos.1);
        let ir = sim4.get_logic_at(ir_pos.0, ir_pos.1);
        println!(
            "    Cycle {}: PC={} Mux=0x{:02X} IR=0x{:02X} ({} deltas)",
            cycle,
            pc,
            mux & 0xFF,
            ir & 0xFF,
            stats.total_deltas
        );
    }

    println!("\n  Expected: PC increments, Mux selects ROM[PC], IR captures Mux output");
    println!("  ROM[0]=0x12, ROM[1]=0x17, ROM[2]=0x31, ROM[3]=0x00");
}
