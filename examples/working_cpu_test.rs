//! Working CPU Test - Properly connected PC -> ROM -> IR
//!
//! Run with: cargo run --example working_cpu_test --release

use engine::simulation::Simulation;
use engine::tile_meta::TileType;

fn main() {
    println!("Working CPU Test: PC -> ROM -> IR\n");

    // Key insights:
    // 1. Default "empty" tiles are Wire type and propagate bidirectionally!
    // 2. WireV reads from both up AND down, causing feedback and bit accumulation
    // 3. WireDown only reads from up, enabling true one-way signal flow
    // 4. Use Const(0) tiles to block potential leak paths

    let mut sim = Simulation::with_size(32, 32);

    // Layout using WireDown for unidirectional selector path:
    //
    //     Col:  7     8     9     10    11
    //   Row 4:             [CLK]             -- Clock for PC
    //   Row 5:             [PC]  [J=0]       -- PC at (9,5)
    //   Row 6:             [WireDown]        -- Reads from PC above only
    //   Row 7:             [WireDown]        -- Continues down, no feedback
    //   Row 8: [ROM] [MUX] [WireDown]        -- ROM(7,8), Mux(8,8), sel(9,8)
    //   Row 9: [C=0] [WireDown] [C=0]        -- Mux output (unidirectional)
    //   Row 10:[C=0] [WireDown] [C=0]        -- Continue down
    //   Row 11:[C=0] [WireDown] [C=0] [CLK2] -- Data path, gap, clock for IR
    //   Row 12:[C=0] [WireDown] [WireRight] [IR] -- WireRight to IR (ignores IR output)
    //
    // WireDown ensures PC signal flows one-way to selector without feedback

    let rom_packed: u64 = 0x0000_0000_0031_1712; // [0x12, 0x17, 0x31, 0x00, ...]

    // === Clock for PC ===
    sim.set_tile(9, 4, TileType::ClockGlobal);

    // === PC with jump disable ===
    sim.set_tile(9, 5, TileType::ProgramCounter);
    sim.set_logic_value(9, 5, 0);
    sim.set_tile(10, 5, TileType::Const); // Jump disable
    sim.set_logic_value(10, 5, 0);

    // === Selector path: PC -> Mux right input using WireDown ===
    // WireDown only reads from UP, preventing feedback from below
    sim.set_tile(9, 6, TileType::WireDown);
    sim.set_tile(9, 7, TileType::WireDown);
    sim.set_tile(9, 8, TileType::WireDown);

    // === Block below selector ===
    sim.set_tile(9, 9, TileType::Const);
    sim.set_logic_value(9, 9, 0);

    // === ROM and Mux ===
    sim.set_tile(7, 8, TileType::Const);
    sim.set_logic_value(7, 8, rom_packed);
    sim.set_tile(8, 8, TileType::Mux8to1); // left=ROM(7,8), right=WireDown(9,8)

    // === Mux output path to IR using WireDown (unidirectional) ===
    sim.set_tile(8, 9, TileType::WireDown); // Mux output down (reads from up only)
    sim.set_tile(8, 10, TileType::WireDown); // Continue down
    sim.set_tile(8, 11, TileType::WireDown); // Continue down
    sim.set_tile(8, 12, TileType::WireDown); // Continue to WireRight

    // === Block data path sides ===
    sim.set_tile(7, 9, TileType::Const);
    sim.set_logic_value(7, 9, 0);
    sim.set_tile(7, 10, TileType::Const);
    sim.set_logic_value(7, 10, 0);
    sim.set_tile(7, 11, TileType::Const);
    sim.set_logic_value(7, 11, 0);
    sim.set_tile(7, 12, TileType::Const);
    sim.set_logic_value(7, 12, 0);

    // === IR clock (not adjacent to data wires) ===
    sim.set_tile(10, 11, TileType::ClockGlobal);

    // === IR at (10, 12) ===
    // Clock from above (10,11), data from left via WireH(9,12)
    sim.set_tile(10, 12, TileType::Register8);

    // === Route data to IR using WireRight (ignores IR output from right) ===
    sim.set_tile(9, 12, TileType::WireRight); // Only reads from left

    // === Block gap between data and clock ===
    sim.set_tile(9, 10, TileType::Const);
    sim.set_logic_value(9, 10, 0);
    sim.set_tile(9, 11, TileType::Const);
    sim.set_logic_value(9, 11, 0);

    // === Block below IR ===
    sim.set_tile(10, 13, TileType::Const);
    sim.set_logic_value(10, 13, 0);
    sim.set_tile(9, 13, TileType::Const);
    sim.set_logic_value(9, 13, 0);
    sim.set_tile(8, 13, TileType::Const);
    sim.set_logic_value(8, 13, 0);

    println!("Layout:");
    println!("  Clock1 at (9, 4) for PC");
    println!("  PC at (9, 5), jump=0 at (10, 5)");
    println!("  Selector: WireDown at (9, 6-8) - unidirectional!");
    println!("  ROM at (7, 8), Mux at (8, 8)");
    println!("  Data path: WireDown at (8, 9-12) -> WireRight(9,12) -> IR");
    println!("  Clock2 at (10, 11) for IR");
    println!("  IR at (10, 12)");
    println!();

    let stats = sim.initialize();
    println!("Init: {} deltas\n", stats.total_deltas);

    // Check values
    let pc_pos = (9, 5);
    let sel_pos = (9, 8);
    let mux_pos = (8, 8);
    let ir_pos = (10, 12);

    let pc = sim.get_logic_at(pc_pos.0, pc_pos.1);
    let selector = sim.get_logic_at(sel_pos.0, sel_pos.1);
    let mux = sim.get_logic_at(mux_pos.0, mux_pos.1);
    let ir = sim.get_logic_at(ir_pos.0, ir_pos.1);

    println!("After init:");
    println!("  PC = {}", pc);
    println!("  Selector = {} (should be {})", selector, pc);
    println!(
        "  Mux = 0x{:02X} (ROM[{}] = 0x{:02X})",
        mux & 0xFF,
        selector & 7,
        (rom_packed >> ((selector & 7) as u32 * 8)) & 0xFF
    );
    println!("  IR = 0x{:02X}", ir & 0xFF);
    println!();

    println!("Running cycles:");
    println!("ROM: [0]=0x12, [1]=0x17, [2]=0x31, [3-7]=0x00\n");

    for cycle in 0..12 {
        let stats = sim.tick_with_delays();
        let clock = if sim.global_clock { "H" } else { "L" };

        let pc = sim.get_logic_at(pc_pos.0, pc_pos.1);
        let selector = sim.get_logic_at(sel_pos.0, sel_pos.1);
        let mux = sim.get_logic_at(mux_pos.0, mux_pos.1);
        let ir = sim.get_logic_at(ir_pos.0, ir_pos.1);

        let expected = (rom_packed >> ((selector & 7) as u32 * 8)) & 0xFF;

        let sel_str = if selector == u64::MAX {
            "MAX".to_string()
        } else {
            selector.to_string()
        };
        let match_str = if (selector & 7) == (pc & 7) {
            ""
        } else {
            " MISMATCH!"
        };

        println!(
            "  Cycle {:2}: clk={} PC={} sel={}{} Mux=0x{:02X}(exp 0x{:02X}) IR=0x{:02X} ({} deltas){}",
            cycle,
            clock,
            pc,
            sel_str,
            match_str,
            mux & 0xFF,
            expected,
            ir & 0xFF,
            stats.total_deltas,
            if !stats.converged {
                " [NOT CONVERGED]"
            } else {
                ""
            }
        );
    }

    println!("\nExpected behavior:");
    println!("  - PC: 0->1->2->3->... (increment on HIGH)");
    println!("  - Selector = PC value (via WireDown, no feedback)");
    println!("  - Mux = ROM[selector]");
    println!("  - IR captures Mux on rising edge (one cycle delay)");
}
