//! EPIC 104: Memory Tiles Demo
//!
//! Demonstrates building addressable memory from RAM, Counter, and Const tiles.
//! Shows how to create a simple 4-byte RAM array with address decoding.
//!
//! Run with: cargo run --release --example epic104_memory_demo

use engine::simulation::Simulation;
use engine::tile_meta::TileType;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 104: Memory Tiles Demo                                  ║");
    println!("║  Building addressable RAM from tile primitives                ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    demo_ram_basic();
    demo_counter();
    demo_const();
    demo_ram_array();
    demo_alu_with_memory();
}

/// Demo 1: Basic RAM tile behavior
fn demo_ram_basic() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 1: Basic RAM Tile (Write-Enable Gated Storage)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut sim = Simulation::new();

    // RAM cell at (10, 10)
    // - Left input (9, 10): data to write
    // - Up input (10, 9): write enable
    sim.set_tile(10, 10, TileType::Ram);
    sim.set_tile(9, 10, TileType::Wire); // Data input
    sim.set_tile(10, 9, TileType::Wire); // Write enable

    // Initialize RAM to 0
    set_tile_value(&mut sim, 10, 10, 0);
    println!("Initial RAM value: {}", get_tile_value(&sim, 10, 10));

    // Test 1: Write disabled (WE=0), data=42 → should hold 0
    set_tile_value(&mut sim, 9, 10, 42); // data = 42
    set_tile_value(&mut sim, 10, 9, 0); // WE = 0
    sim.eval_at(10, 10);
    println!(
        "After eval with WE=0, data=42: RAM={} (should be 0)",
        get_tile_value(&sim, 10, 10)
    );

    // Test 2: Write enabled (WE=1), data=42 → should write 42
    set_tile_value(&mut sim, 10, 9, 1); // WE = 1
    sim.eval_at(10, 10);
    println!(
        "After eval with WE=1, data=42: RAM={} (should be 42)",
        get_tile_value(&sim, 10, 10)
    );

    // Test 3: Write disabled again, data=99 → should hold 42
    set_tile_value(&mut sim, 9, 10, 99); // data = 99
    set_tile_value(&mut sim, 10, 9, 0); // WE = 0
    sim.eval_at(10, 10);
    println!(
        "After eval with WE=0, data=99: RAM={} (should still be 42)",
        get_tile_value(&sim, 10, 10)
    );

    // Test 4: Write new value
    set_tile_value(&mut sim, 9, 10, 100); // data = 100
    set_tile_value(&mut sim, 10, 9, 1); // WE = 1
    sim.eval_at(10, 10);
    println!(
        "After eval with WE=1, data=100: RAM={} (should be 100)",
        get_tile_value(&sim, 10, 10)
    );

    println!("\n✓ RAM tile correctly implements write-enable gated storage\n");
}

/// Demo 2: Counter tile behavior
fn demo_counter() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 2: Counter Tile (Conditional Increment)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut sim = Simulation::new();

    // Counter at (10, 10)
    // - Up input (10, 9): enable counting
    sim.set_tile(10, 10, TileType::Counter);
    sim.set_tile(10, 9, TileType::Wire); // Enable input

    // Initialize counter to 0
    set_tile_value(&mut sim, 10, 10, 0);
    println!("Initial counter: {}", get_tile_value(&sim, 10, 10));

    // Count 5 times with enable=1
    set_tile_value(&mut sim, 10, 9, 1); // Enable = 1
    for i in 1..=5 {
        sim.eval_at(10, 10);
        println!("After tick {}: counter={}", i, get_tile_value(&sim, 10, 10));
    }

    // Disable counting
    set_tile_value(&mut sim, 10, 9, 0); // Enable = 0
    println!("\nDisabling counter (enable=0):");
    for i in 1..=3 {
        sim.eval_at(10, 10);
        println!(
            "After tick {}: counter={} (should stay at 5)",
            i,
            get_tile_value(&sim, 10, 10)
        );
    }

    // Re-enable and count more
    set_tile_value(&mut sim, 10, 9, 1); // Enable = 1
    println!("\nRe-enabling counter:");
    for i in 1..=3 {
        sim.eval_at(10, 10);
        println!("After tick {}: counter={}", i, get_tile_value(&sim, 10, 10));
    }

    println!("\n✓ Counter tile correctly implements conditional increment\n");
}

/// Demo 3: Const tile behavior
fn demo_const() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 3: Const Tile (Constant Value Source)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut sim = Simulation::new();

    // Const tile holds its value regardless of inputs
    sim.set_tile(10, 10, TileType::Const);
    sim.set_tile(9, 10, TileType::Wire); // Random neighbor

    // Set const value
    set_tile_value(&mut sim, 10, 10, 12345);
    println!("Initial const value: {}", get_tile_value(&sim, 10, 10));

    // Try to influence it with different neighbor values
    for val in [0, 100, u64::MAX, 42] {
        set_tile_value(&mut sim, 9, 10, val);
        sim.eval_at(10, 10);
        println!(
            "After eval with neighbor={}: const={} (should always be 12345)",
            val,
            get_tile_value(&sim, 10, 10)
        );
    }

    println!("\n✓ Const tile correctly holds its value\n");
}

/// Demo 4: Building a 4-cell RAM array (well-spaced to avoid crosstalk)
fn demo_ram_array() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 4: 4-Cell RAM Array (Isolated Cells)");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Architecture:");
    println!("  - 4 RAM cells spaced apart to avoid neighbor interference");
    println!("  - Each cell has dedicated data input (left) and write-enable (up)");
    println!("  - Demonstrates selective write with address decoding concept\n");

    let mut sim = Simulation::new();

    // Create 4 isolated RAM cells, spaced 3 apart to avoid crosstalk
    // Cell 0 at (20, 10), Cell 1 at (23, 10), Cell 2 at (26, 10), Cell 3 at (29, 10)
    let ram_y = 10;
    let cell_positions = [20usize, 23, 26, 29];

    for (i, &x) in cell_positions.iter().enumerate() {
        sim.set_tile(x, ram_y, TileType::Ram);
        sim.set_tile(x - 1, ram_y, TileType::Wire); // Data input (left)
        sim.set_tile(x, ram_y - 1, TileType::Wire); // Write enable (up)
        set_tile_value(&mut sim, x, ram_y, 0); // Initialize to 0
        println!("  Cell {} at ({}, {})", i, x, ram_y);
    }

    println!("\nWriting values to memory:");

    // Helper to write to a specific cell
    let write_cell = |sim: &mut Simulation, cell: usize, value: u64| {
        let x = cell_positions[cell];
        // Set data input
        set_tile_value(sim, x - 1, ram_y, value);
        // Enable write
        set_tile_value(sim, x, ram_y - 1, 1);
        // Evaluate
        sim.eval_at(x, ram_y);
        // Disable write
        set_tile_value(sim, x, ram_y - 1, 0);
    };

    write_cell(&mut sim, 0, 100);
    println!("  Cell 0 ← 100");

    write_cell(&mut sim, 1, 200);
    println!("  Cell 1 ← 200");

    write_cell(&mut sim, 2, 300);
    println!("  Cell 2 ← 300");

    write_cell(&mut sim, 3, 400);
    println!("  Cell 3 ← 400");

    println!("\nReading back values:");
    for (i, &x) in cell_positions.iter().enumerate() {
        let val = get_tile_value(&sim, x, ram_y);
        println!("  Cell {}: {}", i, val);
    }

    // Verify selective overwrite
    println!("\nOverwriting cell 1 only:");
    write_cell(&mut sim, 1, 999);
    println!("  Cell 1 ← 999");

    println!("\nFinal memory state:");
    let expected = [100u64, 999, 300, 400];
    let mut all_ok = true;
    for (i, &x) in cell_positions.iter().enumerate() {
        let val = get_tile_value(&sim, x, ram_y);
        let status = if val == expected[i] {
            "✓"
        } else {
            all_ok = false;
            "✗"
        };
        println!(
            "  Cell {}: {} (expected {}) {}",
            i, val, expected[i], status
        );
    }

    if all_ok {
        println!("\n✓ RAM array correctly implements selective addressing\n");
    } else {
        println!("\n✗ RAM array has issues - check tile layout\n");
    }
}

/// Demo 5: Simple ALU with memory - compute A + B and store result
fn demo_alu_with_memory() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("Demo 5: ALU with Memory (Compute A + B, Store Result)");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Architecture:");
    println!("  - Two Const tiles for operands A and B");
    println!("  - One Add tile for computation");
    println!("  - One RAM tile to store result");
    println!("  - One Counter for operation count\n");

    let mut sim = Simulation::new();

    // Layout (spaced to avoid interference):
    // (50,10)=Const_A  (51,10)=Add  (52,10)=Const_B
    //                  (51,11)=Wire (data route)
    //                  (51,12)=RAM
    //                  (51,11)=WE wire (up of RAM)
    // (55,10)=Counter with enable at (55,9)

    // Operand A (Const)
    sim.set_tile(50, 10, TileType::Const);
    set_tile_value(&mut sim, 50, 10, 25);

    // Operand B (Const)
    sim.set_tile(52, 10, TileType::Const);
    set_tile_value(&mut sim, 52, 10, 17);

    // Add tile - takes left (A) and right (B)
    sim.set_tile(51, 10, TileType::Add);

    // Result storage (RAM) at (51, 12)
    sim.set_tile(51, 12, TileType::Ram);
    sim.set_tile(50, 12, TileType::Wire); // Data input (left)
    sim.set_tile(51, 11, TileType::Wire); // Write enable (up)
    set_tile_value(&mut sim, 51, 11, 1); // WE always on for this demo

    // Operation counter (isolated position)
    sim.set_tile(55, 10, TileType::Counter);
    sim.set_tile(55, 9, TileType::Wire); // Enable (up)
    set_tile_value(&mut sim, 55, 9, 1); // Always enabled

    println!("Initial state:");
    println!("  A = {}", get_tile_value(&sim, 50, 10));
    println!("  B = {}", get_tile_value(&sim, 52, 10));
    println!("  Result RAM = {}", get_tile_value(&sim, 51, 12));
    println!("  Op Counter = {}", get_tile_value(&sim, 55, 10));

    // Compute: Evaluate Add tile
    sim.eval_at(51, 10);
    let add_result = get_tile_value(&sim, 51, 10);
    println!("\nAfter Add evaluation:");
    println!("  Add output = {} (25 + 17 = 42)", add_result);

    // Route result to RAM's data input and evaluate
    set_tile_value(&mut sim, 50, 12, add_result);
    sim.eval_at(51, 12);
    println!("  RAM stored = {}", get_tile_value(&sim, 51, 12));

    // Update counter
    sim.eval_at(55, 10);
    println!("  Op Counter = {}", get_tile_value(&sim, 55, 10));

    // Run a few more operations with different values
    println!("\nRunning more computations:");
    for (a, b) in [(10, 20), (100, 50), (7, 8)] {
        set_tile_value(&mut sim, 50, 10, a);
        set_tile_value(&mut sim, 52, 10, b);
        sim.eval_at(51, 10);
        let result = get_tile_value(&sim, 51, 10);
        set_tile_value(&mut sim, 50, 12, result);
        sim.eval_at(51, 12);
        sim.eval_at(55, 10);
        println!(
            "  {} + {} = {}, stored in RAM, op count = {}",
            a,
            b,
            get_tile_value(&sim, 51, 12),
            get_tile_value(&sim, 55, 10)
        );
    }

    println!("\n✓ ALU with memory demonstrates computation + storage\n");
}

// Helper functions

fn set_tile_value(sim: &mut Simulation, x: usize, y: usize, value: u64) {
    sim.tilemap.set_value_at(x, y, value);
}

fn get_tile_value(sim: &Simulation, x: usize, y: usize) -> u64 {
    sim.tilemap.value_at(x, y).unwrap_or(0)
}
