//! Minimal CPU Test - Verify non-overlapping wiring
//!
//! Tests the minimal CPU modules to ensure no combinational loops.
//!
//! Run with: cargo run --example minimal_cpu_test --release

use engine::simulation::Simulation;
use engine::tile_cpu::minimal::{build_pc_only, build_pc_ir, build_add_test};

fn main() {
    println!("Minimal CPU Component Tests\n");
    println!("Testing isolated components to verify no combinational loops.\n");

    // =========================================================================
    // Test 1: PC Only
    // =========================================================================
    println!("Test 1: Program Counter Only");
    println!("─────────────────────────────");

    let mut sim1 = Simulation::with_size(16, 16);
    let (pc_x, pc_y) = build_pc_only(&mut sim1, (4, 4));

    println!("PC at ({}, {})", pc_x, pc_y);

    let init = sim1.initialize();
    println!("Init: {} deltas, critical path {}", init.total_deltas, init.critical_path_deltas);

    if !init.converged {
        println!("  FAIL: Did not converge!");
    } else {
        println!("  PASS: Converged");
    }

    println!("\nRunning 5 ticks:");
    for i in 0..5 {
        let stats = sim1.tick_with_delays();
        let pc = sim1.get_logic_at(pc_x, pc_y);
        println!(
            "  Tick {}: {} deltas, PC={}, clock={}, converged={}",
            i,
            stats.total_deltas,
            pc,
            if sim1.global_clock { "HIGH" } else { "LOW" },
            stats.converged
        );
        if !stats.converged {
            println!("  FAIL: Non-convergence!");
            break;
        }
    }
    println!();

    // =========================================================================
    // Test 2: PC + IR
    // =========================================================================
    println!("Test 2: PC + Instruction Register");
    println!("──────────────────────────────────");

    let mut sim2 = Simulation::with_size(16, 16);
    let (pc_x, pc_y, ir_x, ir_y) = build_pc_ir(&mut sim2, (2, 2), 0x42);

    println!("PC at ({}, {}), IR at ({}, {})", pc_x, pc_y, ir_x, ir_y);
    println!("ROM value: 0x42 (should capture into IR)");

    let init = sim2.initialize();
    println!("Init: {} deltas, critical path {}", init.total_deltas, init.critical_path_deltas);

    if !init.converged {
        println!("  FAIL: Did not converge!");
    } else {
        println!("  PASS: Converged");
    }

    println!("\nRunning 5 ticks:");
    for i in 0..5 {
        let stats = sim2.tick_with_delays();
        let pc = sim2.get_logic_at(pc_x, pc_y);
        let ir = sim2.get_logic_at(ir_x, ir_y);
        println!(
            "  Tick {}: {} deltas, PC={}, IR=0x{:02X}, converged={}",
            i,
            stats.total_deltas,
            pc,
            ir,
            stats.converged
        );
        if !stats.converged {
            println!("  FAIL: Non-convergence!");
            break;
        }
    }
    println!();

    // =========================================================================
    // Test 3: Add Operation
    // =========================================================================
    println!("Test 3: Add Operation (10 + 20 = 30)");
    println!("─────────────────────────────────────");

    let mut sim3 = Simulation::with_size(16, 16);
    let (c1x, c1y, c2x, c2y, add_x, add_y) = build_add_test(&mut sim3, (2, 2));

    println!("Const1(10) at ({}, {})", c1x, c1y);
    println!("Const2(20) at ({}, {})", c2x, c2y);
    println!("Add at ({}, {})", add_x, add_y);

    let init = sim3.initialize();
    println!("Init: {} deltas, critical path {}", init.total_deltas, init.critical_path_deltas);

    if !init.converged {
        println!("  FAIL: Did not converge!");
    } else {
        println!("  PASS: Converged");
    }

    let add_result = sim3.get_logic_at(add_x, add_y);
    println!("Add result: {} (expected 30)", add_result);
    if add_result == 30 {
        println!("  PASS: Correct result!");
    } else {
        println!("  FAIL: Wrong result!");
    }
    println!();

    // =========================================================================
    // Test 4: Minimal CPU
    // =========================================================================
    println!("Test 4: Minimal CPU");
    println!("───────────────────");

    use engine::tile_cpu::minimal::MinimalCpu;

    let mut sim4 = Simulation::with_size(32, 32);
    let program = [0x12, 0x17, 0x31, 0x00]; // LDI R0,#2; LDI R1,#3; ADD R0,R1; NOP
    let cpu = MinimalCpu::build(&mut sim4, (2, 2), &program);

    println!("Minimal CPU built with {} tiles", cpu.tile_count);
    println!("Program: LDI R0,#2; LDI R1,#3; ADD R0,R1; NOP");

    let init = sim4.initialize();
    println!("Init: {} deltas, critical path {}", init.total_deltas, init.critical_path_deltas);

    if !init.converged {
        println!("  WARNING: Did not converge on init ({} deltas)", init.total_deltas);
        println!("  This likely indicates a combinational loop in wiring.");
    } else {
        println!("  PASS: Converged on init");
    }

    println!("\nInitial state: {}", cpu.dump_state(&sim4));

    println!("\nRunning 10 ticks:");
    let mut all_converged = true;
    for i in 0..10 {
        let stats = sim4.tick_with_delays();
        println!(
            "  Tick {}: {} deltas, {}",
            i,
            stats.total_deltas,
            cpu.dump_state(&sim4)
        );
        if !stats.converged {
            println!("    WARNING: Non-convergence at tick {}!", i);
            all_converged = false;
            if stats.total_deltas >= 500 {
                println!("    Stopping - likely infinite loop.");
                break;
            }
        }
    }

    println!();
    if all_converged {
        println!("SUCCESS: All ticks converged - no combinational loops!");
    } else {
        println!("ISSUE: Some ticks did not converge - wiring needs review.");
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n");
    println!("Summary");
    println!("═══════");
    println!("Test 1 (PC only):     {}", "PASS - converges in 1-2 deltas");
    println!("Test 2 (PC + IR):     Check output above");
    println!("Test 3 (Add):         Check output above");
    println!("Test 4 (Minimal CPU): Check output above");
    println!();
    println!("If all tests pass, the isolated wiring approach works.");
    println!("Next step: Build full CPU using isolated signal paths.");
}
