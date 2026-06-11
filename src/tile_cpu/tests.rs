//! Tests for TileCpu
//!
//! These tests verify that the CPU executes correctly via tile simulation.

use super::*;
use crate::simulation::Simulation;
use crate::tile_meta::TileType;

// =============================================================================
// Phase 1: New Tile Type Tests
// =============================================================================

#[test]
fn test_addcarry_tile() {
    let mut sim = Simulation::with_size(8, 8);
    let w = sim.width();

    // Layout: [Const(left)] [AddCarry] [Const(right)]
    // Guard row above and below
    for col in 2..=4 {
        sim.set_tile(col, 2, TileType::Const);
        sim.set_logic_value(col, 2, 0);
        sim.set_tile(col, 4, TileType::Const);
        sim.set_logic_value(col, 4, 0);
    }
    sim.set_tile(2, 3, TileType::Const);
    sim.set_tile(3, 3, TileType::AddCarry);
    sim.set_tile(4, 3, TileType::Const);

    let addcarry_idx = 3 * w + 3;

    // Test: 100 + 50 = 150 (no carry, bit 8 = 0)
    sim.set_logic_value(2, 3, 100);
    sim.set_logic_value(4, 3, 50);
    sim.dirty.mark_dirty(addcarry_idx);
    sim.propagate_combinational();
    let result = sim.get_logic_value_by_idx(addcarry_idx);
    assert_eq!(result & 0xFF, 150, "100 + 50 = 150");
    assert_eq!((result >> 8) & 1, 0, "No carry for 100 + 50");

    // Test: 200 + 100 = 300 → low 8 bits = 44, carry bit = 1
    sim.set_logic_value(2, 3, 200);
    sim.set_logic_value(4, 3, 100);
    sim.dirty.mark_dirty(addcarry_idx);
    sim.propagate_combinational();
    let result = sim.get_logic_value_by_idx(addcarry_idx);
    assert_eq!(result & 0xFF, 44, "200 + 100 low byte = 44");
    assert_eq!((result >> 8) & 1, 1, "Carry set for 200 + 100");
    assert_eq!(result, 300, "Full 9-bit result = 300");

    // Test: 255 + 255 = 510 → low 8 bits = 254, carry = 1
    sim.set_logic_value(2, 3, 255);
    sim.set_logic_value(4, 3, 255);
    sim.dirty.mark_dirty(addcarry_idx);
    sim.propagate_combinational();
    let result = sim.get_logic_value_by_idx(addcarry_idx);
    assert_eq!(result & 0xFF, 254, "255 + 255 low byte = 254");
    assert_eq!((result >> 8) & 1, 1, "Carry set for 255 + 255");

    // Test: 0 + 0 = 0 (no carry)
    sim.set_logic_value(2, 3, 0);
    sim.set_logic_value(4, 3, 0);
    sim.dirty.mark_dirty(addcarry_idx);
    sim.propagate_combinational();
    let result = sim.get_logic_value_by_idx(addcarry_idx);
    assert_eq!(result, 0, "0 + 0 = 0");
}

#[test]
fn test_bitselect_tile() {
    let mut sim = Simulation::with_size(8, 8);
    let w = sim.width();

    // Layout: [Const(value)] [BitSelect] [Const(bit_pos)]
    for col in 2..=4 {
        sim.set_tile(col, 2, TileType::Const);
        sim.set_logic_value(col, 2, 0);
        sim.set_tile(col, 4, TileType::Const);
        sim.set_logic_value(col, 4, 0);
    }
    sim.set_tile(2, 3, TileType::Const);
    sim.set_tile(3, 3, TileType::BitSelect);
    sim.set_tile(4, 3, TileType::Const);

    let bs_idx = 3 * w + 3;

    // Test: value=0b1010_0101, bit 0 → 1 → MAX
    sim.set_logic_value(2, 3, 0xA5);
    sim.set_logic_value(4, 3, 0);
    sim.dirty.mark_dirty(bs_idx);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(bs_idx),
        u64::MAX,
        "Bit 0 of 0xA5 = 1 → MAX"
    );

    // Test: value=0b1010_0101, bit 1 → 0 → 0
    sim.set_logic_value(4, 3, 1);
    sim.dirty.mark_dirty(bs_idx);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(bs_idx),
        0,
        "Bit 1 of 0xA5 = 0 → 0"
    );

    // Test: value=0b1010_0101, bit 2 → 1 → MAX
    sim.set_logic_value(4, 3, 2);
    sim.dirty.mark_dirty(bs_idx);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(bs_idx),
        u64::MAX,
        "Bit 2 of 0xA5 = 1 → MAX"
    );

    // Test: value=0b1010_0101, bit 7 → 1 → MAX
    sim.set_logic_value(4, 3, 7);
    sim.dirty.mark_dirty(bs_idx);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(bs_idx),
        u64::MAX,
        "Bit 7 of 0xA5 = 1 → MAX"
    );

    // Test: value=0b1010_0101, bit 4 → 0 → 0
    sim.set_logic_value(4, 3, 4);
    sim.dirty.mark_dirty(bs_idx);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(bs_idx),
        0,
        "Bit 4 of 0xA5 = 0 → 0"
    );

    // Test: value=0, any bit → 0
    sim.set_logic_value(2, 3, 0);
    sim.set_logic_value(4, 3, 0);
    sim.dirty.mark_dirty(bs_idx);
    sim.propagate_combinational();
    assert_eq!(sim.get_logic_value_by_idx(bs_idx), 0, "Any bit of 0 = 0");
}

#[test]
fn test_builder_creates_tiles() {
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(4, 4)
        .with_program(&[0x12, 0x17, 0x31]) // LDI R0,#2; LDI R1,#3; ADD R0,R1
        .with_rom_size(16)
        .with_ram_size(16)
        .build(&mut sim);

    // Verify tiles were placed
    assert!(cpu.tile_count > 0);
    assert_eq!(cpu.origin, (4, 4));
}

#[test]
fn test_cpu_initial_state() {
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_initial_regs([10, 20, 30, 40])
        .build(&mut sim);

    // Check initial register values
    assert_eq!(cpu.read_reg(&sim, 0), 10);
    assert_eq!(cpu.read_reg(&sim, 1), 20);
    assert_eq!(cpu.read_reg(&sim, 2), 30);
    assert_eq!(cpu.read_reg(&sim, 3), 40);

    // PC should start at 0
    assert_eq!(cpu.read_pc(&sim), 0);
}

#[test]
fn test_tick_returns_timing_stats() {
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x20]) // MOV R0,R0 (NOP replacement)
        .build(&mut sim);

    let stats = cpu.tick(&mut sim);

    // Should have some propagation
    assert!(stats.total_deltas < u32::MAX);
    // Should converge
    assert!(stats.converged);
}

#[test]
fn test_run_collects_metrics() {
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x20, 0x20, 0x20]) // 3x MOV R0,R0 (NOP replacement)
        .build(&mut sim);

    let metrics = cpu.run(&mut sim, 5);

    assert!(metrics.cycles > 0);
    assert!(metrics.estimated_max_mhz > 0.0);
}

#[test]
fn test_control_signals_decode() {
    // NOP
    let signals = ControlSignals::decode(0x00);
    assert!(!signals.reg_write_en);
    assert!(!signals.is_branch());

    // LDI R0, #2
    let signals = ControlSignals::decode(0x12);
    assert!(signals.reg_write_en);
    assert_eq!(signals.reg_dst, 0);
    assert!(signals.use_immediate);
    assert_eq!(signals.immediate, 2);

    // ADD R0, R1
    let signals = ControlSignals::decode(0x31);
    assert!(signals.reg_write_en);
    assert_eq!(signals.alu_op, AluOp::Add);
    assert_eq!(signals.reg_dst, 0);
    assert_eq!(signals.reg_src_a, 0);
    assert_eq!(signals.reg_src_b, 1);
    assert!(signals.update_flags);

    // JMP - the lower 6 bits become the target
    // Note: Current encoding has opcode in bits 7-4, so bits 4-5 overlap with address
    // 0xB5 = opcode 0xB (JMP) with lower bits 0x05, but target = 0xB5 & 0x3F = 0x35
    let signals = ControlSignals::decode(0xB5);
    assert!(signals.jump);
    assert_eq!(signals.jump_target, 0x35); // Lower 6 bits of 0xB5
}

#[test]
fn test_alu_op_from_opcode() {
    assert_eq!(AluOp::from_opcode(0x3), AluOp::Add);
    assert_eq!(AluOp::from_opcode(0x4), AluOp::Sub);
    assert_eq!(AluOp::from_opcode(0x5), AluOp::And);
    assert_eq!(AluOp::from_opcode(0x6), AluOp::Or);
    assert_eq!(AluOp::from_opcode(0x7), AluOp::Xor);
}

#[test]
fn test_datapath_analysis() {
    let datapath = TileCpuDatapath::single_cycle();
    let analysis = datapath.analyze();

    // Single-cycle should have reasonable critical path
    assert!(analysis.total_critical_path > 0);
    assert!(analysis.estimated_max_mhz > 0.0);
    assert!(!analysis.bottleneck_stage.is_empty());
}

#[test]
fn test_pipeline_benefit() {
    let single = TileCpuDatapath::single_cycle();
    let pipelined = TileCpuDatapath::two_stage_pipeline();

    // Pipeline should have shorter critical path per stage
    assert!(pipelined.total_critical_path() < single.total_critical_path());
}

#[test]
fn test_layout_generation() {
    let layout = layout::generate_cpu_layout((0, 0), 16, 16);

    // Should have components
    assert!(!layout.components.is_empty());

    // Should have wires
    assert!(!layout.wires.is_empty());

    // Should have reasonable tile count
    assert!(layout.total_tiles > 0);
}

#[test]
fn test_placed_component_alu() {
    let alu = PlacedComponent::alu(10, 20);

    assert_eq!(alu.name, "ALU");
    assert_eq!(alu.x, 10);
    assert_eq!(alu.y, 20);
    assert!(!alu.tiles.is_empty());

    // ALU should have Add, Sub, And, Or, Xor tiles
    let tile_types: Vec<_> = alu.tiles.iter().map(|(_, _, t)| *t).collect();
    assert!(tile_types.contains(&TileType::Add));
    assert!(tile_types.contains(&TileType::Sub));
    assert!(tile_types.contains(&TileType::And));
    assert!(tile_types.contains(&TileType::Or));
    assert!(tile_types.contains(&TileType::Xor));
}

#[test]
fn test_wire_routing() {
    let h_wire = WireRoute::horizontal(
        ("A", "out"),
        ("B", "in"),
        10, // y
        5,  // x_start
        15, // x_end
    );

    assert_eq!(h_wire.path.len(), 11); // 5 to 15 inclusive
    assert_eq!(h_wire.delay, 11);

    // All tiles should be WireH
    for (_, _, tile_type) in &h_wire.path {
        assert_eq!(*tile_type, TileType::WireH);
    }
}

#[test]
fn test_cpu_dump_state() {
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_initial_regs([0xAB, 0xCD, 0xEF, 0x12])
        .build(&mut sim);

    let state = cpu.dump_state(&sim);

    assert!(state.contains("R0=AB"));
    assert!(state.contains("R1=CD"));
    assert!(state.contains("R2=EF"));
    assert!(state.contains("R3=12"));
    assert!(state.contains("PC=00"));
}

// =============================================================================
// Integration Tests - Full Execution
// =============================================================================

#[test]
fn test_full_execution_nop() {
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x20, 0x20, 0x20]) // 3x MOV R0,R0 (NOP replacement)
        .build(&mut sim);

    let metrics = cpu.run(&mut sim, 10);

    // Should execute without timing violations
    assert!(!metrics.had_timing_violation);
}

#[test]
fn test_ldi_instruction() {
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x12]) // LDI R0, #2
        .build(&mut sim);

    cpu.tick(&mut sim);

    // R0 should be 2
    assert_eq!(cpu.read_reg(&sim, 0), 2);
}

#[test]
fn test_add_instruction() {
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x12, 0x17, 0x31]) // LDI R0,#2; LDI R1,#3; ADD R0,R1
        .build(&mut sim);

    cpu.run(&mut sim, 3);

    // R0 should be 2 + 3 = 5
    assert_eq!(cpu.read_reg(&sim, 0), 5);
}

// =============================================================================
// Flag Tests
// =============================================================================

#[test]
fn test_sub_updates_zero_flag() {
    // SUB R0, R1 where R0 == R1 should set Z=1
    let mut sim = Simulation::with_size(64, 64);

    // LDI R0, #3 = 0x13; LDI R1, #3 = 0x17; SUB R0, R1 = 0x41
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x13, 0x17, 0x41])
        .build(&mut sim);

    cpu.run(&mut sim, 3);

    assert_eq!(cpu.read_reg(&sim, 0), 0, "R0 should be 3-3=0");
    assert!(cpu.read_flag_z(&sim), "Zero flag should be set");
    assert!(
        !cpu.read_flag_c(&sim),
        "Carry flag should be clear (no borrow)"
    );
}

#[test]
fn test_sub_clears_zero_flag() {
    // SUB R0, R1 where R0 != R1 should clear Z
    let mut sim = Simulation::with_size(64, 64);

    // LDI R0, #3 = 0x13; LDI R1, #1 = 0x15; SUB R0, R1 = 0x41
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x13, 0x15, 0x41])
        .build(&mut sim);

    cpu.run(&mut sim, 3);

    assert_eq!(cpu.read_reg(&sim, 0), 2, "R0 should be 3-1=2");
    assert!(!cpu.read_flag_z(&sim), "Zero flag should be clear");
}

#[test]
fn test_sub_sets_carry_on_borrow() {
    // SUB where A < B should set carry (borrow)
    let mut sim = Simulation::with_size(64, 64);

    // LDI R0, #1 = 0x11; LDI R1, #3 = 0x17; SUB R0, R1 = 0x41
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x11, 0x17, 0x41])
        .build(&mut sim);

    cpu.run(&mut sim, 3);

    assert_eq!(
        cpu.read_reg(&sim, 0),
        254,
        "R0 should be 1-3=254 (wrapping)"
    );
    assert!(!cpu.read_flag_z(&sim), "Zero flag should be clear");
    assert!(cpu.read_flag_c(&sim), "Carry flag should be set (borrow)");
}

#[test]
fn test_cmp_updates_flags_without_writeback() {
    // CMP sets flags but doesn't modify registers
    let mut sim = Simulation::with_size(64, 64);

    // LDI R0, #2 = 0x12; LDI R1, #2 = 0x16; CMP R0, R1 = 0xA1
    // CMP: opcode=0xA, rd=0 (R0), rs=1 (R1) → 0xA0 | (0<<2) | 1 = 0xA1
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x12, 0x16, 0xA1])
        .build(&mut sim);

    cpu.run(&mut sim, 3);

    assert_eq!(
        cpu.read_reg(&sim, 0),
        2,
        "R0 should be unchanged (CMP doesn't write)"
    );
    assert!(cpu.read_flag_z(&sim), "Zero flag should be set (R0 == R1)");
}

// =============================================================================
// Branch Tests
// =============================================================================

#[test]
fn test_jz_taken() {
    // JZ should jump when Z flag is set
    let mut sim = Simulation::with_size(64, 64);

    // addr 0: LDI R0, #0 = 0x10
    // addr 1: LDI R1, #0 = 0x14
    // addr 2: CMP R0, R1 = 0xA1 (R0 - R1 = 0, Z=1)
    // addr 3: JZ 5       = 0xC5 (Z set → jump to addr 5, skip addr 4)
    // addr 4: LDI R2, #1 = 0x19 (should be SKIPPED)
    // addr 5: LDI R2, #2 = 0x1A (should execute)
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x10, 0x14, 0xA1, 0xC5, 0x19, 0x1A])
        .build(&mut sim);

    cpu.run(&mut sim, 6);

    assert_eq!(
        cpu.read_reg(&sim, 2),
        2,
        "R2 should be 2 (JZ skipped addr 4)"
    );
}

#[test]
fn test_jz_not_taken() {
    // JZ should NOT jump when Z flag is clear
    let mut sim = Simulation::with_size(64, 64);

    // addr 0: LDI R0, #1 = 0x11
    // addr 1: LDI R1, #0 = 0x14
    // addr 2: CMP R0, R1 = 0xA1 (R0 - R1 = 1, Z=0)
    // addr 3: JZ 5       = 0xC5 (Z clear → NOT taken)
    // addr 4: LDI R2, #1 = 0x19 (should execute)
    // addr 5: LDI R2, #2 = 0x1A (also executes — not a halt)
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x11, 0x14, 0xA1, 0xC5, 0x19, 0x1A])
        .build(&mut sim);

    cpu.run(&mut sim, 5);

    // After 5 ticks: executed addrs 0-4, R2 set by addr 4
    assert_eq!(
        cpu.read_reg(&sim, 2),
        1,
        "R2 should be 1 (JZ not taken, addr 4 executed)"
    );
}

// =============================================================================
// Memory Tests
// =============================================================================

#[test]
fn test_store_and_load() {
    // ST stores Rs to RAM[R0], LD reads RAM[R0] into Rd
    let mut sim = Simulation::with_size(64, 64);

    // addr 0: LDI R0, #0 = 0x10 (address = 0)
    // addr 1: LDI R1, #3 = 0x17 (value to store)
    // addr 2: ST R1       = 0xF1 (RAM[0] = R1 = 3)
    // addr 3: LD R2       = 0xE8 (R2 = RAM[0] = 3)
    //   LD Rd: opcode=0xE, rd=2 → 0xE0 | (2<<2) = 0xE8
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x10, 0x17, 0xF1, 0xE8])
        .build(&mut sim);

    cpu.run(&mut sim, 4);

    // Debug: step manually to pinpoint failure
    // Verify intermediate state at each step
    // (We already ran 4 ticks via run above, just check final state)
    assert_eq!(cpu.read_ram(&sim, 0), 3, "RAM[0] should be 3 after ST");
    assert_eq!(cpu.read_reg(&sim, 2), 3, "R2 should be 3 after LD");
}

#[test]
fn test_ram_survives_tick() {
    // Verify RAM values persist through clock cycles
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x20]) // MOV R0,R0 (NOP replacement)
        .build(&mut sim);

    cpu.write_ram(&mut sim, 0, 42);
    assert_eq!(cpu.read_ram(&sim, 0), 42, "Direct write/read should work");

    sim.tick_with_delays();
    sim.tick_with_delays();
    assert_eq!(cpu.read_ram(&sim, 0), 42, "RAM should survive tick cycle");
}

// =============================================================================
// Loop Test — Countdown using JZ
// =============================================================================

#[test]
fn test_countdown_loop() {
    // Counts R0 down from 3 to 0 using a JZ-based loop.
    //
    // addr 0: LDI R0, #3   = 0x13  (counter)
    // addr 1: LDI R1, #1   = 0x15  (decrement value)
    // addr 2: SUB R0, R1   = 0x41  (R0 -= 1, updates Z)
    // addr 3: JZ 7         = 0xC7  (if zero, exit to addr 7 = past program)
    // addr 4: CMP R1, R1   = 0xA5  (force Z=1 for unconditional JZ)
    //   CMP: opcode=0xA, rd=1, rs=1 → 0xA0 | (1<<2) | 1 = 0xA5
    // addr 5: JZ 2         = 0xC2  (Z=1, jump back to SUB)
    // addr 6: NOP           = 0x00  (never reached)
    // addr 7: NOP           = 0x00  (exit point)
    //
    // Execution trace:
    //   tick  1: addr 0 → R0=3
    //   tick  2: addr 1 → R1=1
    //   tick  3: addr 2 → R0=2, Z=0
    //   tick  4: addr 3 → JZ not taken
    //   tick  5: addr 4 → CMP R1,R1 → Z=1
    //   tick  6: addr 5 → JZ 2 taken
    //   tick  7: addr 2 → R0=1, Z=0
    //   tick  8: addr 3 → JZ not taken
    //   tick  9: addr 4 → CMP R1,R1 → Z=1
    //   tick 10: addr 5 → JZ 2 taken
    //   tick 11: addr 2 → R0=0, Z=1
    //   tick 12: addr 3 → JZ 7 taken (exit)
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x13, 0x15, 0x41, 0xC7, 0xA5, 0xC2, 0x00, 0x00])
        .build(&mut sim);

    cpu.run(&mut sim, 12);

    assert_eq!(cpu.read_reg(&sim, 0), 0, "R0 should be 0 after countdown");
    assert!(cpu.read_flag_z(&sim), "Zero flag should be set");
}

#[test]
fn test_mov_instruction() {
    // MOV Rd, Rs copies a register
    let mut sim = Simulation::with_size(64, 64);

    // addr 0: LDI R1, #3 = 0x17 (R1 = 3)
    // addr 1: MOV R0, R1 = 0x21
    //   MOV: opcode=0x2, rd=0, rs=1 → 0x20 | (0<<2) | 1 = 0x21
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x17, 0x21])
        .build(&mut sim);

    cpu.run(&mut sim, 2);

    assert_eq!(cpu.read_reg(&sim, 0), 3, "R0 should be 3 (copied from R1)");
    assert_eq!(cpu.read_reg(&sim, 1), 3, "R1 should still be 3");
}

// =============================================================================
// Tile ALU Tests — verify ALU operations execute through actual tiles
// =============================================================================

#[test]
fn test_tile_alu_add() {
    // ADD R0, R1 should execute through the Add tile
    let mut sim = Simulation::with_size(64, 64);

    // LDI R0, #3 = 0x13; LDI R1, #2 = 0x16; ADD R0, R1 = 0x31
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x13, 0x16, 0x31])
        .build(&mut sim);

    cpu.run(&mut sim, 3);

    assert_eq!(cpu.read_reg(&sim, 0), 5, "R0 should be 3+2=5 via tile Add");
}

#[test]
fn test_tile_alu_add_overflow() {
    // ADD with overflow: 200 + 100 = 44 (wrapping), carry set
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_initial_regs([200, 100, 0, 0])
        .with_program(&[0x31]) // ADD R0, R1
        .build(&mut sim);

    cpu.tick(&mut sim);

    assert_eq!(
        cpu.read_reg(&sim, 0),
        44,
        "R0 should be 200+100=44 (wrapping)"
    );
    assert!(
        cpu.read_flag_c(&sim),
        "Carry flag should be set on overflow"
    );
}

#[test]
fn test_tile_alu_sub() {
    // SUB R0, R1 should execute through the Sub tile
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_initial_regs([10, 3, 0, 0])
        .with_program(&[0x41]) // SUB R0, R1
        .build(&mut sim);

    cpu.tick(&mut sim);

    assert_eq!(cpu.read_reg(&sim, 0), 7, "R0 should be 10-3=7 via tile Sub");
}

#[test]
fn test_tile_alu_and() {
    // AND R0, R1 should execute through the And tile
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_initial_regs([0xFF, 0x0F, 0, 0])
        .with_program(&[0x51]) // AND R0, R1
        .build(&mut sim);

    cpu.tick(&mut sim);

    assert_eq!(
        cpu.read_reg(&sim, 0),
        0x0F,
        "R0 should be 0xFF & 0x0F = 0x0F via tile And"
    );
}

#[test]
fn test_tile_alu_or_xor() {
    // Test OR and XOR through their respective tiles
    let mut sim = Simulation::with_size(64, 64);

    // OR R0, R1: opcode=0x6, rd=0, rs=1 → 0x61
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_initial_regs([0xA0, 0x05, 0, 0])
        .with_program(&[0x61]) // OR R0, R1
        .build(&mut sim);

    cpu.tick(&mut sim);
    assert_eq!(
        cpu.read_reg(&sim, 0),
        0xA5,
        "R0 should be 0xA0 | 0x05 = 0xA5 via tile Or"
    );

    // Now test XOR
    let mut sim2 = Simulation::with_size(64, 64);

    // XOR R0, R1: opcode=0x7, rd=0, rs=1 → 0x71
    let cpu2 = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_initial_regs([0xFF, 0x0F, 0, 0])
        .with_program(&[0x71]) // XOR R0, R1
        .build(&mut sim2);

    cpu2.tick(&mut sim2);
    assert_eq!(
        cpu2.read_reg(&sim2, 0),
        0xF0,
        "R0 should be 0xFF ^ 0x0F = 0xF0 via tile Xor"
    );
}

// =============================================================================
// Tile ALU Tests — Not, Shl, Shr (Phase 1 expansion)
// =============================================================================

#[test]
fn test_tile_alu_not() {
    let mut sim = Simulation::with_size(64, 64);

    // NOT R0: opcode=0x8, rd=0 → 0x80
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_initial_regs([0xA5, 0, 0, 0])
        .with_program(&[0x80]) // SHL R0... wait, 0x8 is SHL
        .build(&mut sim);

    // Actually NOT doesn't exist as a separate opcode in the ISA!
    // The ISA has SHL=0x8, SHR=0x9. There's no NOT opcode.
    // Not is only used internally via AluOp. Let's test SHL/SHR instead.
    drop(cpu);
    drop(sim);

    // Test NOT via direct tile ALU path with CMP (which uses Sub tile)
    // We can't test Not directly since there's no NOT instruction,
    // but we can verify the tile works via write_alu_operands
}

#[test]
fn test_tile_alu_shl() {
    // SHL R0: opcode=0x8, rd=0 → 0x80
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_initial_regs([0x15, 0, 0, 0])
        .with_program(&[0x80]) // SHL R0
        .build(&mut sim);

    cpu.tick(&mut sim);

    assert_eq!(
        cpu.read_reg(&sim, 0),
        0x2A,
        "R0 should be 0x15 << 1 = 0x2A via tile Shl"
    );
}

#[test]
fn test_tile_alu_shr() {
    // SHR R0: opcode=0x9, rd=0 → 0x90
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_initial_regs([0x2A, 0, 0, 0])
        .with_program(&[0x90]) // SHR R0
        .build(&mut sim);

    cpu.tick(&mut sim);

    assert_eq!(
        cpu.read_reg(&sim, 0),
        0x15,
        "R0 should be 0x2A >> 1 = 0x15 via tile Shr"
    );
}

// =============================================================================
// Mux4to1 Standalone Test
// =============================================================================

#[test]
fn test_mux4to1_tile_standalone() {
    // Place a Mux4to1 tile with known packed data above and select below.
    // Mux4to1 reads up=data, down=select, outputs selected byte.
    let mut sim = Simulation::with_size(16, 16);
    let w = sim.width();

    // Layout (column 5):
    //   row 3: Const(0) — guard above data
    //   row 4: Const(packed_data) — data source (up neighbor of Mux4to1)
    //   row 5: Mux4to1
    //   row 6: Const(select) — select source (down neighbor of Mux4to1)
    //   row 7: Const(0) — guard below select

    // Pack 4 bytes: R0=0xAA at byte 0, R1=0xBB at byte 1, R2=0xCC at byte 2, R3=0xDD at byte 3
    let packed: u64 = 0xAA | (0xBB << 8) | (0xCC << 16) | (0xDD << 24);

    // Guards
    sim.set_tile(5, 3, TileType::Const);
    sim.set_logic_value(5, 3, 0);
    sim.set_tile(5, 7, TileType::Const);
    sim.set_logic_value(5, 7, 0);

    // Data Const above
    sim.set_tile(5, 4, TileType::Const);
    sim.set_logic_value(5, 4, packed);

    // Mux4to1
    sim.set_tile(5, 5, TileType::Mux4to1);

    // Select Const below
    sim.set_tile(5, 6, TileType::Const);

    // Also guard horizontally to prevent Wire leakage
    for row in 3..=7 {
        sim.set_tile(4, row, TileType::Const);
        sim.set_logic_value(4, row, 0);
        sim.set_tile(6, row, TileType::Const);
        sim.set_logic_value(6, row, 0);
    }

    let mux_idx = 5 * w + 5;

    // Test select=0 → byte 0 = 0xAA
    sim.set_logic_value(5, 6, 0);
    sim.dirty.mark_dirty(mux_idx);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(mux_idx) as u8,
        0xAA,
        "Select 0 should yield byte 0 (0xAA)"
    );

    // Test select=1 → byte 1 = 0xBB
    sim.set_logic_value(5, 6, 1);
    sim.dirty.mark_dirty(mux_idx);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(mux_idx) as u8,
        0xBB,
        "Select 1 should yield byte 1 (0xBB)"
    );

    // Test select=2 → byte 2 = 0xCC
    sim.set_logic_value(5, 6, 2);
    sim.dirty.mark_dirty(mux_idx);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(mux_idx) as u8,
        0xCC,
        "Select 2 should yield byte 2 (0xCC)"
    );

    // Test select=3 → byte 3 = 0xDD
    sim.set_logic_value(5, 6, 3);
    sim.dirty.mark_dirty(mux_idx);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(mux_idx) as u8,
        0xDD,
        "Select 3 should yield byte 3 (0xDD)"
    );
}

// =============================================================================
// Binary Mux Tree Tests — 4:1 and 8:1 without packed data
// =============================================================================

#[test]
fn test_mux_tree_4to1_standalone() {
    // Build a 4:1 binary mux tree in isolation and verify all selector combinations.
    let mut sim = Simulation::with_size(32, 32);
    let tree = {
        let mut ctx = WiringContext::new(&mut sim, (0, 0));
        ctx.wire_mux_tree_4to1(4, 4)
    };

    // Initial settle
    sim.dirty.mark_all_dirty(sim.tile_count());
    sim.propagate_combinational();

    // Set data inputs: D0=10, D1=20, D2=30, D3=40
    sim.set_logic_value_by_idx(tree.data_indices[0], 10);
    sim.set_logic_value_by_idx(tree.data_indices[1], 20);
    sim.set_logic_value_by_idx(tree.data_indices[2], 30);
    sim.set_logic_value_by_idx(tree.data_indices[3], 40);

    // Test sel=0 (S0=0, S1=0) -> D0 = 10
    sim.set_logic_value_by_idx(tree.sel0_indices[0], 0);
    sim.set_logic_value_by_idx(tree.sel0_indices[1], 0);
    sim.set_logic_value_by_idx(tree.sel1_idx, 0);
    sim.dirty.mark_dirty(tree.leaf_indices[0]);
    sim.dirty.mark_dirty(tree.leaf_indices[1]);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(tree.root_idx) as u8,
        10,
        "sel=0 should output D0=10"
    );

    // Test sel=1 (S0=MAX, S1=0) -> D1 = 20
    sim.set_logic_value_by_idx(tree.sel0_indices[0], u64::MAX);
    sim.set_logic_value_by_idx(tree.sel0_indices[1], u64::MAX);
    sim.set_logic_value_by_idx(tree.sel1_idx, 0);
    sim.dirty.mark_dirty(tree.leaf_indices[0]);
    sim.dirty.mark_dirty(tree.leaf_indices[1]);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(tree.root_idx) as u8,
        20,
        "sel=1 should output D1=20"
    );

    // Test sel=2 (S0=0, S1=MAX) -> D2 = 30
    sim.set_logic_value_by_idx(tree.sel0_indices[0], 0);
    sim.set_logic_value_by_idx(tree.sel0_indices[1], 0);
    sim.set_logic_value_by_idx(tree.sel1_idx, u64::MAX);
    sim.dirty.mark_dirty(tree.leaf_indices[0]);
    sim.dirty.mark_dirty(tree.leaf_indices[1]);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(tree.root_idx) as u8,
        30,
        "sel=2 should output D2=30"
    );

    // Test sel=3 (S0=MAX, S1=MAX) -> D3 = 40
    sim.set_logic_value_by_idx(tree.sel0_indices[0], u64::MAX);
    sim.set_logic_value_by_idx(tree.sel0_indices[1], u64::MAX);
    sim.set_logic_value_by_idx(tree.sel1_idx, u64::MAX);
    sim.dirty.mark_dirty(tree.leaf_indices[0]);
    sim.dirty.mark_dirty(tree.leaf_indices[1]);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(tree.root_idx) as u8,
        40,
        "sel=3 should output D3=40"
    );
}

#[test]
fn test_mux_tree_8to1_standalone() {
    // Build an 8:1 binary mux tree and verify all 8 selector combinations.
    let mut sim = Simulation::with_size(32, 32);
    let tree = {
        let mut ctx = WiringContext::new(&mut sim, (0, 0));
        ctx.wire_mux_tree_8to1(2, 2)
    };

    // Initial settle
    sim.dirty.mark_all_dirty(sim.tile_count());
    sim.propagate_combinational();

    // Set data inputs: D0=10, D1=20, ..., D7=80
    for i in 0..8 {
        sim.set_logic_value_by_idx(tree.data_indices[i], (i as u64 + 1) * 10);
    }

    // Helper: set selectors and propagate
    let expected = [10u8, 20, 30, 40, 50, 60, 70, 80];
    for sel in 0..8u64 {
        let s0 = if sel & 1 != 0 { u64::MAX } else { 0 };
        let s1 = if sel & 2 != 0 { u64::MAX } else { 0 };
        let s2 = if sel & 4 != 0 { u64::MAX } else { 0 };

        // Write S0 to all 4 leaf pairs
        for idx in &tree.sel0_indices {
            sim.set_logic_value_by_idx(*idx, s0);
        }
        // Write S1 to both middle selectors
        for idx in &tree.sel1_indices {
            sim.set_logic_value_by_idx(*idx, s1);
        }
        // Write S2
        sim.set_logic_value_by_idx(tree.sel2_idx, s2);

        // Mark all leaf muxes dirty
        for idx in &tree.leaf_indices {
            sim.dirty.mark_dirty(*idx);
        }
        sim.propagate_combinational();

        assert_eq!(
            sim.get_logic_value_by_idx(tree.root_idx) as u8,
            expected[sel as usize],
            "sel={} should output D{}={}",
            sel,
            sel,
            expected[sel as usize]
        );
    }
}

#[test]
fn test_arithmetic_chain_shl_debug() {
    // Reproduces the benchmark failure: LDI R0,#2; LDI R1,#3; ADD R0,R1; SHL R0; SHL R0; SUB R0,R1; AND R0,R1
    // Expected: R0 = ((5<<1)<<1 - 3) & 3 = (20 - 3) & 3 = 17 & 3 = 1
    let mut sim = Simulation::with_size(64, 64);
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x12, 0x17, 0x31, 0x80, 0x80, 0x41, 0x51])
        .build(&mut sim);

    // Trace each instruction
    // Tick 1: LDI R0,#2
    cpu.tick(&mut sim);
    assert_eq!(cpu.read_reg(&sim, 0), 2, "After LDI R0,#2: R0 should be 2");
    assert_eq!(cpu.read_reg(&sim, 1), 0, "After LDI R0,#2: R1 should be 0");

    // Tick 2: LDI R1,#3
    cpu.tick(&mut sim);
    assert_eq!(cpu.read_reg(&sim, 0), 2, "After LDI R1,#3: R0 should be 2");
    assert_eq!(cpu.read_reg(&sim, 1), 3, "After LDI R1,#3: R1 should be 3");

    // Tick 3: ADD R0,R1
    cpu.tick(&mut sim);
    assert_eq!(cpu.read_reg(&sim, 0), 5, "After ADD R0,R1: R0 should be 5");

    // Tick 4: SHL R0 (R0 = 5 << 1 = 10)
    cpu.tick(&mut sim);
    assert_eq!(
        cpu.read_reg(&sim, 0),
        10,
        "After SHL R0: R0 should be 10 (5<<1)"
    );

    // Tick 5: SHL R0 (R0 = 10 << 1 = 20)
    cpu.tick(&mut sim);
    assert_eq!(
        cpu.read_reg(&sim, 0),
        20,
        "After SHL R0: R0 should be 20 (10<<1)"
    );

    // Tick 6: SUB R0,R1 (R0 = 20 - 3 = 17)
    cpu.tick(&mut sim);
    assert_eq!(
        cpu.read_reg(&sim, 0),
        17,
        "After SUB R0,R1: R0 should be 17"
    );

    // Tick 7: AND R0,R1 (R0 = 17 & 3 = 1)
    cpu.tick(&mut sim);
    assert_eq!(cpu.read_reg(&sim, 0), 1, "After AND R0,R1: R0 should be 1");
}

// =============================================================================
// Physical CPU Tests — tile-based decoder
// =============================================================================

/// Test that the physical decoder circuit produces correct LUT outputs after settling.
///
/// The physical decoder: IR → Shr(IR,4) → WD chain → 4 Mux8to1 LUTs.
/// After build_physical(), the LUTs should contain the correct ctrl_a/ctrl_b
/// for instruction[0] in the program.
#[test]
fn test_physical_decoder_nop() {
    // NOP = 0x00: opcode=0, ctrl_a_lo[0]=0x00, ctrl_b_lo[0]=0x00
    let mut sim = Simulation::with_size_layered(128, 128, 2);
    let cpu = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&[0x00]) // NOP
        .build_physical(&mut sim);

    // After settling, IR should be the NOP instruction
    let ir = sim.get_logic_value_by_idx(cpu.ir_idx) as u8;
    assert_eq!(ir, 0x00, "IR should be NOP (0x00)");

    // Shr output should be opcode = 0x00 >> 4 = 0
    let opcode = sim.get_logic_value_by_idx(cpu.shr_opcode_idx) as u8;
    assert_eq!(opcode, 0, "Shr output should be opcode 0 for NOP");

    // LUT a_lo[0] = 0x00 (NOP: no alu_sel, no reg_wr, no flags, no imm)
    let ctrl_a_lo = sim.get_logic_value_by_idx(cpu.decoder_lut_indices[0]) as u8;
    assert_eq!(ctrl_a_lo, 0x00, "ctrl_a for NOP should be 0x00");

    // LUT b_lo[0] = 0x00
    let ctrl_b_lo = sim.get_logic_value_by_idx(cpu.decoder_lut_indices[2]) as u8;
    assert_eq!(ctrl_b_lo, 0x00, "ctrl_b for NOP should be 0x00");
}

#[test]
fn test_physical_decoder_ldi() {
    // LDI R0,#2 = 0x12: opcode=1, ctrl_a_lo[1]=0x28, ctrl_b_lo[1]=0x00
    let mut sim = Simulation::with_size_layered(128, 128, 2);
    let cpu = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&[0x12]) // LDI R0, #2
        .build_physical(&mut sim);

    let ir = sim.get_logic_value_by_idx(cpu.ir_idx) as u8;
    assert_eq!(ir, 0x12, "IR should be LDI R0,#2 (0x12)");

    let opcode = sim.get_logic_value_by_idx(cpu.shr_opcode_idx) as u8;
    assert_eq!(opcode, 1, "Shr output should be opcode 1 for LDI");

    // ctrl_a_lo[1] = 0x28: alu_sel=0(PassB implicit), reg_wr=1(bit3),
    // upd_flags=0(bit4), use_imm=1(bit5) → 0x08 | 0x20 = 0x28
    let ctrl_a_lo = sim.get_logic_value_by_idx(cpu.decoder_lut_indices[0]) as u8;
    assert_eq!(
        ctrl_a_lo, 0x28,
        "ctrl_a for LDI should be 0x28 (reg_wr + use_imm)"
    );

    let ctrl_b_lo = sim.get_logic_value_by_idx(cpu.decoder_lut_indices[2]) as u8;
    assert_eq!(ctrl_b_lo, 0x00, "ctrl_b for LDI should be 0x00");
}

#[test]
fn test_physical_decoder_add() {
    // ADD R0,R1 = 0x31: opcode=3, ctrl_a_lo[3]=0x18
    let mut sim = Simulation::with_size_layered(128, 128, 2);
    let cpu = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&[0x31]) // ADD R0, R1
        .build_physical(&mut sim);

    let opcode = sim.get_logic_value_by_idx(cpu.shr_opcode_idx) as u8;
    assert_eq!(opcode, 3, "Shr output should be opcode 3 for ADD");

    // ctrl_a_lo[3] = 0x18: alu_sel=0(Add mapped to 0 in LUT? No, Add=0x18)
    // Wait: LUT encoding = alu_sel | (reg_wr << 3) | (upd_flags << 4) | (use_imm << 5)
    // ADD: alu_sel=0 (Add tile is index 0), reg_wr=1, upd_flags=1, use_imm=0
    // = 0 | 8 | 16 = 0x18
    let ctrl_a_lo = sim.get_logic_value_by_idx(cpu.decoder_lut_indices[0]) as u8;
    assert_eq!(
        ctrl_a_lo, 0x18,
        "ctrl_a for ADD should be 0x18 (Add, reg_wr, upd_flags)"
    );
}

#[test]
fn test_physical_decoder_hi_bank_shl() {
    // SHL R0 = 0x80: opcode=8, ctrl_a_hi[0]=0x1E
    let mut sim = Simulation::with_size_layered(128, 128, 2);
    let cpu = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&[0x80]) // SHL R0
        .build_physical(&mut sim);

    let opcode = sim.get_logic_value_by_idx(cpu.shr_opcode_idx) as u8;
    assert_eq!(opcode, 8, "Shr output should be opcode 8 for SHL");

    // For opcode 8 (bit 3 set), use hi bank
    // ctrl_a_hi[0] = 0x1E: alu_sel=6(Shl), reg_wr=1, upd_flags=1
    // = 6 | 8 | 16 = 0x1E
    let ctrl_a_hi = sim.get_logic_value_by_idx(cpu.decoder_lut_indices[1]) as u8;
    assert_eq!(ctrl_a_hi, 0x1E, "ctrl_a_hi for SHL should be 0x1E");

    // ctrl_b_hi[0] = 0x00
    let ctrl_b_hi = sim.get_logic_value_by_idx(cpu.decoder_lut_indices[3]) as u8;
    assert_eq!(ctrl_b_hi, 0x00, "ctrl_b_hi for SHL should be 0x00");
}

#[test]
fn test_physical_decoder_jmp() {
    // JMP target = 0xB0 + addr. JMP is opcode 0xB, ctrl_b_hi[3]=0x01
    // But JMP addr encoding: instruction & 0x3F. For JMP to addr 0x35: 0xB5
    // opcode = 0xB = 11, opcode & 7 = 3
    let mut sim = Simulation::with_size_layered(128, 128, 2);
    let cpu = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&[0xB5]) // JMP (target=0x35)
        .build_physical(&mut sim);

    let opcode = sim.get_logic_value_by_idx(cpu.shr_opcode_idx) as u8;
    assert_eq!(opcode, 0x0B, "Shr output should be opcode 0x0B for JMP");

    // opcode & 7 = 3, so LUT selects byte[3]
    // ctrl_a_hi[3] = 0x00 (JMP has no ALU, no reg_wr)
    let ctrl_a_hi = sim.get_logic_value_by_idx(cpu.decoder_lut_indices[1]) as u8;
    assert_eq!(ctrl_a_hi, 0x00, "ctrl_a_hi for JMP should be 0x00");

    // ctrl_b_hi[3] = 0x01 (jmp bit)
    let ctrl_b_hi = sim.get_logic_value_by_idx(cpu.decoder_lut_indices[3]) as u8;
    assert_eq!(ctrl_b_hi, 0x01, "ctrl_b_hi for JMP should be 0x01");
}

/// Parity test: physical CPU LDI instruction produces same result as hybrid CPU
#[test]
fn test_physical_parity_ldi() {
    // LDI R0,#2; LDI R1,#3
    let program = [0x12, 0x17];

    // Hybrid CPU
    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build(&mut sim_h);
    cpu_h.tick(&mut sim_h); // LDI R0,#2
    cpu_h.tick(&mut sim_h); // LDI R1,#3

    // Physical CPU
    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);
    cpu_p.tick(&mut sim_p); // LDI R0,#2
    cpu_p.tick(&mut sim_p); // LDI R1,#3

    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        cpu_p.read_reg(&sim_p, 0),
        "R0 parity: hybrid={}, physical={}",
        cpu_h.read_reg(&sim_h, 0),
        cpu_p.read_reg(&sim_p, 0)
    );
    assert_eq!(
        cpu_h.read_reg(&sim_h, 1),
        cpu_p.read_reg(&sim_p, 1),
        "R1 parity: hybrid={}, physical={}",
        cpu_h.read_reg(&sim_h, 1),
        cpu_p.read_reg(&sim_p, 1)
    );
    assert_eq!(
        cpu_h.read_pc(&sim_h),
        cpu_p.read_pc(&sim_p),
        "PC parity: hybrid={}, physical={}",
        cpu_h.read_pc(&sim_h),
        cpu_p.read_pc(&sim_p)
    );
}

/// Parity test: physical CPU ADD instruction produces same result as hybrid CPU
#[test]
fn test_physical_parity_add() {
    // LDI R0,#2; LDI R1,#3; ADD R0,R1
    let program = [0x12, 0x17, 0x31];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build(&mut sim_h);
    for _ in 0..3 {
        cpu_h.tick(&mut sim_h);
    }

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);
    for _ in 0..3 {
        cpu_p.tick(&mut sim_p);
    }

    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        cpu_p.read_reg(&sim_p, 0),
        "R0 after ADD parity: hybrid={}, physical={}",
        cpu_h.read_reg(&sim_h, 0),
        cpu_p.read_reg(&sim_p, 0)
    );
    assert_eq!(cpu_h.read_reg(&sim_h, 0), 5, "2 + 3 = 5");
}

/// Debug: trace writeback bus signal path for ADD
#[test]
fn test_physical_writeback_bus_debug() {
    let program = [0x12, 0x17, 0x31]; // LDI R0,#2; LDI R1,#3; ADD R0,R1
    let ox = 2usize;
    let oy = 2usize;
    let gw = 128usize;
    let layer_size = 128 * 128;

    let mut sim = Simulation::with_size_layered(128, 128, 2);
    let cpu = TileCpuBuilder::new()
        .with_origin(ox, oy)
        .with_program(&program)
        .build_physical(&mut sim);

    // Tick 1: LDI R0,#2
    cpu.tick(&mut sim);
    assert_eq!(cpu.read_reg(&sim, 0), 2, "After tick 1: R0=2");

    // Tick 2: LDI R1,#3
    cpu.tick(&mut sim);
    assert_eq!(cpu.read_reg(&sim, 0), 2, "After tick 2: R0 still 2");
    assert_eq!(cpu.read_reg(&sim, 1), 3, "After tick 2: R1=3");

    // After tick 2, check what the ALU result tree root has
    // ALU root is at L0@(ox+20, oy+54)
    let alu_root_l0_idx = (oy + 54) * gw + (ox + 20);
    let alu_root_val = sim.get_logic_value_by_idx(alu_root_l0_idx);
    eprintln!(
        "After tick 2: ALU root L0@({},{}) = {}",
        ox + 20,
        oy + 54,
        alu_root_val
    );

    // Check L1 ViaDown at same position
    let viadown_l1_idx = 1 * layer_size + (oy + 54) * gw + (ox + 20);
    let viadown_val = sim.get_logic_value_by_idx(viadown_l1_idx);
    eprintln!(
        "After tick 2: ViaDown L1@({},{}) = {}",
        ox + 20,
        oy + 54,
        viadown_val
    );

    // Check L1 WireLeft at (ox+4, oy+54)
    let wl_l1_idx = 1 * layer_size + (oy + 54) * gw + (ox + 4);
    let wl_val = sim.get_logic_value_by_idx(wl_l1_idx);
    eprintln!(
        "After tick 2: WireLeft L1@({},{}) = {}",
        ox + 4,
        oy + 54,
        wl_val
    );

    // Check L1 WireUp spine at (ox+4, oy+28)
    let wu_spine_idx = 1 * layer_size + (oy + 28) * gw + (ox + 4);
    let wu_spine_val = sim.get_logic_value_by_idx(wu_spine_idx);
    eprintln!(
        "After tick 2: WireUp spine L1@({},{}) = {}",
        ox + 4,
        oy + 28,
        wu_spine_val
    );

    // Check L1 WireRight at (ox+39, oy+28)
    let wr_l1_idx = 1 * layer_size + (oy + 28) * gw + (ox + 39);
    let wr_val = sim.get_logic_value_by_idx(wr_l1_idx);
    eprintln!(
        "After tick 2: WireRight L1@({},{}) = {}",
        ox + 39,
        oy + 28,
        wr_val
    );

    // Check Reg0 WireUp branch at (ox+39, oy+21)
    let reg0_wu_idx = 1 * layer_size + (oy + 21) * gw + (ox + 39);
    let reg0_wu_val = sim.get_logic_value_by_idx(reg0_wu_idx);
    eprintln!(
        "After tick 2: Reg0 WireUp L1@({},{}) = {}",
        ox + 39,
        oy + 21,
        reg0_wu_val
    );

    // Check L1 merge Mux at (ox+40, oy+21)
    let merge_mux_idx = 1 * layer_size + (oy + 21) * gw + (ox + 40);
    let merge_mux_val = sim.get_logic_value_by_idx(merge_mux_idx);
    eprintln!(
        "After tick 2: Merge Mux L1@({},{}) = {}",
        ox + 40,
        oy + 21,
        merge_mux_val
    );

    // Check L1 mem_read selector at (ox+40, oy+20)
    let mem_read_idx = 1 * layer_size + (oy + 20) * gw + (ox + 40);
    let mem_read_val = sim.get_logic_value_by_idx(mem_read_idx);
    eprintln!(
        "After tick 2: mem_read L1@({},{}) = {}",
        ox + 40,
        oy + 20,
        mem_read_val
    );

    // Check L0 ViaUp at (ox+40, oy+21) — should read L1 merge Mux
    let viaup_l0_idx = (oy + 21) * gw + (ox + 40);
    let viaup_val = sim.get_logic_value_by_idx(viaup_l0_idx);
    eprintln!(
        "After tick 2: ViaUp L0@({},{}) = {}",
        ox + 40,
        oy + 21,
        viaup_val
    );

    // Check L0 Mux(wb) at (ox+41, oy+21) — feedback Mux for R0
    let muxwb_l0_idx = (oy + 21) * gw + (ox + 41);
    let muxwb_val = sim.get_logic_value_by_idx(muxwb_l0_idx);
    eprintln!(
        "After tick 2: Mux(wb) L0@({},{}) = {}",
        ox + 41,
        oy + 21,
        muxwb_val
    );

    // Check L0 Register8 at (ox+42, oy+21)
    let reg8_l0_idx = (oy + 21) * gw + (ox + 42);
    let reg8_val = sim.get_logic_value_by_idx(reg8_l0_idx);
    eprintln!(
        "After tick 2: Register8 L0@({},{}) = {}",
        ox + 42,
        oy + 21,
        reg8_val
    );

    // Check AddCarry tile at L0@(ox+38, oy+43)
    let addcarry_idx = (oy + 43) * gw + (ox + 38);
    let addcarry_val = sim.get_logic_value_by_idx(addcarry_idx);
    eprintln!(
        "After tick 2: AddCarry L0@({},{}) = {} (low8={})",
        ox + 38,
        oy + 43,
        addcarry_val,
        addcarry_val & 0xFF
    );

    // Check alu_sel2 BitSelect at L0@(ox+14, oy+15) — should be bit 2 of ctrl_a
    let asel2_bs_idx = (oy + 15) * gw + (ox + 14);
    let asel2_bs_val = sim.get_logic_value_by_idx(asel2_bs_idx);
    eprintln!(
        "After tick 2: alu_sel2 BitSelect L0@({},{}) = {}",
        ox + 14,
        oy + 15,
        asel2_bs_val
    );

    // Check L1 ViaDown for alu_sel2 at L1@(ox+14, oy+15)
    let asel2_vd_idx = 1 * layer_size + (oy + 15) * gw + (ox + 14);
    let asel2_vd_val = sim.get_logic_value_by_idx(asel2_vd_idx);
    eprintln!(
        "After tick 2: alu_sel2 ViaDown L1@({},{}) = {}",
        ox + 14,
        oy + 15,
        asel2_vd_val
    );

    // Check top of alu_sel2 WD chain at L1@(ox+55, oy+16)
    let asel2_wd_top_idx = 1 * layer_size + (oy + 16) * gw + (ox + 55);
    let asel2_wd_top_val = sim.get_logic_value_by_idx(asel2_wd_top_idx);
    eprintln!(
        "After tick 2: alu_sel2 WD top L1@({},{}) = {}",
        ox + 55,
        oy + 16,
        asel2_wd_top_val
    );

    // Check end of alu_sel2 WD chain at L1@(ox+55, oy+52)
    let asel2_wd_bot_idx = 1 * layer_size + (oy + 52) * gw + (ox + 55);
    let asel2_wd_bot_val = sim.get_logic_value_by_idx(asel2_wd_bot_idx);
    eprintln!(
        "After tick 2: alu_sel2 WD bottom L1@({},{}) = {}",
        ox + 55,
        oy + 52,
        asel2_wd_bot_val
    );

    // Check S2 ViaUp at L0@(ox+20, oy+52)
    let s2_via_idx = (oy + 52) * gw + (ox + 20);
    let s2_val = sim.get_logic_value_by_idx(s2_via_idx);
    eprintln!(
        "After tick 2: S2 ViaUp L0@({},{}) = {}",
        ox + 20,
        oy + 52,
        s2_val
    );

    // Check L1@(ox+20, oy+52) — should be WireLeft
    let s2_l1_idx = 1 * layer_size + (oy + 52) * gw + (ox + 20);
    let s2_l1_val = sim.get_logic_value_by_idx(s2_l1_idx);
    eprintln!(
        "After tick 2: L1@({},{}) (S2 source) = {}",
        ox + 20,
        oy + 52,
        s2_l1_val
    );

    // Check what's at intermediate WL position on the alu_sel2 WL chain
    let wl_mid_idx = 1 * layer_size + (oy + 52) * gw + (ox + 40);
    let wl_mid_val = sim.get_logic_value_by_idx(wl_mid_idx);
    eprintln!(
        "After tick 2: alu_sel2 WL mid L1@({},{}) = {}",
        ox + 40,
        oy + 52,
        wl_mid_val
    );

    eprintln!("After tick 2: ALU root value = {}", alu_root_val);

    // Now tick 3: ADD R0,R1
    cpu.tick(&mut sim);

    let alu_root_val3 = sim.get_logic_value_by_idx(alu_root_l0_idx);
    let merge_mux_val3 = sim.get_logic_value_by_idx(merge_mux_idx);
    let mem_read_val3 = sim.get_logic_value_by_idx(mem_read_idx);
    let viaup_val3 = sim.get_logic_value_by_idx(viaup_l0_idx);
    eprintln!("After tick 3: ALU root = {}", alu_root_val3);
    eprintln!("After tick 3: Merge Mux L1 = {}", merge_mux_val3);
    eprintln!("After tick 3: mem_read sel = {}", mem_read_val3);
    eprintln!("After tick 3: ViaUp L0 = {}", viaup_val3);
    eprintln!("After tick 3: R0 = {}", cpu.read_reg(&sim, 0));
    eprintln!("After tick 3: R1 = {}", cpu.read_reg(&sim, 1));
    assert_eq!(cpu.read_reg(&sim, 0), 5, "R0 = 2+3 = 5");
}

/// Parity test: physical CPU full arithmetic chain
#[test]
fn test_physical_parity_arithmetic_chain() {
    // Same program as test_arithmetic_chain_shl_debug
    let program = [
        0x12, // LDI R0, #2
        0x17, // LDI R1, #3
        0x31, // ADD R0, R1   → R0 = 5
        0x80, // SHL R0       → R0 = 10
        0x80, // SHL R0       → R0 = 20
        0x41, // SUB R0, R1   → R0 = 17
        0x51, // AND R0, R1   → R0 = 1
    ];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    for step in 0..7 {
        cpu_h.tick(&mut sim_h);
        cpu_p.tick(&mut sim_p);

        let h_state = cpu_h.dump_state(&sim_h);
        let p_state = cpu_p.dump_state(&sim_p);

        assert_eq!(
            cpu_h.read_pc(&sim_h),
            cpu_p.read_pc(&sim_p),
            "PC mismatch at step {}: hybrid='{}', physical='{}'",
            step,
            h_state,
            p_state
        );

        for reg in 0..4 {
            assert_eq!(
                cpu_h.read_reg(&sim_h, reg),
                cpu_p.read_reg(&sim_p, reg),
                "R{} mismatch at step {}: hybrid='{}', physical='{}'",
                reg,
                step,
                h_state,
                p_state
            );
        }
    }

    assert_eq!(cpu_p.read_reg(&sim_p, 0), 1, "Final R0 = 1");
}

/// Parity test: physical CPU countdown loop with JZ-trampoline branches (full loop)
#[test]
fn test_physical_parity_countdown() {
    // Same program as test_countdown_loop (JZ trampoline for backward branch):
    // addr 0: LDI R0, #3   = 0x13  ; counter = 3
    // addr 1: LDI R1, #1   = 0x15  ; decrement = 1
    // addr 2: SUB R0, R1   = 0x41  ; R0 -= 1, updates Z
    // addr 3: JZ  7        = 0xC7  ; if Z, exit to addr 7
    // addr 4: CMP R1, R1   = 0xA5  ; force Z=1 (unconditional JZ trick)
    // addr 5: JZ  2        = 0xC2  ; Z=1 -> jump to addr 2 (loop back)
    // addr 6: NOP          = 0x00
    // addr 7: NOP (exit)   = 0x00
    let program = [0x13, 0x15, 0x41, 0xC7, 0xA5, 0xC2, 0x00, 0x00];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    for step in 0..12 {
        cpu_h.tick(&mut sim_h);
        cpu_p.tick(&mut sim_p);

        assert_eq!(
            cpu_h.read_reg(&sim_h, 0),
            cpu_p.read_reg(&sim_p, 0),
            "R0 mismatch at tick {}: hybrid={}, physical={}",
            step + 1,
            cpu_h.read_reg(&sim_h, 0),
            cpu_p.read_reg(&sim_p, 0)
        );
        assert_eq!(
            cpu_h.read_flag_z(&sim_h),
            cpu_p.read_flag_z(&sim_p),
            "Z flag mismatch at tick {}",
            step + 1
        );
        assert_eq!(
            cpu_h.read_pc(&sim_h),
            cpu_p.read_pc(&sim_p),
            "PC mismatch at tick {}: hybrid={}, physical={}",
            step + 1,
            cpu_h.read_pc(&sim_h),
            cpu_p.read_pc(&sim_p)
        );
    }

    assert_eq!(
        cpu_p.read_reg(&sim_p, 0),
        0,
        "R0 should be 0 after countdown"
    );
    assert!(
        cpu_p.read_flag_z(&sim_p),
        "Z flag should be set after countdown"
    );
}

/// Fibonacci sequence on hybrid CPU
#[test]
fn test_fibonacci() {
    // Computes Fibonacci numbers using a loop.
    // R0 = current fib, R1 = previous fib, R2 = temp
    //
    // addr 0: LDI R0, #1   = 0x11  ; R0 = 1 (fib(1))
    // addr 1: LDI R1, #0   = 0x14  ; R1 = 0 (fib(0))
    // addr 2: MOV R2, R0   = 0x28  ; save current
    // addr 3: ADD R0, R1   = 0x31  ; new current = old current + previous
    // addr 4: MOV R1, R2   = 0x26  ; previous = old current
    // addr 5: CMP R3, R3   = 0xAF  ; force Z=1 (unconditional JZ trick)
    // addr 6: JZ  2        = 0xC2  ; loop back to addr 2
    // addr 7: NOP          = 0x00
    //
    // Loop body = 5 instructions (addr 2-6). Setup = 2 ticks.
    // After 9 iterations (47 ticks): R0 = fib(10) = 55, R1 = fib(9) = 34
    let mut sim = Simulation::with_size(64, 64);
    let program = [0x11, 0x14, 0x28, 0x31, 0x26, 0xAF, 0xC2, 0x00];
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim);

    for _ in 0..47 {
        cpu.tick(&mut sim);
    }

    assert_eq!(cpu.read_reg(&sim, 0), 55, "R0 = fib(10) = 55");
    assert_eq!(cpu.read_reg(&sim, 1), 34, "R1 = fib(9) = 34");
}

/// Fibonacci parity: hybrid vs physical CPU produce identical results
#[test]
fn test_physical_fibonacci() {
    let program = [0x11, 0x14, 0x28, 0x31, 0x26, 0xAF, 0xC2, 0x00];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    for step in 0..47 {
        cpu_h.tick(&mut sim_h);
        cpu_p.tick(&mut sim_p);

        assert_eq!(
            cpu_h.read_reg(&sim_h, 0),
            cpu_p.read_reg(&sim_p, 0),
            "R0 mismatch at tick {}: hybrid={}, physical={}",
            step + 1,
            cpu_h.read_reg(&sim_h, 0),
            cpu_p.read_reg(&sim_p, 0)
        );
        assert_eq!(
            cpu_h.read_pc(&sim_h),
            cpu_p.read_pc(&sim_p),
            "PC mismatch at tick {}: hybrid={}, physical={}",
            step + 1,
            cpu_h.read_pc(&sim_h),
            cpu_p.read_pc(&sim_p)
        );
    }

    assert_eq!(cpu_p.read_reg(&sim_p, 0), 55, "Physical R0 = fib(10) = 55");
    assert_eq!(cpu_p.read_reg(&sim_p, 1), 34, "Physical R1 = fib(9) = 34");
}

// =============================================================================
// Sprint 82: SubBorrow + Mux16to1 + ROM Expansion Tests
// =============================================================================

/// SubBorrow tile: verifies diff in low 8 bits and borrow in bit 8
#[test]
fn test_subborrow_tile() {
    let mut sim = Simulation::with_size(8, 8);
    let w = sim.width();

    // Guard neighbors to prevent Wire leakage
    for col in 2..=4 {
        sim.set_tile(col, 2, TileType::Const);
        sim.set_logic_value(col, 2, 0);
        sim.set_tile(col, 4, TileType::Const);
        sim.set_logic_value(col, 4, 0);
    }
    sim.set_tile(2, 3, TileType::Const); // left
    sim.set_tile(3, 3, TileType::SubBorrow);
    sim.set_tile(4, 3, TileType::Const); // right

    let sb_idx = 3 * w + 3;

    // 100 - 50 = 50, no borrow
    sim.set_logic_value(2, 3, 100);
    sim.set_logic_value(4, 3, 50);
    sim.dirty.mark_dirty(sb_idx);
    sim.propagate_combinational();
    let result = sim.get_logic_value_by_idx(sb_idx);
    assert_eq!(result & 0xFF, 50, "100 - 50 = 50");
    assert_eq!((result >> 8) & 1, 0, "No borrow for 100 - 50");

    // 1 - 3 = 254 (wrapping), borrow = 1
    sim.set_logic_value(2, 3, 1);
    sim.set_logic_value(4, 3, 3);
    sim.dirty.mark_dirty(sb_idx);
    sim.propagate_combinational();
    let result = sim.get_logic_value_by_idx(sb_idx);
    assert_eq!(result & 0xFF, 254, "1 - 3 wraps to 254");
    assert_eq!((result >> 8) & 1, 1, "Borrow set for 1 - 3");

    // 0 - 0 = 0, no borrow
    sim.set_logic_value(2, 3, 0);
    sim.set_logic_value(4, 3, 0);
    sim.dirty.mark_dirty(sb_idx);
    sim.propagate_combinational();
    assert_eq!(
        sim.get_logic_value_by_idx(sb_idx),
        0,
        "0 - 0 = 0, no borrow"
    );

    // 255 - 255 = 0, no borrow
    sim.set_logic_value(2, 3, 255);
    sim.set_logic_value(4, 3, 255);
    sim.dirty.mark_dirty(sb_idx);
    sim.propagate_combinational();
    let result = sim.get_logic_value_by_idx(sb_idx);
    assert_eq!(result & 0xFF, 0, "255 - 255 = 0");
    assert_eq!((result >> 8) & 1, 0, "No borrow for equal values");
}

/// Mux16to1 tile: verifies all 16 selector values
#[test]
fn test_mux16to1_tile() {
    let mut sim = Simulation::with_size(8, 8);
    let w = sim.width();

    // Guard neighbors
    for col in 2..=4 {
        sim.set_tile(col, 1, TileType::Const);
        sim.set_logic_value(col, 1, 0);
        sim.set_tile(col, 4, TileType::Const);
        sim.set_logic_value(col, 4, 0);
    }
    sim.set_tile(2, 3, TileType::Const); // left = ROM-A (bytes 0-7)
    sim.set_tile(3, 3, TileType::Mux16to1);
    sim.set_tile(4, 3, TileType::Const); // right = selector
    sim.set_tile(3, 2, TileType::Const); // up = ROM-B (bytes 8-15)

    let mux_idx = 3 * w + 3;

    // Pack: bytes 0-7 = values 10,20,30,40,50,60,70,80
    let rom_a: u64 = 10
        | (20 << 8)
        | (30 << 16)
        | (40 << 24)
        | (50 << 32)
        | (60 << 40)
        | (70 << 48)
        | (80 << 56);
    // Pack: bytes 8-15 = values 90,100,110,120,130,140,150,160
    let rom_b: u64 = 90
        | (100 << 8)
        | (110 << 16)
        | (120 << 24)
        | (130 << 32)
        | (140 << 40)
        | (150 << 48)
        | (160 << 56);

    sim.set_logic_value(2, 3, rom_a);
    sim.set_logic_value(3, 2, rom_b);

    let expected = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
    ];
    for sel in 0..16u64 {
        sim.set_logic_value(4, 3, sel);
        sim.dirty.mark_dirty(mux_idx);
        sim.propagate_combinational();
        let result = sim.get_logic_value_by_idx(mux_idx);
        assert_eq!(
            result, expected[sel as usize],
            "Mux16to1 sel={} expected {} got {}",
            sel, expected[sel as usize], result
        );
    }
}

/// Overflow counter: counts 0→255→0 using 16-instruction ROM, exits to addr 8
#[test]
fn test_overflow_counter_16rom() {
    // addr 0: LDI R0, #0   = 0x10  ; counter = 0
    // addr 1: LDI R1, #1   = 0x15  ; increment
    // addr 2: ADD R0, R1   = 0x31  ; counter++ (sets Z on wrap)
    // addr 3: JZ  8        = 0xC8  ; if wrapped to 0, exit to addr 8 (ROM-B!)
    // addr 4: CMP R3, R3   = 0xAF  ; force Z=1
    // addr 5: JZ  2        = 0xC2  ; loop back
    // addr 6-7: NOP
    // addr 8: NOP (halt)   -- in ROM-B range, proves 16-byte fetch
    let program = [
        0x10, 0x15, 0x31, 0xC8, 0xAF, 0xC2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ];

    let mut sim = Simulation::with_size(64, 64);
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(16)
        .build(&mut sim);

    // 2 setup + 255*4 (loop) + 2 (final ADD + JZ taken) = 1024 ticks
    cpu.run(&mut sim, 1030);

    assert_eq!(cpu.read_reg(&sim, 0), 0, "R0 wrapped to 0");
    assert!(cpu.read_flag_z(&sim), "Z flag set after wrap");
}

/// Physical parity: overflow counter on both CPUs
#[test]
fn test_physical_parity_overflow_counter() {
    let program = [
        0x10, 0x15, 0x31, 0xC8, 0xAF, 0xC2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(16)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .with_rom_size(16)
        .build_physical(&mut sim_p);

    cpu_h.run(&mut sim_h, 1030);
    cpu_p.run(&mut sim_p, 1030);

    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        cpu_p.read_reg(&sim_p, 0),
        "R0 parity: hybrid={}, physical={}",
        cpu_h.read_reg(&sim_h, 0),
        cpu_p.read_reg(&sim_p, 0)
    );
    assert_eq!(
        cpu_h.read_flag_z(&sim_h),
        cpu_p.read_flag_z(&sim_p),
        "Z flag parity"
    );
    assert_eq!(cpu_p.read_reg(&sim_p, 0), 0, "Physical R0 = 0");
}

/// RAM read/write: store a value and load it back
#[test]
fn test_ram_store_load() {
    // addr 0: LDI R0, #0   = 0x10  ; address = 0
    // addr 1: LDI R1, #3   = 0x17  ; value = 3
    // addr 2: ST  R1        = 0xF5  ; RAM[R0] = R1
    // addr 3: LDI R1, #0   = 0x14  ; clear R1
    // addr 4: LD  R1        = 0xE4  ; R1 = RAM[R0]
    let program = [0x10, 0x17, 0xF5, 0x14, 0xE4, 0x00, 0x00, 0x00];

    let mut sim = Simulation::with_size(64, 64);
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim);

    cpu.run(&mut sim, 5);

    assert_eq!(cpu.read_reg(&sim, 1), 3, "R1 = 3 (loaded from RAM)");
}

// =============================================================================
// Sprint 83: Physical RAM subsystem tests
// =============================================================================

#[test]
fn test_physical_ram_store_basic() {
    // LDI R0, #2 (address)   = 0x12
    // LDI R1, #3 (value)     = 0x17
    // ST R1                   = 0xF1 (RAM[R0=2] = R1=3)
    let mut sim = Simulation::with_size_layered(128, 128, 2);
    let cpu = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&[0x12, 0x17, 0xF1])
        .build_physical(&mut sim);

    cpu.run(&mut sim, 3);

    assert_eq!(
        cpu.read_ram(&sim, 2),
        3,
        "Physical RAM[2] should be 3 after ST R1"
    );
}

#[test]
fn test_physical_ram_store_load() {
    // LDI R0, #1 (address)   = 0x11
    // LDI R1, #3 (value)     = 0x17
    // ST R1                   = 0xF1 (RAM[1] = 3)
    // LD R2                   = 0xE8 (R2 = RAM[R0=1] = 3)
    let mut sim = Simulation::with_size_layered(128, 128, 2);
    let cpu = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&[0x11, 0x17, 0xF1, 0xE8])
        .build_physical(&mut sim);

    for _ in 0..4 {
        cpu.tick(&mut sim);
    }

    assert_eq!(
        cpu.read_ram(&sim, 1),
        3,
        "Physical RAM[1] should be 3 after ST"
    );
    assert_eq!(
        cpu.read_reg(&sim, 2),
        3,
        "R2 should be 3 after LD from RAM[1]"
    );
}

#[test]
fn test_physical_ram_address_decode() {
    // Write different values to addresses 0-3, verify only target cell changes.
    // Program: for each address a in {0,1,2,3}:
    //   LDI R0, #a; LDI R1, #(a+1); ST R1
    // Addr 0: LDI R0,#0=0x10, LDI R1,#1=0x15, ST R1=0xF1
    // Addr 1: LDI R0,#1=0x11, LDI R1,#2=0x16, ST R1=0xF1
    // Addr 2: LDI R0,#2=0x12, LDI R1,#3=0x17, ST R1=0xF1
    // Addr 3: LDI R0,#3=0x13, LDI R1,#0=0x14, ST R1=0xF1
    // (12 instructions)
    let program = [
        0x10, 0x15, 0xF1, // RAM[0] = 1
        0x11, 0x16, 0xF1, // RAM[1] = 2
        0x12, 0x17, 0xF1, // RAM[2] = 3
        0x13, 0x14, 0xF1, // RAM[3] = 0 (imm wraps: LDI R1,#0)
    ];
    let mut sim = Simulation::with_size_layered(128, 128, 2);
    let cpu = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim);

    cpu.run(&mut sim, 12);

    assert_eq!(cpu.read_ram(&sim, 0), 1, "RAM[0] should be 1");
    assert_eq!(cpu.read_ram(&sim, 1), 2, "RAM[1] should be 2");
    assert_eq!(cpu.read_ram(&sim, 2), 3, "RAM[2] should be 3");
    assert_eq!(cpu.read_ram(&sim, 3), 0, "RAM[3] should be 0");
    // Untouched cells should remain 0
    assert_eq!(cpu.read_ram(&sim, 4), 0, "RAM[4] should be untouched (0)");
    assert_eq!(cpu.read_ram(&sim, 5), 0, "RAM[5] should be untouched (0)");
    assert_eq!(cpu.read_ram(&sim, 6), 0, "RAM[6] should be untouched (0)");
    assert_eq!(cpu.read_ram(&sim, 7), 0, "RAM[7] should be untouched (0)");
}

#[test]
fn test_physical_ram_read_mux_all_addresses() {
    // Verify L1 RAM read Mux tree (Sprint 87 Part B) for addresses 0-3.
    // Each address exercises different bit0 (leaf) and bit1 (mid) paths.
    // Per address: 5-instruction program (LDI R0; LDI R1; ST; LDI R0; LD R2).
    let expected_vals = [1u64, 2, 3, 0]; // LDI R1 imm values for each address
    for addr in 0u8..4 {
        let val = expected_vals[addr as usize];
        let prog = [
            0x10 | addr,              // LDI R0, #addr
            0x14 | ((val as u8) & 3), // LDI R1, #val
            0xF1,                     // ST R1 → RAM[addr] = val
            0x10 | addr,              // LDI R0, #addr (restore after ST may change R0)
            0xE8,                     // LD R2 → R2 = RAM[addr]
        ];
        let mut s = Simulation::with_size_layered(128, 128, 2);
        let c = TileCpuBuilder::new()
            .with_origin(2, 2)
            .with_program(&prog)
            .build_physical(&mut s);
        c.run(&mut s, 5);
        assert_eq!(
            c.read_reg(&s, 2),
            val,
            "LD from addr {}: expected {} got {}",
            addr,
            val,
            c.read_reg(&s, 2)
        );
    }
}

#[test]
fn test_physical_accumulator_preloaded_ram_regression() {
    // Regression from tile_cpu_metrics physical accumulator benchmark:
    // preloaded RAM values [10,20,30,40] should sum to 100 in R2.
    let program = [
        0x10, // LDI R0, #0
        0xE8, // LD  R2 = RAM[0]
        0x11, // LDI R0, #1
        0xEC, // LD  R3 = RAM[1]
        0x3B, // ADD R2, R3
        0x12, // LDI R0, #2
        0xEC, // LD  R3 = RAM[2]
        0x3B, // ADD R2, R3
        0x13, // LDI R0, #3
        0xEC, // LD  R3 = RAM[3]
        0x3B, // ADD R2, R3
        0x22, // MOV R0, R2
    ];

    let mut sim = Simulation::with_size_layered(128, 128, 2);
    let cpu = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim);

    cpu.write_ram(&mut sim, 0, 10);
    cpu.write_ram(&mut sim, 1, 20);
    cpu.write_ram(&mut sim, 2, 30);
    cpu.write_ram(&mut sim, 3, 40);
    cpu.run(&mut sim, 12);

    assert_eq!(
        cpu.read_reg(&sim, 2),
        100,
        "Accumulator regression: expected R2=100 from 10+20+30+40"
    );
}

#[test]
fn test_physical_ram_we_gating() {
    // Non-ST instructions should NOT corrupt RAM.
    // LDI R0, #2 → LDI R1, #3 → NOP → NOP
    // RAM[2] should stay 0 (no ST executed)
    let mut sim = Simulation::with_size_layered(128, 128, 2);
    let cpu = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&[0x12, 0x17, 0x00, 0x00])
        .build_physical(&mut sim);

    cpu.run(&mut sim, 4);

    for addr in 0..8 {
        assert_eq!(
            cpu.read_ram(&sim, addr),
            0,
            "RAM[{}] should be 0 (no ST executed)",
            addr
        );
    }
}

#[test]
fn test_physical_ram_overwrite() {
    // Write to same address twice, verify latest value wins.
    // LDI R0, #1 (addr)      = 0x11
    // LDI R1, #2 (first val) = 0x16
    // ST R1                   = 0xF1 (RAM[1] = 2)
    // LDI R1, #3 (new val)   = 0x17
    // ST R1                   = 0xF1 (RAM[1] = 3, overwrites 2)
    let mut sim = Simulation::with_size_layered(128, 128, 2);
    let cpu = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&[0x11, 0x16, 0xF1, 0x17, 0xF1])
        .build_physical(&mut sim);

    cpu.run(&mut sim, 5);

    assert_eq!(
        cpu.read_ram(&sim, 1),
        3,
        "RAM[1] should be 3 (overwritten from 2)"
    );
}

// =============================================================================
// Sprint 86: Physical Flags + Physical Operand B Data
// =============================================================================

/// Physical flag Z: ADD producing zero and non-zero results
#[test]
fn test_physical_flag_z() {
    // LDI R0, #3 = 0x13; LDI R1, #3 = 0x17; SUB R0, R1 = 0x41 (R0=0, Z=1)
    // then LDI R0, #1 = 0x11; ADD R0, R1 = 0x31 (R0=4, Z=0)
    let program = [0x13, 0x17, 0x41, 0x11, 0x31, 0x00, 0x00, 0x00];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    // After SUB R0,R1: R0=0, Z=1
    cpu_h.run(&mut sim_h, 3);
    cpu_p.run(&mut sim_p, 3);
    assert_eq!(cpu_p.read_reg(&sim_p, 0), 0, "Physical R0 = 0 after SUB");
    assert!(
        cpu_p.read_flag_z(&sim_p),
        "Physical Z=1 after SUB producing 0"
    );
    assert_eq!(
        cpu_h.read_flag_z(&sim_h),
        cpu_p.read_flag_z(&sim_p),
        "Z parity after SUB"
    );

    // After ADD R0,R1: R0=4, Z=0
    cpu_h.run(&mut sim_h, 2);
    cpu_p.run(&mut sim_p, 2);
    assert_eq!(cpu_p.read_reg(&sim_p, 0), 4, "Physical R0 = 4 after ADD");
    assert!(
        !cpu_p.read_flag_z(&sim_p),
        "Physical Z=0 after ADD producing non-zero"
    );
    assert_eq!(
        cpu_h.read_flag_z(&sim_h),
        cpu_p.read_flag_z(&sim_p),
        "Z parity after ADD"
    );
}

/// Physical flag Z masking: ADD producing 0x100 should set Z=1 (masked to 0x00)
#[test]
fn test_physical_flag_z_mask() {
    // LDI R0, #2 = 0x12; LDI R1, #1 = 0x15
    // SHL R0, R1 = 0x81 × 6 → R0 = 4 → 8 → 16 → 32 → 64 → 128
    // ADD R0, R0 = 0x30 → R0 = 128+128 = 256 → 0 (masked), raw bit 8 = 1
    // After this ADD, result & 0xFF = 0 → Z should be 1
    let program = [
        0x12, 0x15, 0x81, 0x81, 0x81, 0x81, 0x81, 0x81,
        // second ROM bank needed for 8+ instructions — use 16-byte ROM
        0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(16)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .with_rom_size(16)
        .build_physical(&mut sim_p);

    cpu_h.run(&mut sim_h, 9);
    cpu_p.run(&mut sim_p, 9);

    assert_eq!(
        cpu_p.read_reg(&sim_p, 0),
        0,
        "Physical R0 = 0 (256 masked to 8 bits)"
    );
    assert!(
        cpu_p.read_flag_z(&sim_p),
        "Physical Z=1 (result masked to 0x00)"
    );
    assert_eq!(
        cpu_h.read_flag_z(&sim_h),
        cpu_p.read_flag_z(&sim_p),
        "Z parity (mask test)"
    );
}

/// Physical flag C: ADD with and without carry
#[test]
fn test_physical_flag_c() {
    // Step 1: ADD without carry (2+3=5, no overflow)
    // LDI R0, #2 = 0x12; LDI R1, #3 = 0x17; ADD R0, R1 = 0x31 (R0=5, C=0)
    // Step 2: SUB with borrow (1-3=254, borrow)
    // LDI R0, #1 = 0x11; SUB R0, R1 = 0x41 (R0=254, C=1 borrow)
    let program = [0x12, 0x17, 0x31, 0x11, 0x41, 0x00, 0x00, 0x00];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    // After ADD R0,R1: R0=5, C=0
    cpu_h.run(&mut sim_h, 3);
    cpu_p.run(&mut sim_p, 3);
    assert_eq!(cpu_p.read_reg(&sim_p, 0), 5, "Physical R0 = 5 after ADD");
    assert!(
        !cpu_p.read_flag_c(&sim_p),
        "Physical C=0 after ADD without overflow"
    );
    assert_eq!(
        cpu_h.read_flag_c(&sim_h),
        cpu_p.read_flag_c(&sim_p),
        "C parity after ADD"
    );

    // After SUB R0,R1: R0=254, C=1 (borrow)
    cpu_h.run(&mut sim_h, 2);
    cpu_p.run(&mut sim_p, 2);
    assert_eq!(
        cpu_p.read_reg(&sim_p, 0),
        254,
        "Physical R0 = 254 after SUB 1-3"
    );
    assert!(
        cpu_p.read_flag_c(&sim_p),
        "Physical C=1 after SUB with borrow"
    );
    assert_eq!(
        cpu_h.read_flag_c(&sim_h),
        cpu_p.read_flag_c(&sim_p),
        "C parity after SUB"
    );
}

/// Physical operand B: register values physically route to Tree B
#[test]
fn test_physical_operand_b() {
    // Load distinct values into all registers, then use them as rs operands
    // LDI R0, #1 = 0x11; LDI R1, #2 = 0x16; LDI R2, #3 = 0x1B; LDI R3, #0 = 0x1C
    // ADD R3, R0 = 0x3C (R3 = 0+1 = 1, tests R0 as rs=0)
    // ADD R3, R1 = 0x3D (R3 = 1+2 = 3, tests R1 as rs=1)
    // ADD R3, R2 = 0x3E (R3 = 3+3 = 6, tests R2 as rs=2)
    // ADD R3, R3 = 0x3F (R3 = 6+6 = 12, tests R3 as rs=3)
    let program = [0x11, 0x16, 0x1B, 0x1C, 0x3C, 0x3D, 0x3E, 0x3F];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    // After 4 LDIs: R0=1, R1=2, R2=3, R3=0
    cpu_h.run(&mut sim_h, 4);
    cpu_p.run(&mut sim_p, 4);

    for step in 0..4 {
        cpu_h.tick(&mut sim_h);
        cpu_p.tick(&mut sim_p);
        assert_eq!(
            cpu_h.read_reg(&sim_h, 3),
            cpu_p.read_reg(&sim_p, 3),
            "R3 mismatch at ADD step {}: hybrid={}, physical={}",
            step,
            cpu_h.read_reg(&sim_h, 3),
            cpu_p.read_reg(&sim_p, 3)
        );
    }

    assert_eq!(cpu_p.read_reg(&sim_p, 3), 12, "Physical R3 = 12 (1+2+3+6)");
}

/// Regression: overflow ADD followed by SHR must produce 0 (Sprint 86.1 audit Finding 1)
///
/// Before the Register8 8-bit mask fix, AddCarry(128,128)=256 leaked into the register
/// as a raw 9-bit value. Subsequent SHR(256,1)=128 in physical vs SHR(0,1)=0 in hybrid.
#[test]
fn test_overflow_add_then_shr_regression() {
    // LDI R0, #2 = 0x12; LDI R1, #1 = 0x15
    // SHL R0, R1 x6 → R0 = 128
    // ADD R0, R0 = 0x30 → 128+128 = 256 → R0 must be 0 (8-bit masked)
    // SHR R0, R1 = 0x91 → 0 >> 1 = 0 (must NOT be 128)
    let program = [
        0x12, 0x15, 0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0x30, 0x91, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(16)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .with_rom_size(16)
        .build_physical(&mut sim_p);

    // Run 9 ticks: 2 LDI + 6 SHL + 1 ADD (overflow)
    cpu_h.run(&mut sim_h, 9);
    cpu_p.run(&mut sim_p, 9);

    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        0,
        "Hybrid R0 = 0 after overflow ADD"
    );
    assert_eq!(
        cpu_p.read_reg(&sim_p, 0),
        0,
        "Physical R0 = 0 after overflow ADD"
    );

    // Tick 10: SHR R0, R1
    cpu_h.tick(&mut sim_h);
    cpu_p.tick(&mut sim_p);

    assert_eq!(cpu_h.read_reg(&sim_h, 0), 0, "Hybrid R0 = 0 after SHR(0,1)");
    assert_eq!(
        cpu_p.read_reg(&sim_p, 0),
        0,
        "Physical R0 = 0 after SHR(0,1) — must not be 128"
    );
    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        cpu_p.read_reg(&sim_p, 0),
        "Overflow-then-SHR parity"
    );
}

/// Regression: carry flag parity after overflow history (Sprint 86.1 audit Finding 2)
///
/// Before the Register8 8-bit mask fix, a previous overflow left >8-bit residue in the
/// register. Subsequent ADD saw input 256 instead of 0, producing a false carry.
#[test]
fn test_overflow_carry_parity_regression() {
    // LDI R0, #2 = 0x12; LDI R1, #1 = 0x15
    // SHL R0, R1 x6 → R0 = 128
    // ADD R0, R0 = 0x30 → 128+128 = 256 → R0 = 0 (overflow, C=1)
    // ADD R0, R1 = 0x31 → 0+1 = 1 → R0 = 1, C must be 0
    let program = [
        0x12, 0x15, 0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0x30, 0x31, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(16)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .with_rom_size(16)
        .build_physical(&mut sim_p);

    // Run 10 ticks: 2 LDI + 6 SHL + 1 ADD(overflow) + 1 ADD(R0+R1)
    cpu_h.run(&mut sim_h, 10);
    cpu_p.run(&mut sim_p, 10);

    assert_eq!(cpu_h.read_reg(&sim_h, 0), 1, "Hybrid R0 = 1 after 0+1");
    assert_eq!(cpu_p.read_reg(&sim_p, 0), 1, "Physical R0 = 1 after 0+1");
    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        cpu_p.read_reg(&sim_p, 0),
        "R0 parity after overflow history"
    );
    assert_eq!(
        cpu_h.read_flag_c(&sim_h),
        cpu_p.read_flag_c(&sim_p),
        "Carry flag parity after overflow history: hybrid C={}, physical C={}",
        cpu_h.read_flag_c(&sim_h),
        cpu_p.read_flag_c(&sim_p)
    );
    assert!(
        !cpu_p.read_flag_c(&sim_p),
        "Physical C=0 after 0+1 (no overflow)"
    );
}

// =============================================================================
// Sprint 88: EXT Prefix + Extended ISA
// =============================================================================

#[test]
fn test_ext_halt() {
    // EXT; HALT; LDI R0,#3 (must not execute)
    let program = [0x00, 0x10, 0x13];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    let metrics_h = cpu_h.run(&mut sim_h, 8);
    let metrics_p = cpu_p.run(&mut sim_p, 8);

    assert!(
        cpu_h.is_halted(&sim_h),
        "Hybrid CPU should halt on EXT HALT"
    );
    assert!(
        cpu_p.is_halted(&sim_p),
        "Physical CPU should halt on EXT HALT"
    );
    assert_eq!(
        metrics_h.cycles, 2,
        "Hybrid HALT should stop after prefix+execute"
    );
    assert_eq!(
        metrics_p.cycles, 2,
        "Physical HALT should stop after prefix+execute"
    );
    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        0,
        "Hybrid R0 unchanged after HALT"
    );
    assert_eq!(
        cpu_p.read_reg(&sim_p, 0),
        0,
        "Physical R0 unchanged after HALT"
    );
}

#[test]
fn test_ext_inc_dec() {
    // LDI R0,#3; EXT INC R0; EXT DEC R0 -> R0 returns to 3
    let program = [0x13, 0x00, 0x20, 0x00, 0x30];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    cpu_h.run(&mut sim_h, 5);
    cpu_p.run(&mut sim_p, 5);

    assert_eq!(cpu_h.read_reg(&sim_h, 0), 3, "Hybrid R0 should return to 3");
    assert_eq!(
        cpu_p.read_reg(&sim_p, 0),
        3,
        "Physical R0 should return to 3"
    );
    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        cpu_p.read_reg(&sim_p, 0),
        "INC/DEC parity"
    );
}

#[test]
fn test_ext_not() {
    // LDI R0,#0; EXT NOT R0 -> 0xFF
    let program = [0x10, 0x00, 0x40];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    cpu_h.run(&mut sim_h, 3);
    cpu_p.run(&mut sim_p, 3);

    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        0xFF,
        "Hybrid NOT should produce 0xFF"
    );
    assert_eq!(
        cpu_p.read_reg(&sim_p, 0),
        0xFF,
        "Physical NOT should produce 0xFF"
    );
    assert_eq!(
        cpu_h.read_flag_z(&sim_h),
        cpu_p.read_flag_z(&sim_p),
        "NOT Z parity"
    );
    assert_eq!(
        cpu_h.read_flag_c(&sim_h),
        cpu_p.read_flag_c(&sim_p),
        "NOT C parity"
    );
}

#[test]
fn test_ext_addi_subi() {
    // LDI R0,#1; EXT ADDI R0,#3; EXT SUBI R0,#2 -> R0=2
    let program = [0x11, 0x00, 0x63, 0x00, 0x72];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    cpu_h.run(&mut sim_h, 5);
    cpu_p.run(&mut sim_p, 5);

    assert_eq!(cpu_h.read_reg(&sim_h, 0), 2, "Hybrid R0 should be 2");
    assert_eq!(cpu_p.read_reg(&sim_p, 0), 2, "Physical R0 should be 2");
    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        cpu_p.read_reg(&sim_p, 0),
        "ADDI/SUBI parity"
    );
}

#[test]
fn test_ext_flags() {
    // LDI R0,#1; EXT DEC R0 -> R0=0, Z=1
    let program = [0x11, 0x00, 0x30];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    cpu_h.run(&mut sim_h, 3);
    cpu_p.run(&mut sim_p, 3);

    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        0,
        "Hybrid R0 should be 0 after DEC"
    );
    assert_eq!(
        cpu_p.read_reg(&sim_p, 0),
        0,
        "Physical R0 should be 0 after DEC"
    );
    assert!(cpu_h.read_flag_z(&sim_h), "Hybrid Z should be set");
    assert!(cpu_p.read_flag_z(&sim_p), "Physical Z should be set");
    assert_eq!(
        cpu_h.read_flag_c(&sim_h),
        cpu_p.read_flag_c(&sim_p),
        "DEC C parity"
    );
}

#[test]
fn test_ext_parity() {
    // Mixed EXT sequence: INC, ADDI, NOT, NEG, SUBI, DEC
    let program = [
        0x11, 0x00, 0x20, 0x00, 0x63, 0x00, 0x40, 0x00, 0x50, 0x00, 0x72, 0x00, 0x30,
    ];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    cpu_h.run(&mut sim_h, 13);
    cpu_p.run(&mut sim_p, 13);

    for reg in 0..4 {
        assert_eq!(
            cpu_h.read_reg(&sim_h, reg),
            cpu_p.read_reg(&sim_p, reg),
            "Register parity mismatch at R{}",
            reg
        );
    }
    assert_eq!(
        cpu_h.read_flag_z(&sim_h),
        cpu_p.read_flag_z(&sim_p),
        "Z parity"
    );
    assert_eq!(
        cpu_h.read_flag_c(&sim_h),
        cpu_p.read_flag_c(&sim_p),
        "C parity"
    );
}

#[test]
fn test_ext_branch_into_prefix() {
    // Jump directly to byte 5 (the second byte of an EXT pair at bytes 4-5).
    // The target byte must execute as a base MOV R0,R0, not as EXT INC R0.
    let program = [0x11, 0xA0, 0xC5, 0x13, 0x00, 0x20];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    cpu_h.run(&mut sim_h, 4);
    cpu_p.run(&mut sim_p, 4);

    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        1,
        "Hybrid target byte should behave as base MOV"
    );
    assert_eq!(
        cpu_p.read_reg(&sim_p, 0),
        1,
        "Physical target byte should behave as base MOV"
    );
}

#[test]
fn test_ext_nop_replacement() {
    // MOV R0,R0 should be a safe NOP replacement (idempotent write, no flag update).
    let program = [0x11, 0x16, 0xA1, 0x20];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .build_physical(&mut sim_p);

    cpu_h.run(&mut sim_h, 4);
    cpu_p.run(&mut sim_p, 4);

    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        1,
        "Hybrid R0 unchanged by MOV R0,R0"
    );
    assert_eq!(
        cpu_p.read_reg(&sim_p, 0),
        1,
        "Physical R0 unchanged by MOV R0,R0"
    );
    assert!(!cpu_h.read_flag_z(&sim_h), "Hybrid Z remains clear");
    assert!(!cpu_p.read_flag_z(&sim_p), "Physical Z remains clear");
    assert!(cpu_h.read_flag_c(&sim_h), "Hybrid C remains set");
    assert!(cpu_p.read_flag_c(&sim_p), "Physical C remains set");
}

// =============================================================================
// Sprint 89: EXT CALL/RET + Link Register
// =============================================================================

#[test]
fn test_ext_call_ret() {
    // LDI R0,#1; EXT CALL 6; EXT HALT; sub: EXT INC R0; EXT RET
    let program = [
        0x11, 0x00, 0x86, 0x00, 0x10, 0x20, 0x00, 0x20, 0x00, 0x90, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20,
    ];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(16)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .with_rom_size(16)
        .build_physical(&mut sim_p);

    cpu_h.run(&mut sim_h, 24);
    cpu_p.run(&mut sim_p, 24);

    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        2,
        "Hybrid CALL/RET should increment once"
    );
    assert_eq!(
        cpu_p.read_reg(&sim_p, 0),
        2,
        "Physical CALL/RET should increment once"
    );
    assert!(cpu_h.is_halted(&sim_h), "Hybrid should halt after return");
    assert!(cpu_p.is_halted(&sim_p), "Physical should halt after return");
}

#[test]
fn test_ext_call_ret_parity() {
    let program = [
        0x11, 0x00, 0x86, 0x00, 0x10, 0x20, 0x00, 0x20, 0x00, 0x90, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20,
    ];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(16)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .with_rom_size(16)
        .build_physical(&mut sim_p);

    cpu_h.run(&mut sim_h, 24);
    cpu_p.run(&mut sim_p, 24);

    for reg in 0..4 {
        assert_eq!(
            cpu_h.read_reg(&sim_h, reg),
            cpu_p.read_reg(&sim_p, reg),
            "Register parity mismatch at R{}",
            reg
        );
    }
    assert_eq!(
        cpu_h.read_flag_z(&sim_h),
        cpu_p.read_flag_z(&sim_p),
        "Z parity"
    );
    assert_eq!(
        cpu_h.read_flag_c(&sim_h),
        cpu_p.read_flag_c(&sim_p),
        "C parity"
    );
    assert_eq!(cpu_h.read_lr(), cpu_p.read_lr(), "LR parity");
}

#[test]
fn test_ext_mflr_mtlr() {
    // Part A: MFLR after CALL captures return address in R2
    let program_mflr = [
        0x00, 0x86, 0x00, 0x10, 0x20, 0x20, 0x00, 0xA8, 0x00, 0x90, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20,
    ];

    let mut sim_h_a = Simulation::with_size(64, 64);
    let cpu_h_a = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program_mflr)
        .with_rom_size(16)
        .build(&mut sim_h_a);

    let mut sim_p_a = Simulation::with_size_layered(128, 128, 2);
    let cpu_p_a = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program_mflr)
        .with_rom_size(16)
        .build_physical(&mut sim_p_a);

    cpu_h_a.run(&mut sim_h_a, 24);
    cpu_p_a.run(&mut sim_p_a, 24);

    assert_eq!(
        cpu_h_a.read_reg(&sim_h_a, 2),
        2,
        "Hybrid MFLR should read LR=2"
    );
    assert_eq!(
        cpu_p_a.read_reg(&sim_p_a, 2),
        2,
        "Physical MFLR should read LR=2"
    );

    // Part B: MTLR writes LR from Rs
    let program_mtlr = [0x17, 0x00, 0xB1, 0x00, 0x10, 0x20, 0x20, 0x20];

    let mut sim_h_b = Simulation::with_size(64, 64);
    let cpu_h_b = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program_mtlr)
        .with_rom_size(16)
        .build(&mut sim_h_b);

    let mut sim_p_b = Simulation::with_size_layered(128, 128, 2);
    let cpu_p_b = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program_mtlr)
        .with_rom_size(16)
        .build_physical(&mut sim_p_b);

    cpu_h_b.run(&mut sim_h_b, 16);
    cpu_p_b.run(&mut sim_p_b, 16);

    assert_eq!(cpu_h_b.read_lr(), 3, "Hybrid MTLR should set LR from R1");
    assert_eq!(cpu_p_b.read_lr(), 3, "Physical MTLR should set LR from R1");
}

#[test]
fn test_ext_nested_call() {
    // Main: LDI R0,#1; CALL sub1; HALT
    // sub1: MFLR R2; CALL inner_ret; MTLR R2; RET
    // inner_ret: EXT RET
    //
    // inner_ret and outer RET share the same [EXT,RET] pair at addresses 12-13.
    // If nested CALL executes correctly, total cycles to HALT is 15 (not 13).
    let program = [
        0x11, 0x00, 0x86, 0x00, 0x10, 0x20, 0x00, 0xA8, 0x00, 0x8C, 0x00, 0xB2, 0x00, 0x90, 0x20,
        0x20,
    ];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(32)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .with_rom_size(32)
        .build_physical(&mut sim_p);

    let metrics_h = cpu_h.run(&mut sim_h, 40);
    let metrics_p = cpu_p.run(&mut sim_p, 40);

    assert!(
        (14..=15).contains(&metrics_h.cycles),
        "Hybrid nested call path length (expected 14-15), got {}",
        metrics_h.cycles
    );
    assert!(
        (14..=15).contains(&metrics_p.cycles),
        "Physical nested call path length (expected 14-15), got {}",
        metrics_p.cycles
    );
    assert!(cpu_h.is_halted(&sim_h), "Hybrid should halt");
    assert!(cpu_p.is_halted(&sim_p), "Physical should halt");
    assert_eq!(
        cpu_h.read_reg(&sim_h, 0),
        1,
        "Hybrid R0 unchanged by ret-only nested call"
    );
    assert_eq!(
        cpu_p.read_reg(&sim_p, 0),
        1,
        "Physical R0 unchanged by ret-only nested call"
    );
    assert_eq!(
        cpu_h.read_reg(&sim_h, 2),
        3,
        "Hybrid save/restore LR register"
    );
    assert_eq!(
        cpu_p.read_reg(&sim_p, 2),
        3,
        "Physical save/restore LR register"
    );
    assert_eq!(
        cpu_h.read_lr(),
        3,
        "Hybrid final LR restored to outer return"
    );
    assert_eq!(
        cpu_p.read_lr(),
        3,
        "Physical final LR restored to outer return"
    );
}

#[test]
fn test_ext_ret_without_call() {
    // EXT RET with default LR=0 should loop safely to address 0.
    let program = [0x00, 0x90, 0x20, 0x20, 0x20, 0x20];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(16)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .with_rom_size(16)
        .build_physical(&mut sim_p);

    cpu_h.run(&mut sim_h, 6);
    cpu_p.run(&mut sim_p, 6);

    assert_eq!(
        cpu_h.read_pc(&sim_h),
        0,
        "Hybrid RET-without-CALL should return to 0"
    );
    assert_eq!(
        cpu_p.read_pc(&sim_p),
        0,
        "Physical RET-without-CALL should return to 0"
    );
    assert!(!cpu_h.is_halted(&sim_h), "Hybrid should not halt");
    assert!(!cpu_p.is_halted(&sim_p), "Physical should not halt");
}

#[test]
fn test_ext_call_preserves_regs() {
    // CALL/RET should preserve GPR values (LR-only control flow state change).
    // Preload registers via instructions to avoid build-time initialization ambiguity.
    let program = [0x11, 0x16, 0x1B, 0x1C, 0x00, 0x88, 0x00, 0x10, 0x00, 0x90];

    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(16)
        .build(&mut sim_h);

    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&program)
        .with_rom_size(16)
        .build_physical(&mut sim_p);

    cpu_h.run(&mut sim_h, 20);
    cpu_p.run(&mut sim_p, 20);

    let expected = [1, 2, 3, 0];
    for reg in 0..4 {
        assert_eq!(
            cpu_h.read_reg(&sim_h, reg),
            expected[reg],
            "Hybrid reg preserved"
        );
        assert_eq!(
            cpu_p.read_reg(&sim_p, reg),
            expected[reg],
            "Physical reg preserved"
        );
    }
}
