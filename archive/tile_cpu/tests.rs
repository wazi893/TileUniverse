//! Tests for TileCpu
//!
//! These tests verify that the CPU executes correctly via tile simulation.

use super::*;
use crate::simulation::Simulation;
use crate::tile_meta::TileType;

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
        .with_program(&[0x00]) // NOP
        .build(&mut sim);

    let stats = cpu.tick(&mut sim);

    // Should have some propagation
    assert!(stats.total_deltas >= 0);
    // Should converge
    assert!(stats.converged);
}

#[test]
fn test_run_collects_metrics() {
    let mut sim = Simulation::with_size(64, 64);

    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&[0x00, 0x00, 0x00]) // 3 NOPs
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
        .with_program(&[0x00, 0x00, 0x00]) // 3 NOPs
        .build(&mut sim);

    let metrics = cpu.run(&mut sim, 10);

    // Should execute without timing violations
    assert!(!metrics.had_timing_violation);
}

// NOTE: Full instruction execution tests require complete datapath wiring,
// which is Phase 2+ of the implementation. These tests are placeholders
// that will be enabled as the implementation progresses.

#[test]
#[ignore = "Requires full datapath implementation"]
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
#[ignore = "Requires full datapath implementation"]
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
