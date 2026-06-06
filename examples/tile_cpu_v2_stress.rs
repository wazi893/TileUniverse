use std::error::Error;

use engine::simulation::Simulation;
use engine::tile_cpu::v2_components::{
    v2_add, v2_addi, v2_cmp, v2_halt, v2_jmp, v2_jnz, v2_ld, v2_ldi, v2_mov, v2_nop, v2_st,
    v2_with_reg_ext,
};
use engine::tile_cpu::{V2Builder, V2DiffConfig, run_v2_differential};

struct StressCase {
    name: &'static str,
    program: Vec<u32>,
    max_ticks: u64,
}

fn case_arithmetic_chain() -> StressCase {
    let program = vec![
        v2_ldi(0, 0xFF),
        v2_addi(0, 1),
        v2_addi(0, 1),
        v2_addi(0, 1),
        v2_addi(0, 1),
        v2_addi(0, 1),
        v2_addi(0, 1),
        v2_addi(0, 1),
        v2_addi(0, 1),
        v2_addi(0, 1),
        v2_addi(0, 1),
        v2_with_reg_ext(v2_ldi(0, 0x80), true, false),
        v2_with_reg_ext(v2_addi(0, 0x7F), true, false),
        v2_with_reg_ext(v2_add(0, 0), true, true),
        v2_halt(),
    ];

    StressCase {
        name: "arith_chain",
        program,
        max_ticks: 512,
    }
}

fn case_mixed_bank_mem() -> StressCase {
    let program = vec![
        v2_with_reg_ext(v2_ldi(0, 0x5A), true, false), // R8 = data
        v2_ldi(0, 9),                                  // R0 = addr
        v2_with_reg_ext(v2_st(0, 0), true, false),     // [R0] <- R8 (mixed)
        v2_with_reg_ext(v2_ldi(1, 9), true, false),    // R9 = addr
        v2_with_reg_ext(v2_ld(1, 1), false, true),     // R1 <- [R9] (mixed)
        v2_with_reg_ext(v2_mov(2, 0), false, true),    // R2 <- R8 (mixed)
        v2_with_reg_ext(v2_add(0, 1), true, false),    // R8 += R1 (mixed)
        v2_with_reg_ext(v2_st(0, 1), true, false),     // [R1] <- R8 (mixed)
        v2_halt(),
    ];

    StressCase {
        name: "mixed_bank_mem",
        program,
        max_ticks: 1024,
    }
}

fn case_branch_loop() -> StressCase {
    let program = vec![
        v2_ldi(0, 0),  // R0 = counter
        v2_ldi(1, 1),  // R1 = step
        v2_ldi(2, 40), // R2 = limit
        v2_add(0, 1),  // loop: R0 += 1
        v2_cmp(0, 2),  // compare
        v2_jnz(3),     // loop until R0 == R2
        v2_jmp(8),     // bank-edge jump
        v2_nop(),
        v2_halt(),
    ];

    StressCase {
        name: "branch_loop",
        program,
        max_ticks: 4096,
    }
}

fn hash_state(cpu: &engine::tile_cpu::TileCpuV2, sim: &Simulation) -> u64 {
    fn mix(acc: u64, v: u64) -> u64 {
        let x = acc ^ v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x.rotate_left(27).wrapping_mul(0x94D0_49BB_1331_11EB)
    }

    let mut h = 0xCBF2_9CE4_8422_2325u64;
    h = mix(h, cpu.read_pc(sim) as u64);
    h = mix(h, cpu.read_lr() as u64);
    h = mix(h, cpu.read_flag_z(sim) as u64);
    h = mix(h, cpu.read_flag_c(sim) as u64);
    h = mix(h, cpu.is_halted() as u64);
    for i in 0..16usize {
        h = mix(h, cpu.read_reg(sim, i));
    }
    for i in 0..64usize {
        h = mix(h, cpu.read_ram(sim, i));
    }
    h
}

fn run_case(case: &StressCase) -> Result<(), Box<dyn Error>> {
    let diff = run_v2_differential(
        &case.program,
        V2DiffConfig {
            max_ticks: case.max_ticks,
        },
    )
    .map_err(|e| format!("{} differential mismatch: {}", case.name, e))?;

    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&case.program)
        .with_rom_size(64)
        .with_ram_size(64)
        .build(&mut sim);
    let metrics = cpu.run(&mut sim, case.max_ticks);
    let assists = cpu.read_hybrid_assist_counters();
    let hash = hash_state(&cpu, &sim);

    println!(
        "{:<16} ticks={:<5} retired={:<5} cycles={:<5} deltas={:<8} mixed_x={:<4} mixed_f={:<4} bank_sw={:<4} ram_hi_rd={:<4} rom_upper={:<4} hash=0x{:016X}",
        case.name,
        diff.ticks,
        diff.retired,
        metrics.cycles,
        metrics.total_deltas,
        assists.stage_x_mixed_software,
        assists.stage_f_mixed_dual_capture,
        assists.stage_f_bank_switches,
        assists.ram_high_bank_read_swaps,
        assists.rom_upper_bank_group_select,
        hash
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cases = vec![
        case_arithmetic_chain(),
        case_mixed_bank_mem(),
        case_branch_loop(),
    ];

    for case in &cases {
        run_case(case)?;
    }

    Ok(())
}
