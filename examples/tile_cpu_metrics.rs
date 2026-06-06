//! Tile CPU Architecture Metrics Benchmark (Sprint 84)
//!
//! Measures real tile-based CPU execution with actual tile propagation timing:
//! - Critical path from tile simulation (tick_with_delays)
//! - Maximum frequency derived from measured critical path
//! - TEPS (Tile Evaluations Per Second) from actual sparse evaluation
//! - Sparsity ratio showing what fraction of tiles actually evaluate per tick
//!
//! Run with: cargo run --example tile_cpu_metrics --release

use engine::simulation::Simulation;
use engine::tile_cpu::v2_components::{v2_add, v2_cmp, v2_halt, v2_inc, v2_jnz, v2_ldi, v2_mov};
use engine::tile_cpu::{PhysicalCpu, TileCpu, TileCpuBuilder, TileCpuV2, V2Builder};
use std::time::Instant;

/// Delta time in nanoseconds per delta cycle.
const DELTA_TIME_NS: f64 = 1.0;

fn main() {
    println!();
    println!("================================================================");
    println!("         TILE CPU ARCHITECTURE METRICS BENCHMARK");
    println!("================================================================");
    println!();
    println!("Delta time: {} ns per delta cycle", DELTA_TIME_NS);
    println!("All metrics from REAL tile propagation (tick_with_delays)");
    println!("TEPS = actual tile evaluations / wall time (sparse evaluation)");
    println!();

    // ==========================================================
    //  HYBRID CPU BENCHMARKS (64x64 grid)
    // ==========================================================
    println!("================================================================");
    println!("         HYBRID CPU BENCHMARKS (64x64)");
    println!("================================================================");
    println!();

    // Benchmark 1: Simple ADD (LDI R0,#2; LDI R1,#3; ADD R0,R1)
    run_bench_hybrid(
        "Simple ADD (3 ticks)",
        &[0x12, 0x17, 0x31],
        3,
        |cpu, sim| {
            assert_eq!(cpu.read_reg(sim, 0), 5, "R0 = 2+3 = 5");
        },
    );

    // Benchmark 2: Arithmetic chain
    run_bench_hybrid(
        "Arithmetic Chain (7 ticks)",
        &[0x12, 0x17, 0x31, 0x80, 0x80, 0x41, 0x51],
        7,
        |cpu, sim| {
            assert_eq!(cpu.read_reg(sim, 0), 1, "((5<<1)<<1)-3)&3 = 1");
        },
    );

    // Benchmark 3: Fibonacci (the reference workload)
    // LDI R0,#1; LDI R1,#0; MOV R2,R0; ADD R0,R1; MOV R1,R2; CMP R3,R3; JZ 2; MOV R0,R0
    let fib_program = [0x11, 0x14, 0x28, 0x31, 0x26, 0xAF, 0xC2, 0x20];
    run_bench_hybrid("Fibonacci (47 ticks)", &fib_program, 47, |cpu, sim| {
        assert_eq!(cpu.read_reg(sim, 0), 55, "R0 = fib(10) = 55");
    });

    // Benchmark 4: Extended increment (EXT + INC pairs)
    let ext_inc_program = [
        0x10, 0x00, 0x20, 0x00, 0x20, 0x00, 0x20, 0x00, 0x20, 0x00, 0x20,
    ];
    run_bench_hybrid(
        "Extended Increment (11 ticks)",
        &ext_inc_program,
        11,
        |cpu, sim| {
            assert_eq!(cpu.read_reg(sim, 0), 5, "R0 incremented five times");
        },
    );

    let subroutine_program = [
        0x11, 0x00, 0x86, 0x00, 0x10, 0x20, 0x00, 0x20, 0x00, 0x90, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20,
    ];
    run_bench_hybrid(
        "Subroutine Call/Return (9 ticks)",
        &subroutine_program,
        9,
        |cpu, sim| {
            assert_eq!(cpu.read_reg(sim, 0), 2, "R0 incremented in subroutine");
        },
    );

    // ==========================================================
    //  V2 HYBRID CPU BENCHMARKS (128x128, 16-bit ISA)
    // ==========================================================
    println!("================================================================");
    println!("         V2 HYBRID CPU BENCHMARKS (128x128, 16-bit)");
    println!("================================================================");
    println!();

    let v2_fib_program = [
        v2_ldi(0, 0),  // a
        v2_ldi(1, 1),  // b
        v2_ldi(2, 0),  // i
        v2_ldi(3, 10), // limit
        v2_mov(4, 0),  // t=a
        v2_add(0, 1),  // a=a+b
        v2_mov(1, 4),  // b=t
        v2_inc(2),     // i++
        v2_cmp(2, 3),  // i==limit?
        v2_jnz(4),     // loop
        v2_halt(),
    ];
    run_bench_v2(
        "V2 Fibonacci (11 words)",
        &v2_fib_program,
        256,
        |cpu, sim| {
            assert_eq!(cpu.read_reg(sim, 0), 55, "R0 = fib(10) = 55");
        },
    );

    // ==========================================================
    //  PHYSICAL CPU BENCHMARKS (128x128x2 grid)
    // ==========================================================
    println!("================================================================");
    println!("         PHYSICAL CPU BENCHMARKS (128x128x2)");
    println!("================================================================");
    println!();

    // Physical Fibonacci
    run_bench_physical(
        "Fibonacci (47 ticks)",
        &fib_program,
        47,
        |_cpu, _sim| {},
        |cpu, sim| {
            assert_eq!(cpu.read_reg(sim, 0), 55, "R0 = fib(10) = 55");
        },
    );

    // Physical Overflow Counter (long-running workload)
    let overflow_program: [u8; 16] = [
        0x10, 0x15, 0x31, 0xC8, // LDI R0,#0; LDI R1,#1; ADD R0,R1; JZ 8
        0xAF, 0xC2, 0x20, 0x20, // CMP R3,R3; JZ 2; MOV R0,R0; MOV R0,R0
        0x20, 0x20, 0x20, 0x20, // MOV R0,R0 (halt target in ROM-B)
        0x20, 0x20, 0x20, 0x20,
    ];
    run_bench_physical(
        "Overflow Counter (~1030 ticks)",
        &overflow_program,
        1030,
        |_cpu, _sim| {},
        |cpu, sim| {
            assert_eq!(cpu.read_reg(sim, 0), 0, "R0 wrapped to 0");
        },
    );

    // Physical Accumulator Sum (exercises RAM + LD)
    let accum_program: [u8; 16] = [
        0x10, // addr 0: LDI R0, #0 (address=0)
        0xE8, // addr 1: LD R2 = RAM[0]
        0x11, // addr 2: LDI R0, #1 (address=1)
        0xEC, // addr 3: LD R3 = RAM[1]
        0x3B, // addr 4: ADD R2, R3
        0x12, // addr 5: LDI R0, #2 (address=2)
        0xEC, // addr 6: LD R3 = RAM[2]
        0x3B, // addr 7: ADD R2, R3
        0x13, // addr 8: LDI R0, #3 (address=3)
        0xEC, // addr 9: LD R3 = RAM[3]
        0x3B, // addr10: ADD R2, R3
        0x22, // addr11: MOV R0, R2 (result in R0)
        0x20, 0x20, 0x20, 0x20,
    ];
    run_bench_physical(
        "Accumulator Sum (RAM, 12 ticks)",
        &accum_program,
        12,
        |cpu, sim| {
            // Pre-load RAM before timing starts
            cpu.write_ram(sim, 0, 10);
            cpu.write_ram(sim, 1, 20);
            cpu.write_ram(sim, 2, 30);
            cpu.write_ram(sim, 3, 40);
        },
        |cpu, sim| {
            assert_eq!(cpu.read_reg(sim, 2), 100, "R2 = 10+20+30+40 = 100");
        },
    );

    run_bench_physical(
        "Extended Increment (11 ticks)",
        &ext_inc_program,
        11,
        |_cpu, _sim| {},
        |cpu, sim| {
            assert_eq!(cpu.read_reg(sim, 0), 5, "R0 incremented five times");
        },
    );

    run_bench_physical(
        "Subroutine Call/Return (9 ticks)",
        &subroutine_program,
        9,
        |_cpu, _sim| {},
        |cpu, sim| {
            assert_eq!(cpu.read_reg(sim, 0), 2, "R0 incremented in subroutine");
        },
    );

    // ==========================================================
    //  SCALE COMPARISON: Hybrid vs Physical (Fibonacci)
    // ==========================================================
    println!("================================================================");
    println!("         SCALE COMPARISON: Hybrid vs Physical (Fibonacci)");
    println!("================================================================");
    println!();

    // Hybrid: 64x64 grid
    let mut sim_h = Simulation::with_size(64, 64);
    let cpu_h = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(&fib_program)
        .build(&mut sim_h);
    let start_h = Instant::now();
    let metrics_h = cpu_h.run(&mut sim_h, 47);
    let wall_h = start_h.elapsed();
    assert_eq!(cpu_h.read_reg(&sim_h, 0), 55);

    // Physical: 128x128x2 grid
    let mut sim_p = Simulation::with_size_layered(128, 128, 2);
    let cpu_p = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(&fib_program)
        .build_physical(&mut sim_p);
    let start_p = Instant::now();
    let metrics_p = cpu_p.run(&mut sim_p, 47);
    let wall_p = start_p.elapsed();
    assert_eq!(cpu_p.read_reg(&sim_p, 0), 55);

    let wall_h_ms = wall_h.as_secs_f64() * 1000.0;
    let wall_p_ms = wall_p.as_secs_f64() * 1000.0;

    let teps_h = if wall_h_ms > 0.0 {
        metrics_h.total_tiles_evaluated as f64 / (wall_h_ms / 1000.0)
    } else {
        0.0
    };
    let teps_p = if wall_p_ms > 0.0 {
        metrics_p.total_tiles_evaluated as f64 / (wall_p_ms / 1000.0)
    } else {
        0.0
    };

    let avg_eval_h = metrics_h.total_tiles_evaluated as f64 / metrics_h.cycles as f64;
    let avg_eval_p = metrics_p.total_tiles_evaluated as f64 / metrics_p.cycles as f64;

    let grid_tiles_h = 64 * 64;
    let grid_tiles_p = 128 * 128 * 2;

    println!(
        "| {:<22}| {:>15} | {:>20} |",
        "Metric", "Hybrid (64x64)", "Physical (128x128x2)"
    );
    println!("|{:-<23}|{:-<17}|{:-<22}|", "", "", "");
    println!(
        "| {:<22}| {:>15} | {:>20} |",
        "Grid tiles", grid_tiles_h, grid_tiles_p
    );
    println!(
        "| {:<22}| {:>15} | {:>20} |",
        "Max critical path", metrics_h.max_critical_path, metrics_p.max_critical_path
    );
    println!(
        "| {:<22}| {:>12.3} ms | {:>17.3} ms |",
        "Wall time", wall_h_ms, wall_p_ms
    );
    println!(
        "| {:<22}| {:>15} | {:>20} |",
        "TEPS",
        format_rate(teps_h),
        format_rate(teps_p)
    );
    println!(
        "| {:<22}| {:>15.0} | {:>20.0} |",
        "Avg eval/tick", avg_eval_h, avg_eval_p
    );
    println!(
        "| {:<22}| {:>15} | {:>20} |",
        "Total evaluated", metrics_h.total_tiles_evaluated, metrics_p.total_tiles_evaluated
    );
    println!(
        "| {:<22}| {:>15} | {:>20} |",
        "Total switched", metrics_h.total_tiles_switched, metrics_p.total_tiles_switched
    );
    println!();

    println!("================================================================");
    println!("  All benchmarks complete.");
    println!("================================================================");
    println!();
}

fn run_bench_hybrid(
    name: &str,
    program: &[u8],
    max_cycles: u64,
    verify: impl FnOnce(&TileCpu, &Simulation),
) {
    let mut sim = Simulation::with_size(64, 64);
    let cpu = TileCpuBuilder::new()
        .with_origin(0, 0)
        .with_program(program)
        .build(&mut sim);

    let start = Instant::now();
    let metrics = cpu.run(&mut sim, max_cycles);
    let wall_time = start.elapsed();
    let wall_time_ms = wall_time.as_secs_f64() * 1000.0;

    verify(&cpu, &sim);
    print_metrics(name, cpu.tile_count, 64 * 64, &metrics, wall_time_ms);
}

fn run_bench_physical(
    name: &str,
    program: &[u8],
    max_cycles: u64,
    setup: impl FnOnce(&PhysicalCpu, &mut Simulation),
    verify: impl FnOnce(&PhysicalCpu, &Simulation),
) {
    let mut sim = Simulation::with_size_layered(128, 128, 2);
    let cpu = TileCpuBuilder::new()
        .with_origin(2, 2)
        .with_program(program)
        .build_physical(&mut sim);

    setup(&cpu, &mut sim);

    let start = Instant::now();
    let metrics = cpu.run(&mut sim, max_cycles);
    let wall_time = start.elapsed();
    let wall_time_ms = wall_time.as_secs_f64() * 1000.0;

    verify(&cpu, &sim);
    print_metrics(
        &format!("[Physical] {}", name),
        cpu.tile_count,
        128 * 128 * 2,
        &metrics,
        wall_time_ms,
    );
}

fn run_bench_v2(
    name: &str,
    program: &[u32],
    max_cycles: u64,
    verify: impl FnOnce(&TileCpuV2, &Simulation),
) {
    let mut sim = Simulation::with_size_layered(128, 128, 3);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(program)
        .with_rom_size(16)
        .with_ram_size(8)
        .build(&mut sim);

    let start = Instant::now();
    let metrics = cpu.run(&mut sim, max_cycles);
    let wall_time = start.elapsed();
    let wall_time_ms = wall_time.as_secs_f64() * 1000.0;

    verify(&cpu, &sim);
    print_metrics(
        &format!("[V2] {}", name),
        cpu.tile_count,
        128 * 128 * 3,
        &metrics,
        wall_time_ms,
    );
}

fn print_metrics(
    name: &str,
    tile_count: usize,
    grid_tiles: usize,
    metrics: &engine::tile_cpu::TileCpuMetrics,
    wall_time_ms: f64,
) {
    let max_freq_mhz = if metrics.max_critical_path > 0 {
        1000.0 / (metrics.max_critical_path as f64 * DELTA_TIME_NS)
    } else {
        f64::INFINITY
    };

    let teps = if wall_time_ms > 0.0 {
        metrics.total_tiles_evaluated as f64 / (wall_time_ms / 1000.0)
    } else {
        0.0
    };

    let ticks_per_sec = if wall_time_ms > 0.0 {
        metrics.cycles as f64 / (wall_time_ms / 1000.0)
    } else {
        0.0
    };

    let avg_eval_per_tick = if metrics.cycles > 0 {
        metrics.total_tiles_evaluated as f64 / metrics.cycles as f64
    } else {
        0.0
    };

    let avg_switched_per_tick = if metrics.cycles > 0 {
        metrics.total_tiles_switched as f64 / metrics.cycles as f64
    } else {
        0.0
    };

    // Sparsity: fraction of grid tiles evaluated per tick on average
    let sparsity = avg_eval_per_tick / grid_tiles as f64;

    println!("+---------------------------------------------------------------+");
    println!("| {:<61} |", name);
    println!("+---------------------------------------------------------------+");
    let tile_info = format!(
        "Tile count:          {:>8}  ({} grid tiles)",
        tile_count, grid_tiles
    );
    println!("| {:<61} |", tile_info);
    println!(
        "| Critical path:       {:>8} deltas (max)                    |",
        metrics.max_critical_path
    );
    println!(
        "| Avg critical path:   {:>8.1} deltas                        |",
        metrics.avg_critical_path
    );
    println!(
        "| Simulated freq:      {:>8.1} MHz                           |",
        max_freq_mhz
    );
    println!(
        "| Instructions:        {:>8}                                |",
        metrics.cycles
    );
    println!(
        "| Wall time:           {:>8.3} ms                            |",
        wall_time_ms
    );
    println!(
        "| Ticks/second:        {:>8}                                |",
        format_rate(ticks_per_sec)
    );
    println!(
        "| TEPS:                {:>8}                                |",
        format_rate(teps)
    );
    println!(
        "| Avg eval/tick:       {:>8.0}                                |",
        avg_eval_per_tick
    );
    println!(
        "| Avg switched/tick:   {:>8.0}                                |",
        avg_switched_per_tick
    );
    println!(
        "| Sparsity:            {:>8.4}                                |",
        sparsity
    );
    if metrics.had_timing_violation {
        println!("| WARNING: Timing violations detected!                          |");
    }
    println!("+---------------------------------------------------------------+");
    println!();
}

fn format_rate(rate: f64) -> String {
    if rate >= 1e12 {
        format!("{:.1} T/s", rate / 1e12)
    } else if rate >= 1e9 {
        format!("{:.1} G/s", rate / 1e9)
    } else if rate >= 1e6 {
        format!("{:.1} M/s", rate / 1e6)
    } else if rate >= 1e3 {
        format!("{:.1} K/s", rate / 1e3)
    } else {
        format!("{:.1} /s", rate)
    }
}
