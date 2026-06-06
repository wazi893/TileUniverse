//! V2 benchmark corpus and execution harness.

use std::error::Error;
use std::time::{Duration, Instant};

use crate::simulation::Simulation;
use crate::tile_cpu::{TileCpuMetrics, V2Builder, V2HybridAssistCounters, V2TraceLog, assemble_v2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V2BenchmarkCase {
    pub name: &'static str,
    pub source: &'static str,
    pub max_cycles: u64,
    pub rom_size: usize,
}

#[derive(Debug, Clone)]
pub struct V2BenchmarkOutcome {
    pub case_name: &'static str,
    pub trace_enabled: bool,
    pub metrics: TileCpuMetrics,
    pub elapsed: Duration,
    pub trace_entries: usize,
    pub trace_hash: Option<u64>,
    pub final_state_hash: u64,
    pub hybrid: V2HybridAssistCounters,
}

impl V2BenchmarkOutcome {
    pub fn deltas_per_sec(&self) -> f64 {
        rate(self.metrics.total_deltas, self.elapsed)
    }

    pub fn evals_per_sec(&self) -> f64 {
        rate(self.metrics.total_tiles_evaluated, self.elapsed)
    }

    pub fn switched_per_sec(&self) -> f64 {
        rate(self.metrics.total_tiles_switched, self.elapsed)
    }
}

const fn case(name: &'static str, source: &'static str, max_cycles: u64) -> V2BenchmarkCase {
    V2BenchmarkCase {
        name,
        source,
        max_cycles,
        rom_size: 64,
    }
}

const fn case_rom128(name: &'static str, source: &'static str, max_cycles: u64) -> V2BenchmarkCase {
    V2BenchmarkCase {
        name,
        source,
        max_cycles,
        rom_size: 128,
    }
}

pub const V2_BENCHMARK_CASES: [V2BenchmarkCase; 10] = [
    case(
        "fibonacci",
        r#"
        LDI R0, 0
        LDI R1, 1
        LDI R2, 0
        LDI R3, 10
    loop:
        MOV R4, R0
        ADD R0, R1
        MOV R1, R4
        INC R2
        CMP R2, R3
        JNZ loop
        HALT
        "#,
        256,
    ),
    case(
        "branch_counter",
        r#"
        LDI R0, 0
        LDI R1, 1
        LDI R2, 24
        LDI R3, 0
    loop:
        ADD R0, R1
        XORI R3, 1
        CMP R3, R1
        JZ skip
        INC R0
    skip:
        DEC R2
        JNZ loop
        HALT
        "#,
        512,
    ),
    case(
        "memory_stream",
        r#"
        LDI R0, 0
        LDI R1, 1
        LDI R2, 16
    loop:
        ST [R0], R1
        LD R3, [R0]
        ADD R1, R3
        INC R0
        DEC R2
        JNZ loop
        STB [40], R1
        LDB R4, [40]
        HALT
        "#,
        512,
    ),
    case(
        "mixed_bank_mem",
        r#"
        LDI R8, 1
        LDI R9, 2
        LDI R0, 4
        LDI R1, 8
    loop:
        ADD R8, R9
        ADD R9, R0
        STB [50], R8
        LDB R10, [50]
        ADD R0, R10
        DEC R1
        JNZ loop
        HALT
        "#,
        512,
    ),
    // Sprint 152: multi-bank ROM benchmark — crosses bank boundary at address 16.
    // Exercises Bank 1 ROM fetch (PC >= 16) with physical westback routes.
    case(
        "cross_bank_loop",
        r#"
        LDI R0, 0
        LDI R1, 1
        LDI R2, 0
        LDI R3, 0
        LDI R4, 6
        NOP
        NOP
        NOP
        NOP
        NOP
        NOP
        NOP
        NOP
        NOP
        NOP
        JMP bank1
    bank1:
        ADD R2, R1
        ADD R3, R2
        INC R0
        CMP R0, R4
        JNZ bank1
        HALT
        "#,
        512,
    ),
    // Sprint 181: CLZ/CTZ/POPCNT benchmark (Sprint 176 features).
    case(
        "bit_ops",
        r#"
        LDI R0, 0
        LDI R1, 128
        CLZ R1
        LDI R2, 16
        CTZ R2
        LDI R3, 255
        POPCNT R3
        ST [R0], R1
        INC R0
        ST [R0], R2
        INC R0
        ST [R0], R3
        INC R0
        LDI R4, 0
        ADD R4, R1
        ADD R4, R2
        ADD R4, R3
        ST [R0], R4
        HALT
        "#,
        128,
    ),
    // Sprint 181: Branchless min/max/clamp via MOVZ/MOVC/MOVNC (Sprint 177 features).
    case(
        "cond_move",
        r#"
        LDI R0, 30
        LDI R1, 50
        CMP R0, R1
        MOVNC R0, R1
        LDI R2, 0
        ST [R2], R0
        LDI R0, 30
        LDI R1, 50
        CMP R0, R1
        MOVC R0, R1
        INC R2
        ST [R2], R0
        LDI R0, 5
        LDI R1, 10
        CMP R0, R1
        MOVC R0, R1
        LDI R1, 40
        CMP R0, R1
        MOVNC R0, R1
        INC R2
        ST [R2], R0
        LDI R0, 7
        LDI R1, 7
        LDI R3, 99
        CMP R0, R1
        MOVZ R4, R3
        INC R2
        ST [R2], R4
        HALT
        "#,
        192,
    ),
    // Sprint 181: Wide immediates + indexed memory (Sprints 174+175 features).
    case(
        "wide_indexed",
        r#"
        LDI.W R5, 1000
        LDI R6, 8
        LDI R0, 10
        ST [R6+0], R0
        LDI R1, 20
        ST [R6+1], R1
        LDI R2, 30
        ST [R6+2], R2
        LDI R3, 40
        ST [R6+3], R3
        LD R0, [R6+0]
        LD R1, [R6+1]
        LD R2, [R6+2]
        LD R3, [R6+3]
        LDI R4, 0
        ADD R4, R0
        ADD R4, R1
        ADD R4, R2
        ADD R4, R3
        ADDI.W R4, 256
        LDI R7, 0
        ST [R7], R4
        HALT
        "#,
        192,
    ),
    // Sprint 199: Upper-bank benchmark — main loop runs at PC >= 64 (bank_group==1).
    // Exercises ROM upper bank hybrid decode path (Sprint 187) and validates that
    // synth gates handle bank_group==1 correctly.
    case_rom128(
        "upper_bank",
        r#"
        JMP upper_start
        .org 64
    upper_start:
        LDI R0, 0
        LDI R1, 1
        LDI R2, 8
        LDI R3, 0
    loop:
        ADD R0, R1
        XOR R3, R0
        ST [R0], R3
        INC R0
        DEC R2
        JNZ loop
        HALT
        "#,
        512,
    ),
    // Sprint 202: cross-bank compute — CALL/RET across bank boundary, diverse ALU + memory
    // + bit manipulation at PC >= 64. Exercises ctrl_b bank_group==1 decode path under load.
    case_rom128(
        "cross_bank_compute",
        r#"
        ; Bank 0: setup + two cross-bank CALL/RET pairs
        LDI R0, 0          ; accumulator
        LDI R1, 5          ; loop count for sub_a
        LDI R2, 3          ; loop count for sub_b
        LDI R3, 0          ; scratch
        CALL sub_a          ; -> bank 1 subroutine A
        ; After sub_a: R0 = accumulated ALU result
        MOV R8, R0          ; save result A
        LDI R0, 0          ; reset accumulator
        CALL sub_b          ; -> bank 1 subroutine B
        ; After sub_b: R0 = accumulated memory result
        ADD R0, R8          ; combine both results
        LDI R9, 0
        ST [R9], R0         ; store final at addr 0
        HALT

        .org 64
    sub_a:
        ; Subroutine A (bank 1): ALU-intensive loop
        ; Computes: R0 += (R1 * iter) XOR iter, with shifts and conditional moves
        LDI R4, 0          ; iter counter
    sa_loop:
        MOV R5, R1          ; R5 = R1 (param)
        MUL R5, R4          ; R5 = R1 * iter
        XOR R5, R4          ; R5 ^= iter
        ADD R0, R5          ; accumulate
        SHL R5, 1           ; R5 <<= 1
        ADD R0, R5          ; accumulate shifted
        INC R4
        CMP R4, R1          ; iter < loop_count?
        JNZ sa_loop
        ; Bit manipulation on accumulated result
        MOV R6, R0
        CLZ R6              ; count leading zeros
        MOV R7, R0
        POPCNT R7           ; population count
        ADD R0, R6
        ADD R0, R7
        ; Conditional moves based on flags
        CMP R0, R1
        MOVNC R3, R0       ; if R0 >= R1: R3 = R0
        ADD R0, R3
        RET

    sub_b:
        ; Subroutine B (bank 1): memory-intensive loop
        ; Stores pattern to RAM, reads back, accumulates
        LDI R4, 0          ; addr counter
        LDI R5, 10         ; base address for indexed stores
    sb_loop:
        MOV R6, R4
        ADD R6, R5          ; addr = base + iter
        MOV R7, R4
        SHL R7, 2           ; value = iter << 2
        XORI R7, 7          ; value ^= 7
        ST [R6], R7         ; store
        LD R8, [R6]         ; load back
        ADD R0, R8          ; accumulate
        INC R4
        CMP R4, R2          ; iter < loop_count?
        JNZ sb_loop
        ; Read back all stored values and XOR-fold
        LDI R4, 0
    sb_verify:
        MOV R6, R4
        ADD R6, R5
        LD R7, [R6]
        XOR R0, R7
        INC R4
        CMP R4, R2
        JNZ sb_verify
        RET
        "#,
        512,
    ),
];

// Filled during Sprint 140 Phase 3 from the benchmark suite on this codebase state.
pub const V2_BENCHMARK_GOLDENS: [(&str, u64); 10] = [
    ("fibonacci", 0xA38A_5D5B_22A2_A5A1),
    ("branch_counter", 0x2CAB_FACF_2A98_CE7B),
    ("memory_stream", 0xDB5A_D614_316D_5C6D),
    ("mixed_bank_mem", 0x378A_02D2_FD04_895E),
    ("cross_bank_loop", 0x5D0F_F075_9CF7_0B42), // Sprint 152: multi-bank ROM benchmark
    ("bit_ops", 0x3D3C_B7B7_8328_6268),         // Sprint 181: CLZ/CTZ/POPCNT
    ("cond_move", 0xFD49_A3F7_418C_1E2A),       // Sprint 181: branchless min/max/clamp
    ("wide_indexed", 0x25F6_8EB3_1C2D_3998),    // Sprint 181: wide imm + indexed mem
    ("upper_bank", 0x1F39_62F8_82CD_8F0F),      // Sprint 199: upper-bank ROM
    ("cross_bank_compute", 0x227A_8634_C7AC_4006), // Sprint 202: cross-bank CALL/RET
];

// Quick-win gate (Sprint 141): performance regression guardrails on deterministic cycle counts.
pub const V2_BENCHMARK_CYCLE_GOLDENS: [(&str, u64); 10] = [
    ("fibonacci", 66),
    ("branch_counter", 162),
    ("memory_stream", 103),
    ("mixed_bank_mem", 62),
    ("cross_bank_loop", 48),     // Sprint 152: multi-bank ROM benchmark
    ("bit_ops", 20),             // Sprint 181: CLZ/CTZ/POPCNT
    ("cond_move", 30),           // Sprint 181: branchless min/max/clamp
    ("wide_indexed", 24),        // Sprint 181: wide imm + indexed mem
    ("upper_bank", 55),          // Sprint 199: upper-bank ROM
    ("cross_bank_compute", 127), // Sprint 202: cross-bank CALL/RET
];

// Sprint 153: Exact assist counter values for regression detection.
// [bank_sw, dual_cap, mixed_x, ram_swap, rom_upper]
// Sprint 186: zero_shift_carry_overrides removed (was always 0, now fixed in software).
// Sprint 187: rom_upper_bank_group_select added (5th element).
pub const V2_BENCHMARK_ASSIST_GOLDENS: [(&str, [u64; 5]); 10] = [
    ("fibonacci", [0, 0, 0, 0, 0]),
    ("branch_counter", [0, 0, 0, 0, 0]),
    ("memory_stream", [0, 0, 0, 0, 0]),
    ("mixed_bank_mem", [0, 0, 0, 0, 0]),
    ("cross_bank_loop", [0, 0, 0, 0, 0]),
    ("bit_ops", [0, 0, 0, 0, 0]),            // Sprint 181
    ("cond_move", [0, 0, 0, 0, 0]),          // Sprint 181
    ("wide_indexed", [0, 0, 0, 0, 0]),       // Sprint 181
    ("upper_bank", [0, 0, 0, 0, 0]), // Sprint 211: staged Super Mux/direct-set — zero assists
    ("cross_bank_compute", [0, 0, 0, 0, 0]), // Sprint 211: staged Super Mux/direct-set — zero assists
];

pub fn benchmark_cases() -> &'static [V2BenchmarkCase] {
    &V2_BENCHMARK_CASES
}

pub fn expected_hash_for_case(name: &str) -> Option<u64> {
    V2_BENCHMARK_GOLDENS
        .iter()
        .find_map(|(case, hash)| (*case == name).then_some(*hash))
}

pub fn expected_cycles_for_case(name: &str) -> Option<u64> {
    V2_BENCHMARK_CYCLE_GOLDENS
        .iter()
        .find_map(|(case, cycles)| (*case == name).then_some(*cycles))
}

pub fn expected_assists_for_case(name: &str) -> Option<[u64; 5]> {
    V2_BENCHMARK_ASSIST_GOLDENS
        .iter()
        .find_map(|(case, assists)| (*case == name).then_some(*assists))
}

pub fn run_v2_benchmark_case(
    case: &V2BenchmarkCase,
    trace_enabled: bool,
) -> Result<V2BenchmarkOutcome, Box<dyn Error>> {
    let program = assemble_v2(case.source)?;
    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(case.rom_size)
        .with_ram_size(128)
        .build(&mut sim);

    let mut metrics = TileCpuMetrics::default();
    let mut total_critical = 0u64;
    let mut trace_log = V2TraceLog::new();

    let start = Instant::now();
    for _ in 0..case.max_cycles {
        if cpu.is_halted() {
            break;
        }

        let stats = if trace_enabled {
            cpu.step_with_trace(&mut sim, &mut trace_log)
        } else {
            cpu.step(&mut sim)
        };

        metrics.cycles += 1;
        if cpu.last_stage_x_valid() {
            metrics.instructions_executed += 1;
        }
        metrics.total_deltas += stats.total_deltas as u64;
        metrics.total_tiles_evaluated += stats.tiles_evaluated as u64;
        metrics.total_tiles_switched += stats.tiles_switched as u64;
        metrics.max_critical_path = metrics.max_critical_path.max(stats.critical_path_deltas);
        total_critical += stats.critical_path_deltas as u64;
        if !stats.converged {
            metrics.had_timing_violation = true;
        }
    }
    let elapsed = start.elapsed();

    if !cpu.is_halted() {
        return Err(format!(
            "benchmark '{}' did not halt within {} cycles",
            case.name, case.max_cycles
        )
        .into());
    }

    if metrics.cycles > 0 {
        metrics.avg_critical_path = total_critical as f64 / metrics.cycles as f64;
        metrics.ipc = metrics.instructions_executed as f64 / metrics.cycles as f64;
        if metrics.max_critical_path > 0 {
            metrics.estimated_max_mhz = 1000.0 / metrics.max_critical_path as f64;
        }
    }

    let trace_hash = if trace_enabled {
        Some(hash_bytes(trace_log.to_lines().as_bytes()))
    } else {
        None
    };

    Ok(V2BenchmarkOutcome {
        case_name: case.name,
        trace_enabled,
        metrics,
        elapsed,
        trace_entries: trace_log.len(),
        trace_hash,
        final_state_hash: hash_v2_final_state(&cpu, &sim),
        hybrid: cpu.read_hybrid_assist_counters(),
    })
}

pub fn validate_benchmark_outcome(outcome: &V2BenchmarkOutcome) -> Result<(), String> {
    let expected = expected_hash_for_case(outcome.case_name)
        .ok_or_else(|| format!("missing benchmark golden for '{}'", outcome.case_name))?;
    if expected != outcome.final_state_hash {
        return Err(format!(
            "golden mismatch for '{}': expected 0x{:016X}, got 0x{:016X}",
            outcome.case_name, expected, outcome.final_state_hash
        ));
    }
    Ok(())
}

pub fn validate_benchmark_performance(outcome: &V2BenchmarkOutcome) -> Result<(), String> {
    let expected_cycles = expected_cycles_for_case(outcome.case_name)
        .ok_or_else(|| format!("missing cycle golden for '{}'", outcome.case_name))?;
    if outcome.metrics.cycles != expected_cycles {
        return Err(format!(
            "cycle regression for '{}': expected {}, got {}",
            outcome.case_name, expected_cycles, outcome.metrics.cycles
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 1 (Sprint 146): Per-cycle detailed metrics for baseline measurement
// ---------------------------------------------------------------------------

/// Per-cycle telemetry captured during a benchmark run.
#[derive(Debug, Clone)]
pub struct V2CycleMetrics {
    pub wall_us: f64,
    pub deltas: u32,
    pub tiles_evaluated: u32,
    pub tiles_switched: u32,
    // Sprint 154: per-stage wall-clock timing (microseconds)
    pub stage_f_us: f64,
    pub stage_x_us: f64,
    pub branch_us: f64,
    pub commit_us: f64,
    pub clock_us: f64,
    pub ram_us: f64,
}

/// Sprint 154: averaged per-stage timing for profiling output.
#[derive(Debug, Clone, Default)]
pub struct V2AvgStageTiming {
    pub stage_f_us: f64,
    pub stage_x_us: f64,
    pub branch_us: f64,
    pub commit_us: f64,
    pub clock_us: f64,
    pub ram_us: f64,
}

/// Detailed benchmark outcome including per-cycle granularity.
#[derive(Debug, Clone)]
pub struct V2BenchmarkDetailedOutcome {
    pub outcome: V2BenchmarkOutcome,
    pub per_cycle: Vec<V2CycleMetrics>,
}

impl V2BenchmarkDetailedOutcome {
    pub fn total_wall_ms(&self) -> f64 {
        self.outcome.elapsed.as_secs_f64() * 1000.0
    }

    pub fn avg_us_per_cycle(&self) -> f64 {
        if self.per_cycle.is_empty() {
            return 0.0;
        }
        let total: f64 = self.per_cycle.iter().map(|c| c.wall_us).sum();
        total / self.per_cycle.len() as f64
    }

    pub fn avg_deltas_per_cycle(&self) -> f64 {
        if self.per_cycle.is_empty() {
            return 0.0;
        }
        let total: u64 = self.per_cycle.iter().map(|c| c.deltas as u64).sum();
        total as f64 / self.per_cycle.len() as f64
    }

    pub fn avg_evals_per_cycle(&self) -> f64 {
        if self.per_cycle.is_empty() {
            return 0.0;
        }
        let total: u64 = self
            .per_cycle
            .iter()
            .map(|c| c.tiles_evaluated as u64)
            .sum();
        total as f64 / self.per_cycle.len() as f64
    }

    /// Sprint 154: average per-stage timing in microseconds.
    pub fn avg_stage_us(&self) -> V2AvgStageTiming {
        let n = self.per_cycle.len() as f64;
        if n == 0.0 {
            return V2AvgStageTiming::default();
        }
        V2AvgStageTiming {
            stage_f_us: self.per_cycle.iter().map(|c| c.stage_f_us).sum::<f64>() / n,
            stage_x_us: self.per_cycle.iter().map(|c| c.stage_x_us).sum::<f64>() / n,
            branch_us: self.per_cycle.iter().map(|c| c.branch_us).sum::<f64>() / n,
            commit_us: self.per_cycle.iter().map(|c| c.commit_us).sum::<f64>() / n,
            clock_us: self.per_cycle.iter().map(|c| c.clock_us).sum::<f64>() / n,
            ram_us: self.per_cycle.iter().map(|c| c.ram_us).sum::<f64>() / n,
        }
    }

    pub fn avg_activity_ratio(&self) -> f64 {
        let total_tiles = 128 * 128 * 4; // V2 grid size
        if self.per_cycle.is_empty() {
            return 0.0;
        }
        let total_evals: u64 = self
            .per_cycle
            .iter()
            .map(|c| c.tiles_evaluated as u64)
            .sum();
        let total_deltas: u64 = self.per_cycle.iter().map(|c| c.deltas as u64).sum();
        if total_deltas == 0 {
            return 0.0;
        }
        let evals_per_delta = total_evals as f64 / total_deltas as f64;
        evals_per_delta / total_tiles as f64
    }
}

pub fn run_v2_benchmark_detailed(
    case: &V2BenchmarkCase,
) -> Result<V2BenchmarkDetailedOutcome, Box<dyn Error>> {
    let program = assemble_v2(case.source)?;
    let mut sim = Simulation::with_size_layered(128, 128, 4);
    let mut cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(case.rom_size)
        .with_ram_size(128)
        .build(&mut sim);
    // Sprint 167: enable per-stage timing for benchmark profiling.
    cpu.set_stage_timing(true);

    let mut metrics = TileCpuMetrics::default();
    let mut total_critical = 0u64;
    let mut per_cycle = Vec::new();

    let start = Instant::now();
    for _ in 0..case.max_cycles {
        if cpu.is_halted() {
            break;
        }

        let cycle_start = Instant::now();
        let stats = cpu.step(&mut sim);
        let cycle_elapsed = cycle_start.elapsed();

        let timing = cpu.read_last_stage_timing();
        per_cycle.push(V2CycleMetrics {
            wall_us: cycle_elapsed.as_secs_f64() * 1_000_000.0,
            deltas: stats.total_deltas,
            tiles_evaluated: stats.tiles_evaluated,
            tiles_switched: stats.tiles_switched,
            stage_f_us: timing.stage_f_ns as f64 / 1_000.0,
            stage_x_us: timing.stage_x_ns as f64 / 1_000.0,
            branch_us: timing.branch_ns as f64 / 1_000.0,
            commit_us: timing.commit_ns as f64 / 1_000.0,
            clock_us: timing.clock_ns as f64 / 1_000.0,
            ram_us: timing.ram_ns as f64 / 1_000.0,
        });

        metrics.cycles += 1;
        if cpu.last_stage_x_valid() {
            metrics.instructions_executed += 1;
        }
        metrics.total_deltas += stats.total_deltas as u64;
        metrics.total_tiles_evaluated += stats.tiles_evaluated as u64;
        metrics.total_tiles_switched += stats.tiles_switched as u64;
        metrics.max_critical_path = metrics.max_critical_path.max(stats.critical_path_deltas);
        total_critical += stats.critical_path_deltas as u64;
        if !stats.converged {
            metrics.had_timing_violation = true;
        }
    }
    let elapsed = start.elapsed();

    if !cpu.is_halted() {
        return Err(format!(
            "benchmark '{}' did not halt within {} cycles",
            case.name, case.max_cycles
        )
        .into());
    }

    if metrics.cycles > 0 {
        metrics.avg_critical_path = total_critical as f64 / metrics.cycles as f64;
        metrics.ipc = metrics.instructions_executed as f64 / metrics.cycles as f64;
        if metrics.max_critical_path > 0 {
            metrics.estimated_max_mhz = 1000.0 / metrics.max_critical_path as f64;
        }
    }

    Ok(V2BenchmarkDetailedOutcome {
        outcome: V2BenchmarkOutcome {
            case_name: case.name,
            trace_enabled: false,
            metrics,
            elapsed,
            trace_entries: 0,
            trace_hash: None,
            final_state_hash: hash_v2_final_state(&cpu, &sim),
            hybrid: cpu.read_hybrid_assist_counters(),
        },
        per_cycle,
    })
}

pub fn hash_v2_final_state(cpu: &crate::tile_cpu::TileCpuV2, sim: &Simulation) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    h = mix_hash(h, cpu.read_pc(sim) as u64);
    h = mix_hash(h, cpu.read_lr() as u64);
    h = mix_hash(h, cpu.read_flag_z(sim) as u64);
    h = mix_hash(h, cpu.read_flag_c(sim) as u64);
    h = mix_hash(h, cpu.is_halted() as u64);
    for reg in 0..16usize {
        h = mix_hash(h, cpu.read_reg(sim, reg));
    }
    for addr in 0..128usize {
        h = mix_hash(h, cpu.read_ram(sim, addr));
    }
    h
}

fn rate(total: u64, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs == 0.0 {
        total as f64
    } else {
        total as f64 / secs
    }
}

fn mix_hash(acc: u64, v: u64) -> u64 {
    let x = acc ^ v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x.rotate_left(27).wrapping_mul(0x94D0_49BB_1331_11EB)
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01B3);
    }
    h
}

// ---------------------------------------------------------------------------
// Sprint 294: V2FastCpu benchmark harness
// ---------------------------------------------------------------------------

/// Run a benchmark case using V2FastCpu (ISS-backed, no tile simulation).
///
/// Returns a lightweight outcome with cycle count, retired count, final-state
/// hash, and wall-clock time.  No Simulation or tile grid is allocated.
pub fn run_v2_benchmark_fast(
    case: &V2BenchmarkCase,
) -> Result<V2FastBenchmarkOutcome, Box<dyn Error>> {
    let program = crate::tile_cpu::assemble_v2(case.source)?;
    let mut cpu = crate::tile_cpu::V2FastCpu::from_program(&program);

    let start = Instant::now();
    for _ in 0..case.max_cycles {
        if cpu.is_halted() {
            break;
        }
        cpu.tick();
    }
    let elapsed = start.elapsed();

    Ok(V2FastBenchmarkOutcome {
        case_name: case.name,
        cycles: cpu.cycle_count(),
        retired: cpu.retired_count(),
        final_state_hash: crate::tile_cpu::hash_v2_iss_state(cpu.state()),
        elapsed,
    })
}

/// Lightweight benchmark outcome for V2FastCpu (no tile metrics).
#[derive(Debug, Clone)]
pub struct V2FastBenchmarkOutcome {
    pub case_name: &'static str,
    pub cycles: u64,
    pub retired: u64,
    pub final_state_hash: u64,
    pub elapsed: Duration,
}

impl V2FastBenchmarkOutcome {
    pub fn cycles_per_sec(&self) -> f64 {
        rate(self.cycles, self.elapsed)
    }
}

// ---------------------------------------------------------------------------
// Sprint 298: Long-running benchmark cases (ISS-only golden tests)
// ---------------------------------------------------------------------------
//
// Separate from V2_BENCHMARK_CASES because the existing golden test suites
// iterate all cases through the physical tile path.  A 30K-cycle benchmark
// at ~10K cyc/sec = 3 seconds per run — unacceptable for the main suite.
// These get their own ISS-only golden tests.

pub const V2_LONG_BENCHMARK_CASES: [V2BenchmarkCase; 3] = [
    case(
        "fib_iterative_1k",
        r#"
        LDI R0, 0
        LDI R1, 1
        LDI R2, 1000
    loop:
        MOV R3, R0
        ADD R0, R1
        MOV R1, R3
        DEC R2
        JNZ loop
        HALT
        "#,
        8000,
    ),
    case(
        "bubble_sort_16",
        r#"
        LDI R0, 16
        LDI R1, 0
    init:
        ST [R1], R0
        DEC R0
        INC R1
        LDI R5, 16
        CMP R1, R5
        JNZ init
        LDI R6, 15
    outer:
        LDI R1, 0
    inner:
        LD R2, [R1+0]
        LD R3, [R1+1]
        CMP R2, R3
        JC no_swap
        ST [R1+0], R3
        ST [R1+1], R2
    no_swap:
        INC R1
        LDI R5, 15
        CMP R1, R5
        JNZ inner
        DEC R6
        JNZ outer
        HALT
        "#,
        5000,
    ),
    case(
        "accumulator_10k",
        r#"
        LDI R0, 0
        LDI R1, 1
        LDI R2, 10000
    loop:
        ADD R0, R1
        DEC R2
        JNZ loop
        HALT
        "#,
        40000,
    ),
];

pub const V2_LONG_BENCHMARK_GOLDENS: [(&str, u64); 3] = [
    ("fib_iterative_1k", 0x6127_200E_20D8_73CD),
    ("bubble_sort_16", 0xC784_277B_6C5F_C63C),
    ("accumulator_10k", 0x8747_1FF9_5A5F_B3C2),
];

pub const V2_LONG_BENCHMARK_CYCLE_GOLDENS: [(&str, u64); 3] = [
    ("fib_iterative_1k", 5005),
    ("bubble_sort_16", 2186),
    ("accumulator_10k", 30005),
];

pub fn long_benchmark_cases() -> &'static [V2BenchmarkCase] {
    &V2_LONG_BENCHMARK_CASES
}

pub fn expected_long_hash_for_case(name: &str) -> Option<u64> {
    V2_LONG_BENCHMARK_GOLDENS
        .iter()
        .find_map(|(case, hash)| (*case == name).then_some(*hash))
}

pub fn expected_long_cycles_for_case(name: &str) -> Option<u64> {
    V2_LONG_BENCHMARK_CYCLE_GOLDENS
        .iter()
        .find_map(|(case, cycles)| (*case == name).then_some(*cycles))
}
