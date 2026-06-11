//! Sprint 377 (Perf Track P1): a measurement harness for **compiled** programs.
//!
//! Mirrors `v2_benchmarks.rs` (which benchmarks hand-written assembly) but compiles a
//! C-like source program with [`compile_source`] and runs it on the physical tile CPU
//! in the configuration the compiler targets (`extended_pc` + `max_authority` + MMIO).
//! It captures the metrics that isolate the three perf levers:
//!
//! 1. **Compiler codegen** — `program_words`, `instructions_executed`.
//! 2. **ISA / microarchitecture** — also `instructions_executed` (e.g. a real `DIV`
//!    would collapse `print_int`'s count).
//! 3. **Simulation engine** — `cycles` (tile-CPU ticks), `ticks/instruction` (= 1/IPC),
//!    and `total_tiles_evaluated` per program.
//!
//! The corpus + goldens let every later perf change be judged against a fixed workload
//! with correctness and golden parity intact.

use std::error::Error;
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::simulation::Simulation;
use crate::tile_cpu::v2_mmio::V2MmioHandle;
use crate::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
use crate::tile_cpu::v2_parser::compile_source;
use crate::tile_cpu::{TileCpuMetrics, V2Builder, V2SynthConfig};

/// A compiled-program benchmark: C-like source + expected results.
#[derive(Debug, Clone, Copy)]
pub struct V2CompiledBenchCase {
    pub name: &'static str,
    pub source: &'static str,
    pub max_cycles: u64,
    /// Expected console output (empty if the program prints nothing).
    pub expected_console: &'static str,
    /// Expected value left in R0 at halt.
    pub expected_r0: u64,
}

/// The measured outcome of running a compiled benchmark.
#[derive(Debug, Clone)]
pub struct V2CompiledBenchOutcome {
    pub case_name: &'static str,
    pub program_words: usize,
    pub metrics: TileCpuMetrics,
    pub console: String,
    pub r0: u64,
    pub elapsed: Duration,
    /// Sprint 381 (Perf L3 profiling): per-phase tile-eval counts (cumulative over the
    /// run) — `(calls, evals)` for cone, settle, constswap, branch, commit. The sim
    /// cost (lever 3) lives here; this attributes it to phases for COMPILED programs.
    pub per_path: [(u64, u64); 5],
    pub settle_scan: u64,
}

/// Names of the five propagation phases in [`V2CompiledBenchOutcome::per_path`].
pub const PER_PATH_NAMES: [&str; 5] = ["cone", "settle", "constswap", "branch", "commit"];

impl V2CompiledBenchOutcome {
    /// Tile-CPU ticks per retired instruction (1/IPC). The compact evaluator retires
    /// one instruction per tick, so this is ~1.0; the meaningful simulation-engine
    /// metric is [`Self::tiles_per_instruction`].
    pub fn ticks_per_instruction(&self) -> f64 {
        if self.metrics.instructions_executed == 0 {
            0.0
        } else {
            self.metrics.cycles as f64 / self.metrics.instructions_executed as f64
        }
    }

    /// Tiles evaluated per retired instruction — the simulation-engine cost (lever 3):
    /// how much of the tile fabric is re-evaluated to retire one instruction.
    pub fn tiles_per_instruction(&self) -> f64 {
        if self.metrics.instructions_executed == 0 {
            0.0
        } else {
            self.metrics.total_tiles_evaluated as f64 / self.metrics.instructions_executed as f64
        }
    }
}

/// Compile `case.source` and run it on the physical tile CPU, collecting metrics.
pub fn run_v2_compiled_benchmark(
    case: &V2CompiledBenchCase,
) -> Result<V2CompiledBenchOutcome, Box<dyn Error>> {
    let program = compile_source(case.source).map_err(|e| e.to_string())?;

    let console = Rc::new(V2MmioCombinedDevice::new(0x000B_E5C0));
    let mut sim = Simulation::with_size_layered(128, 640, 16);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(&program)
        .with_rom_size(128)
        .with_ram_size(128)
        .with_extended_pc()
        .with_mmio(V2MmioHandle::from_rc(console.clone()))
        .with_synth_blocks(V2SynthConfig::max_authority())
        .build(&mut sim);

    let mut metrics = TileCpuMetrics::default();
    let mut total_critical = 0u64;

    let start = Instant::now();
    for _ in 0..case.max_cycles {
        if cpu.is_halted() {
            break;
        }
        let stats = cpu.step(&mut sim);
        metrics.cycles += 1;
        if cpu.last_stage_x_valid() {
            metrics.instructions_executed += 1;
        }
        metrics.total_deltas += stats.total_deltas as u64;
        metrics.total_tiles_evaluated += stats.tiles_evaluated as u64;
        metrics.total_tiles_switched += stats.tiles_switched as u64;
        metrics.max_critical_path = metrics.max_critical_path.max(stats.critical_path_deltas);
        total_critical += stats.critical_path_deltas as u64;
    }
    let elapsed = start.elapsed();

    if !cpu.is_halted() {
        return Err(format!(
            "compiled benchmark '{}' did not halt within {} cycles",
            case.name, case.max_cycles
        )
        .into());
    }
    if metrics.cycles > 0 {
        metrics.avg_critical_path = total_critical as f64 / metrics.cycles as f64;
        metrics.ipc = metrics.instructions_executed as f64 / metrics.cycles as f64;
    }

    Ok(V2CompiledBenchOutcome {
        case_name: case.name,
        program_words: program.len(),
        metrics,
        console: console.ref_pack.console_string(),
        r0: cpu.read_reg(&sim, 0),
        elapsed,
        per_path: cpu.per_path_prop_stats(),
        settle_scan: cpu.settle_scan_total(),
    })
}

/// The compiled benchmark corpus. Each case stresses a specific lever; inputs are
/// sized for a few-thousand ticks so the suite runs quickly. (Array kernels — sort,
/// sieve, matrix — arrive once the language has arrays.)
pub const V2_COMPILED_BENCHMARKS: &[V2CompiledBenchCase] = &[
    // Recursion / call-frame overhead.
    V2CompiledBenchCase {
        name: "fib_recursive",
        source: r#"
            int fib(int n) {
                if (n < 2) { return n; }
                return fib(n - 1) + fib(n - 2);
            }
            int main() { return fib(10); }
        "#,
        max_cycles: 400_000,
        expected_console: "",
        expected_r0: 55,
    },
    // Loop + MUL.
    V2CompiledBenchCase {
        name: "fact_iterative",
        source: r#"
            int main() {
                int f = 1;
                int i = 1;
                while (i <= 10) { f = f * i; i = i + 1; }
                return f;
            }
        "#,
        max_cycles: 100_000,
        expected_console: "",
        expected_r0: 3_628_800,
    },
    // Loop + RAM-backed locals.
    V2CompiledBenchCase {
        name: "sum_loop",
        source: r#"
            int main() {
                int s = 0;
                int i = 1;
                while (i <= 50) { s = s + i; i = i + 1; }
                return s;
            }
        "#,
        max_cycles: 200_000,
        expected_console: "",
        expected_r0: 1275,
    },
    // While + subtraction, many iterations.
    V2CompiledBenchCase {
        name: "gcd_subtract",
        source: r#"
            int gcd(int a, int b) {
                while (a != b) {
                    if (a > b) { a = a - b; } else { b = b - a; }
                }
                return a;
            }
            int main() { return gcd(1000, 12); }
        "#,
        max_cycles: 400_000,
        expected_console: "",
        expected_r0: 4,
    },
    // MUL in a loop.
    V2CompiledBenchCase {
        name: "sum_squares",
        source: r#"
            int main() {
                int s = 0;
                int i = 1;
                while (i <= 20) { s = s + i * i; i = i + 1; }
                return s;
            }
        "#,
        max_cycles: 200_000,
        expected_console: "",
        expected_r0: 2870,
    },
    // print_int / div10 (O(n) repeated subtraction) cost.
    V2CompiledBenchCase {
        name: "print_count",
        source: r#"
            int main() {
                int i = 1;
                while (i <= 20) { print_int(i); putchar(32); i = i + 1; }
                return 0;
            }
        "#,
        max_cycles: 300_000,
        expected_console: "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ",
        expected_r0: 0,
    },
    // Large-number printing — where O(n) div10 is brutal (54321/10 ≈ 5400 subtractions)
    // and the chunked div10 (P3) is essential.
    V2CompiledBenchCase {
        name: "print_large",
        source: r#"
            int main() {
                int i = 0;
                while (i < 5) { print_int(54321); putchar(32); i = i + 1; }
                return 0;
            }
        "#,
        max_cycles: 300_000,
        expected_console: "54321 54321 54321 54321 54321 ",
        expected_r0: 0,
    },
    // Sprint 382 (Gate F): array kernel — fill a[i]=i*i, sum it. Exercises StoreIndex,
    // Index, and base+offset RAM addressing on the physical CPU. sum(i*i, 0..16) = 1240.
    V2CompiledBenchCase {
        name: "array_sum_squares",
        source: r#"
            int a[16];
            int main() {
                int i = 0;
                while (i < 16) { a[i] = i * i; i = i + 1; }
                int s = 0;
                int k = 0;
                while (k < 16) { s = s + a[k]; k = k + 1; }
                return s;
            }
        "#,
        max_cycles: 200_000,
        expected_console: "",
        expected_r0: 1240,
    },
];

/// Performance goldens: `(name, program_words, instructions_executed, cycles)`.
/// These are the **baseline measured 2026-06-01** (naive codegen, repeated-subtraction
/// div10, software MUL). Perf-track sprints drive `instructions_executed`/`cycles` down
/// and update these — the test makes every change intentional and visible.
pub const V2_COMPILED_BENCH_GOLDENS: &[(&str, usize, u64, u64)] = &[
    // Cumulative baseline -> P4. P2 (S378): register allocation + spill elimination.
    // P3 (S379): chunked div10 + remainder via (n - q*10). P4 (S380): single-arg calls
    // (no PUSH/POP) + leaf functions return via RET (no LR save). Totals vs the original
    // baseline: fib -38%, sum_loop -60%, fact -58%, gcd -54%, sum_squares -53%,
    // print_count -48%; print_large (5-digit) is now feasible where O(n) div10 was ~180k.
    ("fib_recursive", 45, 4254, 4255),
    ("fact_iterative", 31, 153, 154),
    ("sum_loop", 31, 673, 674),
    ("gcd_subtract", 52, 1312, 1313),
    ("sum_squares", 37, 403, 404),
    ("print_count", 143, 1806, 1807),
    ("print_large", 143, 4537, 4538),
    // Sprint 382 (Gate F): array kernel. Golden captured 2026-06-04.
    ("array_sum_squares", 80, 660, 661),
];

/// Render a one-line report row for a benchmark outcome.
/// Sprint 384: wall-clock columns added — under the JIT settle, `tiles/instr`
/// counts the unconditional 5,492-op native scan as evals, so wall time is
/// the honest cross-build comparison metric.
pub fn format_bench_row(o: &V2CompiledBenchOutcome) -> String {
    let wall_ms = o.elapsed.as_secs_f64() * 1_000.0;
    let cyc_per_sec = if o.elapsed.as_secs_f64() > 0.0 {
        o.metrics.cycles as f64 / o.elapsed.as_secs_f64()
    } else {
        0.0
    };
    format!(
        "{:<16} {:>6} {:>9} {:>9} {:>14} {:>11.0} {:>9.1} {:>8.0} {:>10}",
        o.case_name,
        o.program_words,
        o.metrics.instructions_executed,
        o.metrics.cycles,
        o.metrics.total_tiles_evaluated,
        o.tiles_per_instruction(),
        wall_ms,
        cyc_per_sec,
        o.r0,
    )
}

/// Sprint 381 (Perf L3 profiling): per-phase tile-eval breakdown for one outcome —
/// `(phase, evals, evals_per_instr, pct_of_total)` for cone/settle/constswap/branch/commit.
pub fn phase_breakdown(o: &V2CompiledBenchOutcome) -> Vec<(&'static str, u64, f64, f64)> {
    let total: u64 = o.per_path.iter().map(|(_, e)| *e).sum();
    o.per_path
        .iter()
        .enumerate()
        .map(|(i, (_, e))| {
            let per_instr = if o.metrics.instructions_executed > 0 {
                *e as f64 / o.metrics.instructions_executed as f64
            } else {
                0.0
            };
            let pct = if total > 0 {
                *e as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            (PER_PATH_NAMES[i], *e, per_instr, pct)
        })
        .collect()
}

/// One row of the per-phase profiling table: evals/instruction for each phase.
pub fn format_phase_row(o: &V2CompiledBenchOutcome) -> String {
    let bd = phase_breakdown(o);
    let v = |i: usize| bd[i].2; // evals per instruction
    format!(
        "{:<16} {:>10.0} {:>10.0} {:>10.0} {:>8.1} {:>8.1}",
        o.case_name,
        v(0), // cone
        v(1), // settle
        v(2), // constswap
        v(3), // branch
        v(4), // commit
    )
}

/// Header for [`format_phase_row`].
pub fn phase_report_header() -> String {
    format!(
        "{:<16} {:>10} {:>10} {:>10} {:>8} {:>8}  (tile-evals per instruction)",
        "benchmark", "cone", "settle", "constswap", "branch", "commit",
    )
}

/// The report header matching [`format_bench_row`].
pub fn bench_report_header() -> String {
    format!(
        "{:<16} {:>6} {:>9} {:>9} {:>14} {:>11} {:>9} {:>8} {:>10}",
        "benchmark", "words", "instrs", "ticks", "tiles_eval", "tiles/instr", "wall_ms", "cyc/s", "R0",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_cpu::v2_iss::V2Iss;

    /// Run a source program on the **ISS** (the authoritative reference) with MMIO,
    /// returning `(r0, console)`.
    fn iss_run(src: &str, max_steps: usize) -> (u64, String) {
        let words = compile_source(src).expect("compile_source failed");
        let mut iss = V2Iss::from_program_with_mmio(&words);
        iss.enable_wide_pc();
        for _ in 0..max_steps {
            if iss.state().halted {
                return (iss.state().regs[0], iss.console_string());
            }
            iss.step();
        }
        panic!("ISS program did not halt within {max_steps} steps");
    }

    /// Run a source program on the **physical tile CPU** (the configuration the compiler
    /// targets), returning `(r0, console)`. Mirrors [`run_v2_compiled_benchmark`] but
    /// accepts a non-`'static` source (so tests can build programs at runtime).
    fn phys_run(src: &str, max_cycles: u64) -> (u64, String) {
        let program = compile_source(src).expect("compile_source failed");
        let console = Rc::new(V2MmioCombinedDevice::new(0x000B_E5C0));
        let mut sim = Simulation::with_size_layered(128, 640, 16);
        // Wide PC (16-bit) so the physical PC sequences past the 256-instruction line —
        // the `extended_pc` path masks the physical PC to 8 bits, which silently wraps a
        // program > 256 words. `with_wide_pc()` implies `extended_pc`. This matches the
        // ISS reference (`enable_wide_pc`).
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_wide_pc()
            .with_mmio(V2MmioHandle::from_rc(console.clone()))
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);
        for _ in 0..max_cycles {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
        }
        assert!(cpu.is_halted(), "physical program did not halt in {max_cycles} cycles");
        (cpu.read_reg(&sim, 0), console.ref_pack.console_string())
    }

    /// Sprint 382 (Gate F): the rigor bar — every array program must produce **identical**
    /// R0 and console on the ISS (reference) and the physical tile CPU. This is the
    /// "the gates are real" guarantee for the new array path (StoreIndex / Index /
    /// base+offset RAM addressing).
    #[test]
    fn test_v2_gate_f_array_iss_physical_differential() {
        // (name, source, max_cycles). Kept small so physical sim stays fast.
        let cases: &[(&str, &str, u64)] = &[
            (
                "fill_sum",
                r#"
                    int a[8];
                    int main() {
                        int i = 0;
                        while (i < 8) { a[i] = i + 1; i = i + 1; }
                        int s = 0;
                        int k = 0;
                        while (k < 8) { s = s + a[k]; k = k + 1; }
                        return s;
                    }
                "#,
                100_000,
            ),
            (
                "initializer_readback",
                r#"
                    int data[] = { 100, 200, 44 };
                    int main() { return data[0] + data[1] + data[2]; }
                "#,
                50_000,
            ),
            (
                "reverse_in_place",
                r#"
                    int a[] = { 1, 2, 3, 4, 5, 6 };
                    int main() {
                        int i = 0;
                        int j = 5;
                        while (i < j) {
                            int t = a[i];
                            a[i] = a[j];
                            a[j] = t;
                            i = i + 1;
                            j = j - 1;
                        }
                        // Fold to a single number: sum of i*a[i].
                        int s = 0;
                        int k = 0;
                        while (k < 6) { s = s + k * a[k]; k = k + 1; }
                        return s;
                    }
                "#,
                100_000,
            ),
        ];
        for &(name, src, max_cycles) in cases {
            let (iss_r0, iss_con) = iss_run(src, max_cycles as usize);
            let case = V2CompiledBenchCase {
                name,
                source: src,
                max_cycles,
                expected_console: "",
                expected_r0: 0,
            };
            let phys = run_v2_compiled_benchmark(&case)
                .unwrap_or_else(|e| panic!("physical run '{name}' failed: {e}"));
            assert_eq!(phys.r0, iss_r0, "{name}: R0 ISS({iss_r0}) != physical({})", phys.r0);
            assert_eq!(
                phys.console, iss_con,
                "{name}: console ISS({iss_con:?}) != physical({:?})",
                phys.console
            );
        }
    }

    /// The Brainfuck interpreter, written in the C-like language. `prog`/`tape` are
    /// global arrays; bracket matching is a depth-scan (no extra memory). Locals
    /// (plen, pc, ptr, op, depth) fit register mode. Sprint 382 (Gate F).
    const BF_INTERPRETER: &str = r#"
        int run(int plen) {
            int pc = 0;
            int ptr = 0;
            while (pc < plen) {
                int op = prog[pc];
                if (op == 62) { ptr = ptr + 1; }
                if (op == 60) { ptr = ptr - 1; }
                if (op == 43) { tape[ptr] = tape[ptr] + 1; }
                if (op == 45) { tape[ptr] = tape[ptr] - 1; }
                if (op == 46) { putchar(tape[ptr]); }
                if (op == 91) {
                    if (tape[ptr] == 0) {
                        int depth = 1;
                        while (depth > 0) {
                            pc = pc + 1;
                            if (prog[pc] == 91) { depth = depth + 1; }
                            if (prog[pc] == 93) { depth = depth - 1; }
                        }
                    }
                }
                if (op == 93) {
                    if (tape[ptr] != 0) {
                        int depth = 1;
                        while (depth > 0) {
                            pc = pc - 1;
                            if (prog[pc] == 91) { depth = depth - 1; }
                            if (prog[pc] == 93) { depth = depth + 1; }
                        }
                    }
                }
                pc = pc + 1;
            }
            return 0;
        }
        int main() { return run(PLEN); }
    "#;

    /// Assemble a full compilable program from a Brainfuck source string: encode each BF
    /// byte as an `int` initializer for `prog`, declare a zeroed `tape`, append the
    /// interpreter with `PLEN` substituted.
    fn bf_program(bf: &str, tape_len: usize) -> String {
        let codes: Vec<String> = bf.bytes().map(|b| (b as u32).to_string()).collect();
        let body = BF_INTERPRETER.replace("PLEN", &bf.len().to_string());
        format!(
            "int prog[] = {{ {} }};\nint tape[{}];\n{}",
            codes.join(", "),
            tape_len,
            body
        )
    }

    /// **The flagship (Gate F, F3).** A Brainfuck interpreter — written in our language,
    /// compiled by our compiler — runs a BF program (a multiply loop + increments) that
    /// prints "12345" on the **physical tile CPU**, bit-for-bit identical to the ISS
    /// reference. This is the self-hosting-class proof: the thing running on the logic
    /// tiles is itself a universal language processor. (Program + tape stay under 41
    /// cells so they fit one contiguous non-MMIO RAM region.)
    #[test]
    fn test_v2_gate_f_brainfuck_count_iss_physical() {
        // cell0=7; [>+++++++<-] makes cell1 = 49 ('1'); then +.+.+.+. prints 2,3,4,5.
        let bf = "+++++++[>+++++++<-]>.+.+.+.+.";
        let src = bf_program(bf, 4);

        let (iss_r0, iss_con) = iss_run(&src, 2_000_000);
        assert_eq!(iss_con, "12345", "ISS console");
        assert_eq!(iss_r0, 0, "ISS r0");

        let (phys_r0, phys_con) = phys_run(&src, 1_000_000);
        assert_eq!(phys_con, iss_con, "console: physical({phys_con:?}) != ISS({iss_con:?})");
        assert_eq!(phys_r0, iss_r0, "r0: physical({phys_r0}) != ISS({iss_r0})");
    }

    /// A second, independent BF program — proves the interpreter is general, not tuned to
    /// one input. A multiply loop prints "AB". Sprint 382 (Gate F, F4).
    #[test]
    fn test_v2_gate_f_brainfuck_second_program_iss_physical() {
        // cell0=13, [>+++++<-] -> cell1=65 ('A'); >.+. prints 'A' then 'B'.
        let bf = "+++++++++++++[>+++++<-]>.+.";
        let src = bf_program(bf, 4);
        let (iss_r0, iss_con) = iss_run(&src, 2_000_000);
        assert_eq!(iss_con, "AB", "ISS console");
        let (phys_r0, phys_con) = phys_run(&src, 1_000_000);
        assert_eq!(phys_con, iss_con, "console parity");
        assert_eq!(phys_r0, iss_r0, "r0 parity");
    }

    #[test]
    fn test_v2_compiled_bench_correctness_and_goldens() {
        for case in V2_COMPILED_BENCHMARKS {
            let o = run_v2_compiled_benchmark(case)
                .unwrap_or_else(|e| panic!("benchmark '{}' failed: {e}", case.name));
            // Correctness is strict.
            assert_eq!(o.r0, case.expected_r0, "{}: R0", case.name);
            assert_eq!(o.console, case.expected_console, "{}: console", case.name);

            // Metrics vs golden (skip while the golden is the 0 placeholder).
            if let Some(&(_, gw, gi, gc)) = V2_COMPILED_BENCH_GOLDENS
                .iter()
                .find(|(n, ..)| *n == case.name)
            {
                if (gw, gi, gc) != (0, 0, 0) {
                    assert_eq!(o.program_words, gw, "{}: program_words", case.name);
                    assert_eq!(
                        o.metrics.instructions_executed, gi,
                        "{}: instructions_executed",
                        case.name
                    );
                    assert_eq!(o.metrics.cycles, gc, "{}: cycles", case.name);
                }
            }
        }
    }
}
