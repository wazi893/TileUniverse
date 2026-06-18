//! Step 0 of the V2-on-CUDA migration: the **GPU charter-set audit**.
//!
//! Before writing any CUDA, decompose the compiled benchmark corpus on the CPU and
//! tabulate exactly what each program needs from the tile CPU per cycle. The output
//! tells us which programs are already device-portable under the restricted first-win
//! regime (op-clean, register-resident/RAM-light, MMIO-free, no upper-bank fallback,
//! runs correctly) versus which need later compatibility layers.
//!
//! Columns map 1:1 to the requested audit:
//!   * per-phase compact op counts (cone/settle/branch/commit/clock)
//!   * unsupported COP_GENERIC / COP_WIRE by phase (via `DensePipelinePreflight`)
//!   * LD/ST count + MMIO usage (static decode + dynamic console)
//!   * software-authority ops, especially MUL (opcode 0x04, imm5==1 — no device op)
//!   * upper-bank / compact_eval_inhibit / constswap incidence (runtime counter)
//!
//! This mirrors `run_v2_compiled_benchmark`'s build config exactly (extended_pc +
//! max_authority + MMIO), so the preflight is the same one the 389a oracle validates.
//!
//! Run: `cargo run --release --example v2_gpu_charter_audit`

use std::rc::Rc;

use engine::simulation::Simulation;
use engine::tile_cpu::v2_compiler_bench::{V2_COMPILED_BENCHMARKS, V2CompiledBenchCase};
use engine::tile_cpu::{
    DensePipelineStepper, TileCpuV2, V2Builder, V2MmioCombinedDevice, V2MmioHandle, V2SynthConfig,
    compile_source, decode_v2_word,
};

/// Static per-program instruction profile (no build, no run).
struct StaticProfile {
    words: usize,
    ld: usize,
    st: usize,
    mul: usize,
}

fn static_profile(program: &[u32]) -> StaticProfile {
    let mut ld = 0;
    let mut st = 0;
    let mut mul = 0;
    for &word in program {
        let d = decode_v2_word(word);
        match d.opcode {
            0x16 | 0x18 => ld += 1,          // LD, LDB
            0x17 | 0x19 => st += 1,          // ST, STB
            0x04 if d.imm5 == 1 => mul += 1, // ADD opcode, imm5==1 == MUL (software authority)
            _ => {}
        }
    }
    StaticProfile {
        words: program.len(),
        ld,
        st,
        mul,
    }
}

/// Build the tile CPU exactly as `run_v2_compiled_benchmark` does (and the 389a
/// dense-stepper oracle), so the preflight + inhibit counters are apples-to-apples.
fn build_case(program: &[u32]) -> (TileCpuV2, Simulation, Rc<V2MmioCombinedDevice>) {
    let console = Rc::new(V2MmioCombinedDevice::new(0x000B_E5C0));
    let mut sim = Simulation::with_size_layered(128, 640, 16);
    let cpu = V2Builder::new()
        .with_origin(0, 0)
        .with_program(program)
        .with_rom_size(128)
        .with_ram_size(128)
        .with_extended_pc()
        .with_mmio(V2MmioHandle::from_rc(console.clone()))
        .with_synth_blocks(V2SynthConfig::max_authority())
        .build(&mut sim);
    (cpu, sim, console)
}

/// Per-phase unsupported-op tallies (COP_GENERIC / COP_WIRE), in phase order
/// cone/settle/branch/commit/clock.
fn unsupported_by_phase(pf: &engine::tile_cpu::DensePipelinePreflight) -> [usize; 5] {
    let mut by = [0usize; 5];
    for u in &pf.unsupported_ops {
        let i = match u.phase {
            "cone" => 0,
            "settle" => 1,
            "branch" => 2,
            "commit" => 3,
            "clock" => 4,
            _ => continue,
        };
        by[i] += 1;
    }
    by
}

struct Row {
    name: &'static str,
    s: StaticProfile,
    // preflight
    cone: usize,
    settle: usize,
    branch: usize,
    commit: usize,
    clock: usize,
    unsupported: [usize; 5],
    op_clean: bool,
    preflight_ok: bool,
    // dynamic
    cycles: u64,
    r0: u64,
    expected_r0: u64,
    inhibit: u64,
    console_len: usize,
    correct: bool,
}

fn audit_case(case: &V2CompiledBenchCase) -> Row {
    let program = compile_source(case.source).expect("compile_source failed");
    let s = static_profile(&program);

    let (cpu, mut sim, console) = build_case(&program);

    // Preflight via the public stepper constructor. If the settle kernel won't build,
    // the program is disqualified outright (and we record why).
    let (cone, settle, branch, commit, clock, unsupported, op_clean, preflight_ok) =
        match DensePipelineStepper::new(&cpu, &sim) {
            Ok(stepper) => {
                let pf = stepper.preflight();
                (
                    pf.cone_ops,
                    pf.settle_ops,
                    pf.branch_ops,
                    pf.commit_ops,
                    pf.clock_ops,
                    unsupported_by_phase(pf),
                    pf.is_gpu_op_clean(),
                    true,
                )
            }
            Err(e) => {
                eprintln!("  [{}] settle kernel build failed: {e}", case.name);
                (0, 0, 0, 0, 0, [0; 5], false, false)
            }
        };

    // Dynamic run to halt (bounded by the case's max_cycles; the corpus halts fast).
    let mut cycles = 0u64;
    for _ in 0..case.max_cycles {
        if cpu.is_halted() {
            break;
        }
        cpu.step(&mut sim);
        cycles += 1;
    }
    let r0 = cpu.read_reg(&sim, 0);
    let inhibit = cpu.compact_eval_inhibit_count();
    let console_str = console.ref_pack.console_string();
    let correct = cpu.is_halted() && r0 == case.expected_r0 && console_str == case.expected_console;

    Row {
        name: case.name,
        s,
        cone,
        settle,
        branch,
        commit,
        clock,
        unsupported,
        op_clean,
        preflight_ok,
        cycles,
        r0,
        expected_r0: case.expected_r0,
        inhibit,
        console_len: console_str.len(),
        correct,
    }
}

fn main() {
    println!("V2 -> CUDA migration, Step 0: GPU charter-set audit\n");

    let rows: Vec<Row> = V2_COMPILED_BENCHMARKS.iter().map(audit_case).collect();

    // --- 1a. Static instruction scan ---
    println!("=== 1a. STATIC INSTRUCTION SCAN ===");
    println!(
        "{:<18}{:>6}{:>5}{:>5}{:>5}   note",
        "program", "words", "LD", "ST", "MUL"
    );
    println!("{}", "-".repeat(70));
    for r in &rows {
        let note = if r.s.mul > 0 {
            "MUL = software authority (no COP_MUL on device)"
        } else if r.s.ld + r.s.st == 0 {
            "register-resident"
        } else {
            "RAM/MMIO traffic"
        };
        println!(
            "{:<18}{:>6}{:>5}{:>5}{:>5}   {}",
            r.name, r.s.words, r.s.ld, r.s.st, r.s.mul, note
        );
    }

    // --- 1b. Build + preflight: per-phase op counts + unsupported COP_GENERIC/COP_WIRE ---
    println!("\n=== 1b. PREFLIGHT: per-phase compact op counts + unsupported(GENERIC/WIRE) ===");
    println!(
        "{:<18}{:>6}{:>7}{:>7}{:>7}{:>6}   {:<22}op_clean",
        "program", "cone", "settle", "branch", "commit", "clock", "unsupp[c/s/b/m/k]"
    );
    println!("{}", "-".repeat(85));
    for r in &rows {
        if !r.preflight_ok {
            println!(
                "{:<18}  (settle kernel build failed — disqualified)",
                r.name
            );
            continue;
        }
        let u = r.unsupported;
        println!(
            "{:<18}{:>6}{:>7}{:>7}{:>7}{:>6}   {:<22}{}",
            r.name,
            r.cone,
            r.settle,
            r.branch,
            r.commit,
            r.clock,
            format!("{}/{}/{}/{}/{}", u[0], u[1], u[2], u[3], u[4]),
            if r.op_clean { "yes" } else { "NO" }
        );
    }

    // --- 1c. Dynamic run: cycles, correctness, upper-bank inhibit, MMIO ---
    println!("\n=== 1c. DYNAMIC RUN (to halt): inhibit/constswap incidence + MMIO ===");
    println!(
        "{:<18}{:>8}{:>14}{:>9}{:>12}   mmio?",
        "program", "cycles", "R0(got/exp)", "inhibit", "console_len"
    );
    println!("{}", "-".repeat(78));
    for r in &rows {
        let mmio = if r.console_len > 0 { "yes" } else { "no" };
        let r0col = if r.r0 == r.expected_r0 {
            format!("{}", r.r0)
        } else {
            format!("{}/{}", r.r0, r.expected_r0)
        };
        println!(
            "{:<18}{:>8}{:>14}{:>9}{:>12}   {}",
            r.name, r.cycles, r0col, r.inhibit, r.console_len, mmio
        );
    }

    // --- 2. Charter-set verdict ---
    // KEY FINDING: there is NO zero-memory program — the V2 call ABI spills frames to the
    // RAM stack, so even leaf loops do LD/ST. Per-lane RAM SoA is therefore a REQUIRED
    // capability for the very first megakernel (not a phase-2 add-on). "mem" is shown as
    // an informational footprint column, not a disqualifier. The charter bar is the set of
    // gates that are genuinely deferrable: MMIO, upper-bank inhibit, and software MUL.
    println!("\n=== 2. GPU CHARTER-SET VERDICT (first-win regime) ===");
    println!(
        "charter = op-clean & MMIO-free & no-inhibit & correct & MUL-free   (RAM SoA assumed; all programs need it)"
    );
    println!(
        "{:<18}{:>9}{:>6}{:>10}{:>10}{:>9}{:>9}   CHARTER",
        "program", "op_clean", "mem", "mmio_free", "no_inhib", "correct", "mul_free"
    );
    println!("{}", "-".repeat(92));
    let mut charter: Vec<&str> = Vec::new();
    for r in &rows {
        let op_clean = r.preflight_ok && r.op_clean;
        let mem = r.s.ld + r.s.st; // informational: RAM footprint (stack + spills + arrays)
        let mmio_free = r.console_len == 0;
        let no_inhib = r.inhibit == 0;
        let mul_free = r.s.mul == 0;
        let member = op_clean && mmio_free && no_inhib && r.correct && mul_free;
        if member {
            charter.push(r.name);
        }
        let yn = |b: bool| if b { "yes" } else { "NO" };
        println!(
            "{:<18}{:>9}{:>6}{:>10}{:>10}{:>9}{:>9}   {}",
            r.name,
            yn(op_clean),
            mem,
            yn(mmio_free),
            yn(no_inhib),
            yn(r.correct),
            yn(mul_free),
            if member { "** YES **" } else { "" }
        );
    }

    println!(
        "\nCharter set (megakernel first target): {}",
        if charter.is_empty() {
            "(none)".to_string()
        } else {
            charter.join(", ")
        }
    );
    println!(
        "\nSequencing implied by the data:\n  \
         - RAM SoA    -> REQUIRED for the first win (no zero-mem program exists; ABI uses the stack)\n  \
         - MUL users  -> need a device COP_MUL (stage-X software authority has no tile op): fact, sum_squares, print_*, array\n  \
         - MMIO users -> device->host event ring drained every N cycles: print_count, print_large\n  \
         - inhibit>0  -> per-lane hold/freeze masks for upper-bank constswap (>128 words): print_count, print_large\n  \
         note: preflight is identical for every program (cone/settle/branch/commit/clock + 0 unsupported) —\n        \
         op-cleanliness is a property of the BUILD CONFIG, not the program; the GENERIC/WIRE gate is already closed."
    );
}
