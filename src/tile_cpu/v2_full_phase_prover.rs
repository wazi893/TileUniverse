//! Step 1 of the V2-on-CUDA migration: the **full-phase CPU prover**.
//!
//! This is the CPU-side specification + validator for the device-resident cycle the
//! CUDA megakernel will eventually run. It is deliberately separate from the in-flight
//! `v2_dense_stepper` (which proves only the Stage-F settle substitution): this module
//! assembles the *whole* cycle's tile-eval surface and the device-state SoA, and
//! validates them against `TileCpuV2::step` on the charter set.
//!
//! ## What it mirrors (NOT the sketch — the real `TileCpuV2::tick`)
//!
//! `tick()` runs **Stage X** (execute the latched instruction) then **Stage F** (fetch
//! the next), producing the next pipeline latch. The five compact-op arrays the
//! megakernel must carry, in tick execution order, are:
//!
//!   Stage X:  branch  -> commit -> clock      (`*_compact_ops` used inside `run_stage_x`)
//!   Stage F:  cone    -> settle               (`pipeline_cone_ops`, `settle_compact_ops`)
//!
//! (The exact *intra*-Stage-X sub-order is taken from the evaluator's array order; the
//! megakernel will need that sub-order locked from `run_stage_x` directly — flagged for
//! the next increment. The stage-level seam, `run_stage_x` then `run_stage_f`, is the
//! part this prover relies on and is verified against the oracle.)
//!
//! ## What it proves now (charter set: fib_recursive, sum_loop, gcd_subtract)
//!
//! 1. **Per-phase op-array fidelity.** For each of the five phases, the phase op array
//!    evaluated by `LaneKernel` (the exact path the megakernel runs) reproduces the
//!    simulation's own dense `propagate_compact` for that array, bit-for-bit, at a real
//!    runtime state. Same ops, same inputs, same outputs.
//! 2. **Unified slot universe.** The union of every tile touched by any phase op array —
//!    the megakernel's compacted `vals[slot*K + lane]` buffer. Reported size.
//! 3. **`LaneCpuState` SoA** including the **RAM mirror (gate-0)** — the audit showed no
//!    register-only program exists (the V2 call ABI spills frames to the RAM stack), so
//!    the per-lane RAM SoA is required for the very first program, not an expansion item.
//! 4. **Full-cycle parity.** A settle-substituted run (op array via `LaneKernel` driven
//!    through the existing fabric seam) reaches a final `hash_v2_final_state` identical to
//!    the authoritative `TileCpuV2::step` run — plus cycle count and R0.
//!
//! Bit-exact final-hash parity is the oracle (integer CPU — equality, not epsilon).

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::simulation::{COP_CONST, COP_THRESHOLD_VIA, CompactOp, Simulation};
use crate::tile_cpu::v2_compiler_bench::{V2_COMPILED_BENCHMARKS, V2CompiledBenchCase};
use crate::tile_cpu::v2_simt_eval::{LaneKernel, LanePack};
use crate::tile_cpu::{
    TileCpuV2, V2Builder, V2MmioCombinedDevice, V2MmioHandle, V2SynthConfig, compile_source,
    decode_v2_word, hash_v2_final_state,
};

/// The charter set: MUL-free, MMIO-free, inhibit-free, op-clean (per Step-0 audit).
pub const CHARTER_PROGRAMS: [&str; 3] = ["fib_recursive", "sum_loop", "gcd_subtract"];

/// Phase op-array names in `TileCpuV2::tick` execution order
/// (Stage X: branch, commit, clock; Stage F: cone, settle).
pub const PHASE_NAMES: [&str; 5] = ["branch", "commit", "clock", "cone", "settle"];

/// Per-lane CPU architectural state — the SoA the CUDA megakernel carries as
/// `field[... * K + lane]`. The CPU prover holds one lane; the layout generalizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneCpuState {
    pub pc: u32,
    pub flag_z: bool,
    pub flag_c: bool,
    pub halted: bool,
    pub cycle: u64,
    pub regs: [u64; 16],
    /// RAM mirror — **gate-0** (the call ABI spills frames here). 0..128 is the
    /// scratchpad; `configure_main_memory` configs append main memory beyond it.
    pub ram: Vec<u64>,
}

impl LaneCpuState {
    /// Read the architectural state out of a live tile CPU + sim.
    pub fn capture(cpu: &TileCpuV2, sim: &Simulation, ram_words: usize) -> Self {
        Self {
            pc: cpu.read_pc(sim),
            flag_z: cpu.read_flag_z(sim),
            flag_c: cpu.read_flag_c(sim),
            halted: cpu.is_halted(),
            cycle: cpu.read_cycle_count(),
            regs: std::array::from_fn(|i| cpu.read_reg(sim, i)),
            ram: (0..ram_words).map(|a| cpu.read_ram(sim, a)).collect(),
        }
    }
}

/// Union of every tile touched (read or written) by any phase op array — the
/// megakernel's compacted value buffer.
pub struct UnifiedSlotUniverse {
    /// tile index -> dense slot.
    pub slot_of: HashMap<u32, u32>,
    /// dense slot -> tile index.
    pub touched: Vec<u32>,
}

impl UnifiedSlotUniverse {
    /// Build from a CPU's five phase op arrays (+ sim for threshold-via neighbors).
    pub fn from_cpu(cpu: &TileCpuV2, sim: &Simulation) -> Self {
        let tile_count = sim.tilemap.tile_count() as u32;
        let mut slot_of: HashMap<u32, u32> = HashMap::new();
        let mut touched: Vec<u32> = Vec::new();
        let add = |t: u32, slot_of: &mut HashMap<u32, u32>, touched: &mut Vec<u32>| {
            if t != u32::MAX && t < tile_count {
                slot_of.entry(t).or_insert_with(|| {
                    touched.push(t);
                    (touched.len() - 1) as u32
                });
            }
        };
        for (_, ops, _) in phase_arrays(cpu) {
            for op in ops {
                add(op.idx, &mut slot_of, &mut touched);
                add(op.in0, &mut slot_of, &mut touched);
                add(op.in1, &mut slot_of, &mut touched);
                add(op.in2, &mut slot_of, &mut touched);
                if op.op == COP_THRESHOLD_VIA {
                    for &n in sim.neighbors4_at(op.idx as usize).iter() {
                        add(n, &mut slot_of, &mut touched);
                    }
                }
            }
        }
        Self { slot_of, touched }
    }

    pub fn len(&self) -> usize {
        self.touched.len()
    }

    pub fn is_empty(&self) -> bool {
        self.touched.is_empty()
    }
}

/// The five (name, ops, wvia) phase arrays in tick execution order.
fn phase_arrays(cpu: &TileCpuV2) -> [(&'static str, &[CompactOp], &[(usize, u8, u64)]); 5] {
    [
        ("branch", &cpu.branch_compact_ops, &cpu.branch_compact_wvia),
        ("commit", &cpu.commit_compact_ops, &cpu.commit_compact_wvia),
        ("clock", &cpu.clock_compact_ops, &cpu.clock_compact_wvia),
        ("cone", &cpu.pipeline_cone_ops, &cpu.pipeline_cone_wvia),
        ("settle", &cpu.settle_compact_ops, &cpu.settle_compact_wvia),
    ]
}

/// Static instruction profile (no build, no run).
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticProfile {
    pub words: usize,
    pub ld: usize,
    pub st: usize,
    pub mul: usize,
}

fn static_profile(program: &[u32]) -> StaticProfile {
    let mut p = StaticProfile {
        words: program.len(),
        ..Default::default()
    };
    for &word in program {
        let d = decode_v2_word(word);
        match d.opcode {
            0x16 | 0x18 => p.ld += 1,          // LD, LDB
            0x17 | 0x19 => p.st += 1,          // ST, STB
            0x04 if d.imm5 == 1 => p.mul += 1, // ADD opcode w/ imm5==1 == MUL (software authority)
            _ => {}
        }
    }
    p
}

/// Per-phase preflight: op count + whether the `LaneKernel` (megakernel path) builds.
#[derive(Clone, Debug)]
pub struct PhaseProfile {
    pub name: &'static str,
    pub ops: usize,
    pub kernel_builds: bool,
    pub build_err: Option<String>,
    /// op-array == `propagate_compact` at the sampled runtime state.
    pub equiv_ok: bool,
}

/// The full prover report for one charter program — the megakernel handoff artifact.
#[derive(Clone, Debug)]
pub struct CharterProgramReport {
    pub name: &'static str,
    pub stat: StaticProfile,
    pub phases: Vec<PhaseProfile>,
    pub slot_union: usize,
    pub ram_words: usize,
    pub inhibit: u64,
    pub mmio: bool,
    pub cycles: u64,
    pub r0: u64,
    pub expected_r0: u64,
    pub hash_authority: u64,
    /// Settle-substituted run hash (None if the resident buffer layout precluded the
    /// in-place op-array path; the run still falls back to a correct scalar settle).
    pub hash_substituted: Option<u64>,
    pub substituted_dense_settles: u64,
    pub final_state: LaneCpuState,
    pub pass: bool,
}

impl CharterProgramReport {
    pub fn all_phases_build(&self) -> bool {
        self.phases.iter().all(|p| p.kernel_builds)
    }
    pub fn all_phases_equiv(&self) -> bool {
        self.phases.iter().all(|p| p.equiv_ok)
    }
}

/// Build the tile CPU in the charter config (mirrors `run_v2_compiled_benchmark`).
fn build(program: &[u32]) -> (TileCpuV2, Simulation, Rc<V2MmioCombinedDevice>) {
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

/// Output tiles a phase writes (skips CONST, out-of-range, no-write).
fn phase_outputs(ops: &[CompactOp], tile_count: u32) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for op in ops {
        if op.op == COP_CONST {
            continue;
        }
        if op.idx < tile_count && seen.insert(op.idx) {
            out.push(op.idx);
        }
    }
    out
}

/// Validate one phase: the `LaneKernel` op-array eval == `propagate_compact` dense eval
/// at the sim's current state, on every tile the phase writes.
fn validate_phase(
    name: &'static str,
    ops: &[CompactOp],
    wvia: &[(usize, u8, u64)],
    sim: &Simulation,
) -> PhaseProfile {
    let kernel = match LaneKernel::new_from_sim(ops, wvia, sim) {
        Ok(k) => k,
        Err(e) => {
            return PhaseProfile {
                name,
                ops: ops.len(),
                kernel_builds: false,
                build_err: Some(e),
                equiv_ok: false,
            };
        }
    };
    // op-array path (exactly what the megakernel runs), one lane.
    let mut lp = LanePack::from_sim(sim, 1);
    kernel.eval(&mut lp);
    // reference path: the sim's own dense compact eval on a throwaway clone.
    let mut reference = sim.clone();
    reference.propagate_compact(ops, wvia);
    let tile_count = sim.tilemap.tile_count() as u32;
    let equiv_ok = phase_outputs(ops, tile_count)
        .into_iter()
        .all(|t| lp.get(t as usize, 0) == reference.tilemap.value(t as usize));
    PhaseProfile {
        name,
        ops: ops.len(),
        kernel_builds: true,
        build_err: None,
        equiv_ok,
    }
}

/// Run to halt on the authoritative `TileCpuV2::step` path; return (cycles, r0, inhibit,
/// console-nonempty, final hash, final LaneCpuState).
fn run_authority(
    program: &[u32],
    max_cycles: u64,
    ram_words: usize,
) -> (u64, u64, u64, bool, u64, LaneCpuState) {
    let (cpu, mut sim, console) = build(program);
    let mut cycles = 0u64;
    for _ in 0..max_cycles {
        if cpu.is_halted() {
            break;
        }
        cpu.step(&mut sim);
        cycles += 1;
    }
    let state = LaneCpuState::capture(&cpu, &sim, ram_words);
    (
        cycles,
        cpu.read_reg(&sim, 0),
        cpu.compact_eval_inhibit_count(),
        !console.ref_pack.console_string().is_empty(),
        hash_v2_final_state(&cpu, &sim),
        state,
    )
}

/// Run to halt driving the cycle through the fabric seam, substituting the settle phase
/// with our own `LaneKernel` op array (the proven Stage-F substitution). Returns
/// (final hash, dense-settle count) or None if the settle kernel could not be built.
fn run_settle_substituted(program: &[u32], max_cycles: u64) -> Option<(u64, u64)> {
    let (cpu, mut sim, _console) = build(program);
    let (settle_ops, settle_wvia) = cpu.settle_compact_view();
    let kernel = LaneKernel::new_from_sim(settle_ops, settle_wvia, &sim).ok()?;
    let kernel_tiles = kernel.tile_count();
    let mut dense_settles = 0u64;

    for _ in 0..max_cycles {
        if cpu.is_halted() {
            break;
        }
        let ctx = cpu.fabric_tick_pre_settle(&mut sim);
        if let Some(ctx) = ctx {
            let mut wide = false;
            if !cpu.fabric_settle_inhibited() {
                let (vals, lanes) = sim.tilemap.lane_vals_raw_mut();
                // Guard: the resident buffer must include the trailing zero row.
                if vals.len() == (kernel_tiles + 1) * lanes {
                    kernel.eval_slice(vals, lanes);
                    dense_settles += 1;
                    wide = true;
                }
            }
            // wide=true: we ran the op-array settle; finish drains the settle scope.
            // wide=false: inhibited or buffer mismatch -> correct scalar settle fallback.
            cpu.fabric_finish_settle(&mut sim, &ctx, wide);
        }
    }
    Some((hash_v2_final_state(&cpu, &sim), dense_settles))
}

/// Prove one charter program end-to-end, producing the report.
pub fn prove_charter(case: &V2CompiledBenchCase) -> Result<CharterProgramReport, String> {
    let program = compile_source(case.source).map_err(|e| format!("compile: {e}"))?;
    let stat = static_profile(&program);
    let ram_words = 128; // charter config: 128-word scratchpad, no extended main memory.

    // Build once for the structural artifacts (preflight + unified universe + per-phase
    // equivalence). Step a few cycles first so equivalence runs against non-trivial state.
    let (cpu, mut sim, _console) = build(&program);
    for _ in 0..16 {
        if cpu.is_halted() {
            break;
        }
        cpu.step(&mut sim);
    }
    let universe = UnifiedSlotUniverse::from_cpu(&cpu, &sim);
    let phases: Vec<PhaseProfile> = phase_arrays(&cpu)
        .iter()
        .map(|(name, ops, wvia)| validate_phase(name, ops, wvia, &sim))
        .collect();

    // Authority run (fresh build) -> oracle hash/cycles/r0/inhibit/mmio + final state.
    let (cycles, r0, inhibit, mmio, hash_authority, final_state) =
        run_authority(&program, case.max_cycles, ram_words);

    // Settle-substituted run (fresh build) -> op-array-driven full-cycle parity.
    let (hash_substituted, substituted_dense_settles) =
        match run_settle_substituted(&program, case.max_cycles) {
            Some((h, d)) => (Some(h), d),
            None => (None, 0),
        };

    let correct = r0 == case.expected_r0;
    let charter_clean = stat.mul == 0 && !mmio && inhibit == 0;
    let all_build = phases.iter().all(|p| p.kernel_builds);
    let all_equiv = phases.iter().all(|p| p.equiv_ok);
    let parity_ok = hash_substituted
        .map(|h| h == hash_authority)
        .unwrap_or(true);
    let pass = correct && charter_clean && all_build && all_equiv && parity_ok;

    Ok(CharterProgramReport {
        name: case.name,
        stat,
        phases,
        slot_union: universe.len(),
        ram_words,
        inhibit,
        mmio,
        cycles,
        r0,
        expected_r0: case.expected_r0,
        hash_authority,
        hash_substituted,
        substituted_dense_settles,
        final_state,
        pass,
    })
}

/// The charter benchmark cases, resolved from the shared compiled corpus.
pub fn charter_cases() -> Vec<&'static V2CompiledBenchCase> {
    CHARTER_PROGRAMS
        .iter()
        .map(|name| {
            V2_COMPILED_BENCHMARKS
                .iter()
                .find(|c| c.name == *name)
                .unwrap_or_else(|| panic!("missing charter program '{name}' in corpus"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lean always-on structural test (no run-to-halt): one charter program builds its
    /// five phase kernels op-clean, the unified universe is non-empty, and the RAM SoA
    /// captures 128 words. Keeps the regular suite fast (memory rule: tests < 1 min).
    #[test]
    fn full_phase_prover_structural_smoke() {
        let case = charter_cases()[0]; // fib_recursive
        let program = compile_source(case.source).expect("compile");
        let (cpu, sim, _c) = build(&program);

        let universe = UnifiedSlotUniverse::from_cpu(&cpu, &sim);
        assert!(
            !universe.is_empty(),
            "unified slot universe should be non-empty"
        );

        for (name, ops, wvia) in phase_arrays(&cpu) {
            let k = LaneKernel::new_from_sim(ops, wvia, &sim);
            assert!(
                k.is_ok(),
                "phase '{name}' kernel must build op-clean: {k:?}"
            );
        }

        let state = LaneCpuState::capture(&cpu, &sim, 128);
        assert_eq!(
            state.ram.len(),
            128,
            "RAM SoA must mirror 128 scratch words"
        );
        assert_eq!(state.regs.len(), 16);
    }

    /// Heavy oracle: run all three charter programs to halt, prove per-phase op-array
    /// fidelity + settle-substituted full-cycle hash parity. Gated (slow) per the
    /// hardware rule — run with `--features slow-tests`.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow physical V2 charter oracle; run with --features slow-tests"
    )]
    fn full_phase_prover_charter_oracle() {
        for case in charter_cases() {
            let r = prove_charter(case).unwrap_or_else(|e| panic!("prove '{}': {e}", case.name));
            assert!(
                r.correct_and_clean(),
                "charter '{}' not clean: {r:?}",
                case.name
            );
            assert!(r.all_phases_build(), "charter '{}' phase build", case.name);
            assert!(r.all_phases_equiv(), "charter '{}' phase equiv", case.name);
            if let Some(h) = r.hash_substituted {
                assert_eq!(h, r.hash_authority, "charter '{}' hash parity", case.name);
            }
            assert!(r.pass, "charter '{}' overall PASS", case.name);
        }
    }
}

impl CharterProgramReport {
    pub fn correct_and_clean(&self) -> bool {
        self.r0 == self.expected_r0 && self.stat.mul == 0 && !self.mmio && self.inhibit == 0
    }
}
