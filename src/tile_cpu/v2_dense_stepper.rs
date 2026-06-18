//! Sprint 389a: one-lane dense pipeline stepper.
//!
//! This is the CPU-side prover for the Tier 3b device-resident cycle design.
//! It keeps the existing `TileCpuV2` stage logic as the authority, but replaces
//! the Stage-F combined settle with the same dense lane kernel used by the SIMT
//! fabric. The oracle is final architectural state parity with `TileCpuV2::step`.

use crate::simulation::{COP_GENERIC, COP_WIRE, CompactOp, Simulation};
use crate::tile_cpu::TileCpuV2;
use crate::tile_cpu::v2_simt_eval::LaneKernel;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePipelineUnsupportedOp {
    pub phase: &'static str,
    pub op_index: usize,
    pub tile_idx: u32,
    pub op: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePipelinePreflight {
    pub tile_count: usize,
    pub cone_ops: usize,
    pub settle_ops: usize,
    pub branch_ops: usize,
    pub commit_ops: usize,
    pub clock_ops: usize,
    pub unsupported_ops: Vec<DensePipelineUnsupportedOp>,
}

impl DensePipelinePreflight {
    pub fn is_gpu_op_clean(&self) -> bool {
        self.unsupported_ops.is_empty()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DensePipelineStepOutcome {
    pub halted_before: bool,
    pub halted_after: bool,
    pub dense_settle: bool,
    pub scalar_fallback_settle: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DensePipelineRunOutcome {
    pub cycles: u64,
    pub dense_settles: u64,
    pub scalar_fallback_settles: u64,
    pub halted: bool,
}

pub struct DensePipelineStepper {
    settle_kernel: LaneKernel,
    preflight: DensePipelinePreflight,
    dense_settles: u64,
    scalar_fallback_settles: u64,
}

impl DensePipelineStepper {
    pub fn new(cpu: &TileCpuV2, sim: &Simulation) -> Result<Self, String> {
        let (settle_ops, settle_wvia) = cpu.settle_compact_view();
        let settle_kernel = LaneKernel::new_from_sim(settle_ops, settle_wvia, sim)?;
        let preflight = DensePipelinePreflight::from_cpu(cpu, sim);
        Ok(Self {
            settle_kernel,
            preflight,
            dense_settles: 0,
            scalar_fallback_settles: 0,
        })
    }

    pub fn preflight(&self) -> &DensePipelinePreflight {
        &self.preflight
    }

    pub fn dense_settles(&self) -> u64 {
        self.dense_settles
    }

    pub fn scalar_fallback_settles(&self) -> u64 {
        self.scalar_fallback_settles
    }

    pub fn step(&mut self, cpu: &TileCpuV2, sim: &mut Simulation) -> DensePipelineStepOutcome {
        let halted_before = cpu.is_halted();
        let ctx = cpu.fabric_tick_pre_settle(sim);
        let mut dense_settle = false;
        let mut scalar_fallback_settle = false;

        if let Some(ctx) = ctx {
            if cpu.fabric_settle_inhibited() {
                self.scalar_fallback_settles += 1;
                scalar_fallback_settle = true;
                cpu.fabric_finish_settle(sim, &ctx, false);
            } else {
                let (vals, lanes) = sim.tilemap.lane_vals_raw_mut();
                assert_eq!(lanes, 1, "DensePipelineStepper is a one-lane CPU prover");
                self.settle_kernel.eval_slice(vals, lanes);
                self.dense_settles += 1;
                dense_settle = true;
                cpu.fabric_finish_settle(sim, &ctx, true);
            }
        }

        DensePipelineStepOutcome {
            halted_before,
            halted_after: cpu.is_halted(),
            dense_settle,
            scalar_fallback_settle,
        }
    }

    pub fn run_until_halt(
        &mut self,
        cpu: &TileCpuV2,
        sim: &mut Simulation,
        max_cycles: u64,
    ) -> DensePipelineRunOutcome {
        let dense_before = self.dense_settles;
        let fallback_before = self.scalar_fallback_settles;
        let mut cycles = 0u64;
        for _ in 0..max_cycles {
            if cpu.is_halted() {
                break;
            }
            self.step(cpu, sim);
            cycles += 1;
        }
        DensePipelineRunOutcome {
            cycles,
            dense_settles: self.dense_settles - dense_before,
            scalar_fallback_settles: self.scalar_fallback_settles - fallback_before,
            halted: cpu.is_halted(),
        }
    }
}

impl DensePipelinePreflight {
    fn from_cpu(cpu: &TileCpuV2, sim: &Simulation) -> Self {
        let mut unsupported_ops = Vec::new();
        record_unsupported("cone", &cpu.pipeline_cone_ops, &mut unsupported_ops);
        record_unsupported("settle", &cpu.settle_compact_ops, &mut unsupported_ops);
        record_unsupported("branch", &cpu.branch_compact_ops, &mut unsupported_ops);
        record_unsupported("commit", &cpu.commit_compact_ops, &mut unsupported_ops);
        record_unsupported("clock", &cpu.clock_compact_ops, &mut unsupported_ops);
        Self {
            tile_count: sim.tilemap.tile_count(),
            cone_ops: cpu.pipeline_cone_ops.len(),
            settle_ops: cpu.settle_compact_ops.len(),
            branch_ops: cpu.branch_compact_ops.len(),
            commit_ops: cpu.commit_compact_ops.len(),
            clock_ops: cpu.clock_compact_ops.len(),
            unsupported_ops,
        }
    }
}

fn record_unsupported(
    phase: &'static str,
    ops: &[CompactOp],
    unsupported: &mut Vec<DensePipelineUnsupportedOp>,
) {
    for (op_index, op) in ops.iter().enumerate() {
        if op.op == COP_GENERIC || op.op == COP_WIRE {
            unsupported.push(DensePipelineUnsupportedOp {
                phase,
                op_index,
                tile_idx: op.idx,
                op: op.op,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::Simulation;
    use crate::tile_cpu::v2_compiler_bench::{V2_COMPILED_BENCHMARKS, V2CompiledBenchCase};
    use crate::tile_cpu::v2_mmio::V2MmioHandle;
    use crate::tile_cpu::v2_mmio_devices::V2MmioCombinedDevice;
    use crate::tile_cpu::v2_parser::compile_source;
    use crate::tile_cpu::{V2Builder, V2SynthConfig, hash_v2_final_state};
    use std::rc::Rc;

    fn case_named(name: &str) -> &'static V2CompiledBenchCase {
        V2_COMPILED_BENCHMARKS
            .iter()
            .find(|case| case.name == name)
            .unwrap_or_else(|| panic!("missing compiled benchmark case '{name}'"))
    }

    fn build_case(case: &V2CompiledBenchCase) -> (TileCpuV2, Simulation) {
        let program = compile_source(case.source).unwrap_or_else(|e| {
            panic!("compile '{}' failed: {e}", case.name);
        });
        let console = Rc::new(V2MmioCombinedDevice::new(0x000B_E5C0));
        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_mmio(V2MmioHandle::from_rc(console))
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);
        (cpu, sim)
    }

    fn run_step_reference(cpu: &TileCpuV2, sim: &mut Simulation, max_cycles: u64) -> u64 {
        let mut cycles = 0u64;
        for _ in 0..max_cycles {
            if cpu.is_halted() {
                break;
            }
            cpu.step(sim);
            cycles += 1;
        }
        cycles
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow physical V2 389a corpus oracle; run with --features slow-tests"
    )]
    fn dense_pipeline_stepper_matches_tilecpu_step_on_389a_corpus() {
        for name in ["fact_iterative", "sum_loop", "gcd_subtract"] {
            let case = case_named(name);

            let (ref_cpu, mut ref_sim) = build_case(case);
            let ref_cycles = run_step_reference(&ref_cpu, &mut ref_sim, case.max_cycles);
            assert!(ref_cpu.is_halted(), "reference '{}' did not halt", name);
            assert_eq!(
                ref_cpu.read_reg(&ref_sim, 0),
                case.expected_r0,
                "reference '{}' R0",
                name
            );
            let ref_hash = hash_v2_final_state(&ref_cpu, &ref_sim);

            let (dense_cpu, mut dense_sim) = build_case(case);
            let mut stepper = DensePipelineStepper::new(&dense_cpu, &dense_sim)
                .unwrap_or_else(|e| panic!("dense stepper preflight '{}' failed: {e}", name));
            let preflight = stepper.preflight().clone();
            assert!(
                preflight.settle_ops > 0,
                "'{}' should expose a non-empty settle op array",
                name
            );

            let dense_run = stepper.run_until_halt(&dense_cpu, &mut dense_sim, case.max_cycles);
            assert!(dense_run.halted, "dense '{}' did not halt", name);
            assert_eq!(dense_run.cycles, ref_cycles, "'{}' cycle count", name);
            assert_eq!(
                dense_run.scalar_fallback_settles, 0,
                "'{}' should stay in the dense zero-inhibit regime",
                name
            );
            assert_eq!(
                dense_cpu.compact_eval_inhibit_count(),
                0,
                "'{}' should not trigger compact-eval inhibit",
                name
            );
            assert_eq!(
                dense_cpu.read_reg(&dense_sim, 0),
                case.expected_r0,
                "dense '{}' R0",
                name
            );
            let dense_hash = hash_v2_final_state(&dense_cpu, &dense_sim);
            assert_eq!(
                dense_hash, ref_hash,
                "'{}' DensePipelineStepper hash differs from TileCpuV2::step",
                name
            );
        }
    }
}
