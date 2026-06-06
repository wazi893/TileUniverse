//! Sprint 294: V2FastCpu — ISS-backed architectural execution backend.
//!
//! A standalone CPU executor that delegates to the V2 ISS (and optionally
//! the Cranelift JIT) for high-throughput instruction execution without
//! tile propagation.  Designed as a separate backend from TileCpuV2,
//! sharing the same architectural state contract (V2IssState) but owning
//! its own pipeline and cycle accounting.

use crate::tile_cpu::v2_iss::{V2Iss, V2IssState};

#[cfg(feature = "cranelift_jit")]
use crate::tile_cpu::v2_jit::{V2JitProgram, V2JitState, compile_v2_program};
#[cfg(feature = "cranelift_jit")]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Tick result
// ---------------------------------------------------------------------------

/// Minimal per-tick result from V2FastCpu (no tile metrics).
#[derive(Debug, Clone, Copy)]
pub struct V2FastTickResult {
    /// True if an instruction was retired this cycle.
    pub retired: bool,
    /// Cumulative cycle count after this tick.
    pub cycle_count: u64,
}

// ---------------------------------------------------------------------------
// V2FastCpu
// ---------------------------------------------------------------------------

/// High-throughput V2 CPU executor backed by the ISS / Cranelift JIT.
///
/// Models the same 2-stage pipeline as TileCpuV2 (one fill cycle before
/// first retirement) so cycle counts match physical execution exactly.
pub struct V2FastCpu {
    iss: V2Iss,
    program_words: Vec<u32>,

    // Pipeline state
    cycle_count: u64,
    retired_count: u64,
    last_stage_x_valid: bool,
    pipeline_filled: bool,
    halted: bool,

    // MMIO policy: set by constructor, not by static analysis.
    // Read by JIT path (cranelift_jit feature); suppress warning when feature is off.
    #[allow(dead_code)]
    uses_mmio: bool,

    // JIT cache (feature-gated)
    #[cfg(feature = "cranelift_jit")]
    jit_program: Option<Arc<V2JitProgram>>,
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl V2FastCpu {
    /// Create from a plain program (no MMIO). JIT eligible.
    pub fn from_program(program: &[u32]) -> Self {
        let iss = V2Iss::from_program(program);
        Self::new_inner(iss, program, false)
    }

    /// Create with standard MMIO device mirrors. JIT disabled.
    pub fn from_program_with_mmio(program: &[u32]) -> Self {
        let iss = V2Iss::from_program_with_mmio(program);
        Self::new_inner(iss, program, true)
    }

    /// Create with MMIO + dataset device. JIT disabled.
    pub fn from_program_with_dataset(program: &[u32], samples: Vec<(u64, u64)>) -> Self {
        let iss = V2Iss::from_program_with_dataset(program, samples);
        Self::new_inner(iss, program, true)
    }

    /// Create with MMIO + dataset + SNN bridge (M11). JIT disabled.
    pub fn from_program_with_snn_model(
        program: &[u32],
        samples: Vec<(u64, u64)>,
        weights: crate::snn::MlpWeights,
        cached_rates: crate::snn::CachedRates,
    ) -> Self {
        let iss = V2Iss::from_program_with_snn_model(program, samples, weights, cached_rates);
        Self::new_inner(iss, program, true)
    }

    /// Create with MMIO + dataset + live SNN model (M12). JIT disabled.
    pub fn from_program_with_live_snn_model(
        program: &[u32],
        samples: Vec<(u64, u64)>,
        live_model: crate::snn::mlp_weights::LiveSnnModel,
        images: Vec<Vec<u8>>,
        seed: u64,
    ) -> Self {
        let iss =
            V2Iss::from_program_with_live_snn_model(program, samples, live_model, images, seed);
        Self::new_inner(iss, program, true)
    }

    fn new_inner(iss: V2Iss, program: &[u32], uses_mmio: bool) -> Self {
        let program_words = program.to_vec();

        #[cfg(feature = "cranelift_jit")]
        let jit_program = if !uses_mmio {
            compile_v2_program(&program_words).ok()
        } else {
            None
        };

        Self {
            iss,
            program_words,
            cycle_count: 0,
            retired_count: 0,
            last_stage_x_valid: false,
            pipeline_filled: false,
            halted: false,
            uses_mmio,
            #[cfg(feature = "cranelift_jit")]
            jit_program,
        }
    }
}

// ---------------------------------------------------------------------------
// State migration: Physical ↔ Fast
// ---------------------------------------------------------------------------

impl V2FastCpu {
    /// Snapshot from a running TileCpuV2 into a V2FastCpu.
    ///
    /// Captures architectural state (regs, ram, pc, lr, flags, halted) plus
    /// pipeline state (cycle_count, retired_count, pipeline_filled).
    /// The program words must be supplied externally since TileCpuV2 does
    /// not store them.
    pub fn from_tile_cpu(
        cpu: &crate::tile_cpu::TileCpuV2,
        sim: &crate::simulation::Simulation,
        program: &[u32],
    ) -> Self {
        let mut fast = Self::from_program(program);
        // Overwrite ISS state from CPU mirrors
        {
            let s = fast.iss.state_mut();
            s.pc = cpu.read_pc(sim);
            s.lr = cpu.read_lr();
            s.flag_z = cpu.read_flag_z(sim);
            s.flag_c = cpu.read_flag_c(sim);
            s.halted = cpu.is_halted();
            for i in 0..16 {
                s.regs[i] = cpu.read_reg(sim, i);
            }
            for i in 0..128 {
                s.ram[i] = cpu.read_ram(sim, i);
            }
        }
        fast.cycle_count = cpu.read_cycle_count();
        fast.retired_count = cpu.read_retired_count();
        fast.halted = cpu.is_halted();
        fast.pipeline_filled = cpu.read_cycle_count() >= 1;
        fast.last_stage_x_valid = cpu.last_stage_x_valid();
        fast
    }

    /// Write V2FastCpu state into a TileCpuV2's tiles for visualization.
    pub fn sync_to_tile_cpu(
        &self,
        cpu: &crate::tile_cpu::TileCpuV2,
        sim: &mut crate::simulation::Simulation,
    ) {
        let s = self.iss.state();
        for i in 0..16 {
            cpu.write_reg(sim, i, s.regs[i]);
        }
        for i in 0..128 {
            cpu.write_ram(sim, i, s.ram[i]);
        }
        cpu.write_pc(sim, s.pc);
        cpu.write_lr(s.lr);
        cpu.write_flag_z(sim, s.flag_z);
        cpu.write_flag_c(sim, s.flag_c);
        cpu.set_halted(s.halted);
    }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

impl V2FastCpu {
    /// Execute one cycle, modeling the 2-stage pipeline.
    ///
    /// Cycle 1 is the pipeline fill (no retirement). Subsequent cycles each
    /// retire one instruction via `iss.step()`.
    pub fn tick(&mut self) -> V2FastTickResult {
        if self.halted {
            self.last_stage_x_valid = false;
            return V2FastTickResult {
                retired: false,
                cycle_count: self.cycle_count,
            };
        }

        self.cycle_count += 1;

        // Pipeline fill: first cycle produces no retirement.
        if !self.pipeline_filled {
            self.pipeline_filled = true;
            self.last_stage_x_valid = false;
            return V2FastTickResult {
                retired: false,
                cycle_count: self.cycle_count,
            };
        }

        // Step the ISS. Sprint 299: use step_no_mmio() for non-MMIO CPUs.
        let stepped = if self.uses_mmio {
            self.iss.step()
        } else {
            self.iss.step_no_mmio()
        };
        if stepped.is_some() {
            self.retired_count += 1;
            self.last_stage_x_valid = true;
        } else {
            // ISS was already halted before we called step — shouldn't happen
            // after the self.halted guard, but be defensive.
            self.last_stage_x_valid = false;
        }

        // Sync halted from ISS.
        if self.iss.state().halted {
            self.halted = true;
        }

        V2FastTickResult {
            retired: self.last_stage_x_valid,
            cycle_count: self.cycle_count,
        }
    }

    /// Run up to `max_cycles` in batch mode. Returns total cycles executed
    /// (including the pipeline fill cycle if it hadn't fired yet).
    ///
    /// Uses JIT when available (non-MMIO + cranelift_jit feature), otherwise
    /// falls back to a tight ISS loop.
    pub fn run_batch(&mut self, max_cycles: u64) -> u64 {
        if self.halted || max_cycles == 0 {
            return 0;
        }

        let mut cycles_used: u64 = 0;

        // Handle pipeline fill if needed.
        if !self.pipeline_filled {
            self.pipeline_filled = true;
            self.cycle_count += 1;
            self.last_stage_x_valid = false;
            cycles_used += 1;
            if cycles_used >= max_cycles {
                return cycles_used;
            }
        }

        let remaining = max_cycles - cycles_used;

        // Try JIT path.
        #[cfg(feature = "cranelift_jit")]
        if let Some(ref jit) = self.jit_program {
            let executed = self.run_jit_batch(jit.clone(), remaining);
            cycles_used += executed;
            return cycles_used;
        }

        // ISS fallback.
        let executed = self.run_iss_batch(remaining);
        cycles_used += executed;
        cycles_used
    }

    fn run_iss_batch(&mut self, max_instructions: u64) -> u64 {
        // Sprint 299: split MMIO/non-MMIO loops so the MMIO check is outside the tight loop.
        let count = if self.uses_mmio {
            let mut count: u64 = 0;
            for _ in 0..max_instructions {
                if self.iss.step().is_none() {
                    break;
                }
                count += 1;
                if self.iss.state().halted {
                    break;
                }
            }
            count
        } else {
            let mut count: u64 = 0;
            for _ in 0..max_instructions {
                if self.iss.step_no_mmio().is_none() {
                    break;
                }
                count += 1;
                if self.iss.state().halted {
                    break;
                }
            }
            count
        };
        self.retired_count += count;
        self.cycle_count += count;
        self.halted = self.iss.state().halted;
        self.last_stage_x_valid = count > 0;
        count
    }

    #[cfg(feature = "cranelift_jit")]
    fn run_jit_batch(&mut self, jit: Arc<V2JitProgram>, max_instructions: u64) -> u64 {
        let cap = max_instructions.min(u32::MAX as u64) as u32;
        let mut jit_state = V2JitState::from_iss_state(self.iss.state());
        let executed = unsafe { (jit.func_ptr)(&mut jit_state, cap) };
        let executed = executed as u64;

        // Transfer back.
        *self.iss.state_mut() = jit_state.to_iss_state();

        self.retired_count += executed;
        self.cycle_count += executed;
        self.halted = self.iss.state().halted;
        self.last_stage_x_valid = executed > 0;
        executed
    }
}

// ---------------------------------------------------------------------------
// State accessors
// ---------------------------------------------------------------------------

impl V2FastCpu {
    pub fn read_reg(&self, reg: usize) -> u64 {
        self.iss.state().regs[reg]
    }

    pub fn read_ram(&self, addr: usize) -> u64 {
        if addr < 128 {
            self.iss.state().ram[addr]
        } else {
            0
        }
    }

    pub fn read_pc(&self) -> u32 {
        self.iss.state().pc
    }

    pub fn read_lr(&self) -> u32 {
        self.iss.state().lr
    }

    pub fn read_flag_z(&self) -> bool {
        self.iss.state().flag_z
    }

    pub fn read_flag_c(&self) -> bool {
        self.iss.state().flag_c
    }

    pub fn is_halted(&self) -> bool {
        self.halted
    }

    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }

    pub fn retired_count(&self) -> u64 {
        self.retired_count
    }

    pub fn last_stage_x_valid(&self) -> bool {
        self.last_stage_x_valid
    }

    /// Direct access to the ISS architectural state.
    pub fn state(&self) -> &V2IssState {
        self.iss.state()
    }

    /// Mutable access to the ISS architectural state.
    pub fn state_mut(&mut self) -> &mut V2IssState {
        self.iss.state_mut()
    }

    /// The program words this CPU was constructed with.
    pub fn program_words(&self) -> &[u32] {
        &self.program_words
    }

    // ── Sprint 297: Mailbox accessors for fabric ──

    pub fn mailbox_in_send(&self) -> u64 {
        self.iss.mailbox_in_send()
    }
    pub fn mailbox_out_send(&self) -> u64 {
        self.iss.mailbox_out_send()
    }
    pub fn set_mailbox_in_recv(&mut self, v: u64) {
        self.iss.set_mailbox_in_recv(v);
    }
    pub fn set_mailbox_out_recv(&mut self, v: u64) {
        self.iss.set_mailbox_out_recv(v);
    }
    pub fn clear_mailbox_sends(&mut self) {
        self.iss.clear_mailbox_sends();
    }
}

// ---------------------------------------------------------------------------
// Hash — identical algorithm to hash_v2_final_state in v2_benchmarks.rs
// ---------------------------------------------------------------------------

/// Compute the same golden hash as `hash_v2_final_state` but from a V2IssState.
pub fn hash_v2_iss_state(s: &V2IssState) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    h = mix_hash(h, s.pc as u64);
    h = mix_hash(h, s.lr as u64);
    h = mix_hash(h, s.flag_z as u64);
    h = mix_hash(h, s.flag_c as u64);
    h = mix_hash(h, s.halted as u64);
    for reg in 0..16usize {
        h = mix_hash(h, s.regs[reg]);
    }
    for addr in 0..128usize {
        h = mix_hash(h, s.ram[addr]);
    }
    h
}

fn mix_hash(acc: u64, v: u64) -> u64 {
    let x = acc ^ v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x.rotate_left(27).wrapping_mul(0x94D0_49BB_1331_11EB)
}
