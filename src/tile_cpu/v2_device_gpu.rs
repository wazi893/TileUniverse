//! 1d — V2 → CUDA megakernel: **K bit-exact `TileCpuV2` instances advanced in parallel**,
//! one CUDA thread per lane, zero cross-lane sync.
//!
//! This is the GPU transliteration of the CPU-proven device cycle (`DeviceCpu` /
//! `DeviceCyclePlan` in `v2_device_cycle.rs`). The payoff is **batch / RL-rollout /
//! parameter-sweep throughput** (K independent CPUs at once), NOT single-lane speedup —
//! one lane on the GPU is slower than the CPU.
//!
//! ## Structure (built bottom-up, each step GPU-validated)
//! - **1d.1 `DeviceAluProgram` + `GpuDeviceAlu`**: the tile ALU. The three op arrays the
//!   ALU runs (`cone`, `settle`, `trunk_held`) are unioned into ONE shared dense slot
//!   universe so cone's outputs feed settle's inputs through the same slots. The ALU body
//!   (`device_alu_eval`): write PC → run cone → super-mux bridge → run settle → inject
//!   operands → run trunk_held → read the ALU root. Acceptance: `out[lane]` ==
//!   `standalone_alu`.
//! - **1d.2/1d.3 `GpuDeviceCpu`**: the scalar shell (`exec_one`, a verbatim port of
//!   `DeviceCpu::step`) + the per-thread run-to-halt loop (`device_run`). Per-lane arch
//!   state (pc/regs/flags/lr/ram/halted) lives in registers/SoA; the program (ROM) is
//!   shared. Acceptance: GPU final `DeviceCycleState` per lane == `DeviceCpu` for that
//!   lane's input, over the charter set.
//!
//! Constraints honored: the `COP_*` switch is **copied** from `v2_simt_gpu` (that file is
//! concurrent WIP — reused by API only, never edited); `v2_execute.rs` is untouched.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use logic_fabric_core::cuda::{CudaError, CudaResult, CudaRuntime};

use crate::simulation::{
    COP_ADD, COP_AND, COP_BITSEL, COP_CARRY, COP_CONST, COP_DEC3, COP_MUX, COP_MUX4, COP_MUX16,
    COP_NOT, COP_OR, COP_RAM, COP_SHL, COP_SHR, COP_SUB, COP_THRESHOLD_VIA, COP_VIA, COP_WIRE_D,
    COP_WIRE_H, COP_WIRE_L, COP_WIRE_R, COP_WIRE_U, COP_WIRE_V, COP_WVIA, COP_XOR, COP_ZERO,
    Simulation,
};
use crate::tile_cpu::v2_device_cycle::{DeviceCyclePlan, DeviceCycleState};

/// Keep only the elements of `v` whose aligned `keep[i]` is true.
fn filter_keep<T: Clone>(v: &[T], keep: &[bool]) -> Vec<T> {
    v.iter()
        .zip(keep.iter())
        .filter_map(|(x, &k)| k.then(|| x.clone()))
        .collect()
}

/// One backward-liveness step for dead-op elimination: if op `j`'s output slot is live,
/// keep it and mark its input slots live (COP_RAM also consumes its own output = `cur`).
#[allow(clippy::too_many_arguments)]
fn live_process(
    j: usize,
    live: &mut [bool],
    keep: &mut [bool],
    op_code: &[u32],
    op_out: &[u32],
    op_in0: &[u32],
    op_in1: &[u32],
    op_in2: &[u32],
    op_n0: &[u32],
    op_n1: &[u32],
    op_n2: &[u32],
    op_n3: &[u32],
) {
    let out = op_out[j] as usize;
    if !live[out] {
        return;
    }
    keep[j] = true;
    live[out] = false;
    for &inp in &[
        op_in0[j], op_in1[j], op_in2[j], op_n0[j], op_n1[j], op_n2[j], op_n3[j],
    ] {
        let s = inp as usize;
        if s < live.len() {
            live[s] = true;
        }
    }
    if op_code[j] == COP_RAM as u32 {
        live[out] = true; // COP_RAM consumes cur (= its own output slot)
    }
}

/// Get-or-insert a dense slot for a real tile index. `u32::MAX` is reserved by the
/// caller for the zero row (patched after all slots are assigned).
fn assign_slot(t: u32, slot_of: &mut HashMap<u32, u32>, touched: &mut Vec<u32>) -> u32 {
    if let Some(&s) = slot_of.get(&t) {
        return s;
    }
    let s = touched.len() as u32;
    slot_of.insert(t, s);
    touched.push(t);
    s
}

/// Host-side compacted device-ALU program: the `cone`, `settle`, and `trunk_held` op
/// arrays remapped into ONE shared dense slot universe (`vals[slot * K + lane]`), plus the
/// inject/readback slot indices the kernel writes/reads each cycle.
///
/// The op stream is concatenated: `[0, cone_end)` = cone, `[cone_end, settle_end)` =
/// settle, `[settle_end, n)` = trunk_held. CONST ops are dropped (the kernel skips them
/// at eval time, so injected Const tiles are auto-held); the non-const trunk terminals are
/// already dropped from `trunk_held` upstream (`DeviceCyclePlan::new`).
pub struct DeviceAluProgram {
    /// dense slot -> original tile index (the trailing zero slot is excluded).
    pub touched: Vec<u32>,
    pub n_slots: usize,
    op_code: Vec<u32>,
    op_out: Vec<u32>,
    op_in0: Vec<u32>,
    op_in1: Vec<u32>,
    op_in2: Vec<u32>,
    op_n0: Vec<u32>,
    op_n1: Vec<u32>,
    op_n2: Vec<u32>,
    op_n3: Vec<u32>,
    op_threshold: Vec<u32>,
    op_shift: Vec<u32>,
    op_mask: Vec<u64>,
    /// op-stream phase boundaries (cone = [0,cone_end), settle = [cone_end,settle_end),
    /// trunk_held = [settle_end, op_code.len())).
    cone_end: usize,
    settle_end: usize,
    // Inject / readback slots.
    pc_slot: u32,
    final_mux_slots: [u32; 4],
    super_mux_inject_right_slots: [u32; 4],
    alu_a_term_slot: u32,
    alu_b_term_slot: u32,
    wb_alu_root_slot: u32,
}

impl DeviceAluProgram {
    /// Build the shared-universe ALU program from a device-cycle plan (dead-op eliminated).
    pub fn from_plan(plan: &DeviceCyclePlan) -> Self {
        Self::from_plan_opt(plan, true)
    }

    /// Build with optional dead-op elimination (`prune`): keep only ops whose output
    /// transitively feeds `wb_alu_root` (the sole tile output the device reads) or
    /// `final_mux` (consumed by the mandatory super-mux bridge). Always bit-exact — a
    /// dropped op's output is never read by a kept op — and independent of lane lockstep.
    /// settle is ~75% of the raw stream and mostly decode logic dead to the ALU result.
    pub fn from_plan_opt(plan: &DeviceCyclePlan, prune: bool) -> Self {
        let (cone, settle, trunk) = plan.alu_kernels();
        let layout = plan.layout_indices();
        let zero_row = cone.tile_count() as u32;
        debug_assert_eq!(
            settle.tile_count() as u32,
            zero_row,
            "kernel tile_count mismatch"
        );
        debug_assert_eq!(
            trunk.tile_count() as u32,
            zero_row,
            "kernel tile_count mismatch"
        );

        let mut slot_of: HashMap<u32, u32> = HashMap::new();
        let mut touched: Vec<u32> = Vec::new();

        let mut op_code: Vec<u32> = Vec::new();
        let mut op_out: Vec<u32> = Vec::new();
        let mut op_in0: Vec<u32> = Vec::new();
        let mut op_in1: Vec<u32> = Vec::new();
        let mut op_in2: Vec<u32> = Vec::new();
        let mut op_n0: Vec<u32> = Vec::new();
        let mut op_n1: Vec<u32> = Vec::new();
        let mut op_n2: Vec<u32> = Vec::new();
        let mut op_n3: Vec<u32> = Vec::new();
        let mut op_threshold: Vec<u32> = Vec::new();
        let mut op_shift: Vec<u32> = Vec::new();
        let mut op_mask: Vec<u64> = Vec::new();

        // zero_row inputs are emitted as u32::MAX and patched to the zero slot afterward.
        let idx_of = |t: u32, slot_of: &mut HashMap<u32, u32>, touched: &mut Vec<u32>| -> u32 {
            if t == zero_row {
                u32::MAX
            } else {
                assign_slot(t, slot_of, touched)
            }
        };

        let mut boundaries = [0usize; 3];
        for (ki, kernel) in [cone, settle, trunk].iter().enumerate() {
            let (ops, wvia, tvia) = kernel.remapped_program();
            for ((op, &(wshift, wmask)), tv) in ops.iter().zip(wvia.iter()).zip(tvia.iter()) {
                if op.op == COP_CONST {
                    continue;
                }
                op_out.push(idx_of(op.idx, &mut slot_of, &mut touched));
                op_in0.push(idx_of(op.in0, &mut slot_of, &mut touched));
                op_in1.push(idx_of(op.in1, &mut slot_of, &mut touched));
                op_in2.push(idx_of(op.in2, &mut slot_of, &mut touched));
                if op.op == COP_THRESHOLD_VIA {
                    op_n0.push(idx_of(tv.neighbors[0], &mut slot_of, &mut touched));
                    op_n1.push(idx_of(tv.neighbors[1], &mut slot_of, &mut touched));
                    op_n2.push(idx_of(tv.neighbors[2], &mut slot_of, &mut touched));
                    op_n3.push(idx_of(tv.neighbors[3], &mut slot_of, &mut touched));
                    op_threshold.push(tv.threshold as u32);
                } else {
                    op_n0.push(u32::MAX);
                    op_n1.push(u32::MAX);
                    op_n2.push(u32::MAX);
                    op_n3.push(u32::MAX);
                    op_threshold.push(0);
                }
                op_code.push(op.op as u32);
                op_shift.push(wshift as u32);
                op_mask.push(wmask);
            }
            boundaries[ki] = op_code.len();
        }
        let raw_cone_end = boundaries[0];
        let raw_settle_end = boundaries[1];

        // Inject / readback tiles → ensure slots (idempotent: most already appear in the
        // op arrays as inputs/outputs; any that don't get fresh slots and get seeded).
        let mut pc_slot = assign_slot(layout.pc_idx as u32, &mut slot_of, &mut touched);
        let mut final_mux_slots = layout
            .final_mux_indices
            .map(|t| assign_slot(t as u32, &mut slot_of, &mut touched));
        let mut super_mux_inject_right_slots = layout
            .super_mux_inject_right_indices
            .map(|t| assign_slot(t as u32, &mut slot_of, &mut touched));
        let mut alu_a_term_slot = assign_slot(
            layout.alu_a_trunk_terminal_idx as u32,
            &mut slot_of,
            &mut touched,
        );
        let mut alu_b_term_slot = assign_slot(
            layout.alu_b_trunk_terminal_idx as u32,
            &mut slot_of,
            &mut touched,
        );
        let mut wb_alu_root_slot =
            assign_slot(layout.wb_alu_root_idx as u32, &mut slot_of, &mut touched);

        // Zero slot is the final slot; patch every u32::MAX input to it.
        let zero_slot = touched.len() as u32;
        let mut n_slots = touched.len() + 1;
        for v in op_out
            .iter_mut()
            .chain(op_in0.iter_mut())
            .chain(op_in1.iter_mut())
            .chain(op_in2.iter_mut())
            .chain(op_n0.iter_mut())
            .chain(op_n1.iter_mut())
            .chain(op_n2.iter_mut())
            .chain(op_n3.iter_mut())
        {
            if *v == u32::MAX {
                *v = zero_slot;
            }
        }

        // Dead-op elimination via backward liveness from `wb_alu_root` (the sole tile
        // output the device reads). The super-mux bridge and operand injections are NOT ops
        // — they write/read slots BETWEEN phases — so they must be modeled as boundary
        // actions at their real positions (seeding final_mux at the END instead is wrong if
        // a later op overwrites a final_mux slot). Always bit-exact (a dropped op's output
        // is never read before overwrite by a kept op) and lockstep-independent.
        let (cone_end, settle_end) = if prune {
            let n_ops = op_code.len();
            let mut live = vec![false; n_slots];
            let mut keep = vec![false; n_ops];
            live[wb_alu_root_slot as usize] = true;
            // trunk_held: reads the injected operands + the ALU op-select from settle.
            for j in (raw_settle_end..n_ops).rev() {
                live_process(
                    j, &mut live, &mut keep, &op_code, &op_out, &op_in0, &op_in1, &op_in2, &op_n0,
                    &op_n1, &op_n2, &op_n3,
                );
            }
            // Boundary settle→trunk: operands are INJECTED into the trunk terminals, so any
            // earlier op writing them is dead (the injection wins).
            live[alu_a_term_slot as usize] = false;
            live[alu_b_term_slot as usize] = false;
            // settle: decode → ALU op-select (most of it is dead to wb_alu_root).
            for j in (raw_cone_end..raw_settle_end).rev() {
                live_process(
                    j, &mut live, &mut keep, &op_code, &op_out, &op_in0, &op_in1, &op_in2, &op_n0,
                    &op_n1, &op_n2, &op_n3,
                );
            }
            // Boundary cone→settle: the bridge writes super_mux_inject_right from final_mux,
            // so smir is defined here (earlier writers dead) and final_mux is consumed here
            // (cone must produce it).
            for &s in &super_mux_inject_right_slots {
                live[s as usize] = false;
            }
            for &f in &final_mux_slots {
                live[f as usize] = true;
            }
            // cone: fetch + decode from ROM at PC.
            for j in (0..raw_cone_end).rev() {
                live_process(
                    j, &mut live, &mut keep, &op_code, &op_out, &op_in0, &op_in1, &op_in2, &op_n0,
                    &op_n1, &op_n2, &op_n3,
                );
            }
            let kept_cone = (0..raw_cone_end).filter(|&j| keep[j]).count();
            let kept_settle = (raw_cone_end..raw_settle_end).filter(|&j| keep[j]).count();
            op_code = filter_keep(&op_code, &keep);
            op_out = filter_keep(&op_out, &keep);
            op_in0 = filter_keep(&op_in0, &keep);
            op_in1 = filter_keep(&op_in1, &keep);
            op_in2 = filter_keep(&op_in2, &keep);
            op_n0 = filter_keep(&op_n0, &keep);
            op_n1 = filter_keep(&op_n1, &keep);
            op_n2 = filter_keep(&op_n2, &keep);
            op_n3 = filter_keep(&op_n3, &keep);
            op_threshold = filter_keep(&op_threshold, &keep);
            op_shift = filter_keep(&op_shift, &keep);
            op_mask = filter_keep(&op_mask, &keep);
            (kept_cone, kept_cone + kept_settle)
        } else {
            (raw_cone_end, raw_settle_end)
        };

        // Slot-pruning: after dead-op elimination most of the slot universe is untouched.
        // Compact to only the slots the surviving ops + inject/readback taps reference (a
        // consistent renumber → correctness-preserving), shrinking `vals` (memory, traffic,
        // and the max K that fits). The zero slot stays last.
        if prune {
            let mut used = vec![false; n_slots];
            for v in op_out
                .iter()
                .chain(op_in0.iter())
                .chain(op_in1.iter())
                .chain(op_in2.iter())
                .chain(op_n0.iter())
                .chain(op_n1.iter())
                .chain(op_n2.iter())
                .chain(op_n3.iter())
            {
                used[*v as usize] = true;
            }
            for &s in &[pc_slot, alu_a_term_slot, alu_b_term_slot, wb_alu_root_slot] {
                used[s as usize] = true;
            }
            for &s in final_mux_slots
                .iter()
                .chain(super_mux_inject_right_slots.iter())
            {
                used[s as usize] = true;
            }
            let old_zero = (n_slots - 1) as u32;
            used[old_zero as usize] = true;

            let mut new_of = vec![u32::MAX; n_slots];
            let mut new_touched = Vec::new();
            for old in 0..(n_slots - 1) {
                if used[old] {
                    new_of[old] = new_touched.len() as u32;
                    new_touched.push(touched[old]);
                }
            }
            let new_zero = new_touched.len() as u32;
            new_of[old_zero as usize] = new_zero;

            for v in op_out
                .iter_mut()
                .chain(op_in0.iter_mut())
                .chain(op_in1.iter_mut())
                .chain(op_in2.iter_mut())
                .chain(op_n0.iter_mut())
                .chain(op_n1.iter_mut())
                .chain(op_n2.iter_mut())
                .chain(op_n3.iter_mut())
            {
                *v = new_of[*v as usize];
            }
            pc_slot = new_of[pc_slot as usize];
            alu_a_term_slot = new_of[alu_a_term_slot as usize];
            alu_b_term_slot = new_of[alu_b_term_slot as usize];
            wb_alu_root_slot = new_of[wb_alu_root_slot as usize];
            for s in final_mux_slots.iter_mut() {
                *s = new_of[*s as usize];
            }
            for s in super_mux_inject_right_slots.iter_mut() {
                *s = new_of[*s as usize];
            }
            touched = new_touched;
            n_slots = new_zero as usize + 1;
        }

        Self {
            touched,
            n_slots,
            op_code,
            op_out,
            op_in0,
            op_in1,
            op_in2,
            op_n0,
            op_n1,
            op_n2,
            op_n3,
            op_threshold,
            op_shift,
            op_mask,
            cone_end,
            settle_end,
            pc_slot,
            final_mux_slots,
            super_mux_inject_right_slots,
            alu_a_term_slot,
            alu_b_term_slot,
            wb_alu_root_slot,
        }
    }

    pub fn op_count(&self) -> usize {
        self.op_code.len()
    }

    /// `(cone, settle, trunk_held)` op counts — the per-ALU-instruction op-eval breakdown.
    pub fn phase_op_counts(&self) -> (usize, usize, usize) {
        (
            self.cone_end,
            self.settle_end - self.cone_end,
            self.op_code.len() - self.settle_end,
        )
    }

    /// Estimated logical `vals` traffic per op-eval after opcode-specific operand loads.
    /// Metadata traffic is reported separately by profilers; this is the benchmark-facing
    /// value-buffer traffic model (operand reads + the always-present output store).
    pub fn estimated_value_bytes_per_eval(&self) -> f64 {
        fn value_reads(code: u32) -> usize {
            match code as u8 {
                COP_WIRE_R | COP_VIA | COP_WIRE_L | COP_WIRE_D | COP_WIRE_U | COP_NOT
                | COP_ZERO | COP_DEC3 | COP_WVIA => 1,
                COP_WIRE_H | COP_OR | COP_WIRE_V | COP_AND | COP_XOR | COP_ADD | COP_SUB
                | COP_SHR | COP_SHL | COP_BITSEL | COP_CARRY | COP_MUX4 => 2,
                COP_MUX | COP_MUX16 | COP_RAM => 3,
                COP_THRESHOLD_VIA => 5,
                _ => 1,
            }
        }
        let bytes: usize = self
            .op_code
            .iter()
            .map(|&code| (value_reads(code) + 1) * std::mem::size_of::<u64>())
            .sum();
        bytes as f64 / self.op_code.len() as f64
    }

    /// `(pre_pack, packed)` estimated logical metadata traffic per op-eval for the lazy
    /// operand-loader. The packed format stores `code/out/in0/in1/in2` in one u64; `extra`
    /// is read only for WVIA/THRESHOLD_VIA, and `mask` only for WVIA.
    pub fn estimated_metadata_bytes_per_eval(&self) -> (f64, f64) {
        fn old_bytes(code: u32) -> usize {
            let base = 8; // op_code + op_out
            match code as u8 {
                COP_WIRE_R | COP_VIA | COP_WIRE_L | COP_WIRE_D | COP_WIRE_U | COP_NOT
                | COP_ZERO | COP_DEC3 => base + 4,
                COP_WIRE_H | COP_OR | COP_WIRE_V | COP_AND | COP_XOR | COP_ADD | COP_SUB
                | COP_SHR | COP_SHL | COP_BITSEL | COP_CARRY | COP_MUX4 => base + 8,
                COP_MUX | COP_MUX16 | COP_RAM => base + 12,
                COP_WVIA => base + 4 + 4 + 8,
                COP_THRESHOLD_VIA => base + 4 + 16 + 4,
                _ => base,
            }
        }
        fn packed_bytes(code: u32) -> usize {
            match code as u8 {
                COP_WVIA => 24,          // meta + extra(shift) + mask
                COP_THRESHOLD_VIA => 16, // meta + extra(neighbors/threshold)
                _ => 8,
            }
        }
        let old: usize = self.op_code.iter().map(|&code| old_bytes(code)).sum();
        let packed: usize = self.op_code.iter().map(|&code| packed_bytes(code)).sum();
        (
            old as f64 / self.op_code.len() as f64,
            packed as f64 / self.op_code.len() as f64,
        )
    }

    fn packed_metadata(&self) -> (Vec<u64>, Vec<u64>) {
        const SLOT_BITS: u32 = 14;
        const SLOT_LIMIT: u32 = 1 << SLOT_BITS;

        let slot = |name: &str, v: u32| -> u64 {
            assert!(
                v < SLOT_LIMIT,
                "{name} slot {v} exceeds {SLOT_BITS}-bit packed metadata limit"
            );
            v as u64
        };

        let mut meta = Vec::with_capacity(self.op_code.len());
        let mut extra = Vec::with_capacity(self.op_code.len());
        for j in 0..self.op_code.len() {
            assert!(self.op_code[j] <= u8::MAX as u32, "op code does not fit u8");
            let m = (self.op_code[j] as u64)
                | (slot("out", self.op_out[j]) << 8)
                | (slot("in0", self.op_in0[j]) << 22)
                | (slot("in1", self.op_in1[j]) << 36)
                | (slot("in2", self.op_in2[j]) << 50);
            let e = if self.op_code[j] == COP_THRESHOLD_VIA as u32 {
                assert!(
                    self.op_threshold[j] <= u8::MAX as u32,
                    "threshold does not fit u8"
                );
                slot("n0", self.op_n0[j])
                    | (slot("n1", self.op_n1[j]) << 14)
                    | (slot("n2", self.op_n2[j]) << 28)
                    | (slot("n3", self.op_n3[j]) << 42)
                    | ((self.op_threshold[j] as u64) << 56)
            } else {
                assert!(self.op_shift[j] <= u8::MAX as u32, "shift does not fit u8");
                self.op_shift[j] as u64
            };
            meta.push(m);
            extra.push(e);
        }
        (meta, extra)
    }

    /// Host-side lane buffer seeded from a sim's current values, replicated into every
    /// lane: `vals[slot * lanes + lane]` (the trailing zero slot stays 0).
    pub fn seed_from_sim(&self, sim: &Simulation, lanes: usize) -> Vec<u64> {
        let mut vals = vec![0u64; self.n_slots * lanes];
        for (s, &tile) in self.touched.iter().enumerate() {
            let v = sim.tilemap.value(tile as usize);
            for lane in 0..lanes {
                vals[s * lanes + lane] = v;
            }
        }
        vals
    }
}

/// The `run_range` device function: one op-array range, per lane. The `COP_*` switch is
/// COPIED from `v2_simt_gpu` so CPU and GPU evaluate identical op semantics; the constants
/// are formatted in from the crate so no drift is possible. (Only this part needs the
/// `format!` placeholder escaping; everything else is the literal `DEVICE_REST`.)
fn run_range_src() -> String {
    format!(
        r#"
__device__ __forceinline__ void run_range_packed(
    unsigned long long* __restrict__ vals,
    const unsigned long long* __restrict__ op_meta,
    const unsigned long long* __restrict__ op_extra,
    const unsigned long long* __restrict__ op_mask,
    int lo, int hi, long long lane, int lanes)
{{
    const unsigned long long SLOT_MASK = 0x3FFFULL;
    for (int j = lo; j < hi; ++j) {{
        unsigned long long meta = op_meta[j];
        unsigned int code = (unsigned int)(meta & 0xFFULL);
        unsigned int out = (unsigned int)((meta >> 8) & SLOT_MASK);
        unsigned int s0 = (unsigned int)((meta >> 22) & SLOT_MASK);
        size_t o = (size_t)out * lanes + lane;
        unsigned long long r;
        switch (code) {{
            case {wire_r}: case {via}:
                r = vals[(size_t)s0 * lanes + lane];
                break;
            case {wire_l}:
                r = vals[(size_t)((meta >> 36) & SLOT_MASK) * lanes + lane];
                break;
            case {wire_d}: case {wire_u}:
                r = vals[(size_t)((meta >> 50) & SLOT_MASK) * lanes + lane];
                break;
            case {wire_h}: case {cop_or}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long v1 = vals[(size_t)((meta >> 36) & SLOT_MASK) * lanes + lane];
                r = v0 | v1;
                break;
            }}
            case {wire_v}: {{
                unsigned long long v1 = vals[(size_t)((meta >> 36) & SLOT_MASK) * lanes + lane];
                unsigned long long v2 = vals[(size_t)((meta >> 50) & SLOT_MASK) * lanes + lane];
                r = v1 | v2;
                break;
            }}
            case {cop_and}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long v1 = vals[(size_t)((meta >> 36) & SLOT_MASK) * lanes + lane];
                r = v0 & v1;
                break;
            }}
            case {cop_xor}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long v1 = vals[(size_t)((meta >> 36) & SLOT_MASK) * lanes + lane];
                r = v0 ^ v1;
                break;
            }}
            case {cop_mux}: {{
                unsigned long long v2 = vals[(size_t)((meta >> 50) & SLOT_MASK) * lanes + lane];
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long v1 = vals[(size_t)((meta >> 36) & SLOT_MASK) * lanes + lane];
                r = (v2 != 0ULL) ? v0 : v1;
                break;
            }}
            case {cop_not}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                r = ~v0;
                break;
            }}
            case {cop_zero}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                r = (v0 == 0ULL) ? ~0ULL : 0ULL;
                break;
            }}
            case {cop_add}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long v1 = vals[(size_t)((meta >> 36) & SLOT_MASK) * lanes + lane];
                r = v0 + v1;
                break;
            }}
            case {cop_sub}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long v1 = vals[(size_t)((meta >> 36) & SLOT_MASK) * lanes + lane];
                r = v0 - v1;
                break;
            }}
            case {cop_shr}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long v1 = vals[(size_t)((meta >> 36) & SLOT_MASK) * lanes + lane];
                r = v0 >> (v1 & 63ULL);
                break;
            }}
            case {cop_shl}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long v1 = vals[(size_t)((meta >> 36) & SLOT_MASK) * lanes + lane];
                r = v0 << (v1 & 63ULL);
                break;
            }}
            case {cop_mux16}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long v1 = vals[(size_t)((meta >> 36) & SLOT_MASK) * lanes + lane];
                unsigned long long v2 = vals[(size_t)((meta >> 50) & SLOT_MASK) * lanes + lane];
                unsigned int sel = (unsigned int)(v1 & 0xFULL);
                unsigned long long data = (sel < 8u) ? v0 : v2;
                r = (data >> ((sel & 7u) * 8u)) & 0xFFULL;
                break;
            }}
            case {cop_dec3}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                r = 1ULL << (v0 & 7ULL);
                break;
            }}
            case {cop_bitsel}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long v1 = vals[(size_t)((meta >> 36) & SLOT_MASK) * lanes + lane];
                r = ((v0 >> (v1 & 63ULL)) & 1ULL) ? ~0ULL : 0ULL;
                break;
            }}
            case {cop_carry}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long v1 = vals[(size_t)((meta >> 36) & SLOT_MASK) * lanes + lane];
                r = (v0 > v1) ? ~0ULL : 0ULL;
                break;
            }}
            case {cop_wvia}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned int shift = (unsigned int)(op_extra[j] & 0xFFULL);
                r = (v0 >> shift) & op_mask[j];
                break;
            }}
            case {cop_threshold_via}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long ex = op_extra[j];
                unsigned int active =
                    (vals[(size_t)((ex >> 0) & SLOT_MASK) * lanes + lane] != 0ULL) +
                    (vals[(size_t)((ex >> 14) & SLOT_MASK) * lanes + lane] != 0ULL) +
                    (vals[(size_t)((ex >> 28) & SLOT_MASK) * lanes + lane] != 0ULL) +
                    (vals[(size_t)((ex >> 42) & SLOT_MASK) * lanes + lane] != 0ULL);
                r = (active >= (unsigned int)(ex >> 56)) ? v0 : 0ULL;
                break;
            }}
            case {cop_mux4}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long v2 = vals[(size_t)((meta >> 50) & SLOT_MASK) * lanes + lane];
                unsigned int sel = (unsigned int)(v2 & 3ULL);
                r = (v0 >> (sel * 8u)) & 0xFFULL;
                break;
            }}
            case {cop_ram}: {{
                unsigned long long v0 = vals[(size_t)s0 * lanes + lane];
                unsigned long long v2 = vals[(size_t)((meta >> 50) & SLOT_MASK) * lanes + lane];
                unsigned long long cur = vals[o];
                r = (v2 != 0ULL) ? v0 : cur;
                break;
            }}
            default:
                r = vals[o];
                break;
        }}
        vals[o] = r;
    }}
}}

__device__ __forceinline__ void run_range_split(
    unsigned long long* __restrict__ vals,
    const unsigned int* __restrict__ op_code,
    const unsigned int* __restrict__ op_out,
    const unsigned int* __restrict__ op_in0,
    const unsigned int* __restrict__ op_in1,
    const unsigned int* __restrict__ op_in2,
    const unsigned int* __restrict__ op_n0,
    const unsigned int* __restrict__ op_n1,
    const unsigned int* __restrict__ op_n2,
    const unsigned int* __restrict__ op_n3,
    const unsigned int* __restrict__ op_threshold,
    const unsigned int* __restrict__ op_shift,
    const unsigned long long* __restrict__ op_mask,
    int lo, int hi, long long lane, int lanes)
{{
    for (int j = lo; j < hi; ++j) {{
        unsigned int code = op_code[j];
        size_t o = (size_t)op_out[j] * lanes + lane;
        unsigned long long r;
        switch (code) {{
            case {wire_r}: case {via}:
                r = vals[(size_t)op_in0[j] * lanes + lane];
                break;
            case {wire_l}:
                r = vals[(size_t)op_in1[j] * lanes + lane];
                break;
            case {wire_d}: case {wire_u}:
                r = vals[(size_t)op_in2[j] * lanes + lane];
                break;
            case {wire_h}: case {cop_or}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned long long v1 = vals[(size_t)op_in1[j] * lanes + lane];
                r = v0 | v1;
                break;
            }}
            case {wire_v}: {{
                unsigned long long v1 = vals[(size_t)op_in1[j] * lanes + lane];
                unsigned long long v2 = vals[(size_t)op_in2[j] * lanes + lane];
                r = v1 | v2;
                break;
            }}
            case {cop_and}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned long long v1 = vals[(size_t)op_in1[j] * lanes + lane];
                r = v0 & v1;
                break;
            }}
            case {cop_xor}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned long long v1 = vals[(size_t)op_in1[j] * lanes + lane];
                r = v0 ^ v1;
                break;
            }}
            case {cop_mux}: {{
                unsigned long long v2 = vals[(size_t)op_in2[j] * lanes + lane];
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned long long v1 = vals[(size_t)op_in1[j] * lanes + lane];
                r = (v2 != 0ULL) ? v0 : v1;
                break;
            }}
            case {cop_not}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                r = ~v0;
                break;
            }}
            case {cop_zero}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                r = (v0 == 0ULL) ? ~0ULL : 0ULL;
                break;
            }}
            case {cop_add}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned long long v1 = vals[(size_t)op_in1[j] * lanes + lane];
                r = v0 + v1;
                break;
            }}
            case {cop_sub}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned long long v1 = vals[(size_t)op_in1[j] * lanes + lane];
                r = v0 - v1;
                break;
            }}
            case {cop_shr}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned long long v1 = vals[(size_t)op_in1[j] * lanes + lane];
                r = v0 >> (v1 & 63ULL);
                break;
            }}
            case {cop_shl}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned long long v1 = vals[(size_t)op_in1[j] * lanes + lane];
                r = v0 << (v1 & 63ULL);
                break;
            }}
            case {cop_mux16}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned long long v1 = vals[(size_t)op_in1[j] * lanes + lane];
                unsigned long long v2 = vals[(size_t)op_in2[j] * lanes + lane];
                unsigned int sel = (unsigned int)(v1 & 0xFULL);
                unsigned long long data = (sel < 8u) ? v0 : v2;
                r = (data >> ((sel & 7u) * 8u)) & 0xFFULL;
                break;
            }}
            case {cop_dec3}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                r = 1ULL << (v0 & 7ULL);
                break;
            }}
            case {cop_bitsel}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned long long v1 = vals[(size_t)op_in1[j] * lanes + lane];
                r = ((v0 >> (v1 & 63ULL)) & 1ULL) ? ~0ULL : 0ULL;
                break;
            }}
            case {cop_carry}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned long long v1 = vals[(size_t)op_in1[j] * lanes + lane];
                r = (v0 > v1) ? ~0ULL : 0ULL;
                break;
            }}
            case {cop_wvia}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                r = (v0 >> op_shift[j]) & op_mask[j];
                break;
            }}
            case {cop_threshold_via}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned int active =
                    (vals[(size_t)op_n0[j] * lanes + lane] != 0ULL) +
                    (vals[(size_t)op_n1[j] * lanes + lane] != 0ULL) +
                    (vals[(size_t)op_n2[j] * lanes + lane] != 0ULL) +
                    (vals[(size_t)op_n3[j] * lanes + lane] != 0ULL);
                r = (active >= op_threshold[j]) ? v0 : 0ULL;
                break;
            }}
            case {cop_mux4}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned long long v2 = vals[(size_t)op_in2[j] * lanes + lane];
                unsigned int sel = (unsigned int)(v2 & 3ULL);
                r = (v0 >> (sel * 8u)) & 0xFFULL;
                break;
            }}
            case {cop_ram}: {{
                unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
                unsigned long long v2 = vals[(size_t)op_in2[j] * lanes + lane];
                unsigned long long cur = vals[o];
                r = (v2 != 0ULL) ? v0 : cur;
                break;
            }}
            default:
                r = vals[o];
                break;
        }}
        vals[o] = r;
    }}
}}
"#,
        wire_r = COP_WIRE_R,
        via = COP_VIA,
        wire_l = COP_WIRE_L,
        wire_d = COP_WIRE_D,
        wire_u = COP_WIRE_U,
        wire_h = COP_WIRE_H,
        cop_or = COP_OR,
        wire_v = COP_WIRE_V,
        cop_and = COP_AND,
        cop_xor = COP_XOR,
        cop_mux = COP_MUX,
        cop_not = COP_NOT,
        cop_zero = COP_ZERO,
        cop_add = COP_ADD,
        cop_sub = COP_SUB,
        cop_shr = COP_SHR,
        cop_shl = COP_SHL,
        cop_mux16 = COP_MUX16,
        cop_dec3 = COP_DEC3,
        cop_bitsel = COP_BITSEL,
        cop_carry = COP_CARRY,
        cop_wvia = COP_WVIA,
        cop_threshold_via = COP_THRESHOLD_VIA,
        cop_mux4 = COP_MUX4,
        cop_ram = COP_RAM,
    )
}

/// The rest of the kernel (no `format!` placeholders — literal C). Defines the shared
/// ALU context, the ALU body (`device_alu_eval`), the scalar-shell helpers + `exec_one`
/// (a verbatim port of `DeviceCpu::step`, charter scope), and the three entry kernels.
const DEVICE_REST: &str = r#"
// Shared op-array contexts (the inject/readback slots + the concatenated op stream).
struct AluCtxPacked {
    unsigned long long* vals;
    const unsigned long long* op_meta;
    const unsigned long long* op_extra;
    const unsigned long long* op_mask;
    int cone_end; int settle_end; int n_total;
    int pc_slot;
    int fm0; int fm1; int fm2; int fm3;
    int smir0; int smir1; int smir2; int smir3;
    int a_term; int b_term; int wb_root;
    int lanes;
};

struct AluCtxSplit {
    unsigned long long* vals;
    const unsigned int* op_code;
    const unsigned int* op_out;
    const unsigned int* op_in0;
    const unsigned int* op_in1;
    const unsigned int* op_in2;
    const unsigned int* op_n0;
    const unsigned int* op_n1;
    const unsigned int* op_n2;
    const unsigned int* op_n3;
    const unsigned int* op_threshold;
    const unsigned int* op_shift;
    const unsigned long long* op_mask;
    int cone_end; int settle_end; int n_total;
    int pc_slot;
    int fm0; int fm1; int fm2; int fm3;
    int smir0; int smir1; int smir2; int smir3;
    int a_term; int b_term; int wb_root;
    int lanes;
};

// One device-ALU evaluation per lane: PC -> cone -> super-mux bridge -> settle ->
// operand inject -> trunk_held -> read wb_alu_root. Mirrors DeviceCpu::alu.
__device__ unsigned long long device_alu_eval(
    const AluCtxPacked* c, unsigned long long pc, unsigned long long a, unsigned long long b, long long lane)
{
    int lanes = c->lanes;
    c->vals[(size_t)c->pc_slot * lanes + lane] = pc;
    run_range_packed(c->vals, c->op_meta, c->op_extra, c->op_mask,
                     0, c->cone_end, lane, lanes);
    // Super-mux bridge: final_mux -> super_mux_inject_right (low-bank, pc < 64).
    c->vals[(size_t)c->smir0 * lanes + lane] = c->vals[(size_t)c->fm0 * lanes + lane];
    c->vals[(size_t)c->smir1 * lanes + lane] = c->vals[(size_t)c->fm1 * lanes + lane];
    c->vals[(size_t)c->smir2 * lanes + lane] = c->vals[(size_t)c->fm2 * lanes + lane];
    c->vals[(size_t)c->smir3 * lanes + lane] = c->vals[(size_t)c->fm3 * lanes + lane];
    run_range_packed(c->vals, c->op_meta, c->op_extra, c->op_mask,
                     c->cone_end, c->settle_end, lane, lanes);
    c->vals[(size_t)c->a_term * lanes + lane] = a;
    c->vals[(size_t)c->b_term * lanes + lane] = b;
    run_range_packed(c->vals, c->op_meta, c->op_extra, c->op_mask,
                     c->settle_end, c->n_total, lane, lanes);
    return c->vals[(size_t)c->wb_root * lanes + lane];
}

__device__ unsigned long long device_alu_eval(
    const AluCtxSplit* c, unsigned long long pc, unsigned long long a, unsigned long long b, long long lane)
{
    int lanes = c->lanes;
    c->vals[(size_t)c->pc_slot * lanes + lane] = pc;
    run_range_split(c->vals, c->op_code, c->op_out, c->op_in0, c->op_in1, c->op_in2,
                    c->op_n0, c->op_n1, c->op_n2, c->op_n3, c->op_threshold,
                    c->op_shift, c->op_mask, 0, c->cone_end, lane, lanes);
    // Super-mux bridge: final_mux -> super_mux_inject_right (low-bank, pc < 64).
    c->vals[(size_t)c->smir0 * lanes + lane] = c->vals[(size_t)c->fm0 * lanes + lane];
    c->vals[(size_t)c->smir1 * lanes + lane] = c->vals[(size_t)c->fm1 * lanes + lane];
    c->vals[(size_t)c->smir2 * lanes + lane] = c->vals[(size_t)c->fm2 * lanes + lane];
    c->vals[(size_t)c->smir3 * lanes + lane] = c->vals[(size_t)c->fm3 * lanes + lane];
    run_range_split(c->vals, c->op_code, c->op_out, c->op_in0, c->op_in1, c->op_in2,
                    c->op_n0, c->op_n1, c->op_n2, c->op_n3, c->op_threshold,
                    c->op_shift, c->op_mask, c->cone_end, c->settle_end, lane, lanes);
    c->vals[(size_t)c->a_term * lanes + lane] = a;
    c->vals[(size_t)c->b_term * lanes + lane] = b;
    run_range_split(c->vals, c->op_code, c->op_out, c->op_in0, c->op_in1, c->op_in2,
                    c->op_n0, c->op_n1, c->op_n2, c->op_n3, c->op_threshold,
                    c->op_shift, c->op_mask, c->settle_end, c->n_total, lane, lanes);
    return c->vals[(size_t)c->wb_root * lanes + lane];
}

// opcode -> branch_kind (== ctrl_b & 0x07). JMP=1,JZ=2,JNZ=3,JC=4,JNC=5,CALL=6,RET=7.
__device__ unsigned int branch_kind(unsigned int opcode) {
    switch (opcode) {
        case 0x1A: return 1;
        case 0x1B: return 2;
        case 0x1C: return 3;
        case 0x1D: return 4;
        case 0x1E: return 5;
        case 0x1F: return 6;
        case 0x15: return 7;
        default: return 0;
    }
}

// Pure next-PC resolution from branch_kind + flags (replica of compute_branch_pc).
__device__ unsigned int compute_branch_pc(
    unsigned int bk, unsigned int z, unsigned int c,
    unsigned int target, unsigned int fall, unsigned int lr)
{
    switch (bk) {
        case 1: case 6: return target;
        case 2: return z ? target : fall;
        case 3: return (!z) ? target : fall;
        case 4: return c ? target : fall;
        case 5: return (!c) ? target : fall;
        case 7: return lr & 0x7F;
        default: return fall;
    }
}

// Per-opcode carry (device_carry): add-like a>result, sub-like result>a, bitwise unchanged,
// SHL/SHR + non-ALU -> no update (*has=0).
__device__ void device_carry(
    unsigned int opcode, unsigned long long a, unsigned long long result,
    unsigned int carry_before, int* has, unsigned int* val)
{
    switch (opcode) {
        case 0x04: case 0x0B: case 0x10: *has = 1; *val = (a > result) ? 1u : 0u; break;
        case 0x05: case 0x0A: case 0x0C: case 0x0F: case 0x11:
            *has = 1; *val = (result > a) ? 1u : 0u; break;
        case 0x06: case 0x07: case 0x08: case 0x09: case 0x12: case 0x13: case 0x14:
            *has = 1; *val = carry_before; break;
        default: *has = 0; *val = 0; break;
    }
}

// Execute ONE instruction for this lane, mutating local arch state + global ram/vals.
// Verbatim port of DeviceCpu::step (charter scope). regs[16] is local; ram is global SoA.
template <typename AluCtxT>
__device__ void exec_one(
    const AluCtxT* c,
    const unsigned int* program, int prog_len,
    unsigned int* pc, unsigned long long* regs,
    unsigned int* flag_z, unsigned int* flag_c, unsigned int* lr,
    unsigned long long* ram, unsigned int* halted, unsigned long long* retired,
    long long lane)
{
    if (*halted) return;
    int lanes = c->lanes;
    unsigned int p = *pc;
    unsigned int word = (p < (unsigned int)prog_len) ? program[p] : 0u;

    unsigned int opcode = (word >> 11) & 0x1F;
    unsigned int rd_lo = (word >> 8) & 0x07;
    unsigned int rs_lo = (word >> 5) & 0x07;
    unsigned int imm5 = word & 0x1F;
    unsigned int imm8 = word & 0xFF;
    unsigned int ir_ext = (word >> 16) & 0xFFFF;
    unsigned int rd_hi = (ir_ext & 0x0001u) != 0;
    unsigned int rs_hi = (ir_ext & 0x0002u) != 0;
    unsigned int wide_imm = (ir_ext & 0x0004u) != 0;
    unsigned int has_offset = (ir_ext & 0x0008u) != 0;
    unsigned int reg_indirect = (ir_ext & 0x0010u) != 0;
    unsigned int rd = (rd_lo & 7u) | (rd_hi ? 8u : 0u);
    unsigned int rs = (rs_lo & 7u) | (rs_hi ? 8u : 0u);
    unsigned int offset = has_offset ? ((ir_ext >> 8) & 0xFF) : 0u;
    unsigned int imm16 = wide_imm ? ((((ir_ext >> 8) & 0xFF) << 8) | imm8) : imm8;

    unsigned int fz0 = *flag_z;
    unsigned int fc0 = *flag_c;
    unsigned long long a = regs[rd];
    unsigned long long b = regs[rs];

    // GUARDED (charter-absent, modeled wrong): MUL (0x04 imm5==1), cmov (0x02 sub-fn 1..4).
    // Halt the lane so a stray one surfaces as a divergence, never silently-wrong state.
    if ((opcode == 0x04 && imm5 == 1) || (opcode == 0x02 && imm5 >= 1 && imm5 <= 4)) {
        *halted = 1;
        *retired += 1;
        return;
    }

    int has_wb = 0; unsigned int wb_r = 0; unsigned long long wb_v = 0;
    int has_z = 0; unsigned int z_v = 0;
    int has_c = 0; unsigned int c_v = 0;

    if (opcode == 0x00) {
        // NOP
    } else if (opcode == 0x01) {
        *halted = 1;                                  // HALT
    } else if (opcode == 0x02) {
        wb_v = (imm5 == 5) ? (unsigned long long)(*lr) : b;   // MOV / MFLR
        has_wb = 1; wb_r = rd;
    } else if (opcode == 0x03) {
        wb_v = (unsigned long long)imm16;             // LDI (writes Z; C unchanged)
        has_wb = 1; wb_r = rd;
        has_z = 1; z_v = (wb_v == 0) ? 1u : 0u;
    } else if (opcode >= 0x04 && opcode <= 0x0E) {
        unsigned long long r = device_alu_eval(c, (unsigned long long)p, a, b, lane);  // ALU reg
        has_wb = 1; wb_r = rd; wb_v = r;
        has_z = 1; z_v = (r == 0) ? 1u : 0u;
        device_carry(opcode, a, r, fc0, &has_c, &c_v);
    } else if (opcode == 0x0F) {
        unsigned long long r = device_alu_eval(c, (unsigned long long)p, a, b, lane);  // CMP (flags only)
        has_z = 1; z_v = (r == 0) ? 1u : 0u;
        device_carry(0x0F, a, r, fc0, &has_c, &c_v);
    } else if (opcode >= 0x10 && opcode <= 0x14) {
        unsigned long long r = device_alu_eval(c, (unsigned long long)p, a, b, lane);  // ALU imm
        has_wb = 1; wb_r = rd; wb_v = r;
        has_z = 1; z_v = (r == 0) ? 1u : 0u;
        device_carry(opcode, a, r, fc0, &has_c, &c_v);
    } else if (opcode == 0x16 || opcode == 0x18) {
        unsigned int byte_variant = (opcode == 0x18);                // LD / LDB
        unsigned int addr;
        if (has_offset) addr = (unsigned int)((b + offset) & 0x7F);
        else if (byte_variant) addr = imm8 & 0x7F;
        else addr = (unsigned int)(b & 0x7F);
        unsigned long long v = ram[(size_t)addr * lanes + lane];
        has_wb = 1; wb_r = rd; wb_v = v;
        has_z = 1; z_v = (v == 0) ? 1u : 0u;
    } else if (opcode == 0x17 || opcode == 0x19) {
        unsigned int byte_variant = (opcode == 0x19);                // ST / STB (store rd)
        unsigned int addr;
        if (has_offset) addr = (unsigned int)((b + offset) & 0x7F);
        else if (byte_variant) addr = imm8 & 0x7F;
        else addr = (unsigned int)(b & 0x7F);
        ram[(size_t)addr * lanes + lane] = a;
    } else {
        // 0x15 RET, 0x1A..0x1F branches: control-only (resolved in next-PC below).
    }

    if (has_wb) regs[wb_r] = wb_v;
    if (has_z) *flag_z = z_v;
    if (has_c) *flag_c = c_v;

    if (!*halted) {
        unsigned int bk = branch_kind(opcode);
        unsigned int PCM = 0xFF;
        unsigned int pc_full = p & PCM;
        unsigned int fall = (pc_full + 1) & PCM;
        unsigned int next;
        if (reg_indirect && (bk == 1 || bk == 6)) {
            next = (unsigned int)(regs[rd] & PCM);                    // JMPR / CALLR
        } else if (bk == 7) {
            next = (*lr) & PCM;                                       // RET
        } else {
            next = compute_branch_pc(bk, fz0, fc0, imm16 & PCM, fall, *lr);
        }
        *pc = next;
        if (bk == 6) {
            *lr = ((p & 0xFF) + 1) & 0xFF;                            // CALL/CALLR set LR
        }
    }
    *retired += 1;
}

// 1d.1 entry: one ALU evaluation per lane (per-lane pc/a/b -> out).
extern "C" __global__ void device_alu_packed(
    unsigned long long* __restrict__ vals,
    const unsigned long long* __restrict__ op_meta,
    const unsigned long long* __restrict__ op_extra,
    const unsigned long long* __restrict__ op_mask,
    int cone_end, int settle_end, int n_total,
    int pc_slot,
    int fm0, int fm1, int fm2, int fm3,
    int smir0, int smir1, int smir2, int smir3,
    int a_term, int b_term, int wb_root,
    const unsigned long long* __restrict__ pc_in,
    const unsigned long long* __restrict__ a_in,
    const unsigned long long* __restrict__ b_in,
    unsigned long long* __restrict__ out,
    int lanes)
{
    long long lane = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    AluCtxPacked c;
    c.vals = vals;
    c.op_meta = op_meta; c.op_extra = op_extra;
    c.op_mask = op_mask;
    c.cone_end = cone_end; c.settle_end = settle_end; c.n_total = n_total;
    c.pc_slot = pc_slot; c.fm0 = fm0; c.fm1 = fm1; c.fm2 = fm2; c.fm3 = fm3;
    c.smir0 = smir0; c.smir1 = smir1; c.smir2 = smir2; c.smir3 = smir3;
    c.a_term = a_term; c.b_term = b_term; c.wb_root = wb_root; c.lanes = lanes;
    out[lane] = device_alu_eval(&c, pc_in[lane], a_in[lane], b_in[lane], lane);
}

extern "C" __global__ void device_alu_split(
    unsigned long long* __restrict__ vals,
    const unsigned int* __restrict__ op_code,
    const unsigned int* __restrict__ op_out,
    const unsigned int* __restrict__ op_in0,
    const unsigned int* __restrict__ op_in1,
    const unsigned int* __restrict__ op_in2,
    const unsigned int* __restrict__ op_n0,
    const unsigned int* __restrict__ op_n1,
    const unsigned int* __restrict__ op_n2,
    const unsigned int* __restrict__ op_n3,
    const unsigned int* __restrict__ op_threshold,
    const unsigned int* __restrict__ op_shift,
    const unsigned long long* __restrict__ op_mask,
    int cone_end, int settle_end, int n_total,
    int pc_slot,
    int fm0, int fm1, int fm2, int fm3,
    int smir0, int smir1, int smir2, int smir3,
    int a_term, int b_term, int wb_root,
    const unsigned long long* __restrict__ pc_in,
    const unsigned long long* __restrict__ a_in,
    const unsigned long long* __restrict__ b_in,
    unsigned long long* __restrict__ out,
    int lanes)
{
    long long lane = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    AluCtxSplit c;
    c.vals = vals;
    c.op_code = op_code; c.op_out = op_out;
    c.op_in0 = op_in0; c.op_in1 = op_in1; c.op_in2 = op_in2;
    c.op_n0 = op_n0; c.op_n1 = op_n1; c.op_n2 = op_n2; c.op_n3 = op_n3;
    c.op_threshold = op_threshold; c.op_shift = op_shift; c.op_mask = op_mask;
    c.cone_end = cone_end; c.settle_end = settle_end; c.n_total = n_total;
    c.pc_slot = pc_slot; c.fm0 = fm0; c.fm1 = fm1; c.fm2 = fm2; c.fm3 = fm3;
    c.smir0 = smir0; c.smir1 = smir1; c.smir2 = smir2; c.smir3 = smir3;
    c.a_term = a_term; c.b_term = b_term; c.wb_root = wb_root; c.lanes = lanes;
    out[lane] = device_alu_eval(&c, pc_in[lane], a_in[lane], b_in[lane], lane);
}

// 1d.2/1d.3 entry: advance each lane up to max_instrs (or until halt). max_instrs==1 is a
// single step. No host interaction inside the loop -> this is the megakernel.
extern "C" __global__ void device_run_packed(
    unsigned long long* __restrict__ vals,
    const unsigned long long* __restrict__ op_meta,
    const unsigned long long* __restrict__ op_extra,
    const unsigned long long* __restrict__ op_mask,
    int cone_end, int settle_end, int n_total,
    int pc_slot,
    int fm0, int fm1, int fm2, int fm3,
    int smir0, int smir1, int smir2, int smir3,
    int a_term, int b_term, int wb_root,
    const unsigned int* __restrict__ program, int prog_len,
    unsigned int* __restrict__ st_pc,
    unsigned long long* __restrict__ st_regs,
    unsigned int* __restrict__ st_fz,
    unsigned int* __restrict__ st_fc,
    unsigned int* __restrict__ st_lr,
    unsigned long long* __restrict__ st_ram,
    unsigned int* __restrict__ st_halted,
    unsigned long long* __restrict__ st_retired,
    long long max_instrs, int lanes)
{
    long long lane = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    AluCtxPacked c;
    c.vals = vals;
    c.op_meta = op_meta; c.op_extra = op_extra;
    c.op_mask = op_mask;
    c.cone_end = cone_end; c.settle_end = settle_end; c.n_total = n_total;
    c.pc_slot = pc_slot; c.fm0 = fm0; c.fm1 = fm1; c.fm2 = fm2; c.fm3 = fm3;
    c.smir0 = smir0; c.smir1 = smir1; c.smir2 = smir2; c.smir3 = smir3;
    c.a_term = a_term; c.b_term = b_term; c.wb_root = wb_root; c.lanes = lanes;

    unsigned int pc = st_pc[lane];
    unsigned long long regs[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) regs[r] = st_regs[(size_t)r * lanes + lane];
    unsigned int fz = st_fz[lane];
    unsigned int fc = st_fc[lane];
    unsigned int lr = st_lr[lane];
    unsigned int halted = st_halted[lane];
    unsigned long long retired = st_retired[lane];

    for (long long i = 0; i < max_instrs; ++i) {
        if (halted) break;
        exec_one(&c, program, prog_len, &pc, regs, &fz, &fc, &lr,
                 st_ram, &halted, &retired, lane);
    }

    st_pc[lane] = pc;
    #pragma unroll
    for (int r = 0; r < 16; ++r) st_regs[(size_t)r * lanes + lane] = regs[r];
    st_fz[lane] = fz;
    st_fc[lane] = fc;
    st_lr[lane] = lr;
    st_halted[lane] = halted;
    st_retired[lane] = retired;
}

extern "C" __global__ void device_run_split(
    unsigned long long* __restrict__ vals,
    const unsigned int* __restrict__ op_code,
    const unsigned int* __restrict__ op_out,
    const unsigned int* __restrict__ op_in0,
    const unsigned int* __restrict__ op_in1,
    const unsigned int* __restrict__ op_in2,
    const unsigned int* __restrict__ op_n0,
    const unsigned int* __restrict__ op_n1,
    const unsigned int* __restrict__ op_n2,
    const unsigned int* __restrict__ op_n3,
    const unsigned int* __restrict__ op_threshold,
    const unsigned int* __restrict__ op_shift,
    const unsigned long long* __restrict__ op_mask,
    int cone_end, int settle_end, int n_total,
    int pc_slot,
    int fm0, int fm1, int fm2, int fm3,
    int smir0, int smir1, int smir2, int smir3,
    int a_term, int b_term, int wb_root,
    const unsigned int* __restrict__ program, int prog_len,
    unsigned int* __restrict__ st_pc,
    unsigned long long* __restrict__ st_regs,
    unsigned int* __restrict__ st_fz,
    unsigned int* __restrict__ st_fc,
    unsigned int* __restrict__ st_lr,
    unsigned long long* __restrict__ st_ram,
    unsigned int* __restrict__ st_halted,
    unsigned long long* __restrict__ st_retired,
    long long max_instrs, int lanes)
{
    long long lane = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    AluCtxSplit c;
    c.vals = vals;
    c.op_code = op_code; c.op_out = op_out;
    c.op_in0 = op_in0; c.op_in1 = op_in1; c.op_in2 = op_in2;
    c.op_n0 = op_n0; c.op_n1 = op_n1; c.op_n2 = op_n2; c.op_n3 = op_n3;
    c.op_threshold = op_threshold; c.op_shift = op_shift; c.op_mask = op_mask;
    c.cone_end = cone_end; c.settle_end = settle_end; c.n_total = n_total;
    c.pc_slot = pc_slot; c.fm0 = fm0; c.fm1 = fm1; c.fm2 = fm2; c.fm3 = fm3;
    c.smir0 = smir0; c.smir1 = smir1; c.smir2 = smir2; c.smir3 = smir3;
    c.a_term = a_term; c.b_term = b_term; c.wb_root = wb_root; c.lanes = lanes;

    unsigned int pc = st_pc[lane];
    unsigned long long regs[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) regs[r] = st_regs[(size_t)r * lanes + lane];
    unsigned int fz = st_fz[lane];
    unsigned int fc = st_fc[lane];
    unsigned int lr = st_lr[lane];
    unsigned int halted = st_halted[lane];
    unsigned long long retired = st_retired[lane];

    for (long long i = 0; i < max_instrs; ++i) {
        if (halted) break;
        exec_one(&c, program, prog_len, &pc, regs, &fz, &fc, &lr,
                 st_ram, &halted, &retired, lane);
    }

    st_pc[lane] = pc;
    #pragma unroll
    for (int r = 0; r < 16; ++r) st_regs[(size_t)r * lanes + lane] = regs[r];
    st_fz[lane] = fz;
    st_fc[lane] = fc;
    st_lr[lane] = lr;
    st_halted[lane] = halted;
    st_retired[lane] = retired;
}
"#;

/// The full megakernel source (run_range + the rest).
fn megakernel_source() -> String {
    format!("{}\n{}", run_range_src(), DEVICE_REST)
}

/// Compile the megakernel module (compute_89 PTX forward-JITs onto every supported device,
/// incl. Blackwell/sm_120) and load the named entry point.
fn load_kernel(rt: &CudaRuntime, name: &str) -> CudaResult<cudarc::driver::CudaFunction> {
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    let opts = CompileOptions {
        arch: Some("compute_89"),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(&megakernel_source(), opts)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("megakernel: {e:?}")))?;
    let module = rt
        .ctx()
        .load_module(ptx)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("module: {e:?}")))?;
    module
        .load_function(name)
        .map_err(|e| CudaError::KernelCompilationFailed(format!("function {name}: {e:?}")))
}

/// Uploaded op-program buffers + their scalar slot indices — shared by `GpuDeviceAlu` and
/// `GpuDeviceCpu`. (`d_vals` is allocated/seeded by the caller per use.)
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GpuMetadataMode {
    Packed,
    Split,
}

impl GpuMetadataMode {
    fn for_lanes(lanes: usize) -> Self {
        // Empirical 5090 result: packed wins small/mid batches and the largest sampled K,
        // while split-array lazy metadata preserves the current saturation headline.
        if (32_768..=98_304).contains(&lanes) {
            Self::Split
        } else {
            Self::Packed
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Packed => "packed",
            Self::Split => "split",
        }
    }

    fn alu_kernel_name(self) -> &'static str {
        match self {
            Self::Packed => "device_alu_packed",
            Self::Split => "device_alu_split",
        }
    }

    fn run_kernel_name(self) -> &'static str {
        match self {
            Self::Packed => "device_run_packed",
            Self::Split => "device_run_split",
        }
    }
}

enum GpuOpStorage {
    Packed {
        d_meta: cudarc::driver::CudaSlice<u64>,
        d_extra: cudarc::driver::CudaSlice<u64>,
        d_mask: cudarc::driver::CudaSlice<u64>,
    },
    Split {
        d_code: cudarc::driver::CudaSlice<u32>,
        d_out: cudarc::driver::CudaSlice<u32>,
        d_in0: cudarc::driver::CudaSlice<u32>,
        d_in1: cudarc::driver::CudaSlice<u32>,
        d_in2: cudarc::driver::CudaSlice<u32>,
        d_n0: cudarc::driver::CudaSlice<u32>,
        d_n1: cudarc::driver::CudaSlice<u32>,
        d_n2: cudarc::driver::CudaSlice<u32>,
        d_n3: cudarc::driver::CudaSlice<u32>,
        d_threshold: cudarc::driver::CudaSlice<u32>,
        d_shift: cudarc::driver::CudaSlice<u32>,
        d_mask: cudarc::driver::CudaSlice<u64>,
    },
}

struct GpuOpBuffers {
    storage: GpuOpStorage,
    cone_end: i32,
    settle_end: i32,
    n_total: i32,
    pc_slot: i32,
    fm: [i32; 4],
    smir: [i32; 4],
    a_term: i32,
    b_term: i32,
    wb_root: i32,
}

impl GpuOpBuffers {
    fn upload(
        rt: &CudaRuntime,
        program: &DeviceAluProgram,
        mode: GpuMetadataMode,
    ) -> CudaResult<Self> {
        let storage = match mode {
            GpuMetadataMode::Packed => {
                let (op_meta, op_extra) = program.packed_metadata();
                GpuOpStorage::Packed {
                    d_meta: rt.upload(&op_meta)?,
                    d_extra: rt.upload(&op_extra)?,
                    d_mask: rt.upload(&program.op_mask)?,
                }
            }
            GpuMetadataMode::Split => GpuOpStorage::Split {
                d_code: rt.upload(&program.op_code)?,
                d_out: rt.upload(&program.op_out)?,
                d_in0: rt.upload(&program.op_in0)?,
                d_in1: rt.upload(&program.op_in1)?,
                d_in2: rt.upload(&program.op_in2)?,
                d_n0: rt.upload(&program.op_n0)?,
                d_n1: rt.upload(&program.op_n1)?,
                d_n2: rt.upload(&program.op_n2)?,
                d_n3: rt.upload(&program.op_n3)?,
                d_threshold: rt.upload(&program.op_threshold)?,
                d_shift: rt.upload(&program.op_shift)?,
                d_mask: rt.upload(&program.op_mask)?,
            },
        };
        Ok(Self {
            storage,
            cone_end: program.cone_end as i32,
            settle_end: program.settle_end as i32,
            n_total: program.op_code.len() as i32,
            pc_slot: program.pc_slot as i32,
            fm: program.final_mux_slots.map(|s| s as i32),
            smir: program.super_mux_inject_right_slots.map(|s| s as i32),
            a_term: program.alu_a_term_slot as i32,
            b_term: program.alu_b_term_slot as i32,
            wb_root: program.wb_alu_root_slot as i32,
        })
    }
}

/// Device-side compiled device-ALU (1d.1): uploaded program + lane value buffer.
pub struct GpuDeviceAlu {
    func: cudarc::driver::CudaFunction,
    ops: GpuOpBuffers,
    d_vals: cudarc::driver::CudaSlice<u64>,
    lanes: usize,
}

impl GpuDeviceAlu {
    pub fn new(
        rt: &CudaRuntime,
        program: &DeviceAluProgram,
        host_vals: &[u64],
        lanes: usize,
    ) -> CudaResult<Self> {
        Self::new_with_metadata_mode(
            rt,
            program,
            host_vals,
            lanes,
            GpuMetadataMode::for_lanes(lanes),
        )
    }

    fn new_with_metadata_mode(
        rt: &CudaRuntime,
        program: &DeviceAluProgram,
        host_vals: &[u64],
        lanes: usize,
        mode: GpuMetadataMode,
    ) -> CudaResult<Self> {
        assert_eq!(
            host_vals.len(),
            program.n_slots * lanes,
            "host vals size mismatch"
        );
        Ok(Self {
            func: load_kernel(rt, mode.alu_kernel_name())?,
            ops: GpuOpBuffers::upload(rt, program, mode)?,
            d_vals: rt.upload(host_vals)?,
            lanes,
        })
    }

    /// Inject per-lane `(pc, a, b)`, launch, and download the per-lane ALU result.
    pub fn run(
        &mut self,
        rt: &CudaRuntime,
        pc: &[u64],
        a: &[u64],
        b: &[u64],
    ) -> CudaResult<Vec<u64>> {
        assert_eq!(pc.len(), self.lanes, "pc len mismatch");
        assert_eq!(a.len(), self.lanes, "a len mismatch");
        assert_eq!(b.len(), self.lanes, "b len mismatch");
        use cudarc::driver::PushKernelArg;
        let d_pc = rt.upload(pc)?;
        let d_a = rt.upload(a)?;
        let d_b = rt.upload(b)?;
        let d_result: cudarc::driver::CudaSlice<u64> = rt.alloc_zeros(self.lanes)?;

        let threads = 256u32;
        let blocks = (self.lanes as u32).div_ceil(threads);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let lanes = self.lanes as i32;
        let o = &self.ops;
        let (fm0, fm1, fm2, fm3) = (o.fm[0], o.fm[1], o.fm[2], o.fm[3]);
        let (smir0, smir1, smir2, smir3) = (o.smir[0], o.smir[1], o.smir[2], o.smir[3]);
        unsafe {
            match &o.storage {
                GpuOpStorage::Packed {
                    d_meta,
                    d_extra,
                    d_mask,
                } => {
                    rt.stream()
                        .launch_builder(&self.func)
                        .arg(&self.d_vals)
                        .arg(d_meta)
                        .arg(d_extra)
                        .arg(d_mask)
                        .arg(&o.cone_end)
                        .arg(&o.settle_end)
                        .arg(&o.n_total)
                        .arg(&o.pc_slot)
                        .arg(&fm0)
                        .arg(&fm1)
                        .arg(&fm2)
                        .arg(&fm3)
                        .arg(&smir0)
                        .arg(&smir1)
                        .arg(&smir2)
                        .arg(&smir3)
                        .arg(&o.a_term)
                        .arg(&o.b_term)
                        .arg(&o.wb_root)
                        .arg(&d_pc)
                        .arg(&d_a)
                        .arg(&d_b)
                        .arg(&d_result)
                        .arg(&lanes)
                        .launch(cfg)
                        .map_err(|e| CudaError::LaunchFailed(format!("device_alu: {e:?}")))?;
                }
                GpuOpStorage::Split {
                    d_code,
                    d_out,
                    d_in0,
                    d_in1,
                    d_in2,
                    d_n0,
                    d_n1,
                    d_n2,
                    d_n3,
                    d_threshold,
                    d_shift,
                    d_mask,
                } => {
                    rt.stream()
                        .launch_builder(&self.func)
                        .arg(&self.d_vals)
                        .arg(d_code)
                        .arg(d_out)
                        .arg(d_in0)
                        .arg(d_in1)
                        .arg(d_in2)
                        .arg(d_n0)
                        .arg(d_n1)
                        .arg(d_n2)
                        .arg(d_n3)
                        .arg(d_threshold)
                        .arg(d_shift)
                        .arg(d_mask)
                        .arg(&o.cone_end)
                        .arg(&o.settle_end)
                        .arg(&o.n_total)
                        .arg(&o.pc_slot)
                        .arg(&fm0)
                        .arg(&fm1)
                        .arg(&fm2)
                        .arg(&fm3)
                        .arg(&smir0)
                        .arg(&smir1)
                        .arg(&smir2)
                        .arg(&smir3)
                        .arg(&o.a_term)
                        .arg(&o.b_term)
                        .arg(&o.wb_root)
                        .arg(&d_pc)
                        .arg(&d_a)
                        .arg(&d_b)
                        .arg(&d_result)
                        .arg(&lanes)
                        .launch(cfg)
                        .map_err(|e| CudaError::LaunchFailed(format!("device_alu: {e:?}")))?;
                }
            }
        }
        rt.stream()
            .synchronize()
            .map_err(|e| CudaError::LaunchFailed(format!("sync: {e:?}")))?;
        rt.download(&d_result)
    }
}

/// Device-side full V2 CPU (1d.2/1d.3): the unified op program + ROM, advancing K lanes of
/// per-lane architectural state to halt with no host interaction in the loop.
pub struct GpuDeviceCpu {
    func: cudarc::driver::CudaFunction,
    ops: GpuOpBuffers,
    d_program: cudarc::driver::CudaSlice<u32>,
    prog_len: i32,
    /// Per-lane vals seed (ROM + scratch), replicated; re-uploaded fresh each run.
    host_vals: Vec<u64>,
    ram_words: usize,
    lanes: usize,
}

impl GpuDeviceCpu {
    pub fn new(
        rt: &CudaRuntime,
        program: &DeviceAluProgram,
        sim: &Simulation,
        rom_words: &[u32],
        ram_words: usize,
        lanes: usize,
    ) -> CudaResult<Self> {
        Self::new_with_metadata_mode(
            rt,
            program,
            sim,
            rom_words,
            ram_words,
            lanes,
            GpuMetadataMode::for_lanes(lanes),
        )
    }

    fn new_with_metadata_mode(
        rt: &CudaRuntime,
        program: &DeviceAluProgram,
        sim: &Simulation,
        rom_words: &[u32],
        ram_words: usize,
        lanes: usize,
        mode: GpuMetadataMode,
    ) -> CudaResult<Self> {
        Ok(Self {
            func: load_kernel(rt, mode.run_kernel_name())?,
            ops: GpuOpBuffers::upload(rt, program, mode)?,
            d_program: rt.upload(rom_words)?,
            prog_len: rom_words.len() as i32,
            host_vals: program.seed_from_sim(sim, lanes),
            ram_words,
            lanes,
        })
    }

    /// Advance every lane from its initial `DeviceCycleState` by up to `max_instrs` (or
    /// until halt), returning each lane's final `DeviceCycleState`. `max_instrs == 1` is a
    /// single step (== one `DeviceCpu::step`); a large value runs to halt (the megakernel).
    pub fn run(
        &mut self,
        rt: &CudaRuntime,
        init: &[DeviceCycleState],
        max_instrs: u64,
    ) -> CudaResult<Vec<DeviceCycleState>> {
        self.run_timed(rt, init, max_instrs).map(|(s, _)| s)
    }

    /// Like `run`, but also returns the kernel-only elapsed seconds (launch→sync, excluding
    /// H2D/D2H transfers) — for throughput benchmarking.
    pub fn run_timed(
        &mut self,
        rt: &CudaRuntime,
        init: &[DeviceCycleState],
        max_instrs: u64,
    ) -> CudaResult<(Vec<DeviceCycleState>, f64)> {
        assert_eq!(init.len(), self.lanes, "init states != lanes");
        let lanes = self.lanes;
        let rw = self.ram_words;
        for s in init {
            assert_eq!(s.ram.len(), rw, "init ram_words mismatch");
        }

        // Pack arch state into SoA host buffers (field[i * lanes + lane]).
        let mut h_pc = vec![0u32; lanes];
        let mut h_regs = vec![0u64; 16 * lanes];
        let mut h_fz = vec![0u32; lanes];
        let mut h_fc = vec![0u32; lanes];
        let mut h_lr = vec![0u32; lanes];
        let mut h_ram = vec![0u64; rw * lanes];
        let mut h_halted = vec![0u32; lanes];
        let mut h_retired = vec![0u64; lanes];
        for (lane, s) in init.iter().enumerate() {
            h_pc[lane] = s.pc;
            for r in 0..16 {
                h_regs[r * lanes + lane] = s.regs[r];
            }
            h_fz[lane] = s.flag_z as u32;
            h_fc[lane] = s.flag_c as u32;
            h_lr[lane] = s.lr;
            for a in 0..rw {
                h_ram[a * lanes + lane] = s.ram[a];
            }
            h_halted[lane] = s.halted as u32;
            h_retired[lane] = s.retired;
        }

        use cudarc::driver::PushKernelArg;
        // Fresh vals (ROM + scratch) for a clean run, mirroring DeviceCpu's per-instance buf.
        let d_vals = rt.upload(&self.host_vals)?;
        let mut d_pc = rt.upload(&h_pc)?;
        let mut d_regs = rt.upload(&h_regs)?;
        let mut d_fz = rt.upload(&h_fz)?;
        let mut d_fc = rt.upload(&h_fc)?;
        let mut d_lr = rt.upload(&h_lr)?;
        let mut d_ram = rt.upload(&h_ram)?;
        let mut d_halted = rt.upload(&h_halted)?;
        let mut d_retired = rt.upload(&h_retired)?;

        let threads = 256u32;
        let blocks = (lanes as u32).div_ceil(threads);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let lanes_i = lanes as i32;
        let max_i = max_instrs as i64;
        let o = &self.ops;
        let (fm0, fm1, fm2, fm3) = (o.fm[0], o.fm[1], o.fm[2], o.fm[3]);
        let (smir0, smir1, smir2, smir3) = (o.smir[0], o.smir[1], o.smir[2], o.smir[3]);
        let t0 = std::time::Instant::now();
        unsafe {
            match &o.storage {
                GpuOpStorage::Packed {
                    d_meta,
                    d_extra,
                    d_mask,
                } => {
                    rt.stream()
                        .launch_builder(&self.func)
                        .arg(&d_vals)
                        .arg(d_meta)
                        .arg(d_extra)
                        .arg(d_mask)
                        .arg(&o.cone_end)
                        .arg(&o.settle_end)
                        .arg(&o.n_total)
                        .arg(&o.pc_slot)
                        .arg(&fm0)
                        .arg(&fm1)
                        .arg(&fm2)
                        .arg(&fm3)
                        .arg(&smir0)
                        .arg(&smir1)
                        .arg(&smir2)
                        .arg(&smir3)
                        .arg(&o.a_term)
                        .arg(&o.b_term)
                        .arg(&o.wb_root)
                        .arg(&self.d_program)
                        .arg(&self.prog_len)
                        .arg(&mut d_pc)
                        .arg(&mut d_regs)
                        .arg(&mut d_fz)
                        .arg(&mut d_fc)
                        .arg(&mut d_lr)
                        .arg(&mut d_ram)
                        .arg(&mut d_halted)
                        .arg(&mut d_retired)
                        .arg(&max_i)
                        .arg(&lanes_i)
                        .launch(cfg)
                        .map_err(|e| CudaError::LaunchFailed(format!("device_run: {e:?}")))?;
                }
                GpuOpStorage::Split {
                    d_code,
                    d_out,
                    d_in0,
                    d_in1,
                    d_in2,
                    d_n0,
                    d_n1,
                    d_n2,
                    d_n3,
                    d_threshold,
                    d_shift,
                    d_mask,
                } => {
                    rt.stream()
                        .launch_builder(&self.func)
                        .arg(&d_vals)
                        .arg(d_code)
                        .arg(d_out)
                        .arg(d_in0)
                        .arg(d_in1)
                        .arg(d_in2)
                        .arg(d_n0)
                        .arg(d_n1)
                        .arg(d_n2)
                        .arg(d_n3)
                        .arg(d_threshold)
                        .arg(d_shift)
                        .arg(d_mask)
                        .arg(&o.cone_end)
                        .arg(&o.settle_end)
                        .arg(&o.n_total)
                        .arg(&o.pc_slot)
                        .arg(&fm0)
                        .arg(&fm1)
                        .arg(&fm2)
                        .arg(&fm3)
                        .arg(&smir0)
                        .arg(&smir1)
                        .arg(&smir2)
                        .arg(&smir3)
                        .arg(&o.a_term)
                        .arg(&o.b_term)
                        .arg(&o.wb_root)
                        .arg(&self.d_program)
                        .arg(&self.prog_len)
                        .arg(&mut d_pc)
                        .arg(&mut d_regs)
                        .arg(&mut d_fz)
                        .arg(&mut d_fc)
                        .arg(&mut d_lr)
                        .arg(&mut d_ram)
                        .arg(&mut d_halted)
                        .arg(&mut d_retired)
                        .arg(&max_i)
                        .arg(&lanes_i)
                        .launch(cfg)
                        .map_err(|e| CudaError::LaunchFailed(format!("device_run: {e:?}")))?;
                }
            }
        }
        rt.stream()
            .synchronize()
            .map_err(|e| CudaError::LaunchFailed(format!("sync: {e:?}")))?;
        let kernel_secs = t0.elapsed().as_secs_f64();

        let r_pc = rt.download(&d_pc)?;
        let r_regs = rt.download(&d_regs)?;
        let r_fz = rt.download(&d_fz)?;
        let r_fc = rt.download(&d_fc)?;
        let r_lr = rt.download(&d_lr)?;
        let r_ram = rt.download(&d_ram)?;
        let r_halted = rt.download(&d_halted)?;
        let r_retired = rt.download(&d_retired)?;

        let mut out = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let regs = std::array::from_fn(|r| r_regs[r * lanes + lane]);
            let ram = (0..rw).map(|a| r_ram[a * lanes + lane]).collect();
            out.push(DeviceCycleState {
                pc: r_pc[lane],
                lr: r_lr[lane],
                flag_z: r_fz[lane] != 0,
                flag_c: r_fc[lane] != 0,
                halted: r_halted[lane] != 0,
                // Non-pipelined: one device cycle per retired instruction.
                cycle: r_retired[lane],
                retired: r_retired[lane],
                regs,
                ram,
            });
        }
        Ok((out, kernel_secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::Simulation;
    use crate::tile_cpu::v2_device_cycle::DeviceCpu;
    use crate::tile_cpu::v2_full_phase_prover::charter_cases;
    use crate::tile_cpu::{
        TileCpuV2, V2Builder, V2MmioCombinedDevice, V2MmioHandle, V2SynthConfig, compile_source,
        decode_v2_word,
    };
    use std::rc::Rc;

    /// Build a charter-config CPU + its device-cycle plan from explicit program words
    /// (mirrors the v2_device_cycle test builder; derives indices on a throwaway sim BEFORE
    /// `build` consumes the builder).
    fn build_plan_from_words(program: &[u32]) -> (TileCpuV2, Simulation, DeviceCyclePlan) {
        let console = Rc::new(V2MmioCombinedDevice::new(0x000B_E5C0));
        let builder = V2Builder::new()
            .with_origin(0, 0)
            .with_program(program)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_extended_pc()
            .with_mmio(V2MmioHandle::from_rc(console))
            .with_synth_blocks(V2SynthConfig::max_authority());
        let mut scratch = Simulation::with_size_layered(128, 640, 16);
        let layout = builder.derive_indices(&mut scratch);
        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = builder.build(&mut sim);
        let plan = DeviceCyclePlan::new(&cpu, &sim, layout).expect("plan");
        (cpu, sim, plan)
    }

    fn build_charter_plan(source: &str) -> (TileCpuV2, Simulation, DeviceCyclePlan, Vec<u32>) {
        let program = compile_source(source).expect("compile");
        let (cpu, sim, plan) = build_plan_from_words(&program);
        (cpu, sim, plan, program)
    }

    fn cuda_or_skip() -> Option<CudaRuntime> {
        if !logic_fabric_core::cuda::is_nvrtc_available() {
            eprintln!("CUDA/NVRTC not available, skipping GPU test");
            return None;
        }
        match CudaRuntime::new() {
            Ok(rt) => Some(rt),
            Err(_) => {
                eprintln!("CUDA not available, skipping GPU test");
                None
            }
        }
    }

    /// First binary-register ALU instruction (ADD/SUB/AND/OR/XOR, not MUL) at pc < 64.
    fn first_binary_alu(program: &[u32]) -> (u32, u8) {
        program
            .iter()
            .enumerate()
            .find_map(|(i, &w)| {
                let d = decode_v2_word(w);
                let binary = matches!(d.opcode, 0x05 | 0x06 | 0x07 | 0x08)
                    || (d.opcode == 0x04 && d.imm5 != 1);
                (i < 64 && binary).then_some((i as u32, d.opcode))
            })
            .expect("no binary-register ALU instruction at pc < 64")
    }

    /// 1d.1 acceptance: the GPU device-ALU, K lanes with DIFFERENT `(a, b)`, equals
    /// `DeviceCyclePlan::standalone_alu` (the CPU oracle) for every lane.
    #[test]
    fn device_alu_gpu_matches_standalone_alu() -> Result<(), Box<dyn std::error::Error>> {
        let Some(rt) = cuda_or_skip() else {
            return Ok(());
        };
        const LANES: usize = 64;
        let case = charter_cases()[1]; // sum_loop
        let (_cpu, sim, plan, program) = build_charter_plan(case.source);
        let (pc, _op) = first_binary_alu(&program);

        let dprog = DeviceAluProgram::from_plan(&plan);
        let host = dprog.seed_from_sim(&sim, LANES);
        let mut gpu = GpuDeviceAlu::new(&rt, &dprog, &host, LANES)?;

        let pcs = vec![pc as u64; LANES];
        let a: Vec<u64> = (0..LANES)
            .map(|l| 0x0000_0000_1234_5678u64 ^ (l as u64).wrapping_mul(0x9E37_79B9))
            .collect();
        let b: Vec<u64> = (0..LANES)
            .map(|l| 0x0000_0000_0000_abcdu64 ^ ((l as u64) << 8))
            .collect();

        let out = gpu.run(&rt, &pcs, &a, &b)?;
        for l in 0..LANES {
            let expected = plan.standalone_alu(&sim, pc, a[l], b[l]);
            assert_eq!(
                out[l], expected,
                "lane {l}: GPU {:#018x} != oracle {:#018x}",
                out[l], expected
            );
        }
        Ok(())
    }

    /// A fresh zeroed device state with `ram_words` scratch words.
    fn zero_state(ram_words: usize) -> DeviceCycleState {
        DeviceCycleState {
            pc: 0,
            lr: 0,
            flag_z: false,
            flag_c: false,
            halted: false,
            cycle: 0,
            retired: 0,
            regs: [0; 16],
            ram: vec![0; ram_words],
        }
    }

    /// 1d.2 acceptance: ONE GPU instruction updates per-lane state identically to one
    /// `DeviceCpu::step()`, for K lanes with DISTINCT initial register state, across MANY
    /// real charter instruction/state combinations (every pc the program walks). Proves the
    /// on-device scalar shell (decode + dispatch + control + flags + writeback + RAM).
    #[test]
    fn device_step_gpu_matches_devicecpu_step() -> Result<(), Box<dyn std::error::Error>> {
        let Some(rt) = cuda_or_skip() else {
            return Ok(());
        };
        const LANES: usize = 32;
        const RW: usize = 128;
        let case = charter_cases()[1]; // sum_loop (loop: LDI/ADD/CMP/JZ/JMP/...)
        let (_cpu, sim, plan, program) = build_charter_plan(case.source);
        let dprog = DeviceAluProgram::from_plan(&plan);
        let mut gpu = GpuDeviceCpu::new(&rt, &dprog, &sim, &program, RW, LANES)?;

        // Two reference CPUs (each owns a plan, built ONCE — never per lane): `refcpu`
        // walks the real program to generate genuine pre-step states; `oracle` one-steps
        // each perturbed lane state. The persistent buf is safe to reuse across arbitrary
        // state resets (each ALU eval recomputes the decode from the ROM in the buffer).
        let (_rc, rsim, rplan) = build_plan_from_words(&program);
        let mut refcpu = DeviceCpu::new(rplan, &rsim, program.clone());
        let (_oc, osim, oplan) = build_plan_from_words(&program);
        let mut oracle = DeviceCpu::new(oplan, &osim, program.clone());

        let mut steps = 0u64;
        let mut checked = 0u64;
        while !refcpu.state.halted && steps < case.max_cycles && checked < 40 {
            let base = refcpu.state.clone();
            // Build K distinct lane states by perturbing registers (pc/flags/lr/ram kept so
            // the instruction is real; perturbing regs exercises the datapath with distinct
            // data, and the per-lane oracle uses the SAME perturbation).
            let mut inits = Vec::with_capacity(LANES);
            for l in 0..LANES {
                let mut s = base.clone();
                for r in 0..16 {
                    s.regs[r] = s.regs[r].wrapping_add((l as u64).wrapping_mul(0x100 + r as u64));
                }
                inits.push(s);
            }
            let gpu_out = gpu.run(&rt, &inits, 1)?;
            for (l, init) in inits.iter().enumerate() {
                oracle.state = init.clone();
                oracle.step();
                let diff = gpu_out[l].diff(&oracle.state);
                assert!(
                    diff.is_empty(),
                    "{}: step {steps} lane {l} GPU != DeviceCpu::step: {diff:?}",
                    case.name
                );
            }
            checked += 1;
            // Advance the reference by one real step.
            refcpu.step();
            steps += 1;
        }
        assert!(checked > 0, "no steps were validated");
        Ok(())
    }

    /// 1d.3 acceptance (the deliverable): the full megakernel runs each charter program to
    /// halt for K lanes with no host interaction in the loop, and every lane's final
    /// `DeviceCycleState` == the `DeviceCpu` oracle (and the expected R0). Reports aggregate
    /// lane-instructions/sec + the memory-bounded max K.
    #[test]
    fn device_run_gpu_matches_devicecpu_charter() -> Result<(), Box<dyn std::error::Error>> {
        let Some(rt) = cuda_or_skip() else {
            return Ok(());
        };
        const LANES: usize = 64;
        const RW: usize = 128;
        for case in charter_cases() {
            let (_cpu, sim, plan, program) = build_charter_plan(case.source);
            let dprog = DeviceAluProgram::from_plan(&plan);
            let slots = dprog.n_slots;
            let mut gpu = GpuDeviceCpu::new(&rt, &dprog, &sim, &program, RW, LANES)?;

            // CPU oracle: one DeviceCpu run to halt (all lanes share the zero init here).
            let mut oracle = DeviceCpu::new(plan, &sim, program.clone());
            oracle.run_to_halt(case.max_cycles);
            assert!(oracle.state.halted, "{}: oracle did not halt", case.name);
            assert_eq!(
                oracle.state.regs[0], case.expected_r0,
                "{}: oracle R0",
                case.name
            );

            // GPU: K lanes, same program, run to halt (megakernel).
            let inits = vec![zero_state(RW); LANES];
            let t0 = std::time::Instant::now();
            let out = gpu.run(&rt, &inits, case.max_cycles)?;
            let secs = t0.elapsed().as_secs_f64();

            for (l, st) in out.iter().enumerate() {
                let diff = st.diff(&oracle.state);
                assert!(
                    diff.is_empty(),
                    "{}: lane {l} GPU != oracle: {diff:?}",
                    case.name
                );
            }
            let total_instrs: u64 = out.iter().map(|s| s.retired).sum();
            let mips = (total_instrs as f64 / secs) / 1.0e6;
            // Memory-bounded max K: vals(slots) + ram(128) dominate, 8 B each, on ~30 GB.
            let bytes_per_lane = (slots + RW + 32) * 8;
            let max_k = (30_usize * 1_000_000_000) / bytes_per_lane;
            eprintln!(
                "1d.3 {}: K={LANES} retired/lane={} slots={slots} -> {:.1} M lane-instr/s in {:.3}s; \
                 mem ~{} B/lane => max K ~{}",
                case.name, oracle.state.retired, mips, secs, bytes_per_lane, max_k
            );
        }
        Ok(())
    }

    /// 1d.3 lane-independence + data-divergence: a tiny program that READS its initial
    /// registers (`R0 = R0 + R1 + R2; HALT`) gives genuinely DISTINCT per-lane outputs.
    /// Each lane's full final state == `DeviceCpu` for that lane's distinct input, and the
    /// outputs are not all equal — proving zero cross-lane interference with real data
    /// divergence.
    #[test]
    fn device_run_gpu_distinct_inputs() -> Result<(), Box<dyn std::error::Error>> {
        let Some(rt) = cuda_or_skip() else {
            return Ok(());
        };
        const LANES: usize = 48;
        const RW: usize = 128;
        // ADD R0,R1 ; ADD R0,R2 ; HALT  (two-operand ALU: rd = rd + rs).
        let add_r0_r1 = (0x04u32 << 11) | (0 << 8) | (1 << 5); // imm5=0 => plain ADD
        let add_r0_r2 = (0x04u32 << 11) | (0 << 8) | (2 << 5);
        let halt = 0x01u32 << 11;
        let program = vec![add_r0_r1, add_r0_r2, halt];
        // Sanity-check the hand encoding before trusting it.
        let d0 = decode_v2_word(program[0]);
        assert!(
            d0.opcode == 0x04 && d0.imm5 != 1 && d0.rd == 0 && d0.rs == 1,
            "{d0:?}"
        );

        let (_cpu, sim, plan) = build_plan_from_words(&program);
        let dprog = DeviceAluProgram::from_plan(&plan);
        let mut gpu = GpuDeviceCpu::new(&rt, &dprog, &sim, &program, RW, LANES)?;

        let mut inits = Vec::with_capacity(LANES);
        for l in 0..LANES {
            let mut s = zero_state(RW);
            s.regs[0] = (l as u64) * 7 + 1;
            s.regs[1] = (l as u64) * 13 + 2;
            s.regs[2] = (l as u64) * 100 + 3;
            inits.push(s);
        }
        let out = gpu.run(&rt, &inits, 16)?;

        // One oracle DeviceCpu (built once), state reset per lane.
        let (_oc, osim, oplan) = build_plan_from_words(&program);
        let mut oracle = DeviceCpu::new(oplan, &osim, program.clone());
        for (l, init) in inits.iter().enumerate() {
            oracle.state = init.clone();
            oracle.run_to_halt(16);
            let diff = out[l].diff(&oracle.state);
            assert!(diff.is_empty(), "lane {l} GPU != oracle: {diff:?}");
            let expected = init.regs[0]
                .wrapping_add(init.regs[1])
                .wrapping_add(init.regs[2]);
            assert_eq!(out[l].regs[0], expected, "lane {l} R0");
        }
        // Outputs genuinely differ across lanes (data divergence, not a constant).
        let distinct: std::collections::HashSet<u64> = out.iter().map(|s| s.regs[0]).collect();
        assert!(
            distinct.len() == LANES,
            "expected distinct per-lane outputs, got {}",
            distinct.len()
        );
        Ok(())
    }

    /// 1d.3 report (the deliverable's throughput/divergence numbers): scale K on a single
    /// charter program and print aggregate lane-instructions/sec. At small K the 5090 is
    /// almost idle (a few threads); aggregate throughput climbs ~linearly with K until the
    /// device fills — that scaling IS the batch payoff. Divergence proxy: with the SAME
    /// program+input every lane retires the IDENTICAL instruction count (perfect lockstep,
    /// zero control divergence — SPMD). Bounded (<1 min), correctness-sampled.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "informational perf/scaling report; run with --features slow-tests"
    )]
    fn device_run_gpu_throughput_report() -> Result<(), Box<dyn std::error::Error>> {
        let Some(rt) = cuda_or_skip() else {
            return Ok(());
        };
        const RW: usize = 128;
        // RTX 5090: 32 GB GDDR7, ~1792 GB/s peak DRAM bandwidth.
        const PEAK_GBS: f64 = 1792.0;
        const PRE_LAZY_BYTES_PER_OPEVAL: f64 = 40.0;

        let case = charter_cases()[1]; // sum_loop (cheapest charter program)
        let (_cpu, sim, plan, program) = build_charter_plan(case.source);
        let dprog_full = DeviceAluProgram::from_plan_opt(&plan, false);
        let dprog = DeviceAluProgram::from_plan(&plan); // dead-op eliminated (default)
        let n_total = dprog.op_count();
        let (n_cone, n_settle, n_trunk) = dprog.phase_op_counts();
        let (f_cone, f_settle, f_trunk) = dprog_full.phase_op_counts();
        let n_full = f_cone + f_settle + f_trunk;
        let slots = dprog.n_slots;
        let bytes_per_opeval = dprog.estimated_value_bytes_per_eval();
        let (meta_prepack, meta_packed) = dprog.estimated_metadata_bytes_per_eval();

        // CPU oracle: final state + count ALU-class instructions executed per lane (only
        // ALU ops 0x04..=0x14 run the tile op arrays; the rest are pure scalar shell).
        let mut oracle = DeviceCpu::new(plan, &sim, program.clone());
        let mut alu_instrs = 0u64;
        let mut steps = 0u64;
        while !oracle.state.halted && steps < case.max_cycles {
            let op = decode_v2_word(program[oracle.state.pc as usize]).opcode;
            if (0x04..=0x14).contains(&op) {
                alu_instrs += 1;
            }
            oracle.step();
            steps += 1;
        }
        assert_eq!(oracle.state.regs[0], case.expected_r0);
        let retired = oracle.state.retired;

        eprintln!(
            "1d.3 scaling — {} | retired/lane={retired}, ALU(tile) instrs/lane={alu_instrs} | \
             op stream cone={n_cone}+settle={n_settle}+trunk={n_trunk}={n_total} ops over {slots} slots",
            case.name
        );
        eprintln!(
            "   dead-op elimination: {n_full} -> {n_total} ops ({:.2}x fewer; settle {f_settle}->{n_settle}); \
             slots {} -> {slots} ({:.2}x smaller vals)",
            n_full as f64 / n_total as f64,
            dprog_full.n_slots,
            dprog_full.n_slots as f64 / slots as f64
        );
        eprintln!(
            "   per ALU instr = {n_total} op-evals; per lane = {} tile-op-evals total",
            alu_instrs * n_total as u64
        );
        eprintln!(
            "   lazy value loads: {:.1} -> {bytes_per_opeval:.1} logical vals bytes/op-eval",
            PRE_LAZY_BYTES_PER_OPEVAL
        );
        eprintln!(
            "   packed metadata: {meta_prepack:.1} -> {meta_packed:.1} logical metadata bytes/op-eval"
        );
        let mut prev_mips = 0.0_f64;
        for &k in &[256usize, 1024, 4096, 16384, 65536, 131072, 262144] {
            let mode = GpuMetadataMode::for_lanes(k);
            let mut gpu =
                GpuDeviceCpu::new_with_metadata_mode(&rt, &dprog, &sim, &program, RW, k, mode)?;
            let inits = vec![zero_state(RW); k];
            let (out, ksecs) = gpu.run_timed(&rt, &inits, case.max_cycles)?;

            // Correctness + lockstep (divergence) check while we're here.
            assert_eq!(out[0].regs[0], case.expected_r0, "K={k} lane 0 R0");
            assert_eq!(out[k - 1].regs[0], case.expected_r0, "K={k} last lane R0");
            let rmin = out.iter().map(|s| s.retired).min().unwrap();
            let rmax = out.iter().map(|s| s.retired).max().unwrap();
            assert_eq!(
                rmin, rmax,
                "same-input lanes must be lockstep (zero divergence)"
            );

            let total_instrs = retired * k as u64;
            let mips = (total_instrs as f64 / ksecs) / 1.0e6;
            let op_evals = alu_instrs * n_total as u64 * k as u64;
            let gops = (op_evals as f64 / ksecs) / 1.0e9;
            let gbs = (op_evals as f64 * bytes_per_opeval / ksecs) / 1.0e9;
            let pct = gbs / PEAK_GBS * 100.0;
            let scale = if prev_mips > 0.0 {
                format!(" | x{:.2} vs prev K", mips / prev_mips)
            } else {
                String::new()
            };
            eprintln!(
                "   K={k:>6} [{:>6}]: {mips:>8.1} M lane-instr/s | {gops:>6.1} G tile-op-evals/s | \
                 ~{gbs:>6.0} logical vals GB/s ({pct:>4.1}% peak) | kernel {ksecs:.3}s{scale}",
                mode.label()
            );
            prev_mips = mips;
        }

        // Backend A/B at the empirically sensitive saturation band.
        let kmode = 65_536usize;
        let inits = vec![zero_state(RW); kmode];
        let mut g_packed = GpuDeviceCpu::new_with_metadata_mode(
            &rt,
            &dprog,
            &sim,
            &program,
            RW,
            kmode,
            GpuMetadataMode::Packed,
        )?;
        let (out_packed, packed_secs) = g_packed.run_timed(&rt, &inits, case.max_cycles)?;
        let mut g_split = GpuDeviceCpu::new_with_metadata_mode(
            &rt,
            &dprog,
            &sim,
            &program,
            RW,
            kmode,
            GpuMetadataMode::Split,
        )?;
        let (out_split, split_secs) = g_split.run_timed(&rt, &inits, case.max_cycles)?;
        assert_eq!(out_packed, out_split, "metadata backends must be bit-exact");
        eprintln!(
            "   metadata A/B @ K={kmode}: packed {packed_secs:.3}s vs split {split_secs:.3}s = {:.2}x split/packed",
            packed_secs / split_secs
        );

        // A/B at a fixed K: full vs pruned op stream (same slot layout, fewer op-evals).
        let kab = 16384usize;
        let inits = vec![zero_state(RW); kab];
        let mut g_full = GpuDeviceCpu::new_with_metadata_mode(
            &rt,
            &dprog_full,
            &sim,
            &program,
            RW,
            kab,
            GpuMetadataMode::for_lanes(kab),
        )?;
        let (out_f, full_secs) = g_full.run_timed(&rt, &inits, case.max_cycles)?;
        let mut g_pruned = GpuDeviceCpu::new_with_metadata_mode(
            &rt,
            &dprog,
            &sim,
            &program,
            RW,
            kab,
            GpuMetadataMode::for_lanes(kab),
        )?;
        let (out_p, pruned_secs) = g_pruned.run_timed(&rt, &inits, case.max_cycles)?;
        assert_eq!(out_f[0].regs[0], case.expected_r0, "full A/B R0");
        assert_eq!(out_p[0].regs[0], case.expected_r0, "pruned A/B R0");
        assert_eq!(out_f, out_p, "pruned must be bit-exact vs full"); // same final states
        eprintln!(
            "   A/B @ K={kab}: full {full_secs:.3}s vs pruned {pruned_secs:.3}s = {:.2}x speedup",
            full_secs / pruned_secs
        );
        Ok(())
    }
}
