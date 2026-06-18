//! SIMT Tier 2a — lane-packed dense compact-op evaluation kernel.
//!
//! The SIMT roadmap's central empirical question (§6) is dense-across-lanes vs
//! sparse-per-instance: a lane-packed tick must evaluate a FIXED op set for all
//! lanes, trading per-instance dirty skipping for SIMD width. This module
//! answers it on the dominant per-step cost — the settle compact-op array
//! (~72% of tile evals per the S377 dashboard) — without touching the
//! evaluator: it is a standalone kernel over the Sprint-385 SoA value layout.
//!
//! Semantics mirror `Simulation::propagate_compact` exactly for the
//! SIMT-supported op subset (no `COP_GENERIC`, no `COP_WIRE`; threshold-via
//! uses side metadata captured from `Simulation`). Lane 0 of a kernel run must be byte-identical
//! to a scalar `propagate_compact` run, and every lane must equal its OWN
//! scalar run when seeded differently — that is the SIMT thesis (program/data
//! divergence is data divergence, not control divergence) stated as a test.
//!
//! Layout: `vals[tile_idx * LANES + lane]`, plus one trailing ZERO ROW; op
//! inputs of `u32::MAX` (no neighbor) are remapped to the zero row at kernel
//! build time so the lane loop is branchless.

use crate::simulation::{
    COP_ADD, COP_AND, COP_BITSEL, COP_CARRY, COP_CONST, COP_DEC3, COP_GENERIC, COP_MUX, COP_MUX4,
    COP_MUX16, COP_NOT, COP_OR, COP_RAM, COP_SHL, COP_SHR, COP_SUB, COP_THRESHOLD_VIA, COP_VIA,
    COP_WIRE, COP_WIRE_D, COP_WIRE_H, COP_WIRE_L, COP_WIRE_R, COP_WIRE_U, COP_WIRE_V, COP_WVIA,
    COP_XOR, COP_ZERO, CompactOp, Simulation,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ThresholdViaParams {
    pub(crate) neighbors: [u32; 4],
    pub(crate) threshold: u8,
}

/// Lane-packed value storage: `vals[idx * lanes + lane]`, one extra all-zero
/// row at `idx == tile_rows - 1` for boundary (u32::MAX) inputs.
pub struct LanePack {
    lanes: usize,
    /// tile_count + 1 (last row is the permanent zero row).
    tile_rows: usize,
    vals: Vec<u64>,
}

impl LanePack {
    /// Replicate a simulation's current values into all lanes.
    pub fn from_sim(sim: &Simulation, lanes: usize) -> Self {
        let tile_count = sim.tilemap.tile_count();
        let tile_rows = tile_count + 1;
        let mut vals = vec![0u64; tile_rows * lanes];
        for idx in 0..tile_count {
            let v = sim.tilemap.value(idx);
            for lane in 0..lanes {
                vals[idx * lanes + lane] = v;
            }
        }
        Self {
            lanes,
            tile_rows,
            vals,
        }
    }

    #[inline]
    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// Number of addressable tiles (excludes the internal zero row).
    #[inline]
    pub fn tile_count(&self) -> usize {
        self.tile_rows - 1
    }

    #[inline]
    pub fn get(&self, idx: usize, lane: usize) -> u64 {
        self.vals[idx * self.lanes + lane]
    }

    #[inline]
    pub fn set(&mut self, idx: usize, lane: usize, v: u64) {
        debug_assert!(idx < self.tile_rows - 1, "cannot write the zero row");
        self.vals[idx * self.lanes + lane] = v;
    }
}

/// A preflighted, zero-row-remapped compact-op program for lane evaluation.
#[derive(Debug)]
pub struct LaneKernel {
    ops: Vec<CompactOp>,
    /// (shift, mask) per op, aligned with `ops` (unused slots are (0, 0)).
    wvia: Vec<(u8, u64)>,
    /// Threshold-via metadata per op. Only meaningful when `op == COP_THRESHOLD_VIA`.
    threshold_via: Vec<ThresholdViaParams>,
    tile_count: usize,
}

impl LaneKernel {
    /// Build from a compact-op array. Fails on ops the kernel cannot evaluate
    /// without extra simulation metadata (`COP_GENERIC`, `COP_WIRE`, and
    /// `COP_THRESHOLD_VIA` in this constructor).
    /// `wvia_params` is consumed in op order, matching `propagate_compact`.
    pub fn new(
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
        tile_count: usize,
    ) -> Result<Self, String> {
        Self::new_inner(ops, wvia_params, tile_count, None)
    }

    /// Build a lane kernel from a live simulation. Required for ops whose
    /// inputs do not fit inside `CompactOp` itself, currently `COP_THRESHOLD_VIA`.
    pub fn new_from_sim(
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
        sim: &Simulation,
    ) -> Result<Self, String> {
        Self::new_inner(ops, wvia_params, sim.tilemap.tile_count(), Some(sim))
    }

    fn new_inner(
        ops: &[CompactOp],
        wvia_params: &[(usize, u8, u64)],
        tile_count: usize,
        sim: Option<&Simulation>,
    ) -> Result<Self, String> {
        let zero_row = tile_count as u32;
        let mut out_ops = Vec::with_capacity(ops.len());
        let mut wvia = Vec::with_capacity(ops.len());
        let mut threshold_via = Vec::with_capacity(ops.len());
        let mut wvia_idx = 0usize;
        for (i, op) in ops.iter().enumerate() {
            match op.op {
                COP_GENERIC => {
                    return Err(format!("op {i} (tile {}) is COP_GENERIC", op.idx));
                }
                COP_WIRE => {
                    return Err(format!("op {i} (tile {}) is COP_WIRE", op.idx));
                }
                COP_THRESHOLD_VIA if sim.is_none() => {
                    return Err(format!(
                        "op {i} (tile {}) is COP_THRESHOLD_VIA; use LaneKernel::new_from_sim",
                        op.idx
                    ));
                }
                _ => {}
            }
            if op.idx as usize >= tile_count && op.op != COP_CONST {
                return Err(format!("op {i} writes out-of-range tile {}", op.idx));
            }
            let remap = |i: u32| -> u32 {
                if i == u32::MAX || i as usize >= tile_count {
                    zero_row
                } else {
                    i
                }
            };
            out_ops.push(CompactOp {
                op: op.op,
                _pad: [0; 3],
                idx: op.idx,
                in0: remap(op.in0),
                in1: remap(op.in1),
                in2: remap(op.in2),
            });
            if op.op == COP_WVIA {
                let (_, shift, mask) = *wvia_params
                    .get(wvia_idx)
                    .ok_or_else(|| format!("op {i}: missing wvia params"))?;
                wvia_idx += 1;
                wvia.push((shift, mask));
            } else {
                wvia.push((0, 0));
            }
            if op.op == COP_THRESHOLD_VIA {
                let idx = op.idx as usize;
                let sim = sim.expect("COP_THRESHOLD_VIA preflight requires simulation metadata");
                let neighbors = (*sim.neighbors4_at(idx)).map(|n| remap(n));
                threshold_via.push(ThresholdViaParams {
                    neighbors,
                    threshold: sim.get_tile_threshold(idx),
                });
            } else {
                threshold_via.push(ThresholdViaParams::default());
            }
        }
        Ok(Self {
            ops: out_ops,
            wvia,
            threshold_via,
            tile_count,
        })
    }

    #[inline]
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    #[inline]
    pub fn tile_count(&self) -> usize {
        self.tile_count
    }

    /// Sprint 387 (Tier 3a): the preflighted, zero-row-remapped program —
    /// shared verbatim with the GPU host harness so CPU and GPU evaluate the
    /// identical op stream. Returns (ops, per-op (shift, mask)).
    #[cfg(feature = "cuda")]
    pub(crate) fn remapped_program(&self) -> (&[CompactOp], &[(u8, u64)], &[ThresholdViaParams]) {
        (&self.ops, &self.wvia, &self.threshold_via)
    }

    /// Evaluate every op once, in order, across all lanes (dense — no dirty
    /// tracking). Per-lane semantics are exactly `propagate_compact`'s.
    pub fn eval(&self, lp: &mut LanePack) {
        assert_eq!(
            lp.tile_rows,
            self.tile_count + 1,
            "LanePack tile count does not match kernel"
        );
        let lanes = lp.lanes;
        self.eval_slice(&mut lp.vals, lanes);
    }

    /// Sprint 386: evaluate against a raw interleaved buffer (e.g. the
    /// Tilemap's resident lane storage via `lane_vals_raw_mut`). `vals.len()`
    /// must be `(tile_count + 1) * lanes` (trailing zero row included).
    pub fn eval_slice(&self, vals: &mut [u64], lanes: usize) {
        assert_eq!(
            vals.len(),
            (self.tile_count + 1) * lanes,
            "lane buffer does not match kernel tile count"
        );
        match lanes {
            1 => self.eval_const::<1>(vals),
            2 => self.eval_const::<2>(vals),
            4 => self.eval_const::<4>(vals),
            8 => self.eval_const::<8>(vals),
            16 => self.eval_const::<16>(vals),
            32 => self.eval_const::<32>(vals),
            n => self.eval_dyn(vals, n),
        }
    }

    /// Monomorphized lane loop: fixed L lets LLVM unroll/vectorize each arm.
    fn eval_const<const L: usize>(&self, vals: &mut [u64]) {
        #[inline(always)]
        fn row<const L: usize>(vals: &[u64], base: usize) -> [u64; L] {
            let mut out = [0u64; L];
            out.copy_from_slice(&vals[base..base + L]);
            out
        }

        for ((op, &(wshift, wmask)), tvia) in self
            .ops
            .iter()
            .zip(self.wvia.iter())
            .zip(self.threshold_via.iter())
        {
            if op.op == COP_CONST {
                continue;
            }
            let b0 = op.in0 as usize * L;
            let b1 = op.in1 as usize * L;
            let b2 = op.in2 as usize * L;
            let bo = op.idx as usize * L;

            let mut r = [0u64; L];
            match op.op {
                COP_WIRE_R | COP_VIA => {
                    r = row::<L>(vals, b0);
                }
                COP_WIRE_L => {
                    r = row::<L>(vals, b1);
                }
                COP_WIRE_D | COP_WIRE_U => {
                    r = row::<L>(vals, b2);
                }
                COP_WIRE_H | COP_OR => {
                    let a = row::<L>(vals, b0);
                    let b = row::<L>(vals, b1);
                    for i in 0..L {
                        r[i] = a[i] | b[i];
                    }
                }
                COP_WIRE_V => {
                    let b = row::<L>(vals, b1);
                    let c = row::<L>(vals, b2);
                    for i in 0..L {
                        r[i] = b[i] | c[i];
                    }
                }
                COP_AND => {
                    let a = row::<L>(vals, b0);
                    let b = row::<L>(vals, b1);
                    for i in 0..L {
                        r[i] = a[i] & b[i];
                    }
                }
                COP_XOR => {
                    let a = row::<L>(vals, b0);
                    let b = row::<L>(vals, b1);
                    for i in 0..L {
                        r[i] = a[i] ^ b[i];
                    }
                }
                COP_MUX => {
                    let a = row::<L>(vals, b0);
                    let b = row::<L>(vals, b1);
                    let c = row::<L>(vals, b2);
                    for i in 0..L {
                        let m = ((c[i] != 0) as u64).wrapping_neg();
                        r[i] = (a[i] & m) | (b[i] & !m);
                    }
                }
                COP_NOT => {
                    let a = row::<L>(vals, b0);
                    for i in 0..L {
                        r[i] = !a[i];
                    }
                }
                COP_ZERO => {
                    let a = row::<L>(vals, b0);
                    for i in 0..L {
                        r[i] = ((a[i] == 0) as u64).wrapping_neg();
                    }
                }
                COP_ADD => {
                    let a = row::<L>(vals, b0);
                    let b = row::<L>(vals, b1);
                    for i in 0..L {
                        r[i] = a[i].wrapping_add(b[i]);
                    }
                }
                COP_SUB => {
                    let a = row::<L>(vals, b0);
                    let b = row::<L>(vals, b1);
                    for i in 0..L {
                        r[i] = a[i].wrapping_sub(b[i]);
                    }
                }
                COP_SHR => {
                    let a = row::<L>(vals, b0);
                    let b = row::<L>(vals, b1);
                    for i in 0..L {
                        r[i] = a[i] >> (b[i] & 63);
                    }
                }
                COP_SHL => {
                    let a = row::<L>(vals, b0);
                    let b = row::<L>(vals, b1);
                    for i in 0..L {
                        r[i] = a[i] << (b[i] & 63);
                    }
                }
                COP_MUX16 => {
                    let a = row::<L>(vals, b0);
                    let b = row::<L>(vals, b1);
                    let c = row::<L>(vals, b2);
                    for i in 0..L {
                        let sel = (b[i] & 0xF) as u32;
                        let m = ((sel < 8) as u64).wrapping_neg();
                        let data = (a[i] & m) | (c[i] & !m);
                        r[i] = (data >> ((sel & 7) * 8)) & 0xFF;
                    }
                }
                COP_DEC3 => {
                    let a = row::<L>(vals, b0);
                    for i in 0..L {
                        r[i] = 1u64 << (a[i] & 7);
                    }
                }
                COP_BITSEL => {
                    let a = row::<L>(vals, b0);
                    let b = row::<L>(vals, b1);
                    for i in 0..L {
                        r[i] = (((a[i] >> (b[i] & 63)) & 1) as u64).wrapping_neg();
                    }
                }
                COP_CARRY => {
                    let a = row::<L>(vals, b0);
                    let b = row::<L>(vals, b1);
                    for i in 0..L {
                        r[i] = ((a[i] > b[i]) as u64).wrapping_neg();
                    }
                }
                COP_WVIA => {
                    let a = row::<L>(vals, b0);
                    for i in 0..L {
                        r[i] = (a[i] >> wshift) & wmask;
                    }
                }
                COP_THRESHOLD_VIA => {
                    let source = row::<L>(vals, b0);
                    let n0 = row::<L>(vals, tvia.neighbors[0] as usize * L);
                    let n1 = row::<L>(vals, tvia.neighbors[1] as usize * L);
                    let n2 = row::<L>(vals, tvia.neighbors[2] as usize * L);
                    let n3 = row::<L>(vals, tvia.neighbors[3] as usize * L);
                    for i in 0..L {
                        let active = (n0[i] != 0) as u8
                            + (n1[i] != 0) as u8
                            + (n2[i] != 0) as u8
                            + (n3[i] != 0) as u8;
                        r[i] = if active >= tvia.threshold {
                            source[i]
                        } else {
                            0
                        };
                    }
                }
                COP_MUX4 => {
                    let a = row::<L>(vals, b0);
                    let c = row::<L>(vals, b2);
                    for i in 0..L {
                        let sel = (c[i] & 0b11) as u32;
                        r[i] = (a[i] >> (sel * 8)) & 0xFF;
                    }
                }
                COP_RAM => {
                    let a = row::<L>(vals, b0);
                    let c = row::<L>(vals, b2);
                    let cur = row::<L>(vals, bo);
                    for i in 0..L {
                        let m = ((c[i] != 0) as u64).wrapping_neg();
                        r[i] = (a[i] & m) | (cur[i] & !m);
                    }
                }
                _ => {
                    // Preflight rejects GENERIC/WIRE; remaining codes keep the
                    // tile stable, matching propagate_compact's `_ => current`.
                    r = row::<L>(vals, bo);
                }
            }
            vals[bo..bo + L].copy_from_slice(&r);
        }
    }

    /// Fallback for arbitrary lane counts (no monomorphized width).
    fn eval_dyn(&self, vals: &mut [u64], lanes: usize) {
        for ((op, &(wshift, wmask)), tvia) in self
            .ops
            .iter()
            .zip(self.wvia.iter())
            .zip(self.threshold_via.iter())
        {
            if op.op == COP_CONST {
                continue;
            }
            let b0 = op.in0 as usize * lanes;
            let b1 = op.in1 as usize * lanes;
            let b2 = op.in2 as usize * lanes;
            let bo = op.idx as usize * lanes;
            for i in 0..lanes {
                let v0 = vals[b0 + i];
                let v1 = vals[b1 + i];
                let v2 = vals[b2 + i];
                let current = vals[bo + i];
                let result = match op.op {
                    COP_WIRE_R | COP_VIA => v0,
                    COP_WIRE_L => v1,
                    COP_WIRE_D | COP_WIRE_U => v2,
                    COP_WIRE_H | COP_OR => v0 | v1,
                    COP_WIRE_V => v1 | v2,
                    COP_AND => v0 & v1,
                    COP_XOR => v0 ^ v1,
                    COP_MUX => {
                        if v2 != 0 {
                            v0
                        } else {
                            v1
                        }
                    }
                    COP_NOT => !v0,
                    COP_ZERO => {
                        if v0 == 0 {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    COP_ADD => v0.wrapping_add(v1),
                    COP_SUB => v0.wrapping_sub(v1),
                    COP_SHR => v0 >> (v1 & 63),
                    COP_SHL => v0 << (v1 & 63),
                    COP_MUX16 => {
                        let sel = (v1 & 0xF) as usize;
                        let data = if sel < 8 { v0 } else { v2 };
                        (data >> ((sel & 7) * 8)) & 0xFF
                    }
                    COP_DEC3 => 1u64 << (v0 & 7),
                    COP_BITSEL => {
                        if (v0 >> (v1 & 63)) & 1 != 0 {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    COP_CARRY => {
                        if v0 > v1 {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    COP_WVIA => (v0 >> wshift) & wmask,
                    COP_THRESHOLD_VIA => {
                        let mut active = 0u8;
                        for &n in &tvia.neighbors {
                            if vals[n as usize * lanes + i] != 0 {
                                active += 1;
                            }
                        }
                        if active >= tvia.threshold { v0 } else { 0 }
                    }
                    COP_MUX4 => {
                        let sel = (v2 & 0b11) as u32;
                        (v0 >> (sel * 8)) & 0xFF
                    }
                    COP_RAM => {
                        if v2 != 0 {
                            v0
                        } else {
                            current
                        }
                    }
                    _ => current,
                };
                vals[bo + i] = result;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_meta::TileType;

    /// Deterministic synthetic op DAG exercising every supported op type.
    /// Tiles 0..64 are inputs (never written); op j writes tile 64+j.
    fn synthetic_ops(n_ops: usize) -> (Vec<CompactOp>, Vec<(usize, u8, u64)>) {
        const TYPES: [u8; 18] = [
            COP_WIRE_R, COP_WIRE_L, COP_WIRE_D, COP_WIRE_H, COP_WIRE_V, COP_AND, COP_OR, COP_XOR,
            COP_MUX, COP_NOT, COP_ZERO, COP_ADD, COP_SUB, COP_SHR, COP_SHL, COP_BITSEL, COP_CARRY,
            COP_WVIA,
        ];
        let mut ops = Vec::with_capacity(n_ops);
        let mut wvia = Vec::new();
        for j in 0..n_ops {
            let op = TYPES[j % TYPES.len()];
            let idx = (64 + j) as u32;
            // Inputs reach back into earlier outputs (and seeds) deterministically.
            let in0 = (j * 7 + 3) % (64 + j);
            let in1 = (j * 13 + 11) % (64 + j);
            let in2 = if j % 5 == 0 {
                u32::MAX as usize // exercise the boundary/zero-row path
            } else {
                (j * 29 + 17) % (64 + j)
            };
            if op == COP_WVIA {
                wvia.push((idx as usize, (j % 64) as u8, 0x00FF_FF00_FF00_FFFFu64));
            }
            ops.push(CompactOp {
                op,
                _pad: [0; 3],
                idx,
                in0: in0 as u32,
                in1: in1 as u32,
                in2: if in2 == u32::MAX as usize {
                    u32::MAX
                } else {
                    in2 as u32
                },
            });
        }
        (ops, wvia)
    }

    fn splitmix(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Scalar reference: run the ops through Simulation::propagate_compact.
    fn scalar_reference(
        ops: &[CompactOp],
        wvia: &[(usize, u8, u64)],
        seeds: &[u64],
        iters: usize,
        tile_count: usize,
    ) -> Vec<u64> {
        let mut sim = Simulation::new();
        assert!(sim.tilemap.tile_count() >= tile_count);
        for (i, &s) in seeds.iter().enumerate() {
            sim.set_logic_value_by_idx(i, s);
        }
        for _ in 0..iters {
            sim.propagate_compact(ops, wvia);
        }
        (0..tile_count).map(|i| sim.tilemap.value(i)).collect()
    }

    #[test]
    fn lane0_matches_scalar_propagate_compact() {
        let (ops, wvia) = synthetic_ops(500);
        let tile_count = 64 + 500;
        let kernel = LaneKernel::new(&ops, &wvia, tile_count).unwrap();

        let mut seed = 0xC0FF_EE11_D00D_F00Du64;
        let seeds: Vec<u64> = (0..64).map(|_| splitmix(&mut seed)).collect();

        // Lane-packed run (single lane).
        let sim = Simulation::new();
        let mut lp = LanePack {
            lanes: 1,
            tile_rows: tile_count + 1,
            vals: vec![0u64; tile_count + 1],
        };
        drop(sim);
        for (i, &s) in seeds.iter().enumerate() {
            lp.set(i, 0, s);
        }
        for _ in 0..3 {
            kernel.eval(&mut lp);
        }

        let reference = scalar_reference(&ops, &wvia, &seeds, 3, tile_count);
        for idx in 0..tile_count {
            assert_eq!(
                lp.get(idx, 0),
                reference[idx],
                "tile {idx} diverged from scalar propagate_compact"
            );
        }
    }

    #[test]
    fn every_lane_matches_its_own_scalar_run() {
        // The SIMT thesis: lanes with DIFFERENT data run in lockstep and each
        // must equal its own independent scalar evaluation.
        const LANES: usize = 8;
        let (ops, wvia) = synthetic_ops(300);
        let tile_count = 64 + 300;
        let kernel = LaneKernel::new(&ops, &wvia, tile_count).unwrap();

        let mut seed = 0xDEAD_BEEF_1234_5678u64;
        let lane_seeds: Vec<Vec<u64>> = (0..LANES)
            .map(|_| (0..64).map(|_| splitmix(&mut seed)).collect())
            .collect();

        let mut lp = LanePack {
            lanes: LANES,
            tile_rows: tile_count + 1,
            vals: vec![0u64; (tile_count + 1) * LANES],
        };
        for lane in 0..LANES {
            for (i, &s) in lane_seeds[lane].iter().enumerate() {
                lp.set(i, lane, s);
            }
        }
        for _ in 0..3 {
            kernel.eval(&mut lp);
        }

        for lane in 0..LANES {
            let reference = scalar_reference(&ops, &wvia, &lane_seeds[lane], 3, tile_count);
            for idx in 0..tile_count {
                assert_eq!(
                    lp.get(idx, lane),
                    reference[idx],
                    "lane {lane} tile {idx} diverged from its own scalar run"
                );
            }
        }
    }

    #[test]
    fn threshold_via_lanes_match_scalar_compact() {
        const LANES: usize = 8;
        let w = 8usize;
        let layer_size = w * w;
        let (cx, cy) = (3usize, 3usize);
        let cell = cy * w + cx;
        let neighbor_xy = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)];

        for &(via_z, src_z, via_type) in &[
            (0usize, 1usize, TileType::ThresholdViaUp),
            (1usize, 0usize, TileType::ThresholdViaDown),
        ] {
            for threshold in 0u8..=4 {
                let mut sim = Simulation::with_size_layered(w, w, 2);
                let via_idx = via_z * layer_size + cell;
                let src_idx = src_z * layer_size + cell;

                sim.set_tile_3d(cx, cy, src_z, TileType::Const);
                sim.set_tile_3d(cx, cy, via_z, via_type);
                sim.set_tile_threshold(via_idx, threshold);
                for &(nx, ny) in &neighbor_xy {
                    sim.set_tile_3d(nx, ny, via_z, TileType::Const);
                }
                sim.rebuild_via_connections();

                let (ops, wvia) = sim.build_compact_ops(&[via_idx]);
                let kernel = LaneKernel::new_from_sim(&ops, &wvia, &sim).unwrap();
                let mut lp = LanePack::from_sim(&sim, LANES);

                for lane in 0..LANES {
                    let combo = lane as u32;
                    let src = if lane % 3 == 0 {
                        0
                    } else {
                        0xABCD_0000_0000_0000u64 | lane as u64
                    };
                    lp.set(src_idx, lane, src);
                    lp.set(via_idx, lane, 0);
                    for (j, &(nx, ny)) in neighbor_xy.iter().enumerate() {
                        let nidx = via_z * layer_size + ny * w + nx;
                        let val = if (combo >> j) & 1 != 0 {
                            10 + lane as u64 + j as u64
                        } else {
                            0
                        };
                        lp.set(nidx, lane, val);
                    }
                }

                kernel.eval(&mut lp);

                for lane in 0..LANES {
                    let mut scalar = sim.clone();
                    let combo = lane as u32;
                    let src = if lane % 3 == 0 {
                        0
                    } else {
                        0xABCD_0000_0000_0000u64 | lane as u64
                    };
                    scalar.set_logic_value_by_idx(src_idx, src);
                    scalar.set_logic_value_by_idx(via_idx, 0);
                    for (j, &(nx, ny)) in neighbor_xy.iter().enumerate() {
                        let nidx = via_z * layer_size + ny * w + nx;
                        let val = if (combo >> j) & 1 != 0 {
                            10 + lane as u64 + j as u64
                        } else {
                            0
                        };
                        scalar.set_logic_value_by_idx(nidx, val);
                    }
                    scalar.propagate_compact(&ops, &wvia);
                    assert_eq!(
                        lp.get(via_idx, lane),
                        scalar.tilemap.value(via_idx),
                        "{via_type:?} threshold={threshold} lane={lane}"
                    );
                }
            }
        }
    }

    #[test]
    fn preflight_rejects_generic_and_wire() {
        let bad = vec![CompactOp {
            op: COP_GENERIC,
            _pad: [0; 3],
            idx: 64,
            in0: 0,
            in1: 1,
            in2: 2,
        }];
        assert!(LaneKernel::new(&bad, &[], 128).is_err());
        let bad = vec![CompactOp {
            op: COP_WIRE,
            _pad: [0; 3],
            idx: 64,
            in0: 0,
            in1: 1,
            in2: 2,
        }];
        assert!(LaneKernel::new(&bad, &[], 128).is_err());
    }

    #[test]
    fn threshold_via_requires_sim_metadata() {
        let bad = vec![CompactOp {
            op: COP_THRESHOLD_VIA,
            _pad: [0; 3],
            idx: 64,
            in0: 0,
            in1: u32::MAX,
            in2: u32::MAX,
        }];
        assert!(LaneKernel::new(&bad, &[], 128).is_err());
    }
}
