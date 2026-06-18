//! SIMT Tier 3a — GPU lane-packed settle kernel (Sprint 387).
//!
//! One CUDA thread = one lane. Every thread walks the SAME preflighted,
//! zero-row-remapped op program (taken verbatim from `LaneKernel`, so CPU and
//! GPU evaluate the identical stream) sequentially for its own lane. Lanes
//! never touch other lanes' slots — no synchronization anywhere; the
//! iteration loop is device-resident (one sync at the end of a run).
//!
//! Values live in global memory as `vals[slot * K + lane]` (u64): adjacent
//! threads → adjacent addresses → fully coalesced. Op metadata reads are
//! warp-uniform broadcasts.
//!
//! The op program's tile universe is COMPACTED host-side to the ~7k tiles the
//! ops actually touch (`GpuLaneProgram`), so the value buffer is
//! `slots × K × 8 B` (~230 MB at K=4096) instead of the absurd full-grid
//! `1.31 M × K × 8 B`.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use logic_fabric_core::cuda::{CudaError, CudaResult, CudaRuntime};

use crate::simulation::{
    COP_ADD, COP_AND, COP_BITSEL, COP_CARRY, COP_CONST, COP_DEC3, COP_MUX, COP_MUX4, COP_MUX16,
    COP_NOT, COP_OR, COP_RAM, COP_SHL, COP_SHR, COP_SUB, COP_THRESHOLD_VIA, COP_VIA, COP_WIRE_D,
    COP_WIRE_H, COP_WIRE_L, COP_WIRE_R, COP_WIRE_U, COP_WIRE_V, COP_WVIA, COP_XOR, COP_ZERO,
    Simulation,
};
use crate::tile_cpu::v2_simt_eval::LaneKernel;

/// The kernel source, with the COP_* opcode values formatted in from the
/// authoritative crate constants (no drift possible).
fn kernel_source() -> String {
    format!(
        r#"
extern "C" __global__ void simt_lane_settle(
    unsigned long long* __restrict__ vals,      // slots * K, [slot*K + lane]
    const unsigned int* __restrict__ op_code,
    const unsigned int* __restrict__ op_out,    // compact slots
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
    int n_ops,
    int lanes,
    int iters)
{{
    long long lane = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;

    for (int it = 0; it < iters; ++it) {{
        for (int j = 0; j < n_ops; ++j) {{
            unsigned int code = op_code[j];
            unsigned long long v0 = vals[(size_t)op_in0[j] * lanes + lane];
            unsigned long long v1 = vals[(size_t)op_in1[j] * lanes + lane];
            unsigned long long v2 = vals[(size_t)op_in2[j] * lanes + lane];
            size_t o = (size_t)op_out[j] * lanes + lane;
            unsigned long long cur = vals[o];
            unsigned long long r;
            switch (code) {{
                case {wire_r}: case {via}: r = v0; break;
                case {wire_l}: r = v1; break;
                case {wire_d}: case {wire_u}: r = v2; break;
                case {wire_h}: case {cop_or}: r = v0 | v1; break;
                case {wire_v}: r = v1 | v2; break;
                case {cop_and}: r = v0 & v1; break;
                case {cop_xor}: r = v0 ^ v1; break;
                case {cop_mux}: r = (v2 != 0ULL) ? v0 : v1; break;
                case {cop_not}: r = ~v0; break;
                case {cop_zero}: r = (v0 == 0ULL) ? ~0ULL : 0ULL; break;
                case {cop_add}: r = v0 + v1; break;
                case {cop_sub}: r = v0 - v1; break;
                case {cop_shr}: r = v0 >> (v1 & 63ULL); break;
                case {cop_shl}: r = v0 << (v1 & 63ULL); break;
                case {cop_mux16}: {{
                    unsigned int sel = (unsigned int)(v1 & 0xFULL);
                    unsigned long long data = (sel < 8u) ? v0 : v2;
                    r = (data >> ((sel & 7u) * 8u)) & 0xFFULL;
                    break;
                }}
                case {cop_dec3}: r = 1ULL << (v0 & 7ULL); break;
                case {cop_bitsel}: r = ((v0 >> (v1 & 63ULL)) & 1ULL) ? ~0ULL : 0ULL; break;
                case {cop_carry}: r = (v0 > v1) ? ~0ULL : 0ULL; break;
                case {cop_wvia}: r = (v0 >> op_shift[j]) & op_mask[j]; break;
                case {cop_threshold_via}: {{
                    unsigned int active =
                        (vals[(size_t)op_n0[j] * lanes + lane] != 0ULL) +
                        (vals[(size_t)op_n1[j] * lanes + lane] != 0ULL) +
                        (vals[(size_t)op_n2[j] * lanes + lane] != 0ULL) +
                        (vals[(size_t)op_n3[j] * lanes + lane] != 0ULL);
                    r = (active >= op_threshold[j]) ? v0 : 0ULL;
                    break;
                }}
                case {cop_mux4}: {{
                    unsigned int sel = (unsigned int)(v2 & 3ULL);
                    r = (v0 >> (sel * 8u)) & 0xFFULL;
                    break;
                }}
                case {cop_ram}: r = (v2 != 0ULL) ? v0 : cur; break;
                default: r = cur; break;
            }}
            vals[o] = r;
        }}
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

/// The op program with its tile universe renumbered to dense slots
/// (0..n_slots; the last slot is the zero row).
pub struct GpuLaneProgram {
    /// Original tile index per compact slot (zero slot excluded — it has no
    /// original tile).
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
}

impl GpuLaneProgram {
    /// Build from a (preflighted, remapped) `LaneKernel`. CONST ops are
    /// filtered out (the CPU kernel skips them at eval time).
    pub fn from_kernel(kernel: &LaneKernel) -> Self {
        let (ops, wvia, threshold_via) = kernel.remapped_program();
        let zero_row = kernel.tile_count() as u32;

        let mut slot_of: HashMap<u32, u32> = HashMap::new();
        let mut touched: Vec<u32> = Vec::new();
        // The zero row gets the FINAL slot; assign real tiles first.
        let slot = |tile: u32, slot_of: &mut HashMap<u32, u32>, touched: &mut Vec<u32>| -> u32 {
            *slot_of.entry(tile).or_insert_with(|| {
                touched.push(tile);
                (touched.len() - 1) as u32
            })
        };

        let mut op_code = Vec::new();
        let mut op_out = Vec::new();
        let mut op_in0 = Vec::new();
        let mut op_in1 = Vec::new();
        let mut op_in2 = Vec::new();
        let mut op_n0 = Vec::new();
        let mut op_n1 = Vec::new();
        let mut op_n2 = Vec::new();
        let mut op_n3 = Vec::new();
        let mut op_threshold = Vec::new();
        let mut op_shift = Vec::new();
        let mut op_mask = Vec::new();

        // First pass: assign slots for all non-zero-row tiles in op order.
        for ((op, &(wshift, wmask)), tvia) in ops.iter().zip(wvia.iter()).zip(threshold_via.iter())
        {
            if op.op == COP_CONST {
                continue;
            }
            let mut idx_of = |t: u32| -> u32 {
                if t == zero_row {
                    u32::MAX // patched to the zero slot below
                } else {
                    slot(t, &mut slot_of, &mut touched)
                }
            };
            op_out.push(idx_of(op.idx));
            op_in0.push(idx_of(op.in0));
            op_in1.push(idx_of(op.in1));
            op_in2.push(idx_of(op.in2));
            if op.op == COP_THRESHOLD_VIA {
                op_n0.push(idx_of(tvia.neighbors[0]));
                op_n1.push(idx_of(tvia.neighbors[1]));
                op_n2.push(idx_of(tvia.neighbors[2]));
                op_n3.push(idx_of(tvia.neighbors[3]));
                op_threshold.push(tvia.threshold as u32);
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

        // Zero slot is the last one.
        let zero_slot = touched.len() as u32;
        let n_slots = touched.len() + 1;
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
        }
    }

    pub fn op_count(&self) -> usize {
        self.op_code.len()
    }

    /// Host-side lane buffer initialized from a sim's current values,
    /// replicated into every lane: `vals[slot * lanes + lane]`.
    pub fn lanes_from_sim(&self, sim: &Simulation, lanes: usize) -> Vec<u64> {
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

/// Device-side instance: uploaded program + lane values, ready to run.
pub struct GpuLaneSettle {
    func: cudarc::driver::CudaFunction,
    d_vals: cudarc::driver::CudaSlice<u64>,
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
    n_ops: i32,
    lanes: usize,
}

impl GpuLaneSettle {
    pub fn new(
        rt: &CudaRuntime,
        program: &GpuLaneProgram,
        host_vals: &[u64],
        lanes: usize,
    ) -> CudaResult<Self> {
        assert_eq!(host_vals.len(), program.n_slots * lanes);
        use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
        let opts = CompileOptions {
            arch: Some("compute_89"),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(&kernel_source(), opts)
            .map_err(|e| CudaError::KernelCompilationFailed(format!("simt lane: {e:?}")))?;
        let module = rt
            .ctx()
            .load_module(ptx)
            .map_err(|e| CudaError::KernelCompilationFailed(format!("module: {e:?}")))?;
        let func = module
            .load_function("simt_lane_settle")
            .map_err(|e| CudaError::KernelCompilationFailed(format!("function: {e:?}")))?;

        Ok(Self {
            func,
            d_vals: rt.upload(host_vals)?,
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
            n_ops: program.op_count() as i32,
            lanes,
        })
    }

    /// Launch `iters` device-resident settle iterations and synchronize.
    pub fn run(&mut self, rt: &CudaRuntime, iters: i32) -> CudaResult<()> {
        use cudarc::driver::PushKernelArg;
        let threads = 256u32;
        let blocks = (self.lanes as u32).div_ceil(threads);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let lanes = self.lanes as i32;
        unsafe {
            rt.stream()
                .launch_builder(&self.func)
                .arg(&self.d_vals)
                .arg(&self.d_code)
                .arg(&self.d_out)
                .arg(&self.d_in0)
                .arg(&self.d_in1)
                .arg(&self.d_in2)
                .arg(&self.d_n0)
                .arg(&self.d_n1)
                .arg(&self.d_n2)
                .arg(&self.d_n3)
                .arg(&self.d_threshold)
                .arg(&self.d_shift)
                .arg(&self.d_mask)
                .arg(&self.n_ops)
                .arg(&lanes)
                .arg(&iters)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchFailed(format!("simt lane: {e:?}")))?;
        }
        rt.stream()
            .synchronize()
            .map_err(|e| CudaError::LaunchFailed(format!("sync: {e:?}")))?;
        Ok(())
    }

    /// Download the full lane buffer (`slot * lanes + lane`).
    pub fn download(&self, rt: &CudaRuntime) -> CudaResult<Vec<u64>> {
        rt.download(&self.d_vals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{COP_THRESHOLD_VIA, Simulation};
    use crate::tile_cpu::v2_simt_eval::LanePack;
    use crate::tile_meta::TileType;

    #[derive(Clone, Copy, Debug)]
    struct ThresholdViaCase {
        x: usize,
        y: usize,
        via_z: usize,
        via_type: TileType,
        threshold: u8,
    }

    fn idx(width: usize, height: usize, x: usize, y: usize, z: usize) -> usize {
        z * width * height + y * width + x
    }

    fn source_z(case: ThresholdViaCase) -> usize {
        match case.via_type {
            TileType::ThresholdViaUp => case.via_z + 1,
            TileType::ThresholdViaDown => case.via_z - 1,
            other => panic!("not a threshold via type: {other:?}"),
        }
    }

    fn neighbor_positions(
        width: usize,
        height: usize,
        case: ThresholdViaCase,
    ) -> [Option<(usize, usize)>; 4] {
        [
            case.x.checked_sub(1).map(|x| (x, case.y)),
            if case.x + 1 < width {
                Some((case.x + 1, case.y))
            } else {
                None
            },
            case.y.checked_sub(1).map(|y| (case.x, y)),
            if case.y + 1 < height {
                Some((case.x, case.y + 1))
            } else {
                None
            },
        ]
    }

    fn seed_lane_values(
        sim: &Simulation,
        cases: &[ThresholdViaCase],
        lanes: usize,
        lp: &mut LanePack,
    ) {
        let width = sim.width();
        let height = sim.height();
        for (case_idx, &case) in cases.iter().enumerate() {
            let via_idx = idx(width, height, case.x, case.y, case.via_z);
            let src_idx = idx(width, height, case.x, case.y, source_z(case));
            let neighbors = neighbor_positions(width, height, case);
            for lane in 0..lanes {
                let combo = (lane & 0xF) as u32;
                let source = if lane < 16 {
                    0xA5A5_0000_0000_0000u64
                        ^ ((case_idx as u64) << 32)
                        ^ ((lane as u64) << 8)
                        ^ case.threshold as u64
                } else {
                    0
                };
                lp.set(src_idx, lane, source);
                lp.set(via_idx, lane, 0);
                for (bit, pos) in neighbors.iter().enumerate() {
                    if let Some((nx, ny)) = pos {
                        let nidx = idx(width, height, *nx, *ny, case.via_z);
                        let value = if ((combo >> bit) & 1) != 0 {
                            ((case_idx as u64 + 1) << 16) | lane as u64 | 1
                        } else {
                            0
                        };
                        lp.set(nidx, lane, value);
                    }
                }
            }
        }
    }

    fn expected_threshold_via(
        sim: &Simulation,
        case: ThresholdViaCase,
        lane: usize,
        case_idx: usize,
    ) -> u64 {
        let width = sim.width();
        let height = sim.height();
        let combo = (lane & 0xF) as u32;
        let source = if lane < 16 {
            0xA5A5_0000_0000_0000u64
                ^ ((case_idx as u64) << 32)
                ^ ((lane as u64) << 8)
                ^ case.threshold as u64
        } else {
            0
        };
        let active = neighbor_positions(width, height, case)
            .iter()
            .enumerate()
            .filter(|(bit, pos)| pos.is_some() && ((combo >> bit) & 1) != 0)
            .count() as u8;
        if active >= case.threshold { source } else { 0 }
    }

    #[test]
    fn threshold_via_gpu_settle_matches_cpu_lane_kernel() -> Result<(), Box<dyn std::error::Error>>
    {
        if !logic_fabric_core::cuda::is_nvrtc_available() {
            eprintln!("CUDA/NVRTC not available, skipping threshold-via GPU parity test");
            return Ok(());
        }
        let rt = match CudaRuntime::new() {
            Ok(rt) => rt,
            Err(_) => {
                eprintln!("CUDA not available, skipping threshold-via GPU parity test");
                return Ok(());
            }
        };

        const LANES: usize = 32;
        const ITERS: i32 = 3;
        let width = 12usize;
        let height = 12usize;
        let mut sim = Simulation::with_size_layered(width, height, 2);

        let mut cases = Vec::new();
        let up_positions = [(2, 2), (5, 2), (8, 2), (2, 5), (5, 5)];
        let down_positions = [(8, 5), (2, 8), (5, 8), (8, 8), (0, 0)];
        for (threshold, &(x, y)) in (0u8..=4).zip(up_positions.iter()) {
            cases.push(ThresholdViaCase {
                x,
                y,
                via_z: 0,
                via_type: TileType::ThresholdViaUp,
                threshold,
            });
        }
        for (threshold, &(x, y)) in (0u8..=4).zip(down_positions.iter()) {
            cases.push(ThresholdViaCase {
                x,
                y,
                via_z: 1,
                via_type: TileType::ThresholdViaDown,
                threshold,
            });
        }

        let mut eval_order = Vec::new();
        for &case in &cases {
            let via_idx = idx(width, height, case.x, case.y, case.via_z);
            let src_idx = idx(width, height, case.x, case.y, source_z(case));
            sim.set_tile_3d(case.x, case.y, case.via_z, case.via_type);
            sim.set_tile_threshold(via_idx, case.threshold);
            sim.set_tile_3d(case.x, case.y, source_z(case), TileType::Const);
            for pos in neighbor_positions(width, height, case).iter().flatten() {
                sim.set_tile_3d(pos.0, pos.1, case.via_z, TileType::Const);
            }
            sim.set_logic_value_by_idx(src_idx, 0);
            eval_order.push(via_idx);
        }
        sim.rebuild_via_connections();

        let (ops, wvia) = sim.build_compact_ops(&eval_order);
        assert!(
            ops.iter().all(|op| op.op == COP_THRESHOLD_VIA),
            "test corpus should isolate threshold-via compact ops"
        );
        let kernel = LaneKernel::new_from_sim(&ops, &wvia, &sim)?;
        let program = GpuLaneProgram::from_kernel(&kernel);

        let mut cpu_lp = LanePack::from_sim(&sim, LANES);
        seed_lane_values(&sim, &cases, LANES, &mut cpu_lp);

        let mut host = vec![0u64; program.n_slots * LANES];
        for (slot, &tile) in program.touched.iter().enumerate() {
            for lane in 0..LANES {
                host[slot * LANES + lane] = cpu_lp.get(tile as usize, lane);
            }
        }

        for _ in 0..ITERS {
            kernel.eval(&mut cpu_lp);
        }

        let mut gpu = GpuLaneSettle::new(&rt, &program, &host, LANES)?;
        gpu.run(&rt, ITERS)?;
        let out = gpu.download(&rt)?;

        for (slot, &tile) in program.touched.iter().enumerate() {
            for lane in 0..LANES {
                assert_eq!(
                    out[slot * LANES + lane],
                    cpu_lp.get(tile as usize, lane),
                    "GPU diverged from CPU lane kernel at tile {tile} slot {slot} lane {lane}"
                );
            }
        }

        for (case_idx, &case) in cases.iter().enumerate() {
            let via_idx = idx(width, height, case.x, case.y, case.via_z);
            for lane in 0..LANES {
                assert_eq!(
                    cpu_lp.get(via_idx, lane),
                    expected_threshold_via(&sim, case, lane, case_idx),
                    "{:?} threshold={} lane={}",
                    case.via_type,
                    case.threshold,
                    lane
                );
            }
        }

        Ok(())
    }
}
