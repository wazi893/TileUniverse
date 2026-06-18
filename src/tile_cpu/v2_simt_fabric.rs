//! SIMT Tier 2b — `V2SimtFabric`: K full V2 CPUs advanced in lane-lockstep.
//!
//! One topology (built once), K `TileCpuV2` clones (per-lane software state),
//! one Simulation whose value storage is in lane mode (Sprint 386 interleaved
//! buffer). Each cycle:
//!
//!   1. per lane: stage X + stage F up to the combined-settle barrier
//!      (per-lane DirtyBitset + clock state swapped in around each lane);
//!   2. ONE wide dense settle across all lanes via `LaneKernel` (the ~73%
//!      eval share) — unless any lane is on an upper-bank Const-swap cycle,
//!      in which case every lane settles scalar (compact ops are stale);
//!   3. per lane: settle-scope dirty drain + stage F post.
//!
//! JIT eval is disabled per lane (it walks stride-1 storage); memoization
//! stays on — each lane's cpu clone owns its caches and sees only its own
//! lane's values through the active-lane redirect.
//!
//! Correctness oracle: a fabric lane running the scalar golden program must
//! produce a byte-identical `hash_v2_final_state` (asserted in tests and the
//! Gate-2b benchmark).

use crate::dirty_bitset::DirtyBitset;
use crate::simulation::Simulation;
use crate::tile_cpu::TileCpuV2;
use crate::tile_cpu::v2_simt_eval::LaneKernel;

pub struct V2SimtFabric {
    pub sim: Simulation,
    cpus: Vec<TileCpuV2>,
    kernel: LaneKernel,
    lane_dirty: Vec<DirtyBitset>,
    lane_clock: Vec<(bool, bool)>, // (global_clock, prev_clock)
    lanes: usize,
    /// Wide settles vs scalar-fallback settle cycles (honest accounting).
    pub wide_settles: u64,
    pub fallback_settles: u64,
}

fn clone_dirty(src: &DirtyBitset, tile_count: usize) -> DirtyBitset {
    let d = DirtyBitset::new(tile_count);
    for (dst, s) in d.segments.iter().zip(src.segments.iter()) {
        dst.set(s.get());
    }
    d
}

impl V2SimtFabric {
    /// Take a freshly built (template) cpu + sim and fan it out to `lanes`
    /// lanes. The template cpu/sim are consumed; lane 0 starts as an exact
    /// copy of the template state, as do all other lanes (SPMD).
    pub fn new(template_cpu: TileCpuV2, mut sim: Simulation, lanes: usize) -> Result<Self, String> {
        assert!(lanes >= 1);
        let (ops, wvia) = template_cpu.settle_compact_view();
        let kernel = LaneKernel::new_from_sim(ops, wvia, &sim)?;

        let tile_count = sim.tilemap.tile_count();
        let mut cpus = Vec::with_capacity(lanes);
        let mut lane_dirty = Vec::with_capacity(lanes);
        let mut lane_clock = Vec::with_capacity(lanes);
        for _ in 0..lanes {
            let mut cpu = template_cpu.clone();
            cpu.fabric_disable_jit();
            cpus.push(cpu);
            lane_dirty.push(clone_dirty(&sim.dirty, tile_count));
            lane_clock.push((sim.global_clock, sim.prev_clock));
        }

        sim.tilemap.enter_lane_mode(lanes);
        Ok(Self {
            sim,
            cpus,
            kernel,
            lane_dirty,
            lane_clock,
            lanes,
            wide_settles: 0,
            fallback_settles: 0,
        })
    }

    #[inline]
    pub fn lanes(&self) -> usize {
        self.lanes
    }

    pub fn cpu(&self, lane: usize) -> &TileCpuV2 {
        &self.cpus[lane]
    }

    /// Point the sim's scalar accessors + dirty/clock state at `lane`.
    fn activate(&mut self, lane: usize) {
        self.sim.tilemap.set_active_lane(lane);
        std::mem::swap(&mut self.sim.dirty, &mut self.lane_dirty[lane]);
        let (gc, pc) = self.lane_clock[lane];
        self.sim.global_clock = gc;
        self.sim.prev_clock = pc;
    }

    /// Store `lane`'s dirty/clock state back after its per-lane phase.
    fn deactivate(&mut self, lane: usize) {
        std::mem::swap(&mut self.sim.dirty, &mut self.lane_dirty[lane]);
        self.lane_clock[lane] = (self.sim.global_clock, self.sim.prev_clock);
    }

    /// Advance every live lane by one cycle. Returns the number of lanes
    /// still running.
    pub fn step_all(&mut self) -> usize {
        let lanes = self.lanes;
        let mut ctxs = Vec::with_capacity(lanes);
        let mut any_inhibit = false;

        // Phase A: per-lane stage X + stage F front (F0..F2).
        for lane in 0..lanes {
            self.activate(lane);
            let ctx = self.cpus[lane].fabric_tick_pre_settle(&mut self.sim);
            if ctx.is_some() && self.cpus[lane].fabric_settle_inhibited() {
                any_inhibit = true;
            }
            self.deactivate(lane);
            ctxs.push(ctx);
        }

        // Phase B: the settle barrier.
        let wide = !any_inhibit && ctxs.iter().any(|c| c.is_some());
        if wide {
            self.wide_settles += 1;
            let (vals, k) = self.sim.tilemap.lane_vals_raw_mut();
            debug_assert_eq!(k, lanes);
            self.kernel.eval_slice(vals, k);
        } else if ctxs.iter().any(|c| c.is_some()) {
            self.fallback_settles += 1;
        }

        // Phase C: per-lane settle finish (drain or scalar settle) + F post.
        let mut live = 0usize;
        for lane in 0..lanes {
            if let Some(ctx) = &ctxs[lane] {
                self.activate(lane);
                self.cpus[lane].fabric_finish_settle(&mut self.sim, ctx, wide);
                self.deactivate(lane);
            }
            if !self.cpus[lane].is_halted() {
                live += 1;
            }
        }
        live
    }

    /// Run until every lane halts or `max_cycles` per lane is reached.
    /// Returns total cycles advanced (summed over lanes).
    pub fn run_to_halt(&mut self, max_cycles: u64) -> u64 {
        let mut cycles_total = 0u64;
        for _ in 0..max_cycles {
            let live_before: u64 = self.cpus.iter().filter(|c| !c.is_halted()).count() as u64;
            if live_before == 0 {
                break;
            }
            self.step_all();
            cycles_total += live_before;
        }
        cycles_total
    }

    /// Read a lane's final state hash (the oracle). Temporarily activates
    /// the lane so the accessors read its slots.
    pub fn lane_state_hash(&mut self, lane: usize) -> u64 {
        self.sim.tilemap.set_active_lane(lane);
        crate::tile_cpu::hash_v2_final_state(&self.cpus[lane], &self.sim)
    }

    /// Read a register from a lane.
    pub fn lane_reg(&mut self, lane: usize, reg: usize) -> u64 {
        self.sim.tilemap.set_active_lane(lane);
        self.cpus[lane].read_reg(&self.sim, reg)
    }
}
