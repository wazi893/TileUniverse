//! The AlphaFabric placement environment.
//!
//! A [`PlacementEnv`] wraps a [`Circuit`] in a **slot-grid canvas**: a row-major
//! lattice of legal positions at the placer's pitch. Gates occupy distinct slots,
//! so non-overlap is guaranteed by construction and the actions are simple:
//!
//! - [`relocate`](PlacementEnv::relocate) — move a gate to an empty slot.
//! - [`swap`](PlacementEnv::swap) — exchange two gates' slots (always legal).
//! - [`reset`](PlacementEnv::reset) — return to the row-major baseline.
//! - [`score`](PlacementEnv::score) — route the current layout and cost it.
//!
//! The baseline assignment (gate `g` -> slot `g`) reproduces the deterministic
//! row-major placement of [`place_mapped_netlist`], which is the layout the
//! later SA / learned placers must beat.

use crate::synth::aig::Aig;
use crate::synth::mapping::evaluate_aig;
use crate::synth::placement::{
    GatePlacement, PinSide, PlaceConfig, PlacedNetlist, PlacementError, PlacementRegion,
    place_mapped_netlist,
};
use crate::synth::routing::{RouteConfig, route_placed_netlist};
use crate::synth::timing::{TimingConstraint, analyze_timing};
use crate::tiles::tile_meta::TileType;

use super::circuit::Circuit;
use super::reward::{PlacementMetrics, PlacementScore, RewardWeights};

/// Per-gate metadata that is independent of where the gate is placed.
#[derive(Clone, Debug)]
struct GateMeta {
    tile_type: TileType,
    input_pins: Vec<PinSide>,
    output_pin: PinSide,
}

/// A slot grid. Gates occupy distinct slots; slot `s` sits at a fixed pitch.
#[derive(Clone, Copy, Debug)]
pub struct Canvas {
    pub min_x: usize,
    pub min_y: usize,
    pub z: usize,
    pub halo: usize,
    pub pitch: usize,
    pub rows: usize,
    pub cols: usize,
    pub max_z: usize,
}

impl Canvas {
    /// Total number of legal slots.
    pub fn num_slots(&self) -> usize {
        self.rows * self.cols
    }

    /// Grid coordinate of a slot's gate center.
    pub fn slot_coord(&self, slot: usize) -> (usize, usize) {
        let row = slot / self.cols;
        let col = slot % self.cols;
        (
            self.min_x + self.halo + col * self.pitch,
            self.min_y + self.halo + row * self.pitch,
        )
    }

    /// The placement region spanned by the canvas.
    fn region(&self) -> PlacementRegion {
        PlacementRegion {
            min_x: self.min_x,
            min_y: self.min_y,
            width: self.cols * self.pitch,
            height: self.rows * self.pitch,
            z: self.z,
            max_z: self.max_z,
        }
    }
}

/// Errors from environment actions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnvError {
    Placement(PlacementError),
    GateOutOfRange { gate: usize },
    SlotOutOfRange { slot: usize },
    SlotOccupied { slot: usize },
}

impl From<PlacementError> for EnvError {
    fn from(e: PlacementError) -> Self {
        EnvError::Placement(e)
    }
}

impl std::fmt::Display for EnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnvError::Placement(e) => write!(f, "placement error: {e}"),
            EnvError::GateOutOfRange { gate } => write!(f, "gate index {gate} out of range"),
            EnvError::SlotOutOfRange { slot } => write!(f, "slot index {slot} out of range"),
            EnvError::SlotOccupied { slot } => write!(f, "slot {slot} already occupied"),
        }
    }
}

impl std::error::Error for EnvError {}

/// A placement-optimization environment over one circuit.
pub struct PlacementEnv<'c> {
    circuit: &'c Circuit,
    weights: RewardWeights,
    route_config: RouteConfig,
    canvas: Canvas,
    gate_meta: Vec<GateMeta>,
    /// `slot_of_gate[g]` = the slot currently holding gate `g`.
    slot_of_gate: Vec<usize>,
    /// `gate_of_slot[s]` = `Some(g)` if slot `s` holds gate `g`, else `None`.
    gate_of_slot: Vec<Option<usize>>,
    baseline_slot_of_gate: Vec<usize>,
    /// Net hyperedges as gate-index sets (driver + consumer gates), for the
    /// cheap half-perimeter wirelength estimate. Nets touching < 2 gates are
    /// dropped (they contribute nothing to a gate-only bounding box).
    hyperedges: Vec<Vec<usize>>,
    /// Per-hyperedge timing criticality in `[0, 1]`, parallel to `hyperedges`,
    /// derived once from STA slack (1.0 = worst-slack/most-critical net). Used to
    /// weight net wirelength in the placement-dependent timing term.
    net_crit: Vec<f64>,
    /// Placement-invariant timing, computed once.
    circuit_delay: f64,
    area_cells: usize,
}

impl<'c> PlacementEnv<'c> {
    /// Build an environment with default placement/routing presets and reward
    /// weights. Routing forbids single-layer crossings (`Cross` tiles use a
    /// bit-split encoding that does not export to boolean tiles) and detours via
    /// layers instead; four layers (`max_z = 3`) keep larger blocks routable.
    /// This is the config under which exported layouts verify on real tiles.
    pub fn new(circuit: &'c Circuit) -> Result<Self, EnvError> {
        let route_config = RouteConfig {
            prefer_horizontal_first: true,
            no_crossings: true,
            max_z: 3,
        };
        Self::with_config(
            circuit,
            PlaceConfig::standalone(),
            route_config,
            RewardWeights::default(),
        )
    }

    /// Build an environment with explicit configuration.
    pub fn with_config(
        circuit: &'c Circuit,
        place_config: PlaceConfig,
        route_config: RouteConfig,
        weights: RewardWeights,
    ) -> Result<Self, EnvError> {
        let netlist = &circuit.netlist;
        let n = netlist.num_gates();

        // Baseline placement validates each gate and yields its (placement-
        // invariant) pin metadata. We reuse the metadata; coordinates are
        // recomputed from the canvas slot grid.
        let baseline = place_mapped_netlist(netlist, &circuit.lib, &place_config)?;
        let mut meta: Vec<Option<GateMeta>> = vec![None; n];
        for g in &baseline.gates {
            meta[g.gate_idx] = Some(GateMeta {
                tile_type: g.tile_type,
                input_pins: g.input_pins.clone(),
                output_pin: g.output_pin,
            });
        }
        let gate_meta: Vec<GateMeta> = meta
            .into_iter()
            .map(|m| m.expect("baseline places every gate"))
            .collect();

        // Canvas: row-major slot grid with one extra row and column of slack so
        // there is room to relocate/swap.
        let halo = place_config.halo.max(1);
        let pitch = halo * 2 + 2;
        let cols = auto_cols(n) + 1;
        let rows = n.div_ceil(cols) + 1;
        let canvas = Canvas {
            min_x: place_config.min_x,
            min_y: place_config.min_y,
            z: place_config.z,
            halo,
            pitch,
            rows,
            cols,
            max_z: route_config.max_z.max(place_config.z),
        };

        let slot_of_gate: Vec<usize> = (0..n).collect();
        let mut gate_of_slot = vec![None; canvas.num_slots()];
        for (g, &s) in slot_of_gate.iter().enumerate() {
            gate_of_slot[s] = Some(g);
        }

        let report = analyze_timing(netlist, &circuit.lib, &TimingConstraint::intrinsic());

        // Build net hyperedges (driver gate + consumer gates) for the HPWL proxy.
        let num_pi = netlist.primary_inputs.len();
        let mut touch: Vec<Vec<usize>> = vec![Vec::new(); netlist.num_nets as usize];
        for g in 0..n {
            touch[num_pi + g].push(g); // gate g drives net (num_pi + g)
        }
        for (g, gate) in netlist.gates.iter().enumerate() {
            for &fanin in &gate.fanins {
                touch[fanin as usize].push(g);
            }
        }
        // Keep only multi-gate nets (the HPWL proxy), carrying each net's timing
        // criticality (from STA slack) alongside its gate set.
        let mut hyperedges: Vec<Vec<usize>> = Vec::new();
        let mut net_crit: Vec<f64> = Vec::new();
        for (net_id, gates) in touch.into_iter().enumerate() {
            if gates.len() >= 2 {
                net_crit.push(net_criticality(
                    report.slack(net_id as u32),
                    report.worst_slack,
                ));
                hyperedges.push(gates);
            }
        }

        Ok(Self {
            circuit,
            weights,
            route_config,
            canvas,
            gate_meta,
            baseline_slot_of_gate: slot_of_gate.clone(),
            slot_of_gate,
            gate_of_slot,
            hyperedges,
            net_crit,
            circuit_delay: report.circuit_delay,
            area_cells: report.area_cells,
        })
    }

    /// The circuit being placed.
    pub fn circuit(&self) -> &Circuit {
        self.circuit
    }

    /// Number of gates to place.
    pub fn num_gates(&self) -> usize {
        self.gate_meta.len()
    }

    /// The slot-grid canvas.
    pub fn canvas(&self) -> Canvas {
        self.canvas
    }

    /// Current `slot_of_gate` assignment (read-only).
    pub fn assignment(&self) -> &[usize] {
        &self.slot_of_gate
    }

    /// Net hyperedges as gate-index sets (driver + consumer gates). Used by the
    /// learned placer to derive gate adjacency and connectivity features.
    pub fn hyperedges(&self) -> &[Vec<usize>] {
        &self.hyperedges
    }

    /// Slots not currently holding a gate.
    pub fn empty_slots(&self) -> Vec<usize> {
        self.gate_of_slot
            .iter()
            .enumerate()
            .filter_map(|(s, g)| g.is_none().then_some(s))
            .collect()
    }

    /// Reset to the deterministic row-major baseline placement.
    pub fn reset(&mut self) {
        self.slot_of_gate
            .copy_from_slice(&self.baseline_slot_of_gate);
        for s in self.gate_of_slot.iter_mut() {
            *s = None;
        }
        for (g, &s) in self.slot_of_gate.iter().enumerate() {
            self.gate_of_slot[s] = Some(g);
        }
    }

    /// Overwrite the current assignment (e.g. to restore the best layout found
    /// by an optimizer). `slot_of_gate` must have one distinct slot per gate.
    pub fn restore_assignment(&mut self, slot_of_gate: &[usize]) {
        debug_assert_eq!(slot_of_gate.len(), self.num_gates());
        self.slot_of_gate.copy_from_slice(slot_of_gate);
        for s in self.gate_of_slot.iter_mut() {
            *s = None;
        }
        for (g, &s) in self.slot_of_gate.iter().enumerate() {
            self.gate_of_slot[s] = Some(g);
        }
    }

    /// Move `gate` to an empty `slot`. No-op if it already sits there.
    pub fn relocate(&mut self, gate: usize, slot: usize) -> Result<(), EnvError> {
        if gate >= self.num_gates() {
            return Err(EnvError::GateOutOfRange { gate });
        }
        if slot >= self.canvas.num_slots() {
            return Err(EnvError::SlotOutOfRange { slot });
        }
        match self.gate_of_slot[slot] {
            None => {}
            Some(g) if g == gate => return Ok(()),
            Some(_) => return Err(EnvError::SlotOccupied { slot }),
        }
        let old = self.slot_of_gate[gate];
        self.gate_of_slot[old] = None;
        self.gate_of_slot[slot] = Some(gate);
        self.slot_of_gate[gate] = slot;
        Ok(())
    }

    /// Exchange the slots of two gates. Always legal (it is a permutation).
    pub fn swap(&mut self, a: usize, b: usize) -> Result<(), EnvError> {
        if a >= self.num_gates() {
            return Err(EnvError::GateOutOfRange { gate: a });
        }
        if b >= self.num_gates() {
            return Err(EnvError::GateOutOfRange { gate: b });
        }
        if a == b {
            return Ok(());
        }
        let (sa, sb) = (self.slot_of_gate[a], self.slot_of_gate[b]);
        self.slot_of_gate[a] = sb;
        self.slot_of_gate[b] = sa;
        self.gate_of_slot[sa] = Some(b);
        self.gate_of_slot[sb] = Some(a);
        Ok(())
    }

    /// Materialize the current assignment as a `PlacedNetlist` (unrouted).
    pub fn placed(&self) -> PlacedNetlist {
        let mut gates = Vec::with_capacity(self.num_gates());
        for (g, meta) in self.gate_meta.iter().enumerate() {
            let (x, y) = self.canvas.slot_coord(self.slot_of_gate[g]);
            gates.push(GatePlacement {
                gate_idx: g,
                tile_type: meta.tile_type,
                x,
                y,
                z: self.canvas.z,
                input_pins: meta.input_pins.clone(),
                output_pin: meta.output_pin,
            });
        }
        PlacedNetlist {
            region: self.canvas.region(),
            halo: self.canvas.halo,
            gates,
            routes: Vec::new(),
            input_anchors: Vec::new(),
            output_anchors: Vec::new(),
        }
    }

    /// Cheap half-perimeter wirelength (HPWL) of the current placement: the sum
    /// over net hyperedges of their gate-center bounding-box half-perimeter.
    ///
    /// This is the canonical placement proxy — O(pins), no routing — and it
    /// correlates strongly with routed wirelength, so an optimizer can anneal
    /// against it and route only to validate. Gate-only (PI/PO terminals are
    /// router-assigned and excluded).
    pub fn hpwl(&self) -> usize {
        let mut total = 0usize;
        for edge in &self.hyperedges {
            let mut min_x = usize::MAX;
            let mut max_x = 0usize;
            let mut min_y = usize::MAX;
            let mut max_y = 0usize;
            for &g in edge {
                let (x, y) = self.canvas.slot_coord(self.slot_of_gate[g]);
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
            total += (max_x - min_x) + (max_y - min_y);
        }
        total
    }

    /// Criticality-weighted wirelength: Σ over nets of (timing criticality × net
    /// HPWL). Same gate-only bounding boxes as [`hpwl`](Self::hpwl), but each net
    /// is scaled by its STA-derived criticality, so long *timing-critical* nets
    /// dominate. This is the placement-dependent timing term — minimizing it
    /// pulls low-slack nets short. O(pins), no routing.
    pub fn timing_cost(&self) -> f64 {
        let mut total = 0.0f64;
        for (edge, &crit) in self.hyperedges.iter().zip(&self.net_crit) {
            if crit == 0.0 {
                continue;
            }
            let mut min_x = usize::MAX;
            let mut max_x = 0usize;
            let mut min_y = usize::MAX;
            let mut max_y = 0usize;
            for &g in edge {
                let (x, y) = self.canvas.slot_coord(self.slot_of_gate[g]);
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
            let hp = (max_x - min_x) + (max_y - min_y);
            total += crit * hp as f64;
        }
        total
    }

    /// Route the current placement and score it. Lower cost is better;
    /// unroutable layouts pay the flat penalty.
    pub fn score(&self) -> PlacementScore {
        let placed = self.placed();
        let mut metrics =
            match route_placed_netlist(&self.circuit.netlist, &placed, &self.route_config) {
                Ok(routed) => {
                    PlacementMetrics::from_routed(&routed, self.circuit_delay, self.area_cells)
                }
                Err(_) => PlacementMetrics::unroutable(
                    placed.region.width * placed.region.height,
                    self.circuit_delay,
                    self.area_cells,
                ),
            };
        // The bare tallies leave the timing term 0; fill it from the precomputed
        // per-net criticality weights (placement-dependent, route-independent).
        metrics.timing_cost = self.timing_cost();
        self.weights.score(metrics)
    }

    /// Physically verify the current placement on real tiles (tier-2 legality).
    pub fn verify_physical(&self) -> bool {
        verify_physical(self.circuit, &self.placed(), &self.route_config)
    }

    /// Diagnose the current placement's physical-legality tier, splitting the
    /// [`verify_physical`](Self::verify_physical) `false` into its cause.
    pub fn diagnose_physical(&self) -> PhysicalDiagnosis {
        diagnose_physical(self.circuit, &self.placed(), &self.route_config)
    }
}

/// Timing criticality of a net in `[0, 1]` from its STA slack. The worst-slack
/// (tightest) net is fully critical (1.0); criticality decays smoothly as slack
/// grows. Unconstrained nets (infinite slack) are not critical (0.0). The unit
/// decay band keeps the kernel scale-free and deterministic; it is a net
/// *weight*, not a calibrated delay.
fn net_criticality(slack: f64, worst_slack: f64) -> f64 {
    if !slack.is_finite() {
        return 0.0;
    }
    1.0 / (1.0 + (slack - worst_slack).max(0.0))
}

/// Square-ish column count for `n` gates (matches the row placer's heuristic).
fn auto_cols(num_gates: usize) -> usize {
    let target = num_gates.max(1);
    let mut cols = 1usize;
    while cols * cols < target {
        cols += 1;
    }
    cols
}

/// Which physical-legality tier a placed layout reaches — the failure causes
/// that [`verify_physical`] collapses into one `false`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicalDiagnosis {
    /// The router could not connect every net.
    Unroutable,
    /// Routed, but the exported tile evaluation never settled (no fixed point).
    NotConverged,
    /// Routed and settled, but a tile output disagreed with the AIG — a genuine
    /// "routable but functionally wrong" layout (would break `routable ⇒ correct`).
    WrongOutput,
    /// Routed, settled, and every checked vector matched the AIG.
    Correct,
}

/// Diagnose a placed layout's physical legality, distinguishing the causes
/// [`verify_physical`] flattens into a single `false`: routing failure,
/// non-convergence of the tile evaluation, or a wrong output. Same routing,
/// export, and vector set as [`verify_physical`]; convergence is checked before
/// output, since a non-settled evaluation makes the output meaningless.
pub fn diagnose_physical(
    circuit: &Circuit,
    placed: &PlacedNetlist,
    route_config: &RouteConfig,
) -> PhysicalDiagnosis {
    use crate::synth::export::{evaluate_exported, export_to_simulation};

    let routed = match route_placed_netlist(&circuit.netlist, placed, route_config) {
        Ok(r) => r,
        Err(_) => return PhysicalDiagnosis::Unroutable,
    };
    let mut export = export_to_simulation(&routed, &circuit.netlist);
    let n = circuit.aig.num_inputs() as usize;
    for inputs in test_vectors(&circuit.aig, n) {
        let got = evaluate_exported(&mut export, &inputs);
        let expected = evaluate_aig(&circuit.aig, &inputs);
        if !export.last_converged {
            return PhysicalDiagnosis::NotConverged;
        }
        if got != expected {
            return PhysicalDiagnosis::WrongOutput;
        }
    }
    PhysicalDiagnosis::Correct
}

/// Physically verify a placed layout: route it, export to real tiles, and check
/// the truth table against the AIG. Exhaustive for small input counts, sampled
/// deterministically above that. Returns `false` if routing fails, a vector
/// disagrees, or the tile evaluation fails to converge.
///
/// This is the strongest legality gate — it proves a *re-placed* circuit still
/// computes correctly on the fabric, honoring the physical-authority standard.
/// See [`diagnose_physical`] for the specific failure cause.
pub fn verify_physical(
    circuit: &Circuit,
    placed: &PlacedNetlist,
    route_config: &RouteConfig,
) -> bool {
    matches!(
        diagnose_physical(circuit, placed, route_config),
        PhysicalDiagnosis::Correct
    )
}

/// Input vectors for physical verification: exhaustive for <= 12 inputs, else a
/// deterministic stride sample.
fn test_vectors(_aig: &Aig, n: usize) -> Vec<Vec<bool>> {
    if n <= 12 {
        (0..(1u32 << n))
            .map(|a| (0..n).map(|i| (a >> i) & 1 != 0).collect())
            .collect()
    } else {
        let total: u64 = 1u64 << n.min(63);
        let samples: u64 = 1024;
        let stride = (total / samples).max(1);
        (0..samples)
            .map(|k| {
                let a = k.wrapping_mul(stride);
                (0..n).map(|i| (a >> i) & 1 != 0).collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::alphafabric::corpus::build_ripple_adder;

    fn adder4() -> Circuit {
        Circuit::from_aig("adder4", build_ripple_adder(4))
    }

    #[test]
    fn baseline_is_measurable_and_deterministic() {
        let c = adder4();
        let env = PlacementEnv::new(&c).expect("env builds");
        let s1 = env.score();
        let s2 = env.score();
        assert!(s1.metrics.routable, "baseline adder4 should route");
        assert!(
            s1.cost < env_default_penalty(),
            "routable cost should be far below the unroutable penalty"
        );
        assert_eq!(s1, s2, "scoring must be deterministic");
        assert!(s1.metrics.wire_tiles > 0);
    }

    fn env_default_penalty() -> f64 {
        RewardWeights::default().unroutable_penalty
    }

    #[test]
    fn reset_restores_baseline() {
        let c = adder4();
        let mut env = PlacementEnv::new(&c).expect("env builds");
        let baseline: Vec<usize> = env.assignment().to_vec();
        env.swap(0, 1).unwrap();
        assert_ne!(env.assignment(), baseline.as_slice());
        env.reset();
        assert_eq!(env.assignment(), baseline.as_slice());
    }

    #[test]
    fn relocate_respects_occupancy() {
        let c = adder4();
        let mut env = PlacementEnv::new(&c).expect("env builds");
        let empty = env.empty_slots();
        assert!(!empty.is_empty(), "canvas should have slack slots");
        let target = empty[0];
        // Moving gate 0 to an empty slot succeeds.
        env.relocate(0, target).unwrap();
        assert_eq!(env.assignment()[0], target);
        // Another gate cannot take the now-occupied slot.
        assert_eq!(
            env.relocate(1, target),
            Err(EnvError::SlotOccupied { slot: target })
        );
    }

    #[test]
    fn mapping_is_correct_for_adder4() {
        assert!(adder4().mapping_is_correct());
    }

    #[test]
    fn baseline_is_physically_correct() {
        let c = adder4();
        let env = PlacementEnv::new(&c).expect("env builds");
        assert!(
            env.verify_physical(),
            "baseline placement must compute the adder on real tiles"
        );
    }

    #[test]
    fn diagnose_physical_is_correct_for_baseline() {
        // The diagnostic refinement of verify_physical: a legal baseline reports
        // Correct (and verify_physical agrees). Cheap (adder4).
        let c = adder4();
        let env = PlacementEnv::new(&c).expect("env builds");
        assert_eq!(env.diagnose_physical(), PhysicalDiagnosis::Correct);
        assert!(env.verify_physical());
    }

    #[test]
    fn routable_implies_correct_after_swap() {
        // The core legality invariant: any layout the router accepts must still
        // compute correctly on the fabric, even after a placement perturbation.
        let c = adder4();
        let mut env = PlacementEnv::new(&c).expect("env builds");
        env.swap(0, 1).unwrap();
        if env.score().metrics.routable {
            assert!(
                env.verify_physical(),
                "a routable re-placement must be physically correct"
            );
        }
    }

    #[test]
    fn timing_cost_is_deterministic_and_positive() {
        let c = adder4();
        let env = PlacementEnv::new(&c).expect("env builds");
        let t1 = env.timing_cost();
        let t2 = env.timing_cost();
        assert_eq!(t1, t2, "timing_cost must be deterministic");
        assert!(t1 >= 0.0, "timing_cost is non-negative");
        assert!(
            t1 > 0.0,
            "adder4 has timing-critical multi-slot nets, so the term is live"
        );
        // It is also surfaced on the scored metrics.
        assert_eq!(env.score().metrics.timing_cost, t1);
    }

    #[test]
    fn default_weights_ignore_timing_but_weight_adds_it() {
        let c = adder4();
        let env = PlacementEnv::new(&c).expect("env builds");
        let m = env.score().metrics;
        assert!(m.routable);

        // Default weights (timing = 0) reproduce the legacy cost exactly: adding
        // the timing field must not perturb existing scores.
        let base = RewardWeights::default();
        let legacy = base.wire * m.wire_tiles as f64
            + base.via * m.vias as f64
            + base.crossing * m.crossings as f64
            + base.area * m.canvas_area as f64;
        assert_eq!(base.cost(&m), legacy);

        // A positive timing weight adds exactly weight * timing_cost.
        let timed = RewardWeights {
            timing: 3.0,
            ..RewardWeights::default()
        };
        assert!((timed.cost(&m) - (legacy + 3.0 * m.timing_cost)).abs() < 1e-9);
    }

    #[test]
    fn timing_cost_responds_to_placement() {
        // The term is placement-dependent: at least one gate swap must change it.
        let c = adder4();
        let mut env = PlacementEnv::new(&c).expect("env builds");
        let base = env.timing_cost();
        let n = env.num_gates();
        let mut changed = false;
        for i in 0..n {
            for j in (i + 1)..n {
                env.swap(i, j).unwrap();
                if (env.timing_cost() - base).abs() > 1e-9 {
                    changed = true;
                }
                env.swap(i, j).unwrap(); // swap is its own inverse — restore
            }
        }
        assert!(
            changed,
            "timing_cost must respond to placement (some swap changes it)"
        );
    }
}
