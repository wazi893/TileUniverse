//! Per-layout quality metrics and the scalar cost the optimizer minimizes.
//!
//! Placement moves wirelength, vias, crossings, and routability — *not* logic
//! depth (that is fixed by the mapped netlist). So the cost is built from the
//! placement-dependent terms; `circuit_delay` and `area_cells` are carried for
//! reporting but are placement-invariant and weighted 0 by default.
//!
//! ## Timing-aware term (`timing_cost`)
//!
//! The combinational `circuit_delay` cannot be improved by placement, but the
//! *physical* critical-path delay can: longer routes on timing-critical nets add
//! wire delay. The standard timing-driven-placement objective captures this
//! without re-running STA per layout — **criticality-weighted wirelength**:
//! weight each net's length by its timing criticality (derived from STA slack,
//! computed once), so pulling low-slack nets short is rewarded. This term *is*
//! placement-dependent; it is opt-in via [`RewardWeights::timing`] (weighted 0
//! by default, so the legacy cost is unchanged).

use crate::synth::placement::PlacedNetlist;
use crate::tiles::tile_meta::TileType;

/// Quality-of-result metrics for one placed (and routed) layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacementMetrics {
    /// Whether the router connected every net.
    pub routable: bool,
    /// Total routed wire tiles — the primary placement objective.
    pub wire_tiles: usize,
    /// Layer-transition tiles (`ViaUp` / `ViaDown`).
    pub vias: usize,
    /// Wire-crossing tiles (`Cross`).
    pub crossings: usize,
    /// Placement canvas area in tiles (`region.width * region.height`).
    pub canvas_area: usize,
    /// Combinational delay of the mapped circuit. Placement-invariant; reported.
    pub circuit_delay: f64,
    /// Cell count (area proxy). Placement-invariant; reported.
    pub area_cells: usize,
    /// Criticality-weighted wirelength: Σ over nets of (STA-slack-derived
    /// criticality × net HPWL). A placement-**dependent** timing proxy — pulling
    /// timing-critical nets short lowers it. The bare [`from_routed`] tally has
    /// no netlist-slack context and leaves this 0; the environment fills it from
    /// its precomputed per-net criticality weights. Opt in via
    /// [`RewardWeights::timing`].
    ///
    /// [`from_routed`]: PlacementMetrics::from_routed
    pub timing_cost: f64,
}

impl PlacementMetrics {
    /// Metrics for a layout the router could not connect.
    pub fn unroutable(canvas_area: usize, circuit_delay: f64, area_cells: usize) -> Self {
        Self {
            routable: false,
            wire_tiles: 0,
            vias: 0,
            crossings: 0,
            canvas_area,
            circuit_delay,
            area_cells,
            timing_cost: 0.0,
        }
    }

    /// Tally routed-tile metrics from a successfully routed netlist.
    ///
    /// `timing_cost` is left 0 here (this tally has no slack context); the
    /// environment fills it in [`PlacementEnv::score`](super::env::PlacementEnv::score).
    pub fn from_routed(routed: &PlacedNetlist, circuit_delay: f64, area_cells: usize) -> Self {
        let mut wire_tiles = 0usize;
        let mut vias = 0usize;
        let mut crossings = 0usize;
        for seg in &routed.routes {
            wire_tiles += 1;
            match seg.tile_type {
                TileType::ViaUp | TileType::ViaDown => vias += 1,
                TileType::Cross => crossings += 1,
                _ => {}
            }
        }
        Self {
            routable: true,
            wire_tiles,
            vias,
            crossings,
            canvas_area: routed.region.width * routed.region.height,
            circuit_delay,
            area_cells,
            timing_cost: 0.0,
        }
    }
}

/// Weights for the scalar cost. Lower cost is better.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardWeights {
    pub wire: f64,
    pub via: f64,
    pub crossing: f64,
    pub area: f64,
    /// Weight on the criticality-weighted-wirelength timing term
    /// ([`PlacementMetrics::timing_cost`]). 0 by default — turning it positive
    /// makes the optimizer timing-driven (pull low-slack nets short).
    pub timing: f64,
    /// Flat cost charged when a layout fails to route, so any routable layout is
    /// strictly preferred to any unroutable one.
    pub unroutable_penalty: f64,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            wire: 1.0,
            via: 2.0,
            crossing: 4.0,
            area: 0.0,
            timing: 0.0,
            unroutable_penalty: 1.0e9,
        }
    }
}

/// A scored layout: its metrics plus the scalar cost to minimize.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacementScore {
    pub metrics: PlacementMetrics,
    pub cost: f64,
}

impl RewardWeights {
    /// Scalar cost (lower is better). Unroutable layouts pay the flat penalty.
    pub fn cost(&self, m: &PlacementMetrics) -> f64 {
        if !m.routable {
            return self.unroutable_penalty;
        }
        self.wire * m.wire_tiles as f64
            + self.via * m.vias as f64
            + self.crossing * m.crossings as f64
            + self.area * m.canvas_area as f64
            + self.timing * m.timing_cost
    }

    /// Bundle metrics with their cost.
    pub fn score(&self, m: PlacementMetrics) -> PlacementScore {
        let cost = self.cost(&m);
        PlacementScore { metrics: m, cost }
    }
}
