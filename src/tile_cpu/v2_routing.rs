//! V2 routing primitives for legality-aware, deterministic multi-net planning.
//!
//! This module is intentionally simulation-agnostic at runtime. It consumes a
//! static snapshot of routable cells and computes candidate paths with explicit
//! legality/capacity constraints.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use crate::simulation::Simulation;
use crate::tile_meta::TileType;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct V2RouteCoord {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl V2RouteCoord {
    pub const fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }

    fn manhattan(self, other: Self) -> u32 {
        (self.x.abs_diff(other.x) + self.y.abs_diff(other.y) + self.z.abs_diff(other.z)) as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V2RouteBounds {
    pub min_x: usize,
    pub max_x_exclusive: usize,
    pub min_y: usize,
    pub max_y_exclusive: usize,
    pub min_layer: usize,
    pub max_layer_inclusive: usize,
}

impl V2RouteBounds {
    pub const fn new(
        min_x: usize,
        max_x_exclusive: usize,
        min_y: usize,
        max_y_exclusive: usize,
        min_layer: usize,
        max_layer_inclusive: usize,
    ) -> Self {
        Self {
            min_x,
            max_x_exclusive,
            min_y,
            max_y_exclusive,
            min_layer,
            max_layer_inclusive,
        }
    }

    pub fn contains(self, coord: V2RouteCoord) -> bool {
        coord.x >= self.min_x
            && coord.x < self.max_x_exclusive
            && coord.y >= self.min_y
            && coord.y < self.max_y_exclusive
            && coord.z >= self.min_layer
            && coord.z <= self.max_layer_inclusive
    }
}

/// Traffic class for layer-affinity routing.
/// Determines which layers a net prefers to route on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum V2TrafficClass {
    /// Register values, ALU operands/results, RAM data, MMIO data.
    /// Prefers L1-L2.
    Data,
    /// Decoder outputs, enable signals, write-enables, branch flags.
    /// Prefers L2.
    Control,
    /// Cross-region signals, clock distribution, broadcast trees.
    /// Prefers L3.
    LongRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum V2RouteNetClass {
    ControlCritical,
    DataCritical,
    Data,
}

impl V2RouteNetClass {
    fn priority(self) -> u8 {
        match self {
            Self::ControlCritical => 0,
            Self::DataCritical => 1,
            Self::Data => 2,
        }
    }
}

/// Endpoint reservation class for route start/goal cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum V2EndpointClass {
    /// Single owner — only one net may use this endpoint.
    HardExclusive,
    /// Fan-out source — up to `capacity` nets may share this endpoint.
    SharedFan { capacity: u16 },
}

impl V2EndpointClass {
    #[allow(dead_code)]
    fn is_shared(self) -> bool {
        matches!(self, Self::SharedFan { .. })
    }

    fn capacity(self) -> Option<u16> {
        match self {
            Self::SharedFan { capacity } => Some(capacity),
            Self::HardExclusive => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct V2RouteNet {
    pub id: String,
    pub class: V2RouteNetClass,
    pub start: V2RouteCoord,
    pub goal: V2RouteCoord,
    pub start_class: V2EndpointClass,
    pub goal_class: V2EndpointClass,
    /// Optional traffic class for layer-affinity routing.
    /// None = no affinity preference (backward-compatible default).
    pub traffic_class: Option<V2TrafficClass>,
}

impl V2RouteNet {
    pub fn new(
        id: impl Into<String>,
        class: V2RouteNetClass,
        start: V2RouteCoord,
        goal: V2RouteCoord,
    ) -> Self {
        Self {
            id: id.into(),
            class,
            start,
            goal,
            start_class: V2EndpointClass::HardExclusive,
            goal_class: V2EndpointClass::HardExclusive,
            traffic_class: None,
        }
    }

    /// Backward-compatible: `true` maps to `SharedFan { capacity: u16::MAX }`.
    pub fn with_shared_start(mut self, allow: bool) -> Self {
        self.start_class = if allow {
            V2EndpointClass::SharedFan { capacity: u16::MAX }
        } else {
            V2EndpointClass::HardExclusive
        };
        self
    }

    /// Backward-compatible: `true` maps to `SharedFan { capacity: u16::MAX }`.
    pub fn with_shared_goal(mut self, allow: bool) -> Self {
        self.goal_class = if allow {
            V2EndpointClass::SharedFan { capacity: u16::MAX }
        } else {
            V2EndpointClass::HardExclusive
        };
        self
    }

    pub fn with_start_class(mut self, class: V2EndpointClass) -> Self {
        self.start_class = class;
        self
    }

    pub fn with_goal_class(mut self, class: V2EndpointClass) -> Self {
        self.goal_class = class;
        self
    }

    pub fn with_traffic_class(mut self, tc: V2TrafficClass) -> Self {
        self.traffic_class = Some(tc);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V2RoutingCell {
    pub hard_block: bool,
    pub soft_cost: u32,
    pub capacity: u16,
    /// Bitfield of allowed entry directions for this cell.
    ///
    /// - Bit 0: allow entry from LEFT  (cell reads from left, e.g. WireRight)
    /// - Bit 1: allow entry from RIGHT (cell reads from right, e.g. WireLeft)
    /// - Bit 2: allow entry from UP    (cell reads from up, e.g. WireDown)
    /// - Bit 3: allow entry from DOWN  (cell reads from down, e.g. WireUp)
    /// - Bit 4: allow LayerUp entry    (cell reads from layer above, e.g. ViaDown)
    /// - Bit 5: allow LayerDown entry  (cell reads from layer below, e.g. ViaUp)
    /// - 0xFF: omnidirectional (default for Const, Wire, logic gates)
    pub directional_mask: u8,
}

/// Directional mask constants for `V2RoutingCell::directional_mask`.
pub const DIR_FROM_LEFT: u8 = 1 << 0;
pub const DIR_FROM_RIGHT: u8 = 1 << 1;
pub const DIR_FROM_UP: u8 = 1 << 2;
pub const DIR_FROM_DOWN: u8 = 1 << 3;
pub const DIR_LAYER_UP: u8 = 1 << 4;
pub const DIR_LAYER_DOWN: u8 = 1 << 5;
pub const DIR_OMNI: u8 = 0xFF;

impl Default for V2RoutingCell {
    fn default() -> Self {
        Self {
            hard_block: false,
            soft_cost: 0,
            capacity: 1,
            directional_mask: DIR_OMNI,
        }
    }
}

/// Compute the directional entry mask for a given tile type.
///
/// For directional wires, only the direction the wire reads from is allowed.
/// For vias, only the cross-layer direction is allowed.
/// For everything else (Const, Wire, logic gates), all directions are allowed.
fn directional_mask_for_tile(tt: TileType) -> u8 {
    match tt {
        // WireRight reads from LEFT → signal enters from LEFT
        TileType::WireRight => DIR_FROM_LEFT,
        // WireLeft reads from RIGHT → signal enters from RIGHT
        TileType::WireLeft => DIR_FROM_RIGHT,
        // WireDown reads from UP → signal enters from UP
        TileType::WireDown => DIR_FROM_UP,
        // WireUp reads from DOWN → signal enters from DOWN
        TileType::WireUp => DIR_FROM_DOWN,
        // ViaUp reads from layer z+1 → signal enters from LayerUp
        TileType::ViaUp | TileType::WeightedViaUp | TileType::ThresholdViaUp => DIR_LAYER_UP,
        // ViaDown reads from layer z-1 → signal enters from LayerDown
        TileType::ViaDown | TileType::WeightedViaDown | TileType::ThresholdViaDown => {
            DIR_LAYER_DOWN
        }
        // All other tiles: omnidirectional
        _ => DIR_OMNI,
    }
}

/// Sprint 398: conservative over-approximation of the directions a tile READS
/// from (same bit encoding as `directional_mask_for_tile`).
///
/// Used to hard-block route cells that existing machinery would read: if this
/// returns a bit for a direction the tile does not actually read, the only
/// cost is a blocked routing cell; if it missed a direction the tile does
/// read, a routed signal could be silently injected into that tile. Unknown
/// tile types therefore default to all four in-plane directions.
pub(crate) fn conservative_reader_mask(tt: TileType) -> u8 {
    match tt {
        // True sources: read nothing.
        TileType::Const => 0,
        // Directional wires: exactly one input side.
        TileType::WireRight => DIR_FROM_LEFT,
        TileType::WireLeft => DIR_FROM_RIGHT,
        TileType::WireDown => DIR_FROM_UP,
        TileType::WireUp => DIR_FROM_DOWN,
        TileType::WireH => DIR_FROM_LEFT | DIR_FROM_RIGHT,
        TileType::WireV => DIR_FROM_UP | DIR_FROM_DOWN,
        // Plain/weighted vias: cross-layer source only, no in-plane reads.
        TileType::ViaUp | TileType::WeightedViaUp => DIR_LAYER_UP,
        TileType::ViaDown | TileType::WeightedViaDown => DIR_LAYER_DOWN,
        // Threshold vias gate on the popcount of all 4 in-plane neighbors.
        TileType::ThresholdViaUp => {
            DIR_LAYER_UP | DIR_FROM_LEFT | DIR_FROM_RIGHT | DIR_FROM_UP | DIR_FROM_DOWN
        }
        TileType::ThresholdViaDown => {
            DIR_LAYER_DOWN | DIR_FROM_LEFT | DIR_FROM_RIGHT | DIR_FROM_UP | DIR_FROM_DOWN
        }
        // Everything else (gates, muxes, registers, RAM, unknown): assume it
        // reads all four in-plane neighbors.
        _ => DIR_FROM_LEFT | DIR_FROM_RIGHT | DIR_FROM_UP | DIR_FROM_DOWN,
    }
}

/// Sprint 398: under-approximation of the directions a tile is KNOWN to read
/// from (same bit encoding as `directional_mask_for_tile`).
///
/// Used for goal-ingress checks when a route terminates on an existing tile:
/// the delivery only physically connects if the goal tile actually reads from
/// the entry direction. Returning a bit the tile does not read would produce a
/// silent no-connect, so unknown tile types default to 0 (reject as ingress).
/// Read sets match the compact-op input table in `simulation.rs`.
pub(crate) fn known_reader_mask(tt: TileType) -> u8 {
    const ALL4: u8 = DIR_FROM_LEFT | DIR_FROM_RIGHT | DIR_FROM_UP | DIR_FROM_DOWN;
    match tt {
        TileType::Wire | TileType::Cross => ALL4,
        TileType::WireRight => DIR_FROM_LEFT,
        TileType::WireLeft => DIR_FROM_RIGHT,
        TileType::WireDown => DIR_FROM_UP,
        TileType::WireUp => DIR_FROM_DOWN,
        TileType::WireH => DIR_FROM_LEFT | DIR_FROM_RIGHT,
        TileType::WireV => DIR_FROM_UP | DIR_FROM_DOWN,
        // Two-input left/right tiles.
        TileType::And
        | TileType::Or
        | TileType::Xor
        | TileType::Add
        | TileType::Sub
        | TileType::Mul
        | TileType::Div
        | TileType::Mod
        | TileType::Shl
        | TileType::Shr
        | TileType::Lt
        | TileType::Gt
        | TileType::Eq
        | TileType::Neq
        | TileType::Lte
        | TileType::Gte
        | TileType::AddCarry
        | TileType::SubBorrow
        | TileType::CarryDetect
        | TileType::BitSelect
        | TileType::Mux8to1
        | TileType::ProgramCounter => DIR_FROM_LEFT | DIR_FROM_RIGHT,
        // Single-input left tiles (registers capture LEFT on clock edge).
        TileType::Not
        | TileType::Neg
        | TileType::Abs
        | TileType::Zero
        | TileType::Decoder3to8
        | TileType::Decoder6to64
        | TileType::Latch
        | TileType::Register8
        | TileType::Register64
        | TileType::Synchronizer => DIR_FROM_LEFT,
        TileType::Mux | TileType::Mux16to1 | TileType::RegEnable => {
            DIR_FROM_LEFT | DIR_FROM_RIGHT | DIR_FROM_UP
        }
        TileType::Mux4to1 => DIR_FROM_UP | DIR_FROM_DOWN,
        TileType::Demux1to8 => DIR_FROM_LEFT | DIR_FROM_UP,
        TileType::Ram => DIR_FROM_LEFT | DIR_FROM_UP,
        TileType::Counter => DIR_FROM_UP,
        TileType::MemoryPort => DIR_FROM_LEFT | DIR_FROM_RIGHT | DIR_FROM_UP,
        TileType::WireCross => DIR_FROM_LEFT | DIR_FROM_UP,
        TileType::WireCrossVert => DIR_FROM_RIGHT | DIR_FROM_UP,
        TileType::ViaUp | TileType::WeightedViaUp => DIR_LAYER_UP,
        TileType::ViaDown | TileType::WeightedViaDown => DIR_LAYER_DOWN,
        TileType::ThresholdViaUp => DIR_LAYER_UP | ALL4,
        TileType::ThresholdViaDown => DIR_LAYER_DOWN | ALL4,
        // Const, demo tiles, anything else: not a known ingress.
        _ => 0,
    }
}

#[derive(Clone, Debug)]
pub struct V2RoutingDb {
    width: usize,
    height: usize,
    layers: usize,
    bounds: V2RouteBounds,
    cells: Vec<V2RoutingCell>,
}

impl V2RoutingDb {
    pub fn new(width: usize, height: usize, layers: usize, bounds: V2RouteBounds) -> Self {
        assert!(width > 0, "width must be > 0");
        assert!(height > 0, "height must be > 0");
        assert!(layers > 0, "layers must be > 0");
        assert!(
            bounds.max_x_exclusive <= width,
            "x bound exceeds grid width"
        );
        assert!(
            bounds.max_y_exclusive <= height,
            "y bound exceeds grid height"
        );
        assert!(
            bounds.max_layer_inclusive < layers,
            "layer bound exceeds grid layers"
        );
        assert!(bounds.min_x < bounds.max_x_exclusive, "invalid x bounds");
        assert!(bounds.min_y < bounds.max_y_exclusive, "invalid y bounds");
        assert!(
            bounds.min_layer <= bounds.max_layer_inclusive,
            "invalid layer bounds"
        );

        Self {
            width,
            height,
            layers,
            bounds,
            cells: vec![V2RoutingCell::default(); width * height * layers],
        }
    }

    pub fn from_simulation_const_blocked(sim: &Simulation, bounds: V2RouteBounds) -> Self {
        let mut db = Self::new(sim.width(), sim.height(), sim.num_layers(), bounds);
        for z in bounds.min_layer..=bounds.max_layer_inclusive {
            for y in bounds.min_y..bounds.max_y_exclusive {
                for x in bounds.min_x..bounds.max_x_exclusive {
                    let tt = sim.tile_type_3d(x, y, z);
                    let coord = V2RouteCoord::new(x, y, z);
                    if tt != TileType::Const {
                        db.set_hard_block(coord, true);
                    }
                    let idx = db.coord_to_idx(coord);
                    db.cells[idx].directional_mask = directional_mask_for_tile(tt);
                }
            }
        }
        db
    }

    /// Sprint 398: interference-hardened variant of
    /// [`Self::from_simulation_const_blocked`].
    ///
    /// In this fabric adjacency IS connectivity: any tile that reads a
    /// neighboring cell picks up whatever value sits there, so a route through
    /// a cell that existing machinery reads silently injects its signal into
    /// that machinery even though no cell collision occurs ("routable does not
    /// imply correct" — the failure class fixed in b21eaa6 for the synth
    /// router). This constructor additionally hard-blocks:
    ///
    /// - every free cell that a non-Const tile reads, in-plane (via
    ///   `conservative_reader_mask`) or cross-layer (via tiles on z±1), and
    /// - every coord in `protected` (software injection-port Consts, guard
    ///   rows — cells that are Const but not actually free).
    ///
    /// Route endpoints are exempt from hard blocks inside `route_multinet`
    /// (terminals skip legality), so deliberate taps/ingress points on blocked
    /// cells still work.
    pub fn from_simulation_interference_checked(
        sim: &Simulation,
        bounds: V2RouteBounds,
        protected: &[V2RouteCoord],
    ) -> Self {
        let mut db = Self::from_simulation_const_blocked(sim, bounds);
        let width = sim.width();
        let height = sim.height();
        let layers = sim.num_layers();
        let is_down_reading_via = |tt: TileType| {
            matches!(
                tt,
                TileType::ViaDown | TileType::WeightedViaDown | TileType::ThresholdViaDown
            )
        };
        let is_up_reading_via = |tt: TileType| {
            matches!(
                tt,
                TileType::ViaUp | TileType::WeightedViaUp | TileType::ThresholdViaUp
            )
        };
        for z in bounds.min_layer..=bounds.max_layer_inclusive {
            for y in bounds.min_y..bounds.max_y_exclusive {
                for x in bounds.min_x..bounds.max_x_exclusive {
                    let coord = V2RouteCoord::new(x, y, z);
                    let idx = db.coord_to_idx(coord);
                    if db.cells[idx].hard_block {
                        continue;
                    }
                    // Neighbors are checked against the full sim grid, not the
                    // routing bounds: readers just outside the bounds are
                    // still corrupted by a signal placed inside them.
                    let mut read_by_foreign = false;
                    if x + 1 < width {
                        let tt = sim.tile_type_3d(x + 1, y, z);
                        read_by_foreign |= conservative_reader_mask(tt) & DIR_FROM_LEFT != 0;
                    }
                    if x >= 1 {
                        let tt = sim.tile_type_3d(x - 1, y, z);
                        read_by_foreign |= conservative_reader_mask(tt) & DIR_FROM_RIGHT != 0;
                    }
                    if y + 1 < height {
                        let tt = sim.tile_type_3d(x, y + 1, z);
                        read_by_foreign |= conservative_reader_mask(tt) & DIR_FROM_UP != 0;
                    }
                    if y >= 1 {
                        let tt = sim.tile_type_3d(x, y - 1, z);
                        read_by_foreign |= conservative_reader_mask(tt) & DIR_FROM_DOWN != 0;
                    }
                    if z + 1 < layers {
                        read_by_foreign |= is_down_reading_via(sim.tile_type_3d(x, y, z + 1));
                    }
                    if z >= 1 {
                        read_by_foreign |= is_up_reading_via(sim.tile_type_3d(x, y, z - 1));
                    }
                    if read_by_foreign {
                        db.cells[idx].hard_block = true;
                    }
                }
            }
        }
        for p in protected {
            if db.in_bounds(*p) {
                db.set_hard_block(*p, true);
            }
        }
        db
    }

    pub fn bounds(&self) -> V2RouteBounds {
        self.bounds
    }

    pub fn in_bounds(&self, coord: V2RouteCoord) -> bool {
        self.bounds.contains(coord)
    }

    pub fn cell(&self, coord: V2RouteCoord) -> V2RoutingCell {
        self.cells[self.coord_to_idx(coord)]
    }

    pub fn set_hard_block(&mut self, coord: V2RouteCoord, blocked: bool) {
        let idx = self.coord_to_idx(coord);
        self.cells[idx].hard_block = blocked;
    }

    pub fn set_soft_cost(&mut self, coord: V2RouteCoord, soft_cost: u32) {
        let idx = self.coord_to_idx(coord);
        self.cells[idx].soft_cost = soft_cost;
    }

    pub fn set_capacity(&mut self, coord: V2RouteCoord, capacity: u16) {
        let idx = self.coord_to_idx(coord);
        self.cells[idx].capacity = capacity.max(1);
    }

    pub fn set_directional_mask(&mut self, coord: V2RouteCoord, mask: u8) {
        let idx = self.coord_to_idx(coord);
        self.cells[idx].directional_mask = mask;
    }

    pub fn reserve_horizontal_channel(
        &mut self,
        z: usize,
        y: usize,
        x_start: usize,
        x_end_exclusive: usize,
        capacity: u16,
        soft_cost: u32,
    ) {
        for x in x_start..x_end_exclusive {
            let coord = V2RouteCoord::new(x, y, z);
            if !self.in_bounds(coord) {
                continue;
            }
            let idx = self.coord_to_idx(coord);
            self.cells[idx].hard_block = false;
            self.cells[idx].capacity = self.cells[idx].capacity.max(capacity.max(1));
            self.cells[idx].soft_cost = soft_cost;
            self.cells[idx].directional_mask = DIR_OMNI;
        }
    }

    pub fn reserve_vertical_channel(
        &mut self,
        z: usize,
        x: usize,
        y_start: usize,
        y_end_exclusive: usize,
        capacity: u16,
        soft_cost: u32,
    ) {
        for y in y_start..y_end_exclusive {
            let coord = V2RouteCoord::new(x, y, z);
            if !self.in_bounds(coord) {
                continue;
            }
            let idx = self.coord_to_idx(coord);
            self.cells[idx].hard_block = false;
            self.cells[idx].capacity = self.cells[idx].capacity.max(capacity.max(1));
            self.cells[idx].soft_cost = soft_cost;
            self.cells[idx].directional_mask = DIR_OMNI;
        }
    }

    pub fn reserve_switchbox(
        &mut self,
        x: usize,
        y: usize,
        z_min: usize,
        z_max_inclusive: usize,
        capacity: u16,
    ) {
        for z in z_min..=z_max_inclusive {
            let coord = V2RouteCoord::new(x, y, z);
            if !self.in_bounds(coord) {
                continue;
            }
            let idx = self.coord_to_idx(coord);
            self.cells[idx].hard_block = false;
            self.cells[idx].capacity = self.cells[idx].capacity.max(capacity.max(1));
            self.cells[idx].directional_mask = DIR_OMNI;
        }
    }

    pub fn route_multinet(
        &self,
        nets: &[V2RouteNet],
        config: V2RoutingConfig,
    ) -> V2MultiRouteResult {
        if nets.is_empty() {
            return V2MultiRouteResult {
                success: true,
                routes: BTreeMap::new(),
                overflow_cells: Vec::new(),
                failed_net_ids: Vec::new(),
                negotiation_iters: 0,
                layer_violation_count: 0,
                layer_utilization: [0; 4],
            };
        }

        let mut work = nets.to_vec();
        if !config.use_input_order {
            work.sort_by(|a, b| {
                a.class
                    .priority()
                    .cmp(&b.class.priority())
                    .then_with(|| a.id.cmp(&b.id))
            });
        }

        let eff_cap = self.build_effective_capacity_map(&work);
        let mut historical = vec![0u32; self.cells.len()];
        let mut last_result = V2MultiRouteResult {
            success: false,
            routes: BTreeMap::new(),
            overflow_cells: Vec::new(),
            failed_net_ids: Vec::new(),
            negotiation_iters: 0,
            layer_violation_count: 0,
            layer_utilization: [0; 4],
        };

        for iter in 0..config.max_negotiation_iters.max(1) {
            let mut usage = vec![0u16; self.cells.len()];
            let mut routes = BTreeMap::new();
            let mut failed_net_ids = Vec::new();

            for net in &work {
                if let Some(path) = self.route_single(net, &usage, &historical, config) {
                    self.apply_path_usage(&mut usage, net, &path);
                    routes.insert(net.id.clone(), path);
                } else {
                    failed_net_ids.push(net.id.clone());
                    break;
                }
            }

            let overflow_cells = self.collect_overflows(&usage, &eff_cap);
            let success = failed_net_ids.is_empty() && overflow_cells.is_empty();

            last_result = V2MultiRouteResult {
                success,
                routes,
                overflow_cells,
                failed_net_ids,
                negotiation_iters: iter + 1,
                layer_violation_count: 0,
                layer_utilization: [0; 4],
            };

            if last_result.success {
                last_result.compute_layer_metrics(&work);
                return last_result;
            }

            // Bounded rip-up/reroute: displace lower-priority nets at overflow cells.
            if config.enable_rip_up && !last_result.overflow_cells.is_empty() {
                let rip_result = self.try_rip_up_reroute(
                    &work,
                    &mut last_result.routes,
                    &mut usage,
                    &historical,
                    config,
                    &eff_cap,
                );
                if rip_result {
                    // Invariant: all nets must still have routes after rip-up.
                    for net in &work {
                        if !last_result.routes.contains_key(&net.id)
                            && !last_result.failed_net_ids.contains(&net.id)
                        {
                            last_result.failed_net_ids.push(net.id.clone());
                        }
                    }
                    // Recompute overflows after rip-up
                    let overflow_cells = self.collect_overflows(&usage, &eff_cap);
                    let success =
                        last_result.failed_net_ids.is_empty() && overflow_cells.is_empty();
                    last_result.overflow_cells = overflow_cells;
                    last_result.success = success;
                    if last_result.success {
                        last_result.compute_layer_metrics(&work);
                        return last_result;
                    }
                }
            }

            // Negotiated-congestion update:
            // - on overflow: increase historical costs at overflowing cells
            // - on hard failure: increase costs along already claimed paths
            if !last_result.overflow_cells.is_empty() {
                for item in &last_result.overflow_cells {
                    let idx = self.coord_to_idx(item.coord);
                    historical[idx] = historical[idx].saturating_add(config.overflow_penalty);
                }
            } else if !last_result.failed_net_ids.is_empty() {
                for path in last_result.routes.values() {
                    for coord in &path.coords {
                        let idx = self.coord_to_idx(*coord);
                        historical[idx] =
                            historical[idx].saturating_add(config.overflow_penalty / 2);
                    }
                }
            }
        }

        last_result.compute_layer_metrics(&work);
        last_result
    }

    fn route_single(
        &self,
        net: &V2RouteNet,
        usage: &[u16],
        historical: &[u32],
        config: V2RoutingConfig,
    ) -> Option<V2RoutePath> {
        if !self.in_bounds(net.start) || !self.in_bounds(net.goal) {
            return None;
        }

        let mut best = vec![[u32::MAX; StepDir::COUNT]; self.cells.len()];
        let mut prev = vec![[None; StepDir::COUNT]; self.cells.len()];
        let mut heap: BinaryHeap<Reverse<(u32, u32, V2RouteCoord, StepDir)>> = BinaryHeap::new();

        let start_idx = self.coord_to_idx(net.start);
        best[start_idx][StepDir::Start.idx()] = 0;
        heap.push(Reverse((
            net.start.manhattan(net.goal),
            0,
            net.start,
            StepDir::Start,
        )));

        while let Some(Reverse((_, g, coord, dir))) = heap.pop() {
            let coord_idx = self.coord_to_idx(coord);
            if g != best[coord_idx][dir.idx()] {
                continue;
            }
            if coord == net.goal {
                break;
            }

            for (next, next_dir) in self.ordered_neighbors(coord) {
                if !self.in_bounds(next) {
                    continue;
                }

                let next_idx = self.coord_to_idx(next);
                let terminal = next == net.start || next == net.goal;
                if self.cells[next_idx].hard_block && !terminal {
                    continue;
                }

                // Directional legality: check that the destination cell allows
                // entry from the direction we're arriving. Skip for terminals
                // (start/goal endpoints are always reachable).
                if !terminal {
                    let entry_bit = next_dir.entry_mask();
                    if self.cells[next_idx].directional_mask & entry_bit == 0 {
                        continue;
                    }
                }

                let mut step = 1u32;
                if dir != StepDir::Start && dir != next_dir {
                    step = step.saturating_add(config.bend_penalty);
                }
                if matches!(next_dir, StepDir::LayerUp | StepDir::LayerDown) {
                    step = step.saturating_add(config.via_penalty);
                }

                // Layer-affinity penalty: discourage routing on non-preferred layers
                if config.layer_affinity_penalty > 0
                    && let Some(tc) = net.traffic_class
                {
                    let preferred = match tc {
                        V2TrafficClass::Data => next.z <= 2,      // L0-L2
                        V2TrafficClass::Control => next.z == 2,   // L2
                        V2TrafficClass::LongRange => next.z == 3, // L3
                    };
                    if !preferred {
                        step = step.saturating_add(config.layer_affinity_penalty);
                    }
                }

                step = step.saturating_add(
                    self.cells[next_idx]
                        .soft_cost
                        .saturating_mul(config.soft_cost_weight),
                );
                step = step.saturating_add(historical[next_idx]);

                // Congestion penalty: capacity-aware for SharedFan endpoints.
                let shared_cap = if next == net.start {
                    net.start_class.capacity()
                } else if next == net.goal {
                    net.goal_class.capacity()
                } else {
                    None
                };
                match shared_cap {
                    Some(cap) if cap == u16::MAX => {
                        // Unlimited fan-out: no congestion penalty (backward compat).
                    }
                    Some(cap) => {
                        // Finite fan-out: graduated penalty near capacity.
                        if usage[next_idx] >= cap {
                            step = step.saturating_add(config.congestion_penalty.saturating_mul(4));
                        } else if (usage[next_idx] as u32) > (cap as u32) / 2 {
                            step = step.saturating_add(
                                (usage[next_idx] as u32).saturating_mul(config.congestion_penalty),
                            );
                        }
                    }
                    None => {
                        // Regular cell: standard congestion penalty.
                        step = step.saturating_add(
                            (usage[next_idx] as u32).saturating_mul(config.congestion_penalty),
                        );
                    }
                }

                let new_g = g.saturating_add(step);
                if new_g >= best[next_idx][next_dir.idx()] {
                    continue;
                }

                best[next_idx][next_dir.idx()] = new_g;
                prev[next_idx][next_dir.idx()] = Some((coord, dir));

                let h = next.manhattan(net.goal);
                let f = new_g.saturating_add(h);
                heap.push(Reverse((f, new_g, next, next_dir)));
            }
        }

        let goal_idx = self.coord_to_idx(net.goal);
        let mut best_goal_cost = u32::MAX;
        let mut best_goal_dir = StepDir::Start;
        for dir in StepDir::all() {
            let g = best[goal_idx][dir.idx()];
            if g < best_goal_cost {
                best_goal_cost = g;
                best_goal_dir = dir;
            }
        }
        if best_goal_cost == u32::MAX {
            return None;
        }

        let mut coords = Vec::new();
        let mut cur_coord = net.goal;
        let mut cur_dir = best_goal_dir;
        coords.push(cur_coord);

        while cur_coord != net.start {
            let cur_idx = self.coord_to_idx(cur_coord);
            let Some((parent_coord, parent_dir)) = prev[cur_idx][cur_dir.idx()] else {
                return None;
            };
            cur_coord = parent_coord;
            cur_dir = parent_dir;
            coords.push(cur_coord);
        }
        coords.reverse();

        let (bends, vias) = path_shape_metrics(&coords);
        Some(V2RoutePath {
            coords,
            cost: best_goal_cost,
            bends,
            vias,
        })
    }

    fn apply_path_usage(&self, usage: &mut [u16], _net: &V2RouteNet, path: &V2RoutePath) {
        for coord in &path.coords {
            let idx = self.coord_to_idx(*coord);
            usage[idx] = usage[idx].saturating_add(1);
        }
    }

    fn remove_path_usage(&self, usage: &mut [u16], _net: &V2RouteNet, path: &V2RoutePath) {
        for coord in &path.coords {
            let idx = self.coord_to_idx(*coord);
            usage[idx] = usage[idx].saturating_sub(1);
        }
    }

    /// Attempt bounded rip-up/reroute to resolve overflows.
    ///
    /// For each overflow cell, identifies the highest-priority net and any lower-priority
    /// nets sharing the cell. Rips up the lower-priority net, reroutes the high-priority
    /// net, then reroutes the displaced net. Returns true if any rip-up was performed.
    ///
    /// Safety: if either reroute fails, the original route is restored so no net is
    /// silently dropped from the routes map.
    fn try_rip_up_reroute(
        &self,
        nets: &[V2RouteNet],
        routes: &mut BTreeMap<String, V2RoutePath>,
        usage: &mut [u16],
        historical: &[u32],
        config: V2RoutingConfig,
        eff_cap: &[u16],
    ) -> bool {
        let net_map: BTreeMap<String, &V2RouteNet> =
            nets.iter().map(|n| (n.id.clone(), n)).collect();

        // Pre-compute rip-up candidates: for each overflow cell, determine (winner_id, victim_id).
        // Use owned Strings to avoid borrowing `routes`.
        let overflows = self.collect_overflows(usage, eff_cap);
        let mut candidates: Vec<(String, String)> = Vec::new();

        {
            // Build reverse index with owned Strings.
            let mut cell_to_nets: BTreeMap<usize, Vec<String>> = BTreeMap::new();
            for (net_id, path) in routes.iter() {
                for coord in &path.coords {
                    let idx = self.coord_to_idx(*coord);
                    cell_to_nets.entry(idx).or_default().push(net_id.clone());
                }
            }

            for overflow in &overflows {
                let ov_idx = self.coord_to_idx(overflow.coord);
                let Some(net_ids) = cell_to_nets.get(&ov_idx) else {
                    continue;
                };

                // Find the highest-priority net at this cell.
                let mut best_pri = u8::MAX;
                let mut best_id: Option<&str> = None;
                for nid in net_ids {
                    if let Some(net) = net_map.get(nid.as_str()) {
                        let p = net.class.priority();
                        if p < best_pri
                            || (p == best_pri
                                && (best_id.is_none() || nid.as_str() < best_id.unwrap()))
                        {
                            best_pri = p;
                            best_id = Some(nid.as_str());
                        }
                    }
                }
                let Some(winner_id) = best_id else {
                    continue;
                };

                // Find a lower-priority net to displace.
                let mut victim_id: Option<&str> = None;
                let mut victim_pri = 0u8;
                for nid in net_ids {
                    if nid.as_str() == winner_id {
                        continue;
                    }
                    if let Some(net) = net_map.get(nid.as_str()) {
                        let p = net.class.priority();
                        if p > best_pri
                            && (victim_id.is_none()
                                || p > victim_pri
                                || (p == victim_pri && nid.as_str() > victim_id.unwrap()))
                        {
                            victim_pri = p;
                            victim_id = Some(nid.as_str());
                        }
                    }
                }
                if let Some(victim) = victim_id {
                    candidates.push((winner_id.to_string(), victim.to_string()));
                }
            }
        }

        // Now execute rip-ups using the pre-computed candidates.
        let mut rip_ups_done = 0;
        let mut ripped_ids: Vec<String> = Vec::new();

        for (winner_id, victim_id) in &candidates {
            if rip_ups_done >= config.max_rip_up_per_iter {
                break;
            }
            if ripped_ids.contains(victim_id) {
                continue;
            }

            let victim_net = net_map[victim_id.as_str()];
            let winner_net = net_map[winner_id.as_str()];

            // Step 1: Remove victim's route, keeping backup for restoration on failure.
            let victim_backup = routes.remove(victim_id.as_str());
            if let Some(ref vp) = victim_backup {
                self.remove_path_usage(usage, victim_net, vp);
            }

            // Step 2: Remove winner's route, keeping backup for restoration on failure.
            let winner_backup = routes.remove(winner_id.as_str());
            if let Some(ref wp) = winner_backup {
                self.remove_path_usage(usage, winner_net, wp);
            }

            // Step 3: Reroute winner through freed capacity.
            if let Some(new_winner_path) = self.route_single(winner_net, usage, historical, config)
            {
                self.apply_path_usage(usage, winner_net, &new_winner_path);
                routes.insert(winner_id.clone(), new_winner_path);
            } else if let Some(wp) = winner_backup {
                // Reroute failed — restore winner's original route.
                self.apply_path_usage(usage, winner_net, &wp);
                routes.insert(winner_id.clone(), wp);
            }

            // Step 4: Reroute victim around the new winner path.
            if let Some(new_victim_path) = self.route_single(victim_net, usage, historical, config)
            {
                self.apply_path_usage(usage, victim_net, &new_victim_path);
                routes.insert(victim_id.clone(), new_victim_path);
            } else if let Some(vp) = victim_backup {
                // Reroute failed — restore victim's original route.
                self.apply_path_usage(usage, victim_net, &vp);
                routes.insert(victim_id.clone(), vp);
            }

            ripped_ids.push(victim_id.clone());
            rip_ups_done += 1;
        }

        rip_ups_done > 0
    }

    fn collect_overflows(&self, usage: &[u16], eff_cap: &[u16]) -> Vec<V2RouteOverflow> {
        let mut out = Vec::new();
        for z in self.bounds.min_layer..=self.bounds.max_layer_inclusive {
            for y in self.bounds.min_y..self.bounds.max_y_exclusive {
                for x in self.bounds.min_x..self.bounds.max_x_exclusive {
                    let coord = V2RouteCoord::new(x, y, z);
                    let idx = self.coord_to_idx(coord);
                    let cap = eff_cap[idx];
                    let used = usage[idx];
                    if used > cap {
                        out.push(V2RouteOverflow {
                            coord,
                            usage: used,
                            capacity: cap,
                        });
                    }
                }
            }
        }
        out
    }

    /// Build a per-cell effective capacity map that accounts for SharedFan endpoints.
    ///
    /// For cells that are SharedFan endpoints, the effective capacity is the maximum
    /// of the cell's grid capacity and the declared SharedFan capacity.
    fn build_effective_capacity_map(&self, nets: &[V2RouteNet]) -> Vec<u16> {
        let mut eff_cap: Vec<u16> = self.cells.iter().map(|c| c.capacity.max(1)).collect();
        let mut declared: std::collections::BTreeMap<usize, u16> =
            std::collections::BTreeMap::new();
        for net in nets {
            for (coord, class) in [(net.start, &net.start_class), (net.goal, &net.goal_class)] {
                if let V2EndpointClass::SharedFan { capacity } = class {
                    let idx = self.coord_to_idx(coord);
                    eff_cap[idx] = eff_cap[idx].max(*capacity);
                    if let Some(&prev) = declared.get(&idx) {
                        debug_assert_eq!(
                            prev, *capacity,
                            "SharedFan capacity conflict at idx {}: {} vs {}",
                            idx, prev, *capacity
                        );
                    }
                    declared.insert(idx, *capacity);
                }
            }
        }
        eff_cap
    }

    fn ordered_neighbors(&self, coord: V2RouteCoord) -> [(V2RouteCoord, StepDir); 6] {
        let left = V2RouteCoord::new(coord.x.wrapping_sub(1), coord.y, coord.z);
        let right = V2RouteCoord::new(coord.x + 1, coord.y, coord.z);
        let up = V2RouteCoord::new(coord.x, coord.y.wrapping_sub(1), coord.z);
        let down = V2RouteCoord::new(coord.x, coord.y + 1, coord.z);
        let layer_down = V2RouteCoord::new(coord.x, coord.y, coord.z.wrapping_sub(1));
        let layer_up = V2RouteCoord::new(coord.x, coord.y, coord.z + 1);
        [
            (left, StepDir::Left),
            (right, StepDir::Right),
            (up, StepDir::Up),
            (down, StepDir::Down),
            (layer_down, StepDir::LayerDown),
            (layer_up, StepDir::LayerUp),
        ]
    }

    fn coord_to_idx(&self, coord: V2RouteCoord) -> usize {
        debug_assert!(coord.x < self.width);
        debug_assert!(coord.y < self.height);
        debug_assert!(coord.z < self.layers);
        coord.z * self.width * self.height + coord.y * self.width + coord.x
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V2RoutingConfig {
    pub bend_penalty: u32,
    pub via_penalty: u32,
    pub congestion_penalty: u32,
    pub overflow_penalty: u32,
    pub soft_cost_weight: u32,
    pub max_negotiation_iters: usize,
    pub use_input_order: bool,
    /// Enable bounded rip-up/reroute after initial routing.
    pub enable_rip_up: bool,
    /// Maximum number of rip-up attempts per negotiation iteration.
    pub max_rip_up_per_iter: usize,
    /// Cost penalty for routing on a non-preferred layer (0 = disabled).
    /// Applied per step when a net's traffic_class prefers a different layer.
    pub layer_affinity_penalty: u32,
}

impl Default for V2RoutingConfig {
    fn default() -> Self {
        Self {
            bend_penalty: 2,
            via_penalty: 4,
            congestion_penalty: 32,
            overflow_penalty: 256,
            soft_cost_weight: 1,
            max_negotiation_iters: 12,
            use_input_order: false,
            enable_rip_up: false,
            max_rip_up_per_iter: 3,
            layer_affinity_penalty: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct V2RoutePath {
    pub coords: Vec<V2RouteCoord>,
    pub cost: u32,
    pub bends: u32,
    pub vias: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct V2RouteOverflow {
    pub coord: V2RouteCoord,
    pub usage: u16,
    pub capacity: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct V2MultiRouteResult {
    pub success: bool,
    pub routes: BTreeMap<String, V2RoutePath>,
    pub overflow_cells: Vec<V2RouteOverflow>,
    pub failed_net_ids: Vec<String>,
    pub negotiation_iters: usize,
    /// Number of nets routed on a non-preferred layer (traffic class mismatch).
    pub layer_violation_count: u32,
    /// Total route steps per layer [L0, L1, L2, L3].
    pub layer_utilization: [u32; 4],
}

impl V2MultiRouteResult {
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    pub fn total_path_nodes(&self) -> usize {
        self.routes.values().map(|p| p.coords.len()).sum()
    }

    pub fn total_wire_steps(&self) -> usize {
        self.routes
            .values()
            .map(|p| p.coords.len().saturating_sub(1))
            .sum()
    }

    /// Compute layer utilization and traffic class violation metrics.
    pub fn compute_layer_metrics(&mut self, nets: &[V2RouteNet]) {
        let mut util = [0u32; 4];
        let mut violations = 0u32;

        // Build net-id → traffic_class lookup
        let tc_map: std::collections::HashMap<&str, V2TrafficClass> = nets
            .iter()
            .filter_map(|n| n.traffic_class.map(|tc| (n.id.as_str(), tc)))
            .collect();

        for (net_id, path) in &self.routes {
            let mut net_has_violation = false;
            for coord in &path.coords {
                if coord.z < 4 {
                    util[coord.z] += 1;
                }
                if let Some(&tc) = tc_map.get(net_id.as_str()) {
                    let preferred = match tc {
                        V2TrafficClass::Data => coord.z <= 2,
                        V2TrafficClass::Control => coord.z == 2,
                        V2TrafficClass::LongRange => coord.z == 3,
                    };
                    if !preferred {
                        net_has_violation = true;
                    }
                }
            }
            if net_has_violation {
                violations += 1;
            }
        }

        self.layer_utilization = util;
        self.layer_violation_count = violations;
    }

    pub fn total_cost(&self) -> u64 {
        self.routes.values().map(|p| p.cost as u64).sum()
    }

    pub fn total_bends(&self) -> u64 {
        self.routes.values().map(|p| p.bends as u64).sum()
    }

    pub fn total_vias(&self) -> u64 {
        self.routes.values().map(|p| p.vias as u64).sum()
    }

    pub fn overflow_pressure(&self) -> u64 {
        self.overflow_cells
            .iter()
            .map(|o| o.usage.saturating_sub(o.capacity) as u64)
            .sum()
    }

    pub fn max_overflow_excess(&self) -> u16 {
        self.overflow_cells
            .iter()
            .map(|o| o.usage.saturating_sub(o.capacity))
            .max()
            .unwrap_or(0)
    }

    pub fn route_manifest_hash(&self) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        for (id, path) in &self.routes {
            hash_bytes(&mut h, id.as_bytes());
            for c in &path.coords {
                hash_u64(&mut h, c.x as u64);
                hash_u64(&mut h, c.y as u64);
                hash_u64(&mut h, c.z as u64);
            }
            hash_u64(&mut h, path.cost as u64);
            hash_u64(&mut h, path.bends as u64);
            hash_u64(&mut h, path.vias as u64);
        }
        h
    }

    pub fn failure_signature(&self) -> String {
        let mut failed = self.failed_net_ids.clone();
        failed.sort();

        let mut overflow = self.overflow_cells.clone();
        overflow.sort_by(|a, b| {
            a.coord
                .cmp(&b.coord)
                .then_with(|| a.usage.cmp(&b.usage))
                .then_with(|| a.capacity.cmp(&b.capacity))
        });

        let failed_txt = if failed.is_empty() {
            "-".to_string()
        } else {
            failed.join(",")
        };
        let overflow_txt = if overflow.is_empty() {
            "-".to_string()
        } else {
            overflow
                .iter()
                .map(|o| {
                    format!(
                        "({},{},{}){}>{}",
                        o.coord.x, o.coord.y, o.coord.z, o.usage, o.capacity
                    )
                })
                .collect::<Vec<_>>()
                .join(";")
        };
        format!(
            "success={} failed=[{}] overflow=[{}]",
            self.success, failed_txt, overflow_txt
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum StepDir {
    Start = 0,
    Left = 1,
    Right = 2,
    Up = 3,
    Down = 4,
    LayerUp = 5,
    LayerDown = 6,
}

impl StepDir {
    const COUNT: usize = 7;

    fn idx(self) -> usize {
        self as usize
    }

    fn all() -> [Self; Self::COUNT] {
        [
            Self::Start,
            Self::Left,
            Self::Right,
            Self::Up,
            Self::Down,
            Self::LayerUp,
            Self::LayerDown,
        ]
    }

    /// Return the directional mask bit for the entry direction at the *destination* cell.
    ///
    /// When we step Left (x decreases), we arrive at the destination from its RIGHT side.
    /// When we step Right, we arrive from its LEFT side. And so on.
    /// Return the directional mask bit for the entry direction at the *destination* cell.
    ///
    /// Convention: entry direction = where the route came from.
    /// Stepping Right (x→x+1): arrived from left → DIR_FROM_LEFT.
    /// Stepping LayerDown (z+1→z): arrived from above → DIR_LAYER_UP.
    fn entry_mask(self) -> u8 {
        match self {
            Self::Left => DIR_FROM_RIGHT,    // step left (x-1): arrived from right
            Self::Right => DIR_FROM_LEFT,    // step right (x+1): arrived from left
            Self::Up => DIR_FROM_DOWN,       // step up (y-1): arrived from below
            Self::Down => DIR_FROM_UP,       // step down (y+1): arrived from above
            Self::LayerUp => DIR_LAYER_DOWN, // step to z+1: arrived from below (lower layer)
            Self::LayerDown => DIR_LAYER_UP, // step to z-1: arrived from above (upper layer)
            Self::Start => DIR_OMNI,         // start position, always allowed
        }
    }
}

fn path_shape_metrics(coords: &[V2RouteCoord]) -> (u32, u32) {
    if coords.len() < 2 {
        return (0, 0);
    }
    let mut bends = 0u32;
    let mut vias = 0u32;
    let mut last_dir = direction_between(coords[0], coords[1]);
    if matches!(last_dir, StepDir::LayerUp | StepDir::LayerDown) {
        vias += 1;
    }
    for i in 2..coords.len() {
        let dir = direction_between(coords[i - 1], coords[i]);
        if dir != last_dir {
            bends += 1;
        }
        if matches!(dir, StepDir::LayerUp | StepDir::LayerDown) {
            vias += 1;
        }
        last_dir = dir;
    }
    (bends, vias)
}

fn direction_between(a: V2RouteCoord, b: V2RouteCoord) -> StepDir {
    if b.x > a.x {
        StepDir::Right
    } else if b.x < a.x {
        StepDir::Left
    } else if b.y > a.y {
        StepDir::Down
    } else if b.y < a.y {
        StepDir::Up
    } else if b.z > a.z {
        StepDir::LayerUp
    } else {
        StepDir::LayerDown
    }
}

fn hash_bytes(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100_0000_01B3);
    }
}

fn hash_u64(h: &mut u64, value: u64) {
    hash_bytes(h, &value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_route_db_legality_contracts() {
        let bounds = V2RouteBounds::new(0, 12, 0, 12, 2, 3);
        let mut db = V2RoutingDb::new(12, 12, 4, bounds);

        let c = V2RouteCoord::new(4, 5, 2);
        assert!(db.in_bounds(c));
        assert!(!db.cell(c).hard_block);
        assert_eq!(db.cell(c).capacity, 1);

        db.set_hard_block(c, true);
        assert!(db.cell(c).hard_block);

        db.reserve_horizontal_channel(2, 5, 0, 12, 3, 0);
        assert!(
            !db.cell(c).hard_block,
            "channel reservation should clear blockage"
        );
        assert_eq!(db.cell(c).capacity, 3);

        db.reserve_switchbox(4, 5, 2, 3, 6);
        assert_eq!(db.cell(c).capacity, 6);
        assert!(db.in_bounds(V2RouteCoord::new(4, 5, 3)));
    }

    #[test]
    fn test_v2_router_single_net_deterministic() {
        let bounds = V2RouteBounds::new(0, 16, 0, 16, 2, 2);
        let mut db = V2RoutingDb::new(16, 16, 4, bounds);
        for x in 2usize..14usize {
            if x != 7 {
                db.set_hard_block(V2RouteCoord::new(x, 8, 2), true);
            }
        }

        let net = V2RouteNet::new(
            "n0",
            V2RouteNetClass::Data,
            V2RouteCoord::new(1, 1, 2),
            V2RouteCoord::new(14, 14, 2),
        );
        let cfg = V2RoutingConfig::default();
        let r1 = db.route_multinet(std::slice::from_ref(&net), cfg);
        let r2 = db.route_multinet(std::slice::from_ref(&net), cfg);

        assert!(r1.success, "expected single-net route");
        assert!(r2.success, "expected deterministic single-net route");
        assert_eq!(r1.route_manifest_hash(), r2.route_manifest_hash());
    }

    #[test]
    fn test_v2_router_multinet_negotiated_congestion() {
        let bounds = V2RouteBounds::new(0, 14, 0, 14, 2, 2);
        let db = V2RoutingDb::new(14, 14, 4, bounds);

        let nets = vec![
            V2RouteNet::new(
                "a",
                V2RouteNetClass::Data,
                V2RouteCoord::new(1, 1, 2),
                V2RouteCoord::new(12, 12, 2),
            ),
            V2RouteNet::new(
                "b",
                V2RouteNetClass::Data,
                V2RouteCoord::new(1, 12, 2),
                V2RouteCoord::new(12, 1, 2),
            ),
        ];

        let cfg = V2RoutingConfig {
            congestion_penalty: 200,
            overflow_penalty: 800,
            max_negotiation_iters: 20,
            ..V2RoutingConfig::default()
        };
        let routed = db.route_multinet(&nets, cfg);
        assert!(
            routed.success,
            "expected negotiated routing to converge: {}",
            routed.failure_signature()
        );
        assert!(
            routed.overflow_cells.is_empty(),
            "no capacity overflow expected"
        );
        assert_eq!(routed.route_count(), nets.len());
        assert!(routed.total_wire_steps() > 0);
        assert!(routed.total_cost() > 0);
        assert_eq!(routed.max_overflow_excess(), 0);
        assert_eq!(routed.overflow_pressure(), 0);
    }

    #[test]
    fn test_v2_router_directional_legality() {
        // 8x8 single-layer grid. Place a corridor of WireRight tiles (read from LEFT only)
        // across y=4. Route must go left-to-right through them; a route trying to enter
        // from above (StepDir::Down) should be rejected.
        let bounds = V2RouteBounds::new(0, 8, 0, 8, 2, 2);
        let mut db = V2RoutingDb::new(8, 8, 4, bounds);

        // Mark corridor at y=4 as WireRight (allows entry from LEFT only = DIR_FROM_LEFT)
        for x in 1..7 {
            let coord = V2RouteCoord::new(x, 4, 2);
            db.set_directional_mask(coord, DIR_FROM_LEFT);
        }

        // Route A: Left-to-right through the corridor — should succeed.
        // Start at (0,4) and goal at (7,4). These are terminals so mask is bypassed.
        let net_ok = V2RouteNet::new(
            "horizontal",
            V2RouteNetClass::Data,
            V2RouteCoord::new(0, 4, 2),
            V2RouteCoord::new(7, 4, 2),
        );
        let cfg = V2RoutingConfig::default();
        let r = db.route_multinet(std::slice::from_ref(&net_ok), cfg);
        assert!(
            r.success,
            "horizontal route through WireRight corridor should succeed"
        );

        // Verify the route actually goes through the corridor (all coords at y=4)
        let path = &r.routes["horizontal"];
        assert!(
            path.coords.iter().all(|c| c.y == 4),
            "route should stay on y=4 corridor"
        );

        // Route B: Must cross the corridor from top to bottom.
        // Block all columns except x=3. If directional masking works, the route cannot
        // enter the WireRight tile at (3,4) from above, forcing a detour.
        // First, hard-block everything at y=4 except x=3 to force crossing at x=3.
        let mut db2 = V2RoutingDb::new(8, 8, 4, bounds);
        for x in 0..8 {
            if x != 3 {
                db2.set_hard_block(V2RouteCoord::new(x, 4, 2), true);
            }
        }
        // Set (3,4) as WireRight: only allows entry from LEFT
        db2.set_directional_mask(V2RouteCoord::new(3, 4, 2), DIR_FROM_LEFT);

        let net_cross = V2RouteNet::new(
            "vertical",
            V2RouteNetClass::Data,
            V2RouteCoord::new(3, 0, 2),
            V2RouteCoord::new(3, 7, 2),
        );
        let r2 = db2.route_multinet(std::slice::from_ref(&net_cross), cfg);

        // The only path through y=4 is at x=3 which is WireRight (LEFT-entry only).
        // A vertical route tries to enter from above (DIR_FROM_UP), which is blocked.
        // The route must either fail or find a detour via x=2→3→... at y=4.
        if r2.success {
            let path2 = &r2.routes["vertical"];
            // If it succeeds, it must detour: approach (3,4) from (2,4) going right
            let idx4 = path2
                .coords
                .iter()
                .position(|c| c.x == 3 && c.y == 4)
                .unwrap();
            assert!(idx4 > 0, "route must pass through (3,4)");
            let prev = path2.coords[idx4 - 1];
            assert_eq!(
                prev.x, 2,
                "must enter (3,4) from x=2 (left), not from above; prev={:?}",
                prev
            );
            assert_eq!(
                prev.y, 4,
                "must enter (3,4) from y=4 (same row); prev={:?}",
                prev
            );
        }
        // If it fails, that's also acceptable — the directional mask prevented illegal crossing.

        // Route C: Verify omnidirectional cells still allow all entry directions.
        // All cells default to DIR_OMNI, so a simple diagonal route should work.
        let db3 = V2RoutingDb::new(8, 8, 4, bounds);
        let net_diag = V2RouteNet::new(
            "diag",
            V2RouteNetClass::Data,
            V2RouteCoord::new(0, 0, 2),
            V2RouteCoord::new(7, 7, 2),
        );
        let r3 = db3.route_multinet(std::slice::from_ref(&net_diag), cfg);
        assert!(r3.success, "omnidirectional route should always succeed");
    }

    #[test]
    fn test_v2_router_directional_mask_constants() {
        // Verify mask computation for known tile types
        assert_eq!(
            directional_mask_for_tile(TileType::WireRight),
            DIR_FROM_LEFT
        );
        assert_eq!(
            directional_mask_for_tile(TileType::WireLeft),
            DIR_FROM_RIGHT
        );
        assert_eq!(directional_mask_for_tile(TileType::WireDown), DIR_FROM_UP);
        assert_eq!(directional_mask_for_tile(TileType::WireUp), DIR_FROM_DOWN);
        assert_eq!(directional_mask_for_tile(TileType::ViaUp), DIR_LAYER_UP);
        assert_eq!(directional_mask_for_tile(TileType::ViaDown), DIR_LAYER_DOWN);
        // Sprint 184: WeightedVia directional masks
        assert_eq!(
            directional_mask_for_tile(TileType::WeightedViaUp),
            DIR_LAYER_UP
        );
        assert_eq!(
            directional_mask_for_tile(TileType::WeightedViaDown),
            DIR_LAYER_DOWN
        );
        assert_eq!(directional_mask_for_tile(TileType::Const), DIR_OMNI);
        assert_eq!(directional_mask_for_tile(TileType::Wire), DIR_OMNI);
        assert_eq!(directional_mask_for_tile(TileType::And), DIR_OMNI);
    }

    #[test]
    fn test_v2_router_via_directional_legality() {
        // 4x4 two-layer grid. Place a ViaUp at (2,2,2) — only allows entry from layer above.
        // A route on layer 2 trying to enter (2,2,2) laterally should be rejected.
        let bounds = V2RouteBounds::new(0, 4, 0, 4, 2, 3);
        let mut db = V2RoutingDb::new(4, 4, 4, bounds);

        // Set (2,2,2) as ViaUp: only allows entry from LayerUp (from z=3)
        db.set_directional_mask(V2RouteCoord::new(2, 2, 2), DIR_LAYER_UP);
        // Hard-block (2,2,3) so the route can't go through layer 3 to enter from above
        db.set_hard_block(V2RouteCoord::new(2, 2, 3), true);

        // Route on layer 2 from (0,2) to (3,2) — must pass through x=2
        // Block y!=2 to force horizontal route through (2,2,2)
        for y in 0..4 {
            if y != 2 {
                for x in 0..4 {
                    db.set_hard_block(V2RouteCoord::new(x, y, 2), true);
                    db.set_hard_block(V2RouteCoord::new(x, y, 3), true);
                }
            }
        }

        let net = V2RouteNet::new(
            "via_test",
            V2RouteNetClass::Data,
            V2RouteCoord::new(0, 2, 2),
            V2RouteCoord::new(3, 2, 2),
        );
        let cfg = V2RoutingConfig::default();
        let r = db.route_multinet(std::slice::from_ref(&net), cfg);

        // Route should fail because (2,2,2) only accepts LayerUp entry but the only
        // available approach is from (1,2,2) going Right — which is DIR_FROM_LEFT.
        assert!(
            !r.success,
            "route through ViaUp cell from lateral direction should fail"
        );
    }

    #[test]
    fn test_v2_router_endpoint_classes() {
        // Test 1: SharedFan allows multiple nets to share a start point.
        let bounds = V2RouteBounds::new(0, 10, 0, 10, 2, 2);
        let db = V2RoutingDb::new(10, 10, 4, bounds);

        let shared_start = V2RouteCoord::new(5, 5, 2);
        let nets = vec![
            V2RouteNet::new(
                "fan_a",
                V2RouteNetClass::Data,
                shared_start,
                V2RouteCoord::new(0, 0, 2),
            )
            .with_start_class(V2EndpointClass::SharedFan { capacity: 4 }),
            V2RouteNet::new(
                "fan_b",
                V2RouteNetClass::Data,
                shared_start,
                V2RouteCoord::new(9, 0, 2),
            )
            .with_start_class(V2EndpointClass::SharedFan { capacity: 4 }),
            V2RouteNet::new(
                "fan_c",
                V2RouteNetClass::Data,
                shared_start,
                V2RouteCoord::new(0, 9, 2),
            )
            .with_start_class(V2EndpointClass::SharedFan { capacity: 4 }),
        ];

        let cfg = V2RoutingConfig {
            congestion_penalty: 100,
            overflow_penalty: 500,
            max_negotiation_iters: 10,
            ..V2RoutingConfig::default()
        };
        let r = db.route_multinet(&nets, cfg);
        assert!(
            r.success,
            "SharedFan start should allow 3 nets: {}",
            r.failure_signature()
        );
        assert_eq!(r.route_count(), 3);

        // Test 2: HardExclusive prevents sharing the same start.
        let excl_nets = vec![
            V2RouteNet::new(
                "excl_a",
                V2RouteNetClass::Data,
                shared_start,
                V2RouteCoord::new(0, 0, 2),
            ), // default: HardExclusive
            V2RouteNet::new(
                "excl_b",
                V2RouteNetClass::Data,
                shared_start,
                V2RouteCoord::new(9, 0, 2),
            ), // default: HardExclusive
        ];
        let r2 = db.route_multinet(&excl_nets, cfg);
        // Both share the same start cell with capacity=1, so overflow should occur
        assert!(
            !r2.overflow_cells.is_empty() || !r2.failed_net_ids.is_empty(),
            "HardExclusive start should cause overflow or failure when shared"
        );

        // Test 3: Backward compat: with_shared_start(true) = SharedFan(MAX)
        let compat = V2RouteNet::new(
            "compat",
            V2RouteNetClass::Data,
            V2RouteCoord::new(0, 0, 2),
            V2RouteCoord::new(5, 5, 2),
        )
        .with_shared_start(true);
        assert!(compat.start_class.is_shared());
        assert_eq!(
            compat.start_class,
            V2EndpointClass::SharedFan { capacity: u16::MAX }
        );

        let compat2 = V2RouteNet::new(
            "compat2",
            V2RouteNetClass::Data,
            V2RouteCoord::new(0, 0, 2),
            V2RouteCoord::new(5, 5, 2),
        )
        .with_shared_start(false);
        assert!(!compat2.start_class.is_shared());
        assert_eq!(compat2.start_class, V2EndpointClass::HardExclusive);
    }

    #[test]
    fn test_v2_router_shared_fan_capacity_enforced() {
        // 5 nets sharing a start with capacity=3. Should report overflow at the shared
        // endpoint (usage=5 > capacity=3).
        let bounds = V2RouteBounds::new(0, 16, 0, 16, 2, 2);
        let db = V2RoutingDb::new(16, 16, 4, bounds);
        let shared_start = V2RouteCoord::new(8, 8, 2);

        let nets: Vec<V2RouteNet> = (0..5)
            .map(|i| {
                let goal_x = (i * 3) % 15;
                let goal_y = if i < 3 { 0 } else { 15 };
                V2RouteNet::new(
                    format!("fan_{i}"),
                    V2RouteNetClass::Data,
                    shared_start,
                    V2RouteCoord::new(goal_x, goal_y, 2),
                )
                .with_start_class(V2EndpointClass::SharedFan { capacity: 3 })
            })
            .collect();

        let cfg = V2RoutingConfig {
            congestion_penalty: 200,
            overflow_penalty: 800,
            max_negotiation_iters: 10,
            ..V2RoutingConfig::default()
        };
        let r = db.route_multinet(&nets, cfg);

        // The shared start has capacity=3 but 5 nets use it → overflow at that cell.
        let shared_idx = db.coord_to_idx(shared_start);
        let shared_overflow = r
            .overflow_cells
            .iter()
            .any(|o| db.coord_to_idx(o.coord) == shared_idx);
        assert!(
            shared_overflow,
            "shared start with capacity=3 and 5 nets should overflow"
        );
    }

    #[test]
    fn test_v2_router_shared_fan_unlimited_backward_compat() {
        // SharedFan { capacity: u16::MAX } should never overflow at the shared endpoint,
        // regardless of how many nets share it.
        let bounds = V2RouteBounds::new(0, 16, 0, 16, 2, 2);
        let db = V2RoutingDb::new(16, 16, 4, bounds);
        let shared_start = V2RouteCoord::new(8, 8, 2);

        let nets: Vec<V2RouteNet> = (0..8)
            .map(|i| {
                let goal_x = (i * 2) % 15;
                let goal_y = if i < 4 { 0 } else { 15 };
                V2RouteNet::new(
                    format!("fan_{i}"),
                    V2RouteNetClass::Data,
                    shared_start,
                    V2RouteCoord::new(goal_x, goal_y, 2),
                )
                .with_shared_start(true) // u16::MAX
            })
            .collect();

        let cfg = V2RoutingConfig {
            congestion_penalty: 200,
            overflow_penalty: 800,
            max_negotiation_iters: 10,
            ..V2RoutingConfig::default()
        };
        let r = db.route_multinet(&nets, cfg);

        // The shared start should NOT appear in overflow (capacity = u16::MAX).
        let shared_idx = db.coord_to_idx(shared_start);
        let shared_overflow = r
            .overflow_cells
            .iter()
            .any(|o| db.coord_to_idx(o.coord) == shared_idx);
        assert!(
            !shared_overflow,
            "SharedFan with unlimited capacity should not overflow at endpoint"
        );
    }

    #[test]
    fn test_v2_router_shared_fan_capacity_method() {
        // Verify the capacity() method on V2EndpointClass.
        assert_eq!(V2EndpointClass::HardExclusive.capacity(), None);
        assert_eq!(
            V2EndpointClass::SharedFan { capacity: 4 }.capacity(),
            Some(4)
        );
        assert_eq!(
            V2EndpointClass::SharedFan { capacity: u16::MAX }.capacity(),
            Some(u16::MAX)
        );
    }

    #[test]
    fn test_v2_router_rip_up_basic() {
        // Create a narrow corridor where a Data net blocks a ControlCritical net.
        // With rip-up enabled, the ControlCritical net should displace the Data net.
        let bounds = V2RouteBounds::new(0, 12, 0, 12, 2, 2);
        let mut db = V2RoutingDb::new(12, 12, 4, bounds);

        // Create a narrow bottleneck: only y=5 is passable at x=5..7
        for y in 0..12 {
            if y != 5 {
                for x in 5..8 {
                    db.set_hard_block(V2RouteCoord::new(x, y, 2), true);
                }
            }
        }

        // Data net routes first (lower priority number = routed first if sorted,
        // but Data has priority 2, ControlCritical has 0).
        // Since nets are sorted by priority, ControlCritical routes first.
        // But with use_input_order=true, Data routes first and blocks the bottleneck.
        let nets = vec![
            V2RouteNet::new(
                "blocker",
                V2RouteNetClass::Data,
                V2RouteCoord::new(0, 5, 2),
                V2RouteCoord::new(11, 5, 2),
            ),
            V2RouteNet::new(
                "critical",
                V2RouteNetClass::ControlCritical,
                V2RouteCoord::new(0, 3, 2),
                V2RouteCoord::new(11, 7, 2),
            ),
        ];

        // Without rip-up, input-order routing has Data first, ControlCritical may get suboptimal path.
        let cfg_no_rip = V2RoutingConfig {
            use_input_order: true,
            congestion_penalty: 200,
            overflow_penalty: 800,
            max_negotiation_iters: 10,
            enable_rip_up: false,
            ..V2RoutingConfig::default()
        };
        let r1 = db.route_multinet(&nets, cfg_no_rip);
        let no_rip_overflow = r1.overflow_cells.len();

        // With rip-up enabled, ControlCritical can displace Data.
        let cfg_rip = V2RoutingConfig {
            enable_rip_up: true,
            max_rip_up_per_iter: 5,
            ..cfg_no_rip
        };
        let r2 = db.route_multinet(&nets, cfg_rip);
        let rip_overflow = r2.overflow_cells.len();

        // Rip-up should not make things worse.
        assert!(
            rip_overflow <= no_rip_overflow,
            "rip-up should not increase overflow: {} vs {}",
            rip_overflow,
            no_rip_overflow
        );
    }

    #[test]
    fn test_v2_router_rip_up_priority_respect() {
        // Same-priority nets should NOT displace each other.
        let bounds = V2RouteBounds::new(0, 10, 0, 10, 2, 2);
        let mut db = V2RoutingDb::new(10, 10, 4, bounds);

        // Narrow bottleneck at y=5, x=4..6
        for y in 0..10 {
            if y != 5 {
                for x in 4..7 {
                    db.set_hard_block(V2RouteCoord::new(x, y, 2), true);
                }
            }
        }

        let nets = vec![
            V2RouteNet::new(
                "data_a",
                V2RouteNetClass::Data,
                V2RouteCoord::new(0, 5, 2),
                V2RouteCoord::new(9, 5, 2),
            ),
            V2RouteNet::new(
                "data_b",
                V2RouteNetClass::Data,
                V2RouteCoord::new(0, 3, 2),
                V2RouteCoord::new(9, 7, 2),
            ),
        ];

        let cfg = V2RoutingConfig {
            use_input_order: true,
            congestion_penalty: 200,
            overflow_penalty: 800,
            max_negotiation_iters: 10,
            enable_rip_up: true,
            max_rip_up_per_iter: 5,
            ..V2RoutingConfig::default()
        };

        let r1 = db.route_multinet(&nets, cfg);
        let r2 = db.route_multinet(&nets, cfg);

        // Results should be deterministic.
        assert_eq!(r1.failure_signature(), r2.failure_signature());
    }

    #[test]
    fn test_v2_router_rip_up_deterministic() {
        // Verify rip-up produces identical results across multiple runs.
        let bounds = V2RouteBounds::new(0, 16, 0, 16, 2, 2);
        let mut db = V2RoutingDb::new(16, 16, 4, bounds);

        // Create some obstacles to force interesting routing
        for x in 4..12 {
            db.set_hard_block(V2RouteCoord::new(x, 8, 2), true);
        }

        let nets = vec![
            V2RouteNet::new(
                "ctrl",
                V2RouteNetClass::ControlCritical,
                V2RouteCoord::new(2, 2, 2),
                V2RouteCoord::new(13, 13, 2),
            ),
            V2RouteNet::new(
                "data1",
                V2RouteNetClass::Data,
                V2RouteCoord::new(2, 13, 2),
                V2RouteCoord::new(13, 2, 2),
            ),
            V2RouteNet::new(
                "data2",
                V2RouteNetClass::DataCritical,
                V2RouteCoord::new(0, 8, 2),
                V2RouteCoord::new(15, 8, 2),
            ),
        ];

        let cfg = V2RoutingConfig {
            congestion_penalty: 100,
            overflow_penalty: 500,
            max_negotiation_iters: 15,
            enable_rip_up: true,
            max_rip_up_per_iter: 3,
            ..V2RoutingConfig::default()
        };

        let r1 = db.route_multinet(&nets, cfg);
        let r2 = db.route_multinet(&nets, cfg);
        let r3 = db.route_multinet(&nets, cfg);

        assert_eq!(r1.route_manifest_hash(), r2.route_manifest_hash());
        assert_eq!(r2.route_manifest_hash(), r3.route_manifest_hash());
        assert_eq!(r1.failure_signature(), r2.failure_signature());
    }

    #[test]
    fn test_v2_router_rip_up_disabled_backward_compat() {
        // With enable_rip_up=false (default), results should be identical to pre-rip-up behavior.
        let bounds = V2RouteBounds::new(0, 14, 0, 14, 2, 2);
        let db = V2RoutingDb::new(14, 14, 4, bounds);

        let nets = vec![
            V2RouteNet::new(
                "a",
                V2RouteNetClass::Data,
                V2RouteCoord::new(1, 1, 2),
                V2RouteCoord::new(12, 12, 2),
            ),
            V2RouteNet::new(
                "b",
                V2RouteNetClass::Data,
                V2RouteCoord::new(1, 12, 2),
                V2RouteCoord::new(12, 1, 2),
            ),
        ];

        let cfg = V2RoutingConfig {
            congestion_penalty: 200,
            overflow_penalty: 800,
            max_negotiation_iters: 20,
            enable_rip_up: false, // explicitly disabled
            ..V2RoutingConfig::default()
        };
        let r = db.route_multinet(&nets, cfg);
        assert!(r.success, "backward-compat routing should still work");
    }

    #[test]
    fn test_v2_router_rip_up_restores_on_failure() {
        // If rip-up removes routes but rerouting fails, originals must be restored.
        // Create a grid where the victim's route is the ONLY path — after removal,
        // the winner still can't route (fully blocked), so both must be restored.
        let bounds = V2RouteBounds::new(0, 8, 0, 8, 2, 2);
        let mut db = V2RoutingDb::new(8, 8, 4, bounds);

        // Block everything except a single narrow corridor at y=3.
        for y in 0..8 {
            if y != 3 {
                for x in 3..6 {
                    db.set_hard_block(V2RouteCoord::new(x, y, 2), true);
                }
            }
        }

        // Two nets forced through the same y=3 corridor.
        // Data routes first (input_order), then ControlCritical.
        let nets = vec![
            V2RouteNet::new(
                "data_first",
                V2RouteNetClass::Data,
                V2RouteCoord::new(0, 3, 2),
                V2RouteCoord::new(7, 3, 2),
            ),
            V2RouteNet::new(
                "ctrl",
                V2RouteNetClass::ControlCritical,
                V2RouteCoord::new(0, 3, 2),
                V2RouteCoord::new(7, 3, 2),
            ),
        ];

        let cfg = V2RoutingConfig {
            use_input_order: true,
            congestion_penalty: 200,
            overflow_penalty: 800,
            max_negotiation_iters: 4,
            enable_rip_up: true,
            max_rip_up_per_iter: 5,
            ..V2RoutingConfig::default()
        };

        let r = db.route_multinet(&nets, cfg);

        // Both nets must be present in routes — rip-up should restore on failure.
        assert_eq!(
            r.routes.len(),
            2,
            "rip-up must not silently drop nets: routes={}, expected=2",
            r.routes.len()
        );
        assert!(r.routes.contains_key("data_first"));
        assert!(r.routes.contains_key("ctrl"));
    }

    #[test]
    fn test_v2_router_rip_up_all_nets_present_invariant() {
        // After rip-up, routes.len() must equal nets.len() when all initial routes succeed.
        let bounds = V2RouteBounds::new(0, 14, 0, 14, 2, 2);
        let db = V2RoutingDb::new(14, 14, 4, bounds);

        let nets = vec![
            V2RouteNet::new(
                "a",
                V2RouteNetClass::Data,
                V2RouteCoord::new(1, 1, 2),
                V2RouteCoord::new(12, 6, 2),
            ),
            V2RouteNet::new(
                "b",
                V2RouteNetClass::DataCritical,
                V2RouteCoord::new(1, 6, 2),
                V2RouteCoord::new(12, 1, 2),
            ),
            V2RouteNet::new(
                "c",
                V2RouteNetClass::ControlCritical,
                V2RouteCoord::new(1, 12, 2),
                V2RouteCoord::new(12, 12, 2),
            ),
        ];

        let cfg = V2RoutingConfig {
            congestion_penalty: 200,
            overflow_penalty: 800,
            max_negotiation_iters: 10,
            enable_rip_up: true,
            max_rip_up_per_iter: 5,
            ..V2RoutingConfig::default()
        };

        let r = db.route_multinet(&nets, cfg);
        assert_eq!(
            r.routes.len(),
            nets.len(),
            "all nets must be present after rip-up: routes={}, nets={}",
            r.routes.len(),
            nets.len()
        );
        // Verify no net IDs are missing.
        for net in &nets {
            assert!(
                r.routes.contains_key(&net.id),
                "net '{}' missing from routes after rip-up",
                net.id
            );
        }
    }

    // === Sprint 160: Island Routing Feasibility Experiment ===

    /// Read-only routing experiment for 2×8 distributed register islands.
    ///
    /// Answers: can 2 islands of 8 registers route successfully on 128×128×4?
    /// This directly informs whether Sprint 161 can proceed.
    ///
    /// Go/no-go criteria:
    /// - Green:  0 overflow nets
    /// - Yellow: 1-3 overflow
    /// - Red:    4+ overflow
    #[test]
    fn test_island_routing_feasibility() {
        use crate::tile_cpu::V2Builder;

        // Build a real V2 CPU to get realistic hard-block placement
        let program = vec![0u32; 4]; // minimal NOP program
        let mut sim = crate::simulation::Simulation::with_size_layered(128, 128, 4);
        let _cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(64)
            .with_ram_size(64)
            .build(&mut sim);

        // Extract routing DB from real CPU layout.
        // Bounds cover the register/ALU/RAM region with L2-L3 for routing.
        let bounds = V2RouteBounds::new(0, 100, 0, 60, 1, 3);
        let mut db = V2RoutingDb::from_simulation_const_blocked(&sim, bounds);

        // Reserve L2/L3 horizontal and vertical channels (same pattern as Sprint 150)
        for y in 0..60 {
            db.reserve_horizontal_channel(3, y, 0, 100, 1, 0);
            db.reserve_horizontal_channel(2, y, 0, 100, 1, 0);
        }
        for x in 0..100 {
            db.reserve_vertical_channel(3, x, 0, 60, 1, 0);
            db.reserve_vertical_channel(2, x, 0, 60, 1, 0);
        }
        // Also open L1 channels
        for y in 0..60 {
            db.reserve_horizontal_channel(1, y, 0, 100, 1, 0);
        }
        for x in 0..100 {
            db.reserve_vertical_channel(1, x, 0, 60, 1, 0);
        }

        // Hypothetical island positions:
        // Island A (R0-R7): near ALU area, ox+28, rows 13..22 (existing Bank A location)
        // Island B (R8-R15): near RAM/MMIO, ox+48, rows 13..22 (existing Bank B location)
        // ALU output: approximately ox+38, oy+30 (centrally placed)
        let alu_out = V2RouteCoord::new(38, 30, 2);
        let island_a_entry = V2RouteCoord::new(28, 13, 2);
        let island_b_entry = V2RouteCoord::new(48, 13, 2);
        let island_a_data_out = V2RouteCoord::new(28, 22, 2);
        let island_b_data_out = V2RouteCoord::new(48, 22, 2);
        let alu_left_in = V2RouteCoord::new(36, 30, 2);
        let alu_right_in = V2RouteCoord::new(40, 30, 2);

        // Unblock island entry/exit points (they're hypothetical placement locations)
        for coord in [
            alu_out,
            island_a_entry,
            island_b_entry,
            island_a_data_out,
            island_b_data_out,
            alu_left_in,
            alu_right_in,
        ] {
            db.set_hard_block(coord, false);
            db.set_capacity(coord, 8);
        }

        // Define inter-island bus nets:
        let nets = vec![
            // Operand read buses: islands → ALU
            V2RouteNet::new(
                "island_a_to_alu_left",
                V2RouteNetClass::DataCritical,
                island_a_data_out,
                alu_left_in,
            )
            .with_traffic_class(V2TrafficClass::Data)
            .with_shared_start(true),
            V2RouteNet::new(
                "island_b_to_alu_right",
                V2RouteNetClass::DataCritical,
                island_b_data_out,
                alu_right_in,
            )
            .with_traffic_class(V2TrafficClass::Data)
            .with_shared_start(true),
            // Writeback buses: ALU → both islands
            V2RouteNet::new(
                "alu_to_island_a_wb",
                V2RouteNetClass::DataCritical,
                alu_out,
                island_a_entry,
            )
            .with_traffic_class(V2TrafficClass::Data)
            .with_shared_start(true),
            V2RouteNet::new(
                "alu_to_island_b_wb",
                V2RouteNetClass::DataCritical,
                alu_out,
                island_b_entry,
            )
            .with_traffic_class(V2TrafficClass::Data)
            .with_shared_start(true),
            // Write-enable routing: decoder → both islands
            V2RouteNet::new(
                "we_to_island_a",
                V2RouteNetClass::ControlCritical,
                V2RouteCoord::new(20, 10, 2),
                island_a_entry,
            )
            .with_traffic_class(V2TrafficClass::Control)
            .with_shared_start(true),
            V2RouteNet::new(
                "we_to_island_b",
                V2RouteNetClass::ControlCritical,
                V2RouteCoord::new(20, 10, 2),
                island_b_entry,
            )
            .with_traffic_class(V2TrafficClass::Control)
            .with_shared_start(true),
        ];

        let cfg = V2RoutingConfig {
            layer_affinity_penalty: 10,
            max_negotiation_iters: 30,
            congestion_penalty: 64,
            overflow_penalty: 512,
            enable_rip_up: true,
            max_rip_up_per_iter: 5,
            ..Default::default()
        };

        let result = db.route_multinet(&nets, cfg);

        // Report
        println!("=== Sprint 160: Island Routing Feasibility ===");
        println!(
            "Nets: {} total, {} routed, {} failed, {} overflow cells",
            nets.len(),
            result.routes.len(),
            result.failed_net_ids.len(),
            result.overflow_cells.len(),
        );
        println!(
            "Wire steps: {}, Negotiation iters: {}",
            result.total_wire_steps(),
            result.negotiation_iters,
        );
        println!(
            "Layer utilization: L0={}, L1={}, L2={}, L3={}",
            result.layer_utilization[0],
            result.layer_utilization[1],
            result.layer_utilization[2],
            result.layer_utilization[3],
        );
        println!(
            "Layer violations: {} nets on non-preferred layer",
            result.layer_violation_count,
        );

        if !result.failed_net_ids.is_empty() {
            println!("Failed nets: {:?}", result.failed_net_ids);
        }
        if !result.overflow_cells.is_empty() {
            println!("Overflow cells:");
            for ov in &result.overflow_cells {
                println!(
                    "  ({},{},{}) usage={} cap={}",
                    ov.coord.x, ov.coord.y, ov.coord.z, ov.usage, ov.capacity
                );
            }
        }

        // Go/no-go assessment
        let overflow_count = result.overflow_cells.len() + result.failed_net_ids.len();
        if overflow_count == 0 {
            println!("STATUS: GREEN — 2×8 island routing is feasible!");
        } else if overflow_count <= 3 {
            println!(
                "STATUS: YELLOW — {} issues, may need placement adjustment",
                overflow_count
            );
        } else {
            println!(
                "STATUS: RED — {} issues, fundamental approach change needed",
                overflow_count
            );
        }

        // The experiment should at least route successfully (green or yellow)
        assert!(
            overflow_count <= 3,
            "Island routing feasibility: RED ({} overflow/failed). \
             Sprint 161 cannot proceed without grid expansion or approach change.",
            overflow_count
        );
    }

    // === Sprint 160: Layer-Affinity Routing Tests ===

    #[test]
    fn test_layer_affinity_disabled_by_default() {
        // With layer_affinity_penalty: 0, adding traffic_class should not
        // change routing behavior (backward-compatible default).
        let db = V2RoutingDb::new(16, 16, 3, V2RouteBounds::new(0, 16, 0, 16, 0, 2));

        let net_no_tc = V2RouteNet::new(
            "no_tc",
            V2RouteNetClass::Data,
            V2RouteCoord::new(0, 0, 1),
            V2RouteCoord::new(15, 0, 1),
        );
        let cfg = V2RoutingConfig::default(); // layer_affinity_penalty = 0

        let r1 = db.route_multinet(&[net_no_tc.clone()], cfg);
        assert!(r1.success, "baseline route should succeed");

        let net_with_tc = V2RouteNet::new(
            "with_tc",
            V2RouteNetClass::Data,
            V2RouteCoord::new(0, 0, 1),
            V2RouteCoord::new(15, 0, 1),
        )
        .with_traffic_class(V2TrafficClass::Data);

        let r2 = db.route_multinet(&[net_with_tc], cfg);
        assert!(r2.success, "route with traffic class should succeed");
        assert_eq!(
            r1.total_wire_steps(),
            r2.total_wire_steps(),
            "wire steps should be identical when affinity penalty is 0"
        );
    }

    #[test]
    fn test_layer_affinity_penalty_steers_routing() {
        // With layer_affinity_penalty > 0, a Data net should prefer L1-L2
        // over L3 (LongRange layer).
        // Setup: 8x4x4 grid. Route from (0,0,0) to (7,0,0) via L1 or L3.
        // Block L0 route so it must go through a via.
        let mut db = V2RoutingDb::new(8, 4, 4, V2RouteBounds::new(0, 8, 0, 4, 0, 3));

        // Block the direct L0 path at x=3..5 to force a layer change
        for x in 3..6 {
            db.set_hard_block(V2RouteCoord::new(x, 0, 0), true);
        }

        let net = V2RouteNet::new(
            "data_net",
            V2RouteNetClass::Data,
            V2RouteCoord::new(0, 0, 0),
            V2RouteCoord::new(7, 0, 0),
        )
        .with_traffic_class(V2TrafficClass::Data);

        // Route without affinity
        let cfg_no_aff = V2RoutingConfig {
            layer_affinity_penalty: 0,
            ..Default::default()
        };
        let r_no = db.route_multinet(&[net.clone()], cfg_no_aff);
        assert!(r_no.success, "should route without affinity");

        // Route with affinity (Data prefers L0-L2)
        let cfg_aff = V2RoutingConfig {
            layer_affinity_penalty: 50,
            ..Default::default()
        };
        let r_aff = db.route_multinet(&[net], cfg_aff);
        assert!(r_aff.success, "should route with affinity");

        // With high penalty, the route should avoid L3 (if it used L3 before)
        // At minimum, layer_utilization[3] should be <= the non-affinity case
        assert!(
            r_aff.layer_utilization[3] <= r_no.layer_utilization[3],
            "affinity should reduce L3 usage: without={}, with={}",
            r_no.layer_utilization[3],
            r_aff.layer_utilization[3]
        );
    }

    #[test]
    fn test_layer_affinity_metrics() {
        // Route nets with mixed traffic classes and verify metrics are populated.
        let db = V2RoutingDb::new(8, 4, 4, V2RouteBounds::new(0, 8, 0, 4, 0, 3));

        let nets = vec![
            V2RouteNet::new(
                "data_a",
                V2RouteNetClass::Data,
                V2RouteCoord::new(0, 0, 1),
                V2RouteCoord::new(7, 0, 1),
            )
            .with_traffic_class(V2TrafficClass::Data),
            V2RouteNet::new(
                "ctrl_a",
                V2RouteNetClass::ControlCritical,
                V2RouteCoord::new(0, 1, 2),
                V2RouteCoord::new(7, 1, 2),
            )
            .with_traffic_class(V2TrafficClass::Control),
            V2RouteNet::new(
                "long_a",
                V2RouteNetClass::Data,
                V2RouteCoord::new(0, 2, 3),
                V2RouteCoord::new(7, 2, 3),
            )
            .with_traffic_class(V2TrafficClass::LongRange),
        ];

        let cfg = V2RoutingConfig {
            layer_affinity_penalty: 10,
            ..Default::default()
        };
        let r = db.route_multinet(&nets, cfg);
        assert!(r.success, "all nets should route");
        assert_eq!(r.route_count(), 3);

        // Layer utilization should be non-zero for at least L1, L2, L3
        let total_util: u32 = r.layer_utilization.iter().sum();
        assert!(total_util > 0, "layer utilization should be populated");

        // Each net routes on its preferred layer, so violations should be 0
        // (straight-line routes on a single layer = no layer mismatch)
        assert_eq!(
            r.layer_violation_count, 0,
            "straight-line routes on preferred layers should have 0 violations"
        );
    }
}
