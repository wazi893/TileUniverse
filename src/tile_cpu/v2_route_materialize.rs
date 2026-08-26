//! Sprint 398: route materialization — the last mile between the A* router
//! (`v2_routing`) and the physical fabric.
//!
//! `route_multinet` returns coordinate paths; nothing in the V2 stack turned
//! those into tiles before this module. `materialize_route` converts a
//! `V2RoutePath` into directional wire / via placements with validate-then-
//! commit semantics (no partial mutation on error), and
//! `validate_materialized_route` re-checks the placed chain as a standalone
//! invariant pass (portable to hand-placed routes).
//!
//! Correctness model (why this is more than coordinate → tile mapping):
//! - **Adjacency is connectivity.** A foreign tile that reads a routed cell
//!   picks up the routed signal even though no cell collision occurs. Routing
//!   must therefore use `V2RoutingDb::from_simulation_interference_checked`,
//!   which hard-blocks reader-adjacent cells; the materializer additionally
//!   refuses to overwrite any non-Const cell.
//! - **One via per source.** `via_fwd` holds a single forward slot per source
//!   tile and the conflict check in `rebuild_via_connections` is a
//!   `debug_assert` (compiled out in release), so the conflict is checked
//!   explicitly here before any tile is placed.
//! - **Directional dirty propagation (S133).** A unidirectional wire never
//!   marks its input-side neighbor dirty, so a route must not exit a
//!   directional driver through the driver's input side (`StaleStartTap`).

use std::collections::BTreeMap;

use crate::simulation::Simulation;
use crate::tile_meta::TileType;

use super::v2_routing::{
    DIR_FROM_DOWN, DIR_FROM_LEFT, DIR_FROM_RIGHT, DIR_FROM_UP, DIR_LAYER_DOWN, DIR_LAYER_UP,
    V2MultiRouteResult, V2RouteCoord, V2RoutePath, known_reader_mask,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V2MaterializeError {
    /// Fewer than 2 coordinates — nothing to place.
    PathTooShort { len: usize },
    /// Consecutive coordinates differ by anything other than exactly one step
    /// on exactly one axis.
    NonUnitStep {
        index: usize,
        from: V2RouteCoord,
        to: V2RouteCoord,
    },
    /// The path visits the same cell twice.
    RevisitedCell { coord: V2RouteCoord },
    /// A coordinate does not exist on the simulation grid.
    OutOfBounds { coord: V2RouteCoord },
    /// An intermediate cell is already occupied by a non-Const tile.
    OccupiedCell {
        coord: V2RouteCoord,
        found: TileType,
    },
    /// The route terminates on an existing tile that does not read from the
    /// entry direction — the delivery would be a silent no-connect.
    GoalIngressMismatch {
        coord: V2RouteCoord,
        goal_type: TileType,
        required_entry: u8,
    },
    /// The first step exits the start tile through its input side; the driver
    /// never marks that neighbor dirty (S133), so the tap would go stale.
    StaleStartTap {
        start_type: TileType,
        first: V2RouteCoord,
    },
    /// Placing this via would give its source cell a second via reader;
    /// `via_fwd` holds one forward slot per source and silently overwrites in
    /// release builds.
    ViaSourceConflict {
        source: V2RouteCoord,
        existing_via: V2RouteCoord,
    },
    /// Two routes in a multi-route set claim the same physical cell.
    OverlappingRoutes {
        coord: V2RouteCoord,
        first_net: String,
        second_net: String,
    },
    /// Post-placement validation: the tile on the path no longer reads from
    /// its predecessor (clobbered or never placed).
    ChainMismatch {
        coord: V2RouteCoord,
        required_entry: u8,
        found: TileType,
    },
}

impl std::fmt::Display for V2MaterializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PathTooShort { len } => write!(f, "path too short: {len} coords"),
            Self::NonUnitStep { index, from, to } => {
                write!(f, "non-unit step at index {index}: {from:?} -> {to:?}")
            }
            Self::RevisitedCell { coord } => write!(f, "path revisits cell {coord:?}"),
            Self::OutOfBounds { coord } => write!(f, "coord {coord:?} outside simulation grid"),
            Self::OccupiedCell { coord, found } => {
                write!(f, "intermediate cell {coord:?} occupied by {found:?}")
            }
            Self::GoalIngressMismatch {
                coord,
                goal_type,
                required_entry,
            } => write!(
                f,
                "goal {coord:?} ({goal_type:?}) does not read from entry mask {required_entry:#04x}"
            ),
            Self::StaleStartTap { start_type, first } => write!(
                f,
                "first step to {first:?} exits the input side of start tile {start_type:?} (stale tap)"
            ),
            Self::ViaSourceConflict {
                source,
                existing_via,
            } => write!(
                f,
                "via source {source:?} already feeds via at {existing_via:?} (one via per source)"
            ),
            Self::OverlappingRoutes {
                coord,
                first_net,
                second_net,
            } => write!(
                f,
                "routes '{first_net}' and '{second_net}' both claim cell {coord:?}"
            ),
            Self::ChainMismatch {
                coord,
                required_entry,
                found,
            } => write!(
                f,
                "chain broken at {coord:?}: {found:?} does not read from entry mask {required_entry:#04x}"
            ),
        }
    }
}

impl std::error::Error for V2MaterializeError {}

/// Result of materializing one route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2MaterializedRoute {
    /// Tiles written into the sim, in path order (excludes the start cell and
    /// a reused goal cell).
    pub placed: Vec<(V2RouteCoord, TileType)>,
    /// Tile indices of `placed` entries.
    pub placed_indices: Vec<usize>,
    /// True when the goal cell already held a compatible consumer tile and was
    /// left untouched (deliberate ingress reuse).
    pub goal_reused: bool,
    /// Number of via tiles placed.
    pub vias_placed: usize,
    /// Non-consecutive path cells stacked on adjacent layers at the same
    /// (x, y). Harmless in this fabric (cross-layer connectivity is explicit
    /// via via tiles), but reported because connectivity-inference tooling
    /// (the b21eaa6 failure class) can misread the shape.
    pub self_stack_count: usize,
}

/// Signal-flow direction of one path step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    East,
    West,
    South,
    North,
    LayerUp,
    LayerDown,
}

impl Step {
    fn between(
        index: usize,
        from: V2RouteCoord,
        to: V2RouteCoord,
    ) -> Result<Self, V2MaterializeError> {
        let dx = to.x as i64 - from.x as i64;
        let dy = to.y as i64 - from.y as i64;
        let dz = to.z as i64 - from.z as i64;
        match (dx, dy, dz) {
            (1, 0, 0) => Ok(Self::East),
            (-1, 0, 0) => Ok(Self::West),
            (0, 1, 0) => Ok(Self::South),
            (0, -1, 0) => Ok(Self::North),
            (0, 0, 1) => Ok(Self::LayerUp),
            (0, 0, -1) => Ok(Self::LayerDown),
            _ => Err(V2MaterializeError::NonUnitStep { index, from, to }),
        }
    }

    /// Tile type placed at the step's destination: it must read from the
    /// direction the signal arrives from. Note the via naming convention:
    /// ViaDown READS from z-1, so a step upward in z places a ViaDown at the
    /// upper cell (see `rebuild_via_connections` in simulation.rs).
    fn dest_tile(self) -> TileType {
        match self {
            Self::East => TileType::WireRight,
            Self::West => TileType::WireLeft,
            Self::South => TileType::WireDown,
            Self::North => TileType::WireUp,
            Self::LayerUp => TileType::ViaDown,
            Self::LayerDown => TileType::ViaUp,
        }
    }

    /// Direction bit, from the destination's perspective, that the signal
    /// enters from (encoding of `directional_mask_for_tile`).
    fn entry_bit(self) -> u8 {
        match self {
            Self::East => DIR_FROM_LEFT,
            Self::West => DIR_FROM_RIGHT,
            Self::South => DIR_FROM_UP,
            Self::North => DIR_FROM_DOWN,
            Self::LayerUp => DIR_LAYER_DOWN,
            Self::LayerDown => DIR_LAYER_UP,
        }
    }
}

fn is_via(tt: TileType) -> bool {
    matches!(
        tt,
        TileType::ViaUp
            | TileType::ViaDown
            | TileType::WeightedViaUp
            | TileType::WeightedViaDown
            | TileType::ThresholdViaUp
            | TileType::ThresholdViaDown
    )
}

fn coord_index(sim: &Simulation, coord: V2RouteCoord) -> Result<usize, V2MaterializeError> {
    sim.tilemap
        .index_3d(coord.x, coord.y, coord.z)
        .ok_or(V2MaterializeError::OutOfBounds { coord })
}

/// Check that placing a via whose SOURCE is `source` does not give that source
/// a second via reader. The via being placed sits at `via_coord` (one of
/// source's z-neighbors); the check inspects the other z-neighbor.
fn check_via_source_exclusive(
    sim: &Simulation,
    source: V2RouteCoord,
    via_coord: V2RouteCoord,
) -> Result<(), V2MaterializeError> {
    let layers = sim.num_layers();
    if source.z + 1 < layers && source.z + 1 != via_coord.z {
        let above = sim.tile_type_3d(source.x, source.y, source.z + 1);
        if matches!(
            above,
            TileType::ViaDown | TileType::WeightedViaDown | TileType::ThresholdViaDown
        ) {
            return Err(V2MaterializeError::ViaSourceConflict {
                source,
                existing_via: V2RouteCoord::new(source.x, source.y, source.z + 1),
            });
        }
    }
    if source.z >= 1 && source.z - 1 != via_coord.z {
        let below = sim.tile_type_3d(source.x, source.y, source.z - 1);
        if matches!(
            below,
            TileType::ViaUp | TileType::WeightedViaUp | TileType::ThresholdViaUp
        ) {
            return Err(V2MaterializeError::ViaSourceConflict {
                source,
                existing_via: V2RouteCoord::new(source.x, source.y, source.z - 1),
            });
        }
    }
    Ok(())
}

struct RoutePlan {
    /// (coord, tile, sim index) for every cell that will be written.
    placements: Vec<(V2RouteCoord, TileType, usize)>,
    goal_reused: bool,
    vias_placed: usize,
    self_stack_count: usize,
}

/// Validate a path against the current sim and compute the placements,
/// without mutating anything.
fn plan_route(sim: &Simulation, path: &V2RoutePath) -> Result<RoutePlan, V2MaterializeError> {
    let coords = &path.coords;
    if coords.len() < 2 {
        return Err(V2MaterializeError::PathTooShort { len: coords.len() });
    }

    let mut seen = std::collections::BTreeSet::new();
    for &c in coords {
        coord_index(sim, c)?;
        if !seen.insert(c) {
            return Err(V2MaterializeError::RevisitedCell { coord: c });
        }
    }

    let mut steps = Vec::with_capacity(coords.len() - 1);
    for i in 1..coords.len() {
        steps.push(Step::between(i, coords[i - 1], coords[i])?);
    }

    // Stale-tap guard: the first step must not exit the start tile through
    // its input side (directional wires and vias never dirty that neighbor).
    let start_tt = sim.tile_type_3d(coords[0].x, coords[0].y, coords[0].z);
    let input_side_step = match start_tt {
        TileType::WireRight => Some(Step::West),
        TileType::WireLeft => Some(Step::East),
        TileType::WireDown => Some(Step::North),
        TileType::WireUp => Some(Step::South),
        TileType::ViaUp | TileType::WeightedViaUp | TileType::ThresholdViaUp => Some(Step::LayerUp),
        TileType::ViaDown | TileType::WeightedViaDown | TileType::ThresholdViaDown => {
            Some(Step::LayerDown)
        }
        _ => None,
    };
    if input_side_step == Some(steps[0]) {
        return Err(V2MaterializeError::StaleStartTap {
            start_type: start_tt,
            first: coords[1],
        });
    }

    let last = coords.len() - 1;
    let goal_tt = sim.tile_type_3d(coords[last].x, coords[last].y, coords[last].z);
    let goal_reused = goal_tt != TileType::Const;
    if goal_reused {
        let entry = steps[last - 1].entry_bit();
        if known_reader_mask(goal_tt) & entry == 0 {
            return Err(V2MaterializeError::GoalIngressMismatch {
                coord: coords[last],
                goal_type: goal_tt,
                required_entry: entry,
            });
        }
    }

    let mut placements = Vec::with_capacity(coords.len() - 1);
    let mut vias_placed = 0;
    for i in 1..coords.len() {
        let coord = coords[i];
        if i == last && goal_reused {
            break;
        }
        let existing = sim.tile_type_3d(coord.x, coord.y, coord.z);
        if existing != TileType::Const {
            return Err(V2MaterializeError::OccupiedCell {
                coord,
                found: existing,
            });
        }
        let tt = steps[i - 1].dest_tile();
        if is_via(tt) {
            check_via_source_exclusive(sim, coords[i - 1], coord)?;
            vias_placed += 1;
        }
        placements.push((coord, tt, coord_index(sim, coord)?));
    }

    // Count non-consecutive same-(x,y) cells on adjacent layers (reported,
    // not rejected: cross-layer connectivity is explicit in this fabric).
    let mut columns: BTreeMap<(usize, usize), Vec<(usize, usize)>> = BTreeMap::new();
    for (i, c) in coords.iter().enumerate() {
        columns.entry((c.x, c.y)).or_default().push((c.z, i));
    }
    let mut self_stack_count = 0;
    for zs in columns.values() {
        for (a, ai) in zs {
            for (b, bi) in zs {
                if a + 1 == *b && ai.abs_diff(*bi) != 1 {
                    self_stack_count += 1;
                }
            }
        }
    }

    Ok(RoutePlan {
        placements,
        goal_reused,
        vias_placed,
        self_stack_count,
    })
}

/// Materialize a routed path into directional wire / via tiles.
///
/// Validate-then-commit: any error leaves the simulation untouched. The start
/// cell (existing driver) is never overwritten; the goal cell is placed if it
/// is free Const, or reused untouched if it already holds a tile that reads
/// from the entry direction. If any via was placed, `rebuild_via_connections`
/// is invoked so dirty propagation reaches the new via.
pub fn materialize_route(
    sim: &mut Simulation,
    path: &V2RoutePath,
) -> Result<V2MaterializedRoute, V2MaterializeError> {
    let plan = plan_route(sim, path)?;
    commit_plan(sim, &plan);
    Ok(materialized_from_plan(plan))
}

fn commit_plan(sim: &mut Simulation, plan: &RoutePlan) {
    for &(coord, tt, _) in &plan.placements {
        sim.set_tile_3d(coord.x, coord.y, coord.z, tt);
    }
    if plan.vias_placed > 0 {
        sim.rebuild_via_connections();
    }
}

fn materialized_from_plan(plan: RoutePlan) -> V2MaterializedRoute {
    V2MaterializedRoute {
        placed: plan
            .placements
            .iter()
            .map(|&(coord, tt, _)| (coord, tt))
            .collect(),
        placed_indices: plan.placements.iter().map(|&(_, _, idx)| idx).collect(),
        goal_reused: plan.goal_reused,
        vias_placed: plan.vias_placed,
        self_stack_count: plan.self_stack_count,
    }
}

/// Materialize every route in a `route_multinet` result (deterministic
/// BTreeMap order). All routes are planned against the unmutated sim and
/// checked pairwise for cell overlap and shared via sources before anything
/// is committed, so an error leaves the simulation untouched.
///
/// Note: physical wires carry exactly one signal, so overlapping paths are
/// rejected even where the routing DB granted capacity > 1 (channel capacity
/// is a planning construct; materialization requires disjoint paths).
pub fn materialize_routes(
    sim: &mut Simulation,
    result: &V2MultiRouteResult,
) -> Result<BTreeMap<String, V2MaterializedRoute>, (String, V2MaterializeError)> {
    let mut planned: Vec<(String, RoutePlan)> = Vec::new();
    let mut claimed: BTreeMap<V2RouteCoord, String> = BTreeMap::new();
    let mut via_sources: BTreeMap<V2RouteCoord, String> = BTreeMap::new();

    for (net_id, path) in &result.routes {
        let plan = plan_route(sim, path).map_err(|e| (net_id.clone(), e))?;
        for &(coord, tt, _) in &plan.placements {
            if let Some(first) = claimed.get(&coord) {
                return Err((
                    net_id.clone(),
                    V2MaterializeError::OverlappingRoutes {
                        coord,
                        first_net: first.clone(),
                        second_net: net_id.clone(),
                    },
                ));
            }
            claimed.insert(coord, net_id.clone());
            if is_via(tt) {
                // The via's source is the path cell preceding this placement.
                let pos = path.coords.iter().position(|&c| c == coord).unwrap();
                let source = path.coords[pos - 1];
                if via_sources.contains_key(&source) {
                    return Err((
                        net_id.clone(),
                        V2MaterializeError::ViaSourceConflict {
                            source,
                            existing_via: coord,
                        },
                    ));
                }
                via_sources.insert(source, net_id.clone());
            }
        }
        planned.push((net_id.clone(), plan));
    }

    let mut out = BTreeMap::new();
    let mut any_via = false;
    for (net_id, plan) in planned {
        for &(coord, tt, _) in &plan.placements {
            sim.set_tile_3d(coord.x, coord.y, coord.z, tt);
            any_via |= is_via(tt);
        }
        out.insert(net_id, materialized_from_plan(plan));
    }
    if any_via {
        sim.rebuild_via_connections();
    }
    Ok(out)
}

/// Standalone invariant pass over an already-materialized (or hand-placed)
/// route: every cell on the path must read from its predecessor, layer
/// transitions must be carried by a correctly-oriented via, and no via source
/// on the path may feed a second via. Usable as a regression check after any
/// later fabric mutation.
pub fn validate_materialized_route(
    sim: &Simulation,
    path: &V2RoutePath,
) -> Result<(), V2MaterializeError> {
    let coords = &path.coords;
    if coords.len() < 2 {
        return Err(V2MaterializeError::PathTooShort { len: coords.len() });
    }
    for i in 1..coords.len() {
        let step = Step::between(i, coords[i - 1], coords[i])?;
        let coord = coords[i];
        coord_index(sim, coord)?;
        let tt = sim.tile_type_3d(coord.x, coord.y, coord.z);
        let entry = step.entry_bit();
        if known_reader_mask(tt) & entry == 0 {
            return Err(V2MaterializeError::ChainMismatch {
                coord,
                required_entry: entry,
                found: tt,
            });
        }
        if is_via(tt) {
            check_via_source_exclusive(sim, coords[i - 1], coord)?;
        }
    }
    Ok(())
}

/// Mark every cell of the path dirty and run combinational propagation,
/// repeated `passes` times. Use path length + slack for `passes` so the value
/// reaches the goal regardless of the propagation engine's per-call depth.
pub fn settle_route(sim: &mut Simulation, path: &V2RoutePath, passes: usize) {
    let indices: Vec<usize> = path
        .coords
        .iter()
        .filter_map(|c| sim.tilemap.index_3d(c.x, c.y, c.z))
        .collect();
    for _ in 0..passes.max(1) {
        for &idx in &indices {
            sim.dirty.mark_dirty(idx);
        }
        sim.propagate_combinational();
    }
}

#[cfg(test)]
mod tests {
    use super::super::v2_routing::{
        V2RouteBounds, V2RouteNet, V2RouteNetClass, V2RoutingConfig, V2RoutingDb,
    };
    use super::*;

    fn const_field(w: usize, h: usize, layers: usize) -> Simulation {
        let mut sim = Simulation::with_size_layered(w, h, layers);
        for z in 0..layers {
            for y in 0..h {
                for x in 0..w {
                    sim.set_tile_3d(x, y, z, TileType::Const);
                }
            }
        }
        sim
    }

    fn path_of(coords: &[(usize, usize, usize)]) -> V2RoutePath {
        V2RoutePath {
            coords: coords
                .iter()
                .map(|&(x, y, z)| V2RouteCoord::new(x, y, z))
                .collect(),
            cost: 0,
            bends: 0,
            vias: 0,
        }
    }

    fn settle_all(sim: &mut Simulation, passes: usize) {
        for _ in 0..passes {
            sim.dirty.mark_all_dirty(sim.tile_count());
            sim.propagate_combinational();
        }
    }

    #[test]
    fn straight_east_run_carries_signal() {
        let mut sim = const_field(12, 8, 4);
        sim.set_logic_value_3d(1, 3, 2, 0xABCD);
        let path = path_of(&[(1, 3, 2), (2, 3, 2), (3, 3, 2), (4, 3, 2), (5, 3, 2)]);

        let mat = materialize_route(&mut sim, &path).expect("materialize");
        assert_eq!(mat.placed.len(), 4);
        assert!(mat.placed.iter().all(|&(_, tt)| tt == TileType::WireRight));
        assert!(!mat.goal_reused);
        assert_eq!(mat.vias_placed, 0);
        validate_materialized_route(&sim, &path).expect("validate");

        settle_all(&mut sim, 6);
        assert_eq!(sim.get_logic_at_3d(5, 3, 2), 0xABCD);
    }

    #[test]
    fn bend_places_correct_corner_orientation() {
        let mut sim = const_field(12, 12, 4);
        sim.set_logic_value_3d(2, 2, 2, 0x77);
        // East, East, then South, South, South.
        let path = path_of(&[
            (2, 2, 2),
            (3, 2, 2),
            (4, 2, 2),
            (4, 3, 2),
            (4, 4, 2),
            (4, 5, 2),
        ]);
        materialize_route(&mut sim, &path).expect("materialize");

        assert_eq!(sim.tile_type_3d(3, 2, 2), TileType::WireRight);
        assert_eq!(sim.tile_type_3d(4, 2, 2), TileType::WireRight);
        assert_eq!(sim.tile_type_3d(4, 3, 2), TileType::WireDown);
        assert_eq!(sim.tile_type_3d(4, 5, 2), TileType::WireDown);

        settle_all(&mut sim, 8);
        assert_eq!(sim.get_logic_at_3d(4, 5, 2), 0x77);
    }

    #[test]
    fn layer_hop_round_trip_uses_correct_via_orientation() {
        let mut sim = const_field(10, 6, 4);
        sim.set_logic_value_3d(1, 1, 1, 0x5A5A);
        // East, LayerUp, East, LayerDown, East.
        let path = path_of(&[
            (1, 1, 1),
            (2, 1, 1),
            (2, 1, 2),
            (3, 1, 2),
            (3, 1, 1),
            (4, 1, 1),
        ]);
        let mat = materialize_route(&mut sim, &path).expect("materialize");
        assert_eq!(mat.vias_placed, 2);

        // Step up in z places a ViaDown (reads z-1); step down places a ViaUp.
        assert_eq!(sim.tile_type_3d(2, 1, 2), TileType::ViaDown);
        assert_eq!(sim.tile_type_3d(3, 1, 1), TileType::ViaUp);
        validate_materialized_route(&sim, &path).expect("validate");

        settle_route(&mut sim, &path, path.coords.len() + 2);
        assert_eq!(sim.get_logic_at_3d(4, 1, 1), 0x5A5A);
    }

    #[test]
    fn occupied_intermediate_rejected_without_mutation() {
        let mut sim = const_field(12, 8, 4);
        sim.set_tile_3d(4, 3, 2, TileType::Or);
        let path = path_of(&[(1, 3, 2), (2, 3, 2), (3, 3, 2), (4, 3, 2), (5, 3, 2)]);

        let err = materialize_route(&mut sim, &path).unwrap_err();
        assert_eq!(
            err,
            V2MaterializeError::OccupiedCell {
                coord: V2RouteCoord::new(4, 3, 2),
                found: TileType::Or,
            }
        );
        // Validate-then-commit: nothing else was placed.
        assert_eq!(sim.tile_type_3d(2, 3, 2), TileType::Const);
        assert_eq!(sim.tile_type_3d(3, 3, 2), TileType::Const);
    }

    #[test]
    fn goal_ingress_reuse_checks_entry_direction() {
        // Goal already holds a WireRight (reads LEFT). Arriving eastward
        // enters from LEFT -> compatible, reused untouched.
        let mut sim = const_field(12, 8, 4);
        sim.set_tile_3d(5, 3, 2, TileType::WireRight);
        let east = path_of(&[(2, 3, 2), (3, 3, 2), (4, 3, 2), (5, 3, 2)]);
        let mat = materialize_route(&mut sim, &east).expect("compatible ingress");
        assert!(mat.goal_reused);
        assert_eq!(mat.placed.len(), 2);
        assert_eq!(sim.tile_type_3d(5, 3, 2), TileType::WireRight);

        // Arriving from the north enters from UP -> WireRight does not read
        // UP -> rejected (silent no-connect otherwise).
        let mut sim2 = const_field(12, 8, 4);
        sim2.set_tile_3d(5, 3, 2, TileType::WireRight);
        let south = path_of(&[(5, 0, 2), (5, 1, 2), (5, 2, 2), (5, 3, 2)]);
        let err = materialize_route(&mut sim2, &south).unwrap_err();
        assert!(matches!(
            err,
            V2MaterializeError::GoalIngressMismatch { .. }
        ));
    }

    #[test]
    fn stale_start_tap_rejected() {
        // WireRight reads LEFT; exiting west sits on its input side, which it
        // never marks dirty (S133).
        let mut sim = const_field(12, 8, 4);
        sim.set_tile_3d(5, 5, 2, TileType::WireRight);
        let path = path_of(&[(5, 5, 2), (4, 5, 2), (3, 5, 2)]);
        let err = materialize_route(&mut sim, &path).unwrap_err();
        assert!(matches!(err, V2MaterializeError::StaleStartTap { .. }));
    }

    #[test]
    fn second_via_on_same_source_rejected() {
        // Existing ViaUp at z=1 reads (3,3,2). Our path would place a ViaDown
        // at z=3 reading the same source -> via_fwd conflict (silent
        // overwrite in release builds without this check).
        let mut sim = const_field(10, 10, 4);
        sim.set_tile_3d(3, 3, 1, TileType::ViaUp);
        let path = path_of(&[(2, 3, 2), (3, 3, 2), (3, 3, 3), (4, 3, 3)]);
        let err = materialize_route(&mut sim, &path).unwrap_err();
        assert_eq!(
            err,
            V2MaterializeError::ViaSourceConflict {
                source: V2RouteCoord::new(3, 3, 2),
                existing_via: V2RouteCoord::new(3, 3, 1),
            }
        );
    }

    #[test]
    fn hardened_db_blocks_reader_adjacent_and_protected_cells() {
        let mut sim = const_field(12, 12, 4);
        sim.set_tile_3d(5, 3, 2, TileType::Or);
        sim.set_tile_3d(2, 2, 3, TileType::ViaDown); // reads (2,2,2) from below

        let bounds = V2RouteBounds::new(0, 12, 0, 12, 0, 3);
        let plain = V2RoutingDb::from_simulation_const_blocked(&sim, bounds);
        let protected = [V2RouteCoord::new(8, 8, 2)];
        let hard = V2RoutingDb::from_simulation_interference_checked(&sim, bounds, &protected);

        // The Or gate is conservatively treated as reading all four in-plane
        // neighbors; the plain DB leaves those cells routable.
        for (x, y) in [(4, 3), (6, 3), (5, 2), (5, 4)] {
            let c = V2RouteCoord::new(x, y, 2);
            assert!(!plain.cell(c).hard_block, "plain DB should not block {c:?}");
            assert!(hard.cell(c).hard_block, "hardened DB must block {c:?}");
        }
        // Cross-layer: the ViaDown at z=3 reads (2,2,2).
        let under_via = V2RouteCoord::new(2, 2, 2);
        assert!(!plain.cell(under_via).hard_block);
        assert!(hard.cell(under_via).hard_block);
        // Protected injection-port Const.
        assert!(!plain.cell(protected[0]).hard_block);
        assert!(hard.cell(protected[0]).hard_block);
        // Far free cell stays routable.
        let far = V2RouteCoord::new(10, 10, 2);
        assert!(!plain.cell(far).hard_block);
        assert!(!hard.cell(far).hard_block);
    }

    #[test]
    fn hardened_route_does_not_disturb_adjacent_reader() {
        // An Or gate sits one row above the naive straight path. The plain DB
        // would happily route through (5,3): the Or reads DOWN and would pick
        // up the routed signal. The hardened DB forces a detour; driving
        // all-ones through the materialized route leaves the Or at 0.
        let mut sim = const_field(14, 10, 4);
        sim.set_tile_3d(5, 2, 2, TileType::Or);

        let bounds = V2RouteBounds::new(0, 14, 0, 10, 2, 2);
        let db = V2RoutingDb::from_simulation_interference_checked(&sim, bounds, &[]);
        let net = V2RouteNet::new(
            "probe",
            V2RouteNetClass::Data,
            V2RouteCoord::new(1, 3, 2),
            V2RouteCoord::new(11, 3, 2),
        );
        let result = db.route_multinet(std::slice::from_ref(&net), V2RoutingConfig::default());
        assert!(result.success, "hardened routing should still succeed");
        let path = &result.routes["probe"];
        assert!(
            !path.coords.contains(&V2RouteCoord::new(5, 3, 2)),
            "path must avoid the cell the Or gate reads"
        );

        sim.set_logic_value_3d(1, 3, 2, u64::MAX);
        materialize_route(&mut sim, path).expect("materialize");
        settle_all(&mut sim, 20);

        assert_eq!(sim.get_logic_at_3d(11, 3, 2), u64::MAX, "goal got signal");
        assert_eq!(sim.get_logic_at_3d(5, 2, 2), 0, "Or gate undisturbed");
    }

    #[test]
    fn overlapping_multi_routes_rejected() {
        let mut sim = const_field(12, 12, 4);
        let mut routes = BTreeMap::new();
        routes.insert(
            "a".to_string(),
            path_of(&[(1, 5, 2), (2, 5, 2), (3, 5, 2), (4, 5, 2)]),
        );
        routes.insert(
            "b".to_string(),
            path_of(&[(3, 3, 2), (3, 4, 2), (3, 5, 2), (3, 6, 2)]),
        );
        let result = V2MultiRouteResult {
            success: true,
            routes,
            overflow_cells: Vec::new(),
            failed_net_ids: Vec::new(),
            negotiation_iters: 1,
            layer_violation_count: 0,
            layer_utilization: [0; 4],
        };
        let (net, err) = materialize_routes(&mut sim, &result).unwrap_err();
        assert_eq!(net, "b");
        assert!(matches!(err, V2MaterializeError::OverlappingRoutes { .. }));
        // Nothing committed.
        assert_eq!(sim.tile_type_3d(2, 5, 2), TileType::Const);
        assert_eq!(sim.tile_type_3d(3, 4, 2), TileType::Const);
    }

    #[test]
    fn validator_detects_post_hoc_clobber() {
        let mut sim = const_field(12, 8, 4);
        let path = path_of(&[(1, 3, 2), (2, 3, 2), (3, 3, 2), (4, 3, 2)]);
        materialize_route(&mut sim, &path).expect("materialize");
        validate_materialized_route(&sim, &path).expect("fresh route validates");

        sim.set_tile_3d(3, 3, 2, TileType::Const);
        let err = validate_materialized_route(&sim, &path).unwrap_err();
        assert!(matches!(err, V2MaterializeError::ChainMismatch { .. }));
    }

    // === Sprint 398 integration: real V2 CPU fabric ===

    fn build_real_cpu(program: &[u32]) -> (Simulation, super::super::TileCpuV2) {
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = super::super::V2Builder::new()
            .with_origin(0, 0)
            .with_program(program)
            .with_rom_size(64)
            .with_ram_size(64)
            .build(&mut sim);
        (sim, cpu)
    }

    /// Deterministically pick a start/goal pair of unblocked cells on the
    /// same row with a minimum span, scanning top layers first.
    fn pick_free_net(db: &V2RoutingDb, bounds: V2RouteBounds) -> (V2RouteCoord, V2RouteCoord) {
        for z in (bounds.min_layer..=bounds.max_layer_inclusive).rev() {
            for y in bounds.min_y..bounds.max_y_exclusive {
                for x in bounds.min_x..bounds.max_x_exclusive {
                    let start = V2RouteCoord::new(x, y, z);
                    if db.cell(start).hard_block {
                        continue;
                    }
                    for gx in (x + 16..bounds.max_x_exclusive).rev() {
                        let goal = V2RouteCoord::new(gx, y, z);
                        if !db.cell(goal).hard_block {
                            return (start, goal);
                        }
                    }
                }
            }
        }
        panic!("no free start/goal pair in bounds {bounds:?}");
    }

    #[test]
    fn s398_hardened_db_on_real_cpu_blocks_and_still_routes() {
        use crate::tile_cpu::v2_components::{v2_halt, v2_nop};
        let (sim, _cpu) = build_real_cpu(&[v2_nop(), v2_halt()]);

        let bounds = V2RouteBounds::new(0, 96, 0, 64, 1, 3);
        let plain = V2RoutingDb::from_simulation_const_blocked(&sim, bounds);
        let hard = V2RoutingDb::from_simulation_interference_checked(&sim, bounds, &[]);

        // The hardening must actually bite on the real fabric.
        let mut newly_blocked = 0usize;
        for z in bounds.min_layer..=bounds.max_layer_inclusive {
            for y in bounds.min_y..bounds.max_y_exclusive {
                for x in bounds.min_x..bounds.max_x_exclusive {
                    let c = V2RouteCoord::new(x, y, z);
                    if hard.cell(c).hard_block && !plain.cell(c).hard_block {
                        newly_blocked += 1;
                    }
                }
            }
        }
        assert!(
            newly_blocked > 0,
            "reader-adjacency hardening blocked no cells on the real CPU"
        );

        // And free space must remain routable end-to-end.
        let (start, goal) = pick_free_net(&hard, bounds);
        let net = V2RouteNet::new("s398_probe", V2RouteNetClass::Data, start, goal);
        let result = hard.route_multinet(std::slice::from_ref(&net), V2RoutingConfig::default());
        assert!(
            result.success,
            "hardened DB failed to route free-space net {start:?}->{goal:?}: {}",
            result.failure_signature()
        );
    }

    #[test]
    fn s398_materialized_route_does_not_perturb_cpu_execution() {
        use crate::tile_cpu::v2_components::{v2_add, v2_halt, v2_ldi};

        let program = [v2_ldi(0, 7), v2_ldi(1, 35), v2_add(0, 1), v2_halt()];
        let run = |mutate: bool| -> ([u64; 16], u32, u64) {
            let (mut sim, cpu) = build_real_cpu(&program);
            let mut driven = 0u64;
            if mutate {
                let bounds = V2RouteBounds::new(0, 96, 0, 64, 1, 3);
                let db = V2RoutingDb::from_simulation_interference_checked(&sim, bounds, &[]);
                let (start, goal) = pick_free_net(&db, bounds);
                let net = V2RouteNet::new("s398_live", V2RouteNetClass::Data, start, goal);
                let result =
                    db.route_multinet(std::slice::from_ref(&net), V2RoutingConfig::default());
                assert!(result.success, "{}", result.failure_signature());
                let path = &result.routes["s398_live"];
                materialize_route(&mut sim, path).expect("materialize on live CPU");
                validate_materialized_route(&sim, path).expect("validate on live CPU");
                // Drive an aggressive all-ones pattern through the new route
                // before the program runs.
                sim.set_logic_value_3d(start.x, start.y, start.z, u64::MAX);
                settle_route(&mut sim, path, path.coords.len() + 2);
                driven = sim.get_logic_at_3d(goal.x, goal.y, goal.z);
            }
            let mut guard = 0;
            while !cpu.is_halted() && guard < 200 {
                cpu.tick(&mut sim);
                guard += 1;
            }
            assert!(cpu.is_halted(), "program did not halt");
            (
                std::array::from_fn(|i| cpu.read_reg(&sim, i)),
                cpu.read_pc(&sim),
                driven,
            )
        };

        let (regs_control, pc_control, _) = run(false);
        let (regs_routed, pc_routed, driven) = run(true);

        assert_eq!(driven, u64::MAX, "routed net must deliver the driven value");
        assert_eq!(regs_control, regs_routed, "register state perturbed");
        assert_eq!(pc_control, pc_routed, "pc perturbed");
        assert_eq!(regs_control[0], 42, "sanity: 7 + 35");
    }
}
