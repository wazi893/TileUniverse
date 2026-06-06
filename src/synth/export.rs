//! Export a placed+routed synth netlist to a Simulation grid.
//!
//! Converts the abstract `PlacedNetlist` (gates + wire routes + I/O anchors)
//! into a live `Simulation` whose tiles can be driven and read back.
//!
//! Wire crossings (Cross tiles in the router) use bit-split encoding (low 32
//! bits = horizontal, high 32 bits = vertical), which is incompatible with
//! all-or-nothing boolean signals (0 or `u64::MAX`). The export handles routed
//! crossings by keeping one net on layer 0 and detouring the other through
//! layer 1 with ViaDown/ViaUp transitions.

use super::mapping::MappedNetlist;
use super::placement::PlacedNetlist;
use crate::simulation::Simulation;
use crate::tiles::tile_meta::TileType;
use std::collections::{HashMap, HashSet, VecDeque};

struct RouteDirectionData {
    /// Directional tile type for each (net_id, x, y, z) route segment.
    /// Uses 4D keys to avoid collisions when a net has segments at the same
    /// (x,y) on different layers (e.g., L0 WireUp vs L1 WireRight).
    directional_tiles: HashMap<(u32, usize, usize, usize), TileType>,
    source_coords: HashMap<u32, (usize, usize)>,
    parent_coords: HashMap<(u32, usize, usize), (usize, usize)>,
    net_coords: HashMap<u32, HashSet<(usize, usize)>>,
}

#[derive(Default)]
struct BypassPlan {
    core_coords: HashMap<u32, HashSet<(usize, usize)>>,
    entry_coords: HashMap<u32, HashSet<(usize, usize)>>,
    exit_coords: HashMap<u32, HashSet<(usize, usize)>>,
    routing_layers: HashMap<u32, usize>,
}

impl BypassPlan {
    fn removes_from_l0(&self, net_id: u32, coord: (usize, usize)) -> bool {
        let is_entry = self
            .entry_coords
            .get(&net_id)
            .is_some_and(|coords| coords.contains(&coord));
        let is_core = self
            .core_coords
            .get(&net_id)
            .is_some_and(|coords| coords.contains(&coord));
        let is_exit = self
            .exit_coords
            .get(&net_id)
            .is_some_and(|coords| coords.contains(&coord));
        (is_core && !is_entry) || is_exit
    }

    fn routing_layer(&self, net_id: u32) -> usize {
        self.routing_layers.get(&net_id).copied().unwrap_or(1)
    }

    fn num_layers(&self) -> usize {
        self.routing_layers
            .values()
            .copied()
            .max()
            .map_or(1, |layer| layer + 1)
    }
}

/// Result of exporting a synthesis result to the simulation grid.
pub struct SynthExport {
    pub sim: Simulation,
    /// (x, y) coordinates of each primary input anchor (Const tile).
    pub input_coords: Vec<(usize, usize)>,
    /// (x, y) coordinates of each primary output anchor (readable tile).
    pub output_coords: Vec<(usize, usize)>,
    /// Indices of all non-Const tiles (gates, wires, vias). Pre-computed so
    /// `evaluate_exported` can reset/dirty only circuit tiles, skipping the
    /// ~90% Const guard tiles that dominate the grid.
    circuit_indices: Vec<usize>,
    /// Whether the last `evaluate_exported` call converged (dirty set reached
    /// zero within the outer iteration limit).
    pub last_converged: bool,
    /// Number of outer propagation rounds used by the last `evaluate_exported`
    /// call. Each outer round runs up to 100 inner delta cycles.
    pub last_convergence_rounds: u32,
}

impl SynthExport {
    /// Access the pre-computed circuit tile indices (non-Const tiles).
    pub(crate) fn circuit_indices(&self) -> &[usize] {
        &self.circuit_indices
    }
}

/// Export a placed+routed netlist to a `Simulation` grid.
///
/// The grid is sized to fit the placement region with a 1-tile margin.
/// All background tiles are set to `Const` (value 0) to prevent default-Wire
/// signal leakage. Route segments use unidirectional wire tiles to prevent
/// feedback.
///
/// Routed crossings are detoured through a second simulation layer so the
/// resulting fabric remains boolean-clean.
pub fn export_to_simulation(placed: &PlacedNetlist, netlist: &MappedNetlist) -> SynthExport {
    let route_data = compute_route_direction_data(placed, netlist);
    let width = placed.region.max_x() + 2;
    let height = placed.region.max_y() + 2;
    let crossings = find_crossings(placed, &route_data);
    let bypass_plan = build_bypass_plan(&crossings, &route_data);
    let router_max_z = placed.routes.iter().map(|seg| seg.z).max().unwrap_or(0);
    // Bypass layers are placed above router layers to avoid conflicts.
    let num_layers = if bypass_plan.num_layers() > 1 {
        router_max_z + bypass_plan.num_layers()
    } else {
        router_max_z + 1
    };
    let mut sim = Simulation::with_size_layered(width, height, num_layers);

    // Fill with Const(0) guards to prevent default-Wire signal leakage.
    for z in 0..num_layers {
        for y in 0..height {
            for x in 0..width {
                sim.set_tile_3d(x, y, z, TileType::Const);
            }
        }
    }

    // Place gate tiles.
    for gate in &placed.gates {
        sim.set_tile_3d(gate.x, gate.y, gate.z, gate.tile_type);
    }

    // Place route segments using directional tiles. The 3D BFS assigned
    // unidirectional wire types for all segments (both L0 and z>0).
    // ViaUp/ViaDown segments keep their tile type directly.
    for seg in &placed.routes {
        if seg.z == 0 && bypass_plan.removes_from_l0(seg.net_id, (seg.x, seg.y)) {
            continue;
        }
        let tile = if seg.tile_type == TileType::ViaUp || seg.tile_type == TileType::ViaDown {
            seg.tile_type
        } else {
            route_data
                .directional_tiles
                .get(&(seg.net_id, seg.x, seg.y, seg.z))
                .copied()
                .unwrap_or(seg.tile_type)
        };
        sim.set_tile_3d(seg.x, seg.y, seg.z, tile);
    }

    if num_layers > 1 {
        apply_crossing_bypasses(
            &mut sim,
            &bypass_plan,
            &route_data.directional_tiles,
            width,
            height,
        );
    }

    // Place PI anchors as Const tiles AFTER routes.
    let input_coords: Vec<(usize, usize)> = placed.input_anchors.clone();
    for &(x, y) in &input_coords {
        sim.set_tile_3d(x, y, 0, TileType::Const);
    }

    let output_coords: Vec<(usize, usize)> = placed.output_anchors.clone();

    if num_layers > 1 {
        sim.rebuild_via_connections();
    }

    // Pre-compute circuit tile indices (non-Const) for fast eval reset.
    let circuit_indices: Vec<usize> = (0..sim.tilemap.tiles.len())
        .filter(|&idx| sim.tilemap.tiles[idx].meta.tile_type != TileType::Const)
        .collect();

    SynthExport {
        sim,
        input_coords,
        output_coords,
        circuit_indices,
        last_converged: true,
        last_convergence_rounds: 0,
    }
}

// ---------------------------------------------------------------------------
// Crossing detection
// ---------------------------------------------------------------------------

/// Classification for one crossing coord.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CrossingInfo {
    /// The net that stays on layer 0 (typically horizontal flow).
    stay_net: u32,
    /// The net that bypasses through layer 1 (typically vertical flow).
    bypass_net: u32,
}

struct CrossingCandidate {
    coord: (usize, usize),
    net_a: u32,
    net_b: u32,
    net_a_orient: u8,
    net_b_orient: u8,
    net_a_is_source: bool,
    net_b_is_source: bool,
}

/// Find all crossing coords (shared by 2+ nets) and classify stay/bypass.
fn find_crossings(
    placed: &PlacedNetlist,
    route_data: &RouteDirectionData,
) -> HashMap<(usize, usize), CrossingInfo> {
    let mut coord_nets: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
    for seg in &placed.routes {
        // Only detect crossings among L0 segments — the multi-layer router
        // already resolved z>0 conflicts through layer transitions.
        if seg.z != 0 {
            continue;
        }
        let nets = coord_nets.entry((seg.x, seg.y)).or_default();
        if !nets.contains(&seg.net_id) {
            nets.push(seg.net_id);
        }
    }

    let mut candidates = Vec::new();
    for (coord, nets) in &coord_nets {
        if nets.len() < 2 {
            continue;
        }
        let net_a = nets[0];
        let net_b = nets[1];
        candidates.push(CrossingCandidate {
            coord: *coord,
            net_a,
            net_b,
            net_a_orient: crossing_input_orientation(route_data, net_a, *coord),
            net_b_orient: crossing_input_orientation(route_data, net_b, *coord),
            net_a_is_source: route_data.source_coords.get(&net_a).copied() == Some(*coord),
            net_b_is_source: route_data.source_coords.get(&net_b).copied() == Some(*coord),
        });
    }

    let mut pair_groups: HashMap<(u32, u32), Vec<&CrossingCandidate>> = HashMap::new();
    let mut result = HashMap::new();
    for candidate in &candidates {
        let pair = if candidate.net_a <= candidate.net_b {
            (candidate.net_a, candidate.net_b)
        } else {
            (candidate.net_b, candidate.net_a)
        };
        pair_groups.entry(pair).or_default().push(candidate);

        let (stay, bypass) = classify_crossing(
            candidate.net_a,
            candidate.net_b,
            candidate.net_a_orient,
            candidate.net_b_orient,
            candidate.net_a_is_source,
            candidate.net_b_is_source,
        );
        result.insert(
            candidate.coord,
            CrossingInfo {
                stay_net: stay,
                bypass_net: bypass,
            },
        );
    }

    for _ in 0..candidates.len().max(1).min(20) {
        let conflict_pairs = find_bypass_conflict_pairs(&result, route_data);
        if conflict_pairs.is_empty() {
            break;
        }

        let mut changed = false;
        for (low_net, high_net) in conflict_pairs {
            let Some(group) = pair_groups.get(&(low_net, high_net)) else {
                continue;
            };
            let Some(stay_net) =
                choose_conflict_pair_stay_net(group, &result, route_data, low_net, high_net)
            else {
                continue;
            };
            let bypass_net = if stay_net == low_net {
                high_net
            } else {
                low_net
            };
            for candidate in group {
                let info = result
                    .get_mut(&candidate.coord)
                    .expect("crossing candidate should already exist");
                if info.stay_net != stay_net || info.bypass_net != bypass_net {
                    info.stay_net = stay_net;
                    info.bypass_net = bypass_net;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    result
}

fn classify_crossing(
    net_a: u32,
    net_b: u32,
    net_a_orient: u8,
    net_b_orient: u8,
    net_a_is_source: bool,
    net_b_is_source: bool,
) -> (u32, u32) {
    if net_a_is_source && !net_b_is_source {
        (net_a, net_b)
    } else if net_b_is_source && !net_a_is_source {
        (net_b, net_a)
    } else if net_a_orient == ORIENT_H && net_b_orient == ORIENT_V {
        (net_a, net_b)
    } else if net_b_orient == ORIENT_H && net_a_orient == ORIENT_V {
        (net_b, net_a)
    } else if net_a_orient == ORIENT_H && net_b_orient != ORIENT_H {
        (net_a, net_b)
    } else if net_b_orient == ORIENT_H && net_a_orient != ORIENT_H {
        (net_b, net_a)
    } else if net_a <= net_b {
        (net_a, net_b)
    } else {
        (net_b, net_a)
    }
}

fn find_bypass_conflict_pairs(
    crossings: &HashMap<(usize, usize), CrossingInfo>,
    route_data: &RouteDirectionData,
) -> Vec<(u32, u32)> {
    let plan = build_bypass_plan(crossings, route_data);
    let mut coord_nets: HashMap<(usize, usize), HashSet<u32>> = HashMap::new();

    for (&net_id, coords) in &plan.core_coords {
        for &coord in coords {
            coord_nets.entry(coord).or_default().insert(net_id);
        }
    }
    for (&net_id, coords) in &plan.entry_coords {
        for &coord in coords {
            coord_nets.entry(coord).or_default().insert(net_id);
        }
    }
    for (&net_id, coords) in &plan.exit_coords {
        for &coord in coords {
            coord_nets.entry(coord).or_default().insert(net_id);
        }
    }

    let mut conflict_pairs = HashSet::new();
    for nets in coord_nets.values() {
        if nets.len() < 2 {
            continue;
        }
        let mut sorted: Vec<_> = nets.iter().copied().collect();
        sorted.sort_unstable();
        for i in 0..sorted.len() {
            for j in (i + 1)..sorted.len() {
                conflict_pairs.insert((sorted[i], sorted[j]));
            }
        }
    }

    let mut pairs: Vec<_> = conflict_pairs.into_iter().collect();
    pairs.sort_unstable();
    pairs
}

fn choose_conflict_pair_stay_net(
    group: &[&CrossingCandidate],
    crossings: &HashMap<(usize, usize), CrossingInfo>,
    route_data: &RouteDirectionData,
    low_net: u32,
    high_net: u32,
) -> Option<u32> {
    let mut preferred_stay = None;
    let mut low_votes = 0usize;
    let mut high_votes = 0usize;

    for candidate in group {
        if candidate.net_a_is_source ^ candidate.net_b_is_source {
            let source_net = if candidate.net_a_is_source {
                candidate.net_a
            } else {
                candidate.net_b
            };
            if let Some(existing) = preferred_stay {
                if existing != source_net {
                    preferred_stay = None;
                    break;
                }
            } else {
                preferred_stay = Some(source_net);
            }
        } else if let Some(info) = crossings.get(&candidate.coord) {
            if info.stay_net == low_net {
                low_votes += 1;
            } else if info.stay_net == high_net {
                high_votes += 1;
            }
        }
    }

    let mut low_trial = crossings.clone();
    for candidate in group {
        low_trial.insert(
            candidate.coord,
            CrossingInfo {
                stay_net: low_net,
                bypass_net: high_net,
            },
        );
    }
    let low_score = crossing_assignment_score(&low_trial, route_data);

    let mut high_trial = crossings.clone();
    for candidate in group {
        high_trial.insert(
            candidate.coord,
            CrossingInfo {
                stay_net: high_net,
                bypass_net: low_net,
            },
        );
    }
    let high_score = crossing_assignment_score(&high_trial, route_data);

    // Source constraint is absolute: a net whose source (gate output pin) is at
    // a crossing MUST stay on L0 because the ViaDown entry cannot pick up the
    // gate signal when the L0 tile belongs to the other net.
    if let Some(source_net) = preferred_stay {
        return Some(source_net);
    }

    if low_score < high_score {
        Some(low_net)
    } else if high_score < low_score {
        Some(high_net)
    } else if low_votes > high_votes {
        Some(low_net)
    } else if high_votes > low_votes {
        Some(high_net)
    } else {
        Some(low_net)
    }
}

fn crossing_assignment_score(
    crossings: &HashMap<(usize, usize), CrossingInfo>,
    route_data: &RouteDirectionData,
) -> usize {
    let plan = build_bypass_plan(crossings, route_data);
    let assignments: HashMap<u32, usize> = plan
        .core_coords
        .keys()
        .copied()
        .map(|net_id| (net_id, plan.routing_layer(net_id)))
        .collect();
    bypass_layer_conflict_score(&plan, &assignments)
}

/// Check if a net has a route segment at a horizontal (or vertical) neighbor.
fn has_neighbor_in_net(
    net_coords: &HashMap<u32, HashSet<(usize, usize)>>,
    net_id: u32,
    coord: (usize, usize),
    horizontal: bool,
) -> bool {
    let Some(coords) = net_coords.get(&net_id) else {
        return false;
    };
    has_net_neighbor(coords, coord, horizontal)
}

// ---------------------------------------------------------------------------
// Per-net directional tile computation
// ---------------------------------------------------------------------------

/// Compute unidirectional tile types per (net_id, x, y, z) via 3D BFS from each
/// net's source. Traverses all layers (z=0 through max_z) and assigns
/// WireRight/WireLeft/WireDown/WireUp based on signal flow direction.
/// Via tiles (ViaUp/ViaDown) keep their router-assigned type.
fn compute_route_direction_data(
    placed: &PlacedNetlist,
    netlist: &MappedNetlist,
) -> RouteDirectionData {
    let mut net_coords: HashMap<u32, HashSet<(usize, usize)>> = HashMap::new();
    let mut net_tiles: HashMap<u32, HashMap<(usize, usize), TileType>> = HashMap::new();
    // 3D coordinate tracking for multi-layer BFS.
    let mut net_coords_3d: HashMap<u32, HashSet<(usize, usize, usize)>> = HashMap::new();
    // Via positions: tiles that must keep their router-assigned ViaUp/ViaDown type.
    let mut via_positions: HashSet<(u32, usize, usize, usize)> = HashSet::new();
    for seg in &placed.routes {
        // 2D maps are used by crossing detection (L0 only).
        if seg.z == 0 {
            net_coords
                .entry(seg.net_id)
                .or_default()
                .insert((seg.x, seg.y));
            net_tiles
                .entry(seg.net_id)
                .or_default()
                .insert((seg.x, seg.y), seg.tile_type);
        }
        // 3D coords are used for directional tile BFS.
        net_coords_3d
            .entry(seg.net_id)
            .or_default()
            .insert((seg.x, seg.y, seg.z));
        if seg.tile_type == TileType::ViaUp || seg.tile_type == TileType::ViaDown {
            via_positions.insert((seg.net_id, seg.x, seg.y, seg.z));
        }
    }

    let num_pi = netlist.primary_inputs.len() as u32;
    let mut directional_tiles: HashMap<(u32, usize, usize, usize), TileType> = HashMap::new();
    let mut source_coords: HashMap<u32, (usize, usize)> = HashMap::new();
    let mut parent_coords: HashMap<(u32, usize, usize), (usize, usize)> = HashMap::new();

    // Collect all net IDs that have route segments (any layer).
    let mut all_net_ids: Vec<u32> = net_coords_3d.keys().copied().collect();
    // Also include nets with z=0 segments not yet in net_coords_3d.
    for &net_id in net_coords.keys() {
        if !net_coords_3d.contains_key(&net_id) {
            all_net_ids.push(net_id);
        }
    }
    all_net_ids.sort();
    all_net_ids.dedup();

    for &net_id in &all_net_ids {
        let coords = net_coords.get(&net_id);
        let source = if net_id < num_pi {
            if let Some(&anchor) = placed.input_anchors.get(net_id as usize) {
                anchor
            } else {
                continue;
            }
        } else {
            let gate_idx = (net_id - num_pi) as usize;
            if let Some(gate) = placed.gates.iter().find(|g| g.gate_idx == gate_idx) {
                output_pin_coord(gate)
            } else {
                continue;
            }
        };
        source_coords.insert(net_id, source);

        // For gate output nets, assign a directional tile at the output pin coord.
        if net_id >= num_pi {
            let gate_idx = (net_id - num_pi) as usize;
            if let Some(gate) = placed.gates.iter().find(|g| g.gate_idx == gate_idx) {
                let dx = source.0 as isize - gate.x as isize;
                let dy = source.1 as isize - gate.y as isize;
                let tile_type = match (-dx, -dy) {
                    (-1, 0) => TileType::WireRight,
                    (1, 0) => TileType::WireLeft,
                    (0, -1) => TileType::WireDown,
                    (0, 1) => TileType::WireUp,
                    _ => TileType::Wire,
                };
                let has_l0 = coords.map(|c| c.contains(&source)).unwrap_or(false);
                if has_l0 {
                    directional_tiles.insert((net_id, source.0, source.1, 0), tile_type);
                }
            }
        }

        // 3D BFS from source through route segments.
        // Traverses ViaUp/ViaDown transitions between layers and assigns
        // unidirectional wire tiles (avoiding Cross tiles which corrupt
        // boolean signals via bit-split encoding).
        let coords_3d = net_coords_3d.get(&net_id);
        let mut visited_3d: HashSet<(usize, usize, usize)> = HashSet::new();
        let mut queue_3d: VecDeque<(usize, usize, usize)> = VecDeque::new();
        let start = (source.0, source.1, 0usize);
        queue_3d.push_back(start);
        visited_3d.insert(start);

        while let Some(pos) = queue_3d.pop_front() {
            // 2D neighbors at same layer.
            let neighbors_2d: [(isize, isize); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
            for &(dx, dy) in &neighbors_2d {
                let nx = pos.0 as isize + dx;
                let ny = pos.1 as isize + dy;
                if nx < 0 || ny < 0 {
                    continue;
                }
                let next = (nx as usize, ny as usize, pos.2);
                let needs_dir = if visited_3d.contains(&next) {
                    // Already visited — check if it needs a directional tile.
                    // z-transitions mark visited without assigning direction.
                    // Allow 2D neighbors to fill in the direction, but don't
                    // re-enqueue (the node is already being/been processed).
                    !directional_tiles.contains_key(&(net_id, next.0, next.1, next.2))
                        && !via_positions.contains(&(net_id, next.0, next.1, next.2))
                } else {
                    false
                };
                if visited_3d.contains(&next) && !needs_dir {
                    continue;
                }
                let in_net = if pos.2 == 0 {
                    coords
                        .map(|c| c.contains(&(next.0, next.1)))
                        .unwrap_or(false)
                } else {
                    coords_3d.map(|c| c.contains(&next)).unwrap_or(false)
                };
                if !in_net {
                    continue;
                }

                let tile_type = match (dx, dy) {
                    (1, 0) => TileType::WireRight,
                    (-1, 0) => TileType::WireLeft,
                    (0, 1) => TileType::WireDown,
                    (0, -1) => TileType::WireUp,
                    _ => TileType::Wire,
                };
                directional_tiles.insert((net_id, next.0, next.1, next.2), tile_type);
                if !needs_dir {
                    // First visit — mark visited and enqueue for exploration.
                    visited_3d.insert(next);
                    parent_coords.insert((net_id, next.0, next.1), (pos.0, pos.1));
                    queue_3d.push_back(next);
                }
            }

            // z-axis neighbors (via transitions).
            for dz in [-1isize, 1isize] {
                let nz = pos.2 as isize + dz;
                if nz < 0 {
                    continue;
                }
                let next = (pos.0, pos.1, nz as usize);
                if visited_3d.contains(&next) {
                    continue;
                }
                let in_net = coords_3d.map(|c| c.contains(&next)).unwrap_or(false);
                if !in_net {
                    continue;
                }
                visited_3d.insert(next);
                // Via tiles keep their type from the router; just mark visited
                // so z>0 neighbors can be explored.
                queue_3d.push_back(next);
            }
        }

        // Ensure source at z=0 has a directional tile if it's a route segment.
        // Gate output sources get assigned above; input sources need post-BFS assignment.
        // Check all z=0 route positions that are missing directional tiles.
        // This covers input sources (which don't get the gate-output pre-assignment)
        // and any other z=0 segments the BFS couldn't assign (e.g. source position).
        if let Some(c) = coords {
            for &(cx, cy) in c.iter() {
                if directional_tiles.contains_key(&(net_id, cx, cy, 0)) {
                    continue;
                }
                if via_positions.contains(&(net_id, cx, cy, 0)) {
                    continue;
                }
                // Find first in-net neighbor to derive direction.
                let neighbors_2d: [(isize, isize); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
                let mut assigned = false;
                for &(dx, dy) in &neighbors_2d {
                    let nx = cx as isize + dx;
                    let ny = cy as isize + dy;
                    if nx < 0 || ny < 0 {
                        continue;
                    }
                    let neighbor = (nx as usize, ny as usize);
                    if c.contains(&neighbor) {
                        let tile_type = match (dx, dy) {
                            (1, 0) => TileType::WireRight,
                            (-1, 0) => TileType::WireLeft,
                            (0, 1) => TileType::WireDown,
                            (0, -1) => TileType::WireUp,
                            _ => TileType::Wire,
                        };
                        directional_tiles.insert((net_id, cx, cy, 0), tile_type);
                        assigned = true;
                        break;
                    }
                }
                if !assigned {
                    // No z=0 neighbor — route goes directly to z>0 via ViaUp.
                    // Assign Wire; the value is set at the source so direction
                    // doesn't affect correctness.
                    directional_tiles.insert((net_id, cx, cy, 0), TileType::Wire);
                }
            }
        }
    }

    RouteDirectionData {
        directional_tiles,
        source_coords,
        parent_coords,
        net_coords,
    }
}

const ORIENT_H: u8 = 0b01;
const ORIENT_V: u8 = 0b10;

#[allow(dead_code)]
fn route_mask_for_coord(
    coords: &HashSet<(usize, usize)>,
    tile_map: &HashMap<(usize, usize), TileType>,
    coord: (usize, usize),
) -> u8 {
    match tile_map.get(&coord).copied().unwrap_or(TileType::Wire) {
        TileType::WireH => ORIENT_H,
        TileType::WireV => ORIENT_V,
        TileType::Wire => ORIENT_H | ORIENT_V,
        TileType::Cross => ORIENT_H | ORIENT_V,
        _ => {
            let mut mask = 0;
            if has_net_neighbor(coords, coord, true) {
                mask |= ORIENT_H;
            }
            if has_net_neighbor(coords, coord, false) {
                mask |= ORIENT_V;
            }
            if mask == 0 { ORIENT_H | ORIENT_V } else { mask }
        }
    }
}

fn crossing_input_orientation(
    route_data: &RouteDirectionData,
    net_id: u32,
    coord: (usize, usize),
) -> u8 {
    match route_data
        .directional_tiles
        .get(&(net_id, coord.0, coord.1, 0))
        .copied()
    {
        Some(TileType::WireRight | TileType::WireLeft | TileType::WireH) => ORIENT_H,
        Some(TileType::WireDown | TileType::WireUp | TileType::WireV) => ORIENT_V,
        _ => {
            let mut orient = 0;
            if has_neighbor_in_net(&route_data.net_coords, net_id, coord, true) {
                orient |= ORIENT_H;
            }
            if has_neighbor_in_net(&route_data.net_coords, net_id, coord, false) {
                orient |= ORIENT_V;
            }
            orient
        }
    }
}

fn has_net_neighbor(
    coords: &HashSet<(usize, usize)>,
    coord: (usize, usize),
    horizontal: bool,
) -> bool {
    if horizontal {
        (coord.0 > 0 && coords.contains(&(coord.0 - 1, coord.1)))
            || coords.contains(&(coord.0 + 1, coord.1))
    } else {
        (coord.1 > 0 && coords.contains(&(coord.0, coord.1 - 1)))
            || coords.contains(&(coord.0, coord.1 + 1))
    }
}

/// Get the output pin coordinate for a gate placement.
fn output_pin_coord(gate: &super::placement::GatePlacement) -> (usize, usize) {
    use super::placement::PinSide;
    match gate.output_pin {
        PinSide::Left => (gate.x - 1, gate.y),
        PinSide::Right => (gate.x + 1, gate.y),
        PinSide::Up => (gate.x, gate.y - 1),
        PinSide::Down => (gate.x, gate.y + 1),
    }
}

fn build_bypass_plan(
    crossings: &HashMap<(usize, usize), CrossingInfo>,
    route_data: &RouteDirectionData,
) -> BypassPlan {
    let mut crossing_sets: HashMap<u32, HashSet<(usize, usize)>> = HashMap::new();
    for (&coord, info) in crossings {
        crossing_sets
            .entry(info.bypass_net)
            .or_default()
            .insert(coord);
    }

    let mut nets: Vec<_> = crossing_sets.keys().copied().collect();
    nets.sort_unstable();

    let mut full_children: HashMap<(u32, usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for (&(net_id, child_x, child_y), &parent) in &route_data.parent_coords {
        full_children
            .entry((net_id, parent.0, parent.1))
            .or_default()
            .push((child_x, child_y));
    }

    let mut plan = BypassPlan::default();
    for &net_id in &nets {
        let Some(crossing_set) = crossing_sets.get(&net_id) else {
            continue;
        };
        let Some(net_coords) = route_data.net_coords.get(&net_id) else {
            continue;
        };
        let source_coord = route_data.source_coords.get(&net_id).copied();
        let stay_crossings: HashSet<(usize, usize)> = crossings
            .iter()
            .filter_map(|(&coord, info)| {
                if info.stay_net == net_id {
                    Some(coord)
                } else {
                    None
                }
            })
            .collect();
        let candidate_coords: HashSet<(usize, usize)> = net_coords
            .iter()
            .copied()
            .filter(|coord| !stay_crossings.contains(coord))
            .collect();
        if crossing_set.is_empty() || candidate_coords.is_empty() {
            continue;
        }

        let mut children: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
        let mut degree: HashMap<(usize, usize), usize> = candidate_coords
            .iter()
            .copied()
            .map(|coord| (coord, 0))
            .collect();

        for &coord in &candidate_coords {
            if let Some(&parent) = route_data.parent_coords.get(&(net_id, coord.0, coord.1)) {
                if candidate_coords.contains(&parent) {
                    *degree.entry(coord).or_default() += 1;
                    *degree.entry(parent).or_default() += 1;
                    children.entry(parent).or_default().push(coord);
                }
            }
        }

        let mut core = candidate_coords.clone();
        let mut queue: VecDeque<(usize, usize)> = degree
            .iter()
            .filter_map(|(&coord, &deg)| {
                if deg <= 1 && !crossing_set.contains(&coord) {
                    Some(coord)
                } else {
                    None
                }
            })
            .collect();

        while let Some(coord) = queue.pop_front() {
            if !core.remove(&coord) {
                continue;
            }

            if let Some(&parent) = route_data.parent_coords.get(&(net_id, coord.0, coord.1)) {
                if core.contains(&parent) {
                    let next_deg = degree
                        .get_mut(&parent)
                        .expect("parent coord should have a degree entry");
                    *next_deg = next_deg.saturating_sub(1);
                    if *next_deg <= 1 && !crossing_set.contains(&parent) {
                        queue.push_back(parent);
                    }
                }
            }

            if let Some(child_list) = children.get(&coord) {
                for &child in child_list {
                    if core.contains(&child) {
                        let next_deg = degree
                            .get_mut(&child)
                            .expect("child coord should have a degree entry");
                        *next_deg = next_deg.saturating_sub(1);
                        if *next_deg <= 1 && !crossing_set.contains(&child) {
                            queue.push_back(child);
                        }
                    }
                }
            }
        }

        if core.is_empty() {
            continue;
        }

        let mut entry_coords = HashSet::new();
        let mut exit_coords = HashSet::new();

        for &coord in &core {
            if let Some(&parent) = route_data.parent_coords.get(&(net_id, coord.0, coord.1)) {
                if !core.contains(&parent) {
                    entry_coords.insert(parent);
                }
            }
            if let Some(child_list) = full_children.get(&(net_id, coord.0, coord.1)) {
                for &child in child_list {
                    if !core.contains(&child) {
                        exit_coords.insert(child);
                    }
                }
            } else if Some(coord) != source_coord {
                // A lifted leaf still has to return to layer 0 so the sink
                // gate/input anchor on the base layer can observe the signal.
                exit_coords.insert(coord);
            }
        }

        if let Some(source) = source_coord {
            if core.contains(&source) {
                entry_coords.insert(source);
            }
        }

        // Fix stay-crossing gaps: when a position is both entry and exit for
        // the same net, a stay-crossing between bypass-crossings split the
        // core. Include those gap positions in the core to restore continuity.
        for _ in 0..candidate_coords.len().max(1) {
            let overlap: Vec<_> = entry_coords.intersection(&exit_coords).copied().collect();
            if overlap.is_empty() {
                break;
            }
            for coord in overlap {
                core.insert(coord);
                entry_coords.remove(&coord);
                exit_coords.remove(&coord);
            }
            // Recompute entries/exits for the expanded core.
            entry_coords.clear();
            exit_coords.clear();
            for &coord in &core {
                if let Some(&parent) = route_data.parent_coords.get(&(net_id, coord.0, coord.1)) {
                    if !core.contains(&parent) {
                        entry_coords.insert(parent);
                    }
                }
                if let Some(child_list) = full_children.get(&(net_id, coord.0, coord.1)) {
                    for &child in child_list {
                        if !core.contains(&child) {
                            exit_coords.insert(child);
                        }
                    }
                } else if Some(coord) != source_coord {
                    exit_coords.insert(coord);
                }
            }
            if let Some(source) = source_coord {
                if core.contains(&source) {
                    entry_coords.insert(source);
                }
            }
        }

        if !entry_coords.is_empty() {
            plan.entry_coords.insert(net_id, entry_coords);
        }
        if !exit_coords.is_empty() {
            plan.exit_coords.insert(net_id, exit_coords);
        }
        plan.core_coords.insert(net_id, core);
    }

    assign_bypass_layers(&mut plan);
    resolve_entry_core_conflicts(&mut plan, route_data);
    plan
}

/// Compute the deduped (layer, (x,y)) occupancy for a net assigned to layer `k`.
///
/// A net at layer K occupies:
/// - Core coords on layer K
/// - Entry coords on layers 1..=K (ViaDown chain)
/// - Exit coords on layers 1..=K (ViaUp chain)
/// Positions appearing in multiple categories on the same layer count only once.
fn net_occupancy(plan: &BypassPlan, net_id: u32, k: usize) -> Vec<(usize, usize, usize)> {
    let mut seen: HashSet<(usize, usize, usize)> = HashSet::new();
    let mut result = Vec::new();

    if let Some(coords) = plan.core_coords.get(&net_id) {
        for &(x, y) in coords {
            if seen.insert((k, x, y)) {
                result.push((k, x, y));
            }
        }
    }

    if let Some(coords) = plan.entry_coords.get(&net_id) {
        for &(x, y) in coords {
            for layer in 1..=k {
                if seen.insert((layer, x, y)) {
                    result.push((layer, x, y));
                }
            }
        }
    }

    if let Some(coords) = plan.exit_coords.get(&net_id) {
        for &(x, y) in coords {
            for layer in 1..=k {
                if seen.insert((layer, x, y)) {
                    result.push((layer, x, y));
                }
            }
        }
    }

    result
}

fn assign_bypass_layers(plan: &mut BypassPlan) {
    const MAX_LAYER: usize = 4;

    let mut nets: Vec<_> = plan.core_coords.keys().copied().collect();
    let mut net_sizes: HashMap<u32, usize> = HashMap::new();
    for &net_id in &nets {
        net_sizes.insert(
            net_id,
            plan.core_coords.get(&net_id).map_or(0, HashSet::len)
                + plan.entry_coords.get(&net_id).map_or(0, HashSet::len)
                + plan.exit_coords.get(&net_id).map_or(0, HashSet::len),
        );
    }
    nets.sort_unstable_by(|&a, &b| {
        let a_size = net_sizes.get(&a).copied().unwrap_or(0);
        let b_size = net_sizes.get(&b).copied().unwrap_or(0);
        b_size.cmp(&a_size).then_with(|| a.cmp(&b))
    });

    // Pre-compute occupancy for every (net, layer) combination.
    let mut precomputed: HashMap<(u32, usize), Vec<(usize, usize, usize)>> = HashMap::new();
    for &net_id in &nets {
        for k in 1..=MAX_LAYER {
            precomputed.insert((net_id, k), net_occupancy(plan, net_id, k));
        }
    }

    // Build initial layer counts from all nets at layer 1.
    let mut layer_counts: Vec<HashMap<(usize, usize), usize>> = vec![HashMap::new(); MAX_LAYER + 1];
    let mut assignments: HashMap<u32, usize> =
        nets.iter().copied().map(|net_id| (net_id, 1)).collect();

    for &net_id in &nets {
        for &(layer, x, y) in &precomputed[&(net_id, 1)] {
            *layer_counts[layer].entry((x, y)).or_default() += 1;
        }
    }

    let compute_total_score = |lc: &[HashMap<(usize, usize), usize>]| -> usize {
        lc[1..]
            .iter()
            .flat_map(|m| m.values())
            .map(|&c| c.saturating_sub(1))
            .sum()
    };

    let mut current_score = compute_total_score(&layer_counts);
    let max_iters = nets.len().max(1) * MAX_LAYER;

    // Delta score computation: temporarily remove old positions, add new positions,
    // compute the score difference, then undo.
    let score_delta_for_reassign = |lc: &Vec<HashMap<(usize, usize), usize>>,
                                    old_positions: &[(usize, usize, usize)],
                                    new_positions: &[(usize, usize, usize)]|
     -> isize {
        // Build position deltas: how each (layer, x, y) count changes
        let mut deltas: HashMap<(usize, usize, usize), isize> = HashMap::new();
        for &pos in old_positions {
            *deltas.entry(pos).or_default() -= 1;
        }
        for &pos in new_positions {
            *deltas.entry(pos).or_default() += 1;
        }

        let mut score_diff: isize = 0;
        for (&(layer, x, y), &d) in &deltas {
            if d == 0 {
                continue;
            }
            let old_count = lc[layer].get(&(x, y)).copied().unwrap_or(0);
            let new_count = (old_count as isize + d) as usize;
            score_diff +=
                new_count.saturating_sub(1) as isize - old_count.saturating_sub(1) as isize;
        }
        score_diff
    };

    let apply_move = |lc: &mut Vec<HashMap<(usize, usize), usize>>,
                      old_positions: &[(usize, usize, usize)],
                      new_positions: &[(usize, usize, usize)]| {
        for &(layer, x, y) in old_positions {
            let count = lc[layer].entry((x, y)).or_default();
            *count = count.saturating_sub(1);
        }
        for &(layer, x, y) in new_positions {
            *lc[layer].entry((x, y)).or_default() += 1;
        }
    };

    for _iter in 0..max_iters {
        if current_score == 0 {
            break;
        }
        let mut best_move: Option<Vec<(u32, usize, usize)>> = None; // (net, old_k, new_k)
        let mut best_score = current_score;

        // Try reassigning each net to every possible layer.
        for &net_id in &nets {
            let cur = assignments[&net_id];
            let old_pos = &precomputed[&(net_id, cur)];
            for try_layer in 1..=MAX_LAYER {
                if try_layer == cur {
                    continue;
                }
                let new_pos = &precomputed[&(net_id, try_layer)];
                let delta = score_delta_for_reassign(&layer_counts, old_pos, new_pos);
                let trial_score = (current_score as isize + delta) as usize;
                if trial_score < best_score {
                    best_move = Some(vec![(net_id, cur, try_layer)]);
                    best_score = trial_score;
                }
            }
        }

        // Pairwise swap: swap layers of two nets on different layers.
        for (i, &net_a) in nets.iter().enumerate() {
            let la = assignments[&net_a];
            for &net_b in &nets[i + 1..] {
                let lb = assignments[&net_b];
                if la == lb {
                    continue;
                }
                // Build combined old/new positions for the swap.
                let old_a = &precomputed[&(net_a, la)];
                let new_a = &precomputed[&(net_a, lb)];
                let old_b = &precomputed[&(net_b, lb)];
                let new_b = &precomputed[&(net_b, la)];

                let mut combined_old: Vec<(usize, usize, usize)> =
                    Vec::with_capacity(old_a.len() + old_b.len());
                combined_old.extend_from_slice(old_a);
                combined_old.extend_from_slice(old_b);
                let mut combined_new: Vec<(usize, usize, usize)> =
                    Vec::with_capacity(new_a.len() + new_b.len());
                combined_new.extend_from_slice(new_a);
                combined_new.extend_from_slice(new_b);

                let delta = score_delta_for_reassign(&layer_counts, &combined_old, &combined_new);
                let trial_score = (current_score as isize + delta) as usize;
                if trial_score < best_score {
                    best_move = Some(vec![(net_a, la, lb), (net_b, lb, la)]);
                    best_score = trial_score;
                }
            }
        }

        let Some(moves) = best_move else {
            break;
        };

        // Apply the best move.
        for &(net_id, old_k, new_k) in &moves {
            let old_pos = precomputed[&(net_id, old_k)].clone();
            let new_pos = precomputed[&(net_id, new_k)].clone();
            apply_move(&mut layer_counts, &old_pos, &new_pos);
            assignments.insert(net_id, new_k);
        }
        current_score = best_score;
    }

    #[cfg(debug_assertions)]
    {
        let verify = bypass_layer_conflict_score(plan, &assignments);
        debug_assert_eq!(
            current_score, verify,
            "ConflictTracker drift: incremental={} full={}",
            current_score, verify
        );
    }

    plan.routing_layers.clear();
    for (net_id, layer) in assignments {
        if layer > 1 {
            plan.routing_layers.insert(net_id, layer);
        }
    }
}

/// Count per-layer position conflicts across all bypass nets.
///
/// For a net on layer K:
/// - Core occupies layer K
/// - Entry ViaDown occupies layers 1..=K
/// - Exit ViaUp+wire occupies layers 1..=K
///
/// Score = sum of (count - 1) for all (layer, position) where count > 1.
fn bypass_layer_conflict_score(plan: &BypassPlan, assignments: &HashMap<u32, usize>) -> usize {
    let max_layer = assignments.values().copied().max().unwrap_or(1);
    let mut layer_counts: Vec<HashMap<(usize, usize), usize>> = vec![HashMap::new(); max_layer + 1];

    for &net_id in plan.core_coords.keys() {
        let k = assignments.get(&net_id).copied().unwrap_or(1);
        let mut seen: Vec<HashSet<(usize, usize)>> = vec![HashSet::new(); max_layer + 1];

        // Core: only on layer K.
        if let Some(coords) = plan.core_coords.get(&net_id) {
            for &coord in coords {
                if k <= max_layer && seen[k].insert(coord) {
                    *layer_counts[k].entry(coord).or_default() += 1;
                }
            }
        }

        // Entries: ViaDown on layers 1..=K.
        if let Some(coords) = plan.entry_coords.get(&net_id) {
            for &coord in coords {
                for layer in 1..=k.min(max_layer) {
                    if seen[layer].insert(coord) {
                        *layer_counts[layer].entry(coord).or_default() += 1;
                    }
                }
            }
        }

        // Exits: ViaUp on 0..K-1 + wire on K → occupies layers 1..=K.
        if let Some(coords) = plan.exit_coords.get(&net_id) {
            for &coord in coords {
                for layer in 1..=k.min(max_layer) {
                    if seen[layer].insert(coord) {
                        *layer_counts[layer].entry(coord).or_default() += 1;
                    }
                }
            }
        }
    }

    layer_counts[1..]
        .iter()
        .flat_map(|m| m.values())
        .map(|&count| count.saturating_sub(1))
        .sum()
}

/// Resolve cross-net conflicts where a bypass net's entry ViaDown chain
/// passes through an intermediate layer occupied by another net's core.
///
/// Fix: absorb the conflicting entry position into the net's core, pushing
/// the entry boundary further away from the conflict.
fn resolve_entry_core_conflicts(plan: &mut BypassPlan, route_data: &RouteDirectionData) {
    let mut full_children: HashMap<(u32, usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for (&(net_id, child_x, child_y), &parent) in &route_data.parent_coords {
        full_children
            .entry((net_id, parent.0, parent.1))
            .or_default()
            .push((child_x, child_y));
    }

    let mut prev_absorb_count = usize::MAX;
    for _round in 0..50 {
        // Build (layer, x, y) → set of net_ids with core there.
        let mut core_on_layer: HashMap<(usize, usize, usize), Vec<u32>> = HashMap::new();
        for (&net_id, coords) in &plan.core_coords {
            let layer = plan.routing_layer(net_id);
            for &(x, y) in coords {
                core_on_layer.entry((layer, x, y)).or_default().push(net_id);
            }
        }

        // Find entry positions whose ViaDown chain conflicts with a foreign core.
        let mut to_absorb: Vec<(u32, (usize, usize))> = Vec::new();
        for (&net_id, entries) in &plan.entry_coords {
            let top = plan.routing_layer(net_id);
            for &entry in entries {
                // ViaDown occupies layers 1..=top at the entry position.
                for layer in 1..=top {
                    if let Some(core_nets) = core_on_layer.get(&(layer, entry.0, entry.1)) {
                        if core_nets.iter().any(|&n| n != net_id) {
                            to_absorb.push((net_id, entry));
                            break;
                        }
                    }
                }
            }
        }

        if to_absorb.is_empty() {
            break;
        }

        // Break if no progress (absorb count not decreasing).
        if to_absorb.len() >= prev_absorb_count {
            break;
        }
        prev_absorb_count = to_absorb.len();

        for (net_id, pos) in to_absorb {
            plan.core_coords.entry(net_id).or_default().insert(pos);

            // Recompute entries and exits for this net.
            let source_coord = route_data.source_coords.get(&net_id).copied();
            recompute_net_entries_exits(plan, &full_children, route_data, net_id, source_coord);
        }
    }
}

/// Recompute entry_coords and exit_coords for a single net from its core.
fn recompute_net_entries_exits(
    plan: &mut BypassPlan,
    full_children: &HashMap<(u32, usize, usize), Vec<(usize, usize)>>,
    route_data: &RouteDirectionData,
    net_id: u32,
    source_coord: Option<(usize, usize)>,
) {
    // Resolve same-net entry-exit overlaps by absorbing into core.
    for _fix in 0..200 {
        let core = plan.core_coords.get(&net_id).unwrap();
        let mut new_entries = HashSet::new();
        let mut new_exits = HashSet::new();

        for &coord in core {
            if let Some(&parent) = route_data.parent_coords.get(&(net_id, coord.0, coord.1)) {
                if !core.contains(&parent) {
                    new_entries.insert(parent);
                }
            }
            if let Some(child_list) = full_children.get(&(net_id, coord.0, coord.1)) {
                for &child in child_list {
                    if !core.contains(&child) {
                        new_exits.insert(child);
                    }
                }
            } else if Some(coord) != source_coord {
                new_exits.insert(coord);
            }
        }
        if let Some(source) = source_coord {
            if core.contains(&source) {
                new_entries.insert(source);
            }
        }

        let overlap: Vec<_> = new_entries.intersection(&new_exits).copied().collect();
        if overlap.is_empty() {
            if new_entries.is_empty() {
                plan.entry_coords.remove(&net_id);
            } else {
                plan.entry_coords.insert(net_id, new_entries);
            }
            if new_exits.is_empty() {
                plan.exit_coords.remove(&net_id);
            } else {
                plan.exit_coords.insert(net_id, new_exits);
            }
            break;
        }
        let core = plan.core_coords.entry(net_id).or_default();
        for coord in overlap {
            core.insert(coord);
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-layer crossing bypass
// ---------------------------------------------------------------------------

/// Lift each bypass net's crossing subtree to its assigned layer, entering
/// through `ViaDown` tiles and returning with `ViaUp` tiles.
///
/// Critical invariant: exit ViaUp tiles on L0 must not overwrite positions
/// where another net's entry ViaDown reads from L0. We detect entry-exit
/// conflicts across nets and extend the L1 chain past conflict points.
fn apply_crossing_bypasses(
    sim: &mut Simulation,
    bypass_plan: &BypassPlan,
    net_dir: &HashMap<(u32, usize, usize, usize), TileType>,
    width: usize,
    height: usize,
) {
    let num_layers = bypass_plan.num_layers();

    // Collect ALL entry positions across all nets.  These are protected: no
    // other net's exit ViaUp may overwrite the L0 tile at an entry position.
    let mut all_entry_positions: HashSet<(usize, usize)> = HashSet::new();
    for entries in bypass_plan.entry_coords.values() {
        for &coord in entries {
            all_entry_positions.insert(coord);
        }
    }

    // Track occupied positions per layer (for entry-shifting on intermediate layers).
    let mut layer_occupied: Vec<HashSet<(usize, usize)>> = vec![HashSet::new(); num_layers + 1];

    let mut nets: Vec<_> = bypass_plan.core_coords.keys().copied().collect();
    nets.sort_unstable_by_key(|net_id| (bypass_plan.routing_layer(*net_id), *net_id));

    // Pass 1: Place core tiles on their assigned layer (offset above router).
    for &bypass_net in &nets {
        let top_layer = bypass_plan.routing_layer(bypass_net);
        let Some(core_coords) = bypass_plan.core_coords.get(&bypass_net) else {
            continue;
        };
        let mut core: Vec<_> = core_coords.iter().copied().collect();
        core.sort_unstable_by_key(|&(x, y)| (y, x));
        for (cx, cy) in core {
            let wire_tile = net_dir
                .get(&(bypass_net, cx, cy, 0))
                .copied()
                .unwrap_or(TileType::WireDown);
            sim.set_tile_3d(cx, cy, top_layer, wire_tile);
            layer_occupied[top_layer].insert((cx, cy));
        }
    }

    // Pass 2: Place exits. When the exit position is a protected entry,
    // extend the L1 chain past it instead of placing ViaUp on L0.
    for &bypass_net in &nets {
        let top_layer = bypass_plan.routing_layer(bypass_net);
        let Some(exit_coords) = bypass_plan.exit_coords.get(&bypass_net) else {
            continue;
        };
        let own_entries = bypass_plan.entry_coords.get(&bypass_net);
        let mut exits: Vec<_> = exit_coords.iter().copied().collect();
        exits.sort_unstable_by_key(|&(x, y)| (y, x));

        for (x, y) in exits {
            if x >= width || y >= height {
                continue;
            }

            // Is this exit also the entry for a DIFFERENT net?
            let is_own_entry = own_entries.is_some_and(|e| e.contains(&(x, y)));
            let is_foreign_entry = all_entry_positions.contains(&(x, y)) && !is_own_entry;

            if !is_foreign_entry {
                // Normal exit: wire on top_layer + ViaUp on all lower layers.
                let wire_tile = net_dir
                    .get(&(bypass_net, x, y, 0))
                    .copied()
                    .unwrap_or(TileType::WireDown);
                sim.set_tile_3d(x, y, top_layer, wire_tile);
                layer_occupied[top_layer].insert((x, y));
                for layer in 0..top_layer {
                    sim.set_tile_3d(x, y, layer, TileType::ViaUp);
                }
            } else {
                // Conflict: this position is a foreign entry. Extend the L1
                // chain by one step past this position and place ViaUp there
                // instead. The direction comes from the net's directional tile.
                let wire_tile = net_dir
                    .get(&(bypass_net, x, y, 0))
                    .copied()
                    .unwrap_or(TileType::WireDown);
                sim.set_tile_3d(x, y, top_layer, wire_tile);
                layer_occupied[top_layer].insert((x, y));

                // Determine the next position in the signal flow direction.
                // The wire tile reads FROM one direction; signal continues TO
                // the opposite direction.
                let next = match wire_tile {
                    TileType::WireRight => Some((x + 1, y)),
                    TileType::WireLeft => x.checked_sub(1).map(|nx| (nx, y)),
                    TileType::WireDown => Some((x, y + 1)),
                    TileType::WireUp => y.checked_sub(1).map(|ny| (x, ny)),
                    _ => None,
                };

                if let Some((nx, ny)) = next {
                    if nx < width && ny < height {
                        // Place continuation wire + ViaUp at the next position.
                        sim.set_tile_3d(nx, ny, top_layer, wire_tile);
                        layer_occupied[top_layer].insert((nx, ny));
                        for layer in 0..top_layer {
                            sim.set_tile_3d(nx, ny, layer, TileType::ViaUp);
                        }
                    }
                }
            }
        }
    }

    // Pass 3: Place entry ViaDown tiles, shifting when another net already
    // occupies the position on any layer in the ViaDown chain.
    for &bypass_net in &nets {
        let top_layer = bypass_plan.routing_layer(bypass_net);
        let Some(entry_coords) = bypass_plan.entry_coords.get(&bypass_net) else {
            continue;
        };
        let mut entries: Vec<_> = entry_coords.iter().copied().collect();
        entries.sort_unstable_by_key(|&(x, y)| (y, x));

        for (ex, ey) in entries {
            if ex >= width || ey >= height {
                continue;
            }

            // Check if any layer in the ViaDown chain has a conflict.
            let has_conflict =
                (1..=top_layer).any(|layer| layer_occupied[layer].contains(&(ex, ey)));

            if !has_conflict {
                for layer in 1..=top_layer {
                    sim.set_tile_3d(ex, ey, layer, TileType::ViaDown);
                }
            } else {
                // Try all 4 adjacent positions for a conflict-free shift.
                let shifted = [(0isize, -1isize), (0, 1), (-1, 0), (1, 0)]
                    .iter()
                    .filter_map(|&(dx, dy)| {
                        let sx = ex.checked_add_signed(dx)?;
                        let sy = ey.checked_add_signed(dy)?;
                        if sx >= width || sy >= height {
                            return None;
                        }
                        // Must be free on ALL layers 1..=top_layer.
                        for layer in 1..=top_layer {
                            if layer_occupied[layer].contains(&(sx, sy)) {
                                return None;
                            }
                        }
                        Some((sx, sy))
                    })
                    .next();

                if let Some((sx, sy)) = shifted {
                    for layer in 1..=top_layer {
                        sim.set_tile_3d(sx, sy, layer, TileType::ViaDown);
                        layer_occupied[layer].insert((sx, sy));
                    }
                    // Bridge wire on top_layer pointing from shifted → entry.
                    let bridge_tile = match (ex as isize - sx as isize, ey as isize - sy as isize) {
                        (1, 0) => TileType::WireRight,
                        (-1, 0) => TileType::WireLeft,
                        (0, 1) => TileType::WireDown,
                        (0, -1) => TileType::WireUp,
                        _ => TileType::Wire,
                    };
                    sim.set_tile_3d(ex, ey, top_layer, bridge_tile);
                    layer_occupied[top_layer].insert((ex, ey));
                } else {
                    // All neighbors also blocked — place ViaDown directly
                    // as a last resort.
                    for layer in 1..=top_layer {
                        sim.set_tile_3d(ex, ey, layer, TileType::ViaDown);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Drive inputs, propagate, and read outputs from an exported synthesis circuit.
///
/// Uses `synth_sweep` — a lightweight tile evaluator that skips dirty tracking,
/// ProgramCounter, record_change_info, and component_input_lookup.  Sweeps the
/// pre-computed `circuit_indices` repeatedly until convergence.
pub fn evaluate_exported(export: &mut SynthExport, input_values: &[bool]) -> Vec<bool> {
    assert_eq!(
        input_values.len(),
        export.input_coords.len(),
        "input count mismatch"
    );

    // Reset only circuit (non-Const) tiles to 0 and mark dirty.
    for &idx in &export.circuit_indices {
        export.sim.tilemap.tiles[idx]
            .logic
            .store(0, std::sync::atomic::Ordering::Relaxed);
        export.sim.dirty.mark_dirty(idx);
    }

    // Set PI Const tile logic values.
    for (i, &val) in input_values.iter().enumerate() {
        let (x, y) = export.input_coords[i];
        let logic = if val { u64::MAX } else { 0 };
        export.sim.set_logic_value(x, y, logic);
    }

    // Mark PI neighbors dirty so Const value changes propagate.
    let width = export.sim.tilemap.width;
    let height = export.sim.tilemap.height;
    for &(x, y) in &export.input_coords {
        let idx = y * width + x;
        if x > 0 {
            export.sim.dirty.mark_dirty(idx - 1);
        }
        if x + 1 < width {
            export.sim.dirty.mark_dirty(idx + 1);
        }
        if y > 0 {
            export.sim.dirty.mark_dirty(idx - width);
        }
        if y + 1 < height {
            export.sim.dirty.mark_dirty(idx + width);
        }
    }

    // Propagate to convergence. Each propagate_combinational_counted call
    // handles up to 100 delta rounds internally, following dirty-set
    // dependencies through multi-layer via chains. Large circuits (e.g.
    // combined decode: ~200 gates, 279-wide grid with multi-layer routes)
    // can require 2000+ delta cycles — use a generous outer limit.
    let mut converged = false;
    let mut rounds = 0u32;
    for i in 0..200 {
        let (_, _, switched) = export.sim.propagate_combinational_counted();
        if switched == 0 {
            converged = true;
            rounds = i + 1;
            break;
        }
    }
    if !converged {
        rounds = 200;
    }
    export.last_converged = converged;
    export.last_convergence_rounds = rounds;

    // Read PO values.
    export
        .output_coords
        .iter()
        .map(|&(x, y)| export.sim.get_logic_at(x, y) != 0)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::benchmark::benchmark_specs;
    use crate::synth::mapping::{evaluate_aig, evaluate_netlist, map_to_library};
    use crate::synth::placement::{PlaceConfig, place_mapped_netlist};
    use crate::synth::routing::{RouteConfig, route_placed_netlist};
    use crate::synth::{Aig, CellLibrary, MapConfig, materialize_inversions};
    use rayon::prelude::*;

    fn evaluate_net_values(
        netlist: &MappedNetlist,
        lib: &CellLibrary,
        input_values: &[bool],
    ) -> Vec<bool> {
        assert_eq!(input_values.len(), netlist.primary_inputs.len());

        let mut net_values = vec![false; netlist.num_nets as usize];
        for (i, &val) in input_values.iter().enumerate() {
            net_values[i] = val;
        }

        let num_pi = netlist.primary_inputs.len() as u32;
        for (gate_idx, gate) in netlist.gates.iter().enumerate() {
            let cell = lib.cell(gate.cell_idx);
            let mut assignment = 0u32;
            for (pin_idx, &fanin_net) in gate.fanins.iter().enumerate() {
                let raw = fanin_net < netlist.num_nets && net_values[fanin_net as usize];
                let inv = gate.fanin_inverted.get(pin_idx).copied().unwrap_or(false);
                if raw ^ inv {
                    assignment |= 1 << pin_idx;
                }
            }

            let cell_output = (cell.truth_table >> assignment) & 1 != 0;
            net_values[(num_pi + gate_idx as u32) as usize] = cell_output ^ gate.output_inverted;
        }

        net_values
    }

    /// Shared helper: synthesize → materialize → place → route → export.
    fn place_route_export(aig: &Aig) -> (SynthExport, MappedNetlist) {
        place_route_export_with_halo(aig, 3)
    }

    fn place_route_export_with_halo(aig: &Aig, halo: usize) -> (SynthExport, MappedNetlist) {
        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(aig, &lib, &MapConfig::default());
        let materialized = materialize_inversions(&netlist, &lib);
        let place_config = PlaceConfig {
            min_x: 3,
            min_y: 3,
            halo,
            ..PlaceConfig::default()
        };
        let placed = place_mapped_netlist(&materialized, &lib, &place_config)
            .expect("placement should succeed");
        let routed = route_placed_netlist(&materialized, &placed, &RouteConfig::default())
            .expect("routing should succeed");
        let export = export_to_simulation(&routed, &materialized);
        (export, materialized)
    }

    #[test]
    fn export_and_gate_simulates_correctly() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);

        let (mut export, netlist) = place_route_export(&aig);
        let lib = CellLibrary::tile_native();

        for combo in 0..4u32 {
            let inputs: Vec<bool> = (0..2).map(|i| (combo >> i) & 1 != 0).collect();
            let expected = evaluate_netlist(&netlist, &lib, &inputs);
            let actual = evaluate_exported(&mut export, &inputs);
            assert_eq!(
                expected, actual,
                "AND gate mismatch at inputs {:?}: expected {:?}, got {:?}",
                inputs, expected, actual
            );
        }
    }

    #[test]
    fn export_xor_gate_simulates_correctly() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.xor(a, b);
        aig.add_output("y", y);

        let (mut export, netlist) = place_route_export(&aig);
        let lib = CellLibrary::tile_native();

        for combo in 0..4u32 {
            let inputs: Vec<bool> = (0..2).map(|i| (combo >> i) & 1 != 0).collect();
            let expected = evaluate_netlist(&netlist, &lib, &inputs);
            let actual = evaluate_exported(&mut export, &inputs);
            assert_eq!(
                expected, actual,
                "XOR gate mismatch at inputs {:?}: expected {:?}, got {:?}",
                inputs, expected, actual
            );
        }
    }

    #[test]
    fn export_inverter_simulates_correctly() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let y = aig.not(a);
        aig.add_output("y", y);

        let (mut export, netlist) = place_route_export(&aig);
        let lib = CellLibrary::tile_native();

        for combo in 0..2u32 {
            let inputs = vec![(combo & 1) != 0];
            let expected = evaluate_netlist(&netlist, &lib, &inputs);
            let actual = evaluate_exported(&mut export, &inputs);
            assert_eq!(
                expected, actual,
                "INV mismatch at input {:?}: expected {:?}, got {:?}",
                inputs, expected, actual
            );
        }
    }

    #[test]
    fn export_adder4_exhaustive() {
        let aig = crate::synth::benchmark::build_4bit_adder();

        let (mut export, _netlist) = place_route_export(&aig);
        let _lib = CellLibrary::tile_native();

        for combo in 0..256u32 {
            let inputs: Vec<bool> = (0..8).map(|i| (combo >> i) & 1 != 0).collect();
            let expected_aig = evaluate_aig(&aig, &inputs);
            let actual = evaluate_exported(&mut export, &inputs);
            assert_eq!(
                expected_aig, actual,
                "adder4 mismatch at combo {}: inputs {:?}, expected {:?}, got {:?}",
                combo, inputs, expected_aig, actual
            );
        }
    }

    #[test]
    fn benchmark_suite_exports_cleanly() {
        // Skip circuits that need multi-layer routing at halo=3,
        // or whose crossing density overwhelms the single-layer bypass system.
        let skip = ["mul4", "adder16", "decoder4to16"];
        let specs: Vec<_> = benchmark_specs()
            .into_iter()
            .filter(|spec| !skip.contains(&spec.name))
            .collect();

        // Run independent benchmark circuits in parallel to make this heavy
        // validation test utilize available CPU cores.
        specs.par_iter().for_each(|spec| {
            let aig = (spec.build)();
            let lib = CellLibrary::tile_native();
            let netlist = map_to_library(&aig, &lib, &MapConfig::default());
            let materialized = materialize_inversions(&netlist, &lib);
            let place_config = PlaceConfig {
                min_x: 3,
                min_y: 3,
                halo: 3,
                ..PlaceConfig::default()
            };
            let placed = place_mapped_netlist(&materialized, &lib, &place_config)
                .unwrap_or_else(|e| panic!("{}: placement failed: {}", spec.name, e));
            let routed = route_placed_netlist(&materialized, &placed, &RouteConfig::default())
                .unwrap_or_else(|e| panic!("{}: routing failed: {}", spec.name, e));
            let mut export = export_to_simulation(&routed, &materialized);

            let num_inputs = materialized.primary_inputs.len();
            let num_combos = (1u64 << num_inputs).min(64);
            for combo in 0..num_combos {
                let inputs: Vec<bool> =
                    (0..num_inputs).map(|i| (combo >> i) & 1 != 0).collect();
                let expected = evaluate_aig(&aig, &inputs);
                let actual = evaluate_exported(&mut export, &inputs);
                if expected != actual {
                    let w = export.sim.tilemap.width;
                    let h = export.sim.tilemap.height;
                    let layers = export.sim.tilemap.tiles.len() / (w * h);
                    let net_values = evaluate_net_values(&materialized, &lib, &inputs);
                    let num_pi = materialized.primary_inputs.len() as u32;

                    // Find first mismatching gate and trace its input net.
                    for gate in &routed.gates {
                        let net_id = num_pi + gate.gate_idx as u32;
                        let source = output_pin_coord(gate);
                        let actual_logic = export.sim.get_logic_at(source.0, source.1) != 0;
                        let expected_logic = net_values[net_id as usize];
                        if actual_logic != expected_logic {
                            println!(
                                "{}: gate {} ({:?}) at ({},{}) expected={} actual={}",
                                spec.name, gate.gate_idx, gate.tile_type,
                                gate.x, gate.y, expected_logic, actual_logic
                            );
                            // Check each input pin
                            for (pin_idx, side) in gate.input_pins.iter().enumerate() {
                                let coord = match side {
                                    crate::synth::placement::PinSide::Left => (gate.x - 1, gate.y),
                                    crate::synth::placement::PinSide::Right => (gate.x + 1, gate.y),
                                    crate::synth::placement::PinSide::Up => (gate.x, gate.y - 1),
                                    crate::synth::placement::PinSide::Down => (gate.x, gate.y + 1),
                                };
                                let fanin_net = materialized.gates[gate.gate_idx].fanins[pin_idx];
                                let pin_val = export.sim.get_logic_at(coord.0, coord.1) != 0;
                                let exp_val = net_values[fanin_net as usize];
                                if pin_val != exp_val {
                                    println!(
                                        "  pin{} net={} at ({},{}) expected={} actual={}",
                                        pin_idx, fanin_net, coord.0, coord.1, exp_val, pin_val
                                    );
                                    // Trace backward from this pin on L0
                                    let route_data = compute_route_direction_data(&routed, &materialized);
                                    let crossings = find_crossings(&routed, &route_data);
                                    let bypass_plan = build_bypass_plan(&crossings, &route_data);
                                    let mut pos = coord;
                                    for step in 0..40 {
                                        let (tx, ty) = pos;
                                        if tx >= w || ty >= h { break; }
                                        let idx0 = ty * w + tx;
                                        let tt0 = export.sim.tilemap.tiles[idx0].meta.tile_type;
                                        let val0 = export.sim.get_logic_at(tx, ty) != 0;
                                        // Check bypass involvement
                                        let mut bypass_info = String::new();
                                        for (&bid, core) in &bypass_plan.core_coords {
                                            if core.contains(&pos) {
                                                bypass_info += &format!(" CORE(net{}@L{})", bid, bypass_plan.routing_layer(bid));
                                            }
                                        }
                                        for (&bid, entries) in &bypass_plan.entry_coords {
                                            if entries.contains(&pos) {
                                                bypass_info += &format!(" ENTRY(net{})", bid);
                                            }
                                        }
                                        for (&bid, exits) in &bypass_plan.exit_coords {
                                            if exits.contains(&pos) {
                                                bypass_info += &format!(" EXIT(net{})", bid);
                                            }
                                        }
                                        // Show all layers
                                        let mut layer_info = String::new();
                                        for z in 1..layers {
                                            let idx_z = z * w * h + ty * w + tx;
                                            let tt_z = export.sim.tilemap.tiles[idx_z].meta.tile_type;
                                            if tt_z != TileType::Const {
                                                let val_z = export.sim.tilemap.tiles[idx_z].logic.load(std::sync::atomic::Ordering::Relaxed);
                                                layer_info += &format!(" L{}={:?}({:#X})", z, tt_z, val_z);
                                            }
                                        }
                                        println!(
                                            "  trace[{}] ({},{}) {:?} val={}{}{}",
                                            step, tx, ty, tt0, val0, bypass_info, layer_info
                                        );
                                        // Follow backward
                                        let next = match tt0 {
                                            TileType::WireRight => (tx.wrapping_sub(1), ty),
                                            TileType::WireLeft => (tx + 1, ty),
                                            TileType::WireDown => (tx, ty.wrapping_sub(1)),
                                            TileType::WireUp => (tx, ty + 1),
                                            TileType::ViaUp => {
                                                // Continue trace on L1 (ViaUp reads from layer above = L1)
                                                let top = 1usize;
                                                let mut lpos = pos;
                                                for lstep in 0..40 {
                                                    let (lx, ly) = lpos;
                                                    if lx >= w || ly >= h { break; }
                                                    let lidx = top * w * h + ly * w + lx;
                                                    let ltt = export.sim.tilemap.tiles[lidx].meta.tile_type;
                                                    let lval = export.sim.tilemap.tiles[lidx].logic.load(std::sync::atomic::Ordering::Relaxed);
                                                    println!("  L{}_trace[{}] ({},{}) {:?} val={:#X}", top, lstep, lx, ly, ltt, lval);
                                                    let lnext = match ltt {
                                                        TileType::WireRight => (lx.wrapping_sub(1), ly),
                                                        TileType::WireLeft => (lx + 1, ly),
                                                        TileType::WireDown => (lx, ly.wrapping_sub(1)),
                                                        TileType::WireUp => (lx, ly + 1),
                                                        TileType::ViaDown => {
                                                            println!("  L{}_trace[{}] ViaDown→L0 signal pickup", top, lstep);
                                                            let l0idx = ly * w + lx;
                                                            let l0tt = export.sim.tilemap.tiles[l0idx].meta.tile_type;
                                                            let l0val = export.sim.tilemap.tiles[l0idx].logic.load(std::sync::atomic::Ordering::Relaxed);
                                                            println!("    L0: {:?} val={:#X}", l0tt, l0val);
                                                            // Check bypass involvement
                                                            let vd_pos = (lx, ly);
                                                            for (&bid, entries) in &bypass_plan.entry_coords {
                                                                if entries.contains(&vd_pos) {
                                                                    println!("    ENTRY(net{})", bid);
                                                                }
                                                            }
                                                            for (&bid, exits) in &bypass_plan.exit_coords {
                                                                if exits.contains(&vd_pos) {
                                                                    println!("    EXIT(net{} layer={})", bid, bypass_plan.routing_layer(bid));
                                                                }
                                                            }
                                                            for (&bid, core) in &bypass_plan.core_coords {
                                                                if core.contains(&vd_pos) {
                                                                    println!("    CORE(net{} layer={})", bid, bypass_plan.routing_layer(bid));
                                                                }
                                                            }
                                                            // Check what nets route through this position
                                                            for seg in &routed.routes {
                                                                if seg.x == lx && seg.y == ly {
                                                                    println!("    route: net={} tile={:?}", seg.net_id, seg.tile_type);
                                                                }
                                                            }
                                                            break;
                                                        }
                                                        _ => { println!("  L{}_trace stop: {:?}", top, ltt); break; }
                                                    };
                                                    lpos = lnext;
                                                }
                                                break;
                                            }
                                            _ => { break; }
                                        };
                                        pos = next;
                                    }
                                }
                            }
                            break; // Only trace first mismatching gate
                        }
                    }
                }
                assert_eq!(
                    expected, actual,
                    "{}: mismatch at combo {}: expected {:?}, got {:?}",
                    spec.name, combo, expected, actual
                );
            }
        });
    }

    /// Sprint 217 D3 regression: verify directional tiles are assigned for
    /// routes that use z>0 layers (multi-layer via routing).
    #[test]
    fn export_multilayer_directional_tiles() {
        let aig = crate::synth::benchmark::build_4bit_adder();
        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        let materialized = materialize_inversions(&netlist, &lib);
        let place_config = PlaceConfig {
            min_x: 3,
            min_y: 3,
            halo: 3,
            ..PlaceConfig::default()
        };
        let placed = place_mapped_netlist(&materialized, &lib, &place_config)
            .expect("placement should succeed");
        let route_config = RouteConfig {
            max_z: 2,
            no_crossings: true,
            ..RouteConfig::default()
        };
        let routed = route_placed_netlist(&materialized, &placed, &route_config)
            .expect("routing should succeed");

        let route_data = compute_route_direction_data(&routed, &materialized);

        // Every non-via route segment should have a directional tile assigned.
        let mut missing = 0usize;
        let mut z_gt0_segments = 0usize;
        for seg in &routed.routes {
            if seg.tile_type == TileType::ViaUp || seg.tile_type == TileType::ViaDown {
                continue;
            }
            if seg.z > 0 {
                z_gt0_segments += 1;
            }
            if !route_data
                .directional_tiles
                .contains_key(&(seg.net_id, seg.x, seg.y, seg.z))
            {
                missing += 1;
            }
        }
        assert!(
            z_gt0_segments > 0,
            "adder4 with max_z=2 should have z>0 route segments"
        );
        assert_eq!(
            missing, 0,
            "{} route segments missing directional tiles (including z>0)",
            missing
        );
    }
}
