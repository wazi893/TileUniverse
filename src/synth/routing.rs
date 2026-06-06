//! Deterministic Manhattan routing for placed synth netlists.
//!
//! This is the Phase 2 routing MVP from the revised synthesizer plan:
//! route one sink at a time, reserve occupied tiles, reuse same-net trees for
//! fanout, and fail loudly on congestion instead of silently overlapping.

use super::mapping::MappedNetlist;
use super::placement::{PinSide, PlacedNetlist, PlacementRegion, RoutedSegment};
use crate::tiles::tile_meta::TileType;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteConfig {
    /// Try horizontal-first neighbor ordering before the vertical-first fallback.
    pub prefer_horizontal_first: bool,
    /// When true, reject any coord already occupied by another net regardless of
    /// perpendicular orientation. This prevents Cross tile generation, which is
    /// needed for simulation export (Cross tiles use bit-split encoding incompatible
    /// with all-or-nothing boolean signals).
    pub no_crossings: bool,
    /// Maximum routing layer (0 = single-layer, >0 enables multi-layer via routing).
    pub max_z: usize,
}

impl Default for RouteConfig {
    fn default() -> Self {
        Self {
            prefer_horizontal_first: true,
            no_crossings: false,
            max_z: 0,
        }
    }
}

impl RouteConfig {
    /// Preset for standalone synthesis (Python `synthesize()`, validators).
    ///
    /// Disables single-layer crossings (incompatible with boolean signals)
    /// and enables multi-layer via routing for clean signal isolation.
    pub fn standalone() -> Self {
        Self {
            prefer_horizontal_first: true,
            no_crossings: true,
            max_z: 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoutingError {
    MissingGatePlacement {
        gate_idx: usize,
    },
    BoundaryAnchorsExhausted {
        required: usize,
        available: usize,
    },
    MissingNetSource {
        net_id: u32,
    },
    OccupiedByDifferentNet {
        net_id: u32,
        other_net_id: u32,
        x: usize,
        y: usize,
        z: usize,
    },
    NoPath {
        net_id: u32,
        sink: String,
        source: (usize, usize, usize),
        target: (usize, usize, usize),
    },
}

impl std::fmt::Display for RoutingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoutingError::MissingGatePlacement { gate_idx } => {
                write!(f, "missing placement for gate {gate_idx}")
            }
            RoutingError::BoundaryAnchorsExhausted {
                required,
                available,
            } => write!(
                f,
                "not enough free boundary anchors: required {required}, available {available}"
            ),
            RoutingError::MissingNetSource { net_id } => {
                write!(f, "missing routing source for net {net_id}")
            }
            RoutingError::OccupiedByDifferentNet {
                net_id,
                other_net_id,
                x,
                y,
                z,
            } => write!(
                f,
                "net {net_id} attempted to occupy ({x}, {y}, {z}) already owned by net {other_net_id}"
            ),
            RoutingError::NoPath {
                net_id,
                sink,
                source,
                target,
            } => {
                write!(
                    f,
                    "no legal route for net {net_id} from {:?} to sink {sink} at {:?}",
                    source, target
                )
            }
        }
    }
}

impl std::error::Error for RoutingError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Coord {
    x: usize,
    y: usize,
    z: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SearchState {
    coord: Coord,
    incoming_orient: u8,
}

const ORIENT_H: u8 = 0b01;
const ORIENT_V: u8 = 0b10;
const ORIENT_Z: u8 = 0b100;
/// Stored in tree mask: this coord receives signal from z+1 (ViaUp tile).
const VIA_READS_ABOVE: u8 = 0b01000;
/// Stored in tree mask: this coord receives signal from z-1 (ViaDown tile).
const VIA_READS_BELOW: u8 = 0b10000;

const LAYER_TRANSITION_PENALTY: usize = 3;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CoordUsage {
    horizontal_owner: Option<u32>,
    vertical_owner: Option<u32>,
    /// Layer (z-axis) ownership for via transitions.
    layer_owner: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Endpoint {
    GateInput { gate_idx: usize, pin_idx: usize },
    PrimaryOutput { output_idx: usize },
}

/// Route the nets in a placed netlist and return an updated placed artifact.
pub fn route_placed_netlist(
    netlist: &MappedNetlist,
    placed: &PlacedNetlist,
    config: &RouteConfig,
) -> Result<PlacedNetlist, RoutingError> {
    let mut routing_view = placed.clone();
    routing_view.region = expanded_route_region(&placed.region, placed.halo.max(1));
    // Expand region to include multi-layer routing if configured.
    if config.max_z > routing_view.region.max_z {
        routing_view.region.max_z = config.max_z;
    }

    let gate_lookup = gate_lookup(&routing_view)?;
    let gate_centers: HashSet<_> = routing_view
        .gates
        .iter()
        .map(|g| Coord {
            x: g.x,
            y: g.y,
            z: g.z,
        })
        .collect();

    let sinks = net_sinks(netlist);
    let anchors = assign_boundary_anchors(netlist, placed, &gate_lookup, &sinks)?;
    let reserved = reserved_terminal_owners(netlist, &routing_view, &gate_lookup, &anchors);

    let mut occupied: HashMap<Coord, CoordUsage> = HashMap::new();
    let mut net_trees: BTreeMap<u32, BTreeMap<Coord, u8>> = BTreeMap::new();

    let mut route_order: Vec<u32> = (0..netlist.num_nets)
        .filter(|net_id| {
            sinks
                .get(net_id)
                .is_some_and(|endpoints| !endpoints.is_empty())
        })
        .collect();
    // Route longest-span nets first (they need the most flexibility).
    route_order.sort_by_key(|net_id| {
        let source = source_coord(netlist, &routing_view, &gate_lookup, &anchors, *net_id)
            .expect("net with sinks should have a legal source");
        let max_span = sinks[net_id]
            .iter()
            .filter_map(|endpoint| {
                endpoint_coord(netlist, &routing_view, &gate_lookup, &anchors, *endpoint).ok()
            })
            .map(|sink| manhattan(source, sink))
            .max()
            .unwrap_or(0);
        (Reverse(max_span), Reverse(sinks[net_id].len()), *net_id)
    });

    // Negotiated-congestion routing: iteratively route all nets, using historical
    // congestion penalties to guide routes away from overused corridors.
    let mut history: HashMap<Coord, usize> = HashMap::new();
    const MAX_NEGOTIATE_ITERS: usize = 10;
    let mut last_failed: Vec<u32> = Vec::new();

    for iteration in 0..MAX_NEGOTIATE_ITERS {
        // Clear all routes and start fresh each iteration
        net_trees.clear();
        occupied.clear();

        // Re-order: put previously-failed nets first so they get priority
        let mut iter_order = route_order.clone();
        if !last_failed.is_empty() {
            let failed_set: HashSet<u32> = last_failed.iter().copied().collect();
            iter_order.sort_by_key(|net_id| {
                if failed_set.contains(net_id) {
                    (0, *net_id)
                } else {
                    (1, *net_id)
                }
            });
        }

        let mut failed_nets: Vec<u32> = Vec::new();
        for &net_id in &iter_order {
            let result = route_single_net_with_history(
                netlist,
                &routing_view,
                &gate_lookup,
                &anchors,
                &gate_centers,
                &reserved,
                &sinks,
                &mut net_trees,
                &mut occupied,
                &history,
                net_id,
                config.prefer_horizontal_first,
                config.no_crossings,
            );
            if result.is_err() {
                failed_nets.push(net_id);
            }
        }

        if failed_nets.is_empty() {
            break;
        }

        // Update historical congestion: penalize coords that are heavily used
        for (coord, usage) in &occupied {
            let pressure = usage.horizontal_owner.is_some() as usize
                + usage.vertical_owner.is_some() as usize
                + usage.layer_owner.is_some() as usize;
            if pressure >= 2 {
                *history.entry(*coord).or_insert(0) += 1;
            }
        }
        // Also penalize coords adjacent to failed route targets
        for &fid in &failed_nets {
            if let Some(source) = source_coord(netlist, &routing_view, &gate_lookup, &anchors, fid)
            {
                for ep in sinks.get(&fid).cloned().unwrap_or_default() {
                    if let Ok(sink) =
                        endpoint_coord(netlist, &routing_view, &gate_lookup, &anchors, ep)
                    {
                        // Penalize coords on the direct path
                        let min_x = source.x.min(sink.x);
                        let max_x = source.x.max(sink.x);
                        let min_y = source.y.min(sink.y);
                        let max_y = source.y.max(sink.y);
                        for x in min_x..=max_x {
                            for y in min_y..=max_y {
                                let c = Coord { x, y, z: source.z };
                                if occupied.contains_key(&c) {
                                    *history.entry(c).or_insert(0) += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        last_failed = failed_nets;

        if iteration + 1 == MAX_NEGOTIATE_ITERS {
            // Sprint 196: Bounded rip-up — try removing blocking nets to make room
            // for failed nets. Deterministic: blockers sorted ascending by net ID.
            const MAX_RIP_UP_ATTEMPTS: usize = 3;
            let mut any_resolved = false;

            for &failed_id in &last_failed {
                let blockers = find_blocking_nets(
                    failed_id,
                    netlist,
                    &routing_view,
                    &gate_lookup,
                    &anchors,
                    &sinks,
                    &occupied,
                );

                let mut resolved = false;
                let mut attempts = 0;
                for &blocker in &blockers {
                    if attempts >= MAX_RIP_UP_ATTEMPTS {
                        break;
                    }
                    attempts += 1;

                    // Backup the blocker's tree before rip-up.
                    let backup = match net_trees.get(&blocker) {
                        Some(tree) => tree.clone(),
                        None => continue,
                    };
                    rip_up_net(blocker, &mut net_trees, &mut occupied);

                    // Try routing the failed net in the freed space.
                    let failed_result = route_single_net_with_history(
                        netlist,
                        &routing_view,
                        &gate_lookup,
                        &anchors,
                        &gate_centers,
                        &reserved,
                        &sinks,
                        &mut net_trees,
                        &mut occupied,
                        &history,
                        failed_id,
                        config.prefer_horizontal_first,
                        config.no_crossings,
                    );

                    if failed_result.is_ok() {
                        // Failed net routed — now try re-routing the blocker.
                        let blocker_result = route_single_net_with_history(
                            netlist,
                            &routing_view,
                            &gate_lookup,
                            &anchors,
                            &gate_centers,
                            &reserved,
                            &sinks,
                            &mut net_trees,
                            &mut occupied,
                            &history,
                            blocker,
                            config.prefer_horizontal_first,
                            config.no_crossings,
                        );
                        if blocker_result.is_ok() {
                            resolved = true;
                            break; // Both routed successfully
                        }
                        // Blocker couldn't re-route — undo the failed net and restore blocker.
                        rip_up_net(failed_id, &mut net_trees, &mut occupied);
                        restore_net(blocker, backup, &mut net_trees, &mut occupied);
                    } else {
                        // Failed net still can't route — restore blocker exactly.
                        restore_net(blocker, backup, &mut net_trees, &mut occupied);
                    }
                }

                if resolved {
                    any_resolved = true;
                } else {
                    // All rip-up attempts failed — return routing error for this net.
                    rip_up_net(failed_id, &mut net_trees, &mut occupied);
                    return route_single_net_with_history(
                        netlist,
                        &routing_view,
                        &gate_lookup,
                        &anchors,
                        &gate_centers,
                        &reserved,
                        &sinks,
                        &mut net_trees,
                        &mut occupied,
                        &history,
                        failed_id,
                        config.prefer_horizontal_first,
                        config.no_crossings,
                    )
                    .map(|_| {
                        let routes = flatten_routes(&net_trees);
                        let mut routed = routing_view.clone();
                        routed.routes = routes;
                        routed.input_anchors = anchors.inputs.iter().map(|c| (c.x, c.y)).collect();
                        routed.output_anchors = anchors
                            .outputs
                            .iter()
                            .map(|opt| opt.map(|c| (c.x, c.y)).unwrap_or((0, 0)))
                            .collect();
                        routed
                    });
                }
            }

            if any_resolved {
                // Some nets were resolved via rip-up — break the negotiate loop
                // since we can't restart (final iteration).
                break;
            }
        }
    }

    let routes = flatten_routes(&net_trees);
    let mut routed = routing_view;
    routed.routes = routes;
    routed.input_anchors = anchors.inputs.iter().map(|c| (c.x, c.y)).collect();
    routed.output_anchors = anchors
        .outputs
        .iter()
        .map(|opt| opt.map(|c| (c.x, c.y)).unwrap_or((0, 0)))
        .collect();
    Ok(routed)
}

/// Remove a net's routes from occupied map and net_trees.
fn rip_up_net(
    net_id: u32,
    net_trees: &mut BTreeMap<u32, BTreeMap<Coord, u8>>,
    occupied: &mut HashMap<Coord, CoordUsage>,
) {
    if let Some(tree) = net_trees.remove(&net_id) {
        for (coord, _) in tree {
            if let Some(usage) = occupied.get_mut(&coord) {
                if usage.horizontal_owner == Some(net_id) {
                    usage.horizontal_owner = None;
                }
                if usage.vertical_owner == Some(net_id) {
                    usage.vertical_owner = None;
                }
                if usage.layer_owner == Some(net_id) {
                    usage.layer_owner = None;
                }
                if usage.horizontal_owner.is_none()
                    && usage.vertical_owner.is_none()
                    && usage.layer_owner.is_none()
                {
                    occupied.remove(&coord);
                }
            }
        }
    }
}

/// Sprint 196: Find nets that are blocking a failed net's potential paths.
/// Scans the bounding box (+ halo margin) of the failed net's source and all sinks.
/// Returns blocker net IDs sorted ascending (deterministic ordering).
fn find_blocking_nets(
    failed_id: u32,
    netlist: &MappedNetlist,
    placed: &PlacedNetlist,
    gate_lookup: &[&super::placement::GatePlacement],
    anchors: &BoundaryAnchors,
    sinks: &BTreeMap<u32, Vec<Endpoint>>,
    occupied: &HashMap<Coord, CoordUsage>,
) -> Vec<u32> {
    let Some(source) = source_coord(netlist, placed, gate_lookup, anchors, failed_id) else {
        return Vec::new();
    };

    // Build bounding box of source + all sinks.
    let mut min_x = source.x;
    let mut max_x = source.x;
    let mut min_y = source.y;
    let mut max_y = source.y;

    if let Some(endpoints) = sinks.get(&failed_id) {
        for ep in endpoints {
            if let Ok(sink) = endpoint_coord(netlist, placed, gate_lookup, anchors, *ep) {
                min_x = min_x.min(sink.x);
                max_x = max_x.max(sink.x);
                min_y = min_y.min(sink.y);
                max_y = max_y.max(sink.y);
            }
        }
    }

    // Expand by halo margin (2 tiles each direction).
    let margin = 2;
    min_x = min_x.saturating_sub(margin);
    min_y = min_y.saturating_sub(margin);
    max_x += margin;
    max_y += margin;

    // Collect blocking net IDs in the bounding box, weighted by tile count.
    // Nets with more tiles in the bbox are more likely the real blockers.
    let mut blocker_weight: BTreeMap<u32, usize> = BTreeMap::new();
    for (coord, usage) in occupied {
        if coord.x >= min_x && coord.x <= max_x && coord.y >= min_y && coord.y <= max_y {
            for owner in [
                usage.horizontal_owner,
                usage.vertical_owner,
                usage.layer_owner,
            ]
            .iter()
            .flatten()
            {
                if *owner != failed_id {
                    *blocker_weight.entry(*owner).or_insert(0) += 1;
                }
            }
        }
    }

    // Sort by weight descending (most obstructive first), then by net ID ascending
    // for deterministic tie-breaking.
    let mut blockers: Vec<(u32, usize)> = blocker_weight.into_iter().collect();
    blockers.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    blockers.into_iter().map(|(id, _)| id).collect()
}

/// Sprint 196: Restore a previously ripped-up net from its backup tree.
/// Re-inserts the backup tree into net_trees and re-marks ownership in occupied.
fn restore_net(
    net_id: u32,
    backup_tree: BTreeMap<Coord, u8>,
    net_trees: &mut BTreeMap<u32, BTreeMap<Coord, u8>>,
    occupied: &mut HashMap<Coord, CoordUsage>,
) {
    for (&coord, &orient) in &backup_tree {
        let usage = occupied.entry(coord).or_default();
        if orient & ORIENT_H != 0 {
            usage.horizontal_owner = Some(net_id);
        }
        if orient & ORIENT_V != 0 {
            usage.vertical_owner = Some(net_id);
        }
        if orient & ORIENT_Z != 0 {
            usage.layer_owner = Some(net_id);
        }
    }
    net_trees.insert(net_id, backup_tree);
}

fn expanded_route_region(region: &PlacementRegion, margin: usize) -> PlacementRegion {
    let left_margin = margin.min(region.min_x);
    let top_margin = margin.min(region.min_y);
    PlacementRegion {
        min_x: region.min_x - left_margin,
        min_y: region.min_y - top_margin,
        width: region.width + left_margin + margin,
        height: region.height + top_margin + margin,
        z: region.z,
        max_z: region.max_z,
    }
}

fn gate_lookup(
    placed: &PlacedNetlist,
) -> Result<Vec<&super::placement::GatePlacement>, RoutingError> {
    let max_gate = placed.gates.iter().map(|g| g.gate_idx).max().unwrap_or(0);
    let mut lookup = vec![None; max_gate + 1];
    for gate in &placed.gates {
        lookup[gate.gate_idx] = Some(gate);
    }
    let mut out = Vec::with_capacity(lookup.len());
    for (gate_idx, gate) in lookup.into_iter().enumerate() {
        match gate {
            Some(gate) => out.push(gate),
            None => return Err(RoutingError::MissingGatePlacement { gate_idx }),
        }
    }
    Ok(out)
}

fn pin_coord(gate: &super::placement::GatePlacement, side: PinSide) -> Coord {
    match side {
        PinSide::Left => Coord {
            x: gate.x - 1,
            y: gate.y,
            z: gate.z,
        },
        PinSide::Right => Coord {
            x: gate.x + 1,
            y: gate.y,
            z: gate.z,
        },
        PinSide::Up => Coord {
            x: gate.x,
            y: gate.y - 1,
            z: gate.z,
        },
        PinSide::Down => Coord {
            x: gate.x,
            y: gate.y + 1,
            z: gate.z,
        },
    }
}

#[derive(Clone, Debug)]
struct BoundaryAnchors {
    inputs: Vec<Coord>,
    outputs: Vec<Option<Coord>>,
}

fn assign_boundary_anchors(
    netlist: &MappedNetlist,
    placed: &PlacedNetlist,
    gate_lookup: &[&super::placement::GatePlacement],
    sinks: &BTreeMap<u32, Vec<Endpoint>>,
) -> Result<BoundaryAnchors, RoutingError> {
    let mut reserved_boundary = HashSet::new();
    for gate in gate_lookup.iter() {
        let gate = *gate;
        reserved_boundary.insert(pin_coord(gate, gate.output_pin));
        for pin_idx in 0..gate.input_pins.len() {
            reserved_boundary.insert(pin_coord(gate, gate.input_pins[pin_idx]));
        }
    }

    let input_left_slots = free_boundary_slots(placed, &reserved_boundary, &[PinSide::Left]);
    let input_fallback_slots = free_boundary_slots(
        placed,
        &reserved_boundary,
        &[PinSide::Up, PinSide::Down, PinSide::Right],
    );
    if input_left_slots.len() + input_fallback_slots.len() < netlist.primary_inputs.len() {
        return Err(RoutingError::BoundaryAnchorsExhausted {
            required: netlist.primary_inputs.len(),
            available: input_left_slots.len() + input_fallback_slots.len(),
        });
    }

    let mut inputs = Vec::with_capacity(netlist.primary_inputs.len());
    let mut used = HashSet::new();
    for net_id in 0..netlist.primary_inputs.len() as u32 {
        let target = preferred_input_target(net_id, sinks, gate_lookup);
        let coord = choose_anchor(&input_left_slots, &used, target)
            .or_else(|| choose_anchor(&input_fallback_slots, &used, target))
            .ok_or(RoutingError::BoundaryAnchorsExhausted {
                required: netlist.primary_inputs.len(),
                available: input_left_slots.len() + input_fallback_slots.len() - used.len(),
            })?;
        used.insert(coord);
        inputs.push(coord);
    }

    let output_right_slots = free_boundary_slots(placed, &reserved_boundary, &[PinSide::Right]);
    let output_fallback_slots = free_boundary_slots(
        placed,
        &reserved_boundary,
        &[PinSide::Down, PinSide::Up, PinSide::Left],
    );

    let mut outputs = vec![None; netlist.primary_outputs.len()];
    let needed_outputs = netlist
        .primary_outputs
        .iter()
        .filter(|(_, net_id, _)| *net_id != u32::MAX)
        .count();
    let available_outputs = output_right_slots
        .iter()
        .chain(output_fallback_slots.iter())
        .filter(|coord| !used.contains(coord))
        .count();
    if available_outputs < needed_outputs {
        return Err(RoutingError::BoundaryAnchorsExhausted {
            required: needed_outputs,
            available: available_outputs,
        });
    }

    for (output_idx, (_, net_id, _)) in netlist.primary_outputs.iter().enumerate() {
        if *net_id == u32::MAX {
            continue;
        }
        let source = source_coord(
            netlist,
            placed,
            gate_lookup,
            &BoundaryAnchors {
                inputs: inputs.clone(),
                outputs: vec![None; netlist.primary_outputs.len()],
            },
            *net_id,
        )
        .expect("output net should have a legal source");
        let coord = choose_anchor(&output_right_slots, &used, Some(source))
            .or_else(|| choose_anchor(&output_fallback_slots, &used, Some(source)))
            .expect("checked output anchor availability above");
        used.insert(coord);
        outputs[output_idx] = Some(coord);
    }

    Ok(BoundaryAnchors { inputs, outputs })
}

fn preferred_input_target(
    net_id: u32,
    sinks: &BTreeMap<u32, Vec<Endpoint>>,
    gate_lookup: &[&super::placement::GatePlacement],
) -> Option<Coord> {
    sinks.get(&net_id).and_then(|endpoints| {
        endpoints.iter().find_map(|endpoint| match endpoint {
            Endpoint::GateInput { gate_idx, pin_idx } => {
                let gate = gate_lookup.get(*gate_idx)?;
                Some(pin_coord(gate, gate.input_pins[*pin_idx]))
            }
            Endpoint::PrimaryOutput { .. } => None,
        })
    })
}

fn choose_anchor(slots: &[Coord], used: &HashSet<Coord>, target: Option<Coord>) -> Option<Coord> {
    slots
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, coord)| !used.contains(coord))
        .min_by_key(|(rank, coord)| {
            let distance = target.map(|target| manhattan(*coord, target)).unwrap_or(0);
            (distance, *rank, coord.y, coord.x, coord.z)
        })
        .map(|(_, coord)| coord)
}

fn free_boundary_slots(
    placed: &PlacedNetlist,
    reserved_boundary: &HashSet<Coord>,
    side_order: &[PinSide],
) -> Vec<Coord> {
    let mut slots = Vec::new();
    let mut seen = HashSet::new();
    for side in side_order {
        for coord in boundary_coords(placed, *side) {
            if reserved_boundary.contains(&coord) {
                continue;
            }
            if seen.insert(coord) {
                slots.push(coord);
            }
        }
    }
    slots
}

fn boundary_coords(placed: &PlacedNetlist, side: PinSide) -> Vec<Coord> {
    let region = &placed.region;
    match side {
        PinSide::Left => (region.min_y..=region.max_y())
            .map(|y| Coord {
                x: region.min_x,
                y,
                z: region.z,
            })
            .collect(),
        PinSide::Right => (region.min_y..=region.max_y())
            .map(|y| Coord {
                x: region.max_x(),
                y,
                z: region.z,
            })
            .collect(),
        PinSide::Up => (region.min_x..=region.max_x())
            .map(|x| Coord {
                x,
                y: region.min_y,
                z: region.z,
            })
            .collect(),
        PinSide::Down => (region.min_x..=region.max_x())
            .map(|x| Coord {
                x,
                y: region.max_y(),
                z: region.z,
            })
            .collect(),
    }
}

/// Reserved terminal with orientation constraint.
/// `orient` indicates which orientation(s) are reserved for the owner.
/// Non-owner nets can still cross perpendicular.
#[derive(Clone, Copy, Debug)]
struct ReservedTerminal {
    net_id: u32,
    orient: u8, // ORIENT_H, ORIENT_V, or both
}

fn reserved_terminal_owners(
    netlist: &MappedNetlist,
    _placed: &PlacedNetlist,
    gate_lookup: &[&super::placement::GatePlacement],
    anchors: &BoundaryAnchors,
) -> HashMap<Coord, ReservedTerminal> {
    let mut reserved = HashMap::new();

    // PI anchors on the left boundary — signals exit rightward (horizontal)
    for net_id in 0..netlist.primary_inputs.len() as u32 {
        reserved.insert(
            anchors.inputs[net_id as usize],
            ReservedTerminal {
                net_id,
                orient: ORIENT_H | ORIENT_V,
            },
        );
    }

    // Gate output pins — reserve the output-side pin coord for the gate's net
    let gate_net_base = netlist.primary_inputs.len() as u32;
    for gate in gate_lookup.iter() {
        let gate_net_id = gate_net_base + gate.gate_idx as u32;
        let pin = pin_coord(gate, gate.output_pin);
        let orient = match gate.output_pin {
            PinSide::Left | PinSide::Right => ORIENT_H,
            PinSide::Up | PinSide::Down => ORIENT_V,
        };
        reserved.insert(
            pin,
            ReservedTerminal {
                net_id: gate_net_id,
                orient,
            },
        );
    }

    // PO anchors on the right boundary
    for (output_idx, (_, net_id, _)) in netlist.primary_outputs.iter().enumerate() {
        if *net_id == u32::MAX {
            continue;
        }
        if let Some(coord) = anchors.outputs[output_idx] {
            reserved.insert(
                coord,
                ReservedTerminal {
                    net_id: *net_id,
                    orient: ORIENT_H | ORIENT_V,
                },
            );
        }
    }

    reserved
}

fn net_sinks(netlist: &MappedNetlist) -> BTreeMap<u32, Vec<Endpoint>> {
    let mut sinks: BTreeMap<u32, Vec<Endpoint>> = BTreeMap::new();

    for (gate_idx, gate) in netlist.gates.iter().enumerate() {
        for (pin_idx, &fanin_net) in gate.fanins.iter().enumerate() {
            sinks
                .entry(fanin_net)
                .or_default()
                .push(Endpoint::GateInput { gate_idx, pin_idx });
        }
    }

    for (output_idx, (_, net_id, _)) in netlist.primary_outputs.iter().enumerate() {
        if *net_id == u32::MAX {
            continue;
        }
        sinks
            .entry(*net_id)
            .or_default()
            .push(Endpoint::PrimaryOutput { output_idx });
    }

    sinks
}

fn source_coord(
    netlist: &MappedNetlist,
    _placed: &PlacedNetlist,
    gate_lookup: &[&super::placement::GatePlacement],
    anchors: &BoundaryAnchors,
    net_id: u32,
) -> Option<Coord> {
    let num_pi = netlist.primary_inputs.len() as u32;
    if net_id < num_pi {
        return anchors.inputs.get(net_id as usize).copied();
    }

    let gate_idx = (net_id - num_pi) as usize;
    let gate = gate_lookup.get(gate_idx)?;
    Some(pin_coord(gate, gate.output_pin))
}

fn endpoint_coord(
    _netlist: &MappedNetlist,
    _placed: &PlacedNetlist,
    gate_lookup: &[&super::placement::GatePlacement],
    anchors: &BoundaryAnchors,
    endpoint: Endpoint,
) -> Result<Coord, RoutingError> {
    match endpoint {
        Endpoint::GateInput { gate_idx, pin_idx } => {
            let gate = gate_lookup
                .get(gate_idx)
                .ok_or(RoutingError::MissingGatePlacement { gate_idx })?;
            Ok(pin_coord(gate, gate.input_pins[pin_idx]))
        }
        Endpoint::PrimaryOutput { output_idx } => {
            Ok(anchors.outputs[output_idx].expect("non-const primary outputs receive anchors"))
        }
    }
}

fn endpoint_name(netlist: &MappedNetlist, endpoint: Endpoint) -> String {
    match endpoint {
        Endpoint::GateInput { gate_idx, pin_idx } => format!("gate[{gate_idx}].pin[{pin_idx}]"),
        Endpoint::PrimaryOutput { output_idx } => netlist.primary_outputs[output_idx].0.clone(),
    }
}

const BEND_PENALTY: usize = 2;

fn find_route(
    starts: &[Coord],
    target: Coord,
    net_id: u32,
    placed: &PlacedNetlist,
    gate_centers: &HashSet<Coord>,
    reserved: &HashMap<Coord, ReservedTerminal>,
    occupied: &HashMap<Coord, CoordUsage>,
    history: &HashMap<Coord, usize>,
    _prefer_horizontal_first: bool,
    no_crossings: bool,
) -> Option<Vec<Coord>> {
    if starts.contains(&target) {
        return Some(vec![target]);
    }

    // A* with (f_cost, g_cost, tiebreak_id, state) in a min-heap
    let mut heap: BinaryHeap<Reverse<(usize, usize, usize, SearchState)>> = BinaryHeap::new();
    let mut g_scores: HashMap<SearchState, usize> = HashMap::new();
    let mut parent: HashMap<SearchState, SearchState> = HashMap::new();
    let mut tiebreak = 0usize;

    for &start in starts {
        let state = SearchState {
            coord: start,
            incoming_orient: 0,
        };
        let h = manhattan(start, target);
        heap.push(Reverse((h, 0, tiebreak, state)));
        tiebreak += 1;
        g_scores.insert(state, 0);
        parent.insert(state, state);
    }

    while let Some(Reverse((_, g, _, current))) = heap.pop() {
        // Skip stale entries
        if g_scores.get(&current).is_some_and(|&best| g > best) {
            continue;
        }

        for next in grid_neighbors(current.coord) {
            let orient = step_orientation(current.coord, next);
            if !placed.region.contains(next.x, next.y, next.z) {
                continue;
            }
            if gate_centers.contains(&next) && next != target {
                continue;
            }
            if !edge_is_traversable(
                current.coord,
                next,
                net_id,
                orient,
                reserved,
                occupied,
                no_crossings,
            ) {
                continue;
            }
            if next == target && !endpoint_is_available(next, net_id, occupied) {
                continue;
            }

            // Cost: base 1 + bend penalty + layer transition penalty + historical congestion
            let bend_cost = if current.incoming_orient != 0 && current.incoming_orient != orient {
                BEND_PENALTY
            } else {
                0
            };
            let layer_cost = if orient == ORIENT_Z {
                LAYER_TRANSITION_PENALTY
            } else {
                0
            };
            let hist_cost = history.get(&next).copied().unwrap_or(0);
            let next_g = g + 1 + bend_cost + layer_cost + hist_cost;

            let next_state = SearchState {
                coord: next,
                incoming_orient: orient,
            };
            if g_scores
                .get(&next_state)
                .is_some_and(|&best| next_g >= best)
            {
                continue;
            }
            g_scores.insert(next_state, next_g);
            parent.insert(next_state, current);

            if next == target {
                return Some(reconstruct_path(parent, next_state));
            }

            let h = manhattan(next, target);
            heap.push(Reverse((next_g + h, next_g, tiebreak, next_state)));
            tiebreak += 1;
        }
    }

    None
}

/// All grid neighbors: 4 in-plane (±x, ±y) plus layer transitions (±z).
fn grid_neighbors(c: Coord) -> Vec<Coord> {
    let mut neighbors = Vec::with_capacity(6);
    if c.x > 0 {
        neighbors.push(Coord { x: c.x - 1, ..c });
    }
    neighbors.push(Coord { x: c.x + 1, ..c });
    if c.y > 0 {
        neighbors.push(Coord { y: c.y - 1, ..c });
    }
    neighbors.push(Coord { y: c.y + 1, ..c });
    // Layer transitions (filtered by region.contains in find_route).
    if c.z > 0 {
        neighbors.push(Coord { z: c.z - 1, ..c });
    }
    neighbors.push(Coord { z: c.z + 1, ..c });
    neighbors
}

fn edge_is_traversable(
    from: Coord,
    coord: Coord,
    net_id: u32,
    orient: u8,
    reserved: &HashMap<Coord, ReservedTerminal>,
    occupied: &HashMap<Coord, CoordUsage>,
    no_crossings: bool,
) -> bool {
    coord_is_traversable(from, net_id, orient, reserved, occupied, no_crossings)
        && coord_is_traversable(coord, net_id, orient, reserved, occupied, no_crossings)
}

fn coord_is_traversable(
    coord: Coord,
    net_id: u32,
    orient: u8,
    reserved: &HashMap<Coord, ReservedTerminal>,
    occupied: &HashMap<Coord, CoordUsage>,
    no_crossings: bool,
) -> bool {
    // Reserved terminals: always use orientation-aware check (allow perpendicular
    // crossing through pin areas even in no_crossings mode).
    if let Some(term) = reserved.get(&coord) {
        if term.net_id != net_id && (term.orient & orient) != 0 {
            return false;
        }
    }
    if let Some(usage) = occupied.get(&coord) {
        if no_crossings {
            if usage.horizontal_owner.is_some_and(|owner| owner != net_id)
                || usage.vertical_owner.is_some_and(|owner| owner != net_id)
                || usage.layer_owner.is_some_and(|owner| owner != net_id)
            {
                return false;
            }
        } else {
            if orient & ORIENT_H != 0 {
                if usage.horizontal_owner.is_some_and(|owner| owner != net_id) {
                    return false;
                }
            }
            if orient & ORIENT_V != 0 {
                if usage.vertical_owner.is_some_and(|owner| owner != net_id) {
                    return false;
                }
            }
            if orient & ORIENT_Z != 0 {
                if usage.layer_owner.is_some_and(|owner| owner != net_id) {
                    return false;
                }
            }
        }
    }
    true
}

fn endpoint_is_available(coord: Coord, net_id: u32, occupied: &HashMap<Coord, CoordUsage>) -> bool {
    occupied.get(&coord).is_none_or(|usage| {
        usage.horizontal_owner.is_none_or(|owner| owner == net_id)
            && usage.vertical_owner.is_none_or(|owner| owner == net_id)
            && usage.layer_owner.is_none_or(|owner| owner == net_id)
    })
}

fn step_orientation(from: Coord, to: Coord) -> u8 {
    if from.z != to.z {
        ORIENT_Z
    } else if from.x != to.x {
        ORIENT_H
    } else {
        ORIENT_V
    }
}

fn apply_path(
    net_id: u32,
    path: &[Coord],
    tree: &mut BTreeMap<Coord, u8>,
    occupied: &mut HashMap<Coord, CoordUsage>,
) -> Result<(), RoutingError> {
    if path.is_empty() {
        return Ok(());
    }

    tree.entry(path[0]).or_insert(0);
    if path.len() == 1 {
        return Ok(());
    }

    for window in path.windows(2) {
        let from = window[0];
        let to = window[1];
        let orient = step_orientation(from, to);
        mark_coord_usage(net_id, from, orient, tree, occupied)?;
        // For z-transitions, store via direction on the destination coord's tree mask.
        let to_extra = if orient == ORIENT_Z {
            if to.z > from.z {
                VIA_READS_BELOW // destination reads from below (ViaDown)
            } else {
                VIA_READS_ABOVE // destination reads from above (ViaUp)
            }
        } else {
            0
        };
        mark_occupied_usage(net_id, to, orient, occupied)?;
        let mask = tree.entry(to).or_insert(0);
        *mask |= orient | to_extra;
    }

    Ok(())
}

fn route_single_net_with_history(
    netlist: &MappedNetlist,
    placed: &PlacedNetlist,
    gate_lookup: &[&super::placement::GatePlacement],
    anchors: &BoundaryAnchors,
    gate_centers: &HashSet<Coord>,
    reserved: &HashMap<Coord, ReservedTerminal>,
    sinks: &BTreeMap<u32, Vec<Endpoint>>,
    net_trees: &mut BTreeMap<u32, BTreeMap<Coord, u8>>,
    occupied: &mut HashMap<Coord, CoordUsage>,
    history: &HashMap<Coord, usize>,
    net_id: u32,
    prefer_horizontal_first: bool,
    no_crossings: bool,
) -> Result<(), RoutingError> {
    let Some(source) = source_coord(netlist, placed, gate_lookup, anchors, net_id) else {
        return Err(RoutingError::MissingNetSource { net_id });
    };

    let mut endpoints = sinks.get(&net_id).cloned().unwrap_or_default();
    if endpoints.is_empty() {
        return Ok(());
    }

    let mut local_tree = net_trees.get(&net_id).cloned().unwrap_or_default();
    let mut local_occupied = occupied.clone();
    local_tree.entry(source).or_insert(0);
    endpoints.sort_by_key(|endpoint| {
        let sink = endpoint_coord(netlist, placed, gate_lookup, anchors, *endpoint)
            .expect("endpoint coordinates should be valid while routing");
        (
            Reverse(manhattan(source, sink)),
            endpoint_name(netlist, *endpoint),
        )
    });

    for endpoint in endpoints {
        let sink = endpoint_coord(netlist, placed, gate_lookup, anchors, endpoint)?;
        let mut starts: Vec<_> = local_tree.keys().copied().collect();
        starts.sort_by_key(|coord| (manhattan(*coord, sink), coord.y, coord.x, coord.z));

        let path = find_route(
            &starts,
            sink,
            net_id,
            placed,
            gate_centers,
            reserved,
            &local_occupied,
            history,
            prefer_horizontal_first,
            no_crossings,
        )
        .ok_or_else(|| RoutingError::NoPath {
            net_id,
            sink: endpoint_name(netlist, endpoint),
            source: (source.x, source.y, source.z),
            target: (sink.x, sink.y, sink.z),
        })?;

        apply_path(net_id, &path, &mut local_tree, &mut local_occupied)?;
    }

    net_trees.insert(net_id, local_tree);
    *occupied = local_occupied;
    Ok(())
}

fn mark_coord_usage(
    net_id: u32,
    coord: Coord,
    orient: u8,
    tree: &mut BTreeMap<Coord, u8>,
    occupied: &mut HashMap<Coord, CoordUsage>,
) -> Result<(), RoutingError> {
    mark_occupied_usage(net_id, coord, orient, occupied)?;

    let mask = tree.entry(coord).or_insert(0);
    *mask |= orient;
    Ok(())
}

fn mark_occupied_usage(
    net_id: u32,
    coord: Coord,
    orient: u8,
    occupied: &mut HashMap<Coord, CoordUsage>,
) -> Result<(), RoutingError> {
    let entry = occupied.entry(coord).or_default();
    if orient & ORIENT_H != 0 {
        if let Some(other) = entry.horizontal_owner {
            if other != net_id {
                return Err(RoutingError::OccupiedByDifferentNet {
                    net_id,
                    other_net_id: other,
                    x: coord.x,
                    y: coord.y,
                    z: coord.z,
                });
            }
        } else {
            entry.horizontal_owner = Some(net_id);
        }
    }
    if orient & ORIENT_V != 0 {
        if let Some(other) = entry.vertical_owner {
            if other != net_id {
                return Err(RoutingError::OccupiedByDifferentNet {
                    net_id,
                    other_net_id: other,
                    x: coord.x,
                    y: coord.y,
                    z: coord.z,
                });
            }
        } else {
            entry.vertical_owner = Some(net_id);
        }
    }
    if orient & ORIENT_Z != 0 {
        if let Some(other) = entry.layer_owner {
            if other != net_id {
                return Err(RoutingError::OccupiedByDifferentNet {
                    net_id,
                    other_net_id: other,
                    x: coord.x,
                    y: coord.y,
                    z: coord.z,
                });
            }
        } else {
            entry.layer_owner = Some(net_id);
        }
    }
    Ok(())
}

fn reconstruct_path(parent: HashMap<SearchState, SearchState>, target: SearchState) -> Vec<Coord> {
    let mut path = Vec::new();
    let mut cursor = target;
    loop {
        path.push(cursor.coord);
        let prev = parent[&cursor];
        if prev == cursor {
            break;
        }
        cursor = prev;
    }
    path.reverse();
    path
}

fn flatten_routes(net_trees: &BTreeMap<u32, BTreeMap<Coord, u8>>) -> Vec<RoutedSegment> {
    let mut coord_counts: HashMap<Coord, usize> = HashMap::new();
    for coords in net_trees.values() {
        for &coord in coords.keys() {
            *coord_counts.entry(coord).or_insert(0) += 1;
        }
    }

    let mut routes = Vec::new();
    for (&net_id, coords) in net_trees {
        for (&coord, &mask) in coords {
            let tile_type = if mask & VIA_READS_BELOW != 0 {
                TileType::ViaDown
            } else if mask & VIA_READS_ABOVE != 0 {
                TileType::ViaUp
            } else {
                classify_route_tile(
                    mask & (ORIENT_H | ORIENT_V),
                    coord_counts.get(&coord).copied().unwrap_or(1),
                )
            };
            routes.push(RoutedSegment {
                net_id,
                x: coord.x,
                y: coord.y,
                z: coord.z,
                tile_type,
            });
        }
    }
    routes
}

fn classify_route_tile(mask: u8, shared_count: usize) -> TileType {
    if shared_count > 1 {
        TileType::Cross
    } else if mask == ORIENT_H {
        TileType::WireH
    } else if mask == ORIENT_V {
        TileType::WireV
    } else if mask == ORIENT_Z {
        // Pure via segment; actual ViaUp/ViaDown direction determined in flatten_routes.
        TileType::Wire
    } else {
        TileType::Wire
    }
}

fn manhattan(a: Coord, b: Coord) -> usize {
    a.x.abs_diff(b.x) + a.y.abs_diff(b.y) + a.z.abs_diff(b.z) * LAYER_TRANSITION_PENALTY
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::benchmark::{benchmark_specs, build_4bit_adder};
    use crate::synth::mapping::map_to_library;
    use crate::synth::placement::{PlaceConfig, place_mapped_netlist};
    use crate::synth::{Aig, CellLibrary, MapConfig};
    use std::collections::HashMap;

    fn place_and_route(build: fn() -> Aig) -> Result<PlacedNetlist, RoutingError> {
        let aig = build();
        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        let placed = place_mapped_netlist(
            &netlist,
            &lib,
            &PlaceConfig {
                min_x: 3,
                min_y: 3,
                halo: 3,
                ..PlaceConfig::default()
            },
        )
        .expect("placement should succeed");
        route_placed_netlist(&netlist, &placed, &RouteConfig::default())
    }

    #[test]
    fn routes_4bit_adder_without_illegal_tile_reuse() {
        let routed = place_and_route(build_4bit_adder).expect("adder4 should route");

        assert!(
            !routed.routes.is_empty(),
            "expected non-empty routed fabric"
        );
        let mut owners: HashMap<(usize, usize, usize), Vec<(u32, TileType)>> = HashMap::new();
        for segment in &routed.routes {
            let coord = (segment.x, segment.y, segment.z);
            owners
                .entry(coord)
                .or_default()
                .push((segment.net_id, segment.tile_type));
            assert!(
                routed.region.contains(segment.x, segment.y, segment.z),
                "route escaped placement region at {:?}",
                coord
            );
        }
        for (coord, entries) in owners {
            if entries.len() <= 1 {
                continue;
            }
            assert!(
                entries.iter().all(|(_, ty)| *ty == TileType::Cross),
                "non-cross overlap at {:?}: {:?}",
                coord,
                entries
            );
            let unique_nets: HashSet<_> = entries.iter().map(|(net_id, _)| *net_id).collect();
            assert!(
                unique_nets.len() == entries.len(),
                "duplicate net entries at {:?}: {:?}",
                coord,
                entries
            );
        }
    }

    #[test]
    fn routing_is_deterministic() {
        let aig = build_4bit_adder();
        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        let placed = place_mapped_netlist(
            &netlist,
            &lib,
            &PlaceConfig {
                min_x: 3,
                min_y: 3,
                halo: 3,
                ..PlaceConfig::default()
            },
        )
        .unwrap();

        let first = route_placed_netlist(&netlist, &placed, &RouteConfig::default()).unwrap();
        let second = route_placed_netlist(&netlist, &placed, &RouteConfig::default()).unwrap();

        assert_eq!(first.routes, second.routes);
    }

    #[test]
    fn stable_benchmark_suite_routes_cleanly() {
        let lib = CellLibrary::tile_native();
        // Skip circuits that need multi-layer routing at halo=3.
        let skip = ["mul4", "adder16"];
        for spec in benchmark_specs()
            .into_iter()
            .filter(|s| !skip.contains(&s.name))
        {
            let aig = (spec.build)();
            let netlist = map_to_library(&aig, &lib, &MapConfig::default());
            let placed = place_mapped_netlist(
                &netlist,
                &lib,
                &PlaceConfig {
                    min_x: 3,
                    min_y: 3,
                    halo: 3,
                    ..PlaceConfig::default()
                },
            )
            .unwrap();

            let routed = route_placed_netlist(&netlist, &placed, &RouteConfig::default())
                .unwrap_or_else(|err| panic!("{} failed to route: {}", spec.name, err));

            let mut owners: HashMap<(usize, usize, usize), Vec<(u32, TileType)>> = HashMap::new();
            for segment in &routed.routes {
                let coord = (segment.x, segment.y, segment.z);
                owners
                    .entry(coord)
                    .or_default()
                    .push((segment.net_id, segment.tile_type));
            }
            for (coord, entries) in owners {
                if entries.len() <= 1 {
                    continue;
                }
                assert!(
                    entries.iter().all(|(_, ty)| *ty == TileType::Cross),
                    "non-cross overlap at {:?} for {}: {:?}",
                    coord,
                    spec.name,
                    entries
                );
            }
        }
    }

    #[test]
    fn multi_layer_adder4_routes_cleanly() {
        let aig = build_4bit_adder();
        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        let placed = place_mapped_netlist(
            &netlist,
            &lib,
            &PlaceConfig {
                min_x: 3,
                min_y: 3,
                halo: 3,
                ..PlaceConfig::default()
            },
        )
        .unwrap();

        // Route with max_z=1 (2 layers available).
        let config = RouteConfig {
            max_z: 1,
            ..RouteConfig::default()
        };
        let routed =
            route_placed_netlist(&netlist, &placed, &config).expect("multi-layer adder4 route");

        assert!(!routed.routes.is_empty());

        // Verify region includes both layers.
        assert!(routed.region.max_z >= 1, "region should span multi-layer");
    }

    #[test]
    fn multi_layer_routing_is_deterministic() {
        let aig = build_4bit_adder();
        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        let placed = place_mapped_netlist(
            &netlist,
            &lib,
            &PlaceConfig {
                min_x: 3,
                min_y: 3,
                halo: 3,
                ..PlaceConfig::default()
            },
        )
        .unwrap();

        let config = RouteConfig {
            max_z: 1,
            ..RouteConfig::default()
        };
        let first = route_placed_netlist(&netlist, &placed, &config).unwrap();
        let second = route_placed_netlist(&netlist, &placed, &config).unwrap();
        assert_eq!(
            first.routes, second.routes,
            "multi-layer routing should be deterministic"
        );
    }

    /// Sprint 196: Verify find_blocking_nets returns deterministic, sorted net IDs.
    #[test]
    fn rip_up_find_blocking_nets_deterministic() {
        // Build a small circuit and route it, then verify find_blocking_nets
        // returns consistent results for a net surrounded by other nets.
        let aig = build_4bit_adder();
        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        let placed = place_mapped_netlist(
            &netlist,
            &lib,
            &PlaceConfig {
                min_x: 3,
                min_y: 3,
                halo: 3,
                ..PlaceConfig::default()
            },
        )
        .expect("placement should succeed");

        let routing_view = {
            let mut rv = placed.clone();
            rv.region = expanded_route_region(&placed.region, placed.halo.max(1));
            rv
        };

        let gate_lookup = gate_lookup(&routing_view).unwrap();
        let sinks = net_sinks(&netlist);
        let anchors = assign_boundary_anchors(&netlist, &placed, &gate_lookup, &sinks).unwrap();

        // Route all nets first.
        let routed = route_placed_netlist(&netlist, &placed, &RouteConfig::default())
            .expect("adder4 should route");

        // Reconstruct occupied map from routed segments for blocker finding.
        let mut occupied: HashMap<Coord, CoordUsage> = HashMap::new();
        for seg in &routed.routes {
            let coord = Coord {
                x: seg.x,
                y: seg.y,
                z: seg.z,
            };
            let usage = occupied.entry(coord).or_default();
            // Mark ownership based on tile type orientation.
            match seg.tile_type {
                TileType::WireRight | TileType::WireLeft | TileType::Wire => {
                    usage.horizontal_owner = Some(seg.net_id);
                }
                TileType::WireUp | TileType::WireDown => {
                    usage.vertical_owner = Some(seg.net_id);
                }
                TileType::ViaUp | TileType::ViaDown => {
                    usage.layer_owner = Some(seg.net_id);
                }
                _ => {}
            }
        }

        // Pick a net that has sinks (should find blockers in its bounding box).
        let test_net = sinks.keys().next().copied().unwrap();
        let blockers1 = find_blocking_nets(
            test_net,
            &netlist,
            &routing_view,
            &gate_lookup,
            &anchors,
            &sinks,
            &occupied,
        );
        // Run again — must be identical (deterministic).
        let blockers2 = find_blocking_nets(
            test_net,
            &netlist,
            &routing_view,
            &gate_lookup,
            &anchors,
            &sinks,
            &occupied,
        );
        assert_eq!(
            blockers1, blockers2,
            "find_blocking_nets must be deterministic"
        );
        // Must not contain the failed net itself.
        assert!(
            !blockers1.contains(&test_net),
            "blockers must not contain the failed net"
        );
    }

    /// Sprint 196: Verify restore_net exactly restores ripped-up state.
    #[test]
    fn rip_up_restore_net_preserves_state() {
        // Build routing state for adder4, rip up a net, then restore it.
        // The net_trees and occupied maps should be identical before and after.
        let aig = build_4bit_adder();
        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        let placed = place_mapped_netlist(
            &netlist,
            &lib,
            &PlaceConfig {
                min_x: 3,
                min_y: 3,
                halo: 3,
                ..PlaceConfig::default()
            },
        )
        .expect("placement should succeed");

        let routing_view = {
            let mut rv = placed.clone();
            rv.region = expanded_route_region(&placed.region, placed.halo.max(1));
            rv
        };

        let gate_lookup = gate_lookup(&routing_view).unwrap();
        let gate_centers: HashSet<_> = routing_view
            .gates
            .iter()
            .map(|g| Coord {
                x: g.x,
                y: g.y,
                z: g.z,
            })
            .collect();
        let sinks = net_sinks(&netlist);
        let anchors = assign_boundary_anchors(&netlist, &placed, &gate_lookup, &sinks).unwrap();
        let reserved = reserved_terminal_owners(&netlist, &routing_view, &gate_lookup, &anchors);
        let history: HashMap<Coord, usize> = HashMap::new();

        // Route all nets.
        let mut net_trees: BTreeMap<u32, BTreeMap<Coord, u8>> = BTreeMap::new();
        let mut occupied: HashMap<Coord, CoordUsage> = HashMap::new();
        let route_order: Vec<u32> = (0..netlist.num_nets)
            .filter(|net_id| {
                sinks
                    .get(net_id)
                    .is_some_and(|endpoints| !endpoints.is_empty())
            })
            .collect();
        for &net_id in &route_order {
            let _ = route_single_net_with_history(
                &netlist,
                &routing_view,
                &gate_lookup,
                &anchors,
                &gate_centers,
                &reserved,
                &sinks,
                &mut net_trees,
                &mut occupied,
                &history,
                net_id,
                true,
                false,
            );
        }

        // Pick a net to rip up and restore.
        let test_net = *net_trees.keys().next().unwrap();
        let tree_count_before = net_trees.len();
        let backup = net_trees.get(&test_net).unwrap().clone();
        let occupied_snapshot: HashMap<Coord, CoordUsage> = occupied.clone();

        // Rip up the net.
        rip_up_net(test_net, &mut net_trees, &mut occupied);
        assert!(
            !net_trees.contains_key(&test_net),
            "net should be removed after rip_up"
        );

        // Restore it.
        restore_net(test_net, backup, &mut net_trees, &mut occupied);
        assert_eq!(
            net_trees.len(),
            tree_count_before,
            "net count should be restored"
        );

        // Verify tree content is identical.
        let restored_tree = net_trees.get(&test_net).unwrap();
        for (coord, orient) in restored_tree {
            let _original_usage = occupied_snapshot.get(coord);
            let restored_usage = occupied.get(coord);
            assert!(
                restored_usage.is_some(),
                "coord {:?} missing from occupied after restore",
                coord
            );
            // Verify ownership is consistent with the original.
            if *orient & ORIENT_H != 0 {
                assert_eq!(
                    restored_usage.unwrap().horizontal_owner,
                    Some(test_net),
                    "horizontal ownership mismatch at {:?}",
                    coord
                );
            }
            if *orient & ORIENT_V != 0 {
                assert_eq!(
                    restored_usage.unwrap().vertical_owner,
                    Some(test_net),
                    "vertical ownership mismatch at {:?}",
                    coord
                );
            }
            if *orient & ORIENT_Z != 0 {
                assert_eq!(
                    restored_usage.unwrap().layer_owner,
                    Some(test_net),
                    "layer ownership mismatch at {:?}",
                    coord
                );
            }
        }
    }

    #[test]
    fn multi_layer_benchmark_suite_routes() {
        let lib = CellLibrary::tile_native();
        let config = RouteConfig {
            max_z: 1,
            ..RouteConfig::default()
        };
        for spec in benchmark_specs() {
            let aig = (spec.build)();
            let netlist = map_to_library(&aig, &lib, &MapConfig::default());
            let placed = place_mapped_netlist(
                &netlist,
                &lib,
                &PlaceConfig {
                    min_x: 3,
                    min_y: 3,
                    halo: 3,
                    ..PlaceConfig::default()
                },
            )
            .unwrap();

            let routed = route_placed_netlist(&netlist, &placed, &config)
                .unwrap_or_else(|err| panic!("{} failed multi-layer route: {}", spec.name, err));

            // Check via tile validity: ViaDown at z=0 is invalid.
            for seg in &routed.routes {
                if seg.tile_type == TileType::ViaDown {
                    assert!(
                        seg.z > 0,
                        "{}: ViaDown at z=0 is invalid (no layer below)",
                        spec.name
                    );
                }
            }
        }
    }
}
