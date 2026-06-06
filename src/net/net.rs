use crate::field::FieldGrid;
use crate::simulation::Simulation;
use crate::tile_meta::TileType;
use crate::tilemap::{HEIGHT, WIDTH};

pub type NetId = u32;
const NET_ID_UNASSIGNED: NetId = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetKind {
    Data,
    Clock,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetStats {
    pub id: NetId,
    pub kind: NetKind,
    pub node_count: u32,
    pub driver_count: u32,
    pub sink_count: u32,
    pub fanout_max: u32,
    pub region: Option<u32>,
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub has_clock: bool,
    pub is_floating: bool,
    pub is_mixed_region: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetReport {
    pub nets: Vec<NetStats>,
}

struct NetWork<'a> {
    sim: &'a Simulation,
    net_ids: FieldGrid<NetId>,
}

impl<'a> NetWork<'a> {
    fn new(sim: &'a Simulation) -> Self {
        Self {
            sim,
            net_ids: FieldGrid::new(WIDTH, HEIGHT, NET_ID_UNASSIGNED),
        }
    }

    fn run(&mut self) -> NetReport {
        let mut nets = Vec::new();
        let mut next_id: NetId = 1;

        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                if self.net_ids.get(x, y) != NET_ID_UNASSIGNED {
                    continue;
                }
                let tile = match self.sim.tilemap.get_tile(x, y) {
                    Some(t) => t,
                    None => continue,
                };
                let logic = tile.logic.load(std::sync::atomic::Ordering::Relaxed);
                if !is_participating_tile(tile.meta.tile_type, logic) {
                    continue;
                }
                let stats = self.bfs_net(next_id, x, y);
                nets.push(stats);
                next_id += 1;
            }
        }

        NetReport { nets }
    }

    fn bfs_net(&mut self, id: NetId, start_x: usize, start_y: usize) -> NetStats {
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((start_x, start_y));
        self.net_ids.set(start_x, start_y, id);

        let mut node_count = 0u32;
        let mut driver_count = 0u32;
        let mut sink_count = 0u32;
        let mut fanout_max = 0u32;
        let mut has_clock = false;
        let mut region: Option<u32> = None;
        let mut x0 = start_x as u32;
        let mut y0 = start_y as u32;
        let mut x1 = start_x as u32;
        let mut y1 = start_y as u32;

        while let Some((x, y)) = queue.pop_front() {
            node_count += 1;
            x0 = x0.min(x as u32);
            y0 = y0.min(y as u32);
            x1 = x1.max(x as u32);
            y1 = y1.max(y as u32);

            if let Some(r) = self.sim.region_at(x as u32, y as u32) {
                region = match region {
                    None => Some(r),
                    Some(prev) if prev == r => Some(r),
                    _ => None,
                };
            }

            let tile = self.sim.tilemap.get_tile(x, y).unwrap();
            let ttype = tile.meta.tile_type;
            let logic = tile.logic.load(std::sync::atomic::Ordering::Relaxed);
            if matches!(ttype, TileType::ClockGlobal) {
                has_clock = true;
            }
            if is_driver(ttype, logic) {
                driver_count += 1;
            }
            if is_sink(ttype) {
                sink_count += 1;
            }

            let neighbors = connected_neighbors(self.sim, x, y, ttype);
            let mut fanout = 0u32;
            for (nx, ny) in neighbors {
                fanout += 1;
                if self.net_ids.get(nx, ny) == NET_ID_UNASSIGNED {
                    let ntile = self.sim.tilemap.get_tile(nx, ny).unwrap();
                    let nlogic = ntile.logic.load(std::sync::atomic::Ordering::Relaxed);
                    if is_participating_tile(ntile.meta.tile_type, nlogic) {
                        self.net_ids.set(nx, ny, id);
                        queue.push_back((nx, ny));
                    }
                }
            }
            fanout_max = fanout_max.max(fanout);
        }

        let is_floating = driver_count == 0 && sink_count > 0;
        let is_mixed_region = region.is_none() && node_count > 0;
        let kind = if has_clock {
            NetKind::Clock
        } else {
            NetKind::Data
        };

        NetStats {
            id,
            kind,
            node_count,
            driver_count,
            sink_count,
            fanout_max,
            region,
            x0,
            y0,
            x1,
            y1,
            has_clock,
            is_floating,
            is_mixed_region,
        }
    }
}

fn is_participating_tile(tile_type: TileType, logic: u64) -> bool {
    match tile_type {
        TileType::Wire => logic != 0,
        TileType::And
        | TileType::Or
        | TileType::Xor
        | TileType::Not
        | TileType::Latch
        | TileType::Register8
        | TileType::ClockGlobal => true,
        TileType::VmSpawner | TileType::VmStatus | TileType::QDemo => false,
        // EPIC 103: Arithmetic tiles participate
        TileType::Add
        | TileType::Sub
        | TileType::Mul
        | TileType::Div
        | TileType::Mod
        | TileType::Shl
        | TileType::Shr => true,
        // EPIC 103: Comparison tiles participate
        TileType::Lt
        | TileType::Gt
        | TileType::Eq
        | TileType::Neq
        | TileType::Lte
        | TileType::Gte => true,
        // EPIC 103: Routing & Special
        TileType::Mux | TileType::Zero | TileType::Neg | TileType::Abs => true,
        // EPIC 104: Memory tiles participate
        TileType::Ram | TileType::Counter | TileType::Const => true,
        // Wire Crossing Tiles - participate like Wire
        TileType::Cross
        | TileType::WireH
        | TileType::WireV
        | TileType::WireDown
        | TileType::WireRight
        | TileType::WireUp
        | TileType::WireLeft => logic != 0,
        TileType::ComponentOutput => true,
        // Bus Architecture
        TileType::BusInterface => true,
        // Memory Controller
        TileType::MemoryPort => true,
        // EPIC 116: CPU Tiles
        TileType::CpuHead | TileType::Register | TileType::Console => true,
        // CPU Building Blocks - always participate
        TileType::Decoder3to8
        | TileType::Mux8to1
        | TileType::Mux4to1
        | TileType::Demux1to8
        | TileType::RegEnable
        | TileType::ProgramCounter
        | TileType::Selector => true,
        // Ising Mode Tiles - always participate
        TileType::IsingNode | TileType::IsingBias => true,
        // Multi-Clock Domains
        TileType::ClockDivider | TileType::Synchronizer => true,
        // Phase 1: Fully Tile-Based CPU
        TileType::AddCarry | TileType::BitSelect => true,
        // Wire Crossing Tiles (unidirectional)
        TileType::WireCross | TileType::WireCrossVert | TileType::VBusIn | TileType::VBusOut => {
            logic != 0
        }
        TileType::ViaUp | TileType::ViaDown => logic != 0,
        TileType::WeightedViaUp | TileType::WeightedViaDown => logic != 0,
        TileType::ThresholdViaUp | TileType::ThresholdViaDown => logic != 0,
        TileType::SubBorrow => true,
        TileType::Mux16to1 => true,
        // Sprint 127: 64-bit Scaling Primitives
        TileType::Register64 => true,
        TileType::CarryDetect => true,
        TileType::Decoder6to64 => true,
    }
}

fn is_driver(tile_type: TileType, logic: u64) -> bool {
    match tile_type {
        TileType::ClockGlobal | TileType::Register8 => true,
        TileType::ComponentOutput => true,
        TileType::BusInterface => true,
        TileType::MemoryPort => true,
        TileType::ClockDivider | TileType::Synchronizer => true,
        TileType::VmSpawner | TileType::VmStatus | TileType::QDemo => false,
        _ => logic != 0,
    }
}

fn is_sink(tile_type: TileType) -> bool {
    matches!(
        tile_type,
        TileType::And
            | TileType::Or
            | TileType::Xor
            | TileType::Not
            | TileType::Latch
            | TileType::Register8
    )
}

fn connected_neighbors(
    sim: &Simulation,
    x: usize,
    y: usize,
    tile_type: TileType,
) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(4);
    let xi = x as isize;
    let yi = y as isize;

    let neighbors = [(xi - 1, yi), (xi + 1, yi), (xi, yi - 1), (xi, yi + 1)];

    let in_bounds = |x: isize, y: isize| -> Option<(usize, usize)> {
        if x < 0 || y < 0 {
            None
        } else {
            let (xu, yu) = (x as usize, y as usize);
            if xu < WIDTH && yu < HEIGHT {
                Some((xu, yu))
            } else {
                None
            }
        }
    };

    match tile_type {
        TileType::Wire => {
            for (nx, ny) in neighbors.iter().filter_map(|(nx, ny)| in_bounds(*nx, *ny)) {
                out.push((nx, ny));
            }
        }
        TileType::And | TileType::Or | TileType::Xor => {
            for (nx, ny) in [(-1, 0), (1, 0)]
                .iter()
                .filter_map(|(dx, dy)| in_bounds(xi + dx, yi + dy))
            {
                out.push((nx, ny));
            }
        }
        TileType::Not => {
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
        }
        TileType::Latch | TileType::Register8 => {
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
            // Clock neighbors: any adjacent ClockGlobal counts
            for (nx, ny) in neighbors.iter().filter_map(|(nx, ny)| in_bounds(*nx, *ny)) {
                if let Some(t) = sim.tilemap.get_tile(nx, ny) {
                    if matches!(t.meta.tile_type, TileType::ClockGlobal) {
                        out.push((nx, ny));
                    }
                }
            }
        }
        TileType::ClockGlobal => {
            for (nx, ny) in neighbors.iter().filter_map(|(nx, ny)| in_bounds(*nx, *ny)) {
                out.push((nx, ny));
            }
        }
        TileType::VmSpawner | TileType::VmStatus | TileType::QDemo => {}
        // EPIC 103: Arithmetic tiles take left/right inputs
        TileType::Add
        | TileType::Sub
        | TileType::Mul
        | TileType::Div
        | TileType::Mod
        | TileType::Shl
        | TileType::Shr => {
            for (nx, ny) in [(-1, 0), (1, 0)]
                .iter()
                .filter_map(|(dx, dy)| in_bounds(xi + dx, yi + dy))
            {
                out.push((nx, ny));
            }
        }
        // EPIC 103: Comparison tiles take left/right inputs
        TileType::Lt
        | TileType::Gt
        | TileType::Eq
        | TileType::Neq
        | TileType::Lte
        | TileType::Gte => {
            for (nx, ny) in [(-1, 0), (1, 0)]
                .iter()
                .filter_map(|(dx, dy)| in_bounds(xi + dx, yi + dy))
            {
                out.push((nx, ny));
            }
        }
        // EPIC 103: Mux takes up as selector, left/right as data
        TileType::Mux => {
            for (nx, ny) in [(-1, 0), (1, 0), (0, -1)]
                .iter()
                .filter_map(|(dx, dy)| in_bounds(xi + dx, yi + dy))
            {
                out.push((nx, ny));
            }
        }
        // EPIC 103: Zero, Neg, Abs take only left input
        TileType::Zero | TileType::Neg | TileType::Abs => {
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
        }
        // EPIC 104: Ram/Counter take up as enable, left as data
        TileType::Ram | TileType::Counter => {
            for (nx, ny) in [(-1, 0), (0, -1)]
                .iter()
                .filter_map(|(dx, dy)| in_bounds(xi + dx, yi + dy))
            {
                out.push((nx, ny));
            }
        }
        // EPIC 104: Const has no inputs
        TileType::Const => {}
        // Wire Crossing Tiles
        // Cross: connects all 4 neighbors (but signals don't mix at runtime)
        TileType::Cross => {
            for (nx, ny) in neighbors.iter().filter_map(|(nx, ny)| in_bounds(*nx, *ny)) {
                out.push((nx, ny));
            }
        }
        // WireH: only connects horizontally (left/right)
        TileType::WireH => {
            for (nx, ny) in [(-1, 0), (1, 0)]
                .iter()
                .filter_map(|(dx, dy)| in_bounds(xi + dx, yi + dy))
            {
                out.push((nx, ny));
            }
        }
        // WireV: only connects vertically (up/down)
        TileType::WireV => {
            for (nx, ny) in [(0, -1), (0, 1)]
                .iter()
                .filter_map(|(dx, dy)| in_bounds(xi + dx, yi + dy))
            {
                out.push((nx, ny));
            }
        }
        // WireDown: only reads from up (unidirectional, signal flows downward)
        TileType::WireDown => {
            if let Some((nx, ny)) = in_bounds(xi, yi - 1) {
                out.push((nx, ny));
            }
        }
        // WireRight: only reads from left (unidirectional, signal flows rightward)
        TileType::WireRight => {
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
        }
        TileType::WireUp => {
            // Only reads from down
            if let Some((nx, ny)) = in_bounds(xi, yi + 1) {
                out.push((nx, ny));
            }
        }
        TileType::WireLeft => {
            // Only reads from right
            if let Some((nx, ny)) = in_bounds(xi + 1, yi) {
                out.push((nx, ny));
            }
        }
        TileType::ComponentOutput => {
            // Connects to all 4 neighbors
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi + 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi - 1) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi + 1) {
                out.push((nx, ny));
            }
        }
        TileType::BusInterface => {
            // BusInterface reads from all 4 directions (like Wire)
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi + 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi - 1) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi + 1) {
                out.push((nx, ny));
            }
        }
        TileType::MemoryPort => {
            // MemoryPort reads from left (address), right (data_in), up (write_enable)
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi + 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi - 1) {
                out.push((nx, ny));
            }
        }
        // EPIC 116: CPU Tiles (Generic 4-way)
        TileType::CpuHead | TileType::Register | TileType::Console => {
            for (nx, ny) in neighbors.iter().filter_map(|(nx, ny)| in_bounds(*nx, *ny)) {
                out.push((nx, ny));
            }
        }
        // CPU Building Blocks (Generic 4-way connectivity for net analysis)
        TileType::Decoder3to8
        | TileType::Mux8to1
        | TileType::Mux4to1
        | TileType::Demux1to8
        | TileType::RegEnable
        | TileType::ProgramCounter
        | TileType::Selector => {
            for (nx, ny) in neighbors.iter().filter_map(|(nx, ny)| in_bounds(*nx, *ny)) {
                out.push((nx, ny));
            }
        }
        // Ising Mode Tiles (4-way connectivity for Ising dynamics)
        TileType::IsingNode | TileType::IsingBias => {
            for (nx, ny) in neighbors.iter().filter_map(|(nx, ny)| in_bounds(*nx, *ny)) {
                out.push((nx, ny));
            }
        }
        TileType::ClockDivider => {
            // ClockDivider outputs to all 4 neighbors (like ClockGlobal)
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi + 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi - 1) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi + 1) {
                out.push((nx, ny));
            }
        }
        TileType::Synchronizer => {
            // Synchronizer reads from left, outputs to all
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi + 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi - 1) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi + 1) {
                out.push((nx, ny));
            }
        }
        // Phase 1: Fully Tile-Based CPU - read left+right, output to all
        TileType::AddCarry | TileType::BitSelect => {
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi + 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi - 1) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi + 1) {
                out.push((nx, ny));
            }
        }
        // WireCross: reads from left (horizontal) and up (vertical)
        TileType::WireCross => {
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi - 1) {
                out.push((nx, ny));
            }
        }
        // WireCrossVert: reads from right (horizontal, for WL chains) and up (vertical)
        TileType::WireCrossVert => {
            if let Some((nx, ny)) = in_bounds(xi + 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi - 1) {
                out.push((nx, ny));
            }
        }
        // VBusIn/VBusOut: reads from up only (vertical bus bridge)
        TileType::VBusIn | TileType::VBusOut => {
            if let Some((nx, ny)) = in_bounds(xi, yi - 1) {
                out.push((nx, ny));
            }
        }
        // Via tiles: cross-layer, not captured by 2D net analysis
        TileType::ViaUp | TileType::ViaDown => {}
        TileType::WeightedViaUp | TileType::WeightedViaDown => {}
        TileType::ThresholdViaUp | TileType::ThresholdViaDown => {}
        // SubBorrow: same as Sub (left, right inputs)
        TileType::SubBorrow => {
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi + 1, yi) {
                out.push((nx, ny));
            }
        }
        // Mux16to1: left (data 0-7), right (selector), up (data 8-15)
        TileType::Mux16to1 => {
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi + 1, yi) {
                out.push((nx, ny));
            }
            if let Some((nx, ny)) = in_bounds(xi, yi - 1) {
                out.push((nx, ny));
            }
        }
        // Sprint 127: 64-bit Scaling Primitives
        // Register64: like Register8, takes left input
        TileType::Register64 => {
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
            // Clock neighbors: any adjacent ClockGlobal counts
            for (nx, ny) in neighbors.iter().filter_map(|(nx, ny)| in_bounds(*nx, *ny)) {
                if let Some(t) = sim.tilemap.get_tile(nx, ny) {
                    if matches!(t.meta.tile_type, TileType::ClockGlobal) {
                        out.push((nx, ny));
                    }
                }
            }
        }
        // CarryDetect: comparison operator, takes left and right inputs
        TileType::CarryDetect => {
            for (nx, ny) in [(-1, 0), (1, 0)]
                .iter()
                .filter_map(|(dx, dy)| in_bounds(xi + dx, yi + dy))
            {
                out.push((nx, ny));
            }
        }
        // Decoder6to64: takes left input (6-bit address)
        TileType::Decoder6to64 => {
            if let Some((nx, ny)) = in_bounds(xi - 1, yi) {
                out.push((nx, ny));
            }
        }
    }

    out
}

pub fn analyze_nets(sim: &Simulation) -> NetReport {
    let mut work = NetWork::new(sim);
    work.run()
}

pub fn summarize(report: &NetReport) -> crate::net::net_summary::GlobalSummary {
    crate::net::net_summary::summarize_nets(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::net_summary;
    use crate::simulation::Simulation;
    use crate::tile_meta::TileType;

    fn find_net_by_kind(report: &NetReport, kind: NetKind) -> &NetStats {
        report
            .nets
            .iter()
            .find(|n| n.kind == kind)
            .expect("net not found")
    }

    #[test]
    fn single_wire_net_is_detected() {
        let sim = Simulation::new();
        // Wire line from (1,1) to (3,1) with logic set so wires participate
        sim.set_logic_value(1, 1, 1);
        sim.set_logic_value(2, 1, 1);
        sim.set_logic_value(3, 1, 1);

        let report = analyze_nets(&sim);
        assert_eq!(report.nets.len(), 1);
        let net = &report.nets[0];
        assert_eq!(net.node_count, 3);
        assert_eq!((net.x0, net.y0, net.x1, net.y1), (1, 1, 3, 1));
        assert!(!net.is_floating);
        assert!(net.driver_count > 0);
        assert!(net.fanout_max > 0);
    }

    #[test]
    fn clock_net_detected() {
        let mut sim = Simulation::new();
        sim.set_tile(5, 5, TileType::ClockGlobal);
        sim.set_tile(6, 5, TileType::Register8);
        let report = analyze_nets(&sim);
        let clock_net = find_net_by_kind(&report, NetKind::Clock);
        assert!(clock_net.has_clock);
        assert_eq!(clock_net.kind, NetKind::Clock);
    }

    #[test]
    fn floating_vs_driven() {
        let mut sim = Simulation::new();
        // Driven net: set logic on a wire cluster
        sim.set_logic_value(1, 1, 1);
        sim.set_logic_value(2, 1, 1);
        // Floating net: isolated gate with no driver
        sim.set_tile(10, 10, TileType::Or);
        let report = analyze_nets(&sim);
        assert_eq!(report.nets.len(), 2);
        let driven = report.nets.iter().find(|n| n.driver_count > 0).unwrap();
        let floating = report.nets.iter().find(|n| n.is_floating).unwrap();
        assert!(driven.driver_count > 0);
        assert!(floating.is_floating);
    }

    #[test]
    fn region_consistency() {
        let mut sim = Simulation::new();
        sim.assign_region_rect(7, 0, 0, 2, 1).expect("assign ok");
        sim.set_logic_value(0, 0, 1);
        sim.set_logic_value(1, 0, 1);

        let report = analyze_nets(&sim);
        assert_eq!(report.nets.len(), 1);
        let net = &report.nets[0];
        assert_eq!(net.region, Some(7));

        // Mixed region: span across region boundary
        let mut sim2 = Simulation::new();
        sim2.assign_region_rect(3, 0, 0, 1, 1).expect("assign ok");
        sim2.set_logic_value(0, 0, 1);
        sim2.set_logic_value(1, 0, 1);
        let report2 = analyze_nets(&sim2);
        let net2 = &report2.nets[0];
        assert_eq!(net2.region, None);
        assert!(net2.is_mixed_region);
    }

    #[test]
    fn determinism_and_no_mutation() {
        let sim = Simulation::new();
        sim.set_logic_value(2, 2, 1);
        sim.set_logic_value(3, 2, 1);

        let report1 = analyze_nets(&sim);
        let report2 = analyze_nets(&sim);
        assert_eq!(report1, report2);

        // Ensure sim unchanged
        assert_eq!(sim.region_at(0, 0), Some(0));
        let tile = sim.tilemap.get_tile(2, 2).unwrap();
        assert_eq!(tile.logic.load(std::sync::atomic::Ordering::Relaxed), 1);

        // Summaries deterministic too
        let summary1 = net_summary::summarize(&report1);
        let summary2 = net_summary::summarize(&report2);
        assert_eq!(summary1, summary2);
    }
}
