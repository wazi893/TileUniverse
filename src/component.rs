//! Component Hierarchy System
//!
//! Supports two modes:
//! - Behavioral (Combinational): Black-box Fn(&[u64]) -> Vec<u64>
//! - Structural: Places actual tiles on the grid with unidirectional wire port isolation

use crate::tile_meta::TileType;
use std::cell::Cell;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortDirection {
    Input,
    Output,
}

#[derive(Debug, Clone)]
pub struct PortDef {
    pub name: String,
    pub edge: Edge,
    pub offset: u32,
    pub direction: PortDirection,
}

/// A tile placement relative to component origin (for structural components).
pub struct TilePlacement {
    pub x: u32,
    pub y: u32,
    pub tile_type: TileType,
    pub initial_logic: Option<u64>,
}

/// Implementation variant for a component.
pub enum ComponentImpl {
    /// Pure combinational function: reads input port values, returns output port values.
    /// Input slice ordered by port definition order (inputs only).
    /// Output vec ordered by port definition order (outputs only).
    Combinational(Box<dyn Fn(&[u64]) -> Vec<u64>>),
    /// Structural: places actual tiles on the grid at registration time.
    /// After placement, tiles evaluate normally via the standard simulation loop.
    Structural(Vec<TilePlacement>),
}

/// Definition of a reusable component.
pub struct ComponentDef {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub ports: Vec<PortDef>,
    pub implementation: ComponentImpl,
    /// Propagation delay for behavioral components (default: 5).
    /// Ignored for structural components (individual tiles have their own delays).
    pub propagation_delay: u8,
}

/// A placed instance of a component on the grid.
pub struct ComponentInstance {
    /// Index into Simulation::component_defs
    pub def_idx: usize,
    /// Top-left corner (x, y) on the grid
    pub origin: (usize, usize),
    /// Tile indices of input port tiles (ordered by port def order, inputs only)
    pub input_port_indices: Vec<usize>,
    /// Tile indices of output port tiles (ordered by port def order, outputs only)
    pub output_port_indices: Vec<usize>,
    /// Cached output values from behavioral function (one per output port)
    pub output_cache: Vec<Cell<u64>>,
    /// Whether the output cache is valid (invalidated when any input changes)
    pub cache_valid: Cell<bool>,
}

/// Convert a port definition to grid coordinates.
pub fn port_to_grid_coords(
    origin_x: usize,
    origin_y: usize,
    width: usize,
    height: usize,
    port: &PortDef,
) -> (usize, usize) {
    match port.edge {
        Edge::Left => (origin_x, origin_y + port.offset as usize),
        Edge::Right => (origin_x + width - 1, origin_y + port.offset as usize),
        Edge::Top => (origin_x + port.offset as usize, origin_y),
        Edge::Bottom => (origin_x + port.offset as usize, origin_y + height - 1),
    }
}

/// Get the unidirectional wire type for an input port on the given edge.
/// Signal flows INTO the component from the external neighbor.
pub fn input_wire_type_for_edge(edge: Edge) -> TileType {
    match edge {
        Edge::Left => TileType::WireRight, // reads from left (external), flows right (into component)
        Edge::Right => TileType::WireLeft, // reads from right (external), flows left (into component)
        Edge::Top => TileType::WireDown,   // reads from up (external), flows down (into component)
        Edge::Bottom => TileType::WireUp,  // reads from down (external), flows up (into component)
    }
}

/// Get the unidirectional wire type for an output port on the given edge (structural components).
/// Signal flows OUT of the component to the external neighbor.
pub fn output_wire_type_for_edge(edge: Edge) -> TileType {
    match edge {
        Edge::Left => TileType::WireLeft,   // signal exits leftward
        Edge::Right => TileType::WireRight, // signal exits rightward
        Edge::Top => TileType::WireUp,      // signal exits upward
        Edge::Bottom => TileType::WireDown, // signal exits downward
    }
}
