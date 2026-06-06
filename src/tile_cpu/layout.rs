//! Tile Layout - Physical placement of CPU components
//!
//! This module handles the spatial arrangement of tiles on the grid,
//! including component placement and wire routing.

use crate::tile_meta::TileType;

/// A complete tile layout for a CPU
#[derive(Debug, Clone)]
pub struct TileLayout {
    /// Width of the layout in tiles
    pub width: usize,
    /// Height of the layout in tiles
    pub height: usize,
    /// Placed components
    pub components: Vec<PlacedComponent>,
    /// Wire routes
    pub wires: Vec<WireRoute>,
    /// Total tile count
    pub total_tiles: usize,
}

/// A component placed on the grid
#[derive(Debug, Clone)]
pub struct PlacedComponent {
    /// Component name
    pub name: String,
    /// Top-left corner position
    pub x: usize,
    pub y: usize,
    /// Component width in tiles
    pub width: usize,
    /// Component height in tiles
    pub height: usize,
    /// Tile types used (row-major order)
    pub tiles: Vec<(usize, usize, TileType)>,
}

/// A wire route connecting components
#[derive(Debug, Clone)]
pub struct WireRoute {
    /// Source component and port
    pub from: (String, String),
    /// Destination component and port
    pub to: (String, String),
    /// Wire tiles (positions)
    pub path: Vec<(usize, usize, TileType)>,
    /// Total delay through this route
    pub delay: u32,
}

impl TileLayout {
    /// Create an empty layout
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            components: Vec::new(),
            wires: Vec::new(),
            total_tiles: 0,
        }
    }

    /// Add a component to the layout
    pub fn add_component(&mut self, component: PlacedComponent) {
        self.total_tiles += component.tiles.len();
        self.components.push(component);
    }

    /// Add a wire route
    pub fn add_wire(&mut self, wire: WireRoute) {
        self.total_tiles += wire.path.len();
        self.wires.push(wire);
    }

    /// Get all tiles as a flat list for rendering
    pub fn all_tiles(&self) -> Vec<(usize, usize, TileType)> {
        let mut tiles = Vec::with_capacity(self.total_tiles);

        for component in &self.components {
            tiles.extend(component.tiles.iter().cloned());
        }

        for wire in &self.wires {
            tiles.extend(wire.path.iter().cloned());
        }

        tiles
    }

    /// Calculate layout utilization
    pub fn utilization(&self) -> f64 {
        let total_area = self.width * self.height;
        if total_area > 0 {
            self.total_tiles as f64 / total_area as f64
        } else {
            0.0
        }
    }
}

impl PlacedComponent {
    /// Create a new placed component
    pub fn new(name: &str, x: usize, y: usize) -> Self {
        Self {
            name: name.into(),
            x,
            y,
            width: 0,
            height: 0,
            tiles: Vec::new(),
        }
    }

    /// Add a tile to the component
    pub fn add_tile(&mut self, rel_x: usize, rel_y: usize, tile_type: TileType) {
        let abs_x = self.x + rel_x;
        let abs_y = self.y + rel_y;
        self.tiles.push((abs_x, abs_y, tile_type));

        // Update bounding box
        self.width = self.width.max(rel_x + 1);
        self.height = self.height.max(rel_y + 1);
    }
}

// =============================================================================
// Standard Component Layouts
// =============================================================================

impl PlacedComponent {
    /// Create an 8-bit register component
    pub fn register_8bit(name: &str, x: usize, y: usize) -> Self {
        let mut comp = Self::new(name, x, y);
        // Single RegEnable tile handles 8-bit value
        comp.add_tile(0, 0, TileType::RegEnable);
        comp
    }

    /// Create a program counter component
    pub fn program_counter(x: usize, y: usize) -> Self {
        let mut comp = Self::new("PC", x, y);
        comp.add_tile(0, 0, TileType::ProgramCounter);
        comp
    }

    /// Create an ALU component
    pub fn alu(x: usize, y: usize) -> Self {
        let mut comp = Self::new("ALU", x, y);

        // Row of ALU operation tiles
        comp.add_tile(0, 0, TileType::Add);
        comp.add_tile(2, 0, TileType::Sub);
        comp.add_tile(4, 0, TileType::And);
        comp.add_tile(6, 0, TileType::Or);
        comp.add_tile(8, 0, TileType::Xor);

        // Output mux
        comp.add_tile(4, 1, TileType::Mux8to1);

        comp
    }

    /// Create a ROM block
    pub fn rom(x: usize, y: usize, size: usize) -> Self {
        let mut comp = Self::new("ROM", x, y);

        for addr in 0..size {
            let rel_x = addr % 8;
            let rel_y = addr / 8;
            comp.add_tile(rel_x, rel_y, TileType::Const);
        }

        comp
    }

    /// Create a RAM block
    pub fn ram(x: usize, y: usize, size: usize) -> Self {
        let mut comp = Self::new("RAM", x, y);

        for addr in 0..size {
            let rel_x = addr % 8;
            let rel_y = addr / 8;
            comp.add_tile(rel_x, rel_y, TileType::Ram);
        }

        comp
    }

    /// Create a register file (4 registers)
    pub fn register_file(x: usize, y: usize) -> Self {
        let mut comp = Self::new("RegFile", x, y);

        for reg in 0..4 {
            comp.add_tile(0, reg, TileType::RegEnable);
        }

        // Address decoder
        comp.add_tile(2, 0, TileType::Decoder3to8);

        comp
    }
}

// =============================================================================
// Wire Routing
// =============================================================================

impl WireRoute {
    /// Create a horizontal wire
    pub fn horizontal(
        from: (&str, &str),
        to: (&str, &str),
        y: usize,
        x_start: usize,
        x_end: usize,
    ) -> Self {
        let mut path = Vec::new();
        for x in x_start..=x_end {
            path.push((x, y, TileType::WireH));
        }
        let delay = path.len() as u32;

        Self {
            from: (from.0.into(), from.1.into()),
            to: (to.0.into(), to.1.into()),
            path,
            delay,
        }
    }

    /// Create a vertical wire
    pub fn vertical(
        from: (&str, &str),
        to: (&str, &str),
        x: usize,
        y_start: usize,
        y_end: usize,
    ) -> Self {
        let mut path = Vec::new();
        for y in y_start..=y_end {
            path.push((x, y, TileType::WireV));
        }
        let delay = path.len() as u32;

        Self {
            from: (from.0.into(), from.1.into()),
            to: (to.0.into(), to.1.into()),
            path,
            delay,
        }
    }

    /// Create an L-shaped wire (horizontal then vertical)
    pub fn l_route_hv(
        from: (&str, &str),
        to: (&str, &str),
        start: (usize, usize),
        corner: (usize, usize),
        end: (usize, usize),
    ) -> Self {
        let mut path = Vec::new();

        // Horizontal segment
        let (x_min, x_max) = if start.0 < corner.0 {
            (start.0, corner.0)
        } else {
            (corner.0, start.0)
        };
        for x in x_min..=x_max {
            path.push((x, start.1, TileType::WireH));
        }

        // Vertical segment
        let (y_min, y_max) = if corner.1 < end.1 {
            (corner.1, end.1)
        } else {
            (end.1, corner.1)
        };
        for y in y_min..=y_max {
            if y != corner.1 {
                // Skip corner, already placed
                path.push((corner.0, y, TileType::WireV));
            }
        }

        // Corner crossing
        path.push((corner.0, corner.1, TileType::Cross));

        let delay = path.len() as u32;

        Self {
            from: (from.0.into(), from.1.into()),
            to: (to.0.into(), to.1.into()),
            path,
            delay,
        }
    }
}

// =============================================================================
// Layout Generation
// =============================================================================

/// Generate a complete CPU layout
#[allow(dead_code)]
pub fn generate_cpu_layout(origin: (usize, usize), rom_size: usize, ram_size: usize) -> TileLayout {
    let (ox, oy) = origin;

    let mut layout = TileLayout::new(24, 48);

    // Program Counter
    layout.add_component(PlacedComponent::program_counter(ox + 4, oy));

    // Register File
    layout.add_component(PlacedComponent::register_file(ox + 4, oy + 4));

    // ALU
    layout.add_component(PlacedComponent::alu(ox + 2, oy + 10));

    // ROM
    layout.add_component(PlacedComponent::rom(ox, oy + 16, rom_size));

    // RAM
    layout.add_component(PlacedComponent::ram(ox, oy + 32, ram_size));

    // Wire routes
    layout.add_wire(WireRoute::vertical(
        ("PC", "out"),
        ("ROM", "addr"),
        ox + 4,
        oy,
        oy + 16,
    ));

    layout.add_wire(WireRoute::vertical(
        ("RegFile", "out"),
        ("ALU", "a"),
        ox + 4,
        oy + 7,
        oy + 10,
    ));

    layout.add_wire(WireRoute::horizontal(
        ("ALU", "out"),
        ("RegFile", "in"),
        oy + 12,
        ox + 2,
        ox + 12,
    ));

    layout
}
