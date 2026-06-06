use crate::simulation::Simulation;
use crate::tile_meta::TileType;
use crate::tilemap::{HEIGHT, WIDTH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitKind {
    WireBus,
    LatchStrip,
    RingOscillator,
    RegisterFile8,
    // BitCounter4, // optional later
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortRole {
    Input,
    Output,
    Bidirectional,
    Clock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CircuitPort {
    pub name: &'static str,
    pub role: PortRole,
    pub local_x: u16,
    pub local_y: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileSpec {
    pub x: u16,
    pub y: u16,
    pub ty: TileType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicSpec {
    pub x: u16,
    pub y: u16,
    pub value: u64,
}

#[derive(Debug, Clone)]
pub struct CircuitTemplate {
    pub kind: CircuitKind,
    pub width: u16,
    pub height: u16,
    pub tiles: &'static [TileSpec],
    pub logic: &'static [LogicSpec],
    pub ports: &'static [CircuitPort],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitError {
    UnknownCircuit,
    OutOfBounds { x: u32, y: u32 },
    Collision { x: u32, y: u32 },
    InvalidParams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementOptions {
    pub allow_overwrite: bool,
}

impl Default for PlacementOptions {
    fn default() -> Self {
        Self {
            allow_overwrite: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlacementSummary {
    pub kind: CircuitKind,
    pub origin_x: u32,
    pub origin_y: u32,
    pub width: u16,
    pub height: u16,
    pub ports_world: Vec<(&'static str, PortRole, u32, u32)>,
}

fn is_occupied(tt: TileType) -> bool {
    !matches!(tt, TileType::Wire)
}

// Templates
const WIRE_BUS_TILES: &[TileSpec] = &[
    TileSpec {
        x: 0,
        y: 0,
        ty: TileType::Wire,
    },
    TileSpec {
        x: 1,
        y: 0,
        ty: TileType::Wire,
    },
    TileSpec {
        x: 2,
        y: 0,
        ty: TileType::Wire,
    },
    TileSpec {
        x: 3,
        y: 0,
        ty: TileType::Wire,
    },
];
const WIRE_BUS_LOGIC: &[LogicSpec] = &[];
const WIRE_BUS_PORTS: &[CircuitPort] = &[
    CircuitPort {
        name: "in",
        role: PortRole::Input,
        local_x: 0,
        local_y: 0,
    },
    CircuitPort {
        name: "out",
        role: PortRole::Output,
        local_x: 3,
        local_y: 0,
    },
];
const WIRE_BUS_TEMPLATE: CircuitTemplate = CircuitTemplate {
    kind: CircuitKind::WireBus,
    width: 4,
    height: 1,
    tiles: WIRE_BUS_TILES,
    logic: WIRE_BUS_LOGIC,
    ports: WIRE_BUS_PORTS,
};

// Latch strip: 4 latches in a row with wires at ends
const LATCH_STRIP_TILES: &[TileSpec] = &[
    TileSpec {
        x: 0,
        y: 0,
        ty: TileType::Wire,
    },
    TileSpec {
        x: 1,
        y: 0,
        ty: TileType::Latch,
    },
    TileSpec {
        x: 2,
        y: 0,
        ty: TileType::Latch,
    },
    TileSpec {
        x: 3,
        y: 0,
        ty: TileType::Latch,
    },
    TileSpec {
        x: 4,
        y: 0,
        ty: TileType::Latch,
    },
    TileSpec {
        x: 5,
        y: 0,
        ty: TileType::Wire,
    },
];
const LATCH_STRIP_LOGIC: &[LogicSpec] = &[];
const LATCH_STRIP_PORTS: &[CircuitPort] = &[
    CircuitPort {
        name: "data_in",
        role: PortRole::Input,
        local_x: 0,
        local_y: 0,
    },
    CircuitPort {
        name: "data_out",
        role: PortRole::Output,
        local_x: 5,
        local_y: 0,
    },
    CircuitPort {
        name: "clock",
        role: PortRole::Clock,
        local_x: 0,
        local_y: 0,
    },
];
const LATCH_STRIP_TEMPLATE: CircuitTemplate = CircuitTemplate {
    kind: CircuitKind::LatchStrip,
    width: 6,
    height: 1,
    tiles: LATCH_STRIP_TILES,
    logic: LATCH_STRIP_LOGIC,
    ports: LATCH_STRIP_PORTS,
};

// Ring oscillator: 3x3 ring with one NOT
const RING_TILES: &[TileSpec] = &[
    TileSpec {
        x: 0,
        y: 1,
        ty: TileType::Wire,
    },
    TileSpec {
        x: 1,
        y: 1,
        ty: TileType::Not,
    },
    TileSpec {
        x: 2,
        y: 1,
        ty: TileType::Wire,
    },
    TileSpec {
        x: 1,
        y: 0,
        ty: TileType::Wire,
    },
    TileSpec {
        x: 1,
        y: 2,
        ty: TileType::Wire,
    },
];
const RING_LOGIC: &[LogicSpec] = &[];
const RING_PORTS: &[CircuitPort] = &[CircuitPort {
    name: "tap_out",
    role: PortRole::Output,
    local_x: 2,
    local_y: 1,
}];
const RING_TEMPLATE: CircuitTemplate = CircuitTemplate {
    kind: CircuitKind::RingOscillator,
    width: 3,
    height: 3,
    tiles: RING_TILES,
    logic: RING_LOGIC,
    ports: RING_PORTS,
};

// Register File: 8 registers x 8 bits using packed storage
// Layout (5 wide x 3 tall):
//   Row 0: Write path - address decoder
//   Row 1: Storage - packed 64-bit RAM (8 bytes = 8 registers)
//   Row 2: Read path - multiplexer
//
// The RAM tile stores all 8 registers packed:
//   R0 = bits 0-7, R1 = bits 8-15, ..., R7 = bits 56-63
//
// Write: Use external logic to create (old & ~mask) | (new << shift)
// Read: Mux8to1 extracts the selected byte
const REGFILE8_TILES: &[TileSpec] = &[
    // Row 0: Write address decoder
    TileSpec {
        x: 0,
        y: 0,
        ty: TileType::Wire,
    }, // write addr input
    TileSpec {
        x: 1,
        y: 0,
        ty: TileType::Decoder3to8,
    }, // decode to one-hot
    TileSpec {
        x: 2,
        y: 0,
        ty: TileType::Wire,
    }, // decoder output
    // Row 1: Storage
    TileSpec {
        x: 0,
        y: 1,
        ty: TileType::Wire,
    }, // write data input
    TileSpec {
        x: 1,
        y: 1,
        ty: TileType::Ram,
    }, // packed register storage
    TileSpec {
        x: 2,
        y: 1,
        ty: TileType::Wire,
    }, // write enable
    // Row 2: Read multiplexer
    TileSpec {
        x: 0,
        y: 2,
        ty: TileType::Wire,
    }, // read addr input
    TileSpec {
        x: 1,
        y: 2,
        ty: TileType::Mux8to1,
    }, // select register to read
    TileSpec {
        x: 2,
        y: 2,
        ty: TileType::Wire,
    }, // read output
];
const REGFILE8_LOGIC: &[LogicSpec] = &[];
const REGFILE8_PORTS: &[CircuitPort] = &[
    CircuitPort {
        name: "wr_addr",
        role: PortRole::Input,
        local_x: 0,
        local_y: 0,
    },
    CircuitPort {
        name: "wr_data",
        role: PortRole::Input,
        local_x: 0,
        local_y: 1,
    },
    CircuitPort {
        name: "wr_enable",
        role: PortRole::Input,
        local_x: 2,
        local_y: 1,
    },
    CircuitPort {
        name: "rd_addr",
        role: PortRole::Input,
        local_x: 0,
        local_y: 2,
    },
    CircuitPort {
        name: "rd_data",
        role: PortRole::Output,
        local_x: 2,
        local_y: 2,
    },
];
const REGFILE8_TEMPLATE: CircuitTemplate = CircuitTemplate {
    kind: CircuitKind::RegisterFile8,
    width: 3,
    height: 3,
    tiles: REGFILE8_TILES,
    logic: REGFILE8_LOGIC,
    ports: REGFILE8_PORTS,
};

pub fn list_templates() -> &'static [CircuitTemplate] {
    &[
        WIRE_BUS_TEMPLATE,
        LATCH_STRIP_TEMPLATE,
        RING_TEMPLATE,
        REGFILE8_TEMPLATE,
    ]
}

fn find_template(kind: CircuitKind) -> Option<&'static CircuitTemplate> {
    list_templates().iter().find(|t| t.kind == kind)
}

pub fn world_ports(
    template: &CircuitTemplate,
    origin_x: u32,
    origin_y: u32,
) -> Vec<(&'static str, PortRole, u32, u32)> {
    template
        .ports
        .iter()
        .map(|p| {
            (
                p.name,
                p.role,
                origin_x + p.local_x as u32,
                origin_y + p.local_y as u32,
            )
        })
        .collect()
}

pub fn place_circuit(
    sim: &mut Simulation,
    kind: CircuitKind,
    origin_x: u32,
    origin_y: u32,
    opts: &PlacementOptions,
) -> Result<PlacementSummary, CircuitError> {
    let t = find_template(kind).ok_or(CircuitError::UnknownCircuit)?;
    // Dry-run: bounds and collision check
    for spec in t.tiles {
        let wx = origin_x + spec.x as u32;
        let wy = origin_y + spec.y as u32;
        if wx as usize >= WIDTH || wy as usize >= HEIGHT {
            return Err(CircuitError::OutOfBounds { x: wx, y: wy });
        }
        if let Some(tile) = sim.tilemap.get_tile(wx as usize, wy as usize) {
            let occupied = is_occupied(tile.meta.tile_type);
            if occupied && !opts.allow_overwrite {
                return Err(CircuitError::Collision { x: wx, y: wy });
            }
        }
    }

    // Commit: apply tiles and logic deterministically
    for spec in t.tiles {
        let wx = (origin_x + spec.x as u32) as usize;
        let wy = (origin_y + spec.y as u32) as usize;
        sim.set_tile(wx, wy, spec.ty);
    }
    for ls in t.logic {
        let wx = (origin_x + ls.x as u32) as usize;
        let wy = (origin_y + ls.y as u32) as usize;
        let _ = sim.set_logic_value(wx, wy, ls.value);
    }

    Ok(PlacementSummary {
        kind,
        origin_x,
        origin_y,
        width: t.width,
        height: t.height,
        ports_world: world_ports(t, origin_x, origin_y),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::Simulation;

    #[test]
    fn templates_list_nonempty() {
        assert!(!list_templates().is_empty());
    }

    #[test]
    fn place_wire_bus_ok() {
        let mut sim = Simulation::new();
        let sum = place_circuit(
            &mut sim,
            CircuitKind::WireBus,
            10,
            10,
            &PlacementOptions::default(),
        )
        .unwrap();
        assert_eq!(sum.width, 4);
        // Check a tile type
        assert!(matches!(
            sim.tilemap.get_tile(10, 10).unwrap().meta.tile_type,
            TileType::Wire
        ));
    }

    #[test]
    fn collision_detects_non_wire() {
        let mut sim = Simulation::new();
        // Pre-seed a gate at target
        sim.set_tile(12, 10, TileType::And);
        let err = place_circuit(
            &mut sim,
            CircuitKind::WireBus,
            10,
            10,
            &PlacementOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, CircuitError::Collision { .. }));
    }

    #[test]
    fn out_of_bounds_detected() {
        let mut sim = Simulation::new();
        let err = place_circuit(
            &mut sim,
            CircuitKind::RingOscillator,
            (WIDTH as u32) - 1,
            (HEIGHT as u32) - 1,
            &PlacementOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, CircuitError::OutOfBounds { .. }));
    }
}
