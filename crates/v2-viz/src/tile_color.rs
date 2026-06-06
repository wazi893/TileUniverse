use bevy::prelude::Color;
use engine::tiles::tile_meta::TileType;

/// Category for grouping tile types in the color legend.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TileCategory {
    Wire,
    Register,
    Arithmetic,
    MuxDecode,
    Const,
    Other,
}

pub fn tile_category(tt: TileType) -> TileCategory {
    use TileType::*;
    match tt {
        Wire | WireH | WireV | WireDown | WireRight | WireUp | WireLeft | Cross | WireCross
        | WireCrossVert | VBusIn | VBusOut | ViaUp | ViaDown | WeightedViaUp | WeightedViaDown
        | ThresholdViaUp | ThresholdViaDown => TileCategory::Wire,

        Register8 | Register64 | Register | RegEnable | Latch | Counter | MemoryPort | Ram => {
            TileCategory::Register
        }

        Add | Sub | Mul | Div | Mod | Shl | Shr | AddCarry | SubBorrow | CarryDetect | And | Or
        | Xor | Not | Neg | Abs | Lt | Gt | Eq | Neq | Lte | Gte => TileCategory::Arithmetic,

        Mux | Mux4to1 | Mux8to1 | Mux16to1 | Decoder3to8 | Decoder6to64 | ProgramCounter
        | ClockGlobal | CpuHead | BitSelect | Zero | Selector | Demux1to8 | BusInterface
        | ClockDivider | Synchronizer => TileCategory::MuxDecode,

        Const => TileCategory::Const,

        _ => TileCategory::Other,
    }
}

/// Base color for a tile category (RGB 0-255).
fn category_base(cat: TileCategory) -> [u8; 3] {
    match cat {
        TileCategory::Wire => [40, 90, 185],       // Blue
        TileCategory::Register => [45, 165, 90],   // Green
        TileCategory::Arithmetic => [190, 70, 55], // Red-brown
        TileCategory::MuxDecode => [200, 145, 45], // Orange
        TileCategory::Const => [95, 95, 95],       // Gray
        TileCategory::Other => [120, 120, 120],    // Light gray
    }
}

/// Returns a Bevy color for a tile given its type and whether its logic value is nonzero.
pub fn tile_bevy_color(tt: TileType, logic_nonzero: bool) -> Color {
    let cat = tile_category(tt);
    let [r, g, b] = category_base(cat);
    let (r, g, b) = if logic_nonzero {
        (
            r.saturating_add(50),
            g.saturating_add(50),
            b.saturating_add(50),
        )
    } else {
        (
            r.saturating_sub(25),
            g.saturating_sub(25),
            b.saturating_sub(25),
        )
    };
    Color::srgb_u8(r, g, b)
}

/// Bright highlight color for tiles that changed this cycle.
pub fn active_highlight_color() -> Color {
    Color::srgba(1.0, 0.6, 0.0, 0.7)
}

/// Category legend entry for UI.
pub fn category_legend() -> &'static [(TileCategory, &'static str, [u8; 3])] {
    &[
        (TileCategory::Wire, "Wire / Via", [40, 90, 185]),
        (TileCategory::Register, "Register / RAM", [45, 165, 90]),
        (TileCategory::Arithmetic, "ALU / Logic", [190, 70, 55]),
        (TileCategory::MuxDecode, "Mux / Decode / PC", [200, 145, 45]),
        (TileCategory::Const, "Const", [95, 95, 95]),
    ]
}
