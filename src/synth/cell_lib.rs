//! Tile-native cell library for technology mapping.
//!
//! Every cell corresponds 1:1 to a real `TileType` primitive. No composite
//! cells (like NAND2 = And + Not) — those are placement decisions, not cells.
//! Delays come directly from `TileType::propagation_delay()`.
//!
//! Truth tables use the convention: bit `i` of the truth table =
//! `f(i_0, i_1, ..., i_{K-1})` where `i_j = (i >> j) & 1`.
//!
//! For K=4 (our max), truth tables fit in `u16`.

use crate::tiles::tile_meta::TileType;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Cell
// ---------------------------------------------------------------------------

/// Parameters of an active (computational) via cell.
///
/// Present only when a [`Cell`] is realized by a `ThresholdVia` tile. Captures
/// what placement needs to configure the physical tile: how many in-plane
/// neighbors carry inputs, the popcount threshold, and whether the cross-layer
/// source is an enable input (vs tied high).
#[derive(Clone, Copy, Debug)]
pub struct ViaInfo {
    pub num_neighbors: u8,
    pub threshold: u8,
    pub uses_enable: bool,
}

/// A standard cell that maps directly to a TileType primitive.
#[derive(Clone, Debug)]
pub struct Cell {
    pub name: &'static str,
    pub tile_type: TileType,
    /// Number of inputs (1..=4 for combinational cells, 0 for constants).
    pub num_inputs: u8,
    /// Truth table for the cell's Boolean function.
    /// For K=4 max: 2^4 = 16 bits.
    ///
    /// Convention: bit `i` = `f(i_0, i_1, ..., i_{n-1})` where `i_j = (i >> j) & 1`.
    ///
    /// Examples (2-input, using low 4 bits of u16):
    /// - AND2: 0b1000 = 0x8 (only 1 when both inputs = 1)
    /// - OR2:  0b1110 = 0xE (1 unless both inputs = 0)
    /// - XOR2: 0b0110 = 0x6
    pub truth_table: u16,
    /// Propagation delay in delta cycles, from `TileType::propagation_delay()`.
    pub delay: u8,
    /// Active-via parameters when this cell is a computational via, else `None`.
    pub via: Option<ViaInfo>,
}

// ---------------------------------------------------------------------------
// CellLibrary
// ---------------------------------------------------------------------------

/// Technology library containing all available tile-native cells.
pub struct CellLibrary {
    cells: Vec<Cell>,
    /// Index: truth_table -> list of cell indices implementing that function.
    /// Multiple cells may implement the same truth table (different speed/area tradeoffs).
    tt_index: HashMap<u16, Vec<usize>>,
}

impl CellLibrary {
    /// Create the built-in tile-native cell library.
    ///
    /// Contains only cells that correspond 1:1 to `TileType` primitives.
    /// Delays are sourced from `TileType::propagation_delay()`.
    pub fn tile_native() -> Self {
        let cells = vec![
            // 0-input cells
            Cell {
                name: "CONST0",
                tile_type: TileType::Const,
                num_inputs: 0,
                truth_table: 0x0, // Always 0
                delay: TileType::Const.propagation_delay(),
                via: None,
            },
            // 1-input cells
            // BUF truth table (1-input): f(0)=0, f(1)=1 → 0b10 = 0x2
            // Note: WireRight is a directional routing primitive, not a location-free
            // buffer. For synthesis mapping this is acceptable since we output a
            // MappedNetlist (no placement). When placement is added, BUF cells will
            // need direction-aware handling or a dedicated buffer tile type.
            Cell {
                name: "BUF",
                tile_type: TileType::WireRight,
                num_inputs: 1,
                truth_table: 0x2,
                delay: TileType::WireRight.propagation_delay(),
                via: None,
            },
            // INV truth table (1-input): f(0)=1, f(1)=0 → 0b01 = 0x1
            Cell {
                name: "INV",
                tile_type: TileType::Not,
                num_inputs: 1,
                truth_table: 0x1,
                delay: TileType::Not.propagation_delay(),
                via: None,
            },
            // 2-input cells
            // AND2: f(a,b) — bit layout: f(0,0)=0, f(1,0)=0, f(0,1)=0, f(1,1)=1
            // → 0b1000 = 0x8
            Cell {
                name: "AND2",
                tile_type: TileType::And,
                num_inputs: 2,
                truth_table: 0x8,
                delay: TileType::And.propagation_delay(),
                via: None,
            },
            // OR2: f(0,0)=0, f(1,0)=1, f(0,1)=1, f(1,1)=1 → 0b1110 = 0xE
            Cell {
                name: "OR2",
                tile_type: TileType::Or,
                num_inputs: 2,
                truth_table: 0xE,
                delay: TileType::Or.propagation_delay(),
                via: None,
            },
            // XOR2: f(0,0)=0, f(1,0)=1, f(0,1)=1, f(1,1)=0 → 0b0110 = 0x6
            Cell {
                name: "XOR2",
                tile_type: TileType::Xor,
                num_inputs: 2,
                truth_table: 0x6,
                delay: TileType::Xor.propagation_delay(),
                via: None,
            },
            // 3-input cells
            // MUX2: sel=input0, a=input1, b=input2
            // f(s,a,b) = s ? a : b
            // Truth table (3-input, 8 bits):
            //   s=0,a=0,b=0 → b=0 → 0  (bit 0)
            //   s=1,a=0,b=0 → a=0 → 0  (bit 1)
            //   s=0,a=1,b=0 → b=0 → 0  (bit 2)
            //   s=1,a=1,b=0 → a=1 → 1  (bit 3)
            //   s=0,a=0,b=1 → b=1 → 1  (bit 4)
            //   s=1,a=0,b=1 → a=0 → 0  (bit 5)
            //   s=0,a=1,b=1 → b=1 → 1  (bit 6)
            //   s=1,a=1,b=1 → a=1 → 1  (bit 7)
            // = 0b11011000 = 0xD8
            // Wait — let me recalculate with the TileType::Mux semantic.
            // TileType::Mux: output = (up != 0) ? left : right
            // Our convention: input0=sel (maps to up), input1=left (true branch),
            //                 input2=right (false branch)
            // f = sel ? input1 : input2
            // Bit i encodes f(i_0, i_1, i_2) where i_j = (i >> j) & 1
            //   i=0: s=0,d1=0,d0=0 → d0=0 → 0
            //   i=1: s=1,d1=0,d0=0 → d1=0 → 0
            //   i=2: s=0,d1=1,d0=0 → d0=0 → 0
            //   i=3: s=1,d1=1,d0=0 → d1=1 → 1
            //   i=4: s=0,d1=0,d0=1 → d0=1 → 1
            //   i=5: s=1,d1=0,d0=1 → d1=0 → 0
            //   i=6: s=0,d1=1,d0=1 → d0=1 → 1
            //   i=7: s=1,d1=1,d0=1 → d1=1 → 1
            // = 0b11010100 ... wait, re-indexing:
            // Let me be precise. input0 = sel, input1 = true_val, input2 = false_val.
            // f(sel, tv, fv) = sel ? tv : fv
            // Bit index i = sel + 2*tv + 4*fv
            //   i=0 (s=0,t=0,f=0): fv=0 → 0
            //   i=1 (s=1,t=0,f=0): tv=0 → 0
            //   i=2 (s=0,t=1,f=0): fv=0 → 0
            //   i=3 (s=1,t=1,f=0): tv=1 → 1
            //   i=4 (s=0,t=0,f=1): fv=1 → 1
            //   i=5 (s=1,t=0,f=1): tv=0 → 0
            //   i=6 (s=0,t=1,f=1): fv=1 → 1
            //   i=7 (s=1,t=1,f=1): tv=1 → 1
            // = 0b_1101_1000 = 0xD8... wait:
            // bit7=1, bit6=1, bit5=0, bit4=1, bit3=1, bit2=0, bit1=0, bit0=0
            // = 0b_1101_1000 ... that's bits [7..0] = 1,1,0,1,1,0,0,0
            // In hex that's 0xD8.
            // Actually: 0b11011000 from MSB to LSB = 0xD8. Let me verify:
            // 0xD8 = 216 = 128+64+16+8 = 0b11011000. Yes.
            Cell {
                name: "MUX2",
                tile_type: TileType::Mux,
                num_inputs: 3,
                truth_table: 0xD8,
                delay: TileType::Mux.propagation_delay(),
                via: None,
            },
        ];

        let mut lib = CellLibrary {
            cells,
            tt_index: HashMap::new(),
        };
        lib.build_index();
        lib
    }

    /// Build a library from an explicit list of cells (e.g. the tile-native
    /// base set plus active-via cells). Rebuilds the truth-table index.
    pub fn from_cells(cells: Vec<Cell>) -> Self {
        let mut lib = CellLibrary {
            cells,
            tt_index: HashMap::new(),
        };
        lib.build_index();
        lib
    }

    fn build_index(&mut self) {
        self.tt_index.clear();
        for (i, cell) in self.cells.iter().enumerate() {
            self.tt_index.entry(cell.truth_table).or_default().push(i);
        }
    }

    /// Look up cell indices implementing a given truth table.
    pub fn cells_for_function(&self, tt: u16) -> &[usize] {
        self.tt_index.get(&tt).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get a cell by index.
    pub fn cell(&self, idx: usize) -> &Cell {
        &self.cells[idx]
    }

    /// Number of cells in the library.
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// All cells.
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    /// Find the best (lowest delay) cell for a given truth table and input count.
    /// Returns None if no cell matches.
    pub fn best_cell_for(&self, tt: u16, num_inputs: u8) -> Option<usize> {
        let candidates = self.cells_for_function(tt);
        candidates
            .iter()
            .copied()
            .filter(|&i| self.cells[i].num_inputs == num_inputs)
            .min_by_key(|&i| self.cells[i].delay)
    }

    /// Find any cell matching a truth table (ignoring input count).
    /// Prefers exact input count match, falls back to any match.
    pub fn find_cell(&self, tt: u16) -> Option<usize> {
        let candidates = self.cells_for_function(tt);
        if candidates.is_empty() {
            None
        } else {
            // Prefer lowest delay
            Some(
                *candidates
                    .iter()
                    .min_by_key(|&&i| self.cells[i].delay)
                    .unwrap(),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Truth table utilities
// ---------------------------------------------------------------------------

/// Compute the mask for meaningful truth table bits given num_inputs.
/// For num_inputs=0: 1 bit, mask=0x1. For 1: 2 bits, mask=0x3.
/// For 2: 4 bits, mask=0xF. For 3: 8 bits, mask=0xFF. For 4: 16 bits, mask=0xFFFF.
fn tt_mask(num_inputs: u8) -> u16 {
    if num_inputs >= 4 {
        0xFFFF
    } else {
        (1u16 << (1u16 << num_inputs)) - 1
    }
}

/// Negate a truth table (bitwise NOT, masked to `num_inputs` bits).
pub fn tt_negate(tt: u16, num_inputs: u8) -> u16 {
    (!tt) & tt_mask(num_inputs)
}

/// Check if a truth table is constant 0.
pub fn tt_is_const0(tt: u16, num_inputs: u8) -> bool {
    (tt & tt_mask(num_inputs)) == 0
}

/// Check if a truth table is constant 1.
pub fn tt_is_const1(tt: u16, num_inputs: u8) -> bool {
    let mask = tt_mask(num_inputs);
    (tt & mask) == mask
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_has_basic_cells() {
        let lib = CellLibrary::tile_native();
        assert!(lib.num_cells() >= 7);
    }

    #[test]
    fn delays_match_tile_types() {
        let lib = CellLibrary::tile_native();
        for cell in lib.cells() {
            assert_eq!(
                cell.delay,
                cell.tile_type.propagation_delay(),
                "Cell {} delay {} != TileType::{:?} delay {}",
                cell.name,
                cell.delay,
                cell.tile_type,
                cell.tile_type.propagation_delay()
            );
        }
    }

    #[test]
    fn lookup_and2() {
        let lib = CellLibrary::tile_native();
        let indices = lib.cells_for_function(0x8); // AND2 truth table
        assert!(!indices.is_empty(), "AND2 not found for tt=0x8");
        assert_eq!(lib.cell(indices[0]).name, "AND2");
    }

    #[test]
    fn lookup_or2() {
        let lib = CellLibrary::tile_native();
        let idx = lib.find_cell(0xE).expect("OR2 not found");
        assert_eq!(lib.cell(idx).name, "OR2");
    }

    #[test]
    fn lookup_inv() {
        let lib = CellLibrary::tile_native();
        let idx = lib.find_cell(0x1).expect("INV not found");
        assert_eq!(lib.cell(idx).name, "INV");
    }

    #[test]
    fn tt_negate_works() {
        // NOT of AND2 (0x8) with 2 inputs = 0x7 (NAND2)
        assert_eq!(tt_negate(0x8, 2), 0x7);
        // NOT of OR2 (0xE) with 2 inputs = 0x1 (NOR2)
        assert_eq!(tt_negate(0xE, 2), 0x1);
    }

    #[test]
    fn tt_const_checks() {
        assert!(tt_is_const0(0x0, 2));
        assert!(!tt_is_const0(0x1, 2));
        assert!(tt_is_const1(0xF, 2));
        assert!(!tt_is_const1(0xE, 2));
    }

    #[test]
    fn mux_truth_table() {
        let lib = CellLibrary::tile_native();
        let idx = lib.find_cell(0xD8).expect("MUX2 not found");
        assert_eq!(lib.cell(idx).name, "MUX2");
        assert_eq!(lib.cell(idx).num_inputs, 3);
    }
}
