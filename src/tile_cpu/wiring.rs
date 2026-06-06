//! Datapath Wiring - Pure tile CPU implementation
//!
//! This module creates the tile-based CPU datapath using binary mux trees.
//! With Register8 fixed to capture only at delta 0, feedback loops work correctly.
//!
//! Two modes:
//! - **Hybrid** (`wire_complete_cpu`): Tiles handle fetch, operand selection, ALU,
//!   result selection. Software handles decode/writeback.
//! - **Physical** (`wire_physical_cpu`): Physical decoder (Shr → LUTs → bank select).
//!   Software still handles operand staging, ALU result staging, writeback, branches.
//!
//! Binary mux trees use standard Mux tiles (if up!=0 { left } else { right })
//! with WireDown/WireLeft/WireRight routing between levels. No packed u64s.

use crate::simulation::Simulation;
use crate::tile_cpu::NUM_REGISTERS;
use crate::tile_meta::TileType;

/// Wiring context for building a CPU
pub struct WiringContext<'a> {
    sim: &'a mut Simulation,
    origin: (usize, usize),
    pub grid_width: usize,
    tiles_placed: usize,
    /// Current layer for tile placement (default 0)
    pub current_layer: usize,

    pub pc_idx: usize,
    pub ir_idx: usize,
    pub reg_indices: [usize; NUM_REGISTERS],
    /// ALU operation tiles [Add, Sub, And, Or, Xor, Not, Shl, Shr]
    pub alu_tile_indices: [usize; 8],

    // Operand A mux tree (4:1 binary tree, selects one of 4 registers)
    pub op_a_data_indices: [usize; 4],
    pub op_a_sel0_indices: [usize; 2],
    pub op_a_sel1_idx: usize,
    pub op_a_leaf_indices: [usize; 2],
    pub op_a_root_idx: usize,

    // Operand B mux tree (same structure as A)
    pub op_b_data_indices: [usize; 4],
    pub op_b_sel0_indices: [usize; 2],
    pub op_b_sel1_idx: usize,
    pub op_b_leaf_indices: [usize; 2],
    pub op_b_root_idx: usize,
    pub flag_z_idx: usize,
    pub flag_c_idx: usize,
    pub rom_indices: Vec<usize>,
    pub ram_indices: Vec<usize>,
    // PC circuit
    /// Const tile for next_pc — software writes next PC value here before tick
    pub next_pc_const_idx: usize,
    // Decoder tiles
    /// Const tile for opcode_lo select — software writes (opcode & 7) here
    pub decoder_opcode_lo_idx: usize,
    /// Mux8to1 for ctrl_a lookup (opcodes 0-7)
    pub decoder_ctrl_a_lo_idx: usize,
    /// Mux8to1 for ctrl_a lookup (opcodes 8-F)
    pub decoder_ctrl_a_hi_idx: usize,
    /// Mux8to1 for ctrl_b lookup (opcodes 0-7)
    pub decoder_ctrl_b_lo_idx: usize,
    /// Mux8to1 for ctrl_b lookup (opcodes 8-F)
    pub decoder_ctrl_b_hi_idx: usize,
    // ALU result mux tree (8:1 binary tree)
    pub alu_result_data_indices: [usize; 8],
    pub alu_result_sel0_indices: [usize; 4],
    pub alu_result_sel1_indices: [usize; 2],
    pub alu_result_sel2_idx: usize,
    pub alu_result_leaf_indices: [usize; 4],
    pub alu_result_mid_indices: [usize; 2],
    pub alu_result_root_idx: usize,
    // Register writeback
    /// Const tiles for write-enable per register (above each Mux)
    pub reg_we_indices: [usize; NUM_REGISTERS],
    /// Const tiles for result data per register (left of each Mux)
    pub reg_result_indices: [usize; NUM_REGISTERS],
    /// Mux tiles for writeback selection per register
    pub reg_mux_indices: [usize; NUM_REGISTERS],
    // Flag circuits
    /// Const tile for result value fed to Zero tile
    pub flag_z_result_idx: usize,
    /// Const tile for update_flags signal above Mux_Z
    pub flag_z_update_idx: usize,
    /// Zero tile that computes z_raw from result
    pub flag_z_zero_idx: usize,
    /// Mux tile for Z flag writeback
    pub flag_z_mux_idx: usize,
    /// Const tile for carry value fed to Mux_C
    pub flag_c_carry_idx: usize,
    /// Const tile for update_flags signal above Mux_C
    pub flag_c_update_idx: usize,
    /// Mux tile for C flag writeback
    pub flag_c_mux_idx: usize,
}

impl<'a> WiringContext<'a> {
    pub fn new(sim: &'a mut Simulation, origin: (usize, usize)) -> Self {
        let grid_width = sim.width();

        Self {
            sim,
            origin,
            grid_width,
            tiles_placed: 0,
            current_layer: 0,
            pc_idx: 0,
            ir_idx: 0,
            reg_indices: [0; NUM_REGISTERS],
            alu_tile_indices: [0; 8],
            op_a_data_indices: [0; 4],
            op_a_sel0_indices: [0; 2],
            op_a_sel1_idx: 0,
            op_a_leaf_indices: [0; 2],
            op_a_root_idx: 0,
            op_b_data_indices: [0; 4],
            op_b_sel0_indices: [0; 2],
            op_b_sel1_idx: 0,
            op_b_leaf_indices: [0; 2],
            op_b_root_idx: 0,
            flag_z_idx: 0,
            flag_c_idx: 0,
            rom_indices: Vec::new(),
            ram_indices: Vec::new(),
            next_pc_const_idx: 0,
            decoder_opcode_lo_idx: 0,
            decoder_ctrl_a_lo_idx: 0,
            decoder_ctrl_a_hi_idx: 0,
            decoder_ctrl_b_lo_idx: 0,
            decoder_ctrl_b_hi_idx: 0,
            alu_result_data_indices: [0; 8],
            alu_result_sel0_indices: [0; 4],
            alu_result_sel1_indices: [0; 2],
            alu_result_sel2_idx: 0,
            alu_result_leaf_indices: [0; 4],
            alu_result_mid_indices: [0; 2],
            alu_result_root_idx: 0,
            reg_we_indices: [0; NUM_REGISTERS],
            reg_result_indices: [0; NUM_REGISTERS],
            reg_mux_indices: [0; NUM_REGISTERS],
            flag_z_result_idx: 0,
            flag_z_update_idx: 0,
            flag_z_zero_idx: 0,
            flag_z_mux_idx: 0,
            flag_c_carry_idx: 0,
            flag_c_update_idx: 0,
            flag_c_mux_idx: 0,
        }
    }

    fn place(&mut self, x: usize, y: usize, tile_type: TileType) -> usize {
        self.sim.set_tile_3d(x, y, self.current_layer, tile_type);
        self.tiles_placed += 1;
        self.current_layer * self.sim.tilemap.layer_size + y * self.grid_width + x
    }

    fn place_with_value(&mut self, x: usize, y: usize, tile_type: TileType, value: u64) -> usize {
        let idx = self.place(x, y, tile_type);
        self.sim.set_logic_value_3d(x, y, self.current_layer, value);
        idx
    }

    /// Place a tile on a specific layer (ignores current_layer)
    pub fn place_on_layer(&mut self, x: usize, y: usize, z: usize, tile_type: TileType) -> usize {
        self.sim.set_tile_3d(x, y, z, tile_type);
        self.tiles_placed += 1;
        z * self.sim.tilemap.layer_size + y * self.grid_width + x
    }

    /// Place a tile with a value on a specific layer (ignores current_layer)
    pub fn place_on_layer_with_value(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        tile_type: TileType,
        value: u64,
    ) -> usize {
        let idx = self.place_on_layer(x, y, z, tile_type);
        self.sim.set_logic_value_3d(x, y, z, value);
        idx
    }

    /// Place a ViaUp/ViaDown pair connecting layer z to layer z+1 at (x,y)
    /// Returns (via_up_idx on layer z, via_down_idx on layer z+1)
    pub fn place_via_pair(&mut self, x: usize, y: usize, z: usize) -> (usize, usize) {
        let up_idx = self.place_on_layer(x, y, z, TileType::ViaUp);
        let dn_idx = self.place_on_layer(x, y, z + 1, TileType::ViaDown);
        (up_idx, dn_idx)
    }

    /// Place a chain of via tiles to connect from_z to to_z at (x,y).
    /// For upward (from_z < to_z): places ViaUp on each intermediate layer.
    /// For downward (from_z > to_z): places ViaDown on each intermediate layer.
    /// Returns the index of the final via tile on to_z.
    pub fn place_via_to_layer(&mut self, x: usize, y: usize, from_z: usize, to_z: usize) -> usize {
        if from_z < to_z {
            // Going up: place ViaUp on from_z..to_z-1, ViaDown on to_z
            for z in from_z..to_z {
                self.place_on_layer(x, y, z, TileType::ViaUp);
            }
            self.place_on_layer(x, y, to_z, TileType::ViaDown)
        } else if from_z > to_z {
            // Going down: place ViaDown on from_z..to_z+1, ViaUp on to_z
            for z in (to_z + 1..=from_z).rev() {
                self.place_on_layer(x, y, z, TileType::ViaDown);
            }
            self.place_on_layer(x, y, to_z, TileType::ViaUp)
        } else {
            // Same layer, nothing to do — return the tile index at (x,y,z)
            from_z * self.sim.tilemap.layer_size + y * self.grid_width + x
        }
    }

    /// Place a guard (Const 0) tile
    fn guard(&mut self, x: usize, y: usize) -> usize {
        self.place_with_value(x, y, TileType::Const, 0)
    }

    /// Fill a rectangular region with guard tiles
    #[allow(dead_code)]
    fn guard_rect(&mut self, x: usize, y: usize, w: usize, h: usize) {
        for row in y..y + h {
            for col in x..x + w {
                self.guard(col, row);
            }
        }
    }

    pub fn total_tiles(&self) -> usize {
        self.tiles_placed
    }
}

/// Indices returned by `wire_mux_tree_4to1`.
///
/// Each data input is an individual Const tile (not packed).
/// Selectors use 0 for "select right" and u64::MAX for "select left".
pub struct MuxTree4to1 {
    /// D0..D3 Const tiles holding individual data values
    pub data_indices: [usize; 4],
    /// S0 Const tiles above each leaf mux — set both to the same value
    pub sel0_indices: [usize; 2],
    /// S1 Const tile above root mux
    pub sel1_idx: usize,
    /// Leaf Mux tiles [M23, M01] — mark dirty after writing inputs
    pub leaf_indices: [usize; 2],
    /// Root Mux tile — read final output from here
    pub root_idx: usize,
}

/// Indices returned by `wire_mux_tree_8to1`.
///
/// Three levels of binary mux: 4 leaves, 2 middle, 1 root.
/// Selectors use 0 for "select right" and u64::MAX for "select left".
pub struct MuxTree8to1 {
    /// D0..D7 Const tiles holding individual data values
    pub data_indices: [usize; 8],
    /// S0 Const tiles above each leaf mux (4 pairs = 4 tiles)
    pub sel0_indices: [usize; 4],
    /// S1 Const tiles above each middle mux (1 per middle mux)
    pub sel1_indices: [usize; 2],
    /// S2 Const tile above root mux
    pub sel2_idx: usize,
    /// Leaf Mux tiles [M67, M45, M23, M01]
    pub leaf_indices: [usize; 4],
    /// Middle Mux tiles [M4567, M0123]
    pub mid_indices: [usize; 2],
    /// Root Mux tile — read final output from here
    pub root_idx: usize,
}

impl<'a> WiringContext<'a> {
    /// Wire a 4:1 binary mux tree using standard Mux tiles (no packed data).
    ///
    /// Occupies 7 columns x 5 rows starting at (x, y).
    /// Output at (x+2, y+4).
    ///
    /// Layout:
    /// ```text
    ///     x     x+1    x+2    x+3    x+4    x+5    x+6
    /// y:   Guard  S0     Guard  Guard  Guard  S0     Guard
    /// y+1: D3     M23    D2     Guard  D1     M01    D0
    /// y+2: Guard  WD     Guard  Guard  Guard  WD     Guard
    /// y+3: Guard  WD     S1     Guard  Guard  WD     Guard
    /// y+4: Guard  WD     Root   WL     WL     WD     Guard
    /// ```
    ///
    /// Selector mapping (S0=bit0, S1=bit1, values 0 or u64::MAX):
    ///   sel=0: S0=0, S1=0 -> D0
    ///   sel=1: S0=MAX, S1=0 -> D1
    ///   sel=2: S0=0, S1=MAX -> D2
    ///   sel=3: S0=MAX, S1=MAX -> D3
    pub fn wire_mux_tree_4to1(&mut self, x: usize, y: usize) -> MuxTree4to1 {
        // Row 0: selectors + guards
        self.guard(x, y);
        let sel0_left = self.place_with_value(x + 1, y, TileType::Const, 0);
        self.guard(x + 2, y);
        self.guard(x + 3, y);
        self.guard(x + 4, y);
        let sel0_right = self.place_with_value(x + 5, y, TileType::Const, 0);
        self.guard(x + 6, y);

        // Row 1: data inputs + leaf muxes
        let d3 = self.place_with_value(x, y + 1, TileType::Const, 0);
        let m23 = self.place(x + 1, y + 1, TileType::Mux);
        let d2 = self.place_with_value(x + 2, y + 1, TileType::Const, 0);
        self.guard(x + 3, y + 1);
        let d1 = self.place_with_value(x + 4, y + 1, TileType::Const, 0);
        let m01 = self.place(x + 5, y + 1, TileType::Mux);
        let d0 = self.place_with_value(x + 6, y + 1, TileType::Const, 0);

        // Row 2: wire-down from leaf muxes + guards
        self.guard(x, y + 2);
        self.place(x + 1, y + 2, TileType::WireDown);
        self.guard(x + 2, y + 2);
        self.guard(x + 3, y + 2);
        self.guard(x + 4, y + 2);
        self.place(x + 5, y + 2, TileType::WireDown);
        self.guard(x + 6, y + 2);

        // Row 3: wire-down continues + S1 selector for root
        self.guard(x, y + 3);
        self.place(x + 1, y + 3, TileType::WireDown);
        let sel1 = self.place_with_value(x + 2, y + 3, TileType::Const, 0);
        self.guard(x + 3, y + 3);
        self.guard(x + 4, y + 3);
        self.place(x + 5, y + 3, TileType::WireDown);
        self.guard(x + 6, y + 3);

        // Row 4: root mux + wire routing
        self.guard(x, y + 4);
        self.place(x + 1, y + 4, TileType::WireDown);
        let root = self.place(x + 2, y + 4, TileType::Mux);
        self.place(x + 3, y + 4, TileType::WireLeft);
        self.place(x + 4, y + 4, TileType::WireLeft);
        self.place(x + 5, y + 4, TileType::WireDown);
        self.guard(x + 6, y + 4);

        MuxTree4to1 {
            data_indices: [d0, d1, d2, d3],
            sel0_indices: [sel0_left, sel0_right],
            sel1_idx: sel1,
            leaf_indices: [m23, m01],
            root_idx: root,
        }
    }

    /// Wire an 8:1 binary mux tree using standard Mux tiles (no packed data).
    ///
    /// Built from two 4:1 subtrees (left=D7-D4, right=D3-D0) plus a root level.
    /// Occupies 15 columns x 9 rows starting at (x, y).
    /// Output at (x+7, y+8).
    ///
    /// Selector mapping (S0=bit0, S1=bit1, S2=bit2, values 0 or u64::MAX):
    ///   sel=0..3: S2=0, right subtree (D0-D3)
    ///   sel=4..7: S2=MAX, left subtree (D4-D7)
    pub fn wire_mux_tree_8to1(&mut self, x: usize, y: usize) -> MuxTree8to1 {
        // Left subtree: D7-D4 at columns x..x+6
        let left = self.wire_mux_tree_4to1(x, y);

        // Guard column between subtrees at x+7
        for row in 0..5 {
            self.guard(x + 7, y + row);
        }

        // Right subtree: D3-D0 at columns x+8..x+14
        let right = self.wire_mux_tree_4to1(x + 8, y);

        // Row 5: guard row below subtrees, with WireDown pass-throughs at root positions
        for col in 0..15 {
            if col == 2 || col == 10 {
                self.place(x + col, y + 5, TileType::WireDown);
            } else {
                self.guard(x + col, y + 5);
            }
        }

        // Row 6: WD from roots + S2 selector
        self.guard(x, y + 6);
        self.guard(x + 1, y + 6);
        self.place(x + 2, y + 6, TileType::WireDown);
        for col in 3..7 {
            self.guard(x + col, y + 6);
        }
        let sel2 = self.place_with_value(x + 7, y + 6, TileType::Const, 0);
        for col in 8..10 {
            self.guard(x + col, y + 6);
        }
        self.place(x + 10, y + 6, TileType::WireDown);
        for col in 11..15 {
            self.guard(x + col, y + 6);
        }

        // Row 7: WD continues + WD for root's up
        self.guard(x, y + 7);
        self.guard(x + 1, y + 7);
        self.place(x + 2, y + 7, TileType::WireDown);
        for col in 3..7 {
            self.guard(x + col, y + 7);
        }
        self.place(x + 7, y + 7, TileType::WireDown);
        for col in 8..10 {
            self.guard(x + col, y + 7);
        }
        self.place(x + 10, y + 7, TileType::WireDown);
        for col in 11..15 {
            self.guard(x + col, y + 7);
        }

        // Row 8: Root mux with WL chain from right, WD from left
        self.guard(x, y + 8);
        self.guard(x + 1, y + 8);
        self.place(x + 2, y + 8, TileType::WireDown);
        self.place(x + 3, y + 8, TileType::WireRight);
        self.place(x + 4, y + 8, TileType::WireRight);
        self.place(x + 5, y + 8, TileType::WireRight);
        self.place(x + 6, y + 8, TileType::WireRight);
        let root_8 = self.place(x + 7, y + 8, TileType::Mux);
        self.place(x + 8, y + 8, TileType::WireLeft);
        self.place(x + 9, y + 8, TileType::WireLeft);
        self.place(x + 10, y + 8, TileType::WireDown);
        for col in 11..15 {
            self.guard(x + col, y + 8);
        }

        MuxTree8to1 {
            data_indices: [
                right.data_indices[0],
                right.data_indices[1],
                right.data_indices[2],
                right.data_indices[3],
                left.data_indices[0],
                left.data_indices[1],
                left.data_indices[2],
                left.data_indices[3],
            ],
            sel0_indices: [
                left.sel0_indices[0],
                left.sel0_indices[1],
                right.sel0_indices[0],
                right.sel0_indices[1],
            ],
            sel1_indices: [left.sel1_idx, right.sel1_idx],
            sel2_idx: sel2,
            leaf_indices: [
                left.leaf_indices[0],
                left.leaf_indices[1],
                right.leaf_indices[0],
                right.leaf_indices[1],
            ],
            mid_indices: [left.root_idx, right.root_idx],
            root_idx: root_8,
        }
    }
}

// =============================================================================
// Physical CPU Wiring — Fully Tile-Based
// =============================================================================

/// Indices for the fully tile-based CPU.
///
/// The entire datapath is physical: decoder, operand routing, ALU, result
/// selection, register writeback, flags, and branch logic all use tiles.
/// Only memory operations (LD/ST) remain in software.
#[allow(dead_code)]
pub struct PhysicalCpuIndices {
    pub pc_idx: usize,
    pub ir_idx: usize,
    pub reg_indices: [usize; NUM_REGISTERS],
    pub flag_z_idx: usize,
    pub flag_c_idx: usize,
    pub rom_indices: Vec<usize>,
    pub ram_indices: Vec<usize>,
    pub grid_width: usize,
    pub tile_count: usize,
    /// Shr tile that computes IR >> 4 (opcode)
    pub shr_opcode_idx: usize,
    /// Mux8to1 LUT indices [a_lo, a_hi, b_lo, b_hi]
    pub decoder_lut_indices: [usize; 4],
    /// Bank-merged ctrl_a output (selects lo/hi based on opcode bit 3)
    pub merged_ctrl_a_idx: usize,
    /// Bank-merged ctrl_b output
    pub merged_ctrl_b_idx: usize,
    /// BitSelect outputs for control signals from ctrl_a:
    /// [alu_sel0, alu_sel1, alu_sel2, reg_write_en, update_flags, use_immediate]
    pub ctrl_a_bits: [usize; 6],
    /// BitSelect outputs for control signals from ctrl_b:
    /// [is_jmp, is_jz, is_jnz, mem_read, mem_write]
    pub ctrl_b_bits: [usize; 5],
    /// BitSelect outputs for IR field bits:
    /// [rd_bit0 (bit2), rd_bit1 (bit3), rs_bit0 (bit0), rs_bit1 (bit1)]
    pub ir_field_bits: [usize; 4],
    /// Operand A mux tree root (outputs reg[rd])
    pub op_a_root_idx: usize,
    /// Operand B mux tree root (outputs reg[rs])
    pub op_b_root_idx: usize,
    /// ALU result mux tree root (outputs ALU result selected by alu_sel)
    pub alu_result_root_idx: usize,
    /// Register write-enable Mux tiles (one per reg, gated by decoded rd)
    pub reg_we_mux_indices: [usize; NUM_REGISTERS],
    /// Register writeback Mux tiles (feedback mux per register)
    pub reg_mux_indices: [usize; NUM_REGISTERS],
    /// Branch taken output tile
    pub branch_taken_idx: usize,
    /// Jump target tile (IR & 0x3F mask)
    pub jump_target_idx: usize,
    /// Flag Z Zero tile
    pub flag_z_zero_idx: usize,
    /// Flag Z Mux tile
    pub flag_z_mux_idx: usize,
    /// Flag C AddCarry tile
    pub flag_c_addcarry_idx: usize,
    /// Flag C BitSelect (carry bit extraction)
    pub flag_c_bit_idx: usize,
    /// Flag C Mux tile
    pub flag_c_mux_idx: usize,
    /// Register result Const tiles (for memory operations only)
    pub reg_result_indices: [usize; NUM_REGISTERS],
    /// Operand A data Const tiles (for software fallback)
    pub op_a_data_indices: [usize; 4],
    /// Operand A selector Const tiles
    pub op_a_sel0_indices: [usize; 2],
    pub op_a_sel1_idx: usize,
    /// Operand A leaf mux tiles
    pub op_a_leaf_indices: [usize; 2],
    /// Operand B data Const tiles
    pub op_b_data_indices: [usize; 4],
    /// Operand B selector Const tiles
    pub op_b_sel0_indices: [usize; 2],
    pub op_b_sel1_idx: usize,
    /// Operand B leaf mux tiles
    pub op_b_leaf_indices: [usize; 2],
    /// ALU tile indices [Add, Sub, And, Or, Xor, Not, Shl, Shr]
    pub alu_tile_indices: [usize; 8],
    /// ALU result tree data Const tiles
    pub alu_result_data_indices: [usize; 8],
    /// ALU result tree selector Const tiles
    pub alu_result_sel0_indices: [usize; 4],
    pub alu_result_sel1_indices: [usize; 2],
    pub alu_result_sel2_idx: usize,
    /// ALU result tree leaf mux tiles
    pub alu_result_leaf_indices: [usize; 4],
    /// ALU result tree mid mux tiles
    pub alu_result_mid_indices: [usize; 2],
    /// Next PC Const tile (for software PC writes during memory ops)
    pub next_pc_const_idx: usize,
    /// Flag update Const tiles (for software flag updates during memory ops)
    pub flag_z_update_idx: usize,
    pub flag_c_update_idx: usize,
    pub flag_z_result_idx: usize,
    pub flag_c_carry_idx: usize,
    /// Const tile for mem_write control signal (software-written before tick)
    pub mem_write_const_idx: usize,
    /// Const tile for write data (software-written before tick)
    pub write_data_const_idx: usize,
    /// Physical Ram tile indices (8 cells, addresses 0-7)
    pub physical_ram_indices: [usize; 8],
    /// L1 merge Mux indices — select between physical ALU result and software LD data
    pub wb_merge_mux_l1_indices: [usize; NUM_REGISTERS],
    /// L1 LD data Const indices — software writes RAM value here on LD ticks
    pub ld_data_l1_indices: [usize; NUM_REGISTERS],
    /// L1 mem_read selector Const indices — MAX=ALU (default), 0=LD data
    pub mem_read_l1_indices: [usize; NUM_REGISTERS],
    /// L1 RAM read Mux tree root index — outputs addressed Ram cell value (Sprint 87)
    pub ram_read_mux_root_idx: usize,
}

/// Wire the fully physical CPU datapath.
///
/// After `tick_with_delays()` × 2, the entire execution pipeline settles
/// through tiles alone. Software only handles LD/ST memory operations.
///
/// # Layout (128×128 grid, origin at ox, oy)
///
/// ```text
/// Row  0:     Clock
/// Row  1:     PC (Register8 + Const_NPC)
/// Row  2:     IR fetch (packed ROM + Mux8to1)
/// Row  3:     IR WD fan-out
/// Row  4:     Shr(IR, 4) — opcode extraction + IR WR fan-out
/// Row  5:     Opcode WD/WR routing
/// Rows 6-9:   Decoder LUTs (4 Mux8to1)
/// Row 10:     WD chains end
/// Row 11:     Bank-merge BitSelect: bit3 extraction at cols 13, 21
/// Row 12:     Bank-merge Mux: Mux(bit3, hi, lo) at cols 13, 21
/// Row 13:     Guard
/// Rows 14-19: BitSelect columns (ctrl_a bits, ctrl_b bits, IR field bits)
/// Row 20:     Guard
/// Rows 21-28: Register file (4 regs with writeback Mux)
/// Row 29:     Guard
/// Rows 30-34: Operand mux trees (A and B)
/// Row 35:     Bus routing
/// Row 36:     Guard
/// Rows 37-44: ALU (8 operations)
/// Row 45:     Guard
/// Rows 46-54: ALU result mux tree (8:1)
/// Row 55:     Guard
/// Rows 56-59: Flag circuit (Z + C)
/// Row 60:     Guard
/// Rows 63-70: Branch gate network (physical branch_taken)
/// Rows 71-74: Physical PC update circuit (Add, Mux, L1 routing)
/// Row  75:    Guard
/// Rows 76+:   ROM + RAM
/// ```
pub fn wire_physical_cpu(
    ctx: &mut WiringContext,
    program: &[u8],
    rom_size: usize,
    ram_size: usize,
    _initial_regs: &[u64; NUM_REGISTERS],
) -> PhysicalCpuIndices {
    let (ox, oy) = ctx.origin;

    // Physical CPU requires at least 2 layers for L1 PC routing
    assert!(
        ctx.sim.num_layers() >= 2,
        "Physical CPU requires at least 2 layers (use Simulation::with_size_layered)"
    );

    // =========================================================================
    // ROW 0: CLOCK + GUARDS
    // =========================================================================
    ctx.guard(ox, oy);
    ctx.guard(ox + 1, oy);
    ctx.guard(ox + 2, oy);
    ctx.place(ox + 3, oy, TileType::ClockGlobal);
    for col in 4..30 {
        ctx.guard(ox + col, oy);
    }

    // =========================================================================
    // ROW 1: PC CIRCUIT
    // =========================================================================
    ctx.guard(ox, oy + 1);
    // ViaUp reads next_pc from L1 return path — no software writes needed
    ctx.next_pc_const_idx = ctx.place(ox + 2, oy + 1, TileType::ViaUp);
    ctx.pc_idx = ctx.place_with_value(ox + 3, oy + 1, TileType::Register8, 0);
    for col in 4..30 {
        ctx.guard(ox + col, oy + 1);
    }

    // =========================================================================
    // ROW 2: INSTRUCTION FETCH (16-byte ROM via Mux16to1)
    // =========================================================================
    let effective_rom = rom_size.min(16);
    // Pack instructions 0-7 into ROM-A
    let mut packed_rom_a: u64 = 0;
    for addr in 0..effective_rom.min(8) {
        let byte = if addr < program.len() {
            program[addr] as u64
        } else {
            0
        };
        packed_rom_a |= byte << (addr * 8);
    }
    // Pack instructions 8-15 into ROM-B
    let mut packed_rom_b: u64 = 0;
    for addr in 8..effective_rom {
        let byte = if addr < program.len() {
            program[addr] as u64
        } else {
            0
        };
        packed_rom_b |= byte << ((addr - 8) * 8);
    }
    // ROM-B at (ox+1, oy+1) — above Mux16to1 (up input)
    ctx.place_with_value(ox + 1, oy + 1, TileType::Const, packed_rom_b);
    ctx.place_with_value(ox, oy + 2, TileType::Const, packed_rom_a);
    ctx.ir_idx = ctx.place(ox + 1, oy + 2, TileType::Mux16to1);
    ctx.place(ox + 2, oy + 2, TileType::WireLeft);
    ctx.place(ox + 3, oy + 2, TileType::WireDown); // PC down
    for col in 4..30 {
        ctx.guard(ox + col, oy + 2);
    }

    // =========================================================================
    // ROW 3: IR FAN-OUT
    // =========================================================================
    ctx.guard(ox, oy + 3);
    ctx.place(ox + 1, oy + 3, TileType::WireDown); // IR flows down
    for col in 2..30 {
        ctx.guard(ox + col, oy + 3);
    }

    // =========================================================================
    // ROW 4: OPCODE EXTRACTION — Shr(IR, 4)
    // =========================================================================
    // IR WD@(1,3)→WD@(1,4)→WR chain→Shr@(7,4)
    ctx.guard(ox, oy + 4);
    ctx.place(ox + 1, oy + 4, TileType::WireDown); // IR continues
    for col in 2..=6 {
        ctx.place(ox + col, oy + 4, TileType::WireRight);
    }
    let shr_opcode_idx = ctx.place(ox + 7, oy + 4, TileType::Shr);
    ctx.place_with_value(ox + 8, oy + 4, TileType::Const, 4);
    for col in 9..30 {
        ctx.guard(ox + col, oy + 4);
    }

    // =========================================================================
    // ROW 5: OPCODE WD CHAIN START
    // =========================================================================
    // Shr@(7,4)→WD@(7,5)→WR@(8,5)→WD chain at col 8 for LUTs
    ctx.guard(ox, oy + 5);
    ctx.place(ox + 1, oy + 5, TileType::WireDown); // IR continues
    for col in 2..=6 {
        ctx.guard(ox + col, oy + 5);
    }
    ctx.place(ox + 7, oy + 5, TileType::WireDown); // Shr output down
    ctx.place(ox + 8, oy + 5, TileType::WireRight); // routes to col 8
    for col in 9..30 {
        ctx.guard(ox + col, oy + 5);
    }

    // =========================================================================
    // ROWS 6-9: DECODER LUTs — 4 Mux8to1 stacked vertically
    // =========================================================================
    // Each row: Const(packed)@6, Mux8to1@7, WD(opcode)@8
    // LUT order: a_lo, a_hi, b_lo, b_hi (rows 6, 7, 8, 9)
    let packed_ctrl_a_lo: u64 = 0x00
        | (0x28 << 8)
        | (0x08 << 16)
        | (0x18 << 24)
        | (0x19u64 << 32)
        | (0x1A << 40)
        | (0x1B << 48)
        | (0x1C << 56);
    let packed_ctrl_a_hi: u64 = 0x1E
        | (0x1F << 8)
        | (0x11 << 16)
        | (0x00 << 24)
        | (0x00u64 << 32)
        | (0x00 << 40)
        | (0x08 << 48)
        | (0x00 << 56);
    let packed_ctrl_b_lo: u64 = 0;
    let packed_ctrl_b_hi: u64 = 0x00
        | (0x00 << 8)
        | (0x00 << 16)
        | (0x01 << 24)
        | (0x02u64 << 32)
        | (0x04 << 40)
        | (0x08 << 48)
        | (0x10 << 56);

    let packed_values = [
        packed_ctrl_a_lo,
        packed_ctrl_a_hi,
        packed_ctrl_b_lo,
        packed_ctrl_b_hi,
    ];
    let mut decoder_lut_indices = [0usize; 4];

    for (i, &packed) in packed_values.iter().enumerate() {
        let row = oy + 6 + i;
        ctx.guard(ox, row);
        ctx.place(ox + 1, row, TileType::WireDown); // IR continues
        for col in 2..=5 {
            ctx.guard(ox + col, row);
        }
        ctx.place_with_value(ox + 6, row, TileType::Const, packed);
        decoder_lut_indices[i] = ctx.place(ox + 7, row, TileType::Mux8to1);
        ctx.place(ox + 8, row, TileType::WireDown); // opcode_lo bus
        for col in 9..30 {
            ctx.guard(ox + col, row);
        }
    }

    ctx.decoder_ctrl_a_lo_idx = decoder_lut_indices[0];
    ctx.decoder_ctrl_a_hi_idx = decoder_lut_indices[1];
    ctx.decoder_ctrl_b_lo_idx = decoder_lut_indices[2];
    ctx.decoder_ctrl_b_hi_idx = decoder_lut_indices[3];
    ctx.decoder_opcode_lo_idx = shr_opcode_idx;

    // =========================================================================
    // REVISED LAYOUT: Spread LUTs horizontally (rows 5-6 overwrite)
    // =========================================================================
    // The stacked LUT layout (rows 6-9) doesn't allow routing individual LUT
    // outputs down — each row overwrites col 7. Instead, spread all 4 LUTs
    // across row 6 in separate column groups, each with its own output column:
    //   a_hi: Const@(10,6), Mux8to1@(11,6), WD:opcode@(12,6)
    //   a_lo: Const@(14,6), Mux8to1@(15,6), WD:opcode@(16,6)
    //   b_hi: Const@(18,6), Mux8to1@(19,6), WD:opcode@(20,6)
    //   b_lo: Const@(22,6), Mux8to1@(23,6), WD:opcode@(24,6)
    //
    // hi banks at lower columns so merge Mux(left=hi, right=lo) works correctly.

    // Guards initially placed at rows 6-9 and row 10 are overwritten below.

    // Overwrite row 5 cols 8-24 with WR chain to fan opcode across to all LUTs:
    ctx.place(ox + 8, oy + 5, TileType::WireRight); // overwrite: was WR already
    for col in 9..=24 {
        ctx.place(ox + col, oy + 5, TileType::WireRight);
    }
    ctx.guard(ox + 25, oy + 5);

    // Row 6: Spread LUTs (overwrite old stacked layout)
    ctx.guard(ox, oy + 6);
    ctx.place(ox + 1, oy + 6, TileType::WireDown); // IR continues
    for col in 2..=9 {
        ctx.guard(ox + col, oy + 6);
    }
    ctx.place_with_value(ox + 10, oy + 6, TileType::Const, packed_ctrl_a_hi);
    decoder_lut_indices[1] = ctx.place(ox + 11, oy + 6, TileType::Mux8to1); // a_hi
    ctx.place(ox + 12, oy + 6, TileType::WireDown); // opcode from row 5
    ctx.guard(ox + 13, oy + 6);
    ctx.place_with_value(ox + 14, oy + 6, TileType::Const, packed_ctrl_a_lo);
    decoder_lut_indices[0] = ctx.place(ox + 15, oy + 6, TileType::Mux8to1); // a_lo
    ctx.place(ox + 16, oy + 6, TileType::WireDown);
    ctx.guard(ox + 17, oy + 6);
    ctx.place_with_value(ox + 18, oy + 6, TileType::Const, packed_ctrl_b_hi);
    decoder_lut_indices[3] = ctx.place(ox + 19, oy + 6, TileType::Mux8to1); // b_hi
    ctx.place(ox + 20, oy + 6, TileType::WireDown);
    ctx.guard(ox + 21, oy + 6);
    ctx.place_with_value(ox + 22, oy + 6, TileType::Const, packed_ctrl_b_lo);
    decoder_lut_indices[2] = ctx.place(ox + 23, oy + 6, TileType::Mux8to1); // b_lo
    ctx.place(ox + 24, oy + 6, TileType::WireDown);
    for col in 25..30 {
        ctx.guard(ox + col, oy + 6);
    }

    ctx.decoder_ctrl_a_lo_idx = decoder_lut_indices[0];
    ctx.decoder_ctrl_a_hi_idx = decoder_lut_indices[1];
    ctx.decoder_ctrl_b_lo_idx = decoder_lut_indices[2];
    ctx.decoder_ctrl_b_hi_idx = decoder_lut_indices[3];

    // Rows 7-10: guard rows with IR WD chain at col 1 and LUT output WD chains
    for row in (oy + 7)..=(oy + 10) {
        ctx.guard(ox, row);
        ctx.place(ox + 1, row, TileType::WireDown); // IR continues
        for col in 2..30 {
            // Skip cols that get WD chains below — they'll be overwritten
            if col == 7
                || col == 11
                || col == 12
                || col == 15
                || col == 19
                || col == 20
                || col == 23
            {
                continue;
            }
            ctx.guard(ox + col, row);
        }
        // LUT output WD chains at cols 11, 15, 19, 23
        ctx.place(ox + 11, row, TileType::WireDown);
        ctx.place(ox + 15, row, TileType::WireDown);
        ctx.place(ox + 19, row, TileType::WireDown);
        ctx.place(ox + 23, row, TileType::WireDown);
        // Shr opcode WD chain at col 7
        ctx.place(ox + 7, row, TileType::WireDown);
        // Opcode WD chains at cols 12, 20 for bank merge BitSelect tiles
        ctx.place(ox + 12, row, TileType::WireDown);
        ctx.place(ox + 20, row, TileType::WireDown);
    }

    // =========================================================================
    // ROW 11: BANK MERGE — BitSelect(opcode, 3) for physical bank selection
    // =========================================================================
    // Two BitSelect tiles extract bit 3 from opcode (routed down via WD at
    // cols 12 and 20). Each BitSelect sits directly above its merge Mux,
    // avoiding any horizontal crossing with LUT output WD chains.
    //
    // BitSelect_a@col13: left=opcode@col12, right=Const(3)@col14 → bit3 for Mux_a
    // BitSelect_b@col21: left=opcode@col20, right=Const(3)@col22 → bit3 for Mux_b
    ctx.guard(ox, oy + 11);
    ctx.place(ox + 1, oy + 11, TileType::WireDown); // IR continues
    for col in 2..=10 {
        ctx.guard(ox + col, oy + 11);
    }
    ctx.place(ox + 11, oy + 11, TileType::WireDown); // a_hi continues
    ctx.place(ox + 12, oy + 11, TileType::WireDown); // opcode continues for BitSelect_a
    ctx.place(ox + 13, oy + 11, TileType::BitSelect); // bit3 for Mux_a
    ctx.place_with_value(ox + 14, oy + 11, TileType::Const, 3);
    ctx.place(ox + 15, oy + 11, TileType::WireDown); // a_lo continues
    for col in 16..=18 {
        ctx.guard(ox + col, oy + 11);
    }
    ctx.place(ox + 19, oy + 11, TileType::WireDown); // b_hi continues
    ctx.place(ox + 20, oy + 11, TileType::WireDown); // opcode continues for BitSelect_b
    ctx.place(ox + 21, oy + 11, TileType::BitSelect); // bit3 for Mux_b
    ctx.place_with_value(ox + 22, oy + 11, TileType::Const, 3);
    ctx.place(ox + 23, oy + 11, TileType::WireDown); // b_lo continues
    for col in 24..30 {
        ctx.guard(ox + col, oy + 11);
    }

    // =========================================================================
    // ROW 12: BANK MERGE MUX — physical bank select
    // =========================================================================
    // Mux(up=bit3, left=hi, right=lo): bit3!=0 → hi bank, bit3==0 → lo bank
    // WR carries hi value from WD chain; WL carries lo value from WD chain.
    ctx.guard(ox, oy + 12);
    ctx.place(ox + 1, oy + 12, TileType::WireDown); // IR continues
    for col in 2..=10 {
        ctx.guard(ox + col, oy + 12);
    }
    ctx.place(ox + 11, oy + 12, TileType::WireDown); // a_hi WD (feeds WR@col12)
    ctx.place(ox + 12, oy + 12, TileType::WireRight); // carries a_hi → Mux left
    let merged_ctrl_a_idx = ctx.place(ox + 13, oy + 12, TileType::Mux); // merge Mux_a
    ctx.place(ox + 14, oy + 12, TileType::WireLeft); // carries a_lo ← Mux right
    ctx.place(ox + 15, oy + 12, TileType::WireDown); // a_lo WD (feeds WL@col14)
    for col in 16..=18 {
        ctx.guard(ox + col, oy + 12);
    }
    ctx.place(ox + 19, oy + 12, TileType::WireDown); // b_hi WD (feeds WR@col20)
    ctx.place(ox + 20, oy + 12, TileType::WireRight); // carries b_hi → Mux left
    let merged_ctrl_b_idx = ctx.place(ox + 21, oy + 12, TileType::Mux); // merge Mux_b
    ctx.place(ox + 22, oy + 12, TileType::WireLeft); // carries b_lo ← Mux right
    ctx.place(ox + 23, oy + 12, TileType::WireDown); // b_lo WD (feeds WL@col22)
    for col in 24..30 {
        ctx.guard(ox + col, oy + 12);
    }

    // Row 13: guard row + ctrl_a/ctrl_b bit extraction start
    // ctrl_a flows down from merged_ctrl_a Mux@(13, 12) via WD chain at col 13.
    // ctrl_b flows down from merged_ctrl_b Mux@(21, 12) via WD chain at col 21.
    // BitSelect extracts individual bits; Const selects bit position.
    let mut ctrl_a_bits = [0usize; 6]; // [alu_sel0, alu_sel1, alu_sel2, reg_write_en, update_flags, use_immediate]
    let mut ctrl_b_bits = [0usize; 5]; // [is_jmp, is_jz, is_jnz, mem_read, mem_write]
    ctx.guard(ox, oy + 13);
    ctx.place(ox + 1, oy + 13, TileType::WireDown); // IR continues
    for col in 2..21 {
        ctx.guard(ox + col, oy + 13);
    }
    // ctrl_a bit extraction at cols 13-15 (overwrites guards)
    ctx.place(ox + 13, oy + 13, TileType::WireDown); // ctrl_a from merge Mux
    ctrl_a_bits[0] = ctx.place(ox + 14, oy + 13, TileType::BitSelect); // alu_sel bit 0
    ctx.place_with_value(ox + 15, oy + 13, TileType::Const, 0);
    // ctrl_b bit extraction at cols 21-23
    ctx.place(ox + 21, oy + 13, TileType::WireDown); // ctrl_b from merge Mux
    ctrl_b_bits[0] = ctx.place(ox + 22, oy + 13, TileType::BitSelect); // is_jmp (bit 0)
    // (23, 13) is Const(0) from guard — correct for bit 0 extraction
    ctx.guard(ox + 23, oy + 13);
    for col in 24..30 {
        ctx.guard(ox + col, oy + 13);
    }

    // =========================================================================
    // ROWS 14-17: IR FIELD EXTRACTION via BitSelect
    // =========================================================================
    // Extract rd/rs bits from IR for mux tree selectors.
    // BitSelect(IR, bit_pos) → 0 or u64::MAX, perfect for Mux selector inputs.

    let mut ir_field_bits = [0usize; 4]; // [rd_bit0, rd_bit1, rs_bit0, rs_bit1]
    let ir_bit_positions = [2, 3, 0, 1]; // bit positions in instruction byte

    for (i, &bit_pos) in ir_bit_positions.iter().enumerate() {
        let row = oy + 14 + i;
        ctx.guard(ox, row);
        ctx.place(ox + 1, row, TileType::WireDown); // IR continues
        ir_field_bits[i] = ctx.place(ox + 2, row, TileType::BitSelect);
        ctx.place_with_value(ox + 3, row, TileType::Const, bit_pos as u64);
        for col in 4..21 {
            ctx.guard(ox + col, row);
        }
        // ctrl_a bit extraction at cols 13-15 (overwrites guards)
        ctx.place(ox + 13, row, TileType::WireDown); // ctrl_a continues
        ctrl_a_bits[i + 1] = ctx.place(ox + 14, row, TileType::BitSelect);
        ctx.place_with_value(ox + 15, row, TileType::Const, (i + 1) as u64);
        // ctrl_b bit extraction continues at cols 21-23
        ctx.place(ox + 21, row, TileType::WireDown); // ctrl_b continues
        ctrl_b_bits[i + 1] = ctx.place(ox + 22, row, TileType::BitSelect);
        ctx.place_with_value(ox + 23, row, TileType::Const, (i + 1) as u64); // bit position 1-4
        for col in 24..30 {
            ctx.guard(ox + col, row);
        }
    }

    // Row 18: jump_target = And(IR, 0x3F) + ctrl_a bit 5 extraction
    ctx.guard(ox, oy + 18);
    ctx.place(ox + 1, oy + 18, TileType::WireDown); // IR continues
    let jump_target_idx = ctx.place(ox + 2, oy + 18, TileType::And);
    ctx.place_with_value(ox + 3, oy + 18, TileType::Const, 0x3F);
    for col in 4..30 {
        ctx.guard(ox + col, oy + 18);
    }
    // ctrl_a bit 5 extraction at cols 13-15 (overwrites guards)
    ctx.place(ox + 13, oy + 18, TileType::WireDown); // ctrl_a continues
    ctrl_a_bits[5] = ctx.place(ox + 14, oy + 18, TileType::BitSelect); // use_immediate
    ctx.place_with_value(ox + 15, oy + 18, TileType::Const, 5);

    // Row 19: IR fan-out bus for physical selector routing
    // IR→WR chain feeds BitSelect tiles at row 20 for physical rd AND rs selectors.
    // Extended to col 54 for rs BitSelect tiles at cols 47, 50, 53.
    ctx.guard(ox, oy + 19);
    ctx.place(ox + 1, oy + 19, TileType::WireDown); // IR continues
    for col in 2..=54 {
        ctx.place(ox + col, oy + 19, TileType::WireRight); // IR fan-out rightward
    }
    for col in 55..=56 {
        ctx.guard(ox + col, oy + 19);
    }

    // =========================================================================
    // ROWS 20-28: REGISTER FILE (spread layout, 4-column spacing)
    // =========================================================================
    // 4 registers spread across cols 28-42 for physical operand routing.
    // Each register group: result Const, Mux (+ WE above), Register8.
    //   Reg0: result@40, Mux@41, Register8@42
    //   Reg1: result@36, Mux@37, Register8@38
    //   Reg2: result@32, Mux@33, Register8@34
    //   Reg3: result@28, Mux@29, Register8@30
    //
    // 4-column spacing ensures Register8 cols {30,34,38,42} don't overlap
    // result cols {28,32,36,40}. WD chains from Register8 tiles flow down
    // into operand tree data inputs (Phase 3 Step 2).

    let reg_cols: [(usize, usize, usize); NUM_REGISTERS] = [
        (40, 41, 42), // Reg0
        (36, 37, 38), // Reg1
        (32, 33, 34), // Reg2
        (28, 29, 30), // Reg3
    ];

    for reg in 0..NUM_REGISTERS {
        let we_row = oy + 20 + reg * 2;
        let reg_row = we_row + 1;
        let (result_col, mux_col, reg8_col) = reg_cols[reg];

        // Guard entire WE row in register area
        for col in 27..=43 {
            ctx.guard(ox + col, we_row);
        }
        // Place WE Const above Mux
        ctx.reg_we_indices[reg] = ctx.place_with_value(ox + mux_col, we_row, TileType::Const, 0);

        // Guard entire Reg row in register area
        for col in 27..=43 {
            ctx.guard(ox + col, reg_row);
        }
        // Place register tiles
        ctx.reg_result_indices[reg] =
            ctx.place_with_value(ox + result_col, reg_row, TileType::Const, 0);
        ctx.reg_mux_indices[reg] = ctx.place(ox + mux_col, reg_row, TileType::Mux);
        ctx.reg_indices[reg] = ctx.place_with_value(ox + reg8_col, reg_row, TileType::Register8, 0);
    }

    // Guard row below registers
    for col in 27..=43 {
        ctx.guard(ox + col, oy + 28);
    }

    // =========================================================================
    // REGISTER WD CHAINS: Register8 → row 28 (physical operand routing)
    // =========================================================================
    // WD chains carry register values downward to Tree A data inputs.
    // Each Register8 column gets WD tiles from its row+1 down through
    // other registers' guard positions to row 28.
    //   Reg0@(42,21) → WD at rows 22-28 (7 tiles)
    //   Reg1@(38,23) → WD at rows 24-28 (5 tiles)
    //   Reg2@(34,25) → WD at rows 26-28 (3 tiles)
    //   Reg3@(30,27) → WD at row 28 (1 tile)
    for reg in 0..NUM_REGISTERS {
        let reg8_col = reg_cols[reg].2;
        let reg_row = oy + 20 + reg * 2 + 1;
        for row in (reg_row + 1)..=(oy + 28) {
            ctx.place(ox + reg8_col, row, TileType::WireDown);
        }
    }

    // =========================================================================
    // PHYSICAL rd SELECTOR ROUTING: IR BitSelect → Tree A selectors
    // =========================================================================
    // IR is fanned out rightward at row 19 via WR chain. At row 20 (Reg0 WE),
    // BitSelect tiles extract rd bits and WD chains carry them to Tree A.
    //
    //   rd_bit0 (bit 2): BS@(31,20) → WD col 31 → Tree A S0 at row 29
    //                     BS@(39,20) → WD col 39 → Tree A S0 at row 29
    //   rd_bit1 (bit 3): BS@(35,20) → WD col 35 → WR@(36,31) → Tree A S1
    //
    // Each BitSelect reads: left=WD(IR from row 19), right=Const(bit_pos)
    // Cols 31/35/39 are guards throughout rows 20-28 (between register groups).

    // Row 20: BitSelect tiles for rd extraction (overwrites guards placed above)
    // rd_bit0 at col 31: WD(IR)@30, BitSelect@31, Const(2)@32
    ctx.place(ox + 30, oy + 20, TileType::WireDown); // IR from WR@(30,19)
    ctx.place(ox + 31, oy + 20, TileType::BitSelect); // rd_bit0
    ctx.place_with_value(ox + 32, oy + 20, TileType::Const, 2); // bit position 2
    // rd_bit1 at col 35: WD(IR)@34, BitSelect@35, Const(3)@36
    ctx.place(ox + 34, oy + 20, TileType::WireDown); // IR from WR@(34,19)
    ctx.place(ox + 35, oy + 20, TileType::BitSelect); // rd_bit1
    ctx.place_with_value(ox + 36, oy + 20, TileType::Const, 3); // bit position 3
    // rd_bit0 copy at col 39: WD(IR)@38, BitSelect@39, Const(2)@40
    ctx.place(ox + 38, oy + 20, TileType::WireDown); // IR from WR@(38,19)
    ctx.place(ox + 39, oy + 20, TileType::BitSelect); // rd_bit0 copy
    ctx.place_with_value(ox + 40, oy + 20, TileType::Const, 2); // bit position 2

    // WD chains: carry rd selector signals through register area (rows 21-28)
    // Cols 31, 35, 39 are guards in all register rows (between register groups).
    for row in (oy + 21)..=(oy + 28) {
        ctx.place(ox + 31, row, TileType::WireDown); // rd_bit0 → Tree A S0
        ctx.place(ox + 35, row, TileType::WireDown); // rd_bit1 → Tree A S1
        ctx.place(ox + 39, row, TileType::WireDown); // rd_bit0 → Tree A S0
    }

    // =========================================================================
    // PHYSICAL rs SELECTOR ROUTING: IR BitSelect → Tree B selectors
    // =========================================================================
    // New Tree B is relocated to cols 46-54. rs BitSelect tiles extract rs bits
    // from IR at row 20, and WD chains carry signals to Tree B at rows 33-37.
    //
    //   rs_bit0 (bit 0): BS@(47,20) → WD col 47 → Tree B S0 at row 33
    //                     BS@(53,20) → WD col 53 → Tree B S0 at row 33
    //   rs_bit1 (bit 1): BS@(50,20) → WD col 50 → Tree B S1 at row 36

    // Row 20: BitSelect tiles for rs extraction
    // rs_bit0 at col 47: WD(IR)@46, BitSelect@47, Const(0)@48
    ctx.place(ox + 46, oy + 20, TileType::WireDown); // IR from WR@(46,19)
    ctx.place(ox + 47, oy + 20, TileType::BitSelect); // rs_bit0
    ctx.place_with_value(ox + 48, oy + 20, TileType::Const, 0); // bit position 0
    // rs_bit1 at col 50: WD(IR)@49, BitSelect@50, Const(1)@51
    ctx.place(ox + 49, oy + 20, TileType::WireDown); // IR from WR@(49,19)
    ctx.place(ox + 50, oy + 20, TileType::BitSelect); // rs_bit1
    ctx.place_with_value(ox + 51, oy + 20, TileType::Const, 1); // bit position 1
    // rs_bit0 copy at col 53: WD(IR)@52, BitSelect@53, Const(0)@54
    ctx.place(ox + 52, oy + 20, TileType::WireDown); // IR from WR@(52,19)
    ctx.place(ox + 53, oy + 20, TileType::BitSelect); // rs_bit0 copy
    ctx.place_with_value(ox + 54, oy + 20, TileType::Const, 0); // bit position 0

    // WD chains: carry rs selector signals through register area (rows 21-32)
    // Cols 47, 50, 53 are outside register area (27-43), so they're in the
    // eastern zone that gets guarded later.
    for row in (oy + 21)..=(oy + 32) {
        ctx.place(ox + 47, row, TileType::WireDown); // rs_bit0 → Tree B S0
        ctx.place(ox + 50, row, TileType::WireDown); // rs_bit1 → Tree B S1
        ctx.place(ox + 53, row, TileType::WireDown); // rs_bit0 → Tree B S0
    }

    // =========================================================================
    // ROWS 29-32: OPERAND MUX TREE A (physical selectors, physical data)
    // =========================================================================
    // S0 at cols 31/39 and S1 at col 36 are now physical WD/WR from rd bits.
    // Data inputs are WD tiles fed by register WD chains.
    //   Row 29: WD(rd_bit0)@31, WD(rd_bit0)@39, WD chains + WL routing
    //   Row 30: D3(WD)@30, M23@31, D2(WD)@32, D1(WD)@38, M01@39, D0(WD)@40
    //   Row 31: WD@31, S1@36, WD@39
    //   Row 32: WD@31, WR@32-35, Root@36, WL@37-38, WD@39

    // Row 29: S0 selectors + register WD routing
    // Direct-column regs (Reg3@30, Reg1@38): WD continues straight down
    // Offset regs (Reg2@34→32, Reg0@42→40): WD then WL to shift 2 cols left
    for col in 27..=43 {
        ctx.guard(ox + col, oy + 29);
    }
    // S0 selectors: physical WD from rd_bit0 BitSelect chain (was Const)
    ctx.op_a_sel0_indices[0] = ctx.place(ox + 31, oy + 29, TileType::WireDown);
    ctx.op_a_sel0_indices[1] = ctx.place(ox + 39, oy + 29, TileType::WireDown);
    // Reg3 chain: col 30 straight down
    ctx.place(ox + 30, oy + 29, TileType::WireDown);
    // Reg1 chain: col 38 straight down
    ctx.place(ox + 38, oy + 29, TileType::WireDown);
    // Reg2 chain: col 34 → WL@33 → WL@32 (shift left 2)
    ctx.place(ox + 34, oy + 29, TileType::WireDown);
    ctx.place(ox + 33, oy + 29, TileType::WireLeft);
    ctx.place(ox + 32, oy + 29, TileType::WireLeft);
    // Reg0 chain: col 42 → WL@41 → WL@40 (shift left 2)
    ctx.place(ox + 42, oy + 29, TileType::WireDown);
    ctx.place(ox + 41, oy + 29, TileType::WireLeft);
    ctx.place(ox + 40, oy + 29, TileType::WireLeft);
    // rd_bit1 WD continues through row 29
    ctx.place(ox + 35, oy + 29, TileType::WireDown);

    // Row 30: WD data inputs as direct Mux neighbors (physical from registers)
    // D3(WD)@30, M23@31, D2(WD)@32, ..guards.., D1(WD)@38, M01@39, D0(WD)@40
    for col in 27..=43 {
        ctx.guard(ox + col, oy + 30);
    }
    ctx.op_a_data_indices[3] = ctx.place(ox + 30, oy + 30, TileType::WireDown);
    ctx.op_a_leaf_indices[0] = ctx.place(ox + 31, oy + 30, TileType::Mux); // M23
    ctx.op_a_data_indices[2] = ctx.place(ox + 32, oy + 30, TileType::WireDown);
    ctx.op_a_data_indices[1] = ctx.place(ox + 38, oy + 30, TileType::WireDown);
    ctx.op_a_leaf_indices[1] = ctx.place(ox + 39, oy + 30, TileType::Mux); // M01
    ctx.op_a_data_indices[0] = ctx.place(ox + 40, oy + 30, TileType::WireDown);
    // rd_bit1 WD continues through row 30
    ctx.place(ox + 35, oy + 30, TileType::WireDown);

    // Row 31: WD from leaves + S1 + VBusIn entries for register crossing
    for col in 27..=43 {
        ctx.guard(ox + col, oy + 31);
    }
    ctx.place(ox + 30, oy + 31, TileType::VBusIn); // Reg3 crossing entry
    ctx.place(ox + 31, oy + 31, TileType::WireDown);
    ctx.place(ox + 32, oy + 31, TileType::VBusIn); // D2 crossing entry
    // S1 selector: physical WR reads rd_bit1 from WD@col35 (was Const)
    ctx.place(ox + 35, oy + 31, TileType::WireDown); // rd_bit1 arrives from chain above
    ctx.op_a_sel1_idx = ctx.place(ox + 36, oy + 31, TileType::WireRight); // carries rd_bit1 to root
    ctx.place(ox + 38, oy + 31, TileType::VBusIn); // Reg1 crossing entry
    ctx.place(ox + 39, oy + 31, TileType::WireDown);
    ctx.place(ox + 40, oy + 31, TileType::WireDown); // D0 (no crossing needed)

    // Row 32: Root + WR/WL routing + crossing tiles
    for col in 27..=43 {
        ctx.guard(ox + col, oy + 32);
    }
    ctx.place(ox + 30, oy + 32, TileType::WireDown); // Reg3 (no conflict at root row)
    ctx.place(ox + 31, oy + 32, TileType::WireDown);
    ctx.place(ox + 32, oy + 32, TileType::WireCross); // D2: WR replaced with WireCross
    ctx.place(ox + 33, oy + 32, TileType::WireRight);
    ctx.place(ox + 34, oy + 32, TileType::WireRight);
    ctx.place(ox + 35, oy + 32, TileType::WireRight);
    ctx.op_a_root_idx = ctx.place(ox + 36, oy + 32, TileType::Mux);
    ctx.place(ox + 37, oy + 32, TileType::WireLeft);
    ctx.place(ox + 38, oy + 32, TileType::WireCrossVert); // Reg1: WL replaced with WireCrossVert
    ctx.place(ox + 39, oy + 32, TileType::WireDown);
    ctx.place(ox + 40, oy + 32, TileType::WireDown); // D0 (no conflict)

    // =========================================================================
    // ROW 33: TREE A OUTPUT ROUTING + crossing tiles
    // =========================================================================
    // Route Tree A root output (col 36) leftward via WL chain to col 10.
    // Cols 30 and 32 use WireCrossVert to pass vertical signals through the WL chain.
    for col in 8..=43 {
        ctx.guard(ox + col, oy + 33);
    }
    for col in 10..=29 {
        ctx.place(ox + col, oy + 33, TileType::WireLeft);
    }
    ctx.place(ox + 30, oy + 33, TileType::WireCrossVert); // Reg3: WL replaced with WireCrossVert
    for col in 31..=31 {
        ctx.place(ox + col, oy + 33, TileType::WireLeft);
    }
    ctx.place(ox + 32, oy + 33, TileType::WireCrossVert); // D2: WL replaced with WireCrossVert
    for col in 33..=35 {
        ctx.place(ox + col, oy + 33, TileType::WireLeft);
    }
    ctx.place(ox + 36, oy + 33, TileType::WireDown);
    ctx.place(ox + 38, oy + 33, TileType::VBusOut); // Reg1 crossing exit
    ctx.place(ox + 40, oy + 33, TileType::WireDown); // D0

    // =========================================================================
    // ROWS 34-37: A BUS FAN-OUT + WD DROPS (cols 0-45)
    // =========================================================================
    // A bus source: Tree A root output at WD@(10,34), from WL chain at row 33.
    // Fan out horizontally at row 34 via WL/WR chains to A bus cols.
    // Rows 35-36: WD passthrough at A bus cols.
    // Row 37: VBusIn at A bus cols 7+ (shift to high bits for B bus crossing at row 38).
    //         Col 2 stays WD (left of B bus range, no crossing needed).
    // Col 21 handled by ctrl_b spine (WireCross@34, WD@35-37).
    let a_bus_cols: [usize; 8] = [2, 7, 12, 17, 22, 27, 32, 37];
    let b_bus_cols: [usize; 7] = [4, 9, 14, 19, 24, 29, 34]; // excludes col 39 (clean)

    // Row 34: A bus WL/WR fan-out from WD@10
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 34);
    }
    for col in 2..=9 {
        ctx.place(ox + col, oy + 34, TileType::WireLeft);
    }
    ctx.place(ox + 10, oy + 34, TileType::WireDown);
    for col in 11..=37 {
        if col != 21 {
            ctx.place(ox + col, oy + 34, TileType::WireRight);
        }
    }

    // Rows 35-36: WD at A bus cols
    for row in (oy + 35)..=(oy + 36) {
        for col in 0..=45 {
            ctx.guard(ox + col, row);
        }
        for &col in &a_bus_cols {
            if col != 21 {
                ctx.place(ox + col, row, TileType::WireDown);
            }
        }
    }

    // Row 37: VBusIn at A bus cols 7+ (prep for B bus crossing), WD at col 2
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 37);
    }
    ctx.place(ox + 2, oy + 37, TileType::WireDown);
    for &col in &a_bus_cols[1..] {
        if col != 21 {
            ctx.place(ox + col, oy + 37, TileType::VBusIn);
        }
    }

    // =========================================================================
    // ROWS 33-37: NEW OPERAND MUX TREE B (relocated to cols 46-54)
    // =========================================================================
    // Selects reg[rs]. Physical selectors via BitSelect WD chains from row 20.
    // Data inputs are software-written Consts (register values written by execute.rs).
    //
    // Layout (6-column gap between leaf muxes for selector routing):
    //     46     47     48     49     50     51     52     53     54
    // 33: Guard  S0(WD) Guard  Guard  WD     Guard  Guard  S0(WD) Guard
    // 34: D3     M23    D2     Guard  WD     Guard  D1     M01    D0
    // 35: Guard  WD     Guard  Guard  WD     Guard  Guard  WD     Guard
    // 36: Guard  WD     Guard  Guard  S1(WD) Guard  Guard  WD     Guard
    // 37: Guard  WD     WR     WR     Root   WL     WL     WD     Guard

    // Row 33: S0 selectors arrive via WD from rs_bit0 BitSelect chains
    // (Tree A output WL chain at row 33 occupies cols 10-36 only, no conflict)
    ctx.guard(ox + 46, oy + 33);
    ctx.op_b_sel0_indices[0] = ctx.place(ox + 47, oy + 33, TileType::WireDown); // rs_bit0 → S0 M23
    ctx.guard(ox + 48, oy + 33);
    ctx.guard(ox + 49, oy + 33);
    ctx.place(ox + 50, oy + 33, TileType::WireDown); // rs_bit1 continues
    ctx.guard(ox + 51, oy + 33);
    ctx.guard(ox + 52, oy + 33);
    ctx.op_b_sel0_indices[1] = ctx.place(ox + 53, oy + 33, TileType::WireDown); // rs_bit0 → S0 M01
    ctx.guard(ox + 54, oy + 33);

    // Row 34: Data Consts + Leaf Muxes
    ctx.op_b_data_indices[3] = ctx.place_with_value(ox + 46, oy + 34, TileType::Const, 0); // D3
    ctx.op_b_leaf_indices[0] = ctx.place(ox + 47, oy + 34, TileType::Mux); // M23
    ctx.op_b_data_indices[2] = ctx.place_with_value(ox + 48, oy + 34, TileType::Const, 0); // D2
    ctx.guard(ox + 49, oy + 34);
    ctx.place(ox + 50, oy + 34, TileType::WireDown); // rs_bit1 continues
    ctx.guard(ox + 51, oy + 34);
    ctx.op_b_data_indices[1] = ctx.place_with_value(ox + 52, oy + 34, TileType::Const, 0); // D1
    ctx.op_b_leaf_indices[1] = ctx.place(ox + 53, oy + 34, TileType::Mux); // M01
    ctx.op_b_data_indices[0] = ctx.place_with_value(ox + 54, oy + 34, TileType::Const, 0); // D0

    // Row 35: WD from leaf muxes
    ctx.guard(ox + 46, oy + 35);
    ctx.place(ox + 47, oy + 35, TileType::WireDown); // M23 output
    ctx.guard(ox + 48, oy + 35);
    ctx.guard(ox + 49, oy + 35);
    ctx.place(ox + 50, oy + 35, TileType::WireDown); // rs_bit1 continues
    ctx.guard(ox + 51, oy + 35);
    ctx.guard(ox + 52, oy + 35);
    ctx.place(ox + 53, oy + 35, TileType::WireDown); // M01 output
    ctx.guard(ox + 54, oy + 35);

    // Row 36: WD continues + S1 from rs_bit1 WD chain
    ctx.guard(ox + 46, oy + 36);
    ctx.place(ox + 47, oy + 36, TileType::WireDown); // M23 continues
    ctx.guard(ox + 48, oy + 36);
    ctx.guard(ox + 49, oy + 36);
    ctx.op_b_sel1_idx = ctx.place(ox + 50, oy + 36, TileType::WireDown); // rs_bit1 → S1
    ctx.guard(ox + 51, oy + 36);
    ctx.guard(ox + 52, oy + 36);
    ctx.place(ox + 53, oy + 36, TileType::WireDown); // M01 continues
    ctx.guard(ox + 54, oy + 36);

    // Row 37: Root Mux + WR/WL routing from leaf outputs
    ctx.guard(ox + 46, oy + 37);
    ctx.place(ox + 47, oy + 37, TileType::WireDown); // M23 output arrives
    ctx.place(ox + 48, oy + 37, TileType::WireRight); // carries M23 rightward
    ctx.place(ox + 49, oy + 37, TileType::WireRight);
    ctx.op_b_root_idx = ctx.place(ox + 50, oy + 37, TileType::Mux); // Root
    ctx.place(ox + 51, oy + 37, TileType::WireLeft); // carries M01 leftward
    ctx.place(ox + 52, oy + 37, TileType::WireLeft);
    ctx.place(ox + 53, oy + 37, TileType::WireDown); // M01 output arrives
    ctx.guard(ox + 54, oy + 37);

    // =========================================================================
    // ROW 38: B BUS WL CHAIN WITH VERTICAL CROSSINGS
    // =========================================================================
    // B bus from Tree B root@50 flows leftward via WL to col 4.
    // WireCrossVert at A bus cols + ctrl_b@21 allows vertical crossing.
    for col in 0..=54 {
        ctx.guard(ox + col, oy + 38);
    }
    ctx.place(ox + 2, oy + 38, TileType::WireDown); // A bus (left of B bus range)
    ctx.place(ox + 50, oy + 38, TileType::WireDown); // Tree B root output continues
    for col in 4..=49 {
        if a_bus_cols.contains(&col) {
            ctx.place(ox + col, oy + 38, TileType::WireCrossVert);
        } else if col != 21 {
            // ctrl_b spine handles col 21
            ctx.place(ox + col, oy + 38, TileType::WireLeft);
        }
    }

    // =========================================================================
    // ROWS 39-42: BUS CLEANUP + WD PASSTHROUGH
    // =========================================================================
    // Row 39: VBusOut at A cols (extract clean A bus from high bits).
    //         VBusIn at B cols (strip high-bit contamination).
    //         VBusOut@21 by ctrl_b spine (end of extended crossing zone).
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 39);
    }
    ctx.place(ox + 2, oy + 39, TileType::WireDown); // A col 2 (never crossed)
    for &col in &a_bus_cols[1..] {
        if col != 21 {
            ctx.place(ox + col, oy + 39, TileType::VBusOut);
        }
    }
    for &col in &b_bus_cols {
        ctx.place(ox + col, oy + 39, TileType::VBusIn);
    }
    ctx.place(ox + 39, oy + 39, TileType::WireDown); // B col 39 (never crossed)

    // Row 40: VBusOut at B cols (clean B bus). WD at A cols + B col 39.
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 40);
    }
    for &col in &a_bus_cols {
        if col != 21 {
            ctx.place(ox + col, oy + 40, TileType::WireDown);
        }
    }
    for &col in &b_bus_cols {
        ctx.place(ox + col, oy + 40, TileType::VBusOut);
    }
    ctx.place(ox + 39, oy + 40, TileType::WireDown);

    // Rows 41-42: WD passthrough — all bus signals descend to ALU
    for row in (oy + 41)..=(oy + 42) {
        for col in 0..=45 {
            ctx.guard(ox + col, row);
        }
        for &col in &a_bus_cols {
            if col != 21 {
                ctx.place(ox + col, row, TileType::WireDown);
            }
        }
        for &col in &b_bus_cols {
            ctx.place(ox + col, row, TileType::WireDown);
        }
        ctx.place(ox + 39, row, TileType::WireDown);
    }

    // =========================================================================
    // ROW 43: SPREAD ALU (8 units, 5-column spacing, offset +1)
    // =========================================================================
    // Each unit: [guard, WD(A_bus), ALU_tile, WD(B_bus)/Const, guard]
    // D7(Shr)→D0(AddCarry) from left to right.
    // Col 21 = ctrl_b WD (placed by spine), sacrifices Unit 4 left guard.
    let alu_out_cols: [usize; 8] = [3, 8, 13, 18, 23, 28, 33, 38];
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 43);
    }

    // Unit 0: Shr (D7) — cols 1-5
    ctx.place(ox + 2, oy + 43, TileType::WireDown);
    ctx.alu_tile_indices[7] = ctx.place(ox + 3, oy + 43, TileType::Shr);
    ctx.place_with_value(ox + 4, oy + 43, TileType::Const, 1);

    // Unit 1: Shl (D6) — cols 6-10
    ctx.place(ox + 7, oy + 43, TileType::WireDown);
    ctx.alu_tile_indices[6] = ctx.place(ox + 8, oy + 43, TileType::Shl);
    ctx.place_with_value(ox + 9, oy + 43, TileType::Const, 1);

    // Unit 2: Not (D5) — cols 11-15
    ctx.place(ox + 12, oy + 43, TileType::WireDown);
    ctx.alu_tile_indices[5] = ctx.place(ox + 13, oy + 43, TileType::Not);
    ctx.place_with_value(ox + 14, oy + 43, TileType::Const, 0);

    // Unit 3: Xor (D4) — cols 16-20
    ctx.place(ox + 17, oy + 43, TileType::WireDown);
    ctx.alu_tile_indices[4] = ctx.place(ox + 18, oy + 43, TileType::Xor);
    ctx.place(ox + 19, oy + 43, TileType::WireDown);

    // Unit 4: Or (D3) — cols 21-25 (col 21 = ctrl_b WD from spine)
    ctx.place(ox + 22, oy + 43, TileType::WireDown);
    ctx.alu_tile_indices[3] = ctx.place(ox + 23, oy + 43, TileType::Or);
    ctx.place(ox + 24, oy + 43, TileType::WireDown);

    // Unit 5: And (D2) — cols 26-30
    ctx.place(ox + 27, oy + 43, TileType::WireDown);
    ctx.alu_tile_indices[2] = ctx.place(ox + 28, oy + 43, TileType::And);
    ctx.place(ox + 29, oy + 43, TileType::WireDown);

    // Unit 6: Sub (D1) — cols 31-35
    ctx.place(ox + 32, oy + 43, TileType::WireDown);
    ctx.alu_tile_indices[1] = ctx.place(ox + 33, oy + 43, TileType::SubBorrow);
    ctx.place(ox + 34, oy + 43, TileType::WireDown);

    // Unit 7: AddCarry (D0) — cols 36-40
    ctx.place(ox + 37, oy + 43, TileType::WireDown);
    ctx.alu_tile_indices[0] = ctx.place(ox + 38, oy + 43, TileType::AddCarry);
    ctx.place(ox + 39, oy + 43, TileType::WireDown);

    // Data indices: ALU tiles are the physical data source (no software staging)
    ctx.alu_result_data_indices = ctx.alu_tile_indices;

    // =========================================================================
    // ROWS 44-55: CUSTOM RESULT TREE (physical data, software selectors)
    // =========================================================================
    // ALU outputs descend via WD to leaf muxes. Leaf→Mid→Root via WR/WL.
    // S0/S1/S2 are Const tiles written by software before propagate_combinational.
    // ctrl_b at col 21 (WD, then VBusIn/WireCrossVert/VBusOut at root level).

    // Rows 44-45: WD passthrough from ALU outputs + ctrl_b
    for row in (oy + 44)..=(oy + 45) {
        for col in 0..=45 {
            ctx.guard(ox + col, row);
        }
        for &col in &alu_out_cols {
            ctx.place(ox + col, row, TileType::WireDown);
        }
    }

    // Row 46: S0 Const selectors at leaf mux cols + WD data passthrough
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 46);
    }
    for &col in &alu_out_cols {
        ctx.place(ox + col, oy + 46, TileType::WireDown);
    }
    // S0 selectors: software-written Consts (physical L1 routing has cross-contamination)
    ctx.alu_result_sel0_indices[0] = ctx.place(ox + 5, oy + 46, TileType::Const);
    ctx.alu_result_sel0_indices[1] = ctx.place(ox + 15, oy + 46, TileType::Const);
    ctx.alu_result_sel0_indices[2] = ctx.place(ox + 25, oy + 46, TileType::Const);
    ctx.alu_result_sel0_indices[3] = ctx.place(ox + 35, oy + 46, TileType::Const);

    // Row 47: Leaf muxes + WR/WL routing
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 47);
    }
    // M67@5: WR@4(←WD@3=D7), Mux@5(up=S0), WL@6→WL@7(←WD@8=D6)
    ctx.place(ox + 3, oy + 47, TileType::WireDown);
    ctx.place(ox + 4, oy + 47, TileType::WireRight);
    ctx.alu_result_leaf_indices[0] = ctx.place(ox + 5, oy + 47, TileType::Mux);
    ctx.place(ox + 6, oy + 47, TileType::WireLeft);
    ctx.place(ox + 7, oy + 47, TileType::WireLeft);
    ctx.place(ox + 8, oy + 47, TileType::WireDown);
    // M45@15: WR@14(←WD@13=D5), Mux@15(up=S0), WL@16→WL@17(←WD@18=D4)
    ctx.place(ox + 13, oy + 47, TileType::WireDown);
    ctx.place(ox + 14, oy + 47, TileType::WireRight);
    ctx.alu_result_leaf_indices[1] = ctx.place(ox + 15, oy + 47, TileType::Mux);
    ctx.place(ox + 16, oy + 47, TileType::WireLeft);
    ctx.place(ox + 17, oy + 47, TileType::WireLeft);
    ctx.place(ox + 18, oy + 47, TileType::WireDown);
    // M23@25: WR@24(←WD@23=D3), Mux@25(up=S0), WL@26→WL@27(←WD@28=D2)
    ctx.place(ox + 23, oy + 47, TileType::WireDown);
    ctx.place(ox + 24, oy + 47, TileType::WireRight);
    ctx.alu_result_leaf_indices[2] = ctx.place(ox + 25, oy + 47, TileType::Mux);
    ctx.place(ox + 26, oy + 47, TileType::WireLeft);
    ctx.place(ox + 27, oy + 47, TileType::WireLeft);
    ctx.place(ox + 28, oy + 47, TileType::WireDown);
    // M01@35: WR@34(←WD@33=D1), Mux@35(up=S0), WL@36→WL@37(←WD@38=D0)
    ctx.place(ox + 33, oy + 47, TileType::WireDown);
    ctx.place(ox + 34, oy + 47, TileType::WireRight);
    ctx.alu_result_leaf_indices[3] = ctx.place(ox + 35, oy + 47, TileType::Mux);
    ctx.place(ox + 36, oy + 47, TileType::WireLeft);
    ctx.place(ox + 37, oy + 47, TileType::WireLeft);
    ctx.place(ox + 38, oy + 47, TileType::WireDown);

    // Row 48: WD from leaf muxes
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 48);
    }
    for &col in &[5, 15, 25, 35] {
        ctx.place(ox + col, oy + 48, TileType::WireDown);
    }

    // Row 49: S1 Consts at mid mux cols + WD continues
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 49);
    }
    for &col in &[5, 15, 25, 35] {
        ctx.place(ox + col, oy + 49, TileType::WireDown);
    }
    // S1 selectors: software-written Consts (physical L1 routing has cross-contamination)
    ctx.alu_result_sel1_indices[0] = ctx.place(ox + 10, oy + 49, TileType::Const);
    ctx.alu_result_sel1_indices[1] = ctx.place(ox + 30, oy + 49, TileType::Const);

    // Row 50: Mid muxes + WR/WL routing
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 50);
    }
    // M4567@10: WD@5→WR@6-9→Mux@10←WL@11-14←WD@15
    ctx.place(ox + 5, oy + 50, TileType::WireDown);
    for col in 6..=9 {
        ctx.place(ox + col, oy + 50, TileType::WireRight);
    }
    ctx.alu_result_mid_indices[0] = ctx.place(ox + 10, oy + 50, TileType::Mux);
    for col in 11..=14 {
        ctx.place(ox + col, oy + 50, TileType::WireLeft);
    }
    ctx.place(ox + 15, oy + 50, TileType::WireDown);
    // M0123@30: WD@25→WR@26-29→Mux@30←WL@31-34←WD@35
    ctx.place(ox + 25, oy + 50, TileType::WireDown);
    for col in 26..=29 {
        ctx.place(ox + col, oy + 50, TileType::WireRight);
    }
    ctx.alu_result_mid_indices[1] = ctx.place(ox + 30, oy + 50, TileType::Mux);
    for col in 31..=34 {
        ctx.place(ox + col, oy + 50, TileType::WireLeft);
    }
    ctx.place(ox + 35, oy + 50, TileType::WireDown);

    // Row 51: WD from mid muxes
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 51);
    }
    ctx.place(ox + 10, oy + 51, TileType::WireDown);
    ctx.place(ox + 30, oy + 51, TileType::WireDown);

    // Row 52: S2 Const + WD continues
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 52);
    }
    ctx.place(ox + 10, oy + 52, TileType::WireDown);
    // S2 selector: software-written Const (physical L1 routing has cross-contamination)
    ctx.alu_result_sel2_idx = ctx.place(ox + 20, oy + 52, TileType::Const);
    ctx.place(ox + 30, oy + 52, TileType::WireDown);

    // Row 53: WD passthrough (VBusIn@21 for ctrl_b root crossing placed by spine)
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 53);
    }
    ctx.place(ox + 10, oy + 53, TileType::WireDown);
    ctx.place(ox + 20, oy + 53, TileType::WireDown); // S2 → Root
    ctx.place(ox + 30, oy + 53, TileType::WireDown);

    // Row 54: Root mux + WR/WL routing
    for col in 0..=45 {
        ctx.guard(ox + col, oy + 54);
    }
    ctx.place(ox + 10, oy + 54, TileType::WireDown);
    for col in 11..=19 {
        ctx.place(ox + col, oy + 54, TileType::WireRight);
    }
    ctx.alu_result_root_idx = ctx.place(ox + 20, oy + 54, TileType::Mux);
    // col 21: WireCrossVert placed by ctrl_b spine (root WL + ctrl_b vertical)
    for col in 22..=29 {
        ctx.place(ox + col, oy + 54, TileType::WireLeft);
    }
    ctx.place(ox + 30, oy + 54, TileType::WireDown);

    // Rows 55-56: Guard rows (VBusOut@21 at row 55 placed by ctrl_b spine)
    for row in (oy + 55)..=(oy + 56) {
        for col in 0..=45 {
            ctx.guard(ox + col, row);
        }
    }

    // =========================================================================
    // ROW 57-58: FLAGS
    // =========================================================================
    // Z flag: Zero tile checks result, Mux selects update or hold
    // C flag: AddCarry computes carry, BitSelect extracts carry bit, Mux selects

    // Guard row
    for col in 0..30 {
        ctx.guard(ox + col, oy + 57);
    }

    // Z flag circuit at row 58-59
    ctx.guard(ox + 20, oy + 58);
    ctx.guard(ox + 21, oy + 58);
    ctx.flag_z_update_idx = ctx.place_with_value(ox + 22, oy + 58, TileType::Const, 0);
    ctx.guard(ox + 23, oy + 58);
    ctx.guard(ox + 24, oy + 58);

    ctx.flag_z_result_idx = ctx.place_with_value(ox + 20, oy + 59, TileType::Const, 0);
    let flag_z_zero_idx = ctx.place(ox + 21, oy + 59, TileType::Zero);
    ctx.flag_z_zero_idx = flag_z_zero_idx;
    let flag_z_mux_idx = ctx.place(ox + 22, oy + 59, TileType::Mux);
    ctx.flag_z_mux_idx = flag_z_mux_idx;
    ctx.flag_z_idx = ctx.place_with_value(ox + 23, oy + 59, TileType::Register8, 0);
    ctx.guard(ox + 24, oy + 59);

    // C flag circuit at row 60-61
    ctx.guard(ox + 20, oy + 60);
    ctx.flag_c_update_idx = ctx.place_with_value(ox + 21, oy + 60, TileType::Const, 0);
    ctx.guard(ox + 22, oy + 60);
    ctx.guard(ox + 23, oy + 60);
    ctx.guard(ox + 24, oy + 60);

    ctx.flag_c_carry_idx = ctx.place_with_value(ox + 20, oy + 61, TileType::Const, 0);
    let flag_c_mux_idx = ctx.place(ox + 21, oy + 61, TileType::Mux);
    ctx.flag_c_mux_idx = flag_c_mux_idx;
    ctx.flag_c_idx = ctx.place_with_value(ox + 22, oy + 61, TileType::Register8, 0);
    ctx.guard(ox + 23, oy + 61);
    ctx.guard(ox + 24, oy + 61);

    for col in 20..=24 {
        ctx.guard(ox + col, oy + 62);
    }

    // =========================================================================
    // ROWS 63-70: PHYSICAL BRANCH GATE NETWORK
    // =========================================================================
    // Computes: branch_taken = is_jmp | (is_jz & flag_z) | (is_jnz & !flag_z)
    //
    // Inputs arriving at row 63:
    //   flag_z at col 23 (WD chain from Register8@(23,59) via Phase 3b)
    //   ctrl_b at col 25 (WD spine from Phase 3b)
    //
    // Row 63: flag_z WD extension + ctrl_b WR fan-out east (cols 26-35)
    ctx.place(ox + 23, oy + 63, TileType::WireDown); // flag_z
    // ctrl_b WD@(25,63) already placed by Phase 3b
    for col in 26..=35 {
        ctx.place(ox + col, oy + 63, TileType::WireRight); // ctrl_b fan-out
    }

    // Row 64: Bit extraction — 3 triplets [WD(ctrl_b), BitSelect, Const(bit)]
    ctx.place(ox + 23, oy + 64, TileType::WireDown); // flag_z continues
    // is_jz (bit 1)
    ctx.place(ox + 29, oy + 64, TileType::WireDown); // ctrl_b from WR@(29,63)
    ctx.place(ox + 30, oy + 64, TileType::BitSelect);
    ctx.place_with_value(ox + 31, oy + 64, TileType::Const, 1);
    // is_jnz (bit 2)
    ctx.place(ox + 32, oy + 64, TileType::WireDown); // ctrl_b from WR@(32,63)
    ctx.place(ox + 33, oy + 64, TileType::BitSelect);
    ctx.place_with_value(ox + 34, oy + 64, TileType::Const, 2);
    // is_jmp (bit 0)
    ctx.place(ox + 35, oy + 64, TileType::WireDown); // ctrl_b from WR@(35,63)
    ctx.place(ox + 36, oy + 64, TileType::BitSelect);
    ctx.place_with_value(ox + 37, oy + 64, TileType::Const, 0);

    // Row 65: And_jz = flag_z & is_jz
    //   flag_z WR chain east: cols 24-28
    //   And@29 reads left=WR(flag_z)@28, right=WD(is_jz)@30
    ctx.place(ox + 23, oy + 65, TileType::WireDown); // flag_z continues
    for col in 24..=28 {
        ctx.place(ox + col, oy + 65, TileType::WireRight); // flag_z east
    }
    ctx.place(ox + 29, oy + 65, TileType::And); // And_jz
    ctx.place(ox + 30, oy + 65, TileType::WireDown); // is_jz from BS@(30,64)
    ctx.place(ox + 33, oy + 65, TileType::WireDown); // is_jnz continues
    ctx.place(ox + 36, oy + 65, TileType::WireDown); // is_jmp continues

    // Row 66: VBusIn — shifts And_jz to high 32 bits for crossing
    ctx.place(ox + 23, oy + 66, TileType::WireDown); // flag_z continues
    ctx.place(ox + 29, oy + 66, TileType::VBusIn); // And_jz → high 32
    ctx.place(ox + 33, oy + 66, TileType::WireDown); // is_jnz continues
    ctx.place(ox + 36, oy + 66, TileType::WireDown); // is_jmp continues

    // Row 67: flag_z WR east + WireCross + Zero(!flag_z)
    //   WireCross@29: horiz=flag_z (from WR chain), vert=And_jz (high bits)
    //   Zero@32: reads packed left → !flag_z (safe: complementary masking)
    ctx.place(ox + 23, oy + 67, TileType::WireDown); // flag_z continues
    for col in 24..=28 {
        ctx.place(ox + col, oy + 67, TileType::WireRight); // flag_z east
    }
    ctx.place(ox + 29, oy + 67, TileType::WireCross); // flag_z horiz + And_jz vert
    ctx.place(ox + 30, oy + 67, TileType::WireRight); // packed east
    ctx.place(ox + 31, oy + 67, TileType::WireRight); // packed east
    ctx.place(ox + 32, oy + 67, TileType::Zero); // !flag_z
    ctx.place(ox + 33, oy + 67, TileType::WireDown); // is_jnz continues
    ctx.place(ox + 34, oy + 67, TileType::WireRight); // is_jnz east copy
    ctx.place(ox + 36, oy + 67, TileType::WireDown); // is_jmp continues

    // Row 68: VBusOut + And_jnz = !flag_z & is_jnz
    //   VBusOut@29: extracts And_jz back to low 32 bits
    //   And@33: left=WD(!flag_z)@32, right=WD(is_jnz)@34
    ctx.place(ox + 29, oy + 68, TileType::VBusOut); // And_jz low 32
    ctx.place(ox + 30, oy + 68, TileType::WireRight); // And_jz east
    ctx.place(ox + 32, oy + 68, TileType::WireDown); // !flag_z from Zero@(32,67)
    ctx.place(ox + 33, oy + 68, TileType::And); // And_jnz = !flag_z & is_jnz
    ctx.place(ox + 34, oy + 68, TileType::WireDown); // is_jnz from WR@(34,67)
    ctx.place(ox + 36, oy + 68, TileType::WireDown); // is_jmp continues

    // Row 69: Or(And_jz, And_jnz)
    //   Or@31: left=WD(And_jz)@30, right=WL(And_jnz)@32
    ctx.place(ox + 30, oy + 69, TileType::WireDown); // And_jz from WR@(30,68)
    ctx.place(ox + 31, oy + 69, TileType::Or); // And_jz | And_jnz
    ctx.place(ox + 32, oy + 69, TileType::WireLeft); // And_jnz from WD@(33,69)
    ctx.place(ox + 33, oy + 69, TileType::WireDown); // And_jnz from And@(33,68)
    ctx.place(ox + 36, oy + 69, TileType::WireDown); // is_jmp continues

    // Row 70: branch_taken = Or_result | is_jmp
    //   Or@32: left=WD(Or1)@31, right=WL(is_jmp)@33
    ctx.place(ox + 31, oy + 70, TileType::WireDown); // Or1 from Or@(31,69)
    let branch_taken_idx = ctx.place(ox + 32, oy + 70, TileType::Or); // branch_taken
    ctx.place(ox + 33, oy + 70, TileType::WireLeft); // is_jmp from WL@(34,70)
    ctx.place(ox + 34, oy + 70, TileType::WireLeft); // is_jmp from WL@(35,70)
    ctx.place(ox + 35, oy + 70, TileType::WireLeft); // is_jmp from WD@(36,70)
    ctx.place(ox + 36, oy + 70, TileType::WireDown); // is_jmp from WD@(36,69)

    // =========================================================================
    // ROM ENTRIES
    // =========================================================================
    let rom_start = oy + 76;
    for addr in 0..effective_rom {
        let val = if addr < program.len() {
            program[addr] as u64
        } else {
            0
        };
        let idx = ctx.place_with_value(
            ox + (addr % 8),
            rom_start + (addr / 8),
            TileType::Const,
            val,
        );
        ctx.rom_indices.push(idx);
    }

    // Guard row
    let guard_row = rom_start + 2;
    for col in 0..8 {
        ctx.sim.set_tile(ox + col, guard_row, TileType::Const);
        ctx.sim.set_logic_value(ox + col, guard_row, 0);
    }

    // RAM
    let ram_start = guard_row + 1;
    let effective_ram = ram_size.min(64);
    for addr in 0..effective_ram {
        let idx = ctx.place_with_value(ox + (addr % 8), ram_start + (addr / 8), TileType::Const, 0);
        ctx.ram_indices.push(idx);
    }

    // =========================================================================
    // PHASE 3b: CTRL_B ROUTING SPINE — col 21 from row 18 to row 63
    // =========================================================================
    // Extends the ctrl_b WD chain (from BitSelect extraction at rows 13-17)
    // downward through the entire CPU, with crossings and detours as needed.
    // This is prep infrastructure for a future fully-physical branch gate network.

    // Row 18: VBusIn — shifts ctrl_b to high 32 bits before IR fan-out crossing
    ctx.place(ox + 21, oy + 18, TileType::VBusIn);

    // Row 19: WireCross — horizontal=IR fan-out (from left), vertical=ctrl_b (high bits)
    // Overwrites WR tile in the IR fan-out chain. WireCross passes both signals:
    //   bits 0-31 = IR from left (horizontal bus), bits 32-63 = ctrl_b from up (vertical bus)
    ctx.place(ox + 21, oy + 19, TileType::WireCross);

    // Row 20: WD — ctrl_b stays in high bits through extended crossing zone (rows 18-39)
    ctx.place(ox + 21, oy + 20, TileType::WireDown);

    // Rows 21-56: Extended crossing zone at col 21.
    // VBusIn@18 shifted ctrl_b to high bits. It stays packed through rows 20-38.
    // Crossings: Tree A WL @33, A bus WR @34, B bus WL @38.
    // VBusOut@39 extracts clean ctrl_b. Then local crossing at rows 53-55
    // for the result tree root WL chain.
    for row in (oy + 21)..=(oy + 56) {
        if row == oy + 33 {
            // WireCrossVert — Tree A WL from right, ctrl_b in high bits
            ctx.place(ox + 21, row, TileType::WireCrossVert);
        } else if row == oy + 34 {
            // WireCross — A bus WR fan-out from left, ctrl_b in high bits
            ctx.place(ox + 21, row, TileType::WireCross);
        } else if row == oy + 38 {
            // WireCrossVert — B bus WL from right, ctrl_b in high bits
            ctx.place(ox + 21, row, TileType::WireCrossVert);
        } else if row == oy + 39 {
            // VBusOut — extract clean ctrl_b from extended crossing zone
            ctx.place(ox + 21, row, TileType::VBusOut);
        } else if row == oy + 53 {
            // VBusIn — ctrl_b → high bits for result tree root WL crossing
            ctx.place(ox + 21, row, TileType::VBusIn);
        } else if row == oy + 54 {
            // WireCrossVert — result tree root WL from right, ctrl_b high
            ctx.place(ox + 21, row, TileType::WireCrossVert);
        } else if row == oy + 55 {
            // VBusOut — extract clean ctrl_b after result tree root crossing
            ctx.place(ox + 21, row, TileType::VBusOut);
        } else {
            ctx.place(ox + 21, row, TileType::WireDown);
        }
    }

    // Row 57: WD@21 then WR detour rightward to col 25 (avoid flag_z at col 21 row 59)
    // Row 57 is entirely guards from cols 0..30.
    ctx.place(ox + 21, oy + 57, TileType::WireDown); // ctrl_b arrives
    ctx.place(ox + 22, oy + 57, TileType::WireRight); // detour starts
    ctx.place(ox + 23, oy + 57, TileType::WireRight);
    ctx.place(ox + 24, oy + 57, TileType::WireRight);
    ctx.place(ox + 25, oy + 57, TileType::WireRight); // ctrl_b arrives at col 25

    // Rows 58-63: WD chain at col 25 (ctrl_b descends to future branch area)
    for row in (oy + 58)..=(oy + 63) {
        ctx.place(ox + 25, row, TileType::WireDown);
    }
    // Guard col 26 alongside the spine to prevent Wire leakage from the east
    for row in (oy + 58)..=(oy + 63) {
        if ctx.sim.tile_type_xy(ox + 26, row) == TileType::Wire {
            ctx.guard(ox + 26, row);
        }
    }

    // =========================================================================
    // PHASE 3b: FLAG_Z ROUTING — Register8@(23,59) down to row 62
    // =========================================================================
    // flag_z Register8 is at (23, 59). Carry its output downward for the
    // future branch gate network.
    // Row 60: col 23 currently is guard → place WD
    // Row 61: col 23 currently is guard → place WD
    // Row 62: col 23 guard → place WD (flag_z arrives at row 62 for later WR routing)
    ctx.place(ox + 23, oy + 60, TileType::WireDown); // flag_z continues
    ctx.place(ox + 23, oy + 61, TileType::WireDown);
    ctx.place(ox + 23, oy + 62, TileType::WireDown); // flag_z at row 62

    // =========================================================================
    // PHASE 3b: JUMP_TARGET DUPLICATION — east of spine
    // =========================================================================
    // Extend IR fan-out at row 19 to col 60 (currently ends at col 54,
    // cols 55-56 are guards). Place And@(61,19) + Const(0x3F)@(62,19)
    // for a duplicate jump_target east of the main circuit.
    for col in 55..=60 {
        ctx.place(ox + col, oy + 19, TileType::WireRight); // extend IR fan-out
    }
    let _jump_target_east_idx = ctx.place(ox + 61, oy + 19, TileType::And);
    ctx.place_with_value(ox + 62, oy + 19, TileType::Const, 0x3F);
    // Guard col 63 to prevent Wire leakage from the far east
    ctx.guard(ox + 63, oy + 19);

    // WD chain at col 61 from row 20 to row 63 (jump_target descends)
    for row in (oy + 20)..=(oy + 63) {
        ctx.place(ox + 61, row, TileType::WireDown);
    }
    // Guard col 62 alongside the jump_target spine
    for row in (oy + 20)..=(oy + 63) {
        if ctx.sim.tile_type_xy(ox + 62, row) == TileType::Wire {
            ctx.guard(ox + 62, row);
        }
    }
    // Guard col 60 alongside the jump_target spine
    for row in (oy + 20)..=(oy + 63) {
        if ctx.sim.tile_type_xy(ox + 60, row) == TileType::Wire {
            ctx.guard(ox + 60, row);
        }
    }

    // =========================================================================
    // DEAD ZONE GUARD: Fill unplaced Wire tiles with Const(0) guards
    // =========================================================================
    // Two zones need guarding:
    // 1. Western dead zone: cols 0-26 at rows 20-56 (original)
    // 2. Eastern zone: cols 44-55 at rows 19-38 (new Tree B area + routing)
    //
    // Wire tiles here OR their neighbors, causing silent signal contamination.
    // Guard any remaining Wire tile in these zones to prevent this.
    for row in (oy + 20)..=(oy + 56) {
        for col in ox..=(ox + 26) {
            if ctx.sim.tile_type_xy(col, row) == TileType::Wire {
                ctx.guard(col, row);
            }
        }
    }
    // Eastern zone: guard cols 44-55 around new Tree B and rs selector WD chains.
    // Rows 19-38 covers IR fan-out extension, BitSelect tiles, WD chains, Tree B, output.
    for row in (oy + 19)..=(oy + 38) {
        for col in (ox + 44)..=(ox + 55) {
            if ctx.sim.tile_type_xy(col, row) == TileType::Wire {
                ctx.guard(col, row);
            }
        }
    }
    // Phase 3b: extended eastern zone around jump_target spine (cols 56-63, rows 19-63)
    for row in (oy + 19)..=(oy + 63) {
        for col in (ox + 56)..=(ox + 63) {
            if ctx.sim.tile_type_xy(col, row) == TileType::Wire {
                ctx.guard(col, row);
            }
        }
    }
    // Phase 3b: guard western spine detour area (cols 20-26 at rows 57-63)
    for row in (oy + 57)..=(oy + 63) {
        for col in (ox + 20)..=(ox + 26) {
            if ctx.sim.tile_type_xy(col, row) == TileType::Wire {
                ctx.guard(col, row);
            }
        }
    }
    // Phase 4: guard gate network area (cols 20-40 at rows 63-75)
    // Extended to row 75 to cover PC update zone (rows 71-74) before ROM at row 76
    for row in (oy + 63)..=(oy + 75) {
        for col in (ox + 20)..=(ox + 40) {
            if ctx.sim.tile_type_xy(col, row) == TileType::Wire {
                ctx.guard(col, row);
            }
        }
    }

    // =========================================================================
    // PHASE 5: PHYSICAL PC UPDATE CIRCUIT (Multi-Layer)
    // =========================================================================
    // PC feedback loop: Register8 → L1 down → Add(PC,1) → Mux(branch_taken) → L1 up → ViaUp → Register8
    // Timing: settles during tick_with_delays() (MAX_DELTA=500), captured at next rising edge.

    // --- Phase 5a: L1 guard sweep ---
    // Guard entire L1 in CPU region with Const(0) to prevent Wire default leakage.
    // Routing tiles placed afterward will overwrite specific guard positions.
    for y in oy..oy + 80 {
        for x in ox..ox + 66 {
            ctx.place_on_layer(x, y, 1, TileType::Const);
        }
    }

    // --- Phase 5b: Extend jump_target WD chain on L0 ---
    // Existing chain: col 61, rows 20-63. Extend to row 73 for Mux input.
    for row in (oy + 64)..=(oy + 73) {
        ctx.place(ox + 61, row, TileType::WireDown);
    }

    // --- Phase 5c: L0 PC update logic (rows 71-74) ---
    // Row 71: branch_taken routing down from Or@(32,70)
    ctx.place(ox + 32, oy + 71, TileType::WireDown);

    // Row 72: branch_taken continues + pc+1 computation
    ctx.place(ox + 32, oy + 72, TileType::WireDown); // branch_taken → Mux selector
    ctx.place(ox + 33, oy + 72, TileType::ViaUp); // PC from L1
    ctx.place(ox + 34, oy + 72, TileType::Add); // pc+1 = Add(PC, 1) — wrapping, no carry leak
    ctx.place_with_value(ox + 35, oy + 72, TileType::Const, 1); // constant 1

    // Row 73: Mux selects next_pc
    ctx.place(ox + 31, oy + 73, TileType::ViaUp); // jump_target from L1
    ctx.place(ox + 32, oy + 73, TileType::Mux); // up=branch_taken, left=jump_target, right=pc+1
    ctx.place(ox + 33, oy + 73, TileType::WireLeft); // pc+1 from right (WireDown@34)
    ctx.place(ox + 34, oy + 73, TileType::WireDown); // pc+1 from Add@(34,72) above

    // Row 74: Mux output descends to L1
    ctx.place(ox + 32, oy + 74, TileType::WireDown); // next_pc → ViaDown on L1

    // --- Phase 5d: L1 routing tiles ---
    // Signal: PC value (Register8@(3,1) → Add@(34,72))
    //   ViaDown on L1 reads PC from L0 Register8
    ctx.place_on_layer(ox + 3, oy + 1, 1, TileType::ViaDown);
    //   WireDown chain on L1 col 3, rows 2-72
    for row in (oy + 2)..=(oy + 72) {
        ctx.place_on_layer(ox + 3, row, 1, TileType::WireDown);
    }
    //   WireRight chain on L1 row 72, cols 4-33
    for col in (ox + 4)..=(ox + 33) {
        ctx.place_on_layer(col, oy + 72, 1, TileType::WireRight);
    }

    // Signal: jump_target (And@(61,19) via WD col 61 → L1 → ViaUp@(31,73))
    //   ViaDown on L1 reads jump_target from L0 WireDown@(61,73)
    ctx.place_on_layer(ox + 61, oy + 73, 1, TileType::ViaDown);
    //   WireLeft chain on L1 row 73, cols 31-60 (must reach col 31 where ViaUp reads)
    for col in (ox + 31)..=(ox + 60) {
        ctx.place_on_layer(col, oy + 73, 1, TileType::WireLeft);
    }

    // Signal: result (Mux@(32,73) output → Register8@(3,1) left input)
    //   ViaDown on L1 reads next_pc from L0 WireDown@(32,74)
    ctx.place_on_layer(ox + 32, oy + 74, 1, TileType::ViaDown);
    //   WireLeft chain on L1 row 74, cols 2-31 (includes col 2 to connect WireUp)
    for col in (ox + 2)..=(ox + 31) {
        ctx.place_on_layer(col, oy + 74, 1, TileType::WireLeft);
    }
    //   WireUp chain on L1 col 2, rows 1-73 (signal ascends to row 1)
    for row in (oy + 1)..=(oy + 73) {
        ctx.place_on_layer(ox + 2, row, 1, TileType::WireUp);
    }

    // --- Phase 5f: L1 routing for ALU result tree selectors ---
    // Routes alu_sel bits from decoder BitSelect tiles to result tree S0/S1/S2 ViaUp positions.
    // Uses L1 cols 40-42 for vertical drops, rows 13/14/15 for horizontal pickup,
    // rows 46/49/52 for horizontal fan-out to target positions.
    //
    // CRITICAL: Column assignment is REVERSED — the route on the highest row (S2, row 15)
    // gets the lowest drop column (40), and the route on the lowest row (S0, row 13)
    // gets the highest drop column (42). This prevents horizontal WireRight chains
    // from crossing and overwriting other routes' vertical WireDown chains.
    // Without this, S1's WireRight@row14 would overwrite S0's WireDown@col40 at (40,14).

    // alu_sel2: BitSelect@L0(14,15) → S2 ViaUp@{(20,52)} — drop col 55
    // (Sprint 85: relocated from col 40 to free cols 40-42 for writeback bus)
    ctx.place_on_layer(ox + 14, oy + 15, 1, TileType::ViaDown);
    for col in (ox + 15)..=(ox + 55) {
        ctx.place_on_layer(col, oy + 15, 1, TileType::WireRight);
    }
    for row in (oy + 16)..=(oy + 52) {
        ctx.place_on_layer(ox + 55, row, 1, TileType::WireDown);
    }
    for col in (ox + 20)..=(ox + 54) {
        ctx.place_on_layer(col, oy + 52, 1, TileType::WireLeft);
    }

    // alu_sel1: BitSelect@L0(14,14) → S1 ViaUp@{(10,49),(30,49)} — drop col 56
    ctx.place_on_layer(ox + 14, oy + 14, 1, TileType::ViaDown);
    for col in (ox + 15)..=(ox + 56) {
        ctx.place_on_layer(col, oy + 14, 1, TileType::WireRight);
    }
    for row in (oy + 15)..=(oy + 49) {
        ctx.place_on_layer(ox + 56, row, 1, TileType::WireDown);
    }
    for col in (ox + 10)..=(ox + 55) {
        ctx.place_on_layer(col, oy + 49, 1, TileType::WireLeft);
    }

    // alu_sel0: BitSelect@L0(14,13) → S0 ViaUp@{(5,46),(15,46),(25,46),(35,46)} — drop col 57
    ctx.place_on_layer(ox + 14, oy + 13, 1, TileType::ViaDown);
    for col in (ox + 15)..=(ox + 57) {
        ctx.place_on_layer(col, oy + 13, 1, TileType::WireRight);
    }
    for row in (oy + 14)..=(oy + 46) {
        ctx.place_on_layer(ox + 57, row, 1, TileType::WireDown);
    }
    for col in (ox + 5)..=(ox + 56) {
        ctx.place_on_layer(col, oy + 46, 1, TileType::WireLeft);
    }

    // =========================================================================
    // PHASE 5g: L1 WRITEBACK BUS (Sprint 85)
    // =========================================================================
    // Routes ALU result from tree root @L0(20,54) physically to each register's
    // result input position via L1, eliminating 4 software Const writes per ALU tick.
    //
    // Route: L1 ViaDown@(20,54) → WireLeft west to col 4 @row 54
    //        → WireUp north on col 4 from row 53 to row 28
    //        → WireRight east on row 28 from col 5 to col 44
    //        → Per-register WireUp branches north from row 28:
    //           Reg0: col 40, rows 22-27 (target row 21)
    //           Reg1: col 36, rows 24-27 (target row 23)
    //           Reg2: col 32, rows 26-27 (target row 25)
    //           Reg3: col 28, row 27    (target row 27)
    //
    // Col 4 avoids ALL alu_sel horizontal fan-outs (row 46 starts at col 5,
    // row 49 at col 10, row 52 at col 20). Row 28 is below all goh WD chains
    // (col 43 ends at row 20, col 38 at row 22, col 34 at row 24, col 30 at row 26).
    // Register branch cols (40,36,32,28) are free after alu_sel relocation (Part 1).

    // ViaDown picks up ALU result root from L0@(20,54)
    ctx.place_on_layer(ox + 20, oy + 54, 1, TileType::ViaDown);

    // WireLeft west at row 54, cols 4-19 (signal flows from ViaDown@20 westward)
    for col in (ox + 4)..=(ox + 19) {
        ctx.place_on_layer(col, oy + 54, 1, TileType::WireLeft);
    }

    // WireUp spine at col 4, rows 28-53 (signal flows from row 54 northward)
    for row in (oy + 28)..=(oy + 53) {
        ctx.place_on_layer(ox + 4, row, 1, TileType::WireUp);
    }

    // WireRight horizontal bus at row 28, cols 5-44 (signal flows from spine eastward)
    for col in (ox + 5)..=(ox + 44) {
        ctx.place_on_layer(col, oy + 28, 1, TileType::WireRight);
    }

    // Per-register WireUp branches from row 28 northward.
    // Each branch goes to col (result_col - 1) so the ALU result arrives at
    // the Mux's LEFT input at L1@(result_col, reg_row).
    //   Reg0: col 39, rows 21-27 → left of L1 Mux@(40,21)
    //   Reg1: col 35, rows 23-27 → left of L1 Mux@(36,23)
    //   Reg2: col 31, rows 25-27 → left of L1 Mux@(32,25)
    //   Reg3: col 27, row 27     → left of L1 Mux@(28,27)
    for row in (oy + 21)..=(oy + 27) {
        ctx.place_on_layer(ox + 39, row, 1, TileType::WireUp);
    }
    for row in (oy + 23)..=(oy + 27) {
        ctx.place_on_layer(ox + 35, row, 1, TileType::WireUp);
    }
    for row in (oy + 25)..=(oy + 27) {
        ctx.place_on_layer(ox + 31, row, 1, TileType::WireUp);
    }
    ctx.place_on_layer(ox + 27, oy + 27, 1, TileType::WireUp);

    // --- Phase 5g.2: L1 merge Mux per register (Sprint 85) ---
    // Each register gets an L1 Mux that selects between physical ALU result
    // (from writeback bus, default) and software LD data (for memory reads).
    //
    // L1 Mux@(result_col, reg_row): left=ALU (WireUp), right=LD_data (Const), up=mem_read (Const)
    //   up==0 → right (ALU result, physical) — default for ALU ops
    //   up!=0 → left  ... wait, Mux selects left when up!=0.
    // We want: default=ALU (most ticks), LD override when mem_read=1.
    // Mux: up!=0 → left, up==0 → right.
    // So: left=LD_data, right=ALU_result? No — ALU is from WireUp on the LEFT.
    //
    // The WireUp branch is at col (result_col - 1), feeding the Mux's LEFT input.
    // So: left=ALU, right=LD_data. When up==0 → right=LD_data (wrong!).
    // We need inverted logic. Solutions:
    //   A) Swap: put LD_data at left, ALU at right. But ALU comes from WireUp on left.
    //   B) Use Const=MAX as default mem_read, write 0 for LD ticks.
    //   C) Add a Zero tile to invert the selector.
    //
    // Solution B: mem_read defaults to MAX (select left=ALU). On LD ticks, write 0
    // (select right=LD_data). This inverts the convention but works cleanly.
    // Actually simpler: just swap which side gets which signal.
    //
    // REVISED: Route ALU to RIGHT of Mux, LD to LEFT.
    // WireUp branch at col (result_col - 1) carries ALU result.
    // Place WireRight at (result_col, reg_row-ish) to relay ALU from left to right of Mux?
    // No — that would require the Mux to be further right.
    //
    // SIMPLEST: Keep left=ALU (from WireUp), right=LD_data (Const).
    // mem_read Const defaults to MAX → up!=0 → selects left=ALU. ✓
    // On LD tick, write mem_read=0 → up==0 → selects right=LD_data. ✓
    //
    // Convention: mem_read = MAX means "use ALU" (normal), mem_read = 0 means "use LD data".
    // This is inverted from the name but matches Mux semantics perfectly.

    let wb_reg_cols: [(usize, usize); NUM_REGISTERS] = [
        (40, 21), // Reg0: result_col, reg_row
        (36, 23), // Reg1
        (32, 25), // Reg2
        (28, 27), // Reg3
    ];

    let mut wb_merge_mux_l1_indices = [0usize; NUM_REGISTERS];
    let mut ld_data_l1_indices = [0usize; NUM_REGISTERS];
    let mut mem_read_l1_indices = [0usize; NUM_REGISTERS];

    for reg in 0..NUM_REGISTERS {
        let (result_col, reg_row) = wb_reg_cols[reg];

        // L1 Mux at (result_col, reg_row): merge ALU result and LD data
        wb_merge_mux_l1_indices[reg] =
            ctx.place_on_layer(ox + result_col, oy + reg_row, 1, TileType::Mux);

        // L1 Const (LD data) at (result_col + 1, reg_row) — Mux RIGHT input
        // Software writes LD value here on LD ticks
        ld_data_l1_indices[reg] =
            ctx.place_on_layer_with_value(ox + result_col + 1, oy + reg_row, 1, TileType::Const, 0);

        // L1 Const (mem_read selector) at (result_col, reg_row - 1) — Mux UP input
        // Default = MAX → selects LEFT = ALU result (physical bus)
        // Software writes 0 on LD ticks → selects RIGHT = LD data
        mem_read_l1_indices[reg] = ctx.place_on_layer_with_value(
            ox + result_col,
            oy + reg_row - 1,
            1,
            TileType::Const,
            u64::MAX,
        );
    }

    // Replace L0 register result Consts with ViaUp tiles.
    // ViaUp reads from L1 at same (x,y) = the merge Mux output.
    for reg in 0..NUM_REGISTERS {
        let (result_col, reg_row) = wb_reg_cols[reg];
        // Overwrite the Const placed earlier with ViaUp
        ctx.reg_result_indices[reg] = ctx.place(ox + result_col, oy + reg_row, TileType::ViaUp);
    }

    // =========================================================================
    // PHASE 6: PHYSICAL RAM SUBSYSTEM (Sprint 83 — 8-Cell Proof of Concept)
    // =========================================================================
    // Strategy: Route R0 (address) via L1 to bypass CPU core congestion.
    // Place RAM and decoder in clean "Eastern Zone" (cols 68+).
    // Write path is physical (Decoder3to8 → And → BitSelect → Ram).
    // Write data and mem_write are software pre-written Consts.
    // Read remains software-managed (get_logic_value_by_idx on Ram tiles).
    //
    // L1 routing: R0 from Register8@(42,21) → WR@(43-46,21) → L1 ViaDown@(46,21)
    //   → L1 WR@(47-62,21) → L1 WD@(62,22-91) → L0 ViaUp@(62,92)
    //   → WR@(63-70,92) to Decoder3to8@(71,92).
    //
    // RAM subsystem (rows 90-99, cols 68-98):
    //   Row 90-92: R0 arrival + Decoder3to8 + And(one_hot, mem_write)
    //   Row 93-95: Gated one-hot fan-out + BitSelect per-cell WE extraction
    //   Row 96-97: Write data fan-out + WireCross/VBusOut for WE/data crossing
    //   Row 98:    Ram array (8 cells, 3-column spacing)

    // --- Phase 6a: Extend L1 guard sweep ---
    // The existing L1 guard covers cols 0-65, rows 0-79.
    // Extend to cover L1 routing column 62 through rows 80-95,
    // and the RAM area for any future L1 usage.
    for y in (oy + 80)..=(oy + 99) {
        for x in (ox + 62)..=(ox + 98) {
            ctx.place_on_layer(x, y, 1, TileType::Const);
        }
    }

    // --- Phase 6b: L0 guard sweep for RAM area ---
    // Guard cols 62-98, rows 89-99 on L0 before placing active tiles.
    for y in (oy + 89)..=(oy + 99) {
        for x in (ox + 62)..=(ox + 98) {
            ctx.guard(x, y);
        }
    }

    // --- Phase 6c: L0 WireRight chain to extend R0 eastward ---
    // R0 Register8 is at (42,21). Cols 43-46 at row 21 are guard Consts
    // (within the register guard range 27..=43 and eastern guard range 44-55).
    // rs BitSelect WD chains at cols 47,50,53 block further L0 extension,
    // so we hop onto L1 at col 46.
    for x in (ox + 43)..=(ox + 46) {
        ctx.place(x, oy + 21, TileType::WireRight);
    }

    // --- Phase 6d: L1 routing for R0 address ---
    // ViaDown on L1 at (46,21) reads R0 from L0 WireRight@(46,21)
    ctx.place_on_layer(ox + 46, oy + 21, 1, TileType::ViaDown);

    // L1 WireRight at row 21, cols 47-62 (horizontal hop past rs WD chains)
    // Row 21 on L1 has no existing routes (all existing L1 horizontals are at rows 13-15,46,49,52,72-74)
    for x in (ox + 47)..=(ox + 62) {
        ctx.place_on_layer(x, oy + 21, 1, TileType::WireRight);
    }

    // L1 WireDown at col 62, rows 22-92 (vertical transport to RAM area)
    // Col 62 on L1 is free (existing L1 verticals at cols 2,3,40,41,42 only).
    // L1 row 73 WireLeft goes cols 31-60, so col 62 clears it.
    // Must extend to row 92 because ViaUp@L0(62,92) reads from L1(62,92) (same position).
    for y in (oy + 22)..=(oy + 92) {
        ctx.place_on_layer(ox + 62, y, 1, TileType::WireDown);
    }

    // --- Phase 6e: R0 arrival at decoder row ---
    // L0 ViaUp at (62,92) reads R0 from L1 WireDown@(62,91).
    // Then WireRight carries R0 east to Decoder3to8 input.
    ctx.place(ox + 62, oy + 92, TileType::ViaUp);
    for x in (ox + 63)..=(ox + 70) {
        ctx.place(x, oy + 92, TileType::WireRight);
    }
    // Decoder3to8 at (71,92): left = WR@(70,92) = R0. ✓

    // --- Phase 6f: Decoder subsystem ---
    // Decoder3to8 at (71,92): left = WR@(70,92) = R0[2:0] → one-hot output
    ctx.place(ox + 71, oy + 92, TileType::Decoder3to8);
    // WireRight at (72,92): carries decoder one-hot east
    ctx.place(ox + 72, oy + 92, TileType::WireRight);
    // And at (73,92): left = WR@(72,92) = one_hot, right = WL@(74,92) = mem_write
    // When mem_write=MAX: output = one_hot (WE active for addressed cell)
    // When mem_write=0: output = 0 (all WEs off)
    ctx.place(ox + 73, oy + 92, TileType::And);
    // WireLeft at (74,92): right = Const@(75,92) = mem_write signal
    ctx.place(ox + 74, oy + 92, TileType::WireLeft);
    // Const at (75,92): software-written mem_write control (MAX for ST, 0 otherwise)
    let mem_write_const_idx = ctx.place_with_value(ox + 75, oy + 92, TileType::Const, 0);

    // --- Phase 6g: Gated one-hot fan-out ---
    // And@(73,92) output = gated_one_hot. Fan out via WireDown + WireRight.
    ctx.place(ox + 73, oy + 93, TileType::WireDown); // gated_oh from And above
    // WireRight fan-out at row 93, cols 74-97 (covers all 8 BitSelect positions)
    for x in (ox + 74)..=(ox + 97) {
        ctx.place(x, oy + 93, TileType::WireRight);
    }

    // --- Phase 6h: Per-cell WE extraction via BitSelect ---
    // 8 cells at 3-column spacing. Each cell: [WD(gated_oh), BitSelect, Const(bit_pos)]
    // Cell i at cols (74+3*i, 75+3*i, 76+3*i), row 95.
    // WD drops from WR fan-out at row 93 to BitSelect at row 95.
    let ram_cell_cols: [usize; 8] = [75, 78, 81, 84, 87, 90, 93, 96]; // BitSelect columns
    for (i, &bs_col) in ram_cell_cols.iter().enumerate() {
        let wd_col = bs_col - 1; // WD column = one left of BitSelect
        // WD at row 94: gated_oh drops from WR@(wd_col,93)
        ctx.place(ox + wd_col, oy + 94, TileType::WireDown);
        // WD at row 95: gated_oh continues to BitSelect left input
        ctx.place(ox + wd_col, oy + 95, TileType::WireDown);
        // BitSelect at (bs_col, 95): left = WD@(wd_col,95) = gated_oh, right = Const(i)
        ctx.place(ox + bs_col, oy + 95, TileType::BitSelect);
        // Const(i) at (bs_col+1, 95): bit position for extraction
        ctx.place_with_value(ox + bs_col + 1, oy + 95, TileType::Const, i as u64);
    }

    // --- Phase 6i: Write data fan-out + WE/data crossing ---
    // Write data Const at (68,96): software-written Rs value before tick_with_delays()
    let write_data_const_idx = ctx.place_with_value(ox + 68, oy + 96, TileType::Const, 0);
    // WireRight fan-out at row 96, cols 69-97 (data flows east to all Ram cells)
    for x in (ox + 69)..=(ox + 97) {
        ctx.place(x, oy + 96, TileType::WireRight);
    }
    // Overwrite WE column positions with WireCross tiles:
    // WireCross: output = (left & 0xFFFFFFFF) | (up & 0xFFFFFFFF_00000000)
    // left = WR(write_data) in low 32 bits, up = BitSelect(WE) in high 32 bits
    // (BitSelect outputs MAX = all bits set, so high 32 bits are naturally set)
    for &bs_col in &ram_cell_cols {
        ctx.place(ox + bs_col, oy + 96, TileType::WireCross);
    }

    // --- Phase 6j: VBusOut + data WD + Ram array ---
    // Row 97: VBusOut at WE columns extracts WE from high bits.
    //         WireDown at data columns carries write_data down.
    // Row 98: Ram tiles at WE columns, WireDown(data) at data columns.
    let mut physical_ram_indices = [0usize; 8];
    for (i, &bs_col) in ram_cell_cols.iter().enumerate() {
        let data_col = bs_col - 1; // Data WD column (one left of Ram)
        // Row 97: VBusOut extracts WE, WireDown carries data
        ctx.place(ox + bs_col, oy + 97, TileType::VBusOut); // WE for Ram up
        ctx.place(ox + data_col, oy + 97, TileType::WireDown); // data for Ram left
        // Row 98: Ram tile + data WireDown
        ctx.place(ox + data_col, oy + 98, TileType::WireDown); // data continues down
        // Ram at (bs_col, 98): up = VBusOut@(bs_col,97) = WE, left = WD@(data_col,98) = data
        physical_ram_indices[i] = ctx.place(ox + bs_col, oy + 98, TileType::Ram);
    }

    // Update ram_indices to point to physical Ram tiles (replaces old Const-based RAM)
    ctx.ram_indices.clear();
    for &idx in &physical_ram_indices {
        ctx.ram_indices.push(idx);
    }

    // =========================================================================
    // PHASE 7: PHYSICAL REGISTER WRITE-ENABLE DECODE (Sprint 84)
    // =========================================================================
    // Replaces 5 software writes per tick (4 WE resets + 1 WE set) with a
    // physical decode circuit. Uses "Western Flank" approach:
    //
    //   1. WireCross@(13,19) packs ctrl_a into IR fan-out high bits
    //   2. Western decoder (cols 4-12, rows 20-24) extracts rd_bit0, rd_bit1,
    //      rwen from IR+packed, computes Decoder3to8 → And(one_hot, rwen) = goh
    //   3. L1 routes goh ABOVE the ALU selector cage (rows 8-11) to each
    //      register's target column, then WD drops to each WE row
    //   4. Per-register BitSelect(Const(2^(1<<I)), goh) at Mux position
    //
    // The reversed L1 row assignment (longest route → highest row) ensures
    // no WD drops cross any WR routes. No WireCross intersections needed.

    // --- Phase 7a: L0 guard sweep for western decoder area ---
    // Guard cols 4-13 at rows 20-24 to prevent Wire default leakage.
    // These positions are currently Wire tiles (not placed by any earlier code).
    for y in (oy + 20)..=(oy + 24) {
        for x in (ox + 4)..=(ox + 16) {
            if ctx.sim.tile_type_xy(x, y) == TileType::Wire {
                ctx.guard(x, y);
            }
        }
    }

    // --- Phase 7b: Western decoder — input extraction (row 20) ---
    // Extract rd_bit0, rd_bit1 from IR fan-out at row 19.
    //   (4,20): WD(IR)    (5,20): BitSelect(IR,2)=rd0  (6,20): Const(2)
    //   (7,20): WD(IR)    (8,20): BitSelect(IR,3)=rd1  (9,20): Const(3)
    // rwen (reg_write_en) arrives via L1 from ctrl_a_bits[3] at BitSelect@(14,16).
    ctx.place(ox + 4, oy + 20, TileType::WireDown);
    ctx.place(ox + 5, oy + 20, TileType::BitSelect);
    ctx.place_with_value(ox + 6, oy + 20, TileType::Const, 2);
    ctx.place(ox + 7, oy + 20, TileType::WireDown);
    ctx.place(ox + 8, oy + 20, TileType::BitSelect);
    ctx.place_with_value(ox + 9, oy + 20, TileType::Const, 3);

    // --- Phase 7c: Mask rd bits to address values (row 21) ---
    //   And(rd0, 1) → 0 or 1,  And(rd1, 2) → 0 or 2
    ctx.place(ox + 5, oy + 21, TileType::WireDown); // rd0 from BitSelect
    ctx.place(ox + 6, oy + 21, TileType::And); // And(rd0, 1)
    ctx.place_with_value(ox + 7, oy + 21, TileType::Const, 1);
    ctx.place(ox + 8, oy + 21, TileType::WireDown); // rd1 from BitSelect
    ctx.place(ox + 9, oy + 21, TileType::And); // And(rd1, 2)
    ctx.place_with_value(ox + 10, oy + 21, TileType::Const, 2);

    // --- Phase 7d: Combine address (row 22) ---
    //   Or(masked_rd0, masked_rd1) → rd address 0-3
    ctx.place(ox + 6, oy + 22, TileType::WireDown); // masked_rd0
    ctx.place(ox + 7, oy + 22, TileType::Or); // Or(masked_rd0, masked_rd1)
    ctx.place(ox + 8, oy + 22, TileType::WireLeft); // carries masked_rd1 west
    ctx.place(ox + 9, oy + 22, TileType::WireDown); // masked_rd1

    // --- Phase 7e: Decoder + gating (row 23) ---
    //   Decoder3to8(address) → one_hot, And(one_hot, rwen) = gated_one_hot
    //   rwen arrives via L1 route from ctrl_a_bits[3]@(14,16) → ViaUp@(14,23) → WL chain
    ctx.place(ox + 7, oy + 23, TileType::WireDown); // address from Or
    ctx.place(ox + 8, oy + 23, TileType::WireRight); // carries address east
    ctx.place(ox + 9, oy + 23, TileType::Decoder3to8); // 1 << (address & 7)
    ctx.place(ox + 10, oy + 23, TileType::WireRight); // carries one_hot east
    ctx.place(ox + 11, oy + 23, TileType::And); // And(one_hot, rwen) = goh
    ctx.place(ox + 12, oy + 23, TileType::WireLeft); // rwen WL ← (13,23)
    ctx.place(ox + 13, oy + 23, TileType::WireLeft); // rwen WL ← (14,23)
    ctx.place(ox + 14, oy + 23, TileType::ViaUp); // rwen from L1(14,23)

    // --- Phase 7g: goh output (row 24) ---
    ctx.place(ox + 11, oy + 24, TileType::WireDown); // goh from And@(11,23)

    // --- Phase 7h: L1 rwen routing ---
    // Route reg_write_en from ctrl_a_bits[3] = BitSelect@L0(14,16) via L1 to ViaUp@L0(14,23).
    // L1(14,16) is free — ALU selector ViaDowns are at L1(14,13/14/15) only.
    // L1 col 14 at rows 17-23 is free (no ALU selector WD chains, which are at cols 40-42).
    ctx.place_on_layer(ox + 14, oy + 16, 1, TileType::ViaDown); // picks up rwen from L0
    for row in (oy + 17)..=(oy + 23) {
        ctx.place_on_layer(ox + 14, row, 1, TileType::WireDown); // rwen south on L1
    }

    // --- Phase 7i: L1 routing — goh distribution ---
    // ViaDown picks up goh from L0(11,24). WireUp chain on L1 col 11 goes north
    // to row 8 (above ALU selector horizontal routes at rows 13-15).
    // Col 11 < 15, so WU chain passes safely through rows 13-15.
    ctx.place_on_layer(ox + 11, oy + 24, 1, TileType::ViaDown);
    for row in (oy + 8)..=(oy + 23) {
        ctx.place_on_layer(ox + 11, row, 1, TileType::WireUp);
    }

    // Reg0 (target col 43, row 20): L1 row 8 WR → L1 col 43 WD → ViaUp
    for col in (ox + 12)..=(ox + 43) {
        ctx.place_on_layer(col, oy + 8, 1, TileType::WireRight);
    }
    for row in (oy + 9)..=(oy + 20) {
        ctx.place_on_layer(ox + 43, row, 1, TileType::WireDown);
    }

    // Reg1 (target col 38, row 22): L1 row 9 WR → L1 col 38 WD → ViaUp
    for col in (ox + 12)..=(ox + 38) {
        ctx.place_on_layer(col, oy + 9, 1, TileType::WireRight);
    }
    for row in (oy + 10)..=(oy + 22) {
        ctx.place_on_layer(ox + 38, row, 1, TileType::WireDown);
    }

    // Reg2 (target col 34, row 24): L1 row 10 WR → L1 col 34 WD → ViaUp
    for col in (ox + 12)..=(ox + 34) {
        ctx.place_on_layer(col, oy + 10, 1, TileType::WireRight);
    }
    for row in (oy + 11)..=(oy + 24) {
        ctx.place_on_layer(ox + 34, row, 1, TileType::WireDown);
    }

    // Reg3 (target col 30, row 26): L1 row 11 WR → L1 col 30 WD → ViaUp
    for col in (ox + 12)..=(ox + 30) {
        ctx.place_on_layer(col, oy + 11, 1, TileType::WireRight);
    }
    for row in (oy + 12)..=(oy + 26) {
        ctx.place_on_layer(ox + 30, row, 1, TileType::WireDown);
    }

    // --- Phase 7j: L0 per-register WE extraction ---
    // Each register uses BitSelect(Const(2^(1<<I)), goh) at the Mux position.
    // BitSelect: if (left >> (right & 63)) & 1 { MAX } else { 0 }
    //   left = Const(2^(1<<I)) — value with exactly one bit set at position (1<<I)
    //   right = goh — Decoder3to8 output (1<<rd) when rwen=MAX, 0 otherwise
    // This returns MAX only when the specific register's one-hot bit is set.

    // Reg0 (Mux@(41,21)): BitSelect@(41,20), left=Const(2)@(40,20) [existing!]
    //   right=WL@(42,20)←ViaUp@(43,20)←L1(43,20)
    ctx.place(ox + 43, oy + 20, TileType::ViaUp); // goh from L1
    ctx.place(ox + 42, oy + 20, TileType::WireLeft); // carries goh west
    ctx.place(ox + 41, oy + 20, TileType::BitSelect); // WE_reg0
    // Const(2)@(40,20) already exists from rd_bit0 copy — reused as BitSelect value!

    // Reg1 (Mux@(37,23)): BitSelect@(37,22), left=Const(4)@(36,22), right=ViaUp@(38,22)
    ctx.place(ox + 38, oy + 22, TileType::ViaUp); // goh from L1
    ctx.place_with_value(ox + 36, oy + 22, TileType::Const, 4);
    ctx.place(ox + 37, oy + 22, TileType::BitSelect); // WE_reg1

    // Reg2 (Mux@(33,25)): BitSelect@(33,24), left=Const(16)@(32,24), right=ViaUp@(34,24)
    ctx.place(ox + 34, oy + 24, TileType::ViaUp); // goh from L1
    ctx.place_with_value(ox + 32, oy + 24, TileType::Const, 16);
    ctx.place(ox + 33, oy + 24, TileType::BitSelect); // WE_reg2

    // Reg3 (Mux@(29,27)): BitSelect@(29,26), left=Const(256)@(28,26), right=ViaUp@(30,26)
    ctx.place(ox + 30, oy + 26, TileType::ViaUp); // goh from L1
    ctx.place_with_value(ox + 28, oy + 26, TileType::Const, 256);
    ctx.place(ox + 29, oy + 26, TileType::BitSelect); // WE_reg3

    // =========================================================================
    // PHASE 8: PHYSICAL OPERAND B DATA (Sprint 86)
    // =========================================================================
    // Routes register values physically from Register8 → L1 → ViaUp at Tree B data positions,
    // eliminating 4 writes + 2 dirty per tick (pre_write_tree_b_data).
    //
    // Flag physicalization (Parts A-C) DEFERRED: L1 routing for ctrl_a to flag area is blocked
    // by L1 congestion (PC chains at cols 2-3, writeback bus at col 4, S0/S1 WL at rows 46/49).
    // Partial flag physicalization (z_result physical, z_update software) has a timing issue:
    // the post-tick S0/S1/S2 propagate uses stale z_update from the previous tick.
    // Both z_result AND z_update must be physical simultaneously — deferred to Sprint 87+.

    // --- Phase 8i: L1 Operand B routes (Register → Tree B via stacked WR rows) ---
    // Route register values from L0 WD chains at row 29 via L1 to Tree B data positions.
    // Below-row-28 approach avoids Sprint 85 WU branches and row 28 WR bus.

    // Reg0 → D0@(54,34): ViaDown@L1(42,29) → WD@(42,30) → WR row 30 cols 43-54 → WD col 54 rows 31-34
    ctx.place_on_layer(ox + 42, oy + 29, 1, TileType::ViaDown);
    ctx.place_on_layer(ox + 42, oy + 30, 1, TileType::WireDown);
    for col in (ox + 43)..=(ox + 54) {
        ctx.place_on_layer(col, oy + 30, 1, TileType::WireRight);
    }
    for row in (oy + 31)..=(oy + 34) {
        ctx.place_on_layer(ox + 54, row, 1, TileType::WireDown);
    }

    // Reg1 → D1@(52,34): ViaDown@L1(38,29) → WD@(38,30-31) → WR row 31 cols 39-52 → WD col 52 rows 32-34
    ctx.place_on_layer(ox + 38, oy + 29, 1, TileType::ViaDown);
    for row in (oy + 30)..=(oy + 31) {
        ctx.place_on_layer(ox + 38, row, 1, TileType::WireDown);
    }
    for col in (ox + 39)..=(ox + 52) {
        ctx.place_on_layer(col, oy + 31, 1, TileType::WireRight);
    }
    for row in (oy + 32)..=(oy + 34) {
        ctx.place_on_layer(ox + 52, row, 1, TileType::WireDown);
    }

    // Reg2 → D2@(48,34): ViaDown@L1(34,29) → WD@(34,30-32) → WR row 32 cols 35-48 → WD col 48 rows 33-34
    ctx.place_on_layer(ox + 34, oy + 29, 1, TileType::ViaDown);
    for row in (oy + 30)..=(oy + 32) {
        ctx.place_on_layer(ox + 34, row, 1, TileType::WireDown);
    }
    for col in (ox + 35)..=(ox + 48) {
        ctx.place_on_layer(col, oy + 32, 1, TileType::WireRight);
    }
    for row in (oy + 33)..=(oy + 34) {
        ctx.place_on_layer(ox + 48, row, 1, TileType::WireDown);
    }

    // Reg3 → D3@(46,34): ViaDown@L1(30,29) → WD@(30,30-33) → WR row 33 cols 31-46 → WD col 46 row 34
    ctx.place_on_layer(ox + 30, oy + 29, 1, TileType::ViaDown);
    for row in (oy + 30)..=(oy + 33) {
        ctx.place_on_layer(ox + 30, row, 1, TileType::WireDown);
    }
    for col in (ox + 31)..=(ox + 46) {
        ctx.place_on_layer(col, oy + 33, 1, TileType::WireRight);
    }
    ctx.place_on_layer(ox + 46, oy + 34, 1, TileType::WireDown);

    // --- Phase 8j: ViaUp at Tree B data positions ---
    // Replace Tree B data Consts with ViaUp tiles that read from L1.
    ctx.op_b_data_indices[0] = ctx.place(ox + 54, oy + 34, TileType::ViaUp); // D0 = Reg0
    ctx.op_b_data_indices[1] = ctx.place(ox + 52, oy + 34, TileType::ViaUp); // D1 = Reg1
    ctx.op_b_data_indices[2] = ctx.place(ox + 48, oy + 34, TileType::ViaUp); // D2 = Reg2
    ctx.op_b_data_indices[3] = ctx.place(ox + 46, oy + 34, TileType::ViaUp); // D3 = Reg3

    // =========================================================================
    // PHASE 9: PHYSICAL RAM READ MUX ON L1 (Sprint 87)
    // =========================================================================
    // Build an 8:1 binary Mux tree on L1 below the RAM area.
    // R0 address bits select which Ram cell's stored value appears at the root.
    //
    // Architecture:
    //   - ViaDown at each Ram col picks up stored value from L0
    //   - Binary Mux tree: 4 leaf (row 102) → 2 mid (row 105) → 1 root (row 108)
    //   - Inverted selectors (Zero tiles) because Mux(up!=0→left, up==0→right)
    //     and even-indexed Rams land on LEFT (physically western)
    //   - BitSelect uses RIGHT for bit position: (left >> (right & 63)) & 1
    //   - Per-pair extraction: R0 drops on col E+1, BitSelect at col E+2,
    //     guard Const(0) at odd Ram col E+3 serves as RIGHT=0 for bit0
    //   - For bit1/bit2: replace guard Const at Ram col with Const(1)/Const(2)
    //   - R0 distributed east via WR chain on row 91 (above Ram zone)
    //   - Zero crossings — all signals on non-overlapping columns

    // --- Phase 9a: L1 guard sweep (rows 91-110, cols 62-98) ---
    for y in (oy + 91)..=(oy + 110) {
        for x in (ox + 62)..=(ox + 98) {
            ctx.place_on_layer(x, y, 1, TileType::Const);
        }
    }

    // --- Phase 9b: Extend R0 L1 WireDown from row 91 to row 108 ---
    // R0 is already on L1 WD at col 62, rows 22-90. Extend south.
    for y in (oy + 91)..=(oy + 108) {
        ctx.place_on_layer(ox + 62, y, 1, TileType::WireDown);
    }

    // --- Phase 9c: R0 east distribution on L1 row 91 ---
    // R0 flows east above entire Ram zone. No crossings.
    for col in 63..=95 {
        ctx.place_on_layer(ox + col, oy + 91, 1, TileType::WireRight);
    }

    // --- Phase 9d: ViaDown picks up Ram outputs from L0 ---
    for &col in &ram_cell_cols {
        ctx.place_on_layer(ox + col, oy + 98, 1, TileType::ViaDown);
    }

    // --- Phase 9e: Ram WD chains straight down (rows 99-102) ---
    for &col in &ram_cell_cols {
        for row in 99..=102 {
            ctx.place_on_layer(ox + col, oy + row, 1, TileType::WireDown);
        }
    }

    // --- Phase 9f: Leaf Mux level (row 102) ---
    // 4 pairs: (Ram[0]@75,Ram[1]@78), (Ram[2]@81,Ram[3]@84),
    //          (Ram[4]@87,Ram[5]@90), (Ram[6]@93,Ram[7]@96).
    //
    // Per pair layout (even=E, odd=E+3, drop=E+1, bs=E+2):
    //   Row 91: WR chain carries R0 east (already placed in 9c)
    //   Row 92: (drop) = WD(R0)             — R0 drops from WR chain
    //   Row 93: (drop) = WD(R0)             — R0 continues to BitSelect LEFT
    //           (bs)   = BitSelect           — LEFT=R0@drop, RIGHT=Const(0)@odd
    //           (odd)  = Const(0)            — guard Const, already value 0
    //   Row 94: (bs)   = WD(bit0)           — bit0 output south
    //           (drop) = WL(bit0←bs)        — bit0 shifts west
    //   Rows 95-101: (drop) = WD(bit0)      — bit0 south to Zero level
    //   Row 101: (bs) = Zero                — NOT(bit0), reads LEFT=drop
    //   Row 102: (drop) = WR(Ram[even]←E)   — Ram[even] east to Mux LEFT
    //            (bs)   = Mux               — LEFT=Ram[even], RIGHT=Ram[odd]
    let leaf_pairs: [(usize, usize); 4] = [(75, 78), (81, 84), (87, 90), (93, 96)];
    for &(even_col, _odd_col) in &leaf_pairs {
        let drop_col = even_col + 1;
        let bs_col = even_col + 2;
        // R0 WD drop from row 91 WR chain
        ctx.place_on_layer(ox + drop_col, oy + 92, 1, TileType::WireDown);
        ctx.place_on_layer(ox + drop_col, oy + 93, 1, TileType::WireDown);
        // BitSelect: LEFT=(drop,93)=R0, RIGHT=(odd,93)=guard Const(0) → bit0
        ctx.place_on_layer(ox + bs_col, oy + 93, 1, TileType::BitSelect);
        // bit0 WD south from BitSelect, then WL west to drop_col
        ctx.place_on_layer(ox + bs_col, oy + 94, 1, TileType::WireDown);
        ctx.place_on_layer(ox + drop_col, oy + 94, 1, TileType::WireLeft);
        // bit0 WD south on drop_col to Zero level
        for row in 95..=101 {
            ctx.place_on_layer(ox + drop_col, oy + row, 1, TileType::WireDown);
        }
        // Zero: NOT(bit0)
        ctx.place_on_layer(ox + bs_col, oy + 101, 1, TileType::Zero);
        // Ram[even] WR east to Mux LEFT
        ctx.place_on_layer(ox + drop_col, oy + 102, 1, TileType::WireRight);
        // Leaf Mux
        ctx.place_on_layer(ox + bs_col, oy + 102, 1, TileType::Mux);
    }

    // --- Phase 9g: Mid Mux level (row 105) ---
    // M0123@(80,105): left from leaf0@(77,102), right from leaf1@(83,102).
    // M4567@(92,105): left from leaf2@(89,102), right from leaf3@(95,102).
    //
    // Per mid pair: R0 drops at r0_drop, BitSelect at bs_col,
    // Const(1) replaces guard at const_col (a Ram column).
    // Leaf outputs route south via WD, then WR/WL to mid Mux.
    let mid_params: [(usize, usize, usize, usize, usize); 2] = [
        // (left_leaf, right_leaf, r0_drop, bs_col, const_col)
        (77, 83, 79, 80, 81),
        (89, 95, 91, 92, 93),
    ];
    for &(left_leaf, right_leaf, r0_drop, bs_col, const_col) in &mid_params {
        // R0 WD drop from row 91 WR chain
        ctx.place_on_layer(ox + r0_drop, oy + 92, 1, TileType::WireDown);
        ctx.place_on_layer(ox + r0_drop, oy + 93, 1, TileType::WireDown);
        // Const(1) at Ram column — replaces guard Const(0) with bit position 1
        ctx.place_on_layer_with_value(ox + const_col, oy + 93, 1, TileType::Const, 1);
        // BitSelect: LEFT=(r0_drop,93)=R0, RIGHT=(const_col,93)=Const(1) → bit1
        ctx.place_on_layer(ox + bs_col, oy + 93, 1, TileType::BitSelect);
        // bit1 WD south, WL west to r0_drop
        ctx.place_on_layer(ox + bs_col, oy + 94, 1, TileType::WireDown);
        ctx.place_on_layer(ox + r0_drop, oy + 94, 1, TileType::WireLeft);
        // bit1 WD south on r0_drop to Zero level (row 104)
        for row in 95..=104 {
            ctx.place_on_layer(ox + r0_drop, oy + row, 1, TileType::WireDown);
        }
        // Zero: NOT(bit1)
        ctx.place_on_layer(ox + bs_col, oy + 104, 1, TileType::Zero);

        // Leaf Mux WD chains to mid level (rows 103-105)
        for row in 103..=105 {
            ctx.place_on_layer(ox + left_leaf, oy + row, 1, TileType::WireDown);
            ctx.place_on_layer(ox + right_leaf, oy + row, 1, TileType::WireDown);
        }
        // Left leaf → WR east to Mux LEFT
        for c in (left_leaf + 1)..bs_col {
            ctx.place_on_layer(ox + c, oy + 105, 1, TileType::WireRight);
        }
        // Right leaf → WL west to Mux RIGHT
        for c in (bs_col + 1)..right_leaf {
            ctx.place_on_layer(ox + c, oy + 105, 1, TileType::WireLeft);
        }
        // Mid Mux
        ctx.place_on_layer(ox + bs_col, oy + 105, 1, TileType::Mux);
    }

    // --- Phase 9h: Root Mux level (row 108) ---
    // Root@(86,108): left from mid0@(80,105), right from mid1@(92,105).
    let root_r0_drop = 85usize;
    let root_bs = 86usize;
    let root_const_col = 87usize; // Ram[4] column
    let left_mid = 80usize;
    let right_mid = 92usize;

    // R0 WD drop from row 91 WR chain
    ctx.place_on_layer(ox + root_r0_drop, oy + 92, 1, TileType::WireDown);
    ctx.place_on_layer(ox + root_r0_drop, oy + 93, 1, TileType::WireDown);
    // Const(2) at Ram column — replaces guard Const(0) with bit position 2
    ctx.place_on_layer_with_value(ox + root_const_col, oy + 93, 1, TileType::Const, 2);
    // BitSelect: LEFT=(root_r0_drop,93)=R0, RIGHT=(root_const_col,93)=Const(2) → bit2
    ctx.place_on_layer(ox + root_bs, oy + 93, 1, TileType::BitSelect);
    // bit2 WD south, WL west to r0_drop
    ctx.place_on_layer(ox + root_bs, oy + 94, 1, TileType::WireDown);
    ctx.place_on_layer(ox + root_r0_drop, oy + 94, 1, TileType::WireLeft);
    // bit2 WD south on r0_drop to Zero level (row 107)
    for row in 95..=107 {
        ctx.place_on_layer(ox + root_r0_drop, oy + row, 1, TileType::WireDown);
    }
    // Zero: NOT(bit2)
    ctx.place_on_layer(ox + root_bs, oy + 107, 1, TileType::Zero);

    // Mid Mux WD chains to root level (rows 106-108)
    for row in 106..=108 {
        ctx.place_on_layer(ox + left_mid, oy + row, 1, TileType::WireDown);
        ctx.place_on_layer(ox + right_mid, oy + row, 1, TileType::WireDown);
    }
    // Left mid → WR east to root LEFT
    for c in (left_mid + 1)..root_bs {
        ctx.place_on_layer(ox + c, oy + 108, 1, TileType::WireRight);
    }
    // Right mid → WL west to root RIGHT
    for c in (root_bs + 1)..right_mid {
        ctx.place_on_layer(ox + c, oy + 108, 1, TileType::WireLeft);
    }
    // Root Mux
    let ram_read_mux_root_idx = ctx.place_on_layer(ox + root_bs, oy + 108, 1, TileType::Mux);

    // --- Phase 9i: Rebuild via connections for cross-layer dirty propagation ---
    ctx.sim.rebuild_via_connections();

    PhysicalCpuIndices {
        pc_idx: ctx.pc_idx,
        ir_idx: ctx.ir_idx,
        reg_indices: ctx.reg_indices,
        flag_z_idx: ctx.flag_z_idx,
        flag_c_idx: ctx.flag_c_idx,
        rom_indices: ctx.rom_indices.clone(),
        ram_indices: ctx.ram_indices.clone(),
        grid_width: ctx.grid_width,
        tile_count: ctx.tiles_placed,
        shr_opcode_idx,
        decoder_lut_indices,
        merged_ctrl_a_idx,
        merged_ctrl_b_idx,
        ctrl_a_bits,
        ctrl_b_bits,
        ir_field_bits,
        op_a_root_idx: ctx.op_a_root_idx,
        op_b_root_idx: ctx.op_b_root_idx,
        reg_we_mux_indices: ctx.reg_mux_indices,
        reg_mux_indices: ctx.reg_mux_indices,
        branch_taken_idx,
        jump_target_idx,
        flag_z_zero_idx,
        flag_z_mux_idx,
        flag_c_addcarry_idx: ctx.alu_tile_indices[0], // AddCarry tile — read bit 8 for carry
        flag_c_bit_idx: 0,                            // not used yet
        flag_c_mux_idx,
        reg_result_indices: ctx.reg_result_indices,
        op_a_data_indices: ctx.op_a_data_indices,
        op_a_sel0_indices: ctx.op_a_sel0_indices,
        op_a_sel1_idx: ctx.op_a_sel1_idx,
        op_a_leaf_indices: ctx.op_a_leaf_indices,
        op_b_data_indices: ctx.op_b_data_indices,
        op_b_sel0_indices: ctx.op_b_sel0_indices,
        op_b_sel1_idx: ctx.op_b_sel1_idx,
        op_b_leaf_indices: ctx.op_b_leaf_indices,
        alu_tile_indices: ctx.alu_tile_indices,
        alu_result_data_indices: ctx.alu_result_data_indices,
        alu_result_sel0_indices: ctx.alu_result_sel0_indices,
        alu_result_sel1_indices: ctx.alu_result_sel1_indices,
        alu_result_sel2_idx: ctx.alu_result_sel2_idx,
        alu_result_leaf_indices: ctx.alu_result_leaf_indices,
        alu_result_mid_indices: ctx.alu_result_mid_indices,
        alu_result_root_idx: ctx.alu_result_root_idx,
        next_pc_const_idx: ctx.next_pc_const_idx,
        flag_z_update_idx: ctx.flag_z_update_idx,
        flag_c_update_idx: ctx.flag_c_update_idx,
        flag_z_result_idx: ctx.flag_z_result_idx,
        flag_c_carry_idx: ctx.flag_c_carry_idx,
        mem_write_const_idx,
        write_data_const_idx,
        physical_ram_indices,
        wb_merge_mux_l1_indices,
        ld_data_l1_indices,
        mem_read_l1_indices,
        ram_read_mux_root_idx,
    }
}

/// Wire the CPU datapath (hybrid mode — software decode/writeback).
///
/// Layout (origin at ox, oy):
/// ```text
/// Row 0:      Clock at col 3 + guards; Decoder LUT at ox+15..17
/// Row 1:      PC circuit at ox+0..4; Decoder continues
/// Row 2:      Fetch: ROM + Mux8to1 + WireLeft + WireDown
/// Row 3-11:   Register file at ox+5..8; Flags at ox+20..24
/// Row 12-16:  Operand A mux tree (ox+8..14), Operand B mux tree (ox+16..22)
/// Row 17:     Bus routing (A bus WD, B bus WL chain)
/// Row 18:     Guard row (pass-through at ox+10, ox+12)
/// Row 19-26:  Compressed ALU (8 ops, 1 row each) at ox+10..12
/// Row 27:     ALU bottom guard
/// Row 28-36:  ALU result 8:1 mux tree (ox+2..16)
/// Row 37+:    ROM, guard, RAM
/// ```
pub fn wire_complete_cpu(
    ctx: &mut WiringContext,
    program: &[u8],
    rom_size: usize,
    ram_size: usize,
    _initial_regs: &[u64; NUM_REGISTERS],
) {
    let (ox, oy) = ctx.origin;

    // =========================================================================
    // ROW 0: CLOCK + GUARDS
    // =========================================================================
    ctx.guard(ox, oy);
    ctx.guard(ox + 1, oy);
    ctx.guard(ox + 2, oy);
    ctx.place(ox + 3, oy, TileType::ClockGlobal);
    ctx.guard(ox + 4, oy);

    // =========================================================================
    // ROW 1: PC CIRCUIT
    // =========================================================================
    ctx.guard(ox, oy + 1);
    ctx.next_pc_const_idx = ctx.place_with_value(ox + 2, oy + 1, TileType::Const, 0);
    ctx.pc_idx = ctx.place_with_value(ox + 3, oy + 1, TileType::Register8, 0);
    ctx.guard(ox + 4, oy + 1);

    // =========================================================================
    // ROW 2: INSTRUCTION FETCH (16-byte ROM via Mux16to1)
    // =========================================================================
    let effective_rom = rom_size.min(16);
    // Pack instructions 0-7 into ROM-A
    let mut packed_rom_a: u64 = 0;
    for addr in 0..effective_rom.min(8) {
        let byte = if addr < program.len() {
            program[addr] as u64
        } else {
            0
        };
        packed_rom_a |= byte << (addr * 8);
    }
    // Pack instructions 8-15 into ROM-B
    let mut packed_rom_b: u64 = 0;
    for addr in 8..effective_rom {
        let byte = if addr < program.len() {
            program[addr] as u64
        } else {
            0
        };
        packed_rom_b |= byte << ((addr - 8) * 8);
    }
    // ROM-B at (ox+1, oy+1) — above Mux16to1 (up input)
    ctx.place_with_value(ox + 1, oy + 1, TileType::Const, packed_rom_b);

    ctx.place_with_value(ox, oy + 2, TileType::Const, packed_rom_a);
    ctx.ir_idx = ctx.place(ox + 1, oy + 2, TileType::Mux16to1);
    ctx.place(ox + 2, oy + 2, TileType::WireLeft);
    ctx.place(ox + 3, oy + 2, TileType::WireDown);
    ctx.guard(ox + 4, oy + 2);

    // =========================================================================
    // ROW 3: GUARDS
    // =========================================================================
    for col in 0..NUM_REGISTERS {
        ctx.guard(ox + col, oy + 3);
    }

    // =========================================================================
    // REGISTER FILE — Register8 with feedback Mux at cols ox+5..ox+8
    // =========================================================================
    for reg in 0..NUM_REGISTERS {
        let we_row = oy + 3 + reg * 2;
        let reg_row = we_row + 1;

        ctx.guard(ox + 5, we_row);
        ctx.reg_we_indices[reg] = ctx.place_with_value(ox + 6, we_row, TileType::Const, 0);
        ctx.guard(ox + 7, we_row);
        ctx.guard(ox + 8, we_row);

        ctx.reg_result_indices[reg] = ctx.place_with_value(ox + 5, reg_row, TileType::Const, 0);
        ctx.reg_mux_indices[reg] = ctx.place(ox + 6, reg_row, TileType::Mux);
        ctx.reg_indices[reg] = ctx.place_with_value(ox + 7, reg_row, TileType::Register8, 0);
        ctx.guard(ox + 8, reg_row);
    }

    // Bottom guard row
    for col in 5..=8 {
        ctx.guard(ox + col, oy + 11);
    }

    // =========================================================================
    // FLAGS — Register8 with Mux + Zero/Const feeders at ox+20..24
    // =========================================================================

    // Z flag circuit — oy+3..oy+4
    ctx.guard(ox + 20, oy + 3);
    ctx.guard(ox + 21, oy + 3);
    ctx.flag_z_update_idx = ctx.place_with_value(ox + 22, oy + 3, TileType::Const, 0);
    ctx.guard(ox + 23, oy + 3);
    ctx.guard(ox + 24, oy + 3);

    ctx.flag_z_result_idx = ctx.place_with_value(ox + 20, oy + 4, TileType::Const, 0);
    ctx.flag_z_zero_idx = ctx.place(ox + 21, oy + 4, TileType::Zero);
    ctx.flag_z_mux_idx = ctx.place(ox + 22, oy + 4, TileType::Mux);
    ctx.flag_z_idx = ctx.place_with_value(ox + 23, oy + 4, TileType::Register8, 0);
    ctx.guard(ox + 24, oy + 4);

    // C flag circuit — oy+5..oy+6
    ctx.guard(ox + 20, oy + 5);
    ctx.flag_c_update_idx = ctx.place_with_value(ox + 21, oy + 5, TileType::Const, 0);
    ctx.guard(ox + 22, oy + 5);
    ctx.guard(ox + 23, oy + 5);
    ctx.guard(ox + 24, oy + 5);

    ctx.flag_c_carry_idx = ctx.place_with_value(ox + 20, oy + 6, TileType::Const, 0);
    ctx.flag_c_mux_idx = ctx.place(ox + 21, oy + 6, TileType::Mux);
    ctx.flag_c_idx = ctx.place_with_value(ox + 22, oy + 6, TileType::Register8, 0);
    ctx.guard(ox + 23, oy + 6);
    ctx.guard(ox + 24, oy + 6);

    // Bottom guard row
    for col in 20..=24 {
        ctx.guard(ox + col, oy + 7);
    }

    // =========================================================================
    // OPERAND MUX TREES
    // =========================================================================
    let tree_a = ctx.wire_mux_tree_4to1(ox + 8, oy + 12);
    ctx.op_a_data_indices = tree_a.data_indices;
    ctx.op_a_sel0_indices = tree_a.sel0_indices;
    ctx.op_a_sel1_idx = tree_a.sel1_idx;
    ctx.op_a_leaf_indices = tree_a.leaf_indices;
    ctx.op_a_root_idx = tree_a.root_idx;

    let tree_b = ctx.wire_mux_tree_4to1(ox + 16, oy + 12);
    ctx.op_b_data_indices = tree_b.data_indices;
    ctx.op_b_sel0_indices = tree_b.sel0_indices;
    ctx.op_b_sel1_idx = tree_b.sel1_idx;
    ctx.op_b_leaf_indices = tree_b.leaf_indices;
    ctx.op_b_root_idx = tree_b.root_idx;

    // =========================================================================
    // BUS ROUTING ROW — oy+17
    // =========================================================================
    ctx.guard(ox + 8, oy + 17);
    ctx.guard(ox + 9, oy + 17);
    ctx.place(ox + 10, oy + 17, TileType::WireDown);
    ctx.guard(ox + 11, oy + 17);
    for col in 12..=17 {
        ctx.place(ox + col, oy + 17, TileType::WireLeft);
    }
    ctx.place(ox + 18, oy + 17, TileType::WireDown);
    for col in 19..=22 {
        ctx.guard(ox + col, oy + 17);
    }

    // =========================================================================
    // GUARD ROW — oy+18
    // =========================================================================
    for col in 8..=22 {
        if col == 10 || col == 12 {
            ctx.place(ox + col, oy + 18, TileType::WireDown);
        } else {
            ctx.guard(ox + col, oy + 18);
        }
    }

    // =========================================================================
    // COMPRESSED ALU — 8 operations at oy+19..oy+26
    // =========================================================================
    let alu_types = [
        TileType::AddCarry,
        TileType::SubBorrow,
        TileType::And,
        TileType::Or,
        TileType::Xor,
        TileType::Not,
        TileType::Shl,
        TileType::Shr,
    ];
    for (i, &tt) in alu_types.iter().enumerate() {
        let alu_row = oy + 19 + i;
        ctx.place(ox + 10, alu_row, TileType::WireDown);
        ctx.alu_tile_indices[i] = ctx.place(ox + 11, alu_row, tt);
        if i == 5 {
            ctx.place_with_value(ox + 12, alu_row, TileType::Const, 0);
        } else if i == 6 || i == 7 {
            ctx.place_with_value(ox + 12, alu_row, TileType::Const, 1);
        } else {
            ctx.place(ox + 12, alu_row, TileType::WireDown);
        }
    }

    // Bottom guard row for ALU
    for col in 10..=12 {
        ctx.guard(ox + col, oy + 27);
    }

    // =========================================================================
    // INSTRUCTION DECODER — LUT-based
    // =========================================================================
    //                  NOP   LDI   MOV   ADD   SUB   AND   OR    XOR
    let packed_ctrl_a_lo: u64 = 0x00
        | (0x28 << 8)
        | (0x08 << 16)
        | (0x18 << 24)
        | (0x19u64 << 32)
        | (0x1A << 40)
        | (0x1B << 48)
        | (0x1C << 56);
    //                  SHL   SHR   CMP   JMP   JZ    JNZ   LD    ST
    let packed_ctrl_a_hi: u64 = 0x1E
        | (0x1F << 8)
        | (0x11 << 16)
        | (0x00 << 24)
        | (0x00u64 << 32)
        | (0x00 << 40)
        | (0x08 << 48)
        | (0x00 << 56);
    let packed_ctrl_b_lo: u64 = 0;
    //                  SHL   SHR   CMP   JMP   JZ    JNZ   LD    ST
    let packed_ctrl_b_hi: u64 = 0x00
        | (0x00 << 8)
        | (0x00 << 16)
        | (0x01 << 24)
        | (0x02u64 << 32)
        | (0x04 << 40)
        | (0x08 << 48)
        | (0x10 << 56);

    ctx.guard(ox + 15, oy);
    ctx.guard(ox + 16, oy);
    ctx.decoder_opcode_lo_idx = ctx.place_with_value(ox + 17, oy, TileType::Const, 0);

    ctx.place_with_value(ox + 15, oy + 1, TileType::Const, packed_ctrl_a_lo);
    ctx.decoder_ctrl_a_lo_idx = ctx.place(ox + 16, oy + 1, TileType::Mux8to1);
    ctx.place(ox + 17, oy + 1, TileType::WireDown);

    ctx.place_with_value(ox + 15, oy + 2, TileType::Const, packed_ctrl_a_hi);
    ctx.decoder_ctrl_a_hi_idx = ctx.place(ox + 16, oy + 2, TileType::Mux8to1);
    ctx.place(ox + 17, oy + 2, TileType::WireDown);

    ctx.place_with_value(ox + 15, oy + 3, TileType::Const, packed_ctrl_b_lo);
    ctx.decoder_ctrl_b_lo_idx = ctx.place(ox + 16, oy + 3, TileType::Mux8to1);
    ctx.place(ox + 17, oy + 3, TileType::WireDown);

    ctx.place_with_value(ox + 15, oy + 4, TileType::Const, packed_ctrl_b_hi);
    ctx.decoder_ctrl_b_hi_idx = ctx.place(ox + 16, oy + 4, TileType::Mux8to1);
    ctx.place(ox + 17, oy + 4, TileType::WireDown);

    ctx.guard(ox + 15, oy + 5);
    ctx.guard(ox + 16, oy + 5);
    ctx.guard(ox + 17, oy + 5);

    // =========================================================================
    // ALU RESULT MUX TREE
    // =========================================================================
    let result_tree = ctx.wire_mux_tree_8to1(ox + 2, oy + 28);
    ctx.alu_result_data_indices = result_tree.data_indices;
    ctx.alu_result_sel0_indices = result_tree.sel0_indices;
    ctx.alu_result_sel1_indices = result_tree.sel1_indices;
    ctx.alu_result_sel2_idx = result_tree.sel2_idx;
    ctx.alu_result_leaf_indices = result_tree.leaf_indices;
    ctx.alu_result_mid_indices = result_tree.mid_indices;
    ctx.alu_result_root_idx = result_tree.root_idx;

    // =========================================================================
    // ROM ENTRIES
    // =========================================================================
    let rom_start = oy + 37;
    for addr in 0..effective_rom {
        let val = if addr < program.len() {
            program[addr] as u64
        } else {
            0
        };
        let idx = ctx.place_with_value(
            ox + (addr % 8),
            rom_start + (addr / 8),
            TileType::Const,
            val,
        );
        ctx.rom_indices.push(idx);
    }

    // =========================================================================
    // GUARD ROW — isolate ROM from RAM
    // =========================================================================
    let guard_row = rom_start + 2;
    for col in 0..8 {
        ctx.sim.set_tile(ox + col, guard_row, TileType::Const);
        ctx.sim.set_logic_value(ox + col, guard_row, 0);
    }

    // =========================================================================
    // RAM — Const tiles (software-managed)
    // =========================================================================
    let ram_start = guard_row + 1;
    let effective_ram = ram_size.min(64);
    for addr in 0..effective_ram {
        let idx = ctx.place_with_value(ox + (addr % 8), ram_start + (addr / 8), TileType::Const, 0);
        ctx.ram_indices.push(idx);
    }
}
