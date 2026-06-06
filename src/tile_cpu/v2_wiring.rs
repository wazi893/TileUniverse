//! V2 tile wiring and index capture.
//!
//! Sprint 94 adds clocked commit fabrics for registers/flags while keeping
//! operand and decode selectors software-bridged.

use crate::simulation::Simulation;
use crate::tile_meta::TileType;

/// Sprint 198: ctrl_a encoding for all 32 opcodes, shared between wiring builder
/// and execute-time validation. Single source of truth (Codex finding 3 mitigation).
/// Sprint 111: NEG (index 10) ctrl_a changed from 0xF9 to 0xB9.
pub const CTRL_A_LUT: [u8; 32] = [
    0x00, 0x00, 0x08, 0x58, 0xB8, 0xB9, 0x9A, 0x9B, 0x9C, 0x9D, 0xB9, 0xF8, 0xF9, 0xFE, 0xFF, 0xB1,
    0xF8, 0xF9, 0xDA, 0xDB, 0xDC, 0x00, 0x18, 0x00, 0x58, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Captured tile indices for the V2 hybrid CPU.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct V2CpuIndices {
    pub pc_idx: usize,
    pub pc_next_mux_idx: usize,
    pub pc_ingress_idx: usize, // Sprint 115: L0 ViaUp between Mux and Register8
    // Sprint 152: rom_low_mux_idx / rom_high_mux_idx removed (overwrite eliminated).

    // Sprint 133: per-bank Mux16to1 indices for hybrid bank selection
    pub bank_low_mux_indices: [usize; 4],
    pub bank_high_mux_indices: [usize; 4],
    // bank_byte2/3_mux_indices removed — Sprint 185 physical ir_ext

    // Sprint 150: physical selector mux outputs (Final Mux per lane)
    pub rom_selected_low_idx: usize,
    pub rom_selected_high_idx: usize,
    // Sprint 185: physical byte2/byte3 selector outputs (lanes 2/3)
    pub rom_selected_byte2_idx: usize,
    pub rom_selected_byte3_idx: usize,

    /// Sprint 187: Bank 4-7 Mux16to1 output indices [bank_within_group][lane]
    /// lane order: low(0), high(1), byte2(2), byte3(3)
    pub bank47_mux_indices: [[usize; 4]; 4],

    /// Sprint 211: Original Final Mux indices (Banks 0-3 physical selector output).
    pub final_mux_indices: [usize; 4],
    /// Sprint 211: Super Mux output indices (one per lane) — unified ROM selector.
    pub super_mux_indices: [usize; 4],
    /// Sprint 211: Const injection tiles for Super Mux LEFT input (Banks 4-7 value).
    pub super_mux_inject_indices: [usize; 4],
    /// Sprint 211: Const injection tiles for Super Mux RIGHT input (Banks 0-3 value).
    pub super_mux_inject_right_indices: [usize; 4],

    // Sprint 150 Phase 2E: hybrid injection points for extraction pipeline delivery.
    // Selector outputs are injected into L2 Const guards; ViaUp chain propagates to L0.
    // Sprint 151: ir_high_l2_inject_idx, ir_low_l2_const_idx, ir_low_l2_wire_idx,
    // ir_high_l1_via_idx, ir_low_l1_via_idx removed — physical westback routes
    // deliver ir_low/ir_high through tile propagation, no injection needed.
    pub reg_indices: [usize; 16],
    // Phase 97.0 groundwork: L1 taps for committed register values.
    pub reg_tap_l1_indices: [usize; 16],
    pub flag_z_idx: usize,
    pub flag_c_idx: usize,
    pub ram_indices: [usize; 128],

    // Physical extraction outputs (Sprint 95)
    pub extract_opcode_shr_idx: usize,
    pub extract_opcode_bit4_idx: usize,
    pub extract_rd_bit_indices: [usize; 3], // [rd0, rd1, rd2]
    pub extract_rs_field_idx: usize,

    // Decoder outputs
    pub ctrl_a_mux_idx: usize,
    pub ctrl_b_mux_idx: usize,
    pub branch_ctrl_b_l1_tap_idx: usize,
    pub branch_flag_z_l1_tap_idx: usize,
    pub branch_flag_c_l1_tap_idx: usize,
    pub branch_taken_core_idx: usize,
    pub branch_dirty_indices: Vec<usize>,
    /// Sprint 154: Flag-only subset of branch_dirty_indices (flag_z + flag_c routes).
    /// Decode routes (ctrl_b, selector_low, bank_sel) are stable from Stage-F.
    pub branch_flag_dirty_indices: Vec<usize>,

    // Operand tree A/B roots.
    pub op_a_root_idx: usize,
    pub op_b_root_idx: usize,
    // Full dirty propagation set: fetch -> extraction -> decode -> operand trees.
    // Retained for full-resync fallback.
    pub pipeline_dirty_indices: Vec<usize>,
    // Always-dirty frontend path.
    pub pipeline_backbone_dirty_indices: Vec<usize>,
    // Register-value dependent data routes (per source register).
    pub pipeline_reg_data_dirty_indices: [Vec<usize>; 16],

    // Sprint 195: physical Decoder3to8 tile for rd field one-hot decode.
    pub rd_onehot_decode_idx: usize,
    // Sprint 197: L0 WireRight chain downstream of Decoder3to8, for synth dirty seeding.
    pub rd_decode_l0_chain: Vec<usize>,

    // Sprint 198: RAM write address decoder + gate for synth replacement.
    pub ram_write_decode_idx: usize,
    pub ram_write_gate_idx: usize,

    // Sprint 94: clocked commit fabric
    pub we_mask_const_idx: usize,
    pub commit_dirty_indices: Vec<usize>,
    // Sprint 169: register writeback path (conditional on reg_write).
    pub reg_wb_dirty_indices: Vec<usize>,

    pub flag_we_mask_const_idx: usize,
    pub flag_commit_dirty_indices: Vec<usize>,

    // Sprint 125: physical LR and Target Selection Mux
    pub lr_idx: usize,
    pub lr_dirty_indices: Vec<usize>,

    /// Sprint 218: ALU 8:1 result mux root — carries the selected ALU output.
    pub wb_alu_root_idx: usize,
    /// Sprint 218: wb_data Mux at (ox+38, oy+10) — final writeback data source.
    pub wb_data_mux_idx: usize,

    /// Sprint 235: Add tile L-operand enable Const at (ox+16, oy+37).
    /// Set to MAX for MOV/LDI (zeroes out A operand: Add output = 0+B = B).
    /// Set to 0 for ALU ops (A operand flows through: Add output = A+B).
    pub alu_add_l_enable_idx: usize,
    /// Sprint 235: Add tile L-operand Mux at (ox+16, oy+38).
    /// Downstream of alu_add_l_enable_idx — must be dirtied after injection.
    pub alu_add_l_mux_idx: usize,

    /// Sprint 236: Per-register wb_data Const injection points for R8-R15.
    /// Injected with the writeback value before commit, restored to 0 after.
    pub high_reg_wb_data_const_indices: [usize; 8],
    /// Sprint 236: Per-register WE bit Const injection points for R8-R15.
    /// Injected with MAX (write enabled) or left at 0 (no write).
    pub high_reg_we_const_indices: [usize; 8],
    /// Sprint 236: Merge mux tiles for R8-R15 (dirty targets after injection).
    pub high_reg_merge_mux_indices: [usize; 8],

    /// Sprint 237: High-register operand tree A root (R8-R15 via rd[0:2]).
    pub op_a_high_root_idx: usize,
    /// Sprint 237: High-register operand tree B root (R8-R15 via rs[0:2]).
    pub op_b_high_root_idx: usize,
    /// Sprint 237: Top Mux A — selects low tree root vs high tree root via rd_hi.
    pub top_mux_a_idx: usize,
    /// Sprint 237: Top Mux B — selects low tree root vs high tree root via rs_hi.
    pub top_mux_b_idx: usize,
    /// Sprint 237: rd_hi selector Const for Top Mux A.
    pub top_mux_a_sel_const_idx: usize,
    /// Sprint 237: rs_hi selector Const for Top Mux B.
    pub top_mux_b_sel_const_idx: usize,
    /// Sprint 237: High Tree A selector Const indices (7 Consts: 4×sel0 + 2×sel1 + 1×sel2).
    pub high_tree_a_sel_const_indices: [usize; 7],
    /// Sprint 237: High Tree B selector Const indices.
    pub high_tree_b_sel_const_indices: [usize; 7],
    /// Sprint 237: High Tree A data injection Consts (8 Consts, indexed [0]=R8 .. [7]=R15).
    pub high_tree_a_data_const_indices: [usize; 8],
    /// Sprint 237: High Tree B data injection Consts (bank-swapped: [0]=R12 .. [3]=R15, [4]=R8 .. [7]=R11).
    pub high_tree_b_data_const_indices: [usize; 8],
    /// Sprint 237: Dirty sets for Const-injection propagation (Option 2-plus).
    /// Stored because mark_dirty on a Const tile itself is a no-op (eval sees current==current).
    pub high_tree_a_dirty_indices: Vec<usize>,
    pub high_tree_b_dirty_indices: Vec<usize>,
    pub top_mux_a_dirty_indices: Vec<usize>,
    pub top_mux_b_dirty_indices: Vec<usize>,

    /// Sprint 239: Terminal L1 trunk tiles that feed the ALU horizontal rows.
    /// Injection targets for high-register operand re-source.
    pub alu_a_trunk_terminal_idx: usize,
    pub alu_b_trunk_terminal_idx: usize,
    /// Sprint 239: Downstream dirty seeds — horizontal distribution + ALU input tiles.
    pub alu_a_downstream_dirty: Vec<usize>,
    pub alu_b_downstream_dirty: Vec<usize>,

    /// Sprint 243: R Mux output WireDown tiles — last tile before each ALU column's RIGHT input.
    /// Index i corresponds to ALU column i (0=Add, 1=Sub, 2=And, 3=Or, 4=Xor, 5=Not, 6=Shl, 7=Shr).
    /// Injection target for wide-immediate override.
    pub alu_r_mux_output_indices: [usize; 8],

    pub tile_count: usize,
    pub grid_width: usize,

    // Sprint 146: zone closure scopes for scoped propagation (computed by builder)
    pub pipeline_scope: Vec<u32>,
    pub branch_scope: Vec<u32>,
    pub commit_scope: Vec<u32>,

    // Sprint 147: bitset-form scope masks for O(L1+active) masked drain
    pub pipeline_scope_mask: Vec<u64>,
    pub branch_scope_mask: Vec<u64>,
    pub commit_scope_mask: Vec<u64>,

    // Sprint 262: topologically sorted evaluation order for levelized propagation
    pub pipeline_eval_order: Vec<usize>,
    // Sprint 263: branch + commit scope eval orders
    pub branch_eval_order: Vec<usize>,
    pub commit_eval_order: Vec<usize>,

    // Sprint 166: unified clock scope = pipeline | branch | commit (for masked clock tick)
    pub clock_scope_mask: Vec<u64>,
    // Sprint 276: topological eval order for compact clock edge propagation
    pub clock_eval_order: Vec<usize>,

    // Sprint 168: Pre-filtered clock-sensitive tiles within clock_scope_mask.
    pub in_scope_clock_cache: Vec<usize>,

    // Sprint 147: bitset of all tiles placed during wiring (for restricted BFS)
    pub placed_mask: Vec<u64>,
}

#[derive(Debug, Clone)]
struct V2OperandTreeIndices {
    root_idx: usize,
    dirty_indices: Vec<usize>,
    /// Selector Const indices: [sel0_0, sel0_1, sel0_2, sel0_3, sel1_left, sel1_right, sel2]
    sel_const_indices: [usize; 7],
}

fn pack_8(bytes: &[u8]) -> u64 {
    let mut packed = 0u64;
    for (i, &b) in bytes.iter().take(8).enumerate() {
        packed |= (b as u64) << (i * 8);
    }
    packed
}

// 8:1 binary mux tree with dedicated selector Consts.
//
// Layout is 15x9 with data anchors at row y+1 and root at (x+7, y+8):
//   D0..D3 on right subtree (x+14,12,10,8)
//   D4..D7 on left subtree  (x+6,4,2,0)
fn wire_operand_tree_8to1(
    sim: &mut Simulation,
    grid_width: usize,
    tile_count: &mut usize,
    placed_mask: &mut [u64],
    x: usize,
    y: usize,
) -> V2OperandTreeIndices {
    let mut dirty_indices = Vec::new();

    macro_rules! place {
        ($px:expr, $py:expr, $tt:expr) => {{
            sim.set_tile($px, $py, $tt);
            *tile_count += 1;
            let _idx = $py * grid_width + $px;
            placed_mask[_idx / 64] |= 1u64 << (_idx % 64);
            _idx
        }};
    }
    macro_rules! place_val {
        ($px:expr, $py:expr, $tt:expr, $v:expr) => {{
            sim.set_tile($px, $py, $tt);
            sim.set_logic_value($px, $py, $v);
            *tile_count += 1;
            let _idx = $py * grid_width + $px;
            placed_mask[_idx / 64] |= 1u64 << (_idx % 64);
            _idx
        }};
    }

    // Row y: selector Consts for leaf muxes
    let _sel0_0 = place_val!(x + 1, y, TileType::Const, 0);
    let _sel0_1 = place_val!(x + 5, y, TileType::Const, 0);
    let _sel0_2 = place_val!(x + 9, y, TileType::Const, 0);
    let _sel0_3 = place_val!(x + 13, y, TileType::Const, 0);

    // Row y+1: data anchors + leaf muxes
    // D ordering is canonical: D0 rightmost, D7 leftmost.
    let idx = place!(x, y + 1, TileType::WireDown);
    dirty_indices.push(idx);
    let l_m67 = place!(x + 1, y + 1, TileType::Mux);
    dirty_indices.push(l_m67);
    let idx = place!(x + 2, y + 1, TileType::WireDown);
    dirty_indices.push(idx);
    let idx = place!(x + 4, y + 1, TileType::WireDown);
    dirty_indices.push(idx);
    let l_m45 = place!(x + 5, y + 1, TileType::Mux);
    dirty_indices.push(l_m45);
    let idx = place!(x + 6, y + 1, TileType::WireDown);
    dirty_indices.push(idx);

    let idx = place!(x + 8, y + 1, TileType::WireDown);
    dirty_indices.push(idx);
    let l_m23 = place!(x + 9, y + 1, TileType::Mux);
    dirty_indices.push(l_m23);
    let idx = place!(x + 10, y + 1, TileType::WireDown);
    dirty_indices.push(idx);
    let idx = place!(x + 12, y + 1, TileType::WireDown);
    dirty_indices.push(idx);
    let l_m01 = place!(x + 13, y + 1, TileType::Mux);
    dirty_indices.push(l_m01);
    let idx = place!(x + 14, y + 1, TileType::WireDown);
    dirty_indices.push(idx);

    // Row y+2: leaf outputs descend
    for cx in [x + 1, x + 5, x + 9, x + 13] {
        let idx = place!(cx, y + 2, TileType::WireDown);
        dirty_indices.push(idx);
    }

    // Row y+3: leaf outputs descend + mid selector Consts
    for cx in [x + 1, x + 5, x + 9, x + 13] {
        let idx = place!(cx, y + 3, TileType::WireDown);
        dirty_indices.push(idx);
    }
    let _sel1_left = place_val!(x + 2, y + 3, TileType::Const, 0);
    let _sel1_right = place_val!(x + 10, y + 3, TileType::Const, 0);

    // Row y+4: middle mux layer
    for cx in [x + 1, x + 5, x + 9, x + 13] {
        let idx = place!(cx, y + 4, TileType::WireDown);
        dirty_indices.push(idx);
    }
    for cx in [x + 3, x + 4, x + 11, x + 12] {
        let idx = place!(cx, y + 4, TileType::WireLeft);
        dirty_indices.push(idx);
    }
    let m_left = place!(x + 2, y + 4, TileType::Mux);
    dirty_indices.push(m_left);
    let m_right = place!(x + 10, y + 4, TileType::Mux);
    dirty_indices.push(m_right);

    // Row y+5: middle outputs descend
    for cx in [x + 2, x + 10] {
        let idx = place!(cx, y + 5, TileType::WireDown);
        dirty_indices.push(idx);
    }

    // Row y+6: middle outputs descend + root selector Const
    for cx in [x + 2, x + 10] {
        let idx = place!(cx, y + 6, TileType::WireDown);
        dirty_indices.push(idx);
    }
    let _sel2 = place_val!(x + 7, y + 6, TileType::Const, 0);

    // Row y+7: selector descends + middle outputs descend
    for cx in [x + 2, x + 7, x + 10] {
        let idx = place!(cx, y + 7, TileType::WireDown);
        dirty_indices.push(idx);
    }

    // Row y+8: root layer
    for cx in [x + 2, x + 10] {
        let idx = place!(cx, y + 8, TileType::WireDown);
        dirty_indices.push(idx);
    }
    for cx in [x + 3, x + 4, x + 5, x + 6] {
        let idx = place!(cx, y + 8, TileType::WireRight);
        dirty_indices.push(idx);
    }
    for cx in [x + 8, x + 9] {
        let idx = place!(cx, y + 8, TileType::WireLeft);
        dirty_indices.push(idx);
    }
    let root = place!(x + 7, y + 8, TileType::Mux);
    dirty_indices.push(root);

    V2OperandTreeIndices {
        root_idx: root,
        dirty_indices,
        sel_const_indices: [
            _sel0_0,
            _sel0_1,
            _sel0_2,
            _sel0_3,
            _sel1_left,
            _sel1_right,
            _sel2,
        ],
    }
}

/// Place V2 hybrid tiles and return all required indices.
pub(crate) fn wire_v2_cpu(
    sim: &mut Simulation,
    origin: (usize, usize),
    program: &[u32],
    rom_size: usize,
    _ram_size: usize,
    pc_addr_bits: u8,
) -> V2CpuIndices {
    assert!(
        sim.num_layers() >= 4,
        "V2 Sprint 118 requires at least 4 layers (use Simulation::with_size_layered(..., ..., 4))"
    );

    let (ox, oy) = origin;
    let grid_width = sim.width();
    let layer_size = sim.width() * sim.height();
    let mut tile_count = 0usize;
    let idx3d = |x: usize, y: usize, z: usize| -> usize { z * layer_size + y * grid_width + x };

    // Sprint 147: track all placed tile indices for restricted BFS.
    let num_segments = (sim.tile_count() + 63) / 64;
    let mut placed_mask = vec![0u64; num_segments];

    macro_rules! place {
        ($x:expr, $y:expr, $tt:expr) => {{
            sim.set_tile($x, $y, $tt);
            tile_count += 1;
            let _idx = idx3d($x, $y, 0);
            placed_mask[_idx / 64] |= 1u64 << (_idx % 64);
            _idx
        }};
    }
    macro_rules! place_val {
        ($x:expr, $y:expr, $tt:expr, $v:expr) => {{
            sim.set_tile($x, $y, $tt);
            sim.set_logic_value($x, $y, $v);
            tile_count += 1;
            let _idx = idx3d($x, $y, 0);
            placed_mask[_idx / 64] |= 1u64 << (_idx % 64);
            _idx
        }};
    }
    macro_rules! place_l1 {
        ($x:expr, $y:expr, $tt:expr) => {{
            sim.set_tile_3d($x, $y, 1, $tt);
            tile_count += 1;
            let _idx = idx3d($x, $y, 1);
            placed_mask[_idx / 64] |= 1u64 << (_idx % 64);
            _idx
        }};
    }
    macro_rules! place_l1_val {
        ($x:expr, $y:expr, $tt:expr, $v:expr) => {{
            sim.set_tile_3d($x, $y, 1, $tt);
            sim.set_logic_value_3d($x, $y, 1, $v);
            tile_count += 1;
            let _idx = idx3d($x, $y, 1);
            placed_mask[_idx / 64] |= 1u64 << (_idx % 64);
            _idx
        }};
    }
    macro_rules! place_l2 {
        ($x:expr, $y:expr, $tt:expr) => {{
            sim.set_tile_3d($x, $y, 2, $tt);
            tile_count += 1;
            let _idx = idx3d($x, $y, 2);
            placed_mask[_idx / 64] |= 1u64 << (_idx % 64);
            _idx
        }};
    }
    macro_rules! place_l2_val {
        ($x:expr, $y:expr, $tt:expr, $v:expr) => {{
            sim.set_tile_3d($x, $y, 2, $tt);
            sim.set_logic_value_3d($x, $y, 2, $v);
            tile_count += 1;
            let _idx = idx3d($x, $y, 2);
            placed_mask[_idx / 64] |= 1u64 << (_idx % 64);
            _idx
        }};
    }
    macro_rules! place_l3 {
        ($x:expr, $y:expr, $tt:expr) => {{
            sim.set_tile_3d($x, $y, 3, $tt);
            tile_count += 1;
            let _idx = idx3d($x, $y, 3);
            placed_mask[_idx / 64] |= 1u64 << (_idx % 64);
            _idx
        }};
    }
    macro_rules! place_l3_val {
        ($x:expr, $y:expr, $tt:expr, $v:expr) => {{
            sim.set_tile_3d($x, $y, 3, $tt);
            sim.set_logic_value_3d($x, $y, 3, $v);
            tile_count += 1;
            let _idx = idx3d($x, $y, 3);
            placed_mask[_idx / 64] |= 1u64 << (_idx % 64);
            _idx
        }};
    }

    // Guard the V2 CPU region with Const(0) to prevent default Wire leakage.
    // Real tiles placed below intentionally overwrite guards where needed.
    for y in oy..(oy + 64) {
        for x in ox..(ox + 96) {
            let _ = place_val!(x, y, TileType::Const, 0);
        }
    }

    // L1 routing guards (corridor-focused, deduplicated at intersections)
    let mut l1_guarded = vec![false; layer_size];
    macro_rules! guard_l1_rect {
        ($x0:expr, $x1:expr, $y0:expr, $y1:expr) => {{
            for gy in $y0..=$y1 {
                for gx in $x0..=$x1 {
                    let local = gy * grid_width + gx;
                    if !l1_guarded[local] {
                        let _ = place_l1_val!(gx, gy, TileType::Const, 0);
                        l1_guarded[local] = true;
                    }
                }
            }
        }};
    }
    // Horizontal wb-data corridor
    guard_l1_rect!(ox + 15, ox + 47, oy + 9, oy + 11);
    // Bank A / Bank B vertical delivery corridors
    guard_l1_rect!(ox + 24, ox + 26, oy + 10, oy + 23);
    guard_l1_rect!(ox + 44, ox + 46, oy + 10, oy + 23);
    // Flag wb-data south route
    guard_l1_rect!(ox + 36, ox + 38, oy + 10, oy + 51);
    // Flag row fanout corridor
    guard_l1_rect!(ox + 16, ox + 37, oy + 48, oy + 51);
    // Sprint 95 Northern Corridor guard lanes around row oy+1
    guard_l1_rect!(ox + 1, ox + 56, oy, oy); // row oy+0
    guard_l1_rect!(ox + 3, ox + 54, oy + 2, oy + 2); // row oy+2
    guard_l1_rect!(ox + 1, ox + 3, oy + 1, oy + 5); // lift flanks
    guard_l1_rect!(ox + 54, ox + 56, oy + 1, oy + 29); // descent flanks
    guard_l1_rect!(ox + 13, ox + 13, oy + 1, oy + 1); // above ir_high tap
    // Sprint 133: extend L1 guard band for ROM area — prevents leakage from
    // ViaDown taps placed at bank Mux16to1 positions. Fills in ALL unguarded
    // L1 positions at oy+1..5 with Const(0). Existing guards are skipped
    // (l1_guarded check). Subsequent routing code overwrites specific guards.
    guard_l1_rect!(ox, ox + 56, oy + 1, oy + 5);
    // Sprint 107: guard L1 corridor for branch selector physical assembly.
    guard_l1_rect!(ox + 38, ox + 69, oy + 49, oy + 53);

    // Sprint 97: guard L2 routing plane to avoid default-Wire leakage.
    for y in oy..(oy + 64) {
        for x in ox..(ox + 96) {
            let _ = place_l2_val!(x, y, TileType::Const, 0);
        }
    }

    // Sprint 118: guard L3 routing plane to avoid default-Wire leakage.
    for y in oy..(oy + 64) {
        for x in ox..(ox + 96) {
            let _ = place_l3_val!(x, y, TileType::Const, 0);
        }
    }

    // L1 route-ownership map to detect accidental coordinate reuse.
    let mut l1_reserved = vec![false; layer_size];
    macro_rules! reserve_l1 {
        ($x:expr, $y:expr) => {{
            let local = $y * grid_width + $x;
            assert!(
                !l1_reserved[local],
                "Sprint95 L1 collision at ({}, {})",
                $x, $y
            );
            l1_reserved[local] = true;
        }};
    }
    macro_rules! place_l1_route {
        ($x:expr, $y:expr, $tt:expr) => {{
            reserve_l1!($x, $y);
            place_l1!($x, $y, $tt)
        }};
    }
    let mut l2_reserved = vec![false; layer_size];
    macro_rules! reserve_l2 {
        ($x:expr, $y:expr) => {{
            let local = $y * grid_width + $x;
            assert!(
                !l2_reserved[local],
                "Sprint97 L2 collision at ({}, {})",
                $x, $y
            );
            l2_reserved[local] = true;
        }};
    }
    macro_rules! place_l2_route {
        ($x:expr, $y:expr, $tt:expr) => {{
            reserve_l2!($x, $y);
            place_l2!($x, $y, $tt)
        }};
    }
    let mut l3_reserved = vec![false; layer_size];
    macro_rules! reserve_l3 {
        ($x:expr, $y:expr) => {{
            let local = $y * grid_width + $x;
            assert!(
                !l3_reserved[local],
                "Sprint118 L3 collision at ({}, {})",
                $x, $y
            );
            l3_reserved[local] = true;
        }};
    }
    macro_rules! place_l3_route {
        ($x:expr, $y:expr, $tt:expr) => {{
            reserve_l3!($x, $y);
            place_l3!($x, $y, $tt)
        }};
    }

    let mut pipeline_dirty_indices = Vec::new();
    let mut pipeline_backbone_dirty_indices = Vec::new();
    let mut pipeline_reg_data_dirty_indices: [Vec<usize>; 16] = std::array::from_fn(|_| Vec::new());
    macro_rules! push_pipeline {
        ($idx:expr) => {{
            pipeline_dirty_indices.push($idx);
            pipeline_backbone_dirty_indices.push($idx);
        }};
    }
    macro_rules! push_reg_data {
        ($reg:expr, $idx:expr) => {{
            pipeline_dirty_indices.push($idx);
            pipeline_reg_data_dirty_indices[$reg].push($idx);
        }};
    }

    // -------------------------------------------------------------------------
    // PC ingress + fallthrough compute (Sprint 100)
    // -------------------------------------------------------------------------
    // L0 ingress tile consumed by PC Register8.LEFT.
    let pc_ingress_idx = place!(ox + 2, oy + 1, TileType::ViaUp);
    // Sprint 370 (Gate B.3): the PC register is Register64 in wide mode (holds a
    // 16-bit value), else Register8. The mask-via below caps the architectural width.
    let pc_reg_type = if pc_addr_bits > 8 {
        TileType::Register64
    } else {
        TileType::Register8
    };
    let pc_idx = place_val!(ox + 3, oy + 1, pc_reg_type, 0);
    // Fallthrough compute: pc_plus_one = Add(pc, 1).
    let pc_plus_one_add_idx = place!(ox + 4, oy + 1, TileType::Add);
    let _pc_plus_one_const_idx = place_val!(ox + 5, oy + 1, TileType::Const, 1);

    // L1/L2 ingress mux fabric:
    //   pc_next = override_enable ? override_target : ((pc_plus_one) & 0x0F)
    // Route ownership must be explicit to avoid silent coordinate reuse.
    //
    // Sprint 110: PC override enable is now physical (branch_taken_core via Or gate).
    // L0 (ox+1, oy): legacy software enable Const (kept at 0; physical path owns PC enable)
    // L0 (ox+2, oy): Or gate [LEFT=software, RIGHT=physical branch_taken_core]
    // L0 (ox+3, oy): ViaUp (reads L1 ViaUp = L2 branch_taken_core arrival)
    // L1 (ox+2, oy): ViaDown (reads L0 Or → feeds pc_next Mux UP)
    // L1 (ox+3, oy): ViaUp (reads L2 → delivers branch_taken_core to L0)
    let _pc_override_software_enable_idx = place_val!(ox + 1, oy, TileType::Const, 0);
    let _pc_enable_or_idx = place!(ox + 2, oy, TileType::Or);
    let _pc_enable_physical_via_l0 = place!(ox + 3, oy, TileType::ViaUp);
    reserve_l1!(ox + 2, oy);
    let _pc_enable_viadown_l1 = place_l1!(ox + 2, oy, TileType::ViaDown);
    let _pc_enable_physical_via_l1 = place_l1_route!(ox + 3, oy, TileType::ViaUp);

    // Sprint 125: changed from WireUp (ir_low direct) to ViaUp (reads L2 Target Mux output).
    // ir_low now reaches pc_next_mux via L1/L2 east route → Target Selection Mux → L3 → L2/L1.
    let _pc_target_viaup_l1 = place_l1_route!(ox + 1, oy + 1, TileType::ViaUp);
    let pc_plus_one_tap_l1 = place_l1_route!(ox + 4, oy + 1, TileType::ViaDown);
    reserve_l2!(ox + 4, oy + 1);
    let pc_plus_one_tap_l2 = place_l2!(ox + 4, oy + 1, TileType::ViaDown);
    // Sprint 207: PC mask fused via — replaces L2 Const(0x7F) + L2 And with
    // L2 WireRight (forwards pc_plus_one) + L1 WeightedViaUp(shift=0, mask=0x7F).
    // Frees L2(ox+2, oy+1) formerly occupied by Const(0x7F).
    // (Sprint 187: 7-bit PC mask, was 0x3F.)
    let pc_mask_wire_l2 = place_l2_route!(ox + 3, oy + 1, TileType::WireLeft);
    let pc_mask_up_l1 = place_l1_route!(ox + 3, oy + 1, TileType::WeightedViaUp);
    // Sprint 369/370 (Gate B.2/B.3): set the fallthrough PC mask to the architectural
    // width (7-bit default, 8-bit extended, 16-bit wide) so PC+1 wraps at the right
    // boundary. Must be set here (before build_compact_ops bakes the mask into the
    // hot path). pc_addr_bits in {7,8,16}.
    let pc_mask_val: u64 = (1u64 << pc_addr_bits) - 1;
    sim.set_tile_mask(pc_mask_up_l1, pc_mask_val);
    // shift defaults to 0 — no call to set_tile_shift needed
    let pc_next_mux_idx = place_l1_route!(ox + 2, oy + 1, TileType::Mux);

    let mut pc_path_dirty_indices = vec![
        pc_plus_one_add_idx,
        pc_plus_one_tap_l1,
        pc_plus_one_tap_l2,
        pc_mask_wire_l2,
        pc_mask_up_l1,
        pc_next_mux_idx,
        pc_ingress_idx,
        _pc_enable_or_idx,
        _pc_enable_physical_via_l0,
        _pc_enable_viadown_l1,
        _pc_enable_physical_via_l1,
    ];

    // -------------------------------------------------------------------------
    // Sprint 115: physical ir_low delivery to PC override target
    // Sprint 125: L1(ox+1, oy+1) changed from WireUp to ViaUp. ir_low chain
    // from oy+3 to oy+2 still works (ViaDown -> WireUp). The WireUp at oy+2
    // is now orphaned (nobody reads it from L1), but ir_low is tapped at oy+3
    // via the new L1 east route (Phase 8) to feed the Target Selection Mux.
    // Sprint 138: HALT also uses this physical target path (self-target encoded
    // into ROM low byte at build time).
    // -------------------------------------------------------------------------
    let _ir_low_tap_l1 = place_l1_route!(ox + 1, oy + 3, TileType::ViaDown);
    let _ir_low_wu_l1 = place_l1_route!(ox + 1, oy + 2, TileType::WireUp);
    // Sprint 125: L1(ox+1, oy+1) is ViaUp now, see above.

    // -------------------------------------------------------------------------
    // Sprint 133: 64-entry quad-lane ROM fetch using banked Mux16to1 tiles
    // -------------------------------------------------------------------------
    // Sprint 187: ROM expanded to 128 entries (was 64).
    let effective_rom = rom_size.min(128);
    let mut low_bytes = [0u8; 128];
    let mut high_bytes = [0u8; 128];
    let mut byte2_bytes = [0u8; 128];
    let mut byte3_bytes = [0u8; 128];
    for addr in 0..effective_rom {
        let mut word = if addr < program.len() {
            program[addr]
        } else {
            0
        };
        // Sprint 138: encode HALT target physically as "jump to self" by
        // patching ir_low to the instruction address at ROM build time.
        // Sprint 187: 7-bit PC mask (was 0x3F).
        if ((word >> 11) & 0x1F) as u8 == 0x01 {
            word = (word & !0xFF) | ((addr as u32) & 0x7F);
        }
        low_bytes[addr] = (word & 0xFF) as u8;
        high_bytes[addr] = ((word >> 8) & 0xFF) as u8;
        byte2_bytes[addr] = ((word >> 16) & 0xFF) as u8;
        byte3_bytes[addr] = ((word >> 24) & 0xFF) as u8;
    }

    // Bank 0 — Low lane: Const(up)=8..15, Const(left)=0..7, selector from right(PC[3:0]).
    let _ = place_val!(ox + 1, oy + 1, TileType::Const, pack_8(&low_bytes[8..16]));
    let _ = place_val!(ox, oy + 2, TileType::Const, pack_8(&low_bytes[0..8]));
    let bank0_low_mux_idx = place!(ox + 1, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank0_low_mux_idx);
    let rom_low_sel_wl_idx = place!(ox + 2, oy + 2, TileType::WireLeft);
    push_pipeline!(rom_low_sel_wl_idx);
    let rom_low_sel_wd_idx = place!(ox + 3, oy + 2, TileType::WireDown); // PC down
    push_pipeline!(rom_low_sel_wd_idx);

    // Bank 0 — High lane (offset east to avoid overlap).
    let _ = place_val!(ox + 13, oy + 1, TileType::Const, pack_8(&high_bytes[8..16]));
    let _ = place_val!(ox + 12, oy + 2, TileType::Const, pack_8(&high_bytes[0..8]));
    let bank0_high_mux_idx = place!(ox + 13, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank0_high_mux_idx);
    for x in (ox + 4)..=(ox + 11) {
        let idx = place!(x, oy + 2, TileType::WireRight);
        push_pipeline!(idx);
    }
    let rom_high_detour_down_idx = place!(ox + 11, oy + 3, TileType::WireDown);
    push_pipeline!(rom_high_detour_down_idx);
    for x in (ox + 12)..=(ox + 14) {
        let idx = place!(x, oy + 3, TileType::WireRight);
        push_pipeline!(idx);
    }
    let rom_high_sel_wu_idx = place!(ox + 14, oy + 2, TileType::WireUp);
    push_pipeline!(rom_high_sel_wu_idx);

    // Bank 0 — Byte2 lane (extension low byte, bits 23-16 of 32-bit instruction)
    let _ = place_val!(
        ox + 16,
        oy + 1,
        TileType::Const,
        pack_8(&byte2_bytes[8..16])
    );
    let _ = place_val!(ox + 15, oy + 2, TileType::Const, pack_8(&byte2_bytes[0..8]));
    let bank0_byte2_mux_idx = place!(ox + 16, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank0_byte2_mux_idx);
    let byte2_sel_wu = place!(ox + 17, oy + 2, TileType::WireUp);
    push_pipeline!(byte2_sel_wu);

    // Bank 0 — Byte3 lane (extension high byte, bits 31-24 of 32-bit instruction)
    let _ = place_val!(
        ox + 19,
        oy + 1,
        TileType::Const,
        pack_8(&byte3_bytes[8..16])
    );
    let _ = place_val!(ox + 18, oy + 2, TileType::Const, pack_8(&byte3_bytes[0..8]));
    let bank0_byte3_mux_idx = place!(ox + 19, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank0_byte3_mux_idx);
    let byte3_sel_wu = place!(ox + 20, oy + 2, TileType::WireUp);
    push_pipeline!(byte3_sel_wu);

    // Sprint 132: Extend PC distribution at oy+3 for byte2/byte3 selectors
    for x in (ox + 15)..=(ox + 20) {
        let idx = place!(x, oy + 3, TileType::WireRight);
        push_pipeline!(idx);
    }

    // Sprint 133: Extend PC distribution at oy+3 to reach Banks 1-3 selectors.
    for x in (ox + 21)..=(ox + 56) {
        let idx = place!(x, oy + 3, TileType::WireRight);
        push_pipeline!(idx);
    }

    // Sprint 133: Bank 1 (instructions 16-31) — 4 lanes.
    let _ = place_val!(ox + 22, oy + 1, TileType::Const, pack_8(&low_bytes[24..32]));
    let _ = place_val!(ox + 21, oy + 2, TileType::Const, pack_8(&low_bytes[16..24]));
    let bank1_low_mux_idx = place!(ox + 22, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank1_low_mux_idx);
    let idx = place!(ox + 23, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 25,
        oy + 1,
        TileType::Const,
        pack_8(&high_bytes[24..32])
    );
    let _ = place_val!(
        ox + 24,
        oy + 2,
        TileType::Const,
        pack_8(&high_bytes[16..24])
    );
    let bank1_high_mux_idx = place!(ox + 25, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank1_high_mux_idx);
    let idx = place!(ox + 26, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 28,
        oy + 1,
        TileType::Const,
        pack_8(&byte2_bytes[24..32])
    );
    let _ = place_val!(
        ox + 27,
        oy + 2,
        TileType::Const,
        pack_8(&byte2_bytes[16..24])
    );
    let bank1_byte2_mux_idx = place!(ox + 28, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank1_byte2_mux_idx);
    let idx = place!(ox + 29, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 31,
        oy + 1,
        TileType::Const,
        pack_8(&byte3_bytes[24..32])
    );
    let _ = place_val!(
        ox + 30,
        oy + 2,
        TileType::Const,
        pack_8(&byte3_bytes[16..24])
    );
    let bank1_byte3_mux_idx = place!(ox + 31, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank1_byte3_mux_idx);
    let idx = place!(ox + 32, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    // Sprint 133: Bank 2 (instructions 32-47) — 4 lanes.
    let _ = place_val!(ox + 34, oy + 1, TileType::Const, pack_8(&low_bytes[40..48]));
    let _ = place_val!(ox + 33, oy + 2, TileType::Const, pack_8(&low_bytes[32..40]));
    let bank2_low_mux_idx = place!(ox + 34, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank2_low_mux_idx);
    let idx = place!(ox + 35, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 37,
        oy + 1,
        TileType::Const,
        pack_8(&high_bytes[40..48])
    );
    let _ = place_val!(
        ox + 36,
        oy + 2,
        TileType::Const,
        pack_8(&high_bytes[32..40])
    );
    let bank2_high_mux_idx = place!(ox + 37, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank2_high_mux_idx);
    let idx = place!(ox + 38, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 40,
        oy + 1,
        TileType::Const,
        pack_8(&byte2_bytes[40..48])
    );
    let _ = place_val!(
        ox + 39,
        oy + 2,
        TileType::Const,
        pack_8(&byte2_bytes[32..40])
    );
    let bank2_byte2_mux_idx = place!(ox + 40, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank2_byte2_mux_idx);
    let idx = place!(ox + 41, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 43,
        oy + 1,
        TileType::Const,
        pack_8(&byte3_bytes[40..48])
    );
    let _ = place_val!(
        ox + 42,
        oy + 2,
        TileType::Const,
        pack_8(&byte3_bytes[32..40])
    );
    let bank2_byte3_mux_idx = place!(ox + 43, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank2_byte3_mux_idx);
    let idx = place!(ox + 44, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    // Sprint 133: Bank 3 (instructions 48-63) — 4 lanes.
    let _ = place_val!(ox + 46, oy + 1, TileType::Const, pack_8(&low_bytes[56..64]));
    let _ = place_val!(ox + 45, oy + 2, TileType::Const, pack_8(&low_bytes[48..56]));
    let bank3_low_mux_idx = place!(ox + 46, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank3_low_mux_idx);
    let idx = place!(ox + 47, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 49,
        oy + 1,
        TileType::Const,
        pack_8(&high_bytes[56..64])
    );
    let _ = place_val!(
        ox + 48,
        oy + 2,
        TileType::Const,
        pack_8(&high_bytes[48..56])
    );
    let bank3_high_mux_idx = place!(ox + 49, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank3_high_mux_idx);
    let idx = place!(ox + 50, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 52,
        oy + 1,
        TileType::Const,
        pack_8(&byte2_bytes[56..64])
    );
    let _ = place_val!(
        ox + 51,
        oy + 2,
        TileType::Const,
        pack_8(&byte2_bytes[48..56])
    );
    let bank3_byte2_mux_idx = place!(ox + 52, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank3_byte2_mux_idx);
    let idx = place!(ox + 53, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 55,
        oy + 1,
        TileType::Const,
        pack_8(&byte3_bytes[56..64])
    );
    let _ = place_val!(
        ox + 54,
        oy + 2,
        TileType::Const,
        pack_8(&byte3_bytes[48..56])
    );
    let bank3_byte3_mux_idx = place!(ox + 55, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank3_byte3_mux_idx);
    let idx = place!(ox + 56, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    // -------------------------------------------------------------------------
    // Sprint 187: Banks 4-7 (instructions 64-127) — 4 lanes each
    // -------------------------------------------------------------------------
    // Placed east of selector tree. Each bank follows Banks 1-3 pattern:
    //   Const(upper_8) at (mux_x, oy+1), Const(lower_8) at (mux_x-1, oy+2),
    //   Mux16to1 at (mux_x, oy+2), WireUp at (mux_x+1, oy+2) for PC tap.
    // Bank layout: lane spacing = 3, bank spacing = 12 (4 lanes × 3).
    //   Bank4: ox+80,83,86,89   Bank5: ox+92,95,98,101
    //   Bank6: ox+104,107,110,113   Bank7: ox+116,119,122,125

    // Bank 4 (instructions 64-79):
    let _ = place_val!(ox + 80, oy + 1, TileType::Const, pack_8(&low_bytes[72..80]));
    let _ = place_val!(ox + 79, oy + 2, TileType::Const, pack_8(&low_bytes[64..72]));
    let bank4_low_mux_idx = place!(ox + 80, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank4_low_mux_idx);
    let idx = place!(ox + 81, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 83,
        oy + 1,
        TileType::Const,
        pack_8(&high_bytes[72..80])
    );
    let _ = place_val!(
        ox + 82,
        oy + 2,
        TileType::Const,
        pack_8(&high_bytes[64..72])
    );
    let bank4_high_mux_idx = place!(ox + 83, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank4_high_mux_idx);
    let idx = place!(ox + 84, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 86,
        oy + 1,
        TileType::Const,
        pack_8(&byte2_bytes[72..80])
    );
    let _ = place_val!(
        ox + 85,
        oy + 2,
        TileType::Const,
        pack_8(&byte2_bytes[64..72])
    );
    let bank4_byte2_mux_idx = place!(ox + 86, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank4_byte2_mux_idx);
    let idx = place!(ox + 87, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 89,
        oy + 1,
        TileType::Const,
        pack_8(&byte3_bytes[72..80])
    );
    let _ = place_val!(
        ox + 88,
        oy + 2,
        TileType::Const,
        pack_8(&byte3_bytes[64..72])
    );
    let bank4_byte3_mux_idx = place!(ox + 89, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank4_byte3_mux_idx);
    let idx = place!(ox + 90, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    // Bank 5 (instructions 80-95):
    let _ = place_val!(ox + 92, oy + 1, TileType::Const, pack_8(&low_bytes[88..96]));
    let _ = place_val!(ox + 91, oy + 2, TileType::Const, pack_8(&low_bytes[80..88]));
    let bank5_low_mux_idx = place!(ox + 92, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank5_low_mux_idx);
    let idx = place!(ox + 93, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 95,
        oy + 1,
        TileType::Const,
        pack_8(&high_bytes[88..96])
    );
    let _ = place_val!(
        ox + 94,
        oy + 2,
        TileType::Const,
        pack_8(&high_bytes[80..88])
    );
    let bank5_high_mux_idx = place!(ox + 95, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank5_high_mux_idx);
    let idx = place!(ox + 96, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 98,
        oy + 1,
        TileType::Const,
        pack_8(&byte2_bytes[88..96])
    );
    let _ = place_val!(
        ox + 97,
        oy + 2,
        TileType::Const,
        pack_8(&byte2_bytes[80..88])
    );
    let bank5_byte2_mux_idx = place!(ox + 98, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank5_byte2_mux_idx);
    let idx = place!(ox + 99, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 101,
        oy + 1,
        TileType::Const,
        pack_8(&byte3_bytes[88..96])
    );
    let _ = place_val!(
        ox + 100,
        oy + 2,
        TileType::Const,
        pack_8(&byte3_bytes[80..88])
    );
    let bank5_byte3_mux_idx = place!(ox + 101, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank5_byte3_mux_idx);
    let idx = place!(ox + 102, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    // Bank 6 (instructions 96-111):
    let _ = place_val!(
        ox + 104,
        oy + 1,
        TileType::Const,
        pack_8(&low_bytes[104..112])
    );
    let _ = place_val!(
        ox + 103,
        oy + 2,
        TileType::Const,
        pack_8(&low_bytes[96..104])
    );
    let bank6_low_mux_idx = place!(ox + 104, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank6_low_mux_idx);
    let idx = place!(ox + 105, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 107,
        oy + 1,
        TileType::Const,
        pack_8(&high_bytes[104..112])
    );
    let _ = place_val!(
        ox + 106,
        oy + 2,
        TileType::Const,
        pack_8(&high_bytes[96..104])
    );
    let bank6_high_mux_idx = place!(ox + 107, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank6_high_mux_idx);
    let idx = place!(ox + 108, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 110,
        oy + 1,
        TileType::Const,
        pack_8(&byte2_bytes[104..112])
    );
    let _ = place_val!(
        ox + 109,
        oy + 2,
        TileType::Const,
        pack_8(&byte2_bytes[96..104])
    );
    let bank6_byte2_mux_idx = place!(ox + 110, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank6_byte2_mux_idx);
    let idx = place!(ox + 111, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 113,
        oy + 1,
        TileType::Const,
        pack_8(&byte3_bytes[104..112])
    );
    let _ = place_val!(
        ox + 112,
        oy + 2,
        TileType::Const,
        pack_8(&byte3_bytes[96..104])
    );
    let bank6_byte3_mux_idx = place!(ox + 113, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank6_byte3_mux_idx);
    let idx = place!(ox + 114, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    // Bank 7 (instructions 112-127):
    let _ = place_val!(
        ox + 116,
        oy + 1,
        TileType::Const,
        pack_8(&low_bytes[120..128])
    );
    let _ = place_val!(
        ox + 115,
        oy + 2,
        TileType::Const,
        pack_8(&low_bytes[112..120])
    );
    let bank7_low_mux_idx = place!(ox + 116, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank7_low_mux_idx);
    let idx = place!(ox + 117, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 119,
        oy + 1,
        TileType::Const,
        pack_8(&high_bytes[120..128])
    );
    let _ = place_val!(
        ox + 118,
        oy + 2,
        TileType::Const,
        pack_8(&high_bytes[112..120])
    );
    let bank7_high_mux_idx = place!(ox + 119, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank7_high_mux_idx);
    let idx = place!(ox + 120, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 122,
        oy + 1,
        TileType::Const,
        pack_8(&byte2_bytes[120..128])
    );
    let _ = place_val!(
        ox + 121,
        oy + 2,
        TileType::Const,
        pack_8(&byte2_bytes[112..120])
    );
    let bank7_byte2_mux_idx = place!(ox + 122, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank7_byte2_mux_idx);
    let idx = place!(ox + 123, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    let _ = place_val!(
        ox + 125,
        oy + 1,
        TileType::Const,
        pack_8(&byte3_bytes[120..128])
    );
    let _ = place_val!(
        ox + 124,
        oy + 2,
        TileType::Const,
        pack_8(&byte3_bytes[112..120])
    );
    let bank7_byte3_mux_idx = place!(ox + 125, oy + 2, TileType::Mux16to1);
    push_pipeline!(bank7_byte3_mux_idx);
    let idx = place!(ox + 126, oy + 2, TileType::WireUp);
    push_pipeline!(idx);

    // Sprint 187: bank47_mux_indices — [bank_within_group][lane]
    // lane order: low(0), high(1), byte2(2), byte3(3)
    let bank47_mux_indices: [[usize; 4]; 4] = [
        [
            bank4_low_mux_idx,
            bank4_high_mux_idx,
            bank4_byte2_mux_idx,
            bank4_byte3_mux_idx,
        ],
        [
            bank5_low_mux_idx,
            bank5_high_mux_idx,
            bank5_byte2_mux_idx,
            bank5_byte3_mux_idx,
        ],
        [
            bank6_low_mux_idx,
            bank6_high_mux_idx,
            bank6_byte2_mux_idx,
            bank6_byte3_mux_idx,
        ],
        [
            bank7_low_mux_idx,
            bank7_high_mux_idx,
            bank7_byte2_mux_idx,
            bank7_byte3_mux_idx,
        ],
    ];

    // -------------------------------------------------------------------------
    // Sprint 133: L1+L2 ViaDown taps — lift each bank output to L2
    // -------------------------------------------------------------------------
    // Sprint 109 L2 east rerouted to oy+1, freeing L2 oy+2 for bank taps.
    // Three cases:
    //  (a) Bank0 Lane0 (ox+1): L1(ox+1,oy+2) reserved (Sprint 125 WireUp).
    //      Tap from WireRight chain at ox+9 instead (carries same Mux16to1 output).
    //  (b) Bank2 Lane1 (ox+37): L2(ox+37,oy+2) is Sprint 109 south WireDown.
    //      L1 detour: ViaDown(ox+37) → WireLeft(ox+36) → L2 ViaDown(ox+36).
    //  (c) All others: direct L1+L2 ViaDown at Mux16to1 position.

    // (a) Bank0 Lane0: canonical tap from WireRight chain at ox+9.
    // Keep an auxiliary tap at ox+1 for local ir_low fanout routes.
    let _b0l0_l1_tap = place_l1_route!(ox + 9, oy + 2, TileType::ViaDown);
    let _b0l0_l2_tap = place_l2_route!(ox + 9, oy + 2, TileType::ViaDown);
    let _b0l0_l2_aux_tap = place_l2_route!(ox + 1, oy + 2, TileType::ViaDown);
    push_pipeline!(_b0l0_l1_tap);
    push_pipeline!(_b0l0_l2_tap);
    push_pipeline!(_b0l0_l2_aux_tap);

    // Bank0 Lane1 (ox+13): L1 ViaDown already exists (ir_high tap, line ~772).
    // Only add L2 ViaDown (freed by Sprint 109 reroute).
    let _b0l1_l2_tap = place_l2_route!(ox + 13, oy + 2, TileType::ViaDown);
    push_pipeline!(_b0l1_l2_tap);

    // (c) Direct taps for Bank0 Lane2-3, Bank1 all, Bank2 Lane0/2/3, Bank3 all.
    let direct_tap_positions: &[usize] = &[
        ox + 16,
        ox + 19, // Bank0 Lane2, Lane3
        ox + 22,
        ox + 25,
        ox + 28,
        ox + 31, // Bank1 Lanes 0-3
        ox + 34, // Bank2 Lane0
        ox + 40,
        ox + 43, // Bank2 Lane2, Lane3
        ox + 46,
        ox + 49,
        ox + 52, // Bank3 Lanes 0-2
    ];
    for &bx in direct_tap_positions {
        let l1_tap = place_l1_route!(bx, oy + 2, TileType::ViaDown);
        let l2_tap = place_l2_route!(bx, oy + 2, TileType::ViaDown);
        push_pipeline!(l1_tap);
        push_pipeline!(l2_tap);
    }

    // (b) Bank2 Lane1 (ox+37): L1 detour via ox+36.
    let _b2l1_l1_tap = place_l1_route!(ox + 37, oy + 2, TileType::ViaDown);
    let _b2l1_l1_wl = place_l1_route!(ox + 36, oy + 2, TileType::WireLeft);
    let _b2l1_l2_tap = place_l2_route!(ox + 36, oy + 2, TileType::ViaDown);
    push_pipeline!(_b2l1_l1_tap);
    push_pipeline!(_b2l1_l1_wl);
    push_pipeline!(_b2l1_l2_tap);

    // -------------------------------------------------------------------------
    // Sprint 133: PC[4] and PC[5] extraction via BitSelect
    // -------------------------------------------------------------------------
    // PC[4] extraction: tap oy+3 WireRight at ox+25, route south, BitSelect bit 4.
    let _pc4_wd1 = place!(ox + 25, oy + 4, TileType::WireDown);
    let _pc4_wd2 = place!(ox + 25, oy + 5, TileType::WireDown);
    let pc4_bit_idx = place!(ox + 26, oy + 5, TileType::BitSelect);
    let _pc4_const = place_val!(ox + 27, oy + 5, TileType::Const, 4);
    push_pipeline!(_pc4_wd1);
    push_pipeline!(_pc4_wd2);
    push_pipeline!(pc4_bit_idx);

    // PC[5] extraction: tap oy+3 WireRight at ox+28, route south, BitSelect bit 5.
    let _pc5_wd1 = place!(ox + 28, oy + 4, TileType::WireDown);
    let _pc5_wd2 = place!(ox + 28, oy + 5, TileType::WireDown);
    let pc5_bit_idx = place!(ox + 29, oy + 5, TileType::BitSelect);
    let _pc5_const = place_val!(ox + 30, oy + 5, TileType::Const, 5);
    push_pipeline!(_pc5_wd1);
    push_pipeline!(_pc5_wd2);
    push_pipeline!(pc5_bit_idx);

    // -------------------------------------------------------------------------
    // Sprint 133: Bank selector mux tree — 4 lanes x 3 muxes (L1A/L1B/Final)
    // -------------------------------------------------------------------------
    // Sprint 144 Phase A preflight anchors (ox=0, oy=0):
    // Lane0/low sources on L2: B0=(9,2), B1=(22,2), B2=(34,2), B3=(46,2)
    // Lane1/high sources on L2: B0=(13,2), B1=(25,2), B2=(36,2), B3=(49,2)
    // Lane0 selector inputs on L0: L1A.L=(56,7) L1A.R=(58,7) L1B.L=(59,7) L1B.R=(61,7)
    // Lane1 selector inputs on L0: L1A.L=(62,7) L1A.R=(64,7) L1B.L=(65,7) L1B.R=(67,7)
    // Selector bits: PC[4] roots at (26,5), PC[5] roots at (29,5)
    // Low/high final roots: Lane0=(58,9), Lane1=(64,9)
    // Planned ingress targets: ir_low=(1,3), ir_high=(13,4)
    //
    // Route feasibility is guarded by:
    //   tile_cpu::v2_tests::test_v2_s144_route_preflight_low_high_candidate_nets
    // which checks static L2/L3 reachability for all low/high selector nets.
    // Layout per lane at base_x, oy+7..9:
    //   (base_x, 7)=L1A Mux, (base_x+3, 7)=L1B Mux
    //   (base_x, 8)=WireDown(L1A), (base_x+3, 8)=WireDown(L1B)
    //   (base_x, 9)=WireDown(L1A→Final LEFT), (base_x+1, 9)=Final Mux,
    //   (base_x+2, 9)=WireLeft(L1B→Final RIGHT), (base_x+3, 9)=WireDown(L1B)
    // Mux semantics: UP!=0 → LEFT, else RIGHT.
    // Sprint 150 corrected mapping:
    //   L1A: PC[4]!=0→LEFT(Bank3), PC[4]=0→RIGHT(Bank2) — selected when PC[5]!=0
    //   L1B: PC[4]!=0→LEFT(Bank1), PC[4]=0→RIGHT(Bank0) — selected when PC[5]=0
    //   Final: PC[5]!=0→LEFT(L1A=banks 2/3), PC[5]=0→RIGHT(L1B=banks 0/1 via WireLeft)
    let lane_bases: [usize; 4] = [ox + 57, ox + 63, ox + 69, ox + 75];
    let mut final_mux_indices = [0usize; 4];
    let mut l1a_mux_indices = [0usize; 4];
    let mut l1b_mux_indices = [0usize; 4];

    for (lane, &bx) in lane_bases.iter().enumerate() {
        let l1a = place!(bx, oy + 7, TileType::Mux);
        let l1b = place!(bx + 3, oy + 7, TileType::Mux);
        let l1a_wd1 = place!(bx, oy + 8, TileType::WireDown);
        let l1b_wd1 = place!(bx + 3, oy + 8, TileType::WireDown);
        let l1a_wd2 = place!(bx, oy + 9, TileType::WireDown);
        let l1b_wd2 = place!(bx + 3, oy + 9, TileType::WireDown);
        let l1b_wl = place!(bx + 2, oy + 9, TileType::WireLeft);
        let final_mux = place!(bx + 1, oy + 9, TileType::Mux);

        push_pipeline!(l1a);
        push_pipeline!(l1b);
        push_pipeline!(l1a_wd1);
        push_pipeline!(l1b_wd1);
        push_pipeline!(l1a_wd2);
        push_pipeline!(l1b_wd2);
        push_pipeline!(l1b_wl);
        push_pipeline!(final_mux);

        l1a_mux_indices[lane] = l1a;
        l1b_mux_indices[lane] = l1b;
        final_mux_indices[lane] = final_mux;
    }

    // -------------------------------------------------------------------------
    // Sprint 150: Physical ROM bank selector input delivery (~383 tiles)
    // -------------------------------------------------------------------------
    // Connects bank L2 taps → selector mux L/R inputs (8 bank data routes),
    // PC[4] → L1A/L1B UP inputs (4 routes), PC[5] → Final UP inputs (2 routes).
    // Policy: L3 horizontal only, L2 vertical only (with short doglegs where blocked).

    // --- Phase 2A: Extend PC bus east on L0 y=3 for local BitSelect extraction ---
    // PC bus currently ends at ox+56. Extend to ox+65 so selector-area BitSelects
    // can tap PC value locally (avoids routing PC[4]/PC[5] signal 30+ columns east).
    for x in (ox + 57)..=(ox + 65) {
        let idx = place!(x, oy + 3, TileType::WireRight);
        push_pipeline!(idx);
    }
    // Sprint 185: extend PC bus east for lanes 2/3 byte2/byte3 selector delivery
    for x in (ox + 66)..=(ox + 78) {
        let idx = place!(x, oy + 3, TileType::WireRight);
        push_pipeline!(idx);
    }
    // Sprint 187: extend PC bus east to reach Banks 4-7 selectors
    for x in (ox + 79)..=(ox + 126) {
        let idx = place!(x, oy + 3, TileType::WireRight);
        push_pipeline!(idx);
    }

    // --- Phase 2B: PC[4] delivery to L1A/L1B UP inputs (4 independent BitSelects) ---
    // Each L1A/L1B Mux at oy+7 needs PC[4] on its UP input at oy+6.
    // Strategy: tap PC bus south at 4 columns, BitSelect bit 4, WireDown to oy+6.
    // Source columns: x=56,59,62,65 (one left of each lane_base).
    // BitSelect at (src+1, oy+5), Const(4) at (src+2, oy+5), WireDown at (src+1, oy+6).
    let pc4_source_cols: [(usize, usize); 4] = [
        (ox + 56, ox + 57), // → L1A lane0 UP at (57, oy+6)
        (ox + 59, ox + 60), // → L1A lane1 UP at (60, oy+6)
        (ox + 62, ox + 63), // → L1A lane2 UP at (63, oy+6)
        (ox + 65, ox + 66), // → L1A lane3 UP at (66, oy+6)
    ];
    for &(src_x, bs_x) in &pc4_source_cols {
        // South route from PC bus at oy+3 to oy+5
        let wd1 = place!(src_x, oy + 4, TileType::WireDown);
        let wd2 = place!(src_x, oy + 5, TileType::WireDown);
        push_pipeline!(wd1);
        push_pipeline!(wd2);
        // BitSelect(bit=4) reads LEFT=PC value, RIGHT=Const(4)
        let bs = place!(bs_x, oy + 5, TileType::BitSelect);
        let _ = place_val!(bs_x + 1, oy + 5, TileType::Const, 4);
        push_pipeline!(bs);
        // WireDown carries PC[4] to mux UP input position at oy+6
        let wd3 = place!(bs_x, oy + 6, TileType::WireDown);
        push_pipeline!(wd3);
    }

    // Sprint 185: PC[4] delivery to lanes 2/3 L1A/L1B UP inputs (4 more entries)
    let pc4_lanes23_cols: [(usize, usize); 4] = [
        (ox + 68, ox + 69), // → L1A lane2 UP at (69, oy+6)
        (ox + 71, ox + 72), // → L1B lane2 UP at (72, oy+6)
        (ox + 74, ox + 75), // → L1A lane3 UP at (75, oy+6)
        (ox + 77, ox + 78), // → L1B lane3 UP at (78, oy+6)
    ];
    for &(src_x, bs_x) in &pc4_lanes23_cols {
        let wd1 = place!(src_x, oy + 4, TileType::WireDown);
        let wd2 = place!(src_x, oy + 5, TileType::WireDown);
        push_pipeline!(wd1);
        push_pipeline!(wd2);
        let bs = place!(bs_x, oy + 5, TileType::BitSelect);
        let _ = place_val!(bs_x + 1, oy + 5, TileType::Const, 4);
        push_pipeline!(bs);
        let wd3 = place!(bs_x, oy + 6, TileType::WireDown);
        push_pipeline!(wd3);
    }

    // --- Phase 2C: PC[5] delivery to Final mux UP inputs (2 routes) ---
    // Final muxes at (bx+1, oy+9) need PC[5] on UP at (bx+1, oy+8).
    // Lane0 final at (58, oy+9), lane1 final at (64, oy+9).
    // Strategy: BitSelect at L0(57, oy+4), L1 trunk south+east to (58,8) and (64,8).
    //
    // PC source: tap oy+3 WireRight at ox+56, south route to (56, oy+4).
    let _pc5_src_wd = place!(ox + 56, oy + 4, TileType::WireDown);
    push_pipeline!(_pc5_src_wd);
    // BitSelect(bit=5) at (57, oy+4): LEFT = PC from WireDown, RIGHT = Const(5)
    let _pc5_bs = place!(ox + 57, oy + 4, TileType::BitSelect);
    let _ = place_val!(ox + 58, oy + 4, TileType::Const, 5);
    push_pipeline!(_pc5_bs);
    // Lift to L1 at (57, oy+4)
    let _pc5_l1_vd = place_l1_route!(ox + 57, oy + 4, TileType::ViaDown);
    push_pipeline!(_pc5_l1_vd);
    // L1 south descent from (57, oy+5) to (57, oy+8)
    for y_off in 5..=8 {
        let idx = place_l1_route!(ox + 57, oy + y_off, TileType::WireDown);
        push_pipeline!(idx);
    }
    // L1 east trunk at oy+8 from x=58 to x=64 (bypasses CtrlB on L2)
    for x in (ox + 58)..=(ox + 64) {
        let idx = place_l1_route!(x, oy + 8, TileType::WireRight);
        push_pipeline!(idx);
    }
    // ViaUp at L0(58, oy+8) — Final mux lane0 UP input
    let _pc5_vu_lane0 = place!(ox + 58, oy + 8, TileType::ViaUp);
    push_pipeline!(_pc5_vu_lane0);
    // ViaUp at L0(64, oy+8) — Final mux lane1 UP input
    let _pc5_vu_lane1 = place!(ox + 64, oy + 8, TileType::ViaUp);
    push_pipeline!(_pc5_vu_lane1);
    // Sprint 185: extend L1 PC[5] trunk east for lanes 2/3 Final mux
    for x in (ox + 65)..=(ox + 76) {
        let idx = place_l1_route!(x, oy + 8, TileType::WireRight);
        push_pipeline!(idx);
    }
    // ViaUp at L0(70, oy+8) — Final mux lane2 UP input
    let _pc5_vu_lane2 = place!(ox + 70, oy + 8, TileType::ViaUp);
    push_pipeline!(_pc5_vu_lane2);
    // ViaUp at L0(76, oy+8) — Final mux lane3 UP input
    let _pc5_vu_lane3 = place!(ox + 76, oy + 8, TileType::ViaUp);
    push_pipeline!(_pc5_vu_lane3);

    // --- Phase 2D: Bank data delivery (8 routes from L2 taps to selector mux L/R) ---
    // Each route: L2 tap (already placed) → L2 vertical → ViaDown@L3 → L3 horizontal
    // → ViaUp@L2 → L2 vertical descent → ViaUp@L1 → ViaUp@L0 at mux input.
    //
    // CORRECTED mux mapping (Mux: UP!=0 → LEFT, UP=0 → RIGHT):
    //   Final: PC[5]!=0 → LEFT (L1A) → banks 2/3;  PC[5]=0 → RIGHT (L1B) → banks 0/1
    //   L1A: PC[4]!=0 → LEFT (Bank3), PC[4]=0 → RIGHT (Bank2)
    //   L1B: PC[4]!=0 → LEFT (Bank1), PC[4]=0 → RIGHT (Bank0)
    //
    // Lane0 (low byte):  L1A@(57,7) L/R = B3/B2;  L1B@(60,7) L/R = B1/B0
    // Lane1 (high byte): L1A@(63,7) L/R = B3/B2;  L1B@(66,7) L/R = B1/B0

    // Route B0.low: source from live Bank0 low tap at L2(1,2) → (61,7) L1B.R.
    // L2-only dogleg avoids congested selector corridors while keeping legal
    // directional turns (cornering happens across adjacent cells, not in-cell).
    {
        // Step 1: source hop to x=2, then south to y=6.
        let hop = place_l2_route!(ox + 2, oy + 2, TileType::WireRight);
        push_pipeline!(hop);
        for y_off in 3..=6 {
            let idx = place_l2_route!(ox + 2, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        // Sprint 151: Override L2(2,4) from WireDown to ViaUp for westback
        // ir_low delivery. ViaUp reads L3(2,4) = physical westback signal.
        sim.set_tile_3d(ox + 2, oy + 4, 2, TileType::ViaUp);
        // Sprint 151: Override L2(2,6) from WireDown to WireRight.
        // B0.low delivery re-routed through x=0 bypass column.
        sim.set_tile_3d(ox + 2, oy + 6, 2, TileType::WireRight);

        // Sprint 151: B0.low bypass column via L2 x=0.
        // Original path through L2(2,3..6) is broken at L2(2,4) ViaUp.
        // New path: L2(1,2) → L2(0,2) WireLeft → L2(0,3..6) WireDown
        //         → L2(1,6) WireRight → L2(2,6) WireRight → existing east trunk.
        let _s151_b0_bypass_entry = place_l2_route!(ox + 0, oy + 2, TileType::WireLeft);
        push_pipeline!(_s151_b0_bypass_entry);
        for y_off in 3..=6 {
            let idx = place_l2_route!(ox + 0, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        let _s151_b0_bypass_exit = place_l2_route!(ox + 1, oy + 6, TileType::WireRight);
        push_pipeline!(_s151_b0_bypass_exit);

        // Step 2: east to x=10 at y=6.
        for x in (ox + 3)..=(ox + 10) {
            let idx = place_l2_route!(x, oy + 6, TileType::WireRight);
            push_pipeline!(idx);
        }
        // Step 3: north to row y=0 at x=10.
        for y_off in (0..=5).rev() {
            let idx = place_l2_route!(ox + 10, oy + y_off, TileType::WireUp);
            push_pipeline!(idx);
        }
        // Step 4: east trunk to x=61 on y=0.
        for x in (ox + 11)..=(ox + 61) {
            let idx = place_l2_route!(x, oy, TileType::WireRight);
            push_pipeline!(idx);
        }
        // Step 5: descend to sink row y=7 at x=61.
        for y_off in 1..=7 {
            let idx = place_l2_route!(ox + 61, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        // Step 6: lift to selector mux input on L0.
        let l1_vu = place_l1_route!(ox + 61, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        // Sprint 184: first live computational via — WeightedViaUp identity
        let l0_vu = place!(ox + 61, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B1.low: (22,2) → (59,7) L1B.L [L3 y=1, x=22..59]
    {
        // L2 tap at (22,2) already placed. ViaDown@L3(22,2) reads L2(22,2).
        let l3_entry = place_l3_route!(ox + 22, oy + 2, TileType::ViaDown);
        push_pipeline!(l3_entry);
        // L3 north to y=1: WireUp at (22,1)
        let l3_wu = place_l3_route!(ox + 22, oy + 1, TileType::WireUp);
        push_pipeline!(l3_wu);
        // L3 east x=23..59 at y=1
        for x in (ox + 23)..=(ox + 59) {
            let idx = place_l3_route!(x, oy + 1, TileType::WireRight);
            push_pipeline!(idx);
        }
        // ViaUp@L2(59,1) reads L3(59,1)
        let l2_exit = place_l2_route!(ox + 59, oy + 1, TileType::ViaUp);
        push_pipeline!(l2_exit);
        // L2 south y=2..7
        for y_off in 2..=7 {
            let idx = place_l2_route!(ox + 59, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        // ViaUp@L1(59,7) reads L2(59,7)
        let l1_vu = place_l1_route!(ox + 59, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        // ViaUp@L0(59,7) reads L1(59,7) — L1B.L input
        let l0_vu = place!(ox + 59, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B3.high: (49,2) → (62,7) L1A.L [L3 y=2, x=50..62]
    {
        let l3_entry = place_l3_route!(ox + 49, oy + 2, TileType::ViaDown);
        push_pipeline!(l3_entry);
        // L3 east x=50..62 at y=2
        for x in (ox + 50)..=(ox + 62) {
            let idx = place_l3_route!(x, oy + 2, TileType::WireRight);
            push_pipeline!(idx);
        }
        // ViaUp@L2(62,2) reads L3(62,2)
        let l2_exit = place_l2_route!(ox + 62, oy + 2, TileType::ViaUp);
        push_pipeline!(l2_exit);
        // L2 south y=3..7
        for y_off in 3..=7 {
            let idx = place_l2_route!(ox + 62, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        let l1_vu = place_l1_route!(ox + 62, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        let l0_vu = place!(ox + 62, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B0.high: (13,2) → (67,7) L1B.R [L3 y=3, x=14..67]
    {
        let l3_entry = place_l3_route!(ox + 13, oy + 2, TileType::ViaDown);
        push_pipeline!(l3_entry);
        // L3 south to y=3: WireDown at (13,3)
        let l3_wd = place_l3_route!(ox + 13, oy + 3, TileType::WireDown);
        push_pipeline!(l3_wd);
        // L3 east x=14..67 at y=3
        for x in (ox + 14)..=(ox + 67) {
            let idx = place_l3_route!(x, oy + 3, TileType::WireRight);
            push_pipeline!(idx);
        }
        // ViaUp@L2(67,3) reads L3(67,3)
        let l2_exit = place_l2_route!(ox + 67, oy + 3, TileType::ViaUp);
        push_pipeline!(l2_exit);
        // L2 south y=4..7
        for y_off in 4..=7 {
            let idx = place_l2_route!(ox + 67, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        let l1_vu = place_l1_route!(ox + 67, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        let l0_vu = place!(ox + 67, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B1.high: (25,2) → (65,7) L1B.L [L3 y=4, x=26..65]
    {
        // L2 south from (25,2) to (25,4)
        let l2_wd3 = place_l2_route!(ox + 25, oy + 3, TileType::WireDown);
        let l2_wd4 = place_l2_route!(ox + 25, oy + 4, TileType::WireDown);
        push_pipeline!(l2_wd3);
        push_pipeline!(l2_wd4);
        // ViaDown@L3(25,4) reads L2(25,4)
        let l3_entry = place_l3_route!(ox + 25, oy + 4, TileType::ViaDown);
        push_pipeline!(l3_entry);
        // L3 east x=26..65 at y=4
        for x in (ox + 26)..=(ox + 65) {
            let idx = place_l3_route!(x, oy + 4, TileType::WireRight);
            push_pipeline!(idx);
        }
        // ViaUp@L2(65,4) reads L3(65,4)
        let l2_exit = place_l2_route!(ox + 65, oy + 4, TileType::ViaUp);
        push_pipeline!(l2_exit);
        // L2 south y=5..7
        for y_off in 5..=7 {
            let idx = place_l2_route!(ox + 65, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        let l1_vu = place_l1_route!(ox + 65, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        let l0_vu = place!(ox + 65, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B2.low: (34,2) → (58,7) L1A.R [L3 y=5, x=35..58]
    {
        // L2 south from (34,2) to (34,5)
        for y_off in 3..=5 {
            let idx = place_l2_route!(ox + 34, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        let l3_entry = place_l3_route!(ox + 34, oy + 5, TileType::ViaDown);
        push_pipeline!(l3_entry);
        // L3 east x=35..58 at y=5
        for x in (ox + 35)..=(ox + 58) {
            let idx = place_l3_route!(x, oy + 5, TileType::WireRight);
            push_pipeline!(idx);
        }
        let l2_exit = place_l2_route!(ox + 58, oy + 5, TileType::ViaUp);
        push_pipeline!(l2_exit);
        // L2 south y=6..7
        for y_off in 6..=7 {
            let idx = place_l2_route!(ox + 58, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        let l1_vu = place_l1_route!(ox + 58, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        let l0_vu = place!(ox + 58, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B2.high: (36,2) → (64,7) L1A.R [L3 y=6, x=37..64]
    // Note: B2.high L2 tap is at (36,2) via L1 detour — actual L2 ViaDown is at (36,2).
    {
        // L2 south from (36,2) to (36,6)
        for y_off in 3..=6 {
            let idx = place_l2_route!(ox + 36, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        let l3_entry = place_l3_route!(ox + 36, oy + 6, TileType::ViaDown);
        push_pipeline!(l3_entry);
        // L3 east x=37..64 at y=6
        for x in (ox + 37)..=(ox + 64) {
            let idx = place_l3_route!(x, oy + 6, TileType::WireRight);
            push_pipeline!(idx);
        }
        let l2_exit = place_l2_route!(ox + 64, oy + 6, TileType::ViaUp);
        push_pipeline!(l2_exit);
        // L2 south y=7
        let l2_wd7 = place_l2_route!(ox + 64, oy + 7, TileType::WireDown);
        push_pipeline!(l2_wd7);
        let l1_vu = place_l1_route!(ox + 64, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        let l0_vu = place!(ox + 64, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B3.low: (46,2) → (56,7) L1A.L [L2 south + L2 east y=7 + L3 y=7 x=55..56]
    // Special: uses L2 east at y=7 from x=47..55, then L3 at x=55..56 (avoids Sprint 125
    // L3 y=7 WireLeft at x=9..53 and WireUp at x=54).
    {
        // L2 south from (46,2) to (46,7)
        for y_off in 3..=7 {
            let idx = place_l2_route!(ox + 46, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        // L2 east at y=7 from x=47 to x=55
        for x in (ox + 47)..=(ox + 55) {
            let idx = place_l2_route!(x, oy + 7, TileType::WireRight);
            push_pipeline!(idx);
        }
        // ViaDown@L3(55,7) reads L2(55,7)
        let l3_entry = place_l3_route!(ox + 55, oy + 7, TileType::ViaDown);
        push_pipeline!(l3_entry);
        // L3 east x=56 at y=7
        let l3_wr = place_l3_route!(ox + 56, oy + 7, TileType::WireRight);
        push_pipeline!(l3_wr);
        // ViaUp@L2(56,7) reads L3(56,7)
        let l2_exit = place_l2_route!(ox + 56, oy + 7, TileType::ViaUp);
        push_pipeline!(l2_exit);
        // ViaUp@L1(56,7) reads L2(56,7)
        let l1_vu = place_l1_route!(ox + 56, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        // ViaUp@L0(56,7) reads L1(56,7) — L1A.L input
        let l0_vu = place!(ox + 56, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // -------------------------------------------------------------------------
    // Sprint 185: Byte2/Byte3 bank data delivery (8 routes to lanes 2/3)
    // -------------------------------------------------------------------------
    // Lanes 2/3 of the 4-lane mux scaffold (lane_bases 69, 75) were dead code since
    // Sprint 133. These routes deliver byte2/byte3 bank outputs to selector mux L/R
    // inputs, completing the physical 32-bit fetch fabric.
    //
    // Strategy: L2-only paths (L3 blocked at x=68 by Sprint 151 ir_high WireUp column).
    //   L2 tap → L2 south descent → L2 east trunk → L2 north ascent → L1 ViaUp → L0 WeightedViaUp.
    // Each route uses a separate L2 y-row to avoid signal leakage through default Wire tiles.
    //
    // Mux input mapping (same as lanes 0/1):
    //   L1A.LEFT = Bank3, L1A.RIGHT = Bank2, L1B.LEFT = Bank1, L1B.RIGHT = Bank0.
    //
    // Lane 2 (byte2): L1A@(69,7), L1B@(72,7), Final@(70,9)
    // Lane 3 (byte3): L1A@(75,7), L1B@(78,7), Final@(76,9)

    // Route B3.byte2: (52,2) → (68,7) L1A.LEFT [L2 y=6 → y=8 pocket]
    // Avoid the congested top rows entirely: descend on the existing L2 tap, use y=6
    // until x=57, duck to y=8 under the blocked top-right descent columns, then return
    // north at the sink.
    {
        for y_off in 3..=6 {
            let idx = place_l2_route!(ox + 52, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        for x in (ox + 53)..=(ox + 56) {
            let idx = place_l2_route!(x, oy + 6, TileType::WireRight);
            push_pipeline!(idx);
        }
        let l2_pocket_turn = place_l2_route!(ox + 57, oy + 6, TileType::WireRight);
        let l2_pocket_down1 = place_l2_route!(ox + 57, oy + 7, TileType::WireDown);
        let l3_pocket_entry = place_l3_route!(ox + 57, oy + 7, TileType::ViaDown);
        push_pipeline!(l2_pocket_turn);
        push_pipeline!(l2_pocket_down1);
        push_pipeline!(l3_pocket_entry);
        let l3_pocket_down = place_l3_route!(ox + 57, oy + 8, TileType::WireDown);
        push_pipeline!(l3_pocket_down);
        for x in (ox + 58)..=(ox + 62) {
            let idx = place_l3_route!(x, oy + 8, TileType::WireRight);
            push_pipeline!(idx);
        }
        let l2_pocket_resume = place_l2_route!(ox + 62, oy + 8, TileType::ViaUp);
        push_pipeline!(l2_pocket_resume);
        for x in (ox + 63)..=(ox + 68) {
            let idx = place_l2_route!(x, oy + 8, TileType::WireRight);
            push_pipeline!(idx);
        }
        let l2_sink_up = place_l2_route!(ox + 68, oy + 7, TileType::WireUp);
        push_pipeline!(l2_sink_up);
        let l1_vu = place_l1_route!(ox + 68, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        let l0_vu = place!(ox + 68, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B2.byte2: (40,2) → (70,7) L1A.RIGHT [L2 y=12, x=41..70]
    // Sprint 116 Phase 8 WireLeft at L2 y=11 skips x=40 (see Sprint 185 hop below).
    // East trunk at y=12 skips x=43,44 for B2.byte3 descent; L1 hop bridges the gap.
    {
        for y_off in 3..=12 {
            let idx = place_l2_route!(ox + 40, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        for x in (ox + 41)..=(ox + 70) {
            if x == ox + 43 || x == ox + 44 {
                continue;
            }
            let idx = place_l2_route!(x, oy + 12, TileType::WireRight);
            push_pipeline!(idx);
        }
        // L1 hop over B2.byte3 descent at x=43: L2(42)→L1(42)→L1(43)→L1(44)→L2(44)
        let _b2b2_hop_l1_vu = place_l1_route!(ox + 42, oy + 12, TileType::ViaUp);
        let _b2b2_hop_l1_wr1 = place_l1_route!(ox + 43, oy + 12, TileType::WireRight);
        let _b2b2_hop_l1_wr2 = place_l1_route!(ox + 44, oy + 12, TileType::WireRight);
        let l2_hop_vd = place_l2_route!(ox + 44, oy + 12, TileType::ViaDown);
        push_pipeline!(_b2b2_hop_l1_vu);
        push_pipeline!(_b2b2_hop_l1_wr1);
        push_pipeline!(_b2b2_hop_l1_wr2);
        push_pipeline!(l2_hop_vd);
        for y_off in 7..=11 {
            let idx = place_l2_route!(ox + 70, oy + y_off, TileType::WireUp);
            push_pipeline!(idx);
        }
        let l1_vu = place_l1_route!(ox + 70, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        let l0_vu = place!(ox + 70, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B1.byte2: (28,2) → (71,7) L1B.LEFT [L1 y=0 east, L2 descent at x=71]
    // L2 y=14 blocked by Sprint 108 pre-existing tiles at x=36-38,45-47,54-56,62.
    // L1 east at y=0 (completely clear), then L2 descent at x=71 to avoid crossing
    // B3.byte3 L1 east run at y=2 (x=56..74) and B1.byte3 L1 east run at y=4.
    // Signal starts at L1(28,2) ViaDown tap (reads L0 Mux16to1 output).
    {
        // L1 north from (28,2) to (28,0)
        let l1_wu1 = place_l1_route!(ox + 28, oy + 1, TileType::WireUp);
        push_pipeline!(l1_wu1);
        let l1_wu2 = place_l1_route!(ox + 28, oy + 0, TileType::WireUp);
        push_pipeline!(l1_wu2);
        // L1 east at y=0 from x=29 to x=71
        for x in (ox + 29)..=(ox + 71) {
            let idx = place_l1_route!(x, oy + 0, TileType::WireRight);
            push_pipeline!(idx);
        }
        // Drop to L2 for south descent (avoids L1 east run collisions at y=2,4)
        let l2_vd = place_l2_route!(ox + 71, oy + 0, TileType::ViaDown);
        push_pipeline!(l2_vd);
        // L2 south descent from (71,1) to (71,7)
        for y_off in 1..=7 {
            let idx = place_l2_route!(ox + 71, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        // Via stack to L0: L1 ViaUp reads L2, L0 WeightedViaUp reads L1
        let l1_vu = place_l1_route!(ox + 71, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        let l0_vu = place!(ox + 71, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B0.byte2: (16,2) → (73,7) L1B.RIGHT [L3 y=16 across BFS zone]
    // L1 bypass at y=3 over Sprint 152 L2 east trunk; keep the BFS-active span off
    // L2 so Sprint 97 cross-bank routes can traverse y=16/y=17 without encirclement.
    {
        // L1 bypass over L2 WireRight trunk at y=3 (L2(14..23,3))
        let l1_b0b2_wd1 = place_l1_route!(ox + 16, oy + 3, TileType::WireDown);
        push_pipeline!(l1_b0b2_wd1);
        let l1_b0b2_wd2 = place_l1_route!(ox + 16, oy + 4, TileType::WireDown);
        push_pipeline!(l1_b0b2_wd2);
        let l2_b0b2_resume = place_l2_route!(ox + 16, oy + 4, TileType::ViaDown);
        push_pipeline!(l2_b0b2_resume);
        for y_off in 5..=16 {
            let idx = place_l2_route!(ox + 16, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        // Lift to L3 immediately and stay there until east of the BFS corridor.
        let l3_b0b2_entry = place_l3_route!(ox + 16, oy + 16, TileType::ViaDown);
        push_pipeline!(l3_b0b2_entry);
        for x in (ox + 17)..=(ox + 51) {
            let idx = place_l3_route!(x, oy + 16, TileType::WireRight);
            push_pipeline!(idx);
        }
        let l2_b0b2_resume = place_l2_route!(ox + 51, oy + 16, TileType::ViaUp);
        push_pipeline!(l2_b0b2_resume);
        for x in (ox + 52)..=(ox + 73) {
            let idx = place_l2_route!(x, oy + 16, TileType::WireRight);
            push_pipeline!(idx);
        }
        // L1 bypass: trunk(y=16) → L1 over y=15,13 → L2 at y=12
        let l1_bypass_vu = place_l1_route!(ox + 73, oy + 16, TileType::ViaUp);
        push_pipeline!(l1_bypass_vu);
        for y_off in [12usize, 13, 14, 15] {
            let idx = place_l1_route!(ox + 73, oy + y_off, TileType::WireUp);
            push_pipeline!(idx);
        }
        let l2_bypass_resume = place_l2_route!(ox + 73, oy + 12, TileType::ViaDown);
        push_pipeline!(l2_bypass_resume);
        for y_off in 7..=11 {
            let idx = place_l2_route!(ox + 73, oy + y_off, TileType::WireUp);
            push_pipeline!(idx);
        }
        let l1_vu = place_l1_route!(ox + 73, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        let l0_vu = place!(ox + 73, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B3.byte3: (55,2) → (74,7) L1A.LEFT [L1 east at y=2, L2 descent at x=74]
    // L2 descent at x=55 blocked by B3.low trunk at L2(55,7). L2(55,2) ViaDown already
    // placed by rs relay (line 2374). Use L1 east from existing L1(55,2) ViaDown tap.
    {
        // L1 east from x=56 to x=74 at y=2 (reads from L1(55,2) ViaDown tap)
        for x in (ox + 56)..=(ox + 74) {
            let idx = place_l1_route!(x, oy + 2, TileType::WireRight);
            push_pipeline!(idx);
        }
        // L2 ViaDown at (74,2) reads from L1(74,2)
        let l2_vd = place_l2_route!(ox + 74, oy + 2, TileType::ViaDown);
        push_pipeline!(l2_vd);
        // L2 south descent at x=74 from y=3 to y=7
        for y_off in 3..=7 {
            let idx = place_l2_route!(ox + 74, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        let l1_vu = place_l1_route!(ox + 74, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        let l0_vu = place!(ox + 74, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B2.byte3: (43,2) → (76,7) L1A.RIGHT [L2 y=13, x=44..76]
    // Straight L2 descent y=3..13: Sprint 116 skips x=43 at y=11, B2.byte2 hops
    // over x=43 at y=12, so the descent column is unobstructed on L2.
    {
        for y_off in 3..=13 {
            let idx = place_l2_route!(ox + 43, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        // L2 east at y=13: split around R4 register tap at x=48 and Sprint 116 at x=57.
        // Both L1 and L2 occupied at register positions; bypass entirely on L3.
        for x in (ox + 44)..=(ox + 47) {
            let idx = place_l2_route!(x, oy + 13, TileType::WireRight);
            push_pipeline!(idx);
        }
        // L3 bypass: x=48 (R4 L0+L1+L2 register) through x=58 (Sprint 116 L1+L2 at x=57)
        let l3_b2b3_entry = place_l3_route!(ox + 47, oy + 13, TileType::ViaDown);
        push_pipeline!(l3_b2b3_entry);
        for x in (ox + 48)..=(ox + 58) {
            let idx = place_l3_route!(x, oy + 13, TileType::WireRight);
            push_pipeline!(idx);
        }
        let l2_b2b3_resume = place_l2_route!(ox + 58, oy + 13, TileType::ViaUp);
        push_pipeline!(l2_b2b3_resume);
        // Resume L2 east at y=13
        for x in (ox + 59)..=(ox + 76) {
            let idx = place_l2_route!(x, oy + 13, TileType::WireRight);
            push_pipeline!(idx);
        }
        for y_off in 7..=12 {
            let idx = place_l2_route!(ox + 76, oy + y_off, TileType::WireUp);
            push_pipeline!(idx);
        }
        let l1_vu = place_l1_route!(ox + 76, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        let l0_vu = place!(ox + 76, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B1.byte3: (31,2) → (77,7) L1B.LEFT [L1 y=4, x=32..77, L2 bypass at x=57]
    // L2 y=15 blocked by ViaDown at L2(57,15) + WireUp at L1(57,15).
    // Route on L1 y=4 which is clear except PC[5] ViaDown at L1(57,4) — bypassed via L2.
    // Signal starts at L1(31,2) ViaDown tap (reads L0 Mux16to1 output).
    {
        // L1 south from (31,2) to (31,4)
        let l1_wd1 = place_l1_route!(ox + 31, oy + 3, TileType::WireDown);
        push_pipeline!(l1_wd1);
        let l1_wd2 = place_l1_route!(ox + 31, oy + 4, TileType::WireDown);
        push_pipeline!(l1_wd2);
        // L1 east at y=4 from x=32 to x=56 (before PC[5] ViaDown at x=57)
        for x in (ox + 32)..=(ox + 56) {
            let idx = place_l1_route!(x, oy + 4, TileType::WireRight);
            push_pipeline!(idx);
        }
        // L2 bypass over L1(57,4) PC[5] ViaDown
        let l2_bypass_vd = place_l2_route!(ox + 56, oy + 4, TileType::ViaDown);
        push_pipeline!(l2_bypass_vd);
        let l2_bypass_wr1 = place_l2_route!(ox + 57, oy + 4, TileType::WireRight);
        push_pipeline!(l2_bypass_wr1);
        let l2_bypass_wr2 = place_l2_route!(ox + 58, oy + 4, TileType::WireRight);
        push_pipeline!(l2_bypass_wr2);
        let l1_bypass_resume = place_l1_route!(ox + 58, oy + 4, TileType::ViaUp);
        push_pipeline!(l1_bypass_resume);
        // L1 east at y=4 from x=59 to x=77 (after bypass)
        for x in (ox + 59)..=(ox + 77) {
            let idx = place_l1_route!(x, oy + 4, TileType::WireRight);
            push_pipeline!(idx);
        }
        // L1 south descent from (77,5) to (77,7)
        for y_off in 5..=7 {
            let idx = place_l1_route!(ox + 77, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        // L0 WeightedViaUp reads from L1(77,7)
        let l0_vu = place!(ox + 77, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // Route B0.byte3: (19,2) → (79,7) L1B.RIGHT [L3 descent west of Sprint 106]
    // L1 bypass at y=3 over Sprint 152 L2 east trunk, then move the descent onto L3
    // at x=15 so Sprint 106 can claim L2(19,11..15) for its selector launch fabric.
    {
        // L1 bypass over L2 WireRight trunk at y=3 (L2(14..23,3))
        let l1_b0b3_wd1 = place_l1_route!(ox + 19, oy + 3, TileType::WireDown);
        push_pipeline!(l1_b0b3_wd1);
        let l1_b0b3_wd2 = place_l1_route!(ox + 19, oy + 4, TileType::WireDown);
        push_pipeline!(l1_b0b3_wd2);
        let l2_b0b3_resume = place_l2_route!(ox + 19, oy + 4, TileType::ViaDown);
        push_pipeline!(l2_b0b3_resume);
        for y_off in 5..=10 {
            let idx = place_l2_route!(ox + 19, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        let l3_b0b3_entry = place_l3_route!(ox + 19, oy + 10, TileType::ViaDown);
        push_pipeline!(l3_b0b3_entry);
        for x in (ox + 15..=(ox + 18)).rev() {
            let idx = place_l3_route!(x, oy + 10, TileType::WireLeft);
            push_pipeline!(idx);
        }
        let l2_b0b3_mid = place_l2_route!(ox + 15, oy + 10, TileType::ViaUp);
        push_pipeline!(l2_b0b3_mid);
        let l2_b0b3_wd1 = place_l2_route!(ox + 15, oy + 11, TileType::WireDown);
        let l2_b0b3_wd2 = place_l2_route!(ox + 15, oy + 12, TileType::WireDown);
        push_pipeline!(l2_b0b3_wd1);
        push_pipeline!(l2_b0b3_wd2);
        let l3_b0b3_reentry = place_l3_route!(ox + 15, oy + 12, TileType::ViaDown);
        push_pipeline!(l3_b0b3_reentry);
        for y_off in 13..=17 {
            let idx = place_l3_route!(ox + 15, oy + y_off, TileType::WireDown);
            push_pipeline!(idx);
        }
        for x in (ox + 16)..=(ox + 51) {
            let idx = place_l3_route!(x, oy + 17, TileType::WireRight);
            push_pipeline!(idx);
        }
        let l2_b0b3_resume = place_l2_route!(ox + 51, oy + 17, TileType::ViaUp);
        push_pipeline!(l2_b0b3_resume);
        for x in (ox + 52)..=(ox + 79) {
            let idx = place_l2_route!(x, oy + 17, TileType::WireRight);
            push_pipeline!(idx);
        }
        for y_off in 7..=16 {
            let idx = place_l2_route!(ox + 79, oy + y_off, TileType::WireUp);
            push_pipeline!(idx);
        }
        let l1_vu = place_l1_route!(ox + 79, oy + 7, TileType::ViaUp);
        push_pipeline!(l1_vu);
        let l0_vu = place!(ox + 79, oy + 7, TileType::WeightedViaUp);
        sim.set_tile_mask(l0_vu, u64::MAX);
        push_pipeline!(l0_vu);
    }

    // -------------------------------------------------------------------------
    // Sprint 211: Super Mux — 4th selector level using PC[6]
    // -------------------------------------------------------------------------
    // Placed in ox+83..93 area (clear of Sprint 122/151 routing). PC[6]!=0
    // selects LEFT (Banks 4-7), PC[6]=0 selects RIGHT (Banks 0-3).
    // Both inputs are Const injection tiles — software injects correct values
    // before pipeline settle. PC[6] physically drives the selection.
    //
    // Layout: PC[6] BitSelect at (81, oy+4), descent to (81, oy+10), east trunk.
    // Per lane at sx = [83, 86, 89, 92]:
    //   (sx-1, oy+11) = Const (LEFT: Banks 4-7 injection)
    //   (sx,   oy+11) = Mux   (Super Mux, UP reads PC[6] trunk at oy+10)
    //   (sx+1, oy+11) = Const (RIGHT: Banks 0-3 injection)

    // Phase 3A: PC[6] extraction
    let _pc6_src_wd = place!(ox + 80, oy + 4, TileType::WireDown);
    push_pipeline!(_pc6_src_wd);
    let _pc6_bit_idx = place!(ox + 81, oy + 4, TileType::BitSelect);
    let _ = place_val!(ox + 82, oy + 4, TileType::Const, 6);
    push_pipeline!(_pc6_bit_idx);

    // Phase 3B: PC[6] descent (81, oy+5..10) and east trunk (82..92, oy+10)
    for y_off in 5..=10 {
        let idx = place!(ox + 81, oy + y_off, TileType::WireDown);
        push_pipeline!(idx);
    }
    for x in (ox + 82)..=(ox + 92) {
        let idx = place!(x, oy + 10, TileType::WireRight);
        push_pipeline!(idx);
    }

    // Phase 3C: Super Mux tiles (dual-injection)
    let super_lane_bases: [usize; 4] = [ox + 83, ox + 86, ox + 89, ox + 92];
    let mut super_mux_indices = [0usize; 4];
    let mut super_mux_inject_indices = [0usize; 4];
    let mut super_mux_inject_right_indices = [0usize; 4];

    for (lane, &sx) in super_lane_bases.iter().enumerate() {
        let inject_left = place_val!(sx - 1, oy + 11, TileType::Const, 0);
        let super_mux = place!(sx, oy + 11, TileType::Mux);
        let inject_right = place_val!(sx + 1, oy + 11, TileType::Const, 0);
        push_pipeline!(inject_left);
        push_pipeline!(super_mux);
        push_pipeline!(inject_right);

        super_mux_indices[lane] = super_mux;
        super_mux_inject_indices[lane] = inject_left;
        super_mux_inject_right_indices[lane] = inject_right;
    }

    // --- Phase 2E: Final selector output delivery to legacy ingress points ---
    // Low/high final westback routes are materialized after collision-free closure.

    // -------------------------------------------------------------------------
    // Sprint 95: physical extraction zone (IR fields)
    // -------------------------------------------------------------------------
    // ir_high tap via L1 bridge because high ROM mux is boxed on L0.
    let ir_high_tap_via_down_l1 = place_l1_route!(ox + 13, oy + 2, TileType::ViaDown);
    let ir_high_tap_wd1_l1 = place_l1_route!(ox + 13, oy + 3, TileType::WireDown);
    // Sprint 150: inject selected high at L2(13,4) -> L1(13,4) -> L0(13,4).
    let ir_high_tap_wd2_l1 = place_l1_route!(ox + 13, oy + 4, TileType::ViaUp);
    let ir_high_tap_via_up_l0 = place!(ox + 13, oy + 4, TileType::ViaUp);
    push_pipeline!(ir_high_tap_via_down_l1);
    push_pipeline!(ir_high_tap_wd1_l1);
    push_pipeline!(ir_high_tap_wd2_l1);
    push_pipeline!(ir_high_tap_via_up_l0);

    // Sprint 150: inject selected low at L2(1,4) -> L1(1,4) -> L0(1,4).
    let ir_low_l1_ingress_idx = place_l1_route!(ox + 1, oy + 4, TileType::ViaUp);
    let ir_low_wd1_idx = place!(ox + 1, oy + 3, TileType::WireDown);
    let ir_low_wd2_idx = place!(ox + 1, oy + 4, TileType::ViaUp);
    let ir_low_wd3_idx = place!(ox + 1, oy + 5, TileType::WireDown);
    push_pipeline!(ir_low_l1_ingress_idx);
    push_pipeline!(ir_low_wd1_idx);
    push_pipeline!(ir_low_wd2_idx);
    push_pipeline!(ir_low_wd3_idx);

    // opcode_shr = ir_high >> 3 (5-bit opcode).
    let extract_opcode_shr_idx = place!(ox + 14, oy + 4, TileType::Shr);
    let _extract_opcode_shr_amt_idx = place_val!(ox + 15, oy + 4, TileType::Const, 3);
    push_pipeline!(extract_opcode_shr_idx);

    // Westward raw ir_high distribution for rd bit extraction.
    for x in (ox + 3)..=(ox + 12) {
        let idx = place!(x, oy + 4, TileType::WireLeft);
        push_pipeline!(idx);
    }

    // rd bit extraction and opcode[4].
    let rd2_wd_1 = place!(ox + 5, oy + 5, TileType::WireDown);
    let rd2_wd_2 = place!(ox + 5, oy + 6, TileType::WireDown);
    let extract_rd2_idx = place!(ox + 6, oy + 6, TileType::BitSelect);
    let _extract_rd2_bit_idx = place_val!(ox + 7, oy + 6, TileType::Const, 2);
    push_pipeline!(rd2_wd_1);
    push_pipeline!(rd2_wd_2);
    push_pipeline!(extract_rd2_idx);

    let rd1_wd_1 = place!(ox + 8, oy + 5, TileType::WireDown);
    let rd1_wd_2 = place!(ox + 8, oy + 6, TileType::WireDown);
    let extract_rd1_idx = place!(ox + 9, oy + 6, TileType::BitSelect);
    let _extract_rd1_bit_idx = place_val!(ox + 10, oy + 6, TileType::Const, 1);
    push_pipeline!(rd1_wd_1);
    push_pipeline!(rd1_wd_2);
    push_pipeline!(extract_rd1_idx);

    let rd0_wd_1 = place!(ox + 11, oy + 5, TileType::WireDown);
    let rd0_wd_2 = place!(ox + 11, oy + 6, TileType::WireDown);
    let extract_rd0_idx = place!(ox + 12, oy + 6, TileType::BitSelect);
    let _extract_rd0_bit_idx = place_val!(ox + 13, oy + 6, TileType::Const, 0);
    push_pipeline!(rd0_wd_1);
    push_pipeline!(rd0_wd_2);
    push_pipeline!(extract_rd0_idx);

    let op4_wd_1 = place!(ox + 14, oy + 5, TileType::WireDown);
    let op4_wd_2 = place!(ox + 14, oy + 6, TileType::WireDown);
    let extract_opcode_bit4_idx = place!(ox + 15, oy + 6, TileType::BitSelect);
    let _extract_opcode_bit4_pos_idx = place_val!(ox + 16, oy + 6, TileType::Const, 4);
    push_pipeline!(op4_wd_1);
    push_pipeline!(op4_wd_2);
    push_pipeline!(extract_opcode_bit4_idx);

    // rs_field = ir_low >> 5 (packed 3-bit field).
    let extract_rs_field_idx = place!(ox + 2, oy + 5, TileType::Shr);
    let _extract_rs_shift_idx = place_val!(ox + 3, oy + 5, TileType::Const, 5);
    push_pipeline!(extract_rs_field_idx);

    // -------------------------------------------------------------------------
    // Physical decoder LUTs (Sprint 92)
    // -------------------------------------------------------------------------
    // ctrl_a / ctrl_b encoding for all 32 opcodes.
    // Sprint 198: ctrl_a_values now references module-level CTRL_A_LUT (single source of truth).
    let ctrl_a_values = CTRL_A_LUT;
    // Sprint 125: ctrl_b bits 6-7 encode is_call/is_ret for physical LR/Target Mux.
    //   bit 6 = is_call (opcode 31 = CALL)
    //   bit 7 = is_ret  (opcode 21 = RET)
    // Existing consumers read bits 0-5 only; new bits are orthogonal.
    let ctrl_b_values: [u8; 32] = [
        0x00, 0x21, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x87, 0x08, 0x10, 0x08, 0x10, 0x01, 0x02, 0x03, 0x04,
        0x05, 0x46,
    ];

    // ctrl_a tiles
    let _ctrl_a_b_up = place_val!(
        ox + 20,
        oy + 6,
        TileType::Const,
        pack_8(&ctrl_a_values[24..32])
    );
    let _ctrl_a_a_up = place_val!(
        ox + 24,
        oy + 6,
        TileType::Const,
        pack_8(&ctrl_a_values[8..16])
    );
    let _ctrl_a_b_left = place_val!(
        ox + 19,
        oy + 7,
        TileType::Const,
        pack_8(&ctrl_a_values[16..24])
    );
    let ctrl_a_lut_b_idx = place!(ox + 20, oy + 7, TileType::Mux16to1);
    push_pipeline!(ctrl_a_lut_b_idx);
    let ctrl_a_sel_b_idx = place!(ox + 21, oy + 7, TileType::ViaUp);
    push_pipeline!(ctrl_a_sel_b_idx);
    let ctrl_a_bit4_idx = place!(ox + 22, oy + 7, TileType::ViaUp);
    push_pipeline!(ctrl_a_bit4_idx);
    let _ctrl_a_a_left = place_val!(
        ox + 23,
        oy + 7,
        TileType::Const,
        pack_8(&ctrl_a_values[0..8])
    );
    let ctrl_a_lut_a_idx = place!(ox + 24, oy + 7, TileType::Mux16to1);
    push_pipeline!(ctrl_a_lut_a_idx);
    let ctrl_a_sel_a_idx = place!(ox + 25, oy + 7, TileType::ViaUp);
    push_pipeline!(ctrl_a_sel_a_idx);
    let ctrl_a_wd_b_idx = place!(ox + 20, oy + 8, TileType::WireDown);
    let ctrl_a_wr_idx = place!(ox + 21, oy + 8, TileType::WireRight);
    let ctrl_a_mux_idx = place!(ox + 22, oy + 8, TileType::Mux);
    let ctrl_a_wl_idx = place!(ox + 23, oy + 8, TileType::WireLeft);
    let ctrl_a_wd_a_idx = place!(ox + 24, oy + 8, TileType::WireDown);
    push_pipeline!(ctrl_a_wd_b_idx);
    push_pipeline!(ctrl_a_wr_idx);
    push_pipeline!(ctrl_a_mux_idx);
    push_pipeline!(ctrl_a_wl_idx);
    push_pipeline!(ctrl_a_wd_a_idx);

    // ctrl_b tiles (same pattern +10 columns east)
    let _ctrl_b_b_up = place_val!(
        ox + 30,
        oy + 6,
        TileType::Const,
        pack_8(&ctrl_b_values[24..32])
    );
    let _ctrl_b_a_up = place_val!(
        ox + 34,
        oy + 6,
        TileType::Const,
        pack_8(&ctrl_b_values[8..16])
    );
    let _ctrl_b_b_left = place_val!(
        ox + 29,
        oy + 7,
        TileType::Const,
        pack_8(&ctrl_b_values[16..24])
    );
    let ctrl_b_lut_b_idx = place!(ox + 30, oy + 7, TileType::Mux16to1);
    push_pipeline!(ctrl_b_lut_b_idx);
    let ctrl_b_sel_b_idx = place!(ox + 31, oy + 7, TileType::ViaUp);
    push_pipeline!(ctrl_b_sel_b_idx);
    let ctrl_b_bit4_idx = place!(ox + 32, oy + 7, TileType::ViaUp);
    push_pipeline!(ctrl_b_bit4_idx);
    let _ctrl_b_a_left = place_val!(
        ox + 33,
        oy + 7,
        TileType::Const,
        pack_8(&ctrl_b_values[0..8])
    );
    let ctrl_b_lut_a_idx = place!(ox + 34, oy + 7, TileType::Mux16to1);
    push_pipeline!(ctrl_b_lut_a_idx);
    let ctrl_b_sel_a_idx = place!(ox + 35, oy + 7, TileType::ViaUp);
    push_pipeline!(ctrl_b_sel_a_idx);
    let ctrl_b_wd_b_idx = place!(ox + 30, oy + 8, TileType::WireDown);
    let ctrl_b_wr_idx = place!(ox + 31, oy + 8, TileType::WireRight);
    let ctrl_b_mux_idx = place!(ox + 32, oy + 8, TileType::Mux);
    let ctrl_b_wl_idx = place!(ox + 33, oy + 8, TileType::WireLeft);
    let ctrl_b_wd_a_idx = place!(ox + 34, oy + 8, TileType::WireDown);
    push_pipeline!(ctrl_b_wd_b_idx);
    push_pipeline!(ctrl_b_wr_idx);
    push_pipeline!(ctrl_b_mux_idx);
    push_pipeline!(ctrl_b_wl_idx);
    push_pipeline!(ctrl_b_wd_a_idx);

    // -------------------------------------------------------------------------
    // Sprint 95: physical decoder selector delivery on L1
    // -------------------------------------------------------------------------
    // opcode_shr transport to four LUT selector sites.
    let opcode_shr_via_down_l1 = place_l1_route!(ox + 14, oy + 4, TileType::ViaDown);
    let opcode_shr_wd_l1 = place_l1_route!(ox + 14, oy + 5, TileType::WireDown);
    push_pipeline!(opcode_shr_via_down_l1);
    push_pipeline!(opcode_shr_wd_l1);
    for x in (ox + 15)..=(ox + 35) {
        let idx = place_l1_route!(x, oy + 5, TileType::WireRight);
        push_pipeline!(idx);
    }
    for x in [ox + 21, ox + 25, ox + 31, ox + 35] {
        let d1 = place_l1_route!(x, oy + 6, TileType::WireDown);
        let d2 = place_l1_route!(x, oy + 7, TileType::WireDown);
        push_pipeline!(d1);
        push_pipeline!(d2);
    }

    // opcode[4] transport to two bank-select selector sites.
    let opcode_bit4_via_down_l1 = place_l1_route!(ox + 15, oy + 6, TileType::ViaDown);
    let opcode_bit4_wd_1_l1 = place_l1_route!(ox + 15, oy + 7, TileType::WireDown);
    let opcode_bit4_wd_2_l1 = place_l1_route!(ox + 15, oy + 8, TileType::WireDown);
    push_pipeline!(opcode_bit4_via_down_l1);
    push_pipeline!(opcode_bit4_wd_1_l1);
    push_pipeline!(opcode_bit4_wd_2_l1);
    // Sprint 107: shorten bus to ox+31 so L1 (ox+32, oy+8) is free for ctrl_b ViaDown.
    for x in (ox + 16)..=(ox + 31) {
        let idx = place_l1_route!(x, oy + 8, TileType::WireRight);
        push_pipeline!(idx);
    }
    // ctrl_a bank Mux: opcode[4] still delivered via L1 WireUp at (ox+22, oy+7).
    let bit4_up_22 = place_l1_route!(ox + 22, oy + 7, TileType::WireUp);
    push_pipeline!(bit4_up_22);
    // ctrl_b bank Mux: opcode[4] delivered via L2 relay (Sprint 107).
    // L1 ViaUp at (ox+32, oy+7) reads L2 → L0 ViaUp reads L1 → bank Mux UP.
    let bit4_up_32 = place_l1_route!(ox + 32, oy + 7, TileType::ViaUp);
    push_pipeline!(bit4_up_32);
    // L2 relay: tap opcode_bit4 from L1 (ox+31, oy+8), route to (ox+32, oy+7).
    let bit4_l2_tap = place_l2_route!(ox + 31, oy + 8, TileType::ViaDown);
    let bit4_l2_wu = place_l2_route!(ox + 31, oy + 7, TileType::WireUp);
    let bit4_l2_wr = place_l2_route!(ox + 32, oy + 7, TileType::WireRight);
    push_pipeline!(bit4_l2_tap);
    push_pipeline!(bit4_l2_wu);
    push_pipeline!(bit4_l2_wr);

    // -------------------------------------------------------------------------
    // Register file: 16 permanent Register64 tiles.
    // Keep legacy 8-tile coordinates for lanes 0..7 so existing operand/commit
    // wiring remains intact; add 8 dedicated tiles for registers 8..15.
    // -------------------------------------------------------------------------
    let mut reg_indices = [0usize; 16];
    let reg_rows = [oy + 13, oy + 16, oy + 19, oy + 22];
    let reg_col_a = ox + 28;
    let reg_col_b = ox + 48;
    let reg_col_c = ox + 88;
    let reg_col_d = ox + 92;
    for (i, row) in reg_rows.iter().copied().enumerate() {
        // Legacy operand-tree-visible lanes (R0..R7).
        reg_indices[i] = place_val!(reg_col_a, row, TileType::Register64, 0);
        reg_indices[i + 4] = place_val!(reg_col_b, row, TileType::Register64, 0);
        // Dedicated permanent tiles for high registers (R8..R15).
        reg_indices[i + 8] = place_val!(reg_col_c, row, TileType::Register64, 0);
        reg_indices[i + 12] = place_val!(reg_col_d, row, TileType::Register64, 0);
    }
    // Phase 97.0: L1 register taps used to validate cross-bank routing viability.
    // These taps are passive in Sprint 96/97.0 and become data-route sources in 97.1+.
    let mut reg_tap_l1_indices = [0usize; 16];
    for i in 0..16usize {
        let (x, y) = if i < 4 {
            (reg_col_a, reg_rows[i])
        } else if i < 8 {
            (reg_col_b, reg_rows[i - 4])
        } else if i < 12 {
            (reg_col_c, reg_rows[i - 8])
        } else {
            (reg_col_d, reg_rows[i - 12])
        };
        let tap_idx = place_l1_route!(x, y, TileType::ViaDown);
        reg_tap_l1_indices[i] = tap_idx;
        push_reg_data!(i, tap_idx);
    }

    // -------------------------------------------------------------------------
    // Sprint 94: clocked writeback commit fabric
    // -------------------------------------------------------------------------
    // Sprint 102: wb_data source is now a mux with software override.
    // Sprint 109: enable at (ox+38, oy+9) is now physical (!use_alu via WireDown).
    // Sprint 116: wb_data override value is now a physical L2 cascade (MOV/LDI).
    // L0 (ox+37, oy+10) changed from Const to WireUp — reads from L0 ViaUp at oy+11.
    // Sprint 122: wb_data is now fully physical for all opcodes (LD/LDB via L2 route).
    let _wb_data_wireup_l0 = place!(ox + 37, oy + 10, TileType::WireUp);
    let wb_data_mux_idx = place!(ox + 38, oy + 10, TileType::Mux);
    let mut commit_dirty_indices = Vec::new();
    commit_dirty_indices.append(&mut pc_path_dirty_indices);
    // Sprint 169: reg-WB tiles separated for conditional dirty seeding.
    let mut reg_wb_dirty_indices = Vec::new();
    reg_wb_dirty_indices.push(wb_data_mux_idx);

    // L1 wb-data bus fanout from source at (ox+38, oy+10)
    let wb_via_down_l1 = place_l1_route!(ox + 38, oy + 10, TileType::ViaDown);
    reg_wb_dirty_indices.push(wb_via_down_l1);
    for x in (ox + 25)..=(ox + 37) {
        let idx = place_l1_route!(x, oy + 10, TileType::WireLeft);
        reg_wb_dirty_indices.push(idx);
    }
    for x in (ox + 39)..=(ox + 45) {
        let idx = place_l1_route!(x, oy + 10, TileType::WireRight);
        reg_wb_dirty_indices.push(idx);
    }
    let wb_drop_a_l1 = place_l1_route!(ox + 25, oy + 11, TileType::WireDown);
    reg_wb_dirty_indices.push(wb_drop_a_l1);
    let wb_drop_b_l1 = place_l1_route!(ox + 45, oy + 11, TileType::WireDown);
    reg_wb_dirty_indices.push(wb_drop_b_l1);

    // L0 delivery from L1 into register rows
    let wb_via_up_a = place!(ox + 25, oy + 11, TileType::ViaUp);
    reg_wb_dirty_indices.push(wb_via_up_a);
    let wb_via_up_b = place!(ox + 45, oy + 11, TileType::ViaUp);
    reg_wb_dirty_indices.push(wb_via_up_b);
    for y in (oy + 12)..=(oy + 22) {
        let idx_a = place!(ox + 25, y, TileType::WireDown);
        reg_wb_dirty_indices.push(idx_a);
        let idx_b = place!(ox + 45, y, TileType::WireDown);
        reg_wb_dirty_indices.push(idx_b);
    }

    // Per-register merge muxes + one-hot WE mask distribution.
    // Sprint 101: we_mask source is now physical:
    //   we_mask = Decoder3to8(rd_field) & reg_write(ctrl_a[3])
    // Source remains at (ox+35, oy+9) to preserve existing fanout wiring.
    // L0 ctrl_a tap + local reg_write extraction path.
    // This avoids L1 selector corridors already occupied by decoder routing.
    let ctrl_a_down_l0 = place!(ox + 22, oy + 9, TileType::WireDown);
    let ctrl_a_stage_x_wd_l0 = place!(ox + 22, oy + 10, TileType::WireDown);
    push_pipeline!(ctrl_a_down_l0);
    push_pipeline!(ctrl_a_stage_x_wd_l0);
    for x in (ox + 23)..=(ox + 31) {
        let idx = place!(x, oy + 10, TileType::WireRight);
        push_pipeline!(idx);
    }
    let reg_write_bit_idx = place!(ox + 32, oy + 10, TileType::BitSelect);
    let _reg_write_pos_idx = place_val!(ox + 33, oy + 10, TileType::Const, 3);
    let reg_write_drop_idx = place!(ox + 32, oy + 11, TileType::WireDown);
    for x in (ox + 33)..=(ox + 36) {
        let idx = place!(x, oy + 11, TileType::WireRight);
        push_pipeline!(idx);
    }
    let reg_write_up_1_idx = place!(ox + 36, oy + 10, TileType::WireUp);
    let reg_write_up_2_idx = place!(ox + 36, oy + 9, TileType::WireUp);
    push_pipeline!(reg_write_bit_idx);
    push_pipeline!(reg_write_drop_idx);
    push_pipeline!(reg_write_up_1_idx);
    push_pipeline!(reg_write_up_2_idx);

    // L1 ctrl_a tap retained for Sprint 101 flag WE physical source.
    let ctrl_a_tap_l1 = place_l1_route!(ox + 22, oy + 9, TileType::ViaDown);
    push_pipeline!(ctrl_a_tap_l1);

    // rd packed field path (L2 bridge) from ir_high tap to local decoder input.
    // Sprint 152: changed ViaDown→WireUp so this tile reads from L2(13,4) = westback
    // ir_high (correct bank) instead of L1(13,3)→L1(13,2)→L0(13,2) Bank0 Mux.
    let rd_tap_l2 = place_l2_route!(ox + 13, oy + 3, TileType::WireUp);
    push_pipeline!(rd_tap_l2);
    for x in (ox + 14)..=(ox + 23) {
        let idx = place_l2_route!(x, oy + 3, TileType::WireRight);
        push_pipeline!(idx);
    }
    for y in (oy + 4)..=(oy + 9) {
        let idx = place_l2_route!(ox + 23, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    let rd_l2_to_l1_idx = place_l1_route!(ox + 23, oy + 9, TileType::ViaUp);
    let rd_field_via_up_idx = place!(ox + 23, oy + 9, TileType::ViaUp);
    let rd_onehot_decode_idx = place!(ox + 24, oy + 9, TileType::Decoder3to8);
    let mut rd_decode_l0_chain = Vec::new();
    for x in (ox + 25)..=(ox + 34) {
        let idx = place!(x, oy + 9, TileType::WireRight);
        push_pipeline!(idx);
        rd_decode_l0_chain.push(idx);
    }
    let we_mask_const_idx = place!(ox + 35, oy + 9, TileType::And);
    push_pipeline!(rd_l2_to_l1_idx);
    push_pipeline!(rd_field_via_up_idx);
    push_pipeline!(rd_onehot_decode_idx);
    push_pipeline!(we_mask_const_idx);

    let we_mask_via_down_l1 = place_l1_route!(ox + 35, oy + 9, TileType::ViaDown);
    reg_wb_dirty_indices.push(we_mask_via_down_l1);
    for x in (ox + 24)..=(ox + 34) {
        let idx = place_l1_route!(x, oy + 9, TileType::WireLeft);
        reg_wb_dirty_indices.push(idx);
    }
    for x in (ox + 36)..=(ox + 46) {
        let idx = place_l1_route!(x, oy + 9, TileType::WireRight);
        reg_wb_dirty_indices.push(idx);
    }
    for y in (oy + 10)..=(oy + 21) {
        let idx_a = place_l1_route!(ox + 24, y, TileType::WireDown);
        reg_wb_dirty_indices.push(idx_a);
        let idx_b = place_l1_route!(ox + 46, y, TileType::WireDown);
        reg_wb_dirty_indices.push(idx_b);
    }

    let mut _merge_mux_indices = [0usize; 8];
    let reg_rows = [oy + 13, oy + 16, oy + 19, oy + 22];
    for (i, row) in reg_rows.iter().copied().enumerate() {
        let wr_idx = place!(ox + 26, row, TileType::WireRight);
        reg_wb_dirty_indices.push(wr_idx);
        let we_hop_a1_idx = place_l1_route!(ox + 25, row - 1, TileType::WireRight);
        let we_hop_a2_idx = place_l1_route!(ox + 26, row - 1, TileType::WireRight);
        reg_wb_dirty_indices.push(we_hop_a1_idx);
        reg_wb_dirty_indices.push(we_hop_a2_idx);
        let we_via_idx = place!(ox + 26, row - 1, TileType::ViaUp);
        let we_idx = place!(ox + 27, row - 1, TileType::BitSelect);
        let _we_bit_idx = place_val!(ox + 28, row - 1, TileType::Const, i as u64);
        reg_wb_dirty_indices.push(we_via_idx);
        reg_wb_dirty_indices.push(we_idx);
        let mux_idx = place!(ox + 27, row, TileType::Mux);
        _merge_mux_indices[i] = mux_idx;
        reg_wb_dirty_indices.push(mux_idx);
    }
    for (i, row) in reg_rows.iter().copied().enumerate() {
        let wr_idx = place!(ox + 46, row, TileType::WireRight);
        reg_wb_dirty_indices.push(wr_idx);
        let we_via_idx = place!(ox + 46, row - 1, TileType::ViaUp);
        let we_idx = place!(ox + 47, row - 1, TileType::BitSelect);
        let _we_bit_idx = place_val!(ox + 48, row - 1, TileType::Const, (i + 4) as u64);
        reg_wb_dirty_indices.push(we_via_idx);
        reg_wb_dirty_indices.push(we_idx);
        let mux_idx = place!(ox + 47, row, TileType::Mux);
        _merge_mux_indices[i + 4] = mux_idx;
        reg_wb_dirty_indices.push(mux_idx);
    }

    // -------------------------------------------------------------------------
    // Sprint 236: High-register commit fabric (R8-R15 merge muxes)
    // -------------------------------------------------------------------------
    // The mid-zone (ox+46..70) is heavily congested across L1-L3 at oy+9..11
    // (Sprint 107/116/122/151 routes). Instead of routing wb_data and WE mask
    // physically through 40+ columns, we use Const-injection — the same pattern
    // as Super Mux (S234) and L-enable (S235).
    //
    // Per-register structure:
    //   Const(0) at (col-2, row):    wb_data injection → feeds merge mux LEFT
    //   Const(0) at (col-1, row-1):  WE bit injection  → feeds merge mux UP
    //   Mux at (col-1, row):         merge mux (UP!=0 → LEFT=wb_data, UP==0 → RIGHT=reg)
    //   Register64 at (col, row):    captures Mux output on rising edge
    //
    // Software injects wb_data and WE bit before commit, restores to 0 after.
    let mut high_reg_wb_data_const_indices = [0usize; 8];
    let mut high_reg_we_const_indices = [0usize; 8];
    let mut high_reg_merge_mux_indices = [0usize; 8];

    // R8-R11: register column at ox+88
    for (i, row) in reg_rows.iter().copied().enumerate() {
        let wb_const_idx = place_val!(ox + 86, row, TileType::Const, 0);
        high_reg_wb_data_const_indices[i] = wb_const_idx;
        reg_wb_dirty_indices.push(wb_const_idx);
        let we_const_idx = place_val!(ox + 87, row - 1, TileType::Const, 0);
        high_reg_we_const_indices[i] = we_const_idx;
        reg_wb_dirty_indices.push(we_const_idx);
        let mux_idx = place!(ox + 87, row, TileType::Mux);
        high_reg_merge_mux_indices[i] = mux_idx;
        reg_wb_dirty_indices.push(mux_idx);
    }
    // R12-R15: register column at ox+92
    for (i, row) in reg_rows.iter().copied().enumerate() {
        let wb_const_idx = place_val!(ox + 90, row, TileType::Const, 0);
        high_reg_wb_data_const_indices[i + 4] = wb_const_idx;
        reg_wb_dirty_indices.push(wb_const_idx);
        let we_const_idx = place_val!(ox + 91, row - 1, TileType::Const, 0);
        high_reg_we_const_indices[i + 4] = we_const_idx;
        reg_wb_dirty_indices.push(we_const_idx);
        let mux_idx = place!(ox + 91, row, TileType::Mux);
        high_reg_merge_mux_indices[i + 4] = mux_idx;
        reg_wb_dirty_indices.push(mux_idx);
    }

    // -------------------------------------------------------------------------
    // Operand selection trees (Sprint 93)
    // -------------------------------------------------------------------------
    // Tree A (rd): canonical mapping D0..D7 == R0..R7.
    // Tree B (rs): bank-swapped mapping D = R XOR 0b100 to simplify routing.
    let tree_row = oy + 24;
    let op_a_tree = wire_operand_tree_8to1(
        sim,
        grid_width,
        &mut tile_count,
        &mut placed_mask,
        ox + 20,
        tree_row,
    );
    let op_b_tree = wire_operand_tree_8to1(
        sim,
        grid_width,
        &mut tile_count,
        &mut placed_mask,
        ox + 40,
        tree_row,
    );

    // Sprint 97 ingress points:
    //   Tree A D4..D7 remain WireDown at row oy+25 and read physical data via ViaUp at row oy+24.
    //   Tree B D4..D7 switch to WireUp at row oy+25 and read physical data via ViaUp at row oy+26.
    let op_a_d4_ingress = place!(ox + 26, oy + 24, TileType::ViaUp); // D4 ingress (R4)
    let op_a_d5_ingress = place!(ox + 24, oy + 24, TileType::ViaUp); // D5 ingress (R5)
    let op_a_d6_ingress = place!(ox + 22, oy + 24, TileType::ViaUp); // D6 ingress (R6)
    let op_a_d7_ingress = place!(ox + 20, oy + 24, TileType::ViaUp); // D7 ingress (R7)
    push_reg_data!(4, op_a_d4_ingress);
    push_reg_data!(5, op_a_d5_ingress);
    push_reg_data!(6, op_a_d6_ingress);
    push_reg_data!(7, op_a_d7_ingress);

    let _ = place!(ox + 46, oy + 25, TileType::WireUp); // D4
    let _ = place!(ox + 44, oy + 25, TileType::WireUp); // D5
    let _ = place!(ox + 42, oy + 25, TileType::WireUp); // D6
    let _ = place!(ox + 40, oy + 25, TileType::WireUp); // D7
    let op_b_d4_ingress = place!(ox + 46, oy + 26, TileType::ViaUp); // D4 ingress (R0)
    let op_b_d5_ingress = place!(ox + 44, oy + 26, TileType::ViaUp); // D5 ingress (R1)
    let op_b_d6_ingress = place!(ox + 42, oy + 26, TileType::ViaUp); // D6 ingress (R2)
    let op_b_d7_ingress = place!(ox + 40, oy + 26, TileType::ViaUp); // D7 ingress (R3)
    push_reg_data!(0, op_b_d4_ingress);
    push_reg_data!(1, op_b_d5_ingress);
    push_reg_data!(2, op_b_d6_ingress);
    push_reg_data!(3, op_b_d7_ingress);

    for &idx in &op_a_tree.dirty_indices {
        push_pipeline!(idx);
    }
    for &idx in &op_b_tree.dirty_indices {
        push_pipeline!(idx);
    }

    macro_rules! place_tree_dirty {
        ($reg:expr, $x:expr, $y:expr, $tt:expr) => {{
            let idx = place!($x, $y, $tt);
            push_reg_data!($reg, idx);
            idx
        }};
    }

    // Tree A routing ----------------------------------------------------------
    // Bank A (R0..R3) -> D0..D3 at cols 34,32,30,28 (direct)
    for x in (ox + 29)..=(ox + 34) {
        let _ = place_tree_dirty!(0, x, oy + 13, TileType::WireRight); // R0 -> D0
    }
    for y in (oy + 14)..=(oy + 25) {
        let _ = place_tree_dirty!(0, ox + 34, y, TileType::WireDown);
    }

    for x in (ox + 29)..=(ox + 32) {
        let _ = place_tree_dirty!(1, x, oy + 16, TileType::WireRight); // R1 -> D1
    }
    for y in (oy + 17)..=(oy + 25) {
        let _ = place_tree_dirty!(1, ox + 32, y, TileType::WireDown);
    }

    for x in (ox + 29)..=(ox + 30) {
        let _ = place_tree_dirty!(2, x, oy + 19, TileType::WireRight); // R2 -> D2
    }
    for y in (oy + 20)..=(oy + 25) {
        let _ = place_tree_dirty!(2, ox + 30, y, TileType::WireDown);
    }

    for y in (oy + 23)..=(oy + 25) {
        let _ = place_tree_dirty!(3, ox + 28, y, TileType::WireDown); // R3 -> D3
    }

    // Sprint 97: cross-bank data route on L2.
    // L1 ingress taps that feed L0 ViaUp ingress points.
    for (x, y, reg) in [
        (ox + 26, oy + 24, 4usize),
        (ox + 24, oy + 24, 5usize),
        (ox + 22, oy + 24, 6usize),
        (ox + 20, oy + 24, 7usize),
        (ox + 46, oy + 26, 0usize),
        (ox + 44, oy + 26, 1usize),
        (ox + 42, oy + 26, 2usize),
        (ox + 40, oy + 26, 3usize),
    ] {
        let idx = place_l1_route!(x, y, TileType::ViaUp);
        push_reg_data!(reg, idx);
    }

    // Tree B routing ----------------------------------------------------------
    // Tree B data mapping:
    //   D0..D3 <- R4..R7 (physical routes below)
    //   D4..D7 <- R0..R3 (Sprint 97 physical cross-bank routes below)

    // Bank B (R4..R7) -> D0..D3
    for x in (ox + 49)..=(ox + 54) {
        let _ = place_tree_dirty!(4, x, oy + 13, TileType::WireRight); // R4 -> D0
    }
    for y in (oy + 14)..=(oy + 25) {
        let _ = place_tree_dirty!(4, ox + 54, y, TileType::WireDown);
    }

    for x in (ox + 49)..=(ox + 52) {
        let _ = place_tree_dirty!(5, x, oy + 16, TileType::WireRight); // R5 -> D1
    }
    for y in (oy + 17)..=(oy + 25) {
        let _ = place_tree_dirty!(5, ox + 52, y, TileType::WireDown);
    }

    for x in (ox + 49)..=(ox + 50) {
        let _ = place_tree_dirty!(6, x, oy + 19, TileType::WireRight); // R6 -> D2
    }
    for y in (oy + 20)..=(oy + 25) {
        let _ = place_tree_dirty!(6, ox + 50, y, TileType::WireDown);
    }

    for y in (oy + 23)..=(oy + 25) {
        let _ = place_tree_dirty!(7, ox + 48, y, TileType::WireDown); // R7 -> D3
    }

    // Sprint 108: pre-reserve 3 L2 cells at (ox+45..47, oy+14) for use_imm L2 hop
    // around WE mask Bank B column (ox+46). Without this, BFS R0→B.D4 claims (46,14).
    for x in (ox + 45)..=(ox + 47) {
        reserve_l2!(x, oy + 14);
    }
    // Sprint 106 later consumes fixed BitSelect tap vias at (ox+23, oy+11..13),
    // west launch rows at oy+11..13, and south columns at ox+17..19. Reserve the
    // BFS-exposed prefix now so Sprint 97 cannot claim those fixed lanes.
    for y in (oy + 11)..=(oy + 13) {
        reserve_l2!(ox + 23, y);
    }
    for x in (ox + 17)..=(ox + 22) {
        reserve_l2!(x, oy + 11);
    }
    for x in (ox + 18)..=(ox + 22) {
        reserve_l2!(x, oy + 12);
    }
    for x in (ox + 19)..=(ox + 22) {
        reserve_l2!(x, oy + 13);
    }
    for y in (oy + 12)..=(oy + 26) {
        reserve_l2!(ox + 17, y);
    }
    for y in (oy + 13)..=(oy + 26) {
        reserve_l2!(ox + 18, y);
    }
    for y in (oy + 14)..=(oy + 26) {
        reserve_l2!(ox + 19, y);
    }

    // Sprint 109: !use_alu route uses L1 west (oy+15) + L1 north (ox+11) then L2 north
    // (ox+11, oy+1..8) → L2 east (oy+1, ox+12..37) → L2 south (ox+37, oy+2..7).
    // Routes ABOVE the rd_field L2 wall (oy+3..9 at ox+23) and WEST of Sprint 106 walls.
    // L1 WireLeft at oy+15 hops over rd0 column (ox+12) via 3 L2 tiles.
    // Sprint 133: Rerouted east row from oy+2→oy+1 to free L2 oy+2 for bank taps.
    // Pre-reserve all L2 positions to prevent BFS claims.
    for y in (oy + 1)..=(oy + 9) {
        reserve_l2!(ox + 11, y);
    }
    for x in (ox + 12)..=(ox + 37) {
        reserve_l2!(x, oy + 1);
    }
    for y in (oy + 2)..=(oy + 7) {
        reserve_l2!(ox + 37, y);
    }
    // L2 hop over rd0 at (ox+12, oy+15): 3 cells.
    for x in (ox + 11)..=(ox + 13) {
        reserve_l2!(x, oy + 15);
    }

    // Sprint 118: branch_taken_core migrated from L2 to L3.
    // Pre-reserve L3 positions (L2 positions freed for future routes).
    for y in oy..=(oy + 55) {
        reserve_l3!(ox, y); // ox+0 north column
    }
    for x in (ox + 1)..=(ox + 3) {
        reserve_l3!(x, oy); // oy+0 east row
    }
    for x in ox..=(ox + 65) {
        reserve_l3!(x, oy + 56); // oy+56 west row (includes ox+0..3 for turn)
    }

    // Sprint 119: ctrl_a flag route migrated from L2 to L3.
    // Pre-reserve L3 positions (L2 positions freed, removing the #1 congestion blocker).
    reserve_l3!(ox + 22, oy + 9); // entry ViaDown
    reserve_l3!(ox + 22, oy + 10); // WireDown drop
    for x in (ox + 23)..=(ox + 60) {
        reserve_l3!(x, oy + 10); // east wall (was L2 WireRight)
    }
    for y in (oy + 11)..=(oy + 48) {
        reserve_l3!(ox + 60, y); // south column (was L2 WireDown)
    }
    for x in (ox + 22)..=(ox + 59) {
        reserve_l3!(x, oy + 48); // west bus (was L2 WireLeft)
    }

    // Sprint 120: imm8 route migrated from L2 to L3.
    reserve_l3!(ox + 1, oy + 5); // entry ViaDown
    for y in (oy + 6)..=(oy + 55) {
        reserve_l3!(ox + 1, y); // south column (was L2 WireDown)
    }
    for x in (ox + 2)..=(ox + 85) {
        reserve_l3!(x, oy + 55); // east row (was L2 WireRight)
    }
    reserve_l3!(ox + 85, oy + 54); // north WireUp
    reserve_l3!(ox + 85, oy + 53); // north WireUp (exit)

    // Sprint 121: use_imm pure L2 segment migrated from L2 to L3.
    // L2 ViaDown(62,14) stays as relay; only the east bus + south column migrate.
    reserve_l3!(ox + 62, oy + 14); // entry ViaDown
    for x in (ox + 63)..=(ox + 86) {
        reserve_l3!(x, oy + 14); // east bus (was L2 WireRight)
    }
    for y in (oy + 15)..=(oy + 52) {
        reserve_l3!(ox + 86, y); // south column (was L2 WireDown)
    }

    // Sprint 121: Phase C enable route body migrated from L2 to L3.
    // L2 BitSelect+Const+WireDown(62,12..13) stay; only the east bus + south column migrate.
    reserve_l3!(ox + 62, oy + 13); // entry ViaDown (L2 WireDown stays)
    for x in (ox + 63)..=(ox + 93) {
        reserve_l3!(x, oy + 13); // east bus (was L2 WireRight)
    }
    for y in (oy + 14)..=(oy + 45) {
        reserve_l3!(ox + 93, y); // south column (was L2 WireDown)
    }

    // Sprint 116: No L2 pre-reservation needed — all cascade positions are outside
    // the BFS routing area (north of y=13, south of y=26, or east of x=48).
    // Sprint 116 uses place_l2! (no reservation) for all L2 tiles.

    // Sprint 97: cross-bank data route on L2.
    // Use ordered BFS routing with start-point blocking only (targets remain passable
    // until their owning route claims them) so all 8 routes can be embedded without
    // collisions.
    let cross_bank_routes = [
        ((ox + 48, oy + 13), (ox + 26, oy + 24)), // R4 -> A.D4
        ((ox + 48, oy + 16), (ox + 24, oy + 24)), // R5 -> A.D5
        ((ox + 48, oy + 19), (ox + 22, oy + 24)), // R6 -> A.D6
        ((ox + 48, oy + 22), (ox + 20, oy + 24)), // R7 -> A.D7
        ((ox + 28, oy + 13), (ox + 46, oy + 26)), // R0 -> B.D4
        ((ox + 28, oy + 16), (ox + 44, oy + 26)), // R1 -> B.D5
        ((ox + 28, oy + 19), (ox + 42, oy + 26)), // R2 -> B.D6
        ((ox + 28, oy + 22), (ox + 40, oy + 26)), // R3 -> B.D7
    ];
    let route_source_regs = [4usize, 5, 6, 7, 0, 1, 2, 3];

    {
        let src_via = place_l2_route!(ox + 48, oy + 13, TileType::ViaDown);
        push_reg_data!(4, src_via);
        let x49 = place_l2_route!(ox + 49, oy + 13, TileType::WireRight);
        push_reg_data!(4, x49);
        for y in (oy + 14)..=(oy + 18) {
            let idx = place_l2_route!(ox + 49, y, TileType::WireDown);
            push_reg_data!(4, idx);
        }
        let l3_entry = place_l3_route!(ox + 49, oy + 18, TileType::ViaDown);
        push_reg_data!(4, l3_entry);
        for y in (oy + 19)..=(oy + 23) {
            let idx = place_l3_route!(ox + 49, y, TileType::WireDown);
            push_reg_data!(4, idx);
        }
        for x in ((ox + 26)..=(ox + 48)).rev() {
            let idx = place_l3_route!(x, oy + 23, TileType::WireLeft);
            push_reg_data!(4, idx);
        }
        let l2_reentry = place_l2_route!(ox + 26, oy + 23, TileType::ViaUp);
        push_reg_data!(4, l2_reentry);
        let l2_final = place_l2_route!(ox + 26, oy + 24, TileType::WireDown);
        push_reg_data!(4, l2_final);
    }

    {
        let src_via = place_l2_route!(ox + 48, oy + 16, TileType::ViaDown);
        push_reg_data!(5, src_via);
        let west = place_l2_route!(ox + 47, oy + 16, TileType::WireLeft);
        push_reg_data!(5, west);
        for y in (oy + 17)..=(oy + 18) {
            let idx = place_l2_route!(ox + 47, y, TileType::WireDown);
            push_reg_data!(5, idx);
        }
        let l3_entry = place_l3_route!(ox + 47, oy + 18, TileType::ViaDown);
        push_reg_data!(5, l3_entry);
        for x in ((ox + 24)..=(ox + 46)).rev() {
            let idx = place_l3_route!(x, oy + 18, TileType::WireLeft);
            push_reg_data!(5, idx);
        }
        let l2_reentry = place_l2_route!(ox + 24, oy + 18, TileType::ViaUp);
        push_reg_data!(5, l2_reentry);
        for y in (oy + 19)..=(oy + 24) {
            let idx = place_l2_route!(ox + 24, y, TileType::WireDown);
            push_reg_data!(5, idx);
        }
    }

    {
        let src_via = place_l2_route!(ox + 48, oy + 19, TileType::ViaDown);
        push_reg_data!(6, src_via);
        let l3_entry = place_l3_route!(ox + 48, oy + 19, TileType::ViaDown);
        push_reg_data!(6, l3_entry);
        for x in ((ox + 22)..=(ox + 47)).rev() {
            let idx = place_l3_route!(x, oy + 19, TileType::WireLeft);
            push_reg_data!(6, idx);
        }
        let l2_reentry = place_l2_route!(ox + 22, oy + 19, TileType::ViaUp);
        push_reg_data!(6, l2_reentry);
        for y in (oy + 20)..=(oy + 24) {
            let idx = place_l2_route!(ox + 22, y, TileType::WireDown);
            push_reg_data!(6, idx);
        }
    }

    {
        let src_via = place_l2_route!(ox + 48, oy + 22, TileType::ViaDown);
        push_reg_data!(7, src_via);
        let l3_entry = place_l3_route!(ox + 48, oy + 22, TileType::ViaDown);
        push_reg_data!(7, l3_entry);
        let l3_up = place_l3_route!(ox + 48, oy + 21, TileType::WireUp);
        push_reg_data!(7, l3_up);
        for x in ((ox + 20)..=(ox + 47)).rev() {
            let idx = place_l3_route!(x, oy + 21, TileType::WireLeft);
            push_reg_data!(7, idx);
        }
        let l2_reentry = place_l2_route!(ox + 20, oy + 21, TileType::ViaUp);
        push_reg_data!(7, l2_reentry);
        for y in (oy + 22)..=(oy + 24) {
            let idx = place_l2_route!(ox + 20, y, TileType::WireDown);
            push_reg_data!(7, idx);
        }
    }

    let route_order = [5usize, 4, 6, 7];

    let mut l2_start_blocked = vec![false; layer_size];
    let mut l2_endpoint_blocked = vec![false; layer_size];
    for &((sx, sy), _) in &cross_bank_routes {
        l2_start_blocked[sy * grid_width + sx] = true;
    }
    for &(_, (tx, ty)) in &cross_bank_routes {
        l2_endpoint_blocked[ty * grid_width + tx] = true;
    }

    for &route_idx in &route_order {
        let (start, end) = cross_bank_routes[route_idx];
        let start_local = start.1 * grid_width + start.0;
        let end_local = end.1 * grid_width + end.0;
        // Permit the owning route to claim its endpoint; all other unclaimed
        // endpoints stay blocked so routes cannot steal each other's sinks.
        l2_endpoint_blocked[end_local] = false;
        let mut prev = vec![usize::MAX; layer_size];
        let mut seen = vec![false; layer_size];
        let mut q = std::collections::VecDeque::new();
        q.push_back(start_local);
        seen[start_local] = true;

        while let Some(cur) = q.pop_front() {
            if cur == end_local {
                break;
            }
            let x = cur % grid_width;
            let y = cur / grid_width;
            let neighbors = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            for (nx, ny) in neighbors {
                if nx < ox || nx >= ox + 96 || ny < oy || ny >= oy + 64 {
                    continue;
                }
                let n = ny * grid_width + nx;
                if seen[n] {
                    continue;
                }
                if l2_reserved[n] && n != end_local {
                    continue;
                }
                if l2_start_blocked[n] && n != start_local && n != end_local {
                    continue;
                }
                if l2_endpoint_blocked[n] && n != start_local && n != end_local {
                    continue;
                }
                seen[n] = true;
                prev[n] = cur;
                q.push_back(n);
            }
        }

        assert!(
            seen[end_local],
            "Sprint97 failed to route L2 path from ({}, {}) to ({}, {})",
            start.0, start.1, end.0, end.1
        );

        let mut path = Vec::new();
        let mut cur = end_local;
        path.push(cur);
        while cur != start_local {
            cur = prev[cur];
            path.push(cur);
        }
        path.reverse();
        let src_reg = route_source_regs[route_idx];
        let src_via = place_l2_route!(start.0, start.1, TileType::ViaDown);
        push_reg_data!(src_reg, src_via);
        for step in 1..path.len() {
            let prev_idx = path[step - 1];
            let cur_idx = path[step];
            let px = prev_idx % grid_width;
            let py = prev_idx / grid_width;
            let cx = cur_idx % grid_width;
            let cy = cur_idx / grid_width;
            let tt = if cx > px {
                TileType::WireRight
            } else if cx < px {
                TileType::WireLeft
            } else if cy > py {
                TileType::WireDown
            } else {
                TileType::WireUp
            };
            let idx = place_l2_route!(cx, cy, tt);
            push_reg_data!(src_reg, idx);
        }
    }

    // Tree A selector delivery (rd bits: direct boolean MAX/0).
    // rd[0] from extract at (ox+12, oy+6) -> sel0 at (ox+21/25/29/33, oy+24)
    let rd0_via_down_l1 = place_l1_route!(ox + 12, oy + 6, TileType::ViaDown);
    push_pipeline!(rd0_via_down_l1);
    for y in (oy + 7)..=(oy + 23) {
        let idx = place_l1_route!(ox + 12, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    for x in (ox + 13)..=(ox + 33) {
        let idx = place_l1_route!(x, oy + 23, TileType::WireRight);
        push_pipeline!(idx);
    }
    for x in [ox + 21, ox + 25, ox + 29, ox + 33] {
        let drop_idx = place_l1_route!(x, oy + 24, TileType::WireDown);
        let via_up_idx = place!(x, oy + 24, TileType::ViaUp);
        push_pipeline!(drop_idx);
        push_pipeline!(via_up_idx);
    }

    // rd[1] from extract at (ox+9, oy+6) -> sel1 at (ox+22/30, oy+27)
    let rd1_via_down_l1 = place_l1_route!(ox + 9, oy + 6, TileType::ViaDown);
    push_pipeline!(rd1_via_down_l1);
    for y in (oy + 7)..=(oy + 26) {
        let idx = place_l1_route!(ox + 9, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    for x in (ox + 10)..=(ox + 30) {
        let idx = place_l1_route!(x, oy + 26, TileType::WireRight);
        push_pipeline!(idx);
    }
    for x in [ox + 22, ox + 30] {
        let drop_idx = place_l1_route!(x, oy + 27, TileType::WireDown);
        let via_up_idx = place!(x, oy + 27, TileType::ViaUp);
        push_pipeline!(drop_idx);
        push_pipeline!(via_up_idx);
    }

    // rd[2] from extract at (ox+6, oy+6) -> sel2 at (ox+27, oy+30)
    let rd2_via_down_l1 = place_l1_route!(ox + 6, oy + 6, TileType::ViaDown);
    push_pipeline!(rd2_via_down_l1);
    for y in (oy + 7)..=(oy + 29) {
        let idx = place_l1_route!(ox + 6, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    for x in (ox + 7)..=(ox + 27) {
        let idx = place_l1_route!(x, oy + 29, TileType::WireRight);
        push_pipeline!(idx);
    }
    let rd2_drop_l1 = place_l1_route!(ox + 27, oy + 30, TileType::WireDown);
    let rd2_sel_via_up = place!(ox + 27, oy + 30, TileType::ViaUp);
    push_pipeline!(rd2_drop_l1);
    push_pipeline!(rd2_sel_via_up);

    // Tree B selector delivery (packed rs field through Northern Corridor).
    // Sprint 100: shift this corridor east so L1 (ox+2, oy+1) is reserved for
    // PC ingress mux return path.
    let rs_via_down_l1 = place_l1_route!(ox + 2, oy + 5, TileType::ViaDown);
    push_pipeline!(rs_via_down_l1);
    for x in (ox + 3)..=(ox + 5) {
        let idx = place_l1_route!(x, oy + 5, TileType::WireRight);
        push_pipeline!(idx);
    }
    for y in (oy + 1)..=(oy + 4) {
        let idx = place_l1_route!(ox + 5, y, TileType::WireUp);
        push_pipeline!(idx);
    }
    for x in (ox + 6)..=(ox + 12) {
        let idx = place_l1_route!(x, oy + 1, TileType::WireRight);
        push_pipeline!(idx);
    }
    // Bypass x=ox+13 (L0 high-ROM upper-byte Const at y=oy+1) via row oy+0.
    let rs_bypass_up = place_l1_route!(ox + 12, oy, TileType::WireUp);
    let rs_bypass_wr_13 = place_l1_route!(ox + 13, oy, TileType::WireRight);
    let rs_bypass_wr_14 = place_l1_route!(ox + 14, oy, TileType::WireRight);
    let rs_bypass_down = place_l1_route!(ox + 14, oy + 1, TileType::WireDown);
    push_pipeline!(rs_bypass_up);
    push_pipeline!(rs_bypass_wr_13);
    push_pipeline!(rs_bypass_wr_14);
    push_pipeline!(rs_bypass_down);

    for x in (ox + 15)..=(ox + 27) {
        let idx = place_l1_route!(x, oy + 1, TileType::WireRight);
        push_pipeline!(idx);
    }
    // Step around B1.byte2's ascent at L1(28,1) by ducking under it on row oy+3.
    let rs_detour_down1 = place_l1_route!(ox + 27, oy + 2, TileType::WireDown);
    let rs_detour_down2 = place_l1_route!(ox + 27, oy + 3, TileType::WireDown);
    let rs_detour_right1 = place_l1_route!(ox + 28, oy + 3, TileType::WireRight);
    let rs_detour_right2 = place_l1_route!(ox + 29, oy + 3, TileType::WireRight);
    let rs_detour_up1 = place_l1_route!(ox + 29, oy + 2, TileType::WireUp);
    let rs_detour_up2 = place_l1_route!(ox + 29, oy + 1, TileType::WireUp);
    push_pipeline!(rs_detour_down1);
    push_pipeline!(rs_detour_down2);
    push_pipeline!(rs_detour_right1);
    push_pipeline!(rs_detour_right2);
    push_pipeline!(rs_detour_up1);
    push_pipeline!(rs_detour_up2);
    for x in (ox + 30)..=(ox + 55) {
        let idx = place_l1_route!(x, oy + 1, TileType::WireRight);
        push_pipeline!(idx);
    }
    // Sprint 133: Rs column splice at (ox+55, oy+2..3) to allow Bank3 Lane3 tap.
    // L1(ox+55, oy+2) = ViaDown (reads L0 Mux16to1 = Bank3 byte3 output).
    // L1(ox+55, oy+3) = ViaUp (reads L2 rs relay = rs signal).
    // L2 rs relay: detour via ox+56 to bypass bank tap ViaDown at L2(ox+55, oy+2).
    let _b3l3_l1_tap = place_l1_route!(ox + 55, oy + 2, TileType::ViaDown);
    push_pipeline!(_b3l3_l1_tap);
    let _b3l3_l2_tap = place_l2_route!(ox + 55, oy + 2, TileType::ViaDown);
    push_pipeline!(_b3l3_l2_tap);
    // L2 rs relay: ox+55 oy+1 → ox+56 oy+1..3 → ox+55 oy+3
    let _rs_relay_l2_entry = place_l2_route!(ox + 55, oy + 1, TileType::ViaDown);
    let _rs_relay_l2_wr = place_l2_route!(ox + 56, oy + 1, TileType::WireRight);
    let _rs_relay_l2_wd1 = place_l2_route!(ox + 56, oy + 2, TileType::WireDown);
    let _rs_relay_l2_wd2 = place_l2_route!(ox + 56, oy + 3, TileType::WireDown);
    let _rs_relay_l2_wl = place_l2_route!(ox + 55, oy + 3, TileType::WireLeft);
    push_pipeline!(_rs_relay_l2_entry);
    push_pipeline!(_rs_relay_l2_wr);
    push_pipeline!(_rs_relay_l2_wd1);
    push_pipeline!(_rs_relay_l2_wd2);
    push_pipeline!(_rs_relay_l2_wl);
    let _rs_resume = place_l1_route!(ox + 55, oy + 3, TileType::ViaUp);
    push_pipeline!(_rs_resume);
    // Duck under B1.byte3's L1 row at oy+4, then resume the rs column on L1.
    let rs_drop_l2_y4 = place_l2_route!(ox + 55, oy + 4, TileType::WireDown);
    let rs_drop_l2_y5 = place_l2_route!(ox + 55, oy + 5, TileType::WireDown);
    let rs_resume_l1_y5 = place_l1_route!(ox + 55, oy + 5, TileType::ViaUp);
    push_pipeline!(rs_drop_l2_y4);
    push_pipeline!(rs_drop_l2_y5);
    push_pipeline!(rs_resume_l1_y5);
    // Continue rs WireDown column from oy+6.
    for y in (oy + 6)..=(oy + 29) {
        let idx = place_l1_route!(ox + 55, y, TileType::WireDown);
        push_pipeline!(idx);
    }

    // rs sel0 branches -> L1 local BitSelect, L0 selector is ViaUp.
    for x in (ox + 40)..=(ox + 54) {
        let idx = place_l1_route!(x, oy + 23, TileType::WireLeft);
        push_pipeline!(idx);
    }
    for x in [ox + 41, ox + 45, ox + 49, ox + 53] {
        let src_drop = place_l1_route!(x - 1, oy + 24, TileType::WireDown);
        let bit_idx = place_l1_route!(x, oy + 24, TileType::BitSelect);
        let _bit_pos = place_l1_val!(x + 1, oy + 24, TileType::Const, 0);
        let via_up_idx = place!(x, oy + 24, TileType::ViaUp);
        push_pipeline!(src_drop);
        push_pipeline!(bit_idx);
        push_pipeline!(via_up_idx);
    }

    // rs sel1 branches -> L1 local BitSelect, L0 selector is ViaUp.
    for x in (ox + 41)..=(ox + 54) {
        let idx = place_l1_route!(x, oy + 25, TileType::WireLeft);
        push_pipeline!(idx);
    }
    for x in [ox + 42, ox + 50] {
        let src_drop = place_l1_route!(x - 1, oy + 26, TileType::WireDown);
        let src_wd = place_l1_route!(x - 1, oy + 27, TileType::WireDown);
        let bit_idx = place_l1_route!(x, oy + 27, TileType::BitSelect);
        let _bit_pos = place_l1_val!(x + 1, oy + 27, TileType::Const, 1);
        let via_up_idx = place!(x, oy + 27, TileType::ViaUp);
        push_pipeline!(src_drop);
        push_pipeline!(src_wd);
        push_pipeline!(bit_idx);
        push_pipeline!(via_up_idx);
    }

    // rs sel2 branch -> L1 BitSelect(rs2) + Zero inversion, L0 selector is ViaUp.
    for x in (ox + 45)..=(ox + 54) {
        let idx = place_l1_route!(x, oy + 28, TileType::WireLeft);
        push_pipeline!(idx);
    }
    let rs2_src_drop = place_l1_route!(ox + 45, oy + 29, TileType::WireDown);
    let rs2_bit = place_l1_route!(ox + 46, oy + 29, TileType::BitSelect);
    let _rs2_bit_pos = place_l1_val!(ox + 47, oy + 29, TileType::Const, 2);
    let rs2_bit_wd = place_l1_route!(ox + 46, oy + 30, TileType::WireDown);
    let rs2_inv = place_l1_route!(ox + 47, oy + 30, TileType::Zero);
    let rs2_sel_via_up = place!(ox + 47, oy + 30, TileType::ViaUp);
    push_pipeline!(rs2_src_drop);
    push_pipeline!(rs2_bit);
    push_pipeline!(rs2_bit_wd);
    push_pipeline!(rs2_inv);
    push_pipeline!(rs2_sel_via_up);

    // -------------------------------------------------------------------------
    // Sprint 237: High-register operand trees (R8-R15) + top muxes
    // -------------------------------------------------------------------------
    // High Tree A (rd): canonical mapping D0..D3 <- R8..R11, D4..D7 <- R12..R15.
    // Placed at (ox+80, oy+24) — mirrors low tree Y coordinate.
    let high_tree_a = wire_operand_tree_8to1(
        sim,
        grid_width,
        &mut tile_count,
        &mut placed_mask,
        ox + 80,
        tree_row,
    );
    // High Tree B (rs): bank-swapped mapping D0..D3 <- R12..R15, D4..D7 <- R8..R11.
    // Placed at (ox+96, oy+24) — east of High Tree A.
    let high_tree_b = wire_operand_tree_8to1(
        sim,
        grid_width,
        &mut tile_count,
        &mut placed_mask,
        ox + 96,
        tree_row,
    );
    for &idx in &high_tree_a.dirty_indices {
        push_pipeline!(idx);
    }
    for &idx in &high_tree_b.dirty_indices {
        push_pipeline!(idx);
    }

    // High Tree A/B data delivery via Const injection.
    // Physical routing from R8-R15 register columns to tree anchors is infeasible:
    // 4 registers share column ox+88 and 4 share ox+92, and their L1 ViaDown taps
    // block descent routes for other registers on the same column. Instead, we use
    // the proven Const-injection pattern (S234/S235/S236): place Const tiles at each
    // tree data ingress point and inject register values before propagation.
    // The trees physically select the correct value via their selector logic.

    // Tree A: canonical D0=R8..D7=R15. Data ingress at row oy+24.
    // Anchor columns: D0=ox+94, D1=ox+92, D2=ox+90, D3=ox+88,
    //                 D4=ox+86, D5=ox+84, D6=ox+82, D7=ox+80.
    let high_tree_a_data_anchors: [usize; 8] = [
        ox + 94,
        ox + 92,
        ox + 90,
        ox + 88, // D0..D3 = R8..R11
        ox + 86,
        ox + 84,
        ox + 82,
        ox + 80, // D4..D7 = R12..R15
    ];
    let mut high_tree_a_data_const_indices = [0usize; 8];
    for (i, &anchor_col) in high_tree_a_data_anchors.iter().enumerate() {
        let reg = 8 + i; // R8..R15
        let idx = place_val!(anchor_col, oy + 24, TileType::Const, 0);
        high_tree_a_data_const_indices[i] = idx;
        push_reg_data!(reg, idx);
    }

    // Tree B: bank-swapped D0=R12..D3=R15, D4=R8..D7=R11.
    // Anchor columns: D0=ox+110, D1=ox+108, D2=ox+106, D3=ox+104,
    //                 D4=ox+102, D5=ox+100, D6=ox+98, D7=ox+96.
    let high_tree_b_data_anchors: [usize; 8] = [
        ox + 110,
        ox + 108,
        ox + 106,
        ox + 104, // D0..D3 = R12..R15
        ox + 102,
        ox + 100,
        ox + 98,
        ox + 96, // D4..D7 = R8..R11
    ];
    let high_tree_b_reg_map: [usize; 8] = [12, 13, 14, 15, 8, 9, 10, 11];
    let mut high_tree_b_data_const_indices = [0usize; 8];
    for (i, &anchor_col) in high_tree_b_data_anchors.iter().enumerate() {
        let reg = high_tree_b_reg_map[i];
        let idx = place_val!(anchor_col, oy + 24, TileType::Const, 0);
        high_tree_b_data_const_indices[i] = idx;
        push_reg_data!(reg, idx);
    }

    // -------------------------------------------------------------------------
    // Sprint 237: Top Mux A/B — select low tree root vs high tree root.
    // -------------------------------------------------------------------------
    // Top Mux A at (ox+27, oy+35): rd_hi selects high (LEFT) vs low (RIGHT) tree.
    //   UP = (ox+27, oy+34): selector Const (rd_hi injection)
    //   LEFT = (ox+26, oy+35): high Tree A root via L2 horizontal
    //   RIGHT = (ox+28, oy+35): low Tree A root via L1 branch from ALU trunk
    // Top Mux B at (ox+47, oy+35): same structure for rs side.
    //   UP = (ox+47, oy+34): selector Const (rs_hi injection)
    //   LEFT = (ox+46, oy+35): high Tree B root via L3 horizontal
    //   RIGHT = (ox+48, oy+35): low Tree B root via L1 branch from ALU trunk
    //
    // Why oy+35 not oy+34: the ALU Tree A/B distribution trunks occupy L1 at
    // columns ox+27/ox+47 from oy+33..oy+38. Placing top muxes at oy+35 lets
    // us branch off those L1 trunks for low root delivery without any L1 conflict.
    // High root horizontal uses L2 (Tree A) / L3 (Tree B) to avoid both L1
    // trunk conflicts and mutual overlap at oy+35.

    // --- Top Mux A ---
    // Low root: ALU L1 trunk at (ox+27, oy+35) carries low tree A root value.
    // Branch right on L1, ViaUp to L0 for Mux RIGHT input.
    let low_root_a_branch = place_l1_route!(ox + 28, oy + 35, TileType::WireRight);
    push_pipeline!(low_root_a_branch);
    let low_root_a_via = place!(ox + 28, oy + 35, TileType::ViaUp);
    push_pipeline!(low_root_a_via);

    // High root: L2 route from high Tree A root at (ox+87, oy+32).
    // L2 horizontal at oy+35 with L1 detour around ctrl_b south column at (ox+61).
    // L1 at (ox+87, oy+32) is occupied by Sprint 108 opB route (WireRight, cols 48..87).
    // Reroute: descend root on L0 to (87, 33), step right to (88, 33), tap L1/L2 at col 88.
    let hta_root_l0_wd = place!(ox + 87, oy + 33, TileType::WireDown);
    push_pipeline!(hta_root_l0_wd);
    let hta_root_l0_wr = place!(ox + 88, oy + 33, TileType::WireRight);
    push_pipeline!(hta_root_l0_wr);
    let hta_root_l1 = place_l1_route!(ox + 88, oy + 33, TileType::ViaDown);
    push_pipeline!(hta_root_l1);
    let hta_root_l2 = place_l2_route!(ox + 88, oy + 33, TileType::ViaDown);
    push_pipeline!(hta_root_l2);
    for y in (oy + 34)..=(oy + 35) {
        let idx = place_l2_route!(ox + 88, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    // East L2 segment: ox+62..87 at oy+35 (connects to L2 WireDown at ox+88).
    // Split around Sprint 122 RAM L2 column at ox+77 (oy+18..61 WireUp).
    for x in (ox + 78)..=(ox + 87) {
        let idx = place_l2_route!(x, oy + 35, TileType::WireLeft);
        push_pipeline!(idx);
    }
    // L1 hop over RAM L2 column at ox+77.
    let hta_ram_l1_entry = place_l1_route!(ox + 78, oy + 35, TileType::ViaUp);
    push_pipeline!(hta_ram_l1_entry);
    let hta_ram_l1_hop = place_l1_route!(ox + 77, oy + 35, TileType::WireLeft);
    push_pipeline!(hta_ram_l1_hop);
    let hta_ram_l1_exit = place_l1_route!(ox + 76, oy + 35, TileType::WireLeft);
    push_pipeline!(hta_ram_l1_exit);
    let hta_ram_l2_return = place_l2_route!(ox + 76, oy + 35, TileType::ViaDown);
    push_pipeline!(hta_ram_l2_return);
    for x in (ox + 62)..=(ox + 75) {
        let idx = place_l2_route!(x, oy + 35, TileType::WireLeft);
        push_pipeline!(idx);
    }
    // L1 detour around ctrl_b L2 column at ox+61.
    let hta_detour_l1_entry = place_l1_route!(ox + 62, oy + 35, TileType::ViaUp);
    push_pipeline!(hta_detour_l1_entry);
    let hta_detour_l1_mid = place_l1_route!(ox + 61, oy + 35, TileType::WireLeft);
    push_pipeline!(hta_detour_l1_mid);
    let hta_detour_l1_exit = place_l1_route!(ox + 60, oy + 35, TileType::WireLeft);
    push_pipeline!(hta_detour_l1_exit);
    let hta_detour_l2_return = place_l2_route!(ox + 60, oy + 35, TileType::ViaDown);
    push_pipeline!(hta_detour_l2_return);
    // West L2 segment: ox+27..59 at oy+35.
    for x in (ox + 27)..=(ox + 59) {
        let idx = place_l2_route!(x, oy + 35, TileType::WireLeft);
        push_pipeline!(idx);
    }
    let hta_root_l2_end = place_l2_route!(ox + 26, oy + 35, TileType::WireLeft);
    push_pipeline!(hta_root_l2_end);
    let hta_root_l1_up = place_l1_route!(ox + 26, oy + 35, TileType::ViaUp);
    push_pipeline!(hta_root_l1_up);
    let hta_via_up = place!(ox + 26, oy + 35, TileType::ViaUp);
    push_pipeline!(hta_via_up);

    // Top Mux A selector + Mux.
    let top_mux_a_sel_const_idx = place_val!(ox + 27, oy + 34, TileType::Const, 0);
    push_pipeline!(top_mux_a_sel_const_idx);
    let top_mux_a_idx = place!(ox + 27, oy + 35, TileType::Mux);
    push_pipeline!(top_mux_a_idx);

    // --- Top Mux B ---
    // Placed at (ox+107, oy+33) — near the high Tree B root, avoiding mid-zone
    // congestion at oy+35..36 (Sprint 111 R-enable bus, ctrl_a/ctrl_b columns).
    //
    // Mux at (ox+107, oy+33):
    //   UP  = (ox+107, oy+32) = selector Const (rs_hi injection)
    //   LEFT = (ox+106, oy+33) = high root (L0 chain from Tree B root)
    //   RIGHT = (ox+108, oy+33) = low root (L2 east from Sprint 108 opB bus)
    //
    // High root delivery (L0): Tree B root at (ox+103, oy+32) → WireDown → 3× WireRight.
    let htb_high_wd = place!(ox + 103, oy + 33, TileType::WireDown);
    push_pipeline!(htb_high_wd);
    for col in [ox + 104, ox + 105, ox + 106] {
        let idx = place!(col, oy + 33, TileType::WireRight);
        push_pipeline!(idx);
    }

    // Low root delivery (L2 east): Sprint 108 opB L1 bus at (ox+87, oy+32) carries
    // low Tree B root on L1. Tap to L2, route east to ox+108, descend to L0.
    let htb_low_l2_tap = place_l2_route!(ox + 87, oy + 32, TileType::ViaDown);
    push_pipeline!(htb_low_l2_tap);
    for col in (ox + 88)..=(ox + 108) {
        let idx = place_l2_route!(col, oy + 32, TileType::WireRight);
        push_pipeline!(idx);
    }
    let htb_low_l1_via = place_l1_route!(ox + 108, oy + 32, TileType::ViaUp);
    push_pipeline!(htb_low_l1_via);
    let htb_low_l0_via = place!(ox + 108, oy + 32, TileType::ViaUp);
    push_pipeline!(htb_low_l0_via);
    let htb_low_wd = place!(ox + 108, oy + 33, TileType::WireDown);
    push_pipeline!(htb_low_wd);

    // Top Mux B selector + Mux.
    let top_mux_b_sel_const_idx = place_val!(ox + 107, oy + 32, TileType::Const, 0);
    push_pipeline!(top_mux_b_sel_const_idx);
    let top_mux_b_idx = place!(ox + 107, oy + 33, TileType::Mux);
    push_pipeline!(top_mux_b_idx);

    // Sprint 237 Option 2-plus: build dirty seed sets for Const-injection propagation.
    // mark_dirty on a Const tile itself is a no-op (eval sees current==current).
    // Instead, seed the downstream tiles that READ from those Consts.
    let high_tree_a_dirty_indices = high_tree_a.dirty_indices.clone();
    let high_tree_b_dirty_indices = high_tree_b.dirty_indices.clone();

    // Top Mux A: low root delivery + high root delivery + mux tile.
    let top_mux_a_dirty_indices = vec![
        low_root_a_branch,
        low_root_a_via, // low root
        hta_root_l0_wd,
        hta_root_l0_wr,
        hta_via_up,    // high root endpoints
        top_mux_a_idx, // mux itself
    ];
    // Top Mux B: low root delivery + high root delivery + mux tile.
    let top_mux_b_dirty_indices = vec![
        htb_low_l2_tap,
        htb_low_l1_via,
        htb_low_l0_via,
        htb_low_wd,    // low root
        htb_high_wd,   // high root start
        top_mux_b_idx, // mux itself
    ];

    // -------------------------------------------------------------------------
    // Sprint 99: physical ALU operand delivery with per-op override muxes.
    // Order: [AddCarry, Sub, And, Or, Xor, Not, Shl, Shr]
    // -------------------------------------------------------------------------
    let alu_row = oy + 39;
    let alu_base_col = ox + 18;

    // L1 distribution: Tree A root -> row oy+38.
    let tree_a_tap_l1 = place_l1_route!(ox + 27, oy + 32, TileType::ViaDown);
    push_pipeline!(tree_a_tap_l1);
    for y in (oy + 33)..=(oy + 38) {
        let idx = place_l1_route!(ox + 27, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    // Sprint 239: capture terminal trunk tile (last WireDown before horizontal).
    let alu_a_trunk_terminal_idx = (oy + 38) * grid_width + (ox + 27) + layer_size;
    let mut alu_a_downstream_dirty = Vec::new();
    for x in (ox + 17)..=(ox + 26) {
        let idx = place_l1_route!(x, oy + 38, TileType::WireLeft);
        push_pipeline!(idx);
        alu_a_downstream_dirty.push(idx); // Sprint 239: west horizontal
    }
    for x in (ox + 28)..=(ox + 36) {
        let idx = place_l1_route!(x, oy + 38, TileType::WireRight);
        push_pipeline!(idx);
        alu_a_downstream_dirty.push(idx); // Sprint 239: east horizontal
    }
    // Deterministic detour around L1 flag spine at col ox+37.
    let tree_a_detour_entry_l2 = place_l2_route!(ox + 36, oy + 38, TileType::ViaDown);
    let tree_a_detour_step_l2 = place_l2_route!(ox + 37, oy + 38, TileType::WireRight);
    let tree_a_detour_exit_l2 = place_l2_route!(ox + 38, oy + 38, TileType::WireRight);
    let tree_a_detour_exit_l1 = place_l1_route!(ox + 38, oy + 38, TileType::ViaUp);
    push_pipeline!(tree_a_detour_entry_l2);
    push_pipeline!(tree_a_detour_step_l2);
    push_pipeline!(tree_a_detour_exit_l2);
    push_pipeline!(tree_a_detour_exit_l1);
    // Sprint 239: detour tiles in downstream dirty set
    alu_a_downstream_dirty.push(tree_a_detour_entry_l2);
    alu_a_downstream_dirty.push(tree_a_detour_step_l2);
    alu_a_downstream_dirty.push(tree_a_detour_exit_l2);
    alu_a_downstream_dirty.push(tree_a_detour_exit_l1);
    for x in (ox + 39)..=(ox + 46) {
        let idx = place_l1_route!(x, oy + 38, TileType::WireRight);
        push_pipeline!(idx);
        alu_a_downstream_dirty.push(idx); // Sprint 239: east horizontal continued
    }

    // L1 distribution: Tree B root -> row oy+37.
    let tree_b_tap_l1 = place_l1_route!(ox + 47, oy + 32, TileType::ViaDown);
    push_pipeline!(tree_b_tap_l1);
    for y in (oy + 33)..=(oy + 37) {
        let idx = place_l1_route!(ox + 47, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    // Sprint 239: capture terminal trunk tile (last WireDown before horizontal).
    let alu_b_trunk_terminal_idx = (oy + 37) * grid_width + (ox + 47) + layer_size;
    let mut alu_b_downstream_dirty = Vec::new();
    let tree_b_right_l1 = place_l1_route!(ox + 48, oy + 37, TileType::WireRight);
    push_pipeline!(tree_b_right_l1);
    alu_b_downstream_dirty.push(tree_b_right_l1); // Sprint 239
    for x in (ox + 38)..=(ox + 46) {
        let idx = place_l1_route!(x, oy + 37, TileType::WireLeft);
        push_pipeline!(idx);
        alu_b_downstream_dirty.push(idx); // Sprint 239
    }
    // Deterministic detour around L1 flag spine at col ox+37.
    let tree_b_detour_entry_l2 = place_l2_route!(ox + 38, oy + 37, TileType::ViaDown);
    let tree_b_detour_step_l2 = place_l2_route!(ox + 37, oy + 37, TileType::WireLeft);
    let tree_b_detour_exit_l2 = place_l2_route!(ox + 36, oy + 37, TileType::WireLeft);
    let tree_b_detour_exit_l1 = place_l1_route!(ox + 36, oy + 37, TileType::ViaUp);
    push_pipeline!(tree_b_detour_entry_l2);
    push_pipeline!(tree_b_detour_step_l2);
    push_pipeline!(tree_b_detour_exit_l2);
    push_pipeline!(tree_b_detour_exit_l1);
    // Sprint 239: detour tiles in downstream dirty set
    alu_b_downstream_dirty.push(tree_b_detour_entry_l2);
    alu_b_downstream_dirty.push(tree_b_detour_step_l2);
    alu_b_downstream_dirty.push(tree_b_detour_exit_l2);
    alu_b_downstream_dirty.push(tree_b_detour_exit_l1);
    // Avoid L1 ownership collision with Tree A vertical trunk at (ox+27, oy+37):
    // keep the left-moving chain on both sides and bridge across the blocked cell on L2.
    for x in (ox + 28)..=(ox + 35) {
        let idx = place_l1_route!(x, oy + 37, TileType::WireLeft);
        push_pipeline!(idx);
        alu_b_downstream_dirty.push(idx); // Sprint 239
    }
    let tree_b_bridge_entry_l2 = place_l2_route!(ox + 29, oy + 37, TileType::ViaDown);
    let tree_b_bridge_mid1_l2 = place_l2_route!(ox + 28, oy + 37, TileType::WireLeft);
    let tree_b_bridge_mid2_l2 = place_l2_route!(ox + 27, oy + 37, TileType::WireLeft);
    let tree_b_bridge_exit_l2 = place_l2_route!(ox + 26, oy + 37, TileType::WireLeft);
    let tree_b_bridge_exit_l1 = place_l1_route!(ox + 26, oy + 37, TileType::ViaUp);
    push_pipeline!(tree_b_bridge_entry_l2);
    push_pipeline!(tree_b_bridge_mid1_l2);
    push_pipeline!(tree_b_bridge_mid2_l2);
    push_pipeline!(tree_b_bridge_exit_l2);
    push_pipeline!(tree_b_bridge_exit_l1);
    // Sprint 239: bridge tiles in downstream dirty set
    alu_b_downstream_dirty.push(tree_b_bridge_entry_l2);
    alu_b_downstream_dirty.push(tree_b_bridge_mid1_l2);
    alu_b_downstream_dirty.push(tree_b_bridge_mid2_l2);
    alu_b_downstream_dirty.push(tree_b_bridge_exit_l2);
    alu_b_downstream_dirty.push(tree_b_bridge_exit_l1);
    for x in (ox + 20)..=(ox + 25) {
        let idx = place_l1_route!(x, oy + 37, TileType::WireLeft);
        push_pipeline!(idx);
        alu_b_downstream_dirty.push(idx); // Sprint 239
    }

    let mut _alu_tile_indices = [0usize; 8];
    let mut alu_l_override_indices = [0usize; 8];
    let mut alu_r_override_indices = [0usize; 8];
    let mut alu_l_enable_indices = [0usize; 8];
    let mut alu_l_mux_indices = [0usize; 8];
    let mut alu_r_mux_indices = [0usize; 8];

    let alu_types = [
        TileType::Add, // Sprint 128: AddCarry → Add + CarryDetect
        TileType::Sub, // Sprint 128: SubBorrow → Sub + CarryDetect
        TileType::And,
        TileType::Or,
        TileType::Xor,
        TileType::Not,
        TileType::Shl,
        TileType::Shr,
    ];
    for i in 0..8usize {
        let x = alu_base_col + i * 4;
        _alu_tile_indices[i] = place!(x, alu_row, alu_types[i]);

        if i == 0 {
            // AddCarry special-case keeps Const(0) above ALU for carry-in.
            alu_l_override_indices[i] = place_val!(x - 3, alu_row - 1, TileType::Const, 0);
            alu_l_enable_indices[i] = place_val!(x - 2, alu_row - 2, TileType::Const, 0);
            alu_l_mux_indices[i] = place!(x - 2, alu_row - 1, TileType::Mux);
            let l_tree_via = place!(x - 1, alu_row - 1, TileType::ViaUp);
            let l_wd = place!(x - 2, alu_row, TileType::WireDown);
            let l_wr = place!(x - 1, alu_row, TileType::WireRight);
            let _carry_zero = place_val!(x, alu_row - 1, TileType::Const, 0);
            push_pipeline!(alu_l_mux_indices[i]);
            push_pipeline!(l_tree_via);
            push_pipeline!(l_wd);
            push_pipeline!(l_wr);

            alu_r_override_indices[i] = place!(x, alu_row - 2, TileType::WireDown);
            // Sprint 111: R enable is now physical ViaUp reading L1 use_imm bus.
            let _r_enable = place!(x + 1, alu_row - 3, TileType::ViaUp);
            push_pipeline!(_r_enable);
            alu_r_mux_indices[i] = place!(x + 1, alu_row - 2, TileType::Mux);
            let r_tree_via = place!(x + 2, alu_row - 2, TileType::ViaUp);
            let r_wd_1 = place!(x + 1, alu_row - 1, TileType::WireDown);
            let r_wd_2 = place!(x + 1, alu_row, TileType::WireDown);
            push_pipeline!(alu_r_mux_indices[i]);
            push_pipeline!(r_tree_via);
            push_pipeline!(r_wd_1);
            push_pipeline!(r_wd_2);
        } else {
            alu_l_override_indices[i] = place_val!(x - 2, alu_row - 1, TileType::Const, 0);
            // Sprint 113: i=1 (Sub) L enable is now physical ViaUp reading is_neg LUT.
            if i == 1 {
                alu_l_enable_indices[i] = place!(x - 1, alu_row - 2, TileType::ViaUp);
            } else {
                alu_l_enable_indices[i] = place_val!(x - 1, alu_row - 2, TileType::Const, 0);
            }
            alu_l_mux_indices[i] = place!(x - 1, alu_row - 1, TileType::Mux);
            let l_tree_via = place!(x, alu_row - 1, TileType::ViaUp);
            let l_wd = place!(x - 1, alu_row, TileType::WireDown);
            push_pipeline!(alu_l_mux_indices[i]);
            push_pipeline!(l_tree_via);
            push_pipeline!(l_wd);

            alu_r_override_indices[i] = place!(x, alu_row - 2, TileType::WireDown);
            // Sprint 111: R enable at (x+1, alu_row-3) is now physical.
            // i=2 and i=7 need special relay tiles (Tree A/B trunk blocks L1);
            // handled in Sprint 111 route section below. Standard cases use ViaUp.
            if i == 2 {
                // Tree A trunk at ox+27 blocks L1 ViaUp.  Place WireLeft to read
                // from ViaUp relay at (ox+28, oy+36).
                let _r_enable = place!(x + 1, alu_row - 3, TileType::WireLeft);
                push_pipeline!(_r_enable);
            } else if i == 7 {
                // Tree B trunk at ox+47 blocks L1 ViaUp.  Place WireRight to read
                // from ViaUp relay at (ox+46, oy+36).
                let _r_enable = place!(x + 1, alu_row - 3, TileType::WireRight);
                push_pipeline!(_r_enable);
            } else {
                let _r_enable = place!(x + 1, alu_row - 3, TileType::ViaUp);
                push_pipeline!(_r_enable);
            }
            alu_r_mux_indices[i] = place!(x + 1, alu_row - 2, TileType::Mux);
            let r_tree_via = place!(x + 2, alu_row - 2, TileType::ViaUp);
            let r_wd_1 = place!(x + 1, alu_row - 1, TileType::WireDown);
            let r_wd_2 = place!(x + 1, alu_row, TileType::WireDown);
            push_pipeline!(alu_r_mux_indices[i]);
            push_pipeline!(r_tree_via);
            push_pipeline!(r_wd_1);
            push_pipeline!(r_wd_2);
        }
    }

    // Sprint 114: Local arrays still needed for ALU loop placement but not returned.
    let _ = alu_r_override_indices;
    let _ = alu_l_mux_indices;
    let _ = alu_r_mux_indices;

    // Sprint 243: Capture R Mux output WireDown indices — the tile at (x+1, alu_row)
    // feeds each ALU column's RIGHT input. These are injection targets for wide-immediate.
    let mut alu_r_mux_output_indices = [0usize; 8];
    for i in 0..8usize {
        let x = alu_base_col + i * 4;
        alu_r_mux_output_indices[i] = alu_row * grid_width + (x + 1);
    }

    // -------------------------------------------------------------------------
    // Sprint 123 Phase B: Physical SHR carry = (A >> (B-1)) & 1.
    // Sub(B,1) at L0(ox+48, alu_row). A and B-1 relayed through L1 to avoid
    // collision with Sprint 106 ALU WB selector tiles at oy+40 (Const(MAX) at
    // ox+45, WireDown at ox+46). BitSelect at L0(ox+48, alu_row+1).
    // -------------------------------------------------------------------------
    let s123_sub = place!(ox + 48, alu_row, TileType::Sub); // Sprint 128: SubBorrow → Sub
    let _s123_one = place_val!(ox + 49, alu_row, TileType::Const, 1);
    // L1 relay: tap A from L0(ox+45,alu_row) and route east to (ox+47,alu_row+1).
    let _ = place_l1_route!(ox + 45, alu_row, TileType::ViaDown);
    let _ = place_l1_route!(ox + 46, alu_row, TileType::WireRight);
    let _ = place_l1_route!(ox + 47, alu_row, TileType::WireRight);
    let _ = place_l1_route!(ox + 47, alu_row + 1, TileType::WireDown);
    // L1 relay: tap B-1 from L0(ox+48,alu_row)=Sub and route to (ox+49,alu_row+1).
    let _ = place_l1_route!(ox + 48, alu_row, TileType::ViaDown);
    let _ = place_l1_route!(ox + 48, alu_row + 1, TileType::WireDown);
    // Sprint 131: Mux selects bit position — LEFT=B-1 (SHR), RIGHT=8-B (SHL).
    // UP=ctrl_a[0] from L2 delivery at L1(ox+49,alu_row). ctrl_a[0]=1→SHR, 0→SHL.
    let _ = place_l1_route!(ox + 49, alu_row + 1, TileType::Mux);
    // L0 computation: ViaUp(A), BitSelect(A, pos), ViaUp(pos from Mux).
    let _ = place!(ox + 47, alu_row + 1, TileType::ViaUp);
    let _ = place!(ox + 48, alu_row + 1, TileType::BitSelect);
    let _ = place!(ox + 49, alu_row + 1, TileType::ViaUp);

    // Sprint 123 Phase B: SHR carry south column from BitSelect to oy+53.
    // Skip oy+48 — occupied by WB ALU east bus (WireRight, line 1878). L1 hop instead.
    for y in (alu_row + 2)..=(oy + 47) {
        let _ = place!(ox + 48, y, TileType::WireDown);
    }
    // L1 hop over WB ALU bus at oy+48: ViaDown(47)→WireDown(48)→WireDown(49), ViaUp(49).
    let _ = place_l1_route!(ox + 48, oy + 47, TileType::ViaDown);
    let _ = place_l1_route!(ox + 48, oy + 48, TileType::WireDown);
    let _ = place_l1_route!(ox + 48, oy + 49, TileType::WireDown);
    let _ = place!(ox + 48, oy + 49, TileType::ViaUp);
    for y in (oy + 50)..=(oy + 53) {
        let _ = place!(ox + 48, y, TileType::WireDown);
    }
    // Via descent L0→L1→L2 at (ox+48, oy+53).
    let _ = place_l1_route!(ox + 48, oy + 53, TileType::ViaDown);
    let _ = place_l2_route!(ox + 48, oy + 53, TileType::ViaDown);
    // L2 west bus at oy+53 from ox+21 to ox+47.
    for x in (ox + 21)..=(ox + 47) {
        let _ = place_l2_route!(x, oy + 53, TileType::WireLeft);
    }
    // L2 north at ox+21 from oy+52 to oy+51 (oy+50 replaced in flag section below).
    let _ = place_l2_route!(ox + 21, oy + 52, TileType::WireUp);
    let _ = place_l2_route!(ox + 21, oy + 51, TileType::WireUp);

    // -------------------------------------------------------------------------
    // Sprint 106: physical ALU WB selector — extract ctrl_a[2:0] and distribute
    // inverted bits to 7 Mux tree selector positions via L2 south + L1 east routing.
    // Mux-inverter pattern: Const(0)→Mux←Const(MAX), UP=raw bit from ViaUp.
    //   UP!=0 → LEFT=0 (bit=1→0), UP==0 → RIGHT=MAX (bit=0→MAX).
    // -------------------------------------------------------------------------
    let mut wb_alu_dirty_indices = Vec::new();

    // Step 1: BitSelect extraction from ctrl_a at L0 (ox+22, oy+11..13).
    // ctrl_a descends from (ox+22, oy+10) WireDown placed in Sprint 101.
    let s106_wd_11 = place!(ox + 22, oy + 11, TileType::WireDown);
    let s106_bit0 = place!(ox + 23, oy + 11, TileType::BitSelect);
    let _s106_bit0_pos = place_val!(ox + 24, oy + 11, TileType::Const, 0);
    push_pipeline!(s106_wd_11);
    push_pipeline!(s106_bit0);

    let s106_wd_12 = place!(ox + 22, oy + 12, TileType::WireDown);
    let s106_bit1 = place!(ox + 23, oy + 12, TileType::BitSelect);
    let _s106_bit1_pos = place_val!(ox + 24, oy + 12, TileType::Const, 1);
    push_pipeline!(s106_wd_12);
    push_pipeline!(s106_bit1);

    let s106_wd_13 = place!(ox + 22, oy + 13, TileType::WireDown);
    let s106_bit2 = place!(ox + 23, oy + 13, TileType::BitSelect);
    let _s106_bit2_pos = place_val!(ox + 24, oy + 13, TileType::Const, 2);
    push_pipeline!(s106_wd_13);
    push_pipeline!(s106_bit2);

    // Step 2: L1+L2 ViaDown taps at BitSelect outputs.
    let s106_b0_l1 = place_l1_route!(ox + 23, oy + 11, TileType::ViaDown);
    let s106_b0_l2 = place_l2!(ox + 23, oy + 11, TileType::ViaDown);
    push_pipeline!(s106_b0_l1);
    push_pipeline!(s106_b0_l2);

    let s106_b1_l1 = place_l1_route!(ox + 23, oy + 12, TileType::ViaDown);
    let s106_b1_l2 = place_l2!(ox + 23, oy + 12, TileType::ViaDown);
    push_pipeline!(s106_b1_l1);
    push_pipeline!(s106_b1_l2);

    let s106_b2_l1 = place_l1_route!(ox + 23, oy + 13, TileType::ViaDown);
    let s106_b2_l2 = place_l2!(ox + 23, oy + 13, TileType::ViaDown);
    push_pipeline!(s106_b2_l1);
    push_pipeline!(s106_b2_l2);

    // Step 3: L2 south columns + L1 east buses with flag-spine detours.
    // Column assignment: bit0→ox+17, bit1→ox+18, bit2→ox+19 (furthest west first
    // so east buses on later bits don't cross earlier south columns on L2).

    // Bit 0: L2 west to col ox+17, south to oy+39, then L1 east to 4 sel0 targets.
    for x in (ox + 17)..=(ox + 22) {
        let idx = place_l2!(x, oy + 11, TileType::WireLeft);
        push_pipeline!(idx);
    }
    for y in (oy + 12)..=(oy + 26) {
        let idx = place_l2!(ox + 17, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    for y in (oy + 12)..=(oy + 39) {
        if y <= oy + 26 {
            continue;
        }
        let idx = place_l2_route!(ox + 17, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    // Transition L2→L1 and east bus on L1 row oy+39.
    let s106_b0_l1_entry = place_l1_route!(ox + 17, oy + 39, TileType::ViaUp);
    push_pipeline!(s106_b0_l1_entry);
    for x in (ox + 18)..=(ox + 36) {
        let idx = place_l1_route!(x, oy + 39, TileType::WireRight);
        push_pipeline!(idx);
    }
    // L2 detour around flag spine at col ox+37.
    // Entry at ox+35 (not ox+36) to avoid dual-via conflict: L0 ViaUp + L2 ViaDown
    // cannot both read the same L1 tile. ox+36 has an L0 ViaUp drop, so detour enters
    // one col earlier.
    let s106_b0_det_entry = place_l2_route!(ox + 35, oy + 39, TileType::ViaDown);
    let s106_b0_det_wr36 = place_l2_route!(ox + 36, oy + 39, TileType::WireRight);
    let s106_b0_det_mid = place_l2_route!(ox + 37, oy + 39, TileType::WireRight);
    let s106_b0_det_exit = place_l2_route!(ox + 38, oy + 39, TileType::WireRight);
    let s106_b0_det_ret = place_l1_route!(ox + 38, oy + 39, TileType::ViaUp);
    push_pipeline!(s106_b0_det_entry);
    push_pipeline!(s106_b0_det_wr36);
    push_pipeline!(s106_b0_det_mid);
    push_pipeline!(s106_b0_det_exit);
    push_pipeline!(s106_b0_det_ret);
    for x in (ox + 39)..=(ox + 44) {
        let idx = place_l1_route!(x, oy + 39, TileType::WireRight);
        push_pipeline!(idx);
    }
    // Drop bit 0 from L1 to L0 at 4 sel0 positions.
    for &x in &[ox + 20, ox + 28, ox + 36, ox + 44] {
        let l0v = place!(x, oy + 39, TileType::ViaUp);
        push_pipeline!(l0v);
    }

    // -------------------------------------------------------------------------
    // Sprint 123 Phase A: Extend sel0 (ctrl_a[0]) L2 south column to oy+46,
    // then L3 hop east to ox+23, L2 descent to (ox+23, oy+49).
    // Feeds And gate at L2(ox+22, oy+49) to compute is_shr = And(is_shift, bit0).
    // -------------------------------------------------------------------------
    for y in (oy + 40)..=(oy + 46) {
        let _ = place_l2_route!(ox + 17, y, TileType::WireDown);
    }
    // L3 hop east from ox+17 to ox+23 at oy+46.
    let _ = place_l3_route!(ox + 17, oy + 46, TileType::ViaDown);
    for x in (ox + 18)..=(ox + 23) {
        let _ = place_l3_route!(x, oy + 46, TileType::WireRight);
    }
    let _ = place_l2_route!(ox + 23, oy + 46, TileType::ViaUp);
    // L2 south from (23,46) to (23,49) — delivers sel0 to And gate.
    for y in (oy + 47)..=(oy + 49) {
        let _ = place_l2_route!(ox + 23, y, TileType::WireDown);
    }

    // Bit 1: L2 west to col ox+18, south to oy+42, then L1 east to 2 sel1 targets.
    for x in (ox + 18)..=(ox + 22) {
        let idx = place_l2!(x, oy + 12, TileType::WireLeft);
        push_pipeline!(idx);
    }
    for y in (oy + 13)..=(oy + 26) {
        let idx = place_l2!(ox + 18, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    for y in (oy + 13)..=(oy + 42) {
        if y <= oy + 26 {
            continue;
        }
        let idx = place_l2_route!(ox + 18, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    let s106_b1_l1_entry = place_l1_route!(ox + 18, oy + 42, TileType::ViaUp);
    push_pipeline!(s106_b1_l1_entry);
    for x in (ox + 19)..=(ox + 36) {
        let idx = place_l1_route!(x, oy + 42, TileType::WireRight);
        push_pipeline!(idx);
    }
    // L2 detour around flag spine.
    let s106_b1_det_entry = place_l2_route!(ox + 36, oy + 42, TileType::ViaDown);
    let s106_b1_det_mid = place_l2_route!(ox + 37, oy + 42, TileType::WireRight);
    let s106_b1_det_exit = place_l2_route!(ox + 38, oy + 42, TileType::WireRight);
    let s106_b1_det_ret = place_l1_route!(ox + 38, oy + 42, TileType::ViaUp);
    push_pipeline!(s106_b1_det_entry);
    push_pipeline!(s106_b1_det_mid);
    push_pipeline!(s106_b1_det_exit);
    push_pipeline!(s106_b1_det_ret);
    for x in (ox + 39)..=(ox + 40) {
        let idx = place_l1_route!(x, oy + 42, TileType::WireRight);
        push_pipeline!(idx);
    }
    for &x in &[ox + 24, ox + 40] {
        let l0v = place!(x, oy + 42, TileType::ViaUp);
        push_pipeline!(l0v);
    }

    // Bit 2: L2 west to col ox+19, south to oy+45, then L1 east to 1 sel2 target.
    // No flag-spine crossing needed (target at ox+32 is west of ox+37).
    for x in (ox + 19)..=(ox + 22) {
        let idx = place_l2!(x, oy + 13, TileType::WireLeft);
        push_pipeline!(idx);
    }
    for y in (oy + 14)..=(oy + 26) {
        let idx = place_l2!(ox + 19, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    for y in (oy + 14)..=(oy + 45) {
        if y <= oy + 26 {
            continue;
        }
        let idx = place_l2_route!(ox + 19, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    let s106_b2_l1_entry = place_l1_route!(ox + 19, oy + 45, TileType::ViaUp);
    push_pipeline!(s106_b2_l1_entry);
    for x in (ox + 20)..=(ox + 32) {
        let idx = place_l1_route!(x, oy + 45, TileType::WireRight);
        push_pipeline!(idx);
    }
    {
        let l0v = place!(ox + 32, oy + 45, TileType::ViaUp);
        push_pipeline!(l0v);
    }

    // Step 4: Mux-inverter tiles at selector positions.
    // LEFT neighbor = guard Const(0) (unchanged), RIGHT = new Const(MAX).
    // Sel0: 4 sites at oy+40 (bit 0 inverted).
    for &x in &[ox + 20, ox + 28, ox + 36, ox + 44] {
        let mux = place!(x, oy + 40, TileType::Mux);
        let _max = place_val!(x + 1, oy + 40, TileType::Const, u64::MAX);
        wb_alu_dirty_indices.push(mux);
        push_pipeline!(mux);
    }
    // Sel1: 2 sites at oy+43 (bit 1 inverted).
    for &x in &[ox + 24, ox + 40] {
        let mux = place!(x, oy + 43, TileType::Mux);
        let _max = place_val!(x + 1, oy + 43, TileType::Const, u64::MAX);
        wb_alu_dirty_indices.push(mux);
        push_pipeline!(mux);
    }
    // Sel2: 1 site at oy+46 (bit 2 inverted).
    {
        let mux = place!(ox + 32, oy + 46, TileType::Mux);
        let _max = place_val!(ox + 33, oy + 46, TileType::Const, u64::MAX);
        wb_alu_dirty_indices.push(mux);
        push_pipeline!(mux);
    }

    // ALU outputs -> leaf layer data anchors.
    for x in [
        ox + 18,
        ox + 22,
        ox + 26,
        ox + 30,
        ox + 34,
        ox + 38,
        ox + 42,
        ox + 46,
    ] {
        let idx = place!(x, oy + 40, TileType::WireDown);
        wb_alu_dirty_indices.push(idx);
        let idx = place!(x, oy + 41, TileType::WireDown);
        wb_alu_dirty_indices.push(idx);
    }
    for x in [ox + 19, ox + 27, ox + 35, ox + 43] {
        let idx = place!(x, oy + 41, TileType::WireRight);
        wb_alu_dirty_indices.push(idx);
    }
    for x in [ox + 21, ox + 29, ox + 37, ox + 45] {
        let idx = place!(x, oy + 41, TileType::WireLeft);
        wb_alu_dirty_indices.push(idx);
    }

    // Leaf muxes (D0..D7 paired): [0/1], [2/3], [4/5], [6/7]
    for x in [ox + 20, ox + 28, ox + 36, ox + 44] {
        let idx = place!(x, oy + 41, TileType::Mux);
        wb_alu_dirty_indices.push(idx);
    }

    // Mid-layer descent from leaf outputs.
    for x in [ox + 20, ox + 28, ox + 36, ox + 44] {
        for y in (oy + 42)..=(oy + 44) {
            let idx = place!(x, y, TileType::WireDown);
            wb_alu_dirty_indices.push(idx);
        }
    }

    // Mid mux A (0..3): left from x20, right from x28.
    for x in [ox + 21, ox + 22, ox + 23] {
        let idx = place!(x, oy + 44, TileType::WireRight);
        wb_alu_dirty_indices.push(idx);
    }
    for x in [ox + 27, ox + 26, ox + 25] {
        let idx = place!(x, oy + 44, TileType::WireLeft);
        wb_alu_dirty_indices.push(idx);
    }
    let idx = place!(ox + 24, oy + 44, TileType::Mux);
    wb_alu_dirty_indices.push(idx);

    // Mid mux B (4..7): left from x36, right from x44.
    for x in [ox + 37, ox + 38, ox + 39] {
        let idx = place!(x, oy + 44, TileType::WireRight);
        wb_alu_dirty_indices.push(idx);
    }
    for x in [ox + 43, ox + 42, ox + 41] {
        let idx = place!(x, oy + 44, TileType::WireLeft);
        wb_alu_dirty_indices.push(idx);
    }
    let idx = place!(ox + 40, oy + 44, TileType::Mux);
    wb_alu_dirty_indices.push(idx);

    // Root layer descent.
    for x in [ox + 24, ox + 40] {
        for y in (oy + 45)..=(oy + 47) {
            let idx = place!(x, y, TileType::WireDown);
            wb_alu_dirty_indices.push(idx);
        }
    }

    // Root mux (0..7): left from x24, right from x40.
    for x in (ox + 25)..=(ox + 31) {
        let idx = place!(x, oy + 47, TileType::WireRight);
        wb_alu_dirty_indices.push(idx);
    }
    for x in (ox + 33)..=(ox + 39) {
        let idx = place!(x, oy + 47, TileType::WireLeft);
        wb_alu_dirty_indices.push(idx);
    }
    let wb_alu_root_idx = place!(ox + 32, oy + 47, TileType::Mux);
    wb_alu_dirty_indices.push(wb_alu_root_idx);

    // Root -> mask stage -> long ingress route to wb_data mux RIGHT input (x39, y10).
    let idx = place!(ox + 32, oy + 48, TileType::WireDown);
    wb_alu_dirty_indices.push(idx);
    for x in (ox + 33)..=(ox + 57) {
        let idx = place!(x, oy + 48, TileType::WireRight);
        wb_alu_dirty_indices.push(idx);
    }
    let wb_mask_and_idx = place!(ox + 58, oy + 48, TileType::And);
    // Sprint 129: Remove 8-bit mask. Const(u64::MAX) makes And a pass-through.
    // Previously Const(0xFF) truncated writeback to 8 bits.
    let _wb_mask_const_idx = place_val!(ox + 59, oy + 48, TileType::Const, u64::MAX);
    wb_alu_dirty_indices.push(wb_mask_and_idx);

    for y in (oy + 10)..=(oy + 47) {
        let idx = place!(ox + 58, y, TileType::WireUp);
        wb_alu_dirty_indices.push(idx);
    }
    for x in (ox + 39)..=(ox + 57) {
        let idx = place!(x, oy + 10, TileType::WireLeft);
        wb_alu_dirty_indices.push(idx);
    }

    reg_wb_dirty_indices.extend(wb_alu_dirty_indices.iter().copied());

    // -------------------------------------------------------------------------
    // Sprint 94: flag commit fabric
    // -------------------------------------------------------------------------
    // Flag C moved from (ox+22, oy+50) -> (ox+24, oy+50) to make room for mux chain.
    let flag_z_idx = place_val!(ox + 20, oy + 50, TileType::Register8, 0);
    let flag_c_idx = place_val!(ox + 24, oy + 50, TileType::Register8, 0);

    // Sprint 101/119: physical packed flag WE source from ctrl_a bits [5:4].
    // Source value presented to existing flag WE split path:
    //   flag_we_mask = ctrl_a >> 4
    // This preserves bit semantics:
    //   bit0 -> update_flags_z, bit1 -> update_flags_c.
    //
    // Sprint 119: Route migrated from L2 to L3. Entry/exit via chains now go
    // L1→L2 ViaDown (relay)→L3 ViaDown→(L3 route)→L2 ViaUp→L1 ViaUp.
    // Frees 114 L2 positions, eliminating the #1 congestion blocker (oy+10 wall).

    // Entry: L2 ViaDown stays as relay (reads L1), L3 ViaDown reads L2 relay.
    let ctrl_a_tap_l2 = place_l2_route!(ox + 22, oy + 9, TileType::ViaDown);
    push_pipeline!(ctrl_a_tap_l2);
    // L3 entry (already pre-reserved in Sprint 119 block above, use place_l3!).
    let ctrl_a_tap_l3 = place_l3!(ox + 22, oy + 9, TileType::ViaDown);
    push_pipeline!(ctrl_a_tap_l3);

    // L3 WireDown drop from entry to east wall.
    let ctrl_a_l3_drop = place_l3!(ox + 22, oy + 10, TileType::WireDown);
    push_pipeline!(ctrl_a_l3_drop);

    // L3 east wall at oy+10 (ox+23..60). [was L2]
    for x in (ox + 23)..=(ox + 60) {
        let idx = place_l3!(x, oy + 10, TileType::WireRight);
        push_pipeline!(idx);
    }
    // L3 south column at ox+60 (oy+11..48). [was L2]
    for y in (oy + 11)..=(oy + 48) {
        let idx = place_l3!(ox + 60, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    // L3 west bus at oy+48 (ox+22..59). [was L2]
    for x in (ox + 22)..=(ox + 59) {
        let idx = place_l3!(x, oy + 48, TileType::WireLeft);
        push_pipeline!(idx);
    }

    // Exit: L2 ViaUp relays L3 signal to L1. L1 ViaUp (existing) now reads L2 ViaUp.
    let ctrl_a_l2_exit = place_l2!(ox + 22, oy + 48, TileType::ViaUp);
    push_pipeline!(ctrl_a_l2_exit);
    let ctrl_a_flag_via_up_l1 = place_l1_route!(ox + 22, oy + 48, TileType::ViaUp);
    push_pipeline!(ctrl_a_flag_via_up_l1);
    // Sprint 206: Fused shift+mask via — replaces L1 Shr + L1 Const(4) + L0 ViaUp
    // with L1 WireRight (forwards ctrl_a) + L0 WeightedViaUp (shift=4, mask=0x03).
    // Frees L1(ox+24, oy+48) formerly occupied by Const(4).
    let flag_we_wire_l1 = place_l1_route!(ox + 23, oy + 48, TileType::WireRight);
    let flag_we_mask_const_idx = place!(ox + 23, oy + 48, TileType::WeightedViaUp);
    sim.set_tile_shift(flag_we_mask_const_idx, 4);
    sim.set_tile_mask(flag_we_mask_const_idx, 0x03);
    push_pipeline!(flag_we_wire_l1);
    push_pipeline!(flag_we_mask_const_idx);

    let mut flag_we_route_indices = Vec::new();
    for x in (ox + 18)..=(ox + 22) {
        let idx = place!(x, oy + 48, TileType::WireLeft);
        flag_we_route_indices.push(idx);
        push_pipeline!(idx); // Sprint 208: flag WE routes in pipeline scope
    }
    let flag_we_drop_z_idx = place!(ox + 18, oy + 49, TileType::WireDown);
    push_pipeline!(flag_we_drop_z_idx); // Sprint 208
    let flag_we_drop_c_idx = place!(ox + 22, oy + 49, TileType::WireDown);
    push_pipeline!(flag_we_drop_c_idx); // Sprint 208
    let flag_z_we_idx = place!(ox + 19, oy + 49, TileType::BitSelect);
    push_pipeline!(flag_z_we_idx); // Sprint 208
    let _flag_z_we_bit_idx = place_val!(ox + 20, oy + 49, TileType::Const, 0);
    let flag_c_we_idx = place!(ox + 23, oy + 49, TileType::BitSelect);
    push_pipeline!(flag_c_we_idx); // Sprint 208
    let _flag_c_we_bit_idx = place_val!(ox + 24, oy + 49, TileType::Const, 1);
    let _carry_const_idx = place!(ox + 22, oy + 50, TileType::ViaUp);
    // Sprint 123: Physical SHR carry replaces software Const at L2(ox+21, oy+50).
    // WireUp reads DOWN = L2(ox+21, oy+51) = north end of Phase B L2 route.
    let _s123_carry_mux_left = place_l2_route!(ox + 21, oy + 50, TileType::WireUp);
    let carry_mux_l2_idx = place_l2_route!(ox + 22, oy + 50, TileType::Mux);
    let carry_or_l1_idx = place_l1_route!(ox + 22, oy + 50, TileType::ViaUp);
    let flag_z_data_via_idx = place!(ox + 17, oy + 50, TileType::ViaUp);
    let flag_z_zero_idx = place!(ox + 18, oy + 50, TileType::Zero);
    let flag_z_mux_idx = place!(ox + 19, oy + 50, TileType::Mux);
    let flag_c_mux_idx = place!(ox + 23, oy + 50, TileType::Mux);

    let mut flag_commit_dirty_indices = vec![
        wb_via_down_l1, // shared source for wb_data
        flag_z_data_via_idx,
        flag_z_zero_idx,
        flag_we_drop_z_idx,
        flag_we_drop_c_idx,
        flag_z_we_idx,
        flag_c_we_idx,
        flag_z_mux_idx,
        flag_c_mux_idx,
        _carry_const_idx,
        carry_mux_l2_idx,
        carry_or_l1_idx,
        s123_sub, // Sprint 123: kick off SHR carry cascade during flag commit
    ];
    flag_commit_dirty_indices.extend(flag_we_route_indices.iter().copied());

    // L1 route: wb_data source col -> south to flag row -> west to flag via
    for y in (oy + 11)..=(oy + 49) {
        let idx = place_l1_route!(ox + 37, y, TileType::WireDown);
        flag_commit_dirty_indices.push(idx);
    }
    for x in (ox + 18)..=(ox + 36) {
        let idx = place_l1_route!(x, oy + 49, TileType::WireLeft);
        flag_commit_dirty_indices.push(idx);
    }
    let idx_down = place_l1_route!(ox + 18, oy + 50, TileType::WireDown);
    flag_commit_dirty_indices.push(idx_down);
    let idx_left = place_l1_route!(ox + 17, oy + 50, TileType::WireLeft);
    flag_commit_dirty_indices.push(idx_left);

    // -------------------------------------------------------------------------
    // Sprint 107: physical branch LUT selector assembly
    // -------------------------------------------------------------------------
    // Replace 3 software-written L2 Const tiles with physical computation:
    //   selector_low = ctrl_b[2:0] | (flag_z << 3)
    //   bank_sel     = inverted flag_c (MAX when c=0, 0 when c=1)
    let mut branch_dirty_indices = Vec::new();
    // Sprint 154: Flag-only subset — only flag_z and flag_c routes.
    let mut branch_flag_dirty_indices = Vec::new();

    // --- 1. ctrl_b L2 east highway ---
    // L1 (ox+32, oy+8) freed by shortening opcode_bit4 bus to ox+31 (see above).
    // Direct via chain: L1 ViaDown reads ctrl_b Mux, L2 ViaDown reads L1 ViaDown.
    let _branch_ctrl_b_l1_tap_idx = place_l1_route!(ox + 32, oy + 8, TileType::ViaDown);
    let s107_ctrl_b_l2_entry = place_l2_route!(ox + 32, oy + 8, TileType::ViaDown);
    branch_dirty_indices.push(_branch_ctrl_b_l1_tap_idx);
    branch_dirty_indices.push(s107_ctrl_b_l2_entry);
    // L2 east at oy+8 from ox+33 to ox+61.
    // Bridge over the bank-2 byte2/byte3 descents on L3, then resume the
    // control row before the later taps at x>=45.
    for x in (ox + 33)..=(ox + 39) {
        let idx = place_l2_route!(x, oy + 8, TileType::WireRight);
        branch_dirty_indices.push(idx);
    }
    let s107_ctrl_b_l3_bridge_entry = place_l3_route!(ox + 39, oy + 8, TileType::ViaDown);
    branch_dirty_indices.push(s107_ctrl_b_l3_bridge_entry);
    for x in (ox + 40)..=(ox + 44) {
        let idx = place_l3_route!(x, oy + 8, TileType::WireRight);
        branch_dirty_indices.push(idx);
    }
    let s107_ctrl_b_l2_bridge_resume = place_l2_route!(ox + 44, oy + 8, TileType::ViaUp);
    branch_dirty_indices.push(s107_ctrl_b_l2_bridge_resume);
    for x in (ox + 45)..=(ox + 61) {
        let idx = place_l2_route!(x, oy + 8, TileType::WireRight);
        branch_dirty_indices.push(idx);
    }
    // L2 south at ox+61 from oy+9 to oy+49.
    // Bypass the byte-delivery cross rows on L3 from oy+12 through oy+18.
    for y in (oy + 9)..=(oy + 11) {
        let idx = place_l2_route!(ox + 61, y, TileType::WireDown);
        branch_dirty_indices.push(idx);
    }
    let s107_ctrl_b_l3_entry = place_l3_route!(ox + 61, oy + 11, TileType::ViaDown);
    branch_dirty_indices.push(s107_ctrl_b_l3_entry);
    for y in (oy + 12)..=(oy + 18) {
        let idx = place_l3_route!(ox + 61, y, TileType::WireDown);
        branch_dirty_indices.push(idx);
    }
    let s107_ctrl_b_l2_resume = place_l2_route!(ox + 61, oy + 18, TileType::ViaUp);
    branch_dirty_indices.push(s107_ctrl_b_l2_resume);
    for y in (oy + 19)..=(oy + 49) {
        let idx = place_l2_route!(ox + 61, y, TileType::WireDown);
        branch_dirty_indices.push(idx);
    }
    // Drop to L0 at (ox+61, oy+49): L1 ViaUp reads L2, L0 ViaUp reads L1.
    let s107_ctrl_b_l1_drop = place_l1_route!(ox + 61, oy + 49, TileType::ViaUp);
    let s107_ctrl_b_l0_drop = place!(ox + 61, oy + 49, TileType::ViaUp);
    branch_dirty_indices.push(s107_ctrl_b_l1_drop);
    branch_dirty_indices.push(s107_ctrl_b_l0_drop);

    // --- 2. flag_z L1 east route ---
    // Tap flag_z Register8 at L0 (ox+20, oy+50) via L1 ViaDown.
    let _branch_flag_z_l1_tap_idx = place_l1_route!(ox + 20, oy + 50, TileType::ViaDown);
    let s107_fz_l1_wd = place_l1_route!(ox + 20, oy + 51, TileType::WireDown);
    branch_dirty_indices.push(_branch_flag_z_l1_tap_idx);
    branch_dirty_indices.push(s107_fz_l1_wd);
    branch_flag_dirty_indices.push(_branch_flag_z_l1_tap_idx);
    branch_flag_dirty_indices.push(s107_fz_l1_wd);
    // L1 east at oy+51 from ox+21 to ox+60.
    for x in (ox + 21)..=(ox + 60) {
        let idx = place_l1_route!(x, oy + 51, TileType::WireRight);
        branch_dirty_indices.push(idx);
        branch_flag_dirty_indices.push(idx);
    }
    // Drop to L0 at (ox+60, oy+51).
    let s107_fz_l0_via = place!(ox + 60, oy + 51, TileType::ViaUp);
    branch_dirty_indices.push(s107_fz_l0_via);
    branch_flag_dirty_indices.push(s107_fz_l0_via);

    // --- 3. flag_c L1 east route ---
    // Tap flag_c Register8 at L0 (ox+24, oy+50) via L1 ViaDown.
    let _branch_flag_c_l1_tap_idx = place_l1_route!(ox + 24, oy + 50, TileType::ViaDown);
    branch_dirty_indices.push(_branch_flag_c_l1_tap_idx);
    branch_flag_dirty_indices.push(_branch_flag_c_l1_tap_idx);
    // L1 east at oy+50 from ox+25 to ox+64.
    for x in (ox + 25)..=(ox + 64) {
        let idx = place_l1_route!(x, oy + 50, TileType::WireRight);
        branch_dirty_indices.push(idx);
        branch_flag_dirty_indices.push(idx);
    }
    // Drop to L0 at (ox+64, oy+50).
    let s107_fc_l0_via = place!(ox + 64, oy + 50, TileType::ViaUp);
    branch_dirty_indices.push(s107_fc_l0_via);
    branch_flag_dirty_indices.push(s107_fc_l0_via);

    // --- 4. Assembly zone (L0) ---
    // Step A: ctrl_b masking — And(ctrl_b, 7) → ctrl_b_masked.
    let s107_ctrl_b_wd = place!(ox + 61, oy + 50, TileType::WireDown);
    let s107_ctrl_b_and = place!(ox + 62, oy + 50, TileType::And);
    let _s107_mask7 = place_val!(ox + 63, oy + 50, TileType::Const, 7);
    // Step B: flag_c inversion — Zero(flag_c) → bank_sel.
    // flag_c arrives at L0 (ox+64, oy+50) ViaUp; Zero reads LEFT.
    let s107_bank_sel_zero = place!(ox + 65, oy + 50, TileType::Zero);
    // Step C: flag_z processing — And(Const(8), flag_z) → flag_z_bit3.
    let _s107_mask8 = place_val!(ox + 58, oy + 51, TileType::Const, 8);
    let s107_fz_and = place!(ox + 59, oy + 51, TileType::And);
    // Step D: Combine ctrl_b_masked | flag_z_bit3 → selector_low.
    let s107_masked_wd1 = place!(ox + 62, oy + 51, TileType::WireDown);
    let s107_masked_wd2 = place!(ox + 62, oy + 52, TileType::WireDown);
    let s107_bit3_wd = place!(ox + 59, oy + 52, TileType::WireDown);
    let s107_bit3_wr = place!(ox + 60, oy + 52, TileType::WireRight);
    let s107_sel_or = place!(ox + 61, oy + 52, TileType::Or);
    branch_dirty_indices.extend_from_slice(&[
        s107_ctrl_b_wd,
        s107_ctrl_b_and,
        s107_bank_sel_zero,
        s107_fz_and,
        s107_masked_wd1,
        s107_masked_wd2,
        s107_bit3_wd,
        s107_bit3_wr,
        s107_sel_or,
    ]);

    // --- 5. selector_low delivery: L0 Or → L1 → L2 ViaDown ---
    let s107_sel_l1_tap = place_l1_route!(ox + 61, oy + 52, TileType::ViaDown);
    let s107_sel_l1_wd = place_l1_route!(ox + 61, oy + 53, TileType::WireDown);
    branch_dirty_indices.push(s107_sel_l1_tap);
    branch_dirty_indices.push(s107_sel_l1_wd);
    for x in (ox + 62)..=(ox + 68) {
        let idx = place_l1_route!(x, oy + 53, TileType::WireRight);
        branch_dirty_indices.push(idx);
    }

    // --- 6. bank_sel delivery: L0 Zero → L1 → L2 south column ---
    let s107_bank_l1_tap = place_l1_route!(ox + 65, oy + 50, TileType::ViaDown);
    let s107_bank_l2_entry = place_l2_route!(ox + 65, oy + 50, TileType::ViaDown);
    let s107_bank_l2_wd1 = place_l2_route!(ox + 65, oy + 51, TileType::WireDown);
    let s107_bank_l2_wd2 = place_l2_route!(ox + 65, oy + 52, TileType::WireDown);
    branch_dirty_indices.extend_from_slice(&[
        s107_bank_l1_tap,
        s107_bank_l2_entry,
        s107_bank_l2_wd1,
        s107_bank_l2_wd2,
    ]);

    // -------------------------------------------------------------------------
    // Sprint 103: branch decision LUT core (stage-X selector driven)
    // -------------------------------------------------------------------------
    let mut branch_taken_table = [0u8; 32];
    for sel in 0..32usize {
        let kind = (sel & 0x07) as u8;
        let z = ((sel >> 3) & 0x01) != 0;
        let c = ((sel >> 4) & 0x01) != 0;
        let taken = match kind {
            1 => true, // JMP
            2 => z,    // JZ
            3 => !z,   // JNZ
            4 => c,    // JC
            5 => !c,   // JNC
            6 => true, // CALL (Sprint 124)
            7 => true, // RET (Sprint 124)
            _ => false,
        };
        branch_taken_table[sel] = u8::from(taken);
    }

    let branch_bx = ox + 62;
    let branch_by = oy + 52;

    reserve_l2!(branch_bx + 1, branch_by);
    let _branch_low_up = place_l2_val!(
        branch_bx + 1,
        branch_by,
        TileType::Const,
        pack_8(&branch_taken_table[8..16])
    );
    reserve_l2!(branch_bx + 5, branch_by);
    let _branch_high_up = place_l2_val!(
        branch_bx + 5,
        branch_by,
        TileType::Const,
        pack_8(&branch_taken_table[24..32])
    );

    reserve_l2!(branch_bx, branch_by + 1);
    let _branch_low_left = place_l2_val!(
        branch_bx,
        branch_by + 1,
        TileType::Const,
        pack_8(&branch_taken_table[0..8])
    );
    reserve_l2!(branch_bx + 4, branch_by + 1);
    let _branch_high_left = place_l2_val!(
        branch_bx + 4,
        branch_by + 1,
        TileType::Const,
        pack_8(&branch_taken_table[16..24])
    );

    let branch_low_mux_idx = place_l2_route!(branch_bx + 1, branch_by + 1, TileType::Mux16to1);
    let branch_high_mux_idx = place_l2_route!(branch_bx + 5, branch_by + 1, TileType::Mux16to1);

    // Sprint 107: physical delivery replaces software-written Consts.
    // sel_low/sel_high: L2 ViaDown reads L1 WireRight = selector_low.
    let s107_sel_low_l2 = place_l2_route!(branch_bx + 2, branch_by + 1, TileType::ViaDown);
    let s107_sel_high_l2 = place_l2_route!(branch_bx + 6, branch_by + 1, TileType::ViaDown);
    // bank_sel: L2 WireDown reads L2 south column = inverted flag_c.
    let s107_bank_sel_l2 = place_l2_route!(branch_bx + 3, branch_by + 1, TileType::WireDown);

    let branch_low_down_idx = place_l2_route!(branch_bx + 1, branch_by + 2, TileType::WireDown);
    let branch_low_to_bank_idx = place_l2_route!(branch_bx + 2, branch_by + 2, TileType::WireRight);
    let branch_high_down_idx = place_l2_route!(branch_bx + 5, branch_by + 2, TileType::WireDown);
    let branch_high_to_bank_idx = place_l2_route!(branch_bx + 4, branch_by + 2, TileType::WireLeft);
    let branch_taken_core_idx = place_l2_route!(branch_bx + 3, branch_by + 2, TileType::Mux);

    branch_dirty_indices.extend_from_slice(&[
        s107_sel_low_l2,
        s107_sel_high_l2,
        s107_bank_sel_l2,
        branch_low_mux_idx,
        branch_high_mux_idx,
        branch_low_down_idx,
        branch_low_to_bank_idx,
        branch_high_down_idx,
        branch_high_to_bank_idx,
        branch_taken_core_idx,
    ]);

    // -------------------------------------------------------------------------
    // Sprint 162: 64 permanent RAM tiles in 8 banks of 8
    // Sprint 189: expanded to 128 cells (16 banks of 8)
    // -------------------------------------------------------------------------
    // Bank 0 (cells 0-7): legacy positions at (ox+70+i*2, oy+54).
    // Banks 1-7 (cells 8-63): east-side region at (ox+96+i*2, oy+56+bank).
    // Banks 8-15 (cells 64-127): further east at (ox+112+i*2, oy+56+bank).
    let mut ram_indices = [0usize; 128];
    for i in 0..8usize {
        let x = ox + 70 + i * 2;
        let y = oy + 54;
        ram_indices[i] = place_val!(x, y, TileType::Ram, 0);
    }
    for bank in 1..8usize {
        for i in 0..8usize {
            let addr = bank * 8 + i;
            let x = ox + 96 + i * 2;
            let y = oy + 56 + bank;
            ram_indices[addr] = place_val!(x, y, TileType::Ram, 0);
        }
    }
    // Sprint 189: Banks 8-15 (cells 64-127)
    for bank in 0..8usize {
        for i in 0..8usize {
            let addr = 64 + bank * 8 + i;
            let x = ox + 112 + i * 2;
            let y = oy + 56 + bank;
            ram_indices[addr] = place_val!(x, y, TileType::Ram, 0);
        }
    }

    // -------------------------------------------------------------------------
    // Sprint 108: physical RAM read address Mux (replaces software Const)
    // -------------------------------------------------------------------------
    // Mux at (ox+86, oy+53): UP=use_imm, LEFT=imm8, RIGHT=opB.
    // Output descends via WireDown chain to existing BitSelect extraction at oy+55.
    // Avoids L1 oy+54 WireLeft bus (RAM write data, ox+69..89).
    //
    // Route 1 — opB: L1 east from Tree B root tap at (ox+47, oy+32).
    //   L1 WireRight oy+32 (ox+48..87) → L1 WireDown ox+87 (oy+33..53) → L0 ViaUp.
    // Route 2 — imm8: L2 south from ir_low at (ox+1, oy+5).
    //   L2 WireDown ox+1 (oy+6..55) → L2 WireRight oy+55 (ox+2..85) → L2 WireUp
    //   (ox+85, oy+53..54) → L1 ViaUp → L0 ViaUp.
    // Route 3 — use_imm: ctrl_a[6] extraction at (ox+23, oy+14), L1/L2 hybrid.
    //   L1 east oy+14 (ox+24..36) → L2 flag-spine detour (ox+36..38) → L1 east
    //   oy+14 (ox+39..62) → L2 east oy+14 (ox+63..86) → L2 south ox+86 (oy+15..52)
    //   → L1/L0 ViaUp at (ox+86, oy+52).

    // --- Route 1: opB (Tree B root → Mux RIGHT) ---
    // Existing tap: L1 ViaDown at (ox+47, oy+32). Branch east from there.
    for x in (ox + 48)..=(ox + 87) {
        let idx = place_l1_route!(x, oy + 32, TileType::WireRight);
        push_pipeline!(idx);
    }
    for y in (oy + 33)..=(oy + 53) {
        let idx = place_l1_route!(ox + 87, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    let s108_opb_l0_via = place!(ox + 87, oy + 53, TileType::ViaUp);
    push_pipeline!(s108_opb_l0_via);

    // --- Route 2: imm8 (ir_low → Mux LEFT) ---
    // Sprint 108/120: Tap ir_low WireDown chain at (ox+1, oy+5) via L1+L2+L3 ViaDown.
    // Sprint 120: Route migrated from L2 to L3. Entry/exit via chains now go
    // L1→L2 ViaDown (relay)→L3 ViaDown→(L3 route)→L2 ViaUp→L1 ViaUp→L0 ViaUp.
    // Frees 135 L2 positions (ox+1 column + oy+55 east corridor).
    let s108_imm8_l1_tap = place_l1_route!(ox + 1, oy + 5, TileType::ViaDown);
    let s108_imm8_l2_tap = place_l2_route!(ox + 1, oy + 5, TileType::ViaDown);
    push_pipeline!(s108_imm8_l1_tap);
    push_pipeline!(s108_imm8_l2_tap);
    // L3 entry (already pre-reserved in Sprint 120 block, use place_l3!).
    let s120_imm8_l3_tap = place_l3!(ox + 1, oy + 5, TileType::ViaDown);
    push_pipeline!(s120_imm8_l3_tap);
    // L3 south at ox+1 from oy+6 to oy+55. [was L2]
    for y in (oy + 6)..=(oy + 55) {
        let idx = place_l3!(ox + 1, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    // L3 east at oy+55 from ox+2 to ox+85. [was L2]
    for x in (ox + 2)..=(ox + 85) {
        let idx = place_l3!(x, oy + 55, TileType::WireRight);
        push_pipeline!(idx);
    }
    // L3 north at ox+85 from oy+54 to oy+53 (WireUp reads south neighbor). [was L2]
    let s120_imm8_l3_wu1 = place_l3!(ox + 85, oy + 54, TileType::WireUp);
    let s120_imm8_l3_wu2 = place_l3!(ox + 85, oy + 53, TileType::WireUp);
    push_pipeline!(s120_imm8_l3_wu1);
    push_pipeline!(s120_imm8_l3_wu2);
    // Exit: L2 ViaUp relays L3→L2. L1 ViaUp + L0 ViaUp unchanged.
    let s120_imm8_l2_exit = place_l2!(ox + 85, oy + 53, TileType::ViaUp);
    push_pipeline!(s120_imm8_l2_exit);
    let s108_imm8_l1_drop = place_l1_route!(ox + 85, oy + 53, TileType::ViaUp);
    let s108_imm8_l0_drop = place!(ox + 85, oy + 53, TileType::ViaUp);
    push_pipeline!(s108_imm8_l1_drop);
    push_pipeline!(s108_imm8_l0_drop);

    // Sprint 120: L2 ViaUp bridge at (ox+1, oy+33) relays L3 imm8 signal to L2.
    // Sprint 114 L2 east bus at oy+33 taps the south column at (ox+1, oy+33).
    // Without this bridge, L2 at (ox+1, oy+33) is guard Const(0) after migration.
    let s120_imm8_l2_bridge = place_l2!(ox + 1, oy + 33, TileType::ViaUp);
    push_pipeline!(s120_imm8_l2_bridge);

    // --- Route 3: use_imm (ctrl_a[6] extraction + L1/L2 hybrid delivery) ---
    // L1 runs east at oy+14 (above rd selector buses at oy+23/26/29, avoids
    // L2 BFS zone x=20..48). L2 detour at flag spine (ox+36..38). Transition
    // to L2 at ox+62 (east of Sprint 107 ctrl_b L2 column at ox+61). L2 east
    // to ox+86, then L2 south to oy+52. Drops to L0 at Mux UP position.
    //
    // Extraction on L0: extend ctrl_a WireDown from oy+13 to oy+14.
    let s108_ctrl_a_wd14 = place!(ox + 22, oy + 14, TileType::WireDown);
    let s108_bit6 = place!(ox + 23, oy + 14, TileType::BitSelect);
    let _s108_bit6_pos = place_val!(ox + 24, oy + 14, TileType::Const, 6);
    // Sprint 109: extract ctrl_a[7] (use_alu bit) at oy+15.
    // Replaces Sprint 108 guard Const(0) — BitSelect ignores UP/DOWN so it
    // naturally blocks bit6 leakage south. This also extends ctrl_a WireDown.
    let s109_ctrl_a_wd15 = place!(ox + 22, oy + 15, TileType::WireDown);
    let s109_bit7 = place!(ox + 23, oy + 15, TileType::BitSelect);
    let _s109_bit7_pos = place_val!(ox + 24, oy + 15, TileType::Const, 7);
    push_pipeline!(s109_ctrl_a_wd15);
    push_pipeline!(s109_bit7);
    push_pipeline!(s108_ctrl_a_wd14);
    push_pipeline!(s108_bit6);
    // Tap bit6 to L1 at (ox+23, oy+14).
    let s108_use_imm_l1_tap = place_l1_route!(ox + 23, oy + 14, TileType::ViaDown);
    push_pipeline!(s108_use_imm_l1_tap);
    // L2 hop over (ox+24, oy+14) — reserved by WE mask L1 WireDown (oy+10..21).
    let s108_hop_l2_entry = place_l2_route!(ox + 23, oy + 14, TileType::ViaDown);
    let s108_hop_l2_wr24 = place_l2_route!(ox + 24, oy + 14, TileType::WireRight);
    let s108_hop_l2_wr25 = place_l2_route!(ox + 25, oy + 14, TileType::WireRight);
    push_pipeline!(s108_hop_l2_entry);
    push_pipeline!(s108_hop_l2_wr24);
    push_pipeline!(s108_hop_l2_wr25);
    let s108_hop_l1_exit = place_l1_route!(ox + 25, oy + 14, TileType::ViaUp);
    push_pipeline!(s108_hop_l1_exit);
    // L1 east at oy+14 from ox+26 to ox+36 (up to flag spine at ox+37).
    for x in (ox + 26)..=(ox + 36) {
        let idx = place_l1_route!(x, oy + 14, TileType::WireRight);
        push_pipeline!(idx);
    }
    // L2 detour around flag spine at (ox+36..38, oy+14).
    // Sprint 101 reserves L1 (ox+37, oy+14) as WireDown. Cross on L2.
    let s108_spine_l2_entry = place_l2_route!(ox + 36, oy + 14, TileType::ViaDown);
    let s108_spine_l2_wr37 = place_l2_route!(ox + 37, oy + 14, TileType::WireRight);
    let s108_spine_l2_wr38 = place_l2_route!(ox + 38, oy + 14, TileType::WireRight);
    push_pipeline!(s108_spine_l2_entry);
    push_pipeline!(s108_spine_l2_wr37);
    push_pipeline!(s108_spine_l2_wr38);
    let s108_spine_l1_exit = place_l1_route!(ox + 38, oy + 14, TileType::ViaUp);
    push_pipeline!(s108_spine_l1_exit);
    // L1 east at oy+14 from ox+39 to ox+45 (up to WE mask Bank B at ox+46).
    for x in (ox + 39)..=(ox + 45) {
        let idx = place_l1_route!(x, oy + 14, TileType::WireRight);
        push_pipeline!(idx);
    }
    // L2 hop over WE mask Bank B column at (ox+45..47, oy+14). Pre-reserved.
    let s108_we_l2_entry = place_l2!(ox + 45, oy + 14, TileType::ViaDown);
    let s108_we_l2_wr46 = place_l2!(ox + 46, oy + 14, TileType::WireRight);
    let s108_we_l2_wr47 = place_l2!(ox + 47, oy + 14, TileType::WireRight);
    push_pipeline!(s108_we_l2_entry);
    push_pipeline!(s108_we_l2_wr46);
    push_pipeline!(s108_we_l2_wr47);
    let s108_we_l1_exit = place_l1_route!(ox + 47, oy + 14, TileType::ViaUp);
    push_pipeline!(s108_we_l1_exit);
    // L1 east at oy+14 from ox+48 to ox+54 (up to rs delivery column at ox+55).
    for x in (ox + 48)..=(ox + 54) {
        let idx = place_l1_route!(x, oy + 14, TileType::WireRight);
        push_pipeline!(idx);
    }
    // L2 hop over rs delivery column at (ox+54..56, oy+14).
    let s108_rs_l2_entry = place_l2_route!(ox + 54, oy + 14, TileType::ViaDown);
    let s108_rs_l2_wr55 = place_l2_route!(ox + 55, oy + 14, TileType::WireRight);
    let s108_rs_l2_wr56 = place_l2_route!(ox + 56, oy + 14, TileType::WireRight);
    push_pipeline!(s108_rs_l2_entry);
    push_pipeline!(s108_rs_l2_wr55);
    push_pipeline!(s108_rs_l2_wr56);
    let s108_rs_l1_exit = place_l1_route!(ox + 56, oy + 14, TileType::ViaUp);
    push_pipeline!(s108_rs_l1_exit);
    // L1 east at oy+14 from ox+57 to ox+62 (past Sprint 107 L2 column at ox+61).
    for x in (ox + 57)..=(ox + 62) {
        let idx = place_l1_route!(x, oy + 14, TileType::WireRight);
        push_pipeline!(idx);
    }
    // L1→L2 transition at (ox+62, oy+14) — east of both BFS zone and S107 column.
    // Sprint 121: L2 ViaDown stays as relay. L3 ViaDown reads L2 relay.
    let s108_use_imm_l2_entry = place_l2_route!(ox + 62, oy + 14, TileType::ViaDown);
    push_pipeline!(s108_use_imm_l2_entry);
    let s121_use_imm_l3_entry = place_l3!(ox + 62, oy + 14, TileType::ViaDown);
    push_pipeline!(s121_use_imm_l3_entry);
    // Sprint 121: L3 east at oy+14 from ox+63 to ox+86. [was L2]
    for x in (ox + 63)..=(ox + 86) {
        let idx = place_l3!(x, oy + 14, TileType::WireRight);
        push_pipeline!(idx);
    }
    // Sprint 121: L3 south at ox+86 from oy+15 to oy+52. [was L2]
    for y in (oy + 15)..=(oy + 52) {
        let idx = place_l3!(ox + 86, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    // Sprint 121: L2 ViaUp exit relay at (ox+86, oy+52) reads L3.
    let s121_use_imm_l2_exit = place_l2!(ox + 86, oy + 52, TileType::ViaUp);
    push_pipeline!(s121_use_imm_l2_exit);
    // Drop: L1 ViaUp + L0 ViaUp at (ox+86, oy+52). L1 now reads L2 ViaUp.
    let s108_use_imm_l1_drop = place_l1_route!(ox + 86, oy + 52, TileType::ViaUp);
    let s108_use_imm_l0_drop = place!(ox + 86, oy + 52, TileType::ViaUp);
    push_pipeline!(s108_use_imm_l1_drop);
    push_pipeline!(s108_use_imm_l0_drop);

    // --- Mux + WireDown chain ---
    // Mux at (ox+86, oy+53): UP=use_imm, LEFT=imm8, RIGHT=opB.
    let s108_addr_mux = place!(ox + 86, oy + 53, TileType::Mux);
    push_pipeline!(s108_addr_mux);
    // WireDown relay: Mux output → oy+54 → oy+55 → existing BitSelect chain.
    let s108_relay_54 = place!(ox + 86, oy + 54, TileType::WireDown);
    push_pipeline!(s108_relay_54);

    // -------------------------------------------------------------------------
    // Sprint 109: physical wb_data override enable (!use_alu)
    // -------------------------------------------------------------------------
    // Replace software enable Const at (ox+38, oy+9) with physical !use_alu
    // derived from ctrl_a[7].
    //
    // Route: extraction at oy+15 → L1 west to ox+11 → L1 north to oy+9 →
    // L2 north (ox+11, oy+1..8) → L2 east at oy+1 (ox+12..37) →
    // L2 south (ox+37, oy+2..7) → drop L2→L1→L0 at (ox+37, oy+7) →
    // L0 south at (ox+37, oy+8..9) → Zero at (ox+38, oy+9) inverts.
    //
    // L1 column ox+11 is clear (between rd1 at ox+9 and rd0 at ox+12).
    // Sprint 133: L2 east rerouted from oy+2 to oy+1 to free oy+2 for bank taps.
    // All L2 positions pre-reserved to prevent BFS claims.
    //
    // Mux semantics: UP != 0 → LEFT (override), UP == 0 → RIGHT (ALU default).
    // When use_alu=1: !use_alu=0 → Mux selects RIGHT (ALU). Correct.
    // When use_alu=0: !use_alu=MAX → Mux selects LEFT (override). Correct.
    // WE mask = 0 for non-writing instructions, so stale LEFT is harmless.

    // Phase 1: L1 tap at (ox+23, oy+15) — extraction done earlier at line ~2053.
    let s109_l1_tap = place_l1_route!(ox + 23, oy + 15, TileType::ViaDown);
    push_pipeline!(s109_l1_tap);

    // Phase 2: L1 west at oy+15 from ox+22 to ox+13, with L2 hop over rd0 column (ox+12).
    // L1 at these positions is default Wire (no guard_l1_rect coverage west of ox+15).
    // No L0 ViaUp reads from these L1 tiles — L0 at oy+15 is guard Const(0).
    for x in (ox + 13)..=(ox + 22) {
        let idx = place_l1_route!(x, oy + 15, TileType::WireLeft);
        push_pipeline!(idx);
    }
    // L2 hop over rd0 at ox+12: ViaDown(13) → WireLeft(12) → WireLeft(11) → L1 ViaUp(11).
    let s109_hop_entry = place_l2!(ox + 13, oy + 15, TileType::ViaDown); // L1→L2
    push_pipeline!(s109_hop_entry);
    let s109_hop_12 = place_l2!(ox + 12, oy + 15, TileType::WireLeft); // west over rd0
    push_pipeline!(s109_hop_12);
    let s109_hop_11 = place_l2!(ox + 11, oy + 15, TileType::WireLeft); // land
    push_pipeline!(s109_hop_11);
    let s109_hop_drop = place_l1_route!(ox + 11, oy + 15, TileType::ViaUp); // L2→L1
    push_pipeline!(s109_hop_drop);

    // Phase 3: L1 north at ox+11 from oy+14 to oy+9.
    // ox+11 is completely clear on L1 (between rd1 at ox+9 and rd0 at ox+12).
    for y in (oy + 9)..=(oy + 14) {
        let idx = place_l1_route!(ox + 11, y, TileType::WireUp);
        push_pipeline!(idx);
    }

    // Phase 4: L2 ViaDown at (ox+11, oy+9) — L1→L2 transition. Pre-reserved.
    let s109_l2_entry = place_l2!(ox + 11, oy + 9, TileType::ViaDown);
    push_pipeline!(s109_l2_entry);

    // Phase 5: L2 north at ox+11 from oy+8 to oy+1 — rises above rd_field L2 wall.
    // Sprint 133: Extended from oy+2 to oy+1 to free L2 oy+2 for bank taps.
    for y in (oy + 1)..=(oy + 8) {
        let idx = place_l2!(ox + 11, y, TileType::WireUp); // pre-reserved
        push_pipeline!(idx);
    }

    // Phase 6: L2 east at oy+1 from ox+12 to ox+37 — above all L2 barriers.
    // Sprint 133: Moved from oy+2 to oy+1 to free L2 oy+2 for bank taps.
    for x in (ox + 12)..=(ox + 37) {
        let idx = place_l2!(x, oy + 1, TileType::WireRight); // pre-reserved
        push_pipeline!(idx);
    }

    // Phase 7: L2 south at ox+37 from oy+2 to oy+7.
    // Sprint 133: Extended from oy+3 to oy+2 to connect to rerouted east at oy+1.
    for y in (oy + 2)..=(oy + 7) {
        let idx = place_l2!(ox + 37, y, TileType::WireDown); // pre-reserved
        push_pipeline!(idx);
    }

    // Phase 8: Drop L2→L1→L0 at (ox+37, oy+7).
    let s109_l1_drop = place_l1_route!(ox + 37, oy + 7, TileType::ViaUp);
    push_pipeline!(s109_l1_drop);
    let s109_l0_drop = place!(ox + 37, oy + 7, TileType::ViaUp);
    push_pipeline!(s109_l0_drop);

    // Phase 9: L0 south delivery from (ox+37, oy+8..9).
    // Both positions are guard Const(0) — overwrite with WireDown.
    let s109_wd_8 = place!(ox + 37, oy + 8, TileType::WireDown);
    push_pipeline!(s109_wd_8);
    let s109_wd_9 = place!(ox + 37, oy + 9, TileType::WireDown);
    push_pipeline!(s109_wd_9);

    // Phase 10: Zero inverts use_alu → !use_alu. Replaces software enable Const.
    let _s109_enable = place!(ox + 38, oy + 9, TileType::Zero);
    push_pipeline!(_s109_enable);

    // -------------------------------------------------------------------------
    // Sprint 110/118: physical PC override enable (branch_taken_core → Or gate)
    // -------------------------------------------------------------------------
    // Sprint 118: Route migrated from L2 to L3. Entry/exit via chains now go
    // L1→L2 ViaDown→L3 ViaDown→(L3 route)→L2 ViaUp→L1 ViaUp→L0 ViaUp.

    // Phase 1: Tap branch_taken_core from L2 into L1. [unchanged]
    let s110_branch_tap_l1 = place_l1_route!(ox + 65, oy + 54, TileType::ViaUp);
    push_pipeline!(s110_branch_tap_l1);
    branch_dirty_indices.push(s110_branch_tap_l1);

    // Phase 2: L1 south to oy+56 for L2/L3 entry. [unchanged]
    let s110_l1_wd_55 = place_l1_route!(ox + 65, oy + 55, TileType::WireDown);
    let s110_l1_wd_56 = place_l1_route!(ox + 65, oy + 56, TileType::WireDown);
    push_pipeline!(s110_l1_wd_55);
    push_pipeline!(s110_l1_wd_56);
    branch_dirty_indices.push(s110_l1_wd_55);
    branch_dirty_indices.push(s110_l1_wd_56);

    // Phase 3: L2 ViaDown relay at (ox+65, oy+56). [keep — reads L1, relays to L3]
    let s110_l2_relay = place_l2!(ox + 65, oy + 56, TileType::ViaDown);
    push_pipeline!(s110_l2_relay);
    branch_dirty_indices.push(s110_l2_relay);

    // Phase 3b: L3 ViaDown entry at (ox+65, oy+56). [new — reads L2 relay]
    // (Already pre-reserved in Sprint 118 block above, use place_l3! directly.)
    let s118_l3_entry = place_l3!(ox + 65, oy + 56, TileType::ViaDown);
    push_pipeline!(s118_l3_entry);
    branch_dirty_indices.push(s118_l3_entry);

    // Phase 4: L3 west at oy+56 from ox+64 to ox+0. [was L2]
    for x in ox..=(ox + 64) {
        let idx = place_l3!(x, oy + 56, TileType::WireLeft);
        push_pipeline!(idx);
        branch_dirty_indices.push(idx);
    }

    // Phase 5: L3 north column at ox+0 from oy+55 to oy+0. [was L2]
    for y in oy..=(oy + 55) {
        let idx = place_l3!(ox, y, TileType::WireUp);
        push_pipeline!(idx);
        branch_dirty_indices.push(idx);
    }

    // Phase 6: L3 east at oy+0 from ox+1 to ox+3. [was L2]
    for x in (ox + 1)..=(ox + 3) {
        let idx = place_l3!(x, oy, TileType::WireRight);
        push_pipeline!(idx);
        branch_dirty_indices.push(idx);
    }

    // Phase 7: Exit L3→L2→L1→L0 at (ox+3, oy+0).
    // L2 ViaUp at (ox+3, oy+0) reads L3 WireRight. [new — replaces old L2 WireRight]
    let _s118_l2_exit = place_l2!(ox + 3, oy, TileType::ViaUp);
    push_pipeline!(_s118_l2_exit);
    branch_dirty_indices.push(_s118_l2_exit);
    // L1 ViaUp at (ox+3, oy+0) reads L2 ViaUp → already placed in PC ingress section.
    // L0 ViaUp at (ox+3, oy+0) reads L1 → already placed in PC ingress section.
    // Or gate at L0 (ox+2, oy+0) reads RIGHT = L0 ViaUp(ox+3) = branch_taken_core. ✓
    // L1 ViaDown at (ox+2, oy+0) reads L0 Or → feeds pc_next Mux UP. ✓
    branch_dirty_indices.push(_pc_enable_physical_via_l1);
    branch_dirty_indices.push(_pc_enable_physical_via_l0);
    branch_dirty_indices.push(_pc_enable_or_idx);
    branch_dirty_indices.push(_pc_enable_viadown_l1);

    // -------------------------------------------------------------------------
    // Sprint 111: physical ALU R enable (use_imm → 8 R Muxes)
    // -------------------------------------------------------------------------
    // Route: tap L1 use_imm bus at (ox+30, oy+14) → L1 south column (ox+30,
    // oy+15..36) → L1 east/west bus at oy+36 with L2 hops over Tree A trunk
    // (ox+27) and flag spine (ox+37). L0 drops at each R enable position.
    //
    // R Mux layout: (x+1, oy+37) = Mux, (x+1, oy+36) = R enable [UP].
    // When use_imm=1: Mux selects LEFT (software R override value).
    // When use_imm=0: Mux selects RIGHT (Tree B physical default).

    // Phase 1: L1 south column (ox+34, oy+15..36) — 22 tiles.
    // Reads UP = L1 WireRight at (ox+34, oy+14) = use_imm signal from Sprint 108.
    // ox+34 is clear on L1 (rd0 bus ends at ox+33, rd1/rd2 end well before ox+34).
    for y in (oy + 15)..=(oy + 36) {
        let idx = place_l1_route!(ox + 34, y, TileType::WireDown);
        push_pipeline!(idx);
    }

    // Phase 2a: Westbound bus at oy+36 (to i=0,1,2 R enables at ox+19, ox+23, ox+27).
    // L1 WireLeft from ox+33 (reads RIGHT = south column) back to ox+19.
    for x in (ox + 29)..=(ox + 33) {
        let idx = place_l1_route!(x, oy + 36, TileType::WireLeft);
        push_pipeline!(idx);
    }
    // L2 hop over Tree A trunk at ox+27 (L1 occupied by WireDown).
    let s111_west_l2_entry = place_l2_route!(ox + 29, oy + 36, TileType::ViaDown);
    let s111_west_l2_wr28 = place_l2_route!(ox + 28, oy + 36, TileType::WireLeft);
    let s111_west_l2_wr27 = place_l2_route!(ox + 27, oy + 36, TileType::WireLeft);
    let s111_west_l2_wr26 = place_l2_route!(ox + 26, oy + 36, TileType::WireLeft);
    push_pipeline!(s111_west_l2_entry);
    push_pipeline!(s111_west_l2_wr28);
    push_pipeline!(s111_west_l2_wr27);
    push_pipeline!(s111_west_l2_wr26);
    let s111_west_l1_exit = place_l1_route!(ox + 26, oy + 36, TileType::ViaUp);
    push_pipeline!(s111_west_l1_exit);
    // Continue L1 west from ox+25 to ox+19.
    for x in (ox + 19)..=(ox + 25) {
        let idx = place_l1_route!(x, oy + 36, TileType::WireLeft);
        push_pipeline!(idx);
    }

    // Phase 2b: Eastbound bus at oy+36 (to i=3..7 R enables at ox+31..ox+47).
    // L1 WireRight from ox+35 to ox+36 (up to flag spine at ox+37).
    for x in (ox + 35)..=(ox + 36) {
        let idx = place_l1_route!(x, oy + 36, TileType::WireRight);
        push_pipeline!(idx);
    }
    // L2 hop over flag spine at ox+37 (L1 occupied by WireDown).
    let s111_east_l2_entry = place_l2_route!(ox + 36, oy + 36, TileType::ViaDown);
    let s111_east_l2_wr37 = place_l2_route!(ox + 37, oy + 36, TileType::WireRight);
    let s111_east_l2_wr38 = place_l2_route!(ox + 38, oy + 36, TileType::WireRight);
    push_pipeline!(s111_east_l2_entry);
    push_pipeline!(s111_east_l2_wr37);
    push_pipeline!(s111_east_l2_wr38);
    let s111_east_l1_exit = place_l1_route!(ox + 38, oy + 36, TileType::ViaUp);
    push_pipeline!(s111_east_l1_exit);
    // Continue L1 east from ox+39 to ox+46.
    for x in (ox + 39)..=(ox + 46) {
        let idx = place_l1_route!(x, oy + 36, TileType::WireRight);
        push_pipeline!(idx);
    }

    // Phase 3: Special relay drops for i=2 and i=7.
    // i=2: R enable at (ox+27, oy+36) is WireLeft (placed in ALU loop above).
    //   It reads RIGHT from L0 ViaUp at (ox+28, oy+36), which reads L1 ViaUp.
    let _s111_i2_l1_via = place_l1_route!(ox + 28, oy + 36, TileType::ViaUp);
    let _s111_i2_l0_via = place!(ox + 28, oy + 36, TileType::ViaUp);
    push_pipeline!(_s111_i2_l1_via);
    push_pipeline!(_s111_i2_l0_via);

    // i=7: R enable at (ox+47, oy+36) is WireRight (placed in ALU loop above).
    //   It reads LEFT from L0 ViaUp at (ox+46, oy+36), which reads L1 WireRight.
    let _s111_i7_l0_via = place!(ox + 46, oy + 36, TileType::ViaUp);
    push_pipeline!(_s111_i7_l0_via);

    // -------------------------------------------------------------------------
    // Sprint 112: Physical carry ingress (Add/Sub class)
    // -------------------------------------------------------------------------
    // Or→Mux upgrade at L2 (ox+22, oy+50). Mux UP=use_shift selects:
    //   LEFT = software carry (SHL/SHR only), RIGHT = physical carry (bit 8).
    // And(ctrl_a[1], ctrl_a[2]) uniquely identifies Shl(110)/Shr(111) among 8 ops.
    //
    // Phase A: bit-8 extraction from wb_alu chain → L2 south+west to Mux RIGHT.
    // Tap 9-bit ALU result from WireDown at (ox+32, oy+48).
    let _s112_wd_49 = place!(ox + 32, oy + 49, TileType::WireDown);
    // Sprint 130: Remove old BitSelect circuit for bit-8 extraction.
    // Change BitSelect to Const(0).
    let _s130_bsel_old = place_val!(ox + 33, oy + 49, TileType::Const, 0);
    // Change bit position selector from Const(8) to Const(0).
    let _s130_bsel_pos_old = place_val!(ox + 34, oy + 49, TileType::Const, 0);
    // BitSelect output south chain removed.
    let _s130_wd_50_old = place_val!(ox + 33, oy + 50, TileType::Const, 0);
    let _s130_wd_51_old = place_val!(ox + 33, oy + 51, TileType::Const, 0);
    // L2 west from ox+31 to ox+24 at oy+52 (ox+23 is now ViaUp for selected carry).
    // Sprint 130: Shortened from ox+32 to ox+31 — ox+32 is now NOT(sel[0]) WireDown.
    for x in (ox + 24)..=(ox + 31) {
        let _ = place_l2_route!(x, oy + 52, TileType::WireLeft);
    }
    // L2 north from oy+51 to oy+50 at ox+23 → feeds carry Mux RIGHT.
    // Sprint 130: Change ox+23 from WireLeft to ViaUp for selected carry input.
    let _s130_carry_via = place_l2_route!(ox + 23, oy + 52, TileType::ViaUp);
    let _ = place_l2_route!(ox + 23, oy + 51, TileType::WireUp);
    let _ = place_l2_route!(ox + 23, oy + 50, TileType::WireUp);

    // Sprint 130: Extend ALU result south to oy+52 for CarryDetect input.
    let _s130_alu_wd_50 = place!(ox + 32, oy + 50, TileType::WireDown);
    let _s130_alu_wd_51 = place!(ox + 32, oy + 51, TileType::WireDown);
    let _s130_alu_wd_52 = place!(ox + 32, oy + 52, TileType::WireDown);

    // Sprint 130: West-side operand_a south column.
    // Source: L1 Tree A bus at oy+38 carries op_a. Tap via L2 ViaDown at (ox+29, oy+38).
    // L2 south from oy+38 to oy+43 bypasses L0 Const(MAX) at oy+40 and WireLeft at oy+41.
    // L0 south from oy+44 to oy+46 (free positions).
    // L1 hop at oy+46..48 bypasses WB ALU root mux WireRight at L0(ox+29, oy+47).
    // L0 south from oy+49 to oy+52, then WireRight east to CarryDetect_add LEFT.
    let _ = place_l2_route!(ox + 29, oy + 38, TileType::ViaDown);
    for y in (oy + 39)..=(oy + 43) {
        let _ = place_l2_route!(ox + 29, y, TileType::WireDown);
    }
    let _ = place_l1_route!(ox + 29, oy + 43, TileType::ViaUp);
    let _ = place!(ox + 29, oy + 43, TileType::ViaUp);
    for y in (oy + 44)..=(oy + 46) {
        let _ = place!(ox + 29, y, TileType::WireDown);
    }
    // L1 hop over WB ALU root mux bus at L0(ox+29, oy+47).
    let _ = place_l1_route!(ox + 29, oy + 46, TileType::ViaDown);
    let _ = place_l1_route!(ox + 29, oy + 47, TileType::WireDown);
    let _ = place_l1_route!(ox + 29, oy + 48, TileType::WireDown);
    let _ = place!(ox + 29, oy + 48, TileType::ViaUp);
    for y in (oy + 49)..=(oy + 52) {
        let _ = place!(ox + 29, y, TileType::WireDown);
    }
    // East step to (ox+30, oy+52) for CarryDetect_add LEFT input.
    let _s130_op_a_west_east = place!(ox + 30, oy + 52, TileType::WireRight);

    // Sprint 130: East-side operand_a south column (ox+33, oy+40..45).
    // Source: WireRight at (ox+33, oy+39) carries operand_a for Xor ALU tile.
    // L1 hop at oy+46..47 bypasses Const(MAX) and WireLeft (WB ALU) on L0.
    for y in (oy + 40)..=(oy + 45) {
        let _ = place!(ox + 33, y, TileType::WireDown);
    }
    let _ = place_l1_route!(ox + 33, oy + 45, TileType::ViaDown);
    let _ = place_l1_route!(ox + 33, oy + 46, TileType::WireDown);
    // L1 WireDown at oy+47 continues hop (was ViaDown — changed for L1 bypass).
    // L2 ViaDown at oy+47 reads L1 WireDown output = operand_a.
    let _s130_op_a_east_l1_via = place_l1_route!(ox + 33, oy + 47, TileType::WireDown);
    let _s130_op_a_east_l2_via = place_l2_route!(ox + 33, oy + 47, TileType::ViaDown);
    for y in (oy + 48)..=(oy + 52) {
        let _ = place_l2_route!(ox + 33, y, TileType::WireDown);
    }
    // L2 east step to (ox+34, oy+52).
    let _s130_op_a_east_l2_east = place_l2_route!(ox + 34, oy + 52, TileType::WireRight);
    // Drop from L2 to L0 at (ox+34, oy+52).
    let _s130_op_a_east_l1_via2 = place_l1_route!(ox + 34, oy + 52, TileType::ViaUp);
    let _s130_op_a_east_l0_via = place!(ox + 34, oy + 52, TileType::ViaUp);

    // Sprint 130: Two CarryDetect tiles for Add and Sub carry.
    let s130_carry_add_idx = place!(ox + 31, oy + 52, TileType::CarryDetect);
    // CarryDetect_sub replaces old WireDown at (ox+33, oy+52).
    let s130_carry_sub_idx = place!(ox + 33, oy + 52, TileType::CarryDetect);

    // Sprint 130: NOT(ctrl_a[0]) delivery for carry selection Mux.
    // L2 east bus at oy+49 carrying ctrl_a[0].
    for x in (ox + 24)..=(ox + 31) {
        let _ = place_l2_route!(x, oy + 49, TileType::WireRight);
    }
    // Zero tile inverts ctrl_a[0] → NOT(sel[0]).
    let _s130_not_sel0_zero = place_l2_route!(ox + 32, oy + 49, TileType::Zero);
    // L2 south column carrying NOT(sel[0]) — stops at oy+52 (oy+53 blocked by SHR carry L2 bus).
    for y in (oy + 50)..=(oy + 52) {
        let _ = place_l2_route!(ox + 32, y, TileType::WireDown);
    }
    // L1 hop over L2 SHR carry bus at oy+53:
    //   L1 ViaUp at oy+52 reads L2 NOT(sel[0]), L1 WireDown at oy+53 carries it south.
    let s130_not_sel0_l1_entry = place_l1_route!(ox + 32, oy + 52, TileType::ViaUp);
    let s130_not_sel0_l1 = place_l1_route!(ox + 32, oy + 53, TileType::WireDown);
    let s130_not_sel0_l0 = place!(ox + 32, oy + 53, TileType::ViaUp);

    // Sprint 130: Carry selection chains and Mux.
    let _s130_add_wd_53 = place!(ox + 31, oy + 53, TileType::WireDown);
    let _s130_sub_wd_53 = place!(ox + 33, oy + 53, TileType::WireDown);
    let _s130_add_wd_54 = place!(ox + 31, oy + 54, TileType::WireDown);
    let _s130_sub_wd_54 = place!(ox + 33, oy + 54, TileType::WireDown);
    let s130_carry_mux_idx = place!(ox + 32, oy + 54, TileType::Mux);

    // Sprint 130: Selected carry output delivery to L2 carry Mux.
    // South from Mux to L2 entry.
    let _s130_mux_wd_55 = place!(ox + 32, oy + 55, TileType::WireDown);
    let s130_mux_l1_via = place_l1_route!(ox + 32, oy + 55, TileType::ViaDown);
    let s130_mux_l2_via = place_l2_route!(ox + 32, oy + 55, TileType::ViaDown);
    // L2 west bus at oy+55 from ox+20 to ox+31.
    for x in (ox + 20)..=(ox + 31) {
        let _ = place_l2_route!(x, oy + 55, TileType::WireLeft);
    }
    // L2 north at ox+20 from oy+54 to oy+52 (west of SHR column at ox+21).
    let _ = place_l2_route!(ox + 20, oy + 54, TileType::WireUp);
    let _ = place_l2_route!(ox + 20, oy + 53, TileType::WireUp);
    let _ = place_l2_route!(ox + 20, oy + 52, TileType::WireUp);
    // L3 hop east over SHR carry column (L2 ox+21 oy+50..52 is WireUp).
    let _ = place_l3_route!(ox + 20, oy + 52, TileType::ViaDown);
    let _ = place_l3_route!(ox + 21, oy + 52, TileType::WireRight);
    let _ = place_l3_route!(ox + 22, oy + 52, TileType::WireRight);
    let _ = place_l3_route!(ox + 23, oy + 52, TileType::WireRight);

    // -------------------------------------------------------------------------
    // Sprint 131: Physical SHL carry — mux BitSelect position for SHL vs SHR.
    // SHR uses pos=B-1 (existing Sprint 123). SHL uses pos=8-B = 7-(B-1).
    // Position Mux at L1(ox+49, alu_row+1): LEFT=B-1, RIGHT=8-B, UP=ctrl_a[0].
    // -------------------------------------------------------------------------

    // Step 1: Deliver ctrl_a[0] to L1(ox+49, alu_row) for Mux UP input.
    // Source: Sprint 106 bit 0 L1 east bus at alu_row, ending at L1(ox+44, alu_row).
    // L1(ox+45..48) occupied by Sprint 123 relay tiles. L2 hop from ox+43 to ox+49.
    let _ = place_l2_route!(ox + 43, alu_row, TileType::ViaDown);
    for x in (ox + 44)..=(ox + 49) {
        let _ = place_l2_route!(x, alu_row, TileType::WireRight);
    }
    let _ = place_l1_route!(ox + 49, alu_row, TileType::ViaUp);

    // Step 2: Compute 8-B = Sub(7, B-1) for SHL carry position.
    // B-1 available at L1(ox+48, alu_row+1) = WireDown. Relay via L2 east to ox+52.
    let _ = place_l2_route!(ox + 48, alu_row + 1, TileType::ViaDown);
    for x in (ox + 49)..=(ox + 52) {
        let _ = place_l2_route!(x, alu_row + 1, TileType::WireRight);
    }
    // Drop B-1 from L2 to L0 at (ox+52, alu_row+1).
    let _ = place_l1_route!(ox + 52, alu_row + 1, TileType::ViaUp);
    let _ = place!(ox + 52, alu_row + 1, TileType::ViaUp);
    // Sub(7, B-1) at L0(ox+51, alu_row+1): LEFT=Const(7), RIGHT=B-1.
    let _ = place_val!(ox + 50, alu_row + 1, TileType::Const, 7);
    let _ = place!(ox + 51, alu_row + 1, TileType::Sub);

    // Step 3: Deliver 8-B to L1(ox+50, alu_row+1) for Mux RIGHT input.
    // Sub output at L0(ox+51) → L1 ViaDown → L1 WireLeft → Mux RIGHT.
    let _ = place_l1_route!(ox + 51, alu_row + 1, TileType::ViaDown);
    let _ = place_l1_route!(ox + 50, alu_row + 1, TileType::WireLeft);

    // Phase B: use_shift = And(ctrl_a[1], ctrl_a[2]).
    // Extend bit1 L2 south column (ox+18) from oy+43 to oy+48.
    for y in (oy + 43)..=(oy + 48) {
        let _ = place_l2_route!(ox + 18, y, TileType::WireDown);
    }
    // Tap bit2 from L1 east bus at (ox+20, oy+45) via L2 ViaDown, route south.
    let _ = place_l2_route!(ox + 20, oy + 45, TileType::ViaDown);
    for y in (oy + 46)..=(oy + 48) {
        let _ = place_l2_route!(ox + 20, y, TileType::WireDown);
    }
    // L2 And(bit1, bit2) at (ox+19, oy+48): LEFT=bit1, RIGHT=bit2.
    let _ = place_l2_route!(ox + 19, oy + 48, TileType::And);
    // And output south to oy+49, then east to ox+22 → Mux UP at (ox+22, oy+50).
    let _ = place_l2_route!(ox + 19, oy + 49, TileType::WireDown);
    for x in (ox + 20)..=(ox + 21) {
        let _ = place_l2_route!(x, oy + 49, TileType::WireRight);
    }
    // Sprint 131: WireRight passes is_shift directly (was And(is_shift, ctrl_a[0]) = is_shr).
    // Both SHL and SHR now use BitSelect carry path; position muxed on L1 by ctrl_a[0].
    let _ = place_l2_route!(ox + 22, oy + 49, TileType::WireRight);

    // -------------------------------------------------------------------------
    // Sprint 113: Physical NEG L override — "Physical NEG Autonomy"
    // -------------------------------------------------------------------------
    // Dedicated is_neg Mux16to1 LUT in decoder gap (ox+26..28) identifies NEG
    // (opcode 10) via opcode_shr selector. Routes to Sub tile's L enable at
    // (ox+21, oy+37) via L2 west+south+L1 hop. Tree B L2 bridge preserves
    // AddCarry R tree delivery at (ox+20, oy+37).
    //
    // Phase 1: is_neg LUT at (ox+27, oy+6..7)
    let _s113_lut_up = place_val!(
        ox + 27,
        oy + 6,
        TileType::Const,
        pack_8(&[0, 0, 0xFF, 0, 0, 0, 0, 0])
    );
    // LEFT data at (ox+26, oy+7) is existing guard Const(0) — correct for opcodes 0-7.
    let _s113_lut = place!(ox + 27, oy + 7, TileType::Mux16to1);
    let _s113_sel = place!(ox + 28, oy + 7, TileType::ViaUp);
    push_pipeline!(_s113_lut);
    push_pipeline!(_s113_sel);
    // opcode_shr WireDown drop at ox+28 from L1 east bus at oy+5.
    let _s113_shr_wd1 = place_l1_route!(ox + 28, oy + 6, TileType::WireDown);
    let _s113_shr_wd2 = place_l1_route!(ox + 28, oy + 7, TileType::WireDown);
    push_pipeline!(_s113_shr_wd1);
    push_pipeline!(_s113_shr_wd2);

    // Phase 2: L1/L2 route from LUT to L enable at (ox+21, oy+37).
    // Tap LUT output: L1 ViaDown + L2 ViaDown at (ox+27, oy+7).
    let _s113_tap_l1 = place_l1_route!(ox + 27, oy + 7, TileType::ViaDown);
    let _s113_tap_l2 = place_l2_route!(ox + 27, oy + 7, TileType::ViaDown);
    push_pipeline!(_s113_tap_l1);
    // L2 west at oy+7 from ox+26 to ox+14, with L1 hop over rd column at ox+23.
    // The rd packed field path occupies L2 ox+23 from oy+4 to oy+9 (WireDown).
    // East segment: WireLeft from ox+26 to ox+24 (before rd column).
    for x in (ox + 24)..=(ox + 26) {
        let _ = place_l2_route!(x, oy + 7, TileType::WireLeft);
    }
    // L1 hop at oy+6 over rd column: L2 WireUp → L1 ViaUp → L1 WireLeft×2 → L2 ViaDown → L2 WireDown.
    let _ = place_l2_route!(ox + 24, oy + 6, TileType::WireUp);
    let _ = place_l1_route!(ox + 24, oy + 6, TileType::ViaUp);
    let _ = place_l1_route!(ox + 23, oy + 6, TileType::WireLeft);
    let _ = place_l1_route!(ox + 22, oy + 6, TileType::WireLeft);
    let _ = place_l2_route!(ox + 22, oy + 6, TileType::ViaDown);
    let _ = place_l3_route!(ox + 22, oy + 6, TileType::ViaDown);
    // West segment on L3 y=6 avoids the byte2/byte3 trunks now occupying L2 y=7.
    for x in (ox + 14)..=(ox + 21) {
        let _ = place_l3_route!(x, oy + 6, TileType::WireLeft);
    }
    let _ = place_l2_route!(ox + 14, oy + 6, TileType::ViaUp);
    let _ = place_l2_route!(ox + 14, oy + 7, TileType::WireDown);
    // L2 south at ox+14 from oy+8 to oy+35 (28 tiles).
    for y in (oy + 8)..=(oy + 35) {
        let _ = place_l2_route!(ox + 14, y, TileType::WireDown);
    }
    // L1 hop at oy+35: ViaUp + 6 WireRight (ox+15..20).
    let _s113_hop_l1 = place_l1_route!(ox + 14, oy + 35, TileType::ViaUp);
    for x in (ox + 15)..=(ox + 20) {
        let _ = place_l1_route!(x, oy + 35, TileType::WireRight);
    }
    // Re-enter L2: ViaDown + WireRight + 2 WireDown to delivery at (ox+21, oy+37).
    let _s113_reenter_l2 = place_l2_route!(ox + 20, oy + 35, TileType::ViaDown);
    let _ = place_l2_route!(ox + 21, oy + 35, TileType::WireRight);
    let _ = place_l2_route!(ox + 21, oy + 36, TileType::WireDown);
    let _ = place_l2_route!(ox + 21, oy + 37, TileType::WireDown);

    // Phase 3: Tree B L2 bridge — reroute Tree B around replaced L1 tiles.
    // Tap Tree B from L1(ox+22, oy+37) via L2, route south+west+north to (ox+20, oy+37).
    let _ = place_l2_route!(ox + 22, oy + 37, TileType::ViaDown);
    let _ = place_l2_route!(ox + 22, oy + 38, TileType::WireDown);
    let _ = place_l2_route!(ox + 21, oy + 38, TileType::WireLeft);
    let _ = place_l2_route!(ox + 20, oy + 38, TileType::WireLeft);
    let _ = place_l2_route!(ox + 20, oy + 37, TileType::WireUp);
    // L1 changes: WireLeft → ViaUp for Tree B bridge and is_neg delivery.
    // These REPLACE the existing WireLeft tiles placed at line ~1375.
    // L1(ox+20, oy+37) → ViaUp reads L2 WireUp = Tree B.
    // L1(ox+21, oy+37) → ViaUp reads L2 WireDown = is_neg.
    sim.set_tile_3d(ox + 20, oy + 37, 1, TileType::ViaUp);
    sim.set_tile_3d(ox + 21, oy + 37, 1, TileType::ViaUp);

    // -------------------------------------------------------------------------
    // Sprint 114: Physical ALU R override value (ir_low delivery)
    // -------------------------------------------------------------------------
    // Delivers ir_low from L2 ox+1 south column via L2 east bus at oy+33,
    // with L1 hop over Sprint 106/113 L2 columns, then L0 drops to all 8
    // R override positions at oy+37. Eliminates ~0.30 writes/tick (9 opcodes).

    // Phase A: L2 east bus at oy+33, west segment (ox+2 to ox+13)
    for x in (ox + 2)..=(ox + 13) {
        let _ = place_l2_route!(x, oy + 33, TileType::WireRight);
    }

    // Phase B: L1 hop at oy+33 over Sprint 106/113 L2 columns (ox+14,17,18,19)
    let _ = place_l1_route!(ox + 13, oy + 33, TileType::ViaUp); // L2→L1
    for x in (ox + 14)..=(ox + 20) {
        let _ = place_l1_route!(x, oy + 33, TileType::WireRight);
    }
    let _ = place_l2_route!(ox + 20, oy + 33, TileType::ViaDown); // L1→L2

    // Phase C: L2 east bus at oy+33, east segment (ox+21 to ox+46)
    for x in (ox + 21)..=(ox + 46) {
        let _ = place_l2_route!(x, oy + 33, TileType::WireRight);
    }

    // Phase D: Drops to L0 + WireDown chains for all 8 R override positions
    let alu_r_columns: [usize; 8] = std::array::from_fn(|i| ox + 18 + i * 4);

    for i in 0..8 {
        let col = alu_r_columns[i];

        if i == 0 {
            // i=0: L1 at this column is part of the hop chain (WireRight).
            // L0 ViaUp reads L1 WireRight = ir_low. No separate L1 ViaUp needed.
            let _ = place!(col, oy + 33, TileType::ViaUp); // L0 ViaUp reads L1
        } else if i == 4 {
            // i=4: L1 at ox+34 is occupied by Sprint 111 WireDown (use_imm signal).
            // Offset drop to ox+35, then L0 WireLeft jog to ox+34.
            let _ = place_l1_route!(col + 1, oy + 33, TileType::ViaUp); // L1 ViaUp at ox+35
            let _ = place!(col + 1, oy + 33, TileType::ViaUp); // L0 ViaUp at ox+35
            let _ = place!(col, oy + 33, TileType::WireLeft); // L0 WireLeft at ox+34
        } else {
            // Standard: L1 ViaUp + L0 ViaUp at the R override column
            let _ = place_l1_route!(col, oy + 33, TileType::ViaUp); // L1 ViaUp reads L2
            let _ = place!(col, oy + 33, TileType::ViaUp); // L0 ViaUp reads L1
        }

        // L0 WireDown chain from oy+34 to oy+36 (replaces guard Const(0) tiles).
        // Sprint 237 fix: column ox+26 (i=2) has a conflict at oy+35 where the
        // Top Mux A ViaUp was placed. Detour the ir_low chain via column ox+25:
        //   WD(26,34) → WL(25,34) → WD(25,35) → WD(25,36) → WR(26,36) → WD(26,37).
        if i == 2 {
            let _ = place!(col, oy + 34, TileType::WireDown);
            // (col, oy+35) = Sprint 237 ViaUp — do NOT overwrite
            let _ = place!(col - 1, oy + 34, TileType::WireLeft); // reads RIGHT=(26,34)
            let _ = place!(col - 1, oy + 35, TileType::WireDown); // reads UP=(25,34)
            let _ = place!(col - 1, oy + 36, TileType::WireDown); // reads UP=(25,35)
            let _ = place!(col, oy + 36, TileType::WireRight); // reads LEFT=(25,36)
        } else {
            for y in (oy + 34)..=(oy + 36) {
                let _ = place!(col, y, TileType::WireDown);
            }
        }
        // Note: oy+37 (the R override position) was already changed to WireDown
        // in Step 1 above (the ALU loop tile conversion).
    }

    // -------------------------------------------------------------------------
    // Sprint 104: physical RAM read 8:1 tree (LD/LDB source)
    // -------------------------------------------------------------------------
    // Compact L0 layout (inside guarded oy..oy+63):
    //   RAM cells:    y+54, x=(70,72,74,76,78,80,82,84)
    //   Leaf muxes:   y+56, x=(71,75,79,83), selectors at y+55
    //   Mid muxes:    y+59, x=(73,81), selectors at y+58
    //   Root mux:     y+62, x=77, selector at y+61
    //
    // Mux polarity: up!=0 selects LEFT.
    // Sprint 108: addr source is now the physical Mux at (ox+86, oy+53).
    // WireDown chain carries addr value from Mux to BitSelect extraction.
    let ram_cols = [70usize, 72, 74, 76, 78, 80, 82, 84];
    let leaf_cols = [71usize, 75, 79, 83];

    // ---- Sprint 106: physical RAM read selector (addr[2:0] extraction) ----
    // Sprint 108: replaced software-written Const with WireDown from Mux chain.
    // BitSelect×3 + Zero×3 inversion pipeline unchanged.
    // L1 distribution of inverted bits to 7 ViaUp selector sites.
    let s108_addr_relay_55 = place!(ox + 86, oy + 55, TileType::WireDown);
    push_pipeline!(s108_addr_relay_55);

    // Bit 0 extraction: BitSelect(addr, 0) → WireDown → Zero (invert)
    let _s106r_bs0 = place!(ox + 87, oy + 55, TileType::BitSelect);
    let _s106r_bp0 = place_val!(ox + 88, oy + 55, TileType::Const, 0);
    let _s106r_wd0a = place!(ox + 86, oy + 56, TileType::WireDown);
    let _s106r_wd0b = place!(ox + 87, oy + 56, TileType::WireDown);
    let _s106r_inv0 = place!(ox + 88, oy + 56, TileType::Zero);

    // Bit 1 extraction
    let _s106r_wd1a = place!(ox + 86, oy + 57, TileType::WireDown);
    let _s106r_bs1 = place!(ox + 87, oy + 57, TileType::BitSelect);
    let _s106r_bp1 = place_val!(ox + 88, oy + 57, TileType::Const, 1);
    let _s106r_wd1b = place!(ox + 86, oy + 58, TileType::WireDown);
    let _s106r_wd1c = place!(ox + 87, oy + 58, TileType::WireDown);
    let _s106r_inv1 = place!(ox + 88, oy + 58, TileType::Zero);

    // Bit 2 extraction
    let _s106r_wd2a = place!(ox + 86, oy + 59, TileType::WireDown);
    let _s106r_bs2 = place!(ox + 87, oy + 59, TileType::BitSelect);
    let _s106r_bp2 = place_val!(ox + 88, oy + 59, TileType::Const, 2);
    let _s106r_wd2b = place!(ox + 87, oy + 60, TileType::WireDown);
    let _s106r_inv2 = place!(ox + 88, oy + 60, TileType::Zero);

    // L1 distribution: inverted bit 0 → oy+55 WireLeft bus → 4 leaf ViaUp
    let _s106r_b0_tap = place_l1_route!(ox + 88, oy + 56, TileType::ViaDown);
    let _s106r_b0_up = place_l1_route!(ox + 88, oy + 55, TileType::WireUp);
    for x in (ox + 71)..=(ox + 87) {
        let _idx = place_l1_route!(x, oy + 55, TileType::WireLeft);
    }

    // L1 distribution: inverted bit 1 → oy+58 WireLeft bus → 2 mid ViaUp
    let _s106r_b1_tap = place_l1_route!(ox + 88, oy + 58, TileType::ViaDown);
    for x in (ox + 73)..=(ox + 87) {
        let _idx = place_l1_route!(x, oy + 58, TileType::WireLeft);
    }

    // L1 distribution: inverted bit 2 → oy+61 WireLeft bus → 1 root ViaUp
    let _s106r_b2_tap = place_l1_route!(ox + 88, oy + 60, TileType::ViaDown);
    let _s106r_b2_dn = place_l1_route!(ox + 88, oy + 61, TileType::WireDown);
    for x in (ox + 77)..=(ox + 87) {
        let _idx = place_l1_route!(x, oy + 61, TileType::WireLeft);
    }

    // RAM outputs descend into leaf data anchors.
    for &col in &ram_cols {
        for y in (oy + 55)..=(oy + 56) {
            let _idx = place!(ox + col, y, TileType::WireDown);
        }
    }

    // Leaf selectors (ViaUp from L1 inverted bit 0) and muxes.
    for &col in &leaf_cols {
        let _via = place!(ox + col, oy + 55, TileType::ViaUp);
        let _idx = place!(ox + col, oy + 56, TileType::Mux);
    }

    // Leaf outputs descend to mid layer.
    for &col in &leaf_cols {
        for y in (oy + 57)..=(oy + 59) {
            let _idx = place!(ox + col, y, TileType::WireDown);
        }
    }

    // Mid selectors (ViaUp from L1 inverted bit 1), input shims, and muxes.
    let _sel_mid_l = place!(ox + 73, oy + 58, TileType::ViaUp);
    let _sel_mid_r = place!(ox + 81, oy + 58, TileType::ViaUp);

    for &(x, tt) in &[
        (72usize, TileType::WireRight),
        (74usize, TileType::WireLeft),
        (80usize, TileType::WireRight),
        (82usize, TileType::WireLeft),
    ] {
        let _idx = place!(ox + x, oy + 59, tt);
    }

    let _mid_left_idx = place!(ox + 73, oy + 59, TileType::Mux);
    let _mid_right_idx = place!(ox + 81, oy + 59, TileType::Mux);

    // Mid outputs descend to root row.
    for &col in &[73usize, 81usize] {
        for y in (oy + 60)..=(oy + 62) {
            let _idx = place!(ox + col, y, TileType::WireDown);
        }
    }

    // Root selector (ViaUp from L1 inverted bit 2), fan-in shims, and root mux.
    let _sel_root = place!(ox + 77, oy + 61, TileType::ViaUp);
    for x in (74usize)..=(76usize) {
        let _idx = place!(ox + x, oy + 62, TileType::WireRight);
    }
    for &x in &[80usize, 79usize, 78usize] {
        let _idx = place!(ox + x, oy + 62, TileType::WireLeft);
    }
    let _ram_read_root_idx = place!(ox + 77, oy + 62, TileType::Mux);

    // Sprint 122: Physical RAM read → wb_data route.
    // L1+L2 ViaDown tap L0 RAM root output at (ox+77, oy+62).
    // L2 north column at ox+77 (oy+10..61) carries signal upward.
    // L2 west bus at oy+10 (ox+46..76) delivers to outer Mux LEFT at (ox+46, oy+9).
    // L1 hop over ctrl_b south column at L2(ox+61, oy+10).
    let _s122_ram_l1_tap = place_l1_route!(ox + 77, oy + 62, TileType::ViaDown);
    let _s122_ram_l2_tap = place_l2_route!(ox + 77, oy + 62, TileType::ViaDown);
    // L2 north column at ox+77 (oy+10..61).
    // Hop over B0.byte3's L2 east run at oy+17 using a short L3 splice.
    for y in (oy + 18)..=(oy + 61) {
        let _ = place_l2_route!(ox + 77, y, TileType::WireUp);
    }
    let _s122_ram_l3_entry = place_l3_route!(ox + 77, oy + 18, TileType::ViaDown);
    let _s122_ram_l3_wu1 = place_l3_route!(ox + 77, oy + 17, TileType::WireUp);
    let _s122_ram_l3_wu2 = place_l3_route!(ox + 77, oy + 16, TileType::WireUp);
    let _s122_ram_l2_resume = place_l2_route!(ox + 77, oy + 16, TileType::ViaUp);
    for y in (oy + 10)..=(oy + 15) {
        let _ = place_l2_route!(ox + 77, y, TileType::WireUp);
    }
    // L2 west bus at oy+10 (ox+46..76). Uses Sprint 119 freed corridor.
    // Split into segments with an L1 bridge over the byte sink ascents at
    // L2(ox+70) and L2(ox+73), plus the existing ctrl_b south-column hop.
    let _s122_b2_hop_l1_entry = place_l1_route!(ox + 77, oy + 10, TileType::ViaUp);
    for x in (ox + 69)..=(ox + 76) {
        let _ = place_l1_route!(x, oy + 10, TileType::WireLeft);
    }
    let _s122_b2_hop_l2_reentry = place_l2_route!(ox + 69, oy + 10, TileType::ViaDown);
    for x in (ox + 62)..=(ox + 68) {
        let _ = place_l2_route!(x, oy + 10, TileType::WireLeft);
    }
    // L1 hop over ctrl_b south column at ox+61:
    // L1 ViaUp(62,10) reads L2 → L1 WireLeft(61,10) → L1 WireLeft(60,10) → L2 ViaDown(60,10) reads L1.
    let _s122_hop_l1_entry = place_l1_route!(ox + 62, oy + 10, TileType::ViaUp);
    let _s122_hop_l1_mid = place_l1_route!(ox + 61, oy + 10, TileType::WireLeft);
    let _s122_hop_l1_exit = place_l1_route!(ox + 60, oy + 10, TileType::WireLeft);
    let _s122_hop_l2_reentry = place_l2_route!(ox + 60, oy + 10, TileType::ViaDown);
    // West segment: L2 WireLeft from ox+46 to ox+59.
    for x in (ox + 46)..=(ox + 59) {
        let _ = place_l2_route!(x, oy + 10, TileType::WireLeft);
    }

    // -------------------------------------------------------------------------
    // Sprint 105: physical RAM write ingress (ST/STB source)
    // -------------------------------------------------------------------------
    // Address decode + one-hot WE fanout + L1 data broadcast into RAM.LEFT.
    // This avoids collisions with Sprint 104 RAM read tree (y+55..y+62).

    // Control bridge — Sprint 117: address + enable are now physical ViaUp (not software Const).
    let _s117_addr_via_l0 = place!(ox + 90, oy + 45, TileType::ViaUp);
    let ram_write_decode_idx = place!(ox + 91, oy + 45, TileType::Decoder3to8);
    let _s117_enable_via_l0 = place!(ox + 93, oy + 45, TileType::ViaUp);
    let ram_write_gate_idx = place!(ox + 92, oy + 45, TileType::And);
    // Lift gated one-hot one row up to keep row y+46 free for WE extraction.
    let _trunk_up_idx = place!(ox + 92, oy + 44, TileType::WireUp);

    // One-hot distribution bus on row y+44 (westward from lifted trunk).
    for x in (69usize..=91usize).rev() {
        let _idx = place!(ox + x, oy + 44, TileType::WireLeft);
    }

    // Vertical one-hot trunks on odd columns down to y+53.
    for col in (69usize..=83usize).step_by(2) {
        for y in (oy + 45)..=(oy + 53) {
            let _idx = place!(ox + col, y, TileType::WireDown);
        }
    }

    // Per-cell WE extraction (high->low rows avoids odd-column aliasing).
    let we_rows = [53usize, 52, 51, 50, 49, 48, 47, 46];
    for i in 0..8usize {
        let ram_col = 70usize + i * 2;
        let row = oy + we_rows[i];

        let _we_bs_idx = place!(ox + ram_col, row, TileType::BitSelect);
        let _we_bit_idx = place_val!(ox + ram_col + 1, row, TileType::Const, i as u64);

        if row < (oy + 53) {
            for y in (row + 1)..=(oy + 53) {
                let _idx = place!(ox + ram_col, y, TileType::WireDown);
            }
        }
    }

    // L1 data broadcast to RAM.LEFT inputs via L0 ViaUp tiles at odd columns.
    // Sprint 117: data source is now physical WireDown (reads from L1 tree_a above).
    reserve_l1!(ox + 90, oy + 54);
    let _s117_data_l1 = place_l1!(ox + 90, oy + 54, TileType::WireDown);
    for x in (ox + 69)..=(ox + 89) {
        let _idx = place_l1_route!(x, oy + 54, TileType::WireLeft);
    }
    for col in (69usize..=83usize).step_by(2) {
        let _idx = place!(ox + col, oy + 54, TileType::ViaUp);
    }

    // -------------------------------------------------------------------------
    // Sprint 116: Physical wb_data override value — "The Cascade"
    // -------------------------------------------------------------------------
    // Two-level L2 Mux cascade replaces the software Const at (ox+37, oy+10).
    // Inner Mux (L2 ox+47 oy+33): UP=use_imm → LEFT=ir_low, RIGHT=Tree_B.
    // Outer Mux (L2 ox+47 oy+9):  UP=ctrl_b  → LEFT=sw_Const, RIGHT=inner.
    //   ctrl_b=0 (MOV/LDI): RIGHT = inner cascade (physical, 0 writes).
    //   ctrl_b≠0 (LD/LDB):  LEFT  = sw Const (1 write).
    //   ctrl_b≠0 (branches): LEFT (stale), but WE=0 → harmless.
    //
    // Delivery: L0 ViaUp(ox+38, oy+11) → L0 WireLeft(ox+37, oy+11) →
    //   L0 WireUp(ox+37, oy+10) → wb_data Mux LEFT.
    // Two L1 hops cross the L2 ctrl_a wall at oy+10 (ox+23..60):
    //   ox+47: outer Mux output (L2 oy+9 → L1 south → L2 oy+11)
    //   ox+48: inner cascade    (L2 oy+11 → L1 north → L2 oy+9)

    // --- Phase 1: L0/L1 delivery chain through (ox+38, oy+11) ---
    // L0 WireUp at (ox+37, oy+10) already placed above (reads DOWN = oy+11).
    let s116_wireleft_l0 = place!(ox + 37, oy + 11, TileType::WireLeft);
    reg_wb_dirty_indices.push(s116_wireleft_l0);
    let s116_via_up_l0 = place!(ox + 38, oy + 11, TileType::ViaUp);
    reg_wb_dirty_indices.push(s116_via_up_l0);
    let s116_via_up_l1 = place_l1_route!(ox + 38, oy + 11, TileType::ViaUp);
    reg_wb_dirty_indices.push(s116_via_up_l1);

    // --- Phase 2: Outer Mux at L2 (ox+47, oy+9) ---
    // UP = ctrl_b from Sprint 107 L2 highway WireRight at (ox+47, oy+8).
    // LEFT = physical RAM read data (Sprint 122: WireUp reads L2 west bus at oy+10).
    // RIGHT = inner cascade output via L1 hop at ox+48 (Phase 3).
    // Sprint 122: Const replaced by WireUp — RAM data arrives physically from L2(46,10).
    let _s122_wb_data_ram_via = place_l2_route!(ox + 46, oy + 9, TileType::WireUp);
    let wb_data_l2_mux_idx = place_l2!(ox + 47, oy + 9, TileType::Mux);
    reg_wb_dirty_indices.push(wb_data_l2_mux_idx);

    // --- Phase 3: Inner cascade L1 hop (ox+48, L2 oy+11 → L2 oy+9) ---
    // Inner output arrives at L2 (ox+48, oy+11) WireUp from Phase 6.
    // L1 hop crosses the L2 ctrl_a wall at oy+10.
    let _s116_inner_l1_via = place_l1_route!(ox + 48, oy + 11, TileType::ViaUp);
    let _s116_inner_l1_10 = place_l1_route!(ox + 48, oy + 10, TileType::WireUp);
    let _s116_inner_l1_09 = place_l1_route!(ox + 48, oy + 9, TileType::WireUp);
    let _s116_inner_l2_entry = place_l2!(ox + 48, oy + 9, TileType::ViaDown);

    // --- Phase 4: Inner Mux at L2 (ox+47, oy+33) ---
    // UP=use_imm, LEFT=ir_low (Sprint 114 L2 east bus end), RIGHT=Tree B root.
    // Sprint 114 L2 east bus ends at (ox+46, oy+33) with WireRight.
    let _s116_inner_mux = place_l2!(ox + 47, oy + 33, TileType::Mux);
    // Tree B root: L1 opB bus at (ox+48, oy+32) = WireRight. L2 ViaDown reads L1.
    let _s116_tree_b_via = place_l2!(ox + 48, oy + 32, TileType::ViaDown);
    let _s116_tree_b_wd = place_l2!(ox + 48, oy + 33, TileType::WireDown);

    // --- Phase 5: use_imm delivery to inner Mux UP ---
    // Sprint 111 L1 WireDown column at ox+34 carries use_imm south (oy+15..36).
    // Tap at oy+32 via L2 ViaDown, then L2 WireRight east to ox+47.
    let _s116_uimm_via = place_l2!(ox + 34, oy + 32, TileType::ViaDown);
    for x in (ox + 35)..=(ox + 47) {
        let _ = place_l2!(x, oy + 32, TileType::WireRight);
    }

    // --- Phase 6: Inner output route ---
    // Route inner Mux output from L2 (ox+47, oy+33) to Phase 3 entry at
    // L2 (ox+48, oy+11) via an L1 corridor at ox+57.
    //
    // BFS auto-router places tiles across L2 in roughly oy+12..18 / ox+48..60.
    // A pure L2 north column would silently overwrite BFS routes, breaking
    // cross-bank register delivery to Tree B. Instead, route through L1
    // (which BFS never uses) with L2 hops over L1 obstacles.
    //
    // L1 column ox+57 has TWO L1 obstacles:
    //   oy+32: opB bus (Sprint 108, WireRight)
    //   oy+14: use_imm east (Sprint 108, WireRight)
    // Both have guard Const on L2, allowing L2 hops.
    //
    // Signal path:
    //   L2 inner Mux (47,33) → L2 south to (47,34) → L2 east to (57,34)
    //   → L1 ViaUp (57,34) → L1 WireUp (57,33)
    //   → L2 hop over opB bus: ViaDown(57,33) → WireUp(57,32..31)
    //   → L1 ViaUp (57,31) → L1 WireUp (57,30..16)
    //   → L1 WireUp (57,15)
    //   → L2 hop over use_imm: ViaDown(57,15) → WireUp(57,14..13)
    //   → L1 ViaUp (57,13) → L1 WireUp (57,12..11)
    //   → L2 ViaDown (57,11) → L2 WireLeft (56..48,11)
    //   → L1 ViaUp (48,11) [Phase 3]

    // L2 south + east from inner Mux to ox+57
    let _s116_inner_out_s = place_l2!(ox + 47, oy + 34, TileType::WireDown);
    for x in (ox + 48)..=(ox + 57) {
        let _ = place_l2!(x, oy + 34, TileType::WireRight);
    }

    // Enter L1 at (ox+57, oy+34)
    let _ = place_l1_route!(ox + 57, oy + 34, TileType::ViaUp);
    let _ = place_l1_route!(ox + 57, oy + 33, TileType::WireUp);

    // L2 hop #1: over L1 opB bus at oy+32 (L1 ox+57 oy+32 = WireRight)
    let _ = place_l2!(ox + 57, oy + 33, TileType::ViaDown);
    let _ = place_l2!(ox + 57, oy + 32, TileType::WireUp);
    let _ = place_l2!(ox + 57, oy + 31, TileType::WireUp);
    let _ = place_l1_route!(ox + 57, oy + 31, TileType::ViaUp);

    // L1 north from oy+30 to oy+16 (between opB bus and use_imm)
    for y in (oy + 16)..=(oy + 30) {
        let _ = place_l1_route!(ox + 57, y, TileType::WireUp);
    }
    let _ = place_l1_route!(ox + 57, oy + 15, TileType::WireUp);

    // L2 hop #2: over L1 use_imm east at oy+14 (L1 ox+57 oy+14 = WireRight)
    let _ = place_l2!(ox + 57, oy + 15, TileType::ViaDown);
    let _ = place_l2!(ox + 57, oy + 14, TileType::WireUp);
    let _ = place_l2!(ox + 57, oy + 13, TileType::WireUp);
    let _ = place_l1_route!(ox + 57, oy + 13, TileType::ViaUp);

    // L1 north from oy+12 to oy+11 (above use_imm, into delivery zone)
    let _ = place_l1_route!(ox + 57, oy + 12, TileType::WireUp);
    let _ = place_l1_route!(ox + 57, oy + 11, TileType::WireUp);

    // Exit L1 back to L2 at (ox+57, oy+11)
    let _ = place_l2!(ox + 57, oy + 11, TileType::ViaDown);

    // L2 west bus from ox+56 to ox+48 at oy+11 — delivers to Phase 3
    for x in (ox + 48)..=(ox + 56) {
        let _ = place_l2!(x, oy + 11, TileType::WireLeft);
    }

    // --- Phase 7: Outer Mux output L1 hop (ox+47, L2 oy+9 → L2 oy+11) ---
    // L1 ViaUp reads L2 Mux output. WireDown south through oy+10 wall.
    // L2 ViaDown at oy+11 re-enters L2 for the west delivery bus.
    let _s116_outer_l1_via = place_l1_route!(ox + 47, oy + 9, TileType::ViaUp);
    let _s116_outer_l1_10 = place_l1_route!(ox + 47, oy + 10, TileType::WireDown);
    let _s116_outer_l1_11 = place_l1_route!(ox + 47, oy + 11, TileType::WireDown);
    let _s116_outer_l2_entry = place_l2!(ox + 47, oy + 11, TileType::ViaDown);

    // --- Phase 8: L2 west bus at oy+11 (ox+38..46) ---
    // Delivers outer Mux output to L1/L0 ViaUp chain at (ox+38, oy+11).
    // Sprint 185: skip x=39,40,42,43 — B2 byte2/byte3 L2 descents cross y=11 there.
    // L1 hops bridge the gaps to maintain chain continuity.
    for x in (ox + 38)..=(ox + 46) {
        if x == ox + 39 || x == ox + 40 || x == ox + 42 || x == ox + 43 {
            continue;
        }
        let _ = place_l2!(x, oy + 11, TileType::WireLeft);
    }
    // Sprint 185: L1 hop over B2.byte3 descent at x=43
    // Chain: L2(44,11)→L1(44,11)→L1(43,11)→L1(42,11)→L2(42,11)
    let _ = place_l1_route!(ox + 44, oy + 11, TileType::ViaUp);
    let _ = place_l1_route!(ox + 43, oy + 11, TileType::WireLeft);
    let _ = place_l1_route!(ox + 42, oy + 11, TileType::WireLeft);
    let _ = place_l2!(ox + 42, oy + 11, TileType::ViaDown);
    // Sprint 185: L1 hop over B2.byte2 descent at x=40
    // Chain: L2(41,11)→L1(41,11)→L1(40,11)→L1(39,11)→L2(39,11)
    let _ = place_l1_route!(ox + 41, oy + 11, TileType::ViaUp);
    let _ = place_l1_route!(ox + 40, oy + 11, TileType::WireLeft);
    let _ = place_l1_route!(ox + 39, oy + 11, TileType::WireLeft);
    let _ = place_l2!(ox + 39, oy + 11, TileType::ViaDown);

    // -------------------------------------------------------------------------
    // Sprint 117: Physical RAM write — "Memory Autonomy"
    // -------------------------------------------------------------------------
    // Physicalizes all 3 RAM write software bridges (address, data, enable),
    // eliminating set_ram_write_addr/data/enable and prev_ram_write_enabled.
    //
    // Phase A: Route RAM read address Mux output (ox+86,oy+53) to Decoder3to8
    //   input at (ox+90,oy+45) via L1+L2 lift → L2 east → L2 north → L1/L0 ViaUp.
    // Phase B: L1 east at oy+38 to ox+86, north to oy+33, west to ox+85,
    //   L2 hop over opB bus at oy+32, east to ox+91, south to oy+53, west to ox+90.
    // Phase C: Extract ctrl_b[4] (mem_write) via BitSelect at L2 (ox+62,oy+11),
    //   route L2 east at oy+13 → south at ox+93 to (ox+93,oy+45) for enable.

    // --- Phase A: Physical RAM write address (~16 tiles) ---
    // Source: L0 Mux at (ox+86, oy+53) = use_imm ? imm8 : opB.
    // Destination: L0 ViaUp at (ox+90, oy+45) → Decoder3to8 LEFT.
    let _s117_addr_l1_tap = place_l1_route!(ox + 86, oy + 53, TileType::ViaDown);
    let _s117_addr_l2_tap = place_l2_route!(ox + 86, oy + 53, TileType::ViaDown);
    for x in (ox + 87)..=(ox + 90) {
        let _ = place_l2_route!(x, oy + 53, TileType::WireRight);
    }
    for y in (oy + 45)..=(oy + 52) {
        let _ = place_l2_route!(ox + 90, y, TileType::WireUp);
    }
    let _s117_addr_l1_drop = place_l1_route!(ox + 90, oy + 45, TileType::ViaUp);

    // --- Phase B: Physical RAM write data (~79 tiles) ---
    // Source: tree_a L1 bus at oy+38 (ends at ox+46).
    // Destination: L1 WireDown at (ox+90, oy+54) reads from (ox+90, oy+53).
    // Route: L1 east at oy+38 to ox+86, L1 north at ox+86 to oy+33 (above opB
    //   south start), L1 west to ox+85, L2 hop over opB east bus at oy+32
    //   (ViaDown→WireUp→WireUp at ox+85), L1 ViaUp at (ox+85,oy+31), L1 east
    //   at oy+31 to ox+91, L1 south at ox+91 to oy+53, L1 west to (ox+90,oy+53).
    for x in (ox + 47)..=(ox + 86) {
        let idx = place_l1_route!(x, oy + 38, TileType::WireRight);
        push_pipeline!(idx);
    }
    for y in (oy + 33)..=(oy + 37) {
        let idx = place_l1_route!(ox + 86, y, TileType::WireUp);
        push_pipeline!(idx);
    }
    {
        let idx = place_l1_route!(ox + 85, oy + 33, TileType::WireLeft);
        push_pipeline!(idx);
    }
    // L2 hop over opB east bus at oy+32 (ox+48..87 blocks L1).
    let _s117_data_l2_hop_entry = place_l2_route!(ox + 85, oy + 33, TileType::ViaDown);
    push_pipeline!(_s117_data_l2_hop_entry);
    let _s117_data_l2_hop_mid = place_l2_route!(ox + 85, oy + 32, TileType::WireUp);
    push_pipeline!(_s117_data_l2_hop_mid);
    let _s117_data_l2_hop_exit = place_l2_route!(ox + 85, oy + 31, TileType::WireUp);
    push_pipeline!(_s117_data_l2_hop_exit);
    {
        let idx = place_l1_route!(ox + 85, oy + 31, TileType::ViaUp);
        push_pipeline!(idx);
    }
    for x in (ox + 86)..=(ox + 91) {
        let idx = place_l1_route!(x, oy + 31, TileType::WireRight);
        push_pipeline!(idx);
    }
    for y in (oy + 32)..=(oy + 53) {
        let idx = place_l1_route!(ox + 91, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    {
        let idx = place_l1_route!(ox + 90, oy + 53, TileType::WireLeft);
        push_pipeline!(idx);
    }

    // --- Phase C: Physical RAM write enable (~69 tiles) ---
    // Source: ctrl_b[4] (mem_write) from Sprint 107 L2 south column at ox+61.
    // BitSelect extracts bit 4, route east at oy+13 then south at ox+93.
    // Destination: L0 ViaUp at (ox+93, oy+45) → And gate RIGHT.
    // Sprint 121: Head stays on L2 only for the BitSelect read from ctrl_b.
    // Route body drops immediately to L3 to clear the byte-delivery rows.
    let _s117_en_bs = place_l2_route!(ox + 62, oy + 11, TileType::BitSelect);
    let _s117_en_const = place_l2_val!(ox + 63, oy + 11, TileType::Const, 4);
    reserve_l2!(ox + 63, oy + 11); // manual reserve for place_l2_val
    let s121_en_l3_entry = place_l3!(ox + 62, oy + 11, TileType::ViaDown);
    push_pipeline!(s121_en_l3_entry);
    let s121_en_l3_wd1 = place_l3!(ox + 62, oy + 12, TileType::WireDown);
    let s121_en_l3_wd2 = place_l3!(ox + 62, oy + 13, TileType::WireDown);
    push_pipeline!(s121_en_l3_wd1);
    push_pipeline!(s121_en_l3_wd2);
    // Sprint 121: L3 east at oy+13 from ox+63 to ox+93. [was L2]
    for x in (ox + 63)..=(ox + 93) {
        let _ = place_l3!(x, oy + 13, TileType::WireRight);
    }
    // Sprint 121: L3 south at ox+93 from oy+14 to oy+45. [was L2]
    for y in (oy + 14)..=(oy + 45) {
        let _ = place_l3!(ox + 93, y, TileType::WireDown);
    }
    // Sprint 121: L2 ViaUp exit relay at (ox+93, oy+45) reads L3.
    let s121_en_l2_exit = place_l2!(ox + 93, oy + 45, TileType::ViaUp);
    push_pipeline!(s121_en_l2_exit);
    let _s117_en_l1_drop = place_l1_route!(ox + 93, oy + 45, TileType::ViaUp);

    // -------------------------------------------------------------------------
    // Sprint 125: Physical RET — Link Register & Target Selection Mux
    // -------------------------------------------------------------------------
    // Eliminates the last meaningful software bridge: RET's post-propagation
    // target write to pc_next_mux_idx / pc_ingress_idx.
    //
    // Components:
    //   1. LR Register8 at L0(ox+8, oy+8) with feedback Mux at L0(ox+7, oy+8)
    //   2. Target Selection Mux at L2(ox+6, oy+1)
    //   3. is_ret/is_call extracted from ctrl_b L2 highway (Sprint 107) at oy+9
    //   4. L3 long-haul routes for control signals + PC+1 delivery
    //
    // Extraction moved to L2 (from ctrl_b east highway at oy+8) to avoid
    // L0 collisions with we_mask/reg_write at oy+9..10 and L1 collisions
    // with WE mask WireLeft corridor at L1(ox+24..34, oy+9).
    //
    // L1(ox+1, oy+1) changed from WireUp to ViaUp (see PC core above).
    let mut lr_dirty_indices = Vec::new();

    // --- Phase 2: is_ret/is_call extraction (6 L2 tiles at oy+9) ---
    // Tap ctrl_b from Sprint 107 L2 east highway at oy+8. WireDown reads
    // UP=ctrl_b from the highway WireRight tiles. BitSelect extracts the bit.
    // is_ret = ctrl_b[7], is_call = ctrl_b[6].
    let _s125_ret_wd = place_l2_route!(ox + 50, oy + 9, TileType::WireDown);
    let _s125_ret_bs = place_l2_route!(ox + 51, oy + 9, TileType::BitSelect);
    let _s125_ret_const = place_l2_val!(ox + 52, oy + 9, TileType::Const, 7);
    reserve_l2!(ox + 52, oy + 9);
    lr_dirty_indices.push(_s125_ret_wd);
    lr_dirty_indices.push(_s125_ret_bs);

    let _s125_call_wd = place_l2_route!(ox + 53, oy + 9, TileType::WireDown);
    let _s125_call_bs = place_l2_route!(ox + 54, oy + 9, TileType::BitSelect);
    let _s125_call_const = place_l2_val!(ox + 55, oy + 9, TileType::Const, 6);
    reserve_l2!(ox + 55, oy + 9);
    lr_dirty_indices.push(_s125_call_wd);
    lr_dirty_indices.push(_s125_call_bs);

    // --- Phase 4: is_ret L3 long-haul route (~57 tiles) ---
    // L2(ox+51,oy+9) BitSelect → L3 ViaDown → L3 west at oy+9 →
    // detour around ctrl_a at (ox+22,oy+9) → north at ox+8 →
    // west at oy+0 → L2 ViaUp at (ox+6,oy+0) → Target Mux UP
    let _s125_ret_l3_entry = place_l3!(ox + 51, oy + 9, TileType::ViaDown);
    lr_dirty_indices.push(_s125_ret_l3_entry);
    // L3 west at oy+9 from ox+50 to ox+24
    for x in (ox + 24)..=(ox + 50) {
        let idx = place_l3!(x, oy + 9, TileType::WireLeft);
        lr_dirty_indices.push(idx);
    }
    // Bridge tile at (ox+23, oy+9): WireLeft reads RIGHT=(ox+24) chain,
    // feeds WireUp at (ox+23, oy+8) via DOWN. Without this, guard Const(0) breaks chain.
    let _s125_ret_bridge = place_l3!(ox + 23, oy + 9, TileType::WireLeft);
    lr_dirty_indices.push(_s125_ret_bridge);
    // Detour around ctrl_a L3 entry at L3(ox+22, oy+9)
    let _s125_ret_detour_up = place_l3!(ox + 23, oy + 8, TileType::WireUp);
    let _s125_ret_detour_w1 = place_l3!(ox + 22, oy + 8, TileType::WireLeft);
    let _s125_ret_detour_w2 = place_l3!(ox + 21, oy + 8, TileType::WireLeft);
    let _s125_ret_detour_dn = place_l3!(ox + 21, oy + 9, TileType::WireDown);
    lr_dirty_indices.push(_s125_ret_detour_up);
    lr_dirty_indices.push(_s125_ret_detour_w1);
    lr_dirty_indices.push(_s125_ret_detour_w2);
    lr_dirty_indices.push(_s125_ret_detour_dn);
    // L3 west at oy+9 from ox+20 to ox+8
    for x in (ox + 8)..=(ox + 20) {
        let idx = place_l3!(x, oy + 9, TileType::WireLeft);
        lr_dirty_indices.push(idx);
    }
    // L3 north at ox+8 from oy+8 to oy+0
    for y in oy..=(oy + 8) {
        let idx = place_l3!(ox + 8, y, TileType::WireUp);
        lr_dirty_indices.push(idx);
    }
    // L3 west at oy+0 (ox+7 and ox+6)
    let _s125_ret_l3_w0a = place_l3!(ox + 7, oy, TileType::WireLeft);
    let _s125_ret_l3_w0b = place_l3!(ox + 6, oy, TileType::WireLeft);
    lr_dirty_indices.push(_s125_ret_l3_w0a);
    lr_dirty_indices.push(_s125_ret_l3_w0b);
    // Exit: L2 ViaUp at (ox+6, oy+0) → Target Mux UP
    let _s125_ret_l2_exit = place_l2!(ox + 6, oy, TileType::ViaUp);
    lr_dirty_indices.push(_s125_ret_l2_exit);

    // --- Phase 5: is_call L3+L2 route (~51 tiles) ---
    // L2(ox+54,oy+9) BitSelect → L3 ViaDown → L3 north to oy+7 →
    // L3 west at oy+7 → L2 ViaUp → L2 WireLeft to (ox+7,oy+7) →
    // L1/L0 ViaUp → LR Capture Mux UP
    // Uses L3 oy+7 (completely free) to avoid is_ret route at oy+9
    // and ctrl_a L3 east wall at oy+10.
    let _s125_call_l3_entry = place_l3!(ox + 54, oy + 9, TileType::ViaDown);
    lr_dirty_indices.push(_s125_call_l3_entry);
    // L3 north from oy+8 to oy+7 at ox+54
    let _s125_call_l3_n1 = place_l3!(ox + 54, oy + 8, TileType::WireUp);
    let _s125_call_l3_n2 = place_l3!(ox + 54, oy + 7, TileType::WireUp);
    lr_dirty_indices.push(_s125_call_l3_n1);
    lr_dirty_indices.push(_s125_call_l3_n2);
    // L3 west at oy+7 from ox+53 to ox+9
    for x in (ox + 9)..=(ox + 53) {
        let idx = place_l3!(x, oy + 7, TileType::WireLeft);
        lr_dirty_indices.push(idx);
    }
    // Exit to L2/L1/L0 at (ox+7, oy+7):
    // L2 ViaUp at (ox+9,oy+7) reads L3 → L2 WireLeft to (ox+7,oy+7)
    let _s125_call_l2_entry = place_l2_route!(ox + 9, oy + 7, TileType::ViaUp);
    let _s125_call_l2_wl1 = place_l2_route!(ox + 8, oy + 7, TileType::WireLeft);
    let _s125_call_l2_wl2 = place_l2_route!(ox + 7, oy + 7, TileType::WireLeft);
    lr_dirty_indices.push(_s125_call_l2_entry);
    lr_dirty_indices.push(_s125_call_l2_wl1);
    lr_dirty_indices.push(_s125_call_l2_wl2);
    // L1/L0 ViaUp chain at (ox+7, oy+7)
    let _s125_call_l1_exit = place_l1_route!(ox + 7, oy + 7, TileType::ViaUp);
    let _s125_call_l0_exit = place!(ox + 7, oy + 7, TileType::ViaUp);
    lr_dirty_indices.push(_s125_call_l1_exit);
    lr_dirty_indices.push(_s125_call_l0_exit);

    // --- Phase 6: PC+1 delivery to LR via L3 (~14 tiles) ---
    // PC+1 at L2(ox+4,oy+1) → L3 south → east → ViaUp chain to L0(ox+6,oy+8)
    let _s125_pc1_l3_tap = place_l3!(ox + 4, oy + 1, TileType::ViaDown);
    lr_dirty_indices.push(_s125_pc1_l3_tap);
    // L3 south at ox+4 from oy+2 to oy+8
    for y in (oy + 2)..=(oy + 8) {
        let idx = place_l3!(ox + 4, y, TileType::WireDown);
        lr_dirty_indices.push(idx);
    }
    // L3 east at oy+8: only ox+5 (ox+6 blocked by rd2 L1 column)
    let _s125_pc1_l3_e1 = place_l3!(ox + 5, oy + 8, TileType::WireRight);
    lr_dirty_indices.push(_s125_pc1_l3_e1);
    // Exit to L0: L2/L1/L0 ViaUp at (ox+5, oy+8), then WireRight relay at (ox+6, oy+8)
    let _s125_pc1_l2_exit = place_l2!(ox + 5, oy + 8, TileType::ViaUp);
    let _s125_pc1_l1_exit = place_l1_route!(ox + 5, oy + 8, TileType::ViaUp);
    let _s125_pc1_l0_via = place!(ox + 5, oy + 8, TileType::ViaUp);
    let _s125_pc1_l0_relay = place!(ox + 6, oy + 8, TileType::WireRight);
    lr_dirty_indices.push(_s125_pc1_l2_exit);
    lr_dirty_indices.push(_s125_pc1_l1_exit);
    lr_dirty_indices.push(_s125_pc1_l0_via);
    lr_dirty_indices.push(_s125_pc1_l0_relay);

    // --- Phase 7: LR subsystem (2 L0 tiles) ---
    // Mux at L0(ox+7, oy+8): UP=is_call, LEFT=PC+1(ViaUp), RIGHT=LR(Register8)
    // Register8 at L0(ox+8, oy+8): captures Mux output on clock edge.
    let _s125_lr_mux = place!(ox + 7, oy + 8, TileType::Mux);
    // Sprint 370 (Gate B.3): LR is Register64 in wide mode to hold wide return addresses.
    let lr_idx = place_val!(ox + 8, oy + 8, pc_reg_type, 0);
    lr_dirty_indices.push(_s125_lr_mux);
    lr_dirty_indices.push(lr_idx);

    // --- Phase 8: ir_low delivery to Target Mux (L2+L3, Sprint 150 cutover) ---
    // Goal: feed Target Mux RIGHT neighbor at L2(ox+7, oy+1).
    // Sprint 150 feeds selected low byte into L2(ox+2, oy+4) from selector Final.
    // Fanout on L2 row oy+4:
    //   west to extraction ingress at L2(ox+1, oy+4),
    //   east to Target Mux route at L2(ox+7, oy+4).
    let _s125_irlow_l2_west = place_l2_route!(ox + 1, oy + 4, TileType::WireLeft);
    lr_dirty_indices.push(_s125_irlow_l2_west);
    for x in (ox + 3)..=(ox + 7) {
        let idx = place_l2_route!(x, oy + 4, TileType::WireRight);
        lr_dirty_indices.push(idx);
    }
    // L3 hop north at ox+7 to bypass Phase 9 L2 tiles at oy+3
    let _s125_irlow_l3_entry = place_l3!(ox + 7, oy + 4, TileType::ViaDown);
    lr_dirty_indices.push(_s125_irlow_l3_entry);
    for y in (oy + 1)..=(oy + 3) {
        let idx = place_l3!(ox + 7, y, TileType::WireUp);
        lr_dirty_indices.push(idx);
    }
    // L2 ViaUp at (ox+7, oy+1): reads L3, feeds Target Mux RIGHT
    let _s125_irlow_l2_exit = place_l2_route!(ox + 7, oy + 1, TileType::ViaUp);
    lr_dirty_indices.push(_s125_irlow_l2_exit);

    // --- Phase 9: LR delivery to Target Mux (L1+L2, ~13 tiles) ---
    // LR at L0(ox+8,oy+8) → L1 north → L2 west → L2 north → Mux LEFT
    let _s125_lr_l1_tap = place_l1_route!(ox + 8, oy + 8, TileType::ViaDown);
    lr_dirty_indices.push(_s125_lr_l1_tap);
    // L1 north at ox+8 from oy+7 to oy+2 (signal flows north via WireUp)
    for y in (oy + 2)..=(oy + 7) {
        let idx = place_l1_route!(ox + 8, y, TileType::WireUp);
        lr_dirty_indices.push(idx);
    }
    // L2 entry at (ox+8, oy+3)
    let _s125_lr_l2_tap = place_l2_route!(ox + 8, oy + 3, TileType::ViaDown);
    lr_dirty_indices.push(_s125_lr_l2_tap);
    // L2 west at oy+3 from ox+7 to ox+5
    for x in (ox + 5)..=(ox + 7) {
        let idx = place_l2_route!(x, oy + 3, TileType::WireLeft);
        lr_dirty_indices.push(idx);
    }
    // L2 north at ox+5 from oy+2 to oy+1
    let _s125_lr_l2_n1 = place_l2_route!(ox + 5, oy + 2, TileType::WireUp);
    let _s125_lr_l2_n2 = place_l2_route!(ox + 5, oy + 1, TileType::WireUp);
    lr_dirty_indices.push(_s125_lr_l2_n1);
    lr_dirty_indices.push(_s125_lr_l2_n2);

    // --- Phase 10: Target Selection Mux (1 L2 tile) ---
    // UP=is_ret(L2 ViaUp), LEFT=LR(WireUp), RIGHT=ir_low(WireUp)
    let _s125_target_mux = place_l2_route!(ox + 6, oy + 1, TileType::Mux);
    lr_dirty_indices.push(_s125_target_mux);

    // --- Phase 11: Target Mux output route via L3+L2 (~17 tiles) ---
    // L2 Mux → L3 ViaDown → L3 south at ox+6 to oy+5 → L2 hop west
    // (ox+6→ox+3 at oy+5) → L3 ViaDown → L3 west to ox+1 at oy+5 →
    // L3 north at ox+1 from oy+4 to oy+1 → L2/L1 ViaUp.
    //
    // CRITICAL: Must NOT use L3 at oy=0 (ox+1..3) — those are Sprint 118's
    // branch_taken_core east delivery (WireRight). Overwriting them broke
    // PC enable (L2(ox+3,oy) ViaUp reads L3 → contaminates Or gate).
    // Phase 6 (PC+1) blocks L3 ox+4 from oy+1..8, crossed via L2 hop at oy+5.
    // L2 at oy+2..4 blocked by Phase 8/9 at ox+3..5, so hop at oy+5.
    let _s125_out_l3_entry = place_l3!(ox + 6, oy + 1, TileType::ViaDown);
    lr_dirty_indices.push(_s125_out_l3_entry);
    // L3 south at ox+6 from oy+2 to oy+5
    for y in (oy + 2)..=(oy + 5) {
        let idx = place_l3!(ox + 6, y, TileType::WireDown);
        lr_dirty_indices.push(idx);
    }
    // L2 hop west at oy+5: ViaUp(ox+6) → WireLeft(ox+5,4,3)
    let _s125_out_l2_hop_entry = place_l2_route!(ox + 6, oy + 5, TileType::ViaUp);
    lr_dirty_indices.push(_s125_out_l2_hop_entry);
    for x in (ox + 3)..=(ox + 5) {
        let idx = place_l2_route!(x, oy + 5, TileType::WireLeft);
        lr_dirty_indices.push(idx);
    }
    // Back to L3 at (ox+3, oy+5)
    let _s125_out_l3_reentry = place_l3!(ox + 3, oy + 5, TileType::ViaDown);
    lr_dirty_indices.push(_s125_out_l3_reentry);
    // L3 west at oy+5 from ox+2 to ox+1
    let _s125_out_l3_w2 = place_l3!(ox + 2, oy + 5, TileType::WireLeft);
    lr_dirty_indices.push(_s125_out_l3_w2);
    let _s125_out_l3_w1 = place_l3!(ox + 1, oy + 5, TileType::WireLeft);
    lr_dirty_indices.push(_s125_out_l3_w1);
    // L3 north at ox+1 from oy+4 to oy+1
    for y in (oy + 1)..=(oy + 4) {
        let idx = place_l3!(ox + 1, y, TileType::WireUp);
        lr_dirty_indices.push(idx);
    }
    // L2 exit: ViaUp at (ox+1, oy+1) → feeds L1 ViaUp (already changed above)
    let _s125_out_l2_exit = place_l2!(ox + 1, oy + 1, TileType::ViaUp);
    lr_dirty_indices.push(_s125_out_l2_exit);
    // L1(ox+1, oy+1) is already ViaUp (changed above in PC core).
    // pc_next_mux LEFT = L1(ox+1, oy+1) = ViaUp = Target Mux output. ✓

    // Add LR dirty indices to commit path.
    // NOTE: pc_path_dirty_indices was already drained by .append() at line ~811,
    // so we push directly to commit_dirty_indices.
    commit_dirty_indices.extend_from_slice(&lr_dirty_indices);

    // Sprint 130: Add CarryDetect and selection Mux to flag commit dirty list.
    flag_commit_dirty_indices.push(s130_carry_add_idx);
    flag_commit_dirty_indices.push(s130_carry_sub_idx);
    flag_commit_dirty_indices.push(s130_carry_mux_idx);
    flag_commit_dirty_indices.push(s130_not_sel0_l0);
    flag_commit_dirty_indices.push(s130_not_sel0_l1);
    flag_commit_dirty_indices.push(s130_not_sel0_l1_entry);
    flag_commit_dirty_indices.push(s130_mux_l1_via);
    flag_commit_dirty_indices.push(s130_mux_l2_via);

    // -------------------------------------------------------------------------
    // Sprint 151: Physical westback routes — ir_high and ir_low
    // -------------------------------------------------------------------------
    // Carry selected ROM bank ir_low/ir_high from selector Final Mux outputs
    // back to extraction pipeline injection points, eliminating the software
    // inject_selector_outputs() and reapply_rom_bank_overwrite() assists.

    // --- ir_high westback: L0(64,9) → L2(13,4) via L3 y=0 transit ---
    // Via stack at (64,9): bring selector output from L0 up to L3.
    let _s151_irh_l1 = place_l1_route!(ox + 64, oy + 9, TileType::ViaDown);
    let _s151_irh_l2 = place_l2_route!(ox + 64, oy + 9, TileType::ViaDown);
    let _s151_irh_l3 = place_l3_route!(ox + 64, oy + 9, TileType::ViaDown);
    push_pipeline!(_s151_irh_l1);
    push_pipeline!(_s151_irh_l2);
    push_pipeline!(_s151_irh_l3);
    // L3 east y=9 from x=65 to x=68.
    for x in (ox + 65)..=(ox + 68) {
        let idx = place_l3_route!(x, oy + 9, TileType::WireRight);
        push_pipeline!(idx);
    }
    // L3 north x=68 from y=0 to y=8.
    for y in oy..=(oy + 8) {
        let idx = place_l3_route!(ox + 68, y, TileType::WireUp);
        push_pipeline!(idx);
    }
    // L3 west y=0 from x=12 to x=67.
    for x in (ox + 12)..=(ox + 67) {
        let idx = place_l3_route!(x, oy, TileType::WireLeft);
        push_pipeline!(idx);
    }
    // L3 south x=12 from y=1 to y=4.
    for y in (oy + 1)..=(oy + 4) {
        let idx = place_l3_route!(ox + 12, y, TileType::WireDown);
        push_pipeline!(idx);
    }
    // L3(13,4) WireRight reads LEFT = L3(12,4).
    let _s151_irh_l3_exit = place_l3_route!(ox + 13, oy + 4, TileType::WireRight);
    push_pipeline!(_s151_irh_l3_exit);
    // L2(13,4): Const(0) guard → ViaUp (reads L3(13,4) = westback ir_high).
    let _s151_irh_l2_exit = place_l2_route!(ox + 13, oy + 4, TileType::ViaUp);
    push_pipeline!(_s151_irh_l2_exit);

    // --- ir_low westback: L0(58,9) → L2(2,4) via L3 y=11 transit ---
    // Route: via stack(58,9) → L3 east y=9 x=59..63 → south to y=10 →
    // L3 east y=10 x=64..69 (past ctrl_a) → south to y=11 →
    // L3 west y=11 x=7..68 (completely free row) →
    // L3 south x=7 y=3..10 (free: bank delivery ≥x=14, call ≥x=9) →
    // L2 bridge (7,3)→(9,3) → via stack down → L0 tunnel → pocket.

    // Via stack at (58,9): bring selector output from L0 up to L3.
    let _s151_irl_l1 = place_l1_route!(ox + 58, oy + 9, TileType::ViaDown);
    let _s151_irl_l2 = place_l2_route!(ox + 58, oy + 9, TileType::ViaDown);
    let _s151_irl_l3 = place_l3_route!(ox + 58, oy + 9, TileType::ViaDown);
    push_pipeline!(_s151_irl_l1);
    push_pipeline!(_s151_irl_l2);
    push_pipeline!(_s151_irl_l3);
    // L3 east y=9 from x=59 to x=63 (ir_high starts at x=64).
    for x in (ox + 59)..=(ox + 63) {
        let idx = place_l3_route!(x, oy + 9, TileType::WireRight);
        push_pipeline!(idx);
    }
    // South step to y=10 at x=63 (ctrl_a L3 at y=10 only x=22..60).
    let _s151_irl_step10 = place_l3_route!(ox + 63, oy + 10, TileType::WireDown);
    push_pipeline!(_s151_irl_step10);
    // L3 east y=10 from x=64 to x=69 (outside ctrl_a range).
    for x in (ox + 64)..=(ox + 69) {
        let idx = place_l3_route!(x, oy + 10, TileType::WireRight);
        push_pipeline!(idx);
    }
    // South step to y=11 at x=69.
    let _s151_irl_step11 = place_l3_route!(ox + 69, oy + 11, TileType::WireDown);
    push_pipeline!(_s151_irl_step11);
    // L3 west y=11: obstacles at x=60 (Sprint 119 ctrl_a L3 south column),
    // x=61..63 (Sprint 107/117 on L2). Bridge via L1 from x=64 to x=59.
    //
    // East segment: L3 west from x=64 to x=68.
    for x in (ox + 64)..=(ox + 68) {
        let idx = place_l3_route!(x, oy + 11, TileType::WireLeft);
        push_pipeline!(idx);
    }
    // L1 bridge: L3(64,11) → L2(64,11) ViaUp → L1(64,11) ViaUp →
    // L1 WireLeft x=63..59 → L2(59,11) ViaDown → L3(59,11) ViaDown.
    let _s151_irl_y11_l2_entry = place_l2_route!(ox + 64, oy + 11, TileType::ViaUp);
    let _s151_irl_y11_l1_entry = place_l1_route!(ox + 64, oy + 11, TileType::ViaUp);
    let _s151_irl_y11_l1_w1 = place_l1_route!(ox + 63, oy + 11, TileType::WireLeft);
    let _s151_irl_y11_l1_w2 = place_l1_route!(ox + 62, oy + 11, TileType::WireLeft);
    let _s151_irl_y11_l1_w3 = place_l1_route!(ox + 61, oy + 11, TileType::WireLeft);
    let _s151_irl_y11_l1_w4 = place_l1_route!(ox + 60, oy + 11, TileType::WireLeft);
    let _s151_irl_y11_l1_w5 = place_l1_route!(ox + 59, oy + 11, TileType::WireLeft);
    let _s151_irl_y11_l2_exit = place_l2_route!(ox + 59, oy + 11, TileType::ViaDown);
    let _s151_irl_y11_l3_reentry = place_l3_route!(ox + 59, oy + 11, TileType::ViaDown);
    push_pipeline!(_s151_irl_y11_l2_entry);
    push_pipeline!(_s151_irl_y11_l1_entry);
    push_pipeline!(_s151_irl_y11_l1_w1);
    push_pipeline!(_s151_irl_y11_l1_w2);
    push_pipeline!(_s151_irl_y11_l1_w3);
    push_pipeline!(_s151_irl_y11_l1_w4);
    push_pipeline!(_s151_irl_y11_l1_w5);
    push_pipeline!(_s151_irl_y11_l2_exit);
    push_pipeline!(_s151_irl_y11_l3_reentry);
    // West segment: L3 west from x=7 to x=58.
    for x in (ox + 7)..=(ox + 58) {
        let idx = place_l3_route!(x, oy + 11, TileType::WireLeft);
        push_pipeline!(idx);
    }
    // L3 south x=7 from y=3 to y=10 (x=7 avoids Sprint 125 ret at x=8
    // and call at x=9..53; bank delivery starts at x≥14).
    for y in (oy + 3)..=(oy + 10) {
        let idx = place_l3_route!(ox + 7, y, TileType::WireUp);
        push_pipeline!(idx);
    }
    // L2 bridge from L3(7,5) to L2(9,3):
    // avoid Sprint 125 occupancy at L2(7,3) by lifting at y=5, jogging east,
    // then climbing north to rejoin the existing L0 tunnel ingress at (9,3).
    let _s151_irl_l2_bridge = place_l2_route!(ox + 7, oy + 5, TileType::ViaUp);
    let _s151_irl_l2_hop1 = place_l2_route!(ox + 8, oy + 5, TileType::WireRight);
    let _s151_irl_l2_hop2 = place_l2_route!(ox + 9, oy + 5, TileType::WireRight);
    let _s151_irl_l2_up1 = place_l2_route!(ox + 9, oy + 4, TileType::WireUp);
    let _s151_irl_l2_up2 = place_l2_route!(ox + 9, oy + 3, TileType::WireUp);
    push_pipeline!(_s151_irl_l2_bridge);
    push_pipeline!(_s151_irl_l2_hop1);
    push_pipeline!(_s151_irl_l2_hop2);
    push_pipeline!(_s151_irl_l2_up1);
    push_pipeline!(_s151_irl_l2_up2);
    // Via stack DOWN at (9,3): L2 → L1 ViaUp → L0 ViaUp.
    let _s151_irl_l1_dn = place_l1_route!(ox + 9, oy + 3, TileType::ViaUp);
    let _s151_irl_l0_dn = place!(ox + 9, oy + 3, TileType::ViaUp);
    push_pipeline!(_s151_irl_l1_dn);
    push_pipeline!(_s151_irl_l0_dn);
    // L0 west y=3 from x=3 to x=8 (tunnel under Sprint 125 wall).
    for x in (ox + 3)..=(ox + 8) {
        let idx = place!(x, oy + 3, TileType::WireLeft);
        push_pipeline!(idx);
    }
    // Via stack UP at (3,3): L0 → L1 → L2 → L3 (inside sealed pocket).
    let _s151_irl_l1_up = place_l1_route!(ox + 3, oy + 3, TileType::ViaDown);
    let _s151_irl_l2_up = place_l2_route!(ox + 3, oy + 3, TileType::ViaDown);
    let _s151_irl_l3_up = place_l3_route!(ox + 3, oy + 3, TileType::ViaDown);
    push_pipeline!(_s151_irl_l1_up);
    push_pipeline!(_s151_irl_l2_up);
    push_pipeline!(_s151_irl_l3_up);
    // Inside pocket: L3(2,3) WireLeft, L3(2,4) WireDown → L2(2,4) ViaUp.
    let _s151_irl_pocket_w = place_l3_route!(ox + 2, oy + 3, TileType::WireLeft);
    let _s151_irl_pocket_d = place_l3_route!(ox + 2, oy + 4, TileType::WireDown);
    push_pipeline!(_s151_irl_pocket_w);
    push_pipeline!(_s151_irl_pocket_d);
    // L2(2,4) already changed to ViaUp in B0.low re-route above.
    // Reads L3(2,4) = westback ir_low, feeding:
    //   - L2(1,4) WireLeft → L1 ViaUp → L0 ViaUp (extraction)
    //   - L2(3,4) WireRight → Sprint 125 Phase 8 → Target Mux

    pipeline_dirty_indices.sort_unstable();
    pipeline_dirty_indices.dedup();
    pipeline_backbone_dirty_indices.sort_unstable();
    pipeline_backbone_dirty_indices.dedup();
    for routes in &mut pipeline_reg_data_dirty_indices {
        routes.sort_unstable();
        routes.dedup();
    }

    // Rebuild cross-layer via forwarding after all ViaUp/ViaDown placement.
    sim.rebuild_via_connections();

    V2CpuIndices {
        pc_idx,
        pc_next_mux_idx,
        pc_ingress_idx,
        // Sprint 152: rom_low/high_mux_idx removed.
        bank_low_mux_indices: [
            bank0_low_mux_idx,
            bank1_low_mux_idx,
            bank2_low_mux_idx,
            bank3_low_mux_idx,
        ],
        bank_high_mux_indices: [
            bank0_high_mux_idx,
            bank1_high_mux_idx,
            bank2_high_mux_idx,
            bank3_high_mux_idx,
        ],
        // bank_byte2/3_mux_indices removed — Sprint 185 physical ir_ext
        // Sprint 211: rom_selected now points to Super Mux outputs (unified 8-bank selector).
        rom_selected_low_idx: super_mux_indices[0],
        rom_selected_high_idx: super_mux_indices[1],
        rom_selected_byte2_idx: super_mux_indices[2],
        rom_selected_byte3_idx: super_mux_indices[3],
        bank47_mux_indices,
        final_mux_indices,
        super_mux_indices,
        super_mux_inject_indices,
        super_mux_inject_right_indices,
        // Sprint 151: injection indices removed (physical westback routes).
        reg_indices,
        reg_tap_l1_indices,
        flag_z_idx,
        flag_c_idx,
        ram_indices,
        extract_opcode_shr_idx,
        extract_opcode_bit4_idx,
        extract_rd_bit_indices: [extract_rd0_idx, extract_rd1_idx, extract_rd2_idx],
        extract_rs_field_idx,
        ctrl_a_mux_idx,
        ctrl_b_mux_idx,
        branch_ctrl_b_l1_tap_idx: _branch_ctrl_b_l1_tap_idx,
        branch_flag_z_l1_tap_idx: _branch_flag_z_l1_tap_idx,
        branch_flag_c_l1_tap_idx: _branch_flag_c_l1_tap_idx,
        branch_taken_core_idx,
        branch_dirty_indices,
        branch_flag_dirty_indices,
        op_a_root_idx: top_mux_a_idx,
        op_b_root_idx: top_mux_b_idx,
        pipeline_dirty_indices,
        pipeline_backbone_dirty_indices,
        pipeline_reg_data_dirty_indices,
        rd_onehot_decode_idx,
        rd_decode_l0_chain,
        ram_write_decode_idx,
        ram_write_gate_idx,
        we_mask_const_idx,
        commit_dirty_indices,
        reg_wb_dirty_indices,
        flag_we_mask_const_idx,
        flag_commit_dirty_indices,
        lr_idx,
        lr_dirty_indices,
        wb_alu_root_idx,
        wb_data_mux_idx,
        alu_add_l_enable_idx: alu_l_enable_indices[0],
        alu_add_l_mux_idx: alu_l_mux_indices[0],
        high_reg_wb_data_const_indices,
        high_reg_we_const_indices,
        high_reg_merge_mux_indices,
        op_a_high_root_idx: high_tree_a.root_idx,
        op_b_high_root_idx: high_tree_b.root_idx,
        top_mux_a_idx,
        top_mux_b_idx,
        top_mux_a_sel_const_idx,
        top_mux_b_sel_const_idx,
        high_tree_a_sel_const_indices: high_tree_a.sel_const_indices,
        high_tree_b_sel_const_indices: high_tree_b.sel_const_indices,
        high_tree_a_data_const_indices,
        high_tree_b_data_const_indices,
        high_tree_a_dirty_indices,
        high_tree_b_dirty_indices,
        top_mux_a_dirty_indices,
        top_mux_b_dirty_indices,
        alu_a_trunk_terminal_idx,
        alu_b_trunk_terminal_idx,
        alu_a_downstream_dirty,
        alu_b_downstream_dirty,
        alu_r_mux_output_indices,
        tile_count,
        grid_width,
        // Sprint 146: scopes computed by builder after wiring
        pipeline_scope: Vec::new(),
        branch_scope: Vec::new(),
        commit_scope: Vec::new(),
        // Sprint 147: bitset masks computed by builder
        pipeline_scope_mask: Vec::new(),
        // Sprint 262: eval order computed by builder
        pipeline_eval_order: Vec::new(),
        branch_eval_order: Vec::new(),
        commit_eval_order: Vec::new(),
        branch_scope_mask: Vec::new(),
        commit_scope_mask: Vec::new(),
        // Sprint 166: clock scope computed by builder
        clock_scope_mask: Vec::new(),
        // Sprint 276: clock eval order computed by builder
        clock_eval_order: Vec::new(),
        // Sprint 168: computed by builder after scope masks are built
        in_scope_clock_cache: Vec::new(),
        // Sprint 147: placed tile bitset for restricted BFS
        placed_mask,
    }
}
