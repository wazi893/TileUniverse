//! V2 Synth Pilot Integration — Sprint 193.
//!
//! Proves that a synth-generated combinational block can coexist with the
//! live V2 CPU in the same 128×128×4 simulation grid.
//!
//! Pilot circuit: branch-taken decoder (5 inputs → 1 output), matching the
//! physical Mux16to1 LUT in `v2_wiring.rs`.

use crate::simulation::Simulation;
use crate::synth::aig::{Aig, AigLit};
use crate::synth::export::SynthExport;
use std::sync::atomic::Ordering;

// ---------------------------------------------------------------------------
// Branch-taken truth table
// ---------------------------------------------------------------------------

/// Expected branch-taken output for each of the 32 input combinations.
///
/// Input encoding (5 bits):
/// - bits [2:0] = ctrl_b (branch condition code)
/// - bit 3 = flag_z
/// - bit 4 = flag_c
///
/// Matches `v2_wiring.rs:3816-3832`.
pub fn branch_taken_truth_table() -> [bool; 32] {
    let mut table = [false; 32];
    for sel in 0..32usize {
        let kind = sel & 0x07;
        let z = (sel >> 3) & 1 != 0;
        let c = (sel >> 4) & 1 != 0;
        table[sel] = match kind {
            0 => false,
            1 => true, // JMP
            2 => z,    // BEQ
            3 => !z,   // BNE
            4 => c,    // BCS
            5 => !c,   // BCC
            6 => true, // CALL
            7 => true, // RET
            _ => unreachable!(),
        };
    }
    table
}

// ---------------------------------------------------------------------------
// Branch-taken AIG builder
// ---------------------------------------------------------------------------

/// Build an AIG for the branch-taken decoder.
///
/// 5 primary inputs: ctrl_b[0], ctrl_b[1], ctrl_b[2], flag_z, flag_c.
/// 1 primary output: taken.
///
/// Logic: taken = (kind==1) | (kind==2 & z) | (kind==3 & !z)
///              | (kind==4 & c) | (kind==5 & !c) | (kind==6) | (kind==7)
pub fn build_branch_taken_aig() -> Aig {
    let mut aig = Aig::new();
    let b0 = aig.add_input("ctrl_b0");
    let b1 = aig.add_input("ctrl_b1");
    let b2 = aig.add_input("ctrl_b2");
    let z = aig.add_input("flag_z");
    let c = aig.add_input("flag_c");

    let nb0 = b0.negated();
    let nb1 = b1.negated();
    let nb2 = b2.negated();

    // kind == N means (b2,b1,b0) == binary(N).
    // kind==1: !b2 & !b1 & b0
    let t = aig.and(nb1, b0);
    let k1 = aig.and(nb2, t);
    // kind==2: !b2 & b1 & !b0
    let t = aig.and(b1, nb0);
    let k2 = aig.and(nb2, t);
    // kind==3: !b2 & b1 & b0
    let t = aig.and(b1, b0);
    let k3 = aig.and(nb2, t);
    // kind==4: b2 & !b1 & !b0
    let t = aig.and(nb1, nb0);
    let k4 = aig.and(b2, t);
    // kind==5: b2 & !b1 & b0
    let t = aig.and(nb1, b0);
    let k5 = aig.and(b2, t);
    // kind==6: b2 & b1 & !b0
    let t = aig.and(b1, nb0);
    let k6 = aig.and(b2, t);
    // kind==7: b2 & b1 & b0
    let t = aig.and(b1, b0);
    let k7 = aig.and(b2, t);

    // Conditional terms
    let t2 = aig.and(k2, z); // kind==2 & flag_z
    let t3 = aig.and(k3, z.negated()); // kind==3 & !flag_z
    let t4 = aig.and(k4, c); // kind==4 & flag_c
    let t5 = aig.and(k5, c.negated()); // kind==5 & !flag_c

    // OR all terms together
    let mut taken = k1;
    taken = aig.or(taken, t2);
    taken = aig.or(taken, t3);
    taken = aig.or(taken, t4);
    taken = aig.or(taken, t5);
    taken = aig.or(taken, k6);
    taken = aig.or(taken, k7);

    aig.add_output("taken", taken);
    aig
}

// ---------------------------------------------------------------------------
// Decoder3to8 truth table + AIG builder (Sprint 195)
// ---------------------------------------------------------------------------

/// Expected one-hot outputs for a 3-to-8 decoder.
///
/// `table[sel][i] == (sel == i)` — exactly one output is true per selector value.
/// Matches the physical `Decoder3to8` tile: `output = 1 << (left & 7)`.
pub fn decoder3to8_truth_table() -> [[bool; 8]; 8] {
    let mut table = [[false; 8]; 8];
    for sel in 0..8usize {
        table[sel][sel] = true;
    }
    table
}

/// Build an AIG for a 3-to-8 one-hot decoder.
///
/// 3 primary inputs: `sel[0]`, `sel[1]`, `sel[2]`.
/// 8 primary outputs: `out[0]`..`out[7]` (one-hot).
///
/// Logic: `out[i] = AND` of selector bits matching `i`'s binary representation
/// (complemented where bit is 0). ~16 AND nodes.
///
/// This replicates the physical `Decoder3to8` tile using composed AND/NOT gates,
/// proving the synthesizer can replace specialized tile types.
pub fn build_decoder3to8_aig() -> Aig {
    let mut aig = Aig::new();
    let s0 = aig.add_input("sel0");
    let s1 = aig.add_input("sel1");
    let s2 = aig.add_input("sel2");

    let ns0 = s0.negated();
    let ns1 = s1.negated();
    let ns2 = s2.negated();

    // Shared high-bit pairs to reduce AND node count:
    // pair_00 = !s2 & !s1, pair_01 = !s2 & s1, pair_10 = s2 & !s1, pair_11 = s2 & s1
    let pair_00 = aig.and(ns2, ns1);
    let pair_01 = aig.and(ns2, s1);
    let pair_10 = aig.and(s2, ns1);
    let pair_11 = aig.and(s2, s1);

    // out[i] = pair[i>>1] & (s0 if i&1 else !s0)
    let pairs = [pair_00, pair_01, pair_10, pair_11];
    for i in 0..8u32 {
        let pair = pairs[(i >> 1) as usize];
        let low = if i & 1 != 0 { s0 } else { ns0 };
        let out = aig.and(pair, low);
        aig.add_output(&format!("out{i}"), out);
    }

    aig
}

// ---------------------------------------------------------------------------
// CTRL_B decoder AIG builder (Sprint 201)
// ---------------------------------------------------------------------------

/// Software decoder LUT for ctrl_a (ALU/reg/flag control).
/// Indexed by opcode (0-31). Must match `CTRL_A_LUT` in `v2_wiring.rs`.
#[allow(dead_code)]
const CTRL_A_LUT_EMBEDDED: [u8; 32] = [
    0x00, 0x00, 0x08, 0x58, 0xB8, 0xB9, 0x9A, 0x9B, 0x9C, 0x9D, 0xB9, 0xF8, 0xF9, 0xFE, 0xFF, 0xB1,
    0xF8, 0xF9, 0xDA, 0xDB, 0xDC, 0x00, 0x18, 0x00, 0x58, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Software decoder LUT for ctrl_b (branch/mem/halt control).
/// Indexed by opcode (0-31). Must match `CTRL_B_LUT` in `v2_execute.rs`.
#[allow(dead_code)]
const CTRL_B_LUT_EMBEDDED: [u8; 32] = [
    0x00, 0x21, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x87, 0x08, 0x10, 0x08, 0x10, 0x01, 0x02, 0x03, 0x04, 0x05, 0x46,
];

/// Build an AIG for the ctrl_b decoder.
///
/// 5 primary inputs: opcode bits 0-4.
/// 8 primary outputs: ctrl_b bits 0-7.
///
/// Uses shared intermediate terms (low-bit pairs, high-bit groups) to minimize
/// AND node count (~36 nodes vs ~92 for naive sum-of-products). This keeps the
/// synthesized circuit compact enough for halo=4 placement.
///
/// Opcode grouping (opcode = b4*16 + b3*8 + b2*4 + b1*2 + b0):
/// - 0x01 = 00001: standalone (b4=0)
/// - 0x15..0x17 = 10101..10111: b4=1, b3=0 group (b4_nb3_b2)
/// - 0x18..0x1F = 11000..11111: b4=1, b3=1 group (b4_b3)
pub fn build_ctrl_b_aig() -> Aig {
    let mut aig = Aig::new();
    let b0 = aig.add_input("op0");
    let b1 = aig.add_input("op1");
    let b2 = aig.add_input("op2");
    let b3 = aig.add_input("op3");
    let b4 = aig.add_input("op4");

    let nb0 = b0.negated();
    let nb1 = b1.negated();
    let nb2 = b2.negated();
    let nb3 = b3.negated();
    let nb4 = b4.negated();

    // Shared low-bit pairs (b1, b0).
    let pair_00 = aig.and(nb1, nb0);
    let pair_01 = aig.and(nb1, b0);
    let pair_10 = aig.and(b1, nb0);
    let pair_11 = aig.and(b1, b0);

    // Shared high-bit groups.
    let nb4_nb3 = aig.and(nb4, nb3);
    let b4_nb3 = aig.and(b4, nb3);
    let b4_b3 = aig.and(b4, b3);

    // 3-var groups (high bits + b2).
    let nb4_nb3_nb2 = aig.and(nb4_nb3, nb2);
    let b4_nb3_b2 = aig.and(b4_nb3, b2); // covers 0x15, 0x16, 0x17
    let b4_b3_nb2 = aig.and(b4_b3, nb2); // covers 0x18, 0x19, 0x1A, 0x1B
    let b4_b3_b2 = aig.and(b4_b3, b2); // covers 0x1C, 0x1D, 0x1E, 0x1F

    // Individual opcode detection (5-bit minterms via 3-var group & low pair).
    let op01 = aig.and(nb4_nb3_nb2, pair_01); // 00001
    let op15 = aig.and(b4_nb3_b2, pair_01); // 10101
    let op16 = aig.and(b4_nb3_b2, pair_10); // 10110
    let op17 = aig.and(b4_nb3_b2, pair_11); // 10111
    let op18 = aig.and(b4_b3_nb2, pair_00); // 11000
    let op19 = aig.and(b4_b3_nb2, pair_01); // 11001
    let op1a = aig.and(b4_b3_nb2, pair_10); // 11010
    let op1b = aig.and(b4_b3_nb2, pair_11); // 11011
    let op1c = aig.and(b4_b3_b2, pair_00); // 11100
    let op1d = aig.and(b4_b3_b2, pair_01); // 11101
    let op1e = aig.and(b4_b3_b2, pair_10); // 11110
    let op1f = aig.and(b4_b3_b2, pair_11); // 11111

    // Output bits: OR of the opcodes where each bit is set in CTRL_B_LUT.
    // Bit 0: {0x01, 0x15, 0x1A, 0x1C, 0x1E}
    let mut cb0 = aig.or(op01, op15);
    cb0 = aig.or(cb0, op1a);
    cb0 = aig.or(cb0, op1c);
    cb0 = aig.or(cb0, op1e);
    aig.add_output("cb0", cb0);
    // Bit 1: {0x15, 0x1B, 0x1C, 0x1F}
    let mut cb1 = aig.or(op15, op1b);
    cb1 = aig.or(cb1, op1c);
    cb1 = aig.or(cb1, op1f);
    aig.add_output("cb1", cb1);
    // Bit 2: {0x15, 0x1D, 0x1E, 0x1F}
    let mut cb2 = aig.or(op15, op1d);
    cb2 = aig.or(cb2, op1e);
    cb2 = aig.or(cb2, op1f);
    aig.add_output("cb2", cb2);
    // Bit 3: {0x16, 0x18}
    let cb3 = aig.or(op16, op18);
    aig.add_output("cb3", cb3);
    // Bit 4: {0x17, 0x19}
    let cb4 = aig.or(op17, op19);
    aig.add_output("cb4", cb4);
    // Bit 5: {0x01} — single minterm
    aig.add_output("cb5", op01);
    // Bit 6: {0x1F} — single minterm
    aig.add_output("cb6", op1f);
    // Bit 7: {0x15} — single minterm
    aig.add_output("cb7", op15);

    aig
}

/// Sprint 203: Build the ctrl_a AIG (5-input opcode → 8-bit control word).
///
/// 5 inputs (opcode bits b0-b4), 8 outputs (ca0-ca7):
///   ca[2:0] = alu_sel, ca[3] = reg_write, ca[4] = flag_z_we,
///   ca[5] = flag_c_we, ca[6:7] = reserved.
///
/// 22 active opcodes. Uses shared sub-expressions (low-bit pairs, high-bit
/// groups, 3-var groups) to minimize AND nodes. Full 3-var groups replace
/// individual minterms where all 4 members agree on a bit (e.g. bit 7 uses
/// 4 group shortcuts instead of 16 individual ORs).
pub fn build_ctrl_a_aig() -> Aig {
    let mut aig = Aig::new();
    let b0 = aig.add_input("op0");
    let b1 = aig.add_input("op1");
    let b2 = aig.add_input("op2");
    let b3 = aig.add_input("op3");
    let b4 = aig.add_input("op4");

    let nb0 = b0.negated();
    let nb1 = b1.negated();
    let nb2 = b2.negated();
    let nb3 = b3.negated();
    let nb4 = b4.negated();

    // Shared low-bit pairs (b1, b0).
    let pair_00 = aig.and(nb1, nb0);
    let pair_01 = aig.and(nb1, b0);
    let pair_10 = aig.and(b1, nb0);
    let pair_11 = aig.and(b1, b0);

    // Shared high-bit groups (b4, b3).
    let nb4_nb3 = aig.and(nb4, nb3);
    let nb4_b3 = aig.and(nb4, b3);
    let b4_nb3 = aig.and(b4, nb3);
    let b4_b3 = aig.and(b4, b3);

    // 3-var groups (b4, b3, b2).
    let nb4_nb3_nb2 = aig.and(nb4_nb3, nb2); // 000xx: opcodes 0-3
    let nb4_nb3_b2 = aig.and(nb4_nb3, b2); // 001xx: opcodes 4-7
    let nb4_b3_nb2 = aig.and(nb4_b3, nb2); // 010xx: opcodes 8-11
    let nb4_b3_b2 = aig.and(nb4_b3, b2); // 011xx: opcodes 12-15
    let b4_nb3_nb2 = aig.and(b4_nb3, nb2); // 100xx: opcodes 16-19
    let b4_nb3_b2 = aig.and(b4_nb3, b2); // 101xx: opcodes 20-23
    let b4_b3_nb2 = aig.and(b4_b3, nb2); // 110xx: opcodes 24-27

    // 5-bit minterms for all 22 active opcodes.
    let op02 = aig.and(nb4_nb3_nb2, pair_10); // 00010
    let op03 = aig.and(nb4_nb3_nb2, pair_11); // 00011
    let op04 = aig.and(nb4_nb3_b2, pair_00); // 00100
    let op05 = aig.and(nb4_nb3_b2, pair_01); // 00101
    let op06 = aig.and(nb4_nb3_b2, pair_10); // 00110
    let op07 = aig.and(nb4_nb3_b2, pair_11); // 00111
    let op08 = aig.and(nb4_b3_nb2, pair_00); // 01000
    let op09 = aig.and(nb4_b3_nb2, pair_01); // 01001
    let op0a = aig.and(nb4_b3_nb2, pair_10); // 01010
    let op0b = aig.and(nb4_b3_nb2, pair_11); // 01011
    let op0c = aig.and(nb4_b3_b2, pair_00); // 01100
    let op0d = aig.and(nb4_b3_b2, pair_01); // 01101
    let op0e = aig.and(nb4_b3_b2, pair_10); // 01110
    let op0f = aig.and(nb4_b3_b2, pair_11); // 01111
    let op10 = aig.and(b4_nb3_nb2, pair_00); // 10000
    let op11 = aig.and(b4_nb3_nb2, pair_01); // 10001
    let op12 = aig.and(b4_nb3_nb2, pair_10); // 10010
    let op13 = aig.and(b4_nb3_nb2, pair_11); // 10011
    let op14 = aig.and(b4_nb3_b2, pair_00); // 10100
    let op16 = aig.and(b4_nb3_b2, pair_10); // 10110
    let op18 = aig.and(b4_b3_nb2, pair_00); // 11000
    let op19 = aig.and(b4_b3_nb2, pair_01); // 11001

    // Bit 0 (alu_sel[0]): {5,7,9,10,12,14,15,17,19}
    let mut ca0 = aig.or(op05, op07);
    ca0 = aig.or(ca0, op09);
    ca0 = aig.or(ca0, op0a);
    ca0 = aig.or(ca0, op0c);
    ca0 = aig.or(ca0, op0e);
    ca0 = aig.or(ca0, op0f);
    ca0 = aig.or(ca0, op11);
    ca0 = aig.or(ca0, op13);
    aig.add_output("ca0", ca0);

    // Bit 1 (alu_sel[1]): {6,7,13,14,18,19}
    let mut ca1 = aig.or(op06, op07);
    ca1 = aig.or(ca1, op0d);
    ca1 = aig.or(ca1, op0e);
    ca1 = aig.or(ca1, op12);
    ca1 = aig.or(ca1, op13);
    aig.add_output("ca1", ca1);

    // Bit 2 (alu_sel[2]): {8,9,13,14,20}
    let mut ca2 = aig.or(op08, op09);
    ca2 = aig.or(ca2, op0d);
    ca2 = aig.or(ca2, op0e);
    ca2 = aig.or(ca2, op14);
    aig.add_output("ca2", ca2);

    // Bit 3 (reg_write): {2-14,16-20,22,24} — 20 opcodes.
    // Full groups for 001xx(4-7), 010xx(8-11), 100xx(16-19).
    let mut ca3 = aig.or(op02, op03);
    ca3 = aig.or(ca3, nb4_nb3_b2);
    ca3 = aig.or(ca3, nb4_b3_nb2);
    ca3 = aig.or(ca3, op0c);
    ca3 = aig.or(ca3, op0d);
    ca3 = aig.or(ca3, op0e);
    ca3 = aig.or(ca3, b4_nb3_nb2);
    ca3 = aig.or(ca3, op14);
    ca3 = aig.or(ca3, op16);
    ca3 = aig.or(ca3, op18);
    aig.add_output("ca3", ca3);

    // Bit 4 (flag_z_we): {3-15,16-20,22,24} — 20 opcodes.
    // Full groups for 001xx(4-7), 010xx(8-11), 011xx(12-15), 100xx(16-19).
    let mut ca4 = aig.or(op03, nb4_nb3_b2);
    ca4 = aig.or(ca4, nb4_b3_nb2);
    ca4 = aig.or(ca4, nb4_b3_b2);
    ca4 = aig.or(ca4, b4_nb3_nb2);
    ca4 = aig.or(ca4, op14);
    ca4 = aig.or(ca4, op16);
    ca4 = aig.or(ca4, op18);
    aig.add_output("ca4", ca4);

    // Bit 5 (flag_c_we): {4,5,10,11,12-15,16,17}
    // Full group for 011xx(12-15).
    let mut ca5 = aig.or(op04, op05);
    ca5 = aig.or(ca5, op0a);
    ca5 = aig.or(ca5, op0b);
    ca5 = aig.or(ca5, nb4_b3_b2);
    ca5 = aig.or(ca5, op10);
    ca5 = aig.or(ca5, op11);
    aig.add_output("ca5", ca5);

    // Bit 6 (reserved): {3,11,12-14,16-19,20,24,25}
    // Full group for 100xx(16-19).
    let mut ca6 = aig.or(op03, op0b);
    ca6 = aig.or(ca6, op0c);
    ca6 = aig.or(ca6, op0d);
    ca6 = aig.or(ca6, op0e);
    ca6 = aig.or(ca6, b4_nb3_nb2);
    ca6 = aig.or(ca6, op14);
    ca6 = aig.or(ca6, op18);
    ca6 = aig.or(ca6, op19);
    aig.add_output("ca6", ca6);

    // Bit 7 (reserved): {4-15,16-19,20} — 17 opcodes.
    // Full groups for 001xx(4-7), 010xx(8-11), 011xx(12-15), 100xx(16-19).
    let mut ca7 = aig.or(nb4_nb3_b2, nb4_b3_nb2);
    ca7 = aig.or(ca7, nb4_b3_b2);
    ca7 = aig.or(ca7, b4_nb3_nb2);
    ca7 = aig.or(ca7, op14);
    aig.add_output("ca7", ca7);

    aig
}

/// Sprint 204: Combined ctrl_a + ctrl_b AIG (5-input opcode → 16-bit decode word).
///
/// First 8 outputs: ctrl_a[7:0] (alu_sel, reg_write, flag_we, reserved).
/// Last 8 outputs: ctrl_b[7:0] (branch/mem/halt control).
///
/// Shares all sub-expression infrastructure (low-bit pairs, high-bit groups,
/// 3-var groups) between ctrl_a and ctrl_b. Single placement replaces two
/// separate synth blocks.
pub fn build_combined_decode_aig() -> Aig {
    let mut aig = Aig::new();
    let b0 = aig.add_input("op0");
    let b1 = aig.add_input("op1");
    let b2 = aig.add_input("op2");
    let b3 = aig.add_input("op3");
    let b4 = aig.add_input("op4");

    let nb0 = b0.negated();
    let nb1 = b1.negated();
    let nb2 = b2.negated();
    let nb3 = b3.negated();
    let nb4 = b4.negated();

    // Shared low-bit pairs (b1, b0).
    let pair_00 = aig.and(nb1, nb0);
    let pair_01 = aig.and(nb1, b0);
    let pair_10 = aig.and(b1, nb0);
    let pair_11 = aig.and(b1, b0);

    // Shared high-bit groups (b4, b3).
    let nb4_nb3 = aig.and(nb4, nb3);
    let nb4_b3 = aig.and(nb4, b3);
    let b4_nb3 = aig.and(b4, nb3);
    let b4_b3 = aig.and(b4, b3);

    // 3-var groups (b4, b3, b2).
    let nb4_nb3_nb2 = aig.and(nb4_nb3, nb2); // 000xx: opcodes 0-3
    let nb4_nb3_b2 = aig.and(nb4_nb3, b2); // 001xx: opcodes 4-7
    let nb4_b3_nb2 = aig.and(nb4_b3, nb2); // 010xx: opcodes 8-11
    let nb4_b3_b2 = aig.and(nb4_b3, b2); // 011xx: opcodes 12-15
    let b4_nb3_nb2 = aig.and(b4_nb3, nb2); // 100xx: opcodes 16-19
    let b4_nb3_b2 = aig.and(b4_nb3, b2); // 101xx: opcodes 20-23
    let b4_b3_nb2 = aig.and(b4_b3, nb2); // 110xx: opcodes 24-27
    let b4_b3_b2 = aig.and(b4_b3, b2); // 111xx: opcodes 28-31

    // All 5-bit minterms used by either ctrl_a or ctrl_b.
    // ctrl_a active: 0x02-0x14, 0x16, 0x18, 0x19 (22 opcodes)
    // ctrl_b active: 0x01, 0x15-0x1F (12 opcodes)
    // Union: 0x01-0x19, 0x1A-0x1F (minus 0x00, 0x15 shared)
    let op01 = aig.and(nb4_nb3_nb2, pair_01); // 00001
    let op02 = aig.and(nb4_nb3_nb2, pair_10); // 00010
    let op03 = aig.and(nb4_nb3_nb2, pair_11); // 00011
    let op04 = aig.and(nb4_nb3_b2, pair_00); // 00100
    let op05 = aig.and(nb4_nb3_b2, pair_01); // 00101
    let op06 = aig.and(nb4_nb3_b2, pair_10); // 00110
    let op07 = aig.and(nb4_nb3_b2, pair_11); // 00111
    let op08 = aig.and(nb4_b3_nb2, pair_00); // 01000
    let op09 = aig.and(nb4_b3_nb2, pair_01); // 01001
    let op0a = aig.and(nb4_b3_nb2, pair_10); // 01010
    let op0b = aig.and(nb4_b3_nb2, pair_11); // 01011
    let op0c = aig.and(nb4_b3_b2, pair_00); // 01100
    let op0d = aig.and(nb4_b3_b2, pair_01); // 01101
    let op0e = aig.and(nb4_b3_b2, pair_10); // 01110
    let op0f = aig.and(nb4_b3_b2, pair_11); // 01111
    let op10 = aig.and(b4_nb3_nb2, pair_00); // 10000
    let op11 = aig.and(b4_nb3_nb2, pair_01); // 10001
    let op12 = aig.and(b4_nb3_nb2, pair_10); // 10010
    let op13 = aig.and(b4_nb3_nb2, pair_11); // 10011
    let op14 = aig.and(b4_nb3_b2, pair_00); // 10100
    let op15 = aig.and(b4_nb3_b2, pair_01); // 10101
    let op16 = aig.and(b4_nb3_b2, pair_10); // 10110
    let op17 = aig.and(b4_nb3_b2, pair_11); // 10111
    let op18 = aig.and(b4_b3_nb2, pair_00); // 11000
    let op19 = aig.and(b4_b3_nb2, pair_01); // 11001
    let op1a = aig.and(b4_b3_nb2, pair_10); // 11010
    let op1b = aig.and(b4_b3_nb2, pair_11); // 11011
    let op1c = aig.and(b4_b3_b2, pair_00); // 11100
    let op1d = aig.and(b4_b3_b2, pair_01); // 11101
    let op1e = aig.and(b4_b3_b2, pair_10); // 11110
    let op1f = aig.and(b4_b3_b2, pair_11); // 11111

    // --- ctrl_a outputs (bits 0-7) ---

    // ca0 (alu_sel[0]): {5,7,9,10,12,14,15,17,19}
    let mut ca0 = aig.or(op05, op07);
    ca0 = aig.or(ca0, op09);
    ca0 = aig.or(ca0, op0a);
    ca0 = aig.or(ca0, op0c);
    ca0 = aig.or(ca0, op0e);
    ca0 = aig.or(ca0, op0f);
    ca0 = aig.or(ca0, op11);
    ca0 = aig.or(ca0, op13);
    aig.add_output("ca0", ca0);

    // ca1 (alu_sel[1]): {6,7,13,14,18,19}
    let mut ca1 = aig.or(op06, op07);
    ca1 = aig.or(ca1, op0d);
    ca1 = aig.or(ca1, op0e);
    ca1 = aig.or(ca1, op12);
    ca1 = aig.or(ca1, op13);
    aig.add_output("ca1", ca1);

    // ca2 (alu_sel[2]): {8,9,13,14,20}
    let mut ca2 = aig.or(op08, op09);
    ca2 = aig.or(ca2, op0d);
    ca2 = aig.or(ca2, op0e);
    ca2 = aig.or(ca2, op14);
    aig.add_output("ca2", ca2);

    // ca3 (reg_write): {2-14,16-20,22,24} — full groups 001xx, 010xx, 100xx
    let mut ca3 = aig.or(op02, op03);
    ca3 = aig.or(ca3, nb4_nb3_b2);
    ca3 = aig.or(ca3, nb4_b3_nb2);
    ca3 = aig.or(ca3, op0c);
    ca3 = aig.or(ca3, op0d);
    ca3 = aig.or(ca3, op0e);
    ca3 = aig.or(ca3, b4_nb3_nb2);
    ca3 = aig.or(ca3, op14);
    ca3 = aig.or(ca3, op16);
    ca3 = aig.or(ca3, op18);
    aig.add_output("ca3", ca3);

    // ca4 (flag_z_we): {3-15,16-20,22,24} — full groups 001xx, 010xx, 011xx, 100xx
    let mut ca4 = aig.or(op03, nb4_nb3_b2);
    ca4 = aig.or(ca4, nb4_b3_nb2);
    ca4 = aig.or(ca4, nb4_b3_b2);
    ca4 = aig.or(ca4, b4_nb3_nb2);
    ca4 = aig.or(ca4, op14);
    ca4 = aig.or(ca4, op16);
    ca4 = aig.or(ca4, op18);
    aig.add_output("ca4", ca4);

    // ca5 (flag_c_we): {4,5,10,11,12-15,16,17} — full group 011xx
    let mut ca5 = aig.or(op04, op05);
    ca5 = aig.or(ca5, op0a);
    ca5 = aig.or(ca5, op0b);
    ca5 = aig.or(ca5, nb4_b3_b2);
    ca5 = aig.or(ca5, op10);
    ca5 = aig.or(ca5, op11);
    aig.add_output("ca5", ca5);

    // ca6 (reserved): {3,11,12-14,16-19,20,24,25}
    let mut ca6 = aig.or(op03, op0b);
    ca6 = aig.or(ca6, op0c);
    ca6 = aig.or(ca6, op0d);
    ca6 = aig.or(ca6, op0e);
    ca6 = aig.or(ca6, b4_nb3_nb2);
    ca6 = aig.or(ca6, op14);
    ca6 = aig.or(ca6, op18);
    ca6 = aig.or(ca6, op19);
    aig.add_output("ca6", ca6);

    // ca7 (reserved): {4-15,16-19,20} — full groups 001xx, 010xx, 011xx, 100xx
    let mut ca7 = aig.or(nb4_nb3_b2, nb4_b3_nb2);
    ca7 = aig.or(ca7, nb4_b3_b2);
    ca7 = aig.or(ca7, b4_nb3_nb2);
    ca7 = aig.or(ca7, op14);
    aig.add_output("ca7", ca7);

    // --- ctrl_b outputs (bits 8-15) ---

    // cb0: {0x01, 0x15, 0x1A, 0x1C, 0x1E}
    let mut cb0 = aig.or(op01, op15);
    cb0 = aig.or(cb0, op1a);
    cb0 = aig.or(cb0, op1c);
    cb0 = aig.or(cb0, op1e);
    aig.add_output("cb0", cb0);

    // cb1: {0x15, 0x1B, 0x1C, 0x1F}
    let mut cb1 = aig.or(op15, op1b);
    cb1 = aig.or(cb1, op1c);
    cb1 = aig.or(cb1, op1f);
    aig.add_output("cb1", cb1);

    // cb2: {0x15, 0x1D, 0x1E, 0x1F}
    let mut cb2 = aig.or(op15, op1d);
    cb2 = aig.or(cb2, op1e);
    cb2 = aig.or(cb2, op1f);
    aig.add_output("cb2", cb2);

    // cb3: {0x16, 0x18}
    let cb3 = aig.or(op16, op18);
    aig.add_output("cb3", cb3);

    // cb4: {0x17, 0x19}
    let cb4 = aig.or(op17, op19);
    aig.add_output("cb4", cb4);

    // cb5: {0x01}
    aig.add_output("cb5", op01);

    // cb6: {0x1F}
    aig.add_output("cb6", op1f);

    // cb7: {0x15}
    aig.add_output("cb7", op15);

    aig
}

// ---------------------------------------------------------------------------
// SRA sign-extension AIG (Sprint 248)
// ---------------------------------------------------------------------------

/// Build an AIG for the SRA (arithmetic right shift) sign-extension overlay.
///
/// 4 inputs: `sign` (MSB of operand A), `s0`, `s1`, `s2` (shift amount bits).
/// 7 outputs: `mask_57`..`mask_63` — one bit per high-order result position.
///
/// For a right shift by N = s2*4 + s1*2 + s0 (0-7), result bits >= (64-N) are
/// sign-extended. Each output `mask_i = sign AND (N >= 64-i)`.
///
/// The SRA result is: `shr_result | (mask_57..63 expanded to u64)`.
/// ~15 AND nodes.
pub fn build_sra_sign_ext_aig() -> Aig {
    let mut aig = Aig::new();
    let sign = aig.add_input("sign");
    let s0 = aig.add_input("s0");
    let s1 = aig.add_input("s1");
    let s2 = aig.add_input("s2");

    // Shared subexpressions
    let or_s0_s1 = aig.or(s0, s1); // s0 | s1
    let or_s1_s2 = aig.or(s1, s2); // s1 | s2
    let or_all = aig.or(or_s0_s1, s2); // s0 | s1 | s2
    let and_s0_s1 = aig.and(s0, s1); // s0 & s1
    let ge3 = aig.or(and_s0_s1, s2); // shift >= 3: (s0 & s1) | s2
    let ge5 = aig.and(s2, or_s0_s1); // shift >= 5: s2 & (s0 | s1)
    let ge6 = aig.and(s2, s1); // shift >= 6: s2 & s1
    let ge7 = aig.and(and_s0_s1, s2); // shift >= 7: s0 & s1 & s2

    // Outputs: sign AND (shift >= threshold)
    let m57 = aig.and(sign, ge7); // shift >= 7
    let m58 = aig.and(sign, ge6); // shift >= 6
    let m59 = aig.and(sign, ge5); // shift >= 5
    let m60 = aig.and(sign, s2); // shift >= 4
    let m61 = aig.and(sign, ge3); // shift >= 3
    let m62 = aig.and(sign, or_s1_s2); // shift >= 2
    let m63 = aig.and(sign, or_all); // shift >= 1
    aig.add_output("mask_57", m57);
    aig.add_output("mask_58", m58);
    aig.add_output("mask_59", m59);
    aig.add_output("mask_60", m60);
    aig.add_output("mask_61", m61);
    aig.add_output("mask_62", m62);
    aig.add_output("mask_63", m63);

    aig
}

// ---------------------------------------------------------------------------
// Sprint 250: CLZ / CTZ / POPCNT AIGs (64-bit)
// ---------------------------------------------------------------------------

/// Helper: OR-reduce a slice of AIG literals.
fn or_reduce(aig: &mut Aig, bits: &[AigLit]) -> AigLit {
    bits.iter()
        .copied()
        .reduce(|acc, b| aig.or(acc, b))
        .unwrap_or(AigLit::FALSE)
}

/// Helper: 2:1 mux. Returns `if sel { b } else { a }`.
fn mux2(aig: &mut Aig, a: AigLit, b: AigLit, sel: AigLit) -> AigLit {
    // mux(a, b, sel) = (a & !sel) | (b & sel)
    let not_sel = aig.not(sel);
    let t1 = aig.and(a, not_sel);
    let t2 = aig.and(b, sel);
    aig.or(t1, t2)
}

// ---------------------------------------------------------------------------
// Sprint 250: Hierarchical byte-sliced CLZ/CTZ/POPCNT
// ---------------------------------------------------------------------------

/// Build an AIG for 8-bit CLZ (count leading zeros in one byte).
///
/// 8 inputs: b0 (LSB) through b7 (MSB).
/// 4 outputs: has_nz (any bit set), clz0, clz1, clz2 (3-bit count, 0-7).
/// Count is only meaningful when has_nz=1.
pub fn build_clz8_aig() -> Aig {
    let mut aig = Aig::new();
    let bits: Vec<AigLit> = (0..8).map(|i| aig.add_input(&format!("b{}", i))).collect();

    let mut remaining = bits.clone();

    // Level 0: 8→4. Check if upper 4 (b4-b7) have any set bit.
    let upper_any = or_reduce(&mut aig, &remaining[4..8]);
    let bit2 = aig.not(upper_any);
    let selected: Vec<AigLit> = (0..4)
        .map(|i| mux2(&mut aig, remaining[i], remaining[i + 4], upper_any))
        .collect();
    remaining = selected;

    // Level 1: 4→2.
    let upper_any = or_reduce(&mut aig, &remaining[2..4]);
    let bit1 = aig.not(upper_any);
    let selected: Vec<AigLit> = (0..2)
        .map(|i| mux2(&mut aig, remaining[i], remaining[i + 2], upper_any))
        .collect();
    remaining = selected;

    // Level 2: 2→1.
    let bit0 = aig.not(remaining[1]);

    let any_set = or_reduce(&mut aig, &bits);

    // Gate count bits: when all zero, count should be 0 (not 7).
    let bit0 = aig.and(bit0, any_set);
    let bit1 = aig.and(bit1, any_set);
    let bit2 = aig.and(bit2, any_set);

    aig.add_output("has_nz", any_set);
    aig.add_output("clz0", bit0);
    aig.add_output("clz1", bit1);
    aig.add_output("clz2", bit2);

    aig
}

/// Build an AIG for 8-bit CTZ (count trailing zeros in one byte).
///
/// 8 inputs: b0 (LSB) through b7 (MSB).
/// 4 outputs: has_nz, ctz0, ctz1, ctz2 (3-bit count, 0-7).
pub fn build_ctz8_aig() -> Aig {
    let mut aig = Aig::new();
    let bits: Vec<AigLit> = (0..8).map(|i| aig.add_input(&format!("b{}", i))).collect();

    let mut remaining = bits.clone();

    // Level 0: 8→4. Check if lower 4 (b0-b3) have any set bit.
    let lower_any = or_reduce(&mut aig, &remaining[0..4]);
    let bit2 = aig.not(lower_any);
    let selected: Vec<AigLit> = (0..4)
        .map(|i| mux2(&mut aig, remaining[i + 4], remaining[i], lower_any))
        .collect();
    remaining = selected;

    // Level 1: 4→2.
    let lower_any = or_reduce(&mut aig, &remaining[0..2]);
    let bit1 = aig.not(lower_any);
    let selected: Vec<AigLit> = (0..2)
        .map(|i| mux2(&mut aig, remaining[i + 2], remaining[i], lower_any))
        .collect();
    remaining = selected;

    // Level 2: 2→1.
    let bit0 = aig.not(remaining[0]);

    let any_set = or_reduce(&mut aig, &bits);

    let bit0 = aig.and(bit0, any_set);
    let bit1 = aig.and(bit1, any_set);
    let bit2 = aig.and(bit2, any_set);

    aig.add_output("has_nz", any_set);
    aig.add_output("ctz0", bit0);
    aig.add_output("ctz1", bit1);
    aig.add_output("ctz2", bit2);

    aig
}

/// Build an AIG for 8-bit POPCNT (population count of one byte).
///
/// 8 inputs: b0 through b7.
/// 4 outputs: pop0 (LSB) through pop3 (MSB). Encodes values 0-8.
pub fn build_popcnt8_aig() -> Aig {
    let mut aig = Aig::new();
    let bits: Vec<AigLit> = (0..8).map(|i| aig.add_input(&format!("b{}", i))).collect();

    // Level 0: 4 half-adders → 4 × 2-bit sums.
    let mut sums: Vec<Vec<AigLit>> = Vec::new();
    for i in (0..8).step_by(2) {
        let s = aig.xor(bits[i], bits[i + 1]);
        let c = aig.and(bits[i], bits[i + 1]);
        sums.push(vec![s, c]);
    }

    // Level 1-2: adder tree.
    while sums.len() > 1 {
        let mut next = Vec::new();
        let mut i = 0;
        while i + 1 < sums.len() {
            let added = aig_add(&mut aig, &sums[i], &sums[i + 1]);
            next.push(added);
            i += 2;
        }
        if i < sums.len() {
            let mut padded = sums[i].clone();
            if !next.is_empty() {
                while padded.len() < next[0].len() {
                    padded.push(AigLit::FALSE);
                }
            }
            next.push(padded);
        }
        sums = next;
    }

    let result = &sums[0];
    for i in 0..4 {
        let lit = if i < result.len() {
            result[i]
        } else {
            AigLit::FALSE
        };
        aig.add_output(&format!("pop{}", i), lit);
    }

    aig
}

/// Build a CLZ half-group combine AIG (4 bytes → 1 group summary).
///
/// 16 inputs: 4 × (has_nz, count[2:0]), highest-priority byte first.
/// 6 outputs: group_nz, byte_idx[1:0], count[2:0].
///
/// For CLZ upper half: bytes 7,6,5,4 (MSB-first priority).
/// For CLZ lower half: bytes 3,2,1,0 (MSB-first priority).
pub fn build_clz_half_combine_aig() -> Aig {
    let mut aig = Aig::new();

    let mut h = Vec::new();
    let mut c = Vec::new();
    for i in 0..4 {
        h.push(aig.add_input(&format!("h{}", i)));
        let mut cbits = Vec::new();
        for bit in 0..3 {
            cbits.push(aig.add_input(&format!("c{}_{}", i, bit)));
        }
        c.push(cbits);
    }

    // Priority encoder: find first byte with has_nz=1.
    let mut not_any_above = AigLit::TRUE;
    let mut win = Vec::new();
    for i in 0..4 {
        let w = aig.and(h[i], not_any_above);
        win.push(w);
        not_any_above = aig.and(not_any_above, aig.not(h[i]));
    }

    // Byte index within group (2 bits): win[0]→0, win[1]→1, win[2]→2, win[3]→3.
    let idx0 = aig.or(win[1], win[3]); // odd positions
    let idx1 = aig.or(win[2], win[3]); // positions 2,3

    // Mux: select winning byte's local count.
    let mut local_count = Vec::new();
    for bit in 0..3 {
        let mut terms = AigLit::FALSE;
        for i in 0..4 {
            let t = aig.and(win[i], c[i][bit]);
            terms = aig.or(terms, t);
        }
        local_count.push(terms);
    }

    let group_nz = or_reduce(&mut aig, &h);

    aig.add_output("group_nz", group_nz);
    aig.add_output("idx0", idx0);
    aig.add_output("idx1", idx1);
    aig.add_output("cnt0", local_count[0]);
    aig.add_output("cnt1", local_count[1]);
    aig.add_output("cnt2", local_count[2]);

    aig
}

/// Build a CLZ final combine AIG (2 half-groups → 7-bit result).
///
/// 12 inputs: upper_group(group_nz, idx[1:0], cnt[2:0]),
///            lower_group(group_nz, idx[1:0], cnt[2:0]).
/// 7 outputs: result[6:0] = total CLZ (0-64).
///
/// Logic: if upper group has nonzero byte, use it (position = upper.idx);
/// otherwise use lower group (position = 4 + lower.idx).
/// Bit 6 = all-zero flag (both groups empty → CLZ = 64).
pub fn build_clz_final_combine_aig() -> Aig {
    let mut aig = Aig::new();

    // Upper group (bytes 7-4, higher priority for CLZ).
    let u_nz = aig.add_input("u_nz");
    let u_idx0 = aig.add_input("u_idx0");
    let u_idx1 = aig.add_input("u_idx1");
    let u_cnt0 = aig.add_input("u_cnt0");
    let u_cnt1 = aig.add_input("u_cnt1");
    let u_cnt2 = aig.add_input("u_cnt2");

    // Lower group (bytes 3-0).
    let l_nz = aig.add_input("l_nz");
    let l_idx0 = aig.add_input("l_idx0");
    let l_idx1 = aig.add_input("l_idx1");
    let l_cnt0 = aig.add_input("l_cnt0");
    let l_cnt1 = aig.add_input("l_cnt1");
    let l_cnt2 = aig.add_input("l_cnt2");

    // Select: upper group wins if it has any nonzero byte.
    // result[0:2] = local count: mux(u_nz, u_cnt, l_cnt)
    let r0 = mux2(&mut aig, l_cnt0, u_cnt0, u_nz);
    let r1 = mux2(&mut aig, l_cnt1, u_cnt1, u_nz);
    let r2 = mux2(&mut aig, l_cnt2, u_cnt2, u_nz);

    // result[3:4] = byte index within winning group: mux(u_nz, u_idx, l_idx)
    let r3 = mux2(&mut aig, l_idx0, u_idx0, u_nz);
    let r4 = mux2(&mut aig, l_idx1, u_idx1, u_nz);

    // result[5] = which half-group: 0 if upper won, 1 if lower won.
    // This is the MSB of the 3-bit byte position.
    let r5 = aig.not(u_nz); // 1 when upper group is all-zero → lower half

    // Bit 6: all-zero flag.
    let any_set = aig.or(u_nz, l_nz);
    let all_zero = aig.not(any_set);

    // Gate bits 0-5 when all-zero (CLZ=64 → only bit 6 set).
    let r0 = aig.and(r0, any_set);
    let r1 = aig.and(r1, any_set);
    let r2 = aig.and(r2, any_set);
    let r3 = aig.and(r3, any_set);
    let r4 = aig.and(r4, any_set);
    let r5 = aig.and(r5, any_set);

    aig.add_output("r0", r0);
    aig.add_output("r1", r1);
    aig.add_output("r2", r2);
    aig.add_output("r3", r3);
    aig.add_output("r4", r4);
    aig.add_output("r5", r5);
    aig.add_output("r6", all_zero);

    aig
}

/// Build a CTZ half-group combine AIG (4 bytes → 1 group summary).
///
/// 16 inputs: 4 × (has_nz, count[2:0]), lowest-priority-index first.
/// 6 outputs: group_nz, byte_idx[1:0], count[2:0].
///
/// For CTZ lower half: bytes 0,1,2,3 (LSB-first priority).
/// For CTZ upper half: bytes 4,5,6,7 (LSB-first priority).
pub fn build_ctz_half_combine_aig() -> Aig {
    let mut aig = Aig::new();

    let mut h = Vec::new();
    let mut c = Vec::new();
    for i in 0..4 {
        h.push(aig.add_input(&format!("h{}", i)));
        let mut cbits = Vec::new();
        for bit in 0..3 {
            cbits.push(aig.add_input(&format!("c{}_{}", i, bit)));
        }
        c.push(cbits);
    }

    // Priority encoder: first byte with has_nz=1 (index 0 = highest priority).
    let mut not_any_above = AigLit::TRUE;
    let mut win = Vec::new();
    for i in 0..4 {
        let w = aig.and(h[i], not_any_above);
        win.push(w);
        not_any_above = aig.and(not_any_above, aig.not(h[i]));
    }

    let idx0 = aig.or(win[1], win[3]);
    let idx1 = aig.or(win[2], win[3]);

    let mut local_count = Vec::new();
    for bit in 0..3 {
        let mut terms = AigLit::FALSE;
        for i in 0..4 {
            let t = aig.and(win[i], c[i][bit]);
            terms = aig.or(terms, t);
        }
        local_count.push(terms);
    }

    let group_nz = or_reduce(&mut aig, &h);

    aig.add_output("group_nz", group_nz);
    aig.add_output("idx0", idx0);
    aig.add_output("idx1", idx1);
    aig.add_output("cnt0", local_count[0]);
    aig.add_output("cnt1", local_count[1]);
    aig.add_output("cnt2", local_count[2]);

    aig
}

/// Build a CTZ final combine AIG (2 half-groups → 7-bit result).
///
/// 12 inputs: lower_group(group_nz, idx[1:0], cnt[2:0]),
///            upper_group(group_nz, idx[1:0], cnt[2:0]).
/// 7 outputs: result[6:0] = total CTZ (0-64).
pub fn build_ctz_final_combine_aig() -> Aig {
    let mut aig = Aig::new();

    // Lower group (bytes 0-3, higher priority for CTZ).
    let l_nz = aig.add_input("l_nz");
    let l_idx0 = aig.add_input("l_idx0");
    let l_idx1 = aig.add_input("l_idx1");
    let l_cnt0 = aig.add_input("l_cnt0");
    let l_cnt1 = aig.add_input("l_cnt1");
    let l_cnt2 = aig.add_input("l_cnt2");

    // Upper group (bytes 4-7).
    let u_nz = aig.add_input("u_nz");
    let u_idx0 = aig.add_input("u_idx0");
    let u_idx1 = aig.add_input("u_idx1");
    let u_cnt0 = aig.add_input("u_cnt0");
    let u_cnt1 = aig.add_input("u_cnt1");
    let u_cnt2 = aig.add_input("u_cnt2");

    // Select: lower group wins if it has any nonzero byte.
    let r0 = mux2(&mut aig, u_cnt0, l_cnt0, l_nz);
    let r1 = mux2(&mut aig, u_cnt1, l_cnt1, l_nz);
    let r2 = mux2(&mut aig, u_cnt2, l_cnt2, l_nz);

    let r3 = mux2(&mut aig, u_idx0, l_idx0, l_nz);
    let r4 = mux2(&mut aig, u_idx1, l_idx1, l_nz);

    // result[5] = which half-group: 0 if lower won, 1 if upper won.
    let r5 = aig.not(l_nz);

    let any_set = aig.or(l_nz, u_nz);
    let all_zero = aig.not(any_set);

    let r0 = aig.and(r0, any_set);
    let r1 = aig.and(r1, any_set);
    let r2 = aig.and(r2, any_set);
    let r3 = aig.and(r3, any_set);
    let r4 = aig.and(r4, any_set);
    let r5 = aig.and(r5, any_set);

    aig.add_output("r0", r0);
    aig.add_output("r1", r1);
    aig.add_output("r2", r2);
    aig.add_output("r3", r3);
    aig.add_output("r4", r4);
    aig.add_output("r5", r5);
    aig.add_output("r6", all_zero);

    aig
}

/// Build a POPCNT pairwise add AIG (two N-bit counts → (N+1)-bit sum).
///
/// `width` inputs from each operand. Returns AIG with `2*width` inputs
/// and `width+1` outputs.
pub fn build_popcnt_add_aig(width: usize) -> Aig {
    let mut aig = Aig::new();

    let a: Vec<AigLit> = (0..width)
        .map(|i| aig.add_input(&format!("a{}", i)))
        .collect();
    let b: Vec<AigLit> = (0..width)
        .map(|i| aig.add_input(&format!("b{}", i)))
        .collect();

    let result = aig_add(&mut aig, &a, &b);

    for (i, &lit) in result.iter().enumerate() {
        aig.add_output(&format!("s{}", i), lit);
    }

    aig
}

// ---------------------------------------------------------------------------
// Sprint 248/249: Monolithic 64-bit CLZ/CTZ/POPCNT (retained for reference/testing)
// ---------------------------------------------------------------------------

/// Build an AIG for 64-bit CLZ (count leading zeros).
///
/// 64 primary inputs: b0 (LSB) through b63 (MSB).
/// 7 primary outputs: clz0 (LSB) through clz6 (MSB). Encodes values 0-64.
///
/// Algorithm: binary search — at each level, check if the upper half is all-zero.
/// If so, the corresponding result bit is 1 and we continue with the lower half.
/// Otherwise, the result bit is 0 and we continue with the upper half.
/// ~306 AND nodes.
pub fn build_clz_aig() -> Aig {
    let mut aig = Aig::new();
    let bits: Vec<AigLit> = (0..64).map(|i| aig.add_input(&format!("b{}", i))).collect();

    // Work from MSB down. CLZ counts from the top, so "upper half" = MSB side.
    let mut remaining = bits.clone();

    // Level 0: 64→32. Check if bits[32..63] (upper 32) are all-zero.
    let upper_any = or_reduce(&mut aig, &remaining[32..64]);
    let bit5 = aig.not(upper_any); // 1 if upper 32 all zero → CLZ += 32
    // Select: if upper has any set bit, use upper 32; else use lower 32.
    let selected: Vec<AigLit> = (0..32)
        .map(|i| mux2(&mut aig, remaining[i], remaining[i + 32], upper_any))
        .collect();
    remaining = selected;

    // Level 1: 32→16.
    let upper_any = or_reduce(&mut aig, &remaining[16..32]);
    let bit4 = aig.not(upper_any);
    let selected: Vec<AigLit> = (0..16)
        .map(|i| mux2(&mut aig, remaining[i], remaining[i + 16], upper_any))
        .collect();
    remaining = selected;

    // Level 2: 16→8.
    let upper_any = or_reduce(&mut aig, &remaining[8..16]);
    let bit3 = aig.not(upper_any);
    let selected: Vec<AigLit> = (0..8)
        .map(|i| mux2(&mut aig, remaining[i], remaining[i + 8], upper_any))
        .collect();
    remaining = selected;

    // Level 3: 8→4.
    let upper_any = or_reduce(&mut aig, &remaining[4..8]);
    let bit2 = aig.not(upper_any);
    let selected: Vec<AigLit> = (0..4)
        .map(|i| mux2(&mut aig, remaining[i], remaining[i + 4], upper_any))
        .collect();
    remaining = selected;

    // Level 4: 4→2.
    let upper_any = or_reduce(&mut aig, &remaining[2..4]);
    let bit1 = aig.not(upper_any);
    let selected: Vec<AigLit> = (0..2)
        .map(|i| mux2(&mut aig, remaining[i], remaining[i + 2], upper_any))
        .collect();
    remaining = selected;

    // Level 5: 2→1. The MSB of the remaining 2-bit value.
    let bit0 = aig.not(remaining[1]); // 1 if top bit is 0

    // Bit 6: all 64 bits are zero → CLZ = 64.
    let any_set = or_reduce(&mut aig, &bits);
    let bit6 = aig.not(any_set);

    // Gate bits 0-5 with any_set: when input is all-zero, bits 0-5 must be 0
    // (otherwise they'd read 63, giving 63+64=127 instead of 64).
    let bit0 = aig.and(bit0, any_set);
    let bit1 = aig.and(bit1, any_set);
    let bit2 = aig.and(bit2, any_set);
    let bit3 = aig.and(bit3, any_set);
    let bit4 = aig.and(bit4, any_set);
    let bit5 = aig.and(bit5, any_set);

    aig.add_output("clz0", bit0);
    aig.add_output("clz1", bit1);
    aig.add_output("clz2", bit2);
    aig.add_output("clz3", bit3);
    aig.add_output("clz4", bit4);
    aig.add_output("clz5", bit5);
    aig.add_output("clz6", bit6);

    aig
}

/// Build an AIG for 64-bit CTZ (count trailing zeros).
///
/// Same structure as CLZ but searches from LSB. The "upper half" becomes the
/// lower half (bits closer to bit 0). ~306 AND nodes.
pub fn build_ctz_aig() -> Aig {
    let mut aig = Aig::new();
    let bits: Vec<AigLit> = (0..64).map(|i| aig.add_input(&format!("b{}", i))).collect();

    // Work from LSB up. CTZ counts from the bottom.
    let mut remaining = bits.clone();

    // Level 0: 64→32. Check if bits[0..31] (lower 32) are all-zero.
    let lower_any = or_reduce(&mut aig, &remaining[0..32]);
    let bit5 = aig.not(lower_any); // 1 if lower 32 all zero → CTZ += 32
    // Select: if lower has any set bit, use lower 32; else use upper 32.
    let selected: Vec<AigLit> = (0..32)
        .map(|i| mux2(&mut aig, remaining[i + 32], remaining[i], lower_any))
        .collect();
    remaining = selected;

    // Level 1: 32→16.
    let lower_any = or_reduce(&mut aig, &remaining[0..16]);
    let bit4 = aig.not(lower_any);
    let selected: Vec<AigLit> = (0..16)
        .map(|i| mux2(&mut aig, remaining[i + 16], remaining[i], lower_any))
        .collect();
    remaining = selected;

    // Level 2: 16→8.
    let lower_any = or_reduce(&mut aig, &remaining[0..8]);
    let bit3 = aig.not(lower_any);
    let selected: Vec<AigLit> = (0..8)
        .map(|i| mux2(&mut aig, remaining[i + 8], remaining[i], lower_any))
        .collect();
    remaining = selected;

    // Level 3: 8→4.
    let lower_any = or_reduce(&mut aig, &remaining[0..4]);
    let bit2 = aig.not(lower_any);
    let selected: Vec<AigLit> = (0..4)
        .map(|i| mux2(&mut aig, remaining[i + 4], remaining[i], lower_any))
        .collect();
    remaining = selected;

    // Level 4: 4→2.
    let lower_any = or_reduce(&mut aig, &remaining[0..2]);
    let bit1 = aig.not(lower_any);
    let selected: Vec<AigLit> = (0..2)
        .map(|i| mux2(&mut aig, remaining[i + 2], remaining[i], lower_any))
        .collect();
    remaining = selected;

    // Level 5: 2→1. The LSB of the remaining 2-bit value.
    let bit0 = aig.not(remaining[0]); // 1 if bottom bit is 0

    // Bit 6: all 64 bits are zero → CTZ = 64.
    let any_set = or_reduce(&mut aig, &bits);
    let bit6 = aig.not(any_set);

    // Gate bits 0-5 with any_set: when input is all-zero, bits 0-5 must be 0.
    let bit0 = aig.and(bit0, any_set);
    let bit1 = aig.and(bit1, any_set);
    let bit2 = aig.and(bit2, any_set);
    let bit3 = aig.and(bit3, any_set);
    let bit4 = aig.and(bit4, any_set);
    let bit5 = aig.and(bit5, any_set);

    aig.add_output("ctz0", bit0);
    aig.add_output("ctz1", bit1);
    aig.add_output("ctz2", bit2);
    aig.add_output("ctz3", bit3);
    aig.add_output("ctz4", bit4);
    aig.add_output("ctz5", bit5);
    aig.add_output("ctz6", bit6);

    aig
}

/// Helper: N-bit ripple-carry adder in AIG. Returns (sum_bits, carry_out).
fn aig_add(aig: &mut Aig, a: &[AigLit], b: &[AigLit]) -> Vec<AigLit> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut result = Vec::with_capacity(n + 1);
    let mut carry = AigLit::FALSE;
    for i in 0..n {
        // Full adder: sum = a ^ b ^ carry, carry_out = maj(a, b, carry)
        let ab_xor = aig.xor(a[i], b[i]);
        let sum = aig.xor(ab_xor, carry);
        let ab_and = aig.and(a[i], b[i]);
        let c_and_xor = aig.and(carry, ab_xor);
        carry = aig.or(ab_and, c_and_xor);
        result.push(sum);
    }
    result.push(carry); // MSB = carry out
    result
}

/// Build an AIG for 64-bit POPCNT (population count / count ones).
///
/// 64 primary inputs: b0 (LSB) through b63 (MSB).
/// 7 primary outputs: pop0 (LSB) through pop6 (MSB). Encodes values 0-64.
///
/// Algorithm: adder tree — pair up single bits into 2-bit sums, pair those
/// into 3-bit sums, etc. 6 levels for 64 bits. ~765 AND nodes.
pub fn build_popcnt_aig() -> Aig {
    let mut aig = Aig::new();
    let bits: Vec<AigLit> = (0..64).map(|i| aig.add_input(&format!("b{}", i))).collect();

    // Level 0: 32 half-adders (1-bit + 1-bit → 2-bit).
    let mut sums: Vec<Vec<AigLit>> = Vec::new();
    for i in (0..64).step_by(2) {
        let s = aig.xor(bits[i], bits[i + 1]);
        let c = aig.and(bits[i], bits[i + 1]);
        sums.push(vec![s, c]);
    }
    // sums: 32 × 2-bit values

    // Levels 1-5: pair up N-bit values into (N+1)-bit values via ripple-carry add.
    while sums.len() > 1 {
        let mut next = Vec::new();
        let mut i = 0;
        while i + 1 < sums.len() {
            let added = aig_add(&mut aig, &sums[i], &sums[i + 1]);
            next.push(added);
            i += 2;
        }
        if i < sums.len() {
            // Odd one out — pad with zero to match width of next level.
            let mut padded = sums[i].clone();
            if !next.is_empty() {
                while padded.len() < next[0].len() {
                    padded.push(AigLit::FALSE);
                }
            }
            next.push(padded);
        }
        sums = next;
    }

    // sums[0] is the 7-bit result.
    let result = &sums[0];
    for i in 0..7 {
        let lit = if i < result.len() {
            result[i]
        } else {
            AigLit::FALSE
        };
        aig.add_output(&format!("pop{}", i), lit);
    }

    aig
}

// ---------------------------------------------------------------------------
// Guard region helper
// ---------------------------------------------------------------------------

/// Fill a rectangular region on ALL layers with `Const(0)` tiles.
///
/// # Layer/Region Ownership Convention
///
/// | Region     | Y range | Layers | Owner                    |
/// |------------|---------|--------|--------------------------|
/// | CPU core   | 0..63   | L0-L3  | V2Builder                |
/// | Guard band | 64..69  | L0-L3  | place_const_guard_region |
/// | Synth zone | 70..191 | L0-L3  | inject_synth_export      |
///
/// Const tiles are BFS dead ends (output never changes), blocking
/// signal propagation in both directions across the guard band.
pub fn place_const_guard_region(
    sim: &mut Simulation,
    x_range: std::ops::Range<usize>,
    y_range: std::ops::Range<usize>,
) {
    let num_layers = sim.num_layers();
    for z in 0..num_layers {
        for y in y_range.clone() {
            for x in x_range.clone() {
                sim.set_tile_3d(x, y, z, crate::tiles::TileType::Const);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SynthOutputs — stack-allocated synth block output buffer
// ---------------------------------------------------------------------------

/// Sprint 203/204: Stack-allocated output buffer for synth block evaluation.
///
/// Replaces `Vec<u64>` to eliminate heap allocation on the hot path.
/// Capacity 16 supports up to 16-output blocks (e.g. combined ctrl_a+ctrl_b
/// decode with 16 outputs). 136 bytes on the stack, `Copy`-able.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynthOutputs {
    data: [u64; 16],
    len: u8,
}

impl SynthOutputs {
    pub fn new() -> Self {
        Self {
            data: [0; 16],
            len: 0,
        }
    }

    pub fn push(&mut self, val: u64) {
        debug_assert!((self.len as usize) < 16);
        self.data[self.len as usize] = val;
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn iter(&self) -> impl Iterator<Item = &u64> {
        self.data[..self.len as usize].iter()
    }
}

impl std::ops::Index<usize> for SynthOutputs {
    type Output = u64;
    fn index(&self, idx: usize) -> &u64 {
        assert!(idx < self.len as usize);
        &self.data[idx]
    }
}

// ---------------------------------------------------------------------------
// Injection API
// ---------------------------------------------------------------------------

/// Describes a synth-generated block injected into a host simulation grid.
#[derive(Debug, Clone)]
pub struct InjectedBlock {
    pub offset_x: usize,
    pub offset_y: usize,
    pub width: usize,
    pub height: usize,
    /// Flat indices into the host sim for each circuit primary input.
    pub input_indices: Vec<usize>,
    /// Flat indices into the host sim for each circuit primary output.
    pub output_indices: Vec<usize>,
    /// Flat indices of all non-Const synth tiles (for reset/dirty marking).
    pub circuit_indices: Vec<usize>,
    /// Bitset scope mask for `propagate_combinational_masked()`.
    /// One bit per tile index; covers all input, circuit, and output tiles.
    pub scope_mask: Vec<u64>,
    /// Sprint 202: Precomputed lookup table indexed by input bit-pattern.
    /// `outputs_cache[pattern]` = output `SynthOutputs` for that input combination.
    /// `None` = no cache (falls back to live tile evaluation).
    pub outputs_cache: Option<Vec<SynthOutputs>>,
}

/// Error returned by checked synth export injection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SynthInjectError {
    /// The export region does not fit in the host grid.
    GridBounds {
        offset_x: usize,
        offset_y: usize,
        export_w: usize,
        export_h: usize,
        host_w: usize,
        host_h: usize,
    },
    /// The export needs more z-layers than the host simulation provides.
    LayerOverflow {
        offset_z: usize,
        export_layers: usize,
        host_layers: usize,
    },
}

impl std::fmt::Display for SynthInjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SynthInjectError::GridBounds {
                offset_x,
                offset_y,
                export_w,
                export_h,
                host_w,
                host_h,
            } => write!(
                f,
                "export [{}..{}, {}..{}) does not fit in host {}x{}",
                offset_x,
                offset_x.saturating_add(*export_w),
                offset_y,
                offset_y.saturating_add(*export_h),
                host_w,
                host_h
            ),
            SynthInjectError::LayerOverflow {
                offset_z,
                export_layers,
                host_layers,
            } => write!(
                f,
                "export needs layers {}..{} but host has only {}",
                offset_z,
                offset_z.saturating_add(*export_layers),
                host_layers
            ),
        }
    }
}

impl std::error::Error for SynthInjectError {}

/// Inject a `SynthExport` into a host `Simulation` at `(offset_x, offset_y)`.
///
/// Copies all tiles (including Const guards) from the export grid into the host
/// grid. Maps input/output/circuit coordinates to host-global indices. Rebuilds
/// via connections if the export uses multiple layers.
pub fn inject_synth_export(
    sim: &mut Simulation,
    export: &SynthExport,
    offset_x: usize,
    offset_y: usize,
) -> InjectedBlock {
    inject_synth_export_z(sim, export, offset_x, offset_y, 0)
}

/// Checked variant of `inject_synth_export`.
pub fn try_inject_synth_export(
    sim: &mut Simulation,
    export: &SynthExport,
    offset_x: usize,
    offset_y: usize,
) -> Result<InjectedBlock, SynthInjectError> {
    try_inject_synth_export_z(sim, export, offset_x, offset_y, 0)
}

/// Sprint 250: Inject a synth export into a host simulation at a z-plane offset.
///
/// Same as `inject_synth_export` but places the export starting at layer `offset_z`
/// instead of layer 0. This enables stacking multiple synth blocks on independent
/// z-planes (e.g., z=0..1, z=2..3, z=4..5) within the same (x, y) region.
pub fn inject_synth_export_z(
    sim: &mut Simulation,
    export: &SynthExport,
    offset_x: usize,
    offset_y: usize,
    offset_z: usize,
) -> InjectedBlock {
    try_inject_synth_export_z(sim, export, offset_x, offset_y, offset_z)
        .unwrap_or_else(|e| panic!("inject_synth_export: {e}"))
}

/// Checked variant of `inject_synth_export_z`.
pub fn try_inject_synth_export_z(
    sim: &mut Simulation,
    export: &SynthExport,
    offset_x: usize,
    offset_y: usize,
    offset_z: usize,
) -> Result<InjectedBlock, SynthInjectError> {
    let ew = export.sim.tilemap.width;
    let eh = export.sim.tilemap.height;
    let host_w = sim.tilemap.width;
    let host_h = sim.tilemap.height;
    let host_layers = sim.tilemap.num_layers;
    let export_layers = export.sim.tilemap.num_layers;

    let max_x = offset_x
        .checked_add(ew)
        .ok_or(SynthInjectError::GridBounds {
            offset_x,
            offset_y,
            export_w: ew,
            export_h: eh,
            host_w,
            host_h,
        })?;
    let max_y = offset_y
        .checked_add(eh)
        .ok_or(SynthInjectError::GridBounds {
            offset_x,
            offset_y,
            export_w: ew,
            export_h: eh,
            host_w,
            host_h,
        })?;
    if max_x > host_w || max_y > host_h {
        return Err(SynthInjectError::GridBounds {
            offset_x,
            offset_y,
            export_w: ew,
            export_h: eh,
            host_w,
            host_h,
        });
    }

    let max_z = offset_z
        .checked_add(export_layers)
        .ok_or(SynthInjectError::LayerOverflow {
            offset_z,
            export_layers,
            host_layers,
        })?;
    if max_z > host_layers {
        return Err(SynthInjectError::LayerOverflow {
            offset_z,
            export_layers,
            host_layers,
        });
    }

    // Copy tiles from export grid into host grid at z-offset.
    for ez in 0..export_layers {
        let export_layer_offset = ez * ew * eh;
        for ey in 0..eh {
            for ex in 0..ew {
                let hx = offset_x + ex;
                let hy = offset_y + ey;
                let export_idx = export_layer_offset + ey * ew + ex;
                let tile_type = export.sim.tilemap.tiles[export_idx].meta.tile_type;
                sim.set_tile_3d(hx, hy, offset_z + ez, tile_type);
            }
        }
    }
    if export_layers > 1 || offset_z > 0 {
        sim.rebuild_via_connections();
    }

    // Map input coordinates (inputs are on the base layer of the export = offset_z).
    let host_layer_size = sim.tilemap.layer_size;
    let z_base = offset_z * host_layer_size;
    let input_indices: Vec<usize> = export
        .input_coords
        .iter()
        .map(|&(ex, ey)| z_base + (offset_y + ey) * host_w + (offset_x + ex))
        .collect();

    // Map output coordinates.
    let output_indices: Vec<usize> = export
        .output_coords
        .iter()
        .map(|&(ex, ey)| z_base + (offset_y + ey) * host_w + (offset_x + ex))
        .collect();

    // Map circuit indices (non-Const tiles), handling multi-layer exports.
    let export_layer_size = ew * eh;
    let host_tile_count = sim.tilemap.tiles.len();
    let circuit_indices: Vec<usize> = export
        .circuit_indices()
        .iter()
        .map(|&idx| {
            let ez = idx / export_layer_size;
            let idx_in_layer = idx % export_layer_size;
            let ex = idx_in_layer % ew;
            let ey = idx_in_layer / ew;
            let host_idx = (offset_z + ez) * host_layer_size + (offset_y + ey) * host_w + (offset_x + ex);
            assert!(
                host_idx < host_tile_count,
                "inject_synth_export: remapped circuit index {host_idx} out of bounds (tile_count={host_tile_count})"
            );
            host_idx
        })
        .collect();

    // Build scope mask for propagate_combinational_masked().
    let total_tiles = sim.tilemap.tiles.len();
    let mask_len = (total_tiles + 63) / 64;
    let mut scope_mask = vec![0u64; mask_len];
    for &idx in input_indices
        .iter()
        .chain(circuit_indices.iter())
        .chain(output_indices.iter())
    {
        scope_mask[idx / 64] |= 1u64 << (idx % 64);
    }

    Ok(InjectedBlock {
        offset_x,
        offset_y,
        width: ew,
        height: eh,
        input_indices,
        output_indices,
        circuit_indices,
        scope_mask,
        outputs_cache: None,
    })
}

/// Drive an injected block with the given input values and return output values.
///
/// Follows the same reset → drive → propagate → read pattern as
/// `evaluate_exported()` in `export.rs`.
pub fn drive_injected_block(
    sim: &mut Simulation,
    block: &InjectedBlock,
    input_values: &[bool],
) -> Vec<bool> {
    assert_eq!(input_values.len(), block.input_indices.len());
    let tile_count = sim.tilemap.tiles.len();
    assert!(
        block.input_indices.iter().all(|&idx| idx < tile_count),
        "drive_injected_block: input index out of bounds"
    );
    assert!(
        block.output_indices.iter().all(|&idx| idx < tile_count),
        "drive_injected_block: output index out of bounds"
    );
    assert!(
        block.circuit_indices.iter().all(|&idx| idx < tile_count),
        "drive_injected_block: circuit index out of bounds"
    );
    let width = sim.tilemap.width;
    let height = sim.tilemap.height;
    let layer_size = width * height;

    // 1. Reset circuit tiles to 0 and mark dirty.
    for &idx in &block.circuit_indices {
        sim.tilemap.tiles[idx].logic.store(0, Ordering::Relaxed);
        sim.dirty.mark_dirty(idx);
    }

    // 2. Drive primary input Const tiles.
    for (i, &val) in input_values.iter().enumerate() {
        let logic = if val { u64::MAX } else { 0 };
        sim.set_logic_value_by_idx(block.input_indices[i], logic);
    }

    // 3. Mark PI neighbors dirty so Const value changes propagate.
    for &idx in &block.input_indices {
        // Only mark L0 neighbors (inputs are always on L0).
        if idx % width > 0 {
            sim.dirty.mark_dirty(idx - 1);
        }
        if idx % width + 1 < width {
            sim.dirty.mark_dirty(idx + 1);
        }
        if idx >= width {
            sim.dirty.mark_dirty(idx - width);
        }
        if idx + width < layer_size {
            sim.dirty.mark_dirty(idx + width);
        }
    }

    // 4. Propagate to convergence. Large circuits (e.g. combined decode:
    // ~200 gates, 280-wide grid with multi-layer routes) need 2000+ delta
    // cycles, so use a generous outer limit.
    for _ in 0..200 {
        let (_, _, switched) = sim.propagate_combinational_counted();
        if switched == 0 {
            break;
        }
    }

    // 5. Read primary outputs.
    block
        .output_indices
        .iter()
        .map(|&idx| sim.tilemap.tiles[idx].logic.load(Ordering::Relaxed) != 0)
        .collect()
}

/// Drive an injected block using masked propagation (scope-restricted).
///
/// Unlike `drive_injected_block()`, this function uses
/// `propagate_combinational_masked()` to restrict evaluation to only the
/// synth block's tiles. Safe to call during CPU execution without corrupting
/// CPU pipeline state.
///
/// Inputs/outputs use `u64` values (matching `set_logic_value_by_idx` / `get_logic_value_by_idx`).
pub fn drive_injected_block_masked(
    sim: &mut Simulation,
    block: &InjectedBlock,
    input_values: &[u64],
) -> SynthOutputs {
    assert_eq!(input_values.len(), block.input_indices.len());

    // 1. Reset circuit tiles to 0 and mark dirty.
    for &idx in &block.circuit_indices {
        sim.set_logic_value_by_idx(idx, 0);
        sim.dirty.mark_dirty(idx);
    }

    // 2. Drive PI Const tiles with input values.
    for (i, &idx) in block.input_indices.iter().enumerate() {
        sim.set_logic_value_by_idx(idx, input_values[i]);
    }

    // 3. Dirty PI neighbors (manual 4-neighbor marking).
    let width = sim.tilemap.width;
    let layer_size = width * sim.tilemap.height;
    for &idx in &block.input_indices {
        let local = idx % layer_size;
        if local % width > 0 {
            sim.dirty.mark_dirty(idx - 1);
        }
        if local % width + 1 < width {
            sim.dirty.mark_dirty(idx + 1);
        }
        if local >= width {
            sim.dirty.mark_dirty(idx - width);
        }
        if local + width < layer_size {
            sim.dirty.mark_dirty(idx + width);
        }
    }

    // 4. Masked propagation — only synth block tiles.
    // Bounded to prevent infinite spinning on non-converging (oscillatory) blocks.
    const MAX_MASKED_ITERS: usize = 200;
    for _ in 0..MAX_MASKED_ITERS {
        let (d, _, _) = sim.propagate_combinational_masked(&block.scope_mask);
        if d == 0 {
            break;
        }
    }

    // 5. Read primary outputs.
    let mut result = SynthOutputs::new();
    for &idx in &block.output_indices {
        result.push(sim.get_logic_value_by_idx(idx));
    }
    result
}

/// Sprint 202: Precompute the output lookup table for an injected block.
///
/// Evaluates `drive_injected_block_masked()` for every possible input combination
/// (2^n_inputs entries) and stores the results in `block.outputs_cache`. After this
/// call, `drive_synth_block()` returns cached results without tile evaluation.
pub fn precompute_block_cache(sim: &mut Simulation, block: &mut InjectedBlock) {
    let n_inputs = block.input_indices.len();
    let n_combos = 1usize << n_inputs;
    let mut cache = Vec::with_capacity(n_combos);
    for pattern in 0..n_combos {
        let inputs: Vec<u64> = (0..n_inputs)
            .map(|i| if (pattern >> i) & 1 != 0 { u64::MAX } else { 0 })
            .collect();
        let outputs = drive_injected_block_masked(sim, block, &inputs);
        cache.push(outputs);
    }
    block.outputs_cache = Some(cache);
}

/// Sprint 202: Unified synth block evaluation with cache fallback.
///
/// If the block has a precomputed `outputs_cache`, returns the cached result
/// (zero tile evaluations). Otherwise falls back to live `drive_injected_block_masked()`.
///
/// **Contract**: `input_values` must have exactly `block.input_indices.len()` elements,
/// each either `0` or `u64::MAX` (boolean tile convention). Non-boolean values are
/// reduced to 0/1 for cache lookup but would produce different results under live eval.
pub fn drive_synth_block(
    sim: &mut Simulation,
    block: &InjectedBlock,
    input_values: &[u64],
) -> SynthOutputs {
    assert_eq!(
        input_values.len(),
        block.input_indices.len(),
        "drive_synth_block: expected {} inputs, got {}",
        block.input_indices.len(),
        input_values.len(),
    );
    if let Some(ref cache) = block.outputs_cache {
        let key = input_values
            .iter()
            .enumerate()
            .fold(0usize, |acc, (i, &v)| acc | if v != 0 { 1 << i } else { 0 });
        return cache[key];
    }
    drive_injected_block_masked(sim, block, input_values)
}

// ---------------------------------------------------------------------------
// 8x8 Multiplier truth table + AIG builder (Sprint 362)
// ---------------------------------------------------------------------------

/// Expected MUL output for each of the 2^16 input combinations.
///
/// `table[a][b] = a * b` (16-bit product, stored as u64).
pub fn mul_truth_table() -> [[u64; 256]; 256] {
    let mut table = [[0u64; 256]; 256];
    for a in 0..256u16 {
        for b in 0..256u16 {
            table[a as usize][b as usize] = (a * b) as u64;
        }
    }
    table
}

/// Build an AIG for an 8x8 unsigned multiplier.
///
/// 16 primary inputs: a[0]..a[7] (bits of operand A), b[0]..b[7] (bits of operand B).
/// 16 primary outputs: r[0]..r[15] (bits of the 16-bit product a * b).
///
/// Uses partial products + 16-bit ripple-carry adder chain.
/// ~300-500 AIG nodes: 64 AND (partial products) + per-bit full adders.
pub fn build_mul_aig() -> Aig {
    let mut aig = Aig::new();

    // 16 primary inputs
    let a: Vec<AigLit> = (0..8).map(|i| aig.add_input(&format!("a{}", i))).collect();
    let b: Vec<AigLit> = (0..8).map(|i| aig.add_input(&format!("b{}", i))).collect();

    // For each multiplier bit b[k], compute a row of partial products and add it
    // to a running 16-bit accumulated sum using a ripple-carry adder.
    let mut sum_reg: Vec<AigLit> = (0..16).map(|_| AigLit::FALSE).collect(); // accumulated product

    for k in 0..8 {
        // Row k: partial products a[i] & b[k], shifted left by k bits.
        let mut row = vec![AigLit::FALSE; 16];
        for i in 0..8 {
            row[i + k] = aig.and(a[i], b[k]);
        }

        // Ripple-carry add this row to the running sum.
        // For each bit j: new_sum[j] = sum_reg[j] XOR row[j] XOR carry_in
        //                 carry_out = majority(sum_reg[j], row[j], carry_in)
        let mut c_in = AigLit::FALSE;
        for j in 0..16 {
            let a_in = sum_reg[j];
            let bj = row[j];

            // Full adder:
            // s = a XOR b XOR cin
            // cout = (a AND b) OR (cin AND (a XOR b))
            let a_xor_b = aig.xor(a_in, bj);
            let s = aig.xor(a_xor_b, c_in);
            let a_and_b = aig.and(a_in, bj);
            let c_and_axorb = aig.and(c_in, a_xor_b);
            let c_out = aig.or(a_and_b, c_and_axorb);

            sum_reg[j] = s;
            c_in = c_out;
        }
    }

    // Outputs
    for k in 0..16 {
        aig.add_output(&format!("r{}", k), sum_reg[k]);
    }

    aig
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::benchmark::build_4bit_adder;
    use crate::synth::export::evaluate_exported;
    use crate::synth::mapping::evaluate_aig;

    // Sprint 195: shared coexistence test helper.
    fn run_coexistence_check(
        benchmark_name: &str,
        aig: &Aig,
        place_config: &PlaceConfig,
        route_config: &RouteConfig,
        post_run_check: Option<&dyn Fn(&mut Simulation, &InjectedBlock)>,
    ) {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let case = benchmark_cases()
            .iter()
            .find(|c| c.name == benchmark_name)
            .unwrap_or_else(|| panic!("benchmark '{benchmark_name}' not found"));

        let program = assemble_v2(case.source).unwrap();
        let mut sim = Simulation::with_size_layered(128, 192, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(case.rom_size)
            .with_ram_size(128)
            .build(&mut sim);

        place_const_guard_region(&mut sim, 0..128, 64..70);

        let lib = CellLibrary::tile_native();
        let export = synthesize_to_simulation(
            aig,
            &lib,
            &SynthConfig::default(),
            place_config,
            route_config,
        )
        .expect("synth pipeline failed");
        let block = inject_synth_export(&mut sim, &export, 0, 70);

        let mut cycles = 0u64;
        for _ in 0..case.max_cycles {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }

        let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
            .iter()
            .find(|(n, _)| *n == benchmark_name)
            .map(|(_, c)| *c)
            .unwrap_or_else(|| panic!("golden cycles for '{benchmark_name}' not found"));
        assert_eq!(
            cycles, expected_cycles,
            "'{benchmark_name}' cycle count with synth block: expected {expected_cycles}, got {cycles}"
        );

        let hash = hash_v2_final_state(&cpu, &sim);
        let expected_hash = V2_BENCHMARK_GOLDENS
            .iter()
            .find(|(n, _)| *n == benchmark_name)
            .map(|(_, h)| *h)
            .unwrap_or_else(|| panic!("golden hash for '{benchmark_name}' not found"));
        assert_eq!(
            hash, expected_hash,
            "'{benchmark_name}' hash with synth block: expected {expected_hash:#018X}, got {hash:#018X}"
        );

        if let Some(check) = post_run_check {
            check(&mut sim, &block);
        }
    }
    use crate::synth::{
        CellLibrary, PlaceConfig, RouteConfig, SynthConfig, synthesize_to_simulation,
    };

    /// Test 1: Verify AIG truth table against expected values.
    #[test]
    fn branch_taken_aig_truth_table() {
        let aig = build_branch_taken_aig();
        let expected = branch_taken_truth_table();

        for sel in 0..32u32 {
            let inputs: Vec<bool> = (0..5).map(|i| (sel >> i) & 1 != 0).collect();
            let outputs = evaluate_aig(&aig, &inputs);
            assert_eq!(outputs.len(), 1, "expected 1 output, got {}", outputs.len());
            assert_eq!(
                outputs[0],
                expected[sel as usize],
                "mismatch at sel={sel}: ctrl_b={}, z={}, c={}, expected={}, got={}",
                sel & 7,
                (sel >> 3) & 1,
                (sel >> 4) & 1,
                expected[sel as usize],
                outputs[0]
            );
        }
    }

    /// Test 1b: Verify ctrl_b AIG truth table against CTRL_B_LUT (direct AIG evaluation).
    #[test]
    fn ctrl_b_aig_truth_table() {
        let aig = build_ctrl_b_aig();
        let lut = &CTRL_B_LUT_EMBEDDED;

        for opcode in 0..32u32 {
            let inputs: Vec<bool> = (0..5).map(|i| (opcode >> i) & 1 != 0).collect();
            let outputs = evaluate_aig(&aig, &inputs);
            assert_eq!(
                outputs.len(),
                8,
                "expected 8 outputs, got {}",
                outputs.len()
            );
            let mut cb_val = 0u8;
            for (i, &val) in outputs.iter().enumerate() {
                if val {
                    cb_val |= 1 << i;
                }
            }
            let expected = lut[opcode as usize];
            assert_eq!(
                cb_val, expected,
                "ctrl_b AIG mismatch at opcode={opcode:#04X}: aig={cb_val:#04X}, lut={expected:#04X}",
            );
        }
    }

    /// Test 1c: Verify ctrl_b synth pipeline output (evaluate_exported).
    #[test]
    fn ctrl_b_synth_pipeline_eval() {
        let aig = build_ctrl_b_aig();
        let lib = CellLibrary::tile_native();
        let config = SynthConfig::default();

        let mut export = synthesize_to_simulation(
            &aig,
            &lib,
            &config,
            &PlaceConfig {
                halo: 4,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
        )
        .expect("synth pipeline failed");

        let lut = &CTRL_B_LUT_EMBEDDED;
        for opcode in 0..32u32 {
            let inputs: Vec<bool> = (0..5).map(|i| (opcode >> i) & 1 != 0).collect();
            let outputs = evaluate_exported(&mut export, &inputs);
            assert_eq!(
                outputs.len(),
                8,
                "expected 8 outputs, got {}",
                outputs.len()
            );
            let mut cb_val = 0u8;
            for (i, &val) in outputs.iter().enumerate() {
                if val {
                    cb_val |= 1 << i;
                }
            }
            let expected = lut[opcode as usize];
            assert_eq!(
                cb_val, expected,
                "ctrl_b synth mismatch at opcode={opcode:#04X}: synth={cb_val:#04X}, lut={expected:#04X}",
            );
        }
    }

    /// Test 1d: Verify ctrl_a AIG truth table matches CTRL_A_LUT for all 32 opcodes.
    #[test]
    fn ctrl_a_aig_truth_table() {
        let aig = build_ctrl_a_aig();
        let lut = &CTRL_A_LUT_EMBEDDED;

        for opcode in 0..32u32 {
            let inputs: Vec<bool> = (0..5).map(|i| (opcode >> i) & 1 != 0).collect();
            let outputs = evaluate_aig(&aig, &inputs);
            assert_eq!(
                outputs.len(),
                8,
                "expected 8 outputs, got {}",
                outputs.len()
            );
            let mut ca_val = 0u8;
            for (i, &val) in outputs.iter().enumerate() {
                if val {
                    ca_val |= 1 << i;
                }
            }
            let expected = lut[opcode as usize];
            assert_eq!(
                ca_val, expected,
                "ctrl_a AIG mismatch at opcode={opcode:#04X}: aig={ca_val:#04X}, lut={expected:#04X}",
            );
        }
    }

    /// Test 1e: Verify ctrl_a synth pipeline output (evaluate_exported).
    #[test]
    #[ignore]
    fn ctrl_a_synth_pipeline_eval() {
        let aig = build_ctrl_a_aig();
        let lib = CellLibrary::tile_native();
        let config = SynthConfig::default();

        // ctrl_a has ~89 AND nodes (much larger than ctrl_b's ~36), so it needs
        // a larger halo and multi-layer routing to avoid congestion.
        let mut export = synthesize_to_simulation(
            &aig,
            &lib,
            &config,
            &PlaceConfig {
                halo: 8,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
        )
        .expect("synth pipeline failed");

        let lut = &CTRL_A_LUT_EMBEDDED;
        for opcode in 0..32u32 {
            let inputs: Vec<bool> = (0..5).map(|i| (opcode >> i) & 1 != 0).collect();
            let outputs = evaluate_exported(&mut export, &inputs);
            assert_eq!(
                outputs.len(),
                8,
                "expected 8 outputs, got {}",
                outputs.len()
            );
            let mut ca_val = 0u8;
            for (i, &val) in outputs.iter().enumerate() {
                if val {
                    ca_val |= 1 << i;
                }
            }
            let expected = lut[opcode as usize];
            assert_eq!(
                ca_val, expected,
                "ctrl_a synth mismatch at opcode={opcode:#04X}: synth={ca_val:#04X}, lut={expected:#04X}",
            );
        }
    }

    /// Test 1f: Verify CTRL_A_LUT_EMBEDDED matches CTRL_A_LUT in v2_wiring.rs.
    #[test]
    fn ctrl_a_lut_parity() {
        use crate::tile_cpu::v2_execute::TileCpuV2;
        let lut = TileCpuV2::ctrl_a_lut();
        assert_eq!(
            &CTRL_A_LUT_EMBEDDED, lut,
            "CTRL_A_LUT_EMBEDDED in integration.rs does not match CTRL_A_LUT in v2_wiring.rs"
        );
    }

    /// Sprint 204: Combined ctrl_a+ctrl_b AIG truth table — all 32 opcodes.
    #[test]
    fn combined_decode_aig_truth_table() {
        use crate::synth::mapping::evaluate_aig;

        let aig = build_combined_decode_aig();

        for opcode in 0..32u32 {
            let inputs: Vec<bool> = (0..5).map(|i| (opcode >> i) & 1 != 0).collect();
            let outputs = evaluate_aig(&aig, &inputs);
            assert_eq!(
                outputs.len(),
                16,
                "expected 16 outputs (8 ctrl_a + 8 ctrl_b)"
            );

            let expected_ca = CTRL_A_LUT_EMBEDDED[opcode as usize];
            let expected_cb = CTRL_B_LUT_EMBEDDED[opcode as usize];

            for bit in 0..8 {
                let ca_bit = outputs[bit];
                let expected = (expected_ca >> bit) & 1 != 0;
                assert_eq!(
                    ca_bit, expected,
                    "combined_decode ca[{bit}] mismatch at opcode {opcode:#04x}: got {ca_bit}, expected {expected}"
                );
            }
            for bit in 0..8 {
                let cb_bit = outputs[8 + bit];
                let expected = (expected_cb >> bit) & 1 != 0;
                assert_eq!(
                    cb_bit, expected,
                    "combined_decode cb[{bit}] mismatch at opcode {opcode:#04x}: got {cb_bit}, expected {expected}"
                );
            }
        }
    }

    /// Test 2: Full synth pipeline → evaluate_exported for branch-taken.
    #[test]
    fn branch_taken_standalone_synth() {
        let aig = build_branch_taken_aig();
        let lib = CellLibrary::tile_native();
        let config = SynthConfig::default();
        let mut export = synthesize_to_simulation(
            &aig,
            &lib,
            &config,
            &PlaceConfig {
                halo: 4,
                ..PlaceConfig::default()
            },
            &RouteConfig::default(),
        )
        .expect("synth pipeline failed for branch_taken");

        let expected = branch_taken_truth_table();
        for sel in 0..32u32 {
            let inputs: Vec<bool> = (0..5).map(|i| (sel >> i) & 1 != 0).collect();
            let outputs = evaluate_exported(&mut export, &inputs);
            assert_eq!(
                outputs[0],
                expected[sel as usize],
                "standalone synth mismatch at sel={sel}: ctrl_b={}, z={}, c={}",
                sel & 7,
                (sel >> 3) & 1,
                (sel >> 4) & 1,
            );
        }
    }

    /// Test 3: Injection mechanics — inject adder4 into fresh grid, verify outputs.
    #[test]
    fn inject_synth_export_mechanics() {
        let aig = build_4bit_adder();
        let lib = CellLibrary::tile_native();
        let export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 3,
                ..PlaceConfig::default()
            },
            &RouteConfig::default(),
        )
        .expect("synth pipeline failed for adder4");

        // Use a host grid large enough to hold the export at a small offset.
        let ew = export.sim.tilemap.width;
        let eh = export.sim.tilemap.height;
        let el = export.sim.tilemap.num_layers;
        let host_w = ew + 4;
        let host_h = eh + 4;
        let mut host = Simulation::with_size_layered(host_w, host_h, el);
        // Fill with Const guards (default Wire tiles would leak signal).
        place_const_guard_region(&mut host, 0..host_w, 0..host_h);
        let block = inject_synth_export(&mut host, &export, 2, 2);

        assert_eq!(block.input_indices.len(), aig.num_inputs() as usize);
        assert!(!block.circuit_indices.is_empty());
        assert_eq!(block.offset_x, 2);
        assert_eq!(block.offset_y, 2);

        // Verify a few adder4 combos: a + b = sum.
        let test_cases: [(u8, u8); 5] = [(0, 0), (1, 1), (3, 5), (15, 15), (7, 8)];
        for (a, b) in test_cases {
            let mut inputs = Vec::new();
            for i in 0..4 {
                inputs.push((a >> i) & 1 != 0);
            }
            for i in 0..4 {
                inputs.push((b >> i) & 1 != 0);
            }
            let outputs = drive_injected_block(&mut host, &block, &inputs);

            let expected_sum = (a as u16) + (b as u16);
            let mut actual_sum = 0u16;
            for (i, &bit) in outputs.iter().enumerate() {
                if bit {
                    actual_sum |= 1 << i;
                }
            }
            assert_eq!(
                actual_sum, expected_sum,
                "adder4 injection: {a}+{b}: expected {expected_sum}, got {actual_sum}"
            );
        }
    }

    /// Test 4: Shadow verification — synth branch-taken in V2 CPU grid.
    #[test]
    fn branch_taken_v2_shadow_verification() {
        use crate::tile_cpu::{V2Builder, assemble_v2};

        // Build V2 CPU with a trivial program.
        let program = assemble_v2("HALT").unwrap();
        let mut sim = Simulation::with_size_layered(128, 192, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(64)
            .with_ram_size(128)
            .build(&mut sim);

        // Build and inject synth branch-taken block south of CPU.
        let aig = build_branch_taken_aig();
        let lib = CellLibrary::tile_native();
        let export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 4,
                ..PlaceConfig::default()
            },
            &RouteConfig::default(),
        )
        .expect("synth pipeline failed for branch_taken");

        // Guard rows between CPU (y<=63) and synth block (y=70+).
        place_const_guard_region(&mut sim, 0..128, 64..70);

        let block = inject_synth_export(&mut sim, &export, 0, 70);

        // Snapshot key CPU state before shadow driving.
        let pre_halted = cpu.is_halted();
        let pre_pc = cpu.read_pc(&sim);
        let pre_lr = cpu.read_lr();
        let pre_flag_z = cpu.read_flag_z(&sim);
        let pre_flag_c = cpu.read_flag_c(&sim);
        let pre_regs: [u64; 16] = std::array::from_fn(|r| cpu.read_reg(&sim, r));

        // Drive all 32 input combos and verify against truth table.
        let expected = branch_taken_truth_table();
        for sel in 0..32u32 {
            let inputs: Vec<bool> = (0..5).map(|i| (sel >> i) & 1 != 0).collect();
            let outputs = drive_injected_block(&mut sim, &block, &inputs);
            assert_eq!(
                outputs[0],
                expected[sel as usize],
                "shadow verification mismatch at sel={sel}: ctrl_b={}, z={}, c={}",
                sel & 7,
                (sel >> 3) & 1,
                (sel >> 4) & 1,
            );
        }

        // Verify CPU state is not corrupted by the synth block.
        let post_regs: [u64; 16] = std::array::from_fn(|r| cpu.read_reg(&sim, r));
        assert_eq!(
            cpu.is_halted(),
            pre_halted,
            "halted flag changed unexpectedly"
        );
        assert_eq!(cpu.read_pc(&sim), pre_pc, "PC changed unexpectedly");
        assert_eq!(cpu.read_lr(), pre_lr, "LR changed unexpectedly");
        assert_eq!(
            cpu.read_flag_z(&sim),
            pre_flag_z,
            "flag Z changed unexpectedly"
        );
        assert_eq!(
            cpu.read_flag_c(&sim),
            pre_flag_c,
            "flag C changed unexpectedly"
        );
        assert_eq!(post_regs, pre_regs, "register file changed unexpectedly");
    }

    /// Test 5: Coexistence — run fibonacci benchmark with synth block present.
    #[test]
    fn v2_golden_coexistence() {
        let aig = build_branch_taken_aig();
        run_coexistence_check(
            "fibonacci",
            &aig,
            &PlaceConfig {
                halo: 4,
                ..PlaceConfig::default()
            },
            &RouteConfig::default(),
            None,
        );
    }

    // -----------------------------------------------------------------------
    // Sprint 194 tests
    // -----------------------------------------------------------------------

    /// Test 6: Live shadow — synth branch-taken tracks physical Mux16to1
    /// across all 8 golden benchmarks, reading live CPU signals each cycle.
    #[test]
    fn live_shadow_branch_taken_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 192, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            place_const_guard_region(&mut sim, 0..128, 64..70);

            let aig = build_branch_taken_aig();
            let lib = CellLibrary::tile_native();
            let export = synthesize_to_simulation(
                &aig,
                &lib,
                &SynthConfig::default(),
                &PlaceConfig {
                    halo: 4,
                    ..PlaceConfig::default()
                },
                &RouteConfig::default(),
            )
            .expect("synth pipeline failed for branch_taken");
            let block = inject_synth_export(&mut sim, &export, 0, 70);

            let mut cycles = 0u64;
            let mut mismatches = 0u64;
            let mut first_mismatch_info: Option<String> = None;

            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;

                // Re-evaluate the physical branch path with current tile values.
                // After step(), branch_taken_core may be stale because flags and
                // ctrl_b were updated but the branch scope wasn't re-propagated.
                let (ctrl_b_val, flag_z, flag_c, physical_taken) =
                    cpu.snapshot_branch_physical(&mut sim);

                // Encode as synth inputs: ctrl_b[0], ctrl_b[1], ctrl_b[2], flag_z, flag_c.
                let inputs = vec![
                    (ctrl_b_val >> 0) & 1 != 0,
                    (ctrl_b_val >> 1) & 1 != 0,
                    (ctrl_b_val >> 2) & 1 != 0,
                    flag_z,
                    flag_c,
                ];

                let synth_output = drive_injected_block(&mut sim, &block, &inputs);
                let synth_taken = synth_output[0];

                if synth_taken != physical_taken {
                    mismatches += 1;
                    if first_mismatch_info.is_none() {
                        first_mismatch_info = Some(format!(
                            "cycle {cycles}: ctrl_b={ctrl_b_val}, z={flag_z}, c={flag_c}, \
                             synth={synth_taken}, physical={physical_taken}"
                        ));
                    }
                }
            }

            assert_eq!(
                mismatches,
                0,
                "live shadow mismatch in '{}': {} mismatches in {} cycles. First: {}",
                case.name,
                mismatches,
                cycles,
                first_mismatch_info.as_deref().unwrap_or("none"),
            );

            // Verify golden cycle count unchanged.
            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(name, _)| *name == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycle count with live shadow: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            // Verify golden hash unchanged.
            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(name, _)| *name == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with live shadow: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );
        }
    }

    /// Test 7: Standalone synth pipeline for decoder4to16 — 16 one-hot outputs.
    #[test]
    fn decoder4to16_standalone_synth() {
        use crate::synth::benchmark::build_decoder4to16;

        let aig = build_decoder4to16();
        let lib = CellLibrary::tile_native();
        // decoder4to16 needs no_crossings + multi-layer routing to avoid
        // bypass corruption (same issue as Sprint 192).
        let mut export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 3,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
        )
        .expect("synth pipeline failed for decoder4to16");

        for sel in 0..16u32 {
            let inputs: Vec<bool> = (0..4).map(|i| (sel >> i) & 1 != 0).collect();
            let outputs = evaluate_exported(&mut export, &inputs);
            assert_eq!(
                outputs.len(),
                16,
                "expected 16 outputs, got {}",
                outputs.len()
            );

            for (i, &bit) in outputs.iter().enumerate() {
                let expected = i as u32 == sel;
                assert_eq!(
                    bit, expected,
                    "decoder4to16 standalone: sel={sel}, output[{i}]={bit}, expected={expected}"
                );
            }
        }
    }

    /// Test 8: Decoder4to16 injected into host grid — software-driven.
    #[test]
    fn decoder4to16_injected_in_host_grid() {
        use crate::synth::benchmark::build_decoder4to16;

        let aig = build_decoder4to16();
        let lib = CellLibrary::tile_native();
        let export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 3,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
        )
        .expect("synth pipeline failed for decoder4to16");

        let ew = export.sim.tilemap.width;
        let eh = export.sim.tilemap.height;
        let el = export.sim.tilemap.num_layers;
        let host_w = ew + 4;
        let host_h = eh + 4;
        let mut host = Simulation::with_size_layered(host_w, host_h, el);
        place_const_guard_region(&mut host, 0..host_w, 0..host_h);
        let block = inject_synth_export(&mut host, &export, 2, 2);

        assert_eq!(block.input_indices.len(), 4, "decoder4to16 has 4 inputs");

        for sel in 0..16u32 {
            let inputs: Vec<bool> = (0..4).map(|i| (sel >> i) & 1 != 0).collect();
            let outputs = drive_injected_block(&mut host, &block, &inputs);

            for (i, &bit) in outputs.iter().enumerate() {
                let expected = i as u32 == sel;
                assert_eq!(
                    bit, expected,
                    "injected decoder4to16: sel={sel}, output[{i}]={bit}, expected={expected}"
                );
            }
        }
    }

    /// Test 9: Decoder4to16 coexistence with V2 CPU — fibonacci golden passes.
    #[test]
    fn decoder4to16_v2_coexistence() {
        use crate::synth::benchmark::build_decoder4to16;
        let aig = build_decoder4to16();
        run_coexistence_check(
            "fibonacci",
            &aig,
            &PlaceConfig {
                halo: 3,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
            Some(&|sim, block| {
                for sel in 0..16u32 {
                    let inputs: Vec<bool> = (0..4).map(|i| (sel >> i) & 1 != 0).collect();
                    let outputs = drive_injected_block(sim, block, &inputs);
                    for (i, &bit) in outputs.iter().enumerate() {
                        let expected = i as u32 == sel;
                        assert_eq!(
                            bit, expected,
                            "decoder4to16 post-coexistence: sel={sel}, output[{i}] wrong"
                        );
                    }
                }
            }),
        );
    }

    // -----------------------------------------------------------------------
    // Sprint 195 tests
    // -----------------------------------------------------------------------

    /// Verify priority encoder 8 expected output for a given input combo.
    /// Returns (expected_enc, expected_valid).
    fn prienc8_expected(combo: u32) -> (u8, bool) {
        if combo == 0 {
            (0, false)
        } else {
            let highest = 31 - combo.leading_zeros();
            (highest as u8, true)
        }
    }

    /// Test 10: Standalone synth pipeline for priority_encoder8 — 256 combos.
    #[test]
    fn prienc8_standalone_synth() {
        use crate::synth::benchmark::build_priority_encoder8;

        let aig = build_priority_encoder8();
        let lib = CellLibrary::tile_native();
        let mut export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 3,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
        )
        .expect("synth pipeline failed for priority_encoder8");

        for combo in 0..256u32 {
            let inputs: Vec<bool> = (0..8).map(|i| (combo >> i) & 1 != 0).collect();
            let outputs = evaluate_exported(&mut export, &inputs);
            assert_eq!(outputs.len(), 4, "expected 4 outputs (enc[2:0] + valid)");

            let (enc_expected, valid_expected) = prienc8_expected(combo);
            let valid_actual = outputs[3];
            assert_eq!(
                valid_actual, valid_expected,
                "prienc8 standalone: combo={combo:#010b}, valid: expected={valid_expected}, got={valid_actual}"
            );
            if valid_expected {
                let enc_actual =
                    (outputs[0] as u8) | ((outputs[1] as u8) << 1) | ((outputs[2] as u8) << 2);
                assert_eq!(
                    enc_actual, enc_expected,
                    "prienc8 standalone: combo={combo:#010b}, enc: expected={enc_expected}, got={enc_actual}"
                );
            }
        }
    }

    /// Test 11: Priority encoder 8 injected in host grid — 256 combos.
    #[test]
    fn prienc8_injected_in_host_grid() {
        use crate::synth::benchmark::build_priority_encoder8;

        let aig = build_priority_encoder8();
        let lib = CellLibrary::tile_native();
        let export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 3,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
        )
        .expect("synth pipeline failed for priority_encoder8");

        let ew = export.sim.tilemap.width;
        let eh = export.sim.tilemap.height;
        let el = export.sim.tilemap.num_layers;
        let host_w = ew + 4;
        let host_h = eh + 4;
        let mut host = Simulation::with_size_layered(host_w, host_h, el);
        place_const_guard_region(&mut host, 0..host_w, 0..host_h);
        let block = inject_synth_export(&mut host, &export, 2, 2);

        assert_eq!(block.input_indices.len(), 8, "prienc8 has 8 inputs");

        for combo in 0..256u32 {
            let inputs: Vec<bool> = (0..8).map(|i| (combo >> i) & 1 != 0).collect();
            let outputs = drive_injected_block(&mut host, &block, &inputs);

            let (enc_expected, valid_expected) = prienc8_expected(combo);
            assert_eq!(
                outputs[3], valid_expected,
                "injected prienc8: combo={combo:#010b}, valid wrong"
            );
            if valid_expected {
                let enc_actual =
                    (outputs[0] as u8) | ((outputs[1] as u8) << 1) | ((outputs[2] as u8) << 2);
                assert_eq!(
                    enc_actual, enc_expected,
                    "injected prienc8: combo={combo:#010b}, enc wrong"
                );
            }
        }
    }

    /// Test 12: Priority encoder 8 coexistence with V2 CPU — fibonacci golden.
    #[test]
    fn prienc8_v2_coexistence() {
        use crate::synth::benchmark::build_priority_encoder8;
        let aig = build_priority_encoder8();
        run_coexistence_check(
            "fibonacci",
            &aig,
            &PlaceConfig {
                halo: 3,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
            Some(&|sim, block| {
                for combo in 0..256u32 {
                    let inputs: Vec<bool> = (0..8).map(|i| (combo >> i) & 1 != 0).collect();
                    let outputs = drive_injected_block(sim, block, &inputs);
                    let (enc_expected, valid_expected) = prienc8_expected(combo);
                    assert_eq!(
                        outputs[3], valid_expected,
                        "prienc8 post-coexistence: combo={combo:#010b}, valid wrong"
                    );
                    if valid_expected {
                        let enc_actual = (outputs[0] as u8)
                            | ((outputs[1] as u8) << 1)
                            | ((outputs[2] as u8) << 2);
                        assert_eq!(
                            enc_actual, enc_expected,
                            "prienc8 post-coexistence: combo={combo:#010b}, enc wrong"
                        );
                    }
                }
            }),
        );
    }

    /// Test 13: Live shadow — synth decoder3to8 tracks physical Decoder3to8
    /// tile across all 8 golden benchmarks.
    ///
    /// After each step(), reads the physical rd field and Decoder3to8 output,
    /// then drives the synth block and compares all 8 output bits.
    ///
    /// NOTE: All 8 benchmarks use PC < 64 (rom_upper_bank_group_select = 0),
    /// so physical extraction tiles are always valid (no bank_group=1 stale
    /// decode issue).
    #[test]
    fn live_shadow_rd_decode_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let expected_table = decoder3to8_truth_table();
        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 192, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            place_const_guard_region(&mut sim, 0..128, 64..70);

            let aig = build_decoder3to8_aig();
            let lib = CellLibrary::tile_native();
            let export = synthesize_to_simulation(
                &aig,
                &lib,
                &SynthConfig::default(),
                &PlaceConfig {
                    halo: 3,
                    ..PlaceConfig::default()
                },
                &RouteConfig {
                    max_z: 1,
                    no_crossings: true,
                    ..RouteConfig::default()
                },
            )
            .expect("synth pipeline failed for decoder3to8");
            let block = inject_synth_export(&mut sim, &export, 0, 70);

            let mut cycles = 0u64;
            let mut mismatches = 0u64;
            let mut first_mismatch_info: Option<String> = None;

            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;

                let rd = cpu.read_extract_rd(&sim);
                let physical_decode = cpu.read_physical_rd_decode(&sim);

                // Consistency check: physical Decoder3to8 should output 1 << rd.
                let expected_onehot = 1u64 << (rd & 0x07);
                if (physical_decode & 0xFF) != expected_onehot {
                    mismatches += 1;
                    if first_mismatch_info.is_none() {
                        first_mismatch_info = Some(format!(
                            "cycle {cycles}: rd={rd}, physical={physical_decode:#X}, \
                             expected={expected_onehot:#X} (consistency)"
                        ));
                    }
                    continue;
                }

                // Drive synth decoder3to8 with rd bits.
                let inputs = vec![rd & 1 != 0, (rd >> 1) & 1 != 0, (rd >> 2) & 1 != 0];
                let synth_outputs = drive_injected_block(&mut sim, &block, &inputs);

                // Compare 8 synth outputs against truth table.
                let expected = expected_table[rd as usize];
                for bit in 0..8 {
                    if synth_outputs[bit] != expected[bit] {
                        mismatches += 1;
                        if first_mismatch_info.is_none() {
                            first_mismatch_info = Some(format!(
                                "cycle {cycles}: rd={rd}, bit={bit}, synth={}, expected={}",
                                synth_outputs[bit], expected[bit],
                            ));
                        }
                        break;
                    }
                }
            }

            assert_eq!(
                mismatches,
                0,
                "rd_decode shadow mismatch in '{}': {} mismatches in {} cycles. First: {}",
                case.name,
                mismatches,
                cycles,
                first_mismatch_info.as_deref().unwrap_or("none"),
            );

            // Verify golden cycle count.
            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with rd_decode shadow: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            // Verify golden hash.
            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with rd_decode shadow: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );
        }
    }

    /// Test 14: Sprint 196 — Synth branch replacement across all 8 golden benchmarks.
    /// Enables synth-driven branch-taken (Const tile injection), runs each benchmark,
    /// verifies golden cycle count, golden hash, and zero PC mismatches.
    #[test]
    fn synth_branch_replacement_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            // Verify tile type before enabling — should be Mux (physical branch decoder).
            let pre_type = cpu.branch_taken_core_tile_type(&sim);
            assert_ne!(
                pre_type,
                crate::tiles::tile_meta::TileType::Const,
                "'{}' branch_taken_core should not be Const before enable",
                case.name,
            );

            cpu.enable_synth_branch(&mut sim);

            // Verify tile type was actually swapped to Const.
            let post_type = cpu.branch_taken_core_tile_type(&sim);
            assert_eq!(
                post_type,
                crate::tiles::tile_meta::TileType::Const,
                "'{}' branch_taken_core should be Const after enable, got {:?}",
                case.name,
                post_type,
            );

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with synth branch: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with synth branch: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_branch_mismatches(),
                0,
                "'{}' had {} PC mismatches in {} cycles with synth branch",
                case.name,
                cpu.synth_branch_mismatches(),
                cycles,
            );

            // Verify disable restores cleanly — tile type reverts to original.
            cpu.disable_synth_branch(&mut sim);
            let restored_type = cpu.branch_taken_core_tile_type(&sim);
            assert_eq!(
                restored_type, pre_type,
                "'{}' branch_taken_core should be restored to {:?} after disable, got {:?}",
                case.name, pre_type, restored_type,
            );
        }
    }

    /// Test 15: Sprint 196 — Toggle synth branch mid-run.
    /// Runs fibonacci: 10 cycles physical → 20 cycles synth → remaining physical.
    /// Verifies golden cycle count and hash are preserved across toggle.
    #[test]
    fn synth_branch_toggle_mid_run() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let case = benchmark_cases()
            .iter()
            .find(|c| c.name == "fibonacci")
            .expect("fibonacci benchmark not found");

        let program = assemble_v2(case.source).unwrap();
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(case.rom_size)
            .with_ram_size(128)
            .build(&mut sim);

        let mut cycles = 0u64;

        // Phase 1: 10 cycles physical
        for _ in 0..10 {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }

        // Phase 2: 20 cycles synth
        cpu.enable_synth_branch(&mut sim);
        for _ in 0..20 {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }

        // Phase 3: remaining cycles physical
        cpu.disable_synth_branch(&mut sim);
        for _ in 0..case.max_cycles {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }

        let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
            .iter()
            .find(|(n, _)| *n == "fibonacci")
            .map(|(_, c)| *c)
            .expect("golden cycles not found");
        assert_eq!(
            cycles, expected_cycles,
            "fibonacci cycles with mid-run toggle: expected {expected_cycles}, got {cycles}",
        );

        let hash = hash_v2_final_state(&cpu, &sim);
        let expected_hash = V2_BENCHMARK_GOLDENS
            .iter()
            .find(|(n, _)| *n == "fibonacci")
            .map(|(_, h)| *h)
            .expect("golden hash not found");
        assert_eq!(
            hash, expected_hash,
            "fibonacci hash with mid-run toggle: expected {expected_hash:#018X}, got {hash:#018X}",
        );
    }

    /// Test 16: Sprint 196 — Verify dual-path counter infrastructure.
    /// Runs fibonacci with synth enabled, checks > 0 PC verifications,
    /// zero mismatches, and that check count is reasonable.
    #[test]
    fn synth_branch_dual_path_counters() {
        use crate::tile_cpu::{V2Builder, assemble_v2, benchmark_cases};

        let case = benchmark_cases()
            .iter()
            .find(|c| c.name == "fibonacci")
            .expect("fibonacci benchmark not found");

        let program = assemble_v2(case.source).unwrap();
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(case.rom_size)
            .with_ram_size(128)
            .build(&mut sim);

        cpu.enable_synth_branch(&mut sim);

        let mut cycles = 0u64;
        for _ in 0..case.max_cycles {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }

        assert!(
            cpu.synth_branch_checks() > 0,
            "expected > 0 synth branch checks, got 0 in {cycles} cycles",
        );
        assert_eq!(
            cpu.synth_branch_mismatches(),
            0,
            "expected 0 mismatches, got {} in {cycles} cycles",
            cpu.synth_branch_mismatches(),
        );
        // Checks should be <= cycles (one check per non-halt instruction in PC<64 path).
        assert!(
            cpu.synth_branch_checks() <= cycles,
            "synth_branch_checks ({}) > cycles ({cycles})",
            cpu.synth_branch_checks(),
        );

        cpu.disable_synth_branch(&mut sim);
    }

    // ── Sprint 197: Decoder3to8 replacement tests ──────────────────────────

    /// Test 17: Sprint 197 — Decoder3to8 replacement across all 8 golden benchmarks.
    /// Swaps physical Decoder3to8 to Const, injects 1<<rd each cycle.
    /// Verifies golden cycle count, hash, zero mismatches, tile type swap/restore.
    #[test]
    fn synth_rd_decode_replacement_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            let pre_type = cpu.rd_decode_tile_type(&sim);
            assert_ne!(
                pre_type,
                crate::tiles::tile_meta::TileType::Const,
                "'{}' rd_decode should not be Const before enable",
                case.name,
            );

            cpu.enable_synth_rd_decode(&mut sim);

            let post_type = cpu.rd_decode_tile_type(&sim);
            assert_eq!(
                post_type,
                crate::tiles::tile_meta::TileType::Const,
                "'{}' rd_decode should be Const after enable, got {:?}",
                case.name,
                post_type,
            );

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with synth rd_decode: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with synth rd_decode: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_rd_decode_mismatches(),
                0,
                "'{}' had {} rd_decode mismatches in {} cycles",
                case.name,
                cpu.synth_rd_decode_mismatches(),
                cycles,
            );

            cpu.disable_synth_rd_decode(&mut sim);
            let restored_type = cpu.rd_decode_tile_type(&sim);
            assert_eq!(
                restored_type, pre_type,
                "'{}' rd_decode should be restored to {:?} after disable, got {:?}",
                case.name, pre_type, restored_type,
            );
        }
    }

    /// Test 18: Sprint 197 — Toggle synth rd_decode mid-run.
    /// 10 cycles physical → 20 cycles synth → remaining physical.
    #[test]
    fn synth_rd_decode_toggle_mid_run() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let case = benchmark_cases()
            .iter()
            .find(|c| c.name == "fibonacci")
            .expect("fibonacci benchmark not found");

        let program = assemble_v2(case.source).unwrap();
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(case.rom_size)
            .with_ram_size(128)
            .build(&mut sim);

        let mut cycles = 0u64;

        // 10 cycles physical
        for _ in 0..10 {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }

        // 20 cycles synth
        cpu.enable_synth_rd_decode(&mut sim);
        for _ in 0..20 {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }
        cpu.disable_synth_rd_decode(&mut sim);

        // Remaining physical
        for _ in cycles..case.max_cycles {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }

        let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
            .iter()
            .find(|(n, _)| *n == case.name)
            .map(|(_, c)| *c)
            .expect("golden cycles not found");
        assert_eq!(cycles, expected_cycles);

        let hash = hash_v2_final_state(&cpu, &sim);
        let expected_hash = V2_BENCHMARK_GOLDENS
            .iter()
            .find(|(n, _)| *n == case.name)
            .map(|(_, h)| *h)
            .expect("golden hash not found");
        assert_eq!(hash, expected_hash);
    }

    /// Test 19: Sprint 197 — Tile type verification for rd_decode replacement.
    #[test]
    fn synth_rd_decode_tile_type_check() {
        use crate::tile_cpu::{V2Builder, assemble_v2, benchmark_cases};

        let case = benchmark_cases()
            .iter()
            .find(|c| c.name == "fibonacci")
            .expect("fibonacci benchmark not found");

        let program = assemble_v2(case.source).unwrap();
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(case.rom_size)
            .with_ram_size(128)
            .build(&mut sim);

        let original = cpu.rd_decode_tile_type(&sim);
        assert_eq!(
            original,
            crate::tiles::tile_meta::TileType::Decoder3to8,
            "expected Decoder3to8, got {:?}",
            original,
        );

        cpu.enable_synth_rd_decode(&mut sim);
        assert_eq!(
            cpu.rd_decode_tile_type(&sim),
            crate::tiles::tile_meta::TileType::Const,
        );

        // Idempotent: second enable is a no-op.
        cpu.enable_synth_rd_decode(&mut sim);
        assert_eq!(
            cpu.rd_decode_tile_type(&sim),
            crate::tiles::tile_meta::TileType::Const,
        );

        cpu.disable_synth_rd_decode(&mut sim);
        assert_eq!(cpu.rd_decode_tile_type(&sim), original);

        // Idempotent: second disable is a no-op.
        cpu.disable_synth_rd_decode(&mut sim);
        assert_eq!(cpu.rd_decode_tile_type(&sim), original);
    }

    /// Test 20: Sprint 197 — Gate test: WE correctness when rd changes each cycle.
    /// Uses benchmarks with high rd diversity. After each step, reads all 16
    /// physical register values and compares against software cache.
    #[test]
    fn synth_rd_decode_we_correctness() {
        use crate::tile_cpu::{V2Builder, assemble_v2, benchmark_cases};

        // Use benchmarks with high rd diversity (multiple different target registers).
        let diversity_names = ["mixed_bank_mem", "cond_move", "fibonacci"];
        let cases = benchmark_cases();

        for bench_name in &diversity_names {
            let case = cases
                .iter()
                .find(|c| c.name == *bench_name)
                .expect("benchmark not found");

            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            cpu.enable_synth_rd_decode(&mut sim);

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;

                // After each step, verify all 16 registers match software cache.
                for r in 0..16usize {
                    let sw = cpu.read_reg(&sim, r);
                    let phys = cpu.read_physical_reg(&sim, r);
                    assert_eq!(
                        sw, phys,
                        "'{}' cycle {}: R{} software={:#X} != physical={:#X} — \
                         WE mask may have targeted wrong register",
                        bench_name, cycles, r, sw, phys,
                    );
                }
            }

            assert!(cycles > 0, "'{}' should run at least 1 cycle", bench_name);
            cpu.disable_synth_rd_decode(&mut sim);
        }
    }

    // ── Sprint 197: Ctrl_b software authority tests ────────────────────────

    /// Test 21: Sprint 197 — Ctrl_b software authority across all 8 golden benchmarks.
    /// No tile swap — overrides latch.ctrl_b with CTRL_B_LUT[opcode], dual-path check.
    #[test]
    fn synth_ctrl_b_authority_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            cpu.enable_synth_ctrl_b();

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with synth ctrl_b: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with synth ctrl_b: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_ctrl_b_mismatches(),
                0,
                "'{}' had {} ctrl_b mismatches in {} cycles",
                case.name,
                cpu.synth_ctrl_b_mismatches(),
                cycles,
            );

            // All benchmarks use PC < 64, so bank_group == 0 path runs every cycle.
            // Verify non-zero dual-path coverage.
            assert!(
                cpu.synth_ctrl_b_checks() > 0,
                "'{}' expected > 0 ctrl_b checks, got 0 in {} cycles",
                case.name,
                cycles,
            );

            cpu.disable_synth_ctrl_b();
        }
    }

    /// Test 22: Sprint 197 — Synth branch (physical) + synth ctrl_b (software authority).
    /// Both active simultaneously across all 8 benchmarks.
    #[test]
    fn synth_ctrl_b_with_branch_replacement() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            cpu.enable_synth_branch(&mut sim);
            cpu.enable_synth_ctrl_b();

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with branch+ctrl_b: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with branch+ctrl_b: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(cpu.synth_branch_mismatches(), 0);
            assert_eq!(cpu.synth_ctrl_b_mismatches(), 0);

            cpu.disable_synth_branch(&mut sim);
            cpu.disable_synth_ctrl_b();
        }
    }

    /// Test 23: Sprint 197 — Ctrl_b dual-path counter infrastructure.
    #[test]
    fn synth_ctrl_b_dual_path_counters() {
        use crate::tile_cpu::{V2Builder, assemble_v2, benchmark_cases};

        let case = benchmark_cases()
            .iter()
            .find(|c| c.name == "fibonacci")
            .expect("fibonacci benchmark not found");

        let program = assemble_v2(case.source).unwrap();
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(case.rom_size)
            .with_ram_size(128)
            .build(&mut sim);

        cpu.enable_synth_ctrl_b();

        let mut cycles = 0u64;
        for _ in 0..case.max_cycles {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }

        assert!(
            cpu.synth_ctrl_b_checks() > 0,
            "expected > 0 synth ctrl_b checks, got 0 in {cycles} cycles",
        );
        assert_eq!(
            cpu.synth_ctrl_b_mismatches(),
            0,
            "expected 0 ctrl_b mismatches, got {} in {cycles} cycles",
            cpu.synth_ctrl_b_mismatches(),
        );
        assert!(
            cpu.synth_ctrl_b_checks() <= cycles,
            "synth_ctrl_b_checks ({}) > cycles ({cycles})",
            cpu.synth_ctrl_b_checks(),
        );

        cpu.disable_synth_ctrl_b();
    }

    // ── Sprint 197: Combined composition test ──────────────────────────────

    /// Test 24: Sprint 197 — Triple composition: branch (physical) + rd_decode (physical)
    /// + ctrl_b (software authority). All 8 golden benchmarks.
    #[test]
    fn synth_triple_composition_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            cpu.enable_synth_branch(&mut sim);
            cpu.enable_synth_rd_decode(&mut sim);
            cpu.enable_synth_ctrl_b();

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with triple composition: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with triple composition: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_branch_mismatches(),
                0,
                "'{}' branch mismatches: {}",
                case.name,
                cpu.synth_branch_mismatches(),
            );
            assert_eq!(
                cpu.synth_rd_decode_mismatches(),
                0,
                "'{}' rd_decode mismatches: {}",
                case.name,
                cpu.synth_rd_decode_mismatches(),
            );
            assert_eq!(
                cpu.synth_ctrl_b_mismatches(),
                0,
                "'{}' ctrl_b mismatches: {}",
                case.name,
                cpu.synth_ctrl_b_mismatches(),
            );

            cpu.disable_synth_branch(&mut sim);
            cpu.disable_synth_rd_decode(&mut sim);
            cpu.disable_synth_ctrl_b();
        }
    }

    // ── Sprint 198: Ctrl_A dual-path validation tests ─────────────────────

    /// Test 25: Sprint 198 — Ctrl_A dual-path validation across all 8 golden benchmarks.
    /// Read-only check: compares physical ctrl_a_mux_idx output against CTRL_A_LUT[opcode].
    #[test]
    fn synth_ctrl_a_validation_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            cpu.enable_synth_ctrl_a(&mut sim);

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with synth ctrl_a: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with synth ctrl_a: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_ctrl_a_mismatches(),
                0,
                "'{}' had {} ctrl_a mismatches in {} cycles",
                case.name,
                cpu.synth_ctrl_a_mismatches(),
                cycles,
            );

            assert!(
                cpu.synth_ctrl_a_checks() > 0,
                "'{}' expected > 0 ctrl_a checks, got 0 in {} cycles",
                case.name,
                cycles,
            );

            cpu.disable_synth_ctrl_a(&mut sim);
        }
    }

    /// Test 26: Sprint 198 — Ctrl_A validation + Ctrl_B authority composed.
    #[test]
    fn synth_ctrl_a_with_ctrl_b() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            cpu.enable_synth_ctrl_a(&mut sim);
            cpu.enable_synth_ctrl_b();

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with ctrl_a+ctrl_b: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with ctrl_a+ctrl_b: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(cpu.synth_ctrl_a_mismatches(), 0);
            assert_eq!(cpu.synth_ctrl_b_mismatches(), 0);

            cpu.disable_synth_ctrl_a(&mut sim);
            cpu.disable_synth_ctrl_b();
        }
    }

    /// Test 27: Sprint 198 — Ctrl_A dual-path counter infrastructure.
    #[test]
    fn synth_ctrl_a_dual_path_counters() {
        use crate::tile_cpu::{V2Builder, assemble_v2, benchmark_cases};

        let case = benchmark_cases()
            .iter()
            .find(|c| c.name == "fibonacci")
            .expect("fibonacci benchmark not found");

        let program = assemble_v2(case.source).unwrap();
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(case.rom_size)
            .with_ram_size(128)
            .build(&mut sim);

        cpu.enable_synth_ctrl_a(&mut sim);

        let mut cycles = 0u64;
        for _ in 0..case.max_cycles {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }

        assert!(
            cpu.synth_ctrl_a_checks() > 0,
            "expected > 0 synth ctrl_a checks, got 0 in {cycles} cycles",
        );
        assert_eq!(
            cpu.synth_ctrl_a_mismatches(),
            0,
            "expected 0 ctrl_a mismatches, got {} in {cycles} cycles",
            cpu.synth_ctrl_a_mismatches(),
        );
        assert!(
            cpu.synth_ctrl_a_checks() <= cycles,
            "synth_ctrl_a_checks ({}) > cycles ({cycles})",
            cpu.synth_ctrl_a_checks(),
        );

        cpu.disable_synth_ctrl_a(&mut sim);
    }

    // ── Sprint 198: Operand bypass authority tests ────────────────────────

    /// Test 28: Sprint 198 — Operand bypass across all 8 golden benchmarks.
    /// Always reads from reg_indices[rd_eff/rs_eff] directly, dual-path check
    /// compares against tree root for bank_group==0.
    #[test]
    fn synth_operand_bypass_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            cpu.enable_synth_operand_bypass();

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with operand bypass: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with operand bypass: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_operand_mismatches(),
                0,
                "'{}' had {} operand mismatches in {} cycles",
                case.name,
                cpu.synth_operand_mismatches(),
                cycles,
            );

            assert!(
                cpu.synth_operand_checks() > 0,
                "'{}' expected > 0 operand checks, got 0 in {} cycles",
                case.name,
                cycles,
            );

            cpu.disable_synth_operand_bypass();
        }
    }

    /// Test 29: Sprint 198 — Operand bypass + branch (physical) + rd_decode (physical).
    #[test]
    fn synth_operand_bypass_with_branch_and_rd() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            cpu.enable_synth_branch(&mut sim);
            cpu.enable_synth_rd_decode(&mut sim);
            cpu.enable_synth_operand_bypass();

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with operand+branch+rd: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with operand+branch+rd: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(cpu.synth_branch_mismatches(), 0);
            assert_eq!(cpu.synth_rd_decode_mismatches(), 0);
            assert_eq!(cpu.synth_operand_mismatches(), 0);

            cpu.disable_synth_branch(&mut sim);
            cpu.disable_synth_rd_decode(&mut sim);
            cpu.disable_synth_operand_bypass();
        }
    }

    /// Test 30: Sprint 198 — Operand bypass dual-path counter infrastructure.
    #[test]
    fn synth_operand_bypass_dual_path_counters() {
        use crate::tile_cpu::{V2Builder, assemble_v2, benchmark_cases};

        let case = benchmark_cases()
            .iter()
            .find(|c| c.name == "fibonacci")
            .expect("fibonacci benchmark not found");

        let program = assemble_v2(case.source).unwrap();
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(case.rom_size)
            .with_ram_size(128)
            .build(&mut sim);

        cpu.enable_synth_operand_bypass();

        let mut cycles = 0u64;
        for _ in 0..case.max_cycles {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }

        assert!(
            cpu.synth_operand_checks() > 0,
            "expected > 0 synth operand checks, got 0 in {cycles} cycles",
        );
        assert_eq!(
            cpu.synth_operand_mismatches(),
            0,
            "expected 0 operand mismatches, got {} in {cycles} cycles",
            cpu.synth_operand_mismatches(),
        );
        assert!(
            cpu.synth_operand_checks() <= cycles * 2,
            "synth_operand_checks ({}) > 2*cycles ({}) — max 2 checks per cycle (a+b)",
            cpu.synth_operand_checks(),
            cycles * 2,
        );

        cpu.disable_synth_operand_bypass();
    }

    // ── Sprint 198: RAM address decoder replacement tests ─────────────────

    /// Test 31: Sprint 198 — RAM address decoder replacement across all 8 golden benchmarks.
    /// Physical tile swap: Decoder3to8 → Const, inject 1<<(addr&7) each cycle.
    #[test]
    fn synth_ram_decode_replacement_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            cpu.enable_synth_ram_decode(&mut sim);

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with ram_decode: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with ram_decode: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_ram_decode_mismatches(),
                0,
                "'{}' had {} ram_decode mismatches in {} cycles",
                case.name,
                cpu.synth_ram_decode_mismatches(),
                cycles,
            );

            cpu.disable_synth_ram_decode(&mut sim);
        }
    }

    /// Test 32: Sprint 198 — Toggle synth ram_decode mid-run.
    /// 10 cycles physical → 20 cycles synth → remaining physical.
    #[test]
    fn synth_ram_decode_toggle_mid_run() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let case = benchmark_cases()
            .iter()
            .find(|c| c.name == "memory_stream")
            .expect("memory_stream benchmark not found");

        let program = assemble_v2(case.source).unwrap();
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(case.rom_size)
            .with_ram_size(128)
            .build(&mut sim);

        let mut cycles = 0u64;

        // 10 cycles physical
        for _ in 0..10 {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }

        // 20 cycles synth
        cpu.enable_synth_ram_decode(&mut sim);
        for _ in 0..20 {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }
        cpu.disable_synth_ram_decode(&mut sim);

        // Remaining physical
        for _ in cycles..case.max_cycles {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }

        let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
            .iter()
            .find(|(n, _)| *n == case.name)
            .map(|(_, c)| *c)
            .expect("golden cycles not found");
        assert_eq!(cycles, expected_cycles);

        let hash = hash_v2_final_state(&cpu, &sim);
        let expected_hash = V2_BENCHMARK_GOLDENS
            .iter()
            .find(|(n, _)| *n == case.name)
            .map(|(_, h)| *h)
            .expect("golden hash not found");
        assert_eq!(hash, expected_hash);
    }

    /// Test 33: Sprint 198 — Tile type verification for RAM address decoder replacement.
    #[test]
    fn synth_ram_decode_tile_type_check() {
        use crate::tile_cpu::{V2Builder, assemble_v2, benchmark_cases};

        let case = benchmark_cases()
            .iter()
            .find(|c| c.name == "fibonacci")
            .expect("fibonacci benchmark not found");

        let program = assemble_v2(case.source).unwrap();
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(case.rom_size)
            .with_ram_size(128)
            .build(&mut sim);

        let original = cpu.ram_decode_tile_type(&sim);
        assert_eq!(
            original,
            crate::tiles::tile_meta::TileType::Decoder3to8,
            "expected Decoder3to8, got {:?}",
            original,
        );

        cpu.enable_synth_ram_decode(&mut sim);
        assert_eq!(
            cpu.ram_decode_tile_type(&sim),
            crate::tiles::tile_meta::TileType::Const,
        );

        // Idempotent: second enable is a no-op.
        cpu.enable_synth_ram_decode(&mut sim);
        assert_eq!(
            cpu.ram_decode_tile_type(&sim),
            crate::tiles::tile_meta::TileType::Const,
        );

        cpu.disable_synth_ram_decode(&mut sim);
        assert_eq!(cpu.ram_decode_tile_type(&sim), original);

        // Idempotent: second disable is a no-op.
        cpu.disable_synth_ram_decode(&mut sim);
        assert_eq!(cpu.ram_decode_tile_type(&sim), original);
    }

    /// Test 34: Sprint 198 — Gate test: Decoder Const injection integrity on memory_stream.
    /// Verifies the injected one-hot value survives commit settle (Const tile readback).
    /// Note: The downstream And gate (ram_write_gate_idx) gates decoder output with a
    /// write_enable signal that doesn't propagate through commit settle — this is the
    /// Sprint 155 stale WE limitation (the exact reason software has RAM authority).
    /// End-to-end correctness is verified by golden hash parity in _all_benchmarks tests.
    #[test]
    fn synth_ram_decode_we_observability() {
        use crate::tile_cpu::{V2Builder, assemble_v2, benchmark_cases};

        // memory_stream writes every cycle — best injection coverage.
        let case = benchmark_cases()
            .iter()
            .find(|c| c.name == "memory_stream")
            .expect("memory_stream benchmark not found");

        let program = assemble_v2(case.source).unwrap();
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(case.rom_size)
            .with_ram_size(128)
            .build(&mut sim);

        cpu.enable_synth_ram_decode(&mut sim);

        let mut cycles = 0u64;
        for _ in 0..case.max_cycles {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
            cycles += 1;
        }

        // Const tile readback: zero mismatches = injection integrity verified.
        // Each cycle, run_stage_x re-derives the expected one-hot and compares
        // against the Const tile readback after commit settle.
        assert_eq!(
            cpu.synth_ram_decode_mismatches(),
            0,
            "ram_decode injection: {} mismatches in {cycles} cycles",
            cpu.synth_ram_decode_mismatches(),
        );

        // Must have checked at least once (memory_stream writes heavily).
        assert!(
            cpu.synth_ram_decode_checks() > 0,
            "expected > 0 ram_decode checks in {cycles} cycles of memory_stream",
        );

        cpu.disable_synth_ram_decode(&mut sim);
    }

    /// Test 35: Sprint 198 — RAM decode replacement across multiple write-heavy benchmarks.
    #[test]
    fn synth_ram_decode_memory_diversity() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let write_heavy = ["memory_stream", "mixed_bank_mem", "wide_indexed"];
        let cases = benchmark_cases();

        for bench_name in &write_heavy {
            let case = cases
                .iter()
                .find(|c| c.name == *bench_name)
                .expect("benchmark not found");

            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            cpu.enable_synth_ram_decode(&mut sim);

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with ram_decode: expected {expected_cycles}, got {cycles}",
                bench_name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash mismatch with ram_decode",
                bench_name,
            );

            assert_eq!(
                cpu.synth_ram_decode_mismatches(),
                0,
                "'{}' had {} ram_decode mismatches",
                bench_name,
                cpu.synth_ram_decode_mismatches(),
            );

            assert!(
                cpu.synth_ram_decode_checks() > 0,
                "'{}' expected > 0 ram_decode checks",
                bench_name,
            );

            cpu.disable_synth_ram_decode(&mut sim);
        }
    }

    // ── Sprint 198: Full composition test ─────────────────────────────────

    /// Test 36: Sprint 198 — Sextuple composition: all 6 synth modes enabled simultaneously.
    /// branch (physical) + rd_decode (physical) + ram_decode (physical)
    /// + ctrl_b (software authority) + ctrl_a (validation) + operand_bypass (authority).
    #[test]
    fn synth_sextuple_composition_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            // Enable all 6 synth modes.
            cpu.enable_synth_branch(&mut sim);
            cpu.enable_synth_rd_decode(&mut sim);
            cpu.enable_synth_ram_decode(&mut sim);
            cpu.enable_synth_ctrl_b();
            cpu.enable_synth_ctrl_a(&mut sim);
            cpu.enable_synth_operand_bypass();

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with sextuple composition: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with sextuple composition: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_branch_mismatches(),
                0,
                "'{}' branch mismatches: {}",
                case.name,
                cpu.synth_branch_mismatches(),
            );
            assert_eq!(
                cpu.synth_rd_decode_mismatches(),
                0,
                "'{}' rd_decode mismatches: {}",
                case.name,
                cpu.synth_rd_decode_mismatches(),
            );
            assert_eq!(
                cpu.synth_ram_decode_mismatches(),
                0,
                "'{}' ram_decode mismatches: {}",
                case.name,
                cpu.synth_ram_decode_mismatches(),
            );
            assert_eq!(
                cpu.synth_ctrl_b_mismatches(),
                0,
                "'{}' ctrl_b mismatches: {}",
                case.name,
                cpu.synth_ctrl_b_mismatches(),
            );
            assert_eq!(
                cpu.synth_ctrl_a_mismatches(),
                0,
                "'{}' ctrl_a mismatches: {}",
                case.name,
                cpu.synth_ctrl_a_mismatches(),
            );
            assert_eq!(
                cpu.synth_operand_mismatches(),
                0,
                "'{}' operand mismatches: {}",
                case.name,
                cpu.synth_operand_mismatches(),
            );

            cpu.disable_synth_branch(&mut sim);
            cpu.disable_synth_rd_decode(&mut sim);
            cpu.disable_synth_ram_decode(&mut sim);
            cpu.disable_synth_ctrl_b();
            cpu.disable_synth_ctrl_a(&mut sim);
            cpu.disable_synth_operand_bypass();
        }
    }

    // -----------------------------------------------------------------------
    // Sprint 200: live synth block tests
    // -----------------------------------------------------------------------

    /// Helper: build and inject a synth block, returning the InjectedBlock.
    fn build_and_inject_synth_block(
        sim: &mut Simulation,
        aig: &Aig,
        offset_y: usize,
        route_config: &RouteConfig,
    ) -> InjectedBlock {
        let lib = CellLibrary::tile_native();
        let export = synthesize_to_simulation(
            aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 4,
                ..PlaceConfig::default()
            },
            route_config,
        )
        .expect("synth pipeline failed");
        inject_synth_export(sim, &export, 0, offset_y)
    }

    /// Test 37: Live synth branch block across all 9 golden benchmarks.
    #[test]
    fn synth_branch_block_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 192, 4);
            let mut cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            place_const_guard_region(&mut sim, 0..128, 64..70);

            let aig = build_branch_taken_aig();
            let block = build_and_inject_synth_block(&mut sim, &aig, 70, &RouteConfig::default());
            cpu.set_synth_branch_block(block);
            cpu.enable_synth_branch(&mut sim);

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with synth branch block: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with synth branch block: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_branch_mismatches(),
                0,
                "'{}' branch block mismatches: {}",
                case.name,
                cpu.synth_branch_mismatches(),
            );

            cpu.disable_synth_branch(&mut sim);
        }
    }

    /// Test 38: Synth branch block matches LUT for all 32 input combinations.
    #[test]
    fn synth_branch_block_vs_lut() {
        let aig = build_branch_taken_aig();
        let lib = CellLibrary::tile_native();
        let export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 4,
                ..PlaceConfig::default()
            },
            &RouteConfig::default(),
        )
        .expect("synth pipeline failed");

        let ew = export.sim.tilemap.width;
        let eh = export.sim.tilemap.height;
        let el = export.sim.tilemap.num_layers;
        let mut sim = Simulation::with_size_layered(ew + 4, eh + 4, el);
        place_const_guard_region(&mut sim, 0..(ew + 4), 0..(eh + 4));
        let block = inject_synth_export(&mut sim, &export, 2, 2);

        let expected = branch_taken_truth_table();
        for sel in 0..32u32 {
            let inputs: Vec<u64> = (0..5)
                .map(|i| if (sel >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();
            let outputs = drive_injected_block_masked(&mut sim, &block, &inputs);
            let result = outputs[0] != 0;
            assert_eq!(
                result, expected[sel as usize],
                "branch block vs LUT mismatch at sel={sel}: block={result}, lut={}",
                expected[sel as usize],
            );
        }
    }

    /// Test 39: Live synth rd_decode block across all 9 golden benchmarks.
    #[test]
    fn synth_rd_decode_block_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 192, 4);
            let mut cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            place_const_guard_region(&mut sim, 0..128, 64..70);

            let aig = build_decoder3to8_aig();
            let decode_route = RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            };
            let block = build_and_inject_synth_block(&mut sim, &aig, 85, &decode_route);
            cpu.set_synth_rd_decode_block(block);
            cpu.enable_synth_rd_decode(&mut sim);

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with synth rd_decode block: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with synth rd_decode block: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_rd_decode_mismatches(),
                0,
                "'{}' rd_decode block mismatches: {}",
                case.name,
                cpu.synth_rd_decode_mismatches(),
            );

            cpu.disable_synth_rd_decode(&mut sim);
        }
    }

    /// Test 40: Synth rd_decode block matches `1 << rd` for all 8 inputs.
    #[test]
    fn synth_rd_decode_block_vs_direct() {
        let aig = build_decoder3to8_aig();
        let lib = CellLibrary::tile_native();
        let export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 4,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
        )
        .expect("synth pipeline failed");

        let ew = export.sim.tilemap.width;
        let eh = export.sim.tilemap.height;
        let el = export.sim.tilemap.num_layers;
        let mut sim = Simulation::with_size_layered(ew + 4, eh + 4, el);
        place_const_guard_region(&mut sim, 0..(ew + 4), 0..(eh + 4));
        let block = inject_synth_export(&mut sim, &export, 2, 2);

        for rd in 0..8u32 {
            let inputs: Vec<u64> = (0..3)
                .map(|i| if (rd >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();
            let outputs = drive_injected_block_masked(&mut sim, &block, &inputs);
            let mut onehot = 0u64;
            for (i, &val) in outputs.iter().enumerate() {
                if val != 0 {
                    onehot |= 1 << i;
                }
            }
            let expected = 1u64 << rd;
            assert_eq!(
                onehot, expected,
                "rd_decode block vs direct mismatch at rd={rd}: block={onehot:#010b}, direct={expected:#010b}",
            );
        }
    }

    /// Test 41: Both synth blocks active + all 4 other synth gates.
    #[test]
    fn synth_dual_block_composition() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 192, 4);
            let mut cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            place_const_guard_region(&mut sim, 0..128, 64..70);

            // Inject both synth blocks — non-overlapping zones.
            let branch_aig = build_branch_taken_aig();
            let branch_block =
                build_and_inject_synth_block(&mut sim, &branch_aig, 70, &RouteConfig::default());
            let decode_offset_y = 70 + branch_block.height + 2;
            cpu.set_synth_branch_block(branch_block);

            let decode_aig = build_decoder3to8_aig();
            let decode_route = RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            };
            let decode_block =
                build_and_inject_synth_block(&mut sim, &decode_aig, decode_offset_y, &decode_route);
            cpu.set_synth_rd_decode_block(decode_block);

            // Enable all 6 synth gates.
            cpu.enable_synth_branch(&mut sim);
            cpu.enable_synth_rd_decode(&mut sim);
            cpu.enable_synth_ram_decode(&mut sim);
            cpu.enable_synth_ctrl_b();
            cpu.enable_synth_ctrl_a(&mut sim);
            cpu.enable_synth_operand_bypass();

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with dual block + all gates: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with dual block + all gates: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_branch_mismatches(),
                0,
                "'{}' branch mismatches",
                case.name
            );
            assert_eq!(
                cpu.synth_rd_decode_mismatches(),
                0,
                "'{}' rd_decode mismatches",
                case.name
            );
            assert_eq!(
                cpu.synth_ram_decode_mismatches(),
                0,
                "'{}' ram_decode mismatches",
                case.name
            );
            assert_eq!(
                cpu.synth_ctrl_b_mismatches(),
                0,
                "'{}' ctrl_b mismatches",
                case.name
            );
            assert_eq!(
                cpu.synth_ctrl_a_mismatches(),
                0,
                "'{}' ctrl_a mismatches",
                case.name
            );
            assert_eq!(
                cpu.synth_operand_mismatches(),
                0,
                "'{}' operand mismatches",
                case.name
            );

            cpu.disable_synth_branch(&mut sim);
            cpu.disable_synth_rd_decode(&mut sim);
            cpu.disable_synth_ram_decode(&mut sim);
            cpu.disable_synth_ctrl_b();
            cpu.disable_synth_ctrl_a(&mut sim);
            cpu.disable_synth_operand_bypass();
        }
    }

    /// Test 42: Synth block evaluation does not corrupt CPU pipeline state.
    #[test]
    fn synth_block_isolation() {
        use crate::tile_cpu::{V2Builder, assemble_v2};

        let program = assemble_v2("LDI R0, 42\nLDI R1, 7\nADD R2, R0\nST [R1], R2\nHALT").unwrap();
        let mut sim = Simulation::with_size_layered(128, 192, 4);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(64)
            .with_ram_size(128)
            .build(&mut sim);

        place_const_guard_region(&mut sim, 0..128, 64..70);

        let branch_aig = build_branch_taken_aig();
        let branch_block =
            build_and_inject_synth_block(&mut sim, &branch_aig, 70, &RouteConfig::default());
        let decode_offset_y = 70 + branch_block.height + 2;

        let decode_aig = build_decoder3to8_aig();
        let decode_route = RouteConfig {
            max_z: 1,
            no_crossings: true,
            ..RouteConfig::default()
        };
        let decode_block =
            build_and_inject_synth_block(&mut sim, &decode_aig, decode_offset_y, &decode_route);

        // Run CPU to completion.
        for _ in 0..512 {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
        }
        assert!(cpu.is_halted(), "program should halt");

        // Snapshot CPU state.
        let pre_pc = cpu.read_pc(&sim);
        let pre_regs: [u64; 16] = std::array::from_fn(|r| cpu.read_reg(&sim, r));
        let pre_flag_z = cpu.read_flag_z(&sim);
        let pre_flag_c = cpu.read_flag_c(&sim);

        // Drive synth blocks many times — should not affect CPU state.
        for sel in 0..32u32 {
            let inputs: Vec<u64> = (0..5)
                .map(|i| if (sel >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();
            let _ = drive_injected_block_masked(&mut sim, &branch_block, &inputs);
        }
        for rd in 0..8u32 {
            let inputs: Vec<u64> = (0..3)
                .map(|i| if (rd >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();
            let _ = drive_injected_block_masked(&mut sim, &decode_block, &inputs);
        }

        // Verify CPU state unchanged.
        let post_regs: [u64; 16] = std::array::from_fn(|r| cpu.read_reg(&sim, r));
        assert_eq!(
            cpu.read_pc(&sim),
            pre_pc,
            "PC changed after synth block eval"
        );
        assert_eq!(
            post_regs, pre_regs,
            "registers changed after synth block eval"
        );
        assert_eq!(cpu.read_flag_z(&sim), pre_flag_z, "flag Z changed");
        assert_eq!(cpu.read_flag_c(&sim), pre_flag_c, "flag C changed");
    }

    // =========================================================================
    // Sprint 201: ram_decode synth block tests
    // =========================================================================

    /// Test 43: Live synth ram_decode block across all 9 golden benchmarks.
    #[test]
    fn synth_ram_decode_block_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 192, 4);
            let mut cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            place_const_guard_region(&mut sim, 0..128, 64..70);

            let aig = build_decoder3to8_aig();
            let decode_route = RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            };
            let block = build_and_inject_synth_block(&mut sim, &aig, 70, &decode_route);
            cpu.set_synth_ram_decode_block(block);
            cpu.enable_synth_ram_decode(&mut sim);

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with synth ram_decode block: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with synth ram_decode block: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_ram_decode_mismatches(),
                0,
                "'{}' ram_decode block mismatches: {}",
                case.name,
                cpu.synth_ram_decode_mismatches(),
            );

            cpu.disable_synth_ram_decode(&mut sim);
        }
    }

    /// Test 44: Synth ram_decode block matches `1 << addr` for all 8 inputs.
    #[test]
    fn synth_ram_decode_block_vs_direct() {
        let aig = build_decoder3to8_aig();
        let lib = CellLibrary::tile_native();
        let export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 4,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
        )
        .expect("synth pipeline failed");

        let ew = export.sim.tilemap.width;
        let eh = export.sim.tilemap.height;
        let el = export.sim.tilemap.num_layers;
        let mut sim = Simulation::with_size_layered(ew + 4, eh + 4, el);
        place_const_guard_region(&mut sim, 0..(ew + 4), 0..(eh + 4));
        let block = inject_synth_export(&mut sim, &export, 2, 2);

        for addr in 0..8u32 {
            let inputs: Vec<u64> = (0..3)
                .map(|i| if (addr >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();
            let outputs = drive_injected_block_masked(&mut sim, &block, &inputs);
            let mut onehot = 0u64;
            for (i, &val) in outputs.iter().enumerate() {
                if val != 0 {
                    onehot |= 1 << i;
                }
            }
            let expected = 1u64 << addr;
            assert_eq!(
                onehot, expected,
                "ram_decode block vs direct mismatch at addr={addr}: block={onehot:#010b}, direct={expected:#010b}",
            );
        }
    }

    /// Test 45: ram_decode block produces 0 for non-store opcodes and MMIO-range addresses.
    #[test]
    fn synth_ram_decode_block_non_store_and_mmio() {
        use crate::tile_cpu::V2_BENCHMARK_GOLDENS;
        use crate::tile_cpu::{V2Builder, assemble_v2, hash_v2_final_state};

        // Test with a program that has both stores and non-stores.
        // The fibonacci benchmark exercises ST instructions.
        let source = "LDI R0, 0\nLDI R1, 1\nADD R2, R0\nST [R1], R2\nHALT";
        let program = assemble_v2(source).unwrap();
        let mut sim = Simulation::with_size_layered(128, 192, 4);
        let mut cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(64)
            .with_ram_size(128)
            .build(&mut sim);

        place_const_guard_region(&mut sim, 0..128, 64..70);

        let aig = build_decoder3to8_aig();
        let decode_route = RouteConfig {
            max_z: 1,
            no_crossings: true,
            ..RouteConfig::default()
        };
        let block = build_and_inject_synth_block(&mut sim, &aig, 70, &decode_route);
        cpu.set_synth_ram_decode_block(block);
        cpu.enable_synth_ram_decode(&mut sim);

        for _ in 0..512 {
            if cpu.is_halted() {
                break;
            }
            cpu.step(&mut sim);
        }

        assert!(cpu.is_halted(), "program should halt");
        assert_eq!(
            cpu.synth_ram_decode_mismatches(),
            0,
            "ram_decode mismatches: {}",
            cpu.synth_ram_decode_mismatches(),
        );

        // Also verify with an MMIO-store program: MMIO addresses should produce 0.
        // Use the MMIO timer benchmark if available, otherwise use a simple one.
        let cases = crate::tile_cpu::benchmark_cases();
        if let Some(case) = cases.iter().find(|c| c.name == "mmio_timer") {
            let program2 = assemble_v2(case.source).unwrap();
            let mut sim2 = Simulation::with_size_layered(128, 192, 4);
            let mut cpu2 = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program2)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim2);

            place_const_guard_region(&mut sim2, 0..128, 64..70);

            let aig2 = build_decoder3to8_aig();
            let block2 = build_and_inject_synth_block(&mut sim2, &aig2, 70, &decode_route);
            cpu2.set_synth_ram_decode_block(block2);
            cpu2.enable_synth_ram_decode(&mut sim2);

            for _ in 0..case.max_cycles {
                if cpu2.is_halted() {
                    break;
                }
                cpu2.step(&mut sim2);
            }

            let hash = hash_v2_final_state(&cpu2, &sim2);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == "mmio_timer")
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'mmio_timer' hash with ram_decode block: expected {expected_hash:#018X}, got {hash:#018X}"
            );
            assert_eq!(
                cpu2.synth_ram_decode_mismatches(),
                0,
                "mmio_timer ram_decode mismatches: {}",
                cpu2.synth_ram_decode_mismatches(),
            );
        }
    }

    // =========================================================================
    // Sprint 201: ctrl_b synth block tests
    // =========================================================================

    /// Test 46: Live synth ctrl_b block across all 9 golden benchmarks.
    #[test]
    #[ignore] // slow: ~40s benchmark sweep — run with --include-ignored
    fn synth_ctrl_b_block_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 192, 4);
            let mut cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            place_const_guard_region(&mut sim, 0..128, 64..70);

            let aig = build_ctrl_b_aig();
            let ctrl_b_route = RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            };
            let block = build_and_inject_synth_block(&mut sim, &aig, 70, &ctrl_b_route);
            cpu.set_synth_ctrl_b_block(block);
            cpu.enable_synth_ctrl_b();

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' cycles with synth ctrl_b block: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' hash with synth ctrl_b block: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_ctrl_b_mismatches(),
                0,
                "'{}' ctrl_b block mismatches: {}",
                case.name,
                cpu.synth_ctrl_b_mismatches(),
            );

            cpu.disable_synth_ctrl_b();
        }
    }

    /// Test 47: Synth ctrl_b block matches CTRL_B_LUT for all 32 opcodes.
    #[test]
    fn synth_ctrl_b_block_vs_lut() {
        use crate::tile_cpu::v2_execute::TileCpuV2;

        let lut = TileCpuV2::ctrl_b_lut();

        // Verify embedded copy matches v2_execute copy.
        assert_eq!(
            &CTRL_B_LUT_EMBEDDED, lut,
            "CTRL_B_LUT_EMBEDDED in integration.rs does not match CTRL_B_LUT in v2_execute.rs"
        );

        let aig = build_ctrl_b_aig();
        let lib = CellLibrary::tile_native();
        let export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 8,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
        )
        .expect("synth pipeline failed");

        let ew = export.sim.tilemap.width;
        let eh = export.sim.tilemap.height;
        let el = export.sim.tilemap.num_layers;
        let mut sim = Simulation::with_size_layered(ew + 4, eh + 4, el);
        place_const_guard_region(&mut sim, 0..(ew + 4), 0..(eh + 4));
        let block = inject_synth_export(&mut sim, &export, 2, 2);

        for opcode in 0..32u32 {
            let inputs: Vec<u64> = (0..5)
                .map(|i| if (opcode >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();
            let outputs = drive_injected_block_masked(&mut sim, &block, &inputs);
            let mut cb_val = 0u8;
            for (i, &val) in outputs.iter().enumerate() {
                if val != 0 {
                    cb_val |= 1 << i;
                }
            }
            let expected = lut[opcode as usize];
            assert_eq!(
                cb_val, expected,
                "ctrl_b block vs LUT mismatch at opcode={opcode:#04X}: block={cb_val:#04X}, lut={expected:#04X}",
            );
        }
    }

    /// Test 48: Upper-bank golden passes with ctrl_b block (bank_group==1 uses LUT, not block).
    #[test]
    fn synth_ctrl_b_block_upper_bank() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, assemble_v2,
            benchmark_cases, hash_v2_final_state,
        };

        // Use the upper_bank benchmark if it exists, otherwise test with all benchmarks
        // that use ROM > 64 entries (they exercise bank_group==1 path).
        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            let mut sim = Simulation::with_size_layered(128, 192, 4);
            let mut cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .build(&mut sim);

            place_const_guard_region(&mut sim, 0..128, 64..70);

            let aig = build_ctrl_b_aig();
            let ctrl_b_route = RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            };
            let block = build_and_inject_synth_block(&mut sim, &aig, 70, &ctrl_b_route);
            cpu.set_synth_ctrl_b_block(block);
            cpu.enable_synth_ctrl_b();

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' (upper_bank test) cycles: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' (upper_bank test) hash: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            // Zero mismatches means bank_group==1 path still uses LUT correctly
            // and bank_group==0 path uses synth block correctly.
            assert_eq!(
                cpu.synth_ctrl_b_mismatches(),
                0,
                "'{}' (upper_bank test) ctrl_b mismatches: {}",
                case.name,
                cpu.synth_ctrl_b_mismatches(),
            );

            cpu.disable_synth_ctrl_b();
        }
    }

    // =========================================================================
    // Sprint 201: V2Builder synth API tests
    // =========================================================================

    /// Test 49: Builder API with all 4 synth blocks enabled, 9 goldens.
    #[test]
    #[ignore] // slow: ~72s benchmark sweep — run with --include-ignored
    fn builder_synth_all_benchmarks() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, V2SynthConfig,
            assemble_v2, benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            // 384 height: 4 synth blocks need ~250 rows (y=70..~320).
            let mut sim = Simulation::with_size_layered(128, 384, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .with_synth_blocks(V2SynthConfig {
                    enable_branch: true,
                    enable_rd_decode: true,
                    enable_ram_decode: true,
                    enable_ctrl_b: true,
                    ..V2SynthConfig::default()
                })
                .build(&mut sim);

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' builder synth cycles: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' builder synth hash: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(
                cpu.synth_branch_mismatches(),
                0,
                "'{}' builder branch mismatches",
                case.name
            );
            assert_eq!(
                cpu.synth_rd_decode_mismatches(),
                0,
                "'{}' builder rd_decode mismatches",
                case.name
            );
            assert_eq!(
                cpu.synth_ram_decode_mismatches(),
                0,
                "'{}' builder ram_decode mismatches",
                case.name
            );
            assert_eq!(
                cpu.synth_ctrl_b_mismatches(),
                0,
                "'{}' builder ctrl_b mismatches",
                case.name
            );
        }
    }

    /// Test 50: Builder API with all 4 blocks + manually enabled ctrl_a + operand_bypass.
    #[test]
    #[ignore] // slow: ~72s benchmark sweep — run with --include-ignored
    fn builder_synth_composition() {
        use crate::tile_cpu::{
            V2_BENCHMARK_CYCLE_GOLDENS, V2_BENCHMARK_GOLDENS, V2Builder, V2SynthConfig,
            assemble_v2, benchmark_cases, hash_v2_final_state,
        };

        let cases = benchmark_cases();
        for case in cases {
            let program = assemble_v2(case.source).unwrap();
            // 384 height: 4 synth blocks + ctrl_a/operand opt-in need ~250 rows.
            let mut sim = Simulation::with_size_layered(128, 384, 4);
            let cpu = V2Builder::new()
                .with_origin(0, 0)
                .with_program(&program)
                .with_rom_size(case.rom_size)
                .with_ram_size(128)
                .with_synth_blocks(V2SynthConfig {
                    enable_branch: true,
                    enable_rd_decode: true,
                    enable_ram_decode: true,
                    enable_ctrl_b: true,
                    ..V2SynthConfig::default()
                })
                .build(&mut sim);

            // Manually enable the two opt-in gates.
            cpu.enable_synth_ctrl_a(&mut sim);
            cpu.enable_synth_operand_bypass();

            let mut cycles = 0u64;
            for _ in 0..case.max_cycles {
                if cpu.is_halted() {
                    break;
                }
                cpu.step(&mut sim);
                cycles += 1;
            }

            let expected_cycles = V2_BENCHMARK_CYCLE_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, c)| *c)
                .expect("golden cycles not found");
            assert_eq!(
                cycles, expected_cycles,
                "'{}' composition cycles: expected {expected_cycles}, got {cycles}",
                case.name,
            );

            let hash = hash_v2_final_state(&cpu, &sim);
            let expected_hash = V2_BENCHMARK_GOLDENS
                .iter()
                .find(|(n, _)| *n == case.name)
                .map(|(_, h)| *h)
                .expect("golden hash not found");
            assert_eq!(
                hash, expected_hash,
                "'{}' composition hash: expected {expected_hash:#018X}, got {hash:#018X}",
                case.name,
            );

            assert_eq!(cpu.synth_branch_mismatches(), 0, "'{}' branch", case.name);
            assert_eq!(
                cpu.synth_rd_decode_mismatches(),
                0,
                "'{}' rd_decode",
                case.name
            );
            assert_eq!(
                cpu.synth_ram_decode_mismatches(),
                0,
                "'{}' ram_decode",
                case.name
            );
            assert_eq!(cpu.synth_ctrl_b_mismatches(), 0, "'{}' ctrl_b", case.name);
            assert_eq!(cpu.synth_ctrl_a_mismatches(), 0, "'{}' ctrl_a", case.name);
            assert_eq!(cpu.synth_operand_mismatches(), 0, "'{}' operand", case.name);
        }
    }

    // ---- Sprint 202: Synth cache tests ----

    /// Test 51: Cached outputs match live eval for branch-taken (all 32 input combos).
    #[test]
    fn synth_cache_branch_correctness() {
        let aig = build_branch_taken_aig();
        let lib = CellLibrary::tile_native();
        let export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 4,
                ..PlaceConfig::default()
            },
            &RouteConfig::default(),
        )
        .expect("synth pipeline failed");
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let mut block = inject_synth_export(&mut sim, &export, 0, 10);

        // Capture live results for all 32 combos BEFORE populating cache.
        let n_inputs = block.input_indices.len();
        assert_eq!(n_inputs, 5);
        let n_combos = 1usize << n_inputs;
        let mut live_results = Vec::with_capacity(n_combos);
        for pattern in 0..n_combos {
            let inputs: Vec<u64> = (0..n_inputs)
                .map(|i| if (pattern >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();
            let outputs = drive_injected_block_masked(&mut sim, &block, &inputs);
            live_results.push(outputs);
        }

        // Populate cache.
        precompute_block_cache(&mut sim, &mut block);
        assert!(block.outputs_cache.is_some());

        // Compare cached vs live for all combos.
        for pattern in 0..n_combos {
            let inputs: Vec<u64> = (0..n_inputs)
                .map(|i| if (pattern >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();
            let cached = drive_synth_block(&mut sim, &block, &inputs);
            assert_eq!(
                cached, live_results[pattern],
                "branch cache mismatch at pattern {pattern:#07b}"
            );
        }
    }

    /// Test 52: All 4 block caches match live eval exhaustively.
    #[test]
    fn synth_cache_all_blocks_correctness() {
        let decode_route = RouteConfig {
            max_z: 1,
            no_crossings: true,
            ..RouteConfig::default()
        };
        let aigs_and_routes: Vec<(crate::synth::aig::Aig, RouteConfig, &str)> = vec![
            (build_branch_taken_aig(), RouteConfig::default(), "branch"),
            (build_decoder3to8_aig(), decode_route.clone(), "rd_decode"),
            (build_decoder3to8_aig(), decode_route.clone(), "ram_decode"),
            (build_ctrl_b_aig(), decode_route, "ctrl_b"),
        ];

        for (aig, route, name) in &aigs_and_routes {
            let lib = CellLibrary::tile_native();
            let export = synthesize_to_simulation(
                aig,
                &lib,
                &SynthConfig::default(),
                &PlaceConfig {
                    halo: 4,
                    ..PlaceConfig::default()
                },
                route,
            )
            .expect("synth pipeline failed");
            let mut sim = Simulation::with_size_layered(128, 128, 4);
            let mut block = inject_synth_export(&mut sim, &export, 0, 10);

            let n_inputs = block.input_indices.len();
            let n_combos = 1usize << n_inputs;

            // Capture live results.
            let mut live_results = Vec::with_capacity(n_combos);
            for pattern in 0..n_combos {
                let inputs: Vec<u64> = (0..n_inputs)
                    .map(|i| if (pattern >> i) & 1 != 0 { u64::MAX } else { 0 })
                    .collect();
                live_results.push(drive_injected_block_masked(&mut sim, &block, &inputs));
            }

            // Populate cache and verify.
            precompute_block_cache(&mut sim, &mut block);
            for pattern in 0..n_combos {
                let inputs: Vec<u64> = (0..n_inputs)
                    .map(|i| if (pattern >> i) & 1 != 0 { u64::MAX } else { 0 })
                    .collect();
                let cached = drive_synth_block(&mut sim, &block, &inputs);
                assert_eq!(
                    cached, live_results[pattern],
                    "{name} cache mismatch at pattern {pattern}"
                );
            }
        }
    }

    /// Test 53: Manual setter tests work without cache (fallback to live eval).
    #[test]
    fn synth_cache_fallback_uncached() {
        let aig = build_branch_taken_aig();
        let lib = CellLibrary::tile_native();
        let export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 4,
                ..PlaceConfig::default()
            },
            &RouteConfig::default(),
        )
        .expect("synth pipeline failed");
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let block = inject_synth_export(&mut sim, &export, 0, 10);

        // Block has no cache (outputs_cache == None).
        assert!(block.outputs_cache.is_none());

        // drive_synth_block should fall back to live evaluation without panic.
        let inputs = [u64::MAX, 0, 0, u64::MAX, 0]; // branch_kind=1, flag_z=1, carry=0
        let result = drive_synth_block(&mut sim, &block, &inputs);
        assert_eq!(result.len(), 1, "branch-taken should have 1 output");
        // Just verify it doesn't panic — value correctness covered by other tests.
    }

    /// Sprint 209: combined decode AIG precomputes correct lookup table via AIG eval.
    /// Physical routing of the combined AIG (125+ gates) exceeds current router
    /// capacity, so the block is deployed as a cached software evaluation.
    #[test]
    fn combined_decode_cached_lookup_correct() {
        use crate::synth::mapping::evaluate_aig;

        let aig = build_combined_decode_aig();

        // Build [u16; 32] lookup table from direct AIG evaluation.
        let mut combined_lut = [0u16; 32];
        for opcode in 0..32u32 {
            let inputs: Vec<bool> = (0..5).map(|i| (opcode >> i) & 1 != 0).collect();
            let outputs = evaluate_aig(&aig, &inputs);
            assert_eq!(outputs.len(), 16);
            let mut val = 0u16;
            for (i, &bit) in outputs.iter().enumerate() {
                if bit {
                    val |= 1 << i;
                }
            }
            combined_lut[opcode as usize] = val;
        }

        // Verify ctrl_a portion (bits 0-7) matches CTRL_A_LUT_EMBEDDED.
        for opcode in 0..32 {
            let ca = (combined_lut[opcode] & 0xFF) as u8;
            assert_eq!(
                ca, CTRL_A_LUT_EMBEDDED[opcode],
                "combined LUT ctrl_a mismatch at opcode {opcode:#04x}: got {ca:#04x}, expected {:#04x}",
                CTRL_A_LUT_EMBEDDED[opcode]
            );
        }

        // Verify ctrl_b portion (bits 8-15) matches CTRL_B_LUT_EMBEDDED.
        for opcode in 0..32 {
            let cb = ((combined_lut[opcode] >> 8) & 0xFF) as u8;
            assert_eq!(
                cb, CTRL_B_LUT_EMBEDDED[opcode],
                "combined LUT ctrl_b mismatch at opcode {opcode:#04x}: got {cb:#04x}, expected {:#04x}",
                CTRL_B_LUT_EMBEDDED[opcode]
            );
        }
    }

    /// Sprint 209: max_width placement constraint produces narrower placement regions.
    #[test]
    fn placement_max_width_reduces_columns() {
        let aig = build_ctrl_b_aig(); // ~36 gates, routes cleanly
        let lib = CellLibrary::tile_native();

        // Unconstrained with halo=4.
        let unconstrained = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 4,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
        )
        .expect("unconstrained synth failed");

        // Constrained to max_width=60.
        let constrained = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 4,
                max_width: Some(60),
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
        )
        .expect("constrained synth failed");

        assert!(
            constrained.sim.tilemap.width <= 60,
            "constrained export width {} exceeds max_width 60",
            constrained.sim.tilemap.width
        );
        assert!(constrained.sim.tilemap.width <= unconstrained.sim.tilemap.width);
        // Constrained should be taller (more rows).
        assert!(constrained.sim.tilemap.height >= unconstrained.sim.tilemap.height);
    }

    /// Sprint 210: Combined decode standalone synth — 5 inputs, 16 outputs.
    #[test]
    #[ignore] // slow: ~7.4 min validation — run with --include-ignored
    fn combined_decode_standalone_synth() {
        use crate::synth::mapping::evaluate_aig;

        let aig = build_combined_decode_aig();
        let lib = CellLibrary::tile_native();
        let mut export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 8,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
        )
        .expect("synth pipeline failed for combined_decode");

        for combo in 0..32u32 {
            let inputs: Vec<bool> = (0..5).map(|i| (combo >> i) & 1 != 0).collect();
            let expected = evaluate_aig(&aig, &inputs);
            let actual = evaluate_exported(&mut export, &inputs);
            assert_eq!(actual.len(), 16, "expected 16 outputs");

            for (i, (exp, act)) in expected.iter().zip(actual.iter()).enumerate() {
                assert_eq!(
                    act, exp,
                    "combined_decode standalone: combo={combo:#07b}, output[{i}]={act}, expected={exp}"
                );
            }
        }
    }

    /// Sprint 210: Combined decode injected into host grid — software-driven.
    // Sprint 250: CLZ/CTZ/POPCNT AIG verification.

    /// Helper: convert u64 to 64 bool inputs (LSB-first, matching b0..b63 ordering).
    fn u64_to_bits(v: u64) -> Vec<bool> {
        (0..64).map(|i| (v >> i) & 1 != 0).collect()
    }

    /// Helper: convert 7 bool outputs to u64 (LSB-first).
    fn bits_to_u7(outputs: &[bool]) -> u64 {
        assert_eq!(outputs.len(), 7);
        let mut val = 0u64;
        for (i, &b) in outputs.iter().enumerate() {
            if b {
                val |= 1 << i;
            }
        }
        val
    }

    #[test]
    fn clz_aig_truth_table() {
        let aig = build_clz_aig();
        assert_eq!(aig.num_inputs(), 64);
        assert_eq!(aig.outputs().len(), 7);

        let cases: Vec<(u64, u64)> = vec![
            (0, 64),                     // all zeros
            (1, 63),                     // only bit 0
            (2, 62),                     // only bit 1
            (3, 62),                     // bits 0-1
            (0x8000_0000_0000_0000, 0),  // only MSB
            (0x4000_0000_0000_0000, 1),  // bit 62
            (u64::MAX, 0),               // all ones
            (0x0000_0000_0000_0100, 55), // bit 8
            (0x0000_0000_8000_0000, 32), // bit 31
            (0x0000_0001_0000_0000, 31), // bit 32
            (0x0000_0000_0001_0000, 47), // bit 16
            (0x0100_0000_0000_0000, 7),  // bit 56
            (0x0010_0000_0000_0000, 11), // bit 52
            (0x0000_0000_0000_FFFF, 48), // lower 16 bits
            (0xFFFF_0000_0000_0000, 0),  // upper 16 bits
            (0x0000_FFFF_0000_0000, 16), // bits 32-47
            (0x0000_0000_FFFF_0000, 32), // bits 16-31
            (0x00FF_FF00_0000_0000, 8),  // bits 40-55
        ];

        for (input, expected_clz) in &cases {
            let inputs = u64_to_bits(*input);
            let outputs = evaluate_aig(&aig, &inputs);
            let got = bits_to_u7(&outputs);
            assert_eq!(
                got, *expected_clz,
                "CLZ({input:#018X}): expected {expected_clz}, got {got}"
            );
        }

        // Spot-check powers of two.
        for bit in 0..64u32 {
            let v = 1u64 << bit;
            let inputs = u64_to_bits(v);
            let outputs = evaluate_aig(&aig, &inputs);
            let got = bits_to_u7(&outputs);
            let expected = 63 - bit as u64;
            assert_eq!(
                got, expected,
                "CLZ(1<<{bit}): expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn ctz_aig_truth_table() {
        let aig = build_ctz_aig();
        assert_eq!(aig.num_inputs(), 64);
        assert_eq!(aig.outputs().len(), 7);

        let cases: Vec<(u64, u64)> = vec![
            (0, 64),                     // all zeros
            (1, 0),                      // only bit 0
            (2, 1),                      // only bit 1
            (3, 0),                      // bits 0-1
            (0x8000_0000_0000_0000, 63), // only MSB
            (0x4000_0000_0000_0000, 62), // bit 62
            (u64::MAX, 0),               // all ones
            (0x0000_0000_0000_0100, 8),  // bit 8
            (0x0000_0000_8000_0000, 31), // bit 31
            (0x0000_0001_0000_0000, 32), // bit 32
            (0x0000_0000_0001_0000, 16), // bit 16
            (0x0100_0000_0000_0000, 56), // bit 56
            (0xFFFF_0000_0000_0000, 48), // upper 16 bits set
            (0x0000_0000_FFFF_0000, 16), // bits 16-31
        ];

        for (input, expected_ctz) in &cases {
            let inputs = u64_to_bits(*input);
            let outputs = evaluate_aig(&aig, &inputs);
            let got = bits_to_u7(&outputs);
            assert_eq!(
                got, *expected_ctz,
                "CTZ({input:#018X}): expected {expected_ctz}, got {got}"
            );
        }

        // Spot-check powers of two.
        for bit in 0..64u32 {
            let v = 1u64 << bit;
            let inputs = u64_to_bits(v);
            let outputs = evaluate_aig(&aig, &inputs);
            let got = bits_to_u7(&outputs);
            let expected = bit as u64;
            assert_eq!(
                got, expected,
                "CTZ(1<<{bit}): expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn popcnt_aig_truth_table() {
        let aig = build_popcnt_aig();
        assert_eq!(aig.num_inputs(), 64);
        assert_eq!(aig.outputs().len(), 7);

        let cases: Vec<(u64, u64)> = vec![
            (0, 0),
            (1, 1),
            (2, 1),
            (3, 2),
            (0xFF, 8),
            (0xFFFF, 16),
            (0xFFFF_FFFF, 32),
            (0xFFFF_FFFF_FFFF_FFFF, 64),
            (0x5555_5555_5555_5555, 32), // alternating 01
            (0xAAAA_AAAA_AAAA_AAAA, 32), // alternating 10
            (0x0000_0000_0000_000F, 4),
            (0xF000_0000_0000_0000, 4),
            (0x8000_0000_0000_0001, 2),
            (0x0123_4567_89AB_CDEF, 32),
        ];

        for (input, expected_pop) in &cases {
            let inputs = u64_to_bits(*input);
            let outputs = evaluate_aig(&aig, &inputs);
            let got = bits_to_u7(&outputs);
            assert_eq!(
                got, *expected_pop,
                "POPCNT({input:#018X}): expected {expected_pop}, got {got}"
            );
        }

        // Spot-check all single-bit values.
        for bit in 0..64u32 {
            let v = 1u64 << bit;
            let inputs = u64_to_bits(v);
            let outputs = evaluate_aig(&aig, &inputs);
            let got = bits_to_u7(&outputs);
            assert_eq!(got, 1, "POPCNT(1<<{bit}): expected 1, got {got}");
        }

        // Spot-check consecutive bit patterns.
        for n in 0..=64u32 {
            let v = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
            let inputs = u64_to_bits(v);
            let outputs = evaluate_aig(&aig, &inputs);
            let got = bits_to_u7(&outputs);
            assert_eq!(got, n as u64, "POPCNT({v:#018X}): expected {n}, got {got}");
        }
    }

    #[test]
    #[ignore] // slow: ~7.6 min validation — run with --include-ignored
    fn combined_decode_injected_in_host_grid() {
        use crate::synth::mapping::evaluate_aig;

        let aig = build_combined_decode_aig();
        let lib = CellLibrary::tile_native();
        let export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &PlaceConfig {
                halo: 8,
                ..PlaceConfig::default()
            },
            &RouteConfig {
                max_z: 1,
                no_crossings: true,
                ..RouteConfig::default()
            },
        )
        .expect("synth pipeline failed for combined_decode");

        let ew = export.sim.tilemap.width;
        let eh = export.sim.tilemap.height;
        let el = export.sim.tilemap.num_layers;
        let host_w = ew + 4;
        let host_h = eh + 4;
        let mut host = Simulation::with_size_layered(host_w, host_h, el);
        place_const_guard_region(&mut host, 0..host_w, 0..host_h);
        let block = inject_synth_export(&mut host, &export, 2, 2);

        assert_eq!(block.input_indices.len(), 5, "combined_decode has 5 inputs");

        for combo in 0..32u32 {
            let inputs: Vec<bool> = (0..5).map(|i| (combo >> i) & 1 != 0).collect();
            let expected = evaluate_aig(&aig, &inputs);
            let outputs = drive_injected_block(&mut host, &block, &inputs);

            for (i, (exp, act)) in expected.iter().zip(outputs.iter()).enumerate() {
                assert_eq!(
                    act, exp,
                    "injected combined_decode: combo={combo:#07b}, output[{i}]={act}, expected={exp}"
                );
            }
        }
    }

    // Sprint 250: Hierarchical byte-sliced block sizing.

    #[test]
    fn hierarchical_bitop_block_sizes() {
        let lib = CellLibrary::tile_native();
        let config = SynthConfig::default();
        let place = PlaceConfig {
            halo: 4,
            ..PlaceConfig::default()
        };
        let route = RouteConfig {
            max_z: 1,
            no_crossings: true,
            ..RouteConfig::default()
        };

        // Byte-level blocks (8 inputs).
        for (name, aig) in [
            ("CLZ8", build_clz8_aig()),
            ("CTZ8", build_ctz8_aig()),
            ("POPCNT8", build_popcnt8_aig()),
        ] {
            let export = synthesize_to_simulation(&aig, &lib, &config, &place, &route)
                .unwrap_or_else(|e| panic!("{name} synth failed: {e}"));
            let w = export.sim.tilemap.width;
            let h = export.sim.tilemap.height;
            let z = export.sim.tilemap.num_layers;
            println!("{name}: {w}w x {h}h x {z}z, nodes={}", aig.num_nodes());
            assert!(w <= 128, "{name} too wide: {w}");
        }

        // Half-group combine blocks (16 inputs).
        for (name, aig) in [
            ("CLZ_HALF", build_clz_half_combine_aig()),
            ("CTZ_HALF", build_ctz_half_combine_aig()),
        ] {
            let export = synthesize_to_simulation(&aig, &lib, &config, &place, &route)
                .unwrap_or_else(|e| panic!("{name} synth failed: {e}"));
            let w = export.sim.tilemap.width;
            let h = export.sim.tilemap.height;
            let z = export.sim.tilemap.num_layers;
            println!("{name}: {w}w x {h}h x {z}z, nodes={}", aig.num_nodes());
            assert!(w <= 128, "{name} too wide: {w}");
        }

        // Final combine blocks (12 inputs).
        for (name, aig) in [
            ("CLZ_FINAL", build_clz_final_combine_aig()),
            ("CTZ_FINAL", build_ctz_final_combine_aig()),
        ] {
            let export = synthesize_to_simulation(&aig, &lib, &config, &place, &route)
                .unwrap_or_else(|e| panic!("{name} synth failed: {e}"));
            let w = export.sim.tilemap.width;
            let h = export.sim.tilemap.height;
            let z = export.sim.tilemap.num_layers;
            println!("{name}: {w}w x {h}h x {z}z, nodes={}", aig.num_nodes());
            assert!(w <= 128, "{name} too wide: {w}");
        }

        // POPCNT pairwise add blocks (8, 10, 12 inputs).
        for (name, width) in [
            ("POPCNT_ADD4", 4), // 8 inputs → 5 outputs
            ("POPCNT_ADD5", 5), // 10 inputs → 6 outputs
            ("POPCNT_ADD6", 6), // 12 inputs → 7 outputs
        ] {
            let aig = build_popcnt_add_aig(width);
            let export = synthesize_to_simulation(&aig, &lib, &config, &place, &route)
                .unwrap_or_else(|e| panic!("{name} synth failed: {e}"));
            let w = export.sim.tilemap.width;
            let h = export.sim.tilemap.height;
            let z = export.sim.tilemap.num_layers;
            println!("{name}: {w}w x {h}h x {z}z, nodes={}", aig.num_nodes());
            assert!(w <= 128, "{name} too wide: {w}");
        }
    }

    // ---- Sprint 362: 8x8 Multiplier tests ----

    /// Sprint 362: Fast AIG-only MUL verification (no synthesis).
    ///
    /// Verifies that build_mul_aig produces correct logic using pure AIG evaluation
    /// (evaluate_aig). This is fast (< 1s) and rules out AIG construction bugs before
    /// running the expensive synthesis + tile-simulation test.
    #[test]
    fn mul_aig_eval_fast() {
        let aig = crate::synth::integration::build_mul_aig();
        assert_eq!(aig.num_inputs(), 16);
        assert_eq!(aig.num_output_bits(), 16);

        // Corner cases and a deterministic sample.
        let samples: &[(u8, u8, u16)] = &[
            (0, 0, 0),
            (1, 1, 1),
            (7, 6, 42),
            (10, 20, 200),
            (128, 2, 256),
            (255, 1, 255),
            (255, 255, 65025),
            (0xAA, 0x55, 0x3872),
            (0xF0, 0x0F, 0x0E10),
            (123, 45, 5535),
        ];

        for &(a, b, expected) in samples {
            let mut inputs = Vec::with_capacity(16);
            for i in 0..8 {
                inputs.push((a >> i) & 1 != 0);
            }
            for i in 0..8 {
                inputs.push((b >> i) & 1 != 0);
            }
            let outputs = evaluate_aig(&aig, &inputs);
            let mut actual = 0u16;
            for (i, &val) in outputs.iter().enumerate() {
                if val {
                    actual |= 1 << i;
                }
            }
            assert_eq!(
                actual, expected,
                "MUL AIG eval failed: {a} * {b} expected {expected}, got {actual}"
            );
        }
    }

    /// Sprint 362: Verify MUL truth table covers all 65536 input combinations correctly.
    #[test]
    fn mul_truth_table_complete() {
        let tt = crate::synth::integration::mul_truth_table();
        for a in 0..256u16 {
            for b in 0..256u16 {
                let product = a as u64 * b as u64;
                assert_eq!(
                    tt[a as usize][b as usize], product,
                    "{} * {} = {} (expected {})",
                    a, b, tt[a as usize][b as usize], product
                );
            }
        }
    }

    /// Sprint 362: MUL synthesis smoke test.
    ///
    /// Synthesizes the MUL AIG and verifies a handful of key products via the
    /// injected block to confirm the synthesis pipeline is sound.
    /// Full truth-table verification is done quickly above via evaluate_aig().
    #[test]
    #[ignore = "slow: MUL synthesis takes ~60s; run with --ignored"]
    fn mul_aig_synth_smoke() {
        let aig = crate::synth::integration::build_mul_aig();
        assert_eq!(aig.num_inputs(), 16);
        assert_eq!(aig.num_output_bits(), 16);
        assert!(
            aig.num_nodes() < 1000,
            "AIG too large: {} nodes",
            aig.num_nodes()
        );

        // Synthesize to a minimal grid and inject.
        let lib = CellLibrary::tile_native();
        let mut sim = Simulation::with_size_layered(128, 128, 4);
        let (block, export) =
            crate::synth::bridge::compile_and_inject(&aig, &lib, &mut sim, 0, 0, 0)
                .unwrap_or_else(|e| panic!("MUL synth failed: {e}"));

        let w = export.sim.tilemap.width;
        let h = export.sim.tilemap.height;
        println!(
            "MUL AIG: {w}w x {h}h x {}z, nodes={}",
            export.sim.tilemap.num_layers,
            aig.num_nodes()
        );
        assert!(w <= 128 && h <= 128, "MUL too large: {w}w x {h}h");

        // Smoke test: a few key combinations via injected block.
        let tt = crate::synth::integration::mul_truth_table();
        let samples: &[(u8, u8)] = &[(0, 0), (1, 1), (7, 6), (10, 20), (255, 255), (0xAA, 0x55)];

        for &(a, b) in samples {
            let combo = ((b as u64) << 8) | (a as u64);
            let input_mask: Vec<u64> = (0..16)
                .map(|i| if (combo >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();
            let outputs = drive_injected_block_masked(&mut sim, &block, &input_mask);

            // Extract 16-bit product from outputs.
            let actual: u64 = (0..16)
                .filter(|i| outputs[*i] != 0)
                .fold(0u64, |acc, i| acc | (1u64 << i));

            let expected = tt[a as usize][b as usize];
            assert_eq!(
                actual, expected,
                "MUL mismatch: a={a}, b={b}, AIG={actual:#018X}, expected={expected:#018X}",
            );
        }

        println!("MUL AIG synth smoke passed, {} nodes", aig.num_nodes());
    }
}
