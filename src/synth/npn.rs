//! NPN4 equivalence classification for 4-input Boolean functions.
//!
//! Every 4-input Boolean function (represented as a u16 truth table) belongs to
//! one of 222 NPN equivalence classes. Two functions are NPN-equivalent if one
//! can be obtained from the other by:
//! - **N**egating inputs (any subset of the 4 inputs)
//! - **P**ermuting inputs (any of the 24 permutations)
//! - **N**egating the output
//!
//! This module provides O(1) classification via a precomputed lookup table
//! (65536 entries, one per possible truth table), built on first access using
//! `LazyLock`.

use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Npn4Class — classification result
// ---------------------------------------------------------------------------

/// Classification of a 4-input truth table into its NPN4 equivalence class.
///
/// Stores the transform that maps **canonical → this truth table**:
/// `npn4_apply_transform(canonical_tt, perm, neg_in, neg_out) = this_tt`
///
/// This is the direction needed by AIG rewriting: given a cut's truth table,
/// look up the canonical class, get the optimal subgraph (which computes canonical),
/// then apply the stored transform to wire the subgraph inputs correctly.
#[derive(Clone, Copy, Debug)]
pub struct Npn4Class {
    /// Class identifier (0..221). There are exactly 222 NPN4 classes.
    pub class_id: u8,
    /// The canonical (lexicographically smallest) truth table for this class.
    pub canonical_tt: u16,
    /// Input permutation: `npn4_apply_transform(canonical, perm, neg_in, neg_out) = this_tt`
    pub perm: [u8; 4],
    /// Bitmask of input negations.
    pub neg_in: u8,
    /// Whether the output was negated.
    pub neg_out: bool,
}

// ---------------------------------------------------------------------------
// Static lookup table
// ---------------------------------------------------------------------------

/// Precomputed NPN4 classification for all 65536 4-input truth tables.
/// Built on first access (~1ms). Thread-safe via LazyLock.
static NPN4_TABLE: LazyLock<Box<[Npn4Class; 65536]>> = LazyLock::new(build_npn4_table);

/// Classify a 4-input truth table into its NPN4 equivalence class.
/// O(1) — single array lookup.
pub fn npn4_classify(tt: u16) -> &'static Npn4Class {
    &NPN4_TABLE[tt as usize]
}

/// Return the number of distinct NPN4 classes (should be 222).
pub fn npn4_num_classes() -> usize {
    let table = &*NPN4_TABLE;
    let mut seen = [false; 256];
    for entry in table.iter() {
        seen[entry.class_id as usize] = true;
    }
    seen.iter().filter(|&&s| s).count()
}

// ---------------------------------------------------------------------------
// Truth table transforms
// ---------------------------------------------------------------------------

/// The 24 permutations of 4 elements.
const PERMS_4: [[u8; 4]; 24] = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
];

/// Permute the inputs of a 4-input truth table.
///
/// `perm[i]` specifies which original input feeds position `i` in the result.
/// Result bit `j` = original bit at index where input `i` gets value `(j >> i) & 1`
/// mapped through the permutation.
pub fn tt_permute(tt: u16, perm: [u8; 4]) -> u16 {
    let mut result = 0u16;
    for j in 0..16u32 {
        // For each output bit position j, compute the source bit position
        // by permuting the input variable assignments
        let mut src = 0u32;
        for i in 0..4 {
            let bit = (j >> i) & 1;
            src |= bit << perm[i];
        }
        if (tt >> src) & 1 != 0 {
            result |= 1 << j;
        }
    }
    result
}

/// Negate input `input` (0..3) of a 4-input truth table.
///
/// Swaps the value of f(..., x_i=0, ...) with f(..., x_i=1, ...) for each
/// combination of the other inputs.
pub fn tt_negate_input(tt: u16, input: u8) -> u16 {
    let mut result = 0u16;
    for j in 0..16u32 {
        let bit_val = (j >> input) & 1;
        let swapped = j ^ (1 << input);
        if bit_val == 0 {
            // Position j has input=0; take value from position with input=1
            if (tt >> swapped) & 1 != 0 {
                result |= 1 << j;
            }
        } else {
            // Position j has input=1; take value from position with input=0
            if (tt >> swapped) & 1 != 0 {
                result |= 1 << j;
            }
        }
    }
    result
}

/// Apply a full NPN transform: permute inputs, negate selected inputs, optionally negate output.
pub fn npn4_apply_transform(tt: u16, perm: [u8; 4], neg_in: u8, neg_out: bool) -> u16 {
    let mut result = tt_permute(tt, perm);
    for i in 0..4u8 {
        if (neg_in >> i) & 1 != 0 {
            result = tt_negate_input(result, i);
        }
    }
    if neg_out {
        result = !result;
    }
    result
}

/// Compute the inverse permutation of a 4-element permutation.
pub fn invert_perm(perm: [u8; 4]) -> [u8; 4] {
    let mut inv = [0u8; 4];
    for i in 0..4 {
        inv[perm[i] as usize] = i as u8;
    }
    inv
}

// ---------------------------------------------------------------------------
// Table construction
// ---------------------------------------------------------------------------

fn build_npn4_table() -> Box<[Npn4Class; 65536]> {
    let identity = Npn4Class {
        class_id: 0,
        canonical_tt: 0,
        perm: [0, 1, 2, 3],
        neg_in: 0,
        neg_out: false,
    };

    let table = vec![identity; 65536].into_boxed_slice();
    // Safety: Vec<T; 65536> -> Box<[T; 65536]> (known size)
    let table: Box<[Npn4Class; 65536]> = unsafe {
        let ptr = Box::into_raw(table) as *mut [Npn4Class; 65536];
        Box::from_raw(ptr)
    };
    let mut table = table;

    let mut classified = vec![false; 65536];
    let mut next_class_id: u8 = 0;

    for tt in 0..=0xFFFFu16 {
        if classified[tt as usize] {
            continue;
        }

        let canonical_tt = tt;
        let class_id = next_class_id;
        next_class_id += 1;

        // Try all 768 transforms, find which tt values belong to this class.
        // For each transformed tt, store the FORWARD transform (canonical → transformed)
        // so that `npn4_apply_transform(canonical, perm, neg_in, neg_out) = transformed`.
        for &perm in &PERMS_4 {
            for neg_in in 0..16u8 {
                for &neg_out in &[false, true] {
                    let transformed = npn4_apply_transform(tt, perm, neg_in, neg_out);
                    if !classified[transformed as usize] {
                        classified[transformed as usize] = true;

                        // Store the forward transform: canonical → transformed
                        table[transformed as usize] = Npn4Class {
                            class_id,
                            canonical_tt,
                            perm,
                            neg_in,
                            neg_out,
                        };
                    }
                }
            }
        }

        // The canonical tt itself: identity transform (canonical → canonical)
        table[tt as usize] = Npn4Class {
            class_id,
            canonical_tt,
            perm: [0, 1, 2, 3],
            neg_in: 0,
            neg_out: false,
        };
    }

    table
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npn4_table_has_222_classes() {
        assert_eq!(npn4_num_classes(), 222);
    }

    #[test]
    fn npn4_and_class() {
        let c = npn4_classify(0x8); // AND: f(a,b,c,d) = a & b (2-input embedded in 4)
        assert!(c.class_id < 222);
        // Stored transform maps canonical → 0x8
        let restored = npn4_apply_transform(c.canonical_tt, c.perm, c.neg_in, c.neg_out);
        assert_eq!(restored, 0x8);
    }

    #[test]
    fn npn4_xor_xnor_same_class() {
        let c_xor = npn4_classify(0x6); // XOR: a ^ b
        let c_xnor = npn4_classify(0x9); // XNOR: !(a ^ b) = 0xF ^ 0x6 = 0x9
        assert_eq!(
            c_xor.class_id, c_xnor.class_id,
            "XOR and XNOR should be NPN-equivalent (output negation)"
        );
    }

    #[test]
    fn npn4_const_classes() {
        let c0 = npn4_classify(0x0000); // CONST0
        let c1 = npn4_classify(0xFFFF); // CONST1
        // CONST0 and CONST1 are NPN-equivalent (output negation)
        assert_eq!(c0.class_id, c1.class_id);
    }

    #[test]
    fn npn4_transform_roundtrip() {
        // For every truth table, the stored transform maps canonical → tt
        for tt in 0..=0xFFFFu16 {
            let c = npn4_classify(tt);
            let restored = npn4_apply_transform(c.canonical_tt, c.perm, c.neg_in, c.neg_out);
            assert_eq!(
                restored, tt,
                "Round-trip failed for tt=0x{:04X}: canonical=0x{:04X}, restored=0x{:04X}",
                tt, c.canonical_tt, restored
            );
        }
    }

    #[test]
    fn npn4_all_entries_valid() {
        for tt in 0..=0xFFFFu16 {
            let c = npn4_classify(tt);
            assert!(
                c.class_id < 222,
                "class_id {} >= 222 for tt=0x{:04X}",
                c.class_id,
                tt
            );
            // Canonical should also map to the same class
            let cc = npn4_classify(c.canonical_tt);
            assert_eq!(
                c.class_id, cc.class_id,
                "tt=0x{:04X} class={} but canonical=0x{:04X} class={}",
                tt, c.class_id, c.canonical_tt, cc.class_id
            );
        }
    }

    #[test]
    fn npn4_complement_same_class() {
        // f and !f should always be in the same NPN class (output negation)
        for tt in 0..=0xFFFFu16 {
            let c = npn4_classify(tt);
            let c_neg = npn4_classify(!tt);
            assert_eq!(
                c.class_id, c_neg.class_id,
                "tt=0x{:04X} and !tt=0x{:04X} should be same NPN class",
                tt, !tt
            );
        }
    }

    #[test]
    fn npn4_permutation_same_class() {
        // Permuting inputs should give the same class
        let tt = 0x8u16; // AND: a & b
        let c_orig = npn4_classify(tt);

        // Swap inputs 0 and 1: f(a,b,c,d) → f(b,a,c,d)
        let tt_swapped = tt_permute(tt, [1, 0, 2, 3]);
        let c_swapped = npn4_classify(tt_swapped);
        assert_eq!(c_orig.class_id, c_swapped.class_id);

        // Try a more complex permutation on XOR
        let tt_xor = 0x6u16;
        let c_xor = npn4_classify(tt_xor);
        let tt_perm = tt_permute(tt_xor, [1, 0, 3, 2]);
        let c_perm = npn4_classify(tt_perm);
        assert_eq!(c_xor.class_id, c_perm.class_id);
    }
}
