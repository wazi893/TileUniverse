//! K=4 cut enumeration for technology mapping.
//!
//! A "cut" of a node is a set of at most K leaves in the node's transitive
//! fanin cone such that every path from a primary input to the node passes
//! through at least one leaf. The truth table of the cut encodes the Boolean
//! function from leaves to the cut root.
//!
//! We use K=4 exclusively. Truth tables are `u16` (2^4 = 16 bits).
//! Cut structs use fixed-size arrays (no heap allocation per cut).

use super::aig::{Aig, AigLit};

// ---------------------------------------------------------------------------
// Cut
// ---------------------------------------------------------------------------

/// Maximum cut size (K).
pub const MAX_K: usize = 4;

/// Maximum cuts stored per node.
pub const MAX_CUTS_PER_NODE: usize = 8;

/// A K-cut: a set of leaf variables and the truth table from leaves to root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cut {
    /// Leaf variable IDs, sorted ascending. Unused slots are 0.
    pub leaves: [u32; MAX_K],
    /// Number of active leaves (1..=4).
    pub num_leaves: u8,
    /// Truth table: bit `i` = f(i_0, i_1, ..., i_{n-1}) where `i_j = (i >> j) & 1`.
    /// Only the low `2^num_leaves` bits are meaningful.
    pub truth_table: u16,
}

impl Cut {
    /// Create a trivial (single-leaf) cut for a variable.
    fn trivial(var: u32) -> Self {
        Cut {
            leaves: [var, 0, 0, 0],
            num_leaves: 1,
            truth_table: 0b10, // Identity: f(0)=0, f(1)=1
        }
    }

    /// Get the active leaf slice.
    pub fn leaf_slice(&self) -> &[u32] {
        &self.leaves[..self.num_leaves as usize]
    }

    /// Number of meaningful truth table bits.
    pub fn tt_bits(&self) -> u32 {
        1 << self.num_leaves
    }

    /// Mask for meaningful truth table bits.
    pub fn tt_mask(&self) -> u16 {
        ((1u32 << self.tt_bits()) - 1) as u16
    }
}

// ---------------------------------------------------------------------------
// Per-node cut set
// ---------------------------------------------------------------------------

/// All cuts for a single node (bounded by MAX_CUTS_PER_NODE).
#[derive(Clone, Debug)]
pub struct NodeCuts {
    pub cuts: Vec<Cut>,
}

impl NodeCuts {
    fn new() -> Self {
        NodeCuts {
            cuts: Vec::with_capacity(MAX_CUTS_PER_NODE),
        }
    }

    /// Add a cut at the default budget. Retained as a convenience for tests;
    /// enumeration uses [`Self::add_with_limit`] with the configured budget.
    #[allow(dead_code)]
    fn add(&mut self, cut: Cut) {
        self.add_with_limit(cut, MAX_CUTS_PER_NODE);
    }

    fn add_with_limit(&mut self, cut: Cut, limit: usize) {
        let new_leaves = cut.leaf_slice();

        // Dedup: skip if an existing cut has identical leaves
        if self.cuts.iter().any(|c| c.leaf_slice() == new_leaves) {
            return;
        }

        // Dominance: skip if an existing cut has a strict subset of leaves
        // (fewer leaves covering the same cone = strictly better)
        if self.cuts.iter().any(|c| {
            c.num_leaves < cut.num_leaves && c.leaf_slice().iter().all(|l| new_leaves.contains(l))
        }) {
            return;
        }

        if self.cuts.len() < limit {
            self.cuts.push(cut);
        }
    }
}

// ---------------------------------------------------------------------------
// Cut enumeration
// ---------------------------------------------------------------------------

/// Enumerate K=4 cuts for every AND node in the AIG.
///
/// Returns a Vec indexed by AND node index (0-based).
/// Primary inputs have implicit trivial cuts (not stored here).
///
/// Algorithm:
/// 1. For each primary input, the trivial cut is {self}.
/// 2. For each AND node in topological order:
///    a. Cross-product of fanin0 cuts × fanin1 cuts.
///    b. Merge leaf sets. Skip if merged > K.
///    c. Compute truth table by composing fanin truth tables.
///    d. Also add the trivial cut {self}.
pub fn enumerate_cuts(aig: &Aig) -> Vec<NodeCuts> {
    enumerate_cuts_with_limit(aig, MAX_CUTS_PER_NODE)
}

/// Enumerate K=4 cuts with a configurable per-node cut budget.
///
/// `enumerate_cuts` uses the default [`MAX_CUTS_PER_NODE`]; via-aware mapping
/// passes a larger limit so deeply-expanded threshold cuts (e.g. the `{a,b,cin}`
/// majority cut of an adder carry) survive eviction and can be folded into a
/// single computational via. A larger limit never changes correctness — only
/// which equivalent mapping the delay-oriented DP can choose.
pub fn enumerate_cuts_with_limit(aig: &Aig, limit: usize) -> Vec<NodeCuts> {
    let limit = limit.max(1);
    let num_inputs = aig.num_inputs();
    let num_nodes = aig.num_nodes();

    // We need to look up cuts for any variable (input or node).
    // Inputs have a single trivial cut each; nodes have computed cuts.
    let mut node_cuts: Vec<NodeCuts> = Vec::with_capacity(num_nodes);

    for (node_idx, node) in aig.nodes_topo() {
        let mut cuts = NodeCuts::new();

        // Guarantee trivial cut {self} in slot 0 — always present, never evicted.
        let self_var = aig.node_var(node_idx);
        cuts.cuts.push(Cut::trivial(self_var));

        // Get cuts for each fanin
        let cuts0 = fanin_cuts(aig, node.fanin0, num_inputs, &node_cuts);
        let cuts1 = fanin_cuts(aig, node.fanin1, num_inputs, &node_cuts);

        let inv0 = node.fanin0.is_complement();
        let inv1 = node.fanin1.is_complement();

        // Cross-product merge
        for c0 in &cuts0 {
            let tt0 = if inv0 {
                (!c0.truth_table) & c0.tt_mask()
            } else {
                c0.truth_table
            };

            for c1 in &cuts1 {
                let tt1 = if inv1 {
                    (!c1.truth_table) & c1.tt_mask()
                } else {
                    c1.truth_table
                };

                // Try to merge leaf sets
                if let Some(merged) = merge_cuts(c0, c1, tt0, tt1) {
                    cuts.add_with_limit(merged, limit);
                    if cuts.cuts.len() >= limit {
                        break;
                    }
                }
            }
            if cuts.cuts.len() >= limit {
                break;
            }
        }

        node_cuts.push(cuts);
    }

    node_cuts
}

/// Get cuts for a literal's variable. Inputs get a single trivial cut.
fn fanin_cuts(_aig: &Aig, lit: AigLit, num_inputs: u32, node_cuts: &[NodeCuts]) -> Vec<Cut> {
    let var = lit.var();
    if var == 0 {
        // Constant: trivial cut (will be handled by truth table = all-0 or all-1)
        vec![Cut {
            leaves: [0, 0, 0, 0],
            num_leaves: 0,
            truth_table: 0x0, // Constant false
        }]
    } else if var <= num_inputs {
        // Primary input: single trivial cut
        vec![Cut::trivial(var)]
    } else {
        // AND node: use computed cuts
        let idx = (var - num_inputs - 1) as usize;
        if idx < node_cuts.len() {
            node_cuts[idx].cuts.clone()
        } else {
            vec![Cut::trivial(var)]
        }
    }
}

/// Merge two cuts, producing an AND of their truth tables.
/// Returns None if the merged leaf count exceeds MAX_K.
fn merge_cuts(c0: &Cut, c1: &Cut, tt0: u16, tt1: u16) -> Option<Cut> {
    // Handle constant cuts (0 leaves)
    if c0.num_leaves == 0 {
        // c0 is constant; the AND result depends on the constant value
        // tt0 is either 0x0 (false) or 0x1 (true, after inversion)
        if tt0 == 0 {
            return Some(Cut {
                leaves: [0; 4],
                num_leaves: 0,
                truth_table: 0,
            });
        } else {
            return Some(c1.clone_with_tt(tt1));
        }
    }
    if c1.num_leaves == 0 {
        if tt1 == 0 {
            return Some(Cut {
                leaves: [0; 4],
                num_leaves: 0,
                truth_table: 0,
            });
        } else {
            return Some(c0.clone_with_tt(tt0));
        }
    }

    // Merge sorted leaf arrays
    let s0 = c0.leaf_slice();
    let s1 = c1.leaf_slice();
    let mut merged = [0u32; MAX_K];
    let mut n = 0usize;
    let mut i = 0usize;
    let mut j = 0usize;

    while i < s0.len() && j < s1.len() {
        if n >= MAX_K {
            // Check if we can still fit
            if i < s0.len() || j < s1.len() {
                // Remaining elements won't fit
            }
        }
        if s0[i] == s1[j] {
            if n >= MAX_K {
                return None;
            }
            merged[n] = s0[i];
            n += 1;
            i += 1;
            j += 1;
        } else if s0[i] < s1[j] {
            if n >= MAX_K {
                return None;
            }
            merged[n] = s0[i];
            n += 1;
            i += 1;
        } else {
            if n >= MAX_K {
                return None;
            }
            merged[n] = s1[j];
            n += 1;
            j += 1;
        }
    }
    while i < s0.len() {
        if n >= MAX_K {
            return None;
        }
        merged[n] = s0[i];
        n += 1;
        i += 1;
    }
    while j < s1.len() {
        if n >= MAX_K {
            return None;
        }
        merged[n] = s1[j];
        n += 1;
        j += 1;
    }

    // Compute the composed truth table.
    // For each of the 2^n input assignments to the merged leaves,
    // look up the corresponding bit in tt0 and tt1, AND them.
    let num_bits = 1u32 << n;
    let mut tt = 0u16;

    for assignment in 0..num_bits {
        // Map assignment bits to c0's leaf positions
        let bit0 = project_truth_table_bit(assignment, &merged[..n], s0, tt0, c0.num_leaves);
        let bit1 = project_truth_table_bit(assignment, &merged[..n], s1, tt1, c1.num_leaves);
        if bit0 & bit1 != 0 {
            tt |= 1 << assignment;
        }
    }

    Some(Cut {
        leaves: merged,
        num_leaves: n as u8,
        truth_table: tt,
    })
}

/// Given an assignment to merged leaves, project to a sub-cut's leaf ordering
/// and look up the truth table bit.
fn project_truth_table_bit(
    assignment: u32,
    merged_leaves: &[u32],
    sub_leaves: &[u32],
    sub_tt: u16,
    _sub_num_leaves: u8,
) -> u32 {
    let mut sub_assignment = 0u32;
    for (sub_pos, &sub_leaf) in sub_leaves.iter().enumerate() {
        // Find this leaf in the merged array
        if let Some(merged_pos) = merged_leaves.iter().position(|&m| m == sub_leaf) {
            let bit = (assignment >> merged_pos) & 1;
            sub_assignment |= bit << sub_pos;
        }
    }
    (sub_tt as u32 >> sub_assignment) & 1
}

impl Cut {
    fn clone_with_tt(&self, tt: u16) -> Cut {
        Cut {
            leaves: self.leaves,
            num_leaves: self.num_leaves,
            truth_table: tt,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::aig::Aig;

    #[test]
    fn cuts_single_and() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);

        let all_cuts = enumerate_cuts(&aig);
        assert_eq!(all_cuts.len(), 1); // 1 AND node

        let node_cuts = &all_cuts[0];
        assert!(!node_cuts.cuts.is_empty());

        // Should have a 2-leaf cut {a, b} with AND truth table
        let two_leaf = node_cuts.cuts.iter().find(|c| c.num_leaves == 2);
        assert!(two_leaf.is_some(), "expected 2-leaf cut");
        let cut = two_leaf.unwrap();
        assert_eq!(cut.leaves[0], 1); // var for 'a'
        assert_eq!(cut.leaves[1], 2); // var for 'b'
        assert_eq!(cut.truth_table, 0x8); // AND2
    }

    #[test]
    fn cuts_or() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.or(a, b);
        aig.add_output("y", y);

        let all_cuts = enumerate_cuts(&aig);
        // OR creates 1 AND node: !(!a & !b)
        assert_eq!(all_cuts.len(), 1);

        // The AND node computes !a & !b. Its 2-leaf cut should be NOR (0x1).
        // But the output literal is complemented — the mapping stage handles that.
        let cut = all_cuts[0]
            .cuts
            .iter()
            .find(|c| c.num_leaves == 2)
            .expect("2-leaf cut");
        // !a & !b: both inputs negated AND → truth table:
        // f(a,b) = !a & !b → f(0,0)=1, f(1,0)=0, f(0,1)=0, f(1,1)=0 = 0x1
        assert_eq!(cut.truth_table, 0x1);
    }

    #[test]
    fn cuts_chain_4_inputs() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let c = aig.add_input("c");
        let d = aig.add_input("d");
        let ab = aig.and(a, b);
        let cd = aig.and(c, d);
        let y = aig.and(ab, cd);
        aig.add_output("y", y);

        let all_cuts = enumerate_cuts(&aig);
        assert_eq!(all_cuts.len(), 3); // 3 AND nodes

        // Root node should have a 4-leaf cut {a, b, c, d}
        let root_cuts = &all_cuts[2];
        let four_leaf = root_cuts.cuts.iter().find(|c| c.num_leaves == 4);
        assert!(four_leaf.is_some(), "expected 4-leaf cut at root");
        let cut = four_leaf.unwrap();
        assert_eq!(cut.leaf_slice(), &[1, 2, 3, 4]);
        // 4-input AND: only bit 15 set (all inputs = 1)
        assert_eq!(cut.truth_table, 1 << 15);
    }

    #[test]
    fn cuts_bounded() {
        // Build a wider circuit to test the per-node cut bound
        let mut aig = Aig::new();
        let inputs: Vec<_> = (0..8).map(|i| aig.add_input(&format!("x{}", i))).collect();
        let mut prev = aig.and(inputs[0], inputs[1]);
        for &inp in &inputs[2..] {
            prev = aig.and(prev, inp);
        }
        aig.add_output("y", prev);

        let all_cuts = enumerate_cuts(&aig);
        // Every node should have at most MAX_CUTS_PER_NODE cuts
        for nc in &all_cuts {
            assert!(nc.cuts.len() <= MAX_CUTS_PER_NODE);
        }
    }

    #[test]
    fn cuts_xor() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.xor(a, b);
        aig.add_output("y", y);

        let all_cuts = enumerate_cuts(&aig);
        // XOR creates 3 AND nodes
        assert_eq!(all_cuts.len(), 3);
    }

    #[test]
    fn cuts_trivial_always_present() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);

        let all_cuts = enumerate_cuts(&aig);
        // The trivial cut {self} should always be present
        let node_var = aig.node_var(0);
        let trivial = all_cuts[0]
            .cuts
            .iter()
            .find(|c| c.num_leaves == 1 && c.leaves[0] == node_var);
        assert!(trivial.is_some(), "trivial cut should always be present");
    }

    #[test]
    fn trivial_cut_always_slot_zero() {
        // The trivial cut must be in slot 0, never evicted by cross-product merging
        let mut aig = Aig::new();
        let inputs: Vec<_> = (0..8).map(|i| aig.add_input(&format!("x{}", i))).collect();
        let mut prev = aig.and(inputs[0], inputs[1]);
        for &inp in &inputs[2..] {
            prev = aig.and(prev, inp);
        }
        aig.add_output("y", prev);

        let all_cuts = enumerate_cuts(&aig);
        for (node_idx, nc) in all_cuts.iter().enumerate() {
            let self_var = aig.node_var(node_idx);
            assert_eq!(
                nc.cuts[0].num_leaves, 1,
                "slot 0 should be trivial cut for node {}",
                node_idx
            );
            assert_eq!(
                nc.cuts[0].leaves[0], self_var,
                "trivial cut leaf should be self var for node {}",
                node_idx
            );
        }
    }

    #[test]
    fn dedup_identical_leaves() {
        // Two cuts with identical leaves should not be duplicated
        let mut nc = NodeCuts::new();
        nc.cuts.push(Cut::trivial(99)); // slot 0 reserved
        let c1 = Cut {
            leaves: [1, 2, 0, 0],
            num_leaves: 2,
            truth_table: 0x8,
        };
        let c2 = Cut {
            leaves: [1, 2, 0, 0],
            num_leaves: 2,
            truth_table: 0x6,
        };
        nc.add(c1);
        nc.add(c2); // same leaves, different tt — should be deduped
        assert_eq!(nc.cuts.len(), 2, "identical-leaf cut should be deduped");
    }

    #[test]
    fn dominance_prunes_supersets() {
        // A 2-leaf cut should block a dominated 3-leaf cut that contains those leaves
        let mut nc = NodeCuts::new();
        nc.cuts.push(Cut::trivial(99)); // slot 0
        let small = Cut {
            leaves: [1, 2, 0, 0],
            num_leaves: 2,
            truth_table: 0x8,
        };
        let big = Cut {
            leaves: [1, 2, 3, 0],
            num_leaves: 3,
            truth_table: 0x80,
        };
        nc.add(small);
        nc.add(big); // dominated by small — should be pruned
        assert_eq!(nc.cuts.len(), 2, "dominated cut should be pruned");
    }
}
