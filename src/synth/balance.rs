//! AIG balancing — rebuild AND chains as balanced binary trees.
//!
//! Detects chains of AND nodes connected by non-complemented edges with fanout == 1,
//! collects their leaves, and rebuilds as a balanced binary tree to minimize depth.
//!
//! Example: `(((a&b)&c)&d)` depth=3 → `(a&b)&(c&d)` depth=2.
//!
//! **AND chains only** (v1). OR chains (`!(!a & !b)`) require polarity-aware
//! detection because OR is encoded as complemented AND in this AIG representation.
//! A node serving both as OR-input (complemented) and AND-input (non-complemented)
//! cannot be safely reshaped. OR-balancing deferred to a future version.

use super::aig::{Aig, AigLit};

/// Balance an AIG by rebuilding AND chains as balanced binary trees.
///
/// Returns a new AIG with the same function but potentially reduced depth.
/// The new AIG has structural hashing enabled, so any shared subexpressions
/// that were duplicated during chain collection are automatically merged.
pub fn balance(aig: &Aig) -> Aig {
    let num_inputs = aig.num_inputs();
    let num_nodes = aig.num_nodes();
    let ref_counts = aig.ref_counts();

    // Pre-compute which nodes are chain members (absorbed by a higher chain root).
    // A chain member should NOT be treated as its own chain root.
    let mut is_chain_member = vec![false; num_nodes];
    for (_node_idx, node) in aig.nodes_topo() {
        if is_chain_member_fanin(aig, node.fanin0, &ref_counts)
            && let Some(fi) = aig.var_to_node_idx(node.fanin0.var())
        {
            is_chain_member[fi] = true;
        }
        if is_chain_member_fanin(aig, node.fanin1, &ref_counts)
            && let Some(fi) = aig.var_to_node_idx(node.fanin1.var())
        {
            is_chain_member[fi] = true;
        }
    }

    // lit_map maps old variable IDs → new literals
    let total_vars = num_inputs as usize + 1 + num_nodes;
    let mut lit_map: Vec<AigLit> = vec![AigLit::FALSE; total_vars];

    // Map constant and inputs directly
    lit_map[0] = AigLit::FALSE;
    for var in 1..=num_inputs {
        lit_map[var as usize] = AigLit::new(var, false);
    }

    // Build new AIG with same inputs
    let mut new_aig = Aig::new();
    for port in aig.inputs() {
        if port.bits.len() == 1 {
            new_aig.add_input(&port.name);
        } else {
            new_aig.add_input_bus(&port.name, port.bits.len() as u32);
        }
    }

    // Process nodes in topological order
    for (node_idx, node) in aig.nodes_topo() {
        let var = aig.node_var(node_idx);

        // Skip chain members — they're absorbed by their chain root's balanced tree.
        // (Chain members have fanout == 1 and only feed their parent chain node.)
        if is_chain_member[node_idx] {
            // lit_map left as FALSE — won't be referenced (fanout == 1, only by chain root)
            continue;
        }

        // A chain root: has chain member fanins AND is not itself a chain member
        if is_and_chain_root(aig, node_idx, &ref_counts) {
            // Collect all leaves of the AND chain
            let mut leaves = Vec::new();
            collect_and_chain_leaves(aig, var, &ref_counts, &mut leaves);

            // Map leaves to new literals
            let mut new_leaves: Vec<AigLit> =
                leaves.iter().map(|&lit| remap_lit(lit, &lit_map)).collect();

            // Build balanced tree from leaves
            let result = build_balanced_and(&mut new_aig, &mut new_leaves);
            lit_map[var as usize] = result;
        } else {
            // Rebuild as-is in the new AIG
            let f0 = remap_lit(node.fanin0, &lit_map);
            let f1 = remap_lit(node.fanin1, &lit_map);
            let new_lit = new_aig.and(f0, f1);
            lit_map[var as usize] = new_lit;
        }
    }

    // Map outputs
    for port in aig.outputs() {
        if port.bits.len() == 1 {
            let new_lit = remap_lit(port.bits[0], &lit_map);
            new_aig.add_output(&port.name, new_lit);
        } else {
            let new_bits: Vec<AigLit> = port.bits.iter().map(|&l| remap_lit(l, &lit_map)).collect();
            new_aig.add_output_bus(&port.name, &new_bits);
        }
    }

    new_aig
}

/// Remap a literal through the lit_map, preserving complement edges.
fn remap_lit(lit: AigLit, lit_map: &[AigLit]) -> AigLit {
    if lit.is_const() {
        return if lit.is_complement() {
            AigLit::TRUE
        } else {
            AigLit::FALSE
        };
    }
    let base = lit_map[lit.var() as usize];
    if lit.is_complement() {
        base.negated()
    } else {
        base
    }
}

/// Check if a node is the root of an AND chain (has at least one chain member below).
fn is_and_chain_root(aig: &Aig, node_idx: usize, ref_counts: &[u32]) -> bool {
    let node = aig.node(node_idx);

    // A chain root has at least one chain member fanin.
    // The caller pre-filters: chain members are skipped before calling this.
    is_chain_member_fanin(aig, node.fanin0, ref_counts)
        || is_chain_member_fanin(aig, node.fanin1, ref_counts)
}

/// Check if a fanin literal points to a chain-eligible AND node.
fn is_chain_member_fanin(aig: &Aig, lit: AigLit, ref_counts: &[u32]) -> bool {
    if lit.is_complement() || lit.is_const() {
        return false;
    }
    let var = lit.var();
    if var <= aig.num_inputs() {
        return false;
    }
    // Must have fanout == 1 (only used by the parent)
    ref_counts[var as usize] == 1
}

/// Collect all leaves of an AND chain rooted at `root_var`.
///
/// A leaf is any literal that is NOT a chain member (complemented, shared, or PI).
/// Leaves are pushed in the order they're discovered (depth-first).
fn collect_and_chain_leaves(
    aig: &Aig,
    root_var: u32,
    ref_counts: &[u32],
    leaves: &mut Vec<AigLit>,
) {
    if let Some(node_idx) = aig.var_to_node_idx(root_var) {
        let node = aig.node(node_idx);
        collect_fanin_leaf(aig, node.fanin0, ref_counts, leaves);
        collect_fanin_leaf(aig, node.fanin1, ref_counts, leaves);
    }
}

/// Recursively collect a fanin: if it's a chain member, recurse; otherwise it's a leaf.
fn collect_fanin_leaf(aig: &Aig, lit: AigLit, ref_counts: &[u32], leaves: &mut Vec<AigLit>) {
    if is_chain_member_fanin(aig, lit, ref_counts) {
        // Recurse into chain member
        collect_and_chain_leaves(aig, lit.var(), ref_counts, leaves);
    } else {
        // This is a leaf (PI, complemented, shared, or constant)
        leaves.push(lit);
    }
}

/// Build a balanced AND tree from a set of leaves.
///
/// Uses a simple bottom-up approach: repeatedly combine pairs.
/// For n leaves, produces a tree of depth ceil(log2(n)).
fn build_balanced_and(aig: &mut Aig, leaves: &mut Vec<AigLit>) -> AigLit {
    if leaves.is_empty() {
        return AigLit::TRUE; // empty AND = TRUE
    }
    if leaves.len() == 1 {
        return leaves[0];
    }

    // Bottom-up: combine pairs until one remains
    while leaves.len() > 1 {
        let mut next = Vec::with_capacity(leaves.len().div_ceil(2));
        let mut i = 0;
        while i + 1 < leaves.len() {
            let combined = aig.and(leaves[i], leaves[i + 1]);
            next.push(combined);
            i += 2;
        }
        if i < leaves.len() {
            // Odd element: carry forward
            next.push(leaves[i]);
        }
        *leaves = next;
    }

    leaves[0]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::mapping::evaluate_aig;

    #[test]
    fn balance_and_chain_reduces_depth() {
        // Build a left-skewed chain: (((a & b) & c) & d) — depth 3
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let c = aig.add_input("c");
        let d = aig.add_input("d");
        let ab = aig.and(a, b);
        let abc = aig.and(ab, c);
        let abcd = aig.and(abc, d);
        aig.add_output("y", abcd);

        assert_eq!(aig.depth(), 3);

        let balanced = balance(&aig);

        // Balanced: (a&b) & (c&d) — depth 2
        assert!(
            balanced.depth() <= 2,
            "balanced depth {} should be <= 2",
            balanced.depth()
        );
    }

    #[test]
    fn balance_already_balanced() {
        // Already balanced: (a&b) & (c&d) — depth 2
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let c = aig.add_input("c");
        let d = aig.add_input("d");
        let ab = aig.and(a, b);
        let cd = aig.and(c, d);
        let _y = aig.and(ab, cd);
        aig.add_output("y", _y);

        let balanced = balance(&aig);
        assert_eq!(balanced.depth(), aig.depth());
    }

    #[test]
    fn balance_preserves_equiv_adder() {
        // 4-bit ripple-carry adder — verify balancing preserves function
        let mut aig = Aig::new();
        let a = aig.add_input_bus("a", 4);
        let b = aig.add_input_bus("b", 4);
        let mut carry = AigLit::FALSE;
        let mut sum_bits = Vec::new();

        for i in 0..4 {
            let axb = aig.xor(a[i], b[i]);
            let sum = aig.xor(axb, carry);
            let ab = aig.and(a[i], b[i]);
            let caxb = aig.and(carry, axb);
            carry = aig.or(ab, caxb);
            sum_bits.push(sum);
        }
        sum_bits.push(carry);
        aig.add_output_bus("sum", &sum_bits);

        let balanced = balance(&aig);

        // Exhaustive equivalence check (8 inputs = 256 combinations)
        for assignment in 0..256u32 {
            let inputs: Vec<bool> = (0..8).map(|i| (assignment >> i) & 1 != 0).collect();
            let orig = evaluate_aig(&aig, &inputs);
            let bal = evaluate_aig(&balanced, &inputs);
            assert_eq!(
                orig, bal,
                "Mismatch at input {:08b}: orig={:?} bal={:?}",
                assignment, orig, bal
            );
        }
    }

    #[test]
    fn balance_single_node_unchanged() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);

        let balanced = balance(&aig);
        assert_eq!(balanced.num_nodes(), 1);
        assert_eq!(balanced.depth(), 1);
    }
}
