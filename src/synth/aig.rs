//! And-Inverter Graph (AIG) — the core IR for logic synthesis.
//!
//! Every node is a 2-input AND gate. Inversion is encoded in edge attributes
//! (LSB of the literal), not as separate nodes. This is the standard AIGER
//! convention used by ABC, mockturtle, and the broader EDA community.
//!
//! Structural hashing ensures that logically equivalent subexpressions share
//! a single node. The `nodes` Vec is always in topological order by construction.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AigLit — complement-encoded literal
// ---------------------------------------------------------------------------

/// A literal in the AIG. The LSB encodes complementation.
///
/// - `lit >> 1` = variable ID
/// - `lit & 1`  = complement flag (1 = inverted)
///
/// Variable 0 is the constant FALSE.
/// Literal 0 = constant FALSE, literal 1 = constant TRUE.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AigLit(u32);

impl AigLit {
    pub const FALSE: AigLit = AigLit(0);
    pub const TRUE: AigLit = AigLit(1);

    #[inline]
    pub const fn new(var: u32, complement: bool) -> Self {
        AigLit((var << 1) | complement as u32)
    }

    #[inline]
    pub const fn var(self) -> u32 {
        self.0 >> 1
    }

    #[inline]
    pub const fn is_complement(self) -> bool {
        self.0 & 1 != 0
    }

    #[inline]
    pub const fn negated(self) -> Self {
        AigLit(self.0 ^ 1)
    }

    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn from_raw(v: u32) -> Self {
        AigLit(v)
    }

    #[inline]
    pub const fn is_const(self) -> bool {
        self.var() == 0
    }
}

impl std::fmt::Debug for AigLit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_const() {
            write!(
                f,
                "{}",
                if self.is_complement() {
                    "TRUE"
                } else {
                    "FALSE"
                }
            )
        } else if self.is_complement() {
            write!(f, "!v{}", self.var())
        } else {
            write!(f, "v{}", self.var())
        }
    }
}

impl std::fmt::Display for AigLit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

// ---------------------------------------------------------------------------
// AigNode — 8-byte AND gate
// ---------------------------------------------------------------------------

/// An AND gate node. Output = fanin0 AND fanin1.
/// Complement edges are encoded in the LSB of each literal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AigNode {
    pub fanin0: AigLit,
    pub fanin1: AigLit,
}

// ---------------------------------------------------------------------------
// AigPort — named I/O
// ---------------------------------------------------------------------------

/// A named primary input or output port.
#[derive(Clone, Debug)]
pub struct AigPort {
    pub name: String,
    /// Each bit of the port is a literal.
    /// Single-bit ports have exactly one element.
    pub bits: Vec<AigLit>,
}

// ---------------------------------------------------------------------------
// AigStats
// ---------------------------------------------------------------------------

/// Summary statistics for an AIG.
#[derive(Clone, Debug)]
pub struct AigStats {
    pub num_inputs: u32,
    pub num_outputs: usize,
    pub num_and_nodes: usize,
    pub depth: u32,
}

impl std::fmt::Display for AigStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AIG: i/o = {}/{}, and = {}, depth = {}",
            self.num_inputs, self.num_outputs, self.num_and_nodes, self.depth
        )
    }
}

// ---------------------------------------------------------------------------
// Aig — the graph
// ---------------------------------------------------------------------------

/// And-Inverter Graph.
///
/// # Variable numbering
///
/// - Variable 0: constant FALSE
/// - Variables `1..=num_inputs`: primary inputs
/// - Variable `num_inputs + 1 + i`: AND node at index `i`
///
/// # Invariants
///
/// - `nodes` is always in topological order (guaranteed by bottom-up construction).
/// - `strash` is only used for point lookups, never iterated (determinism).
pub struct Aig {
    num_inputs: u32,
    nodes: Vec<AigNode>,
    inputs: Vec<AigPort>,
    outputs: Vec<AigPort>,
    strash: HashMap<(u32, u32), u32>,
}

impl Aig {
    /// Create an empty AIG with no inputs, outputs, or nodes.
    pub fn new() -> Self {
        Aig {
            num_inputs: 0,
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            strash: HashMap::new(),
        }
    }

    // -- Primary inputs --

    /// Add a single-bit primary input. Returns its positive literal.
    pub fn add_input(&mut self, name: &str) -> AigLit {
        self.num_inputs += 1;
        let lit = AigLit::new(self.num_inputs, false);
        self.inputs.push(AigPort {
            name: name.to_string(),
            bits: vec![lit],
        });
        lit
    }

    /// Add a multi-bit input bus. Returns literals for each bit (LSB first).
    pub fn add_input_bus(&mut self, name: &str, width: u32) -> Vec<AigLit> {
        let mut bits = Vec::with_capacity(width as usize);
        for _ in 0..width {
            self.num_inputs += 1;
            bits.push(AigLit::new(self.num_inputs, false));
        }
        self.inputs.push(AigPort {
            name: name.to_string(),
            bits: bits.clone(),
        });
        bits
    }

    // -- Primary outputs --

    /// Add a single-bit primary output.
    pub fn add_output(&mut self, name: &str, lit: AigLit) {
        self.outputs.push(AigPort {
            name: name.to_string(),
            bits: vec![lit],
        });
    }

    /// Add a multi-bit output bus.
    pub fn add_output_bus(&mut self, name: &str, bits: &[AigLit]) {
        self.outputs.push(AigPort {
            name: name.to_string(),
            bits: bits.to_vec(),
        });
    }

    // -- Gate construction with structural hashing --

    /// Create an AND node with structural hashing.
    ///
    /// Performs constant folding and trivial simplifications:
    /// - `a & 0 = 0`
    /// - `a & 1 = a`
    /// - `a & a = a`
    /// - `a & !a = 0`
    pub fn and(&mut self, a: AigLit, b: AigLit) -> AigLit {
        // Constant folding
        if a == AigLit::FALSE || b == AigLit::FALSE {
            return AigLit::FALSE;
        }
        if a == AigLit::TRUE {
            return b;
        }
        if b == AigLit::TRUE {
            return a;
        }

        // Trivial simplification
        if a == b {
            return a;
        }
        if a == b.negated() {
            return AigLit::FALSE;
        }

        // Canonical ordering for structural hashing
        let (f0, f1) = if a.raw() <= b.raw() { (a, b) } else { (b, a) };
        let key = (f0.raw(), f1.raw());

        // Check strash
        if let Some(&node_idx) = self.strash.get(&key) {
            return self.node_lit(node_idx as usize);
        }

        // Create new node
        let node_idx = self.nodes.len();
        self.nodes.push(AigNode {
            fanin0: f0,
            fanin1: f1,
        });
        self.strash.insert(key, node_idx as u32);

        self.node_lit(node_idx)
    }

    /// NOT = complement the literal. No new node created.
    #[inline]
    pub fn not(&self, a: AigLit) -> AigLit {
        a.negated()
    }

    /// OR via DeMorgan: `a | b = !(!a & !b)`.
    pub fn or(&mut self, a: AigLit, b: AigLit) -> AigLit {
        self.and(a.negated(), b.negated()).negated()
    }

    /// XOR: `a ^ b = (a & !b) | (!a & b)`.
    pub fn xor(&mut self, a: AigLit, b: AigLit) -> AigLit {
        let t1 = self.and(a, b.negated());
        let t2 = self.and(a.negated(), b);
        self.or(t1, t2)
    }

    /// MUX: `sel ? a : b = (sel & a) | (!sel & b)`.
    pub fn mux(&mut self, sel: AigLit, a: AigLit, b: AigLit) -> AigLit {
        let t1 = self.and(sel, a);
        let t2 = self.and(sel.negated(), b);
        self.or(t1, t2)
    }

    // -- Accessors --

    /// Build an AIG from a truth table via Shannon decomposition.
    ///
    /// `num_inputs` must be 1..=6 (truth table up to 64 bits).
    /// Produces a single-output AIG named "f".
    pub fn from_truth_table(tt: u64, num_inputs: u32) -> Aig {
        assert!((1..=6).contains(&num_inputs));
        let mut aig = Aig::new();
        let inputs: Vec<AigLit> = (0..num_inputs)
            .map(|i| aig.add_input(&format!("x{}", i)))
            .collect();
        let out = aig.build_tt_recursive(tt, &inputs, num_inputs);
        aig.add_output("f", out);
        aig
    }

    /// Build a multi-output AIG from a list of truth tables sharing the same inputs.
    ///
    /// Each `(tt, num_inputs)` pair defines one output. All must have the same `num_inputs`.
    /// Uses the correct LSB-first cofactor decomposition (Sprint 213).
    pub fn from_truth_tables(truth_tables: &[(u64, u32)]) -> Aig {
        assert!(!truth_tables.is_empty());
        let num_inputs = truth_tables[0].1;
        assert!((1..=6).contains(&num_inputs));
        for &(_, n) in truth_tables {
            assert_eq!(n, num_inputs, "all truth tables must have same num_inputs");
        }

        let mut aig = Aig::new();
        let inputs: Vec<AigLit> = (0..num_inputs)
            .map(|i| aig.add_input(&format!("x{}", i)))
            .collect();

        for (out_idx, &(tt, _)) in truth_tables.iter().enumerate() {
            let lit = aig.build_tt_recursive(tt, &inputs, num_inputs);
            aig.add_output(&format!("y{}", out_idx), lit);
        }
        aig
    }

    /// Compute the cofactor of a truth table w.r.t. variable `var`.
    ///
    /// Truth table convention: bit index = x0 + 2*x1 + 4*x2 + ...
    /// Cofactor w.r.t. x_var=val extracts bits where x_var==val and
    /// compresses them into a truth table over the remaining variables.
    fn cofactor(tt: u64, num_vars: u32, var: u32, val: bool) -> u64 {
        let num_entries = 1u32 << num_vars;
        let half_entries = 1u32 << (num_vars - 1);
        let mut result = 0u64;
        let mut dst = 0u32;
        for src in 0..num_entries {
            let bit_var = (src >> var) & 1;
            if (bit_var != 0) == val {
                if (tt >> src) & 1 != 0 {
                    result |= 1u64 << dst;
                }
                dst += 1;
            }
        }
        debug_assert_eq!(dst, half_entries);
        result
    }

    fn build_tt_recursive(&mut self, tt: u64, inputs: &[AigLit], num_vars: u32) -> AigLit {
        // Base cases: constant 0 or constant 1
        let mask = if num_vars >= 6 {
            u64::MAX
        } else {
            (1u64 << (1u32 << num_vars)) - 1
        };
        let relevant = tt & mask;
        if relevant == 0 {
            return AigLit::FALSE;
        }
        if relevant == mask {
            return AigLit::TRUE;
        }

        // Single variable — find which one
        if num_vars == 1 {
            let bit0 = tt & 1;
            let bit1 = (tt >> 1) & 1;
            // The single remaining variable — find it
            let x = inputs[0];
            return match (bit0, bit1) {
                (0, 0) => AigLit::FALSE,
                (1, 1) => AigLit::TRUE,
                (0, 1) => x,
                (1, 0) => self.not(x),
                _ => unreachable!(),
            };
        }

        // Pick the first variable to split on (x0, which is inputs[0])
        let tt_low = Self::cofactor(tt, num_vars, 0, false);
        let tt_high = Self::cofactor(tt, num_vars, 0, true);

        let f_low = self.build_tt_recursive(tt_low, &inputs[1..], num_vars - 1);
        let f_high = self.build_tt_recursive(tt_high, &inputs[1..], num_vars - 1);

        if f_low == f_high {
            return f_low;
        }

        let x = inputs[0];
        let branch_hi = self.and(x, f_high);
        let branch_lo = self.and(self.not(x), f_low);
        self.or(branch_lo, branch_hi)
    }

    /// Number of primary inputs.
    #[inline]
    pub fn num_inputs(&self) -> u32 {
        self.num_inputs
    }

    /// Number of AND nodes.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of output bits across all output ports.
    pub fn num_output_bits(&self) -> usize {
        self.outputs.iter().map(|p| p.bits.len()).sum()
    }

    /// Input ports.
    pub fn inputs(&self) -> &[AigPort] {
        &self.inputs
    }

    /// Output ports.
    pub fn outputs(&self) -> &[AigPort] {
        &self.outputs
    }

    /// Get the AND node at a given index.
    #[inline]
    pub fn node(&self, idx: usize) -> &AigNode {
        &self.nodes[idx]
    }

    /// All AND nodes (in topological order).
    pub fn nodes(&self) -> &[AigNode] {
        &self.nodes
    }

    /// Variable ID for the AND node at the given index.
    #[inline]
    pub fn node_var(&self, node_idx: usize) -> u32 {
        self.num_inputs + 1 + node_idx as u32
    }

    /// Positive literal for the AND node at the given index.
    #[inline]
    pub fn node_lit(&self, node_idx: usize) -> AigLit {
        AigLit::new(self.node_var(node_idx), false)
    }

    /// Convert a variable ID to a node index (None if it's an input or constant).
    #[inline]
    pub fn var_to_node_idx(&self, var: u32) -> Option<usize> {
        if var > self.num_inputs {
            let idx = (var - self.num_inputs - 1) as usize;
            if idx < self.nodes.len() {
                Some(idx)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Iterate AND nodes in topological order with their indices.
    pub fn nodes_topo(&self) -> impl Iterator<Item = (usize, &AigNode)> {
        self.nodes.iter().enumerate()
    }

    /// Iterate all output literals (flat, across all output ports).
    pub fn output_lits(&self) -> impl Iterator<Item = AigLit> + '_ {
        self.outputs.iter().flat_map(|p| p.bits.iter().copied())
    }

    // -- Analysis --

    /// Compute topological levels (depth) for all nodes.
    /// Level of an input = 0. Level of AND node = max(level(fanin0), level(fanin1)) + 1.
    /// Returns a Vec indexed by node index.
    pub fn compute_levels(&self) -> Vec<u32> {
        let mut levels = vec![0u32; self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            let l0 = self.lit_level(&levels, node.fanin0);
            let l1 = self.lit_level(&levels, node.fanin1);
            levels[i] = l0.max(l1) + 1;
        }
        levels
    }

    /// Helper: get the level of a literal (0 for inputs/constants).
    fn lit_level(&self, levels: &[u32], lit: AigLit) -> u32 {
        if lit.is_const() || lit.var() <= self.num_inputs {
            0
        } else {
            let idx = (lit.var() - self.num_inputs - 1) as usize;
            levels[idx]
        }
    }

    /// AIG depth = max level across all output literals.
    pub fn depth(&self) -> u32 {
        let levels = self.compute_levels();
        self.output_lits()
            .map(|lit| {
                if lit.is_const() || lit.var() <= self.num_inputs {
                    0
                } else {
                    let idx = (lit.var() - self.num_inputs - 1) as usize;
                    levels[idx]
                }
            })
            .max()
            .unwrap_or(0)
    }

    /// Compute reference counts for each variable.
    /// Index 0..=num_inputs are inputs (includes constant).
    /// Index num_inputs+1+i are AND nodes.
    pub fn ref_counts(&self) -> Vec<u32> {
        let total_vars = self.num_inputs as usize + 1 + self.nodes.len();
        let mut refs = vec![0u32; total_vars];

        // Count refs from AND node fanins
        for node in &self.nodes {
            refs[node.fanin0.var() as usize] += 1;
            refs[node.fanin1.var() as usize] += 1;
        }

        // Count refs from outputs
        for lit in self.output_lits() {
            if !lit.is_const() {
                refs[lit.var() as usize] += 1;
            }
        }

        refs
    }

    /// Summary statistics.
    pub fn stats(&self) -> AigStats {
        AigStats {
            num_inputs: self.num_inputs,
            num_outputs: self.num_output_bits(),
            num_and_nodes: self.nodes.len(),
            depth: self.depth(),
        }
    }

    // -- Compaction --

    /// Rebuild this AIG keeping only nodes reachable from outputs.
    ///
    /// Removes dead (orphaned) nodes that don't contribute to any output.
    /// Returns a new AIG with the same function and potentially fewer nodes.
    pub fn compact(&self) -> Aig {
        // Mark reachable nodes
        let mut reachable = vec![false; self.nodes.len()];
        for lit in self.output_lits() {
            self.mark_reachable(lit, &mut reachable);
        }

        // Build new AIG with only reachable nodes
        let total_vars = self.num_inputs as usize + 1 + self.nodes.len();
        let mut lit_map = vec![AigLit::FALSE; total_vars];
        lit_map[0] = AigLit::FALSE;
        for var in 1..=self.num_inputs {
            lit_map[var as usize] = AigLit::new(var, false);
        }

        let mut new_aig = Aig::new();
        for port in self.inputs() {
            if port.bits.len() == 1 {
                new_aig.add_input(&port.name);
            } else {
                new_aig.add_input_bus(&port.name, port.bits.len() as u32);
            }
        }

        for (node_idx, node) in self.nodes.iter().enumerate() {
            let var = self.node_var(node_idx);
            if !reachable[node_idx] {
                // Dead node — map to FALSE (shouldn't be referenced)
                lit_map[var as usize] = AigLit::FALSE;
                continue;
            }
            let f0 = Self::remap_compact(node.fanin0, &lit_map);
            let f1 = Self::remap_compact(node.fanin1, &lit_map);
            let new_lit = new_aig.and(f0, f1);
            lit_map[var as usize] = new_lit;
        }

        for port in self.outputs() {
            if port.bits.len() == 1 {
                let new_lit = Self::remap_compact(port.bits[0], &lit_map);
                new_aig.add_output(&port.name, new_lit);
            } else {
                let new_bits: Vec<AigLit> = port
                    .bits
                    .iter()
                    .map(|&l| Self::remap_compact(l, &lit_map))
                    .collect();
                new_aig.add_output_bus(&port.name, &new_bits);
            }
        }

        new_aig
    }

    fn mark_reachable(&self, lit: AigLit, reachable: &mut [bool]) {
        if lit.is_const() || lit.var() <= self.num_inputs {
            return;
        }
        let idx = (lit.var() - self.num_inputs - 1) as usize;
        if idx >= reachable.len() || reachable[idx] {
            return; // Already visited or out of bounds
        }
        reachable[idx] = true;
        let node = &self.nodes[idx];
        self.mark_reachable(node.fanin0, reachable);
        self.mark_reachable(node.fanin1, reachable);
    }

    fn remap_compact(lit: AigLit, lit_map: &[AigLit]) -> AigLit {
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

    // -- Construction helpers (used by aiger.rs) --

    /// Create an AIG with a pre-allocated number of inputs (no names).
    /// Used by the AIGER reader.
    pub(crate) fn with_inputs(num_inputs: u32) -> Self {
        let mut inputs = Vec::with_capacity(num_inputs as usize);
        for i in 1..=num_inputs {
            inputs.push(AigPort {
                name: format!("i{}", i),
                bits: vec![AigLit::new(i, false)],
            });
        }
        Aig {
            num_inputs,
            nodes: Vec::new(),
            inputs,
            outputs: Vec::new(),
            strash: HashMap::new(),
        }
    }

    /// Push a raw AND node (bypasses structural hashing).
    /// Used by the AIGER reader where nodes are already canonical.
    pub(crate) fn push_node_raw(&mut self, fanin0: AigLit, fanin1: AigLit) {
        let node_idx = self.nodes.len() as u32;
        let f0_raw = fanin0.raw();
        let f1_raw = fanin1.raw();
        let key = if f0_raw <= f1_raw {
            (f0_raw, f1_raw)
        } else {
            (f1_raw, f0_raw)
        };
        self.strash.insert(key, node_idx);
        self.nodes.push(AigNode { fanin0, fanin1 });
    }

    /// Push a raw output literal.
    /// Used by the AIGER reader.
    pub(crate) fn push_output_raw(&mut self, name: String, lit: AigLit) {
        self.outputs.push(AigPort {
            name,
            bits: vec![lit],
        });
    }
}

impl Default for Aig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aig_empty() {
        let aig = Aig::new();
        assert_eq!(aig.num_nodes(), 0);
        assert_eq!(aig.num_inputs(), 0);
        assert_eq!(aig.num_output_bits(), 0);
        assert_eq!(aig.depth(), 0);
    }

    #[test]
    fn aig_single_and() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);
        assert_eq!(aig.num_inputs(), 2);
        assert_eq!(aig.num_nodes(), 1);
        assert_eq!(aig.num_output_bits(), 1);
        assert_eq!(aig.depth(), 1);
    }

    #[test]
    fn aig_constant_folding() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");

        // a & FALSE = FALSE
        assert_eq!(aig.and(a, AigLit::FALSE), AigLit::FALSE);
        assert_eq!(aig.num_nodes(), 0);

        // a & TRUE = a
        assert_eq!(aig.and(a, AigLit::TRUE), a);
        assert_eq!(aig.num_nodes(), 0);

        // FALSE & a = FALSE
        assert_eq!(aig.and(AigLit::FALSE, a), AigLit::FALSE);

        // TRUE & a = a
        assert_eq!(aig.and(AigLit::TRUE, a), a);
        assert_eq!(aig.num_nodes(), 0);
    }

    #[test]
    fn aig_trivial_simplification() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");

        // a & a = a
        assert_eq!(aig.and(a, a), a);
        assert_eq!(aig.num_nodes(), 0);

        // a & !a = FALSE
        assert_eq!(aig.and(a, a.negated()), AigLit::FALSE);
        assert_eq!(aig.num_nodes(), 0);

        // !a & a = FALSE (commuted)
        assert_eq!(aig.and(a.negated(), a), AigLit::FALSE);
        assert_eq!(aig.num_nodes(), 0);
    }

    #[test]
    fn aig_structural_hashing() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y1 = aig.and(a, b);
        let y2 = aig.and(b, a); // Same node, different argument order
        assert_eq!(y1, y2);
        assert_eq!(aig.num_nodes(), 1);
    }

    #[test]
    fn aig_or() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let _y = aig.or(a, b);
        aig.add_output("y", _y);
        // OR = !(!a & !b) — one AND node
        assert_eq!(aig.num_nodes(), 1);
    }

    #[test]
    fn aig_xor() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let _y = aig.xor(a, b);
        aig.add_output("y", _y);
        // XOR = (a & !b) | (!a & b) = !( !(a & !b) & !(!a & b) )
        // = 3 AND nodes: (a & !b), (!a & b), and the final OR (inverted AND)
        assert_eq!(aig.num_nodes(), 3);
    }

    #[test]
    fn aig_mux() {
        let mut aig = Aig::new();
        let s = aig.add_input("s");
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let _y = aig.mux(s, a, b);
        aig.add_output("y", _y);
        // MUX = (s & a) | (!s & b) = 3 AND nodes
        assert_eq!(aig.num_nodes(), 3);
    }

    #[test]
    fn aig_depth_chain() {
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
    }

    #[test]
    fn aig_depth_balanced() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let c = aig.add_input("c");
        let d = aig.add_input("d");
        let ab = aig.and(a, b);
        let cd = aig.and(c, d);
        let abcd = aig.and(ab, cd);
        aig.add_output("y", abcd);
        assert_eq!(aig.depth(), 2);
    }

    #[test]
    fn aig_multi_bit_bus() {
        let mut aig = Aig::new();
        let a_bits = aig.add_input_bus("a", 4);
        assert_eq!(a_bits.len(), 4);
        assert_eq!(aig.num_inputs(), 4);
        // Each bit should have a distinct variable
        for i in 0..4 {
            assert_eq!(a_bits[i].var(), (i + 1) as u32);
        }
    }

    #[test]
    fn aig_ref_counts() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let c = aig.add_input("c");
        let ab = aig.and(a, b);
        let ac = aig.and(a, c);
        aig.add_output("y1", ab);
        aig.add_output("y2", ac);

        let refs = aig.ref_counts();
        // a is used by both AND nodes = 2 refs
        assert_eq!(refs[a.var() as usize], 2);
        // b and c each used once
        assert_eq!(refs[b.var() as usize], 1);
        assert_eq!(refs[c.var() as usize], 1);
    }

    #[test]
    fn aig_lit_debug_format() {
        assert_eq!(format!("{:?}", AigLit::FALSE), "FALSE");
        assert_eq!(format!("{:?}", AigLit::TRUE), "TRUE");
        let v1 = AigLit::new(1, false);
        assert_eq!(format!("{:?}", v1), "v1");
        let v1_neg = AigLit::new(1, true);
        assert_eq!(format!("{:?}", v1_neg), "!v1");
    }

    // -- D1: Shannon decomposition correctness (Sprint 213) --

    fn verify_truth_table(tt: u64, num_inputs: u32) {
        use crate::synth::mapping::evaluate_aig;
        let aig = Aig::from_truth_table(tt, num_inputs);
        let n = num_inputs as usize;
        for combo in 0..(1u32 << n) {
            let inputs: Vec<bool> = (0..n).map(|i| (combo >> i) & 1 != 0).collect();
            let expected = (tt >> combo) & 1 != 0;
            let actual = evaluate_aig(&aig, &inputs);
            assert_eq!(
                actual.len(),
                1,
                "tt=0x{:X} n={}: expected 1 output, got {}",
                tt,
                num_inputs,
                actual.len()
            );
            assert_eq!(
                actual[0], expected,
                "tt=0x{:X} n={} combo={}: expected {} got {}",
                tt, num_inputs, combo, expected, actual[0]
            );
        }
    }

    #[test]
    fn from_truth_table_all_2input() {
        for tt in 0..16u64 {
            verify_truth_table(tt, 2);
        }
    }

    #[test]
    fn from_truth_table_all_3input() {
        for tt in 0..256u64 {
            verify_truth_table(tt, 3);
        }
    }

    #[test]
    fn from_truth_table_known_functions() {
        // 2-input AND = 0x8, OR = 0xE, XOR = 0x6
        verify_truth_table(0x8, 2);
        verify_truth_table(0xE, 2);
        verify_truth_table(0x6, 2);
        // 3-input majority = 0xE8
        verify_truth_table(0xE8, 3);
        // 4-input AND = 0x8000
        verify_truth_table(0x8000, 4);
    }

    #[test]
    fn from_truth_tables_multi_output() {
        use crate::synth::mapping::evaluate_aig;

        // 2-bit increment: inputs [s0, s1], outputs [ns0=NOT s0, ns1=XOR(s1, s0)]
        // ns0: 0→1, 1→0, 2→1, 3→0 → 0b0101 = 5
        // ns1: 0→0, 1→1, 2→1, 3→0 → 0b0110 = 6
        let aig = Aig::from_truth_tables(&[(5, 2), (6, 2)]);
        assert_eq!(aig.num_inputs(), 2);
        assert_eq!(aig.output_lits().count(), 2);

        for combo in 0..4u32 {
            let inputs = vec![(combo & 1) != 0, (combo >> 1) != 0];
            let outputs = evaluate_aig(&aig, &inputs);
            let expected_next = (combo + 1) % 4;
            let expected = vec![(expected_next & 1) != 0, (expected_next >> 1) != 0];
            assert_eq!(outputs, expected, "combo={}", combo);
        }
    }

    /// Sprint 215 D4: 2-bit adder — 4 inputs, 3 outputs (s0, s1, carry).
    #[test]
    fn multi_output_synth_2bit_adder() {
        use crate::synth::mapping::evaluate_aig;
        use crate::synth::{PlaceConfig, RouteConfig, validate_synth_pipeline};

        // inputs = [a0, a1, b0, b1], A = a0+2*a1, B = b0+2*b1, result = A+B
        // s0 = 0x5A5A, s1 = 0x936C, cout = 0xEC80
        let aig = Aig::from_truth_tables(&[(0x5A5A, 4), (0x936C, 4), (0xEC80, 4)]);
        assert_eq!(aig.num_inputs(), 4);
        assert_eq!(aig.output_lits().count(), 3);

        // Verify AIG evaluation against reference
        for combo in 0..16u32 {
            let a0 = combo & 1;
            let a1 = (combo >> 1) & 1;
            let b0 = (combo >> 2) & 1;
            let b1 = (combo >> 3) & 1;
            let a_val = a0 + 2 * a1;
            let b_val = b0 + 2 * b1;
            let sum = a_val + b_val;
            let expected = vec![(sum & 1) != 0, ((sum >> 1) & 1) != 0, ((sum >> 2) & 1) != 0];
            let inputs: Vec<bool> = (0..4).map(|i| (combo >> i) & 1 != 0).collect();
            let actual = evaluate_aig(&aig, &inputs);
            assert_eq!(
                actual, expected,
                "adder combo={}: A={} B={} sum={}",
                combo, a_val, b_val, sum
            );
        }

        // Validate full synth pipeline
        let result =
            validate_synth_pipeline(&aig, &PlaceConfig::standalone(), &RouteConfig::standalone())
                .expect("synth pipeline");
        assert!(
            result.passed(),
            "2-bit adder: {} mismatches out of {} combos",
            result.mismatches.len(),
            result.combos_tested,
        );
    }

    /// Sprint 215 D4: 3-to-8 decoder — 3 inputs, 8 outputs.
    #[test]
    fn multi_output_synth_decoder_3to8() {
        use crate::synth::mapping::evaluate_aig;
        use crate::synth::{PlaceConfig, RouteConfig, validate_synth_pipeline};

        // Decoder: for input combo i, output[i] = 1 and all others = 0.
        // output[j] truth table: bit at position i is 1 iff i == j.
        let tts: Vec<(u64, u32)> = (0..8u64).map(|j| (1u64 << j, 3)).collect();
        let aig = Aig::from_truth_tables(&tts);
        assert_eq!(aig.num_inputs(), 3);
        assert_eq!(aig.output_lits().count(), 8);

        // Verify AIG
        for combo in 0..8u32 {
            let inputs: Vec<bool> = (0..3).map(|i| (combo >> i) & 1 != 0).collect();
            let actual = evaluate_aig(&aig, &inputs);
            let expected: Vec<bool> = (0..8u32).map(|j| j == combo).collect();
            assert_eq!(actual, expected, "decoder combo={}", combo);
        }

        // Validate full synth pipeline — all 8 outputs must pass.
        let result =
            validate_synth_pipeline(&aig, &PlaceConfig::standalone(), &RouteConfig::standalone())
                .expect("synth pipeline");
        assert!(
            result.mismatches.is_empty(),
            "3-to-8 decoder: {} mismatches ({:?})",
            result.mismatches.len(),
            result
                .mismatches
                .iter()
                .map(|m| m.output_idx)
                .collect::<Vec<_>>(),
        );
    }

    /// Sprint 215 D4: AIGER round-trip for multi-output circuit.
    #[test]
    fn aiger_roundtrip_multi_output_synth() {
        use crate::synth::aiger::{read_aiger, write_aiger};
        use crate::synth::mapping::evaluate_aig;
        use std::io::BufReader;

        // 2-bit adder
        let aig = Aig::from_truth_tables(&[(0x5A5A, 4), (0x936C, 4), (0xEC80, 4)]);

        // Write binary AIGER
        let mut buf = Vec::new();
        write_aiger(&aig, &mut buf).expect("write");
        // Read back
        let aig2 = read_aiger(&mut BufReader::new(&buf[..])).expect("read");
        assert_eq!(aig2.num_inputs(), 4);
        assert_eq!(aig2.output_lits().count(), 3);

        // Verify functional equivalence
        for combo in 0..16u32 {
            let inputs: Vec<bool> = (0..4).map(|i| (combo >> i) & 1 != 0).collect();
            let orig = evaluate_aig(&aig, &inputs);
            let roundtrip = evaluate_aig(&aig2, &inputs);
            assert_eq!(
                orig, roundtrip,
                "AIGER round-trip mismatch at combo={}",
                combo
            );
        }
    }

    /// Sprint 215 D4: AIGER export → import → synth pipeline → validate.
    #[test]
    fn aiger_export_import_synth() {
        use crate::synth::aiger::{read_aiger, write_aiger};
        use crate::synth::{PlaceConfig, RouteConfig, validate_synth_pipeline};
        use std::io::BufReader;

        // 2-input XOR + AND (2 outputs)
        let aig = Aig::from_truth_tables(&[(0x6, 2), (0x8, 2)]);

        // Export → import
        let mut buf = Vec::new();
        write_aiger(&aig, &mut buf).expect("write");
        let aig2 = read_aiger(&mut BufReader::new(&buf[..])).expect("read");

        // Synth the imported AIG
        let result = validate_synth_pipeline(
            &aig2,
            &PlaceConfig::standalone(),
            &RouteConfig::standalone(),
        )
        .expect("synth pipeline");
        assert!(
            result.passed(),
            "AIGER import synth: {} mismatches",
            result.mismatches.len(),
        );
    }
}
