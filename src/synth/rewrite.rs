//! AIG rewriting using NPN4 optimal subgraphs.
//!
//! For each AND node, examines its K=4 cuts, classifies each cut's truth table
//! via NPN4, looks up a precomputed optimal AIG subgraph for that class, and
//! replaces the subgraph if it uses fewer AND nodes (measured by MFFC).
//!
//! Builds a new AIG in a single topological pass — never modifies the old AIG.

use super::aig::{Aig, AigLit};
use super::cuts::enumerate_cuts;
use super::npn::{invert_perm, npn4_classify};

// ---------------------------------------------------------------------------
// Optimal subgraph database
// ---------------------------------------------------------------------------

/// Max AND nodes per recipe. 4-input functions need at most 7 AND nodes
/// via Shannon expansion (empirically, most need ≤6).
const MAX_RECIPE_STEPS: usize = 9;

/// Recipe for building an optimal AIG subgraph for an NPN4 class.
///
/// Each step is `(fanin0, fanin1)` where:
/// - Indices 0-3 refer to the 4 inputs of the canonical function
/// - Indices 4+ refer to AND nodes created by earlier steps
/// - Encoding: `idx * 2` = positive, `idx * 2 + 1` = complement
///
/// The recipe computes the canonical truth table. To compute a different NPN4
/// variant, the caller applies the NPN4 transform to remap inputs and output.
#[derive(Clone, Copy, Debug)]
struct SubgraphRecipe {
    steps: [(u8, u8); MAX_RECIPE_STEPS],
    num_steps: u8,
    output_neg: bool, // Is the recipe output complemented?
}

impl SubgraphRecipe {
    const fn new(steps: [(u8, u8); MAX_RECIPE_STEPS], num_steps: u8, output_neg: bool) -> Self {
        SubgraphRecipe {
            steps,
            num_steps,
            output_neg,
        }
    }
}

/// Evaluate a subgraph recipe for a given 4-input assignment.
/// Returns the output value (before output_neg).
fn eval_recipe(recipe: &SubgraphRecipe, inputs: [bool; 4]) -> bool {
    let mut vals = [false; 4 + MAX_RECIPE_STEPS]; // 4 inputs + up to MAX_RECIPE_STEPS AND nodes
    vals[0] = inputs[0];
    vals[1] = inputs[1];
    vals[2] = inputs[2];
    vals[3] = inputs[3];

    for step_idx in 0..recipe.num_steps as usize {
        let (f0, f1) = recipe.steps[step_idx];
        let v0 = vals[(f0 >> 1) as usize] ^ (f0 & 1 != 0);
        let v1 = vals[(f1 >> 1) as usize] ^ (f1 & 1 != 0);
        vals[4 + step_idx] = v0 && v1;
    }

    let base = if recipe.num_steps == 0 {
        false // No steps → const 0
    } else {
        vals[4 + recipe.num_steps as usize - 1]
    };

    base ^ recipe.output_neg
}

/// Verify a recipe produces the expected truth table.
fn verify_recipe(recipe: &SubgraphRecipe, expected_tt: u16) -> bool {
    for j in 0..16u32 {
        let inputs = [
            (j & 1) != 0,
            (j >> 1 & 1) != 0,
            (j >> 2 & 1) != 0,
            (j >> 3 & 1) != 0,
        ];
        let val = eval_recipe(recipe, inputs);
        let expected = (expected_tt >> j) & 1 != 0;
        if val != expected {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Database of optimal subgraphs for common NPN4 classes
// ---------------------------------------------------------------------------

/// Get the optimal subgraph recipe for an NPN4 canonical truth table.
///
/// Returns None for classes without a known recipe (the rewriter will
/// rebuild the node as-is). The database covers the most common classes
/// that appear in arithmetic and logic circuits.
///
/// Step encoding:
/// - Inputs 0-3: encoded as 0,2,4,6 (positive) or 1,3,5,7 (complement)
/// - Step results: step j → (4+j)*2 (positive), (4+j)*2+1 (complement)
fn get_optimal_recipe(canonical_tt: u16) -> Option<SubgraphRecipe> {
    // Build the database on first call.
    // We use the NPN4 table to discover which canonical TTs exist,
    // then construct recipes using a small AIG builder.
    static RECIPES: LazyLock<std::collections::HashMap<u16, SubgraphRecipe>> =
        LazyLock::new(build_recipe_database);

    RECIPES.get(&canonical_tt).copied()
}

use std::sync::LazyLock;

/// Build recipes for all 222 NPN4 canonical truth tables by constructing
/// each function with a mini AIG builder.
fn build_recipe_database() -> std::collections::HashMap<u16, SubgraphRecipe> {
    let mut db = std::collections::HashMap::new();

    // Discover all canonical TTs from the NPN4 table
    let mut canonicals = std::collections::HashSet::new();
    for tt in 0..=0xFFFFu16 {
        let npn = npn4_classify(tt);
        canonicals.insert(npn.canonical_tt);
    }

    for &canonical in &canonicals {
        if let Some(recipe) = build_recipe_for(canonical) {
            // Verify correctness
            debug_assert!(
                verify_recipe(&recipe, canonical),
                "Recipe for 0x{:04X} is incorrect",
                canonical
            );
            db.insert(canonical, recipe);
        }
    }

    db
}

/// Build a recipe for a single canonical truth table.
fn build_recipe_for(tt: u16) -> Option<SubgraphRecipe> {
    // Trivial cases
    if tt == 0x0000 {
        // Const 0 — no recipe needed (0 AND nodes)
        return None;
    }
    if tt == 0xFFFF {
        return None; // Const 1 (output negation of const 0)
    }

    // Check single-variable functions
    for i in 0..4u8 {
        let var_tt = var_truth_table(i);
        if tt == var_tt || tt == !var_tt {
            return None; // Just a wire or inverter, no AND nodes
        }
    }

    // Build using iterative mini AIG
    let mut builder = RecipeBuilder::new();
    if let Some(output_lit) = builder.build_function(tt, 0) {
        builder.to_recipe(output_lit)
    } else {
        None
    }
}

/// Truth table for variable i (4-input): bit j set when (j >> i) & 1 != 0.
fn var_truth_table(i: u8) -> u16 {
    let mut tt = 0u16;
    for j in 0..16u16 {
        if (j >> i) & 1 != 0 {
            tt |= 1 << j;
        }
    }
    tt
}

#[derive(Clone)]
struct RecipeBuilder {
    steps: Vec<(u8, u8)>,
    cache: Vec<(u16, u8)>,
}

impl RecipeBuilder {
    fn new() -> Self {
        RecipeBuilder {
            steps: Vec::new(),
            cache: Vec::new(),
        }
    }

    fn input_lit(i: u8, neg: bool) -> u8 {
        i * 2 + neg as u8
    }

    fn step_lit(step_idx: usize, neg: bool) -> u8 {
        ((4 + step_idx) * 2 + neg as usize) as u8
    }

    fn and(&mut self, a: u8, b: u8) -> Option<u8> {
        if self.steps.len() >= MAX_RECIPE_STEPS {
            return None;
        }
        let (f0, f1) = if a <= b { (a, b) } else { (b, a) };

        // Check if this AND already exists
        for (i, &(s0, s1)) in self.steps.iter().enumerate() {
            if s0 == f0 && s1 == f1 {
                return Some(Self::step_lit(i, false));
            }
        }

        let idx = self.steps.len();
        self.steps.push((f0, f1));
        Some(Self::step_lit(idx, false))
    }

    fn build_function(&mut self, tt: u16, depth: u8) -> Option<u8> {
        if depth > 8 {
            return None; // Prevent deep recursion
        }

        // Check cache
        for &(cached_tt, cached_lit) in &self.cache {
            if cached_tt == tt {
                return Some(cached_lit);
            }
            if cached_tt == !tt {
                return Some(cached_lit ^ 1);
            }
        }

        // Single variable
        for i in 0..4u8 {
            let vtt = var_truth_table(i);
            if tt == vtt {
                return Some(Self::input_lit(i, false));
            }
            if tt == !vtt {
                return Some(Self::input_lit(i, true));
            }
        }

        // 2-input functions
        for i in 0..4u8 {
            for j in (i + 1)..4u8 {
                let vi = var_truth_table(i);
                let vj = var_truth_table(j);

                // AND
                if tt == vi & vj {
                    let lit = self.and(Self::input_lit(i, false), Self::input_lit(j, false))?;
                    self.cache.push((tt, lit));
                    return Some(lit);
                }
                if tt == !(vi & vj) {
                    let lit = self.and(Self::input_lit(i, false), Self::input_lit(j, false))?;
                    self.cache.push((!tt, lit));
                    return Some(lit ^ 1);
                }
                // NOR = !a & !b
                if tt == !vi & !vj {
                    let lit = self.and(Self::input_lit(i, true), Self::input_lit(j, true))?;
                    self.cache.push((tt, lit));
                    return Some(lit);
                }
                // OR = !(!a & !b)
                if tt == vi | vj {
                    let lit = self.and(Self::input_lit(i, true), Self::input_lit(j, true))?;
                    self.cache.push((!tt, lit));
                    return Some(lit ^ 1);
                }
                // XOR: (a & !b) | (!a & b) = !( !(a & !b) & !(!a & b) )
                if tt == vi ^ vj {
                    let t1 = self.and(Self::input_lit(i, false), Self::input_lit(j, true))?;
                    let t2 = self.and(Self::input_lit(i, true), Self::input_lit(j, false))?;
                    let t3 = self.and(t1 ^ 1, t2 ^ 1)?;
                    let lit = t3 ^ 1;
                    self.cache.push((tt, lit));
                    return Some(lit);
                }
                // XNOR = !(a ^ b)
                if tt == !(vi ^ vj) {
                    let t1 = self.and(Self::input_lit(i, false), Self::input_lit(j, true))?;
                    let t2 = self.and(Self::input_lit(i, true), Self::input_lit(j, false))?;
                    let t3 = self.and(t1 ^ 1, t2 ^ 1)?;
                    let lit = t3; // NOT of XOR
                    self.cache.push((!tt, t3 ^ 1));
                    return Some(lit);
                }
            }
        }

        // Shannon expansion: try ALL 4 variable orderings, keep the one
        // with fewest AND steps. This dramatically improves NPN4 coverage
        // compared to the single best-heuristic variable approach.
        let mut best: Option<(u8, RecipeBuilder)> = None;

        for var in 0..4u8 {
            let pos_cof = cofactor(tt, var, true);
            let neg_cof = cofactor(tt, var, false);
            // Skip if this variable provides no decomposition progress
            if pos_cof == tt && neg_cof == tt {
                continue;
            }
            if pos_cof == neg_cof && pos_cof == tt {
                continue;
            }

            let mut fork = self.clone();
            if let Some(result) = fork.try_shannon(tt, var, pos_cof, neg_cof, depth) {
                let is_better = match &best {
                    None => true,
                    Some((_, best_fork)) => fork.steps.len() < best_fork.steps.len(),
                };
                if is_better {
                    best = Some((result, fork));
                }
            }
        }

        if let Some((result, best_fork)) = best {
            *self = best_fork;
            self.cache.push((tt, result));
            return Some(result);
        }
        None
    }

    /// Shannon expansion with constant-cofactor shortcuts.
    ///
    /// When one cofactor is const 0 or const 1, we need only 1 AND node
    /// instead of 3 for the generic MUX construction.
    fn try_shannon(
        &mut self,
        _tt: u16,
        var: u8,
        pos_cof: u16,
        neg_cof: u16,
        depth: u8,
    ) -> Option<u8> {
        // Case: cofactors are equal → f doesn't depend on var
        if pos_cof == neg_cof {
            return self.build_function(pos_cof, depth + 1);
        }

        // Case: f|x=1 = 0 → f = !x & f|x=0
        if pos_cof == 0x0000 {
            let nc = self.build_function(neg_cof, depth + 1)?;
            let result = self.and(Self::input_lit(var, true), nc)?;
            return Some(result);
        }

        // Case: f|x=0 = 0 → f = x & f|x=1
        if neg_cof == 0x0000 {
            let pc = self.build_function(pos_cof, depth + 1)?;
            let result = self.and(Self::input_lit(var, false), pc)?;
            return Some(result);
        }

        // Case: f|x=1 = 1 → f = x | f|x=0 = !(!x & !f|x=0)
        if pos_cof == 0xFFFF {
            let nc = self.build_function(neg_cof, depth + 1)?;
            let result = self.and(Self::input_lit(var, true), nc ^ 1)?;
            return Some(result ^ 1);
        }

        // Case: f|x=0 = 1 → f = !x | f|x=1 = !(x & !f|x=1)
        if neg_cof == 0xFFFF {
            let pc = self.build_function(pos_cof, depth + 1)?;
            let result = self.and(Self::input_lit(var, false), pc ^ 1)?;
            return Some(result ^ 1);
        }

        // Case: f|x=1 = !f|x=0 → f = XOR(x, f|x=0)
        // XOR(a,b) = !( !(a & !b) & !(!a & b) ) — 3 ANDs
        if pos_cof == !neg_cof {
            let nc = self.build_function(neg_cof, depth + 1)?;
            let t1 = self.and(Self::input_lit(var, false), nc ^ 1)?;
            let t2 = self.and(Self::input_lit(var, true), nc)?;
            let t3 = self.and(t1 ^ 1, t2 ^ 1)?;
            return Some(t3 ^ 1);
        }

        // Generic MUX: f = (x & pos) | (!x & neg) = !( !(x & pos) & !(!x & neg) )
        let pc = self.build_function(pos_cof, depth + 1)?;
        let nc = self.build_function(neg_cof, depth + 1)?;

        let t1 = self.and(Self::input_lit(var, false), pc)?;
        let t2 = self.and(Self::input_lit(var, true), nc)?;
        let t3 = self.and(t1 ^ 1, t2 ^ 1)?;
        Some(t3 ^ 1)
    }

    fn to_recipe(&self, output_lit: u8) -> Option<SubgraphRecipe> {
        if self.steps.is_empty() {
            return None;
        }
        if self.steps.len() > MAX_RECIPE_STEPS {
            return None;
        }

        let mut steps = [(0u8, 0u8); MAX_RECIPE_STEPS];
        for (i, &(f0, f1)) in self.steps.iter().enumerate() {
            steps[i] = (f0, f1);
        }

        let output_step = (output_lit >> 1) as usize;
        if output_step < 4 {
            return None; // Output is an input
        }
        let output_step_idx = output_step - 4;
        if output_step_idx >= self.steps.len() {
            return None;
        }

        // Truncate to only the steps up to the output
        let num = output_step_idx + 1;
        Some(SubgraphRecipe::new(steps, num as u8, output_lit & 1 != 0))
    }
}

/// Compute the cofactor of tt: set variable `var` to `value`, producing a
/// truth table that's constant in that variable (both halves identical).
fn cofactor(tt: u16, var: u8, value: bool) -> u16 {
    let mut result = 0u16;
    for j in 0..16u32 {
        // For this output position, force var to `value` and look up the source
        let src = if value {
            j | (1 << var) // Force var=1
        } else {
            j & !(1 << var) // Force var=0
        };
        if (tt >> src) & 1 != 0 {
            result |= 1 << j;
        }
    }
    result
}

/// Simple complexity metric for cofactor selection.
#[allow(dead_code)]
fn popcount_complexity(tt: u16) -> u32 {
    let ones = tt.count_ones();
    // Prefer cofactors close to const (0 or 16 ones)
    ones.min(16 - ones)
}

// ---------------------------------------------------------------------------
// MFFC computation
// ---------------------------------------------------------------------------

/// Compute the MFFC (Maximum Fanout-Free Cone) size for a cut.
///
/// Uses recursive dereference: starting from root, decrement fanin ref counts.
/// Any fanin whose ref count drops to 0 is also in the MFFC.
/// Returns the number of AND nodes that would become dead if the root's subgraph
/// were replaced.
fn mffc_size(aig: &Aig, root_var: u32, cut_leaves: &[u32], refs: &mut [u32]) -> usize {
    let mut size = 1; // Root itself is always in MFFC

    // Decrement refs for root's fanins, recurse into any that become dead
    if let Some(node_idx) = aig.var_to_node_idx(root_var) {
        let node = aig.node(node_idx);
        deref_fanin(aig, node.fanin0.var(), cut_leaves, refs, &mut size);
        deref_fanin(aig, node.fanin1.var(), cut_leaves, refs, &mut size);
    }

    size
}

fn deref_fanin(aig: &Aig, var: u32, cut_leaves: &[u32], refs: &mut [u32], size: &mut usize) {
    // Stop at cut leaves, primary inputs, or constants
    if var == 0 || var <= aig.num_inputs() || cut_leaves.contains(&var) {
        return;
    }

    let v = var as usize;
    if v >= refs.len() {
        return;
    }

    refs[v] = refs[v].saturating_sub(1);
    if refs[v] == 0 {
        // This node is now dead — part of the MFFC
        *size += 1;
        if let Some(node_idx) = aig.var_to_node_idx(var) {
            let node = aig.node(node_idx);
            deref_fanin(aig, node.fanin0.var(), cut_leaves, refs, size);
            deref_fanin(aig, node.fanin1.var(), cut_leaves, refs, size);
        }
    }
}

// ---------------------------------------------------------------------------
// Rewrite pass
// ---------------------------------------------------------------------------

/// Rewrite an AIG using NPN4 optimal subgraphs.
///
/// For each AND node, examines its K=4 cuts, classifies each cut's truth table
/// via NPN4, and replaces with an optimal subgraph if profitable.
///
/// `zero_cost`: if true, also accept replacements with gain == 0 (reshaping
/// for future optimization passes).
///
/// Returns a new, potentially smaller AIG.
pub fn rewrite(aig: &Aig, zero_cost: bool) -> Aig {
    let num_inputs = aig.num_inputs();
    let num_nodes = aig.num_nodes();
    let all_cuts = enumerate_cuts(aig);
    let base_refs = aig.ref_counts();

    // lit_map: old variable → new literal
    let total_vars = num_inputs as usize + 1 + num_nodes;
    let mut lit_map: Vec<AigLit> = vec![AigLit::FALSE; total_vars];

    // Map constants and inputs
    lit_map[0] = AigLit::FALSE;
    for var in 1..=num_inputs {
        lit_map[var as usize] = AigLit::new(var, false);
    }

    // Build new AIG
    let mut new_aig = Aig::new();
    for port in aig.inputs() {
        if port.bits.len() == 1 {
            new_aig.add_input(&port.name);
        } else {
            new_aig.add_input_bus(&port.name, port.bits.len() as u32);
        }
    }

    // Process each AND node in topological order
    for (node_idx, node) in aig.nodes_topo() {
        let var = aig.node_var(node_idx);
        let node_cuts = &all_cuts[node_idx];

        // Find the best replacement
        let mut best_gain: i32 = if zero_cost { 0 } else { 1 };
        let mut best_recipe: Option<SubgraphRecipe> = None;
        let mut best_cut_leaves: Option<Vec<u32>> = None;
        let mut best_npn_perm: [u8; 4] = [0, 1, 2, 3];
        let mut best_npn_neg_in: u8 = 0;
        let mut best_npn_neg_out: bool = false;

        for cut in &node_cuts.cuts {
            // Skip trivial cut (self-reference)
            if cut.num_leaves == 1 && cut.leaves[0] == var {
                continue;
            }

            // Skip cuts with < 2 leaves (constants handled elsewhere)
            if cut.num_leaves < 2 {
                continue;
            }

            // Classify the cut's truth table via NPN4
            let npn = npn4_classify(cut.truth_table);

            // Look up optimal recipe for this class
            if let Some(recipe) = get_optimal_recipe(npn.canonical_tt) {
                // Compute MFFC size (use a scratch copy of ref counts)
                let mut scratch_refs = base_refs.clone();
                let mffc = mffc_size(aig, var, cut.leaf_slice(), &mut scratch_refs);

                let gain = mffc as i32 - recipe.num_steps as i32;

                if gain >= best_gain {
                    best_gain = gain;
                    best_recipe = Some(recipe);
                    best_cut_leaves = Some(cut.leaf_slice().to_vec());
                    best_npn_perm = npn.perm;
                    best_npn_neg_in = npn.neg_in;
                    best_npn_neg_out = npn.neg_out;
                }
            }
        }

        if let (Some(recipe), Some(cut_leaves)) = (best_recipe, best_cut_leaves) {
            // Build the replacement subgraph
            let new_lit = build_subgraph(
                &mut new_aig,
                &recipe,
                &cut_leaves,
                &lit_map,
                num_inputs,
                best_npn_perm,
                best_npn_neg_in,
                best_npn_neg_out,
            );
            lit_map[var as usize] = new_lit;
        } else {
            // No profitable replacement — rebuild as-is
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

/// Build a subgraph from a recipe, mapping canonical inputs through the NPN4
/// transform to the actual cut leaves.
///
/// The recipe computes the canonical truth table. The NPN4 transform stored
/// in the table maps canonical → actual truth table:
///   actual_tt = npn4_apply_transform(canonical_tt, perm, neg_in, neg_out)
///
/// `tt_permute(canonical, perm)` produces `result(v) = canonical(v_{inv[0]}, ..., v_{inv[3]})`
/// where `inv = invert_perm(perm)`. So canonical input k receives the value from
/// actual input `inv_perm[k]`, negated if `neg_in` bit `inv_perm[k]` is set.
///
/// To wire the recipe:
/// - Recipe input k corresponds to actual cut leaf at position inv_perm[k]
/// - If neg_in bit inv_perm[k] is set, negate that input to the recipe
/// - If neg_out, negate the recipe's output
fn build_subgraph(
    new_aig: &mut Aig,
    recipe: &SubgraphRecipe,
    cut_leaves: &[u32],
    lit_map: &[AigLit],
    num_inputs: u32,
    npn_perm: [u8; 4],
    npn_neg_in: u8,
    npn_neg_out: bool,
) -> AigLit {
    // Map recipe inputs to actual literals.
    //
    // tt_permute(canonical, perm) gives: result(v) = canonical(v_{inv[0]}, ..., v_{inv[3]})
    // where inv = invert_perm(perm). After neg_in is applied to the permuted result:
    //   canonical input k = actual input inv_perm[k], negated if neg_in bit inv_perm[k] is set.
    //
    // Recipe input k (= canonical input k) wires to cut_leaves[inv_perm[k]].
    let inv_perm = invert_perm(npn_perm);
    let num_cut_leaves = cut_leaves.len();
    let mut input_lits = [AigLit::FALSE; 4];

    for canonical_pos in 0..4u8 {
        let actual_pos = inv_perm[canonical_pos as usize] as usize;
        let base_lit = if actual_pos < num_cut_leaves {
            let leaf_var = cut_leaves[actual_pos];
            if leaf_var == 0 {
                AigLit::FALSE
            } else if leaf_var <= num_inputs {
                lit_map[leaf_var as usize]
            } else if (leaf_var as usize) < lit_map.len() {
                lit_map[leaf_var as usize]
            } else {
                AigLit::FALSE
            }
        } else {
            AigLit::FALSE // Unused input
        };

        // Apply input negation from NPN transform.
        // neg_in bit inv_perm[k] controls whether canonical input k is negated.
        input_lits[canonical_pos as usize] = if (npn_neg_in >> actual_pos) & 1 != 0 {
            base_lit.negated()
        } else {
            base_lit
        };
    }

    // Build the recipe steps
    let mut step_lits = [AigLit::FALSE; MAX_RECIPE_STEPS];
    for step_idx in 0..recipe.num_steps as usize {
        let (f0_enc, f1_enc) = recipe.steps[step_idx];
        let f0 = decode_recipe_lit(f0_enc, &input_lits, &step_lits);
        let f1 = decode_recipe_lit(f1_enc, &input_lits, &step_lits);
        step_lits[step_idx] = new_aig.and(f0, f1);
    }

    // Get the recipe output
    let mut result = if recipe.num_steps > 0 {
        step_lits[recipe.num_steps as usize - 1]
    } else {
        AigLit::FALSE
    };

    // Apply recipe output negation
    if recipe.output_neg {
        result = result.negated();
    }

    // Apply NPN output negation
    if npn_neg_out {
        result = result.negated();
    }

    result
}

/// Decode a recipe literal encoding to an AigLit.
fn decode_recipe_lit(
    enc: u8,
    input_lits: &[AigLit; 4],
    step_lits: &[AigLit; MAX_RECIPE_STEPS],
) -> AigLit {
    let idx = (enc >> 1) as usize;
    let neg = enc & 1 != 0;
    let base = if idx < 4 {
        input_lits[idx]
    } else {
        step_lits[idx - 4]
    };
    if neg { base.negated() } else { base }
}

/// Remap a literal through lit_map.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::mapping::evaluate_aig;

    #[test]
    fn rewrite_single_and_unchanged() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);

        let rewritten = rewrite(&aig, false);
        // Single AND is already optimal — should be 1 node
        assert_eq!(rewritten.num_nodes(), 1);
    }

    #[test]
    fn rewrite_preserves_equiv_adder() {
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

        let rewritten = rewrite(&aig, false);

        // Exhaustive equivalence check
        for assignment in 0..256u32 {
            let inputs: Vec<bool> = (0..8).map(|i| (assignment >> i) & 1 != 0).collect();
            let orig = evaluate_aig(&aig, &inputs);
            let rw = evaluate_aig(&rewritten, &inputs);
            assert_eq!(
                orig, rw,
                "Mismatch at input {:08b}: orig={:?} rw={:?}",
                assignment, orig, rw
            );
        }
    }

    #[test]
    fn rewrite_constant_passthrough() {
        let mut aig = Aig::new();
        let _a = aig.add_input("a");
        aig.add_output("y", AigLit::TRUE);
        aig.add_output("z", AigLit::FALSE);

        let rewritten = rewrite(&aig, false);
        assert_eq!(rewritten.num_nodes(), 0);

        let orig = evaluate_aig(&aig, &[false]);
        let rw = evaluate_aig(&rewritten, &[false]);
        assert_eq!(orig, rw);
    }

    #[test]
    fn rewrite_shared_fanout_safe() {
        // a & b used by two outputs — shared node should not be broken
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let c = aig.add_input("c");
        let ab = aig.and(a, b);
        let y1 = aig.and(ab, c);
        aig.add_output("y1", y1);
        aig.add_output("y2", ab); // ab is shared

        let rewritten = rewrite(&aig, false);

        for assignment in 0..8u32 {
            let inputs: Vec<bool> = (0..3).map(|i| (assignment >> i) & 1 != 0).collect();
            let orig = evaluate_aig(&aig, &inputs);
            let rw = evaluate_aig(&rewritten, &inputs);
            assert_eq!(orig, rw);
        }
    }

    #[test]
    fn rewrite_zero_cost_mode() {
        // Zero-cost should accept gain == 0 replacements
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.xor(a, b);
        aig.add_output("y", y);

        let rw_normal = rewrite(&aig, false);
        let rw_zero = rewrite(&aig, true);

        // Both should preserve equivalence
        for assignment in 0..4u32 {
            let inputs: Vec<bool> = (0..2).map(|i| (assignment >> i) & 1 != 0).collect();
            let orig = evaluate_aig(&aig, &inputs);
            assert_eq!(orig, evaluate_aig(&rw_normal, &inputs));
            assert_eq!(orig, evaluate_aig(&rw_zero, &inputs));
        }
    }

    #[test]
    fn mffc_recursive_deref() {
        // Build a chain: y = (a & b) & c where (a & b) has fanout 1
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let c = aig.add_input("c");
        let ab = aig.and(a, b);
        let y = aig.and(ab, c);
        aig.add_output("y", y);

        let refs = aig.ref_counts();
        let var_y = aig.node_var(1); // y is the second AND node

        // MFFC of y with cut leaves {a, b, c}: both AND nodes are in MFFC
        let mut scratch = refs.clone();
        let size = mffc_size(&aig, var_y, &[1, 2, 3], &mut scratch);
        assert_eq!(size, 2, "MFFC should include both AND nodes");

        // MFFC of y with cut leaves {ab_var, c}: only y itself
        let var_ab = aig.node_var(0);
        let mut scratch2 = refs.clone();
        let size2 = mffc_size(&aig, var_y, &[var_ab, 3], &mut scratch2);
        assert_eq!(size2, 1, "MFFC should include only y when ab is a leaf");
    }

    #[test]
    fn recipe_verification() {
        // Verify some known recipes
        let _and_tt = 0x0001u16; // canonical for the class containing AND-like functions

        // Check that get_optimal_recipe returns valid recipes for
        // whatever canonical TTs exist in the NPN4 table
        let mut tested = 0;
        let mut seen_canonicals = std::collections::HashSet::new();
        for tt in 0..=0xFFFFu16 {
            let npn = npn4_classify(tt);
            if seen_canonicals.insert(npn.canonical_tt) {
                if let Some(recipe) = get_optimal_recipe(npn.canonical_tt) {
                    assert!(
                        verify_recipe(&recipe, npn.canonical_tt),
                        "Recipe for canonical 0x{:04X} does not produce correct truth table",
                        npn.canonical_tt,
                    );
                    tested += 1;
                }
            }
        }
        assert!(tested > 0, "Should have verified at least one recipe");
        let total = seen_canonicals.len();
        let mut no_recipe = Vec::new();
        for &ct in &seen_canonicals {
            // Skip trivial ones (const, single var)
            if ct == 0x0000 || ct == 0xFFFF {
                continue;
            }
            let is_single_var = (0..4u8).any(|i| {
                let v = var_truth_table(i);
                ct == v || ct == !v
            });
            if is_single_var {
                continue;
            }
            if get_optimal_recipe(ct).is_none() {
                no_recipe.push(ct);
            }
        }
        eprintln!(
            "Recipe coverage: {}/{} canonical TTs ({:.0}%)",
            tested,
            total,
            tested as f64 / total as f64 * 100.0,
        );
        if !no_recipe.is_empty() {
            no_recipe.sort();
            eprintln!(
                "Missing recipes for first 20: {:?}",
                &no_recipe[..no_recipe.len().min(20)]
            )
        }
    }

    #[test]
    fn recipe_coverage_improved() {
        // After multi-variable Shannon decomposition, coverage should be >= 130/222.
        let mut canonicals = std::collections::HashSet::new();
        for tt in 0..=0xFFFFu16 {
            let npn = npn4_classify(tt);
            canonicals.insert(npn.canonical_tt);
        }
        let total = canonicals.len(); // 222
        let mut covered = 0;
        let mut trivial = 0;
        for &ct in &canonicals {
            if ct == 0x0000 || ct == 0xFFFF {
                trivial += 1;
                continue;
            }
            let is_single_var = (0..4u8).any(|i| {
                let v = var_truth_table(i);
                ct == v || ct == !v
            });
            if is_single_var {
                trivial += 1;
                continue;
            }
            if get_optimal_recipe(ct).is_some() {
                covered += 1;
            }
        }
        let non_trivial = total - trivial;
        eprintln!(
            "NPN4 recipe coverage: {}/{} non-trivial ({:.0}%), {}/{} total",
            covered,
            non_trivial,
            covered as f64 / non_trivial as f64 * 100.0,
            covered + trivial,
            total
        );
        assert!(
            covered >= 130,
            "Expected >= 130 non-trivial recipes, got {}",
            covered
        );
    }

    #[test]
    fn recipe_all_orderings_correct() {
        // For each covered canonical TT, verify recipe against all 16 inputs.
        let mut canonicals = std::collections::HashSet::new();
        for tt in 0..=0xFFFFu16 {
            let npn = npn4_classify(tt);
            canonicals.insert(npn.canonical_tt);
        }
        for &ct in &canonicals {
            if let Some(recipe) = get_optimal_recipe(ct) {
                assert!(
                    verify_recipe(&recipe, ct),
                    "Recipe for canonical 0x{:04X} is incorrect",
                    ct,
                );
            }
        }
    }

    #[test]
    fn rewrite_suboptimal_aig() {
        // Build a 4-input AND using a suboptimal deep chain: ((a&b)&c)&d
        // The rewriter may create orphaned nodes from intermediate replacements;
        // these get cleaned up by balance in the full optimize() pipeline.
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let c = aig.add_input("c");
        let d = aig.add_input("d");
        let ab = aig.and(a, b);
        let abc = aig.and(ab, c);
        let abcd = aig.and(abc, d);
        aig.add_output("y", abcd);

        assert_eq!(aig.num_nodes(), 3);

        let opt = super::super::optimize(&aig, &super::super::OptConfig::default());
        eprintln!(
            "4-AND chain optimized: {} -> {} nodes, depth {} -> {}",
            aig.num_nodes(),
            opt.num_nodes(),
            aig.depth(),
            opt.depth()
        );

        // Should not increase nodes
        assert!(
            opt.num_nodes() <= aig.num_nodes(),
            "Optimized should have <= nodes: {} vs {}",
            opt.num_nodes(),
            aig.num_nodes(),
        );

        // Verify equivalence
        for assignment in 0..16u32 {
            let inputs: Vec<bool> = (0..4).map(|i| (assignment >> i) & 1 != 0).collect();
            let orig = evaluate_aig(&aig, &inputs);
            let opt_out = evaluate_aig(&opt, &inputs);
            assert_eq!(orig, opt_out, "Mismatch at {:04b}", assignment);
        }
    }
}
