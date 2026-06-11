//! Delay-oriented technology mapping using K=4 cuts.
//!
//! Maps an AIG to tile-native cells via dynamic programming. For each AND node,
//! we select the K=4 cut that minimizes the critical path delay.
//!
//! Output is a `MappedNetlist` — gate instances + net connectivity with
//! polarity tracking. Every gate net carries the correct AIG variable value;
//! the `output_inverted` flag compensates when the cell implements the
//! negated truth table.
//!
//! No area accounting (DAG sharing makes naive sums wrong).
//! No placement coordinates — that's a separate problem.

use super::aig::Aig;
#[cfg(test)]
use super::aig::AigLit;
use super::cell_lib::{CellLibrary, tt_is_const0, tt_is_const1, tt_negate};
use super::cuts::{Cut, enumerate_cuts_with_limit};

// ---------------------------------------------------------------------------
// MappedNetlist
// ---------------------------------------------------------------------------

/// A gate instance in the mapped netlist.
///
/// The gate's net value is defined as:
///   `net_value = cell_function(fanin_net_values XOR fanin_inverted) XOR output_inverted`
///
/// This ensures every gate's net carries the correct AIG variable value,
/// regardless of complement edges on the AIG node's fanins.
#[derive(Clone, Debug)]
pub struct MappedGate {
    /// Index of the cell in the CellLibrary.
    pub cell_idx: usize,
    /// Net IDs feeding this gate's inputs (ordered to match cell pin order).
    pub fanins: Vec<u32>,
    /// Per-fanin inversion. If `fanin_inverted[i]` is true, the value on
    /// `fanins[i]` is complemented before entering the cell.
    /// Empty means no inversions (all false).
    pub fanin_inverted: Vec<bool>,
    /// If true, the physical gate output is complemented before driving the net.
    /// This compensates when the selected cell implements the negated truth table.
    pub output_inverted: bool,
    /// Optional name (for debugging).
    pub name: Option<String>,
}

/// A mapped netlist: the output of technology mapping.
///
/// Net value model:
/// - PI nets carry raw input values.
/// - Gate nets: `net[gate] = cell_function(fanin_nets) XOR gate.output_inverted`
/// - Primary output value: `output = net[net_id] XOR inverted`
#[derive(Clone, Debug)]
pub struct MappedNetlist {
    /// All gate instances.
    pub gates: Vec<MappedGate>,
    /// Primary input names (net IDs 0..num_inputs-1).
    pub primary_inputs: Vec<String>,
    /// Primary outputs: (name, net_id, inverted).
    /// `output_value = net[net_id] XOR inverted`.
    pub primary_outputs: Vec<(String, u32, bool)>,
    /// Total number of nets. Net IDs: 0..num_inputs-1 = PIs,
    /// then each gate produces a net at index num_inputs + gate_idx.
    pub num_nets: u32,
}

impl MappedNetlist {
    /// Number of gates.
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }
}

// ---------------------------------------------------------------------------
// Mapping configuration
// ---------------------------------------------------------------------------

/// Technology mapping configuration.
pub struct MapConfig {
    /// If true, prefer area reduction (not yet implemented — delay-only for now).
    pub area_oriented: bool,
    /// Per-node cut budget for enumeration. Larger budgets surface more cuts
    /// (e.g. deeply-expanded threshold cuts foldable into computational vias)
    /// at the cost of mapping time. Default is [`crate::synth::cuts::MAX_CUTS_PER_NODE`].
    pub cut_limit: usize,
}

impl Default for MapConfig {
    fn default() -> Self {
        MapConfig {
            area_oriented: false,
            cut_limit: crate::synth::cuts::MAX_CUTS_PER_NODE,
        }
    }
}

impl MapConfig {
    /// Mapping config tuned for active-via libraries: a larger cut budget so
    /// threshold cuts (majority, wide AND/OR over deep supports) survive
    /// enumeration and can collapse into single delay-1 vias.
    pub fn via_aware() -> Self {
        MapConfig {
            area_oriented: false,
            cut_limit: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Mapping state (DP)
// ---------------------------------------------------------------------------

/// Per-variable DP state during mapping.
#[derive(Clone, Debug)]
struct VarState {
    /// Best delay to reach this variable.
    arrival: f64,
    /// The cut selected for this variable (None for PIs).
    best_cut: Option<Cut>,
    /// Cell index selected for this cut (None for PIs or unmapped).
    cell_idx: Option<usize>,
    /// Whether the cell implements the negated truth table.
    /// When true, gate output = !cell_function, so net = AIG_value.
    inverted: bool,
    /// Per-fanin complement edges (from the AIG AND node).
    /// Non-empty only for fallback AND2 mapping where complement edges
    /// can't be absorbed into the truth table.
    fanin_inverted: Vec<bool>,
}

// ---------------------------------------------------------------------------
// map_to_library
// ---------------------------------------------------------------------------

/// Map an AIG to a cell library using delay-oriented DP.
///
/// Algorithm:
/// 1. Enumerate K=4 cuts for all AND nodes.
/// 2. For each variable (inputs: arrival=0, nodes: DP), select the cut
///    that minimizes `max(arrival[leaf]) + cell_delay`.
///    Cell lookup is arity-aware: only cells matching the cut's input count.
/// 3. Trace back from outputs to instantiate gates with polarity tracking.
pub fn map_to_library(aig: &Aig, lib: &CellLibrary, config: &MapConfig) -> MappedNetlist {
    let num_inputs = aig.num_inputs();
    let num_nodes = aig.num_nodes();
    let total_vars = num_inputs as usize + 1 + num_nodes; // var 0 = const

    // Step 1: enumerate cuts (cut budget from config — larger for via-aware maps)
    let all_cuts = enumerate_cuts_with_limit(aig, config.cut_limit);

    // Step 2: DP — compute best arrival for each variable
    let mut state = vec![
        VarState {
            arrival: 0.0,
            best_cut: None,
            cell_idx: None,
            inverted: false,
            fanin_inverted: Vec::new(),
        };
        total_vars
    ];

    // Primary inputs have arrival = 0 (already set)

    // Process AND nodes in topological order
    for (node_idx, _node) in aig.nodes_topo() {
        let var = aig.node_var(node_idx);
        let node_cuts = &all_cuts[node_idx];

        let mut best_arrival = f64::INFINITY;
        let mut best_cut_idx = None;
        let mut best_cell = None;
        let mut best_inverted = false;

        for (cut_idx, cut) in node_cuts.cuts.iter().enumerate() {
            // Skip trivial cuts (single leaf = self). These exist so parent
            // nodes can use this node as a leaf, but they don't represent
            // a valid mapping for this node itself (would be circular).
            if cut.num_leaves == 1 && cut.leaves[0] == var {
                continue;
            }

            if cut.num_leaves == 0 {
                // Constant cut — delay = 0
                if let Some(cell) = lib.best_cell_for(cut.truth_table, 0) {
                    if 0.0 < best_arrival {
                        best_arrival = 0.0;
                        best_cut_idx = Some(cut_idx);
                        best_cell = Some(cell);
                        best_inverted = false;
                    }
                }
                continue;
            }

            // Try both positive and negated truth tables
            for inverted in [false, true] {
                let tt = if inverted {
                    tt_negate(cut.truth_table, cut.num_leaves)
                } else {
                    cut.truth_table
                };

                // Skip constant truth tables — they don't need a gate
                if tt_is_const0(tt, cut.num_leaves) || tt_is_const1(tt, cut.num_leaves) {
                    continue;
                }

                // Arity-aware cell lookup: only match cells with the right input count
                let cell_idx = lib.best_cell_for(tt, cut.num_leaves);
                if cell_idx.is_none() {
                    continue;
                }
                let cell_idx = cell_idx.unwrap();
                let cell = lib.cell(cell_idx);

                // Compute arrival: max(arrival[leaf]) + cell_delay
                let mut max_leaf_arrival = 0.0f64;
                for &leaf in cut.leaf_slice() {
                    if (leaf as usize) < state.len() {
                        max_leaf_arrival = max_leaf_arrival.max(state[leaf as usize].arrival);
                    }
                }
                let arrival = max_leaf_arrival + cell.delay as f64;

                if arrival < best_arrival {
                    best_arrival = arrival;
                    best_cut_idx = Some(cut_idx);
                    best_cell = Some(cell_idx);
                    best_inverted = inverted;
                }
            }
        }

        // Fallback: if no cut matched a cell (even negated), decompose using
        // the AND node's raw 2-input AND function with per-fanin complement edges.
        // Every AIG AND node computes `fanin0 & fanin1` where each fanin may be
        // complemented. We map to AND2 and record the complement edges in
        // `fanin_inverted`, which the evaluator applies before the cell function.
        if best_cell.is_none() {
            let node = aig.node(node_idx);
            let f0_var = node.fanin0.var();
            let f1_var = node.fanin1.var();
            let f0_arrival = if (f0_var as usize) < state.len() {
                state[f0_var as usize].arrival
            } else {
                0.0
            };
            let f1_arrival = if (f1_var as usize) < state.len() {
                state[f1_var as usize].arrival
            } else {
                0.0
            };

            // AND2 cell: raw 2-input AND (tt=0x8). Complement edges handled
            // by fanin_inverted, not by the truth table.
            let and2_idx = lib.best_cell_for(0x8, 2).expect("AND2 must exist");
            let delay = lib.cell(and2_idx).delay as f64;

            // Sort fanins by variable ID for canonical ordering
            let (lo, hi, inv_lo, inv_hi) = if f0_var <= f1_var {
                (
                    f0_var,
                    f1_var,
                    node.fanin0.is_complement(),
                    node.fanin1.is_complement(),
                )
            } else {
                (
                    f1_var,
                    f0_var,
                    node.fanin1.is_complement(),
                    node.fanin0.is_complement(),
                )
            };

            state[var as usize].best_cut = Some(Cut {
                leaves: [lo, hi, 0, 0],
                num_leaves: 2,
                truth_table: 0x8, // Raw AND — complements in fanin_inverted
            });
            state[var as usize].arrival = f0_arrival.max(f1_arrival) + delay;
            state[var as usize].cell_idx = Some(and2_idx);
            state[var as usize].inverted = false;
            state[var as usize].fanin_inverted = vec![inv_lo, inv_hi];
        } else {
            state[var as usize].arrival = best_arrival;
            state[var as usize].best_cut = best_cut_idx.map(|i| node_cuts.cuts[i].clone());
            state[var as usize].cell_idx = best_cell;
            state[var as usize].inverted = best_inverted;
        }
    }

    // Step 3: Trace back from outputs to build the netlist
    trace_back(aig, &state)
}

/// Trace back from outputs, instantiating gates for selected cuts.
///
/// Each gate's `output_inverted` is set from `state[var].inverted`.
/// This ensures every gate net carries the correct AIG variable value.
/// Primary output `inverted` is simply the AIG output literal's complement bit.
fn trace_back(aig: &Aig, state: &[VarState]) -> MappedNetlist {
    let num_inputs = aig.num_inputs();

    // Collect primary input names
    let primary_inputs: Vec<String> = aig
        .inputs()
        .iter()
        .flat_map(|p| {
            p.bits.iter().enumerate().map(move |(i, _)| {
                if p.bits.len() == 1 {
                    p.name.clone()
                } else {
                    format!("{}[{}]", p.name, i)
                }
            })
        })
        .collect();

    let mut gates: Vec<MappedGate> = Vec::new();
    let mut var_to_net: Vec<Option<u32>> = vec![None; state.len()];

    // Assign PI net IDs (var 1..=num_inputs → net 0..num_inputs-1)
    for var in 1..=num_inputs {
        var_to_net[var as usize] = Some(var - 1);
    }
    // Constant net (sentinel)
    var_to_net[0] = Some(u32::MAX);

    let mut next_net_id = num_inputs;

    // Collect all variables referenced by outputs (BFS into cut leaves)
    let mut work: Vec<u32> = Vec::new();
    let mut visited = vec![false; state.len()];

    let mut primary_outputs: Vec<(String, u32, bool)> = Vec::new();

    for port in aig.outputs() {
        for (i, &lit) in port.bits.iter().enumerate() {
            let name = if port.bits.len() == 1 {
                port.name.clone()
            } else {
                format!("{}[{}]", port.name, i)
            };
            let var = lit.var();

            if var == 0 {
                // Constant output: inverted = lit complement
                primary_outputs.push((name, u32::MAX, lit.is_complement()));
                continue;
            }

            if !visited[var as usize] && var > num_inputs {
                work.push(var);
                visited[var as usize] = true;
            }
            // Output inverted = literal's complement bit.
            // Gate net already carries AIG_var value (output_inverted handles it).
            primary_outputs.push((name, var, lit.is_complement()));
        }
    }

    // BFS: discover all needed variables by recursing into cut leaves
    let mut queue_idx = 0;
    while queue_idx < work.len() {
        let var = work[queue_idx];
        queue_idx += 1;

        let vs = &state[var as usize];
        if let Some(ref cut) = vs.best_cut {
            for &leaf in cut.leaf_slice() {
                if leaf > num_inputs && !visited[leaf as usize] {
                    work.push(leaf);
                    visited[leaf as usize] = true;
                }
            }
        }
    }

    // Instantiate gates in topological order.
    // AIG variable IDs are already in topological order by construction
    // (var = num_inputs + 1 + node_idx, where nodes are topo-ordered).
    // Simple sort is correct and avoids BFS-reversal ordering bugs
    // that occur when multiple output cones share internal nodes.
    work.sort_unstable();
    for &var in work.iter() {
        let vs = &state[var as usize];
        if let (Some(cut), Some(cell_idx)) = (&vs.best_cut, vs.cell_idx) {
            let mut fanins = Vec::with_capacity(cut.num_leaves as usize);
            for &leaf in cut.leaf_slice() {
                let net_id = match var_to_net[leaf as usize] {
                    Some(id) => id,
                    None => {
                        if leaf <= num_inputs {
                            leaf - 1
                        } else {
                            0 // fallback (shouldn't happen)
                        }
                    }
                };
                fanins.push(net_id);
            }

            let gate_net_id = next_net_id;
            next_net_id += 1;
            var_to_net[var as usize] = Some(gate_net_id);

            gates.push(MappedGate {
                cell_idx,
                fanins,
                fanin_inverted: vs.fanin_inverted.clone(),
                output_inverted: vs.inverted,
                name: Some(format!("g{}", var)),
            });
        } else {
            // No mapping found — assign a net anyway
            let gate_net_id = next_net_id;
            next_net_id += 1;
            var_to_net[var as usize] = Some(gate_net_id);
        }
    }

    // Resolve output net IDs
    let resolved_outputs: Vec<(String, u32, bool)> = primary_outputs
        .into_iter()
        .map(|(name, var, inv)| {
            if var == u32::MAX {
                (name, u32::MAX, inv)
            } else {
                let net_id = var_to_net[var as usize].unwrap_or(var - 1);
                (name, net_id, inv)
            }
        })
        .collect();

    MappedNetlist {
        gates,
        primary_inputs,
        primary_outputs: resolved_outputs,
        num_nets: next_net_id,
    }
}

// ---------------------------------------------------------------------------
// Netlist evaluator (for equivalence testing)
// ---------------------------------------------------------------------------

/// Evaluate a mapped netlist for a given set of primary input values.
///
/// Returns the primary output values as a Vec of bools.
///
/// Net value model:
/// - PI nets: `net[i] = inputs[i]`
/// - Gate nets: `net[gate] = cell_tt_bit(fanin_values) XOR output_inverted`
/// - Output: `net[net_id] XOR inverted`
pub fn evaluate_netlist(
    netlist: &MappedNetlist,
    lib: &CellLibrary,
    input_values: &[bool],
) -> Vec<bool> {
    assert_eq!(input_values.len(), netlist.primary_inputs.len());

    let mut net_values = vec![false; netlist.num_nets as usize];

    // Set PI net values
    for (i, &val) in input_values.iter().enumerate() {
        net_values[i] = val;
    }

    // Evaluate gates in order (they are in topological order from trace-back)
    let num_pi = netlist.primary_inputs.len();
    for (gate_idx, gate) in netlist.gates.iter().enumerate() {
        let cell = lib.cell(gate.cell_idx);

        // Build input assignment from fanin net values, applying per-fanin inversion
        let mut assignment = 0u32;
        for (pin_idx, &fanin_net) in gate.fanins.iter().enumerate() {
            let raw = fanin_net < netlist.num_nets && net_values[fanin_net as usize];
            let inv = gate.fanin_inverted.get(pin_idx).copied().unwrap_or(false);
            if raw ^ inv {
                assignment |= 1 << pin_idx;
            }
        }

        // Look up truth table bit
        let cell_output = (cell.truth_table >> assignment) & 1 != 0;

        // Apply output inversion
        let net_value = cell_output ^ gate.output_inverted;

        let net_id = num_pi + gate_idx;
        if net_id < net_values.len() {
            net_values[net_id] = net_value;
        }
    }

    // Compute primary output values
    netlist
        .primary_outputs
        .iter()
        .map(|(_, net_id, inverted)| {
            if *net_id == u32::MAX {
                // Constant: false XOR inverted
                *inverted
            } else {
                let val = if (*net_id as usize) < net_values.len() {
                    net_values[*net_id as usize]
                } else {
                    false
                };
                val ^ inverted
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// AIG evaluator (reference for equivalence testing)
// ---------------------------------------------------------------------------

/// Evaluate an AIG for a given set of primary input values.
/// Returns the primary output values as a Vec of bools.
pub fn evaluate_aig(aig: &Aig, input_values: &[bool]) -> Vec<bool> {
    assert_eq!(input_values.len(), aig.num_inputs() as usize);

    let total_vars = aig.num_inputs() as usize + 1 + aig.num_nodes();
    let mut var_values = vec![false; total_vars];

    // Variable 0 = constant FALSE (already false)

    // Set input variables
    for (i, &val) in input_values.iter().enumerate() {
        var_values[i + 1] = val; // var 1..=num_inputs
    }

    // Evaluate AND nodes in topological order
    for (node_idx, node) in aig.nodes_topo() {
        let v0 = var_values[node.fanin0.var() as usize] ^ node.fanin0.is_complement();
        let v1 = var_values[node.fanin1.var() as usize] ^ node.fanin1.is_complement();
        let var = aig.node_var(node_idx);
        var_values[var as usize] = v0 && v1;
    }

    // Compute output values
    aig.output_lits()
        .map(|lit| {
            if lit.is_const() {
                lit.is_complement() // FALSE=false, TRUE=true
            } else {
                var_values[lit.var() as usize] ^ lit.is_complement()
            }
        })
        .collect()
}

/// Verify that a mapped netlist computes the same function as the AIG.
///
/// Exhaustive for n <= 20 inputs. For larger circuits, uses deterministic
/// sampling (10000 stride-based patterns).
pub fn verify_equivalence(aig: &Aig, netlist: &MappedNetlist, lib: &CellLibrary) -> bool {
    let n = aig.num_inputs() as usize;

    if n <= 20 {
        for assignment in 0..(1u32 << n) {
            let inputs: Vec<bool> = (0..n).map(|i| (assignment >> i) & 1 != 0).collect();

            let aig_outputs = evaluate_aig(aig, &inputs);
            let netlist_outputs = evaluate_netlist(netlist, lib, &inputs);

            if aig_outputs != netlist_outputs {
                eprintln!(
                    "Mismatch at input {:0width$b}: AIG={:?} netlist={:?}",
                    assignment,
                    aig_outputs,
                    netlist_outputs,
                    width = n,
                );
                return false;
            }
        }
    } else {
        let total_combos: u64 = 1u64 << n.min(63);
        let num_samples: u64 = 10_000;
        let stride = total_combos / num_samples;

        for sample in 0..num_samples {
            let assignment = sample * stride;
            let inputs: Vec<bool> = (0..n).map(|i| (assignment >> i) & 1 != 0).collect();

            let aig_outputs = evaluate_aig(aig, &inputs);
            let netlist_outputs = evaluate_netlist(netlist, lib, &inputs);

            if aig_outputs != netlist_outputs {
                eprintln!(
                    "Mismatch at input {}: AIG={:?} netlist={:?}",
                    assignment, aig_outputs, netlist_outputs,
                );
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::aig::Aig;
    use crate::synth::cell_lib::CellLibrary;

    // ===== Structural tests =====

    #[test]
    fn map_single_and() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());

        assert!(!netlist.gates.is_empty(), "should have at least 1 gate");
        assert_eq!(netlist.primary_inputs.len(), 2);
        assert_eq!(netlist.primary_outputs.len(), 1);

        let gate = &netlist.gates[0];
        let cell = lib.cell(gate.cell_idx);
        assert_eq!(cell.name, "AND2");
        assert!(!gate.output_inverted);
    }

    #[test]
    fn map_or_uses_correct_cell() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.or(a, b);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());

        // OR = !(!a & !b). The AND node computes NOR (tt=0x1 for 2 inputs).
        // Mapper should select OR2 (tt=0xE) with output_inverted=true,
        // or find the NOR truth table directly.
        assert!(!netlist.gates.is_empty());
    }

    #[test]
    fn map_constant_passthrough() {
        let mut aig = Aig::new();
        let _a = aig.add_input("a");
        aig.add_output("y", AigLit::TRUE);

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());

        assert_eq!(netlist.primary_outputs.len(), 1);
        let (_, _, inv) = &netlist.primary_outputs[0];
        assert!(*inv, "TRUE should be inverted FALSE");
    }

    // ===== Equivalence tests (exhaustive truth table verification) =====

    #[test]
    fn equiv_and() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(verify_equivalence(&aig, &netlist, &lib));
    }

    #[test]
    fn equiv_or() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.or(a, b);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(verify_equivalence(&aig, &netlist, &lib));
    }

    #[test]
    fn equiv_xor() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.xor(a, b);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(verify_equivalence(&aig, &netlist, &lib));
    }

    #[test]
    fn equiv_not() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        aig.add_output("y", aig.not(a));

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(verify_equivalence(&aig, &netlist, &lib));
    }

    #[test]
    fn equiv_mux() {
        let mut aig = Aig::new();
        let s = aig.add_input("s");
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.mux(s, a, b);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(verify_equivalence(&aig, &netlist, &lib));
    }

    #[test]
    fn equiv_4input_and_chain() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let c = aig.add_input("c");
        let d = aig.add_input("d");
        let ab = aig.and(a, b);
        let cd = aig.and(c, d);
        let y = aig.and(ab, cd);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(verify_equivalence(&aig, &netlist, &lib));
    }

    #[test]
    fn equiv_nand_nor_patterns() {
        // NAND: !(a & b)
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let ab = aig.and(a, b);
        aig.add_output("y", aig.not(ab));

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(verify_equivalence(&aig, &netlist, &lib));
    }

    #[test]
    fn equiv_nor_pattern() {
        // NOR: !(a | b)
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let ab_or = aig.or(a, b);
        aig.add_output("y", aig.not(ab_or));

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(verify_equivalence(&aig, &netlist, &lib));
    }

    #[test]
    fn equiv_multi_output() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let c = aig.add_input("c");
        let y1 = aig.and(a, b);
        let y2 = aig.or(b, c);
        let y3 = aig.xor(a, c);
        aig.add_output("y1", y1);
        aig.add_output("y2", y2);
        aig.add_output("y3", y3);

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(verify_equivalence(&aig, &netlist, &lib));
    }

    #[test]
    fn equiv_4bit_adder() {
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

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(
            verify_equivalence(&aig, &netlist, &lib),
            "4-bit adder equivalence failed"
        );
    }

    #[test]
    fn equiv_constant_true() {
        let mut aig = Aig::new();
        let _a = aig.add_input("a");
        aig.add_output("y", AigLit::TRUE);

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(verify_equivalence(&aig, &netlist, &lib));
    }

    #[test]
    fn equiv_constant_false() {
        let mut aig = Aig::new();
        let _a = aig.add_input("a");
        aig.add_output("y", AigLit::FALSE);

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(verify_equivalence(&aig, &netlist, &lib));
    }

    #[test]
    fn equiv_passthrough() {
        // Output = input directly (no gates)
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        aig.add_output("y", a);

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(verify_equivalence(&aig, &netlist, &lib));
    }

    #[test]
    fn equiv_inverted_passthrough() {
        // Output = !input (complement, no AND nodes)
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        aig.add_output("y", aig.not(a));

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert!(verify_equivalence(&aig, &netlist, &lib));
    }

    // ===== AIG evaluator tests =====

    #[test]
    fn aig_eval_and() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);

        assert_eq!(evaluate_aig(&aig, &[false, false]), vec![false]);
        assert_eq!(evaluate_aig(&aig, &[true, false]), vec![false]);
        assert_eq!(evaluate_aig(&aig, &[false, true]), vec![false]);
        assert_eq!(evaluate_aig(&aig, &[true, true]), vec![true]);
    }

    #[test]
    fn aig_eval_or() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.or(a, b);
        aig.add_output("y", y);

        assert_eq!(evaluate_aig(&aig, &[false, false]), vec![false]);
        assert_eq!(evaluate_aig(&aig, &[true, false]), vec![true]);
        assert_eq!(evaluate_aig(&aig, &[false, true]), vec![true]);
        assert_eq!(evaluate_aig(&aig, &[true, true]), vec![true]);
    }

    #[test]
    fn aig_eval_xor() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.xor(a, b);
        aig.add_output("y", y);

        assert_eq!(evaluate_aig(&aig, &[false, false]), vec![false]);
        assert_eq!(evaluate_aig(&aig, &[true, false]), vec![true]);
        assert_eq!(evaluate_aig(&aig, &[false, true]), vec![true]);
        assert_eq!(evaluate_aig(&aig, &[true, true]), vec![false]);
    }
}
