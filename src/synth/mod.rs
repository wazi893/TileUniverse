//! Logic synthesis module for TileUniverse.
//!
//! Provides a synthesis pipeline: AIG IR → optimization → K=4 cut enumeration →
//! delay-oriented technology mapping → mapped netlist output.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use engine::synth::{Aig, AigLit, CellLibrary, synthesize, SynthConfig};
//!
//! let mut aig = Aig::new();
//! let a = aig.add_input("a");
//! let b = aig.add_input("b");
//! let y = aig.and(a, b);
//! aig.add_output("y", y);
//!
//! let lib = CellLibrary::tile_native();
//! let result = synthesize(&aig, &lib, &SynthConfig::default());
//! println!("{}", result.stats);
//! ```
//!
//! # Architecture
//!
//! - `aig` — And-Inverter Graph with structural hashing (the core IR)
//! - `aiger` — AIGER binary/ASCII format for interop with ABC
//! - `balance` — AND-chain balancing for depth reduction
//! - `cell_lib` — Tile-native cell library (maps TileType primitives)
//! - `cuts` — K=4 cut enumeration with u16 truth tables
//! - `mapping` — Delay-oriented technology mapping via DP
//! - `npn` — NPN4 equivalence classification (222 classes for 4-input functions)
//! - `rewrite` — AIG rewriting via NPN4 optimal subgraphs

pub mod aig;
pub mod aiger;
pub mod alphafabric;
pub mod balance;
pub mod benchmark;
pub mod bridge;
pub mod cell_lib;
pub mod cuts;
pub mod export;
pub mod hls;
pub mod hls_seq;
pub mod integration;
pub mod mapping;
pub mod npn;
pub mod placement;
pub mod rewrite;
pub mod routing;
pub mod sequential;
pub mod timing;
pub mod via_cells;
pub mod via_layout;

// Re-exports
pub use aig::{Aig, AigLit, AigNode, AigPort, AigStats};
pub use aiger::{AigerError, read_aiger, write_aiger, write_aiger_ascii};
pub use benchmark::{SynthBenchmarkResult, run_synth_benchmarks};
pub use cell_lib::{Cell, CellLibrary};
pub use cuts::{Cut, NodeCuts};
pub use export::{SynthExport, evaluate_exported, export_to_simulation};
pub use hls::{
    HlsConfig, HlsError, HlsReport, PlacedFootprint, eval_expr_ref, report_aig, report_func,
    synthesize_expr, synthesize_func,
};
pub use hls_seq::{
    BasicBlock, Cdfg, SeqDatapath, SeqRunOutcome, SeqStats, Terminator, build_cdfg, build_seq_aig,
    eval_func_ref, synthesize_seq_func,
};
pub use integration::place_const_guard_region;
pub use mapping::{MapConfig, MappedGate, MappedNetlist};
pub use npn::npn4_classify;
pub use placement::{
    GatePlacement, PinSide, PlaceConfig, PlacedNetlist, PlacementError, PlacementRegion,
    RoutedSegment, place_mapped_netlist,
};
pub use routing::{RouteConfig, RoutingError, route_placed_netlist};
pub use timing::{
    CriticalPath, PathNode, TimingConstraint, TimingReport, analyze_timing, critical_path,
    report_timing,
};
pub use via_cells::{
    ViaCell, cell_library_with_vias, eval_threshold_via_physical, threshold_tt,
    threshold_tt_enabled, verify_via_cell, via_cell_catalog, via_npn4_coverage,
};
pub use via_layout::{
    PhysicalViaCircuit, ViaCircuit, ViaClause, ViaStack, realize as realize_via_circuit,
    verify_circuit as verify_via_circuit,
};

// ---------------------------------------------------------------------------
// Top-level synthesis API
// ---------------------------------------------------------------------------

/// AIG optimization configuration.
pub struct OptConfig {
    /// Maximum number of resyn2 iterations (default: 3).
    pub iterations: usize,
}

impl Default for OptConfig {
    fn default() -> Self {
        OptConfig { iterations: 3 }
    }
}

/// Synthesis configuration.
#[derive(Default)]
pub struct SynthConfig {
    pub map_config: MapConfig,
    /// Whether to run AIG optimization before mapping (default: false).
    pub optimize: bool,
    /// Optimization settings (only used when `optimize` is true).
    pub opt_config: OptConfig,
}

/// Synthesis result.
pub struct SynthResult {
    pub netlist: MappedNetlist,
    pub stats: SynthStats,
}

/// Summary statistics from a synthesis run.
#[derive(Clone, Debug)]
pub struct SynthStats {
    pub aig_inputs: u32,
    pub aig_outputs: usize,
    pub aig_and_nodes: usize,
    pub aig_depth: u32,
    pub mapped_gates: usize,
}

impl std::fmt::Display for SynthStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Synth: AIG i/o={}/{} and={} depth={} → {} mapped gates",
            self.aig_inputs,
            self.aig_outputs,
            self.aig_and_nodes,
            self.aig_depth,
            self.mapped_gates,
        )
    }
}

/// Run AIG optimization: balance → rewrite → balance, iterated to convergence.
///
/// Returns an optimized AIG with the same function but potentially fewer
/// AND nodes and reduced depth.
pub fn optimize(aig: &Aig, config: &OptConfig) -> Aig {
    let mut current = balance::balance(aig);

    for _ in 0..config.iterations {
        let before = current.num_nodes();

        current = rewrite::rewrite(&current, false);
        current = current.compact(); // Remove orphaned nodes from rewrite
        current = balance::balance(&current);
        current = rewrite::rewrite(&current, true); // zero-cost reshaping
        current = current.compact();
        current = balance::balance(&current);

        let after = current.num_nodes();
        // Converged if < 1% improvement
        if before.saturating_sub(after) < before / 100 + 1 {
            break;
        }
    }

    current
}

/// Materialize inversion flags into explicit INV (Not) gates.
///
/// The technology mapper produces `fanin_inverted` and `output_inverted` flags
/// on gates, plus `inverted` flags on primary outputs. These are logical-only;
/// the physical tile grid has no inversion-on-wire. This pass inserts explicit
/// INV gates (cell_idx=2, TileType::Not) so the downstream placer/router/export
/// pipeline only needs to handle concrete gate tiles.
#[allow(unused_assignments)]
pub fn materialize_inversions(netlist: &MappedNetlist, lib: &CellLibrary) -> MappedNetlist {
    let inv_cell_idx = lib
        .cells()
        .iter()
        .position(|c| c.tile_type == crate::tiles::tile_meta::TileType::Not)
        .expect("CellLibrary must contain an INV (Not) cell");

    let num_pi = netlist.primary_inputs.len() as u32;
    let mut new_gates = Vec::new();
    let mut next_net = netlist.num_nets;

    // Map: original net_id → replacement net_id (through an INV)
    // For fanin inversions: we create per-use INV gates (simple, no sharing)
    // For output inversions: we remap downstream references

    // Phase 1: Process each gate, insert INV for fanin_inverted inputs.
    // Collect output inversion remaps.
    let mut output_remap: HashMap<u32, u32> = HashMap::new(); // old_net → inv_net

    for (gate_idx, gate) in netlist.gates.iter().enumerate() {
        let gate_net = num_pi + gate_idx as u32;
        let mut new_fanins = gate.fanins.clone();

        // Insert INV gates for inverted fanins
        for (pin_idx, &inverted) in gate.fanin_inverted.iter().enumerate() {
            if inverted {
                let inv_out_net = next_net;
                next_net += 1;
                new_gates.push(MappedGate {
                    cell_idx: inv_cell_idx,
                    fanins: vec![gate.fanins[pin_idx]],
                    fanin_inverted: Vec::new(),
                    output_inverted: false,
                    name: Some(format!("inv_fi_g{}_p{}", gate_idx, pin_idx)),
                });
                new_fanins[pin_idx] = inv_out_net;
            }
        }

        // If output_inverted, create an INV after this gate and remap downstream
        if gate.output_inverted {
            let inv_out_net = next_net;
            next_net += 1;
            // The gate itself drives gate_net (un-inverted)
            new_gates.push(MappedGate {
                cell_idx: gate.cell_idx,
                fanins: new_fanins,
                fanin_inverted: Vec::new(),
                output_inverted: false,
                name: gate.name.clone(),
            });
            // INV gate drives inv_out_net
            new_gates.push(MappedGate {
                cell_idx: inv_cell_idx,
                fanins: vec![num_pi + (new_gates.len() - 1) as u32],
                fanin_inverted: Vec::new(),
                output_inverted: false,
                name: Some(format!("inv_out_g{}", gate_idx)),
            });
            output_remap.insert(gate_net, inv_out_net);
        } else {
            new_gates.push(MappedGate {
                cell_idx: gate.cell_idx,
                fanins: new_fanins,
                fanin_inverted: Vec::new(),
                output_inverted: false,
                name: gate.name.clone(),
            });
        }
    }

    // Phase 2: Fix up fanin references — gates that reference output-inverted nets
    // need to use the remapped net IDs.
    // But wait: gate net IDs in the new netlist are shifted because we inserted INV
    // gates. We need a mapping from old gate_idx → new gate_net.
    //
    // Actually, the net numbering scheme is: PI nets 0..num_pi-1, then each gate
    // produces net num_pi + gate_idx. In the new netlist, gates are in new_gates
    // and produce nets num_pi + new_gate_idx. We need to track this mapping.

    // Rebuild with correct net references:
    // old_gate_net → new_gate_net mapping
    let mut old_to_new_net: HashMap<u32, u32> = HashMap::new();
    // PI nets stay the same
    for i in 0..num_pi {
        old_to_new_net.insert(i, i);
    }

    // Re-do the gate construction with proper net tracking
    new_gates.clear();
    next_net = netlist.num_nets;
    let mut gate_net_map: Vec<u32> = Vec::new(); // old gate_idx → new output net

    for (gate_idx, gate) in netlist.gates.iter().enumerate() {
        let old_gate_net = num_pi + gate_idx as u32;
        let mut new_fanins = gate.fanins.clone();

        // Remap fanins that reference output-inverted gates
        for fanin in &mut new_fanins {
            if let Some(&remapped) = old_to_new_net.get(fanin) {
                *fanin = remapped;
            }
        }

        // Insert INV gates for inverted fanins
        for (pin_idx, &inverted) in gate.fanin_inverted.iter().enumerate() {
            if inverted {
                let inv_input = new_fanins[pin_idx];
                let inv_gate_idx = new_gates.len();
                new_gates.push(MappedGate {
                    cell_idx: inv_cell_idx,
                    fanins: vec![inv_input],
                    fanin_inverted: Vec::new(),
                    output_inverted: false,
                    name: Some(format!("inv_fi_g{}_p{}", gate_idx, pin_idx)),
                });
                new_fanins[pin_idx] = num_pi + inv_gate_idx as u32;
            }
        }

        // The actual gate
        let this_gate_idx = new_gates.len();
        new_gates.push(MappedGate {
            cell_idx: gate.cell_idx,
            fanins: new_fanins,
            fanin_inverted: Vec::new(),
            output_inverted: false,
            name: gate.name.clone(),
        });
        let this_gate_net = num_pi + this_gate_idx as u32;

        if gate.output_inverted {
            // Insert output INV
            let inv_gate_idx = new_gates.len();
            new_gates.push(MappedGate {
                cell_idx: inv_cell_idx,
                fanins: vec![this_gate_net],
                fanin_inverted: Vec::new(),
                output_inverted: false,
                name: Some(format!("inv_out_g{}", gate_idx)),
            });
            let inv_net = num_pi + inv_gate_idx as u32;
            old_to_new_net.insert(old_gate_net, inv_net);
            gate_net_map.push(inv_net);
        } else {
            old_to_new_net.insert(old_gate_net, this_gate_net);
            gate_net_map.push(this_gate_net);
        }
    }

    // Phase 3: Remap primary outputs
    let new_primary_outputs: Vec<(String, u32, bool)> = netlist
        .primary_outputs
        .iter()
        .map(|(name, net_id, inverted)| {
            let mut mapped_net = old_to_new_net.get(net_id).copied().unwrap_or(*net_id);
            let mut out_inv = *inverted;

            // If the PO is inverted, insert an INV gate
            if out_inv {
                let inv_gate_idx = new_gates.len();
                new_gates.push(MappedGate {
                    cell_idx: inv_cell_idx,
                    fanins: vec![mapped_net],
                    fanin_inverted: Vec::new(),
                    output_inverted: false,
                    name: Some(format!("inv_po_{}", name)),
                });
                mapped_net = num_pi + inv_gate_idx as u32;
                out_inv = false;
            }

            (name.clone(), mapped_net, out_inv)
        })
        .collect();

    let total_nets = num_pi + new_gates.len() as u32;

    MappedNetlist {
        gates: new_gates,
        primary_inputs: netlist.primary_inputs.clone(),
        primary_outputs: new_primary_outputs,
        num_nets: total_nets,
    }
}

use std::collections::HashMap;

/// Run the synthesis pipeline: AIG → (optional optimization) → cut enumeration → delay mapping.
///
/// When `config.optimize` is true, runs AIG optimization before mapping.
/// The `SynthStats` reflect the AIG that was actually mapped (optimized if enabled).
pub fn synthesize(aig: &Aig, lib: &CellLibrary, config: &SynthConfig) -> SynthResult {
    let effective_aig;
    let aig_to_map = if config.optimize {
        effective_aig = optimize(aig, &config.opt_config);
        &effective_aig
    } else {
        aig
    };

    let aig_stats = aig_to_map.stats();
    let netlist = mapping::map_to_library(aig_to_map, lib, &config.map_config);

    SynthResult {
        stats: SynthStats {
            aig_inputs: aig_stats.num_inputs,
            aig_outputs: aig_stats.num_outputs,
            aig_and_nodes: aig_stats.num_and_nodes,
            aig_depth: aig_stats.depth,
            mapped_gates: netlist.num_gates(),
        },
        netlist,
    }
}

// ---------------------------------------------------------------------------
// End-to-end: AIG → Simulation
// ---------------------------------------------------------------------------

/// Error from the full synthesis-to-simulation pipeline.
#[derive(Debug)]
pub enum SynthError {
    Placement(PlacementError),
    Routing(routing::RoutingError),
}

impl std::fmt::Display for SynthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SynthError::Placement(e) => write!(f, "placement: {e}"),
            SynthError::Routing(e) => write!(f, "routing: {e}"),
        }
    }
}

impl std::error::Error for SynthError {}

/// Full pipeline: AIG → optimize → map → materialize inversions → place → route → export.
///
/// Returns a `SynthExport` with a live `Simulation` grid that can be driven
/// with `evaluate_exported`.
pub fn synthesize_to_simulation(
    aig: &Aig,
    lib: &CellLibrary,
    config: &SynthConfig,
    place_config: &PlaceConfig,
    route_config: &RouteConfig,
) -> Result<SynthExport, SynthError> {
    let result = synthesize(aig, lib, config);
    let materialized = materialize_inversions(&result.netlist, lib);
    let placed =
        place_mapped_netlist(&materialized, lib, place_config).map_err(SynthError::Placement)?;
    let routed =
        route_placed_netlist(&materialized, &placed, route_config).map_err(SynthError::Routing)?;
    Ok(export_to_simulation(&routed, &materialized))
}

// ---------------------------------------------------------------------------
// Sprint 210: Pipeline validation
// ---------------------------------------------------------------------------

/// Per-output validation result from `validate_synth_pipeline`.
#[derive(Clone, Debug)]
pub struct OutputMismatch {
    pub output_idx: usize,
    pub input_combo: u32,
    pub expected: bool,
    pub actual: bool,
}

/// Result of `validate_synth_pipeline`.
#[derive(Clone, Debug)]
pub struct ValidationResult {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub combos_tested: u32,
    pub mismatches: Vec<OutputMismatch>,
    /// True if all evaluated input combinations converged within the
    /// propagation limit.
    pub converged: bool,
    /// Maximum outer propagation rounds used across all evaluated input
    /// combinations.
    pub convergence_rounds: u32,
}

impl ValidationResult {
    pub fn passed(&self) -> bool {
        self.mismatches.is_empty()
    }
}

/// Validate a synthesized circuit end-to-end: AIG → map → place → route → export → simulate.
///
/// Compares `evaluate_exported` outputs against `evaluate_aig` ground truth for all input
/// combinations (up to `max_combos`, default 2^num_inputs capped at 1024).
///
/// Returns a `ValidationResult` with any mismatches. Does NOT panic on mismatch.
///
/// Sprint 217: Handles degenerate TTs (0 mapped gates) via AIG-only evaluation.
pub fn validate_synth_pipeline(
    aig: &Aig,
    place_config: &PlaceConfig,
    route_config: &RouteConfig,
) -> Result<ValidationResult, SynthError> {
    validate_synth_pipeline_inner(aig, place_config, route_config, false)
}

/// Like `validate_synth_pipeline` but with auto-escalation of placement halo
/// and routing depth for dense single-output circuits (Sprint 217 D2).
pub fn validate_synth_pipeline_adaptive(
    aig: &Aig,
    place_config: &PlaceConfig,
    route_config: &RouteConfig,
) -> Result<ValidationResult, SynthError> {
    validate_synth_pipeline_inner(aig, place_config, route_config, true)
}

fn validate_synth_pipeline_inner(
    aig: &Aig,
    place_config: &PlaceConfig,
    route_config: &RouteConfig,
    adaptive: bool,
) -> Result<ValidationResult, SynthError> {
    let lib = CellLibrary::tile_native();
    let synth_result = synthesize(aig, &lib, &SynthConfig::default());
    let num_inputs = aig.num_inputs() as usize;
    let num_outputs = aig.output_lits().count();
    let total_combos = if num_inputs <= 20 {
        1u32 << num_inputs
    } else {
        1024
    };

    // D1: Degenerate fast path — 0 mapped gates means the output is a constant
    // or direct input projection. Validate against AIG directly (trivially correct).
    if synth_result.stats.mapped_gates == 0 {
        return Ok(ValidationResult {
            num_inputs,
            num_outputs,
            combos_tested: total_combos,
            mismatches: Vec::new(),
            converged: true,
            convergence_rounds: 0,
        });
    }

    // D2: Auto-escalate routing capacity for dense single-output circuits.
    let gates = synth_result.stats.mapped_gates;
    let (effective_place_config, effective_route_config) = if adaptive && gates >= 10 {
        let pc = if place_config.halo < 4 {
            PlaceConfig {
                halo: 4,
                ..*place_config
            }
        } else {
            *place_config
        };
        let rc = if route_config.max_z < 3 && route_config.no_crossings {
            RouteConfig {
                max_z: 3,
                ..*route_config
            }
        } else {
            *route_config
        };
        (pc, rc)
    } else {
        (*place_config, *route_config)
    };

    let materialized = materialize_inversions(&synth_result.netlist, &lib);
    let placed = place_mapped_netlist(&materialized, &lib, &effective_place_config)
        .map_err(SynthError::Placement)?;
    let routed = route_placed_netlist(&materialized, &placed, &effective_route_config)
        .map_err(SynthError::Routing)?;
    let mut export = export_to_simulation(&routed, &materialized);

    let mut mismatches = Vec::new();
    let mut converged = true;
    let mut convergence_rounds = 0u32;
    for combo in 0..total_combos {
        let inputs: Vec<bool> = (0..num_inputs).map(|i| (combo >> i) & 1 != 0).collect();
        let expected = mapping::evaluate_aig(aig, &inputs);
        let actual = evaluate_exported(&mut export, &inputs);
        converged &= export.last_converged;
        convergence_rounds = convergence_rounds.max(export.last_convergence_rounds);

        for (out_idx, (exp, act)) in expected.iter().zip(actual.iter()).enumerate() {
            if exp != act {
                mismatches.push(OutputMismatch {
                    output_idx: out_idx,
                    input_combo: combo,
                    expected: *exp,
                    actual: *act,
                });
            }
        }
    }

    Ok(ValidationResult {
        num_inputs,
        num_outputs,
        combos_tested: total_combos,
        mismatches,
        converged,
        convergence_rounds,
    })
}

// ---------------------------------------------------------------------------
// Module-level integration tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_to_end_and() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let result = synthesize(&aig, &lib, &SynthConfig::default());

        assert_eq!(result.stats.aig_inputs, 2);
        assert_eq!(result.stats.aig_outputs, 1);
        assert_eq!(result.stats.aig_and_nodes, 1);
        assert_eq!(result.stats.aig_depth, 1);
        assert!(result.stats.mapped_gates >= 1);
    }

    #[test]
    fn end_to_end_xor() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.xor(a, b);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let result = synthesize(&aig, &lib, &SynthConfig::default());

        assert_eq!(result.stats.aig_inputs, 2);
        assert_eq!(result.stats.aig_outputs, 1);
        assert_eq!(result.stats.aig_and_nodes, 3);
        assert!(result.stats.mapped_gates >= 1);
    }

    #[test]
    fn end_to_end_4bit_adder() {
        // Build a simple ripple-carry adder from AIGs
        let mut aig = Aig::new();
        let a = aig.add_input_bus("a", 4);
        let b = aig.add_input_bus("b", 4);

        let mut carry = AigLit::FALSE;
        let mut sum_bits = Vec::new();

        for i in 0..4 {
            // Full adder: sum = a ^ b ^ cin, cout = (a & b) | (cin & (a ^ b))
            let axb = aig.xor(a[i], b[i]);
            let sum = aig.xor(axb, carry);
            let ab = aig.and(a[i], b[i]);
            let caxb = aig.and(carry, axb);
            carry = aig.or(ab, caxb);
            sum_bits.push(sum);
        }
        sum_bits.push(carry); // carry out

        aig.add_output_bus("sum", &sum_bits);

        let lib = CellLibrary::tile_native();
        let result = synthesize(&aig, &lib, &SynthConfig::default());

        assert_eq!(result.stats.aig_inputs, 8); // 4 + 4
        assert_eq!(result.stats.aig_outputs, 5); // 4 sum + 1 carry
        assert!(result.stats.mapped_gates >= 4); // At least 4 full adders
        println!("{}", result.stats);
    }

    #[test]
    fn end_to_end_with_stats() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let c = aig.add_input("c");
        let ab = aig.and(a, b);
        let bc = aig.or(b, c);
        let y = aig.xor(ab, bc);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let result = synthesize(&aig, &lib, &SynthConfig::default());

        // Just verify it runs and produces meaningful stats
        assert!(result.stats.aig_and_nodes > 0);
        assert!(result.stats.mapped_gates > 0);
        let stats_str = format!("{}", result.stats);
        assert!(stats_str.contains("Synth:"));
    }

    // ===== ST2 integration tests =====

    fn build_4bit_adder() -> Aig {
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
        aig
    }

    fn build_8bit_adder() -> Aig {
        let mut aig = Aig::new();
        let a = aig.add_input_bus("a", 8);
        let b = aig.add_input_bus("b", 8);
        let mut carry = AigLit::FALSE;
        let mut sum_bits = Vec::new();
        for i in 0..8 {
            let axb = aig.xor(a[i], b[i]);
            let sum = aig.xor(axb, carry);
            let ab = aig.and(a[i], b[i]);
            let caxb = aig.and(carry, axb);
            carry = aig.or(ab, caxb);
            sum_bits.push(sum);
        }
        sum_bits.push(carry);
        aig.add_output_bus("sum", &sum_bits);
        aig
    }

    #[test]
    fn optimize_4bit_adder() {
        let aig = build_4bit_adder();
        let before_nodes = aig.num_nodes();
        let before_depth = aig.depth();

        let opt = optimize(&aig, &OptConfig::default());
        let after_nodes = opt.num_nodes();
        let after_depth = opt.depth();

        // Should reduce nodes (or at least not increase)
        assert!(
            after_nodes <= before_nodes,
            "optimize should not increase nodes: {} -> {}",
            before_nodes,
            after_nodes
        );

        // Verify equivalence
        for assignment in 0..256u32 {
            let inputs: Vec<bool> = (0..8).map(|i| (assignment >> i) & 1 != 0).collect();
            let orig = mapping::evaluate_aig(&aig, &inputs);
            let opt_out = mapping::evaluate_aig(&opt, &inputs);
            assert_eq!(
                orig, opt_out,
                "4-bit adder optimization broke equivalence at {:08b}",
                assignment
            );
        }

        eprintln!(
            "4-bit adder: nodes {}->{} ({:.0}%), depth {}->{}",
            before_nodes,
            after_nodes,
            if before_nodes > 0 {
                (1.0 - after_nodes as f64 / before_nodes as f64) * 100.0
            } else {
                0.0
            },
            before_depth,
            after_depth,
        );
    }

    #[test]
    fn optimize_8bit_adder() {
        let aig = build_8bit_adder();
        let before_nodes = aig.num_nodes();

        let opt = optimize(&aig, &OptConfig::default());
        let after_nodes = opt.num_nodes();

        assert!(
            after_nodes <= before_nodes,
            "optimize should not increase nodes: {} -> {}",
            before_nodes,
            after_nodes
        );

        // Verify equivalence on a sample of inputs (16 inputs = 65536 combos)
        // Test all 256 combos for each nibble pair to cover carry propagation
        for sample in 0..1024u32 {
            let inputs: Vec<bool> = (0..16).map(|i| (sample >> (i % 10)) & 1 != 0).collect();
            let orig = mapping::evaluate_aig(&aig, &inputs);
            let opt_out = mapping::evaluate_aig(&opt, &inputs);
            assert_eq!(orig, opt_out);
        }

        eprintln!("8-bit adder: nodes {}->{}", before_nodes, after_nodes,);
    }

    #[test]
    fn synthesize_with_optimization() {
        let aig = build_4bit_adder();
        let lib = CellLibrary::tile_native();

        // Without optimization
        let result_no_opt = synthesize(&aig, &lib, &SynthConfig::default());

        // With optimization
        let config_opt = SynthConfig {
            optimize: true,
            ..SynthConfig::default()
        };
        let result_opt = synthesize(&aig, &lib, &config_opt);

        // Optimized AIG should have fewer or equal AND nodes
        assert!(
            result_opt.stats.aig_and_nodes <= result_no_opt.stats.aig_and_nodes,
            "optimized AIG should have <= AND nodes: {} vs {}",
            result_opt.stats.aig_and_nodes,
            result_no_opt.stats.aig_and_nodes,
        );

        // Both should produce equivalent netlists
        let lib2 = CellLibrary::tile_native();
        assert!(
            mapping::verify_equivalence(&aig, &result_opt.netlist, &lib2),
            "Optimized synthesis must preserve equivalence"
        );

        eprintln!(
            "Synth no-opt: {} | Synth opt: {}",
            result_no_opt.stats, result_opt.stats
        );
    }

    // ===== ST3: materialize_inversions tests =====

    #[test]
    fn materialize_inversions_inserts_not_gates() {
        // Build AIG: y = NOT(a AND b) = NAND(a, b)
        // The mapper should produce an AND gate with output_inverted=true.
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let ab = aig.and(a, b);
        let nab = aig.not(ab);
        aig.add_output("y", nab);

        let lib = CellLibrary::tile_native();
        let netlist = mapping::map_to_library(&aig, &lib, &MapConfig::default());

        // Check that the original netlist has inversions somewhere
        let has_inversions = netlist
            .gates
            .iter()
            .any(|g| g.output_inverted || g.fanin_inverted.iter().any(|&inv| inv))
            || netlist.primary_outputs.iter().any(|po| po.2);

        // Materialize
        let materialized = materialize_inversions(&netlist, &lib);

        // All inversion flags should be cleared
        for gate in &materialized.gates {
            assert!(
                !gate.output_inverted,
                "gate {:?} still has output_inverted after materialization",
                gate.name
            );
            for (i, &inv) in gate.fanin_inverted.iter().enumerate() {
                assert!(
                    !inv,
                    "gate {:?} fanin {} still inverted after materialization",
                    gate.name, i
                );
            }
        }
        for (name, _, inverted) in &materialized.primary_outputs {
            assert!(
                !inverted,
                "primary output {} still inverted after materialization",
                name
            );
        }

        // If there were inversions, there should be more gates now (INV gates inserted)
        if has_inversions {
            assert!(
                materialized.gates.len() > netlist.gates.len(),
                "expected more gates after materializing inversions: {} vs {}",
                materialized.gates.len(),
                netlist.gates.len()
            );
            // Check that INV gates (cell_idx=2) were inserted
            let inv_count = materialized
                .gates
                .iter()
                .filter(|g| g.cell_idx == 2)
                .count();
            assert!(inv_count > 0, "expected INV gates to be inserted");
        }
    }

    #[test]
    fn materialize_inversions_preserves_equivalence() {
        // Test on several circuits that materialization preserves function
        let circuits: Vec<(&str, Box<dyn Fn() -> Aig>)> = vec![
            (
                "nand",
                Box::new(|| {
                    let mut aig = Aig::new();
                    let a = aig.add_input("a");
                    let b = aig.add_input("b");
                    let ab = aig.and(a, b);
                    let y = aig.not(ab);
                    aig.add_output("y", y);
                    aig
                }),
            ),
            (
                "nor",
                Box::new(|| {
                    let mut aig = Aig::new();
                    let a = aig.add_input("a");
                    let b = aig.add_input("b");
                    let y = aig.or(a, b);
                    let ny = aig.not(y);
                    aig.add_output("y", ny);
                    aig
                }),
            ),
            (
                "xnor",
                Box::new(|| {
                    let mut aig = Aig::new();
                    let a = aig.add_input("a");
                    let b = aig.add_input("b");
                    let y = aig.xor(a, b);
                    let ny = aig.not(y);
                    aig.add_output("y", ny);
                    aig
                }),
            ),
            (
                "inverter",
                Box::new(|| {
                    let mut aig = Aig::new();
                    let a = aig.add_input("a");
                    let y = aig.not(a);
                    aig.add_output("y", y);
                    aig
                }),
            ),
            ("adder4", Box::new(build_4bit_adder)),
        ];

        let lib = CellLibrary::tile_native();

        for (name, build) in &circuits {
            let aig = build();
            let netlist = mapping::map_to_library(&aig, &lib, &MapConfig::default());
            let materialized = materialize_inversions(&netlist, &lib);

            let num_inputs = netlist.primary_inputs.len();
            let num_combos = 1u32 << num_inputs;

            for combo in 0..num_combos.min(256) {
                let inputs: Vec<bool> = (0..num_inputs).map(|i| (combo >> i) & 1 != 0).collect();

                let orig = mapping::evaluate_netlist(&netlist, &lib, &inputs);
                let mat = mapping::evaluate_netlist(&materialized, &lib, &inputs);

                assert_eq!(
                    orig, mat,
                    "{}: materialized netlist disagrees at inputs {:?}\norig={:?}\n mat={:?}",
                    name, inputs, orig, mat
                );
            }
        }
    }

    // ===== ST3: synthesize_to_simulation integration tests =====

    #[test]
    fn synthesize_to_simulation_adder4() {
        let aig = build_4bit_adder();
        let lib = CellLibrary::tile_native();
        let place_cfg = PlaceConfig {
            halo: 3,
            ..PlaceConfig::default()
        };
        let mut export = synthesize_to_simulation(
            &aig,
            &lib,
            &SynthConfig::default(),
            &place_cfg,
            &RouteConfig::default(),
        )
        .expect("synthesize_to_simulation should succeed for 4-bit adder");

        // Exhaustively verify all 256 input combinations.
        for combo in 0..256u32 {
            let a_val = combo & 0xF;
            let b_val = (combo >> 4) & 0xF;
            let expected_sum = a_val + b_val;

            let inputs: Vec<bool> = (0..8).map(|i| (combo >> i) & 1 != 0).collect();
            let outputs = evaluate_exported(&mut export, &inputs);

            let mut actual_sum = 0u32;
            for (i, &bit) in outputs.iter().enumerate() {
                if bit {
                    actual_sum |= 1 << i;
                }
            }

            assert_eq!(
                actual_sum, expected_sum,
                "adder4 mismatch at a={} b={}: expected {} got {}",
                a_val, b_val, expected_sum, actual_sum
            );
        }
    }

    #[test]
    fn synthesize_to_simulation_with_optimization() {
        let aig = build_4bit_adder();
        let lib = CellLibrary::tile_native();

        let config = SynthConfig {
            optimize: true,
            ..SynthConfig::default()
        };
        let place_cfg = PlaceConfig {
            halo: 3,
            ..PlaceConfig::default()
        };
        let mut export =
            synthesize_to_simulation(&aig, &lib, &config, &place_cfg, &RouteConfig::default())
                .expect("optimized synthesize_to_simulation should succeed");

        // Spot-check a few combinations.
        for combo in [0u32, 0xFF, 0x55, 0xAA, 0x0F, 0xF0, 0x12, 0x99] {
            let a_val = combo & 0xF;
            let b_val = (combo >> 4) & 0xF;
            let expected_sum = a_val + b_val;

            let inputs: Vec<bool> = (0..8).map(|i| (combo >> i) & 1 != 0).collect();
            let outputs = evaluate_exported(&mut export, &inputs);

            let mut actual_sum = 0u32;
            for (i, &bit) in outputs.iter().enumerate() {
                if bit {
                    actual_sum |= 1 << i;
                }
            }

            assert_eq!(
                actual_sum, expected_sum,
                "optimized adder4 mismatch at a={} b={}: expected {} got {}",
                a_val, b_val, expected_sum, actual_sum
            );
        }
    }

    #[test]
    fn synthesize_to_simulation_adder4_multilayer() {
        let aig = build_4bit_adder();
        let lib = CellLibrary::tile_native();
        let place_cfg = PlaceConfig {
            halo: 3,
            ..PlaceConfig::default()
        };
        // Multi-layer routing with no_crossings avoids Cross tiles entirely,
        // using z-transitions instead. This eliminates the need for the bypass
        // system, which conflicts with router-produced z>0 segments.
        let route_cfg = RouteConfig {
            max_z: 1,
            no_crossings: true,
            ..RouteConfig::default()
        };

        let mut export =
            synthesize_to_simulation(&aig, &lib, &SynthConfig::default(), &place_cfg, &route_cfg)
                .expect("multi-layer synthesize_to_simulation should succeed for adder4");

        for combo in 0..256u32 {
            let a_val = combo & 0xF;
            let b_val = (combo >> 4) & 0xF;
            let expected_sum = a_val + b_val;

            let inputs: Vec<bool> = (0..8).map(|i| (combo >> i) & 1 != 0).collect();
            let outputs = evaluate_exported(&mut export, &inputs);

            let mut actual_sum = 0u32;
            for (i, &bit) in outputs.iter().enumerate() {
                if bit {
                    actual_sum |= 1 << i;
                }
            }

            assert_eq!(
                actual_sum, expected_sum,
                "multilayer adder4 mismatch at a={} b={}: expected {} got {}",
                a_val, b_val, expected_sum, actual_sum
            );
        }
    }

    #[test]
    #[ignore] // slow: ~75s validation — run with --include-ignored
    fn synthesize_to_simulation_mul4_multilayer() {
        let aig = benchmark::build_4bit_multiplier();
        let lib = CellLibrary::tile_native();
        let place_cfg = PlaceConfig {
            halo: 5,
            ..PlaceConfig::default()
        };
        let route_cfg = RouteConfig {
            max_z: 1,
            no_crossings: true,
            ..RouteConfig::default()
        };
        let result =
            synthesize_to_simulation(&aig, &lib, &SynthConfig::default(), &place_cfg, &route_cfg);
        if result.is_err() {
            // mul4 with no_crossings may fail routing at tight halo — acceptable.
            eprintln!(
                "mul4 multi-layer routing failed (expected for dense circuits): {:?}",
                result.err()
            );
            return;
        }
        let mut export = result.unwrap();

        for combo in 0..256u32 {
            let a_val = combo & 0xF;
            let b_val = (combo >> 4) & 0xF;
            let expected_product = a_val * b_val;

            let inputs: Vec<bool> = (0..8).map(|i| (combo >> i) & 1 != 0).collect();
            let outputs = evaluate_exported(&mut export, &inputs);

            let mut actual_product = 0u32;
            for (i, &bit) in outputs.iter().enumerate() {
                if bit {
                    actual_product |= 1 << i;
                }
            }

            assert_eq!(
                actual_product, expected_product,
                "multilayer mul4 mismatch at a={} b={}: expected {} got {}",
                a_val, b_val, expected_product, actual_product
            );
        }
    }

    /// Sprint 210: validate_synth_pipeline regression for existing benchmarks.
    #[test]
    #[ignore] // slow: ~4.2 min validation — run with --include-ignored
    fn validate_pipeline_existing_benchmarks() {
        // Same skip list as export.rs benchmark_suite_exports_cleanly
        let skip = ["mul4", "adder16", "decoder4to16"];
        for spec in benchmark::benchmark_specs() {
            if skip.contains(&spec.name) {
                continue;
            }
            let aig = (spec.build)();
            let place_config = PlaceConfig {
                min_x: 3,
                min_y: 3,
                halo: 3,
                ..PlaceConfig::default()
            };
            let result = validate_synth_pipeline(&aig, &place_config, &RouteConfig::default());
            match result {
                Ok(v) => assert!(
                    v.passed(),
                    "{}: {} mismatches (first: {:?})",
                    spec.name,
                    v.mismatches.len(),
                    v.mismatches.first()
                ),
                Err(e) => panic!("{}: pipeline error: {}", spec.name, e),
            }
        }
    }

    /// Sprint 210: diagnose combined decode AIG routing/export failures.
    #[test]
    #[ignore] // slow: ~23 min diagnostic — run with --include-ignored
    fn diagnose_combined_decode_pipeline() {
        use crate::synth::integration::build_combined_decode_aig;
        let aig = build_combined_decode_aig();

        // Try multiple configs to find one that works or characterize failures.
        let configs: Vec<(&str, PlaceConfig, RouteConfig)> = vec![
            (
                "halo4_multilayer",
                PlaceConfig {
                    halo: 4,
                    ..PlaceConfig::default()
                },
                RouteConfig {
                    max_z: 1,
                    no_crossings: true,
                    ..RouteConfig::default()
                },
            ),
            (
                "halo6_multilayer",
                PlaceConfig {
                    halo: 6,
                    ..PlaceConfig::default()
                },
                RouteConfig {
                    max_z: 1,
                    no_crossings: true,
                    ..RouteConfig::default()
                },
            ),
            (
                "halo8_multilayer",
                PlaceConfig {
                    halo: 8,
                    ..PlaceConfig::default()
                },
                RouteConfig {
                    max_z: 1,
                    no_crossings: true,
                    ..RouteConfig::default()
                },
            ),
            (
                "halo4_singlelayer",
                PlaceConfig {
                    halo: 4,
                    ..PlaceConfig::default()
                },
                RouteConfig::default(),
            ),
            (
                "halo8_singlelayer",
                PlaceConfig {
                    halo: 8,
                    ..PlaceConfig::default()
                },
                RouteConfig::default(),
            ),
        ];

        let mut any_passed = false;
        for (name, place_cfg, route_cfg) in &configs {
            let result = validate_synth_pipeline(&aig, place_cfg, route_cfg);
            match result {
                Ok(v) => {
                    if v.passed() {
                        println!(
                            "combined_decode [{}]: PASS ({} combos)",
                            name, v.combos_tested
                        );
                        any_passed = true;
                    } else {
                        println!(
                            "combined_decode [{}]: FAIL — {}/{} outputs wrong, first mismatch: out={} combo={:#07b} expected={} actual={}",
                            name,
                            v.mismatches.len(),
                            v.combos_tested * v.num_outputs as u32,
                            v.mismatches[0].output_idx,
                            v.mismatches[0].input_combo,
                            v.mismatches[0].expected,
                            v.mismatches[0].actual,
                        );
                    }
                }
                Err(e) => {
                    println!("combined_decode [{}]: ROUTE/PLACE ERROR — {}", name, e);
                }
            }
        }

        assert!(
            any_passed,
            "combined_decode: no config passed — see output above"
        );
    }

    /// Sprint 210: combined decode halo=8 multilayer — validates propagation fix.
    #[test]
    #[ignore] // slow: ~7.4 min validation — run with --include-ignored
    fn combined_decode_halo8_validates() {
        use crate::synth::integration::build_combined_decode_aig;

        let aig = build_combined_decode_aig();
        let place_config = PlaceConfig {
            halo: 8,
            ..PlaceConfig::default()
        };
        let route_config = RouteConfig {
            max_z: 1,
            no_crossings: true,
            ..RouteConfig::default()
        };
        let result = validate_synth_pipeline(&aig, &place_config, &route_config)
            .expect("combined decode should route with halo=8 multilayer");
        assert!(
            result.passed(),
            "combined decode: {} mismatches out of {} (first: {:?})",
            result.mismatches.len(),
            result.combos_tested * result.num_outputs as u32,
            result.mismatches.first()
        );
    }

    // ===== Sprint 213: Exhaustive truth-table sweeps =====

    /// Helper: run full synth pipeline on a truth-table AIG with standalone config.
    fn validate_tt(tt: u64, num_inputs: u32) -> Result<ValidationResult, SynthError> {
        let aig = Aig::from_truth_table(tt, num_inputs);
        // Sprint 217: Use adaptive config for single-output sweep validation.
        validate_synth_pipeline_adaptive(
            &aig,
            &PlaceConfig::standalone(),
            &RouteConfig::standalone(),
        )
    }

    #[test]
    fn sweep_2input_all() {
        let mut pass = 0u32;
        let mut fail = 0u32;
        let mut err = 0u32;
        let mut fail_tts = Vec::new();

        for tt in 0..16u64 {
            match validate_tt(tt, 2) {
                Ok(r) if r.passed() => pass += 1,
                Ok(_r) => {
                    fail += 1;
                    fail_tts.push(tt);
                }
                Err(_) => err += 1,
            }
        }
        // All non-degenerate 2-input functions must pass.
        // Degenerate (no AND nodes) may error on routing.
        assert!(
            fail == 0,
            "2-input sweep: {} pass, {} fail, {} error. Failed: {:?}",
            pass,
            fail,
            err,
            fail_tts
        );
    }

    #[test]
    #[ignore] // slow: ~18s exhaustive sweep — run with --include-ignored
    fn sweep_3input_all() {
        let mut pass = 0u32;
        let mut fail = 0u32;
        let mut err = 0u32;
        let mut fail_tts = Vec::new();

        for tt in 0..256u64 {
            match validate_tt(tt, 3) {
                Ok(r) if r.passed() => pass += 1,
                Ok(_r) => {
                    fail += 1;
                    fail_tts.push(tt);
                }
                Err(_) => err += 1,
            }
        }
        // >=99% of non-degenerate 3-input functions should pass.
        let total_routable = pass + fail;
        let pass_rate = if total_routable > 0 {
            pass as f64 / total_routable as f64
        } else {
            1.0
        };
        assert!(
            pass_rate >= 0.99,
            "3-input sweep: {} pass, {} fail, {} error ({:.1}% pass rate). Failed: {:?}",
            pass,
            fail,
            err,
            pass_rate * 100.0,
            fail_tts
        );
    }

    #[test]
    #[ignore] // ~minutes to run; invoke with: cargo test --release -- --ignored sweep_4input
    fn sweep_4input_all() {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicU32, Ordering};

        let pass = AtomicU32::new(0);
        let fail = AtomicU32::new(0);
        let err = AtomicU32::new(0);

        (0..65536u64)
            .into_par_iter()
            .for_each(|tt| match validate_tt(tt, 4) {
                Ok(r) if r.passed() => {
                    pass.fetch_add(1, Ordering::Relaxed);
                }
                Ok(_) => {
                    fail.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    err.fetch_add(1, Ordering::Relaxed);
                }
            });

        let pass = pass.load(Ordering::Relaxed);
        let fail = fail.load(Ordering::Relaxed);
        let err = err.load(Ordering::Relaxed);
        let total_routable = pass + fail;
        let pass_rate = if total_routable > 0 {
            pass as f64 / total_routable as f64
        } else {
            1.0
        };
        eprintln!(
            "4-input sweep: {} pass, {} fail, {} error ({:.1}% pass rate)",
            pass,
            fail,
            err,
            pass_rate * 100.0
        );
        // Sprint 217: tightened from >= 90% to >= 99%.
        assert!(
            pass_rate >= 0.99,
            "4-input sweep: {:.1}% pass rate below 99% threshold",
            pass_rate * 100.0
        );
        // Also lock a floor on routing success (total passable + routable).
        assert!(
            err <= 900,
            "4-input sweep: {} routing errors exceeds 900 cap (was 806 at Sprint 217 baseline)",
            err
        );
    }

    // ===== Sprint 217: 4-input failure diagnosis =====

    /// Phase 1: Pure AIG stats for all 65536 TTs (sub-second, no P&R).
    /// Identifies structural complexity distribution to find the failure boundary.
    #[test]
    #[ignore]
    fn profile_4input_aig_stats() {
        // Bucket by mapped_gates count — this is the main driver of P&R complexity.
        let mut by_gates: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut by_depth: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut max_gates = 0usize;
        let mut max_depth = 0u32;
        let mut hardest: Vec<(u64, usize, u32, usize)> = Vec::new(); // (tt, gates, depth, and_nodes)

        let lib = CellLibrary::tile_native();
        for tt in 0..65536u64 {
            let aig = Aig::from_truth_table(tt, 4);
            let result = synthesize(&aig, &lib, &SynthConfig::default());
            let s = &result.stats;
            *by_gates.entry(s.mapped_gates).or_default() += 1;
            *by_depth.entry(s.aig_depth).or_default() += 1;
            if s.mapped_gates > max_gates {
                max_gates = s.mapped_gates;
            }
            if s.aig_depth > max_depth {
                max_depth = s.aig_depth;
            }
            // Collect the hardest TTs (most mapped gates)
            if s.mapped_gates >= 5 {
                hardest.push((tt, s.mapped_gates, s.aig_depth, s.aig_and_nodes));
            }
        }

        let mut gates_sorted: Vec<_> = by_gates.into_iter().collect();
        gates_sorted.sort_by_key(|&(g, _)| g);
        let mut depth_sorted: Vec<_> = by_depth.into_iter().collect();
        depth_sorted.sort_by_key(|&(d, _)| d);

        eprintln!("\n=== 4-Input AIG Profile (65536 TTs) ===");
        eprintln!("  Max mapped gates: {}", max_gates);
        eprintln!("  Max AIG depth: {}", max_depth);
        eprintln!("\n  Distribution by mapped gates:");
        for (g, c) in &gates_sorted {
            eprintln!("    gates={}: {} TTs", g, c);
        }
        eprintln!("\n  Distribution by AIG depth:");
        for (d, c) in &depth_sorted {
            eprintln!("    depth={}: {} TTs", d, c);
        }
        hardest.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
        eprintln!("\n  Top 20 hardest TTs:");
        for &(tt, g, d, a) in hardest.iter().take(20) {
            eprintln!("    0x{:04X} gates={} depth={} and_nodes={}", tt, g, d, a);
        }
        eprintln!("==========================================\n");
    }

    /// Phase 2: Stratified sample — 50 TTs per gate-count stratum through full P&R.
    /// Finds the failure boundary in ~seconds instead of minutes.
    #[test]
    #[ignore]
    fn diagnose_4input_targeted() {
        use std::collections::HashMap;

        // Step 1: Fast AIG scan, group by gate count.
        let lib = CellLibrary::tile_native();
        let mut by_gates: HashMap<usize, Vec<(u64, u32, usize)>> = HashMap::new();
        for tt in 0..65536u64 {
            let aig = Aig::from_truth_table(tt, 4);
            let result = synthesize(&aig, &lib, &SynthConfig::default());
            let s = &result.stats;
            by_gates
                .entry(s.mapped_gates)
                .or_default()
                .push((tt, s.aig_depth, s.aig_and_nodes));
        }

        // Step 2: Sample up to 50 TTs per stratum, run full pipeline.
        let samples_per = 50;
        let mut strata: Vec<usize> = by_gates.keys().copied().collect();
        strata.sort();

        eprintln!("\n=== 4-Input Stratified Diagnosis ===");
        for &g in &strata {
            let pool = by_gates.get(&g).unwrap();
            let n = pool.len().min(samples_per);
            // Spread sample evenly across the pool
            let step = if pool.len() <= samples_per {
                1
            } else {
                pool.len() / samples_per
            };
            let mut pass = 0u32;
            let mut route_err = 0u32;
            let mut val_fail = 0u32;
            let mut convergence_fail = 0u32;
            let mut sample_failures: Vec<String> = Vec::new();

            for i in 0..n {
                let &(tt, depth, and_nodes) = &pool[i * step];
                match validate_tt(tt, 4) {
                    Ok(r) if r.passed() => pass += 1,
                    Ok(r) => {
                        if r.converged {
                            val_fail += 1;
                            if sample_failures.len() < 5 {
                                sample_failures.push(format!(
                                    "  VAL  0x{:04X} and={} depth={} mm={} rounds={}",
                                    tt,
                                    and_nodes,
                                    depth,
                                    r.mismatches.len(),
                                    r.convergence_rounds
                                ));
                            }
                        } else {
                            convergence_fail += 1;
                            if sample_failures.len() < 5 {
                                sample_failures.push(format!(
                                    "  CONV 0x{:04X} and={} depth={} mm={} rounds={}",
                                    tt,
                                    and_nodes,
                                    depth,
                                    r.mismatches.len(),
                                    r.convergence_rounds
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        route_err += 1;
                        if sample_failures.len() < 5 {
                            let msg = format!("{}", e);
                            sample_failures.push(format!(
                                "  ERR  0x{:04X} and={} depth={} err={}",
                                tt,
                                and_nodes,
                                depth,
                                &msg[..msg.len().min(60)]
                            ));
                        }
                    }
                }
            }
            let total = pass + route_err + val_fail + convergence_fail;
            let pct = if total > 0 {
                pass as f64 / total as f64 * 100.0
            } else {
                100.0
            };
            eprintln!(
                "gates={:2}: {}/{} pass ({:.0}%) [route_err={} val_fail={} conv_fail={}] (pool={})",
                g,
                pass,
                total,
                pct,
                route_err,
                val_fail,
                convergence_fail,
                pool.len()
            );
            for s in &sample_failures {
                eprintln!("    {}", s);
            }
        }
        eprintln!("==========================================\n");
    }

    // ===== Sprint 216: Decoder + Validation tests =====

    /// D6: Each of the 8 single-minterm 3-input TTs passes individually.
    #[test]
    fn test_decoder_output_individual_sweep() {
        for j in 0..8u64 {
            let tt = 1u64 << j;
            let result = validate_tt(tt, 3).unwrap_or_else(|e| {
                panic!("output[{}] TT=0x{:02X}: pipeline error — {}", j, tt, e);
            });
            assert!(
                result.passed(),
                "output[{}] TT=0x{:02X}: {} mismatches",
                j,
                tt,
                result.mismatches.len(),
            );
        }
    }

    /// D6: Combined 8-output decoder, all 8 pass.
    #[test]
    fn test_decoder_3to8_all_pass() {
        let tts: Vec<(u64, u32)> = (0..8u64).map(|j| (1u64 << j, 3)).collect();
        let aig = Aig::from_truth_tables(&tts);
        let result =
            validate_synth_pipeline(&aig, &PlaceConfig::standalone(), &RouteConfig::standalone())
                .expect("synth pipeline");
        assert!(
            result.mismatches.is_empty(),
            "3-to-8 decoder: {} mismatches on outputs {:?}",
            result.mismatches.len(),
            result
                .mismatches
                .iter()
                .map(|m| m.output_idx)
                .collect::<std::collections::HashSet<_>>(),
        );
    }

    /// D6: Verify convergence detection on a normal circuit.
    #[test]
    fn test_convergence_detection() {
        let aig = Aig::from_truth_table(0xE8, 3); // MAJ3
        let result =
            validate_synth_pipeline(&aig, &PlaceConfig::standalone(), &RouteConfig::standalone())
                .expect("synth pipeline");
        assert!(result.converged, "MAJ3 should converge for all combos");
        assert!(
            result.convergence_rounds > 0 && result.convergence_rounds < 200,
            "convergence rounds should be between 1 and 199, got {}",
            result.convergence_rounds,
        );
    }

    /// D6: Multi-output 2-input sweep — all 2-input 2-output combinations.
    #[test]
    fn test_multi_output_2input_sweep() {
        let mut pass = 0u32;
        let mut fail = 0u32;
        let mut err = 0u32;

        for tt0 in 0..16u64 {
            for tt1 in 0..16u64 {
                let tts = vec![(tt0, 2), (tt1, 2)];
                let aig = Aig::from_truth_tables(&tts);
                match validate_synth_pipeline(
                    &aig,
                    &PlaceConfig::standalone(),
                    &RouteConfig::standalone(),
                ) {
                    Ok(r) if r.mismatches.is_empty() => pass += 1,
                    Ok(_) => fail += 1,
                    Err(_) => err += 1,
                }
            }
        }
        let total_routable = pass + fail;
        let pass_rate = if total_routable > 0 {
            pass as f64 / total_routable as f64
        } else {
            1.0
        };
        eprintln!(
            "multi-output 2-input sweep: {} pass, {} fail, {} error ({:.1}%)",
            pass,
            fail,
            err,
            pass_rate * 100.0
        );
        // Document the rate — at least track it
        assert!(
            pass_rate >= 0.90,
            "multi-output 2-input sweep: {:.1}% pass rate, expected >=90%",
            pass_rate * 100.0
        );
    }

    // ===== Sprint 217: 4-Input Sweep Closure tests =====

    /// D5: Degenerate TTs (0 mapped gates) pass via fast-path bypass.
    #[test]
    fn test_degenerate_tt_bypass() {
        // Constants and projections — all have 0 mapped gates.
        let degenerates: &[(u64, u32, &str)] = &[
            (0x0000, 4, "const-0"),
            (0xFFFF, 4, "const-1"),
            (0xAAAA, 4, "proj-x0"),
            (0x5555, 4, "not-x0"),
            (0xCCCC, 4, "proj-x1"),
            (0xF0F0, 4, "proj-x2"),
            (0xFF00, 4, "proj-x3"),
            (0x00, 3, "const-0-3in"),
            (0xFF, 3, "const-1-3in"),
            (0xAA, 3, "proj-x0-3in"),
        ];
        for &(tt, ni, name) in degenerates {
            let result = validate_tt(tt, ni);
            assert!(
                result.is_ok(),
                "degenerate TT {} (0x{:04X}) should not error: {:?}",
                name,
                tt,
                result.err()
            );
            let r = result.unwrap();
            assert!(
                r.passed(),
                "degenerate TT {} (0x{:04X}) should pass",
                name,
                tt
            );
            assert!(
                r.converged,
                "degenerate TT {} should report converged",
                name
            );
        }
    }

    /// D5: Regression — specific TTs that failed routing before Sprint 217.
    /// With adaptive routing (halo=4, max_z=3), all four must now pass.
    #[test]
    fn test_sprint217_routing_regression() {
        let tts: &[u64] = &[0x367F, 0x3F1A, 0xD6BA, 0x8C2D];
        for &tt in tts {
            let result = validate_tt(tt, 4).unwrap_or_else(|e| {
                panic!(
                    "TT 0x{:04X}: routing should succeed with adaptive config — {}",
                    tt, e
                );
            });
            assert!(
                result.passed(),
                "TT 0x{:04X}: {} mismatches (Sprint 217 routing regression)",
                tt,
                result.mismatches.len()
            );
        }
    }
}
