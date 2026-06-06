//! Bridge between AIG-level logic synthesis and V2 tile simulation.
//!
//! Provides a unified API to compile an AIG circuit through the full synthesis
//! pipeline (map -> place -> route -> export -> inject) and evaluate the result
//! at runtime.
//!
//! # Example
//!
//! ```rust,ignore
//! use engine::synth::{bridge, Aig, CellLibrary};
//! use engine::simulation::Simulation;
//!
//! let mut sim = Simulation::with_size_layered(128, 640, 16);
//! let aig = build_my_circuit();
//! let lib = CellLibrary::tile_native();
//!
//! // Compile and inject at grid offset (0, 0), layer 0.
//! let (block, _export) = bridge::compile_and_inject(
//!     &aig, &lib, &mut sim, 0, 0, 0,
//! ).expect("synth block compiled");
//!
//! // Drive inputs (high bits = 1, low bits = 0).
//! let inputs = [u64::MAX, u64::MAX, 0, u64::MAX, 0];
//! let outputs = bridge::evaluate_block(&mut sim, &block, &inputs)
//!     .expect("synth block evaluated");
//! ```

use crate::simulation::Simulation;
use crate::synth::aig::Aig;
use crate::synth::cell_lib::CellLibrary;
use crate::synth::export::SynthExport;
use crate::synth::integration::{
    SynthInjectError, drive_injected_block_masked, try_inject_synth_export_z,
};
use crate::synth::placement::PlaceConfig;
use crate::synth::routing::RouteConfig;
use crate::synth::{SynthConfig, SynthError, synthesize_to_simulation};

/// Error from the synthesis-to-simulation bridge pipeline.
#[derive(Debug)]
pub enum BridgeError {
    /// Mapping or synthesis error.
    Mapping(String),
    /// Placement failed (no valid placement found).
    Placement(String),
    /// Routing failed (congestion or net loss).
    Routing(String),
    /// Grid bounds error (export doesn't fit in host).
    GridBounds {
        offset_x: usize,
        offset_y: usize,
        export_w: usize,
        export_h: usize,
        host_w: usize,
        host_h: usize,
    },
    /// Layer overflow (export needs more layers than available).
    LayerOverflow {
        offset_z: usize,
        export_layers: usize,
        host_layers: usize,
    },
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::Mapping(e) => write!(f, "mapping error: {e}"),
            BridgeError::Placement(e) => write!(f, "placement error: {e}"),
            BridgeError::Routing(e) => write!(f, "routing error: {e}"),
            BridgeError::GridBounds {
                offset_x,
                offset_y,
                export_w,
                export_h,
                host_w,
                host_h,
            } => write!(
                f,
                "export [{offset_x}..{}, {offset_y}..{}) doesn't fit in host {host_w}x{host_h}",
                offset_x + export_w,
                offset_y + export_h
            ),
            BridgeError::LayerOverflow {
                offset_z,
                export_layers,
                host_layers,
            } => write!(
                f,
                "needs layers {offset_z}..{} but host has {host_layers}",
                offset_z + export_layers
            ),
        }
    }
}

impl std::error::Error for BridgeError {}

impl From<SynthError> for BridgeError {
    fn from(err: SynthError) -> Self {
        match err {
            SynthError::Placement(e) => BridgeError::Placement(e.to_string()),
            SynthError::Routing(e) => BridgeError::Routing(e.to_string()),
        }
    }
}

impl From<SynthInjectError> for BridgeError {
    fn from(err: SynthInjectError) -> Self {
        match err {
            SynthInjectError::GridBounds {
                offset_x,
                offset_y,
                export_w,
                export_h,
                host_w,
                host_h,
            } => BridgeError::GridBounds {
                offset_x,
                offset_y,
                export_w,
                export_h,
                host_w,
                host_h,
            },
            SynthInjectError::LayerOverflow {
                offset_z,
                export_layers,
                host_layers,
            } => BridgeError::LayerOverflow {
                offset_z,
                export_layers,
                host_layers,
            },
        }
    }
}

/// V2-appropriate defaults for placement and routing.
///
/// Halo=4 with multi-layer routing (no_crossings) is the standard config
/// used by the existing synth integration in `v2_builder.rs`.
#[derive(Clone, Debug)]
pub struct BridgeConfig {
    /// Placement halo around each gate (default: 4).
    pub halo: usize,
    /// Maximum routing layer (default: 1, enables multi-layer).
    pub max_z: usize,
    /// Use no_crossings mode (detours via layer 1 for crossings).
    pub no_crossings: bool,
    /// Whether to optimize the AIG before mapping (default: false).
    pub optimize: bool,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        BridgeConfig {
            halo: 4,
            max_z: 1,
            no_crossings: true,
            optimize: false,
        }
    }
}

impl BridgeConfig {
    fn place_config(&self) -> PlaceConfig {
        PlaceConfig {
            halo: self.halo,
            ..PlaceConfig::default()
        }
    }

    fn route_config(&self) -> RouteConfig {
        RouteConfig {
            max_z: self.max_z,
            no_crossings: self.no_crossings,
            ..RouteConfig::default()
        }
    }
}

/// Compile an AIG through the full synthesis pipeline and inject the result
/// into a host `Simulation` at `(offset_x, offset_y, offset_z)`.
///
/// Runs: AIG -> optimize (optional) -> map -> materialize inversions -> place ->
/// route -> export -> inject into host grid.
///
/// Returns `(InjectedBlock, SynthExport)` where:
/// - `InjectedBlock` provides input/output indices and scope mask for evaluation
/// - `SynthExport` provides the raw simulation grid for inspection
///
/// Uses `BridgeConfig::default()` for placement/routing. Call
/// `compile_and_inject_with()` for custom config.
pub fn compile_and_inject(
    aig: &Aig,
    lib: &CellLibrary,
    sim: &mut Simulation,
    offset_x: usize,
    offset_y: usize,
    offset_z: usize,
) -> Result<(super::integration::InjectedBlock, SynthExport), BridgeError> {
    compile_and_inject_with(
        aig,
        lib,
        sim,
        offset_x,
        offset_y,
        offset_z,
        &BridgeConfig::default(),
    )
}

/// Compile with a custom `BridgeConfig`.
pub fn compile_and_inject_with(
    aig: &Aig,
    lib: &CellLibrary,
    sim: &mut Simulation,
    offset_x: usize,
    offset_y: usize,
    offset_z: usize,
    config: &BridgeConfig,
) -> Result<(super::integration::InjectedBlock, SynthExport), BridgeError> {
    let export = compile_aig_to_export(aig, lib, config)?;
    let block = inject_into(sim, &export, offset_x, offset_y, offset_z)?;

    Ok((block, export))
}

/// Compile an AIG to a standalone `SynthExport` without injecting it.
pub fn compile_aig_to_export(
    aig: &Aig,
    lib: &CellLibrary,
    config: &BridgeConfig,
) -> Result<SynthExport, BridgeError> {
    let synth_config = SynthConfig {
        optimize: config.optimize,
        ..SynthConfig::default()
    };
    let export = synthesize_to_simulation(
        aig,
        lib,
        &synth_config,
        &config.place_config(),
        &config.route_config(),
    )?;
    Ok(export)
}

/// Inject a `SynthExport` into a host `Simulation` at `(offset_x, offset_y, offset_z)`.
///
/// Copies all tiles from the export grid into the host grid. Maps input/output/
/// circuit indices from export-local to host-global flat indices.
///
/// Returns `InjectedBlock` with I/O indices and scope mask for evaluation.
///
/// Returns a typed `BridgeError` if the export does not fit in the host grid.
pub fn inject_into(
    sim: &mut Simulation,
    export: &SynthExport,
    offset_x: usize,
    offset_y: usize,
    offset_z: usize,
) -> Result<super::integration::InjectedBlock, BridgeError> {
    try_inject_synth_export_z(sim, export, offset_x, offset_y, offset_z).map_err(BridgeError::from)
}

/// Drive a synth block's inputs, propagate, and read outputs.
///
/// Wraps `drive_injected_block_masked` (masked propagation, safe during
/// CPU execution) and returns outputs as `Vec<u64>` (0 = logic 0, `u64::MAX` = logic 1).
///
/// `input_values[i]` must be 0 (logic 0) or non-zero (logic 1).
/// Length must equal `block.input_indices.len()`.
pub fn evaluate_block(
    sim: &mut Simulation,
    block: &super::integration::InjectedBlock,
    input_values: &[u64],
) -> Result<Vec<u64>, BridgeError> {
    let outputs = drive_injected_block_masked(sim, block, input_values);
    // Convert SynthOutputs → Vec<u64> by collecting the populated entries.
    Ok(outputs.iter().copied().collect())
}

/// Evaluate a synth block and interpret outputs as boolean bits.
///
/// Like `evaluate_block` but returns boolean values (true = non-zero, false = 0).
pub fn evaluate_block_bool(
    sim: &mut Simulation,
    block: &super::integration::InjectedBlock,
    input_values: &[u64],
) -> Result<Vec<bool>, BridgeError> {
    let raw = evaluate_block(sim, block, input_values)?;
    Ok(raw.iter().map(|v| *v != 0).collect())
}

// Re-export InjectedBlock and SynthOutputs for convenience.
pub use super::integration::InjectedBlock;
pub use super::integration::SynthOutputs;

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::aig::AigLit;
    use crate::synth::integration::{
        branch_taken_truth_table, build_branch_taken_aig, build_decoder3to8_aig,
        decoder3to8_truth_table,
    };
    use crate::synth::mapping::evaluate_aig;

    fn make_sim() -> Simulation {
        Simulation::with_size_layered(256, 1280, 16)
    }

    #[test]
    fn bridge_compile_branch_taken() {
        let aig = build_branch_taken_aig();
        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        let (block, _export) = compile_and_inject(&aig, &lib, &mut sim, 64, 100, 0)
            .expect("branch-taken AIG compiled");

        assert!(!block.input_indices.is_empty(), "must have inputs");
        assert!(!block.output_indices.is_empty(), "must have outputs");
        assert!(block.width > 0, "width > 0");
        assert!(block.height > 0, "height > 0");
    }

    #[test]
    fn bridge_compile_decoder3to8() {
        let aig = build_decoder3to8_aig();
        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        let (block, _export) =
            compile_and_inject(&aig, &lib, &mut sim, 64, 160, 0).expect("decoder3to8 AIG compiled");

        assert_eq!(block.input_indices.len(), 3);
        assert_eq!(block.output_indices.len(), 8);
    }

    #[test]
    fn bridge_evaluate_branch_taken_all_combos() {
        let aig = build_branch_taken_aig();
        let truth_table = branch_taken_truth_table();
        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        let (block, _export) =
            compile_and_inject(&aig, &lib, &mut sim, 64, 220, 0).expect("branch-taken compiled");

        // Verify all 32 input combinations.
        for combo in 0..32u32 {
            let inputs: Vec<u64> = (0..5)
                .map(|i| if (combo >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();

            let outputs = evaluate_block(&mut sim, &block, &inputs).expect("evaluate succeeded");

            assert_eq!(outputs.len(), 1, "branch-taken has 1 output");

            let expected = truth_table[combo as usize];
            let actual = outputs[0] != 0;
            assert_eq!(
                actual,
                expected,
                "branch-taken mismatch at ctrl_b={:b}, flag_z={}, flag_c={}",
                combo & 0x07,
                (combo >> 3) & 1,
                (combo >> 4) & 1,
            );
        }
    }

    #[test]
    fn bridge_evaluate_decoder3to8_all_combos() {
        let aig = build_decoder3to8_aig();
        let truth_table = decoder3to8_truth_table();
        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        let (block, _export) =
            compile_and_inject(&aig, &lib, &mut sim, 64, 280, 0).expect("decoder3to8 compiled");

        for sel in 0..8u32 {
            let inputs: Vec<u64> = (0..3)
                .map(|i| if (sel >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();

            let outputs = evaluate_block(&mut sim, &block, &inputs).expect("evaluate succeeded");

            assert_eq!(outputs.len(), 8, "decoder has 8 outputs");

            for (i, &out_val) in outputs.iter().enumerate() {
                let expected = truth_table[sel as usize][i];
                let actual = out_val != 0;
                assert_eq!(
                    actual, expected,
                    "decoder3to8 output[{}] mismatch at sel={}: expected={} got={}",
                    i, sel, expected, actual
                );
            }
        }
    }

    #[test]
    fn bridge_evaluate_vs_aig_equivalence() {
        // Verify that bridge evaluation matches AIG ground truth for branch-taken.
        let aig = build_branch_taken_aig();
        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        let (block, _export) =
            compile_and_inject(&aig, &lib, &mut sim, 64, 340, 0).expect("compiled");

        for combo in 0..32u32 {
            let inputs: Vec<u64> = (0..5)
                .map(|i| if (combo >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();

            // AIG ground truth.
            let aig_bools: Vec<bool> = (0..5).map(|i| (combo >> i) & 1 != 0).collect();
            let aig_out = evaluate_aig(&aig, &aig_bools);
            let expected = aig_out[0];

            // Bridge evaluation.
            let outputs =
                evaluate_block_bool(&mut sim, &block, &inputs).expect("evaluate succeeded");
            let actual = outputs[0];

            assert_eq!(
                actual, expected,
                "bridge output disagrees with AIG at combo={:05b}",
                combo
            );
        }
    }

    #[test]
    fn bridge_evaluate_simple_xor() {
        // Build a simple XOR from AIG and verify via bridge.
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.xor(a, b);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        let (block, _export) =
            compile_and_inject(&aig, &lib, &mut sim, 64, 400, 0).expect("XOR compiled");

        for combo in 0..4u32 {
            let a_val = if (combo >> 0) & 1 != 0 { u64::MAX } else { 0 };
            let b_val = if (combo >> 1) & 1 != 0 { u64::MAX } else { 0 };

            let outputs = evaluate_block_bool(&mut sim, &block, &[a_val, b_val]).expect("evaluate");
            let actual = outputs[0];

            let expected = (combo >> 0) & 1 != (combo >> 1) & 1;
            assert_eq!(actual, expected, "XOR mismatch at combo={:02b}", combo);
        }
    }

    #[test]
    fn bridge_evaluate_simple_and() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        let (block, _export) =
            compile_and_inject(&aig, &lib, &mut sim, 64, 460, 0).expect("AND compiled");

        for combo in 0..4u32 {
            let a_val = if (combo >> 0) & 1 != 0 { u64::MAX } else { 0 };
            let b_val = if (combo >> 1) & 1 != 0 { u64::MAX } else { 0 };

            let outputs = evaluate_block_bool(&mut sim, &block, &[a_val, b_val]).expect("evaluate");
            let expected = ((combo >> 0) & 1 != 0) && ((combo >> 1) & 1 != 0);
            assert_eq!(outputs[0], expected, "AND mismatch at combo={:02b}", combo);
        }
    }

    #[test]
    fn bridge_evaluate_simple_or() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.or(a, b);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        let (block, _export) =
            compile_and_inject(&aig, &lib, &mut sim, 64, 520, 0).expect("OR compiled");

        for combo in 0..4u32 {
            let a_val = if (combo >> 0) & 1 != 0 { u64::MAX } else { 0 };
            let b_val = if (combo >> 1) & 1 != 0 { u64::MAX } else { 0 };

            let outputs = evaluate_block_bool(&mut sim, &block, &[a_val, b_val]).expect("evaluate");
            let expected = ((combo >> 0) & 1 != 0) || ((combo >> 1) & 1 != 0);
            assert_eq!(outputs[0], expected, "OR mismatch at combo={:02b}", combo);
        }
    }

    #[test]
    fn bridge_evaluate_not_gate() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let y = aig.not(a);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        let (block, _export) =
            compile_and_inject(&aig, &lib, &mut sim, 64, 580, 0).expect("NOT compiled");

        for val in 0..2u32 {
            let input_val = if val == 1 { u64::MAX } else { 0 };
            let outputs = evaluate_block_bool(&mut sim, &block, &[input_val]).expect("evaluate");
            let expected = val == 0; // NOT: 0->1, 1->0
            assert_eq!(outputs[0], expected, "NOT mismatch at input={}", val);
        }
    }

    #[test]
    fn bridge_multi_output_mux16to1() {
        // Build a 4-to-1 MUX (2-bit selector, 4 data inputs) from AIG.
        // Output = data[sel]
        let mut aig = Aig::new();
        let s0 = aig.add_input("s0");
        let s1 = aig.add_input("s1");
        let d0 = aig.add_input("d0");
        let d1 = aig.add_input("d1");
        let d2 = aig.add_input("d2");
        let d3 = aig.add_input("d3");

        let ns0 = s0.negated();
        let ns1 = s1.negated();

        let out0 = aig.and(ns1, ns0); // sel=0
        let out1 = aig.and(ns1, s0); // sel=1
        let out2 = aig.and(s1, ns0); // sel=2
        let out3 = aig.and(s1, s0); // sel=3

        let y0_0 = aig.and(out0, d0);
        let y0_1 = aig.and(out1, d1);
        let y0 = aig.or(y0_0, y0_1);
        let y1_0 = aig.and(out2, d2);
        let y1_1 = aig.and(out3, d3);
        let y1 = aig.or(y1_0, y1_1);

        aig.add_output("y0", y0);
        aig.add_output("y1", y1);

        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        let (block, _export) =
            compile_and_inject(&aig, &lib, &mut sim, 64, 640, 0).expect("MUX compiled");

        // Test all 16 combinations of (sel[1:0], d[3:0])
        for combo in 0..64u32 {
            let s1 = (combo >> 5) & 1 != 0;
            let s0 = (combo >> 4) & 1 != 0;
            let sel = ((s1 as u32) << 1) | ((s0 as u32) << 0);

            let d0 = (combo >> 0) & 1 != 0;
            let d1 = (combo >> 1) & 1 != 0;
            let d2 = (combo >> 2) & 1 != 0;
            let d3 = (combo >> 3) & 1 != 0;

            let inputs = vec![
                if s0 { u64::MAX } else { 0 },
                if s1 { u64::MAX } else { 0 },
                if d0 { u64::MAX } else { 0 },
                if d1 { u64::MAX } else { 0 },
                if d2 { u64::MAX } else { 0 },
                if d3 { u64::MAX } else { 0 },
            ];

            let outputs = evaluate_block_bool(&mut sim, &block, &inputs).expect("evaluate");

            let expected_y0 = match sel {
                0 => d0,
                1 => d1,
                _ => false,
            };
            let expected_y1 = match sel {
                2 => d2,
                3 => d3,
                _ => false,
            };

            assert_eq!(
                outputs[0],
                expected_y0,
                "MUX y0 mismatch at combo={:06b} (sel={}, data={})",
                combo,
                sel,
                combo & 0xF
            );
            assert_eq!(
                outputs[1],
                expected_y1,
                "MUX y1 mismatch at combo={:06b} (sel={}, data={})",
                combo,
                sel,
                combo & 0xF
            );
        }
    }

    #[test]
    fn bridge_inject_bounds_check() {
        let aig = build_branch_taken_aig();
        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        // Inject near the right edge — export width is ~65 tiles, so 200+65=265 > 256.
        let result = compile_and_inject(&aig, &lib, &mut sim, 200, 1100, 0);
        assert!(
            matches!(result, Err(BridgeError::GridBounds { .. })),
            "should fail with grid bounds error"
        );
    }

    #[test]
    fn bridge_inject_layer_overflow() {
        let aig = build_branch_taken_aig();
        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        // Use offset_z=15 on a 16-layer sim — export needs at least 1 more layer.
        let result = compile_and_inject(&aig, &lib, &mut sim, 0, 0, 15);
        assert!(
            matches!(result, Err(BridgeError::LayerOverflow { .. })),
            "should fail with layer overflow"
        );
    }

    /// Sprint 359: test_bridge_adder4 — compile adder4, evaluate all 256 inputs, verify against AIG.
    #[test]
    fn test_bridge_adder4() {
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
        let mut sim = make_sim();

        let (block, _export) =
            compile_and_inject(&aig, &lib, &mut sim, 0, 0, 0).expect("adder4 compiled");

        for combo in 0..256u32 {
            let a_val = combo & 0xF;
            let b_val = (combo >> 4) & 0xF;
            let expected_sum = a_val + b_val;

            let inputs: Vec<u64> = (0..8)
                .map(|i| if (combo >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();

            let outputs = evaluate_block(&mut sim, &block, &inputs).expect("evaluate succeeded");

            let mut actual_sum: u32 = 0;
            for (i, &bit) in outputs.iter().enumerate().take(9) {
                if bit != 0 {
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

    /// Sprint 359: test_bridge_decoder — compile branch decoder, verify against truth table.
    #[test]
    fn test_bridge_decoder() {
        let aig = build_branch_taken_aig();
        let truth_table = branch_taken_truth_table();
        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        let (block, _export) =
            compile_and_inject(&aig, &lib, &mut sim, 64, 700, 0).expect("decoder compiled");

        for combo in 0..32u32 {
            let inputs: Vec<u64> = (0..5)
                .map(|i| if (combo >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();

            let outputs = evaluate_block(&mut sim, &block, &inputs).expect("evaluate succeeded");

            let expected = truth_table[combo as usize];
            let actual = outputs[0] != 0;
            assert_eq!(actual, expected, "decoder mismatch at combo={:05b}", combo);
        }
    }

    /// Sprint 359: test_driver_roundtrip — compile AND gate to sim, drive const tiles, read back.
    #[test]
    fn test_driver_roundtrip() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("out", y);

        let lib = CellLibrary::tile_native();
        let mut sim = make_sim();

        let (block, _export) =
            compile_and_inject(&aig, &lib, &mut sim, 0, 0, 0).expect("AND compiled");

        for combo in 0..4u32 {
            let inputs = vec![
                if (combo >> 0) & 1 != 0 { u64::MAX } else { 0 },
                if (combo >> 1) & 1 != 0 { u64::MAX } else { 0 },
            ];
            let outputs =
                evaluate_block_bool(&mut sim, &block, &inputs).expect("evaluate succeeded");
            let expected = ((combo >> 0) & 1 != 0) && ((combo >> 1) & 1 != 0);
            assert_eq!(
                outputs[0], expected,
                "roundtrip mismatch at combo={:02b}",
                combo
            );
        }
    }
}
