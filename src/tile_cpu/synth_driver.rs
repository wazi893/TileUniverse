//! V2-side driver for synth-generated logic blocks.
//!
//! Bridges runtime V2 values to/from AIG inputs and synth block outputs.
//! Handles compilation, placement, and evaluation of synth blocks within
//! a V2 CPU simulation grid.
//!
//! # Example
//!
//! ```rust,ignore
//! use engine::tile_cpu::synth_driver::{SynthDriver, SynthBlockConfig};
//! use engine::synth::CellLibrary;
//! use engine::simulation::Simulation;
//!
//! let mut sim = Simulation::with_size_layered(128, 640, 16);
//! let mut cpu = V2Builder::new().build(&mut sim);
//!
//! // Configure a synth block for branch-taken decode.
//! let config = SynthBlockConfig::new("branch_taken")
//!     .with_aig(build_branch_taken_aig())
//!     .with_placement(64, 100, 0);
//!
//! // Compile and inject.
//! let driver = SynthDriver::compile_and_inject(&config, &mut sim)
//!     .expect("synth block compiled");
//!
//! // Evaluate during execution.
//! let outputs = driver.evaluate(&mut sim, &[u64::MAX, 0, 0, u64::MAX, 0])
//!     .expect("evaluation succeeded");
//! ```

use crate::simulation::Simulation;
use crate::synth::aig::Aig;
use crate::synth::bridge::{
    BridgeConfig, compile_and_inject_with, evaluate_block, evaluate_block_bool,
};
use crate::synth::cell_lib::CellLibrary;
use crate::synth::integration::{SynthOutputs, drive_injected_block_masked};

/// Configuration for compiling and placing a synth block.
pub struct SynthBlockConfig {
    /// Name for debugging.
    pub name: String,
    /// The AIG circuit to compile.
    pub aig: Aig,
    /// Grid placement config (offset_x, offset_y).
    pub placement_x: usize,
    pub placement_y: usize,
    pub placement_z: usize,
    /// Bridge config for placement/routing parameters.
    pub bridge_config: BridgeConfig,
}

impl SynthBlockConfig {
    /// Create a new config with a name.
    pub fn new(name: &str) -> Self {
        SynthBlockConfig {
            name: name.to_string(),
            aig: Aig::new(),
            placement_x: 0,
            placement_y: 0,
            placement_z: 0,
            bridge_config: BridgeConfig::default(),
        }
    }

    /// Set the AIG circuit.
    pub fn with_aig(mut self, aig: Aig) -> Self {
        self.aig = aig;
        self
    }

    /// Set grid X offset.
    pub fn with_placement_x(mut self, x: usize) -> Self {
        self.placement_x = x;
        self
    }

    /// Set grid Y offset.
    pub fn with_placement_y(mut self, y: usize) -> Self {
        self.placement_y = y;
        self
    }

    /// Set grid Z offset.
    pub fn with_placement_z(mut self, z: usize) -> Self {
        self.placement_z = z;
        self
    }

    /// Set grid placement offset.
    pub fn with_placement(mut self, x: usize, y: usize, z: usize) -> Self {
        self.placement_x = x;
        self.placement_y = y;
        self.placement_z = z;
        self
    }

    /// Set bridge config.
    pub fn with_bridge_config(mut self, config: BridgeConfig) -> Self {
        self.bridge_config = config;
        self
    }
}

/// Manages the full lifecycle of a synth block within a V2 CPU.
///
/// Holds the compiled `InjectedBlock` (with I/O indices) and provides
/// evaluation methods that are safe to call during V2 execution.
#[derive(Debug)]
pub struct SynthDriver {
    /// Name for debugging.
    pub name: String,
    /// The injected block with I/O indices and scope mask.
    pub block: crate::synth::bridge::InjectedBlock,
    /// Whether this driver has been compiled successfully.
    pub compiled: bool,
}

impl SynthDriver {
    /// Compile a synth block from config and inject it into the host sim.
    ///
    /// Uses the placement and bridge settings stored in `SynthBlockConfig`.
    pub fn compile_and_inject(
        config: &SynthBlockConfig,
        sim: &mut Simulation,
    ) -> Result<Self, crate::synth::bridge::BridgeError> {
        Self::compile_and_inject_with(
            config,
            sim,
            config.placement_x,
            config.placement_y,
            config.placement_z,
            &config.bridge_config,
        )
    }

    /// Compile using the config's bridge settings but an explicit placement.
    pub fn compile_and_inject_at(
        config: &SynthBlockConfig,
        sim: &mut Simulation,
        offset_x: usize,
        offset_y: usize,
        offset_z: usize,
    ) -> Result<Self, crate::synth::bridge::BridgeError> {
        Self::compile_and_inject_with(
            config,
            sim,
            offset_x,
            offset_y,
            offset_z,
            &config.bridge_config,
        )
    }

    /// Compile with explicit placement and bridge settings.
    pub fn compile_and_inject_with(
        config: &SynthBlockConfig,
        sim: &mut Simulation,
        offset_x: usize,
        offset_y: usize,
        offset_z: usize,
        bridge_cfg: &BridgeConfig,
    ) -> Result<Self, crate::synth::bridge::BridgeError> {
        let lib = CellLibrary::tile_native();
        let (block, _export) = compile_and_inject_with(
            &config.aig,
            &lib,
            sim,
            offset_x,
            offset_y,
            offset_z,
            bridge_cfg,
        )?;

        Ok(SynthDriver {
            name: config.name.clone(),
            block,
            compiled: true,
        })
    }

    /// Evaluate the synth block with the given input values.
    ///
    /// `input_values[i]` must be 0 (logic 0) or non-zero (logic 1).
    /// Returns outputs as `Vec<u64>` (0 = logic 0, `u64::MAX` = logic 1).
    pub fn evaluate(&self, sim: &mut Simulation, input_values: &[u64]) -> Result<Vec<u64>, String> {
        if !self.compiled {
            return Err("synth driver not compiled".into());
        }
        evaluate_block(sim, &self.block, input_values)
            .map_err(|e| format!("synth evaluate error: {e}"))
    }

    /// Evaluate and return boolean outputs.
    pub fn evaluate_bool(
        &self,
        sim: &mut Simulation,
        input_values: &[u64],
    ) -> Result<Vec<bool>, String> {
        if !self.compiled {
            return Err("synth driver not compiled".into());
        }
        evaluate_block_bool(sim, &self.block, input_values)
            .map_err(|e| format!("synth evaluate error: {e}"))
    }

    /// Evaluate and return raw `SynthOutputs` (stack-allocated, no heap).
    pub fn evaluate_raw(
        &self,
        sim: &mut Simulation,
        input_values: &[u64],
    ) -> Result<SynthOutputs, String> {
        if !self.compiled {
            return Err("synth driver not compiled".into());
        }
        Ok(drive_injected_block_masked(sim, &self.block, input_values))
    }

    /// Check if the block is compiled.
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }

    /// Get the block width in tiles.
    pub fn width(&self) -> usize {
        self.block.width
    }

    /// Get the block height in tiles.
    pub fn height(&self) -> usize {
        self.block.height
    }

    /// Get the number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.block.input_indices.len()
    }

    /// Get the number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.block.output_indices.len()
    }

    /// Return a reference to the underlying injected block.
    pub fn block(&self) -> &crate::synth::bridge::InjectedBlock {
        &self.block
    }

    /// Consume self and return the underlying injected block.
    pub fn into_block(self) -> crate::synth::bridge::InjectedBlock {
        self.block
    }
}

// ===== Synth I/O Binding =====

/// Source for a single synth block input bit.
#[derive(Clone, Debug)]
pub enum SynthInputSource {
    /// Fixed immediate bit (always 0 or 1).
    ImmediateBit(bool),
    /// A single simulation tile (read as 0 or u64::MAX).
    LogicIndex(usize),
    /// A bit extracted from a u64 value (used when the source is known at bind time).
    PackedValueBit { value: u64, bit: u8 },
}

/// Target for a single synth block output bit.
#[derive(Clone, Debug)]
pub enum SynthOutputTarget {
    /// Discard the output.
    None,
    /// Write the output to a simulation tile index.
    LogicIndex(usize),
}

/// Maps V2-side values to synth block inputs and outputs.
///
/// Used to drive a synth block without hand-construction of the input vector.
#[derive(Clone, Debug)]
pub struct SynthIoBinding {
    /// Input sources. Each entry produces one `u64` input value.
    pub inputs: Vec<SynthInputSource>,
    /// Output targets. Each entry consumes one `u64` output value.
    pub outputs: Vec<SynthOutputTarget>,
}

impl SynthIoBinding {
    /// Gather all input values from a simulation.
    ///
    /// `LogicIndex` reads a single tile value from `sim`.
    /// `ImmediateBit` and `PackedValueBit` are evaluated immediately.
    pub fn gather_inputs(&self, sim: &Simulation) -> Vec<u64> {
        self.inputs
            .iter()
            .map(|src| match src {
                SynthInputSource::ImmediateBit(b) => {
                    if *b {
                        u64::MAX
                    } else {
                        0
                    }
                }
                SynthInputSource::LogicIndex(idx) => sim.get_logic_value_by_idx(*idx),
                SynthInputSource::PackedValueBit { value, bit } => {
                    if (value >> bit) & 1 != 0 {
                        u64::MAX
                    } else {
                        0
                    }
                }
            })
            .collect()
    }

    /// Deliver output values back into the simulation.
    ///
    /// `LogicIndex` writes `u64::MAX` (logic 1) or `0` (logic 0) to the tile.
    pub fn deliver_outputs(&self, sim: &mut Simulation, outputs: &[u64]) {
        for (i, target) in self.outputs.iter().enumerate() {
            if let SynthOutputTarget::LogicIndex(idx) = target {
                let val = if i < outputs.len() && outputs[i] != 0 {
                    u64::MAX
                } else {
                    0
                };
                sim.set_logic_value_by_idx(*idx, val);
            }
        }
    }
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::integration::{branch_taken_truth_table, build_branch_taken_aig};

    fn make_sim() -> Simulation {
        Simulation::with_size_layered(256, 1280, 16)
    }

    #[test]
    fn driver_compile_branch_taken() {
        let aig = build_branch_taken_aig();
        let config = SynthBlockConfig::new("branch_test")
            .with_aig(aig)
            .with_placement(64, 100, 0);
        let mut sim = make_sim();

        let driver =
            SynthDriver::compile_and_inject(&config, &mut sim).expect("branch-taken compiled");

        assert!(driver.compiled);
        assert!(driver.width() > 0);
        assert!(driver.height() > 0);
        assert_eq!(driver.block.offset_x, 64);
        assert_eq!(driver.block.offset_y, 100);
    }

    #[test]
    fn driver_uses_configured_layer() {
        let aig = build_branch_taken_aig();
        let config = SynthBlockConfig::new("branch_layer")
            .with_aig(aig)
            .with_placement(0, 0, 15);
        let mut sim = make_sim();

        let result = SynthDriver::compile_and_inject(&config, &mut sim);
        assert!(
            matches!(
                result,
                Err(crate::synth::bridge::BridgeError::LayerOverflow { .. })
            ),
            "driver should honor placement_z from config"
        );
    }

    #[test]
    fn driver_evaluate_branch_taken() {
        let aig = build_branch_taken_aig();
        let truth_table = branch_taken_truth_table();
        let config = SynthBlockConfig::new("branch_eval")
            .with_aig(aig)
            .with_placement(64, 160, 0);
        let mut sim = make_sim();

        let driver = SynthDriver::compile_and_inject(&config, &mut sim).expect("compiled");

        // Verify all 32 input combinations.
        for combo in 0..32u32 {
            let inputs: Vec<u64> = (0..5)
                .map(|i| if (combo >> i) & 1 != 0 { u64::MAX } else { 0 })
                .collect();

            let outputs = driver
                .evaluate(&mut sim, &inputs)
                .expect("evaluate succeeded");

            let expected = truth_table[combo as usize];
            assert_eq!(
                outputs[0] != 0,
                expected,
                "branch-taken mismatch at ctrl_b={:b}",
                combo & 0x07
            );
        }
    }

    #[test]
    fn driver_evaluate_xor() {
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.xor(a, b);
        aig.add_output("y", y);

        let config = SynthBlockConfig::new("xor_test")
            .with_aig(aig)
            .with_placement(64, 220, 0);
        let mut sim = make_sim();

        let driver = SynthDriver::compile_and_inject(&config, &mut sim).expect("compiled");

        for combo in 0..4u32 {
            let inputs = vec![
                if (combo >> 0) & 1 != 0 { u64::MAX } else { 0 },
                if (combo >> 1) & 1 != 0 { u64::MAX } else { 0 },
            ];
            let outputs = driver.evaluate(&mut sim, &inputs).expect("evaluate");
            let expected = ((combo >> 0) & 1 != 0) ^ ((combo >> 1) & 1 != 0);
            assert_eq!(
                outputs[0] != 0,
                expected,
                "XOR mismatch at combo={:02b}",
                combo
            );
        }
    }

    #[test]
    fn driver_not_compiled_fails() {
        // Build a valid AIG so compilation succeeds.
        let mut aig = Aig::new();
        let a = aig.add_input("a");
        let b = aig.add_input("b");
        let y = aig.and(a, b);
        aig.add_output("y", y);
        let config = SynthBlockConfig::new("compiled").with_aig(aig);
        let mut sim = make_sim();

        let driver = SynthDriver::compile_and_inject(&config, &mut sim).expect("compiled");
        assert!(driver.is_compiled());

        // Now test the uncompiled case.
        let uncompiled = SynthDriver {
            name: "test".into(),
            block: crate::synth::bridge::InjectedBlock {
                offset_x: 0,
                offset_y: 0,
                width: 0,
                height: 0,
                input_indices: Vec::new(),
                output_indices: Vec::new(),
                circuit_indices: Vec::new(),
                scope_mask: Vec::new(),
                outputs_cache: None,
            },
            compiled: false,
        };
        let mut sim2 = make_sim();
        assert_eq!(
            uncompiled.evaluate(&mut sim2, &[]),
            Err("synth driver not compiled".into())
        );
    }

    #[test]
    fn driver_accessor_methods() {
        let aig = build_branch_taken_aig();
        let config = SynthBlockConfig::new("accessor_test")
            .with_aig(aig)
            .with_placement(64, 280, 0);
        let mut sim = make_sim();

        let driver = SynthDriver::compile_and_inject(&config, &mut sim).expect("compiled");

        assert_eq!(driver.num_inputs(), 5);
        assert!(driver.num_outputs() >= 1);
        assert!(driver.width() > 0);
        assert!(driver.height() > 0);
    }

    #[test]
    fn synth_io_binding_pilot() {
        use super::*;

        // Build branch-taken AIG (5 inputs, 1 output, truth table: ctrl_b[0]|ctrl_b[1]|ctrl_b[2]).
        let truth_table = branch_taken_truth_table();

        // Place the synth block so we can reference its input/output tiles.
        let config = SynthBlockConfig::new("binding_pilot")
            .with_aig(build_branch_taken_aig())
            .with_placement(64, 340, 0);
        let mut sim = make_sim();
        let driver =
            SynthDriver::compile_and_inject(&config, &mut sim).expect("binding_pilot compiled");

        // Build binding that reads all 5 inputs from the injected block's input tiles.
        let binding = SynthIoBinding {
            inputs: (0..driver.num_inputs())
                .map(|i| SynthInputSource::LogicIndex(driver.block().input_indices[i]))
                .collect(),
            outputs: vec![SynthOutputTarget::LogicIndex(
                driver.block().output_indices[0],
            )],
        };

        // Verify all 32 combinations against the truth table.
        for combo in 0..32u32 {
            // Drive inputs via simulation grid.
            for i in 0..driver.num_inputs() {
                let idx = driver.block().input_indices[i];
                let val = if (combo >> i) & 1 != 0 { u64::MAX } else { 0 };
                sim.set_logic_value_by_idx(idx, val);
            }

            let inputs = binding.gather_inputs(&sim);
            let outputs = driver
                .evaluate(&mut sim, &inputs)
                .expect("binding_pilot evaluate");

            let expected = truth_table[combo as usize];
            assert_eq!(
                outputs[0] != 0,
                expected,
                "binding_pilot mismatch at combo={:05b}",
                combo
            );
        }
    }
}
