//! SPRINT 80: FPGA Synthesis Module
//!
//! This module provides hardware description language (HDL) generation for
//! synthesizing TileUniverse circuits to FPGAs using RustHDL.
//!
//! # Architecture
//!
//! The FPGA module bridges the simulation tile system to synthesizable hardware:
//!
//! ```text
//! TileType (simulation) → LogicBlock (RustHDL) → Verilog → FPGA Bitstream
//! ```
//!
//! # Submodules
//!
//! - `types`: Type mappings between simulation and HDL
//! - `primitives`: Hardware implementations of all 51 TileTypes
//! - `synthesis`: Grid-to-Verilog transpilation
//!
//! # Usage
//!
//! ```ignore
//! use engine::fpga::prelude::*;
//!
//! // Create a simple AND gate
//! let mut and_gate = AndGate::default();
//! and_gate.connect_all();
//!
//! // Generate Verilog
//! let verilog = and_gate.generate_verilog();
//! ```
//!
//! # Feature Flag
//!
//! This module requires the `fpga` feature:
//! ```toml
//! [dependencies]
//! engine = { version = "0.1", features = ["fpga"] }
//! ```

pub mod primitives;
pub mod synthesis;
pub mod types;

// Re-export RustHDL prelude for convenience
pub use rust_hdl::prelude::*;

// Re-export our types
pub use types::*;

/// Prelude for convenient imports
pub mod prelude {
    pub use super::primitives::*;
    pub use super::synthesis::*;
    pub use super::types::*;
    pub use rust_hdl::prelude::*;
}

/// Trait for converting simulation tiles to synthesizable HDL
pub trait Synthesizable {
    /// The RustHDL LogicBlock type for this tile
    type HdlBlock: Block;

    /// Create a new HDL block instance
    fn to_hdl(&self) -> Self::HdlBlock;

    /// Get the propagation delay in nanoseconds (for timing constraints)
    fn propagation_delay_ns(&self) -> f64;

    /// Get port definitions for this tile
    fn port_count(&self) -> (usize, usize); // (inputs, outputs)
}

/// Configuration for Verilog generation
#[derive(Debug, Clone)]
pub struct SynthesisConfig {
    /// Target clock frequency in MHz
    pub target_clock_mhz: f64,

    /// Generate timing constraints (SDC format)
    pub generate_timing: bool,

    /// Infer BRAM for RAM tiles
    pub infer_bram: bool,

    /// Use DSP blocks for arithmetic
    pub use_dsp: bool,

    /// Target FPGA family (for optimization hints)
    pub target_family: FpgaFamily,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            target_clock_mhz: 100.0,
            generate_timing: true,
            infer_bram: true,
            use_dsp: true,
            target_family: FpgaFamily::Xilinx7Series,
        }
    }
}

/// Supported FPGA families
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpgaFamily {
    /// Xilinx 7-Series (Artix-7, Kintex-7, Virtex-7)
    Xilinx7Series,
    /// Xilinx UltraScale/UltraScale+
    XilinxUltraScale,
    /// Intel/Altera Cyclone V
    IntelCycloneV,
    /// Intel/Altera Stratix 10
    IntelStratix10,
    /// Lattice ECP5
    LatticeEcp5,
    /// Generic (no vendor-specific optimizations)
    Generic,
}

/// Result of synthesis operation
#[derive(Debug)]
pub struct SynthesisResult {
    /// Generated Verilog code
    pub verilog: String,

    /// Timing constraints (SDC format) if requested
    pub timing_constraints: Option<String>,

    /// Estimated resource usage
    pub resource_estimate: ResourceEstimate,
}

/// Estimated FPGA resource usage
#[derive(Debug, Default)]
pub struct ResourceEstimate {
    /// Lookup tables (LUTs)
    pub luts: usize,

    /// Flip-flops (FFs)
    pub flip_flops: usize,

    /// Block RAM (18Kb blocks)
    pub bram_18k: usize,

    /// DSP blocks
    pub dsp_blocks: usize,

    /// Estimated critical path delay in ns
    pub critical_path_ns: f64,

    /// Maximum estimated clock frequency in MHz
    pub max_clock_mhz: f64,
}
