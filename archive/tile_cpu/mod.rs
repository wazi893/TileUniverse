//! TileFabric CPU - A processor built from simulation tiles
//!
//! Unlike the tile8 module which uses software emulation, TileFabric CPUs
//! execute by propagating signals through actual tile connections. Every
//! gate, register, and wire is a discrete tile evaluated by the simulation
//! engine.
//!
//! # Architecture
//!
//! ```text
//! +------------------+     +------------------+
//! |   Program ROM    |---->|   Instruction    |
//! |   (Const tiles)  |     |   Decoder        |
//! +------------------+     +------------------+
//!          ^                       |
//!          |                       v
//! +------------------+     +------------------+
//! | Program Counter  |<----|   Control Unit   |
//! | (ProgramCounter) |     |   (And/Or/Mux)   |
//! +------------------+     +------------------+
//!                                  |
//!          +----------+------------+----------+
//!          |          |            |          |
//!          v          v            v          v
//!     +--------+  +--------+  +--------+  +--------+
//!     | Reg 0  |  | Reg 1  |  | Reg 2  |  | Reg 3  |
//!     +--------+  +--------+  +--------+  +--------+
//!          |          |            |          |
//!          +----------+-----+------+----------+
//!                           |
//!                           v
//!                    +------------+
//!                    |    ALU     |
//!                    | Add/Sub/.. |
//!                    +------------+
//! ```
//!
//! # Key Difference from tile8
//!
//! - tile8: `cpu.step()` runs a Rust match statement
//! - tile_cpu: `cpu.tick(sim)` calls `sim.tick_with_delays()` - tiles execute
//!
//! # Example
//!
//! ```rust,ignore
//! use engine::tile_cpu::{TileCpuBuilder, TileCpu};
//! use engine::simulation::Simulation;
//!
//! let mut sim = Simulation::new();
//! let cpu = TileCpuBuilder::new()
//!     .with_program(&[0x12, 0x17, 0x31]) // LDI R0,#2; LDI R1,#3; ADD R0,R1
//!     .build(&mut sim);
//!
//! // Execute via tile simulation - no software shortcuts
//! for _ in 0..10 {
//!     let stats = cpu.tick(&mut sim);
//!     println!("Critical path: {} deltas", stats.critical_path_deltas);
//! }
//!
//! assert_eq!(cpu.read_reg(&sim, 0), 5); // 2 + 3 = 5
//! ```

mod builder;
mod components;
mod datapath;
mod execute;
mod layout;
pub mod minimal;
mod wiring;

pub use builder::TileCpuBuilder;
pub use wiring::WiringContext;
pub use components::{AluOp, ControlSignals};
pub use datapath::TileCpuDatapath;
pub use execute::{TileCpu, TileCpuMetrics};
pub use layout::{TileLayout, PlacedComponent, WireRoute};
pub use minimal::MinimalCpu;

/// Width of the CPU datapath in bits
pub const DATAPATH_WIDTH: usize = 8;

/// Number of general-purpose registers
pub const NUM_REGISTERS: usize = 4;

/// Maximum ROM size in bytes
pub const MAX_ROM_SIZE: usize = 64;

/// Maximum RAM size in bytes
pub const MAX_RAM_SIZE: usize = 64;

/// Maximum propagation deltas before declaring non-convergence
pub const MAX_PROPAGATION_DELTAS: u32 = 1000;

#[cfg(test)]
mod tests;
