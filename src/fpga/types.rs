//! SPRINT 80: Type mappings between simulation and HDL
//!
//! This module defines the bridge types between TileUniverse's simulation
//! types and RustHDL's hardware types.

use rust_hdl::prelude::*;

// ============================================================================
// Bit Width Type Aliases
// ============================================================================

/// Standard tile value width (64 bits)
pub type TileBits = Bits<64>;

/// 8-bit data path (for CPU registers, instructions)
pub type DataBits8 = Bits<8>;

/// 16-bit data path
pub type DataBits16 = Bits<16>;

/// 32-bit data path
pub type DataBits32 = Bits<32>;

/// 3-bit selector (for 8-way mux/demux)
pub type Select3 = Bits<3>;

/// 8-bit one-hot encoding (decoder output)
pub type OneHot8 = Bits<8>;

// ============================================================================
// Signal Direction Aliases
// ============================================================================

/// Input signal with 64-bit tile value
pub type TileIn = Signal<In, TileBits>;

/// Output signal with 64-bit tile value
pub type TileOut = Signal<Out, TileBits>;

/// Input signal with 8-bit data
pub type DataIn8 = Signal<In, DataBits8>;

/// Output signal with 8-bit data
pub type DataOut8 = Signal<Out, DataBits8>;

/// Clock input signal
pub type ClockIn = Signal<In, Clock>;

/// Single-bit control input
pub type ControlIn = Signal<In, Bit>;

/// Single-bit control output
pub type ControlOut = Signal<Out, Bit>;

// ============================================================================
// Tile Type Codes (matching cuda_tiles.rs)
// ============================================================================

/// Tile type codes matching the CUDA kernel definitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TileTypeCode {
    // Routing
    Wire = 0,
    WireH = 37,
    WireV = 38,
    WireDown = 39,
    WireRight = 51,
    Cross = 36,

    // Logic gates
    And = 1,
    Or = 2,
    Xor = 3,
    Not = 4,

    // Sequential
    Latch = 5,
    Register8 = 6,
    ClockGlobal = 7,
    RegEnable = 46,
    ProgramCounter = 47,

    // Arithmetic
    Add = 11,
    Sub = 12,
    Mul = 13,
    Div = 14,
    Mod = 15,
    Shl = 16,
    Shr = 17,
    Neg = 29,
    Abs = 30,

    // Comparison
    Lt = 21,
    Gt = 22,
    Eq = 23,
    Neq = 24,
    Lte = 25,
    Gte = 26,

    // Routing/Special
    Mux = 27,
    Zero = 28,
    Decoder3to8 = 43,
    Mux8to1 = 44,
    Demux1to8 = 45,

    // Memory
    Ram = 31,
    Counter = 32,
    Const = 33,
}

impl TileTypeCode {
    /// Get the propagation delay in delta cycles
    pub fn propagation_delay(&self) -> u8 {
        match self {
            // 0 deltas (synchronous)
            Self::Latch
            | Self::Register8
            | Self::ClockGlobal
            | Self::Ram
            | Self::Counter
            | Self::Const
            | Self::RegEnable
            | Self::ProgramCounter => 0,

            // 1 delta
            Self::Wire
            | Self::WireH
            | Self::WireV
            | Self::WireDown
            | Self::WireRight
            | Self::Cross => 1,

            // 2 deltas
            Self::And | Self::Or | Self::Not | Self::Mux | Self::Decoder3to8 | Self::Demux1to8 => 2,

            // 3 deltas
            Self::Xor | Self::Zero | Self::Shl | Self::Shr | Self::Mux8to1 => 3,

            // 4 deltas
            Self::Lt | Self::Gt | Self::Eq | Self::Neq | Self::Lte | Self::Gte => 4,

            // 5 deltas
            Self::Add | Self::Sub | Self::Neg | Self::Abs => 5,

            // 8 deltas
            Self::Mul => 8,

            // 12 deltas
            Self::Div | Self::Mod => 12,
        }
    }

    /// Convert propagation delay to nanoseconds (assuming 10ns per delta)
    pub fn propagation_delay_ns(&self) -> f64 {
        self.propagation_delay() as f64 * 1.0 // 1ns per delta for 100MHz base
    }
}

// ============================================================================
// Port Definitions (documentation-only, not synthesizable blocks)
// ============================================================================

// Note: Port configurations are defined inline in each primitive.
// These type aliases above (TileIn, TileOut, etc.) are used directly.

// ============================================================================
// Utility Functions
// ============================================================================

/// Convert a u64 constant to Bits<64>
pub fn const_to_bits(value: u64) -> Bits<64> {
    value.into()
}

/// All ones (0xFFFFFFFFFFFFFFFF) - used for comparison true result
pub fn all_ones() -> Bits<64> {
    Bits::<64>::from(u64::MAX)
}

/// All zeros - used for comparison false result
pub fn all_zeros() -> Bits<64> {
    Bits::<64>::from(0u64)
}
