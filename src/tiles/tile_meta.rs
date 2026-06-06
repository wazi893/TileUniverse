#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TileType {
    // === Original Logic Gates (0-10) ===
    Wire,        // 0: output = left | right | up | down
    And,         // 1: output = left & right
    Or,          // 2: output = left | right
    Xor,         // 3: output = left ^ right
    Not,         // 4: output = ~left
    Latch,       // 5: output = clock ? left : current
    Register8,   // 6: output = rising_edge ? left : current
    ClockGlobal, // 7: output = clock ? MAX : 0
    VmSpawner,   // 8: passthrough (demo)
    VmStatus,    // 9: passthrough (demo)
    QDemo,       // 10: quantum feedback (demo)

    // === EPIC 103: Arithmetic Tiles (11-20) ===
    Add, // 11: output = left + right (wrapping)
    Sub, // 12: output = left - right (wrapping)
    Mul, // 13: output = left * right (lower bits)
    Div, // 14: output = left / right (0 if right == 0)
    Mod, // 15: output = left % right (0 if right == 0)
    Shl, // 16: output = left << (right & 63)
    Shr, // 17: output = left >> (right & 63)

    // === EPIC 103: Comparison Tiles (21-26) ===
    Lt,  // 21: output = (left < right) ? MAX : 0
    Gt,  // 22: output = (left > right) ? MAX : 0
    Eq,  // 23: output = (left == right) ? MAX : 0
    Neq, // 24: output = (left != right) ? MAX : 0
    Lte, // 25: output = (left <= right) ? MAX : 0
    Gte, // 26: output = (left >= right) ? MAX : 0

    // === EPIC 103: Routing & Special Tiles (27-30) ===
    Mux,  // 27: output = (up != 0) ? left : right
    Zero, // 28: output = (left == 0) ? MAX : 0
    Neg,  // 29: output = wrapping negation (-left)
    Abs,  // 30: output = if left negative (MSB set), -left else left

    // === EPIC 104: Memory Tiles (31-35) ===
    Ram,     // 31: output = (up != 0) ? left : current (write-enable gated)
    Counter, // 32: output = (up != 0) ? current + 1 : current
    Const,   // 33: output = preset constant (configured externally)

    // === Wire Crossing Tiles (36-38) ===
    // These enable bus routing without signal interference
    Cross, // 36: horizontal/vertical crossing - bits 0-31 = left|right, bits 32-63 = up|down
    WireH, // 37: horizontal wire only - output = left | right (ignores up/down)
    WireV, // 38: vertical wire only - output = up | down (ignores left/right)

    // === Unidirectional Wires (39-40) ===
    // These enable one-way signal flow to prevent feedback loops
    WireDown = 39,  // 39: reads from up only (signal flows downward)
    WireRight = 51, // 51: reads from left only (signal flows rightward)

    // === Component System Tiles (52-54) ===
    WireUp = 52,          // 52: reads from down only (signal flows upward)
    WireLeft = 53,        // 53: reads from right only (signal flows leftward)
    ComponentOutput = 54, // 54: output port of a behavioral component

    // === Bus Architecture Tiles (55) ===
    BusInterface = 55, // 55: bridge between grid tile and shared bus

    // === Memory Controller Tiles (56) ===
    MemoryPort = 56, // 56: addressable memory port (addr=left, data=right, we=up)

    // === EPIC 116: CPU Tiles (40-42) ===
    CpuHead = 40,  // 40: CPU Control Unit (Stores IP)
    Register = 41, // 41: General Purpose Register (Stores Value)
    Console = 42,  // 42: Output Device (Displays Value)

    // === CPU Building Blocks (43-47) ===
    // Essential components for constructing CPUs from tiles
    Decoder3to8 = 43,    // 43: 3-bit address (left bits 0-2) → 8-bit one-hot output
    Mux8to1 = 44,        // 44: 8 packed bytes (left) + 3-bit select (right) → selected byte
    Demux1to8 = 45,      // 45: 1 byte (up) + 3-bit select (left) → byte at selected position
    RegEnable = 46,      // 46: register with enable - captures left when up!=0 AND right&1!=0
    ProgramCounter = 47, // 47: PC - on clock: if jump(right&1) then load left, else PC+1

    // === SPRINT 66: Evolutionary Selection (48) ===
    Selector = 48, // 48: Fitness-based selection - adopts neighbor with highest popcount

    // === Ising Mode Tiles (49-50) ===
    IsingNode = 49, // 49: P-bit node - stochastic spin based on neighbor field
    IsingBias = 50, // 50: External field source for Ising nodes

    // === Multi-Clock Domain Tiles (57-58) ===
    ClockDivider = 57, // 57: outputs assigned domain's clock signal
    Synchronizer = 58, // 58: 2-FF CDC synchronizer (captures left on dest domain rising edge)

    // === Vertical-I/O Mux (59) ===
    Mux4to1 = 59, // 59: 4 packed bytes (up) + 2-bit select (down) → selected byte

    // === Phase 1: Fully Tile-Based CPU (60-61) ===
    AddCarry = 60, // 60: output = ((left as u16 + right as u16) & 0x1FF) as u64 (bit 8 = carry-out)
    BitSelect = 61, // 61: output = if (left >> (right & 63)) & 1 != 0 { u64::MAX } else { 0 }

    // === Wire Crossing Tiles (62-64) ===
    // Enable signal routing through crossing points without interference.
    // Uses bit-partitioned buses: bits 0-31 = horizontal, bits 32-63 = vertical.
    WireCross = 62,     // 62: unidirectional crossing - h=left[0:31], v=up[32:63]
    VBusIn = 63,        // 63: shift signal from low bus to high bus for vertical crossing entry
    VBusOut = 64,       // 64: shift signal from high bus to low bus for vertical crossing exit
    WireCrossVert = 65, // 65: unidirectional crossing - h=right[0:31], v=up[32:63] (for WL chains)

    // === Sprint 80: Multi-Layer Via Tiles (66-67) ===
    // Enable signal routing between layers without in-plane crossing hacks.
    // Layers are routing resources: layer 0 = logic, layers 1+ = routing planes.
    ViaUp = 66, // 66: reads from same (x,y) on layer z+1 (signal flows upward between layers)
    ViaDown = 67, // 67: reads from same (x,y) on layer z-1 (signal flows downward between layers)

    // === Sprint 82: SubBorrow + Mux16to1 ===
    SubBorrow = 68, // 68: output = (left - right) & 0xFF | (borrow << 8), borrow = (left & 0xFF) < (right & 0xFF)
    Mux16to1 = 69, // 69: 16 packed bytes (left=0-7, up=8-15) + 4-bit select (right) → selected byte

    // === Sprint 127: 64-bit Scaling Primitives (70-72) ===
    Register64 = 70,  // 70: output = rising_edge ? left : current (full u64, no mask)
    CarryDetect = 71, // 71: output = (left > right) ? MAX : 0 (unsigned overflow detector)
    Decoder6to64 = 72, // 72: 6-bit address (left bits 0-5) → 64-bit one-hot output

    // === Sprint 160: Weighted Via Tiles (73-74) ===
    // Smart vias that apply a stored mask during inter-layer signal transfer.
    // output = source_layer_value & tile_mask[idx]. Mask defaults to u64::MAX (identity).
    WeightedViaUp = 73,   // 73: reads from layer z+1, applies mask
    WeightedViaDown = 74, // 74: reads from layer z-1, applies mask

    // === Sprint 183: Threshold Via Tiles (75-76) ===
    // Gated vias: output = cross_layer_signal if nonzero_neighbor_count >= threshold else 0.
    // Threshold stored in tile_threshold[idx] (default 1).
    ThresholdViaUp = 75, // 75: reads from layer z+1, gated by in-plane neighbor count
    ThresholdViaDown = 76, // 76: reads from layer z-1, gated by in-plane neighbor count
}

impl TileType {
    /// Propagation delay in delta cycles.
    ///
    /// - 0 = synchronous (edge-triggered), evaluates immediately on clock
    /// - 1+ = combinational delay through this tile type
    ///
    /// These values model realistic gate delays where wires have minimal
    /// delay, simple gates take 2 cycles, and complex arithmetic takes more.
    #[inline]
    pub const fn propagation_delay(&self) -> u8 {
        match self {
            // === Wires: 1 delta per tile (RC delay of interconnect) ===
            TileType::Wire => 1,
            TileType::WireH => 1,
            TileType::WireV => 1,
            TileType::WireDown => 1,
            TileType::WireRight => 1,
            TileType::WireUp => 1,
            TileType::WireLeft => 1,
            TileType::Cross => 1,

            // === Simple gates: 2 deltas (transistor switching) ===
            TileType::Not => 2,
            TileType::And => 2,
            TileType::Or => 2,
            TileType::Xor => 3, // XOR is typically 2 gates deep

            // === Comparison: 3-4 deltas (ripple through bits) ===
            TileType::Lt | TileType::Gt | TileType::Eq | TileType::Neq => 4,
            TileType::Lte | TileType::Gte => 4,
            TileType::Zero => 3,

            // === Arithmetic: scales with bit width ===
            TileType::Add | TileType::Sub | TileType::SubBorrow => 5, // Carry chain
            TileType::Neg | TileType::Abs => 5,
            TileType::Shl | TileType::Shr => 3,  // Barrel shifter
            TileType::Mul => 8,                  // Array multiplier
            TileType::Div | TileType::Mod => 12, // Iterative division

            // === Routing: 2-3 deltas (mux tree) ===
            TileType::Mux => 2,
            TileType::Decoder3to8 => 2,
            TileType::Mux8to1 | TileType::Mux16to1 => 3,
            TileType::Mux4to1 => 2,
            TileType::Demux1to8 => 2,

            // === Sequential: 0 (synchronous - captures on clock edge) ===
            TileType::Register8 => 0,
            TileType::Latch => 0,
            TileType::RegEnable => 0,
            TileType::ProgramCounter => 0,
            TileType::Ram => 0,
            TileType::Counter => 0,
            TileType::ClockGlobal => 0,
            TileType::Const => 0,

            // === CPU/Special: 0 (synchronous) ===
            TileType::CpuHead => 0,
            TileType::Register => 0,
            TileType::Console => 0,

            // === Component System ===
            TileType::ComponentOutput => 5,

            // === Bus Architecture ===
            TileType::BusInterface => 2,

            // === Memory Controller ===
            TileType::MemoryPort => 3,

            // === Multi-Clock Domains ===
            TileType::ClockDivider => 0,
            TileType::Synchronizer => 0,

            // === Phase 1: Fully Tile-Based CPU ===
            TileType::AddCarry => 5,  // Same as Add (carry chain)
            TileType::BitSelect => 2, // Simple shift + mask

            // === Sprint 127: 64-bit Scaling Primitives ===
            TileType::Register64 => 0,   // Sequential (edge-triggered)
            TileType::CarryDetect => 4,  // Comparison (same as Lt/Gt)
            TileType::Decoder6to64 => 3, // Wider decode tree than Decoder3to8

            // === Wire Crossing Tiles ===
            TileType::WireCross => 1,
            TileType::WireCrossVert => 1,
            TileType::VBusIn => 1,
            TileType::VBusOut => 1,

            // === Multi-Layer Via Tiles ===
            TileType::ViaUp => 1, // Inter-layer via, same delay as wire
            TileType::ViaDown => 1,

            // === Sprint 160: Weighted Via Tiles ===
            TileType::WeightedViaUp => 1, // Same delay as plain via
            TileType::WeightedViaDown => 1,

            // === Sprint 183: Threshold Via Tiles ===
            TileType::ThresholdViaUp => 1, // Same delay as plain via
            TileType::ThresholdViaDown => 1,

            // === Other ===
            TileType::Selector => 2,
            TileType::IsingNode => 1,
            TileType::IsingBias => 0,
            TileType::VmSpawner | TileType::VmStatus | TileType::QDemo => 1,
        }
    }

    /// Returns true if this tile is edge-triggered (synchronous).
    /// Sequential tiles have delay=0 because they capture on clock edge,
    /// not after combinational propagation.
    #[inline]
    pub const fn is_sequential(&self) -> bool {
        self.propagation_delay() == 0
    }

    /// Returns true if this tile is clock-sensitive (needs evaluation on clock edge)
    #[inline]
    pub const fn is_clock_sensitive(&self) -> bool {
        matches!(
            self,
            TileType::ClockGlobal
                | TileType::Latch
                | TileType::Register8
                | TileType::Register64
                | TileType::RegEnable
                | TileType::ProgramCounter
                | TileType::Counter
                | TileType::Ram
                | TileType::CpuHead
                | TileType::Register
                | TileType::ClockDivider
                | TileType::Synchronizer
        )
    }

    /// Returns true if this tile type is a wire variant.
    ///
    /// Used for distance-based wire delay calculations where wire tiles
    /// can have per-tile delay overrides based on segment length.
    #[inline]
    pub const fn is_wire(&self) -> bool {
        matches!(
            self,
            TileType::Wire
                | TileType::WireH
                | TileType::WireV
                | TileType::WireDown
                | TileType::WireRight
                | TileType::WireUp
                | TileType::WireLeft
                | TileType::Cross
                | TileType::WireCross
                | TileType::WireCrossVert
                | TileType::VBusIn
                | TileType::VBusOut
                | TileType::ViaUp
                | TileType::ViaDown
                | TileType::WeightedViaUp
                | TileType::WeightedViaDown
                | TileType::ThresholdViaUp
                | TileType::ThresholdViaDown
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TileMeta {
    pub tile_type: TileType,
    // future fields can be added later
}
