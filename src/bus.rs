use std::cell::Cell;

/// How to resolve multiple writers in the same tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusArbitration {
    /// Lowest connection index wins (deterministic, simple).
    Priority,
    /// Bitwise OR of all writer values (open-drain bus).
    OrMerge,
}

/// Direction of a bus connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusDirection {
    /// Tile reads from bus. BusInterface output = bus data.
    Reader,
    /// Tile writes to bus. Bus data = tile's left neighbor input.
    Writer,
    /// Tile can both read and write (bidirectional).
    ReadWriter,
}

/// Definition of a bus.
pub struct BusDef {
    pub name: String,
    /// Number of u64 words on this bus (typically 1).
    pub width: usize,
    pub arbitration: BusArbitration,
}

/// Runtime state of a bus.
pub struct BusState {
    /// Current bus data (one u64 per word).
    pub data: Vec<Cell<u64>>,
    /// Indices into the bus_connections vec for connections to this bus.
    pub connection_indices: Vec<usize>,
    /// Per-word flag: has this word been written this tick (for Priority arbitration).
    pub word_written: Vec<Cell<bool>>,
}

/// A connection between a grid tile and a bus.
pub struct BusConnection {
    /// Which bus this connects to (index into bus_defs/bus_states).
    pub bus_idx: usize,
    /// Grid tile index of the BusInterface tile.
    pub tile_idx: usize,
    /// Which word of the bus this connection accesses (0 for single-word buses).
    pub word_offset: usize,
    /// Direction of data flow.
    pub direction: BusDirection,
}
