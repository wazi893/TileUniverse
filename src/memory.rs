use std::cell::Cell;

/// Definition of a memory bank.
pub struct MemoryBankDef {
    pub name: String,
    /// Number of u64 words in this bank.
    pub size: usize,
    /// Initial contents (if shorter than size, remaining words are 0).
    pub initial_data: Vec<u64>,
}

/// Runtime state of a memory bank.
pub struct MemoryBank {
    /// The backing store — Cell for interior mutability (accessed through &self in compute_tile_output).
    pub data: Vec<Cell<u64>>,
}

/// A connection between a MemoryPort tile and a memory bank.
pub struct MemoryPortConnection {
    /// Which bank this connects to (index into memory_banks).
    pub bank_idx: usize,
    /// Grid tile index of the MemoryPort tile.
    pub tile_idx: usize,
}
