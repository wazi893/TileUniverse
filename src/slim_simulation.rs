//! EPIC 105b: SlimSimulation - Memory-optimized simulation for maximum scale
//!
//! Drops debug/physics overhead to minimize memory footprint:
//! - No last_change tracking (saves 88 bytes/tile)
//! - No physics fields (heat, charge, etc.) (saves ~48 bytes/tile)
//! - No quantum tile lookup (saves 16 bytes/tile)
//! - No coupled/reaction fields
//!
//! Memory per tile: ~49 bytes (vs 197 bytes in full Simulation)
//! Target: 4x larger grids in same RAM

use crate::dirty_bitset::DirtyBitset;
use crate::field::FieldGrid;
use crate::tile::Tile;
use crate::tile_meta::TileType;
use crate::tilemap::Tilemap;
use std::sync::atomic::Ordering;

/// Memory-optimized simulation for maximum grid scale.
///
/// Uses ~49 bytes/tile vs ~197 bytes/tile in full Simulation.
/// Supports grids up to 4x larger in the same memory.
pub struct SlimSimulation {
    pub tilemap: Tilemap,
    pub dirty: DirtyBitset,
    pub global_clock: bool,
    prev_clock: bool,
    // EPIC 38: fast-path caches (kept - essential for performance)
    pub meta_fast: Vec<TileType>,
    neighbors4: Vec<[u32; 4]>,
    // Minimal fields - just logic and region
    #[allow(dead_code)]
    logic_field: FieldGrid<u32>,
    #[allow(dead_code)]
    region_field: FieldGrid<u32>,
    // Sprint 206: Weighted via mask + shift (parity with Simulation)
    tile_mask: Vec<u64>,
    tile_shift: Vec<u8>,
}

impl SlimSimulation {
    /// Create a new slim simulation with custom dimensions (1 layer)
    pub fn with_size(width: usize, height: usize) -> Self {
        Self::with_size_layered(width, height, 1)
    }

    /// Create a new slim simulation with custom dimensions and multiple layers
    pub fn with_size_layered(width: usize, height: usize, num_layers: usize) -> Self {
        let tilemap = Tilemap::with_size_layered(width, height, num_layers);
        let tile_count = tilemap.tile_count();
        let layer_size = tilemap.layer_size;
        let dirty = DirtyBitset::new(tile_count);

        // Precompute neighbor indices (layer-aware — Sprint 80)
        let mut neighbors4: Vec<[u32; 4]> = Vec::with_capacity(tile_count);
        for idx in 0..tile_count {
            let layer_base = (idx / layer_size) * layer_size;
            let within = idx % layer_size;
            let x = within % width;
            let local_y = within / width;
            let left = if x > 0 {
                (layer_base + local_y * width + (x - 1)) as u32
            } else {
                u32::MAX
            };
            let right = if x + 1 < width {
                (layer_base + local_y * width + (x + 1)) as u32
            } else {
                u32::MAX
            };
            let up = if local_y > 0 {
                (layer_base + (local_y - 1) * width + x) as u32
            } else {
                u32::MAX
            };
            let down = if local_y + 1 < height {
                (layer_base + (local_y + 1) * width + x) as u32
            } else {
                u32::MAX
            };
            neighbors4.push([left, right, up, down]);
        }

        let meta_fast = vec![TileType::Wire; tile_count];
        let logic_field = FieldGrid::new(width, height, 0u32);
        let region_field = FieldGrid::new(width, height, 0u32);

        Self {
            tilemap,
            dirty,
            global_clock: false,
            prev_clock: false,
            meta_fast,
            neighbors4,
            logic_field,
            region_field,
            tile_mask: vec![u64::MAX; tile_count],
            tile_shift: vec![0u8; tile_count],
        }
    }

    /// Get the width of this simulation's tilemap
    #[inline]
    pub fn width(&self) -> usize {
        self.tilemap.width
    }

    /// Get the height of this simulation's tilemap
    #[inline]
    pub fn height(&self) -> usize {
        self.tilemap.height
    }

    /// Get the total tile count
    #[inline]
    pub fn tile_count(&self) -> usize {
        self.tilemap.tile_count()
    }

    /// Set tile type at position
    pub fn set_tile(&mut self, x: usize, y: usize, tile_type: TileType) {
        if let Some(t) = self.tilemap.get_tile_mut(x, y) {
            t.meta.tile_type = tile_type;
            let idx = y * self.tilemap.width + x;
            if let Some(m) = self.meta_fast.get_mut(idx) {
                *m = tile_type;
            }
        }
    }

    /// Get logic value at position
    pub fn get_logic_at(&self, x: usize, y: usize) -> u64 {
        if let Some(t) = self.tilemap.get_tile(x, y) {
            t.logic.load(Ordering::Relaxed)
        } else {
            0
        }
    }

    /// Set logic value at position
    pub fn set_logic_value(&self, x: usize, y: usize, value: u64) -> bool {
        if let Some(t) = self.tilemap.get_tile(x, y) {
            t.logic.store(value, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    #[inline(always)]
    fn tile_type_at(&self, idx: usize) -> TileType {
        self.meta_fast.get(idx).copied().unwrap_or(TileType::Wire)
    }

    /// Get tile type at coordinates (for visualization)
    pub fn get_tile_type(&self, x: usize, y: usize) -> TileType {
        let idx = y * self.tilemap.width + x;
        self.tile_type_at(idx)
    }

    /// Get tile value at coordinates (for visualization)
    pub fn get_tile_value(&self, x: usize, y: usize) -> u64 {
        self.get_logic_at(x, y)
    }

    /// Set the mask for a WeightedViaUp/WeightedViaDown tile by index.
    pub fn set_tile_mask(&mut self, idx: usize, mask: u64) {
        if idx < self.tile_mask.len() {
            self.tile_mask[idx] = mask;
        }
    }

    /// Set the right-shift for a WeightedViaUp/WeightedViaDown tile by index.
    pub fn set_tile_shift(&mut self, idx: usize, shift: u8) {
        if idx < self.tile_shift.len() {
            self.tile_shift[idx] = shift;
        }
    }

    #[inline(always)]
    fn load_logic_idx(&self, idx_u32: u32) -> u64 {
        if idx_u32 == u32::MAX {
            0
        } else {
            self.tilemap.tiles[idx_u32 as usize]
                .logic
                .load(Ordering::Relaxed)
        }
    }

    /// Evaluate a single tile and return whether it changed
    #[inline(always)]
    pub fn eval_tile(&mut self, idx: usize) -> bool {
        if idx >= self.tilemap.tiles.len() {
            return false;
        }

        let n = &self.neighbors4[idx];
        let left = self.load_logic_idx(n[0]);
        let right = self.load_logic_idx(n[1]);
        let up = self.load_logic_idx(n[2]);
        let down = self.load_logic_idx(n[3]);

        let tile = &self.tilemap.tiles[idx];
        let current = tile.logic.load(Ordering::Relaxed);
        let tt = self.tile_type_at(idx);

        let new_out = self.compute_tile_output(tt, left, right, up, down, current, idx);

        if new_out != current {
            tile.logic.store(new_out, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Compute tile output based on type and inputs
    #[inline(always)]
    fn compute_tile_output(
        &self,
        tt: TileType,
        left: u64,
        right: u64,
        up: u64,
        down: u64,
        current: u64,
        idx: usize,
    ) -> u64 {
        match tt {
            TileType::Wire => left | right | up | down,
            TileType::And => left & right,
            TileType::Or => left | right,
            TileType::Xor => left ^ right,
            TileType::Not => !left,
            TileType::Latch => {
                if up != 0 {
                    left
                } else {
                    current
                }
            }
            TileType::Register8 => {
                // Simplified: just pass through on clock
                // Mask to 8 bits: Register8 enforces architectural width (Sprint 86.1)
                if self.global_clock && !self.prev_clock {
                    left & 0xFF
                } else {
                    current
                }
            }
            TileType::Register64 => {
                if self.global_clock && !self.prev_clock {
                    left // Full u64, no mask
                } else {
                    current
                }
            }
            TileType::ClockGlobal => {
                if self.global_clock {
                    1
                } else {
                    0
                }
            }
            // Arithmetic tiles
            TileType::Add => left.wrapping_add(right),
            TileType::Sub => left.wrapping_sub(right),
            TileType::Mul => left.wrapping_mul(right),
            TileType::Div => {
                if right != 0 {
                    left / right
                } else {
                    0
                }
            }
            TileType::Mod => {
                if right != 0 {
                    left % right
                } else {
                    0
                }
            }
            TileType::Shl => left.wrapping_shl((right & 63) as u32),
            TileType::Shr => left.wrapping_shr((right & 63) as u32),
            // Comparison tiles
            TileType::Lt => {
                if left < right {
                    1
                } else {
                    0
                }
            }
            TileType::Gt => {
                if left > right {
                    1
                } else {
                    0
                }
            }
            TileType::Eq => {
                if left == right {
                    1
                } else {
                    0
                }
            }
            TileType::Neq => {
                if left != right {
                    1
                } else {
                    0
                }
            }
            TileType::Lte => {
                if left <= right {
                    1
                } else {
                    0
                }
            }
            TileType::Gte => {
                if left >= right {
                    1
                } else {
                    0
                }
            }
            // Routing/special
            TileType::Mux => {
                if up != 0 {
                    right
                } else {
                    left
                }
            }
            TileType::Zero => {
                if left == 0 {
                    1
                } else {
                    0
                }
            }
            TileType::Neg => (-(left as i64)) as u64,
            TileType::Abs => (left as i64).unsigned_abs(),
            // Memory tiles
            TileType::Ram => {
                if up != 0 {
                    left
                } else {
                    current
                }
            }
            TileType::Counter => {
                if up != 0 {
                    current.wrapping_add(1)
                } else {
                    current
                }
            }
            TileType::Const => current,

            // === Wire Crossing Tiles ===
            TileType::Cross => {
                let h_mask: u64 = 0x0000_0000_FFFF_FFFF;
                let v_mask: u64 = 0xFFFF_FFFF_0000_0000;
                let h_signal = (left & h_mask) | (right & h_mask);
                let v_signal = (up & v_mask) | (down & v_mask);
                h_signal | v_signal
            }
            TileType::WireH => left | right,
            TileType::WireV => up | down,
            // WireDown: unidirectional, reads only from up
            TileType::WireDown => up,
            // WireRight: unidirectional, reads only from left
            TileType::WireRight => left,
            TileType::WireUp => down,
            TileType::WireLeft => right,
            TileType::ComponentOutput => current,
            TileType::BusInterface => current, // No bus support in slim mode
            TileType::MemoryPort => current,   // No memory controller support in slim mode
            TileType::ClockDivider => current, // No multi-clock support in slim mode
            TileType::Synchronizer => current,

            // === CPU Building Blocks ===
            TileType::Decoder3to8 => {
                let addr = (left & 0b111) as u32;
                1u64 << addr
            }
            TileType::CarryDetect => {
                if left > right {
                    u64::MAX
                } else {
                    0
                }
            }
            TileType::Decoder6to64 => 1u64 << (left & 63),
            TileType::Mux8to1 => {
                let sel = (right & 0b111) as u32;
                let shift = sel * 8;
                (left >> shift) & 0xFF
            }
            TileType::Mux16to1 => {
                let sel = (right & 0xF) as usize;
                let data = if sel < 8 { left } else { up };
                (data >> ((sel & 7) * 8)) & 0xFF
            }
            TileType::Mux4to1 => {
                let sel = (down & 0b11) as u32;
                let shift = sel * 8;
                (up >> shift) & 0xFF
            }
            TileType::Demux1to8 => {
                let data = up & 0xFF;
                let sel = (left & 0b111) as u32;
                let shift = sel * 8;
                data << shift
            }
            TileType::RegEnable => {
                if up != 0 && (right & 1) != 0 {
                    left
                } else {
                    current
                }
            }
            TileType::ProgramCounter => {
                if up != 0 {
                    if (right & 1) != 0 {
                        left
                    } else {
                        current.wrapping_add(1)
                    }
                } else {
                    current
                }
            }

            // === SPRINT 66: Evolutionary Selection ===
            TileType::Selector => {
                // Fitness = popcount (number of set bits)
                let my_fitness = (current as u32).count_ones();
                let mut best_val = current;
                let mut best_fitness = my_fitness;

                // Compare with neighbors (only if they're also Selector tiles)
                // In CPU mode, we check all 4 neighbors and pick highest fitness
                let neighbors = [left, right, up, down];
                for &n in &neighbors {
                    let n_fitness = (n as u32).count_ones();
                    if n_fitness > best_fitness {
                        best_fitness = n_fitness;
                        best_val = n;
                    }
                }
                best_val
            }

            // Unsupported in slim mode
            TileType::VmSpawner | TileType::VmStatus | TileType::QDemo => current,
            // CPU Tiles (Identity logic for CA, handled by Higher Level CPU sim)
            TileType::CpuHead | TileType::Register | TileType::Console => current,

            // === Ising Mode Tiles ===
            TileType::IsingNode => {
                // P-bit node: compute local field from neighbors and flip stochastically
                // Spins encoded: 0 = spin down (-1), 1+ = spin up (+1)
                // Local field = sum of neighbor spins (each neighbor contributes ±1)
                let s_left = if left != 0 { 1i64 } else { -1i64 };
                let s_right = if right != 0 { 1i64 } else { -1i64 };
                let s_up = if up != 0 { 1i64 } else { -1i64 };
                let s_down = if down != 0 { 1i64 } else { -1i64 };

                // Local field (negative J = antiferromagnetic for MaxCut)
                let local_field = -(s_left + s_right + s_up + s_down);

                // Deterministic threshold for tile-based simulation
                // (For stochastic: use IsingGrid which has proper RNG)
                // If local field > 0, prefer spin up; if < 0, prefer spin down
                if local_field > 0 {
                    1 // spin up
                } else if local_field < 0 {
                    0 // spin down
                } else {
                    current // tie - keep current
                }
            }
            TileType::IsingBias => {
                // External field source - outputs constant bias value
                current
            }

            // === Phase 1: Fully Tile-Based CPU ===
            TileType::AddCarry => {
                let sum = (left as u16).wrapping_add(right as u16);
                (sum & 0x1FF) as u64
            }
            TileType::SubBorrow => {
                let a = left & 0xFF;
                let b = right & 0xFF;
                let diff = (a as u16).wrapping_sub(b as u16);
                let borrow = if a < b { 1u64 } else { 0u64 };
                (diff & 0xFF) as u64 | (borrow << 8)
            }
            TileType::BitSelect => {
                if (left >> (right & 63)) & 1 != 0 {
                    u64::MAX
                } else {
                    0
                }
            }

            // === Wire Crossing Tiles ===
            TileType::WireCross => (left & 0xFFFF_FFFF) | (up & 0xFFFF_FFFF_0000_0000),
            TileType::WireCrossVert => (right & 0xFFFF_FFFF) | (up & 0xFFFF_FFFF_0000_0000),
            TileType::VBusIn => (up & 0xFFFF_FFFF) << 32,
            TileType::VBusOut => (up >> 32) & 0xFFFF_FFFF,

            // === Multi-Layer Via Tiles ===
            TileType::ViaUp => {
                let layer_size = self.tilemap.layer_size;
                let target = idx + layer_size;
                if target < self.tilemap.tiles.len() {
                    self.tilemap.tiles[target].logic.load(Ordering::Relaxed)
                } else {
                    0
                }
            }
            TileType::ViaDown => {
                let layer_size = self.tilemap.layer_size;
                if idx >= layer_size {
                    self.tilemap.tiles[idx - layer_size]
                        .logic
                        .load(Ordering::Relaxed)
                } else {
                    0
                }
            }

            // Sprint 160/206: Weighted Vias with shift+mask (parity with Simulation)
            TileType::WeightedViaUp => {
                let layer_size = self.tilemap.layer_size;
                let target = idx + layer_size;
                if target < self.tilemap.tiles.len() {
                    let source = self.tilemap.tiles[target].logic.load(Ordering::Relaxed);
                    (source >> self.tile_shift[idx]) & self.tile_mask[idx]
                } else {
                    0
                }
            }
            TileType::WeightedViaDown => {
                let layer_size = self.tilemap.layer_size;
                if idx >= layer_size {
                    let source = self.tilemap.tiles[idx - layer_size]
                        .logic
                        .load(Ordering::Relaxed);
                    (source >> self.tile_shift[idx]) & self.tile_mask[idx]
                } else {
                    0
                }
            }

            // Sprint 183: Threshold Vias — slim mode treats as identity (no threshold storage)
            TileType::ThresholdViaUp => {
                let layer_size = self.tilemap.layer_size;
                let target = idx + layer_size;
                if target < self.tilemap.tiles.len() {
                    self.tilemap.tiles[target].logic.load(Ordering::Relaxed)
                } else {
                    0
                }
            }
            TileType::ThresholdViaDown => {
                let layer_size = self.tilemap.layer_size;
                if idx >= layer_size {
                    self.tilemap.tiles[idx - layer_size]
                        .logic
                        .load(Ordering::Relaxed)
                } else {
                    0
                }
            }
        }
    }

    /// Evaluate all tiles in the grid (benchmark mode - no delta cycle limit)
    pub fn eval_full_grid(&mut self) -> (u32, u32) {
        let mut eval_count: u32 = 0;
        let mut change_count: u32 = 0;

        for idx in 0..self.tilemap.tiles.len() {
            let n = &self.neighbors4[idx];
            let left = self.load_logic_idx(n[0]);
            let right = self.load_logic_idx(n[1]);
            let up = self.load_logic_idx(n[2]);
            let down = self.load_logic_idx(n[3]);

            let tile = &self.tilemap.tiles[idx];
            let current = tile.logic.load(Ordering::Relaxed);
            let tt = self.tile_type_at(idx);
            let new_out = self.compute_tile_output(tt, left, right, up, down, current, idx);

            eval_count += 1;
            if new_out != current {
                tile.logic.store(new_out, Ordering::Relaxed);
                change_count += 1;
            }
        }

        (eval_count, change_count)
    }

    /// Toggle global clock
    pub fn tick_clock(&mut self) {
        self.prev_clock = self.global_clock;
        self.global_clock = !self.global_clock;
    }

    /// Calculate actual memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let tile_count = self.tile_count();

        // tilemap.tiles: Vec<Tile> - 16 bytes each
        let tiles_bytes = tile_count * std::mem::size_of::<Tile>();

        // dirty bitset: ~1 bit per tile
        let dirty_bytes = (tile_count + 7) / 8;

        // meta_fast: Vec<TileType> - 1 byte each
        let meta_bytes = tile_count;

        // neighbors4: Vec<[u32; 4]> - 16 bytes each
        let neighbors_bytes = tile_count * 16;

        // logic_field: FieldGrid<u32> - 4 bytes each
        let logic_field_bytes = tile_count * 4;

        // region_field: FieldGrid<u32> - 4 bytes each
        let region_field_bytes = tile_count * 4;

        tiles_bytes
            + dirty_bytes
            + meta_bytes
            + neighbors_bytes
            + logic_field_bytes
            + region_field_bytes
    }

    /// Calculate memory usage in MB
    pub fn memory_usage_mb(&self) -> f64 {
        self.memory_usage_bytes() as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slim_basic_operations() {
        let mut sim = SlimSimulation::with_size(64, 64);
        assert_eq!(sim.width(), 64);
        assert_eq!(sim.height(), 64);
        assert_eq!(sim.tile_count(), 4096);

        // Set and get logic
        sim.set_logic_value(10, 10, 42);
        assert_eq!(sim.get_logic_at(10, 10), 42);

        // Set tile type
        sim.set_tile(10, 10, TileType::And);
    }

    #[test]
    fn test_slim_memory_much_smaller() {
        let slim = SlimSimulation::with_size(512, 512);
        let bytes_per_tile = slim.memory_usage_bytes() as f64 / slim.tile_count() as f64;

        // Should be well under 100 bytes per tile (vs 197 in full sim)
        assert!(
            bytes_per_tile < 100.0,
            "bytes_per_tile = {}",
            bytes_per_tile
        );
        println!("SlimSimulation: {:.1} bytes/tile", bytes_per_tile);
    }

    #[test]
    fn test_slim_eval_and_gate() {
        let mut sim = SlimSimulation::with_size(64, 64);

        // Set up: wire - AND - wire
        sim.set_tile(9, 10, TileType::Wire);
        sim.set_tile(10, 10, TileType::And);
        sim.set_tile(11, 10, TileType::Wire);

        // Set inputs
        sim.set_logic_value(9, 10, 0xFF);
        sim.set_logic_value(11, 10, 0x0F);

        // Eval AND gate
        let idx = 10 * 64 + 10;
        sim.eval_tile(idx);

        assert_eq!(sim.get_logic_at(10, 10), 0xFF & 0x0F);
    }

    /// Sprint 207 Deliverable C: Verify Simulation and SlimSimulation produce
    /// identical WeightedVia outputs for shift+mask configurations.
    #[test]
    fn test_slim_weighted_via_parity() {
        use crate::simulation::Simulation;

        let w = 8;
        let h = 8;

        // Test cases: (source_value, shift, mask, expected)
        let cases: Vec<(u64, u8, u64, u64)> = vec![
            (0xABCD_1234, 4, 0x0F, (0xABCD_1234 >> 4) & 0x0F), // extract nibble
            (0xFF, 0, 0x7F, 0xFF & 0x7F),                      // PC mask pattern
            (0x1234, 4, 0x03, (0x1234 >> 4) & 0x03),           // flag-WE pattern
            (0xDEAD_BEEF, 0, u64::MAX, 0xDEAD_BEEF),           // identity
            (0xFF00, 8, 0xFF, 0xFF),                           // full byte extract
        ];

        for (i, &(source, shift, mask, expected)) in cases.iter().enumerate() {
            // --- Full Simulation ---
            let mut full = Simulation::with_size_layered(w, h, 2);
            let layer_size = w * h;
            // L1 source: Const at (4,4)
            let src_idx = layer_size + 4 * w + 4;
            full.set_tile_3d(4, 4, 1, TileType::Const);
            full.set_logic_value_by_idx(src_idx, source);
            // L0 WeightedViaUp at (4,4) reads L1
            let via_idx = 4 * w + 4;
            full.set_tile_3d(4, 4, 0, TileType::WeightedViaUp);
            full.set_tile_mask(via_idx, mask);
            full.set_tile_shift(via_idx, shift);
            full.eval_tile(via_idx);
            let full_result = full.get_logic_value_by_idx(via_idx);

            // --- Slim Simulation ---
            let mut slim = SlimSimulation::with_size_layered(w, h, 2);
            // L1 source: Const at (4,4)
            let slim_src_idx = layer_size + 4 * w + 4;
            slim.tilemap.tiles[slim_src_idx].meta.tile_type = TileType::Const;
            slim.meta_fast[slim_src_idx] = TileType::Const;
            slim.tilemap.tiles[slim_src_idx]
                .logic
                .store(source, std::sync::atomic::Ordering::Relaxed);
            // L0 WeightedViaUp at (4,4)
            let slim_via_idx = 4 * w + 4;
            slim.tilemap.tiles[slim_via_idx].meta.tile_type = TileType::WeightedViaUp;
            slim.meta_fast[slim_via_idx] = TileType::WeightedViaUp;
            slim.set_tile_mask(slim_via_idx, mask);
            slim.set_tile_shift(slim_via_idx, shift);
            slim.eval_tile(slim_via_idx);
            let slim_result = slim.tilemap.tiles[slim_via_idx]
                .logic
                .load(std::sync::atomic::Ordering::Relaxed);

            assert_eq!(
                full_result, expected,
                "case {i}: Simulation result {full_result:#x} != expected {expected:#x}"
            );
            assert_eq!(
                slim_result, expected,
                "case {i}: SlimSimulation result {slim_result:#x} != expected {expected:#x}"
            );
            assert_eq!(
                full_result, slim_result,
                "case {i}: parity mismatch: full={full_result:#x} slim={slim_result:#x}"
            );
        }
    }
}
