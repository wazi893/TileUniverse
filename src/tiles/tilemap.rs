use crate::tile::Tile;
use crate::tile_meta::{TileMeta, TileType};
use std::cell::Cell;

// Default dimensions (kept for backward compatibility)
pub const WIDTH: usize = 512;
pub const HEIGHT: usize = 512;
pub const TILE_COUNT: usize = WIDTH * HEIGHT;

pub struct Tilemap {
    pub tiles: Vec<Tile>,
    /// Per-tile logic values, SoA and lane-interleaved (Sprints 385/386):
    /// `vals[idx * lane_stride + lane_offset]` is the addressed tile's value.
    /// Normal mode: stride 1, offset 0 — a plain dense array. Lane mode
    /// (SIMT fabric): stride = lanes, offset = active lane, and the wide
    /// kernels evaluate the whole buffer. Length is always
    /// `(tile_count + 1) * lane_stride`; the trailing row is a permanent
    /// ZERO ROW (lane-kernel convention for boundary inputs).
    ///
    /// `Cell<u64>` (single-threaded interior mutability — the same invariant
    /// the old Relaxed-only `AtomicU64` actually relied on) keeps `&self`
    /// accessors and lets plain-u64 eval loops vectorize.
    vals: Vec<Cell<u64>>,
    lane_stride: usize,
    lane_offset: usize,
    /// Actual width of this tilemap (may differ from default WIDTH)
    pub width: usize,
    /// Actual height of this tilemap (may differ from default HEIGHT)
    pub height: usize,
    /// Number of vertical layers (default 1)
    pub num_layers: usize,
    /// Precomputed width * height (one layer's worth of tiles)
    pub layer_size: usize,
}

impl Tilemap {
    /// Create a new tilemap with default dimensions (512x512, 1 layer)
    pub fn new() -> Self {
        Self::with_size(WIDTH, HEIGHT)
    }

    /// Create a new tilemap with custom dimensions (1 layer)
    pub fn with_size(width: usize, height: usize) -> Self {
        Self::with_size_layered(width, height, 1)
    }

    /// Create a new tilemap with custom dimensions and multiple layers.
    ///
    /// Tiles are stored in a single flat array partitioned by layer:
    /// - Layer 0: indices [0, layer_size)
    /// - Layer z: indices [z*layer_size, (z+1)*layer_size)
    ///
    /// Within each layer, index = y * width + x (row-major).
    pub fn with_size_layered(width: usize, height: usize, num_layers: usize) -> Self {
        let num_layers = num_layers.max(1);
        let layer_size = width * height;
        let tile_count = layer_size * num_layers;
        debug_assert_eq!(tile_count, width * height * num_layers);
        debug_assert_eq!(layer_size, width * height);
        let mut tiles = Vec::with_capacity(tile_count);
        for _ in 0..tile_count {
            tiles.push(Tile {
                meta: TileMeta {
                    tile_type: TileType::Wire,
                },
            });
        }
        let vals = vec![Cell::new(0u64); tile_count + 1];
        Self {
            tiles,
            vals,
            lane_stride: 1,
            lane_offset: 0,
            width,
            height,
            num_layers,
            layer_size,
        }
    }

    /// Total number of tiles across all layers
    #[inline]
    pub fn tile_count(&self) -> usize {
        self.layer_size * self.num_layers
    }

    /// Logic value of the tile at flat index `idx` (the active lane's slot in
    /// lane mode). Panics on out-of-bounds like the pre-SoA `tiles[idx]` did.
    /// The stride-1 fast path keeps normal-mode codegen identical to a plain
    /// dense array (the branch predicts perfectly; lane mode is the rare arm).
    #[inline(always)]
    pub fn value(&self, idx: usize) -> u64 {
        if self.lane_stride == 1 {
            self.vals[idx].get()
        } else {
            self.vals[idx * self.lane_stride + self.lane_offset].get()
        }
    }

    /// Store the logic value of the tile at flat index `idx` (active lane's
    /// slot in lane mode).
    #[inline(always)]
    pub fn set_value(&self, idx: usize, v: u64) {
        if self.lane_stride == 1 {
            self.vals[idx].set(v);
        } else {
            self.vals[idx * self.lane_stride + self.lane_offset].set(v);
        }
    }

    /// Unchecked variants for perf-gated eval paths (callers guarantee
    /// `idx < tile_count`).
    ///
    /// # Safety
    /// `idx` must be in bounds for the addressed lane slot.
    #[inline(always)]
    pub(crate) unsafe fn value_unchecked(&self, idx: usize) -> u64 {
        let slot = if self.lane_stride == 1 {
            idx
        } else {
            idx * self.lane_stride + self.lane_offset
        };
        unsafe { self.vals.get_unchecked(slot).get() }
    }

    /// # Safety
    /// `idx` must be in bounds for the addressed lane slot.
    #[inline(always)]
    pub(crate) unsafe fn set_value_unchecked(&self, idx: usize, v: u64) {
        let slot = if self.lane_stride == 1 {
            idx
        } else {
            idx * self.lane_stride + self.lane_offset
        };
        unsafe { self.vals.get_unchecked(slot).set(v) };
    }

    /// Logic value at (x, y) on layer 0; None when out of bounds.
    #[inline]
    pub fn value_at(&self, x: usize, y: usize) -> Option<u64> {
        self.index(x, y).map(|i| self.value(i))
    }

    /// Logic value at (x, y, z); None when out of bounds.
    #[inline]
    pub fn value_at_3d(&self, x: usize, y: usize, z: usize) -> Option<u64> {
        self.index_3d(x, y, z).map(|i| self.value(i))
    }

    /// Store the logic value at (x, y) on layer 0. Returns false when out of
    /// bounds.
    #[inline]
    pub fn set_value_at(&self, x: usize, y: usize, v: u64) -> bool {
        match self.index(x, y) {
            Some(i) => {
                self.set_value(i, v);
                true
            }
            None => false,
        }
    }

    /// Snapshot all tile values for the active lane (batch IO/diagnostics).
    pub fn values_snapshot(&self) -> Vec<u64> {
        (0..self.tile_count()).map(|i| self.value(i)).collect()
    }

    /// Raw value-array pointer for JIT-compiled eval (walks dense u64 at
    /// stride 8). Only valid in normal (stride-1) mode — the fabric disables
    /// JIT in lane mode.
    pub fn jit_values_ptr(&mut self) -> *mut u8 {
        assert_eq!(
            self.lane_stride, 1,
            "JIT eval is incompatible with lane mode"
        );
        self.vals.as_mut_ptr() as *mut u8
    }

    // --- Sprint 386: lane mode (SIMT fabric) ---

    /// Enter lane mode: re-layout to `lanes`-interleaved storage, replicating
    /// the current values into every lane. Scalar accessors then address the
    /// active lane (default 0).
    pub fn enter_lane_mode(&mut self, lanes: usize) {
        assert!(lanes >= 1, "lane mode needs at least one lane");
        assert_eq!(self.lane_stride, 1, "already in lane mode");
        let tile_count = self.tile_count();
        let vals = vec![Cell::new(0u64); (tile_count + 1) * lanes];
        for idx in 0..tile_count {
            let v = self.vals[idx].get();
            for lane in 0..lanes {
                vals[idx * lanes + lane].set(v);
            }
        }
        self.vals = vals;
        self.lane_stride = lanes;
        self.lane_offset = 0;
    }

    /// Leave lane mode, keeping the ACTIVE lane's values.
    pub fn exit_lane_mode(&mut self) {
        assert!(self.lane_stride > 1, "not in lane mode");
        let tile_count = self.tile_count();
        let vals = vec![Cell::new(0u64); tile_count + 1];
        for (idx, v) in vals.iter().enumerate().take(tile_count) {
            v.set(self.vals[idx * self.lane_stride + self.lane_offset].get());
        }
        self.vals = vals;
        self.lane_stride = 1;
        self.lane_offset = 0;
    }

    /// Select which lane the scalar accessors address.
    #[inline]
    pub fn set_active_lane(&mut self, lane: usize) {
        assert!(
            lane < self.lane_stride,
            "lane {lane} out of range ({})",
            self.lane_stride
        );
        self.lane_offset = lane;
    }

    #[inline]
    pub fn active_lane(&self) -> usize {
        self.lane_offset
    }

    #[inline]
    pub fn lane_mode_active(&self) -> bool {
        self.lane_stride > 1
    }

    #[inline]
    pub fn lane_count(&self) -> usize {
        self.lane_stride
    }

    /// Raw interleaved buffer for the wide lane kernels: `(vals, lanes)` with
    /// `vals.len() == (tile_count + 1) * lanes`. `Cell<u64>` is
    /// `repr(transparent)` over `u64` and we hold `&mut self`, so the
    /// reinterpretation is sound.
    pub fn lane_vals_raw_mut(&mut self) -> (&mut [u64], usize) {
        let lanes = self.lane_stride;
        let slice = self.vals.as_mut_slice();
        let raw = unsafe { &mut *(slice as *mut [Cell<u64>] as *mut [u64]) };
        (raw, lanes)
    }

    /// Copy every tile value from `src` (same dimensions, both in normal
    /// mode) — used by `Simulation::clone`.
    pub(crate) fn copy_values_from(&mut self, src: &Tilemap) {
        assert_eq!(self.lane_stride, src.lane_stride);
        for (dst, s) in self.vals.iter().zip(src.vals.iter()) {
            dst.set(s.get());
        }
    }

    /// 2D index (layer 0 only, backward compatible)
    #[inline]
    pub fn index(&self, x: usize, y: usize) -> Option<usize> {
        if x < self.width && y < self.height {
            Some(y * self.width + x)
        } else {
            None
        }
    }

    /// 3D index: returns flat index for tile at (x, y, z)
    #[inline]
    pub fn index_3d(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        if x < self.width && y < self.height && z < self.num_layers {
            Some(z * self.layer_size + y * self.width + x)
        } else {
            None
        }
    }

    /// Convert flat index back to (x, y, z) coordinates
    #[inline]
    pub fn coords_from_idx(&self, idx: usize) -> (usize, usize, usize) {
        let z = idx / self.layer_size;
        let within = idx % self.layer_size;
        let y = within / self.width;
        let x = within % self.width;
        (x, y, z)
    }

    /// Get tile at (x, y) on layer 0 (backward compatible)
    pub fn get_tile(&self, x: usize, y: usize) -> Option<&Tile> {
        self.index(x, y).and_then(|i| self.tiles.get(i))
    }

    /// Get mutable tile at (x, y) on layer 0 (backward compatible)
    pub fn get_tile_mut(&mut self, x: usize, y: usize) -> Option<&mut Tile> {
        let i = self.index(x, y)?;
        self.tiles.get_mut(i)
    }

    /// Get tile at (x, y, z) on specified layer
    pub fn get_tile_3d(&self, x: usize, y: usize, z: usize) -> Option<&Tile> {
        self.index_3d(x, y, z).and_then(|i| self.tiles.get(i))
    }

    /// Get mutable tile at (x, y, z) on specified layer
    pub fn get_tile_3d_mut(&mut self, x: usize, y: usize, z: usize) -> Option<&mut Tile> {
        let i = self.index_3d(x, y, z)?;
        self.tiles.get_mut(i)
    }
}
