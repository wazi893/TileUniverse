use crate::tile::Tile;
use crate::tile_meta::{TileMeta, TileType};
use std::sync::atomic::AtomicU64;

// Default dimensions (kept for backward compatibility)
pub const WIDTH: usize = 512;
pub const HEIGHT: usize = 512;
pub const TILE_COUNT: usize = WIDTH * HEIGHT;

pub struct Tilemap {
    pub tiles: Vec<Tile>,
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
                logic: AtomicU64::new(0),
                meta: TileMeta {
                    tile_type: TileType::Wire,
                },
            });
        }
        Self {
            tiles,
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

    /// 2D index (layer 0 only, backward compatible)
    #[inline]
    fn index(&self, x: usize, y: usize) -> Option<usize> {
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
