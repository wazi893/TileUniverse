use crate::tile_meta::TileMeta;

/// Topology-only tile. The per-tile logic VALUE lives in `Tilemap::values`
/// (SoA, same index), relocated in Sprint 385 so lane-packed evaluation can
/// address all values contiguously.
pub struct Tile {
    pub meta: TileMeta,
}
