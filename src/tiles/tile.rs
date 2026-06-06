use std::sync::atomic::AtomicU64;

use crate::tile_meta::TileMeta;

pub struct Tile {
    pub logic: AtomicU64, // 64 lanes
    pub meta: TileMeta,
}
