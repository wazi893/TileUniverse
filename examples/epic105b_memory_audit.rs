//! EPIC 105b: Memory Audit
//!
//! Calculates exact memory usage per tile to identify optimization targets.
//!
//! Run with: cargo run --release --example epic105b_memory_audit

use std::mem::size_of;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  EPIC 105b: Memory Audit - Where Does It All Go?              ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Calculate sizes of key types
    println!("=== Type Sizes ===\n");

    println!("Primitive types:");
    println!("  u8:  {} bytes", size_of::<u8>());
    println!("  u32: {} bytes", size_of::<u32>());
    println!("  u64: {} bytes", size_of::<u64>());
    println!("  usize: {} bytes", size_of::<usize>());
    println!("  Option<usize>: {} bytes", size_of::<Option<usize>>());
    println!("  [u32; 4]: {} bytes", size_of::<[u32; 4]>());

    println!("\nAtomic types:");
    println!(
        "  AtomicU64: {} bytes",
        size_of::<std::sync::atomic::AtomicU64>()
    );

    // Tile and TileMeta
    println!("\nTile types:");
    println!(
        "  TileType (enum): {} bytes",
        size_of::<engine::tile_meta::TileType>()
    );
    println!(
        "  TileMeta: {} bytes",
        size_of::<engine::tile_meta::TileMeta>()
    );
    println!("  Tile: {} bytes", size_of::<engine::tile::Tile>());

    // ChangeInfo
    println!("\nChangeInfo:");
    println!(
        "  ChangeInfo: {} bytes",
        size_of::<engine::simulation::ChangeInfo>()
    );
    println!(
        "  Option<ChangeInfo>: {} bytes",
        size_of::<Option<engine::simulation::ChangeInfo>>()
    );

    println!("\n=== Memory Per Tile (for N tiles) ===\n");

    // Per-tile allocations in Simulation struct:
    let tile_size = size_of::<engine::tile::Tile>();
    let neighbors_size = size_of::<[u32; 4]>();
    let meta_fast_size = size_of::<engine::tile_meta::TileType>();
    let change_info_size = size_of::<Option<engine::simulation::ChangeInfo>>();
    let qtile_lookup_size = size_of::<Option<usize>>();

    // FieldGrid allocations (per cell):
    // From the struct: logic(u32), power(u8), clock(u8), region(u32),
    //                  logic_next(u32), power_next(u8), clock_next(u8),
    //                  logic_coupled(u64), power_coupled(u32),
    //                  heat(u32), charge(u32), heat_next(u32), charge_next(u32),
    //                  heat_react(u32), charge_react(u32), heat_interact(u32), charge_interact(u32)
    let field_bytes: usize = 4 +  // logic_field: u32
        1 +  // power_field: u8
        1 +  // clock_field: u8
        4 +  // region_field: u32
        4 +  // logic_field_next: u32
        1 +  // power_field_next: u8
        1 +  // clock_field_next: u8
        8 +  // logic_field_coupled: u64
        4 +  // power_field_coupled: u32
        4 +  // heat_field: u32
        4 +  // charge_field: u32
        4 +  // heat_field_next: u32
        4 +  // charge_field_next: u32
        4 +  // heat_field_react: u32
        4 +  // charge_field_react: u32
        4 +  // heat_field_interact: u32
        4; // charge_field_interact: u32

    // Dirty bitset: 1 bit per tile = 0.125 bytes
    let dirty_bits = 0.125_f64;

    println!("Component breakdown (bytes per tile):");
    println!("  tilemap.tiles (Vec<Tile>):     {:>3} bytes", tile_size);
    println!(
        "  neighbors4 (Vec<[u32;4]>):     {:>3} bytes",
        neighbors_size
    );
    println!(
        "  meta_fast (Vec<TileType>):     {:>3} bytes",
        meta_fast_size
    );
    println!(
        "  last_change (Vec<Option<CI>>): {:>3} bytes  <-- BIG!",
        change_info_size
    );
    println!(
        "  qtile_lookup (Vec<Opt<usize>>):{:>3} bytes",
        qtile_lookup_size
    );
    println!(
        "  17x FieldGrids:                {:>3} bytes  (total)",
        field_bytes
    );
    println!("  dirty bitset:                  {:.2} bytes", dirty_bits);

    let total_per_tile = tile_size
        + neighbors_size
        + meta_fast_size
        + change_info_size
        + qtile_lookup_size
        + field_bytes;
    let total_with_dirty = total_per_tile as f64 + dirty_bits;

    println!(
        "\n  TOTAL per tile:               ~{:.1} bytes",
        total_with_dirty
    );

    println!("\n=== Projected Memory Usage ===\n");

    let configs = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
        (32768, 32768),
    ];

    println!(
        "{:>12} {:>14} {:>12} {:>12}",
        "Grid", "Tiles", "Memory (MB)", "Memory (GB)"
    );
    println!("{:->12} {:->14} {:->12} {:->12}", "", "", "", "");

    for (w, h) in configs {
        let tiles = w * h;
        let bytes = (tiles as f64) * total_with_dirty;
        let mb = bytes / (1024.0 * 1024.0);
        let gb = mb / 1024.0;
        println!("{:>5}x{:<6} {:>14} {:>12.1} {:>12.2}", w, h, tiles, mb, gb);
    }

    println!("\n=== Optimization Targets ===\n");

    println!("Biggest memory hogs (per tile):");
    let mut items = vec![
        ("last_change (Option<ChangeInfo>)", change_info_size),
        ("tilemap.tiles (Tile)", tile_size),
        ("neighbors4 ([u32;4])", neighbors_size),
        ("17x FieldGrids", field_bytes),
        ("qtile_lookup", qtile_lookup_size),
        ("meta_fast", meta_fast_size),
    ];
    items.sort_by(|a, b| b.1.cmp(&a.1));

    for (name, size) in &items {
        let pct = (*size as f64 / total_with_dirty) * 100.0;
        println!("  {:>3} bytes ({:>5.1}%) - {}", size, pct, name);
    }

    println!("\n=== Optimization Ideas ===\n");
    println!(
        "1. last_change: {} bytes/tile - Make lazy/optional for benchmarks",
        change_info_size
    );
    println!(
        "2. FieldGrids: {} bytes/tile - Many unused in pure logic sim",
        field_bytes
    );
    println!(
        "3. qtile_lookup: {} bytes/tile - Only needed if using quantum tiles",
        qtile_lookup_size
    );
    println!("4. neighbors4: Can be computed on-the-fly (trade CPU for RAM)");

    // Calculate potential savings
    let minimal_bytes = tile_size + meta_fast_size + dirty_bits as usize;
    println!(
        "\nMinimal config (tiles + meta + dirty): {} bytes/tile",
        minimal_bytes
    );
    println!("Current config: {:.0} bytes/tile", total_with_dirty);
    println!(
        "Potential savings: {:.1}x reduction possible!",
        total_with_dirty / minimal_bytes as f64
    );
}
