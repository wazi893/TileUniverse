//! Multi-Layer Tile Support Tests (Sprint 80)
//!
//! Verifies the multi-layer tilemap, via tiles, cross-layer dirty propagation,
//! blueprint v2 format, and backward compatibility.
//!
//! Run with: cargo test --test multi_layer_tests -- --nocapture

use engine::blueprint::{Blueprint, BlueprintTile};
use engine::simulation::Simulation;
use engine::slim_simulation::SlimSimulation;
use engine::tile_meta::TileType;
use std::io::BufReader;
use std::sync::atomic::Ordering;

// ============================================================================
// Test 1: Single-layer identity
// ============================================================================

#[test]
fn single_layer_identity() {
    let sim1 = Simulation::with_size(32, 32);
    let sim2 = Simulation::with_size_layered(32, 32, 1);

    assert_eq!(sim1.tilemap.tile_count(), sim2.tilemap.tile_count());
    assert_eq!(sim1.tilemap.width, sim2.tilemap.width);
    assert_eq!(sim1.tilemap.height, sim2.tilemap.height);
    assert_eq!(sim2.tilemap.num_layers, 1);
    assert_eq!(sim2.tilemap.layer_size, 32 * 32);

    // Place the same tile in both sims
    let mut sim1 = sim1;
    let mut sim2 = sim2;
    sim1.set_tile(5, 5, TileType::Const);
    sim1.set_logic_value(5, 5, 42);
    sim2.set_tile(5, 5, TileType::Const);
    sim2.set_logic_value(5, 5, 42);

    assert_eq!(
        sim1.tilemap.value(5 * 32 + 5),
        sim2.tilemap.value(5 * 32 + 5),
    );
}

// ============================================================================
// Test 2: Layer isolation
// ============================================================================

#[test]
fn layer_isolation() {
    let mut sim = Simulation::with_size_layered(8, 8, 2);

    // Place Const with value 0xFF on layer 0 at (3,3)
    sim.set_tile_3d(3, 3, 0, TileType::Const);
    sim.set_logic_value_3d(3, 3, 0, 0xFF);

    // Layer 1 at (3,3) is a Wire (default)
    assert_eq!(sim.tile_type_3d(3, 3, 1), TileType::Wire);

    // Settle
    sim.dirty.mark_all_dirty(sim.tilemap.tile_count());
    for _ in 0..20 {
        sim.propagate_combinational();
    }

    // Layer 0 should have the value
    assert_eq!(sim.get_logic_at_3d(3, 3, 0), 0xFF);

    // Layer 1 at (3,3) — Wire on layer 1 only sees its own layer-1 neighbors
    let layer1_val = sim.get_logic_at_3d(3, 3, 1);
    assert_eq!(
        layer1_val, 0,
        "Layer 1 should not see layer 0's Const value"
    );
}

// ============================================================================
// Test 3: ViaUp signal transfer
// ============================================================================

#[test]
fn via_up_reads_from_layer_above() {
    let mut sim = Simulation::with_size_layered(8, 8, 2);

    // Place a Const with value 0xBEEF on layer 1 at (4,4)
    sim.set_tile_3d(4, 4, 1, TileType::Const);
    sim.set_logic_value_3d(4, 4, 1, 0xBEEF);

    // Place ViaUp on layer 0 at (4,4) — it reads from layer 1
    sim.set_tile_3d(4, 4, 0, TileType::ViaUp);

    // Settle
    sim.dirty.mark_all_dirty(sim.tilemap.tile_count());
    for _ in 0..20 {
        sim.propagate_combinational();
    }

    let via_val = sim.get_logic_at_3d(4, 4, 0);
    assert_eq!(via_val, 0xBEEF, "ViaUp should read from layer above");
}

// ============================================================================
// Test 4: ViaDown signal transfer
// ============================================================================

#[test]
fn via_down_reads_from_layer_below() {
    let mut sim = Simulation::with_size_layered(8, 8, 2);

    // Place a Const with value 0xCAFE on layer 0 at (2,2)
    sim.set_tile_3d(2, 2, 0, TileType::Const);
    sim.set_logic_value_3d(2, 2, 0, 0xCAFE);

    // Place ViaDown on layer 1 at (2,2) — it reads from layer 0
    sim.set_tile_3d(2, 2, 1, TileType::ViaDown);

    // Settle
    sim.dirty.mark_all_dirty(sim.tilemap.tile_count());
    for _ in 0..20 {
        sim.propagate_combinational();
    }

    let via_val = sim.get_logic_at_3d(2, 2, 1);
    assert_eq!(via_val, 0xCAFE, "ViaDown should read from layer below");
}

// ============================================================================
// Test 5: Cross-layer dirty propagation
// ============================================================================

#[test]
fn cross_layer_dirty_propagation() {
    let mut sim = Simulation::with_size_layered(8, 8, 2);

    // Const on layer 1 -> ViaUp on layer 0 -> WireDown on layer 0
    sim.set_tile_3d(3, 1, 1, TileType::Const);
    sim.set_logic_value_3d(3, 1, 1, 0x42);

    sim.set_tile_3d(3, 1, 0, TileType::ViaUp); // reads from layer 1
    sim.set_tile_3d(3, 2, 0, TileType::WireDown); // reads from above (ViaUp)

    // Guard neighbors to prevent Wire leakage
    sim.set_tile_3d(2, 1, 0, TileType::Const);
    sim.set_tile_3d(4, 1, 0, TileType::Const);
    sim.set_tile_3d(2, 2, 0, TileType::Const);
    sim.set_tile_3d(4, 2, 0, TileType::Const);
    sim.set_tile_3d(3, 3, 0, TileType::Const);
    sim.set_tile_3d(3, 0, 0, TileType::Const);

    // Rebuild via connections and settle
    sim.rebuild_via_connections();
    sim.dirty.mark_all_dirty(sim.tilemap.tile_count());
    for _ in 0..30 {
        sim.propagate_combinational();
    }

    let wd_val = sim.get_logic_at_3d(3, 2, 0);
    assert_eq!(
        wd_val, 0x42,
        "WireDown should receive value from ViaUp which reads from layer 1"
    );

    // Now change the Const on layer 1 and verify dirty propagation.
    // set_logic_value_3d updates the atomic but doesn't evaluate — same pattern
    // as software-written Const: must dirty the consumer (ViaUp) explicitly.
    sim.set_logic_value_3d(3, 1, 1, 0x99);
    let via_up_idx = 8 + 3; // ViaUp on layer 0 at (x=3, y=1), gw=8
    sim.dirty.mark_dirty(via_up_idx);

    for _ in 0..30 {
        sim.propagate_combinational();
    }

    let wd_val2 = sim.get_logic_at_3d(3, 2, 0);
    assert_eq!(
        wd_val2, 0x99,
        "Dirty propagation should flow through Via to WireDown"
    );
}

// ============================================================================
// Test 6: Boundary safety
// ============================================================================

#[test]
fn via_boundary_safety() {
    let mut sim = Simulation::with_size_layered(4, 4, 2);

    // ViaUp on top layer should output 0 (no layer above)
    sim.set_tile_3d(1, 1, 1, TileType::ViaUp);

    // ViaDown on layer 0 should output 0 (no layer below)
    sim.set_tile_3d(2, 2, 0, TileType::ViaDown);

    sim.dirty.mark_all_dirty(sim.tilemap.tile_count());
    for _ in 0..10 {
        sim.propagate_combinational();
    }

    let up_val = sim.get_logic_at_3d(1, 1, 1);
    assert_eq!(up_val, 0, "ViaUp on top layer should output 0");

    let down_val = sim.get_logic_at_3d(2, 2, 0);
    assert_eq!(down_val, 0, "ViaDown on layer 0 should output 0");
}

// ============================================================================
// Test 7: Two-layer routing
// ============================================================================

#[test]
fn two_layer_routing() {
    // Signal goes up to layer 1, across, and back down to layer 0.
    // ViaDown on layer 1 reads from layer 0 (brings signal up).
    // ViaUp on layer 0 reads from layer 1 (brings signal down).
    let mut sim = Simulation::with_size_layered(8, 8, 2);

    // Source signal on layer 0
    sim.set_tile_3d(1, 1, 0, TileType::Const);
    sim.set_logic_value_3d(1, 1, 0, 0xAA);

    // WireRight on layer 0 to carry signal to via point
    sim.set_tile_3d(2, 1, 0, TileType::WireRight);

    // ViaDown on layer 1 reads from layer 0 at (2,1)
    sim.set_tile_3d(2, 1, 1, TileType::ViaDown);

    // Route across on layer 1
    sim.set_tile_3d(3, 1, 1, TileType::WireRight);
    sim.set_tile_3d(4, 1, 1, TileType::WireRight);
    sim.set_tile_3d(5, 1, 1, TileType::WireRight);

    // ViaUp on layer 0 reads from layer 1 at (5,1)
    sim.set_tile_3d(5, 1, 0, TileType::ViaUp);

    // Continue on layer 0
    sim.set_tile_3d(5, 2, 0, TileType::WireDown);

    // Guard all Wire tiles on both layers to prevent leakage
    for layer in 0..2 {
        for x in 0..8 {
            for y in 0..8 {
                let tt = sim.tile_type_3d(x, y, layer);
                if tt == TileType::Wire {
                    sim.set_tile_3d(x, y, layer, TileType::Const);
                }
            }
        }
    }

    sim.rebuild_via_connections();
    sim.dirty.mark_all_dirty(sim.tilemap.tile_count());
    for _ in 0..50 {
        sim.propagate_combinational();
    }

    // ViaUp at (5,1,0) should see the signal
    let via_up_val = sim.get_logic_at_3d(5, 1, 0);
    assert_eq!(
        via_up_val, 0xAA,
        "ViaUp should carry signal from layer 1 routing"
    );

    // WireDown at (5,2,0) should receive from ViaUp above
    let wd_val = sim.get_logic_at_3d(5, 2, 0);
    assert_eq!(
        wd_val, 0xAA,
        "Signal should route through layer 1 and back down"
    );
}

// ============================================================================
// Test 8: Blueprint v2 roundtrip
// ============================================================================

#[test]
fn blueprint_v2_roundtrip() {
    let bp = Blueprint {
        width: 8,
        height: 8,
        num_layers: 2,
        tiles: vec![
            BlueprintTile {
                x: 1,
                y: 1,
                z: 0,
                tile_type: TileType::Const,
                logic: Some(0xFF),
            },
            BlueprintTile {
                x: 2,
                y: 2,
                z: 0,
                tile_type: TileType::ViaUp,
                logic: None,
            },
            BlueprintTile {
                x: 2,
                y: 2,
                z: 1,
                tile_type: TileType::ViaDown,
                logic: None,
            },
            BlueprintTile {
                x: 3,
                y: 3,
                z: 1,
                tile_type: TileType::WireRight,
                logic: None,
            },
        ],
    };

    let mut buf = Vec::new();
    bp.save_to_writer(&mut buf).expect("save ok");

    let text = String::from_utf8_lossy(&buf);
    assert!(
        text.starts_with("LF-BLUEPRINT 2"),
        "Multi-layer should use v2 format"
    );
    assert!(
        text.contains("SIZE 8 8 2"),
        "SIZE line should include layer count"
    );

    let loaded = Blueprint::load_from_reader(BufReader::new(&buf[..])).expect("load ok");
    assert_eq!(bp, loaded);
}

// ============================================================================
// Test 9: Blueprint v1 backward compatibility
// ============================================================================

#[test]
fn blueprint_v1_backward_compat() {
    let v1_data = b"LF-BLUEPRINT 1\nSIZE 4 4\nTILE 1 1 Or 0x0000000000000001\n";
    let loaded = Blueprint::load_from_reader(BufReader::new(&v1_data[..])).expect("load ok");

    assert_eq!(
        loaded.num_layers, 1,
        "V1 blueprint should force num_layers=1"
    );
    assert_eq!(loaded.tiles.len(), 1);
    assert_eq!(loaded.tiles[0].z, 0, "V1 tiles should have z=0");
    assert_eq!(loaded.tiles[0].tile_type, TileType::Or);
    assert_eq!(loaded.tiles[0].logic, Some(1));
}

// ============================================================================
// Test 11: No cross-layer neighbor contamination
// ============================================================================

#[test]
fn no_cross_layer_neighbor_contamination() {
    let mut sim = Simulation::with_size_layered(8, 8, 2);

    // Layer 0: fill row 3 with Const tiles holding 0xAA
    for x in 0..8 {
        sim.set_tile_3d(x, 3, 0, TileType::Const);
        sim.set_logic_value_3d(x, 3, 0, 0xAA);
    }

    // Layer 1: all tiles are Const 0 (guards) except one And at (4,3,1)
    for x in 0..8 {
        for y in 0..8 {
            sim.set_tile_3d(x, y, 1, TileType::Const);
        }
    }
    sim.set_tile_3d(4, 3, 1, TileType::And);

    sim.dirty.mark_all_dirty(sim.tilemap.tile_count());
    for _ in 0..20 {
        sim.propagate_combinational();
    }

    let and_val = sim.get_logic_at_3d(4, 3, 1);
    assert_eq!(and_val, 0, "Layer 1 And should not see layer 0 values");
}

// ============================================================================
// Test 12: Delay consistency (Via has same delay behavior as Wire)
// ============================================================================

#[test]
fn via_delay_consistency() {
    assert_eq!(TileType::ViaUp.propagation_delay(), 1);
    assert_eq!(TileType::ViaDown.propagation_delay(), 1);
    assert!(TileType::ViaUp.is_wire());
    assert!(TileType::ViaDown.is_wire());
    assert!(!TileType::ViaUp.is_clock_sensitive());
    assert!(!TileType::ViaDown.is_clock_sensitive());
}

// ============================================================================
// Test 13: SlimSimulation single-layer identity
// ============================================================================

#[test]
fn slim_simulation_single_layer() {
    let sim1 = SlimSimulation::with_size(8, 8);
    let sim2 = SlimSimulation::with_size_layered(8, 8, 1);
    assert_eq!(sim1.tile_count(), sim2.tile_count());
}

// ============================================================================
// Test 14: SlimSimulation via evaluation
// ============================================================================

#[test]
fn slim_simulation_via_eval() {
    let mut sim = SlimSimulation::with_size_layered(8, 8, 2);
    let layer_size = 8 * 8;

    // Place Const on layer 1 at (3,3) using tilemap directly
    let const_idx = layer_size + 3 * 8 + 3;
    sim.tilemap.tiles[const_idx].meta.tile_type = TileType::Const;
    sim.tilemap.set_value(const_idx, 0xBEEF);
    sim.meta_fast[const_idx] = TileType::Const;

    // Place ViaUp on layer 0 at (3,3) — reads from layer 1
    let via_idx = 3 * 8 + 3;
    sim.tilemap.tiles[via_idx].meta.tile_type = TileType::ViaUp;
    sim.meta_fast[via_idx] = TileType::ViaUp;

    // Evaluate the via tile
    sim.eval_tile(via_idx);

    let via_val = sim.tilemap.value(via_idx);
    assert_eq!(
        via_val, 0xBEEF,
        "SlimSimulation ViaUp should read from layer above"
    );
}

// ============================================================================
// Test 15: Tilemap coords_from_idx roundtrip
// ============================================================================

#[test]
fn tilemap_coords_roundtrip() {
    use engine::tilemap::Tilemap;

    let tm = Tilemap::with_size_layered(16, 8, 3);
    for idx in 0..tm.tile_count() {
        let (x, y, z) = tm.coords_from_idx(idx);
        let back = tm.index_3d(x, y, z);
        assert_eq!(
            back,
            Some(idx),
            "coords_from_idx/index_3d roundtrip failed for idx={}",
            idx
        );
    }
}

// ============================================================================
// Test 16: Blueprint v1 save for single-layer
// ============================================================================

#[test]
fn blueprint_v1_save_for_single_layer() {
    let mut bp = Blueprint::new(4, 4);
    bp.tiles.push(BlueprintTile {
        x: 1,
        y: 1,
        z: 0,
        tile_type: TileType::And,
        logic: Some(0x1),
    });

    let mut buf = Vec::new();
    bp.save_to_writer(&mut buf).expect("save ok");

    let text = String::from_utf8_lossy(&buf);
    assert!(
        text.starts_with("LF-BLUEPRINT 1"),
        "Single-layer should use v1 format"
    );
    assert!(
        !text.contains("SIZE 4 4 1"),
        "V1 SIZE should not include layer count"
    );
    assert!(
        text.contains("SIZE 4 4"),
        "V1 SIZE should have width and height"
    );
}
