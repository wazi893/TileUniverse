//! Deterministic placement legalization for mapped synth netlists.
//!
//! Workstream B starts with a deliberately simple placer: place mapped gates
//! in stable topological order on a row-major grid with a reserved routing halo
//! around every gate. Routing and export come later; this module establishes the
//! intermediate placed-netlist artifact that those later phases will consume.

use super::CellLibrary;
use super::mapping::MappedNetlist;
use crate::tiles::tile_meta::TileType;

/// Bounding box reserved for synthesized placement (single or multi-layer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementRegion {
    pub min_x: usize,
    pub min_y: usize,
    pub width: usize,
    pub height: usize,
    pub z: usize,
    /// Highest routing layer (defaults to `z` for single-layer).
    pub max_z: usize,
}

impl PlacementRegion {
    /// Inclusive maximum x coordinate covered by the region.
    pub fn max_x(&self) -> usize {
        self.min_x + self.width.saturating_sub(1)
    }

    /// Inclusive maximum y coordinate covered by the region.
    pub fn max_y(&self) -> usize {
        self.min_y + self.height.saturating_sub(1)
    }

    /// Returns true if the coordinate lies inside the region.
    pub fn contains(&self, x: usize, y: usize, z: usize) -> bool {
        z >= self.z
            && z <= self.max_z
            && x >= self.min_x
            && x <= self.max_x()
            && y >= self.min_y
            && y <= self.max_y()
    }
}

/// Preferred side of a tile used by the future router.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinSide {
    Left,
    Right,
    Up,
    Down,
}

/// Legalized placement for one mapped gate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatePlacement {
    pub gate_idx: usize,
    pub tile_type: TileType,
    pub x: usize,
    pub y: usize,
    pub z: usize,
    /// Ordered to match the mapped gate's fanin order.
    pub input_pins: Vec<PinSide>,
    /// Preferred egress side for routing the gate output.
    pub output_pin: PinSide,
}

/// Reserved routed tile segment.
///
/// Phase 1 placement leaves this empty. The type lives here so routing can add
/// to the same artifact in Workstream B Phase 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutedSegment {
    pub net_id: u32,
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub tile_type: TileType,
}

/// Intermediate artifact between mapping and tile export.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacedNetlist {
    pub region: PlacementRegion,
    pub halo: usize,
    pub gates: Vec<GatePlacement>,
    pub routes: Vec<RoutedSegment>,
    /// Boundary anchor coordinates for primary inputs (set by router).
    pub input_anchors: Vec<(usize, usize)>,
    /// Boundary anchor coordinates for primary outputs (set by router).
    pub output_anchors: Vec<(usize, usize)>,
}

impl PlacedNetlist {
    /// Returns the placement for a gate index, if present.
    pub fn gate(&self, gate_idx: usize) -> Option<&GatePlacement> {
        self.gates.iter().find(|g| g.gate_idx == gate_idx)
    }
}

/// Placement configuration for the deterministic row placer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaceConfig {
    pub min_x: usize,
    pub min_y: usize,
    pub z: usize,
    /// Optional cap on the number of gate columns before wrapping to the next row.
    pub max_columns: Option<usize>,
    /// Empty tiles reserved on each side of a gate for future routing.
    pub halo: usize,
    /// Sprint 209: Optional maximum width for the final exported block (in tiles).
    /// Accounts for routing expansion (halo margin on each side of placement region).
    /// When set, the placer computes the maximum number of columns that fit within
    /// `max_width - 2 * routing_margin` and uses that as the column cap.
    pub max_width: Option<usize>,
}

impl Default for PlaceConfig {
    fn default() -> Self {
        Self {
            min_x: 0,
            min_y: 0,
            z: 0,
            max_columns: None,
            halo: 1,
            max_width: None,
        }
    }
}

impl PlaceConfig {
    /// Preset for standalone synthesis (Python `synthesize()`, `validate_synth_pipeline`).
    ///
    /// Uses `halo=3` and `min_x=3, min_y=3` to provide sufficient guard clearance
    /// between gates and grid edges. Matches the configuration used by all passing
    /// Rust export integration tests.
    pub fn standalone() -> Self {
        Self {
            min_x: 3,
            min_y: 3,
            halo: 3,
            ..Self::default()
        }
    }
}

/// Placement legalization failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlacementError {
    InvalidConfig(&'static str),
    SequentialTile {
        gate_idx: usize,
        tile_type: TileType,
    },
    UnsupportedTile {
        gate_idx: usize,
        tile_type: TileType,
    },
    InvalidPinArity {
        gate_idx: usize,
        tile_type: TileType,
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for PlacementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlacementError::InvalidConfig(msg) => write!(f, "invalid placement config: {msg}"),
            PlacementError::SequentialTile {
                gate_idx,
                tile_type,
            } => {
                write!(
                    f,
                    "gate {gate_idx} uses sequential tile {:?}, which is out of scope for placement MVP",
                    tile_type
                )
            }
            PlacementError::UnsupportedTile {
                gate_idx,
                tile_type,
            } => {
                write!(
                    f,
                    "gate {gate_idx} uses unsupported tile {:?} for placement MVP",
                    tile_type
                )
            }
            PlacementError::InvalidPinArity {
                gate_idx,
                tile_type,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "gate {gate_idx} expects {expected} inputs for {:?} but netlist uses {actual}",
                    tile_type
                )
            }
        }
    }
}

impl std::error::Error for PlacementError {}

/// Place a mapped netlist in deterministic row-major order.
///
/// Gates are placed in mapped-netlist order, which is already topological.
/// Every gate sits at the center of a reserved routing cell with one extra
/// inter-cell aisle to make Phase 2 Manhattan routing feasible on one layer.
pub fn place_mapped_netlist(
    netlist: &MappedNetlist,
    lib: &CellLibrary,
    config: &PlaceConfig,
) -> Result<PlacedNetlist, PlacementError> {
    let mut columns = match config.max_columns {
        Some(0) => return Err(PlacementError::InvalidConfig("max_columns must be >= 1")),
        Some(cols) => cols,
        None => auto_columns(netlist.gates.len()),
    };
    let pitch = halo_pitch(config.halo)?;

    // Sprint 209: constrain columns so final exported width fits within max_width.
    // Export width = min_x + placement_width + routing_margin + 1, where:
    //   routing_margin = halo.max(1)  (from expanded_route_region)
    //   +1 comes from export.rs: region.max_x() + 2 = min_x + width + margin - 1 + 2
    // So overhead = min_x + routing_margin + 1, and placement_width = columns * pitch.
    if let Some(max_w) = config.max_width {
        let routing_margin = config.halo.max(1);
        let overhead = config.min_x + routing_margin + 1;
        if max_w <= overhead + pitch {
            return Err(PlacementError::InvalidConfig(
                "max_width too small for even 1 column after routing/export margin",
            ));
        }
        let max_cols_from_width = (max_w - overhead) / pitch;
        if max_cols_from_width == 0 {
            return Err(PlacementError::InvalidConfig(
                "max_width too small for even 1 column",
            ));
        }
        columns = columns.min(max_cols_from_width);
    }
    let rows = if netlist.gates.is_empty() {
        1
    } else {
        netlist.gates.len().div_ceil(columns)
    };

    let region = PlacementRegion {
        min_x: config.min_x,
        min_y: config.min_y,
        width: columns * pitch,
        height: rows * pitch,
        z: config.z,
        max_z: config.z,
    };

    let mut gates = Vec::with_capacity(netlist.gates.len());
    for (gate_idx, gate) in netlist.gates.iter().enumerate() {
        let cell = lib.cell(gate.cell_idx);
        let (input_pins, output_pin) = pin_layout_for(gate_idx, cell.tile_type, gate.fanins.len())?;
        let row = gate_idx / columns;
        let col = gate_idx % columns;
        let x = config.min_x + config.halo + col * pitch;
        let y = config.min_y + config.halo + row * pitch;

        gates.push(GatePlacement {
            gate_idx,
            tile_type: cell.tile_type,
            x,
            y,
            z: config.z,
            input_pins,
            output_pin,
        });
    }

    Ok(PlacedNetlist {
        region,
        halo: config.halo,
        gates,
        routes: Vec::new(),
        input_anchors: Vec::new(),
        output_anchors: Vec::new(),
    })
}

fn halo_pitch(halo: usize) -> Result<usize, PlacementError> {
    halo.checked_mul(2)
        .and_then(|x| x.checked_add(2))
        .ok_or(PlacementError::InvalidConfig("halo is too large"))
}

fn auto_columns(num_gates: usize) -> usize {
    let target = num_gates.max(1);
    let mut cols = 1usize;
    while cols.saturating_mul(cols) < target {
        cols += 1;
    }
    cols
}

fn pin_layout_for(
    gate_idx: usize,
    tile_type: TileType,
    actual_inputs: usize,
) -> Result<(Vec<PinSide>, PinSide), PlacementError> {
    if tile_type.is_sequential() {
        return Err(PlacementError::SequentialTile {
            gate_idx,
            tile_type,
        });
    }

    let (expected, inputs, output) = match tile_type {
        TileType::Const => (0, Vec::new(), PinSide::Right),
        TileType::WireRight => (1, vec![PinSide::Left], PinSide::Right),
        TileType::Not => (1, vec![PinSide::Left], PinSide::Right),
        TileType::And | TileType::Or | TileType::Xor => {
            (2, vec![PinSide::Left, PinSide::Right], PinSide::Down)
        }
        TileType::Mux => (
            3,
            vec![PinSide::Up, PinSide::Left, PinSide::Right],
            PinSide::Down,
        ),
        _ => {
            return Err(PlacementError::UnsupportedTile {
                gate_idx,
                tile_type,
            });
        }
    };

    if actual_inputs != expected {
        return Err(PlacementError::InvalidPinArity {
            gate_idx,
            tile_type,
            expected,
            actual: actual_inputs,
        });
    }

    Ok((inputs, output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::benchmark::{benchmark_specs, build_4bit_adder};
    use crate::synth::mapping::map_to_library;
    use crate::synth::{Aig, CellLibrary, MapConfig};
    use std::collections::HashSet;

    fn map_benchmark(build: fn() -> Aig) -> (MappedNetlist, CellLibrary) {
        let aig = build();
        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        (netlist, lib)
    }

    #[test]
    fn row_placer_handles_benchmark_suite_without_overlap() {
        for spec in benchmark_specs() {
            let (netlist, lib) = map_benchmark(spec.build);
            let placed = place_mapped_netlist(&netlist, &lib, &PlaceConfig::default())
                .expect("benchmark netlist should place cleanly");

            assert_eq!(placed.gates.len(), netlist.gates.len(), "{}", spec.name);

            let mut seen = HashSet::new();
            for gate in &placed.gates {
                assert!(
                    placed.region.contains(gate.x, gate.y, gate.z),
                    "gate {:?} escaped region for {}",
                    gate,
                    spec.name
                );
                assert!(
                    seen.insert((gate.x, gate.y, gate.z)),
                    "duplicate gate placement at ({}, {}, {}) for {}",
                    gate.x,
                    gate.y,
                    gate.z,
                    spec.name
                );
            }

            for (i, lhs) in placed.gates.iter().enumerate() {
                for rhs in &placed.gates[i + 1..] {
                    let dx = lhs.x.abs_diff(rhs.x);
                    let dy = lhs.y.abs_diff(rhs.y);
                    assert!(
                        dx > placed.halo * 2 || dy > placed.halo * 2,
                        "routing halos overlap between gate {} and gate {} for {}",
                        lhs.gate_idx,
                        rhs.gate_idx,
                        spec.name
                    );
                }
            }
        }
    }

    #[test]
    fn row_placer_is_deterministic() {
        let (netlist, lib) = map_benchmark(build_4bit_adder);
        let config = PlaceConfig {
            max_columns: Some(4),
            ..PlaceConfig::default()
        };

        let first = place_mapped_netlist(&netlist, &lib, &config).unwrap();
        let second = place_mapped_netlist(&netlist, &lib, &config).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn max_width_constrains_placement_columns() {
        let (netlist, lib) = map_benchmark(build_4bit_adder);
        let halo = 4usize;
        let pitch = halo * 2 + 2; // 10

        // Without constraint: auto_columns picks sqrt(N) columns.
        let unconstrained = place_mapped_netlist(
            &netlist,
            &lib,
            &PlaceConfig {
                halo,
                ..PlaceConfig::default()
            },
        )
        .unwrap();
        let _unconstrained_cols = unconstrained.region.width / pitch;

        // With max_width=60, min_x=0: overhead = 0 + 4 + 1 = 5.
        // Available for placement = 60 - 5 = 55 → 55/10 = 5 columns max.
        let constrained = place_mapped_netlist(
            &netlist,
            &lib,
            &PlaceConfig {
                halo,
                max_width: Some(60),
                ..PlaceConfig::default()
            },
        )
        .unwrap();
        let constrained_cols = constrained.region.width / pitch;

        assert!(
            constrained_cols <= 5,
            "expected <= 5 columns with max_width=60, got {constrained_cols}"
        );
        // Constrained should be narrower (or equal if already small).
        assert!(constrained.region.width <= unconstrained.region.width);
        // Constrained should be taller (more rows needed for same gates).
        assert!(constrained.region.height >= unconstrained.region.height);
    }

    #[test]
    fn max_width_too_small_returns_error() {
        let (netlist, lib) = map_benchmark(build_4bit_adder);
        let result = place_mapped_netlist(
            &netlist,
            &lib,
            &PlaceConfig {
                halo: 4,
                max_width: Some(5), // way too small
                ..PlaceConfig::default()
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn max_width_accounts_for_min_x() {
        let (netlist, lib) = map_benchmark(build_4bit_adder);
        let halo = 4usize;
        let pitch = halo * 2 + 2; // 10

        // min_x=20, max_width=60: overhead = 20 + 4 + 1 = 25.
        // Available = 60 - 25 = 35 → 35/10 = 3 columns max.
        let placed = place_mapped_netlist(
            &netlist,
            &lib,
            &PlaceConfig {
                min_x: 20,
                halo,
                max_width: Some(60),
                ..PlaceConfig::default()
            },
        )
        .unwrap();
        let cols = placed.region.width / pitch;
        assert!(
            cols <= 3,
            "expected <= 3 columns with min_x=20 + max_width=60, got {cols}"
        );

        // Same max_width at min_x=0 should allow more columns.
        let placed_zero = place_mapped_netlist(
            &netlist,
            &lib,
            &PlaceConfig {
                halo,
                max_width: Some(60),
                ..PlaceConfig::default()
            },
        )
        .unwrap();
        assert!(
            placed_zero.region.width >= placed.region.width,
            "min_x=0 should allow at least as many columns as min_x=20"
        );
    }

    #[test]
    fn mux_pin_order_is_preserved_for_routing() {
        let mut aig = Aig::new();
        let sel = aig.add_input("sel");
        let when_true = aig.add_input("t");
        let when_false = aig.add_input("f");
        let y = aig.mux(sel, when_true, when_false);
        aig.add_output("y", y);

        let lib = CellLibrary::tile_native();
        let netlist = map_to_library(&aig, &lib, &MapConfig::default());
        assert_eq!(netlist.gates.len(), 1, "expected direct MUX2 mapping");

        let placed = place_mapped_netlist(&netlist, &lib, &PlaceConfig::default()).unwrap();
        let mux = &placed.gates[0];

        assert_eq!(mux.tile_type, TileType::Mux);
        assert_eq!(
            mux.input_pins,
            vec![PinSide::Up, PinSide::Left, PinSide::Right]
        );
        assert_eq!(mux.output_pin, PinSide::Down);
    }
}
