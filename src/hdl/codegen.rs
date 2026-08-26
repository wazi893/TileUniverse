use std::collections::{HashMap, HashSet};

use crate::blueprint::{Blueprint, BlueprintTile};
use crate::tile_meta::TileType;

use super::HdlError;
use super::ast::WireEndpoint;
use super::semantic::CheckedModule;

/// Generate a Blueprint from a list of checked modules.
///
/// The last module in the list is treated as the top-level module and is
/// instantiated at origin (0, 0).
pub fn generate(modules: &[CheckedModule]) -> Result<Blueprint, Vec<HdlError>> {
    let top = modules.last().ok_or_else(|| {
        vec![HdlError {
            message: "no modules to generate".to_string(),
            line: 0,
            col: 0,
        }]
    })?;

    let (width, height) = estimate_grid_size(top, modules);
    let mut blueprint = Blueprint::new(width, height);
    let mut occupied: HashSet<(u32, u32)> = HashSet::new();

    place_module(top, modules, &mut blueprint, &mut occupied, 0, 0)?;

    Ok(blueprint)
}

/// Estimate the required grid size for a module and its sub-instances.
fn estimate_grid_size(module: &CheckedModule, all_modules: &[CheckedModule]) -> (u32, u32) {
    let mut max_x: u32 = 0;
    let mut max_y: u32 = 0;

    // Account for explicit tile positions
    for tile in &module.tiles {
        if let Some(ref pos) = tile.position {
            max_x = max_x.max(pos.x);
            max_y = max_y.max(pos.y);
        }
    }

    // Account for auto-placed tiles (we need space)
    let auto_count = module.tiles.iter().filter(|t| t.position.is_none()).count() as u32;
    // Rough estimate: sqrt(auto_count) * 3 spacing
    let side = ((auto_count as f64).sqrt().ceil() as u32 + 1) * 3;
    max_x = max_x.max(side);
    max_y = max_y.max(side);

    // Account for sub-instances
    for inst in &module.instances {
        let sub = &all_modules[inst.module_idx];
        let (sw, sh) = estimate_grid_size(sub, all_modules);
        let ox = inst.position.as_ref().map(|p| p.x).unwrap_or(0);
        let oy = inst.position.as_ref().map(|p| p.y).unwrap_or(0);
        max_x = max_x.max(ox + sw);
        max_y = max_y.max(oy + sh);
    }

    // Add margin for wire routing and ports
    let width = (max_x + 10).max(20);
    let height = (max_y + 10).max(20);
    (width, height)
}

/// Place a module's tiles and wires into the blueprint.
///
/// `offset_x` and `offset_y` are the origin of this module's placement
/// (used for sub-module instantiation).
fn place_module(
    module: &CheckedModule,
    all_modules: &[CheckedModule],
    blueprint: &mut Blueprint,
    occupied: &mut HashSet<(u32, u32)>,
    offset_x: u32,
    offset_y: u32,
) -> Result<HashMap<String, (u32, u32)>, Vec<HdlError>> {
    let mut errors = Vec::new();
    // Map from name -> grid position for tiles and ports
    let mut name_pos: HashMap<String, (u32, u32)> = HashMap::new();

    // Auto-placement cursor for tiles without explicit positions
    let mut cursor_x: u32 = offset_x + 1;
    let mut cursor_y: u32 = offset_y + 1;
    let auto_wrap_x: u32 = blueprint.width.saturating_sub(2);

    // Place port "virtual" positions along the edges
    // Input ports along the left edge, output ports along the right edge
    let mut input_y = offset_y + 1;
    let mut output_y = offset_y + 1;
    for port in &module.ports {
        match port.direction {
            super::ast::PortDir::Input => {
                let pos = (offset_x, input_y);
                name_pos.insert(port.name.clone(), pos);
                input_y += 2;
            }
            super::ast::PortDir::Output => {
                let pos = (
                    blueprint.width.saturating_sub(1).min(offset_x + 15),
                    output_y,
                );
                name_pos.insert(port.name.clone(), pos);
                output_y += 2;
            }
        }
    }

    // Place tiles
    for tile in &module.tiles {
        let (x, y) = if let Some(ref pos) = tile.position {
            (pos.x + offset_x, pos.y + offset_y)
        } else {
            // Auto-place with 3-cell spacing
            let pos = (cursor_x, cursor_y);
            cursor_x += 3;
            if cursor_x >= auto_wrap_x {
                cursor_x = offset_x + 1;
                cursor_y += 3;
            }
            pos
        };

        // Clamp to grid bounds
        if x >= blueprint.width || y >= blueprint.height {
            errors.push(HdlError {
                message: format!(
                    "tile '{}' position ({}, {}) is out of grid bounds ({}x{})",
                    tile.name, x, y, blueprint.width, blueprint.height
                ),
                line: 0,
                col: 0,
            });
            continue;
        }

        blueprint.tiles.push(BlueprintTile {
            x,
            y,
            z: 0,
            tile_type: tile.tile_type,
            logic: tile.initial_value,
        });
        occupied.insert((x, y));
        name_pos.insert(tile.name.clone(), (x, y));
    }

    // Place sub-module instances
    for inst in &module.instances {
        let sub = &all_modules[inst.module_idx];
        let ox = inst
            .position
            .as_ref()
            .map(|p| p.x + offset_x)
            .unwrap_or(offset_x + 5);
        let oy = inst
            .position
            .as_ref()
            .map(|p| p.y + offset_y)
            .unwrap_or(offset_y + 5);

        match place_module(sub, all_modules, blueprint, occupied, ox, oy) {
            Ok(sub_positions) => {
                // Map qualified port references: inst_name.port_name -> position
                for port in &sub.ports {
                    if let Some(&pos) = sub_positions.get(&port.name) {
                        let qualified_key = format!("{}.{}", inst.name, port.name);
                        name_pos.insert(qualified_key, pos);
                    }
                }
                // Also store the instance name itself at its origin
                name_pos.insert(inst.name.clone(), (ox, oy));
            }
            Err(mut errs) => errors.append(&mut errs),
        }
    }

    // Route wires
    for wire in &module.wires {
        let from_pos = resolve_endpoint_pos(&wire.from, &name_pos);
        let to_pos = resolve_endpoint_pos(&wire.to, &name_pos);

        match (from_pos, to_pos) {
            (Some(from), Some(to)) => {
                route_wire(from, to, blueprint, occupied);
            }
            (None, _) => {
                errors.push(HdlError {
                    message: format!(
                        "cannot resolve position for wire '{}' source endpoint",
                        wire.name
                    ),
                    line: 0,
                    col: 0,
                });
            }
            (_, None) => {
                errors.push(HdlError {
                    message: format!(
                        "cannot resolve position for wire '{}' destination endpoint",
                        wire.name
                    ),
                    line: 0,
                    col: 0,
                });
            }
        }
    }

    if errors.is_empty() {
        Ok(name_pos)
    } else {
        Err(errors)
    }
}

/// Resolve a wire endpoint to a grid position.
fn resolve_endpoint_pos(
    endpoint: &WireEndpoint,
    name_pos: &HashMap<String, (u32, u32)>,
) -> Option<(u32, u32)> {
    match endpoint {
        WireEndpoint::Port(name) | WireEndpoint::Tile(name) => name_pos.get(name).copied(),
        WireEndpoint::QualifiedPort(inst_name, port_name) => {
            let key = format!("{}.{}", inst_name, port_name);
            name_pos.get(&key).copied()
        }
    }
}

/// Route a wire between two grid positions using simple Manhattan (L-shaped) routing.
///
/// Places Wire tiles along the path, going horizontal first then vertical.
/// Skips cells that are already occupied.
fn route_wire(
    from: (u32, u32),
    to: (u32, u32),
    blueprint: &mut Blueprint,
    occupied: &mut HashSet<(u32, u32)>,
) {
    let (x0, y0) = from;
    let (x1, y1) = to;

    // Don't route if same position
    if x0 == x1 && y0 == y1 {
        return;
    }

    // Horizontal segment
    let x_start = x0.min(x1);
    let x_end = x0.max(x1);
    for x in x_start..=x_end {
        let pos = (x, y0);
        if pos != from && pos != (x1, y0) || (x == x1 && y0 == y1) {
            // Skip source and corner (corner handled in vertical)
            // But place intermediate Wire tiles
        }
        if pos != from && !occupied.contains(&pos) && x < blueprint.width && y0 < blueprint.height {
            blueprint.tiles.push(BlueprintTile {
                x,
                y: y0,
                z: 0,
                tile_type: TileType::Wire,
                logic: None,
            });
            occupied.insert(pos);
        }
    }

    // Vertical segment (from the corner at (x1, y0) to destination)
    if y0 != y1 {
        let y_start = y0.min(y1);
        let y_end = y0.max(y1);
        for y in y_start..=y_end {
            let pos = (x1, y);
            if pos != to
                && pos != (x1, y0)
                && !occupied.contains(&pos)
                && x1 < blueprint.width
                && y < blueprint.height
            {
                blueprint.tiles.push(BlueprintTile {
                    x: x1,
                    y,
                    z: 0,
                    tile_type: TileType::Wire,
                    logic: None,
                });
                occupied.insert(pos);
            }
        }
    }
}
