use std::collections::{HashMap, HashSet};

use crate::blueprint::parse_tile_type_token;
use crate::tile_meta::TileType;

use super::HdlError;
use super::ast::*;

/// A semantically checked module with resolved types.
#[derive(Debug, Clone)]
pub struct CheckedModule {
    pub name: String,
    pub ports: Vec<PortDecl>,
    pub tiles: Vec<CheckedTile>,
    pub wires: Vec<CheckedWire>,
    pub instances: Vec<CheckedInstance>,
    pub constants: Vec<(String, u64)>,
}

/// A tile declaration with resolved TileType.
#[derive(Debug, Clone)]
pub struct CheckedTile {
    pub name: String,
    pub tile_type: TileType,
    pub position: Option<Position>,
    pub initial_value: Option<u64>,
}

/// A checked wire with resolved endpoints.
#[derive(Debug, Clone)]
pub struct CheckedWire {
    pub name: String,
    pub from: WireEndpoint,
    pub to: WireEndpoint,
}

/// A checked module instantiation.
#[derive(Debug, Clone)]
pub struct CheckedInstance {
    pub name: String,
    pub module_idx: usize,
    pub args: Vec<String>,
    pub position: Option<Position>,
}

/// Perform semantic analysis on parsed modules.
///
/// Checks:
/// 1. Tile type name resolution (string -> TileType via parse_tile_type_token)
/// 2. Name uniqueness within each module (no duplicate tile/wire/instance names)
/// 3. Wire endpoint resolution (referenced tiles/ports/instances must exist)
/// 4. Port arity for instance arguments
/// 5. Module existence for instantiation
/// 6. No circular instantiation
pub fn check(modules: &[ModuleDecl]) -> Result<Vec<CheckedModule>, Vec<HdlError>> {
    let mut errors = Vec::new();

    // Build a name -> index map for module lookups
    let module_map: HashMap<&str, usize> = modules
        .iter()
        .enumerate()
        .map(|(i, m)| (m.name.as_str(), i))
        .collect();

    let mut checked = Vec::new();

    for module in modules {
        match check_module(module, &module_map, modules) {
            Ok(cm) => checked.push(cm),
            Err(mut errs) => errors.append(&mut errs),
        }
    }

    // Check for circular instantiation
    if let Some(cycle_err) = check_no_cycles(&checked, &module_map) {
        errors.push(cycle_err);
    }

    if errors.is_empty() {
        Ok(checked)
    } else {
        Err(errors)
    }
}

fn check_module(
    module: &ModuleDecl,
    module_map: &HashMap<&str, usize>,
    all_modules: &[ModuleDecl],
) -> Result<CheckedModule, Vec<HdlError>> {
    let mut errors = Vec::new();
    let mut names: HashSet<String> = HashSet::new();

    // Port names are valid names within the module
    let mut port_names: HashSet<String> = HashSet::new();
    for port in &module.ports {
        if !port_names.insert(port.name.clone()) {
            errors.push(HdlError {
                message: format!("duplicate port name: '{}'", port.name),
                line: 0,
                col: 0,
            });
        }
    }

    let mut tiles = Vec::new();
    let mut wires = Vec::new();
    let mut instances = Vec::new();
    let mut constants = Vec::new();
    let mut tile_names: HashSet<String> = HashSet::new();
    let mut instance_names: HashSet<String> = HashSet::new();

    for stmt in &module.body {
        match stmt {
            Statement::TileDecl {
                name,
                tile_type,
                position,
                initial_value,
            } => {
                // Check name uniqueness
                if !names.insert(name.clone()) {
                    errors.push(HdlError {
                        message: format!("duplicate name '{}' in module '{}'", name, module.name),
                        line: 0,
                        col: 0,
                    });
                    continue;
                }

                // Resolve tile type
                match parse_tile_type_token(tile_type) {
                    Some(tt) => {
                        tile_names.insert(name.clone());
                        tiles.push(CheckedTile {
                            name: name.clone(),
                            tile_type: tt,
                            position: position.clone(),
                            initial_value: *initial_value,
                        });
                    }
                    None => {
                        errors.push(HdlError {
                            message: format!(
                                "unknown tile type '{}' in module '{}'",
                                tile_type, module.name
                            ),
                            line: 0,
                            col: 0,
                        });
                    }
                }
            }

            Statement::WireDecl { name, from, to } => {
                if !names.insert(name.clone()) {
                    errors.push(HdlError {
                        message: format!("duplicate name '{}' in module '{}'", name, module.name),
                        line: 0,
                        col: 0,
                    });
                    continue;
                }
                wires.push(CheckedWire {
                    name: name.clone(),
                    from: from.clone(),
                    to: to.clone(),
                });
            }

            Statement::ConstDecl { name, value } => {
                if !names.insert(name.clone()) {
                    errors.push(HdlError {
                        message: format!("duplicate name '{}' in module '{}'", name, module.name),
                        line: 0,
                        col: 0,
                    });
                    continue;
                }
                constants.push((name.clone(), *value));
            }

            Statement::InstDecl {
                name,
                module_name,
                args,
                position,
            } => {
                if !names.insert(name.clone()) {
                    errors.push(HdlError {
                        message: format!("duplicate name '{}' in module '{}'", name, module.name),
                        line: 0,
                        col: 0,
                    });
                    continue;
                }

                // Check that the target module exists
                match module_map.get(module_name.as_str()) {
                    Some(&idx) => {
                        let target = &all_modules[idx];
                        // Check port arity: argument count must match input port count
                        let input_count = target
                            .ports
                            .iter()
                            .filter(|p| p.direction == PortDir::Input)
                            .count();
                        if args.len() != input_count {
                            errors.push(HdlError {
                                message: format!(
                                    "instance '{}' has {} arguments but module '{}' has {} input ports",
                                    name,
                                    args.len(),
                                    module_name,
                                    input_count
                                ),
                                line: 0,
                                col: 0,
                            });
                        }
                        instance_names.insert(name.clone());
                        instances.push(CheckedInstance {
                            name: name.clone(),
                            module_idx: idx,
                            args: args.clone(),
                            position: position.clone(),
                        });
                    }
                    None => {
                        errors.push(HdlError {
                            message: format!(
                                "unknown module '{}' instantiated as '{}'",
                                module_name, name
                            ),
                            line: 0,
                            col: 0,
                        });
                    }
                }
            }

            Statement::AsmDecl { name, source: _ } => {
                if !names.insert(name.clone()) {
                    errors.push(HdlError {
                        message: format!("duplicate name '{}' in module '{}'", name, module.name),
                        line: 0,
                        col: 0,
                    });
                }
                // TODO: Asm block validation (assembler integration)
            }
        }
    }

    // Validate wire endpoints
    for wire in &wires {
        if let Err(e) = validate_endpoint(
            &wire.from,
            &port_names,
            &tile_names,
            &instance_names,
            &module.name,
            &wire.name,
        ) {
            errors.push(e);
        }
        if let Err(e) = validate_endpoint(
            &wire.to,
            &port_names,
            &tile_names,
            &instance_names,
            &module.name,
            &wire.name,
        ) {
            errors.push(e);
        }
    }

    if errors.is_empty() {
        Ok(CheckedModule {
            name: module.name.clone(),
            ports: module.ports.clone(),
            tiles,
            wires,
            instances,
            constants,
        })
    } else {
        Err(errors)
    }
}

fn validate_endpoint(
    endpoint: &WireEndpoint,
    port_names: &HashSet<String>,
    tile_names: &HashSet<String>,
    instance_names: &HashSet<String>,
    module_name: &str,
    wire_name: &str,
) -> Result<(), HdlError> {
    match endpoint {
        WireEndpoint::Port(name) => {
            // Could be a port or a tile name
            if !port_names.contains(name) && !tile_names.contains(name) {
                return Err(HdlError {
                    message: format!(
                        "wire '{}' in module '{}' references unknown endpoint '{}'",
                        wire_name, module_name, name
                    ),
                    line: 0,
                    col: 0,
                });
            }
            Ok(())
        }
        WireEndpoint::Tile(name) => {
            if !tile_names.contains(name) {
                return Err(HdlError {
                    message: format!(
                        "wire '{}' in module '{}' references unknown tile '{}'",
                        wire_name, module_name, name
                    ),
                    line: 0,
                    col: 0,
                });
            }
            Ok(())
        }
        WireEndpoint::QualifiedPort(inst_name, _port_name) => {
            if !instance_names.contains(inst_name) {
                return Err(HdlError {
                    message: format!(
                        "wire '{}' in module '{}' references unknown instance '{}'",
                        wire_name, module_name, inst_name
                    ),
                    line: 0,
                    col: 0,
                });
            }
            // TODO: validate that the port_name exists on the target module
            Ok(())
        }
    }
}

/// Check for circular module instantiation using DFS.
fn check_no_cycles(
    checked: &[CheckedModule],
    _module_map: &HashMap<&str, usize>,
) -> Option<HdlError> {
    let n = checked.len();
    let mut visited = vec![0u8; n]; // 0=unvisited, 1=in-progress, 2=done

    for i in 0..n {
        if visited[i] == 0 && has_cycle(i, checked, &mut visited) {
            return Some(HdlError {
                message: format!(
                    "circular module instantiation detected involving '{}'",
                    checked[i].name
                ),
                line: 0,
                col: 0,
            });
        }
    }
    None
}

fn has_cycle(idx: usize, modules: &[CheckedModule], visited: &mut Vec<u8>) -> bool {
    visited[idx] = 1; // in-progress
    for inst in &modules[idx].instances {
        let target = inst.module_idx;
        if visited[target] == 1 {
            return true; // back edge = cycle
        }
        if visited[target] == 0 && has_cycle(target, modules, visited) {
            return true;
        }
    }
    visited[idx] = 2; // done
    false
}
