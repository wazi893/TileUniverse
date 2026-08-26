use std::collections::HashSet;

use crate::simulation::Simulation;
use crate::tile_meta::TileType;
use crate::tilemap::{HEIGHT, WIDTH};

#[derive(Debug, Clone)]
pub struct SimVm {
    pub id: u32,
    pub rack_x: u16,
    pub rack_y: u16,
    pub cpu_millis: u32,
    pub ram_mb: u32,
    pub uptime_ticks: u32,
    pub is_healthy: bool,
}

#[derive(Debug, Clone)]
pub struct DemoPresetInfo {
    pub name: String,
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
}

#[derive(Debug, Clone)]
pub struct DemoError {
    pub reason: String,
}

#[derive(Debug)]
pub struct DemoState {
    pub vms: Vec<SimVm>,
    pub spawned_spawners: HashSet<(u16, u16)>,
    pub next_vm_id: u32,
    pub region: (usize, usize, usize, usize),
}

impl DemoState {
    pub fn new(info: &DemoPresetInfo) -> Self {
        Self {
            vms: Vec::new(),
            spawned_spawners: HashSet::new(),
            next_vm_id: 1,
            region: (info.x0, info.y0, info.w, info.h),
        }
    }
}

pub fn build_preset(sim: &mut Simulation, preset: u32) -> Result<DemoPresetInfo, DemoError> {
    match preset {
        0 => build_preset_seed(sim),
        1 => build_preset_datacenter(sim),
        _ => Err(DemoError {
            reason: format!("unknown preset {}", preset),
        }),
    }
}

fn build_preset_seed(sim: &mut Simulation) -> Result<DemoPresetInfo, DemoError> {
    let x0 = 10usize;
    let y0 = 10usize;
    let w = 8usize;
    let h = 8usize;
    if x0 + w > WIDTH || y0 + h > HEIGHT {
        return Err(DemoError {
            reason: "out-of-bounds preset region".to_string(),
        });
    }
    for y in y0..(y0 + h) {
        for x in x0..(x0 + w) {
            sim.set_tile_type(x, y, TileType::Wire);
            sim.set_logic_value(x, y, 0);
        }
    }
    // A single row with a spawner, vm slot, and status
    let row = y0 + 3;
    sim.set_tile_type(x0, row, TileType::VmSpawner);
    sim.set_tile_type(x0 + 1, row, TileType::Wire);
    sim.set_tile_type(x0 + 3, row, TileType::VmStatus);
    Ok(DemoPresetInfo {
        name: "seed".to_string(),
        x0,
        y0,
        w,
        h,
    })
}

fn build_preset_datacenter(sim: &mut Simulation) -> Result<DemoPresetInfo, DemoError> {
    let base_x = 10usize;
    let base_y = 10usize;
    let rows = 4usize;
    for r in 0..rows {
        let y = base_y + r;
        sim.set_tile_type(base_x, y, TileType::VmSpawner);
        sim.set_tile_type(base_x + 1, y, TileType::Wire);
        sim.set_tile_type(base_x + 3, y, TileType::VmStatus);
    }
    Ok(DemoPresetInfo {
        name: "datacenter".to_string(),
        x0: base_x,
        y0: base_y,
        w: 6,
        h: rows,
    })
}

pub fn step_demo(sim: &mut Simulation, state: &mut DemoState) -> Result<(), DemoError> {
    let (x0, y0, w, h) = state.region;
    // Spawners: rising edge simplified to: logic high and not yet spawned at this coord
    for y in y0..(y0 + h) {
        for x in x0..(x0 + w) {
            if sim.tile_type_xy(x, y) == TileType::VmSpawner {
                let key = (x as u16, y as u16);
                let logic = sim.get_logic_at(x, y);
                if logic != 0 && !state.spawned_spawners.contains(&key) {
                    let vm = SimVm {
                        id: state.next_vm_id,
                        rack_x: (x + 1) as u16,
                        rack_y: y as u16,
                        cpu_millis: 500,
                        ram_mb: 1024,
                        uptime_ticks: 0,
                        is_healthy: true,
                    };
                    state.next_vm_id = state.next_vm_id.saturating_add(1);
                    state.vms.push(vm);
                    state.spawned_spawners.insert(key);
                    // initial heat pulse
                    sim.add_heat_at(x + 1, y, 10);
                }
            }
        }
    }
    // VMs update
    for vm in &mut state.vms {
        vm.uptime_ticks = vm.uptime_ticks.saturating_add(1);
        let phase = (vm.uptime_ticks / 32) % 3;
        let extra_heat = match phase {
            0 => 2,
            1 => 8,
            _ => 4,
        };
        let vx = vm.rack_x as usize;
        let vy = vm.rack_y as usize;
        if vx < WIDTH && vy < HEIGHT {
            sim.add_heat_at(vx, vy, extra_heat);
        }
        if vm.uptime_ticks > 300 {
            vm.is_healthy = false;
        }
    }
    // Status tiles: write logic value
    for y in y0..(y0 + h) {
        for x in x0..(x0 + w) {
            if sim.tile_type_xy(x, y) == TileType::VmStatus {
                let target_x = x.saturating_sub(2);
                let target_y = y;
                let mut val = 0u64;
                for vm in &state.vms {
                    if vm.rack_x as usize == target_x && vm.rack_y as usize == target_y {
                        val = if vm.is_healthy { 1 } else { 2 };
                        break;
                    }
                }
                let _ = sim.set_logic_value(x, y, val);
            }
        }
    }
    Ok(())
}
