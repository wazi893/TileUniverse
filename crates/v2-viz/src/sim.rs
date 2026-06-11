//! Simulation state management — builds the V2 CPU, steps it, tracks changes.

use bevy::prelude::*;
use engine::simulation::Simulation;
use engine::tile_cpu::{
    TileCpuV2, V2Builder, V2SynthConfig, V2TraceLog, assemble_v2, benchmark_cases,
};

/// Wrapper holding the simulation and CPU state.
/// Not Send/Sync due to Cell<> usage in TileCpuV2 — use NonSend/NonSendMut.
pub struct SimState {
    pub sim: Simulation,
    pub cpu: TileCpuV2,
    pub trace: V2TraceLog,
}

/// Send+Sync snapshot of CPU state, updated each cycle for the UI and grid renderer.
#[derive(Resource)]
pub struct SimSnapshot {
    pub cycle: u64,
    pub pc: u32,
    pub flag_z: bool,
    pub flag_c: bool,
    pub halted: bool,
    pub regs: [u64; 16],
    pub ram: [u64; 16], // First 16 cells for display
    pub asm: String,
    /// Indices of tiles whose logic value changed this cycle.
    pub changed_indices: Vec<usize>,
    /// Previous logic values for diff detection (kept here, updated by step).
    prev_logic: Vec<u64>,
    /// Grid dimensions (cached from sim).
    pub width: usize,
    pub height: usize,
    pub num_layers: usize,
}

/// Controls for stepping / running the simulation.
#[derive(Resource)]
pub struct SimControls {
    pub running: bool,
    pub speed: f64,
    pub accum: f64,
    pub step_requested: bool,
}

impl Default for SimControls {
    fn default() -> Self {
        Self {
            running: false,
            speed: 10.0,
            accum: 0.0,
            step_requested: false,
        }
    }
}

pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        // Build the V2 CPU during app init (main thread).
        let cases = benchmark_cases();
        let case = &cases[0];
        let program = assemble_v2(case.source).expect("benchmark asm failed");

        let mut sim = Simulation::with_size_layered(128, 640, 16);
        let cpu = V2Builder::new()
            .with_origin(0, 0)
            .with_program(&program)
            .with_rom_size(128)
            .with_ram_size(128)
            .with_synth_blocks(V2SynthConfig::max_authority())
            .build(&mut sim);

        // Build initial snapshot.
        let tile_count = sim.tile_count();
        let mut prev_logic = Vec::with_capacity(tile_count);
        for i in 0..tile_count {
            prev_logic.push(sim.get_logic_value_by_idx(i));
        }

        let mut regs = [0u64; 16];
        for r in 0..16 {
            regs[r] = cpu.read_reg(&sim, r);
        }
        let mut ram = [0u64; 16];
        for a in 0..16 {
            ram[a] = cpu.read_ram(&sim, a);
        }

        let width = sim.width();
        let height = sim.height();
        let num_layers = sim.num_layers();

        let snapshot = SimSnapshot {
            cycle: 0,
            pc: cpu.read_pc(&sim),
            flag_z: cpu.read_flag_z(&sim),
            flag_c: cpu.read_flag_c(&sim),
            halted: cpu.is_halted(),
            regs,
            ram,
            asm: String::new(),
            changed_indices: Vec::new(),
            prev_logic,
            width,
            height,
            num_layers,
        };

        app.insert_non_send_resource(SimState {
            sim,
            cpu,
            trace: V2TraceLog::new(),
        });

        app.insert_resource(snapshot);
        app.insert_resource(SimControls::default());
        app.add_systems(Update, step_sim);
    }
}

fn step_sim(
    mut state: NonSendMut<SimState>,
    mut snapshot: ResMut<SimSnapshot>,
    mut controls: ResMut<SimControls>,
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        controls.step_requested = true;
    }
    if keyboard.just_pressed(KeyCode::Enter) {
        controls.running = !controls.running;
        controls.accum = 0.0;
    }
    if keyboard.just_pressed(KeyCode::Escape) {
        controls.running = false;
    }

    let should_step = if controls.running {
        controls.accum += time.delta_secs_f64() * controls.speed;
        if controls.accum >= 1.0 {
            controls.accum -= 1.0;
            true
        } else {
            false
        }
    } else if controls.step_requested {
        controls.step_requested = false;
        true
    } else {
        false
    };

    if !should_step || state.cpu.is_halted() {
        return;
    }

    // Step the CPU.
    let s = &mut *state;
    s.cpu.step_with_trace(&mut s.sim, &mut s.trace);
    snapshot.cycle += 1;

    // Detect which tiles changed.
    snapshot.changed_indices.clear();
    let tile_count = s.sim.tile_count();
    for i in 0..tile_count {
        let val = s.sim.get_logic_value_by_idx(i);
        if val != snapshot.prev_logic[i] {
            snapshot.changed_indices.push(i);
            snapshot.prev_logic[i] = val;
        }
    }

    // Update snapshot.
    snapshot.pc = s.cpu.read_pc(&s.sim);
    snapshot.flag_z = s.cpu.read_flag_z(&s.sim);
    snapshot.flag_c = s.cpu.read_flag_c(&s.sim);
    snapshot.halted = s.cpu.is_halted();
    for r in 0..16 {
        snapshot.regs[r] = s.cpu.read_reg(&s.sim, r);
    }
    for a in 0..16 {
        snapshot.ram[a] = s.cpu.read_ram(&s.sim, a);
    }

    // Get last trace entry ASM.
    let entries = s.trace.entries();
    if let Some(entry) = entries.last() {
        snapshot.asm.clear();
        snapshot.asm.push_str(&entry.asm);
    }
}
