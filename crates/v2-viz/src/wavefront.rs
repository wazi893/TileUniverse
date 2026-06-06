//! Wavefront animation — tiles light up in topological eval order.
//!
//! After each cycle step, the wavefront replays the eval order with a time-based
//! sweep. Each tile gets an "activation time" based on its position in the eval
//! order, and the sweep progresses over a configurable duration.

use bevy::prelude::*;

use crate::grid::{ActiveLayer, TILE_SIZE};
use crate::sim::{SimSnapshot, SimState};

/// Controls for wavefront visualization.
#[derive(Resource)]
pub struct WavefrontState {
    /// If true, wavefront animation is active.
    pub enabled: bool,
    /// Duration (seconds) for the full sweep across all eval-order tiles.
    pub sweep_duration: f32,
    /// Current sweep progress (0.0 = start, 1.0 = complete).
    pub progress: f32,
    /// Which scope to visualize.
    pub scope: EvalScope,
    /// Cached eval order for current scope (tile indices).
    eval_order: Vec<usize>,
    /// Whether the sweep is currently playing.
    pub playing: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EvalScope {
    Pipeline,
    Branch,
    Commit,
}

impl Default for WavefrontState {
    fn default() -> Self {
        Self {
            enabled: false,
            sweep_duration: 2.0,
            progress: 0.0,
            scope: EvalScope::Pipeline,
            eval_order: Vec::new(),
            playing: false,
        }
    }
}

/// Marker for wavefront overlay sprites.
#[derive(Component)]
pub struct WavefrontOverlay;

pub struct WavefrontPlugin;

impl Plugin for WavefrontPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WavefrontState>().add_systems(
            Update,
            (toggle_wavefront, advance_wavefront, render_wavefront),
        );
    }
}

/// Toggle wavefront mode with W key. Cycle scope with S key.
fn toggle_wavefront(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut wf: ResMut<WavefrontState>,
    state: NonSend<SimState>,
) {
    if keyboard.just_pressed(KeyCode::KeyW) {
        wf.enabled = !wf.enabled;
        if wf.enabled {
            // Cache the eval order for the selected scope.
            refresh_eval_order(&mut wf, &state);
            wf.progress = 0.0;
            wf.playing = true;
        }
    }

    if keyboard.just_pressed(KeyCode::KeyS) && wf.enabled {
        wf.scope = match wf.scope {
            EvalScope::Pipeline => EvalScope::Branch,
            EvalScope::Branch => EvalScope::Commit,
            EvalScope::Commit => EvalScope::Pipeline,
        };
        refresh_eval_order(&mut wf, &state);
        wf.progress = 0.0;
        wf.playing = true;
    }

    // R = restart sweep
    if keyboard.just_pressed(KeyCode::KeyR) && wf.enabled {
        wf.progress = 0.0;
        wf.playing = true;
    }
}

fn refresh_eval_order(wf: &mut WavefrontState, state: &SimState) {
    wf.eval_order = match wf.scope {
        EvalScope::Pipeline => state.cpu.pipeline_eval_order().to_vec(),
        EvalScope::Branch => state.cpu.branch_eval_order().to_vec(),
        EvalScope::Commit => state.cpu.commit_eval_order().to_vec(),
    };
}

/// Advance the sweep timer.
fn advance_wavefront(mut wf: ResMut<WavefrontState>, time: Res<Time>) {
    if !wf.enabled || !wf.playing {
        return;
    }
    wf.progress += time.delta_secs() / wf.sweep_duration;
    if wf.progress >= 1.0 {
        wf.progress = 1.0;
        wf.playing = false;
    }
}

/// Render wavefront overlays — tiles colored by their position in the eval order.
fn render_wavefront(
    mut commands: Commands,
    wf: Res<WavefrontState>,
    snapshot: Res<SimSnapshot>,
    layer: Res<ActiveLayer>,
    existing: Query<Entity, With<WavefrontOverlay>>,
) {
    // Despawn old overlays.
    for entity in existing.iter() {
        commands.entity(entity).despawn();
    }

    if !wf.enabled || wf.eval_order.is_empty() {
        return;
    }

    let width = snapshot.width;
    let height = snapshot.height;
    let layer_size = width * height;
    let base_idx = layer.z * layer_size;
    let end_idx = base_idx + layer_size;

    let total = wf.eval_order.len();
    let visible_count = ((wf.progress * total as f32) as usize).min(total);

    for (order_pos, &idx) in wf.eval_order.iter().enumerate().take(visible_count) {
        if idx < base_idx || idx >= end_idx {
            continue;
        }

        let local = idx - base_idx;
        let x = local % width;
        let y = local / width;

        // Color gradient: blue (early) → red (late) in the eval order.
        let t = order_pos as f32 / total.max(1) as f32;
        let r = t;
        let g = 0.2 * (1.0 - t);
        let b = 1.0 - t;

        let world_x = x as f32 * TILE_SIZE + TILE_SIZE * 0.5;
        let world_y = -(y as f32 * TILE_SIZE + TILE_SIZE * 0.5);

        commands.spawn((
            Sprite {
                color: Color::srgba(r, g, b, 0.6),
                custom_size: Some(Vec2::new(TILE_SIZE - 0.5, TILE_SIZE - 0.5)),
                ..default()
            },
            Transform::from_xyz(world_x, world_y, 2.0), // Above active highlights
            WavefrontOverlay,
        ));
    }
}
