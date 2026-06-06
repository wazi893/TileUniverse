//! Scope overlay — dims tiles outside the selected evaluation scope.
//!
//! When enabled, tiles NOT in the pipeline/branch/commit scope are darkened,
//! making the active evaluation region visually obvious.

use bevy::prelude::*;

use crate::grid::{ActiveLayer, TILE_SIZE};
use crate::sim::{SimSnapshot, SimState};
use crate::wavefront::EvalScope;

/// Controls for scope overlay.
#[derive(Resource)]
pub struct ScopeOverlayState {
    pub enabled: bool,
    pub scope: EvalScope,
    /// Cached scope membership bitset (L1 summary, 64-tile segments).
    scope_mask: Vec<u64>,
}

impl Default for ScopeOverlayState {
    fn default() -> Self {
        Self {
            enabled: false,
            scope: EvalScope::Pipeline,
            scope_mask: Vec::new(),
        }
    }
}

/// Marker for scope dimming overlay sprites.
#[derive(Component)]
pub struct ScopeDimOverlay;

pub struct ScopePlugin;

impl Plugin for ScopePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScopeOverlayState>()
            .add_systems(Update, (toggle_scope, render_scope_overlay));
    }
}

fn toggle_scope(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut scope_state: ResMut<ScopeOverlayState>,
    state: NonSend<SimState>,
) {
    if keyboard.just_pressed(KeyCode::KeyP) {
        scope_state.enabled = !scope_state.enabled;
        if scope_state.enabled {
            refresh_scope_mask(&mut scope_state, &state);
        }
    }

    // Cycle scope with Shift+P
    if keyboard.just_pressed(KeyCode::KeyP)
        && (keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight))
    {
        scope_state.scope = match scope_state.scope {
            EvalScope::Pipeline => EvalScope::Branch,
            EvalScope::Branch => EvalScope::Commit,
            EvalScope::Commit => EvalScope::Pipeline,
        };
        refresh_scope_mask(&mut scope_state, &state);
    }
}

fn refresh_scope_mask(scope_state: &mut ScopeOverlayState, state: &SimState) {
    scope_state.scope_mask = match scope_state.scope {
        EvalScope::Pipeline => state.cpu.pipeline_scope_mask().to_vec(),
        EvalScope::Branch => state.cpu.branch_scope_mask().to_vec(),
        EvalScope::Commit => state.cpu.commit_scope_mask().to_vec(),
    };
}

/// Check if a tile index is in the scope mask.
fn is_in_scope(idx: usize, mask: &[u64]) -> bool {
    let seg = idx / 64;
    let bit = idx % 64;
    seg < mask.len() && (mask[seg] & (1u64 << bit)) != 0
}

fn render_scope_overlay(
    mut commands: Commands,
    scope_state: Res<ScopeOverlayState>,
    snapshot: Res<SimSnapshot>,
    layer: Res<ActiveLayer>,
    existing: Query<Entity, With<ScopeDimOverlay>>,
) {
    for entity in existing.iter() {
        commands.entity(entity).despawn();
    }

    if !scope_state.enabled || scope_state.scope_mask.is_empty() {
        return;
    }

    let width = snapshot.width;
    let height = snapshot.height;
    let layer_size = width * height;
    let base_idx = layer.z * layer_size;

    // Dark overlay on tiles NOT in scope.
    let dim_color = Color::srgba(0.0, 0.0, 0.0, 0.6);

    for y in 0..height {
        for x in 0..width {
            let idx = base_idx + y * width + x;
            if is_in_scope(idx, &scope_state.scope_mask) {
                continue; // In scope — don't dim.
            }

            let world_x = x as f32 * TILE_SIZE + TILE_SIZE * 0.5;
            let world_y = -(y as f32 * TILE_SIZE + TILE_SIZE * 0.5);

            commands.spawn((
                Sprite {
                    color: dim_color,
                    custom_size: Some(Vec2::new(TILE_SIZE - 0.5, TILE_SIZE - 0.5)),
                    ..default()
                },
                Transform::from_xyz(world_x, world_y, 4.0), // Above heatmap
                ScopeDimOverlay,
            ));
        }
    }
}
