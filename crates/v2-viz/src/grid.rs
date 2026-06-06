//! Tile grid rendering — one colored quad per tile, updated each frame.

use bevy::prelude::*;

use crate::sim::{SimSnapshot, SimState};
use crate::tile_color;

/// Pixel size of each tile quad in world space.
pub const TILE_SIZE: f32 = 4.0;

/// Which layer is currently being displayed.
#[derive(Resource)]
pub struct ActiveLayer {
    pub z: usize,
}

impl Default for ActiveLayer {
    fn default() -> Self {
        Self { z: 0 }
    }
}

/// Marker component on tile sprite entities.
#[derive(Component)]
pub struct TileQuad {
    /// Flat index into the simulation tile array.
    pub idx: usize,
}

/// Marker for the "changed this cycle" highlight overlay sprites.
#[derive(Component)]
pub struct ActiveHighlight;

pub struct GridPlugin;

impl Plugin for GridPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveLayer>()
            .add_systems(PostStartup, spawn_grid)
            .add_systems(
                Update,
                (
                    update_tile_colors,
                    update_active_highlights,
                    handle_layer_switch,
                ),
            );
    }
}

fn spawn_grid(mut commands: Commands, state: NonSend<SimState>) {
    let width = state.sim.width();
    let height = state.sim.height();
    let layer_size = width * height;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x; // Layer 0
            let tt = state.sim.tilemap.tiles[idx].meta.tile_type;
            let logic = state.sim.get_logic_value_by_idx(idx);
            let color = tile_color::tile_bevy_color(tt, logic != 0);

            let world_x = x as f32 * TILE_SIZE + TILE_SIZE * 0.5;
            let world_y = -(y as f32 * TILE_SIZE + TILE_SIZE * 0.5);

            commands.spawn((
                Sprite {
                    color,
                    custom_size: Some(Vec2::new(TILE_SIZE - 0.5, TILE_SIZE - 0.5)),
                    ..default()
                },
                Transform::from_xyz(world_x, world_y, 0.0),
                TileQuad { idx },
            ));
        }
    }

    let _ = layer_size; // suppress warning
}

/// Update tile sprite colors when the simulation steps.
fn update_tile_colors(
    state: NonSend<SimState>,
    snapshot: Res<SimSnapshot>,
    layer: Res<ActiveLayer>,
    mut query: Query<(&TileQuad, &mut Sprite)>,
) {
    if snapshot.changed_indices.is_empty() {
        return;
    }

    let layer_size = snapshot.width * snapshot.height;
    let base_idx = layer.z * layer_size;
    let end_idx = base_idx + layer_size;

    for (tq, mut sprite) in query.iter_mut() {
        if tq.idx >= base_idx && tq.idx < end_idx {
            let tt = state.sim.tilemap.tiles[tq.idx].meta.tile_type;
            let logic = state.sim.get_logic_value_by_idx(tq.idx);
            sprite.color = tile_color::tile_bevy_color(tt, logic != 0);
        }
    }
}

/// Manage highlight overlays for tiles that changed this cycle.
fn update_active_highlights(
    mut commands: Commands,
    snapshot: Res<SimSnapshot>,
    layer: Res<ActiveLayer>,
    existing: Query<Entity, With<ActiveHighlight>>,
) {
    for entity in existing.iter() {
        commands.entity(entity).despawn();
    }

    if snapshot.changed_indices.is_empty() {
        return;
    }

    let layer_size = snapshot.width * snapshot.height;
    let base_idx = layer.z * layer_size;
    let end_idx = base_idx + layer_size;

    let highlight_color = tile_color::active_highlight_color();

    for &idx in &snapshot.changed_indices {
        if idx < base_idx || idx >= end_idx {
            continue;
        }
        let local = idx - base_idx;
        let x = local % snapshot.width;
        let y = local / snapshot.width;

        let world_x = x as f32 * TILE_SIZE + TILE_SIZE * 0.5;
        let world_y = -(y as f32 * TILE_SIZE + TILE_SIZE * 0.5);

        commands.spawn((
            Sprite {
                color: highlight_color,
                custom_size: Some(Vec2::new(TILE_SIZE - 0.5, TILE_SIZE - 0.5)),
                ..default()
            },
            Transform::from_xyz(world_x, world_y, 1.0),
            ActiveHighlight,
        ));
    }
}

/// Switch layers with number keys 0-9.
fn handle_layer_switch(
    mut layer: ResMut<ActiveLayer>,
    state: NonSend<SimState>,
    snapshot: Res<SimSnapshot>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut TileQuad, &mut Sprite)>,
) {
    let key_map: &[(KeyCode, usize)] = &[
        (KeyCode::Digit0, 0),
        (KeyCode::Digit1, 1),
        (KeyCode::Digit2, 2),
        (KeyCode::Digit3, 3),
        (KeyCode::Digit4, 4),
        (KeyCode::Digit5, 5),
        (KeyCode::Digit6, 6),
        (KeyCode::Digit7, 7),
        (KeyCode::Digit8, 8),
        (KeyCode::Digit9, 9),
    ];

    let mut new_z = None;
    for &(key, z) in key_map {
        if keyboard.just_pressed(key) && z < snapshot.num_layers {
            new_z = Some(z);
            break;
        }
    }

    let Some(z) = new_z else { return };
    if z == layer.z {
        return;
    }
    layer.z = z;

    let layer_size = snapshot.width * snapshot.height;
    let base_idx = z * layer_size;

    let mut i = 0;
    for (mut tq, mut sprite) in query.iter_mut() {
        let idx = base_idx + i;
        tq.idx = idx;
        let tt = state.sim.tilemap.tiles[idx].meta.tile_type;
        let logic = state.sim.get_logic_value_by_idx(idx);
        sprite.color = tile_color::tile_bevy_color(tt, logic != 0);
        i += 1;
        if i >= layer_size {
            break;
        }
    }
}
