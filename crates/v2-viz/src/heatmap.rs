//! Heatmap mode — accumulate activation counts over many cycles.
//!
//! Toggle with H key. Shows which tiles are hot (frequently changing) vs cold
//! (rarely or never active). Reset with Ctrl+H.

use bevy::prelude::*;

use crate::grid::{ActiveLayer, TILE_SIZE};
use crate::sim::SimSnapshot;

/// Heatmap state — tracks per-tile activation counts.
#[derive(Resource)]
pub struct HeatmapState {
    pub enabled: bool,
    /// Per-tile activation count (flat array, all layers).
    pub counts: Vec<u32>,
    /// Maximum count seen (for normalization).
    pub max_count: u32,
    /// Total cycles accumulated.
    pub total_cycles: u64,
}

impl Default for HeatmapState {
    fn default() -> Self {
        Self {
            enabled: false,
            counts: Vec::new(),
            max_count: 0,
            total_cycles: 0,
        }
    }
}

/// Marker for heatmap overlay sprites.
#[derive(Component)]
pub struct HeatmapOverlay;

pub struct HeatmapPlugin;

impl Plugin for HeatmapPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HeatmapState>()
            .add_systems(Update, (toggle_heatmap, accumulate_heatmap, render_heatmap));
    }
}

fn toggle_heatmap(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut hm: ResMut<HeatmapState>,
    snapshot: Res<SimSnapshot>,
) {
    if keyboard.just_pressed(KeyCode::KeyH) {
        if keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight) {
            // Ctrl+H = reset
            hm.counts.fill(0);
            hm.max_count = 0;
            hm.total_cycles = 0;
        } else {
            hm.enabled = !hm.enabled;
            // Lazy init counts.
            let total_tiles = snapshot.width * snapshot.height * snapshot.num_layers;
            if hm.counts.len() != total_tiles {
                hm.counts = vec![0u32; total_tiles];
            }
        }
    }
}

fn accumulate_heatmap(mut hm: ResMut<HeatmapState>, snapshot: Res<SimSnapshot>) {
    if !hm.enabled || snapshot.changed_indices.is_empty() {
        return;
    }

    // Lazy init.
    let total_tiles = snapshot.width * snapshot.height * snapshot.num_layers;
    if hm.counts.len() != total_tiles {
        hm.counts = vec![0u32; total_tiles];
    }

    for &idx in &snapshot.changed_indices {
        if idx < hm.counts.len() {
            hm.counts[idx] = hm.counts[idx].saturating_add(1);
            if hm.counts[idx] > hm.max_count {
                hm.max_count = hm.counts[idx];
            }
        }
    }
    hm.total_cycles += 1;
}

fn render_heatmap(
    mut commands: Commands,
    hm: Res<HeatmapState>,
    snapshot: Res<SimSnapshot>,
    layer: Res<ActiveLayer>,
    existing: Query<Entity, With<HeatmapOverlay>>,
) {
    // Despawn old overlays.
    for entity in existing.iter() {
        commands.entity(entity).despawn();
    }

    if !hm.enabled || hm.max_count == 0 {
        return;
    }

    let width = snapshot.width;
    let height = snapshot.height;
    let layer_size = width * height;
    let base_idx = layer.z * layer_size;

    let max = hm.max_count as f32;

    for y in 0..height {
        for x in 0..width {
            let idx = base_idx + y * width + x;
            let count = if idx < hm.counts.len() {
                hm.counts[idx]
            } else {
                0
            };
            if count == 0 {
                continue;
            }

            let t = (count as f32 / max).sqrt(); // sqrt for better dynamic range

            // Cold (blue) → warm (yellow) → hot (red)
            let (r, g, b) = if t < 0.5 {
                let s = t * 2.0;
                (s, s, 1.0 - s) // blue → yellow
            } else {
                let s = (t - 0.5) * 2.0;
                (1.0, 1.0 - s, 0.0) // yellow → red
            };

            let world_x = x as f32 * TILE_SIZE + TILE_SIZE * 0.5;
            let world_y = -(y as f32 * TILE_SIZE + TILE_SIZE * 0.5);

            commands.spawn((
                Sprite {
                    color: Color::srgba(r, g, b, 0.5),
                    custom_size: Some(Vec2::new(TILE_SIZE - 0.5, TILE_SIZE - 0.5)),
                    ..default()
                },
                Transform::from_xyz(world_x, world_y, 3.0), // Above wavefront
                HeatmapOverlay,
            ));
        }
    }
}
