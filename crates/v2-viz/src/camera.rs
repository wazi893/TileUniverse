//! 2D camera with zoom (scroll) and pan (middle-drag or right-drag).

use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;

/// Marker for the main visualization camera.
#[derive(Component)]
pub struct VizCamera;

/// Tracks drag state for panning.
#[derive(Resource, Default)]
struct DragState {
    active: bool,
    last_pos: Vec2,
}

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DragState>()
            .add_systems(Startup, spawn_camera)
            .add_systems(Update, camera_pan);
    }
}

fn spawn_camera(mut commands: Commands) {
    let tile_size = 4.0;
    let center_x = 128.0 * tile_size / 2.0;
    let center_y = -(640.0 * tile_size / 2.0);

    commands.spawn((
        Camera2d,
        Transform::from_xyz(center_x, center_y, 999.0),
        VizCamera,
    ));
}

// Zoom is handled in camera_pan via Transform scale.

fn camera_pan(
    mut query: Query<&mut Transform, With<VizCamera>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut drag: ResMut<DragState>,
    mut scroll_events: EventReader<MouseWheel>,
    windows: Query<&Window>,
) {
    let Ok(window) = windows.single() else {
        return;
    };

    // Zoom via scroll — adjust camera Z scale.
    let mut zoom_delta = 0.0f32;
    for ev in scroll_events.read() {
        zoom_delta += match ev.unit {
            MouseScrollUnit::Line => ev.y * 0.15,
            MouseScrollUnit::Pixel => ev.y * 0.002,
        };
    }
    if zoom_delta != 0.0 {
        for mut transform in &mut query {
            let factor = (1.0 - zoom_delta).clamp(0.5, 2.0);
            let new_scale = (transform.scale.x * factor).clamp(0.05, 30.0);
            transform.scale = Vec3::splat(new_scale);
        }
    }

    // Pan via middle/right drag.
    let pressed = mouse.pressed(MouseButton::Middle) || mouse.pressed(MouseButton::Right);

    if let Some(pos) = window.cursor_position() {
        if pressed {
            if drag.active {
                let delta = pos - drag.last_pos;
                for mut transform in &mut query {
                    let scale = transform.scale.x;
                    transform.translation.x -= delta.x * scale;
                    transform.translation.y += delta.y * scale;
                }
            }
            drag.active = true;
            drag.last_pos = pos;
        } else {
            drag.active = false;
        }
    } else {
        drag.active = false;
    }
}
