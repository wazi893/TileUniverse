mod camera;
mod grid;
mod heatmap;
mod inspector;
mod scope;
mod sim;
mod tile_color;
mod wavefront;

use bevy::prelude::*;
use bevy_egui::EguiPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "TileUniverse V2 — Tile CPU Visualizer".into(),
                resolution: (1600.0, 900.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .insert_resource(ClearColor(Color::srgb(0.06, 0.06, 0.08)))
        .add_plugins(sim::SimPlugin)
        .add_plugins(grid::GridPlugin)
        .add_plugins(camera::CameraPlugin)
        .add_plugins(wavefront::WavefrontPlugin)
        .add_plugins(heatmap::HeatmapPlugin)
        .add_plugins(scope::ScopePlugin)
        .add_plugins(inspector::InspectorPlugin)
        .run();
}
