//! egui inspector panel — registers, flags, PC, RAM, disassembly, tile info,
//! wavefront/heatmap/scope controls.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

use crate::grid::ActiveLayer;
use crate::heatmap::HeatmapState;
use crate::scope::ScopeOverlayState;
use crate::sim::{SimControls, SimSnapshot};
use crate::tile_color;
use crate::wavefront::{EvalScope, WavefrontState};

pub struct InspectorPlugin;

impl Plugin for InspectorPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, draw_inspector);
    }
}

fn scope_label(scope: EvalScope) -> &'static str {
    match scope {
        EvalScope::Pipeline => "Pipeline",
        EvalScope::Branch => "Branch",
        EvalScope::Commit => "Commit",
    }
}

fn draw_inspector(
    mut contexts: EguiContexts,
    snapshot: Res<SimSnapshot>,
    mut controls: ResMut<SimControls>,
    mut layer: ResMut<ActiveLayer>,
    mut wf: ResMut<WavefrontState>,
    mut hm: ResMut<HeatmapState>,
    mut scope_overlay: ResMut<ScopeOverlayState>,
    mut initialized: Local<bool>,
) {
    if !*initialized {
        *initialized = true;
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    egui::SidePanel::left("inspector")
        .default_width(280.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.heading("TileUniverse V2 CPU");
            ui.separator();

            // ── Sim Controls ──
            ui.label(egui::RichText::new("Controls").strong());
            ui.horizontal(|ui| {
                if ui
                    .button(if controls.running { "Pause" } else { "Run" })
                    .clicked()
                {
                    controls.running = !controls.running;
                    controls.accum = 0.0;
                }
                if ui.button("Step").clicked() {
                    controls.step_requested = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Speed:");
                ui.add(
                    egui::Slider::new(&mut controls.speed, 1.0..=120.0)
                        .logarithmic(true)
                        .suffix(" cyc/s"),
                );
            });
            ui.separator();

            // ── Status ──
            ui.label(egui::RichText::new("Status").strong());
            ui.horizontal(|ui| {
                ui.label(format!("Cycle: {}", snapshot.cycle));
                ui.label(format!("PC: 0x{:02X}", snapshot.pc));
            });
            ui.horizontal(|ui| {
                ui.label(format!("Z: {}", if snapshot.flag_z { "1" } else { "0" }));
                ui.label(format!("C: {}", if snapshot.flag_c { "1" } else { "0" }));
                if snapshot.halted {
                    ui.label(egui::RichText::new("HALTED").color(egui::Color32::RED));
                }
            });
            if !snapshot.asm.is_empty() {
                ui.label(format!("ASM: {}", snapshot.asm));
            }
            ui.label(format!("Active tiles: {}", snapshot.changed_indices.len()));
            ui.separator();

            // ── Visualization Modes ──
            ui.label(egui::RichText::new("Visualization").strong());

            // Wavefront
            ui.horizontal(|ui| {
                let wf_label = if wf.enabled {
                    "Wavefront: ON"
                } else {
                    "Wavefront: OFF"
                };
                if ui.button(wf_label).clicked() {
                    wf.enabled = !wf.enabled;
                }
                if wf.enabled {
                    ui.label(format!("[{}]", scope_label(wf.scope)));
                }
            });
            if wf.enabled {
                ui.horizontal(|ui| {
                    ui.label("Sweep:");
                    ui.add(egui::Slider::new(&mut wf.sweep_duration, 0.5..=10.0).suffix("s"));
                });
                ui.horizontal(|ui| {
                    let pct = (wf.progress * 100.0) as u32;
                    ui.label(format!("{}%", pct));
                    if ui.button("Restart").clicked() {
                        wf.progress = 0.0;
                        wf.playing = true;
                    }
                });
                ui.label(
                    egui::RichText::new("W=toggle  S=scope  R=restart")
                        .small()
                        .color(egui::Color32::GRAY),
                );
            }

            // Heatmap
            ui.horizontal(|ui| {
                let hm_label = if hm.enabled {
                    "Heatmap: ON"
                } else {
                    "Heatmap: OFF"
                };
                if ui.button(hm_label).clicked() {
                    hm.enabled = !hm.enabled;
                }
                if hm.enabled {
                    ui.label(format!("{} cyc", hm.total_cycles));
                }
            });
            if hm.enabled {
                ui.label(
                    egui::RichText::new("H=toggle  Ctrl+H=reset")
                        .small()
                        .color(egui::Color32::GRAY),
                );
            }

            // Scope overlay
            ui.horizontal(|ui| {
                let sc_label = if scope_overlay.enabled {
                    "Scope: ON"
                } else {
                    "Scope: OFF"
                };
                if ui.button(sc_label).clicked() {
                    scope_overlay.enabled = !scope_overlay.enabled;
                }
                if scope_overlay.enabled {
                    ui.label(format!("[{}]", scope_label(scope_overlay.scope)));
                }
            });
            if scope_overlay.enabled {
                ui.label(
                    egui::RichText::new("P=toggle  Shift+P=scope")
                        .small()
                        .color(egui::Color32::GRAY),
                );
            }
            ui.separator();

            // ── Registers ──
            ui.label(egui::RichText::new("Registers").strong());
            egui::Grid::new("reg_grid")
                .num_columns(3)
                .striped(true)
                .show(ui, |ui| {
                    for r in 0..16 {
                        let val = snapshot.regs[r];
                        ui.label(format!("R{:<2}", r));
                        ui.label(format!("0x{:016X}", val));
                        ui.label(format!("{}", val));
                        ui.end_row();
                    }
                });
            ui.separator();

            // ── RAM (first 16 cells) ──
            ui.label(egui::RichText::new("RAM (0..15)").strong());
            egui::Grid::new("ram_grid")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    for addr in 0..16 {
                        let val = snapshot.ram[addr];
                        ui.label(format!("[{:02X}]", addr));
                        ui.label(format!("0x{:016X}", val));
                        ui.end_row();
                    }
                });
            ui.separator();

            // ── Layer selector ──
            ui.label(egui::RichText::new("Layer").strong());
            ui.horizontal(|ui| {
                for z in 0..snapshot.num_layers.min(16) {
                    let label = format!("L{}", z);
                    if ui.selectable_label(layer.z == z, &label).clicked() {
                        layer.z = z;
                    }
                }
            });
            ui.separator();

            // ── Legend ──
            ui.label(egui::RichText::new("Legend").strong());
            for &(_cat, name, [r, g, b]) in tile_color::category_legend() {
                ui.horizontal(|ui| {
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                    ui.painter()
                        .rect_filled(rect, 2.0, egui::Color32::from_rgb(r, g, b));
                    ui.label(name);
                });
            }

            // ── Keybindings ──
            ui.separator();
            ui.label(egui::RichText::new("Keys").strong());
            ui.label(
                egui::RichText::new(
                    "Space=step  Enter=run  Esc=pause\n\
                     0-9=layer  W=wavefront  H=heatmap\n\
                     P=scope  S=cycle scope  R=restart\n\
                     Scroll=zoom  RMB/MMB=pan",
                )
                .small()
                .color(egui::Color32::GRAY),
            );
        });
}
