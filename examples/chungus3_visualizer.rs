//! CHUNGUS 3 Visualizer (ADDM Prototype)
//!
//! Interactive visualization of the Async Distributed Dataflow Mesh.
//! Verify token passing and FU execution.
//!
//! Run with: cargo run --example chungus3_visualizer --features "visualizer gpu-prototype"

use egui_macroquad::egui;
use macroquad::prelude::*;

use engine::chungus3_compiler::{Compiler, Instruction};

// Token Packing Constants
const META_VALID_MASK: u32 = 0x80;

fn pack_token(val: u32, meta: u32, tag: u32) -> u32 {
    (tag << 16) | (meta << 8) | (val & 0xFF)
}

struct Visualizer {
    camera_pos: Vec2,
    zoom: f32,

    // Grid
    grid_width: u32,
    grid_height: u32,

    // GPU Engine
    #[cfg(feature = "gpu-prototype")]
    gpu: engine::gpu_engine::GpuSimulation,
    #[cfg(not(feature = "gpu-prototype"))]
    _placeholder: (),

    display_texture: Option<Texture2D>,

    // Interaction
    current_tool: u32, // 0=Wire, 1=ADD, 2=Token
    input_val: u32,
}

impl Visualizer {
    async fn new() -> Self {
        // Grid Size: 256 for better alignment and more space
        let width = 256;
        let height = 256;

        #[cfg(feature = "gpu-prototype")]
        let (mut gpu, _shader_src) = {
            let shader_src = include_str!("../src/shaders/chungus3_simulation.wgsl");
            let gpu = engine::gpu_engine::GpuSimulation::new(width, height, Some(shader_src)).await;
            (gpu, shader_src)
        };

        // --- Compiler: Physics Demo (Falling Block) ---
        let mut compiler = Compiler::new();

        // ARITHMETIC DEMO
        // Place ADD Unit at Center
        // Inputs will be spawned at NW and NE.
        let center_x = 128;
        compiler.add_instr(Instruction::AddUnit(center_x as i32, 64)); // (128, 64)

        // Add a "Floor" to catch the result
        for c in -10..10 {
            compiler.add_instr(Instruction::Pixel(
                center_x as i32 + c,
                100,
                (3u32 << 16) | (0x80 << 8) | 255,
            ));
        }
        // Compile (using relative offsets, so base can be 0,0)
        let (tiles, states) = compiler.compile(0, 0);

        #[cfg(feature = "gpu-prototype")]
        {
            // Clear Grid (Default to C3_TOKEN / Air for Gravity)
            let mut tile_array = vec![62u8; (width * height) as usize];
            let mut state_array = vec![0u32; (width * height) as usize];

            // Apply Compiler Output
            for ((x, y), tile_type) in tiles {
                let idx = (y * width as u32 + x) as usize;
                if idx < tile_array.len() {
                    tile_array[idx] = tile_type;
                }
            }
            for ((x, y), state) in &states {
                let idx = (y * width as u32 + x) as usize;
                if idx < state_array.len() {
                    state_array[idx] = *state;
                }
            }

            gpu.upload_meta(&tile_array);

            // Upload Initial States (Sparse)
            for ((x, y), state) in &states {
                gpu.write_state(*x, *y, *state);
            }
        }

        Self {
            // Focus on logic at (20, 20).
            // Camera Target is in World Units (Pixels).
            // Grid Step = 10.0. Pos = 20 * 10 = 200.
            // Center camera near (250, 250).
            camera_pos: vec2(250.0, 250.0),
            zoom: 0.05, // A bit higher to see details? No, zoom works inversely in my logic?
            // "zoom: vec2(self.zoom * 2.0 / screen_width(), ...)"
            // If scale is 1.0, we see 2.0 / 1.0 = 2 units?
            // If self.zoom = 0.05 -> vec2(0.1 / 800) -> Tiny?
            // Wait, macroquad default Camera2D zoom is 1.0 = 1 unit per screen height?
            // Let's stick closer to what usually works. 0.05 might be fine if tiles are 10px.
            // But user said "scaled too high".
            // Let's try 0.1? Or 0.02?
            // If user sees "just the starting grid mesh", maybe they are zoomed IN too much?
            // Let's try 0.02 (Zoomed Out more, seeing more area).
            // Width 128x10 = 1280 pixels world.
            // If screen 800.
            // We want to see maybe 500 pixels.
            grid_width: width,
            grid_height: height,
            #[cfg(feature = "gpu-prototype")]
            gpu,
            #[cfg(not(feature = "gpu-prototype"))]
            _placeholder: (),

            display_texture: None,
            current_tool: 2, // Default Token
            input_val: 1,
        }
    }

    async fn update(&mut self) {
        // Pan/Zoom
        // Pan/Zoom (Speed up)
        if is_key_down(KeyCode::W) {
            self.camera_pos.y -= 20.0 / self.zoom;
        }
        if is_key_down(KeyCode::S) {
            self.camera_pos.y += 20.0 / self.zoom;
        }
        if is_key_down(KeyCode::A) {
            self.camera_pos.x -= 20.0 / self.zoom;
        }
        if is_key_down(KeyCode::D) {
            self.camera_pos.x += 20.0 / self.zoom;
        }

        let (_, wheel_y) = mouse_wheel();
        if wheel_y != 0.0 {
            self.zoom *= if wheel_y > 0.0 { 1.1 } else { 0.9 };
        }

        #[cfg(feature = "gpu-prototype")]
        {
            // Input Injection (Space -> Spawn Stream)
            // Throttle: Max 1 token every 5 frames to prevent Apex Jamming.
            // (Apex Pin split rate is < 0.5/tick, so Input > 0.5/tick jams).
            if is_key_pressed(KeyCode::Space) {
                // Spawn Operands for ADD Unit at (128, 64)
                // Input A: 2 (West input -> 127, 63)
                // Input B: 3 (East input -> 129, 63)

                let val_a = (0 << 16) | (0x80 << 8) | 2;
                let val_b = (1 << 16) | (0x80 << 8) | 3; // Tag 1 for visual difference

                self.gpu.write_state(127, 63, val_a);
                self.gpu.write_state(129, 63, val_b);

                println!("Injected 2 and 3 into ADD Unit");
            }
            // Mouse Interaction
            if is_mouse_button_down(MouseButton::Left) || is_mouse_button_down(MouseButton::Right) {
                let mpos = mouse_position();
                let world_x = (mpos.0 - screen_width() / 2.0) / self.zoom + self.camera_pos.x;
                let world_y = (mpos.1 - screen_height() / 2.0) / self.zoom + self.camera_pos.y;
                let tx = (world_x / 10.0) as u32;
                let ty = (world_y / 10.0) as u32;

                if tx < self.grid_width && ty < self.grid_height {
                    // Left Click: Place Tool
                    if is_mouse_button_down(MouseButton::Left) {
                        let type_to_place = if self.current_tool == 0 {
                            60
                        }
                        // ADD
                        else if self.current_tool == 2 {
                            62
                        }
                        // TOKEN
                        else {
                            0
                        }; // Wire?

                        self.gpu.update_tile(tx, ty, type_to_place as u8);

                        // If Token tool, also inject value?
                        if self.current_tool == 2 {
                            let token = pack_token(self.input_val, META_VALID_MASK, 1);
                            self.gpu.update_state(tx, ty, token);
                        }
                    }
                }
            }

            self.gpu.step();

            // Fibonacci Verification.
            // Nodes were placed at X = 20 + ID*6.
            // 0 (SrcA): 20
            // 1 (SrcB): 26
            // 2 (MergeA): 32
            // 3 (MergeB): 38
            // 4 (ADD): 44. Output at 45.

            let roi = self
                .gpu
                .capture_frame_roi(0, 0, self.grid_width, self.grid_height)
                .await;
            if !roi.is_empty() {
                let img = Image {
                    width: self.grid_width as u16,
                    height: self.grid_height as u16,
                    bytes: roi,
                };
                if let Some(tex) = &self.display_texture {
                    tex.update(&img);
                } else {
                    self.display_texture = Some(Texture2D::from_image(&img));
                    self.display_texture
                        .as_ref()
                        .unwrap()
                        .set_filter(FilterMode::Nearest);
                }
            }

            // Verification Log (kept)
            let roi_check = self.gpu.capture_state_roi(45, 20, 1, 1).await;
            if !roi_check.is_empty() {
                let val = roi_check[0] & 0xFF;
                let valid = (roi_check[0] >> 8) & 0x80 != 0;
                if valid {
                    println!("FIBO OUT: {}", val);
                }
            }
        }
    }

    fn draw(&mut self) {
        set_camera(&Camera2D {
            target: self.camera_pos,
            zoom: vec2(
                self.zoom * 2.0 / screen_width(),
                self.zoom * 2.0 / screen_height(),
            ),
            ..Default::default()
        });

        // Draw Texture
        if let Some(tex) = &self.display_texture {
            draw_texture_ex(
                tex,
                0.,
                0.,
                WHITE,
                DrawTextureParams {
                    dest_size: Some(vec2(
                        self.grid_width as f32 * 10.0,
                        self.grid_height as f32 * 10.0,
                    )),
                    ..Default::default()
                },
            );
        }

        draw_grid(self.grid_width as usize, 10.0, 0.5, GRAY, DARKGRAY);

        // UI
        set_default_camera();
        egui_macroquad::ui(|egui_ctx| {
            egui::Window::new("ADDM Control").show(egui_ctx, |ui| {
                ui.label("CHUNGUS 3: ADDM Prototype");
                ui.label("Tools:");
                ui.radio_value(&mut self.current_tool, 2, "Token (Wire)");
                ui.radio_value(&mut self.current_tool, 0, "ADD FU");

                ui.add(egui::Slider::new(&mut self.input_val, 0..=7).text("Input Value"));
            });
        });
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "CHUNGUS 3".to_string(),
        window_width: 800,
        window_height: 600,
        platform: miniquad::conf::Platform {
            ..Default::default()
        },
        // Macroquad doesn't expose strict "no audio" easily in Conf 0.3?
        // Wait, miniquad backend handles it.
        // Actually, the panic is in `quad-snd`.
        // Macroquad uses `quad-snd` by default.
        // There is no config option to fully disable audio init in Conf struct itself easily exposed?
        // Let's check Conf docs or assume `Conf::default()`?
        // The error `GetDefaultAudioEndPoint failed` suggests it tries accessing WASAPI.
        // Maybe we cannot disable it via Conf easily without a custom build feature or miniquad config.
        // BUT, usually `high_dpi`, `fullscreen` etc are here.
        // Let's try to just define the function first.
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // Actually, `quad-snd` panic might be unavoidable if the library unconditionally inits it.
    // However, on some versions, setting "blocking_event_loop" helps?
    // Let's at least define the config correctly.
    let mut vis = Visualizer::new().await;
    loop {
        clear_background(BLACK);
        vis.update().await;
        vis.draw();
        egui_macroquad::draw();
        next_frame().await;
    }
}
// Helpers
fn draw_grid(w: usize, step: f32, thick: f32, c: Color, bg: Color) {
    for i in 0..=w {
        let p = i as f32 * step;
        draw_line(p, 0.0, p, w as f32 * step, thick, c);
        draw_line(0.0, p, w as f32 * step, p, thick, c);
    }
}
