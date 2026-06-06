//! TileAnneal Phase 3: Interactive Segmentation Visualizer
//!
//! An interactive demo showing QUBO-based image segmentation with:
//! - User seed painting (left-click = foreground, right-click = background)
//! - Live annealing visualization (watch segmentation emerge)
//! - Real-time energy/metrics dashboard
//! - Edge-aware couplings option
//!
//! Run with: cargo run --release --features cuda,visualizer --example segmentation_visualizer
//!
//! Controls:
//!   Left-click: Paint foreground seeds (green)
//!   Right-click: Paint background seeds (red)
//!   Space: Run/pause annealing
//!   R: Reset spins (keep seeds)
//!   C: Clear seeds
//!   E: Toggle edge-aware couplings
//!   1/2/3: Load demo datasets (synthetic, real, ambiguous)

#[cfg(all(feature = "cuda", feature = "visualizer"))]
use egui_macroquad::egui;
#[cfg(all(feature = "cuda", feature = "visualizer"))]
use macroquad::prelude::*;

#[cfg(all(feature = "cuda", feature = "visualizer"))]
use engine::cuda::CudaRuntime;
#[cfg(all(feature = "cuda", feature = "visualizer"))]
use engine::cuda_tiles::{
    CouplingMode, GpuIsingGrid, IsingConfig, compute_weighted_energy, run_ising_update_weighted,
};

#[cfg(all(feature = "cuda", feature = "visualizer"))]
const GRID_SIZE: usize = 128; // 128x128 for interactive speed

#[cfg(all(feature = "cuda", feature = "visualizer"))]
const BRUSH_RADIUS: i32 = 3;

#[cfg(all(feature = "cuda", feature = "visualizer"))]
#[derive(Clone, Copy, PartialEq)]
enum SeedType {
    None,
    Foreground, // +1 spin
    Background, // -1 spin
}

#[cfg(all(feature = "cuda", feature = "visualizer"))]
struct SegmentationVisualizer {
    // CUDA resources
    rt: CudaRuntime,

    // Image data (grayscale 0-255)
    image: Vec<u8>,

    // Seeds from user painting
    seeds: Vec<SeedType>,

    // Ising model state
    grid: Option<GpuIsingGrid>,

    // Annealing state
    running: bool,
    beta: f32,
    beta_max: f32,
    sweeps_per_frame: u32,
    total_sweeps: u32,

    // Metrics (flight recorder)
    energy_history: Vec<i64>,
    current_energy: i64,
    best_energy: i64,

    // Parameters
    lambda: f32,        // Coupling strength (smoothness)
    seed_strength: f32, // How strong are seed biases
    edge_aware: bool,   // Use edge-aware couplings
    edge_alpha: f32,    // Edge sensitivity

    // Display
    image_texture: Option<Texture2D>,
    result_texture: Option<Texture2D>,

    // UI state
    show_help: bool,
}

#[cfg(all(feature = "cuda", feature = "visualizer"))]
impl SegmentationVisualizer {
    fn new() -> Self {
        let rt = CudaRuntime::new().expect("CUDA runtime init failed");

        // Start with synthetic demo image (circle on noisy background)
        let image = Self::generate_synthetic_circle();

        Self {
            rt,
            image,
            seeds: vec![SeedType::None; GRID_SIZE * GRID_SIZE],
            grid: None,
            running: false,
            beta: 0.1,
            beta_max: 20.0,
            sweeps_per_frame: 50,
            total_sweeps: 0,
            energy_history: Vec::new(),
            current_energy: 0,
            best_energy: i64::MAX,
            lambda: 2.0,
            seed_strength: 4.0,
            edge_aware: true,
            edge_alpha: 0.05,
            image_texture: None,
            result_texture: None,
            show_help: true,
        }
    }

    /// Generate synthetic test image: bright circle on dark noisy background
    fn generate_synthetic_circle() -> Vec<u8> {
        let mut img = vec![0u8; GRID_SIZE * GRID_SIZE];
        let cx = GRID_SIZE as f32 / 2.0;
        let cy = GRID_SIZE as f32 / 2.0;
        let radius = GRID_SIZE as f32 * 0.3;

        let mut rng = 42u64;

        for y in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();

                // Add noise
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let noise = ((rng % 40) as i32 - 20) as f32;

                let val = if dist < radius {
                    200.0 + noise // Bright circle
                } else {
                    50.0 + noise // Dark background
                };

                img[y * GRID_SIZE + x] = val.clamp(0.0, 255.0) as u8;
            }
        }
        img
    }

    /// Generate "real" test: gradient with soft edge (harder than circle)
    fn generate_gradient_blob() -> Vec<u8> {
        let mut img = vec![0u8; GRID_SIZE * GRID_SIZE];
        let cx = GRID_SIZE as f32 * 0.4;
        let cy = GRID_SIZE as f32 * 0.5;

        let mut rng = 123u64;

        for y in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();

                // Soft edge blob with gradient
                let blob_val = 255.0 * (-dist / 40.0).exp();

                // Add some texture
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let noise = ((rng % 30) as i32 - 15) as f32;

                // Background gradient
                let bg = (x as f32 / GRID_SIZE as f32) * 80.0 + 30.0;

                let val = blob_val.max(bg) + noise;
                img[y * GRID_SIZE + x] = val.clamp(0.0, 255.0) as u8;
            }
        }
        img
    }

    /// Generate ambiguous test: low contrast, needs seeds to resolve
    fn generate_ambiguous() -> Vec<u8> {
        let mut img = vec![0u8; GRID_SIZE * GRID_SIZE];
        let cx = GRID_SIZE as f32 / 2.0;
        let cy = GRID_SIZE as f32 / 2.0;

        let mut rng = 456u64;

        for y in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();

                // Very low contrast
                let base = if dist < GRID_SIZE as f32 * 0.35 {
                    140.0
                } else {
                    120.0
                };

                // Heavy noise
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let noise = ((rng % 60) as i32 - 30) as f32;

                img[y * GRID_SIZE + x] = (base + noise).clamp(0.0, 255.0) as u8;
            }
        }
        img
    }

    /// Build Ising model from image + seeds
    fn build_model(&mut self) {
        let size = GRID_SIZE * GRID_SIZE;

        // Compute bias from seeds (primary) and intensity (secondary)
        let mut h_bias = vec![0i8; size];
        for i in 0..size {
            h_bias[i] = match self.seeds[i] {
                SeedType::Foreground => self.seed_strength as i8, // Strong +
                SeedType::Background => -(self.seed_strength as i8), // Strong -
                SeedType::None => {
                    // Mild bias from intensity (maps 0-255 to roughly -1 to +1)
                    let intensity = self.image[i] as f32;
                    ((intensity - 128.0) / 128.0 * 1.0).round().clamp(-1.0, 1.0) as i8
                }
            };
        }

        // Compute couplings (ferromagnetic for smoothness)
        let (j_h, j_v) = if self.edge_aware {
            self.compute_edge_aware_couplings()
        } else {
            // Uniform ferromagnetic coupling
            let j = self.lambda.round().clamp(1.0, 4.0) as i8;
            (vec![j; size], vec![j; size])
        };

        // Create weighted grid with QUBO
        let coupling = CouplingMode::weighted(j_h, j_v);
        let config = IsingConfig::new(GRID_SIZE, GRID_SIZE).with_seed(42);
        let mut grid = GpuIsingGrid::new_weighted(&self.rt, &config, coupling).expect("grid");
        grid.set_bias(&self.rt, Some(&h_bias)).expect("set bias");
        grid.beta = self.beta;

        self.grid = Some(grid);
        self.total_sweeps = 0;
        self.energy_history.clear();
        self.best_energy = i64::MAX;

        // Compute initial energy
        if let Some(ref grid) = self.grid {
            self.current_energy = compute_weighted_energy(&self.rt, grid).expect("energy");
            self.best_energy = self.current_energy;
            self.energy_history.push(self.current_energy);
        }
    }

    /// Compute edge-aware couplings: J_ij = lambda * exp(-alpha * (I_i - I_j)^2)
    fn compute_edge_aware_couplings(&self) -> (Vec<i8>, Vec<i8>) {
        let size = GRID_SIZE * GRID_SIZE;
        let mut j_h = vec![0i8; size]; // Horizontal (to right neighbor)
        let mut j_v = vec![0i8; size]; // Vertical (to down neighbor)

        for y in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                let idx = y * GRID_SIZE + x;
                let i_center = self.image[idx] as f32;

                // Horizontal coupling (to right)
                if x + 1 < GRID_SIZE {
                    let i_right = self.image[idx + 1] as f32;
                    let diff = (i_center - i_right) / 255.0;
                    let coupling =
                        self.lambda * (-self.edge_alpha * diff * diff * 255.0 * 255.0).exp();
                    j_h[idx] = coupling.round().clamp(0.0, 4.0) as i8;
                }

                // Vertical coupling (to down)
                if y + 1 < GRID_SIZE {
                    let i_down = self.image[(y + 1) * GRID_SIZE + x] as f32;
                    let diff = (i_center - i_down) / 255.0;
                    let coupling =
                        self.lambda * (-self.edge_alpha * diff * diff * 255.0 * 255.0).exp();
                    j_v[idx] = coupling.round().clamp(0.0, 4.0) as i8;
                }
            }
        }

        (j_h, j_v)
    }

    /// Run one annealing step
    fn step_annealing(&mut self) {
        if let Some(ref mut grid) = self.grid {
            // Adaptive beta increase
            let beta_rate = 0.02; // 2% increase per frame
            self.beta = (self.beta * (1.0 + beta_rate)).min(self.beta_max);
            grid.beta = self.beta;

            run_ising_update_weighted(&self.rt, grid, self.sweeps_per_frame).expect("update");
            self.total_sweeps += self.sweeps_per_frame;

            let energy = compute_weighted_energy(&self.rt, grid).expect("energy");
            self.current_energy = energy;
            if energy < self.best_energy {
                self.best_energy = energy;
            }

            // Record every 10 frames to avoid huge history
            if self.energy_history.len() < 1000 {
                self.energy_history.push(energy);
            }
        }
    }

    /// Paint seeds at mouse position
    fn paint_seed(&mut self, mx: f32, my: f32, seed_type: SeedType, cell_size: f32) {
        let gx = (mx / cell_size) as i32;
        let gy = (my / cell_size) as i32;

        for dy in -BRUSH_RADIUS..=BRUSH_RADIUS {
            for dx in -BRUSH_RADIUS..=BRUSH_RADIUS {
                let x = gx + dx;
                let y = gy + dy;
                if x >= 0 && x < GRID_SIZE as i32 && y >= 0 && y < GRID_SIZE as i32 {
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq <= BRUSH_RADIUS * BRUSH_RADIUS {
                        self.seeds[(y as usize) * GRID_SIZE + (x as usize)] = seed_type;
                    }
                }
            }
        }
    }

    /// Get spins as image (RGBA bytes)
    fn get_spin_image(&self) -> Vec<u8> {
        let mut rgba = vec![0u8; GRID_SIZE * GRID_SIZE * 4];

        if let Some(ref grid) = self.grid {
            let spins = self.rt.download(&grid.spins).expect("download spins");
            let words_per_row = (GRID_SIZE + 63) / 64;

            for y in 0..GRID_SIZE {
                for x in 0..GRID_SIZE {
                    let word_idx = y * words_per_row + x / 64;
                    let bit_pos = x % 64;
                    let spin = if (spins[word_idx] >> bit_pos) & 1 == 1 {
                        1
                    } else {
                        -1
                    };

                    let idx = (y * GRID_SIZE + x) * 4;
                    if spin == 1 {
                        // Foreground: blue-ish
                        rgba[idx] = 100;
                        rgba[idx + 1] = 150;
                        rgba[idx + 2] = 255;
                        rgba[idx + 3] = 255;
                    } else {
                        // Background: dark gray
                        rgba[idx] = 40;
                        rgba[idx + 1] = 40;
                        rgba[idx + 2] = 40;
                        rgba[idx + 3] = 255;
                    }
                }
            }
        }
        rgba
    }

    /// Get image with seeds overlay (RGBA)
    fn get_image_with_seeds(&self) -> Vec<u8> {
        let mut rgba = vec![0u8; GRID_SIZE * GRID_SIZE * 4];

        for y in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                let idx = y * GRID_SIZE + x;
                let gray = self.image[idx];
                let pidx = idx * 4;

                match self.seeds[idx] {
                    SeedType::Foreground => {
                        // Green overlay
                        rgba[pidx] = (gray / 2) as u8;
                        rgba[pidx + 1] = 200;
                        rgba[pidx + 2] = (gray / 2) as u8;
                        rgba[pidx + 3] = 255;
                    }
                    SeedType::Background => {
                        // Red overlay
                        rgba[pidx] = 200;
                        rgba[pidx + 1] = (gray / 2) as u8;
                        rgba[pidx + 2] = (gray / 2) as u8;
                        rgba[pidx + 3] = 255;
                    }
                    SeedType::None => {
                        // Grayscale
                        rgba[pidx] = gray;
                        rgba[pidx + 1] = gray;
                        rgba[pidx + 2] = gray;
                        rgba[pidx + 3] = 255;
                    }
                }
            }
        }
        rgba
    }

    fn update(&mut self) {
        // Keyboard shortcuts
        if is_key_pressed(KeyCode::Space) {
            if self.grid.is_none() {
                self.build_model();
            }
            self.running = !self.running;
        }

        if is_key_pressed(KeyCode::R) {
            self.beta = 0.1;
            self.build_model();
        }

        if is_key_pressed(KeyCode::C) {
            self.seeds = vec![SeedType::None; GRID_SIZE * GRID_SIZE];
            self.grid = None;
            self.running = false;
        }

        if is_key_pressed(KeyCode::E) {
            self.edge_aware = !self.edge_aware;
            if self.grid.is_some() {
                self.beta = 0.1;
                self.build_model();
            }
        }

        if is_key_pressed(KeyCode::H) {
            self.show_help = !self.show_help;
        }

        // Load demo datasets
        if is_key_pressed(KeyCode::Key1) {
            self.image = Self::generate_synthetic_circle();
            self.seeds = vec![SeedType::None; GRID_SIZE * GRID_SIZE];
            self.grid = None;
            self.running = false;
        }
        if is_key_pressed(KeyCode::Key2) {
            self.image = Self::generate_gradient_blob();
            self.seeds = vec![SeedType::None; GRID_SIZE * GRID_SIZE];
            self.grid = None;
            self.running = false;
        }
        if is_key_pressed(KeyCode::Key3) {
            self.image = Self::generate_ambiguous();
            self.seeds = vec![SeedType::None; GRID_SIZE * GRID_SIZE];
            self.grid = None;
            self.running = false;
        }

        // Run annealing
        if self.running {
            self.step_annealing();
        }

        // Update textures
        let img_with_seeds = self.get_image_with_seeds();
        let img = Image {
            width: GRID_SIZE as u16,
            height: GRID_SIZE as u16,
            bytes: img_with_seeds,
        };
        if let Some(ref tex) = self.image_texture {
            tex.update(&img);
        } else {
            self.image_texture = Some(Texture2D::from_image(&img));
            if let Some(ref tex) = self.image_texture {
                tex.set_filter(FilterMode::Nearest);
            }
        }

        if self.grid.is_some() {
            let spin_img = self.get_spin_image();
            let img = Image {
                width: GRID_SIZE as u16,
                height: GRID_SIZE as u16,
                bytes: spin_img,
            };
            if let Some(ref tex) = self.result_texture {
                tex.update(&img);
            } else {
                self.result_texture = Some(Texture2D::from_image(&img));
                if let Some(ref tex) = self.result_texture {
                    tex.set_filter(FilterMode::Nearest);
                }
            }
        }
    }

    fn draw(&mut self) {
        let sw = screen_width();
        let sh = screen_height();

        // Calculate display sizes
        let panel_width = 280.0;
        let available_width = sw - panel_width;
        let cell_size: f32;
        let img_x: f32;
        let img_y: f32;
        let result_x: f32;

        // Split view: image on left, result on right
        let half_width = available_width / 2.0;
        cell_size = (half_width.min(sh) / GRID_SIZE as f32).floor();
        let grid_px = cell_size * GRID_SIZE as f32;

        img_x = (half_width - grid_px) / 2.0;
        img_y = (sh - grid_px) / 2.0;
        result_x = half_width + (half_width - grid_px) / 2.0;

        // Mouse painting (only on left panel - image area)
        let (mx, my) = mouse_position();
        if mx >= img_x && mx < img_x + grid_px && my >= img_y && my < img_y + grid_px {
            if is_mouse_button_down(MouseButton::Left) {
                self.paint_seed(mx - img_x, my - img_y, SeedType::Foreground, cell_size);
            }
            if is_mouse_button_down(MouseButton::Right) {
                self.paint_seed(mx - img_x, my - img_y, SeedType::Background, cell_size);
            }
        }

        // Draw image with seeds
        if let Some(ref tex) = self.image_texture {
            draw_texture_ex(
                tex,
                img_x,
                img_y,
                WHITE,
                DrawTextureParams {
                    dest_size: Some(vec2(grid_px, grid_px)),
                    ..Default::default()
                },
            );
        }

        // Draw border
        draw_rectangle_lines(img_x, img_y, grid_px, grid_px, 2.0, WHITE);
        draw_text("Image + Seeds", img_x, img_y - 10.0, 20.0, WHITE);

        // Draw segmentation result
        if let Some(ref tex) = self.result_texture {
            draw_texture_ex(
                tex,
                result_x,
                img_y,
                WHITE,
                DrawTextureParams {
                    dest_size: Some(vec2(grid_px, grid_px)),
                    ..Default::default()
                },
            );
            draw_rectangle_lines(result_x, img_y, grid_px, grid_px, 2.0, WHITE);
            draw_text("Segmentation", result_x, img_y - 10.0, 20.0, WHITE);
        } else {
            draw_rectangle_lines(result_x, img_y, grid_px, grid_px, 2.0, GRAY);
            draw_text(
                "(Press Space)",
                result_x + grid_px / 4.0,
                img_y + grid_px / 2.0,
                20.0,
                GRAY,
            );
        }

        // Draw energy graph at bottom
        if !self.energy_history.is_empty() {
            let graph_h = 80.0;
            let graph_w = available_width - 40.0;
            let graph_x = 20.0;
            let graph_y = sh - graph_h - 20.0;

            draw_rectangle(
                graph_x,
                graph_y,
                graph_w,
                graph_h,
                Color::new(0.1, 0.1, 0.1, 0.8),
            );
            draw_rectangle_lines(graph_x, graph_y, graph_w, graph_h, 1.0, WHITE);

            let min_e = self.energy_history.iter().min().copied().unwrap_or(0);
            let max_e = self.energy_history.iter().max().copied().unwrap_or(1);
            let range = (max_e - min_e).max(1) as f32;

            let n = self.energy_history.len();
            if n > 1 {
                for i in 1..n {
                    let x0 = graph_x + (i - 1) as f32 / (n - 1) as f32 * graph_w;
                    let x1 = graph_x + i as f32 / (n - 1) as f32 * graph_w;
                    let y0 = graph_y + graph_h
                        - (self.energy_history[i - 1] - min_e) as f32 / range * graph_h;
                    let y1 = graph_y + graph_h
                        - (self.energy_history[i] - min_e) as f32 / range * graph_h;
                    draw_line(x0, y0, x1, y1, 2.0, GREEN);
                }
            }

            draw_text(
                &format!(
                    "Energy: {} (best: {})",
                    self.current_energy, self.best_energy
                ),
                graph_x + 5.0,
                graph_y + 15.0,
                16.0,
                WHITE,
            );
        }

        // UI Panel
        egui_macroquad::ui(|ctx| {
            egui::Window::new("TileAnneal Segmentation")
                .fixed_pos([sw - panel_width + 10.0, 10.0])
                .fixed_size([panel_width - 20.0, sh - 20.0])
                .show(ctx, |ui| {
                    ui.heading("Controls");

                    ui.horizontal(|ui| {
                        if ui
                            .button(if self.running { "Pause" } else { "Run" })
                            .clicked()
                        {
                            if self.grid.is_none() {
                                self.build_model();
                            }
                            self.running = !self.running;
                        }
                        if ui.button("Reset").clicked() {
                            self.beta = 0.1;
                            self.build_model();
                        }
                        if ui.button("Clear").clicked() {
                            self.seeds = vec![SeedType::None; GRID_SIZE * GRID_SIZE];
                            self.grid = None;
                            self.running = false;
                        }
                    });

                    ui.separator();
                    ui.heading("Parameters");

                    ui.horizontal(|ui| {
                        ui.label("Lambda:");
                        ui.add(egui::Slider::new(&mut self.lambda, 0.5..=4.0));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Seed strength:");
                        ui.add(egui::Slider::new(&mut self.seed_strength, 1.0..=4.0));
                    });

                    ui.checkbox(&mut self.edge_aware, "Edge-aware couplings");

                    if self.edge_aware {
                        ui.horizontal(|ui| {
                            ui.label("Edge alpha:");
                            ui.add(egui::Slider::new(&mut self.edge_alpha, 0.01..=0.2));
                        });
                    }

                    ui.separator();
                    ui.heading("Annealing");

                    ui.horizontal(|ui| {
                        ui.label("Beta:");
                        ui.add(egui::Slider::new(&mut self.beta, 0.1..=50.0).logarithmic(true));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Beta max:");
                        ui.add(egui::Slider::new(&mut self.beta_max, 5.0..=100.0));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Sweeps/frame:");
                        ui.add(egui::Slider::new(&mut self.sweeps_per_frame, 10..=200));
                    });

                    ui.separator();
                    ui.heading("Metrics");

                    ui.label(format!("Total sweeps: {}", self.total_sweeps));
                    ui.label(format!("Current energy: {}", self.current_energy));
                    ui.label(format!("Best energy: {}", self.best_energy));
                    ui.label(format!("Temperature: {:.3}", 1.0 / self.beta));

                    let seed_count: usize =
                        self.seeds.iter().filter(|s| **s != SeedType::None).count();
                    ui.label(format!("Seeds: {} painted", seed_count));

                    ui.separator();
                    ui.heading("Demo Datasets");

                    ui.horizontal(|ui| {
                        if ui.button("1: Circle").clicked() {
                            self.image = Self::generate_synthetic_circle();
                            self.seeds = vec![SeedType::None; GRID_SIZE * GRID_SIZE];
                            self.grid = None;
                            self.running = false;
                        }
                        if ui.button("2: Blob").clicked() {
                            self.image = Self::generate_gradient_blob();
                            self.seeds = vec![SeedType::None; GRID_SIZE * GRID_SIZE];
                            self.grid = None;
                            self.running = false;
                        }
                        if ui.button("3: Hard").clicked() {
                            self.image = Self::generate_ambiguous();
                            self.seeds = vec![SeedType::None; GRID_SIZE * GRID_SIZE];
                            self.grid = None;
                            self.running = false;
                        }
                    });

                    if self.show_help {
                        ui.separator();
                        ui.heading("Help (H to hide)");
                        ui.label("Left-click: Foreground seeds");
                        ui.label("Right-click: Background seeds");
                        ui.label("Space: Run/pause");
                        ui.label("R: Reset spins");
                        ui.label("C: Clear all");
                        ui.label("E: Toggle edge-aware");
                        ui.label("");
                        ui.label("Paint seeds BEFORE running!");
                        ui.label("Seeds show optimizer > blur.");
                    }
                });
        });
    }
}

#[cfg(all(feature = "cuda", feature = "visualizer"))]
fn window_conf() -> Conf {
    Conf {
        window_title: "TileAnneal Segmentation Demo".to_string(),
        window_width: 1200,
        window_height: 700,
        ..Default::default()
    }
}

#[cfg(all(feature = "cuda", feature = "visualizer"))]
#[macroquad::main(window_conf)]
async fn main() {
    let mut vis = SegmentationVisualizer::new();

    loop {
        clear_background(Color::new(0.15, 0.15, 0.15, 1.0));
        vis.update();
        vis.draw();
        egui_macroquad::draw();
        next_frame().await;
    }
}

#[cfg(not(all(feature = "cuda", feature = "visualizer")))]
fn main() {
    eprintln!("This example requires both 'cuda' and 'visualizer' features.");
    eprintln!(
        "Run with: cargo run --release --features cuda,visualizer --example segmentation_visualizer"
    );
    std::process::exit(1);
}
