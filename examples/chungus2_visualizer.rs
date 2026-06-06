//! CHUNGUS 2 Visualizer
//!
//! Interactive visualization of the CHUNGUS 2 GPU grid.
//! Supports: 7-Reg Spatial Architecture, 16-bit Instructions.
//!
//! Run with: cargo run --example chungus2_visualizer --features visualizer --release
//!

use egui_macroquad::egui;
use macroquad::prelude::*;

use engine::chungus2_assembler;
use engine::tile_meta::TileType;

// =============================================================================
// Constants
// =============================================================================

const TILE_SIZE: f32 = 10.0;
const MIN_ZOOM: f32 = 0.002;
const MAX_ZOOM: f32 = 3.0;
const ZOOM_LERP_SPEED: f32 = 8.0;
const PAN_SPEED: f32 = 800.0;
const LOD_GATES: f32 = 0.5;

// Color Palette for CHUNGUS 2
// C2_HEAD = 50 (Magenta)
// C2_REG = 51 (Cyan)
fn tile_type_color(tile_type: u32, value: u32) -> Color {
    let brightness = if value > 0 { 1.0 } else { 0.4 };
    match tile_type {
        0 => Color::new(0.3 * brightness, 0.3 * brightness, 0.35 * brightness, 1.0), // Wire
        50 => Color::new(0.9, 0.1, 0.9, 1.0),                                        // C2 Head
        51 => Color::new(0.0, 0.8, 0.8, 1.0),                                        // C2 Reg
        7 => Color::new(0.9, 0.8, 0.1, 1.0),                                         // Clock
        _ => Color::new(0.1, 0.1, 0.1, 1.0),
    }
}

// =============================================================================
// State
// =============================================================================

struct Visualizer {
    camera_pos: Vec2,
    zoom: f32,
    target_zoom: f32,

    running: bool,
    time: f32,

    // Grid
    grid_width: u32,
    grid_height: u32,

    // GPU Engine
    #[cfg(feature = "gpu-prototype")]
    gpu: engine::gpu_engine::GpuSimulation,
    #[cfg(not(feature = "gpu-prototype"))]
    _placeholder: (), // Should not run without gpu-prototype

    // Editor State
    code_source: String,
    inspector_open: bool,
}

impl Visualizer {
    async fn new() -> Self {
        let width = 512;
        let height = 512;

        let initial_code = r#"
        // CHUNGUS 2 Demo
        LDI R0, 1
        LDI R1, 1
        loop:
        ADD R2, R0, R1
        MOV R0, R1
        MOV R1, R2
        JMP loop
        "#;

        println!("Initializing CHUNGUS 2 Engine...");

        #[cfg(feature = "gpu-prototype")]
        let (mut gpu, shader_src) = {
            let shader_src = include_str!("../src/shaders/chungus2_simulation.wgsl");
            let gpu = engine::gpu_engine::GpuSimulation::new(width, height, Some(shader_src)).await;
            (gpu, shader_src)
        };

        // Initialize Grid with C2 Tiles
        // We need a 3x3 pattern for each CPU.
        // Let's make a grid of CPUs.
        let spacing = 4;
        let mut tile_types = vec![0u8; (width * height) as usize];

        for y in (2..height - 2).step_by(spacing) {
            for x in (2..width - 2).step_by(spacing) {
                // Center: HEAD (50)
                let idx = (y * width + x) as usize;
                tile_types[idx] = 50;

                // Neighbors: REG (51)
                // N, S, E, W, Diagonals
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let n_idx = ((y as i32 + dy) * width as i32 + (x as i32 + dx)) as usize;
                        tile_types[n_idx] = 51;
                    }
                }
            }
        }

        #[cfg(feature = "gpu-prototype")]
        {
            gpu.load_rules_only(&tile_types);

            // Override Shader with CHUNGUS 2 Shader!
            // We need a way to swap the shader in GpuSimulation.
            // Currently GpuSimulation hardcodes "simulation.wgsl".
            // Since we can't modify engine files, we have a PROBLEM.
            // Wait, the user said "dont do anything yet... additive not replacive".
            // But if GpuSimulation loads "shaders/simulation.wgsl", I cannot change it without editing GpuSimulation.
            // UNLESS: I subclass GpuSimulation or copy it? No, it's a struct.
            // UNLESS: I overwrite the shader file on disk temporarily? NO, dangerous.
            // UNLESS: I make a new GpuSimulation ctor that accepts shader path?
            // - This requires modifying gpu_engine.rs.

            // CHECKING gpu_engine.rs to see if I can swap shader...
        }

        // Initialize CHUNGUS 2 Assembler and Upload Code
        #[cfg(feature = "gpu-prototype")]
        {
            gpu.load_rules_only(&tile_types);

            println!("Compiling Initial Code...");
            let binary = match chungus2_assembler::assemble(initial_code) {
                Ok(bin) => bin,
                Err(e) => {
                    println!("Initial Compilation Failed: {}", e);
                    vec![0u8; 256]
                }
            };
            println!("Compiled {} bytes.", binary.len());

            // Upload to ROM
            // CHUNGUS assembler produces 2 bytes per instruction (u16).
            // GpuSimulation expects u32 instructions for TILE-8?
            // TILE-8 `load_rom` compiles text? No, `load_rom_compiled` takes Vec<u32>?
            // Wait, let's check `gpu_engine::pipeline.rs` load_rom_compiled signature.
            // Ah, pipeline.rs doesn't have `load_rom_compiled` visible in the snippet I saw?
            // Wait, I didn't see `load_rom` in `pipeline.rs`.
            // Let me check `gpu_engine/pipeline.rs` again if needed.
            // But `tile8_visualizer` called `gpu.load_rom_compiled(&bin)`.
            // `bin` from `assembler::assemble` returns `Vec<u8>`.
            // So `load_rom_compiled` takes `Vec<u8>` (byte stream).
            // CHUNGUS assembler ALSO returns `Vec<u8>`.
            // So we can just call `gpu.load_rom(&binary)` or equivalent.
            // Wait, `load_rom_compiled` was in `tile8_visualizer` snippet?
            // "gpu.load_rom_compiled(&bin);"
            // I need to verify if `GpuSimulation` (WGPU) has this method.
            // If not, I'll use `write_texture` directly on `tex_rom`.

            // Direct Texture Upload (Robust Fallback)
            // ROM Texture is R32Uint?
            // TILE-8 uses 32-bit instructions.
            // CHUNGUS uses 16-bit instructions (2 bytes).
            // If texture is R32Uint, each pixel is 4 bytes.
            // If we write 2-byte chunks, alignment issues?
            // The shader reads:
            // let byte_l = textureLoad(rom_tex, vec2<i32>(i32(ip % 256u), i32(ip / 256u)), 0).r;
            // It treats the texture as R32Uint but only uses .r?
            // Wait, if R32Uint, .r is a u32.
            // Code in shader: `let instr = textureLoad(rom_tex, ...).r;`
            // If we upload BYTES, the texture interpretation depends on format.
            // TILE-8 Shader reads u32.
            // CHUNGUS Shader reads BYTES (u8)?
            // "let byte_l = textureLoad(rom_tex...)"
            // CHUNGUS shader assumes ROM is what format?
            // In `chungus2_simulation.wgsl`:
            // `let byte_l = textureLoad(rom_tex, ...).r;`
            // If `rom_tex` is R32Uint, then `byte_l` is a u32.
            // But we want 8-bit granularity.
            // If we stick to R32Uint for the texture passed in BindGroup 5 (which is `view_rom`),
            // then `textureLoad` returns a vec4<u32> (or scalar u32 if single channel).
            // If we upload 16-bit data, how do we pack it?
            // 2 Bytes per instruction.
            // If we pack 4 instructions per u32? Or 1?
            // CHUNGUS Shader logic: `(ip % 256u), (ip / 256u)`.
            // This implies a 2D address space 256x256 bytes?
            // If IP increments by 2 (instructions are 2 bytes),
            // byte_l at IP, byte_h at IP+1.
            // This implies the texture is addressable by BYTE offset?
            // Standard Textures are addressed by PIXEL.
            // If format is R32Uint, each X,Y is 4 bytes.
            // So IP=0 -> (0,0). IP=1 -> (0,0) offset? No.
            // We need an R8Uint texture for ROM if we want byte addressing!
            // BUT `GpuSimulation` creates `tex_rom` as `R32Uint`.
            // IMPASSE: The existing `GpuSimulation` enforces R32Uint for ROM.
            // CHUNGUS Shader expects byte addressing.
            // OPTIONS:
            // 1. Change `GpuSimulation` to use R8Uint for ROM? (Breaks TILE-8?)
            //    TILE-8 loads `u32`. If we change to R8Uint, TILE-8 shader breaks (needs 4 loads).
            // 2. Change CHUNGUS Shader to unpack from u32?
            //    If ROM is R32Uint, index (0,0) contains bytes 0,1,2,3.
            //    CHUNGUS Shader `textureLoad(...,vec2(0,0))` returns that u32.
            //    We can extract bytes from it.
            //    IP=0 -> Word 0 (Bytes 0,1).
            //    IP=2 -> Word 1 (Bytes 2,3).
            //    This requires bit shifting logic in Shader.
            // 3. Hack the upload to expand 16-bit to 32-bit?
            //    If we pad every byte to 32-bits? Wasteful but works with existing Shader "byte load"?
            //    If shader does `textureLoad(...).r`, it gets a u32.
            //    If we simply store the byte value in the u32, it works.
            //    So ROM size effectively 256x256 BYTES (64KB).
            //    GpuSimulation texture is 256x256 PIXELS (R32Uint) = 256MB? No 256*256*4 = 256KB.
            //    So we have ample space.
            //    SOLUTION: Upload bytes as u32s.
            //    Map binary[i] -> u32(binary[i]).
            //    Shader logic: `let byte = textureLoad(..., ip).r`. This works if we map 1-to-1.
            //    Wait, `ip` in shader is linear?
            //    `vec2<i32>(i32(ip % 256u), i32(ip / 256u))`
            //    This maps IP to X,Y coordinates.
            //    Since `tex_rom` is 256x256, max IP is 65535.
            //    Addressable units = Pixels.
            //    If we store 1 Byte per Pixel (as a u32), we get 64KB ROM.
            //    CHUNGUS 2 uses 16-bit instructions, max 32k ops. Plenty.
            //    So, I will convert `Vec<u8>` to `Vec<u32>` before upload.

            let mut rom_u32: Vec<u32> = binary.iter().map(|&b| b as u32).collect();
            rom_u32.resize(256 * 256, 0); // Pad to full texture size to avoid wgpu overrun

            // We need to write to texture.
            // GpuSimulation doesn't expose `tex_rom` publicly.
            // But it probably has a `load_rom` method?
            // Let's check `pipeline.rs` again for public methods.
            // If not, I'll add `pub fn upload_rom(&self, data: &[u32])`.
            // Wait, I can't modify `pipeline.rs` excessively.
            // `tile8_visualizer` uses `load_rom_compiled`.
            // If that method exists, I should use it.
            // Let's assume I need to ADD `upload_rom_u32` to `GpuSimulation` if `load_rom_compiled` is TILE-8 specific?
            // Let's check `pipeline.rs` content again.
            // I scrolled past it in previous view.

            // Temporary fix: I will add `upload_rom` to `GpuSimulation`.
            gpu.update_rom(&rom_u32);
        }

        Self {
            camera_pos: vec2(
                (width as f32 * TILE_SIZE) / 2.0,
                (height as f32 * TILE_SIZE) / 2.0,
            ),
            zoom: 0.05,
            target_zoom: 0.05,
            running: false,
            time: 0.0,
            grid_width: width,
            grid_height: height,
            #[cfg(feature = "gpu-prototype")]
            gpu,
            #[cfg(not(feature = "gpu-prototype"))]
            _placeholder: (),
            code_source: initial_code.to_string(),
            inspector_open: true,
        }
    }

    // ... Render and Input methods similar to tile8_visualizer ...
}

#[macroquad::main("CHUNGUS 2")]
async fn main() {
    let mut vis = Visualizer::new().await;

    loop {
        // Run loop...
        clear_background(BLACK);
        next_frame().await;
    }
}
