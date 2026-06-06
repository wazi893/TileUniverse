struct SimulationParams {
    width: u32,
    height: u32,
    frame_count: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> params: SimulationParams;
@group(0) @binding(1) var input_tex: texture_2d<u32>;
@group(0) @binding(2) var output_tex: texture_storage_2d<r32uint, write>;
@group(0) @binding(3) var meta_tex: texture_2d<u32>;
@group(0) @binding(4) var color_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var rom_tex: texture_2d<u32>;

// ============================================================================
// SPRINT 66: FITNESS-BASED SELECTION
// ============================================================================

// Popcount (count set bits) - fitness function for selection
fn popcount(v: u32) -> u32 {
    var x = v;
    x = x - ((x >> 1u) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2u) & 0x33333333u);
    x = (x + (x >> 4u)) & 0x0F0F0F0Fu;
    x = x + (x >> 8u);
    x = x + (x >> 16u);
    return x & 0x3Fu;
}

// Simple hash for mutation and tie-breaking
fn hash(x: u32, y: u32, frame: u32) -> u32 {
    var h = x * 374761393u + y * 668265263u + frame * 2147483647u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
}

// Mutate a value - higher rate to maintain diversity
fn mutate(val: u32, rand: u32) -> u32 {
    var result = val;
    // ~12% mutation rate per bit position checked
    // Flip 1-3 bits based on randomness
    if ((rand & 0xFFu) < 32u) {
        let bit1 = (rand >> 8u) & 0x1Fu;
        result = result ^ (1u << bit1);
    }
    if (((rand >> 16u) & 0xFFu) < 32u) {
        let bit2 = (rand >> 24u) & 0x1Fu;
        result = result ^ (1u << bit2);
    }
    return result;
}

fn get_color(type_id: u32, state: u32, xy: vec2<i32>) -> vec4<f32> {
    // 0=Dead, 1=Live
    // Logic colors based on Rust implementation
    // Wire=0, And=1, Or=2, Xor=3, Not=4, Clock=7
    
    let active_state = f32(state);
    let base = 0.2 + active_state * 0.8; // Brightness boost when active
    
    switch type_id {
        case 0u: { return vec4<f32>(0.3 * base, 0.3 * base, 0.35 * base, 1.0); } // Wire (Grey)
        case 1u: { return vec4<f32>(0.9 * base, 0.2 * base, 0.3 * base, 1.0); }  // And (Red)
        case 2u: { return vec4<f32>(0.2 * base, 0.9 * base, 0.3 * base, 1.0); }  // Or (Green)
        case 3u: { return vec4<f32>(0.3 * base, 0.5 * base, 0.95 * base, 1.0); } // Xor (Blue)
        case 4u: { return vec4<f32>(0.95 * base, 0.9 * base, 0.2 * base, 1.0); } // Not (Yellowish)
        case 7u: { return vec4<f32>(0.9 * base, 0.8 * base, 0.1 * base, 1.0); }  // Clock (Gold)
        case 30u: { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }                      // Const (White)
        case 40u: { // CPU Head - Use Sum of Neighbors for Density Visualization
            let val_n = textureLoad(input_tex, xy + vec2(0, -1), 0).r;
            let val_s = textureLoad(input_tex, xy + vec2(0, 1), 0).r;
            let val_e = textureLoad(input_tex, xy + vec2(1, 0), 0).r;
            let val_w = textureLoad(input_tex, xy + vec2(-1, 0), 0).r;
            let density = f32(val_n + val_s + val_e + val_w) / 1024.0; // Normalizing to 0-1 (approx)
            
            // Thermal/Fluid Gradient: Blue (Low) -> Cyan -> Green -> Yellow -> Red (High)
            if (density < 0.25) { return mix(vec4(0.1, 0.1, 0.3, 1.0), vec4(0.0, 1.0, 1.0, 1.0), density * 4.0); }
            else if (density < 0.5) { return mix(vec4(0.0, 1.0, 1.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), (density - 0.25) * 4.0); }
            else if (density < 0.75) { return mix(vec4(0.0, 1.0, 0.0, 1.0), vec4(1.0, 1.0, 0.0, 1.0), (density - 0.5) * 4.0); }
            else { return mix(vec4(1.0, 1.0, 0.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0), (density - 0.75) * 4.0); }
        }
        case 41u: { return vec4<f32>(0.0, 0.8, 0.8, 1.0); }                      // Register (Cyan)
        case 42u: { return vec4<f32>(1.0, 0.5, 0.0, 1.0); }                      // Console (Orange)
        case 48u: { // SELECTOR - Fitness-based selection (Sprint 66)
            // Color by fitness: low (purple) -> medium (blue) -> high (green) -> max (gold)
            let fitness = f32(popcount(state)) / 32.0; // Normalize to 0-1
            if (fitness < 0.25) {
                return mix(vec4(0.3, 0.0, 0.5, 1.0), vec4(0.0, 0.3, 0.8, 1.0), fitness * 4.0);
            } else if (fitness < 0.5) {
                return mix(vec4(0.0, 0.3, 0.8, 1.0), vec4(0.0, 0.8, 0.3, 1.0), (fitness - 0.25) * 4.0);
            } else if (fitness < 0.75) {
                return mix(vec4(0.0, 0.8, 0.3, 1.0), vec4(0.9, 0.9, 0.0, 1.0), (fitness - 0.5) * 4.0);
            } else {
                return mix(vec4(0.9, 0.9, 0.0, 1.0), vec4(1.0, 0.8, 0.0, 1.0), (fitness - 0.75) * 4.0);
            }
        }
        case 255u: { return vec4<f32>(0.0, 0.0, 0.0, 0.0); } // Eraser/Empty (Transparent)
        default: { return vec4<f32>(0.1, 0.1, 0.1, 1.0); } // Unknown (Dark Grey)
    }
}

fn decode_and_apply_write(instr: u32, my_reg_id: u32, cpu_pos: vec2<i32>, current_state: u32, next_state: ptr<function, u32>) -> bool {
    let op = instr & 0xFFu;
    let flags = (instr >> 8u) & 0xFFu;
    let imm = ((instr >> 16u) & 0xFFu) << 8u | ((instr >> 24u) & 0xFFu);
    
    let r_dest = (flags >> 4u) & 0xFu;
    let r_src = flags & 0xFu;
    let use_imm = (flags & 0x80u) != 0u;

    if (r_dest == my_reg_id) {
        if (op == 1u) { // MOV
            var src_val = 0u;
            if (use_imm) { src_val = imm; } else {
                var src_offset = vec2<i32>(0,0);
                if (r_src == 0u) { src_offset = vec2(0, -1); } 
                else if (r_src == 1u) { src_offset = vec2(0, 1); } 
                else if (r_src == 2u) { src_offset = vec2(1, 0); } 
                else if (r_src == 3u) { src_offset = vec2(-1, 0); } 
                src_val = textureLoad(input_tex, cpu_pos + src_offset, 0).r;
            }
            *next_state = src_val;
            return true;
        }
        if (op == 2u) { // ADD
            var src_val = 0u;
            if (use_imm) { src_val = imm; } else {
                var src_offset = vec2<i32>(0,0);
                if (r_src == 0u) { src_offset = vec2(0, -1); } 
                else if (r_src == 1u) { src_offset = vec2(0, 1); } 
                else if (r_src == 2u) { src_offset = vec2(1, 0); } 
                else if (r_src == 3u) { src_offset = vec2(-1, 0); } 
                src_val = textureLoad(input_tex, cpu_pos + src_offset, 0).r;
            }
            *next_state = current_state + src_val;
            return true;
        }
        if (op == 3u) { // SUB
            var src_val = 0u;
            if (use_imm) { src_val = imm; } else {
                var src_offset = vec2<i32>(0,0);
                if (r_src == 0u) { src_offset = vec2(0, -1); } 
                else if (r_src == 1u) { src_offset = vec2(0, 1); } 
                else if (r_src == 2u) { src_offset = vec2(1, 0); } 
                else if (r_src == 3u) { src_offset = vec2(-1, 0); } 
                src_val = textureLoad(input_tex, cpu_pos + src_offset, 0).r;
            }
            *next_state = current_state - src_val;
            return true;
        }
        if (op == 7u) { // SHR Dest, Imm/Src
            var src_val = 0u;
            if (use_imm) { src_val = imm; } else {
                var src_offset = vec2<i32>(0,0);
                if (r_src == 0u) { src_offset = vec2(0, -1); } 
                else if (r_src == 1u) { src_offset = vec2(0, 1); } 
                else if (r_src == 2u) { src_offset = vec2(1, 0); } 
                else if (r_src == 3u) { src_offset = vec2(-1, 0); } 
                src_val = textureLoad(input_tex, cpu_pos + src_offset, 0).r;
            }
            *next_state = current_state >> src_val;
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let xy = vec2<i32>(global_id.xy);
    let dims = textureDimensions(input_tex);
    
    if (xy.x >= i32(dims.x) || xy.y >= i32(dims.y)) {
        return;
    }

    // 1. Read Meta-Type
    let ops = textureLoad(meta_tex, xy, 0).r;
    
    // 2. Read Self State
    let current_state = textureLoad(input_tex, xy, 0).r;
    
    // 2b. Read Neighbors
    var in_n = false; var in_s = false; var in_e = false; var in_w = false;
    let x = xy.x; let y = xy.y;
    var val_n = 0u; var val_s = 0u; var val_e = 0u; var val_w = 0u;

    if (y > 0) { val_n = textureLoad(input_tex, xy + vec2(0, -1), 0).r; if (val_n > 0u) { in_n = true; } }
    if (y < i32(dims.y) - 1) { val_s = textureLoad(input_tex, xy + vec2(0, 1), 0).r; if (val_s > 0u) { in_s = true; } }
    if (x < i32(dims.x) - 1) { val_e = textureLoad(input_tex, xy + vec2(1, 0), 0).r; if (val_e > 0u) { in_e = true; } }
    if (x > 0) { val_w = textureLoad(input_tex, xy + vec2(-1, 0), 0).r; if (val_w > 0u) { in_w = true; } }
    
    var next_state = 0u;

    switch ops {
        case 0u: { if (in_n || in_s || in_e || in_w) { next_state = 1u; } }
        case 1u: { if ((in_w && in_e) || (in_n && in_s)) { next_state = 1u; } }
        case 2u: { if (in_n || in_s || in_e || in_w) { next_state = 1u; } }
        case 3u: { if ((in_w != in_e) || (in_n != in_s)) { next_state = 1u; } }
        case 4u: { if (!(in_n || in_s || in_e || in_w)) { next_state = 1u; } }
        case 7u: { if ((params.frame_count / 32u) % 2u == 0u) { next_state = 1u; } }
        case 40u: { // CPU HEAD
            let ip = current_state;
            let ip_idx = ip / 4u;
            let instr = textureLoad(rom_tex, vec2<i32>(i32(ip_idx % 256u), i32(ip_idx / 256u)), 0).r;
            let op = instr & 0xFFu;
            let imm = ((instr >> 16u) & 0xFFu) << 8u | ((instr >> 24u) & 0xFFu);
            
            if (op == 4u) { next_state = imm; } // JMP
            else if (op == 5u) { if (val_n == 0u) { next_state = imm; } else { next_state = ip + 4u; } } // JZ
            else if (op == 6u) { if (val_n != 0u) { next_state = imm; } else { next_state = ip + 4u; } } // JNZ
            else { next_state = ip + 4u; }
        }
        case 41u: { // REGISTER
            next_state = current_state;
            // Check neighbors for write
            if (textureLoad(meta_tex, xy + vec2(0, 1), 0).r == 40u) {
                 if (decode_and_apply_write(textureLoad(rom_tex, vec2<i32>(i32((textureLoad(input_tex, xy + vec2(0, 1), 0).r / 4u) % 256u), i32((textureLoad(input_tex, xy + vec2(0, 1), 0).r / 4u) / 256u)), 0).r, 0u, xy + vec2(0, 1), current_state, &next_state)) { textureStore(output_tex, xy, vec4<u32>(next_state, 0u, 0u, 0u)); let color = get_color(ops, next_state, xy); textureStore(color_tex, xy, color); return; }
            }
            if (textureLoad(meta_tex, xy + vec2(0, -1), 0).r == 40u) {
                 if (decode_and_apply_write(textureLoad(rom_tex, vec2<i32>(i32((textureLoad(input_tex, xy + vec2(0, -1), 0).r / 4u) % 256u), i32((textureLoad(input_tex, xy + vec2(0, -1), 0).r / 4u) / 256u)), 0).r, 1u, xy + vec2(0, -1), current_state, &next_state)) { textureStore(output_tex, xy, vec4<u32>(next_state, 0u, 0u, 0u)); let color = get_color(ops, next_state, xy); textureStore(color_tex, xy, color); return; }
            }
            if (textureLoad(meta_tex, xy + vec2(-1, 0), 0).r == 40u) {
                 if (decode_and_apply_write(textureLoad(rom_tex, vec2<i32>(i32((textureLoad(input_tex, xy + vec2(-1, 0), 0).r / 4u) % 256u), i32((textureLoad(input_tex, xy + vec2(-1, 0), 0).r / 4u) / 256u)), 0).r, 2u, xy + vec2(-1, 0), current_state, &next_state)) { textureStore(output_tex, xy, vec4<u32>(next_state, 0u, 0u, 0u)); let color = get_color(ops, next_state, xy); textureStore(color_tex, xy, color); return; }
            }
            if (textureLoad(meta_tex, xy + vec2(1, 0), 0).r == 40u) {
                 if (decode_and_apply_write(textureLoad(rom_tex, vec2<i32>(i32((textureLoad(input_tex, xy + vec2(1, 0), 0).r / 4u) % 256u), i32((textureLoad(input_tex, xy + vec2(1, 0), 0).r / 4u) / 256u)), 0).r, 3u, xy + vec2(1, 0), current_state, &next_state)) { textureStore(output_tex, xy, vec4<u32>(next_state, 0u, 0u, 0u)); let color = get_color(ops, next_state, xy); textureStore(color_tex, xy, color); return; }
            }
        }
        case 42u: { // CONSOLE
            next_state = current_state;
            var cpu_pos = vec2<i32>(0,0);
            if (textureLoad(meta_tex, xy + vec2(0, 1), 0).r == 40u) { cpu_pos = xy + vec2(0, 1); }
            else if (textureLoad(meta_tex, xy + vec2(0, -1), 0).r == 40u) { cpu_pos = xy + vec2(0, -1); }
            else if (textureLoad(meta_tex, xy + vec2(-1, 0), 0).r == 40u) { cpu_pos = xy + vec2(-1, 0); }
            else if (textureLoad(meta_tex, xy + vec2(1, 0), 0).r == 40u) { cpu_pos = xy + vec2(1, 0); }
            
            if (cpu_pos.x != 0 || cpu_pos.y != 0) {
                 let instr = textureLoad(rom_tex, vec2<i32>(i32((textureLoad(input_tex, cpu_pos, 0).r / 4u) % 256u), i32((textureLoad(input_tex, cpu_pos, 0).r / 4u) / 256u)), 0).r;
                 if ((instr & 0xFFu) == 255u) {
                     var src_offset = vec2<i32>(0,0);
                     let r_src = (instr >> 8u) & 0xFu;
                     if (r_src == 0u) { src_offset = vec2(0, -1); } 
                     else if (r_src == 1u) { src_offset = vec2(0, 1); } 
                     else if (r_src == 2u) { src_offset = vec2(1, 0); } 
                     else if (r_src == 3u) { src_offset = vec2(-1, 0); } 
                     next_state = textureLoad(input_tex, cpu_pos + src_offset, 0).r;
                 }
            }
        }
        case 30u: { next_state = 1u; }
        case 48u: { // SELECTOR - Fitness-based selection (Sprint 66)
            // Compute my fitness
            let my_fitness = popcount(current_state);

            // Get neighbor values and fitness (only check SELECTOR neighbors)
            var best_val = current_state;
            var best_fitness = my_fitness;
            var best_pos = xy;

            // Check North
            if (y > 0 && textureLoad(meta_tex, xy + vec2(0, -1), 0).r == 48u) {
                let n_fitness = popcount(val_n);
                if (n_fitness > best_fitness) {
                    best_fitness = n_fitness;
                    best_val = val_n;
                    best_pos = xy + vec2(0, -1);
                }
            }
            // Check South
            if (y < i32(dims.y) - 1 && textureLoad(meta_tex, xy + vec2(0, 1), 0).r == 48u) {
                let s_fitness = popcount(val_s);
                if (s_fitness > best_fitness) {
                    best_fitness = s_fitness;
                    best_val = val_s;
                    best_pos = xy + vec2(0, 1);
                }
            }
            // Check East
            if (x < i32(dims.x) - 1 && textureLoad(meta_tex, xy + vec2(1, 0), 0).r == 48u) {
                let e_fitness = popcount(val_e);
                if (e_fitness > best_fitness) {
                    best_fitness = e_fitness;
                    best_val = val_e;
                    best_pos = xy + vec2(1, 0);
                }
            }
            // Check West
            if (x > 0 && textureLoad(meta_tex, xy + vec2(-1, 0), 0).r == 48u) {
                let w_fitness = popcount(val_w);
                if (w_fitness > best_fitness) {
                    best_fitness = w_fitness;
                    best_val = val_w;
                    best_pos = xy + vec2(-1, 0);
                }
            }

            // Probabilistic selection with mutation and catastrophic reset
            let rand = hash(u32(xy.x), u32(xy.y), params.frame_count);
            let rand2 = hash(u32(xy.y), u32(xy.x), params.frame_count + 12345u);

            // ~0.5% chance of catastrophic reset (death) - creates ongoing dynamics
            if ((rand2 & 0xFFu) < 2u) {
                // Reset to random low-fitness value
                next_state = rand & 0x0000FFFFu; // Only lower 16 bits = ~16 fitness
            } else {
                // 60% chance to adopt better neighbor, 40% keeps current (more drift)
                let dominated = (best_fitness > my_fitness) && ((rand & 0xFFu) < 153u);

                if (dominated) {
                    // Take winner's value with mutation
                    next_state = mutate(best_val, rand);
                } else {
                    // Keep my value with mutation
                    next_state = mutate(current_state, rand >> 8u);
                }
            }
        }
        default: { next_state = 0u; }
    }
    
    // Write State
    textureStore(output_tex, xy, vec4<u32>(next_state, 0u, 0u, 0u));

    // Write Color
    let color = get_color(ops, next_state, xy);
    textureStore(color_tex, xy, color);
}
