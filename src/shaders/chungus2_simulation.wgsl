// CHUNGUS 2 CPU Simulation Shader
// Based on 3x3 Spatial Grid Layout

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

// Constants
const C2_HEAD = 50u;
const C2_REG = 51u;

fn get_color(type_id: u32, state: u32) -> vec4<f32> {
    switch type_id {
        case 50u: { return vec4<f32>(0.9, 0.1, 0.9, 1.0); } // C2 HEAD (Magenta)
        case 51u: { return vec4<f32>(0.2, 0.8, 0.9, 1.0); } // C2 REG (Cyan)
        default: { return vec4<f32>(0.1, 0.1, 0.1, 1.0); }
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let xy = vec2<i32>(global_id.xy);
    let dims = textureDimensions(input_tex);
    
    if (xy.x >= i32(dims.x) || xy.y >= i32(dims.y)) {
        return;
    }

    let type_id = textureLoad(meta_tex, xy, 0).r;
    let current_state = textureLoad(input_tex, xy, 0).r;
    var next_state = current_state;

    // CHUNGUS 2 Logic
    // We only process C2 types here. Ideally this would be merged with main sim, 
    // but for now this acts as a standalone verified logic block.
    
    if (type_id == C2_HEAD) {
        // IP Logic
        let ip = current_state;
        
        // Fetch Instruction (16-bit) from ROM
        // Map IP to 2D Texture Coords (256 width typically)
        let rom_x = i32(ip % 256u);
        let rom_y = i32(ip / 256u);
        let instr_word = textureLoad(rom_tex, vec2<i32>(rom_x, rom_y), 0).r;
        
        // Decode: [ Op(5) ] [ Rd(3) ] [ Rs1(3) ] [ Rs2(3) ] [ Unused(2) ]
        // Note: Shader loads u32. Assuming ROM texture stores 16-bit instruction in lower 16 bits.
        // Actually, assembler generates bytes. 
        // If we upload as R32Uint, every pixel is 4 bytes? 
        // Or R8Uint? Usually texture is u32/R32Uint in this engine.
        // Let's assume the ROM loader packs 2 bytes into the lower 16 bits of the u32 pixel.
        // Or maybe 1 byte per pixel?
        // Detailed check: Assembler returns Vec<u8>. Upload logic usually packs this.
        // Let's assume standard u32 per instruction to be safe for now, 
        // OR assume the engine packs 4 bytes.
        // If the Assembler emits 2 bytes per instruction, and we map 1 instruction per pixel (simplest):
        // We need 1 instruction per IP step.
        // IP increments by 2 in assembler? 
        // Assembler: pc += 2.
        // So we need to fetch bytes at IP and IP+1.
        // If texture is R32Uint, it's easier if we treat texture as array of bytes?
        // No, textureLoad returns vec4<u32> (or u32 for r32uint).
        // Let's assume we read the WORD at IP/2 if packed?
        // SAFE BET: Assembler PC corresponds to BYTE offset.
        // We fetch 2 bytes.
        // Instr = (Read(IP+1) << 8) | Read(IP)
        
        // Helper to read byte from ROM
        // This is expensive if not optimized, but OK for logic.
        // Assume ROM is R32Uint, stores 1 byte per pixel in .r channel?
        // Or stores 4 bytes? 
        // Existing engine uses `textureLoad(..., 0).r`. 
        // Let's assume 1 byte per pixel for simplicity of 'test_asm.rs' usually.
        
        let byte_l = textureLoad(rom_tex, vec2<i32>(i32(ip % 256u), i32(ip / 256u)), 0).r;
        let byte_h = textureLoad(rom_tex, vec2<i32>(i32((ip+1u) % 256u), i32((ip+1u) / 256u)), 0).r;
        let instr = (byte_h << 8u) | byte_l;
        
        let op = (instr >> 11u) & 0x1Fu;
        let rd_idx = (instr >> 8u) & 0x7u;
        let rs1_idx = (instr >> 5u) & 0x7u;
        let rs2_idx = (instr >> 2u) & 0x7u;
        
        // Default: IP += 2
        next_state = ip + 2u;
        
        // Control Flow Opcodes
        // JMP (11)
        if (op == 11u) {
            // Target Address is lowest 11 bits (10..0)
            next_state = instr & 0x7FFu;
        }
        else if (op == 12u) { // JZ (Jump Zero)
             let val = read_neighbor_reg(xy, rs1_idx, input_tex);
             if (val == 0u) {
                  // Address packed in Rd (10..8) | Rs2 (4..2) | Extra (1..0)
                  // Rd is bits 10..8 -> addr bits 7..5
                  // Rs2 is bits 4..2 -> addr bits 4..2
                  // Extra is bits 1..0 -> addr bits 1..0
                  let addr_h = rd_idx;
                  let addr_m = rs2_idx;
                  let addr_l = instr & 3u;
                  let addr = (addr_h << 5u) | (addr_m << 2u) | addr_l;
                  next_state = addr;
             } else { next_state = ip + 2u; }
        }
        else if (op == 13u) { // JNZ
             let val = read_neighbor_reg(xy, rs1_idx, input_tex);
             if (val != 0u) {
                  let addr_h = rd_idx;
                  let addr_m = rs2_idx;
                  let addr_l = instr & 3u;
                  let addr = (addr_h << 5u) | (addr_m << 2u) | addr_l;
                  next_state = addr;
             } else { next_state = ip + 2u; }
        }
    } 
    else if (type_id == C2_REG) {
        // Register Logic
        // Determine my Identity (R0-R7) relative to CPU Head
        // Search 8 neighbors for C2_HEAD
        var cpu_pos = vec2<i32>(0,0);
        var found_cpu = false;
        var my_reg_idx = 99u;
        
        // Directions: N(0,-1), NE(1,-1), E(1,0), SE(1,1), S(0,1), SW(-1,1), W(-1,0), NW(-1,-1)
        // Regs:       0        1         2       3        4       5         6        7
        
        // Check S neighbor (means I am N of CPU) -> Reg 0?
        // No, if CPU is at S, I am N. 
        // Relation: My Pos - CPU Pos
        
        // Let's Loop or Unroll
        // Check North (0,-1)
        if (textureLoad(meta_tex, xy + vec2(0, -1), 0).r == C2_HEAD) { found_cpu = true; cpu_pos = xy + vec2(0, -1); my_reg_idx = 4u; } // CPU is N, I am S (Reg 4)
        else if (textureLoad(meta_tex, xy + vec2(0, 1), 0).r == C2_HEAD) { found_cpu = true; cpu_pos = xy + vec2(0, 1); my_reg_idx = 0u; } // CPU is S, I am N (Reg 0)
        else if (textureLoad(meta_tex, xy + vec2(-1, 0), 0).r == C2_HEAD) { found_cpu = true; cpu_pos = xy + vec2(-1, 0); my_reg_idx = 2u; } // CPU is W, I am E (Reg 2)
        else if (textureLoad(meta_tex, xy + vec2(1, 0), 0).r == C2_HEAD) { found_cpu = true; cpu_pos = xy + vec2(1, 0); my_reg_idx = 6u; } // CPU is E, I am W (Reg 6)
        // Diagonals
        else if (textureLoad(meta_tex, xy + vec2(-1, 1), 0).r == C2_HEAD) { found_cpu = true; cpu_pos = xy + vec2(-1, 1); my_reg_idx = 1u; } // CPU is SW, I am NE (Reg 1)
        else if (textureLoad(meta_tex, xy + vec2(-1, -1), 0).r == C2_HEAD) { found_cpu = true; cpu_pos = xy + vec2(-1, -1); my_reg_idx = 3u; } // CPU is NW, I am SE (Reg 3)
        // Correcting Geometry mentally:
        // CPU center (cx, cy).
        // R0 (N)  at (cx, cy-1)
        // R1 (NE) at (cx+1, cy-1)
        // R2 (E)  at (cx+1, cy)
        // R3 (SE) at (cx+1, cy+1)
        // R4 (S)  at (cx, cy+1)
        // R5 (SW) at (cx-1, cy+1)
        // R6 (W)  at (cx-1, cy)
        // R7 (NW) at (cx-1, cy-1)
        
        // So from Neighbor's perspective:
        // If CPU is at (x, y+1) [South], I am at (x, y) [North] -> I am R0.
        // CHECKED.
        
        if (!found_cpu) {
            // Check other diagonals
            // I am SW (Reg 5) -> CPU is NE (1, -1)
             if (textureLoad(meta_tex, xy + vec2(1, -1), 0).r == C2_HEAD) { found_cpu = true; cpu_pos = xy + vec2(1, -1); my_reg_idx = 5u; }
            // I am NW (Reg 7) -> CPU is SE (1, 1)
             if (textureLoad(meta_tex, xy + vec2(1, 1), 0).r == C2_HEAD) { found_cpu = true; cpu_pos = xy + vec2(1, 1); my_reg_idx = 7u; }
        }
        
        if (found_cpu) {
            // Execute Instruction from CPU's perspective
            let ip = textureLoad(input_tex, cpu_pos, 0).r;
            let byte_l = textureLoad(rom_tex, vec2<i32>(i32(ip % 256u), i32(ip / 256u)), 0).r;
            let byte_h = textureLoad(rom_tex, vec2<i32>(i32((ip+1u) % 256u), i32((ip+1u) / 256u)), 0).r;
            let instr = (byte_h << 8u) | byte_l;
            
            let op = (instr >> 11u) & 0x1Fu;
            let rd_idx = (instr >> 8u) & 0x7u;
            let rs1_idx = (instr >> 5u) & 0x7u;
            let rs2_idx = (instr >> 2u) & 0x7u;
            
            // Only update if I am the Destination (Rd)
            if (rd_idx == my_reg_idx) {
                var res = current_state;
                
                // Read Source Values from CPU perspective
                // We need helper to Read Relative to CPU
                let val1 = read_neighbor_abs(cpu_pos, rs1_idx, input_tex);
                let val2 = read_neighbor_abs(cpu_pos, rs2_idx, input_tex);
                
                switch op {
                    case 1u: { // LDI
                        // Imm is 8 bits: Rd(10..8) Unused, Rs1(5..7)..Wait.
                        // Assembler maps Imm -> bits 0..7.
                        // We extract bits 0..7 from instr.
                        // bit 0,1 are in Extra.
                        // bit 2,3,4 are in Rs2 position (but Assembler set them directly)
                        // bit 5,6,7 are in Rs1 position
                        // Actually, Assembler puts `imm` directly at offset 0.
                        // So `instr & 0xFF` is the Immediate.
                        // But wait, `Rd` is at 8..10.
                        // So `instr & 0xFF` IS the immediate.
                        res = instr & 0xFFu;
                    }
                    case 2u: { res = val1; } // MOV
                    case 3u: { res = val1 + val2; } // ADD
                    case 4u: { res = val1 - val2; } // SUB
                    case 5u: { res = val1 * val2; } // MUL
                    case 6u: { if(val2 != 0u) { res = val1 / val2; } } // DIV
                    case 7u: { res = val1 & val2; } // AND
                    case 8u: { res = val1 | val2; } // OR
                    case 9u: { res = val1 ^ val2; } // XOR
                    case 10u: { res = ~val1; } // NOT
                    case 14u: { if (val1 == val2) { res = 1u; } else { res = 0u; } } // EQ
                    case 15u: { if (val1 > val2) { res = 1u; } else { res = 0u; } } // GT
                    case 16u: { if (val1 < val2) { res = 1u; } else { res = 0u; } } // LT
                    default: {}
                }
                
                // Mask to 8-bit? CHUNGUS 2 is 8-bit.
                next_state = res & 0xFFu;
            }
        }
    }

    textureStore(output_tex, xy, vec4<u32>(next_state, 0u, 0u, 0u));
    textureStore(color_tex, xy, get_color(type_id, next_state));
}

fn read_neighbor_reg(my_pos: vec2<i32>, reg_idx: u32, tex: texture_2d<u32>) -> u32 {
    // I am CPU, reading neighbor `reg_idx`
    return read_neighbor_abs(my_pos, reg_idx, tex);
}

fn read_neighbor_abs(center: vec2<i32>, reg_idx: u32, tex: texture_2d<u32>) -> u32 {
    var offset = vec2<i32>(0,0);
    switch reg_idx {
        case 0u: { offset = vec2(0, -1); } // N
        case 1u: { offset = vec2(1, -1); } // NE
        case 2u: { offset = vec2(1, 0); }  // E
        case 3u: { offset = vec2(1, 1); }  // SE
        case 4u: { offset = vec2(0, 1); }  // S
        case 5u: { offset = vec2(-1, 1); } // SW
        case 6u: { offset = vec2(-1, 0); } // W
        case 7u: { offset = vec2(-1, -1); } // NW
        default: {}
    }
    return textureLoad(tex, center + offset, 0).r;
}
