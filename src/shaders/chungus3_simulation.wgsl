// CHUNGUS 3: Planetary Edition (ADDM)
// Async Distributed Dataflow Mesh Simulation Shader

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
// We might use ROM later for static graph definitions, but for now purely dynamic.
@group(0) @binding(5) var rom_tex: texture_2d<u32>; 

// Constants / Tile Types
const TYPE_WIRE = 0u;
const C3_FU_ADD = 60u;
const C3_FU_MUL = 61u;
const C3_TOKEN  = 62u; // General purpose storage/wire
const C3_OPTICAL = 63u;
const C3_MERGE = 66u;
const C3_SOURCE = 67u;
const WIRE_W = 70u;
const WIRE_N = 71u;
const WIRE_S = 72u;

// Token Layout (32-bit packed)
// [ Value (8) ] [ Meta (8) ] [ Tag (16) ]
// Value: The data.
// Meta: Validity, Ack, Direction?
// Tag: Context ID.

// Meta Flags
const META_VALID_MASK = 0x80u; // Bit 7 of Meta (Bit 23 of full word)
const META_ACK_MASK   = 0x40u; // Bit 6 of Meta

fn pack_token(val: u32, metadata: u32, tag: u32) -> u32 {
    // Val: 0..7, Meta: 8..15, Tag: 16..31
    // Actually, design said: 
    // [ Tag (16) ] [ Meta (8) ] [ Value (8) ] makes getting value easier (mask 0xFF)
    return (tag << 16u) | (metadata << 8u) | (val & 0xFFu);
}

fn unpack_val(token: u32) -> u32 { return token & 0xFFu; }
fn unpack_meta(token: u32) -> u32 { return (token >> 8u) & 0xFFu; }
fn unpack_tag(token: u32) -> u32 { return (token >> 16u) & 0xFFFFu; }

fn is_valid(token: u32) -> bool {
    return (unpack_meta(token) & META_VALID_MASK) != 0u;
}

fn get_color(type_id: u32, state: u32) -> vec4<f32> {
    switch type_id {
        case 60u: { return vec4<f32>(0.9, 0.4, 0.1, 1.0); } // ADD (Orange)
        case 61u: { return vec4<f32>(0.9, 0.1, 0.1, 1.0); } // MUL (Red)
        case 62u: { // TOKEN / WIRE
            if (is_valid(state)) {
                let tag = unpack_tag(state);
                let r = f32((tag * 37u) % 255u) / 255.0;
                let g = f32((tag * 113u) % 255u) / 255.0;
                let b = f32((tag * 19u) % 255u) / 255.0;
                return vec4<f32>(0.5 + r*0.5, 0.5 + g*0.5, 0.5 + b*0.5, 1.0);
            } else {
                return vec4<f32>(0.2, 0.2, 0.2, 1.0); // Dark Gray
            }
        }
        case 63u: { return vec4<f32>(0.1, 0.8, 0.8, 1.0); } // OPTICAL (Cyan)
        case 66u: { return vec4<f32>(0.6, 0.2, 0.8, 1.0); } // MERGE (Purple)
        case 67u: { return vec4<f32>(1.0, 1.0, 1.0, 1.0); } // SOURCE (White)
        case 70u: { return vec4<f32>(0.2, 0.2, 0.5, 1.0); } // WIRE_W (Blue)
        case 71u: { return vec4<f32>(0.5, 0.5, 0.0, 1.0); } // Wire N
        case 72u: { return vec4<f32>(0.5, 0.0, 0.5, 1.0); } // Wire S
        case 75u: { return vec4<f32>(1.0, 0.0, 1.0, 1.0); } // Scatter (Pink)
        default: { return vec4<f32>(0.05, 0.05, 0.05, 1.0); }
    }
}

// --- Helper for RNG ---
fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let xy = vec2<i32>(global_id.xy);
    let dims = textureDimensions(input_tex);
    if (xy.x >= i32(dims.x) || xy.y >= i32(dims.y)) { return; }

    let type_id = textureLoad(meta_tex, xy, 0).r;
    let current_state = textureLoad(input_tex, xy, 0).r;
    var next_state = current_state;

    // Default decay/clear if not persistent?
    // Dataflow tokens flow. If I am a wire, and I have a valid token, 
    // I should pass it to neighbor and clear myself IF neighbor accepted it (Ack).
    // For prototype, let's just do "Flow Forward" (Copy from previous).
    
    // Simplest flow: Grid behaves like a shift register for Valid tokens?
    // We need directionality.
    // Let's assume C3_TOKEN acts like a "Cell".
    // 
    // Rule 1 (Wire): Pull from valid neighbor if I am empty.
    // Rule 2 (FU): Pull from inputs, compute, push to output.
    
    // Directional convention: Left -> Right flow for prototype?
    


    if (type_id == C3_TOKEN) {
        // PHYSICS MODE: Gravity & Scatter
        let C3_SCATTER: u32 = 75u;
        let C3_ADD: u32 = 60u;
        let C3_MUL: u32 = 61u;
        
        let me_valid = is_valid(current_state);

        // 1. Pull Logic (Incoming)
        if (!me_valid) {
             let n_pos = xy + vec2(0, -1);
             let n_state = textureLoad(input_tex, n_pos, 0).r;
             let n_type = textureLoad(meta_tex, n_pos, 0).r;

             // Check Direct Gravity (North)
             // Case: North is Token (Gravity)
             // Case: North is FU (ADD/MUL) outputting result?
             if (is_valid(n_state)) {
                  next_state = n_state;
             }
             // Case: North is FU (ADD/MUL) holding a result, attempting to push?
             else if (n_type == C3_ADD || n_type == C3_MUL) {
                  // If FU has valid result, pull it.
                  let fu_state = textureLoad(input_tex, n_pos, 0).r;
                  if (is_valid(fu_state)) {
                      next_state = fu_state;
                  }
             }
             else {
                 // Check Scatter Inputs (Diagonal)
                 let nw_pos = xy + vec2(-1, -1);
                 let nw_state = textureLoad(input_tex, nw_pos, 0).r;
                 let w_pos = xy + vec2(-1, 0); 
                 let w_type = textureLoad(meta_tex, w_pos, 0).r;
                 
                 var pulled = false;
                 
                 if (is_valid(nw_state) && w_type == C3_SCATTER) {
                      let tag = unpack_tag(nw_state);
                      let seed = tag ^ u32(nw_pos.x) ^ u32(nw_pos.y);
                      if ((pcg_hash(seed) % 2u) == 1u) { // 1 = East (To Me)
                          next_state = nw_state;
                          pulled = true;
                      }
                 }
                 
                 if (!pulled) {
                     let ne_pos = xy + vec2(1, -1);
                     let ne_state = textureLoad(input_tex, ne_pos, 0).r;
                     let e_pos = xy + vec2(1, 0);
                     let e_type = textureLoad(meta_tex, e_pos, 0).r;
                     
                     if (is_valid(ne_state) && e_type == C3_SCATTER) {
                          let tag = unpack_tag(ne_state);
                          let seed = tag ^ u32(ne_pos.x) ^ u32(ne_pos.y);
                          if ((pcg_hash(seed) % 2u) == 0u) { // 0 = West (To Me)
                              next_state = ne_state;
                          }
                     }
                 }
             }

        } else {
             // 2. Clear Logic (Outgoing)
             let s_pos = xy + vec2(0, 1);
             let dims = textureDimensions(input_tex);
             
             if (s_pos.y < i32(dims.y)) {
                 let s_type = textureLoad(meta_tex, s_pos, 0).r;
                 let s_state = textureLoad(input_tex, s_pos, 0).r;
                 
                 // Case A: Gravity (South is Empty Air)
                 if (s_type != C3_SOURCE && s_type != C3_SCATTER && !is_valid(s_state)) {
                     next_state = 0u; 
                 }
                  // Case B: Scatter (South is Scatter Pin)
                  else if (s_type == C3_SCATTER) {
                      // RNG: Where do I go?
                      let tag = unpack_tag(current_state);
                      let seed = tag ^ u32(xy.x) ^ u32(xy.y);
                      
                      var target_pos = xy; 
                      var blocked = false;
 
                      if ((pcg_hash(seed) % 2u) == 0u) { // West (SW)
                          target_pos = xy + vec2(-1, 1);
                          
                          // Priority Check: I am NE of Target.
                          let competitor_pos = xy + vec2(-2, 0);
                          let c_state = textureLoad(input_tex, competitor_pos, 0).r;
                          let cs_pos = xy + vec2(-2, 1); 
                          let cs_type = textureLoad(meta_tex, cs_pos, 0).r;
                          
                          if (is_valid(c_state) && cs_type == C3_SCATTER) {
                              let c_tag = unpack_tag(c_state);
                              let c_seed = c_tag ^ u32(competitor_pos.x) ^ u32(competitor_pos.y);
                              if ((pcg_hash(c_seed) % 2u) == 1u) { 
                                  blocked = true; 
                              }
                          }
                      } else { // East (SE)
                          target_pos = xy + vec2(1, 1);
                      }
                      
                      let t_state = textureLoad(input_tex, target_pos, 0).r;
                      let t_type = textureLoad(meta_tex, target_pos, 0).r;
                      
                      if (!blocked && t_type != C3_SOURCE && !is_valid(t_state)) {
                          next_state = 0u; 
                      }
                  }
                  // Case C: Functional Unit (ADD/MUL) Consumption
                  else if (s_type == C3_ADD || s_type == C3_MUL) {
                      // I am North of the FU? No wait.
                      // Logic: FU inputs are NW and NE.
                      // So if South is ADD, I am North of ADD.
                      // ADD reads NW and NE.
                      // So for ME (at X, Y) to be consumed, ADD must be at (X+1, Y+1) [I am NW] OR (X-1, Y+1) [I am NE].
                      // THIS BLOCK checks `s_pos = xy + vec2(0, 1)`. So ADD is directly South.
                      // If ADD is South, I am NOT an input (I am "North").
                      // Is "North" an input? Maybe?
                      // Design Decision: Let's make Inputs NW and NE.
                      // So `s_type == C3_ADD` checks if *Gravity* pulls me into it?
                      // If Gravity pulls me into ADD -> I Block? Or I slide?
                      // If ADD inputs are Diagonal, then "Top" is just a roof?
                      // Let's say Top is BLOCKED. (Tokens stack on top of ADD).
                      // So do nothing. next_state = current_state.
                  }
                  
                  // Case D: Diagonal Consumption (I am NW or NE of an ADD unit)
                  // Check SE (x+1, y+1).
                  let se_pos = xy + vec2(1, 1);
                  let se_type = textureLoad(meta_tex, se_pos, 0).r;
                  if (se_type == C3_ADD || se_type == C3_MUL) {
                      // I am NW input.
                      // Check if NE input (Partner) exists.
                      // Partner is at (Answer: ADD is at X+1. Partner is East of ADD -> X+2 of Me).
                      // partner_pos = xy + vec2(2, 0).
                      let p_pos = xy + vec2(2, 0);
                      let p_state = textureLoad(input_tex, p_pos, 0).r;
                      
                      // Also check if ADD is ready (Empty).
                      let fu_state = textureLoad(input_tex, se_pos, 0).r;
                      
                      if (is_valid(p_state) && !is_valid(fu_state)) {
                          next_state = 0u; // Consumed!
                      }
                  }
                  
                  // Check SW (x-1, y+1).
                  let sw_pos = xy + vec2(-1, 1);
                  let sw_type = textureLoad(meta_tex, sw_pos, 0).r;
                  if (sw_type == C3_ADD || sw_type == C3_MUL) {
                      // I am NE input.
                      // Partner is West of ADD -> X-2 of Me.
                      let p_pos = xy + vec2(-2, 0);
                      let p_state = textureLoad(input_tex, p_pos, 0).r;
                      let fu_state = textureLoad(input_tex, sw_pos, 0).r;
                      
                      if (is_valid(p_state) && !is_valid(fu_state)) {
                          next_state = 0u; // Consumed!
                      }
                  }
             }
        }
        // Legacy West->East Logic Removed for Physics Mode
    }
    else if (type_id == WIRE_W) {
        // Wire Logic: Flow East -> West
        let me_valid = is_valid(current_state);
        let w_valid = is_valid(textureLoad(input_tex, xy + vec2(-1, 0), 0).r);
        let w_type = textureLoad(meta_tex, xy + vec2(-1, 0), 0).r;
        
        let e_state = textureLoad(input_tex, xy + vec2(1, 0), 0).r;
        let e_valid = is_valid(e_state);

        if (!me_valid) {
            if (e_valid) { next_state = e_state; }
        } else {
            // Clearing Logic (Generic)
            var consumed = false;
            // Check West (WireW) - Natural Flow
            if (w_type == WIRE_W) { if (!w_valid) { consumed = true; } }
            // Check East (WireE) - Turn Back?
            let e_type = textureLoad(meta_tex, xy + vec2(1, 0), 0).r;
            if (e_type == C3_TOKEN || e_type == C3_MERGE) { if (!e_valid) { consumed = true; } }
            // Check South
            let s_pos = xy + vec2(0, 1);
            let s_type = textureLoad(meta_tex, s_pos, 0).r;
            let s_valid = is_valid(textureLoad(input_tex, s_pos, 0).r);
            if (s_type == WIRE_S || s_type == C3_FU_ADD) { if (!s_valid) { consumed = true; } }
            // Check North
            let n_pos = xy + vec2(0, -1);
            let n_type = textureLoad(meta_tex, n_pos, 0).r;
            let n_valid = is_valid(textureLoad(input_tex, n_pos, 0).r);
            if (n_type == WIRE_N || n_type == C3_FU_ADD) { if (!n_valid) { consumed = true; } }
            
            if (consumed) { next_state = 0u; } else { next_state = current_state; }
        }
    }
    else if (type_id == WIRE_N) {
        // Wire Logic: Flow South -> North
        let me_valid = is_valid(current_state);
        let s_state = textureLoad(input_tex, xy + vec2(0, 1), 0).r;
        let s_valid = is_valid(s_state);

        if (!me_valid) {
            if (s_valid) { next_state = s_state; }
        } else {
            var consumed = false;
             // Check North (WireN) - Natural Flow
            let n_pos = xy + vec2(0, -1);
            let n_type = textureLoad(meta_tex, n_pos, 0).r;
            let n_valid = is_valid(textureLoad(input_tex, n_pos, 0).r);
            
            if (n_type == WIRE_N) { 
                if (!n_valid) { consumed = true; } 
            } else if (n_type == C3_FU_ADD) {
                 // Check ADD Partner (North Check -> I am South input. Partner is North input).
                 // Partner is at n_pos + (0, -1)
                 let p_valid = is_valid(textureLoad(input_tex, n_pos + vec2(0, -1), 0).r);
                 if (!n_valid && p_valid) { consumed = true; }
            }
            
             // Check South?
            let s_type = textureLoad(meta_tex, xy + vec2(0, 1), 0).r; 
            if (s_type == WIRE_S || s_type == C3_FU_ADD) { if (!s_valid) { consumed = true; } } 
             // Check West
            let w_pos = xy + vec2(-1, 0);
            let w_type = textureLoad(meta_tex, w_pos, 0).r;
            let w_valid = is_valid(textureLoad(input_tex, w_pos, 0).r);
            if (w_type == WIRE_W) { if (!w_valid) { consumed = true; } }
             // Check East
            let e_pos = xy + vec2(1, 0);
            let e_type = textureLoad(meta_tex, e_pos, 0).r;
            let e_valid = is_valid(textureLoad(input_tex, e_pos, 0).r);
            if (e_type == C3_TOKEN || e_type == C3_MERGE) { if (!e_valid) { consumed = true; } }
            
            if (consumed) { next_state = 0u; } else { next_state = current_state; }
        }
    }
     else if (type_id == WIRE_S) {
        // Wire Logic: Flow North -> South
        let me_valid = is_valid(current_state);
        let n_state = textureLoad(input_tex, xy + vec2(0, -1), 0).r;
        let n_valid = is_valid(n_state);

        if (!me_valid) {
            if (n_valid) { next_state = n_state; }
        } else {
             var consumed = false;
             // Check South (WireS) - Natural Flow
            let s_pos = xy + vec2(0, 1);
            let s_type = textureLoad(meta_tex, s_pos, 0).r;
            let s_valid = is_valid(textureLoad(input_tex, s_pos, 0).r);
            
            if (s_type == WIRE_S) { 
                if (!s_valid) { consumed = true; } 
            } else if (s_type == C3_FU_ADD) {
                 // Check ADD Partner (South Check -> I am North input. Partner is South input).
                 // Partner is at s_pos + (0, 1)
                 let p_valid = is_valid(textureLoad(input_tex, s_pos + vec2(0, 1), 0).r);
                 if (!s_valid && p_valid) { consumed = true; }
            } else if (s_type == C3_MERGE) {
                 // Check Merge Priority (West)
                 // If West is valid, Merge takes West -> I hold.
                 // Partner is at s_pos + (-1, 0)
                 let p_valid = is_valid(textureLoad(input_tex, s_pos + vec2(-1, 0), 0).r);
                 if (!s_valid && !p_valid) { consumed = true; }
            }

             // Check North
            let n_pos = xy + vec2(0, -1);
            let n_type = textureLoad(meta_tex, n_pos, 0).r;
            let n_valid = is_valid(textureLoad(input_tex, n_pos, 0).r);
            if (n_type == WIRE_N || n_type == C3_FU_ADD) { if (!n_valid) { consumed = true; } }
             // Check West
            let w_pos = xy + vec2(-1, 0);
            let w_type = textureLoad(meta_tex, w_pos, 0).r;
            let w_valid = is_valid(textureLoad(input_tex, w_pos, 0).r);
            if (w_type == WIRE_W) { if (!w_valid) { consumed = true; } }
             // Check East
            let e_pos = xy + vec2(1, 0);
            let e_type = textureLoad(meta_tex, e_pos, 0).r;
            let e_valid = is_valid(textureLoad(input_tex, e_pos, 0).r);
            if (e_type == C3_TOKEN || e_type == C3_MERGE) { if (!e_valid) { consumed = true; } }

            if (consumed) { next_state = 0u; } else { next_state = current_state; }
        }
    }
    else if (type_id == C3_OPTICAL) {
        // ... (Optical Logic kept as is) ...
        let me_valid = is_valid(current_state);
        
        if (!me_valid) {
            // Pull Logic: Raycast West up to 20 blocks
            for (var i: i32 = 1; i <= 20; i++) {
                let pos = xy + vec2(-1, 0) * i;
                let s_state = textureLoad(input_tex, pos, 0).r;
                if (is_valid(s_state)) {
                    next_state = s_state; // Copy instantly
                    break;
                }
            }
        } else {
             // Push/Clear Logic: Check East
             let e_pos = xy + vec2(1, 0);
             let e_state = textureLoad(input_tex, e_pos, 0).r;
             if (!is_valid(e_state)) {
                 next_state = 0u;
             } else {
                 next_state = current_state; 
             }
        }
    }
    else if (type_id == 66u) { // C3_MERGE
        // Merge Logic: Input West (Priority) OR Input North.
        // Output: East.
        // Useful for Loops (Initial Value vs Feedback).
        
        let me_valid = is_valid(current_state);
        let e_pos = xy + vec2(1, 0);
        let e_state = textureLoad(input_tex, e_pos, 0).r;
        let e_valid = is_valid(e_state);

        if (!me_valid) {
            // Priority: West, then North
            let w_pos = xy + vec2(-1, 0);
            let w_state = textureLoad(input_tex, w_pos, 0).r;
            
            if (is_valid(w_state)) {
                next_state = w_state;
            } else {
                let n_pos = xy + vec2(0, -1);
                let n_state = textureLoad(input_tex, n_pos, 0).r;
                if (is_valid(n_state)) {
                    next_state = n_state;
                }
            }
        } else {
            // Push to East
            if (!e_valid) {
                next_state = 0u;
            } else {
                next_state = current_state;
            }
        }
    }
    else if (type_id == C3_SOURCE) {
        // Source Logic: Always maintain state (Persistent).
        // Downstream logic (Wire/Optical) pulls from here.
        next_state = current_state;
    }
    else if (type_id == 60u || type_id == 61u) { // C3_ADD / C3_MUL
        let C3_ADD: u32 = 60u;
        
        // Logic: Inputs NW, NE. Output: South (Gravity).
        let me_valid = is_valid(current_state);
        
        let nw_pos = xy + vec2(-1, -1);
        let nw_state = textureLoad(input_tex, nw_pos, 0).r;
        let ne_pos = xy + vec2(1, -1);
        let ne_state = textureLoad(input_tex, ne_pos, 0).r;
        
        if (!me_valid) {
            // Execution Phase
            if (is_valid(nw_state) && is_valid(ne_state)) {
                // Unpack Values (Lower 8 bits?)
                // Assuming Tag(16) | Meta(8) | Value(8)
                let val_a = nw_state & 0xFFu;
                let val_b = ne_state & 0xFFu;
                var res = 0u;
                
                if (type_id == C3_ADD) {
                    res = (val_a + val_b) % 255u;
                } else {
                    res = (val_a * val_b) % 255u;
                }
                
                // Construct Result
                // Use input tags? Or fixed tag?
                // Visualizer needs Tag for Color?
                // Let's preserve NW tag.
                let tag = unpack_tag(nw_state);
                next_state = (tag << 16u) | (0x80u << 8u) | res;
            }
        } else {
            // Output Phase (Push South)
            let s_pos = xy + vec2(0, 1);
            let s_type = textureLoad(meta_tex, s_pos, 0).r;
            let s_state = textureLoad(input_tex, s_pos, 0).r;
            
            // If South is Token (62) and Empty, it pulls me.
            // I clear myself.
            if (s_type == 62u && !is_valid(s_state)) {
                next_state = 0u; 
            }
        }
    }

    textureStore(output_tex, xy, vec4<u32>(next_state, 0u, 0u, 0u));
    textureStore(color_tex, xy, get_color(type_id, next_state));
}
