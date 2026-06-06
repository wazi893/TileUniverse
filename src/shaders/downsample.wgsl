@group(0) @binding(0) var source_tex: texture_2d<f32>;
@group(0) @binding(1) var sat_tex: texture_storage_2d<rgba8unorm, write>;

// Source Dimensions (Uniform)
struct DownsampleParams {
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
}
@group(0) @binding(2) var<uniform> params: DownsampleParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tx = global_id.x;
    let ty = global_id.y;
    
    if (tx >= params.target_width || ty >= params.target_height) {
        return;
    }

    // Calculate source region to average
    let scale_x = f32(params.source_width) / f32(params.target_width);
    let scale_y = f32(params.source_height) / f32(params.target_height);
    
    let start_x = u32(f32(tx) * scale_x);
    let start_y = u32(f32(ty) * scale_y);
    let end_x = u32(f32(tx + 1u) * scale_x);
    let end_y = u32(f32(ty + 1u) * scale_y);
    
    // Accumulate color
    var acc = vec4<f32>(0.0);
    var count = 0.0;
    
    for (var y = start_y; y < end_y; y++) {
        for (var x = start_x; x < end_x; x++) {
             // textureLoad from texture_2d requires mip level (0)
             acc += textureLoad(source_tex, vec2<i32>(i32(x), i32(y)), 0);
             count += 1.0;
        }
    }
    
    if (count > 0.0) {
        textureStore(sat_tex, vec2<i32>(i32(tx), i32(ty)), acc / count);
    } else {
        textureStore(sat_tex, vec2<i32>(i32(tx), i32(ty)), textureLoad(source_tex, vec2<i32>(i32(start_x), i32(start_y)), 0));
    }
}
