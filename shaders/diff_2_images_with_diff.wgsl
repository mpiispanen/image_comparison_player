struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

struct Uniforms {
    cursor_x: f32,
    cursor_y: f32,
    image1_size: vec2<f32>,
    image2_size: vec2<f32>,
    flip_diff_size: vec2<f32>,
    show_flip_diff: f32,
    zoom_level: f32,
    zoom_center: vec2<f32>,
}

@group(1) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(0) @binding(2)
var t_diffuse2: texture_2d<f32>;
@group(0) @binding(3)
var s_diffuse2: sampler;
@group(0) @binding(4)
var t_flip_diff: texture_2d<f32>;
@group(0) @binding(5)
var s_flip_diff: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply zoom
    let zoom_offset = (in.tex_coords - uniforms.zoom_center) / uniforms.zoom_level;
    let zoomed_tex_coords = uniforms.zoom_center + zoom_offset;
    
    // Clamp the zoomed coordinates to [0, 1]
    let clamped_tex_coords = clamp(zoomed_tex_coords, vec2(0.0), vec2(1.0));

    let color1 = textureSample(t_diffuse, s_diffuse, clamped_tex_coords);
    let color2 = textureSample(t_diffuse2, s_diffuse2, clamped_tex_coords);
    let color_diff = textureSample(t_flip_diff, s_flip_diff, clamped_tex_coords);

    let t_x = step(uniforms.cursor_x, in.tex_coords.x);
    let show_flip_diff = step(uniforms.cursor_y, in.tex_coords.y);

    let color_top = mix(color1, color2, t_x);
    let final_color = mix(color_top, color_diff, show_flip_diff);

    // Calculate alpha based on whether the zoomed coordinates are within bounds
    let alpha = 1.0 - step(1.0, max(abs(zoomed_tex_coords.x - 0.5), abs(zoomed_tex_coords.y - 0.5)) * 2.0);

    return vec4<f32>(final_color.rgb, final_color.a * alpha);
}
