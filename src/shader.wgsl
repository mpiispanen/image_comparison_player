// Vertex shader

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

struct Uniforms {
    cursor_x: f32,
    image1_size: vec2<f32>,
    image2_size: vec2<f32>,
};

@group(1) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    out.tex_coords = position * 0.5 + 0.5;
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse1: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse1: sampler;
@group(0) @binding(2)
var t_diffuse2: texture_2d<f32>;
@group(0) @binding(3)
var s_diffuse2: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect_ratio = uniforms.image1_size.x / uniforms.image1_size.y;
    let scaled_tex_coords = vec2<f32>(in.tex_coords.x * aspect_ratio, in.tex_coords.y);
    
    let color1 = textureSample(t_diffuse1, s_diffuse1, scaled_tex_coords);
    let color2 = textureSample(t_diffuse2, s_diffuse2, scaled_tex_coords);
    
    return select(color2, color1, in.tex_coords.x < uniforms.cursor_x);
}