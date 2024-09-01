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
    let aspect_ratio = 1.0;
    let scaled_tex_coords = vec2<f32>(in.tex_coords.x * aspect_ratio, in.tex_coords.y);

    let flip_diff_aspect_ratio = 1.0;
    let flip_diff_tex_coords = vec2<f32>(in.tex_coords.x * flip_diff_aspect_ratio, in.tex_coords.y);

    let color1 = textureSample(t_diffuse, s_diffuse, scaled_tex_coords);
    let color2 = textureSample(t_diffuse2, s_diffuse2, scaled_tex_coords);
    let color_diff = textureSample(t_flip_diff, s_flip_diff, flip_diff_tex_coords);

    let t_x = step(uniforms.cursor_x, in.tex_coords.x);
    let show_flip_diff = step(uniforms.cursor_y, in.tex_coords.y);

    let color_top = mix(color1, color2, t_x);
    let final_color = mix(color_top, color_diff, show_flip_diff);

    return final_color;
}
