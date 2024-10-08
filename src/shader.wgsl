// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>, // Add this line
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

struct Uniforms {
    cursor_x: f32,
    cursor_y: f32,
    image1_size: vec2<f32>,
    image2_size: vec2<f32>,
    flip_diff_size: vec2<f32>,
};

@group(1) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 1.0);
    out.tex_coords = model.tex_coords; // Add this line
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
@group(0) @binding(4)
var t_flip_diff: texture_2d<f32>;
@group(0) @binding(5)
var s_flip_diff: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect_ratio = uniforms.image1_size.x / uniforms.image1_size.y;
    let scaled_tex_coords = vec2<f32>(in.tex_coords.x * aspect_ratio, in.tex_coords.y);
    
    let color1 = textureSample(t_diffuse1, s_diffuse1, scaled_tex_coords);
    let color2 = textureSample(t_diffuse2, s_diffuse2, scaled_tex_coords);
    
    let flip_diff_aspect_ratio = uniforms.flip_diff_size.x / uniforms.flip_diff_size.y;
    let flip_diff_tex_coords = vec2<f32>(in.tex_coords.x * flip_diff_aspect_ratio, in.tex_coords.y);
    let flip_diff_color = textureSample(t_flip_diff, s_flip_diff, flip_diff_tex_coords);
    
    let show_flip_diff = uniforms.cursor_y < 1.0 && in.tex_coords.y > uniforms.cursor_y;
    let use_right_image = in.tex_coords.x >= uniforms.cursor_x;
    
    var color: vec4<f32>;
    if (show_flip_diff) {
        color = flip_diff_color;
    } else if (use_right_image) {
        let scaled_y = in.tex_coords.y / select(1.0, uniforms.cursor_y, uniforms.cursor_y < 1.0);
        if (scaled_y <= 1.0) {
            color = textureSample(t_diffuse2, s_diffuse2, vec2<f32>(scaled_tex_coords.x, scaled_y));
        } else {
            color = vec4<f32>(0.0, 0.0, 0.0, 1.0); // Black for the area below the right image
        }
    } else {
        color = color1;
    }
    
    return color;
}