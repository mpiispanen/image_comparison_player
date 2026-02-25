// Vertex shader

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
    comparison_mode: f32,
    zoom_level: f32,
    zoom_center: vec2<f32>,
    window_size: vec2<f32>,
    show_image1: f32,
    show_image2: f32,
    show_split_line: f32,
}

@group(1) @binding(0)
var<uniform> uniforms: Uniforms;

const MODE_NORMAL: f32 = 0.0;
const MODE_FLIP: f32 = 1.0;
const MODE_OVERLAY: f32 = 2.0;
const MODE_ABS_DIFF: f32 = 3.0;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    
    // Calculate the aspect ratios
    let window_aspect_ratio = uniforms.window_size.x / uniforms.window_size.y;
    let image_aspect_ratio = uniforms.image1_size.x / uniforms.image1_size.y;
    
    // Calculate the scale factor to fit the image within the window
    let scale_x = select(1.0, image_aspect_ratio / window_aspect_ratio, window_aspect_ratio > image_aspect_ratio);
    let scale_y = select(window_aspect_ratio / image_aspect_ratio, 1.0, window_aspect_ratio > image_aspect_ratio);
    let scale = vec2<f32>(scale_x, scale_y);
    
    // Apply the scale to the position
    let scaled_position = model.position.xy * scale;
    
    out.clip_position = vec4<f32>(scaled_position, 0.0, 1.0);
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
    // Apply zoom
    let zoom_offset = (in.tex_coords - uniforms.zoom_center) / uniforms.zoom_level;
    let zoomed_tex_coords = uniforms.zoom_center + zoom_offset;
    
    // Clamp the zoomed coordinates to [0, 1]
    let clamped_tex_coords = clamp(zoomed_tex_coords, vec2(0.0), vec2(1.0));

    let color1 = textureSample(t_diffuse1, s_diffuse1, clamped_tex_coords);
    let color2 = textureSample(t_diffuse2, s_diffuse2, clamped_tex_coords);

    // Calculate alpha based on whether the zoomed coordinates are within bounds
    let alpha = 1.0 - step(1.0, max(abs(zoomed_tex_coords.x - 0.5), abs(zoomed_tex_coords.y - 0.5)) * 2.0);

    let mode = uniforms.comparison_mode;

    // Determine split factor based on which images are enabled
    let both_shown = uniforms.show_image1 * uniforms.show_image2;
    let only_image2 = (1.0 - uniforms.show_image1) * uniforms.show_image2;
    let t = both_shown * step(uniforms.cursor_x, in.tex_coords.x) + only_image2;
    let mixed_color = mix(color1 * uniforms.show_image1, color2 * uniforms.show_image2, t);

    // Overlay/Abs Diff are shown below cursor_y so you can compare against originals interactively.
    if mode == MODE_OVERLAY || mode == MODE_ABS_DIFF {
        let overlay_color = mix(color1, color2, 0.5);
        let diff = abs(color1 - color2);
        let abs_diff_color = vec4<f32>(clamp(diff.rgb * 10.0, vec3(0.0), vec3(1.0)), 1.0);
        let mode_color = select(abs_diff_color, overlay_color, mode == MODE_OVERLAY);
        let show_mode = step(uniforms.cursor_y, in.tex_coords.y);
        let combined = mix(mixed_color, mode_color, show_mode);
        return vec4<f32>(combined.rgb, combined.a * alpha);
    }

    // Add a 1-pixel-wide white line at the split position using screen-space coordinates
    // so the line stays a constant width regardless of image resolution or window size.
    let window_aspect_ratio_fs = uniforms.window_size.x / uniforms.window_size.y;
    let image_aspect_ratio_fs = uniforms.image1_size.x / uniforms.image1_size.y;
    let scale_x_fs = select(1.0, image_aspect_ratio_fs / window_aspect_ratio_fs, window_aspect_ratio_fs > image_aspect_ratio_fs);
    let render_width_fs = scale_x_fs * uniforms.window_size.x;
    let x_offset_fs = (uniforms.window_size.x - render_width_fs) / 2.0;
    let cursor_screen_x = x_offset_fs + uniforms.cursor_x * render_width_fs;
    let in_original_region = mode <= MODE_FLIP || in.tex_coords.y < uniforms.cursor_y;
    if (both_shown > 0.5 && uniforms.show_split_line > 0.5 && in_original_region && abs(in.clip_position.x - cursor_screen_x) < 1.0) {
        return vec4<f32>(1.0, 1.0, 1.0, alpha); // White color for the line
    }

    return vec4<f32>(mixed_color.rgb, mixed_color.a * alpha);
}
