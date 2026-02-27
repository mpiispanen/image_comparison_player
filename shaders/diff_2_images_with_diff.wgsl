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
    peek_active: f32,
    peek_factor: f32,
    peek_radius: f32,
    diff_multiplier: f32,
    pump_active: f32,
    time: f32,
    _padding: f32,
}

@group(1) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;

    // Match regular pipeline behavior: preserve source image aspect ratio with letterboxing.
    let window_aspect_ratio = uniforms.window_size.x / uniforms.window_size.y;
    let image_aspect_ratio = uniforms.image1_size.x / uniforms.image1_size.y;
    let scale_x = select(1.0, image_aspect_ratio / window_aspect_ratio, window_aspect_ratio > image_aspect_ratio);
    let scale_y = select(window_aspect_ratio / image_aspect_ratio, 1.0, window_aspect_ratio > image_aspect_ratio);
    let scale = vec2<f32>(scale_x, scale_y);
    let scaled_position = model.position.xy * scale;

    out.clip_position = vec4<f32>(scaled_position, 0.0, 1.0);
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

    // Determine split factor based on which images are enabled
    let both_shown = uniforms.show_image1 * uniforms.show_image2;
    let only_image2 = (1.0 - uniforms.show_image1) * uniforms.show_image2;
    let t_x = both_shown * step(uniforms.cursor_x, in.tex_coords.x) + only_image2;

    let show_flip_diff = step(uniforms.cursor_y, in.tex_coords.y);

    let color_top = mix(color1 * uniforms.show_image1, color2 * uniforms.show_image2, t_x);
    let final_color = mix(color_top, color_diff, show_flip_diff);

    // Calculate alpha based on whether the zoomed coordinates are within bounds
    let alpha = 1.0 - step(1.0, max(abs(zoomed_tex_coords.x - 0.5), abs(zoomed_tex_coords.y - 0.5)) * 2.0);

    // Peek zoom magnifier: active while Z is held
    if (uniforms.peek_active > 0.5) {
        let w_ar = uniforms.window_size.x / uniforms.window_size.y;
        let i_ar = uniforms.image1_size.x / uniforms.image1_size.y;
        let sx = select(1.0, i_ar / w_ar, w_ar > i_ar);
        let sy = select(w_ar / i_ar, 1.0, w_ar > i_ar);
        let rw = sx * uniforms.window_size.x;
        let rh = sy * uniforms.window_size.y;
        let xo = (uniforms.window_size.x - rw) / 2.0;
        let yo = (uniforms.window_size.y - rh) / 2.0;
        let cx = xo + uniforms.cursor_x * rw;
        let cy = yo + uniforms.cursor_y * rh;
        let dx = in.clip_position.x - cx;
        let dy = in.clip_position.y - cy;
        let dist = sqrt(dx * dx + dy * dy);
        if (dist < uniforms.peek_radius) {
            // White border ring (2 px)
            if (dist > uniforms.peek_radius - 2.0) {
                return vec4<f32>(1.0, 1.0, 1.0, 1.0);
            }
            // Magnify: contract screen offset by peek_factor to zoom in
            let peek_sx = cx + dx / uniforms.peek_factor;
            let peek_sy = cy + dy / uniforms.peek_factor;
            let tc = vec2((peek_sx - xo) / rw, (peek_sy - yo) / rh);
            // Apply main zoom on top of peek zoom
            let zoom_ofs = (tc - uniforms.zoom_center) / uniforms.zoom_level;
            let ztc = clamp(uniforms.zoom_center + zoom_ofs, vec2(0.0), vec2(1.0));
            let p1 = textureSampleLevel(t_diffuse, s_diffuse, ztc, 0.0);
            let p2 = textureSampleLevel(t_diffuse2, s_diffuse2, ztc, 0.0);
            let t_p = both_shown * step(uniforms.cursor_x, tc.x) + only_image2;
            let peek_color = mix(p1 * uniforms.show_image1, p2 * uniforms.show_image2, t_p);
            return vec4<f32>(peek_color.rgb, peek_color.a);
        }
    }

    // Add a 1-pixel-wide white line at the split position using screen-space coordinates
    // so the line stays a constant width regardless of image resolution or window size.
    let window_aspect_ratio_fs = uniforms.window_size.x / uniforms.window_size.y;
    let image_aspect_ratio_fs = uniforms.image1_size.x / uniforms.image1_size.y;
    let scale_x_fs = select(1.0, image_aspect_ratio_fs / window_aspect_ratio_fs, window_aspect_ratio_fs > image_aspect_ratio_fs);
    let scale_y_fs = select(window_aspect_ratio_fs / image_aspect_ratio_fs, 1.0, window_aspect_ratio_fs > image_aspect_ratio_fs);
    let render_width_fs = scale_x_fs * uniforms.window_size.x;
    let render_height_fs = scale_y_fs * uniforms.window_size.y;
    let x_offset_fs = (uniforms.window_size.x - render_width_fs) / 2.0;
    let y_offset_fs = (uniforms.window_size.y - render_height_fs) / 2.0;
    let cursor_screen_x = x_offset_fs + uniforms.cursor_x * render_width_fs;
    let cursor_screen_y = y_offset_fs + uniforms.cursor_y * render_height_fs;
    let in_original_region = uniforms.comparison_mode <= 0.5 || in.tex_coords.y < uniforms.cursor_y;
    if (both_shown > 0.5 && uniforms.show_split_line > 0.5 && in_original_region && abs(in.clip_position.x - cursor_screen_x) < 1.0) {
        return vec4<f32>(1.0, 1.0, 1.0, alpha); // White color for the lines
    }
    if (uniforms.comparison_mode > 0.5 && uniforms.show_split_line > 0.5 && abs(in.clip_position.y - cursor_screen_y) < 1.0) {
        return vec4<f32>(1.0, 1.0, 1.0, alpha); // White color for the lines
    }

    return vec4<f32>(final_color.rgb, final_color.a * alpha);
}
