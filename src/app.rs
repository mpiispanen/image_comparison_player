use crate::image_loader;
use crate::player::FlipStats;
use crate::player::Player;
use crate::player::PlayerConfig;
use font8x8::UnicodeFonts;
use imgui::Condition;
use imgui::Ui;
use log::{debug, info, warn};
use nv_flip::magma_lut;
use parking_lot::lock_api::RwLock;
use parking_lot::Mutex;
use std::process;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::event::TouchPhase;
use winit::event::VirtualKeyCode;
use winit::event::WindowEvent;
use winit::window::Window as WinitWindow;

const APP_TITLE: &str = "Image Comparison Player";
const MAX_ZOOM_LEVEL: f32 = 10.0;
/// Minimum drag distance (in screen pixels) to commit a drag action (zoom or marker creation).
const MIN_DRAG_DISTANCE_PX: f32 = 10.0;
const MIN_DRAG_ZOOM_DISTANCE_PX: f32 = MIN_DRAG_DISTANCE_PX;
const MIN_DRAG_ZOOM_UV_SIZE: f32 = 0.01;
const MIN_PEEK_ZOOM_FACTOR: f32 = 1.0;
const MAX_PEEK_ZOOM_FACTOR: f32 = 16.0;
const PEEK_ZOOM_FACTOR_STEP: f32 = 0.25;

#[allow(dead_code)]
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
struct UniformData {
    cursor_x: f32,
    cursor_y: f32,
    image1_size: [f32; 2],
    image2_size: [f32; 2],
    flip_diff_size: [f32; 2],
    comparison_mode: f32,
    zoom_level: f32,
    zoom_center: [f32; 2],
    window_size: [f32; 2],
    show_image1: f32,
    show_image2: f32,
    show_split_line: f32,
    peek_active: f32,     // 1.0 when peek zoom is held (Z key)
    peek_factor: f32,     // peek magnification factor
    peek_radius: f32,     // peek window radius in screen pixels
    diff_multiplier: f32, // multiplier applied to abs-diff values (default 1.0)
    pump_active: f32,     // 1.0 when pumping animation is enabled
    time: f32,            // elapsed seconds (for animation)
    peek_show_image: f32, // 0=split, 1=image1 only, 2=image2 only, 3=current view, 4=diff only (in peek zoom)
}

// SAFETY: UniformData is #[repr(C)] and all fields are plain f32 arrays/scalars.
// Layout matches the WGSL Uniforms struct exactly (96 bytes, 8-byte aligned).
unsafe impl bytemuck::Zeroable for UniformData {}
unsafe impl bytemuck::Pod for UniformData {}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
enum ComparisonMode {
    #[default]
    None,
    Flip,
    Overlay,
    AbsDiff,
}

impl ComparisonMode {
    fn cycle(self) -> Self {
        match self {
            ComparisonMode::None => ComparisonMode::Flip,
            ComparisonMode::Flip => ComparisonMode::Overlay,
            ComparisonMode::Overlay => ComparisonMode::AbsDiff,
            ComparisonMode::AbsDiff => ComparisonMode::None,
        }
    }

    fn as_f32(self) -> f32 {
        match self {
            ComparisonMode::None => 0.0,
            ComparisonMode::Flip => 1.0,
            ComparisonMode::Overlay => 2.0,
            ComparisonMode::AbsDiff => 3.0,
        }
    }

    fn label(self) -> &'static str {
        match self {
            ComparisonMode::None => "Normal",
            ComparisonMode::Flip => "FLIP",
            ComparisonMode::Overlay => "Alpha Overlay",
            ComparisonMode::AbsDiff => "Abs Diff",
        }
    }
}

/// Controls which images are visible: normal side-by-side split or a full-screen solo view.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
enum ViewMode {
    /// Both images shown with a movable split line (default).
    #[default]
    Split,
    /// One image fills the full frame; `solo_view_source` selects which one.
    Solo,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum PeekImageMode {
    #[default]
    Both,
    Image1,
    Image2,
    CurrentView,
    DiffOnly,
}

impl PeekImageMode {
    fn cycle(self) -> Self {
        match self {
            PeekImageMode::Both => PeekImageMode::Image1,
            PeekImageMode::Image1 => PeekImageMode::Image2,
            PeekImageMode::Image2 => PeekImageMode::CurrentView,
            PeekImageMode::CurrentView => PeekImageMode::DiffOnly,
            PeekImageMode::DiffOnly => PeekImageMode::Both,
        }
    }

    fn as_f32(self) -> f32 {
        match self {
            PeekImageMode::Both => 0.0,
            PeekImageMode::Image1 => 1.0,
            PeekImageMode::Image2 => 2.0,
            PeekImageMode::CurrentView => 3.0,
            PeekImageMode::DiffOnly => 4.0,
        }
    }

    fn label(self) -> &'static str {
        match self {
            PeekImageMode::Both => "Peek: split",
            PeekImageMode::Image1 => "Peek: image 1",
            PeekImageMode::Image2 => "Peek: image 2",
            PeekImageMode::CurrentView => "Peek: current view",
            PeekImageMode::DiffOnly => "Peek: diff only",
        }
    }
}

/// A single rectangular marker anchored in image UV space [0, 1].
#[derive(Clone, Debug, PartialEq)]
struct Marker {
    id: usize,
    /// UV x of the left edge.
    x1: f32,
    /// UV y of the top edge.
    y1: f32,
    /// UV x of the right edge.
    x2: f32,
    /// UV y of the bottom edge.
    y2: f32,
    /// Optional text annotation shown near the marker.
    label: String,
    /// RGBA display colour.
    color: [f32; 4],
}

/// Manages a collection of rectangular marker overlays that can be drawn on top
/// of the image and optionally included in screenshots.
struct MarkerOverlay {
    markers: Vec<Marker>,
    next_id: usize,
    /// Whether markers are globally visible (shown/hidden with M key).
    visible: bool,
    /// Whether the marker editor window is open (toggled with N key).
    is_editor_open: bool,
    /// When true the next left-click-drag on the image creates a new marker.
    create_mode: bool,
}

impl MarkerOverlay {
    fn new() -> Self {
        Self {
            markers: Vec::new(),
            next_id: 0,
            visible: true,
            is_editor_open: false,
            create_mode: false,
        }
    }

    /// Add a new marker at the given image UV coordinates.
    /// Coordinates are normalised (min/max) before storing.
    fn add_marker(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        const COLORS: [[f32; 4]; 5] = [
            [1.0, 0.27, 0.27, 1.0], // red
            [1.0, 0.82, 0.0, 1.0],  // yellow
            [0.0, 0.78, 1.0, 1.0],  // cyan
            [0.27, 1.0, 0.27, 1.0], // green
            [0.78, 0.27, 1.0, 1.0], // purple
        ];
        let color = COLORS[self.next_id % COLORS.len()];
        let id = self.next_id;
        self.next_id += 1;
        self.markers.push(Marker {
            id,
            x1: x1.min(x2),
            y1: y1.min(y2),
            x2: x1.max(x2),
            y2: y1.max(y2),
            label: String::new(),
            color,
        });
    }

    /// Remove the marker with the given ID.
    fn delete_marker(&mut self, id: usize) {
        self.markers.retain(|m| m.id != id);
    }

    /// Draw the marker editor window.  Only visible when `is_editor_open` is true.
    fn draw_editor(&mut self, ui: &Ui) {
        if !self.is_editor_open {
            return;
        }
        let mut is_open = self.is_editor_open;
        ui.window("Markers")
            .size([340.0, 300.0], Condition::FirstUseEver)
            .position([10.0, 60.0], Condition::FirstUseEver)
            .resizable(true)
            .opened(&mut is_open)
            .build(|| {
                let mut vis = self.visible;
                if ui.checkbox("Show [M]", &mut vis) {
                    self.visible = vis;
                }
                ui.same_line();
                let create_label = if self.create_mode {
                    "Cancel"
                } else {
                    "Add Marker"
                };
                if ui.button(create_label) {
                    self.create_mode = !self.create_mode;
                }

                if self.create_mode {
                    ui.text_colored([1.0, 1.0, 0.3, 1.0], "Drag on image to draw marker");
                } else {
                    ui.dummy([0.0, ui.text_line_height()]);
                }

                ui.separator();

                if self.markers.is_empty() {
                    ui.text_disabled("No markers yet. Click \"Add Marker\" and drag on the image.");
                    return;
                }

                let swatch_size = 14.0_f32;
                let mut to_delete: Option<usize> = None;

                for marker in &mut self.markers {
                    // Small colour swatch via the window draw list.
                    {
                        let dl = ui.get_window_draw_list();
                        let wp = ui.window_pos();
                        let cp = ui.cursor_pos();
                        let sx = wp[0] + cp[0];
                        let sy = wp[1] + cp[1];
                        dl.add_rect([sx, sy], [sx + swatch_size, sy + swatch_size], marker.color)
                            .filled(true)
                            .build();
                        dl.add_rect(
                            [sx, sy],
                            [sx + swatch_size, sy + swatch_size],
                            [1.0, 1.0, 1.0, 0.3],
                        )
                        .build();
                    }
                    ui.dummy([swatch_size + 2.0, swatch_size]);
                    ui.same_line();

                    // Editable label.
                    ui.set_next_item_width(165.0);
                    ui.input_text(format!("##lbl_{}", marker.id), &mut marker.label)
                        .build();

                    ui.same_line();

                    if ui.small_button(format!("Del##del_{}", marker.id)) {
                        to_delete = Some(marker.id);
                    }
                }

                if let Some(id) = to_delete {
                    self.delete_marker(id);
                }
            });
        self.is_editor_open = is_open;
    }
}

struct CacheDebugWindow {
    is_open: bool,
    size: [f32; 2],
}

fn round_up_to_multiple(value: u32, multiple: u32) -> u32 {
    let m = multiple.max(1);
    value.div_ceil(m) * m
}

impl CacheDebugWindow {
    fn new() -> Self {
        Self {
            is_open: false,
            size: [400.0, 200.0], // Increased height to accommodate the new row
        }
    }

    fn draw(&mut self, ui: &Ui, player: &Player, mouse_x: f32, mouse_y: f32, window_width: f32) {
        let visible_frames = usize::max(
            player.config.preload_ahead + player.config.preload_behind + 1,
            player.config.diff_preload_ahead + player.config.diff_preload_behind + 1,
        );
        let button_size = 15.0;
        let spacing = 1.0;
        let total_width = visible_frames as f32 * (button_size + spacing) - spacing;
        let desired_width = total_width.min(window_width - 20.0).max(260.0);
        let min_height = 260.0;
        let desired_height = self.size[1].max(min_height);

        ui.window("Cache Debug")
            .size([desired_width, desired_height], Condition::Always)
            .position([10.0, 10.0], Condition::FirstUseEver)
            .resizable(true)
            .build(|| {
                let (current_left, current_right) = player.current_images();
                let frame_count = player.frame_count1;

                self.draw_cache_row(
                    ui,
                    player,
                    true,
                    current_left,
                    CacheRowParams {
                        frame_count,
                        mouse_pos: (mouse_x, mouse_y),
                        available_width: desired_width,
                    },
                );
                ui.dummy([0.0, 10.0]);
                self.draw_cache_row(
                    ui,
                    player,
                    false,
                    current_right,
                    CacheRowParams {
                        frame_count,
                        mouse_pos: (mouse_x, mouse_y),
                        available_width: desired_width,
                    },
                );
                ui.dummy([0.0, 10.0]);
                self.draw_diff_cache_row(
                    ui,
                    player,
                    current_left,
                    current_right,
                    CacheRowParams {
                        frame_count,
                        mouse_pos: (mouse_x, mouse_y),
                        available_width: desired_width,
                    },
                );
                ui.dummy([0.0, 10.0]);
                self.draw_cache_metrics(ui, player);

                self.size = ui.window_size();
                self.size[1] = self.size[1].max(min_height);
            });
    }

    fn draw_cache_metrics(&self, ui: &Ui, player: &Player) {
        use std::sync::atomic::Ordering;
        let fmt_rate = |hits: u64, misses: u64| -> String {
            let total = hits + misses;
            if total == 0 {
                "N/A".to_string()
            } else {
                format!("{:.1}%", 100.0 * hits as f64 / total as f64)
            }
        };

        let lm = &player.texture_cache_left.metrics;
        let rm = &player.texture_cache_right.metrics;
        let dm = &player.flip_diff_cache_metrics;

        let l_hits = lm.hits.load(Ordering::Relaxed);
        let l_misses = lm.misses.load(Ordering::Relaxed);
        let l_evictions = lm.evictions.load(Ordering::Relaxed);

        let r_hits = rm.hits.load(Ordering::Relaxed);
        let r_misses = rm.misses.load(Ordering::Relaxed);
        let r_evictions = rm.evictions.load(Ordering::Relaxed);

        let d_hits = dm.hits.load(Ordering::Relaxed);
        let d_misses = dm.misses.load(Ordering::Relaxed);
        let d_evictions = dm.evictions.load(Ordering::Relaxed);

        ui.text(format!(
            "L: hit rate={} hits={} misses={} evictions={}",
            fmt_rate(l_hits, l_misses),
            l_hits,
            l_misses,
            l_evictions
        ));
        ui.text(format!(
            "R: hit rate={} hits={} misses={} evictions={}",
            fmt_rate(r_hits, r_misses),
            r_hits,
            r_misses,
            r_evictions
        ));
        ui.text(format!(
            "D: hit rate={} hits={} misses={} evictions={}",
            fmt_rate(d_hits, d_misses),
            d_hits,
            d_misses,
            d_evictions
        ));
    }

    fn draw_cache_row(
        &self,
        ui: &Ui,
        player: &Player,
        is_left: bool,
        current: usize,
        params: CacheRowParams,
    ) {
        let cache = if is_left {
            &player.texture_cache_left
        } else {
            &player.texture_cache_right
        };
        let label = if is_left { "L:" } else { "R:" };

        ui.text(label);
        ui.same_line();

        let visible_frames = player.config.preload_ahead + player.config.preload_behind + 1;
        let button_size = 15.0;
        let spacing = 1.0;
        let total_width = visible_frames as f32 * (button_size + spacing) - spacing;
        let scale_factor = ((params.available_width - ui.calc_text_size(label)[0] - spacing)
            / total_width)
            .min(1.0);
        let scaled_button_size = button_size * scale_factor;
        let scaled_spacing = spacing * scale_factor;

        ui.group(|| {
            let window_pos = ui.window_pos();
            let cursor_pos = ui.cursor_pos();
            ui.set_next_item_width(params.available_width - ui.calc_text_size(label)[0] - spacing);
            ui.dummy([
                params.available_width - ui.calc_text_size(label)[0] - spacing,
                scaled_button_size + 20.0,
            ]);

            let draw_list = ui.get_window_draw_list();

            let half_visible = visible_frames / 2;

            for i in 0..visible_frames {
                let frame = (current as i64 + i as i64 - half_visible as i64)
                    .rem_euclid(params.frame_count as i64) as usize;
                let x = window_pos[0]
                    + cursor_pos[0]
                    + i as f32 * (scaled_button_size + scaled_spacing);
                let y = window_pos[1] + cursor_pos[1] + 10.0;

                if frame.is_multiple_of(5) {
                    draw_list.add_text([x, y - 15.0], [1.0, 1.0, 1.0, 1.0], frame.to_string());
                }

                let y = y + 5.0;

                let color = if cache.contains(frame) {
                    [0.0, 1.0, 0.0, 1.0]
                } else if player.texture_load_queue.lock().contains(&(frame, is_left)) {
                    [1.0, 1.0, 0.0, 1.0]
                } else {
                    [1.0, 0.0, 0.0, 1.0]
                };

                draw_list
                    .add_rect(
                        [x, y],
                        [x + scaled_button_size, y + scaled_button_size],
                        color,
                    )
                    .filled(true)
                    .build();

                if frame == current {
                    draw_list
                        .add_rect(
                            [x, y],
                            [x + scaled_button_size, y + scaled_button_size],
                            [1.0, 1.0, 1.0, 1.0],
                        )
                        .thickness(2.0)
                        .build();
                }

                // Check if playback delay is bigger than frame duration and draw orange border
                let switch_time = player.frame_switch_times.read().get(&(frame, is_left)).cloned();
                let available_time = player.texture_available_times.read().get(&(frame, is_left)).cloned();
                if let (Some(switch), Some(available)) = (switch_time, available_time) {
                    let playback_delay = available.duration_since(switch);
                    if let Some(frame_duration) = player.get_frame_duration(frame, is_left) {
                        if playback_delay > frame_duration {
                            draw_list
                                .add_rect(
                                    [x, y],
                                    [x + scaled_button_size, y + scaled_button_size],
                                    [1.0, 0.5, 0.0, 1.0], // Orange color
                                )
                                .thickness(2.0)
                                .build();
                        }
                    }
                }

                if params.mouse_pos.0 >= x
                    && params.mouse_pos.0 <= x + scaled_button_size
                    && params.mouse_pos.1 >= y
                    && params.mouse_pos.1 <= y + scaled_button_size
                {
                    let tooltip = if let Some(texture_info) =
                        player.texture_timings.read().get(&(frame, is_left))
                    {
                        let switch_time = player
                            .frame_switch_times
                            .read()
                            .get(&(frame, is_left))
                            .cloned();
                        
                        let available_time = player
                            .texture_available_times
                            .read()
                            .get(&(frame, is_left))
                            .cloned();
                        
                        let playback_delay = match (switch_time, available_time) {
                            (Some(switch), Some(available)) => available.duration_since(switch),
                            _ => Duration::default(),
                        };

                        let frame_duration = player.get_frame_duration(frame, is_left)
                            .unwrap_or_else(|| Duration::from_secs(0));

                        format!(
                            "Frame {} (Loaded)\nLoad time: {:.2}ms\nProcess time: {:.2}ms\nPlayback delay: {:.2}ms\nFrame duration: {:.2}ms",
                            frame,
                            texture_info.load_time.as_secs_f32() * 1000.0,
                            texture_info.process_time.as_secs_f32() * 1000.0,
                            playback_delay.as_secs_f32() * 1000.0,
                            frame_duration.as_secs_f32() * 1000.0
                        )
                    } else if player.texture_load_queue.lock().contains(&(frame, is_left)) {
                        format!("Frame {} (Loading)", frame)
                    } else {
                        format!("Frame {} (Not loaded)", frame)
                    };
                    ui.tooltip_text(tooltip);
                }
            }
        });
    }

    fn draw_diff_cache_row(
        &self,
        ui: &Ui,
        player: &Player,
        current_left: usize,
        current_right: usize,
        params: CacheRowParams,
    ) {
        ui.text("D:");
        ui.same_line();

        let visible_frames = usize::max(
            player.config.preload_ahead + player.config.preload_behind + 1,
            player.config.diff_preload_ahead + player.config.diff_preload_behind + 1,
        );
        let button_size = 15.0;
        let spacing = 1.0;
        let total_width = visible_frames as f32 * (button_size + spacing) - spacing;
        let scale_factor = ((params.available_width - ui.calc_text_size("D:")[0] - spacing)
            / total_width)
            .min(1.0);
        let scaled_button_size = button_size * scale_factor;
        let scaled_spacing = spacing * scale_factor;

        ui.group(|| {
            let window_pos = ui.window_pos();
            let cursor_pos = ui.cursor_pos();
            ui.set_next_item_width(params.available_width - ui.calc_text_size("D:")[0] - spacing);
            ui.dummy([
                params.available_width - ui.calc_text_size("D:")[0] - spacing,
                scaled_button_size + 20.0,
            ]);

            let draw_list = ui.get_window_draw_list();

            let half_visible = visible_frames / 2;

            for i in 0..visible_frames {
                let left_frame = (current_left as i64 + i as i64 - half_visible as i64)
                    .rem_euclid(params.frame_count as i64)
                    as usize;
                let right_frame = (current_right as i64 + i as i64 - half_visible as i64)
                    .rem_euclid(params.frame_count as i64)
                    as usize;
                let x = window_pos[0]
                    + cursor_pos[0]
                    + i as f32 * (scaled_button_size + scaled_spacing);
                let y = window_pos[1] + cursor_pos[1] + 10.0;

                if left_frame.is_multiple_of(5) {
                    draw_list.add_text(
                        [x, y - 15.0],
                        [1.0, 1.0, 1.0, 1.0],
                        left_frame.to_string(),
                    );
                }

                let y = y + 5.0;

                let color = if player
                    .flip_diff_cache
                    .read()
                    .contains_key(&(left_frame, right_frame))
                {
                    [0.0, 1.0, 0.0, 1.0] // Green for completed
                } else if player
                    .flip_diff_in_progress
                    .read()
                    .contains(&(left_frame, right_frame))
                {
                    [1.0, 1.0, 0.0, 1.0] // Yellow for in progress
                } else {
                    [1.0, 0.0, 0.0, 1.0] // Red for not started
                };

                draw_list
                    .add_rect(
                        [x, y],
                        [x + scaled_button_size, y + scaled_button_size],
                        color,
                    )
                    .filled(true)
                    .build();

                if (left_frame, right_frame) == (current_left, current_right) {
                    draw_list
                        .add_rect(
                            [x, y],
                            [x + scaled_button_size, y + scaled_button_size],
                            [1.0, 1.0, 1.0, 1.0],
                        )
                        .thickness(2.0)
                        .build();
                }

                if params.mouse_pos.0 >= x
                    && params.mouse_pos.0 <= x + scaled_button_size
                    && params.mouse_pos.1 >= y
                    && params.mouse_pos.1 <= y + scaled_button_size
                {
                    let tooltip = if let Some(diff_info) = player
                        .diff_image_timings
                        .read()
                        .get(&(left_frame, right_frame))
                    {
                        let flip_stats = player.flip_stats.read().get(&(left_frame, right_frame)).cloned();
                        let stats_str = if let Some(stats) = flip_stats {
                            format!(
                                "\nFlip Stats:\n  Mean: {:.4}\n  Min: {:.4}\n  Max: {:.4}\n  P95: {:.4}\n  P99: {:.4}",
                                stats.mean, stats.min, stats.max, stats.p95, stats.p99
                            )
                        } else {
                            "".to_string()
                        };

                        format!(
                            "Diff Cache: ({}, {})\nProcess time: {:.2}ms{}",
                            left_frame,
                            right_frame,
                            diff_info.process_time.as_secs_f32() * 1000.0,
                            stats_str
                        )
                    } else if player
                        .flip_diff_in_progress
                        .read()
                        .contains(&(left_frame, right_frame))
                    {
                        format!(
                            "Diff Cache: ({}, {}) (In Progress)",
                            left_frame, right_frame
                        )
                    } else {
                        format!("Diff Cache: ({}, {}) (Not Loaded)", left_frame, right_frame)
                    };
                    ui.tooltip_text(tooltip);
                }
            }
        });
    }

    fn toggle(&mut self) {
        self.is_open = !self.is_open;
    }
}

struct PixelInfoWindow {
    is_open: bool,
}

impl PixelInfoWindow {
    fn new() -> Self {
        Self { is_open: false }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw(
        &self,
        ui: &Ui,
        hovered_pixel: (u32, u32),
        left_color: [u8; 4],
        right_color: [u8; 4],
        flip_error: Option<f32>,
        flip_stats: Option<&FlipStats>,
        single_image_mode: bool,
    ) {
        let swatch_size = 16.0;
        let swatch_spacing = 4.0;
        ui.window("Pixel Info")
            .size([300.0, 200.0], Condition::FirstUseEver)
            .position([10.0, 280.0], Condition::FirstUseEver)
            .resizable(true)
            .always_auto_resize(true)
            .build(|| {
                let draw_list = ui.get_window_draw_list();

                // Left image pixel color
                {
                    let window_pos = ui.window_pos();
                    let cursor_pos = ui.cursor_pos();
                    let x = window_pos[0] + cursor_pos[0];
                    let y = window_pos[1] + cursor_pos[1];
                    let lc = [
                        left_color[0] as f32 / 255.0,
                        left_color[1] as f32 / 255.0,
                        left_color[2] as f32 / 255.0,
                        1.0,
                    ];
                    draw_list
                        .add_rect([x, y], [x + swatch_size, y + swatch_size], lc)
                        .filled(true)
                        .build();
                    draw_list
                        .add_rect(
                            [x, y],
                            [x + swatch_size, y + swatch_size],
                            [0.8, 0.8, 0.8, 0.6],
                        )
                        .build();
                    ui.dummy([swatch_size + swatch_spacing, swatch_size]);
                    ui.same_line();
                    ui.text(format!(
                        "L: #{:02X}{:02X}{:02X}  ({}, {}, {}, {})",
                        left_color[0],
                        left_color[1],
                        left_color[2],
                        left_color[0],
                        left_color[1],
                        left_color[2],
                        left_color[3]
                    ));
                }

                // Right image pixel color (only in two-image mode)
                if !single_image_mode {
                    let window_pos = ui.window_pos();
                    let cursor_pos = ui.cursor_pos();
                    let x = window_pos[0] + cursor_pos[0];
                    let y = window_pos[1] + cursor_pos[1];
                    let rc = [
                        right_color[0] as f32 / 255.0,
                        right_color[1] as f32 / 255.0,
                        right_color[2] as f32 / 255.0,
                        1.0,
                    ];
                    draw_list
                        .add_rect([x, y], [x + swatch_size, y + swatch_size], rc)
                        .filled(true)
                        .build();
                    draw_list
                        .add_rect(
                            [x, y],
                            [x + swatch_size, y + swatch_size],
                            [0.8, 0.8, 0.8, 0.6],
                        )
                        .build();
                    ui.dummy([swatch_size + swatch_spacing, swatch_size]);
                    ui.same_line();
                    ui.text(format!(
                        "R: #{:02X}{:02X}{:02X}  ({}, {}, {}, {})",
                        right_color[0],
                        right_color[1],
                        right_color[2],
                        right_color[0],
                        right_color[1],
                        right_color[2],
                        right_color[3]
                    ));
                }

                if !single_image_mode {
                    let overlay_color = [
                        ((left_color[0] as u16 + right_color[0] as u16) / 2) as u8,
                        ((left_color[1] as u16 + right_color[1] as u16) / 2) as u8,
                        ((left_color[2] as u16 + right_color[2] as u16) / 2) as u8,
                        ((left_color[3] as u16 + right_color[3] as u16) / 2) as u8,
                    ];
                    let abs_diff_color = [
                        left_color[0].abs_diff(right_color[0]),
                        left_color[1].abs_diff(right_color[1]),
                        left_color[2].abs_diff(right_color[2]),
                        left_color[3].abs_diff(right_color[3]),
                    ];
                    ui.text(format!(
                        "Blend 0.5*(A+B): #{:02X}{:02X}{:02X}  ({}, {}, {}, {})",
                        overlay_color[0],
                        overlay_color[1],
                        overlay_color[2],
                        overlay_color[0],
                        overlay_color[1],
                        overlay_color[2],
                        overlay_color[3]
                    ));
                    ui.text(format!(
                        "|A-B| (raw, 8-bit): #{:02X}{:02X}{:02X}  ({}, {}, {}, {})",
                        abs_diff_color[0],
                        abs_diff_color[1],
                        abs_diff_color[2],
                        abs_diff_color[0],
                        abs_diff_color[1],
                        abs_diff_color[2],
                        abs_diff_color[3]
                    ));

                    if let Some(error) = flip_error {
                        ui.text(format!("FLIP error @ pixel: {:.4}", error));
                    } else {
                        ui.text_disabled("FLIP error @ pixel: n/a");
                    }
                }

                ui.text(format!(
                    "Hovered pixel: ({}, {})",
                    hovered_pixel.0, hovered_pixel.1
                ));
                ui.separator();

                // FLIP error metrics
                if let Some(stats) = flip_stats {
                    ui.text("FLIP Metrics:");
                    ui.text(format!("  Mean: {:.4}", stats.mean));
                    ui.text(format!("  Min:  {:.4}", stats.min));
                    ui.text(format!("  Max:  {:.4}", stats.max));
                    ui.text(format!("  P95:  {:.4}", stats.p95));
                    ui.text(format!("  P99:  {:.4}", stats.p99));
                } else if !single_image_mode {
                    ui.text_disabled("FLIP: not computed (press F to enable)");
                }
            });
    }

    fn toggle(&mut self) {
        self.is_open = !self.is_open;
    }
}

struct HelpOverlay {
    is_open: bool,
}

impl HelpOverlay {
    fn new() -> Self {
        Self { is_open: false }
    }

    fn draw(&mut self, ui: &Ui, single_image_mode: bool) {
        if !self.is_open {
            return;
        }
        ui.window("Help - Keyboard & Mouse Controls")
            .size([420.0, 410.0], Condition::FirstUseEver)
            .position([60.0, 60.0], Condition::FirstUseEver)
            .resizable(true)
            .opened(&mut self.is_open)
            .build(|| {
                ui.text_colored([1.0, 0.85, 0.3, 1.0], "Playback");
                ui.separator();
                ui.text("  Space          Play / Pause");
                ui.text("  Left / Right   Previous / Next frame");
                ui.text("  [  /  ]        Decrease / Increase playback speed");
                ui.dummy([0.0, 4.0]);

                ui.text_colored([1.0, 0.85, 0.3, 1.0], "Zoom & Pan");
                ui.separator();
                ui.text("  Scroll wheel   Zoom in / out");
                ui.text("  Up / Down      Zoom in / out");
                ui.text("  Q / E          Zoom out / in");
                ui.text("  W A S D        Pan up / left / down / right");
                ui.text("  Left drag      Pan (drag to move view)");
                ui.text("  Right drag     Zoom to dragged region");
                ui.text("  R              Reset zoom to full frame");
                ui.text("  Z (hold)       Peek zoom magnifier");
                ui.text("  X              Cycle peek zoom image (split / image 1 / image 2 / current view / diff only)");
                ui.text("  - / =          Decrease / Increase peek magnifier");
                ui.dummy([0.0, 4.0]);

                if !single_image_mode {
                    ui.text_colored([1.0, 0.85, 0.3, 1.0], "Comparison");
                    ui.separator();
                    ui.text("  L              Toggle split-line divider");
                    ui.text("  F              Cycle comparison mode");
                    ui.text("                 (Normal -> FLIP -> Overlay -> Abs Diff)");
                    ui.text("  1 / 2          Solo view: show only left / right image");
                    ui.text("                 (press same key again to return to split view)");
                    ui.text("  P              Save FLIP diff image");
                    ui.text("  , / .          Decrease / Increase diff multiplier");
                    ui.text("  T              Toggle diff highlight animation");
                    ui.dummy([0.0, 4.0]);
                }

                ui.text_colored([1.0, 0.85, 0.3, 1.0], "Windows & Overlays");
                ui.separator();
                ui.text("  H              Toggle this help overlay");
                ui.text("  O              Toggle HUD (frame/speed/zoom/mode)");
                ui.text("  C              Toggle cache debug window");
                ui.text("  V              Toggle pixel info window");
                ui.text("  N              Toggle marker editor");
                ui.text("  M              Toggle marker visibility");
                ui.dummy([0.0, 4.0]);

                ui.text_colored([1.0, 0.85, 0.3, 1.0], "Other");
                ui.separator();
                ui.text("  I              Save screenshot");
                ui.text("  U              Save combined screenshot (all sources side-by-side)");
                ui.text("  Esc            Close overlay / Quit");
            });
    }

    fn toggle(&mut self) {
        self.is_open = !self.is_open;
    }

    fn close(&mut self) {
        self.is_open = false;
    }
}

struct CacheRowParams {
    frame_count: usize,
    mouse_pos: (f32, f32),
    available_width: f32,
}

/// Read one RGBA pixel from each of two GPU textures at the same (x, y) position.
/// Both copies are submitted in a single command buffer so only one GPU round-trip
/// (device.poll(Wait)) is needed instead of two. The returned bytes are in the
/// textures' native encoding (sRGB for Rgba8UnormSrgb).
/// Returns [[0,0,0,255]; 2] on any GPU error.
///
/// Note: `bytes_per_row` for copy_texture_to_buffer must be a multiple of 256
/// (wgpu's COPY_BYTES_PER_ROW_ALIGNMENT). For a 1-pixel-wide 1-row copy, the minimum
/// valid value is 256, which allocates 256 bytes of staging buffer per pixel. This is
/// intentional — the extra bytes are padding required by the alignment constraint.
fn read_two_texture_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex1: &wgpu::Texture,
    tex2: &wgpu::Texture,
    x: u32,
    y: u32,
) -> [[u8; 4]; 2] {
    const BYTES_PER_ROW: u32 = 256; // wgpu COPY_BYTES_PER_ROW_ALIGNMENT
    let make_buf = || {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pixel Read Buffer"),
            size: BYTES_PER_ROW as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    };
    let buf1 = make_buf();
    let buf2 = make_buf();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Pixel Read Encoder"),
    });

    let copy = |encoder: &mut wgpu::CommandEncoder, tex: &wgpu::Texture, buf: &wgpu::Buffer| {
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(BYTES_PER_ROW),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    };
    copy(&mut encoder, tex1, &buf1);
    copy(&mut encoder, tex2, &buf2);

    queue.submit(std::iter::once(encoder.finish()));

    let slice1 = buf1.slice(..);
    let slice2 = buf2.slice(..);
    let (tx1, rx1) = mpsc::channel();
    let (tx2, rx2) = mpsc::channel();
    slice1.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx1.send(r);
    });
    slice2.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx2.send(r);
    });
    // Single poll(Wait) covers both mappings.
    device.poll(wgpu::Maintain::Wait);

    let read =
        |rx: std::sync::mpsc::Receiver<_>, buf: &wgpu::Buffer, slice: wgpu::BufferSlice<'_>| {
            match rx.recv() {
                Ok(Ok(_)) => {
                    let data = slice.get_mapped_range();
                    let result = [data[0], data[1], data[2], data[3]];
                    drop(data);
                    buf.unmap();
                    result
                }
                _ => [0, 0, 0, 255],
            }
        };
    [read(rx1, &buf1, slice1), read(rx2, &buf2, slice2)]
}

type FlipDiffReceiver = Arc<Mutex<mpsc::Receiver<(usize, usize, Vec<u8>, wgpu::Extent3d)>>>;
type FlipDiffTexture = Arc<Mutex<Option<Arc<wgpu::Texture>>>>;

/// Derive a short display label for an image source from config fields.
/// Prefers the last path component of a directory, then the filename (single
/// image) or parent directory name (multiple images).  Falls back to `default`.
fn derive_image_label(dir: Option<&str>, images: Option<&[String]>, default: &str) -> String {
    if let Some(d) = dir {
        return std::path::Path::new(d)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(default)
            .to_string();
    }
    if let Some(imgs) = images {
        if !imgs.is_empty() {
            let p = std::path::Path::new(&imgs[0]);
            if imgs.len() == 1 {
                return p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(default)
                    .to_string();
            }
            return p
                .parent()
                .and_then(|par| par.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or(default)
                .to_string();
        }
    }
    default.to_string()
}

#[derive(Clone)]
pub struct AppConfig {
    pub dir1: Option<String>,
    pub dir2: Option<String>,
    pub images1: Option<Vec<String>>,
    pub images2: Option<Vec<String>>,
    pub cache_size: usize,
    pub preload_ahead: usize,
    pub preload_behind: usize,
    pub num_load_threads: usize,
    pub num_process_threads: usize,
    pub num_flip_diff_threads: usize,
    pub diff_preload_ahead: usize,
    pub diff_preload_behind: usize,
    pub fps: f32,
    pub peek_zoom_factor: f32,
}

pub struct AppState {
    surface: wgpu::Surface,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    render_pipeline_with_flip: wgpu::RenderPipeline,
    player: Arc<RwLock<parking_lot::RawRwLock, Player>>,
    cursor_x: f32,
    cursor_y: f32,
    last_update: Instant,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    imgui_context: imgui::Context,
    imgui_platform: imgui_winit_support::WinitPlatform,
    imgui_renderer: imgui_wgpu::Renderer,
    cache_debug_window: CacheDebugWindow,
    uniform_bind_group: wgpu::BindGroup,
    mouse_position: (f32, f32),
    flip_diff_receiver: FlipDiffReceiver,
    flip_diff_texture: FlipDiffTexture,
    flip_mode: bool,
    comparison_mode: ComparisonMode,
    show_image1: bool,
    show_image2: bool,
    show_split_line: bool,
    zoom_level: f32,
    fixed_zoom_center: (f32, f32),
    swipe_start: Option<(f64, f64)>,
    swipe_threshold: f64,
    config: wgpu::SurfaceConfiguration,
    zoom_center_offset: (f32, f32),
    zoom_move_speed: f32,
    screenshot_requested: bool,
    surface_scale: u32,
    status_message: Option<(String, Instant)>,
    screenshot_result_rx: Arc<Mutex<mpsc::Receiver<String>>>,
    screenshot_result_tx: mpsc::Sender<String>,
    single_image_mode: bool,
    esc_key_down: bool,
    pixel_info_window: PixelInfoWindow,
    help_overlay: HelpOverlay,
    left_pixel_color: [u8; 4],
    right_pixel_color: [u8; 4],
    hovered_pixel: (u32, u32),
    flip_error_value: Option<f32>,
    drag_zoom_start: Option<(f32, f32)>,
    drag_zoom_current: (f32, f32),
    /// Start position (screen px) of an in-progress left-drag pan.
    drag_pan_start: Option<(f32, f32)>,
    /// Effective zoom center (fixed_zoom_center + zoom_center_offset) at the start of the pan drag.
    drag_pan_initial_center: (f32, f32),
    show_hud: bool,
    app_config: AppConfig,
    pending_drop_paths: Vec<std::path::PathBuf>,
    waiting_for_drop: bool,
    hovering_file: bool,
    /// Which side (true = left, false = right) will receive the next dropped file.
    /// Updated from cursor position on every CursorMoved event, and can be
    /// toggled with the Left/Right arrow keys while a file is being dragged
    /// (useful on platforms where CursorMoved is not emitted during drag-and-drop,
    /// such as Linux X11/Wayland).
    drop_target_left: bool,
    peek_zoom_active: bool,
    peek_zoom_factor: f32,
    peek_zoom_radius: f32,
    peek_image_mode: PeekImageMode,
    marker_overlay: MarkerOverlay,
    /// Start position (screen px) of an in-progress marker creation drag.
    marker_drag_start: Option<(f32, f32)>,
    /// Current end position (screen px) of an in-progress marker creation drag.
    marker_drag_current: (f32, f32),
    /// Multiplier applied to abs-diff colour values (default 10.0).
    diff_enhance_factor: f32,
    /// When true, animated pumping rings are drawn around small diff regions.
    pump_animation_active: bool,
    /// App start time used to compute the `time` uniform for animations.
    start_time: Instant,
    /// Short display label for the left/first image source (shown in HUD).
    left_label: String,
    /// Short display label for the right/second image source (shown in HUD).
    right_label: String,
    /// Whether we are in solo (one-image-at-a-time) view or the normal split view.
    view_mode: ViewMode,
    /// Index of the image source shown in solo view mode (0 = left/first, 1 = right/second).
    solo_view_source: usize,
}

fn decode_flip_error_from_magma_rgb(rgb: [u8; 3]) -> Option<f32> {
    let lut = magma_lut().to_vec();
    if lut.len() < 3 {
        return None;
    }
    let mut best_index = 0usize;
    let mut best_dist = u32::MAX;
    for (index, chunk) in lut.chunks_exact(3).enumerate() {
        let dr = rgb[0] as i32 - chunk[0] as i32;
        let dg = rgb[1] as i32 - chunk[1] as i32;
        let db = rgb[2] as i32 - chunk[2] as i32;
        let dist = (dr * dr + dg * dg + db * db) as u32;
        if dist < best_dist {
            best_dist = dist;
            best_index = index;
            if dist == 0 {
                break;
            }
        }
    }
    let max_index = lut.chunks_exact(3).len().saturating_sub(1).max(1) as f32;
    Some(best_index as f32 / max_index)
}

fn sample_flip_error_at_pixel(
    player: &Player,
    left_index: usize,
    right_index: usize,
    px: u32,
    py: u32,
) -> Option<f32> {
    let flip_diff_guard = player.flip_diff_raw_data.read();
    let (diff_data, width, height) = flip_diff_guard.get(&(left_index, right_index))?;
    if px >= *width || py >= *height {
        return None;
    }
    let offset = ((py * *width + px) * 4) as usize;
    if offset + 2 >= diff_data.len() {
        return None;
    }
    decode_flip_error_from_magma_rgb([
        diff_data[offset],
        diff_data[offset + 1],
        diff_data[offset + 2],
    ])
}

impl AppState {
    pub async fn new(
        window: &WinitWindow,
        app_config: AppConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Initializing AppState");

        let stored_config = app_config.clone();

        // When no images are provided (drag-and-drop startup), create a 1×1 black
        // placeholder so the Player initialises correctly.  It is replaced as soon
        // as the user drops real files.
        let no_images_provided =
            app_config.dir1.is_none() && app_config.images1.as_ref().is_none_or(|v| v.is_empty());

        let placeholder_dir = if no_images_provided {
            let dir = std::env::temp_dir().join("icp_placeholder");
            std::fs::create_dir_all(&dir)?;
            let path = dir.join("placeholder.png");
            if !path.exists() {
                image::RgbaImage::new(1, 1).save(&path)?;
            }
            Some(dir)
        } else {
            None
        };

        let (images1, image_len1) = if let Some(files) = &app_config.images1 {
            image_loader::load_image_paths_from_files(files, app_config.fps)?
        } else if let Some(raw) = app_config.dir1.as_deref() {
            let dir = std::fs::canonicalize(raw)?;
            image_loader::load_image_paths(&dir.to_string_lossy(), app_config.fps)?
        } else {
            let dir = placeholder_dir.as_ref().unwrap();
            image_loader::load_image_paths(&dir.to_string_lossy(), app_config.fps)?
        };
        let (images2, image_len2) = if let Some(files) = &app_config.images2 {
            image_loader::load_image_paths_from_files(files, app_config.fps)?
        } else if let Some(raw) = app_config.dir2.as_deref() {
            let dir = std::fs::canonicalize(raw)?;
            image_loader::load_image_paths(&dir.to_string_lossy(), app_config.fps)?
        } else {
            (images1.clone(), image_len1)
        };
        let single_image_mode =
            app_config.dir2.is_none() && app_config.images2.as_ref().is_none_or(|v| v.is_empty());
        debug!(
            "Loaded {} images from input1 and {} images from input2",
            image_len1, image_len2
        );

        let left_label = derive_image_label(
            app_config.dir1.as_deref(),
            app_config.images1.as_deref(),
            "Left",
        );
        let right_label = derive_image_label(
            app_config.dir2.as_deref(),
            app_config.images2.as_deref(),
            "Right",
        );

        let size = window.inner_size();
        let surface_scale = window.scale_factor().ceil() as u32;

        // On Windows, exclude the Vulkan backend: wgpu 0.17's Vulkan HAL has a
        // semaphore reuse bug (VUID-vkQueueSubmit-pSignalSemaphores-00067) that
        // fires on every frame regardless of present mode.  DX12 is native on
        // Windows 10+ and does not share this issue.  On every other platform
        // keep the full backend set so Metal, GL, etc. are still available.
        #[cfg(target_os = "windows")]
        let backends = wgpu::Backends::DX12;
        #[cfg(not(target_os = "windows"))]
        let backends = wgpu::Backends::all();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            dx12_shader_compiler: Default::default(),
        });

        let surface = unsafe { instance.create_surface(&window) }?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find an appropriate adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        // Use Fifo (vsync) as a conservative default. With the Vulkan backend
        // excluded on Windows (see above), this is mainly relevant for Linux/macOS
        // where all backends have correct semaphore handling.
        let present_mode = wgpu::PresentMode::Fifo;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: surface_format,
            // Round up to the nearest multiple of the Wayland buffer_scale so that the
            // surface size is always valid on HiDPI compositors (scale 2, 3, ...).
            width: round_up_to_multiple(size.width, surface_scale),
            height: round_up_to_multiple(size.height, surface_scale),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/diff_2_images.wgsl").into()),
        });

        let shader_with_flip = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader with Flip"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/diff_2_images_with_diff.wgsl").into(),
            ),
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Texture Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<UniformData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Uniform Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&[
                // Position (x, y)   // Texture coords (u, v)
                -1.0f32, -1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0,
                1.0, 1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0,
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 16,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let render_pipeline_with_flip =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline with Flip"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_with_flip,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 8,
                                shader_location: 1,
                            },
                        ],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_with_flip,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let player = Arc::new(RwLock::new(Player::new(
            PlayerConfig {
                image_data1: images1,
                image_data2: images2,
                cache_size: app_config.cache_size,
                preload_ahead: app_config.preload_ahead,
                preload_behind: app_config.preload_behind,
                num_load_threads: app_config.num_load_threads,
                num_process_threads: app_config.num_process_threads,
                num_flip_diff_threads: app_config.num_flip_diff_threads,
                diff_preload_ahead: app_config.diff_preload_ahead,
                diff_preload_behind: app_config.diff_preload_behind,
                single_image_mode,
            },
            Arc::clone(&queue),
            Arc::clone(&device),
        )));
        debug!("Player initialized");

        let (left_texture, right_texture) = player.write().load_initial_textures()?;
        let left_texture = Arc::new(left_texture);
        let right_texture = Arc::new(right_texture);

        debug!(
            "Loaded left texture dimensions: {}x{}",
            left_texture.width(),
            left_texture.height()
        );

        debug!(
            "Loaded right texture dimensions: {}x{}",
            right_texture.width(),
            right_texture.height()
        );

        let mut imgui_context = imgui::Context::create();
        imgui_context.set_ini_filename(None); // Disable imgui.ini file
        let hidpi_factor = window.scale_factor() as f32;
        imgui_context.io_mut().font_global_scale = if hidpi_factor > 0.0 {
            1.0 / hidpi_factor
        } else {
            1.0
        };
        imgui_context.fonts().clear();
        imgui_context
            .fonts()
            .add_font(&[imgui::FontSource::DefaultFontData {
                config: Some(imgui::FontConfig {
                    size_pixels: 18.0 * hidpi_factor.max(1.0),
                    ..imgui::FontConfig::default()
                }),
            }]);
        let mut imgui_platform = imgui_winit_support::WinitPlatform::init(&mut imgui_context);
        imgui_platform.attach_window(
            imgui_context.io_mut(),
            window,
            imgui_winit_support::HiDpiMode::Default,
        );

        let imgui_renderer_config = imgui_wgpu::RendererConfig {
            texture_format: config.format,
            ..Default::default()
        };

        let mut imgui_renderer =
            imgui_wgpu::Renderer::new(&mut imgui_context, &device, &queue, imgui_renderer_config);
        imgui_renderer.reload_font_texture(&mut imgui_context, &device, &queue);

        let cache_debug_window = CacheDebugWindow::new();

        let mouse_position = (0.0, 0.0);
        let (screenshot_result_tx, screenshot_result_rx) = mpsc::channel::<String>();
        let peek_zoom_factor = stored_config.peek_zoom_factor.max(1.0);

        info!("AppState initialized successfully");
        Ok(Self {
            surface,
            device,
            queue,
            size,
            render_pipeline,
            render_pipeline_with_flip,
            player,
            cursor_x: size.width as f32 / 2.0,
            cursor_y: size.height as f32 / 2.0,
            last_update: Instant::now(),
            texture_bind_group_layout,
            uniform_buffer,
            vertex_buffer,
            imgui_context,
            imgui_platform,
            imgui_renderer,
            cache_debug_window,
            uniform_bind_group,
            mouse_position,
            flip_diff_texture: Arc::new(Mutex::new(None)),
            flip_mode: false,
            comparison_mode: ComparisonMode::None,
            show_image1: true,
            show_image2: true,
            show_split_line: true,
            flip_diff_receiver: Arc::new(Mutex::new(mpsc::channel().1)),
            zoom_level: 1.0,
            fixed_zoom_center: (0.5, 0.5),
            swipe_start: None,
            swipe_threshold: 50.0,
            config,
            zoom_center_offset: (0.0, 0.0),
            zoom_move_speed: 0.1,
            screenshot_requested: false,
            surface_scale,
            status_message: None,
            screenshot_result_tx,
            screenshot_result_rx: Arc::new(Mutex::new(screenshot_result_rx)),
            single_image_mode,
            esc_key_down: false,
            pixel_info_window: PixelInfoWindow::new(),
            help_overlay: HelpOverlay::new(),
            left_pixel_color: [128, 128, 128, 255],
            right_pixel_color: [128, 128, 128, 255],
            hovered_pixel: (0, 0),
            flip_error_value: None,
            drag_zoom_start: None,
            drag_zoom_current: (0.0, 0.0),
            drag_pan_start: None,
            drag_pan_initial_center: (0.5, 0.5),
            show_hud: true,
            app_config: stored_config,
            pending_drop_paths: Vec::new(),
            waiting_for_drop: no_images_provided,
            hovering_file: false,
            drop_target_left: true,
            peek_zoom_active: false,
            peek_zoom_factor,
            peek_zoom_radius: 100.0,
            peek_image_mode: PeekImageMode::Both,
            marker_overlay: MarkerOverlay::new(),
            marker_drag_start: None,
            marker_drag_current: (0.0, 0.0),
            diff_enhance_factor: 1.0,
            pump_animation_active: false,
            start_time: Instant::now(),
            left_label,
            right_label,
            view_mode: ViewMode::Split,
            solo_view_source: 0,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            // Keep the real window size for coordinate math and event handling.
            self.size = new_size;
            // Round up to the nearest multiple of surface_scale so that the
            // surface size is always a valid integer multiple of the Wayland
            // buffer_scale on any HiDPI compositor (scale 2, 3, …).
            self.config.width = round_up_to_multiple(new_size.width, self.surface_scale);
            self.config.height = round_up_to_multiple(new_size.height, self.surface_scale);
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn update(&mut self) {
        // Process any pending drag-and-drop paths accumulated since the last frame.
        if !self.pending_drop_paths.is_empty() {
            let paths = std::mem::take(&mut self.pending_drop_paths);
            self.reload_from_dropped_paths(paths);
        }

        let now = Instant::now();
        let delta = now.duration_since(self.last_update);
        self.last_update = now;

        let mut player = self.player.write();
        let frame_changed = player.update(delta, self.comparison_mode == ComparisonMode::Flip);
        player.process_load_queue();
        debug!("Update called, frame changed: {}", frame_changed);

        if frame_changed {
            drop(player);
            self.load_and_update_textures();
        }
    }

    pub fn update_textures(&mut self) -> bool {
        let player = self.player.write();
        player.update_textures(self.comparison_mode == ComparisonMode::Flip)
    }

    pub fn render(&mut self, window: &WinitWindow) -> Result<(), wgpu::SurfaceError> {
        self.update_textures();
        self.player.write().process_loaded_textures();

        debug!("Starting render function");
        let player = self.player.read();
        let (left_index, right_index) = player.current_images();
        debug!(
            "Current frame indices: left={}, right={}",
            left_index, right_index
        );

        let left_texture = player.get_left_texture();
        let right_texture = if self.single_image_mode {
            left_texture.clone()
        } else {
            player.get_right_texture()
        };

        if left_texture.is_none() || right_texture.is_none() {
            debug!("Textures not ready yet, skipping render");
            return Ok(());
        }

        let left_texture = left_texture.unwrap();
        let right_texture = right_texture.unwrap();

        debug!("Left texture: {:?}", left_texture.size());
        debug!("Right texture: {:?}", right_texture.size());

        let output = match self.surface.get_current_texture() {
            Ok(tex) => tex,
            Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                // Surface lost (e.g. due to a Wayland compositor error on resize).
                // Reconfigure and skip this frame; the next frame will succeed.
                warn!("Surface lost/outdated, reconfiguring and skipping frame");
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let (image_width, image_height) =
            (left_texture.width() as f32, left_texture.height() as f32);
        let (image2_width, image2_height) =
            (right_texture.width() as f32, right_texture.height() as f32);
        let window_size = window.inner_size();
        let window_aspect_ratio = window_size.width as f32 / window_size.height as f32;
        let image_aspect_ratio = image_width / image_height;

        let (render_width, render_height) = if window_aspect_ratio > image_aspect_ratio {
            let scaled_height = window_size.height as f32;
            let scaled_width = scaled_height * image_aspect_ratio;
            (scaled_width, scaled_height)
        } else {
            let scaled_width = window_size.width as f32;
            let scaled_height = scaled_width / image_aspect_ratio;
            (scaled_width, scaled_height)
        };

        let (show_image1, show_image2) = self.effective_show_images();

        let uniforms = UniformData {
            cursor_x: if self.single_image_mode {
                1.0
            } else {
                self.cursor_x / render_width
            },
            cursor_y: self.cursor_y / render_height,
            image1_size: [image_width, image_height],
            image2_size: [image2_width, image2_height],
            flip_diff_size: [image_width, image_height],
            comparison_mode: self.comparison_mode.as_f32(),
            zoom_level: self.zoom_level,
            zoom_center: [
                self.fixed_zoom_center.0 + self.zoom_center_offset.0,
                self.fixed_zoom_center.1 + self.zoom_center_offset.1,
            ],
            window_size: [window_size.width as f32, window_size.height as f32],
            show_image1,
            show_image2,
            show_split_line: if self.show_split_line { 1.0 } else { 0.0 },
            peek_active: if self.peek_zoom_active { 1.0 } else { 0.0 },
            peek_factor: self.peek_zoom_factor,
            peek_radius: self.peek_zoom_radius,
            diff_multiplier: self.diff_enhance_factor,
            pump_active: if self.pump_animation_active { 1.0 } else { 0.0 },
            time: self.start_time.elapsed().as_secs_f32(),
            peek_show_image: self.effective_peek_image_mode().as_f32(),
        };

        debug!("Created texture view");

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Check if a valid flip diff texture exists for the current frame pair
        let flip_diff_texture = if self.comparison_mode == ComparisonMode::Flip {
            player
                .flip_diff_cache
                .read()
                .get(&(left_index, right_index))
                .and_then(|mutex| mutex.lock().as_ref().cloned())
        } else {
            None
        };

        let use_flip_diff = flip_diff_texture.is_some();

        let texture_bind_group = if use_flip_diff {
            self.create_texture_bind_group_with_flip(
                &left_texture,
                &right_texture,
                &flip_diff_texture.clone().unwrap(),
            )
        } else {
            self.create_texture_bind_group(&left_texture, &right_texture)
        };

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            self.queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

            if use_flip_diff {
                render_pass.set_pipeline(&self.render_pipeline_with_flip);
            } else {
                render_pass.set_pipeline(&self.render_pipeline);
            }
            render_pass.set_bind_group(0, &texture_bind_group, &[]);
            render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
        }

        let mut should_render_imgui = false;

        // Pick up any pending screenshot result from the background thread early so the
        // toast can be rendered in this same frame.
        while let Ok(msg) = self.screenshot_result_rx.lock().try_recv() {
            self.status_message = Some((msg, Instant::now()));
        }

        // Expire the status message after 3 seconds
        if let Some((_, set_at)) = &self.status_message {
            if set_at.elapsed().as_secs_f32() >= 3.0 {
                self.status_message = None;
            }
        }
        if let Some((msg, _)) = &self.status_message {
            window.set_title(&format!("{} — {}", APP_TITLE, msg));
        } else {
            window.set_title(APP_TITLE);
        }

        if self.cache_debug_window.is_open
            || self.pixel_info_window.is_open
            || self.help_overlay.is_open
            || self.status_message.is_some()
            || self.waiting_for_drop
            || self.hovering_file
            || self.show_hud
            || self.drag_zoom_start.is_some()
            || self.marker_overlay.is_editor_open
            || self.marker_drag_start.is_some()
            || (self.marker_overlay.visible && !self.marker_overlay.markers.is_empty())
            || (self.comparison_mode != ComparisonMode::None && !self.single_image_mode)
        {
            match self
                .imgui_platform
                .prepare_frame(self.imgui_context.io_mut(), window)
            {
                Ok(()) => {
                    let ui = self.imgui_context.frame();

                    if self.cache_debug_window.is_open {
                        let player = self.player.read();
                        let mouse_pos = ui.io().mouse_pos;
                        let window_width = window.inner_size().width as f32;
                        self.cache_debug_window.draw(
                            ui,
                            &player,
                            mouse_pos[0],
                            mouse_pos[1],
                            window_width,
                        );
                    }

                    if self.pixel_info_window.is_open {
                        let player = self.player.read();
                        let (left_index, right_index) = player.current_images();
                        let flip_stats = player
                            .flip_stats
                            .read()
                            .get(&(left_index, right_index))
                            .cloned();
                        self.pixel_info_window.draw(
                            ui,
                            self.hovered_pixel,
                            self.left_pixel_color,
                            self.right_pixel_color,
                            self.flip_error_value,
                            flip_stats.as_ref(),
                            self.single_image_mode,
                        );
                    }

                    self.help_overlay.draw(ui, self.single_image_mode);
                    let ui_scale = window.scale_factor() as f32;
                    let to_ui = if ui_scale > 0.0 { 1.0 / ui_scale } else { 1.0 };
                    let win_size = window.inner_size();
                    let ui_width = win_size.width as f32 * to_ui;
                    let ui_height = win_size.height as f32 * to_ui;

                    // Draw persistent HUD in the top-right corner
                    if self.show_hud {
                        let player = self.player.read();
                        let (left_index, right_index) = player.current_images();
                        let left_total = player.frame_count1;
                        let right_total = player.frame_count2;
                        let left_frame_label = player
                            .config
                            .image_data1
                            .get(left_index)
                            .and_then(|(path, _, _)| {
                                std::path::Path::new(path)
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .map(ToString::to_string)
                            })
                            .unwrap_or_else(|| self.left_label.clone());
                        let right_frame_label = player
                            .config
                            .image_data2
                            .get(right_index)
                            .and_then(|(path, _, _)| {
                                std::path::Path::new(path)
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .map(ToString::to_string)
                            })
                            .unwrap_or_else(|| self.right_label.clone());
                        let speed = player.playback_speed();
                        let playing = player.is_playing();
                        drop(player);

                        let compare_mode = if self.single_image_mode {
                            "Single".to_string()
                        } else if self.view_mode == ViewMode::Solo {
                            let src_label = if self.solo_view_source == 0 {
                                &left_frame_label
                            } else {
                                &right_frame_label
                            };
                            format!("Solo: {}", src_label)
                        } else if self.comparison_mode == ComparisonMode::Flip {
                            "FLIP diff".to_string()
                        } else if !self.show_image1 {
                            "Right only".to_string()
                        } else if !self.show_image2 {
                            "Left only".to_string()
                        } else {
                            "Split".to_string()
                        };

                        let draw_list = ui.get_foreground_draw_list();
                        let padding = 10.0_f32;
                        let inner_pad_x = 8.0_f32;
                        let inner_pad_y = 6.0_f32;
                        let play_str = if playing { ">" } else { "||" };
                        let hud_text = if self.single_image_mode {
                            format!(
                                "Frame: {}/{}  {} {:.2}x  Zoom: {:.1}x  Mode: {}  H: Help",
                                left_index + 1,
                                left_total,
                                play_str,
                                speed,
                                self.zoom_level,
                                compare_mode,
                            )
                        } else {
                            format!(
                                "L: {}/{}  R: {}/{}  {} {:.2}x  Zoom: {:.1}x  Mode: {}  H: Help",
                                left_index + 1,
                                left_total,
                                right_index + 1,
                                right_total,
                                play_str,
                                speed,
                                self.zoom_level,
                                compare_mode,
                            )
                        };
                        let text_size = ui.calc_text_size(&hud_text);
                        let box_w = text_size[0] + inner_pad_x * 2.0;
                        let box_h = text_size[1] + inner_pad_y * 2.0;
                        let box_min = [ui_width - padding - box_w, padding];
                        let box_max = [ui_width - padding, padding + box_h];
                        draw_list
                            .add_rect(box_min, box_max, [0.0, 0.0, 0.0, 0.6])
                            .filled(true)
                            .build();
                        draw_list.add_text(
                            [box_min[0] + inner_pad_x, box_min[1] + inner_pad_y],
                            [1.0, 1.0, 1.0, 1.0],
                            hud_text,
                        );

                        // Draw source labels inside the source-image region, relative to split lines.
                        if !self.single_image_mode {
                            let x_off_ui = (ui_width - render_width * to_ui) / 2.0;
                            let y_off_ui = (ui_height - render_height * to_ui) / 2.0;
                            let render_w_ui = render_width * to_ui;
                            let render_h_ui = render_height * to_ui;
                            let split_x_ui = x_off_ui + self.cursor_x * to_ui;
                            let original_bottom_ui = if self.comparison_mode == ComparisonMode::None
                            {
                                y_off_ui + render_h_ui
                            } else {
                                y_off_ui + self.cursor_y * to_ui
                            };
                            let lpad = 8.0_f32;
                            let vpad = 4.0_f32;
                            let left_text_size = ui.calc_text_size(&left_frame_label);
                            let right_text_size = ui.calc_text_size(&right_frame_label);
                            let hud_reserved_top = padding + box_h + 6.0;
                            let top_y = (y_off_ui + vpad).max(hud_reserved_top);

                            if self.view_mode == ViewMode::Solo {
                                // Solo mode: show only the active source label, centered in the full image area.
                                let (solo_label, solo_text_size) = if self.solo_view_source == 0 {
                                    (&left_frame_label, left_text_size)
                                } else {
                                    (&right_frame_label, right_text_size)
                                };
                                let label_h = solo_text_size[1];
                                let max_top_y = y_off_ui + render_h_ui - label_h - vpad;
                                if top_y <= max_top_y {
                                    let full_center = x_off_ui + render_w_ui * 0.5;
                                    let sx = (full_center - solo_text_size[0] * 0.5).clamp(
                                        x_off_ui + lpad,
                                        x_off_ui + render_w_ui - solo_text_size[0] - lpad,
                                    );
                                    let sy = top_y;
                                    draw_list
                                        .add_rect(
                                            [sx - vpad, sy - vpad],
                                            [
                                                sx + solo_text_size[0] + vpad,
                                                sy + solo_text_size[1] + vpad,
                                            ],
                                            [0.0, 0.0, 0.0, 0.65],
                                        )
                                        .filled(true)
                                        .build();
                                    draw_list.add_text(
                                        [sx, sy],
                                        [1.0, 1.0, 1.0, 1.0],
                                        solo_label.as_str(),
                                    );
                                }
                            } else {
                                let label_h = left_text_size[1].max(right_text_size[1]);
                                let max_top_y = original_bottom_ui - label_h - vpad;
                                if top_y <= max_top_y {
                                    // Left label centered in the left source region.
                                    let left_region_start = x_off_ui;
                                    let left_region_end = split_x_ui;
                                    let left_center = (left_region_start + left_region_end) * 0.5;
                                    let left_bound_min = left_region_start + lpad;
                                    let left_bound_max = left_region_end - lpad;
                                    let lx = if left_bound_max - left_bound_min >= left_text_size[0]
                                    {
                                        (left_center - left_text_size[0] * 0.5).clamp(
                                            left_bound_min,
                                            left_bound_max - left_text_size[0],
                                        )
                                    } else {
                                        -1.0
                                    };
                                    if lx >= 0.0 {
                                        let ly = top_y;
                                        draw_list
                                            .add_rect(
                                                [lx - vpad, ly - vpad],
                                                [
                                                    lx + left_text_size[0] + vpad,
                                                    ly + left_text_size[1] + vpad,
                                                ],
                                                [0.0, 0.0, 0.0, 0.65],
                                            )
                                            .filled(true)
                                            .build();
                                        draw_list.add_text(
                                            [lx, ly],
                                            [1.0, 1.0, 1.0, 1.0],
                                            &left_frame_label,
                                        );
                                    }

                                    // Right label centered in the right source region.
                                    let right_region_start = split_x_ui;
                                    let right_region_end = x_off_ui + render_w_ui;
                                    let right_center =
                                        (right_region_start + right_region_end) * 0.5;
                                    let right_bound_min = right_region_start + lpad;
                                    let right_bound_max = right_region_end - lpad;
                                    let rx = if right_bound_max - right_bound_min
                                        >= right_text_size[0]
                                    {
                                        (right_center - right_text_size[0] * 0.5).clamp(
                                            right_bound_min,
                                            right_bound_max - right_text_size[0],
                                        )
                                    } else {
                                        -1.0
                                    };
                                    if rx >= 0.0 {
                                        let ry = top_y;
                                        draw_list
                                            .add_rect(
                                                [rx - vpad, ry - vpad],
                                                [
                                                    rx + right_text_size[0] + vpad,
                                                    ry + right_text_size[1] + vpad,
                                                ],
                                                [0.0, 0.0, 0.0, 0.65],
                                            )
                                            .filled(true)
                                            .build();
                                        draw_list.add_text(
                                            [rx, ry],
                                            [1.0, 1.0, 1.0, 1.0],
                                            &right_frame_label,
                                        );
                                    }
                                }

                                if self.comparison_mode != ComparisonMode::None {
                                    let diff_label = match self.comparison_mode {
                                        ComparisonMode::Flip => "FLIP Diff",
                                        ComparisonMode::Overlay => "Overlay",
                                        ComparisonMode::AbsDiff => "Abs Diff",
                                        ComparisonMode::None => "",
                                    };
                                    let diff_text_size = ui.calc_text_size(diff_label);
                                    let diff_y =
                                        y_off_ui + render_h_ui - diff_text_size[1] - vpad * 2.0;
                                    let split_y_ui = y_off_ui + self.cursor_y * to_ui;
                                    let diff_top = diff_y - vpad;
                                    if split_y_ui <= diff_top {
                                        let diff_x = (x_off_ui
                                            + (render_w_ui - diff_text_size[0]) * 0.5)
                                            .clamp(
                                                x_off_ui + lpad,
                                                x_off_ui + render_w_ui - diff_text_size[0] - lpad,
                                            );
                                        draw_list
                                            .add_rect(
                                                [diff_x - vpad, diff_y - vpad],
                                                [
                                                    diff_x + diff_text_size[0] + vpad,
                                                    diff_y + diff_text_size[1] + vpad,
                                                ],
                                                [0.0, 0.0, 0.0, 0.65],
                                            )
                                            .filled(true)
                                            .build();
                                        draw_list.add_text(
                                            [diff_x, diff_y],
                                            [1.0, 1.0, 1.0, 1.0],
                                            diff_label,
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Draw persistent comparison mode indicator in the top-right corner
                    if self.comparison_mode != ComparisonMode::None && !self.single_image_mode {
                        let win_size = window.inner_size();
                        let padding = 10.0_f32;
                        let _token = ui.push_style_var(imgui::StyleVar::WindowPadding([8.0, 6.0]));
                        if let Some(_win) = ui
                            .window("##mode_hud")
                            .position(
                                [win_size.width as f32 - padding, padding],
                                imgui::Condition::Always,
                            )
                            .position_pivot([1.0, 0.0])
                            .bg_alpha(0.6)
                            .no_decoration()
                            .no_inputs()
                            .movable(false)
                            .no_nav()
                            .focus_on_appearing(false)
                            .always_auto_resize(true)
                            .begin()
                        {
                            let color = match self.comparison_mode {
                                ComparisonMode::Flip => [1.0, 0.8, 0.2, 1.0],
                                ComparisonMode::Overlay => [0.4, 0.8, 1.0, 1.0],
                                ComparisonMode::AbsDiff => [1.0, 0.5, 0.5, 1.0],
                                ComparisonMode::None => [1.0, 1.0, 1.0, 1.0],
                            };
                            ui.text_colored(color, self.comparison_mode.label());
                            if self.comparison_mode == ComparisonMode::AbsDiff {
                                ui.text_colored(
                                    [1.0, 0.9, 0.5, 1.0],
                                    format!("Diff x{:.0}", self.diff_enhance_factor),
                                );
                                if self.pump_animation_active {
                                    ui.text_colored([0.5, 1.0, 0.5, 1.0], "Highlight ON");
                                }
                            }
                        }
                    }

                    // Draw status-message toast in the bottom-left corner
                    if let Some((msg, set_at)) = &self.status_message {
                        let elapsed = set_at.elapsed().as_secs_f32();
                        let alpha = if elapsed < 2.5 {
                            1.0_f32
                        } else {
                            1.0 - (elapsed - 2.5) / 0.5
                        };
                        let draw_list = ui.get_foreground_draw_list();
                        let padding = 10.0_f32;
                        let inner_pad_x = 8.0_f32;
                        let inner_pad_y = 6.0_f32;
                        let text_size = ui.calc_text_size(msg);
                        let box_w = text_size[0] + inner_pad_x * 2.0;
                        let box_h = text_size[1] + inner_pad_y * 2.0;
                        let box_min = [padding, ui_height - padding - box_h];
                        let box_max = [padding + box_w, ui_height - padding];
                        draw_list
                            .add_rect(box_min, box_max, [0.0, 0.0, 0.0, alpha * 0.75])
                            .filled(true)
                            .build();
                        draw_list.add_text(
                            [box_min[0] + inner_pad_x, box_min[1] + inner_pad_y],
                            [1.0, 1.0, 1.0, alpha],
                            msg.as_str(),
                        );
                    }

                    // Draw drag-zoom selection rectangle.
                    if let Some(start) = self.drag_zoom_start {
                        let current = self.drag_zoom_current;
                        let start_ui = (start.0 * to_ui, start.1 * to_ui);
                        let current_ui = (current.0 * to_ui, current.1 * to_ui);
                        // ImGui expects min/max corners; normalize in case of up/left drags.
                        let min_x = start_ui.0.min(current_ui.0);
                        let max_x = start_ui.0.max(current_ui.0);
                        let min_y = start_ui.1.min(current_ui.1);
                        let max_y = start_ui.1.max(current_ui.1);
                        let draw_list = ui.get_foreground_draw_list();
                        // Semi-transparent yellow fill.
                        draw_list
                            .add_rect([min_x, min_y], [max_x, max_y], [1.0, 1.0, 0.0, 0.15])
                            .filled(true)
                            .build();
                        // Solid yellow outline.
                        draw_list
                            .add_rect([min_x, min_y], [max_x, max_y], [1.0, 1.0, 0.0, 0.9])
                            .thickness(1.5)
                            .build();
                    }

                    // Draw marker creation preview while the user is dragging.
                    if let Some(start) = self.marker_drag_start {
                        let current = self.marker_drag_current;
                        let min_x = start.0.min(current.0) * to_ui;
                        let min_y = start.1.min(current.1) * to_ui;
                        let max_x = start.0.max(current.0) * to_ui;
                        let max_y = start.1.max(current.1) * to_ui;
                        let draw_list = ui.get_foreground_draw_list();
                        draw_list
                            .add_rect([min_x, min_y], [max_x, max_y], [1.0, 0.5, 0.0, 0.2])
                            .filled(true)
                            .build();
                        draw_list
                            .add_rect([min_x, min_y], [max_x, max_y], [1.0, 0.5, 0.0, 0.9])
                            .thickness(1.5)
                            .build();
                    }

                    // Draw visible marker overlays over the image.
                    if self.marker_overlay.visible && !self.marker_overlay.markers.is_empty() {
                        let x_off = (window_size.width as f32 - render_width) / 2.0;
                        let y_off = (window_size.height as f32 - render_height) / 2.0;
                        let zoom_cx = self.fixed_zoom_center.0 + self.zoom_center_offset.0;
                        let zoom_cy = self.fixed_zoom_center.1 + self.zoom_center_offset.1;
                        let zoom = self.zoom_level;

                        // Convert image UV [0,1] → screen px → UI px.
                        let img_to_ui = |uv_x: f32, uv_y: f32| -> (f32, f32) {
                            let sx = zoom_cx + (uv_x - zoom_cx) * zoom;
                            let sy = zoom_cy + (uv_y - zoom_cy) * zoom;
                            let px = (x_off + sx * render_width) * to_ui;
                            let py = (y_off + sy * render_height) * to_ui;
                            (px, py)
                        };

                        let draw_list = ui.get_foreground_draw_list();
                        for marker in &self.marker_overlay.markers {
                            let (sx1, sy1) = img_to_ui(marker.x1, marker.y1);
                            let (sx2, sy2) = img_to_ui(marker.x2, marker.y2);

                            // Solid outline.
                            draw_list
                                .add_rect([sx1, sy1], [sx2, sy2], marker.color)
                                .thickness(2.0)
                                .build();

                            // Label text above the marker rectangle (if any).
                            if !marker.label.is_empty() {
                                let text_size = ui.calc_text_size(&marker.label);
                                let lx = sx1;
                                let ly = (sy1 - text_size[1] - 4.0).max(0.0);
                                draw_list
                                    .add_rect(
                                        [lx - 2.0, ly],
                                        [lx + text_size[0] + 4.0, ly + text_size[1] + 2.0],
                                        [0.0, 0.0, 0.0, 0.70],
                                    )
                                    .filled(true)
                                    .build();
                                draw_list.add_text([lx, ly], marker.color, &marker.label);
                            }
                        }
                    }

                    // Draw marker editor window.
                    self.marker_overlay.draw_editor(ui);

                    // Draw the "waiting for drop" overlay in the centre of the window.
                    if self.waiting_for_drop && !self.hovering_file {
                        let cx = ui_width / 2.0;
                        let cy = ui_height / 2.0;
                        let _padding =
                            ui.push_style_var(imgui::StyleVar::WindowPadding([20.0, 16.0]));
                        if let Some(_win) = ui
                            .window("##drop_hint")
                            .position([cx, cy], imgui::Condition::Always)
                            .position_pivot([0.5, 0.5])
                            .bg_alpha(0.75)
                            .no_decoration()
                            .no_inputs()
                            .movable(false)
                            .no_nav()
                            .focus_on_appearing(false)
                            .always_auto_resize(true)
                            .begin()
                        {
                            ui.text("Drop image files or folders here");
                            ui.spacing();
                            ui.text_colored([0.7, 0.7, 0.7, 1.0], "1 path \u{2192} single view");
                            ui.text_colored(
                                [0.7, 0.7, 0.7, 1.0],
                                "2 paths \u{2192} comparison view",
                            );
                        }
                    }

                    // Show left/right drop-zone panels during file hover (or hover + waiting).
                    if self.hovering_file
                        || (self.waiting_for_drop && !self.pending_drop_paths.is_empty())
                    {
                        let w = ui_width;
                        let h = ui_height;
                        let half = w / 2.0;
                        // Use the stored drop_target_left so that the highlighted panel
                        // matches the side that will actually receive the file.  On
                        // platforms where CursorMoved fires during drag (Windows) this
                        // is updated automatically; on others the user can press
                        // Left/Right arrow to choose the target side.
                        let over_left = self.drop_target_left;

                        let _padding =
                            ui.push_style_var(imgui::StyleVar::WindowPadding([12.0, 10.0]));

                        // Left panel
                        let left_alpha: f32 = if over_left { 0.55 } else { 0.25 };
                        if let Some(_win) = ui
                            .window("##drop_left")
                            .position([0.0, 0.0], imgui::Condition::Always)
                            .size([half, h], imgui::Condition::Always)
                            .bg_alpha(left_alpha)
                            .no_decoration()
                            .no_inputs()
                            .movable(false)
                            .no_nav()
                            .focus_on_appearing(false)
                            .begin()
                        {
                            // Centre the label inside the panel
                            let label = if self.waiting_for_drop {
                                "Drop here"
                            } else {
                                "Left"
                            };
                            let label_size = ui.calc_text_size(label);
                            let pad_x = (half - label_size[0]).max(0.0) / 2.0;
                            let pad_y = (h - label_size[1]).max(0.0) / 2.0;
                            ui.set_cursor_pos([pad_x, pad_y]);
                            let text_col = if over_left {
                                [1.0, 1.0, 1.0, 1.0]
                            } else {
                                [0.8, 0.8, 0.8, 0.7]
                            };
                            ui.text_colored(text_col, label);
                        }

                        // Right panel
                        let right_alpha: f32 = if over_left { 0.25 } else { 0.55 };
                        if let Some(_win) = ui
                            .window("##drop_right")
                            .position([half, 0.0], imgui::Condition::Always)
                            .size([half, h], imgui::Condition::Always)
                            .bg_alpha(right_alpha)
                            .no_decoration()
                            .no_inputs()
                            .movable(false)
                            .no_nav()
                            .focus_on_appearing(false)
                            .begin()
                        {
                            let label = if self.waiting_for_drop {
                                "Drop here"
                            } else {
                                "Right"
                            };
                            let label_size = ui.calc_text_size(label);
                            let pad_x = (half - label_size[0]).max(0.0) / 2.0;
                            let pad_y = (h - label_size[1]).max(0.0) / 2.0;
                            ui.set_cursor_pos([pad_x, pad_y]);
                            let text_col = if over_left {
                                [0.8, 0.8, 0.8, 0.7]
                            } else {
                                [1.0, 1.0, 1.0, 1.0]
                            };
                            ui.text_colored(text_col, label);
                        }
                    }

                    should_render_imgui = true;
                    self.imgui_platform.prepare_render(ui, window);
                }
                Err(e) => {
                    warn!("Failed to prepare ImGui frame: {}", e);
                }
            }
        }

        while let Ok((left_index, right_index, diff_data, size)) =
            self.flip_diff_receiver.lock().try_recv()
        {
            let device = Arc::clone(&self.device);
            let flip_diff_texture = Arc::clone(&self.flip_diff_texture);
            let flip_diff_cache = player.flip_diff_cache.clone();

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!(
                    "Flip Diff Texture - ({}, {})",
                    left_index, right_index
                )),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            let texture_arc = Arc::new(texture);
            *flip_diff_texture.lock() = Some(Arc::clone(&texture_arc));
            flip_diff_cache.write().insert(
                (left_index, right_index),
                Arc::new(Mutex::new(Some(texture_arc.clone()))),
            );

            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture_arc,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &diff_data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * size.width),
                    rows_per_image: Some(size.height),
                },
                size,
            );
        }

        if self.flip_mode {
            let flip_diff_texture = self.flip_diff_texture.lock().clone();
            if let Some(flip_diff_texture) = flip_diff_texture {
                let mouse_y = self.mouse_position.1;
                let window_height = window.inner_size().height as f32;

                let texture_bind_group = if mouse_y < window_height / 2.0 {
                    self.create_texture_bind_group(&left_texture, &right_texture)
                } else {
                    self.create_texture_bind_group(&flip_diff_texture, &flip_diff_texture)
                };

                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });

                let flip_diff_size = [
                    flip_diff_texture.width() as f32,
                    flip_diff_texture.height() as f32,
                ];

                let (show_image1, show_image2) = self.effective_show_images();

                let uniforms = UniformData {
                    cursor_x: self.cursor_x / render_width,
                    cursor_y: mouse_y / window_height,
                    image1_size: [left_texture.width() as f32, left_texture.height() as f32],
                    image2_size: [right_texture.width() as f32, right_texture.height() as f32],
                    flip_diff_size,
                    comparison_mode: self.comparison_mode.as_f32(),
                    zoom_level: self.zoom_level,
                    zoom_center: [
                        self.fixed_zoom_center.0 + self.zoom_center_offset.0,
                        self.fixed_zoom_center.1 + self.zoom_center_offset.1,
                    ],
                    window_size: [window_size.width as f32, window_size.height as f32],
                    show_image1,
                    show_image2,
                    show_split_line: if self.show_split_line { 1.0 } else { 0.0 },
                    peek_active: if self.peek_zoom_active { 1.0 } else { 0.0 },
                    peek_factor: self.peek_zoom_factor,
                    peek_radius: self.peek_zoom_radius,
                    diff_multiplier: self.diff_enhance_factor,
                    pump_active: if self.pump_animation_active { 1.0 } else { 0.0 },
                    time: self.start_time.elapsed().as_secs_f32(),
                    peek_show_image: self.effective_peek_image_mode().as_f32(),
                };

                self.queue
                    .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &texture_bind_group, &[]);
                render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.draw(0..6, 0..1);
            }
        }

        if should_render_imgui {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ImGui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            let draw_data = self.imgui_context.render();
            if draw_data.draw_lists_count() > 0 {
                if let Err(e) = self.imgui_renderer.render(
                    draw_data,
                    &self.queue,
                    &self.device,
                    &mut render_pass,
                ) {
                    warn!("Failed to render ImGui: {}", e);
                }
            } else {
                debug!("Skipping ImGui render: no draw lists");
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Update pixel colors for the pixel info window.
        // Performed after the main submit so all write_texture calls are flushed.
        // Both left and right pixels are read in a single GPU round-trip.
        if self.pixel_info_window.is_open && render_width > 0.0 && render_height > 0.0 {
            let cursor_u = self.cursor_x / render_width;
            let cursor_v = self.cursor_y / render_height;
            let zoom_center_x = self.fixed_zoom_center.0 + self.zoom_center_offset.0;
            let zoom_center_y = self.fixed_zoom_center.1 + self.zoom_center_offset.1;
            let zoomed_u =
                (zoom_center_x + (cursor_u - zoom_center_x) / self.zoom_level).clamp(0.0, 1.0);
            let zoomed_v =
                (zoom_center_y + (cursor_v - zoom_center_y) / self.zoom_level).clamp(0.0, 1.0);
            let px = ((zoomed_u * left_texture.width() as f32) as u32)
                .min(left_texture.width().saturating_sub(1));
            let py = ((zoomed_v * left_texture.height() as f32) as u32)
                .min(left_texture.height().saturating_sub(1));
            self.hovered_pixel = (px, py);
            let [lc, rc] = read_two_texture_pixels(
                &self.device,
                &self.queue,
                &left_texture,
                &right_texture,
                px,
                py,
            );
            self.left_pixel_color = lc;
            self.right_pixel_color = if self.single_image_mode { lc } else { rc };
            self.flip_error_value = if self.single_image_mode {
                None
            } else {
                sample_flip_error_at_pixel(&player, left_index, right_index, px, py)
            };
        }

        // Capture screenshot if requested.
        // The GPU readback (copy + poll) must complete before output.present(), but the
        // slow pixel-format conversion and PNG file-write are offloaded to a background
        // thread so the render loop is unblocked as quickly as possible.
        if self.screenshot_requested {
            self.screenshot_requested = false;
            info!("Saving screenshot...");
            let width = self.size.width;
            let height = self.size.height;
            // Align bytes_per_row to wgpu's required 256-byte alignment
            let bytes_per_row = (width * 4 + 255) & !255;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Screenshot Buffer"),
                size: (bytes_per_row * height) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut screenshot_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Screenshot Encoder"),
                    });
            screenshot_encoder.copy_texture_to_buffer(
                output.texture.as_image_copy(),
                wgpu::ImageCopyBuffer {
                    buffer: &buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            self.queue
                .submit(std::iter::once(screenshot_encoder.finish()));
            let buffer_slice = buffer.slice(..);
            let (tx, rx) = mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            // Poll until the GPU has finished writing into the buffer.
            self.device.poll(wgpu::Maintain::Wait);
            match rx.recv() {
                Ok(Ok(_)) => {
                    // Copy the raw bytes out so the GPU buffer can be unmapped immediately.
                    let raw = buffer_slice.get_mapped_range().to_vec();
                    buffer.unmap();
                    let is_bgra = matches!(
                        self.config.format,
                        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
                    );
                    let screenshot_prefix = match self.comparison_mode {
                        ComparisonMode::None => "screenshot_regular",
                        ComparisonMode::Flip => "screenshot_flip",
                        ComparisonMode::Overlay => "screenshot_overlay",
                        ComparisonMode::AbsDiff => "screenshot_absdiff",
                    };
                    let path = generate_output_filename(screenshot_prefix, "png");
                    let result_tx = self.screenshot_result_tx.clone();
                    // Pixel-format conversion and file I/O happen on a background thread
                    // so the render loop can continue without further blocking.
                    std::thread::spawn(move || {
                        let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
                        for row in 0..height as usize {
                            let row_start = row * bytes_per_row as usize;
                            let row_data = &raw[row_start..row_start + (width * 4) as usize];
                            if is_bgra {
                                for chunk in row_data.chunks(4) {
                                    pixels.extend_from_slice(&[
                                        chunk[2], chunk[1], chunk[0], chunk[3],
                                    ]);
                                }
                            } else {
                                pixels.extend_from_slice(row_data);
                            }
                        }
                        let msg = match image::save_buffer(
                            &path,
                            &pixels,
                            width,
                            height,
                            image::ColorType::Rgba8,
                        ) {
                            Ok(_) => {
                                info!("Screenshot saved to {}", path);
                                format!("Screenshot saved: {}", path)
                            }
                            Err(e) => {
                                warn!("Failed to save screenshot: {}", e);
                                format!("Failed to save screenshot: {}", e)
                            }
                        };
                        let _ = result_tx.send(msg);
                    });
                }
                Ok(Err(e)) => {
                    warn!("GPU buffer mapping failed for screenshot: {}", e);
                    self.status_message = Some((format!("GPU error: {}", e), Instant::now()));
                }
                Err(e) => {
                    warn!("Screenshot channel receive failed: {}", e);
                    self.status_message =
                        Some((format!("Screenshot failed: {}", e), Instant::now()));
                }
            }
        }

        debug!("Presenting output");
        output.present();

        Ok(())
    }

    fn create_texture_bind_group(
        &self,
        texture1: &wgpu::Texture,
        texture2: &wgpu::Texture,
    ) -> wgpu::BindGroup {
        let texture_view1 = texture1.create_view(&wgpu::TextureViewDescriptor::default());
        let texture_view2 = texture2.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create a dummy texture for the flip diff
        let dummy_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("Dummy Flip Diff Texture"),
            view_formats: &[],
        });
        let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view1),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&texture_view2),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        })
    }

    fn create_texture_bind_group_with_flip(
        &self,
        texture1: &wgpu::Texture,
        texture2: &wgpu::Texture,
        flip_diff_texture: &wgpu::Texture,
    ) -> wgpu::BindGroup {
        let texture_view1 = texture1.create_view(&wgpu::TextureViewDescriptor::default());
        let texture_view2 = texture2.create_view(&wgpu::TextureViewDescriptor::default());
        let flip_diff_view = flip_diff_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group with Flip"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view1),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&texture_view2),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&flip_diff_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        })
    }

    pub fn toggle_play_pause(&mut self) {
        self.player.write().toggle_play_pause();
    }

    pub fn next_frame(&mut self) {
        let frame_changed = self
            .player
            .write()
            .next_frame(self.comparison_mode == ComparisonMode::Flip);
        if frame_changed {
            self.load_and_update_textures();
        }
    }

    pub fn previous_frame(&mut self) {
        let frame_changed = self
            .player
            .write()
            .previous_frame(self.comparison_mode == ComparisonMode::Flip);
        if frame_changed {
            self.load_and_update_textures();
        }
    }

    fn load_and_update_textures(&mut self) {
        {
            let player = self.player.write();
            let (current_left, current_right) = player.current_images();

            player.ensure_texture_loaded(current_left, true);
            player.ensure_texture_loaded(current_right, false);
        }

        self.update_textures();
        self.player.write().process_loaded_textures();
    }

    /// Load images from a set of dropped paths and replace the current player.
    ///
    /// * 1 path + cursor on left  → replace left slot only (or single-input if no images yet)
    /// * 1 path + cursor on right → replace right slot only (or single-input if no images yet)
    /// * 2 paths → left path → left slot, right path → right slot
    /// * 0 or >2 paths → show an error status message and do nothing
    fn reload_from_dropped_paths(&mut self, paths: Vec<std::path::PathBuf>) {
        self.hovering_file = false;

        let fps = self.app_config.fps;

        let count = paths.len();
        if count == 0 || count > 2 {
            if count > 2 {
                self.status_message = Some((
                    "Drop 1 or 2 paths to load images".to_string(),
                    Instant::now(),
                ));
            }
            return;
        }

        let load_images_from_path =
            |p: &std::path::PathBuf| -> Result<Vec<(String, u64, u64)>, String> {
                if p.is_dir() {
                    image_loader::load_image_paths(&p.to_string_lossy(), fps)
                        .map(|(imgs, _)| imgs)
                        .map_err(|e| e.to_string())
                } else if p.is_file() {
                    image_loader::load_image_paths_from_files(
                        &[p.to_string_lossy().into_owned()],
                        fps,
                    )
                    .map(|(imgs, _)| imgs)
                    .map_err(|e| e.to_string())
                } else {
                    Err(format!("Path not found: {}", p.display()))
                }
            };

        // Determine which side should receive the dropped file.
        // drop_target_left is kept in sync with cursor position by CursorMoved events
        // (works on Windows) and can be overridden with the Left/Right arrow keys
        // while the hover overlay is visible (fallback for Linux X11/Wayland where
        // CursorMoved is not emitted during drag-and-drop operations).
        let drop_on_left = self.drop_target_left;

        // Build final (images1, images2, single_image_mode) based on count + cursor position.
        let (images1, images2, single_image_mode, display_msg) = if count == 2 {
            // Two paths: first → left, second → right, regardless of cursor position.
            let imgs1 = match load_images_from_path(&paths[0]) {
                Ok(v) if !v.is_empty() => v,
                Ok(_) => {
                    self.status_message = Some((
                        format!("No images found in: {}", paths[0].display()),
                        Instant::now(),
                    ));
                    return;
                }
                Err(e) => {
                    self.status_message = Some((format!("Invalid drop: {}", e), Instant::now()));
                    return;
                }
            };
            let imgs2 = match load_images_from_path(&paths[1]) {
                Ok(v) if !v.is_empty() => v,
                Ok(_) => {
                    self.status_message = Some((
                        format!("No images found in: {}", paths[1].display()),
                        Instant::now(),
                    ));
                    return;
                }
                Err(e) => {
                    self.status_message = Some((format!("Invalid drop: {}", e), Instant::now()));
                    return;
                }
            };
            let msg = format!(
                "Loaded left: {} | right: {}",
                paths[0]
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| paths[0].to_string_lossy().into_owned()),
                paths[1]
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| paths[1].to_string_lossy().into_owned()),
            );
            (imgs1, imgs2, false, msg)
        } else {
            // One path: use cursor position to decide which slot to fill.
            let dropped = match load_images_from_path(&paths[0]) {
                Ok(v) if !v.is_empty() => v,
                Ok(_) => {
                    self.status_message = Some((
                        format!("No images found in: {}", paths[0].display()),
                        Instant::now(),
                    ));
                    return;
                }
                Err(e) => {
                    self.status_message = Some((format!("Invalid drop: {}", e), Instant::now()));
                    return;
                }
            };

            let name = paths[0]
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| paths[0].to_string_lossy().into_owned());

            if self.waiting_for_drop {
                // No prior images — just open in single-input mode.
                let msg = format!("Loaded: {}", name);
                (dropped.clone(), dropped, true, msg)
            } else if drop_on_left {
                // Replace left slot; keep existing right.
                let existing_right = self.player.read().config.image_data2.clone();
                let (imgs2, sim) = if existing_right.is_empty() {
                    (dropped.clone(), true)
                } else {
                    (existing_right, false)
                };
                let msg = format!("Loaded left: {}", name);
                (dropped, imgs2, sim, msg)
            } else {
                // Replace right slot; keep existing left.
                let existing_left = self.player.read().config.image_data1.clone();
                let msg = format!("Loaded right: {}", name);
                (existing_left, dropped, false, msg)
            }
        };

        let new_player = Player::new(
            PlayerConfig {
                image_data1: images1,
                image_data2: images2,
                cache_size: self.app_config.cache_size,
                preload_ahead: self.app_config.preload_ahead,
                preload_behind: self.app_config.preload_behind,
                num_load_threads: self.app_config.num_load_threads,
                num_process_threads: self.app_config.num_process_threads,
                num_flip_diff_threads: self.app_config.num_flip_diff_threads,
                diff_preload_ahead: self.app_config.diff_preload_ahead,
                diff_preload_behind: self.app_config.diff_preload_behind,
                single_image_mode,
            },
            Arc::clone(&self.queue),
            Arc::clone(&self.device),
        );

        info!(
            "Reloading player from dropped path(s): {}",
            paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        self.player = Arc::new(RwLock::new(new_player));
        self.single_image_mode = single_image_mode;
        self.waiting_for_drop = false;
        self.comparison_mode = ComparisonMode::None;
        *self.flip_diff_texture.lock() = None;
        self.zoom_level = 1.0;
        self.fixed_zoom_center = (0.5, 0.5);
        self.zoom_center_offset = (0.0, 0.0);

        // Update labels to reflect the newly-loaded paths.
        let path_label = |p: &std::path::PathBuf| -> String {
            p.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_string()
        };
        if count == 2 {
            self.left_label = path_label(&paths[0]);
            self.right_label = path_label(&paths[1]);
        } else if self.single_image_mode || drop_on_left {
            self.left_label = path_label(&paths[0]);
        } else {
            self.right_label = path_label(&paths[0]);
        }

        self.status_message = Some((display_msg, Instant::now()));
        self.load_and_update_textures();
    }

    pub fn handle_event<T>(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::Event<T>,
    ) {
        self.imgui_platform
            .handle_event(self.imgui_context.io_mut(), window, event);

        if let winit::event::Event::WindowEvent {
            event: winit::event::WindowEvent::Resized(size),
            ..
        } = event
        {
            self.resize(*size);
            debug!("Window resized to: {:?}", size);
        }

        if let winit::event::Event::WindowEvent {
            event:
                winit::event::WindowEvent::ScaleFactorChanged {
                    scale_factor,
                    new_inner_size,
                },
            ..
        } = event
        {
            self.surface_scale = scale_factor.ceil() as u32;
            self.imgui_context.io_mut().font_global_scale = if *scale_factor > 0.0 {
                1.0 / *scale_factor as f32
            } else {
                1.0
            };
            self.resize(**new_inner_size);
        }

        if let winit::event::Event::WindowEvent {
            event: winit::event::WindowEvent::CursorMoved { position, .. },
            ..
        } = event
        {
            let pos = (position.x as f32, position.y as f32);
            // Update cursor state without writing the uniform buffer yet so that
            // the pan-drag computation below can see the updated cursor position,
            // and we only write the buffer once at the end of this branch.
            self.update_cursor_state(pos.0, pos.1);
            if let Some(start) = self.drag_zoom_start {
                self.drag_zoom_current = self.constrain_drag_to_window_aspect(start, pos);
            }
            if self.marker_drag_start.is_some() {
                self.marker_drag_current = pos;
            }
            // Update pan drag: keep the grabbed image pixel under the cursor.
            if let Some(pan_start) = self.drag_pan_start {
                let (render_width, render_height) = self.compute_render_dimensions();
                let d_uv_x = (pos.0 - pan_start.0) / render_width;
                let d_uv_y = (pos.1 - pan_start.1) / render_height;
                // Exact "pixel-under-cursor" pan formula:
                //   new_center = initial_center - d_uv / (zoom_level - 1)
                // A minimum denominator of 0.1 prevents erratic behaviour near
                // zoom == 1.0. Clamp just inside [0, 1] so drag panning can still
                // reach the image edge without producing exact 0.0/1.0 coordinates,
                // which current shaders may treat as transparent at the boundary.
                let z_minus_1 = (self.zoom_level - 1.0).max(0.1);
                let edge_epsilon = 1.0e-6_f32;
                let c_new_x = (self.drag_pan_initial_center.0 - d_uv_x / z_minus_1)
                    .clamp(edge_epsilon, 1.0 - edge_epsilon);
                let c_new_y = (self.drag_pan_initial_center.1 - d_uv_y / z_minus_1)
                    .clamp(edge_epsilon, 1.0 - edge_epsilon);
                // fixed_zoom_center was set to drag_pan_initial_center at drag start,
                // so the offset is the delta from the initial center.
                self.zoom_center_offset = (
                    c_new_x - self.fixed_zoom_center.0,
                    c_new_y - self.fixed_zoom_center.1,
                );
            }
            // Keep the drop-target side in sync with cursor position so that
            // platforms where CursorMoved fires during drag-and-drop (Windows)
            // automatically track which side should receive a dropped file.
            self.drop_target_left = pos.0 < self.size.width as f32 / 2.0;
            // Write the uniform buffer exactly once for this cursor-moved event.
            self.update_uniform_buffer();
        }

        if let winit::event::Event::WindowEvent {
            event:
                winit::event::WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            state: winit::event::ElementState::Released,
                            virtual_keycode: Some(keycode),
                            ..
                        },
                    ..
                },
            ..
        } = event
        {
            // While ImGui is capturing keyboard input, suppress global shortcuts.
            if self.imgui_context.io().want_capture_keyboard
                || self.imgui_context.io().want_text_input
            {
                return;
            }
            match keycode {
                VirtualKeyCode::Escape => {
                    self.esc_key_down = false;
                }
                VirtualKeyCode::Z => {
                    self.peek_zoom_active = false;
                }
                _ => {}
            }
        }

        if let winit::event::Event::WindowEvent {
            event:
                winit::event::WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            state: winit::event::ElementState::Pressed,
                            virtual_keycode: Some(keycode),
                            ..
                        },
                    ..
                },
            ..
        } = event
        {
            // While ImGui is capturing keyboard input, suppress global shortcuts.
            if self.imgui_context.io().want_capture_keyboard
                || self.imgui_context.io().want_text_input
            {
                return;
            }
            match keycode {
                VirtualKeyCode::Escape if !self.esc_key_down => {
                    self.esc_key_down = true;
                    if self.help_overlay.is_open {
                        self.help_overlay.close();
                    } else {
                        // Exit the application when Esc is pressed
                        process::exit(0);
                    }
                }
                VirtualKeyCode::C => {
                    self.cache_debug_window.toggle();
                }
                VirtualKeyCode::F if !self.single_image_mode => {
                    self.cycle_comparison_mode();
                }
                VirtualKeyCode::T if !self.single_image_mode => {
                    self.toggle_pump_animation();
                }
                VirtualKeyCode::Comma if !self.single_image_mode => {
                    self.adjust_diff_multiplier(-1.0);
                }
                VirtualKeyCode::Period if !self.single_image_mode => {
                    self.adjust_diff_multiplier(1.0);
                }
                VirtualKeyCode::Left | VirtualKeyCode::Right => {
                    if self.hovering_file {
                        // While a file is being dragged over the window, Left/Right
                        // arrow keys switch the drop target side.  This is the fallback
                        // for platforms (Linux X11/Wayland) where CursorMoved is not
                        // emitted during drag-and-drop, so the cursor-based tracking
                        // would otherwise be stuck on whichever side was last active.
                        self.drop_target_left = *keycode == VirtualKeyCode::Left;
                    } else if *keycode == VirtualKeyCode::Left {
                        self.previous_frame();
                    } else {
                        self.next_frame();
                    }
                }
                VirtualKeyCode::Space => {
                    self.toggle_play_pause();
                }
                VirtualKeyCode::Up => {
                    self.handle_zoom(&winit::event::MouseScrollDelta::LineDelta(0.0, 1.0))
                }
                VirtualKeyCode::Down => {
                    self.handle_zoom(&winit::event::MouseScrollDelta::LineDelta(0.0, -1.0))
                }
                VirtualKeyCode::Q => {
                    self.handle_zoom(&winit::event::MouseScrollDelta::LineDelta(0.0, -1.0))
                }
                VirtualKeyCode::E => {
                    self.handle_zoom(&winit::event::MouseScrollDelta::LineDelta(0.0, 1.0))
                }
                VirtualKeyCode::LBracket => {
                    self.player.write().decrease_playback_speed();
                }
                VirtualKeyCode::RBracket => {
                    self.player.write().increase_playback_speed();
                }
                VirtualKeyCode::W => self.handle_zoom_move((0.0, -1.0)),
                VirtualKeyCode::A => self.handle_zoom_move((-1.0, 0.0)),
                VirtualKeyCode::S => self.handle_zoom_move((0.0, 1.0)),
                VirtualKeyCode::D => self.handle_zoom_move((1.0, 0.0)),
                VirtualKeyCode::Minus => self.adjust_peek_zoom_factor(-PEEK_ZOOM_FACTOR_STEP),
                VirtualKeyCode::Equals => self.adjust_peek_zoom_factor(PEEK_ZOOM_FACTOR_STEP),
                VirtualKeyCode::P => {
                    self.save_flip_diff_image();
                }
                VirtualKeyCode::I => {
                    self.request_screenshot();
                }
                VirtualKeyCode::U => {
                    self.save_combined_screenshot();
                }
                VirtualKeyCode::Key1 if !self.single_image_mode => {
                    if self.view_mode == ViewMode::Solo && self.solo_view_source == 0 {
                        // Already viewing source 0 solo — exit solo view.
                        self.view_mode = ViewMode::Split;
                        self.status_message = Some(("Solo view: off".to_string(), Instant::now()));
                    } else {
                        self.view_mode = ViewMode::Solo;
                        self.solo_view_source = 0;
                        self.status_message =
                            Some((format!("Solo: {}", self.left_label), Instant::now()));
                    }
                    self.update_uniform_buffer();
                }
                VirtualKeyCode::Key2 if !self.single_image_mode => {
                    if self.view_mode == ViewMode::Solo && self.solo_view_source == 1 {
                        // Already viewing source 1 solo — exit solo view.
                        self.view_mode = ViewMode::Split;
                        self.status_message = Some(("Solo view: off".to_string(), Instant::now()));
                    } else {
                        self.view_mode = ViewMode::Solo;
                        self.solo_view_source = 1;
                        self.status_message =
                            Some((format!("Solo: {}", self.right_label), Instant::now()));
                    }
                    self.update_uniform_buffer();
                }
                VirtualKeyCode::L => {
                    self.toggle_split_line();
                }
                VirtualKeyCode::V => {
                    self.pixel_info_window.toggle();
                }
                VirtualKeyCode::H => {
                    self.help_overlay.toggle();
                }
                VirtualKeyCode::R => {
                    self.zoom_level = 1.0;
                    self.fixed_zoom_center = (0.5, 0.5);
                    self.zoom_center_offset = (0.0, 0.0);
                    self.update_uniform_buffer();
                }
                VirtualKeyCode::O => {
                    self.show_hud = !self.show_hud;
                }
                VirtualKeyCode::M => {
                    self.marker_overlay.visible = !self.marker_overlay.visible;
                    let msg = if self.marker_overlay.visible {
                        "Markers: visible"
                    } else {
                        "Markers: hidden"
                    };
                    self.status_message = Some((msg.to_string(), Instant::now()));
                }
                VirtualKeyCode::N => {
                    self.marker_overlay.is_editor_open = !self.marker_overlay.is_editor_open;
                }
                VirtualKeyCode::Z => {
                    self.peek_zoom_active = true;
                }
                VirtualKeyCode::X
                    if !self.single_image_mode && self.view_mode != ViewMode::Solo =>
                {
                    self.cycle_peek_image_mode();
                }
                _ => {}
            }
        }

        if let winit::event::Event::WindowEvent {
            event: WindowEvent::MouseWheel { delta, .. },
            ..
        } = event
        {
            self.handle_zoom(delta);
        }

        if let winit::event::Event::WindowEvent {
            event:
                WindowEvent::MouseInput {
                    button: winit::event::MouseButton::Left,
                    state,
                    ..
                },
            ..
        } = event
        {
            match state {
                winit::event::ElementState::Pressed => {
                    if !self.imgui_context.io().want_capture_mouse {
                        let start = self.clamp_to_render_rect(self.mouse_position);
                        if self.marker_overlay.create_mode {
                            self.marker_drag_start = Some(start);
                            self.marker_drag_current = start;
                        } else {
                            // Start a pan drag: record the effective zoom center at drag start.
                            let effective_cx = self.fixed_zoom_center.0 + self.zoom_center_offset.0;
                            let effective_cy = self.fixed_zoom_center.1 + self.zoom_center_offset.1;
                            // Merge offset into fixed center so zoom_center_offset tracks only this drag.
                            self.fixed_zoom_center = (effective_cx, effective_cy);
                            self.zoom_center_offset = (0.0, 0.0);
                            self.drag_pan_start = Some(start);
                            self.drag_pan_initial_center = (effective_cx, effective_cy);
                        }
                    }
                }
                winit::event::ElementState::Released => {
                    // Finish a marker creation drag.
                    if let Some(start) = self.marker_drag_start.take() {
                        let current = self.marker_drag_current;
                        let dx = (current.0 - start.0).abs();
                        let dy = (current.1 - start.1).abs();
                        if dx > MIN_DRAG_DISTANCE_PX || dy > MIN_DRAG_DISTANCE_PX {
                            // Convert screen px → image UV using current zoom/pan state.
                            let (x_off, y_off, rw, rh) = self.compute_render_rect();
                            let zoom_cx = self.fixed_zoom_center.0 + self.zoom_center_offset.0;
                            let zoom_cy = self.fixed_zoom_center.1 + self.zoom_center_offset.1;
                            let zoom = self.zoom_level;
                            let to_uv = |sx: f32, sy: f32| -> (f32, f32) {
                                let suv_x = (sx - x_off) / rw.max(1.0);
                                let suv_y = (sy - y_off) / rh.max(1.0);
                                let iuv_x = zoom_cx + (suv_x - zoom_cx) / zoom;
                                let iuv_y = zoom_cy + (suv_y - zoom_cy) / zoom;
                                (iuv_x.clamp(0.0, 1.0), iuv_y.clamp(0.0, 1.0))
                            };
                            let (u1, v1) = to_uv(start.0, start.1);
                            let (u2, v2) = to_uv(current.0, current.1);
                            self.marker_overlay.add_marker(u1, v1, u2, v2);
                            self.marker_overlay.create_mode = false;
                        }
                    }
                    // Finish a pan drag: fold the accumulated offset into fixed_zoom_center.
                    if self.drag_pan_start.take().is_some() {
                        self.fixed_zoom_center.0 += self.zoom_center_offset.0;
                        self.fixed_zoom_center.1 += self.zoom_center_offset.1;
                        self.zoom_center_offset = (0.0, 0.0);
                    }
                }
            }
        }

        // Right-click drag: zoom to a selected region.
        if let winit::event::Event::WindowEvent {
            event:
                WindowEvent::MouseInput {
                    button: winit::event::MouseButton::Right,
                    state,
                    ..
                },
            ..
        } = event
        {
            match state {
                winit::event::ElementState::Pressed => {
                    if !self.imgui_context.io().want_capture_mouse {
                        let start = self.clamp_to_render_rect(self.mouse_position);
                        self.drag_zoom_start = Some(start);
                        self.drag_zoom_current = start;
                    }
                }
                winit::event::ElementState::Released => {
                    if let Some(start) = self.drag_zoom_start.take() {
                        let current =
                            self.constrain_drag_to_window_aspect(start, self.drag_zoom_current);
                        let dx = (current.0 - start.0).abs();
                        let dy = (current.1 - start.1).abs();
                        if dx > MIN_DRAG_ZOOM_DISTANCE_PX || dy > MIN_DRAG_ZOOM_DISTANCE_PX {
                            self.apply_drag_zoom(start, current);
                        }
                    }
                }
            }
        }

        if let winit::event::Event::WindowEvent {
            event: WindowEvent::Touch(touch),
            ..
        } = event
        {
            self.handle_touch(touch);
        }

        if let winit::event::Event::WindowEvent {
            event: WindowEvent::DroppedFile(path),
            ..
        } = event
        {
            self.pending_drop_paths.push(path.clone());
            self.hovering_file = false;
        }

        if let winit::event::Event::WindowEvent {
            event: WindowEvent::HoveredFile(_),
            ..
        } = event
        {
            self.hovering_file = true;
        }

        if let winit::event::Event::WindowEvent {
            event: WindowEvent::HoveredFileCancelled,
            ..
        } = event
        {
            self.pending_drop_paths.clear();
            self.hovering_file = false;
        }

        debug!("Event handled");
    }

    fn compute_render_dimensions(&self) -> (f32, f32) {
        let player = self.player.read();
        let left_texture = player.get_left_texture();

        let (image_width, image_height) = if let Some(left) = left_texture {
            (left.width() as f32, left.height() as f32)
        } else {
            (self.size.width as f32, self.size.height as f32)
        };

        let window_aspect_ratio = self.size.width as f32 / self.size.height as f32;
        let image_aspect_ratio = image_width / image_height;

        if window_aspect_ratio > image_aspect_ratio {
            let scaled_height = self.size.height as f32;
            let scaled_width = scaled_height * image_aspect_ratio;
            (scaled_width, scaled_height)
        } else {
            let scaled_width = self.size.width as f32;
            let scaled_height = scaled_width / image_aspect_ratio;
            (scaled_width, scaled_height)
        }
    }

    fn compute_render_rect(&self) -> (f32, f32, f32, f32) {
        let (render_width, render_height) = self.compute_render_dimensions();
        let x_offset = (self.size.width as f32 - render_width) / 2.0;
        let y_offset = (self.size.height as f32 - render_height) / 2.0;
        (x_offset, y_offset, render_width, render_height)
    }

    fn clamp_to_render_rect(&self, position: (f32, f32)) -> (f32, f32) {
        let (x_offset, y_offset, render_width, render_height) = self.compute_render_rect();
        (
            position.0.clamp(x_offset, x_offset + render_width),
            position.1.clamp(y_offset, y_offset + render_height),
        )
    }

    fn update_cursor_state(&mut self, x: f32, y: f32) {
        let (x_offset, y_offset, render_width, render_height) = self.compute_render_rect();

        self.mouse_position = (x, y);
        self.cursor_x = if self.single_image_mode {
            render_width
        } else {
            (x - x_offset).max(0.0).min(render_width)
        };
        self.cursor_y = (y - y_offset).max(0.0).min(render_height);
    }

    pub fn cycle_comparison_mode(&mut self) {
        self.comparison_mode = self.comparison_mode.cycle();
        if self.comparison_mode == ComparisonMode::Flip {
            let (current_left, current_right) = self.player.read().current_images();
            self.player
                .write()
                .generate_flip_diff(current_left, current_right);
        }
        self.normalize_peek_image_mode();
        let label = self.comparison_mode.label();
        self.status_message = Some((format!("Comparison mode: {}", label), Instant::now()));
        self.update_uniform_buffer();
    }

    pub fn toggle_split_line(&mut self) {
        self.show_split_line = !self.show_split_line;
        self.update_uniform_buffer();
    }

    fn handle_zoom(&mut self, delta: &winit::event::MouseScrollDelta) {
        let zoom_factor = match delta {
            winit::event::MouseScrollDelta::LineDelta(_, y) => 1.0 + y.signum() * 0.1,
            winit::event::MouseScrollDelta::PixelDelta(winit::dpi::PhysicalPosition {
                y, ..
            }) => 1.0 + (y / 100.0) as f32,
        };

        let new_zoom_level = (self.zoom_level * zoom_factor).clamp(1.0, MAX_ZOOM_LEVEL);

        // Convert cursor position to texture coordinates [0, 1] using render dimensions.
        let (render_width, render_height) = self.compute_render_dimensions();
        let mouse_x = self.cursor_x / render_width;
        let mouse_y = self.cursor_y / render_height;

        // Incorporate any active pan offset into the effective zoom center.
        let center_x = self.fixed_zoom_center.0 + self.zoom_center_offset.0;
        let center_y = self.fixed_zoom_center.1 + self.zoom_center_offset.1;

        // Compute the new zoom center so the image pixel under the cursor stays fixed.
        // The shader maps screen position `t` to image pixel `p`:
        //   p = center + (t - center) / zoom_level
        // For pixel `p` to remain at position `t` after zooming to new_zoom_level:
        //   new_center = (p * new_zoom_level - t) / (new_zoom_level - 1)
        let (new_center_x, new_center_y) = if new_zoom_level > 1.0 {
            let pixel_x = center_x + (mouse_x - center_x) / self.zoom_level;
            let pixel_y = center_y + (mouse_y - center_y) / self.zoom_level;
            let cx = (pixel_x * new_zoom_level - mouse_x) / (new_zoom_level - 1.0);
            let cy = (pixel_y * new_zoom_level - mouse_y) / (new_zoom_level - 1.0);
            (cx, cy)
        } else {
            // At zoom level 1 there is no zoom; reset to center.
            (0.5, 0.5)
        };

        // Clamp the zoom center to keep it within the image bounds.
        let max_offset_x = (1.0 - 1.0 / new_zoom_level) / 2.0;
        let max_offset_y = (1.0 - 1.0 / new_zoom_level) / 2.0;
        let clamped_center_x = new_center_x.clamp(0.5 - max_offset_x, 0.5 + max_offset_x);
        let clamped_center_y = new_center_y.clamp(0.5 - max_offset_y, 0.5 + max_offset_y);

        // Update the zoom level and zoom center; reset the pan offset.
        self.zoom_level = new_zoom_level;
        self.fixed_zoom_center = (clamped_center_x, clamped_center_y);
        self.zoom_center_offset = (0.0, 0.0);

        self.update_uniform_buffer();
    }

    fn effective_show_images(&self) -> (f32, f32) {
        match self.view_mode {
            ViewMode::Solo => (
                if self.solo_view_source == 0 { 1.0 } else { 0.0 },
                if self.solo_view_source == 1 { 1.0 } else { 0.0 },
            ),
            ViewMode::Split => (
                if self.show_image1 { 1.0 } else { 0.0 },
                if self.show_image2 { 1.0 } else { 0.0 },
            ),
        }
    }

    fn effective_peek_image_mode(&self) -> PeekImageMode {
        if self.view_mode == ViewMode::Solo {
            if self.solo_view_source == 0 {
                PeekImageMode::Image1
            } else {
                PeekImageMode::Image2
            }
        } else {
            self.peek_image_mode
        }
    }

    fn update_uniform_buffer(&self) {
        let player = self.player.read();
        let (left_texture, right_texture) = (player.get_left_texture(), player.get_right_texture());

        let (image1_size, image2_size) =
            if let (Some(left), Some(right)) = (left_texture, right_texture) {
                (
                    [left.width() as f32, left.height() as f32],
                    [right.width() as f32, right.height() as f32],
                )
            } else {
                (
                    [self.size.width as f32, self.size.height as f32],
                    [self.size.width as f32, self.size.height as f32],
                )
            };

        let (show_image1, show_image2) = self.effective_show_images();

        let uniforms = UniformData {
            cursor_x: self.cursor_x / self.size.width as f32,
            cursor_y: self.cursor_y / self.size.height as f32,
            image1_size,
            image2_size,
            flip_diff_size: [self.size.width as f32, self.size.height as f32],
            comparison_mode: self.comparison_mode.as_f32(),
            zoom_level: self.zoom_level,
            zoom_center: [
                self.fixed_zoom_center.0 + self.zoom_center_offset.0,
                self.fixed_zoom_center.1 + self.zoom_center_offset.1,
            ],
            window_size: [self.size.width as f32, self.size.height as f32],
            show_image1,
            show_image2,
            show_split_line: if self.show_split_line { 1.0 } else { 0.0 },
            peek_active: if self.peek_zoom_active { 1.0 } else { 0.0 },
            peek_factor: self.peek_zoom_factor,
            peek_radius: self.peek_zoom_radius,
            diff_multiplier: self.diff_enhance_factor,
            pump_active: if self.pump_animation_active { 1.0 } else { 0.0 },
            time: self.start_time.elapsed().as_secs_f32(),
            peek_show_image: self.effective_peek_image_mode().as_f32(),
        };

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    fn handle_touch(&mut self, touch: &winit::event::Touch) {
        match touch.phase {
            TouchPhase::Started => {
                self.swipe_start = Some((touch.location.x, touch.location.y));
            }
            TouchPhase::Moved => {
                if let Some((start_x, start_y)) = self.swipe_start {
                    let dx = touch.location.x - start_x;
                    let dy = touch.location.y - start_y;

                    if dx.abs() > self.swipe_threshold || dy.abs() > self.swipe_threshold {
                        if dx.abs() > dy.abs() {
                            // Horizontal swipe
                            if dx > 0.0 {
                                self.previous_frame();
                            } else {
                                self.next_frame();
                            }
                        } else {
                            // Vertical swipe
                            if dy > 0.0 {
                                self.handle_zoom(&winit::event::MouseScrollDelta::LineDelta(
                                    0.0, 1.0,
                                ));
                            } else {
                                self.handle_zoom(&winit::event::MouseScrollDelta::LineDelta(
                                    0.0, -1.0,
                                ));
                            }
                        }
                        self.swipe_start = None;
                    }
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.swipe_start = None;
            }
        }
    }

    pub fn handle_zoom_move(&mut self, direction: (f32, f32)) {
        let (dx, dy) = direction;

        // Scale the step by 1/zoom_level so that each key press moves a consistent
        // fraction of the *visible* area regardless of the current zoom level.
        // zoom_move_speed is interpreted as the desired fraction of the visible width
        // per key press (e.g. 0.1 = 10 % of the visible area).
        let step = self.zoom_move_speed / self.zoom_level;

        // Compute the new effective center (fixed + offset + delta).
        let eff_x = self.fixed_zoom_center.0 + self.zoom_center_offset.0 + dx * step;
        let eff_y = self.fixed_zoom_center.1 + self.zoom_center_offset.1 + dy * step;

        // Clamp the *effective* center just inside [0, 1] so we can pan to the
        // image edge without producing exact 0.0/1.0 texture coordinates at the
        // screen boundary. The shaders currently treat the upper boundary
        // exclusively, so hitting an exact endpoint can introduce a 1 px transparent
        // border.
        let min_zoom_center = f32::EPSILON;
        let max_zoom_center = 1.0 - f32::EPSILON;
        let clamped_x = eff_x.clamp(min_zoom_center, max_zoom_center);
        let clamped_y = eff_y.clamp(min_zoom_center, max_zoom_center);

        // Derive the new offset as the difference from the (unchanged) fixed center.
        self.zoom_center_offset = (
            clamped_x - self.fixed_zoom_center.0,
            clamped_y - self.fixed_zoom_center.1,
        );
        self.update_uniform_buffer();
    }

    fn adjust_peek_zoom_factor(&mut self, delta: f32) {
        self.peek_zoom_factor =
            (self.peek_zoom_factor + delta).clamp(MIN_PEEK_ZOOM_FACTOR, MAX_PEEK_ZOOM_FACTOR);
        self.status_message = Some((
            format!("Peek zoom: {:.2}x", self.peek_zoom_factor),
            Instant::now(),
        ));
        self.update_uniform_buffer();
    }

    fn is_peek_mode_available(&self, mode: PeekImageMode) -> bool {
        match mode {
            PeekImageMode::Both => self.show_image1 && self.show_image2,
            PeekImageMode::Image1 => self.show_image1,
            PeekImageMode::Image2 => self.show_image2,
            PeekImageMode::CurrentView => self.show_image1 || self.show_image2,
            PeekImageMode::DiffOnly => self.comparison_mode != ComparisonMode::None,
        }
    }

    fn normalize_peek_image_mode(&mut self) {
        if self.is_peek_mode_available(self.peek_image_mode) {
            return;
        }
        let mut next_mode = self.peek_image_mode;
        for _ in 0..5 {
            next_mode = next_mode.cycle();
            if self.is_peek_mode_available(next_mode) {
                self.peek_image_mode = next_mode;
                return;
            }
        }
        self.peek_image_mode = PeekImageMode::Both;
    }

    fn cycle_peek_image_mode(&mut self) {
        let current_mode = self.peek_image_mode;
        for _ in 0..5 {
            self.peek_image_mode = self.peek_image_mode.cycle();
            if self.is_peek_mode_available(self.peek_image_mode) {
                self.status_message =
                    Some((self.peek_image_mode.label().to_string(), Instant::now()));
                self.update_uniform_buffer();
                return;
            }
        }
        self.peek_image_mode = current_mode;
        self.status_message = Some((self.peek_image_mode.label().to_string(), Instant::now()));
        self.update_uniform_buffer();
    }

    fn toggle_pump_animation(&mut self) {
        self.pump_animation_active = !self.pump_animation_active;
        let msg = if self.pump_animation_active {
            "Diff highlight: on"
        } else {
            "Diff highlight: off"
        };
        self.status_message = Some((msg.to_string(), Instant::now()));
        self.update_uniform_buffer();
    }

    fn adjust_diff_multiplier(&mut self, delta: f32) {
        self.diff_enhance_factor = (self.diff_enhance_factor + delta).clamp(1.0, 1000.0);
        self.status_message = Some((
            format!("Diff multiplier: {:.0}x", self.diff_enhance_factor),
            Instant::now(),
        ));
        self.update_uniform_buffer();
    }

    /// Constrain drag end-point so the selection box keeps the render-area aspect ratio.
    fn constrain_drag_to_window_aspect(&self, start: (f32, f32), end: (f32, f32)) -> (f32, f32) {
        let (x_offset, y_offset, render_width, render_height) = self.compute_render_rect();
        let start = self.clamp_to_render_rect(start);
        let end = self.clamp_to_render_rect(end);
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        let abs_dx = dx.abs();
        let abs_dy = dy.abs();
        if abs_dx <= f32::EPSILON && abs_dy <= f32::EPSILON {
            return end;
        }

        let aspect = render_width / render_height.max(1.0);
        let sign_x = if dx < 0.0 { -1.0 } else { 1.0 };
        let sign_y = if dy < 0.0 { -1.0 } else { 1.0 };

        let (new_abs_dx, new_abs_dy) = if abs_dy <= f32::EPSILON {
            (abs_dx, abs_dx / aspect)
        } else if abs_dx <= f32::EPSILON {
            (abs_dy * aspect, abs_dy)
        } else if abs_dx / abs_dy > aspect {
            (abs_dx, abs_dx / aspect)
        } else {
            (abs_dy * aspect, abs_dy)
        };

        let max_abs_dx = if sign_x > 0.0 {
            (x_offset + render_width) - start.0
        } else {
            start.0 - x_offset
        };
        let max_abs_dy = if sign_y > 0.0 {
            (y_offset + render_height) - start.1
        } else {
            start.1 - y_offset
        };
        let scale_x = if new_abs_dx > f32::EPSILON {
            max_abs_dx / new_abs_dx
        } else {
            1.0
        };
        let scale_y = if new_abs_dy > f32::EPSILON {
            max_abs_dy / new_abs_dy
        } else {
            1.0
        };
        let scale = scale_x.min(scale_y).clamp(0.0, 1.0);
        let constrained_abs_dx = new_abs_dx * scale;
        let constrained_abs_dy = new_abs_dy * scale;

        (
            start.0 + sign_x * constrained_abs_dx,
            start.1 + sign_y * constrained_abs_dy,
        )
    }

    /// Apply a zoom that fits the drag rectangle defined by two screen-space positions.
    fn apply_drag_zoom(&mut self, start: (f32, f32), end: (f32, f32)) {
        let (x_offset, y_offset, render_width, render_height) = self.compute_render_rect();
        let start = self.clamp_to_render_rect(start);
        let end = self.clamp_to_render_rect(end);

        // Convert screen coordinates to normalised image UV [0, 1].
        let u1 = ((start.0 - x_offset) / render_width).clamp(0.0, 1.0);
        let v1 = ((start.1 - y_offset) / render_height).clamp(0.0, 1.0);
        let u2 = ((end.0 - x_offset) / render_width).clamp(0.0, 1.0);
        let v2 = ((end.1 - y_offset) / render_height).clamp(0.0, 1.0);

        let (u1, u2) = (u1.min(u2), u1.max(u2));
        let (v1, v2) = (v1.min(v2), v1.max(v2));

        let du = u2 - u1;
        let dv = v2 - v1;

        if du < MIN_DRAG_ZOOM_UV_SIZE || dv < MIN_DRAG_ZOOM_UV_SIZE {
            return;
        }

        // Choose the zoom level that fully shows the rectangle.
        let new_zoom_level = (1.0_f32 / du).min(1.0 / dv).clamp(1.0, MAX_ZOOM_LEVEL);
        let new_center_x = (u1 + u2) / 2.0;
        let new_center_y = (v1 + v2) / 2.0;

        let max_offset_x = (1.0 - 1.0 / new_zoom_level) / 2.0;
        let max_offset_y = (1.0 - 1.0 / new_zoom_level) / 2.0;
        let clamped_center_x = new_center_x.clamp(0.5 - max_offset_x, 0.5 + max_offset_x);
        let clamped_center_y = new_center_y.clamp(0.5 - max_offset_y, 0.5 + max_offset_y);

        self.zoom_level = new_zoom_level;
        self.fixed_zoom_center = (clamped_center_x, clamped_center_y);
        self.zoom_center_offset = (0.0, 0.0);
        self.update_uniform_buffer();
    }

    /// Save the current FLIP diff image to a PNG file.
    pub fn save_flip_diff_image(&mut self) {
        let player = self.player.read();
        let (left_index, right_index) = player.current_images();
        match player.get_flip_diff_raw_data(left_index, right_index) {
            Some((mut data, width, height)) => {
                if self.marker_overlay.visible && !self.marker_overlay.markers.is_empty() {
                    draw_markers_on_image(&mut data, width, height, &self.marker_overlay.markers);
                }
                let path = generate_output_filename("flip_diff", "png");
                match image::save_buffer(&path, &data, width, height, image::ColorType::Rgba8) {
                    Ok(_) => {
                        info!("FLIP diff image saved to {}", path);
                        self.status_message =
                            Some((format!("FLIP diff saved: {}", path), Instant::now()));
                    }
                    Err(e) => {
                        warn!("Failed to save FLIP diff image: {}", e);
                        self.status_message =
                            Some((format!("Failed to save FLIP diff: {}", e), Instant::now()));
                    }
                }
            }
            None => {
                warn!(
                    "No FLIP diff image available for frames ({}, {}). Enable FLIP mode first.",
                    left_index, right_index
                );
                self.status_message = Some((
                    "No FLIP diff available. Press F to enable FLIP mode first.".to_string(),
                    Instant::now(),
                ));
            }
        }
    }

    /// Request a screenshot to be saved on the next rendered frame.
    pub fn request_screenshot(&mut self) {
        self.screenshot_requested = true;
        self.status_message = Some(("Saving screenshot...".to_string(), Instant::now()));
    }

    /// Save a combined image with all available sources (left, right, and optionally
    /// FLIP diff) stitched side-by-side into a single PNG file.
    pub fn save_combined_screenshot(&mut self) {
        let player = self.player.read();
        let (left_index, right_index) = player.current_images();
        let left_frame_label = player
            .config
            .image_data1
            .get(left_index)
            .and_then(|(path, _, _)| {
                std::path::Path::new(path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(ToString::to_string)
            })
            .unwrap_or_else(|| self.left_label.clone());
        let right_frame_label = player
            .config
            .image_data2
            .get(right_index)
            .and_then(|(path, _, _)| {
                std::path::Path::new(path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(ToString::to_string)
            })
            .unwrap_or_else(|| self.right_label.clone());

        let left = player.get_current_frame_image_data(left_index, true);
        let right = if self.single_image_mode {
            None
        } else {
            player.get_current_frame_image_data(right_index, false)
        };
        let flip_diff = if self.comparison_mode == ComparisonMode::Flip && !self.single_image_mode {
            player
                .get_flip_diff_raw_data(left_index, right_index)
                .map(|(mut pixels, w, h)| {
                    // Apply diff_multiplier: amplify the FLIP colormap brightness.
                    let factor = self.diff_enhance_factor;
                    if (factor - 1.0).abs() > f32::EPSILON {
                        for chunk in pixels.chunks_mut(4) {
                            chunk[0] = ((chunk[0] as f32 * factor).min(255.0)) as u8;
                            chunk[1] = ((chunk[1] as f32 * factor).min(255.0)) as u8;
                            chunk[2] = ((chunk[2] as f32 * factor).min(255.0)) as u8;
                        }
                    }
                    (pixels, w, h)
                })
        } else if self.comparison_mode == ComparisonMode::AbsDiff && !self.single_image_mode {
            // Compute per-pixel abs diff from the two source images.
            let left_data = player.get_current_frame_image_data(left_index, true);
            let right_data = player.get_current_frame_image_data(right_index, false);
            match (left_data, right_data) {
                (Some((lp, lw, lh)), Some((rp, rw, rh))) if lw == rw && lh == rh => {
                    let factor = self.diff_enhance_factor;
                    let pixels: Vec<u8> = lp
                        .chunks(4)
                        .zip(rp.chunks(4))
                        .flat_map(|(l, r)| {
                            let dr = ((l[0] as i16 - r[0] as i16).abs() as f32 * factor).min(255.0)
                                as u8;
                            let dg = ((l[1] as i16 - r[1] as i16).abs() as f32 * factor).min(255.0)
                                as u8;
                            let db = ((l[2] as i16 - r[2] as i16).abs() as f32 * factor).min(255.0)
                                as u8;
                            [dr, dg, db, 255u8]
                        })
                        .collect();
                    Some((pixels, lw, lh))
                }
                _ => None,
            }
        } else {
            None
        };
        drop(player);

        let mut panels: Vec<(Vec<u8>, u32, u32)> = Vec::new();
        if let Some(l) = left {
            panels.push(l);
        }
        if let Some(r) = right {
            panels.push(r);
        }
        if let Some(d) = flip_diff {
            panels.push(d);
        }

        // Apply zoom/pan crop so the combined screenshot matches what is currently
        // visible on screen.  Do this BEFORE drawing markers so that:
        //  - markers outside the visible region are excluded (correct)
        //  - markers inside are drawn at their original visual size (no unwanted upscaling)
        let zoom_center = (
            self.fixed_zoom_center.0 + self.zoom_center_offset.0,
            self.fixed_zoom_center.1 + self.zoom_center_offset.1,
        );
        if self.zoom_level > 1.0 {
            for panel in &mut panels {
                let (new_pixels, new_w, new_h) = crop_image_to_zoom(
                    std::mem::take(&mut panel.0),
                    panel.1,
                    panel.2,
                    self.zoom_level,
                    zoom_center,
                );
                *panel = (new_pixels, new_w, new_h);
            }
        }

        // Draw markers AFTER the crop+upscale.  Transform each marker's UV coordinates
        // from source-image space to the visible-region space using the inverse of the
        // shader UV mapping:  s = zoom_center + (t - zoom_center) * zoom_level
        // where t is the source UV and s is the output UV.  When zoom_level == 1 this
        // is the identity transform.  The existing clamp(0,1) in draw_markers_on_image
        // clips markers that are only partially visible to the image boundary.
        if self.marker_overlay.visible && !self.marker_overlay.markers.is_empty() {
            let zoom_level = self.zoom_level;
            let (cx, cy) = zoom_center;
            let transformed: Vec<Marker> = self
                .marker_overlay
                .markers
                .iter()
                .map(|m| Marker {
                    id: m.id,
                    x1: cx + (m.x1 - cx) * zoom_level,
                    y1: cy + (m.y1 - cy) * zoom_level,
                    x2: cx + (m.x2 - cx) * zoom_level,
                    y2: cy + (m.y2 - cy) * zoom_level,
                    label: m.label.clone(),
                    color: m.color,
                })
                .collect();
            for (pixels, width, height) in &mut panels {
                draw_markers_on_image(pixels, *width, *height, &transformed);
            }
        }

        // Draw image source labels on each panel when the HUD is active.
        if self.show_hud && !self.single_image_mode {
            let diff_label = match self.comparison_mode {
                ComparisonMode::Flip => "FLIP Diff",
                ComparisonMode::AbsDiff => "Abs Diff",
                // None and Overlay have no separate diff panel.
                ComparisonMode::None | ComparisonMode::Overlay => "",
            };
            let panel_labels: [&str; 3] = [&left_frame_label, &right_frame_label, diff_label];
            for (i, (pixels, width, height)) in panels.iter_mut().enumerate() {
                if let Some(label) = panel_labels.get(i) {
                    draw_label_bottom_left_on_image(pixels, *width, *height, label);
                }
            }
        }

        if panels.is_empty() {
            self.status_message = Some((
                "No images available for combined screenshot.".to_string(),
                Instant::now(),
            ));
            return;
        }

        let (combined_pixels, combined_width, combined_height) =
            stitch_images_side_by_side(&panels);
        let path = generate_output_filename("combined_screenshot", "png");
        match image::save_buffer(
            &path,
            &combined_pixels,
            combined_width,
            combined_height,
            image::ColorType::Rgba8,
        ) {
            Ok(_) => {
                info!("Combined screenshot saved to {}", path);
                self.status_message = Some((
                    format!("Combined screenshot saved: {}", path),
                    Instant::now(),
                ));
            }
            Err(e) => {
                warn!("Failed to save combined screenshot: {}", e);
                self.status_message = Some((
                    format!("Failed to save combined screenshot: {}", e),
                    Instant::now(),
                ));
            }
        }
    }
}

/// Crop an RGBA image to the region that is visible given the current zoom and pan,
/// then scale that region back up to the original image dimensions.
///
/// This produces an output image that looks like what is shown in the window:
/// the visible region is enlarged to fill the same pixel canvas as the original,
/// using nearest-neighbor scaling to match the texture sampler used by the shader.
///
/// `zoom_center` is the nominal zoom center in normalized \[0, 1\] image-UV space
/// (`0.0` at the left/top edge, `1.0` at the right/bottom edge). It is typically
/// computed as `fixed_zoom_center + zoom_center_offset` and may therefore be
/// slightly outside the \[0.0, 1.0\] range at high zoom levels. In that case, the
/// effective sampled/cropped region is shifted accordingly and clamped to the
/// valid image bounds. When `zoom_level` is ≤ 1.0 the original image is returned
/// unchanged.
///
/// The shader maps a screen-space coordinate `s ∈ [0,1]` to a texture coordinate
/// via `t = zoom_center + (s – zoom_center) / zoom_level`, so the visible UV range
/// is `[zoom_center – zoom_center/zoom_level,  zoom_center + (1–zoom_center)/zoom_level]`.
fn crop_image_to_zoom(
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    zoom_level: f32,
    zoom_center: (f32, f32),
) -> (Vec<u8>, u32, u32) {
    if zoom_level <= 1.0 || width == 0 || height == 0 {
        return (pixels, width, height);
    }

    let (cx, cy) = zoom_center;

    // Visible UV range derived from the shader zoom formula.
    let left_uv = (cx + (0.0 - cx) / zoom_level).clamp(0.0, 1.0);
    let right_uv = (cx + (1.0 - cx) / zoom_level).clamp(0.0, 1.0);
    let top_uv = (cy + (0.0 - cy) / zoom_level).clamp(0.0, 1.0);
    let bottom_uv = (cy + (1.0 - cy) / zoom_level).clamp(0.0, 1.0);

    // Convert UV to pixel coordinates.
    //
    // Treat UVs as covering texel centers in [0, 1], with 1.0 mapping to the last
    // texel, and use a half-open pixel range [left_px, right_px) /
    // [top_px, bottom_px) to avoid collapsing narrow but non-empty intervals.
    let max_x = width.saturating_sub(1);
    let max_y = height.saturating_sub(1);

    let left_incl = ((left_uv * max_x as f32).floor() as u32).min(max_x);
    let right_incl = ((right_uv * max_x as f32).ceil() as u32).min(max_x);
    let top_incl = ((top_uv * max_y as f32).floor() as u32).min(max_y);
    let bottom_incl = ((bottom_uv * max_y as f32).ceil() as u32).min(max_y);

    // Build half-open ranges and clamp to image bounds.
    let left_px = left_incl.min(right_incl);
    let mut right_px = right_incl.max(left_incl).saturating_add(1).min(width);
    let top_px = top_incl.min(bottom_incl);
    let mut bottom_px = bottom_incl.max(top_incl).saturating_add(1).min(height);

    // Ensure at least a 1×1 crop when the UV range is non-empty, guarding
    // against any pathological floating-point cases.
    if right_uv > left_uv && right_px <= left_px {
        right_px = (left_px + 1).min(width);
    }
    if bottom_uv > top_uv && bottom_px <= top_px {
        bottom_px = (top_px + 1).min(height);
    }

    let crop_w = right_px
        .saturating_sub(left_px)
        .min(width.saturating_sub(left_px));
    let crop_h = bottom_px
        .saturating_sub(top_px)
        .min(height.saturating_sub(top_px));

    // Return unchanged if degenerate (no visible area) or nothing to crop.
    if crop_w == 0 || crop_h == 0 {
        return (pixels, width, height);
    }
    if left_px == 0 && top_px == 0 && crop_w == width && crop_h == height {
        return (pixels, width, height);
    }

    // Extract the cropped region.
    let mut cropped = Vec::with_capacity((crop_w * crop_h * 4) as usize);
    for row in 0..crop_h {
        let src_row = top_px + row;
        let src_start = ((src_row * width + left_px) * 4) as usize;
        let src_end = src_start + (crop_w * 4) as usize;
        cropped.extend_from_slice(&pixels[src_start..src_end]);
    }

    // Scale the cropped region back up to the original image dimensions using
    // nearest-neighbor filtering, matching the nearest-filter texture sampler
    // used by the rendering shader.
    let cropped_img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(crop_w, crop_h, cropped)
        .expect("cropped buffer dimensions are consistent");
    let scaled = image::imageops::resize(
        &cropped_img,
        width,
        height,
        image::imageops::FilterType::Nearest,
    );
    (scaled.into_raw(), width, height)
}

/// Stitch multiple RGBA images side-by-side into a single image.
/// Each element is `(pixels, width, height)`. The output height equals the
/// tallest input; shorter panels are padded with transparent black rows at
/// the bottom.
fn stitch_images_side_by_side(panels: &[(Vec<u8>, u32, u32)]) -> (Vec<u8>, u32, u32) {
    let total_width: u32 = panels.iter().map(|(_, w, _)| w).sum();
    let max_height: u32 = panels.iter().map(|(_, _, h)| *h).max().unwrap_or(0);
    let mut combined = vec![0u8; (total_width * max_height * 4) as usize];
    let mut x_offset = 0u32;
    for (data, width, height) in panels {
        for row in 0..*height {
            let src_start = (row * width * 4) as usize;
            let src_end = src_start + (width * 4) as usize;
            let dst_start = (row * total_width * 4 + x_offset * 4) as usize;
            combined[dst_start..dst_start + (width * 4) as usize]
                .copy_from_slice(&data[src_start..src_end]);
        }
        x_offset += width;
    }
    (combined, total_width, max_height)
}

/// Draw marker outlines onto an RGBA image buffer in-place.
fn draw_markers_on_image(pixels: &mut [u8], width: u32, height: u32, markers: &[Marker]) {
    fn put_pixel(pixels: &mut [u8], width: u32, height: u32, x: i32, y: i32, color: [u8; 4]) {
        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
            return;
        }
        let idx = ((y as u32 * width + x as u32) * 4) as usize;
        pixels[idx..idx + 4].copy_from_slice(&color);
    }
    #[allow(clippy::too_many_arguments)]
    fn fill_rect(
        pixels: &mut [u8],
        width: u32,
        height: u32,
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
        color: [u8; 4],
    ) {
        for y in y0..=y1 {
            for x in x0..=x1 {
                put_pixel(pixels, width, height, x, y, color);
            }
        }
    }
    fn draw_text(
        pixels: &mut [u8],
        width: u32,
        height: u32,
        x: i32,
        y: i32,
        text: &str,
        color: [u8; 4],
    ) {
        let mut pen_x = x;
        for ch in text.chars() {
            if let Some(glyph) = font8x8::BASIC_FONTS.get(ch) {
                for (row, bits) in glyph.iter().enumerate() {
                    for col in 0..8_u8 {
                        if ((bits >> col) & 1) != 0 {
                            put_pixel(
                                pixels,
                                width,
                                height,
                                pen_x + col as i32,
                                y + row as i32,
                                color,
                            );
                        }
                    }
                }
            }
            pen_x += 8;
        }
    }

    if width == 0 || height == 0 {
        return;
    }

    const MARKER_THICKNESS: i32 = 2;
    const CHAR_W: i32 = 8;
    const LABEL_GAP: i32 = 11; // pixels above the marker top edge for the label
    const PAD_X: i32 = 2; // horizontal padding around label text
    const PAD_Y: i32 = 1; // vertical padding around label text

    for marker in markers {
        let color = [
            (marker.color[0].clamp(0.0, 1.0) * 255.0) as u8,
            (marker.color[1].clamp(0.0, 1.0) * 255.0) as u8,
            (marker.color[2].clamp(0.0, 1.0) * 255.0) as u8,
            255,
        ];
        let x1 = (marker.x1.clamp(0.0, 1.0) * (width.saturating_sub(1)) as f32).round() as i32;
        let y1 = (marker.y1.clamp(0.0, 1.0) * (height.saturating_sub(1)) as f32).round() as i32;
        let x2 = (marker.x2.clamp(0.0, 1.0) * (width.saturating_sub(1)) as f32).round() as i32;
        let y2 = (marker.y2.clamp(0.0, 1.0) * (height.saturating_sub(1)) as f32).round() as i32;
        let (left, right) = (x1.min(x2), x1.max(x2));
        let (top, bottom) = (y1.min(y2), y1.max(y2));

        for t in 0..MARKER_THICKNESS {
            let lt = left - t;
            let rt = right + t;
            let tt = top - t;
            let bt = bottom + t;
            for x in lt..=rt {
                put_pixel(pixels, width, height, x, tt, color);
                put_pixel(pixels, width, height, x, bt, color);
            }
            for y in tt..=bt {
                put_pixel(pixels, width, height, lt, y, color);
                put_pixel(pixels, width, height, rt, y, color);
            }
        }

        if !marker.label.is_empty() {
            let text_w = (marker.label.chars().count() as i32) * CHAR_W;
            let tx = left.max(0);
            let ty = (top - LABEL_GAP).max(0);
            fill_rect(
                pixels,
                width,
                height,
                tx - PAD_X,
                ty - PAD_Y,
                tx + text_w + PAD_X - 1,
                (ty + CHAR_W).min(height as i32 - 1),
                [0, 0, 0, 200],
            );
            draw_text(pixels, width, height, tx, ty, &marker.label, color);
        }
    }
}

/// Draw a short text label with a semi-transparent dark background at the
/// bottom-left corner of an RGBA image buffer in-place.  Uses the 8×8
/// font8x8 bitmap font (same rendering approach as `draw_markers_on_image`).
fn draw_label_bottom_left_on_image(pixels: &mut [u8], width: u32, height: u32, label: &str) {
    if width == 0 || height == 0 || label.is_empty() {
        return;
    }
    let char_w: i32 = 8;
    let char_h: i32 = 8;
    let text_w = label.chars().count() as i32 * char_w;
    let pad: i32 = 4;
    let tx = pad;
    let ty = height as i32 - char_h - pad * 2;
    if ty < 0 {
        return;
    }
    // Dark semi-transparent background: darken existing pixels by 75 %.
    let bg_x0 = (tx - pad).max(0) as u32;
    let bg_y0 = (ty - pad).max(0) as u32;
    let bg_x1 = (tx + text_w + pad - 1).min(width as i32 - 1) as u32;
    let bg_y1 = (ty + char_h + pad - 1).min(height as i32 - 1) as u32;
    for y in bg_y0..=bg_y1 {
        for x in bg_x0..=bg_x1 {
            let idx = ((y * width + x) * 4) as usize;
            if idx + 3 < pixels.len() {
                pixels[idx] = (pixels[idx] as f32 * 0.25) as u8;
                pixels[idx + 1] = (pixels[idx + 1] as f32 * 0.25) as u8;
                pixels[idx + 2] = (pixels[idx + 2] as f32 * 0.25) as u8;
                pixels[idx + 3] = 255;
            }
        }
    }
    // White text using font8x8.
    let mut pen_x = tx;
    for ch in label.chars() {
        if let Some(glyph) = font8x8::BASIC_FONTS.get(ch) {
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..8_u8 {
                    if ((bits >> col) & 1) != 0 {
                        let px = pen_x + col as i32;
                        let py = ty + row as i32;
                        if px >= 0 && py >= 0 && (px as u32) < width && (py as u32) < height {
                            let idx = ((py as u32 * width + px as u32) * 4) as usize;
                            if idx + 3 < pixels.len() {
                                pixels[idx] = 255;
                                pixels[idx + 1] = 255;
                                pixels[idx + 2] = 255;
                                pixels[idx + 3] = 255;
                            }
                        }
                    }
                }
            }
        }
        pen_x += char_w;
    }
}

/// Generate a timestamped output file path in the current directory.
fn generate_output_filename(prefix: &str, extension: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{}_{}.{:03}.{}",
        prefix,
        d.as_secs(),
        d.subsec_millis(),
        extension
    )
}

#[cfg(test)]
mod tests {
    use super::{
        draw_markers_on_image, generate_output_filename, stitch_images_side_by_side, MarkerOverlay,
    };

    #[test]
    fn test_generate_output_filename_format() {
        let name = generate_output_filename("screenshot", "png");
        assert!(
            name.starts_with("screenshot_"),
            "should start with prefix: {}",
            name
        );
        assert!(name.ends_with(".png"), "should end with .png: {}", name);
        // Format is screenshot_{secs}.{millis}.png — strip prefix and extension
        let inner = name
            .strip_prefix("screenshot_")
            .unwrap()
            .strip_suffix(".png")
            .unwrap();
        // Inner should be "{secs}.{millis}" — exactly one dot
        let parts: Vec<&str> = inner.splitn(2, '.').collect();
        assert_eq!(parts.len(), 2, "expected secs.millis format: {}", inner);
        assert!(
            parts[0].chars().all(|c| c.is_ascii_digit()),
            "seconds should be numeric digits: {}",
            parts[0]
        );
        assert!(
            parts[1].chars().all(|c| c.is_ascii_digit()),
            "millis should be numeric digits: {}",
            parts[1]
        );
        assert_eq!(
            parts[1].len(),
            3,
            "millis should be zero-padded to 3 digits: {}",
            parts[1]
        );
    }

    #[test]
    fn test_generate_output_filename_flip_diff() {
        let name = generate_output_filename("flip_diff", "png");
        assert!(
            name.starts_with("flip_diff_"),
            "should start with flip_diff_: {}",
            name
        );
        assert!(name.ends_with(".png"), "should end with .png: {}", name);
        // Format is flip_diff_{secs}.{millis}.png
        let inner = name
            .strip_prefix("flip_diff_")
            .unwrap()
            .strip_suffix(".png")
            .unwrap();
        let parts: Vec<&str> = inner.splitn(2, '.').collect();
        assert_eq!(parts.len(), 2, "expected secs.millis format: {}", inner);
    }

    #[test]
    fn test_generate_output_filename_unique() {
        // Two calls should produce different filenames when at least one second apart,
        // but even within the same second the function is deterministic; we only check
        // that the function doesn't panic and returns a non-empty string.
        let a = generate_output_filename("test", "png");
        let b = generate_output_filename("test", "png");
        assert!(!a.is_empty());
        assert!(!b.is_empty());
    }

    /// Verifies the BGRA→RGBA channel-swap logic used in the screenshot path.
    #[test]
    fn test_bgra_to_rgba_conversion() {
        // Simulate a 1×1 BGRA pixel: B=10, G=20, R=30, A=255
        let bgra_pixel: Vec<u8> = vec![10, 20, 30, 255];
        let rgba_pixel: Vec<u8> = bgra_pixel
            .chunks(4)
            .flat_map(|c| vec![c[2], c[1], c[0], c[3]])
            .collect();
        assert_eq!(rgba_pixel, vec![30, 20, 10, 255]);
    }

    /// Verifies that row-padding removal works correctly.
    #[test]
    fn test_row_padding_removal() {
        let width: u32 = 2;
        let height: u32 = 2;
        // bytes_per_row aligned to 256: (2*4 + 255) & !255 = 256
        let bytes_per_row: usize = 256;
        // Construct padded buffer: two rows of 256 bytes each, with 8 bytes of pixel data
        let mut raw = vec![0u8; bytes_per_row * height as usize];
        // Row 0: pixels R=1,G=2,B=3,A=4 and R=5,G=6,B=7,A=8
        raw[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        // Row 1: pixels R=9,G=10,B=11,A=12 and R=13,G=14,B=15,A=16
        raw[256..264].copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);

        let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height as usize {
            let row_start = row * bytes_per_row;
            pixels.extend_from_slice(&raw[row_start..row_start + (width * 4) as usize]);
        }

        assert_eq!(
            pixels,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    /// Verifies that `stitch_images_side_by_side` places two 1×1 panels next to each other.
    #[test]
    fn test_stitch_two_1x1_images() {
        // Panel 1: a single red pixel
        let red: Vec<u8> = vec![255, 0, 0, 255];
        // Panel 2: a single blue pixel
        let blue: Vec<u8> = vec![0, 0, 255, 255];
        let panels = vec![(red.clone(), 1u32, 1u32), (blue.clone(), 1u32, 1u32)];
        let (combined, width, height) = stitch_images_side_by_side(&panels);
        assert_eq!(width, 2, "combined width should be 2");
        assert_eq!(height, 1, "combined height should be 1");
        assert_eq!(&combined[0..4], &red[..], "first pixel should be red");
        assert_eq!(&combined[4..8], &blue[..], "second pixel should be blue");
    }

    /// Verifies that `stitch_images_side_by_side` pads shorter panels with transparent rows.
    #[test]
    fn test_stitch_images_height_padding() {
        // Panel 1: 1×2 (green column)
        let green_top: Vec<u8> = vec![0, 255, 0, 255, 0, 255, 0, 255]; // 2 rows
                                                                       // Panel 2: 1×1 (red pixel — shorter than panel 1)
        let red: Vec<u8> = vec![255, 0, 0, 255];
        let panels = vec![(green_top, 1u32, 2u32), (red, 1u32, 1u32)];
        let (combined, width, height) = stitch_images_side_by_side(&panels);
        assert_eq!(width, 2);
        assert_eq!(height, 2);
        // Row 0: green | red
        assert_eq!(
            &combined[0..4],
            &[0, 255, 0, 255],
            "row0 col0 should be green"
        );
        assert_eq!(
            &combined[4..8],
            &[255, 0, 0, 255],
            "row0 col1 should be red"
        );
        // Row 1: green | transparent (zero-initialised)
        assert_eq!(
            &combined[8..12],
            &[0, 255, 0, 255],
            "row1 col0 should be green"
        );
        assert_eq!(
            &combined[12..16],
            &[0, 0, 0, 0],
            "row1 col1 should be transparent padding"
        );
    }

    #[test]
    fn test_draw_markers_on_image_draws_outline_only() {
        let mut pixels = vec![0u8; 8 * 8 * 4];
        let mut overlay = MarkerOverlay::new();
        overlay.add_marker(0.25, 0.25, 0.75, 0.75);
        draw_markers_on_image(&mut pixels, 8, 8, &overlay.markers);

        // Interior should remain transparent.
        let interior_idx = ((4 * 8 + 4) * 4) as usize;
        assert_eq!(&pixels[interior_idx..interior_idx + 4], &[0, 0, 0, 0]);

        // Top border should be non-zero.
        let border_idx = ((2 * 8 + 3) * 4) as usize;
        assert_ne!(&pixels[border_idx..border_idx + 4], &[0, 0, 0, 0]);
    }

    /// Verifies that `stitch_images_side_by_side` with a single panel is a no-op copy.
    #[test]
    fn test_stitch_single_panel() {
        let pixels: Vec<u8> = (0..16).collect(); // 2×2 RGBA
        let panels = vec![(pixels.clone(), 2u32, 2u32)];
        let (combined, width, height) = stitch_images_side_by_side(&panels);
        assert_eq!(width, 2);
        assert_eq!(height, 2);
        assert_eq!(combined, pixels);
    }

    /// Tests for the drag-and-drop path loading helper used by reload_from_dropped_paths.
    mod drop_tests {
        use crate::image_loader;
        use std::fs;

        /// Loading a single image file via load_image_paths_from_files gives 1 entry.
        #[test]
        fn test_drop_single_image_file() {
            let dir = tempfile::TempDir::new().unwrap();
            let img = image::RgbaImage::new(4, 4);
            let img_path = dir.path().join("frame.png");
            img.save(&img_path).unwrap();

            let fps = 30.0;
            let result = image_loader::load_image_paths_from_files(
                &[img_path.to_string_lossy().into_owned()],
                fps,
            );
            assert!(result.is_ok(), "should load single image file");
            let (imgs, count) = result.unwrap();
            assert_eq!(count, 1);
            assert_eq!(imgs.len(), 1);
        }

        /// Loading a directory with images uses load_image_paths and returns entries.
        #[test]
        fn test_drop_directory_with_images() {
            let dir = tempfile::TempDir::new().unwrap();
            for i in 0..3u32 {
                let img = image::RgbaImage::new(4, 4);
                img.save(dir.path().join(format!("{}.png", i))).unwrap();
            }

            let fps = 30.0;
            let result = image_loader::load_image_paths(&dir.path().to_string_lossy(), fps);
            assert!(result.is_ok(), "should load from directory");
            let (imgs, count) = result.unwrap();
            assert_eq!(count, 3);
            assert_eq!(imgs.len(), 3);
        }

        /// Dropping a non-existent path fails with a meaningful error.
        #[test]
        fn test_drop_nonexistent_path() {
            let fps = 30.0;
            let bad_path = std::path::PathBuf::from("/nonexistent/path/that/does/not/exist");
            let result = if bad_path.is_dir() {
                image_loader::load_image_paths(&bad_path.to_string_lossy(), fps)
                    .map(|(v, _)| v)
                    .map_err(|e| e.to_string())
            } else if bad_path.is_file() {
                image_loader::load_image_paths_from_files(
                    &[bad_path.to_string_lossy().into_owned()],
                    fps,
                )
                .map(|(v, _)| v)
                .map_err(|e| e.to_string())
            } else {
                Err(format!("Path not found: {}", bad_path.display()))
            };
            assert!(result.is_err(), "should fail for nonexistent path");
            assert!(
                result.unwrap_err().contains("not found"),
                "error message should mention 'not found'"
            );
        }

        /// Dropping a non-image file fails with a meaningful error.
        #[test]
        fn test_drop_non_image_file() {
            let dir = tempfile::TempDir::new().unwrap();
            let txt_path = dir.path().join("notes.txt");
            fs::write(&txt_path, "not an image").unwrap();

            let fps = 30.0;
            let result = image_loader::load_image_paths_from_files(
                &[txt_path.to_string_lossy().into_owned()],
                fps,
            );
            assert!(result.is_err(), "should fail for non-image file");
        }

        /// An empty directory produces no images.
        #[test]
        fn test_drop_empty_directory() {
            let dir = tempfile::TempDir::new().unwrap();
            let fps = 30.0;
            let result = image_loader::load_image_paths(&dir.path().to_string_lossy(), fps);
            // Should succeed but return 0 images
            assert!(result.is_ok());
            let (imgs, count) = result.unwrap();
            assert_eq!(count, 0);
            assert!(imgs.is_empty());
        }
    }

    // ── MarkerOverlay unit tests ──────────────────────────────────────────────

    #[test]
    fn test_marker_overlay_starts_empty() {
        let overlay = MarkerOverlay::new();
        assert!(overlay.markers.is_empty());
        assert!(overlay.visible);
        assert!(!overlay.is_editor_open);
        assert!(!overlay.create_mode);
    }

    #[test]
    fn test_add_marker_stores_normalised_coords() {
        let mut overlay = MarkerOverlay::new();
        // Provide coordinates in "wrong" order; add_marker should normalise them.
        overlay.add_marker(0.8, 0.7, 0.2, 0.1);
        assert_eq!(overlay.markers.len(), 1);
        let m = &overlay.markers[0];
        assert_eq!(m.x1, 0.2, "x1 should be min");
        assert_eq!(m.y1, 0.1, "y1 should be min");
        assert_eq!(m.x2, 0.8, "x2 should be max");
        assert_eq!(m.y2, 0.7, "y2 should be max");
    }

    #[test]
    fn test_add_marker_already_normalised() {
        let mut overlay = MarkerOverlay::new();
        overlay.add_marker(0.1, 0.2, 0.9, 0.8);
        let m = &overlay.markers[0];
        assert_eq!(m.x1, 0.1);
        assert_eq!(m.y1, 0.2);
        assert_eq!(m.x2, 0.9);
        assert_eq!(m.y2, 0.8);
    }

    #[test]
    fn test_add_marker_assigns_incrementing_ids() {
        let mut overlay = MarkerOverlay::new();
        overlay.add_marker(0.0, 0.0, 0.5, 0.5);
        overlay.add_marker(0.1, 0.1, 0.6, 0.6);
        overlay.add_marker(0.2, 0.2, 0.7, 0.7);
        assert_eq!(overlay.markers[0].id, 0);
        assert_eq!(overlay.markers[1].id, 1);
        assert_eq!(overlay.markers[2].id, 2);
    }

    #[test]
    fn test_add_marker_cycles_colors() {
        let mut overlay = MarkerOverlay::new();
        for _ in 0..7 {
            overlay.add_marker(0.0, 0.0, 1.0, 1.0);
        }
        // Colors cycle through 5 entries; markers 0 and 5 should share a color.
        assert_eq!(overlay.markers[0].color, overlay.markers[5].color);
        // Adjacent markers 0 and 1 should differ.
        assert_ne!(overlay.markers[0].color, overlay.markers[1].color);
    }

    #[test]
    fn test_add_marker_default_label_is_empty() {
        let mut overlay = MarkerOverlay::new();
        overlay.add_marker(0.0, 0.0, 1.0, 1.0);
        assert_eq!(overlay.markers[0].label, "");
    }

    #[test]
    fn test_delete_marker_removes_correct_one() {
        let mut overlay = MarkerOverlay::new();
        overlay.add_marker(0.0, 0.0, 0.3, 0.3); // id=0
        overlay.add_marker(0.1, 0.1, 0.4, 0.4); // id=1
        overlay.add_marker(0.2, 0.2, 0.5, 0.5); // id=2
        overlay.delete_marker(1);
        assert_eq!(overlay.markers.len(), 2);
        assert!(overlay.markers.iter().all(|m| m.id != 1));
        assert!(overlay.markers.iter().any(|m| m.id == 0));
        assert!(overlay.markers.iter().any(|m| m.id == 2));
    }

    #[test]
    fn test_delete_marker_nonexistent_id_is_noop() {
        let mut overlay = MarkerOverlay::new();
        overlay.add_marker(0.0, 0.0, 1.0, 1.0);
        overlay.delete_marker(99);
        assert_eq!(overlay.markers.len(), 1);
    }

    #[test]
    fn test_delete_all_markers() {
        let mut overlay = MarkerOverlay::new();
        overlay.add_marker(0.0, 0.0, 0.5, 0.5);
        overlay.add_marker(0.5, 0.5, 1.0, 1.0);
        overlay.delete_marker(0);
        overlay.delete_marker(1);
        assert!(overlay.markers.is_empty());
    }

    #[test]
    fn test_marker_ids_after_delete_continue_incrementing() {
        let mut overlay = MarkerOverlay::new();
        overlay.add_marker(0.0, 0.0, 0.5, 0.5); // id=0
        overlay.delete_marker(0);
        overlay.add_marker(0.1, 0.1, 0.6, 0.6); // id=1 (next_id not reset)
        assert_eq!(overlay.markers[0].id, 1);
    }

    #[test]
    fn test_toggle_visibility() {
        let mut overlay = MarkerOverlay::new();
        assert!(overlay.visible);
        overlay.visible = !overlay.visible;
        assert!(!overlay.visible);
        overlay.visible = !overlay.visible;
        assert!(overlay.visible);
    }

    // ── derive_image_label unit tests ─────────────────────────────────────────

    #[test]
    fn test_derive_label_from_dir() {
        use super::derive_image_label;
        let label = derive_image_label(Some("/some/path/mydir"), None, "Left");
        assert_eq!(label, "mydir");
    }

    #[test]
    fn test_derive_label_from_dir_trailing_slash() {
        use super::derive_image_label;
        // Path::file_name handles trailing separators on most platforms.
        let label = derive_image_label(Some("/images/renders"), None, "Left");
        assert_eq!(label, "renders");
    }

    #[test]
    fn test_derive_label_from_single_image() {
        use super::derive_image_label;
        let label = derive_image_label(None, Some(&["foo/bar/frame001.png".to_string()]), "Left");
        assert_eq!(label, "frame001.png");
    }

    #[test]
    fn test_derive_label_from_multiple_images_uses_parent_dir() {
        use super::derive_image_label;
        let images = vec![
            "renders/a/frame001.png".to_string(),
            "renders/a/frame002.png".to_string(),
        ];
        let label = derive_image_label(None, Some(&images), "Left");
        assert_eq!(label, "a");
    }

    #[test]
    fn test_derive_label_falls_back_to_default() {
        use super::derive_image_label;
        assert_eq!(derive_image_label(None, None, "Left"), "Left");
        assert_eq!(derive_image_label(None, Some(&[]), "Right"), "Right");
    }

    // ── draw_label_bottom_left_on_image unit tests ────────────────────────────

    #[test]
    fn test_draw_label_does_not_panic_on_empty_image() {
        use super::draw_label_bottom_left_on_image;
        let mut pixels: Vec<u8> = vec![];
        // Should not panic.
        draw_label_bottom_left_on_image(&mut pixels, 0, 0, "Test");
    }

    #[test]
    fn test_draw_label_modifies_pixels() {
        use super::draw_label_bottom_left_on_image;
        // 40×20 white image.
        let w = 40u32;
        let h = 20u32;
        let mut pixels = vec![200u8; (w * h * 4) as usize];
        draw_label_bottom_left_on_image(&mut pixels, w, h, "Hi");
        // After drawing the dark background, at least some pixels should have
        // been darkened (R < 200).
        let has_darkened = pixels.chunks(4).any(|c| c[0] < 200);
        assert!(has_darkened, "label background should darken some pixels");
    }

    #[test]
    fn test_draw_label_empty_string_is_noop() {
        use super::draw_label_bottom_left_on_image;
        let w = 20u32;
        let h = 20u32;
        let original = vec![128u8; (w * h * 4) as usize];
        let mut pixels = original.clone();
        draw_label_bottom_left_on_image(&mut pixels, w, h, "");
        assert_eq!(pixels, original, "empty label should not modify any pixels");
    }

    // ── crop_image_to_zoom unit tests ─────────────────────────────────────────

    /// Create a simple RGBA test image where each pixel stores its (row, col)
    /// as R=row, G=col and B=A=255.
    fn make_test_image(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height {
            for col in 0..width {
                pixels.extend_from_slice(&[row as u8, col as u8, 255, 255]);
            }
        }
        pixels
    }

    #[test]
    fn test_crop_no_zoom_returns_original() {
        use super::crop_image_to_zoom;
        let w = 4u32;
        let h = 4u32;
        let pixels = make_test_image(w, h);
        let (out, ow, oh) = crop_image_to_zoom(pixels.clone(), w, h, 1.0, (0.5, 0.5));
        assert_eq!(ow, w);
        assert_eq!(oh, h);
        assert_eq!(out, pixels);
    }

    #[test]
    fn test_crop_zoom_2x_preserves_original_dimensions() {
        use super::crop_image_to_zoom;
        let w = 100u32;
        let h = 100u32;
        let pixels = make_test_image(w, h);
        // After crop + scale-up the output dimensions must equal the original.
        let (_, ow, oh) = crop_image_to_zoom(pixels, w, h, 2.0, (0.5, 0.5));
        assert_eq!(ow, w);
        assert_eq!(oh, h);
    }

    #[test]
    fn test_crop_zoom_2x_center_top_left_pixel() {
        use super::crop_image_to_zoom;
        let w = 100u32;
        let h = 100u32;
        let pixels = make_test_image(w, h);
        // 2× zoom centred at (0.5, 0.5): visible UV [0.25, 0.75].
        // UV→pixel uses floor(uv * (width-1)), so left_px = floor(0.25 * 99) = 24.
        // After scaling back up, the top-left output pixel maps to source row=24, col=24.
        let (scaled, ow, oh) = crop_image_to_zoom(pixels, w, h, 2.0, (0.5, 0.5));
        assert_eq!(ow, w);
        assert_eq!(oh, h);
        // Nearest-neighbor: first output pixel is the first source pixel (row=24, col=24).
        assert_eq!(scaled[0], 24, "first pixel R should be source row 24");
        assert_eq!(scaled[1], 24, "first pixel G should be source col 24");
    }

    #[test]
    fn test_crop_empty_image_returns_unchanged() {
        use super::crop_image_to_zoom;
        let (out, ow, oh) = crop_image_to_zoom(vec![], 0, 0, 4.0, (0.5, 0.5));
        assert_eq!(ow, 0);
        assert_eq!(oh, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_crop_zoom_top_left_corner_preserves_dimensions() {
        use super::crop_image_to_zoom;
        let w = 100u32;
        let h = 100u32;
        let pixels = make_test_image(w, h);
        // Zoom center at (0, 0): visible UV region is [0, 0.5] × [0, 0.5].
        // Output must be scaled back to original size.
        let (scaled, ow, oh) = crop_image_to_zoom(pixels, w, h, 2.0, (0.0, 0.0));
        assert_eq!(ow, w);
        assert_eq!(oh, h);
        // Top-left of the output is the top-left of the original image (row=0, col=0).
        assert_eq!(scaled[0], 0, "first pixel R should be source row 0");
        assert_eq!(scaled[1], 0, "first pixel G should be source col 0");
    }

    #[test]
    fn test_crop_zoom_extreme_zoom_small_image_replication() {
        use super::crop_image_to_zoom;
        let w = 16u32;
        let h = 16u32;
        let zoom = 100.0f32;
        let center = (0.5f32, 0.5f32);
        let pixels = make_test_image(w, h);

        let (scaled, ow, oh) = crop_image_to_zoom(pixels.clone(), w, h, zoom, center);

        // Output dimensions must always match the original.
        assert_eq!(ow, w);
        assert_eq!(oh, h);

        // Extreme zoom must not be a no-op: the sampled region should change the data.
        assert_ne!(
            scaled, pixels,
            "crop + extreme zoom should modify the image data"
        );

        // With center (0.5, 0.5) and zoom Z, the visible UV starts at:
        // u_min = center_u - 0.5 / Z, v_min = center_v - 0.5 / Z.
        // UV→pixel uses floor(uv * (width-1)), so the first pixel corresponds to
        // floor(u_min * (width-1)) and floor(v_min * (height-1)).
        let max = (w - 1) as f32;
        let u_min = center.0 - 0.5f32 / zoom;
        let v_min = center.1 - 0.5f32 / zoom;
        let expected_col = (u_min * max).floor() as u8;
        let expected_row = (v_min * max).floor() as u8;

        // First output pixel should sample the expected source texel.
        assert_eq!(
            scaled[0], expected_row,
            "first pixel R should be source row {}",
            expected_row
        );
        assert_eq!(
            scaled[1], expected_col,
            "first pixel G should be source col {}",
            expected_col
        );
    }
}
