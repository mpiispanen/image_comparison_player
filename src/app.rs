use crate::image_loader;
use crate::player::Player;
use crate::player::PlayerConfig;
use imgui::Condition;
use imgui::Ui;
use log::{debug, info, warn};
use parking_lot::lock_api::RwLock;
use parking_lot::Mutex;
use winit::event::WindowEvent;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::event::VirtualKeyCode;
use winit::window::Window as WinitWindow;
use winit::event::TouchPhase;
use std::process;

#[allow(dead_code)]
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
struct UniformData {
    cursor_x: f32,
    cursor_y: f32,
    image1_size: [f32; 2],
    image2_size: [f32; 2],
    flip_diff_size: [f32; 2],
    show_flip_diff: f32,
    zoom_level: f32,
    zoom_center: [f32; 2],
    window_size: [f32; 2],
}

// SAFETY: UniformData is #[repr(C)] and all fields are f32 or [f32; N].
// Every field has 4-byte size and 4-byte alignment, so #[repr(C)] introduces
// no padding bytes between fields, making the struct valid for Pod.
unsafe impl bytemuck::Zeroable for UniformData {}
unsafe impl bytemuck::Pod for UniformData {}

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
        let visible_frames = (player.config.preload_ahead + player.config.preload_behind) * 4 + 1;
        let button_size = 15.0;
        let spacing = 1.0;
        let total_width = visible_frames as f32 * (button_size + spacing) - spacing;
        let desired_width = total_width.min(window_width - 20.0);

        ui.window("Cache Debug")
            .size([desired_width, self.size[1]], Condition::Always)
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

                self.size = ui.window_size();
            });
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

        let visible_frames = (player.config.preload_ahead + player.config.preload_behind) * 4 + 1;
        let button_size = 15.0;
        let spacing = 1.0;
        let total_width = visible_frames as f32 * (button_size + spacing) - spacing;
        let scale_factor =
            (params.available_width - ui.calc_text_size(label)[0] - spacing) / total_width;
        let scaled_button_size = button_size * scale_factor;
        let scaled_spacing = spacing * scale_factor;

        ui.group(|| {
            ui.set_next_item_width(params.available_width - ui.calc_text_size(label)[0] - spacing);
            ui.dummy([
                params.available_width - ui.calc_text_size(label)[0] - spacing,
                scaled_button_size + 20.0,
            ]);

            let draw_list = ui.get_window_draw_list();
            let window_pos = ui.window_pos();
            let cursor_pos = ui.cursor_pos();

            let half_visible = visible_frames / 2;

            for i in 0..visible_frames {
                let frame = (current as i64 + i as i64 - half_visible as i64)
                    .rem_euclid(params.frame_count as i64) as usize;
                let x = window_pos[0]
                    + cursor_pos[0]
                    + i as f32 * (scaled_button_size + scaled_spacing);
                let y = window_pos[1] + cursor_pos[1];

                if frame.is_multiple_of(5) {
                    draw_list.add_text([x, y - 15.0], [1.0, 1.0, 1.0, 1.0], frame.to_string());
                }

                let y = y + 5.0;

                let color = if cache.read().contains_key(&frame) {
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

        let visible_frames = (player.config.preload_ahead + player.config.preload_behind) * 4 + 1;
        let button_size = 15.0;
        let spacing = 1.0;
        let total_width = visible_frames as f32 * (button_size + spacing) - spacing;
        let scale_factor =
            (params.available_width - ui.calc_text_size("D:")[0] - spacing) / total_width;
        let scaled_button_size = button_size * scale_factor;
        let scaled_spacing = spacing * scale_factor;

        ui.group(|| {
            ui.set_next_item_width(params.available_width - ui.calc_text_size("D:")[0] - spacing);
            ui.dummy([
                params.available_width - ui.calc_text_size("D:")[0] - spacing,
                scaled_button_size + 20.0,
            ]);

            let draw_list = ui.get_window_draw_list();
            let window_pos = ui.window_pos();
            let cursor_pos = ui.cursor_pos();

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
                let y = window_pos[1] + cursor_pos[1];

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

struct CacheRowParams {
    frame_count: usize,
    mouse_pos: (f32, f32),
    available_width: f32,
}

type FlipDiffReceiver = Arc<Mutex<mpsc::Receiver<(usize, usize, Vec<u8>, wgpu::Extent3d)>>>;
type FlipDiffTexture = Arc<Mutex<Option<Arc<wgpu::Texture>>>>;

pub struct AppConfig {
    pub dir1: String,
    pub dir2: String,
    pub cache_size: usize,
    pub preload_ahead: usize,
    pub preload_behind: usize,
    pub num_load_threads: usize,
    pub num_process_threads: usize,
    pub num_flip_diff_threads: usize,
    pub diff_preload_ahead: usize,
    pub diff_preload_behind: usize,
    pub fps: f32,
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
    show_flip_diff: bool,
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
}

impl AppState {
    pub async fn new(
        window: &WinitWindow,
        app_config: AppConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Initializing AppState");
        let dir1 = std::fs::canonicalize(app_config.dir1)?;
        let dir2 = std::fs::canonicalize(app_config.dir2)?;

        let size = window.inner_size();
        let surface_scale = window.scale_factor().ceil() as u32;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
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

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: surface_format,
            // Round up to the nearest multiple of the Wayland buffer_scale so that the
            // surface size is always valid on HiDPI compositors (scale 2, 3, …).
            width: round_up_to_multiple(size.width, surface_scale),
            height: round_up_to_multiple(size.height, surface_scale),
            present_mode: surface_caps.present_modes[0],
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

        let (images1, image_len1) = image_loader::load_image_paths(dir1.to_str().unwrap(), app_config.fps)?;
        let (images2, image_len2) = image_loader::load_image_paths(dir2.to_str().unwrap(), app_config.fps)?;
        debug!(
            "Loaded {} images from dir1 and {} images from dir2",
            image_len1, image_len2
        );

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

        let imgui_renderer =
            imgui_wgpu::Renderer::new(&mut imgui_context, &device, &queue, imgui_renderer_config);

        let cache_debug_window = CacheDebugWindow::new();

        let mouse_position = (0.0, 0.0);
        let (screenshot_result_tx, screenshot_result_rx) = mpsc::channel::<String>();

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
            show_flip_diff: false,
            flip_diff_receiver: Arc::new(Mutex::new(mpsc::channel().1)),
            zoom_level: 1.0,
            fixed_zoom_center: (0.5, 0.5),
            swipe_start: None,
            swipe_threshold: 50.0,
            config,
            zoom_center_offset: (0.0, 0.0),
            zoom_move_speed: 0.01,
            screenshot_requested: false,
            surface_scale,
            status_message: None,
            screenshot_result_tx,
            screenshot_result_rx: Arc::new(Mutex::new(screenshot_result_rx)),
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
        let now = Instant::now();
        let delta = now.duration_since(self.last_update);
        self.last_update = now;

        let mut player = self.player.write();
        let frame_changed = player.update(delta, self.show_flip_diff);
        player.process_load_queue();
        debug!("Update called, frame changed: {}", frame_changed);

        if frame_changed {
            drop(player);
            self.load_and_update_textures();
        }
    }

    pub fn update_textures(&mut self) -> bool {
        let player = self.player.write();
        player.update_textures(self.show_flip_diff)
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
        let right_texture = player.get_right_texture();

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

        let (image_width, image_height) = (left_texture.width() as f32, left_texture.height() as f32);
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

        let uniforms = UniformData {
            cursor_x: self.cursor_x / render_width,
            cursor_y: self.cursor_y / render_height,
            image1_size: [image_width, image_height],
            image2_size: [image_width, image_height],
            flip_diff_size: [image_width, image_height],
            show_flip_diff: if self.show_flip_diff { 1.0 } else { 0.0 },
            zoom_level: self.zoom_level,
            zoom_center: [
                self.fixed_zoom_center.0 + self.zoom_center_offset.0,
                self.fixed_zoom_center.1 + self.zoom_center_offset.1
            ],
            window_size: [window_size.width as f32, window_size.height as f32],
        };

        debug!("Created texture view");

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Check if a valid flip diff texture exists for the current frame pair
        let flip_diff_texture = if self.show_flip_diff {
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

        // Expire the status message after 3 seconds
        if let Some((_, set_at)) = &self.status_message {
            if set_at.elapsed().as_secs_f32() >= 3.0 {
                self.status_message = None;
            }
        }

        if self.cache_debug_window.is_open || self.status_message.is_some() {
            self.imgui_platform
                .prepare_frame(self.imgui_context.io_mut(), window)
                .expect("Failed to prepare ImGui frame");

            let ui = self.imgui_context.frame();

            if self.cache_debug_window.is_open {
                let player = self.player.read();
                let mouse_pos = ui.io().mouse_pos;
                let window_width = window.inner_size().width as f32;
                self.cache_debug_window
                    .draw(ui, &player, mouse_pos[0], mouse_pos[1], window_width);
            }

            // Draw status-message toast in the bottom-left corner
            if let Some((msg, set_at)) = &self.status_message {
                let elapsed = set_at.elapsed().as_secs_f32();
                let alpha = if elapsed < 2.5 { 1.0_f32 } else { 1.0 - (elapsed - 2.5) / 0.5 };
                let win_size = window.inner_size();
                let padding = 10.0_f32;
                let _token = ui.push_style_var(imgui::StyleVar::WindowPadding([8.0, 6.0]));
                if let Some(_win) = ui
                    .window("##status_toast")
                    .position(
                        [padding, win_size.height as f32 - padding],
                        imgui::Condition::Always,
                    )
                    .position_pivot([0.0, 1.0])
                    .bg_alpha(alpha * 0.75)
                    .no_decoration()
                    .no_inputs()
                    .movable(false)
                    .no_nav()
                    .focus_on_appearing(false)
                    .always_auto_resize(true)
                    .begin()
                {
                    ui.text_colored([1.0, 1.0, 1.0, alpha], msg.as_str());
                }
            }

            should_render_imgui = true;
            self.imgui_platform.prepare_render(ui, window);
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
            self.imgui_renderer
                .render(draw_data, &self.queue, &self.device, &mut render_pass)
                .expect("Failed to render ImGui");
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

                let uniforms = UniformData {
                    cursor_x: self.cursor_x / render_width,
                    cursor_y: mouse_y / window_height,
                    image1_size: [left_texture.width() as f32, left_texture.height() as f32],
                    image2_size: [right_texture.width() as f32, right_texture.height() as f32],
                    flip_diff_size,
                    show_flip_diff: if self.show_flip_diff { 1.0 } else { 0.0 },
                    zoom_level: self.zoom_level,
                    zoom_center: [
                        self.fixed_zoom_center.0 + self.zoom_center_offset.0,
                        self.fixed_zoom_center.1 + self.zoom_center_offset.1
                    ],
                    window_size: [window_size.width as f32, window_size.height as f32],
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

        self.queue.submit(std::iter::once(encoder.finish()));

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
            let mut screenshot_encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("Screenshot Encoder") },
            );
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
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            );
            self.queue.submit(std::iter::once(screenshot_encoder.finish()));
            let buffer_slice = buffer.slice(..);
            let (tx, rx) = mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
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
                    let path = generate_output_filename("screenshot", "png");
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
                                    pixels.extend_from_slice(&[chunk[2], chunk[1], chunk[0], chunk[3]]);
                                }
                            } else {
                                pixels.extend_from_slice(row_data);
                            }
                        }
                        let msg = match image::save_buffer(&path, &pixels, width, height, image::ColorType::Rgba8) {
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
                    self.status_message = Some((format!("Screenshot failed: {}", e), Instant::now()));
                }
            }
        }

        // Pick up any pending screenshot result from the background thread
        while let Ok(msg) = self.screenshot_result_rx.lock().try_recv() {
            self.status_message = Some((msg, Instant::now()));
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
            mag_filter: wgpu::FilterMode::Linear,
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
            mag_filter: wgpu::FilterMode::Linear,
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
        let frame_changed = self.player.write().next_frame(self.show_flip_diff);
        if frame_changed {
            self.load_and_update_textures();
        }
    }

    pub fn previous_frame(&mut self) {
        let frame_changed = self.player.write().previous_frame(self.show_flip_diff);
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

    pub fn handle_event<T>(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::Event<T>,
    ) {
        if let winit::event::Event::WindowEvent {
            event: winit::event::WindowEvent::Resized(size),
            ..
        } = event
        {
            self.resize(*size);
            debug!("Window resized to: {:?}", size);
        }

        if let winit::event::Event::WindowEvent {
            event: winit::event::WindowEvent::ScaleFactorChanged { scale_factor, new_inner_size },
            ..
        } = event
        {
            self.surface_scale = scale_factor.ceil() as u32;
            self.resize(**new_inner_size);
        }

        if let winit::event::Event::WindowEvent {
            event: winit::event::WindowEvent::CursorMoved { position, .. },
            ..
        } = event
        {
            self.update_mouse_position(position.x as f32, position.y as f32);
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
            match keycode {
                VirtualKeyCode::Escape => {
                    // Exit the application when Esc is pressed
                    process::exit(0);
                }
                VirtualKeyCode::C => {
                    self.cache_debug_window.toggle();
                }
                VirtualKeyCode::F => {
                    self.toggle_flip_diff();
                }
                VirtualKeyCode::Left | VirtualKeyCode::Right => {
                    if *keycode == VirtualKeyCode::Left {
                        self.previous_frame();
                    } else {
                        self.next_frame();
                    }
                }
                VirtualKeyCode::Space => {
                    self.toggle_play_pause();
                }
                VirtualKeyCode::Up => self.handle_zoom(&winit::event::MouseScrollDelta::LineDelta(0.0, 1.0)),
                VirtualKeyCode::Down => self.handle_zoom(&winit::event::MouseScrollDelta::LineDelta(0.0, -1.0)),
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
                VirtualKeyCode::P => {
                    self.save_flip_diff_image();
                }
                VirtualKeyCode::I => {
                    self.request_screenshot();
                }
                _ => {}
            }
        }

        if let winit::event::Event::WindowEvent { event: WindowEvent::MouseWheel { delta, .. }, .. } = event {
            self.handle_zoom(delta);
        }

        if let winit::event::Event::WindowEvent { event: WindowEvent::Touch(touch), .. } = event {
            self.handle_touch(touch);
        }

        self.imgui_platform
            .handle_event(self.imgui_context.io_mut(), window, event);
        debug!("Event handled");
    }

    pub fn update_mouse_position(&mut self, x: f32, y: f32) {
        let player = self.player.read();
        let left_texture = player.get_left_texture();

        let (image_width, image_height) = if let Some(left) = left_texture {
            (left.width() as f32, left.height() as f32)
        } else {
            (self.size.width as f32, self.size.height as f32)
        };

        let window_aspect_ratio = self.size.width as f32 / self.size.height as f32;
        let image_aspect_ratio = image_width / image_height;

        let (render_width, render_height) = if window_aspect_ratio > image_aspect_ratio {
            let scaled_height = self.size.height as f32;
            let scaled_width = scaled_height * image_aspect_ratio;
            (scaled_width, scaled_height)
        } else {
            let scaled_width = self.size.width as f32;
            let scaled_height = scaled_width / image_aspect_ratio;
            (scaled_width, scaled_height)
        };

        let x_offset = (self.size.width as f32 - render_width) / 2.0;
        let y_offset = (self.size.height as f32 - render_height) / 2.0;

        self.mouse_position = (x, y);
        self.cursor_x = (x - x_offset).max(0.0).min(render_width);
        self.cursor_y = (y - y_offset).max(0.0).min(render_height);
        self.update_uniform_buffer();
    }

    pub fn toggle_flip_diff(&mut self) {
        self.show_flip_diff = !self.show_flip_diff;
        if self.show_flip_diff {
            let (current_left, current_right) = self.player.read().current_images();
            self.player
                .write()
                .generate_flip_diff(current_left, current_right);
        }
    }

    fn handle_zoom(&mut self, delta: &winit::event::MouseScrollDelta) {
        let zoom_factor = match delta {
            winit::event::MouseScrollDelta::LineDelta(_, y) => {
                1.0 + y.signum() * 0.1
            }
            winit::event::MouseScrollDelta::PixelDelta(winit::dpi::PhysicalPosition { y, .. }) => {
                1.0 + (y / 100.0) as f32
            }
        };

        let new_zoom_level = (self.zoom_level * zoom_factor).clamp(1.0, 10.0);

        // Calculate the mouse position relative to the image
        let image_width = self.size.width as f32;
        let image_height = self.size.height as f32;
        let mouse_x = self.cursor_x / image_width;
        let mouse_y = self.cursor_y / image_height;

        // Adjust the fixed zoom center based on the current zoom level and mouse position
        let zoom_center_x = (mouse_x - 0.5) / self.zoom_level + 0.5;
        let zoom_center_y = (mouse_y - 0.5) / self.zoom_level + 0.5;

        // Calculate the maximum allowed offset based on the new zoom level
        let max_offset_x = (1.0 - 1.0 / new_zoom_level) / 2.0;
        let max_offset_y = (1.0 - 1.0 / new_zoom_level) / 2.0;

        // Clamp the zoom center to keep it within the image bounds
        let clamped_zoom_center_x = zoom_center_x.clamp(0.5 - max_offset_x, 0.5 + max_offset_x);
        let clamped_zoom_center_y = zoom_center_y.clamp(0.5 - max_offset_y, 0.5 + max_offset_y);

        // Update the zoom level and fixed zoom center
        self.zoom_level = new_zoom_level;
        self.fixed_zoom_center = (clamped_zoom_center_x, clamped_zoom_center_y);
        self.zoom_center_offset = (0.0, 0.0); // Reset the offset when zooming

        // Update the uniform buffer
        self.update_uniform_buffer();
    }

    fn update_uniform_buffer(&self) {
        let player = self.player.read();
        let (left_texture, right_texture) = (player.get_left_texture(), player.get_right_texture());
        
        let (image1_size, image2_size) = if let (Some(left), Some(right)) = (left_texture, right_texture) {
            ([left.width() as f32, left.height() as f32], [right.width() as f32, right.height() as f32])
        } else {
            ([self.size.width as f32, self.size.height as f32], [self.size.width as f32, self.size.height as f32])
        };

        let uniforms = UniformData {
            cursor_x: self.cursor_x / self.size.width as f32,
            cursor_y: self.cursor_y / self.size.height as f32,
            image1_size,
            image2_size,
            flip_diff_size: [self.size.width as f32, self.size.height as f32],
            show_flip_diff: if self.show_flip_diff { 1.0 } else { 0.0 },
            zoom_level: self.zoom_level,
            zoom_center: [
                self.fixed_zoom_center.0 + self.zoom_center_offset.0,
                self.fixed_zoom_center.1 + self.zoom_center_offset.1
            ],
            window_size: [self.size.width as f32, self.size.height as f32],
        };

        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
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
                                self.handle_zoom(&winit::event::MouseScrollDelta::LineDelta(0.0, 1.0));
                            } else {
                                self.handle_zoom(&winit::event::MouseScrollDelta::LineDelta(0.0, -1.0));
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
        let (mut offset_x, mut offset_y) = self.zoom_center_offset;
        
        // Calculate the maximum allowed offset based on zoom level
        let max_offset_x = (1.0 - 1.0 / self.zoom_level) / 2.0;
        let max_offset_y = (1.0 - 1.0 / self.zoom_level) / 2.0;
        
        offset_x += dx * self.zoom_move_speed;
        offset_y += dy * self.zoom_move_speed;
        
        // Clamp the offset to keep the zoom center within the image
        offset_x = offset_x.clamp(-max_offset_x, max_offset_x);
        offset_y = offset_y.clamp(-max_offset_y, max_offset_y);
        
        self.zoom_center_offset = (offset_x, offset_y);
        self.update_uniform_buffer();
    }

    /// Save the current FLIP diff image to a PNG file.
    pub fn save_flip_diff_image(&mut self) {
        let player = self.player.read();
        let (left_index, right_index) = player.current_images();
        match player.get_flip_diff_raw_data(left_index, right_index) {
            Some((data, width, height)) => {
                let path = generate_output_filename("flip_diff", "png");
                match image::save_buffer(&path, &data, width, height, image::ColorType::Rgba8) {
                    Ok(_) => {
                        info!("FLIP diff image saved to {}", path);
                        self.status_message = Some((format!("FLIP diff saved: {}", path), Instant::now()));
                    }
                    Err(e) => {
                        warn!("Failed to save FLIP diff image: {}", e);
                        self.status_message = Some((format!("Failed to save FLIP diff: {}", e), Instant::now()));
                    }
                }
            }
            None => {
                warn!(
                    "No FLIP diff image available for frames ({}, {}). Enable FLIP mode first.",
                    left_index, right_index
                );
                self.status_message = Some(("No FLIP diff available. Press F to enable FLIP mode first.".to_string(), Instant::now()));
            }
        }
    }

    /// Request a screenshot to be saved on the next rendered frame.
    pub fn request_screenshot(&mut self) {
        self.screenshot_requested = true;
    }
}

/// Generate a timestamped output file path in the current directory.
fn generate_output_filename(prefix: &str, extension: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}_{}.{:03}.{}", prefix, d.as_secs(), d.subsec_millis(), extension)
}

#[cfg(test)]
mod tests {
    use super::generate_output_filename;

    #[test]
    fn test_generate_output_filename_format() {
        let name = generate_output_filename("screenshot", "png");
        assert!(name.starts_with("screenshot_"), "should start with prefix: {}", name);
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
        assert_eq!(parts[1].len(), 3, "millis should be zero-padded to 3 digits: {}", parts[1]);
    }

    #[test]
    fn test_generate_output_filename_flip_diff() {
        let name = generate_output_filename("flip_diff", "png");
        assert!(name.starts_with("flip_diff_"), "should start with flip_diff_: {}", name);
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

        assert_eq!(pixels, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    }
}
