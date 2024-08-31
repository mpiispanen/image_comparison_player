use crate::image_loader;
use crate::player::Player;
use crate::player::PlayerConfig;
use imgui::Condition;
use imgui::Ui;
use log::{debug, info}; // Add error to the import list
use parking_lot::lock_api::RwLock;
use parking_lot::Mutex;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::event::VirtualKeyCode;
use winit::window::Window as WinitWindow;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default, Debug)]
struct UniformData {
    cursor_x: f32,
    cursor_y: f32,
    image1_size: [f32; 2],
    image2_size: [f32; 2],
    flip_diff_size: [f32; 2],
    show_flip_diff: f32,
}

struct CacheDebugWindow {
    is_open: bool,
    size: [f32; 2],
}

impl CacheDebugWindow {
    fn new() -> Self {
        Self {
            is_open: false,
            size: [400.0, 200.0], // Increased height to accommodate the new row
        }
    }

    fn draw(&mut self, ui: &Ui, player: &Player, mouse_x: f32, mouse_y: f32, window_width: f32) {
        let visible_frames = player.config.preload_ahead * 2 + player.config.preload_behind * 2 + 1;
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

        let visible_frames = player.config.preload_ahead * 2 + player.config.preload_behind * 2 + 1;
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

                if frame % 5 == 0 {
                    draw_list.add_text([x, y - 15.0], [1.0, 1.0, 1.0, 1.0], &frame.to_string());
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

                if params.mouse_pos.0 >= x
                    && params.mouse_pos.0 <= x + scaled_button_size
                    && params.mouse_pos.1 >= y
                    && params.mouse_pos.1 <= y + scaled_button_size
                {
                    let tooltip = if let Some(texture_info) =
                        player.texture_timings.read().get(&(frame, is_left))
                    {
                        format!(
                            "Frame {} (Loaded)\nLoad time: {:.2}ms\nProcess time: {:.2}ms",
                            frame,
                            texture_info.load_time.as_secs_f32() * 1000.0,
                            texture_info.process_time.as_secs_f32() * 1000.0
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

        let visible_frames = player.config.preload_ahead * 2 + player.config.preload_behind * 2 + 1;
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

                if left_frame % 5 == 0 {
                    draw_list.add_text(
                        [x, y - 15.0],
                        [1.0, 1.0, 1.0, 1.0],
                        &left_frame.to_string(),
                    );
                }

                let y = y + 5.0;

                let color = if player
                    .flip_diff_cache
                    .read()
                    .contains_key(&(left_frame, right_frame))
                {
                    [0.0, 1.0, 0.0, 1.0]
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
                    ui.tooltip_text(format!("Diff Cache: ({}, {})", left_frame, right_frame));
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
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

        let (images1, image_len1) = image_loader::load_image_paths(dir1.to_str().unwrap())?;
        let (images2, image_len2) = image_loader::load_image_paths(dir2.to_str().unwrap())?;
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
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        let delta = now.duration_since(self.last_update);
        self.last_update = now;

        let mut player = self.player.write();
        let frame_changed = player.update(delta);
        player.process_load_queue();
        debug!("Update called, frame changed: {}", frame_changed);

        if frame_changed {
            drop(player);
            self.load_and_update_textures();
        }
    }

    pub fn update_textures(&mut self) -> bool {
        let player = self.player.write();
        let frame_changed = player.update_textures();
        if frame_changed {
            self.flip_mode = false;
            self.show_flip_diff = false;
        }
        frame_changed
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

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

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
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            let uniforms = UniformData {
                cursor_x: self.cursor_x / self.size.width as f32,
                cursor_y: self.mouse_position.1 / self.size.height as f32,
                image1_size: [left_texture.width() as f32, left_texture.height() as f32],
                image2_size: [right_texture.width() as f32, right_texture.height() as f32],
                flip_diff_size: if use_flip_diff {
                    let flip_diff_texture = flip_diff_texture.clone().unwrap();
                    [
                        flip_diff_texture.width() as f32,
                        flip_diff_texture.height() as f32,
                    ]
                } else {
                    [0.0, 0.0]
                },
                show_flip_diff: if use_flip_diff { 1.0 } else { 0.0 },
            };

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

        if self.cache_debug_window.is_open {
            self.imgui_platform
                .prepare_frame(self.imgui_context.io_mut(), window)
                .expect("Failed to prepare ImGui frame");

            let ui = self.imgui_context.frame();

            let player = self.player.read();
            let mouse_pos = ui.io().mouse_pos;
            let window_width = window.inner_size().width as f32;
            self.cache_debug_window
                .draw(ui, &player, mouse_pos[0], mouse_pos[1], window_width);
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
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
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
                    cursor_x: self.cursor_x / self.size.width as f32,
                    cursor_y: mouse_y / window_height,
                    image1_size: [left_texture.width() as f32, left_texture.height() as f32],
                    image2_size: [right_texture.width() as f32, right_texture.height() as f32],
                    flip_diff_size,
                    show_flip_diff: if self.show_flip_diff { 1.0 } else { 0.0 },
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
        let frame_changed = self.player.write().next_frame();
        if frame_changed {
            self.load_and_update_textures();
        }
    }

    pub fn previous_frame(&mut self) {
        let frame_changed = self.player.write().previous_frame();
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

    pub fn update_cursor_position(&mut self, x: f32, y: f32) {
        self.cursor_x = x;
        self.cursor_y = y;
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[UniformData {
                cursor_x: x / self.size.width as f32,
                cursor_y: y / self.size.height as f32,
                image1_size: [0.0, 0.0],
                image2_size: [0.0, 0.0],
                flip_diff_size: [0.0, 0.0],
                show_flip_diff: if self.show_flip_diff { 1.0 } else { 0.0 },
            }]),
        );
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
                VirtualKeyCode::C => {
                    self.cache_debug_window.toggle();
                }
                VirtualKeyCode::F => {
                    self.show_flip_diff = !self.show_flip_diff;
                    if self.show_flip_diff {
                        self.generate_flip_diff();
                    }
                }
                VirtualKeyCode::Left | VirtualKeyCode::Right => {
                    if *keycode == VirtualKeyCode::Left {
                        self.previous_frame();
                    } else {
                        self.next_frame();
                    }
                }
                _ => {}
            }
        }

        self.imgui_platform
            .handle_event(self.imgui_context.io_mut(), window, event);
        debug!("Event handled");
    }

    pub fn update_mouse_position(&mut self, x: f32, y: f32) {
        self.mouse_position = (x, y);
        self.update_cursor_position(self.mouse_position.0, self.mouse_position.1);
    }

    fn generate_flip_diff(&mut self) {
        let (left_index, right_index) = {
            let player = self.player.read();
            player.current_images()
        };

        // Clear the old flip diff texture
        *self.flip_diff_texture.lock() = None;

        // Trigger the flip diff generation in the Player
        self.player
            .read()
            .generate_flip_diff(left_index, right_index);
    }
}
