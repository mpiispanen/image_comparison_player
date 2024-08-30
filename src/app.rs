use crate::image_loader;
use crate::player::Player;
use imgui::{Condition, StyleVar};
use imgui::{StyleColor, Ui};
use log::{debug, error, info}; // Add error to the import list
use parking_lot::lock_api::RwLock;
use parking_lot::Mutex;
use parking_lot::RwLock as PLRwLock;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::event::VirtualKeyCode;
use winit::window::Window as WinitWindow;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default, Debug)]
struct UniformData {
    cursor_x: f32,
    image1_size: [f32; 2],
    image2_size: [f32; 2],
}

struct CacheDebugWindow {
    is_open: bool,
    size: [f32; 2],
}

impl CacheDebugWindow {
    fn new() -> Self {
        Self {
            is_open: false,
            size: [400.0, 150.0],
        }
    }

    fn draw(&mut self, ui: &Ui, player: &Player, mouse_x: f32, mouse_y: f32) {
        ui.window("Cache Debug")
            .size(self.size, Condition::Always)
            .resizable(true)
            .build(|| {
                let (current_left, current_right) = player.current_images();
                let frame_count = player.frame_count1;
                let visible_frames = player.preload_ahead * 2 + player.preload_behind * 2 + 1;

                self.draw_cache_row(
                    ui,
                    player,
                    true,
                    current_left,
                    frame_count,
                    mouse_x,
                    mouse_y,
                );
                ui.dummy([0.0, 10.0]); // Add some space between rows
                self.draw_cache_row(
                    ui,
                    player,
                    false,
                    current_right,
                    frame_count,
                    mouse_x,
                    mouse_y,
                );

                // Update size after potential resize
                self.size = ui.window_size();
            });
    }

    fn draw_cache_row(
        &self,
        ui: &Ui,
        player: &Player,
        is_left: bool,
        current: usize,
        frame_count: usize,
        mouse_x: f32,
        mouse_y: f32,
    ) {
        let cache = if is_left {
            &player.texture_cache_left
        } else {
            &player.texture_cache_right
        };
        let label = if is_left { "L:" } else { "R:" };

        ui.text(label);
        ui.same_line();

        let visible_frames = player.preload_ahead * 2 + player.preload_behind * 2 + 1;
        let button_size = 15.0;
        let spacing = 1.0;
        let total_width = visible_frames as f32 * (button_size + spacing) - spacing;

        ui.group(|| {
            ui.set_next_item_width(total_width);
            ui.dummy([total_width, button_size + 20.0]);

            let draw_list = ui.get_window_draw_list();
            let window_pos = ui.window_pos();
            let cursor_pos = ui.cursor_pos();

            let half_visible = visible_frames / 2;

            for i in 0..visible_frames {
                let frame = (current as i64 + i as i64 - half_visible as i64)
                    .rem_euclid(frame_count as i64) as usize;
                let x = window_pos[0] + cursor_pos[0] + i as f32 * (button_size + spacing);
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
                    .add_rect([x, y], [x + button_size, y + button_size], color)
                    .filled(true)
                    .build();

                if frame == current {
                    draw_list
                        .add_rect(
                            [x, y],
                            [x + button_size, y + button_size],
                            [1.0, 1.0, 1.0, 1.0],
                        )
                        .thickness(2.0)
                        .build();
                }

                if mouse_x >= x
                    && mouse_x <= x + button_size
                    && mouse_y >= y
                    && mouse_y <= y + button_size
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

    fn toggle(&mut self) {
        self.is_open = !self.is_open;
    }
}

pub struct AppState {
    surface: wgpu::Surface,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    player: Arc<RwLock<parking_lot::RawRwLock, Player>>,
    cursor_x: f32,
    last_update: Instant,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    imgui_context: imgui::Context,
    imgui_platform: imgui_winit_support::WinitPlatform,
    imgui_renderer: imgui_wgpu::Renderer,
    last_frame: std::time::Instant,
    cache_debug_window: CacheDebugWindow,
    uniform_bind_group: wgpu::BindGroup,
    mouse_position: (f32, f32),
}

impl AppState {
    pub async fn new(
        window: &WinitWindow,
        dir1: String,
        dir2: String,
        cache_size: usize,
        preload_ahead: usize,
        preload_behind: usize,
        num_load_threads: usize,
        num_process_threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Initializing AppState");
        let dir1 = std::fs::canonicalize(dir1)?;
        let dir2 = std::fs::canonicalize(dir2)?;

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
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
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
                ],
            });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: 24,
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
                -1.0f32, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
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
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
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

        let (images1, image_len1) = image_loader::load_image_paths(&dir1.to_str().unwrap())?;
        let (images2, image_len2) = image_loader::load_image_paths(&dir2.to_str().unwrap())?;
        debug!(
            "Loaded {} images from dir1 and {} images from dir2",
            image_len1, image_len2
        );

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let player = Arc::new(RwLock::new(Player::new(
            images1,
            images2,
            cache_size,
            preload_ahead,
            preload_behind,
            Arc::clone(&queue),
            Arc::clone(&device),
            num_load_threads,
            num_process_threads,
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

        let last_frame = std::time::Instant::now();

        let cache_debug_window = CacheDebugWindow::new();

        let mouse_position = (0.0, 0.0);

        info!("AppState initialized successfully");
        Ok(Self {
            surface,
            device,
            queue,
            size,
            render_pipeline,
            player,
            cursor_x: size.width as f32 / 2.0,
            last_update: Instant::now(),
            texture_bind_group_layout,
            uniform_buffer,
            uniform_bind_group_layout,
            vertex_buffer,
            imgui_context,
            imgui_platform,
            imgui_renderer,
            last_frame,
            cache_debug_window,
            uniform_bind_group,
            mouse_position,
        })
    }

    fn load_image_data_from_path(
        path: String,
    ) -> Result<(Vec<u8>, wgpu::Extent3d), Box<dyn std::error::Error>> {
        let img = image::open(path)?;
        let rgba = img.to_rgba8();
        let dimensions = rgba.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        Ok((rgba.into_raw(), size))
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
        player.update_textures()
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

        let texture_bind_group = self.create_texture_bind_group(&left_texture, &right_texture);

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
                image1_size: [left_texture.width() as f32, left_texture.height() as f32],
                image2_size: [right_texture.width() as f32, right_texture.height() as f32],
            };

            self.queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &texture_bind_group, &[]);
            render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
        }

        let mut should_render_imgui = false;

        if self.cache_debug_window.is_open {
            // Prepare ImGui frame only if needed
            self.imgui_platform
                .prepare_frame(self.imgui_context.io_mut(), window)
                .expect("Failed to prepare ImGui frame");

            let ui = self.imgui_context.frame();

            let player = self.player.read();
            let mouse_pos = ui.io().mouse_pos;
            self.cache_debug_window
                .draw(&ui, &player, mouse_pos[0], mouse_pos[1]);
            should_render_imgui = true;

            // Explicitly end the frame
            self.imgui_platform.prepare_render(&ui, window);
        }

        // Render ImGui only if there are UI elements
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

        // Submit the command encoder
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
        } // player is dropped here

        self.update_textures();
        self.player.write().process_loaded_textures();
    }

    pub fn update_cursor_position(&mut self, x: f32, _y: f32) {
        self.cursor_x = x;
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[UniformData {
                cursor_x: x / self.size.width as f32,
                image1_size: [0.0, 0.0], // These will be updated in the render function
                image2_size: [0.0, 0.0],
            }]),
        );
    }

    pub fn handle_mouse_click(&mut self, window: &winit::window::Window) {
        let cursor_position = window.inner_position().unwrap();
        let window_size = window.inner_size();
        if cursor_position.x >= 0
            && cursor_position.x <= window_size.width as i32
            && cursor_position.y >= 0
            && cursor_position.y <= window_size.height as i32
        {
            self.player.write().toggle_play_pause();
        }
    }

    pub fn handle_event<T>(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::Event<T>,
    ) {
        // Check if the event is a resize event and update the surface configuration
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
                            virtual_keycode: Some(VirtualKeyCode::C),
                            ..
                        },
                    ..
                },
            ..
        } = event
        {
            self.cache_debug_window.toggle();
        }

        // Handle the event
        self.imgui_platform
            .handle_event(self.imgui_context.io_mut(), window, event);
        debug!("Event handled");
    }

    pub fn update_mouse_position(&mut self, x: f32, y: f32) {
        self.mouse_position = (x, y);
        self.update_cursor_position(self.mouse_position.0, self.mouse_position.1);
    }
}
