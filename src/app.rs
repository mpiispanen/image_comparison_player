use crate::image_loader;
use crate::player::Player;
use log::{debug, info};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::window::Window;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default, Debug)]
struct UniformData {
    cursor_x: f32,
    image1_size: [f32; 2],
    image2_size: [f32; 2],
}

type TextureLoadRequest = (String, usize, bool);
type TextureLoadResponse = (usize, bool, Vec<u8>, wgpu::Extent3d);

pub struct AppState {
    surface: wgpu::Surface,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    player: Player,
    cursor_x: f32,
    last_update: Instant,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    left_texture: Arc<RwLock<Arc<wgpu::Texture>>>,
    right_texture: Arc<RwLock<Arc<wgpu::Texture>>>,
    texture_load_sender: std::sync::mpsc::Sender<TextureLoadRequest>,
    texture_load_receiver: std::sync::mpsc::Receiver<TextureLoadResponse>,
}

impl AppState {
    pub async fn new(
        window: &Window,
        dir1: String,
        dir2: String,
        cache_size: usize,
        preload_ahead: usize,
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
        let (texture_load_sender, texture_load_receiver) =
            std::sync::mpsc::channel::<TextureLoadRequest>();
        let (texture_ready_sender, texture_ready_receiver) =
            std::sync::mpsc::channel::<TextureLoadResponse>();

        let device_clone = Arc::clone(&device);
        let queue_clone = Arc::clone(&queue);

        std::thread::spawn(move || {
            while let Ok((path, index, is_left)) = texture_load_receiver.recv() {
                let (image_data, size) = Self::load_image_data_from_path(&path).unwrap();
                texture_ready_sender
                    .send((index, is_left, image_data, size))
                    .unwrap();
            }
        });

        let texture_load_sender_clone = texture_load_sender.clone();
        let player = Player::new(
            images1,
            images2,
            cache_size,
            preload_ahead,
            texture_load_sender_clone,
            Arc::clone(&queue),
            Arc::clone(&device),
        );
        debug!("Player initialized");

        let (left_texture, right_texture) = player.load_initial_textures()?;

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

        info!("AppState initialized successfully");
        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            player,
            cursor_x: size.width as f32 / 2.0,
            last_update: Instant::now(),
            texture_bind_group_layout,
            uniform_buffer,
            uniform_bind_group_layout,
            vertex_buffer,
            left_texture: Arc::new(RwLock::new(Arc::new(left_texture))),
            right_texture: Arc::new(RwLock::new(Arc::new(right_texture))),
            texture_load_sender,
            texture_load_receiver: texture_ready_receiver,
        })
    }

    fn load_image_data_from_path(
        path: &str,
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
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn update(&mut self) -> bool {
        let now = Instant::now();
        let delta = now.duration_since(self.last_update);
        self.last_update = now;

        let frame_changed = self.player.update(delta);
        debug!("Update called, frame changed: {}", frame_changed);

        // Always update textures, regardless of frame change
        let textures_updated = self.update_textures();

        if frame_changed {
            debug!("Frame changed");
            // Perform any additional frame change specific actions here
        }

        textures_updated
    }

    fn update_textures(&mut self) -> bool {
        let (left_index, right_index) = self.player.current_images();

        let mut updated = false;

        while let Ok((loaded_index, is_left, image_data, size)) =
            self.texture_load_receiver.try_recv()
        {
            self.player
                .add_texture_to_cache(loaded_index, is_left, &image_data, size);

            // Check if the loaded texture is for the current frame
            if (is_left && loaded_index == left_index) || (!is_left && loaded_index == right_index)
            {
                updated = true;
            }
        }

        // Ensure current textures are loaded
        if self.player.ensure_texture_loaded(left_index, true)
            || self.player.ensure_texture_loaded(right_index, false)
        {
            updated = true;
        }

        // Update the displayed textures if they are available or have been updated
        if updated
            || (self.player.get_texture(left_index, true).is_some()
                && self.player.get_texture(right_index, false).is_some())
        {
            *self.left_texture.write().unwrap() = self
                .player
                .get_texture(left_index, true)
                .unwrap_or_else(|| {
                    debug!(
                        "Using fallback texture for left image: index={}",
                        left_index
                    );
                    Arc::clone(&self.left_texture.read().unwrap())
                });
            *self.right_texture.write().unwrap() = self
                .player
                .get_texture(right_index, false)
                .unwrap_or_else(|| {
                    debug!(
                        "Using fallback texture for right image: index={}",
                        right_index
                    );
                    Arc::clone(&self.right_texture.read().unwrap())
                });
        }

        // Preload next textures
        self.player.preload_textures(left_index, right_index);

        updated
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        debug!("Starting render function");
        let (left_index, right_index) = self.player.current_images();
        debug!(
            "Current frame indices: left={}, right={}",
            left_index, right_index
        );

        let left_texture = self.player.get_texture(left_index, true);
        let right_texture = self.player.get_texture(right_index, false);

        debug!(
            "Left texture: {:?}",
            left_texture.as_ref().map(|t| t.size())
        );
        debug!(
            "Right texture: {:?}",
            right_texture.as_ref().map(|t| t.size())
        );

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

        let left_texture = self
            .player
            .get_texture(left_index, true)
            .unwrap_or_else(|| {
                debug!(
                    "Using fallback texture for left image: index={}",
                    left_index
                );
                Arc::clone(&self.left_texture.read().unwrap())
            });
        let right_texture = self
            .player
            .get_texture(right_index, false)
            .unwrap_or_else(|| {
                debug!(
                    "Using fallback texture for right image: index={}",
                    right_index
                );
                Arc::clone(&self.right_texture.read().unwrap())
            });

        let texture_bind_group = self.create_texture_bind_group(&left_texture, &right_texture);

        let uniforms = UniformData {
            cursor_x: self.cursor_x / self.size.width as f32,
            image1_size: [left_texture.width() as f32, left_texture.height() as f32],
            image2_size: [right_texture.width() as f32, right_texture.height() as f32],
        };

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let uniform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &self.uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniform_buffer.as_entire_binding(),
            }],
        });

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

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &texture_bind_group, &[]);
        render_pass.set_bind_group(1, &uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..6, 0..1);

        debug!("Issued draw calls");

        drop(render_pass);

        debug!("Finishing command encoder");
        let command_buffer = encoder.finish();

        debug!("Submitting command buffer");
        self.queue.submit(std::iter::once(command_buffer));

        debug!("Presenting output");
        output.present();

        Ok(())
    }

    fn create_texture_bind_group(
        &self,
        texture1: &Arc<wgpu::Texture>,
        texture2: &Arc<wgpu::Texture>,
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
        self.player.toggle_play_pause();
    }

    pub fn next_frame(&mut self) {
        self.player.next_frame();
        self.update_textures();
    }

    pub fn previous_frame(&mut self) {
        self.player.previous_frame();
        self.update_textures();
    }

    pub fn update_cursor_position(&mut self, x: f32, _y: f32) {
        self.cursor_x = x;
    }

    pub fn handle_mouse_click(&mut self) {
        self.player.toggle_play_pause();
    }
}
