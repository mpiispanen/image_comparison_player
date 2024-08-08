use log::debug;
use std::sync::{Arc, Mutex};
use std::time::Instant;

type PreloadRequest = (
    String,
    usize,
    Arc<Mutex<Option<Arc<wgpu::Texture>>>>,
    Arc<Mutex<bool>>,
);

type PreloadResponse = (
    usize,
    Vec<u8>,
    Arc<Mutex<Option<Arc<wgpu::Texture>>>>,
    Arc<Mutex<bool>>,
    u32,
    u32,
);

type TextureLoadRequest = (String, usize);

pub struct Player {
    image_data1: Vec<(String, u64, u64)>,
    image_data2: Vec<(String, u64, u64)>,
    current_time: u64,
    is_playing: bool,
    last_update: Instant,
    cache_size: usize,
    preload_ahead: usize,
    frame_count1: usize,
    frame_count2: usize,
    last_index1: usize,
    last_index2: usize,
    texture_load_sender: std::sync::mpsc::Sender<TextureLoadRequest>,
    queue: Arc<wgpu::Queue>,
    device: Arc<wgpu::Device>,
    current_frame1: usize,
    current_frame2: usize,
}

impl Player {
    pub fn new(
        image_data1: Vec<(String, u64, u64)>,
        image_data2: Vec<(String, u64, u64)>,
        cache_size: usize,
        preload_ahead: usize,
        texture_load_sender: std::sync::mpsc::Sender<TextureLoadRequest>,
        queue: Arc<wgpu::Queue>,
        device: Arc<wgpu::Device>,
    ) -> Self {
        let frame_count1 = image_data1.len();
        let frame_count2 = image_data2.len();
        Self {
            image_data1,
            image_data2,
            current_time: 0,
            is_playing: false,
            last_update: Instant::now(),
            cache_size,
            preload_ahead,
            frame_count1,
            frame_count2,
            last_index1: 0,
            last_index2: 0,
            texture_load_sender,
            queue,
            device,
            current_frame1: 0,
            current_frame2: 0,
        }
    }

    pub fn current_images(&mut self) -> (usize, usize) {
        (self.current_frame1 * 2, self.current_frame2 * 2 + 1)
    }

    fn request_texture_load(&self, path: &str, index: usize) {
        self.texture_load_sender
            .send((path.to_string(), index))
            .unwrap();
    }

    fn trigger_preload(&self, index1: usize, index2: usize) {
        for i in 1..=self.preload_ahead {
            let preload_index1 = (index1 + i) % self.frame_count1;
            let preload_index2 = (index2 + i) % self.frame_count2;

            self.request_texture_load(&self.image_data1[preload_index1].0, preload_index1 * 2);
            self.request_texture_load(&self.image_data2[preload_index2].0, preload_index2 * 2 + 1);
        }
    }

    fn get_current_index(&self, image_data: &[(String, u64, u64)]) -> usize {
        image_data
            .iter()
            .position(|(_, start, end)| *start <= self.current_time && self.current_time < *end)
            .unwrap_or(0)
    }

    pub fn is_playing(&self) -> bool {
        self.is_playing
    }

    pub fn toggle_play_pause(&mut self) {
        self.is_playing = !self.is_playing;
        self.last_update = Instant::now();
        debug!(
            "Player is now {}",
            if self.is_playing { "playing" } else { "paused" }
        );
    }

    pub fn next_frame(&mut self) {
        self.current_frame1 = (self.current_frame1 + 1) % self.frame_count1;
        self.current_frame2 = (self.current_frame2 + 1) % self.frame_count2;
        self.current_time = self.image_data1[self.current_frame1].1;
        self.update_textures();
    }

    pub fn previous_frame(&mut self) {
        self.current_frame1 = (self.current_frame1 + self.frame_count1 - 1) % self.frame_count1;
        self.current_frame2 = (self.current_frame2 + self.frame_count2 - 1) % self.frame_count2;
        self.current_time = self.image_data1[self.current_frame1].1;
        self.update_textures();
    }

    fn update_textures(&mut self) {
        let (index1, index2) = self.current_images();
        self.request_texture_load(&self.image_data1[self.current_frame1].0, index1);
        self.request_texture_load(&self.image_data2[self.current_frame2].0, index2);
        self.trigger_preload(self.current_frame1, self.current_frame2);
    }

    pub fn update(&mut self, delta: std::time::Duration) {
        if self.is_playing {
            self.advance_frame(delta.as_micros() as u64);
            self.update_textures();
        }
    }

    fn advance_frame(&mut self, delta_micros: u64) {
        self.current_time += delta_micros;
        if self.current_time >= self.total_duration() {
            self.current_time = 0;
        }
        self.current_frame1 = self.get_current_index(&self.image_data1);
        self.current_frame2 = self.get_current_index(&self.image_data2);
    }

    fn total_duration(&self) -> u64 {
        self.image_data1.last().map(|(_, _, end)| *end).unwrap_or(0)
    }

    pub fn load_initial_textures(
        &self,
    ) -> Result<(wgpu::Texture, wgpu::Texture), Box<dyn std::error::Error>> {
        let left_texture = self.load_texture(0)?;
        let right_texture = self.load_texture(1)?;
        Ok((left_texture, right_texture))
    }

    fn load_texture(&self, index: usize) -> Result<wgpu::Texture, Box<dyn std::error::Error>> {
        let image_data = if index % 2 == 0 {
            &self.image_data1
        } else {
            &self.image_data2
        };

        let path = &image_data[index / 2].0;
        let img = image::open(path)?;
        let rgba = img.to_rgba8();
        let dimensions = rgba.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Image Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );

        Ok(texture)
    }

    fn check_and_load_textures(&mut self) {
        let (index1, index2) = self.current_images();

        for i in 0..=self.preload_ahead {
            let preload_index1 = (index1 / 2 + i) % self.frame_count1;
            let preload_index2 = (index2 / 2 + i) % self.frame_count2;

            self.request_texture_load(&self.image_data1[preload_index1].0, preload_index1 * 2);
            self.request_texture_load(&self.image_data2[preload_index2].0, preload_index2 * 2 + 1);
        }
    }
}
