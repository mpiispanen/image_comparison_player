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
        }
    }

    pub fn current_images(&mut self) -> (usize, usize) {
        let index1 = self.get_current_index(&self.image_data1);
        let index2 = self.get_current_index(&self.image_data2);

        if index1 != self.last_index1 {
            self.last_index1 = index1;
            self.request_texture_load(&self.image_data1[index1].0, index1 * 2);
        }

        if index2 != self.last_index2 {
            self.last_index2 = index2;
            self.request_texture_load(&self.image_data2[index2].0, index2 * 2 + 1);
        }

        self.trigger_preload(index1, index2);

        (index1 * 2, index2 * 2 + 1)
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
        log::info!(
            "Player is now {}",
            if self.is_playing { "playing" } else { "paused" }
        );
    }

    pub fn next_frame(&mut self) {
        let next_time = self
            .image_data1
            .iter()
            .find(|(_, _, end)| *end > self.current_time)
            .map(|(_, _, end)| *end)
            .unwrap_or_else(|| self.image_data1.last().map(|(_, _, end)| *end).unwrap_or(0));
        self.current_time = next_time;
        let total_duration = self.total_duration();
        debug!(
            "Current time: {} / {} ms",
            self.current_time, total_duration
        );
    }

    pub fn previous_frame(&mut self) {
        let prev_time = self
            .image_data1
            .iter()
            .rev()
            .find(|(_, _, start)| *start < self.current_time)
            .map(|(_, _, start)| *start)
            .unwrap_or(0);
        self.current_time = prev_time;
        let total_duration = self.total_duration();
        debug!(
            "Current time: {} / {} ms",
            self.current_time, total_duration
        );
    }

    pub fn advance_frame(&mut self, delta_time: u64) {
        let old_time = self.current_time;
        self.current_time += delta_time;
        let total_duration = self.total_duration();
        if self.current_time >= total_duration {
            self.current_time = self.current_time % total_duration;
        }
        debug!(
            "Advanced frame: old_time = {}, delta_time = {}, new_time = {}, total_duration = {}",
            old_time, delta_time, self.current_time, total_duration
        );
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

    pub fn update(&mut self) {
        if self.is_playing {
            let now = Instant::now();
            let delta = now.duration_since(self.last_update);
            self.last_update = now;
            self.advance_frame(delta.as_micros() as u64);
        }
        self.check_and_load_textures();
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
