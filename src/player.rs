use image::GenericImageView;
use log::{debug, warn};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use wgpu::util::DeviceExt;

type PreloadRequest = (
    String,
    usize,
    Arc<Mutex<Option<Arc<wgpu::Texture>>>>,
    Arc<Mutex<bool>>,
);

type PreloadResponse = (
    usize,
    Vec<u8>,
    u32,
    u32,
    Arc<Mutex<Option<Arc<wgpu::Texture>>>>,
);

pub struct Player {
    image_data1: Vec<(String, u64, u64)>,
    image_data2: Vec<(String, u64, u64)>,
    image_cache1: HashMap<usize, Arc<Mutex<Option<Arc<wgpu::Texture>>>>>,
    image_cache2: HashMap<usize, Arc<Mutex<Option<Arc<wgpu::Texture>>>>>,
    cache_order1: VecDeque<usize>,
    cache_order2: VecDeque<usize>,
    current_time: u64,
    is_playing: bool,
    last_update: Instant,
    preload_sender: std::sync::mpsc::Sender<PreloadRequest>,
    preload_receiver: std::sync::mpsc::Receiver<PreloadResponse>,
    ongoing_preloads1: Vec<Arc<Mutex<bool>>>,
    ongoing_preloads2: Vec<Arc<Mutex<bool>>>,
    preloaded_frames1: Vec<Arc<Mutex<bool>>>,
    preloaded_frames2: Vec<Arc<Mutex<bool>>>,
    cache_size: usize,
    preload_ahead: usize,
    frame_count1: usize,
    frame_count2: usize,
    last_index1: usize,
    last_index2: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl Player {
    pub fn new(
        images1: (Vec<(String, u64)>, usize),
        images2: (Vec<(String, u64)>, usize),
        cache_size: usize,
        preload_ahead: usize,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Self {
        let (image_data1, frame_count1) = Self::accumulate_durations(images1.0);
        let (image_data2, frame_count2) = Self::accumulate_durations(images2.0);

        let image_cache1 = HashMap::new();
        let image_cache2 = HashMap::new();
        let cache_order1 = VecDeque::new();
        let cache_order2 = VecDeque::new();

        let ongoing_preloads1 = vec![Arc::new(Mutex::new(false)); frame_count1];
        let ongoing_preloads2 = vec![Arc::new(Mutex::new(false)); frame_count2];

        let preloaded_frames1 = vec![Arc::new(Mutex::new(false)); frame_count1];
        let preloaded_frames2 = vec![Arc::new(Mutex::new(false)); frame_count2];

        let (sender, receiver) = std::sync::mpsc::channel();
        let (load_sender, load_receiver) = std::sync::mpsc::channel();

        let player = Self {
            image_data1,
            image_data2,
            image_cache1,
            image_cache2,
            cache_order1,
            cache_order2,
            current_time: 0,
            is_playing: false,
            last_update: Instant::now(),
            preload_sender: sender,
            preload_receiver: load_receiver,
            ongoing_preloads1,
            ongoing_preloads2,
            preloaded_frames1,
            preloaded_frames2,
            cache_size,
            preload_ahead,
            frame_count1,
            frame_count2,
            last_index1: usize::MAX,
            last_index2: usize::MAX,
            device,
            queue,
        };

        player.start_preload_thread(receiver, load_sender);
        player
    }

    fn start_preload_thread(
        &self,
        receiver: std::sync::mpsc::Receiver<PreloadRequest>,
        sender: std::sync::mpsc::Sender<PreloadResponse>,
    ) {
        thread::spawn(move || {
            while let Ok((path, index, cache, ongoing)) = receiver.recv() {
                debug!("Preload thread received request for index: {}", index);
                if let Ok(img) = image::open(&path) {
                    let rgba = img.to_rgba8();
                    let width = rgba.width();
                    let height = rgba.height();
                    let rgba_vec = rgba.into_raw();
                    sender
                        .send((index, rgba_vec, width, height, cache.clone()))
                        .unwrap();
                    debug!("Preload thread completed request for index: {}", index);
                }
                *ongoing.lock().unwrap() = false;
            }
        });
    }

    pub fn update(&mut self) {
        while let Ok((index, rgba, width, height, cache)) = self.preload_receiver.try_recv() {
            let texture = self.load_texture(&rgba, width, height);

            let is_cache1 = std::ptr::eq(
                Arc::as_ptr(&cache),
                self.image_cache1
                    .get(&index)
                    .map_or(std::ptr::null(), |arc| Arc::as_ptr(arc)),
            );

            let (cache, order, preloaded_frames) = if is_cache1 {
                (
                    &mut self.image_cache1,
                    &mut self.cache_order1,
                    &mut self.preloaded_frames1,
                )
            } else {
                (
                    &mut self.image_cache2,
                    &mut self.cache_order2,
                    &mut self.preloaded_frames2,
                )
            };

            if cache.len() >= self.cache_size && !cache.contains_key(&index) {
                if let Some(&oldest) = order.front() {
                    cache.remove(&oldest);
                    order.pop_front();
                }
            }

            cache.insert(index, Arc::new(Mutex::new(Some(Arc::new(texture)))));
            if !order.contains(&index) {
                order.push_back(index);
            }
            *preloaded_frames[index].lock().unwrap() = true;

            debug!("Added preloaded image to cache for index: {}", index);
        }
    }

    pub fn current_images(&mut self) -> (Arc<wgpu::Texture>, Arc<wgpu::Texture>) {
        let index1 = self.get_current_index(&self.image_data1);
        let index2 = self.get_current_index(&self.image_data2);

        debug!("Current indices: {} {}", index1, index2);

        let image1 = if index1 != self.last_index1 {
            self.last_index1 = index1;
            Self::get_or_load_image(
                &self.image_data1,
                index1,
                &mut self.image_cache1,
                &mut self.cache_order1,
                &self.device,
                &self.queue,
            )
        } else {
            self.image_cache1
                .get(&index1)
                .unwrap()
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .clone()
        };

        let image2 = if index2 != self.last_index2 {
            self.last_index2 = index2;
            Self::get_or_load_image(
                &self.image_data2,
                index2,
                &mut self.image_cache2,
                &mut self.cache_order2,
                &self.device,
                &self.queue,
            )
        } else {
            self.image_cache2
                .get(&index2)
                .unwrap()
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .clone()
        };

        self.trigger_preload(index1, index2);

        (image1, image2)
    }

    fn get_or_load_image(
        image_data: &[(String, u64, u64)],
        index: usize,
        cache: &mut HashMap<usize, Arc<Mutex<Option<Arc<wgpu::Texture>>>>>,
        cache_order: &mut VecDeque<usize>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Arc<wgpu::Texture> {
        if cache.is_empty() {
            // Handle empty cache case
            let (path, _, _) = &image_data[index];
            debug!(
                "Cache is empty. Loading image from path: {} for index {}",
                path, index
            );
            return Self::load_image(path, index, cache, cache_order, device, queue);
        }

        if let Some(entry) = cache.get(&index) {
            if let Some(image) = entry.lock().unwrap().as_ref() {
                debug!("Image found in cache for index {}", index);
                return Arc::clone(image);
            }
        }

        let (path, _, _) = &image_data[index];
        debug!("Loading image from path: {} for index {}", path, index);
        let texture = Self::load_image(path, index, cache, cache_order, device, queue);
        debug!("Successfully loaded image for index {}", index);
        texture
    }

    fn load_image(
        path: &str,
        index: usize,
        cache: &mut HashMap<usize, Arc<Mutex<Option<Arc<wgpu::Texture>>>>>,
        cache_order: &mut VecDeque<usize>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Arc<wgpu::Texture> {
        debug!("Loading image from path: {} for index {}", path, index);
        let texture = match Self::load_texture_from_path(path, device, queue) {
            Ok(texture) => texture,
            Err(e) => {
                warn!("Failed to load image at {}: {}", path, e);
                // Create a placeholder texture
                let size = wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                };
                device.create_texture(&wgpu::TextureDescriptor {
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    label: Some("placeholder_texture"),
                    view_formats: &[],
                })
            }
        };
        let arc_texture = Arc::new(texture);
        cache.insert(index, Arc::new(Mutex::new(Some(Arc::clone(&arc_texture)))));
        cache_order.push_back(index);
        debug!("Successfully loaded image for index {}", index);
        arc_texture
    }

    fn trigger_preload(&mut self, index1: usize, index2: usize) {
        for i in 1..=self.preload_ahead {
            let preload_index1 = (index1 + i) % self.frame_count1;
            let preload_index2 = (index2 + i) % self.frame_count2;

            Self::handle_preload(
                preload_index1,
                &self.image_data1,
                &mut self.image_cache1,
                &mut self.ongoing_preloads1,
                &mut self.preloaded_frames1,
                &self.preload_sender,
            );
            Self::handle_preload(
                preload_index2,
                &self.image_data2,
                &mut self.image_cache2,
                &mut self.ongoing_preloads2,
                &mut self.preloaded_frames2,
                &self.preload_sender,
            );
        }
    }

    fn handle_preload(
        index: usize,
        image_data: &[(String, u64, u64)],
        cache: &mut HashMap<usize, Arc<Mutex<Option<Arc<wgpu::Texture>>>>>,
        ongoing_preloads: &mut [Arc<Mutex<bool>>],
        preloaded_frames: &mut [Arc<Mutex<bool>>],
        preload_sender: &std::sync::mpsc::Sender<PreloadRequest>,
    ) {
        let in_cache = cache
            .get(&index)
            .map_or(false, |entry| entry.lock().unwrap().is_some());
        let in_ongoing = *ongoing_preloads[index].lock().unwrap();

        if in_cache {
            *ongoing_preloads[index].lock().unwrap() = false;
            *preloaded_frames[index].lock().unwrap() = true;
        } else if !in_ongoing {
            let (path, _, _) = &image_data[index];
            *ongoing_preloads[index].lock().unwrap() = true;
            *preloaded_frames[index].lock().unwrap() = false;

            let cache_slot = cache
                .entry(index)
                .or_insert_with(|| Arc::new(Mutex::new(None)))
                .clone();
            let ongoing_flag = Arc::clone(&ongoing_preloads[index]);

            preload_sender
                .send((path.clone(), index, cache_slot, ongoing_flag))
                .unwrap();
        }
    }

    fn get_next_n_index(
        &self,
        image_data: &[(String, u64, u64)],
        current_index: usize,
        n: usize,
    ) -> usize {
        let mut index = current_index;
        for _ in 0..n {
            index = (index + 1) % image_data.len();
        }
        index
    }

    fn accumulate_durations(images: Vec<(String, u64)>) -> (Vec<(String, u64, u64)>, usize) {
        let len = images.len();
        let mut accumulated = 0;
        let accumulated_images = images
            .into_iter()
            .map(|(path, duration)| {
                let start = accumulated;
                accumulated += duration;
                (path, start, accumulated)
            })
            .collect();
        (accumulated_images, len)
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

    fn get_current_index(&self, image_data: &[(String, u64, u64)]) -> usize {
        image_data
            .iter()
            .position(|(_, start, end)| *start <= self.current_time && self.current_time < *end)
            .unwrap_or(0)
    }

    fn total_duration(&self) -> u64 {
        self.image_data1.last().map(|(_, _, end)| *end).unwrap_or(0)
    }

    fn load_texture(&self, rgba: &[u8], width: u32, height: u32) -> wgpu::Texture {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("image_texture"),
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );

        texture
    }

    fn load_texture_from_path(
        path: &str,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<wgpu::Texture, Box<dyn std::error::Error>> {
        let img = image::open(path)?;
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();

        // Add error logging
        debug!("Loading texture from path: {}", path);
        debug!("Image dimensions: {}x{}", dimensions.0, dimensions.1);

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("diffuse_texture"),
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
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

    pub fn load_initial_textures(
        &mut self,
    ) -> Result<(Arc<wgpu::Texture>, Arc<wgpu::Texture>), Box<dyn std::error::Error>> {
        self.update();
        debug!(
            "Attempting to load left texture from: {}",
            &self.image_data1[0].0
        );
        let left = Arc::new(Self::load_texture_from_path(
            &self.image_data1[0].0,
            &self.device,
            &self.queue,
        )?);
        debug!(
            "Attempting to load right texture from: {}",
            &self.image_data2[0].0
        );
        let right = Arc::new(Self::load_texture_from_path(
            &self.image_data2[0].0,
            &self.device,
            &self.queue,
        )?);
        Ok((left, right))
    }
}
