use log::debug;
use parking_lot::RwLock as PLRwLock;
use std::collections::{HashMap, HashSet};
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

type TextureLoadRequest = (String, usize, bool);

pub struct Player {
    image_data1: Vec<(String, u64, u64)>,
    image_data2: Vec<(String, u64, u64)>,
    current_time: u64,
    is_playing: bool,
    last_update: Instant,
    cache_size: usize,
    preload_ahead: usize,
    preload_behind: usize,
    frame_count1: usize,
    frame_count2: usize,
    last_index1: usize,
    last_index2: usize,
    texture_load_sender: std::sync::mpsc::Sender<TextureLoadRequest>,
    queue: Arc<wgpu::Queue>,
    device: Arc<wgpu::Device>,
    current_frame1: usize,
    current_frame2: usize,
    pub texture_cache_left: Arc<PLRwLock<HashMap<usize, Arc<Mutex<Option<Arc<wgpu::Texture>>>>>>>,
    pub texture_cache_right: Arc<PLRwLock<HashMap<usize, Arc<Mutex<Option<Arc<wgpu::Texture>>>>>>>,
    requested_textures_left: HashSet<usize>,
    requested_textures_right: HashSet<usize>,
    cache_order_left: Vec<usize>,
    cache_order_right: Vec<usize>,
    texture_reuse_pool: Vec<Arc<wgpu::Texture>>,
    frame_changed: bool,
}

impl Player {
    pub fn new(
        image_data1: Vec<(String, u64, u64)>,
        image_data2: Vec<(String, u64, u64)>,
        cache_size: usize,
        preload_ahead: usize,
        preload_behind: usize,
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
            preload_behind,
            frame_count1,
            frame_count2,
            last_index1: 0,
            last_index2: 0,
            texture_load_sender,
            queue,
            device,
            current_frame1: 0,
            current_frame2: 0,
            texture_cache_left: Arc::new(PLRwLock::new(HashMap::with_capacity(cache_size / 2))),
            texture_cache_right: Arc::new(PLRwLock::new(HashMap::with_capacity(cache_size / 2))),
            requested_textures_left: HashSet::new(),
            requested_textures_right: HashSet::new(),
            cache_order_left: Vec::with_capacity(cache_size / 2),
            cache_order_right: Vec::with_capacity(cache_size / 2),
            texture_reuse_pool: Vec::new(),
            frame_changed: false,
        }
    }

    pub fn current_images(&mut self) -> (usize, usize) {
        (self.current_frame1, self.current_frame2)
    }

    pub fn get_texture(&self, index: usize, is_left: bool) -> Option<Arc<wgpu::Texture>> {
        let cache = if is_left {
            &self.texture_cache_left
        } else {
            &self.texture_cache_right
        };

        let cache_read = cache.read();
        cache_read
            .get(&index)
            .and_then(|texture_holder| texture_holder.lock().ok().and_then(|guard| guard.clone()))
    }

    fn request_texture_load(&self, path: &str, index: usize, is_left: bool) {
        self.texture_load_sender
            .send((path.to_string(), index, is_left))
            .unwrap();
    }

    fn trigger_preload(&self, index1: usize, index2: usize) {
        for i in 1..=self.preload_ahead {
            let preload_index1 = (index1 + i) % self.frame_count1;
            let preload_index2 = (index2 + i) % self.frame_count2;

            self.request_texture_load(&self.image_data1[preload_index1].0, preload_index1, true);
            self.request_texture_load(&self.image_data2[preload_index2].0, preload_index2, false);
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

    pub fn next_frame(&mut self) -> bool {
        let old_frame1 = self.current_frame1;
        let old_frame2 = self.current_frame2;
        self.current_frame1 = (self.current_frame1 + 1) % self.frame_count1;
        self.current_frame2 = (self.current_frame2 + 1) % self.frame_count2;
        self.current_time = std::cmp::max(
            self.image_data1[self.current_frame1].1,
            self.image_data2[self.current_frame2].1,
        );
        self.frame_changed = true;
        debug!(
            "Next frame: {} -> {}, {} -> {}",
            old_frame1, self.current_frame1, old_frame2, self.current_frame2
        );
        self.update_textures();
        true
    }

    pub fn previous_frame(&mut self) -> bool {
        let old_frame1 = self.current_frame1;
        let old_frame2 = self.current_frame2;
        self.current_frame1 = (self.current_frame1 + self.frame_count1 - 1) % self.frame_count1;
        self.current_frame2 = (self.current_frame2 + self.frame_count2 - 1) % self.frame_count2;
        self.current_time = std::cmp::min(
            self.image_data1[self.current_frame1].1,
            self.image_data2[self.current_frame2].1,
        );
        self.frame_changed = true;
        debug!(
            "Previous frame: {} -> {}, {} -> {}",
            old_frame1, self.current_frame1, old_frame2, self.current_frame2
        );
        self.update_textures();
        true
    }

    pub fn has_frame_changed(&mut self) -> bool {
        let changed = self.frame_changed;
        self.frame_changed = false;
        changed
    }

    fn update_textures(&mut self) {
        let index1 = self.current_frame1;
        let index2 = self.current_frame2;
        debug!("Updating textures: left={}, right={}", index1, index2);
        debug!(
            "Texture cache state: left={:?}, right={:?}",
            self.texture_cache_left.read().keys(),
            self.texture_cache_right.read().keys()
        );
        self.ensure_texture_loaded(index1, true);
        self.ensure_texture_loaded(index2, false);
        self.preload_textures(index1, index2);
        debug!(
            "After update, texture cache state: left={:?}, right={:?}",
            self.texture_cache_left.read().keys(),
            self.texture_cache_right.read().keys()
        );
    }

    pub fn ensure_texture_loaded(&self, index: usize, is_left: bool) -> bool {
        if !self.is_within_preload_range(index, is_left) {
            return false;
        }

        let cache = if is_left {
            &self.texture_cache_left
        } else {
            &self.texture_cache_right
        };

        let cache_read = cache.read();
        if !cache_read.contains_key(&index) {
            drop(cache_read);
            let mut cache_write = cache.write();
            cache_write.entry(index).or_insert_with(|| {
                let texture_holder = Arc::new(Mutex::new(None));
                let path = if is_left {
                    &self.image_data1[index].0
                } else {
                    &self.image_data2[index].0
                };
                self.texture_load_sender
                    .send((path.to_string(), index, is_left))
                    .unwrap();
                texture_holder
            });
            true
        } else {
            false
        }
    }

    pub fn preload_textures(&mut self, index1: usize, index2: usize) {
        let frame_count1 = self.frame_count1;
        let frame_count2 = self.frame_count2;

        // Preload ahead
        for i in 1..=self.preload_ahead {
            let preload_index1 = (index1 + i) % frame_count1;
            let preload_index2 = (index2 + i) % frame_count2;
            if !self.ensure_texture_loaded(preload_index1, true) {
                break;
            }
            if !self.ensure_texture_loaded(preload_index2, false) {
                break;
            }
        }

        // Preload behind
        for i in 1..=self.preload_behind {
            let preload_index1 = (index1 + frame_count1 - i) % frame_count1;
            let preload_index2 = (index2 + frame_count2 - i) % frame_count2;
            if !self.ensure_texture_loaded(preload_index1, true) {
                break;
            }
            if !self.ensure_texture_loaded(preload_index2, false) {
                break;
            }
        }
    }

    pub fn add_texture_to_cache(
        &mut self,
        index: usize,
        is_left: bool,
        image_data: &[u8],
        size: wgpu::Extent3d,
    ) -> bool {
        if !self.is_within_preload_range(index, is_left) {
            debug!(
                "Skipping texture addition to cache: index={}, is_left={} (out of preload range)",
                index, is_left
            );
            return false;
        }

        debug!(
            "Adding texture to cache: index={}, is_left={}, size={:?}",
            index, is_left, size
        );
        let index_to_evict = self.get_index_to_evict(is_left);
        let texture = self.get_or_create_texture(size);
        let cache = if is_left {
            &mut self.texture_cache_left
        } else {
            &mut self.texture_cache_right
        };
        let requested = if is_left {
            &mut self.requested_textures_left
        } else {
            &mut self.requested_textures_right
        };
        let cache_order = if is_left {
            &mut self.cache_order_left
        } else {
            &mut self.cache_order_right
        };

        if cache.read().len() >= self.cache_size / 2 {
            debug!("Cache full, evicting texture");
            if let Some(index) = index_to_evict {
                let texture = cache.write().remove(&index).unwrap();
                if let Some(inner_texture) = texture.lock().unwrap().take() {
                    self.texture_reuse_pool.push(inner_texture);
                }
                cache_order.retain(|&x| x != index);
            }
        }

        let queue = &self.queue;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            image_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * size.width),
                rows_per_image: Some(size.height),
            },
            size,
        );

        cache
            .write()
            .insert(index, Arc::new(Mutex::new(Some(texture))));
        cache_order.push(index);
        requested.remove(&index);
        debug!(
            "Texture added to cache. New cache state: {:?}",
            cache.read().keys()
        );

        // Check if the loaded texture is for the current frame
        let (current_left_index, current_right_index) = self.current_images();
        (is_left && index == current_left_index) || (!is_left && index == current_right_index)
    }

    fn get_index_to_evict(&self, is_left: bool) -> Option<usize> {
        let cache_order = if is_left {
            &self.cache_order_left
        } else {
            &self.cache_order_right
        };
        let current_frame = if is_left {
            self.current_frame1
        } else {
            self.current_frame2
        };
        let frame_count = if is_left {
            self.frame_count1
        } else {
            self.frame_count2
        };

        cache_order
            .iter()
            .max_by_key(|&&index| {
                let forward_distance = (index + frame_count - current_frame) % frame_count;
                let backward_distance = (current_frame + frame_count - index) % frame_count;
                std::cmp::min(forward_distance, backward_distance)
            })
            .copied()
    }

    fn get_or_create_texture(&mut self, size: wgpu::Extent3d) -> Arc<wgpu::Texture> {
        if let Some(texture) = self.texture_reuse_pool.pop() {
            if texture.size() == size {
                return texture;
            }
            // If size doesn't match, let it drop and create a new one
        }

        Arc::new(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Reused Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }))
    }

    fn distance_from_current_frame(&self, index: usize, is_left: bool) -> usize {
        let current_frame = if is_left {
            self.current_frame1
        } else {
            self.current_frame2
        };
        let frame_count = if is_left {
            self.frame_count1
        } else {
            self.frame_count2
        };

        std::cmp::min(
            (index + frame_count - current_frame) % frame_count,
            (current_frame + frame_count - index) % frame_count,
        )
    }

    pub fn update(&mut self, delta: std::time::Duration) -> bool {
        if self.is_playing {
            self.advance_frame(delta.as_micros() as u64);
            true
        } else {
            false
        }
    }

    fn advance_frame(&mut self, delta_micros: u64) {
        self.current_time += delta_micros;
        if self.current_time >= self.total_duration() {
            self.current_time %= self.total_duration();
        }

        let old_frame1 = self.current_frame1;
        let old_frame2 = self.current_frame2;

        self.current_frame1 = self.get_current_index(&self.image_data1);
        self.current_frame2 = self.get_current_index(&self.image_data2);

        if self.current_frame1 != old_frame1 || self.current_frame2 != old_frame2 {
            self.frame_changed = true;
            self.update_textures();
        }
    }

    fn total_duration(&self) -> u64 {
        self.image_data1.last().map(|(_, _, end)| *end).unwrap_or(0)
    }

    pub fn load_initial_textures(
        &self,
    ) -> Result<(wgpu::Texture, wgpu::Texture), Box<dyn std::error::Error>> {
        let left_texture = self.load_texture(0, true)?;
        let right_texture = self.load_texture(0, false)?;
        Ok((left_texture, right_texture))
    }

    fn load_texture(
        &self,
        index: usize,
        is_left: bool,
    ) -> Result<wgpu::Texture, Box<dyn std::error::Error>> {
        let image_data = if is_left {
            &self.image_data1
        } else {
            &self.image_data2
        };

        let path = &image_data[index].0;
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
            let preload_index1 = (index1 + i) % self.frame_count1;
            let preload_index2 = (index2 + i) % self.frame_count2;

            self.ensure_texture_loaded(preload_index1, true);
            self.ensure_texture_loaded(preload_index2, false);
        }
    }

    pub fn is_within_preload_range(&self, index: usize, is_left: bool) -> bool {
        let current_frame = if is_left {
            self.current_frame1
        } else {
            self.current_frame2
        };
        let frame_count = if is_left {
            self.frame_count1
        } else {
            self.frame_count2
        };

        let forward_distance = (index + frame_count - current_frame) % frame_count;
        let backward_distance = (current_frame + frame_count - index) % frame_count;

        let min_distance = std::cmp::min(forward_distance, backward_distance);

        min_distance <= self.preload_ahead || min_distance <= self.preload_behind
    }
}
