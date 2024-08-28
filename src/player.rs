use log::debug;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use threadpool::ThreadPool;

pub type TextureLoadRequest = (String, usize, bool);

pub struct PriorityTextureLoadQueue {
    queue: Arc<Mutex<VecDeque<TextureLoadRequest>>>,
    unique_requests: Arc<Mutex<HashSet<(usize, bool)>>>,
    frame_count_left: usize,
    frame_count_right: usize,
}

impl PriorityTextureLoadQueue {
    fn new(frame_count_left: usize, frame_count_right: usize) -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            unique_requests: Arc::new(Mutex::new(HashSet::new())),
            frame_count_left,
            frame_count_right,
        }
    }

    pub fn push(&self, request: TextureLoadRequest) {
        let key = (request.1, request.2);
        let mut unique_requests = self.unique_requests.lock();
        if !unique_requests.contains(&key) {
            self.queue.lock().push_back(request);
            unique_requests.insert(key);
        }
    }

    pub fn pop(&self) -> Option<TextureLoadRequest> {
        let mut queue = self.queue.lock();
        let mut unique_requests = self.unique_requests.lock();
        if let Some(request) = queue.pop_front() {
            unique_requests.remove(&(request.1, request.2));
            Some(request)
        } else {
            None
        }
    }

    pub fn clear(&self) {
        self.queue.lock().clear();
        self.unique_requests.lock().clear();
    }

    pub fn reprioritize(&self, current_frame_left: usize, current_frame_right: usize) {
        let mut queue = self.queue.lock();
        queue.make_contiguous().sort_by_key(|req| {
            let (current_frame, frame_count) = if req.2 {
                (current_frame_left, self.frame_count_left)
            } else {
                (current_frame_right, self.frame_count_right)
            };
            std::cmp::min(
                (req.1 as i32 - current_frame as i32).abs() as usize,
                frame_count - (req.1 as i32 - current_frame as i32).abs() as usize,
            )
        });
    }

    pub fn contains(&self, key: &(usize, bool)) -> bool {
        self.unique_requests.lock().contains(key)
    }
}

pub struct Player {
    image_data1: Vec<(String, u64, u64)>,
    image_data2: Vec<(String, u64, u64)>,
    current_time: AtomicU64,
    is_playing: AtomicBool,
    last_update: AtomicU64,
    cache_size: usize,
    pub preload_ahead: usize,
    pub preload_behind: usize,
    pub frame_count1: usize,
    pub frame_count2: usize,
    last_index1: usize,
    last_index2: usize,
    queue: Arc<wgpu::Queue>,
    device: Arc<wgpu::Device>,
    current_frame1: AtomicUsize,
    current_frame2: AtomicUsize,
    pub texture_cache_left: Arc<RwLock<HashMap<usize, Arc<Mutex<Option<Arc<wgpu::Texture>>>>>>>,
    pub texture_cache_right: Arc<RwLock<HashMap<usize, Arc<Mutex<Option<Arc<wgpu::Texture>>>>>>>,
    requested_textures_left: HashSet<usize>,
    requested_textures_right: HashSet<usize>,
    cache_order_left: Vec<usize>,
    cache_order_right: Vec<usize>,
    texture_reuse_pool: Arc<Mutex<Vec<Arc<wgpu::Texture>>>>,
    frame_changed: AtomicBool,
    pub texture_load_queue: Arc<Mutex<PriorityTextureLoadQueue>>,
    texture_load_pool: ThreadPool,
    texture_load_sender: Sender<TextureLoadRequest>,
    texture_load_receiver: Arc<Mutex<Receiver<(String, usize, bool)>>>,
    texture_process_pool: ThreadPool,
    texture_process_sender: Sender<(usize, bool, Vec<u8>, wgpu::Extent3d)>,
    texture_process_receiver: Arc<Mutex<Receiver<(usize, bool, Vec<u8>, wgpu::Extent3d)>>>,
    left_texture: Arc<Mutex<Option<Arc<wgpu::Texture>>>>,
    right_texture: Arc<Mutex<Option<Arc<wgpu::Texture>>>>,
    processing_textures: Arc<Mutex<HashSet<(usize, bool)>>>,
}

impl Player {
    pub fn new(
        image_data1: Vec<(String, u64, u64)>,
        image_data2: Vec<(String, u64, u64)>,
        cache_size: usize,
        preload_ahead: usize,
        preload_behind: usize,
        queue: Arc<wgpu::Queue>,
        device: Arc<wgpu::Device>,
        num_load_threads: usize,
        num_process_threads: usize,
    ) -> Self {
        let frame_count1 = image_data1.len();
        let frame_count2 = image_data2.len();
        let (texture_load_sender, texture_load_receiver) = channel();
        let texture_load_receiver = Arc::new(Mutex::new(texture_load_receiver));
        let (texture_process_sender, texture_process_receiver) = channel();
        let texture_process_receiver = Arc::new(Mutex::new(texture_process_receiver));

        let texture_load_pool = ThreadPool::new(num_load_threads);
        let texture_process_pool = ThreadPool::new(num_process_threads);

        Self {
            image_data1,
            image_data2,
            current_time: AtomicU64::new(0),
            is_playing: AtomicBool::new(false),
            last_update: AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            ),
            cache_size,
            preload_ahead,
            preload_behind,
            frame_count1,
            frame_count2,
            last_index1: 0,
            last_index2: 0,
            queue,
            device,
            current_frame1: AtomicUsize::new(0),
            current_frame2: AtomicUsize::new(0),
            texture_cache_left: Arc::new(RwLock::new(HashMap::with_capacity(cache_size / 2))),
            texture_cache_right: Arc::new(RwLock::new(HashMap::with_capacity(cache_size / 2))),
            requested_textures_left: HashSet::new(),
            requested_textures_right: HashSet::new(),
            cache_order_left: Vec::with_capacity(cache_size / 2),
            cache_order_right: Vec::with_capacity(cache_size / 2),
            texture_reuse_pool: Arc::new(Mutex::new(Vec::new())),
            frame_changed: AtomicBool::new(false),
            texture_load_queue: Arc::new(Mutex::new(PriorityTextureLoadQueue::new(
                frame_count1,
                frame_count2,
            ))),
            texture_load_pool,
            texture_load_sender,
            texture_load_receiver,
            texture_process_pool,
            texture_process_sender,
            texture_process_receiver,
            left_texture: Arc::new(Mutex::new(None)),
            right_texture: Arc::new(Mutex::new(None)),
            processing_textures: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    pub fn current_images(&self) -> (usize, usize) {
        (
            self.current_frame1.load(Ordering::Relaxed),
            self.current_frame2.load(Ordering::Relaxed),
        )
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
            .and_then(|texture_holder| texture_holder.lock().as_ref().cloned())
    }

    fn get_current_index(&self, image_data: &[(String, u64, u64)], current_time: u64) -> usize {
        image_data
            .iter()
            .position(|(_, start, end)| *start <= current_time && current_time < *end)
            .unwrap_or(0)
    }

    pub fn is_playing(&self) -> bool {
        self.is_playing.load(Ordering::Relaxed)
    }

    pub fn toggle_play_pause(&self) {
        self.is_playing.fetch_xor(true, Ordering::Relaxed);
        self.last_update.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            Ordering::Relaxed,
        );
    }

    pub fn next_frame(&self) -> bool {
        let old_frame1 = self.current_frame1.fetch_add(1, Ordering::Relaxed);
        let old_frame2 = self.current_frame2.fetch_add(1, Ordering::Relaxed);

        let new_frame1 = (old_frame1 + 1) % self.frame_count1;
        let new_frame2 = (old_frame2 + 1) % self.frame_count2;

        self.current_frame1.store(new_frame1, Ordering::Relaxed);
        self.current_frame2.store(new_frame2, Ordering::Relaxed);

        let changed = old_frame1 != new_frame1 || old_frame2 != new_frame2;
        self.frame_changed.store(changed, Ordering::Relaxed);

        changed
    }

    pub fn previous_frame(&self) -> bool {
        let old_frame1 = self.current_frame1.load(Ordering::Relaxed);
        let old_frame2 = self.current_frame2.load(Ordering::Relaxed);
        let new_frame1 = (old_frame1 + self.frame_count1 - 1) % self.frame_count1;
        let new_frame2 = (old_frame2 + self.frame_count2 - 1) % self.frame_count2;
        self.current_frame1.store(new_frame1, Ordering::Relaxed);
        self.current_frame2.store(new_frame2, Ordering::Relaxed);
        self.frame_changed.store(true, Ordering::Relaxed);
        true
    }

    pub fn has_frame_changed(&self) -> bool {
        self.frame_changed.swap(false, Ordering::Relaxed)
    }

    pub fn update_textures(&self) -> bool {
        let (current_left, current_right) = self.current_images();

        // Process other textures in the background
        self.process_loaded_textures();

        // Preload textures
        self.preload_textures(current_left, current_right);

        // Check if new textures are available
        let new_left_texture = self.get_texture(current_left, true);
        let new_right_texture = self.get_texture(current_right, false);

        let current_left_texture = self.left_texture.lock().clone();
        let current_right_texture = self.right_texture.lock().clone();

        let textures_updated = match (
            &new_left_texture,
            &new_right_texture,
            &current_left_texture,
            &current_right_texture,
        ) {
            (Some(nl), Some(nr), Some(cl), Some(cr)) => {
                !Arc::ptr_eq(nl, cl) || !Arc::ptr_eq(nr, cr)
            }
            _ => new_left_texture.is_some() || new_right_texture.is_some(),
        };

        textures_updated
    }

    pub fn ensure_texture_loaded(&self, index: usize, is_left: bool) {
        if !self.is_within_preload_range(index, is_left) {
            return;
        }

        let cache = if is_left {
            &self.texture_cache_left
        } else {
            &self.texture_cache_right
        };

        let cache_read = cache.read();
        if !cache_read.contains_key(&index) {
            drop(cache_read);
            let path = if is_left {
                &self.image_data1[index].0
            } else {
                &self.image_data2[index].0
            };

            // Check if the image is already being processed
            let mut processing_textures = self.processing_textures.lock();
            if !processing_textures.contains(&(index, is_left)) {
                processing_textures.insert((index, is_left));
                drop(processing_textures);

                // Add the request to the queue
                let mut queue = self.texture_load_queue.lock();
                if !queue.contains(&(index, is_left)) {
                    queue.push((path.to_string(), index, is_left));
                }
            }
        }
    }

    pub fn process_load_queue(&mut self) {
        let mut queue = self.texture_load_queue.lock();
        queue.reprioritize(
            self.current_frame1.load(Ordering::Relaxed),
            self.current_frame2.load(Ordering::Relaxed),
        );

        while let Some((path, index, is_left)) = queue.pop() {
            let sender = self.texture_load_sender.clone();
            let texture_process_sender = self.texture_process_sender.clone();
            let processing_textures = Arc::clone(&self.processing_textures);
            self.texture_load_pool.execute(move || {
                if let Ok((image_data, size)) = Self::load_image_data_from_path(&path) {
                    sender.send((path.to_string(), index, is_left)).unwrap();
                    texture_process_sender
                        .send((index, is_left, image_data, size))
                        .unwrap();
                }
                processing_textures.lock().remove(&(index, is_left));
            });
        }
    }

    pub fn process_loaded_textures(&self) {
        while let Ok((index, is_left, image_data, size)) =
            self.texture_process_receiver.lock().try_recv()
        {
            let texture = if let Some(reused_texture) = self.texture_reuse_pool.lock().pop() {
                // Reuse an existing texture if available
                reused_texture
            } else {
                // Create a new texture if none are available for reuse
                Arc::new(self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!(
                        "Image Texture - {} (Frame {})",
                        if is_left { "Left" } else { "Right" },
                        index
                    )),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                }))
            };

            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    aspect: wgpu::TextureAspect::All,
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                },
                &image_data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * size.width),
                    rows_per_image: Some(size.height),
                },
                size,
            );

            let cache = if is_left {
                &self.texture_cache_left
            } else {
                &self.texture_cache_right
            };

            // Update cache
            let texture_arc = Arc::clone(&texture);
            let mut cache_write = cache.write();
            if let Some(old_texture) =
                cache_write.insert(index, Arc::new(Mutex::new(Some(texture_arc))))
            {
                // If there was an old texture, add it to the reuse pool
                if let Some(old_texture) = old_texture.lock().take() {
                    self.texture_reuse_pool.lock().push(old_texture);
                }
            }

            // Evict old textures if the cache is full
            if cache_write.len() > self.cache_size / 2 {
                self.evict_old_textures();
            }

            // Notify that the texture is ready
            self.frame_changed.store(true, Ordering::Relaxed);
        }
    }

    fn add_texture_to_cache_background(
        index: usize,
        is_left: bool,
        texture: Arc<wgpu::Texture>,
        texture_cache: Arc<RwLock<HashMap<usize, Arc<Mutex<Option<Arc<wgpu::Texture>>>>>>>,
        texture_load_queue: Arc<Mutex<PriorityTextureLoadQueue>>,
    ) {
        debug!(
            "Adding texture to cache: index={}, is_left={}, size={:?}",
            index,
            is_left,
            texture.size()
        );

        let mut cache_write = texture_cache.write();
        cache_write.insert(index, Arc::new(Mutex::new(Some(Arc::clone(&texture)))));

        // Remove the completed request from the queue
        let mut queue = texture_load_queue.lock();
        queue
            .queue
            .lock()
            .retain(|&(_, req_index, req_is_left)| req_index != index || req_is_left != is_left);
        queue.unique_requests.lock().remove(&(index, is_left));

        debug!(
            "Texture added to cache and removed from queue. New cache state: {:?}",
            cache_write.keys().collect::<Vec<_>>()
        );
    }

    pub fn preload_textures(&self, index1: usize, index2: usize) {
        // Ensure current frames are loaded first
        self.ensure_texture_loaded(index1, true);
        self.ensure_texture_loaded(index2, false);

        let frame_count1 = self.frame_count1;
        let frame_count2 = self.frame_count2;

        // Preload ahead
        for i in 1..=self.preload_ahead {
            let preload_index1 = (index1 + i) % frame_count1;
            let preload_index2 = (index2 + i) % frame_count2;
            self.ensure_texture_loaded(preload_index1, true);
            self.ensure_texture_loaded(preload_index2, false);
        }

        // Preload behind
        for i in 1..=self.preload_behind {
            let preload_index1 = (index1 + frame_count1 - i) % frame_count1;
            let preload_index2 = (index2 + frame_count2 - i) % frame_count2;
            self.ensure_texture_loaded(preload_index1, true);
            self.ensure_texture_loaded(preload_index2, false);
        }
    }

    pub fn update_current_frame(&self, new_frame_left: usize, new_frame_right: usize) {
        let queue = self.texture_load_queue.lock();
        queue.reprioritize(new_frame_left, new_frame_right);
    }

    pub fn is_within_preload_range(&self, index: usize, is_left: bool) -> bool {
        let current_frame = if is_left {
            self.current_frame1.load(Ordering::Relaxed)
        } else {
            self.current_frame2.load(Ordering::Relaxed)
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

    pub fn update(&self, delta: std::time::Duration) -> bool {
        if self.is_playing.load(Ordering::Relaxed) {
            self.advance_frame(delta.as_micros() as u64);
            true
        } else {
            false
        }
    }

    fn advance_frame(&self, delta_micros: u64) {
        let mut current_time = self.current_time.load(Ordering::Relaxed);
        current_time += delta_micros;
        if current_time >= self.total_duration() {
            current_time %= self.total_duration();
        }
        self.current_time.store(current_time, Ordering::Relaxed);

        let old_frame1 = self.current_frame1.load(Ordering::Relaxed);
        let old_frame2 = self.current_frame2.load(Ordering::Relaxed);

        let new_frame1 = self.get_current_index(&self.image_data1, current_time as u64);
        let new_frame2 = self.get_current_index(&self.image_data2, current_time as u64);

        self.current_frame1.store(new_frame1, Ordering::Relaxed);
        self.current_frame2.store(new_frame2, Ordering::Relaxed);

        if new_frame1 != old_frame1 || new_frame2 != old_frame2 {
            self.frame_changed.store(true, Ordering::Relaxed);
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
            label: Some(&format!(
                "Image Texture - {} (Frame {})",
                if is_left { "Left" } else { "Right" },
                index
            )),
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

    pub fn get_left_texture(&self) -> Option<Arc<wgpu::Texture>> {
        self.left_texture.lock().clone()
    }

    pub fn get_right_texture(&self) -> Option<Arc<wgpu::Texture>> {
        self.right_texture.lock().clone()
    }

    fn evict_old_textures(&self) {
        let current_frame_left = self.current_frame1.load(Ordering::Relaxed);
        let current_frame_right = self.current_frame2.load(Ordering::Relaxed);

        let mut cache_left = self.texture_cache_left.write();
        let mut cache_right = self.texture_cache_right.write();

        let mut reuse_pool = self.texture_reuse_pool.lock();

        Self::evict_old_textures_from_cache(
            &mut cache_left,
            current_frame_left,
            self.frame_count1,
            &mut reuse_pool,
        );
        Self::evict_old_textures_from_cache(
            &mut cache_right,
            current_frame_right,
            self.frame_count2,
            &mut reuse_pool,
        );
    }

    fn evict_old_textures_from_cache(
        cache: &mut HashMap<usize, Arc<Mutex<Option<Arc<wgpu::Texture>>>>>,
        current_frame: usize,
        frame_count: usize,
        reuse_pool: &mut Vec<Arc<wgpu::Texture>>,
    ) {
        if cache.len() <= 1 {
            return;
        }

        let mut frames: Vec<usize> = cache.keys().cloned().collect();
        frames.sort_by_key(|&frame| {
            std::cmp::min(
                (frame as i32 - current_frame as i32).abs() as usize,
                frame_count - (frame as i32 - current_frame as i32).abs() as usize,
            )
        });

        while frames.len() > 1 {
            let frame_to_evict = frames.pop().unwrap();
            if frame_to_evict == current_frame {
                continue;
            }

            if let Some(evicted_texture) = cache.remove(&frame_to_evict) {
                if let Some(texture) = evicted_texture.lock().take() {
                    reuse_pool.push(texture);
                }
            }
        }
    }
}
