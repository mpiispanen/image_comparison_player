use log::debug;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};
use threadpool::ThreadPool;

pub struct TextureLoadRequest {
    path: String,
    index: usize,
    is_left: bool,
    load_start: Instant,
}

impl TextureLoadRequest {
    fn new(path: String, index: usize, is_left: bool) -> Self {
        Self {
            path,
            index,
            is_left,
            load_start: Instant::now(),
        }
    }
}

struct TextureProcessRequest {
    index: usize,
    is_left: bool,
    image_data: Vec<u8>,
    size: wgpu::Extent3d,
}

impl TextureProcessRequest {
    fn new(index: usize, is_left: bool, image_data: Vec<u8>, size: wgpu::Extent3d) -> Self {
        Self {
            index,
            is_left,
            image_data,
            size,
        }
    }
}

pub struct TextureTimingInfo {
    pub load_time: Duration,
    pub process_time: Duration,
}

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
        let key = (request.index, request.is_left);
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
            unique_requests.remove(&(request.index, request.is_left));
            Some(request)
        } else {
            None
        }
    }

    pub fn reprioritize(&self, current_frame_left: usize, current_frame_right: usize) {
        let mut queue = self.queue.lock();
        queue.make_contiguous().sort_by_key(|req| {
            let (current_frame, frame_count) = if req.is_left {
                (current_frame_left, self.frame_count_left)
            } else {
                (current_frame_right, self.frame_count_right)
            };
            std::cmp::min(
                usize::abs_diff(req.index, current_frame),
                frame_count - usize::abs_diff(req.index, current_frame),
            )
        });
    }

    pub fn contains(&self, key: &(usize, bool)) -> bool {
        self.unique_requests.lock().contains(key)
    }
}

type TextureCache = Arc<RwLock<HashMap<usize, Arc<Mutex<(Option<Arc<wgpu::Texture>>, Vec<u8>)>>>>>;

type TextureProcessSender = Sender<(usize, bool, Vec<u8>, wgpu::Extent3d)>;
type TextureProcessReceiver = Arc<Mutex<Receiver<(usize, bool, Vec<u8>, wgpu::Extent3d)>>>;
type TextureHolder = Arc<Mutex<Option<Arc<wgpu::Texture>>>>;

type FlipDiffCache = Arc<RwLock<HashMap<(usize, usize), Arc<Mutex<Option<Arc<wgpu::Texture>>>>>>>;

pub struct PlayerConfig {
    pub image_data1: Vec<(String, u64, u64)>,
    pub image_data2: Vec<(String, u64, u64)>,
    pub cache_size: usize,
    pub preload_ahead: usize,
    pub preload_behind: usize,
    pub num_load_threads: usize,
    pub num_process_threads: usize,
}

pub struct Player {
    pub config: PlayerConfig,
    current_time: AtomicU64,
    is_playing: AtomicBool,
    last_update: AtomicU64,
    pub frame_count1: usize,
    pub frame_count2: usize,
    queue: Arc<wgpu::Queue>,
    device: Arc<wgpu::Device>,
    current_frame1: AtomicUsize,
    current_frame2: AtomicUsize,
    pub texture_cache_left: TextureCache,
    pub texture_cache_right: TextureCache,
    texture_reuse_pool: Arc<Mutex<Vec<Arc<wgpu::Texture>>>>,
    frame_changed: Arc<AtomicBool>,
    pub texture_load_queue: Arc<Mutex<PriorityTextureLoadQueue>>,
    texture_load_pool: ThreadPool,
    texture_process_pool: ThreadPool,
    texture_process_sender: TextureProcessSender,
    texture_process_receiver: TextureProcessReceiver,
    left_texture: TextureHolder,
    right_texture: TextureHolder,
    processing_textures: Arc<Mutex<HashSet<(usize, bool)>>>,
    pub texture_timings: Arc<RwLock<HashMap<(usize, bool), TextureTimingInfo>>>,
    current_frame_set_time: Arc<Mutex<Instant>>,
    pub flip_diff_cache: FlipDiffCache,
}

impl Player {
    pub fn new(config: PlayerConfig, queue: Arc<wgpu::Queue>, device: Arc<wgpu::Device>) -> Self {
        let frame_count1 = config.image_data1.len();
        let frame_count2 = config.image_data2.len();
        let (texture_process_sender, texture_process_receiver) = channel();
        let texture_process_receiver = Arc::new(Mutex::new(texture_process_receiver));

        let texture_load_pool = ThreadPool::new(config.num_load_threads);
        let texture_process_pool = ThreadPool::new(config.num_process_threads);

        let cache_size = config.cache_size;

        Self {
            config,
            current_time: AtomicU64::new(0),
            is_playing: AtomicBool::new(false),
            last_update: AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            ),
            frame_count1,
            frame_count2,
            queue,
            device,
            current_frame1: AtomicUsize::new(0),
            current_frame2: AtomicUsize::new(0),
            texture_cache_left: Arc::new(RwLock::new(HashMap::with_capacity(cache_size / 2))),
            texture_cache_right: Arc::new(RwLock::new(HashMap::with_capacity(cache_size / 2))),
            texture_reuse_pool: Arc::new(Mutex::new(Vec::new())),
            frame_changed: Arc::new(AtomicBool::new(false)),
            texture_load_queue: Arc::new(Mutex::new(PriorityTextureLoadQueue::new(
                frame_count1,
                frame_count2,
            ))),
            texture_load_pool,
            texture_process_pool,
            texture_process_sender,
            texture_process_receiver,
            left_texture: Arc::new(Mutex::new(None)),
            right_texture: Arc::new(Mutex::new(None)),
            processing_textures: Arc::new(Mutex::new(HashSet::new())),
            texture_timings: Arc::new(RwLock::new(HashMap::new())),
            current_frame_set_time: Arc::new(Mutex::new(Instant::now())),
            flip_diff_cache: Arc::new(RwLock::new(HashMap::new())),
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
            .and_then(|texture_holder| texture_holder.lock().0.as_ref().cloned())
    }

    fn get_current_index(&self, image_data: &[(String, u64, u64)], current_time: u64) -> usize {
        image_data
            .iter()
            .position(|(_, start, end)| *start <= current_time && current_time < *end)
            .unwrap_or(image_data.len() - 1)
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

    pub fn update_textures(&self) -> bool {
        let (current_left, current_right) = self.current_images();

        // Ensure current frames are loaded
        self.ensure_texture_loaded(current_left, true);
        self.ensure_texture_loaded(current_right, false);

        // Preload textures
        self.preload_textures(current_left, current_right);

        // Process other textures in the background
        self.process_loaded_textures();

        let mut textures_updated = false;

        // Update left texture
        if let Some(new_left) = self.get_texture(current_left, true) {
            let mut left_texture = self.left_texture.lock();
            if left_texture
                .as_ref()
                .map_or(true, |t| !Arc::ptr_eq(t, &new_left))
            {
                *left_texture = Some(new_left);
                textures_updated = true;
            }
        }

        // Update right texture
        if let Some(new_right) = self.get_texture(current_right, false) {
            let mut right_texture = self.right_texture.lock();
            if right_texture
                .as_ref()
                .map_or(true, |t| !Arc::ptr_eq(t, &new_right))
            {
                *right_texture = Some(new_right);
                textures_updated = true;
            }
        }

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
                &self.config.image_data1[index].0
            } else {
                &self.config.image_data2[index].0
            };

            // Check if the image is already being processed
            let mut processing_textures = self.processing_textures.lock();
            if !processing_textures.contains(&(index, is_left)) {
                processing_textures.insert((index, is_left));
                drop(processing_textures);

                // Add the request to the queue
                let queue = self.texture_load_queue.lock();
                if !queue.contains(&(index, is_left)) {
                    queue.push(TextureLoadRequest::new(path.to_string(), index, is_left));
                }
            }
        }
    }

    pub fn process_load_queue(&mut self) {
        let queue = self.texture_load_queue.lock();
        queue.reprioritize(
            self.current_frame1.load(Ordering::Relaxed),
            self.current_frame2.load(Ordering::Relaxed),
        );

        while let Some(request) = queue.pop() {
            let texture_process_sender = self.texture_process_sender.clone();
            let processing_textures = Arc::clone(&self.processing_textures);
            let texture_timings = Arc::clone(&self.texture_timings);
            self.texture_load_pool.execute(move || {
                let request = TextureLoadRequest::new(
                    request.path.to_string(),
                    request.index,
                    request.is_left,
                );
                if let Ok((image_data, size)) = Self::load_image_data_from_path(&request.path) {
                    let load_end = Instant::now();
                    let load_time = load_end - request.load_start;

                    let process_request = TextureProcessRequest::new(
                        request.index,
                        request.is_left,
                        image_data,
                        size,
                    );
                    texture_process_sender
                        .send((
                            process_request.index,
                            process_request.is_left,
                            process_request.image_data,
                            process_request.size,
                        ))
                        .unwrap();

                    let mut texture_timings = texture_timings.write();
                    texture_timings
                        .entry((request.index, request.is_left))
                        .or_insert(TextureTimingInfo {
                            load_time,
                            process_time: Duration::default(),
                        });
                }
                processing_textures
                    .lock()
                    .remove(&(request.index, request.is_left));
            });
        }
    }

    pub fn process_loaded_textures(&self) {
        while let Ok((index, is_left, image_data, size)) =
            self.texture_process_receiver.lock().try_recv()
        {
            let device = Arc::clone(&self.device);
            let queue = Arc::clone(&self.queue);
            let texture_reuse_pool = Arc::clone(&self.texture_reuse_pool);
            let cache = if is_left {
                Arc::clone(&self.texture_cache_left)
            } else {
                Arc::clone(&self.texture_cache_right)
            };
            let cache_size = self.config.cache_size;
            let frame_changed = Arc::clone(&self.frame_changed);
            let texture_timings = Arc::clone(&self.texture_timings);

            self.texture_process_pool.execute(move || {
                let process_start = Instant::now();

                let texture = if let Some(reused_texture) = texture_reuse_pool.lock().pop() {
                    reused_texture
                } else {
                    Arc::new(device.create_texture(&wgpu::TextureDescriptor {
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
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    }))
                };

                queue.write_texture(
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

                let texture_arc = Arc::clone(&texture);
                let mut cache_write = cache.write();
                if let Some(old_entry) =
                    cache_write.insert(index, Arc::new(Mutex::new((Some(texture_arc), image_data))))
                {
                    if let Some(old_texture) = old_entry.lock().0.take() {
                        texture_reuse_pool.lock().push(old_texture);
                    }
                }

                if cache_write.len() > cache_size / 2 {
                    // Call to evict_old_textures would need to be implemented separately
                    // or the logic moved here
                }

                frame_changed.store(true, Ordering::Relaxed);

                let process_end = Instant::now();
                let process_time = process_end - process_start;

                let mut texture_timings = texture_timings.write();
                if let Some(timing) = texture_timings.get_mut(&(index, is_left)) {
                    timing.process_time = process_time;
                }
            });
        }
    }

    pub fn preload_textures(&self, index1: usize, index2: usize) {
        // Ensure current frames are loaded first
        self.ensure_texture_loaded(index1, true);
        self.ensure_texture_loaded(index2, false);

        let frame_count1 = self.frame_count1;
        let frame_count2 = self.frame_count2;

        // Preload ahead
        for i in 1..=self.config.preload_ahead {
            let preload_index1 = (index1 + i) % frame_count1;
            let preload_index2 = (index2 + i) % frame_count2;
            self.ensure_texture_loaded(preload_index1, true);
            self.ensure_texture_loaded(preload_index2, false);
        }

        // Preload behind
        for i in 1..=self.config.preload_behind {
            let preload_index1 = (index1 + frame_count1 - i) % frame_count1;
            let preload_index2 = (index2 + frame_count2 - i) % frame_count2;
            self.ensure_texture_loaded(preload_index1, true);
            self.ensure_texture_loaded(preload_index2, false);
        }
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

        min_distance <= self.config.preload_ahead || min_distance <= self.config.preload_behind
    }

    pub fn update(&self, delta: std::time::Duration) -> bool {
        if self.is_playing.load(Ordering::Relaxed) {
            let (current_left, current_right) = self.current_images();
            let now = Instant::now();
            let elapsed = now.duration_since(*self.current_frame_set_time.lock());

            if self.get_texture(current_left, true).is_some()
                && self.get_texture(current_right, false).is_some()
            {
                debug!("Frame displayed after {:?} delay", elapsed);
                self.advance_frame(delta.as_micros() as u64);
                *self.current_frame_set_time.lock() = now;
                true
            } else {
                debug!(
                    "Waiting for frame to be available. Elapsed time: {:?}",
                    elapsed
                );
                false
            }
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

        let new_frame1 = self.get_current_index(&self.config.image_data1, current_time);
        let new_frame2 = self.get_current_index(&self.config.image_data2, current_time);

        self.current_frame1.store(new_frame1, Ordering::Relaxed);
        self.current_frame2.store(new_frame2, Ordering::Relaxed);

        if new_frame1 != old_frame1 || new_frame2 != old_frame2 {
            self.frame_changed.store(true, Ordering::Relaxed);
            *self.current_frame_set_time.lock() = Instant::now();
        }
    }

    fn total_duration(&self) -> u64 {
        std::cmp::max(
            self.config
                .image_data1
                .last()
                .map(|(_, _, end)| *end)
                .unwrap_or(0),
            self.config
                .image_data2
                .last()
                .map(|(_, _, end)| *end)
                .unwrap_or(0),
        )
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
            &self.config.image_data1
        } else {
            &self.config.image_data2
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
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
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

    pub fn generate_flip_diff(
        &self,
        left_index: usize,
        right_index: usize,
        flip_diff_sender: Sender<(usize, usize, Vec<u8>, wgpu::Extent3d)>,
    ) {
        let left_texture = self.get_texture(left_index, true);
        let right_texture = self.get_texture(right_index, false);

        if let (Some(left_texture), Some(right_texture)) = (left_texture, right_texture) {
            let left_size = left_texture.size();
            let right_size = right_texture.size();

            if left_size.width != right_size.width || left_size.height != right_size.height {
                debug!("Skipping Flip diff generation due to size mismatch");
                return;
            }

            let width = left_size.width;
            let height = left_size.height;

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            let left_buffer =
                self.create_buffer_and_copy_texture(&mut encoder, &left_texture, left_size);
            let right_buffer =
                self.create_buffer_and_copy_texture(&mut encoder, &right_texture, right_size);

            self.queue.submit(std::iter::once(encoder.finish()));

            let left_data = self.read_buffer(&left_buffer, (width * height * 4) as u64);
            let right_data = self.read_buffer(&right_buffer, (width * height * 4) as u64);

            let left_image = nv_flip::FlipImageRgb8::with_data(
                width,
                height,
                &left_data
                    .chunks(4)
                    .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
                    .collect::<Vec<u8>>(),
            );
            let right_image = nv_flip::FlipImageRgb8::with_data(
                width,
                height,
                &right_data
                    .chunks(4)
                    .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
                    .collect::<Vec<u8>>(),
            );

            let error_map =
                nv_flip::flip(left_image, right_image, nv_flip::DEFAULT_PIXELS_PER_DEGREE);
            let visualized = error_map.apply_color_lut(&nv_flip::magma_lut());

            let diff_data: Vec<u8> = visualized
                .to_vec()
                .chunks_exact(3)
                .flat_map(|chunk| chunk.iter().chain(std::iter::once(&255u8)))
                .copied()
                .collect();
            let diff_size = wgpu::Extent3d {
                width: visualized.width(),
                height: visualized.height(),
                depth_or_array_layers: 1,
            };

            flip_diff_sender
                .send((left_index, right_index, diff_data, diff_size))
                .unwrap();
        }
    }

    fn create_buffer_and_copy_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        size: wgpu::Extent3d,
    ) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Texture Buffer"),
            size: (size.width * size.height * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(size.width * 4),
                    rows_per_image: Some(size.height),
                },
            },
            size,
        );

        buffer
    }

    fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let buffer_slice = buffer.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = buffer_slice.get_mapped_range().to_vec();
        buffer.unmap();
        data
    }
}
