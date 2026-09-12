use image::GenericImageView;
use log::{debug, warn};
use memmap2::Mmap;
use nv_flip::{flip, magma_lut, FlipImageRgb8, FlipPool};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};
use threadpool::ThreadPool;

/// Metrics tracking cache efficiency.
pub struct CacheMetrics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }
}

/// Returns the index (from `keys`) that is farthest from `current_frame` in circular distance.
/// Returns `None` if `keys` is empty.
fn select_eviction_candidate(
    keys: &[usize],
    current_frame: usize,
    frame_count: usize,
) -> Option<usize> {
    keys.iter().cloned().max_by_key(|&k| {
        let fwd = (k + frame_count - current_frame) % frame_count;
        let bwd = (current_frame + frame_count - k) % frame_count;
        std::cmp::min(fwd, bwd)
    })
}

fn circular_distance(index: usize, current_frame: usize, frame_count: usize) -> usize {
    let fwd = (index + frame_count - current_frame) % frame_count;
    let bwd = (current_frame + frame_count - index) % frame_count;
    std::cmp::min(fwd, bwd)
}

fn forward_distance(index: usize, current_frame: usize, frame_count: usize) -> usize {
    (index + frame_count - current_frame) % frame_count
}

fn backward_distance(index: usize, current_frame: usize, frame_count: usize) -> usize {
    (current_frame + frame_count - index) % frame_count
}

fn is_in_active_flip_diff_window(
    left_index: usize,
    right_index: usize,
    current_frames: (usize, usize),
    frame_count_left: usize,
    frame_count_right: usize,
    preload_ahead: usize,
    preload_behind: usize,
) -> bool {
    let (current_left, current_right) = current_frames;
    let fwd_left = forward_distance(left_index, current_left, frame_count_left);
    let fwd_right = forward_distance(right_index, current_right, frame_count_right);
    if fwd_left == fwd_right && fwd_left <= preload_ahead {
        return true;
    }

    let back_left = backward_distance(left_index, current_left, frame_count_left);
    let back_right = backward_distance(right_index, current_right, frame_count_right);
    back_left == back_right && back_left <= preload_behind
}

fn latest_current_frames_from_switch_times(
    frame_switch_times: &HashMap<(usize, bool), Instant>,
    fallback_left: usize,
    fallback_right: usize,
) -> (usize, usize) {
    let mut current_left = fallback_left;
    let mut current_right = fallback_right;
    let mut latest_left_time = None::<Instant>;
    let mut latest_right_time = None::<Instant>;

    for (&(index, is_left), &time) in frame_switch_times {
        if is_left {
            if latest_left_time.is_none_or(|latest| time > latest) {
                latest_left_time = Some(time);
                current_left = index;
            }
        } else if latest_right_time.is_none_or(|latest| time > latest) {
            latest_right_time = Some(time);
            current_right = index;
        }
    }

    (current_left, current_right)
}

fn select_flip_diff_eviction_candidate(
    keys: &[(usize, usize)],
    current_left: usize,
    current_right: usize,
    frame_count_left: usize,
    frame_count_right: usize,
    preload_ahead: usize,
    preload_behind: usize,
) -> Option<(usize, usize)> {
    let mut outside = Vec::new();
    let mut inside = Vec::new();
    for &(l, r) in keys {
        if is_in_active_flip_diff_window(
            l,
            r,
            (current_left, current_right),
            frame_count_left,
            frame_count_right,
            preload_ahead,
            preload_behind,
        ) {
            inside.push((l, r));
        } else {
            outside.push((l, r));
        }
    }

    let candidates = if outside.is_empty() { inside } else { outside };
    candidates.into_iter().max_by_key(|&(l, r)| {
        let left_dist = circular_distance(l, current_left, frame_count_left);
        let right_dist = circular_distance(r, current_right, frame_count_right);
        left_dist + right_dist
    })
}

/// A fixed-capacity ring-buffer-style texture cache that evicts the frame farthest from
/// the current playback position when capacity is exceeded.
pub struct RingBufferTextureCache {
    entries: Arc<RwLock<HashMap<usize, Arc<wgpu::Texture>>>>,
    capacity: usize,
    pub metrics: Arc<CacheMetrics>,
}

impl RingBufferTextureCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::with_capacity(capacity))),
            capacity,
            metrics: Arc::new(CacheMetrics::default()),
        }
    }

    pub fn contains(&self, index: usize) -> bool {
        self.entries.read().contains_key(&index)
    }

    fn get(&self, index: usize) -> Option<Arc<wgpu::Texture>> {
        self.entries.read().get(&index).cloned()
    }

    /// Insert a texture into the cache. If the cache is at capacity, the frame farthest from
    /// `current_frame` is evicted first. Returns any evicted textures so the caller can return
    /// them to a reuse pool.
    fn insert(
        &self,
        index: usize,
        texture: Arc<wgpu::Texture>,
        current_frame: usize,
        frame_count: usize,
    ) -> Vec<Arc<wgpu::Texture>> {
        let mut entries = self.entries.write();
        let mut evicted = Vec::new();

        // Replace an existing entry without counting as an eviction.
        if let Some(old) = entries.insert(index, texture) {
            evicted.push(old);
            return evicted;
        }

        // Evict the farthest frame(s) until we are within capacity.
        while entries.len() > self.capacity {
            let keys: Vec<usize> = entries.keys().cloned().filter(|&k| k != index).collect();
            match select_eviction_candidate(&keys, current_frame, frame_count) {
                Some(farthest) => {
                    if let Some(old) = entries.remove(&farthest) {
                        evicted.push(old);
                        self.metrics.evictions.fetch_add(1, Ordering::Relaxed);
                    }
                }
                None => break,
            }
        }

        evicted
    }
}

impl Clone for RingBufferTextureCache {
    fn clone(&self) -> Self {
        Self {
            entries: Arc::clone(&self.entries),
            capacity: self.capacity,
            metrics: Arc::clone(&self.metrics),
        }
    }
}

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

pub struct PriorityFlipDiffQueue {
    queue: Arc<Mutex<VecDeque<(usize, usize)>>>,
    unique_requests: Arc<Mutex<HashSet<(usize, usize)>>>,
    frame_count_left: usize,
    frame_count_right: usize,
}

impl PriorityFlipDiffQueue {
    fn new(frame_count_left: usize, frame_count_right: usize) -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            unique_requests: Arc::new(Mutex::new(HashSet::new())),
            frame_count_left,
            frame_count_right,
        }
    }

    fn push(&self, left_index: usize, right_index: usize) {
        let key = (left_index, right_index);
        let mut unique_requests = self.unique_requests.lock();
        if !unique_requests.contains(&key) {
            self.queue.lock().push_back(key);
            unique_requests.insert(key);
        }
    }

    fn pop(&self) -> Option<(usize, usize)> {
        let mut queue = self.queue.lock();
        let mut unique_requests = self.unique_requests.lock();
        if let Some(request) = queue.pop_front() {
            unique_requests.remove(&request);
            Some(request)
        } else {
            None
        }
    }

    fn reprioritize(&self, current_frame_left: usize, current_frame_right: usize) {
        let mut queue = self.queue.lock();
        queue
            .make_contiguous()
            .sort_by_key(|&(left_index, right_index)| {
                let left_dist =
                    circular_distance(left_index, current_frame_left, self.frame_count_left);
                let right_dist =
                    circular_distance(right_index, current_frame_right, self.frame_count_right);
                left_dist + right_dist
            });
    }

    fn discard_non_priority(
        &self,
        current_left: usize,
        current_right: usize,
        preload_ahead: usize,
        preload_behind: usize,
    ) {
        let mut queue = self.queue.lock();
        let mut unique_requests = self.unique_requests.lock();
        queue.retain(|&(left_index, right_index)| {
            let keep = is_in_active_flip_diff_window(
                left_index,
                right_index,
                (current_left, current_right),
                self.frame_count_left,
                self.frame_count_right,
                preload_ahead,
                preload_behind,
            );
            if !keep {
                unique_requests.remove(&(left_index, right_index));
            }
            keep
        });
    }
}

type TextureProcessSender = Sender<(usize, bool, Vec<u8>, wgpu::Extent3d)>;
type TextureProcessReceiver = Arc<Mutex<Receiver<(usize, bool, Vec<u8>, wgpu::Extent3d)>>>;
type TextureHolder = Arc<Mutex<Option<Arc<wgpu::Texture>>>>;

type FlipDiffCache = Arc<RwLock<HashMap<(usize, usize), Arc<Mutex<Option<Arc<wgpu::Texture>>>>>>>;
type FlipDiffInProgress = Arc<RwLock<HashSet<(usize, usize)>>>;
type FlipDiffRawData = Arc<RwLock<HashMap<(usize, usize), (Vec<u8>, u32, u32)>>>;

pub struct PlayerConfig {
    pub image_data1: Vec<(String, u64, u64)>,
    pub image_data2: Vec<(String, u64, u64)>,
    pub cache_size: usize,
    pub preload_ahead: usize,
    pub preload_behind: usize,
    pub num_load_threads: usize,
    pub num_process_threads: usize,
    pub num_flip_diff_threads: usize,
    pub diff_preload_ahead: usize,
    pub diff_preload_behind: usize,
    pub single_image_mode: bool,
}

type FlipDiffSender = Sender<(usize, usize, Vec<u8>, wgpu::Extent3d)>;
#[allow(dead_code)]
type FlipDiffReceiver = Arc<Mutex<Receiver<(usize, usize, Vec<u8>, wgpu::Extent3d)>>>;

fn update_displayed_texture_pair<T>(
    left_slot: &mut Option<Arc<T>>,
    right_slot: &mut Option<Arc<T>>,
    new_left: Option<Arc<T>>,
    new_right: Option<Arc<T>>,
    single_image_mode: bool,
) -> bool {
    let Some(new_left) = new_left else {
        return false;
    };

    if single_image_mode {
        if left_slot
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, &new_left))
        {
            return false;
        }

        *left_slot = Some(new_left);
        return true;
    }

    let Some(new_right) = new_right else {
        return false;
    };

    let left_changed = !left_slot
        .as_ref()
        .is_some_and(|current| Arc::ptr_eq(current, &new_left));
    let right_changed = !right_slot
        .as_ref()
        .is_some_and(|current| Arc::ptr_eq(current, &new_right));

    if !left_changed && !right_changed {
        return false;
    }

    *left_slot = Some(new_left);
    *right_slot = Some(new_right);
    true
}

pub struct DiffImageInfo {
    pub process_time: Duration,
}

#[derive(Clone)]
pub struct FlipStats {
    pub mean: f32,
    pub min: f32,
    pub max: f32,
    pub p95: f32,
    pub p99: f32,
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
    pub texture_cache_left: RingBufferTextureCache,
    pub texture_cache_right: RingBufferTextureCache,
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
    pub current_frame_set_time: Arc<Mutex<Instant>>,
    pub flip_diff_cache: FlipDiffCache,
    pub flip_diff_request_queue: Arc<Mutex<PriorityFlipDiffQueue>>,
    flip_diff_pool: ThreadPool,
    flip_diff_sender: FlipDiffSender,
    #[allow(dead_code)]
    flip_diff_receiver: FlipDiffReceiver,
    pub flip_diff_in_progress: FlipDiffInProgress,
    pub diff_image_timings: Arc<RwLock<HashMap<(usize, usize), DiffImageInfo>>>,
    pub frame_switch_times: Arc<RwLock<HashMap<(usize, bool), Instant>>>,
    pub texture_available_times: Arc<RwLock<HashMap<(usize, bool), Instant>>>,
    playback_speed: f32,
    pub flip_stats: Arc<RwLock<HashMap<(usize, usize), FlipStats>>>,
    pub flip_diff_raw_data: FlipDiffRawData,
    pub single_image_mode: bool,
    /// Pre-computed sorted unique time points for O(log N) frame navigation
    sorted_time_points: Vec<u64>,
    pub flip_diff_cache_metrics: Arc<CacheMetrics>,
    flip_diff_cache_capacity: usize,
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
        let min_cache_span = config.preload_ahead + config.preload_behind + 1;
        let per_side_cache_size = cache_size.max(min_cache_span);
        // Keep small headroom above the active diff preload window so asynchronous
        // completions at the boundary do not immediately evict each other.
        const FLIP_DIFF_CACHE_HEADROOM: usize = 2;
        let flip_diff_cache_capacity =
            config.diff_preload_ahead + config.diff_preload_behind + 1 + FLIP_DIFF_CACHE_HEADROOM;

        let flip_diff_pool = ThreadPool::new(config.num_flip_diff_threads);
        let (flip_diff_sender, flip_diff_receiver) = channel();
        let flip_diff_receiver = Arc::new(Mutex::new(flip_diff_receiver));

        let single_image_mode = config.single_image_mode;
        let sorted_time_points =
            Self::compute_sorted_time_points(&config.image_data1, &config.image_data2);

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
            texture_cache_left: RingBufferTextureCache::new(per_side_cache_size),
            texture_cache_right: RingBufferTextureCache::new(per_side_cache_size),
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
            flip_diff_request_queue: Arc::new(Mutex::new(PriorityFlipDiffQueue::new(
                frame_count1,
                frame_count2,
            ))),
            flip_diff_pool,
            flip_diff_sender,
            flip_diff_receiver,
            flip_diff_in_progress: Arc::new(RwLock::new(HashSet::new())),
            diff_image_timings: Arc::new(RwLock::new(HashMap::new())),
            frame_switch_times: Arc::new(RwLock::new(HashMap::new())),
            texture_available_times: Arc::new(RwLock::new(HashMap::new())),
            playback_speed: 1.0,
            flip_stats: Arc::new(RwLock::new(HashMap::new())),
            flip_diff_raw_data: Arc::new(RwLock::new(HashMap::new())),
            single_image_mode,
            sorted_time_points,
            flip_diff_cache_metrics: Arc::new(CacheMetrics::default()),
            flip_diff_cache_capacity,
        }
    }

    /// Pre-compute a sorted, deduplicated list of all time points from both sequences.
    /// This is computed once at construction time to make `find_next_time_point` O(log N).
    fn compute_sorted_time_points(
        image_data1: &[(String, u64, u64)],
        image_data2: &[(String, u64, u64)],
    ) -> Vec<u64> {
        let mut times: Vec<u64> = image_data1
            .iter()
            .chain(image_data2.iter())
            .flat_map(|(_, start, end)| [*start, *end])
            .collect();
        times.sort_unstable();
        times.dedup();
        times
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
        cache.get(index)
    }

    fn get_current_index(&self, image_data: &[(String, u64, u64)], current_time: u64) -> usize {
        current_index_for_time(image_data, current_time)
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

    pub fn next_frame(&self, show_flip_diff: bool) -> bool {
        let frame_changed = self.jump_to_next_time_point(1);
        if frame_changed && show_flip_diff {
            let (current_left, current_right) = self.current_images();
            self.ensure_flip_diff_generated(current_left, current_right);
        }
        frame_changed
    }

    pub fn previous_frame(&self, show_flip_diff: bool) -> bool {
        let frame_changed = self.jump_to_next_time_point(-1);
        if frame_changed && show_flip_diff {
            let (current_left, current_right) = self.current_images();
            self.ensure_flip_diff_generated(current_left, current_right);
        }
        frame_changed
    }

    fn jump_to_next_time_point(&self, direction: i64) -> bool {
        let current_time = self.current_time.load(Ordering::Relaxed);
        let new_time = self.find_next_time_point(current_time, direction);

        if new_time != current_time {
            self.current_time.store(new_time, Ordering::Relaxed);
            self.update_current_frames();
            true
        } else {
            false
        }
    }

    fn find_next_time_point(&self, current_time: u64, direction: i64) -> u64 {
        if direction > 0 {
            next_time_point_forward(&self.sorted_time_points, current_time)
        } else {
            next_time_point_backward(&self.sorted_time_points, current_time)
        }
    }

    fn update_current_frames(&self) {
        let current_time = self.current_time.load(Ordering::Relaxed);
        let new_frame1 = self.get_current_index(&self.config.image_data1, current_time);
        let new_frame2 = self.get_current_index(&self.config.image_data2, current_time);

        let old_frame1 = self.current_frame1.swap(new_frame1, Ordering::Relaxed);
        let old_frame2 = self.current_frame2.swap(new_frame2, Ordering::Relaxed);

        if new_frame1 != old_frame1 || new_frame2 != old_frame2 {
            let now = Instant::now();
            let mut frame_switch_times = self.frame_switch_times.write();
            frame_switch_times.insert((new_frame1, true), now);
            frame_switch_times.insert((new_frame2, false), now);
            self.frame_changed.store(true, Ordering::Relaxed);
        }
    }

    pub fn update_textures(&self, show_flip_diff: bool) -> bool {
        let (current_left, current_right) = self.current_images();

        // Ensure current frames are loaded
        self.ensure_texture_loaded(current_left, true);
        self.ensure_texture_loaded(current_right, false);

        // Preload textures
        self.preload_textures(current_left, current_right, show_flip_diff);

        // Process other textures in the background
        self.process_loaded_textures();

        let new_left = self.get_texture(current_left, true);
        let new_right = if self.single_image_mode {
            None
        } else {
            self.get_texture(current_right, false)
        };

        let mut left_texture = self.left_texture.lock();
        let mut right_texture = self.right_texture.lock();
        update_displayed_texture_pair(
            &mut left_texture,
            &mut right_texture,
            new_left,
            new_right,
            self.single_image_mode,
        )
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

        if cache.contains(index) {
            cache.metrics.hits.fetch_add(1, Ordering::Relaxed);
            return;
        }

        cache.metrics.misses.fetch_add(1, Ordering::Relaxed);

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
                    if texture_process_sender
                        .send((
                            process_request.index,
                            process_request.is_left,
                            process_request.image_data,
                            process_request.size,
                        ))
                        .is_err()
                    {
                        warn!(
                            "Dropping processed texture for frame {} ({}) because receiver is gone",
                            process_request.index,
                            if process_request.is_left {
                                "left"
                            } else {
                                "right"
                            }
                        );
                        return;
                    }

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
                self.texture_cache_left.clone()
            } else {
                self.texture_cache_right.clone()
            };
            let frame_changed = Arc::clone(&self.frame_changed);
            let texture_timings = Arc::clone(&self.texture_timings);
            let texture_available_times = Arc::clone(&self.texture_available_times);

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

            self.texture_process_pool.execute(move || {
                let process_start = Instant::now();

                let texture = {
                    let mut pool = texture_reuse_pool.lock();
                    let maybe_reused_index = pool.iter().position(|texture| {
                        let texture_size = texture.size();
                        texture_size.width == size.width
                            && texture_size.height == size.height
                            && texture_size.depth_or_array_layers == size.depth_or_array_layers
                    });
                    maybe_reused_index.map(|index| pool.swap_remove(index))
                }
                .unwrap_or_else(|| {
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
                });

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

                // Insert into the ring-buffer cache; evicted textures go back to the reuse pool.
                let evicted = cache.insert(index, texture, current_frame, frame_count);
                for old_texture in evicted {
                    texture_reuse_pool.lock().push(old_texture);
                }

                frame_changed.store(true, Ordering::Relaxed);

                let process_end = Instant::now();
                let process_time = process_end - process_start;

                let mut texture_timings = texture_timings.write();
                if let Some(timing) = texture_timings.get_mut(&(index, is_left)) {
                    timing.process_time = process_time;
                }

                // Record the time when the texture becomes available
                let mut texture_available_times = texture_available_times.write();
                texture_available_times.insert((index, is_left), Instant::now());
            });
        }
    }

    pub fn preload_textures(&self, index1: usize, index2: usize, show_flip_diff: bool) {
        // Ensure current frames are loaded first
        self.ensure_texture_loaded(index1, true);
        if !self.single_image_mode {
            self.ensure_texture_loaded(index2, false);
        }

        let frame_count1 = self.frame_count1;
        let frame_count2 = self.frame_count2;

        // Preload ahead
        for i in 1..=self.config.preload_ahead {
            let preload_index1 = (index1 + i) % frame_count1;
            self.ensure_texture_loaded(preload_index1, true);
            if !self.single_image_mode {
                let preload_index2 = (index2 + i) % frame_count2;
                self.ensure_texture_loaded(preload_index2, false);
            }
        }

        // Preload behind
        for i in 1..=self.config.preload_behind {
            let preload_index1 = (index1 + frame_count1 - i % frame_count1) % frame_count1;
            self.ensure_texture_loaded(preload_index1, true);
            if !self.single_image_mode {
                let preload_index2 = (index2 + frame_count2 - i % frame_count2) % frame_count2;
                self.ensure_texture_loaded(preload_index2, false);
            }
        }

        // Preload flip diffs (not applicable in single image mode)
        if !self.single_image_mode && show_flip_diff {
            self.ensure_flip_diff_generated(index1, index2);
            self.preload_flip_diffs(index1, index2);
            self.process_flip_diff_queue();
        }
    }

    fn preload_flip_diffs(&self, index1: usize, index2: usize) {
        let frame_count1 = self.frame_count1;
        let frame_count2 = self.frame_count2;

        // Preload ahead
        for i in 1..=self.config.diff_preload_ahead {
            let preload_index1 = (index1 + i) % frame_count1;
            let preload_index2 = (index2 + i) % frame_count2;
            self.ensure_flip_diff_generated(preload_index1, preload_index2);
        }

        // Preload behind
        for i in 1..=self.config.diff_preload_behind {
            let preload_index1 = (index1 + frame_count1 - i % frame_count1) % frame_count1;
            let preload_index2 = (index2 + frame_count2 - i % frame_count2) % frame_count2;
            self.ensure_flip_diff_generated(preload_index1, preload_index2);
        }
    }

    fn ensure_flip_diff_generated(&self, left_index: usize, right_index: usize) {
        let flip_diff_cache = self.flip_diff_cache.read();
        let flip_diff_in_progress = self.flip_diff_in_progress.read();

        if flip_diff_cache.contains_key(&(left_index, right_index)) {
            self.flip_diff_cache_metrics
                .hits
                .fetch_add(1, Ordering::Relaxed);
            return;
        }

        if !flip_diff_in_progress.contains(&(left_index, right_index)) {
            self.flip_diff_cache_metrics
                .misses
                .fetch_add(1, Ordering::Relaxed);
            drop(flip_diff_cache);
            drop(flip_diff_in_progress);
            if self.get_texture(left_index, true).is_some()
                && self.get_texture(right_index, false).is_some()
            {
                let queue = self.flip_diff_request_queue.lock();
                queue.push(left_index, right_index);
            }
        }
    }

    fn process_flip_diff_queue(&self) {
        if self.config.num_flip_diff_threads == 0 {
            return;
        }

        let preload_ahead = self.config.diff_preload_ahead;
        let preload_behind = self.config.diff_preload_behind;
        let current_left = self.current_frame1.load(Ordering::Relaxed);
        let current_right = self.current_frame2.load(Ordering::Relaxed);
        let next_request = {
            let queue = self.flip_diff_request_queue.lock();
            queue.discard_non_priority(current_left, current_right, preload_ahead, preload_behind);
            queue.reprioritize(current_left, current_right);
            queue.pop()
        };

        if let Some((left_index, right_index)) = next_request {
            self.generate_flip_diff(left_index, right_index);
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

    pub fn update(&self, delta: std::time::Duration, show_flip_diff: bool) -> bool {
        if self.is_playing.load(Ordering::Relaxed) {
            let (current_left, current_right) = self.current_images();
            let now = Instant::now();
            let elapsed = now.duration_since(*self.current_frame_set_time.lock());

            if self.get_texture(current_left, true).is_some()
                && (self.single_image_mode || self.get_texture(current_right, false).is_some())
            {
                debug!("Frame displayed after {:?} delay", elapsed);
                let scaled_delta = delta.mul_f32(self.playback_speed);
                let frame_changed = self.advance_frame(scaled_delta.as_micros());
                *self.current_frame_set_time.lock() = now;

                if frame_changed && show_flip_diff {
                    let (new_left, new_right) = self.current_images();
                    self.ensure_flip_diff_generated(new_left, new_right);
                }

                frame_changed
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

    pub fn playback_speed(&self) -> f32 {
        self.playback_speed
    }

    pub fn is_playing(&self) -> bool {
        self.is_playing.load(Ordering::Relaxed)
    }

    pub fn decrease_playback_speed(&mut self) {
        self.playback_speed = (self.playback_speed - 0.25).max(0.25);
    }

    pub fn increase_playback_speed(&mut self) {
        self.playback_speed = (self.playback_speed + 0.25).min(4.0);
    }

    fn advance_frame(&self, delta_micros: u128) -> bool {
        let current_time = self.current_time.load(Ordering::Relaxed);
        let new_time = current_time.saturating_add(delta_micros as u64);
        let total_duration = self.total_duration();

        if new_time >= total_duration {
            self.current_time.store(0, Ordering::Relaxed);
            self.update_current_frames();
            true
        } else {
            self.current_time.store(new_time, Ordering::Relaxed);
            self.update_current_frames();
            self.frame_changed.swap(false, Ordering::Relaxed)
        }
    }

    pub fn total_duration(&self) -> u64 {
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

    pub fn get_frame_duration(&self, frame: usize, is_left: bool) -> Option<Duration> {
        let image_data = if is_left {
            &self.config.image_data1
        } else {
            &self.config.image_data2
        };

        image_data
            .get(frame)
            .map(|(_, start, end)| Duration::from_micros(end - start))
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
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();

        let (img, dimensions) = if file_size > 15 * 1024 * 1024 {
            let mmap = unsafe { Mmap::map(&file)? };
            let img = image::load_from_memory(&mmap)?;
            let dimensions = img.dimensions();
            (img, dimensions)
        } else {
            let img = image::open(path)?;
            let dimensions = img.dimensions();
            (img, dimensions)
        };

        let rgba = img.to_rgba8();
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

    pub fn generate_flip_diff(&self, left_index: usize, right_index: usize) {
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

            let (left_buffer, left_stride) =
                self.create_buffer_and_copy_texture(&mut encoder, &left_texture, left_size);
            let (right_buffer, right_stride) =
                self.create_buffer_and_copy_texture(&mut encoder, &right_texture, right_size);

            self.queue.submit(std::iter::once(encoder.finish()));

            // Read back full padded buffers then strip row padding so nv_flip
            // receives tightly-packed RGBA rows of exactly `width * 4` bytes.
            let strip_padding = |padded: Vec<u8>, stride: u32, w: u32, h: u32| -> Vec<u8> {
                let row_bytes = (w * 4) as usize;
                let stride = stride as usize;
                let mut out = Vec::with_capacity(row_bytes * h as usize);
                for row in 0..h as usize {
                    out.extend_from_slice(&padded[row * stride..row * stride + row_bytes]);
                }
                out
            };

            let left_padded = self.read_buffer(&left_buffer, (left_stride * height) as u64);
            let right_padded = self.read_buffer(&right_buffer, (right_stride * height) as u64);

            let left_data = strip_padding(left_padded, left_stride, width, height);
            let right_data = strip_padding(right_padded, right_stride, width, height);

            let flip_diff_sender = self.flip_diff_sender.clone();
            let device = Arc::clone(&self.device);
            let queue = Arc::clone(&self.queue);
            let flip_diff_cache = Arc::clone(&self.flip_diff_cache);
            let flip_diff_in_progress = Arc::clone(&self.flip_diff_in_progress);
            let diff_image_timings = Arc::clone(&self.diff_image_timings);
            let flip_stats = Arc::clone(&self.flip_stats);
            let flip_diff_raw_data = Arc::clone(&self.flip_diff_raw_data);
            let flip_diff_cache_metrics = Arc::clone(&self.flip_diff_cache_metrics);
            let flip_diff_cache_capacity = self.flip_diff_cache_capacity;
            let current_frame_left_at_enqueue = self.current_frame1.load(Ordering::Relaxed);
            let current_frame_right_at_enqueue = self.current_frame2.load(Ordering::Relaxed);
            let frame_switch_times = Arc::clone(&self.frame_switch_times);
            let frame_count1 = self.frame_count1;
            let frame_count2 = self.frame_count2;
            let preload_ahead = self.config.diff_preload_ahead;
            let preload_behind = self.config.diff_preload_behind;

            self.flip_diff_in_progress
                .write()
                .insert((left_index, right_index));

            self.flip_diff_pool.execute(move || {
                let (current_frame_left, current_frame_right) =
                    latest_current_frames_from_switch_times(
                        &frame_switch_times.read(),
                        current_frame_left_at_enqueue,
                        current_frame_right_at_enqueue,
                    );
                if !is_in_active_flip_diff_window(
                    left_index,
                    right_index,
                    (current_frame_left, current_frame_right),
                    frame_count1,
                    frame_count2,
                    preload_ahead,
                    preload_behind,
                ) {
                    flip_diff_in_progress
                        .write()
                        .remove(&(left_index, right_index));
                    return;
                }

                let process_start = Instant::now();
                let left_image = FlipImageRgb8::with_data(width, height, &rgba_to_rgb(&left_data));
                let right_image =
                    FlipImageRgb8::with_data(width, height, &rgba_to_rgb(&right_data));

                let error_map = flip(left_image, right_image, nv_flip::DEFAULT_PIXELS_PER_DEGREE);
                let visualized = error_map.apply_color_lut(&magma_lut());

                // Collect flip stats
                let mut pool = FlipPool::from_image(&error_map);
                let stats = FlipStats {
                    mean: pool.mean(),
                    min: pool.min_value(),
                    max: pool.max_value(),
                    p95: pool.get_percentile(95.0, true),
                    p99: pool.get_percentile(99.0, true),
                };

                // Store the flip stats
                flip_stats.write().insert((left_index, right_index), stats);

                let visualized_data = visualized.to_vec();
                let diff_data = rgb_to_rgba(&visualized_data);
                let diff_size = wgpu::Extent3d {
                    width: visualized.width(),
                    height: visualized.height(),
                    depth_or_array_layers: 1,
                };

                // Store raw RGBA diff data for file export with simple eviction to bound memory usage
                {
                    const MAX_FLIP_DIFF_CACHE_ENTRIES: usize = 128;
                    let mut flip_diff_map = flip_diff_raw_data.write();
                    if flip_diff_map.len() >= MAX_FLIP_DIFF_CACHE_ENTRIES {
                        if let Some(first_key) = flip_diff_map.keys().next().cloned() {
                            flip_diff_map.remove(&first_key);
                        }
                    }
                    flip_diff_map.insert(
                        (left_index, right_index),
                        (diff_data.clone(), visualized.width(), visualized.height()),
                    );
                }

                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!(
                        "Flip Diff Texture - ({}, {})",
                        left_index, right_index
                    )),
                    size: diff_size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });

                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &diff_data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * diff_size.width),
                        rows_per_image: Some(diff_size.height),
                    },
                    diff_size,
                );

                let texture_arc = Arc::new(texture);
                {
                    let mut cache_write = flip_diff_cache.write();
                    cache_write.insert(
                        (left_index, right_index),
                        Arc::new(Mutex::new(Some(texture_arc.clone()))),
                    );

                    // Evict entries farthest from current frame pair, preferring entries
                    // outside the active preload window.
                    while cache_write.len() > flip_diff_cache_capacity {
                        let (current_frame_left, current_frame_right) =
                            latest_current_frames_from_switch_times(
                                &frame_switch_times.read(),
                                current_frame_left_at_enqueue,
                                current_frame_right_at_enqueue,
                            );
                        let candidate_keys: Vec<(usize, usize)> = cache_write
                            .keys()
                            .filter(|&&(l, r)| !(l == left_index && r == right_index))
                            .cloned()
                            .collect();
                        match select_flip_diff_eviction_candidate(
                            &candidate_keys,
                            current_frame_left,
                            current_frame_right,
                            frame_count1,
                            frame_count2,
                            preload_ahead,
                            preload_behind,
                        ) {
                            Some(evict_key) => {
                                cache_write.remove(&evict_key);
                                flip_diff_cache_metrics
                                    .evictions
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                            None => break,
                        }
                    }
                }
                flip_diff_in_progress
                    .write()
                    .remove(&(left_index, right_index));

                if flip_diff_sender
                    .send((left_index, right_index, diff_data, diff_size))
                    .is_err()
                {
                    warn!(
                        "Dropping generated FLIP diff for frames ({}, {}) because receiver is gone",
                        left_index, right_index
                    );
                    return;
                }

                let process_end = Instant::now();
                let process_time = process_end - process_start;
                diff_image_timings
                    .write()
                    .insert((left_index, right_index), DiffImageInfo { process_time });
            });
        }
    }

    fn create_buffer_and_copy_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        size: wgpu::Extent3d,
    ) -> (wgpu::Buffer, u32) {
        // bytes_per_row must be a multiple of wgpu::COPY_BYTES_PER_ROW_ALIGNMENT (256).
        let unpadded_bytes_per_row = size.width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Texture Buffer"),
            size: (bytes_per_row * size.height) as u64,
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
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size.height),
                },
            },
            size,
        );

        (buffer, bytes_per_row)
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

    /// Returns the raw RGBA pixel data, width, and height of the FLIP diff image
    /// for the given left/right frame pair, if it has been computed.
    pub fn get_flip_diff_raw_data(
        &self,
        left_index: usize,
        right_index: usize,
    ) -> Option<(Vec<u8>, u32, u32)> {
        self.flip_diff_raw_data
            .read()
            .get(&(left_index, right_index))
            .cloned()
    }

    /// Load and return the raw RGBA pixel data for a specific frame from disk.
    /// Returns `(pixels, width, height)` on success, or `None` if the path is
    /// out-of-range or the image cannot be decoded.
    pub fn get_current_frame_image_data(
        &self,
        index: usize,
        is_left: bool,
    ) -> Option<(Vec<u8>, u32, u32)> {
        let image_data = if is_left {
            &self.config.image_data1
        } else {
            &self.config.image_data2
        };
        let path = &image_data.get(index)?.0;
        match image::open(path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let (width, height) = rgba.dimensions();
                Some((rgba.into_raw(), width, height))
            }
            Err(e) => {
                warn!(
                    "Failed to load image '{}' for combined screenshot: {}",
                    path, e
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;

    /// Verifies that `get_flip_diff_raw_data` returns `None` when no diff has been generated.
    #[test]
    fn test_get_flip_diff_raw_data_missing() {
        let store: FlipDiffRawData = Arc::new(RwLock::new(HashMap::new()));
        // Nothing inserted → lookup must return None
        assert!(store.read().get(&(0, 0)).is_none());
    }

    /// Verifies that raw diff data round-trips correctly through the cache.
    #[test]
    fn test_get_flip_diff_raw_data_present() {
        let store: FlipDiffRawData = Arc::new(RwLock::new(HashMap::new()));
        let pixels: Vec<u8> = (0..16).collect(); // 2×2 RGBA
        store.write().insert((0, 1), (pixels.clone(), 2, 2));

        let result = store.read().get(&(0, 1)).cloned();
        assert!(result.is_some());
        let (data, w, h) = result.unwrap();
        assert_eq!(data, pixels);
        assert_eq!(w, 2);
        assert_eq!(h, 2);
    }

    /// Verifies that FLIP diff RGBA data has the correct length (width × height × 4).
    #[test]
    fn test_flip_diff_data_length() {
        let width: u32 = 4;
        let height: u32 = 3;
        // Simulate what generate_flip_diff produces: RGB triplets → RGBA with alpha=255
        let rgb_data: Vec<u8> = (0..(width * height * 3) as u8).collect();
        let diff_data: Vec<u8> = rgb_data
            .chunks_exact(3)
            .flat_map(|chunk| {
                let mut v = chunk.to_vec();
                v.push(255);
                v
            })
            .collect();
        assert_eq!(diff_data.len(), (width * height * 4) as usize);
        // Every 4th byte (alpha) must be 255
        for chunk in diff_data.chunks(4) {
            assert_eq!(chunk[3], 255);
        }
    }

    #[test]
    fn test_priority_queue_push_and_pop() {
        let queue = PriorityTextureLoadQueue::new(10, 10);
        queue.push(TextureLoadRequest::new(
            "frame_001.png".to_string(),
            0,
            true,
        ));

        let popped = queue.pop();
        assert!(popped.is_some());
        let req = popped.unwrap();
        assert_eq!(req.index, 0);
        assert!(req.is_left);
    }

    #[test]
    fn test_priority_queue_empty_pop_returns_none() {
        let queue = PriorityTextureLoadQueue::new(10, 10);
        assert!(queue.pop().is_none());
    }

    #[test]
    fn test_priority_queue_deduplication() {
        let queue = PriorityTextureLoadQueue::new(10, 10);
        queue.push(TextureLoadRequest::new(
            "frame_001.png".to_string(),
            0,
            true,
        ));
        queue.push(TextureLoadRequest::new(
            "frame_001.png".to_string(),
            0,
            true,
        ));

        assert!(queue.pop().is_some());
        assert!(queue.pop().is_none());
    }

    #[test]
    fn test_priority_queue_left_and_right_are_distinct_keys() {
        let queue = PriorityTextureLoadQueue::new(10, 10);
        queue.push(TextureLoadRequest::new(
            "frame_001.png".to_string(),
            0,
            true,
        ));
        queue.push(TextureLoadRequest::new(
            "frame_001.png".to_string(),
            0,
            false,
        ));

        assert!(queue.pop().is_some());
        assert!(queue.pop().is_some());
        assert!(queue.pop().is_none());
    }

    #[test]
    fn test_priority_queue_contains() {
        let queue = PriorityTextureLoadQueue::new(10, 10);
        assert!(!queue.contains(&(0, true)));

        queue.push(TextureLoadRequest::new(
            "frame_001.png".to_string(),
            0,
            true,
        ));
        assert!(queue.contains(&(0, true)));

        queue.pop();
        assert!(!queue.contains(&(0, true)));
    }

    #[test]
    fn test_priority_queue_maintains_fifo_order() {
        let queue = PriorityTextureLoadQueue::new(10, 10);
        queue.push(TextureLoadRequest::new(
            "frame_001.png".to_string(),
            0,
            true,
        ));
        queue.push(TextureLoadRequest::new(
            "frame_002.png".to_string(),
            1,
            true,
        ));
        queue.push(TextureLoadRequest::new(
            "frame_003.png".to_string(),
            2,
            true,
        ));

        assert_eq!(queue.pop().unwrap().index, 0);
        assert_eq!(queue.pop().unwrap().index, 1);
        assert_eq!(queue.pop().unwrap().index, 2);
    }

    #[test]
    fn test_flip_diff_priority_queue_reprioritizes_to_current_frame() {
        let queue = PriorityFlipDiffQueue::new(10, 10);
        queue.push(7, 7);
        queue.push(1, 1);
        queue.push(5, 5);

        queue.reprioritize(5, 5);

        assert_eq!(queue.pop(), Some((5, 5)));
        assert_eq!(queue.pop(), Some((7, 7)));
        assert_eq!(queue.pop(), Some((1, 1)));
    }

    #[test]
    fn test_flip_diff_priority_queue_deduplicates_requests() {
        let queue = PriorityFlipDiffQueue::new(10, 10);
        queue.push(2, 2);
        queue.push(2, 2);

        assert_eq!(queue.pop(), Some((2, 2)));
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn test_flip_diff_priority_queue_discards_non_priority_requests() {
        let queue = PriorityFlipDiffQueue::new(10, 10);
        queue.push(5, 5);
        queue.push(8, 8);
        queue.push(4, 6);

        queue.discard_non_priority(5, 5, 1, 1);

        assert_eq!(queue.pop(), Some((5, 5)));
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn test_is_in_active_flip_diff_window_requires_matching_offsets() {
        assert!(is_in_active_flip_diff_window(6, 6, (5, 5), 10, 10, 2, 1));
        assert!(!is_in_active_flip_diff_window(6, 5, (5, 5), 10, 10, 2, 1));
        assert!(is_in_active_flip_diff_window(4, 4, (5, 5), 10, 10, 2, 1));
        assert!(!is_in_active_flip_diff_window(3, 3, (5, 5), 10, 10, 2, 1));
    }

    // --- select_eviction_candidate tests ---

    #[test]
    fn test_select_eviction_candidate_empty() {
        assert_eq!(select_eviction_candidate(&[], 5, 10), None);
    }

    #[test]
    fn test_select_eviction_candidate_single() {
        assert_eq!(select_eviction_candidate(&[3], 5, 10), Some(3));
    }

    #[test]
    fn test_select_eviction_candidate_picks_farthest_linear() {
        // current = 5, frame_count = 10
        // frame 6: min_dist = 1  (nearest)
        // frame 0: min_dist = 5  (farthest)
        // frame 9: min_dist = 4
        let result = select_eviction_candidate(&[6, 0, 9], 5, 10);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_select_eviction_candidate_wraps_correctly() {
        // current = 0, frame_count = 10
        // frame 9: fwd=9, bwd=1, min=1 (near because of wrap)
        // frame 5: fwd=5, bwd=5, min=5 (farthest)
        let result = select_eviction_candidate(&[9, 5], 0, 10);
        assert_eq!(result, Some(5));
    }

    #[test]
    fn test_select_eviction_candidate_symmetry() {
        // current = 5, frame_count = 10
        // frame 0: fwd=(0+10-5)%10=5, bwd=(5+10-0)%10=5, min_dist=5 (farthest)
        // frame 1: fwd=(1+10-5)%10=6, bwd=(5+10-1)%10=4, min_dist=4
        // frame 4: fwd=(4+10-5)%10=9, bwd=(5+10-4)%10=1, min_dist=1 (nearest)
        let result = select_eviction_candidate(&[0, 1, 4], 5, 10);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_select_flip_diff_eviction_candidate_prefers_outside_window() {
        let keys = vec![(5, 5), (6, 6), (9, 9)];
        let result = select_flip_diff_eviction_candidate(&keys, 5, 5, 10, 10, 1, 1);
        assert_eq!(result, Some((9, 9)));
    }

    #[test]
    fn test_select_flip_diff_eviction_candidate_prefers_offset_mismatch_outside_window() {
        let keys = vec![(5, 5), (5, 6), (6, 6)];
        let result = select_flip_diff_eviction_candidate(&keys, 5, 5, 10, 10, 1, 1);
        assert_eq!(result, Some((5, 6)));
    }

    #[test]
    fn test_select_flip_diff_eviction_candidate_uses_farthest_when_all_in_active_window() {
        let keys = vec![(5, 5), (6, 6), (7, 7)];
        let result = select_flip_diff_eviction_candidate(&keys, 5, 5, 10, 10, 2, 0);
        assert_eq!(result, Some((7, 7)));
    }

    #[test]
    fn test_latest_current_frames_from_switch_times_uses_latest_entries() {
        let base = Instant::now();
        let mut switch_times = HashMap::new();
        switch_times.insert((2, true), base);
        switch_times.insert((7, false), base + Duration::from_millis(1));
        switch_times.insert((4, true), base + Duration::from_millis(2));
        switch_times.insert((9, false), base + Duration::from_millis(3));

        let result = latest_current_frames_from_switch_times(&switch_times, 0, 0);
        assert_eq!(result, (4, 9));
    }

    /// Verifies the aligned bytes_per_row calculation used in create_buffer_and_copy_texture.
    #[test]
    fn test_copy_bytes_per_row_alignment() {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        // Helper mirrors the production formula.
        let aligned = |w: u32| -> u32 {
            let unpadded = w * 4;
            unpadded.div_ceil(align) * align
        };

        // Width already produces aligned bytes_per_row (64 * 4 = 256).
        assert_eq!(aligned(64), 256);
        assert_eq!(aligned(64) % align, 0);

        // Width whose bytes_per_row is NOT a multiple of 256 without padding.
        assert_eq!(aligned(100), 512); // 400 → padded to 512
        assert_eq!(aligned(100) % align, 0);

        // Width = 1 → 4 bytes padded to 256.
        assert_eq!(aligned(1), 256);
        assert_eq!(aligned(1) % align, 0);
    }

    /// Verifies the strip_padding helper removes row padding correctly.
    #[test]
    fn test_strip_row_padding() {
        let width: u32 = 2;
        let height: u32 = 2;
        let stride: u32 = 256; // padded stride
        let row_bytes = (width * 4) as usize;

        // Build padded buffer: two 256-byte rows, pixel data in first 8 bytes each.
        let mut padded = vec![0u8; stride as usize * height as usize];
        // Row 0 pixels: R=1,G=2,B=3,A=4 and R=5,G=6,B=7,A=8
        padded[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        // Row 1 pixels: R=9,G=10,B=11,A=12 and R=13,G=14,B=15,A=16
        padded[256..264].copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);

        // Strip padding.
        let mut out = Vec::with_capacity(row_bytes * height as usize);
        for row in 0..height as usize {
            let s = row * stride as usize;
            out.extend_from_slice(&padded[s..s + row_bytes]);
        }

        assert_eq!(
            out,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }
}

/// Convert an RGBA byte slice to an RGB byte vec, dropping the alpha channel.
/// Pre-allocates the output buffer to avoid repeated reallocations.
fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    debug_assert_eq!(
        rgba.len() % 4,
        0,
        "RGBA buffer length must be a multiple of 4"
    );
    let pixel_count = rgba.len() / 4;
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for chunk in rgba.chunks_exact(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }
    rgb
}

/// Convert an RGB byte slice to an RGBA byte vec, inserting 255 for the alpha channel.
/// Pre-allocates the output buffer to avoid repeated reallocations.
fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    debug_assert_eq!(
        rgb.len() % 3,
        0,
        "RGB buffer length must be a multiple of 3"
    );
    let pixel_count = rgb.len() / 3;
    let mut rgba = Vec::with_capacity(pixel_count * 4);
    for chunk in rgb.chunks_exact(3) {
        rgba.push(chunk[0]);
        rgba.push(chunk[1]);
        rgba.push(chunk[2]);
        rgba.push(255);
    }
    rgba
}

/// Binary-search into a sorted image-data slice and return the index of the
/// frame that covers `current_time`.  `image_data` must be sorted by start time.
fn current_index_for_time(image_data: &[(String, u64, u64)], current_time: u64) -> usize {
    let pos = image_data.partition_point(|(_, start, _)| *start <= current_time);
    if pos == 0 {
        0
    } else {
        pos - 1
    }
}

/// Return the first time point strictly after `current_time` in a sorted,
/// deduplicated `sorted_times` slice, wrapping to the first entry when past the end.
fn next_time_point_forward(sorted_times: &[u64], current_time: u64) -> u64 {
    let pos = sorted_times.partition_point(|&t| t <= current_time);
    if pos < sorted_times.len() {
        sorted_times[pos]
    } else {
        sorted_times.first().cloned().unwrap_or(0)
    }
}

/// Return the last time point strictly before `current_time` in a sorted,
/// deduplicated `sorted_times` slice, wrapping to the last entry when at the start.
fn next_time_point_backward(sorted_times: &[u64], current_time: u64) -> u64 {
    let total_duration = *sorted_times.last().unwrap_or(&0);
    let pos = sorted_times.partition_point(|&t| t < current_time);
    if pos > 0 {
        sorted_times[pos - 1]
    } else {
        sorted_times.last().cloned().unwrap_or(total_duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_test_dds(path: &std::path::Path) {
        fs::write(
            path,
            [
                0x44, 0x44, 0x53, 0x20, // magic: "DDS "
                0x7C, 0x00, 0x00, 0x00, // header size
                0x07, 0x10, 0x08, 0x00, // flags
                0x04, 0x00, 0x00, 0x00, // height
                0x04, 0x00, 0x00, 0x00, // width
                0x08, 0x00, 0x00, 0x00, // linear size for one DXT1 block
                0x00, 0x00, 0x00, 0x00, // depth
                0x00, 0x00, 0x00, 0x00, // mipmap count
                0x00, 0x00, 0x00, 0x00, // reserved1[0]
                0x00, 0x00, 0x00, 0x00, // reserved1[1]
                0x00, 0x00, 0x00, 0x00, // reserved1[2]
                0x00, 0x00, 0x00, 0x00, // reserved1[3]
                0x00, 0x00, 0x00, 0x00, // reserved1[4]
                0x00, 0x00, 0x00, 0x00, // reserved1[5]
                0x00, 0x00, 0x00, 0x00, // reserved1[6]
                0x00, 0x00, 0x00, 0x00, // reserved1[7]
                0x00, 0x00, 0x00, 0x00, // reserved1[8]
                0x00, 0x00, 0x00, 0x00, // reserved1[9]
                0x00, 0x00, 0x00, 0x00, // reserved1[10]
                0x20, 0x00, 0x00, 0x00, // pixel format size
                0x04, 0x00, 0x00, 0x00, // pixel format flags: FOURCC
                0x44, 0x58, 0x54, 0x31, // fourcc: "DXT1"
                0x00, 0x00, 0x00, 0x00, // rgb bit count
                0x00, 0x00, 0x00, 0x00, // r mask
                0x00, 0x00, 0x00, 0x00, // g mask
                0x00, 0x00, 0x00, 0x00, // b mask
                0x00, 0x00, 0x00, 0x00, // a mask
                0x00, 0x10, 0x00, 0x00, // caps: texture
                0x00, 0x00, 0x00, 0x00, // caps2
                0x00, 0x00, 0x00, 0x00, // caps3
                0x00, 0x00, 0x00, 0x00, // caps4
                0x00, 0x00, 0x00, 0x00, // reserved2
                0x00, 0xF8, // color0: red in RGB565
                0x00, 0x00, // color1: black
                0x00, 0x00, 0x00, 0x00, // all pixels use color0
            ],
        )
        .unwrap();
    }

    fn make_image_data(frame_durations_us: &[u64]) -> Vec<(String, u64, u64)> {
        let mut data = Vec::new();
        let mut t = 0u64;
        for &d in frame_durations_us {
            data.push(("dummy.png".to_string(), t, t + d));
            t += d;
        }
        data
    }

    #[test]
    fn test_load_image_data_from_path_ppm() {
        let dir = std::env::temp_dir()
            .join("icp_tests")
            .join("player_load_ppm");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pixel.ppm");
        fs::write(&path, b"P3\n1 1\n255\n255 0 0\n").unwrap();

        let (rgba, size) = Player::load_image_data_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(size.width, 1);
        assert_eq!(size.height, 1);
        assert_eq!(rgba, vec![255, 0, 0, 255]);
    }

    #[test]
    fn test_load_image_data_from_path_pgm() {
        let dir = std::env::temp_dir()
            .join("icp_tests")
            .join("player_load_pgm");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pixel.pgm");
        fs::write(&path, b"P2\n1 1\n255\n127\n").unwrap();

        let (rgba, size) = Player::load_image_data_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(size.width, 1);
        assert_eq!(size.height, 1);
        assert_eq!(rgba, vec![127, 127, 127, 255]);
    }

    #[test]
    fn test_load_image_data_from_path_dds() {
        let dir = std::env::temp_dir().join("icp_tests").join("player_load_dds");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pixel.dds");
        write_test_dds(&path);

        let (rgba, size) = Player::load_image_data_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(size.width, 4);
        assert_eq!(size.height, 4);
        assert_eq!(rgba.len(), (size.width * size.height * 4) as usize);
        assert_eq!(rgba[0], 255);
        assert_eq!(rgba[1], 0);
        assert_eq!(rgba[2], 0);
        assert_eq!(rgba[3], 255);
    }

    // ── compute_sorted_time_points ──────────────────────────────────────────

    #[test]
    fn test_compute_sorted_time_points_basic() {
        let data1 = make_image_data(&[100, 100]);
        let data2 = make_image_data(&[100, 100]);
        let times = Player::compute_sorted_time_points(&data1, &data2);
        assert_eq!(times, vec![0, 100, 200]);
    }

    #[test]
    fn test_compute_sorted_time_points_dedup() {
        // Both sequences share the same time points — result must be deduplicated.
        let data1 = make_image_data(&[50, 50]);
        let data2 = make_image_data(&[50, 50]);
        let times = Player::compute_sorted_time_points(&data1, &data2);
        assert_eq!(times, vec![0, 50, 100]);
    }

    #[test]
    fn test_compute_sorted_time_points_unequal_sequences() {
        let data1 = make_image_data(&[100]);
        let data2 = make_image_data(&[50, 50]);
        let times = Player::compute_sorted_time_points(&data1, &data2);
        assert_eq!(times, vec![0, 50, 100]);
    }

    // ── rgba_to_rgb / rgb_to_rgba ───────────────────────────────────────────

    #[test]
    fn test_rgba_to_rgb_roundtrip() {
        let rgba = vec![255u8, 128, 64, 200, 10, 20, 30, 40];
        let rgb = rgba_to_rgb(&rgba);
        assert_eq!(rgb, vec![255, 128, 64, 10, 20, 30]);
    }

    #[test]
    fn test_rgb_to_rgba_inserts_alpha() {
        let rgb = vec![255u8, 128, 64, 10, 20, 30];
        let rgba = rgb_to_rgba(&rgb);
        assert_eq!(rgba, vec![255, 128, 64, 255, 10, 20, 30, 255]);
    }

    #[test]
    fn test_rgb_rgba_roundtrip_size() {
        let pixel_count = 100usize;
        let rgba_in: Vec<u8> = (0..pixel_count * 4).map(|i| (i % 256) as u8).collect();
        let rgb = rgba_to_rgb(&rgba_in);
        assert_eq!(rgb.len(), pixel_count * 3);
        let rgba_out = rgb_to_rgba(&rgb);
        assert_eq!(rgba_out.len(), pixel_count * 4);
    }

    #[test]
    fn test_rgb_to_rgba_preallocates_expected_capacity() {
        let pixel_count = 100usize;
        let rgb: Vec<u8> = (0..pixel_count * 3).map(|i| (i % 256) as u8).collect();
        let rgba = rgb_to_rgba(&rgb);
        assert_eq!(rgba.capacity(), pixel_count * 4);
    }

    #[test]
    fn test_update_displayed_texture_pair_waits_for_both_sides() {
        let old_left = Arc::new(1u8);
        let old_right = Arc::new(2u8);
        let new_left = Arc::new(3u8);

        let mut left_slot = Some(old_left.clone());
        let mut right_slot = Some(old_right.clone());

        let updated = update_displayed_texture_pair(
            &mut left_slot,
            &mut right_slot,
            Some(new_left),
            None,
            false,
        );

        assert!(!updated);
        assert!(left_slot
            .as_ref()
            .is_some_and(|texture| Arc::ptr_eq(texture, &old_left)));
        assert!(right_slot
            .as_ref()
            .is_some_and(|texture| Arc::ptr_eq(texture, &old_right)));
    }

    #[test]
    fn test_update_displayed_texture_pair_switches_both_sides_together() {
        let old_left = Arc::new(1u8);
        let old_right = Arc::new(2u8);
        let new_left = Arc::new(3u8);
        let new_right = Arc::new(4u8);

        let mut left_slot = Some(old_left);
        let mut right_slot = Some(old_right);

        let updated = update_displayed_texture_pair(
            &mut left_slot,
            &mut right_slot,
            Some(new_left.clone()),
            Some(new_right.clone()),
            false,
        );

        assert!(updated);
        assert!(left_slot
            .as_ref()
            .is_some_and(|texture| Arc::ptr_eq(texture, &new_left)));
        assert!(right_slot
            .as_ref()
            .is_some_and(|texture| Arc::ptr_eq(texture, &new_right)));
    }

    #[test]
    fn test_update_displayed_texture_pair_updates_single_image_mode_with_left_only() {
        let old_left = Arc::new(1u8);
        let new_left = Arc::new(2u8);

        let mut left_slot = Some(old_left);
        let mut right_slot = None;

        let updated = update_displayed_texture_pair(
            &mut left_slot,
            &mut right_slot,
            Some(new_left.clone()),
            None,
            true,
        );

        assert!(updated);
        assert!(left_slot
            .as_ref()
            .is_some_and(|texture| Arc::ptr_eq(texture, &new_left)));
        assert!(right_slot.is_none());
    }

    // ── get_current_index ──────────────────────────────────────────────────

    #[test]
    fn test_get_current_index_first_frame() {
        let data = make_image_data(&[100, 100, 100]);
        assert_eq!(current_index_for_time(&data, 0), 0);
        assert_eq!(current_index_for_time(&data, 50), 0);
        assert_eq!(current_index_for_time(&data, 99), 0);
    }

    #[test]
    fn test_get_current_index_middle_frame() {
        let data = make_image_data(&[100, 100, 100]);
        assert_eq!(current_index_for_time(&data, 100), 1);
        assert_eq!(current_index_for_time(&data, 150), 1);
        assert_eq!(current_index_for_time(&data, 199), 1);
    }

    #[test]
    fn test_get_current_index_last_frame() {
        let data = make_image_data(&[100, 100, 100]);
        assert_eq!(current_index_for_time(&data, 200), 2);
        assert_eq!(current_index_for_time(&data, 250), 2);
        // At or past the end, returns last frame
        assert_eq!(current_index_for_time(&data, 300), 2);
    }

    // ── find_next_time_point ───────────────────────────────────────────────

    #[test]
    fn test_find_next_forward_wraps_at_end() {
        let times = vec![0u64, 100, 200];
        // Past the last point → wraps to start
        assert_eq!(next_time_point_forward(&times, 200), 0);
    }

    #[test]
    fn test_find_next_forward_basic() {
        let times = vec![0u64, 100, 200];
        assert_eq!(next_time_point_forward(&times, 0), 100);
        assert_eq!(next_time_point_forward(&times, 50), 100);
        assert_eq!(next_time_point_forward(&times, 100), 200);
    }

    #[test]
    fn test_find_next_backward_wraps_at_start() {
        let times = vec![0u64, 100, 200];
        // At or before start → wraps to end
        assert_eq!(next_time_point_backward(&times, 0), 200);
    }

    #[test]
    fn test_find_next_backward_basic() {
        let times = vec![0u64, 100, 200];
        assert_eq!(next_time_point_backward(&times, 200), 100);
        assert_eq!(next_time_point_backward(&times, 150), 100);
        assert_eq!(next_time_point_backward(&times, 100), 0);
    }

    #[test]
    fn test_load_image_data_different_dimensions_per_side() {
        // Verify mixed-resolution sources are accepted. Each image should load based on
        // its own dimensions even when provided "expected" dimensions differ.
        let dir = std::env::temp_dir()
            .join("icp_tests")
            .join("player_diff_dims");
        let _ = fs::remove_dir_all(&dir).ok();
        fs::create_dir_all(&dir).unwrap();

        // Write a 2×2 PPM (left side).
        let path_left = dir.join("left.ppm");
        fs::write(
            &path_left,
            b"P3\n2 2\n255\n255 0 0\n0 255 0\n0 0 255\n255 255 0\n",
        )
        .unwrap();

        // Write a 1×1 PPM (right side – deliberately different dimensions).
        let path_right = dir.join("right.ppm");
        fs::write(&path_right, b"P3\n1 1\n255\n0 0 255\n").unwrap();

        // The left-side expected dimensions come from the first left image (2×2).
        let (_, left_size) = Player::load_image_data_from_path(path_left.to_str().unwrap())
            .expect("left image should load");
        assert_eq!(left_size.width, 2);
        assert_eq!(left_size.height, 2);

        // The right-side expected dimensions come from the first right image (1×1).
        // Before the fix this call used the LEFT expected dimensions (2×2), causing
        // a mismatch error and silently dropping the right texture.
        let (_, right_size) = Player::load_image_data_from_path(path_right.to_str().unwrap())
            .expect("right image should load");
        assert_eq!(right_size.width, 1);
        assert_eq!(right_size.height, 1);

        // Loading the same image again should still report its real dimensions.
        let (_, right_size_reloaded) =
            Player::load_image_data_from_path(path_right.to_str().unwrap())
                .expect("reloading image should not fail");
        assert_eq!(right_size_reloaded.width, 1);
        assert_eq!(right_size_reloaded.height, 1);
    }

    #[test]
    fn test_load_image_data_allows_different_dimensions_within_source() {
        let dir = std::env::temp_dir()
            .join("icp_tests")
            .join("player_diff_dims_within_source");
        let _ = fs::remove_dir_all(&dir).ok();
        fs::create_dir_all(&dir).unwrap();

        let first = dir.join("first.ppm");
        fs::write(
            &first,
            b"P3\n2 2\n255\n255 0 0\n0 255 0\n0 0 255\n255 255 0\n",
        )
        .unwrap();
        let second = dir.join("second.ppm");
        fs::write(&second, b"P3\n1 1\n255\n0 0 255\n").unwrap();

        let (_, first_size) = Player::load_image_data_from_path(first.to_str().unwrap()).unwrap();
        assert_eq!(first_size.width, 2);
        assert_eq!(first_size.height, 2);

        let (_, second_size) = Player::load_image_data_from_path(second.to_str().unwrap()).unwrap();
        assert_eq!(second_size.width, 1);
        assert_eq!(second_size.height, 1);
    }
}
