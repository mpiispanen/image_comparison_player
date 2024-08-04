use ggez::graphics::Image;
use image;
use log::{debug, warn};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex as StdMutex};
use std::thread;
use std::time::Instant;

type PreloadRequest = (
    String,
    usize,
    Arc<Mutex<HashMap<usize, Arc<Image>>>>,
    Arc<StdMutex<HashSet<usize>>>,
);

type PreloadResponse = (
    usize,
    Vec<u8>,
    u32,
    u32,
    Arc<Mutex<HashMap<usize, Arc<Image>>>>,
);

pub struct Player {
    image_data1: Vec<(String, u64, u64)>,
    image_data2: Vec<(String, u64, u64)>,
    image_cache1: Arc<Mutex<HashMap<usize, Arc<Image>>>>,
    image_cache2: Arc<Mutex<HashMap<usize, Arc<Image>>>>,
    current_time: u64,
    is_playing: bool,
    last_update: Instant,
    preload_sender: std::sync::mpsc::Sender<PreloadRequest>,
    preload_receiver: std::sync::mpsc::Receiver<PreloadResponse>,
    ongoing_preloads1: Arc<StdMutex<HashSet<usize>>>,
    ongoing_preloads2: Arc<StdMutex<HashSet<usize>>>,
    preloaded_frames1: Arc<StdMutex<HashSet<usize>>>,
    preloaded_frames2: Arc<StdMutex<HashSet<usize>>>,
    cache_size: usize,
    preload_ahead: usize,
}

impl Player {
    pub fn new(
        images1: Vec<(String, u64)>,
        images2: Vec<(String, u64)>,
        cache_size: usize,
        preload_ahead: usize,
    ) -> Self {
        let image_data1 = Self::accumulate_durations(images1);
        let image_data2 = Self::accumulate_durations(images2);
        let image_cache1 = Arc::new(Mutex::new(HashMap::with_capacity(cache_size)));
        let image_cache2 = Arc::new(Mutex::new(HashMap::with_capacity(cache_size)));

        let (sender, receiver) = std::sync::mpsc::channel();
        let (load_sender, load_receiver) = std::sync::mpsc::channel();

        let player = Self {
            image_data1,
            image_data2,
            image_cache1,
            image_cache2,
            current_time: 0,
            is_playing: false,
            last_update: Instant::now(),
            preload_sender: sender,
            preload_receiver: load_receiver,
            ongoing_preloads1: Arc::new(StdMutex::new(HashSet::new())),
            ongoing_preloads2: Arc::new(StdMutex::new(HashSet::new())),
            preloaded_frames1: Arc::new(StdMutex::new(HashSet::new())),
            preloaded_frames2: Arc::new(StdMutex::new(HashSet::new())),
            cache_size,
            preload_ahead,
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
                ongoing.lock().unwrap().remove(&index);
            }
        });
    }

    pub fn update(&mut self, ctx: &mut ggez::Context) {
        while let Ok((index, rgba, width, height, cache)) = self.preload_receiver.try_recv() {
            let mut cache_guard = cache.lock();
            let cache_index = index % self.cache_size;
            let image = Image::from_pixels(
                ctx,
                &rgba,
                ggez::graphics::ImageFormat::Rgba8UnormSrgb,
                width,
                height,
            );
            cache_guard.insert(index, Arc::new(image));

            if cache_guard.len() > self.cache_size {
                if let Some(oldest) = cache_guard.keys().min().cloned() {
                    cache_guard.remove(&oldest);
                }
            }
        }
    }

    pub fn current_images(&mut self, ctx: &mut ggez::Context) -> (Arc<Image>, Arc<Image>) {
        let index1 = self.get_current_index(&self.image_data1);
        let index2 = self.get_current_index(&self.image_data2);

        debug!("Current indices: {} {}", index1, index2);

        let image1 = Self::get_or_load_image(
            ctx,
            &self.image_data1,
            index1,
            &self.image_cache1,
            self.cache_size,
        );
        let image2 = Self::get_or_load_image(
            ctx,
            &self.image_data2,
            index2,
            &self.image_cache2,
            self.cache_size,
        );

        self.trigger_preload(index1, index2);

        (image1, image2)
    }

    fn get_or_load_image(
        ctx: &mut ggez::Context,
        image_data: &[(String, u64, u64)],
        index: usize,
        cache: &Arc<Mutex<HashMap<usize, Arc<Image>>>>,
        cache_size: usize,
    ) -> Arc<Image> {
        let mut cache_guard = cache.lock();

        if let Some(image) = cache_guard.get(&index) {
            debug!("Image found in cache for index {}", index);
            return Arc::clone(image);
        }

        let (path, _, _) = &image_data[index];
        debug!("Loading image from path: {} for index {}", path, index);
        match Image::from_path(ctx, path) {
            Ok(image) => {
                let arc_image = Arc::new(image);
                if cache_guard.len() >= cache_size {
                    if let Some(oldest) = cache_guard.keys().next().cloned() {
                        cache_guard.remove(&oldest);
                    }
                }
                cache_guard.insert(index, Arc::clone(&arc_image));
                debug!("Successfully loaded image for index {}", index);
                arc_image
            }
            Err(e) => {
                warn!("Failed to load image at {}: {}", path, e);
                Arc::new(Image::from_pixels(
                    ctx,
                    &[255, 0, 255, 255],
                    ggez::graphics::ImageFormat::Rgba8UnormSrgb,
                    1,
                    1,
                ))
            }
        }
    }

    fn trigger_preload(&self, index1: usize, index2: usize) {
        for i in 1..=self.preload_ahead {
            let preload_index1 = (index1 + i) % self.image_data1.len();
            let preload_index2 = (index2 + i) % self.image_data2.len();

            debug!("Checking preload for index1: {}", preload_index1);

            let in_cache = {
                let cache_lock = self.image_cache1.lock();
                cache_lock.contains_key(&preload_index1)
            };
            let in_ongoing = {
                let ongoing_lock = self.ongoing_preloads1.lock().unwrap();
                ongoing_lock.contains(&preload_index1)
            };
            let in_preloaded = {
                let preloaded_lock = self.preloaded_frames1.lock().unwrap();
                preloaded_lock.contains(&preload_index1)
            };

            debug!(
                "Index {} - in_cache: {}, in_ongoing: {}, in_preloaded: {}",
                preload_index1, in_cache, in_ongoing, in_preloaded
            );

            let should_preload1 = !in_cache && !in_ongoing && !in_preloaded;

            debug!("should_preload1: {}", should_preload1);

            if should_preload1 {
                let (path1, _, _) = &self.image_data1[preload_index1];
                {
                    let mut ongoing_lock = self.ongoing_preloads1.lock().unwrap();
                    ongoing_lock.insert(preload_index1);
                    debug!("Added {} to ongoing_preloads1", preload_index1);
                }
                {
                    let mut preloaded_lock = self.preloaded_frames1.lock().unwrap();
                    preloaded_lock.insert(preload_index1);
                    debug!("Added {} to preloaded_frames1", preload_index1);
                }
                self.preload_sender
                    .send((
                        path1.clone(),
                        preload_index1,
                        Arc::clone(&self.image_cache1),
                        Arc::clone(&self.ongoing_preloads1),
                    ))
                    .unwrap();
                debug!("Sent preload request for index1: {}", preload_index1);
            }

            // Similar changes for preload_index2...
            debug!("Checking preload for index2: {}", preload_index2);

            let in_cache = {
                let cache_lock = self.image_cache2.lock();
                cache_lock.contains_key(&preload_index2)
            };
            let in_ongoing = {
                let ongoing_lock = self.ongoing_preloads2.lock().unwrap();
                ongoing_lock.contains(&preload_index2)
            };
            let in_preloaded = {
                let preloaded_lock = self.preloaded_frames2.lock().unwrap();
                preloaded_lock.contains(&preload_index2)
            };

            debug!(
                "Index {} - in_cache: {}, in_ongoing: {}, in_preloaded: {}",
                preload_index2, in_cache, in_ongoing, in_preloaded
            );

            let should_preload2 = !in_cache && !in_ongoing && !in_preloaded;

            debug!("should_preload2: {}", should_preload2);

            if should_preload2 {
                let (path2, _, _) = &self.image_data2[preload_index2];
                {
                    let mut ongoing_lock = self.ongoing_preloads2.lock().unwrap();
                    ongoing_lock.insert(preload_index2);
                    debug!("Added {} to ongoing_preloads2", preload_index2);
                }
                {
                    let mut preloaded_lock = self.preloaded_frames2.lock().unwrap();
                    preloaded_lock.insert(preload_index2);
                    debug!("Added {} to preloaded_frames2", preload_index2);
                }
                self.preload_sender
                    .send((
                        path2.clone(),
                        preload_index2,
                        Arc::clone(&self.image_cache2),
                        Arc::clone(&self.ongoing_preloads2),
                    ))
                    .unwrap();
                debug!("Sent preload request for index2: {}", preload_index2);
            }
        }
    }

    fn accumulate_durations(images: Vec<(String, u64)>) -> Vec<(String, u64, u64)> {
        let mut accumulated = 0;
        images
            .into_iter()
            .map(|(path, duration)| {
                let start = accumulated;
                accumulated += duration;
                (path, start, accumulated)
            })
            .collect()
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
}
