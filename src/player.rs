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
    Arc<Mutex<Option<Arc<Image>>>>,
    Arc<Mutex<bool>>,
);

type PreloadResponse = (usize, Vec<u8>, u32, u32, Arc<Mutex<Option<Arc<Image>>>>);

pub struct Player {
    image_data1: Vec<(String, u64, u64)>,
    image_data2: Vec<(String, u64, u64)>,
    image_cache1: HashMap<usize, Arc<Mutex<Option<Arc<Image>>>>>,
    image_cache2: HashMap<usize, Arc<Mutex<Option<Arc<Image>>>>>,
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
}

impl Player {
    pub fn new(
        images1: (Vec<(String, u64)>, usize),
        images2: (Vec<(String, u64)>, usize),
        cache_size: usize,
        preload_ahead: usize,
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
                *ongoing.lock() = false;
            }
        });
    }

    pub fn update(&mut self, ctx: &mut ggez::Context) {
        while let Ok((index, rgba, width, height, cache)) = self.preload_receiver.try_recv() {
            let image = Image::from_pixels(
                ctx,
                &rgba,
                ggez::graphics::ImageFormat::Rgba8UnormSrgb,
                width,
                height,
            );
            let arc_image = Arc::new(image);

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

            cache.insert(index, Arc::new(Mutex::new(Some(Arc::clone(&arc_image)))));
            if !order.contains(&index) {
                order.push_back(index);
            }
            *preloaded_frames[index].lock() = true;

            debug!("Added preloaded image to cache for index: {}", index);
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
            &mut self.image_cache1,
            &mut self.cache_order1,
        );
        let image2 = Self::get_or_load_image(
            ctx,
            &self.image_data2,
            index2,
            &mut self.image_cache2,
            &mut self.cache_order2,
        );

        self.trigger_preload(index1, index2);

        (image1, image2)
    }

    fn get_or_load_image(
        ctx: &mut ggez::Context,
        image_data: &[(String, u64, u64)],
        index: usize,
        cache: &mut HashMap<usize, Arc<Mutex<Option<Arc<Image>>>>>,
        cache_order: &mut VecDeque<usize>,
    ) -> Arc<Image> {
        if cache.is_empty() {
            // Handle empty cache case
            let (path, _, _) = &image_data[index];
            debug!(
                "Cache is empty. Loading image from path: {} for index {}",
                path, index
            );
            return Self::load_image(ctx, path, index, cache, cache_order);
        }

        if let Some(entry) = cache.get(&index) {
            if let Some(image) = entry.lock().as_ref() {
                debug!("Image found in cache for index {}", index);
                return Arc::clone(image);
            }
        }

        let (path, _, _) = &image_data[index];
        debug!("Loading image from path: {} for index {}", path, index);
        match Image::from_path(ctx, path) {
            Ok(image) => {
                let arc_image = Arc::new(image);
                if let Some(entry) = cache.get(&index) {
                    *entry.lock() = Some(Arc::clone(&arc_image));
                } else {
                    cache.insert(index, Arc::new(Mutex::new(Some(Arc::clone(&arc_image)))));
                    cache_order.push_back(index);
                }
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

    fn load_image(
        ctx: &mut ggez::Context,
        path: &str,
        index: usize,
        cache: &mut HashMap<usize, Arc<Mutex<Option<Arc<Image>>>>>,
        cache_order: &mut VecDeque<usize>,
    ) -> Arc<Image> {
        debug!("Loading image from path: {} for index {}", path, index);
        match Image::from_path(ctx, path) {
            Ok(image) => {
                let arc_image = Arc::new(image);
                cache.insert(index, Arc::new(Mutex::new(Some(Arc::clone(&arc_image)))));
                cache_order.push_back(index);
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
        cache: &mut HashMap<usize, Arc<Mutex<Option<Arc<Image>>>>>,
        ongoing_preloads: &mut [Arc<Mutex<bool>>],
        preloaded_frames: &mut [Arc<Mutex<bool>>],
        preload_sender: &std::sync::mpsc::Sender<PreloadRequest>,
    ) {
        let in_cache = cache
            .get(&index)
            .map_or(false, |entry| entry.lock().is_some());
        let in_ongoing = *ongoing_preloads[index].lock();

        if in_cache {
            *ongoing_preloads[index].lock() = false;
            *preloaded_frames[index].lock() = true;
        } else if !in_ongoing {
            let (path, _, _) = &image_data[index];
            *ongoing_preloads[index].lock() = true;
            *preloaded_frames[index].lock() = false;

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
}
