use ggez::graphics::Image;
use image;
use log::{debug, warn};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

const CACHE_SIZE: usize = 20;
const PRELOAD_AHEAD: usize = 5;

pub struct Player {
    image_data1: Vec<(String, u64, u64)>,
    image_data2: Vec<(String, u64, u64)>,
    image_cache1: Arc<Mutex<VecDeque<(usize, Image)>>>,
    image_cache2: Arc<Mutex<VecDeque<(usize, Image)>>>,
    current_time: u64,
    is_playing: bool,
    last_update: Instant,
    preload_sender: std::sync::mpsc::Sender<(String, usize, Arc<Mutex<VecDeque<(usize, Image)>>>)>,
    preload_receiver: std::sync::mpsc::Receiver<(
        usize,
        Vec<u8>,
        u32,
        u32,
        Arc<Mutex<VecDeque<(usize, Image)>>>,
    )>,
}

impl Player {
    pub fn new(images1: Vec<(String, u64)>, images2: Vec<(String, u64)>) -> Self {
        let image_data1 = Self::accumulate_durations(images1);
        let image_data2 = Self::accumulate_durations(images2);
        let image_cache1 = Arc::new(Mutex::new(VecDeque::new()));
        let image_cache2 = Arc::new(Mutex::new(VecDeque::new()));

        let (sender, receiver) = std::sync::mpsc::channel();
        let (load_sender, load_receiver) = std::sync::mpsc::channel();

        let player = Self {
            image_data1,
            image_data2,
            image_cache1: Arc::clone(&image_cache1),
            image_cache2: Arc::clone(&image_cache2),
            current_time: 0,
            is_playing: false,
            last_update: Instant::now(),
            preload_sender: sender,
            preload_receiver: load_receiver,
        };

        player.start_preload_thread(receiver, load_sender);
        player
    }

    fn start_preload_thread(
        &self,
        receiver: std::sync::mpsc::Receiver<(String, usize, Arc<Mutex<VecDeque<(usize, Image)>>>)>,
        sender: std::sync::mpsc::Sender<(
            usize,
            Vec<u8>,
            u32,
            u32,
            Arc<Mutex<VecDeque<(usize, Image)>>>,
        )>,
    ) {
        thread::spawn(move || {
            while let Ok((path, index, cache)) = receiver.recv() {
                if let Ok(img) = image::open(&path) {
                    let rgba = img.to_rgba8();
                    let width = rgba.width();
                    let height = rgba.height();
                    let rgba_vec = rgba.into_raw();
                    sender
                        .send((index, rgba_vec, width, height, cache))
                        .unwrap();
                }
            }
        });
    }

    pub fn update(&mut self, ctx: &mut ggez::Context) {
        while let Ok((index, rgba, width, height, cache)) = self.preload_receiver.try_recv() {
            let mut cache = cache.lock().unwrap();
            if !cache.iter().any(|(i, _)| *i == index) {
                let image = Image::from_pixels(
                    ctx,
                    &rgba,
                    ggez::graphics::ImageFormat::Rgba8UnormSrgb,
                    width,
                    height,
                );
                if cache.len() >= CACHE_SIZE {
                    cache.pop_back();
                }
                cache.push_front((index, image));
            }
        }
    }

    pub fn current_images(&mut self, ctx: &mut ggez::Context) -> (Arc<Image>, Arc<Image>) {
        let index1 = self.get_current_index(&self.image_data1);
        let index2 = self.get_current_index(&self.image_data2);

        let image1 = Self::get_or_load_image(ctx, &self.image_data1, index1, &self.image_cache1);
        let image2 = Self::get_or_load_image(ctx, &self.image_data2, index2, &self.image_cache2);

        self.trigger_preload(index1, index2);

        (image1, image2)
    }

    fn get_or_load_image(
        ctx: &mut ggez::Context,
        image_data: &[(String, u64, u64)],
        index: usize,
        cache: &Arc<Mutex<VecDeque<(usize, Image)>>>,
    ) -> Arc<Image> {
        let mut cache = cache.lock().unwrap();
        if let Some(pos) = cache.iter().position(|(i, _)| *i == index) {
            let (_, image) = cache.remove(pos).unwrap();
            cache.push_front((index, image.clone()));
            Arc::new(cache[0].1.clone())
        } else {
            let (path, _, _) = &image_data[index];
            match Image::from_path(ctx, path) {
                Ok(image) => {
                    if cache.len() >= CACHE_SIZE {
                        cache.pop_back();
                    }
                    let arc_image = Arc::new(image.clone());
                    cache.push_front((index, image));
                    arc_image
                }
                Err(e) => {
                    warn!("Failed to load image at {}: {}", path, e);
                    Arc::new(cache[0].1.clone())
                }
            }
        }
    }

    fn trigger_preload(&self, index1: usize, index2: usize) {
        for i in 1..=PRELOAD_AHEAD {
            let preload_index1 = (index1 + i) % self.image_data1.len();
            let preload_index2 = (index2 + i) % self.image_data2.len();

            let (path1, _, _) = &self.image_data1[preload_index1];
            let (path2, _, _) = &self.image_data2[preload_index2];

            self.preload_sender
                .send((
                    path1.clone(),
                    preload_index1,
                    Arc::clone(&self.image_cache1),
                ))
                .unwrap();
            self.preload_sender
                .send((
                    path2.clone(),
                    preload_index2,
                    Arc::clone(&self.image_cache2),
                ))
                .unwrap();
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
