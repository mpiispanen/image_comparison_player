use ggez::graphics::Image;
use log::{debug, warn};
use std::collections::VecDeque;
use std::time::Instant;

const CACHE_SIZE: usize = 20; // Adjust this value based on your needs and available memory

pub struct Player {
    image_data1: Vec<(String, u64, u64)>,
    image_data2: Vec<(String, u64, u64)>,
    image_cache1: VecDeque<(usize, Image)>,
    image_cache2: VecDeque<(usize, Image)>,
    current_time: u64,
    is_playing: bool,
    last_update: Instant,
}

impl Player {
    pub fn new(images1: Vec<(String, u64)>, images2: Vec<(String, u64)>) -> Self {
        let image_data1 = Self::accumulate_durations(images1);
        let image_data2 = Self::accumulate_durations(images2);
        Self {
            image_data1,
            image_data2,
            image_cache1: VecDeque::new(),
            image_cache2: VecDeque::new(),
            current_time: 0,
            is_playing: false,
            last_update: Instant::now(),
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

    pub fn current_images(&mut self, ctx: &mut ggez::Context) -> (&Image, &Image) {
        let index1 = self.get_current_index(&self.image_data1);
        let index2 = self.get_current_index(&self.image_data2);

        let image1 =
            Self::get_or_load_image(ctx, &self.image_data1, index1, &mut self.image_cache1);
        let image2 =
            Self::get_or_load_image(ctx, &self.image_data2, index2, &mut self.image_cache2);

        (image1, image2)
    }

    fn get_current_index(&self, image_data: &[(String, u64, u64)]) -> usize {
        image_data
            .iter()
            .position(|(_, start, end)| *start <= self.current_time && self.current_time < *end)
            .unwrap_or(0)
    }

    fn get_or_load_image<'a>(
        ctx: &mut ggez::Context,
        image_data: &[(String, u64, u64)],
        index: usize,
        cache: &'a mut VecDeque<(usize, Image)>,
    ) -> &'a Image {
        if let Some(pos) = cache.iter().position(|(i, _)| *i == index) {
            let (_, image) = cache.remove(pos).unwrap();
            cache.push_front((index, image));
            &cache[0].1
        } else {
            let (path, _, _) = &image_data[index];
            match Image::from_path(ctx, path) {
                Ok(image) => {
                    if cache.len() >= CACHE_SIZE {
                        cache.pop_back();
                    }
                    cache.push_front((index, image));
                    &cache[0].1
                }
                Err(e) => {
                    warn!("Failed to load image at {}: {}", path, e);
                    // Return a placeholder or default image
                    &cache[0].1 // Assuming there's at least one image in the cache
                }
            }
        }
    }

    fn total_duration(&self) -> u64 {
        self.image_data1.last().map(|(_, _, end)| *end).unwrap_or(0)
    }
}
