use ggez::graphics::Image;
use log::debug;

pub struct Player {
    images1: Vec<(Image, u64, u64)>,
    images2: Vec<(Image, u64, u64)>,
    current_time: u64,
    is_playing: bool,
}

impl Player {
    pub fn new(images1: Vec<(Image, u64)>, images2: Vec<(Image, u64)>) -> Self {
        let images1 = Self::accumulate_durations(images1);
        let images2 = Self::accumulate_durations(images2);
        Self {
            images1,
            images2,
            current_time: 0,
            is_playing: false,
        }
    }

    fn accumulate_durations(images: Vec<(Image, u64)>) -> Vec<(Image, u64, u64)> {
        let mut accumulated = 0;
        images
            .into_iter()
            .map(|(image, duration)| {
                let start = accumulated;
                accumulated += duration;
                (image, start, accumulated)
            })
            .collect()
    }

    pub fn is_playing(&self) -> bool {
        self.is_playing
    }

    pub fn toggle_play_pause(&mut self) {
        self.is_playing = !self.is_playing;
        debug!(
            "Player is now {}",
            if self.is_playing { "playing" } else { "paused" }
        );
    }

    pub fn next_frame(&mut self) {
        let next_time = self
            .images1
            .iter()
            .find(|(_, _, end)| *end > self.current_time)
            .map(|(_, _, end)| *end)
            .unwrap_or_else(|| self.images1.last().map(|(_, _, end)| *end).unwrap_or(0));
        self.current_time = next_time;
        let total_duration = self.total_duration();
        debug!(
            "Current time: {} / {} ms",
            self.current_time, total_duration
        );
    }

    pub fn previous_frame(&mut self) {
        let prev_time = self
            .images1
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
        if self.current_time > total_duration {
            self.current_time = self.current_time % total_duration;
        }
        debug!(
            "Advanced frame: old_time = {}, delta_time = {}, new_time = {}, total_duration = {}",
            old_time, delta_time, self.current_time, total_duration
        );
    }

    pub fn current_images(&self) -> (&Image, &Image) {
        fn get_image<'a>(images: &'a [(Image, u64, u64)], current_time: u64) -> (&'a Image, usize) {
            debug!("Selecting image for current_time: {}", current_time);
            for (index, (image, start, end)) in images.iter().enumerate() {
                debug!("Checking frame {}: start = {}, end = {}", index, start, end);
                if *start <= current_time && current_time < *end {
                    debug!(
                        "Selected frame {} with start {} and end {}",
                        index, start, end
                    );
                    return (image, index);
                }
            }
            debug!("No matching frame found, defaulting to frame 0");
            (&images[0].0, 0)
        }

        let (image1, frame1) = get_image(&self.images1, self.current_time);
        let (image2, frame2) = get_image(&self.images2, self.current_time);

        debug!("Selected frames: image1 = {}, image2 = {}", frame1, frame2);
        debug!(
            "Current time: {}, Total duration: {}",
            self.current_time,
            self.total_duration()
        );

        (image1, image2)
    }

    fn total_duration(&self) -> u64 {
        self.images1.last().map(|(_, _, end)| *end).unwrap_or(0)
    }
}
