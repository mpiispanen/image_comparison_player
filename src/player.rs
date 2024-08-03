use ggez::graphics::Image;

pub struct Player {
    images1: Vec<(Image, u64)>,
    images2: Vec<(Image, u64)>,
    current_time: u64,
    is_playing: bool,
}

impl Player {
    pub fn new(images1: Vec<(Image, u64)>, images2: Vec<(Image, u64)>) -> Self {
        Self {
            images1,
            images2,
            current_time: 0,
            is_playing: false,
        }
    }

    pub fn is_playing(&self) -> bool {
        self.is_playing
    }

    pub fn toggle_play_pause(&mut self) {
        self.is_playing = !self.is_playing;
    }

    pub fn next_frame(&mut self) {
        let next_time = self
            .images1
            .iter()
            .find(|(_, duration)| *duration > self.current_time)
            .map(|(_, duration)| *duration)
            .unwrap_or_else(|| {
                self.images1
                    .last()
                    .map(|(_, duration)| *duration)
                    .unwrap_or(0)
            });
        self.current_time = next_time;
    }

    pub fn previous_frame(&mut self) {
        let prev_time = self
            .images1
            .iter()
            .rev()
            .find(|(_, duration)| *duration < self.current_time)
            .map(|(_, duration)| *duration)
            .unwrap_or(0);
        self.current_time = prev_time;
    }

    pub fn advance_frame(&mut self, delta_time: u64) {
        self.current_time += delta_time;
        if self.current_time > self.total_duration() {
            self.current_time = 0;
        }
    }

    pub fn current_images(&self) -> (&Image, &Image) {
        fn get_image<'a>(images: &'a [(Image, u64)], current_time: u64) -> &'a Image {
            images
                .iter()
                .rev()
                .find(|(_, duration)| *duration <= current_time)
                .map(|(image, _)| image)
                .unwrap_or_else(|| &images[0].0)
        }

        (
            get_image(&self.images1, self.current_time),
            get_image(&self.images2, self.current_time),
        )
    }

    fn total_duration(&self) -> u64 {
        self.images1
            .last()
            .map(|(_, duration)| *duration)
            .unwrap_or(0)
    }
}
