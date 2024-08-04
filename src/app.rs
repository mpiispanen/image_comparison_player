use crate::image_loader;
use crate::player::Player;
use ggez::event::EventHandler;
use ggez::graphics::{self, DrawParam};
use ggez::input::keyboard::KeyCode;
use ggez::input::mouse::MouseButton;
use ggez::{Context, GameResult};
use log::{debug, error, info};
use std::time::Instant;

pub struct AppState {
    player: Player,
    cursor_x: f32,
    last_update: Instant,
}

impl AppState {
    pub fn new(ctx: &mut Context, dir1: String, dir2: String) -> GameResult<Self> {
        info!("Initializing AppState");
        let images1 = image_loader::load_image_paths(ctx, &dir1).map_err(|e| {
            error!("Error loading images from dir1: {}", e);
            AppError(e)
        })?;
        let images2 = image_loader::load_image_paths(ctx, &dir2).map_err(|e| {
            error!("Error loading images from dir2: {}", e);
            AppError(e)
        })?;
        let player = Player::new(images1, images2);
        info!("AppState initialized successfully");
        Ok(Self {
            player,
            cursor_x: ctx.gfx.drawable_size().0 / 2.0,
            last_update: Instant::now(),
        })
    }

    pub fn handle_input(&mut self, keycode: KeyCode) {
        match keycode {
            KeyCode::Space => {
                self.player.toggle_play_pause();
                debug!(
                    "Toggled play/pause. Is playing: {}",
                    self.player.is_playing()
                );
            }
            KeyCode::Right => {
                self.player.next_frame();
                debug!("Moved to next frame");
            }
            KeyCode::Left => {
                self.player.previous_frame();
                debug!("Moved to previous frame");
            }
            _ => {}
        }
    }
}

pub struct AppError(anyhow::Error);

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError(err)
    }
}

impl From<AppError> for ggez::GameError {
    fn from(err: AppError) -> Self {
        ggez::GameError::CustomError(err.0.to_string())
    }
}

impl EventHandler for AppState {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        let now = Instant::now();
        let delta = now.duration_since(self.last_update);
        self.last_update = now;

        if self.player.is_playing() {
            self.player.advance_frame(delta.as_micros() as u64);
            debug!("Advanced frame, delta: {:?}", delta);
        }
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, None);
        let (left_image, right_image) = self.player.current_images(ctx);

        let (window_width, window_height) = ctx.gfx.drawable_size();
        let scale_x = window_width / left_image.width() as f32;
        let scale_y = window_height / left_image.height() as f32;
        let scale = scale_x.min(scale_y);

        let scaled_width = left_image.width() as f32 * scale;
        let scaled_height = left_image.height() as f32 * scale;
        let x_offset = (window_width - scaled_width) / 2.0;
        let y_offset = (window_height - scaled_height) / 2.0;

        let scaled_cursor_x =
            ((self.cursor_x - x_offset) / scale).clamp(0.0, left_image.width() as f32);
        let cursor_ratio = scaled_cursor_x / left_image.width() as f32;

        // Draw left image
        canvas.draw(
            left_image,
            DrawParam::default()
                .src(graphics::Rect::new(0.0, 0.0, cursor_ratio, 1.0))
                .dest([x_offset, y_offset])
                .scale([scale, scale]),
        );

        // Draw right image
        canvas.draw(
            right_image,
            DrawParam::default()
                .src(graphics::Rect::new(
                    cursor_ratio,
                    0.0,
                    1.0 - cursor_ratio,
                    1.0,
                ))
                .dest([x_offset + scaled_cursor_x * scale, y_offset])
                .scale([scale, scale]),
        );
        canvas.finish(ctx)?;
        Ok(())
    }

    fn mouse_motion_event(
        &mut self,
        _ctx: &mut Context,
        x: f32,
        _y: f32,
        _dx: f32,
        _dy: f32,
    ) -> GameResult {
        self.cursor_x = x;
        Ok(())
    }

    fn mouse_button_down_event(
        &mut self,
        _ctx: &mut Context,
        button: MouseButton,
        _x: f32,
        _y: f32,
    ) -> GameResult {
        match button {
            MouseButton::Left => self.player.toggle_play_pause(),
            MouseButton::Right => self.player.next_frame(),
            MouseButton::Middle => self.player.previous_frame(),
            _ => {}
        }
        Ok(())
    }

    fn key_down_event(
        &mut self,
        _ctx: &mut Context,
        input: ggez::input::keyboard::KeyInput,
        _repeated: bool,
    ) -> GameResult {
        if let Some(keycode) = input.keycode {
            self.handle_input(keycode);
        }
        Ok(())
    }
}
