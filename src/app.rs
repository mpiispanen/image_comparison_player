use crate::image_loader;
use crate::player::Player;
use ggez::event::EventHandler;
use ggez::graphics::{self, DrawParam};
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
        let images1 = image_loader::load_images(ctx, &dir1).map_err(|e| {
            error!("Error loading images from dir1: {}", e);
            AppError(e)
        })?;
        let images2 = image_loader::load_images(ctx, &dir2).map_err(|e| {
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
        let (left_image, right_image) = self.player.current_images();

        // Draw left image
        canvas.draw(
            left_image,
            DrawParam::default().src(graphics::Rect::new(
                0.0,
                0.0,
                self.cursor_x,
                left_image.height() as f32,
            )),
        );

        // Draw right image
        canvas.draw(
            right_image,
            DrawParam::default().src(graphics::Rect::new(
                self.cursor_x,
                0.0,
                right_image.width() as f32 - self.cursor_x,
                right_image.height() as f32,
            )),
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
}
