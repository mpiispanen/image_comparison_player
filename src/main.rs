use clap::{Arg, ArgAction, Command};
use log::{error, info};
use std::time::{Duration, Instant};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
mod app;
mod image_loader;
mod player;
mod test_images;

use crate::app::AppConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let matches = Command::new("image_comparison_player")
        .version("1.0")
        .author("Matias Piispanen")
        .about("Compares images from two directories or file lists")
        .arg(
            Arg::new("test_mode")
                .long("test-mode")
                .action(ArgAction::SetTrue)
                .conflicts_with_all(["dir1", "dir2", "images1", "images2"])
                .help("Run in testing mode with synthetically generated images (no real image files required)"),
        )
        .arg(
            Arg::new("dir1")
                .short('1')
                .long("dir1")
                .action(ArgAction::Set)
                .value_name("DIR")
                .help("First directory containing images (mutually exclusive with --images1)")
                .required(false)
                .conflicts_with("images1"),
        )
        .arg(
            Arg::new("dir2")
                .short('2')
                .long("dir2")
                .action(ArgAction::Set)
                .value_name("DIR")
                .help("Second directory containing images (mutually exclusive with --images2)")
                .required(false)
                .conflicts_with("images2"),
        )
        .arg(
            Arg::new("images1")
                .long("images1")
                .action(ArgAction::Append)
                .value_name("FILE")
                .help("One or more image files for the left side (use instead of --dir1)")
                .required(false)
                .num_args(1..),
        )
        .arg(
            Arg::new("images2")
                .long("images2")
                .action(ArgAction::Append)
                .value_name("FILE")
                .help("One or more image files for the right side (use instead of --dir2)")
                .required(false)
                .num_args(1..),
        )
        .arg(
            Arg::new("window_size")
                .short('w')
                .long("window-size")
                .action(ArgAction::Set)
                .value_name("WIDTHxHEIGHT")
                .help("Window size in format WIDTHxHEIGHT (e.g. 1920x1080)")
                .default_value("1920x1080"),
        )
        .arg(
            Arg::new("cache_size")
                .long("cache-size")
                .action(ArgAction::Set)
                .value_name("SIZE")
                .help("Size of the image cache")
                .default_value("50"),
        )
        .arg(
            Arg::new("preload_ahead")
                .long("preload-ahead")
                .action(ArgAction::Set)
                .value_name("COUNT")
                .help("Number of images to preload ahead")
                .default_value("25"),
        )
        .arg(
            Arg::new("preload_behind")
                .long("preload-behind")
                .action(ArgAction::Set)
                .value_name("COUNT")
                .help("Number of images to preload behind")
                .default_value("25"),
        )
        .arg(
            Arg::new("num_load_threads")
                .long("num-load-threads")
                .action(ArgAction::Set)
                .value_name("COUNT")
                .help("Number of threads to use for loading images")
                .default_value("4"),
        )
        .arg(
            Arg::new("num_process_threads")
                .long("num-process-threads")
                .action(ArgAction::Set)
                .value_name("COUNT")
                .help("Number of threads to use for processing images")
                .default_value("4"),
        )
        .arg(
            Arg::new("num_flip_diff_threads")
                .long("num-flip-diff-threads")
                .action(ArgAction::Set)
                .value_name("COUNT")
                .help("Number of threads to use for generating flip diffs")
                .default_value("4"),
        )
        .arg(
            Arg::new("diff_preload_ahead")
                .long("diff-preload-ahead")
                .action(ArgAction::Set)
                .value_name("COUNT")
                .help("Number of diff images to preload ahead")
                .default_value("5"),
        )
        .arg(
            Arg::new("diff_preload_behind")
                .long("diff-preload-behind")
                .action(ArgAction::Set)
                .value_name("COUNT")
                .help("Number of diff images to preload behind")
                .default_value("1"),
        )
        .arg(
            Arg::new("fps")
                .long("fps")
                .action(ArgAction::Set)
                .value_name("FPS")
                .help("Frames per second (overrides input.txt durations)")
                .default_value("30"),
        )
        .get_matches();

    let test_mode = matches.get_flag("test_mode");

    let (dir1, dir2, images1, images2) = if test_mode {
        info!("Test mode enabled -- generating synthetic test images");
        let (d1, d2) = test_images::generate_test_images()?;
        (
            Some(d1.to_string_lossy().into_owned()),
            Some(d2.to_string_lossy().into_owned()),
            None,
            None,
        )
    } else {
        let dir1 = matches.get_one::<String>("dir1").cloned();
        let dir2 = matches.get_one::<String>("dir2").cloned();
        let images1: Option<Vec<String>> = matches
            .get_many::<String>("images1")
            .map(|vals| vals.cloned().collect());
        let images2: Option<Vec<String>> = matches
            .get_many::<String>("images2")
            .map(|vals| vals.cloned().collect());

        (dir1, dir2, images1, images2)
    };

    let window_size = matches.get_one::<String>("window_size").unwrap();
    let cache_size = matches
        .get_one::<String>("cache_size")
        .unwrap()
        .parse()
        .unwrap_or(50);
    let preload_ahead = matches
        .get_one::<String>("preload_ahead")
        .unwrap()
        .parse()
        .unwrap_or(25);
    let preload_behind = matches
        .get_one::<String>("preload_behind")
        .unwrap()
        .parse()
        .unwrap_or(25);
    let num_load_threads = matches
        .get_one::<String>("num_load_threads")
        .unwrap()
        .parse()
        .unwrap_or(4);
    let num_process_threads = matches
        .get_one::<String>("num_process_threads")
        .unwrap()
        .parse()
        .unwrap_or(4);
    let num_flip_diff_threads = matches
        .get_one::<String>("num_flip_diff_threads")
        .unwrap()
        .parse()
        .unwrap_or(4);
    let diff_preload_ahead = matches
        .get_one::<String>("diff_preload_ahead")
        .unwrap()
        .parse()
        .unwrap_or(5);
    let diff_preload_behind = matches
        .get_one::<String>("diff_preload_behind")
        .unwrap()
        .parse()
        .unwrap_or(1);
    let fps: f32 = matches
        .get_one::<String>("fps")
        .unwrap()
        .parse()
        .unwrap_or(30.0);

    let (width, height) = parse_window_size(window_size)?;

    info!(
        "Starting image comparison player with input1: {}, input2: {}, window size: {}x{}",
        dir1.as_deref()
            .unwrap_or_else(|| images1.as_ref().and_then(|v| v.first().map(|s| s.as_str())).unwrap_or("<none>")),
        dir2.as_deref()
            .unwrap_or_else(|| images2.as_ref().and_then(|v| v.first().map(|s| s.as_str())).unwrap_or("<none>")),
        width,
        height
    );

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Image Comparison Player")
        .with_inner_size(winit::dpi::LogicalSize::new(width, height))
        .build(&event_loop)?;

    let app_config = AppConfig {
        dir1,
        dir2,
        images1,
        images2,
        cache_size,
        preload_ahead,
        preload_behind,
        num_load_threads,
        num_process_threads,
        num_flip_diff_threads,
        diff_preload_ahead,
        diff_preload_behind,
        fps,
    };

    let mut app_state = pollster::block_on(app::AppState::new(&window, app_config))?;

    // Target ~60 fps. The event loop wakes on any window event (e.g. mouse
    // movement) so the split line responds immediately, while sleeping between
    // frames when idle to avoid burning CPU/GPU with an unthrottled busy loop.
    let target_frame_time = Duration::from_secs_f32(1.0 / 60.0);
    let mut next_frame_time = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::WaitUntil(next_frame_time);

        app_state.handle_event(&window, &event);

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                app_state.update();
                next_frame_time = Instant::now() + target_frame_time;
                match app_state.render(&window) {
                    Ok(_) => {}
                    Err(e) => error!("Render error: {}", e),
                }
            }
            _ => {}
        }
    });
}

fn parse_window_size(size: &str) -> Result<(f32, f32), String> {
    let parts: Vec<&str> = size.split('x').collect();
    if parts.len() != 2 {
        return Err("Invalid window size format. Use WIDTHxHEIGHT".to_string());
    }
    let width = parts[0].parse::<f32>().map_err(|_| "Invalid width")?;
    let height = parts[1].parse::<f32>().map_err(|_| "Invalid height")?;
    Ok((width, height))
}

#[cfg(test)]
mod tests {
    use super::parse_window_size;

    #[test]
    fn test_parse_window_size_valid() {
        assert_eq!(parse_window_size("1920x1080"), Ok((1920.0, 1080.0)));
        assert_eq!(parse_window_size("800x600"), Ok((800.0, 600.0)));
        assert_eq!(parse_window_size("3840x2160"), Ok((3840.0, 2160.0)));
    }

    #[test]
    fn test_parse_window_size_missing_separator() {
        assert!(parse_window_size("1920").is_err());
        assert!(parse_window_size("1920 1080").is_err());
    }

    #[test]
    fn test_parse_window_size_missing_dimension() {
        assert!(parse_window_size("x1080").is_err());
        assert!(parse_window_size("1920x").is_err());
    }

    #[test]
    fn test_parse_window_size_non_numeric() {
        assert!(parse_window_size("WIDTHxHEIGHT").is_err());
        assert!(parse_window_size("1920xabc").is_err());
    }
}
