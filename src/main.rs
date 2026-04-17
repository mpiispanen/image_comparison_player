use clap::{Arg, ArgAction, Command};
use log::{error, info};
use std::time::{Duration, Instant};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
mod app;
mod batch_mode;
mod image_loader;
mod player;
mod test_images;

use crate::app::AppConfig;
use crate::batch_mode::{BatchConfig, VideoLayout};

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
        .arg(
            Arg::new("peek_zoom_factor")
                .long("peek-zoom-factor")
                .action(ArgAction::Set)
                .value_name("FACTOR")
                .help("Magnification factor for hold-to-peek zoom (Z key)")
                .default_value("4.0"),
        )
        .arg(
            Arg::new("batch_mode")
                .long("batch-mode")
                .action(ArgAction::SetTrue)
                .help("Run in headless batch mode (diff generation and/or video export)"),
        )
        .arg(
            Arg::new("batch_diff_output")
                .long("batch-diff-output")
                .action(ArgAction::Set)
                .value_name("DIR")
                .help("Output directory for batch diff images"),
        )
        .arg(
            Arg::new("video_output")
                .long("video-output")
                .action(ArgAction::Set)
                .value_name("FILE")
                .help("Output file path for generated video"),
        )
        .arg(
            Arg::new("video_layout")
                .long("video-layout")
                .action(ArgAction::Set)
                .value_name("LAYOUT")
                .help("Video panel layout: single | side-by-side | side-by-side-diff")
                .default_value("single"),
        )
        .arg(
            Arg::new("video_fps")
                .long("video-fps")
                .action(ArgAction::Set)
                .value_name("FPS")
                .help("Output video frame rate")
                .default_value("30"),
        )
        .arg(
            Arg::new("video_crf")
                .long("video-crf")
                .action(ArgAction::Set)
                .value_name("CRF")
                .help("Video quality CRF (0..51, lower is better quality)")
                .default_value("18"),
        )
        .arg(
            Arg::new("video_preset")
                .long("video-preset")
                .action(ArgAction::Set)
                .value_name("PRESET")
                .help("Video encoder preset (e.g. ultrafast, medium, veryslow)")
                .default_value("medium"),
        )
        .arg(
            Arg::new("video_codec")
                .long("video-codec")
                .action(ArgAction::Set)
                .value_name("CODEC")
                .help("Video codec for ffmpeg (e.g. libx264, libx265, mpeg4)")
                .default_value("libx264"),
        )
        .arg(
            Arg::new("video_pixel_format")
                .long("video-pixel-format")
                .action(ArgAction::Set)
                .value_name("PIX_FMT")
                .help("Video pixel format for ffmpeg (e.g. yuv420p)")
                .default_value("yuv420p"),
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
    let peek_zoom_factor: f32 = matches
        .get_one::<String>("peek_zoom_factor")
        .unwrap()
        .parse()
        .unwrap_or(4.0);
    let batch_mode = matches.get_flag("batch_mode");
    let batch_diff_output = matches.get_one::<String>("batch_diff_output").cloned();
    let video_output = matches.get_one::<String>("video_output").cloned();
    let video_layout = matches
        .get_one::<String>("video_layout")
        .unwrap()
        .parse::<VideoLayout>()
        .map_err(|e| format!("Invalid --video-layout: {}", e))?;
    let video_fps: f32 = matches
        .get_one::<String>("video_fps")
        .unwrap()
        .parse()
        .map_err(|_| "Invalid --video-fps value")?;
    let video_crf: u8 = matches
        .get_one::<String>("video_crf")
        .unwrap()
        .parse()
        .map_err(|_| "Invalid --video-crf value")?;
    let video_preset = matches.get_one::<String>("video_preset").unwrap().clone();
    let video_codec = matches.get_one::<String>("video_codec").unwrap().clone();
    let video_pixel_format = matches
        .get_one::<String>("video_pixel_format")
        .unwrap()
        .clone();

    if batch_mode {
        let left_images = resolve_batch_images(dir1.as_deref(), images1.as_deref(), fps)?;
        let right_images = if dir2.is_some() || images2.is_some() {
            Some(resolve_batch_images(
                dir2.as_deref(),
                images2.as_deref(),
                fps,
            )?)
        } else {
            None
        };
        let config = BatchConfig {
            left_images,
            right_images,
            diff_output_dir: batch_diff_output.map(std::path::PathBuf::from),
            video_output_path: video_output.map(std::path::PathBuf::from),
            video_layout,
            video_fps,
            video_crf,
            video_preset,
            video_codec,
            video_pixel_format,
        };
        batch_mode::run_batch_mode(config)?;
        return Ok(());
    }

    let (width, height) = parse_window_size(window_size)?;

    info!(
        "Starting image comparison player with input1: {}, input2: {}, window size: {}x{}",
        dir1.as_deref().unwrap_or_else(|| images1
            .as_ref()
            .and_then(|v| v.first().map(|s| s.as_str()))
            .unwrap_or("<none>")),
        dir2.as_deref().unwrap_or_else(|| images2
            .as_ref()
            .and_then(|v| v.first().map(|s| s.as_str()))
            .unwrap_or("<none>")),
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
        peek_zoom_factor,
    };

    let mut app_state = pollster::block_on(app::AppState::new(&window, app_config))?;

    let mut initialized = false;

    // Target ~60 fps. The event loop wakes on any window event (e.g. mouse
    // movement) so the split line responds immediately, while sleeping between
    // frames when idle to avoid burning CPU/GPU with an unthrottled busy loop.
    let target_frame_time = Duration::from_secs_f32(1.0 / 60.0);
    let mut next_frame_time = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::WaitUntil(next_frame_time);

        if initialized {
            app_state.handle_event(&window, &event);
        }

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
                initialized = true;
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

fn resolve_batch_images(
    dir: Option<&str>,
    images: Option<&[String]>,
    fps: f32,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let paths = if let Some(files) = images {
        image_loader::load_image_paths_from_files(files, fps)?.0
    } else if let Some(input_dir) = dir {
        image_loader::load_image_paths(input_dir, fps)?.0
    } else {
        return Err("Batch mode requires --dir1 or --images1".into());
    };
    Ok(paths.into_iter().map(|(path, _, _)| path).collect())
}

#[cfg(test)]
mod tests {
    use super::{parse_window_size, resolve_batch_images};

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

    #[test]
    fn test_resolve_batch_images_requires_input() {
        let result = resolve_batch_images(None, None, 30.0);
        assert!(result.is_err());
    }
}
