use clap::{Arg, ArgAction, Command};
use log::{error, info};
use winit::{
    event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
mod app;
mod image_loader;
mod player;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let matches = Command::new("image_comparison_player")
        .version("1.0")
        .author("Matias Piispanen")
        .about("Compares images from two directories")
        .arg(
            Arg::new("dir1")
                .short('1')
                .long("dir1")
                .action(ArgAction::Set)
                .value_name("DIR")
                .help("First directory containing images")
                .required(true),
        )
        .arg(
            Arg::new("dir2")
                .short('2')
                .long("dir2")
                .action(ArgAction::Set)
                .value_name("DIR")
                .help("Second directory containing images")
                .required(true),
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
        .get_matches();

    let dir1 = matches.get_one::<String>("dir1").unwrap();
    let dir2 = matches.get_one::<String>("dir2").unwrap();
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

    let (width, height) = parse_window_size(window_size)?;

    info!(
        "Starting image comparison player with dir1: {}, dir2: {}, window size: {}x{}",
        dir1, dir2, width, height
    );

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Image Comparison Player")
        .with_inner_size(winit::dpi::LogicalSize::new(width, height))
        .build(&event_loop)?;

    let mut app_state = pollster::block_on(app::AppState::new(
        &window,
        dir1.clone(),
        dir2.clone(),
        cache_size,
        preload_ahead,
        preload_behind,
        num_load_threads,
        num_process_threads,
    ))?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(keycode),
                            ..
                        },
                    ..
                } => match keycode {
                    VirtualKeyCode::Space => {
                        app_state.toggle_play_pause();
                        window.request_redraw();
                    }
                    VirtualKeyCode::Right => {
                        app_state.next_frame();
                        window.request_redraw();
                    }
                    VirtualKeyCode::Left => {
                        app_state.previous_frame();
                        window.request_redraw();
                    }
                    VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                    _ => {}
                },
                WindowEvent::CursorMoved { position, .. } => {
                    app_state.update_cursor_position(position.x as f32, position.y as f32);
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    app_state.handle_mouse_click();
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                app_state.update();
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
