use clap::{Arg, ArgAction, Command};
use ggez::event::{self};
use ggez::{ContextBuilder, GameResult};
use log::{error, info};
use std::path::PathBuf;
mod app;
mod image_loader;
mod player;

fn main() -> GameResult {
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
        .get_matches();

    let dir1 = matches.get_one::<String>("dir1").unwrap();
    let dir2 = matches.get_one::<String>("dir2").unwrap();
    let window_size = matches.get_one::<String>("window_size").unwrap();

    let (width, height) = parse_window_size(window_size).map_err(|e| {
        error!("Failed to parse window size: {}", e);
        ggez::GameError::ConfigError(e)
    })?;

    info!(
        "Starting image comparison player with dir1: {}, dir2: {}, window size: {}x{}",
        dir1, dir2, width, height
    );

    let (mut ctx, event_loop) = ContextBuilder::new("image_comparison_player", "Matias Piispanen")
        .add_resource_path(PathBuf::from("./resources"))
        .window_mode(ggez::conf::WindowMode::default().dimensions(width, height))
        .window_setup(ggez::conf::WindowSetup::default().title("Image Comparison Player"))
        .build()?;
    let app_state = app::AppState::new(&mut ctx, dir1.clone(), dir2.clone())?;
    event::run(ctx, event_loop, app_state)
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
