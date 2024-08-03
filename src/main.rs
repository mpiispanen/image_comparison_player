use clap::{Arg, ArgAction, Command};
use ggez::event::{self};
use ggez::{ContextBuilder, GameResult};
use log::info;
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
        .get_matches();

    let dir1 = matches.get_one::<String>("dir1").unwrap();
    let dir2 = matches.get_one::<String>("dir2").unwrap();

    info!(
        "Starting image comparison player with dir1: {} and dir2: {}",
        dir1, dir2
    );

    let (mut ctx, event_loop) = ContextBuilder::new("image_comparison_player", "Matias Piispanen")
        .add_resource_path(PathBuf::from("./resources"))
        .window_mode(ggez::conf::WindowMode::default().dimensions(1920.0, 1080.0))
        .build()?;
    let app_state = app::AppState::new(&mut ctx, dir1.clone(), dir2.clone())?;
    event::run(ctx, event_loop, app_state)
}
