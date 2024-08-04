use anyhow::{Context as _, Result};
use ggez::Context;
use log::{debug, error, info};
use std::io::{BufRead, BufReader};

pub fn load_image_paths(ctx: &mut Context, dir: &str) -> Result<(Vec<(String, u64)>, usize)> {
    info!("Loading image paths from directory: {}", dir);
    let ffmpeg_input = format!("{}/input.txt", dir);

    debug!("Attempting to read file at: {}", ffmpeg_input);
    debug!("Current working directory: {:?}", std::env::current_dir()?);

    let file = ctx
        .fs
        .open(&ffmpeg_input)
        .context("Failed to open input file")?;
    let reader = BufReader::new(file);
    let mut images = Vec::new();
    let mut lines = reader.lines();

    while let (Some(Ok(file_path)), Some(Ok(duration_str))) = (lines.next(), lines.next()) {
        let file_path = file_path.trim_start_matches("file '").trim_end_matches("'");
        let duration = duration_str
            .trim_start_matches("duration ")
            .trim_end_matches("us")
            .parse::<u64>()
            .with_context(|| format!("Failed to parse duration '{}'", duration_str))?;

        debug!(
            "Loaded image path: {:?} with duration: {}",
            file_path, duration
        );
        images.push((file_path.to_string(), duration));
    }

    let frame_count = images.len();

    if frame_count == 0 {
        error!("No valid image paths were loaded from the input file");
        return Err(anyhow::anyhow!(
            "No valid image paths were loaded from the input file"
        ));
    }

    info!("Successfully loaded {} image paths", frame_count);
    Ok((images, frame_count))
}
