use anyhow::{Context as _, Result};
use log::{debug, info};
use std::fs::File;
use std::io::{BufRead, BufReader};

type ImageInfo = (String, u64, u64);

pub fn load_image_paths(dir: &str) -> Result<(Vec<ImageInfo>, usize)> {
    info!("Loading image paths from directory: {}", dir);
    let absolute_dir = std::fs::canonicalize(dir)?;
    let ffmpeg_input = absolute_dir.join("input.txt");

    debug!("Attempting to read file at: {:?}", ffmpeg_input);
    debug!("Current working directory: {:?}", std::env::current_dir()?);

    debug!("Attempting to open file: {:?}", &ffmpeg_input);
    let file = File::open(&ffmpeg_input).context("Failed to open input file")?;
    let reader = BufReader::new(file);
    let mut images = Vec::new();
    let mut lines = reader.lines();
    let mut cumulative_duration = 0;

    while let (Some(Ok(file_path)), Some(Ok(duration_str))) = (lines.next(), lines.next()) {
        let file_path = file_path
            .trim_start_matches("file '")
            .trim_end_matches('\'');

        let duration = duration_str
            .trim_start_matches("duration ")
            .trim_end_matches("us")
            .parse::<u64>()
            .with_context(|| format!("Failed to parse duration '{}'", duration_str))?;

        let full_path = absolute_dir.join(file_path);

        cumulative_duration += duration;
        images.push((
            full_path.to_string_lossy().into_owned(),
            cumulative_duration - duration,
            cumulative_duration,
        ));
    }

    let frame_count = images.len();
    Ok((images, frame_count))
}
