use anyhow::{Context as _, Result};
use log::{debug, info, warn};
use regex::Regex;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

type ImageInfo = (String, u64, u64);

pub fn load_image_paths(dir: &str, fps: f32) -> Result<(Vec<ImageInfo>, usize)> {
    info!("Loading image paths from directory: {}", dir);
    let absolute_dir = std::fs::canonicalize(dir)?;
    let ffmpeg_input = absolute_dir.join("input.txt");

    if ffmpeg_input.exists() {
        load_from_input_txt(&ffmpeg_input, fps)
    } else {
        warn!("input.txt not found, searching for image files");
        load_from_directory(&absolute_dir, fps)
    }
}

fn load_from_input_txt(ffmpeg_input: &Path, fps: f32) -> Result<(Vec<ImageInfo>, usize)> {
    debug!("Attempting to open file: {:?}", &ffmpeg_input);
    let file = File::open(ffmpeg_input).context("Failed to open input file")?;
    let reader = BufReader::new(file);
    let mut images = Vec::new();
    let mut lines = reader.lines();
    let mut cumulative_duration = 0;
    let frame_duration = (1_000_000.0 / fps) as u64;

    let current_dir = std::env::current_dir().context("Failed to get current directory")?;
    let input_dir = ffmpeg_input.parent().unwrap_or(Path::new(""));

    while let (Some(Ok(file_path)), Some(Ok(duration_str))) = (lines.next(), lines.next()) {
        let file_path = file_path
            .trim_start_matches("file '")
            .trim_end_matches('\'');

        let duration = if fps != 30.0 {
            frame_duration
        } else {
            duration_str
                .trim_start_matches("duration ")
                .trim_end_matches("us")
                .parse::<u64>()
                .with_context(|| format!("Failed to parse duration '{}'", duration_str))?
        };

        let full_path = input_dir.join(file_path);
        let relative_path = full_path
            .strip_prefix(&current_dir)
            .unwrap_or(&full_path)
            .to_path_buf();

        cumulative_duration += duration;
        images.push((
            relative_path.to_string_lossy().into_owned(),
            cumulative_duration - duration,
            cumulative_duration,
        ));
    }

    let frame_count = images.len();
    Ok((images, frame_count))
}

fn load_from_directory(dir: &Path, fps: f32) -> Result<(Vec<ImageInfo>, usize)> {
    let frame_duration = (1_000_000.0 / fps) as u64;
    let mut images = Vec::new();
    let mut cumulative_duration = 0;

    let re = Regex::new(r"(\d+)").unwrap();
    let mut image_files: Vec<_> = fs::read_dir(dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_file() && is_image_file(&path) {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    image_files.sort_by_key(|path| {
        re.captures(path.file_name().unwrap().to_str().unwrap())
            .and_then(|cap| cap.get(1))
            .and_then(|m| m.as_str().parse::<usize>().ok())
            .unwrap_or(0)
    });

    for path in image_files {
        cumulative_duration += frame_duration;
        images.push((
            path.to_string_lossy().into_owned(),
            cumulative_duration - frame_duration,
            cumulative_duration,
        ));
    }

    let frame_count = images.len();
    Ok((images, frame_count))
}

fn is_image_file(path: &Path) -> bool {
    let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    matches!(
        extension.to_lowercase().as_str(),
        "jpg" | "jpeg" | "png" | "bmp"
    )
}
