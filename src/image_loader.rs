use anyhow::{Context as _, Result};
use ggez::graphics::Image;
use ggez::Context;
use log::{debug, error, info};
use std::io::Read;
use std::path::Path;

pub fn load_images(ctx: &mut Context, dir: &str) -> Result<Vec<(Image, u64)>> {
    info!("Loading images from directory: {}", dir);
    let path = Path::new(dir);
    let ffmpeg_input = path.join("input.txt");

    debug!("Attempting to read file at: {:?}", ffmpeg_input);
    debug!("Current working directory: {:?}", std::env::current_dir()?);

    let mut f = ctx.fs.open(&ffmpeg_input)?;
    let mut input_content = String::new();
    f.read_to_string(&mut input_content)?;

    let lines: Vec<&str> = input_content.lines().collect();

    let mut images = Vec::new();
    for (index, chunk) in lines.chunks(2).enumerate() {
        if chunk.len() == 2 {
            let file_path = chunk[0].trim_start_matches("file '").trim_end_matches("'");
            let duration = chunk[1]
                .trim_start_matches("duration ")
                .trim_end_matches("us")
                .parse::<u64>()
                .with_context(|| {
                    error!("Failed to parse duration '{}' at line {}", chunk[1], index * 2 + 2);
                    format!("Failed to parse duration '{}' at line {}. Ensure the duration is a valid unsigned integer.", chunk[1], index * 2 + 2)
                })?;

            let relative_to_root = "resources".to_string() + file_path;
            let relative_to_root_file = Path::new(&relative_to_root);
            let image = {
                let mut file = ctx.fs.open(&file_path)?;
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes)?;
                Image::from_bytes(ctx, &bytes)
                    .with_context(|| {
                        let metadata = std::fs::metadata(&relative_to_root_file);
                        let exists = relative_to_root_file.exists();
                        let permissions = metadata.as_ref().map(|m| m.permissions());
                        let error_msg = format!(
                            "Failed to load image at {:?}. File exists: {}, Metadata: {:?}, Permissions: {:?}. Error: {}",
                            relative_to_root, exists, metadata, permissions,
                            match Image::from_bytes(ctx, &bytes) {
                                Ok(_) => "No error (unexpected)".to_string(),
                                Err(e) => e.to_string(),
                            }
                        );
                        log::error!("{}", error_msg);
                        error_msg
                    })?
            };

            debug!("Loaded image: {:?} with duration: {}", file_path, duration);
            images.push((image, duration));
        } else {
            error!("Invalid input format at line {}", index * 2 + 1);
            return Err(anyhow::anyhow!("Invalid input format at line {}. Expected file path and duration, but found incomplete data.", index * 2 + 1));
        }
    }

    if images.is_empty() {
        error!("No valid images were loaded from the input file");
        return Err(anyhow::anyhow!("No valid images were loaded from the input file. Ensure the input file is not empty and contains valid data."));
    }

    info!("Successfully loaded {} images", images.len());
    Ok(images)
}
