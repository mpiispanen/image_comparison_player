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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join("icp_tests").join(name);
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    // --- is_image_file tests ---

    #[test]
    fn test_is_image_file_valid_extensions() {
        assert!(is_image_file(Path::new("image.jpg")));
        assert!(is_image_file(Path::new("image.jpeg")));
        assert!(is_image_file(Path::new("image.png")));
        assert!(is_image_file(Path::new("image.bmp")));
    }

    #[test]
    fn test_is_image_file_invalid_extensions() {
        assert!(!is_image_file(Path::new("image.txt")));
        assert!(!is_image_file(Path::new("image.mp4")));
        assert!(!is_image_file(Path::new("image.gif")));
        assert!(!is_image_file(Path::new("image")));
    }

    #[test]
    fn test_is_image_file_case_insensitive() {
        assert!(is_image_file(Path::new("image.JPG")));
        assert!(is_image_file(Path::new("image.PNG")));
        assert!(is_image_file(Path::new("image.JPEG")));
        assert!(is_image_file(Path::new("image.BMP")));
    }

    // --- load_from_directory tests ---

    #[test]
    fn test_load_from_directory_empty() {
        let dir = TempDir::new("load_dir_empty");
        let result = load_from_directory(dir.path(), 30.0);
        assert!(result.is_ok());
        let (images, count) = result.unwrap();
        assert_eq!(count, 0);
        assert_eq!(images.len(), 0);
    }

    #[test]
    fn test_load_from_directory_counts_image_files() {
        let dir = TempDir::new("load_dir_count");
        fs::write(dir.path().join("frame_001.png"), b"").unwrap();
        fs::write(dir.path().join("frame_002.png"), b"").unwrap();
        fs::write(dir.path().join("frame_003.png"), b"").unwrap();

        let result = load_from_directory(dir.path(), 30.0);
        assert!(result.is_ok());
        let (images, count) = result.unwrap();
        assert_eq!(count, 3);
        assert_eq!(images.len(), 3);
    }

    #[test]
    fn test_load_from_directory_filters_non_image_files() {
        let dir = TempDir::new("load_dir_filter");
        fs::write(dir.path().join("image.png"), b"").unwrap();
        fs::write(dir.path().join("readme.txt"), b"").unwrap();
        fs::write(dir.path().join("video.mp4"), b"").unwrap();

        let result = load_from_directory(dir.path(), 30.0);
        assert!(result.is_ok());
        let (images, count) = result.unwrap();
        assert_eq!(count, 1);
        assert!(images[0].0.ends_with("image.png"));
    }

    #[test]
    fn test_load_from_directory_numeric_sorting() {
        let dir = TempDir::new("load_dir_sort");
        fs::write(dir.path().join("frame_10.png"), b"").unwrap();
        fs::write(dir.path().join("frame_2.png"), b"").unwrap();
        fs::write(dir.path().join("frame_1.png"), b"").unwrap();

        let result = load_from_directory(dir.path(), 30.0);
        assert!(result.is_ok());
        let (images, count) = result.unwrap();
        assert_eq!(count, 3);
        assert!(images[0].0.ends_with("frame_1.png"));
        assert!(images[1].0.ends_with("frame_2.png"));
        assert!(images[2].0.ends_with("frame_10.png"));
    }

    #[test]
    fn test_load_from_directory_frame_timing() {
        let dir = TempDir::new("load_dir_timing");
        fs::write(dir.path().join("frame_1.png"), b"").unwrap();
        fs::write(dir.path().join("frame_2.png"), b"").unwrap();

        let fps = 30.0_f32;
        let frame_duration = (1_000_000.0 / fps) as u64;

        let result = load_from_directory(dir.path(), fps);
        assert!(result.is_ok());
        let (images, _) = result.unwrap();
        assert_eq!(images[0].1, 0);
        assert_eq!(images[0].2, frame_duration);
        assert_eq!(images[1].1, frame_duration);
        assert_eq!(images[1].2, 2 * frame_duration);
    }

    // --- load_from_input_txt tests ---

    #[test]
    fn test_load_from_input_txt_parses_file_and_duration() {
        let dir = TempDir::new("load_input_txt_basic");
        let input_txt = dir.path().join("input.txt");
        fs::write(
            &input_txt,
            "file 'frame_0001.png'\nduration 33333us\nfile 'frame_0002.png'\nduration 33333us\n",
        )
        .unwrap();

        let result = load_from_input_txt(&input_txt, 30.0);
        assert!(result.is_ok());
        let (images, count) = result.unwrap();
        assert_eq!(count, 2);
        assert_eq!(images.len(), 2);
    }

    #[test]
    fn test_load_from_input_txt_cumulative_timing() {
        let dir = TempDir::new("load_input_txt_timing");
        let input_txt = dir.path().join("input.txt");
        fs::write(
            &input_txt,
            "file 'frame_0001.png'\nduration 40000us\nfile 'frame_0002.png'\nduration 40000us\n",
        )
        .unwrap();

        let result = load_from_input_txt(&input_txt, 30.0);
        assert!(result.is_ok());
        let (images, _) = result.unwrap();
        // First frame: [0, 40000)
        assert_eq!(images[0].1, 0);
        assert_eq!(images[0].2, 40000);
        // Second frame: [40000, 80000)
        assert_eq!(images[1].1, 40000);
        assert_eq!(images[1].2, 80000);
    }

    #[test]
    fn test_load_from_input_txt_respects_fps_override() {
        let dir = TempDir::new("load_input_txt_fps");
        let input_txt = dir.path().join("input.txt");
        fs::write(
            &input_txt,
            "file 'frame_0001.png'\nduration 33333us\nfile 'frame_0002.png'\nduration 33333us\n",
        )
        .unwrap();

        // When fps != 30.0, duration from file is ignored and frame_duration is used instead
        let fps = 24.0_f32;
        let frame_duration = (1_000_000.0 / fps) as u64;

        let result = load_from_input_txt(&input_txt, fps);
        assert!(result.is_ok());
        let (images, _) = result.unwrap();
        assert_eq!(images[0].2 - images[0].1, frame_duration);
        assert_eq!(images[1].2 - images[1].1, frame_duration);
    }

    // --- load_image_paths integration tests ---

    #[test]
    fn test_load_image_paths_uses_directory_when_no_input_txt() {
        let dir = TempDir::new("load_paths_dir");
        fs::write(dir.path().join("frame_001.png"), b"").unwrap();
        fs::write(dir.path().join("frame_002.png"), b"").unwrap();

        let result = load_image_paths(dir.path().to_str().unwrap(), 30.0);
        assert!(result.is_ok());
        let (images, count) = result.unwrap();
        assert_eq!(count, 2);
        assert_eq!(images.len(), 2);
    }

    #[test]
    fn test_load_image_paths_uses_input_txt_when_present() {
        let dir = TempDir::new("load_paths_input_txt");
        let input_txt = dir.path().join("input.txt");
        fs::write(
            &input_txt,
            "file 'frame_0001.png'\nduration 33333us\nfile 'frame_0002.png'\nduration 33333us\nfile 'frame_0003.png'\nduration 33333us\n",
        )
        .unwrap();

        let result = load_image_paths(dir.path().to_str().unwrap(), 30.0);
        assert!(result.is_ok());
        let (images, count) = result.unwrap();
        assert_eq!(count, 3);
        assert_eq!(images.len(), 3);
    }
}
