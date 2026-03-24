use anyhow::{Context as _, Result};
use log::{debug, info, warn};
use regex::Regex;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

type ImageInfo = (String, u64, u64);

pub fn load_image_paths_from_files(files: &[String], fps: f32) -> Result<(Vec<ImageInfo>, usize)> {
    info!("Loading image paths from file list ({} files)", files.len());
    let frame_duration = (1_000_000.0 / fps) as u64;
    let mut images = Vec::new();
    let mut cumulative_duration = 0u64;

    for file in files {
        let path = std::fs::canonicalize(file)
            .with_context(|| format!("Failed to resolve path '{}'", file))?;
        if !path.is_file() {
            return Err(anyhow::anyhow!("Path is not a file: {}", file));
        }
        if !is_image_file(&path) {
            return Err(anyhow::anyhow!("Not a supported image file: {}", file));
        }
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

pub fn load_image_paths(dir: &str, fps: f32) -> Result<(Vec<ImageInfo>, usize)> {
    info!("Loading image paths from directory: {}", dir);
    let absolute_dir = std::fs::canonicalize(dir)
        .with_context(|| format!("Directory not found: '{}'", dir))?;
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
        // Common formats
        "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp"
            // TIFF
            | "tiff" | "tif"
            // PNM / Netpbm
            | "ppm" | "pbm" | "pgm" | "pam"
            // Other formats supported by the image crate
            | "dds" | "exr" | "ff" | "hdr" | "ico" | "qoi" | "tga"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

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

    fn create_test_image(path: &std::path::Path) {
        let mut f = fs::File::create(path).unwrap();
        // Minimal valid 1x1 PNG
        f.write_all(&[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR length + type
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, // 8-bit RGB, CRC
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT length + type
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, // IDAT data
            0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC, // CRC
            0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, // IEND length + type
            0x44, 0xAE, 0x42, 0x60, 0x82, // IEND CRC
        ])
        .unwrap();
    }

    fn create_test_ppm(path: &std::path::Path) {
        fs::write(path, b"P3\n1 1\n255\n255 0 0\n").unwrap();
    }

    fn write_tmp_input_txt(dir: &std::path::Path, content: &str) {
        let path = dir.join("input.txt");
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    // ── load_from_input_txt ────────────────────────────────────────────────

    #[test]
    fn test_load_from_input_txt_basic() {
        let dir = tempfile::tempdir().unwrap();
        // Note: We don't need real image files; load_from_input_txt only records paths and doesn't validate file existence.
        let content = "\
file 'a.png'\n\
duration 33333us\n\
file 'b.png'\n\
duration 33333us\n";
        write_tmp_input_txt(dir.path(), content);

        let input = dir.path().join("input.txt");
        let (images, count) = load_from_input_txt(&input, 30.0).unwrap();
        assert_eq!(count, 2);
        assert_eq!(images.len(), 2);
        // Verify cumulative timing
        assert_eq!(images[0].1, 0);
        assert_eq!(images[0].2, 33333);
        assert_eq!(images[1].1, 33333);
        assert_eq!(images[1].2, 66666);
    }

    // --- load_image_paths_from_files tests ---

    #[test]
    fn test_load_image_paths_from_files_single() {
        let dir = TempDir::new("from_files_single");
        let path = dir.path().join("frame1.png");
        create_test_image(&path);
        let path_str = path.to_string_lossy().into_owned();
        let (images, count) = load_image_paths_from_files(&[path_str], 30.0).unwrap();
        assert_eq!(count, 1);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].1, 0); // start time
        assert!(images[0].2 > 0); // end time
    }

    #[test]
    fn test_load_image_paths_from_files_multiple() {
        let dir = TempDir::new("from_files_multiple");
        let p1 = dir.path().join("a.png");
        let p2 = dir.path().join("b.png");
        let p3 = dir.path().join("c.png");
        create_test_image(&p1);
        create_test_image(&p2);
        create_test_image(&p3);
        let files = vec![
            p1.to_string_lossy().into_owned(),
            p2.to_string_lossy().into_owned(),
            p3.to_string_lossy().into_owned(),
        ];
        let (images, count) = load_image_paths_from_files(&files, 30.0).unwrap();
        assert_eq!(count, 3);
        assert_eq!(images.len(), 3);
        // Verify timestamps are cumulative
        assert_eq!(images[0].1, 0);
        assert_eq!(images[0].2, images[1].1);
        assert_eq!(images[1].2, images[2].1);
    }

    #[test]
    fn test_load_image_paths_from_files_accepts_ppm_and_pgm() {
        let dir = TempDir::new("from_files_ppm_pgm");
        let ppm = dir.path().join("a.ppm");
        let pgm = dir.path().join("b.pgm");
        create_test_ppm(&ppm);
        fs::write(&pgm, b"P2\n1 1\n255\n127\n").unwrap();

        let files = vec![
            ppm.to_string_lossy().into_owned(),
            pgm.to_string_lossy().into_owned(),
        ];
        let (images, count) = load_image_paths_from_files(&files, 30.0).unwrap();
        assert_eq!(count, 2);
        assert_eq!(images.len(), 2);
    }

    #[test]
    fn test_load_image_paths_from_files_empty() {
        let (images, count) = load_image_paths_from_files(&[], 30.0).unwrap();
        assert_eq!(count, 0);
        assert!(images.is_empty());
    }

    #[test]
    fn test_load_image_paths_from_files_non_existent() {
        let result =
            load_image_paths_from_files(&["/nonexistent/path/img.png".to_string()], 30.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_image_paths_from_files_non_image() {
        let dir = TempDir::new("from_files_non_image");
        let path = dir.path().join("file.txt");
        fs::write(&path, "hello").unwrap();
        let result = load_image_paths_from_files(&[path.to_string_lossy().into_owned()], 30.0);
        assert!(result.is_err());
    }

    // --- is_image_file tests ---

    #[test]
    fn test_is_image_file_valid_extensions() {
        assert!(is_image_file(Path::new("image.jpg")));
        assert!(is_image_file(Path::new("image.jpeg")));
        assert!(is_image_file(Path::new("image.png")));
        assert!(is_image_file(Path::new("image.bmp")));
        assert!(is_image_file(Path::new("image.gif")));
        assert!(is_image_file(Path::new("image.tiff")));
        assert!(is_image_file(Path::new("image.tif")));
        assert!(is_image_file(Path::new("image.webp")));
        assert!(is_image_file(Path::new("image.ico")));
        assert!(is_image_file(Path::new("image.hdr")));
        assert!(is_image_file(Path::new("image.ppm")));
        assert!(is_image_file(Path::new("image.pbm")));
        assert!(is_image_file(Path::new("image.pgm")));
        assert!(is_image_file(Path::new("image.pam")));
        assert!(is_image_file(Path::new("image.tga")));
        assert!(is_image_file(Path::new("image.ff")));
        assert!(is_image_file(Path::new("image.exr")));
        assert!(is_image_file(Path::new("image.qoi")));
        assert!(is_image_file(Path::new("image.dds")));
    }

    #[test]
    fn test_is_image_file_invalid_extensions() {
        assert!(!is_image_file(Path::new("image.txt")));
        assert!(!is_image_file(Path::new("image.mp4")));
        assert!(!is_image_file(Path::new("image")));
    }

    #[test]
    fn test_is_image_file_case_insensitive() {
        assert!(is_image_file(Path::new("image.JPG")));
        assert!(is_image_file(Path::new("image.PNG")));
        assert!(is_image_file(Path::new("image.JPEG")));
        assert!(is_image_file(Path::new("image.BMP")));
        assert!(is_image_file(Path::new("image.PPM")));
        assert!(is_image_file(Path::new("image.TIFF")));
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
