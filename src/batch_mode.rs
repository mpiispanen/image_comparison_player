use anyhow::{bail, Context, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, RgbaImage};
use nv_flip::{flip, magma_lut, FlipImageRgb8};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const EVEN_DIMENSIONS_PAD_FILTER: &str = "pad=ceil(iw/2)*2:ceil(ih/2)*2";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoLayout {
    Single,
    SideBySide,
    SideBySideWithDiff,
    SideBySideWithFlip,
}

impl std::str::FromStr for VideoLayout {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "single" => Ok(Self::Single),
            "side-by-side" => Ok(Self::SideBySide),
            "side-by-side-diff" => Ok(Self::SideBySideWithDiff),
            "side-by-side-flip" => Ok(Self::SideBySideWithFlip),
            _ => Err(format!(
                "Invalid video layout '{}'. Use one of: single, side-by-side, side-by-side-diff, side-by-side-flip",
                s
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchImageLayout {
    Diff,
    Flip,
    SideBySide,
    SideBySideWithDiff,
    SideBySideWithFlip,
}

impl BatchImageLayout {
    fn uses_abs_diff(self) -> bool {
        matches!(self, Self::Diff | Self::SideBySideWithDiff)
    }

    fn uses_flip(self) -> bool {
        matches!(self, Self::Flip | Self::SideBySideWithFlip)
    }
}

impl std::str::FromStr for BatchImageLayout {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "diff" => Ok(Self::Diff),
            "flip" => Ok(Self::Flip),
            "side-by-side" => Ok(Self::SideBySide),
            "side-by-side-diff" => Ok(Self::SideBySideWithDiff),
            "side-by-side-flip" => Ok(Self::SideBySideWithFlip),
            _ => Err(format!(
                "Invalid batch diff layout '{}'. Use one of: diff, flip, side-by-side, side-by-side-diff, side-by-side-flip",
                s
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub left_images: Vec<String>,
    pub right_images: Option<Vec<String>>,
    pub diff_output_dir: Option<PathBuf>,
    pub diff_output_layout: BatchImageLayout,
    pub video_output_path: Option<PathBuf>,
    pub video_layout: VideoLayout,
    pub video_fps: f32,
    pub video_crf: u8,
    pub video_preset: String,
    pub video_codec: String,
    pub video_pixel_format: String,
    pub use_existing_diffs: bool,
}

fn validate_batch_config(config: &BatchConfig) -> Result<()> {
    if config.video_output_path.is_some() {
        if config.video_crf > 51 {
            bail!("--video-crf must be in range 0..=51");
        }
        if !(config.video_fps.is_finite() && config.video_fps > 0.0) {
            bail!("--video-fps must be a positive number");
        }
    }
    if config.use_existing_diffs {
        if config.diff_output_dir.is_none() {
            bail!("--use-existing-diffs requires --batch-diff-output");
        }
        let video_uses_existing_diff = config.video_output_path.is_some()
            && config.video_layout == VideoLayout::SideBySideWithDiff;
        let image_output_uses_existing_diff = config.diff_output_dir.is_some()
            && config.diff_output_layout == BatchImageLayout::SideBySideWithDiff;
        if !video_uses_existing_diff && !image_output_uses_existing_diff {
            bail!(
                "--use-existing-diffs requires side-by-side-diff output via --video-layout and/or --batch-diff-layout"
            );
        }
    }

    Ok(())
}

pub fn run_batch_mode(config: BatchConfig) -> Result<()> {
    validate_batch_config(&config)?;
    if config.left_images.is_empty() {
        bail!("Batch mode requires at least one input image on the left side");
    }
    if config.diff_output_dir.is_none() && config.video_output_path.is_none() {
        bail!("Batch mode requested but neither --batch-diff-output nor --video-output is set");
    }

    let frame_count = config
        .right_images
        .as_ref()
        .map_or(config.left_images.len(), |r| {
            config.left_images.len().min(r.len())
        });
    if frame_count == 0 {
        bail!("No frames available for batch mode output");
    }

    let right_required =
        config.diff_output_dir.is_some() || config.video_layout != VideoLayout::Single;
    if right_required && config.right_images.as_ref().is_none_or(|r| r.is_empty()) {
        bail!(
            "Batch diff and side-by-side video layouts require a second input source (--dir2 or --images2)"
        );
    }

    let diff_dir = if let Some(dir) = &config.diff_output_dir {
        fs::create_dir_all(dir).with_context(|| format!("Failed to create {}", dir.display()))?;
        Some(dir.clone())
    } else {
        None
    };

    let temp_dir = if config.video_output_path.is_some() {
        let dir = tempfile::Builder::new()
            .prefix("icp_batch_frames_")
            .tempdir()
            .context("Failed to create temporary frame directory")?;
        Some(dir)
    } else {
        None
    };

    eprintln!(
        "[batch] Processing {} frame(s){}{}",
        frame_count,
        if diff_dir.is_some() {
            " with image output"
        } else {
            ""
        },
        if config.video_output_path.is_some() {
            " and video export"
        } else {
            ""
        }
    );

    let mut diff_progress = 0usize;
    let mut video_progress = 0usize;
    let needs_abs_diff_image = (diff_dir.is_some() && config.diff_output_layout.uses_abs_diff())
        || (config.video_output_path.is_some() && config.video_layout == VideoLayout::SideBySideWithDiff);
    let needs_flip_image = (diff_dir.is_some() && config.diff_output_layout.uses_flip())
        || (config.video_output_path.is_some() && config.video_layout == VideoLayout::SideBySideWithFlip);
    let report_diff_progress =
        (needs_abs_diff_image && !config.use_existing_diffs) || needs_flip_image;

    for frame in 0..frame_count {
        let left = image::open(&config.left_images[frame])
            .with_context(|| format!("Failed to open left image {}", config.left_images[frame]))?;
        let right = config.right_images.as_ref().map(|r| {
            image::open(&r[frame])
                .with_context(|| format!("Failed to open right image {}", r[frame]))
        });
        let right = right.transpose()?;

        let diff_path = diff_dir
            .as_ref()
            .map(|dir| dir.join(format!("diff_{:06}.png", frame)));
        let abs_diff = if needs_abs_diff_image {
            if config.use_existing_diffs {
                let path = diff_path
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Internal error: diff path is required"))?;
                Some(
                    image::open(path)
                        .with_context(|| {
                            format!(
                                "Failed to open precomputed diff {}. Disable --use-existing-diffs to regenerate",
                                path.display()
                            )
                        })?
                        .to_rgba8(),
                )
            } else {
                let r = right.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Internal error: right image is required for diffs")
                })?;
                Some(compute_abs_diff_image(&left, r)?)
            }
        } else {
            None
        };
        let flip_diff = if needs_flip_image {
            let r = right
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Internal error: right image is required for FLIP"))?;
            Some(compute_flip_diff_image(&left, r)?)
        } else {
            None
        };

        if let Some(dir) = &diff_dir {
            let path = dir.join(format!(
                "{}_{:06}.png",
                config.diff_output_layout.output_prefix(),
                frame
            ));
            match config.diff_output_layout {
                BatchImageLayout::Diff => {
                    let diff = abs_diff.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("Internal error: diff image is required for diff export")
                    })?;
                    diff.save(&path)
                        .with_context(|| format!("Failed to save {}", path.display()))?;
                }
                BatchImageLayout::Flip => {
                    let flip = flip_diff.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("Internal error: FLIP image is required for flip export")
                    })?;
                    flip.save(&path)
                        .with_context(|| format!("Failed to save {}", path.display()))?;
                }
                BatchImageLayout::SideBySide => {
                    let r = right.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("Internal error: right image is required for side-by-side")
                    })?;
                    stitch_panels(&[left.to_rgba8(), r.to_rgba8()])
                        .save(&path)
                        .with_context(|| format!("Failed to save {}", path.display()))?;
                }
                BatchImageLayout::SideBySideWithDiff => {
                    let r = right.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "Internal error: right image is required for side-by-side-diff"
                        )
                    })?;
                    let diff = abs_diff.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "Internal error: diff image is required for side-by-side-diff"
                        )
                    })?;
                    stitch_panels(&[left.to_rgba8(), r.to_rgba8(), diff.clone()])
                        .save(&path)
                        .with_context(|| format!("Failed to save {}", path.display()))?;
                }
                BatchImageLayout::SideBySideWithFlip => {
                    let r = right.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "Internal error: right image is required for side-by-side-flip"
                        )
                    })?;
                    let flip = flip_diff.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "Internal error: FLIP image is required for side-by-side-flip"
                        )
                    })?;
                    stitch_panels(&[left.to_rgba8(), r.to_rgba8(), flip.clone()])
                        .save(&path)
                        .with_context(|| format!("Failed to save {}", path.display()))?;
                }
            }
        }

        if report_diff_progress {
            diff_progress += 1;
            print_progress("Diff creation", diff_progress, frame_count);
        }

        if let Some(dir) = &temp_dir {
            let frame_img = match config.video_layout {
                VideoLayout::Single => left.to_rgba8(),
                VideoLayout::SideBySide => {
                    let r = right.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("Internal error: right image is required for side-by-side")
                    })?;
                    stitch_panels(&[left.to_rgba8(), r.to_rgba8()])
                }
                VideoLayout::SideBySideWithDiff => {
                    let r = right.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "Internal error: right image is required for side-by-side-diff"
                        )
                    })?;
                    let diff = abs_diff.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "Internal error: diff image is required for side-by-side-diff"
                        )
                    })?;
                    stitch_panels(&[left.to_rgba8(), r.to_rgba8(), diff.clone()])
                }
                VideoLayout::SideBySideWithFlip => {
                    let r = right.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "Internal error: right image is required for side-by-side-flip"
                        )
                    })?;
                    let flip = flip_diff.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "Internal error: FLIP image is required for side-by-side-flip"
                        )
                    })?;
                    stitch_panels(&[left.to_rgba8(), r.to_rgba8(), flip.clone()])
                }
            };
            let path = dir.path().join(format!("frame_{:06}.png", frame));
            frame_img
                .save(&path)
                .with_context(|| format!("Failed to save {}", path.display()))?;

            video_progress += 1;
            print_progress("Video frame creation", video_progress, frame_count);
        }
    }

    if let Some(video_path) = &config.video_output_path {
        let frames_dir = temp_dir
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("Internal error: temporary video frame directory missing")
            })?
            .path();
        create_video_from_frames(video_path, frames_dir, &config, frame_count)?;
    }

    eprintln!("[batch] Completed.");

    Ok(())
}

fn create_video_from_frames(
    video_path: &Path,
    frames_dir: &Path,
    config: &BatchConfig,
    frame_count: usize,
) -> Result<()> {
    if let Some(parent) = video_path.parent() {
        if !parent.as_os_str().is_empty() && parent != Path::new(".") {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
    }

    eprintln!(
        "[batch] Encoding {} frame(s) to video: {}",
        frame_count,
        video_path.display()
    );
    let pattern = frames_dir.join("frame_%06d.png");
    let status = build_ffmpeg_video_command(&pattern, video_path, config)
        .status()
        .context("Failed to launch ffmpeg. Ensure ffmpeg is installed and available in PATH")?;

    if !status.success() {
        bail!("ffmpeg exited with non-zero status: {}", status);
    }
    eprintln!("[batch] Video encoding complete: {}", video_path.display());
    Ok(())
}

fn build_ffmpeg_video_command(pattern: &Path, video_path: &Path, config: &BatchConfig) -> Command {
    let mut command = Command::new("ffmpeg");
    command
        .arg("-y")
        .arg("-framerate")
        .arg(config.video_fps.to_string())
        .arg("-i")
        .arg(pattern)
        .arg("-vf")
        .arg(EVEN_DIMENSIONS_PAD_FILTER)
        .arg("-c:v")
        .arg(&config.video_codec)
        .arg("-preset")
        .arg(&config.video_preset)
        .arg("-crf")
        .arg(config.video_crf.to_string())
        .arg("-pix_fmt")
        .arg(&config.video_pixel_format)
        .arg(video_path);
    command
}

fn print_progress(stage: &str, current: usize, total: usize) {
    if total == 0 {
        return;
    }

    let width = 28usize;
    let filled = ((current * width) / total).min(width);
    let percent = (current as f32 / total as f32) * 100.0;
    let bar = format!("{}{}", "#".repeat(filled), "-".repeat(width - filled));
    eprint!("\r[batch] {stage}: [{bar}] {current}/{total} ({percent:.1}%)");
    if let Err(err) = std::io::stderr().flush() {
        eprintln!("\n[batch] Warning: failed to flush progress output: {err}");
    }
    if current >= total {
        eprintln!();
    }
}

impl BatchImageLayout {
    fn output_prefix(self) -> &'static str {
        match self {
            Self::Diff => "diff",
            Self::Flip => "flip",
            Self::SideBySide => "side_by_side",
            Self::SideBySideWithDiff => "side_by_side_diff",
            Self::SideBySideWithFlip => "side_by_side_flip",
        }
    }
}

fn compute_abs_diff_image(left: &DynamicImage, right: &DynamicImage) -> Result<RgbaImage> {
    let (lw, lh) = left.dimensions();
    let (rw, rh) = right.dimensions();
    if lw == 0 || lh == 0 || rw == 0 || rh == 0 {
        bail!("Cannot diff zero-sized images");
    }

    let target_w = lw.min(rw);
    let target_h = lh.min(rh);

    let left = if lw == target_w && lh == target_h {
        left.to_rgba8()
    } else {
        image::imageops::resize(&left.to_rgba8(), target_w, target_h, FilterType::Triangle)
    };
    let right = if rw == target_w && rh == target_h {
        right.to_rgba8()
    } else {
        image::imageops::resize(&right.to_rgba8(), target_w, target_h, FilterType::Triangle)
    };

    let mut diff = ImageBuffer::new(target_w, target_h);
    for (x, y, pixel) in diff.enumerate_pixels_mut() {
        let l = left.get_pixel(x, y);
        let r = right.get_pixel(x, y);
        *pixel = image::Rgba([
            l[0].abs_diff(r[0]),
            l[1].abs_diff(r[1]),
            l[2].abs_diff(r[2]),
            255,
        ]);
    }
    Ok(diff)
}

fn compute_flip_diff_image(left: &DynamicImage, right: &DynamicImage) -> Result<RgbaImage> {
    let (lw, lh) = left.dimensions();
    let (rw, rh) = right.dimensions();
    if lw == 0 || lh == 0 || rw == 0 || rh == 0 {
        bail!("Cannot generate FLIP diff for zero-sized images");
    }

    let target_w = lw.min(rw);
    let target_h = lh.min(rh);

    let left = if lw == target_w && lh == target_h {
        left.to_rgba8()
    } else {
        image::imageops::resize(&left.to_rgba8(), target_w, target_h, FilterType::Triangle)
    };
    let right = if rw == target_w && rh == target_h {
        right.to_rgba8()
    } else {
        image::imageops::resize(&right.to_rgba8(), target_w, target_h, FilterType::Triangle)
    };

    let left_rgb = rgba_to_rgb(left.as_raw());
    let right_rgb = rgba_to_rgb(right.as_raw());
    let left_image = FlipImageRgb8::with_data(target_w, target_h, &left_rgb);
    let right_image = FlipImageRgb8::with_data(target_w, target_h, &right_rgb);
    let error_map = flip(left_image, right_image, nv_flip::DEFAULT_PIXELS_PER_DEGREE);
    let visualized = error_map.apply_color_lut(&magma_lut());
    let rgba = rgb_to_rgba(&visualized.to_vec());
    ImageBuffer::from_vec(target_w, target_h, rgba)
        .ok_or_else(|| anyhow::anyhow!("Failed to create FLIP output image buffer"))
}

fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    debug_assert_eq!(rgba.len() % 4, 0, "RGBA buffer length must be a multiple of 4");
    let mut rgb = Vec::with_capacity((rgba.len() / 4) * 3);
    for chunk in rgba.chunks_exact(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }
    rgb
}

fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    debug_assert_eq!(rgb.len() % 3, 0, "RGB buffer length must be a multiple of 3");
    let mut rgba = Vec::with_capacity((rgb.len() / 3) * 4);
    for chunk in rgb.chunks_exact(3) {
        rgba.push(chunk[0]);
        rgba.push(chunk[1]);
        rgba.push(chunk[2]);
        rgba.push(255);
    }
    rgba
}

fn stitch_panels(panels: &[RgbaImage]) -> RgbaImage {
    let total_width = panels.iter().map(|p| p.width()).sum::<u32>();
    let max_height = panels.iter().map(|p| p.height()).max().unwrap_or(0);
    let mut out = ImageBuffer::new(total_width, max_height);

    let mut x_offset = 0;
    for panel in panels {
        for y in 0..panel.height() {
            for x in 0..panel.width() {
                out.put_pixel(x + x_offset, y, *panel.get_pixel(x, y));
            }
        }
        x_offset += panel.width();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_video_layout() {
        assert_eq!(
            "single".parse::<VideoLayout>().unwrap(),
            VideoLayout::Single
        );
        assert_eq!(
            "side-by-side".parse::<VideoLayout>().unwrap(),
            VideoLayout::SideBySide
        );
        assert_eq!(
            "side-by-side-diff".parse::<VideoLayout>().unwrap(),
            VideoLayout::SideBySideWithDiff
        );
        assert_eq!(
            "side-by-side-flip".parse::<VideoLayout>().unwrap(),
            VideoLayout::SideBySideWithFlip
        );
        assert!("other".parse::<VideoLayout>().is_err());
    }

    #[test]
    fn parse_batch_image_layout() {
        assert_eq!(
            "diff".parse::<BatchImageLayout>().unwrap(),
            BatchImageLayout::Diff
        );
        assert_eq!(
            "flip".parse::<BatchImageLayout>().unwrap(),
            BatchImageLayout::Flip
        );
        assert_eq!(
            "side-by-side".parse::<BatchImageLayout>().unwrap(),
            BatchImageLayout::SideBySide
        );
        assert_eq!(
            "side-by-side-diff".parse::<BatchImageLayout>().unwrap(),
            BatchImageLayout::SideBySideWithDiff
        );
        assert_eq!(
            "side-by-side-flip".parse::<BatchImageLayout>().unwrap(),
            BatchImageLayout::SideBySideWithFlip
        );
        assert!("other".parse::<BatchImageLayout>().is_err());
    }

    #[test]
    fn diff_image_uses_absolute_difference() {
        let left = DynamicImage::ImageRgba8(ImageBuffer::from_fn(1, 1, |_x, _y| {
            image::Rgba([10, 100, 250, 255])
        }));
        let right = DynamicImage::ImageRgba8(ImageBuffer::from_fn(1, 1, |_x, _y| {
            image::Rgba([30, 40, 240, 255])
        }));
        let diff = compute_abs_diff_image(&left, &right).unwrap();
        assert_eq!(diff.get_pixel(0, 0).0, [20, 60, 10, 255]);
    }

    #[test]
    fn flip_image_is_generated_with_alpha() {
        let left = DynamicImage::ImageRgba8(ImageBuffer::from_fn(1, 1, |_x, _y| {
            image::Rgba([10, 100, 250, 255])
        }));
        let right = DynamicImage::ImageRgba8(ImageBuffer::from_fn(1, 1, |_x, _y| {
            image::Rgba([30, 40, 240, 255])
        }));
        let flip = compute_flip_diff_image(&left, &right).unwrap();
        assert_eq!(flip.dimensions(), (1, 1));
        assert_eq!(flip.get_pixel(0, 0).0[3], 255);
    }

    #[test]
    fn stitch_panels_concatenates_horizontally() {
        let red = ImageBuffer::from_pixel(1, 1, image::Rgba([255, 0, 0, 255]));
        let blue = ImageBuffer::from_pixel(1, 1, image::Rgba([0, 0, 255, 255]));
        let stitched = stitch_panels(&[red, blue]);
        assert_eq!(stitched.dimensions(), (2, 1));
        assert_eq!(stitched.get_pixel(0, 0).0, [255, 0, 0, 255]);
        assert_eq!(stitched.get_pixel(1, 0).0, [0, 0, 255, 255]);
    }

    #[test]
    fn batch_mode_requires_output_target() {
        let cfg = BatchConfig {
            left_images: vec!["a.png".to_string()],
            right_images: None,
            diff_output_dir: None,
            diff_output_layout: BatchImageLayout::Diff,
            video_output_path: None,
            video_layout: VideoLayout::Single,
            video_fps: 30.0,
            video_crf: 18,
            video_preset: "medium".to_string(),
            video_codec: "libx264".to_string(),
            video_pixel_format: "yuv420p".to_string(),
            use_existing_diffs: false,
        };
        assert!(run_batch_mode(cfg).is_err());
    }

    #[test]
    fn batch_mode_requires_right_side_for_diff() {
        let cfg = BatchConfig {
            left_images: vec!["a.png".to_string()],
            right_images: None,
            diff_output_dir: Some(std::path::PathBuf::from(
                "/tmp/will-not-be-created-before-validation",
            )),
            diff_output_layout: BatchImageLayout::Diff,
            video_output_path: None,
            video_layout: VideoLayout::Single,
            video_fps: 30.0,
            video_crf: 18,
            video_preset: "medium".to_string(),
            video_codec: "libx264".to_string(),
            video_pixel_format: "yuv420p".to_string(),
            use_existing_diffs: false,
        };
        assert!(run_batch_mode(cfg).is_err());
    }

    #[test]
    fn use_existing_diffs_requires_diff_dir() {
        let cfg = BatchConfig {
            left_images: vec!["a.png".to_string()],
            right_images: Some(vec!["b.png".to_string()]),
            diff_output_dir: None,
            diff_output_layout: BatchImageLayout::SideBySideWithDiff,
            video_output_path: None,
            video_layout: VideoLayout::Single,
            video_fps: 30.0,
            video_crf: 18,
            video_preset: "medium".to_string(),
            video_codec: "libx264".to_string(),
            video_pixel_format: "yuv420p".to_string(),
            use_existing_diffs: true,
        };
        assert!(validate_batch_config(&cfg).is_err());
    }

    #[test]
    fn use_existing_diffs_requires_side_by_side_diff_output() {
        let cfg = BatchConfig {
            left_images: vec!["a.png".to_string()],
            right_images: Some(vec!["b.png".to_string()]),
            diff_output_dir: Some(std::path::PathBuf::from("/tmp/diffs")),
            diff_output_layout: BatchImageLayout::Diff,
            video_output_path: None,
            video_layout: VideoLayout::Single,
            video_fps: 30.0,
            video_crf: 18,
            video_preset: "medium".to_string(),
            video_codec: "libx264".to_string(),
            video_pixel_format: "yuv420p".to_string(),
            use_existing_diffs: true,
        };
        assert!(validate_batch_config(&cfg).is_err());
    }

    #[test]
    fn use_existing_diffs_allows_image_side_by_side_diff_output() {
        let cfg = BatchConfig {
            left_images: vec!["a.png".to_string()],
            right_images: Some(vec!["b.png".to_string()]),
            diff_output_dir: Some(std::path::PathBuf::from("/tmp/diffs")),
            diff_output_layout: BatchImageLayout::SideBySideWithDiff,
            video_output_path: None,
            video_layout: VideoLayout::Single,
            video_fps: 30.0,
            video_crf: 18,
            video_preset: "medium".to_string(),
            video_codec: "libx264".to_string(),
            video_pixel_format: "yuv420p".to_string(),
            use_existing_diffs: true,
        };
        assert!(validate_batch_config(&cfg).is_ok());
    }

    fn make_test_image(color: image::Rgba<u8>, w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgba8(ImageBuffer::from_pixel(w, h, color))
    }

    fn save_test_image(img: &DynamicImage, path: &std::path::Path) {
        img.save(path).expect("failed to save test image");
    }

    #[test]
    fn batch_mode_flip_layout_produces_flip_output_file() {
        let tmp = tempfile::tempdir().unwrap();
        let left_path = tmp.path().join("left_000000.png");
        let right_path = tmp.path().join("right_000000.png");
        let diff_dir = tmp.path().join("diffs");

        let left_img = make_test_image(image::Rgba([100, 150, 200, 255]), 4, 4);
        let right_img = make_test_image(image::Rgba([110, 140, 190, 255]), 4, 4);
        save_test_image(&left_img, &left_path);
        save_test_image(&right_img, &right_path);

        let cfg = BatchConfig {
            left_images: vec![left_path.to_string_lossy().into_owned()],
            right_images: Some(vec![right_path.to_string_lossy().into_owned()]),
            diff_output_dir: Some(diff_dir.clone()),
            diff_output_layout: BatchImageLayout::Flip,
            video_output_path: None,
            video_layout: VideoLayout::Single,
            video_fps: 30.0,
            video_crf: 18,
            video_preset: "medium".to_string(),
            video_codec: "libx264".to_string(),
            video_pixel_format: "yuv420p".to_string(),
            use_existing_diffs: false,
        };
        run_batch_mode(cfg).unwrap();

        let out_path = diff_dir.join("flip_000000.png");
        assert!(out_path.exists(), "flip output file should exist");
        let out = image::open(&out_path).unwrap();
        assert_eq!(out.dimensions(), (4, 4));
        // The FLIP diff image should have full alpha (opaque).
        let rgba = out.to_rgba8();
        for pixel in rgba.pixels() {
            assert_eq!(pixel[3], 255, "all pixels should be fully opaque");
        }
    }

    #[test]
    fn batch_mode_side_by_side_flip_layout_produces_triple_width_output() {
        let tmp = tempfile::tempdir().unwrap();
        let left_path = tmp.path().join("left_000000.png");
        let right_path = tmp.path().join("right_000000.png");
        let diff_dir = tmp.path().join("diffs");

        let left_img = make_test_image(image::Rgba([200, 100, 50, 255]), 4, 4);
        let right_img = make_test_image(image::Rgba([180, 120, 60, 255]), 4, 4);
        save_test_image(&left_img, &left_path);
        save_test_image(&right_img, &right_path);

        let cfg = BatchConfig {
            left_images: vec![left_path.to_string_lossy().into_owned()],
            right_images: Some(vec![right_path.to_string_lossy().into_owned()]),
            diff_output_dir: Some(diff_dir.clone()),
            diff_output_layout: BatchImageLayout::SideBySideWithFlip,
            video_output_path: None,
            video_layout: VideoLayout::Single,
            video_fps: 30.0,
            video_crf: 18,
            video_preset: "medium".to_string(),
            video_codec: "libx264".to_string(),
            video_pixel_format: "yuv420p".to_string(),
            use_existing_diffs: false,
        };
        run_batch_mode(cfg).unwrap();

        let out_path = diff_dir.join("side_by_side_flip_000000.png");
        assert!(out_path.exists(), "side-by-side-flip output file should exist");
        let out = image::open(&out_path).unwrap();
        // Width should be 3x the individual frame width (left + right + flip).
        assert_eq!(
            out.dimensions(),
            (12, 4),
            "side-by-side-flip image should be 3x as wide as a single frame"
        );
    }

    #[test]
    fn batch_mode_video_side_by_side_flip_requires_right_images() {
        let cfg = BatchConfig {
            left_images: vec!["a.png".to_string()],
            right_images: None,
            diff_output_dir: None,
            diff_output_layout: BatchImageLayout::SideBySide,
            video_output_path: Some(std::env::temp_dir().join("test_will_not_be_created.mp4")),
            video_layout: VideoLayout::SideBySideWithFlip,
            video_fps: 30.0,
            video_crf: 18,
            video_preset: "medium".to_string(),
            video_codec: "libx264".to_string(),
            video_pixel_format: "yuv420p".to_string(),
            use_existing_diffs: false,
        };
        let err = run_batch_mode(cfg).unwrap_err();
        assert!(
            err.to_string().contains("second input source"),
            "expected right-side error, got: {err}"
        );
    }

    #[test]
    fn batch_mode_video_side_by_side_flip_layout_generates_frames() {
        let tmp = tempfile::tempdir().unwrap();
        let left_path = tmp.path().join("left_000000.png");
        let right_path = tmp.path().join("right_000000.png");
        let diff_dir = tmp.path().join("diffs");
        // Use image output as a proxy to verify the video-layout flip computation
        // path is exercised (flip is needed for both image and video output here).
        let left_img = make_test_image(image::Rgba([80, 160, 40, 255]), 4, 4);
        let right_img = make_test_image(image::Rgba([90, 150, 50, 255]), 4, 4);
        save_test_image(&left_img, &left_path);
        save_test_image(&right_img, &right_path);

        // Configure image output as side-by-side-flip AND video layout as
        // side-by-side-flip.  The flip diff will be required for both outputs,
        // so compute_flip_diff_image is exercised via the shared needs_flip_image
        // flag regardless of whether ffmpeg is available.
        let video_path = tmp.path().join("out.mp4");
        let cfg = BatchConfig {
            left_images: vec![left_path.to_string_lossy().into_owned()],
            right_images: Some(vec![right_path.to_string_lossy().into_owned()]),
            diff_output_dir: Some(diff_dir.clone()),
            diff_output_layout: BatchImageLayout::SideBySideWithFlip,
            video_output_path: Some(video_path.clone()),
            video_layout: VideoLayout::SideBySideWithFlip,
            video_fps: 30.0,
            video_crf: 18,
            video_preset: "medium".to_string(),
            video_codec: "libx264".to_string(),
            video_pixel_format: "yuv420p".to_string(),
            use_existing_diffs: false,
        };
        // Frame generation (including FLIP computation) always runs before ffmpeg.
        // The run may succeed (ffmpeg available) or fail at the encoding step only.
        let result = run_batch_mode(cfg);
        match result {
            Ok(()) => {
                // ffmpeg was available – full success.
                assert!(video_path.exists(), "video file should exist when ffmpeg succeeds");
            }
            Err(e) => {
                // ffmpeg was not available – that is acceptable in this environment.
                // The important check is that the error is about ffmpeg, not about
                // the FLIP computation or frame stitching.
                let msg = e.to_string();
                assert!(
                    msg.contains("ffmpeg"),
                    "expected ffmpeg error, got: {msg}"
                );
            }
        }
        // Image output (which doesn't need ffmpeg) must always be present.
        let out_path = diff_dir.join("side_by_side_flip_000000.png");
        assert!(out_path.exists(), "image output should always be created");
        let out = image::open(&out_path).unwrap();
        assert_eq!(out.dimensions(), (12, 4));
    }

    #[test]
    fn ffmpeg_video_command_pads_odd_frame_dimensions() {
        let config = BatchConfig {
            left_images: vec!["left.png".to_string()],
            right_images: None,
            diff_output_dir: None,
            diff_output_layout: BatchImageLayout::SideBySide,
            video_output_path: Some(PathBuf::from("out.mp4")),
            video_layout: VideoLayout::Single,
            video_fps: 30.0,
            video_crf: 18,
            video_preset: "medium".to_string(),
            video_codec: "libx264".to_string(),
            video_pixel_format: "yuv420p".to_string(),
            use_existing_diffs: false,
        };
        let command = build_ffmpeg_video_command(
            Path::new("/tmp/frames/frame_%06d.png"),
            Path::new("/tmp/out.mp4"),
            &config,
        );
        let args: Vec<_> = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();

        assert!(
            args.windows(2)
                .any(|window| window == ["-vf", EVEN_DIMENSIONS_PAD_FILTER]),
            "expected ffmpeg args to include even-dimension padding filter, got: {args:?}"
        );
    }
}
