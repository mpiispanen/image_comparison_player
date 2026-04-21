use anyhow::{bail, Context, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, RgbaImage};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoLayout {
    Single,
    SideBySide,
    SideBySideWithDiff,
}

impl std::str::FromStr for VideoLayout {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "single" => Ok(Self::Single),
            "side-by-side" => Ok(Self::SideBySide),
            "side-by-side-diff" => Ok(Self::SideBySideWithDiff),
            _ => Err(format!(
                "Invalid video layout '{}'. Use one of: single, side-by-side, side-by-side-diff",
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
        if config.video_output_path.is_none() {
            bail!("--use-existing-diffs requires --video-output");
        }
        if config.diff_output_dir.is_none() {
            bail!("--use-existing-diffs requires --batch-diff-output");
        }
        if config.video_layout != VideoLayout::SideBySideWithDiff {
            bail!("--use-existing-diffs requires --video-layout side-by-side-diff");
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
            " with diff output"
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
    let needs_diff_image =
        diff_dir.is_some() || config.video_layout == VideoLayout::SideBySideWithDiff;
    let report_diff_progress = needs_diff_image && !config.use_existing_diffs;

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
        let diff = if needs_diff_image {
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

        if let Some(path) = &diff_path {
            if !config.use_existing_diffs {
                let diff = diff.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Internal error: diff image is required for diff export")
                })?;
                diff.save(path)
                    .with_context(|| format!("Failed to save {}", path.display()))?;
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
                    let diff = diff.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "Internal error: diff image is required for side-by-side-diff"
                        )
                    })?;
                    stitch_panels(&[left.to_rgba8(), r.to_rgba8(), diff.clone()])
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
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-framerate")
        .arg(config.video_fps.to_string())
        .arg("-i")
        .arg(pattern)
        .arg("-c:v")
        .arg(&config.video_codec)
        .arg("-preset")
        .arg(&config.video_preset)
        .arg("-crf")
        .arg(config.video_crf.to_string())
        .arg("-pix_fmt")
        .arg(&config.video_pixel_format)
        .arg(video_path)
        .status()
        .context("Failed to launch ffmpeg. Ensure ffmpeg is installed and available in PATH")?;

    if !status.success() {
        bail!("ffmpeg exited with non-zero status: {}", status);
    }
    eprintln!("[batch] Video encoding complete: {}", video_path.display());
    Ok(())
}

fn print_progress(stage: &str, current: usize, total: usize) {
    if total == 0 {
        return;
    }

    let width = 28usize;
    let filled = ((current * width) / total).min(width);
    let percent = (current as f32 / total as f32) * 100.0;
    let bar = format!("{}{}", "#".repeat(filled), "-".repeat(width - filled));
    eprint!(
        "\r[batch] {stage}: [{bar}] {current}/{total} ({percent:.1}%)"
    );
    if let Err(err) = std::io::stderr().flush() {
        eprintln!("\n[batch] Warning: failed to flush progress output: {err}");
    }
    if current >= total {
        eprintln!();
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
        assert!("other".parse::<VideoLayout>().is_err());
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
            video_output_path: Some(std::path::PathBuf::from("out.mp4")),
            video_layout: VideoLayout::SideBySideWithDiff,
            video_fps: 30.0,
            video_crf: 18,
            video_preset: "medium".to_string(),
            video_codec: "libx264".to_string(),
            video_pixel_format: "yuv420p".to_string(),
            use_existing_diffs: true,
        };
        assert!(run_batch_mode(cfg).is_err());
    }
}
