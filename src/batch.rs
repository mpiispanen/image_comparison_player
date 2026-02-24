use anyhow::{Context, Result};
use image::GenericImageView;
use log::{info, warn};
use nv_flip::{flip, magma_lut, FlipImageRgb8, FlipPool};
use std::path::Path;

use crate::image_loader;

/// Diff computation mode for batch processing.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffMode {
    /// Compute FLIP perceptual difference and visualise with the magma LUT.
    Flip,
    /// Skip diff computation; only validate that images can be loaded.
    None,
}

impl std::str::FromStr for DiffMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "flip" => Ok(DiffMode::Flip),
            "none" => Ok(DiffMode::None),
            other => Err(format!(
                "Unknown diff mode '{}'. Valid values are: flip, none",
                other
            )),
        }
    }
}

/// Configuration for a batch processing run.
pub struct BatchConfig {
    /// Ordered pairs of (left_path, right_path) to compare.
    pub image_pairs: Vec<(String, String)>,
    /// Directory where output images and metrics will be written.
    pub output_dir: String,
    /// Which diff algorithm to apply.
    pub diff_mode: DiffMode,
}

/// Per-frame result produced during batch processing.
pub struct FrameResult {
    pub index: usize,
    pub left_path: String,
    pub right_path: String,
    pub flip_mean: Option<f32>,
    pub flip_min: Option<f32>,
    pub flip_max: Option<f32>,
    pub flip_p95: Option<f32>,
    pub flip_p99: Option<f32>,
    /// Path to the saved diff PNG, if one was written.
    pub output_path: Option<String>,
}

/// Build image pairs from the same source descriptors used by the GUI.
///
/// Loads paths from `dir1`/`dir2` or `images1`/`images2` and zips them into
/// `(left, right)` pairs up to the shorter list's length.
pub fn collect_image_pairs(
    dir1: Option<&str>,
    dir2: Option<&str>,
    images1: Option<&[String]>,
    images2: Option<&[String]>,
    fps: f32,
) -> Result<Vec<(String, String)>> {
    let left_paths = if let Some(dir) = dir1 {
        let (data, _) = image_loader::load_image_paths(dir, fps)?;
        data.into_iter().map(|(p, _, _)| p).collect::<Vec<_>>()
    } else if let Some(files) = images1 {
        let (data, _) = image_loader::load_image_paths_from_files(files, fps)?;
        data.into_iter().map(|(p, _, _)| p).collect::<Vec<_>>()
    } else {
        return Err(anyhow::anyhow!(
            "Either --dir1 or --images1 must be provided for batch mode"
        ));
    };

    let right_paths = if let Some(dir) = dir2 {
        let (data, _) = image_loader::load_image_paths(dir, fps)?;
        data.into_iter().map(|(p, _, _)| p).collect::<Vec<_>>()
    } else if let Some(files) = images2 {
        let (data, _) = image_loader::load_image_paths_from_files(files, fps)?;
        data.into_iter().map(|(p, _, _)| p).collect::<Vec<_>>()
    } else {
        return Err(anyhow::anyhow!(
            "Either --dir2 or --images2 must be provided for batch mode"
        ));
    };

    let pairs = left_paths
        .into_iter()
        .zip(right_paths)
        .collect::<Vec<_>>();
    Ok(pairs)
}

/// Run the batch diff pipeline described by `config`.
///
/// Returns `Ok(results)` when all frames succeed, or an error if any frame
/// fails (after attempting the remaining frames).
pub fn run_batch(config: &BatchConfig) -> Result<Vec<FrameResult>> {
    let output_dir = Path::new(&config.output_dir);
    std::fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "Failed to create output directory '{}'",
            config.output_dir
        )
    })?;

    let mut results = Vec::with_capacity(config.image_pairs.len());
    let mut had_error = false;

    for (index, (left_path, right_path)) in config.image_pairs.iter().enumerate() {
        match process_pair(index, left_path, right_path, output_dir, &config.diff_mode) {
            Ok(result) => {
                info!(
                    "Frame {}: processed OK (flip_mean={:?})",
                    index, result.flip_mean
                );
                results.push(result);
            }
            Err(e) => {
                warn!("Frame {}: processing failed – {}", index, e);
                had_error = true;
                results.push(FrameResult {
                    index,
                    left_path: left_path.clone(),
                    right_path: right_path.clone(),
                    flip_mean: None,
                    flip_min: None,
                    flip_max: None,
                    flip_p95: None,
                    flip_p99: None,
                    output_path: None,
                });
            }
        }
    }

    // Always write metrics so the caller has partial data even on failure.
    write_metrics_csv(&results, output_dir)?;

    if had_error {
        return Err(anyhow::anyhow!(
            "One or more image pairs failed to process; see warnings above"
        ));
    }

    Ok(results)
}

/// Process a single image pair and, if the diff mode requires it, save a PNG.
fn process_pair(
    index: usize,
    left_path: &str,
    right_path: &str,
    output_dir: &Path,
    diff_mode: &DiffMode,
) -> Result<FrameResult> {
    let left_img = image::open(left_path)
        .with_context(|| format!("Failed to open left image '{}'", left_path))?;
    let right_img = image::open(right_path)
        .with_context(|| format!("Failed to open right image '{}'", right_path))?;

    let (lw, lh) = left_img.dimensions();
    let (rw, rh) = right_img.dimensions();

    if lw != rw || lh != rh {
        return Err(anyhow::anyhow!(
            "Dimension mismatch: left {}×{}, right {}×{}",
            lw,
            lh,
            rw,
            rh
        ));
    }

    let mut result = FrameResult {
        index,
        left_path: left_path.to_string(),
        right_path: right_path.to_string(),
        flip_mean: None,
        flip_min: None,
        flip_max: None,
        flip_p95: None,
        flip_p99: None,
        output_path: None,
    };

    if *diff_mode == DiffMode::Flip {
        let left_rgb = left_img.to_rgb8();
        let right_rgb = right_img.to_rgb8();

        let left_flip = FlipImageRgb8::with_data(lw, lh, left_rgb.as_raw());
        let right_flip = FlipImageRgb8::with_data(rw, rh, right_rgb.as_raw());

        let error_map = flip(left_flip, right_flip, nv_flip::DEFAULT_PIXELS_PER_DEGREE);
        let mut pool = FlipPool::from_image(&error_map);

        result.flip_mean = Some(pool.mean());
        result.flip_min = Some(pool.min_value());
        result.flip_max = Some(pool.max_value());
        result.flip_p95 = Some(pool.get_percentile(95.0, true)); // true = use linear interpolation
        result.flip_p99 = Some(pool.get_percentile(99.0, true)); // true = use linear interpolation

        let visualized = error_map.apply_color_lut(&magma_lut());
        let rgb_bytes = visualized.to_vec();

        // Convert RGB → RGBA for PNG encoding.
        let mut rgba = Vec::with_capacity(rgb_bytes.len() / 3 * 4);
        for chunk in rgb_bytes.chunks_exact(3) {
            rgba.push(chunk[0]);
            rgba.push(chunk[1]);
            rgba.push(chunk[2]);
            rgba.push(255);
        }

        let filename = format!("flip_diff_{:06}.png", index);
        let out_path = output_dir.join(&filename);
        image::save_buffer(&out_path, &rgba, lw, lh, image::ColorType::Rgba8)
            .with_context(|| format!("Failed to save diff image to '{}'", out_path.display()))?;

        result.output_path = Some(out_path.to_string_lossy().into_owned());
        info!("Saved diff image: {}", out_path.display());
    }

    Ok(result)
}

/// Write per-frame metrics to `<output_dir>/metrics.csv`.
pub fn write_metrics_csv(results: &[FrameResult], output_dir: &Path) -> Result<()> {
    let csv_path = output_dir.join("metrics.csv");
    let mut content =
        String::from("index,left_path,right_path,flip_mean,flip_min,flip_max,flip_p95,flip_p99,output_path\n");

    for r in results {
        content.push_str(&format!(
            "{},{},{},{},{},{},{},{},{}\n",
            r.index,
            csv_quote(&r.left_path),
            csv_quote(&r.right_path),
            r.flip_mean
                .map(|v| v.to_string())
                .unwrap_or_default(),
            r.flip_min
                .map(|v| v.to_string())
                .unwrap_or_default(),
            r.flip_max
                .map(|v| v.to_string())
                .unwrap_or_default(),
            r.flip_p95
                .map(|v| v.to_string())
                .unwrap_or_default(),
            r.flip_p99
                .map(|v| v.to_string())
                .unwrap_or_default(),
            r.output_path.as_deref().map(csv_quote).unwrap_or_default(),
        ));
    }

    std::fs::write(&csv_path, &content)
        .with_context(|| format!("Failed to write metrics CSV to '{}'", csv_path.display()))?;

    info!("Metrics written to {}", csv_path.display());
    Ok(())
}

/// Quote a CSV field, escaping any embedded double-quotes by doubling them.
fn csv_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    // Minimal valid 1×1 PNG bytes.
    fn minimal_png(r: u8, g: u8, b: u8) -> Vec<u8> {
        // Build a tiny PNG programmatically using the `image` crate so we get a
        // real, decodable file rather than hand-crafted bytes that might not match
        // the exact encoder output.
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([r, g, b]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageOutputFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    fn write_png(path: &Path, r: u8, g: u8, b: u8) {
        let data = minimal_png(r, g, b);
        let mut f = fs::File::create(path).unwrap();
        f.write_all(&data).unwrap();
    }

    // ── DiffMode::from_str ────────────────────────────────────────────────

    #[test]
    fn test_diff_mode_from_str_flip() {
        assert_eq!("flip".parse::<DiffMode>().unwrap(), DiffMode::Flip);
        assert_eq!("FLIP".parse::<DiffMode>().unwrap(), DiffMode::Flip);
    }

    #[test]
    fn test_diff_mode_from_str_none() {
        assert_eq!("none".parse::<DiffMode>().unwrap(), DiffMode::None);
        assert_eq!("NONE".parse::<DiffMode>().unwrap(), DiffMode::None);
    }

    #[test]
    fn test_diff_mode_from_str_invalid() {
        assert!("absolute".parse::<DiffMode>().is_err());
        assert!("".parse::<DiffMode>().is_err());
    }

    // ── process_pair (via run_batch) ──────────────────────────────────────

    #[test]
    fn test_run_batch_flip_mode_produces_png_and_csv() {
        let tmp = tempfile::tempdir().unwrap();
        let left = tmp.path().join("left.png");
        let right = tmp.path().join("right.png");
        write_png(&left, 200, 100, 50);
        write_png(&right, 50, 100, 200);

        let out_dir = tmp.path().join("out");
        let config = BatchConfig {
            image_pairs: vec![(
                left.to_string_lossy().into_owned(),
                right.to_string_lossy().into_owned(),
            )],
            output_dir: out_dir.to_string_lossy().into_owned(),
            diff_mode: DiffMode::Flip,
        };

        let results = run_batch(&config).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].flip_mean.is_some());
        assert!(results[0].output_path.is_some());

        let png = Path::new(results[0].output_path.as_ref().unwrap());
        assert!(png.exists(), "diff PNG should exist on disk");

        let csv = out_dir.join("metrics.csv");
        assert!(csv.exists(), "metrics.csv should exist");
        let content = fs::read_to_string(&csv).unwrap();
        assert!(content.contains("flip_mean"), "CSV header missing");
        assert!(content.contains("left.png") || content.contains("right.png"));
    }

    #[test]
    fn test_run_batch_none_mode_skips_diff() {
        let tmp = tempfile::tempdir().unwrap();
        let left = tmp.path().join("l.png");
        let right = tmp.path().join("r.png");
        write_png(&left, 0, 0, 0);
        write_png(&right, 255, 255, 255);

        let out_dir = tmp.path().join("out_none");
        let config = BatchConfig {
            image_pairs: vec![(
                left.to_string_lossy().into_owned(),
                right.to_string_lossy().into_owned(),
            )],
            output_dir: out_dir.to_string_lossy().into_owned(),
            diff_mode: DiffMode::None,
        };

        let results = run_batch(&config).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].flip_mean.is_none(), "None mode should not compute flip_mean");
        assert!(results[0].output_path.is_none(), "None mode should not write a PNG");
    }

    #[test]
    fn test_run_batch_dimension_mismatch_returns_error() {
        let tmp = tempfile::tempdir().unwrap();

        // 1×1 left image
        let left = tmp.path().join("left.png");
        write_png(&left, 100, 100, 100);

        // 2×2 right image
        let right = tmp.path().join("right.png");
        let img = image::RgbImage::from_pixel(2, 2, image::Rgb([0u8, 0, 0]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageOutputFormat::Png).unwrap();
        fs::write(&right, buf.into_inner()).unwrap();

        let out_dir = tmp.path().join("out_mismatch");
        let config = BatchConfig {
            image_pairs: vec![(
                left.to_string_lossy().into_owned(),
                right.to_string_lossy().into_owned(),
            )],
            output_dir: out_dir.to_string_lossy().into_owned(),
            diff_mode: DiffMode::Flip,
        };

        assert!(run_batch(&config).is_err(), "mismatched dimensions should fail");
    }

    #[test]
    fn test_run_batch_missing_file_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let left = tmp.path().join("does_not_exist.png");
        let right = tmp.path().join("also_missing.png");

        let out_dir = tmp.path().join("out_missing");
        let config = BatchConfig {
            image_pairs: vec![(
                left.to_string_lossy().into_owned(),
                right.to_string_lossy().into_owned(),
            )],
            output_dir: out_dir.to_string_lossy().into_owned(),
            diff_mode: DiffMode::Flip,
        };

        assert!(run_batch(&config).is_err(), "missing input should fail");
    }

    #[test]
    fn test_write_metrics_csv_creates_file() {
        let tmp = tempfile::tempdir().unwrap();
        let results = vec![FrameResult {
            index: 0,
            left_path: "a.png".to_string(),
            right_path: "b.png".to_string(),
            flip_mean: Some(0.05),
            flip_min: Some(0.0),
            flip_max: Some(0.9),
            flip_p95: Some(0.3),
            flip_p99: Some(0.7),
            output_path: Some("out/flip_diff_000000.png".to_string()),
        }];
        write_metrics_csv(&results, tmp.path()).unwrap();
        let csv = tmp.path().join("metrics.csv");
        assert!(csv.exists());
        let content = fs::read_to_string(&csv).unwrap();
        assert!(content.contains("0.05"));
        assert!(content.contains("a.png"));
    }

    #[test]
    fn test_csv_quote_escapes_special_chars() {
        assert_eq!(csv_quote("normal.png"), "\"normal.png\"");
        assert_eq!(csv_quote("path,with,commas.png"), "\"path,with,commas.png\"");
        assert_eq!(csv_quote("path\"with\"quotes.png"), "\"path\"\"with\"\"quotes.png\"");
    }
}
