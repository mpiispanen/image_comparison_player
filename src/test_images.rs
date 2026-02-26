use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const TEST_FRAME_COUNT: usize = 30;
pub const TEST_IMAGE_WIDTH: u32 = 640;
pub const TEST_IMAGE_HEIGHT: u32 = 480;
const BAND_HEIGHT: u32 = 20;
const BAND_OFFSET: usize = 10;
/// Maximum brightness value used when building horizontal gradients across the image width.
const GRADIENT_MAX: u32 = 128;

/// Monotonically increasing counter so each call to `generate_test_images()`
/// gets its own unique subdirectory, even within the same process.
static INVOCATION_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Generates two sets of synthetic test images in a **unique** temporary directory
/// so that image sequence playback, image comparison and diff generation can be
/// exercised without requiring real image files.
///
/// Each call creates a fresh subdirectory under `$TMPDIR/image_comparison_player_test/<pid>_<n>/`
/// (where `<n>` is a per-process monotonic counter), so concurrent calls and parallel
/// unit tests never share or corrupt each other's files.
///
/// Returns `(dir1_path, dir2_path)` where:
/// * `dir1` contains frames with a red horizontal band moving down on a warm gradient.
/// * `dir2` contains the same frames but the band is blue and offset by a few pixels,
///   so every frame produces a visible FLIP diff.
pub fn generate_test_images() -> Result<(PathBuf, PathBuf)> {
    let n = INVOCATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir()
        .join("image_comparison_player_test")
        .join(format!("{}_{}", std::process::id(), n));
    let dir1 = base.join("dir1");
    let dir2 = base.join("dir2");

    std::fs::create_dir_all(&dir1)?;
    std::fs::create_dir_all(&dir2)?;

    for i in 0..TEST_FRAME_COUNT {
        let img1 = generate_left_frame(i, TEST_FRAME_COUNT, TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT);
        img1.save(dir1.join(format!("frame_{:04}.png", i)))?;

        let img2 = generate_right_frame(i, TEST_FRAME_COUNT, TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT);
        img2.save(dir2.join(format!("frame_{:04}.png", i)))?;
    }

    Ok((dir1, dir2))
}

/// Generates **four** sets of synthetic test images for exercising multi-view mode.
///
/// Builds on `generate_test_images()` for sequences 1 and 2, and adds:
/// * `dir3` – green horizontal band on a neutral grey gradient.
/// * `dir4` – magenta horizontal band on an inverted gradient.
///
/// Returns `(dir1_path, dir2_path, dir3_path, dir4_path)`.
pub fn generate_test_images_multi() -> Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    let (dir1, dir2) = generate_test_images()?;

    // Re-use the same base directory (same invocation counter slot used above).
    let base = dir1.parent().unwrap().to_path_buf();
    let dir3 = base.join("dir3");
    let dir4 = base.join("dir4");

    std::fs::create_dir_all(&dir3)?;
    std::fs::create_dir_all(&dir4)?;

    for i in 0..TEST_FRAME_COUNT {
        let img3 = generate_third_frame(i, TEST_FRAME_COUNT, TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT);
        img3.save(dir3.join(format!("frame_{:04}.png", i)))?;

        let img4 = generate_fourth_frame(i, TEST_FRAME_COUNT, TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT);
        img4.save(dir4.join(format!("frame_{:04}.png", i)))?;
    }

    Ok((dir1, dir2, dir3, dir4))
}

/// Left frame: warm gradient background with a red horizontal band.
fn generate_left_frame(
    frame: usize,
    total: usize,
    width: u32,
    height: u32,
) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(width, height);
    let band_top = (frame * height as usize / total) as u32;
    let band_height = BAND_HEIGHT;

    for y in 0..height {
        for x in 0..width {
            let bg = (x * GRADIENT_MAX / width) as u8;
            let pixel = if y >= band_top && y < band_top + band_height {
                image::Rgba([255, 50, 50, 255])
            } else {
                image::Rgba([bg, bg / 2, bg / 4, 255])
            };
            img.put_pixel(x, y, pixel);
        }
    }
    img
}

/// Right frame: cool gradient background with a blue horizontal band, shifted 10 px
/// relative to the left frame so that the diff is always visible.
fn generate_right_frame(
    frame: usize,
    total: usize,
    width: u32,
    height: u32,
) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(width, height);
    let band_top_raw = (frame * height as usize / total) + BAND_OFFSET;
    let band_top = band_top_raw.min(height as usize - BAND_HEIGHT as usize) as u32;
    let band_height = BAND_HEIGHT;

    for y in 0..height {
        for x in 0..width {
            let bg = (x * GRADIENT_MAX / width) as u8;
            let pixel = if y >= band_top && y < band_top + band_height {
                image::Rgba([50, 50, 255, 255])
            } else {
                image::Rgba([bg / 4, bg / 2, bg, 255])
            };
            img.put_pixel(x, y, pixel);
        }
    }
    img
}

/// Third frame: neutral grey gradient with a green horizontal band.
fn generate_third_frame(
    frame: usize,
    total: usize,
    width: u32,
    height: u32,
) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(width, height);
    let band_top = (frame * height as usize / total) as u32;
    let band_height = BAND_HEIGHT;

    for y in 0..height {
        for x in 0..width {
            let bg = (x * GRADIENT_MAX / width) as u8;
            let pixel = if y >= band_top && y < band_top + band_height {
                image::Rgba([50, 220, 50, 255])
            } else {
                image::Rgba([bg / 2, bg / 2, bg / 2, 255])
            };
            img.put_pixel(x, y, pixel);
        }
    }
    img
}

/// Fourth frame: inverted gradient with a magenta horizontal band, shifted relative
/// to sequence 3 to make all four panels visually distinct.
fn generate_fourth_frame(
    frame: usize,
    total: usize,
    width: u32,
    height: u32,
) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(width, height);
    let band_top_raw = (frame * height as usize / total) + BAND_OFFSET * 2;
    let band_top = band_top_raw.min(height as usize - BAND_HEIGHT as usize) as u32;
    let band_height = BAND_HEIGHT;

    for y in 0..height {
        for x in 0..width {
            let bg = (GRADIENT_MAX - x * GRADIENT_MAX / width) as u8;
            let pixel = if y >= band_top && y < band_top + band_height {
                image::Rgba([220, 50, 220, 255])
            } else {
                image::Rgba([bg, bg / 2, bg / 4, 255])
            };
            img.put_pixel(x, y, pixel);
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    #[test]
    fn test_generate_test_images_creates_files() {
        let (dir1, dir2) = generate_test_images().expect("generate_test_images failed");

        for i in 0..TEST_FRAME_COUNT {
            let path1 = dir1.join(format!("frame_{:04}.png", i));
            let path2 = dir2.join(format!("frame_{:04}.png", i));
            assert!(path1.exists(), "Missing left frame {}", i);
            assert!(path2.exists(), "Missing right frame {}", i);
        }
    }

    #[test]
    fn test_generated_images_have_correct_dimensions() {
        let (dir1, dir2) = generate_test_images().expect("generate_test_images failed");

        let img1 = image::open(dir1.join("frame_0000.png")).expect("Cannot open left frame 0");
        let img2 = image::open(dir2.join("frame_0000.png")).expect("Cannot open right frame 0");

        assert_eq!(
            img1.dimensions(),
            (TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT),
            "Left frame has wrong dimensions"
        );
        assert_eq!(
            img2.dimensions(),
            (TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT),
            "Right frame has wrong dimensions"
        );
    }

    #[test]
    fn test_left_and_right_frames_differ() {
        let (dir1, dir2) = generate_test_images().expect("generate_test_images failed");

        let img1 = image::open(dir1.join("frame_0000.png"))
            .expect("Cannot open left frame 0")
            .to_rgba8();
        let img2 = image::open(dir2.join("frame_0000.png"))
            .expect("Cannot open right frame 0")
            .to_rgba8();

        assert_ne!(
            img1.as_raw(),
            img2.as_raw(),
            "Left and right frames should differ"
        );
    }

    // ── generate_test_images_multi ──────────────────────────────────────────

    #[test]
    fn test_generate_test_images_multi_creates_files() {
        let (dir1, dir2, dir3, dir4) =
            generate_test_images_multi().expect("generate_test_images_multi failed");

        for i in 0..TEST_FRAME_COUNT {
            for (label, dir) in [("dir1", &dir1), ("dir2", &dir2), ("dir3", &dir3), ("dir4", &dir4)] {
                let path = dir.join(format!("frame_{:04}.png", i));
                assert!(path.exists(), "Missing {} frame {}", label, i);
            }
        }
    }

    #[test]
    fn test_generate_test_images_multi_correct_dimensions() {
        let (dir1, dir2, dir3, dir4) =
            generate_test_images_multi().expect("generate_test_images_multi failed");

        for (label, dir) in [("dir1", &dir1), ("dir2", &dir2), ("dir3", &dir3), ("dir4", &dir4)] {
            let img = image::open(dir.join("frame_0000.png"))
                .unwrap_or_else(|_| panic!("Cannot open {} frame_0000.png", label));
            assert_eq!(
                img.dimensions(),
                (TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT),
                "{} has wrong dimensions",
                label
            );
        }
    }

    #[test]
    fn test_all_four_sequences_differ() {
        let (dir1, dir2, dir3, dir4) =
            generate_test_images_multi().expect("generate_test_images_multi failed");

        let imgs: Vec<Vec<u8>> = [&dir1, &dir2, &dir3, &dir4]
            .iter()
            .map(|d| {
                image::open(d.join("frame_0000.png"))
                    .expect("Cannot open frame")
                    .to_rgba8()
                    .into_raw()
            })
            .collect();

        // Every pair of sequences must produce different pixel data.
        for i in 0..imgs.len() {
            for j in (i + 1)..imgs.len() {
                assert_ne!(imgs[i], imgs[j], "Sequences {} and {} should differ", i + 1, j + 1);
            }
        }
    }

    #[test]
    fn test_generate_test_images_multi_unique_dirs() {
        let (d1a, _, d3a, _) = generate_test_images_multi().expect("first call failed");
        let (d1b, _, d3b, _) = generate_test_images_multi().expect("second call failed");
        assert_ne!(d1a, d1b, "Successive calls should produce unique dir1 paths");
        assert_ne!(d3a, d3b, "Successive calls should produce unique dir3 paths");
    }
}
