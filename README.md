# Image Comparison Player

The Image Comparison Player is a Rust-based application designed to view and compare images from one or two directories. It provides a visual interface for side-by-side comparison, flip difference visualization, and playback control of image sequences.

<p align="center">
  <img src="https://github.com/mpiispanen/image_comparison_player/blob/main/gif/output1.gif" />
</p>

<p align="center">
  <img src="https://github.com/mpiispanen/image_comparison_player/blob/main/gif/output2.gif" />
</p>


## Purpose

This application is useful for:
- Viewing a single image sequence
- Comparing two sets of image sequences
- Analyzing differences between image sequences
- Reviewing visual changes in rendered outputs

## How It Works

The application loads images from one or two specified directories. When two directories are provided, images are displayed side-by-side with comparison tools. When only one directory is given, the single image stream fills the entire window (single image mode).

## Arguments

The application accepts the following command-line arguments:

```
--dir1 <DIR>                  First directory containing images (use instead of --images1)
--dir2 <DIR>                  Second directory containing images (use instead of --images2)
--images1 <FILE>...           One or more image files for the left side (use instead of --dir1)
--images2 <FILE>...           One or more image files for the right side (use instead of --dir2)
--window-size <WIDTHxHEIGHT> Window size (default: 1920x1080)
--cache-size <SIZE>           Size of the image cache (default: 50)
--preload-ahead <COUNT>       Number of images to preload ahead (default: 6)
--preload-behind <COUNT>      Number of images to preload behind (default: 0)
--num-load-threads <COUNT>    Number of threads for loading images (default: 6)
--num-process-threads <COUNT> Number of threads for processing images (default: 6)
--num-flip-diff-threads <COUNT> Number of threads for generating flip diffs (default: 4)
--diff-preload-ahead <COUNT>  Number of diff images to preload ahead (default: 4)
--diff-preload-behind <COUNT> Number of diff images to preload behind (default: 0)
--fps <FPS>                   Frames per second (default: 30)
--peek-zoom-factor <FACTOR>   Magnification factor for hold-to-peek zoom (default: 4.0)
--batch-mode                  Run in headless batch mode (diff generation / video export)
--batch-diff-output <DIR>     Output directory for batch diff images
--video-output <FILE>         Output file path for generated video
--video-layout <LAYOUT>       Video layout: single | side-by-side | side-by-side-diff (default: single)
--video-fps <FPS>             Output video frame rate (default: 30)
--video-crf <CRF>             Video quality CRF 0..51 (default: 18)
--video-preset <PRESET>       ffmpeg encoder preset (default: medium)
--video-codec <CODEC>         ffmpeg video codec (default: libx264)
--video-pixel-format <PIX_FMT> ffmpeg pixel format (default: yuv420p)
--use-existing-diffs          Reuse diff PNGs from --batch-diff-output for side-by-side-diff video export
```

Left side requires either `--dir1` (directory) or `--images1` (explicit file list), but not both.
Right side is optional; when omitted (no `--dir2` and no `--images2`), the app runs in single image mode.

Batch mode requires at least one output target (`--batch-diff-output` and/or `--video-output`).
Diff generation and multi-panel video layouts require both left and right image sources.
`--use-existing-diffs` requires `--video-output`, `--batch-diff-output`, and `--video-layout side-by-side-diff`.

## User Interface Controls

The application supports the following keyboard controls:

- Left Arrow: Previous frame
- Right Arrow: Next frame
- Space: Toggle play/pause
- C: Toggle cache debug window
- F: Toggle flip difference mode
- H: Toggle help overlay
- O: Toggle HUD (frame index, speed, zoom, compare mode)
- U: Save a combined screenshot (left/right/FLIP diff panels side-by-side) to a PNG file
- Esc: Close help overlay (if open), otherwise exit the application
- [ : Decrease playback speed
- ] : Increase playback speed
- Q: Zoom out
- E: Zoom in
- W: Move zoom center up
- A: Move zoom center left
- S: Move zoom center down
- D: Move zoom center right
- - / =: Decrease / Increase peek magnifier zoom factor
- P: Save the current FLIP diff image to a PNG file
- I: Save a screenshot of the current window to a PNG file (`screenshot_regular_*` or `screenshot_flip_*` depending on FLIP mode)
- L: Toggle the split line between images on/off

Mouse controls:
- Move the cursor horizontally to adjust the split between left and right images
- Move the cursor vertically in flip difference mode to switch between normal and difference views
- Scroll to zoom in/out

## Input Options

## Supported Image Formats

The application uses the [image](https://crates.io/crates/image) crate for decoding images, and supports all formats it enables by default:

| Extension(s)       | Format            |
|--------------------|-------------------|
| `.png`             | PNG               |
| `.jpg`, `.jpeg`    | JPEG              |
| `.bmp`             | BMP               |
| `.gif`             | GIF               |
| `.tiff`, `.tif`    | TIFF              |
| `.webp`            | WebP              |
| `.ico`             | ICO               |
| `.hdr`             | Radiance HDR      |
| `.ppm`, `.pbm`, `.pgm`, `.pam` | PNM (Netpbm) |
| `.tga`             | TGA (Targa)       |
| `.ff`              | Farbfeld          |
| `.exr`             | OpenEXR           |
| `.qoi`             | QOI               |
| `.dds`             | DDS               |

## Input File Options

The application supports three methods of specifying input images:

1. Directory of Images:
   - Place images in the specified directories (`--dir1` and `--dir2`)
   - Images should have matching names or sequential numbering

2. Explicit File List:
   - Provide one or more image file paths directly with `--images1` and `--images2`
   - Useful for comparing arbitrary images without organizing them into directories
   - Example:
     ```
     ./target/release/image_comparison_player \
       --images1 /path/to/img_a1.png /path/to/img_a2.png \
       --images2 /path/to/img_b1.png /path/to/img_b2.png
     ```

3. Input.txt File:
   - Create an `input.txt` file in the specified directory
   - Each line should alternate between file path and duration
   - Example:
     ```
     file 'frame_0001.png'
     duration 33333us
     file 'frame_0002.png'
     duration 33333us
     ```

## Building and Running

1. Ensure you have Rust and Cargo installed
2. Clone the repository
3. Navigate to the project directory
4. Run `cargo build --release` to build the application
5. Execute the application with the required arguments:
   ```
    # Using directories
    # Single image mode (view one image sequence):
    ./target/release/image_comparison_player --dir1 /path/to/directory

    # Comparison mode (compare two image sequences side-by-side):
    ./target/release/image_comparison_player --dir1 /path/to/first/directory --dir2 /path/to/second/directory

   # Using explicit image files
   ./target/release/image_comparison_player \
     --images1 /path/to/img1.png /path/to/img2.png \
     --images2 /path/to/ref1.png /path/to/ref2.png

   # Batch diff generation for a full sequence
   ./target/release/image_comparison_player \
     --batch-mode \
     --dir1 /path/to/left \
     --dir2 /path/to/right \
     --batch-diff-output /tmp/diffs

   # Batch video export (left | right | diff panels)
    ./target/release/image_comparison_player \
      --batch-mode \
      --dir1 /path/to/left \
      --dir2 /path/to/right \
      --video-output /tmp/comparison.mp4 \
     --video-layout side-by-side-diff \
      --video-crf 16 \
      --video-preset slow

    # Reuse precomputed diffs and only build side-by-side-diff video
    ./target/release/image_comparison_player \
      --batch-mode \
      --dir1 /path/to/left \
      --dir2 /path/to/right \
      --batch-diff-output /tmp/diffs \
      --video-output /tmp/comparison-from-diffs.mp4 \
      --video-layout side-by-side-diff \
      --use-existing-diffs
    ```

## Dependencies

The application uses several external crates, including wgpu for GPU rendering, winit for window management, clap for argument parsing, and imgui for debug UI. Batch video export also uses `tempfile` to stage generated frame sequences before ffmpeg encoding. For a full list of dependencies, refer to the Cargo.toml file.

## Note

This application is designed for comparing image sequences and may require significant system resources depending on the size and number of images being compared.

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/U7U112X6BE)
