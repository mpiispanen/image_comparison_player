# Image Comparison Player

The Image Comparison Player is a Rust-based application designed to compare images from two directories. It provides a visual interface for side-by-side comparison, flip difference visualization, and playback control of image sequences.

## Purpose

This application is useful for:
- Comparing two sets of image sequences
- Analyzing differences between image sequences
- Reviewing visual changes in rendered outputs

## How It Works

The application loads images from two specified directories and displays them side-by-side. Users can navigate through the images, play them as a sequence, and use various comparison tools to analyze differences.

## Arguments

The application accepts the following command-line arguments:

```
--dir1 <DIR>                First directory containing images (required)
--dir2 <DIR>                Second directory containing images (required)
--window-size <WIDTHxHEIGHT> Window size (default: 1920x1080)
--cache-size <SIZE>         Size of the image cache (default: 50)
--preload-ahead <COUNT>     Number of images to preload ahead (default: 25)
--preload-behind <COUNT>    Number of images to preload behind (default: 25)
--num-load-threads <COUNT>  Number of threads for loading images (default: 4)
--num-process-threads <COUNT> Number of threads for processing images (default: 4)
--num-flip-diff-threads <COUNT> Number of threads for generating flip diffs (default: 4)
--diff-preload-ahead <COUNT> Number of diff images to preload ahead (default: 5)
--diff-preload-behind <COUNT> Number of diff images to preload behind (default: 1)
--fps <FPS>                 Frames per second (default: 30)
```

## User Interface Controls

The application supports the following keyboard controls:

- Left Arrow: Previous frame
- Right Arrow: Next frame
- Space: Toggle play/pause
- C: Toggle cache debug window
- F: Toggle flip difference mode
- Esc: Exit the application
- [ : Decrease playback speed
- ] : Increase playback speed

Mouse controls:
- Move the cursor horizontally to adjust the split between left and right images
- Move the cursor vertically in flip difference mode to switch between normal and difference views
- Scroll to zoom in/out

## Input File Options

The application supports two methods of specifying input images:

1. Directory of Images:
   - Place images in the specified directories (--dir1 and --dir2)
   - Images should have matching names or sequential numbering

2. Input.txt File:
   - Create an `input.txt` file in the specified directory
   - Each line should alternate between file path and duration
   - Example:
     ```
     file 'frame_0001.png'
     duration 33333
     file 'frame_0002.png'
     duration 33333
     ```

## Building and Running

1. Ensure you have Rust and Cargo installed
2. Clone the repository
3. Navigate to the project directory
4. Run `cargo build --release` to build the application
5. Execute the application with the required arguments:
   ```
   ./target/release/image_comparison_player --dir1 /path/to/first/directory --dir2 /path/to/second/directory
   ```

## Dependencies

The application uses several external crates, including wgpu for GPU rendering, winit for window management, clap for argument parsing, and imgui for debug UI. For a full list of dependencies, refer to the Cargo.toml file.

## Note

This application is designed for comparing image sequences and may require significant system resources depending on the size and number of images being compared.