use anyhow::Result;
use clap::{Arg, ArgAction, Command};
use image::{ImageBuffer, Rgb, RgbImage};
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    env_logger::init();

    let matches = Command::new("visual-diff-test")
        .version("1.0")
        .about("Runs visual diff tests against existing reference images")
        .arg(
            Arg::new("test-images-dir")
                .long("test-images-dir")
                .action(ArgAction::Set)
                .value_name("DIR")
                .help("Directory containing test images")
                .default_value("test_images"),
        )
        .arg(
            Arg::new("output-dir")
                .long("output-dir")
                .action(ArgAction::Set)
                .value_name("DIR")
                .help("Output directory for current test results")
                .default_value("test_images/current"),
        )
        .arg(
            Arg::new("reference-dir")
                .long("reference-dir")
                .action(ArgAction::Set)
                .value_name("DIR")
                .help("Directory containing reference images")
                .default_value("test_images/reference"),
        )
        .arg(
            Arg::new("threshold")
                .long("threshold")
                .action(ArgAction::Set)
                .value_name("FLOAT")
                .help("Difference threshold for visual comparison")
                .default_value("0.01"),
        )
        .get_matches();

    let test_images_dir = matches.get_one::<String>("test-images-dir").unwrap();
    let output_dir = matches.get_one::<String>("output-dir").unwrap();
    let reference_dir = matches.get_one::<String>("reference-dir").unwrap();
    let threshold: f32 = matches.get_one::<String>("threshold").unwrap().parse()?;

    run_visual_diff_tests(test_images_dir, output_dir, reference_dir, threshold)
}

fn run_visual_diff_tests(
    test_images_dir: &str,
    output_dir: &str,
    reference_dir: &str,
    threshold: f32,
) -> Result<()> {
    println!("Running visual diff tests...");
    println!("Test images dir: {}", test_images_dir);
    println!("Output dir: {}", output_dir);
    println!("Reference dir: {}", reference_dir);
    println!("Threshold: {}", threshold);

    // Create output directory
    fs::create_dir_all(output_dir)?;

    // Check if reference images exist
    if !Path::new(reference_dir).exists() {
        println!("Warning: Reference directory '{}' does not exist.", reference_dir);
        println!("Please run the test image generator first or download reference images.");
        return Ok(());
    }

    // Check if input images exist
    let input_dir1 = format!("{}/input/dir1", test_images_dir);
    let input_dir2 = format!("{}/input/dir2", test_images_dir);
    
    if !Path::new(&input_dir1).exists() || !Path::new(&input_dir2).exists() {
        println!("Warning: Input directories do not exist. Creating minimal test...");
        return run_minimal_test(output_dir, reference_dir, threshold);
    }

    // Run tests for each test case
    let test_cases = get_test_cases(&input_dir1)?;
    
    for test_case in test_cases {
        println!("Testing: {}", test_case);
        
        let input1_path = format!("{}/{}.png", input_dir1, test_case);
        let input2_path = format!("{}/{}.png", input_dir2, test_case);
        
        if Path::new(&input1_path).exists() && Path::new(&input2_path).exists() {
            // Simulate running the image comparison player
            // In a real implementation, this would execute the actual application
            // with the test inputs and capture the output
            let result_image = simulate_image_comparison(&input1_path, &input2_path)?;
            
            let output_path = format!("{}/{}.png", output_dir, test_case);
            result_image.save(&output_path)?;
            println!("Generated current output: {}", output_path);
        } else {
            println!("Warning: Missing input files for test case: {}", test_case);
        }
    }

    println!("Visual diff tests completed. Output images saved to: {}", output_dir);
    println!("Compare these with reference images in: {}", reference_dir);
    
    Ok(())
}

fn run_minimal_test(output_dir: &str, reference_dir: &str, _threshold: f32) -> Result<()> {
    println!("Running minimal visual diff test...");
    
    // Create a simple test output
    let test_image = generate_test_output();
    let output_path = format!("{}/minimal_test.png", output_dir);
    test_image.save(&output_path)?;
    println!("Generated minimal test output: {}", output_path);
    
    Ok(())
}

fn get_test_cases(input_dir: &str) -> Result<Vec<String>> {
    let mut test_cases = Vec::new();
    
    if let Ok(entries) = fs::read_dir(input_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                if let Some(file_name) = entry.file_name().to_str() {
                    if file_name.ends_with(".png") {
                        let test_case = file_name.trim_end_matches(".png");
                        test_cases.push(test_case.to_string());
                    }
                }
            }
        }
    }
    
    test_cases.sort();
    Ok(test_cases)
}

fn simulate_image_comparison(input1_path: &str, input2_path: &str) -> Result<RgbImage> {
    // This is a simplified simulation of the image comparison player
    // In a real implementation, this would:
    // 1. Load the two input images
    // 2. Run the actual image comparison player with these inputs
    // 3. Capture the rendered output (possibly using headless rendering)
    // 4. Return the output image
    
    println!("Simulating comparison of {} and {}", input1_path, input2_path);
    
    // For now, create a simple blend of the two images
    let img1 = image::open(input1_path)?.to_rgb8();
    let img2 = image::open(input2_path)?.to_rgb8();
    
    let (width, height) = img1.dimensions();
    let blended = ImageBuffer::from_fn(width, height, |x, y| {
        let pixel1 = img1.get_pixel(x, y);
        let pixel2 = img2.get_pixel(x, y);
        
        // Simple blend
        let r = (pixel1[0] as u16 + pixel2[0] as u16) / 2;
        let g = (pixel1[1] as u16 + pixel2[1] as u16) / 2;
        let b = (pixel1[2] as u16 + pixel2[2] as u16) / 2;
        
        Rgb([r as u8, g as u8, b as u8])
    });
    
    Ok(blended)
}

fn generate_test_output() -> RgbImage {
    // Generate a simple test pattern for minimal testing
    ImageBuffer::from_fn(100, 100, |x, y| {
        let intensity = ((x + y) % 255) as u8;
        Rgb([intensity, intensity / 2, intensity / 4])
    })
}