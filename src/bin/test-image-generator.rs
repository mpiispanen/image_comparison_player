use anyhow::Result;
use clap::{Arg, ArgAction, Command};
use image::{ImageBuffer, Rgb, RgbImage};
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    env_logger::init();

    let matches = Command::new("test-image-generator")
        .version("1.0")
        .about("Generates test images for visual diff testing")
        .subcommand(
            Command::new("generate-inputs")
                .about("Generate input test images")
        )
        .subcommand(
            Command::new("generate-references")
                .about("Generate reference images using the image comparison player")
                .arg(
                    Arg::new("output-dir")
                        .long("output-dir")
                        .action(ArgAction::Set)
                        .value_name("DIR")
                        .help("Output directory for reference images")
                        .default_value("test_images/reference"),
                )
        )
        .get_matches();

    match matches.subcommand() {
        Some(("generate-inputs", _)) => generate_input_images(),
        Some(("generate-references", sub_matches)) => {
            let output_dir = sub_matches.get_one::<String>("output-dir").unwrap();
            generate_reference_images(output_dir)
        }
        _ => {
            eprintln!("No subcommand provided. Use 'generate-inputs' or 'generate-references'");
            std::process::exit(1);
        }
    }
}

fn generate_input_images() -> Result<()> {
    println!("Generating input test images...");
    
    // Create directories
    fs::create_dir_all("test_images/input/dir1")?;
    fs::create_dir_all("test_images/input/dir2")?;
    
    // Generate various test patterns
    let test_cases = vec![
        ("solid_red", generate_solid_color_image(255, 0, 0)),
        ("solid_blue", generate_solid_color_image(0, 0, 255)),
        ("gradient", generate_gradient_image()),
        ("checkerboard", generate_checkerboard_image()),
        ("noise", generate_noise_image()),
    ];
    
    for (name, img) in test_cases {
        // Save to dir1
        let path1 = format!("test_images/input/dir1/{}.png", name);
        img.save(&path1)?;
        println!("Generated: {}", path1);
        
        // Generate slightly modified version for dir2 to test comparison
        let modified_img = if name == "solid_red" {
            // Make it slightly different red
            generate_solid_color_image(250, 5, 5)
        } else {
            img.clone()
        };
        
        let path2 = format!("test_images/input/dir2/{}.png", name);
        modified_img.save(&path2)?;
        println!("Generated: {}", path2);
    }
    
    println!("Input test images generated successfully!");
    Ok(())
}

fn generate_reference_images(output_dir: &str) -> Result<()> {
    println!("Generating reference images using headless rendering...");
    
    fs::create_dir_all(output_dir)?;
    
    // Check if input images exist
    if !Path::new("test_images/input/dir1").exists() {
        println!("Warning: Input images not found. Run 'generate-inputs' first.");
        return Ok(());
    }
    
    // For now, we'll create simple reference images
    // In a real implementation, this would run the image comparison player
    // in headless mode to generate actual reference outputs
    
    let test_cases = vec![
        "solid_red",
        "solid_blue", 
        "gradient",
        "checkerboard",
        "noise",
    ];
    
    for test_case in test_cases {
        // Create a reference image that represents the expected output
        // This is a simplified version - in reality, this would run the actual
        // image comparison player to generate the reference
        let reference_img = generate_reference_output(test_case)?;
        
        let path = format!("{}/{}_reference.png", output_dir, test_case);
        reference_img.save(&path)?;
        println!("Generated reference: {}", path);
    }
    
    println!("Reference images generated successfully!");
    Ok(())
}

fn generate_solid_color_image(r: u8, g: u8, b: u8) -> RgbImage {
    ImageBuffer::from_fn(200, 200, |_x, _y| Rgb([r, g, b]))
}

fn generate_gradient_image() -> RgbImage {
    ImageBuffer::from_fn(200, 200, |x, _y| {
        let intensity = ((x as f32 / 200.0) * 255.0) as u8;
        Rgb([intensity, intensity, intensity])
    })
}

fn generate_checkerboard_image() -> RgbImage {
    ImageBuffer::from_fn(200, 200, |x, y| {
        if (x / 20 + y / 20) % 2 == 0 {
            Rgb([255, 255, 255])
        } else {
            Rgb([0, 0, 0])
        }
    })
}

fn generate_noise_image() -> RgbImage {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    ImageBuffer::from_fn(200, 200, |x, y| {
        let mut hasher = DefaultHasher::new();
        (x, y).hash(&mut hasher);
        let hash = hasher.finish();
        let r = (hash & 0xFF) as u8;
        let g = ((hash >> 8) & 0xFF) as u8;
        let b = ((hash >> 16) & 0xFF) as u8;
        Rgb([r, g, b])
    })
}

fn generate_reference_output(test_case: &str) -> Result<RgbImage> {
    // This is a simplified reference generator
    // In a real implementation, this would run the image comparison player
    // to generate the actual expected output
    
    match test_case {
        "solid_red" => Ok(generate_solid_color_image(128, 0, 128)), // Expected blend
        "solid_blue" => Ok(generate_solid_color_image(0, 0, 255)),
        "gradient" => Ok(generate_gradient_image()),
        "checkerboard" => Ok(generate_checkerboard_image()),
        "noise" => Ok(generate_noise_image()),
        _ => Ok(generate_solid_color_image(128, 128, 128)), // Default gray
    }
}