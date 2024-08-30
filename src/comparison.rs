use vulkanalia::prelude::*;

pub fn render_comparison(
    device: &Device,
    left_image: &crate::image_loader::Image,
    right_image: &crate::image_loader::Image,
    mouse_x: f32,
    mouse_y: f32,
    window_width: u32,
    window_height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Calculate the scaling factor based on the mouse position
    let scale_factor = mouse_x / window_width as f32;

    // Scale the image sizes based on the scaling factor
    let left_image_width = (left_image.width() as f32 * scale_factor) as u32;
    let right_image_width = (right_image.width() as f32 * (1.0 - scale_factor)) as u32;

    // Render the comparison view using Vulkan
    // ...

    Ok(())
}
