//! # Color Management Groundwork
//!
//! ## Current Assumptions
//!
//! All image textures are uploaded to the GPU as `wgpu::TextureFormat::Rgba8UnormSrgb`.
//! This means the GPU automatically converts from sRGB-encoded bytes to **linear light**
//! values when the texture is sampled in a fragment shader.  The render surface is
//! configured to prefer an sRGB format (`surface_caps.formats.find(|f| f.is_srgb())`),
//! so the GPU also applies the inverse linear→sRGB conversion when writing to the
//! framebuffer.
//!
//! The net result of the default pipeline is:
//!
//! ```text
//! image bytes (sRGB) ──[GPU tex sample]──► linear ──[shader]──► linear ──[sRGB surface]──► display (sRGB)
//! ```
//!
//! This round-trip is perceptually correct for standard sRGB content: the image is
//! displayed with accurate colors on a typical sRGB monitor.
//!
//! ## Target Model
//!
//! The [`ColorSpace`] enum selects between two display modes:
//!
//! | Mode    | Meaning                                                         |
//! |---------|----------------------------------------------------------------|
//! | `Srgb`  | Default.  GPU-managed sRGB↔linear conversions (see above).     |
//! | `Linear`| Bypass perceptual gamma: display raw linear-light pixel values. |
//!
//! In **Linear** mode the fragment shader applies a manual sRGB→linear conversion
//! *before* writing the output color.  Because the sRGB surface then encodes that
//! value with the standard γ≈2.2 curve, the two operations cancel and the framebuffer
//! receives the unmodified linear values — useful for inspecting HDR content or verifying
//! that rendering pipelines handle linear light correctly.
//!
//! ## Determinism
//!
//! The active [`ColorSpace`] is part of the application state and is propagated to the
//! GPU via a dedicated uniform field (`color_space: f32`, `0.0` = sRGB, `1.0` = Linear).
//! Comparisons are therefore fully deterministic given an explicit mode selection: the
//! same mode always produces the same pixel output for the same input images.

/// Selects the color-space interpretation used when compositing and displaying images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorSpace {
    /// **sRGB mode** (default).
    ///
    /// The GPU-managed sRGB↔linear round-trip is in effect.  Image data is assumed to be
    /// sRGB-encoded; textures are uploaded as `Rgba8UnormSrgb` so the GPU linearises on
    /// sample, and the sRGB render surface re-encodes to sRGB for display.  This is the
    /// correct mode for standard photographic content.
    #[default]
    Srgb,

    /// **Linear mode**.
    ///
    /// The fragment shader manually applies a sRGB→linear conversion to the final color
    /// before outputting it.  Because the sRGB render surface subsequently applies the
    /// linear→sRGB encoding, the two transformations cancel, and the framebuffer receives
    /// the raw linear-light values.  Use this mode to inspect linear-light pixel data or
    /// HDR content without the perceptual γ correction being applied on top.
    Linear,
}

impl ColorSpace {
    /// Returns a short human-readable label suitable for HUD display.
    pub fn label(self) -> &'static str {
        match self {
            ColorSpace::Srgb => "sRGB",
            ColorSpace::Linear => "Linear",
        }
    }

    /// Cycles to the next color-space mode.
    pub fn toggle(self) -> Self {
        match self {
            ColorSpace::Srgb => ColorSpace::Linear,
            ColorSpace::Linear => ColorSpace::Srgb,
        }
    }

    /// Returns the numeric value passed to the GPU uniform (`color_space` field).
    ///
    /// * `0.0` — sRGB (default, no manual conversion in shader)
    /// * `1.0` — Linear (shader applies sRGB→linear before output)
    pub fn as_f32(self) -> f32 {
        match self {
            ColorSpace::Srgb => 0.0,
            ColorSpace::Linear => 1.0,
        }
    }

    /// Parses a color-space name from a CLI/config string (case-insensitive).
    ///
    /// Recognised values: `"srgb"`, `"linear"`.
    /// Returns `None` for unrecognised strings.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "srgb" => Some(ColorSpace::Srgb),
            "linear" => Some(ColorSpace::Linear),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_srgb() {
        assert_eq!(ColorSpace::default(), ColorSpace::Srgb);
    }

    #[test]
    fn test_toggle_cycles() {
        assert_eq!(ColorSpace::Srgb.toggle(), ColorSpace::Linear);
        assert_eq!(ColorSpace::Linear.toggle(), ColorSpace::Srgb);
    }

    #[test]
    fn test_as_f32() {
        assert_eq!(ColorSpace::Srgb.as_f32(), 0.0);
        assert_eq!(ColorSpace::Linear.as_f32(), 1.0);
    }

    #[test]
    fn test_label() {
        assert_eq!(ColorSpace::Srgb.label(), "sRGB");
        assert_eq!(ColorSpace::Linear.label(), "Linear");
    }

    #[test]
    fn test_from_str_valid() {
        assert_eq!(ColorSpace::from_str("srgb"), Some(ColorSpace::Srgb));
        assert_eq!(ColorSpace::from_str("SRGB"), Some(ColorSpace::Srgb));
        assert_eq!(ColorSpace::from_str("linear"), Some(ColorSpace::Linear));
        assert_eq!(ColorSpace::from_str("LINEAR"), Some(ColorSpace::Linear));
        assert_eq!(ColorSpace::from_str("Linear"), Some(ColorSpace::Linear));
    }

    #[test]
    fn test_from_str_invalid() {
        assert_eq!(ColorSpace::from_str(""), None);
        assert_eq!(ColorSpace::from_str("gamma"), None);
        assert_eq!(ColorSpace::from_str("rec2020"), None);
    }

    #[test]
    fn test_toggle_roundtrip() {
        // Toggling twice returns to the original mode.
        let original = ColorSpace::Srgb;
        assert_eq!(original.toggle().toggle(), original);
    }
}
