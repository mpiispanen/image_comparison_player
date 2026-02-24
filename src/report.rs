//! Diff summary report generation (HTML and Markdown).
//!
//! Generates a shareable artifact summarising the comparison session:
//! frame-range metadata, per-pair FLIP statistics, and an embedded
//! (HTML) or linked (Markdown) FLIP diff screenshot.

use crate::player::FlipStats;
use image::{DynamicImage, RgbaImage};
use std::collections::HashMap;
use std::error::Error;

/// Output format for the diff summary report.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReportFormat {
    Html,
    Markdown,
}

impl ReportFormat {
    pub fn file_extension(self) -> &'static str {
        match self {
            ReportFormat::Html => "html",
            ReportFormat::Markdown => "md",
        }
    }
}

/// All data required to build a diff summary report.
pub struct ReportData {
    /// Total number of frames on the left side.
    pub frame_count_left: usize,
    /// Total number of frames on the right side.
    pub frame_count_right: usize,
    /// Currently displayed left frame index.
    pub current_left: usize,
    /// Currently displayed right frame index.
    pub current_right: usize,
    /// All FLIP stats that have been computed so far, keyed by (left, right).
    pub flip_stats: HashMap<(usize, usize), FlipStats>,
    /// Raw RGBA pixel data for the current frame's FLIP diff image, if available.
    pub flip_diff_image_rgba: Option<(Vec<u8>, u32, u32)>,
    /// Human-readable label for the left image source (directory or first file path).
    pub left_source: String,
    /// Human-readable label for the right image source (directory or first file path).
    pub right_source: String,
}

/// Generate a report and write it to `output_path`.
/// Returns the path on success.
pub fn generate_report(
    data: &ReportData,
    format: ReportFormat,
    output_path: &str,
) -> Result<String, Box<dyn Error>> {
    match format {
        ReportFormat::Html => generate_html(data, output_path),
        ReportFormat::Markdown => generate_markdown(data, output_path),
    }
}

// ---------------------------------------------------------------------------
// HTML report
// ---------------------------------------------------------------------------

fn generate_html(data: &ReportData, output_path: &str) -> Result<String, Box<dyn Error>> {
    let timestamp = current_timestamp_str();
    let flip_image_html = match &data.flip_diff_image_rgba {
        Some((rgba, w, h)) => {
            let png_bytes = rgba_to_png_bytes(rgba.clone(), *w, *h)?;
            let b64 = base64_encode(&png_bytes);
            format!(
                r#"<h2>FLIP Diff Screenshot (frame {}/{})</h2>
<img src="data:image/png;base64,{}" alt="FLIP diff" style="max-width:100%;border:1px solid #555;" />"#,
                data.current_left, data.current_right, b64
            )
        }
        None => {
            r#"<h2>FLIP Diff Screenshot</h2>
<p><em>No FLIP diff available for the current frame pair. Enable FLIP mode (press F) to compute it.</em></p>"#
                .to_string()
        }
    };

    let stats_rows = build_html_stats_rows(&data.flip_stats);
    let current_stats_html = match data.flip_stats.get(&(data.current_left, data.current_right)) {
        Some(s) => format!(
            r#"<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Mean</td><td>{:.4}</td></tr>
  <tr><td>Min</td><td>{:.4}</td></tr>
  <tr><td>Max</td><td>{:.4}</td></tr>
  <tr><td>P95</td><td>{:.4}</td></tr>
  <tr><td>P99</td><td>{:.4}</td></tr>
</table>"#,
            s.mean, s.min, s.max, s.p95, s.p99
        ),
        None => "<p><em>FLIP stats not yet computed for this frame pair.</em></p>".to_string(),
    };

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Diff Summary Report</title>
<style>
  body {{ font-family: sans-serif; background: #1e1e1e; color: #d4d4d4; margin: 2em; }}
  h1 {{ color: #d7ba7d; }}
  h2 {{ color: #9cdcfe; margin-top: 1.5em; }}
  table {{ border-collapse: collapse; margin-top: 0.5em; }}
  th, td {{ border: 1px solid #555; padding: 4px 12px; text-align: left; }}
  th {{ background: #2d2d2d; }}
  p {{ margin: 0.4em 0; }}
  .meta td:first-child {{ font-weight: bold; }}
</style>
</head>
<body>
<h1>Diff Summary Report</h1>
<p>Generated: {timestamp}</p>

<h2>Sequence Info</h2>
<table class="meta">
  <tr><td>Left source</td><td>{left_source}</td></tr>
  <tr><td>Right source</td><td>{right_source}</td></tr>
  <tr><td>Frame range (left)</td><td>0 – {last_left} ({total_left} frames)</td></tr>
  <tr><td>Frame range (right)</td><td>0 – {last_right} ({total_right} frames)</td></tr>
  <tr><td>Current frame (left / right)</td><td>{current_left} / {current_right}</td></tr>
</table>

<h2>FLIP Metrics – Current Frame Pair ({current_left} / {current_right})</h2>
{current_stats_html}

{all_stats_section}

{flip_image_html}
</body>
</html>"#,
        timestamp = timestamp,
        left_source = html_escape(&data.left_source),
        right_source = html_escape(&data.right_source),
        last_left = data.frame_count_left.saturating_sub(1),
        total_left = data.frame_count_left,
        last_right = data.frame_count_right.saturating_sub(1),
        total_right = data.frame_count_right,
        current_left = data.current_left,
        current_right = data.current_right,
        current_stats_html = current_stats_html,
        all_stats_section = stats_rows,
        flip_image_html = flip_image_html,
    );

    std::fs::write(output_path, &html)?;
    Ok(output_path.to_string())
}

fn build_html_stats_rows(stats: &HashMap<(usize, usize), FlipStats>) -> String {
    if stats.is_empty() {
        return "<h2>All Computed FLIP Stats</h2>\n<p><em>No FLIP stats computed yet.</em></p>"
            .to_string();
    }
    let mut pairs: Vec<(usize, usize)> = stats.keys().cloned().collect();
    pairs.sort_unstable();

    let mut rows = String::new();
    for (l, r) in &pairs {
        let s = &stats[&(*l, *r)];
        rows.push_str(&format!(
            "  <tr><td>{}/{}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td></tr>\n",
            l, r, s.mean, s.min, s.max, s.p95, s.p99
        ));
    }
    format!(
        r#"<h2>All Computed FLIP Stats ({} frame pair(s))</h2>
<table>
  <tr><th>Frame (L/R)</th><th>Mean</th><th>Min</th><th>Max</th><th>P95</th><th>P99</th></tr>
{}</table>"#,
        pairs.len(),
        rows
    )
}

// ---------------------------------------------------------------------------
// Markdown report
// ---------------------------------------------------------------------------

fn generate_markdown(data: &ReportData, output_path: &str) -> Result<String, Box<dyn Error>> {
    let timestamp = current_timestamp_str();

    // Save FLIP diff PNG alongside the markdown file (same directory, same timestamp stem).
    let diff_image_ref = match &data.flip_diff_image_rgba {
        Some((rgba, w, h)) => {
            let png_path = output_path.replace(".md", "_flip_diff.png");
            let png_bytes = rgba_to_png_bytes(rgba.clone(), *w, *h)?;
            std::fs::write(&png_path, &png_bytes)?;
            let filename = std::path::Path::new(&png_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&png_path)
                .to_string();
            format!(
                "## FLIP Diff Screenshot (frame {}/{})\n\n![FLIP diff]({})\n",
                data.current_left, data.current_right, filename
            )
        }
        None => "## FLIP Diff Screenshot\n\n*No FLIP diff available for the current frame pair. Enable FLIP mode (press F) to compute it.*\n".to_string(),
    };

    let current_stats_md = match data.flip_stats.get(&(data.current_left, data.current_right)) {
        Some(s) => format!(
            "| Metric | Value |\n|--------|-------|\n| Mean | {:.4} |\n| Min  | {:.4} |\n| Max  | {:.4} |\n| P95  | {:.4} |\n| P99  | {:.4} |\n",
            s.mean, s.min, s.max, s.p95, s.p99
        ),
        None => "*FLIP stats not yet computed for this frame pair.*\n".to_string(),
    };

    let all_stats_md = build_markdown_stats_table(&data.flip_stats);

    let md = format!(
        "# Diff Summary Report\n\nGenerated: {timestamp}\n\n\
## Sequence Info\n\n\
| Field | Value |\n|-------|-------|\n\
| Left source | `{left_source}` |\n\
| Right source | `{right_source}` |\n\
| Frame range (left) | 0 – {last_left} ({total_left} frames) |\n\
| Frame range (right) | 0 – {last_right} ({total_right} frames) |\n\
| Current frame (left / right) | {current_left} / {current_right} |\n\n\
## FLIP Metrics – Current Frame Pair ({current_left} / {current_right})\n\n\
{current_stats_md}\n\
{all_stats_md}\n\
{diff_image_ref}",
        timestamp = timestamp,
        left_source = data.left_source,
        right_source = data.right_source,
        last_left = data.frame_count_left.saturating_sub(1),
        total_left = data.frame_count_left,
        last_right = data.frame_count_right.saturating_sub(1),
        total_right = data.frame_count_right,
        current_left = data.current_left,
        current_right = data.current_right,
        current_stats_md = current_stats_md,
        all_stats_md = all_stats_md,
        diff_image_ref = diff_image_ref,
    );

    std::fs::write(output_path, &md)?;
    Ok(output_path.to_string())
}

fn build_markdown_stats_table(stats: &HashMap<(usize, usize), FlipStats>) -> String {
    if stats.is_empty() {
        return "## All Computed FLIP Stats\n\n*No FLIP stats computed yet.*\n".to_string();
    }
    let mut pairs: Vec<(usize, usize)> = stats.keys().cloned().collect();
    pairs.sort_unstable();

    let mut rows = String::new();
    for (l, r) in &pairs {
        let s = &stats[&(*l, *r)];
        rows.push_str(&format!(
            "| {}/{} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            l, r, s.mean, s.min, s.max, s.p95, s.p99
        ));
    }
    format!(
        "## All Computed FLIP Stats ({} frame pair(s))\n\n\
| Frame (L/R) | Mean | Min | Max | P95 | P99 |\n\
|-------------|------|-----|-----|-----|-----|\n\
{}\n",
        pairs.len(),
        rows
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Encode raw RGBA pixel data as an in-memory PNG byte vector.
fn rgba_to_png_bytes(data: Vec<u8>, width: u32, height: u32) -> Result<Vec<u8>, Box<dyn Error>> {
    let img = RgbaImage::from_raw(width, height, data)
        .ok_or("Failed to create RgbaImage from raw data")?;
    let dynamic = DynamicImage::ImageRgba8(img);
    let mut cursor = std::io::Cursor::new(Vec::new());
    dynamic.write_to(&mut cursor, image::ImageOutputFormat::Png)?;
    Ok(cursor.into_inner())
}

/// Minimal Base64 encoder (RFC 4648) – avoids adding a new crate dependency.
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARS[(n >> 18) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 {
            CHARS[((n >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            CHARS[(n & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Escape the characters that have special meaning in HTML.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// ISO-8601-like timestamp for the report header.
fn current_timestamp_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format as YYYY-MM-DD HH:MM:SS UTC (manual arithmetic, no extra dependencies).
    let s = secs;
    let sec = s % 60;
    let min = (s / 60) % 60;
    let hour = (s / 3600) % 24;
    let days = s / 86400; // days since 1970-01-01

    // Gregorian calendar approximation (accurate for dates 1970–2100).
    let (year, month, day) = days_to_ymd(days);
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        year, month, day, hour, min, sec
    )
}

fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // 400-year cycle = 146097 days
    let z = days + 719468;
    let era = z / 146097;
    let doe = z % 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_stats(mean: f32) -> FlipStats {
        FlipStats {
            mean,
            min: 0.0,
            max: 1.0,
            p95: 0.9,
            p99: 0.99,
        }
    }

    fn minimal_data() -> ReportData {
        let mut stats = HashMap::new();
        stats.insert((0, 0), make_stats(0.123));
        ReportData {
            frame_count_left: 10,
            frame_count_right: 10,
            current_left: 0,
            current_right: 0,
            flip_stats: stats,
            flip_diff_image_rgba: None,
            left_source: "/images/left".to_string(),
            right_source: "/images/right".to_string(),
        }
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<b>\"&\"</b>"), "&lt;b&gt;&quot;&amp;&quot;&lt;/b&gt;");
        assert_eq!(html_escape("plain"), "plain");
    }

    #[test]
    fn test_base64_encode_rfc_vectors() {
        // RFC 4648 test vectors
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn test_current_timestamp_str_format() {
        let ts = current_timestamp_str();
        // Expect "YYYY-MM-DD HH:MM:SS UTC"
        assert!(ts.ends_with(" UTC"), "unexpected suffix: {}", ts);
        let parts: Vec<&str> = ts.trim_end_matches(" UTC").splitn(2, ' ').collect();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].len(), 10); // YYYY-MM-DD
        assert_eq!(parts[1].len(), 8);  // HH:MM:SS
    }

    #[test]
    fn test_generate_html_report_creates_file() {
        let dir = std::env::temp_dir().join("icp_tests").join("report_html");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.html").to_string_lossy().to_string();

        let data = minimal_data();
        let result = generate_report(&data, ReportFormat::Html, &path);
        assert!(result.is_ok(), "generate failed: {:?}", result);
        assert_eq!(result.unwrap(), path);

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("<!DOCTYPE html>"));
        assert!(content.contains("Diff Summary Report"));
        assert!(content.contains("0.1230")); // mean value
        assert!(content.contains("10 frames")); // frame count
    }

    #[test]
    fn test_generate_markdown_report_creates_file() {
        let dir = std::env::temp_dir().join("icp_tests").join("report_md");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.md").to_string_lossy().to_string();

        let data = minimal_data();
        let result = generate_report(&data, ReportFormat::Markdown, &path);
        assert!(result.is_ok(), "generate failed: {:?}", result);

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("# Diff Summary Report"));
        assert!(content.contains("0.1230"));
        assert!(content.contains("10 frames"));
    }

    #[test]
    fn test_generate_html_report_with_flip_diff_image() {
        let dir = std::env::temp_dir().join("icp_tests").join("report_html_img");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.html").to_string_lossy().to_string();

        // Create a tiny 2×2 RGBA image.
        let rgba: Vec<u8> = vec![
            255, 0, 0, 255, 0, 255, 0, 255,
            0, 0, 255, 255, 128, 128, 128, 255,
        ];
        let mut data = minimal_data();
        data.flip_diff_image_rgba = Some((rgba, 2, 2));

        let result = generate_report(&data, ReportFormat::Html, &path);
        assert!(result.is_ok(), "generate failed: {:?}", result);

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("data:image/png;base64,"));
    }

    #[test]
    fn test_generate_markdown_report_with_flip_diff_image() {
        let dir = std::env::temp_dir().join("icp_tests").join("report_md_img");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.md").to_string_lossy().to_string();

        let rgba: Vec<u8> = vec![
            255, 0, 0, 255, 0, 255, 0, 255,
            0, 0, 255, 255, 128, 128, 128, 255,
        ];
        let mut data = minimal_data();
        data.flip_diff_image_rgba = Some((rgba, 2, 2));

        let result = generate_report(&data, ReportFormat::Markdown, &path);
        assert!(result.is_ok(), "generate failed: {:?}", result);

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("![FLIP diff]"));
        // PNG file should exist alongside the markdown
        let png_path = path.replace(".md", "_flip_diff.png");
        assert!(std::path::Path::new(&png_path).exists(), "PNG not created: {}", png_path);
    }

    #[test]
    fn test_report_format_extension() {
        assert_eq!(ReportFormat::Html.file_extension(), "html");
        assert_eq!(ReportFormat::Markdown.file_extension(), "md");
    }

    #[test]
    fn test_rgba_to_png_bytes_roundtrip() {
        let w = 2u32;
        let h = 2u32;
        let rgba: Vec<u8> = vec![
            255, 0, 0, 255, 0, 255, 0, 255,
            0, 0, 255, 255, 255, 255, 0, 255,
        ];
        let png = rgba_to_png_bytes(rgba.clone(), w, h).unwrap();
        assert!(!png.is_empty());
        // PNG magic bytes
        assert_eq!(&png[0..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn test_empty_flip_stats_html() {
        let dir = std::env::temp_dir().join("icp_tests").join("report_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.html").to_string_lossy().to_string();

        let data = ReportData {
            frame_count_left: 5,
            frame_count_right: 5,
            current_left: 2,
            current_right: 2,
            flip_stats: HashMap::new(),
            flip_diff_image_rgba: None,
            left_source: "left".to_string(),
            right_source: "right".to_string(),
        };

        let result = generate_report(&data, ReportFormat::Html, &path);
        assert!(result.is_ok());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("No FLIP stats computed yet"));
    }
}
