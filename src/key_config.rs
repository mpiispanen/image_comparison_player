use log::warn;
use std::collections::HashMap;
use std::path::Path;
use winit::event::VirtualKeyCode;

/// All remappable keyboard actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyAction {
    PlayPause,
    NextFrame,
    PreviousFrame,
    ZoomIn,
    ZoomOut,
    PanUp,
    PanLeft,
    PanDown,
    PanRight,
    CycleComparisonMode,
    ToggleSplitLine,
    ToggleShowLeft,
    ToggleShowRight,
    SaveFlipDiff,
    SaveScreenshot,
    SaveCombinedScreenshot,
    ToggleHelp,
    ToggleCacheDebug,
    TogglePixelInfo,
    DecreasePlaybackSpeed,
    IncreasePlaybackSpeed,
    ResetZoom,
    ToggleHud,
    /// Hold-to-peek magnifying-glass zoom (Z key by default).
    /// The action is active while the key is held and deactivates on key release.
    PeekZoom,
    DecreasePeekZoom,
    IncreasePeekZoom,
    Quit,
}

/// Keyboard configuration: maps `VirtualKeyCode` → `KeyAction`.
///
/// Build with `KeyConfig::default()` for built-in bindings, or
/// `KeyConfig::load(path)` to load overrides from a plain-text config file.
pub struct KeyConfig {
    pub bindings: HashMap<VirtualKeyCode, KeyAction>,
}

impl Default for KeyConfig {
    fn default() -> Self {
        let mut bindings = HashMap::new();
        for (key, action) in Self::default_pairs() {
            bindings.insert(key, action);
        }
        Self { bindings }
    }
}

impl KeyConfig {
    /// The canonical set of (key, action) pairs used when no config file is present.
    pub fn default_pairs() -> Vec<(VirtualKeyCode, KeyAction)> {
        vec![
            (VirtualKeyCode::Space, KeyAction::PlayPause),
            (VirtualKeyCode::Right, KeyAction::NextFrame),
            (VirtualKeyCode::Left, KeyAction::PreviousFrame),
            (VirtualKeyCode::E, KeyAction::ZoomIn),
            (VirtualKeyCode::Up, KeyAction::ZoomIn),
            (VirtualKeyCode::Q, KeyAction::ZoomOut),
            (VirtualKeyCode::Down, KeyAction::ZoomOut),
            (VirtualKeyCode::W, KeyAction::PanUp),
            (VirtualKeyCode::A, KeyAction::PanLeft),
            (VirtualKeyCode::S, KeyAction::PanDown),
            (VirtualKeyCode::D, KeyAction::PanRight),
            (VirtualKeyCode::F, KeyAction::CycleComparisonMode),
            (VirtualKeyCode::L, KeyAction::ToggleSplitLine),
            (VirtualKeyCode::Key1, KeyAction::ToggleShowLeft),
            (VirtualKeyCode::Key2, KeyAction::ToggleShowRight),
            (VirtualKeyCode::P, KeyAction::SaveFlipDiff),
            (VirtualKeyCode::I, KeyAction::SaveScreenshot),
            (VirtualKeyCode::U, KeyAction::SaveCombinedScreenshot),
            (VirtualKeyCode::H, KeyAction::ToggleHelp),
            (VirtualKeyCode::C, KeyAction::ToggleCacheDebug),
            (VirtualKeyCode::V, KeyAction::TogglePixelInfo),
            (VirtualKeyCode::LBracket, KeyAction::DecreasePlaybackSpeed),
            (VirtualKeyCode::RBracket, KeyAction::IncreasePlaybackSpeed),
            (VirtualKeyCode::R, KeyAction::ResetZoom),
            (VirtualKeyCode::O, KeyAction::ToggleHud),
            (VirtualKeyCode::Z, KeyAction::PeekZoom),
            (VirtualKeyCode::Minus, KeyAction::DecreasePeekZoom),
            (VirtualKeyCode::Equals, KeyAction::IncreasePeekZoom),
            (VirtualKeyCode::Escape, KeyAction::Quit),
        ]
    }

    /// Load key bindings from `path`, merging overrides on top of the defaults.
    ///
    /// **Format** (one binding per line, `#` starts a comment):
    /// ```text
    /// # action = Key1[, Key2, ...]
    /// zoom_in = E, Up
    /// play_pause = Space
    /// ```
    ///
    /// Unknown action names and unrecognised key names produce a `warn!` log and
    /// are skipped; all other defaults remain unchanged.
    pub fn load(path: &Path) -> Self {
        let content = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to read key config file {:?}: {}", path, e);
                return Self::default();
            }
        };

        // Build action → default-keys map so we can apply per-action overrides.
        let mut by_action: HashMap<KeyAction, Vec<VirtualKeyCode>> = HashMap::new();
        for (key, action) in Self::default_pairs() {
            by_action.entry(action).or_default().push(key);
        }

        // Parse overrides from the file.
        for (line_idx, raw_line) in content.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let Some((lhs, rhs)) = line.split_once('=') else {
                warn!(
                    "Key config line {}: expected 'action = key', got {:?}",
                    line_idx + 1,
                    line
                );
                continue;
            };

            let action_name = lhs.trim().to_lowercase();
            let Some(action) = Self::parse_action(&action_name) else {
                warn!(
                    "Key config line {}: unknown action {:?}, skipping",
                    line_idx + 1,
                    action_name
                );
                continue;
            };

            let mut keys = Vec::new();
            let mut all_valid = true;
            for token in rhs.split(',') {
                let key_name = token.trim();
                match Self::parse_key(key_name) {
                    Some(k) => keys.push(k),
                    None => {
                        warn!(
                            "Key config line {}: unknown key {:?} for action {:?}, skipping action",
                            line_idx + 1,
                            key_name,
                            action_name
                        );
                        all_valid = false;
                        break;
                    }
                }
            }

            if all_valid && !keys.is_empty() {
                by_action.insert(action, keys);
            }
        }

        // Flatten into the key → action map, warning on conflicts.
        let mut bindings: HashMap<VirtualKeyCode, KeyAction> = HashMap::new();
        for (action, keys) in &by_action {
            for &key in keys {
                if let Some(&existing) = bindings.get(&key) {
                    if existing != *action {
                        warn!(
                            "Key config: key {:?} is bound to both {:?} and {:?}; keeping {:?}",
                            key, existing, action, existing
                        );
                        continue;
                    }
                }
                bindings.insert(key, *action);
            }
        }

        Self { bindings }
    }

    /// Convert an action name string (lower-case) to its `KeyAction` variant.
    pub fn parse_action(s: &str) -> Option<KeyAction> {
        match s {
            "play_pause" => Some(KeyAction::PlayPause),
            "next_frame" => Some(KeyAction::NextFrame),
            "previous_frame" => Some(KeyAction::PreviousFrame),
            "zoom_in" => Some(KeyAction::ZoomIn),
            "zoom_out" => Some(KeyAction::ZoomOut),
            "pan_up" => Some(KeyAction::PanUp),
            "pan_left" => Some(KeyAction::PanLeft),
            "pan_down" => Some(KeyAction::PanDown),
            "pan_right" => Some(KeyAction::PanRight),
            "cycle_comparison_mode" => Some(KeyAction::CycleComparisonMode),
            "toggle_split_line" => Some(KeyAction::ToggleSplitLine),
            "toggle_show_left" => Some(KeyAction::ToggleShowLeft),
            "toggle_show_right" => Some(KeyAction::ToggleShowRight),
            "save_flip_diff" => Some(KeyAction::SaveFlipDiff),
            "save_screenshot" => Some(KeyAction::SaveScreenshot),
            "save_combined_screenshot" => Some(KeyAction::SaveCombinedScreenshot),
            "toggle_help" => Some(KeyAction::ToggleHelp),
            "toggle_cache_debug" => Some(KeyAction::ToggleCacheDebug),
            "toggle_pixel_info" => Some(KeyAction::TogglePixelInfo),
            "decrease_playback_speed" => Some(KeyAction::DecreasePlaybackSpeed),
            "increase_playback_speed" => Some(KeyAction::IncreasePlaybackSpeed),
            "reset_zoom" => Some(KeyAction::ResetZoom),
            "toggle_hud" => Some(KeyAction::ToggleHud),
            "peek_zoom" => Some(KeyAction::PeekZoom),
            "decrease_peek_zoom" => Some(KeyAction::DecreasePeekZoom),
            "increase_peek_zoom" => Some(KeyAction::IncreasePeekZoom),
            "quit" => Some(KeyAction::Quit),
            _ => None,
        }
    }

    /// Convert a key-name string to its `VirtualKeyCode` variant.
    ///
    /// Accepts common spellings (case-insensitive for letters, exact for
    /// special keys).  Returns `None` and the caller should emit a warning.
    pub fn parse_key(s: &str) -> Option<VirtualKeyCode> {
        // Single-character letter keys (A–Z, case-insensitive)
        if s.len() == 1 {
            let c = s.chars().next()?;
            match c.to_ascii_uppercase() {
                'A' => return Some(VirtualKeyCode::A),
                'B' => return Some(VirtualKeyCode::B),
                'C' => return Some(VirtualKeyCode::C),
                'D' => return Some(VirtualKeyCode::D),
                'E' => return Some(VirtualKeyCode::E),
                'F' => return Some(VirtualKeyCode::F),
                'G' => return Some(VirtualKeyCode::G),
                'H' => return Some(VirtualKeyCode::H),
                'I' => return Some(VirtualKeyCode::I),
                'J' => return Some(VirtualKeyCode::J),
                'K' => return Some(VirtualKeyCode::K),
                'L' => return Some(VirtualKeyCode::L),
                'M' => return Some(VirtualKeyCode::M),
                'N' => return Some(VirtualKeyCode::N),
                'O' => return Some(VirtualKeyCode::O),
                'P' => return Some(VirtualKeyCode::P),
                'Q' => return Some(VirtualKeyCode::Q),
                'R' => return Some(VirtualKeyCode::R),
                'S' => return Some(VirtualKeyCode::S),
                'T' => return Some(VirtualKeyCode::T),
                'U' => return Some(VirtualKeyCode::U),
                'V' => return Some(VirtualKeyCode::V),
                'W' => return Some(VirtualKeyCode::W),
                'X' => return Some(VirtualKeyCode::X),
                'Y' => return Some(VirtualKeyCode::Y),
                'Z' => return Some(VirtualKeyCode::Z),
                '0' => return Some(VirtualKeyCode::Key0),
                '1' => return Some(VirtualKeyCode::Key1),
                '2' => return Some(VirtualKeyCode::Key2),
                '3' => return Some(VirtualKeyCode::Key3),
                '4' => return Some(VirtualKeyCode::Key4),
                '5' => return Some(VirtualKeyCode::Key5),
                '6' => return Some(VirtualKeyCode::Key6),
                '7' => return Some(VirtualKeyCode::Key7),
                '8' => return Some(VirtualKeyCode::Key8),
                '9' => return Some(VirtualKeyCode::Key9),
                _ => {}
            }
        }

        // Multi-character named keys (case-insensitive comparison)
        match s.to_lowercase().as_str() {
            "escape" | "esc" => Some(VirtualKeyCode::Escape),
            "space" => Some(VirtualKeyCode::Space),
            "left" => Some(VirtualKeyCode::Left),
            "right" => Some(VirtualKeyCode::Right),
            "up" => Some(VirtualKeyCode::Up),
            "down" => Some(VirtualKeyCode::Down),
            "[" | "lbracket" => Some(VirtualKeyCode::LBracket),
            "]" | "rbracket" => Some(VirtualKeyCode::RBracket),
            "-" | "minus" | "hyphen" => Some(VirtualKeyCode::Minus),
            "=" | "equals" => Some(VirtualKeyCode::Equals),
            "return" | "enter" => Some(VirtualKeyCode::Return),
            "tab" => Some(VirtualKeyCode::Tab),
            "back" | "backspace" => Some(VirtualKeyCode::Back),
            "delete" | "del" => Some(VirtualKeyCode::Delete),
            "home" => Some(VirtualKeyCode::Home),
            "end" => Some(VirtualKeyCode::End),
            "pageup" => Some(VirtualKeyCode::PageUp),
            "pagedown" => Some(VirtualKeyCode::PageDown),
            "insert" => Some(VirtualKeyCode::Insert),
            "f1" => Some(VirtualKeyCode::F1),
            "f2" => Some(VirtualKeyCode::F2),
            "f3" => Some(VirtualKeyCode::F3),
            "f4" => Some(VirtualKeyCode::F4),
            "f5" => Some(VirtualKeyCode::F5),
            "f6" => Some(VirtualKeyCode::F6),
            "f7" => Some(VirtualKeyCode::F7),
            "f8" => Some(VirtualKeyCode::F8),
            "f9" => Some(VirtualKeyCode::F9),
            "f10" => Some(VirtualKeyCode::F10),
            "f11" => Some(VirtualKeyCode::F11),
            "f12" => Some(VirtualKeyCode::F12),
            "lshift" | "leftshift" => Some(VirtualKeyCode::LShift),
            "rshift" | "rightshift" => Some(VirtualKeyCode::RShift),
            "lcontrol" | "lctrl" | "leftcontrol" | "leftctrl" => Some(VirtualKeyCode::LControl),
            "rcontrol" | "rctrl" | "rightcontrol" | "rightctrl" => Some(VirtualKeyCode::RControl),
            "lalt" | "leftalt" => Some(VirtualKeyCode::LAlt),
            "ralt" | "rightalt" => Some(VirtualKeyCode::RAlt),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct TempFile {
        path: std::path::PathBuf,
    }
    impl TempFile {
        fn new(name: &str, contents: &str) -> Self {
            let dir = std::env::temp_dir().join("icp_key_config_tests");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(name);
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(contents.as_bytes()).unwrap();
            Self { path }
        }
    }
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn default_bindings_are_complete() {
        let cfg = KeyConfig::default();
        // Check a handful of well-known defaults
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::Space), Some(&KeyAction::PlayPause));
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::Left), Some(&KeyAction::PreviousFrame));
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::Right), Some(&KeyAction::NextFrame));
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::Escape), Some(&KeyAction::Quit));
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::H), Some(&KeyAction::ToggleHelp));
        // New actions
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::F), Some(&KeyAction::CycleComparisonMode));
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::Z), Some(&KeyAction::PeekZoom));
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::R), Some(&KeyAction::ResetZoom));
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::O), Some(&KeyAction::ToggleHud));
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::U), Some(&KeyAction::SaveCombinedScreenshot));
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::Minus), Some(&KeyAction::DecreasePeekZoom));
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::Equals), Some(&KeyAction::IncreasePeekZoom));
    }

    #[test]
    fn load_overrides_single_action() {
        let tf = TempFile::new("override_single.txt", "play_pause = J\n");
        let cfg = KeyConfig::load(&tf.path);
        // J is now play_pause
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::J), Some(&KeyAction::PlayPause));
        // Space is no longer play_pause (override replaced it)
        assert_ne!(cfg.bindings.get(&VirtualKeyCode::Space), Some(&KeyAction::PlayPause));
    }

    #[test]
    fn load_overrides_multiple_keys_for_action() {
        let tf = TempFile::new("override_multi.txt", "zoom_in = J, K\n");
        let cfg = KeyConfig::load(&tf.path);
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::J), Some(&KeyAction::ZoomIn));
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::K), Some(&KeyAction::ZoomIn));
        // Default keys no longer bound to zoom_in
        assert_ne!(cfg.bindings.get(&VirtualKeyCode::E), Some(&KeyAction::ZoomIn));
    }

    #[test]
    fn load_ignores_comments_and_blank_lines() {
        let tf = TempFile::new("comments.txt", "# this is a comment\n\n# another\nplay_pause = N\n");
        let cfg = KeyConfig::load(&tf.path);
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::N), Some(&KeyAction::PlayPause));
    }

    #[test]
    fn load_warns_unknown_action_keeps_defaults() {
        let tf = TempFile::new("bad_action.txt", "nonexistent_action = X\n");
        let cfg = KeyConfig::load(&tf.path);
        // The default for Space (play_pause) should still be present
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::Space), Some(&KeyAction::PlayPause));
    }

    #[test]
    fn load_warns_unknown_key_keeps_defaults() {
        let tf = TempFile::new("bad_key.txt", "play_pause = NotAKey\n");
        let cfg = KeyConfig::load(&tf.path);
        // play_pause should fall back to Space default
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::Space), Some(&KeyAction::PlayPause));
    }

    #[test]
    fn load_missing_file_returns_defaults() {
        let missing = std::env::temp_dir().join("icp_key_config_tests").join("does_not_exist.txt");
        let cfg = KeyConfig::load(&missing);
        assert_eq!(cfg.bindings.get(&VirtualKeyCode::Space), Some(&KeyAction::PlayPause));
    }

    #[test]
    fn parse_key_all_letters() {
        for (ch, expected) in [
            ('A', VirtualKeyCode::A),
            ('z', VirtualKeyCode::Z),
            ('m', VirtualKeyCode::M),
        ] {
            assert_eq!(KeyConfig::parse_key(&ch.to_string()), Some(expected));
        }
    }

    #[test]
    fn parse_key_digits() {
        assert_eq!(KeyConfig::parse_key("0"), Some(VirtualKeyCode::Key0));
        assert_eq!(KeyConfig::parse_key("9"), Some(VirtualKeyCode::Key9));
    }

    #[test]
    fn parse_key_special() {
        assert_eq!(KeyConfig::parse_key("Escape"), Some(VirtualKeyCode::Escape));
        assert_eq!(KeyConfig::parse_key("esc"), Some(VirtualKeyCode::Escape));
        assert_eq!(KeyConfig::parse_key("Space"), Some(VirtualKeyCode::Space));
        assert_eq!(KeyConfig::parse_key("["), Some(VirtualKeyCode::LBracket));
        assert_eq!(KeyConfig::parse_key("]"), Some(VirtualKeyCode::RBracket));
        assert_eq!(KeyConfig::parse_key("-"), Some(VirtualKeyCode::Minus));
        assert_eq!(KeyConfig::parse_key("Minus"), Some(VirtualKeyCode::Minus));
        assert_eq!(KeyConfig::parse_key("="), Some(VirtualKeyCode::Equals));
        assert_eq!(KeyConfig::parse_key("Equals"), Some(VirtualKeyCode::Equals));
        assert_eq!(KeyConfig::parse_key("F1"), Some(VirtualKeyCode::F1));
        assert_eq!(KeyConfig::parse_key("f12"), Some(VirtualKeyCode::F12));
    }

    #[test]
    fn parse_key_unknown_returns_none() {
        assert_eq!(KeyConfig::parse_key("NotAKey"), None);
        assert_eq!(KeyConfig::parse_key(""), None);
    }

    #[test]
    fn parse_action_roundtrip() {
        let actions = [
            ("play_pause", KeyAction::PlayPause),
            ("next_frame", KeyAction::NextFrame),
            ("cycle_comparison_mode", KeyAction::CycleComparisonMode),
            ("peek_zoom", KeyAction::PeekZoom),
            ("reset_zoom", KeyAction::ResetZoom),
            ("toggle_hud", KeyAction::ToggleHud),
            ("save_combined_screenshot", KeyAction::SaveCombinedScreenshot),
            ("decrease_peek_zoom", KeyAction::DecreasePeekZoom),
            ("increase_peek_zoom", KeyAction::IncreasePeekZoom),
            ("quit", KeyAction::Quit),
            ("toggle_help", KeyAction::ToggleHelp),
        ];
        for (name, expected) in actions {
            assert_eq!(KeyConfig::parse_action(name), Some(expected));
        }
        assert_eq!(KeyConfig::parse_action("unknown"), None);
        assert_eq!(KeyConfig::parse_action("toggle_flip_diff"), None);
    }
}
