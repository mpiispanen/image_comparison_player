use std::collections::BTreeMap;

/// A single annotation marker attached to a frame index.
#[derive(Clone, Debug)]
pub struct Annotation {
    pub frame: usize,
    pub note: String,
}

/// Stores all annotation markers keyed by frame index (sorted).
pub struct AnnotationStore {
    pub annotations: BTreeMap<usize, Annotation>,
}

impl Default for AnnotationStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AnnotationStore {
    pub fn new() -> Self {
        Self {
            annotations: BTreeMap::new(),
        }
    }

    /// Toggle the marker on `frame`: adds it if absent, removes it if present.
    /// Returns `true` when the marker was added.
    pub fn toggle(&mut self, frame: usize) -> bool {
        if let std::collections::btree_map::Entry::Vacant(e) = self.annotations.entry(frame) {
            e.insert(Annotation { frame, note: String::new() });
            true
        } else {
            self.annotations.remove(&frame);
            false
        }
    }

    pub fn has(&self, frame: usize) -> bool {
        self.annotations.contains_key(&frame)
    }

    /// Update the note for `frame` (no-op if the frame has no marker).
    pub fn set_note(&mut self, frame: usize, note: String) {
        if let Some(a) = self.annotations.get_mut(&frame) {
            a.note = note;
        }
    }

    /// Return the note text for `frame`, if a marker exists.
    pub fn get_note(&self, frame: usize) -> Option<&str> {
        self.annotations.get(&frame).map(|a| a.note.as_str())
    }

    pub fn remove(&mut self, frame: usize) {
        self.annotations.remove(&frame);
    }

    /// Return the next annotated frame after `current`, wrapping around.
    pub fn next_marker(&self, current: usize) -> Option<usize> {
        if self.annotations.is_empty() {
            return None;
        }
        self.annotations
            .keys()
            .find(|&&f| f > current)
            .copied()
            .or_else(|| self.annotations.keys().next().copied())
    }

    /// Return the previous annotated frame before `current`, wrapping around.
    pub fn prev_marker(&self, current: usize) -> Option<usize> {
        if self.annotations.is_empty() {
            return None;
        }
        self.annotations
            .keys()
            .rev()
            .find(|&&f| f < current)
            .copied()
            .or_else(|| self.annotations.keys().next_back().copied())
    }

    /// Serialize all annotations to a JSON string.
    pub fn export_json(&self) -> String {
        let entries: Vec<String> = self
            .annotations
            .values()
            .map(|a| {
                format!(
                    "{{\"frame\":{},\"note\":\"{}\"}}",
                    a.frame,
                    json_escape(&a.note)
                )
            })
            .collect();
        format!("{{\"annotations\":[{}]}}", entries.join(","))
    }

    /// Parse a JSON string and merge the contained annotations into this store.
    /// Returns the number of annotations loaded, or an error description.
    pub fn import_json(&mut self, json: &str) -> Result<usize, String> {
        let parsed = parse_annotations_json(json)?;
        let count = parsed.len();
        for a in parsed {
            self.annotations.insert(a.frame, a);
        }
        Ok(count)
    }

    /// Load annotations from a JSON file, merging them into this store.
    pub fn load_from_file(&mut self, path: &str) -> Result<usize, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read '{}': {}", path, e))?;
        self.import_json(&contents)
    }

    /// Write the current annotations to a JSON file.
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, self.export_json())
            .map_err(|e| format!("Failed to write '{}': {}", path, e))
    }
}

// ── JSON helpers ─────────────────────────────────────────────────────────────

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

fn json_unescape(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('"') => result.push('"'),
                Some('\\') => result.push('\\'),
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }
    result
}

fn parse_annotations_json(json: &str) -> Result<Vec<Annotation>, String> {
    use regex::Regex;
    use std::sync::OnceLock;

    // Compile the regex once; reuse on subsequent calls.
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(
            r#"\{"frame"\s*:\s*(\d+)\s*,\s*"note"\s*:\s*"((?:[^"\\]|\\.)*)"\}"#,
        )
        .expect("annotation JSON regex is valid")
    });

    let mut results = Vec::new();
    for cap in re.captures_iter(json) {
        let frame: usize = cap[1]
            .parse()
            .map_err(|e: std::num::ParseIntError| e.to_string())?;
        let note = json_unescape(&cap[2]);
        results.push(Annotation { frame, note });
    }
    Ok(results)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toggle_add_remove() {
        let mut store = AnnotationStore::new();
        assert!(!store.has(0));
        assert!(store.toggle(0), "should return true when adding");
        assert!(store.has(0));
        assert!(!store.toggle(0), "should return false when removing");
        assert!(!store.has(0));
    }

    #[test]
    fn test_set_get_note() {
        let mut store = AnnotationStore::new();
        store.toggle(3);
        store.set_note(3, "hello".to_string());
        assert_eq!(store.get_note(3), Some("hello"));
        // set_note on a missing frame is a no-op
        store.set_note(99, "ignored".to_string());
        assert_eq!(store.get_note(99), None);
    }

    #[test]
    fn test_remove() {
        let mut store = AnnotationStore::new();
        store.toggle(5);
        store.remove(5);
        assert!(!store.has(5));
        // removing a non-existent frame is a no-op
        store.remove(5);
    }

    #[test]
    fn test_navigation_next_prev() {
        let mut store = AnnotationStore::new();
        store.toggle(0);
        store.toggle(5);
        store.toggle(10);

        assert_eq!(store.next_marker(0), Some(5));
        assert_eq!(store.next_marker(5), Some(10));
        assert_eq!(store.next_marker(10), Some(0), "should wrap around");

        assert_eq!(store.prev_marker(10), Some(5));
        assert_eq!(store.prev_marker(5), Some(0));
        assert_eq!(store.prev_marker(0), Some(10), "should wrap around");
    }

    #[test]
    fn test_navigation_single_marker() {
        let mut store = AnnotationStore::new();
        store.toggle(7);
        // Only one marker; next/prev wraps back to itself.
        assert_eq!(store.next_marker(7), Some(7));
        assert_eq!(store.prev_marker(7), Some(7));
    }

    #[test]
    fn test_navigation_empty() {
        let store = AnnotationStore::new();
        assert_eq!(store.next_marker(0), None);
        assert_eq!(store.prev_marker(0), None);
    }

    #[test]
    fn test_export_json_empty() {
        let store = AnnotationStore::new();
        assert_eq!(store.export_json(), r#"{"annotations":[]}"#);
    }

    #[test]
    fn test_export_json_with_annotations() {
        let mut store = AnnotationStore::new();
        store.toggle(0);
        store.set_note(0, "first".to_string());
        store.toggle(3);
        store.set_note(3, "second".to_string());

        let json = store.export_json();
        assert!(json.contains(r#""frame":0"#));
        assert!(json.contains(r#""note":"first""#));
        assert!(json.contains(r#""frame":3"#));
        assert!(json.contains(r#""note":"second""#));
    }

    #[test]
    fn test_export_json_escaping() {
        let mut store = AnnotationStore::new();
        store.toggle(1);
        store.set_note(1, "line1\nline2\t\"quoted\"".to_string());
        let json = store.export_json();
        assert!(json.contains(r#"line1\nline2\t\"quoted\""#));
    }

    #[test]
    fn test_import_json_roundtrip() {
        let mut store = AnnotationStore::new();
        store.toggle(2);
        store.set_note(2, "note with\nnewline".to_string());
        store.toggle(8);

        let json = store.export_json();

        let mut store2 = AnnotationStore::new();
        let count = store2.import_json(&json).unwrap();
        assert_eq!(count, 2);
        assert!(store2.has(2));
        assert!(store2.has(8));
        assert_eq!(store2.get_note(2), Some("note with\nnewline"));
        assert_eq!(store2.get_note(8), Some(""));
    }

    #[test]
    fn test_import_json_merges() {
        let mut store = AnnotationStore::new();
        store.toggle(1);
        store.set_note(1, "existing".to_string());

        let json = r#"{"annotations":[{"frame":2,"note":"new"}]}"#;
        let count = store.import_json(json).unwrap();
        assert_eq!(count, 1);
        assert!(store.has(1), "existing annotation preserved");
        assert!(store.has(2), "imported annotation added");
    }

    #[test]
    fn test_import_json_invalid_returns_empty() {
        let mut store = AnnotationStore::new();
        // Completely invalid JSON — regex finds no matches, returns 0 annotations.
        let count = store.import_json("not json at all").unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_save_load_file() {
        let dir = std::env::temp_dir()
            .join("icp_tests")
            .join("annotations_save_load");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("annotations.json");

        let mut store = AnnotationStore::new();
        store.toggle(0);
        store.set_note(0, "saved note".to_string());
        store.save_to_file(path.to_str().unwrap()).unwrap();

        let mut store2 = AnnotationStore::new();
        let count = store2.load_from_file(path.to_str().unwrap()).unwrap();
        assert_eq!(count, 1);
        assert_eq!(store2.get_note(0), Some("saved note"));
    }

    #[test]
    fn test_load_file_missing() {
        let mut store = AnnotationStore::new();
        let result = store.load_from_file("/nonexistent/path/annotations.json");
        assert!(result.is_err());
    }
}
