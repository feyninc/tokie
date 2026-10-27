//! BERT WordPiece. The longest vocabulary prefix wins. A miss on the first
//! character makes the whole word `unk`.

/// Append the ids for `word`. `prefix` is the continuation mark, `##` for BERT.
/// `scratch` holds one continuation candidate at a time.
pub fn wordpiece(
    word: &str,
    unk: u32,
    max_chars: usize,
    prefix: &str,
    scratch: &mut String,
    mut vocab: impl FnMut(&str) -> Option<u32>,
    out: &mut Vec<u32>,
) {
    if word.chars().count() > max_chars {
        out.push(unk);
        return;
    }
    let bytes = word.as_bytes();
    let mut ids = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        let mut end = bytes.len();
        let mut found = None;
        while end > start {
            if !word.is_char_boundary(start) || !word.is_char_boundary(end) {
                end -= 1;
                continue;
            }
            let piece = if start == 0 {
                &word[start..end]
            } else {
                scratch.clear();
                scratch.push_str(prefix);
                scratch.push_str(&word[start..end]);
                scratch.as_str()
            };
            if let Some(id) = vocab(piece) {
                found = Some((end, id));
                break;
            }
            end -= 1;
        }
        let Some((next, id)) = found else {
            out.push(unk);
            return;
        };
        ids.push(id);
        start = next;
    }
    out.extend_from_slice(&ids);
}
