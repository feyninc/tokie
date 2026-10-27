//! Normalize, then fill pretokie spans.

use crate::cut::Cut;
use crate::norm::Norm;
use pretokie::Span;

/// Run `steps` left to right. The result borrows `text` when every step
/// returns `false`, otherwise one of the two buffers.
pub fn apply<'a>(steps: &[Norm], text: &'a str, a: &'a mut String, b: &'a mut String) -> &'a str {
    // 0 reads `text`, 1 reads `a`, 2 reads `b`.
    let mut src = 0u8;
    for step in steps {
        let changed = match src {
            0 => {
                a.clear();
                step.write(text, true, a)
            }
            1 => {
                b.clear();
                step.write(a.as_str(), true, b)
            }
            _ => {
                a.clear();
                step.write(b.as_str(), true, a)
            }
        };
        if changed {
            src = if src == 1 { 2 } else { 1 };
        }
    }
    match src {
        0 => text,
        1 => a.as_str(),
        _ => b.as_str(),
    }
}

/// Replace `spans` with the pieces of `text`.
pub fn split(cut: Cut, text: &str, spans: &mut Vec<Span>) {
    cut.split(text, spans);
}
