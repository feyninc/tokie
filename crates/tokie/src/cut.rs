//! Which pretokie cut produces the spans.

use pretokie::Span;

/// A split pretokie already implements. `None` is one span, the whole string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cut {
    Gpt2,
    SmolLM,
    Cl100k,
    Olmo,
    Voyage,
    Qwen,
    O200k,
    Nemotron,
    Kimi,
    DeepSeek,
    Bert,
    T5,
    None,
}

impl Cut {
    pub const ALL: [Cut; 13] = [
        Cut::Gpt2,
        Cut::SmolLM,
        Cut::Cl100k,
        Cut::Olmo,
        Cut::Voyage,
        Cut::Qwen,
        Cut::O200k,
        Cut::Nemotron,
        Cut::Kimi,
        Cut::DeepSeek,
        Cut::Bert,
        Cut::T5,
        Cut::None,
    ];

    /// Append every piece of `text`.
    pub fn split(self, text: &str, spans: &mut Vec<Span>) {
        spans.clear();
        self.drive(text, |batch| spans.extend_from_slice(batch));
    }

    /// Call `each` with every batch of pieces. A batch is at most 64 spans.
    pub fn drive(self, text: &str, mut each: impl FnMut(&[Span])) {
        if text.is_empty() {
            return;
        }
        let mut batch = [Span { start: 0, end: 0 }; 64];
        match self {
            Cut::None => each(&[Span {
                start: 0,
                end: u32::try_from(text.len()).expect("text is longer than 4 GiB"),
            }]),
            Cut::Gpt2 => {
                let mut walk = pretokie::Gpt2::walk(text);
                batches(|buf| walk.fill(buf), &mut batch, &mut each);
            }
            Cut::SmolLM => {
                let mut walk = pretokie::SmolLM::walk(text);
                batches(|buf| walk.fill(buf), &mut batch, &mut each);
            }
            Cut::Cl100k => {
                let mut walk = pretokie::Cl100k::walk(text);
                batches(|buf| walk.fill(buf), &mut batch, &mut each);
            }
            Cut::Olmo => {
                let mut walk = pretokie::Olmo::walk(text);
                batches(|buf| walk.fill(buf), &mut batch, &mut each);
            }
            Cut::Voyage => {
                let mut walk = pretokie::Voyage::walk(text);
                batches(|buf| walk.fill(buf), &mut batch, &mut each);
            }
            Cut::Qwen => {
                let mut walk = pretokie::Qwen::walk(text);
                batches(|buf| walk.fill(buf), &mut batch, &mut each);
            }
            Cut::O200k => {
                let mut walk = pretokie::O200k::walk(text);
                batches(|buf| walk.fill(buf), &mut batch, &mut each);
            }
            Cut::Nemotron => {
                let mut walk = pretokie::Nemotron::walk(text);
                batches(|buf| walk.fill(buf), &mut batch, &mut each);
            }
            Cut::Kimi => {
                let mut walk = pretokie::Kimi::walk(text);
                batches(|buf| walk.fill(buf), &mut batch, &mut each);
            }
            Cut::DeepSeek => {
                let mut walk = pretokie::DeepSeek::walk(text);
                batches(|buf| walk.fill(buf), &mut batch, &mut each);
            }
            Cut::Bert => {
                let mut walk = pretokie::Bert::walk(text);
                batches(|buf| walk.fill(buf), &mut batch, &mut each);
            }
            Cut::T5 => {
                let mut walk = pretokie::T5::walk(text);
                batches(|buf| walk.fill(buf), &mut batch, &mut each);
            }
        }
    }

    /// Call `each(false, batch)` after a batch is filled, then walk the next
    /// batch, then `each(true, batch)`. The encode loop prefetches on the
    /// first call and probes on the second, so the next walk covers the
    /// prefetch.
    pub fn pipeline(self, text: &str, each: impl FnMut(bool, &[Span])) {
        if text.is_empty() {
            return;
        }
        match self {
            Cut::None => {
                let span = [Span {
                    start: 0,
                    end: u32::try_from(text.len()).expect("text is longer than 4 GiB"),
                }];
                let mut each = each;
                each(false, &span);
                each(true, &span);
            }
            Cut::Gpt2 => {
                let mut walk = pretokie::Gpt2::walk(text);
                pipeline_fill(|buf| walk.fill(buf), each);
            }
            Cut::SmolLM => {
                let mut walk = pretokie::SmolLM::walk(text);
                pipeline_fill(|buf| walk.fill(buf), each);
            }
            Cut::Cl100k => {
                let mut walk = pretokie::Cl100k::walk(text);
                pipeline_fill(|buf| walk.fill(buf), each);
            }
            Cut::Olmo => {
                let mut walk = pretokie::Olmo::walk(text);
                pipeline_fill(|buf| walk.fill(buf), each);
            }
            Cut::Voyage => {
                let mut walk = pretokie::Voyage::walk(text);
                pipeline_fill(|buf| walk.fill(buf), each);
            }
            Cut::Qwen => {
                let mut walk = pretokie::Qwen::walk(text);
                pipeline_fill(|buf| walk.fill(buf), each);
            }
            Cut::O200k => {
                let mut walk = pretokie::O200k::walk(text);
                pipeline_fill(|buf| walk.fill(buf), each);
            }
            Cut::Nemotron => {
                let mut walk = pretokie::Nemotron::walk(text);
                pipeline_fill(|buf| walk.fill(buf), each);
            }
            Cut::Kimi => {
                let mut walk = pretokie::Kimi::walk(text);
                pipeline_fill(|buf| walk.fill(buf), each);
            }
            Cut::DeepSeek => {
                let mut walk = pretokie::DeepSeek::walk(text);
                pipeline_fill(|buf| walk.fill(buf), each);
            }
            Cut::Bert => {
                let mut walk = pretokie::Bert::walk(text);
                pipeline_fill(|buf| walk.fill(buf), each);
            }
            Cut::T5 => {
                let mut walk = pretokie::T5::walk(text);
                pipeline_fill(|buf| walk.fill(buf), each);
            }
        }
    }
}

fn batches(
    mut fill: impl FnMut(&mut [Span]) -> usize,
    batch: &mut [Span],
    each: &mut impl FnMut(&[Span]),
) {
    loop {
        let n = fill(batch);
        if n == 0 {
            break;
        }
        each(&batch[..n]);
        if n < batch.len() {
            break;
        }
    }
}

/// `each(false)` prefetches the batch just filled. The next `fill` runs
/// before `each(true)` probes it.
fn pipeline_fill(
    mut fill: impl FnMut(&mut [Span]) -> usize,
    mut each: impl FnMut(bool, &[Span]),
) {
    let mut current = [Span { start: 0, end: 0 }; 64];
    let mut next = [Span { start: 0, end: 0 }; 64];
    let mut n = fill(&mut current);
    if n == 0 {
        return;
    }
    each(false, &current[..n]);
    loop {
        let m = fill(&mut next);
        each(true, &current[..n]);
        if m == 0 {
            break;
        }
        each(false, &next[..m]);
        std::mem::swap(&mut current, &mut next);
        n = m;
        if m < current.len() {
            each(true, &current[..n]);
            break;
        }
    }
}
