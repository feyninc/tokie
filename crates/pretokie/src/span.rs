/// Start and end byte offsets into the caller's text.
///
/// `start` is inclusive and `end` is exclusive. Both are `u32`, so the text
/// is at most 4 GiB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}
