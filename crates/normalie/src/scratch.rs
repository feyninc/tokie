//! Two buffers for a chain of formulas.
//!
//! A step that returns `false` leaves the next step reading the previous
//! string. `apply2` and `apply3` borrow one of the buffers, or `text` when
//! every step returns `false`.

/// Caller-owned scratch for a stacked normalizer.
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    a: String,
    b: String,
}

impl Scratch {
    /// Empty buffers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run two steps. `out` borrows `text` or one of the two buffers.
    pub fn apply2<'a>(
        &'a mut self,
        text: &'a str,
        at_start: bool,
        first: impl FnOnce(&str, bool, &mut String) -> bool,
        second: impl FnOnce(&str, bool, &mut String) -> bool,
    ) -> &'a str {
        self.a.clear();
        let first_changed = first(text, at_start, &mut self.a);
        self.b.clear();
        if first_changed {
            if second(self.a.as_str(), at_start, &mut self.b) {
                self.b.as_str()
            } else {
                self.a.as_str()
            }
        } else if second(text, at_start, &mut self.b) {
            self.b.as_str()
        } else {
            text
        }
    }

    /// Run three steps, swapping the two buffers.
    pub fn apply3<'a>(
        &'a mut self,
        text: &'a str,
        at_start: bool,
        first: impl FnOnce(&str, bool, &mut String) -> bool,
        second: impl FnOnce(&str, bool, &mut String) -> bool,
        third: impl FnOnce(&str, bool, &mut String) -> bool,
    ) -> &'a str {
        self.a.clear();
        let c1 = first(text, at_start, &mut self.a);
        self.b.clear();
        if c1 {
            let c2 = second(self.a.as_str(), at_start, &mut self.b);
            if c2 {
                self.a.clear();
                if third(self.b.as_str(), at_start, &mut self.a) {
                    self.a.as_str()
                } else {
                    self.b.as_str()
                }
            } else {
                self.b.clear();
                if third(self.a.as_str(), at_start, &mut self.b) {
                    self.b.as_str()
                } else {
                    self.a.as_str()
                }
            }
        } else {
            let c2 = second(text, at_start, &mut self.b);
            if c2 {
                self.a.clear();
                if third(self.b.as_str(), at_start, &mut self.a) {
                    self.a.as_str()
                } else {
                    self.b.as_str()
                }
            } else if third(text, at_start, &mut self.b) {
                self.b.as_str()
            } else {
                text
            }
        }
    }
}
