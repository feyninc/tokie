//! NFC, NFD, NFKC, and NFKD.
//!
//! ASCII is already in each form, so the window copies it. NFC of a long
//! string scans past starters that are already composed. Anything else is
//! checked with ICU. A clean string is not copied.

use crate::form::{self, Kind};
use crate::formula::{Formula, Window};
use crate::scan::ascii_prefix;

pub type Nfc = Uni<0>;
pub type Nfd = Uni<1>;
pub type Nfkc = Uni<2>;
pub type Nfkd = Uni<3>;

pub struct Uni<const K: u8>;

impl<const K: u8> Uni<K> {
    fn kind() -> Kind {
        match K {
            0 => Kind::Nfc,
            1 => Kind::Nfd,
            2 => Kind::Nfkc,
            _ => Kind::Nfkd,
        }
    }

    #[inline(always)]
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl<const K: u8> Formula for Uni<K> {
    #[inline(always)]
    fn drive(&self, text: &str, _at_start: bool, out: &mut String) -> bool {
        if text.is_empty() || text.is_ascii() {
            return false;
        }
        let clean = if K == 0 {
            form::nfc_yes(text)
        } else if K == 1 {
            icu_normalizer::DecomposingNormalizerBorrowed::new_nfd().is_normalized(text)
        } else {
            form::is_clean(Self::kind(), text)
        };
        if clean {
            return false;
        }
        out.reserve(text.len());
        form::emit(Self::kind(), text, out);
        true
    }

    fn unchanged(&self, text: &str, _at_start: bool) -> Option<usize> {
        if form::is_clean(Self::kind(), text) {
            Some(text.len())
        } else {
            None
        }
    }

    fn write_scalar(&self, text: &str, _at_start: bool, out: &mut String) -> usize {
        form::emit(Self::kind(), text, out);
        text.len()
    }

    #[inline(always)]
    fn try_window<const ISA: u8>(&self, bytes: &[u8], _at_start: bool) -> Option<Window> {
        let run = ascii_prefix::<ISA>(bytes);
        if run == 0 {
            return None;
        }
        let followed = bytes.get(run).is_some_and(|b| *b >= 0x80);
        let take = run - usize::from(followed);
        if take == 0 {
            None
        } else {
            Some(Window::Copy(take))
        }
    }
}
