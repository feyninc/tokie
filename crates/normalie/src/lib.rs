//! Normalizers for tokie.
//!
//! Each formula rewrites text into a caller-owned buffer. `false` leaves the
//! buffer empty: the text was already in the formula's form. Pretokie splits
//! the string this crate returns.

mod charsmap;
mod form;
mod formula;
mod impls;
mod kernel;
mod marks;
mod scan;
mod scratch;

pub use charsmap::CharsmapError;
pub use impls::{
    refuse, Albert, BertCased, BertUncased, ByteLevel, Clean, CollapseSpaces, Llama, Lower,
    Metaspace, MetaspaceFirst, MetaspaceNever, MetaspaceWords, Nfc, Nfd, Nfkc, Nfkd, Nmt,
    Precompiled, Prepend, PunctPad, ReplaceSpace, Strip, StripAccents, UnsupportedNormalizer,
};
pub use scratch::Scratch;
