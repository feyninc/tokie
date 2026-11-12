#![doc = include_str!("../README.md")]

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
