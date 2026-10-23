//! Which normalie formula rewrites the string.
//!
//! `Precompiled` is not here. It holds a charsmap, so the model owns it.

pub use crate::prepare::{apply, split};

/// A zero-sized normalie formula.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    Nmt,
    Lower,
    Clean,
    BertCased,
    BertUncased,
    StripAccents,
    Albert,
    Llama,
    Metaspace,
    MetaspaceFirst,
    MetaspaceNever,
    MetaspaceWords,
    ByteLevel,
    Strip,
    CollapseSpaces,
    ReplaceSpace,
    Prepend,
}

impl Norm {
    /// `true` when `out` holds the rewrite. `false` leaves `out` empty.
    pub fn write(self, text: &str, at_start: bool, out: &mut String) -> bool {
        match self {
            Norm::Nfc => normalie::Nfc::write(text, at_start, out),
            Norm::Nfd => normalie::Nfd::write(text, at_start, out),
            Norm::Nfkc => normalie::Nfkc::write(text, at_start, out),
            Norm::Nfkd => normalie::Nfkd::write(text, at_start, out),
            Norm::Nmt => normalie::Nmt::write(text, at_start, out),
            Norm::Lower => normalie::Lower::write(text, at_start, out),
            Norm::Clean => normalie::Clean::write(text, at_start, out),
            Norm::BertCased => normalie::BertCased::write(text, at_start, out),
            Norm::BertUncased => normalie::BertUncased::write(text, at_start, out),
            Norm::StripAccents => normalie::StripAccents::write(text, at_start, out),
            Norm::Albert => normalie::Albert::write(text, at_start, out),
            Norm::Llama => normalie::Llama::write(text, at_start, out),
            Norm::Metaspace => normalie::Metaspace::write(text, at_start, out),
            Norm::MetaspaceFirst => normalie::MetaspaceFirst::write(text, at_start, out),
            Norm::MetaspaceNever => normalie::MetaspaceNever::write(text, at_start, out),
            Norm::MetaspaceWords => normalie::MetaspaceWords::write(text, at_start, out),
            Norm::ByteLevel => normalie::ByteLevel::write(text, at_start, out),
            Norm::Strip => normalie::Strip::write(text, at_start, out),
            Norm::CollapseSpaces => normalie::CollapseSpaces::write(text, at_start, out),
            Norm::ReplaceSpace => normalie::ReplaceSpace::write(text, at_start, out),
            Norm::Prepend => normalie::Prepend::write(text, at_start, out),
        }
    }
}
