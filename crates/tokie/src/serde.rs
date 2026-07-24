//! Binary serialization for fast tokenizer loading.
//!
//! This module provides efficient save/load functionality using a custom binary format
//! that stores pre-built DAAC state, eliminating the need to rebuild the automaton.
//!
//! # File Format
//!
//! ```text
//! Header (88 bytes):
//!   - magic: "TOKI" (4 bytes)
//!   - version: u32 (4 bytes) - currently v12
//!   - encoder_type: u32 (4 bytes) - 0=Backtracking, 1=Simple, 2=WordPiece
//!   - pretokenizer_type: u32 (4 bytes) - 0=None, 1=GPT2, 2=CL100K, 3=O200K, 4=BERT, 5=Voyage
//!   - normalizer_type: u32 (4 bytes) - 0=None, 1=BertUncased, 2=BertCased, 3=Nfc
//!   - post_processor_type: u32 (4 bytes) - 0=None, 1=Bert, 2=Prefix, 3=Template
//!   - vocab_size: u32 (4 bytes)
//!   - num_merges: u32 (4 bytes)
//!   - num_base_tokens: u32 (4 bytes)
//!   - pad_token_id: u32 (4 bytes) - 0xFFFFFFFF = None (v11+)
//!   - token_data_offset: u32, token_data_checksum: u32
//!   - merge_data_offset: u32, merge_data_checksum: u32
//!   - daac_data_offset: u32, daac_data_checksum: u32
//!   - prefix_data_offset: u32, prefix_data_checksum: u32
//!   - pp_data_offset: u32, pp_data_checksum: u32
//!   - charsmap_data_offset: u32, charsmap_data_checksum: u32 (v12+; padding before)
//!
//! Sections:
//!   - TOKEN_DATA: Decoder's flat buffer (offsets + data)
//!   - MERGE_DATA: split_table as raw bytes
//!   - DAAC_DATA: Pre-built DoubleArrayAhoCorasick state (empty for Simple encoder)
//!   - PREFIX_DATA: next_prefix_match table (empty for Simple encoder)
//!   - PP_DATA: Post-processor parameters (empty for None)
//!   - CHARSMAP_DATA: Raw SentencePiece precompiled_charsmap blob (v12+, empty
//!     unless normalizer is SentencePiecePrecompiled)
//! ```

use core::mem::size_of;
use std::io::{Read, Write};

use crate::charsmap::PrecompiledCharsmap;
use crate::encoder::{BacktrackingBytePairEncoder, BytePairEncoder, Encoder, EncoderType, SentencePieceBPE, UnigramEncoder, WordPieceEncoder};
use crate::decoder::{Decoder, DecoderType, VocabDecoder};
use crate::normalizer::Normalizer;
use crate::postprocessor::PostProcessor;
use crate::pretok::PretokType;
use crate::tokenizer::{AddedTokenSpec, Tokenizer};
use crate::types::{Split, TokenId};
use daggrs::DoubleArrayAhoCorasick;
use foldhash::HashMap as FoldHashMap;

const MAGIC: &[u8; 4] = b"TOKI";
/// Current .tkz format version. Public so cache layers can key artifacts by it.
/// v12 added CHARSMAP_DATA; v13 adds ADDED_TOKENS (files are self-contained —
/// no tokenizer.json fetch needed for added/special tokens).
pub(crate) const VERSION: u32 = 13;
const HEADER_SIZE: usize = 88;
/// v13 grows the header by one (offset, checksum) pair for ADDED_TOKENS.
const HEADER_SIZE_V13: usize = 96;

impl PretokType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Gpt2),
            2 => Some(Self::Cl100k),
            3 => Some(Self::O200k),
            4 => Some(Self::Bert),
            5 => Some(Self::Voyage),
            6 => Some(Self::DeepSeek),
            7 => Some(Self::SmolLM),
            8 => Some(Self::Qwen35),
            _ => None,
        }
    }
}

impl Normalizer {
    /// Charsmap-backed ids (8, 9, 12) are not constructible here — they
    /// need the CHARSMAP_DATA section, handled in `Tokenizer::load`.
    fn from_u32(v: u32) -> Option<Self> {
        use crate::normalizer::MetaspacePrepend;
        match v {
            0 => Some(Self::None),
            1 => Some(Self::BertUncased),
            2 => Some(Self::BertCased),
            3 => Some(Self::Nfc),
            4 => Some(Self::Metaspace(MetaspacePrepend::Unconditional)),
            5 => Some(Self::SentencePiece),
            6 => Some(Self::SentencePieceLowercase),
            7 => Some(Self::MetaspaceReplace),
            10 => Some(Self::Metaspace(MetaspacePrepend::IfNotSpaceLed)),
            11 => Some(Self::Metaspace(MetaspacePrepend::FirstSegment)),
            _ => None,
        }
    }

    fn to_u32(&self) -> u32 {
        use crate::normalizer::MetaspacePrepend;
        match self {
            Self::None => 0,
            Self::BertUncased => 1,
            Self::BertCased => 2,
            Self::Nfc => 3,
            Self::Metaspace(MetaspacePrepend::Unconditional) => 4,
            Self::SentencePiece => 5,
            Self::SentencePieceLowercase => 6,
            Self::MetaspaceReplace => 7,
            Self::SentencePiecePrecompiled { whitespace_split: true, .. } => 8,
            Self::SentencePiecePrecompiled { whitespace_split: false, .. } => 9,
            Self::Metaspace(MetaspacePrepend::IfNotSpaceLed) => 10,
            Self::Metaspace(MetaspacePrepend::FirstSegment) => 11,
            Self::SentencePiecePunctPad { .. } => 12,
        }
    }
}

/// Precompiled-charsmap normalizer ids (need the CHARSMAP_DATA section):
/// 8 = with WhitespaceSplit (XLM-R/T5), 9 = Metaspace-only (bge-m3 family),
/// 12 = punctuation-padding chain (potion-multilingual).
const NORMALIZER_SP_PRECOMPILED_WS: u32 = 8;
const NORMALIZER_SP_PRECOMPILED_META: u32 = 9;
const NORMALIZER_SP_PUNCT_PAD: u32 = 12;

impl PostProcessor {
    fn type_id(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Bert { .. } => 1,
            Self::Prefix { .. } => 2,
            Self::Template { .. } => 3,
        }
    }

    fn serialize(&self) -> Vec<u8> {
        match self {
            Self::None => Vec::new(),
            Self::Bert { cls_token, sep_token } => {
                let mut buf = Vec::with_capacity(8);
                buf.extend_from_slice(&cls_token.to_le_bytes());
                buf.extend_from_slice(&sep_token.to_le_bytes());
                buf
            }
            Self::Prefix { bos_token } => {
                bos_token.to_le_bytes().to_vec()
            }
            Self::Template {
                single_prefix,
                single_suffix,
                pair_a_prefix,
                pair_a_suffix,
                pair_b_prefix,
                pair_b_suffix,
            } => {
                // Format: 6 length-prefixed arrays of u32 tokens
                let mut buf = Vec::new();
                for tokens in [
                    single_prefix,
                    single_suffix,
                    pair_a_prefix,
                    pair_a_suffix,
                    pair_b_prefix,
                    pair_b_suffix,
                ] {
                    buf.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
                    for &token in tokens {
                        buf.extend_from_slice(&token.to_le_bytes());
                    }
                }
                buf
            }
        }
    }

    fn deserialize(type_id: u32, data: &[u8]) -> Option<Self> {
        match type_id {
            0 => Some(Self::None),
            1 => {
                if data.len() < 8 {
                    return None;
                }
                let cls_token = u32::from_le_bytes(data[0..4].try_into().ok()?);
                let sep_token = u32::from_le_bytes(data[4..8].try_into().ok()?);
                Some(Self::Bert { cls_token, sep_token })
            }
            2 => {
                if data.len() < 4 {
                    return None;
                }
                let bos_token = u32::from_le_bytes(data[0..4].try_into().ok()?);
                Some(Self::Prefix { bos_token })
            }
            3 => {
                // Parse 6 length-prefixed arrays
                let mut offset = 0;
                let mut arrays = Vec::new();
                for _ in 0..6 {
                    if offset + 4 > data.len() {
                        return None;
                    }
                    let len = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
                    offset += 4;
                    let mut tokens = Vec::with_capacity(len);
                    for _ in 0..len {
                        if offset + 4 > data.len() {
                            return None;
                        }
                        tokens.push(u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?));
                        offset += 4;
                    }
                    arrays.push(tokens);
                }
                Some(Self::Template {
                    single_prefix: arrays.remove(0),
                    single_suffix: arrays.remove(0),
                    pair_a_prefix: arrays.remove(0),
                    pair_a_suffix: arrays.remove(0),
                    pair_b_prefix: arrays.remove(0),
                    pair_b_suffix: arrays.remove(0),
                })
            }
            _ => None,
        }
    }
}

/// Fast CRC32 checksum using hardware acceleration when available.
fn crc32(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

/// Error type for serialization/deserialization.
#[derive(Debug)]
pub enum SerdeError {
    Io(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    InvalidEncoderType(u32),
    InvalidPretokenizer(u32),
    InvalidNormalizer(u32),
    InvalidPostProcessor(u32),
    ChecksumMismatch { section: &'static str },
    InvalidData(&'static str),
}

impl std::fmt::Display for SerdeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::InvalidMagic => write!(f, "Invalid magic bytes (not a TOKI file)"),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported version: {}", v),
            Self::InvalidEncoderType(v) => write!(f, "Invalid encoder type: {}", v),
            Self::InvalidPretokenizer(v) => write!(f, "Invalid pretokenizer type: {}", v),
            Self::InvalidNormalizer(v) => write!(f, "Invalid normalizer type: {}", v),
            Self::InvalidPostProcessor(v) => write!(f, "Invalid post-processor type: {}", v),
            Self::ChecksumMismatch { section } => write!(f, "Checksum mismatch in {}", section),
            Self::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
        }
    }
}

impl std::error::Error for SerdeError {}

impl From<std::io::Error> for SerdeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl Tokenizer {
    /// Save the tokenizer to a file.
    ///
    /// This saves the pre-built DAAC state, enabling fast loading without
    /// rebuilding the automaton.
    pub fn to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), SerdeError> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        self.save(&mut writer)
    }

    /// Save the tokenizer to a writer.
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<(), SerdeError> {
        let encoder_type = self.encoder_type();
        let pretokenizer_type = self.pretokenizer_type();
        let normalizer = self.normalizer();
        let post_processor = self.post_processor();
        let encoder = self.encoder();
        let decoder = self.decoder();

        // Serialize sections based on encoder type
        let token_data = serialize_vocab_decoder(decoder.vocab());

        // Serialize sections based on encoder type
        let (merge_data, daac_data, prefix_data) = match encoder {
            Encoder::Backtracking(enc) => {
                let merge = serialize_splits(enc.split_table());
                let daac = enc.matcher().serialize();
                let prefix = serialize_prefix_match(enc.next_prefix_match_table());
                (merge, daac, prefix)
            }
            Encoder::Simple(enc) => {
                // Simple encoder: serialize pair_lookup as merges, empty DAAC/prefix
                let merge = serialize_pair_lookup(enc);
                let daac = Vec::new();
                let prefix = Vec::new();
                (merge, daac, prefix)
            }
            Encoder::WordPiece(enc) => {
                // WordPiece: serialize DAAC with anchor, empty merge/prefix
                let merge = serialize_wordpiece_config(enc);
                let daac = enc.matcher().serialize();
                let prefix = Vec::new();
                (merge, daac, prefix)
            }
            Encoder::SentencePiece(enc) => {
                // SentencePiece: serialize pair_lookup as merges, empty DAAC/prefix
                let merge = serialize_sentencepiece_config(enc);
                let daac = Vec::new();
                let prefix = Vec::new();
                (merge, daac, prefix)
            }
            Encoder::Unigram(enc) => {
                // Unigram: serialize scores, unk_token, byte_tokens in merge_data
                // DAAC in daac_data, prefix_data empty
                let merge = serialize_unigram_config(enc);
                let daac = enc.matcher().serialize();
                let prefix = Vec::new();
                (merge, daac, prefix)
            }
        };

        // Serialize post-processor
        let pp_data = post_processor.serialize();

        // Serialize charsmap (raw precompiled_charsmap blob, v12+)
        let charsmap_data: &[u8] = match normalizer {
            Normalizer::SentencePiecePrecompiled { charsmap, .. }
            | Normalizer::SentencePiecePunctPad { charsmap } => charsmap.blob(),
            _ => &[],
        };

        // Compute checksums
        let token_checksum = crc32(&token_data);
        let merge_checksum = crc32(&merge_data);
        let daac_checksum = crc32(&daac_data);
        let prefix_checksum = crc32(&prefix_data);
        let pp_checksum = crc32(&pp_data);
        let charsmap_checksum = crc32(charsmap_data);

        // Compute offsets (after header)
        let token_offset = HEADER_SIZE_V13 as u32;
        let merge_offset = token_offset + token_data.len() as u32;
        let daac_offset = merge_offset + merge_data.len() as u32;
        let prefix_offset = daac_offset + daac_data.len() as u32;
        let pp_offset = prefix_offset + prefix_data.len() as u32;
        let charsmap_offset = pp_offset + pp_data.len() as u32;

        // ADDED_TOKENS payload (v13+): count, then (id u32, flags u8, len u32, bytes).
        // flags bit0 = special. The section is written even when empty: its
        // presence marks the file authoritative, so loaders skip tokenizer.json.
        let added_data = serialize_added_tokens(self.added_tokens_raw(), self.special_tokens());
        let added_checksum = crc32(&added_data);
        let added_offset = charsmap_offset + charsmap_data.len() as u32;

        // Write header (88 bytes total)
        // 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + (5 × 8) = 40 + 40 = 80... need 8 more
        // Actually: magic(4) + version(4) + encoder(4) + pretok(4) + norm(4) + pp_type(4)
        //         + vocab(4) + merges(4) + base(4) + reserved(4) = 40 bytes
        //         + 5 sections × 8 bytes = 40 bytes
        //         Total = 80 bytes... let me recalculate for 88
        // We need: 40 bytes metadata + 5 sections × 8 = 80 bytes
        // For 88: add another reserved u32 (4) + padding (4) = 88
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&(encoder_type as u32).to_le_bytes())?;
        writer.write_all(&(pretokenizer_type as u32).to_le_bytes())?;
        writer.write_all(&normalizer.to_u32().to_le_bytes())?;
        writer.write_all(&post_processor.type_id().to_le_bytes())?;
        writer.write_all(&(decoder.vocab_size() as u32).to_le_bytes())?;
        writer.write_all(&((encoder.vocab_size() - encoder.num_base_tokens()) as u32).to_le_bytes())?;
        writer.write_all(&(encoder.num_base_tokens() as u32).to_le_bytes())?;
        // pad_token_id: 0xFFFFFFFF sentinel means None
        let pad_token_id_raw = self.pad_token_id().unwrap_or(0xFFFF_FFFF);
        writer.write_all(&pad_token_id_raw.to_le_bytes())?;

        // Interleaved offsets and checksums (5 sections × 8 bytes = 40 bytes)
        writer.write_all(&token_offset.to_le_bytes())?;
        writer.write_all(&token_checksum.to_le_bytes())?;
        writer.write_all(&merge_offset.to_le_bytes())?;
        writer.write_all(&merge_checksum.to_le_bytes())?;
        writer.write_all(&daac_offset.to_le_bytes())?;
        writer.write_all(&daac_checksum.to_le_bytes())?;
        writer.write_all(&prefix_offset.to_le_bytes())?;
        writer.write_all(&prefix_checksum.to_le_bytes())?;
        writer.write_all(&pp_offset.to_le_bytes())?;
        writer.write_all(&pp_checksum.to_le_bytes())?;
        // v12: 6th section in the former padding bytes (80..88)
        writer.write_all(&charsmap_offset.to_le_bytes())?;
        writer.write_all(&charsmap_checksum.to_le_bytes())?;
        // v13: 7th section pair extends the header to 96 bytes
        writer.write_all(&added_offset.to_le_bytes())?;
        writer.write_all(&added_checksum.to_le_bytes())?;

        // Write sections
        writer.write_all(&token_data)?;
        writer.write_all(&merge_data)?;
        writer.write_all(&daac_data)?;
        writer.write_all(&prefix_data)?;
        writer.write_all(&pp_data)?;
        writer.write_all(charsmap_data)?;
        writer.write_all(&added_data)?;

        Ok(())
    }

    /// Load a tokenizer from a file.
    ///
    /// This loads pre-built DAAC state for instant use without rebuilding.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, SerdeError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        Self::load(&mut reader)
    }

    /// Load a tokenizer from a reader.
    pub fn load<R: Read>(reader: &mut R) -> Result<Self, SerdeError> {
        // Read entire file
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        if data.len() < HEADER_SIZE {
            return Err(SerdeError::InvalidData("file too small"));
        }

        // Parse header
        if &data[0..4] != MAGIC {
            return Err(SerdeError::InvalidMagic);
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if !(10..=VERSION).contains(&version) {
            return Err(SerdeError::UnsupportedVersion(version));
        }

        let encoder_type = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let encoder_type = EncoderType::from_u32(encoder_type)
            .ok_or(SerdeError::InvalidEncoderType(encoder_type))?;

        let pretokenizer_type = u32::from_le_bytes(data[12..16].try_into().unwrap());
        let pretokenizer_type = PretokType::from_u32(pretokenizer_type)
            .ok_or(SerdeError::InvalidPretokenizer(pretokenizer_type))?;

        let normalizer_type = u32::from_le_bytes(data[16..20].try_into().unwrap());
        // Charsmap-backed normalizers need the CHARSMAP_DATA section; built below.
        let normalizer = if normalizer_type == NORMALIZER_SP_PRECOMPILED_WS
            || normalizer_type == NORMALIZER_SP_PRECOMPILED_META
            || normalizer_type == NORMALIZER_SP_PUNCT_PAD
        {
            None
        } else {
            Some(
                Normalizer::from_u32(normalizer_type)
                    .ok_or(SerdeError::InvalidNormalizer(normalizer_type))?,
            )
        };

        let pp_type = u32::from_le_bytes(data[20..24].try_into().unwrap());

        let vocab_size = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;
        let _num_merges = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;
        let num_base_tokens = u32::from_le_bytes(data[32..36].try_into().unwrap()) as usize;
        // pad_token_id: v11+ stores in bytes 36-40, 0xFFFFFFFF = None; v10 has 0 (treat as None)
        let pad_token_id_raw = u32::from_le_bytes(data[36..40].try_into().unwrap());
        let pad_token_id = if version >= 11 && pad_token_id_raw != 0xFFFF_FFFF {
            Some(pad_token_id_raw)
        } else {
            None
        };

        // Section offsets and checksums (5 sections × 8 bytes = 40 bytes)
        let token_offset = u32::from_le_bytes(data[40..44].try_into().unwrap()) as usize;
        let token_checksum = u32::from_le_bytes(data[44..48].try_into().unwrap());
        let merge_offset = u32::from_le_bytes(data[48..52].try_into().unwrap()) as usize;
        let merge_checksum = u32::from_le_bytes(data[52..56].try_into().unwrap());
        let daac_offset = u32::from_le_bytes(data[56..60].try_into().unwrap()) as usize;
        let daac_checksum = u32::from_le_bytes(data[60..64].try_into().unwrap());
        let prefix_offset = u32::from_le_bytes(data[64..68].try_into().unwrap()) as usize;
        let prefix_checksum = u32::from_le_bytes(data[68..72].try_into().unwrap());
        let pp_offset = u32::from_le_bytes(data[72..76].try_into().unwrap()) as usize;
        let pp_checksum = u32::from_le_bytes(data[76..80].try_into().unwrap());
        // v12+: 6th section in bytes 80..88 (padding in v10/v11)
        let (charsmap_offset, charsmap_checksum) = if version >= 12 {
            (
                u32::from_le_bytes(data[80..84].try_into().unwrap()) as usize,
                u32::from_le_bytes(data[84..88].try_into().unwrap()),
            )
        } else {
            (data.len(), 0)
        };
        // v13+: 7th section pair in bytes 88..96
        let (added_offset, added_checksum) = if version >= 13 {
            if data.len() < HEADER_SIZE_V13 {
                return Err(SerdeError::InvalidData("truncated v13 header"));
            }
            (
                u32::from_le_bytes(data[88..92].try_into().unwrap()) as usize,
                u32::from_le_bytes(data[92..96].try_into().unwrap()),
            )
        } else {
            (data.len(), 0)
        };

        // Extract and verify sections
        let token_data = &data[token_offset..merge_offset];
        if crc32(token_data) != token_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "token_data" });
        }

        let merge_data = &data[merge_offset..daac_offset];
        if crc32(merge_data) != merge_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "merge_data" });
        }

        let daac_data = &data[daac_offset..prefix_offset];
        if crc32(daac_data) != daac_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "daac_data" });
        }

        let prefix_data = &data[prefix_offset..pp_offset];
        if crc32(prefix_data) != prefix_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "prefix_data" });
        }

        if charsmap_offset < pp_offset || charsmap_offset > data.len() {
            return Err(SerdeError::InvalidData("charsmap section out of bounds"));
        }
        let pp_data = &data[pp_offset..charsmap_offset];
        if crc32(pp_data) != pp_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "pp_data" });
        }

        if added_offset < charsmap_offset || added_offset > data.len() {
            return Err(SerdeError::InvalidData("added-tokens section out of bounds"));
        }
        let charsmap_data = &data[charsmap_offset..added_offset];
        if version >= 12 && crc32(charsmap_data) != charsmap_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "charsmap_data" });
        }

        let added_data = &data[added_offset..];
        if version >= 13 && crc32(added_data) != added_checksum {
            return Err(SerdeError::ChecksumMismatch { section: "added_tokens" });
        }

        // Build the normalizer, parsing the charsmap blob if required
        let normalizer = match normalizer {
            Some(n) => n,
            None => {
                let charsmap = PrecompiledCharsmap::from_blob(charsmap_data)
                    .map_err(|_| SerdeError::InvalidData("invalid precompiled charsmap"))?;
                let charsmap = std::sync::Arc::new(charsmap);
                if normalizer_type == NORMALIZER_SP_PUNCT_PAD {
                    Normalizer::SentencePiecePunctPad { charsmap }
                } else {
                    Normalizer::SentencePiecePrecompiled {
                        charsmap,
                        whitespace_split: normalizer_type == NORMALIZER_SP_PRECOMPILED_WS,
                    }
                }
            }
        };

        // Deserialize post-processor
        let post_processor = PostProcessor::deserialize(pp_type, pp_data)
            .ok_or(SerdeError::InvalidPostProcessor(pp_type))?;

        // Deserialize decoder
        let (decoder_offsets, decoder_data) = deserialize_decoder(token_data, vocab_size)?;

        // Build encoder based on type
        // OPTIMIZATION: For Simple/SentencePiece, build lookups directly from decoder
        // without intermediate Vec<Vec<u8>> allocation (4x faster for large vocabs)
        let encoder = match encoder_type {
            EncoderType::Backtracking => {
                // Backtracking still needs token_bytes for now
                let token_bytes: Vec<Vec<u8>> = (0..vocab_size)
                    .map(|i| {
                        let start = decoder_offsets[i] as usize;
                        let end = decoder_offsets[i + 1] as usize;
                        decoder_data[start..end].to_vec()
                    })
                    .collect();

                let split_table = deserialize_splits(merge_data)?;
                let (daac, _) = DoubleArrayAhoCorasick::deserialize(daac_data)
                    .ok_or(SerdeError::InvalidData("failed to deserialize DAAC"))?;
                let next_prefix_match = deserialize_prefix_match(prefix_data)?;

                // Rebuild pair_lookup from split_table
                let pair_lookup = rebuild_pair_lookup(&split_table, num_base_tokens);

                // Extract token lengths from decoder offsets
                let token_lengths: Vec<u8> = (0..vocab_size)
                    .map(|i| {
                        let start = decoder_offsets[i] as usize;
                        let end = decoder_offsets[i + 1] as usize;
                        (end - start).min(255) as u8
                    })
                    .collect();

                let enc = BacktrackingBytePairEncoder::from_parts(
                    split_table,
                    pair_lookup,
                    token_lengths,
                    num_base_tokens,
                    daac,
                    next_prefix_match,
                    &token_bytes,
                );
                Encoder::Backtracking(enc)
            }
            EncoderType::Simple => {
                // OPTIMIZED: Build lookups directly from decoder (single copy)
                // Simple encoder doesn't use token_lengths, so we ignore it
                let (byte_lut, token_cache, _, _) = build_token_lookups(&decoder_offsets, &decoder_data, vocab_size);
                let merges = deserialize_merges(merge_data)?;

                let enc = BytePairEncoder::from_parts(
                    &merges,
                    byte_lut,
                    token_cache,
                    vocab_size,
                    num_base_tokens,
                );
                Encoder::Simple(enc)
            }
            EncoderType::WordPiece => {
                // WordPiece needs token_bytes for continuation prefix matching
                let token_bytes: Vec<Vec<u8>> = (0..vocab_size)
                    .map(|i| {
                        let start = decoder_offsets[i] as usize;
                        let end = decoder_offsets[i + 1] as usize;
                        decoder_data[start..end].to_vec()
                    })
                    .collect();

                let (unk_token, continuation_prefix, max_input_chars_per_word) = deserialize_wordpiece_config(merge_data)?;
                let (daac, _) = DoubleArrayAhoCorasick::deserialize(daac_data)
                    .ok_or(SerdeError::InvalidData("failed to deserialize DAAC"))?;

                let enc = WordPieceEncoder::from_parts(
                    daac,
                    unk_token,
                    continuation_prefix,
                    vocab_size,
                    &token_bytes,
                    max_input_chars_per_word,
                );
                Encoder::WordPiece(enc)
            }
            EncoderType::SentencePiece => {
                // OPTIMIZED: Build lookups directly from decoder (single copy)
                let (mut byte_lut, mut token_cache, token_lengths, byte_tokens) = build_token_lookups(&decoder_offsets, &decoder_data, vocab_size);
                let merges = deserialize_merges(merge_data)?;

                // Fix byte_lut/token_cache for byte-fallback collisions.
                // In models like Gemma, both a byte-fallback token (e.g., <0x3C> = id 277)
                // and a real character token (e.g., '<' = id 235322) map to the same byte.
                // Merge rules reference the real token, so we detect which single-byte tokens
                // appear in merges and prefer those.
                fix_byte_fallback_collisions(
                    &mut byte_lut,
                    &mut token_cache,
                    &merges,
                    &byte_tokens,
                );

                let enc = SentencePieceBPE::from_parts(
                    &merges,
                    byte_lut,
                    token_cache,
                    token_lengths,
                    vocab_size,
                    num_base_tokens,
                );
                Encoder::SentencePiece(enc)
            }
            EncoderType::Unigram => {
                // Unigram needs token_bytes for token_cache
                let token_bytes: Vec<Vec<u8>> = (0..vocab_size)
                    .map(|i| {
                        let start = decoder_offsets[i] as usize;
                        let end = decoder_offsets[i + 1] as usize;
                        decoder_data[start..end].to_vec()
                    })
                    .collect();

                let (scores, unk_token, byte_tokens, token_lengths) = deserialize_unigram_config(merge_data, version)?;
                let (daac, _) = DoubleArrayAhoCorasick::deserialize(daac_data)
                    .ok_or(SerdeError::InvalidData("failed to deserialize DAAC"))?;

                let enc = UnigramEncoder::from_parts(
                    daac,
                    scores,
                    unk_token,
                    byte_tokens,
                    token_lengths,
                    &token_bytes,
                );
                Encoder::Unigram(enc)
            }
        };

        // Build decoder
        let decoder_type = DecoderType::from_encoder_type(encoder_type);
        let decoder = Decoder::from_parts(decoder_data, decoder_offsets, decoder_type);

        let mut tokenizer = Tokenizer::new(encoder, decoder, pretokenizer_type, normalizer, post_processor);
        if let Some(pad_id) = pad_token_id {
            tokenizer.set_pad_token_id(pad_id);
        }
        if version >= 13 {
            let (added, specials) = deserialize_added_tokens(added_data)?;
            if !added.is_empty() {
                tokenizer.set_added_tokens(&added);
            }
            if !specials.is_empty() {
                tokenizer.set_special_tokens(specials);
            }
            tokenizer.mark_added_tokens_serialized();
        }
        Ok(tokenizer)
    }
}

/// Serialize added/special tokens (v13 ADDED_TOKENS section).
///
/// Entries: added tokens, plus any special tokens that are not in the added
/// list (flags bit1 = metadata-only, excluded from the matcher).
///
/// Flags byte: bit0 = special, bit1 = metadata-only, bit2 = lstrip,
/// bit3 = rstrip, bit4 = normalized, bit5 = single_word. Files written
/// before the flag bits existed have bits 2-5 zero, which deserializes to
/// the pre-flag matching behavior.
fn serialize_added_tokens(
    added: &[AddedTokenSpec],
    specials: &[(String, TokenId)],
) -> Vec<u8> {
    let is_special = |id: TokenId, bytes: &[u8]| {
        specials.iter().any(|(s, sid)| *sid == id && s.as_bytes() == bytes)
    };
    let mut entries: Vec<(TokenId, &[u8], u8)> = added
        .iter()
        .map(|t| {
            let mut flags = 0u8;
            if t.special || is_special(t.id, &t.bytes) {
                flags |= 0b0000_0001;
            }
            if t.lstrip {
                flags |= 0b0000_0100;
            }
            if t.rstrip {
                flags |= 0b0000_1000;
            }
            if t.normalized {
                flags |= 0b0001_0000;
            }
            if t.single_word {
                flags |= 0b0010_0000;
            }
            (t.id, t.bytes.as_slice(), flags)
        })
        .collect();
    for (s, id) in specials {
        if !added.iter().any(|t| t.id == *id && t.bytes == s.as_bytes()) {
            entries.push((*id, s.as_bytes(), 0b11));
        }
    }
    let mut out = Vec::with_capacity(4 + entries.iter().map(|(_, b, _)| 9 + b.len()).sum::<usize>());
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (id, bytes, flags) in entries {
        out.extend_from_slice(&id.to_le_bytes());
        out.push(flags);
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

#[allow(clippy::type_complexity)]
fn deserialize_added_tokens(
    data: &[u8],
) -> Result<(Vec<AddedTokenSpec>, Vec<(String, TokenId)>), SerdeError> {
    if data.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    if data.len() < 4 {
        return Err(SerdeError::InvalidData("truncated added-tokens section"));
    }
    let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut added = Vec::new();
    let mut specials = Vec::new();
    let mut pos = 4;
    for _ in 0..count {
        if pos + 9 > data.len() {
            return Err(SerdeError::InvalidData("truncated added-tokens entry"));
        }
        let id = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        let flags = data[pos + 4];
        let len = u32::from_le_bytes(data[pos + 5..pos + 9].try_into().unwrap()) as usize;
        pos += 9;
        if pos + len > data.len() {
            return Err(SerdeError::InvalidData("truncated added-tokens bytes"));
        }
        let bytes = data[pos..pos + len].to_vec();
        pos += len;
        let special = flags & 0b0000_0001 != 0;
        if flags & 0b0000_0010 == 0 {
            added.push(AddedTokenSpec {
                id,
                bytes: bytes.clone(),
                special,
                lstrip: flags & 0b0000_0100 != 0,
                rstrip: flags & 0b0000_1000 != 0,
                normalized: flags & 0b0001_0000 != 0,
                single_word: flags & 0b0010_0000 != 0,
            });
        }
        if special {
            if let Ok(s) = String::from_utf8(bytes) {
                specials.push((s, id));
            }
        }
    }
    Ok((added, specials))
}

/// Serialize the vocab decoder's flat buffer.
fn serialize_vocab_decoder(decoder: &VocabDecoder) -> Vec<u8> {
    let (data, offsets) = decoder.as_parts();

    // Format: num_offsets (u32) + offsets + data
    let mut buf = Vec::with_capacity(4 + offsets.len() * 4 + data.len());

    buf.extend_from_slice(&(offsets.len() as u32).to_le_bytes());
    for &offset in offsets {
        buf.extend_from_slice(&offset.to_le_bytes());
    }
    buf.extend_from_slice(data);

    buf
}

/// Maximum token length to cache for early exit lookup.
const MAX_CACHED_TOKEN_LEN: usize = 16;

/// ID range window for byte-fallback cluster detection.
/// SentencePiece models place ~256 byte-fallback tokens in a contiguous ID range;
/// 300 allows slack for gaps/special tokens within the range.
const FALLBACK_CLUSTER_WINDOW: u32 = 300;

/// Minimum single-byte tokens required to identify a byte-fallback cluster.
/// 200 out of 256 possible bytes is a strong signal without requiring full coverage.
const FALLBACK_CLUSTER_MIN_DENSITY: usize = 200;

/// Build token lookups directly from decoder data (single copy, no intermediate Vec<Vec<u8>>).
/// Returns: (byte_lut, token_cache, token_lengths, byte_tokens)
/// `byte_tokens` groups single-byte token IDs by byte value for collision detection.
fn build_token_lookups(
    decoder_offsets: &[u32],
    decoder_data: &[u8],
    vocab_size: usize,
) -> ([TokenId; 256], FoldHashMap<Vec<u8>, TokenId>, Vec<u16>, [Vec<TokenId>; 256]) {
    let mut byte_lut = [u32::MAX; 256];
    let mut byte_tokens: [Vec<TokenId>; 256] = std::array::from_fn(|_| Vec::new());

    // Pre-count short tokens for HashMap capacity
    let short_count: usize = (0..vocab_size)
        .filter(|&i| {
            let len = (decoder_offsets[i + 1] - decoder_offsets[i]) as usize;
            len <= MAX_CACHED_TOKEN_LEN
        })
        .count();

    let mut token_cache: FoldHashMap<Vec<u8>, TokenId> =
        FoldHashMap::with_capacity_and_hasher(short_count, Default::default());

    let mut token_lengths: Vec<u16> = Vec::with_capacity(vocab_size);

    for i in 0..vocab_size {
        let start = decoder_offsets[i] as usize;
        let end = decoder_offsets[i + 1] as usize;
        let bytes = &decoder_data[start..end];
        let len = bytes.len();

        token_lengths.push(len as u16);

        if len == 1 {
            let byte_val = bytes[0] as usize;
            byte_tokens[byte_val].push(i as TokenId);
            // First-wins for byte_lut
            if byte_lut[byte_val] == u32::MAX {
                byte_lut[byte_val] = i as TokenId;
            }
            // First-wins for token_cache
            token_cache.entry(bytes.to_vec()).or_insert(i as TokenId);
        } else if len <= MAX_CACHED_TOKEN_LEN {
            token_cache.insert(bytes.to_vec(), i as TokenId);
        }
    }

    (byte_lut, token_cache, token_lengths, byte_tokens)
}

/// Fix byte_lut and token_cache for SentencePiece models with byte-fallback collisions.
///
/// In models like Gemma, multiple tokens can map to the same single byte:
/// - A byte-fallback token like `<0x3C>` (id 277, byte 0x3C)
/// - A real character token like `<` (id 235322, also byte 0x3C)
///
/// Merge rules reference the real token, not the byte-fallback. If byte_lut returns
/// the byte-fallback ID, merges won't fire and encoding degrades to byte-level output.
///
/// Detection strategy:
/// 1. Find tokens that appear in merge rules (merge operands) — these are "real" tokens
/// 2. For any byte with a collision, prefer the merge-operand token
/// 3. For bytes where neither token is a merge operand (e.g., digits in Gemma),
///    detect the byte-fallback range and prefer the non-fallback token
fn fix_byte_fallback_collisions(
    byte_lut: &mut [TokenId; 256],
    token_cache: &mut FoldHashMap<Vec<u8>, TokenId>,
    merges: &[(TokenId, TokenId, TokenId)],
    byte_tokens: &[Vec<TokenId>; 256],
) {
    // Check if there are any collisions at all
    if !byte_tokens.iter().any(|ids| ids.len() > 1) {
        return;
    }

    // Detect byte-fallback range: find a dense cluster of ~256 single-byte tokens
    // in a contiguous ID range (SentencePiece places byte-fallback tokens together).
    let mut all_single_byte: Vec<(TokenId, u8)> = Vec::new();
    for (byte_val, ids) in byte_tokens.iter().enumerate() {
        for &id in ids {
            all_single_byte.push((id, byte_val as u8));
        }
    }
    all_single_byte.sort_by_key(|(id, _)| *id);

    let mut fallback_ids = foldhash::HashSet::default();
    if all_single_byte.len() >= 256 {
        let mut best_start = 0;
        let mut best_density = 0usize;
        for start_idx in 0..all_single_byte.len().saturating_sub(FALLBACK_CLUSTER_MIN_DENSITY) {
            let start_id = all_single_byte[start_idx].0;
            let count = all_single_byte[start_idx..]
                .iter()
                .take_while(|(id, _)| *id < start_id + FALLBACK_CLUSTER_WINDOW)
                .count();
            if count > best_density && count >= FALLBACK_CLUSTER_MIN_DENSITY {
                best_density = count;
                best_start = start_idx;
            }
        }

        if best_density >= FALLBACK_CLUSTER_MIN_DENSITY {
            let range_start_id = all_single_byte[best_start].0;
            for &(id, _) in &all_single_byte[best_start..] {
                if id < range_start_id + FALLBACK_CLUSTER_WINDOW {
                    fallback_ids.insert(id);
                } else {
                    break;
                }
            }
        }
    }

    // Collect merge operands for tie-breaking
    let mut merge_operands = foldhash::HashSet::default();
    for &(left, right, _) in merges {
        merge_operands.insert(left);
        merge_operands.insert(right);
    }

    // For each byte with multiple tokens, pick the best one
    // Priority: merge operand > non-fallback > fallback
    for (byte_val, ids) in byte_tokens.iter().enumerate() {
        if ids.len() <= 1 {
            continue;
        }

        let mut best = byte_lut[byte_val];
        for &id in ids {
            if merge_operands.contains(&id) && !merge_operands.contains(&best) {
                best = id;
            } else if !fallback_ids.contains(&id) && fallback_ids.contains(&best)
                && !merge_operands.contains(&best)
            {
                best = id;
            }
        }

        if best != byte_lut[byte_val] {
            byte_lut[byte_val] = best;
            token_cache.insert(vec![byte_val as u8], best);
        }
    }
}

/// Deserialize the decoder's flat buffer.
/// Note: We read u32s manually because the slice may not be aligned.
fn deserialize_decoder(data: &[u8], vocab_size: usize) -> Result<(Vec<u32>, Vec<u8>), SerdeError> {
    if data.len() < 4 {
        return Err(SerdeError::InvalidData("decoder data too small"));
    }

    let num_offsets = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    if num_offsets != vocab_size + 1 {
        return Err(SerdeError::InvalidData("offset count mismatch"));
    }

    let offsets_end = 4 + num_offsets * 4;
    if data.len() < offsets_end {
        return Err(SerdeError::InvalidData("decoder data truncated"));
    }

    // Read offsets manually to handle unaligned data
    let mut offsets = Vec::with_capacity(num_offsets);
    for i in 0..num_offsets {
        let start = 4 + i * 4;
        offsets.push(u32::from_le_bytes(data[start..start + 4].try_into().unwrap()));
    }

    let token_data = data[offsets_end..].to_vec();

    Ok((offsets, token_data))
}

/// Serialize the split table.
fn serialize_splits(splits: &[Split]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(splits.len() * 8);
    for split in splits {
        buf.extend_from_slice(&split.left.to_le_bytes());
        buf.extend_from_slice(&split.right.to_le_bytes());
    }
    buf
}

/// Deserialize the split table.
/// Note: We read manually to handle unaligned data from file reads.
fn deserialize_splits(data: &[u8]) -> Result<Vec<Split>, SerdeError> {
    if data.len() % size_of::<Split>() != 0 {
        return Err(SerdeError::InvalidData("split data size not aligned"));
    }

    let num_splits = data.len() / size_of::<Split>();
    let mut splits = Vec::with_capacity(num_splits);

    for i in 0..num_splits {
        let start = i * 8;
        let left = u32::from_le_bytes(data[start..start + 4].try_into().unwrap());
        let right = u32::from_le_bytes(data[start + 4..start + 8].try_into().unwrap());
        splits.push(Split { left, right });
    }

    Ok(splits)
}

/// Serialize the next_prefix_match table.
fn serialize_prefix_match(prefixes: &[TokenId]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(prefixes.len() * 4);
    for &prefix in prefixes {
        buf.extend_from_slice(&prefix.to_le_bytes());
    }
    buf
}

/// Deserialize the next_prefix_match table.
fn deserialize_prefix_match(data: &[u8]) -> Result<Vec<TokenId>, SerdeError> {
    if data.len() % 4 != 0 {
        return Err(SerdeError::InvalidData("prefix data size not aligned"));
    }

    let num_prefixes = data.len() / 4;
    let mut prefixes = Vec::with_capacity(num_prefixes);

    for i in 0..num_prefixes {
        let start = i * 4;
        prefixes.push(u32::from_le_bytes(data[start..start + 4].try_into().unwrap()));
    }

    Ok(prefixes)
}

/// Pack two token IDs into a single u64 key.
#[inline(always)]
fn pack_pair(left: TokenId, right: TokenId) -> u64 {
    ((left as u64) << 32) | (right as u64)
}

/// Unpack u64 key back to two token IDs.
#[inline(always)]
fn unpack_pair(packed: u64) -> (TokenId, TokenId) {
    let left = (packed >> 32) as TokenId;
    let right = (packed & 0xFFFF_FFFF) as TokenId;
    (left, right)
}

/// Serialize Simple encoder's pair_lookup as merge list with merged IDs.
///
/// Format: (left: u32, right: u32, merged_id: u32) per merge, sorted by rank.
/// This allows fast deserialization without rebuilding the token_cache map.
fn serialize_pair_lookup(enc: &BytePairEncoder) -> Vec<u8> {
    let pair_lookup = enc.pair_lookup();

    // Collect all merges with their ranks and merged IDs
    let mut merges: Vec<(u32, TokenId, TokenId, TokenId)> = pair_lookup
        .iter()
        .map(|(&packed, &(merged, rank))| {
            let (left, right) = unpack_pair(packed);
            (rank, left, right, merged)
        })
        .collect();

    // Sort by rank to preserve merge order
    merges.sort_by_key(|(rank, _, _, _)| *rank);

    // Serialize as (left, right, merged_id) tuples
    let mut buf = Vec::with_capacity(merges.len() * 12);
    for (_, left, right, merged) in merges {
        buf.extend_from_slice(&left.to_le_bytes());
        buf.extend_from_slice(&right.to_le_bytes());
        buf.extend_from_slice(&merged.to_le_bytes());
    }
    buf
}

/// Serialize SentencePiece encoder's pair_lookup as merge list with merged IDs.
///
/// Format: (left: u32, right: u32, merged_id: u32) per merge, sorted by rank.
/// This allows fast deserialization without rebuilding the token_cache map.
fn serialize_sentencepiece_config(enc: &SentencePieceBPE) -> Vec<u8> {
    let pair_lookup = enc.pair_lookup();

    // Collect all merges with their ranks and merged IDs
    let mut merges: Vec<(u32, TokenId, TokenId, TokenId)> = pair_lookup
        .iter()
        .map(|(&packed, &(merged, rank))| {
            let (left, right) = unpack_pair(packed);
            (rank, left, right, merged)
        })
        .collect();

    // Sort by rank to preserve merge order
    merges.sort_by_key(|(rank, _, _, _)| *rank);

    // Serialize as (left, right, merged_id) tuples
    let mut buf = Vec::with_capacity(merges.len() * 12);
    for (_, left, right, merged) in merges {
        buf.extend_from_slice(&left.to_le_bytes());
        buf.extend_from_slice(&right.to_le_bytes());
        buf.extend_from_slice(&merged.to_le_bytes());
    }
    buf
}

/// Deserialize merge list for Simple/SentencePiece encoder.
///
/// Format: (left: u32, right: u32, merged_id: u32) per merge.
/// Returns tuples of (left, right, merged_id) for direct pair_lookup construction.
fn deserialize_merges(data: &[u8]) -> Result<Vec<(TokenId, TokenId, TokenId)>, SerdeError> {
    if data.len() % 12 != 0 {
        return Err(SerdeError::InvalidData("merge data size not aligned (expected 12 bytes per merge)"));
    }

    let num_merges = data.len() / 12;
    let mut merges = Vec::with_capacity(num_merges);

    for i in 0..num_merges {
        let start = i * 12;
        let left = u32::from_le_bytes(data[start..start + 4].try_into().unwrap());
        let right = u32::from_le_bytes(data[start + 4..start + 8].try_into().unwrap());
        let merged = u32::from_le_bytes(data[start + 8..start + 12].try_into().unwrap());
        merges.push((left, right, merged));
    }

    Ok(merges)
}

/// Rebuild pair_lookup from split_table using packed u64 keys.
fn rebuild_pair_lookup(
    splits: &[Split],
    num_base_tokens: usize,
) -> FoldHashMap<u64, TokenId> {
    let mut lookup = FoldHashMap::default();

    for (id, split) in splits.iter().enumerate().skip(num_base_tokens) {
        // Split::base entries (added/special tokens, e.g. gpt2's <|endoftext|>)
        // point at themselves and are not merges; from_json never puts them in
        // pair_lookup, and a degenerate (id,id)->id entry would make the rank
        // table's monotonicity check reject the whole vocab.
        if split.left == id as TokenId && split.right == id as TokenId {
            continue;
        }
        lookup.insert(pack_pair(split.left, split.right), id as TokenId);
    }

    lookup
}

/// Serialize WordPiece encoder config (unk_token + continuation_prefix + max_input_chars_per_word).
///
/// Format: unk_token (u32) + prefix_len (u32) + prefix bytes + max_input_chars_per_word (u32)
fn serialize_wordpiece_config(enc: &WordPieceEncoder) -> Vec<u8> {
    let prefix = enc.continuation_prefix();
    let mut buf = Vec::with_capacity(12 + prefix.len());
    buf.extend_from_slice(&enc.unk_token().to_le_bytes());
    buf.extend_from_slice(&(prefix.len() as u32).to_le_bytes());
    buf.extend_from_slice(prefix);
    buf.extend_from_slice(&(enc.max_input_chars_per_word() as u32).to_le_bytes());
    buf
}

/// Deserialize WordPiece encoder config.
fn deserialize_wordpiece_config(data: &[u8]) -> Result<(TokenId, Vec<u8>, usize), SerdeError> {
    if data.len() < 8 {
        return Err(SerdeError::InvalidData("wordpiece config too small"));
    }

    let unk_token = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let prefix_len = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

    if data.len() < 8 + prefix_len {
        return Err(SerdeError::InvalidData("wordpiece prefix truncated"));
    }

    let continuation_prefix = data[8..8 + prefix_len].to_vec();

    // Backward compatible: older .tkz files may not have this field
    let max_input_chars_per_word = if data.len() >= 8 + prefix_len + 4 {
        u32::from_le_bytes(data[8 + prefix_len..12 + prefix_len].try_into().unwrap()) as usize
    } else {
        100 // HF default
    };

    Ok((unk_token, continuation_prefix, max_input_chars_per_word))
}

/// Serialize Unigram encoder config.
///
/// Format:
/// - vocab_size (u32)
/// - unk_token (u32)
/// - byte_tokens (256 × u32 = 1024 bytes)
/// - scores (vocab_size × f64; f32 before v12)
/// - token_lengths (vocab_size × u16)
fn serialize_unigram_config(enc: &UnigramEncoder) -> Vec<u8> {
    let scores = enc.scores();
    let byte_tokens = enc.byte_tokens();
    let token_lengths = enc.token_lengths();
    let vocab_size = enc.vocab_size();

    // Calculate buffer size
    let buf_size = 4 + 4 + (256 * 4) + (vocab_size * 8) + (vocab_size * 2);
    let mut buf = Vec::with_capacity(buf_size);

    // vocab_size
    buf.extend_from_slice(&(vocab_size as u32).to_le_bytes());
    // unk_token
    buf.extend_from_slice(&enc.unk_token().to_le_bytes());
    // byte_tokens (256 u32s)
    for &bt in byte_tokens.iter() {
        buf.extend_from_slice(&bt.to_le_bytes());
    }
    // scores (f64 array, v12+; v11 and earlier stored f32)
    for &score in scores {
        buf.extend_from_slice(&score.to_le_bytes());
    }
    // token_lengths (u16 array)
    for &len in token_lengths {
        buf.extend_from_slice(&len.to_le_bytes());
    }

    buf
}

/// Deserialize Unigram encoder config.
///
/// v12+ stores scores as f64; v10/v11 stored f32 (widened on load — old files
/// keep working, but only freshly generated v12 files match HF bit-for-bit on
/// near-tie Viterbi paths).
fn deserialize_unigram_config(
    data: &[u8],
    version: u32,
) -> Result<(Vec<f64>, TokenId, [TokenId; 256], Vec<u16>), SerdeError> {
    if data.len() < 8 + 1024 {
        return Err(SerdeError::InvalidData("unigram config too small"));
    }

    let vocab_size = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let unk_token = u32::from_le_bytes(data[4..8].try_into().unwrap());

    // Read byte_tokens (256 u32s starting at offset 8)
    let mut byte_tokens = [0u32; 256];
    for i in 0..256 {
        let start = 8 + i * 4;
        byte_tokens[i] = u32::from_le_bytes(data[start..start + 4].try_into().unwrap());
    }

    // Read scores starting at offset 8 + 1024
    let scores_offset = 8 + 1024;
    let score_width = if version >= 12 { 8 } else { 4 };
    let expected_len = scores_offset + vocab_size * score_width + vocab_size * 2;
    if data.len() < expected_len {
        return Err(SerdeError::InvalidData("unigram config truncated"));
    }

    let mut scores = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        let start = scores_offset + i * score_width;
        if score_width == 8 {
            scores.push(f64::from_le_bytes(data[start..start + 8].try_into().unwrap()));
        } else {
            scores.push(f32::from_le_bytes(data[start..start + 4].try_into().unwrap()) as f64);
        }
    }

    // Read token_lengths (vocab_size u16s starting after scores)
    let lengths_offset = scores_offset + vocab_size * score_width;
    let mut token_lengths = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        let start = lengths_offset + i * 2;
        token_lengths.push(u16::from_le_bytes(data[start..start + 2].try_into().unwrap()));
    }

    Ok((scores, unk_token, byte_tokens, token_lengths))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TokenId;

    #[test]
    fn test_added_tokens_roundtrip() {
        let mut tok = make_test_tokenizer();
        tok.set_added_tokens(&[
            AddedTokenSpec { special: true, ..AddedTokenSpec::plain(300, b"<|special|>".to_vec()) },
            AddedTokenSpec {
                lstrip: true,
                rstrip: true,
                normalized: false,
                single_word: true,
                ..AddedTokenSpec::plain(301, b"<mask>".to_vec())
            },
        ]);
        tok.set_special_tokens(vec![("<|special|>".to_string(), 300)]);
        let path = std::env::temp_dir().join("tokie_added_tokens_roundtrip.tkz");
        tok.to_file(&path).unwrap();
        let loaded = Tokenizer::from_file(&path).unwrap();
        assert_eq!(loaded.added_tokens_raw(), tok.added_tokens_raw());
        assert_eq!(loaded.special_tokens(), tok.special_tokens());
        assert!(loaded.added_tokens_serialized(), "v13 files carry added tokens");
        assert_eq!(
            loaded.encode("ab<|special|>cd", false).ids,
            tok.encode("ab<|special|>cd", false).ids,
            "added-token splitting must survive the roundtrip"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_no_added_tokens_roundtrip_flag() {
        let tok = make_test_tokenizer();
        let path = std::env::temp_dir().join("tokie_no_added_tokens_roundtrip.tkz");
        tok.to_file(&path).unwrap();
        let loaded = Tokenizer::from_file(&path).unwrap();
        assert!(loaded.added_tokens_raw().is_empty());
        assert!(loaded.added_tokens_serialized(), "empty section still marks the file authoritative");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_crc32() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"hello"), crc32(b"hello"));
        assert_ne!(crc32(b"hello"), crc32(b"world"));
    }

    #[test]
    fn test_pretokenizer_type_roundtrip() {
        for typ in [
            PretokType::None,
            PretokType::Gpt2,
            PretokType::Cl100k,
            PretokType::O200k,
        ] {
            assert_eq!(PretokType::from_u32(typ as u32), Some(typ));
        }
    }

    fn make_test_tokenizer() -> Tokenizer {
        let base_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let merges: Vec<(TokenId, TokenId)> = vec![
            (b'a' as u32, b'b' as u32), // ab
            (b'c' as u32, b'd' as u32), // cd
            (256, 257),                  // abcd
        ];
        let (encoder, token_bytes) = crate::encoder::BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);
        Tokenizer::new(Encoder::Backtracking(encoder), decoder, PretokType::Gpt2, Normalizer::None, PostProcessor::None)
    }

    fn make_simple_test_tokenizer() -> Tokenizer {
        let base_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let merges: Vec<(TokenId, TokenId)> = vec![
            (b'a' as u32, b'b' as u32), // ab
            (b'c' as u32, b'd' as u32), // cd
            (256, 257),                  // abcd
        ];
        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);
        Tokenizer::new(Encoder::Simple(encoder), decoder, PretokType::Gpt2, Normalizer::None, PostProcessor::None)
    }

    #[test]
    fn test_rank_table_survives_roundtrip_with_added_token() {
        // Non-merge tokens past the base range (e.g. gpt2's <|endoftext|>) get
        // self-referential Split::base entries; the pair_lookup rebuild must not
        // turn those into degenerate (id,id)->id merges, or the rank table's
        // monotonicity check rejects the whole vocab on every .tkz load.
        let base_tokens: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        let merges: Vec<(TokenId, TokenId)> = vec![
            (b'a' as u32, b'b' as u32), // 256 "ab"
            (256, b'c' as u32),         // 257 "abc"
        ];
        let added = vec![(258u32, b"<|endoftext|>".to_vec())];
        let (encoder, token_bytes) = crate::encoder::BacktrackingBytePairEncoder::from_merges_with_added(
            &merges,
            &base_tokens,
            &added,
        );
        assert!(encoder.has_rank_merge(), "precondition: fresh build has rank table");

        let decoder = Decoder::new(token_bytes);
        let tokenizer = Tokenizer::new(
            Encoder::Backtracking(encoder),
            decoder,
            PretokType::Gpt2,
            Normalizer::None,
            PostProcessor::None,
        );

        let mut buf = Vec::new();
        tokenizer.save(&mut buf).expect("save failed");
        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = Tokenizer::load(&mut cursor).expect("load failed");

        match loaded.encoder() {
            Encoder::Backtracking(enc) => {
                assert!(
                    enc.has_rank_merge(),
                    "rank table must survive .tkz roundtrip when the vocab has added tokens"
                );
            }
            _ => panic!("expected backtracking encoder"),
        }
        assert_eq!(
            tokenizer.encode("abcab", false).ids,
            loaded.encode("abcab", false).ids
        );
    }

    #[test]
    fn test_save_load_roundtrip() {
        let tokenizer = make_test_tokenizer();

        // Save to memory buffer
        let mut buf = Vec::new();
        tokenizer
            .save(&mut buf)
            .expect("save failed");

        // Load from buffer
        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = Tokenizer::load(&mut cursor).expect("load failed");

        // Verify same vocab size
        assert_eq!(tokenizer.vocab_size(), loaded.vocab_size());

        // Verify encoding matches
        let test_texts = ["Hello world", "abcd", "test 123", "abcdabcd"];
        for text in test_texts {
            let original_tokens = tokenizer.encode(text, false).ids;
            let loaded_tokens = loaded.encode(text, false).ids;
            assert_eq!(
                original_tokens, loaded_tokens,
                "encoding mismatch for '{}'",
                text
            );
        }

        // Verify decoding matches
        let tokens = tokenizer.encode("Hello world", false).ids;
        let original_decoded = tokenizer.decode(&tokens);
        let loaded_decoded = loaded.decode(&tokens);
        assert_eq!(original_decoded, loaded_decoded);
    }

    #[test]
    fn test_save_load_file() {
        let tokenizer = make_test_tokenizer();

        let temp_path = std::env::temp_dir().join("tokie_test.bin");

        // Save to file
        tokenizer
            .to_file(&temp_path)
            .expect("to_file failed");

        // Load from file
        let loaded = Tokenizer::from_file(&temp_path).expect("from_file failed");

        // Verify encoding matches
        let text = "Hello world test";
        assert_eq!(tokenizer.encode(text, false).ids, loaded.encode(text, false).ids);

        // Cleanup
        std::fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_load_invalid_magic() {
        let mut bad_data = vec![0u8; HEADER_SIZE + 100];
        bad_data[0..4].copy_from_slice(b"BADM");
        let mut cursor = std::io::Cursor::new(&bad_data);
        let result = Tokenizer::load(&mut cursor);
        assert!(matches!(result, Err(SerdeError::InvalidMagic)));
    }

    #[test]
    fn test_load_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&99u32.to_le_bytes()); // Bad version
        data.resize(HEADER_SIZE + 100, 0);

        let mut cursor = std::io::Cursor::new(&data);
        let result = Tokenizer::load(&mut cursor);
        assert!(matches!(result, Err(SerdeError::UnsupportedVersion(99))));
    }

    #[test]
    fn test_pad_token_id_roundtrip() {
        let mut tokenizer = make_test_tokenizer();
        tokenizer.set_pad_token_id(42);

        // Save to memory buffer
        let mut buf = Vec::new();
        tokenizer.save(&mut buf).expect("save failed");

        // Load from buffer
        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = Tokenizer::load(&mut cursor).expect("load failed");

        assert_eq!(loaded.pad_token_id(), Some(42));
    }

    #[test]
    fn test_pad_token_id_none_roundtrip() {
        let tokenizer = make_test_tokenizer();
        assert_eq!(tokenizer.pad_token_id(), None);

        let mut buf = Vec::new();
        tokenizer.save(&mut buf).expect("save failed");

        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = Tokenizer::load(&mut cursor).expect("load failed");

        assert_eq!(loaded.pad_token_id(), None);
    }

    #[test]
    fn test_simple_encoder_save_load_roundtrip() {
        let tokenizer = make_simple_test_tokenizer();

        // Verify it's a Simple encoder
        assert_eq!(tokenizer.encoder_type(), EncoderType::Simple);

        // Save to memory buffer
        let mut buf = Vec::new();
        tokenizer
            .save(&mut buf)
            .expect("save failed");

        // Load from buffer
        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = Tokenizer::load(&mut cursor).expect("load failed");

        // Verify it loaded as Simple encoder
        assert_eq!(loaded.encoder_type(), EncoderType::Simple);

        // Verify same vocab size
        assert_eq!(tokenizer.vocab_size(), loaded.vocab_size());

        // Verify encoding matches
        let test_texts = ["Hello world", "abcd", "test 123", "abcdabcd"];
        for text in test_texts {
            let original_tokens = tokenizer.encode(text, false).ids;
            let loaded_tokens = loaded.encode(text, false).ids;
            assert_eq!(
                original_tokens, loaded_tokens,
                "encoding mismatch for '{}'",
                text
            );
        }

        // Verify decoding matches
        let tokens = tokenizer.encode("Hello world", false).ids;
        let original_decoded = tokenizer.decode(&tokens);
        let loaded_decoded = loaded.decode(&tokens);
        assert_eq!(original_decoded, loaded_decoded);
    }
}
