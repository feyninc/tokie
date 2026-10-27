//! The `.tokie` container. Version 1, little-endian, no struct cast.
//!
//! `docs/FORMAT.md` is the contract. This module reads and writes the bytes.
//! The tokenizer turns the image into the live tables.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::cut::Cut;
use crate::norm::Norm;

pub(crate) const FILE_CAP: usize = 512 * 1024 * 1024;
pub(crate) const PIECE_BUDGET: usize = 128 * 1024 * 1024;
const BLOB_CAP: u16 = 22;
const HEADER_LEN: usize = 128;
const ENTRY_LEN: usize = 40;
const ALIGN: u64 = 64;
const MAGIC: &[u8; 8] = b"TOKIE\x89\r\n";

const TAG_META: u16 = 1;
const TAG_STEPS: u16 = 2;
const TAG_LITERALS: u16 = 3;
const TAG_STRTAB: u16 = 4;
const TAG_ARENA: u16 = 5;
const TAG_SLOTS: u16 = 6;
const TAG_ID_KEYS: u16 = 7;
const TAG_PAIRS: u16 = 8;
const TAG_BYTE_IDS: u16 = 9;
const TAG_SPM_BYTES: u16 = 10;
const TAG_SPM_CHARS: u16 = 11;
const TAG_NODES: u16 = 12;
const TAG_EDGES: u16 = 13;
const TAG_ADDED_RAW: u16 = 14;
const TAG_ADDED_NORM: u16 = 15;
const TAG_POST_PREFIX: u16 = 16;
const TAG_POST_SUFFIX: u16 = 17;
const TAG_CHARSMAP: u16 = 18;
const TAG_BMP: u16 = 19;
const TAG_WP: u16 = 20;
const TAG_PIECES: u16 = 21;

const MODEL_BPE: u8 = 1;
const MODEL_WP: u8 = 2;
const MODEL_UNI: u8 = 3;
const MODEL_SPM: u8 = 4;

const STEP_NORM: u16 = 1;
const STEP_COMPILED: u16 = 2;
const STEP_LITERAL: u16 = 3;

const FLAG_PREFIX: u8 = 1;
const FLAG_MARK: u8 = 2;
const FLAG_STRIP: u8 = 4;
const SLOT_PRESENT: u32 = 1;
const SLOT_SPECIAL: u32 = 2;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Model {
    Bpe,
    WordPiece,
    Unigram,
    Spm,
}

pub(crate) enum StepRec {
    Norm(Norm),
    Compiled(u16),
    Literal(u16),
}

pub(crate) struct CharsmapRec {
    pub blob: Vec<u8>,
    pub bmp: Vec<(u32, u32)>,
}

pub(crate) struct SlotRec {
    pub lead_at: u32,
    pub lead_len: u32,
    pub next_at: u32,
    pub next_len: u32,
    pub token: Option<String>,
    pub special: bool,
}

pub(crate) struct Image {
    pub sha: [u8; 32],
    pub model: Model,
    pub cut: Cut,
    pub prefix_space: bool,
    pub mark: bool,
    pub strip_space: bool,
    pub unk: u32,
    pub max_chars: u32,
    pub vocab_len: u32,
    pub prefix: String,
    pub unk_score: f64,
    pub steps: Vec<StepRec>,
    pub literals: Vec<(String, String)>,
    pub charsmaps: Vec<CharsmapRec>,
    pub arena: Vec<u8>,
    pub slots: Vec<SlotRec>,
    pub pairs: Vec<(u32, u32, u32, u32)>,
    pub byte_ids: Option<[u32; 256]>,
    pub spm_bytes: Option<[u32; 256]>,
    pub spm_chars: Vec<(u32, u32)>,
    pub nodes: Vec<(u32, u16, u32, f64)>,
    pub edges: Vec<(u8, u32)>,
    pub added_raw: Vec<(String, u32)>,
    pub added_norm: Vec<(String, u32)>,
    pub post_prefix: Vec<u32>,
    pub post_suffix: Vec<u32>,
    pub wp: Vec<(String, u32)>,
    pub pieces: Vec<(Vec<u8>, Vec<u32>)>,
}

struct Blob {
    tag: u16,
    count: u32,
    stride: u32,
    bytes: Vec<u8>,
}

pub(crate) fn piece_cost(piece: &[u8]) -> usize {
    if piece.len() <= 15 { 32 } else { 32 + piece.len() }
}

pub(crate) fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Encode an image. The caller writes the bytes with a temp file and a rename.
pub(crate) fn encode(image: &Image) -> Result<Vec<u8>, String> {
    let mut tab = Tab::default();
    let mut blobs = Vec::new();
    blobs.push(meta_blob(image, &mut tab)?);
    if !image.steps.is_empty() {
        blobs.push(steps_blob(image)?);
    }
    if !image.literals.is_empty() {
        blobs.push(literals_blob(image, &mut tab)?);
    }
    blobs.push(arena_blob(image));
    blobs.push(slots_blob(image, &mut tab)?);
    blobs.push(id_keys_blob(image, &mut tab)?);
    push_opt(&mut blobs, pairs_blob(image));
    push_opt(&mut blobs, byte_ids_blob(image));
    push_opt(&mut blobs, spm_bytes_blob(image));
    push_opt(&mut blobs, spm_chars_blob(image));
    push_opt(&mut blobs, nodes_blob(image));
    push_opt(&mut blobs, edges_blob(image));
    push_opt(&mut blobs, added_blob(TAG_ADDED_RAW, &image.added_raw, &mut tab)?);
    push_opt(&mut blobs, added_blob(TAG_ADDED_NORM, &image.added_norm, &mut tab)?);
    push_opt(&mut blobs, ids_blob(TAG_POST_PREFIX, &image.post_prefix));
    push_opt(&mut blobs, ids_blob(TAG_POST_SUFFIX, &image.post_suffix));
    push_opt(&mut blobs, charsmap_blob(image));
    push_opt(&mut blobs, bmp_blob(image));
    push_opt(&mut blobs, wp_blob(image, &mut tab)?);
    push_opt(&mut blobs, pieces_blob(image)?);
    blobs.push(Blob {
        tag: TAG_STRTAB,
        count: 1,
        stride: 0,
        bytes: tab.bytes,
    });
    if blobs.len() > BLOB_CAP as usize {
        return Err(format!("a .tokie file holds at most {BLOB_CAP} blobs"));
    }
    layout(&blobs, image.sha)
}

fn push_opt(blobs: &mut Vec<Blob>, blob: Option<Blob>) {
    if let Some(blob) = blob {
        blobs.push(blob);
    }
}

fn layout(blobs: &[Blob], sha: [u8; 32]) -> Result<Vec<u8>, String> {
    let dir_len = blobs.len() * ENTRY_LEN;
    let mut cursor = align_up((HEADER_LEN + dir_len) as u64, ALIGN);
    let mut entries = Vec::with_capacity(blobs.len());
    for blob in blobs {
        let len = blob.bytes.len() as u64;
        entries.push((cursor, len));
        cursor = align_up(cursor + len, ALIGN);
    }
    let file_len = cursor;
    if file_len > FILE_CAP as u64 {
        return Err(format!("a .tokie file is at most {FILE_CAP} bytes"));
    }
    let mut out = vec![0u8; file_len as usize];
    let mut directory = vec![0u8; dir_len];
    for (i, blob) in blobs.iter().enumerate() {
        let (offset, len) = entries[i];
        let at = i * ENTRY_LEN;
        directory[at..at + 2].copy_from_slice(&blob.tag.to_le_bytes());
        directory[at + 8..at + 12].copy_from_slice(&blob.count.to_le_bytes());
        directory[at + 12..at + 16].copy_from_slice(&blob.stride.to_le_bytes());
        directory[at + 16..at + 24].copy_from_slice(&offset.to_le_bytes());
        directory[at + 24..at + 32].copy_from_slice(&len.to_le_bytes());
        let crc = crc32c(&blob.bytes);
        directory[at + 32..at + 36].copy_from_slice(&crc.to_le_bytes());
        let start = offset as usize;
        out[start..start + blob.bytes.len()].copy_from_slice(&blob.bytes);
    }
    out[HEADER_LEN..HEADER_LEN + dir_len].copy_from_slice(&directory);
    let dir_crc = crc32c(&directory);
    let mut header = [0u8; HEADER_LEN];
    header[..8].copy_from_slice(MAGIC);
    header[8] = 1;
    header[10..12].copy_from_slice(&(HEADER_LEN as u16).to_le_bytes());
    header[12..14].copy_from_slice(&(ENTRY_LEN as u16).to_le_bytes());
    header[14..16].copy_from_slice(&(blobs.len() as u16).to_le_bytes());
    header[16..24].copy_from_slice(&file_len.to_le_bytes());
    header[24..32].copy_from_slice(&(ALIGN as u64).to_le_bytes());
    header[32..36].copy_from_slice(&dir_crc.to_le_bytes());
    header[64..96].copy_from_slice(&sha);
    let header_crc = crc32c(&header);
    header[36..40].copy_from_slice(&header_crc.to_le_bytes());
    out[..HEADER_LEN].copy_from_slice(&header);
    Ok(out)
}

pub(crate) fn decode(bytes: &[u8]) -> Result<Image, String> {
    if bytes.len() < 8 || &bytes[..8] != MAGIC {
        return Err(".tokie magic is missing".to_string());
    }
    if bytes.len() < HEADER_LEN {
        return Err(".tokie header is truncated".to_string());
    }
    if bytes.len() > FILE_CAP {
        return Err(format!("a .tokie file is at most {FILE_CAP} bytes"));
    }
    let header = &bytes[..HEADER_LEN];
    if header[8] != 1 || header[9] != 0 {
        return Err(format!(".tokie version {} is not supported", header[8]));
    }
    let header_bytes = u16::from_le_bytes(header[10..12].try_into().unwrap());
    let entry_bytes = u16::from_le_bytes(header[12..14].try_into().unwrap());
    let blob_count = u16::from_le_bytes(header[14..16].try_into().unwrap());
    let file_len = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let align = u64::from_le_bytes(header[24..32].try_into().unwrap());
    if header_bytes != HEADER_LEN as u16 || entry_bytes != ENTRY_LEN as u16 {
        return Err(".tokie header sizes do not match version 1".to_string());
    }
    if align != ALIGN {
        return Err(".tokie alignment is not 64".to_string());
    }
    if blob_count > BLOB_CAP {
        return Err(format!("a .tokie file holds at most {BLOB_CAP} blobs"));
    }
    if file_len != bytes.len() as u64 {
        return Err(".tokie file length does not match the header".to_string());
    }
    let mut check = [0u8; HEADER_LEN];
    check.copy_from_slice(header);
    check[36..40].fill(0);
    let stored = u32::from_le_bytes(header[36..40].try_into().unwrap());
    if crc32c(&check) != stored {
        return Err(".tokie header checksum failed".to_string());
    }
    let dir_len = blob_count as usize * ENTRY_LEN;
    if HEADER_LEN + dir_len > bytes.len() {
        return Err(".tokie directory is truncated".to_string());
    }
    let directory = &bytes[HEADER_LEN..HEADER_LEN + dir_len];
    let dir_crc = u32::from_le_bytes(header[32..36].try_into().unwrap());
    if crc32c(directory) != dir_crc {
        return Err(".tokie directory checksum failed".to_string());
    }
    let mut seen = [false; 32];
    let mut spans = Vec::with_capacity(blob_count as usize);
    for i in 0..blob_count as usize {
        let at = i * ENTRY_LEN;
        let tag = u16::from_le_bytes(directory[at..at + 2].try_into().unwrap());
        let count = u32::from_le_bytes(directory[at + 8..at + 12].try_into().unwrap());
        let stride = u32::from_le_bytes(directory[at + 12..at + 16].try_into().unwrap());
        let offset = u64::from_le_bytes(directory[at + 16..at + 24].try_into().unwrap());
        let len = u64::from_le_bytes(directory[at + 24..at + 32].try_into().unwrap());
        let crc = u32::from_le_bytes(directory[at + 32..at + 36].try_into().unwrap());
        if tag == 0 || tag > TAG_PIECES {
            return Err(format!(".tokie tag {tag} is unknown"));
        }
        if seen[tag as usize] {
            return Err(format!(".tokie tag {tag} is repeated"));
        }
        seen[tag as usize] = true;
        if offset % ALIGN != 0 {
            return Err(".tokie blob is not 64-byte aligned".to_string());
        }
        let end = offset.checked_add(len).ok_or(".tokie blob range overflows")?;
        if end > file_len {
            return Err(".tokie blob runs past the file".to_string());
        }
        if stride != 0 && stride as u64 * count as u64 != len {
            return Err(".tokie blob length does not match its stride".to_string());
        }
        let body = &bytes[offset as usize..end as usize];
        if crc32c(body) != crc {
            return Err(format!(".tokie blob {tag} checksum failed"));
        }
        spans.push(Span {
            tag,
            count,
            stride,
            offset,
            len,
        });
    }
    check_coverage(bytes, &spans, file_len, (HEADER_LEN + dir_len) as u64)?;
    let mut got = [None; 32];
    for span in &spans {
        got[span.tag as usize] = Some(*span);
    }
    let strtab = body(bytes, got[TAG_STRTAB as usize].ok_or(".tokie file has no string table")?)?;
    let meta = body(bytes, got[TAG_META as usize].ok_or(".tokie file has no meta blob")?)?;
    if meta.len() != 64 {
        return Err(".tokie meta blob is not 64 bytes".to_string());
    }
    let model = match meta[0] {
        MODEL_BPE => Model::Bpe,
        MODEL_WP => Model::WordPiece,
        MODEL_UNI => Model::Unigram,
        MODEL_SPM => Model::Spm,
        other => return Err(format!(".tokie model {other} is unknown")),
    };
    let flags = meta[1];
    let cut = cut_of(u16::from_le_bytes(meta[2..4].try_into().unwrap()))?;
    let unk = u32::from_le_bytes(meta[4..8].try_into().unwrap());
    let max_chars = u32::from_le_bytes(meta[8..12].try_into().unwrap());
    let vocab_len = u32::from_le_bytes(meta[12..16].try_into().unwrap());
    let prefix = text(strtab, u32::from_le_bytes(meta[16..20].try_into().unwrap()), u32::from_le_bytes(meta[20..24].try_into().unwrap()))?.to_string();
    let unk_score = f64::from_bits(u64::from_le_bytes(meta[24..32].try_into().unwrap()));
    let sha = header[64..96].try_into().unwrap();
    let steps = read_steps(bytes, got[TAG_STEPS as usize])?;
    let literals = read_literals(bytes, strtab, got[TAG_LITERALS as usize])?;
    let arena = match got[TAG_ARENA as usize] {
        Some(span) => body(bytes, span)?.to_vec(),
        None => Vec::new(),
    };
    let slots = read_slots(bytes, strtab, got[TAG_SLOTS as usize].ok_or(".tokie file has no slots")?, vocab_len)?;
    let id_span = got[TAG_ID_KEYS as usize].ok_or(".tokie file has no id keys")?;
    let id_keys = read_pairs_str(bytes, strtab, Some(id_span))?;
    check_id_keys(&slots, &id_keys)?;
    let pairs = read_quads(bytes, got[TAG_PAIRS as usize])?;
    let byte_ids = read_u32x256(bytes, got[TAG_BYTE_IDS as usize])?;
    let spm_bytes = read_u32x256(bytes, got[TAG_SPM_BYTES as usize])?;
    let spm_chars = read_pairs_u32(bytes, got[TAG_SPM_CHARS as usize])?;
    let nodes = read_nodes(bytes, got[TAG_NODES as usize])?;
    let edges = read_edges(bytes, got[TAG_EDGES as usize])?;
    let added_raw = read_pairs_str(bytes, strtab, got[TAG_ADDED_RAW as usize])?;
    let added_norm = read_pairs_str(bytes, strtab, got[TAG_ADDED_NORM as usize])?;
    let post_prefix = read_ids(bytes, got[TAG_POST_PREFIX as usize])?;
    let post_suffix = read_ids(bytes, got[TAG_POST_SUFFIX as usize])?;
    let charsmaps = read_charsmaps(bytes, got[TAG_CHARSMAP as usize], got[TAG_BMP as usize])?;
    let wp = read_pairs_str(bytes, strtab, got[TAG_WP as usize])?;
    let pieces = read_pieces(bytes, got[TAG_PIECES as usize], vocab_len)?;
    match model {
        Model::Bpe if byte_ids.is_none() || got[TAG_PAIRS as usize].is_none() => {
            return Err(".tokie bpe file is missing pairs or byte ids".to_string());
        }
        Model::WordPiece if got[TAG_WP as usize].is_none() => {
            return Err(".tokie wordpiece file is missing its vocab".to_string());
        }
        Model::Unigram if got[TAG_NODES as usize].is_none() || got[TAG_EDGES as usize].is_none() => {
            return Err(".tokie unigram file is missing its lattice".to_string());
        }
        Model::Spm if spm_bytes.is_none() || got[TAG_PAIRS as usize].is_none() => {
            return Err(".tokie sentencepiece bpe file is missing pairs or byte tokens".to_string());
        }
        _ => {}
    }
    let compiled = steps.iter().filter(|step| matches!(step, StepRec::Compiled(_))).count();
    if compiled != charsmaps.len() {
        return Err(".tokie compiled steps do not match the charsmap blobs".to_string());
    }
    Ok(Image {
        sha,
        model,
        cut,
        prefix_space: flags & FLAG_PREFIX != 0,
        mark: flags & FLAG_MARK != 0,
        strip_space: flags & FLAG_STRIP != 0,
        unk,
        max_chars,
        vocab_len,
        prefix,
        unk_score,
        steps,
        literals,
        charsmaps,
        arena,
        slots,
        pairs,
        byte_ids,
        spm_bytes,
        spm_chars,
        nodes,
        edges,
        added_raw,
        added_norm,
        post_prefix,
        post_suffix,
        wp,
        pieces,
    })
}

#[derive(Clone, Copy)]
struct Span {
    tag: u16,
    count: u32,
    stride: u32,
    offset: u64,
    len: u64,
}

fn body<'a>(bytes: &'a [u8], span: Span) -> Result<&'a [u8], String> {
    let end = span.offset as usize + span.len as usize;
    Ok(&bytes[span.offset as usize..end])
}

fn check_coverage(bytes: &[u8], spans: &[Span], file_len: u64, header_end: u64) -> Result<(), String> {
    let mut ordered: Vec<Span> = spans.to_vec();
    ordered.sort_by_key(|span| span.offset);
    let mut cursor = header_end;
    for span in ordered {
        if span.offset < cursor {
            return Err(".tokie blobs overlap".to_string());
        }
        if bytes[cursor as usize..span.offset as usize].iter().any(|b| *b != 0) {
            return Err(".tokie gap bytes are not zero".to_string());
        }
        cursor = span.offset + span.len;
    }
    if align_up(cursor, ALIGN) != file_len {
        return Err(".tokie file has trailing bytes".to_string());
    }
    if bytes[cursor as usize..file_len as usize].iter().any(|b| *b != 0) {
        return Err(".tokie gap bytes are not zero".to_string());
    }
    Ok(())
}

fn read_steps(bytes: &[u8], span: Option<Span>) -> Result<Vec<StepRec>, String> {
    let Some(span) = span else {
        return Ok(Vec::new());
    };
    if span.stride != 8 {
        return Err(".tokie steps stride is not 8".to_string());
    }
    let body = body(bytes, span)?;
    let mut steps = Vec::with_capacity(span.count as usize);
    for rec in body.chunks_exact(8) {
        let kind = u16::from_le_bytes(rec[0..2].try_into().unwrap());
        let code = u16::from_le_bytes(rec[2..4].try_into().unwrap());
        steps.push(match kind {
            STEP_NORM => StepRec::Norm(norm_of(code)?),
            STEP_COMPILED => StepRec::Compiled(code),
            STEP_LITERAL => StepRec::Literal(code),
            other => return Err(format!(".tokie step kind {other} is unknown")),
        });
    }
    Ok(steps)
}

fn read_literals(bytes: &[u8], tab: &[u8], span: Option<Span>) -> Result<Vec<(String, String)>, String> {
    let Some(span) = span else {
        return Ok(Vec::new());
    };
    if span.stride != 16 {
        return Err(".tokie literals stride is not 16".to_string());
    }
    let body = body(bytes, span)?;
    let mut out = Vec::with_capacity(span.count as usize);
    for rec in body.chunks_exact(16) {
        let from_off = u32::from_le_bytes(rec[0..4].try_into().unwrap());
        let from_len = u32::from_le_bytes(rec[4..8].try_into().unwrap());
        let to_off = u32::from_le_bytes(rec[8..12].try_into().unwrap());
        let to_len = u32::from_le_bytes(rec[12..16].try_into().unwrap());
        out.push((text(tab, from_off, from_len)?.to_string(), text(tab, to_off, to_len)?.to_string()));
    }
    Ok(out)
}

fn read_slots(bytes: &[u8], tab: &[u8], span: Span, vocab_len: u32) -> Result<Vec<SlotRec>, String> {
    if span.stride != 32 || span.count != vocab_len {
        return Err(".tokie slots do not match the vocab length".to_string());
    }
    let body = body(bytes, span)?;
    let mut slots = Vec::with_capacity(span.count as usize);
    for rec in body.chunks_exact(32) {
        let flags = u32::from_le_bytes(rec[24..28].try_into().unwrap());
        let token = if flags & SLOT_PRESENT != 0 {
            Some(text(tab, u32::from_le_bytes(rec[16..20].try_into().unwrap()), u32::from_le_bytes(rec[20..24].try_into().unwrap()))?.to_string())
        } else {
            None
        };
        slots.push(SlotRec {
            lead_at: u32::from_le_bytes(rec[0..4].try_into().unwrap()),
            lead_len: u32::from_le_bytes(rec[4..8].try_into().unwrap()),
            next_at: u32::from_le_bytes(rec[8..12].try_into().unwrap()),
            next_len: u32::from_le_bytes(rec[12..16].try_into().unwrap()),
            token,
            special: flags & SLOT_SPECIAL != 0,
        });
    }
    Ok(slots)
}

fn read_pairs_str(bytes: &[u8], tab: &[u8], span: Option<Span>) -> Result<Vec<(String, u32)>, String> {
    let Some(span) = span else {
        return Ok(Vec::new());
    };
    if span.stride != 16 {
        return Err(".tokie string record stride is not 16".to_string());
    }
    let body = body(bytes, span)?;
    let mut out = Vec::with_capacity(span.count as usize);
    for rec in body.chunks_exact(16) {
        let off = u32::from_le_bytes(rec[0..4].try_into().unwrap());
        let len = u32::from_le_bytes(rec[4..8].try_into().unwrap());
        let id = u32::from_le_bytes(rec[8..12].try_into().unwrap());
        out.push((text(tab, off, len)?.to_string(), id));
    }
    Ok(out)
}

fn check_id_keys(slots: &[SlotRec], keys: &[(String, u32)]) -> Result<(), String> {
    let present = slots.iter().filter(|slot| slot.token.is_some()).count();
    if present != keys.len() {
        return Err(".tokie id keys do not match the slots".to_string());
    }
    for (token, id) in keys {
        let slot = slots.get(*id as usize).ok_or(".tokie id key is outside the vocab")?;
        if slot.token.as_deref() != Some(token.as_str()) {
            return Err(".tokie id key does not match its slot".to_string());
        }
    }
    Ok(())
}

fn read_quads(bytes: &[u8], span: Option<Span>) -> Result<Vec<(u32, u32, u32, u32)>, String> {
    let Some(span) = span else {
        return Ok(Vec::new());
    };
    if span.stride != 16 {
        return Err(".tokie pair stride is not 16".to_string());
    }
    let body = body(bytes, span)?;
    Ok(body
        .chunks_exact(16)
        .map(|rec| {
            (
                u32::from_le_bytes(rec[0..4].try_into().unwrap()),
                u32::from_le_bytes(rec[4..8].try_into().unwrap()),
                u32::from_le_bytes(rec[8..12].try_into().unwrap()),
                u32::from_le_bytes(rec[12..16].try_into().unwrap()),
            )
        })
        .collect())
}

fn read_u32x256(bytes: &[u8], span: Option<Span>) -> Result<Option<[u32; 256]>, String> {
    let Some(span) = span else {
        return Ok(None);
    };
    if span.stride != 4 || span.count != 256 {
        return Err(".tokie byte table is not 256 ids".to_string());
    }
    let body = body(bytes, span)?;
    let mut ids = [0u32; 256];
    for (i, rec) in body.chunks_exact(4).enumerate() {
        ids[i] = u32::from_le_bytes(rec.try_into().unwrap());
    }
    Ok(Some(ids))
}

fn read_pairs_u32(bytes: &[u8], span: Option<Span>) -> Result<Vec<(u32, u32)>, String> {
    let Some(span) = span else {
        return Ok(Vec::new());
    };
    if span.stride != 8 {
        return Err(".tokie pair stride is not 8".to_string());
    }
    let body = body(bytes, span)?;
    Ok(body
        .chunks_exact(8)
        .map(|rec| {
            (
                u32::from_le_bytes(rec[0..4].try_into().unwrap()),
                u32::from_le_bytes(rec[4..8].try_into().unwrap()),
            )
        })
        .collect())
}

fn read_nodes(bytes: &[u8], span: Option<Span>) -> Result<Vec<(u32, u16, u32, f64)>, String> {
    let Some(span) = span else {
        return Ok(Vec::new());
    };
    if span.stride != 24 {
        return Err(".tokie node stride is not 24".to_string());
    }
    let body = body(bytes, span)?;
    let mut nodes = Vec::with_capacity(span.count as usize);
    for rec in body.chunks_exact(24) {
        let edge_n = u32::from_le_bytes(rec[4..8].try_into().unwrap());
        let edge_n = u16::try_from(edge_n).map_err(|_| "unigram node has too many edges".to_string())?;
        nodes.push((
            u32::from_le_bytes(rec[0..4].try_into().unwrap()),
            edge_n,
            u32::from_le_bytes(rec[8..12].try_into().unwrap()),
            f64::from_bits(u64::from_le_bytes(rec[16..24].try_into().unwrap())),
        ));
    }
    Ok(nodes)
}

fn read_edges(bytes: &[u8], span: Option<Span>) -> Result<Vec<(u8, u32)>, String> {
    let Some(span) = span else {
        return Ok(Vec::new());
    };
    if span.stride != 8 {
        return Err(".tokie edge stride is not 8".to_string());
    }
    let body = body(bytes, span)?;
    Ok(body
        .chunks_exact(8)
        .map(|rec| (rec[0], u32::from_le_bytes(rec[4..8].try_into().unwrap())))
        .collect())
}

fn read_ids(bytes: &[u8], span: Option<Span>) -> Result<Vec<u32>, String> {
    let Some(span) = span else {
        return Ok(Vec::new());
    };
    if span.stride != 4 {
        return Err(".tokie id list stride is not 4".to_string());
    }
    let body = body(bytes, span)?;
    Ok(body
        .chunks_exact(4)
        .map(|rec| u32::from_le_bytes(rec.try_into().unwrap()))
        .collect())
}

fn read_charsmaps(bytes: &[u8], maps: Option<Span>, bmp: Option<Span>) -> Result<Vec<CharsmapRec>, String> {
    match (maps, bmp) {
        (None, None) => Ok(Vec::new()),
        (Some(maps), Some(bmp)) => {
            if maps.stride != 0 || bmp.stride != 8 {
                return Err(".tokie charsmap blobs have the wrong stride".to_string());
            }
            let raw = body(bytes, maps)?;
            let table = body(bytes, bmp)?;
            if raw.len() < 4 {
                return Err(".tokie charsmap blob is truncated".to_string());
            }
            let n = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
            if n > 4 || bmp.count as usize != n * 65536 {
                return Err(".tokie charsmap count is wrong".to_string());
            }
            let mut at = 4usize;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                if at + 4 > raw.len() {
                    return Err(".tokie charsmap frame is truncated".to_string());
                }
                let len = u32::from_le_bytes(raw[at..at + 4].try_into().unwrap()) as usize;
                at += 4;
                if at + len > raw.len() {
                    return Err(".tokie charsmap frame is truncated".to_string());
                }
                let blob = raw[at..at + len].to_vec();
                at += len;
                let base = i * 65536 * 8;
                let mut pairs = Vec::with_capacity(65536);
                for rec in table[base..base + 65536 * 8].chunks_exact(8) {
                    pairs.push((
                        u32::from_le_bytes(rec[0..4].try_into().unwrap()),
                        u32::from_le_bytes(rec[4..8].try_into().unwrap()),
                    ));
                }
                out.push(CharsmapRec { blob, bmp: pairs });
            }
            if at != raw.len() {
                return Err(".tokie charsmap blob has trailing bytes".to_string());
            }
            Ok(out)
        }
        _ => Err(".tokie charsmap and bmp blobs come together".to_string()),
    }
}

fn read_pieces(bytes: &[u8], span: Option<Span>, vocab_len: u32) -> Result<Vec<(Vec<u8>, Vec<u32>)>, String> {
    let Some(span) = span else {
        return Ok(Vec::new());
    };
    if span.stride != 0 || span.count != 1 {
        return Err(".tokie piece blob is not one byte range".to_string());
    }
    let mut body = body(bytes, span)?;
    let mut pieces = Vec::new();
    let mut cost = 0usize;
    while !body.is_empty() {
        if body.len() < 8 {
            return Err(".tokie piece record is truncated".to_string());
        }
        let len = u32::from_le_bytes(body[0..4].try_into().unwrap()) as usize;
        let n = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
        body = &body[8..];
        if len == 0 || body.len() < len + n * 4 {
            return Err(".tokie piece record is truncated".to_string());
        }
        let piece = body[..len].to_vec();
        body = &body[len..];
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            let id = u32::from_le_bytes(body[..4].try_into().unwrap());
            if id >= vocab_len {
                return Err(".tokie piece id is outside the vocab".to_string());
            }
            ids.push(id);
            body = &body[4..];
        }
        cost = cost.saturating_add(piece_cost(&piece));
        if cost > PIECE_BUDGET {
            return Err(".tokie piece list is larger than 128 MiB".to_string());
        }
        pieces.push((piece, ids));
    }
    Ok(pieces)
}

fn text(tab: &[u8], off: u32, len: u32) -> Result<&str, String> {
    let start = off as usize;
    let end = start.checked_add(len as usize).ok_or(".tokie string span overflows")?;
    let bytes = tab.get(start..end).ok_or(".tokie string span is outside the table")?;
    std::str::from_utf8(bytes).map_err(|_| ".tokie string is not utf-8".to_string())
}

struct Tab {
    bytes: Vec<u8>,
    at: HashMap<Vec<u8>, (u32, u32)>,
}

impl Default for Tab {
    fn default() -> Self {
        Self {
            bytes: Vec::new(),
            at: HashMap::new(),
        }
    }
}

impl Tab {
    fn put(&mut self, text: &str) -> Result<(u32, u32), String> {
        if text.is_empty() {
            return Ok((0, 0));
        }
        let bytes = text.as_bytes();
        if let Some(span) = self.at.get(bytes) {
            return Ok(*span);
        }
        let off = u32::try_from(self.bytes.len()).map_err(|_| "string table exceeds 4 GiB".to_string())?;
        let len = u32::try_from(bytes.len()).map_err(|_| "string exceeds 4 GiB".to_string())?;
        self.bytes.extend_from_slice(bytes);
        self.at.insert(bytes.to_vec(), (off, len));
        Ok((off, len))
    }
}

fn meta_blob(image: &Image, tab: &mut Tab) -> Result<Blob, String> {
    let (prefix_off, prefix_len) = tab.put(&image.prefix)?;
    let mut meta = [0u8; 64];
    meta[0] = match image.model {
        Model::Bpe => MODEL_BPE,
        Model::WordPiece => MODEL_WP,
        Model::Unigram => MODEL_UNI,
        Model::Spm => MODEL_SPM,
    };
    if image.prefix_space {
        meta[1] |= FLAG_PREFIX;
    }
    if image.mark {
        meta[1] |= FLAG_MARK;
    }
    if image.strip_space {
        meta[1] |= FLAG_STRIP;
    }
    meta[2..4].copy_from_slice(&cut_code(image.cut).to_le_bytes());
    meta[4..8].copy_from_slice(&image.unk.to_le_bytes());
    meta[8..12].copy_from_slice(&image.max_chars.to_le_bytes());
    meta[12..16].copy_from_slice(&image.vocab_len.to_le_bytes());
    meta[16..20].copy_from_slice(&prefix_off.to_le_bytes());
    meta[20..24].copy_from_slice(&prefix_len.to_le_bytes());
    meta[24..32].copy_from_slice(&image.unk_score.to_bits().to_le_bytes());
    Ok(Blob {
        tag: TAG_META,
        count: 1,
        stride: 64,
        bytes: meta.to_vec(),
    })
}

fn steps_blob(image: &Image) -> Result<Blob, String> {
    let mut bytes = Vec::with_capacity(image.steps.len() * 8);
    for step in &image.steps {
        let (kind, code) = match step {
            StepRec::Norm(norm) => (STEP_NORM, norm_code(*norm)),
            StepRec::Compiled(index) => (STEP_COMPILED, *index),
            StepRec::Literal(index) => (STEP_LITERAL, *index),
        };
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend_from_slice(&code.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
    }
    Ok(Blob {
        tag: TAG_STEPS,
        count: image.steps.len() as u32,
        stride: 8,
        bytes,
    })
}

fn literals_blob(image: &Image, tab: &mut Tab) -> Result<Blob, String> {
    let mut bytes = Vec::with_capacity(image.literals.len() * 16);
    for (from, to) in &image.literals {
        let (from_off, from_len) = tab.put(from)?;
        let (to_off, to_len) = tab.put(to)?;
        bytes.extend_from_slice(&from_off.to_le_bytes());
        bytes.extend_from_slice(&from_len.to_le_bytes());
        bytes.extend_from_slice(&to_off.to_le_bytes());
        bytes.extend_from_slice(&to_len.to_le_bytes());
    }
    Ok(Blob {
        tag: TAG_LITERALS,
        count: image.literals.len() as u32,
        stride: 16,
        bytes,
    })
}

fn arena_blob(image: &Image) -> Blob {
    Blob {
        tag: TAG_ARENA,
        count: 1,
        stride: 0,
        bytes: image.arena.clone(),
    }
}

fn slots_blob(image: &Image, tab: &mut Tab) -> Result<Blob, String> {
    let mut bytes = Vec::with_capacity(image.slots.len() * 32);
    for slot in &image.slots {
        bytes.extend_from_slice(&slot.lead_at.to_le_bytes());
        bytes.extend_from_slice(&slot.lead_len.to_le_bytes());
        bytes.extend_from_slice(&slot.next_at.to_le_bytes());
        bytes.extend_from_slice(&slot.next_len.to_le_bytes());
        let (off, len, flags) = if let Some(token) = &slot.token {
            let (off, len) = tab.put(token)?;
            let mut flags = SLOT_PRESENT;
            if slot.special {
                flags |= SLOT_SPECIAL;
            }
            (off, len, flags)
        } else {
            (0, 0, 0)
        };
        bytes.extend_from_slice(&off.to_le_bytes());
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(&flags.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
    }
    Ok(Blob {
        tag: TAG_SLOTS,
        count: image.slots.len() as u32,
        stride: 32,
        bytes,
    })
}

fn id_keys_blob(image: &Image, tab: &mut Tab) -> Result<Blob, String> {
    let mut keys = Vec::new();
    for (id, slot) in image.slots.iter().enumerate() {
        if let Some(token) = &slot.token {
            keys.push((token.clone(), id as u32));
        }
    }
    Ok(added_blob(TAG_ID_KEYS, &keys, tab)?.unwrap_or(Blob {
        tag: TAG_ID_KEYS,
        count: 0,
        stride: 16,
        bytes: Vec::new(),
    }))
}

fn pairs_blob(image: &Image) -> Option<Blob> {
    if image.pairs.is_empty() && !matches!(image.model, Model::Bpe | Model::Spm) {
        return None;
    }
    if !matches!(image.model, Model::Bpe | Model::Spm) {
        return None;
    }
    let mut bytes = Vec::with_capacity(image.pairs.len() * 16);
    for (left, right, rank, id) in &image.pairs {
        bytes.extend_from_slice(&left.to_le_bytes());
        bytes.extend_from_slice(&right.to_le_bytes());
        bytes.extend_from_slice(&rank.to_le_bytes());
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    Some(Blob {
        tag: TAG_PAIRS,
        count: image.pairs.len() as u32,
        stride: 16,
        bytes,
    })
}

fn byte_ids_blob(image: &Image) -> Option<Blob> {
    let ids = image.byte_ids?;
    Some(u32_table(TAG_BYTE_IDS, &ids))
}

fn spm_bytes_blob(image: &Image) -> Option<Blob> {
    let ids = image.spm_bytes?;
    Some(u32_table(TAG_SPM_BYTES, &ids))
}

fn u32_table(tag: u16, ids: &[u32; 256]) -> Blob {
    let mut bytes = Vec::with_capacity(256 * 4);
    for id in ids {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    Blob {
        tag,
        count: 256,
        stride: 4,
        bytes,
    }
}

fn spm_chars_blob(image: &Image) -> Option<Blob> {
    if image.spm_chars.is_empty() {
        return None;
    }
    let mut bytes = Vec::with_capacity(image.spm_chars.len() * 8);
    for (scalar, id) in &image.spm_chars {
        bytes.extend_from_slice(&scalar.to_le_bytes());
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    Some(Blob {
        tag: TAG_SPM_CHARS,
        count: image.spm_chars.len() as u32,
        stride: 8,
        bytes,
    })
}

fn nodes_blob(image: &Image) -> Option<Blob> {
    if image.nodes.is_empty() && !matches!(image.model, Model::Unigram) {
        return None;
    }
    if !matches!(image.model, Model::Unigram) {
        return None;
    }
    let mut bytes = Vec::with_capacity(image.nodes.len() * 24);
    for (edge_at, edge_n, id, score) in &image.nodes {
        bytes.extend_from_slice(&edge_at.to_le_bytes());
        bytes.extend_from_slice(&(*edge_n as u32).to_le_bytes());
        bytes.extend_from_slice(&id.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&score.to_bits().to_le_bytes());
    }
    Some(Blob {
        tag: TAG_NODES,
        count: image.nodes.len() as u32,
        stride: 24,
        bytes,
    })
}

fn edges_blob(image: &Image) -> Option<Blob> {
    if !matches!(image.model, Model::Unigram) {
        return None;
    }
    let mut bytes = Vec::with_capacity(image.edges.len() * 8);
    for (byte, to) in &image.edges {
        bytes.push(*byte);
        bytes.extend_from_slice(&[0, 0, 0]);
        bytes.extend_from_slice(&to.to_le_bytes());
    }
    Some(Blob {
        tag: TAG_EDGES,
        count: image.edges.len() as u32,
        stride: 8,
        bytes,
    })
}

fn added_blob(tag: u16, rows: &[(String, u32)], tab: &mut Tab) -> Result<Option<Blob>, String> {
    if rows.is_empty() {
        return Ok(None);
    }
    let mut bytes = Vec::with_capacity(rows.len() * 16);
    for (token, id) in rows {
        let (off, len) = tab.put(token)?;
        bytes.extend_from_slice(&off.to_le_bytes());
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(&id.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
    }
    Ok(Some(Blob {
        tag,
        count: rows.len() as u32,
        stride: 16,
        bytes,
    }))
}

fn ids_blob(tag: u16, ids: &[u32]) -> Option<Blob> {
    if ids.is_empty() {
        return None;
    }
    let mut bytes = Vec::with_capacity(ids.len() * 4);
    for id in ids {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    Some(Blob {
        tag,
        count: ids.len() as u32,
        stride: 4,
        bytes,
    })
}

fn charsmap_blob(image: &Image) -> Option<Blob> {
    if image.charsmaps.is_empty() {
        return None;
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(image.charsmaps.len() as u32).to_le_bytes());
    for map in &image.charsmaps {
        bytes.extend_from_slice(&(map.blob.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&map.blob);
    }
    Some(Blob {
        tag: TAG_CHARSMAP,
        count: 1,
        stride: 0,
        bytes,
    })
}

fn bmp_blob(image: &Image) -> Option<Blob> {
    if image.charsmaps.is_empty() {
        return None;
    }
    let mut bytes = Vec::with_capacity(image.charsmaps.len() * 65536 * 8);
    for map in &image.charsmaps {
        for (start, end) in &map.bmp {
            bytes.extend_from_slice(&start.to_le_bytes());
            bytes.extend_from_slice(&end.to_le_bytes());
        }
    }
    Some(Blob {
        tag: TAG_BMP,
        count: (image.charsmaps.len() * 65536) as u32,
        stride: 8,
        bytes,
    })
}

fn wp_blob(image: &Image, tab: &mut Tab) -> Result<Option<Blob>, String> {
    if !matches!(image.model, Model::WordPiece) {
        return Ok(None);
    }
    added_blob(TAG_WP, &image.wp, tab)
}

fn pieces_blob(image: &Image) -> Result<Option<Blob>, String> {
    if image.pieces.is_empty() {
        return Ok(None);
    }
    let mut cost = 0usize;
    let mut bytes = Vec::new();
    for (piece, ids) in &image.pieces {
        if piece.is_empty() {
            return Err(".tokie piece is empty".to_string());
        }
        for id in ids {
            if *id >= image.vocab_len {
                return Err(".tokie piece id is outside the vocab".to_string());
            }
        }
        cost = cost.saturating_add(piece_cost(piece));
        if cost > PIECE_BUDGET {
            return Err(".tokie piece list is larger than 128 MiB".to_string());
        }
        bytes.extend_from_slice(&(piece.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(ids.len() as u32).to_le_bytes());
        bytes.extend_from_slice(piece);
        for id in ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
    }
    Ok(Some(Blob {
        tag: TAG_PIECES,
        count: 1,
        stride: 0,
        bytes,
    }))
}

fn cut_code(cut: Cut) -> u16 {
    match cut {
        Cut::Gpt2 => 1,
        Cut::SmolLM => 2,
        Cut::Cl100k => 3,
        Cut::Olmo => 4,
        Cut::Voyage => 5,
        Cut::Qwen => 6,
        Cut::O200k => 7,
        Cut::Nemotron => 8,
        Cut::Kimi => 9,
        Cut::DeepSeek => 10,
        Cut::Bert => 11,
        Cut::T5 => 12,
        Cut::None => 13,
    }
}

fn cut_of(code: u16) -> Result<Cut, String> {
    Ok(match code {
        1 => Cut::Gpt2,
        2 => Cut::SmolLM,
        3 => Cut::Cl100k,
        4 => Cut::Olmo,
        5 => Cut::Voyage,
        6 => Cut::Qwen,
        7 => Cut::O200k,
        8 => Cut::Nemotron,
        9 => Cut::Kimi,
        10 => Cut::DeepSeek,
        11 => Cut::Bert,
        12 => Cut::T5,
        13 => Cut::None,
        other => return Err(format!(".tokie cut {other} is unknown")),
    })
}

fn norm_code(norm: Norm) -> u16 {
    match norm {
        Norm::Nfc => 1,
        Norm::Nfd => 2,
        Norm::Nfkc => 3,
        Norm::Nfkd => 4,
        Norm::Nmt => 5,
        Norm::Lower => 6,
        Norm::Clean => 7,
        Norm::BertCased => 8,
        Norm::BertUncased => 9,
        Norm::StripAccents => 10,
        Norm::Albert => 11,
        Norm::Llama => 12,
        Norm::Metaspace => 13,
        Norm::MetaspaceFirst => 14,
        Norm::MetaspaceNever => 15,
        Norm::MetaspaceWords => 16,
        Norm::ByteLevel => 17,
        Norm::Strip => 18,
        Norm::CollapseSpaces => 19,
        Norm::ReplaceSpace => 20,
        Norm::Prepend => 21,
    }
}

fn norm_of(code: u16) -> Result<Norm, String> {
    Ok(match code {
        1 => Norm::Nfc,
        2 => Norm::Nfd,
        3 => Norm::Nfkc,
        4 => Norm::Nfkd,
        5 => Norm::Nmt,
        6 => Norm::Lower,
        7 => Norm::Clean,
        8 => Norm::BertCased,
        9 => Norm::BertUncased,
        10 => Norm::StripAccents,
        11 => Norm::Albert,
        12 => Norm::Llama,
        13 => Norm::Metaspace,
        14 => Norm::MetaspaceFirst,
        15 => Norm::MetaspaceNever,
        16 => Norm::MetaspaceWords,
        17 => Norm::ByteLevel,
        18 => Norm::Strip,
        19 => Norm::CollapseSpaces,
        20 => Norm::ReplaceSpace,
        21 => Norm::Prepend,
        other => return Err(format!(".tokie normalizer {other} is unknown")),
    })
}

fn align_up(n: u64, align: u64) -> u64 {
    n.div_ceil(align) * align
}

fn crc32c(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC_TABLE[index] ^ (crc >> 8);
    }
    !crc
}

const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut bit = 0;
        while bit < 8 {
            if c & 1 == 1 {
                c = 0x82F6_3B78 ^ (c >> 1);
            } else {
                c >>= 1;
            }
            bit += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::crc32c;

    #[test]
    fn castagnoli_check_value() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }
}
